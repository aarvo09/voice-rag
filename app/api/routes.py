import os
import sys
import time
import tempfile
import logging
from typing import Dict, Any

from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel

from app.stt.sarvam import SarvamSTTProvider
from app.retrieval.metadata import CorpusMetadataLoader
from app.embeddings.model import MultilingualE5Embedder
from app.retrieval.faiss_index import FaissVectorIndex
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.retriever import ProductionRetriever, VectorRetriever
from app.generation.config import LLMConfig
from app.generation.llm import get_llm_provider
from app.guardrails.policy import GuardrailPolicy, GuardrailPolicyConfig
from app.pipeline.text_rag import TextRAGService

logger = logging.getLogger(__name__)

router = APIRouter()

# Global state to hold services
services = {}

def initialize_services():
    if "rag_service" in services:
        return
        
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    logger.info("Initializing Sarvam STT Provider...")
    services["stt_provider"] = SarvamSTTProvider()
    
    logger.info("Initializing Text RAG components...")
    parquet_path = os.path.join(project_root, "data", "processed", "dev_corpus.parquet")
    faiss_path = os.path.join(project_root, "data", "indexes", "dev.faiss")
    bm25_path = os.path.join(project_root, "data", "indexes", "dev_bm25.pkl")
    
    metadata_loader = CorpusMetadataLoader(parquet_path)
    embedder = MultilingualE5Embedder()
    faiss_idx = FaissVectorIndex()
    faiss_idx.load(faiss_path)
    bm25_idx = BM25Retriever()
    bm25_idx.load(bm25_path)
    
    vector_retriever = VectorRetriever(
        embedder=embedder,
        faiss_index=faiss_idx,
        metadata_loader=metadata_loader
    )
    
    retriever = ProductionRetriever(
        vector_retriever=vector_retriever,
        bm25_retriever=bm25_idx,
        metadata_loader=metadata_loader
    )
    
    os.environ["LLM_PROVIDER"] = "groq"
    llm_config = LLMConfig()
    llm_config.provider = "groq"
    provider = get_llm_provider(config=llm_config)
    
    policy = GuardrailPolicy(config=GuardrailPolicyConfig(max_grounding_retries=0))
    
    services["rag_service"] = TextRAGService(retriever=retriever, provider=provider, policy=policy)
    logger.info("Services initialized.")

@router.on_event("startup")
async def startup_event():
    initialize_services()

@router.post("/api/voice-query")
async def voice_query(audio: UploadFile = File(...)):
    start_e2e = time.perf_counter()
    
    if "stt_provider" not in services or "rag_service" not in services:
        raise HTTPException(status_code=500, detail="Services not initialized")
        
    stt_provider = services["stt_provider"]
    rag_service = services["rag_service"]
    
    # Save uploaded file to a temporary location
    try:
        suffix = os.path.splitext(audio.filename)[1] if audio.filename else ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await audio.read()
            tmp.write(content)
            tmp_path = tmp.name
            
        stt_result = stt_provider.transcribe(tmp_path)
        
    finally:
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.remove(tmp_path)
            
    if not stt_result.transcript.strip():
        raise HTTPException(status_code=400, detail="No speech detected")
        
    lang = stt_result.detected_language
    lang_name = "Hindi" if "hi" in lang.lower() else "English"
    
    # RAG
    res = rag_service.run(stt_result.transcript, metadata={"language": lang_name})
    end_e2e = time.perf_counter()
    
    total_latency_ms = round((end_e2e - start_e2e) * 1000.0, 2)
    telemetry = res.get("telemetry", {})
    
    return {
        "transcript": stt_result.transcript,
        "language": lang,
        "answer": res.get("answer"),
        "grounded": res.get("grounded", False),
        "citations": res.get("citations", []),
        "status": res.get("status"),
        "timings": {
            "stt_ms": stt_result.latency_ms,
            "retrieval_ms": telemetry.get("retrieval_ms", 0) + telemetry.get("retrieval_guardrail_ms", 0),
            "generation_ms": telemetry.get("llm_ms", 0),
            "grounding_ms": telemetry.get("grounding_guardrail_ms", 0),
            "total_rag_ms": telemetry.get("total_ms", 0),
            "total_e2e_ms": total_latency_ms
        },
        "retrieved_documents": res.get("retrieved_documents", [])
    }
