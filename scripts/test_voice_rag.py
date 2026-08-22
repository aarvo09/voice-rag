"""
CLI Utility to test end-to-end Voice RAG pipeline.

Flow: Audio -> Sarvam STT -> Text RAG -> Answer
"""

import os
import sys
import argparse
import time
import logging
from dotenv import load_dotenv

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
load_dotenv(os.path.join(project_root, ".env"))

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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Test Voice RAG Pipeline")
    parser.add_argument("--audio", type=str, required=True, help="Path to local audio file")
    args = parser.parse_args()
    
    if not os.path.exists(args.audio):
        print(f"Error: Audio file not found at {args.audio}")
        sys.exit(1)
        
    try:
        # 1. Initialize STT
        stt_provider = SarvamSTTProvider()
        
        # 2. Run STT
        logger.info(f"Running STT on {args.audio}...")
        stt_result = stt_provider.transcribe(args.audio)
        
        logger.info(f"Transcribed Text: {stt_result.transcript} (Language: {stt_result.detected_language})")
        
        if not stt_result.transcript.strip():
            logger.error("Empty transcript. Aborting.")
            sys.exit(1)
            
        # 3. Initialize Text RAG Pipeline
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
        
        llm_config = LLMConfig()
        provider = get_llm_provider(config=llm_config)
        policy = GuardrailPolicy(config=GuardrailPolicyConfig(max_grounding_retries=1))
        
        rag_service = TextRAGService(retriever=retriever, provider=provider, policy=policy)
        
        # 4. Run Text RAG
        logger.info(f"Executing RAG with query: '{stt_result.transcript}'...")
        res = rag_service.run(stt_result.transcript)
        
        # 5. Output Results & Latency
        print("\n==================================================")
        print("VOICE RAG RESULT")
        print("==================================================")
        print(f"Audio File:         {args.audio}")
        print(f"Transcript:         {stt_result.transcript}")
        print(f"Detected Language:  {stt_result.detected_language}")
        print(f"Generated Answer:   {res.get('generated_answer', res.get('answer', ''))}")
        print(f"Grounded:           {res['grounded']}")
        print(f"Status:             {res['status']}")
        print(f"Final Answer:       {res['answer']}")
        print(f"Citations:          {res['citations']}")
        print("--------------------------------------------------")
        
        stt_ms = stt_result.latency_ms
        retrieval_ms = res['telemetry']['retrieval_ms']
        llm_ms = res['telemetry']['llm_ms']
        grounding_ms = res['telemetry'].get('grounding_guardrail_ms', 0.0)
        total_rag_ms = res['telemetry']['total_ms']
        
        total_pipeline_ms = round(stt_ms + total_rag_ms, 2)
        
        print("LATENCY TELEMETRY BREAKDOWN:")
        print(f"  STT Latency:         {stt_ms} ms")
        print(f"  Retrieval Latency:   {retrieval_ms} ms")
        print(f"  LLM Latency:         {llm_ms} ms")
        print(f"  Grounding Latency:   {grounding_ms} ms")
        print(f"  Total RAG Latency:   {total_rag_ms} ms")
        print(f"  TOTAL END-TO-END:    {total_pipeline_ms} ms")
        print("==================================================\n")
        
    except Exception as e:
        logger.error(f"Pipeline Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
