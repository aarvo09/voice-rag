"""
CLI Utility to test end-to-end Guarded RAG generation with live Google Gemini API (TASK 15).
Supports live execution or --dry-run mode.

Usage:
  python scripts/test_real_generation.py --query "मैनहट्टन परियोजना की सफलता का क्या प्रभाव पड़ा?"
  python scripts/test_real_generation.py --query "What was the Manhattan Project?" --dry-run
"""

import os
import sys
import argparse
import logging
from dotenv import load_dotenv

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

dotenv_path = os.path.join(project_root, ".env")
load_dotenv(dotenv_path=dotenv_path)

from app.retrieval.metadata import CorpusMetadataLoader
from app.embeddings.model import MultilingualE5Embedder
from app.retrieval.faiss_index import FaissVectorIndex
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.retriever import VectorRetriever, ProductionRetriever
from app.generation.config import LLMConfig
from app.generation.llm import get_llm_provider
from app.pipeline.text_rag import TextRAGService

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_CORPUS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "processed", "dev_corpus.parquet")
DEFAULT_FAISS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "indexes", "dev.faiss")
DEFAULT_BM25_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "indexes", "dev_bm25.pkl")


def main():
    parser = argparse.ArgumentParser(description="Test end-to-end guarded RAG generation with Gemini API.")
    parser.add_argument("--query", type=str, required=True, help="User query string for RAG answer generation.")
    parser.add_argument("--dry-run", action="store_true", help="Run retrieval and context formatting without calling live LLM API.")
    parser.add_argument("--provider", type=str, default=os.getenv("LLM_PROVIDER", "gemini"), help="LLM provider name (default: from .env).")
    parser.add_argument("--model", type=str, default=None, help="LLM model name.")
    args = parser.parse_args()

    # Determine correct API key env variable
    api_key_env = "GROQ_API_KEY" if args.provider == "groq" else "GOOGLE_API_KEY"
    if not args.dry_run:
        api_key = os.getenv(api_key_env)
        if not api_key:
            print(f"\n[WARNING] {api_key_env} environment variable is not set!")
            print(f"Falling back to dry-run / mock mode. Set {api_key_env} to run live {args.provider} requests.\n")
            args.dry_run = True

    logger.info("Initializing Production RAG components...")
    metadata_loader = CorpusMetadataLoader(DEFAULT_CORPUS_PATH)
    embedder = MultilingualE5Embedder()
    faiss_idx = FaissVectorIndex()
    faiss_idx.load(DEFAULT_FAISS_PATH)

    bm25_idx = BM25Retriever()
    bm25_idx.load(DEFAULT_BM25_PATH)

    vector_retriever = VectorRetriever(
        embedder=embedder,
        faiss_index=faiss_idx,
        metadata_loader=metadata_loader
    )

    prod_retriever = ProductionRetriever(
        vector_retriever=vector_retriever,
        bm25_retriever=bm25_idx,
        metadata_loader=metadata_loader
    )

    # Initialize LLM config safely
    llm_config = LLMConfig()
    # Override with CLI args if explicitly provided
    if args.provider:
        llm_config.provider = args.provider
    if args.model:
        llm_config.model_name = args.model

    provider_inst = get_llm_provider(llm_config, force_mock=args.dry_run)

    rag_service = TextRAGService(
        retriever=prod_retriever,
        provider=provider_inst,
        config=llm_config
    )

    logger.info(f"Executing Guarded Text RAG (Query: '{args.query}', Dry-Run={args.dry_run})...")
    res = rag_service.run(args.query, dry_run=args.dry_run)

    doc_ids = [d.get("document_id", "") for d in res.get("retrieved_documents", [])]

    print("\n==================================================")
    print("GUARDED TEXT RAG RESULT")
    print("==================================================")
    print(f"Query:              {res['query']}")
    print(f"Generated Answer:   {res.get('generated_answer', res.get('answer', ''))}")
    print(f"Grounded:           {res['grounded']}")
    print(f"Status:             {res['status']}")
    print(f"Final Answer:       {res['answer']}")
    print(f"Citations:          {res['citations']}")
    print(f"Confidence:         {res['confidence']}")
    print(f"Refusal Reason:     {res.get('refusal_reason')}")
    print(f"Provider:           {res['provider']}")
    print(f"Model:              {res['model']}")
    print(f"Retrieved Doc IDs:  {doc_ids}")
    print(f"Retry Count:        {res.get('retry_count', 0)}")
    print("--------------------------------------------------")
    print("LATENCY TELEMETRY BREAKDOWN:")
    print(f"  Input Guardrail:  {res['telemetry'].get('input_guardrail_ms', 0.0)} ms")
    print(f"  Retrieval:        {res['telemetry']['retrieval_ms']} ms")
    print(f"  Relevance Guard:  {res['telemetry'].get('retrieval_guardrail_ms', 0.0)} ms")
    print(f"  Context Build:    {res['telemetry']['context_build_ms']} ms")
    print(f"  LLM Generation:   {res['telemetry']['llm_ms']} ms")
    print(f"  Output Parsing:   {res['telemetry']['parsing_ms']} ms")
    print(f"  Grounding Guard:  {res['telemetry'].get('grounding_guardrail_ms', 0.0)} ms")
    print(f"  TOTAL GUARDRAILS: {res['telemetry'].get('total_guardrail_ms', 0.0)} ms")
    print(f"  TOTAL TEXT RAG:   {res['telemetry']['total_ms']} ms")
    print("==================================================\n")


if __name__ == "__main__":
    main()
