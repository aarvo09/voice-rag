"""
CLI Utility to test end-to-end Text RAG generation (TASK 13).
Supports real LLM execution or --dry-run mode without credentials.

Usage:
  python scripts/test_generation.py --query "मैनहट्टन परियोजना की सफलता का तुरंत क्या प्रभाव पड़ा?"
  python scripts/test_generation.py --query "मैनहट्टन परियोजना" --dry-run
"""

import os
import sys
import argparse
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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


def get_process_memory_mb() -> float:
    """Returns current process RAM usage (VmRSS) in MB."""
    try:
        with open("/proc/self/status", "r") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    parts = line.split()
                    kb = float(parts[1])
                    return round(kb / 1024.0, 2)
    except Exception:
        pass
    return 0.0


def main():
    parser = argparse.ArgumentParser(description="Test end-to-end Text RAG generation pipeline.")
    parser.add_argument("--query", type=str, required=True, help="User query string for RAG answer generation.")
    parser.add_argument("--dry-run", action="store_true", help="Run retrieval and context formatting without calling live LLM API.")
    parser.add_argument("--provider", type=str, default="openai", help="LLM provider name (default: openai).")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="LLM model identifier.")
    args = parser.parse_args()

    mem_start = get_process_memory_mb()
    logger.info("Initializing Production RAG components...")

    # Load components
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

    llm_config = LLMConfig(provider=args.provider, model_name=args.model)
    provider_inst = get_llm_provider(llm_config, force_mock=args.dry_run)

    rag_service = TextRAGService(
        retriever=prod_retriever,
        provider=provider_inst,
        config=llm_config
    )

    logger.info(f"Executing Text RAG for Query: '{args.query}' (Dry-Run={args.dry_run})")
    response = rag_service.run(args.query, dry_run=args.dry_run)

    mem_end = get_process_memory_mb()

    print("\n==================================================")
    print("TEXT RAG GENERATION RESULT")
    print("==================================================")
    print(f"Query:              {response['query']}")
    print(f"Answer:             {response['answer']}")
    print(f"Grounded:           {response['grounded']}")
    print(f"Citations:          {response['citations']}")
    print(f"Confidence:         {response['confidence']}")
    print(f"Provider:           {response['provider']}")
    print(f"Model:              {response['model']}")
    print(f"Dry-Run Mode:       {response.get('dry_run', False)}")
    print(f"Retrieved Docs:     {len(response['retrieved_documents'])}")
    print("--------------------------------------------------")
    print("LATENCY TELEMETRY:")
    print(f"  Retrieval:        {response['telemetry']['retrieval_ms']} ms")
    print(f"  Context Build:    {response['telemetry']['context_build_ms']} ms")
    print(f"  LLM Generation:   {response['telemetry']['llm_ms']} ms")
    print(f"  Output Parsing:   {response['telemetry']['parsing_ms']} ms")
    print(f"  TOTAL TEXT RAG:   {response['telemetry']['total_ms']} ms")
    print("--------------------------------------------------")
    print(f"MEMORY FOOTPRINT:   {mem_end} MB (Initial: {mem_start} MB)")
    print("==================================================\n")

    if args.dry_run or response.get("context_str"):
        print("FORMATTED CONTEXT PAYLOAD:")
        print("--------------------------------------------------")
        print(response.get("context_str", ""))
        print("==================================================\n")


if __name__ == "__main__":
    main()
