"""
Text RAG Live & Dry-Run Benchmark Script (TASK 15).
Runs bounded evaluation across development queries (evaluation/queries.json).
Calculates P50/P70/P100 latency, grounded answer rate, refusal rate, and citation validity.

Usage:
  python scripts/benchmark_text_rag.py --sample-size 10 --dry-run
  python scripts/benchmark_text_rag.py --sample-size 10
"""

import os
import sys
import json
import argparse
import logging
import numpy as np

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
DEFAULT_QUERIES_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "evaluation", "queries.json")


def load_queries(path: str, sample_size: int = 10):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    queries = data.get("queries", data) if isinstance(data, dict) else data
    return queries[:sample_size]


def calc_percentiles(values):
    if not values:
        return {"p50": 0.0, "p70": 0.0, "p100": 0.0}
    arr = np.array(values)
    return {
        "p50": round(float(np.percentile(arr, 50)), 2),
        "p70": round(float(np.percentile(arr, 70)), 2),
        "p100": round(float(np.max(arr)), 2)
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark Text RAG execution on query sample.")
    parser.add_argument("--sample-size", type=int, default=10, help="Number of queries to sample (default: 10).")
    parser.add_argument("--dry-run", action="store_true", help="Run benchmark in dry-run mode using MockLLMProvider.")
    parser.add_argument("--provider", type=str, default="gemini", help="LLM provider name (default: gemini).")
    args = parser.parse_args()

    if not args.dry_run and args.provider == "gemini":
        if not os.getenv("GOOGLE_API_KEY"):
            print("\n[WARNING] GOOGLE_API_KEY not set. Running benchmark in dry-run mode.")
            args.dry_run = True

    queries = load_queries(DEFAULT_QUERIES_PATH, sample_size=args.sample_size)
    print(f"\n==================================================")
    print(f"BENCHMARKING TEXT RAG ({len(queries)} Queries, Dry-Run={args.dry_run})")
    print(f"==================================================")

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

    model_name = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
    llm_config = LLMConfig(provider=args.provider, model_name=model_name)
    provider_inst = get_llm_provider(llm_config, force_mock=args.dry_run)

    rag_service = TextRAGService(
        retriever=prod_retriever,
        provider=provider_inst,
        config=llm_config
    )

    retrieval_lats = []
    llm_lats = []
    total_lats = []

    success_count = 0
    grounded_count = 0
    refusal_count = 0
    valid_citation_count = 0
    total_retries = 0
    api_errors = 0

    results_log = []

    for idx, q_item in enumerate(queries, start=1):
        q_text = q_item.get("query", q_item.get("text", ""))
        print(f"[{idx}/{len(queries)}] Query: '{q_text[:50]}...'")

        res = rag_service.run(q_text, dry_run=args.dry_run)
        telem = res.get("telemetry", {})

        retrieval_lats.append(telem.get("retrieval_ms", 0.0))
        llm_lats.append(telem.get("llm_ms", 0.0))
        total_lats.append(telem.get("total_ms", 0.0))

        if res.get("status") == "success" or res.get("status") == "dry_run":
            success_count += 1
        if res.get("grounded"):
            grounded_count += 1
        if "refused" in res.get("status", ""):
            refusal_count += 1
        if res.get("citations") and not res.get("error"):
            valid_citation_count += 1
        if res.get("error"):
            api_errors += 1

        total_retries += res.get("retry_count", 0)

        results_log.append({
            "query": q_text,
            "status": res.get("status"),
            "grounded": res.get("grounded"),
            "citations": res.get("citations"),
            "retrieval_ms": telem.get("retrieval_ms"),
            "llm_ms": telem.get("llm_ms"),
            "total_ms": telem.get("total_ms")
        })

    total_q = len(queries)
    ret_pct = calc_percentiles(retrieval_lats)
    llm_pct = calc_percentiles(llm_lats)
    tot_pct = calc_percentiles(total_lats)

    print("\n==================================================")
    print("BENCHMARK SUMMARY RESULTS")
    print("==================================================")
    print(f"Evaluated Queries:    {total_q}")
    print(f"Dry-Run Mode:         {args.dry_run}")
    print(f"Provider / Model:     {args.provider} / {model_name}")
    print(f"Success Count / Rate: {success_count}/{total_q} ({round(success_count/total_q*100, 1)}%)")
    print(f"Grounded Rate:        {round(grounded_count/total_q*100, 1)}%")
    print(f"Refusal Rate:         {round(refusal_count/total_q*100, 1)}%")
    print(f"Valid Citation Rate:  {round(valid_citation_count/total_q*100, 1)}%")
    print(f"Total Retries:        {total_retries}")
    print(f"API Errors:           {api_errors}")
    print("--------------------------------------------------")
    print("LATENCY PERCENTILES (ms):")
    print(f"  Retrieval-only:     P50={ret_pct['p50']} ms | P70={ret_pct['p70']} ms | P100={ret_pct['p100']} ms")
    print(f"  LLM Generation:     P50={llm_pct['p50']} ms | P70={llm_pct['p70']} ms | P100={llm_pct['p100']} ms")
    print(f"  Total Text-RAG:     P50={tot_pct['p50']} ms | P70={tot_pct['p70']} ms | P100={tot_pct['p100']} ms")
    print("==================================================\n")


if __name__ == "__main__":
    main()
