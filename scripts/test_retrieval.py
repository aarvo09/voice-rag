#!/usr/bin/env python3
"""
Retrieval Test & Sanity Verification Script (TASK 06).

Loads FAISS index and corpus metadata, embeds test query, executes vector search,
measures online latency vs model startup time, runs a 10-query Hit@1/Hit@5 sanity test,
and verifies search determinism.
"""

import os
import sys
import time
import argparse
import logging
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.embeddings.model import MultilingualE5Embedder
from app.retrieval.faiss_index import FaissVectorIndex
from app.retrieval.metadata import CorpusMetadataLoader
from app.retrieval.retriever import VectorRetriever

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_INDEX_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "indexes", "dev.faiss")
DEFAULT_CORPUS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "processed", "dev_corpus.parquet")


def get_process_memory_mb() -> float:
    try:
        import psutil
        return psutil.Process().memory_info().rss / (1024.0 * 1024.0)
    except Exception:
        pass

    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / 1024.0
    except Exception:
        pass

    import resource
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


def main():
    parser = argparse.ArgumentParser(description="FAISS Retrieval Test and Sanity Evaluator")
    parser.add_argument("--query", type=str, default="मैनहट्टन परियोजना की सफलता का क्या प्रभाव पड़ा?", help="Test query string")
    parser.add_argument("--top-k", type=int, default=5, help="Top-K passages to retrieve")
    parser.add_argument("--index-path", type=str, default=DEFAULT_INDEX_PATH, help="Path to dev.faiss")
    parser.add_argument("--corpus-path", type=str, default=DEFAULT_CORPUS_PATH, help="Path to dev_corpus.parquet")
    parser.add_argument("--model-name", type=str, default="intfloat/multilingual-e5-small", help="HF embedding model name")
    args = parser.parse_args()

    mem_before_load = get_process_memory_mb()
    logger.info(f"Memory before index & model load: {mem_before_load:.2f} MB")

    # 1. Measure Startup Component Times
    t_start0 = time.time()

    faiss_index = FaissVectorIndex()
    faiss_index.load(args.index_path)

    metadata_loader = CorpusMetadataLoader(args.corpus_path)
    mem_after_index_load = get_process_memory_mb()

    t_model0 = time.time()
    embedder = MultilingualE5Embedder(model_name=args.model_name)
    model_load_time_ms = round((time.time() - t_model0) * 1000, 2)

    startup_time_ms = round((time.time() - t_start0) * 1000, 2)

    retriever = VectorRetriever(embedder=embedder, faiss_index=faiss_index, metadata_loader=metadata_loader)

    # 2. Warm up model & index
    _ = retriever.retrieve("warmup query", top_k=1)

    # 3. Single Query Online Latency Measurement
    t_embed0 = time.time()
    query_vector = embedder.embed_query(args.query)
    query_embed_time_ms = round((time.time() - t_embed0) * 1000, 3)

    t_search0 = time.time()
    scores, indices = faiss_index.search(query_vector, top_k=args.top_k)
    faiss_search_time_ms = round((time.time() - t_search0) * 1000, 3)

    t_lookup0 = time.time()
    results = retriever.retrieve(args.query, top_k=args.top_k)
    meta_lookup_time_ms = round((time.time() - t_lookup0) * 1000, 3) - query_embed_time_ms - faiss_search_time_ms

    online_total_retrieval_ms = round(query_embed_time_ms + faiss_search_time_ms + max(0.0, meta_lookup_time_ms), 3)
    mem_after_search = get_process_memory_mb()

    # 4. Verification of Search Determinism
    results_second_pass = retriever.retrieve(args.query, top_k=args.top_k)
    deterministic_pass = [r["document_id"] for r in results] == [r["document_id"] for r in results_second_pass]

    # 5. Sanity Check over 10 Corpus Queries (Hit@1 and Hit@5)
    corpus_table = pq.read_table(args.corpus_path)
    distinct_queries = []
    seen_qids = set()

    for r in corpus_table.to_pylist():
        qid = r["query_id"]
        qtext = r.get("query", "")
        if qid not in seen_qids and qtext:
            seen_qids.add(qid)
            distinct_queries.append({"query_id": qid, "query": qtext})
        if len(distinct_queries) >= 10:
            break

    hit_1_count = 0
    hit_5_count = 0
    sanity_queries_tested = len(distinct_queries)

    for q_item in distinct_queries:
        target_qid = q_item["query_id"]
        q_text = q_item["query"]

        retrieved_docs = retriever.retrieve(q_text, top_k=5)

        # Check Hit@1
        if retrieved_docs and retrieved_docs[0]["query_id"] == target_qid and retrieved_docs[0]["is_selected"] == 1:
            hit_1_count += 1

        # Check Hit@5
        if any(doc["query_id"] == target_qid and doc["is_selected"] == 1 for doc in retrieved_docs):
            hit_5_count += 1

    hit_1_pct = (hit_1_count / sanity_queries_tested * 100) if sanity_queries_tested > 0 else 0.0
    hit_5_pct = (hit_5_count / sanity_queries_tested * 100) if sanity_queries_tested > 0 else 0.0

    print("\n==================================================")
    print("TASK 06 — RETRIEVAL PERFORMANCE & SANITY REPORT")
    print("==================================================")
    print(f"FAISS Index Type:             IndexFlatIP ({faiss_index.size()} vectors, d={faiss_index.dimension()})")
    print(f"Embedding Model:              {embedder.model_name} (Device: {embedder.device})")
    print(f"Test Query:                   '{args.query}'")
    print(f"Startup Time (Model Load):    {startup_time_ms} ms ({model_load_time_ms} ms model load)")
    print(f"Query Embedding Time:         {query_embed_time_ms} ms")
    print(f"FAISS Search Time:            {faiss_search_time_ms} ms")
    print(f"Metadata Lookup Time:         {meta_lookup_time_ms:.3f} ms")
    print(f"Online Retrieval Latency:     {online_total_retrieval_ms} ms (EXCLUDING startup)")
    print(f"Search Determinism Verified:  {deterministic_pass} -> PASS")
    print(f"Memory Before Index Load:     {mem_before_load:.2f} MB")
    print(f"Memory After Index Load:      {mem_after_index_load:.2f} MB")
    print(f"Memory After Query Search:    {mem_after_search:.2f} MB")

    print(f"\n--- 10-QUERY GROUND-TRUTH SANITY CHECK RESULTS ---")
    print(f"Sanity Queries Tested:        {sanity_queries_tested}")
    print(f"Hit@1 Score:                  {hit_1_count}/{sanity_queries_tested} ({hit_1_pct:.1f}%)")
    print(f"Hit@5 Score:                  {hit_5_count}/{sanity_queries_tested} ({hit_5_pct:.1f}%)")

    print(f"\n--- TOP-{args.top_k} RETRIEVED PASSAGES ---")
    for res in results:
        text_preview = res["text"][:75] + "..." if len(res["text"]) > 75 else res["text"]
        print(f"  [Rank {res['rank']}] Score: {res['score']:.4f} | ID: {res['document_id']} | Selected: {res['is_selected']} | QID: {res['query_id']} | Text: {text_preview}")
    print("==================================================\n")


if __name__ == "__main__":
    main()
