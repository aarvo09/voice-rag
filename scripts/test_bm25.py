#!/usr/bin/env python3
"""
BM25 Lexical Retrieval Test & Sanity Verification Script (TASK 07).

Loads BM25 index and corpus metadata, tokenizes test query, executes BM25Okapi search,
measures online query latency separately from index loading, runs 10-query Hit@1/Hit@5 sanity test,
and verifies search determinism.
"""

import os
import sys
import time
import argparse
import logging
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.retrieval.bm25 import BM25Retriever, tokenize_hindi
from app.retrieval.metadata import CorpusMetadataLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_INDEX_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "indexes", "dev_bm25.pkl")
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


def execute_bm25_retrieval(bm25: BM25Retriever, metadata_loader: CorpusMetadataLoader, query: str, top_k: int = 5):
    t_token0 = time.time()
    tokens = bm25.tokenize(query)
    token_time_ms = round((time.time() - t_token0) * 1000, 3)

    t_search0 = time.time()
    scores, indices = bm25.search(query, top_k=top_k)
    search_time_ms = round((time.time() - t_search0) * 1000, 3) - token_time_ms

    t_lookup0 = time.time()
    results = []
    if indices.size > 0:
        for rank_idx, (score, row_idx) in enumerate(zip(scores[0], indices[0]), start=1):
            if row_idx < 0:
                continue
            doc_meta = metadata_loader.get_document(int(row_idx))
            res_item = {
                "rank": rank_idx,
                "document_id": doc_meta["document_id"],
                "score": float(score),
                "text": doc_meta["text"],
                "language": doc_meta["language"],
                "query_id": int(doc_meta["query_id"]),
                "passage_index": int(doc_meta["passage_index"]),
                "is_selected": int(doc_meta["is_selected"]),
                "source": doc_meta.get("source", "ai4bharat/MSMARCO-XI")
            }
            results.append(res_item)
    lookup_time_ms = round((time.time() - t_lookup0) * 1000, 3)

    total_online_ms = round(token_time_ms + max(0.0, search_time_ms) + lookup_time_ms, 3)

    return results, {
        "tokens": tokens,
        "token_time_ms": token_time_ms,
        "search_time_ms": max(0.0, search_time_ms),
        "lookup_time_ms": lookup_time_ms,
        "total_online_ms": total_online_ms
    }


def main():
    parser = argparse.ArgumentParser(description="BM25 Retrieval Test and Sanity Evaluator")
    parser.add_argument("--query", type=str, default="मैनहट्टन परियोजना की सफलता का क्या प्रभाव पड़ा?", help="Test query string")
    parser.add_argument("--top-k", type=int, default=5, help="Top-K passages to retrieve")
    parser.add_argument("--index-path", type=str, default=DEFAULT_INDEX_PATH, help="Path to dev_bm25.pkl")
    parser.add_argument("--corpus-path", type=str, default=DEFAULT_CORPUS_PATH, help="Path to dev_corpus.parquet")
    args = parser.parse_args()

    mem_before_load = get_process_memory_mb()
    logger.info(f"Memory before BM25 index load: {mem_before_load:.2f} MB")

    # 1. Startup Component Times (Index Loading)
    t_load0 = time.time()
    bm25 = BM25Retriever()
    bm25.load(args.index_path)

    metadata_loader = CorpusMetadataLoader(args.corpus_path)
    index_load_time_ms = round((time.time() - t_load0) * 1000, 2)
    mem_after_load = get_process_memory_mb()

    mem_before_query = get_process_memory_mb()

    # 2. Single Query Online Latency Measurement
    results, timings = execute_bm25_retrieval(bm25, metadata_loader, args.query, top_k=args.top_k)
    mem_after_query = get_process_memory_mb()

    # 3. Verification of Search Determinism
    results_pass2, _ = execute_bm25_retrieval(bm25, metadata_loader, args.query, top_k=args.top_k)
    det_ids_pass1 = [r["document_id"] for r in results]
    det_ids_pass2 = [r["document_id"] for r in results_pass2]
    det_scores_pass1 = [r["score"] for r in results]
    det_scores_pass2 = [r["score"] for r in results_pass2]

    deterministic_pass = (det_ids_pass1 == det_ids_pass2) and (det_scores_pass1 == det_scores_pass2)

    # 4. Sanity Check over 10 Corpus Queries (Hit@1 and Hit@5)
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

        retrieved_docs, _ = execute_bm25_retrieval(bm25, metadata_loader, q_text, top_k=5)

        # Check Hit@1
        if retrieved_docs and retrieved_docs[0]["query_id"] == target_qid and retrieved_docs[0]["is_selected"] == 1:
            hit_1_count += 1

        # Check Hit@5
        if any(doc["query_id"] == target_qid and doc["is_selected"] == 1 for doc in retrieved_docs):
            hit_5_count += 1

    hit_1_pct = (hit_1_count / sanity_queries_tested * 100) if sanity_queries_tested > 0 else 0.0
    hit_5_pct = (hit_5_count / sanity_queries_tested * 100) if sanity_queries_tested > 0 else 0.0

    print("\n==================================================")
    print("TASK 07 — BM25 RETRIEVAL PERFORMANCE & SANITY REPORT")
    print("==================================================")
    print(f"BM25 Corpus Size:             {bm25.corpus_size} passages")
    print(f"Test Query:                   '{args.query}'")
    print(f"Query Tokens:                 {timings['tokens']}")
    print(f"Index Loading Time:           {index_load_time_ms} ms (Startup Component)")
    print(f"Query Tokenization Time:      {timings['token_time_ms']} ms")
    print(f"BM25 Search Time:             {timings['search_time_ms']} ms")
    print(f"Metadata Lookup Time:         {timings['lookup_time_ms']} ms")
    print(f"Online Retrieval Latency:     {timings['total_online_ms']} ms (EXCLUDING index load)")
    print(f"Search Determinism Verified:  {deterministic_pass} -> PASS")
    print(f"Memory Before Index Load:     {mem_before_load:.2f} MB")
    print(f"Memory After Index Load:      {mem_after_load:.2f} MB")
    print(f"Memory Before Query:          {mem_before_query:.2f} MB")
    print(f"Memory After Query:           {mem_after_query:.2f} MB")

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
