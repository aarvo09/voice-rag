#!/usr/bin/env python3
"""
Multi-Strategy Chunk Variant Generator & Evaluator (TASK 09).

Loads data/processed/dev_corpus.parquet (1,000 passages), processes one strategy at a time,
saves chunk Parquet files to data/processed/chunks/<strategy>.parquet, records build timing/memory,
and generates reports/chunking_statistics.md and reports/chunk_examples.md.
"""

import os
import sys
import gc
import time
import json
import argparse
import logging
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.chunking.models import ChunkConfig
from app.chunking.registry import registry
from app.embeddings.model import MultilingualE5Embedder

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_CORPUS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "processed", "dev_corpus.parquet")
DEFAULT_CHUNKS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "processed", "chunks")
DEFAULT_REPORTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports")


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


def compute_chunk_statistics(chunks_dicts: list, doc_count: int) -> dict:
    if not chunks_dicts:
        return {
            "output_chunks": 0,
            "avg_chunks_per_doc": 0.0,
            "min_chunk_length": 0,
            "max_chunk_length": 0,
            "avg_chunk_length": 0.0,
            "median_chunk_length": 0.0,
            "empty_chunks": 0,
            "duplicate_chunks": 0,
            "selected_chunks": 0,
            "unselected_chunks": 0,
        }

    lengths = [len(c["text"]) for c in chunks_dicts]
    empty_count = sum(1 for l in lengths if l == 0)

    texts = [c["text"] for c in chunks_dicts]
    duplicate_count = len(texts) - len(set(texts))

    selected_count = sum(1 for c in chunks_dicts if c["is_selected"] == 1)
    unselected_count = sum(1 for c in chunks_dicts if c["is_selected"] == 0)

    return {
        "output_chunks": len(chunks_dicts),
        "avg_chunks_per_doc": round(len(chunks_dicts) / doc_count, 2),
        "min_chunk_length": int(np.min(lengths)),
        "max_chunk_length": int(np.max(lengths)),
        "avg_chunk_length": round(float(np.mean(lengths)), 2),
        "median_chunk_length": round(float(np.median(lengths)), 2),
        "empty_chunks": empty_count,
        "duplicate_chunks": duplicate_count,
        "selected_chunks": selected_count,
        "unselected_chunks": unselected_count,
    }


def save_chunks_parquet(chunks_dicts: list, output_path: str) -> int:
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    schema = pa.schema([
        ("chunk_id", pa.string()),
        ("parent_document_id", pa.string()),
        ("text", pa.string()),
        ("language", pa.string()),
        ("query_id", pa.int64()),
        ("passage_index", pa.int64()),
        ("is_selected", pa.int64()),
        ("chunk_type", pa.string()),
        ("chunk_index", pa.int64()),
        ("start_offset", pa.int64()),
        ("end_offset", pa.int64()),
    ])

    table = pa.Table.from_pylist(chunks_dicts, schema=schema)
    pq.write_table(table, output_path, compression="snappy")
    return os.path.getsize(output_path)


def main():
    parser = argparse.ArgumentParser(description="Multi-Strategy Chunk Variant Generator")
    parser.add_argument("--corpus-path", type=str, default=DEFAULT_CORPUS_PATH, help="Path to dev_corpus.parquet")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_CHUNKS_DIR, help="Output directory for chunk Parquet files")
    parser.add_argument("--reports-dir", type=str, default=DEFAULT_REPORTS_DIR, help="Output directory for markdown reports")
    args = parser.parse_args()

    if not os.path.exists(args.corpus_path):
        logger.error(f"Input corpus file not found: {args.corpus_path}")
        sys.exit(1)

    # 1. Load source corpus document records
    corpus_table = pq.read_table(args.corpus_path)
    documents = corpus_table.to_pylist()
    doc_count = len(documents)
    logger.info(f"Loaded {doc_count} passages from {args.corpus_path}")

    # Shared embedder for semantic chunking
    shared_embedder = None

    strategies = ["native", "sentence_window", "fixed", "semantic"]
    all_stats = {}
    sample_doc_examples = {}  # {doc_id: {strategy: [chunks]}}

    # Pick 3 sample documents for chunk_examples.md inspection
    sample_docs = documents[:3]

    for strat_name in strategies:
        mem_before = get_process_memory_mb()
        t0 = time.time()

        logger.info(f"Processing chunking strategy: '{strat_name}'...")

        if strat_name == "semantic":
            if shared_embedder is None:
                shared_embedder = MultilingualE5Embedder(model_name="intfloat/multilingual-e5-small")
            chunker = registry.get("semantic", embedder=shared_embedder)
        else:
            chunker = registry.get(strat_name)

        # Process chunks document by document (bounded memory)
        chunk_objects = []
        for doc in documents:
            chunks = chunker.chunk_document(doc)
            chunk_objects.extend(chunks)

            if doc["document_id"] in [s["document_id"] for s in sample_docs]:
                d_id = doc["document_id"]
                if d_id not in sample_doc_examples:
                    sample_doc_examples[d_id] = {"doc_text": doc["text"]}
                sample_doc_examples[d_id][strat_name] = [c.text for c in chunks]

        build_time_ms = round((time.time() - t0) * 1000, 2)
        mem_after = get_process_memory_mb()

        # Convert chunks to dicts
        chunk_dicts = [c.to_dict() for c in chunk_objects]

        # Save to Parquet
        out_parquet_path = os.path.join(args.output_dir, f"{strat_name}.parquet")
        file_bytes = save_chunks_parquet(chunk_dicts, out_parquet_path)

        # Compute statistics
        stats = compute_chunk_statistics(chunk_dicts, doc_count)
        stats["strategy"] = strat_name
        stats["build_time_ms"] = build_time_ms
        stats["mem_before_mb"] = round(mem_before, 2)
        stats["mem_after_mb"] = round(mem_after, 2)
        stats["file_size_kb"] = round(file_bytes / 1024.0, 2)

        all_stats[strat_name] = stats

        logger.info(f"Completed '{strat_name}': {stats['output_chunks']} chunks in {build_time_ms} ms. Memory: {mem_after:.2f} MB")

        # Cleanup memory between strategies
        del chunk_objects
        del chunk_dicts
        gc.collect()

    # 2. Generate reports/chunking_statistics.md
    stats_report_path = os.path.join(args.reports_dir, "chunking_statistics.md")
    os.makedirs(args.reports_dir, exist_ok=True)

    stats_md = f"""# Multi-Strategy Chunking Statistics Report (Task 09)

**Project**: Voice-enabled RAG using `ai4bharat/MSMARCO-XI` (Hindi Dev Corpus)  
**Input Documents**: {doc_count} passages  
**Date**: {datetime.now().strftime('%B %d, %Y')}  

---

## 1. Chunking Strategy Comparison Table

| Metric | Native | Sentence Window | Fixed-Size | Semantic Boundary |
| :--- | :---: | :---: | :---: | :---: |
| **Output Chunks** | {all_stats['native']['output_chunks']} | {all_stats['sentence_window']['output_chunks']} | {all_stats['fixed']['output_chunks']} | {all_stats['semantic']['output_chunks']} |
| **Avg Chunks / Doc** | {all_stats['native']['avg_chunks_per_doc']} | {all_stats['sentence_window']['avg_chunks_per_doc']} | {all_stats['fixed']['avg_chunks_per_doc']} | {all_stats['semantic']['avg_chunks_per_doc']} |
| **Min Chunk Length (chars)** | {all_stats['native']['min_chunk_length']} | {all_stats['sentence_window']['min_chunk_length']} | {all_stats['fixed']['min_chunk_length']} | {all_stats['semantic']['min_chunk_length']} |
| **Max Chunk Length (chars)** | {all_stats['native']['max_chunk_length']} | {all_stats['sentence_window']['max_chunk_length']} | {all_stats['fixed']['max_chunk_length']} | {all_stats['semantic']['max_chunk_length']} |
| **Avg Chunk Length (chars)** | {all_stats['native']['avg_chunk_length']} | {all_stats['sentence_window']['avg_chunk_length']} | {all_stats['fixed']['avg_chunk_length']} | {all_stats['semantic']['avg_chunk_length']} |
| **Median Chunk Length (chars)** | {all_stats['native']['median_chunk_length']} | {all_stats['sentence_window']['median_chunk_length']} | {all_stats['fixed']['median_chunk_length']} | {all_stats['semantic']['median_chunk_length']} |
| **Empty Chunks** | {all_stats['native']['empty_chunks']} | {all_stats['sentence_window']['empty_chunks']} | {all_stats['fixed']['empty_chunks']} | {all_stats['semantic']['empty_chunks']} |
| **Duplicate Chunks** | {all_stats['native']['duplicate_chunks']} | {all_stats['sentence_window']['duplicate_chunks']} | {all_stats['fixed']['duplicate_chunks']} | {all_stats['semantic']['duplicate_chunks']} |
| **Selected Chunks (Inherited)** | {all_stats['native']['selected_chunks']} | {all_stats['sentence_window']['selected_chunks']} | {all_stats['fixed']['selected_chunks']} | {all_stats['semantic']['selected_chunks']} |
| **Offline Build Time (ms)** | {all_stats['native']['build_time_ms']} ms | {all_stats['sentence_window']['build_time_ms']} ms | {all_stats['fixed']['build_time_ms']} ms | {all_stats['semantic']['build_time_ms']} ms |
| **Parquet File Size (KB)** | {all_stats['native']['file_size_kb']} KB | {all_stats['sentence_window']['file_size_kb']} KB | {all_stats['fixed']['file_size_kb']} KB | {all_stats['semantic']['file_size_kb']} KB |
| **Peak Memory (MB)** | {all_stats['native']['mem_after_mb']} MB | {all_stats['sentence_window']['mem_after_mb']} MB | {all_stats['fixed']['mem_after_mb']} MB | {all_stats['semantic']['mem_after_mb']} MB |

---

## 2. Key Insights & Ground-Truth Handling
- **Ground-Truth Label Inheritance**: Every chunk inherits `is_selected` directly from its parent passage. *Note*: Inherited relevance labels indicate source document relevance, not guaranteed presence of the answer within a single chunk slice.
- **Deduplication Status**: Chunks are not globally deduplicated yet. Duplicate counts reflect exact text overlaps across overlapping sliding windows.
- **Offline Processing Efficiency**: All chunk datasets were generated offline without introducing query-time overhead.
"""

    with open(stats_report_path, "w", encoding="utf-8") as f:
        f.write(stats_md)
    logger.info(f"Saved chunking statistics report to {stats_report_path}")

    # 3. Generate reports/chunk_examples.md
    examples_report_path = os.path.join(args.reports_dir, "chunk_examples.md")
    examples_md = f"""# Representative Chunking Strategy Visual Examples (Task 09)

This document provides representative visual examples of source Hindi passages converted into chunks across all four strategies: **Native**, **Sentence-Window**, **Fixed-Size**, and **Semantic Boundary**.

---
"""
    for doc_id, data in sample_doc_examples.items():
        examples_md += f"## Document ID: `{doc_id}`\n\n"
        examples_md += f"### Original Source Passage\n> {data['doc_text']}\n\n"

        for strat in strategies:
            chunks = data.get(strat, [])
            examples_md += f"#### Strategy: `{strat.upper()}` ({len(chunks)} Chunks)\n"
            for idx, c_text in enumerate(chunks):
                examples_md += f"- **Chunk {idx}**: {c_text}\n"
            examples_md += "\n"
        examples_md += "---\n\n"

    with open(examples_report_path, "w", encoding="utf-8") as f:
        f.write(examples_md)
    logger.info(f"Saved chunk examples report to {examples_report_path}")

    # 4. Summary Console Output
    print("\n==================================================")
    print("TASK 09 — MULTI-STRATEGY CHUNKING SUMMARY REPORT")
    print("==================================================")
    print(f"Input Document Count:       {doc_count}")
    print(f"Native Chunk Count:         {all_stats['native']['output_chunks']} ({all_stats['native']['avg_chunks_per_doc']} chunks/doc, Build: {all_stats['native']['build_time_ms']} ms)")
    print(f"Sentence Window Count:      {all_stats['sentence_window']['output_chunks']} ({all_stats['sentence_window']['avg_chunks_per_doc']} chunks/doc, Build: {all_stats['sentence_window']['build_time_ms']} ms)")
    print(f"Fixed Size Count:           {all_stats['fixed']['output_chunks']} ({all_stats['fixed']['avg_chunks_per_doc']} chunks/doc, Build: {all_stats['fixed']['build_time_ms']} ms)")
    print(f"Semantic Boundary Count:    {all_stats['semantic']['output_chunks']} ({all_stats['semantic']['avg_chunks_per_doc']} chunks/doc, Build: {all_stats['semantic']['build_time_ms']} ms)")
    print("==================================================\n")


if __name__ == "__main__":
    main()
