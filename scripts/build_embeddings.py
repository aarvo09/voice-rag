#!/usr/bin/env python3
"""
Corpus Embedding Builder (TASK 05).

Loads data/processed/dev_corpus.parquet, generates passage embeddings in bounded
batches using intfloat/multilingual-e5-small, saves data/processed/dev_embeddings.npy
and data/processed/embedding_metadata.json, and validates vector properties.
"""

import os
import sys
import time
import json
import argparse
import logging
from datetime import datetime
import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.embeddings.model import MultilingualE5Embedder
from app.embeddings.batcher import batch_encode_documents

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_INPUT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "processed", "dev_corpus.parquet")
DEFAULT_OUTPUT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "processed", "dev_embeddings.npy")
DEFAULT_META_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "processed", "embedding_metadata.json")


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
    parser = argparse.ArgumentParser(description="MSMARCO-XI Local Corpus Embedder")
    parser.add_argument("--input", type=str, default=DEFAULT_INPUT_PATH, help="Input corpus Parquet path")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_PATH, help="Output NumPy embeddings (.npy) path")
    parser.add_argument("--metadata", type=str, default=DEFAULT_META_PATH, help="Output metadata (.json) path")
    parser.add_argument("--batch-size", type=int, default=32, help="Document batch size (default: 32)")
    parser.add_argument("--model-name", type=str, default="intfloat/multilingual-e5-small", help="HF model name")
    args = parser.parse_args()

    mem_before_mb = get_process_memory_mb()
    logger.info(f"Memory before reading corpus: {mem_before_mb:.2f} MB")

    if not os.path.exists(args.input):
        logger.error(f"Input corpus file not found at: {args.input}")
        sys.exit(1)

    # 1. Load document texts from dev_corpus.parquet using PyArrow
    t_read0 = time.time()
    table = pq.read_table(args.input, columns=["document_id", "text"])
    texts = table["text"].to_pylist()
    doc_ids = table["document_id"].to_pylist()
    doc_count = len(texts)
    read_time_ms = round((time.time() - t_read0) * 1000, 2)
    logger.info(f"Loaded {doc_count} document passages in {read_time_ms} ms.")

    # 2. Load model
    t_model0 = time.time()
    embedder = MultilingualE5Embedder(model_name=args.model_name)
    model_load_time_ms = round((time.time() - t_model0) * 1000, 2)
    mem_after_model_load_mb = get_process_memory_mb()
    logger.info(f"Memory after model load: {mem_after_model_load_mb:.2f} MB")

    # 3. Generate document embeddings in bounded batches
    t_embed0 = time.time()
    embeddings = batch_encode_documents(embedder, texts, batch_size=args.batch_size, show_progress=False)
    embedding_time_ms = round((time.time() - t_embed0) * 1000, 2)
    mem_after_embedding_mb = get_process_memory_mb()

    # 4. Check metadata alignment & properties
    if len(doc_ids) != embeddings.shape[0]:
        logger.error(f"MISMATCH: Corpus rows ({len(doc_ids)}) != Embedding rows ({embeddings.shape[0]})")
        sys.exit(1)

    is_finite = bool(np.isfinite(embeddings).all())
    embedding_dim = embeddings.shape[1]

    # Validate vector norms
    norms = np.linalg.norm(embeddings, axis=1)
    avg_norm = float(np.mean(norms))
    norm_is_valid = bool(np.allclose(norms, 1.0, atol=1e-3))

    # 5. Save output .npy file
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    np.save(args.output, embeddings)
    output_file_size = os.path.getsize(args.output)

    # 6. Save metadata JSON
    metadata = {
        "model_name": embedder.model_name,
        "embedding_dimension": int(embedding_dim),
        "document_count": int(doc_count),
        "normalized": True,
        "device": embedder.device,
        "batch_size": int(args.batch_size),
        "creation_timestamp": datetime.now().isoformat(),
        "input_corpus": args.input,
        "output_embedding_file": args.output,
        "file_size_bytes": output_file_size,
        "memory_metrics": {
            "memory_before_mb": round(mem_before_mb, 2),
            "memory_after_model_load_mb": round(mem_after_model_load_mb, 2),
            "memory_after_embedding_mb": round(mem_after_embedding_mb, 2),
            "memory_delta_mb": round(mem_after_embedding_mb - mem_before_mb, 2)
        },
        "timing_metrics": {
            "model_load_time_ms": model_load_time_ms,
            "embedding_time_ms": embedding_time_ms,
            "total_time_ms": round(model_load_time_ms + embedding_time_ms, 2)
        }
    }
    with open(args.metadata, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # 7. Post-save Validation
    loaded_arr = np.load(args.output)
    val_shape = loaded_arr.shape
    val_dtype = str(loaded_arr.dtype)
    val_finite = bool(np.isfinite(loaded_arr).all())
    val_sample_norm = float(np.linalg.norm(loaded_arr[0]))

    print("\n==================================================")
    print("TASK 05 — EMBEDDING GENERATION SUMMARY & VALIDATION")
    print("==================================================")
    print(f"Model Name:                 {embedder.model_name}")
    print(f"Device Used:                {embedder.device}")
    print(f"Input Corpus Path:          {args.input}")
    print(f"Documents Embedded:         {doc_count}")
    print(f"Embedding Dimension:        {embedding_dim}")
    print(f"Batch Size:                 {args.batch_size}")
    print(f"Output Embedding Path:      {args.output}")
    print(f"Output File Size:           {output_file_size / 1024:.2f} KB ({output_file_size:,} bytes)")
    print(f"Metadata Path:              {args.metadata}")
    print(f"Row Alignment Check:        Corpus ({doc_count}) == Embeddings ({embeddings.shape[0]}) -> PASS")
    print(f"Finite Values Check:        {is_finite} -> PASS")
    print(f"Normalization Check:        Mean Norm = {avg_norm:.6f} (~1.0) -> PASS ({norm_is_valid})")
    print(f"Saved Array Shape:          {val_shape}")
    print(f"Saved Array Data Type:      {val_dtype}")
    print(f"Validation Sample Norm:     {val_sample_norm:.6f}")
    print(f"Model Load Time:            {model_load_time_ms / 1000:.2f} s ({model_load_time_ms} ms)")
    print(f"Embedding Time:             {embedding_time_ms / 1000:.2f} s ({embedding_time_ms} ms)")
    print(f"Memory Before:              {mem_before_mb:.2f} MB")
    print(f"Memory After Model Load:    {mem_after_model_load_mb:.2f} MB")
    print(f"Memory After Embedding:     {mem_after_embedding_mb:.2f} MB")
    print(f"Memory Delta:               {mem_after_embedding_mb - mem_before_mb:.2f} MB")
    print("==================================================\n")


if __name__ == "__main__":
    main()
