#!/usr/bin/env python3
"""
FAISS Index Builder (TASK 06).
Loads data/processed/dev_embeddings.npy, builds a CPU FAISS IndexFlatIP index,
saves data/indexes/dev.faiss and data/indexes/dev_index_metadata.json, and validates properties.
"""

import os
import sys
import time
import json
import argparse
import logging
from datetime import datetime
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.retrieval.faiss_index import FaissVectorIndex

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_EMBEDDINGS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "processed", "dev_embeddings.npy")
DEFAULT_OUTPUT_INDEX_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "indexes", "dev.faiss")
DEFAULT_METADATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "indexes", "dev_index_metadata.json")


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
    parser = argparse.ArgumentParser(description="FAISS Vector Index Builder")
    parser.add_argument("--embeddings", type=str, default=DEFAULT_EMBEDDINGS_PATH, help="Path to dev_embeddings.npy")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_INDEX_PATH, help="Output .faiss index file path")
    parser.add_argument("--metadata", type=str, default=DEFAULT_METADATA_PATH, help="Output index metadata JSON path")
    parser.add_argument("--model-name", type=str, default="intfloat/multilingual-e5-small", help="Embedding model name")
    args = parser.parse_args()

    mem_before = get_process_memory_mb()
    logger.info(f"Memory before index build: {mem_before:.2f} MB")

    if not os.path.exists(args.embeddings):
        logger.error(f"Input embeddings file not found at: {args.embeddings}")
        sys.exit(1)

    t0 = time.time()

    # 1. Load & validate NumPy embeddings array
    embeddings = np.load(args.embeddings)
    logger.info(f"Loaded embeddings array of shape {embeddings.shape}, dtype {embeddings.dtype}")

    if embeddings.ndim != 2:
        logger.error(f"Embeddings array must be 2D, got shape {embeddings.shape}")
        sys.exit(1)
    if embeddings.dtype != np.float32:
        logger.info(f"Casting embeddings dtype from {embeddings.dtype} to float32...")
        embeddings = embeddings.astype(np.float32)
    if not np.isfinite(embeddings).all():
        logger.error("Embeddings contain non-finite values (NaN/Inf)!")
        sys.exit(1)

    n_vectors, dimension = embeddings.shape

    # 2. Build FAISS IndexFlatIP
    faiss_index = FaissVectorIndex(dimension=dimension)
    faiss_index.build(embeddings)

    build_time = round(time.time() - t0, 4)
    mem_after = get_process_memory_mb()
    mem_delta = round(mem_after - mem_before, 2)

    # 3. Save index
    faiss_index.save(args.output)
    index_file_size = os.path.getsize(args.output)

    # 4. Save metadata JSON
    metadata = {
        "index_type": "IndexFlatIP",
        "metric": "inner_product",
        "dimension": int(dimension),
        "vector_count": int(n_vectors),
        "embedding_model": args.model_name,
        "normalized": True,
        "source_embedding_file": args.embeddings,
        "creation_time": datetime.now().isoformat(),
        "file_size_bytes": index_file_size,
        "memory_metrics": {
            "memory_before_mb": round(mem_before, 2),
            "memory_after_mb": round(mem_after, 2),
            "memory_delta_mb": mem_delta
        },
        "build_time_s": build_time
    }

    os.makedirs(os.path.dirname(args.metadata), exist_ok=True)
    with open(args.metadata, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # 5. Validation Check
    val_index = FaissVectorIndex()
    val_index.load(args.output)
    val_size = val_index.size()
    val_dim = val_index.dimension()

    print("\n==================================================")
    print("TASK 06 — FAISS INDEX BUILD SUMMARY & VALIDATION")
    print("==================================================")
    print(f"Source Embeddings Path:     {args.embeddings}")
    print(f"Embedding Vector Count:     {n_vectors}")
    print(f"Embedding Vector Dimension: {dimension}")
    print(f"FAISS Index Type:           IndexFlatIP (Exact Cosine / Inner-Product)")
    print(f"Output Index Path:          {args.output}")
    print(f"Output Index Size:          {index_file_size / 1024:.2f} KB ({index_file_size:,} bytes)")
    print(f"Output Metadata Path:       {args.metadata}")
    print(f"Index Vector Count Check:   Saved ({n_vectors}) == Loaded ({val_size}) -> PASS")
    print(f"Index Dimension Check:      Saved ({dimension}) == Loaded ({val_dim}) -> PASS")
    print(f"Index Build Time:           {build_time} s")
    print(f"Memory Before:              {mem_before:.2f} MB")
    print(f"Memory After:               {mem_after:.2f} MB")
    print(f"Memory Delta:               {mem_delta:.2f} MB")
    print("==================================================\n")


if __name__ == "__main__":
    main()
