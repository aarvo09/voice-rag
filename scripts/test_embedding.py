#!/usr/bin/env python3
"""
Test Embedding Script (TASK 05).
Validates model loading, query prefixing, vector output shape, norms, and memory usage.
"""

import os
import sys
import time
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.embeddings.model import MultilingualE5Embedder

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


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
    mem_before = get_process_memory_mb()
    logger.info(f"Memory before model load: {mem_before:.2f} MB")

    t0 = time.time()
    embedder = MultilingualE5Embedder(model_name="intfloat/multilingual-e5-small")
    model_load_time = round(time.time() - t0, 3)
    mem_after_load = get_process_memory_mb()

    sample_query = "मैनहट्टन परियोजना की सफलता का क्या प्रभाव पड़ा?"
    t1 = time.time()
    query_vector = embedder.embed_query(sample_query)
    embed_time = round(time.time() - t1, 4)
    mem_after_embed = get_process_memory_mb()

    vector_norm = float((query_vector ** 2).sum() ** 0.5)

    print("\n==================================================")
    print("TASK 05 — EMBEDDING MODEL TEST RESULT")
    print("==================================================")
    print(f"Model Name:                 {embedder.model_name}")
    print(f"Device Used:                {embedder.device}")
    print(f"Sample Query:               {sample_query}")
    print(f"Embedding Vector Shape:     {query_vector.shape}")
    print(f"Embedding Dimension:        {embedder.get_embedding_dimension()}")
    print(f"Vector L2 Norm:             {vector_norm:.6f} (Normalized ~1.0)")
    print(f"First 5 Float Values:       {query_vector[:5].tolist()}")
    print(f"Model Load Time:            {model_load_time} s")
    print(f"Query Embed Time:           {embed_time} s")
    print(f"Memory Before Model Load:   {mem_before:.2f} MB")
    print(f"Memory After Model Load:    {mem_after_load:.2f} MB")
    print(f"Memory After Query Embed:   {mem_after_embed:.2f} MB")
    print("==================================================\n")


if __name__ == "__main__":
    main()
