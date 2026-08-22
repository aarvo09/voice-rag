"""
Pytest Unit Test Suite for Chunk Retrieval Evaluation (TASK 10).
Validates loading of chunk Parquet datasets, embedding file alignment,
FAISS index dimensions, ID mappings, and inherited relevance metadata.
"""

import os
import pytest
import numpy as np
import pyarrow.parquet as pq
import faiss

CHUNKS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "processed", "chunks")
EMB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "processed", "chunk_embeddings")
INDEX_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "indexes", "chunk_variants")

STRATEGIES = ["native", "sentence_window", "fixed", "semantic"]


@pytest.mark.parametrize("strat", STRATEGIES)
def test_chunk_dataset_loading_and_counts(strat):
    parquet_path = os.path.join(CHUNKS_DIR, f"{strat}.parquet")
    assert os.path.exists(parquet_path), f"Parquet file for {strat} missing"

    table = pq.read_table(parquet_path)
    rows = table.to_pylist()
    assert len(rows) > 0, f"Chunk count for {strat} is zero"

    # Verify inherited relevance metadata is present
    first_row = rows[0]
    assert "is_selected" in first_row
    assert "parent_document_id" in first_row
    assert "query_id" in first_row
    assert "chunk_id" in first_row


@pytest.mark.parametrize("strat", STRATEGIES)
def test_embeddings_alignment_and_faiss(strat):
    parquet_path = os.path.join(CHUNKS_DIR, f"{strat}.parquet")
    npy_path = os.path.join(EMB_DIR, f"{strat}.npy")
    faiss_path = os.path.join(INDEX_DIR, f"{strat}.faiss")

    if not os.path.exists(npy_path) or not os.path.exists(faiss_path):
        pytest.skip(f"Embedding or FAISS index for {strat} not yet generated (run evaluate_chunk_retrieval.py first)")

    table = pq.read_table(parquet_path)
    rows = table.to_pylist()
    chunk_count = len(rows)

    embeddings = np.load(npy_path)
    assert embeddings.shape[0] == chunk_count, f"Embedding count {embeddings.shape[0]} != chunk count {chunk_count} for {strat}"
    assert embeddings.shape[1] == 384, f"Embedding dimension is not 384 for {strat}"

    index = faiss.read_index(faiss_path)
    assert index.ntotal == chunk_count, f"FAISS ntotal {index.ntotal} != chunk count {chunk_count} for {strat}"
    assert index.d == 384, f"FAISS index dimension {index.d} != 384 for {strat}"


@pytest.mark.parametrize("strat", STRATEGIES)
def test_retrieved_ids_map_to_valid_chunks(strat):
    parquet_path = os.path.join(CHUNKS_DIR, f"{strat}.parquet")
    faiss_path = os.path.join(INDEX_DIR, f"{strat}.faiss")

    if not os.path.exists(faiss_path):
        pytest.skip(f"FAISS index for {strat} not generated")

    table = pq.read_table(parquet_path)
    rows = table.to_pylist()
    index = faiss.read_index(faiss_path)

    # Perform sample search with random vector
    sample_vec = np.random.randn(1, 384).astype(np.float32)
    faiss.normalize_L2(sample_vec)
    scores, indices = index.search(sample_vec, k=3)

    for idx in indices[0]:
        assert 0 <= idx < len(rows), f"Retrieved index {idx} out of range for {strat}"
        assert len(rows[idx]["chunk_id"]) > 0, f"Retrieved chunk at index {idx} has invalid ID"
