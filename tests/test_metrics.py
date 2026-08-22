"""
Pytest Unit Tests for Retrieval Metrics (TASK 11).
"""

import pytest
import math
from evaluation.metrics import (
    calculate_hit_at_k,
    calculate_recall_at_k,
    calculate_mrr_at_k,
    calculate_ndcg_at_k
)


def test_hit_at_k():
    retrieved = ["doc1", "doc2", "doc3", "doc4"]
    gt = {"doc3"}

    assert calculate_hit_at_k(retrieved, gt, k=1) == 0.0
    assert calculate_hit_at_k(retrieved, gt, k=2) == 0.0
    assert calculate_hit_at_k(retrieved, gt, k=3) == 1.0
    assert calculate_hit_at_k(retrieved, gt, k=5) == 1.0


def test_recall_at_k():
    retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
    gt = {"doc2", "doc4", "doc6"}

    # In top 2: doc2 (1 / 3)
    assert pytest.approx(calculate_recall_at_k(retrieved, gt, k=2), 0.001) == 1.0 / 3.0

    # In top 5: doc2, doc4 (2 / 3)
    assert pytest.approx(calculate_recall_at_k(retrieved, gt, k=5), 0.001) == 2.0 / 3.0


def test_mrr_at_k():
    retrieved = ["doc1", "doc2", "doc3", "doc4"]
    gt = {"doc3", "doc1"}

    # First hit at rank 1 (doc1) -> MRR = 1.0 / 1 = 1.0
    assert calculate_mrr_at_k(retrieved, gt, k=5) == 1.0

    retrieved2 = ["doc5", "doc6", "doc1", "doc4"]
    # First hit at rank 3 (doc1) -> MRR = 1.0 / 3.0
    assert calculate_mrr_at_k(retrieved2, gt, k=5) == pytest.approx(1.0 / 3.0, 0.001)


def test_ndcg_at_k():
    retrieved = ["doc1", "doc2", "doc3"]
    gt = {"doc1"}

    # Hit at rank 1 -> DCG = 1/log2(2) = 1.0, IDCG = 1.0 -> nDCG = 1.0
    assert calculate_ndcg_at_k(retrieved, gt, k=3) == 1.0

    retrieved2 = ["doc2", "doc1", "doc3"]
    # Hit at rank 2 -> DCG = 1/log2(3) = 0.6309, IDCG = 1.0 -> nDCG = 0.6309
    expected_ndcg = (1.0 / math.log2(3)) / (1.0 / math.log2(2))
    assert pytest.approx(calculate_ndcg_at_k(retrieved2, gt, k=3), 0.001) == expected_ndcg
