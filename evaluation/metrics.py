"""
Retrieval Metrics Harness (TASK 11).

Provides standard Information Retrieval (IR) evaluation metrics:
- Recall@K
- Hit@K
- MRR@K (Mean Reciprocal Rank)
- nDCG@K (Normalized Discounted Cumulative Gain)
"""

import math
from typing import List, Set, Union


def calculate_hit_at_k(retrieved_ids: List[str], ground_truth_ids: Union[Set[str], List[str]], k: int) -> float:
    """
    Returns 1.0 if at least one retrieved document in top-k is in ground_truth_ids, else 0.0.
    """
    gt_set = set(ground_truth_ids)
    if not gt_set:
        return 0.0
    for doc_id in retrieved_ids[:k]:
        if doc_id in gt_set:
            return 1.0
    return 0.0


def calculate_recall_at_k(retrieved_ids: List[str], ground_truth_ids: Union[Set[str], List[str]], k: int) -> float:
    """
    Calculates proportion of ground_truth_ids retrieved in top-k.
    """
    gt_set = set(ground_truth_ids)
    if not gt_set:
        return 0.0
    hits = sum(1 for doc_id in retrieved_ids[:k] if doc_id in gt_set)
    return float(hits) / float(len(gt_set))


def calculate_mrr_at_k(retrieved_ids: List[str], ground_truth_ids: Union[Set[str], List[str]], k: int) -> float:
    """
    Calculates Reciprocal Rank of the first relevant document retrieved in top-k.
    """
    gt_set = set(ground_truth_ids)
    if not gt_set:
        return 0.0
    for rank, doc_id in enumerate(retrieved_ids[:k], start=1):
        if doc_id in gt_set:
            return 1.0 / float(rank)
    return 0.0


def calculate_ndcg_at_k(retrieved_ids: List[str], ground_truth_ids: Union[Set[str], List[str]], k: int) -> float:
    """
    Calculates Normalized Discounted Cumulative Gain at k (binary relevance).
    """
    gt_set = set(ground_truth_ids)
    if not gt_set:
        return 0.0

    # 1. Calculate DCG@k
    dcg = 0.0
    for rank, doc_id in enumerate(retrieved_ids[:k], start=1):
        rel = 1.0 if doc_id in gt_set else 0.0
        dcg += rel / math.log2(rank + 1)

    # 2. Calculate IDCG@k (Ideal DCG)
    ideal_hits = min(k, len(gt_set))
    idcg = sum(1.0 / math.log2(r + 1) for r in range(1, ideal_hits + 1))

    if idcg == 0.0:
        return 0.0
    return dcg / idcg
