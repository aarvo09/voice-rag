"""
Production Pipeline & Retrieval Policy Configuration (TASK 12).
Defines dataclasses and parameters controlling retrieval policies, confidence thresholds, and fallback behavior.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class RetrievalPolicy:
    """
    Configuration policy for ProductionRetriever.
    Controls top-k limits, confidence thresholds, and fallback flags.
    """
    dense_top_k: int = 5
    fallback_top_k: int = 5
    min_dense_score: float = 0.80
    fallback_enabled: bool = True
    final_top_k: int = 5


DEFAULT_RETRIEVAL_POLICY = RetrievalPolicy()
