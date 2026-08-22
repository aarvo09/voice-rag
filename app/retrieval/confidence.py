"""
Lightweight Retrieval Confidence Assessment Module (TASK 12).
Evaluates dense retrieval confidence using cheap numerical signals.
Uses ZERO neural, CrossEncoder, or LLM calls.
"""

import logging
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class ConfidenceAssessment:
    """
    Structured outcome of retrieval confidence assessment.
    """
    confidence_score: float
    decision: str  # "HIGH_CONFIDENCE" or "LOW_CONFIDENCE"
    reason: str
    top1_score: float
    mean_score: float
    score_gap: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class RetrievalConfidenceEvaluator:
    """
    Lightweight evaluator assessing dense retrieval results.
    Computes top-1 score, top-k mean score, and score gap between top-1 and top-2.
    """

    def __init__(self, min_dense_score: float = 0.75):
        self.min_dense_score = min_dense_score

    def evaluate(
        self,
        dense_results: List[Dict[str, Any]],
        min_score_override: Optional[float] = None
    ) -> ConfidenceAssessment:
        """
        Evaluates confidence of dense retrieval candidates.
        """
        threshold = min_score_override if min_score_override is not None else self.min_dense_score

        if not dense_results:
            return ConfidenceAssessment(
                confidence_score=0.0,
                decision="LOW_CONFIDENCE",
                reason="No dense candidates retrieved.",
                top1_score=0.0,
                mean_score=0.0,
                score_gap=0.0
            )

        scores = [float(item.get("score", 0.0)) for item in dense_results]
        top1_score = scores[0]
        mean_score = float(sum(scores) / len(scores))
        score_gap = top1_score - scores[1] if len(scores) > 1 else top1_score

        # Primary signal: top-1 similarity score compared against min_dense_score threshold
        if top1_score >= threshold:
            decision = "HIGH_CONFIDENCE"
            reason = f"Top-1 dense score ({top1_score:.4f}) >= threshold ({threshold:.4f})."
        else:
            decision = "LOW_CONFIDENCE"
            reason = f"Top-1 dense score ({top1_score:.4f}) < threshold ({threshold:.4f})."

        return ConfidenceAssessment(
            confidence_score=top1_score,
            decision=decision,
            reason=reason,
            top1_score=top1_score,
            mean_score=mean_score,
            score_gap=score_gap
        )
