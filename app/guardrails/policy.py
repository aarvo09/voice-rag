"""
Guardrail Policy and Action Handler (TASK 14).
Defines configurable policy rules for input safety refusals, retrieval sufficiency refusals,
and grounding retry / refusal logic.
"""

import logging
from dataclasses import dataclass
from typing import Dict, Any

from app.guardrails.input import InputSafetyResult
from app.guardrails.relevance import RetrievalRelevanceResult
from app.guardrails.grounding import GroundingResult

logger = logging.getLogger(__name__)

DEFAULT_REFUSAL_MESSAGE = "No relevant context found in the provided knowledge base."


@dataclass
class GuardrailPolicyConfig:
    """Configuration parameters for Guardrail Policy."""
    unsafe_query_action: str = "REFUSE"
    low_relevance_action: str = "REFUSE"
    ungrounded_answer_action: str = "RETRY"
    max_grounding_retries: int = 0
    minimum_grounding_confidence: float = 0.50
    refusal_message: str = DEFAULT_REFUSAL_MESSAGE


class GuardrailPolicy:
    """
    Orchestrates actions (REFUSE, RETRY, RETURN) based on guardrail evaluation results.
    """

    def __init__(self, config: Optional[GuardrailPolicyConfig] = None):
        self.config = config or GuardrailPolicyConfig()

    def build_refusal_response(self, query: str, refusal_reason: str, status: str = "refused") -> Dict[str, Any]:
        """Constructs standardized controlled refusal response dictionary."""
        return {
            "query": query,
            "answer": self.config.refusal_message,
            "grounded": False,
            "confidence": 0.0,
            "citations": [],
            "status": status,
            "refusal_reason": refusal_reason,
            "retrieved_documents": []
        }

    def handle_unsafe_input(self, query: str, safety_result: InputSafetyResult) -> Dict[str, Any]:
        """Refuses unsafe or prompt-injection queries immediately."""
        reason_str = f"input_safety_{safety_result.category.lower()}"
        logger.info(f"Refusing query '{query}': {reason_str}")
        return self.build_refusal_response(query, refusal_reason=reason_str, status="refused_unsafe")

    def handle_insufficient_retrieval(self, query: str, relevance_result: RetrievalRelevanceResult) -> Dict[str, Any]:
        """Refuses queries with insufficient retrieval context evidence."""
        logger.info(f"Refusing query '{query}': insufficient_retrieval")
        return self.build_refusal_response(query, refusal_reason="insufficient_retrieval", status="refused_insufficient")

    def should_retry_grounding(self, grounding_result: GroundingResult, current_retry_count: int) -> bool:
        """Determines if generation should be retried due to ungrounded claims."""
        if not grounding_result.grounded and current_retry_count < self.config.max_grounding_retries:
            return True
        return False

    def handle_ungrounded_final(self, query: str, grounding_result: GroundingResult, generated_answer: str = None) -> Dict[str, Any]:
        """Refuses final response if answer remains ungrounded, but preserves generated answer."""
        logger.info(f"Refusing query '{query}': ungrounded_answer")
        resp = self.build_refusal_response(query, refusal_reason="ungrounded_answer", status="ungrounded")
        if generated_answer:
            resp["answer"] = generated_answer
        return resp
