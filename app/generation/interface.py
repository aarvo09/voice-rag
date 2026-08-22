"""
Abstract LLM Provider interface and result data structures (TASK 13).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class LLMResult:
    """
    Standardized result structure returned by any LLMProvider.
    """
    answer: str
    grounded: bool
    citations: List[str] = field(default_factory=list)
    confidence: float = 0.0
    raw_response: str = ""
    provider: str = "unknown"
    model: str = "unknown"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.answer,
            "grounded": self.grounded,
            "citations": self.citations,
            "confidence": self.confidence,
            "raw_response": self.raw_response,
            "provider": self.provider,
            "model": self.model
        }


class LLMProvider(ABC):
    """
    Abstract Base Class for LLM providers.
    Ensures provider decoupling from pipeline components.
    """

    @abstractmethod
    def generate(
        self,
        query: str,
        context_str: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> LLMResult:
        """
        Generates a grounded response given a query and formatted context string.
        """
        pass
