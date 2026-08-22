"""
RAG Pipeline Execution State Data Structure (TASK 14).
Tracks execution state across retrieval, generation, and guardrail evaluation phases.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class RAGState:
    """
    State object tracking metadata, latency, and guardrail outputs across pipeline execution.
    """
    query: str
    input_safety_status: str = "pending"
    input_safety_category: str = "unknown"
    retrieval_sufficiency: str = "pending"
    retrieval_guardrail_reason: str = ""
    retrieved_documents: List[Dict[str, Any]] = field(default_factory=list)
    formatted_context: str = ""
    llm_raw_response: str = ""
    answer: str = ""
    grounded: bool = False
    citations: List[str] = field(default_factory=list)
    confidence: float = 0.0
    grounding_status: str = "pending"
    grounding_confidence: float = 0.0
    unsupported_claims: List[str] = field(default_factory=list)
    citation_validation: bool = False
    retry_count: int = 0
    status: str = "initialized"
    refusal_reason: Optional[str] = None
    telemetry: Dict[str, float] = field(default_factory=dict)
