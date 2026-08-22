"""
Pydantic data models for structured LLM Generation response (TASK 13).
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class GenerationResponse(BaseModel):
    """
    Pydantic schema for structured RAG generation output.
    Enforces types, range constraints on confidence, and citation list.
    """
    answer: str = Field(..., description="Grounded answer string to user query.")
    grounded: bool = Field(..., description="True if answer is strictly grounded in retrieved context.")
    citations: List[str] = Field(default_factory=list, description="List of document_ids supporting the answer.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Self-reported confidence score between 0.0 and 1.0.")
    status: str = Field(default="success", description="Pipeline status: 'success', 'refused', etc.")
    refusal_reason: Optional[str] = Field(default=None, description="Detailed reason if response was refused.")
    provider: Optional[str] = Field(default=None, description="LLM provider name.")
    model: Optional[str] = Field(default=None, description="LLM model identifier.")
