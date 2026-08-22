"""
Generation package for LLM providers, prompts, context formatting, and structured output parsing (TASK 15).
"""

from app.generation.interface import LLMProvider, LLMResult
from app.generation.config import LLMConfig, DEFAULT_LLM_CONFIG
from app.generation.models import GenerationResponse
from app.generation.context import ContextBuilder
from app.generation.prompts import GROUNDED_SYSTEM_PROMPT, build_user_prompt
from app.generation.parser import GenerationParser
from app.generation.gemini import GeminiLLMProvider
from app.generation.llm import GroqLLMProvider, MockLLMProvider, get_llm_provider

__all__ = [
    "LLMProvider",
    "LLMResult",
    "LLMConfig",
    "DEFAULT_LLM_CONFIG",
    "GenerationResponse",
    "ContextBuilder",
    "GROUNDED_SYSTEM_PROMPT",
    "build_user_prompt",
    "GenerationParser",
    "GeminiLLMProvider",
    "GroqLLMProvider",
    "MockLLMProvider",
    "get_llm_provider"
]
