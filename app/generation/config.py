"""
Configuration dataclass for LLM Providers (TASK 15).
Supports Gemini API and Mock providers with env-var configuration without hardcoding secrets.
"""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LLMConfig:
    """
    Configuration parameters for LLM generation provider.
    Reads provider, model, and API keys dynamically from environment variables.
    """
    provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "gemini"))
    model_name: str = field(
        default_factory=lambda: os.getenv("GROQ_MODEL", "llama-3.1-8b-instant") if os.getenv("LLM_PROVIDER") == "groq" 
        else os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
    )
    temperature: float = 0.0
    max_tokens: int = 128
    request_timeout_ms: int = 15000
    max_context_documents: int = 5
    api_key_env_name: str = field(
        default_factory=lambda: "GROQ_API_KEY" if os.getenv("LLM_PROVIDER") == "groq" else "GOOGLE_API_KEY"
    )
    base_url: Optional[str] = field(
        default_factory=lambda: "https://api.groq.com/openai/v1/chat/completions" if os.getenv("LLM_PROVIDER") == "groq" else None
    )

    def get_api_key(self) -> Optional[str]:
        """Retrieves API key from environment variable safely."""
        return os.getenv(self.api_key_env_name)


DEFAULT_LLM_CONFIG = LLMConfig()
