"""
LLM Provider implementations (TASK 15).
Includes GeminiLLMProvider, HTTPLLMProvider, and MockLLMProvider for tests/dry-run.
"""

import json
import logging
import httpx
from typing import Dict, Any, Optional

from app.generation.interface import LLMProvider, LLMResult
from app.generation.config import LLMConfig, DEFAULT_LLM_CONFIG
from app.generation.prompts import GROUNDED_SYSTEM_PROMPT, build_user_prompt
from app.generation.parser import GenerationParser
from app.generation.gemini import GeminiLLMProvider

logger = logging.getLogger(__name__)


class MockLLMProvider(LLMProvider):
    """
    Mock LLM provider used for unit tests and dry-run execution without active API credentials.
    Explicitly labeled as a test implementation.
    """

    def __init__(self, provider_name: str = "mock_provider", model_name: str = "mock-grounded-v1"):
        self.provider_name = provider_name
        self.model_name = model_name

    def generate(
        self,
        query: str,
        context_str: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> LLMResult:
        logger.info(f"Executing MockLLMProvider for query: '{query}'")

        citations = []
        if "document_id:" in context_str:
            lines = context_str.split("\n")
            for line in lines:
                if line.startswith("document_id:"):
                    doc_id = line.split(":", 1)[1].strip()
                    citations.append(doc_id)
                    break

        if not citations:
            citations = ["1185869_8"]

        mock_payload = {
            "answer": f"[MOCK TEST RESPONSE] Grounded answer based on context for query: {query}",
            "grounded": True,
            "citations": citations,
            "confidence": 0.95
        }
        raw_str = json.dumps(mock_payload)

        parsed = GenerationParser.parse(raw_str, provider=self.provider_name, model=self.model_name)

        return LLMResult(
            answer=parsed.answer,
            grounded=parsed.grounded,
            citations=parsed.citations,
            confidence=parsed.confidence,
            raw_response=raw_str,
            provider=self.provider_name,
            model=self.model_name
        )


class GroqLLMProvider(LLMProvider):
    """
    REST HTTP LLM provider targeting Groq's OpenAI-compatible Chat Completion API.
    """

    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or DEFAULT_LLM_CONFIG
        self.api_key = self.config.get_api_key()
        self.base_url = self.config.base_url or "https://api.groq.com/openai/v1/chat/completions"
        self.provider_name = self.config.provider
        self.model_name = self.config.model_name

    def generate(
        self,
        query: str,
        context_str: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> LLMResult:
        if not self.api_key:
            raise ValueError(
                f"Missing API key in environment variable '{self.config.api_key_env_name}'. "
                "Set credentials or use dry-run/MockLLMProvider mode."
            )

        language = metadata.get("language") if metadata else None
        user_prompt = build_user_prompt(query, context_str, language=language)
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        schema = {
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
                "grounded": {"type": "boolean"},
                "citations": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "confidence": {"type": "number"}
            },
            "required": ["answer", "grounded", "citations", "confidence"],
            "additionalProperties": False
        }

        payload = {
            "model": self.config.model_name,
            "messages": [
                {"role": "system", "content": GROUNDED_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": self.config.temperature,
            "max_completion_tokens": 128,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "rag_response",
                    "strict": True,
                    "schema": schema
                }
            },
            "include_reasoning": False,
            "reasoning_effort": "low"
        }

        timeout_sec = self.config.request_timeout_ms / 1000.0

        try:
            with httpx.Client(timeout=timeout_sec) as client:
                response = client.post(self.base_url, headers=headers, json=payload)
                response.raise_for_status()

            res_json = response.json()
            message_obj = res_json["choices"][0]["message"]
            raw_content = message_obj.get("content", "")

            parsed = GenerationParser.parse(
                raw_content,
                provider=self.config.provider,
                model=self.config.model_name
            )

            return LLMResult(
                answer=parsed.answer,
                grounded=parsed.grounded,
                citations=parsed.citations,
                confidence=parsed.confidence,
                raw_response=raw_content,
                provider=self.config.provider,
                model=self.config.model_name
            )

        except httpx.HTTPStatusError as e:
            logger.error(f"Groq LLM HTTP error: {e.response.status_code} - {e.response.text}")
            raise
        except Exception as err:
            logger.error(f"Groq LLM generation failed: {err}")
            raise


def get_llm_provider(config: Optional[LLMConfig] = None, force_mock: bool = False) -> LLMProvider:
    """
    Factory function for instantiating LLMProvider.
    Uses MockLLMProvider if force_mock=True, provider='mock', or if API key is unconfigured.
    """
    active_config = config or DEFAULT_LLM_CONFIG

    if force_mock or active_config.provider == "mock" or not active_config.get_api_key():
        logger.info("Initializing MockLLMProvider (dry-run/test mode).")
        return MockLLMProvider(provider_name=f"mock_{active_config.provider}", model_name=active_config.model_name)

    if active_config.provider == "groq":
        logger.info(f"Initializing GroqLLMProvider (model={active_config.model_name}).")
        return GroqLLMProvider(config=active_config)

    if active_config.provider == "gemini":
        logger.info(f"Initializing GeminiLLMProvider (model={active_config.model_name}).")
        return GeminiLLMProvider(config=active_config)

    logger.warning(f"Unknown provider '{active_config.provider}', defaulting to Gemini.")
    return GeminiLLMProvider(config=active_config)
