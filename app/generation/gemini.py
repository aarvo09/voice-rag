"""
Google Gemini LLM Provider implementation using official google-genai SDK (TASK 15).
Enforces structured JSON output and handles API errors, timeouts, and quota errors safely.
"""

import time
import logging
from typing import Dict, Any, Optional

from google import genai
from google.genai import types
from google.genai.errors import APIError

from app.generation.interface import LLMProvider, LLMResult
from app.generation.config import LLMConfig, DEFAULT_LLM_CONFIG
from app.generation.prompts import GROUNDED_SYSTEM_PROMPT, build_user_prompt
from app.generation.parser import GenerationParser
from app.generation.models import GenerationResponse

logger = logging.getLogger(__name__)


class GeminiLLMProvider(LLMProvider):
    """
    Official Google Gemini API provider integration.
    Reads GOOGLE_API_KEY from environment and sends structured RAG prompts.
    """

    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or DEFAULT_LLM_CONFIG
        self.api_key = self.config.get_api_key()
        if not self.api_key:
            logger.warning(
                f"Missing API key in environment variable '{self.config.api_key_env_name}'. "
                "Gemini provider calls will fail unless API key is set or mock mode is enabled."
            )
            self.client = None
        else:
            self.client = genai.Client(api_key=self.api_key)

        self.provider_name = "gemini"
        self.model_name = self.config.model_name

    def generate(
        self,
        query: str,
        context_str: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> LLMResult:
        if not self.api_key or not self.client:
            raise ValueError(
                f"Missing environment variable '{self.config.api_key_env_name}'. "
                "Set GOOGLE_API_KEY or use MockLLMProvider for dry-run/test mode."
            )

        t_prep_start = time.perf_counter()
        user_prompt = build_user_prompt(query, context_str)
        prep_ms = round((time.perf_counter() - t_prep_start) * 1000.0, 2)

        gen_config = types.GenerateContentConfig(
            system_instruction=GROUNDED_SYSTEM_PROMPT,
            temperature=self.config.temperature,
            max_output_tokens=max(self.config.max_tokens, 1024),
            response_mime_type="application/json",
            response_schema=GenerationResponse,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)
        )

        max_attempts = 2
        raw_text = ""
        net_ms = 0.0

        for attempt in range(1, max_attempts + 1):
            t_net_start = time.perf_counter()
            try:
                response = self.client.models.generate_content(
                    model=self.config.model_name,
                    contents=user_prompt,
                    config=gen_config
                )
                net_ms = round((time.perf_counter() - t_net_start) * 1000.0, 2)

                if hasattr(response, "text") and response.text:
                    raw_text = response.text
                elif hasattr(response, "candidates") and response.candidates:
                    candidate = response.candidates[0]
                    if hasattr(candidate, "content") and candidate.content and candidate.content.parts:
                        raw_text = candidate.content.parts[0].text or ""

                if not raw_text or not raw_text.strip():
                    raise ValueError("Received empty content response from Gemini API.")

                break

            except APIError as api_err:
                net_ms = round((time.perf_counter() - t_net_start) * 1000.0, 2)
                err_code = getattr(api_err, "code", None)
                err_msg = str(api_err)
                is_503 = (
                    err_code in [503, "503", "UNAVAILABLE"]
                    or "503" in err_msg
                    or "UNAVAILABLE" in err_msg
                    or "high demand" in err_msg.lower()
                )

                if is_503 and attempt < max_attempts:
                    logger.warning(
                        f"Gemini API 503 Service Unavailable (attempt {attempt}/{max_attempts}). "
                        "Retrying in 1.0s..."
                    )
                    time.sleep(1.0)
                    continue

                if is_503:
                    logger.error(f"Gemini API 503 Provider Temporarily Unavailable ({err_code}): {api_err.message}")
                    raise RuntimeError(f"Gemini provider temporarily unavailable (503): {api_err.message}") from api_err

                logger.error(f"Gemini API Error ({err_code}): {api_err.message}")
                raise RuntimeError(f"Gemini API call failed ({err_code}): {api_err.message}") from api_err

            except Exception as err:
                net_ms = round((time.perf_counter() - t_net_start) * 1000.0, 2)
                logger.error(f"Gemini generation request failed: {err}")
                raise

        t_parse_start = time.perf_counter()
        parsed = GenerationParser.parse(
            raw_text,
            provider=self.provider_name,
            model=self.model_name
        )
        parse_ms = round((time.perf_counter() - t_parse_start) * 1000.0, 2)

        total_gen_ms = round(prep_ms + net_ms + parse_ms, 2)

        return LLMResult(
            answer=parsed.answer,
            grounded=parsed.grounded,
            citations=parsed.citations,
            confidence=parsed.confidence,
            raw_response=raw_text,
            provider=self.provider_name,
            model=self.model_name
        )
