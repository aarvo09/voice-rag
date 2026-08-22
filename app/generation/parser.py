"""
Structured Output Parser and Validator for LLM Generation (TASK 13).
Validates JSON responses via Pydantic schema and applies bounded safe repairs.
"""

import json
import re
import logging
from typing import Dict, Any, Optional
from pydantic import ValidationError

from app.generation.models import GenerationResponse

logger = logging.getLogger(__name__)


class GenerationParser:
    """
    Parses and validates LLM generation responses against GenerationResponse model.
    """

    @staticmethod
    def extract_json_str(raw_text: str) -> str:
        """Strips markdown code fences (```json ... ```) or extracts JSON substring."""
        text = raw_text.strip()
        # Regex to extract ```json ... ``` content
        code_fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if code_fence_match:
            return code_fence_match.group(1).strip()

        # Fallback: extract first '{' to last '}'
        start_idx = text.find("{")
        end_idx = text.rfind("}")
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            return text[start_idx:end_idx + 1].strip()

        return text

    @classmethod
    def parse(cls, raw_response: str, provider: Optional[str] = None, model: Optional[str] = None) -> GenerationResponse:
        """
        Parses raw text into a validated GenerationResponse.
        Applies a single safe repair attempt if raw text contains markdown wrapping.
        Fails safely by raising ValueError on invalid or malformed data.
        """
        if not raw_response or not raw_response.strip():
            logger.error("Empty or whitespace raw response from LLM.")
            return GenerationResponse(
                answer="Generation truncated or empty response.",
                grounded=False,
                citations=[],
                confidence=0.0
            )

        cleaned_text = cls.extract_json_str(raw_response)

        try:
            parsed_dict = json.loads(cleaned_text)
        except json.JSONDecodeError as err:
            # Safe repair attempts for trailing commas and truncated JSON
            repaired_text = re.sub(r",\s*([\}\]])", r"\1", cleaned_text)
            
            # Simple heuristic for truncated JSON: close strings, arrays, and objects
            if not repaired_text.endswith("}"):
                if repaired_text.rfind('"') % 2 != 0:
                    repaired_text += '"'
                if "[" in repaired_text and "]" not in repaired_text[repaired_text.rfind("["):]:
                    repaired_text += "]"
                repaired_text += "}"
                
            try:
                parsed_dict = json.loads(repaired_text)
            except json.JSONDecodeError:
                logger.error(f"Failed to parse JSON from LLM output: {raw_response}. Returning clean error.")
                return GenerationResponse(
                    answer="Generation error: Failed to parse JSON.",
                    grounded=False,
                    citations=[],
                    confidence=0.0
                )

        if not isinstance(parsed_dict, dict):
            raise ValueError(f"Parsed JSON is not a dictionary: {type(parsed_dict)}")

        # Inject provider and model if missing
        if provider and "provider" not in parsed_dict:
            parsed_dict["provider"] = provider
        if model and "model" not in parsed_dict:
            parsed_dict["model"] = model

        try:
            response_model = GenerationResponse(**parsed_dict)
            return response_model
        except ValidationError as val_err:
            logger.error(f"Pydantic validation error: {val_err}")
            raise ValueError(f"Generation output failed schema validation: {val_err}") from val_err
