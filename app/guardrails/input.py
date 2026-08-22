"""
Input Safety Guardrail (TASK 14).
Detects empty input, prompt injection attempts, system prompt exfiltration, and unsafe requests.
"""

import re
import logging
from dataclasses import dataclass
from typing import Dict, Any

logger = logging.getLogger(__name__)

# Injection & System Exfiltration Patterns (English + Hindi)
PROMPT_INJECTION_PATTERNS = [
    r"ignore\s+(previous|all)\s+instructions",
    r"disregard\s+all\s+(prior|previous)\s+rules",
    r"forget\s+all\s+instructions",
    r"you\s+are\s+now\s+a",
    r"system\s+prompt",
    r"reveal\s+(your\s+)?(system|initial)\s+(prompt|instructions)",
    r"print\s+(your\s+)?system\s+prompt",
    r"developer\s+mode",
    r"jailbreak",
    r"override\s+safety\s+guidelines",
    r"पिछला\s+निर्देश\s+भूल\s+जाओ",
    r"सिस्टम\s+प्रॉम्प्ट",
    r"नियमों\s+को\s+अमान्य\s+करो"
]

UNSAFE_REQUEST_PATTERNS = [
    r"how\s+to\s+build\s+a\s+bomb",
    r"make\s+(explosives|weapons)",
    r"how\s+to\s+hack\s+into",
    r"suicide\s+instructions",
    r"बम\s+कैसे\s+बनाएं",
    r"हैक\s+कैसे\s+करें"
]


@dataclass
class InputSafetyResult:
    """Result returned by InputSafetyGuardrail."""
    safe: bool
    category: str
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "safe": self.safe,
            "category": self.category,
            "reason": self.reason
        }


class InputSafetyGuardrail:
    """
    Application-level lightweight input safety evaluator.
    """

    def __init__(self, max_query_length: int = 1000):
        self.max_query_length = max_query_length

    def validate(self, query: str) -> InputSafetyResult:
        """Validates user query against safety rules."""
        if not query or not query.strip():
            return InputSafetyResult(
                safe=False,
                category="EMPTY_INPUT",
                reason="User query is empty or contains only whitespace."
            )

        cleaned_query = query.strip()

        if len(cleaned_query) > self.max_query_length:
            return InputSafetyResult(
                safe=False,
                category="UNSAFE_REQUEST",
                reason=f"Query length ({len(cleaned_query)}) exceeds maximum allowed limit ({self.max_query_length})."
            )

        # Check prompt injection
        for pattern in PROMPT_INJECTION_PATTERNS:
            if re.search(pattern, cleaned_query, re.IGNORECASE):
                logger.warning(f"Prompt injection pattern detected: '{pattern}' in query.")
                return InputSafetyResult(
                    safe=False,
                    category="PROMPT_INJECTION",
                    reason=f"Query contains prompt-injection or instruction override attempt ('{pattern}')."
                )

        # Check unsafe requests
        for pattern in UNSAFE_REQUEST_PATTERNS:
            if re.search(pattern, cleaned_query, re.IGNORECASE):
                logger.warning(f"Unsafe request pattern detected: '{pattern}' in query.")
                return InputSafetyResult(
                    safe=False,
                    category="UNSAFE_REQUEST",
                    reason="Query requests unsafe or disallowed activities."
                )

        return InputSafetyResult(
            safe=True,
            category="SAFE",
            reason="Query passed input safety validation."
        )
