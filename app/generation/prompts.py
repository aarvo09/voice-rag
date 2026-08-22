"""
Prompt templates and builders for grounded LLM RAG generation (TASK 13).
"""

from typing import Dict, Any, Optional

GROUNDED_SYSTEM_PROMPT = """You are a precise, factual AI assistant in a grounded Voice RAG system.
CRITICAL RULES:
- answer in 1-3 concise sentences
- use only retrieved context
- include citations
If not in context, set "grounded": false, "answer": "No info.", "citations": [], and "confidence": 0.0.
In "citations", list ONLY the exact document_ids.

Answer ONLY from the supplied retrieved context.
Return ONLY the JSON object matching the provided schema.
Do not include reasoning, analysis, markdown, or explanation outside the JSON.
"""


def build_user_prompt(query: str, context_str: str, language: Optional[str] = None) -> str:
    """
    Combines formatted context string and user query into the final user prompt.
    """
    lang_instruction = f"\n(Please answer in {language})\n" if language else ""
    return f"Retrieved Context:\n{context_str}\n\nUser Query: {query}\n{lang_instruction}\nJSON Output:"
