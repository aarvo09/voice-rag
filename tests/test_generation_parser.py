"""
Unit tests for GenerationParser and Insufficient Retrieval Refusal (TASK 13).
"""

import pytest
from unittest.mock import MagicMock

from app.generation.parser import GenerationParser
from app.generation.models import GenerationResponse
from app.pipeline.text_rag import TextRAGService
from app.generation.llm import MockLLMProvider


def test_parse_clean_json():
    raw_str = '{"answer": "परीक्षण उत्तर", "grounded": true, "citations": ["doc1"], "confidence": 0.95}'
    parsed = GenerationParser.parse(raw_str, provider="test_prov", model="test_mod")

    assert isinstance(parsed, GenerationResponse)
    assert parsed.answer == "परीक्षण उत्तर"
    assert parsed.grounded is True
    assert parsed.citations == ["doc1"]
    assert parsed.confidence == 0.95
    assert parsed.provider == "test_prov"


def test_parse_markdown_code_fence():
    raw_str = """Here is the response:
```json
{
  "answer": "Grounded answer from fence",
  "grounded": true,
  "citations": ["docA"],
  "confidence": 0.88
}
```
Thank you!"""
    parsed = GenerationParser.parse(raw_str)

    assert parsed.answer == "Grounded answer from fence"
    assert parsed.grounded is True
    assert parsed.citations == ["docA"]
    assert parsed.confidence == 0.88


def test_parse_trailing_comma_repair():
    raw_str = '{"answer": "Repaired answer", "grounded": false, "citations": ["docB",], "confidence": 0.50,}'
    parsed = GenerationParser.parse(raw_str)

    assert parsed.answer == "Repaired answer"
    assert parsed.grounded is False
    assert parsed.citations == ["docB"]


def test_parse_malformed_json_rejection():
    raw_str = "This is not JSON at all."
    with pytest.raises(ValueError, match="Invalid JSON string"):
        GenerationParser.parse(raw_str)


def test_parse_invalid_confidence_range():
    raw_str = '{"answer": "Invalid confidence", "grounded": true, "citations": [], "confidence": 2.5}'
    with pytest.raises(ValueError, match="failed schema validation"):
        GenerationParser.parse(raw_str)


def test_insufficient_retrieval_skips_llm():
    mock_retriever = MagicMock()
    # Mock retriever returning empty candidates
    mock_retriever.retrieve.return_value = {
        "results": [],
        "confidence": {"decision": "LOW_CONFIDENCE", "confidence_score": 0.1},
        "fallback_used": False
    }

    mock_llm = MagicMock(spec=MockLLMProvider)
    service = TextRAGService(retriever=mock_retriever, provider=mock_llm)

    res = service.run("अज्ञात प्रश्न")

    assert res["grounded"] is False
    assert res["citations"] == []
    assert res["confidence"] == 0.0
    assert "couldn't find enough relevant information" in res["answer"]
    # Verify LLM call was SKIPPED
    mock_llm.generate.assert_not_called()
