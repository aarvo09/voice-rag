"""
Unit tests for GeminiLLMProvider and LLMProvider factory (TASK 15).
Runs without requiring external API keys.
"""

import os
import pytest
from unittest.mock import MagicMock, patch

from app.generation.config import LLMConfig
from app.generation.gemini import GeminiLLMProvider
from app.generation.llm import MockLLMProvider, get_llm_provider
from app.generation.parser import GenerationParser


def test_missing_google_api_key_raises_error(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    config = LLMConfig(provider="gemini", model_name="gemini-2.5-flash")
    provider = GeminiLLMProvider(config=config)

    assert provider.api_key is None
    with pytest.raises(ValueError, match="Missing environment variable"):
        provider.generate("query", "context")


def test_factory_returns_mock_when_forced():
    config = LLMConfig(provider="gemini", model_name="gemini-2.5-flash")
    provider = get_llm_provider(config=config, force_mock=True)

    assert isinstance(provider, MockLLMProvider)


def test_factory_returns_mock_when_provider_is_mock():
    config = LLMConfig(provider="mock", model_name="mock-model")
    provider = get_llm_provider(config=config)

    assert isinstance(provider, MockLLMProvider)


def test_gemini_response_parsing_success():
    raw_response = '{"answer": "मैनहट्टन परियोजना द्वितीय विश्व युद्ध की परमाणु बम शोध परियोजना थी।", "grounded": true, "citations": ["1185869_8"], "confidence": 0.95}'

    parsed = GenerationParser.parse(raw_response, provider="gemini", model="gemini-2.5-flash")

    assert parsed.answer == "मैनहट्टन परियोजना द्वितीय विश्व युद्ध की परमाणु बम शोध परियोजना थी।"
    assert parsed.grounded is True
    assert parsed.citations == ["1185869_8"]
    assert parsed.confidence == 0.95


def test_gemini_provider_init_with_api_key(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test_mock_key_12345")
    config = LLMConfig(provider="gemini", model_name="gemini-2.5-flash")

    with patch("google.genai.Client") as mock_client_cls:
        provider = GeminiLLMProvider(config=config)
        assert provider.api_key == "test_mock_key_12345"
        mock_client_cls.assert_called_once_with(api_key="test_mock_key_12345")
