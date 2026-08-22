"""
Unit tests for GenerationResponse Pydantic model (TASK 13).
"""

import pytest
from pydantic import ValidationError
from app.generation.models import GenerationResponse


def test_valid_generation_response():
    resp = GenerationResponse(
        answer="मैनहट्टन परियोजना एक शोध परियोजना थी।",
        grounded=True,
        citations=["1185869_0"],
        confidence=0.92,
        provider="openai",
        model="gpt-4o-mini"
    )
    assert resp.answer == "मैनहट्टन परियोजना एक शोध परियोजना थी।"
    assert resp.grounded is True
    assert resp.citations == ["1185869_0"]
    assert resp.confidence == 0.92
    assert resp.provider == "openai"


def test_invalid_confidence_above_one():
    with pytest.raises(ValidationError):
        GenerationResponse(
            answer="Test answer",
            grounded=True,
            citations=["doc1"],
            confidence=1.5
        )


def test_invalid_confidence_below_zero():
    with pytest.raises(ValidationError):
        GenerationResponse(
            answer="Test answer",
            grounded=True,
            citations=["doc1"],
            confidence=-0.1
        )


def test_missing_required_answer():
    with pytest.raises(ValidationError):
        GenerationResponse(
            grounded=True,
            citations=["doc1"],
            confidence=0.8
        )


def test_missing_required_grounded():
    with pytest.raises(ValidationError):
        GenerationResponse(
            answer="Test answer",
            citations=["doc1"],
            confidence=0.8
        )
