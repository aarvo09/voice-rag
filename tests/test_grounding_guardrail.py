"""
Unit tests for GroundingValidator (TASK 14).
"""

import pytest
from app.guardrails.grounding import GroundingValidator


@pytest.fixture
def retrieved_docs():
    return [
        {
            "document_id": "1185869_8",
            "score": 0.88,
            "text": "मैनहट्टन परियोजना द्वितीय विश्व युद्ध के दौरान पहली परमाणु बम विकसित करने के लिए एक शोध उपक्रम था।"
        }
    ]


def test_supported_answer_passes(retrieved_docs):
    validator = GroundingValidator()
    answer = "मैनहट्टन परियोजना द्वितीय विश्व युद्ध में परमाणु बम विकसित करने की शोध परियोजना थी।"
    citations = ["1185869_8"]

    res = validator.validate("प्रश्न", answer, citations, retrieved_docs)

    assert res.grounded is True
    assert res.citations_valid is True
    assert res.unsupported_claims == []


def test_unsupported_answer_fails(retrieved_docs):
    validator = GroundingValidator()
    answer = "मैनहट्टन परियोजना में 10 लाख रोबोट सेना बनाई गई थी जो स्पेस एलियंस से लड़ती थी।"
    citations = ["1185869_8"]

    res = validator.validate("प्रश्न", answer, citations, retrieved_docs)

    assert res.grounded is False
    assert len(res.unsupported_claims) > 0


def test_invalid_citation_fails(retrieved_docs):
    validator = GroundingValidator()
    answer = "मैनहट्टन परियोजना द्वितीय विश्व युद्ध की शोध परियोजना थी।"
    citations = ["fabricated_doc_999"]

    res = validator.validate("प्रश्न", answer, citations, retrieved_docs)

    assert res.grounded is False
    assert res.citations_valid is False


def test_duplicate_citation_fails(retrieved_docs):
    validator = GroundingValidator()
    answer = "मैनहट्टन परियोजना द्वितीय विश्व युद्ध की शोध परियोजना थी।"
    citations = ["1185869_8", "1185869_8"]

    res = validator.validate("प्रश्न", answer, citations, retrieved_docs)

    assert res.grounded is False
    assert res.citations_valid is False
