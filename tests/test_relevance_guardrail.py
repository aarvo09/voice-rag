"""
Unit tests for RetrievalRelevanceGuardrail (TASK 14).
"""

import pytest
from app.guardrails.relevance import RetrievalRelevanceGuardrail


def test_strong_retrieval_passes():
    guard = RetrievalRelevanceGuardrail(min_top1_score=0.60)
    retrieval_output = {
        "results": [{"document_id": "doc1", "score": 0.85}],
        "confidence": {"confidence_score": 0.85, "decision": "HIGH_CONFIDENCE"},
        "fallback_used": False
    }

    res = guard.evaluate(retrieval_output)
    assert res.sufficient is True
    assert res.decision == "SUFFICIENT"
    assert res.confidence == 0.85


def test_weak_retrieval_fails():
    guard = RetrievalRelevanceGuardrail(min_top1_score=0.60)
    retrieval_output = {
        "results": [{"document_id": "doc1", "score": 0.35}],
        "confidence": {"confidence_score": 0.35, "decision": "LOW_CONFIDENCE"},
        "fallback_used": False
    }

    res = guard.evaluate(retrieval_output)
    assert res.sufficient is False
    assert res.decision == "INSUFFICIENT"


def test_empty_retrieval_fails():
    guard = RetrievalRelevanceGuardrail()
    retrieval_output = {
        "results": [],
        "confidence": {"confidence_score": 0.0, "decision": "LOW_CONFIDENCE"},
        "fallback_used": False
    }

    res = guard.evaluate(retrieval_output)
    assert res.sufficient is False
    assert res.decision == "INSUFFICIENT"


def test_threshold_is_configurable():
    guard_high = RetrievalRelevanceGuardrail(min_top1_score=0.90)
    retrieval_output = {
        "results": [{"document_id": "doc1", "score": 0.85}],
        "confidence": {"confidence_score": 0.85, "decision": "HIGH_CONFIDENCE"},
        "fallback_used": False
    }

    res = guard_high.evaluate(retrieval_output)
    assert res.sufficient is False
