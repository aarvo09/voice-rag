"""
Unit tests for GuardrailPolicy (TASK 14).
"""

import pytest
from app.guardrails.policy import GuardrailPolicy, GuardrailPolicyConfig
from app.guardrails.input import InputSafetyResult
from app.guardrails.relevance import RetrievalRelevanceResult
from app.guardrails.grounding import GroundingResult


def test_unsafe_request_refuses():
    policy = GuardrailPolicy()
    safety_res = InputSafetyResult(safe=False, category="PROMPT_INJECTION", reason="Injection detected")

    resp = policy.handle_unsafe_input("test query", safety_res)

    assert resp["status"] == "refused_unsafe"
    assert resp["grounded"] is False
    assert resp["refusal_reason"] == "input_safety_prompt_injection"
    assert "couldn't find enough relevant information" in resp["answer"]


def test_weak_retrieval_refuses():
    policy = GuardrailPolicy()
    rel_res = RetrievalRelevanceResult(sufficient=False, decision="INSUFFICIENT", confidence=0.3, reason="Weak similarity score")

    resp = policy.handle_insufficient_retrieval("test query", rel_res)

    assert resp["status"] == "refused_insufficient"
    assert resp["grounded"] is False
    assert resp["refusal_reason"] == "insufficient_retrieval"


def test_grounding_failure_retries_once():
    policy = GuardrailPolicy(GuardrailPolicyConfig(max_grounding_retries=1))
    ground_res = GroundingResult(grounded=False, confidence=0.3, unsupported_claims=["unsupported claim"])

    # Attempt 0: retry should be True
    should_retry = policy.should_retry_grounding(ground_res, current_retry_count=0)
    assert should_retry is True

    # Attempt 1: max retries reached, retry should be False
    should_retry_max = policy.should_retry_grounding(ground_res, current_retry_count=1)
    assert should_retry_max is False


def test_repeated_grounding_failure_refuses():
    policy = GuardrailPolicy()
    ground_res = GroundingResult(grounded=False, confidence=0.2, unsupported_claims=["claim1"])

    resp = policy.handle_ungrounded_final("test query", ground_res)

    assert resp["status"] == "refused_ungrounded"
    assert resp["refusal_reason"] == "ungrounded_answer"
