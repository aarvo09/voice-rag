"""
Unit tests for InputSafetyGuardrail (TASK 14).
"""

import json
import os
import pytest
from app.guardrails.input import InputSafetyGuardrail

FIXTURES_PATH = os.path.join(os.path.dirname(__file__), "fixtures", "guardrail_cases.json")


@pytest.fixture
def test_cases():
    with open(FIXTURES_PATH, "r") as f:
        return json.load(f)


def test_safe_query_passes(test_cases):
    guard = InputSafetyGuardrail()
    case = test_cases["normal_question"]
    res = guard.validate(case["query"])

    assert res.safe is True
    assert res.category == "SAFE"


def test_empty_input_rejected(test_cases):
    guard = InputSafetyGuardrail()
    case = test_cases["empty_input"]
    res = guard.validate(case["query"])

    assert res.safe is False
    assert res.category == "EMPTY_INPUT"


def test_prompt_injection_detected(test_cases):
    guard = InputSafetyGuardrail()
    case = test_cases["prompt_injection"]
    res = guard.validate(case["query"])

    assert res.safe is False
    assert res.category == "PROMPT_INJECTION"


def test_unsafe_request_rejected(test_cases):
    guard = InputSafetyGuardrail()
    case = test_cases["unsafe_request"]
    res = guard.validate(case["query"])

    assert res.safe is False
    assert res.category == "UNSAFE_REQUEST"
