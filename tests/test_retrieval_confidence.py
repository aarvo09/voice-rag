"""
Unit tests for RetrievalConfidenceEvaluator (TASK 12).
"""

import pytest
from app.retrieval.confidence import RetrievalConfidenceEvaluator, ConfidenceAssessment


def test_high_confidence_score():
    evaluator = RetrievalConfidenceEvaluator(min_dense_score=0.75)
    mock_candidates = [
        {"document_id": "doc1", "score": 0.85},
        {"document_id": "doc2", "score": 0.70},
        {"document_id": "doc3", "score": 0.65}
    ]
    assessment = evaluator.evaluate(mock_candidates)
    assert assessment.decision == "HIGH_CONFIDENCE"
    assert assessment.confidence_score == 0.85
    assert assessment.top1_score == 0.85
    assert assessment.score_gap == pytest.approx(0.15)
    assert ">= threshold" in assessment.reason


def test_low_confidence_score():
    evaluator = RetrievalConfidenceEvaluator(min_dense_score=0.75)
    mock_candidates = [
        {"document_id": "doc1", "score": 0.62},
        {"document_id": "doc2", "score": 0.58}
    ]
    assessment = evaluator.evaluate(mock_candidates)
    assert assessment.decision == "LOW_CONFIDENCE"
    assert assessment.confidence_score == 0.62
    assert assessment.score_gap == pytest.approx(0.04)
    assert "< threshold" in assessment.reason


def test_empty_candidates_confidence():
    evaluator = RetrievalConfidenceEvaluator(min_dense_score=0.75)
    assessment = evaluator.evaluate([])
    assert assessment.decision == "LOW_CONFIDENCE"
    assert assessment.confidence_score == 0.0
    assert assessment.top1_score == 0.0
    assert assessment.mean_score == 0.0
    assert assessment.score_gap == 0.0
    assert "No dense candidates" in assessment.reason


def test_confidence_structured_dict():
    evaluator = RetrievalConfidenceEvaluator(min_dense_score=0.80)
    mock_candidates = [
        {"document_id": "docA", "score": 0.82},
        {"document_id": "docB", "score": 0.80}
    ]
    assessment = evaluator.evaluate(mock_candidates)
    res_dict = assessment.to_dict()
    assert isinstance(res_dict, dict)
    assert "confidence_score" in res_dict
    assert "decision" in res_dict
    assert "reason" in res_dict
    assert "top1_score" in res_dict
    assert "mean_score" in res_dict
    assert "score_gap" in res_dict
