"""
Unit tests for ProductionRetriever (TASK 12).
Validates dense retrieval, confidence evaluation, optional BM25 fallback, metadata preservation,
top-K constraints, and reranker exclusion.
"""

import sys
import pytest
from unittest.mock import MagicMock
from app.pipeline.policies import RetrievalPolicy
from app.retrieval.confidence import RetrievalConfidenceEvaluator
from app.retrieval.retriever import ProductionRetriever, VectorRetriever


class MockMetadataLoader:
    def __init__(self):
        self.docs = {
            1: {"document_id": "doc_dense_1", "text": "Dense Text 1", "language": "hi", "query_id": 100, "passage_index": 0, "is_selected": 1},
            2: {"document_id": "doc_dense_2", "text": "Dense Text 2", "language": "hi", "query_id": 100, "passage_index": 1, "is_selected": 0},
            3: {"document_id": "doc_bm25_1", "text": "Lexical BM25 Text 1", "language": "hi", "query_id": 100, "passage_index": 2, "is_selected": 1},
            4: {"document_id": "doc_bm25_2", "text": "Lexical BM25 Text 2", "language": "hi", "query_id": 100, "passage_index": 3, "is_selected": 0},
        }

    def get_document(self, row_idx: int):
        return self.docs.get(row_idx, {
            "document_id": f"doc_{row_idx}", "text": "Default text", "language": "hi",
            "query_id": 100, "passage_index": row_idx, "is_selected": 0
        })


@pytest.fixture
def mock_retriever_setup():
    metadata_loader = MockMetadataLoader()
    vector_retriever = MagicMock(spec=VectorRetriever)
    bm25_retriever = MagicMock()

    # Default dense search return
    vector_retriever.retrieve.return_value = [
        {"rank": 1, "document_id": "doc_dense_1", "score": 0.82, "text": "Dense Text 1", "language": "hi", "query_id": 100, "passage_index": 0, "is_selected": 1},
        {"rank": 2, "document_id": "doc_dense_2", "score": 0.70, "text": "Dense Text 2", "language": "hi", "query_id": 100, "passage_index": 1, "is_selected": 0},
    ]

    # Default BM25 search return
    bm25_retriever.search.return_value = (
        __import__("numpy").array([[12.5, 9.1]], dtype=__import__("numpy").float32),
        __import__("numpy").array([[3, 4]], dtype=__import__("numpy").int64)
    )

    return vector_retriever, bm25_retriever, metadata_loader


def test_high_confidence_returns_dense_only(mock_retriever_setup):
    vector_retriever, bm25_retriever, metadata_loader = mock_retriever_setup
    policy = RetrievalPolicy(min_dense_score=0.75, fallback_enabled=True, final_top_k=2)
    prod_retriever = ProductionRetriever(
        vector_retriever=vector_retriever,
        bm25_retriever=bm25_retriever,
        metadata_loader=metadata_loader,
        policy=policy
    )

    out = prod_retriever.retrieve("मैनहट्टन परियोजना")
    assert out["fallback_used"] is False
    assert len(out["results"]) == 2
    assert out["results"][0]["document_id"] == "doc_dense_1"
    assert out["confidence"]["decision"] == "HIGH_CONFIDENCE"
    bm25_retriever.search.assert_not_called()


def test_low_confidence_activates_fallback(mock_retriever_setup):
    vector_retriever, bm25_retriever, metadata_loader = mock_retriever_setup
    # Dense results return low score
    vector_retriever.retrieve.return_value = [
        {"rank": 1, "document_id": "doc_dense_1", "score": 0.60, "text": "Dense Text 1", "language": "hi", "query_id": 100, "passage_index": 0, "is_selected": 0},
    ]
    policy = RetrievalPolicy(min_dense_score=0.75, fallback_enabled=True, final_top_k=2)
    prod_retriever = ProductionRetriever(
        vector_retriever=vector_retriever,
        bm25_retriever=bm25_retriever,
        metadata_loader=metadata_loader,
        policy=policy
    )

    out = prod_retriever.retrieve("मैनहट्टन परियोजना")
    assert out["fallback_used"] is True
    assert out["confidence"]["decision"] == "LOW_CONFIDENCE"
    assert len(out["results"]) == 2
    assert out["results"][0]["document_id"] == "doc_bm25_1"
    bm25_retriever.search.assert_called_once()


def test_fallback_can_be_disabled(mock_retriever_setup):
    vector_retriever, bm25_retriever, metadata_loader = mock_retriever_setup
    vector_retriever.retrieve.return_value = [
        {"rank": 1, "document_id": "doc_dense_1", "score": 0.50, "text": "Dense Text 1", "language": "hi", "query_id": 100, "passage_index": 0, "is_selected": 0},
    ]
    policy = RetrievalPolicy(min_dense_score=0.75, fallback_enabled=False, final_top_k=1)
    prod_retriever = ProductionRetriever(
        vector_retriever=vector_retriever,
        bm25_retriever=bm25_retriever,
        metadata_loader=metadata_loader,
        policy=policy
    )

    out = prod_retriever.retrieve("मैनहट्टन परियोजना")
    assert out["fallback_used"] is False
    assert len(out["results"]) == 1
    assert out["results"][0]["document_id"] == "doc_dense_1"
    bm25_retriever.search.assert_not_called()


def test_top_k_respected(mock_retriever_setup):
    vector_retriever, bm25_retriever, metadata_loader = mock_retriever_setup
    policy = RetrievalPolicy(min_dense_score=0.75, fallback_enabled=True, final_top_k=1)
    prod_retriever = ProductionRetriever(
        vector_retriever=vector_retriever,
        bm25_retriever=bm25_retriever,
        metadata_loader=metadata_loader,
        policy=policy
    )

    out = prod_retriever.retrieve("मैनहट्टन परियोजना")
    assert len(out["results"]) == 1


def test_reranker_not_loaded_in_production(mock_retriever_setup):
    sys.modules.pop("app.retrieval.reranker", None)
    vector_retriever, bm25_retriever, metadata_loader = mock_retriever_setup
    prod_retriever = ProductionRetriever(
        vector_retriever=vector_retriever,
        bm25_retriever=bm25_retriever,
        metadata_loader=metadata_loader
    )
    _ = prod_retriever.retrieve("टेस्ट क्वेरी")
    # Verify app.retrieval.reranker module is NOT imported into sys.modules by initializing ProductionRetriever
    assert "app.retrieval.reranker" not in sys.modules


def test_metadata_preservation(mock_retriever_setup):
    vector_retriever, bm25_retriever, metadata_loader = mock_retriever_setup
    prod_retriever = ProductionRetriever(
        vector_retriever=vector_retriever,
        bm25_retriever=bm25_retriever,
        metadata_loader=metadata_loader
    )

    out = prod_retriever.retrieve("टेस्ट क्वेरी")
    doc = out["results"][0]
    assert "document_id" in doc
    assert "text" in doc
    assert "language" in doc
    assert "is_selected" in doc
    assert "rank" in doc


def test_deterministic_behavior(mock_retriever_setup):
    vector_retriever, bm25_retriever, metadata_loader = mock_retriever_setup
    prod_retriever = ProductionRetriever(
        vector_retriever=vector_retriever,
        bm25_retriever=bm25_retriever,
        metadata_loader=metadata_loader
    )

    out1 = prod_retriever.retrieve("टेस्ट क्वेरी")
    out2 = prod_retriever.retrieve("टेस्ट क्वेरी")
    assert out1["results"] == out2["results"]
    assert out1["fallback_used"] == out2["fallback_used"]
    assert out1["confidence"] == out2["confidence"]
