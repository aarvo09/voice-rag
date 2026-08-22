"""
Unit tests for ContextBuilder (TASK 13).
Verifies context formatting, provenance preservation, rank ordering, deduplication, and document limits.
"""

import pytest
from app.generation.context import ContextBuilder


@pytest.fixture
def mock_documents():
    return [
        {"document_id": "doc1", "rank": 1, "score": 0.85, "text": "First passage text."},
        {"document_id": "doc2", "rank": 2, "score": 0.75, "text": "Second passage text."},
        {"document_id": "doc1", "rank": 3, "score": 0.70, "text": "Duplicate doc1 text."},
        {"document_id": "doc3", "rank": 4, "score": 0.65, "text": "Third passage text."},
        {"document_id": "doc4", "rank": 5, "score": 0.60, "text": "Fourth passage text."},
        {"document_id": "doc5", "rank": 6, "score": 0.55, "text": "Fifth passage text."},
        {"document_id": "doc6", "rank": 7, "score": 0.50, "text": "Sixth passage text."},
    ]


def test_context_builder_max_documents_limit(mock_documents):
    builder = ContextBuilder(max_context_documents=3)
    formatted_str, selected_docs = builder.build_context(mock_documents)

    assert len(selected_docs) == 3
    assert [d["document_id"] for d in selected_docs] == ["doc1", "doc2", "doc3"]
    assert "[DOCUMENT 1]" in formatted_str
    assert "[DOCUMENT 3]" in formatted_str
    assert "[DOCUMENT 4]" not in formatted_str


def test_context_builder_deduplication(mock_documents):
    builder = ContextBuilder(max_context_documents=5)
    _, selected_docs = builder.build_context(mock_documents)

    doc_ids = [d["document_id"] for d in selected_docs]
    assert len(doc_ids) == len(set(doc_ids))
    assert doc_ids == ["doc1", "doc2", "doc3", "doc4", "doc5"]


def test_context_builder_ordering_and_formatting(mock_documents):
    builder = ContextBuilder(max_context_documents=2)
    formatted_str, selected_docs = builder.build_context(mock_documents)

    assert selected_docs[0]["document_id"] == "doc1"
    assert selected_docs[1]["document_id"] == "doc2"

    assert "document_id: doc1" in formatted_str
    assert "score: 0.8500" in formatted_str
    assert "text: First passage text." in formatted_str


def test_context_builder_empty_documents():
    builder = ContextBuilder(max_context_documents=5)
    formatted_str, selected_docs = builder.build_context([])

    assert formatted_str == "NO CONTEXT AVAILABLE"
    assert selected_docs == []
