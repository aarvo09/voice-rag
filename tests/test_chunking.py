"""
Pytest Unit Test Suite for Multi-Strategy Chunking (TASK 09).
Validates Native, Sentence-Window, Fixed-Size, and Semantic chunking behavior,
registry instantiation, metadata preservation, ID uniqueness, and label inheritance.
"""

import pytest
from app.chunking.models import ChunkConfig
from app.chunking.registry import registry
from app.chunking.native import NativeChunker
from app.chunking.sentence_window import SentenceWindowChunker
from app.chunking.fixed import FixedSizeChunker
from app.chunking.semantic import SemanticChunker

SAMPLE_DOC = {
    "document_id": "doc_101",
    "text": "मैनहट्टन परियोजना एक अनुसंधान परियोजना थी। इसमें पहला परमाणु बम विकसित किया गया था! द्वितीय विश्व युद्ध में इसका बड़ा प्रभाव पड़ा।",
    "language": "hi",
    "query_id": 1185869,
    "passage_index": 0,
    "is_selected": 1,
}

EMPTY_DOC = {
    "document_id": "doc_empty",
    "text": "   ",
    "language": "hi",
    "query_id": 1185869,
    "passage_index": 1,
    "is_selected": 0,
}


def test_registry_lookup():
    native_c = registry.get("native")
    sw_c = registry.get("sentence_window")
    fix_c = registry.get("fixed")
    sem_c = registry.get("semantic")

    assert isinstance(native_c, NativeChunker)
    assert isinstance(sw_c, SentenceWindowChunker)
    assert isinstance(fix_c, FixedSizeChunker)
    assert isinstance(sem_c, SemanticChunker)

    with pytest.raises(KeyError):
        registry.get("invalid_strategy")


def test_native_chunker():
    chunker = NativeChunker()
    chunks = chunker.chunk_document(SAMPLE_DOC)

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.text == SAMPLE_DOC["text"]
    assert chunk.chunk_id == "doc_101_native"
    assert chunk.is_selected == 1
    assert chunk.query_id == 1185869
    assert chunk.start_offset == 0
    assert chunk.end_offset == len(SAMPLE_DOC["text"])


def test_sentence_window_chunker_overlap():
    config = ChunkConfig(window_size=2, stride=1)
    chunker = SentenceWindowChunker(config=config)
    chunks = chunker.chunk_document(SAMPLE_DOC)

    assert len(chunks) == 2
    assert chunks[0].chunk_type == "sentence_window"
    assert chunks[0].chunk_id == "doc_101_sw0"
    assert chunks[1].chunk_id == "doc_101_sw1"
    # Sentence 2 "इसमें पहला परमाणु बम विकसित किया गया था!" should appear in both chunks 0 and 1
    assert "पहला परमाणु बम" in chunks[0].text
    assert "पहला परमाणु बम" in chunks[1].text
    assert chunks[0].is_selected == 1


def test_fixed_size_chunker_size_and_overlap():
    # 5 words per chunk, 2 words overlap
    config = ChunkConfig(chunk_size=5, overlap=2)
    chunker = FixedSizeChunker(config=config)
    chunks = chunker.chunk_document(SAMPLE_DOC)

    assert len(chunks) > 1
    for idx, chunk in enumerate(chunks):
        assert chunk.chunk_id == f"doc_101_fix{idx}"
        assert chunk.chunk_type == "fixed"
        assert chunk.is_selected == 1
        assert chunk.start_offset < chunk.end_offset


def test_semantic_chunker():
    chunker = SemanticChunker()
    chunks = chunker.chunk_document(SAMPLE_DOC)

    assert len(chunks) >= 1
    for chunk in chunks:
        assert chunk.chunk_type == "semantic"
        assert len(chunk.text) > 0
        assert chunk.is_selected == 1


def test_chunk_ids_uniqueness():
    chunkers = [
        registry.get("native"),
        registry.get("sentence_window"),
        registry.get("fixed"),
    ]
    for chunker in chunkers:
        chunks = chunker.chunk_document(SAMPLE_DOC)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids)), f"Duplicate chunk IDs found in {chunker}"


def test_metadata_preservation():
    chunker = NativeChunker()
    chunks = chunker.chunk_document(SAMPLE_DOC)
    c = chunks[0]

    assert c.parent_document_id == "doc_101"
    assert c.language == "hi"
    assert c.query_id == 1185869
    assert c.passage_index == 0
    assert c.is_selected == 1


def test_inherited_selected_label():
    unselected_doc = dict(SAMPLE_DOC, is_selected=0)
    chunker = SentenceWindowChunker()

    selected_chunks = chunker.chunk_document(SAMPLE_DOC)
    unselected_chunks = chunker.chunk_document(unselected_doc)

    assert all(c.is_selected == 1 for c in selected_chunks)
    assert all(c.is_selected == 0 for c in unselected_chunks)


def test_empty_text_rejection():
    chunkers = [
        NativeChunker(),
        SentenceWindowChunker(),
        FixedSizeChunker(),
        SemanticChunker(),
    ]
    for chunker in chunkers:
        chunks = chunker.chunk_document(EMPTY_DOC)
        assert len(chunks) == 0, f"Expected 0 chunks for empty doc in {chunker}"
