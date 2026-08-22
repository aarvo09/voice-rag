"""
Semantic Boundary Chunker Strategy (TASK 09).
Splits document passages at semantic boundary discontinuities using sentence embedding similarity drops.
Calculated strictly offline. Reuses cached MultilingualE5Embedder.
"""

import numpy as np
from typing import List, Dict, Any, Optional
from app.chunking.base import BaseChunker
from app.chunking.models import Chunk, ChunkConfig
from app.chunking.sentence_window import split_sentences_with_offsets
from app.embeddings.model import MultilingualE5Embedder


class SemanticChunker(BaseChunker):
    """
    Groups passage sentences into semantically coherent chunks based on cosine similarity
    discontinuities between consecutive sentence embeddings.
    """

    def __init__(self, config: ChunkConfig = None, embedder: Optional[MultilingualE5Embedder] = None):
        super().__init__(config or ChunkConfig(strategy="semantic", semantic_threshold=0.75))
        self.embedder = embedder

    def _get_embedder(self) -> MultilingualE5Embedder:
        if self.embedder is None:
            self.embedder = MultilingualE5Embedder(model_name="intfloat/multilingual-e5-small")
        return self.embedder

    def chunk_document(self, doc: Dict[str, Any]) -> List[Chunk]:
        text = doc.get("text", "").strip()
        if not text:
            return []

        doc_id = doc.get("document_id", "")
        sentences = split_sentences_with_offsets(text)
        num_sentences = len(sentences)

        if num_sentences <= 1:
            return [
                Chunk(
                    chunk_id=f"{doc_id}_sem0",
                    parent_document_id=doc_id,
                    text=text,
                    language=doc.get("language", "hi"),
                    query_id=int(doc.get("query_id", -1)),
                    passage_index=int(doc.get("passage_index", -1)),
                    is_selected=int(doc.get("is_selected", 0)),
                    chunk_type="semantic",
                    chunk_index=0,
                    start_offset=0,
                    end_offset=len(text),
                    metadata={"sentence_count": num_sentences, "semantic_boundaries": 0}
                )
            ]

        # 1. Encode sentences using sentence-transformers
        embedder = self._get_embedder()
        sentence_texts = [s[0] for s in sentences]
        sentence_vecs = embedder.embed_documents(sentence_texts)  # (N, 384) normalized vectors

        # 2. Calculate cosine similarity between consecutive sentence embeddings
        # Since vectors are L2-normalized, cosine similarity = dot product
        sims = np.sum(sentence_vecs[:-1] * sentence_vecs[1:], axis=1)  # (N-1,)

        # 3. Identify boundary points where similarity drops below threshold
        # Default threshold: 0.75
        threshold = self.config.semantic_threshold
        split_indices = np.where(sims < threshold)[0] + 1  # 1-indexed split positions

        # 4. Group sentences into semantic chunks
        boundaries = [0] + list(split_indices) + [num_sentences]
        chunks: List[Chunk] = []

        chunk_idx = 0
        for b_start, b_end in zip(boundaries[:-1], boundaries[1:]):
            group_sents = sentences[b_start:b_end]
            if not group_sents:
                continue

            start_off = group_sents[0][1]
            end_off = group_sents[-1][2]
            chunk_text = text[start_off:end_off].strip()

            if not chunk_text:
                continue

            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}_sem{chunk_idx}",
                    parent_document_id=doc_id,
                    text=chunk_text,
                    language=doc.get("language", "hi"),
                    query_id=int(doc.get("query_id", -1)),
                    passage_index=int(doc.get("passage_index", -1)),
                    is_selected=int(doc.get("is_selected", 0)),
                    chunk_type="semantic",
                    chunk_index=chunk_idx,
                    start_offset=start_off,
                    end_offset=end_off,
                    metadata={"sentence_count": len(group_sents), "threshold": threshold}
                )
            )
            chunk_idx += 1

        return chunks
