"""
Fixed-Size Token Chunker Strategy (TASK 09).
Splits document passages into fixed-size token chunks with configurable token overlap.
"""

import re
from typing import List, Dict, Any, Tuple
from app.chunking.base import BaseChunker
from app.chunking.models import Chunk, ChunkConfig

TOKEN_PATTERN = re.compile(r'\S+', re.UNICODE)


def extract_tokens_with_offsets(text: str) -> List[Tuple[str, int, int]]:
    """Extracts (token_string, start_offset, end_offset) for all whitespace-delimited tokens."""
    if not text:
        return []
    return [(m.group(), m.start(), m.end()) for m in TOKEN_PATTERN.finditer(text)]


class FixedSizeChunker(BaseChunker):
    """Splits passages into fixed-token chunks (default: 128 tokens) with overlap (default: 32 tokens)."""

    def __init__(self, config: ChunkConfig = None):
        super().__init__(config or ChunkConfig(strategy="fixed", chunk_size=128, overlap=32))

    def chunk_document(self, doc: Dict[str, Any]) -> List[Chunk]:
        text = doc.get("text", "").strip()
        if not text:
            return []

        doc_id = doc.get("document_id", "")
        tokens = extract_tokens_with_offsets(text)
        num_tokens = len(tokens)

        if num_tokens == 0:
            return []

        c_size = self.config.chunk_size
        overlap = self.config.overlap
        step = max(1, c_size - overlap)

        chunks: List[Chunk] = []

        if num_tokens <= c_size:
            start_off = tokens[0][1]
            end_off = tokens[-1][2]
            chunk_text = text[start_off:end_off].strip()
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}_fix0",
                    parent_document_id=doc_id,
                    text=chunk_text,
                    language=doc.get("language", "hi"),
                    query_id=int(doc.get("query_id", -1)),
                    passage_index=int(doc.get("passage_index", -1)),
                    is_selected=int(doc.get("is_selected", 0)),
                    chunk_type="fixed",
                    chunk_index=0,
                    start_offset=start_off,
                    end_offset=end_off,
                    metadata={"token_count": num_tokens, "chunk_size": c_size, "overlap": overlap}
                )
            )
            return chunks

        chunk_idx = 0
        for i in range(0, num_tokens, step):
            window_tokens = tokens[i : i + c_size]
            if not window_tokens:
                break

            start_off = window_tokens[0][1]
            end_off = window_tokens[-1][2]
            chunk_text = text[start_off:end_off].strip()

            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}_fix{chunk_idx}",
                    parent_document_id=doc_id,
                    text=chunk_text,
                    language=doc.get("language", "hi"),
                    query_id=int(doc.get("query_id", -1)),
                    passage_index=int(doc.get("passage_index", -1)),
                    is_selected=int(doc.get("is_selected", 0)),
                    chunk_type="fixed",
                    chunk_index=chunk_idx,
                    start_offset=start_off,
                    end_offset=end_off,
                    metadata={"token_count": len(window_tokens), "chunk_size": c_size, "overlap": overlap}
                )
            )
            chunk_idx += 1

            # Break if window reached the end of tokens
            if i + c_size >= num_tokens:
                break

        return chunks
