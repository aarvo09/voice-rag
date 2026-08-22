"""
Native Passage Chunker Strategy (TASK 09).
Preserves 1-to-1 passage mapping as the baseline chunking strategy.
"""

from typing import List, Dict, Any
from app.chunking.base import BaseChunker
from app.chunking.models import Chunk, ChunkConfig


class NativeChunker(BaseChunker):
    """Pass-through chunker mapping 1 source passage directly to 1 Chunk object."""

    def __init__(self, config: ChunkConfig = None):
        super().__init__(config or ChunkConfig(strategy="native"))

    def chunk_document(self, doc: Dict[str, Any]) -> List[Chunk]:
        text = doc.get("text", "").strip()
        if not text:
            return []

        doc_id = doc.get("document_id", "")
        return [
            Chunk(
                chunk_id=f"{doc_id}_native",
                parent_document_id=doc_id,
                text=text,
                language=doc.get("language", "hi"),
                query_id=int(doc.get("query_id", -1)),
                passage_index=int(doc.get("passage_index", -1)),
                is_selected=int(doc.get("is_selected", 0)),
                chunk_type="native",
                chunk_index=0,
                start_offset=0,
                end_offset=len(text),
                metadata={"passage_length": len(text)}
            )
        ]
