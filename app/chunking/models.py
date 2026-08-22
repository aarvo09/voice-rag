"""
Chunk Data Models (TASK 09).
Defines Chunk structure and ChunkConfig parameters.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass
class Chunk:
    """Represents a discrete text chunk derived from a parent document passage."""
    chunk_id: str
    parent_document_id: str
    text: str
    language: str
    query_id: int
    passage_index: int
    is_selected: int  # Inherited from parent passage
    chunk_type: str  # "native", "sentence_window", "fixed", "semantic"
    chunk_index: int  # 0-indexed position within parent document
    start_offset: int
    end_offset: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Converts chunk to dictionary matching Parquet schema."""
        return {
            "chunk_id": self.chunk_id,
            "parent_document_id": self.parent_document_id,
            "text": self.text,
            "language": self.language,
            "query_id": self.query_id,
            "passage_index": self.passage_index,
            "is_selected": self.is_selected,
            "chunk_type": self.chunk_type,
            "chunk_index": self.chunk_index,
            "start_offset": self.start_offset,
            "end_offset": self.end_offset,
        }


@dataclass
class ChunkConfig:
    """Configuration options for multi-strategy chunking."""
    strategy: str = "native"
    window_size: int = 3
    stride: int = 1
    chunk_size: int = 128
    overlap: int = 32
    semantic_threshold: float = 0.75
    minimum_chunk_length: int = 10
    maximum_chunk_length: int = 1000
