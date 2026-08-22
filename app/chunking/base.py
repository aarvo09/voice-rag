"""
Base Chunker Abstract Interface (TASK 09).
Defines abstract methods for document passage chunking.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
from app.chunking.models import Chunk, ChunkConfig


class BaseChunker(ABC):
    """Abstract Base Class for document passage chunkers."""

    def __init__(self, config: ChunkConfig = None):
        self.config = config or ChunkConfig()

    @abstractmethod
    def chunk_document(self, doc: Dict[str, Any]) -> List[Chunk]:
        """Chunks a single document dictionary into a list of Chunk objects."""
        pass

    def chunk_batch(self, docs: List[Dict[str, Any]]) -> List[Chunk]:
        """Chunks a batch of document dictionaries."""
        all_chunks: List[Chunk] = []
        for doc in docs:
            chunks = self.chunk_document(doc)
            all_chunks.extend(chunks)
        return all_chunks
