"""
Multi-strategy Chunking Package (TASK 09).
Provides native, sentence_window, fixed, and semantic chunking strategies.
"""

from app.chunking.models import Chunk, ChunkConfig
from app.chunking.base import BaseChunker
from app.chunking.native import NativeChunker
from app.chunking.sentence_window import SentenceWindowChunker
from app.chunking.fixed import FixedSizeChunker
from app.chunking.semantic import SemanticChunker
from app.chunking.registry import ChunkerRegistry, registry

__all__ = [
    "Chunk",
    "ChunkConfig",
    "BaseChunker",
    "NativeChunker",
    "SentenceWindowChunker",
    "FixedSizeChunker",
    "SemanticChunker",
    "ChunkerRegistry",
    "registry"
]
