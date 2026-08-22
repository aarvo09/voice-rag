"""
Chunker Registry Module (TASK 09).
Factory registry for dynamically retrieving chunker strategy instances by name.
"""

from typing import Dict, Type, Optional, Any
from app.chunking.base import BaseChunker
from app.chunking.models import ChunkConfig
from app.chunking.native import NativeChunker
from app.chunking.sentence_window import SentenceWindowChunker
from app.chunking.fixed import FixedSizeChunker
from app.chunking.semantic import SemanticChunker


class ChunkerRegistry:
    """Registry managing available document chunking strategies."""

    def __init__(self):
        self._strategies: Dict[str, Type[BaseChunker]] = {
            "native": NativeChunker,
            "sentence_window": SentenceWindowChunker,
            "fixed": FixedSizeChunker,
            "semantic": SemanticChunker,
        }

    def register(self, name: str, chunker_cls: Type[BaseChunker]) -> None:
        """Registers a new custom chunking strategy."""
        self._strategies[name.lower()] = chunker_cls

    def get(self, name: str, config: Optional[ChunkConfig] = None, **kwargs: Any) -> BaseChunker:
        """Retrieves and instantiates a chunker strategy by name."""
        name_lower = name.lower()
        if name_lower not in self._strategies:
            raise KeyError(f"Unknown chunking strategy '{name}'. Available: {list(self._strategies.keys())}")

        chunker_cls = self._strategies[name_lower]
        if config is None:
            config = ChunkConfig(strategy=name_lower)
        else:
            config.strategy = name_lower

        return chunker_cls(config=config, **kwargs)


# Global default registry instance
registry = ChunkerRegistry()
