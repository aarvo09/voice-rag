"""
Embeddings package for multilingual E5 text vectorization.
"""

from app.embeddings.model import MultilingualE5Embedder
from app.embeddings.batcher import batch_encode_documents

__all__ = ["MultilingualE5Embedder", "batch_encode_documents"]
