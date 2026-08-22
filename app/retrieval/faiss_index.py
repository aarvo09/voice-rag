"""
FAISS Vector Index Implementation (TASK 06).
Uses IndexFlatIP for inner-product (cosine) similarity over normalized embeddings.
"""

import os
import logging
from typing import Tuple, Optional
import numpy as np
import faiss

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class FaissVectorIndex:
    """
    CPU FAISS Vector Index wrapping IndexFlatIP.
    Since embeddings from MultilingualE5Embedder are L2-normalized,
    Inner Product (IP) search is mathematically equivalent to Cosine Similarity.
    """

    def __init__(self, dimension: Optional[int] = None):
        self.dim = dimension
        self.index: Optional[faiss.IndexFlatIP] = None
        if dimension is not None:
            self.index = faiss.IndexFlatIP(dimension)

    def _validate_embeddings(self, embeddings: np.ndarray) -> None:
        """Validates input embeddings matrix properties."""
        if not isinstance(embeddings, np.ndarray):
            raise TypeError("Embeddings must be a numpy.ndarray.")
        if embeddings.ndim != 2:
            raise ValueError(f"Embeddings must be 2-dimensional (got shape {embeddings.shape}).")
        if embeddings.dtype != np.float32:
            raise ValueError(f"Embeddings dtype must be float32 (got {embeddings.dtype}).")
        if not np.isfinite(embeddings).all():
            raise ValueError("Embeddings matrix contains non-finite values (NaN or Inf).")

    def build(self, embeddings: np.ndarray) -> None:
        """Builds the IndexFlatIP from normalized float32 embeddings array."""
        self._validate_embeddings(embeddings)
        n_vectors, d = embeddings.shape

        if self.dim is not None and self.dim != d:
            raise ValueError(f"Embedding dimension {d} does not match index dimension {self.dim}.")

        self.dim = d
        # Re-initialize IndexFlatIP with correct dimension if needed
        self.index = faiss.IndexFlatIP(self.dim)

        logger.info(f"Adding {n_vectors} vectors of dimension {d} to FAISS IndexFlatIP...")
        self.index.add(embeddings)
        logger.info(f"FAISS IndexFlatIP built successfully. Total vectors: {self.index.ntotal}")

    def save(self, file_path: str) -> None:
        """Saves FAISS index to disk."""
        if self.index is None or self.index.ntotal == 0:
            raise ValueError("Cannot save empty or uninitialized FAISS index.")
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        faiss.write_index(self.index, file_path)
        logger.info(f"Saved FAISS index to {file_path} ({os.path.getsize(file_path):,} bytes).")

    def load(self, file_path: str) -> None:
        """Loads FAISS index from disk."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"FAISS index file not found at: {file_path}")
        self.index = faiss.read_index(file_path)
        self.dim = self.index.d
        logger.info(f"Loaded FAISS index from {file_path}. Total vectors: {self.index.ntotal}, Dim: {self.dim}")

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Searches FAISS index for top_k nearest neighbors.
        Accepts query_vector as 1D or 2D array of shape (1, dim) or (dim,).
        Returns tuple of (scores, indices) arrays of shape (1, top_k).
        """
        if self.index is None or self.index.ntotal == 0:
            raise ValueError("FAISS index is not populated or loaded.")

        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)

        if query_vector.ndim != 2 or query_vector.shape[0] != 1:
            raise ValueError(f"Query vector must be 2D array with shape (1, dim), got {query_vector.shape}.")
        if query_vector.dtype != np.float32:
            query_vector = query_vector.astype(np.float32)
        if query_vector.shape[1] != self.dim:
            raise ValueError(f"Query vector dimension {query_vector.shape[1]} does not match index dimension {self.dim}.")
        if not np.isfinite(query_vector).all():
            raise ValueError("Query vector contains non-finite values (NaN or Inf).")

        top_k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query_vector, top_k)
        return scores, indices

    def size(self) -> int:
        """Returns total vector count in index."""
        return self.index.ntotal if self.index is not None else 0

    def dimension(self) -> int:
        """Returns vector dimension of index."""
        return self.dim if self.dim is not None else 0
