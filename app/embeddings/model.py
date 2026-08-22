"""
Multilingual E5 Embedding Model Interface (TASK 05).
"""

import logging
import time
from typing import List, Optional
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class MultilingualE5Embedder:
    """
    Interface for intfloat/multilingual-e5-small (or compatible E5 embedding models).
    Enforces required E5 prefixes:
      - 'passage: <text>' for documents
      - 'query: <text>' for queries
    """

    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-small",
        device: Optional[str] = None,
        normalize_embeddings: bool = True
    ):
        self.model_name = model_name
        self.normalize_embeddings = normalize_embeddings

        # Lazy torch import & device detection
        import torch
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(f"Loading SentenceTransformer model '{self.model_name}' on device '{self.device}'...")
        t0 = time.time()
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(self.model_name, device=self.device)
        self.load_time_s = round(time.time() - t0, 3)
        logger.info(f"Model loaded successfully in {self.load_time_s}s.")

        # Discover dynamic embedding dimension from sentence-transformers model
        if hasattr(self.model, "get_embedding_dimension"):
            self.embedding_dim = self.model.get_embedding_dimension()
        else:
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Discovered dynamic embedding dimension: {self.embedding_dim}")

    def embed_documents(self, texts: List[str], batch_size: int = 32, show_progress: bool = False) -> np.ndarray:
        """
        Embeds a list of document/passage texts in bounded batches.
        Prepends required E5 prefix: 'passage: '
        """
        if not texts:
            return np.empty((0, self.embedding_dim), dtype=np.float32)

        formatted_passages = [f"passage: {t}" for t in texts]
        embeddings = self.model.encode(
            formatted_passages,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True
        )
        return embeddings.astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embeds a single query string.
        Prepends required E5 prefix: 'query: '
        Returns 1D numpy array of shape (embedding_dim,).
        """
        formatted_query = f"query: {query}"
        embedding = self.model.encode(
            formatted_query,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True
        )
        return embedding.astype(np.float32).flatten()

    def get_embedding_dimension(self) -> int:
        return self.embedding_dim
