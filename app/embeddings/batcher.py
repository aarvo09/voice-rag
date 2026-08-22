"""
Batching utilities for sentence embedding generation (TASK 05).
"""

import logging
from typing import List, Generator
import numpy as np
from app.embeddings.model import MultilingualE5Embedder

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def chunk_list(items: List[str], chunk_size: int) -> Generator[List[str], None, None]:
    """Yields successive chunks of size chunk_size from items list."""
    for i in range(0, len(items), chunk_size):
        yield items[i : i + chunk_size]


def batch_encode_documents(
    embedder: MultilingualE5Embedder,
    texts: List[str],
    batch_size: int = 32,
    show_progress: bool = False
) -> np.ndarray:
    """
    Encodes a list of passage texts in bounded batches to prevent memory spikes.
    Appends output batches into a concatenated NumPy array.
    """
    total_texts = len(texts)
    if total_texts == 0:
        return np.empty((0, embedder.get_embedding_dimension()), dtype=np.float32)

    logger.info(f"Encoding {total_texts} document passages with batch_size={batch_size}...")

    embeddings_list: List[np.ndarray] = []
    processed = 0

    for batch in chunk_list(texts, batch_size):
        batch_embeddings = embedder.embed_documents(batch, batch_size=batch_size, show_progress=show_progress)
        embeddings_list.append(batch_embeddings)
        processed += len(batch)

    full_embeddings = np.vstack(embeddings_list)
    logger.info(f"Completed encoding {processed}/{total_texts} passages. Final shape: {full_embeddings.shape}")
    return full_embeddings
