"""
Corpus Metadata Loader (TASK 06).
Maps FAISS 0-based row indices to document metadata records from dev_corpus.parquet.
"""

import os
import logging
from typing import Dict, List, Any
import pyarrow.parquet as pq

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class CorpusMetadataLoader:
    """
    Loads and caches document metadata from dev_corpus.parquet.
    FAISS index row i corresponds exactly to corpus row i.
    """

    def __init__(self, corpus_path: str):
        self.corpus_path = corpus_path
        if not os.path.exists(corpus_path):
            raise FileNotFoundError(f"Corpus Parquet file not found at: {corpus_path}")

        logger.info(f"Loading metadata from corpus: {corpus_path}")
        table = pq.read_table(corpus_path)
        self.documents: List[Dict[str, Any]] = table.to_pylist()
        logger.info(f"Loaded metadata for {len(self.documents)} corpus documents.")

    def get_document(self, row_idx: int) -> Dict[str, Any]:
        """Returns document metadata dictionary for FAISS row_idx."""
        if 0 <= row_idx < len(self.documents):
            return self.documents[row_idx]
        raise IndexError(f"Row index {row_idx} out of range for corpus of size {len(self.documents)}.")

    def __len__(self) -> int:
        return len(self.documents)
