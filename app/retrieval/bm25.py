"""
BM25 Lexical Retriever Implementation (TASK 07).
Uses rank_bm25.BM25Okapi with Unicode-aware Indic/Hindi tokenization.
"""

import os
import re
import pickle
import string
import logging
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from rank_bm25 import BM25Okapi

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

INDIC_PUNCTUATION = string.punctuation + "।॥''\"\"“”‘’"
PUNCT_PATTERN = re.compile("[" + re.escape(INDIC_PUNCTUATION) + "]")


def tokenize_hindi(text: str) -> List[str]:
    """
    Unicode-aware tokenizer for Hindi / Indic text.
      1. Converts to string and lowercases.
      2. Replaces ASCII & Devanagari punctuation with spaces.
      3. Splits on whitespace into intact word tokens.
    """
    if not text:
        return []
    clean_text = PUNCT_PATTERN.sub(" ", text.lower().strip())
    return [token for token in clean_text.split() if token]


class BM25Retriever:
    """
    In-process BM25 lexical retriever wrapping BM25Okapi.
    Preserves exact row-to-document metadata alignment.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.bm25: Optional[BM25Okapi] = None
        self.corpus_size: int = 0

    def tokenize(self, text: str) -> List[str]:
        return tokenize_hindi(text)

    def build(self, documents: List[str]) -> None:
        """
        Tokenizes document corpus and initializes BM25Okapi index.
        """
        self.corpus_size = len(documents)
        logger.info(f"Tokenizing {self.corpus_size} corpus documents for BM25...")
        tokenized_corpus = [self.tokenize(doc) for doc in documents]

        logger.info(f"Building BM25Okapi index (k1={self.k1}, b={self.b})...")
        self.bm25 = BM25Okapi(tokenized_corpus, k1=self.k1, b=self.b)
        logger.info(f"BM25 index built successfully for {self.corpus_size} documents.")

    def search(self, query: str, top_k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Tokenizes query, computes BM25 scores, and returns top-K (scores, indices).
        """
        if self.bm25 is None or self.corpus_size == 0:
            raise ValueError("BM25 index is not populated or loaded.")

        tokenized_query = self.tokenize(query)
        if not tokenized_query:
            logger.warning(f"Query string '{query}' produced 0 tokens.")
            return np.zeros((1, 0), dtype=np.float32), np.zeros((1, 0), dtype=np.int64)

        scores = self.bm25.get_scores(tokenized_query).astype(np.float32)
        top_k = min(top_k, self.corpus_size)

        # Get indices sorted by score descending
        sorted_indices = np.argsort(scores)[::-1][:top_k]
        top_scores = scores[sorted_indices]

        return top_scores.reshape(1, -1), sorted_indices.reshape(1, -1)

    def save(self, file_path: str) -> None:
        """Saves BM25 index object to disk via pickle."""
        if self.bm25 is None:
            raise ValueError("Cannot save uninitialized BM25 index.")
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        data = {
            "bm25": self.bm25,
            "k1": self.k1,
            "b": self.b,
            "corpus_size": self.corpus_size
        }
        with open(file_path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"Saved BM25 index to {file_path} ({os.path.getsize(file_path):,} bytes).")

    def load(self, file_path: str) -> None:
        """Loads BM25 index object from disk via pickle."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"BM25 index file not found at: {file_path}")
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        self.bm25 = data["bm25"]
        self.k1 = data["k1"]
        self.b = data["b"]
        self.corpus_size = data["corpus_size"]
        logger.info(f"Loaded BM25 index from {file_path}. Corpus size: {self.corpus_size}")
