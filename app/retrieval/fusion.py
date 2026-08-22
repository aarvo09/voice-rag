"""
Reciprocal Rank Fusion (RRF) Implementation (TASK 08).
Fuses ranked candidate lists from Dense FAISS and Lexical BM25 retrievers.
Formula: RRF(d) = dense_weight / (k + dense_rank) + bm25_weight / (k + bm25_rank)
"""

import logging
from typing import List, Dict, Any, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class ReciprocalRankFusion:
    """
    Combines ranked result lists from multiple retrieval systems using Reciprocal Rank Fusion (RRF).
    Avoids score scale mismatch by leveraging ordinal ranks.
    """

    def __init__(self, k: int = 60, dense_weight: float = 1.0, bm25_weight: float = 1.0):
        self.k = k
        self.dense_weight = dense_weight
        self.bm25_weight = bm25_weight

    def fuse(
        self,
        dense_results: List[Dict[str, Any]],
        bm25_results: List[Dict[str, Any]],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Fuses dense and BM25 candidate lists into a single ranked list.
        """
        doc_map: Dict[str, Dict[str, Any]] = {}

        # 1. Process Dense Candidate Results
        for rank_idx, doc in enumerate(dense_results, start=1):
            doc_id = doc["document_id"]
            if doc_id not in doc_map:
                doc_map[doc_id] = {
                    "doc_meta": doc,
                    "dense_rank": rank_idx,
                    "dense_score": doc.get("score"),
                    "bm25_rank": None,
                    "bm25_score": None,
                }
            else:
                doc_map[doc_id]["dense_rank"] = rank_idx
                doc_map[doc_id]["dense_score"] = doc.get("score")

        # 2. Process BM25 Candidate Results
        for rank_idx, doc in enumerate(bm25_results, start=1):
            doc_id = doc["document_id"]
            if doc_id not in doc_map:
                doc_map[doc_id] = {
                    "doc_meta": doc,
                    "dense_rank": None,
                    "dense_score": None,
                    "bm25_rank": rank_idx,
                    "bm25_score": doc.get("score"),
                }
            else:
                doc_map[doc_id]["bm25_rank"] = rank_idx
                doc_map[doc_id]["bm25_score"] = doc.get("score")

        # 3. Compute Weighted RRF Scores
        fused_items = []
        for doc_id, entry in doc_map.items():
            rrf_score = 0.0
            if entry["dense_rank"] is not None:
                rrf_score += self.dense_weight / (self.k + entry["dense_rank"])
            if entry["bm25_rank"] is not None:
                rrf_score += self.bm25_weight / (self.k + entry["bm25_rank"])

            meta = entry["doc_meta"]
            fused_item = {
                "document_id": doc_id,
                "rrf_score": float(rrf_score),
                "text": meta["text"],
                "language": meta["language"],
                "query_id": int(meta["query_id"]),
                "passage_index": int(meta["passage_index"]),
                "is_selected": int(meta["is_selected"]),
                "source": meta.get("source", "ai4bharat/MSMARCO-XI"),
                "dense_rank": entry["dense_rank"],
                "bm25_rank": entry["bm25_rank"],
                "dense_score": entry["dense_score"],
                "bm25_score": entry["bm25_score"],
            }
            fused_items.append(fused_item)

        # 4. Sort by RRF score descending, breaking ties deterministically by document_id
        fused_items.sort(key=lambda x: (-x["rrf_score"], x["document_id"]))

        # 5. Assign final 1-indexed ranks & slice top_k
        final_results = []
        for final_rank, item in enumerate(fused_items[:top_k], start=1):
            item["rank"] = final_rank
            final_results.append(item)

        return final_results
