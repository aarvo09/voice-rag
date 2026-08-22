"""
Vector and Hybrid Retriever Module (TASK 08).
Connects E5 embedder, FAISS vector index, BM25 retriever, and RRF fusion layer.
"""

import time
import logging
from typing import List, Dict, Any, Optional
from app.embeddings.model import MultilingualE5Embedder
from app.retrieval.faiss_index import FaissVectorIndex
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.metadata import CorpusMetadataLoader
from app.retrieval.fusion import ReciprocalRankFusion
from app.pipeline.policies import RetrievalPolicy, DEFAULT_RETRIEVAL_POLICY
from app.retrieval.confidence import RetrievalConfidenceEvaluator, ConfidenceAssessment

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class VectorRetriever:
    """
    High-level Vector Retriever combining:
      1. MultilingualE5Embedder (vectorizes query string)
      2. FaissVectorIndex (IndexFlatIP cosine search)
      3. CorpusMetadataLoader (maps row indices to document dicts)
    """

    def __init__(
        self,
        embedder: MultilingualE5Embedder,
        faiss_index: FaissVectorIndex,
        metadata_loader: CorpusMetadataLoader
    ):
        self.embedder = embedder
        self.faiss_index = faiss_index
        self.metadata_loader = metadata_loader

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Executes dense query retrieval:
          query -> E5 query embedding -> FAISS search -> metadata lookup -> structured results.
        """
        query_vec = self.embedder.embed_query(query)
        scores, indices = self.faiss_index.search(query_vec, top_k=top_k)

        results: List[Dict[str, Any]] = []
        if indices.size > 0:
            for rank_idx, (score, row_idx) in enumerate(zip(scores[0], indices[0]), start=1):
                if row_idx < 0:
                    continue
                doc_meta = self.metadata_loader.get_document(int(row_idx))
                result_item = {
                    "rank": rank_idx,
                    "document_id": doc_meta["document_id"],
                    "score": float(score),
                    "text": doc_meta["text"],
                    "language": doc_meta["language"],
                    "query_id": int(doc_meta["query_id"]),
                    "passage_index": int(doc_meta["passage_index"]),
                    "is_selected": int(doc_meta["is_selected"]),
                    "source": doc_meta.get("source", "ai4bharat/MSMARCO-XI"),
                    "english_text": doc_meta.get("english_text", ""),
                    "query": doc_meta.get("query", ""),
                    "query_type": doc_meta.get("query_type", "")
                }
                results.append(result_item)
        return results


class HybridRetriever:
    """
    Hybrid Retriever executing parallel Dense (FAISS) + Lexical (BM25) searches,
    fused via Reciprocal Rank Fusion (RRF).
    """

    def __init__(
        self,
        dense_retriever: VectorRetriever,
        bm25_retriever: BM25Retriever,
        metadata_loader: CorpusMetadataLoader,
        dense_candidate_k: int = 20,
        bm25_candidate_k: int = 20,
        rrf_k: int = 60,
        dense_weight: float = 1.0,
        bm25_weight: float = 1.0
    ):
        self.dense_retriever = dense_retriever
        self.bm25_retriever = bm25_retriever
        self.metadata_loader = metadata_loader
        self.dense_candidate_k = dense_candidate_k
        self.bm25_candidate_k = bm25_candidate_k
        self.fusion = ReciprocalRankFusion(k=rrf_k, dense_weight=dense_weight, bm25_weight=bm25_weight)

    def retrieve_bm25_candidates(self, query: str, top_k: int = 20) -> List[Dict[str, Any]]:
        """Helper to fetch BM25 top_k candidate list with full document metadata."""
        scores, indices = self.bm25_retriever.search(query, top_k=top_k)
        results = []
        if indices.size > 0:
            for rank_idx, (score, row_idx) in enumerate(zip(scores[0], indices[0]), start=1):
                if row_idx < 0:
                    continue
                doc_meta = self.metadata_loader.get_document(int(row_idx))
                results.append({
                    "rank": rank_idx,
                    "document_id": doc_meta["document_id"],
                    "score": float(score),
                    "text": doc_meta["text"],
                    "language": doc_meta["language"],
                    "query_id": int(doc_meta["query_id"]),
                    "passage_index": int(doc_meta["passage_index"]),
                    "is_selected": int(doc_meta["is_selected"]),
                    "source": doc_meta.get("source", "ai4bharat/MSMARCO-XI")
                })
        return results

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Executes hybrid retrieval:
          1. Dense FAISS top-(dense_candidate_k) search
          2. Lexical BM25 top-(bm25_candidate_k) search
          3. RRF rank fusion -> top_k final results
        """
        dense_candidates = self.dense_retriever.retrieve(query, top_k=self.dense_candidate_k)
        bm25_candidates = self.retrieve_bm25_candidates(query, top_k=self.bm25_candidate_k)

        final_fused_results = self.fusion.fuse(
            dense_results=dense_candidates,
            bm25_results=bm25_candidates,
            top_k=top_k
        )
        return final_fused_results


class ProductionRetriever:
    """
    Production-grade low-latency retriever (TASK 12).
    Primary strategy: Dense FAISS retrieval.
    Optional fallback: BM25 lexical retrieval when confidence is LOW.
    Guarantees low latency (<200ms) by excluding heavy neural cross-encoders.
    """

    def __init__(
        self,
        vector_retriever: VectorRetriever,
        bm25_retriever: Optional[BM25Retriever] = None,
        metadata_loader: Optional[CorpusMetadataLoader] = None,
        policy: Optional[RetrievalPolicy] = None,
        confidence_evaluator: Optional[RetrievalConfidenceEvaluator] = None
    ):
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        self.metadata_loader = metadata_loader or getattr(vector_retriever, "metadata_loader", None)
        self.policy = policy or DEFAULT_RETRIEVAL_POLICY
        self.confidence_evaluator = confidence_evaluator or RetrievalConfidenceEvaluator(
            min_dense_score=self.policy.min_dense_score
        )

    def retrieve_bm25_candidates(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Helper to fetch BM25 top_k candidate list with document metadata."""
        if self.bm25_retriever is None or self.metadata_loader is None:
            return []
        scores, indices = self.bm25_retriever.search(query, top_k=top_k)
        results = []
        if indices.size > 0:
            for rank_idx, (score, row_idx) in enumerate(zip(scores[0], indices[0]), start=1):
                if row_idx < 0:
                    continue
                doc_meta = self.metadata_loader.get_document(int(row_idx))
                results.append({
                    "rank": rank_idx,
                    "document_id": doc_meta["document_id"],
                    "score": float(score),
                    "text": doc_meta["text"],
                    "language": doc_meta["language"],
                    "query_id": int(doc_meta["query_id"]),
                    "passage_index": int(doc_meta["passage_index"]),
                    "is_selected": int(doc_meta["is_selected"]),
                    "source": doc_meta.get("source", "ai4bharat/MSMARCO-XI"),
                    "english_text": doc_meta.get("english_text", ""),
                    "query": doc_meta.get("query", ""),
                    "query_type": doc_meta.get("query_type", "")
                })
        return results

    def retrieve(
        self,
        query: str,
        policy_override: Optional[RetrievalPolicy] = None
    ) -> Dict[str, Any]:
        """
        Executes production retrieval flow:
          1. Query embedding -> FAISS dense search.
          2. Lightweight confidence assessment.
          3. If confidence is HIGH or fallback disabled: return dense results.
          4. If confidence is LOW and fallback enabled: execute BM25 fallback.
        """
        t0 = time.time()
        active_policy = policy_override or self.policy

        # Step 1: Perform Dense FAISS Retrieval
        dense_results = self.vector_retriever.retrieve(query, top_k=active_policy.dense_top_k)

        # Step 2: Assess Retrieval Confidence
        confidence = self.confidence_evaluator.evaluate(
            dense_results,
            min_score_override=active_policy.min_dense_score
        )

        fallback_used = False
        final_candidates: List[Dict[str, Any]] = []

        # Step 3: High Confidence or Fallback Disabled or no BM25 retriever
        if confidence.decision == "HIGH_CONFIDENCE" or not active_policy.fallback_enabled or self.bm25_retriever is None:
            final_candidates = [dict(c) for c in dense_results[:active_policy.final_top_k]]
        else:
            # Step 4: Low Confidence -> Activate BM25 Fallback
            fallback_used = True
            bm25_candidates = self.retrieve_bm25_candidates(query, top_k=active_policy.fallback_top_k)

            seen_ids = set()
            for cand in bm25_candidates:
                doc_id = cand["document_id"]
                if doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    final_candidates.append(dict(cand))
                if len(final_candidates) >= active_policy.final_top_k:
                    break

            if len(final_candidates) < active_policy.final_top_k:
                for cand in dense_results:
                    doc_id = cand["document_id"]
                    if doc_id not in seen_ids:
                        seen_ids.add(doc_id)
                        final_candidates.append(dict(cand))
                    if len(final_candidates) >= active_policy.final_top_k:
                        break

        # Re-assign standard rank indices 1..K
        for rank_idx, item in enumerate(final_candidates, start=1):
            item["rank"] = rank_idx

        elapsed_ms = round((time.time() - t0) * 1000, 2)

        return {
            "results": final_candidates,
            "confidence": confidence.to_dict(),
            "fallback_used": fallback_used,
            "dense_count": len(dense_results),
            "latency_ms": elapsed_ms
        }

