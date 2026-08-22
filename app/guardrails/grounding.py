"""
Grounding Validator and Citation Validator (TASK 14).
Performs independent evidence matching and citation validation without extra neural models.
"""

import re
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Set, Tuple

logger = logging.getLogger(__name__)

# Basic stopwords to filter during lexical evidence matching
STOP_WORDS = {
    "is", "was", "are", "were", "the", "a", "an", "and", "or", "in", "on", "at", "to", "for", "of", "with", "by", "that", "this",
    "का", "की", "के", "में", "से", "को", "पर", "और", "है", "हैं", "था", "थी", "थे", "यह", "वह", "द्वारा", "लिए", "एक", "या"
}


@dataclass
class GroundingResult:
    """Result returned by GroundingValidator."""
    grounded: bool
    confidence: float
    unsupported_claims: List[str] = field(default_factory=list)
    citations_valid: bool = True
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "grounded": self.grounded,
            "confidence": self.confidence,
            "unsupported_claims": self.unsupported_claims,
            "citations_valid": self.citations_valid,
            "reason": self.reason
        }


class GroundingValidator:
    """
    Independent Lightweight Grounding and Citation Validator.
    Applies citation validation and sentence-level evidence overlap matching.
    """

    def __init__(self, min_sentence_overlap_ratio: float = 0.35, max_novel_terms: int = 3):
        self.min_sentence_overlap_ratio = min_sentence_overlap_ratio
        self.max_novel_terms = max_novel_terms

    @staticmethod
    def extract_keywords(text: str) -> Set[str]:
        """Extracts normalized words excluding stop words and punctuation."""
        words = re.findall(r"[^\s,.:;!?|\"'\(\)\[\]{}॥।]+", text.lower())
        return {w for w in words if w not in STOP_WORDS and len(w) > 1 and not w.isdigit()}

    def validate_citations(
        self,
        citations: List[str],
        retrieved_documents: List[Dict[str, Any]]
    ) -> Tuple[bool, str]:
        """
        Verifies all cited document IDs exist in retrieved_documents.
        """
        valid_doc_ids = {str(doc.get("document_id", "")) for doc in retrieved_documents if doc.get("document_id")}

        if not citations:
            return False, "No citations provided."

        # Check uniqueness
        if len(citations) != len(set(citations)):
            return False, "Duplicate document IDs found in citations."

        for doc_id in citations:
            if doc_id not in valid_doc_ids:
                logger.warning(f"Fabricated or unretrieved document ID cited: '{doc_id}'")
                return False, f"Cited document ID '{doc_id}' was not retrieved in context."

        return True, "All citations are valid and retrieved."

    def validate(
        self,
        query: str,
        answer: str,
        citations: List[str],
        retrieved_documents: List[Dict[str, Any]]
    ) -> GroundingResult:
        """
        Performs independent grounding and citation validation.
        """
        # Step 1: Validate Citations
        citations_valid, citation_reason = self.validate_citations(citations, retrieved_documents)
        if not citations_valid:
            return GroundingResult(
                grounded=False,
                confidence=0.0,
                unsupported_claims=[f"Invalid citation: {citation_reason}"],
                citations_valid=False,
                reason=f"Citation validation failed: {citation_reason}"
            )

        # Refusal check: if answer is a standard refusal string, it is valid grounded refusal
        if "couldn't find enough relevant information" in answer.lower() or "पर्याप्त जानकारी नहीं" in answer:
            return GroundingResult(
                grounded=False,
                confidence=0.0,
                unsupported_claims=[],
                citations_valid=True,
                reason="Answer is a standard refusal response."
            )

        if not retrieved_documents:
            return GroundingResult(
                grounded=False,
                confidence=0.0,
                unsupported_claims=[answer],
                citations_valid=citations_valid,
                reason="No retrieved context available to support answer."
            )

        # Build context keyword set from retrieved documents
        if citations:
            cited_set = set(citations)
            target_docs = [doc for doc in retrieved_documents if str(doc.get("document_id", "")) in cited_set]
            if not target_docs:
                target_docs = retrieved_documents
        else:
            target_docs = retrieved_documents

        combined_context_text = " ".join([doc.get("text", "") for doc in target_docs])
        context_keywords = self.extract_keywords(combined_context_text)

        # Detect cross-lingual case based on dominant script character counts (e.g. English answer with Hindi context)
        latin_ans = len(re.findall(r"[a-zA-Z]", answer))
        deva_ans = len(re.findall(r"[\u0900-\u097F]", answer))
        latin_ctx = len(re.findall(r"[a-zA-Z]", combined_context_text))
        deva_ctx = len(re.findall(r"[\u0900-\u097F]", combined_context_text))

        ans_is_latin = latin_ans > deva_ans
        ctx_is_deva = deva_ctx > latin_ctx
        ans_is_deva = deva_ans > latin_ans
        ctx_is_latin = latin_ctx > deva_ctx

        is_cross_lingual = (ans_is_latin and ctx_is_deva) or (ans_is_deva and ctx_is_latin)

        # Split answer into sentences
        sentences = [s.strip() for s in re.split(r"[.!?|।]", answer) if s.strip()]
        if not sentences:
            sentences = [answer.strip()]

        unsupported_claims = []
        supported_count = 0

        for sentence in sentences:
            sent_keywords = self.extract_keywords(sentence)
            if not sent_keywords:
                supported_count += 1
                continue

            overlap = sent_keywords.intersection(context_keywords)
            novel_terms = sent_keywords - context_keywords
            overlap_ratio = len(overlap) / float(len(sent_keywords))

            # Sentence evaluation:
            # If cross-lingual with valid citations, accept sentence as grounded in cited context
            if is_cross_lingual and citations and citations_valid:
                supported_count += 1
            elif overlap_ratio >= self.min_sentence_overlap_ratio or len(novel_terms) <= self.max_novel_terms:
                supported_count += 1
            else:
                unsupported_claims.append(sentence)

        confidence_score = round(supported_count / float(len(sentences)), 4) if sentences else 1.0
        is_grounded = (len(unsupported_claims) == 0) and citations_valid

        reason = "Answer is fully supported by retrieved context and valid citations." if is_grounded else f"Found {len(unsupported_claims)} unsupported sentence claims."

        return GroundingResult(
            grounded=is_grounded,
            confidence=confidence_score,
            unsupported_claims=unsupported_claims,
            citations_valid=citations_valid,
            reason=reason
        )
