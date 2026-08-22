"""
Context Builder for formatting retrieved documents into structured prompts (TASK 13).
Preserves provenance, limits document count, deduplicates, and maintains rank order.
"""

from typing import List, Dict, Any, Tuple


class ContextBuilder:
    """
    Formats candidate document metadata into structured prompt context tagged blocks.
    """

    def __init__(self, max_context_documents: int = 5):
        self.max_context_documents = max_context_documents

    def build_context(self, documents: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Processes candidate documents:
          1. Deduplicates by document_id while preserving rank order.
          2. Limits count to max_context_documents.
          3. Formats tagged string blocks.

        Returns:
          (formatted_context_str, selected_documents_list)
        """
        seen_ids = set()
        selected_docs = []

        for doc in documents:
            doc_id = str(doc.get("document_id", ""))
            if not doc_id or doc_id in seen_ids:
                continue
            seen_ids.add(doc_id)
            selected_docs.append(doc)
            if len(selected_docs) >= self.max_context_documents:
                break

        if not selected_docs:
            return "NO CONTEXT AVAILABLE", []

        blocks = []
        for idx, doc in enumerate(selected_docs, start=1):
            doc_id = doc.get("document_id", f"doc_{idx}")
            score = doc.get("score", 0.0)
            text = doc.get("text", "").strip()

            block = (
                f"[DOCUMENT {idx}]\n"
                f"document_id: {doc_id}\n"
                f"score: {score:.4f}\n"
                f"text: {text}"
            )
            blocks.append(block)

        formatted_str = "\n\n".join(blocks)
        return formatted_str, selected_docs
