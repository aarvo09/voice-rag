"""
Data models for corpus ingestion and normalized document schema.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass
class Document:
    """
    Represents a normalized passage document extracted from MSMARCO-XI dataset.
    One passage corresponds to one document.
    """
    document_id: str
    text: str
    language: str
    query_id: int
    passage_index: int
    is_selected: int
    source: str = "ai4bharat/MSMARCO-XI"
    english_text: Optional[str] = ""
    query: Optional[str] = ""
    query_type: Optional[str] = ""
    source_lang: Optional[str] = "en"
    target_lang: Optional[str] = "hi"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_id": self.document_id,
            "text": self.text,
            "language": self.language,
            "query_id": int(self.query_id),
            "passage_index": int(self.passage_index),
            "is_selected": int(self.is_selected),
            "source": self.source,
            "english_text": self.english_text or "",
            "query": self.query or "",
            "query_type": self.query_type or "",
            "source_lang": self.source_lang or "en",
            "target_lang": self.target_lang or "hi"
        }
