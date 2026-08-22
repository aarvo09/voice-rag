"""
Sentence Window Chunker Strategy (TASK 09).
Splits document passages into sentences using Hindi/multilingual punctuation boundaries,
then constructs sliding windows of N sentences with a configurable stride.
"""

import re
from typing import List, Dict, Any, Tuple
from app.chunking.base import BaseChunker
from app.chunking.models import Chunk, ChunkConfig

SENTENCE_PATTERN = re.compile(r'[^।॥.?!]+[।॥.?!]+|[^\s।॥.?!]+.*?(?=[।॥.?!]|$)', re.UNICODE)


def split_sentences_with_offsets(text: str) -> List[Tuple[str, int, int]]:
    """Splits text into (sentence_text, start_offset, end_offset) tuples."""
    if not text:
        return []
    sentences: List[Tuple[str, int, int]] = []
    for match in SENTENCE_PATTERN.finditer(text):
        s_text = match.group().strip()
        if s_text:
            sentences.append((s_text, match.start(), match.end()))
    if not sentences and text.strip():
        sentences = [(text.strip(), 0, len(text))]
    return sentences


class SentenceWindowChunker(BaseChunker):
    """Creates overlapping sentence windows of configurable size and stride."""

    def __init__(self, config: ChunkConfig = None):
        super().__init__(config or ChunkConfig(strategy="sentence_window", window_size=3, stride=1))

    def chunk_document(self, doc: Dict[str, Any]) -> List[Chunk]:
        text = doc.get("text", "").strip()
        if not text:
            return []

        doc_id = doc.get("document_id", "")
        sentences = split_sentences_with_offsets(text)
        num_sentences = len(sentences)

        if num_sentences == 0:
            return []

        w_size = self.config.window_size
        stride = self.config.stride
        chunks: List[Chunk] = []

        # If sentence count is smaller than window size, produce a single chunk
        if num_sentences <= w_size:
            chunk_text = text[sentences[0][1]:sentences[-1][2]].strip()
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}_sw0",
                    parent_document_id=doc_id,
                    text=chunk_text,
                    language=doc.get("language", "hi"),
                    query_id=int(doc.get("query_id", -1)),
                    passage_index=int(doc.get("passage_index", -1)),
                    is_selected=int(doc.get("is_selected", 0)),
                    chunk_type="sentence_window",
                    chunk_index=0,
                    start_offset=sentences[0][1],
                    end_offset=sentences[-1][2],
                    metadata={"sentence_count": num_sentences, "window_size": w_size, "stride": stride}
                )
            )
            return chunks

        chunk_idx = 0
        for i in range(0, num_sentences - w_size + 1, stride):
            window_sents = sentences[i : i + w_size]
            start_off = window_sents[0][1]
            end_off = window_sents[-1][2]
            chunk_text = text[start_off:end_off].strip()

            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}_sw{chunk_idx}",
                    parent_document_id=doc_id,
                    text=chunk_text,
                    language=doc.get("language", "hi"),
                    query_id=int(doc.get("query_id", -1)),
                    passage_index=int(doc.get("passage_index", -1)),
                    is_selected=int(doc.get("is_selected", 0)),
                    chunk_type="sentence_window",
                    chunk_index=chunk_idx,
                    start_offset=start_off,
                    end_offset=end_off,
                    metadata={"sentence_count": len(window_sents), "window_size": w_size, "stride": stride}
                )
            )
            chunk_idx += 1

        return chunks
