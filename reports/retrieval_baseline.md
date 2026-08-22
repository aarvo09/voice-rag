# Retrieval Baseline Comparison Report: Dense FAISS vs. Lexical BM25

**Project**: Voice-enabled RAG using `ai4bharat/MSMARCO-XI` (Hindi Dev Corpus)  
**Corpus Size**: 1,000 document passages across 100 source queries  
**Date**: August 22, 2026  

---

## 1. Overview & Objectives

This report provides an empirical baseline comparison between **Dense Vector Retrieval (FAISS)** and **Lexical Text Search (BM25)** on the 1,000-document Hindi development corpus (`data/processed/dev_corpus.parquet`).

- **Dense Retriever**: `intfloat/multilingual-e5-small` embeddings ($d=384$, float32, L2-normalized) with `FAISS IndexFlatIP`.
- **Lexical Retriever**: `rank_bm25.BM25Okapi` ($k_1=1.5, b=0.75$) with custom Unicode-aware Indic tokenizer.

---

## 2. Empirical Benchmark Comparison Table

| Benchmark Metric | FAISS (Dense Vector) | BM25 (Lexical) |
| :--- | :--- | :--- |
| **Algorithm / Architecture** | `intfloat/multilingual-e5-small` + `FAISS IndexFlatIP` | `BM25Okapi` ($k_1=1.5, b=0.75$) |
| **Hit@1** | **10.0%** (1/10) | **10.0%** (1/10) |
| **Hit@5** | **50.0%** (5/10) | **40.0%** (4/10) |
| **Online Retrieval Latency (excl. startup)** | **11.90 ms** | **10.36 ms** |
| **Startup / Index Loading Time** | 10,214.28 ms (10.21 s) | **145.47 ms** |
| **Index File Size** | 1,500.04 KB (1.50 MB) | **859.05 KB** (0.86 MB) |
| **Active Process Memory (RSS)** | 1,490.61 MB | **188.12 MB** |
| **Search Determinism** | Verified (`True`) | Verified (`True`) |

---

## 3. Key Observations & Findings

### 3.1 Retrieval Quality (Hit@1 & Hit@5)
- **FAISS (Dense)** achieved higher recall at top-5 (**Hit@5 = 50.0%** vs **BM25 = 40.0%**). Dense semantic vectors successfully capture conceptual similarity even when query words are paraphrased or translated in Devanagari.
- **BM25 (Lexical)** achieved equivalent top-1 accuracy (**Hit@1 = 10.0%**). For exact term queries (such as `"मैनहट्टन परियोजना"`), BM25 placed the exact ground-truth selected passage at **Rank 1** with a high score of $27.32$.

### 3.2 Online Latency
- Both systems exhibit sub-12ms online retrieval times:
  - FAISS Vector Search: **11.90 ms** ($11.74$ms query embedding + $0.16$ms vector search).
  - BM25 Lexical Search: **10.36 ms** ($0.01$ms tokenization + $10.32$ms term matching).

### 3.3 Resource Footprint & Startup
- **BM25** is dramatically lighter on memory (**188.12 MB** RAM vs **1,490.61 MB** RAM for PyTorch/E5) and loads nearly instantly (**145.47 ms** vs **10.21 s** for PyTorch Transformer model load).
- **FAISS** index size ($1.50$ MB) reflects raw float32 vector storage ($1000 \times 384 \times 4$ bytes), whereas BM25 pickle ($0.86$ MB) reflects compressed inverted index mappings.

---

## 4. Architectural Recommendation for Hybrid Retrieval

Neither retriever alone is sufficient for optimal RAG performance:
1. **Dense Retrieval** excels at capturing semantic intent, handling synonyms, and resolving cross-lingual concepts.
2. **Lexical Retrieval (BM25)** excels at exact keyword matching, rare proper nouns, and specific names.

**Next Steps**: Combine FAISS and BM25 using **Reciprocal Rank Fusion (RRF)** or **Score Normalization Fusion** in Task 08 to achieve higher Hit@1 and Hit@5 accuracy.
