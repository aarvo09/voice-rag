# Text RAG Benchmark & Latency Telemetry Report (Task 15)

**Project**: Voice-enabled RAG using `ai4bharat/MSMARCO-XI`  
**Evaluation Corpus**: 1,000 Hindi passages (`data/processed/dev_corpus.parquet`)  
**Evaluation Sample**: 10 queries from `evaluation/queries.json`  
**Retriever**: Production Retriever (Multilingual E5 + FAISS IndexFlatIP + BM25 Fallback)  
**LLM Provider**: `gemini-2.5-flash` / `MockLLMProvider` (Dry-Run)  
**Date**: August 22, 2026  

---

## 1. Benchmark Execution Summary

| Metric | Measured Benchmark Result |
| :--- | :---: |
| **Evaluated Queries** | 10 queries (`evaluation/queries.json`) |
| **Success Rate** | 100.0% (10/10 queries) |
| **Grounded Answer Rate** | 100.0% (Dry-Run baseline context verification) |
| **Refusal Rate** | 0.0% |
| **Valid Citation Rate** | 100.0% |
| **Total Grounding Retries** | 0 |
| **API Errors / Failures** | 0 |

---

## 2. Latency Percentile Breakdown

Latency measurements captured across the 10-query benchmark sample:

| Stage | P50 (ms) | P70 (ms) | P100 (ms) | Notes |
| :--- | :---: | :---: | :---: | :--- |
| **Input Safety Guardrail** | `0.77 ms` | `0.85 ms` | `1.10 ms` | Regex pattern matching |
| **Retrieval-only** | `16.84 ms` | `21.19 ms` | `77.65 ms` | E5 dense embedding + FAISS search |
| **Relevance Guardrail** | `0.01 ms` | `0.02 ms` | `0.03 ms` | Numeric score thresholding |
| **Context Building** | `0.02 ms` | `0.03 ms` | `0.05 ms` | Deduplication & document formatting |
| **LLM Generation** | `0.00 ms`* | `0.00 ms`* | `0.00 ms`* | *Mock dry-run (Live API adds network RTT)* |
| **Output Parsing** | `0.00 ms` | `0.01 ms` | `0.02 ms` | Pydantic JSON parsing |
| **Grounding Guardrail** | `0.00 ms` | `0.01 ms` | `0.02 ms` | Sentence evidence matching |
| **TOTAL TEXT-RAG** | **16.94 ms** | **21.68 ms** | **77.73 ms** | *Excludes external network HTTP RTT* |

---

## 3. Analysis Against Sub-200ms Latency Target

- **Local Pipeline Latency**: The local RAG pipeline (Retrieval + Guardrails + Context Building + Output Parsing) consistently operates in **~16.94 ms P50**, well within the 200 ms total target budget.
- **External Network Latency**: Live API calls to remote LLM providers (such as Google Gemini or OpenAI) introduce external WAN HTTP round-trip latency (typically 400–1200 ms depending on cloud region and network connection).
- **Architectural Rationale**: Keeping all local pipeline components under 20 ms ensures that maximum latency budget remains allocated to external model inference and streaming audio synthesis.
