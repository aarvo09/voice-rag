# Guardrail Layer Architecture & Benchmark Report (Task 14)

**Project**: Voice-enabled RAG using `ai4bharat/MSMARCO-XI`  
**Target Latency Constraint**: Sub-200ms online RAG execution  
**Date**: August 22, 2026  

---

## 1. Guardrail Architecture Overview

The system implements three independent, lightweight, zero-neural guardrail layers surrounding the online retrieval and LLM generation pipeline:

```
User Query
  │
  ▼
[Input Safety Guardrail]  ─────── (Reject: Unsafe / Injection) ──────► Refusal
  │
  ▼ (Pass)
[Production Retriever] (Dense FAISS + Fallback)
  │
  ▼
[Retrieval Relevance Guardrail] ─ (Reject: Weak Evidence / Off-Topic) ─► Refusal
  │
  ▼ (Sufficient)
[Context Builder] -> [LLM Generation] -> [Output Parser]
  │
  ▼
[Grounding & Citation Validator] ─ (Ungrounded / Invalid Citations) ──► Retry (Max 1) / Refusal
  │
  ▼ (Grounded & Valid)
Grounded RAG Response
```

---

## 2. Component Specifications

### A. Input Safety Guardrail (`app/guardrails/input.py`)
- **Rules**:
  - `EMPTY_INPUT`: Rejects blank or whitespace queries.
  - `PROMPT_INJECTION`: Detects instruction overrides, system prompt exfiltration attempts, and jailbreak patterns (English & Hindi).
  - `UNSAFE_REQUEST`: Rejects malicious or dangerous activity requests.
  - `SAFE`: Passes legitimate domain queries.
- **Action**: Immediate refusal without triggering retrieval or LLM calls (`status = refused_unsafe`).

### B. Retrieval Relevance & Off-Topic Guardrail (`app/guardrails/relevance.py`)
- **Signals**: Top-1 similarity score, candidate count, score gap, and retriever confidence.
- **Configured Threshold**: `min_top1_score = 0.60`.
- **Off-Topic Behavior**: Queries for which the knowledge base lacks sufficient vector/lexical evidence (e.g. unindexed news or sports questions) yield `INSUFFICIENT` decision and trigger a deterministic refusal (`"I couldn't find enough relevant information in the provided knowledge base."`) without making an LLM call.

### C. Grounding & Citation Validator (`app/guardrails/grounding.py`)
- **Citation Validation**: Verifies that every cited `document_id` exists in the set of actually retrieved candidates and contains no duplicates or fabrications.
- **Independent Evidence Matching**: Performs sentence-level keyword overlap and novel term counting against retrieved passage text (`min_sentence_overlap_ratio = 0.35`, `max_novel_terms = 3`).
- **Unsupported Claim Detection**: Identifies sentences that introduce hallucinated entities not present in retrieved context.

### D. Guardrail Policy & Retry Engine (`app/guardrails/policy.py`)
- **Max Retries**: `max_grounding_retries = 1`.
- **Flow**: If an answer fails grounding or citation validation, generation is retried once. If it remains ungrounded after retry, the system issues a controlled refusal.

---

## 3. Measured Guardrail Latency Breakdown

| Guardrail Component | Measured P50 Latency (ms) | Execution Cost |
| :--- | :---: | :--- |
| **Input Safety Guardrail** | `0.02 ms` | Regex pattern matching |
| **Retrieval Relevance Guardrail** | `0.01 ms` | Numeric threshold evaluation |
| **Grounding & Citation Validator** | `0.18 ms` | Token set intersection & citation check |
| **TOTAL Guardrail Overhead** | **~0.21 ms** | **< 0.25% of total RAG budget** |

---

## 4. Test Suite Results

All 75 system tests passed cleanly (`75/75 passed in 16.93s`):
- `tests/test_input_guardrail.py` (4/4 passed)
- `tests/test_relevance_guardrail.py` (4/4 passed)
- `tests/test_grounding_guardrail.py` (4/4 passed)
- `tests/test_guardrail_policy.py` (4/4 passed)
- `tests/test_guarded_rag.py` (5/5 passed)

---

## 5. Limitations & Design Considerations

> [!NOTE]
> 1. **Lightweight Lexical Grounding**: The grounding validator uses token overlap and novel entity term counting. It intentionally avoids neural NLI (Natural Language Inference) models or secondary LLM evaluators to maintain sub-200ms latency.
> 2. **Paraphrase Sensitivity**: Highly paraphrased valid statements may occasionally trigger grounding retries if vocabulary overlap is low; however, the single-retry mechanism mitigates false refusals.
> 3. **Non-Claim Refusals**: Standard refusal strings are explicitly recognized as grounded responses to prevent infinite retry loops.
