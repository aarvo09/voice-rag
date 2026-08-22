# Repository Cleanup Audit Report

This report outlines the dependencies, usage, and recommended git structure for the `voice-rag` project, prioritizing the final Groq/Sarvam FastAPI + React architecture.

## 1. Current Repository Tree Status
The repository contains a mix of final production code, comprehensive evaluation scripts, and artifacts from deprecated experiments (Gemini LLM, BGE Reranker, RRF Hybrid Retrieval, and chunk size variations). 

## 2. Production-Required Files (KEEP)
These files are strictly required for the final live `live_voice_rag.py` / `serve.py` execution path:
- `app/__init__.py`
- `app/api/__init__.py`, `app/api/routes.py`
- `app/embeddings/__init__.py`, `app/embeddings/model.py`
- `app/generation/__init__.py`, `app/generation/config.py`, `app/generation/context.py`, `app/generation/interface.py`, `app/generation/llm.py`, `app/generation/models.py`, `app/generation/parser.py`, `app/generation/prompts.py`
- `app/guardrails/__init__.py`, `app/guardrails/grounding.py`, `app/guardrails/input.py`, `app/guardrails/policy.py`, `app/guardrails/relevance.py`
- `app/pipeline/__init__.py`, `app/pipeline/policies.py`, `app/pipeline/state.py`, `app/pipeline/text_rag.py`
- `app/retrieval/__init__.py`, `app/retrieval/bm25.py`, `app/retrieval/confidence.py`, `app/retrieval/faiss_index.py`, `app/retrieval/metadata.py`, `app/retrieval/retriever.py`
- `app/stt/__init__.py`, `app/stt/interface.py`, `app/stt/sarvam.py`
- `frontend/*` (All source code for the React UI)
- `scripts/serve.py`
- `scripts/live_voice_rag.py`
- `requirements.txt`, `.env.example`, `Dockerfile`, `README.md`, `pyproject.toml`

## 3. Development / Evaluation Files (KEEP)
These are not required at runtime but are critical for engineering reproducibility:
- **Scripts:** `benchmark_text_rag.py`, `build_bm25_index.py`, `build_dev_corpus.py`, `build_embeddings.py`, `build_faiss_index.py`, `evaluate_production_retriever.py`, `inspect_dataset.py`, `test_bm25.py`, `test_embedding.py`, `test_generation.py`, `test_real_generation.py`, `test_relevance_scores.py`, `test_sarvam_stt.py`, `test_voice_rag.py`
- **Tests:** `tests/test_context_builder.py`, `tests/test_generation_*.py`, `tests/test_grounding_guardrail.py`, `tests/test_guarded_rag.py`, `tests/test_guardrail_policy.py`, `tests/test_input_guardrail.py`, `tests/test_metrics.py`, `tests/test_production_retriever.py`, `tests/test_relevance_guardrail.py`, `tests/test_retrieval_confidence.py`
- **Reports:** `dataset_acquisition.md`, `dataset_forensics.md`, `dev_corpus.md`, `production_retrieval_decision.md`, `production_retriever_thresholds.md`, `retrieval_baseline.md`, `retrieval_evaluation.md`, `text_rag_live_benchmark.md`, `guardrail_evaluation.md`
- **Evaluation:** `evaluation/*` 
- **Base Chunking:** `app/chunking/__init__.py`, `app/chunking/base.py`, `app/chunking/native.py`, `app/chunking/models.py`, `app/chunking/registry.py`

## 4. Obsolete Candidates (DELETE)
These files relate to explicitly abandoned approaches, one-off debugging, or are simply empty (0 bytes).

**Abandoned Approaches:**
- **BGE Reranker:** `app/retrieval/reranker.py`, `scripts/test_reranker.py`, `tests/test_reranker.py`, `reports/reranker_error_analysis.md`
- **Hybrid RRF:** `app/retrieval/fusion.py`, `scripts/test_hybrid_retrieval.py`, `reports/hybrid_retrieval.md` *(Note: `app/retrieval/retriever.py` imports `fusion.py`, so the import must be removed before deleting).*
- **Gemini-only Path:** `app/generation/gemini.py`, `scripts/test_gemini_connection.py`, `tests/test_gemini_provider.py`, `reports/gemini_live_test.md` *(Note: `app/generation/llm.py` imports `gemini.py`, so the factory function needs a cleanup).*
- **Chunk Variations:** `app/chunking/fixed.py`, `app/chunking/semantic.py`, `app/chunking/sentence_window.py`, `scripts/build_chunk_variants.py`, `scripts/evaluate_chunk_retrieval.py`, `tests/test_chunk_retrieval_evaluation.py`, `tests/test_chunking.py`, `reports/chunk_error_analysis.md`, `reports/chunk_examples.md`, `reports/chunking_statistics.md`, `reports/chunk_retrieval_evaluation.md`

**One-off Scripts & Caches:**
- `test_groq_curl.py`, `test_groq_curl2.py`
- `test.wav` (temporary artifact from live mic)

**Empty Files (0 Bytes):**
- `app/api/schemas.py`, `app/ingestion/corpus_builder.py`, `app/ingestion/deduplicator.py`, `app/ingestion/loader.py`, `app/ingestion/normalizer.py`, `app/observability/metrics.py`, `app/observability/tracing.py`, `app/pipeline/controller.py`, `scripts/benchmark.py`, `scripts/build_corpus.py`, `scripts/build_indexes.py`, `scripts/__init__.py`, `tests/test_guardrails.py`, `tests/test_latency.py`, `tests/test_pipeline.py`, `tests/test_retrieval.py`

## 5. Duplicate Candidates
There is no significant structural duplication, but `scripts/evaluate_production_retriever.py` and `scripts/benchmark_text_rag.py` share similar evaluation logic. We recommend keeping both since one focuses on retrieval recall and the other on E2E guardrail latency.

## 6. Data Artifacts
- **KEEP (Git LFS) / REGENERATE:** `data/indexes/dev.faiss`, `data/indexes/dev_bm25.pkl`, `data/processed/dev_corpus.parquet`, `data/processed/dev_embeddings.npy`
- **DELETE (Obsolete):** `data/indexes/chunk_variants/*`, `data/processed/chunk_embeddings/*`, `data/processed/chunks/*`

## 7. Reports Audit
- **FINAL / IMPORTANT:** `dataset_acquisition.md`, `production_retrieval_decision.md`, `production_retriever_thresholds.md`, `guardrail_evaluation.md`
- **OBSOLETE:** `reranker_error_analysis.md`, `hybrid_retrieval.md`, `gemini_live_test.md`, `chunk_*.md`

## 8. Test Audit
- **KEEP:** `test_production_retriever.py`, `test_generation_*.py`, `test_guardrail*.py`, `test_context_builder.py`, `test_metrics.py`
- **OBSOLETE:** `test_reranker.py`, `test_gemini_provider.py`, `test_chunking.py`, `test_chunk_retrieval_evaluation.py`

## 9. Script Audit
| Script | Purpose | Used by Final System? | Keep/Delete | Reason |
|--------|---------|------------------------|-------------|--------|
| `live_voice_rag.py` | Terminal interface | Yes | KEEP | Production Demo |
| `serve.py` | FastAPI Backend | Yes | KEEP | Production API |
| `build_dev_corpus.py` | Ingest corpus | No (Dev) | KEEP | Reproducibility |
| `build_embeddings.py` | Embed corpus | No (Dev) | KEEP | Reproducibility |
| `build_indexes.py` | Empty | No | DELETE | 0 Bytes |
| `test_gemini_connection.py`| Gemini setup test | No | DELETE | Abandoned provider |
| `test_reranker.py` | Reranker test | No | DELETE | Abandoned approach |
| `build_chunk_variants.py` | Chunk experiments | No | DELETE | Abandoned approach |

## 10. App Module Audit
- `app/retrieval/`: Uses `bm25.py`, `faiss_index.py`, `retriever.py`. (Delete `fusion.py`, `reranker.py`).
- `app/generation/`: Uses `llm.py`, `prompts.py`, `parser.py`. (Delete `gemini.py`).
- `app/guardrails/`: All modules used by final runtime.
- `app/stt/`: All modules used by final runtime.

## 11. Frontend/Backend Audit
The stack is highly streamlined:
- **Backend Entry:** `scripts/serve.py` → `app/api/routes.py`
- **Frontend Entry:** `frontend/src/main.tsx` → `frontend/src/App.tsx`
No unused components or routes were found in the API or Frontend layer.

## 12. Git Cleanliness Audit
The `.gitignore` needs to ensure the following are ignored:
```
.env
.venv/
__pycache__/
*.pyc
.pytest_cache/
.vscode/
node_modules/
frontend/dist/
logs/
*.log
test.wav
```
*(No API keys were detected in tracked files).*

## 13. GitHub Size Audit
- **> 100 MB:** None
- **> 50 MB:** None
- **> 10 MB:** None
- **Largest files found:** `data/processed/chunk_embeddings/sentence_window.npy` (5.2 MB) and `data/indexes/chunk_variants/sentence_window.faiss` (5.2 MB)
- **Recommendation:** Although no files exceed GitHub's strict 100MB limit, we recommend tracking all `.faiss`, `.npy`, and `.parquet` files in **Git LFS** or adding them to `.gitignore` so users must regenerate them.

## Final Recommended Repository Tree
```
voice-rag/
├── app/
│   ├── api/
│   ├── chunking/        # Native only
│   ├── embeddings/
│   ├── generation/      # Groq focused
│   ├── guardrails/
│   ├── pipeline/
│   ├── retrieval/       # FAISS + BM25 only
│   └── stt/
├── data/
│   ├── indexes/         # dev.faiss, dev_bm25.pkl
│   └── processed/       # dev_corpus.parquet
├── evaluation/
├── frontend/
├── reports/             # Only final decision reports
├── scripts/             # serve.py, live_voice_rag.py, build_*.py
├── tests/               # Guardrail and pipeline tests
├── .env.example
├── .gitignore
├── Dockerfile
├── README.md
└── requirements.txt
```
