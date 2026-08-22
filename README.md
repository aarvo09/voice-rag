# Voice RAG - Hacker House Goa 2026 

A high-performance, voice-enabled Retrieval-Augmented Generation (RAG) system built with strict guardrails, low-latency dense retrieval, and a dynamic editorial frontend.

## Demo Architecture

Microphone Input → Speech-to-Text (Sarvam API) → RAG Pipeline (FastAPI) → LLM Generation (Groq API) → Hacker House Goa UI

## Features
- **Voice-Native**: Direct audio capture to contextual answers.
- **Production-Grade Retrieval**: Fast FAISS dense search with BM25 lexical fallback when confidence is low.
- **Strict Guardrails**: Prevents hallucinations by refusing queries when context is insufficient or ungrounded.
- **Transparent Latency Telemetry**: Granular logging of local operations vs. external API wait times.
- **Beautiful Frontend**: An editorial, studio-inspired React interface matching the energy of Hacker House Goa.

## System Architecture
The production architecture is heavily optimized to maintain strict grounding:

1. **Microphone Capture** (React Frontend)
2. **Speech-to-Text Transcription** (Sarvam Saaras)
3. **Input Safety Guardrail** (Filters malicious/unsafe prompts)
4. **Embedding Generation** (Multilingual E5 Small)
5. **Retrieval Layer**: FAISS Dense Retrieval + BM25 fallback when retrieval confidence is low
6. **Retrieval Sufficiency Guardrail** (Refuses out-of-domain queries before calling LLM)
7. **Context Builder** (Formats retrieved passages)
8. **Generation** (Groq LLM: `openai/gpt-oss-20b`)
9. **Grounding + Citation Validation Guardrails**
10. **Final Answer / Refusal Response**

## Technology Stack
- **Backend Framework**: FastAPI (Python)
- **Frontend Framework**: React, Vite, TypeScript, Tailwind CSS v4
- **Language Model**: Groq (`openai/gpt-oss-20b`)
- **Speech-to-Text**: Sarvam API (Saaras)
- **Embedding Model**: `intfloat/multilingual-e5-small`
- **Vector Store**: FAISS
- **Lexical Search**: BM25

## Dataset
This project indexes data derived from the **`ai4bharat/MSMARCO-XI`** dataset. For this local development and evaluation environment, a bounded **1,000 passage development corpus** was extracted to maintain a low memory footprint and strict sub-second performance.

## Retrieval Approach
The system uses a **low-latency Dense Retrieval Strategy** (`ProductionRetriever`). It defaults to blazing-fast FAISS vector search, and applies a lightweight confidence assessment. If the retrieval confidence is evaluated as LOW, it conditionally falls back to BM25 lexical search to enhance recall without penalizing standard queries with heavy cross-encoder reranking.

## Guardrails
The pipeline enforces four distinct programmatic guardrails:
1. **Input Safety**: Blocks injection attacks and unsafe queries.
2. **Retrieval Sufficiency**: Gates the LLM; if the vector/BM25 search yields irrelevant context, the pipeline halts and refuses.
3. **Grounding Validation**: Validates that the generated answer is entirely derived from the retrieved context.
4. **Citation Validation**: Ensures that every claim made in the generated text is properly cited with a valid document ID.

## Latency Telemetry
The pipeline exposes detailed, granular telemetry to explicitly separate internal engine speeds from external API network constraints. 
- **Local Retrieval Latency**: Measured in single-digit to low double-digit milliseconds (FAISS + BM25).
- **Remote API Latency**: The STT (Sarvam) and LLM generation (Groq) external network calls heavily dominate the pipeline's overall footprint.
*(Note: While the local context engine is extremely fast, the total end-to-end voice pipeline latency is strictly dependent on the network speeds of the external API providers and is not completely under 200 ms.)*

## Frontend
The user interface is an art-directed localhost dashboard inspired by the Hacker House Goa 2026 design language—featuring bold editorial typography, warm Goa-inspired orange accent colors, and sharp, raw studio aesthetics.

## Project Structure
```text
voice-rag/
├── app/               # FastAPI Backend Pipeline (Retrieval, LLM, Guardrails, STT)
├── data/              # FAISS indexes, BM25 indices, Dev Corpus Parquet
├── evaluation/        # Offline benchmarks and latency tests
├── frontend/          # React + Vite web application
├── reports/           # Architectural decisions and performance metrics
├── scripts/           # Pipeline runners, index builders, terminal UI
└── tests/             # Unit tests for guardrails, chunking, and retrievers
```

## Setup & Running Locally

### 1. Backend Setup
```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Variables
Create your configuration file:
```bash
cp .env.example .env
```
Populate `.env` with your API keys:
```env
GOOGLE_API_KEY=           # optional/legacy provider
GROQ_API_KEY=
GROQ_MODEL=openai/gpt-oss-20b
SARVAM_API_KEY=
LLM_PROVIDER=groq
```
*(Never include real API keys in your version control!)*

### 3. Run the Services
**Terminal 1 (Backend FastAPI):**
```bash
# Start the backend server on port 8000
python scripts/serve.py
```

**Terminal 2 (Frontend React):**
```bash
# Install NPM dependencies and start Vite dev server
cd frontend
npm install
npm run dev
```
Navigate to `http://localhost:5173` in your browser.

## Example Queries & Out-of-Domain Behavior

### Supported Domain Query
*User speaks:* **"What were the consequences of the Manhattan Project?"**
*Behavior:* The system easily finds relevant documents within the development corpus, passes the retrieval sufficiency guardrail, generates a strictly grounded answer using the Groq LLM, and successfully returns accurate citations.

### Out-of-Domain Query
*User speaks:* **"Who won the 2022 FIFA World Cup?"**
*Behavior:* 
1. The query is embedded and searched.
2. The `Retrieval Sufficiency` guardrail evaluates the top returned chunks.
3. It detects that none of the retrieved documents contain sufficient evidence pertaining to the FIFA World Cup.
4. **The pipeline halts BEFORE calling the Groq LLM.**
5. The system safely refuses to answer, returning an immediate "Insufficient Knowledge" response to prevent hallucinations.

## Evaluation & Benchmarks
The pipeline's components underwent rigorous forensic analysis and offline benchmarking. See the `reports/` directory for historical evaluations of baseline retrieval recall, guardrail effectiveness (measured against edge-case JSON fixtures), and latency breakdowns.

## Limitations
- **External Network Dependency**: Complete end-to-end pipeline speeds are bottlenecked by Sarvam STT and Groq API response times. 
- **Dataset Scale**: Currently bounded to a 1,000-passage dev corpus for local memory safety.

## Future Improvements
- Migration to a fully local STT and LLM inference engine to remove remote API overhead.
- Scaling the FAISS index partitioning to support the full 15GB MSMARCO-XI corpus natively.

## Hackathon Attribution
Built for **Hacker House Goa 2026** 
