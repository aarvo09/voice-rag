# MSMARCO-XI Acquisition Analysis

## Training File
- **URL**: `https://huggingface.co/datasets/ai4bharat/MSMARCO-XI/resolve/main/train/hintrain.parquet`
- **Size**: 3,719,813,179 bytes (~3.46 GB)
- **Format**: Parquet (Single row group containing 778,638 query-answer records)

## Validation File
- **URL**: `https://huggingface.co/datasets/ai4bharat/MSMARCO-XI/resolve/main/validation/hinval.parquet`
- **Size**: 461,888,616 bytes (~440.49 MB)
- **Format**: Parquet (Single row group containing 97,941 query-answer records)

## Available Files
Discovered 28 total files in the repository:
- **14 Training Shards** (`train/*train.parquet`): 3.4 GB to 3.8 GB each across Indic languages (`hin`, `asm`, `ben`, `guj`, `kan`, `mal`, `mar`, `nep`, `ori`, `pan`, `san`, `tam`, `tel`, `urd`).
- **14 Validation Shards** (`validation/*val.parquet`): 419 MB to 494 MB each across Indic languages.
- **Micro-shards / Subsets**: None. Hugging Face does not expose pre-built development subsets (<10 MB) for `ai4bharat/MSMARCO-XI`.

## Python Loading Problem
In our local environment, attempting to read `ai4bharat/MSMARCO-XI` via `datasets.load_dataset(..., streaming=True)` or remote HTTP range requests (`fsspec` + `pyarrow`) caused Python memory usage to grow to **~5 GB RAM** before retrieving the first record.

### Root Cause Analysis:
1. **Single Row Group**: Both `hintrain.parquet` and `hinval.parquet` were written with **only 1 row group** containing all 778,638 rows (or 97,941 rows).
2. **Dictionary & Column Page Reads over HTTP**: PyArrow's C++ Parquet decoder must read the dictionary page and column metadata for the entire row group to yield batch 0. Over remote HTTP range requests, this triggers multi-hundred megabyte chunk downloads and heavy buffering in memory.

## Development Options

| Option | Disk Requirement | RAM Implications | Suitability for Rapid Development |
|---|---|---|---|
| **1. Full Training Download** | ~3.46 GB | High (Loads multi-GB Parquet structures into RAM if read via Pandas/Datasets) | **Low** — Large download overhead and slow initial load times for testing. |
| **2. Validation Download** | ~440.49 MB | Moderate (~440 MB download; PyArrow batch reads ~50 MB RAM) | **Medium** — 8x smaller than training shard; suitable if no local cache exists. |
| **3. Smaller Shard/Subset on Hub** | N/A | N/A | **Unfeasible** — No sub-10 MB development shards exist in HF repository. |
| **4. Bounded Local Dev Corpus from Cache** | 0 B additional (Uses cached `hintrain.parquet` or `data/processed/dev_corpus.parquet`) | Extremely Low (<60 MB RAM for bounded PyArrow batch reading) | **Highest** — Instantaneous execution, sub-50 KB file size (`dev_corpus.parquet`), zero memory risk. |

## Recommended Development Strategy

**Recommendation: Option 4 (Existing Local HF Cache / Bounded 100-Record Dev Corpus)**

1. **Primary Strategy**: Utilize the existing local Hugging Face cache (`~/.cache/huggingface/hub/datasets--ai4bharat--MSMARCO-XI/.../train/hintrain.parquet`) to generate a local 100-record development corpus at `data/processed/dev_corpus.parquet` using bounded PyArrow batch reads (`iter_batches(batch_size=10)`).
2. **Fallback Strategy for Clean Environments**: If deploying to a fresh environment without local cache, download `validation/hinval.parquet` (440.49 MB) once and sample 100 records using bounded PyArrow reading, avoiding full training shard downloads completely.
