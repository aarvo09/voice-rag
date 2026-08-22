# Development Corpus Report

> [!NOTE]
> This is a bounded development corpus, not the full MSMARCO-XI corpus.

## Source
- **Local Source Path**: `/home/arvind/.cache/huggingface/hub/datasets--ai4bharat--MSMARCO-XI/snapshots/bf5cdc1f26e581e519018e434db14edd1b77602b/train/hintrain.parquet`
- **Source File Size**: 3.46 GB (3,719,813,179 bytes)
- **Source Format**: Parquet (`ai4bharat/MSMARCO-XI`)

## Acquisition & Extraction Summary
- **Source Records Inspected**: 100
- **Total Source Passages Encountered**: 1000
- **Output Documents Written**: 1000
- **Empty Passages Skipped**: 0

## Output File
- **Corpus Path**: `data/processed/dev_corpus.parquet`
- **File Size**: 473.58 KB (484,949 bytes)
- **Format**: Parquet

## Output Schema
| Column Name | Arrow Data Type | Description / Sample Value |
|---|---|---|
| `document_id` | `string` | `"1185869_0"` |
| `text` | `string` | Primary Hindi passage text |
| `language` | `string` | `"hi"` |
| `query_id` | `int64` | `1185869` |
| `passage_index` | `int64` | `0` |
| `is_selected` | `int64` | Ground-truth relevance (`1` or `0`) |
| `source` | `string` | `"ai4bharat/MSMARCO-XI"` |
| `english_text` | `string` | Parallel English passage snippet |
| `query` | `string` | Source Hindi query |
| `query_type` | `string` | Question classification |
| `source_lang` | `string` | `"en"` |
| `target_lang` | `string` | `"hi"` |

## Document Statistics
- **Total Passages / Documents**: 1000
- **Selected Relevant Documents (`is_selected == 1`)**: 63
- **Unselected Candidate Documents (`is_selected == 0`)**: 937

## Empty Passage Handling
- **Skipped Passages**: 0 empty or whitespace-only passage strings were excluded from document creation.

## Duplicate Check
- **Duplicate Document IDs**: 0
- **Duplicate Text Values**: 6

## Memory Usage & Performance
- **Memory Before**: 62.48 MB
- **Memory After**: 3648.41 MB
- **Memory Delta**: 3585.93 MB
- **Processing Time**: 6.336 seconds

## Validation Result
- **Metadata Inspection**: Successfully re-opened output file using PyArrow metadata.
- **Row Count Verified**: 1000 rows.
- **Schema Verified**: All 12 columns correctly mapped and typed.
