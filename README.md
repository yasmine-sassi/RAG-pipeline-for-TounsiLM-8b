# RAG Pipeline for TounsiLM-8b

A Retrieval-Augmented Generation (RAG) system built on a structured Tunisian Arabic knowledge base, designed to ground [`alabenayed/TounsiLM-8b`](https://huggingface.co/alabenayed/TounsiLM-8b) in verified dialectal knowledge.

## Features

- **Hybrid retrieval** — BM25 (40%) + semantic embeddings (60%) merged via Reciprocal Rank Fusion
- **Query rewriting** — generates up to 3 query variants with Arabizi digit normalization (`7→h`, `5→kh`, `3→a` …)
- **Automatic query routing** — detects entry type from the query and restricts search to the relevant category
- **Confidence scoring** — high / medium / low signal per query based on mean retrieval score
- **Token-based context truncation** — uses TounsiLM's tokenizer to stay within the 4 096-token context window
- **Typed knowledge base** — 1 647 entries across 11 validated types, each with a Pydantic schema

## Project Structure

```
RAG/
├── run_rag.py                        # CLI entry point
├── requirements.txt
├── tounsilm_rag_kaggle.ipynb         # Kaggle test notebook
└── rag_kb/
    ├── data/                         # JSON knowledge base files
    │   ├── expressions.json          # Tunisian expressions
    │   ├── expressions2.json         # Number slang
    │   ├── expressions3.json         # Additional expressions
    │   ├── proverbs.json             # Tunisian proverbs (1 276 entries)
    │   ├── food.json                 # Dishes & ingredients
    │   ├── rituals.json              # Social rituals & greetings
    │   ├── code-switching.json       # French-Tunisian code-switching
    │   ├── series_movies.json        # Tunisian TV & films
    │   └── colors.json               # Colors in Tunisian dialect
    │
    ├── schemas/                      # Pydantic validation schemas
    │   ├── base_schema.py
    │   ├── expression_schema.py
    │   ├── number_slang_schema.py
    │   ├── proverb_schema.py
    │   ├── food_schema.py
    │   ├── ritual_schema.py
    │   ├── code_switch_schema.py
    │   ├── media_schema.py
    │   └── color_schema.py
    │
    ├── pipeline/                     # Core RAG logic
    │   ├── query_rewriter.py         # Arabizi normalization + query routing
    │   ├── build_embed_text.py       # Builds embedding text per entry type
    │   ├── indexer.py                # Loads data → ChromaDB
    │   ├── retriever.py              # Hybrid BM25 + semantic retrieval
    │   ├── validate_entries.py       # JSON validation against schemas
    │   ├── llm_interface.py          # TounsiLM-8b wrapper
    │   └── rag_pipeline.py           # Orchestrates rewrite → retrieve → generate
    │
    ├── scripts/
    │   └── bulk_import.py
    │
    └── db/
        └── chroma_db/                # ChromaDB vector store (auto-created)
```

## Setup

```bash
pip install -r requirements.txt
```

**Dependencies:** `chromadb`, `sentence-transformers`, `transformers`, `torch`, `accelerate`, `pydantic`, `rank_bm25`

## Quick Start

### 1. Build the vector index

Embeds all knowledge base entries and stores them in ChromaDB. Run once per environment:

```bash
python run_rag.py --index
```

To wipe and rebuild from scratch:

```bash
python run_rag.py --index --reset
```

### 2. Test retrieval (no LLM)

Validates retrieval quality before loading the model — fast:

```bash
python run_rag.py --retrieve "ما معنى برشا؟" --top-k 3
python run_rag.py --retrieve "shnow hia el harissa" --top-k 5
python run_rag.py --retrieve "harissa" --type food
```

### 3. Run a RAG query

Retrieves relevant entries and generates an answer with TounsiLM-8b:

```bash
python run_rag.py --query "شنو هي الهريسة التونسية؟"
python run_rag.py --query "shnow maana el k7li fel tounsi?"
python run_rag.py --query "ما معنى اللي فات مات؟" --type proverb
```

### 4. Interactive chat

```bash
python run_rag.py
```

In-session commands:

| Command | Effect |
|---|---|
| `:exit` | Quit |
| `:sources on` / `:sources off` | Toggle source display |
| `:top-k 3` | Change retrieval count |
| `:type food` | Filter by entry type |

## CLI Reference

```
python run_rag.py [OPTIONS]

Subcommands:
  --index                Build / update ChromaDB index
  --index --reset        Wipe and rebuild from scratch
  --query "..."          Run a RAG query (retrieve + generate)
  --retrieve "..."       Retrieve only, no LLM

Retrieval:
  --top-k N              Entries to retrieve (default: 5)
  --type TYPE            Filter by type (see entry types below)
  --min-score F          Minimum similarity score 0–1 (default: 0.0)

Generation:
  --max-tokens N         Max tokens to generate (default: 512)
  --temperature F        Sampling temperature (default: 0.7)
  --max-context-tokens N Max tokens for retrieved context (default: 1500)

Models:
  --embedding-model M    Sentence-transformer model
                         (default: intfloat/multilingual-e5-base)
```

## Architecture

### Query rewriting

Before retrieval, `QueryRewriter` generates up to 3 variants of the user query:

1. **Original** — preserved as-is
2. **Arabizi-normalized** — digit substitutions applied (`7→h`, `5→kh`, `3→a`, `9→q`, `8→gh`)
3. **Cleaned** — punctuation stripped

Results from all variants are merged by keeping the best score per document.

The same module inspects the query for type-specific keywords and automatically sets an `entry_type` filter — e.g. a query mentioning *harissa* or *طبيخ* is routed to `food` without any manual flag.

### Hybrid retrieval

Each query variant is retrieved using two methods in parallel:

| Method | Weight | Strength |
|---|---|---|
| Semantic (`multilingual-e5-base`) | 60% | Meaning, synonymy, cross-language |
| BM25 (`rank_bm25`) | 40% | Exact keyword matches, precise Arabizi terms |

The two ranked lists are merged with **Reciprocal Rank Fusion (RRF)**:

```
RRF(d) = 0.6 / (60 + rank_semantic(d)) + 0.4 / (60 + rank_bm25(d))
```

### Embedding model

`intfloat/multilingual-e5-base` handles Arabic script, Arabizi, and French in a unified embedding space. The E5 asymmetric prefix protocol is applied:

- `"passage: "` prepended at **index time**
- `"query: "` prepended at **retrieval time**

### Confidence scoring

After retrieval, a confidence level is computed from the mean RRF-normalized score:

| Level | Condition |
|---|---|
| High | mean score ≥ 0.75 |
| Medium | 0.50 ≤ mean score < 0.75 |
| Low | mean score < 0.50 |
| None | no results returned |

### Prompt injection

Retrieved entries are injected into TounsiLM-8b's system prompt:

```text
[INST] <<SYS>>
أنت مساعد ذكي متخصص في اللهجة التونسية...

معلومات من قاعدة المعرفة التونسية:
[food | score: 0.91] الهريسة (harissa): معجون الفلفل الأحمر... | مثال: ...
...
<</SYS>>

{user question} [/INST]
```

Context is truncated at **1 500 tokens** using TounsiLM's own tokenizer.

## Knowledge Base

### Entry types

| Type                          | File                  | Count    | Key extra fields                                    |
| ----------------------------- | --------------------- | -------- | --------------------------------------------------- |
| `expression`                  | expressions\*.json    | 83       | `origin`, `severity`, `gender_sensitive`            |
| `number_slang`                | expressions2.json     | 15       | `msa_equivalent`                                    |
| `proverb`                     | proverbs.json         | 1 276    | `literal_meaning`, `real_meaning`, `when_used`      |
| `food`                        | food.json             | 123      | `description`, `regional_variation`, `when_eaten`   |
| `ingredient`                  | food.json             | 26       | (same as food)                                      |
| `ritual`                      | rituals.json          | 26       | `occasion`, `tone`, `expected_response`             |
| `code_switch`                 | code-switching.json   | 23       | `origin_language`, `origin_word`, `domain`          |
| `tv_series` / `movie` / `film`| series_movies.json    | 20       | `era`, `cultural_significance`, `common_references` |
| `color`                       | colors.json           | 20       | `color_family`, `cultural_significance`             |
| **Total**                     |                       | **1 647**|                                                     |

### Validation

Every entry is validated against its Pydantic schema before indexing. All types share `BaseEntry` fields (`id`, `type`, `term_arabic`, `term_arabizi`, `meaning`, `example`, `usage_context`, `region`, `register`, `generation`, `scripts`, `source`, `last_updated`) and extend them with type-specific fields.

Run validation across all files:

```python
from pathlib import Path
from rag_kb.pipeline.validate_entries import validate_file, print_validation_report

for jf in Path("rag_kb/data").glob("*.json"):
    valid, errors = validate_file(str(jf))
    print_validation_report(valid, errors)
```

### Adding a new type

1. Create `rag_kb/schemas/my_type_schema.py` extending `BaseEntry`
2. Add the type to `SCHEMA_MAP` in `rag_kb/pipeline/validate_entries.py`
3. Add a field block to `build_embed_text.py`
4. Add routing keywords to `query_rewriter.py` if needed
5. Re-index: `python run_rag.py --index`

## Running on Kaggle

A ready-to-use notebook is included: **`tounsilm_rag_kaggle.ipynb`**

Upload it to Kaggle, then:

1. Set **Accelerator** → GPU T4, **Internet** → On
2. Add a secret named `HF_TOKEN` with your HuggingFace token
3. Run all cells in order

The notebook tests the query rewriter, BM25 vs semantic comparison, retrieval-only mode, and full RAG queries with confidence output.

## Hardware

| Setup              | VRAM needed |
| ------------------ | ----------- |
| CPU only (slow)    | —           |
| GPU full precision | ~16 GB      |

Kaggle T4 (16 GB) is the recommended free-tier environment.
