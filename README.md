# RAG Pipeline for TounsiLM-8b

A Retrieval-Augmented Generation (RAG) system built on a Tunisian Arabic knowledge base, designed to improve [`alabenayed/TounsiLM-8b`](https://huggingface.co/alabenayed/TounsiLM-8b).

## Structure

```
RAG/
├── run_rag.py                     # CLI entry point
├── requirements.txt               # Dependencies
└── rag_kb/
    ├── data/                      # JSON knowledge base files
    │   ├── expressions.json       # Tunisian expressions
    │   ├── expressions2.json      # Number slang & extra expressions
    │   ├── expressions3.json      # Additional expressions
    │   ├── proverbs.json          # Tunisian proverbs
    │   ├── food.json              # Tunisian dishes & ingredients
    │   ├── rituals.json           # Social rituals & greetings
    │   ├── code-switching.json    # French-Tunisian code-switching
    │   └── series_movies.json     # Tunisian TV & cultural references
    │
    ├── schemas/                   # Pydantic validation schemas
    │   ├── base_schema.py
    │   ├── expression_schema.py
    │   └── proverb_schema.py
    │
    ├── pipeline/                  # Core RAG logic
    │   ├── build_embed_text.py    # Builds embedding text for all entry types
    │   ├── validate_entries.py    # JSON validation
    │   ├── indexer.py             # Loads data → ChromaDB
    │   ├── retriever.py           # Semantic search over ChromaDB
    │   ├── llm_interface.py       # TounsiLM-8b wrapper
    │   └── rag_pipeline.py        # Orchestrates retrieve → generate
    │
    ├── scripts/
    │   └── bulk_import.py         # Bulk import from JSON/CSV
    │
    └── db/
        └── chroma_db/             # ChromaDB vector store (auto-created)
```

## Setup

```bash
pip install -r requirements.txt
```

**Dependencies:** `chromadb`, `sentence-transformers`, `transformers`, `torch`, `accelerate`, `pydantic`

## Quick Start

### 1. Build the vector index

Embeds all knowledge base entries and stores them in ChromaDB (run once):

```bash
python run_rag.py --index
```

To rebuild from scratch:

```bash
python run_rag.py --index --reset
```

### 2. Test retrieval (no LLM)

Validate that the knowledge base returns relevant results before loading the model:

```bash
python run_rag.py --retrieve "ما معنى برشا؟" --top-k 3
python run_rag.py --retrieve "harissa" --top-k 5 --type food
```

### 3. Run a RAG query

Retrieves relevant entries and feeds them as context to TounsiLM-8b:

```bash
python run_rag.py --query "شنو هي الهريسة التونسية؟"
python run_rag.py --query "ما معنى اللي فات مات؟" --type proverb
```

### 4. Interactive chat

```bash
python run_rag.py                  # full precision (CPU/GPU)
python run_rag.py --load-in-4bit   # 4-bit quantization (GPU, needs bitsandbytes)
python run_rag.py --load-in-8bit   # 8-bit quantization (GPU, needs bitsandbytes)
```

In-session commands:

| Command | Effect |
|---|---|
| `:exit` | Quit |
| `:sources on` / `:sources off` | Toggle source display |
| `:top-k 3` | Change retrieval count |
| `:type proverb` | Filter by entry type |

## CLI Reference

```
python run_rag.py [OPTIONS]

Subcommands:
  --index              Build / update ChromaDB index
  --index --reset      Wipe and rebuild from scratch
  --query "..."        Run a RAG query
  --retrieve "..."     Retrieve without LLM

Retrieval:
  --top-k N            Entries to retrieve (default: 5)
  --type TYPE          Filter by type: expression, proverb, food, ritual,
                       code_switch, tv_series, number_slang
  --min-score F        Minimum similarity score 0–1 (default: 0.0)

Generation:
  --max-tokens N       Max tokens to generate (default: 512)
  --temperature F      Sampling temperature (default: 0.7)

Models:
  --load-in-4bit       4-bit quantization (Linux + GPU + bitsandbytes)
  --load-in-8bit       8-bit quantization (Linux + GPU + bitsandbytes)
  --embedding-model M  Sentence-transformer model
                       (default: intfloat/multilingual-e5-base)
```

## Architecture

### Embedding model

`intfloat/multilingual-e5-base` is used for both indexing and retrieval. It handles Arabic script, Arabizi (Arabic in Latin letters), and French in a unified embedding space — essential for a knowledge base that mixes all three.

### Vector store

ChromaDB with cosine similarity, persisted to `rag_kb/db/chroma_db/`. The index survives restarts; only new entries need to be re-indexed.

### Retrieval

Queries are prefixed with `"query: "` as required by E5 models before being embedded. Top-k results are returned with cosine similarity scores.

### Prompt injection

Retrieved entries are formatted and injected into TounsiLM-8b's system prompt so the model answers with grounded Tunisian cultural context rather than relying solely on parametric knowledge.

```
[INST] <<SYS>>
أنت مساعد ذكي متخصص في اللهجة التونسية...

معلومات من قاعدة المعرفة:
[expression] برشا (barcha): كثير | مثال: عندي برشا خدمة اليوم
...
<</SYS>>

{user question} [/INST]
```

### Entry types

| Type | File(s) | Key fields |
|---|---|---|
| `expression` | expressions*.json | term, meaning, origin, severity |
| `proverb` | proverbs.json | literal_meaning, real_meaning, when_used |
| `food` | food.json | description, regional_variation, when_eaten |
| `ritual` | rituals.json | occasion, tone, expected_response |
| `code_switch` | code-switching.json | origin_language, origin_word, domain |
| `tv_series` | series_movies.json | cultural_significance, common_references |
| `number_slang` | expressions2.json | msa_equivalent |

## Schemas

### BaseEntry (all types)

```python
id: str
type: str
term_arabic: str
term_arabizi: str
meaning: str
meaning_fr: Optional[str]
example: str
usage_context: str
region: str          # "national" or region name
register: str        # "formal" | "informal"
generation: str      # "all" | "youth" | …
scripts: List[str]   # ["arabic", "arabizi"]
source: str
last_updated: str
```

### ExpressionEntry

```python
origin: Optional[str]
severity: str        # "neutral" | "offensive" | …
gender_sensitive: bool
```

### ProverbEntry

```python
literal_meaning: str
real_meaning: str
when_used: str
msa_equivalent: Optional[str]
```

## Adding Data

### Validate a file

```python
from rag_kb.pipeline import validate_file, print_validation_report

valid, errors = validate_file("rag_kb/data/expressions.json")
print_validation_report(valid, errors)
```

### Bulk import from JSON

```python
from rag_kb.scripts.bulk_import import bulk_import_from_json

imported, failed = bulk_import_from_json("new_entries.json", "expression")
```

After adding data, re-run indexing to pick up new entries:

```bash
python run_rag.py --index
```

## Hardware Notes

| Setup | Command | VRAM needed |
|---|---|---|
| CPU only | `python run_rag.py` | — |
| GPU full precision | `python run_rag.py` | ~16 GB |
| GPU 8-bit | `python run_rag.py --load-in-8bit` | ~10 GB |
| GPU 4-bit | `python run_rag.py --load-in-4bit` | ~6 GB |

`bitsandbytes` quantization requires Linux with a CUDA GPU. On Windows the pipeline falls back to full precision automatically.
