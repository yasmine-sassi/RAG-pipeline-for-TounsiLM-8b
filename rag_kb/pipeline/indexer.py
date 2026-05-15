import json
import sys
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.utils import embedding_functions

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from rag_kb.pipeline.build_embed_text import build_embed_text

DATA_DIR = Path(__file__).parent.parent / "data"
CHROMA_DIR = Path(__file__).parent.parent / "db" / "chroma_db"
COLLECTION_NAME = "tounsilm_kb"
EMBEDDING_MODEL = "intfloat/multilingual-e5-base"
BATCH_SIZE = 100


def load_all_data(data_dir: Path) -> list:
    entries = []
    for json_file in sorted(data_dir.glob("*.json")):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                entries.extend(data)
                print(f"  Loaded {len(data):>4} entries from {json_file.name}")
        except Exception as e:
            print(f"  Warning: could not load {json_file.name}: {e}")
    return entries


def build_index(
    data_dir: Optional[Path] = None,
    chroma_dir: Optional[Path] = None,
    embedding_model: str = EMBEDDING_MODEL,
    reset: bool = False,
) -> "chromadb.Collection":
    data_dir = data_dir or DATA_DIR
    chroma_dir = chroma_dir or CHROMA_DIR
    chroma_dir.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(chroma_dir))
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=embedding_model)

    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
            print(f"Deleted existing collection '{COLLECTION_NAME}'")
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )

    print(f"\nLoading data from {data_dir}...")
    entries = load_all_data(data_dir)
    print(f"Total entries loaded: {len(entries)}")

    existing_ids: set = set()
    try:
        existing_ids = set(collection.get(include=[])["ids"])
        if existing_ids:
            print(f"Skipping {len(existing_ids)} already-indexed entries")
    except Exception:
        pass

    ids, texts, metadatas = [], [], []

    for entry in entries:
        entry_id = entry.get("id", "")
        if not entry_id or entry_id in existing_ids:
            continue

        embed_text = build_embed_text(entry)
        if not embed_text.strip():
            continue

        # ChromaDB only accepts str/int/float/bool metadata values
        meta = {
            "type": str(entry.get("type", "unknown")),
            "term_arabic": str(entry.get("term_arabic", "")),
            "term_arabizi": str(entry.get("term_arabizi", "")),
            "meaning": str(entry.get("meaning", ""))[:500],
            "meaning_fr": str(entry.get("meaning_fr") or ""),
            "example": str(entry.get("example", ""))[:300],
            "usage_context": str(entry.get("usage_context", ""))[:300],
            "region": str(entry.get("region", "")),
            "register": str(entry.get("register", "")),
            "generation": str(entry.get("generation", "")),
            "source": str(entry.get("source", "")),
        }

        ids.append(entry_id)
        texts.append(embed_text)
        metadatas.append(meta)

    if ids:
        print(f"\nIndexing {len(ids)} new entries in batches of {BATCH_SIZE}...")
        for i in range(0, len(ids), BATCH_SIZE):
            batch_end = min(i + BATCH_SIZE, len(ids))
            collection.add(
                ids=ids[i:batch_end],
                documents=texts[i:batch_end],
                metadatas=metadatas[i:batch_end],
            )
            print(f"  Batch {i // BATCH_SIZE + 1}: entries {i + 1}–{batch_end}")
        print(f"\nDone. Total entries in index: {collection.count()}")
    else:
        print(f"\nNo new entries to index. Collection has {collection.count()} entries.")

    return collection
