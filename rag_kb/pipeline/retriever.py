from pathlib import Path
from typing import Optional

import chromadb
from chromadb.utils import embedding_functions

CHROMA_DIR = Path(__file__).parent.parent / "db" / "chroma_db"
COLLECTION_NAME = "tounsilm_kb"
EMBEDDING_MODEL = "intfloat/multilingual-e5-base"


class Retriever:
    def __init__(
        self,
        chroma_dir: Optional[Path] = None,
        embedding_model: str = EMBEDDING_MODEL,
    ):
        chroma_dir = chroma_dir or CHROMA_DIR

        if not chroma_dir.exists():
            raise FileNotFoundError(
                f"ChromaDB directory not found at {chroma_dir}. "
                "Run `python run_rag.py --index` first."
            )

        client = chromadb.PersistentClient(path=str(chroma_dir))
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model
        )
        self._collection = client.get_collection(name=COLLECTION_NAME, embedding_function=ef)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        entry_type: Optional[str] = None,
        min_score: float = 0.0,
    ) -> list:
        where = {"type": {"$eq": entry_type}} if entry_type else None

        # multilingual-e5 models require a "query: " prefix at retrieval time
        results = self._collection.query(
            query_texts=[f"query: {query}"],
            n_results=min(top_k, self._collection.count()),
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        hits = []
        for i in range(len(results["ids"][0])):
            score = round(1.0 - results["distances"][0][i], 4)
            if score >= min_score:
                hits.append({
                    "id": results["ids"][0][i],
                    "document": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "score": score,
                })
        return hits

    def format_context(self, hits: list, max_chars: int = 2000) -> str:
        lines = []
        total = 0
        for hit in hits:
            meta = hit["metadata"]
            line = (
                f"[{meta.get('type')}] {meta.get('term_arabic')} ({meta.get('term_arabizi')})"
                f": {meta.get('meaning')}"
            )
            if meta.get("example"):
                line += f" | مثال: {meta['example']}"
            if meta.get("usage_context"):
                line += f" | الاستخدام: {meta['usage_context']}"
            total += len(line)
            if total > max_chars:
                break
            lines.append(line)
        return "\n".join(lines)

    def count(self) -> int:
        return self._collection.count()
