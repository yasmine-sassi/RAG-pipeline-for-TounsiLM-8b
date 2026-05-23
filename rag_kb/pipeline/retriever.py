from pathlib import Path
from typing import Optional

import chromadb
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi

CHROMA_DIR = Path(__file__).parent.parent / "db" / "chroma_db"
COLLECTION_NAME = "tounsilm_kb"
EMBEDDING_MODEL = "intfloat/multilingual-e5-base"
_RRF_K = 60          # RRF constant — prevents top ranks from dominating
_SEMANTIC_W = 0.6    # weight for semantic results in RRF fusion
_BM25_W = 0.4        # weight for BM25 results in RRF fusion


class Retriever:
    def __init__(
        self,
        chroma_dir: Optional[Path] = None,
        embedding_model: str = EMBEDDING_MODEL,
        tokenizer=None,
    ):
        chroma_dir = chroma_dir or CHROMA_DIR

        if not chroma_dir.exists():
            raise FileNotFoundError(
                f"ChromaDB directory not found at {chroma_dir}. "
                "Run `python run_rag.py --index` first."
            )

        client = chromadb.PersistentClient(path=str(chroma_dir))
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model,
            device="cpu",   # keep embedding model on CPU so TounsiLM-8b has full VRAM
        )
        self._collection = client.get_collection(name=COLLECTION_NAME, embedding_function=ef)
        self.tokenizer = tokenizer
        self._build_bm25_index()

    def _build_bm25_index(self):
        data = self._collection.get(include=["documents", "metadatas"])
        self._bm25_ids = data["ids"]
        self._bm25_metadatas = data["metadatas"]
        # Strip "passage: " prefix — BM25 should index content, not the e5 prefix
        docs = [
            d[len("passage: "):] if d.startswith("passage: ") else d
            for d in data["documents"]
        ]
        self._bm25 = BM25Okapi([d.split() for d in docs])

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        entry_type: Optional[str] = None,
        min_score: float = 0.0,
    ) -> list:
        fetch_k = min(top_k * 3, self._collection.count())

        semantic_hits = self._semantic_retrieve(query, fetch_k, entry_type)
        bm25_hits = self._bm25_retrieve(query, fetch_k, entry_type)
        merged = self._rrf_merge(
            rankings=[semantic_hits, bm25_hits],
            weights=[_SEMANTIC_W, _BM25_W],
            top_k=top_k,
        )
        return [h for h in merged if h["score"] >= min_score]

    def format_context(
        self,
        hits: list,
        max_tokens: Optional[int] = None,
        max_chars: Optional[int] = None,
    ) -> str:
        if max_tokens is None and max_chars is None:
            max_tokens = 1500

        lines = []
        total_tokens = 0
        total_chars = 0

        for hit in hits:
            meta = hit["metadata"]
            score = hit.get("score", 0.0)
            line = (
                f"[{meta.get('type')} | score: {score}] "
                f"{meta.get('term_arabic')} ({meta.get('term_arabizi')})"
                f": {meta.get('meaning')}"
            )
            if meta.get("example"):
                line += f" | مثال: {meta['example']}"
            if meta.get("usage_context"):
                line += f" | الاستخدام: {meta['usage_context']}"

            if self.tokenizer and max_tokens is not None:
                line_tokens = len(self.tokenizer.encode(line))
                if total_tokens + line_tokens > max_tokens:
                    break
                total_tokens += line_tokens
            elif max_chars is not None:
                if total_chars + len(line) > max_chars:
                    break
                total_chars += len(line)

            lines.append(line)

        return "\n".join(lines)

    def count(self) -> int:
        return self._collection.count()

    def calculate_relevance_score(self, hits: list) -> dict:
        if not hits:
            return {
                "mean_score": 0.0,
                "max_score": 0.0,
                "min_score": 0.0,
                "num_results": 0,
                "confidence_level": "none",
            }

        scores = [h.get("score", 0.0) for h in hits]
        mean_score = sum(scores) / len(scores)

        if mean_score >= 0.75:
            confidence_level = "high"
        elif mean_score >= 0.50:
            confidence_level = "medium"
        else:
            confidence_level = "low"

        return {
            "mean_score": round(mean_score, 4),
            "max_score": round(max(scores), 4),
            "min_score": round(min(scores), 4),
            "num_results": len(hits),
            "confidence_level": confidence_level,
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _semantic_retrieve(self, query: str, k: int, entry_type: Optional[str]) -> list:
        where = {"type": {"$eq": entry_type}} if entry_type else None
        results = self._collection.query(
            query_texts=[f"query: {query}"],
            n_results=k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        hits = []
        for i in range(len(results["ids"][0])):
            score = round(1.0 - results["distances"][0][i], 4)
            hits.append({
                "id": results["ids"][0][i],
                "document": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "score": score,
            })
        return hits

    def _bm25_retrieve(self, query: str, k: int, entry_type: Optional[str]) -> list:
        scores = self._bm25.get_scores(query.split())

        pairs = []
        for idx, score in enumerate(scores):
            if entry_type and self._bm25_metadatas[idx].get("type") != entry_type:
                continue
            pairs.append((idx, float(score)))

        pairs.sort(key=lambda x: x[1], reverse=True)
        pairs = pairs[:k]

        if not pairs or pairs[0][1] == 0:
            return []

        max_score = pairs[0][1]
        return [
            {
                "id": self._bm25_ids[idx],
                "document": "",
                "metadata": self._bm25_metadatas[idx],
                "score": round(score / max_score, 4),
            }
            for idx, score in pairs
        ]

    def _rrf_merge(self, rankings: list, weights: list, top_k: int) -> list:
        """Reciprocal Rank Fusion with per-ranking weights."""
        rrf_scores: dict[str, float] = {}
        id_to_hit: dict[str, dict] = {}

        for ranking, weight in zip(rankings, weights):
            for rank, hit in enumerate(ranking, start=1):
                doc_id = hit["id"]
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + weight / (_RRF_K + rank)
                # Keep the hit with the highest original score for metadata
                if doc_id not in id_to_hit or hit["score"] > id_to_hit[doc_id]["score"]:
                    id_to_hit[doc_id] = hit

        max_rrf = max(rrf_scores.values()) if rrf_scores else 1.0
        ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        results = []
        for doc_id, rrf_score in ranked:
            hit = dict(id_to_hit[doc_id])
            hit["score"] = round(rrf_score / max_rrf, 4)
            results.append(hit)
        return results
