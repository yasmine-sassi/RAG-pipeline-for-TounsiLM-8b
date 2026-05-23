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
            model_name=embedding_model
        )
        self._collection = client.get_collection(name=COLLECTION_NAME, embedding_function=ef)
        self.tokenizer = tokenizer

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

    def format_context(
        self,
        hits: list,
        max_tokens: Optional[int] = None,
        max_chars: Optional[int] = None,
    ) -> str:
        """
        Format context from hits with relevance scores.
        Uses token-based truncation if tokenizer is available and max_tokens is set,
        otherwise falls back to character-based truncation.
        """
        if max_tokens is None and max_chars is None:
            max_tokens = 1500  # Default token limit
        
        lines = []
        total_tokens = 0
        total_chars = 0
        
        for hit in hits:
            meta = hit["metadata"]
            score = hit.get("score", 0.0)
            
            # Build line with relevance score indicator
            line = (
                f"[{meta.get('type')} | score: {score}] {meta.get('term_arabic')} ({meta.get('term_arabizi')})"
                f": {meta.get('meaning')}"
            )
            if meta.get("example"):
                line += f" | مثال: {meta['example']}"
            if meta.get("usage_context"):
                line += f" | الاستخدام: {meta['usage_context']}"
            
            # Check token-based limit if tokenizer available
            if self.tokenizer and max_tokens is not None:
                line_tokens = len(self.tokenizer.encode(line))
                if total_tokens + line_tokens > max_tokens:
                    break
                total_tokens += line_tokens
            # Otherwise check char-based limit
            elif max_chars is not None:
                if total_chars + len(line) > max_chars:
                    break
                total_chars += len(line)
            
            lines.append(line)
        
        return "\n".join(lines)

    def count(self) -> int:
        return self._collection.count()
    
    def calculate_relevance_score(self, hits: list) -> dict:
        """
        Calculate overall relevance metrics from retrieved hits.
        Returns a dict with mean_score, max_score, min_score, and confidence_level.
        """
        if not hits:
            return {
                "mean_score": 0.0,
                "max_score": 0.0,
                "min_score": 0.0,
                "num_results": 0,
                "confidence_level": "none",
            }
        
        scores = [hit.get("score", 0.0) for hit in hits]
        mean_score = sum(scores) / len(scores)
        
        # Confidence levels based on mean relevance score
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
