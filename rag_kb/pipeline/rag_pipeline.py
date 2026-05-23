from pathlib import Path
from typing import Optional

from rag_kb.pipeline.retriever import Retriever
from rag_kb.pipeline.llm_interface import TounsiLM
from rag_kb.pipeline.query_rewriter import QueryRewriter


class RAGPipeline:
    def __init__(
        self,
        chroma_dir: Optional[Path] = None,
        model_id: str = "alabenayed/TounsiLM-8b",
        embedding_model: str = "intfloat/multilingual-e5-base",
        top_k: int = 5,
    ):
        self.top_k = top_k
        self.rewriter = QueryRewriter()

        print("Initializing TounsiLM-8b...")
        self.llm = TounsiLM(model_id=model_id)

        print("Initializing retriever...")
        self.retriever = Retriever(
            chroma_dir=chroma_dir,
            embedding_model=embedding_model,
            tokenizer=self.llm.tokenizer,
        )
        print(f"Knowledge base: {self.retriever.count()} indexed entries\n")

    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
        entry_type: Optional[str] = None,
        min_score: float = 0.0,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        show_sources: bool = True,
        max_context_tokens: int = 1500,
    ) -> dict:
        effective_k = top_k if top_k is not None else self.top_k

        # Auto-route to entry type if not explicitly set
        if entry_type is None:
            entry_type = self.rewriter.detect_type(question)

        # Retrieve across all query variants, keep best score per document
        hits = self._multi_query_retrieve(question, effective_k, entry_type, min_score)

        relevance_metrics = self.retriever.calculate_relevance_score(hits)

        # Only inject context when retrieval is confident enough to be useful.
        # If no type was routed and mean score is below threshold, the KB likely
        # returned keyword noise (e.g. proverbs matching a word in an off-domain query).
        use_context = relevance_metrics["mean_score"] >= 0.60 or entry_type is not None
        context = self.retriever.format_context(hits, max_tokens=max_context_tokens) if use_context else ""
        prompt = self.llm.build_prompt(question, context)
        answer = self.llm.generate(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

        result = {
            "question": question,
            "answer": answer,
            "confidence_level": relevance_metrics["confidence_level"],
            "relevance_metrics": relevance_metrics,
        }
        if show_sources:
            result["context"] = context
            result["sources"] = hits
        return result

    def retrieve_only(
        self,
        question: str,
        top_k: Optional[int] = None,
        entry_type: Optional[str] = None,
        min_score: float = 0.0,
    ) -> dict:
        effective_k = top_k if top_k is not None else self.top_k

        if entry_type is None:
            entry_type = self.rewriter.detect_type(question)

        hits = self._multi_query_retrieve(question, effective_k, entry_type, min_score)
        relevance_metrics = self.retriever.calculate_relevance_score(hits)

        return {
            "question": question,
            "hits": hits,
            "relevance_metrics": relevance_metrics,
            "routed_type": entry_type,
        }

    # ──────────────────────────────────────────────────────────────────────────

    def _multi_query_retrieve(
        self,
        question: str,
        top_k: int,
        entry_type: Optional[str],
        min_score: float,
    ) -> list:
        variants = self.rewriter.rewrite(question)
        best: dict[str, dict] = {}

        for variant in variants:
            hits = self.retriever.retrieve(
                query=variant,
                top_k=top_k,
                entry_type=entry_type,
                min_score=min_score,
            )
            for hit in hits:
                doc_id = hit["id"]
                if doc_id not in best or hit["score"] > best[doc_id]["score"]:
                    best[doc_id] = hit

        return sorted(best.values(), key=lambda h: h["score"], reverse=True)[:top_k]
