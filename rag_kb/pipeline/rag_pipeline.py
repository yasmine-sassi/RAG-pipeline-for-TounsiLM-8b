from pathlib import Path
from typing import Optional

from rag_kb.pipeline.retriever import Retriever
from rag_kb.pipeline.llm_interface import TounsiLM


class RAGPipeline:
    def __init__(
        self,
        chroma_dir: Optional[Path] = None,
        model_id: str = "alabenayed/TounsiLM-8b",
        embedding_model: str = "intfloat/multilingual-e5-base",
        top_k: int = 5,
    ):
        self.top_k = top_k

        print("Initializing TounsiLM-8b...")
        self.llm = TounsiLM(model_id=model_id)

        print("Initializing retriever...")
        self.retriever = Retriever(
            chroma_dir=chroma_dir,
            embedding_model=embedding_model,
            tokenizer=self.llm.tokenizer,  # Pass tokenizer for token-based truncation
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
        """
        Query the RAG pipeline with confidence scoring and token-based context truncation.
        
        Args:
            question: User query
            top_k: Number of results to retrieve
            entry_type: Filter by entry type
            min_score: Minimum relevance score threshold
            max_new_tokens: Max tokens for LLM response
            temperature: LLM temperature
            show_sources: Include source context and hits
            max_context_tokens: Max tokens for context (uses tokenizer-based truncation)
        
        Returns:
            dict with question, answer, confidence_level, and optionally context and sources
        """
        hits = self.retriever.retrieve(
            query=question,
            top_k=top_k if top_k is not None else self.top_k,
            entry_type=entry_type,
            min_score=min_score,
        )
        
        # Calculate relevance metrics
        relevance_metrics = self.retriever.calculate_relevance_score(hits)
        
        # Format context with token-based truncation
        context = self.retriever.format_context(
            hits,
            max_tokens=max_context_tokens,
        )
        
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
        """
        Retrieve only (no generation) with relevance metrics.
        Useful for debugging retrieval quality.
        """
        hits = self.retriever.retrieve(
            query=question,
            top_k=top_k if top_k is not None else self.top_k,
            entry_type=entry_type,
            min_score=min_score,
        )
        
        relevance_metrics = self.retriever.calculate_relevance_score(hits)
        
        return {
            "question": question,
            "hits": hits,
            "relevance_metrics": relevance_metrics,
        }
