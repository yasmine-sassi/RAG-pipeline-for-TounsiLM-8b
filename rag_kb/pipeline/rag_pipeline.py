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
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
    ):
        self.top_k = top_k

        print("Initializing retriever...")
        self.retriever = Retriever(chroma_dir=chroma_dir, embedding_model=embedding_model)
        print(f"Knowledge base: {self.retriever.count()} indexed entries\n")

        print("Initializing TounsiLM-8b...")
        self.llm = TounsiLM(
            model_id=model_id,
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
        )

    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
        entry_type: Optional[str] = None,
        min_score: float = 0.0,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        show_sources: bool = True,
    ) -> dict:
        hits = self.retriever.retrieve(
            query=question,
            top_k=top_k if top_k is not None else self.top_k,
            entry_type=entry_type,
            min_score=min_score,
        )
        context = self.retriever.format_context(hits)
        prompt = self.llm.build_prompt(question, context)
        answer = self.llm.generate(prompt=prompt, max_new_tokens=max_new_tokens, temperature=temperature)

        result = {"question": question, "answer": answer}
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
    ) -> list:
        return self.retriever.retrieve(
            query=question,
            top_k=top_k if top_k is not None else self.top_k,
            entry_type=entry_type,
            min_score=min_score,
        )
