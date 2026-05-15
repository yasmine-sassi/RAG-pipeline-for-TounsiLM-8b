from .build_embed_text import build_embed_text
from .validate_entries import validate_file, print_validation_report

# Heavy imports (torch, transformers, chromadb) are NOT loaded here.
# Import them directly when needed:
#   from rag_kb.pipeline.indexer import build_index
#   from rag_kb.pipeline.retriever import Retriever
#   from rag_kb.pipeline.llm_interface import TounsiLM
#   from rag_kb.pipeline.rag_pipeline import RAGPipeline

__all__ = [
    "build_embed_text",
    "validate_file",
    "print_validation_report",
]
