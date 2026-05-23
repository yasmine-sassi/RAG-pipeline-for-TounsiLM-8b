#!/usr/bin/env python3
"""
run_rag.py — TounsiLM RAG Pipeline CLI

Commands:
  --index              Build / update the ChromaDB vector index
  --index --reset      Wipe and rebuild the index from scratch
  --query "سؤال"       Run a single RAG query (retrieve + generate)
  --retrieve "سؤال"    Retrieve without LLM (test retrieval quality)
  (no args)            Launch interactive chat mode

Options:
  --top-k N            Number of KB entries to retrieve (default: 5)
  --type TYPE          Filter retrieval by entry type (expression, proverb, food…)
  --min-score F        Minimum similarity score 0–1 (default: 0.0)
  --max-tokens N       Max tokens to generate (default: 512)
  --temperature F      Sampling temperature (default: 0.7)
  --max-context-tokens N   Max tokens for RAG context (default: 1500)
  --embedding-model M  HuggingFace embedding model (default: intfloat/multilingual-e5-base)

Examples:
  python run_rag.py --index
  python run_rag.py --retrieve "ما معنى برشا؟" --top-k 3
  python run_rag.py --query "شنو هي الهريسة التونسية؟"
  python run_rag.py --load-in-4bit
"""

import argparse
import sys
from pathlib import Path

# Ensure the RAG root is on the path so imports resolve correctly
sys.path.insert(0, str(Path(__file__).parent))


# ──────────────────────────────────────────────────────────────────────────────
# Subcommand handlers
# ──────────────────────────────────────────────────────────────────────────────

def cmd_index(args):
    from rag_kb.pipeline.indexer import build_index

    print("Building ChromaDB index…")
    collection = build_index(
        embedding_model=args.embedding_model,
        reset=args.reset,
    )
    print(f"\nIndex ready — {collection.count()} entries in ChromaDB.")


def cmd_retrieve(args):
    """Retrieve without generating with relevance metrics."""
    from rag_kb.pipeline.rag_pipeline import RAGPipeline

    pipeline = _build_pipeline(args)
    result = pipeline.retrieve_only(
        question=args.retrieve,
        top_k=args.top_k,
        entry_type=args.type,
        min_score=args.min_score,
    )

    metrics = result["relevance_metrics"]
    conf_icon = {"high": "🟢", "medium": "🟡", "low": "🔴", "none": "⚫"}.get(metrics["confidence_level"], "?")
    
    print(f"\nRetrieve results for: '{result['question']}'")
    print(f"Confidence: {conf_icon} {metrics['confidence_level'].upper()}")
    print(f"  Found: {metrics['num_results']} results")
    print(f"  Scores: mean={metrics['mean_score']:.4f}, "
          f"range=[{metrics['min_score']:.4f}–{metrics['max_score']:.4f}]")
    print("─" * 60)
    
    if not result["hits"]:
        print("No results found.")
        return
    
    for i, hit in enumerate(result["hits"], 1):
        meta = hit["metadata"]
        print(
            f"{i}. [{meta.get('type')}] {meta.get('term_arabic')} "
            f"({meta.get('term_arabizi')})  score={hit['score']:.4f}"
        )
        print(f"   Meaning   : {meta.get('meaning', '')[:120]}")
        print(f"   Example   : {meta.get('example', '')[:100]}")
        print()


def cmd_query(args):
    from rag_kb.pipeline.rag_pipeline import RAGPipeline

    pipeline = _build_pipeline(args)
    result = pipeline.query(
        question=args.query,
        top_k=args.top_k,
        entry_type=args.type,
        min_score=args.min_score,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        show_sources=True,
        max_context_tokens=args.max_context_tokens,
    )
    _print_result(result)


def cmd_interactive(args):
    from rag_kb.pipeline.rag_pipeline import RAGPipeline

    pipeline = _build_pipeline(args)
    print("\nTounsiLM RAG — interactive mode")
    print("Type your question in Arabic, Arabizi, or French.")
    print("Commands: :exit  :sources on/off  :top-k N  :type TYPE")
    print("─" * 60)

    show_sources = True
    top_k = args.top_k
    entry_type = args.type

    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not question:
            continue

        # Simple in-session commands
        if question.startswith(":"):
            parts = question.split()
            cmd = parts[0]
            if cmd in (":exit", ":quit"):
                print("Goodbye!")
                break
            elif cmd == ":sources":
                show_sources = len(parts) < 2 or parts[1].lower() not in ("off", "false", "0")
                print(f"Sources display: {'on' if show_sources else 'off'}")
            elif cmd == ":top-k" and len(parts) == 2:
                try:
                    top_k = int(parts[1])
                    print(f"top_k set to {top_k}")
                except ValueError:
                    print("Usage: :top-k N")
            elif cmd == ":type":
                entry_type = parts[1] if len(parts) > 1 else None
                print(f"entry_type set to {entry_type!r}")
            else:
                print("Unknown command. Available: :exit :sources on/off :top-k N :type TYPE")
            continue

        result = pipeline.query(
            question=question,
            top_k=top_k,
            entry_type=entry_type,
            min_score=args.min_score,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            show_sources=show_sources,
            max_context_tokens=args.max_context_tokens,
        )
        _print_result(result, verbose=show_sources)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _build_pipeline(args):
    from rag_kb.pipeline.rag_pipeline import RAGPipeline

    return RAGPipeline(
        embedding_model=args.embedding_model,
        top_k=args.top_k,
    )


def _print_result(result: dict, verbose: bool = True):
    print("\n" + "═" * 60)
    print(f"Question : {result['question']}")
    
    # Display confidence level and relevance metrics
    if "confidence_level" in result:
        metrics = result["relevance_metrics"]
        conf_icon = {"high": "🟢", "medium": "🟡", "low": "🔴", "none": "⚫"}.get(metrics["confidence_level"], "?")
        print(f"Confidence: {conf_icon} {metrics['confidence_level'].upper()}")
        print(f"  Relevance: mean={metrics['mean_score']:.4f}, "
              f"range=[{metrics['min_score']:.4f}–{metrics['max_score']:.4f}], "
              f"results={metrics['num_results']}")
    
    print("─" * 60)
    print(f"Answer   :\n{result['answer']}")
    
    if verbose and result.get("sources"):
        print("\nSources:")
        for hit in result["sources"]:
            meta = hit["metadata"]
            print(
                f"  [{hit['id']}] {meta.get('term_arabic')} "
                f"({meta.get('term_arabizi')})  score={hit['score']:.3f}"
            )
    print("═" * 60)


# ──────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="TounsiLM RAG Pipeline for Tunisian Arabic",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Subcommand flags
    parser.add_argument("--index", action="store_true", help="Build/update ChromaDB index")
    parser.add_argument("--reset", action="store_true", help="Reset index before rebuilding (with --index)")
    parser.add_argument("--query", "-q", metavar="QUESTION", help="Run a RAG query")
    parser.add_argument("--retrieve", "-r", metavar="QUESTION", help="Retrieve without LLM")

    # Retrieval options
    parser.add_argument("--top-k", type=int, default=5, metavar="N", help="Entries to retrieve (default: 5)")
    parser.add_argument("--type", metavar="TYPE", help="Filter by entry type (expression, proverb, food, ritual…)")
    parser.add_argument("--min-score", type=float, default=0.0, metavar="F", help="Min similarity score 0–1")

    # Generation options
    parser.add_argument("--max-tokens", type=int, default=512, metavar="N", help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, metavar="F", help="Sampling temperature")
    parser.add_argument(
        "--max-context-tokens",
        type=int,
        default=1500,
        metavar="N",
        help="Max tokens for RAG context (default: 1500)",
    )

    # Model options
    parser.add_argument(
        "--embedding-model",
        default="intfloat/multilingual-e5-base",
        metavar="MODEL",
        help="Sentence-transformer model for embeddings",
    )

    args = parser.parse_args()

    if args.index:
        cmd_index(args)
    elif args.retrieve:
        cmd_retrieve(args)
    elif args.query:
        cmd_query(args)
    else:
        cmd_interactive(args)


if __name__ == "__main__":
    main()
