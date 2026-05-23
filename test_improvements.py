#!/usr/bin/env python3
"""
test_improvements.py

Demonstrates the new improvements:
1. Relevance scoring and confidence levels
2. Token-based context truncation
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from rag_kb.pipeline.rag_pipeline import RAGPipeline


def test_relevance_scoring():
    """Test relevance scoring with confidence levels."""
    print("=" * 70)
    print("TEST 1: Relevance Scoring & Confidence Levels")
    print("=" * 70)
    
    pipeline = RAGPipeline()
    
    # Test with a query that should have high confidence
    result = pipeline.query(
        "ما معنى برشا؟",
        top_k=5,
        show_sources=True,
    )
    
    print(f"\nQuestion: {result['question']}")
    print(f"Answer: {result['answer']}")
    print(f"\n✓ Confidence Level: {result['confidence_level'].upper()}")
    print(f"\nRelevance Metrics:")
    for key, val in result['relevance_metrics'].items():
        print(f"  {key}: {val}")
    
    print(f"\nContext (truncated to {result['relevance_metrics']['num_results']} results):")
    print("-" * 70)
    print(result['context'][:500] + "..." if len(result['context']) > 500 else result['context'])
    print()


def test_token_based_truncation():
    """Test token-based context truncation."""
    print("\n" + "=" * 70)
    print("TEST 2: Token-Based Context Truncation")
    print("=" * 70)
    
    pipeline = RAGPipeline()
    
    # Test with different token limits
    for max_tokens in [500, 1500, 3000]:
        result = pipeline.query(
            "شنو هي الهريسة؟",
            top_k=10,
            max_context_tokens=max_tokens,
            show_sources=False,
        )
        
        # Estimate tokens in context
        context_tokens = len(pipeline.llm.tokenizer.encode(result['context']))
        
        print(f"\nMax Context Tokens: {max_tokens}")
        print(f"Actual Context Tokens: {context_tokens}")
        print(f"Context Length: {len(result['context'])} chars")
        print(f"Confidence: {result['confidence_level'].upper()}")
        print(f"Answer Preview: {result['answer'][:100]}...")


def test_low_confidence_query():
    """Test a query that might have low confidence."""
    print("\n" + "=" * 70)
    print("TEST 3: Low Confidence Query Detection")
    print("=" * 70)
    
    pipeline = RAGPipeline()
    
    # Test with a very specific or unusual query
    result = pipeline.query(
        "شنو هي أغرب تقليد تونسي؟",
        top_k=3,
        show_sources=True,
    )
    
    print(f"\nQuestion: {result['question']}")
    print(f"Confidence Level: {result['confidence_level'].upper()}")
    print(f"Mean Relevance Score: {result['relevance_metrics']['mean_score']}")
    print(f"Number of Results: {result['relevance_metrics']['num_results']}")
    
    if result['relevance_metrics']['num_results'] < 3:
        print(f"\n⚠️  WARNING: Only {result['relevance_metrics']['num_results']} relevant results found.")
        print("Consider rephrasing the question or relaxing min_score threshold.")
    
    print(f"\nAnswer: {result['answer']}")


def test_retrieve_only_with_metrics():
    """Test retrieve_only with metrics."""
    print("\n" + "=" * 70)
    print("TEST 4: Retrieve-Only Mode with Metrics")
    print("=" * 70)
    
    pipeline = RAGPipeline()
    
    results = pipeline.retrieve_only(
        "ما معنى البرنوص؟",
        top_k=5,
    )
    
    print(f"\nQuestion: {results['question']}")
    print(f"Retrieved {results['relevance_metrics']['num_results']} results")
    print(f"Confidence Level: {results['relevance_metrics']['confidence_level'].upper()}")
    print(f"\nTop Hit Scores:")
    for i, hit in enumerate(results['hits'][:3], 1):
        print(f"  {i}. [{hit['metadata']['type']}] {hit['metadata']['term_arabic']}: {hit['score']}")


if __name__ == "__main__":
    print("\n🚀 Testing RAG Pipeline Improvements\n")
    
    try:
        # Uncomment individual tests as needed:
        test_relevance_scoring()
        test_token_based_truncation()
        test_low_confidence_query()
        test_retrieve_only_with_metrics()
        
        print("\n✅ All tests completed successfully!")
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("Make sure to build the index first: python run_rag.py --index")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
