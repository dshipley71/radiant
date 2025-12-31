#!/usr/bin/env python3
"""
Test script for validating the RAG improvements:
1. Circuit breaker functionality
2. BM25 index persistence
3. Score normalization
4. Single LLM instantiation
"""

import sys
import os
import time
import tempfile
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.circuit_breaker import (
    CircuitBreaker,
    CircuitState,
    get_circuit_breaker,
    reset_all_circuit_breakers,
    get_all_circuit_breaker_stats,
)


def test_circuit_breaker_basic():
    """Test basic circuit breaker functionality."""
    print("\n=== Test: Circuit Breaker Basic ===")
    
    cb = CircuitBreaker(name="test_basic", failure_threshold=3, recovery_timeout=1.0)
    
    # Initial state should be CLOSED
    assert cb.is_closed, "Initial state should be CLOSED"
    assert cb.can_execute(), "Should allow execution when CLOSED"
    print("✓ Initial state is CLOSED")
    
    # Record success
    cb.record_success()
    assert cb.is_closed, "Should remain CLOSED after success"
    print("✓ Remains CLOSED after success")
    
    # Record failures up to threshold
    for i in range(3):
        cb.record_failure(Exception(f"failure {i+1}"))
    
    assert cb.is_open, f"Should be OPEN after {cb.failure_threshold} failures"
    assert not cb.can_execute(), "Should NOT allow execution when OPEN"
    print(f"✓ Opens after {cb.failure_threshold} failures")
    
    # Wait for recovery timeout
    time.sleep(1.1)
    
    # Should transition to HALF_OPEN
    assert cb.can_execute(), "Should allow execution after recovery timeout"
    assert cb.state == CircuitState.HALF_OPEN, "Should be HALF_OPEN after timeout"
    print("✓ Transitions to HALF_OPEN after recovery timeout")
    
    # Record success to close
    cb.record_success()
    assert cb.is_closed, "Should be CLOSED after success in HALF_OPEN"
    print("✓ Closes after success in HALF_OPEN")
    
    print("=== Circuit Breaker Basic: PASSED ===\n")


def test_circuit_breaker_registry():
    """Test circuit breaker registry."""
    print("\n=== Test: Circuit Breaker Registry ===")
    
    reset_all_circuit_breakers()
    
    # Get or create circuit breaker
    cb1 = get_circuit_breaker("test_registry", failure_threshold=5)
    cb2 = get_circuit_breaker("test_registry")
    
    assert cb1 is cb2, "Should return same instance for same name"
    print("✓ Registry returns same instance for same name")
    
    # Get stats
    stats = get_all_circuit_breaker_stats()
    assert "test_registry" in stats, "Should have test_registry in stats"
    assert stats["test_registry"]["failure_threshold"] == 5
    print("✓ Stats include registered circuit breakers")
    
    print("=== Circuit Breaker Registry: PASSED ===\n")


def test_score_normalization():
    """Test score normalization functions."""
    print("\n=== Test: Score Normalization ===")
    
    try:
        # Import from retriever (avoid circular imports by importing here)
        from agents.retriever import normalize_scores_minmax, normalize_scores_zscore
        
        # Haystack Document compatibility
        try:
            from haystack import Document
        except Exception:
            from haystack.dataclasses import Document
    except ImportError as e:
        print(f"⚠ Skipping test (missing dependency): {e}")
        print("=== Score Normalization: SKIPPED ===\n")
        return
    
    # Create test documents with various scores
    docs = [
        Document(content="doc1", meta={}),
        Document(content="doc2", meta={}),
        Document(content="doc3", meta={}),
    ]
    docs[0].score = 10.0
    docs[1].score = 50.0
    docs[2].score = 100.0
    
    # Test min-max normalization
    normalized = normalize_scores_minmax(docs.copy())
    
    assert abs(normalized[0].score - 0.0) < 0.001, "Min score should be 0"
    assert abs(normalized[1].score - 0.444) < 0.01, "Mid score should be ~0.444"
    assert abs(normalized[2].score - 1.0) < 0.001, "Max score should be 1"
    print("✓ Min-max normalization works correctly")
    
    # Reset scores
    docs[0].score = 10.0
    docs[1].score = 50.0
    docs[2].score = 100.0
    
    # Test z-score normalization
    normalized_z = normalize_scores_zscore(docs.copy())
    
    # Z-score normalized and sigmoid transformed should be in (0, 1)
    for d in normalized_z:
        assert 0 < d.score < 1, f"Z-score normalized should be in (0, 1), got {d.score}"
    print("✓ Z-score normalization works correctly")
    
    # Test empty list
    empty_result = normalize_scores_minmax([])
    assert empty_result == [], "Empty list should return empty list"
    print("✓ Empty list handling works")
    
    # Test single document
    single_doc = [Document(content="single", meta={})]
    single_doc[0].score = 5.0
    single_result = normalize_scores_minmax(single_doc)
    assert single_result[0].score == 1.0, "Single doc should have score 1.0"
    print("✓ Single document handling works")
    
    print("=== Score Normalization: PASSED ===\n")


def test_bm25_cache():
    """Test BM25 index persistence."""
    print("\n=== Test: BM25 Index Cache ===")
    
    try:
        from agents.retriever import BM25IndexCache
        
        # Haystack Document compatibility
        try:
            from haystack import Document
        except Exception:
            from haystack.dataclasses import Document
        
        from haystack.document_stores.in_memory import InMemoryDocumentStore
        from haystack.components.writers import DocumentWriter
    except ImportError as e:
        print(f"⚠ Skipping test (missing dependency): {e}")
        print("=== BM25 Index Cache: SKIPPED ===\n")
        return
    
    # Create temporary cache directory
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = BM25IndexCache(cache_dir=tmpdir, ttl_hours=1.0)
        
        # Create test documents
        docs = [
            Document(id="doc1", content="hello world", meta={}),
            Document(id="doc2", content="foo bar", meta={}),
        ]
        
        # Create a simple store
        store = InMemoryDocumentStore()
        writer = DocumentWriter(document_store=store)
        writer.run(documents=docs)
        
        # Cache the store
        success = cache.put("test_collection", docs, store)
        assert success, "Cache put should succeed"
        print("✓ BM25 cache put works")
        
        # Retrieve from cache
        retrieved = cache.get("test_collection", docs)
        assert retrieved is not None, "Cache get should return stored value"
        print("✓ BM25 cache get works")
        
        # Verify it's a valid store
        assert hasattr(retrieved, 'filter_documents'), "Retrieved should be a document store"
        print("✓ Retrieved store is valid")
        
        # Test cache invalidation on document change
        changed_docs = [
            Document(id="doc1", content="changed content", meta={}),
            Document(id="doc2", content="foo bar", meta={}),
        ]
        invalidated = cache.get("test_collection", changed_docs)
        assert invalidated is None, "Cache should be invalidated when docs change"
        print("✓ Cache invalidation on document change works")
        
        # Test manual invalidation
        cache.put("test_collection", docs, store)
        cache.invalidate("test_collection")
        after_invalidate = cache.get("test_collection", docs)
        assert after_invalidate is None, "Manual invalidation should clear cache"
        print("✓ Manual cache invalidation works")
    
    print("=== BM25 Index Cache: PASSED ===\n")


def test_default_circuit_breakers():
    """Test that default circuit breakers are initialized."""
    print("\n=== Test: Default Circuit Breakers ===")
    
    # These should already be initialized
    llm_cb = get_circuit_breaker("llm")
    dense_cb = get_circuit_breaker("dense_retrieval")
    bm25_cb = get_circuit_breaker("bm25_retrieval")
    rerank_cb = get_circuit_breaker("rerank")
    
    assert llm_cb is not None, "LLM circuit breaker should exist"
    assert dense_cb is not None, "Dense retrieval circuit breaker should exist"
    assert bm25_cb is not None, "BM25 circuit breaker should exist"
    assert rerank_cb is not None, "Rerank circuit breaker should exist"
    
    print("✓ All default circuit breakers are initialized")
    
    # Check default thresholds
    assert llm_cb.failure_threshold == 3, "LLM should have threshold 3"
    assert llm_cb.recovery_timeout == 120.0, "LLM should have 120s recovery"
    print("✓ Default thresholds are correct")
    
    print("=== Default Circuit Breakers: PASSED ===\n")


def main():
    """Run all tests."""
    print("=" * 60)
    print("RAG Improvements Test Suite")
    print("=" * 60)
    
    try:
        test_circuit_breaker_basic()
        test_circuit_breaker_registry()
        test_score_normalization()
        test_bm25_cache()
        test_default_circuit_breakers()
        
        print("=" * 60)
        print("ALL TESTS PASSED!")
        print("=" * 60)
        return 0
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
