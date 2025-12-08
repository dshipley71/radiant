# HybridRetrievalAgent Documentation

## Technical Reference for the Radiant RAG Pipeline Retrieval System

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture Context](#architecture-context)
3. [Class Structure](#class-structure)
4. [Core Components](#core-components)
5. [Configuration System](#configuration-system)
6. [Retrieval Modes](#retrieval-modes)
7. [Caching System](#caching-system)
8. [Data Flow](#data-flow)
9. [Testing Strategies](#testing-strategies)
10. [Recommendations and Improvements](#recommendations-and-improvements)
11. [Usage Examples](#usage-examples)

---

## Overview

### Purpose

The `HybridRetrievalAgent` is the primary document retrieval component within the Radiant RAG pipeline. It implements a sophisticated multi-strategy retrieval system combining dense vector search, lexical BM25 search, and hierarchical document merging to maximize recall and precision.

### Key Responsibilities

- Dense vector retrieval from ChromaDB (leaf and parent indices)
- Hybrid retrieval with BM25 lexical fusion
- Hierarchical auto-merging of leaf chunks into parent documents
- Query expansion and PRF-augmented query handling
- Result caching for performance optimization
- Parent metadata enrichment via sidecar files
- Grouping results by parent document

### Design Philosophy

The agent implements a **multi-signal fusion** approach, combining multiple retrieval strategies to maximize recall while maintaining precision. It supports both hierarchical (dual-index) and flat (leaf-only) retrieval modes, with intelligent fallbacks and caching.

---

## Architecture Context

### Position in the Radiant Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    RetrieverInput                               │
│  query | expanded_queries | prf_augmented_query | plan          │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                   HybridRetrievalAgent                          │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  1. Cache Check                                         │   │
│  │     └─ Return cached results if available               │   │
│  ├─────────────────────────────────────────────────────────┤   │
│  │  2. Query Preparation                                   │   │
│  │     └─ Normalize, deduplicate, expand queries           │   │
│  ├─────────────────────────────────────────────────────────┤   │
│  │  3. Dense Retrieval                                     │   │
│  │     ├─ Leaf ChromaDB (always)                          │   │
│  │     └─ Parent ChromaDB (dual-index mode)               │   │
│  ├─────────────────────────────────────────────────────────┤   │
│  │  4. Lexical Retrieval (if hybrid enabled)              │   │
│  │     ├─ BM25 over enriched documents                    │   │
│  │     └─ Sidecar lexical boost fallback                  │   │
│  ├─────────────────────────────────────────────────────────┤   │
│  │  5. Auto-Merge (dual-index only)                       │   │
│  │     └─ Merge leaf chunks into parents via threshold    │   │
│  ├─────────────────────────────────────────────────────────┤   │
│  │  6. Grouping & Output                                  │   │
│  │     └─ Group by parent ID, build snippets              │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    RetrieverOutput                              │
│  results: List[RetrievalResult]                                 │
│    └─ doc_id | parent_metadata | snippets[]                     │
└─────────────────────────────────────────────────────────────────┘
```

### Component Relationships

```
                    ┌─────────────────────┐
                    │ HybridRetrievalAgent │
                    └──────────┬──────────┘
                               │
        ┌──────────────────────┼──────────────────────┐
        │                      │                      │
        ▼                      ▼                      ▼
┌───────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  QueryCache   │    │ AutoMergeAgent  │    │ RetrieverConfig │
│  (LRU cache)  │    │ (chunk merger)  │    │  (YAML config)  │
└───────────────┘    └────────┬────────┘    └─────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │ AutoMerging-    │
                    │ Retriever       │
                    │ (Haystack)      │
                    └─────────────────┘
```

### Related Components

| Component | Relationship |
|-----------|--------------|
| `RetrieverAgent` | Abstract base class (from `core.interfaces`) |
| `RetrieverInput` | Input schema with query, plan, expanded queries |
| `RetrieverOutput` | Output schema with retrieval results |
| `RetrievalResult` | Per-document result with snippets |
| `Snippet` | Individual text chunk with metadata |
| `ChromaDocumentStore` | Haystack integration for ChromaDB |
| `InMemoryDocumentStore` | Haystack in-memory store for BM25 |
| `AutoMergingRetriever` | Haystack component for hierarchical merging |

---

## Class Structure

### Main Classes

#### HybridRetrievalAgent

```python
class HybridRetrievalAgent(RetrieverAgent):
    """Hybrid retriever supporting dual-index and leaf-only modes."""
```

| Attribute | Type | Description |
|-----------|------|-------------|
| `role` | `str` | `"retriever"` |
| `_cfg` | `RetrieverConfig` | Loaded configuration |
| `_leaf_store` | `ChromaDocumentStore` | Leaf document index |
| `_parent_store` | `ChromaDocumentStore` | Parent document index |
| `_bm25_store` | `InMemoryDocumentStore` | BM25 lexical index |
| `_parent_sidecar` | `Dict[str, Dict]` | Parent metadata sidecar |
| `_auto_merge` | `AutoMergeAgent` | Chunk merger helper |
| `_cache` | `QueryCache` | Result cache reference |

#### QueryCache

```python
@dataclass
class QueryCache:
    """LRU cache for RetrieverOutput payloads."""
```

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_size` | `int` | `128` | Maximum cached entries |
| `stats` | `CacheStats` | - | Hit/miss statistics |
| `_store` | `OrderedDict` | - | LRU storage |
| `_lock` | `Lock` | - | Thread safety lock |

#### CacheStats

```python
@dataclass
class CacheStats:
    """Cache performance statistics."""
```

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | `bool` | `True` | Cache enabled flag |
| `hits` | `int` | `0` | Cache hit count |
| `misses` | `int` | `0` | Cache miss count |
| `stores` | `int` | `0` | Cache store count |

#### RetrieverConfig

```python
@dataclass
class RetrieverConfig:
    """Retrieval configuration from config.fast.yaml."""
```

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `leaf_chroma_path` | `str` | `"../data/database/chroma_leaf_store"` | Leaf index path |
| `leaf_collection` | `str` | `"leaves"` | Leaf collection name |
| `parent_chroma_path` | `str` | `"../data/database/chroma_parents_store"` | Parent index path |
| `parent_collection` | `str` | `"parents"` | Parent collection name |
| `leaf_only` | `bool` | `False` | Leaf-only mode flag |
| `parent_sidecar_path` | `str` | `"../data/metadata/parents_sidecar.json"` | Sidecar file path |
| `leaf_top_k` | `int` | `50` | Documents per retrieval |
| `enable_hybrid` | `bool` | `True` | Enable BM25 fusion |
| `bm25_top_k` | `int` | `200` | BM25 retrieval limit |
| `merge_threshold` | `float` | `0.45` | Auto-merge threshold |

#### AutoMergeAgent

```python
class AutoMergeAgent:
    """Wrapper around Haystack's AutoMergingRetriever."""
```

---

## Core Components

### 1. Dense Retrieval

The agent uses ChromaDB for dense vector retrieval via Haystack's `ChromaQueryTextRetriever`.

**Flow:**
```
Query → ChromaQueryTextRetriever → Top-K Documents (by embedding similarity)
```

**Indices:**
- **Leaf Index:** Fine-grained document chunks
- **Parent Index:** Coarse-grained parent documents

### 2. BM25 Lexical Retrieval

When `enable_hybrid=True`, the agent builds an in-memory BM25 index over documents enriched with metadata.

**Enrichment Fields:**
- `display_summary`
- `vision_caption`
- `title`
- `filename`
- `source_path`

**Purpose:** Capture keyword matches that dense embeddings might miss (e.g., exact product names, technical terms).

### 3. Sidecar Lexical Boost

A fallback lexical search over parent sidecar metadata for robust retrieval of image-heavy documents.

**Algorithm:**
1. Tokenize query, remove stopwords
2. Scan sidecar content and metadata fields
3. Boost matching parent documents with score = 1000.0

### 4. Auto-Merging

Haystack's `AutoMergingRetriever` merges leaf chunks into parent documents when sufficient chunks from the same parent are retrieved.

**Parameters:**
- `threshold`: Minimum proportion of chunks required for merge (default: 0.45)

### 5. Result Grouping

Results are grouped by parent ID, with metadata enriched from:
1. Parent document metadata
2. Sidecar file metadata
3. Leaf document metadata (fallback)

---

## Configuration System

### Configuration File: `config.fast.yaml`

```yaml
vectorstore:
  persist_path: ./chroma_db
  collection_name: leaves

parent_vectorstore:
  persist_path: ./chroma_db_parents
  collection_name: parents

retrieval:
  leaf_only: false
  parent_sidecar_path: ./run_meta/parents_sidecar.json
  leaf_top_k: 50
  enable_hybrid: true
  bm25_top_k: 200
  merge_threshold: 0.45
```

### Configuration Resolution Order

```
1. Explicit config_path parameter
         │
         ▼
2. $AGENTIC_RAG_CONFIG environment variable
         │
         ▼
3. ./config.fast.yaml (relative to module)
         │
         ▼
4. Default values in RetrieverConfig
```

### Path Resolution

All relative paths in configuration are resolved relative to the config file's directory, not the current working directory.

---

## Retrieval Modes

### Dual-Index Mode (Default)

```
leaf_only: false
```

**Behavior:**
1. Query both leaf and parent ChromaDB indices
2. Apply BM25 fusion (if enabled)
3. Auto-merge leaf chunks into parents
4. Group results by parent ID

**Best For:**
- Hierarchical document collections
- Documents with natural parent-child relationships
- Long documents split into chunks

### Leaf-Only Mode

```
leaf_only: true
```

**Behavior:**
1. Query only leaf ChromaDB index
2. Apply BM25 fusion (if enabled)
3. Skip auto-merge step
4. Group results by parent ID (using `__parent_id` metadata)
5. Enrich parent metadata from sidecar file

**Best For:**
- Flat document collections
- Performance-sensitive deployments
- When parent index is unavailable

### Mode Override via Plan

The Plan's `retrieval_mode` can override the config default:

```python
plan.retrieval_mode = RetrievalModeEnum.LEAF_ONLY  # Force leaf-only
plan.retrieval_mode = RetrievalModeEnum.DUAL_INDEX  # Force dual-index
```

---

## Caching System

### Cache Key Components

The cache key is a JSON-serialized dictionary containing:

| Field | Description |
|-------|-------------|
| `query` | Original query text |
| `prf_augmented_query` | PRF-augmented query |
| `expanded_queries` | List of expanded queries |
| `retrieval_mode` | DUAL_INDEX or LEAF_ONLY |
| `use_qe` | Query expansion flag |
| `use_prf` | PRF flag |
| `use_rerank` | Reranking flag |
| `leaf_only` | Config leaf_only setting |
| `enable_hybrid` | Hybrid retrieval flag |
| `merge_threshold` | Auto-merge threshold |
| `leaf_top_k` | Leaf retrieval limit |
| `bm25_top_k` | BM25 retrieval limit |
| `leaf_chroma_path` | Leaf index path |
| `leaf_collection` | Leaf collection name |
| `parent_chroma_path` | Parent index path |
| `parent_collection` | Parent collection name |

### Cache Behavior

- **LRU Eviction:** Oldest entries evicted when `max_size` exceeded
- **Thread-Safe:** Protected by `threading.Lock`
- **Global Instance:** `RETRIEVAL_QUERY_CACHE` shared across pipeline

### Cache Statistics

```python
cache = RETRIEVAL_QUERY_CACHE
print(f"Hit Rate: {cache.stats.hit_rate:.2%}")
print(f"Hits: {cache.stats.hits}, Misses: {cache.stats.misses}")
print(f"Current Size: {cache.size}")
```

---

## Data Flow

### Input Schema: `RetrieverInput`

```python
@dataclass
class RetrieverInput:
    query: str                           # Original query
    plan: Plan                           # Execution plan
    expanded_queries: Optional[List[str]] # QE-expanded queries
    prf_augmented_query: Optional[str]   # PRF-augmented query
```

### Output Schema: `RetrieverOutput`

```python
@dataclass
class RetrieverOutput:
    results: List[RetrievalResult]
```

### RetrievalResult Schema

```python
@dataclass
class RetrievalResult:
    doc_id: str                    # Parent document ID
    parent_metadata: Dict[str, Any] # Parent metadata
    snippets: List[Snippet]        # Retrieved snippets
```

### Snippet Schema

```python
@dataclass
class Snippet:
    chunk_id: str      # Chunk identifier
    score: float       # Retrieval score
    text: str          # Snippet text (max 512 chars)
    lang: Optional[str] # Language code
    page: Optional[int] # Page number
    level: str         # "leaf" or "parent"
```

---

## Testing Strategies

### Unit Tests

#### 1. Configuration Loading Tests

```python
import pytest
import tempfile
from pathlib import Path
from retriever_haystack_agent import _load_retriever_cfg, RetrieverConfig

class TestConfigLoading:
    
    def test_load_yaml_config(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
vectorstore:
  persist_path: ./test_leaf
  collection_name: test_leaves
parent_vectorstore:
  persist_path: ./test_parent
  collection_name: test_parents
retrieval:
  leaf_only: true
  leaf_top_k: 100
  enable_hybrid: false
  bm25_top_k: 50
  merge_threshold: 0.6
""")
        cfg = _load_retriever_cfg(str(config_file))
        
        assert cfg.leaf_only == True
        assert cfg.leaf_top_k == 100
        assert cfg.enable_hybrid == False
        assert cfg.merge_threshold == 0.6
    
    def test_missing_config_uses_defaults(self):
        cfg = _load_retriever_cfg("/nonexistent/config.yaml")
        
        assert cfg.leaf_only == False
        assert cfg.leaf_top_k == 50
        assert cfg.enable_hybrid == True
    
    def test_relative_path_resolution(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
vectorstore:
  persist_path: ./relative/path
""")
        cfg = _load_retriever_cfg(str(config_file))
        
        expected = str((tmp_path / "relative/path").resolve())
        assert cfg.leaf_chroma_path == expected
```

#### 2. Cache Tests

```python
from retriever_haystack_agent import QueryCache, CacheStats
from unittest.mock import Mock

class TestQueryCache:
    
    @pytest.fixture
    def cache(self):
        return QueryCache(max_size=3)
    
    @pytest.fixture
    def mock_input(self):
        inp = Mock()
        inp.query = "test query"
        inp.prf_augmented_query = None
        inp.expanded_queries = []
        inp.plan = Mock()
        inp.plan.retrieval_mode = None
        inp.plan.use_qe = False
        inp.plan.use_prf = False
        inp.plan.use_rerank = False
        return inp
    
    @pytest.fixture
    def mock_config(self):
        return RetrieverConfig()
    
    def test_cache_miss_on_empty(self, cache, mock_input, mock_config):
        result = cache.get(mock_input, mock_config)
        assert result is None
        assert cache.stats.misses == 1
    
    def test_cache_hit_after_put(self, cache, mock_input, mock_config):
        output = Mock()
        cache.put(mock_input, mock_config, output)
        
        result = cache.get(mock_input, mock_config)
        assert result is output
        assert cache.stats.hits == 1
    
    def test_lru_eviction(self, cache, mock_config):
        # Fill cache beyond capacity
        for i in range(5):
            inp = Mock()
            inp.query = f"query_{i}"
            inp.prf_augmented_query = None
            inp.expanded_queries = []
            inp.plan = Mock()
            inp.plan.retrieval_mode = None
            inp.plan.use_qe = False
            inp.plan.use_prf = False
            inp.plan.use_rerank = False
            
            cache.put(inp, mock_config, Mock())
        
        assert cache.size == 3  # max_size
    
    def test_cache_disabled(self, mock_input, mock_config):
        cache = QueryCache(max_size=10)
        cache.stats.enabled = False
        
        output = Mock()
        cache.put(mock_input, mock_config, output)
        result = cache.get(mock_input, mock_config)
        
        assert result is None
    
    def test_hit_rate_calculation(self, cache):
        cache.stats.hits = 3
        cache.stats.misses = 7
        
        assert cache.stats.hit_rate == 0.3
        assert cache.stats.total_lookups == 10
```

#### 3. Query Normalization Tests

```python
from retriever_haystack_agent import _normalize_query_text

class TestQueryNormalization:
    
    def test_strip_whitespace(self):
        assert _normalize_query_text("  hello  ") == "hello"
    
    def test_collapse_spaces(self):
        assert _normalize_query_text("hello    world") == "hello world"
    
    def test_empty_string(self):
        assert _normalize_query_text("") == ""
    
    def test_none_handling(self):
        assert _normalize_query_text(None) == ""
```

#### 4. BM25 Enrichment Tests

```python
from retriever_haystack_agent import _enrich_docs_for_bm25
from haystack import Document

class TestBM25Enrichment:
    
    def test_enrichment_adds_caption(self):
        doc = Document(
            id="1",
            content="Original content",
            meta={"vision_caption": "A beautiful sunset"}
        )
        _enrich_docs_for_bm25([doc])
        
        assert "A beautiful sunset" in doc.content
        assert "Original content" in doc.content
    
    def test_enrichment_adds_title(self):
        doc = Document(
            id="1",
            content="Content",
            meta={"title": "Document Title"}
        )
        _enrich_docs_for_bm25([doc])
        
        assert "Document Title" in doc.content
    
    def test_enrichment_preserves_original(self):
        doc = Document(
            id="1",
            content="Original",
            meta={}
        )
        _enrich_docs_for_bm25([doc])
        
        assert doc.content == "Original"
    
    def test_enrichment_multiple_fields(self):
        doc = Document(
            id="1",
            content="Base",
            meta={
                "display_summary": "Summary",
                "title": "Title",
                "filename": "file.pdf"
            }
        )
        _enrich_docs_for_bm25([doc])
        
        assert "Summary" in doc.content
        assert "Title" in doc.content
        assert "file.pdf" in doc.content
```

#### 5. Deduplication Tests

```python
from retriever_haystack_agent import _dedupe_docs
from haystack import Document

class TestDeduplication:
    
    def test_removes_duplicates(self):
        docs = [
            Document(id="1", content="A"),
            Document(id="2", content="B"),
            Document(id="1", content="A duplicate"),
        ]
        result = _dedupe_docs(docs)
        
        assert len(result) == 2
        assert result[0].id == "1"
        assert result[1].id == "2"
    
    def test_preserves_order(self):
        docs = [
            Document(id="3", content="C"),
            Document(id="1", content="A"),
            Document(id="2", content="B"),
        ]
        result = _dedupe_docs(docs)
        
        assert [d.id for d in result] == ["3", "1", "2"]
```

#### 6. Snippet Building Tests

```python
class TestSnippetBuilding:
    
    @pytest.fixture
    def agent(self):
        # Create agent with mocked stores
        agent = HybridRetrievalAgent.__new__(HybridRetrievalAgent)
        agent._page_field = "page"
        agent._level_field = "__level"
        agent._lang_field = "language"
        return agent
    
    def test_snippet_from_content(self, agent):
        doc = Document(
            id="chunk_1",
            content="This is the content",
            score=0.85,
            meta={"page": 5, "__level": "leaf"}
        )
        snippet = agent._build_snippet(doc)
        
        assert snippet.chunk_id == "chunk_1"
        assert snippet.text == "This is the content"
        assert snippet.score == 0.85
        assert snippet.page == 5
        assert snippet.level == "leaf"
    
    def test_snippet_prefers_display_summary(self, agent):
        doc = Document(
            id="1",
            content="Raw content",
            score=0.5,
            meta={"display_summary": "Nice summary"}
        )
        snippet = agent._build_snippet(doc)
        
        assert snippet.text == "Nice summary"
    
    def test_snippet_truncation(self, agent):
        doc = Document(
            id="1",
            content="A" * 1000,
            score=0.5,
            meta={}
        )
        snippet = agent._build_snippet(doc)
        
        assert len(snippet.text) <= 512
        assert snippet.text.endswith("…")
    
    def test_snippet_fallback_for_images(self, agent):
        doc = Document(
            id="1",
            content="",
            score=0.5,
            meta={"filename": "image.png"}
        )
        snippet = agent._build_snippet(doc)
        
        assert "image.png" in snippet.text
```

#### 7. Integration Tests

```python
class TestRetrievalIntegration:
    
    @pytest.fixture
    def mock_agent(self, tmp_path):
        """Create agent with mocked ChromaDB stores."""
        # This would require setting up test ChromaDB instances
        # or mocking the store interactions
        pass
    
    def test_cache_integration(self, mock_agent):
        """Test that results are cached and returned on repeat queries."""
        pass
    
    def test_mode_override_via_plan(self, mock_agent):
        """Test that plan.retrieval_mode overrides config."""
        pass
    
    def test_query_expansion_handling(self, mock_agent):
        """Test that expanded queries are processed."""
        pass
```

### Test Commands

```bash
# Run all retriever tests
pytest test_retriever_haystack_agent.py -v

# Run with coverage
pytest test_retriever_haystack_agent.py --cov=retriever_haystack_agent --cov-report=html

# Run cache tests only
pytest test_retriever_haystack_agent.py::TestQueryCache -v
```

---

## Recommendations and Improvements

### Critical Issues

#### 1. No Cache Invalidation Mechanism

**Problem:** Cache entries are never invalidated when underlying indices change.

**Recommendation:** Add cache invalidation:

```python
class QueryCache:
    def invalidate_all(self) -> None:
        """Clear all cached entries."""
        with self._lock:
            self._store.clear()
            self.stats.stores = 0
    
    def invalidate_by_index(self, index_path: str) -> None:
        """Invalidate entries referencing a specific index."""
        with self._lock:
            keys_to_remove = [
                k for k in self._store 
                if index_path in k
            ]
            for k in keys_to_remove:
                self._store.pop(k)
```

#### 2. Silent Exception Handling

**Problem:** Many exceptions are silently caught and return empty results, making debugging difficult.

**Current:**
```python
except Exception:
    docs_leaf = []
```

**Recommendation:** Add logging:

```python
import logging
logger = logging.getLogger(__name__)

try:
    res_leaf = leaf_retriever.run(query=qtext)
except Exception as e:
    logger.warning(f"Leaf retrieval failed for query '{qtext}': {e}")
    docs_leaf = []
```

---

### High Priority Improvements

#### 3. Configurable Cache Size

**Problem:** Global cache size is hardcoded.

**Recommendation:** Make cache size configurable:

```yaml
# config.fast.yaml
retrieval:
  cache_max_size: 512
  cache_enabled: true
```

```python
def __init__(self, *, config_path: Optional[str] = None, ...):
    self._cfg = _load_retriever_cfg(config_path)
    cache_size = self._cfg.cache_max_size or 256
    self._cache = QueryCache(max_size=cache_size)
```

#### 4. Retrieval Metrics and Observability

**Problem:** No metrics for retrieval performance monitoring.

**Recommendation:** Add structured metrics:

```python
import time
from dataclasses import dataclass

@dataclass
class RetrievalMetrics:
    query: str
    total_time_ms: float
    dense_time_ms: float
    bm25_time_ms: float
    merge_time_ms: float
    num_leaf_docs: int
    num_parent_docs: int
    cache_hit: bool

def retrieve(self, inp: RetrieverInput) -> RetrieverOutput:
    start = time.perf_counter()
    metrics = RetrievalMetrics(query=inp.query, ...)
    
    # ... retrieval logic with timing ...
    
    metrics.total_time_ms = (time.perf_counter() - start) * 1000
    logger.info("retrieval_complete", extra={"metrics": asdict(metrics)})
    
    return output
```

#### 5. Configurable Stopwords

**Problem:** Stopwords in `_sidecar_lexical_boost` are hardcoded.

**Recommendation:** Make stopwords configurable or use a library:

```python
from nltk.corpus import stopwords

STOPWORDS = set(stopwords.words('english'))

# Or via config:
retrieval:
  sidecar_stopwords:
    - the
    - a
    - an
```

#### 6. Score Normalization

**Problem:** Scores from different sources (dense, BM25, sidecar) are on different scales.

**Recommendation:** Normalize scores before fusion:

```python
def _normalize_scores(self, docs: List[Document], method: str = "minmax") -> None:
    if not docs:
        return
    
    scores = [d.score or 0.0 for d in docs]
    min_score, max_score = min(scores), max(scores)
    
    if max_score == min_score:
        for d in docs:
            d.score = 1.0
        return
    
    for d in docs:
        d.score = (d.score - min_score) / (max_score - min_score)
```

---

### Medium Priority Improvements

#### 7. Async Retrieval Support

**Problem:** All retrieval is synchronous, limiting throughput.

**Recommendation:** Add async support:

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

class HybridRetrievalAgent:
    def __init__(self, ...):
        self._executor = ThreadPoolExecutor(max_workers=4)
    
    async def retrieve_async(self, inp: RetrieverInput) -> RetrieverOutput:
        loop = asyncio.get_event_loop()
        
        # Run dense and BM25 retrieval in parallel
        dense_task = loop.run_in_executor(
            self._executor, self._dense_retrieve, inp
        )
        bm25_task = loop.run_in_executor(
            self._executor, self._bm25_retrieve, inp
        )
        
        dense_results, bm25_results = await asyncio.gather(
            dense_task, bm25_task
        )
        
        return self._merge_results(dense_results, bm25_results)
```

#### 8. Reciprocal Rank Fusion (RRF)

**Problem:** Current fusion simply takes max score, which may not be optimal.

**Recommendation:** Implement RRF for better fusion:

```python
def _reciprocal_rank_fusion(
    self,
    rankings: List[List[Document]],
    k: int = 60
) -> List[Document]:
    """Fuse multiple rankings using RRF."""
    scores: Dict[str, float] = defaultdict(float)
    docs_by_id: Dict[str, Document] = {}
    
    for ranking in rankings:
        for rank, doc in enumerate(ranking):
            scores[doc.id] += 1.0 / (k + rank + 1)
            docs_by_id[doc.id] = doc
    
    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    
    result = []
    for doc_id in sorted_ids:
        doc = docs_by_id[doc_id]
        doc.score = scores[doc_id]
        result.append(doc)
    
    return result
```

#### 9. Query-Dependent top_k

**Problem:** Fixed top_k regardless of query complexity.

**Recommendation:** Adjust top_k based on query:

```python
def _compute_dynamic_top_k(self, query: str, base_top_k: int) -> int:
    """Adjust top_k based on query characteristics."""
    word_count = len(query.split())
    
    if word_count <= 3:
        # Short queries need more candidates
        return int(base_top_k * 1.5)
    elif word_count >= 10:
        # Long queries are more specific
        return int(base_top_k * 0.75)
    
    return base_top_k
```

---

### Low Priority / Future Enhancements

#### 10. Multi-Index Support

**Recommendation:** Support multiple leaf/parent index pairs:

```yaml
indices:
  - name: documents
    leaf_path: ./chroma_docs_leaf
    parent_path: ./chroma_docs_parent
  - name: images
    leaf_path: ./chroma_images_leaf
    parent_path: ./chroma_images_parent
```

#### 11. Embedding Model Flexibility

**Recommendation:** Support different embedding models:

```python
class HybridRetrievalAgent:
    def __init__(
        self,
        *,
        embedding_model: str = "default",
        ...
    ):
        self._embedding_model = embedding_model
```

#### 12. Result Explanation

**Recommendation:** Add retrieval explanations for debugging:

```python
@dataclass
class RetrievalExplanation:
    doc_id: str
    sources: List[str]  # ["dense_leaf", "bm25", "sidecar"]
    scores_by_source: Dict[str, float]
    merged_from: Optional[List[str]]  # Child chunk IDs if merged
```

---

## Usage Examples

### Basic Usage

```python
from retriever_haystack_agent import HybridRetrievalAgent
from core.schemas import RetrieverInput, Plan

# Initialize agent
agent = HybridRetrievalAgent(config_path="config.fast.yaml")

# Create input
plan = Plan(
    retrieval_mode=RetrievalModeEnum.DUAL_INDEX,
    top_k=10,
    # ... other plan fields
)

inp = RetrieverInput(
    query="What are the key features of the product?",
    plan=plan,
    expanded_queries=["product features", "main capabilities"],
    prf_augmented_query=None
)

# Retrieve
output = agent.retrieve(inp)

for result in output.results:
    print(f"Document: {result.doc_id}")
    print(f"Title: {result.parent_metadata.get('title')}")
    for snippet in result.snippets:
        print(f"  - [{snippet.score:.2f}] {snippet.text[:100]}...")
```

### Pipeline Integration

```python
class RAGPipeline:
    def __init__(self, config_path: str):
        self.retriever = HybridRetrievalAgent(config_path=config_path)
        self.reranker = RerankAgent()
        self.generator = GeneratorAgent()
    
    def process(self, query: str, plan: Plan):
        # Step 1: Retrieve
        retriever_input = RetrieverInput(
            query=query,
            plan=plan,
            expanded_queries=plan.expanded_queries,
            prf_augmented_query=plan.prf_query
        )
        retriever_output = self.retriever.retrieve(retriever_input)
        
        # Step 2: Flatten snippets for reranking
        all_snippets = []
        for result in retriever_output.results:
            all_snippets.extend(result.snippets)
        
        # Step 3: Rerank if enabled
        if plan.use_rerank:
            reranked = self.reranker.rerank(query, all_snippets)
            top_snippets = reranked[:plan.rerank_top_k]
        else:
            top_snippets = sorted(
                all_snippets, 
                key=lambda s: s.score, 
                reverse=True
            )[:plan.top_k]
        
        # Step 4: Generate
        return self.generator.generate(query, top_snippets)
```

### Cache Management

```python
from retriever_haystack_agent import RETRIEVAL_QUERY_CACHE

# Check cache stats
cache = RETRIEVAL_QUERY_CACHE
print(f"Cache Size: {cache.size}")
print(f"Hit Rate: {cache.stats.hit_rate:.2%}")

# Disable caching for debugging
cache.stats.enabled = False

# Clear cache after index update
cache.invalidate_all()
```

### Custom Field Configuration

```python
agent = HybridRetrievalAgent(
    config_path="config.yaml",
    parent_id_field="parent_doc_id",      # Custom parent ID field
    parent_title_field="document_title",   # Custom title field
    page_field="page_number",              # Custom page field
    lang_field="doc_language",             # Custom language field
)
```

---

## Appendix

### Glossary

| Term | Definition |
|------|------------|
| **Dense Retrieval** | Vector similarity search using embeddings |
| **BM25** | Best Match 25 - lexical retrieval algorithm |
| **Hybrid Retrieval** | Combination of dense and lexical retrieval |
| **Auto-Merge** | Combining leaf chunks into parent documents |
| **Sidecar** | JSON file with precomputed parent metadata |
| **LRU** | Least Recently Used - cache eviction strategy |
| **ChromaDB** | Vector database for embedding storage |

### Configuration Reference

| Config Path | Type | Default | Description |
|-------------|------|---------|-------------|
| `vectorstore.persist_path` | str | `"./chroma_db"` | Leaf index path |
| `vectorstore.collection_name` | str | `"leaves"` | Leaf collection |
| `parent_vectorstore.persist_path` | str | `"./chroma_db_parents"` | Parent index path |
| `parent_vectorstore.collection_name` | str | `"parents"` | Parent collection |
| `retrieval.leaf_only` | bool | `false` | Leaf-only mode |
| `retrieval.parent_sidecar_path` | str | - | Sidecar JSON path |
| `retrieval.leaf_top_k` | int | `50` | Dense retrieval limit |
| `retrieval.enable_hybrid` | bool | `true` | Enable BM25 |
| `retrieval.bm25_top_k` | int | `200` | BM25 retrieval limit |
| `retrieval.merge_threshold` | float | `0.45` | Auto-merge threshold |

### Metadata Field Reference

| Field | Purpose | Source |
|-------|---------|--------|
| `__parent_id` | Parent document reference | Indexer |
| `__level` | Document hierarchy level | Indexer |
| `title` | Document title | Metadata |
| `filename` | Source filename | Metadata |
| `source_path` | Source file path | Metadata |
| `page` | Page number | Metadata |
| `language` | Document language | Metadata |
| `display_summary` | Human-readable summary | Indexer |
| `vision_caption` | Image caption from vision model | Indexer |

### Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | Initial | Basic ChromaDB retrieval |
| 2.0 | - | Hybrid retrieval with BM25 |
| 3.0 | - | Auto-merge and caching |

### References

- Radiant Repository: https://github.com/dshipley71/radiant
- Haystack Documentation: https://docs.haystack.deepset.ai/
- ChromaDB Documentation: https://docs.trychroma.com/
- Related Files: `hier_indexer.py`, `orchestrator.py`, `config.fast.yaml`

---

*Document generated for the Radiant RAG Pipeline project. For questions or contributions, please refer to the project repository.*
