# BasicPRFAgent Documentation

## Technical Reference for the Radiant RAG Pipeline Pseudo-Relevance Feedback

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture Context](#architecture-context)
3. [Class Structure](#class-structure)
4. [Core Functionality](#core-functionality)
5. [PRF Algorithm](#prf-algorithm)
6. [Configuration System](#configuration-system)
7. [Data Flow](#data-flow)
8. [Testing Strategies](#testing-strategies)
9. [Recommendations and Improvements](#recommendations-and-improvements)
10. [Usage Examples](#usage-examples)

---

## Overview

### Purpose

The `BasicPRFAgent` implements Pseudo-Relevance Feedback (PRF), a query expansion technique within the Radiant RAG pipeline. It automatically expands user queries with high-frequency terms extracted from top BM25 search results, improving recall by capturing relevant vocabulary that users may not have explicitly included.

### Key Responsibilities

- Build and maintain an in-memory BM25 index over leaf documents
- Execute BM25 retrieval to find initially relevant documents
- Extract high-frequency unigrams and bigrams from top results
- Augment the original query with extracted PRF terms
- Provide both raw PRF terms and the augmented query

### Design Philosophy

The agent follows a **blind relevance feedback** approach, assuming top BM25 results are relevant without user confirmation. This enables automatic query expansion at the cost of potentially introducing noise if initial results are poor. The technique is particularly effective for vocabulary mismatch problems.

---

## Architecture Context

### Position in the Radiant Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    Original Query                               │
│               "machine learning models"                         │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                      BasicPRFAgent                              │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  1. BM25 Retrieval                                      │   │
│  │     └─ Top K documents from leaf index                  │   │
│  ├─────────────────────────────────────────────────────────┤   │
│  │  2. Term Extraction                                     │   │
│  │     └─ Tokenize documents                               │   │
│  │     └─ Count unigrams and bigrams                       │   │
│  │     └─ Select top terms by frequency                    │   │
│  ├─────────────────────────────────────────────────────────┤   │
│  │  3. Query Augmentation                                  │   │
│  │     └─ Append PRF terms to original query               │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Augmented Query                              │
│    "machine learning models neural network deep training"       │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
                    Dense Retrieval (improved recall)
```

### PRF in Query Processing Pipeline

```
User Query
    │
    ├─────────────────────┐
    │                     │
    ▼                     ▼
┌─────────┐         ┌─────────┐
│   QE    │         │   PRF   │
│ (LLM)   │         │ (BM25)  │
└────┬────┘         └────┬────┘
     │                   │
     ▼                   ▼
expanded_queries    prf_augmented_query
     │                   │
     └─────────┬─────────┘
               │
               ▼
        Dense Retrieval
        (uses all queries)
```

### Related Components

| Component | Relationship |
|-----------|--------------|
| `PRFAgent` | Abstract base class (from `core.interfaces`) |
| `PRFInput` | Input schema with query and BM25 config |
| `PRFOutput` | Output schema with PRF terms and augmented query |
| `ChromaDocumentStore` | Source of leaf documents for BM25 index |
| `InMemoryBM25Retriever` | Haystack BM25 implementation |
| `PRFRetrievalConfig` | Configuration dataclass |

---

## Class Structure

### Main Class: BasicPRFAgent

```python
class BasicPRFAgent(PRFAgent):
    """PRF agent backed by a BM25 index over the leaf Chroma corpus."""
```

### Class Attributes

| Attribute | Type | Value | Description |
|-----------|------|-------|-------------|
| `role` | `str` | `"prf"` | Agent role identifier |

### Instance Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `_cfg` | `PRFRetrievalConfig` | Loaded configuration |
| `_prf_docs` | `int` | Max documents for PRF |
| `_prf_terms` | `int` | Max terms to extract |
| `_bm25` | `InMemoryBM25Retriever` | BM25 retriever (may be None) |

### Constructor

```python
def __init__(self, config_path: Optional[str] = None) -> None
```

**Initialization Steps:**
1. Load configuration from file
2. Set PRF parameters
3. Build BM25 index (if hybrid enabled)

### Methods

| Method | Visibility | Purpose |
|--------|------------|---------|
| `name` | Property | Returns agent name |
| `describe()` | Public | Returns agent description |
| `compute(inp)` | Public | Main PRF computation method |
| `_build_bm25_index()` | Private | Builds in-memory BM25 index |

### Supporting Classes

#### PRFRetrievalConfig

```python
@dataclass
class PRFRetrievalConfig:
    leaf_chroma_path: str = "./chroma_db"
    leaf_collection: str = "leaves"
    enable_hybrid: bool = True
    bm25_top_k: int = 200
    prf_docs: int = 10
    prf_terms: int = 6
```

---

## Core Functionality

### The `compute()` Method

Primary method that performs PRF computation.

**Signature:**
```python
def compute(self, inp: PRFInput) -> PRFOutput
```

**Parameters:**
- `inp` (`PRFInput`): Contains query and optional BM25 configuration

**Returns:**
- `PRFOutput`: Contains PRF terms and augmented query

**Processing Steps:**

1. **Check BM25 Availability**
   - If `_bm25` is None, return original query unchanged

2. **Determine Limits**
   - Use configured `prf_docs` and `prf_terms`
   - Optionally limit by `inp.bm25_config.top_k`

3. **Run BM25 Retrieval**
   - Execute BM25 search with original query
   - Get top documents

4. **Extract PRF Terms**
   - Tokenize document content
   - Count unigrams and bigrams
   - Select top terms by frequency

5. **Build Augmented Query**
   - Append PRF terms to original query

6. **Return Output**
   - Package PRF terms and augmented query

### The `_build_bm25_index()` Method

Constructs the in-memory BM25 index from ChromaDB.

**Process:**
1. Check if hybrid is enabled
2. Open leaf ChromaDocumentStore
3. Create InMemoryDocumentStore
4. Write all leaf documents to memory store
5. Create BM25 retriever

---

## PRF Algorithm

### Term Extraction Process

```
┌─────────────────────────────────────────────────────────────────┐
│                    Top BM25 Documents                           │
│  Doc1: "Neural networks are powerful machine learning models"   │
│  Doc2: "Deep learning uses neural network architectures"        │
│  Doc3: "Training neural networks requires large datasets"       │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Tokenization                                 │
│  Pattern: [A-Za-z0-9]+  (alphanumeric, case-insensitive)       │
│  Filter: len(token) > 2  (exclude short tokens)                │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Frequency Counting                           │
│  Unigrams: {"neural": 3, "networks": 2, "learning": 2, ...}    │
│  Bigrams: {("neural", "networks"): 2, ("deep", "learning"): 1} │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Term Selection                               │
│  Top unigrams: ["neural", "networks", "learning", ...]         │
│  Top bigrams: ["neural networks", "deep learning", ...]        │
│  Combined (deduplicated): ["neural", "networks", ...]          │
└─────────────────────────────────────────────────────────────────┘
```

### Tokenization Pattern

```python
_TOKEN = re.compile(r"[A-Za-z0-9]+")
```

**Behavior:**
- Matches alphanumeric sequences
- Case-insensitive (converted to lowercase)
- Filters tokens with length ≤ 2

**Examples:**
| Input | Tokens |
|-------|--------|
| `"Hello World!"` | `["hello", "world"]` |
| `"ML-based AI"` | `["based"]` (ML and AI too short) |
| `"test123"` | `["test123"]` |

### Term Selection Algorithm

```python
def _prf_expand_terms(query, bm25_docs, max_docs, max_terms):
    # 1. Concatenate document texts
    texts = [doc.content for doc in bm25_docs[:max_docs]]
    
    # 2. Tokenize and filter
    toks = [t for t in _TOKEN.findall(" ".join(texts).lower()) if len(t) > 2]
    
    # 3. Count frequencies
    uni = Counter(toks)           # Unigram counts
    big = Counter(zip(toks, toks[1:]))  # Bigram counts
    
    # 4. Select top unigrams
    mix = []
    for w, _ in uni.most_common(max_terms * 2):
        mix.append(w)
        if len(mix) >= max_terms:
            break
    
    # 5. Add top bigrams
    for (a, b), _ in big.most_common(max_terms):
        if len(mix) >= max_terms + 2:
            break
        mix.append(f"{a} {b}")
    
    # 6. Deduplicate preserving order
    mix = list(dict.fromkeys(mix))
    
    return mix[:max_terms + 2]
```

### Output Limits

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `prf_docs` | 10 | Documents to analyze |
| `prf_terms` | 6 | Base term count |
| Final limit | `prf_terms + 2` | Max terms returned |

---

## Configuration System

### Configuration File: `config.fast.yaml`

```yaml
vectorstore:
  persist_path: ./chroma_db
  collection_name: leaves

retrieval:
  enable_hybrid: true
  bm25_top_k: 200
  prf_docs: 10
  prf_terms: 6
```

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `leaf_chroma_path` | str | `"./chroma_db"` | ChromaDB path for leaves |
| `leaf_collection` | str | `"leaves"` | Collection name |
| `enable_hybrid` | bool | `True` | Enable/disable PRF |
| `bm25_top_k` | int | `200` | BM25 retrieval limit |
| `prf_docs` | int | `10` | Docs for PRF extraction |
| `prf_terms` | int | `6` | Terms to extract |

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
4. Default values in PRFRetrievalConfig
```

### Configuration Fallbacks

The loader checks multiple keys for backward compatibility:

| Config Key | Fallback Key | Final Fallback |
|------------|--------------|----------------|
| `retrieval.leaf_chroma_path` | `vectorstore.persist_path` | `"./chroma_db"` |
| `retrieval.leaf_collection` | `vectorstore.collection_name` | `"leaves"` |

---

## Data Flow

### Input Schema: `PRFInput`

```python
@dataclass
class PRFInput:
    query: str                    # Original user query
    bm25_config: Optional[BM25Config]  # Optional BM25 parameters
```

### BM25Config Schema

```python
@dataclass
class BM25Config:
    top_k: int  # Override for BM25 retrieval limit
```

### Output Schema: `PRFOutput`

```python
@dataclass
class PRFOutput:
    prf_terms: List[str]      # Extracted PRF terms
    augmented_query: str      # Original query + PRF terms
```

### Example Transformation

**Input:**
```python
PRFInput(query="machine learning models")
```

**Output:**
```python
PRFOutput(
    prf_terms=["neural", "network", "deep", "training", "neural network", "deep learning"],
    augmented_query="machine learning models neural network deep training neural network deep learning"
)
```

---

## Testing Strategies

### Unit Tests

#### 1. Configuration Loading Tests

```python
import pytest
import tempfile
from pathlib import Path
from prf_basic_agent import _load_prf_cfg, PRFRetrievalConfig

class TestConfigLoading:
    
    def test_load_yaml_config(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
vectorstore:
  persist_path: ./test_chroma
  collection_name: test_leaves
retrieval:
  enable_hybrid: true
  prf_docs: 15
  prf_terms: 8
""")
        cfg = _load_prf_cfg(str(config_file))
        
        assert cfg.leaf_chroma_path == "./test_chroma"
        assert cfg.leaf_collection == "test_leaves"
        assert cfg.prf_docs == 15
        assert cfg.prf_terms == 8
    
    def test_missing_config_uses_defaults(self):
        cfg = _load_prf_cfg("/nonexistent/config.yaml")
        
        assert cfg.enable_hybrid == True
        assert cfg.prf_docs == 10
        assert cfg.prf_terms == 6
    
    def test_json_config_loading(self, tmp_path):
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps({
            "retrieval": {
                "enable_hybrid": False,
                "prf_docs": 5
            }
        }))
        cfg = _load_prf_cfg(str(config_file))
        
        assert cfg.enable_hybrid == False
        assert cfg.prf_docs == 5
```

#### 2. Tokenization Tests

```python
from prf_basic_agent import _TOKEN

class TestTokenization:
    
    def test_basic_tokenization(self):
        text = "Hello World Test"
        tokens = _TOKEN.findall(text.lower())
        
        assert tokens == ["hello", "world", "test"]
    
    def test_punctuation_removed(self):
        text = "Hello, World! How are you?"
        tokens = _TOKEN.findall(text.lower())
        
        assert tokens == ["hello", "world", "how", "are", "you"]
    
    def test_numbers_included(self):
        text = "Test123 abc456"
        tokens = _TOKEN.findall(text.lower())
        
        assert tokens == ["test123", "abc456"]
    
    def test_short_tokens_present(self):
        text = "AI ML is good"
        tokens = _TOKEN.findall(text.lower())
        
        # Short tokens ARE captured by regex, filtered later
        assert "ai" in tokens
        assert "ml" in tokens
```

#### 3. Term Extraction Tests

```python
from prf_basic_agent import _prf_expand_terms
from haystack import Document

class TestTermExtraction:
    
    def test_basic_term_extraction(self):
        docs = [
            Document(id="1", content="neural networks are powerful"),
            Document(id="2", content="neural network training methods"),
        ]
        
        terms = _prf_expand_terms(
            query="test",
            bm25_docs=docs,
            max_docs=10,
            max_terms=4
        )
        
        assert "neural" in terms
        assert "network" in terms or "networks" in terms
    
    def test_short_tokens_filtered(self):
        docs = [
            Document(id="1", content="AI ML is the future of NLP"),
        ]
        
        terms = _prf_expand_terms(
            query="test",
            bm25_docs=docs,
            max_docs=10,
            max_terms=10
        )
        
        # "AI", "ML", "is", "NLP" should be filtered (length <= 2)
        assert "the" not in terms  # length > 2 but might be filtered
        assert "future" in terms
    
    def test_bigrams_included(self):
        docs = [
            Document(id="1", content="deep learning deep learning deep learning"),
        ]
        
        terms = _prf_expand_terms(
            query="test",
            bm25_docs=docs,
            max_docs=10,
            max_terms=4
        )
        
        assert "deep learning" in terms
    
    def test_empty_docs_returns_empty(self):
        terms = _prf_expand_terms(
            query="test",
            bm25_docs=[],
            max_docs=10,
            max_terms=4
        )
        
        assert terms == []
    
    def test_max_docs_limit_respected(self):
        docs = [
            Document(id="1", content="first document content"),
            Document(id="2", content="second document content"),
            Document(id="3", content="third unique special terms"),
        ]
        
        # With max_docs=2, "third", "unique", "special" shouldn't appear
        terms = _prf_expand_terms(
            query="test",
            bm25_docs=docs,
            max_docs=2,
            max_terms=10
        )
        
        assert "third" not in terms
        assert "unique" not in terms
    
    def test_deduplication_preserves_order(self):
        docs = [
            Document(id="1", content="test test test duplicate duplicate"),
        ]
        
        terms = _prf_expand_terms(
            query="test",
            bm25_docs=docs,
            max_docs=10,
            max_terms=4
        )
        
        # No duplicates
        assert len(terms) == len(set(terms))
```

#### 4. PRF Agent Tests

```python
from prf_basic_agent import BasicPRFAgent
from core.schemas import PRFInput, PRFOutput
from unittest.mock import Mock, patch

class TestBasicPRFAgent:
    
    def test_disabled_prf_returns_original(self):
        """When BM25 is disabled, return original query unchanged."""
        agent = BasicPRFAgent.__new__(BasicPRFAgent)
        agent._bm25 = None
        agent._prf_docs = 10
        agent._prf_terms = 6
        
        inp = PRFInput(query="original query")
        output = agent.compute(inp)
        
        assert output.prf_terms == []
        assert output.augmented_query == "original query"
    
    @patch.object(BasicPRFAgent, '_build_bm25_index')
    def test_no_bm25_hits_returns_original(self, mock_build):
        agent = BasicPRFAgent(config_path=None)
        agent._bm25 = Mock()
        agent._bm25.run.return_value = {"documents": []}
        
        inp = PRFInput(query="test query")
        output = agent.compute(inp)
        
        assert output.prf_terms == []
        assert output.augmented_query == "test query"
    
    @patch.object(BasicPRFAgent, '_build_bm25_index')
    def test_successful_prf_expansion(self, mock_build):
        agent = BasicPRFAgent(config_path=None)
        agent._prf_docs = 10
        agent._prf_terms = 4
        
        mock_docs = [
            Document(id="1", content="neural network training methods"),
            Document(id="2", content="neural networks deep learning"),
        ]
        agent._bm25 = Mock()
        agent._bm25.run.return_value = {"documents": mock_docs}
        
        inp = PRFInput(query="machine learning")
        output = agent.compute(inp)
        
        assert len(output.prf_terms) > 0
        assert output.augmented_query.startswith("machine learning")
        assert len(output.augmented_query) > len("machine learning")
    
    @patch.object(BasicPRFAgent, '_build_bm25_index')
    def test_bm25_config_limits_docs(self, mock_build):
        agent = BasicPRFAgent(config_path=None)
        agent._prf_docs = 100
        agent._prf_terms = 4
        
        mock_docs = [Document(id=str(i), content=f"doc {i}") for i in range(50)]
        agent._bm25 = Mock()
        agent._bm25.run.return_value = {"documents": mock_docs}
        
        bm25_config = Mock()
        bm25_config.top_k = 5
        
        inp = PRFInput(query="test", bm25_config=bm25_config)
        output = agent.compute(inp)
        
        # Should use min(prf_docs, bm25_config.top_k) = 5
        # This affects which docs are processed for PRF
        assert output is not None
```

#### 5. Agent Interface Tests

```python
class TestAgentInterface:
    
    @patch.object(BasicPRFAgent, '_build_bm25_index')
    def test_name_property(self, mock_build):
        agent = BasicPRFAgent(config_path=None)
        assert agent.name == "BasicPRFAgent"
    
    @patch.object(BasicPRFAgent, '_build_bm25_index')
    def test_describe_method(self, mock_build):
        agent = BasicPRFAgent(config_path=None)
        description = agent.describe()
        
        assert isinstance(description, str)
        assert "PRF" in description or "Pseudo-Relevance" in description
    
    @patch.object(BasicPRFAgent, '_build_bm25_index')
    def test_role_attribute(self, mock_build):
        agent = BasicPRFAgent(config_path=None)
        assert agent.role == "prf"
```

#### 6. Integration Tests

```python
class TestPRFIntegration:
    
    @pytest.fixture
    def temp_chroma_store(self, tmp_path):
        """Create a temporary ChromaDB with test documents."""
        from haystack_integrations.document_stores.chroma import ChromaDocumentStore
        
        store = ChromaDocumentStore(
            persist_path=str(tmp_path / "chroma"),
            collection_name="test_leaves"
        )
        
        docs = [
            Document(id="1", content="Machine learning algorithms process data"),
            Document(id="2", content="Neural networks learn patterns from data"),
            Document(id="3", content="Deep learning models require training data"),
        ]
        store.write_documents(docs)
        
        return str(tmp_path / "chroma")
    
    def test_end_to_end_prf(self, temp_chroma_store, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text(f"""
vectorstore:
  persist_path: {temp_chroma_store}
  collection_name: test_leaves
retrieval:
  enable_hybrid: true
  prf_docs: 3
  prf_terms: 4
""")
        
        agent = BasicPRFAgent(config_path=str(config_file))
        
        inp = PRFInput(query="data processing")
        output = agent.compute(inp)
        
        assert len(output.prf_terms) > 0
        assert "data processing" in output.augmented_query
```

### Test Commands

```bash
# Run all PRF tests
pytest test_prf_basic_agent.py -v

# Run with coverage
pytest test_prf_basic_agent.py --cov=prf_basic_agent --cov-report=html

# Run specific test class
pytest test_prf_basic_agent.py::TestTermExtraction -v
```

---

## Recommendations and Improvements

### Critical Issues

#### 1. No Stopword Filtering

**Problem:** Common words like "the", "and", "for" can dominate term frequencies.

**Current:**
```python
toks = [t for t in _TOKEN.findall(...) if len(t) > 2]
```

**Recommendation:** Add stopword filtering:

```python
STOPWORDS = {
    "the", "and", "for", "are", "but", "not", "you", "all",
    "can", "her", "was", "one", "our", "out", "has", "have",
    "been", "were", "being", "their", "this", "that", "with",
    "from", "they", "will", "would", "there", "what", "about",
}

def _prf_expand_terms(query, bm25_docs, max_docs, max_terms):
    # ... tokenization ...
    toks = [t for t in _TOKEN.findall(...) if len(t) > 2 and t not in STOPWORDS]
```

#### 2. Query Terms Not Excluded

**Problem:** PRF terms may include words already in the query.

**Recommendation:** Filter query terms:

```python
def _prf_expand_terms(query, bm25_docs, max_docs, max_terms):
    query_terms = set(_TOKEN.findall(query.lower()))
    
    # ... term extraction ...
    
    # Filter out query terms
    mix = [t for t in mix if t not in query_terms and 
           not any(qt in t for qt in query_terms)]
```

---

### High Priority Improvements

#### 3. Configurable Stopwords

**Problem:** Stopwords are not configurable.

**Recommendation:** Add to configuration:

```yaml
retrieval:
  prf_stopwords:
    - the
    - and
    - for
  prf_min_token_length: 3
```

```python
@dataclass
class PRFRetrievalConfig:
    # ... existing fields ...
    prf_stopwords: List[str] = field(default_factory=list)
    prf_min_token_length: int = 3
```

#### 4. Logging and Observability

**Problem:** No visibility into PRF operations.

**Recommendation:** Add structured logging:

```python
import logging
logger = logging.getLogger(__name__)

def compute(self, inp: PRFInput) -> PRFOutput:
    logger.info(
        "prf_start",
        extra={
            "query": inp.query,
            "bm25_enabled": self._bm25 is not None,
        }
    )
    
    # ... PRF logic ...
    
    logger.info(
        "prf_complete",
        extra={
            "query": inp.query,
            "num_bm25_hits": len(bm25_hits),
            "num_prf_terms": len(prf_terms),
            "prf_terms": prf_terms,
            "augmented_query": augmented_query,
        }
    )
```

#### 5. Term Weighting

**Problem:** All PRF terms added with equal weight.

**Recommendation:** Add term importance weights:

```python
@dataclass
class PRFOutput:
    prf_terms: List[str]
    augmented_query: str
    term_weights: Dict[str, float] = None  # NEW

def _prf_expand_terms_weighted(...) -> Tuple[List[str], Dict[str, float]]:
    # ... term extraction ...
    
    total_count = sum(uni.values())
    weights = {term: count / total_count for term, count in uni.most_common(max_terms)}
    
    return terms, weights
```

#### 6. Lazy Index Building

**Problem:** BM25 index built at initialization, even if PRF never used.

**Recommendation:** Build index on first use:

```python
def compute(self, inp: PRFInput) -> PRFOutput:
    if self._bm25 is None and self._cfg.enable_hybrid:
        self._build_bm25_index()  # Lazy initialization
    
    # ... rest of compute
```

---

### Medium Priority Improvements

#### 7. TF-IDF Based Term Selection

**Problem:** Simple frequency counting doesn't account for document specificity.

**Recommendation:** Use TF-IDF:

```python
from sklearn.feature_extraction.text import TfidfVectorizer

def _prf_expand_terms_tfidf(query, bm25_docs, max_docs, max_terms):
    texts = [doc.content for doc in bm25_docs[:max_docs]]
    
    vectorizer = TfidfVectorizer(
        max_features=max_terms * 2,
        stop_words='english',
        ngram_range=(1, 2)
    )
    
    tfidf_matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()
    
    # Get top terms by average TF-IDF score
    avg_scores = tfidf_matrix.mean(axis=0).A1
    top_indices = avg_scores.argsort()[-max_terms:][::-1]
    
    return [feature_names[i] for i in top_indices]
```

#### 8. Caching PRF Results

**Problem:** Same query always recomputes PRF.

**Recommendation:** Add caching:

```python
from functools import lru_cache

class BasicPRFAgent:
    @lru_cache(maxsize=256)
    def _compute_cached(self, query: str) -> Tuple[Tuple[str, ...], str]:
        # ... PRF computation ...
        return tuple(prf_terms), augmented_query
    
    def compute(self, inp: PRFInput) -> PRFOutput:
        prf_terms, augmented_query = self._compute_cached(inp.query)
        return PRFOutput(prf_terms=list(prf_terms), augmented_query=augmented_query)
```

#### 9. Negative Feedback Support

**Problem:** PRF assumes top results are relevant (may not be true).

**Recommendation:** Support negative feedback:

```python
def compute(self, inp: PRFInput) -> PRFOutput:
    # Get more docs than needed
    all_hits = self._bm25.run(query=inp.query, top_k=max_docs * 2)
    
    # Use top docs as positive, bottom docs as negative
    positive_docs = all_hits[:max_docs]
    negative_docs = all_hits[-max_docs:]
    
    positive_terms = self._extract_terms(positive_docs)
    negative_terms = self._extract_terms(negative_docs)
    
    # Remove terms that appear frequently in negative docs
    prf_terms = [t for t in positive_terms if t not in negative_terms]
```

---

### Low Priority / Future Enhancements

#### 10. Multi-Language Support

**Recommendation:** Language-specific tokenization and stopwords:

```python
class BasicPRFAgent:
    def __init__(self, config_path=None, language="en"):
        self.language = language
        self.stopwords = self._load_stopwords(language)
        self.tokenizer = self._get_tokenizer(language)
```

#### 11. Semantic PRF

**Recommendation:** Use embeddings for semantic term expansion:

```python
def _semantic_prf(query, docs, embedder):
    """Use embeddings to find semantically similar terms."""
    # Embed query and document terms
    # Find terms with high similarity to query
    # Add those as PRF terms
```

#### 12. PRF Effectiveness Metrics

**Recommendation:** Track PRF impact on retrieval:

```python
@dataclass
class PRFMetrics:
    original_recall: float
    augmented_recall: float
    improvement: float
    terms_that_helped: List[str]
```

---

## Usage Examples

### Basic Usage

```python
from prf_basic_agent import BasicPRFAgent
from core.schemas import PRFInput

# Initialize agent
agent = BasicPRFAgent(config_path="config.fast.yaml")

# Create input
inp = PRFInput(query="machine learning algorithms")

# Compute PRF
output = agent.compute(inp)

print(f"Original: {inp.query}")
print(f"PRF Terms: {output.prf_terms}")
print(f"Augmented: {output.augmented_query}")

# Output:
# Original: machine learning algorithms
# PRF Terms: ['neural', 'network', 'training', 'data', 'neural network']
# Augmented: machine learning algorithms neural network training data neural network
```

### Pipeline Integration

```python
class RAGPipeline:
    def __init__(self, config_path: str):
        self.prf_agent = BasicPRFAgent(config_path=config_path)
        self.retriever = HybridRetrievalAgent(config_path=config_path)
    
    def process(self, query: str, plan: Plan) -> RetrieverOutput:
        # Step 1: Compute PRF if enabled
        prf_query = query
        if plan.use_prf:
            prf_output = self.prf_agent.compute(PRFInput(query=query))
            prf_query = prf_output.augmented_query
        
        # Step 2: Retrieve with augmented query
        retriever_input = RetrieverInput(
            query=query,
            prf_augmented_query=prf_query,
            plan=plan
        )
        
        return self.retriever.retrieve(retriever_input)
```

### With BM25 Configuration

```python
from core.schemas import PRFInput, BM25Config

# Limit PRF to top 5 BM25 results
bm25_config = BM25Config(top_k=5)

inp = PRFInput(
    query="deep learning frameworks",
    bm25_config=bm25_config
)

output = agent.compute(inp)
```

### Disabled PRF

```yaml
# config.fast.yaml
retrieval:
  enable_hybrid: false
```

```python
agent = BasicPRFAgent(config_path="config.fast.yaml")
output = agent.compute(PRFInput(query="test"))

# When disabled, returns original query unchanged
assert output.augmented_query == "test"
assert output.prf_terms == []
```

---

## Appendix

### Glossary

| Term | Definition |
|------|------------|
| **PRF** | Pseudo-Relevance Feedback |
| **BM25** | Best Matching 25 - lexical retrieval algorithm |
| **Unigram** | Single word token |
| **Bigram** | Two consecutive word tokens |
| **Blind Feedback** | Assuming top results are relevant without confirmation |

### Configuration Reference

| Config Path | Type | Default | Description |
|-------------|------|---------|-------------|
| `vectorstore.persist_path` | str | `"./chroma_db"` | ChromaDB path |
| `vectorstore.collection_name` | str | `"leaves"` | Collection name |
| `retrieval.enable_hybrid` | bool | `true` | Enable PRF |
| `retrieval.bm25_top_k` | int | `200` | BM25 retrieval limit |
| `retrieval.prf_docs` | int | `10` | Docs for PRF |
| `retrieval.prf_terms` | int | `6` | Terms to extract |

### Algorithm Parameters

| Parameter | Range | Effect |
|-----------|-------|--------|
| `prf_docs` | 5-20 | More docs = more diverse terms, but more noise |
| `prf_terms` | 4-10 | More terms = more recall, but query dilution |
| Token length | > 2 | Filters short tokens (configurable) |

### Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | Initial | Basic PRF with unigram/bigram extraction |

### References

- Radiant Repository: https://github.com/dshipley71/radiant
- Related Files: `retriever_haystack_agent.py`, `qe_llm_agent.py`, `config.fast.yaml`
- Academic: Rocchio (1971), "Relevance Feedback in Information Retrieval"

---

*Document generated for the Radiant RAG Pipeline project. For questions or contributions, please refer to the project repository.*
