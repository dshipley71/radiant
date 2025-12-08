# BasicRerankAgent Documentation

## Technical Reference for the Radiant RAG Pipeline Cross-Encoder Reranking

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture Context](#architecture-context)
3. [Class Structure](#class-structure)
4. [Core Functionality](#core-functionality)
5. [Text Selection Algorithm](#text-selection-algorithm)
6. [Configuration System](#configuration-system)
7. [Data Flow](#data-flow)
8. [Testing Strategies](#testing-strategies)
9. [Recommendations and Improvements](#recommendations-and-improvements)
10. [Usage Examples](#usage-examples)

---

## Overview

### Purpose

The `BasicRerankAgent` is the reranking component within the Radiant RAG pipeline. It uses a cross-encoder model (SentenceTransformers) to reorder retrieved documents based on their relevance to the query, providing more accurate relevance scores than initial bi-encoder retrieval.

### Key Responsibilities

- Load and initialize cross-encoder reranking model
- Select best representative text for each parent document
- Run cross-encoder scoring against the user query
- Aggregate scores per parent and reorder results
- Propagate reranker scores back to snippets for downstream use

### Design Philosophy

The agent implements a **two-stage retrieval** approach where initial retrieval (bi-encoder) provides recall, and reranking (cross-encoder) provides precision. The text selection algorithm prioritizes image captions and summaries, optimizing for document types with visual content.

---

## Architecture Context

### Position in the Radiant Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    RetrievalResults                             │
│  [Parent1: score=0.7, Parent2: score=0.8, Parent3: score=0.6]  │
│  (Initial bi-encoder scores)                                   │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    BasicRerankAgent                             │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  1. Text Selection                                      │   │
│  │     └─ Choose best text per parent (caption > summary)  │   │
│  ├─────────────────────────────────────────────────────────┤   │
│  │  2. Cross-Encoder Scoring                               │   │
│  │     └─ Score each (query, text) pair                    │   │
│  ├─────────────────────────────────────────────────────────┤   │
│  │  3. Score Aggregation                                   │   │
│  │     └─ Best score per parent_id                         │   │
│  ├─────────────────────────────────────────────────────────┤   │
│  │  4. Reorder & Propagate                                 │   │
│  │     └─ Sort by score, update snippet.score              │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Reranked Results                             │
│  [Parent2: score=0.92, Parent1: score=0.85, Parent3: score=0.71]│
│  (Cross-encoder scores, new ordering)                          │
└─────────────────────────────────────────────────────────────────┘
```

### Bi-Encoder vs Cross-Encoder

```
┌─────────────────────────────────────────────────────────────────┐
│                    Retrieval Stages                             │
├─────────────────────┬───────────────────────────────────────────┤
│    Bi-Encoder       │        Cross-Encoder                      │
│    (Retrieval)      │        (Reranking)                        │
├─────────────────────┼───────────────────────────────────────────┤
│ Encodes query and   │ Encodes query AND document together       │
│ docs independently  │                                           │
├─────────────────────┼───────────────────────────────────────────┤
│ Fast (O(1) lookup)  │ Slow (O(n) for n docs)                    │
├─────────────────────┼───────────────────────────────────────────┤
│ Good recall         │ Better precision                          │
├─────────────────────┼───────────────────────────────────────────┤
│ ~100-1000 docs      │ ~10-100 docs (after filtering)            │
├─────────────────────┼───────────────────────────────────────────┤
│ Use: Initial filter │ Use: Final ranking                        │
└─────────────────────┴───────────────────────────────────────────┘
```

### Related Components

| Component | Relationship |
|-----------|--------------|
| `RerankAgent` | Abstract base class (from `core.interfaces`) |
| `RerankInput` | Input schema with query and results |
| `RerankOutput` | Output schema with reranked results |
| `RetrievalResult` | Parent-level document with snippets |
| `SentenceTransformersSimilarityRanker` | Haystack cross-encoder |

---

## Class Structure

### Inheritance

```python
class BasicRerankAgent(RerankAgent):
    """Cross-encoder-based rerank agent using SentenceTransformers."""
```

### Class Attributes

| Attribute | Type | Value | Description |
|-----------|------|-------|-------------|
| `role` | `str` | `"rerank"` | Agent role identifier |

### Instance Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `_config_path` | `str` | Path to configuration file |
| `_ranker` | `SentenceTransformersSimilarityRanker` | Cross-encoder model (may be None) |

### Constructor

```python
def __init__(self, config_path: Optional[str] = None) -> None
```

**Initialization:**
1. Store config path
2. Initialize cross-encoder ranker

### Methods

| Method | Visibility | Purpose |
|--------|------------|---------|
| `name` | Property | Returns agent name |
| `describe()` | Public | Returns agent description |
| `rerank(inp)` | Public | Main reranking method |
| `_init_ranker()` | Private | Initialize cross-encoder |
| `_best_text_for_parent(result)` | Static | Select best text for parent |
| `_build_documents_for_rerank(results)` | Class | Build pseudo-documents |
| `_aggregate_parent_scores(docs)` | Static | Aggregate scores per parent |

---

## Core Functionality

### The `rerank()` Method

Primary method that reorders retrieval results.

**Signature:**
```python
def rerank(self, inp: RerankInput) -> RerankOutput
```

**Parameters:**
- `inp` (`RerankInput`): Contains query and retrieval results

**Returns:**
- `RerankOutput`: Contains reranked results

**Processing Steps:**

1. **Validation**
   - Check if results exist
   - Check if ranker is initialized
   - Return original order if either fails

2. **Build Documents**
   - For each parent, select best representative text
   - Create Haystack Document objects
   - Skip parents with no usable text

3. **Run Cross-Encoder**
   - Call ranker with query and documents
   - Handle errors gracefully (fallback to original)

4. **Aggregate Scores**
   - Extract best score per parent_id
   - Handle missing/invalid scores

5. **Propagate Scores**
   - Update snippet.score for all snippets of each parent
   - Ensures downstream components see cross-encoder scores

6. **Sort Results**
   - Order by aggregated score (descending)
   - Return reranked results

### The `_init_ranker()` Method

Initializes the cross-encoder model.

**Process:**
1. Load configuration
2. Extract model ID, device, top_k
3. Create `SentenceTransformersSimilarityRanker`
4. Warm up model (reduce first-call latency)
5. Set `_ranker = None` on any failure (graceful degradation)

---

## Text Selection Algorithm

### Priority Order

The agent selects the best text to represent each parent for cross-encoder scoring:

```
┌─────────────────────────────────────────────────────────────────┐
│                    Text Selection Priority                      │
├─────────────────────────────────────────────────────────────────┤
│  1. parent_metadata["display_summary"]                         │
│     └─ Often contains vision_caption already                    │
│                                                                 │
│  2. parent_metadata["vision_caption"]                          │
│     └─ Direct image caption from vision model                   │
│                                                                 │
│  3. parent_metadata["summary_leaf"]                            │
│     parent_metadata["summary_parent"]                          │
│     parent_metadata["summary"]                                 │
│     └─ Various summary fields (first non-empty wins)           │
│                                                                 │
│  4. result.snippets[0].text                                    │
│     └─ Top snippet text (sorted by retriever score)            │
│                                                                 │
│  5. parent_metadata["title"]                                   │
│     └─ Document title as last resort                           │
└─────────────────────────────────────────────────────────────────┘
```

### Rationale

1. **display_summary**: Pre-composed summary that combines multiple signals
2. **vision_caption**: Critical for image-heavy documents where text extraction fails
3. **summary_***: Condensed document content
4. **snippet.text**: Actual retrieved chunk content
5. **title**: Minimal fallback for sparse documents

### Implementation

```python
@staticmethod
def _best_text_for_parent(result: RetrievalResult) -> str:
    meta = result.parent_metadata or {}
    text = ""
    
    # Priority 1: display_summary
    ds = meta.get("display_summary")
    if isinstance(ds, str) and ds.strip():
        text = ds.strip()
    
    # Priority 2: vision_caption
    if not text:
        vc = meta.get("vision_caption")
        if isinstance(vc, str) and vc.strip():
            text = vc.strip()
    
    # Priority 3: summary fields
    if not text:
        for key in ("summary_leaf", "summary_parent", "summary"):
            val = meta.get(key)
            if isinstance(val, str) and val.strip():
                text = val.strip()
                break
    
    # Priority 4: best snippet
    if not text and result.snippets:
        if isinstance(result.snippets[0].text, str):
            text = result.snippets[0].text.strip()
    
    # Priority 5: title
    if not text:
        title = meta.get("title")
        if isinstance(title, str) and title.strip():
            text = title.strip()
    
    return text
```

---

## Configuration System

### Configuration File: `config.fast.yaml`

```yaml
retrieval:
  rerank_model: cross-encoder/ms-marco-MiniLM-L-6-v2
  rerank_device: cpu
  rerank_top_k: 100
```

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `rerank_model` | str | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder model ID |
| `rerank_device` | str | `cpu` | Computation device |
| `rerank_top_k` | int | `100` | Max documents to rerank |

### Supported Devices

| Device | Value | Notes |
|--------|-------|-------|
| CPU | `"cpu"` | Default, always available |
| GPU | `"cuda"`, `"cuda:0"` | Requires CUDA |
| MPS | `"mps"` | Apple Silicon |

### Configuration Resolution Order

```
1. Explicit config_path parameter
         │
         ▼
2. $AGENTIC_RAG_CONFIG environment variable
         │
         ▼
3. ./config.fast.yaml (current working directory)
         │
         ▼
4. Default values (if file missing)
```

### Recommended Cross-Encoder Models

| Model | Size | Speed | Quality |
|-------|------|-------|---------|
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | 23M | Fast | Good |
| `cross-encoder/ms-marco-MiniLM-L-12-v2` | 33M | Medium | Better |
| `cross-encoder/ms-marco-TinyBERT-L-2-v2` | 4.4M | Fastest | Lower |
| `BAAI/bge-reranker-base` | 278M | Slow | Best |

---

## Data Flow

### Input Schema: `RerankInput`

```python
@dataclass
class RerankInput:
    query: str                       # User query for scoring
    results: List[RetrievalResult]   # Results from retriever
```

### Output Schema: `RerankOutput`

```python
@dataclass
class RerankOutput:
    results: List[RetrievalResult]  # Reranked results
```

### RetrievalResult Structure

```python
@dataclass
class RetrievalResult:
    doc_id: str                      # Parent document ID
    parent_metadata: Dict[str, Any]  # Parent-level metadata
    snippets: List[Snippet]          # Child chunks/snippets
```

### Score Propagation

```
Before Reranking:
  Parent1 (doc_id="p1")
    └─ Snippet1 (score=0.7)
    └─ Snippet2 (score=0.65)
  Parent2 (doc_id="p2")
    └─ Snippet3 (score=0.8)

After Reranking:
  Parent2 (doc_id="p2")            ← Moved up (higher cross-encoder score)
    └─ Snippet3 (score=0.92)      ← Score updated to cross-encoder score
  Parent1 (doc_id="p1")
    └─ Snippet1 (score=0.85)      ← Score updated
    └─ Snippet2 (score=0.85)      ← All snippets get same parent score
```

---

## Testing Strategies

### Unit Tests

#### 1. Initialization Tests

```python
import pytest
from unittest.mock import Mock, patch, MagicMock
from rerank_basic_agent import BasicRerankAgent, _load_config

class TestInitialization:
    
    @patch('rerank_basic_agent._load_config')
    @patch('rerank_basic_agent.SentenceTransformersSimilarityRanker')
    def test_init_with_config(self, mock_ranker_class, mock_load_config):
        mock_load_config.return_value = {
            "retrieval": {
                "rerank_model": "test-model",
                "rerank_device": "cuda",
                "rerank_top_k": 50
            }
        }
        mock_ranker = Mock()
        mock_ranker_class.return_value = mock_ranker
        
        agent = BasicRerankAgent(config_path="test.yaml")
        
        mock_ranker_class.assert_called_once()
        call_kwargs = mock_ranker_class.call_args[1]
        assert call_kwargs["model"] == "test-model"
        assert call_kwargs["top_k"] == 50
    
    @patch('rerank_basic_agent._load_config')
    @patch('rerank_basic_agent.SentenceTransformersSimilarityRanker')
    def test_init_with_defaults(self, mock_ranker_class, mock_load_config):
        mock_load_config.return_value = {}
        mock_ranker = Mock()
        mock_ranker_class.return_value = mock_ranker
        
        agent = BasicRerankAgent()
        
        call_kwargs = mock_ranker_class.call_args[1]
        assert call_kwargs["model"] == "cross-encoder/ms-marco-MiniLM-L-6-v2"
        assert call_kwargs["top_k"] == 100
    
    @patch('rerank_basic_agent._load_config')
    @patch('rerank_basic_agent.SentenceTransformersSimilarityRanker')
    def test_init_failure_sets_ranker_none(self, mock_ranker_class, mock_load_config):
        mock_load_config.return_value = {}
        mock_ranker_class.side_effect = Exception("Model load failed")
        
        agent = BasicRerankAgent()
        
        assert agent._ranker is None
```

#### 2. Text Selection Tests

```python
from rerank_basic_agent import BasicRerankAgent
from core.schemas import RetrievalResult, Snippet

class TestTextSelection:
    
    def test_priority_display_summary(self):
        result = RetrievalResult(
            doc_id="p1",
            parent_metadata={
                "display_summary": "Display summary text",
                "vision_caption": "Vision caption",
                "title": "Title"
            },
            snippets=[Snippet(chunk_id="c1", text="Snippet text", score=0.5)]
        )
        
        text = BasicRerankAgent._best_text_for_parent(result)
        assert text == "Display summary text"
    
    def test_priority_vision_caption(self):
        result = RetrievalResult(
            doc_id="p1",
            parent_metadata={
                "vision_caption": "Image shows a diagram",
                "title": "Title"
            },
            snippets=[Snippet(chunk_id="c1", text="Snippet", score=0.5)]
        )
        
        text = BasicRerankAgent._best_text_for_parent(result)
        assert text == "Image shows a diagram"
    
    def test_priority_summary_fields(self):
        result = RetrievalResult(
            doc_id="p1",
            parent_metadata={
                "summary_leaf": "Leaf summary",
                "summary_parent": "Parent summary",
                "summary": "Generic summary"
            },
            snippets=[]
        )
        
        text = BasicRerankAgent._best_text_for_parent(result)
        assert text == "Leaf summary"  # First in priority
    
    def test_priority_snippet_text(self):
        result = RetrievalResult(
            doc_id="p1",
            parent_metadata={"title": "Title"},
            snippets=[
                Snippet(chunk_id="c1", text="Best snippet text", score=0.9),
                Snippet(chunk_id="c2", text="Second snippet", score=0.7)
            ]
        )
        
        text = BasicRerankAgent._best_text_for_parent(result)
        assert text == "Best snippet text"
    
    def test_priority_title_fallback(self):
        result = RetrievalResult(
            doc_id="p1",
            parent_metadata={"title": "Document Title"},
            snippets=[]
        )
        
        text = BasicRerankAgent._best_text_for_parent(result)
        assert text == "Document Title"
    
    def test_empty_metadata_returns_empty(self):
        result = RetrievalResult(
            doc_id="p1",
            parent_metadata={},
            snippets=[]
        )
        
        text = BasicRerankAgent._best_text_for_parent(result)
        assert text == ""
    
    def test_whitespace_only_skipped(self):
        result = RetrievalResult(
            doc_id="p1",
            parent_metadata={
                "display_summary": "   ",
                "vision_caption": "\n\t",
                "title": "Actual Title"
            },
            snippets=[]
        )
        
        text = BasicRerankAgent._best_text_for_parent(result)
        assert text == "Actual Title"
```

#### 3. Reranking Tests

```python
class TestReranking:
    
    @pytest.fixture
    def agent_with_ranker(self):
        agent = BasicRerankAgent.__new__(BasicRerankAgent)
        agent._ranker = Mock()
        return agent
    
    def test_empty_results_returns_empty(self, agent_with_ranker):
        inp = RerankInput(query="test", results=[])
        output = agent_with_ranker.rerank(inp)
        
        assert output.results == []
        agent_with_ranker._ranker.run.assert_not_called()
    
    def test_no_ranker_returns_original(self):
        agent = BasicRerankAgent.__new__(BasicRerankAgent)
        agent._ranker = None
        
        results = [
            RetrievalResult(doc_id="p1", parent_metadata={"title": "A"}, snippets=[])
        ]
        inp = RerankInput(query="test", results=results)
        output = agent.rerank(inp)
        
        assert output.results == results
    
    def test_successful_reranking(self, agent_with_ranker):
        from haystack.dataclasses import Document
        
        results = [
            RetrievalResult(
                doc_id="p1",
                parent_metadata={"title": "Doc 1"},
                snippets=[Snippet(chunk_id="c1", text="text", score=0.5)]
            ),
            RetrievalResult(
                doc_id="p2",
                parent_metadata={"title": "Doc 2"},
                snippets=[Snippet(chunk_id="c2", text="text", score=0.6)]
            ),
        ]
        
        # Mock ranker returns p2 with higher score
        reranked_docs = [
            Document(id="p2", content="Doc 2", meta={"parent_id": "p2"}, score=0.9),
            Document(id="p1", content="Doc 1", meta={"parent_id": "p1"}, score=0.7),
        ]
        agent_with_ranker._ranker.run.return_value = {"documents": reranked_docs}
        
        inp = RerankInput(query="test query", results=results)
        output = agent_with_ranker.rerank(inp)
        
        # p2 should be first (higher score)
        assert output.results[0].doc_id == "p2"
        assert output.results[1].doc_id == "p1"
    
    def test_score_propagation_to_snippets(self, agent_with_ranker):
        from haystack.dataclasses import Document
        
        results = [
            RetrievalResult(
                doc_id="p1",
                parent_metadata={"title": "Doc 1"},
                snippets=[
                    Snippet(chunk_id="c1", text="a", score=0.5),
                    Snippet(chunk_id="c2", text="b", score=0.4),
                ]
            ),
        ]
        
        reranked_docs = [
            Document(id="p1", content="Doc 1", meta={"parent_id": "p1"}, score=0.85),
        ]
        agent_with_ranker._ranker.run.return_value = {"documents": reranked_docs}
        
        inp = RerankInput(query="test", results=results)
        output = agent_with_ranker.rerank(inp)
        
        # All snippets should have cross-encoder score
        assert output.results[0].snippets[0].score == 0.85
        assert output.results[0].snippets[1].score == 0.85
    
    def test_ranker_exception_returns_original(self, agent_with_ranker):
        agent_with_ranker._ranker.run.side_effect = Exception("Ranker failed")
        
        results = [
            RetrievalResult(doc_id="p1", parent_metadata={"title": "A"}, snippets=[])
        ]
        inp = RerankInput(query="test", results=results)
        output = agent_with_ranker.rerank(inp)
        
        assert output.results == results  # Original order preserved
```

#### 4. Document Building Tests

```python
class TestDocumentBuilding:
    
    def test_builds_documents_from_results(self):
        results = [
            RetrievalResult(
                doc_id="p1",
                parent_metadata={"title": "Doc 1"},
                snippets=[]
            ),
            RetrievalResult(
                doc_id="p2",
                parent_metadata={"title": "Doc 2"},
                snippets=[]
            ),
        ]
        
        docs = BasicRerankAgent._build_documents_for_rerank(results)
        
        assert len(docs) == 2
        assert docs[0].id == "p1"
        assert docs[0].content == "Doc 1"
        assert docs[0].meta["parent_id"] == "p1"
    
    def test_skips_results_with_no_text(self):
        results = [
            RetrievalResult(
                doc_id="p1",
                parent_metadata={"title": "Has Text"},
                snippets=[]
            ),
            RetrievalResult(
                doc_id="p2",
                parent_metadata={},  # No text available
                snippets=[]
            ),
        ]
        
        docs = BasicRerankAgent._build_documents_for_rerank(results)
        
        assert len(docs) == 1
        assert docs[0].id == "p1"
```

#### 5. Score Aggregation Tests

```python
class TestScoreAggregation:
    
    def test_aggregates_best_score(self):
        from haystack.dataclasses import Document
        
        docs = [
            Document(id="d1", content="", meta={"parent_id": "p1"}, score=0.7),
            Document(id="d2", content="", meta={"parent_id": "p1"}, score=0.9),
            Document(id="d3", content="", meta={"parent_id": "p2"}, score=0.8),
        ]
        # Set scores manually since Document might not accept score in constructor
        for doc in docs:
            doc.score = float(doc.id.replace("d", "")) * 0.1 + 0.6
        docs[0].score = 0.7
        docs[1].score = 0.9
        docs[2].score = 0.8
        
        scores = BasicRerankAgent._aggregate_parent_scores(docs)
        
        assert scores["p1"] == 0.9  # Best of 0.7 and 0.9
        assert scores["p2"] == 0.8
    
    def test_handles_missing_scores(self):
        from haystack.dataclasses import Document
        
        docs = [
            Document(id="d1", content="", meta={"parent_id": "p1"}),
        ]
        # No score set
        
        scores = BasicRerankAgent._aggregate_parent_scores(docs)
        
        assert "p1" not in scores  # Skipped due to missing score
```

#### 6. Agent Interface Tests

```python
class TestAgentInterface:
    
    @patch('rerank_basic_agent.SentenceTransformersSimilarityRanker')
    @patch('rerank_basic_agent._load_config')
    def test_name_property(self, mock_config, mock_ranker):
        mock_config.return_value = {}
        agent = BasicRerankAgent()
        assert agent.name == "BasicRerankAgent"
    
    @patch('rerank_basic_agent.SentenceTransformersSimilarityRanker')
    @patch('rerank_basic_agent._load_config')
    def test_describe_method(self, mock_config, mock_ranker):
        mock_config.return_value = {}
        agent = BasicRerankAgent()
        description = agent.describe()
        
        assert isinstance(description, str)
        assert "cross-encoder" in description.lower() or "rerank" in description.lower()
    
    @patch('rerank_basic_agent.SentenceTransformersSimilarityRanker')
    @patch('rerank_basic_agent._load_config')
    def test_role_attribute(self, mock_config, mock_ranker):
        mock_config.return_value = {}
        agent = BasicRerankAgent()
        assert agent.role == "rerank"
```

### Test Commands

```bash
# Run all rerank tests
pytest test_rerank_basic_agent.py -v

# Run with coverage
pytest test_rerank_basic_agent.py --cov=rerank_basic_agent --cov-report=html

# Run specific test class
pytest test_rerank_basic_agent.py::TestTextSelection -v
```

---

## Recommendations and Improvements

### Critical Issues

#### 1. Silent Failure on Ranker Init

**Problem:** When ranker initialization fails, agent silently falls back to no-op.

**Recommendation:** Add logging:

```python
import logging
logger = logging.getLogger(__name__)

def _init_ranker(self) -> None:
    try:
        self._ranker = SentenceTransformersSimilarityRanker(...)
    except Exception as e:
        logger.error(f"Failed to initialize reranker: {e}")
        logger.warning("Reranking will be disabled")
        self._ranker = None
```

#### 2. No Reranking Status Indicator

**Problem:** Downstream components don't know if reranking actually happened.

**Recommendation:** Add status to output:

```python
@dataclass
class RerankOutput:
    results: List[RetrievalResult]
    was_reranked: bool = True
    model_used: Optional[str] = None

def rerank(self, inp: RerankInput) -> RerankOutput:
    if self._ranker is None:
        return RerankOutput(results=results, was_reranked=False)
    
    # ... reranking ...
    
    return RerankOutput(
        results=sorted_results,
        was_reranked=True,
        model_used=self._model_id
    )
```

---

### High Priority Improvements

#### 3. Batch Processing for Large Result Sets

**Problem:** All documents processed in single call.

**Recommendation:** Add batching:

```python
def rerank(self, inp: RerankInput) -> RerankOutput:
    batch_size = 32
    all_reranked = []
    
    for i in range(0, len(docs), batch_size):
        batch = docs[i:i + batch_size]
        reranked_batch = self._ranker.run(query=inp.query, documents=batch)
        all_reranked.extend(reranked_batch["documents"])
```

#### 4. Logging and Observability

**Problem:** No visibility into reranking operations.

**Recommendation:** Add structured logging:

```python
def rerank(self, inp: RerankInput) -> RerankOutput:
    logger.info(
        "rerank_start",
        extra={
            "query_length": len(inp.query),
            "num_results": len(results),
            "ranker_available": self._ranker is not None,
        }
    )
    
    # ... reranking ...
    
    logger.info(
        "rerank_complete",
        extra={
            "num_reranked": len(sorted_results),
            "top_score": max(parent_scores.values()) if parent_scores else None,
            "elapsed_ms": elapsed_ms,
        }
    )
```

#### 5. Configurable Text Selection

**Problem:** Text selection priority is hardcoded.

**Recommendation:** Make configurable:

```yaml
retrieval:
  rerank_text_priority:
    - display_summary
    - vision_caption
    - summary_leaf
    - snippet_text
    - title
```

```python
def _best_text_for_parent(self, result: RetrievalResult) -> str:
    for field in self._text_priority:
        text = self._extract_text_field(result, field)
        if text:
            return text
    return ""
```

---

### Medium Priority Improvements

#### 6. Score Normalization

**Problem:** Cross-encoder scores may not be normalized (0-1).

**Recommendation:** Add normalization:

```python
def _normalize_scores(self, docs: List[Document]) -> List[Document]:
    scores = [d.score for d in docs if d.score is not None]
    if not scores:
        return docs
    
    min_score, max_score = min(scores), max(scores)
    if max_score == min_score:
        return docs
    
    for d in docs:
        if d.score is not None:
            d.score = (d.score - min_score) / (max_score - min_score)
    
    return docs
```

#### 7. Caching Rerank Results

**Problem:** Same query/docs always recomputes.

**Recommendation:** Add caching:

```python
from functools import lru_cache
import hashlib

def _cache_key(self, query: str, doc_ids: List[str]) -> str:
    content = f"{query}|{'|'.join(sorted(doc_ids))}"
    return hashlib.md5(content.encode()).hexdigest()

# Cache at method level or use external cache
```

#### 8. Multi-Model Ensemble

**Problem:** Single model may have blind spots.

**Recommendation:** Support ensemble:

```python
class EnsembleRerankAgent(RerankAgent):
    def __init__(self, models: List[str], weights: List[float]):
        self.rankers = [self._init_ranker(m) for m in models]
        self.weights = weights
    
    def rerank(self, inp: RerankInput) -> RerankOutput:
        all_scores = []
        for ranker, weight in zip(self.rankers, self.weights):
            scores = ranker.run(...)
            all_scores.append({d.id: d.score * weight for d in scores})
        
        # Combine scores
        combined = self._combine_scores(all_scores)
```

---

### Low Priority / Future Enhancements

#### 9. Diversity-Aware Reranking

**Recommendation:** Add diversity to prevent redundancy:

```python
def rerank_with_diversity(self, inp: RerankInput, lambda_param: float = 0.5):
    """MMR-style reranking for diversity."""
    # Maximal Marginal Relevance
```

#### 10. Learned Text Selection

**Recommendation:** Use ML to select best text:

```python
def _best_text_for_parent_learned(self, result: RetrievalResult) -> str:
    """Use a small model to select most relevant text field."""
    candidates = self._get_all_text_candidates(result)
    if len(candidates) == 1:
        return candidates[0]
    
    # Score each candidate against query
    scores = self._score_candidates(candidates)
    return max(zip(candidates, scores), key=lambda x: x[1])[0]
```

#### 11. Async Reranking

**Recommendation:** Support async for non-blocking:

```python
async def rerank_async(self, inp: RerankInput) -> RerankOutput:
    # For async-compatible rankers
```

---

## Usage Examples

### Basic Usage

```python
from rerank_basic_agent import BasicRerankAgent
from core.schemas import RerankInput, RetrievalResult, Snippet

# Initialize agent
agent = BasicRerankAgent(config_path="config.fast.yaml")

# Prepare results from retriever
results = [
    RetrievalResult(
        doc_id="doc1",
        parent_metadata={"title": "Introduction to ML"},
        snippets=[Snippet(chunk_id="c1", text="ML basics...", score=0.7)]
    ),
    RetrievalResult(
        doc_id="doc2",
        parent_metadata={"title": "Advanced Neural Networks"},
        snippets=[Snippet(chunk_id="c2", text="Deep learning...", score=0.6)]
    ),
]

# Rerank
inp = RerankInput(query="beginner machine learning tutorial", results=results)
output = agent.rerank(inp)

# Results now ordered by cross-encoder relevance
for r in output.results:
    print(f"{r.doc_id}: {r.snippets[0].score:.3f}")
```

### Pipeline Integration

```python
class RAGPipeline:
    def __init__(self, config_path: str):
        self.retriever = HybridRetrievalAgent(config_path=config_path)
        self.reranker = BasicRerankAgent(config_path=config_path)
    
    def process(self, query: str, plan: Plan) -> List[RetrievalResult]:
        # Step 1: Retrieve
        retriever_output = self.retriever.retrieve(RetrieverInput(
            query=query,
            plan=plan
        ))
        
        # Step 2: Rerank if enabled
        if plan.use_rerank:
            rerank_output = self.reranker.rerank(RerankInput(
                query=query,
                results=retriever_output.results
            ))
            return rerank_output.results
        
        return retriever_output.results
```

### Checking Reranker Availability

```python
agent = BasicRerankAgent()

if agent._ranker is None:
    logger.warning("Reranker not available, using original ordering")
else:
    logger.info(f"Reranker ready: {agent._ranker.model}")
```

---

## Appendix

### Glossary

| Term | Definition |
|------|------------|
| **Cross-Encoder** | Model that encodes query and document together |
| **Bi-Encoder** | Model that encodes query and documents separately |
| **Reranking** | Reordering retrieved documents by relevance |
| **Score Propagation** | Copying reranker scores to snippet objects |

### Configuration Reference

| Config Path | Type | Default | Description |
|-------------|------|---------|-------------|
| `retrieval.rerank_model` | str | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Model ID |
| `retrieval.rerank_device` | str | `cpu` | Computation device |
| `retrieval.rerank_top_k` | int | `100` | Max docs to rerank |

### Text Priority Reference

| Priority | Field | Source |
|----------|-------|--------|
| 1 | `display_summary` | parent_metadata |
| 2 | `vision_caption` | parent_metadata |
| 3 | `summary_leaf` | parent_metadata |
| 4 | `summary_parent` | parent_metadata |
| 5 | `summary` | parent_metadata |
| 6 | `snippets[0].text` | result.snippets |
| 7 | `title` | parent_metadata |

### Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | Initial | Cross-encoder reranking with text priority |

### References

- Radiant Repository: https://github.com/dshipley71/radiant
- Related Files: `retriever_haystack_agent.py`, `core/schemas.py`
- Model: [ms-marco-MiniLM-L-6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2)

---

*Document generated for the Radiant RAG Pipeline project. For questions or contributions, please refer to the project repository.*
