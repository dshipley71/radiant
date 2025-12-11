# HybridRetrievalAgent Documentation

Technical reference for the Radiant RAG pipeline retriever agent.

---

## Overview

The `HaystackChromaRetrieverAgent` performs hybrid retrieval combining dense (ChromaDB) and sparse (BM25) methods with RRF fusion and optional auto-merging.

**Module Location:** `agents/retriever.py`

**Interface:** `RetrieverAgent` (from `core.interfaces`)

---

## Class Definition

```python
class HaystackChromaRetrieverAgent(RetrieverAgent):
    """Hybrid retriever with ChromaDB, BM25, and auto-merging support."""
    
    role = "retriever"
    
    def __init__(self, config_path: Optional[str] = None) -> None:
        ...
    
    @property
    def name(self) -> str:
        return "HaystackChromaRetrieverAgent"
    
    def retrieve(self, inp: RetrieverInput) -> RetrieverOutput:
        ...
```

---

## Functionality

### Main Method: `retrieve()`

**Input:** `RetrieverInput`
- `ctx`: Request context
- `query`: User's query string
- `expanded_queries`: Optional QE variants
- `prf_augmented_query`: Optional PRF-augmented query
- `plan`: Execution plan

**Output:** `RetrieverOutput`
- `results`: List of `RetrievalResult` objects

---

## Retrieval Modes

### DUAL_INDEX Mode

1. Dense retrieval from leaf ChromaDB
2. Dense retrieval from parent ChromaDB
3. BM25 retrieval (if enabled)
4. RRF fusion of results
5. Auto-merging of leaf chunks into parent context

### LEAF_ONLY Mode

1. Dense retrieval from leaf ChromaDB only
2. BM25 retrieval (if enabled)
3. RRF fusion

---

## Configuration

```yaml
retrieval:
  leaf_chroma_path: ./chroma_db/leaves
  leaf_collection: leaves
  parent_chroma_path: ./chroma_db/parents
  parent_collection: parents
  enable_hybrid: true
  bm25_top_k: 200
  leaf_only: false
  merge_threshold: 3
```

---

## Result Caching

The agent implements an LRU cache for retrieval results:

```python
@dataclass
class QueryCache:
    max_size: int = 128
    stats: CacheStats
```

Cache key includes:
- Query text
- PRF augmented query
- Expanded queries
- Plan parameters
- Retrieval configuration

---

## Output Schema

```python
class RetrievalResult(BaseModel):
    doc_id: str
    parent_metadata: Dict[str, Any] = {}
    snippets: List[Snippet] = []

class Snippet(BaseModel):
    chunk_id: str
    score: float
    text: str
    lang: Optional[str] = None
    page: Optional[int] = None
    level: str = "leaf"
```

---

## Related Documentation

- [CoreInterfaces_Documentation.md](CoreInterfaces_Documentation.md) - `RetrieverAgent` interface
- [CoreSchemas_Documentation.md](CoreSchemas_Documentation.md) - `RetrieverInput`, `RetrieverOutput`, `RetrievalResult` schemas
- [BasicPRFAgent_Documentation.md](BasicPRFAgent_Documentation.md) - PRF augmentation
- [BasicRerankAgent_Documentation.md](BasicRerankAgent_Documentation.md) - Result reranking
