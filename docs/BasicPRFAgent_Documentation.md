# BasicPRFAgent Documentation

Technical reference for the Radiant RAG pipeline PRF agent.

---

## Overview

The `BasicPRFAgent` computes pseudo-relevance feedback by running BM25 retrieval and extracting expansion terms from top results.

**Module Location:** `agents/prf.py`

**Interface:** `PRFAgent` (from `core.interfaces`)

---

## Class Definition

```python
class BasicPRFAgent(PRFAgent):
    """PRF agent backed by a BM25 index over the leaf Chroma corpus."""
    
    role = "prf"
    
    def __init__(self, config_path: Optional[str] = None) -> None:
        self._cfg: PRFRetrievalConfig = _load_prf_cfg(config_path)
        self._prf_docs: int = int(self._cfg.prf_docs)
        self._prf_terms: int = int(self._cfg.prf_terms)
        self._bm25: Optional[InMemoryBM25Retriever] = None
        self._build_bm25_index()
    
    @property
    def name(self) -> str:
        return "BasicPRFAgent"
    
    def compute(self, inp: PRFInput) -> PRFOutput:
        ...
```

---

## Functionality

### Main Method: `compute()`

**Input:** `PRFInput`
- `ctx`: Request context
- `query`: User's query string
- `bm25_config`: PRF configuration

**Output:** `PRFOutput`
- `prf_terms`: List of expansion terms
- `augmented_query`: Original query + PRF terms

---

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `leaf_chroma_path` | `./chroma_db` | ChromaDB path |
| `leaf_collection` | `leaves` | Collection name |
| `enable_hybrid` | `true` | Enable PRF |
| `bm25_top_k` | `200` | BM25 candidates |
| `prf_docs` | `10` | Docs for term extraction |
| `prf_terms` | `6` | Number of terms |

---

## PRF Algorithm

1. Run BM25 on query against leaf chunks
2. Extract tokens from top `prf_docs` results
3. Compute unigram and bigram frequencies
4. Select top `prf_terms` by frequency
5. Append PRF terms to original query

---

## Related Documentation

- [CoreInterfaces_Documentation.md](CoreInterfaces_Documentation.md) - `PRFAgent` interface
- [CoreSchemas_Documentation.md](CoreSchemas_Documentation.md) - `PRFInput`, `PRFOutput` schemas
- [HybridRetrievalAgent_Documentation.md](HybridRetrievalAgent_Documentation.md) - Uses PRF output
