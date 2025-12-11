# BasicRerankAgent Documentation

Technical reference for the Radiant RAG pipeline rerank agent.

---

## Overview

The `BasicRerankAgent` reorders retrieval results using a cross-encoder model (SentenceTransformers) for improved precision.

**Module Location:** `agents/rerank.py`

**Interface:** `RerankAgent` (from `core.interfaces`)

---

## Class Definition

```python
class BasicRerankAgent(RerankAgent):
    """Cross-encoder-based rerank agent."""
    
    role = "rerank"
    
    def __init__(self, config_path: Optional[str] = None) -> None:
        self._config_path = config_path
        self._ranker: Optional[SentenceTransformersSimilarityRanker] = None
        self._init_ranker()
    
    @property
    def name(self) -> str:
        return "BasicRerankAgent"
    
    def rerank(self, inp: RerankInput) -> RerankOutput:
        ...
```

---

## Functionality

### Main Method: `rerank()`

**Input:** `RerankInput`
- `ctx`: Request context
- `query`: User's query string
- `results`: List of `RetrievalResult` objects
- `plan`: Execution plan

**Output:** `RerankOutput`
- `results`: Reordered `RetrievalResult` list

---

## Configuration

From `config.fast.yaml`:

```yaml
retrieval:
  rerank_model: cross-encoder/ms-marco-MiniLM-L-6-v2
  rerank_device: cpu  # or cuda
  rerank_top_k: 5
```

---

## Text Selection Priority

For each parent document, text is selected in this order:

1. `parent_metadata["display_summary"]`
2. `parent_metadata["vision_caption"]`
3. `parent_metadata["summary_leaf"]` / `["summary_parent"]` / `["summary"]`
4. Top snippet text
5. `parent_metadata["title"]`

---

## Reranking Process

1. Build pseudo-Document per parent using selected text
2. Run cross-encoder with user query
3. Aggregate scores per parent_id
4. Sort `RetrievalResult` objects by score (descending)
5. Update snippet scores for downstream reporting

---

## Related Documentation

- [CoreInterfaces_Documentation.md](CoreInterfaces_Documentation.md) - `RerankAgent` interface
- [CoreSchemas_Documentation.md](CoreSchemas_Documentation.md) - `RerankInput`, `RerankOutput` schemas
- [HybridRetrievalAgent_Documentation.md](HybridRetrievalAgent_Documentation.md) - Upstream retrieval
