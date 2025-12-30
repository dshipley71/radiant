# Semantic Chunking in Radiant RAG

## Overview

Semantic chunking is an advanced document splitting strategy that breaks documents at natural **topic boundaries** rather than fixed character/word/sentence counts. It uses embedding similarity to detect where the semantic meaning of text shifts significantly.

## How It Works

1. **Sentence Splitting**: Text is first split into individual sentences
2. **Embedding Computation**: Each sentence is embedded using the configured sentence transformer
3. **Similarity Calculation**: Cosine similarity is computed between adjacent sentence windows
4. **Breakpoint Detection**: Where similarity drops below the threshold, a chunk boundary is placed
5. **Hierarchical Output**: Leaf chunks are created at break points; parents are formed by merging N leaves

```
Document Text
    ↓
[Sentence 1] [Sentence 2] [Sentence 3] | [Sentence 4] [Sentence 5] | [Sentence 6]...
                    ↓                          ↓
         Similarity: 0.72              Similarity: 0.38  ← Below threshold (0.5)
                                                ↓
                                          CHUNK BREAK
```

## Benefits Over Fixed-Size Chunking

| Aspect | Fixed-Size | Semantic |
|--------|-----------|----------|
| Coherence | May split mid-topic | Chunks are topically coherent |
| Context | Arbitrary boundaries | Natural topic boundaries |
| Retrieval | May return partial context | Returns complete topic segments |
| Flexibility | Same size for all content | Adapts to content structure |

## Configuration

### Enable Semantic Chunking

In `config.fast.yaml` or `config.semantic.yaml`:

```yaml
indexing:
  split_by: semantic  # Changed from "sentence" or "word"
```

### Tuning Parameters

```yaml
indexing:
  # Similarity threshold for chunk breaks (0.0 - 1.0)
  # Lower = more breaks (smaller chunks)
  # Higher = fewer breaks (larger chunks)
  semantic_similarity_threshold: 0.5

  # Minimum characters per chunk (prevents tiny fragments)
  semantic_min_chunk_size: 100

  # Maximum characters per chunk (force-breaks if exceeded)
  semantic_max_chunk_size: 2000

  # Sentences to consider on each side for smoothing
  # Higher = smoother, more stable boundaries
  semantic_buffer_size: 1

  # Number of leaf chunks to merge for parent level
  semantic_parent_merge_count: 3
```

### Recommended Settings by Content Type

| Content Type | Threshold | Min Size | Max Size | Buffer | Parent Merge |
|--------------|-----------|----------|----------|--------|--------------|
| Technical docs | 0.45 | 150 | 1500 | 2 | 4 |
| News articles | 0.50 | 100 | 1200 | 1 | 3 |
| Legal/formal | 0.55 | 200 | 2000 | 2 | 3 |
| Conversational | 0.40 | 80 | 1000 | 1 | 4 |

## Usage

### Re-index with Semantic Chunking

```bash
# Using the dedicated semantic config
python hierarchical.py config.semantic.yaml

# Or modify config.fast.yaml and run
python hierarchical.py config.fast.yaml
```

### Output

The indexer will log:
```
[INFO] Initialized semantic chunker (threshold=0.5)
[INFO] INDEXING CONFIGURATION
[INFO]   Split strategy:    semantic (threshold=0.5, min=100, max=2000, parent_merge=3)
```

And the final output will include:
```json
{
  "config": {
    "split_by": "semantic"
  }
}
```

## Implementation Details

### SemanticChunker Class

Located in `hierarchical.py`:

```python
class SemanticChunker:
    def __init__(
        self,
        embedder_model: str = "sentence-transformers/all-MiniLM-L12-v2",
        similarity_threshold: float = 0.5,
        min_chunk_size: int = 100,
        max_chunk_size: int = 2000,
        buffer_size: int = 1,
        parent_merge_count: int = 3,
        device: str = "cuda",
    ):
```

### Key Methods

- `chunk_text(text: str) -> List[str]`: Split text into semantic chunks
- `chunk_document(doc: Document) -> Tuple[List[Document], List[Document]]`: Create hierarchical parent/leaf documents
- `_compute_similarities(sentences: List[str]) -> List[float]`: Calculate inter-sentence similarities
- `_find_breakpoints(sentences, similarities) -> List[int]`: Determine where to break

### Metadata

Semantically chunked documents include:
```python
{
    "semantic_chunk": True,
    "__level": 1 or 2,  # 1=parent, 2=leaf
    "__split_id": 0,    # Chunk index
    "__root_id": "...", # Original document ID
    "__parent_id": "..." # Parent chunk ID (for leaves)
}
```

## Performance Considerations

1. **Initial embedding cost**: First run loads the sentence transformer model
2. **Per-document cost**: Each document requires embedding all sentences
3. **GPU acceleration**: Use `embedder_device: cuda` for faster processing
4. **Batch processing**: Semantic chunking is applied per-document, parallelizable via `num_workers`

## Troubleshooting

### Chunks too small
- Increase `semantic_similarity_threshold` (e.g., 0.5 → 0.6)
- Decrease `semantic_min_chunk_size`

### Chunks too large
- Decrease `semantic_similarity_threshold` (e.g., 0.5 → 0.4)
- Decrease `semantic_max_chunk_size`

### Inconsistent boundaries
- Increase `semantic_buffer_size` (e.g., 1 → 2 or 3)
- This smooths the similarity calculation

### Memory issues
- Reduce `num_workers` for parallel processing
- Use `streaming_writes: true` to write incrementally
