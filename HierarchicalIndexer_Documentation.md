# Hierarchical Indexer: Detailed Architecture

## Overview

The hierarchical indexer (`hierarchical.py`) creates a two-level document index: **parent chunks** (larger context windows) and **leaf chunks** (smaller, more precise segments). This enables auto-merging retrieval where related leaf chunks can be combined back into their parent context.

---

## Phase 1: File Parsing

```
Files (PDF, DOCX, TXT, images, etc.)
           ↓
    _parse_files_batch()
           ↓
    Root Documents (one per file/page)
```

**What happens:**
- Files are parsed using format-specific converters (PyMuPDF for PDFs, python-docx for DOCX, etc.)
- Each file becomes one or more "root" documents
- PDFs are split into page-level documents
- Images get OCR text and/or vision captions stored in `meta["vision_caption"]`

**Document structure at this stage:**
```python
Document(
    id="unique-id",
    content="The actual text content of the document...",
    meta={
        "source_path": "/path/to/file.pdf",
        "filename": "file.pdf",
        "page": 1,
        "vision_caption": "Description of image...",  # if image/vision enabled
        # ... other metadata
    }
)
```

---

## Phase 2: Hierarchical Splitting

```
Root Documents
      ↓
_split_hierarchical()
      ↓
┌─────────────────────────────────────┐
│  Parents (larger chunks)            │
│  - e.g., 10 sentences each          │
│  - meta["__level"] = "parent"       │
│  - meta["__doc_id"] = parent's ID   │
└─────────────────────────────────────┘
      ↓
┌─────────────────────────────────────┐
│  Leaves (smaller chunks)            │
│  - e.g., 5 sentences each           │
│  - meta["__level"] = "leaf"         │
│  - meta["__parent_id"] = parent ID  │
└─────────────────────────────────────┘
```

**Splitting modes** (configured via `indexing.split_by`):
- `sentence`: Split by sentence count (default: parent=10, leaf=5)
- `word`: Split by word count
- `page`: Split by page count
- `passage`: Split by passage markers

**Key metadata added:**
```python
# Parent document
meta["__level"] = "parent"
meta["__doc_id"] = "parent-uuid"
meta["__children"] = ["leaf-uuid-1", "leaf-uuid-2", ...]

# Leaf document
meta["__level"] = "leaf"
meta["__parent_id"] = "parent-uuid"
meta["__doc_id"] = "leaf-uuid"
```

---

## Phase 3: Summarization (Optional)

```
                    ┌─────────────────────────────┐
                    │  summarize_leaves: true     │
                    │  summarize_parents: true    │
                    └─────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                   LocalSummarizer                           │
│  - Truncates content to summarizer_max_input_tokens (512)   │
│  - Calls LLM (local HF model or remote API via llm.*)       │
│  - Stores result in meta["display_summary"]                 │
└─────────────────────────────────────────────────────────────┘
```

### When `summarize_leaves: true`

```python
# Before summarization
leaf.content = "The Federal Reserve announced today that interest rates..."
leaf.meta = {"__level": "leaf", "__parent_id": "..."}

# After summarization
leaf.content = "The Federal Reserve announced today that interest rates..."  # UNCHANGED
leaf.meta = {
    "__level": "leaf",
    "__parent_id": "...",
    "display_summary": "Fed announces rate decision with market implications."  # ADDED
}
```

### When `summarize_parents: true`

Same process applied to parent documents.

### TopK Filtering (`summarize_only_topk_leaves`)

If set to N > 0, only the N longest leaves per parent are summarized (saves LLM calls):

```python
# Example: summarize_only_topk_leaves: 8
# Parent has 20 leaves
# → Only 8 longest leaves get display_summary
# → Other 12 leaves have no display_summary
```

### When Summarization is DISABLED

```python
# summarize_leaves: false (default)
# summarize_parents: false (default)

leaf.content = "The Federal Reserve announced today..."
leaf.meta = {"__level": "leaf", "__parent_id": "..."}
# NO display_summary field exists
```

---

## Phase 4: Embedding

```
Documents (with or without display_summary)
                    ↓
         SentenceTransformersDocumentEmbedder
                    ↓
         Documents with embedding vectors
                    ↓
┌───────────────────────────────────────────┐
│  Chroma/pgvector stores the ENTIRE        │
│  document including:                      │
│  - content (original text)                │
│  - meta (including display_summary)       │
│  - embedding vector                       │
└───────────────────────────────────────────┘
```

**What gets embedded:**
- The embedder uses `Document.content` (the actual text) to generate vectors
- The `display_summary` is stored in metadata but NOT used for embedding
- Both leaves and parents are embedded into separate vector stores

---

## Phase 5: Storage

```
┌─────────────────────────────────────────────────────────────┐
│                    LEAF STORE (Chroma/pgvector)             │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ id: "leaf-uuid-1"                                      │ │
│  │ content: "The Federal Reserve announced today..."      │ │
│  │ embedding: [0.123, -0.456, ...]                        │ │
│  │ meta: {                                                │ │
│  │   "__level": "leaf",                                   │ │
│  │   "__parent_id": "parent-uuid",                        │ │
│  │   "display_summary": "Fed rate decision summary...",   │ │  ← Only if summarize_leaves=true
│  │   "source_path": "/path/to/file.pdf",                  │ │
│  │   "page": 1                                            │ │
│  │ }                                                      │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                   PARENT STORE (Chroma/pgvector)            │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ id: "parent-uuid"                                      │ │
│  │ content: "The Federal Reserve announced today that..." │ │  ← Larger chunk
│  │ embedding: [0.789, -0.012, ...]                        │ │
│  │ meta: {                                                │ │
│  │   "__level": "parent",                                 │ │
│  │   "__children": ["leaf-uuid-1", "leaf-uuid-2"],        │ │
│  │   "display_summary": "Comprehensive Fed coverage...",  │ │  ← Only if summarize_parents=true
│  │   "source_path": "/path/to/file.pdf"                   │ │
│  │ }                                                      │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                   PARENT SIDECAR (JSON)                     │
│  {                                                          │
│    "parent-uuid": {                                         │
│      "meta": { ... },                                       │
│      "content": "The Federal Reserve announced..."          │
│    }                                                        │
│  }                                                          │
│  Used for leaf-only retrieval mode to enrich results        │
└─────────────────────────────────────────────────────────────┘
```

---

## Retrieval: How Content and Summaries Are Used

### Step 1: Vector Search (Dense Retrieval)

```
User Query: "What did the Fed announce?"
                    ↓
            Query Embedding
                    ↓
    Chroma/pgvector similarity search
                    ↓
    Returns documents by embedding similarity
    (embeddings were created from content, NOT display_summary)
```

### Step 2: BM25 Search (Lexical/Hybrid Retrieval)

```python
# _enrich_docs_for_bm25() - IN-MEMORY ONLY
# Appends display_summary to content for BM25 indexing

enriched_content = original_content + "\n\n" + display_summary + "\n" + title + "\n" + filename

# This allows BM25 to match keywords in:
# - Original content
# - LLM-generated summary (if exists)
# - Filename/title
```

**Important:** This enrichment is temporary and in-memory. It does NOT modify what's stored in Chroma.

### Step 3: Building Snippets for Response

```python
# _build_snippet() in retriever.py

Snippet(
    chunk_id="leaf-uuid-1",
    score=0.85,
    text="Fed rate decision summary...",      # ← display_summary (for UI)
    content="The Federal Reserve announced...", # ← actual content (for reranking/RAG)
    page=1,
    level="leaf"
)
```

| Field | Source | Used For |
|-------|--------|----------|
| `text` | `meta["display_summary"]` if exists, else `content` | UI display |
| `content` | `Document.content` (always) | Reranking, RAG context |

### Step 4: Reranking

```python
# _best_text_for_parent() in rerank.py

# Priority for TEXT documents:
1. snippet.content     ← Actual document text (PREFERRED)
2. snippet.text        ← Fallback

# Priority for IMAGE documents (no text content):
3. meta["display_summary"]
4. meta["vision_caption"]
5. meta["title"]
```

**Why content over summary for reranking?**
- Cross-encoder needs actual text to score relevance accurately
- Summaries are compressed approximations that may lose important details

### Step 5: RAG Context (LLM Input)

```python
# build_context_snippets_from_results() in orchestrator.py

source_text = snippet.content if snippet.content else snippet.text

# The LLM receives actual document content, NOT summaries
# This ensures grounded, accurate answers from source material
```

---

## Summary: Content vs Display_Summary Usage

| Stage | What's Used | Why |
|-------|-------------|-----|
| **Embedding** | `content` | Semantic vectors should represent actual text |
| **Vector Search** | embedding vectors | Find semantically similar content |
| **BM25 Search** | `content` + `display_summary` + metadata | Expand keyword matching surface |
| **UI Display** | `display_summary` (if exists) else `content` | Concise preview for humans |
| **Reranking** | `content` (preferred) | Accurate relevance scoring |
| **RAG/LLM Context** | `content` | Grounded answers from source material |

---

## Configuration Reference

```yaml
indexing:
  # Splitting
  split_by: sentence              # sentence | word | page | passage
  parent_sentences: 10            # Sentences per parent chunk
  leaf_sentences: 5               # Sentences per leaf chunk
  sentence_overlap: 1             # Overlap between leaves

  # Summarization - summaries stored in meta["display_summary"]
  summarize_leaves: false         # Generate LLM summaries for leaves
  summarize_parents: false        # Generate LLM summaries for parents
  summarizer_batch_size: 16       # Batch size for LLM calls
  summarizer_max_input_tokens: 512  # Truncate input to N tokens
  summarizer_concurrency: 0       # Concurrent API calls (remote only)
  summarize_only_topk_leaves: 0   # 0=all; N>0=only N longest per parent

  # Other
  embed_parents: true             # Whether to embed parent chunks
  num_workers: 0                  # Parallel file parsing (0=sequential)

# LLM for summarization (when models.use_local=false)
llm:
  model: "minimax-m2:cloud"
  api_base: "https://ollama.com/v1"
  api_key: "..."
```

---

## Visual Flow Summary

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           INDEXING PIPELINE                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Files → Parse → Root Docs → Split → Parents + Leaves                       │
│                                            │                                │
│                              ┌─────────────┴─────────────┐                  │
│                              │                           │                  │
│                              ▼                           ▼                  │
│                    [summarize_leaves?]         [summarize_parents?]         │
│                              │                           │                  │
│                              ▼                           ▼                  │
│                    Add display_summary          Add display_summary         │
│                    to leaf.meta                 to parent.meta              │
│                              │                           │                  │
│                              └─────────────┬─────────────┘                  │
│                                            │                                │
│                                            ▼                                │
│                                    Embed (using content)                    │
│                                            │                                │
│                                            ▼                                │
│                              Store in Chroma/pgvector                       │
│                              (content + meta + embedding)                   │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                           RETRIEVAL PIPELINE                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Query → Embed → Vector Search (matches content embeddings)                 │
│            │                                                                │
│            └──→ BM25 Search (matches content + display_summary + metadata)  │
│                              │                                              │
│                              ▼                                              │
│                    Build Snippets:                                          │
│                    - text = display_summary (UI)                            │
│                    - content = actual text (RAG)                            │
│                              │                                              │
│                              ▼                                              │
│                    Rerank using content                                     │
│                              │                                              │
│                              ▼                                              │
│                    LLM receives content for grounded answers                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Key Design Decisions

### Why Separate `text` and `content` in Snippets?

The `Snippet` model has two text fields serving different purposes:

```python
class Snippet(BaseModel):
    text: str                      # Display text (may use display_summary)
    content: Optional[str] = None  # Actual document content
```

This separation ensures:
1. **UI gets readable summaries** - Users see concise, meaningful previews
2. **Reranker gets actual content** - Cross-encoders score true relevance
3. **LLM gets source material** - Answers are grounded in original text, not summaries of summaries

### Why Store Summaries in Metadata, Not Replace Content?

Storing summaries in `meta["display_summary"]` rather than overwriting `content` preserves:
1. **Original text for RAG** - LLM can cite and quote actual sources
2. **Accurate embeddings** - Vectors represent real document semantics
3. **Flexibility** - Different consumers can choose what they need

### Why TopK Filtering for Summarization?

The `summarize_only_topk_leaves` option exists because:
1. **Cost control** - LLM calls are expensive; summarizing every chunk may not be necessary
2. **Quality focus** - Longer chunks often contain more substantive content worth summarizing
3. **Diminishing returns** - Very short chunks may not benefit much from summarization
