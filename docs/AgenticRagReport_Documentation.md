# Agentic RAG Report Documentation

## Technical Reference for the Radiant RAG Pipeline HTML Report Generator

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture Design](#architecture-design)
3. [Execution Modes](#execution-modes)
4. [Core Functions](#core-functions)
5. [Helper Functions](#helper-functions)
6. [Configuration System](#configuration-system)
7. [Report Structure](#report-structure)
8. [Testing and Usage](#testing-and-usage)
9. [Recommendations and Improvements](#recommendations-and-improvements)
10. [Usage Examples](#usage-examples)

---

## Overview

### Purpose

The `agentic_rag_report.py` module is the HTML report generator and smoke test tool for the Radiant RAG pipeline. It provides two execution modes: full Agentic RAG with LLM generation, and retrieval-only BM25 keyword search.

### Key Responsibilities

- Run queries through the full Agentic RAG pipeline or BM25-only retrieval
- Collect and aggregate results, telemetry, and metrics
- Generate comprehensive HTML reports with configuration, answers, sources, and metrics
- Provide CLI and programmatic interfaces for testing and evaluation

### Design Philosophy

The module follows a **report-centric** design where all pipeline outputs are collected, normalized, and rendered into a human-readable HTML report. This enables visual inspection of RAG quality, debugging of retrieval issues, and comparison of different configurations.

---

## Architecture Design

### High-Level Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          agentic_rag_report.py                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │                               │
                    ▼                               ▼
           ┌──────────────┐               ┌──────────────┐
           │  mode="rag"  │               │mode="retrieval"│
           └──────┬───────┘               └──────┬───────┘
                  │                               │
                  ▼                               ▼
    ┌─────────────────────────┐     ┌─────────────────────────┐
    │ run_agentic_smoke_query │     │ run_retrieval_only_query│
    │                         │     │                         │
    │ • Registers agents      │     │ • Builds BM25 retriever │
    │ • Runs full pipeline    │     │ • Keyword search only   │
    │ • Collects telemetry    │     │ • No LLM, no agents     │
    └────────────┬────────────┘     └────────────┬────────────┘
                 │                               │
                 └───────────────┬───────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │   Result Normalization  │
                    │                         │
                    │ • Snippets → rows       │
                    │ • Sources → refs        │
                    │ • Citations → aligned   │
                    │ • Metrics computed      │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │    HTML Rendering       │
                    │                         │
                    │ • Configuration section │
                    │ • Query/Answer blocks   │
                    │ • Snippets table        │
                    │ • Sources table         │
                    │ • Metrics table         │
                    │ • Telemetry table       │
                    │ • Cache summary         │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │  agentic_report_*.html  │
                    └─────────────────────────┘
```

### Module Structure

```
agentic_rag_report.py
├── Constants
│   ├── DEFAULT_QUERIES              # Built-in test queries
│   ├── SHOW_SOURCES_TABLE           # Feature flag
│   └── SHOW_CITATIONS_TABLE         # Feature flag
│
├── Globals
│   ├── _AGENTS_INITIALIZED          # Agent registration state
│   └── _AGENTS_CONFIG_PATH          # Config path cache
│
├── Helper Functions
│   ├── ensure_agents_registered()   # Lazy agent initialization
│   ├── score_to_confidence()        # Score normalization
│   ├── infer_level()                # Hierarchy level detection
│   ├── _make_doc_chunk_key()        # Key normalization
│   ├── _enrich_docs_for_bm25()      # BM25 content enrichment
│   ├── classify_backend_with_cfg()  # Backend classification
│   ├── standardize_phase()          # Phase normalization
│   ├── build_retriever_settings_from_cfg()  # Extract settings
│   ├── get_runtime_retrieval_cfg()  # Runtime config extraction
│   ├── normalize_telemetry_backends()  # Backend display names
│   ├── aggregate_agent_metrics()    # Metrics aggregation
│   ├── compute_high_level_metrics() # Per-query metrics
│   └── get_retrieval_cache_stats()  # Cache statistics
│
├── RAG Mode Functions
│   ├── run_agentic_smoke_query()    # Single query execution
│   └── run_smoke_test()             # Multi-query + report
│
├── Retrieval Mode Functions
│   ├── build_bm25_retriever()       # BM25 retriever setup
│   ├── run_retrieval_only_query()   # Single BM25 query
│   └── run_retrieval_smoke_test()   # Multi-query + report
│
└── Entry Points
    ├── run_smoke_test_entry()       # Programmatic entry
    └── main()                       # CLI entry
```

---

## Execution Modes

### Mode: `rag` (Full Agentic RAG)

```
┌─────────────────────────────────────────────────────────────────┐
│                      RAG Mode Pipeline                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. ensure_agents_registered()                                  │
│     └─ Calls orchestrator.register_default_agents()            │
│                                                                 │
│  2. clear_telemetry()                                          │
│     └─ Resets TELEMETRY_EVENTS buffer                          │
│                                                                 │
│  3. agentic_once_with_metadata(query)                          │
│     └─ Full pipeline: Router → Planner → Retrieval →           │
│        Generator → Critic → Policy → PostProcess               │
│                                                                 │
│  4. Extract results:                                            │
│     ├─ answer_text (generated answer)                          │
│     ├─ context_snippets (retrieved chunks)                     │
│     ├─ citations (used sources)                                │
│     ├─ telemetry (timing events)                               │
│     └─ cache_stats (retrieval cache)                           │
│                                                                 │
│  5. Normalize and render HTML report                           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Features:**
- Full agent pipeline execution
- LLM-generated answers
- Query expansion, PRF, reranking
- Iterative refinement (critic → rewrite)
- Comprehensive telemetry

### Mode: `retrieval` (BM25 Only)

```
┌─────────────────────────────────────────────────────────────────┐
│                   Retrieval-Only Mode                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. build_bm25_retriever(cfg)                                  │
│     ├─ Load documents from ChromaDocumentStore                 │
│     ├─ Enrich content with captions/filenames                  │
│     └─ Build InMemoryBM25Retriever                             │
│                                                                 │
│  2. run_retrieval_only_query(query)                            │
│     ├─ BM25 keyword search                                     │
│     └─ No LLM, no agents                                       │
│                                                                 │
│  3. Index modes:                                                │
│     ├─ leaf_only=true  → Search leaf chunks only               │
│     └─ leaf_only=false → Search leaf + parent (dual-index)     │
│                                                                 │
│  4. Output:                                                     │
│     ├─ snippets (BM25-ranked documents)                        │
│     ├─ sources (unique documents)                              │
│     └─ NO generated answer                                     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Features:**
- Fast keyword search
- No LLM costs
- Dual-index support (leaf + parent)
- Useful for debugging retrieval quality

---

## Core Functions

### run_agentic_smoke_query

Executes a single Agentic RAG query and normalizes results.

```python
def run_agentic_smoke_query(query_text: str, config_path: str) -> Dict[str, Any]:
    """
    Returns:
      {
        "answer": str,                  # Generated answer text
        "snippets": List[Dict],         # Context snippet rows
        "sources": List[Dict],          # Source document rows
        "telemetry": List[Dict],        # Telemetry event rows
        "cache_stats": Dict,            # Cache statistics
        "citations": List[Dict],        # Citation rows
      }
    """
```

**Processing Steps:**
1. Register agents (if not already done)
2. Clear telemetry buffer
3. Run `agentic_once_with_metadata(query)`
4. Extract and normalize snippets with scores
5. Build source reference index
6. Align citations to sources
7. Format telemetry rows
8. Return normalized result dict

### run_retrieval_only_query

Executes a single BM25-only retrieval query.

```python
def run_retrieval_only_query(
    query_text: str,
    cfg: Dict[str, Any],
    retriever: InMemoryBM25Retriever,
    bm25_top_k: Optional[int] = None,
    runtime_cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Returns:
      {
        "answer": str,                  # Placeholder (no generation)
        "snippets": List[Dict],         # BM25-ranked document rows
        "sources": List[Dict],          # Unique document rows
        "telemetry": [],                # Empty (no agents)
        "cache_stats": Dict,            # Cache statistics
        "citations": [],                # Empty (no citations)
        "retrieval_index": Dict,        # Index configuration info
      }
    """
```

### run_smoke_test

Multi-query RAG mode report generation.

```python
def run_smoke_test(
    config_path: str,
    cfg: Dict[str, Any],
    queries: List[str],
) -> Tuple[str, str]:
    """
    Returns:
      (html_content, output_file_path)
    """
```

**Processing:**
1. Render configuration section
2. For each query:
   - Run `run_agentic_smoke_query()`
   - Compute per-query metrics
   - Render query/answer HTML block
3. Normalize all telemetry backends
4. Aggregate agent metrics
5. Render metrics, telemetry, cache sections
6. Write HTML to timestamped file

### run_retrieval_smoke_test

Multi-query retrieval-only report generation.

```python
def run_retrieval_smoke_test(
    config_path: str,
    cfg: Dict[str, Any],
    queries: List[str],
    bm25_top_k: Optional[int] = None,
) -> Tuple[str, str]:
    """
    Returns:
      (html_content, output_file_path)
    """
```

---

## Helper Functions

### Score Normalization

```python
def score_to_confidence(raw_score: float, score_type: str = "crossencoder") -> float:
    """
    Map raw retrieval/reranker score to [0,1] confidence.
    
    Score types:
    - "cosine": [-1, 1] → [0, 1] via linear transform
    - "dot": Unbounded → [0, 1] via scaled sigmoid
    - "bm25": Positive unbounded → [0, 1] via log-compressed sigmoid
    - "crossencoder": Logits → [0, 1] via clipped sigmoid
    - "raw": Direct clamp to [0, 1]
    """
```

### Hierarchy Level Detection

```python
def infer_level(meta: Dict[str, Any]) -> str:
    """
    Convert metadata to human-readable level label.
    
    Priority:
    1. meta["is_leaf"] (explicit boolean)
    2. meta["h_level"] >= 2 → "leaf"
    3. meta["h_level"] < 2 → "parent"
    4. Otherwise → "unknown"
    """
```

### BM25 Content Enrichment

```python
def _enrich_docs_for_bm25(docs: List[Any]) -> None:
    """
    In-place enrichment of Document.content with:
    - vision_caption (for image documents)
    - filename / source_path
    
    Enables BM25 matching on metadata fields.
    """
```

### Backend Classification

```python
def classify_backend_with_cfg(
    model_name: Optional[str],
    backend_raw: Optional[str],
    cfg: Dict[str, Any],
) -> str:
    """
    Config-aware backend classification.
    
    Returns one of:
    - "HF-Local"
    - "OpenAI-Compatible (vLLM / Ollama)"
    - "Ollama-Cloud"
    
    Rules:
    1. api_base contains 'ollama.com' → "Ollama-Cloud"
    2. Any api_base/base_url set → "OpenAI-Compatible"
    3. use_local=True, no base_url → "HF-Local"
    4. Heuristic fallback
    """
```

### Metrics Computation

```python
def compute_high_level_metrics(
    answer_text: str,
    snippet_rows: List[Dict[str, Any]],
    citations: List[Dict[str, Any]],
    query_index: int,
) -> List[Dict[str, Any]]:
    """
    Build per-query metrics for the Evaluation Metrics table.
    
    Categories:
    - Retrieval: num_docs, num_snippets, avg_score, max_score
    - Citations: num_citations, unique_cited_docs
    - Answer: length (chars), length (words)
    """
```

```python
def aggregate_agent_metrics(telemetry_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Compute average elapsed times per agent from telemetry.
    
    Returns rows like:
    {
      "category": "Agents",
      "metric": "BasicRouterAgent – avg elapsed (n=3)",
      "value": "15.23 ms",
      "notes": "Average elapsed wall time per telemetry event."
    }
    """
```

---

## Configuration System

### Config Resolution

```python
def get_runtime_retrieval_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract runtime settings for retrieval.
    
    Supports both patterns:
    
    # Pattern 1: Retrieval-centric
    retrieval:
      leaf_chroma_path: ./chroma_db
      leaf_collection: leaves
      parent_chroma_path: ./chroma_db_parents
      parent_collection: parents
      leaf_only: false
      bm25_top_k: 50
    
    # Pattern 2: Vectorstore-centric
    vectorstore:
      persist_path: ./chroma_db
      collection_name: leaves
    parent_vectorstore:
      persist_path: ./chroma_db_parents
      collection_name: parents
    
    Returns:
    {
      "leaf_path": Path,
      "leaf_collection": str,
      "parent_path": Optional[Path],
      "parent_collection": Optional[str],
      "leaf_only": bool,
      "bm25_top_k": int,
    }
    """
```

### Retriever Settings Extraction

```python
def build_retriever_settings_from_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract retriever settings for display in Configuration section.
    
    Returns:
    {
      "retrieval.leaf_only": bool,
      "retrieval.enable_hybrid": bool,
      "retrieval.leaf_top_k": int,
      "retrieval.bm25_top_k": int,
      "retrieval.enable_rerank": bool,
      "retrieval.rerank_top_k": int,
      "qe.enabled": bool,
      "qe.num_variants": int,
      "prf.enabled": bool,
      "prf.docs": int,
      "prf.terms": int,
    }
    """
```

---

## Report Structure

### HTML Report Sections

```
┌─────────────────────────────────────────────────────────────────┐
│                    Agentic RAG Report                           │
│                    (or Retrieval-Only Report)                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │ Configuration                                             │ │
│  │ • Config file path                                        │ │
│  │ • LLM settings                                            │ │
│  │ • Retrieval settings                                      │ │
│  │ • Number of queries                                       │ │
│  └───────────────────────────────────────────────────────────┘ │
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │ Query 1                                                   │ │
│  │ ┌─────────────────────────────────────────────────────┐  │ │
│  │ │ Question: [user query]                              │  │ │
│  │ │ Answer: [generated answer or retrieval-only note]   │  │ │
│  │ └─────────────────────────────────────────────────────┘  │ │
│  │ ┌─────────────────────────────────────────────────────┐  │ │
│  │ │ Context Snippets Table (if SHOW_CITATIONS_TABLE)    │  │ │
│  │ │ • Rank, Doc ID, Score, Title, Page, Text            │  │ │
│  │ └─────────────────────────────────────────────────────┘  │ │
│  │ ┌─────────────────────────────────────────────────────┐  │ │
│  │ │ Top Documents                                       │  │ │
│  │ │ • Ranked document summaries                         │  │ │
│  │ └─────────────────────────────────────────────────────┘  │ │
│  │ ┌─────────────────────────────────────────────────────┐  │ │
│  │ │ Sources Table (if SHOW_SOURCES_TABLE)               │  │ │
│  │ │ • Ref, Doc ID, Title, Page, Level                   │  │ │
│  │ └─────────────────────────────────────────────────────┘  │ │
│  └───────────────────────────────────────────────────────────┘ │
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │ Query 2, 3, ... (same structure)                         │ │
│  └───────────────────────────────────────────────────────────┘ │
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │ Evaluation Metrics                                        │ │
│  │ • Per-query retrieval metrics                            │ │
│  │ • Per-query citation metrics                             │ │
│  │ • Per-query answer metrics                               │ │
│  │ • Agent timing metrics                                   │ │
│  └───────────────────────────────────────────────────────────┘ │
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │ Telemetry                                                 │ │
│  │ • Agent, Event, Phase, Elapsed, Model, Backend           │ │
│  └───────────────────────────────────────────────────────────┘ │
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │ Cache Summary                                             │ │
│  │ • Hits, Misses, Hit Rate, Capacity, Current Size         │ │
│  └───────────────────────────────────────────────────────────┘ │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Output File Naming

| Mode | Filename Pattern |
|------|------------------|
| RAG | `agentic_report_YYYYMMDDHHMMSS.html` |
| Retrieval | `retrieval_report_YYYYMMDDHHMMSS.html` |

---

## Testing and Usage

### CLI Usage

```bash
# Full Agentic RAG with default queries
python agentic_rag_report.py --config config.fast.yaml

# Agentic RAG with custom queries
python agentic_rag_report.py --config config.fast.yaml \
    --query "What is hierarchical RAG?" \
    --query "Explain the role of multi-agent systems."

# Retrieval-only BM25 mode
python agentic_rag_report.py --mode retrieval --config config.fast.yaml \
    --query "dogs playing cards"

# Retrieval-only with custom top_k
python agentic_rag_report.py --mode retrieval --config config.fast.yaml \
    --bm25-top-k 100 \
    --query "quantum computing"
```

### CLI Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--config` | str | `config/config.fast.yaml` | Path to config file |
| `--query` | str (multi) | DEFAULT_QUERIES | Queries to run |
| `--mode` | str | `rag` | `rag` or `retrieval` |
| `--bm25-top-k` | int | from config | BM25 result count |

### Programmatic Usage (Notebooks)

```python
from agentic_rag_report import run_smoke_test_entry
from IPython.display import HTML

# Full Agentic RAG
html, path = run_smoke_test_entry("config.fast.yaml")
HTML(html)

# With custom queries
html, path = run_smoke_test_entry(
    "config.fast.yaml",
    queries=["What is RAG?", "Explain transformers."]
)

# Retrieval-only mode
html, path = run_smoke_test_entry(
    "config.fast.yaml",
    queries=["dogs playing cards"],
    mode="retrieval",
    bm25_top_k=50,
)
HTML(html)
```

### Default Queries

```python
DEFAULT_QUERIES = [
    "What is hierarchical RAG?",
    "What is Agentic RAG and how is it different from traditional RAG?",
    "Explain the role of multi-agent systems in RAG.",
]
```

---

## Recommendations and Improvements

### Critical Issues

#### 1. Global State for Agent Registration

**Problem:** Uses global `_AGENTS_INITIALIZED` and `_AGENTS_CONFIG_PATH` for state.

**Recommendation:** Encapsulate in a class:

```python
class ReportRunner:
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.agents_initialized = False
    
    def ensure_agents(self):
        if not self.agents_initialized:
            register_default_agents(self.config_path)
            self.agents_initialized = True
```

#### 2. No Error Handling for Missing Config

**Problem:** `get_runtime_retrieval_cfg()` raises on missing config but other functions may not handle it gracefully.

**Recommendation:** Add consistent error handling:

```python
def run_smoke_test_entry(config_path: str, ...):
    try:
        cfg = load_config(config_path)
    except FileNotFoundError:
        return render_error_report(f"Config not found: {config_path}"), None
    except yaml.YAMLError as e:
        return render_error_report(f"Invalid YAML: {e}"), None
```

---

### High Priority Improvements

#### 3. Add Progress Reporting

**Problem:** No progress feedback during long-running reports.

**Recommendation:** Add callbacks:

```python
def run_smoke_test_entry(
    config_path: str,
    queries: List[str] = None,
    progress_callback: Callable[[int, int, str], None] = None,
):
    for i, query in enumerate(queries):
        if progress_callback:
            progress_callback(i + 1, len(queries), query)
        # ... process query
```

#### 4. Add Report Comparison

**Problem:** No way to compare reports from different configurations.

**Recommendation:** Add diff capability:

```python
def compare_reports(report1_path: str, report2_path: str) -> str:
    """Generate comparison HTML showing metric differences."""
    pass
```

#### 5. Add JSON Export

**Problem:** Only HTML output available.

**Recommendation:** Add JSON export for programmatic analysis:

```python
def run_smoke_test_entry(
    ...,
    output_format: str = "html",  # "html" | "json" | "both"
) -> Union[Tuple[str, str], Dict[str, Any]]:
```

---

### Medium Priority Improvements

#### 6. Async Query Execution

**Problem:** Queries run sequentially.

**Recommendation:** Add parallel execution option:

```python
async def run_smoke_test_async(
    config_path: str,
    cfg: Dict[str, Any],
    queries: List[str],
    parallel: bool = False,
):
    if parallel:
        results = await asyncio.gather(*[
            run_query_async(q) for q in queries
        ])
    else:
        results = [await run_query_async(q) for q in queries]
```

#### 7. Add Query Templates

**Problem:** Hard-coded default queries.

**Recommendation:** Support query templates:

```python
QUERY_TEMPLATES = {
    "basic": ["What is X?", "How does X work?"],
    "comparison": ["Compare X and Y", "X vs Y differences"],
    "comprehensive": [...],
}

def run_smoke_test_entry(
    ...,
    template: str = None,  # Use predefined template
    queries: List[str] = None,  # Or custom queries
):
```

#### 8. Add Result Caching

**Problem:** Re-running same queries always hits the pipeline.

**Recommendation:** Add optional caching:

```python
def run_smoke_test_entry(
    ...,
    cache_results: bool = False,
    cache_dir: str = ".report_cache",
):
    if cache_results:
        cache_key = hash_query_config(query, config_path)
        if exists_in_cache(cache_key, cache_dir):
            return load_from_cache(cache_key, cache_dir)
```

---

### Low Priority / Future Enhancements

#### 9. Add Interactive Mode

**Recommendation:** Support interactive HTML with filtering:

```python
def render_interactive_report(...):
    """Generate HTML with JavaScript filtering/sorting."""
    pass
```

#### 10. Add PDF Export

**Recommendation:** Support PDF output:

```python
def export_to_pdf(html_path: str) -> str:
    """Convert HTML report to PDF."""
    pass
```

#### 11. Add Batch Processing

**Recommendation:** Process query files:

```python
def run_batch_from_file(
    query_file: str,  # One query per line
    config_path: str,
) -> Tuple[str, str]:
    queries = Path(query_file).read_text().strip().split('\n')
    return run_smoke_test_entry(config_path, queries=queries)
```

---

## Usage Examples

### Basic CLI Usage

```bash
# Generate RAG report with defaults
python agentic_rag_report.py --config config.fast.yaml

# Custom queries
python agentic_rag_report.py --config config.fast.yaml \
    --query "What is RAG?" \
    --query "Explain transformers."

# Retrieval-only debugging
python agentic_rag_report.py --mode retrieval \
    --config config.fast.yaml \
    --query "dogs playing poker"
```

### Notebook Usage

```python
from agentic_rag_report import run_smoke_test_entry
from IPython.display import HTML, display

# Run RAG pipeline
html, output_path = run_smoke_test_entry(
    config_path="config/config.fast.yaml",
    queries=[
        "What is hierarchical RAG?",
        "Explain the benefits of chunking."
    ],
    mode="rag"
)

# Display inline
display(HTML(html))

# Print output path
print(f"Report saved to: {output_path}")
```

### Retrieval Debugging

```python
# Debug retrieval without LLM costs
html, path = run_smoke_test_entry(
    "config.fast.yaml",
    queries=["dogs playing cards"],  # Known image in corpus
    mode="retrieval",
    bm25_top_k=20,
)

# Check if image documents are retrieved
# (vision_caption and filename matching)
HTML(html)
```

### Programmatic Result Access

```python
from agentic_rag_report import run_agentic_smoke_query, ensure_agents_registered

# Initialize agents
ensure_agents_registered("config.fast.yaml")

# Run single query
result = run_agentic_smoke_query(
    query_text="What is RAG?",
    config_path="config.fast.yaml"
)

# Access structured results
print(f"Answer: {result['answer'][:200]}...")
print(f"Snippets: {len(result['snippets'])}")
print(f"Sources: {len(result['sources'])}")
print(f"Citations: {len(result['citations'])}")

# Analyze telemetry
for event in result['telemetry']:
    print(f"{event['agent']}: {event['elapsed']}")
```

### Custom Report Generation

```python
from agentic_rag_report import (
    run_agentic_smoke_query,
    ensure_agents_registered,
    compute_high_level_metrics,
)

ensure_agents_registered("config.fast.yaml")

queries = ["Q1", "Q2", "Q3"]
all_metrics = []

for i, query in enumerate(queries, 1):
    result = run_agentic_smoke_query(query, "config.fast.yaml")
    
    metrics = compute_high_level_metrics(
        answer_text=result['answer'],
        snippet_rows=result['snippets'],
        citations=result['citations'],
        query_index=i,
    )
    all_metrics.extend(metrics)

# Custom analysis
import pandas as pd
df = pd.DataFrame(all_metrics)
print(df[df['category'] == 'Retrieval'])
```

---

## Appendix

### Glossary

| Term | Definition |
|------|------------|
| **Smoke Test** | Quick validation that pipeline works end-to-end |
| **BM25** | Best Matching 25, keyword-based retrieval algorithm |
| **Dual Index** | Using both leaf and parent chunks for retrieval |
| **Confidence** | Normalized or raw score for ranking |

### Feature Flags

| Flag | Default | Description |
|------|---------|-------------|
| `SHOW_SOURCES_TABLE` | `False` | Show sources table in report |
| `SHOW_CITATIONS_TABLE` | `False` | Show citations table in report |

### Default Query Set

| Index | Query |
|-------|-------|
| 1 | "What is hierarchical RAG?" |
| 2 | "What is Agentic RAG and how is it different from traditional RAG?" |
| 3 | "Explain the role of multi-agent systems in RAG." |

### Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | Initial | RAG + retrieval-only modes |

### References

- Radiant Repository: https://github.com/dshipley71/radiant
- Related Files: `agentic_report_html.py`, `core/orchestrator.py`, `agents/retriever.py`

---

*Document generated for the Radiant RAG Pipeline project. For questions or contributions, please refer to the project repository.*
