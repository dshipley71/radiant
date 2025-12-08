# Agentic Report HTML Documentation

## Technical Reference for the Radiant RAG Pipeline HTML Rendering Utilities

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture Design](#architecture-design)
3. [Rendering Functions](#rendering-functions)
4. [Backend Classification](#backend-classification)
5. [HTML Structure](#html-structure)
6. [Score Normalization](#score-normalization)
7. [Testing Strategies](#testing-strategies)
8. [Recommendations and Improvements](#recommendations-and-improvements)
9. [Usage Examples](#usage-examples)

---

## Overview

### Purpose

The `agentic_report_html.py` module provides HTML rendering utilities for generating visual reports from Radiant RAG pipeline outputs. It converts structured data (snippets, sources, metrics, telemetry) into formatted HTML tables and sections.

### Key Responsibilities

- Render configuration settings as HTML tables
- Format query/answer pairs for display
- Generate context snippet tables with scores
- Create top documents summary with score normalization
- Render sources, citations, metrics, and telemetry tables
- Display cache statistics
- Wrap all sections in a styled HTML document

### Design Philosophy

The module follows a **component-based rendering** approach where each HTML section is rendered by a dedicated function. This enables independent testing, selective rendering, and easy customization of individual report sections.

---

## Architecture Design

### Module Structure

```
agentic_report_html.py
├── Helpers
│   └── _tr(k, v)                        # Table row helper
│
├── Backend Classification
│   ├── _resolve_config_path_for_backend() # Config path resolution
│   ├── _load_config_for_backend()         # Minimal YAML loader
│   └── classify_backend_from_config()     # Backend type detection
│
├── Section Renderers
│   ├── render_configuration_html()       # Config section
│   ├── render_query_and_answer_html()    # Q&A block
│   ├── render_context_snippets_html()    # Snippets table
│   ├── render_top_documents_html()       # Top docs summary
│   ├── render_sources_html()             # Sources table
│   ├── render_eval_metrics_html()        # Metrics table
│   ├── render_telemetry_html()           # Telemetry table
│   └── render_cache_summary_html()       # Cache stats
│
└── Document Wrapper
    └── wrap_full_report_html()           # Full HTML document
```

### Rendering Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    Report Data                                  │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│  │ Config   │  │ Snippets │  │ Sources  │  │Telemetry │       │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘       │
│       │             │             │             │               │
└───────┼─────────────┼─────────────┼─────────────┼───────────────┘
        │             │             │             │
        ▼             ▼             ▼             ▼
┌───────────────┐ ┌───────────────┐ ┌───────────────┐ ┌───────────────┐
│render_config_ │ │render_context_│ │render_sources_│ │render_teleme- │
│     html()    │ │snippets_html()│ │    html()     │ │  try_html()   │
└───────┬───────┘ └───────┬───────┘ └───────┬───────┘ └───────┬───────┘
        │                 │                 │                 │
        │                 │                 │                 │
        ▼                 ▼                 ▼                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                    HTML Section Strings                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│  │<section> │  │<section> │  │<section> │  │<section> │       │
│  │ Config   │  │ Snippets │  │ Sources  │  │Telemetry │       │
│  │</section>│  │</section>│  │</section>│  │</section>│       │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                 ┌─────────────────────────┐
                 │ wrap_full_report_html() │
                 └────────────┬────────────┘
                              │
                              ▼
                 ┌─────────────────────────┐
                 │  <!DOCTYPE html>        │
                 │  <html>                 │
                 │    <head>...</head>     │
                 │    <body>               │
                 │      [sections]         │
                 │    </body>              │
                 │  </html>                │
                 └─────────────────────────┘
```

---

## Rendering Functions

### render_configuration_html

Renders the configuration section with general and retriever settings.

```python
def render_configuration_html(
    config_path: str,
    raw_cfg: Dict[str, Any],
    retriever_settings: Dict[str, Any],
    num_queries: int,
) -> str
```

**Parameters:**
- `config_path`: Path to the configuration file
- `raw_cfg`: Full configuration dictionary
- `retriever_settings`: Extracted retriever-specific settings
- `num_queries`: Number of queries in the report

**Output Structure:**
```html
<section class="agentic-config">
  <h2>Configuration</h2>
  <h3>General</h3>
  <table>
    <tr><th>Config path</th><td>...</td></tr>
    <tr><th>Num queries (report)</th><td>...</td></tr>
    <tr><th>Index vector store path</th><td>...</td></tr>
    <tr><th>Index collection name</th><td>...</td></tr>
  </table>
  <h3>Retriever Settings</h3>
  <table>
    <!-- retriever_settings key-value pairs -->
  </table>
</section>
```

---

### render_query_and_answer_html

Renders a query/answer block.

```python
def render_query_and_answer_html(query_text: str, answer_text: str) -> str
```

**Parameters:**
- `query_text`: User query
- `answer_text`: Generated or placeholder answer

**Output Structure:**
```html
<h3>Query</h3>
<pre>[escaped query]</pre>
<h3>Answer</h3>
<div class="agentic-answer">
  <p>[escaped answer]</p>
</div>
```

---

### render_context_snippets_html

Renders the context snippets table with raw scores.

```python
def render_context_snippets_html(snippet_rows: List[Dict[str, Any]]) -> str
```

**Parameters:**
- `snippet_rows`: List of snippet dictionaries with keys:
  - `rank`: Display rank
  - `score` or `confidence`: Raw retrieval score
  - `title`: Document title
  - `page`: Page number
  - `text`: Snippet text

**Output Structure:**
```html
<section class="agentic-snippets">
  <h3>Context Snippets (top 10)</h3>
  <table>
    <thead>
      <tr><th>#</th><th>Score</th><th>Title / File</th><th>Page</th><th>Snippet</th></tr>
    </thead>
    <tbody>
      <tr><td>1</td><td>0.950</td><td>...</td><td>5</td><td>...</td></tr>
      <!-- more rows -->
    </tbody>
  </table>
  <p><em>Confidence explanation...</em></p>
</section>
```

**Score Display:**
- Uses raw `score` or `confidence` value
- No normalization (matches `retrieval_automerging.py` behavior)
- Format: 3 decimal places

---

### render_top_documents_html

Renders the top documents summary with score normalization.

```python
def render_top_documents_html(
    snippet_rows: List[Dict[str, Any]],
    sources_rows: List[Dict[str, Any]],
    max_docs: int = 10,
) -> str
```

**Parameters:**
- `snippet_rows`: Context snippets
- `sources_rows`: Source documents for title/level lookup
- `max_docs`: Maximum documents to display (default: 10)

**Processing:**
1. Group snippets by `doc_id`
2. Select highest-scoring snippet per document
3. Sort documents by best score (descending)
4. Normalize scores to [0.0, 1.0] based on ceiling of max score
5. Detect metadata-only entries (empty doc_id or text)

**Output Structure:**
```html
<section class="agentic-top-docs">
  <h3>Top Documents (by retrieval score)</h3>
  <p>Explanation...</p>
  <table>
    <thead>
      <tr>
        <th>Rank</th><th>Doc ID</th><th>Title / File</th>
        <th>Score</th><th>Page</th><th>Snippet</th><th>Level</th>
      </tr>
    </thead>
    <tbody>
      <tr><td>1</td><td>doc123</td><td>Title</td><td>0.950</td><td>5</td><td>...</td><td>leaf</td></tr>
    </tbody>
  </table>
</section>
```

---

### render_sources_html

Renders the sources reference table.

```python
def render_sources_html(sources_rows: List[Dict[str, Any]]) -> str
```

**Parameters:**
- `sources_rows`: List of source dictionaries with keys:
  - `ref`: Reference label (e.g., "[S1]")
  - `doc_id`: Document identifier
  - `title`: Document title
  - `page`: Page number
  - `level`: Hierarchy level

**Output Structure:**
```html
<section class="agentic-sources">
  <h3>Sources</h3>
  <table>
    <thead>
      <tr><th>Ref</th><th>Doc ID</th><th>Title / File</th><th>Page</th><th>Level</th></tr>
    </thead>
    <tbody>
      <tr><td>[S1]</td><td>doc123</td><td>Document Title</td><td>5</td><td>leaf</td></tr>
    </tbody>
  </table>
</section>
```

---

### render_eval_metrics_html

Renders the evaluation metrics table.

```python
def render_eval_metrics_html(metrics_rows: List[Dict[str, Any]]) -> str
```

**Parameters:**
- `metrics_rows`: List of metric dictionaries with keys:
  - `category`: Metric category (e.g., "Retrieval", "Citations", "Agents")
  - `metric`: Metric name
  - `value`: Metric value
  - `notes`: Additional notes

**Output Structure:**
```html
<section class="agentic-metrics">
  <h2>Agentic RAG – Evaluation Metrics</h2>
  <table>
    <thead>
      <tr><th>Category</th><th>Metric</th><th>Value</th><th>Notes</th></tr>
    </thead>
    <tbody>
      <tr><td>Retrieval</td><td>Q1 – Total snippets</td><td>10</td><td>...</td></tr>
    </tbody>
  </table>
  <p>Explanation of n in metrics...</p>
</section>
```

---

### render_telemetry_html

Renders the telemetry events table.

```python
def render_telemetry_html(
    telemetry_rows: List[Dict[str, Any]],
    cfg_for_backend: Optional[Dict[str, Any]] = None,
) -> str
```

**Parameters:**
- `telemetry_rows`: List of telemetry event dictionaries with keys:
  - `agent`: Agent name
  - `event`: Event type
  - `phase`: Pipeline phase
  - `elapsed`: Elapsed time string
  - `model`: Model name
  - `backend_display`: Classified backend label
  - `mode`: Runtime mode
  - `iteration`: Loop iteration
  - `timestamp`: ISO timestamp
  - `payload`: Event payload (dict/list)
- `cfg_for_backend`: Optional config for backend classification fallback

**Output Structure:**
```html
<section class="agentic-telemetry">
  <h2>Telemetry</h2>
  <table>
    <thead>
      <tr>
        <th>Agent</th><th>Event</th><th>Phase</th><th>Elapsed</th>
        <th>Model</th><th>Backend</th><th>Mode</th><th>Iteration</th>
        <th>Timestamp</th><th>Payload</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>BasicRouterAgent</td><td>router.decided</td><td>QUERY</td><td>15.23 ms</td>
        <td>gpt-4o-mini</td><td>OpenAI-Compatible</td><td>AGENTIC</td><td>0</td>
        <td>2024-01-01T...</td><td><pre>{...}</pre></td>
      </tr>
    </tbody>
  </table>
  <p>Phase explanations...</p>
</section>
```

**Payload Handling:**
- JSON serialized with 2-space indent
- Truncated to 280 characters with "..." suffix

---

### render_cache_summary_html

Renders the retrieval cache statistics.

```python
def render_cache_summary_html(cache_stats: Dict[str, Any]) -> str
```

**Parameters:**
- `cache_stats`: Dictionary with keys:
  - `enabled`: Whether cache is enabled
  - `backend`: Cache backend name
  - `capacity`: Maximum entries
  - `current_size`: Current entries
  - `hits`: Cache hits
  - `misses`: Cache misses
  - `stores`: Store operations
  - `total_lookups`: Total lookup count
  - `hit_rate`: Hit rate (0.0-1.0)

**Output Structure:**
```html
<section class="agentic-cache">
  <h2>Retrieval Cache Summary</h2>
  <table>
    <tr><th>Enabled</th><td>Yes</td></tr>
    <tr><th>Backend</th><td>QueryCache</td></tr>
    <tr><th>Capacity (entries)</th><td>1000</td></tr>
    <tr><th>Current size (entries)</th><td>50</td></tr>
    <tr><th>Hits</th><td>25</td></tr>
    <tr><th>Misses</th><td>25</td></tr>
    <tr><th>Stores</th><td>25</td></tr>
    <tr><th>Total lookups</th><td>50</td></tr>
    <tr><th>Hit rate</th><td>50.0%</td></tr>
  </table>
  <p>Hit rate explanation...</p>
</section>
```

---

### wrap_full_report_html

Wraps all sections in a complete HTML document.

```python
def wrap_full_report_html(title: str, sections: List[str]) -> str
```

**Parameters:**
- `title`: Document title
- `sections`: List of HTML section strings

**Output Structure:**
```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>[title]</title>
  <style>
    /* Built-in CSS styles */
  </style>
</head>
<body>
  <h1>[title]</h1>
  [section 1]
  [section 2]
  ...
</body>
</html>
```

---

## Backend Classification

### Classification Logic

```python
def classify_backend_from_config(cfg: Dict[str, Any] | None) -> str:
    """
    Returns one of:
    - 'HF-Local'
    - 'OpenAI-Compatible (vLLM / Ollama)'
    - 'Ollama-Cloud'
    """
```

**Decision Tree:**

```
┌─────────────────────────────────────────┐
│        classify_backend_from_config     │
└─────────────────────────────────────────┘
                    │
                    ▼
          ┌─────────────────┐
          │ models.use_local│
          │    == True?     │
          └────────┬────────┘
                   │
         Yes ──────┼────── No
          │        │        │
          ▼        │        ▼
    ┌─────────┐    │   ┌──────────────────┐
    │HF-Local │    │   │api_base contains │
    └─────────┘    │   │ 'ollama.com'?    │
                   │   └────────┬─────────┘
                   │            │
                   │  Yes ──────┼────── No
                   │   │        │        │
                   │   ▼        │        ▼
                   │┌──────────┐│   ┌────────────────┐
                   ││ Ollama-  ││   │ api_base set?  │
                   ││  Cloud   ││   └───────┬────────┘
                   │└──────────┘│           │
                   │            │  Yes ─────┼───── No
                   │            │   │       │       │
                   │            │   ▼       │       ▼
                   │            │┌─────────┐│  ┌─────────┐
                   │            ││OpenAI-  ││  │HF-Local │
                   │            ││Compatible│  └─────────┘
                   │            │└─────────┘│
                   └────────────┴───────────┘
```

### Config Resolution

```python
def _resolve_config_path_for_backend() -> str:
    """
    Resolution order:
    1. AGENTIC_RAG_CONFIG environment variable
    2. 'config/config.fast.yaml' (default)
    """
```

---

## HTML Structure

### CSS Styles

The module includes built-in CSS styles for consistent rendering:

```css
body {
  font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  margin: 1.5rem;
  line-height: 1.5;
}

h1 { margin-bottom: 1rem; }

h2 {
  margin-top: 2rem;
  border-bottom: 1px solid #ccc;
  padding-bottom: 0.25rem;
}

h3 { margin-top: 1.5rem; }

table {
  border-collapse: collapse;
  width: 100%;
  margin-top: 0.5rem;
  margin-bottom: 1rem;
}

th, td {
  border: 1px solid #ccc;
  padding: 0.4rem 0.6rem;
  vertical-align: top;
  text-align: left;
  font-size: 0.9rem;
}

th {
  background-color: #f6f6f6;
  font-weight: 600;
}

pre {
  background-color: #f6f6f6;
  padding: 0.75rem;
  border-radius: 4px;
  overflow-x: auto;
  font-size: 0.8rem;
  max-height: 16rem;
  white-space: pre-wrap;
}

.agentic-answer p { white-space: pre-wrap; }

section.agentic-query-block {
  border: 1px solid #ddd;
  border-radius: 4px;
  padding: 0.75rem 1rem;
  margin-top: 1.5rem;
  background-color: #fafafa;
}
```

### Section CSS Classes

| Class | Purpose |
|-------|---------|
| `agentic-config` | Configuration section |
| `agentic-snippets` | Context snippets table |
| `agentic-top-docs` | Top documents summary |
| `agentic-sources` | Sources table |
| `agentic-metrics` | Evaluation metrics |
| `agentic-telemetry` | Telemetry table |
| `agentic-cache` | Cache summary |
| `agentic-query-block` | Query/answer block |
| `agentic-answer` | Answer text container |

---

## Score Normalization

### Context Snippets (Raw Scores)

The `render_context_snippets_html` function displays **raw scores** without normalization:

```python
raw_score = row.get("score", row.get("confidence", 0.0))
try:
    val = float(raw_score)
except Exception:
    val = 0.0
# Display as-is with 3 decimal places
f"{val:.3f}"
```

### Top Documents (Normalized Scores)

The `render_top_documents_html` function normalizes scores to [0.0, 1.0]:

```python
# 1. Find max raw score
max_raw = max(raw_scores)

# 2. Compute denominator as ceiling of max
denom = math.ceil(max_raw) if max_raw > 0.0 else 1.0

# 3. Normalize each score
norm = raw_val / denom

# 4. Clamp to [0.0, 1.0]
norm = max(0.0, min(1.0, norm))
```

**Example:**
- Raw scores: [3.2, 2.8, 1.5, 0.9]
- Max = 3.2, Ceiling = 4.0
- Normalized: [0.80, 0.70, 0.375, 0.225]

---

## Testing Strategies

### Unit Tests

#### 1. Rendering Tests

```python
import pytest
from agentic_report_html import (
    render_configuration_html,
    render_query_and_answer_html,
    render_context_snippets_html,
    render_top_documents_html,
    render_sources_html,
    render_eval_metrics_html,
    render_telemetry_html,
    render_cache_summary_html,
    wrap_full_report_html,
)

class TestConfigurationRendering:
    
    def test_renders_config_section(self):
        html = render_configuration_html(
            config_path="config.yaml",
            raw_cfg={"vectorstore": {"persist_path": "./db", "collection_name": "docs"}},
            retriever_settings={"retrieval.top_k": 10},
            num_queries=3,
        )
        
        assert '<section class="agentic-config">' in html
        assert "config.yaml" in html
        assert "3" in html  # num_queries
    
    def test_escapes_html_in_values(self):
        html = render_configuration_html(
            config_path="<script>alert('xss')</script>",
            raw_cfg={},
            retriever_settings={},
            num_queries=1,
        )
        
        assert "<script>" not in html
        assert "&lt;script&gt;" in html
```

#### 2. Query/Answer Tests

```python
class TestQueryAnswerRendering:
    
    def test_renders_query_and_answer(self):
        html = render_query_and_answer_html(
            query_text="What is RAG?",
            answer_text="RAG stands for..."
        )
        
        assert "What is RAG?" in html
        assert "RAG stands for..." in html
        assert "<h3>Query</h3>" in html
        assert "<h3>Answer</h3>" in html
    
    def test_escapes_special_characters(self):
        html = render_query_and_answer_html(
            query_text="<script>alert('xss')</script>",
            answer_text="A & B < C > D"
        )
        
        assert "&lt;script&gt;" in html
        assert "&amp;" in html
        assert "&lt;" in html
        assert "&gt;" in html
```

#### 3. Snippets Table Tests

```python
class TestSnippetsRendering:
    
    def test_renders_snippet_table(self):
        snippets = [
            {"rank": 1, "score": 0.95, "title": "Doc1", "page": 5, "text": "Content"},
            {"rank": 2, "score": 0.80, "title": "Doc2", "page": 10, "text": "More content"},
        ]
        
        html = render_context_snippets_html(snippets)
        
        assert '<section class="agentic-snippets">' in html
        assert "0.950" in html  # Score with 3 decimals
        assert "Doc1" in html
    
    def test_empty_snippets(self):
        html = render_context_snippets_html([])
        
        assert "No context snippets" in html
    
    def test_uses_confidence_fallback(self):
        snippets = [{"rank": 1, "confidence": 0.75, "title": "Doc", "text": "Text"}]
        
        html = render_context_snippets_html(snippets)
        
        assert "0.750" in html
```

#### 4. Top Documents Tests

```python
class TestTopDocumentsRendering:
    
    def test_groups_by_doc_id(self):
        snippets = [
            {"doc_id": "doc1", "confidence": 0.9, "title": "Doc 1", "text": "High score"},
            {"doc_id": "doc1", "confidence": 0.7, "title": "Doc 1", "text": "Low score"},
            {"doc_id": "doc2", "confidence": 0.8, "title": "Doc 2", "text": "Medium"},
        ]
        
        html = render_top_documents_html(snippets, [])
        
        # Should show 2 documents, not 3 rows
        assert html.count("<tr>") == 3  # header + 2 data rows
    
    def test_normalizes_scores(self):
        snippets = [
            {"doc_id": "doc1", "confidence": 4.0, "title": "Doc 1", "text": "Text"},
            {"doc_id": "doc2", "confidence": 2.0, "title": "Doc 2", "text": "Text"},
        ]
        
        html = render_top_documents_html(snippets, [])
        
        # Max = 4.0, Ceiling = 4.0
        # doc1: 4.0/4.0 = 1.000
        # doc2: 2.0/4.0 = 0.500
        assert "1.000" in html
        assert "0.500" in html
    
    def test_detects_metadata_only(self):
        snippets = [
            {"doc_id": "", "confidence": 0.5, "text": ""},  # Empty doc_id
            {"doc_id": "None", "confidence": 0.4, "text": ""},  # "None" string
        ]
        
        html = render_top_documents_html(snippets, [])
        
        assert "metadata" in html.lower()
```

#### 5. Backend Classification Tests

```python
from agentic_report_html import classify_backend_from_config

class TestBackendClassification:
    
    def test_hf_local_with_use_local(self):
        cfg = {"models": {"use_local": True}}
        
        result = classify_backend_from_config(cfg)
        
        assert result == "HF-Local"
    
    def test_openai_compatible_with_api_base(self):
        cfg = {"models": {"use_local": False}, "llm": {"api_base": "http://localhost:8000"}}
        
        result = classify_backend_from_config(cfg)
        
        assert result == "OpenAI-Compatible (vLLM / Ollama)"
    
    def test_ollama_cloud_detection(self):
        cfg = {"llm": {"api_base": "https://api.ollama.com/v1"}}
        
        result = classify_backend_from_config(cfg)
        
        assert result == "Ollama-Cloud"
    
    def test_default_is_hf_local(self):
        result = classify_backend_from_config({})
        
        assert result == "HF-Local"
    
    def test_none_config_returns_hf_local(self):
        result = classify_backend_from_config(None)
        
        assert result == "HF-Local"
```

#### 6. Document Wrapper Tests

```python
class TestDocumentWrapper:
    
    def test_wraps_sections(self):
        sections = ["<p>Section 1</p>", "<p>Section 2</p>"]
        
        html = wrap_full_report_html("Test Report", sections)
        
        assert "<!DOCTYPE html>" in html
        assert "<title>Test Report</title>" in html
        assert "<h1>Test Report</h1>" in html
        assert "<p>Section 1</p>" in html
        assert "<p>Section 2</p>" in html
    
    def test_includes_styles(self):
        html = wrap_full_report_html("Test", [])
        
        assert "<style>" in html
        assert "font-family" in html
        assert "border-collapse" in html
    
    def test_escapes_title(self):
        html = wrap_full_report_html("<script>bad</script>", [])
        
        assert "&lt;script&gt;" in html
```

### Test Commands

```bash
# Run all HTML rendering tests
pytest test_agentic_report_html.py -v

# Run with coverage
pytest test_agentic_report_html.py --cov=agentic_report_html --cov-report=html

# Run specific test class
pytest test_agentic_report_html.py::TestTopDocumentsRendering -v
```

---

## Recommendations and Improvements

### High Priority Improvements

#### 1. Add Templating Engine Support

**Problem:** HTML is built with string concatenation.

**Recommendation:** Use Jinja2 templates:

```python
from jinja2 import Environment, FileSystemLoader

env = Environment(loader=FileSystemLoader('templates'))

def render_snippets_html(snippets):
    template = env.get_template('snippets.html')
    return template.render(snippets=snippets)
```

#### 2. Add Dark Mode Support

**Problem:** Only light theme available.

**Recommendation:** Add CSS media query:

```css
@media (prefers-color-scheme: dark) {
  body { background: #1a1a1a; color: #e0e0e0; }
  th { background-color: #2a2a2a; }
  pre { background-color: #2a2a2a; }
}
```

#### 3. Add Accessibility Attributes

**Problem:** Tables lack accessibility markup.

**Recommendation:** Add ARIA attributes:

```html
<table role="table" aria-label="Context Snippets">
  <thead role="rowgroup">
    <tr role="row">
      <th role="columnheader" scope="col">Rank</th>
      ...
    </tr>
  </thead>
  <tbody role="rowgroup">
    <tr role="row">
      <td role="cell">1</td>
      ...
    </tr>
  </tbody>
</table>
```

---

### Medium Priority Improvements

#### 4. Add Sorting/Filtering JavaScript

**Recommendation:** Add interactive table features:

```javascript
function sortTable(tableId, column) {
  // Client-side sorting
}

function filterTable(tableId, query) {
  // Client-side filtering
}
```

#### 5. Add Collapsible Sections

**Recommendation:** Use `<details>` elements:

```html
<details open>
  <summary>Telemetry (50 events)</summary>
  <table>...</table>
</details>
```

#### 6. Add Export Buttons

**Recommendation:** Add CSV/JSON export:

```html
<button onclick="exportTableToCSV('snippets-table', 'snippets.csv')">
  Export CSV
</button>
```

---

### Low Priority / Future Enhancements

#### 7. Add Responsive Design

**Recommendation:** Mobile-friendly tables:

```css
@media (max-width: 768px) {
  table { display: block; overflow-x: auto; }
}
```

#### 8. Add Charts/Visualizations

**Recommendation:** Add Chart.js for metrics:

```javascript
new Chart(ctx, {
  type: 'bar',
  data: { labels: agents, datasets: [{ label: 'Elapsed (ms)', data: times }] }
});
```

#### 9. Add Print Styles

**Recommendation:** Optimize for printing:

```css
@media print {
  .no-print { display: none; }
  body { font-size: 10pt; }
  pre { max-height: none; overflow: visible; }
}
```

---

## Usage Examples

### Basic Rendering

```python
from agentic_report_html import (
    render_configuration_html,
    render_query_and_answer_html,
    render_context_snippets_html,
    wrap_full_report_html,
)

# Render individual sections
config_html = render_configuration_html(
    config_path="config.yaml",
    raw_cfg={"vectorstore": {"persist_path": "./db"}},
    retriever_settings={"top_k": 10},
    num_queries=1,
)

qa_html = render_query_and_answer_html(
    query_text="What is RAG?",
    answer_text="RAG combines retrieval with generation..."
)

snippets_html = render_context_snippets_html([
    {"rank": 1, "score": 0.95, "title": "RAG Paper", "page": 1, "text": "RAG is..."},
    {"rank": 2, "score": 0.80, "title": "Tutorial", "page": 5, "text": "How to use RAG..."},
])

# Wrap in full document
full_html = wrap_full_report_html(
    title="RAG Report",
    sections=[config_html, qa_html, snippets_html]
)

# Save to file
with open("report.html", "w") as f:
    f.write(full_html)
```

### Backend Classification

```python
from agentic_report_html import classify_backend_from_config

# Local HuggingFace
cfg = {"models": {"use_local": True}}
print(classify_backend_from_config(cfg))  # "HF-Local"

# OpenAI-compatible API
cfg = {"llm": {"api_base": "http://localhost:8000/v1"}}
print(classify_backend_from_config(cfg))  # "OpenAI-Compatible (vLLM / Ollama)"

# Ollama Cloud
cfg = {"llm": {"api_base": "https://api.ollama.com/v1"}}
print(classify_backend_from_config(cfg))  # "Ollama-Cloud"
```

### Custom Report Generation

```python
from agentic_report_html import (
    render_eval_metrics_html,
    render_telemetry_html,
    wrap_full_report_html,
)

# Custom metrics
metrics = [
    {"category": "Custom", "metric": "Latency P50", "value": "150ms", "notes": ""},
    {"category": "Custom", "metric": "Latency P99", "value": "500ms", "notes": ""},
]

# Custom telemetry
telemetry = [
    {"agent": "CustomAgent", "event": "process", "phase": "QUERY", 
     "elapsed": "50.00 ms", "model": None, "backend_display": "HF-Local",
     "mode": "RAG", "iteration": 0, "timestamp": "2024-01-01T00:00:00Z",
     "payload": {"custom_key": "value"}},
]

# Render
metrics_html = render_eval_metrics_html(metrics)
telemetry_html = render_telemetry_html(telemetry)

html = wrap_full_report_html("Custom Report", [metrics_html, telemetry_html])
```

---

## Appendix

### Glossary

| Term | Definition |
|------|------------|
| **Snippet** | Retrieved text chunk used as context |
| **Source** | Document from which snippets are extracted |
| **Citation** | Reference to a source used in the answer |
| **Telemetry** | Timing and event data from pipeline execution |

### Function Reference

| Function | Purpose |
|----------|---------|
| `render_configuration_html` | Config section |
| `render_query_and_answer_html` | Q&A block |
| `render_context_snippets_html` | Snippets table |
| `render_top_documents_html` | Top docs summary |
| `render_sources_html` | Sources table |
| `render_eval_metrics_html` | Metrics table |
| `render_telemetry_html` | Telemetry table |
| `render_cache_summary_html` | Cache stats |
| `wrap_full_report_html` | Full document |

### Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | Initial | Core rendering functions |

### References

- Radiant Repository: https://github.com/dshipley71/radiant
- Related Files: `agentic_rag_report.py`, `core/orchestrator.py`

---

*Document generated for the Radiant RAG Pipeline project. For questions or contributions, please refer to the project repository.*
