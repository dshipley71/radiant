# AgenticReportHtml Documentation

Technical reference for the Radiant RAG pipeline HTML rendering utilities.

---

## Overview

The `agentic_report_html.py` module provides HTML rendering functions for RAG pipeline reports, including styling, score normalization, and component rendering.

**Module Location:** `agentic_report_html.py`

---

## Core Functions

### Report Generation

```python
def render_rag_report(
    query: str,
    answer: Answer,
    citations: List[Citation],
    telemetry: List[TelemetryEvent],
    metadata: PostprocessMetadata,
) -> str:
    """Render full RAG report as HTML."""

def render_retrieval_report(
    query: str,
    results: List[RetrievalResult],
) -> str:
    """Render retrieval-only report as HTML."""
```

---

## Component Renderers

### Telemetry Table

```python
def render_telemetry_table(events: List[TelemetryEvent]) -> str:
    """Render telemetry events as HTML table."""
```

Columns: Phase, Iteration, Event Type, Agent, Elapsed (ms)

### Citations Section

```python
def render_citations(citations: List[Citation]) -> str:
    """Render citations as collapsible HTML sections."""
```

### Metrics Summary

```python
def render_metrics_summary(
    num_docs: int,
    avg_score: float,
    coverage: float,
    iterations: int,
) -> str:
    """Render metrics as styled HTML cards."""
```

---

## Score Normalization

```python
def normalize_score(score: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    """Normalize score to 0-1 range for display."""
```

Used for consistent score visualization across different retrieval backends.

---

## CSS Styling

The module includes embedded CSS for:

- Responsive layout
- Score color coding (green → yellow → red)
- Collapsible sections
- Tables with hover effects
- Metric cards
- Code/snippet formatting

---

## Backend Classification

```python
def classify_backend(model_name: str) -> str:
    """Classify model into backend type for display."""
```

| Pattern | Classification |
|---------|---------------|
| `gpt-*` | OpenAI |
| `claude-*` | Anthropic |
| `llama*` | Local/Llama |
| Other | Custom |

---

## Usage Example

```python
from agentic_report_html import render_rag_report
from core.schemas import Answer, Citation, TelemetryEvent, PostprocessMetadata

html = render_rag_report(
    query="What is RAG?",
    answer=Answer(text="RAG combines..."),
    citations=[...],
    telemetry=[...],
    metadata=PostprocessMetadata(critic_summary="", languages=["en"]),
)

with open("report.html", "w") as f:
    f.write(html)
```

---

## Related Documentation

- [AgenticRagReport_Documentation.md](AgenticRagReport_Documentation.md) - Report generation entry point
- [CoreSchemas_Documentation.md](CoreSchemas_Documentation.md) - Data models used in rendering
