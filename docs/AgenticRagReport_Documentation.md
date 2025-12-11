# AgenticRagReport Documentation

Technical reference for the Radiant RAG pipeline report generator.

---

## Overview

The `agentic_rag_report.py` module provides HTML report generation and smoke testing functionality for the Radiant RAG pipeline.

**Module Location:** `agentic_rag_report.py`

---

## Execution Modes

### RAG Mode (Default)

Full agentic RAG pipeline with generation, criticism, and iteration.

```bash
python agentic_rag_report.py --config config.fast.yaml
```

### Retrieval Mode

Retrieval-only mode (BM25) for debugging retrieval quality.

```bash
python agentic_rag_report.py --mode retrieval --config config.fast.yaml
```

---

## CLI Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--config` | str | `config.fast.yaml` | Configuration file path |
| `--mode` | str | `rag` | Execution mode: `rag` or `retrieval` |
| `--query` | str | - | Single query to run |
| `--output` | str | `report.html` | Output HTML file |
| `--quiet` | flag | - | Suppress console output |

---

## Report Contents

### RAG Mode Report

- Query and final answer
- Iteration history
- Critic feedback per iteration
- Policy decisions
- Retrieval metrics
- Telemetry events table
- Citations with snippets

### Retrieval Mode Report

- Query
- BM25 results with scores
- Document metadata
- Snippet previews

---

## Programmatic Usage

```python
from agentic_rag_report import run_rag_report, run_retrieval_report

# RAG mode
html = run_rag_report(
    query="What is RAG?",
    config_path="config.fast.yaml",
)

# Retrieval mode
html = run_retrieval_report(
    query="What is RAG?",
    config_path="config.fast.yaml",
)

# Save report
with open("report.html", "w") as f:
    f.write(html)
```

---

## Metrics Computed

| Metric | Description |
|--------|-------------|
| Total docs | Number of documents retrieved |
| Avg score | Mean retrieval score |
| Coverage | Snippet coverage ratio |
| Hallucination risk | Inverse of coverage |
| Iterations | Number of generation loops |
| Rewrites | Number of query rewrites |

---

## Related Documentation

- [AgenticReportHtml_Documentation.md](AgenticReportHtml_Documentation.md) - HTML rendering utilities
- [Orchestrator_Documentation.md](Orchestrator_Documentation.md) - Pipeline execution
