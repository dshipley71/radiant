from __future__ import annotations

import os
import json
from html import escape
from typing import Any, Dict, List, Iterable


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _tr(k: str, v: Any) -> str:
    return f"<tr><th>{escape(str(k))}</th><td>{escape(str(v))}</td></tr>"


# ---------------------------------------------------------------------------
# Backend classification (config-driven)
# ---------------------------------------------------------------------------


def _resolve_config_path_for_backend() -> str:
    """
    Resolve which config file to use when inferring backend.
    Mirrors orchestrator default: AGENTIC_RAG_CONFIG or config.fast.yaml.
    """
    return os.getenv("AGENTIC_RAG_CONFIG", "config.fast.yaml")


def _load_config_for_backend(config_path: str) -> Dict[str, Any]:
    """
    Minimal YAML loader used only for backend classification.
    Returns {} if file is missing or unreadable.
    """
    if not config_path:
        return {}
    if not os.path.exists(config_path):
        return {}

    try:
        import yaml  # type: ignore
    except Exception:
        return {}

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            return {}
        return data
    except Exception:
        return {}


def classify_backend_from_config(cfg: Dict[str, Any] | None) -> str:
    """
    Map shared YAML config into one of:
      - 'HF-Local'
      - 'OpenAI-Compatible (vLLM / Ollama)'
      - 'Ollama-Cloud'

    Logic:
      - If models.use_local is true → HF-Local
      - Else, look at llm.api_base (or base_url):
          * contains 'ollama.com' → Ollama-Cloud
          * anything non-empty → OpenAI-Compatible (vLLM / Ollama)
      - Fallback → HF-Local
    """
    if not isinstance(cfg, dict):
        return "HF-Local"

    models_cfg = cfg.get("models") or {}
    llm_cfg = cfg.get("llm") or {}

    use_local = bool(models_cfg.get("use_local", False))
    api_base = str(
        llm_cfg.get("api_base")
        or llm_cfg.get("base_url")
        or ""
    ).lower()

    if use_local:
        return "HF-Local"

    if "ollama.com" in api_base:
        return "Ollama-Cloud"

    if api_base:
        return "OpenAI-Compatible (vLLM / Ollama)"

    return "HF-Local"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def render_configuration_html(
    config_path: str,
    raw_cfg: Dict[str, Any],
    retriever_settings: Dict[str, Any],
    num_queries: int,
) -> str:
    vector_cfg = raw_cfg.get("vectorstore", {}) or {}
    idx_vs_path = vector_cfg.get("persist_path", "")
    idx_vs_collection = vector_cfg.get("collection_name", "")

    general_rows = []
    general_rows.append(_tr("Config path", config_path))
    general_rows.append(_tr("Num queries (report)", num_queries))
    general_rows.append(_tr("Index vector store path", idx_vs_path))
    general_rows.append(_tr("Index collection name", idx_vs_collection))

    retr_rows = []
    for k, v in retriever_settings.items():
        retr_rows.append(_tr(k, v))

    html = []
    html.append('<section class="agentic-config">')
    html.append("<h2>Configuration</h2>")

    html.append("<h3>General</h3>")
    html.append('<table border="1" cellspacing="0" cellpadding="4"><tbody>')
    html.extend(general_rows)
    html.append("</tbody></table>")

    if retr_rows:
        html.append("<h3>Retriever Settings</h3>")
        html.append('<table border="1" cellspacing="0" cellpadding="4"><tbody>')
        html.extend(retr_rows)
        html.append("</tbody></table>")

    html.append("</section>")
    return "\n".join(html)


# ---------------------------------------------------------------------------
# Query + Answer
# ---------------------------------------------------------------------------


def render_query_and_answer_html(query_text: str, answer_text: str) -> str:
    html = []
    html.append("<h3>Query</h3>")
    html.append("<pre>" + escape(query_text) + "</pre>")
    html.append("<h3>Answer</h3>")
    html.append('<div class="agentic-answer"><p>')
    html.append(escape(answer_text))
    html.append("</p></div>")
    return "\n".join(html)


# ---------------------------------------------------------------------------
# Context snippets (confidence = raw retrieval score)
# ---------------------------------------------------------------------------


def render_context_snippets_html(snippet_rows: List[Dict[str, Any]]) -> str:
    """
    snippet_rows is expected to contain, per row:
      - 'score'  → raw Document.score from retriever/reranker (preferred), OR
      - 'confidence' → if score is not available (treated as the raw score).

    We DO NOT renormalize or clamp; we simply print the numeric value.
    This mirrors retrieval_automerging.py, which prints Document.score directly.
    """
    if not snippet_rows:
        return "<p><em>No context snippets were recorded.</em></p>"

    html = []
    html.append('<section class="agentic-snippets">')
    html.append("<h3>Context Snippets (top 10)</h3>")
    html.append('<table border="1" cellspacing="0" cellpadding="4">')
    html.append(
        "<thead><tr>"
        "<th>#</th>"
        "<th>Confidence</th>"
        "<th>Title / File</th>"
        "<th>Page</th>"
        "<th>Snippet</th>"
        "</tr></thead>"
    )
    html.append("<tbody>")

    for idx, row in enumerate(snippet_rows, start=1):
        rank = row.get("rank", idx)
        raw_score = row.get("score", row.get("confidence", 0.0))
        try:
            val = float(raw_score)
        except Exception:
            val = 0.0

        title = row.get("title", "")
        page = row.get("page", "")
        text = row.get("text", "")

        html.append(
            "<tr>"
            f"<td>{escape(str(rank))}</td>"
            f"<td>{val:.3f}</td>"
            f"<td>{escape(str(title))}</td>"
            f"<td>{escape(str(page))}</td>"
            f"<td>{escape(str(text))}</td>"
            "</tr>"
        )

    html.append("</tbody></table>")
    html.append(
        "<p><em>Confidence</em> is the raw retrieval / reranker score "
        "(for example, the <code>Document.score</code> produced by Haystack’s "
        "retriever or cross-encoder). It is not renormalized or rescaled within "
        "this table; values are shown exactly as produced by the retrieval stack, "
        "matching the behavior of <code>retrieval_automerging.py</code>.</p>"
    )
    html.append("</section>")
    return "\n".join(html)


# ---------------------------------------------------------------------------
# Sources / citations
# ---------------------------------------------------------------------------


def render_sources_html(sources_rows: List[Dict[str, Any]]) -> str:
    if not sources_rows:
        return "<p><em>No sources were recorded.</em></p>"

    html = []
    html.append('<section class="agentic-sources">')
    html.append("<h3>Sources</h3>")
    html.append('<table border="1" cellspacing="0" cellpadding="4">')
    html.append(
        "<thead><tr>"
        "<th>Ref</th>"
        "<th>Doc ID</th>"
        "<th>Title / File</th>"
        "<th>Page</th>"
        "<th>Level</th>"
        "</tr></thead>"
    )
    html.append("<tbody>")

    for row in sources_rows:
        ref = row.get("ref", "")
        doc_id = row.get("doc_id", "")
        title = row.get("title", "")
        page = row.get("page", "")
        level = row.get("level", "")

        html.append(
            "<tr>"
            f"<td>{escape(str(ref))}</td>"
            f"<td>{escape(str(doc_id))}</td>"
            f"<td>{escape(str(title))}</td>"
            f"<td>{escape(str(page))}</td>"
            f"<td>{escape(str(level))}</td>"
            "</tr>"
        )

    html.append("</tbody></table>")
    html.append("</section>")
    return "\n".join(html)


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------


def render_eval_metrics_html(metrics_rows: List[Dict[str, Any]]) -> str:
    if not metrics_rows:
        return "<p><em>No evaluation metrics were recorded.</em></p>"

    html = []
    html.append('<section class="agentic-metrics">')
    html.append("<h2>Agentic RAG – Evaluation Metrics</h2>")
    html.append('<table border="1" cellspacing="0" cellpadding="4">')
    html.append(
        "<thead><tr>"
        "<th>Category</th>"
        "<th>Metric</th>"
        "<th>Value</th>"
        "<th>Notes</th>"
        "</tr></thead>"
    )
    html.append("<tbody>")

    for row in metrics_rows:
        category = row.get("category", "")
        metric = row.get("metric", "")
        value = row.get("value", "")
        notes = row.get("notes", "")

        html.append(
            "<tr>"
            f"<td>{escape(str(category))}</td>"
            f"<td>{escape(str(metric))}</td>"
            f"<td>{escape(str(value))}</td>"
            f"<td>{escape(str(notes))}</td>"
            "</tr>"
        )

    html.append("</tbody></table>")
    html.append(
        "<p><em>n</em> in the metric description (e.g. <code>avg elapsed (n=4)</code>) "
        "refers to the number of telemetry events (agent invocations) used when computing "
        "the average.</p>"
    )
    html.append("</section>")
    return "\n".join(html)


# ---------------------------------------------------------------------------
# Telemetry
# ---------------------------------------------------------------------------


def render_telemetry_html(telemetry_rows: List[Dict[str, Any]]) -> str:
    if not telemetry_rows:
        return "<p><em>No telemetry events were recorded.</em></p>"

    cfg_path = _resolve_config_path_for_backend()
    cfg_for_backend = _load_config_for_backend(cfg_path)
    backend_label = classify_backend_from_config(cfg_for_backend)

    html = []
    html.append('<section class="agentic-telemetry">')
    html.append("<h2>Telemetry</h2>")
    html.append('<table border="1" cellspacing="0" cellpadding="4">')
    html.append(
        "<thead><tr>"
        "<th>Agent</th>"
        "<th>Event</th>"
        "<th>Phase</th>"
        "<th>Elapsed</th>"
        "<th>Model</th>"
        "<th>Backend</th>"
        "<th>Mode</th>"
        "<th>Iteration</th>"
        "<th>Timestamp</th>"
        "<th>Payload</th>"
        "</tr></thead>"
    )
    html.append("<tbody>")

    for row in telemetry_rows:
        agent = row.get("agent", "")
        event = row.get("event", "")
        phase = row.get("phase", "")
        elapsed = row.get("elapsed", "")
        model = row.get("model", "")
        mode = row.get("mode", "")
        iteration = row.get("iteration", "")
        timestamp = row.get("timestamp", "")

        # Backend is derived purely from config so it matches models.use_local + llm.api_base
        backend = backend_label

        payload = row.get("payload", "")
        if isinstance(payload, (dict, list)):
            try:
                payload_str = json.dumps(payload, indent=2)
            except Exception:
                payload_str = str(payload)
        else:
            payload_str = str(payload)

        # Compact preview to avoid blowing up the table
        max_len = 280
        if len(payload_str) > max_len:
            payload_str = payload_str[: max_len - 3] + "..."

        html.append(
            "<tr>"
            f"<td>{escape(str(agent))}</td>"
            f"<td>{escape(str(event))}</td>"
            f"<td>{escape(str(phase))}</td>"
            f"<td>{escape(str(elapsed))}</td>"
            f"<td>{escape(str(model))}</td>"
            f"<td>{escape(str(backend))}</td>"
            f"<td>{escape(str(mode))}</td>"
            f"<td>{escape(str(iteration))}</td>"
            f"<td>{escape(str(timestamp))}</td>"
            f"<td><pre>{escape(payload_str)}</pre></td>"
            "</tr>"
        )

    html.append("</tbody></table>")

    html.append(
        """
    <p><strong>Phases</strong><br/>
      <code>WARMUP</code>: non-user warmup calls.<br/>
      <code>QUERY</code>: normal user query handling.<br/>
      <code>INDEX</code>: indexing / corpus processing.<br/>
      <code>SYSTEM</code>: diagnostics or background work.
    </p>
    """
    )

    html.append("</section>")
    return "\n".join(html)


# ---------------------------------------------------------------------------
# Cache summary
# ---------------------------------------------------------------------------


def render_cache_summary_html(cache_stats: Dict[str, Any]) -> str:
    if not cache_stats:
        return "<p><em>No cache statistics recorded.</em></p>"

    enabled = cache_stats.get("enabled", False)
    backend = cache_stats.get("backend", "")
    capacity = cache_stats.get("capacity", "")
    current_size = cache_stats.get("current_size", "")
    hits = cache_stats.get("hits", 0)
    misses = cache_stats.get("misses", 0)
    stores = cache_stats.get("stores", 0)
    total_lookups = cache_stats.get("total_lookups", hits + misses)
    hit_rate = cache_stats.get("hit_rate", 0.0)

    html = []
    html.append('<section class="agentic-cache">')
    html.append("<h2>Retrieval Cache Summary</h2>")
    html.append('<table border="1" cellspacing="0" cellpadding="4"><tbody>')

    html.append(_tr("Enabled", "Yes" if enabled else "No"))
    html.append(_tr("Backend", backend))
    html.append(_tr("Capacity (entries)", capacity))
    html.append(_tr("Current size (entries)", current_size))
    html.append(_tr("Hits", hits))
    html.append(_tr("Misses", misses))
    html.append(_tr("Stores", stores))
    html.append(_tr("Total lookups", total_lookups))
    html.append(_tr("Hit rate", f"{hit_rate * 100.0:.1f}%"))

    html.append("</tbody></table>")
    html.append(
        "<p><em>Hit rate</em> is the percentage of retrieval requests served from the "
        "cache instead of hitting the vector store / retriever.</p>"
    )
    html.append("</section>")
    return "\n".join(html)


# ---------------------------------------------------------------------------
# Full document wrapper
# ---------------------------------------------------------------------------


def wrap_full_report_html(title: str, sections: List[str]) -> str:
    body = "\n\n".join(sections)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>{escape(title)}</title>
  <style>
    body {{
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      margin: 1.5rem;
      line-height: 1.5;
    }}
    h1 {{
      margin-bottom: 1rem;
    }}
    h2 {{
      margin-top: 2rem;
      border-bottom: 1px solid #ccc;
      padding-bottom: 0.25rem;
    }}
    h3 {{
      margin-top: 1.5rem;
    }}
    table {{
      border-collapse: collapse;
      width: 100%;
      margin-top: 0.5rem;
      margin-bottom: 1rem;
    }}
    th, td {{
      border: 1px solid #ccc;
      padding: 0.4rem 0.6rem;
      vertical-align: top;
      text-align: left;
      font-size: 0.9rem;
    }}
    th {{
      background-color: #f6f6f6;
      font-weight: 600;
    }}
    pre {{
      background-color: #f6f6f6;
      padding: 0.75rem;
      border-radius: 4px;
      overflow-x: auto;
      font-size: 0.8rem;
      max-height: 16rem;
      white-space: pre-wrap;
    }}
    .agentic-answer p {{
      white-space: pre-wrap;
    }}
    section.agentic-query-block {{
      border: 1px solid #ddd;
      border-radius: 4px;
      padding: 0.75rem 1rem;
      margin-top: 1.5rem;
      background-color: #fafafa;
    }}
  </style>
</head>
<body>
  <h1>{escape(title)}</h1>
  {body}
</body>
</html>
"""
