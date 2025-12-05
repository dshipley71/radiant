from __future__ import annotations

"""
Agentic RAG Report – HTML Generator

This script:
  * Loads the configuration file (e.g. config.fast.yaml)
  * Optionally initializes the agent registry from orchestrator.py (mode="rag")
  * Runs one or more queries in one of two modes:

    - mode="rag" (default)
        * Full Agentic RAG:
            - Router / Decomposition / Planner / Retrieval / Rerank / Generator
            - Collects answer, context snippets, sources, telemetry, cache stats
        * Renders a full HTML report.

    - mode="retrieval"
        * Retrieval-only:
            - BM25 keyword search over the leaf Chroma index
              (vectorstore.leaf_chroma_path / vectorstore.leaf_collection)
            - No agents, no LLM, no RAG orchestration
        * Renders an HTML report that shows queries and retrieved documents
          (snippets + sources), but no generated answer.

  * Saves the report to a timestamped file:
        agentic_report_YYYYMMDDHHMMSS.html

Usage patterns:

1) CLI / shell:

    # Full Agentic RAG using default queries
    python agentic_rag_report.py --config config.fast.yaml

    # Agentic RAG with custom queries
    python agentic_rag_report.py --config config.fast.yaml \
        --query "What is hierarchical RAG?" \
        --query "What is Agentic RAG and how is it different from traditional RAG?"

    # Retrieval-only BM25 keyword search
    python agentic_rag_report.py --mode retrieval --config config.fast.yaml \
        --query "What is hierarchical RAG?" \
        --query "dogs playing cards"

    # Retrieval-only with custom BM25 top_k
    python agentic_rag_report.py --mode retrieval --config config.fast.yaml \
        --bm25-top-k 100 \
        --query "dogs playing cards"

2) Colab / notebook:

    from agentic_rag_report import run_smoke_test_entry
    from IPython.display import HTML

    # Agentic RAG (default mode="rag"), using default queries
    html, path = run_smoke_test_entry("config.fast.yaml")
    HTML(html)

    # Retrieval-only BM25 mode with custom queries
    html, path = run_smoke_test_entry(
        "config.fast.yaml",
        queries=["dogs playing cards"],
        mode="retrieval",
        bm25_top_k=50,
    )
    HTML(html)
"""

import argparse
import datetime as _dt
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import yaml  # type: ignore

from agentic_report_html import (
    render_configuration_html,
    render_query_and_answer_html,
    render_context_snippets_html,
    render_sources_html,
    render_eval_metrics_html,
    render_telemetry_html,
    render_cache_summary_html,
    wrap_full_report_html,
)

# Import orchestrator primitives (used only in mode="rag")
from orchestrator import (
    register_default_agents,
    agentic_once_with_metadata,
    clear_telemetry,
    TELEMETRY_EVENTS,
)

# Retrieval cache (HaystackChromaRetrieverAgent uses this, for mode="rag")
from retriever_haystack_agent import RETRIEVAL_QUERY_CACHE

# BM25 / Chroma imports (used only in mode="retrieval")
from haystack_integrations.document_stores.chroma import ChromaDocumentStore
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.retrievers import InMemoryBM25Retriever


# ---------------------------------------------------------------------------
# Built-in default queries
# ---------------------------------------------------------------------------

DEFAULT_QUERIES: List[str] = [
    "What is hierarchical RAG?",
    "What is Agentic RAG and how is it different from traditional RAG?",
    "Explain the role of multi-agent systems in RAG.",
]


# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------

_AGENTS_INITIALIZED: bool = False
_AGENTS_CONFIG_PATH: Optional[str] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def ensure_agents_registered(config_path: str) -> None:
    """
    Make sure orchestrator.register_default_agents() has been called with the
    correct config path. This will also hydrate orchestrator.CONFIG and
    RUNTIME_CONFIG used by build_request_context().

    Only needed for mode="rag".
    """
    global _AGENTS_INITIALIZED, _AGENTS_CONFIG_PATH

    if _AGENTS_INITIALIZED and _AGENTS_CONFIG_PATH == config_path:
        return

    register_default_agents(config_path=config_path)
    _AGENTS_INITIALIZED = True
    _AGENTS_CONFIG_PATH = config_path


def score_to_confidence(raw_score: float, score_type: str = "crossencoder") -> float:
    """
    Map a raw retrieval/reranker score to a [0,1] confidence.

    NOTE: For snippet tables and high-level metrics we now prefer to use
    the raw score directly (mirroring retrieval_automerging.py). This helper
    remains available as a fallback if we ever need a normalized view.
    """
    try:
        rs = float(raw_score)
    except (TypeError, ValueError):
        return 0.0

    if score_type == "cosine":
        # cosine in [-1, 1] -> [0, 1]
        return max(0.0, min(1.0, 0.5 * (rs + 1.0)))

    if score_type == "dot":
        # dot products can be large; scale and clip before sigmoid
        rs_clipped = max(-20.0, min(20.0, rs))
        return 1.0 / (1.0 + math.exp(-rs_clipped / 4.0))

    if score_type == "bm25":
        # bm25 is positive and unbounded; log-compress and clip
        s_log = math.log1p(max(rs, 0.0))
        s_log = max(0.0, min(6.0, s_log))
        return 1.0 / (1.0 + math.exp(-(s_log - 3.0)))

    if score_type == "crossencoder":
        # Cross-encoder / reranker logits: clip into [-6, 6]
        rs_clipped = max(-6.0, min(6.0, rs))
        return 1.0 / (1.0 + math.exp(-rs_clipped))

    # "raw" or unknown: clamp into [0,1]
    return max(0.0, min(1.0, rs))


def infer_level(meta: Dict[str, Any]) -> str:
    """
    Convert raw hierarchical metadata into a human-readable level label.

    We prioritize explicit 'is_leaf' if present; otherwise we inspect 'h_level'.

    For this hierarchy we treat:
      - level >= 2 as 'leaf'
      - level <  2 as 'parent'
    """
    if meta.get("is_leaf") is True:
        return "leaf"
    if meta.get("is_leaf") is False:
        return "parent"

    h_level = meta.get("h_level")
    if h_level is None:
        return "unknown"

    try:
        h_level_int = int(h_level)
    except (TypeError, ValueError):
        return str(h_level)

    if h_level_int >= 2:
        return "leaf"
    else:
        return "parent"


def _enrich_docs_for_bm25(docs: List[Any]) -> None:
    """
    In-place: extend each Document.content with selected metadata fields
    so BM25 can match on:
      - vision captions (parent chunks)
      - filenames / source paths (e.g., dogs_playing_poker.png)

    Adjust META_TEXT_KEYS and CAPTION_META_KEYS to match your schema.
    """
    CAPTION_META_KEYS = [
        "vision_caption",
        "vision_captions",
    ]
    META_TEXT_KEYS = [
        "filename",
        "source_path",
    ]

    for d in docs:
        meta = getattr(d, "meta", None) or {}
        extra_texts: List[str] = []

        # Vision captions
        for key in CAPTION_META_KEYS:
            if key not in meta:
                continue
            value = meta[key]
            if isinstance(value, str):
                extra_texts.append(value)
            elif isinstance(value, (list, tuple)):
                extra_texts.extend(str(v) for v in value if v)

        # Filename / path
        for key in META_TEXT_KEYS:
            val = meta.get(key)
            if isinstance(val, str):
                extra_texts.append(val)

        if not extra_texts:
            continue

        base_content = getattr(d, "content", "") or ""
        merged = base_content
        if merged:
            merged += "\n\n"
        merged += "\n".join(extra_texts)

        d.content = merged  # type: ignore[attr-defined]


# ---- config-based backend classification helpers --------------------------


def _walk_for_key(d: Dict[str, Any], key: str) -> List[Any]:
    """Recursively collect values for a given key inside a nested dict."""
    found: List[Any] = []
    if key in d:
        found.append(d[key])
    for v in d.values():
        if isinstance(v, dict):
            found.extend(_walk_for_key(v, key))
    return found


def _discover_base_url(cfg: Dict[str, Any]) -> str:
    """
    Try hard to find a base_url for the LLM in the config.

    Priority:
      1) llm.api_base / base_url / openai_base_url / llm_base_url
      2) models.api_base / base_url / ...
      3) any of the above anywhere in cfg
    """
    llm_cfg = cfg.get("llm", {}) or {}
    for key in ("api_base", "base_url", "openai_base_url", "llm_base_url", "api_url"):
        vals = _walk_for_key(llm_cfg, key)
        for val in vals:
            if val:
                return str(val)

    models_cfg = cfg.get("models", {}) or {}
    for key in ("api_base", "base_url", "openai_base_url", "llm_base_url", "api_url"):
        vals = _walk_for_key(models_cfg, key)
        for val in vals:
            if val:
                return str(val)

    for key in ("api_base", "base_url", "openai_base_url", "llm_base_url", "api_url"):
        vals = _walk_for_key(cfg, key)
        for val in vals:
            if val:
                return str(val)

    return ""


def _discover_use_local(cfg: Dict[str, Any]) -> Optional[bool]:
    """
    Discover any use_local flag in the config, preferring llm.* then models.*.
    """
    llm_cfg = cfg.get("llm", {}) or {}
    vals = _walk_for_key(llm_cfg, "use_local")
    if not vals:
        models_cfg = cfg.get("models", {}) or {}
        vals = _walk_for_key(models_cfg, "use_local")

    if vals:
        v = vals[0]
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            vl = v.strip().lower()
            if vl in {"true", "1", "yes"}:
                return True
            if vl in {"false", "0", "no"}:
                return False
    return None


def _fallback_backend_label(model_name: Optional[str], backend_raw: Optional[str]) -> str:
    """
    Fallback heuristic when config does not give a clear answer.
    Always returns one of:
      - HF-Local
      - OpenAI-Compatible (vLLM / Ollama)
      - Ollama-Cloud
    """
    backend_l = (backend_raw or "").lower()
    name_l = (model_name or "").lower()

    if "ollama.com" in backend_l or "ollama.com" in name_l:
        return "Ollama-Cloud"

    if any(
        token in backend_l or token in name_l
        for token in ["ollama", "vllm", "openai", "gpt", "/v1"]
    ):
        return "OpenAI-Compatible (vLLM / Ollama)"

    return "HF-Local"


def classify_backend_with_cfg(
    model_name: Optional[str],
    backend_raw: Optional[str],
    cfg: Dict[str, Any],
) -> str:
    """
    Config-aware backend classification.

    Returns exactly one of:
      - HF-Local
      - OpenAI-Compatible (vLLM / Ollama)
      - Ollama-Cloud

    Rules:
      - If llm.api_base (or any base_url) contains 'ollama.com' -> 'Ollama-Cloud'
      - Else if any base_url is set -> 'OpenAI-Compatible (vLLM / Ollama)'
      - Else if use_local is True  -> 'HF-Local'
      - Else                       -> heuristic based on model_name / backend_raw.
    """
    base_url = _discover_base_url(cfg).strip()
    base_url_l = base_url.lower()
    use_local = _discover_use_local(cfg)

    if base_url_l:
        if "ollama.com" in base_url_l:
            return "Ollama-Cloud"
        return "OpenAI-Compatible (vLLM / Ollama)"

    if use_local is True and not base_url_l:
        return "HF-Local"

    return _fallback_backend_label(model_name, backend_raw)


def standardize_phase(phase_raw: Optional[str]) -> str:
    """
    Standardize the phase field to one of:
      - WARMUP
      - QUERY
      - INDEX
      - SYSTEM
    """
    if not phase_raw:
        return "SYSTEM"

    p = str(phase_raw).strip().upper()

    if p in {"WARMUP", "QUERY", "INDEX", "SYSTEM"}:
        return p

    if p.startswith("WARM"):
        return "WARMUP"
    if p.startswith("Q"):
        return "QUERY"
    if p.startswith("IDX") or p.startswith("INGEST"):
        return "INDEX"

    return "SYSTEM"


def build_retriever_settings_from_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract the retriever-related settings from the config dict so that
    they can be displayed in the Configuration section.
    """
    retrieval_cfg = cfg.get("retrieval", {})
    qe_cfg = cfg.get("query_expansion", {}) or cfg.get("qe", {})
    prf_cfg = cfg.get("prf", {})

    settings: Dict[str, Any] = {
        "retrieval.leaf_only": retrieval_cfg.get("leaf_only"),
        "retrieval.enable_hybrid": retrieval_cfg.get("enable_hybrid"),
        "retrieval.leaf_top_k": retrieval_cfg.get("leaf_top_k"),
        "retrieval.bm25_top_k": retrieval_cfg.get("bm25_top_k"),
        "retrieval.enable_rerank": retrieval_cfg.get("enable_rerank"),
        "retrieval.rerank_top_k": retrieval_cfg.get("rerank_top_k"),
        "qe.enabled": qe_cfg.get("enable") or qe_cfg.get("enabled"),
        "qe.num_variants": qe_cfg.get("num_variants"),
        "prf.enabled": prf_cfg.get("prf_enable") or prf_cfg.get("enabled"),
        "prf.docs": prf_cfg.get("prf_docs"),
        "prf.terms": prf_cfg.get("prf_terms"),
    }
    return {k: v for k, v in settings.items() if v is not None}


def get_runtime_retrieval_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract runtime settings for BM25 retrieval from the config.

    It supports BOTH of the following patterns:

      # Retrieval-centric
      retrieval:
        leaf_chroma_path: ./chroma_db
        leaf_collection: leaves
        parent_chroma_path: ./chroma_db_parents
        parent_collection: parents
        leaf_only: false
        bm25_top_k: 50

      # Vectorstore-centric (leaf only)
      vectorstore:
        persist_path: ./chroma_db
        collection_name: leaves

      parent_vectorstore:
        persist_path: ./chroma_db_parents
        collection_name: parents

    Resolution order for leaf store:
      1) retrieval.leaf_chroma_path / retrieval.leaf_collection
      2) vectorstore.leaf_chroma_path / vectorstore.leaf_collection
      3) vectorstore.persist_path / vectorstore.collection_name

    Resolution order for parent store:
      1) retrieval.parent_chroma_path / retrieval.parent_collection
      2) parent_vectorstore.persist_path / parent_vectorstore.collection_name

    Returns a dict:
      {
        "leaf_path": Path,
        "leaf_collection": str,
        "parent_path": Optional[Path],
        "parent_collection": Optional[str],
        "leaf_only": bool,
        "bm25_top_k": int,
      }
    """
    vs_cfg = cfg.get("vectorstore", {}) or {}
    rt_cfg = cfg.get("retrieval", {}) or {}
    parent_vs_cfg = cfg.get("parent_vectorstore", {}) or {}

    # ---- Leaf store ----
    leaf_path_raw = (
        rt_cfg.get("leaf_chroma_path")
        or vs_cfg.get("leaf_chroma_path")
        or vs_cfg.get("persist_path")
    )
    leaf_collection = (
        rt_cfg.get("leaf_collection")
        or vs_cfg.get("leaf_collection")
        or vs_cfg.get("collection_name")
    )

    if not leaf_path_raw or not leaf_collection:
        raise ValueError(
            "Config is missing a usable leaf Chroma path / collection.\n"
            "Expected one of:\n"
            "  retrieval.leaf_chroma_path + retrieval.leaf_collection\n"
            "  vectorstore.leaf_chroma_path + vectorstore.leaf_collection\n"
            "  vectorstore.persist_path + vectorstore.collection_name"
        )

    leaf_path = Path(str(leaf_path_raw)).expanduser()

    # ---- Parent store (optional) ----
    parent_path_raw = (
        rt_cfg.get("parent_chroma_path")
        or parent_vs_cfg.get("persist_path")
    )
    parent_collection = (
        rt_cfg.get("parent_collection")
        or parent_vs_cfg.get("collection_name")
    )

    parent_path: Optional[Path]
    if parent_path_raw and parent_collection:
        parent_path = Path(str(parent_path_raw)).expanduser()
    else:
        parent_path = None
        parent_collection = None  # type: ignore[assignment]

    leaf_only = bool(rt_cfg.get("leaf_only", True))
    bm25_top_k = int(rt_cfg.get("bm25_top_k", 50))

    return {
        "leaf_path": leaf_path,
        "leaf_collection": str(leaf_collection),
        "parent_path": parent_path,
        "parent_collection": parent_collection,
        "leaf_only": leaf_only,
        "bm25_top_k": bm25_top_k,
    }


def normalize_telemetry_backends(
    telemetry_rows: List[Dict[str, Any]],
    cfg: Dict[str, Any],
) -> None:
    """
    Update telemetry_rows in-place so that backend_display is consistent
    with the configuration (use_local + base_url). The label is ALWAYS one of:
      - HF-Local
      - OpenAI-Compatible (vLLM / Ollama)
      - Ollama-Cloud
    """
    for row in telemetry_rows:
        model = row.get("model")
        backend_raw = row.get("backend")
        row["backend_display"] = classify_backend_with_cfg(
            model_name=model,
            backend_raw=backend_raw,
            cfg=cfg,
        )


def aggregate_agent_metrics(telemetry_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Compute average elapsed times and counts per agent from telemetry rows.

    Each telemetry row is expected to have:
      - agent (str)
      - elapsed_ms (float)  OR elapsed (str with 'ms' suffix)

    Only elapsed times are annotated with units ("ms").
    """
    by_agent: Dict[str, List[float]] = {}

    for row in telemetry_rows:
        agent = str(row.get("agent") or "")
        if not agent:
            continue

        elapsed_ms: Optional[float] = None
        if "elapsed_ms" in row and row["elapsed_ms"] is not None:
            try:
                elapsed_ms = float(row["elapsed_ms"])
            except (TypeError, ValueError):
                elapsed_ms = None
        elif "elapsed" in row and row["elapsed"]:
            text = str(row["elapsed"])
            if text.endswith("ms"):
                text = text[:-2].strip()
            try:
                elapsed_ms = float(text)
            except (TypeError, ValueError):
                elapsed_ms = None

        if elapsed_ms is None:
            continue

        by_agent.setdefault(agent, []).append(elapsed_ms)

    metrics_rows: List[Dict[str, Any]] = []

    for agent, values in sorted(by_agent.items()):
        if not values:
            continue
        avg_ms = sum(values) / len(values)
        n = len(values)
        metrics_rows.append(
            {
                "category": "Agents",
                "metric": f"{agent} – avg elapsed (n={n})",
                "value": f"{avg_ms:.2f} ms",
                "notes": "Average elapsed wall time per telemetry event.",
            }
        )

    return metrics_rows


def compute_high_level_metrics(
    answer_text: str,
    snippet_rows: List[Dict[str, Any]],
    citations: List[Dict[str, Any]],
    query_index: int,
) -> List[Dict[str, Any]]:
    """
    Build per-query Retrieval / Citations / Answer metrics suitable for
    the Evaluation Metrics table. Metric names are prefixed with Q{index}.
    """
    qprefix = f"Q{query_index} – "

    metrics: List[Dict[str, Any]] = []

    # Retrieval metrics
    num_snippets = len(snippet_rows)
    doc_ids = {s.get("doc_id") for s in snippet_rows if s.get("doc_id")}
    num_docs = len(doc_ids)

    confidences = [
        float(s.get("confidence", 0.0))
        for s in snippet_rows
        if s.get("confidence") is not None
    ]
    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
    max_conf = max(confidences) if confidences else 0.0

    metrics.append(
        {
            "category": "Retrieval",
            "metric": qprefix + "Documents with snippets",
            "value": str(num_docs),
            "notes": "",
        }
    )
    metrics.append(
        {
            "category": "Retrieval",
            "metric": qprefix + "Total snippets (context)",
            "value": str(num_snippets),
            "notes": "",
        }
    )
    metrics.append(
        {
            "category": "Retrieval",
            "metric": qprefix + "Average snippet confidence (raw score)",
            "value": f"{avg_conf:.4f}",
            "notes": "Uses raw retrieval/reranker score, mirroring retrieval_automerging.py.",
        }
    )
    metrics.append(
        {
            "category": "Retrieval",
            "metric": qprefix + "Max snippet confidence (raw score)",
            "value": f"{max_conf:.4f}",
            "notes": "Uses raw retrieval/reranker score, mirroring retrieval_automerging.py.",
        }
    )

    # Citation metrics
    num_citations = len(citations)
    cited_docs = {c.get("doc_id") for c in citations if c.get("doc_id")}
    num_cited_docs = len(cited_docs)

    metrics.append(
        {
            "category": "Citations",
            "metric": qprefix + "Number of citations",
            "value": str(num_citations),
            "notes": "",
        }
    )
    metrics.append(
        {
            "category": "Citations",
            "metric": qprefix + "Unique cited documents",
            "value": str(num_cited_docs),
            "notes": "",
        }
    )

    # Answer metrics
    text = answer_text or ""
    answer_chars = len(text)
    answer_words = len(text.split()) if text else 0

    metrics.append(
        {
            "category": "Answer",
            "metric": qprefix + "Answer length (chars)",
            "value": str(answer_chars),
            "notes": "",
        }
    )
    metrics.append(
        {
            "category": "Answer",
            "metric": qprefix + "Answer length (words)",
            "value": str(answer_words),
            "notes": "",
        }
    )

    return metrics


def get_retrieval_cache_stats() -> Dict[str, Any]:
    """
    Extract retrieval cache statistics from RETRIEVAL_QUERY_CACHE so they can
    be rendered in the HTML report.

    This is meaningful primarily in mode="rag" when using HaystackChromaRetrieverAgent.
    In mode="retrieval" (BM25-only), this will typically be all zeros.
    """
    stats = RETRIEVAL_QUERY_CACHE.stats

    try:
        enabled = bool(getattr(stats, "enabled"))
    except Exception:
        enabled = True

    try:
        hits = int(getattr(stats, "hits"))
        misses = int(getattr(stats, "misses"))
        stores = int(getattr(stats, "stores"))
    except Exception:
        hits = misses = stores = 0

    total_lookups = getattr(stats, "total_lookups", hits + misses)
    try:
        hit_rate = float(getattr(stats, "hit_rate"))
    except Exception:
        hit_rate = float(hits) / float(total_lookups) if total_lookups else 0.0

    cache_info: Dict[str, Any] = {
        "enabled": enabled,
        "backend": "QueryCache",
        "capacity": RETRIEVAL_QUERY_CACHE.max_size,
        "current_size": RETRIEVAL_QUERY_CACHE.size,
        "hits": hits,
        "misses": misses,
        "stores": stores,
        "total_lookups": total_lookups,
        "hit_rate": hit_rate,
    }
    return cache_info


# ---------------------------------------------------------------------------
# Agentic hook (mode="rag")
# ---------------------------------------------------------------------------


def _extract_score_and_type(cs: Any) -> Tuple[float, str]:
    """
    Try to extract the 'most raw' score we can find from a context snippet.

    Preference order:
      - raw_score (crossencoder)
      - rerank_score (crossencoder)
      - similarity (cosine)
      - score (crossencoder)
    """
    candidates = [
        ("raw_score", "crossencoder"),
        ("rerank_score", "crossencoder"),
        ("similarity", "cosine"),
        ("score", "crossencoder"),
    ]
    for attr, s_type in candidates:
        val = getattr(cs, attr, None)
        if val is not None:
            try:
                return float(val), s_type
            except (TypeError, ValueError):
                continue
    return 0.0, "crossencoder"


def run_agentic_smoke_query(query_text: str, config_path: str) -> Dict[str, Any]:
    """
    Hook that uses orchestrator.agentic_once_with_metadata()
    to run a single Agentic RAG query and adapt its rich metadata into a
    simplified structure used by the HTML report.

    Used only in mode="rag".
    """
    ensure_agents_registered(config_path=config_path)
    clear_telemetry()

    meta = agentic_once_with_metadata(query_text)

    answer_obj = meta.get("answer")
    answer_text = getattr(answer_obj, "text", "") if answer_obj is not None else ""

    retrieval_results = meta.get("retrieval_results") or []
    context_snippets = meta.get("context_snippets") or []
    citations_obj = meta.get("citations") or []

    # ------------------------------------------------------------------
    # Snippet rows
    # ------------------------------------------------------------------
    snippet_rows: List[Dict[str, Any]] = []
    for idx, cs in enumerate(context_snippets, start=1):
        score_val, score_type = _extract_score_and_type(cs)

        doc_title = getattr(cs, "doc_title", None) or ""
        page = getattr(cs, "page", None)
        source_text = (
            getattr(cs, "translated_text", None)
            or getattr(cs, "source_text", None)
            or ""
        )

        # IMPORTANT: confidence is now just the raw score, mirroring retrieval_automerging.py
        snippet_rows.append(
            {
                "rank": idx,
                "doc_id": getattr(cs, "doc_id", None),
                "score": score_val,
                "score_type": score_type,
                "confidence": score_val,
                "title": doc_title,
                "page": page,
                "text": source_text,
            }
        )

    # ------------------------------------------------------------------
    # Sources
    # ------------------------------------------------------------------
    parent_meta_by_doc: Dict[str, Dict[str, Any]] = {}
    for r in retrieval_results:
        doc_id = str(getattr(r, "doc_id", ""))
        if not doc_id:
            continue
        parent_meta_by_doc[doc_id] = getattr(r, "parent_metadata", {}) or {}

    title_idx: Dict[Tuple[str, str], str] = {}
    level_idx: Dict[Tuple[str, str], Any] = {}
    page_idx: Dict[Tuple[str, str], int] = {}

    for cs in context_snippets:
        doc_id = getattr(cs, "doc_id", None)
        chunk_id = getattr(cs, "chunk_id", None)
        if doc_id is None or chunk_id is None:
            continue
        key = (doc_id, chunk_id)
        if getattr(cs, "doc_title", None):
            title_idx[key] = cs.doc_title  # type: ignore[attr-defined]
        if getattr(cs, "level", None) is not None:
            level_idx[key] = cs.level  # type: ignore[attr-defined]
        if getattr(cs, "page", None) is not None:
            page_idx[key] = cs.page  # type: ignore[attr-defined]

    sources_rows: List[Dict[str, Any]] = []
    citations_rows: List[Dict[str, Any]] = []

    for i, c in enumerate(citations_obj, start=1):
        doc_id_full = getattr(c, "doc_id", "")
        chunk_id = getattr(c, "chunk_id", "")
        page_val = getattr(c, "page", None)

        key = (doc_id_full, chunk_id)

        title = title_idx.get(key)
        if title is None:
            for cs in context_snippets:
                if getattr(cs, "doc_id", None) == doc_id_full and getattr(
                    cs, "doc_title", None
                ):
                    title = cs.doc_title  # type: ignore[attr-defined]
                    break
        if title is None:
            parent_meta = parent_meta_by_doc.get(doc_id_full) or {}
            title = (
                parent_meta.get("title")
                or parent_meta.get("doc_title")
                or parent_meta.get("filename")
                or parent_meta.get("source_path")
                or "(unknown title)"
            )

        if page_val is None:
            page_val = page_idx.get(key)
        page_display = page_val if page_val is not None else "?"

        parent_meta = parent_meta_by_doc.get(doc_id_full) or {}
        level_meta: Dict[str, Any] = {}
        if "__level" in parent_meta:
            level_meta["h_level"] = parent_meta["__level"]
        elif "level" in parent_meta:
            level_meta["h_level"] = parent_meta["level"]

        if key in level_idx:
            level_meta["h_level"] = level_idx[key]

        level_label = infer_level(level_meta)

        ref = f"[S{i}]"
        sources_rows.append(
            {
                "ref": ref,
                "doc_id": doc_id_full,
                "title": title,
                "page": page_display,
                "level": level_label,
            }
        )
        citations_rows.append(
            {
                "ref": ref,
                "doc_id": doc_id_full,
                "chunk_id": chunk_id,
                "page": page_val,
            }
        )

    # ------------------------------------------------------------------
    # Telemetry rows (from global TELEMETRY_EVENTS)
    # ------------------------------------------------------------------
    telemetry_rows: List[Dict[str, Any]] = []
    for ev in TELEMETRY_EVENTS:
        phase_raw = getattr(ev, "phase", None)
        if hasattr(phase_raw, "value"):
            phase_str = str(phase_raw.value)
        else:
            phase_str = str(phase_raw)

        backend_raw = getattr(ev, "backend", None)
        if hasattr(backend_raw, "value"):
            backend_str = str(backend_raw.value if hasattr(backend_raw, "value") else backend_raw)
        else:
            backend_str = str(backend_raw)

        mode_raw = getattr(ev, "mode", None)
        if hasattr(mode_raw, "value"):
            mode_str = str(mode_raw.value)
        else:
            mode_str = str(mode_raw)

        timing = getattr(ev, "timing", None)
        elapsed_ms = getattr(timing, "elapsed_ms", None) if timing is not None else None
        t_iso = getattr(timing, "t_iso", None) if timing is not None else None

        telemetry_rows.append(
            {
                "agent": getattr(ev, "agent", ""),
                "event": getattr(ev, "event_type", ""),
                "phase": standardize_phase(phase_str),
                "elapsed": f"{float(elapsed_ms):.2f} ms" if elapsed_ms is not None else "",
                "elapsed_ms": float(elapsed_ms) if elapsed_ms is not None else None,
                "model": getattr(ev, "model", None),
                "backend": backend_str,
                "backend_display": None,
                "mode": mode_str,
                "iteration": getattr(ev, "iteration", 0),
                "timestamp": t_iso or "",
                "payload": getattr(ev, "payload", None),  # <--- NEW: expose payload
            }
        )

    cache_stats = get_retrieval_cache_stats()

    return {
        "answer": answer_text,
        "snippets": snippet_rows,
        "sources": sources_rows,
        "telemetry": telemetry_rows,
        "cache_stats": cache_stats,
        "citations": citations_rows,
    }


# ---------------------------------------------------------------------------
# Retrieval-only hook (mode="retrieval")
# ---------------------------------------------------------------------------


def build_bm25_retriever(
    cfg: Dict[str, Any],
) -> Tuple[InMemoryBM25Retriever, Dict[str, Any]]:
    rt_cfg = get_runtime_retrieval_cfg(cfg)

    # Leaf store
    leaf_store = ChromaDocumentStore(
        persist_path=str(rt_cfg["leaf_path"]),
        collection_name=rt_cfg["leaf_collection"],
    )
    docs: List[Any] = list(leaf_store.filter_documents())

    # Parent store (if dual-index and configured)
    if not rt_cfg["leaf_only"] and rt_cfg.get("parent_path") and rt_cfg.get("parent_collection"):
        parent_store = ChromaDocumentStore(
            persist_path=str(rt_cfg["parent_path"]),
            collection_name=str(rt_cfg["parent_collection"]),
        )
        parent_docs = parent_store.filter_documents()
        docs.extend(parent_docs)

    # Enrich content with captions + filename/source_path
    _enrich_docs_for_bm25(docs)

    mem_store = InMemoryDocumentStore()
    mem_store.write_documents(docs)

    retriever = InMemoryBM25Retriever(document_store=mem_store)
    return retriever, rt_cfg


def run_retrieval_only_query(
    query_text: str,
    cfg: Dict[str, Any],
    retriever: InMemoryBM25Retriever,
    bm25_top_k: Optional[int] = None,
    runtime_cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Execute a single BM25-only retrieval query and adapt the results into the
    same simplified structure used by the HTML report.

    This completely bypasses the agentic pipeline and LLMs, but it respects
    retrieval.leaf_only:

      - leaf_only = true  → search leaf index only
      - leaf_only = false → search BOTH leaf and parent indexes (dual-index)
    """
    rt_cfg = runtime_cfg or get_runtime_retrieval_cfg(cfg)
    top_k = int(bm25_top_k) if bm25_top_k is not None else int(rt_cfg["bm25_top_k"])

    result = retriever.run(query=query_text, top_k=top_k)
    docs = result.get("documents", []) or []

    # Snippets: one row per retrieved document (leaf or parent)
    snippet_rows: List[Dict[str, Any]] = []
    for idx, d in enumerate(docs, start=1):
        meta = getattr(d, "meta", None) or {}
        doc_id = getattr(d, "id", None) or meta.get("id") or ""
        score_val = float(getattr(d, "score", 0.0) or 0.0)

        title = (
            meta.get("title")
            or meta.get("doc_title")
            or meta.get("filename")
            or meta.get("source_path")
            or ""
        )
        page = meta.get("page")
        text = getattr(d, "content", "") or ""

        # Determine leaf vs parent using existing hierarchy conventions
        level_meta: Dict[str, Any] = {}
        if "__level" in meta:
            level_meta["h_level"] = meta["__level"]
        elif "level" in meta:
            level_meta["h_level"] = meta["level"]
        if "is_leaf" in meta:
            level_meta["is_leaf"] = meta["is_leaf"]

        level_label = infer_level(level_meta)

        snippet_rows.append(
            {
                "rank": idx,
                "doc_id": doc_id,
                "score": score_val,
                "score_type": "bm25",
                "confidence": score_val,  # raw BM25 score as "confidence"
                "title": title,
                "page": page,
                "text": text,
                "level": level_label,
            }
        )

    # Sources: one row per unique document
    sources_rows: List[Dict[str, Any]] = []
    seen_docs: Dict[str, Dict[str, Any]] = {}
    for d in docs:
        meta = getattr(d, "meta", None) or {}
        doc_id = getattr(d, "id", None) or meta.get("id") or ""
        if not doc_id or doc_id in seen_docs:
            continue

        title = (
            meta.get("title")
            or meta.get("doc_title")
            or meta.get("filename")
            or meta.get("source_path")
            or "(unknown title)"
        )
        page = meta.get("page")

        level_meta: Dict[str, Any] = {}
        if "__level" in meta:
            level_meta["h_level"] = meta["__level"]
        elif "level" in meta:
            level_meta["h_level"] = meta["level"]
        if "is_leaf" in meta:
            level_meta["is_leaf"] = meta["is_leaf"]

        level_label = infer_level(level_meta)

        seen_docs[doc_id] = {
            "doc_id": doc_id,
            "title": title,
            "page": page if page is not None else "?",
            "level": level_label,
        }

    for i, (doc_id, row) in enumerate(seen_docs.items(), start=1):
        sources_rows.append(
            {
                "ref": f"[S{i}]",
                "doc_id": doc_id,
                "title": row["title"],
                "page": row["page"],
                "level": row["level"],
            }
        )

    # No citations / telemetry in retrieval-only mode
    citations_rows: List[Dict[str, Any]] = []
    telemetry_rows: List[Dict[str, Any]] = []

    cache_stats = get_retrieval_cache_stats()  # mostly zeros in retrieval-only path

    # We explicitly annotate that there is no generated answer.
    answer_text = (
        "(Retrieval-only mode: no generated answer. "
        "Showing BM25-ranked leaf and parent documents only.)"
    )

    retrieval_index = {
        "mode": "leaf_only" if rt_cfg["leaf_only"] else "dual_index",
        "leaf": {
            "path": str(rt_cfg["leaf_path"]),
            "collection": rt_cfg["leaf_collection"],
        },
        "parent": (
            {
                "path": str(rt_cfg["parent_path"]),
                "collection": rt_cfg["parent_collection"],
            }
            if rt_cfg.get("parent_path") and rt_cfg.get("parent_collection")
            else None
        ),
        "bm25_top_k": top_k,
        "num_results": len(snippet_rows),
    }

    return {
        "answer": answer_text,
        "snippets": snippet_rows,
        "sources": sources_rows,
        "telemetry": telemetry_rows,
        "cache_stats": cache_stats,
        "citations": citations_rows,
        "retrieval_index": retrieval_index,
    }


# ---------------------------------------------------------------------------
# Core reporting logic
# ---------------------------------------------------------------------------


def run_smoke_test(
    config_path: str,
    cfg: Dict[str, Any],
    queries: List[str],
) -> Tuple[str, str]:
    """
    Run the Agentic RAG report generation for the given queries (mode="rag")
    and build an HTML report.

    Returns:
      (report_html, output_file_path)
    """
    retriever_settings = build_retriever_settings_from_cfg(cfg)

    sections: List[str] = []

    sections.append(
        render_configuration_html(
            config_path=config_path,
            raw_cfg=cfg,
            retriever_settings=retriever_settings,
            num_queries=len(queries),
        )
    )

    all_telemetry_rows: List[Dict[str, Any]] = []
    all_metrics_rows: List[Dict[str, Any]] = []
    last_cache_stats: Dict[str, Any] | None = None

    for q_idx, query in enumerate(queries, start=1):
        result = run_agentic_smoke_query(query_text=query, config_path=config_path)

        answer_text = str(result.get("answer") or "")
        snippets_raw: List[Dict[str, Any]] = list(result.get("snippets") or [])
        sources_raw: List[Dict[str, Any]] = list(result.get("sources") or [])
        telemetry_raw: List[Dict[str, Any]] = list(result.get("telemetry") or [])
        cache_stats = result.get("cache_stats")
        citations_rows = list(result.get("citations") or [])

        # Ensure 'confidence' is present; by default it should already be the raw score.
        snippet_rows: List[Dict[str, Any]] = []
        for s in snippets_raw:
            row = dict(s)
            if "confidence" not in row or row["confidence"] is None:
                row["confidence"] = row.get("score", 0.0)
            snippet_rows.append(row)

        telemetry_rows: List[Dict[str, Any]] = telemetry_raw
        all_telemetry_rows.extend(telemetry_rows)

        per_query_metrics = compute_high_level_metrics(
            answer_text=answer_text,
            snippet_rows=snippet_rows,
            citations=citations_rows,
            query_index=q_idx,
        )
        all_metrics_rows.extend(per_query_metrics)

        qa_html = render_query_and_answer_html(
            query_text=query,
            answer_text=answer_text,
        )
        snippets_html = render_context_snippets_html(snippet_rows)
        sources_html = render_sources_html(sources_raw)

        query_block_html = (
            f'<section class="agentic-query-block">'
            f'<h2>Query {q_idx}</h2>'
            f"{qa_html}"
            f"{snippets_html}"
            f"{sources_html}"
            f"</section>"
        )
        sections.append(query_block_html)

        last_cache_stats = cache_stats

    normalize_telemetry_backends(all_telemetry_rows, cfg)

    all_metrics_rows.extend(aggregate_agent_metrics(all_telemetry_rows))
    sections.append(render_eval_metrics_html(all_metrics_rows))
    sections.append(render_telemetry_html(all_telemetry_rows))
    sections.append(render_cache_summary_html(last_cache_stats or {}))

    title = "Agentic RAG Report"
    report_html = wrap_full_report_html(title=title, sections=sections)

    utc_now = _dt.datetime.utcnow()
    timestamp_str = utc_now.strftime("%Y%m%d%H%M%S")
    output_name = f"agentic_report_{timestamp_str}.html"
    output_path = Path(output_name).resolve()
    output_path.write_text(report_html, encoding="utf-8")

    return report_html, str(output_path)


def run_retrieval_smoke_test(
    config_path: str,
    cfg: Dict[str, Any],
    queries: List[str],
    bm25_top_k: Optional[int] = None,
) -> Tuple[str, str]:
    """
    Run the retrieval-only (BM25) report for the given queries (mode="retrieval")
    and build an HTML report.

    This bypasses the agentic pipeline entirely and only exercises BM25 keyword
    search over:
      - leaf index only     when retrieval.leaf_only = true
      - leaf + parent index when retrieval.leaf_only = false (dual-index)
    """
    retriever_settings = build_retriever_settings_from_cfg(cfg)

    sections: List[str] = []

    sections.append(
        render_configuration_html(
            config_path=config_path,
            raw_cfg=cfg,
            retriever_settings=retriever_settings,
            num_queries=len(queries),
        )
    )

    all_metrics_rows: List[Dict[str, Any]] = []
    all_telemetry_rows: List[Dict[str, Any]] = []  # remains empty in retrieval-only
    last_cache_stats: Dict[str, Any] = get_retrieval_cache_stats()

    bm25_retriever, rt_cfg = build_bm25_retriever(cfg)

    for q_idx, query in enumerate(queries, start=1):
        result = run_retrieval_only_query(
            query_text=query,
            cfg=cfg,
            retriever=bm25_retriever,
            bm25_top_k=bm25_top_k,
            runtime_cfg=rt_cfg,
        )

        answer_text = str(result.get("answer") or "")
        snippets_raw: List[Dict[str, Any]] = list(result.get("snippets") or [])
        sources_raw: List[Dict[str, Any]] = list(result.get("sources") or [])
        citations_rows: List[Dict[str, Any]] = list(result.get("citations") or [])

        snippet_rows: List[Dict[str, Any]] = []
        for s in snippets_raw:
            row = dict(s)
            if "confidence" not in row or row["confidence"] is None:
                row["confidence"] = row.get("score", 0.0)
            snippet_rows.append(row)

        per_query_metrics = compute_high_level_metrics(
            answer_text=answer_text,
            snippet_rows=snippet_rows,
            citations=citations_rows,
            query_index=q_idx,
        )
        all_metrics_rows.extend(per_query_metrics)

        qa_html = render_query_and_answer_html(
            query_text=query,
            answer_text=answer_text,
        )
        snippets_html = render_context_snippets_html(snippet_rows)
        sources_html = render_sources_html(sources_raw)

        mode_label = "Leaf Only" if rt_cfg["leaf_only"] else "Dual Index (Leaf + Parent)"

        query_block_html = (
            f'<section class="agentic-query-block retrieval-only">'
            f'<h2>Query {q_idx} – Retrieval Only (BM25, {mode_label})</h2>'
            f"{qa_html}"
            f"{snippets_html}"
            f"{sources_html}"
            f"</section>"
        )
        sections.append(query_block_html)

    sections.append(render_eval_metrics_html(all_metrics_rows))
    sections.append(render_telemetry_html(all_telemetry_rows))
    sections.append(render_cache_summary_html(last_cache_stats or {}))

    title = "Retrieval-Only Report (BM25)"
    report_html = wrap_full_report_html(title=title, sections=sections)

    utc_now = _dt.datetime.utcnow()
    timestamp_str = utc_now.strftime("%Y%m%d%H%M%S")
    output_name = f"retrieval_report_{timestamp_str}.html"
    output_path = Path(output_name).resolve()
    output_path.write_text(report_html, encoding="utf-8")

    return report_html, str(output_path)


# ---------------------------------------------------------------------------
# Convenience entrypoint for Colab / notebooks
# ---------------------------------------------------------------------------


def run_smoke_test_entry(
    config_path: str = "config.fast.yaml",
    queries: Optional[List[str]] = None,
    mode: str = "rag",
    bm25_top_k: Optional[int] = None,
) -> Tuple[str, str]:
    """
    Convenience wrapper for notebooks / Colab.

    Args:
      config_path: Path to the YAML config file.
      queries: Optional list of queries. If None or empty, DEFAULT_QUERIES are used.
      mode: "rag" (full Agentic RAG) or "retrieval" (BM25-only).
      bm25_top_k: Optional override for retrieval.bm25_top_k when mode="retrieval".

    Returns:
      (html_string, output_file_path)
    """
    cfg_path = Path(config_path)
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    if queries is not None and len(queries) > 0:
        query_list = list(queries)
    else:
        query_list = list(DEFAULT_QUERIES)

    if mode == "retrieval":
        # Retrieval-only mode: BM25 over leaf index, no agents / LLM.
        return run_retrieval_smoke_test(
            config_path=str(cfg_path),
            cfg=cfg,
            queries=query_list,
            bm25_top_k=bm25_top_k,
        )

    # Default: full Agentic RAG mode
    ensure_agents_registered(config_path=str(cfg_path))
    return run_smoke_test(
        config_path=str(cfg_path),
        cfg=cfg,
        queries=query_list,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Agentic RAG / Retrieval-Only Report Generator (HTML)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.fast.yaml",
        help="Path to config.fast.yaml (or equivalent). Defaults to 'config.fast.yaml'.",
    )
    parser.add_argument(
        "--query",
        type=str,
        action="append",
        help=(
            "Optional. One or more queries to run. "
            "If provided, these override the built-in default queries. "
            "Can be specified multiple times."
        ),
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["rag", "retrieval"],
        default="rag",
        help=(
            "Execution mode: "
            "'rag' runs the full Agentic RAG pipeline (default), "
            "'retrieval' runs BM25-only keyword retrieval over the leaf index."
        ),
    )
    parser.add_argument(
        "--bm25-top-k",
        type=int,
        default=None,
        help=(
            "Optional override for retrieval.bm25_top_k from config.fast.yaml "
            "when mode='retrieval'. Ignored when mode='rag'."
        ),
    )
    args = parser.parse_args()

    user_queries: Optional[List[str]] = args.query if args.query else None

    html, path = run_smoke_test_entry(
        config_path=args.config,
        queries=user_queries,
        mode=args.mode,
        bm25_top_k=args.bm25_top_k,
    )

    if args.mode == "retrieval":
        print(f"[Retrieval-Only] HTML report written to {path}")
    else:
        print(f"[Agentic] HTML report written to {path}")


if __name__ == "__main__":
    main()
