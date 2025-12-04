from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
import json
import os
from collections import defaultdict, OrderedDict
from threading import Lock

# Haystack Document compatibility
try:
    from haystack import Document  # haystack < 2.18
except Exception:
    from haystack.dataclasses import Document  # haystack >= 2.18

from haystack import Pipeline
from haystack_integrations.document_stores.chroma import ChromaDocumentStore
from haystack_integrations.components.retrievers.chroma import ChromaQueryTextRetriever
from haystack.components.retrievers.auto_merging_retriever import AutoMergingRetriever

from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.components.writers import DocumentWriter

from agents_interfaces import RetrieverAgent
from agents_schemas import (
    RetrieverInput,
    RetrieverOutput,
    RetrievalResult,
    Snippet,
)

# ---------------------------------------------------------------------------
# In-process result cache (retrieval is relatively expensive)
# ---------------------------------------------------------------------------


@dataclass
class CacheStats:
    enabled: bool = True
    hits: int = 0
    misses: int = 0
    stores: int = 0

    @property
    def total_lookups(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        if self.total_lookups == 0:
            return 0.0
        return self.hits / float(self.total_lookups)


@dataclass
class QueryCache:
    """
    Simple in-process LRU cache for RetrieverOutput payloads, keyed
    by (query, plan knobs, retrieval config). Used by both the agent
    and the reporting layer.
    """

    max_size: int = 128
    stats: CacheStats = field(default_factory=CacheStats)
    _store: "OrderedDict[str, RetrieverOutput]" = field(
        default_factory=OrderedDict
    )
    _lock: Lock = field(default_factory=Lock)

    @property
    def size(self) -> int:
        """Number of cached entries (for telemetry/reporting)."""
        with self._lock:
            return len(self._store)

    def _make_key(
        self,
        inp: RetrieverInput,
        cfg: "RetrieverConfig",
    ) -> str:
        """
        Build a deterministic key capturing query text, plan knobs, and major
        retrieval configuration.

        We intentionally include:
          - raw query
          - prf_augmented_query
          - expanded_queries
          - retrieval_mode (plan)
          - use_qe / use_prf / use_rerank (plan)
          - leaf_only / enable_hybrid / merge_threshold
          - top_k knobs
          - leaf/parent chroma paths + collections
        """
        plan = inp.plan

        retrieval_mode = getattr(plan, "retrieval_mode", None)
        if retrieval_mode is not None:
            retrieval_mode_val = getattr(retrieval_mode, "value", str(retrieval_mode))
        else:
            retrieval_mode_val = None

        payload: Dict[str, Any] = {
            "query": inp.query,
            "prf_augmented_query": inp.prf_augmented_query,
            "expanded_queries": inp.expanded_queries or [],
            "retrieval_mode": retrieval_mode_val,
            "use_qe": bool(getattr(plan, "use_qe", False)),
            "use_prf": bool(getattr(plan, "use_prf", False)),
            "use_rerank": bool(getattr(plan, "use_rerank", False)),
            "leaf_only": bool(cfg.leaf_only),
            "enable_hybrid": bool(cfg.enable_hybrid),
            "merge_threshold": float(cfg.merge_threshold),
            "leaf_top_k": int(cfg.leaf_top_k),
            "bm25_top_k": int(cfg.bm25_top_k),
            "leaf_chroma_path": str(cfg.leaf_chroma_path),
            "leaf_collection": str(cfg.leaf_collection),
            "parent_chroma_path": str(cfg.parent_chroma_path),
            "parent_collection": str(cfg.parent_collection),
        }
        return json.dumps(payload, sort_keys=True)

    def get(
        self,
        inp: RetrieverInput,
        cfg: "RetrieverConfig",
    ) -> Optional[RetrieverOutput]:
        if not self.stats.enabled:
            return None
        key = self._make_key(inp, cfg)
        with self._lock:
            if key in self._store:
                # LRU: move to end
                self.stats.hits += 1
                val = self._store.pop(key)
                self._store[key] = val
                return val
            self.stats.misses += 1
        return None

    def put(
        self,
        inp: RetrieverInput,
        cfg: "RetrieverConfig",
        out: RetrieverOutput,
    ) -> None:
        if not self.stats.enabled:
            return
        key = self._make_key(inp, cfg)
        with self._lock:
            if key in self._store:
                self._store.pop(key)
            self._store[key] = out
            self.stats.stores += 1
            # Evict LRU if over capacity
            while len(self._store) > self.max_size:
                self._store.popitem(last=False)


# Global cache instance used by agentic_rag_report.py
RETRIEVAL_QUERY_CACHE = QueryCache(max_size=256)


# ---------------------------------------------------------------------------
# Minimal retrieval config adapter
# ---------------------------------------------------------------------------


@dataclass
class RetrieverConfig:
    """
    Minimal subset of retrieval configuration needed by this agent.

    Maps to config.fast.yaml like:

        vectorstore:
          persist_path: ./chroma_db
          collection_name: leaves

        parent_vectorstore:
          persist_path: ./chroma_db_parents
          collection_name: parents

        retrieval:
          leaf_only: false
          parent_sidecar_path: ./run_meta/parents_sidecar.json
          leaf_top_k: 50
          enable_hybrid: true
          bm25_top_k: 200
          merge_threshold: 0.45
    """

    leaf_chroma_path: str = "./chroma_db"
    leaf_collection: str = "leaves"
    parent_chroma_path: str = "./chroma_db_parents"
    parent_collection: str = "parents"
    leaf_only: bool = False
    parent_sidecar_path: Optional[str] = "./run_meta/parents_sidecar.json"

    leaf_top_k: int = 50

    # Hybrid + auto-merge
    enable_hybrid: bool = True
    bm25_top_k: int = 200
    merge_threshold: float = 0.45


def _load_retriever_cfg(config_path: Optional[str]) -> RetrieverConfig:
    """
    Load RetrieverConfig from YAML/JSON.

    Path resolution order:
      1. Explicit config_path
      2. $AGENTIC_RAG_CONFIG
      3. ./config.fast.yaml next to this file
    """
    here = Path(__file__).resolve().parent

    if config_path is None:
        env_path = os.getenv("AGENTIC_RAG_CONFIG")
        config_path = env_path if env_path else str(here / "config.fast.yaml")

    cfg_file = Path(config_path).resolve()
    raw: Dict[str, Any] = {}

    if cfg_file.exists():
        if cfg_file.suffix.lower() in {".yaml", ".yml"}:
            import yaml  # type: ignore

            with cfg_file.open("r", encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}
        else:
            with cfg_file.open("r", encoding="utf-8") as f:
                raw = json.load(f) or {}
    else:
        raw = {}

    # Newer config.fast.yaml uses "vectorstore" / "parent_vectorstore"
    vs_cfg: Dict[str, Any] = raw.get("vectorstore") or {}
    parent_vs_cfg: Dict[str, Any] = raw.get("parent_vectorstore") or {}

    leaf_chroma_path = vs_cfg.get("persist_path") or "./chroma_db"
    leaf_collection = vs_cfg.get("collection_name") or "leaves"

    parent_chroma_path = parent_vs_cfg.get("persist_path") or "./chroma_db_parents"
    parent_collection = parent_vs_cfg.get("collection_name") or "parents"

    retr_cfg: Dict[str, Any] = raw.get("retrieval") or {}

    leaf_only = bool(retr_cfg.get("leaf_only", False))
    parent_sidecar_path = retr_cfg.get("parent_sidecar_path") or "./run_meta/parents_sidecar.json"
    leaf_top_k = int(retr_cfg.get("leaf_top_k", 50))

    enable_hybrid = bool(retr_cfg.get("enable_hybrid", True))
    bm25_top_k = int(retr_cfg.get("bm25_top_k", 200))
    merge_threshold = float(retr_cfg.get("merge_threshold", 0.45))

    return RetrieverConfig(
        leaf_chroma_path=str(Path(leaf_chroma_path).resolve()),
        leaf_collection=str(leaf_collection),
        parent_chroma_path=str(Path(parent_chroma_path).resolve()),
        parent_collection=str(parent_collection),
        leaf_only=leaf_only,
        parent_sidecar_path=str(parent_sidecar_path) if parent_sidecar_path else None,
        leaf_top_k=leaf_top_k,
        enable_hybrid=enable_hybrid,
        bm25_top_k=bm25_top_k,
        merge_threshold=merge_threshold,
    )


# ---------------------------------------------------------------------------
# Parent sidecar (used primarily in leaf-only mode)
# ---------------------------------------------------------------------------


def _load_parent_sidecar(path: Optional[str]) -> Dict[str, Dict[str, Any]]:
    """
    Load parent sidecar JSON produced by hier_indexer.py.

    Expected per entry:

        {
          "id": "...",
          "meta": {...},
          "content": "...",
          "relationships": {...}
        }

    We mainly care about id + meta to enrich titles/pages.
    """
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    try:
        with p.open("r", encoding="utf-8") as f:
            arr = json.load(f) or []
        out: Dict[str, Dict[str, Any]] = {}
        for rec in arr:
            pid = rec.get("id")
            if not pid:
                continue
            out[str(pid)] = {
                "meta": rec.get("meta") or {},
                "content": rec.get("content") or "",
                "relationships": rec.get("relationships") or {},
            }
        return out
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _normalize_query_text(text: str) -> str:
    """
    Simple query normalization: strip whitespace and collapse spaces.
    """
    t = (text or "").strip()
    return " ".join(t.split())


def _build_bm25_store(docs: List[Document]) -> InMemoryDocumentStore:
    """
    Build an in-memory BM25 index over the given docs.
    """
    store = InMemoryDocumentStore()
    writer = DocumentWriter(document_store=store)
    writer.run(documents=docs)
    return store


def _run_bm25(
    store: Optional[InMemoryDocumentStore],
    query: str,
    top_k: int,
) -> List[Document]:
    """
    Run BM25 lexical retrieval using haystack's InMemoryBM25Retriever.
    """
    if store is None:
        return []
    try:
        retr = InMemoryBM25Retriever(document_store=store, top_k=top_k)
        res = retr.run(query=query)
        docs = res.get("documents", []) or []
        return docs
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Auto-merge helper
# ---------------------------------------------------------------------------


class AutoMergeAgent:
    """
    Thin wrapper around Haystack's AutoMergingRetriever.

    Given a list of leaf Documents and a parent store, it:
      - selects merge candidates (those with a valid parent id),
      - runs AutoMergingRetriever,
      - returns merged leaf documents + non-merge docs.
    """

    def __init__(
        self,
        parent_store: ChromaDocumentStore,
        parent_id_field: str,
        merge_threshold: float,
    ) -> None:
        self._parent_store = parent_store
        self._parent_id_field = parent_id_field
        self._threshold = float(merge_threshold)

    def merge(self, docs: List[Document]) -> List[Document]:
        if not docs or self._threshold <= 0.0:
            return docs

        def _has_parent_id(doc: Document) -> bool:
            m = doc.meta or {}
            pid = m.get(self._parent_id_field)
            return isinstance(pid, (str, int)) and str(pid) != ""

        candidates = [d for d in docs if _has_parent_id(d)]
        non_merge_docs = [d for d in docs if d not in candidates]

        if not candidates:
            return docs

        auto_merge = AutoMergingRetriever(
            threshold=self._threshold, document_store=self._parent_store
        )
        pipe = Pipeline()
        pipe.add_component("merge", auto_merge)
        res = pipe.run({"merge": {"documents": candidates}})
        merged: List[Document] = res.get("merge", {}).get("documents", candidates)
        return _dedupe_docs(merged + non_merge_docs)


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------


def _dedupe_docs(docs: List[Document]) -> List[Document]:
    seen = set()
    out: List[Document] = []
    for d in docs:
        if d.id not in seen:
            seen.add(d.id)
            out.append(d)
    return out


def _sort_by_score_desc(docs: List[Document]) -> List[Document]:
    return sorted(docs, key=lambda d: (d.score or 0.0), reverse=True)


# ---------------------------------------------------------------------------
# HybridRetrievalAgent
# ---------------------------------------------------------------------------


class HybridRetrievalAgent(RetrieverAgent):
    """
    RetrieverAgent that supports BOTH:

      - Dual-index mode (default):
          * Query leaf and parent Chroma indices (dense)
          * Optionally fuse BM25 lexical hits (hybrid retrieval)
          * Optionally auto-merge leaf chunks into parents via AutoMergingRetriever
          * Group results by parent id

      - Leaf-only mode:
          * Query only the leaf index (dense, plus optional BM25 lexical fusion)
          * Group results by parent id
          * Optionally enrich parent metadata via parent sidecar JSON

    Config keys (under retrieval:):

      leaf_chroma_path, parent_chroma_path,
      leaf_collection, parent_collection,
      parent_sidecar_path,
      leaf_only, enable_hybrid, bm25_top_k,
      merge_threshold, leaf_top_k
    """

    role = "retriever"

    def __init__(
        self,
        *,
        config_path: Optional[str] = None,
        parent_id_field: str = "__parent_id",
        parent_title_field: str = "title",
        parent_path_field: str = "source_path",
        level_field: str = "__level",
        page_field: str = "page",
        lang_field: str = "language",
        cache: Optional[QueryCache] = None,
    ) -> None:
        self._cfg = _load_retriever_cfg(config_path)
        self._config_path = config_path

        # Field names (can be overridden via __init__, but config usually suffices)
        self._parent_id_field = parent_id_field
        self._parent_title_field = parent_title_field
        self._parent_path_field = parent_path_field
        self._level_field = level_field
        self._page_field = page_field
        self._lang_field = lang_field

        # Shared cache reference (use global if none passed)
        self._cache = cache or RETRIEVAL_QUERY_CACHE

        # Document stores
        self._leaf_store: Optional[ChromaDocumentStore] = None
        self._parent_store: Optional[ChromaDocumentStore] = None
        self._bm25_store: Optional[InMemoryDocumentStore] = None

        # Parent sidecar (for leaf-only mode)
        self._parent_sidecar: Dict[str, Dict[str, Any]] = {}
        if self._cfg.parent_sidecar_path:
            self._parent_sidecar = _load_parent_sidecar(self._cfg.parent_sidecar_path)

        # Auto-merge helper (dual-index mode only)
        self._auto_merge: Optional[AutoMergeAgent] = None

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "HybridRetrievalAgent"

    def describe(self) -> str:
        return (
            "Hybrid retriever that supports dense dual-index or leaf-only modes, "
            "optional BM25 fusion, and optional auto-merge of leaf chunks into parents."
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_leaf_store(self) -> ChromaDocumentStore:
        if self._leaf_store is None:
            self._leaf_store = ChromaDocumentStore(
                persist_path=str(self._cfg.leaf_chroma_path),
                collection_name=self._cfg.leaf_collection,
            )
        return self._leaf_store

    def _ensure_parent_store(self) -> ChromaDocumentStore:
        if self._parent_store is None:
            self._parent_store = ChromaDocumentStore(
                persist_path=str(self._cfg.parent_chroma_path),
                collection_name=self._cfg.parent_collection,
            )
        return self._parent_store

    def _ensure_bm25_store(self) -> Optional[InMemoryDocumentStore]:
        """
        Build BM25 store lazily over all leaf documents.
        """
        if not self._cfg.enable_hybrid:
            return None
        if self._bm25_store is None:
            leaf_store = self._ensure_leaf_store()
            corpus = leaf_store.filter_documents()
            self._bm25_store = _build_bm25_store(corpus)
        return self._bm25_store

    def _ensure_auto_merge(self) -> Optional[AutoMergeAgent]:
        """
        Lazily construct AutoMergeAgent if merge_threshold > 0 and parent_store is available.
        """
        if self._cfg.merge_threshold <= 0.0:
            return None
        if self._auto_merge is None:
            parent_store = self._ensure_parent_store()
            self._auto_merge = AutoMergeAgent(
                parent_store=parent_store,
                parent_id_field=self._parent_id_field,
                merge_threshold=self._cfg.merge_threshold,
            )
        return self._auto_merge

    # ------------------------------------------------------------------
    # Core retrieval
    # ------------------------------------------------------------------

    def retrieve(self, inp: RetrieverInput) -> RetrieverOutput:
        """
        Main retrieval entrypoint for the agentic pipeline.

        Steps:
          0. Determine retrieval mode (leaf-only vs dual-index)
          1. Prepare queries (expanded, PRF-augmented) + deduplicate
          2. Run dense retrieval on leaf (and parent if dual-index)
          3. Optionally run BM25 lexical fusion (hybrid)
          4. Optionally auto-merge leaf chunks into parents (dual-index only)
          5. Group by parent id and build RetrievalResult objects
        """
        cfg = self._cfg
        plan = inp.plan
        mode = getattr(plan, "retrieval_mode", None)

        leaf_only_mode = bool(cfg.leaf_only)
        if mode is not None:
            # Plan can override default: DUAL_INDEX vs LEAF_ONLY
            name = str(mode)
            if "LEAF_ONLY" in name.upper():
                leaf_only_mode = True
            elif "DUAL_INDEX" in name.upper():
                leaf_only_mode = False

        # Check cache
        cached = self._cache.get(inp, cfg)
        if cached is not None:
            return cached

        # Determine queries: expanded + PRF-augmented fallbacks
        queries: List[str] = [inp.query]
        if inp.expanded_queries:
            for qx in inp.expanded_queries:
                if isinstance(qx, str) and qx.strip():
                    queries.append(qx.strip())
        if inp.prf_augmented_query:
            qprf = inp.prf_augmented_query.strip()
            if qprf:
                queries.append(qprf)

        # Normalize queries if requested
        queries = [_normalize_query_text(q) for q in queries if q.strip()]

        # Deduplicate queries while preserving order
        seen: set = set()
        uniq_queries: List[str] = []
        for q in queries:
            if q not in seen:
                seen.add(q)
                uniq_queries.append(q)

        # Leaf and parent stores
        leaf_store = self._ensure_leaf_store()

        parent_store: Optional[ChromaDocumentStore] = None
        if not leaf_only_mode:
            parent_store = self._ensure_parent_store()

        # Leaf retriever (dense)
        leaf_retriever = ChromaQueryTextRetriever(
            document_store=leaf_store,
            top_k=int(cfg.leaf_top_k),
        )

        # Parent retriever (dense, dual-index only)
        parent_retriever: Optional[ChromaQueryTextRetriever] = None
        if not leaf_only_mode and parent_store is not None:
            parent_retriever = ChromaQueryTextRetriever(
                document_store=parent_store,
                top_k=int(cfg.leaf_top_k),
            )

        # Dense retrieval on leaf + parent (if dual-index)
        leaf_docs_by_id: Dict[str, Document] = {}
        parent_docs_by_id: Dict[str, Document] = {}

        for qtext in uniq_queries:
            # Leaf dense
            try:
                res_leaf = leaf_retriever.run(query=qtext)
                docs_leaf: List[Document] = res_leaf.get("documents", []) or []
            except Exception:
                docs_leaf = []
            for d in docs_leaf:
                doc_id = str(d.id)
                prev = leaf_docs_by_id.get(doc_id)
                if prev is None or (d.score or 0.0) > (prev.score or 0.0):
                    leaf_docs_by_id[doc_id] = d

            # Parent dense (dual-index only)
            if not leaf_only_mode and parent_retriever is not None:
                try:
                    res_parent = parent_retriever.run(query=qtext)
                    docs_parent: List[Document] = res_parent.get("documents", []) or []
                except Exception:
                    docs_parent = []
                for d in docs_parent:
                    doc_id = str(d.id)
                    prev = parent_docs_by_id.get(doc_id)
                    if prev is None or (d.score or 0.0) > (prev.score or 0.0):
                        parent_docs_by_id[doc_id] = d

        # Optional BM25 fusion over leaf docs
        bm25_store = self._ensure_bm25_store()
        if bm25_store is not None:
            for qtext in uniq_queries:
                bm_docs = _run_bm25(
                    store=bm25_store,
                    query=qtext,
                    top_k=int(cfg.bm25_top_k),
                )
                for d in bm_docs:
                    doc_id = str(d.id)
                    prev = leaf_docs_by_id.get(doc_id)
                    if prev is None or (d.score or 0.0) > (prev.score or 0.0):
                        leaf_docs_by_id[doc_id] = d

        # Collect leaf docs
        leaf_docs: List[Document] = list(leaf_docs_by_id.values())

        # Optional auto-merge (dual-index only)
        if not leaf_only_mode:
            auto_merge = self._ensure_auto_merge()
            if auto_merge is not None and leaf_docs:
                leaf_docs = auto_merge.merge(leaf_docs)

        # Group by parent id
        groups = self._group_by_parent(
            leaf_docs=leaf_docs,
            parent_docs=list(parent_docs_by_id.values()),
            leaf_only_mode=leaf_only_mode,
        )

        # Build RetrievalResult objects
        results: List[RetrievalResult] = []
        for parent_id, group in groups.items():
            parent_meta = group["parent_meta"]
            leaf_docs_for_parent: List[Document] = group["leaf_docs"]
            parent_docs_for_parent: List[Document] = group["parent_docs"]

            # Prefer leaf docs for snippets, but fall back to parent docs
            # for image-only parents (e.g., dogs_playing_poker.png) that
            # may not have any leaf chunks. This mirrors the legacy
            # retrieval_automerging.py behavior where parent-only docs
            # are still surfaced as context.
            snippet_sources: List[Document] = (
                leaf_docs_for_parent if leaf_docs_for_parent else parent_docs_for_parent
            )

            snippets: List[Snippet] = []
            for d in snippet_sources:
                snippets.append(self._build_snippet(d))

            if not snippets:
                continue

            results.append(
                RetrievalResult(
                    doc_id=str(parent_id),
                    parent_metadata=parent_meta,
                    snippets=snippets,
                )
            )

        out = RetrieverOutput(results=results)

        # Store in cache
        if self._cache is not None:
            self._cache.put(
                inp,
                cfg,
                out,
            )

        return out

    # ------------------------------------------------------------------
    # Grouping and metadata helpers
    # ------------------------------------------------------------------

    def _group_by_parent(
        self,
        *,
        leaf_docs: List[Document],
        parent_docs: List[Document],
        leaf_only_mode: bool,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Group leaf and parent docs by parent id.

        Returns:
            parent_id -> {
                "leaf_docs": [...],
                "parent_docs": [...],
                "parent_meta": {...},
            }
        """
        groups: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {"leaf_docs": [], "parent_docs": [], "parent_meta": {}}
        )

        def _get_parent_id(doc: Document) -> str:
            meta = doc.meta or {}
            pid = meta.get(self._parent_id_field)
            if pid is None and leaf_only_mode:
                # In leaf-only mode, the leaf doc might itself be the parent
                return str(doc.id)
            return str(pid)

        for d in leaf_docs:
            pid = _get_parent_id(d)
            groups[pid]["leaf_docs"].append(d)

        for d in parent_docs:
            pid = str(d.id)
            groups[pid]["parent_docs"].append(d)

        # Build parent_meta for each group
        for parent_id, group in groups.items():
            parent_docs_for_group = group["parent_docs"]
            leaf_docs_for_group = group["leaf_docs"]

            # Representative doc: prefer explicit parent doc, otherwise any leaf doc
            rep_doc: Optional[Document] = None
            if parent_docs_for_group:
                rep_doc = max(
                    parent_docs_for_group, key=lambda d: (d.score or 0.0)
                )
            elif leaf_docs_for_group:
                rep_doc = max(leaf_docs_for_group, key=lambda d: (d.score or 0.0))

            if rep_doc is None:
                continue

            parent_meta = self._build_parent_metadata(
                parent_id=parent_id,
                doc=rep_doc,
                leaf_only_mode=leaf_only_mode,
            )
            group["parent_meta"] = parent_meta

        return groups

    def _build_parent_metadata(
        self,
        parent_id: str,
        doc: Document,
        *,
        leaf_only_mode: bool,
    ) -> Dict[str, Any]:
        """
        Build parent_metadata from representative Document (+ sidecar when leaf-only).

        parent_metadata["title"] is critical: it's used downstream by
        build_context_snippets_from_results() as ContextSnippet.doc_title.
        """
        meta = dict(doc.meta or {})

        # Leaf-only mode: try to enrich from sidecar
        sidecar_meta: Dict[str, Any] = {}
        if leaf_only_mode and self._parent_sidecar:
            side = self._parent_sidecar.get(str(parent_id))
            if side and isinstance(side.get("meta"), dict):
                sidecar_meta = side["meta"]

        # Merge sidecar meta with doc meta (doc wins on conflicts)
        if sidecar_meta:
            merged_meta = dict(sidecar_meta)
            merged_meta.update(meta)
            meta = merged_meta

        # If we have a vision_caption but no display_summary, promote it.
        vc = meta.get("vision_caption")
        if isinstance(vc, str) and vc.strip():
            ds = meta.get("display_summary")
            if not isinstance(ds, str) or not ds.strip():
                meta["display_summary"] = vc.strip()

        title = (
            meta.get(self._parent_title_field)
            or meta.get("filename")
            or meta.get(self._parent_path_field)
            or str(parent_id)
        )

        path = meta.get(self._parent_path_field)

        parent_meta: Dict[str, Any] = dict(meta)
        parent_meta["title"] = title
        if path is not None:
            parent_meta["path"] = path

        return parent_meta

    def _build_snippet(self, doc: Document) -> Snippet:
        """
        Map a Haystack Document to a Snippet.

        Text priority:
          1. meta["display_summary"] (possibly coming from vision_caption)
          2. meta["vision_caption"]
          3. doc.content
          4. empty string
        """
        meta = doc.meta or {}

        # Prefer display_summary / vision_caption (image captions) over raw content,
        # so that image-only parents like dogs_playing_poker.png are properly surfaced.
        ds = meta.get("display_summary")
        vc = meta.get("vision_caption")

        text = ""
        if isinstance(ds, str) and ds.strip():
            text = ds.strip()
        elif isinstance(vc, str) and vc.strip():
            text = vc.strip()
        else:
            c = doc.content or ""
            if isinstance(c, str):
                text = c.strip()

        # Truncate excessively long text for snippet usage
        max_chars = 512
        if len(text) > max_chars:
            text = text[: max_chars - 1].rstrip() + "…"

        # Page normalization
        page_raw = meta.get(self._page_field)
        page: Optional[int]
        try:
            page = int(page_raw) if page_raw is not None else None
        except Exception:
            page = None

        # Level normalization
        level_val = meta.get(self._level_field, "leaf")
        level = str(level_val)

        # Language normalization
        lang_val = meta.get(self._lang_field)
        lang = str(lang_val) if lang_val is not None else None

        return Snippet(
            chunk_id=str(doc.id),
            score=float(doc.score or 0.0),
            text=text,
            lang=lang,
            page=page,
            level=level,
        )


# ---------------------------------------------------------------------------
# Backwards-compatible alias for orchestrator.py
# ---------------------------------------------------------------------------


class HaystackChromaRetrieverAgent(HybridRetrievalAgent):
    """
    Backwards-compatible alias.

    orchestrator.register_default_agents() currently instantiates
    HaystackChromaRetrieverAgent; by subclassing HybridRetrievalAgent
    we inherit the full hybrid + auto-merge + cache behavior without
    changing orchestrator.py.
    """
    pass
