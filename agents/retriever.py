from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
import json
import os
import re
import time
import logging
import pickle
import hashlib
from collections import defaultdict, OrderedDict
from threading import Lock
from enum import Enum

# Haystack Document compatibility
try:
    from haystack import Document  # haystack < 2.18
except Exception:
    from haystack.dataclasses import Document  # haystack >= 2.18

from haystack import Pipeline
from haystack_integrations.document_stores.chroma import ChromaDocumentStore
from haystack_integrations.components.retrievers.chroma import ChromaQueryTextRetriever
from haystack.components.retrievers.auto_merging_retriever import AutoMergingRetriever

# pgvector imports
from haystack_integrations.document_stores.pgvector import PgvectorDocumentStore
from haystack_integrations.components.retrievers.pgvector import (
    PgvectorEmbeddingRetriever,
    PgvectorKeywordRetriever,
)
from haystack.components.embedders import SentenceTransformersTextEmbedder

from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.components.writers import DocumentWriter

from core.interfaces import RetrieverAgent
from core.schemas import (
    RetrieverInput,
    RetrieverOutput,
    RetrievalResult,
    Snippet,
)
from core.circuit_breaker import get_circuit_breaker

# Module logger
_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Score Normalization utilities for hybrid fusion
# ---------------------------------------------------------------------------


def normalize_scores_minmax(docs: List[Document]) -> List[Document]:
    """
    Normalize document scores using min-max normalization to [0, 1] range.
    
    This ensures fair fusion between dense (typically 0-1 cosine) and 
    sparse (unbounded BM25) scores.
    """
    if not docs:
        return docs
    
    scores = [d.score or 0.0 for d in docs]
    min_score = min(scores)
    max_score = max(scores)
    
    # Avoid division by zero if all scores are the same
    score_range = max_score - min_score
    if score_range == 0:
        for d in docs:
            d.score = 1.0 if max_score > 0 else 0.0
        return docs
    
    for d in docs:
        original_score = d.score or 0.0
        d.score = (original_score - min_score) / score_range
    
    return docs


def normalize_scores_zscore(docs: List[Document]) -> List[Document]:
    """
    Normalize document scores using z-score normalization.
    
    Transforms scores to have mean=0 and std=1, then shifts to positive range.
    """
    if not docs:
        return docs
    
    scores = [d.score or 0.0 for d in docs]
    n = len(scores)
    
    if n == 0:
        return docs
    
    mean = sum(scores) / n
    
    # Calculate standard deviation
    variance = sum((s - mean) ** 2 for s in scores) / n
    std = variance ** 0.5
    
    # Avoid division by zero
    if std == 0:
        for d in docs:
            d.score = 0.5
        return docs
    
    # Z-score normalize and shift to positive range
    for d in docs:
        original_score = d.score or 0.0
        z_score = (original_score - mean) / std
        # Shift to [0, 1] range using sigmoid-like transformation
        d.score = 1 / (1 + 2.718281828 ** (-z_score))
    
    return docs


# ---------------------------------------------------------------------------
# BM25 Index Persistence
# ---------------------------------------------------------------------------


@dataclass 
class BM25IndexCache:
    """
    Persistent cache for BM25 index to avoid rebuilding on startup.
    
    The cache stores:
    - Serialized InMemoryDocumentStore with BM25 index
    - Hash of source documents for invalidation
    - Timestamp for TTL-based expiration
    
    TTL behavior:
    - ttl_hours > 0: Cache expires after specified hours
    - ttl_hours <= 0: Cache never expires based on time (only on document changes)
    """
    cache_dir: str = "data/cache/bm25"
    ttl_hours: float = 24.0  # Cache TTL in hours; 0 or negative = never expire
    _lock: Lock = field(default_factory=Lock, init=False)
    
    def __post_init__(self):
        """Ensure cache directory exists."""
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
    
    def _get_cache_path(self, collection_name: str) -> Path:
        """Get the cache file path for a collection."""
        return Path(self.cache_dir) / f"bm25_{collection_name}.pkl"
    
    def _get_meta_path(self, collection_name: str) -> Path:
        """Get the metadata file path for a collection."""
        return Path(self.cache_dir) / f"bm25_{collection_name}_meta.json"
    
    def _compute_docs_hash(self, docs: List[Document]) -> str:
        """Compute a hash of documents for cache invalidation."""
        # Create a deterministic representation of docs
        doc_data = []
        for d in sorted(docs, key=lambda x: str(x.id)):
            doc_data.append({
                "id": str(d.id),
                "content_hash": hashlib.md5((d.content or "").encode()).hexdigest()[:16],
            })
        data_str = json.dumps(doc_data, sort_keys=True)
        return hashlib.sha256(data_str.encode()).hexdigest()[:32]
    
    def get(self, collection_name: str, docs: List[Document]) -> Optional[InMemoryDocumentStore]:
        """
        Retrieve cached BM25 store if valid.
        
        Returns None if:
        - Cache doesn't exist
        - Cache is expired (TTL) - only if ttl_hours > 0
        - Documents have changed (hash mismatch)
        """
        with self._lock:
            cache_path = self._get_cache_path(collection_name)
            meta_path = self._get_meta_path(collection_name)
            
            if not cache_path.exists() or not meta_path.exists():
                _logger.debug(f"BM25 cache miss for '{collection_name}': cache files not found")
                return None
            
            try:
                # Load metadata
                with meta_path.open("r") as f:
                    meta = json.load(f)
                
                # Check TTL (only if ttl_hours > 0; 0 or negative means never expire)
                if self.ttl_hours > 0:
                    cached_time = meta.get("timestamp", 0)
                    if time.time() - cached_time > self.ttl_hours * 3600:
                        _logger.info(f"BM25 cache expired for '{collection_name}'")
                        return None
                
                # Check document hash
                current_hash = self._compute_docs_hash(docs)
                if meta.get("docs_hash") != current_hash:
                    _logger.info(f"BM25 cache invalidated for '{collection_name}': documents changed")
                    return None
                
                # Load the store
                with cache_path.open("rb") as f:
                    store = pickle.load(f)
                
                _logger.info(f"BM25 cache hit for '{collection_name}'")
                return store
                
            except Exception as e:
                _logger.warning(f"Failed to load BM25 cache for '{collection_name}': {e}")
                return None
    
    def put(self, collection_name: str, docs: List[Document], store: InMemoryDocumentStore) -> bool:
        """
        Cache a BM25 store.
        
        Returns True if successful, False otherwise.
        """
        with self._lock:
            cache_path = self._get_cache_path(collection_name)
            meta_path = self._get_meta_path(collection_name)
            
            try:
                # Save metadata
                meta = {
                    "timestamp": time.time(),
                    "docs_hash": self._compute_docs_hash(docs),
                    "doc_count": len(docs),
                }
                with meta_path.open("w") as f:
                    json.dump(meta, f)
                
                # Save store
                with cache_path.open("wb") as f:
                    pickle.dump(store, f)
                
                _logger.info(f"BM25 index cached for '{collection_name}' ({len(docs)} docs)")
                return True
                
            except Exception as e:
                _logger.warning(f"Failed to cache BM25 store for '{collection_name}': {e}")
                return False
    
    def invalidate(self, collection_name: str) -> None:
        """Invalidate cache for a collection."""
        with self._lock:
            cache_path = self._get_cache_path(collection_name)
            meta_path = self._get_meta_path(collection_name)
            
            for p in [cache_path, meta_path]:
                try:
                    if p.exists():
                        p.unlink()
                except Exception:
                    pass


# Global BM25 cache instance
BM25_INDEX_CACHE = BM25IndexCache()


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
            "backend": str(cfg.backend),
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
# pgvector configuration
# ---------------------------------------------------------------------------


@dataclass
class PgvectorConfig:
    """Configuration for pgvector backend."""
    connection_string: Optional[str] = None
    leaf_table_name: str = "haystack_leaves"
    parent_table_name: str = "haystack_parents"
    embedding_dimension: int = 384
    vector_function: str = "cosine_similarity"
    recreate_table: bool = False
    search_strategy: str = "hnsw"
    hnsw_recreate_index_if_exists: bool = False
    hnsw_index_creation_kwargs: Dict[str, Any] = field(default_factory=dict)
    hnsw_ef_search: Optional[int] = None
    language: str = "english"
    embedder_model: str = "sentence-transformers/all-MiniLM-L12-v2"
    embedder_device: str = "cuda"


# ---------------------------------------------------------------------------
# Minimal retrieval config adapter
# ---------------------------------------------------------------------------


@dataclass
class RetrieverConfig:
    """
    Minimal subset of retrieval configuration needed by this agent.

    Maps to config.fast.yaml like:

        backend: chroma  # or "pgvector"

        vectorstore:
          persist_path: ./chroma_db
          collection_name: leaves

        parent_vectorstore:
          persist_path: ./chroma_db_parents
          collection_name: parents

        pgvector:
          connection_string: postgresql://...
          leaf_table_name: haystack_leaves
          parent_table_name: haystack_parents
          embedding_dimension: 384
          ...

        retrieval:
          leaf_only: false
          parent_sidecar_path: ./run_meta/parents_sidecar.json
          leaf_top_k: 50
          enable_hybrid: true
          bm25_top_k: 200
          bm25_max_index_docs: 50000        # optional cap for BM25 corpus
          merge_threshold: 0.45
          cache_enabled: true               # optional

    """

    # Backend selection
    backend: str = "chroma"  # "chroma" or "pgvector"

    # Chroma paths
    leaf_chroma_path: str = "data/database/chroma_leaf_store"
    leaf_collection: str = "leaves"
    parent_chroma_path: str = "data/database/chroma_parents_store"
    parent_collection: str = "parents"

    # pgvector config
    pgvector: PgvectorConfig = field(default_factory=PgvectorConfig)

    leaf_only: bool = False
    parent_sidecar_path: Optional[str] = "data/metadata/parents_sidecar.json"

    leaf_top_k: int = 50

    # Hybrid + auto-merge
    enable_hybrid: bool = True
    bm25_top_k: int = 200
    merge_threshold: float = 0.45

    # Cache + BM25 corpus sizing
    cache_enabled: bool = True
    bm25_max_index_docs: Optional[int] = None


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

    base_dir = cfg_file.parent

    def _resolve_under_config(p: Optional[str]) -> Optional[str]:
        if not p:
            return None
        p_path = Path(p)
        if not p_path.is_absolute():
            p_path = base_dir / p_path
        return str(p_path.resolve())

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

    # Backend selection (default: chroma)
    backend = str(raw.get("backend", "chroma")).lower()

    # Newer config.fast.yaml uses "vectorstore" / "parent_vectorstore"
    vs_cfg: Dict[str, Any] = raw.get("vectorstore") or {}
    parent_vs_cfg: Dict[str, Any] = raw.get("parent_vectorstore") or {}

    leaf_chroma_path = vs_cfg.get("persist_path") or "data/database/chroma_leaf_store"
    leaf_collection = vs_cfg.get("collection_name") or "leaves"

    parent_chroma_path = parent_vs_cfg.get("persist_path") or "data/database/chroma_parents_store"
    parent_collection = parent_vs_cfg.get("collection_name") or "parents"

    # pgvector configuration
    pg_cfg_raw: Dict[str, Any] = raw.get("pgvector") or {}
    models_cfg: Dict[str, Any] = raw.get("models") or {}
    retr_cfg: Dict[str, Any] = raw.get("retrieval") or {}

    # Build PgvectorConfig
    pg_connection_string = pg_cfg_raw.get("connection_string") or os.getenv("PG_CONN_STR")
    pgvector_cfg = PgvectorConfig(
        connection_string=pg_connection_string,
        leaf_table_name=str(pg_cfg_raw.get("leaf_table_name", "haystack_leaves")),
        parent_table_name=str(pg_cfg_raw.get("parent_table_name", "haystack_parents")),
        embedding_dimension=int(pg_cfg_raw.get("embedding_dimension", 384)),
        vector_function=str(pg_cfg_raw.get("vector_function", "cosine_similarity")),
        recreate_table=bool(pg_cfg_raw.get("recreate_table", False)),
        search_strategy=str(pg_cfg_raw.get("search_strategy", "hnsw")),
        hnsw_recreate_index_if_exists=bool(pg_cfg_raw.get("hnsw_recreate_index_if_exists", False)),
        hnsw_index_creation_kwargs=dict(pg_cfg_raw.get("hnsw_index_creation_kwargs") or {}),
        hnsw_ef_search=pg_cfg_raw.get("hnsw_ef_search"),
        language=str(pg_cfg_raw.get("language", "english")),
        embedder_model=str(models_cfg.get("embedder_model", "sentence-transformers/all-MiniLM-L12-v2")),
        embedder_device=str(models_cfg.get("embedder_device", "cuda")),
    )

    leaf_only = bool(retr_cfg.get("leaf_only", False))
    parent_sidecar_path = retr_cfg.get("parent_sidecar_path") or "data/metadata/parents_sidecar.json"
    leaf_top_k = int(retr_cfg.get("leaf_top_k", 50))

    enable_hybrid = bool(retr_cfg.get("enable_hybrid", True))
    bm25_top_k = int(retr_cfg.get("bm25_top_k", 200))
    merge_threshold = float(retr_cfg.get("merge_threshold", 0.45))

    cache_enabled = bool(retr_cfg.get("cache_enabled", True))
    bm25_max_index_docs = retr_cfg.get("bm25_max_index_docs")
    if bm25_max_index_docs is not None:
        try:
            bm25_max_index_docs = int(bm25_max_index_docs)
        except Exception:
            bm25_max_index_docs = None

    # Env overrides for quick tuning
    if os.getenv("AGENTIC_DISABLE_RETRIEVAL_CACHE", "").strip() in {"1", "true", "TRUE"}:
        cache_enabled = False

    env_bm25_cap = os.getenv("AGENTIC_BM25_MAX_INDEX_DOCS")
    if env_bm25_cap:
        try:
            bm25_max_index_docs = int(env_bm25_cap)
        except Exception:
            pass

    return RetrieverConfig(
        backend=backend,
        leaf_chroma_path=_resolve_under_config(leaf_chroma_path) or str(Path("data/database/chroma_leaf_store").resolve()),
        leaf_collection=str(leaf_collection),
        parent_chroma_path=_resolve_under_config(parent_chroma_path) or str(Path("data/database/chroma_parents_store").resolve()),
        parent_collection=str(parent_collection),
        pgvector=pgvector_cfg,
        leaf_only=leaf_only,
        parent_sidecar_path=_resolve_under_config(parent_sidecar_path) if parent_sidecar_path else None,
        leaf_top_k=leaf_top_k,
        enable_hybrid=enable_hybrid,
        bm25_top_k=bm25_top_k,
        merge_threshold=merge_threshold,
        cache_enabled=cache_enabled,
        bm25_max_index_docs=bm25_max_index_docs,
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


def _enrich_docs_for_bm25(docs: List[Document]) -> None:
    """
    In-place: extend each Document.content with selected metadata fields
    so BM25 can match on:

      - display_summary / vision_caption (esp. for image / parent-like docs)
      - title / filename / source_path

    This enrichment is ONLY used for building the in-memory BM25 index.
    It does NOT modify what is stored in Chroma, and it does NOT affect
    what is fed to the LLM (those still use the original content/meta).
    """
    CAPTION_META_KEYS = [
        "display_summary",
        "vision_caption",
    ]
    META_TEXT_KEYS = [
        "title",
        "filename",
        "source_path",
    ]

    for d in docs:
        meta = d.meta or {}
        extra_texts: List[str] = []

        # Captions / summaries
        for key in CAPTION_META_KEYS:
            if key not in meta:
                continue
            value = meta[key]
            if isinstance(value, str) and value.strip():
                extra_texts.append(value.strip())

        # Filename / path / title
        for key in META_TEXT_KEYS:
            val = meta.get(key)
            if isinstance(val, str) and val.strip():
                extra_texts.append(val.strip())

        if not extra_texts:
            continue

        base_content = d.content or ""
        if not isinstance(base_content, str):
            base_content = str(base_content)

        merged = base_content
        if merged:
            merged += "\n\n"
        merged += "\n".join(extra_texts)

        d.content = merged  # type: ignore[attr-defined]


def _build_bm25_store(
    docs: List[Document],
    collection_name: str = "default",
    use_cache: bool = True,
) -> InMemoryDocumentStore:
    """
    Build an in-memory BM25 index over the given docs.

    Before indexing, we enrich Document.content with selected metadata
    (captions, filenames, titles, etc.) so lexical search can hit those
    fields as well. This enrichment is in-memory only.
    
    If use_cache is True, attempts to load from persistent cache first.
    """
    # Try to load from cache first
    if use_cache:
        cached_store = BM25_INDEX_CACHE.get(collection_name, docs)
        if cached_store is not None:
            return cached_store
    
    # Build new index
    _enrich_docs_for_bm25(docs)

    store = InMemoryDocumentStore()
    writer = DocumentWriter(document_store=store)
    writer.run(documents=docs)
    
    # Cache the new store
    if use_cache:
        BM25_INDEX_CACHE.put(collection_name, docs, store)
    
    return store


def _run_bm25(
    store: Optional[InMemoryDocumentStore],
    query: str,
    top_k: int,
    normalize_scores: bool = True,
) -> List[Document]:
    """
    Run BM25 lexical retrieval using haystack's InMemoryBM25Retriever.
    
    Uses circuit breaker to prevent cascade failures.
    Optionally normalizes scores for fair hybrid fusion.
    """
    if store is None:
        return []
    
    circuit = get_circuit_breaker("bm25_retrieval")
    
    if not circuit.can_execute():
        _logger.warning("BM25 retrieval circuit breaker is OPEN, skipping")
        return []
    
    try:
        retr = InMemoryBM25Retriever(
            document_store=store,
            top_k=top_k,
        )
        res = retr.run(query=query)
        docs = res.get("documents", []) or []
        
        # Normalize BM25 scores for fair fusion with dense scores
        if normalize_scores and docs:
            docs = normalize_scores_minmax(docs)
        
        circuit.record_success()
        return docs
    except Exception as e:
        circuit.record_failure(e)
        _logger.warning(f"BM25 retrieval failed: {e}")
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

    If anything goes wrong (missing collection, empty parent store, etc.),
    it falls back to returning the original docs without merging.
    """

    def __init__(
        self,
        parent_store: Union[ChromaDocumentStore, PgvectorDocumentStore],
        parent_id_field: str,
        merge_threshold: float,
    ) -> None:
        self._parent_store = parent_store
        self._parent_id_field = parent_id_field
        self._threshold = float(merge_threshold)

    def merge(self, docs: List[Document]) -> List[Document]:
        # If nothing to merge or threshold disabled, just return docs
        if not docs or self._threshold <= 0.0:
            return docs

        # Split into candidates (with parent_id) and non-merge docs
        def _has_parent_id(doc: Document) -> bool:
            m = doc.meta or {}
            pid = m.get(self._parent_id_field)
            return isinstance(pid, (str, int)) and str(pid) != ""

        candidates = [d for d in docs if _has_parent_id(d)]
        non_merge_docs = [d for d in docs if d not in candidates]

        if not candidates:
            return docs

        # Run AutoMergingRetriever inside a try/except so missing collections
        # or other Chroma/Haystack issues do NOT kill the pipeline.
        try:
            auto_merge = AutoMergingRetriever(
                threshold=self._threshold, document_store=self._parent_store
            )
            pipe = Pipeline()
            pipe.add_component("merge", auto_merge)
            res = pipe.run({"merge": {"documents": candidates}})
            merged: List[Document] = res.get("merge", {}).get("documents", candidates)
            return _dedupe_docs(merged + non_merge_docs)
        except Exception:
            # Fail-safe: if auto-merge fails, just return the original docs.
            return docs


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
          * Query leaf and parent Chroma/pgvector indices (dense)
          * Optionally fuse BM25 lexical hits (hybrid retrieval)
          * Optionally auto-merge leaf chunks into parents via AutoMergingRetriever
          * Group results by parent id

      - Leaf-only mode:
          * Query only the leaf index (dense, plus optional BM25 lexical fusion)
          * Group results by parent id
          * Optionally enrich parent metadata via parent sidecar JSON

    Supports two backends:
      - "chroma": Uses ChromaDocumentStore and ChromaQueryTextRetriever
      - "pgvector": Uses PgvectorDocumentStore, PgvectorEmbeddingRetriever, and PgvectorKeywordRetriever

    Config keys (under retrieval:):

      leaf_chroma_path, parent_chroma_path,
      leaf_collection, parent_collection,
      parent_sidecar_path,
      leaf_only, enable_hybrid, bm25_top_k,
      bm25_max_index_docs, merge_threshold, leaf_top_k,
      cache_enabled
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

        # Respect cache_enabled from config / env
        self._cache.stats.enabled = bool(self._cfg.cache_enabled)

        # Document stores (Chroma)
        self._leaf_store: Optional[ChromaDocumentStore] = None
        self._parent_store: Optional[ChromaDocumentStore] = None
        self._bm25_store: Optional[InMemoryDocumentStore] = None

        # Document stores (pgvector)
        self._pgvector_leaf_store: Optional[PgvectorDocumentStore] = None
        self._pgvector_parent_store: Optional[PgvectorDocumentStore] = None

        # Text embedder for pgvector (required for PgvectorEmbeddingRetriever)
        self._text_embedder: Optional[SentenceTransformersTextEmbedder] = None

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
            "optional BM25 fusion, and optional auto-merge of leaf chunks into parents. "
            "Supports both Chroma and pgvector backends."
        )

    def warm_up(self) -> None:
        """
        Eagerly initialize all lazy components to avoid first-query latency issues.
        
        This ensures:
        - Document stores are connected
        - BM25 index is built (if hybrid enabled)
        - Text embedder is warmed up (for pgvector)
        
        Call this after agent registration to prevent "run twice" issues.
        """
        try:
            if self._cfg.backend == "pgvector":
                # Initialize pgvector stores
                self._ensure_pgvector_leaf_store()
                try:
                    self._ensure_pgvector_parent_store()
                except Exception:
                    pass  # Parent store is optional
                # Warm up text embedder
                self._ensure_text_embedder()
            else:
                # Initialize Chroma stores
                self._ensure_leaf_store()
                try:
                    self._ensure_parent_store()
                except Exception:
                    pass  # Parent store is optional
            
            # Build BM25 index if hybrid is enabled
            if self._cfg.enable_hybrid:
                self._ensure_bm25_store()
                
        except Exception as e:
            # Log but don't fail - components will be initialized on first use
            import logging
            logging.getLogger(__name__).warning(f"Retriever warm_up encountered error: {e}")

    # ------------------------------------------------------------------
    # Internal helpers - Chroma
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

    # ------------------------------------------------------------------
    # Internal helpers - pgvector
    # ------------------------------------------------------------------

    def _ensure_pgvector_leaf_store(self) -> PgvectorDocumentStore:
        if self._pgvector_leaf_store is None:
            pg_cfg = self._cfg.pgvector
            store_kwargs: Dict[str, Any] = {
                "table_name": pg_cfg.leaf_table_name,
                "embedding_dimension": pg_cfg.embedding_dimension,
                "vector_function": pg_cfg.vector_function,
                "recreate_table": pg_cfg.recreate_table,
                "search_strategy": pg_cfg.search_strategy,
                "hnsw_recreate_index_if_exists": pg_cfg.hnsw_recreate_index_if_exists,
                "language": pg_cfg.language,
            }
            if pg_cfg.connection_string:
                store_kwargs["connection_string"] = pg_cfg.connection_string
            if pg_cfg.hnsw_index_creation_kwargs:
                store_kwargs["hnsw_index_creation_kwargs"] = pg_cfg.hnsw_index_creation_kwargs
            if pg_cfg.hnsw_ef_search is not None:
                store_kwargs["hnsw_ef_search"] = pg_cfg.hnsw_ef_search
            self._pgvector_leaf_store = PgvectorDocumentStore(**store_kwargs)
        return self._pgvector_leaf_store

    def _ensure_pgvector_parent_store(self) -> PgvectorDocumentStore:
        if self._pgvector_parent_store is None:
            pg_cfg = self._cfg.pgvector
            store_kwargs: Dict[str, Any] = {
                "table_name": pg_cfg.parent_table_name,
                "embedding_dimension": pg_cfg.embedding_dimension,
                "vector_function": pg_cfg.vector_function,
                "recreate_table": pg_cfg.recreate_table,
                "search_strategy": pg_cfg.search_strategy,
                "hnsw_recreate_index_if_exists": pg_cfg.hnsw_recreate_index_if_exists,
                "language": pg_cfg.language,
            }
            if pg_cfg.connection_string:
                store_kwargs["connection_string"] = pg_cfg.connection_string
            if pg_cfg.hnsw_index_creation_kwargs:
                store_kwargs["hnsw_index_creation_kwargs"] = pg_cfg.hnsw_index_creation_kwargs
            if pg_cfg.hnsw_ef_search is not None:
                store_kwargs["hnsw_ef_search"] = pg_cfg.hnsw_ef_search
            self._pgvector_parent_store = PgvectorDocumentStore(**store_kwargs)
        return self._pgvector_parent_store

    def _ensure_text_embedder(self) -> SentenceTransformersTextEmbedder:
        """Ensure text embedder is initialized for pgvector embedding retrieval."""
        if self._text_embedder is None:
            pg_cfg = self._cfg.pgvector
            self._text_embedder = SentenceTransformersTextEmbedder(
                model=pg_cfg.embedder_model,
                device=None,  # Will use default based on availability
            )
            self._text_embedder.warm_up()
        return self._text_embedder

    # ------------------------------------------------------------------
    # Internal helpers - BM25
    # ------------------------------------------------------------------

    def _ensure_bm25_store(self) -> Optional[InMemoryDocumentStore]:
        """
        Build BM25 store lazily over leaf *and* parent documents.

        Notes:
        - We enrich Document.content with display_summary / vision_caption /
            title / filename / source_path before indexing, so BM25 can match
            on those fields as well.
        - Parents are included so that image-only parents with captions
            (e.g. vision_caption) become discoverable via lexical queries.
        - On any error (missing collection, Chroma failure), we disable
          hybrid BM25 by returning None.
        """
        if not self._cfg.enable_hybrid:
            return None

        if self._bm25_store is not None:
            return self._bm25_store

        try:
            # Get documents based on backend
            docs: List[Document] = []

            if self._cfg.backend == "pgvector":
                leaf_store = self._ensure_pgvector_leaf_store()
                docs = list(leaf_store.filter_documents())

                parent_store: Optional[PgvectorDocumentStore] = None
                try:
                    parent_store = self._ensure_pgvector_parent_store()
                except Exception:
                    parent_store = None

                if parent_store is not None:
                    parent_docs = list(parent_store.filter_documents())
                    if self._parent_sidecar:
                        for d in parent_docs:
                            side = self._parent_sidecar.get(str(d.id))
                            if side and isinstance(side.get("meta"), dict):
                                side_meta = side["meta"]
                                meta = d.meta or {}
                                for key, val in side_meta.items():
                                    if key not in meta and isinstance(val, str) and val.strip():
                                        meta[key] = val
                                d.meta = meta
                    docs.extend(parent_docs)
            else:
                # Chroma backend
                leaf_store_chroma = self._ensure_leaf_store()
                docs = list(leaf_store_chroma.filter_documents())

                parent_store_chroma: Optional[ChromaDocumentStore] = None
                try:
                    parent_store_chroma = self._ensure_parent_store()
                except Exception:
                    parent_store_chroma = None

                if parent_store_chroma is not None:
                    parent_docs = list(parent_store_chroma.filter_documents())
                    if self._parent_sidecar:
                        for d in parent_docs:
                            side = self._parent_sidecar.get(str(d.id))
                            if side and isinstance(side.get("meta"), dict):
                                side_meta = side["meta"]
                                meta = d.meta or {}
                                for key, val in side_meta.items():
                                    if key not in meta and isinstance(val, str) and val.strip():
                                        meta[key] = val
                                d.meta = meta
                    docs.extend(parent_docs)

            # Optional cap for BM25 corpus size (helps in very large collections)
            if self._cfg.bm25_max_index_docs is not None and self._cfg.bm25_max_index_docs > 0:
                docs = docs[: self._cfg.bm25_max_index_docs]

            if not docs:
                self._bm25_store = None
                return None

            # Build BM25 index with caching
            # Use collection name as cache key for multi-collection support
            collection_name = f"{self._cfg.leaf_collection}_{self._cfg.backend}"
            self._bm25_store = _build_bm25_store(
                docs,
                collection_name=collection_name,
                use_cache=self._cfg.cache_enabled,
            )
            return self._bm25_store
        except Exception as e:
            # On any error, completely disable hybrid BM25.
            _logger.warning(f"Failed to build BM25 store: {e}")
            self._bm25_store = None
            return None

    def _ensure_auto_merge(self) -> Optional[AutoMergeAgent]:
        """
        Lazily construct AutoMergeAgent if merge_threshold > 0 and parent_store is available.
        """
        if self._cfg.merge_threshold <= 0.0:
            return None
        if self._auto_merge is None:
            if self._cfg.backend == "pgvector":
                parent_store = self._ensure_pgvector_parent_store()
            else:
                parent_store = self._ensure_parent_store()
            self._auto_merge = AutoMergeAgent(
                parent_store=parent_store,
                parent_id_field=self._parent_id_field,
                merge_threshold=self._cfg.merge_threshold,
            )
        return self._auto_merge

    def _sidecar_lexical_boost(self, query: str) -> List[Document]:
        """
        Fallback lexical search over the parent sidecar.

        We scan parent_sidecar["content"] and selected meta fields for
        simple substring matches of the query tokens. Any matching parents
        are loaded from the parent Chroma store and returned with a strong
        score boost so they reliably surface.

        This is intentionally simple and robust:
          - works even if dense retrieval misses
          - works even if BM25 ranking isn't ideal
          - only touches parents we already know about via the sidecar
        """
        if not self._parent_sidecar:
            return []

        # Basic tokenization
        raw_tokens = [t.lower() for t in re.findall(r"\w+", query or "")]
        if not raw_tokens:
            return []

        # Filter out common stopwords / question words
        STOPWORDS = {
            "what", "which", "who", "whom", "whose",
            "when", "where", "why", "how",
            "the", "a", "an", "and", "or", "but",
            "are", "is", "was", "were", "be", "been", "being",
            "do", "does", "did",
            "of", "in", "on", "at", "to", "for", "with", "by", "from",
        }

        tokens = [t for t in raw_tokens if len(t) > 2 and t not in STOPWORDS]

        # If everything got filtered out, fall back to raw tokens > 2 chars
        if not tokens:
            tokens = [t for t in raw_tokens if len(t) > 2]

        if not tokens:
            return []

        matching_parent_ids: List[str] = []

        for pid, rec in self._parent_sidecar.items():
            content = rec.get("content") or ""
            meta = rec.get("meta") or {}

            blob_parts: List[str] = []

            # Content from sidecar
            if isinstance(content, str):
                blob_parts.append(content)

            # Meta fields we care about
            for key in ("title", "filename", "vision_caption", "display_summary", "source_path"):
                val = meta.get(key)
                if isinstance(val, str):
                    blob_parts.append(val)

            if not blob_parts:
                continue

            blob = " ".join(blob_parts).lower()

            # Match if ANY meaningful token appears in the blob
            if any(tok in blob for tok in tokens):
                matching_parent_ids.append(str(pid))

        if not matching_parent_ids:
            return []

        # Get documents based on backend
        if self._cfg.backend == "pgvector":
            parent_store = self._ensure_pgvector_parent_store()
        else:
            parent_store = self._ensure_parent_store()

        try:
            docs = parent_store.get_documents_by_id(ids=matching_parent_ids)
        except Exception:
            # Fallback if get_documents_by_id is not available
            try:
                docs = parent_store.filter_documents(filters={"id": {"$in": matching_parent_ids}})
            except Exception:
                return []

        # Give them a big score boost so they bubble to the top
        for d in docs:
            d.score = max(d.score or 0.0, 1_000.0)

        return docs

    # ------------------------------------------------------------------
    # Core retrieval
    # ------------------------------------------------------------------

    def _retrieve_with_chroma(
        self,
        uniq_queries: List[str],
        cfg: RetrieverConfig,
        leaf_only_mode: bool,
    ) -> Tuple[Dict[str, Document], Dict[str, Document]]:
        """Dense retrieval using Chroma backend with circuit breaker protection."""
        circuit = get_circuit_breaker("dense_retrieval")
        
        if not circuit.can_execute():
            _logger.warning("Dense retrieval circuit breaker is OPEN, returning empty results")
            return {}, {}
        
        try:
            leaf_store = self._ensure_leaf_store()
            parent_store: Optional[ChromaDocumentStore] = None
            try:
                parent_store = self._ensure_parent_store()
            except Exception:
                parent_store = None

            # Leaf retriever (dense)
            leaf_retriever = ChromaQueryTextRetriever(
                document_store=leaf_store,
                top_k=int(cfg.leaf_top_k),
            )

            # Parent retriever (dense)
            parent_retriever: Optional[ChromaQueryTextRetriever] = None
            if parent_store is not None:
                parent_retriever = ChromaQueryTextRetriever(
                    document_store=parent_store,
                    top_k=int(cfg.leaf_top_k),
                )

            leaf_docs_by_id: Dict[str, Document] = {}
            parent_docs_by_id: Dict[str, Document] = {}

            for qtext in uniq_queries:
                # Leaf dense
                try:
                    res_leaf = leaf_retriever.run(query=qtext)
                    docs_leaf: List[Document] = res_leaf.get("documents", []) or []
                except Exception as e:
                    _logger.debug(f"Leaf retrieval failed for query: {e}")
                    docs_leaf = []
                for d in docs_leaf:
                    doc_id = str(d.id)
                    prev = leaf_docs_by_id.get(doc_id)
                    if prev is None or (d.score or 0.0) > (prev.score or 0.0):
                        leaf_docs_by_id[doc_id] = d

                # Parent dense
                if parent_retriever is not None:
                    try:
                        res_parent = parent_retriever.run(query=qtext)
                        docs_parent: List[Document] = res_parent.get("documents", []) or []
                    except Exception as e:
                        _logger.debug(f"Parent retrieval failed for query: {e}")
                        docs_parent = []
                    for d in docs_parent:
                        doc_id = str(d.id)
                        prev = parent_docs_by_id.get(doc_id)
                        if prev is None or (d.score or 0.0) > (prev.score or 0.0):
                            parent_docs_by_id[doc_id] = d

            circuit.record_success()
            return leaf_docs_by_id, parent_docs_by_id
            
        except Exception as e:
            circuit.record_failure(e)
            _logger.error(f"Dense retrieval failed: {e}")
            return {}, {}

    def _retrieve_with_pgvector(
        self,
        uniq_queries: List[str],
        cfg: RetrieverConfig,
        leaf_only_mode: bool,
    ) -> Tuple[Dict[str, Document], Dict[str, Document]]:
        """Dense retrieval using pgvector backend with circuit breaker protection."""
        circuit = get_circuit_breaker("dense_retrieval")
        
        if not circuit.can_execute():
            _logger.warning("Dense retrieval circuit breaker is OPEN, returning empty results")
            return {}, {}
        
        try:
            leaf_store = self._ensure_pgvector_leaf_store()
            parent_store: Optional[PgvectorDocumentStore] = None
            try:
                parent_store = self._ensure_pgvector_parent_store()
            except Exception:
                parent_store = None

            text_embedder = self._ensure_text_embedder()

            # Leaf retriever (embedding-based)
            leaf_retriever = PgvectorEmbeddingRetriever(
                document_store=leaf_store,
                top_k=int(cfg.leaf_top_k),
            )

            # Parent retriever (embedding-based)
            parent_retriever: Optional[PgvectorEmbeddingRetriever] = None
            if parent_store is not None:
                parent_retriever = PgvectorEmbeddingRetriever(
                    document_store=parent_store,
                    top_k=int(cfg.leaf_top_k),
                )

            leaf_docs_by_id: Dict[str, Document] = {}
            parent_docs_by_id: Dict[str, Document] = {}

            for qtext in uniq_queries:
                # Embed the query
                try:
                    embed_result = text_embedder.run(text=qtext)
                    query_embedding = embed_result.get("embedding", [])
                except Exception as e:
                    _logger.debug(f"Query embedding failed: {e}")
                    query_embedding = []

                if not query_embedding:
                    continue

                # Leaf dense (embedding-based)
                try:
                    res_leaf = leaf_retriever.run(query_embedding=query_embedding)
                    docs_leaf: List[Document] = res_leaf.get("documents", []) or []
                except Exception as e:
                    _logger.debug(f"Leaf retrieval failed: {e}")
                    docs_leaf = []
                for d in docs_leaf:
                    doc_id = str(d.id)
                    prev = leaf_docs_by_id.get(doc_id)
                    if prev is None or (d.score or 0.0) > (prev.score or 0.0):
                        leaf_docs_by_id[doc_id] = d

                # Parent dense (embedding-based)
                if parent_retriever is not None:
                    try:
                        res_parent = parent_retriever.run(query_embedding=query_embedding)
                        docs_parent: List[Document] = res_parent.get("documents", []) or []
                    except Exception as e:
                        _logger.debug(f"Parent retrieval failed: {e}")
                        docs_parent = []
                    for d in docs_parent:
                        doc_id = str(d.id)
                        prev = parent_docs_by_id.get(doc_id)
                        if prev is None or (d.score or 0.0) > (prev.score or 0.0):
                            parent_docs_by_id[doc_id] = d

            circuit.record_success()
            return leaf_docs_by_id, parent_docs_by_id
            
        except Exception as e:
            circuit.record_failure(e)
            _logger.error(f"Dense retrieval (pgvector) failed: {e}")
            return {}, {}

    def _run_pgvector_keyword_retrieval(
        self,
        uniq_queries: List[str],
        cfg: RetrieverConfig,
        leaf_docs_by_id: Dict[str, Document],
    ) -> None:
        """Run PgvectorKeywordRetriever for BM25-style keyword retrieval."""
        leaf_store = self._ensure_pgvector_leaf_store()

        keyword_retriever = PgvectorKeywordRetriever(
            document_store=leaf_store,
            top_k=int(cfg.bm25_top_k),
        )

        for qtext in uniq_queries:
            try:
                res = keyword_retriever.run(query=qtext)
                docs: List[Document] = res.get("documents", []) or []
            except Exception:
                docs = []
            for d in docs:
                doc_id = str(d.id)
                prev = leaf_docs_by_id.get(doc_id)
                if prev is None or (d.score or 0.0) > (prev.score or 0.0):
                    leaf_docs_by_id[doc_id] = d

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
        seen_queries: set = set()
        uniq_queries: List[str] = []
        for q in queries:
            if q not in seen_queries:
                seen_queries.add(q)
                uniq_queries.append(q)

        # Dense retrieval based on backend
        if cfg.backend == "pgvector":
            leaf_docs_by_id, parent_docs_by_id = self._retrieve_with_pgvector(
                uniq_queries, cfg, leaf_only_mode
            )

            # Normalize dense scores before hybrid fusion
            if leaf_docs_by_id and cfg.enable_hybrid:
                dense_docs = list(leaf_docs_by_id.values())
                normalize_scores_minmax(dense_docs)
                leaf_docs_by_id = {str(d.id): d for d in dense_docs}

            # For pgvector, use PgvectorKeywordRetriever for hybrid retrieval
            if cfg.enable_hybrid:
                self._run_pgvector_keyword_retrieval(uniq_queries, cfg, leaf_docs_by_id)
        else:
            # Chroma backend
            leaf_docs_by_id, parent_docs_by_id = self._retrieve_with_chroma(
                uniq_queries, cfg, leaf_only_mode
            )

            # Normalize dense scores before hybrid fusion
            if leaf_docs_by_id and cfg.enable_hybrid:
                dense_docs = list(leaf_docs_by_id.values())
                normalize_scores_minmax(dense_docs)
                leaf_docs_by_id = {str(d.id): d for d in dense_docs}

            # Optional BM25 fusion over leaf docs (enriched with metadata)
            # BM25 scores are normalized within _run_bm25
            bm25_store = self._ensure_bm25_store()
            if bm25_store is not None:
                for qtext in uniq_queries:
                    bm_docs = _run_bm25(
                        store=bm25_store,
                        query=qtext,
                        top_k=int(cfg.bm25_top_k),
                        normalize_scores=True,  # Normalize BM25 scores for fair fusion
                    )
                    for d in bm_docs:
                        doc_id = str(d.id)
                        prev = leaf_docs_by_id.get(doc_id)
                        if prev is None or (d.score or 0.0) > (prev.score or 0.0):
                            leaf_docs_by_id[doc_id] = d

        # ------------------------------------------------------------------
        # Sidecar lexical fallback over parents
        # ------------------------------------------------------------------
        for qtext in uniq_queries:
            sidecar_parents = self._sidecar_lexical_boost(qtext)
            for d in sidecar_parents:
                doc_id = str(d.id)
                prev = parent_docs_by_id.get(doc_id)
                if prev is None or (d.score or 0.0) > (prev.score or 0.0):
                    parent_docs_by_id[doc_id] = d

        # Collect leaf docs
        leaf_docs: List[Document] = list(leaf_docs_by_id.values())

        # ------------------------------------------------------------------
        # Optional auto-merge (dual-index only)
        # ------------------------------------------------------------------
        if not leaf_only_mode and self._cfg.merge_threshold > 0.0:
            auto_merge = self._ensure_auto_merge()
            if auto_merge is not None and leaf_docs:
                try:
                    leaf_docs = auto_merge.merge(leaf_docs)
                except Exception:
                    # Fail-safe: if auto-merge fails, just skip merging.
                    pass

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
            self._cache.put(inp, cfg, out)

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
            if pid is None:
                # No parent ID found - use the document's own ID as parent
                # This handles both leaf-only mode and orphan documents in dual-index mode
                doc_id = doc.id
                if doc_id is None:
                    # Last resort: generate a unique ID based on content hash
                    import hashlib
                    content_hash = hashlib.md5((doc.content or "").encode()).hexdigest()[:16]
                    return f"orphan_{content_hash}"
                return str(doc_id)
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
        del leaf_only_mode  # currently unused but kept for signature compatibility

        meta = dict(doc.meta or {})

        # Enrich from parent sidecar if available (useful for both leaf-only
        # and dual-index modes; sidecar metadata is exported directly from
        # the parent docs at index time).
        sidecar_meta: Dict[str, Any] = {}
        if self._parent_sidecar:
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
        meta = doc.meta or {}

        # Get actual document content (for reranking/RAG)
        actual_content = ""
        c = doc.content or ""
        if isinstance(c, str):
            actual_content = c.strip()
        else:
            actual_content = str(c)

        # Get display text (prioritize summaries for UI display)
        ds = meta.get("display_summary")
        vc = meta.get("vision_caption")

        display_text = ""
        if isinstance(ds, str) and ds.strip():
            display_text = ds.strip()
        elif isinstance(vc, str) and vc.strip():
            display_text = vc.strip()
        else:
            display_text = actual_content

        # As a last resort, show filename + a note (so we never silently drop)
        if not display_text:
            fname = meta.get("filename") or meta.get("source_path") or ""
            if fname:
                display_text = f"(image-only document: {fname})"
            else:
                display_text = "(no snippet text available)"

        max_chars = 512
        if len(display_text) > max_chars:
            display_text = display_text[: max_chars - 1].rstrip() + "…"

        # Page normalization
        page_raw = meta.get(self._page_field)
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
            text=display_text,
            content=actual_content if actual_content else None,
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
