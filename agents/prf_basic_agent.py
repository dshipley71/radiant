from __future__ import annotations

from typing import List, Dict, Any, Optional
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import os
import re
import json

from core.agents_interfaces import PRFAgent
from core.agents_schemas import PRFInput, PRFOutput

# Haystack imports
try:
    from haystack import Document  # haystack < 2.18 style
except Exception:
    from haystack.dataclasses import Document  # haystack >= 2.18 style

from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.components.writers import DocumentWriter
from haystack_integrations.document_stores.chroma import ChromaDocumentStore

# ---------------------------------------------------------------------------
# Minimal retrieval config for PRF (no dependency on retrieval_automerging)
# ---------------------------------------------------------------------------

@dataclass
class PRFRetrievalConfig:
    """
    Minimal subset of retrieval configuration needed for PRF:

      - leaf_chroma_path / leaf_collection : where to read leaf chunks from
      - enable_hybrid                      : whether PRF/BM25 should run
      - bm25_top_k, prf_docs, prf_terms    : PRF parameters
    """
    leaf_chroma_path: str = "./chroma_db"
    leaf_collection: str = "leaves"
    enable_hybrid: bool = True
    bm25_top_k: int = 200
    prf_docs: int = 10
    prf_terms: int = 6


def _load_prf_cfg(config_path: Optional[str]) -> PRFRetrievalConfig:
    """
    Load a minimal retrieval config for PRF.

    Expects a config file shaped like your existing config.fast.yaml, e.g.:

        vectorstore:
          persist_path: ./chroma_db
          collection_name: leaves

        retrieval:
          enable_hybrid: true
          bm25_top_k: 200
          prf_docs: 10
          prf_terms: 6

    If retrieval.* keys are missing, sensible defaults are used.
    """

    here = Path(__file__).resolve().parent

    # Resolve config path:
    #   - explicit argument
    #   - AGENTIC_RAG_CONFIG env var
    #   - ./config.fast.yaml next to this file
    if config_path is None:
        env_path = os.getenv("AGENTIC_RAG_CONFIG")
        config_path = env_path or str(here / "config.fast.yaml")

    cfg_file = Path(config_path)
    raw: Dict[str, Any] = {}

    if cfg_file.exists():
        if cfg_file.suffix.lower() == ".json":
            with cfg_file.open("r", encoding="utf-8") as f:
                raw = json.load(f) or {}
        else:
            # Try YAML first, fall back to JSON
            try:
                import yaml  # type: ignore
                with cfg_file.open("r", encoding="utf-8") as f:
                    raw = yaml.safe_load(f) or {}
            except Exception:
                with cfg_file.open("r", encoding="utf-8") as f:
                    raw = json.load(f) or {}

    retr = raw.get("retrieval", raw) or {}
    vs = raw.get("vectorstore", {}) or {}

    leaf_chroma_path = retr.get("leaf_chroma_path") or vs.get("persist_path") or "./chroma_db"
    leaf_collection = retr.get("leaf_collection") or vs.get("collection_name") or "leaves"

    enable_hybrid = bool(retr.get("enable_hybrid", True))
    bm25_top_k = int(retr.get("bm25_top_k", 200))
    prf_docs = int(retr.get("prf_docs", 10))
    prf_terms = int(retr.get("prf_terms", 6))

    return PRFRetrievalConfig(
        leaf_chroma_path=str(leaf_chroma_path),
        leaf_collection=str(leaf_collection),
        enable_hybrid=enable_hybrid,
        bm25_top_k=bm25_top_k,
        prf_docs=prf_docs,
        prf_terms=prf_terms,
    )


# ---------------------------------------------------------------------------
# Tokenization for PRF (same pattern as retrieval_automerging._TOKEN)
# ---------------------------------------------------------------------------

_TOKEN = re.compile(r"[A-Za-z0-9]+")


def _prf_expand_terms(
    query: str,
    bm25_docs: List[Document],
    max_docs: int,
    max_terms: int,
) -> List[str]:
    """
    Compute PRF terms from the top BM25 docs.

    This mirrors the behavior of retrieval_automerging._prf_expand, but
    returns just the list of candidate PRF terms instead of the full
    augmented query.
    """
    texts: List[str] = [
        (bm25_docs[i].content or "") for i in range(min(max_docs, len(bm25_docs)))
    ]
    toks: List[str] = [
        t for t in _TOKEN.findall(" ".join(texts).lower()) if len(t) > 2
    ]

    uni = Counter(toks)
    big = Counter(zip(toks, toks[1:]))

    mix: List[str] = []

    # Unigrams
    for w, _ in uni.most_common(max_terms * 2):
        mix.append(w)
        if len(mix) >= max_terms:
            break

    # Bigrams
    for (a, b), _ in big.most_common(max_terms):
        if len(mix) >= max_terms + 2:
            break
        mix.append(f"{a} {b}")

    # Dedupe while preserving order
    mix = list(dict.fromkeys(mix))
    return mix[: max_terms + 2]


# ---------------------------------------------------------------------------
# Basic PRF Agent
# ---------------------------------------------------------------------------

class BasicPRFAgent(PRFAgent):
    """
    PRF agent backed by a BM25 index over the leaf Chroma corpus.

    Behavior:
      - Loads a minimal retrieval config from config.fast.yaml (or JSON)
      - If enable_hybrid is False, PRF is effectively disabled
      - Builds an in-memory BM25 index over all leaf documents
      - For each query, runs BM25 and derives PRF terms using token
        frequencies and simple unigram/bigram statistics
      - Returns both the PRF term list and the augmented query

    This is a faithful migration of the PRF behavior from
    retrieval_automerging.py into an agent implementation that can be
    used by the agentic orchestrator.
    """

    role = "prf"

    def __init__(self, config_path: Optional[str] = None) -> None:
        # Load minimal PRF config
        self._cfg: PRFRetrievalConfig = _load_prf_cfg(config_path)

        # PRF parameters
        self._prf_docs: int = int(self._cfg.prf_docs)
        self._prf_terms: int = int(self._cfg.prf_terms)

        # BM25 retriever (may be None if enable_hybrid is False)
        self._bm25: Optional[InMemoryBM25Retriever] = None
        self._build_bm25_index()

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "BasicPRFAgent"

    def describe(self) -> str:
        return (
            "Pseudo-Relevance Feedback (PRF) agent that builds an in-memory "
            "BM25 index over the leaf Chroma corpus and augments queries with "
            "high-frequency terms from top documents."
        )

    # ------------------------------------------------------------------
    # PRFAgent interface
    # ------------------------------------------------------------------

    def compute(self, inp: PRFInput) -> PRFOutput:
        """
        Compute PRF terms and an augmented query based on BM25 hits.

        - If BM25 is not available or disabled, returns the original query
        - Otherwise uses BM25 hits to derive PRF terms and appends them
        """
        if self._bm25 is None:
            # PRF effectively disabled (no BM25 index or hybrid disabled)
            return PRFOutput(prf_terms=[], augmented_query=inp.query)

        # Effective PRF limits: combine config and input.bm25_config.top_k
        max_docs = max(1, int(self._prf_docs))
        max_terms = max(1, int(self._prf_terms))
        if inp.bm25_config and inp.bm25_config.top_k:
            max_docs = min(max_docs, int(inp.bm25_config.top_k))

        res = self._bm25.run(query=inp.query)
        bm25_hits: List[Document] = res.get("documents", []) or []
        if not bm25_hits:
            return PRFOutput(prf_terms=[], augmented_query=inp.query)

        prf_terms = _prf_expand_terms(
            query=inp.query,
            bm25_docs=bm25_hits,
            max_docs=max_docs,
            max_terms=max_terms,
        )

        if not prf_terms:
            return PRFOutput(prf_terms=[], augmented_query=inp.query)

        augmented_query = f"{inp.query} " + " ".join(prf_terms)

        return PRFOutput(
            prf_terms=prf_terms,
            augmented_query=augmented_query.strip(),
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_bm25_index(self) -> None:
        """
        Build the in-memory BM25 index from the leaf ChromaDocumentStore.

        Mirrors the BM25 construction approach from retrieval_automerging.py
        but restricted to the minimal configuration this agent cares about.
        """
        if not bool(self._cfg.enable_hybrid):
            self._bm25 = None
            return

        # Open leaf Chroma store
        leaf_store = ChromaDocumentStore(
            persist_path=self._cfg.leaf_chroma_path,
            collection_name=self._cfg.leaf_collection,
        )

        # Build in-memory document store and BM25 retriever
        mem_store = InMemoryDocumentStore()
        writer = DocumentWriter(mem_store)
        writer.run(documents=leaf_store.filter_documents())

        self._bm25 = InMemoryBM25Retriever(
            document_store=mem_store,
            top_k=max(self._prf_docs, self._cfg.bm25_top_k),
        )
