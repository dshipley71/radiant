from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from haystack.dataclasses import Document
from haystack.components.rankers import SentenceTransformersSimilarityRanker
from haystack.utils.device import ComponentDevice

from core.interfaces import RerankAgent
from core.schemas import RerankInput, RerankOutput, RetrievalResult, Snippet


def _load_config(config_path: Optional[str]) -> Dict:
    """
    Minimal loader for config.fast.yaml (or similar), mirroring orchestrator
    semantics but kept local to avoid circular imports.

    Resolution order:
      - If config_path is provided, use it as-is.
      - Else, use AGENTIC_RAG_CONFIG env var if set.
      - Else, fall back to "config.fast.yaml" in CWD.
    """
    if config_path is None:
        config_path = os.getenv("AGENTIC_RAG_CONFIG", "config.fast.yaml")

    path = Path(config_path)
    if not path.is_file():
        return {}

    try:
        with path.open("r", encoding="utf-8") as f:
            if path.suffix.lower() in {".yaml", ".yml"}:
                cfg = yaml.safe_load(f) or {}
            else:
                import json
                cfg = json.load(f) or {}
    except Exception:
        cfg = {}

    return cfg


class BasicRerankAgent(RerankAgent):
    """
    RerankAgent that uses Haystack's SentenceTransformersSimilarityRanker
    (cross-encoder) to reorder parent-level RetrievalResult objects.

    Behavior:
      - Loads rerank configuration from config.fast.yaml under "retrieval":
            retrieval.rerank_model
            retrieval.rerank_device
            retrieval.rerank_top_k
      - Builds a pseudo-Document per parent using the best available text.
        Text priority (prefers actual content for accurate relevance scoring):

            For TEXT DOCUMENTS:
            1. Top snippet.content (actual document text)
            2. Top snippet.text (fallback)

            For IMAGE DOCUMENTS (no text content):
            3. parent_metadata["display_summary"]
            4. parent_metadata["vision_caption"]
            5. parent_metadata["summary_leaf"], ["summary_parent"], ["summary"]
            6. parent_metadata["title"]

      - Calls the cross-encoder with the original user query.
      - Aggregates scores per parent_id and:
            * sorts RetrievalResult objects by that score (descending)
            * overwrites Snippet.score for that parent so downstream reporting
              sees the same cross-encoder scores.
    """

    role: str = "rerank"

    def __init__(self, config_path: Optional[str] = None) -> None:
        self._config_path = config_path
        self._ranker: Optional[SentenceTransformersSimilarityRanker] = None
        self._init_ranker()

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "BasicRerankAgent"

    def describe(self) -> str:
        return (
            "Cross-encoder-based rerank agent that reorders RetrievalResult "
            "parents using SentenceTransformersSimilarityRanker, with text "
            "selection that prioritizes parent-level captions/summaries for "
            "image-heavy documents."
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_ranker(self) -> None:
        cfg = _load_config(self._config_path)
        retr_cfg = cfg.get("retrieval", {}) or {}

        model_id = retr_cfg.get("rerank_model", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        device_str = retr_cfg.get("rerank_device", "cpu")
        top_k = int(retr_cfg.get("rerank_top_k", 100))

        try:
            device = ComponentDevice.from_str(device_str or "cpu")
        except Exception:
            device = ComponentDevice.from_str("cpu")

        try:
            self._ranker = SentenceTransformersSimilarityRanker(
                model=model_id,
                top_k=top_k,
                device=device,
            )
            # Warm up to avoid big first-call latency (best-effort)
            try:
                self._ranker.warm_up()
            except Exception:
                pass
        except Exception:
            # If anything fails, we fall back to a no-op rerank
            self._ranker = None

    @staticmethod
    def _best_text_for_parent(result: RetrievalResult) -> str:
        """
        Choose the best text to represent this parent for cross-encoder scoring.

        Priority for TEXT DOCUMENTS (prefer actual content for accurate relevance scoring):
          1. Top snippet.content (actual document text)
          2. Top snippet.text (fallback)

        Priority for IMAGE DOCUMENTS (no actual text content):
          3. parent_metadata["display_summary"]
          4. parent_metadata["vision_caption"]
          5. parent_metadata["summary_leaf"], ["summary_parent"], ["summary"]
          6. parent_metadata["title"]
        """
        meta = result.parent_metadata or {}
        text: str = ""

        # 1) Prefer actual content from snippets (for text documents)
        if result.snippets:
            best: Snippet = result.snippets[0]
            # Try content field first (actual document text)
            if hasattr(best, 'content') and best.content and isinstance(best.content, str) and best.content.strip():
                text = best.content.strip()
            # Fall back to text field
            elif isinstance(best.text, str) and best.text.strip():
                text = best.text.strip()

        # 2) For image-only documents, use display_summary/vision_caption
        if not text:
            ds = meta.get("display_summary")
            if isinstance(ds, str) and ds.strip():
                text = ds.strip()

        if not text:
            vc = meta.get("vision_caption")
            if isinstance(vc, str) and vc.strip():
                text = vc.strip()

        # 3) other summary fields
        if not text:
            for key in ("summary_leaf", "summary_parent", "summary"):
                val = meta.get(key)
                if isinstance(val, str) and val.strip():
                    text = val.strip()
                    break

        # 4) title as last resort
        if not text:
            title = meta.get("title")
            if isinstance(title, str) and title.strip():
                text = title.strip()

        return text

    @classmethod
    def _build_documents_for_rerank(cls, results: List[RetrievalResult]) -> List[Document]:
        """
        Build one pseudo-Document per parent result using _best_text_for_parent.

        Document.id is set to the parent doc_id, and meta["parent_id"] mirrors it.
        Parents with no usable text are skipped (they keep their original
        ordering and snippet scores).
        """
        docs: List[Document] = []
        for r in results:
            text = cls._best_text_for_parent(r)
            if not isinstance(text, str) or not text.strip():
                continue

            d = Document(
                id=r.doc_id,
                content=text.strip(),
                meta={"parent_id": r.doc_id},
            )
            docs.append(d)
        return docs

    @staticmethod
    def _aggregate_parent_scores(docs: List[Document]) -> Dict[str, float]:
        """
        Aggregate best score per parent_id from reranked Documents.
        """
        scores: Dict[str, float] = {}
        for d in docs:
            meta = d.meta or {}
            parent_id = meta.get("parent_id") or d.id
            score = getattr(d, "score", None)
            if score is None:
                continue
            try:
                f = float(score)
            except (TypeError, ValueError):
                continue
            if parent_id not in scores or f > scores[parent_id]:
                scores[parent_id] = f
        return scores

    # ------------------------------------------------------------------
    # RerankAgent interface
    # ------------------------------------------------------------------

    def rerank(self, inp: RerankInput) -> RerankOutput:
        """
        If a cross-encoder is available, rerank parent-level RetrievalResult
        objects using the query and their best text (parent captions/summaries
        + snippets).

        If the ranker is not initialized or there are no results,
        this is a no-op and returns the input ordering.
        """
        results = inp.results or []
        if not results or self._ranker is None:
            return RerankOutput(results=results)

        # Build docs from parents' best text (captions/summaries + snippets)
        docs = self._build_documents_for_rerank(results)
        if not docs:
            # Nothing to score; keep original order
            return RerankOutput(results=results)

        # Run cross-encoder reranker
        try:
            rres = self._ranker.run(query=inp.query, documents=docs)
            reranked_docs: List[Document] = rres.get("documents", docs) or docs
        except Exception:
            # On any failure, gracefully fall back to original ordering
            return RerankOutput(results=results)

        # Aggregate best score per parent
        parent_scores = self._aggregate_parent_scores(reranked_docs)

        # Propagate reranker scores back into snippets for downstream reporting
        for r in results:
            ps = parent_scores.get(r.doc_id)
            if ps is None:
                continue
            for sn in r.snippets:
                sn.score = ps

        # Sort RetrievalResult list by aggregated parent score
        sorted_results = sorted(
            results,
            key=lambda r: parent_scores.get(r.doc_id, 0.0),
            reverse=True,
        )

        return RerankOutput(results=sorted_results)
