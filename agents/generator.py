from __future__ import annotations

import os
import time
from dataclasses import dataclass, asdict, is_dataclass
from typing import Any, Dict, List, Optional, Tuple

from core.schemas import Answer

# Try both haystack imports for Document
try:
    from haystack import Document  # type: ignore
except Exception:  # pragma: no cover
    from haystack.dataclasses import Document  # type: ignore


# =============================================================================
# Config helpers – read from config.fast.yaml (no RetrievalConfig needed)
# =============================================================================

def _load_raw_config(path: Optional[str] = "config.fast.yaml") -> Dict[str, Any]:
    """
    Minimal loader that only cares about:

        llm:
          model: ...
          api_base: ...
          api_key: ...
          temperature: ...
          max_tokens: ...
          context_max_chars: ...

    but we don't enforce a schema here; we just read the YAML.
    """
    if not path:
        return {}
    if not os.path.exists(path):
        return {}
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        return {}
    return data


def _load_llm_config(path: Optional[str] = "config.fast.yaml") -> Dict[str, Any]:
    """
    Extract only the `llm`-related config from the shared YAML,
    plus `retrieval.context_max_chars` if present.
    """
    cfg: Dict[str, Any] = {}
    if path:
        raw = _load_raw_config(path)
        cfg = raw.get("llm", {}) or {}
        retrieval = raw.get("retrieval", {}) or {}
        # carry over context_max_chars if present
        if "context_max_chars" in retrieval:
            cfg.setdefault("context_max_chars", retrieval["context_max_chars"])

    # Defaults if missing in YAML
    cfg.setdefault("model", None)
    cfg.setdefault("api_base", None)
    cfg.setdefault("api_key", None)
    cfg.setdefault("temperature", 0.2)
    cfg.setdefault("max_tokens", 512)
    cfg.setdefault("context_max_chars", 4000)

    # Allow env vars to override some settings
    model_env = os.environ.get("AGENTIC_LLM_MODEL")
    if model_env:
        cfg["model"] = model_env

    api_base_env = os.environ.get("AGENTIC_LLM_API_BASE")
    if api_base_env:
        cfg["api_base"] = api_base_env

    api_key_env = os.environ.get("AGENTIC_LLM_API_KEY")
    if api_key_env:
        cfg["api_key"] = api_key_env

    temp_env = os.environ.get("AGENTIC_LLM_TEMPERATURE")
    if temp_env:
        try:
            cfg["temperature"] = float(temp_env)
        except ValueError:
            pass

    return cfg


# =============================================================================
# Small DTOs for clarity
# =============================================================================

@dataclass
class RAGGeneratorOutput:
    answer_text: str
    refs: List[Tuple[int, str, str]]  # (n, human_title, snippet)

# AnswerWrapper removed; using agents_schemas.Answer instead.


# =============================================================================
# Formatting helpers (adapted from retrieval_automerging style, but standalone)
# =============================================================================

def _format_pages(d: Document) -> str:
    """Human-friendly page label for a doc, handling single page or min/max."""
    page = d.meta.get("page")
    if isinstance(page, int):
        return f"p. {page}"
    if isinstance(page, (list, tuple)) and len(page) == 2:
        return f"pp. {page[0]}–{page[1]}"
    return ""


def _format_doc_title(d: Document, idx: int) -> str:
    title = d.meta.get("doc_title") or d.meta.get("file_name") or f"Document {idx+1}"
    pages = _format_pages(d)
    if pages:
        return f"{title} ({pages})"
    return title


def _build_cited_context(docs: List[Document], max_chars: int = 4000) -> Tuple[str, List[Tuple[int, str, str]]]:
    """
    Build a context string with [S1], [S2], ... markers and associated references.

    Returns:
      context_str, refs
    where refs is a list of (n, human_title, snippet).
    """
    parts: List[str] = []
    refs: List[Tuple[int, str, str]] = []
    total_len = 0

    for i, d in enumerate(docs):
        n = i + 1
        title = _format_doc_title(d, i)
        snippet = d.content or ""
        snippet = snippet.strip()
        if not snippet:
            continue
        tagged = f"[S{n}] {snippet}"
        if total_len + len(tagged) > max_chars and parts:
            break
        parts.append(tagged)
        refs.append((n, title, snippet[:200]))
        total_len += len(tagged)

    return "\n\n".join(parts), refs


# =============================================================================
# LLM routing
# =============================================================================

def _resolve_llm(llm_cfg: Dict[str, Any]):
    """
    Simple helper to construct the LLM client.

    You already have LLMRouter in your repo; here we keep this minimal
    so the file is self-contained for the agent.
    """
    from llm_router import LLMRouter

    return LLMRouter(config=llm_cfg)


def _rag_answer(
    query: str,
    docs: List[Document],
    llm_cfg: Dict[str, Any],
    qe_variants: Optional[List[str]] = None,
) -> RAGGeneratorOutput:
    """
    Build a RAG-style prompt with:

      - The user's query
      - Optional QE variants
      - Context snippets with [S1], [S2], ...
    """
    max_chars = int(llm_cfg.get("context_max_chars", 4000) or 4000)

    context_str, refs = _build_cited_context(docs, max_chars=max_chars)

    qe_block = ""
    if qe_variants:
        lines = [f"- {v}" for v in qe_variants if v and v.strip()]
        if lines:
            qe_block = "Query expansions:\n" + "\n".join(lines) + "\n\n"

    prompt = (
        "You are a helpful AI assistant. Use the context snippets to answer the question.\n"
        "Cite snippets as [S1], [S2], etc., when relevant.\n\n"
        f"Question:\n{query}\n\n"
    )
    if qe_block:
        prompt += qe_block

    if context_str.strip():
        prompt += "Context:\n" + context_str + "\n\n"
    else:
        prompt += "Context:\n(no context provided)\n\n"

    prompt += "Answer:\n"

    route = _resolve_llm(llm_cfg)

    completion = route.generate(
        prompt=prompt,
        max_tokens=int(llm_cfg.get("max_tokens", 512) or 512),
        temperature=float(llm_cfg.get("temperature", 0.2) or 0.2),
    )

    if hasattr(completion, "text"):
        answer_text = completion.text
    elif isinstance(completion, dict) and "text" in completion:
        answer_text = completion["text"]
    else:
        answer_text = str(completion)

    answer_text = answer_text.strip()

    return RAGGeneratorOutput(
        answer_text=answer_text,
        refs=refs,
    )


# =============================================================================
# Generator agent wrapper
# =============================================================================

class LLMGeneratorAgent:
    """
    Generator agent that:
      - Builds a cited context from reranked documents
      - Calls the unified LLM route (remote OpenAI-compatible or local HF)
      - Stores `answer` and `citations` into the shared state
      - Emits telemetry for generator.output
    """

    role: str = "generator"

    def __init__(
        self,
        config_path: Optional[str] = "config.fast.yaml",
        config: Optional[Any] = None,
        name: str = "LLMGeneratorAgent",
    ) -> None:
        """
        `config` is an optional object passed by the orchestrator. We only
        need the subset relevant to LLM config. If it's missing, we use
        `_load_llm_config(config_path)`.
        """
        self.name = name
        if config is not None and hasattr(config, "llm"):
            llm_cfg = getattr(config, "llm")
            if isinstance(llm_cfg, dict):
                self.llm_cfg = dict(llm_cfg)
            else:
                self.llm_cfg = _load_llm_config(config_path)
        else:
            self.llm_cfg = _load_llm_config(config_path)

    def describe(self) -> str:
        return "LLMGeneratorAgent: LLM-backed RAG generator using LLMRouter."

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        query: str = state.get("query") or state.get("user_query") or ""
        if not query:
            # Nothing to do
            state.setdefault("answer_text", "")
            state.setdefault("answer", Answer(text=""))
            state.setdefault("citations", [])
            return state

        docs: List[Document] = (
            state.get("reranked_documents")
            or state.get("retrieved_documents")
            or []
        )
        qe_variants: Optional[List[str]] = state.get("qe_variants")

        # Allow orchestrator or caller to override llm_cfg at runtime
        llm_cfg = dict(self.llm_cfg)
        overrides = state.get("llm_config") or state.get("llm_cfg") or {}
        if isinstance(overrides, dict):
            llm_cfg.update(overrides)

        t0 = time.time()
        output = _rag_answer(
            query=query,
            docs=docs[:6],  # top docs for generation
            llm_cfg=llm_cfg,
            qe_variants=qe_variants,
        )
        elapsed_ms = (time.time() - t0) * 1000.0

        # Write into state for downstream agents / reporting
        state["answer_text"] = output.answer_text            # plain string
        state["answer"] = Answer(text=output.answer_text)    # Pydantic Answer model with .text
        state["citations"] = output.refs

        # Optional: attach model info for metrics/debugging
        route = _resolve_llm(llm_cfg)
        model_label = route.model_name or "unknown"

        # Telemetry hook if available
        telemetry = state.get("telemetry")
        if telemetry is not None:
            # Expecting telemetry.record_event(agent_name, event, phase, elapsed_ms, model=None, extra=None)
            try:
                telemetry.record_event(
                    agent=self.name,
                    event="generator.output",
                    phase="QUERY",
                    elapsed_ms=elapsed_ms,
                    model=model_label,
                    extra={"num_docs": len(docs)},
                )
            except TypeError:
                # Backward-compat: older signature without extra/model kwargs
                try:
                    telemetry.record_event(
                        agent=self.name,
                        event="generator.output",
                        phase="QUERY",
                        elapsed_ms=elapsed_ms,
                    )
                except Exception:
                    pass

        return state

    def generate(self, input_obj: Any) -> Any:
        """
        Compatibility wrapper for orchestrator calls.

        Accepts either:
          - a dict-like state, in which case we call run(state) directly; or
          - a dataclass / simple object (e.g., GeneratorInput), which we
            normalize to a dict, call run(...), then return a new object
            that *does* have `answer` and `citations` attributes.
        """
        # If it's already a dict, just pass through to run() and return the dict
        if isinstance(input_obj, dict):
            return self.run(input_obj)

        # If it's a dataclass, convert via asdict
        if is_dataclass(input_obj):
            state: Dict[str, Any] = asdict(input_obj)
        else:
            # Fallback: build a dict from known attributes / __dict__
            state = {}
            for attr in (
                "query",
                "user_query",
                "retrieved_documents",
                "reranked_documents",
                "qe_variants",
                "llm_config",
                "telemetry",
            ):
                if hasattr(input_obj, attr):
                    state[attr] = getattr(input_obj, attr)

            # As a last resort, merge the object's __dict__ if present
            if hasattr(input_obj, "__dict__"):
                for k, v in vars(input_obj).items():
                    state.setdefault(k, v)

        # Run the core generator logic on the normalized dict state
        state = self.run(state)

        # Return something that *definitely* has answer / citations attrs
        from types import SimpleNamespace
        return SimpleNamespace(**state)
