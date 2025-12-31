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

def _load_raw_config(path: Optional[str] = "../config.fast.yaml") -> Dict[str, Any]:
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


def _load_llm_config(path: Optional[str] = "../config.fast.yaml") -> Dict[str, Any]:
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
    from core.llm_router import LLMRouter

    return LLMRouter(config=llm_cfg)


def _rag_answer(
    query: str,
    docs: List[Document],
    llm_cfg: Dict[str, Any],
    qe_variants: Optional[List[str]] = None,
    history: Optional[List[Any]] = None,
) -> RAGGeneratorOutput:
    """
    Build a RAG-style prompt with:

      - Conversation history (if provided)
      - The user's query
      - Optional QE variants
      - Context snippets with [S1], [S2], ...
    
    If the query already contains embedded context (from orchestrator), 
    we avoid adding duplicate context sections.
    """
    max_chars = int(llm_cfg.get("context_max_chars", 4000) or 4000)

    context_str, refs = _build_cited_context(docs, max_chars=max_chars)

    qe_block = ""
    if qe_variants:
        lines = [f"- {v}" for v in qe_variants if v and v.strip()]
        if lines:
            qe_block = "Query expansions:\n" + "\n".join(lines) + "\n\n"

    # Build history block if provided
    history_block = ""
    if history:
        history_lines = []
        for msg in history:
            # Handle both Message objects (Pydantic) and dicts
            if hasattr(msg, "role"):
                role = msg.role
                content = msg.content
            else:
                role = msg.get("role", "user")
                content = msg.get("content", "")
            
            if role == "user":
                history_lines.append(f"User: {content}")
            elif role == "assistant":
                history_lines.append(f"Assistant: {content}")
        if history_lines:
            history_block = "Conversation history:\n" + "\n".join(history_lines) + "\n\n"

    # Check if the query already has context embedded (from orchestrator)
    # The orchestrator embeds context with patterns like "Context:" and numbered snippets
    query_has_embedded_context = (
        "Context:" in query or 
        "context snippets" in query.lower() or
        "[1]" in query  # Orchestrator uses [1], [2] etc.
    )

    if query_has_embedded_context:
        # Query already has context AND history from orchestrator - use it directly
        # DO NOT add history again - orchestrator already included it
        prompt = query
        
        if qe_block:
            prompt += "\n\n" + qe_block
        
        prompt += "\n\nAnswer:\n"
    else:
        # No embedded context - build traditional RAG prompt
        prompt = (
            "You are a helpful AI assistant. Use ONLY the provided context snippets to "
            "answer the user's question. If the context does not contain enough "
            "information, say so explicitly.\n\n"
            "Important formatting rule:\n"
            "- Do NOT include citation or source tags like [S1], [S2], etc. in your answer.\n"
            "- Just answer in natural language.\n"
        )
        
        # Add history context note if history is present
        if history_block:
            prompt += "- Use the conversation history to resolve pronouns and references in the current question.\n"
        
        prompt += "\n"
        
        # Add history before the current question
        if history_block:
            prompt += history_block
        
        prompt += f"Current question:\n{query}\n\n"
        
        if qe_block:
            prompt += qe_block

        if context_str.strip():
            prompt += "Context:\n" + context_str + "\n\n"
        elif docs:
            # We have docs but couldn't extract context - unusual, just note it
            prompt += "Context:\n(context extraction failed)\n\n"
        else:
            # No docs and no embedded context - this is a genuine no-context situation
            prompt += "Context:\n(no context available - answer based on general knowledge if appropriate)\n\n"

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
    """ Generator agent that:
    - Builds a cited context from reranked documents
    - Calls the unified LLM route (remote OpenAI-compatible or local HF)
    - Stores `answer` and `citations` into the shared state
    - Emits telemetry for generator.output
    """

    role: str = "generator"

    def __init__(
        self,
        config_path: Optional[str] = "../config.fast.yaml",
        config: Optional[Any] = None,
        name: str = "LLMGeneratorAgent",
    ) -> None:
        """
        `config` is usually the full dict loaded from config.fast.yaml
        by orchestrator._load_config(...).

        We prefer that dict if available; otherwise we fall back to
        reading from `config_path` via _load_llm_config().
        """
        self.name = name

        # If we got a full YAML dict, extract llm + retrieval.context_max_chars
        if isinstance(config, dict):
            raw = config or {}
            llm_cfg = raw.get("llm", {}) or {}
            retrieval = raw.get("retrieval", {}) or {}

            # carry over context_max_chars if present
            if "context_max_chars" in retrieval:
                llm_cfg.setdefault("context_max_chars", retrieval["context_max_chars"])

            # Defaults (mirror _load_llm_config)
            llm_cfg.setdefault("model", None)
            llm_cfg.setdefault("api_base", None)
            llm_cfg.setdefault("api_key", None)
            llm_cfg.setdefault("temperature", 0.2)
            llm_cfg.setdefault("max_tokens", 512)
            llm_cfg.setdefault("context_max_chars", 4000)

            # Env overrides
            model_env = os.environ.get("AGENTIC_LLM_MODEL")
            if model_env:
                llm_cfg["model"] = model_env

            api_base_env = os.environ.get("AGENTIC_LLM_API_BASE")
            if api_base_env:
                llm_cfg["api_base"] = api_base_env

            api_key_env = os.environ.get("AGENTIC_LLM_API_KEY")
            if api_key_env:
                llm_cfg["api_key"] = api_key_env

            temp_env = os.environ.get("AGENTIC_LLM_TEMPERATURE")
            if temp_env:
                try:
                    llm_cfg["temperature"] = float(temp_env)
                except ValueError:
                    pass

            self.llm_cfg = llm_cfg

            # Read history config from agentic.history
            hist_cfg = raw.get("agentic", {}).get("history", {})
            self.max_history = hist_cfg.get("max_history", 10)
            self.include_history_in_prompt = hist_cfg.get("include_in_prompt", True)

        # If config is some other object with a `.llm` attribute, keep that path
        elif config is not None and hasattr(config, "llm"):
            llm_cfg = getattr(config, "llm")
            if isinstance(llm_cfg, dict):
                self.llm_cfg = dict(llm_cfg)
            else:
                self.llm_cfg = _load_llm_config(config_path)
            # Default history config when not using full dict config
            self.max_history = 10
            self.include_history_in_prompt = True

        # Fallback: load from a YAML path
        else:
            self.llm_cfg = _load_llm_config(config_path)
            # Default history config when loading from path
            self.max_history = 10
            self.include_history_in_prompt = True

    def describe(self) -> str:
        return "LLMGeneratorAgent: LLM-backed RAG generator using LLMRouter."

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        import logging
        logger = logging.getLogger(__name__)
        
        query: str = state.get("query") or state.get("user_query") or ""
        if not query:
            # Nothing to do
            state.setdefault("answer_text", "")
            state.setdefault("answer", Answer(text=""))
            state.setdefault("citations", [])
            return state

        # Try multiple sources for documents, including context_snippets
        docs: List[Document] = (
            state.get("reranked_documents")
            or state.get("retrieved_documents")
            or []
        )
        
        # If no docs but we have context_snippets, convert them to Documents
        if not docs:
            context_snippets = state.get("context_snippets") or []
            if context_snippets:
                docs = self._convert_context_snippets_to_docs(context_snippets)
        
        qe_variants: Optional[List[str]] = state.get("qe_variants")
        
        # Get conversation history from state and apply config
        history: Optional[List[Dict[str, str]]] = None
        if self.include_history_in_prompt and self.max_history > 0:
            raw_history = state.get("history")
            if raw_history:
                # Truncate to max_history Q&A pairs (max_history * 2 messages)
                max_messages = self.max_history * 2
                history = raw_history[-max_messages:]

        # Allow orchestrator or caller to override llm_cfg at runtime
        llm_cfg = dict(self.llm_cfg)
        overrides = state.get("llm_config") or state.get("llm_cfg") or {}
        if isinstance(overrides, dict):
            llm_cfg.update(overrides)

        # Single LLM router instantiation - reused for all operations in this run
        route = _resolve_llm(llm_cfg)
        model_label = route.model_name or "unknown"

        t0 = time.time()
        output = _rag_answer(
            query=query,
            docs=docs[:6],  # top docs for generation
            llm_cfg=llm_cfg,
            qe_variants=qe_variants,
            history=history,
        )
        elapsed_ms = (time.time() - t0) * 1000.0

        answer_text = output.answer_text
        
        # Detect and handle empty/failed answers
        if not answer_text or not answer_text.strip():
            # Retry once with a simplified prompt using the same router instance
            logger.warning("Empty answer from LLM, attempting retry with simplified prompt")
            
            # Build a simpler prompt for retry
            if docs:
                simple_context = "\n".join([
                    f"[{i+1}] {(d.content or d.meta.get('display_summary', ''))[:500]}"
                    for i, d in enumerate(docs[:3])
                    if d.content or d.meta.get('display_summary')
                ])
                retry_prompt = f"Based on this context:\n{simple_context}\n\nAnswer this question briefly: {query}"
            else:
                retry_prompt = f"Answer this question briefly: {query}"
            
            try:
                # Reuse the same router instance
                retry_response = route.generate(
                    prompt=retry_prompt,
                    max_tokens=int(llm_cfg.get("max_tokens", 512) or 512),
                    temperature=float(llm_cfg.get("temperature", 0.3) or 0.3),
                )
                if retry_response and retry_response.strip():
                    answer_text = retry_response.strip()
                else:
                    answer_text = "(Unable to generate answer - please try again)"
            except Exception as e:
                logger.error(f"Retry also failed: {e}")
                answer_text = "(Error generating answer - please try again)"

        # Write into state for downstream agents / reporting
        state["answer_text"] = answer_text            # plain string
        state["answer"] = Answer(text=answer_text)    # Pydantic Answer model with .text
        state["citations"] = output.refs

        # Telemetry hook if available (model_label already set from single router instance)
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

    def _convert_context_snippets_to_docs(self, context_snippets: List[Any]) -> List[Document]:
        """
        Convert ContextSnippet objects to Haystack Document objects for RAG generation.
        
        ContextSnippet has: doc_id, chunk_id, source_text, translated_text, lang, page, score, level, doc_title
        Document needs: id, content, meta
        """
        docs: List[Document] = []
        for cs in context_snippets:
            # Extract text - prefer translated_text, fall back to source_text
            if hasattr(cs, 'translated_text') and cs.translated_text:
                text = cs.translated_text
            elif hasattr(cs, 'source_text') and cs.source_text:
                text = cs.source_text
            elif isinstance(cs, dict):
                text = cs.get('translated_text') or cs.get('source_text') or ''
            else:
                text = ''
            
            if not text or not text.strip():
                continue
            
            # Extract metadata
            if hasattr(cs, 'doc_id'):
                doc_id = cs.doc_id
                chunk_id = getattr(cs, 'chunk_id', '')
                doc_title = getattr(cs, 'doc_title', None)
                page = getattr(cs, 'page', None)
                score = getattr(cs, 'score', 0.0)
                lang = getattr(cs, 'lang', 'unknown')
            elif isinstance(cs, dict):
                doc_id = cs.get('doc_id', '')
                chunk_id = cs.get('chunk_id', '')
                doc_title = cs.get('doc_title')
                page = cs.get('page')
                score = cs.get('score', 0.0)
                lang = cs.get('lang', 'unknown')
            else:
                doc_id = str(id(cs))
                chunk_id = ''
                doc_title = None
                page = None
                score = 0.0
                lang = 'unknown'
            
            doc = Document(
                id=f"{doc_id}::chunk::{chunk_id}" if chunk_id else str(doc_id),
                content=text.strip(),
                meta={
                    "doc_id": doc_id,
                    "chunk_id": chunk_id,
                    "doc_title": doc_title,
                    "page": page,
                    "score": score,
                    "lang": lang,
                },
            )
            docs.append(doc)
        
        return docs


    def generate(self, input_obj: Any) -> Any:
        """
        Compatibility wrapper for orchestrator calls.

        Accepts either:
          - a dict-like state, in which case we call run(state) directly; or
          - a Pydantic BaseModel (e.g., GeneratorInput), which we convert via model_dump(); or
          - a dataclass / simple object, which we normalize to a dict.
        
        Returns an object with `answer` and `citations` attributes.
        """
        # If it's already a dict, just pass through to run() and return the dict
        if isinstance(input_obj, dict):
            return self.run(input_obj)

        # Check for Pydantic models first (v2 uses model_dump, v1 uses dict)
        if hasattr(input_obj, 'model_dump'):
            # Pydantic v2
            state: Dict[str, Any] = input_obj.model_dump()
        elif hasattr(input_obj, 'dict') and callable(getattr(input_obj, 'dict')):
            # Pydantic v1
            state = input_obj.dict()
        elif is_dataclass(input_obj):
            # Python dataclass
            state = asdict(input_obj)
        else:
            # Fallback: build a dict from known attributes
            state = {}
            for attr in (
                "query",
                "user_query",
                "ctx",
                "plan",
                "context_snippets",      # From GeneratorInput schema
                "retrieved_documents",
                "reranked_documents",
                "qe_variants",
                "llm_config",
                "telemetry",
                "history",
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
