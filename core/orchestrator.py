from __future__ import annotations

import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# Haystack Document compatibility
try:
    from haystack import Document  # haystack < 2.18
except Exception:
    from haystack.dataclasses import Document  # haystack >= 2.18

from .interfaces import BaseAgent
from .schemas import (
    Answer,
    AnswerSection,
    BackendEnum,
    ContextSnippet,
    CriticFeedback,
    CriticInput,
    DecompositionConfig,
    DecompositionInput,
    GlobalConfig,
    GeneratorInput,
    GuardrailInput,
    PhaseEnum,
    Plan,
    PlanIterations,
    PlannerInput,
    PolicyInput,
    PostprocessInput,
    PostprocessPreferences,
    PRFConfig,
    PRFInput,
    QEInput,
    QueryRewriteInput,
    RequestContext,
    RetrieverInput,
    RetrievalMetrics,
    RetrievalResult,
    RerankInput,
    RouterConfig,
    RouterInput,
    RuntimeContext,
    RuntimeModeEnum,
    TelemetryEvent,
    TelemetryTiming,
)

from agents.router import BasicRouterAgent
from agents.decomposition import BasicDecompositionAgent
from agents.planner import BasicPlannerAgent
from agents.guardrail import BasicGuardrailAgent
from agents.critic import BasicCriticAgent
from agents.policy import BasicPolicyAgent
from agents.telemetry import BasicTelemetryAgent
from agents.prf import BasicPRFAgent
from agents.rerank import BasicRerankAgent
from agents.postprocess import BasicPostProcessorAgent
from agents.retriever import HaystackChromaRetrieverAgent
from agents.qe import LLMQEAgent
from agents.generator import LLMGeneratorAgent
from agents.rewrite import LLMQueryRewriteAgent

# ---------------------------------------------------------------------------
# Global config and telemetry buffers
# ---------------------------------------------------------------------------

CONFIG: Dict[str, Any] | None = None
RUNTIME_CONFIG: Dict[str, Any] = {}

# Global in-memory telemetry buffer used by smoke tests / debugging
TELEMETRY_EVENTS: List[TelemetryEvent] = []


# ---------------------------------------------------------------------------
# Agent registry
# ---------------------------------------------------------------------------


class AgentRegistry:
    """
    Minimal global registry: agents are registered by their `.role` attribute.
    """

    def __init__(self) -> None:
        self._agents: Dict[str, BaseAgent] = {}

    def register(self, agent: BaseAgent) -> None:
        role = getattr(agent, "role", None)
        if not role:
            raise ValueError(f"Agent {agent} has no 'role' attribute.")
        self._agents[role] = agent

    def get(self, role: str) -> BaseAgent:
        if role not in self._agents:
            raise ValueError(f"No agent registered for role='{role}'")
        return self._agents[role]


# Global singleton registry
REGISTRY = AgentRegistry()


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------


def _load_config(config_path: str | None) -> Dict[str, Any]:
    """
    Load YAML configuration (config.fast.yaml by default) and return a dict.
    Also caches the result in the global CONFIG for later reuse.
    """
    global CONFIG

    if config_path is None:
        config_path = os.getenv("AGENTIC_RAG_CONFIG", "config.fast.yaml")

    path = Path(config_path)
    if not path.is_file():
        # Minimal default; don't crash if config is missing
        CONFIG = {}
        return CONFIG

    try:
        with path.open("r", encoding="utf-8") as f:
            if path.suffix.lower() in {".yaml", ".yml"}:
                cfg = yaml.safe_load(f) or {}
            else:
                import json

                cfg = json.load(f) or {}
    except Exception:
        cfg = {}

    CONFIG = cfg
    return cfg


def _init_runtime_from_config(cfg: Dict[str, Any]) -> None:
    """
    Extracts runtime settings from config.fast.yaml (if present) and stores
    them in a module-level RUNTIME_CONFIG dict used by build_request_context().
    """
    global RUNTIME_CONFIG

    runtime_cfg = cfg.get("runtime") or {}
    if isinstance(runtime_cfg, dict):
        RUNTIME_CONFIG = runtime_cfg
    else:
        RUNTIME_CONFIG = {}


def _build_global_config_from_yaml(cfg: Dict[str, Any] | None) -> GlobalConfig:
    """
    Construct a GlobalConfig instance from config.fast.yaml.

    Uses the `agentic.planner` section if present, otherwise falls back to
    GlobalConfig's own defaults.
    """
    base = GlobalConfig()
    if not cfg:
        return base

    agentic = cfg.get("agentic") or {}
    planner_cfg = agentic.get("planner") or {}

    # Safely overlay onto the base GlobalConfig
    from copy import copy

    gc = copy(base)

    # Booleans
    if "enable_qe" in planner_cfg:
        gc.enable_qe = bool(planner_cfg["enable_qe"])
    if "enable_prf" in planner_cfg:
        gc.enable_prf = bool(planner_cfg["enable_prf"])
    if "enable_rerank" in planner_cfg:
        gc.enable_rerank = bool(planner_cfg["enable_rerank"])

    # Iteration knobs
    if "max_iters" in planner_cfg:
        try:
            gc.max_iters = int(planner_cfg["max_iters"])
        except Exception:
            pass
    if "max_rewrites" in planner_cfg:
        try:
            gc.max_rewrites = int(planner_cfg["max_rewrites"])
        except Exception:
            pass
    if "max_time_seconds" in planner_cfg:
        try:
            gc.max_time_seconds = int(planner_cfg["max_time_seconds"])
        except Exception:
            pass

    # Top-k knobs
    if "top_k" in planner_cfg:
        try:
            gc.top_k = int(planner_cfg["top_k"])
        except Exception:
            pass
    if "rerank_top_k" in planner_cfg:
        try:
            gc.rerank_top_k = int(planner_cfg["rerank_top_k"])
        except Exception:
            pass

    # Language & tools
    if "language" in planner_cfg:
        gc.language = str(planner_cfg["language"])
    if "allow_online_tools" in planner_cfg:
        gc.allow_online_tools = bool(planner_cfg["allow_online_tools"])

    return gc


# ---------------------------------------------------------------------------
# Telemetry helpers
# ---------------------------------------------------------------------------


def _now_iso_utc() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def _log_telemetry_with_elapsed(
    telem_agent: BaseAgent,
    *,
    ctx: RequestContext,
    phase: PhaseEnum,
    agent_name: str,
    event_type: str,
    start_time: float,
    payload: Dict[str, Any] | None = None,
    iteration: int | None = None,
) -> None:
    """
    Compute elapsed_ms from start_time and emit a TelemetryEvent via the
    TelemetryAgent. Also append to the global TELEMETRY_EVENTS buffer.

    TelemetryEvent now includes a 'model' field populated from CONFIG["llm"].
    """
    end_time = time.perf_counter()
    elapsed_ms = (end_time - start_time) * 1000.0

    timing = TelemetryTiming(
        t_iso=_now_iso_utc(),
        elapsed_ms=elapsed_ms,
    )

    iter_value = iteration if iteration is not None else 0

    # Best-effort: pull model name from llm config so Telemetry table can show it.
    model_name: Optional[str] = None
    try:
        cfg = CONFIG or {}
        if cfg:
            llm_cfg = cfg.get("llm") or {}
            if isinstance(llm_cfg, dict):
                model_name = (
                    llm_cfg.get("model")
                    or llm_cfg.get("model_id")
                    or llm_cfg.get("deployment_name")
                )
    except Exception:
        model_name = None

    evt = TelemetryEvent(
        ctx=ctx,
        phase=phase,
        backend=ctx.runtime.backend,
        mode=ctx.runtime.mode,
        agent=agent_name,
        event_type=event_type,
        iteration=iter_value,
        timing=timing,
        payload=payload or {},
        model=model_name,
    )

    TELEMETRY_EVENTS.append(evt)
    telem_agent.log_event(evt)


def clear_telemetry() -> None:
    """
    Convenience helper: clear the global telemetry buffer.
    """
    TELEMETRY_EVENTS.clear()


# ---------------------------------------------------------------------------
# Context + Document + retrieval metrics helpers
# ---------------------------------------------------------------------------


def build_context_snippets_from_results(
    results: List[RetrievalResult],
    limit: int = 10,
) -> List[ContextSnippet]:
    """
    Convert RetrievalResult objects into ContextSnippet objects, respecting a
    global limit across all snippets.

    ContextSnippet schema:

      - doc_id: str
      - chunk_id: str
      - source_text: str
      - translated_text: str
      - lang: str
      - score: float
      - page: Optional[int]
      - doc_title: Optional[str]
      - level: Optional[str/int]
    """
    contexts: List[ContextSnippet] = []
    if not results or limit <= 0:
        return contexts

    for r in results:
        doc_id = str(r.doc_id)
        parent_meta = r.parent_metadata or {}

        # Try to infer a reasonable document title, mirroring build_documents_from_results.
        doc_title = (
            parent_meta.get("title")
            or parent_meta.get("doc_title")
            or parent_meta.get("filename")
            or parent_meta.get("source_path")
            or doc_id
        )
        level_from_parent = parent_meta.get("__level") or parent_meta.get("level")

        if not r.snippets:
            continue

        for sn in r.snippets:
            source_text = sn.text or ""
            if not source_text.strip():
                continue

            page = getattr(sn, "page", None)
            level_from_sn = getattr(sn, "level", None)

            snippet = ContextSnippet(
                doc_id=doc_id,
                chunk_id=str(sn.chunk_id),
                source_text=source_text,
                translated_text=source_text,  # no translation layer yet
                lang=sn.lang or "unknown",
                score=float(sn.score or 0.0),
                page=page,
                doc_title=doc_title,
                level=level_from_sn if level_from_sn is not None else level_from_parent,
            )
            contexts.append(snippet)
            if len(contexts) >= limit:
                return contexts

    return contexts


def build_documents_from_results(
    results: List[RetrievalResult],
    limit: int = 10,
) -> List[Document]:
    """
    Build Haystack Documents from RetrievalResult + Snippet objects to feed
    into LLMGeneratorAgent (RAG-with-context).
    """
    docs: List[Document] = []
    if not results or limit <= 0:
        return docs

    for r in results:
        parent_meta = r.parent_metadata or {}
        doc_title = (
            parent_meta.get("title")
            or parent_meta.get("doc_title")
            or parent_meta.get("filename")
            or parent_meta.get("source_path")
            or str(r.doc_id)
        )
        source_path = parent_meta.get("path") or parent_meta.get("source_path")

        if not r.snippets:
            continue

        for sn in r.snippets:
            text = sn.text or ""
            if not text.strip():
                continue

            doc_id = str(r.doc_id)
            chunk_id = str(sn.chunk_id)
            doc = Document(
                id=f"{doc_id}::chunk::{chunk_id}",
                content=text,
                meta={
                    "doc_id": doc_id,
                    "chunk_id": chunk_id,
                    "score": float(sn.score or 0.0),
                    "lang": sn.lang or "unknown",
                    "page": sn.page,
                    "doc_title": doc_title,
                    "source_path": source_path,
                },
            )
            docs.append(doc)
            if len(docs) >= limit:
                return docs

    return docs


def build_retrieval_metrics(results: List[RetrievalResult]) -> RetrievalMetrics:
    """
    Build a simple RetrievalMetrics summary from RetrievalResult list.
    """
    if not results:
        return RetrievalMetrics(num_docs=0, avg_score=0.0)

    num_docs = len(results)
    scores: List[float] = []
    for r in results:
        if not r.snippets:
            continue
        best = r.snippets[0]
        try:
            scores.append(float(best.score or 0.0))
        except Exception:
            continue

    avg_score = float(sum(scores) / len(scores)) if scores else 0.0
    return RetrievalMetrics(num_docs=num_docs, avg_score=avg_score)


def build_citations_from_context_snippets(
    contexts: List[ContextSnippet],
    per_doc: int = 1,
    limit: int = 10,
) -> List[ContextSnippet]:
    """
    Derive citations as a subset of ContextSnippet objects (first N per doc).

    This keeps the type compatible with smoke_test_agentic.print_sources_table,
    which expects citation items to have attributes like .doc_id, .chunk_id.
    """
    if not contexts or limit <= 0:
        return []

    by_doc: Dict[str, List[ContextSnippet]] = {}
    for sn in contexts:
        by_doc.setdefault(sn.doc_id, []).append(sn)

    citations: List[ContextSnippet] = []
    for doc_id, snippets in by_doc.items():
        # Preserve order: take the first `per_doc` snippets for each document
        for sn in snippets[:per_doc]:
            citations.append(sn)
            if len(citations) >= limit:
                return citations

    return citations


# ---------------------------------------------------------------------------
# RequestContext builder (runtime settings from config.fast.yaml)
# ---------------------------------------------------------------------------


def build_request_context() -> RequestContext:
    """
    Build a RequestContext whose RuntimeContext is driven by the
    `runtime` section of config.fast.yaml, if present.
    """
    # Defaults
    backend = BackendEnum.HF
    mode = RuntimeModeEnum.RAG
    offline = True
    allow_remote_models = True
    allow_online_tools = False

    # Allow config to override
    if RUNTIME_CONFIG:
        b = RUNTIME_CONFIG.get("backend")
        if isinstance(b, str):
            try:
                backend = BackendEnum[b.upper()]
            except KeyError:
                try:
                    backend = BackendEnum(b)
                except Exception:
                    pass

        m = RUNTIME_CONFIG.get("mode")
        if isinstance(m, str):
            try:
                mode = RuntimeModeEnum[m.upper()]
            except KeyError:
                try:
                    mode = RuntimeModeEnum(m)
                except Exception:
                    pass

        if "offline" in RUNTIME_CONFIG:
            offline = bool(RUNTIME_CONFIG["offline"])
        if "allow_remote_models" in RUNTIME_CONFIG:
            allow_remote_models = bool(RUNTIME_CONFIG["allow_remote_models"])
        if "allow_online_tools" in RUNTIME_CONFIG:
            allow_online_tools = bool(RUNTIME_CONFIG["allow_online_tools"])

    runtime = RuntimeContext(
        offline=offline,
        backend=backend,
        mode=mode,
        allow_remote_models=allow_remote_models,
        allow_online_tools=allow_online_tools,
    )

    return RequestContext(
        request_id=uuid.uuid4(),
        session_id=uuid.uuid4(),
        runtime=runtime,
    )


# ---------------------------------------------------------------------------
# Agent registration
# ---------------------------------------------------------------------------


def register_default_agents(config_path: str | None = None) -> AgentRegistry:
    """
    Register all the default/basic agents in the global REGISTRY.
    """
    cfg = _load_config(config_path)
    _init_runtime_from_config(cfg)

    # Core non-LLM agents
    REGISTRY.register(BasicRouterAgent())
    REGISTRY.register(BasicDecompositionAgent())
    REGISTRY.register(BasicPlannerAgent(config_path=config_path))
    REGISTRY.register(BasicGuardrailAgent())
    REGISTRY.register(BasicCriticAgent())
    REGISTRY.register(BasicPolicyAgent())
    REGISTRY.register(BasicTelemetryAgent(events_sink=TELEMETRY_EVENTS))
    REGISTRY.register(BasicPRFAgent(config_path=config_path))
    REGISTRY.register(BasicRerankAgent(config_path=config_path))
    REGISTRY.register(BasicPostProcessorAgent())

    # LLM-backed agents
    REGISTRY.register(LLMQEAgent(config=cfg))
    REGISTRY.register(LLMQueryRewriteAgent(config=cfg))
    REGISTRY.register(LLMGeneratorAgent(config=cfg))

    # Retrieval agent (Chroma + auto-merge)
    retr_agent = HaystackChromaRetrieverAgent(
        config_path=config_path,
        parent_id_field="__parent_id",
        parent_title_field="filename",
        parent_path_field="source_path",
        level_field="__level",
        page_field="page",
        lang_field="lang",
    )
    REGISTRY.register(retr_agent)

    return REGISTRY


# ---------------------------------------------------------------------------
# Main Agentic RAG orchestrator
# ---------------------------------------------------------------------------


def agentic_once_with_metadata(query: str) -> Dict[str, Any]:
    """
    Run one end-to-end Agentic RAG cycle for a single user query and return
    rich metadata (ctx, plan, answer, citations, critic, policy, etc.).
    """
    ctx = build_request_context()
    telem = REGISTRY.get("telemetry")

    # Global planner config derived from config.fast.yaml
    cfg_dict = CONFIG or _load_config(None)
    global_cfg = _build_global_config_from_yaml(cfg_dict)

    original_query = query

    # ------------------------------ Router ------------------------------
    router = REGISTRY.get("router")
    router_cfg = RouterConfig()
    rin = RouterInput(
        ctx=ctx,
        user_query=query,
        history=[],
        config=router_cfg,
    )
    t0 = time.perf_counter()
    r_out = router.route(rin)
    _log_telemetry_with_elapsed(
        telem_agent=telem,
        ctx=ctx,
        phase=PhaseEnum.QUERY,
        agent_name=router.name,
        event_type="router.decided",
        start_time=t0,
        payload={
            "router_profile": r_out.router_profile.model_dump(),
            "user_query": query,
        },
        iteration=None,
    )

    # --------------------------- Decomposition ---------------------------
    decomposer = REGISTRY.get("decomposition")
    decomp_cfg = DecompositionConfig()
    din = DecompositionInput(
        ctx=ctx,
        user_query=query,
        router_profile=r_out.router_profile,
        config=decomp_cfg,
    )
    t1 = time.perf_counter()
    d_out = decomposer.decompose(din)
    _log_telemetry_with_elapsed(
        telem_agent=telem,
        ctx=ctx,
        phase=PhaseEnum.QUERY,
        agent_name=decomposer.name,
        event_type="decomposition.done",
        start_time=t1,
        payload={"decomposition": d_out.decomposition.model_dump()},
        iteration=None,
    )

    # ----------------------------- Planner ------------------------------
    planner = REGISTRY.get("planner")
    pin = PlannerInput(
        ctx=ctx,
        router_profile=r_out.router_profile,
        decomposition=d_out.decomposition,
        global_config=global_cfg,
    )
    t2 = time.perf_counter()
    p_out = planner.plan(pin)
    plan: Plan = p_out.plan

    # Build a small, structured "scaling" view for telemetry
    rp = r_out.router_profile
    plan_iters = getattr(plan, "iterations", None)

    planner_scaling = {
        "complexity_hint": getattr(rp, "complexity_hint", None),
        "query_type": getattr(rp, "query_type", None),
        "iters": {
            "base": global_cfg.max_iters,
            "scaled": getattr(plan_iters, "max_iters", None)
            if isinstance(plan_iters, PlanIterations)
            else None,
        },
        "rewrites": {
            "base": global_cfg.max_rewrites,
            "scaled": getattr(plan_iters, "max_rewrites", None)
            if isinstance(plan_iters, PlanIterations)
            else None,
        },
        "top_k": {
            "base": global_cfg.top_k,
            "scaled": getattr(plan, "top_k", None),
        },
        "rerank_top_k": {
            "base": global_cfg.rerank_top_k,
            "scaled": getattr(plan, "rerank_top_k", None),
        },
    }

    _log_telemetry_with_elapsed(
        telem_agent=telem,
        ctx=ctx,
        phase=PhaseEnum.QUERY,
        agent_name=planner.name,
        event_type="planner.plan",
        start_time=t2,
        payload={
            "router_profile": rp.model_dump(),
            "plan": plan.model_dump(),
            "planner_scaling": planner_scaling,
        },
        iteration=None,
    )

    # ----------------------------- Guardrail ----------------------------
    guard = REGISTRY.get("guardrail")
    gin = GuardrailInput(ctx=ctx, plan=plan)
    t3 = time.perf_counter()
    g_out = guard.validate_plan(gin)
    plan = g_out.plan
    _log_telemetry_with_elapsed(
        telem_agent=telem,
        ctx=ctx,
        phase=PhaseEnum.QUERY,
        agent_name=guard.name,
        event_type="guardrail.validate_plan",
        start_time=t3,
        payload={"plan": plan.model_dump()},
        iteration=None,
    )

    # ------------------------------ PRF ---------------------------------
    prf_agent = REGISTRY.get("prf")
    prf_in = PRFInput(
        ctx=ctx,
        query=query,
        bm25_config=PRFConfig(top_k=plan.top_k),
    )
    t4 = time.perf_counter()
    prf_out = prf_agent.compute(prf_in)
    prf_aug = prf_out.augmented_query
    prf_terms = prf_out.prf_terms
    _log_telemetry_with_elapsed(
        telem_agent=telem,
        ctx=ctx,
        phase=PhaseEnum.QUERY,
        agent_name=prf_agent.name,
        event_type="prf.compute",
        start_time=t4,
        payload={"term_count": len(prf_terms or [])},
        iteration=None,
    )

    # ------------------------------ QE ----------------------------------
    qe_agent = REGISTRY.get("qe")
    expanded_queries: List[str] = []
    if getattr(plan, "use_qe", False):
        qe_in = QEInput(
            ctx=ctx,
            query=prf_aug or query,
            router_profile=r_out.router_profile,
            plan=plan,
        )
        t5 = time.perf_counter()
        qe_out = qe_agent.expand(qe_in)
        expanded_queries = qe_out.expanded_queries or []
        _log_telemetry_with_elapsed(
            telem_agent=telem,
            ctx=ctx,
            phase=PhaseEnum.QUERY,
            agent_name=qe_agent.name,
            event_type="qe.expand",
            start_time=t5,
            payload={"num_expansions": len(expanded_queries)},
            iteration=None,
        )

    # ----------------------------- Retrieval -----------------------------
    retriever = REGISTRY.get("retriever")
    rin2 = RetrieverInput(
        ctx=ctx,
        query=query,
        expanded_queries=expanded_queries,
        prf_augmented_query=prf_aug,
        plan=plan,
    )
    t6 = time.perf_counter()
    re_out = retriever.retrieve(rin2)
    _log_telemetry_with_elapsed(
        telem_agent=telem,
        ctx=ctx,
        phase=PhaseEnum.QUERY,
        agent_name=retriever.name,
        event_type="retriever.results",
        start_time=t6,
        payload={"num_results": len(re_out.results or [])},
        iteration=None,
    )

    # ----------------------------- Rerank -------------------------------
    rerank_agent = REGISTRY.get("rerank")
    rrin = RerankInput(
        ctx=ctx,
        query=query,
        results=re_out.results,
        plan=plan,
    )
    t7 = time.perf_counter()
    rr_out = rerank_agent.rerank(rrin)
    _log_telemetry_with_elapsed(
        telem_agent=telem,
        ctx=ctx,
        phase=PhaseEnum.QUERY,
        agent_name=rerank_agent.name,
        event_type="rerank.rerank",
        start_time=t7,
        payload={"num_results": len(rr_out.results or [])},
        iteration=None,
    )

    # Build initial context snippets and documents for generator
    contexts: List[ContextSnippet] = build_context_snippets_from_results(
        rr_out.results, limit=plan.top_k
    )
    docs: List[Document] = build_documents_from_results(
        rr_out.results, limit=plan.top_k
    )
    retrieval_metrics: RetrievalMetrics = build_retrieval_metrics(rr_out.results)

    # -------------------- Iteration configuration -----------------------
    # Default to single pass
    max_iters = 1
    min_iters = 1

    plan_iters = getattr(plan, "iterations", None)
    if isinstance(plan_iters, PlanIterations):
        max_from_plan = getattr(plan_iters, "max_iters", None)
        min_from_plan = getattr(plan_iters, "max_rewrites", None)

        if max_from_plan is not None:
            try:
                max_iters = max(1, int(max_from_plan))
            except (TypeError, ValueError):
                pass

        if min_from_plan is not None:
            try:
                min_iters = max(1, int(min_from_plan))
            except (TypeError, ValueError):
                pass

    # Environment overrides
    env_max = os.getenv("AGENTIC_MAX_ITERS")
    env_min = os.getenv("AGENTIC_MIN_ITERS")
    if env_max is not None:
        try:
            max_iters = max(1, int(env_max))
        except ValueError:
            pass
    if env_min is not None:
        try:
            min_iters = max(1, int(env_min))
        except ValueError:
            pass

    if max_iters < min_iters:
        max_iters = min_iters

    # Core agents for the iterative loop
    generator = REGISTRY.get("generator")
    critic = REGISTRY.get("critic")
    policy = REGISTRY.get("policy")

    last_gen = None
    last_cr: Optional[CriticFeedback] = None
    last_pol = None
    iterations_data: List[Dict[str, Any]] = []

    # Keep track of PRF/QE state so we can refresh on rewrite
    prf_augmented = prf_aug
    prf_term_list = prf_terms
    expanded_queries_list = expanded_queries

    # -------------------- Main agentic iteration loop -------------------
    for i in range(max_iters):
        # ---------------------- Generator ----------------------
        # Build a RAG-aware query string that inlines the top-N context snippets.
        if contexts:
            context_text = "\n\n".join(
                f"[{idx + 1}] {sn.source_text}"
                for idx, sn in enumerate(contexts)
                if (sn.source_text or "").strip()
            ).strip()
        else:
            context_text = ""

        if context_text:
            rag_query = (
                "You are a retrieval-augmented assistant. "
                "Use ONLY the context below to answer the user’s question. "
                "If the context is insufficient, say so explicitly.\n\n"
                "Context:\n"
                f"{context_text}\n\n"
                f"Question:\n{query}"
            )
        else:
            # Fallback: no context, just use the raw user query
            rag_query = query

        gin2 = GeneratorInput(
            ctx=ctx,
            query=rag_query,
            plan=plan,
            context_snippets=contexts,
        )

        t8 = time.perf_counter()
        gen_out = generator.generate(gin2)
        last_gen = gen_out
        _log_telemetry_with_elapsed(
            telem_agent=telem,
            ctx=ctx,
            phase=PhaseEnum.QUERY,
            agent_name=generator.name,
            event_type="generator.output",
            start_time=t8,
            payload={
                "iteration": i,
                "answer_len": len(gen_out.answer.text or "")
                if gen_out.answer and gen_out.answer.text
                else 0,
            },
            iteration=i,
        )

        # ------------------------ Critic ------------------------
        cin = CriticInput(
            ctx=ctx,
            query=query,
            plan=plan,
            answer=gen_out.answer,
            citations=gen_out.citations,
            context_snippets=contexts,
        )
        t9 = time.perf_counter()
        cr_out = critic.evaluate(cin)
        last_cr = cr_out
        _log_telemetry_with_elapsed(
            telem_agent=telem,
            ctx=ctx,
            phase=PhaseEnum.QUERY,
            agent_name=critic.name,
            event_type="critic.evaluate",
            start_time=t9,
            payload={"iteration": i},
            iteration=i,
        )

        # ------------------------ Policy ------------------------
        pol_in = PolicyInput(
            ctx=ctx,
            iteration=i,
            plan=plan,
            retrieval_metrics=retrieval_metrics,
            critic_feedback=cr_out,
        )
        t10 = time.perf_counter()
        pol_out = policy.decide(pol_in)
        last_pol = pol_out
        _log_telemetry_with_elapsed(
            telem_agent=telem,
            ctx=ctx,
            phase=PhaseEnum.QUERY,
            agent_name=policy.name,
            event_type="policy.decision",
            start_time=t10,
            payload={"decision": pol_out.model_dump()},
            iteration=i,
        )

        iterations_data.append(
            {
                "iteration": i,
                "answer": gen_out.answer,
                "critic": cr_out,
                "policy": pol_out,
            }
        )

        decision = getattr(pol_out, "decision", None)
        decision_name = getattr(decision, "name", "").upper() if decision is not None else ""

        next_iter_possible = (i + 1) < max_iters
        below_min_iters = (i + 1) < min_iters

        # Does policy want some form of revision / retry / continue?
        wants_revision = "REWRITE" in decision_name or "REVISION" in decision_name or "RETRY" in decision_name

        # If we can iterate again AND the policy wants revision, rewrite the
        # query using critic feedback and re-run PRF/QE/retrieval/rerank.
        if next_iter_possible and wants_revision:
            try:
                rewrite_agent = REGISTRY.get("rewrite")
            except ValueError:
                rewrite_agent = None

            if rewrite_agent is not None and last_cr is not None:
                rwin = QueryRewriteInput(
                    ctx=ctx,
                    original_query=original_query,
                    current_query=query,
                    critic_feedback=last_cr,
                    plan=plan,
                    translation_metadata=None,
                )
                t_rw = time.perf_counter()
                rw_out = rewrite_agent.rewrite(rwin)
                _log_telemetry_with_elapsed(
                    telem_agent=telem,
                    ctx=ctx,
                    phase=PhaseEnum.QUERY,
                    agent_name=rewrite_agent.name,
                    event_type="rewrite.rewrite",
                    start_time=t_rw,
                    payload={"iteration": i},
                    iteration=i,
                )

                new_query = (rw_out.rewritten_query or "").strip() or query
                if new_query != query:
                    query = new_query

                    # --------- Re-run PRF with rewritten query ---------
                    t_prf2 = time.perf_counter()
                    prf_in2 = PRFInput(
                        ctx=ctx,
                        query=query,
                        bm25_config=PRFConfig(top_k=plan.top_k),
                    )
                    prf_out2 = prf_agent.compute(prf_in2)
                    prf_augmented = prf_out2.augmented_query
                    prf_term_list = prf_out2.prf_terms
                    _log_telemetry_with_elapsed(
                        telem_agent=telem,
                        ctx=ctx,
                        phase=PhaseEnum.QUERY,
                        agent_name=prf_agent.name,
                        event_type="prf.compute_rewrite",
                        start_time=t_prf2,
                        payload={"term_count": len(prf_term_list or [])},
                        iteration=i,
                    )

                    # --------- Re-run QE if plan.use_qe ---------
                    expanded_queries_list = []
                    if getattr(plan, "use_qe", False):
                        t_qe2 = time.perf_counter()
                        qe_in2 = QEInput(
                            ctx=ctx,
                            query=prf_augmented or query,
                            router_profile=r_out.router_profile,
                            plan=plan,
                        )
                        qe_out2 = qe_agent.expand(qe_in2)
                        expanded_queries_list = qe_out2.expanded_queries or []
                        _log_telemetry_with_elapsed(
                            telem_agent=telem,
                            ctx=ctx,
                            phase=PhaseEnum.QUERY,
                            agent_name=qe_agent.name,
                            event_type="qe.expand_rewrite",
                            start_time=t_qe2,
                            payload={"num_expansions": len(expanded_queries_list)},
                            iteration=i,
                        )

                    # --------- Re-run retrieval + rerank ---------
                    rin3 = RetrieverInput(
                        ctx=ctx,
                        query=query,
                        expanded_queries=expanded_queries_list,
                        prf_augmented_query=prf_augmented,
                        plan=plan,
                    )
                    t_ret2 = time.perf_counter()
                    re_out2 = retriever.retrieve(rin3)
                    _log_telemetry_with_elapsed(
                        telem_agent=telem,
                        ctx=ctx,
                        phase=PhaseEnum.QUERY,
                        agent_name=retriever.name,
                        event_type="retriever.results_rewrite",
                        start_time=t_ret2,
                        payload={"num_results": len(re_out2.results or [])},
                        iteration=i,
                    )

                    rrin2 = RerankInput(
                        ctx=ctx,
                        query=query,
                        results=re_out2.results,
                        plan=plan,
                    )
                    t_rr2 = time.perf_counter()
                    rr_out2 = rerank_agent.rerank(rrin2)
                    _log_telemetry_with_elapsed(
                        telem_agent=telem,
                        ctx=ctx,
                        phase=PhaseEnum.QUERY,
                        agent_name=rerank_agent.name,
                        event_type="rerank.rerank_rewrite",
                        start_time=t_rr2,
                        payload={"num_results": len(rr_out2.results or [])},
                        iteration=i,
                    )

                    # Refresh contexts/docs/metrics for next iteration
                    contexts = build_context_snippets_from_results(
                        rr_out2.results, limit=plan.top_k
                    )
                    docs = build_documents_from_results(
                        rr_out2.results, limit=plan.top_k
                    )
                    retrieval_metrics = build_retrieval_metrics(rr_out2.results)

            # After rewrite + refresh, go to next iteration
            if next_iter_possible:
                continue

        # Enforce minimum iterations: if below min_iters and we *can* iterate,
        # just keep going regardless of decision.
        if below_min_iters and next_iter_possible:
            continue

        # Stop conditions
        if (not next_iter_possible) or ("FINAL" in decision_name) or ("STOP" in decision_name):
            break

    # -------------------------- Post-processing --------------------------
    post_agent = REGISTRY.get("postprocess")

    # Derive citations as ContextSnippet objects:
    # - prefer generator-produced citations (if present and compatible)
    # - otherwise, pick top snippets per doc from `contexts`
    if last_gen is None:
        final_answer_obj = Answer(
            text="",
            sections=[AnswerSection(title="Final Answer", body="")],
        )
        final_critic = CriticFeedback()
        citations: List[ContextSnippet] = build_citations_from_context_snippets(
            contexts, per_doc=1, limit=10
        )
    else:
        final_answer_obj = last_gen.answer
        final_critic = last_cr
        gen_citations = getattr(last_gen, "citations", None) or []
        if gen_citations:
            citations = gen_citations  # assumes same schema
        else:
            citations = build_citations_from_context_snippets(
                contexts, per_doc=1, limit=10
            )

    # Default preferences; can be wired from config later
    pp_prefs = PostprocessPreferences()

    ppin = PostprocessInput(
        ctx=ctx,
        query=query,
        plan=plan,
        answer=final_answer_obj,
        critic_feedback=final_critic,
        router_profile=r_out.router_profile,
        context_snippets=contexts,
        iterations=len(iterations_data),
        preferences=pp_prefs,
    )
    t11 = time.perf_counter()
    pp_out = post_agent.format(ppin)
    _log_telemetry_with_elapsed(
        telem_agent=telem,
        ctx=ctx,
        phase=PhaseEnum.QUERY,
        agent_name=post_agent.name,
        event_type="postprocess.format",
        start_time=t11,
        payload={},
        iteration=None,
    )

    # --------------------------- Aggregate meta --------------------------
    return {
        "ctx": ctx,
        "router": r_out,
        "decomposition": d_out,
        "plan": plan,
        "guardrail": g_out,
        "prf_terms": prf_term_list,
        "qe_expansions": expanded_queries_list,
        "retrieval_results": rr_out.results,
        "answer": final_answer_obj,
        "citations": citations,
        "context_snippets": contexts,
        "critic": final_critic,
        "policy": last_pol,
        "postprocess": pp_out,
        "iterations": iterations_data,
        "retrieval_metrics": retrieval_metrics,
    }


def agentic_once(query: str) -> str:
    """
    Convenience wrapper: return only the final answer text.
    """
    meta = agentic_once_with_metadata(query)
    ans: Answer = meta["answer"]
    return ans.text or ""


# ---------------------------------------------------------------------------
# CLI entry-point (manual smoke test)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    register_default_agents()
    user_query = "What is hierarchical RAG and why is it useful?"
    print(agentic_once(user_query))
