from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Common enums
# ---------------------------------------------------------------------------

class BackendEnum(str, Enum):
    HF = "hf"
    VLLM = "vllm"
    OLLAMA = "ollama"
    OPENAI_COMPAT = "openai_compat"


class RuntimeModeEnum(str, Enum):
    RETRIEVAL = "RETRIEVAL"
    RAG = "RAG"
    AGENTIC = "AGENTIC"


class PhaseEnum(str, Enum):
    WARMUP = "warmup"
    QUERY = "query"
    ITERATION = "iteration"
    FINAL = "final"
    MAINTENANCE = "maintenance"


class RetrievalModeEnum(str, Enum):
    LEAF_ONLY = "leaf_only"
    DUAL_INDEX = "dual_index"


class DecisionEnum(str, Enum):
    FINALIZE = "finalize"
    REWRITE = "rewrite"
    CONTINUE = "continue"


class SafetyStageEnum(str, Enum):
    INPUT = "input"
    TOOL_REQUEST = "tool_request"
    TOOL_RESPONSE = "tool_response"
    ANSWER = "answer"


# ---------------------------------------------------------------------------
# Common envelope / runtime models
# ---------------------------------------------------------------------------

class RuntimeContext(BaseModel):
    offline: bool = True
    backend: BackendEnum = BackendEnum.HF
    mode: RuntimeModeEnum = RuntimeModeEnum.AGENTIC
    allow_remote_models: bool = False
    allow_online_tools: bool = False


class RequestContext(BaseModel):
    request_id: UUID
    session_id: UUID
    runtime: RuntimeContext


class TelemetryTiming(BaseModel):
    t_iso: str
    elapsed_ms: float


# ---------------------------------------------------------------------------
# Router / Decomposition / Planner / Guardrail
# ---------------------------------------------------------------------------

class RouterConfig(BaseModel):
    default_query_type: Optional[str] = None
    max_hist_turns: int = 10


class Message(BaseModel):
    role: str
    content: str


class RouterProfile(BaseModel):
    query_type: str  # "lookup" | "explanation" | "comparison" | "list" | "other"
    use_qe: bool
    use_prf: bool
    use_rerank: bool
    expected_answer_style: str  # "short" | "paragraph" | "multi_section"
    complexity_hint: str        # "low" | "medium" | "high"


class RouterInput(BaseModel):
    ctx: RequestContext
    user_query: str
    history: List[Message] = Field(default_factory=list)
    config: RouterConfig


class RouterOutput(BaseModel):
    router_profile: RouterProfile


class DecompositionConfig(BaseModel):
    max_subqueries: int = 4
    min_subquery_length: int = 10


class Subquery(BaseModel):
    id: str
    text: str


class ComparisonPair(BaseModel):
    left: str
    right: str


class Decomposition(BaseModel):
    is_multi_part: bool
    subqueries: List[Subquery] = Field(default_factory=list)
    comparison_pairs: List[ComparisonPair] = Field(default_factory=list)


class DecompositionInput(BaseModel):
    ctx: RequestContext
    user_query: str
    router_profile: RouterProfile
    config: DecompositionConfig


class DecompositionOutput(BaseModel):
    decomposition: Decomposition


class GlobalConfig(BaseModel):
    default_retrieval_mode: RetrievalModeEnum = RetrievalModeEnum.DUAL_INDEX
    enable_qe: bool = True
    enable_prf: bool = True
    enable_rerank: bool = True
    max_iters: int = 3
    max_rewrites: int = 2
    max_time_seconds: int = 30
    top_k: int = 10
    rerank_top_k: int = 20
    language: str = "auto"
    allow_online_tools: bool = False


class PlanIterations(BaseModel):
    max_iters: int
    max_rewrites: int


class Plan(BaseModel):
    retrieval_mode: RetrievalModeEnum
    use_qe: bool
    use_prf: bool
    use_rerank: bool
    iterations: PlanIterations
    top_k: int
    rerank_top_k: int
    language: str
    allow_online_tools: bool
    backend: BackendEnum


class PlannerInput(BaseModel):
    ctx: RequestContext
    router_profile: RouterProfile
    decomposition: Decomposition
    global_config: GlobalConfig


class PlannerOutput(BaseModel):
    plan: Plan


class GuardrailInput(BaseModel):
    ctx: RequestContext
    plan: Plan


class GuardrailOutput(BaseModel):
    status: str  # "ok" | "adjusted" | "blocked"
    plan: Plan
    messages: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Translation
# ---------------------------------------------------------------------------

class TranslationItem(BaseModel):
    id: str
    text: str
    type: str  # "query" | "snippet"


class TranslationConfig(BaseModel):
    enabled: bool = True
    detect_only: bool = False
    target_lang: str = "en"
    min_confidence: float = 0.7


class TranslationResultItem(BaseModel):
    id: str
    type: str
    lang: str
    confidence: float
    source_text: str
    translated_text: str


class TranslationMetadata(BaseModel):
    original_lang: Optional[str] = None
    target_lang: str = "en"
    query_id: Optional[str] = None
    snippet_ids: List[str] = Field(default_factory=list)


class TranslationInput(BaseModel):
    ctx: RequestContext
    items: List[TranslationItem]
    config: TranslationConfig


class TranslationOutput(BaseModel):
    normalized_query: Optional[str]
    items: List[TranslationResultItem]
    translation_metadata: TranslationMetadata


# ---------------------------------------------------------------------------
# QE / PRF
# ---------------------------------------------------------------------------

class QEInput(BaseModel):
    ctx: RequestContext
    query: str
    router_profile: RouterProfile
    plan: Plan
    translation_metadata: Optional[TranslationMetadata] = None


class QEOutput(BaseModel):
    expanded_queries: List[str]


class PRFConfig(BaseModel):
    top_k: int = 10


class PRFInput(BaseModel):
    ctx: RequestContext
    query: str
    bm25_config: PRFConfig


class PRFOutput(BaseModel):
    prf_terms: List[str]
    augmented_query: str


# ---------------------------------------------------------------------------
# Retrieval / Rerank
# ---------------------------------------------------------------------------

class Snippet(BaseModel):
    chunk_id: str
    score: float
    text: str
    lang: Optional[str] = None
    page: Optional[int] = None
    level: str = "leaf"


class RetrievalResult(BaseModel):
    doc_id: str
    parent_metadata: Dict[str, Any] = Field(default_factory=dict)
    snippets: List[Snippet] = Field(default_factory=list)


class RetrieverInput(BaseModel):
    ctx: RequestContext
    query: str
    expanded_queries: Optional[List[str]] = None
    prf_augmented_query: Optional[str] = None
    plan: Plan


class RetrieverOutput(BaseModel):
    results: List[RetrievalResult]


class RerankInput(BaseModel):
    ctx: RequestContext
    query: str
    results: List[RetrievalResult]
    plan: Plan


class RerankOutput(BaseModel):
    results: List[RetrievalResult]


# ---------------------------------------------------------------------------
# Policy / Rewrite
# ---------------------------------------------------------------------------

class RetrievalMetrics(BaseModel):
    num_docs: int
    avg_score: float


class CriticFeedback(BaseModel):
    hallucination_risk: float
    coverage_score: float
    missing_topics: List[str] = Field(default_factory=list)
    ambiguities: List[str] = Field(default_factory=list)
    unsupported_claims: List[Dict[str, Any]] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)


class PolicyInput(BaseModel):
    ctx: RequestContext
    iteration: int
    plan: Plan
    retrieval_metrics: RetrievalMetrics
    critic_feedback: CriticFeedback


class PolicyOutput(BaseModel):
    decision: DecisionEnum
    reason: str
    adjustments: Dict[str, Any] = Field(default_factory=dict)


class QueryRewriteInput(BaseModel):
    ctx: RequestContext
    original_query: str
    current_query: str
    critic_feedback: CriticFeedback
    plan: Plan
    translation_metadata: Optional[TranslationMetadata] = None


class QueryRewriteOutput(BaseModel):
    rewritten_query: str
    notes: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Generation / Critic / Post-process
# ---------------------------------------------------------------------------

class ContextSnippet(BaseModel):
    doc_id: str
    chunk_id: str
    source_text: str
    translated_text: str
    lang: str
    page: Optional[int] = None
    score: float
    level: Optional[str] = None
    doc_title: Optional[str] = None


class AnswerSection(BaseModel):
    title: str
    body: str


class Answer(BaseModel):
    text: str
    sections: Optional[List[AnswerSection]] = None


class Citation(BaseModel):
    doc_id: str
    chunk_id: str
    page: Optional[int] = None
    score: float
    lang: str
    snippet: str
    original_snippet: str


class GeneratorInput(BaseModel):
    ctx: RequestContext
    query: str
    plan: Plan
    context_snippets: List[ContextSnippet]


class GeneratorOutput(BaseModel):
    answer: Answer
    citations: List[Citation]


class CriticInput(BaseModel):
    ctx: RequestContext
    query: str
    answer: Answer
    context_snippets: List[ContextSnippet]
    plan: Plan


class CriticOutput(CriticFeedback):
    pass


class PostprocessPreferences(BaseModel):
    format: str = "markdown"  # or "plain"
    include_critic_note: bool = True
    include_language_notes: bool = True


class PostprocessInput(BaseModel):
    ctx: RequestContext
    query: str
    answer: Answer
    critic_feedback: CriticFeedback
    context_snippets: List[ContextSnippet]
    preferences: PostprocessPreferences


class PostprocessMetadata(BaseModel):
    critic_summary: str
    languages: List[str]


class PostprocessOutput(BaseModel):
    final_text: str
    metadata: PostprocessMetadata


# ---------------------------------------------------------------------------
# Telemetry
# ---------------------------------------------------------------------------

class TelemetryEvent(BaseModel):
    ctx: RequestContext
    phase: PhaseEnum
    iteration: int
    event_type: str
    agent: str
    mode: RuntimeModeEnum
    backend: BackendEnum
    model: Optional[str] = None
    timing: TelemetryTiming
    payload: Dict[str, Any] = Field(default_factory=dict)


class TelemetryOutput(BaseModel):
    status: str = "logged"
    trace_id: str
    sink: str


# ---------------------------------------------------------------------------
# Tools / Safety / Index
# ---------------------------------------------------------------------------

class ToolLimits(BaseModel):
    timeout_seconds: int = 20
    max_bytes: int = 2_000_000
    max_items: int = 50


class ToolExecutionInput(BaseModel):
    ctx: RequestContext
    tool_id: str
    arguments: Dict[str, Any]
    limits: ToolLimits


class ToolExecutionOutput(BaseModel):
    status: str  # "ok" | "blocked" | "error"
    result: Any
    error: Optional[str] = None
    tool_metadata: Dict[str, Any] = Field(default_factory=dict)


class SafetyConfig(BaseModel):
    enabled: bool = True
    mode: str = "log_only"  # "log_only" | "enforce"
    policies: Dict[str, Any] = Field(default_factory=dict)


class SafetyInput(BaseModel):
    ctx: RequestContext
    stage: SafetyStageEnum
    text: str
    metadata: Dict[str, Any]
    config: SafetyConfig


class SafetyOutput(BaseModel):
    allowed: bool
    redacted_text: Optional[str]
    action: str           # "allow" | "redact" | "block" | "warn"
    reasons: List[str]
    tags: List[str]


class IndexOperationEnum(str, Enum):
    STATUS = "status"
    REINDEX = "reindex"
    ADD_DOCS = "add_docs"
    REMOVE_DOCS = "remove_docs"


class IndexInput(BaseModel):
    ctx: RequestContext
    operation: IndexOperationEnum
    arguments: Dict[str, Any]


class IndexStatusInfo(BaseModel):
    num_docs: int
    num_chunks: int
    languages: List[str]
    last_updated: Optional[str]


class IndexOutput(BaseModel):
    status: str  # "ok" | "error"
    leaf_index: Optional[IndexStatusInfo] = None
    parent_index: Optional[IndexStatusInfo] = None
    num_processed: Optional[int] = None
    errors: List[str] = Field(default_factory=list)
