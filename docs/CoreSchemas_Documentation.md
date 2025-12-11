# Core Schemas Documentation

Technical reference for Radiant RAG pipeline data models.

---

## Overview

The `core.schemas` module defines all Pydantic data models used throughout the Radiant RAG pipeline. These schemas provide type safety, validation, and serialization for data flowing between agents.

**Module Location:** `core/schemas.py`

---

## Enumerations

### BackendEnum

LLM backend types.

| Value | Description |
|-------|-------------|
| `hf` | HuggingFace local |
| `vllm` | vLLM server |
| `ollama` | Ollama local |
| `openai_compat` | OpenAI-compatible API |

### RuntimeModeEnum

Pipeline execution modes.

| Value | Description |
|-------|-------------|
| `RETRIEVAL` | Retrieval only (no generation) |
| `RAG` | Standard RAG |
| `AGENTIC` | Full agentic with iterations |

### PhaseEnum

Pipeline execution phases.

| Value | Description |
|-------|-------------|
| `warmup` | Initialization |
| `query` | Query processing |
| `iteration` | Loop iteration |
| `final` | Finalization |
| `maintenance` | Index operations |

### RetrievalModeEnum

Retrieval index modes.

| Value | Description |
|-------|-------------|
| `leaf_only` | Only leaf chunks |
| `dual_index` | Leaf + parent chunks |

### DecisionEnum

Policy decisions.

| Value | Description |
|-------|-------------|
| `finalize` | Accept answer |
| `rewrite` | Rewrite query |
| `continue` | Retry same query |

### SafetyStageEnum

Safety check stages.

| Value | Description |
|-------|-------------|
| `input` | User input |
| `tool_request` | Before tool call |
| `tool_response` | After tool call |
| `answer` | Final answer |

### IndexOperationEnum

Index management operations.

| Value | Description |
|-------|-------------|
| `status` | Get index status |
| `reindex` | Full reindex |
| `add_docs` | Add documents |
| `remove_docs` | Remove documents |

---

## Common Models

### RuntimeContext

Runtime environment configuration.

```python
class RuntimeContext(BaseModel):
    offline: bool = True
    backend: BackendEnum = BackendEnum.HF
    mode: RuntimeModeEnum = RuntimeModeEnum.AGENTIC
    allow_remote_models: bool = False
    allow_online_tools: bool = False
```

### RequestContext

Per-request context with identifiers.

```python
class RequestContext(BaseModel):
    request_id: UUID
    session_id: UUID
    runtime: RuntimeContext
```

### TelemetryTiming

Timing information for telemetry.

```python
class TelemetryTiming(BaseModel):
    t_iso: str
    elapsed_ms: float
```

---

## Query Processing Schemas

### Router Schemas

```python
class RouterConfig(BaseModel):
    default_query_type: Optional[str] = None
    max_hist_turns: int = 10

class Message(BaseModel):
    role: str
    content: str

class RouterProfile(BaseModel):
    query_type: str           # "lookup" | "explanation" | "comparison" | "list" | "other"
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
```

### Decomposition Schemas

```python
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
```

### Planner Schemas

```python
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
```

---

## Retrieval Schemas

### QE/PRF Schemas

```python
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
```

### Retriever Schemas

```python
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
```

---

## Generation Schemas

### Context and Answer Schemas

```python
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
```

### Generator Schemas

```python
class GeneratorInput(BaseModel):
    ctx: RequestContext
    query: str
    plan: Plan
    context_snippets: List[ContextSnippet]

class GeneratorOutput(BaseModel):
    answer: Answer
    citations: List[Citation]
```

### Critic Schemas

```python
class CriticFeedback(BaseModel):
    hallucination_risk: float
    coverage_score: float
    missing_topics: List[str] = Field(default_factory=list)
    ambiguities: List[str] = Field(default_factory=list)
    unsupported_claims: List[Dict[str, Any]] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)

class CriticInput(BaseModel):
    ctx: RequestContext
    query: str
    answer: Answer
    context_snippets: List[ContextSnippet]
    plan: Plan

class CriticOutput(CriticFeedback):
    pass
```

### Policy Schemas

```python
class RetrievalMetrics(BaseModel):
    num_docs: int
    avg_score: float

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
```

### PostProcess Schemas

```python
class PostprocessPreferences(BaseModel):
    format: str = "markdown"
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
```

---

## Infrastructure Schemas

### Telemetry Schemas

```python
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
```

### Tool Execution Schemas

```python
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
```

### Safety Schemas

```python
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
    action: str  # "allow" | "redact" | "block" | "warn"
    reasons: List[str]
    tags: List[str]
```

### Index Management Schemas

```python
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
```

---

## Usage Examples

### Creating Schemas

```python
from uuid import uuid4
from core.schemas import (
    RequestContext, RuntimeContext, BackendEnum, RuntimeModeEnum,
    RouterInput, RouterConfig, Message,
)

runtime = RuntimeContext(
    offline=False,
    backend=BackendEnum.OPENAI_COMPAT,
    mode=RuntimeModeEnum.AGENTIC,
)

ctx = RequestContext(
    request_id=uuid4(),
    session_id=uuid4(),
    runtime=runtime,
)

router_input = RouterInput(
    ctx=ctx,
    user_query="What is RAG?",
    history=[Message(role="user", content="Hello")],
    config=RouterConfig(max_hist_turns=5),
)
```

### Serialization

```python
from core.schemas import RouterProfile

profile = RouterProfile(
    query_type="explanation",
    use_qe=True,
    use_prf=True,
    use_rerank=True,
    expected_answer_style="paragraph",
    complexity_hint="medium",
)

# To dictionary
data = profile.model_dump()

# To JSON
json_str = profile.model_dump_json()

# From dictionary
profile2 = RouterProfile.model_validate(data)
```

---

## Related Documentation

- [CoreInterfaces_Documentation.md](CoreInterfaces_Documentation.md) - Agent interface definitions
- [Orchestrator_Documentation.md](Orchestrator_Documentation.md) - Pipeline execution
