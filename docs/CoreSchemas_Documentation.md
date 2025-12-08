# Core Schemas Documentation

## Technical Reference for the Radiant RAG Pipeline Data Models

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture Design](#architecture-design)
3. [Enumerations](#enumerations)
4. [Common Models](#common-models)
5. [Query Processing Schemas](#query-processing-schemas)
6. [Retrieval Schemas](#retrieval-schemas)
7. [Generation Schemas](#generation-schemas)
8. [Infrastructure Schemas](#infrastructure-schemas)
9. [Schema Relationships](#schema-relationships)
10. [Testing Strategies](#testing-strategies)
11. [Recommendations and Improvements](#recommendations-and-improvements)
12. [Usage Examples](#usage-examples)

---

## Overview

### Purpose

The `core.schemas` module defines all Pydantic data models used throughout the Radiant RAG pipeline. These schemas provide type safety, validation, serialization, and documentation for data flowing between agents.

### Key Responsibilities

- Define strongly-typed input/output contracts for all agents
- Provide automatic validation of data structures
- Enable JSON serialization/deserialization
- Document field purposes and constraints
- Support IDE autocompletion and type checking

### Design Philosophy

The module follows **Data Transfer Object (DTO)** patterns where each agent has dedicated input and output schemas. This creates clear contracts between components and enables independent evolution of the data structures.

---

## Architecture Design

### Schema Organization

```
core/schemas.py
├── Enumerations (7 enums)
│   ├── BackendEnum
│   ├── RuntimeModeEnum
│   ├── PhaseEnum
│   ├── RetrievalModeEnum
│   ├── DecisionEnum
│   ├── SafetyStageEnum
│   └── IndexOperationEnum
│
├── Common Models
│   ├── RuntimeContext
│   ├── RequestContext
│   └── TelemetryTiming
│
├── Query Processing
│   ├── Router (Config, Profile, Input, Output)
│   ├── Decomposition (Config, Input, Output, Subquery, etc.)
│   ├── Planner (GlobalConfig, Plan, Input, Output)
│   └── Guardrail (Input, Output)
│
├── Translation
│   ├── TranslationItem, Config, ResultItem
│   ├── TranslationMetadata
│   └── TranslationInput, Output
│
├── Retrieval
│   ├── QE (Input, Output)
│   ├── PRF (Config, Input, Output)
│   ├── Retriever (Snippet, Result, Input, Output)
│   └── Rerank (Input, Output)
│
├── Generation
│   ├── Context (ContextSnippet)
│   ├── Answer (Answer, AnswerSection, Citation)
│   ├── Generator (Input, Output)
│   ├── Critic (Input, Output, Feedback)
│   ├── Policy (Input, Output, Metrics)
│   ├── QueryRewrite (Input, Output)
│   └── Postprocess (Input, Output, Preferences, Metadata)
│
├── Telemetry
│   ├── TelemetryEvent
│   └── TelemetryOutput
│
└── Infrastructure
    ├── Tools (Limits, Input, Output)
    ├── Safety (Config, Input, Output)
    └── Index (Input, Output, StatusInfo)
```

### Schema Count by Category

| Category | Count |
|----------|-------|
| Enumerations | 7 |
| Common/Context | 3 |
| Query Processing | 15 |
| Translation | 6 |
| Retrieval | 10 |
| Generation | 17 |
| Telemetry | 2 |
| Infrastructure | 10 |
| **Total** | **70** |

---

## Enumerations

### BackendEnum

LLM backend types.

```python
class BackendEnum(str, Enum):
    HF = "hf"                    # HuggingFace local
    VLLM = "vllm"                # vLLM server
    OLLAMA = "ollama"            # Ollama local
    OPENAI_COMPAT = "openai_compat"  # OpenAI-compatible API
```

### RuntimeModeEnum

Pipeline execution modes.

```python
class RuntimeModeEnum(str, Enum):
    RETRIEVAL = "RETRIEVAL"  # Retrieval only (no generation)
    RAG = "RAG"              # Standard RAG
    AGENTIC = "AGENTIC"      # Full agentic with iterations
```

### PhaseEnum

Pipeline execution phases.

```python
class PhaseEnum(str, Enum):
    WARMUP = "warmup"           # Initialization
    QUERY = "query"             # Query processing
    ITERATION = "iteration"     # Loop iteration
    FINAL = "final"             # Finalization
    MAINTENANCE = "maintenance" # Index operations
```

### RetrievalModeEnum

Retrieval index modes.

```python
class RetrievalModeEnum(str, Enum):
    LEAF_ONLY = "leaf_only"    # Only leaf chunks
    DUAL_INDEX = "dual_index"  # Leaf + parent chunks
```

### DecisionEnum

Policy decisions.

```python
class DecisionEnum(str, Enum):
    FINALIZE = "finalize"  # Accept answer
    REWRITE = "rewrite"    # Rewrite query
    CONTINUE = "continue"  # Retry same query
```

### SafetyStageEnum

Safety check stages.

```python
class SafetyStageEnum(str, Enum):
    INPUT = "input"               # User input
    TOOL_REQUEST = "tool_request" # Before tool call
    TOOL_RESPONSE = "tool_response"  # After tool call
    ANSWER = "answer"             # Final answer
```

### IndexOperationEnum

Index management operations.

```python
class IndexOperationEnum(str, Enum):
    STATUS = "status"           # Get index status
    REINDEX = "reindex"         # Full reindex
    ADD_DOCS = "add_docs"       # Add documents
    REMOVE_DOCS = "remove_docs" # Remove documents
```

---

## Common Models

### RuntimeContext

Runtime environment configuration.

```python
class RuntimeContext(BaseModel):
    offline: bool = True                    # Offline mode
    backend: BackendEnum = BackendEnum.HF   # LLM backend
    mode: RuntimeModeEnum = RuntimeModeEnum.AGENTIC  # Execution mode
    allow_remote_models: bool = False       # Allow remote model calls
    allow_online_tools: bool = False        # Allow online tool access
```

### RequestContext

Per-request context with identifiers.

```python
class RequestContext(BaseModel):
    request_id: UUID        # Unique request identifier
    session_id: UUID        # Session identifier
    runtime: RuntimeContext # Runtime configuration
```

### TelemetryTiming

Timing information for telemetry.

```python
class TelemetryTiming(BaseModel):
    t_iso: str          # ISO timestamp
    elapsed_ms: float   # Elapsed milliseconds
```

---

## Query Processing Schemas

### Router Schemas

#### RouterConfig

```python
class RouterConfig(BaseModel):
    default_query_type: Optional[str] = None  # Default classification
    max_hist_turns: int = 10                  # Max history turns
```

#### Message

Chat message format.

```python
class Message(BaseModel):
    role: str      # "user" | "assistant" | "system"
    content: str   # Message content
```

#### RouterProfile

Query classification result.

```python
class RouterProfile(BaseModel):
    query_type: str           # "lookup" | "explanation" | "comparison" | "list" | "other"
    use_qe: bool              # Enable query expansion
    use_prf: bool             # Enable PRF
    use_rerank: bool          # Enable reranking
    expected_answer_style: str # "short" | "paragraph" | "multi_section"
    complexity_hint: str      # "low" | "medium" | "high"
```

#### RouterInput / RouterOutput

```python
class RouterInput(BaseModel):
    ctx: RequestContext
    user_query: str
    history: List[Message] = Field(default_factory=list)
    config: RouterConfig

class RouterOutput(BaseModel):
    router_profile: RouterProfile
```

### Decomposition Schemas

#### DecompositionConfig

```python
class DecompositionConfig(BaseModel):
    max_subqueries: int = 4        # Max sub-questions
    min_subquery_length: int = 10  # Min characters per sub-question
```

#### Subquery / ComparisonPair

```python
class Subquery(BaseModel):
    id: str    # Unique identifier
    text: str  # Sub-question text

class ComparisonPair(BaseModel):
    left: str   # First item to compare
    right: str  # Second item to compare
```

#### Decomposition

```python
class Decomposition(BaseModel):
    is_multi_part: bool                              # Has multiple parts
    subqueries: List[Subquery] = Field(default_factory=list)
    comparison_pairs: List[ComparisonPair] = Field(default_factory=list)
```

#### DecompositionInput / DecompositionOutput

```python
class DecompositionInput(BaseModel):
    ctx: RequestContext
    user_query: str
    router_profile: RouterProfile
    config: DecompositionConfig

class DecompositionOutput(BaseModel):
    decomposition: Decomposition
```

### Planner Schemas

#### GlobalConfig

System-wide configuration.

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
```

#### PlanIterations

```python
class PlanIterations(BaseModel):
    max_iters: int      # Maximum iterations
    max_rewrites: int   # Maximum query rewrites
```

#### Plan

Execution plan for a query.

```python
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
```

#### PlannerInput / PlannerOutput

```python
class PlannerInput(BaseModel):
    ctx: RequestContext
    router_profile: RouterProfile
    decomposition: Decomposition
    global_config: GlobalConfig

class PlannerOutput(BaseModel):
    plan: Plan
```

### Guardrail Schemas

```python
class GuardrailInput(BaseModel):
    ctx: RequestContext
    plan: Plan

class GuardrailOutput(BaseModel):
    status: str  # "ok" | "adjusted" | "blocked"
    plan: Plan   # Potentially modified plan
    messages: List[str] = Field(default_factory=list)
```

---

## Retrieval Schemas

### Translation Schemas

#### TranslationItem / ResultItem

```python
class TranslationItem(BaseModel):
    id: str    # Item identifier
    text: str  # Text to translate
    type: str  # "query" | "snippet"

class TranslationResultItem(BaseModel):
    id: str
    type: str
    lang: str              # Detected language
    confidence: float      # Detection confidence
    source_text: str       # Original text
    translated_text: str   # Translated text
```

#### TranslationConfig

```python
class TranslationConfig(BaseModel):
    enabled: bool = True
    detect_only: bool = False    # Only detect, don't translate
    target_lang: str = "en"
    min_confidence: float = 0.7
```

#### TranslationMetadata

```python
class TranslationMetadata(BaseModel):
    original_lang: Optional[str] = None
    target_lang: str = "en"
    query_id: Optional[str] = None
    snippet_ids: List[str] = Field(default_factory=list)
```

#### TranslationInput / TranslationOutput

```python
class TranslationInput(BaseModel):
    ctx: RequestContext
    items: List[TranslationItem]
    config: TranslationConfig

class TranslationOutput(BaseModel):
    normalized_query: Optional[str]
    items: List[TranslationResultItem]
    translation_metadata: TranslationMetadata
```

### QE Schemas

```python
class QEInput(BaseModel):
    ctx: RequestContext
    query: str
    router_profile: RouterProfile
    plan: Plan
    translation_metadata: Optional[TranslationMetadata] = None

class QEOutput(BaseModel):
    expanded_queries: List[str]
```

### PRF Schemas

```python
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

#### Snippet

```python
class Snippet(BaseModel):
    chunk_id: str
    score: float
    text: str
    lang: Optional[str] = None
    page: Optional[int] = None
    level: str = "leaf"
```

#### RetrievalResult

```python
class RetrievalResult(BaseModel):
    doc_id: str
    parent_metadata: Dict[str, Any] = Field(default_factory=dict)
    snippets: List[Snippet] = Field(default_factory=list)
```

#### RetrieverInput / RetrieverOutput

```python
class RetrieverInput(BaseModel):
    ctx: RequestContext
    query: str
    expanded_queries: Optional[List[str]] = None
    prf_augmented_query: Optional[str] = None
    plan: Plan

class RetrieverOutput(BaseModel):
    results: List[RetrievalResult]
```

### Rerank Schemas

```python
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

### Context Schemas

#### ContextSnippet

Rich snippet with metadata for generation.

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
```

### Answer Schemas

#### AnswerSection

```python
class AnswerSection(BaseModel):
    title: str
    body: str
```

#### Answer

```python
class Answer(BaseModel):
    text: str                                # Full answer text
    sections: Optional[List[AnswerSection]] = None  # Structured sections
```

#### Citation

```python
class Citation(BaseModel):
    doc_id: str
    chunk_id: str
    page: Optional[int] = None
    score: float
    lang: str
    snippet: str           # Cited text
    original_snippet: str  # Original (untranslated)
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

#### CriticFeedback

```python
class CriticFeedback(BaseModel):
    hallucination_risk: float           # 0-1 risk score
    coverage_score: float               # 0-1 coverage
    missing_topics: List[str] = Field(default_factory=list)
    ambiguities: List[str] = Field(default_factory=list)
    unsupported_claims: List[Dict[str, Any]] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)
```

#### CriticInput / CriticOutput

```python
class CriticInput(BaseModel):
    ctx: RequestContext
    query: str
    answer: Answer
    context_snippets: List[ContextSnippet]
    plan: Plan

class CriticOutput(CriticFeedback):
    pass  # Inherits all CriticFeedback fields
```

### Policy Schemas

#### RetrievalMetrics

```python
class RetrievalMetrics(BaseModel):
    num_docs: int
    avg_score: float
```

#### PolicyInput / PolicyOutput

```python
class PolicyInput(BaseModel):
    ctx: RequestContext
    iteration: int
    plan: Plan
    retrieval_metrics: RetrievalMetrics
    critic_feedback: CriticFeedback

class PolicyOutput(BaseModel):
    decision: DecisionEnum       # FINALIZE | REWRITE | CONTINUE
    reason: str                  # Human-readable reason
    adjustments: Dict[str, Any] = Field(default_factory=dict)
```

### QueryRewrite Schemas

```python
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

### Postprocess Schemas

#### PostprocessPreferences

```python
class PostprocessPreferences(BaseModel):
    format: str = "markdown"        # "markdown" | "plain"
    include_critic_note: bool = True
    include_language_notes: bool = True
```

#### PostprocessMetadata

```python
class PostprocessMetadata(BaseModel):
    critic_summary: str
    languages: List[str]
```

#### PostprocessInput / PostprocessOutput

```python
class PostprocessInput(BaseModel):
    ctx: RequestContext
    query: str
    answer: Answer
    critic_feedback: CriticFeedback
    context_snippets: List[ContextSnippet]
    preferences: PostprocessPreferences

class PostprocessOutput(BaseModel):
    final_text: str
    metadata: PostprocessMetadata
```

---

## Infrastructure Schemas

### Telemetry Schemas

#### TelemetryEvent

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
```

#### TelemetryOutput

```python
class TelemetryOutput(BaseModel):
    status: str = "logged"
    trace_id: str
    sink: str
```

### Tool Schemas

#### ToolLimits

```python
class ToolLimits(BaseModel):
    timeout_seconds: int = 20
    max_bytes: int = 2_000_000
    max_items: int = 50
```

#### ToolExecutionInput / ToolExecutionOutput

```python
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

#### SafetyConfig

```python
class SafetyConfig(BaseModel):
    enabled: bool = True
    mode: str = "log_only"  # "log_only" | "enforce"
    policies: Dict[str, Any] = Field(default_factory=dict)
```

#### SafetyInput / SafetyOutput

```python
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

### Index Schemas

#### IndexStatusInfo

```python
class IndexStatusInfo(BaseModel):
    num_docs: int
    num_chunks: int
    languages: List[str]
    last_updated: Optional[str]
```

#### IndexInput / IndexOutput

```python
class IndexInput(BaseModel):
    ctx: RequestContext
    operation: IndexOperationEnum
    arguments: Dict[str, Any]

class IndexOutput(BaseModel):
    status: str  # "ok" | "error"
    leaf_index: Optional[IndexStatusInfo] = None
    parent_index: Optional[IndexStatusInfo] = None
    num_processed: Optional[int] = None
    errors: List[str] = Field(default_factory=list)
```

---

## Schema Relationships

### Agent Input/Output Mapping

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Schema Flow Through Pipeline                         │
└─────────────────────────────────────────────────────────────────────────────┘

User Query
    │
    ▼
┌───────────┐     RouterInput          ┌─────────────┐
│  Router   │ ────────────────────────▶│RouterOutput │
└───────────┘                          │ └RouterProfile
    │                                  └─────────────┘
    ▼                                        │
┌───────────┐     DecompositionInput         ▼
│Decomposer │ ◀─────────────────────────────┐│
└───────────┘                               ││
    │                                       ││
    ▼  DecompositionOutput                  ││
┌───────────┐     PlannerInput              ││
│  Planner  │ ◀─────────────────────────────┘│
└───────────┘  (router_profile, decomposition, global_config)
    │                                        │
    ▼  PlannerOutput (Plan)                  │
┌───────────┐     GuardrailInput             │
│ Guardrail │ ◀──────────────────────────────┘
└───────────┘
    │
    ▼  GuardrailOutput (Plan)
┌───────────┐     PRFInput
│    PRF    │ ◀──────────────────── query
└───────────┘
    │
    ▼  PRFOutput (prf_terms, augmented_query)
┌───────────┐     QEInput
│    QE     │ ◀──────────────────── augmented_query + plan
└───────────┘
    │
    ▼  QEOutput (expanded_queries)
┌───────────┐     RetrieverInput
│ Retriever │ ◀──────────────────── query + expanded + prf + plan
└───────────┘
    │
    ▼  RetrieverOutput (List[RetrievalResult])
┌───────────┐     RerankInput
│  Rerank   │ ◀──────────────────── results + query + plan
└───────────┘
    │
    ▼  RerankOutput (List[RetrievalResult])
    │
    │  Convert to ContextSnippet[]
    ▼
┌───────────┐     GeneratorInput
│ Generator │ ◀──────────────────── query + plan + context_snippets
└───────────┘
    │
    ▼  GeneratorOutput (Answer + Citations)
┌───────────┐     CriticInput
│  Critic   │ ◀──────────────────── answer + context + plan
└───────────┘
    │
    ▼  CriticOutput (CriticFeedback)
┌───────────┐     PolicyInput
│  Policy   │ ◀──────────────────── critic_feedback + metrics + plan
└───────────┘
    │
    ▼  PolicyOutput (DecisionEnum)
    │
    ├──── FINALIZE ────▶ PostprocessInput ────▶ PostprocessOutput
    │
    └──── REWRITE ─────▶ QueryRewriteInput
                              │
                              ▼
                        QueryRewriteOutput (rewritten_query)
                              │
                              └──── Loop back to PRF
```

### Shared Schema Usage

| Schema | Used By |
|--------|---------|
| `RequestContext` | All agents (via input schemas) |
| `Plan` | Planner, Guardrail, Retriever, Rerank, Generator, Critic, Policy, Rewrite |
| `RouterProfile` | Router, Decomposition, Planner, QE |
| `CriticFeedback` | Critic, Policy, Rewrite, Postprocess |
| `ContextSnippet` | Generator, Critic, Postprocess |
| `RetrievalResult` | Retriever, Rerank |

---

## Testing Strategies

### Unit Tests

#### 1. Basic Model Tests

```python
import pytest
from uuid import uuid4
from core.schemas import (
    RequestContext, RuntimeContext, BackendEnum, RuntimeModeEnum,
    RouterProfile, Plan, PlanIterations, RetrievalModeEnum,
)

class TestBasicModels:
    
    def test_request_context_creation(self):
        runtime = RuntimeContext(
            offline=True,
            backend=BackendEnum.HF,
            mode=RuntimeModeEnum.RAG
        )
        ctx = RequestContext(
            request_id=uuid4(),
            session_id=uuid4(),
            runtime=runtime
        )
        
        assert ctx.runtime.offline == True
        assert ctx.runtime.backend == BackendEnum.HF
    
    def test_router_profile_defaults(self):
        profile = RouterProfile(
            query_type="lookup",
            use_qe=True,
            use_prf=True,
            use_rerank=True,
            expected_answer_style="short",
            complexity_hint="low"
        )
        
        assert profile.query_type == "lookup"
        assert profile.use_qe == True
    
    def test_plan_structure(self):
        plan = Plan(
            retrieval_mode=RetrievalModeEnum.DUAL_INDEX,
            use_qe=True,
            use_prf=True,
            use_rerank=True,
            iterations=PlanIterations(max_iters=3, max_rewrites=2),
            top_k=10,
            rerank_top_k=20,
            language="en",
            allow_online_tools=False,
            backend=BackendEnum.OPENAI_COMPAT
        )
        
        assert plan.iterations.max_iters == 3
        assert plan.top_k == 10
```

#### 2. Serialization Tests

```python
class TestSerialization:
    
    def test_model_to_dict(self):
        profile = RouterProfile(
            query_type="explanation",
            use_qe=True,
            use_prf=False,
            use_rerank=True,
            expected_answer_style="paragraph",
            complexity_hint="medium"
        )
        
        data = profile.model_dump()
        
        assert isinstance(data, dict)
        assert data["query_type"] == "explanation"
        assert data["use_prf"] == False
    
    def test_model_to_json(self):
        from core.schemas import Snippet
        
        snippet = Snippet(
            chunk_id="c1",
            score=0.95,
            text="Sample text",
            lang="en",
            page=5
        )
        
        json_str = snippet.model_dump_json()
        
        assert '"chunk_id":"c1"' in json_str
        assert '"score":0.95' in json_str
    
    def test_model_from_dict(self):
        data = {
            "chunk_id": "c2",
            "score": 0.8,
            "text": "Another text",
            "level": "parent"
        }
        
        snippet = Snippet(**data)
        
        assert snippet.chunk_id == "c2"
        assert snippet.level == "parent"
```

#### 3. Validation Tests

```python
class TestValidation:
    
    def test_required_fields(self):
        with pytest.raises(Exception):  # ValidationError
            RouterProfile()  # Missing required fields
    
    def test_enum_validation(self):
        with pytest.raises(Exception):
            RuntimeContext(backend="invalid_backend")
    
    def test_type_coercion(self):
        # Pydantic coerces compatible types
        from core.schemas import PRFConfig
        
        config = PRFConfig(top_k="10")  # String coerced to int
        assert config.top_k == 10
        assert isinstance(config.top_k, int)
```

#### 4. Default Value Tests

```python
class TestDefaults:
    
    def test_global_config_defaults(self):
        from core.schemas import GlobalConfig
        
        config = GlobalConfig()
        
        assert config.enable_qe == True
        assert config.enable_prf == True
        assert config.max_iters == 3
        assert config.top_k == 10
    
    def test_list_field_defaults(self):
        from core.schemas import CriticFeedback
        
        feedback = CriticFeedback(
            hallucination_risk=0.2,
            coverage_score=0.8
        )
        
        assert feedback.missing_topics == []
        assert feedback.notes == []
    
    def test_optional_field_defaults(self):
        from core.schemas import Snippet
        
        snippet = Snippet(
            chunk_id="c1",
            score=0.9,
            text="text"
        )
        
        assert snippet.lang is None
        assert snippet.page is None
        assert snippet.level == "leaf"
```

#### 5. Enum Tests

```python
class TestEnums:
    
    def test_decision_enum_values(self):
        from core.schemas import DecisionEnum
        
        assert DecisionEnum.FINALIZE.value == "finalize"
        assert DecisionEnum.REWRITE.value == "rewrite"
        assert DecisionEnum.CONTINUE.value == "continue"
    
    def test_enum_from_string(self):
        from core.schemas import BackendEnum
        
        backend = BackendEnum("hf")
        assert backend == BackendEnum.HF
        
        backend2 = BackendEnum["OPENAI_COMPAT"]
        assert backend2 == BackendEnum.OPENAI_COMPAT
    
    def test_enum_in_model(self):
        from core.schemas import PolicyOutput, DecisionEnum
        
        output = PolicyOutput(
            decision=DecisionEnum.REWRITE,
            reason="Low coverage"
        )
        
        assert output.decision == DecisionEnum.REWRITE
        assert output.decision.value == "rewrite"
```

#### 6. Complex Schema Tests

```python
class TestComplexSchemas:
    
    def test_nested_models(self):
        from core.schemas import (
            PlannerInput, RequestContext, RuntimeContext,
            RouterProfile, Decomposition, GlobalConfig
        )
        from uuid import uuid4
        
        inp = PlannerInput(
            ctx=RequestContext(
                request_id=uuid4(),
                session_id=uuid4(),
                runtime=RuntimeContext()
            ),
            router_profile=RouterProfile(
                query_type="lookup",
                use_qe=True,
                use_prf=True,
                use_rerank=True,
                expected_answer_style="short",
                complexity_hint="low"
            ),
            decomposition=Decomposition(is_multi_part=False),
            global_config=GlobalConfig()
        )
        
        assert inp.router_profile.query_type == "lookup"
        assert inp.global_config.max_iters == 3
    
    def test_retrieval_result_with_snippets(self):
        from core.schemas import RetrievalResult, Snippet
        
        result = RetrievalResult(
            doc_id="doc1",
            parent_metadata={"title": "Test Doc"},
            snippets=[
                Snippet(chunk_id="c1", score=0.9, text="text1"),
                Snippet(chunk_id="c2", score=0.8, text="text2"),
            ]
        )
        
        assert len(result.snippets) == 2
        assert result.snippets[0].score > result.snippets[1].score
```

### Test Commands

```bash
# Run schema tests
pytest test_schemas.py -v

# Run with validation coverage
pytest test_schemas.py --cov=core.schemas --cov-report=html

# Type checking
mypy core/schemas.py
```

---

## Recommendations and Improvements

### High Priority Improvements

#### 1. Add Field Descriptions

**Problem:** Fields lack documentation.

**Recommendation:** Add descriptions:

```python
class RouterProfile(BaseModel):
    query_type: str = Field(
        description="Classification of query intent: lookup, explanation, comparison, list, or other"
    )
    use_qe: bool = Field(
        description="Whether to enable query expansion for this query"
    )
    complexity_hint: str = Field(
        description="Estimated query complexity: low, medium, or high"
    )
```

#### 2. Add Field Constraints

**Problem:** No validation constraints on values.

**Recommendation:** Add validators:

```python
from pydantic import field_validator

class CriticFeedback(BaseModel):
    hallucination_risk: float = Field(ge=0.0, le=1.0)
    coverage_score: float = Field(ge=0.0, le=1.0)
    
    @field_validator('hallucination_risk', 'coverage_score')
    @classmethod
    def validate_score(cls, v):
        if not 0 <= v <= 1:
            raise ValueError(f"Score must be between 0 and 1, got {v}")
        return v
```

#### 3. Add Computed Properties

**Recommendation:** Add useful derived fields:

```python
class CriticFeedback(BaseModel):
    hallucination_risk: float
    coverage_score: float
    
    @property
    def quality_score(self) -> float:
        """Combined quality score (coverage - risk)."""
        return self.coverage_score - self.hallucination_risk
    
    @property
    def needs_revision(self) -> bool:
        """Whether this feedback suggests revision is needed."""
        return self.coverage_score < 0.6 or self.hallucination_risk > 0.4
```

---

### Medium Priority Improvements

#### 4. Add Schema Versioning

**Recommendation:** Track schema versions:

```python
class SchemaVersion:
    MAJOR = 1
    MINOR = 0
    PATCH = 0
    
    @classmethod
    def string(cls) -> str:
        return f"{cls.MAJOR}.{cls.MINOR}.{cls.PATCH}"

class BaseSchema(BaseModel):
    schema_version: str = Field(default=SchemaVersion.string())
```

#### 5. Add Example Values

**Recommendation:** Add examples for documentation:

```python
class RouterProfile(BaseModel):
    query_type: str = Field(
        example="explanation",
        description="Query classification"
    )
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "query_type": "lookup",
                    "use_qe": True,
                    "use_prf": True,
                    "use_rerank": True,
                    "expected_answer_style": "short",
                    "complexity_hint": "low"
                }
            ]
        }
    }
```

#### 6. Add Custom Serializers

**Recommendation:** Control serialization format:

```python
from pydantic import field_serializer

class TelemetryEvent(BaseModel):
    timing: TelemetryTiming
    
    @field_serializer('timing')
    def serialize_timing(self, timing: TelemetryTiming, _info):
        return {
            "timestamp": timing.t_iso,
            "duration_ms": round(timing.elapsed_ms, 2)
        }
```

---

### Low Priority / Future Enhancements

#### 7. Add Immutable Models

**Recommendation:** Make some models immutable:

```python
class Plan(BaseModel):
    model_config = {"frozen": True}
    
    # Fields cannot be modified after creation
```

#### 8. Add Union Types for Flexibility

**Recommendation:** Support alternative schemas:

```python
from typing import Union

class GeneratorInput(BaseModel):
    context: Union[List[ContextSnippet], List[Document], str]
```

#### 9. Add Generic Pagination

**Recommendation:** Standard pagination schema:

```python
from typing import Generic, TypeVar

T = TypeVar('T')

class PaginatedResponse(BaseModel, Generic[T]):
    items: List[T]
    total: int
    page: int
    page_size: int
    has_more: bool
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

# Create runtime context
runtime = RuntimeContext(
    offline=False,
    backend=BackendEnum.OPENAI_COMPAT,
    mode=RuntimeModeEnum.AGENTIC,
    allow_remote_models=True
)

# Create request context
ctx = RequestContext(
    request_id=uuid4(),
    session_id=uuid4(),
    runtime=runtime
)

# Create router input
router_input = RouterInput(
    ctx=ctx,
    user_query="What is RAG?",
    history=[
        Message(role="user", content="Hello"),
        Message(role="assistant", content="Hi! How can I help?")
    ],
    config=RouterConfig(max_hist_turns=5)
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
    complexity_hint="medium"
)

# To dictionary
data = profile.model_dump()
print(data)
# {'query_type': 'explanation', 'use_qe': True, ...}

# To JSON
json_str = profile.model_dump_json()
print(json_str)
# '{"query_type":"explanation","use_qe":true,...}'

# From dictionary
profile2 = RouterProfile(**data)
profile3 = RouterProfile.model_validate(data)
```

### Working with Enums

```python
from core.schemas import DecisionEnum, PolicyOutput

# Create with enum
output = PolicyOutput(
    decision=DecisionEnum.REWRITE,
    reason="Low coverage score"
)

# Check decision
if output.decision == DecisionEnum.FINALIZE:
    print("Accept answer")
elif output.decision == DecisionEnum.REWRITE:
    print("Rewrite query")

# Get enum value
print(output.decision.value)  # "rewrite"
print(output.decision.name)   # "REWRITE"
```

### Nested Schemas

```python
from core.schemas import (
    GeneratorOutput, Answer, AnswerSection, Citation
)

output = GeneratorOutput(
    answer=Answer(
        text="RAG combines retrieval with generation...",
        sections=[
            AnswerSection(title="Overview", body="RAG is..."),
            AnswerSection(title="Benefits", body="The benefits include...")
        ]
    ),
    citations=[
        Citation(
            doc_id="doc1",
            chunk_id="c1",
            page=5,
            score=0.95,
            lang="en",
            snippet="RAG was introduced...",
            original_snippet="RAG was introduced..."
        )
    ]
)

print(f"Answer has {len(output.answer.sections)} sections")
print(f"Found {len(output.citations)} citations")
```

---

## Appendix

### Schema Count Summary

| Category | Models | Enums |
|----------|--------|-------|
| Common | 3 | - |
| Query Processing | 15 | - |
| Translation | 6 | - |
| Retrieval | 10 | - |
| Generation | 17 | - |
| Telemetry | 2 | - |
| Infrastructure | 10 | - |
| Enumerations | - | 7 |
| **Total** | **63** | **7** |

### Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | Initial | 70 schemas and enums defined |

### References

- Radiant Repository: https://github.com/dshipley71/radiant
- Related Files: `core/interfaces.py`, `core/orchestrator.py`, all agent implementations
- Pydantic Documentation: https://docs.pydantic.dev/

---

*Document generated for the Radiant RAG Pipeline project. For questions or contributions, please refer to the project repository.*
