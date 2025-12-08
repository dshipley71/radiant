# Orchestrator Documentation

## Technical Reference for the Radiant RAG Pipeline Coordinator

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture Design](#architecture-design)
3. [Core Components](#core-components)
4. [Pipeline Execution Flow](#pipeline-execution-flow)
5. [Configuration System](#configuration-system)
6. [Agent Registry](#agent-registry)
7. [Telemetry System](#telemetry-system)
8. [Helper Functions](#helper-functions)
9. [Testing Strategies](#testing-strategies)
10. [Recommendations and Improvements](#recommendations-and-improvements)
11. [Usage Examples](#usage-examples)

---

## Overview

### Purpose

The `orchestrator.py` module is the central coordinator for the Radiant RAG pipeline. It manages agent registration, configuration loading, request context creation, telemetry collection, and the main agentic iteration loop that transforms user queries into answers.

### Key Responsibilities

- Load and cache configuration from YAML files
- Register and manage all pipeline agents
- Build request contexts with runtime settings
- Execute the multi-stage RAG pipeline
- Manage the iterative refinement loop (generate → critique → decide → rewrite)
- Collect and emit telemetry events
- Convert retrieval results to context snippets and documents

### Design Philosophy

The orchestrator implements a **centralized coordination** pattern where all pipeline stages are orchestrated from a single location. This provides clear control flow, comprehensive telemetry, and the ability to implement complex iteration patterns (rewrites, retries) without distributed state management.

---

## Architecture Design

### High-Level Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              User Query                                     │
│                    "What is hierarchical RAG?"                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           QUERY PHASE                                       │
│  ┌─────────┐   ┌─────────────┐   ┌─────────┐   ┌───────────┐               │
│  │ Router  │──▶│Decomposition│──▶│ Planner │──▶│ Guardrail │               │
│  └─────────┘   └─────────────┘   └─────────┘   └───────────┘               │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         RETRIEVAL PHASE                                     │
│  ┌─────────┐   ┌─────────┐   ┌───────────┐   ┌─────────┐                   │
│  │   PRF   │──▶│   QE    │──▶│ Retriever │──▶│ Rerank  │                   │
│  └─────────┘   └─────────┘   └───────────┘   └─────────┘                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        ITERATION LOOP                                       │
│                                                                             │
│    ┌──────────────────────────────────────────────────────────────────┐    │
│    │                                                                  │    │
│    │  ┌───────────┐   ┌─────────┐   ┌─────────┐                      │    │
│    │  │ Generator │──▶│ Critic  │──▶│ Policy  │                      │    │
│    │  └───────────┘   └─────────┘   └────┬────┘                      │    │
│    │                                     │                            │    │
│    │                    ┌────────────────┼────────────────┐          │    │
│    │                    │                │                │          │    │
│    │              FINALIZE          REWRITE          CONTINUE        │    │
│    │                    │                │                │          │    │
│    │                    ▼                ▼                │          │    │
│    │               [exit loop]    ┌─────────┐            │          │    │
│    │                              │ Rewrite │────────────┘          │    │
│    │                              │  Agent  │                        │    │
│    │                              └────┬────┘                        │    │
│    │                                   │                             │    │
│    │                    ┌──────────────┴──────────────┐              │    │
│    │                    ▼                             ▼              │    │
│    │              ┌─────────┐                  ┌───────────┐         │    │
│    │              │PRF + QE │                  │ Retriever │         │    │
│    │              └─────────┘                  │ + Rerank  │         │    │
│    │                    │                      └───────────┘         │    │
│    │                    └────────────────────────────┘               │    │
│    │                                   │                             │    │
│    └───────────────────────────────────┴─────────────────────────────┘    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        FINALIZATION                                         │
│  ┌─────────────┐   ┌──────────────────┐                                    │
│  │PostProcessor│──▶│ Return Metadata  │                                    │
│  └─────────────┘   └──────────────────┘                                    │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Module Structure

```
orchestrator.py
├── Global State
│   ├── CONFIG                    # Cached configuration dict
│   ├── RUNTIME_CONFIG            # Runtime settings
│   ├── TELEMETRY_EVENTS         # Global telemetry buffer
│   └── REGISTRY                  # Agent registry singleton
│
├── Classes
│   └── AgentRegistry            # Agent registration/lookup
│
├── Configuration Functions
│   ├── _load_config()           # Load YAML config
│   ├── _init_runtime_from_config()  # Extract runtime settings
│   └── _build_global_config_from_yaml()  # Build GlobalConfig
│
├── Telemetry Functions
│   ├── _now_iso_utc()           # Timestamp helper
│   ├── _log_telemetry_with_elapsed()  # Log telemetry event
│   └── clear_telemetry()        # Clear telemetry buffer
│
├── Helper Functions
│   ├── build_context_snippets_from_results()  # Convert to snippets
│   ├── build_documents_from_results()  # Convert to Documents
│   ├── build_retrieval_metrics()  # Compute metrics
│   ├── build_citations_from_context_snippets()  # Extract citations
│   └── build_request_context()   # Create RequestContext
│
├── Registration Functions
│   └── register_default_agents()  # Register all agents
│
└── Main Functions
    ├── agentic_once_with_metadata()  # Full pipeline with metadata
    └── agentic_once()               # Simple wrapper
```

---

## Core Components

### Global State Variables

| Variable | Type | Purpose |
|----------|------|---------|
| `CONFIG` | `Dict[str, Any]` | Cached configuration from YAML |
| `RUNTIME_CONFIG` | `Dict[str, Any]` | Runtime settings extracted from config |
| `TELEMETRY_EVENTS` | `List[TelemetryEvent]` | Global telemetry buffer |
| `REGISTRY` | `AgentRegistry` | Singleton agent registry |

### AgentRegistry Class

```python
class AgentRegistry:
    """Minimal global registry: agents are registered by their .role attribute."""
    
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
```

### Registered Agents

| Role | Agent Class | Type |
|------|-------------|------|
| `router` | `BasicRouterAgent` | Core |
| `decomposition` | `BasicDecompositionAgent` | Core |
| `planner` | `BasicPlannerAgent` | Core |
| `guardrail` | `BasicGuardrailAgent` | Core |
| `critic` | `BasicCriticAgent` | Core |
| `policy` | `BasicPolicyAgent` | Core |
| `telemetry` | `BasicTelemetryAgent` | Infrastructure |
| `prf` | `BasicPRFAgent` | Retrieval |
| `rerank` | `BasicRerankAgent` | Retrieval |
| `postprocess` | `BasicPostProcessorAgent` | Core |
| `qe` | `LLMQEAgent` | LLM-backed |
| `rewrite` | `LLMQueryRewriteAgent` | LLM-backed |
| `generator` | `LLMGeneratorAgent` | LLM-backed |
| `retriever` | `HaystackChromaRetrieverAgent` | Retrieval |

---

## Pipeline Execution Flow

### Phase 1: Query Processing

```
┌─────────────────────────────────────────────────────────────────┐
│                      Query Processing                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. Router                                                      │
│     Input:  RouterInput(ctx, user_query, history, config)      │
│     Output: RouterOutput(router_profile)                        │
│     Telemetry: "router.decided"                                │
│                                                                 │
│  2. Decomposition                                               │
│     Input:  DecompositionInput(ctx, user_query, router_profile)│
│     Output: DecompositionOutput(decomposition)                  │
│     Telemetry: "decomposition.done"                            │
│                                                                 │
│  3. Planner                                                     │
│     Input:  PlannerInput(ctx, router_profile, decomposition,   │
│                          global_config)                        │
│     Output: PlannerOutput(plan)                                 │
│     Telemetry: "planner.plan"                                  │
│                                                                 │
│  4. Guardrail                                                   │
│     Input:  GuardrailInput(ctx, plan)                          │
│     Output: GuardrailOutput(plan) [potentially modified]       │
│     Telemetry: "guardrail.validate_plan"                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Phase 2: Retrieval Enhancement

```
┌─────────────────────────────────────────────────────────────────┐
│                    Retrieval Enhancement                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  5. PRF (Pseudo-Relevance Feedback)                            │
│     Input:  PRFInput(ctx, query, bm25_config)                  │
│     Output: PRFOutput(augmented_query, prf_terms)              │
│     Telemetry: "prf.compute"                                   │
│                                                                 │
│  6. QE (Query Expansion) [if plan.use_qe]                      │
│     Input:  QEInput(ctx, query, router_profile, plan)          │
│     Output: QEOutput(expanded_queries)                          │
│     Telemetry: "qe.expand"                                     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Phase 3: Document Retrieval

```
┌─────────────────────────────────────────────────────────────────┐
│                    Document Retrieval                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  7. Retriever                                                   │
│     Input:  RetrieverInput(ctx, query, expanded_queries,       │
│                            prf_augmented_query, plan)          │
│     Output: RetrieverOutput(results)                            │
│     Telemetry: "retriever.results"                             │
│                                                                 │
│  8. Rerank                                                      │
│     Input:  RerankInput(ctx, query, results, plan)             │
│     Output: RerankOutput(results) [reordered]                  │
│     Telemetry: "rerank.rerank"                                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Phase 4: Iteration Loop

```
┌─────────────────────────────────────────────────────────────────┐
│                      Iteration Loop                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  for i in range(max_iters):                                    │
│                                                                 │
│    9. Generator                                                 │
│       Input:  GeneratorInput(ctx, rag_query, plan, snippets)   │
│       Output: GeneratorOutput(answer, citations)                │
│       Telemetry: "generator.output"                            │
│                                                                 │
│    10. Critic                                                   │
│        Input:  CriticInput(ctx, query, plan, answer, citations,│
│                            context_snippets)                   │
│        Output: CriticFeedback(coverage_score, hallucination_risk,│
│                              notes, ...)                        │
│        Telemetry: "critic.evaluate"                            │
│                                                                 │
│    11. Policy                                                   │
│        Input:  PolicyInput(ctx, iteration, plan,               │
│                            retrieval_metrics, critic_feedback) │
│        Output: PolicyOutput(decision, adjustments)             │
│        Telemetry: "policy.decision"                            │
│                                                                 │
│    Decision Handling:                                           │
│      - FINALIZE/STOP: break loop                               │
│      - REWRITE/REVISION/RETRY: run rewrite agent               │
│                                re-run PRF, QE, retrieval, rerank│
│                                continue loop                    │
│      - CONTINUE: continue loop (same query)                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Phase 5: Finalization

```
┌─────────────────────────────────────────────────────────────────┐
│                      Finalization                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  12. PostProcessor                                              │
│      Input:  PostprocessInput(ctx, query, plan, answer,        │
│                               critic_feedback, router_profile, │
│                               context_snippets, iterations,    │
│                               preferences)                     │
│      Output: PostprocessOutput(final_text, metadata)           │
│      Telemetry: "postprocess.format"                           │
│                                                                 │
│  13. Return aggregated metadata dictionary                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Configuration System

### Configuration Loading

```python
def _load_config(config_path: str | None) -> Dict[str, Any]:
    """
    Load YAML configuration and cache in global CONFIG.
    
    Resolution order:
    1. Explicit config_path parameter
    2. AGENTIC_RAG_CONFIG environment variable
    3. ../config/config.fast.yaml (default)
    """
```

### Configuration Structure

```yaml
# config.fast.yaml

runtime:
  backend: HF           # HF | OPENAI
  mode: RAG             # RAG | AGENT | HYBRID
  offline: true
  allow_remote_models: true
  allow_online_tools: false

agentic:
  planner:
    enable_qe: true
    enable_prf: true
    enable_rerank: true
    max_iters: 3
    max_rewrites: 2
    max_time_seconds: 60
    top_k: 10
    rerank_top_k: 100
    language: "en"
    allow_online_tools: false

llm:
  api_base: "https://api.openai.com/v1"
  api_key: "${OPENAI_API_KEY}"
  model: "gpt-4o-mini"
  temperature: 0.2
  max_tokens: 256

models:
  use_local: false
  llm_model: "meta-llama/Llama-2-7b-chat-hf"
  llm_device: "cuda"

retrieval:
  # ... retrieval-specific settings
```

### GlobalConfig Building

```python
def _build_global_config_from_yaml(cfg: Dict[str, Any]) -> GlobalConfig:
    """
    Construct GlobalConfig from config.fast.yaml.
    
    Maps agentic.planner.* settings to GlobalConfig fields:
    - enable_qe, enable_prf, enable_rerank
    - max_iters, max_rewrites, max_time_seconds
    - top_k, rerank_top_k
    - language, allow_online_tools
    """
```

### Environment Variable Overrides

| Variable | Purpose |
|----------|---------|
| `AGENTIC_RAG_CONFIG` | Override config file path |
| `AGENTIC_MAX_ITERS` | Override max iterations |
| `AGENTIC_MIN_ITERS` | Override min iterations |

---

## Agent Registry

### Registration Process

```python
def register_default_agents(config_path: str | None = None) -> AgentRegistry:
    """Register all default agents in the global REGISTRY."""
    
    cfg = _load_config(config_path)
    _init_runtime_from_config(cfg)
    
    # Core agents (no LLM dependency)
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
    
    # Retrieval agent
    REGISTRY.register(HaystackChromaRetrieverAgent(...))
    
    return REGISTRY
```

### Agent Retrieval

```python
# Get agent by role
router = REGISTRY.get("router")
generator = REGISTRY.get("generator")

# Missing role raises ValueError
try:
    agent = REGISTRY.get("nonexistent")
except ValueError as e:
    print(e)  # "No agent registered for role='nonexistent'"
```

---

## Telemetry System

### TelemetryEvent Structure

```python
@dataclass
class TelemetryEvent:
    ctx: RequestContext      # Request context
    phase: PhaseEnum         # Pipeline phase
    backend: BackendEnum     # Backend type
    mode: RuntimeModeEnum    # Runtime mode
    agent: str               # Agent name
    event_type: str          # Event type identifier
    iteration: int           # Loop iteration
    timing: TelemetryTiming  # Timing info
    payload: Dict[str, Any]  # Additional data
    model: Optional[str]     # Model name
```

### Event Types

| Agent | Event Type | When |
|-------|------------|------|
| Router | `router.decided` | After classification |
| Decomposition | `decomposition.done` | After decomposition |
| Planner | `planner.plan` | After plan creation |
| Guardrail | `guardrail.validate_plan` | After validation |
| PRF | `prf.compute` | After PRF computation |
| PRF | `prf.compute_rewrite` | After rewrite PRF |
| QE | `qe.expand` | After expansion |
| QE | `qe.expand_rewrite` | After rewrite expansion |
| Retriever | `retriever.results` | After retrieval |
| Retriever | `retriever.results_rewrite` | After rewrite retrieval |
| Rerank | `rerank.rerank` | After reranking |
| Rerank | `rerank.rerank_rewrite` | After rewrite reranking |
| Generator | `generator.output` | After generation |
| Critic | `critic.evaluate` | After evaluation |
| Policy | `policy.decision` | After decision |
| Rewrite | `rewrite.rewrite` | After query rewrite |
| PostProcessor | `postprocess.format` | After formatting |

### Telemetry Logging

```python
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
    Compute elapsed_ms and emit TelemetryEvent.
    Also appends to global TELEMETRY_EVENTS buffer.
    """
```

### Telemetry Buffer Management

```python
# Access events
from core.orchestrator import TELEMETRY_EVENTS

for event in TELEMETRY_EVENTS:
    print(f"{event.agent}: {event.event_type} ({event.timing.elapsed_ms:.2f}ms)")

# Clear buffer
from core.orchestrator import clear_telemetry
clear_telemetry()
```

---

## Helper Functions

### build_context_snippets_from_results

Converts `RetrievalResult` objects to `ContextSnippet` objects.

```python
def build_context_snippets_from_results(
    results: List[RetrievalResult],
    limit: int = 10,
) -> List[ContextSnippet]:
    """
    ContextSnippet fields:
    - doc_id, chunk_id
    - source_text, translated_text
    - lang, score
    - page, doc_title, level
    """
```

### build_documents_from_results

Converts results to Haystack `Document` objects for the generator.

```python
def build_documents_from_results(
    results: List[RetrievalResult],
    limit: int = 10,
) -> List[Document]:
    """
    Document fields:
    - id: "{doc_id}::chunk::{chunk_id}"
    - content: snippet text
    - meta: doc_id, chunk_id, score, lang, page, doc_title, source_path
    """
```

### build_retrieval_metrics

Computes summary metrics from retrieval results.

```python
def build_retrieval_metrics(results: List[RetrievalResult]) -> RetrievalMetrics:
    """
    Returns:
    - num_docs: count of results
    - avg_score: average of best snippet scores
    """
```

### build_citations_from_context_snippets

Extracts citations (first N snippets per document).

```python
def build_citations_from_context_snippets(
    contexts: List[ContextSnippet],
    per_doc: int = 1,
    limit: int = 10,
) -> List[ContextSnippet]:
```

### build_request_context

Creates a `RequestContext` with runtime settings from config.

```python
def build_request_context() -> RequestContext:
    """
    Builds RequestContext with RuntimeContext from RUNTIME_CONFIG.
    
    RuntimeContext fields:
    - backend: HF | OPENAI
    - mode: RAG | AGENT | HYBRID
    - offline: bool
    - allow_remote_models: bool
    - allow_online_tools: bool
    """
```

---

## Testing Strategies

### Unit Tests

#### 1. Configuration Loading Tests

```python
import pytest
from unittest.mock import patch, mock_open
from core.orchestrator import _load_config, CONFIG

class TestConfigLoading:
    
    def test_load_yaml_config(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
agentic:
  planner:
    max_iters: 5
llm:
  model: gpt-4
""")
        cfg = _load_config(str(config_file))
        
        assert cfg["agentic"]["planner"]["max_iters"] == 5
        assert cfg["llm"]["model"] == "gpt-4"
    
    def test_missing_config_returns_empty(self):
        cfg = _load_config("/nonexistent/config.yaml")
        
        assert cfg == {}
    
    def test_env_var_override(self):
        with patch.dict('os.environ', {'AGENTIC_RAG_CONFIG': '/custom/path.yaml'}):
            # Would use env var path
            pass
```

#### 2. Agent Registry Tests

```python
from core.orchestrator import AgentRegistry
from core.interfaces import BaseAgent

class MockAgent(BaseAgent):
    role = "test_role"
    
    @property
    def name(self):
        return "MockAgent"
    
    def describe(self):
        return "Test agent"

class TestAgentRegistry:
    
    def test_register_and_get(self):
        registry = AgentRegistry()
        agent = MockAgent()
        
        registry.register(agent)
        retrieved = registry.get("test_role")
        
        assert retrieved is agent
    
    def test_get_missing_raises(self):
        registry = AgentRegistry()
        
        with pytest.raises(ValueError, match="No agent registered"):
            registry.get("nonexistent")
    
    def test_register_no_role_raises(self):
        registry = AgentRegistry()
        
        class NoRoleAgent:
            pass
        
        with pytest.raises(ValueError, match="no 'role' attribute"):
            registry.register(NoRoleAgent())
```

#### 3. Telemetry Tests

```python
from core.orchestrator import (
    _log_telemetry_with_elapsed,
    TELEMETRY_EVENTS,
    clear_telemetry,
    build_request_context,
)
from core.schemas import PhaseEnum
import time

class TestTelemetry:
    
    def setup_method(self):
        clear_telemetry()
    
    def test_log_telemetry_appends_event(self):
        ctx = build_request_context()
        mock_agent = MockTelemetryAgent()
        
        _log_telemetry_with_elapsed(
            telem_agent=mock_agent,
            ctx=ctx,
            phase=PhaseEnum.QUERY,
            agent_name="TestAgent",
            event_type="test.event",
            start_time=time.perf_counter() - 0.1,
            payload={"key": "value"},
            iteration=0,
        )
        
        assert len(TELEMETRY_EVENTS) == 1
        event = TELEMETRY_EVENTS[0]
        assert event.agent == "TestAgent"
        assert event.event_type == "test.event"
        assert event.timing.elapsed_ms > 0
    
    def test_clear_telemetry(self):
        TELEMETRY_EVENTS.append("dummy")
        
        clear_telemetry()
        
        assert len(TELEMETRY_EVENTS) == 0
```

#### 4. Helper Function Tests

```python
from core.orchestrator import (
    build_context_snippets_from_results,
    build_documents_from_results,
    build_retrieval_metrics,
    build_citations_from_context_snippets,
)
from core.schemas import RetrievalResult, Snippet

class TestHelperFunctions:
    
    @pytest.fixture
    def sample_results(self):
        return [
            RetrievalResult(
                doc_id="doc1",
                parent_metadata={"title": "Document 1"},
                snippets=[
                    Snippet(chunk_id="c1", text="Text 1", score=0.9, lang="en"),
                    Snippet(chunk_id="c2", text="Text 2", score=0.8, lang="en"),
                ]
            ),
            RetrievalResult(
                doc_id="doc2",
                parent_metadata={"title": "Document 2"},
                snippets=[
                    Snippet(chunk_id="c3", text="Text 3", score=0.7, lang="en"),
                ]
            ),
        ]
    
    def test_build_context_snippets(self, sample_results):
        snippets = build_context_snippets_from_results(sample_results, limit=10)
        
        assert len(snippets) == 3
        assert snippets[0].doc_id == "doc1"
        assert snippets[0].source_text == "Text 1"
    
    def test_build_context_snippets_respects_limit(self, sample_results):
        snippets = build_context_snippets_from_results(sample_results, limit=2)
        
        assert len(snippets) == 2
    
    def test_build_documents(self, sample_results):
        docs = build_documents_from_results(sample_results, limit=10)
        
        assert len(docs) == 3
        assert docs[0].id == "doc1::chunk::c1"
        assert docs[0].content == "Text 1"
        assert docs[0].meta["score"] == 0.9
    
    def test_build_retrieval_metrics(self, sample_results):
        metrics = build_retrieval_metrics(sample_results)
        
        assert metrics.num_docs == 2
        assert metrics.avg_score == pytest.approx(0.8, abs=0.01)
    
    def test_build_citations(self):
        snippets = [
            ContextSnippet(doc_id="d1", chunk_id="c1", source_text="t1", ...),
            ContextSnippet(doc_id="d1", chunk_id="c2", source_text="t2", ...),
            ContextSnippet(doc_id="d2", chunk_id="c3", source_text="t3", ...),
        ]
        
        citations = build_citations_from_context_snippets(snippets, per_doc=1, limit=10)
        
        assert len(citations) == 2  # One per doc
```

#### 5. Request Context Tests

```python
from core.orchestrator import build_request_context, RUNTIME_CONFIG
from core.schemas import BackendEnum, RuntimeModeEnum

class TestRequestContext:
    
    def test_default_context(self):
        ctx = build_request_context()
        
        assert ctx.request_id is not None
        assert ctx.session_id is not None
        assert ctx.runtime.backend == BackendEnum.HF
        assert ctx.runtime.mode == RuntimeModeEnum.RAG
    
    def test_runtime_config_override(self):
        global RUNTIME_CONFIG
        RUNTIME_CONFIG = {
            "backend": "OPENAI",
            "mode": "AGENT",
            "offline": False,
        }
        
        ctx = build_request_context()
        
        assert ctx.runtime.backend == BackendEnum.OPENAI
        assert ctx.runtime.mode == RuntimeModeEnum.AGENT
        assert ctx.runtime.offline == False
        
        RUNTIME_CONFIG = {}  # Reset
```

#### 6. Integration Tests

```python
class TestPipelineIntegration:
    
    @pytest.fixture
    def mock_registry(self):
        # Set up registry with mock agents
        pass
    
    def test_full_pipeline_execution(self, mock_registry):
        # Test complete pipeline flow
        pass
    
    def test_iteration_loop(self, mock_registry):
        # Test multiple iterations
        pass
    
    def test_rewrite_flow(self, mock_registry):
        # Test rewrite path
        pass
```

### Test Commands

```bash
# Run orchestrator tests
pytest test_orchestrator.py -v

# Run with coverage
pytest test_orchestrator.py --cov=core.orchestrator --cov-report=html

# Run specific test class
pytest test_orchestrator.py::TestTelemetry -v
```

---

## Recommendations and Improvements

### Critical Issues

#### 1. Global State Management

**Problem:** Multiple global variables (`CONFIG`, `RUNTIME_CONFIG`, `TELEMETRY_EVENTS`, `REGISTRY`) make testing and concurrent use difficult.

**Recommendation:** Encapsulate in a class:

```python
class PipelineContext:
    def __init__(self, config_path: str = None):
        self.config = self._load_config(config_path)
        self.runtime_config = self._init_runtime(self.config)
        self.telemetry_events = []
        self.registry = AgentRegistry()
    
    def register_agents(self):
        # Register agents using self.config
        pass
    
    def run(self, query: str) -> Dict[str, Any]:
        # Execute pipeline
        pass
```

#### 2. No Error Handling in Pipeline

**Problem:** Agent failures can crash the entire pipeline.

**Recommendation:** Add try-catch blocks:

```python
def agentic_once_with_metadata(query: str) -> Dict[str, Any]:
    try:
        # Router
        r_out = router.route(rin)
    except Exception as e:
        logger.error(f"Router failed: {e}")
        return {"error": str(e), "stage": "router"}
    
    # ... continue with error handling for each stage
```

---

### High Priority Improvements

#### 3. Logging and Observability

**Problem:** No structured logging beyond telemetry.

**Recommendation:** Add logging:

```python
import logging
logger = logging.getLogger(__name__)

def agentic_once_with_metadata(query: str) -> Dict[str, Any]:
    logger.info(f"Starting pipeline for query: {query[:100]}...")
    
    # After each stage
    logger.debug(f"Router completed: {r_out.router_profile.query_type}")
    logger.debug(f"Retrieved {len(re_out.results)} results")
    
    # On completion
    logger.info(f"Pipeline completed in {total_ms:.2f}ms, {len(iterations_data)} iterations")
```

#### 4. Timeout Handling

**Problem:** No timeout enforcement despite `max_time_seconds` in config.

**Recommendation:** Add timeout:

```python
import signal

def agentic_once_with_metadata(query: str) -> Dict[str, Any]:
    max_time = global_cfg.max_time_seconds
    
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Pipeline exceeded {max_time}s limit")
    
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(max_time)
    
    try:
        # Execute pipeline
        pass
    finally:
        signal.alarm(0)  # Cancel alarm
```

#### 5. Cancellation Support

**Problem:** No way to cancel a running pipeline.

**Recommendation:** Add cancellation token:

```python
class CancellationToken:
    def __init__(self):
        self.cancelled = False
    
    def cancel(self):
        self.cancelled = True
    
    def check(self):
        if self.cancelled:
            raise CancelledException("Pipeline cancelled")

def agentic_once_with_metadata(query: str, cancel_token: CancellationToken = None):
    for i in range(max_iters):
        if cancel_token:
            cancel_token.check()
        # ... iteration logic
```

---

### Medium Priority Improvements

#### 6. Caching Retrieval Results

**Problem:** Same query always retrieves fresh results.

**Recommendation:** Add caching:

```python
from functools import lru_cache

@lru_cache(maxsize=100)
def _cached_retrieval(query_hash: str, plan_hash: str):
    # Return cached results
    pass
```

#### 7. Async Support

**Problem:** Synchronous execution only.

**Recommendation:** Add async version:

```python
async def agentic_once_async(query: str) -> Dict[str, Any]:
    # Async agent calls where supported
    r_out = await router.route_async(rin)
    # ...
```

#### 8. Progress Callbacks

**Problem:** No progress feedback during execution.

**Recommendation:** Add callbacks:

```python
def agentic_once_with_metadata(
    query: str,
    progress_callback: Callable[[str, float], None] = None
) -> Dict[str, Any]:
    
    def report(stage: str, progress: float):
        if progress_callback:
            progress_callback(stage, progress)
    
    report("router", 0.1)
    r_out = router.route(rin)
    
    report("retrieval", 0.4)
    re_out = retriever.retrieve(rin2)
    # ...
```

---

### Low Priority / Future Enhancements

#### 9. Plugin System

**Recommendation:** Allow custom agents via plugins:

```python
def register_plugin_agents(plugin_paths: List[str]):
    for path in plugin_paths:
        module = importlib.import_module(path)
        for agent in module.agents:
            REGISTRY.register(agent)
```

#### 10. Configuration Validation

**Recommendation:** Validate config on load:

```python
from pydantic import BaseModel

class ConfigSchema(BaseModel):
    agentic: AgenticConfig
    llm: LLMConfig
    # ...

def _load_config(config_path: str) -> Dict[str, Any]:
    raw = yaml.safe_load(...)
    validated = ConfigSchema(**raw)
    return validated.dict()
```

#### 11. Pipeline Profiling

**Recommendation:** Add detailed profiling:

```python
class PipelineProfiler:
    def __init__(self):
        self.stage_times = {}
        self.memory_usage = {}
    
    def profile_stage(self, name: str):
        # Context manager for profiling
        pass
    
    def report(self) -> Dict[str, Any]:
        return {
            "total_time_ms": sum(self.stage_times.values()),
            "stages": self.stage_times,
            "memory": self.memory_usage,
        }
```

---

## Usage Examples

### Basic Usage

```python
from core.orchestrator import register_default_agents, agentic_once

# Register all agents
register_default_agents("config.fast.yaml")

# Run query
answer = agentic_once("What is hierarchical RAG and why is it useful?")
print(answer)
```

### With Full Metadata

```python
from core.orchestrator import register_default_agents, agentic_once_with_metadata

register_default_agents()

result = agentic_once_with_metadata("Explain the benefits of RAG")

print(f"Answer: {result['answer'].text}")
print(f"Iterations: {len(result['iterations'])}")
print(f"Citations: {len(result['citations'])}")
print(f"Coverage: {result['critic'].coverage_score}")
```

### Accessing Telemetry

```python
from core.orchestrator import TELEMETRY_EVENTS, clear_telemetry

# Run query
result = agentic_once_with_metadata("test query")

# Analyze telemetry
for event in TELEMETRY_EVENTS:
    print(f"{event.agent}: {event.event_type} - {event.timing.elapsed_ms:.2f}ms")

# Total time by agent
from collections import defaultdict
by_agent = defaultdict(float)
for event in TELEMETRY_EVENTS:
    by_agent[event.agent] += event.timing.elapsed_ms

for agent, total_ms in sorted(by_agent.items(), key=lambda x: -x[1]):
    print(f"{agent}: {total_ms:.2f}ms")

# Clear for next run
clear_telemetry()
```

### Custom Configuration

```python
import os

# Set custom config path
os.environ["AGENTIC_RAG_CONFIG"] = "/path/to/custom_config.yaml"

# Override iterations
os.environ["AGENTIC_MAX_ITERS"] = "5"
os.environ["AGENTIC_MIN_ITERS"] = "2"

# Register and run
register_default_agents()
result = agentic_once_with_metadata("complex query requiring multiple iterations")
```

### Programmatic Agent Access

```python
from core.orchestrator import register_default_agents, REGISTRY

register_default_agents()

# Access individual agents
router = REGISTRY.get("router")
generator = REGISTRY.get("generator")

# Use directly
from core.schemas import RouterInput, RouterConfig
inp = RouterInput(
    ctx=build_request_context(),
    user_query="test",
    history=[],
    config=RouterConfig(),
)
output = router.route(inp)
print(output.router_profile.query_type)
```

---

## Appendix

### Glossary

| Term | Definition |
|------|------------|
| **Orchestrator** | Central coordinator for pipeline execution |
| **Registry** | Agent registration and lookup system |
| **Telemetry** | Event logging and timing collection |
| **Iteration Loop** | Generate-critique-decide refinement cycle |

### Return Value Structure

```python
{
    "ctx": RequestContext,           # Request context
    "router": RouterOutput,          # Router result
    "decomposition": DecompositionOutput,  # Decomposition result
    "plan": Plan,                    # Execution plan
    "guardrail": GuardrailOutput,    # Guardrail result
    "prf_terms": List[str],          # PRF terms
    "qe_expansions": List[str],      # Query expansions
    "retrieval_results": List[RetrievalResult],  # Retrieved docs
    "answer": Answer,                # Final answer
    "citations": List[ContextSnippet],  # Citations
    "context_snippets": List[ContextSnippet],  # All snippets
    "critic": CriticFeedback,        # Critic evaluation
    "policy": PolicyOutput,          # Policy decision
    "postprocess": PostprocessOutput,  # Formatted output
    "iterations": List[Dict],        # Iteration history
    "retrieval_metrics": RetrievalMetrics,  # Metrics
}
```

### Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | Initial | Full agentic pipeline orchestration |

### References

- Radiant Repository: https://github.com/dshipley71/radiant
- Related Files: All agent implementations, `core/schemas.py`, `core/interfaces.py`

---

*Document generated for the Radiant RAG Pipeline project. For questions or contributions, please refer to the project repository.*
