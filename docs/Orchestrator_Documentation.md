# Orchestrator Documentation

Technical reference for the Radiant RAG pipeline coordinator.

---

## Overview

The `orchestrator.py` module is the central coordinator for the Radiant RAG pipeline. It manages agent registration, configuration loading, request context creation, and the main agentic iteration loop.

**Module Location:** `core/orchestrator.py`

---

## Global State

| Variable | Type | Purpose |
|----------|------|---------|
| `CONFIG` | `Dict[str, Any]` | Cached configuration from YAML |
| `RUNTIME_CONFIG` | `Dict[str, Any]` | Runtime settings |
| `TELEMETRY_EVENTS` | `List[TelemetryEvent]` | Global telemetry buffer |
| `REGISTRY` | `AgentRegistry` | Singleton agent registry |

---

## AgentRegistry

```python
class AgentRegistry:
    """Minimal global registry: agents registered by .role attribute."""
    
    def register(self, agent: BaseAgent) -> None:
        """Register an agent by its role."""
    
    def get(self, role: str) -> BaseAgent:
        """Get agent by role name."""
```

### Registered Agents

| Role | Agent Class |
|------|-------------|
| `router` | `BasicRouterAgent` |
| `decomposition` | `BasicDecompositionAgent` |
| `planner` | `BasicPlannerAgent` |
| `guardrail` | `BasicGuardrailAgent` |
| `critic` | `BasicCriticAgent` |
| `policy` | `BasicPolicyAgent` |
| `telemetry` | `BasicTelemetryAgent` |
| `prf` | `BasicPRFAgent` |
| `rerank` | `BasicRerankAgent` |
| `postprocess` | `BasicPostProcessorAgent` |
| `qe` | `LLMQEAgent` |
| `rewrite` | `LLMQueryRewriteAgent` |
| `generator` | `LLMGeneratorAgent` |

---

## Pipeline Execution

### Main Entry Point

```python
def agentic_once_with_metadata(
    user_query: str,
    history: Optional[List[Message]] = None,
    config_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute full RAG pipeline and return results with metadata."""
```

**Returns:**
```python
{
    "answer": Answer,
    "citations": List[Citation],
    "final_text": str,
    "metadata": PostprocessMetadata,
    "telemetry": List[TelemetryEvent],
}
```

---

## Pipeline Phases

### Phase 1: Query Processing

```
User Query → Router → Decomposition → Planner → Guardrail → Plan
```

1. **Router**: Classify query type, set feature toggles
2. **Decomposition**: Split multi-part queries
3. **Planner**: Create execution plan
4. **Guardrail**: Validate and adjust plan

### Phase 2: Retrieval Enhancement

```
Query → PRF → QE → Enhanced Query
```

1. **PRF**: Compute pseudo-relevance feedback terms
2. **QE**: Generate query expansion variants

### Phase 3: Document Retrieval

```
Enhanced Query → Retriever → Reranker → Ranked Results
```

1. **Retriever**: Hybrid dense+sparse retrieval
2. **Reranker**: Cross-encoder reranking

### Phase 4: Iteration Loop

```
Results → Generator → Critic → Policy → [FINALIZE|REWRITE|CONTINUE]
                                              ↓
                                         Rewriter → Phase 2
```

1. **Generator**: Generate answer from context
2. **Critic**: Evaluate answer quality
3. **Policy**: Decide next action
4. **Rewriter**: Refine query (if REWRITE)

### Phase 5: Finalization

```
Answer → PostProcessor → Final Output
```

---

## Telemetry Events

| Event Type | Agent | Description |
|------------|-------|-------------|
| `router.decided` | Router | Query classification complete |
| `decomposition.done` | Decomposition | Query split complete |
| `planner.plan` | Planner | Plan created |
| `guardrail.check` | Guardrail | Plan validated |
| `prf.compute` | PRF | Expansion terms extracted |
| `qe.expand` | QE | Query variants generated |
| `retriever.results` | Retriever | Documents retrieved |
| `rerank.rerank` | Reranker | Candidates reranked |
| `generator.output` | Generator | Answer generated |
| `critic.evaluate` | Critic | Answer evaluated |
| `policy.decision` | Policy | Decision made |
| `rewrite.rewrite` | Rewriter | Query rewritten |
| `postprocess.format` | PostProcessor | Output formatted |

---

## Configuration

### Loading

```python
def _load_config(config_path: Optional[str]) -> Dict[str, Any]:
    """Load YAML configuration (config.fast.yaml by default)."""
```

Resolution order:
1. Explicit `config_path` parameter
2. `AGENTIC_RAG_CONFIG` environment variable
3. Default: `../config.fast.yaml`

### Environment Variables

| Variable | Purpose |
|----------|---------|
| `AGENTIC_RAG_CONFIG` | Config file path |
| `AGENTIC_MAX_ITERS` | Max iteration override |
| `AGENTIC_MIN_ITERS` | Min iteration override |

---

## Helper Functions

### Context Building

```python
def build_context_snippets_from_results(
    results: List[RetrievalResult]
) -> List[ContextSnippet]:
    """Convert retrieval results to context snippets."""

def build_retrieval_metrics(
    results: List[RetrievalResult]
) -> RetrievalMetrics:
    """Compute retrieval statistics."""

def build_request_context(
    runtime: RuntimeContext
) -> RequestContext:
    """Create request context with unique IDs."""
```

---

## Usage Example

```python
from core.orchestrator import register_default_agents, agentic_once_with_metadata

# Initialize pipeline
register_default_agents(config_path="config.fast.yaml")

# Run query
result = agentic_once_with_metadata("What is hierarchical RAG?")

print(result["answer"].text)
print(f"Citations: {len(result['citations'])}")
print(f"Events: {len(result['telemetry'])}")
```

---

## Related Documentation

- [CoreInterfaces_Documentation.md](CoreInterfaces_Documentation.md) - Agent interfaces
- [CoreSchemas_Documentation.md](CoreSchemas_Documentation.md) - Data models
- Agent documentation files - Individual agent details
