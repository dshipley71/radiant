# Core Interfaces Documentation

Technical reference for the Radiant RAG pipeline agent architecture.

---

## Overview

The `core.interfaces` module defines abstract base classes for all agents in the Radiant RAG pipeline. These interfaces establish contracts that concrete implementations must fulfill, enabling a modular, pluggable architecture.

**Module Location:** `core/interfaces.py`

---

## Base Class

### BaseAgent

The root abstract class for all agents.

```python
class BaseAgent(ABC):
    """Base interface for all agents."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for this agent implementation."""

    @property
    def role(self) -> str:
        """Logical role used for registry lookups."""
        return getattr(self.__class__, "role", self.__class__.__name__)

    @abstractmethod
    def describe(self) -> str:
        """Short description of what this agent does."""
```

| Member | Type | Required | Description |
|--------|------|----------|-------------|
| `name` | Property | Yes | Human-readable agent name |
| `role` | Property | No | Logical role for registry (defaults to class attribute or class name) |
| `describe()` | Method | Yes | Agent description |

---

## Agent Interfaces

### Query Processing Agents

| Agent | Role | Method | Input | Output |
|-------|------|--------|-------|--------|
| `RouterAgent` | `router` | `route()` | `RouterInput` | `RouterOutput` |
| `DecompositionAgent` | `decomposition` | `decompose()` | `DecompositionInput` | `DecompositionOutput` |
| `TranslationAgent` | `translation` | `normalize()` | `TranslationInput` | `TranslationOutput` |
| `QEAgent` | `qe` | `expand()` | `QEInput` | `QEOutput` |

### Planning & Control Agents

| Agent | Role | Method | Input | Output |
|-------|------|--------|-------|--------|
| `PlannerAgent` | `planner` | `plan()` | `PlannerInput` | `PlannerOutput` |
| `GuardrailAgent` | `guardrail` | `validate_plan()` | `GuardrailInput` | `GuardrailOutput` |
| `PolicyAgent` | `policy` | `decide()` | `PolicyInput` | `PolicyOutput` |

### Retrieval Agents

| Agent | Role | Method | Input | Output |
|-------|------|--------|-------|--------|
| `PRFAgent` | `prf` | `compute()` | `PRFInput` | `PRFOutput` |
| `RetrieverAgent` | `retriever` | `retrieve()` | `RetrieverInput` | `RetrieverOutput` |
| `RerankAgent` | `rerank` | `rerank()` | `RerankInput` | `RerankOutput` |

### Generation Agents

| Agent | Role | Method | Input | Output |
|-------|------|--------|-------|--------|
| `GeneratorAgent` | `generator` | `generate()` | `GeneratorInput` | `GeneratorOutput` |
| `CriticAgent` | `critic` | `evaluate()` | `CriticInput` | `CriticOutput` |
| `QueryRewriteAgent` | `rewrite` | `rewrite()` | `QueryRewriteInput` | `QueryRewriteOutput` |
| `PostProcessorAgent` | `postprocess` | `format()` | `PostprocessInput` | `PostprocessOutput` |

### Infrastructure Agents

| Agent | Role | Method | Input | Output |
|-------|------|--------|-------|--------|
| `TelemetryAgent` | `telemetry` | `log_event()` | `TelemetryEvent` | `TelemetryOutput` |
| `ToolExecutionAgent` | `tools` | `execute()` | `ToolExecutionInput` | `ToolExecutionOutput` |
| `SafetyAgent` | `safety` | `check()` | `SafetyInput` | `SafetyOutput` |
| `IndexManagementAgent` | `index` | `manage()` | `IndexInput` | `IndexOutput` |

---

## Interface Definitions

### RouterAgent

Classify queries and set high-level processing toggles.

```python
class RouterAgent(BaseAgent):
    role = "router"

    def describe(self) -> str:
        return "RouterAgent: classify query and set high-level toggles."

    @abstractmethod
    def route(self, inp: RouterInput) -> RouterOutput:
        ...
```

### DecompositionAgent

Detect and split multi-part queries.

```python
class DecompositionAgent(BaseAgent):
    role = "decomposition"

    def describe(self) -> str:
        return "DecompositionAgent: detect multi-part queries."

    @abstractmethod
    def decompose(self, inp: DecompositionInput) -> DecompositionOutput:
        ...
```

### PlannerAgent

Build execution plans from router/decomposition outputs.

```python
class PlannerAgent(BaseAgent):
    role = "planner"

    def describe(self) -> str:
        return "PlannerAgent: build an execution plan from router/decomposition."

    @abstractmethod
    def plan(self, inp: PlannerInput) -> PlannerOutput:
        ...
```

### GuardrailAgent

Validate and adjust execution plans.

```python
class GuardrailAgent(BaseAgent):
    role = "guardrail"

    def describe(self) -> str:
        return "GuardrailAgent: validate and adjust the plan."

    @abstractmethod
    def validate_plan(self, inp: GuardrailInput) -> GuardrailOutput:
        ...
```

### RetrieverAgent

Perform hierarchical hybrid retrieval.

```python
class RetrieverAgent(BaseAgent):
    role = "retriever"

    def describe(self) -> str:
        return "RetrieverAgent: perform hierarchical hybrid retrieval."

    @abstractmethod
    def retrieve(self, inp: RetrieverInput) -> RetrieverOutput:
        ...
```

### RerankAgent

Rerank retrieval results with a cross-encoder.

```python
class RerankAgent(BaseAgent):
    role = "rerank"

    def describe(self) -> str:
        return "RerankAgent: rerank retrieval results with a cross-encoder."

    @abstractmethod
    def rerank(self, inp: RerankInput) -> RerankOutput:
        ...
```

### GeneratorAgent

Generate RAG answers from context snippets.

```python
class GeneratorAgent(BaseAgent):
    role = "generator"

    def describe(self) -> str:
        return "GeneratorAgent: generate a RAG answer from context snippets."

    @abstractmethod
    def generate(self, inp: GeneratorInput) -> GeneratorOutput:
        ...
```

### CriticAgent

Evaluate coverage and hallucination risk.

```python
class CriticAgent(BaseAgent):
    role = "critic"

    def describe(self) -> str:
        return "CriticAgent: evaluate coverage and hallucination risk."

    @abstractmethod
    def evaluate(self, inp: CriticInput) -> CriticOutput:
        ...
```

### PolicyAgent

Decide whether to continue, rewrite, or finalize.

```python
class PolicyAgent(BaseAgent):
    role = "policy"

    def describe(self) -> str:
        return "PolicyAgent: decide whether to continue, rewrite, or finalize."

    @abstractmethod
    def decide(self, inp: PolicyInput) -> PolicyOutput:
        ...
```

### QueryRewriteAgent

Refine queries based on critic feedback.

```python
class QueryRewriteAgent(BaseAgent):
    role = "rewrite"

    def describe(self) -> str:
        return "QueryRewriteAgent: refine the query based on critic feedback."

    @abstractmethod
    def rewrite(self, inp: QueryRewriteInput) -> QueryRewriteOutput:
        ...
```

### PostProcessorAgent

Format the final answer.

```python
class PostProcessorAgent(BaseAgent):
    role = "postprocess"

    def describe(self) -> str:
        return "PostProcessorAgent: format the final answer."

    @abstractmethod
    def format(self, inp: PostprocessInput) -> PostprocessOutput:
        ...
```

---

## Implementation Example

```python
from core.interfaces import RouterAgent
from core.schemas import RouterInput, RouterOutput, RouterProfile

class CustomRouterAgent(RouterAgent):
    """Custom router implementation."""
    
    @property
    def name(self) -> str:
        return "CustomRouterAgent"
    
    def route(self, inp: RouterInput) -> RouterOutput:
        # Classification logic
        profile = RouterProfile(
            query_type="lookup",
            use_qe=True,
            use_prf=False,
            use_rerank=True,
            expected_answer_style="paragraph",
            complexity_hint="medium",
        )
        return RouterOutput(router_profile=profile)
```

---

## Related Documentation

- [CoreSchemas_Documentation.md](CoreSchemas_Documentation.md) - Input/output data models
- [Orchestrator_Documentation.md](Orchestrator_Documentation.md) - Agent registration and pipeline execution
