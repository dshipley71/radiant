# Core Interfaces Documentation

## Technical Reference for the Radiant RAG Pipeline Agent Architecture

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture Design](#architecture-design)
3. [Base Classes](#base-classes)
4. [Agent Interfaces](#agent-interfaces)
5. [Interface Contracts](#interface-contracts)
6. [Implementation Guidelines](#implementation-guidelines)
7. [Testing Strategies](#testing-strategies)
8. [Recommendations and Improvements](#recommendations-and-improvements)
9. [Usage Examples](#usage-examples)

---

## Overview

### Purpose

The `core.interfaces` module defines the abstract base classes for all agents in the Radiant RAG pipeline. These interfaces establish contracts that concrete implementations must fulfill, enabling a modular, pluggable architecture.

### Key Responsibilities

- Define standardized interfaces for each agent role
- Establish input/output contracts via typed schemas
- Enable agent registration and discovery via roles
- Provide default implementations where appropriate

### Design Philosophy

The module follows the **Interface Segregation Principle** where each agent type has its own focused interface. This allows for independent implementation, testing, and replacement of individual pipeline components without affecting others.

---

## Architecture Design

### Agent Hierarchy

```
                         BaseAgent (ABC)
                              │
                              │ @abstractmethod: name, describe()
                              │ @property: role
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        │                     │                     │
   ┌────┴────┐          ┌────┴────┐          ┌────┴────┐
   │ Router  │          │Retriever│          │Generator│
   │  Agent  │          │  Agent  │          │  Agent  │
   └────┬────┘          └────┬────┘          └────┬────┘
        │                    │                    │
        │                    │                    │
   ┌────┴────┐          ┌────┴────┐          ┌────┴────┐
   │ Basic   │          │ Hybrid  │          │  LLM    │
   │ Router  │          │Retrieval│          │Generator│
   │ Agent   │          │  Agent  │          │  Agent  │
   └─────────┘          └─────────┘          └─────────┘
  (Concrete)            (Concrete)           (Concrete)
```

### Complete Agent Type Catalog

```
┌─────────────────────────────────────────────────────────────────┐
│                    Pipeline Agents                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Query Processing:                                              │
│    ├── RouterAgent         → Query classification               │
│    ├── DecompositionAgent  → Multi-part query handling          │
│    ├── TranslationAgent    → Language normalization             │
│    └── QEAgent             → Query expansion                    │
│                                                                 │
│  Planning & Control:                                            │
│    ├── PlannerAgent        → Execution planning                 │
│    ├── GuardrailAgent      → Plan validation                    │
│    └── PolicyAgent         → Decision control                   │
│                                                                 │
│  Retrieval:                                                     │
│    ├── PRFAgent            → Pseudo-relevance feedback          │
│    ├── RetrieverAgent      → Document retrieval                 │
│    └── RerankAgent         → Result reranking                   │
│                                                                 │
│  Generation:                                                    │
│    ├── GeneratorAgent      → Answer generation                  │
│    ├── CriticAgent         → Quality evaluation                 │
│    ├── QueryRewriteAgent   → Query refinement                   │
│    └── PostProcessorAgent  → Output formatting                  │
│                                                                 │
│  Infrastructure:                                                │
│    ├── TelemetryAgent      → Event logging                      │
│    ├── ToolExecutionAgent  → External tool execution            │
│    ├── SafetyAgent         → Safety checks                      │
│    └── IndexManagementAgent→ Index management                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Role Registry Pattern

```python
# Agent discovery via role attribute
agents = {
    "router": BasicRouterAgent(),
    "planner": BasicPlannerAgent(),
    "retriever": HybridRetrievalAgent(),
    # ...
}

def get_agent(role: str) -> BaseAgent:
    return agents[role]

# Usage
router = get_agent("router")
assert router.role == "router"
```

---

## Base Classes

### BaseAgent

The root abstract class for all agents.

```python
class BaseAgent(ABC):
    """Base interface for all agents."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for this agent implementation."""
        ...

    @property
    def role(self) -> str:
        """Logical role used for registry lookups."""
        return getattr(self.__class__, "role", self.__class__.__name__)

    @abstractmethod
    def describe(self) -> str:
        """Short description of what this agent does."""
        ...
```

### BaseAgent Members

| Member | Type | Required | Description |
|--------|------|----------|-------------|
| `name` | Property (abstract) | Yes | Human-readable agent name |
| `role` | Property | No | Logical role for registry (has default) |
| `describe()` | Method (abstract) | Yes | Agent description |

### Role Resolution Logic

```python
@property
def role(self) -> str:
    """
    Resolution order:
    1. Class attribute 'role' if defined
    2. Class name as fallback
    """
    return getattr(self.__class__, "role", self.__class__.__name__)
```

---

## Agent Interfaces

### Query Processing Agents

#### RouterAgent

**Purpose:** Classify queries and set high-level processing toggles.

```python
class RouterAgent(BaseAgent):
    role = "router"

    @abstractmethod
    def route(self, inp: RouterInput) -> RouterOutput:
        ...
```

| Method | Input | Output | Description |
|--------|-------|--------|-------------|
| `route()` | `RouterInput` | `RouterOutput` | Query classification |

---

#### DecompositionAgent

**Purpose:** Detect and split multi-part queries.

```python
class DecompositionAgent(BaseAgent):
    role = "decomposition"

    @abstractmethod
    def decompose(self, inp: DecompositionInput) -> DecompositionOutput:
        ...
```

| Method | Input | Output | Description |
|--------|-------|--------|-------------|
| `decompose()` | `DecompositionInput` | `DecompositionOutput` | Query splitting |

---

#### TranslationAgent

**Purpose:** Detect language and optionally translate text.

```python
class TranslationAgent(BaseAgent):
    role = "translation"

    @abstractmethod
    def normalize(self, inp: TranslationInput) -> TranslationOutput:
        ...
```

| Method | Input | Output | Description |
|--------|-------|--------|-------------|
| `normalize()` | `TranslationInput` | `TranslationOutput` | Language normalization |

---

#### QEAgent

**Purpose:** Perform LLM-based query expansion.

```python
class QEAgent(BaseAgent):
    role = "qe"

    @abstractmethod
    def expand(self, inp: QEInput) -> QEOutput:
        ...
```

| Method | Input | Output | Description |
|--------|-------|--------|-------------|
| `expand()` | `QEInput` | `QEOutput` | Query expansion |

---

### Planning & Control Agents

#### PlannerAgent

**Purpose:** Build execution plans from router/decomposition outputs.

```python
class PlannerAgent(BaseAgent):
    role = "planner"

    @abstractmethod
    def plan(self, inp: PlannerInput) -> PlannerOutput:
        ...
```

| Method | Input | Output | Description |
|--------|-------|--------|-------------|
| `plan()` | `PlannerInput` | `PlannerOutput` | Plan creation |

---

#### GuardrailAgent

**Purpose:** Validate and adjust execution plans.

```python
class GuardrailAgent(BaseAgent):
    role = "guardrail"

    @abstractmethod
    def validate_plan(self, inp: GuardrailInput) -> GuardrailOutput:
        ...
```

| Method | Input | Output | Description |
|--------|-------|--------|-------------|
| `validate_plan()` | `GuardrailInput` | `GuardrailOutput` | Plan validation |

---

#### PolicyAgent

**Purpose:** Decide whether to continue, rewrite, or finalize.

```python
class PolicyAgent(BaseAgent):
    role = "policy"

    @abstractmethod
    def decide(self, inp: PolicyInput) -> PolicyOutput:
        ...
```

| Method | Input | Output | Description |
|--------|-------|--------|-------------|
| `decide()` | `PolicyInput` | `PolicyOutput` | Flow decision |

---

### Retrieval Agents

#### PRFAgent

**Purpose:** Compute pseudo-relevance feedback for query augmentation.

```python
class PRFAgent(BaseAgent):
    role = "prf"

    @abstractmethod
    def compute(self, inp: PRFInput) -> PRFOutput:
        ...
```

| Method | Input | Output | Description |
|--------|-------|--------|-------------|
| `compute()` | `PRFInput` | `PRFOutput` | PRF computation |

---

#### RetrieverAgent

**Purpose:** Perform hierarchical hybrid retrieval.

```python
class RetrieverAgent(BaseAgent):
    role = "retriever"

    @abstractmethod
    def retrieve(self, inp: RetrieverInput) -> RetrieverOutput:
        ...
```

| Method | Input | Output | Description |
|--------|-------|--------|-------------|
| `retrieve()` | `RetrieverInput` | `RetrieverOutput` | Document retrieval |

---

#### RerankAgent

**Purpose:** Rerank retrieval results with a cross-encoder.

```python
class RerankAgent(BaseAgent):
    role = "rerank"

    @abstractmethod
    def rerank(self, inp: RerankInput) -> RerankOutput:
        ...
```

| Method | Input | Output | Description |
|--------|-------|--------|-------------|
| `rerank()` | `RerankInput` | `RerankOutput` | Result reranking |

---

### Generation Agents

#### GeneratorAgent

**Purpose:** Generate RAG answers from context snippets.

```python
class GeneratorAgent(BaseAgent):
    role = "generator"

    @abstractmethod
    def generate(self, inp: GeneratorInput) -> GeneratorOutput:
        ...
```

| Method | Input | Output | Description |
|--------|-------|--------|-------------|
| `generate()` | `GeneratorInput` | `GeneratorOutput` | Answer generation |

---

#### CriticAgent

**Purpose:** Evaluate answer coverage and hallucination risk.

```python
class CriticAgent(BaseAgent):
    role = "critic"

    @abstractmethod
    def evaluate(self, inp: CriticInput) -> CriticOutput:
        ...
```

| Method | Input | Output | Description |
|--------|-------|--------|-------------|
| `evaluate()` | `CriticInput` | `CriticOutput` | Quality evaluation |

---

#### QueryRewriteAgent

**Purpose:** Refine queries based on critic feedback.

```python
class QueryRewriteAgent(BaseAgent):
    role = "rewrite"

    @abstractmethod
    def rewrite(self, inp: QueryRewriteInput) -> QueryRewriteOutput:
        ...
```

| Method | Input | Output | Description |
|--------|-------|--------|-------------|
| `rewrite()` | `QueryRewriteInput` | `QueryRewriteOutput` | Query refinement |

---

#### PostProcessorAgent

**Purpose:** Format the final answer for output.

```python
class PostProcessorAgent(BaseAgent):
    role = "postprocess"

    @abstractmethod
    def format(self, inp: PostprocessInput) -> PostprocessOutput:
        ...
```

| Method | Input | Output | Description |
|--------|-------|--------|-------------|
| `format()` | `PostprocessInput` | `PostprocessOutput` | Output formatting |

---

### Infrastructure Agents

#### TelemetryAgent

**Purpose:** Log events and metrics.

```python
class TelemetryAgent(BaseAgent):
    role = "telemetry"

    @abstractmethod
    def log_event(self, event: TelemetryEvent) -> TelemetryOutput:
        ...
```

| Method | Input | Output | Description |
|--------|-------|--------|-------------|
| `log_event()` | `TelemetryEvent` | `TelemetryOutput` | Event logging |

---

#### ToolExecutionAgent

**Purpose:** Execute external tools with policy checks.

```python
class ToolExecutionAgent(BaseAgent):
    role = "tools"

    @abstractmethod
    def execute(self, inp: ToolExecutionInput) -> ToolExecutionOutput:
        ...
```

| Method | Input | Output | Description |
|--------|-------|--------|-------------|
| `execute()` | `ToolExecutionInput` | `ToolExecutionOutput` | Tool execution |

---

#### SafetyAgent

**Purpose:** Perform safety checks, redaction, or blocking.

```python
class SafetyAgent(BaseAgent):
    role = "safety"

    @abstractmethod
    def check(self, inp: SafetyInput) -> SafetyOutput:
        ...
```

| Method | Input | Output | Description |
|--------|-------|--------|-------------|
| `check()` | `SafetyInput` | `SafetyOutput` | Safety checking |

---

#### IndexManagementAgent

**Purpose:** Manage and report index state.

```python
class IndexManagementAgent(BaseAgent):
    role = "index"

    @abstractmethod
    def manage(self, inp: IndexInput) -> IndexOutput:
        ...
```

| Method | Input | Output | Description |
|--------|-------|--------|-------------|
| `manage()` | `IndexInput` | `IndexOutput` | Index management |

---

## Interface Contracts

### Complete Interface Reference

| Agent Type | Role | Method | Input Schema | Output Schema |
|------------|------|--------|--------------|---------------|
| `RouterAgent` | `router` | `route()` | `RouterInput` | `RouterOutput` |
| `DecompositionAgent` | `decomposition` | `decompose()` | `DecompositionInput` | `DecompositionOutput` |
| `PlannerAgent` | `planner` | `plan()` | `PlannerInput` | `PlannerOutput` |
| `GuardrailAgent` | `guardrail` | `validate_plan()` | `GuardrailInput` | `GuardrailOutput` |
| `TranslationAgent` | `translation` | `normalize()` | `TranslationInput` | `TranslationOutput` |
| `QEAgent` | `qe` | `expand()` | `QEInput` | `QEOutput` |
| `PRFAgent` | `prf` | `compute()` | `PRFInput` | `PRFOutput` |
| `RetrieverAgent` | `retriever` | `retrieve()` | `RetrieverInput` | `RetrieverOutput` |
| `RerankAgent` | `rerank` | `rerank()` | `RerankInput` | `RerankOutput` |
| `GeneratorAgent` | `generator` | `generate()` | `GeneratorInput` | `GeneratorOutput` |
| `CriticAgent` | `critic` | `evaluate()` | `CriticInput` | `CriticOutput` |
| `PolicyAgent` | `policy` | `decide()` | `PolicyInput` | `PolicyOutput` |
| `QueryRewriteAgent` | `rewrite` | `rewrite()` | `QueryRewriteInput` | `QueryRewriteOutput` |
| `PostProcessorAgent` | `postprocess` | `format()` | `PostprocessInput` | `PostprocessOutput` |
| `TelemetryAgent` | `telemetry` | `log_event()` | `TelemetryEvent` | `TelemetryOutput` |
| `ToolExecutionAgent` | `tools` | `execute()` | `ToolExecutionInput` | `ToolExecutionOutput` |
| `SafetyAgent` | `safety` | `check()` | `SafetyInput` | `SafetyOutput` |
| `IndexManagementAgent` | `index` | `manage()` | `IndexInput` | `IndexOutput` |

### Schema Import Reference

All schemas are imported from `core.schemas`:

```python
from .schemas import (
    CriticInput, CriticOutput,
    DecompositionInput, DecompositionOutput,
    GeneratorInput, GeneratorOutput,
    GuardrailInput, GuardrailOutput,
    IndexInput, IndexOutput,
    PlannerInput, PlannerOutput,
    PolicyInput, PolicyOutput,
    PRFInput, PRFOutput,
    QEInput, QEOutput,
    QueryRewriteInput, QueryRewriteOutput,
    RetrieverInput, RetrieverOutput,
    RerankInput, RerankOutput,
    RouterInput, RouterOutput,
    SafetyInput, SafetyOutput,
    TelemetryEvent, TelemetryOutput,
    ToolExecutionInput, ToolExecutionOutput,
    TranslationInput, TranslationOutput,
    PostprocessInput, PostprocessOutput,
)
```

---

## Implementation Guidelines

### Creating a New Agent

#### Step 1: Choose the Appropriate Base Class

```python
from core.interfaces import RetrieverAgent

class MyCustomRetriever(RetrieverAgent):
    """Custom retriever implementation."""
```

#### Step 2: Implement Required Abstract Methods

```python
class MyCustomRetriever(RetrieverAgent):
    
    @property
    def name(self) -> str:
        return "MyCustomRetriever"
    
    def describe(self) -> str:
        return "Custom retriever with special features."
    
    def retrieve(self, inp: RetrieverInput) -> RetrieverOutput:
        # Implementation here
        ...
```

#### Step 3: (Optional) Override Class Attributes

```python
class MyCustomRetriever(RetrieverAgent):
    role = "retriever"  # Inherited, but can override
    
    # Custom class attributes
    supports_batching = True
    max_batch_size = 100
```

### Implementation Checklist

- [ ] Inherit from appropriate base class
- [ ] Implement `name` property
- [ ] Implement `describe()` method
- [ ] Implement domain-specific abstract method
- [ ] Use correct input/output schemas
- [ ] Add type hints
- [ ] Handle errors gracefully
- [ ] Add logging where appropriate

### Common Patterns

#### Pattern 1: Configuration Injection

```python
class ConfigurableAgent(SomeAgent):
    def __init__(self, config: dict):
        self.config = config
        self._setup_from_config()
    
    def _setup_from_config(self):
        # Initialize from config
        pass
```

#### Pattern 2: Lazy Initialization

```python
class LazyAgent(SomeAgent):
    def __init__(self):
        self._model = None
    
    @property
    def model(self):
        if self._model is None:
            self._model = self._load_model()
        return self._model
```

#### Pattern 3: Graceful Degradation

```python
class RobustAgent(SomeAgent):
    def process(self, inp: SomeInput) -> SomeOutput:
        try:
            return self._process_impl(inp)
        except Exception as e:
            logger.error(f"Processing failed: {e}")
            return self._fallback_output(inp)
```

---

## Testing Strategies

### Testing Abstract Interface Compliance

```python
import pytest
from abc import ABC
from core.interfaces import BaseAgent, RouterAgent

class TestInterfaceCompliance:
    
    def test_base_agent_is_abstract(self):
        """BaseAgent cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseAgent()
    
    def test_router_agent_is_abstract(self):
        """RouterAgent cannot be instantiated directly."""
        with pytest.raises(TypeError):
            RouterAgent()
    
    def test_concrete_router_must_implement_route(self):
        """Concrete RouterAgent must implement route()."""
        
        class IncompleteRouter(RouterAgent):
            @property
            def name(self):
                return "Incomplete"
        
        with pytest.raises(TypeError):
            IncompleteRouter()
    
    def test_complete_implementation_instantiates(self):
        """Complete implementation can be instantiated."""
        
        class CompleteRouter(RouterAgent):
            @property
            def name(self):
                return "Complete"
            
            def route(self, inp):
                return RouterOutput(...)
        
        router = CompleteRouter()
        assert router.name == "Complete"
```

### Testing Role Resolution

```python
class TestRoleResolution:
    
    def test_class_attribute_role(self):
        """Role from class attribute."""
        
        class MyAgent(RouterAgent):
            role = "custom_role"
            
            @property
            def name(self):
                return "Test"
            
            def route(self, inp):
                pass
        
        agent = MyAgent()
        assert agent.role == "custom_role"
    
    def test_inherited_role(self):
        """Role inherited from base class."""
        
        class MyRouter(RouterAgent):
            @property
            def name(self):
                return "Test"
            
            def route(self, inp):
                pass
        
        agent = MyRouter()
        assert agent.role == "router"
    
    def test_fallback_to_class_name(self):
        """Role falls back to class name if not defined."""
        
        class CustomAgent(BaseAgent):
            @property
            def name(self):
                return "Test"
            
            def describe(self):
                return "Test"
        
        agent = CustomAgent()
        assert agent.role == "CustomAgent"
```

### Testing Schema Contracts

```python
from core.schemas import RouterInput, RouterOutput

class TestSchemaContracts:
    
    def test_router_input_output_types(self):
        """RouterAgent.route() accepts RouterInput, returns RouterOutput."""
        
        class TestRouter(RouterAgent):
            @property
            def name(self):
                return "Test"
            
            def route(self, inp: RouterInput) -> RouterOutput:
                assert isinstance(inp, RouterInput)
                return RouterOutput(...)
        
        router = TestRouter()
        inp = RouterInput(query="test")
        output = router.route(inp)
        
        assert isinstance(output, RouterOutput)
```

### Testing Default Implementations

```python
class TestDefaultImplementations:
    
    def test_describe_has_default(self):
        """Base classes provide default describe()."""
        
        class TestRouter(RouterAgent):
            @property
            def name(self):
                return "Test"
            
            def route(self, inp):
                pass
        
        router = TestRouter()
        description = router.describe()
        
        assert isinstance(description, str)
        assert len(description) > 0
```

### Test Commands

```bash
# Run interface tests
pytest test_interfaces.py -v

# Check abstract class compliance
pytest test_interfaces.py::TestInterfaceCompliance -v

# Run with type checking
mypy core/interfaces.py
```

---

## Recommendations and Improvements

### Critical Issues

#### 1. Inconsistent `describe()` Pattern

**Problem:** Some base classes have concrete `describe()`, others might expect override.

**Current State:**
```python
class RouterAgent(BaseAgent):
    def describe(self) -> str:
        return "RouterAgent: classify query and set high-level toggles."
```

**Observation:** Base classes provide default descriptions, but concrete implementations often override. This is actually good design (optional override), but should be documented.

---

### High Priority Improvements

#### 2. Add Common Lifecycle Methods

**Problem:** No standard initialization/shutdown hooks.

**Recommendation:** Add lifecycle methods to BaseAgent:

```python
class BaseAgent(ABC):
    def initialize(self) -> None:
        """Called before first use. Override for setup."""
        pass
    
    def shutdown(self) -> None:
        """Called on cleanup. Override for resource release."""
        pass
    
    def health_check(self) -> bool:
        """Return True if agent is operational."""
        return True
```

#### 3. Add Async Variants

**Problem:** No async support in interfaces.

**Recommendation:** Add async method variants:

```python
class RetrieverAgent(BaseAgent):
    @abstractmethod
    def retrieve(self, inp: RetrieverInput) -> RetrieverOutput:
        ...
    
    async def retrieve_async(self, inp: RetrieverInput) -> RetrieverOutput:
        """Async variant. Override for true async, default wraps sync."""
        return self.retrieve(inp)
```

#### 4. Add Batch Processing Interface

**Problem:** No standard batching interface.

**Recommendation:** Add batch methods:

```python
class RetrieverAgent(BaseAgent):
    @abstractmethod
    def retrieve(self, inp: RetrieverInput) -> RetrieverOutput:
        ...
    
    def retrieve_batch(self, inputs: List[RetrieverInput]) -> List[RetrieverOutput]:
        """Process multiple inputs. Override for optimized batching."""
        return [self.retrieve(inp) for inp in inputs]
    
    @property
    def supports_batching(self) -> bool:
        """Whether this agent supports optimized batching."""
        return False
```

---

### Medium Priority Improvements

#### 5. Add Metadata Properties

**Recommendation:** Standard metadata for agent discovery:

```python
class BaseAgent(ABC):
    @property
    def version(self) -> str:
        """Agent version string."""
        return "1.0.0"
    
    @property
    def capabilities(self) -> List[str]:
        """List of supported capabilities."""
        return []
    
    @property
    def config_schema(self) -> Optional[dict]:
        """JSON schema for configuration validation."""
        return None
```

#### 6. Add Input/Output Validation

**Recommendation:** Optional validation hooks:

```python
class BaseAgent(ABC):
    def validate_input(self, inp: Any) -> bool:
        """Validate input before processing. Override for custom validation."""
        return True
    
    def validate_output(self, output: Any) -> bool:
        """Validate output after processing. Override for custom validation."""
        return True
```

#### 7. Add Observability Hooks

**Recommendation:** Standard telemetry integration:

```python
class BaseAgent(ABC):
    def on_start(self, inp: Any) -> None:
        """Called before processing. Override for telemetry."""
        pass
    
    def on_complete(self, inp: Any, output: Any, elapsed_ms: float) -> None:
        """Called after processing. Override for telemetry."""
        pass
    
    def on_error(self, inp: Any, error: Exception) -> None:
        """Called on error. Override for error tracking."""
        pass
```

---

### Low Priority / Future Enhancements

#### 8. Add Configuration Protocol

**Recommendation:** Standardize configuration:

```python
from typing import Protocol

class Configurable(Protocol):
    def configure(self, config: dict) -> None:
        """Apply configuration."""
        ...
    
    def get_config(self) -> dict:
        """Return current configuration."""
        ...

class BaseAgent(ABC):
    # Optionally implement Configurable
    pass
```

#### 9. Add Agent Composition

**Recommendation:** Support agent chaining:

```python
class ComposableAgent(BaseAgent):
    def chain(self, other: BaseAgent) -> "ChainedAgent":
        """Chain this agent with another."""
        return ChainedAgent(self, other)
```

#### 10. Add Caching Protocol

**Recommendation:** Standard caching interface:

```python
class Cacheable(Protocol):
    def cache_key(self, inp: Any) -> str:
        """Generate cache key for input."""
        ...
    
    def is_cacheable(self, inp: Any) -> bool:
        """Whether this input can be cached."""
        ...
```

---

## Usage Examples

### Implementing a Custom Agent

```python
from core.interfaces import RouterAgent
from core.schemas import RouterInput, RouterOutput, RouterProfile

class CustomRouterAgent(RouterAgent):
    """Custom router with ML-based classification."""
    
    role = "router"  # Optional, inherited from RouterAgent
    
    def __init__(self, model_path: str):
        self.model = self._load_model(model_path)
    
    @property
    def name(self) -> str:
        return "CustomRouterAgent"
    
    def describe(self) -> str:
        return "ML-based router using custom classification model."
    
    def route(self, inp: RouterInput) -> RouterOutput:
        # ML-based classification
        prediction = self.model.predict(inp.query)
        
        return RouterOutput(
            profile=RouterProfile(
                query_type=prediction["type"],
                complexity=prediction["complexity"],
                use_qe=prediction["needs_expansion"],
                use_prf=prediction["needs_prf"],
                use_rerank=True,
            )
        )
    
    def _load_model(self, path: str):
        # Load ML model
        pass
```

### Agent Registry

```python
from typing import Dict, Type
from core.interfaces import BaseAgent

class AgentRegistry:
    """Registry for agent implementations."""
    
    _agents: Dict[str, Type[BaseAgent]] = {}
    
    @classmethod
    def register(cls, agent_class: Type[BaseAgent]) -> Type[BaseAgent]:
        """Decorator to register an agent class."""
        role = getattr(agent_class, "role", agent_class.__name__)
        cls._agents[role] = agent_class
        return agent_class
    
    @classmethod
    def get(cls, role: str) -> Type[BaseAgent]:
        """Get agent class by role."""
        return cls._agents[role]
    
    @classmethod
    def create(cls, role: str, **kwargs) -> BaseAgent:
        """Create agent instance by role."""
        agent_class = cls.get(role)
        return agent_class(**kwargs)

# Usage
@AgentRegistry.register
class MyRouterAgent(RouterAgent):
    ...

router = AgentRegistry.create("router", config=config)
```

### Type Checking with Interfaces

```python
from typing import List
from core.interfaces import RetrieverAgent, RerankAgent
from core.schemas import RetrieverInput, RetrieverOutput, RerankInput

def create_pipeline(
    retriever: RetrieverAgent,
    reranker: RerankAgent,
) -> None:
    """Type-safe pipeline creation."""
    
    # Type checker ensures correct types
    def process(query: str) -> List[dict]:
        retriever_output = retriever.retrieve(
            RetrieverInput(query=query, ...)
        )
        
        rerank_output = reranker.rerank(
            RerankInput(query=query, results=retriever_output.results)
        )
        
        return rerank_output.results
```

### Testing with Mock Agents

```python
from unittest.mock import Mock
from core.interfaces import RouterAgent
from core.schemas import RouterInput, RouterOutput, RouterProfile

def test_pipeline_with_mock_router():
    """Test pipeline with mocked router."""
    
    # Create mock that satisfies RouterAgent interface
    mock_router = Mock(spec=RouterAgent)
    mock_router.name = "MockRouter"
    mock_router.role = "router"
    mock_router.route.return_value = RouterOutput(
        profile=RouterProfile(
            query_type="lookup",
            complexity="low",
            use_qe=False,
            use_prf=False,
            use_rerank=True,
        )
    )
    
    # Use in pipeline
    inp = RouterInput(query="test query")
    output = mock_router.route(inp)
    
    assert output.profile.query_type == "lookup"
```

---

## Appendix

### Glossary

| Term | Definition |
|------|------------|
| **Abstract Base Class (ABC)** | Class that cannot be instantiated, defines interface |
| **Interface** | Contract defining methods a class must implement |
| **Role** | Logical identifier for agent type in registry |
| **Schema** | Data structure for agent input/output |

### Agent Role Reference

| Role | Agent Type | Primary Method |
|------|------------|----------------|
| `router` | RouterAgent | `route()` |
| `decomposition` | DecompositionAgent | `decompose()` |
| `planner` | PlannerAgent | `plan()` |
| `guardrail` | GuardrailAgent | `validate_plan()` |
| `translation` | TranslationAgent | `normalize()` |
| `qe` | QEAgent | `expand()` |
| `prf` | PRFAgent | `compute()` |
| `retriever` | RetrieverAgent | `retrieve()` |
| `rerank` | RerankAgent | `rerank()` |
| `generator` | GeneratorAgent | `generate()` |
| `critic` | CriticAgent | `evaluate()` |
| `policy` | PolicyAgent | `decide()` |
| `rewrite` | QueryRewriteAgent | `rewrite()` |
| `postprocess` | PostProcessorAgent | `format()` |
| `telemetry` | TelemetryAgent | `log_event()` |
| `tools` | ToolExecutionAgent | `execute()` |
| `safety` | SafetyAgent | `check()` |
| `index` | IndexManagementAgent | `manage()` |

### Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | Initial | 18 agent interfaces defined |

### References

- Radiant Repository: https://github.com/dshipley71/radiant
- Related Files: `core/schemas.py`, all agent implementations

---

*Document generated for the Radiant RAG Pipeline project. For questions or contributions, please refer to the project repository.*
