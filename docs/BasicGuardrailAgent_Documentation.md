# BasicGuardrailAgent Documentation

## Technical Reference for the Radiant RAG Pipeline Plan Validation

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture Context](#architecture-context)
3. [Class Structure](#class-structure)
4. [Core Functionality](#core-functionality)
5. [Validation Rules](#validation-rules)
6. [Data Flow](#data-flow)
7. [Testing Strategies](#testing-strategies)
8. [Recommendations and Improvements](#recommendations-and-improvements)
9. [Usage Examples](#usage-examples)

---

## Overview

### Purpose

The `BasicGuardrailAgent` is a validation and normalization component within the Radiant RAG pipeline. It ensures that execution plans conform to safe operational limits before being executed by downstream agents, preventing resource exhaustion, invalid configurations, and potential system failures.

### Key Responsibilities

- Validate plan parameters against defined limits
- Normalize out-of-range values to safe defaults
- Track all adjustments made for transparency
- Report validation status (ok/adjusted)

### Design Philosophy

The guardrail follows a **defensive normalization** approach rather than strict rejection. Invalid values are corrected to safe defaults rather than failing the pipeline, ensuring graceful degradation while maintaining audit trails of all changes.

---

## Architecture Context

### Position in the Radiant Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    Plan (from Planner)                          │
│  top_k: -5  |  rerank_top_k: 500  |  max_iters: 0              │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    BasicGuardrailAgent                          │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Validation Checks:                                     │   │
│  │  ✗ top_k <= 0       → Set to 5                         │   │
│  │  ✗ rerank_top_k > 200 → Cap to 200                     │   │
│  │  ✗ max_iters < 1    → Set to 1                         │   │
│  ├─────────────────────────────────────────────────────────┤   │
│  │  Output:                                                │   │
│  │  • status: "adjusted"                                   │   │
│  │  • messages: ["top_k <= 0; set to 5.", ...]            │   │
│  │  • plan: (corrected values)                             │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Validated Plan                               │
│  top_k: 5  |  rerank_top_k: 200  |  max_iters: 1               │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
              Downstream Agents (Retriever, Rewriter, etc.)
```

### Related Components

| Component | Relationship |
|-----------|--------------|
| `GuardrailAgent` | Abstract base class (from `core.interfaces`) |
| `GuardrailInput` | Input schema containing the Plan to validate |
| `GuardrailOutput` | Output schema with status, adjusted plan, and messages |
| `Plan` | Execution plan data structure being validated |
| `PlanIterations` | Nested structure containing iteration limits |

---

## Class Structure

### Inheritance

```python
class BasicGuardrailAgent(GuardrailAgent):
    """Basic guardrails for Plan sanity (limits & normalization)."""
```

### Class Attributes

| Attribute | Type | Value | Description |
|-----------|------|-------|-------------|
| `role` | `str` | `"guardrail"` | Identifies the agent's role in the pipeline |

### Properties

| Property | Return Type | Description |
|----------|-------------|-------------|
| `name` | `str` | Returns `"BasicGuardrailAgent"` |

### Methods

| Method | Visibility | Purpose |
|--------|------------|---------|
| `describe()` | Public | Returns human-readable description |
| `validate_plan(inp)` | Public | Main entry point for plan validation |

---

## Core Functionality

### The `validate_plan()` Method

This is the primary entry point that performs all validation and normalization.

**Signature:**
```python
def validate_plan(self, inp: GuardrailInput) -> GuardrailOutput
```

**Parameters:**
- `inp` (`GuardrailInput`): Contains the `Plan` object to validate

**Returns:**
- `GuardrailOutput`: Contains status, adjusted plan, and messages

**Processing Steps:**

1. **Deep Copy Plan**
   - Create an independent copy to avoid mutating the original
   - Uses Pydantic's `model_copy(deep=True)`

2. **Apply Validation Rules**
   - Check each parameter against defined limits
   - Correct invalid values to safe defaults
   - Record each adjustment in messages list

3. **Determine Status**
   - `"ok"` if no adjustments were needed
   - `"adjusted"` if any values were corrected

4. **Return Output**
   - Package corrected plan, status, and messages

---

## Validation Rules

### Rule Summary Table

| Parameter | Condition | Action | Default |
|-----------|-----------|--------|---------|
| `top_k` | `<= 0` | Set to default | `5` |
| `top_k` | `> 100` | Cap to maximum | `100` |
| `rerank_top_k` | `<= 0` | Set to `top_k` | Dynamic |
| `rerank_top_k` | `> 200` | Cap to maximum | `200` |
| `iterations.max_iters` | `< 1` | Set to minimum | `1` |

### Detailed Rule Specifications

#### Rule 1: top_k Minimum

```python
if adjusted.top_k <= 0:
    adjusted.top_k = 5
    messages.append("top_k <= 0; set to 5.")
```

**Rationale:** Retrieving zero or negative documents is invalid. Default of 5 provides a reasonable baseline.

**Boundary Values:**
| Input | Output | Message |
|-------|--------|---------|
| `-10` | `5` | Yes |
| `0` | `5` | Yes |
| `1` | `1` | No |

#### Rule 2: top_k Maximum

```python
if adjusted.top_k > 100:
    adjusted.top_k = 100
    messages.append("top_k > 100; capped to 100.")
```

**Rationale:** Retrieving excessive documents impacts performance and memory. 100 is a practical upper bound.

**Boundary Values:**
| Input | Output | Message |
|-------|--------|---------|
| `100` | `100` | No |
| `101` | `100` | Yes |
| `1000` | `100` | Yes |

#### Rule 3: rerank_top_k Minimum

```python
if adjusted.rerank_top_k <= 0:
    adjusted.rerank_top_k = adjusted.top_k
    messages.append("rerank_top_k <= 0; set to top_k.")
```

**Rationale:** Invalid rerank count defaults to top_k, ensuring all retrieved documents can be considered for reranking.

**Boundary Values:**
| Input (rerank) | top_k | Output | Message |
|----------------|-------|--------|---------|
| `-5` | `10` | `10` | Yes |
| `0` | `20` | `20` | Yes |
| `1` | `10` | `1` | No |

#### Rule 4: rerank_top_k Maximum

```python
if adjusted.rerank_top_k > 200:
    adjusted.rerank_top_k = 200
    messages.append("rerank_top_k > 200; capped to 200.")
```

**Rationale:** Cross-encoder reranking is computationally expensive. 200 is a practical limit for real-time systems.

**Boundary Values:**
| Input | Output | Message |
|-------|--------|---------|
| `200` | `200` | No |
| `201` | `200` | Yes |
| `500` | `200` | Yes |

#### Rule 5: max_iters Minimum

```python
if adjusted.iterations.max_iters < 1:
    adjusted.iterations.max_iters = 1
    messages.append("max_iters < 1; set to 1.")
```

**Rationale:** At least one iteration is required for the pipeline to produce results.

**Boundary Values:**
| Input | Output | Message |
|-------|--------|---------|
| `-1` | `1` | Yes |
| `0` | `1` | Yes |
| `1` | `1` | No |

### Rule Application Order

The rules are applied sequentially, which can affect outcomes:

```
1. top_k minimum check (sets to 5 if <= 0)
2. top_k maximum check (caps to 100 if > 100)
3. rerank_top_k minimum check (uses current top_k value)
4. rerank_top_k maximum check (caps to 200)
5. max_iters minimum check
```

**Important:** Rule 3 uses the **already-adjusted** `top_k` value, not the original.

---

## Data Flow

### Input Schema: `GuardrailInput`

```python
@dataclass
class GuardrailInput:
    plan: Plan  # Execution plan to validate
```

### Output Schema: `GuardrailOutput`

```python
@dataclass
class GuardrailOutput:
    status: str           # "ok" or "adjusted"
    plan: Plan            # Validated/corrected plan
    messages: List[str]   # List of adjustment messages
```

### Status Values

| Status | Meaning |
|--------|---------|
| `"ok"` | All parameters within valid ranges; no changes made |
| `"adjusted"` | One or more parameters were corrected |

---

## Testing Strategies

### Unit Tests

#### 1. No Adjustment Tests (Valid Plans)

```python
import pytest
from unittest.mock import Mock
from guardrail_basic_agent import BasicGuardrailAgent
from core.schemas import GuardrailInput, Plan, PlanIterations

@pytest.fixture
def agent():
    return BasicGuardrailAgent()

@pytest.fixture
def valid_plan():
    iterations = Mock()
    iterations.max_iters = 3
    
    plan = Mock()
    plan.top_k = 10
    plan.rerank_top_k = 5
    plan.iterations = iterations
    plan.model_copy = lambda deep=False: create_plan_copy(plan)
    return plan

def create_plan_copy(original):
    copy = Mock()
    copy.top_k = original.top_k
    copy.rerank_top_k = original.rerank_top_k
    copy.iterations = Mock()
    copy.iterations.max_iters = original.iterations.max_iters
    return copy

class TestValidPlans:
    
    def test_valid_plan_no_changes(self, agent, valid_plan):
        inp = GuardrailInput(plan=valid_plan)
        output = agent.validate_plan(inp)
        
        assert output.status == "ok"
        assert output.messages == []
        assert output.plan.top_k == 10
        assert output.plan.rerank_top_k == 5
    
    def test_boundary_valid_top_k(self, agent):
        plan = create_mock_plan(top_k=1, rerank_top_k=1, max_iters=1)
        inp = GuardrailInput(plan=plan)
        output = agent.validate_plan(inp)
        
        assert output.status == "ok"
        assert output.plan.top_k == 1
    
    def test_boundary_max_top_k(self, agent):
        plan = create_mock_plan(top_k=100, rerank_top_k=100, max_iters=1)
        inp = GuardrailInput(plan=plan)
        output = agent.validate_plan(inp)
        
        assert output.status == "ok"
        assert output.plan.top_k == 100
    
    def test_boundary_max_rerank_top_k(self, agent):
        plan = create_mock_plan(top_k=100, rerank_top_k=200, max_iters=1)
        inp = GuardrailInput(plan=plan)
        output = agent.validate_plan(inp)
        
        assert output.status == "ok"
        assert output.plan.rerank_top_k == 200
```

#### 2. top_k Validation Tests

```python
class TestTopKValidation:
    
    def test_negative_top_k_set_to_default(self, agent):
        plan = create_mock_plan(top_k=-10, rerank_top_k=5, max_iters=1)
        inp = GuardrailInput(plan=plan)
        output = agent.validate_plan(inp)
        
        assert output.status == "adjusted"
        assert output.plan.top_k == 5
        assert "top_k <= 0; set to 5." in output.messages
    
    def test_zero_top_k_set_to_default(self, agent):
        plan = create_mock_plan(top_k=0, rerank_top_k=5, max_iters=1)
        inp = GuardrailInput(plan=plan)
        output = agent.validate_plan(inp)
        
        assert output.plan.top_k == 5
        assert "top_k <= 0; set to 5." in output.messages
    
    def test_excessive_top_k_capped(self, agent):
        plan = create_mock_plan(top_k=500, rerank_top_k=5, max_iters=1)
        inp = GuardrailInput(plan=plan)
        output = agent.validate_plan(inp)
        
        assert output.status == "adjusted"
        assert output.plan.top_k == 100
        assert "top_k > 100; capped to 100." in output.messages
    
    def test_top_k_101_capped(self, agent):
        plan = create_mock_plan(top_k=101, rerank_top_k=5, max_iters=1)
        inp = GuardrailInput(plan=plan)
        output = agent.validate_plan(inp)
        
        assert output.plan.top_k == 100
```

#### 3. rerank_top_k Validation Tests

```python
class TestRerankTopKValidation:
    
    def test_negative_rerank_top_k_set_to_top_k(self, agent):
        plan = create_mock_plan(top_k=20, rerank_top_k=-5, max_iters=1)
        inp = GuardrailInput(plan=plan)
        output = agent.validate_plan(inp)
        
        assert output.status == "adjusted"
        assert output.plan.rerank_top_k == 20  # Set to top_k
        assert "rerank_top_k <= 0; set to top_k." in output.messages
    
    def test_zero_rerank_top_k_set_to_top_k(self, agent):
        plan = create_mock_plan(top_k=15, rerank_top_k=0, max_iters=1)
        inp = GuardrailInput(plan=plan)
        output = agent.validate_plan(inp)
        
        assert output.plan.rerank_top_k == 15
    
    def test_excessive_rerank_top_k_capped(self, agent):
        plan = create_mock_plan(top_k=50, rerank_top_k=500, max_iters=1)
        inp = GuardrailInput(plan=plan)
        output = agent.validate_plan(inp)
        
        assert output.status == "adjusted"
        assert output.plan.rerank_top_k == 200
        assert "rerank_top_k > 200; capped to 200." in output.messages
    
    def test_rerank_top_k_201_capped(self, agent):
        plan = create_mock_plan(top_k=50, rerank_top_k=201, max_iters=1)
        inp = GuardrailInput(plan=plan)
        output = agent.validate_plan(inp)
        
        assert output.plan.rerank_top_k == 200
```

#### 4. max_iters Validation Tests

```python
class TestMaxItersValidation:
    
    def test_negative_max_iters_set_to_one(self, agent):
        plan = create_mock_plan(top_k=10, rerank_top_k=5, max_iters=-1)
        inp = GuardrailInput(plan=plan)
        output = agent.validate_plan(inp)
        
        assert output.status == "adjusted"
        assert output.plan.iterations.max_iters == 1
        assert "max_iters < 1; set to 1." in output.messages
    
    def test_zero_max_iters_set_to_one(self, agent):
        plan = create_mock_plan(top_k=10, rerank_top_k=5, max_iters=0)
        inp = GuardrailInput(plan=plan)
        output = agent.validate_plan(inp)
        
        assert output.plan.iterations.max_iters == 1
        assert "max_iters < 1; set to 1." in output.messages
    
    def test_valid_max_iters_unchanged(self, agent):
        plan = create_mock_plan(top_k=10, rerank_top_k=5, max_iters=5)
        inp = GuardrailInput(plan=plan)
        output = agent.validate_plan(inp)
        
        assert output.plan.iterations.max_iters == 5
```

#### 5. Multiple Adjustment Tests

```python
class TestMultipleAdjustments:
    
    def test_all_parameters_invalid(self, agent):
        plan = create_mock_plan(top_k=-1, rerank_top_k=-1, max_iters=0)
        inp = GuardrailInput(plan=plan)
        output = agent.validate_plan(inp)
        
        assert output.status == "adjusted"
        assert output.plan.top_k == 5
        assert output.plan.rerank_top_k == 5  # Set to adjusted top_k
        assert output.plan.iterations.max_iters == 1
        assert len(output.messages) == 3
    
    def test_top_k_and_rerank_both_excessive(self, agent):
        plan = create_mock_plan(top_k=500, rerank_top_k=500, max_iters=1)
        inp = GuardrailInput(plan=plan)
        output = agent.validate_plan(inp)
        
        assert output.plan.top_k == 100
        assert output.plan.rerank_top_k == 200
        assert len(output.messages) == 2
    
    def test_cascading_adjustment(self, agent):
        """Test that rerank_top_k uses already-adjusted top_k."""
        plan = create_mock_plan(top_k=-1, rerank_top_k=-1, max_iters=1)
        inp = GuardrailInput(plan=plan)
        output = agent.validate_plan(inp)
        
        # top_k becomes 5, then rerank_top_k should also become 5
        assert output.plan.top_k == 5
        assert output.plan.rerank_top_k == 5
```

#### 6. Immutability Tests

```python
class TestImmutability:
    
    def test_original_plan_not_modified(self, agent):
        original_top_k = -10
        plan = create_mock_plan(top_k=original_top_k, rerank_top_k=5, max_iters=1)
        inp = GuardrailInput(plan=plan)
        
        output = agent.validate_plan(inp)
        
        # Original should be unchanged
        assert inp.plan.top_k == original_top_k
        # Output should have adjusted value
        assert output.plan.top_k == 5
```

#### 7. Agent Interface Tests

```python
class TestAgentInterface:
    
    def test_name_property(self, agent):
        assert agent.name == "BasicGuardrailAgent"
    
    def test_describe_method(self, agent):
        description = agent.describe()
        assert isinstance(description, str)
        assert "guardrail" in description.lower()
    
    def test_role_attribute(self, agent):
        assert agent.role == "guardrail"
```

### Helper Function for Tests

```python
def create_mock_plan(top_k: int, rerank_top_k: int, max_iters: int):
    """Create a mock Plan object for testing."""
    iterations = Mock()
    iterations.max_iters = max_iters
    
    plan = Mock()
    plan.top_k = top_k
    plan.rerank_top_k = rerank_top_k
    plan.iterations = iterations
    
    def model_copy(deep=False):
        copy_iterations = Mock()
        copy_iterations.max_iters = iterations.max_iters
        
        copy = Mock()
        copy.top_k = plan.top_k
        copy.rerank_top_k = plan.rerank_top_k
        copy.iterations = copy_iterations
        return copy
    
    plan.model_copy = model_copy
    return plan
```

### Test Commands

```bash
# Run all guardrail tests
pytest test_guardrail_basic_agent.py -v

# Run with coverage
pytest test_guardrail_basic_agent.py --cov=guardrail_basic_agent --cov-report=html

# Run specific test class
pytest test_guardrail_basic_agent.py::TestMultipleAdjustments -v
```

---

## Recommendations and Improvements

### Critical Issues

#### 1. Missing max_rewrites Validation

**Problem:** The `iterations.max_rewrites` field is not validated, potentially allowing negative or excessive values.

**Recommendation:** Add validation for max_rewrites:

```python
if adjusted.iterations.max_rewrites < 0:
    adjusted.iterations.max_rewrites = 0
    messages.append("max_rewrites < 0; set to 0.")

if adjusted.iterations.max_rewrites > 10:
    adjusted.iterations.max_rewrites = 10
    messages.append("max_rewrites > 10; capped to 10.")
```

#### 2. Missing max_iters Upper Bound

**Problem:** No maximum limit on `max_iters`, allowing potentially infinite loops.

**Recommendation:** Add upper bound:

```python
MAX_ITERS_LIMIT = 20

if adjusted.iterations.max_iters > MAX_ITERS_LIMIT:
    adjusted.iterations.max_iters = MAX_ITERS_LIMIT
    messages.append(f"max_iters > {MAX_ITERS_LIMIT}; capped.")
```

---

### High Priority Improvements

#### 3. Configurable Limits

**Problem:** All limits are hardcoded magic numbers.

**Recommendation:** Make limits configurable:

```python
from dataclasses import dataclass

@dataclass
class GuardrailLimits:
    top_k_min: int = 1
    top_k_max: int = 100
    top_k_default: int = 5
    rerank_top_k_max: int = 200
    max_iters_min: int = 1
    max_iters_max: int = 20
    max_rewrites_min: int = 0
    max_rewrites_max: int = 10

class BasicGuardrailAgent(GuardrailAgent):
    def __init__(self, limits: GuardrailLimits = None):
        self.limits = limits or GuardrailLimits()
    
    def validate_plan(self, inp: GuardrailInput) -> GuardrailOutput:
        # Use self.limits.top_k_max instead of hardcoded 100
        if adjusted.top_k > self.limits.top_k_max:
            adjusted.top_k = self.limits.top_k_max
            messages.append(f"top_k > {self.limits.top_k_max}; capped.")
```

#### 4. Relationship Validation

**Problem:** No validation that `rerank_top_k <= top_k` (logically required).

**Recommendation:** Add relationship constraint:

```python
# After individual validations
if adjusted.rerank_top_k > adjusted.top_k:
    adjusted.rerank_top_k = adjusted.top_k
    messages.append(
        f"rerank_top_k ({original_rerank}) > top_k ({adjusted.top_k}); "
        f"set rerank_top_k to top_k."
    )
```

#### 5. Logging and Observability

**Problem:** No visibility into validation operations.

**Recommendation:** Add structured logging:

```python
import logging
logger = logging.getLogger(__name__)

def validate_plan(self, inp: GuardrailInput) -> GuardrailOutput:
    plan = inp.plan
    original_values = {
        "top_k": plan.top_k,
        "rerank_top_k": plan.rerank_top_k,
        "max_iters": plan.iterations.max_iters,
    }
    
    # ... validation logic ...
    
    if messages:
        logger.warning(
            "plan_adjusted",
            extra={
                "original": original_values,
                "adjusted": {
                    "top_k": adjusted.top_k,
                    "rerank_top_k": adjusted.rerank_top_k,
                    "max_iters": adjusted.iterations.max_iters,
                },
                "adjustments": messages,
            }
        )
    
    return GuardrailOutput(...)
```

---

### Medium Priority Improvements

#### 6. Severity Levels for Adjustments

**Problem:** All adjustments are treated equally; no distinction between minor fixes and major corrections.

**Recommendation:** Add severity classification:

```python
from enum import Enum

class AdjustmentSeverity(Enum):
    INFO = "info"       # Minor normalization
    WARNING = "warning" # Significant correction
    ERROR = "error"     # Critical fix required

@dataclass
class Adjustment:
    field: str
    original: Any
    corrected: Any
    message: str
    severity: AdjustmentSeverity

@dataclass
class GuardrailOutput:
    status: str
    plan: Plan
    adjustments: List[Adjustment]  # Replace messages with structured adjustments
```

#### 7. Validation for Additional Plan Fields

**Problem:** Only a subset of Plan fields are validated.

**Recommendation:** Validate all relevant fields:

```python
def validate_plan(self, inp: GuardrailInput) -> GuardrailOutput:
    # ... existing validations ...
    
    # Language validation
    SUPPORTED_LANGUAGES = {"en", "es", "fr", "de", "zh", "ja"}
    if adjusted.language not in SUPPORTED_LANGUAGES:
        adjusted.language = "en"
        messages.append(f"Unsupported language; defaulting to 'en'.")
    
    # Retrieval mode validation
    if adjusted.retrieval_mode not in RetrievalModeEnum:
        adjusted.retrieval_mode = RetrievalModeEnum.DUAL_INDEX
        messages.append("Invalid retrieval_mode; set to DUAL_INDEX.")
```

#### 8. Pre and Post Validation Hooks

**Problem:** No extensibility for custom validation rules.

**Recommendation:** Add hook mechanism:

```python
from typing import Callable

class BasicGuardrailAgent(GuardrailAgent):
    def __init__(self):
        self._pre_validators: List[Callable] = []
        self._post_validators: List[Callable] = []
    
    def add_pre_validator(self, fn: Callable[[Plan], Tuple[Plan, List[str]]]):
        self._pre_validators.append(fn)
    
    def add_post_validator(self, fn: Callable[[Plan], Tuple[Plan, List[str]]]):
        self._post_validators.append(fn)
    
    def validate_plan(self, inp: GuardrailInput) -> GuardrailOutput:
        adjusted = inp.plan.model_copy(deep=True)
        messages = []
        
        # Run pre-validators
        for validator in self._pre_validators:
            adjusted, msgs = validator(adjusted)
            messages.extend(msgs)
        
        # Core validation
        # ...
        
        # Run post-validators
        for validator in self._post_validators:
            adjusted, msgs = validator(adjusted)
            messages.extend(msgs)
        
        return GuardrailOutput(...)
```

---

### Low Priority / Future Enhancements

#### 9. Strict Mode Option

**Recommendation:** Add option to reject rather than correct:

```python
class BasicGuardrailAgent(GuardrailAgent):
    def __init__(self, strict: bool = False):
        self.strict = strict
    
    def validate_plan(self, inp: GuardrailInput) -> GuardrailOutput:
        if self.strict:
            errors = self._collect_errors(inp.plan)
            if errors:
                return GuardrailOutput(
                    status="rejected",
                    plan=inp.plan,  # Return original unchanged
                    messages=errors,
                )
        # ... normal correction flow
```

#### 10. Validation History/Audit Trail

**Recommendation:** Track validation history for debugging:

```python
@dataclass
class ValidationRecord:
    timestamp: datetime
    original_plan: Plan
    adjusted_plan: Plan
    adjustments: List[str]

class BasicGuardrailAgent(GuardrailAgent):
    def __init__(self):
        self._history: List[ValidationRecord] = []
    
    def get_history(self) -> List[ValidationRecord]:
        return self._history.copy()
```

#### 11. Resource Cost Estimation

**Recommendation:** Estimate and limit computational cost:

```python
def _estimate_cost(self, plan: Plan) -> float:
    """Estimate relative computational cost."""
    base_cost = plan.top_k * 1.0
    rerank_cost = plan.rerank_top_k * 2.0  # Reranking is more expensive
    iteration_multiplier = plan.iterations.max_iters
    return (base_cost + rerank_cost) * iteration_multiplier

def validate_plan(self, inp: GuardrailInput) -> GuardrailOutput:
    # ... other validations ...
    
    cost = self._estimate_cost(adjusted)
    if cost > MAX_ALLOWED_COST:
        # Scale down parameters proportionally
        scale_factor = MAX_ALLOWED_COST / cost
        adjusted.top_k = max(1, int(adjusted.top_k * scale_factor))
        messages.append(f"Plan cost too high; scaled down parameters.")
```

---

## Usage Examples

### Basic Usage

```python
from guardrail_basic_agent import BasicGuardrailAgent
from core.schemas import GuardrailInput, Plan, PlanIterations

# Initialize agent
agent = BasicGuardrailAgent()

# Create a plan with some invalid values
iterations = PlanIterations(max_iters=0, max_rewrites=2)
plan = Plan(
    top_k=-5,           # Invalid: negative
    rerank_top_k=500,   # Invalid: too high
    iterations=iterations,
    # ... other fields
)

# Validate
inp = GuardrailInput(plan=plan)
output = agent.validate_plan(inp)

print(f"Status: {output.status}")  # "adjusted"
print(f"Messages: {output.messages}")
# ["top_k <= 0; set to 5.", "rerank_top_k > 200; capped to 200.", "max_iters < 1; set to 1."]

print(f"Adjusted top_k: {output.plan.top_k}")  # 5
print(f"Adjusted rerank_top_k: {output.plan.rerank_top_k}")  # 200
print(f"Adjusted max_iters: {output.plan.iterations.max_iters}")  # 1
```

### Pipeline Integration

```python
class RAGPipeline:
    def __init__(self):
        self.router = BasicRouterAgent()
        self.planner = BasicPlannerAgent()
        self.guardrail = BasicGuardrailAgent()
        self.retriever = RetrieverAgent()
    
    def process(self, query: str, config: GlobalConfig):
        # Step 1: Route
        router_output = self.router.route(RouterInput(user_query=query, ...))
        
        # Step 2: Plan
        planner_output = self.planner.plan(PlannerInput(
            global_config=config,
            router_profile=router_output.router_profile,
            ...
        ))
        
        # Step 3: Validate and correct plan
        guardrail_output = self.guardrail.validate_plan(
            GuardrailInput(plan=planner_output.plan)
        )
        
        if guardrail_output.status == "adjusted":
            logger.info(
                f"Plan was adjusted: {guardrail_output.messages}"
            )
        
        # Step 4: Use validated plan
        validated_plan = guardrail_output.plan
        documents = self.retriever.retrieve(
            query=query,
            top_k=validated_plan.top_k
        )
        
        # ... continue pipeline
```

### Conditional Processing Based on Status

```python
def process_with_guardrail(plan: Plan) -> ProcessingResult:
    guardrail = BasicGuardrailAgent()
    output = guardrail.validate_plan(GuardrailInput(plan=plan))
    
    if output.status == "ok":
        # Plan was valid, proceed normally
        return execute_plan(output.plan)
    
    elif output.status == "adjusted":
        # Log adjustments for monitoring
        for message in output.messages:
            logger.warning(f"Plan adjustment: {message}")
        
        # Optionally notify user
        if significant_adjustment(output.messages):
            notify_user("Your request was modified for optimal performance")
        
        return execute_plan(output.plan)
```

### Custom Limits Configuration

```python
# Example: More restrictive limits for cost-sensitive deployment
strict_limits = GuardrailLimits(
    top_k_max=50,
    rerank_top_k_max=25,
    max_iters_max=5,
)

guardrail = BasicGuardrailAgent(limits=strict_limits)
```

---

## Appendix

### Glossary

| Term | Definition |
|------|------------|
| **Guardrail** | Safety mechanism that prevents invalid configurations |
| **Normalization** | Converting invalid values to valid defaults |
| **top_k** | Number of documents to retrieve from the index |
| **rerank_top_k** | Number of documents to keep after reranking |
| **max_iters** | Maximum retrieval iterations allowed |

### Limit Reference

| Parameter | Minimum | Maximum | Default |
|-----------|---------|---------|---------|
| `top_k` | 1 | 100 | 5 |
| `rerank_top_k` | 1 | 200 | = top_k |
| `max_iters` | 1 | None* | 1 |
| `max_rewrites` | None* | None* | - |

*Not currently validated - see recommendations

### Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | Initial | Basic limit validation for top_k, rerank_top_k, max_iters |

### References

- Radiant Repository: https://github.com/dshipley71/radiant
- Related Files: `planner_basic_agent.py`, `core/schemas.py`, `core/interfaces.py`

---

*Document generated for the Radiant RAG Pipeline project. For questions or contributions, please refer to the project repository.*
