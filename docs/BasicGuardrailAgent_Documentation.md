# BasicGuardrailAgent Documentation

Technical reference for the Radiant RAG pipeline guardrail agent.

---

## Overview

The `BasicGuardrailAgent` validates and normalizes execution plans by enforcing resource limits and sanity constraints.

**Module Location:** `agents/guardrail.py`

**Interface:** `GuardrailAgent` (from `core.interfaces`)

---

## Class Definition

```python
class BasicGuardrailAgent(GuardrailAgent):
    """Basic guardrails for Plan sanity (limits & normalization)."""
    
    role = "guardrail"
    
    @property
    def name(self) -> str:
        return "BasicGuardrailAgent"
    
    def describe(self) -> str:
        return "Basic guardrails for Plan sanity (limits & normalization)."
    
    def validate_plan(self, inp: GuardrailInput) -> GuardrailOutput:
        ...
```

---

## Functionality

### Main Method: `validate_plan()`

**Input:** `GuardrailInput`
- `ctx`: Request context
- `plan`: Execution plan to validate

**Output:** `GuardrailOutput`
- `status`: `"ok"` | `"adjusted"` | `"blocked"`
- `plan`: Validated/adjusted plan
- `messages`: List of adjustment messages

---

## Validation Rules

| Parameter | Constraint | Action |
|-----------|------------|--------|
| `top_k <= 0` | Invalid | Set to 5 |
| `top_k > 100` | Too high | Cap to 100 |
| `rerank_top_k <= 0` | Invalid | Set to `top_k` |
| `rerank_top_k > 200` | Too high | Cap to 200 |
| `max_iters < 1` | Invalid | Set to 1 |

---

## Implementation

```python
def validate_plan(self, inp: GuardrailInput) -> GuardrailOutput:
    plan = inp.plan
    adjusted = plan.model_copy(deep=True)
    messages: List[str] = []
    status = "ok"

    if adjusted.top_k <= 0:
        adjusted.top_k = 5
        messages.append("top_k <= 0; set to 5.")
    if adjusted.top_k > 100:
        adjusted.top_k = 100
        messages.append("top_k > 100; capped to 100.")

    if adjusted.rerank_top_k <= 0:
        adjusted.rerank_top_k = adjusted.top_k
        messages.append("rerank_top_k <= 0; set to top_k.")
    if adjusted.rerank_top_k > 200:
        adjusted.rerank_top_k = 200
        messages.append("rerank_top_k > 200; capped to 200.")

    if adjusted.iterations.max_iters < 1:
        adjusted.iterations.max_iters = 1
        messages.append("max_iters < 1; set to 1.")

    if messages:
        status = "adjusted"

    return GuardrailOutput(status=status, plan=adjusted, messages=messages)
```

---

## Usage Example

```python
from agents.guardrail import BasicGuardrailAgent
from core.schemas import GuardrailInput, Plan, PlanIterations, RetrievalModeEnum, BackendEnum, RequestContext, RuntimeContext
from uuid import uuid4

guardrail = BasicGuardrailAgent()

ctx = RequestContext(
    request_id=uuid4(),
    session_id=uuid4(),
    runtime=RuntimeContext(),
)

# Plan with invalid values
plan = Plan(
    retrieval_mode=RetrievalModeEnum.DUAL_INDEX,
    use_qe=True,
    use_prf=False,
    use_rerank=True,
    iterations=PlanIterations(max_iters=0, max_rewrites=2),  # Invalid
    top_k=150,  # Too high
    rerank_top_k=10,
    language="en",
    allow_online_tools=False,
    backend=BackendEnum.HF,
)

inp = GuardrailInput(ctx=ctx, plan=plan)
output = guardrail.validate_plan(inp)

# output.status == "adjusted"
# output.plan.top_k == 100 (capped)
# output.plan.iterations.max_iters == 1 (set)
# output.messages == ["top_k > 100; capped to 100.", "max_iters < 1; set to 1."]
```

---

## Related Documentation

- [CoreInterfaces_Documentation.md](CoreInterfaces_Documentation.md) - `GuardrailAgent` interface
- [CoreSchemas_Documentation.md](CoreSchemas_Documentation.md) - `Plan`, `GuardrailInput`, `GuardrailOutput` schemas
- [BasicPlannerAgent_Documentation.md](BasicPlannerAgent_Documentation.md) - Plan creation
