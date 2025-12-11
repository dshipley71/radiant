# BasicPlannerAgent Documentation

Technical reference for the Radiant RAG pipeline planner agent.

---

## Overview

The `BasicPlannerAgent` converts high-level configuration and router hints into a concrete execution plan. It determines retrieval mode, feature toggles, iteration budgets, and top_k parameters.

**Module Location:** `agents/planner.py`

**Interface:** `PlannerAgent` (from `core.interfaces`)

---

## Class Definition

```python
class BasicPlannerAgent(PlannerAgent):
    """Planner that converts high-level config + router hints into a concrete Plan."""
    
    role = "planner"
    
    def __init__(self, config_path: Optional[str] = None) -> None:
        self._cfg_raw: Dict[str, Any] = self._load_config(config_path)
    
    @property
    def name(self) -> str:
        return "BasicPlannerAgent"
    
    def plan(self, inp: PlannerInput) -> PlannerOutput:
        ...
```

---

## Functionality

### Main Method: `plan()`

**Input:** `PlannerInput`
- `ctx`: Request context
- `router_profile`: Router classification results
- `decomposition`: Query decomposition results
- `global_config`: System-wide configuration

**Output:** `PlannerOutput`
- `plan`: Execution plan

### Processing Steps

1. **Retrieval mode selection** - Reads from `config.fast.yaml` or falls back to `GlobalConfig`
2. **Feature toggles** - Combines `GlobalConfig` enables with `RouterProfile` flags
3. **Iteration budget** - Scales based on complexity hint
4. **top_k scaling** - Respects config values with sanity constraints
5. **Plan assembly** - Creates `Plan` object

---

## Configuration

### Retrieval Mode

Controlled by `config.fast.yaml`:

```yaml
retrieval:
  leaf_only: false   # DUAL_INDEX mode
  # or
  leaf_only: true    # LEAF_ONLY mode
```

Falls back to `GlobalConfig.default_retrieval_mode` if not specified.

### Feature Toggles

Feature flags are determined by ANDing `GlobalConfig` and `RouterProfile`:

```python
use_qe = bool(cfg.enable_qe and rp.use_qe)
use_prf = bool(cfg.enable_prf and rp.use_prf)
use_rerank = bool(cfg.enable_rerank and rp.use_rerank)
```

---

## Scaling Logic

### Iteration Budget Scaling

Based on `RouterProfile.complexity_hint`:

| Complexity | max_iters | max_rewrites |
|------------|-----------|--------------|
| `low` | base // 2 (min 1) | base // 2 (min 0) |
| `medium` | base | base |
| `high` | base + 1 | base + 1 |

Additional: For `comparison` or `list` queries at medium/high complexity, `max_rewrites` is increased by 1.

### top_k Constraints

The planner enforces:
- `top_k >= 1`
- `rerank_top_k >= 1`
- `rerank_top_k <= top_k`

---

## Output Schema

### Plan

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

class PlanIterations(BaseModel):
    max_iters: int
    max_rewrites: int
```

---

## Usage Example

```python
from agents.planner import BasicPlannerAgent
from core.schemas import (
    PlannerInput, GlobalConfig, RouterProfile, Decomposition,
    RequestContext, RuntimeContext
)
from uuid import uuid4

planner = BasicPlannerAgent(config_path="config.fast.yaml")

ctx = RequestContext(
    request_id=uuid4(),
    session_id=uuid4(),
    runtime=RuntimeContext(),
)

profile = RouterProfile(
    query_type="comparison",
    use_qe=True,
    use_prf=True,
    use_rerank=True,
    expected_answer_style="paragraph",
    complexity_hint="medium",
)

inp = PlannerInput(
    ctx=ctx,
    router_profile=profile,
    decomposition=Decomposition(is_multi_part=False),
    global_config=GlobalConfig(),
)

output = planner.plan(inp)
# output.plan.retrieval_mode -> RetrievalModeEnum.DUAL_INDEX
# output.plan.use_qe -> True
# output.plan.iterations.max_rewrites -> base + 1 (comparison + medium)
```

---

## Related Documentation

- [CoreInterfaces_Documentation.md](CoreInterfaces_Documentation.md) - `PlannerAgent` interface
- [CoreSchemas_Documentation.md](CoreSchemas_Documentation.md) - `Plan`, `GlobalConfig` schemas
- [BasicRouterAgent_Documentation.md](BasicRouterAgent_Documentation.md) - Upstream classification
- [BasicGuardrailAgent_Documentation.md](BasicGuardrailAgent_Documentation.md) - Plan validation
