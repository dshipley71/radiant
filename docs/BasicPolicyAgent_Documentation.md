# BasicPolicyAgent Documentation

Technical reference for the Radiant RAG pipeline policy agent.

---

## Overview

The `BasicPolicyAgent` makes decisions about whether to finalize an answer, rewrite the query, or continue iterating based on critic feedback and iteration budgets.

**Module Location:** `agents/policy.py`

**Interface:** `PolicyAgent` (from `core.interfaces`)

---

## Class Definition

```python
class BasicPolicyAgent(PolicyAgent):
    """Heuristic policy for finalize / rewrite / continue decisions."""
    
    role = "policy"
    
    @property
    def name(self) -> str:
        return "BasicPolicyAgent"
    
    def describe(self) -> str:
        return "Heuristic policy for finalize / rewrite / continue decisions."
    
    def decide(self, inp: PolicyInput) -> PolicyOutput:
        ...
```

---

## Functionality

### Main Method: `decide()`

**Input:** `PolicyInput`
- `ctx`: Request context
- `iteration`: Current iteration number
- `plan`: Execution plan
- `retrieval_metrics`: Retrieval statistics
- `critic_feedback`: Critic evaluation results

**Output:** `PolicyOutput`
- `decision`: `DecisionEnum` (FINALIZE, REWRITE, CONTINUE)
- `reason`: Explanation string
- `adjustments`: Dict for plan modifications (currently unused)

---

## Decision Logic

```python
def decide(self, inp: PolicyInput) -> PolicyOutput:
    cov = inp.critic_feedback.coverage_score
    risk = inp.critic_feedback.hallucination_risk

    max_rewrites = max(0, inp.plan.iterations.max_rewrites)
    max_iters = max(1, inp.plan.iterations.max_iters)

    # Quality threshold met
    if cov >= 0.6 and risk <= 0.4:
        return PolicyOutput(
            decision=DecisionEnum.FINALIZE,
            reason="Coverage sufficient and hallucination risk acceptable.",
        )

    # Rewrite budget available
    if inp.iteration < max_rewrites:
        return PolicyOutput(
            decision=DecisionEnum.REWRITE,
            reason="Coverage low or hallucination risk high; attempting rewrite.",
        )

    # Iteration budget available
    if inp.iteration + 1 < max_iters:
        return PolicyOutput(
            decision=DecisionEnum.CONTINUE,
            reason="Continuing without rewrite (iteration budget remains).",
        )

    # Budget exhausted
    return PolicyOutput(
        decision=DecisionEnum.FINALIZE,
        reason="Iteration budget exhausted.",
    )
```

---

## Decision Flow

```
┌─────────────────────────────────────────────────────────┐
│                    PolicyInput                          │
│  coverage_score | hallucination_risk | iteration        │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │ coverage >= 0.6 AND    │
              │ risk <= 0.4?           │
              └────────────────────────┘
                    │           │
                   Yes          No
                    │           │
                    ▼           ▼
              ┌─────────┐  ┌────────────────────────┐
              │FINALIZE │  │iteration < max_rewrites?│
              └─────────┘  └────────────────────────┘
                                │           │
                               Yes          No
                                │           │
                                ▼           ▼
                          ┌─────────┐  ┌────────────────────────┐
                          │ REWRITE │  │iteration+1 < max_iters?│
                          └─────────┘  └────────────────────────┘
                                            │           │
                                           Yes          No
                                            │           │
                                            ▼           ▼
                                      ┌──────────┐ ┌─────────┐
                                      │ CONTINUE │ │FINALIZE │
                                      └──────────┘ └─────────┘
```

---

## Thresholds

| Metric | Threshold | Action |
|--------|-----------|--------|
| Coverage | >= 0.6 | Consider acceptable |
| Hallucination Risk | <= 0.4 | Consider acceptable |
| Iteration | < max_rewrites | Allow rewrite |
| Iteration + 1 | < max_iters | Allow continue |

---

## Usage Example

```python
from agents.policy import BasicPolicyAgent
from core.schemas import (
    PolicyInput, CriticFeedback, RetrievalMetrics, Plan, PlanIterations,
    RetrievalModeEnum, BackendEnum, RequestContext, RuntimeContext
)
from uuid import uuid4

policy = BasicPolicyAgent()

ctx = RequestContext(
    request_id=uuid4(),
    session_id=uuid4(),
    runtime=RuntimeContext(),
)

plan = Plan(
    retrieval_mode=RetrievalModeEnum.DUAL_INDEX,
    use_qe=True,
    use_prf=False,
    use_rerank=True,
    iterations=PlanIterations(max_iters=3, max_rewrites=2),
    top_k=10,
    rerank_top_k=5,
    language="en",
    allow_online_tools=False,
    backend=BackendEnum.HF,
)

feedback = CriticFeedback(
    hallucination_risk=0.7,
    coverage_score=0.3,
)

metrics = RetrievalMetrics(num_docs=3, avg_score=0.5)

inp = PolicyInput(
    ctx=ctx,
    iteration=0,
    plan=plan,
    retrieval_metrics=metrics,
    critic_feedback=feedback,
)

output = policy.decide(inp)
# output.decision == DecisionEnum.REWRITE
# output.reason == "Coverage low or hallucination risk high; attempting rewrite."
```

---

## Related Documentation

- [CoreInterfaces_Documentation.md](CoreInterfaces_Documentation.md) - `PolicyAgent` interface
- [CoreSchemas_Documentation.md](CoreSchemas_Documentation.md) - `PolicyInput`, `PolicyOutput`, `DecisionEnum` schemas
- [BasicCriticAgent_Documentation.md](BasicCriticAgent_Documentation.md) - Provides feedback input
- [LLMQueryRewriteAgent_Documentation.md](LLMQueryRewriteAgent_Documentation.md) - Query rewriting on REWRITE decision
