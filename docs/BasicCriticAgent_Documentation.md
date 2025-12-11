# BasicCriticAgent Documentation

Technical reference for the Radiant RAG pipeline critic agent.

---

## Overview

The `BasicCriticAgent` evaluates generated answers for potential quality issues such as hallucination risk and insufficient coverage using lightweight heuristic methods.

**Module Location:** `agents/critic.py`

**Interface:** `CriticAgent` (from `core.interfaces`)

---

## Class Definition

```python
class BasicCriticAgent(CriticAgent):
    """Heuristic critic for coverage and hallucination risk."""
    
    role = "critic"
    
    @property
    def name(self) -> str:
        return "BasicCriticAgent"
    
    def describe(self) -> str:
        return "Heuristic critic for coverage and hallucination risk."
    
    def evaluate(self, inp: CriticInput) -> CriticOutput:
        ...
```

---

## Functionality

### Main Method: `evaluate()`

**Input:** `CriticInput`
- `ctx`: Request context
- `query`: User's query
- `answer`: Generated answer
- `context_snippets`: Retrieved context snippets
- `plan`: Execution plan

**Output:** `CriticOutput` (extends `CriticFeedback`)
- `hallucination_risk`: Float (0.0–1.0)
- `coverage_score`: Float (0.0–1.0)
- `missing_topics`: List[str]
- `ambiguities`: List[str]
- `unsupported_claims`: List[Dict]
- `notes`: List[str]

---

## Evaluation Logic

### Coverage Score

```python
coverage = min(1.0, num_snippets / max_k)
```

Where:
- `num_snippets`: Number of context snippets provided
- `max_k`: Plan's `top_k` value (minimum 1)

### Hallucination Risk

```python
hallucination_risk = 1.0 - coverage
```

Inverse relationship with coverage: fewer context snippets implies higher hallucination risk.

### Quality Notes

Generated when specific conditions are met:

| Condition | Note |
|-----------|------|
| No context snippets | "No retrieval context available; answer may be hallucinated." |
| Answer < 10 words | "Answer is very short; consider elaborating if user needs detail." |
| Coverage < 0.3 | "Low coverage of available context (few relevant snippets)." |

---

## Implementation

```python
def evaluate(self, inp: CriticInput) -> CriticOutput:
    ctx = inp.context_snippets or []
    answer_text = inp.answer.text or ""

    num_snips = len(ctx)
    max_k = max(1, inp.plan.top_k)

    coverage = min(1.0, num_snips / max_k)
    halluc_risk = 1.0 - coverage

    notes: List[str] = []
    if not ctx:
        notes.append("No retrieval context available; answer may be hallucinated.")
    if len(answer_text.split()) < 10:
        notes.append("Answer is very short; consider elaborating if user needs detail.")
    if coverage < 0.3:
        notes.append("Low coverage of available context (few relevant snippets).")

    return CriticOutput(
        hallucination_risk=halluc_risk,
        coverage_score=coverage,
        missing_topics=[],
        ambiguities=[],
        unsupported_claims=[],
        notes=notes,
    )
```

---

## Usage Example

```python
from agents.critic import BasicCriticAgent
from core.schemas import (
    CriticInput, Answer, ContextSnippet, Plan, PlanIterations,
    RetrievalModeEnum, BackendEnum, RequestContext, RuntimeContext
)
from uuid import uuid4

critic = BasicCriticAgent()

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

answer = Answer(text="RAG combines retrieval with generation for better answers.")

snippets = [
    ContextSnippet(
        doc_id="doc1", chunk_id="c1", source_text="RAG is...",
        translated_text="RAG is...", lang="en", score=0.9
    )
]

inp = CriticInput(
    ctx=ctx,
    query="What is RAG?",
    answer=answer,
    context_snippets=snippets,
    plan=plan,
)

output = critic.evaluate(inp)
# output.coverage_score == 0.1 (1 snippet / 10 top_k)
# output.hallucination_risk == 0.9
# output.notes includes "Low coverage of available context..."
```

---

## Related Documentation

- [CoreInterfaces_Documentation.md](CoreInterfaces_Documentation.md) - `CriticAgent` interface
- [CoreSchemas_Documentation.md](CoreSchemas_Documentation.md) - `CriticFeedback`, `CriticInput`, `CriticOutput` schemas
- [BasicPolicyAgent_Documentation.md](BasicPolicyAgent_Documentation.md) - Uses critic feedback for decisions
