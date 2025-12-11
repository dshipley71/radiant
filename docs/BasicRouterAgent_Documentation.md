# BasicRouterAgent Documentation

Technical reference for the Radiant RAG pipeline router agent.

---

## Overview

The `BasicRouterAgent` is a heuristic-based query routing component that analyzes incoming user queries and determines the optimal retrieval and processing strategy.

**Module Location:** `agents/router.py`

**Interface:** `RouterAgent` (from `core.interfaces`)

---

## Class Definition

```python
class BasicRouterAgent(RouterAgent):
    """Heuristic router for query type and high-level retrieval toggles."""
    
    role = "router"
    
    @property
    def name(self) -> str:
        return "BasicRouterAgent"
    
    def describe(self) -> str:
        return "Heuristic router for query type and high-level retrieval toggles."
    
    def route(self, inp: RouterInput) -> RouterOutput:
        ...
```

---

## Functionality

### Main Method: `route()`

**Input:** `RouterInput`
- `ctx`: Request context
- `user_query`: User's query string
- `history`: Conversation history (`List[Message]`)
- `config`: Router configuration (`RouterConfig`)

**Output:** `RouterOutput`
- `router_profile`: Classification results (`RouterProfile`)

### Processing Steps

1. Extract and normalize query (strip whitespace, lowercase for analysis)
2. Truncate history based on `max_hist_turns` configuration
3. Classify query type using heuristics
4. Apply `default_query_type` fallback if classified as "other"
5. Detect follow-up queries based on history
6. Infer expected answer style
7. Assess query complexity
8. Determine QE/PRF/rerank toggles
9. Return `RouterProfile`

---

## Classification Logic

### Query Type Classification

| Type | Trigger Conditions |
|------|-------------------|
| `comparison` | Contains " vs ", " versus ", or "difference between" |
| `list` | Starts with "list " OR contains " list of " OR contains "top " |
| `explanation` | Starts with "how " or "why " OR contains "explain" |
| `lookup` | Starts with "what " OR ends with "?" |
| `other` | Default fallback |

**Priority:** comparison → list → explanation → lookup → other

### Answer Style Inference

| Style | Trigger Conditions |
|-------|-------------------|
| `multi_section` | Contains "overview", "detailed", or "guide" |
| `short` | Token count < 8 AND not a follow-up |
| `paragraph` | Token count < 8 AND is follow-up, OR token count ≥ 8 |

### Complexity Assessment

| Complexity | Token Count |
|------------|-------------|
| `low` | < 8 tokens |
| `medium` | 8–19 tokens |
| `high` | ≥ 20 tokens |

### Follow-up Detection

A query is classified as a follow-up when:
1. Conversation history exists
2. Query has ≤ 5 tokens
3. Query does NOT start with: what, how, why, who, where, when

---

## Toggle Decision Logic

### Query Expansion (QE)

**Enabled when:**
- Complexity is "medium" or "high", OR
- Query is detected as a follow-up

### Pseudo-Relevance Feedback (PRF)

**Enabled when:**
- Query type is "comparison" or "list", AND
- Complexity is NOT "low"

### Reranking

Always enabled.

### Decision Matrix

| Query Type | Complexity | Follow-up | QE | PRF | Rerank |
|------------|------------|-----------|-----|-----|--------|
| comparison | low | No | ✗ | ✗ | ✓ |
| comparison | medium/high | No | ✓ | ✓ | ✓ |
| list | low | No | ✗ | ✗ | ✓ |
| list | medium/high | No | ✓ | ✓ | ✓ |
| explanation | low | No | ✗ | ✗ | ✓ |
| explanation | medium/high | No | ✓ | ✗ | ✓ |
| lookup | low | No | ✗ | ✗ | ✓ |
| lookup | medium/high | No | ✓ | ✗ | ✓ |
| any | any | Yes | ✓ | varies | ✓ |

---

## Configuration

### RouterConfig Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_hist_turns` | `int` | `10` | Maximum conversation turns to consider |
| `default_query_type` | `str` | `None` | Fallback query type for "other" classification |

---

## Usage Example

```python
from agents.router import BasicRouterAgent
from core.schemas import RouterInput, RouterConfig, RequestContext, RuntimeContext
from uuid import uuid4

router = BasicRouterAgent()

ctx = RequestContext(
    request_id=uuid4(),
    session_id=uuid4(),
    runtime=RuntimeContext(),
)

config = RouterConfig(max_hist_turns=5, default_query_type="lookup")

inp = RouterInput(
    ctx=ctx,
    user_query="What is the difference between REST and GraphQL?",
    config=config,
    history=[],
)

output = router.route(inp)
profile = output.router_profile

# profile.query_type == "comparison"
# profile.use_qe == True (medium complexity)
# profile.use_prf == True (comparison + medium)
# profile.use_rerank == True
# profile.expected_answer_style == "paragraph"
# profile.complexity_hint == "medium"
```

---

## Related Documentation

- [CoreInterfaces_Documentation.md](CoreInterfaces_Documentation.md) - `RouterAgent` interface
- [CoreSchemas_Documentation.md](CoreSchemas_Documentation.md) - `RouterInput`, `RouterOutput`, `RouterProfile` schemas
- [Orchestrator_Documentation.md](Orchestrator_Documentation.md) - Pipeline integration
