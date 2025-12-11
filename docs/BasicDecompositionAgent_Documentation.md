# BasicDecompositionAgent Documentation

Technical reference for the Radiant RAG pipeline decomposition agent.

---

## Overview

The `BasicDecompositionAgent` breaks down complex, multi-part queries into simpler, independently retrievable subqueries. It also identifies comparison queries that require information about multiple entities.

**Module Location:** `agents/decomposition.py`

**Interface:** `DecompositionAgent` (from `core.interfaces`)

---

## Class Definition

```python
class BasicDecompositionAgent(DecompositionAgent):
    """Heuristic decomposition for multi-part and comparison queries."""
    
    role = "decomposition"
    
    @property
    def name(self) -> str:
        return "BasicDecompositionAgent"
    
    def describe(self) -> str:
        return "Heuristic decomposition for multi-part and comparison queries."
    
    def decompose(self, inp: DecompositionInput) -> DecompositionOutput:
        ...
```

---

## Functionality

### Main Method: `decompose()`

**Input:** `DecompositionInput`
- `ctx`: Request context
- `user_query`: User's query string
- `router_profile`: Router classification results
- `config`: Decomposition configuration

**Output:** `DecompositionOutput`
- `decomposition`: Decomposition results

### Processing Logic

```python
def decompose(self, inp: DecompositionInput) -> DecompositionOutput:
    q = inp.user_query.strip()

    comparison_pairs: List[ComparisonPair] = []
    subqueries: List[Subquery] = []

    # Detect comparison pattern
    lower = q.lower()
    if " vs " in lower:
        left, right = q.split(" vs ", 1)
        comparison_pairs.append(ComparisonPair(left=left.strip(), right=right.strip()))

    # Split on "and" / "&"
    parts = [p.strip() for p in q.replace(" & ", " and ").split(" and ") if p.strip()]
    if len(parts) > 1:
        for i, p in enumerate(parts, start=1):
            subqueries.append(Subquery(id=f"sub-{i}", text=p))

    is_multi_part = bool(subqueries or comparison_pairs)

    dec = Decomposition(
        is_multi_part=is_multi_part,
        subqueries=subqueries,
        comparison_pairs=comparison_pairs,
    )
    return DecompositionOutput(decomposition=dec)
```

---

## Decomposition Patterns

### Comparison Detection

Pattern: `" vs "` (case-insensitive)

**Example:**
- Input: `"Python vs JavaScript"`
- Output: `ComparisonPair(left="Python", right="JavaScript")`

### Subquery Splitting

Pattern: `" and "` or `" & "`

**Example:**
- Input: `"features and pricing"`
- Output: `[Subquery(id="sub-1", text="features"), Subquery(id="sub-2", text="pricing")]`

### Combined Example

- Input: `"Compare Python vs JavaScript and their frameworks"`
- Output:
  - `comparison_pairs: [{left: "Compare Python", right: "JavaScript and their frameworks"}]`
  - `subqueries: [{id: "sub-1", text: "Compare Python vs JavaScript"}, {id: "sub-2", text: "their frameworks"}]`
  - `is_multi_part: true`

---

## Output Schema

### Decomposition

```python
class Decomposition(BaseModel):
    is_multi_part: bool
    subqueries: List[Subquery] = []
    comparison_pairs: List[ComparisonPair] = []

class Subquery(BaseModel):
    id: str   # "sub-1", "sub-2", etc.
    text: str

class ComparisonPair(BaseModel):
    left: str
    right: str
```

---

## Usage Example

```python
from agents.decomposition import BasicDecompositionAgent
from core.schemas import DecompositionInput, DecompositionConfig, RouterProfile, RequestContext, RuntimeContext
from uuid import uuid4

agent = BasicDecompositionAgent()

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

inp = DecompositionInput(
    ctx=ctx,
    user_query="Python vs JavaScript",
    router_profile=profile,
    config=DecompositionConfig(),
)

output = agent.decompose(inp)
# output.decomposition.is_multi_part == True
# output.decomposition.comparison_pairs == [ComparisonPair(left="Python", right="JavaScript")]
```

---

## Related Documentation

- [CoreInterfaces_Documentation.md](CoreInterfaces_Documentation.md) - `DecompositionAgent` interface
- [CoreSchemas_Documentation.md](CoreSchemas_Documentation.md) - `Decomposition`, `Subquery`, `ComparisonPair` schemas
- [BasicRouterAgent_Documentation.md](BasicRouterAgent_Documentation.md) - Upstream query classification
