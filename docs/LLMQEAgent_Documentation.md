# LLMQEAgent Documentation

Technical reference for the Radiant RAG pipeline query expansion agent.

---

## Overview

The `LLMQEAgent` generates query paraphrases using an LLM to improve retrieval recall through semantic variations.

**Module Location:** `agents/qe.py`

**Interface:** `QEAgent` (from `core.interfaces`)

---

## Class Definition

```python
class LLMQEAgent(QEAgent):
    """QEAgent that uses LLMRouter for query expansion."""
    
    role = "qe"
    
    def __init__(self, config: dict):
        self.cfg = config or {}
        self.router = LLMRouter(self.cfg)
        qe_cfg = self.cfg.get("retrieval", {}).get("query_expansion", {})
        self.default_num_variants = qe_cfg.get("num_variants", 5)
        self.temperature = qe_cfg.get("temperature", 0.2)
        self.max_new_tokens = qe_cfg.get("max_new_tokens", 64)
    
    @property
    def name(self) -> str:
        return "LLMQEAgent"
    
    def expand(self, inp: QEInput) -> QEOutput:
        ...
```

---

## Functionality

### Main Method: `expand()`

**Input:** `QEInput`
- `ctx`: Request context
- `query`: User's query string
- `router_profile`: Router classification
- `plan`: Execution plan
- `translation_metadata`: Optional translation info

**Output:** `QEOutput`
- `expanded_queries`: List of paraphrased queries

---

## Behavior

- If `plan.use_qe` is `False`, returns empty list
- Generates N paraphrases using LLM
- N determined by: `plan.max_qe_variants` → config → default (5)

---

## Configuration

```yaml
retrieval:
  query_expansion:
    num_variants: 5
    temperature: 0.2
    max_new_tokens: 64
```

---

## LLM Prompt

```
System: You are a query expansion assistant for a retrieval system.
Given a user query, generate several diverse paraphrases that preserve the
original meaning but use different wording or focus on complementary aspects.
Do NOT introduce new facts, constraints, or assumptions.
Return ONLY the paraphrased queries, one per line, with no bullets or numbering.

User: Original query: <query>
Number of paraphrases: <N>
```

---

## Related Documentation

- [CoreInterfaces_Documentation.md](CoreInterfaces_Documentation.md) - `QEAgent` interface
- [CoreSchemas_Documentation.md](CoreSchemas_Documentation.md) - `QEInput`, `QEOutput` schemas
- [LLMRouter_Documentation.md](LLMRouter_Documentation.md) - LLM backend abstraction
