# LLMGeneratorAgent Documentation

Technical reference for the Radiant RAG pipeline generator agent.

---

## Overview

The `LLMGeneratorAgent` generates answers from retrieved context snippets using an LLM, with citation tracking.

**Module Location:** `agents/generator.py`

**Interface:** `GeneratorAgent` (from `core.interfaces`)

---

## Class Definition

```python
class LLMGeneratorAgent:
    """Generator agent that builds cited context and calls LLM."""
    
    role = "generator"
    
    def __init__(
        self,
        config_path: Optional[str] = "../config.fast.yaml",
        config: Optional[Any] = None,
        name: str = "LLMGeneratorAgent",
    ) -> None:
        ...
    
    def generate(self, inp: GeneratorInput) -> GeneratorOutput:
        ...
```

---

## Functionality

### Main Method: `generate()`

**Input:** `GeneratorInput`
- `ctx`: Request context
- `query`: User's query string
- `plan`: Execution plan
- `context_snippets`: Retrieved context snippets

**Output:** `GeneratorOutput`
- `answer`: Generated answer (`Answer` object)
- `citations`: List of `Citation` objects

---

## Configuration

```yaml
llm:
  model: <model_name>
  api_base: <api_url>
  api_key: <key>
  temperature: 0.2
  max_tokens: 512

retrieval:
  context_max_chars: 4000
```

### Environment Overrides

| Variable | Purpose |
|----------|---------|
| `AGENTIC_LLM_MODEL` | Model name override |
| `AGENTIC_LLM_API_BASE` | API URL override |
| `AGENTIC_LLM_API_KEY` | API key override |
| `AGENTIC_LLM_TEMPERATURE` | Temperature override |

---

## Prompt Template

```
You are a helpful AI assistant. Use ONLY the provided context snippets to 
answer the user's question. If the context does not contain enough 
information, say so explicitly.

Important formatting rule:
- Do NOT include citation or source tags like [S1], [S2], etc. in your answer.
- Just answer in natural language.

Question:
<query>

Context:
<numbered context snippets>

Answer:
```

---

## Citation Tracking

Citations are built from context snippets with:
- Document ID and chunk ID
- Page number (if available)
- Relevance score
- Language
- Original and translated text

---

## Related Documentation

- [CoreInterfaces_Documentation.md](CoreInterfaces_Documentation.md) - `GeneratorAgent` interface
- [CoreSchemas_Documentation.md](CoreSchemas_Documentation.md) - `GeneratorInput`, `GeneratorOutput`, `Answer`, `Citation` schemas
- [LLMRouter_Documentation.md](LLMRouter_Documentation.md) - LLM backend abstraction
- [BasicCriticAgent_Documentation.md](BasicCriticAgent_Documentation.md) - Evaluates generated answers
