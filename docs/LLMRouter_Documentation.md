# LLMRouter Documentation

Technical reference for the Radiant RAG pipeline LLM abstraction layer.

---

## Overview

The `LLMRouter` provides a unified interface for LLM calls, supporting both local HuggingFace models and OpenAI-compatible APIs.

**Module Location:** `core/llm_router.py`

---

## Class Definition

```python
class LLMRouter:
    """Unified LLM router supporting local HF models and OpenAI-compatible APIs."""
    
    def __init__(self, config: dict):
        ...
    
    def chat(self, messages: List[Dict[str, str]], **overrides) -> str:
        ...
    
    def generate(self, prompt: str, max_tokens: Optional[int] = None, 
                 temperature: Optional[float] = None) -> str:
        ...
```

---

## Public Interface

### `chat(messages, **overrides)`

Routes to HF or OpenAI-compatible backend based on configuration.

**Parameters:**
- `messages`: List of message dicts following OpenAI chat schema
- `**overrides`: Optional `max_tokens`, `temperature` overrides

**Returns:** Generated text string

### `generate(prompt, max_tokens, temperature)`

Simple text-generation interface (delegates to `chat()`).

**Parameters:**
- `prompt`: Input prompt string
- `max_tokens`: Optional token limit
- `temperature`: Optional sampling temperature

**Returns:** Generated text string

---

## Configuration

### Full Config Mode (config.fast.yaml)

```yaml
models:
  use_local: true
  llm_model: <huggingface_model_id>
  llm_device: cuda  # or cpu
  llm_max_new_tokens: 256
  llm_temperature: 0.3

llm:
  api_base: <api_url>
  api_key: <key>
  model: <model_name>
  temperature: 0.2
  max_tokens: 256
```

### Bare LLM Config Mode

```python
config = {
    "api_base": "https://api.example.com/v1",
    "api_key": "sk-...",
    "model": "gpt-3.5-turbo",
    "temperature": 0.2,
    "max_tokens": 512,
}
router = LLMRouter(config)
```

---

## Backend Selection

| Config | Backend |
|--------|---------|
| `models.use_local: true` | HuggingFace local |
| `models.use_local: false` | OpenAI-compatible |
| Bare LLM config (no `models` key) | OpenAI-compatible |

---

## HuggingFace Backend

- Uses `transformers` library
- Lazy loads model on first call
- Auto-detects CUDA availability
- Converts chat messages to prompt format

### Message-to-Prompt Conversion

```
SYSTEM: <system content>
USER: <user content>
ASSISTANT:
```

---

## OpenAI-Compatible Backend

- Uses `openai` Python client
- Supports any OpenAI-compatible API (vLLM, Ollama, etc.)
- Lazy initializes client

---

## Usage Example

```python
from core.llm_router import LLMRouter

# OpenAI-compatible mode
config = {
    "api_base": "http://localhost:8000/v1",
    "api_key": "dummy",
    "model": "llama-7b",
    "temperature": 0.2,
}
router = LLMRouter(config)

messages = [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "What is RAG?"},
]

response = router.chat(messages)
print(response)

# Simple generation
text = router.generate("Explain RAG:", max_tokens=256)
```

---

## Related Documentation

- [LLMGeneratorAgent_Documentation.md](LLMGeneratorAgent_Documentation.md) - Uses LLMRouter for generation
- [LLMQEAgent_Documentation.md](LLMQEAgent_Documentation.md) - Uses LLMRouter for query expansion
- [LLMQueryRewriteAgent_Documentation.md](LLMQueryRewriteAgent_Documentation.md) - Uses LLMRouter for rewriting
