# LLMRouter Documentation

## Technical Reference for the Radiant RAG Pipeline LLM Abstraction Layer

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture Context](#architecture-context)
3. [Class Structure](#class-structure)
4. [Configuration System](#configuration-system)
5. [Backend Implementations](#backend-implementations)
6. [Public Interface](#public-interface)
7. [Data Flow](#data-flow)
8. [Testing Strategies](#testing-strategies)
9. [Recommendations and Improvements](#recommendations-and-improvements)
10. [Usage Examples](#usage-examples)

---

## Overview

### Purpose

The `LLMRouter` is the unified LLM abstraction layer for the Radiant RAG pipeline. It provides a consistent interface for calling language models regardless of whether they're hosted locally (HuggingFace) or accessed via API (OpenAI-compatible).

### Key Responsibilities

- Route LLM requests to appropriate backend (local HF or remote API)
- Manage model loading and client initialization (lazy loading)
- Convert between message formats (chat messages ↔ text prompts)
- Apply default and override parameters (temperature, max_tokens)
- Provide consistent interface across different model types

### Design Philosophy

The router follows the **Strategy Pattern** where the backend implementation is selected at runtime based on configuration. Lazy loading ensures resources are only allocated when first needed, and the unified interface allows agents to remain backend-agnostic.

---

## Architecture Context

### Position in the Radiant Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    Agents Using LLMRouter                       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │LLMGenerator │  │  LLMQEAgent │  │ LLMRewrite  │             │
│  │   Agent     │  │             │  │   Agent     │             │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘             │
│         │                │                │                     │
│         └────────────────┼────────────────┘                     │
│                          │                                      │
│                          ▼                                      │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                      LLMRouter                          │   │
│  │  ┌───────────────────────────────────────────────────┐  │   │
│  │  │  chat(messages) / generate(prompt)                │  │   │
│  │  └───────────────────┬───────────────────────────────┘  │   │
│  │                      │                                   │   │
│  │          ┌───────────┴───────────┐                      │   │
│  │          │    use_local?         │                      │   │
│  │          └───────────┬───────────┘                      │   │
│  │                      │                                   │   │
│  │        ┌─────────────┴─────────────┐                    │   │
│  │        │                           │                    │   │
│  │        ▼                           ▼                    │   │
│  │  ┌─────────────┐            ┌─────────────┐            │   │
│  │  │ HF Backend  │            │OpenAI Backend│            │   │
│  │  │ _chat_hf()  │            │_chat_openai()│            │   │
│  │  └──────┬──────┘            └──────┬──────┘            │   │
│  │         │                          │                    │   │
│  └─────────┼──────────────────────────┼────────────────────┘   │
│            │                          │                         │
└────────────┼──────────────────────────┼─────────────────────────┘
             │                          │
             ▼                          ▼
      ┌─────────────┐            ┌─────────────┐
      │ HuggingFace │            │ OpenAI API  │
      │Local Models │            │ (or compat) │
      └─────────────┘            └─────────────┘
```

### Backend Selection

```
┌─────────────────────────────────────────────────────────────────┐
│                    Backend Selection Logic                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Config Type 1: Full config.fast.yaml                          │
│    Has: "models" or "llm" keys                                 │
│    Default: use_local = True (local HF)                        │
│    Override: models.use_local = False → OpenAI                 │
│                                                                 │
│  Config Type 2: Bare LLM config dict                           │
│    Has: model, api_base, api_key directly                      │
│    Default: use_local = False (OpenAI-compatible)              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Related Components

| Component | Relationship |
|-----------|--------------|
| `LLMGeneratorAgent` | Uses router for RAG generation |
| `LLMQEAgent` | Uses router for query expansion |
| `LLMQueryRewriteAgent` | Uses router for query refinement |
| `config.fast.yaml` | Configuration source |
| `OpenAI` | Python client for API calls |
| `transformers` | HuggingFace model loading |

---

## Class Structure

### Instance Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `cfg` | `dict` | Full configuration dictionary |
| `use_local` | `bool` | Whether to use local HF models |
| `local_model_id` | `str` | HuggingFace model identifier |
| `local_device` | `str` | Device for local model (cuda/cpu) |
| `local_max_tokens` | `int` | Default max tokens for HF |
| `local_temperature` | `float` | Default temperature for HF |
| `api_base` | `str` | OpenAI-compatible API base URL |
| `api_key` | `str` | API key for authentication |
| `api_model` | `str` | Model name for API calls |
| `api_temperature` | `float` | Default temperature for API |
| `api_max_tokens` | `int` | Default max tokens for API |
| `model_name` | `str` | Unified name for telemetry |
| `_hf_model` | `AutoModelForCausalLM` | Lazy-loaded HF model |
| `_hf_tokenizer` | `AutoTokenizer` | Lazy-loaded HF tokenizer |
| `_hf_pipe` | `Pipeline` | Lazy-loaded HF pipeline |
| `_client` | `OpenAI` | Lazy-loaded OpenAI client |

### Constructor

```python
def __init__(self, config: dict)
```

**Initialization Steps:**
1. Store configuration
2. Detect config format (full vs bare)
3. Set appropriate defaults based on format
4. Extract HF model settings
5. Extract OpenAI API settings
6. Set unified model name for telemetry

### Methods

| Method | Visibility | Purpose |
|--------|------------|---------|
| `chat(messages, **overrides)` | Public | Main chat interface |
| `generate(prompt, max_tokens, temperature)` | Public | Simple text generation |
| `_load_hf()` | Private | Lazy load HuggingFace model |
| `_chat_hf(messages, **overrides)` | Private | HF backend implementation |
| `_load_openai()` | Private | Lazy load OpenAI client |
| `_chat_openai(messages, **overrides)` | Private | OpenAI backend implementation |
| `_messages_to_prompt(messages)` | Static | Convert messages to text prompt |

---

## Configuration System

### Full Configuration Format

```yaml
# config.fast.yaml

models:
  use_local: false           # false = use OpenAI, true = use HF
  llm_model: "meta-llama/Llama-2-7b-chat-hf"  # HF model ID
  llm_device: "cuda"         # cuda or cpu
  llm_max_new_tokens: 256    # Default max tokens for HF
  llm_temperature: 0.3       # Default temperature for HF

llm:
  api_base: "https://api.openai.com/v1"
  api_key: "${OPENAI_API_KEY}"
  model: "gpt-4o-mini"
  temperature: 0.2           # Default temperature for API
  max_tokens: 256            # Default max tokens for API
```

### Bare LLM Configuration Format

```python
# Direct LLM config (used by some agents)
config = {
    "api_base": "https://api.openai.com/v1",
    "api_key": "sk-...",
    "model": "gpt-4o-mini",
    "temperature": 0.2,
    "max_tokens": 256,
}
```

### Configuration Parameters

#### Local HF Settings

| Parameter | Config Path | Default | Description |
|-----------|-------------|---------|-------------|
| `use_local` | `models.use_local` | `True` (full config) | Use local HF |
| `llm_model` | `models.llm_model` | None | HF model ID |
| `llm_device` | `models.llm_device` | Auto-detect | cuda/cpu |
| `llm_max_new_tokens` | `models.llm_max_new_tokens` | 256 | Max tokens |
| `llm_temperature` | `models.llm_temperature` | 0.3 | Temperature |

#### OpenAI API Settings

| Parameter | Config Path | Default | Description |
|-----------|-------------|---------|-------------|
| `api_base` | `llm.api_base` | None | API endpoint |
| `api_key` | `llm.api_key` | None | API key |
| `model` | `llm.model` | None | Model name |
| `temperature` | `llm.temperature` | 0.2 | Temperature |
| `max_tokens` | `llm.max_tokens` | 256 | Max tokens |

### Environment Variables

| Variable | Purpose |
|----------|---------|
| `OPENAI_API_KEY` | API key (can be referenced in config) |

### Device Auto-Detection

```python
# Automatic GPU detection for HF models
self.local_device = models_cfg.get(
    "llm_device",
    "cuda" if torch.cuda.is_available() else "cpu",
)
```

---

## Backend Implementations

### HuggingFace Backend

#### Model Loading (`_load_hf`)

```python
def _load_hf(self):
    if self._hf_pipe is not None:
        return  # Already loaded
    
    # Load tokenizer
    self._hf_tokenizer = AutoTokenizer.from_pretrained(self.local_model_id)
    
    # Load model with appropriate dtype
    self._hf_model = AutoModelForCausalLM.from_pretrained(
        self.local_model_id,
        torch_dtype=torch.float16 if "cuda" in self.local_device else torch.float32,
        device_map="auto" if "cuda" in self.local_device else None,
    )
    
    # Create pipeline
    self._hf_pipe = pipeline(
        "text-generation",
        model=self._hf_model,
        tokenizer=self._hf_tokenizer,
        device=0 if "cuda" in self.local_device else -1,
    )
```

#### Message Conversion

```python
@staticmethod
def _messages_to_prompt(messages: List[Dict[str, str]]) -> str:
    """Convert chat messages to HF prompt format."""
    out = []
    for msg in messages:
        role = msg["role"].upper()
        out.append(f"{role}: {msg['content']}")
    out.append("ASSISTANT:")
    return "\n".join(out)
```

**Example Conversion:**
```
Input:
[
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "Hello!"}
]

Output:
SYSTEM: You are helpful.
USER: Hello!
ASSISTANT:
```

#### Chat Implementation

```python
def _chat_hf(self, messages, **overrides):
    self._load_hf()
    
    max_tokens = overrides.get("max_tokens", self.local_max_tokens)
    temperature = overrides.get("temperature", self.local_temperature)
    
    prompt = self._messages_to_prompt(messages)
    
    out = self._hf_pipe(
        prompt,
        max_new_tokens=max_tokens,
        do_sample=(temperature > 0),
        temperature=temperature,
    )
    
    # Extract generated text (excluding prompt)
    full = out[0]["generated_text"]
    return full[len(prompt):].strip()
```

### OpenAI Backend

#### Client Loading (`_load_openai`)

```python
def _load_openai(self):
    if self._client is None:
        self._client = OpenAI(
            base_url=self.api_base,
            api_key=self.api_key
        )
```

#### Chat Implementation

```python
def _chat_openai(self, messages, **overrides):
    self._load_openai()
    
    temperature = overrides.get("temperature", self.api_temperature)
    max_tokens = overrides.get("max_tokens", self.api_max_tokens)
    
    resp = self._client.chat.completions.create(
        model=self.api_model,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=messages,
    )
    
    return resp.choices[0].message.content.strip()
```

---

## Public Interface

### `chat()` Method

Primary method for chat-style interactions.

**Signature:**
```python
def chat(self, messages: List[Dict[str, str]], **overrides) -> str
```

**Parameters:**
- `messages`: List of chat messages in OpenAI format
- `**overrides`: Parameter overrides (temperature, max_tokens)

**Returns:**
- `str`: Generated text response

**Message Format:**
```python
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"},
]
```

### `generate()` Method

Simplified text generation interface.

**Signature:**
```python
def generate(
    self,
    prompt: str,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
) -> str
```

**Parameters:**
- `prompt`: Text prompt for generation
- `max_tokens`: Override for max tokens
- `temperature`: Override for temperature

**Returns:**
- `str`: Generated text

**Implementation:**
```python
def generate(self, prompt, max_tokens=None, temperature=None):
    messages = [{"role": "user", "content": prompt}]
    overrides = {}
    if max_tokens is not None:
        overrides["max_tokens"] = max_tokens
    if temperature is not None:
        overrides["temperature"] = temperature
    return self.chat(messages, **overrides)
```

---

## Data Flow

### Chat Request Flow

```
┌────────────────────────────────────────────────────────────────────┐
│                        chat(messages, **overrides)                  │
└────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
                    ┌─────────────────────────┐
                    │      use_local?         │
                    └─────────────────────────┘
                         │              │
                    True │              │ False
                         ▼              ▼
              ┌──────────────┐   ┌──────────────┐
              │  _chat_hf()  │   │_chat_openai()│
              └──────┬───────┘   └──────┬───────┘
                     │                  │
                     ▼                  ▼
              ┌──────────────┐   ┌──────────────┐
              │  _load_hf()  │   │_load_openai()│
              │ (if needed)  │   │ (if needed)  │
              └──────┬───────┘   └──────┬───────┘
                     │                  │
                     ▼                  ▼
              ┌──────────────┐   ┌──────────────┐
              │ Convert to   │   │ OpenAI API   │
              │ HF prompt    │   │ call         │
              └──────┬───────┘   └──────┬───────┘
                     │                  │
                     ▼                  ▼
              ┌──────────────┐   ┌──────────────┐
              │ HF pipeline  │   │ Parse resp-  │
              │ generation   │   │ onse.content │
              └──────┬───────┘   └──────┬───────┘
                     │                  │
                     ▼                  ▼
              ┌──────────────┐   ┌──────────────┐
              │ Strip prompt │   │ Strip white- │
              │ from output  │   │ space        │
              └──────┬───────┘   └──────┬───────┘
                     │                  │
                     └────────┬─────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │  Return string  │
                    └─────────────────┘
```

### Lazy Loading Pattern

```
First Call:
  chat() → _chat_hf() → _load_hf() [loads model] → generate
  
Subsequent Calls:
  chat() → _chat_hf() → _load_hf() [returns immediately] → generate
```

---

## Testing Strategies

### Unit Tests

#### 1. Configuration Parsing Tests

```python
import pytest
from unittest.mock import Mock, patch
from llm_router import LLMRouter

class TestConfigurationParsing:
    
    def test_full_config_defaults_to_local(self):
        """Full config with 'models' key defaults to local."""
        config = {
            "models": {"llm_model": "test-model"},
            "llm": {"api_base": "http://test"}
        }
        
        router = LLMRouter(config)
        
        assert router.use_local == True
    
    def test_full_config_can_override_to_remote(self):
        """Full config can set use_local=False."""
        config = {
            "models": {"use_local": False},
            "llm": {"api_base": "http://test", "api_key": "key", "model": "gpt-4"}
        }
        
        router = LLMRouter(config)
        
        assert router.use_local == False
    
    def test_bare_config_defaults_to_remote(self):
        """Bare LLM config defaults to remote."""
        config = {
            "api_base": "http://test",
            "api_key": "key",
            "model": "gpt-4"
        }
        
        router = LLMRouter(config)
        
        assert router.use_local == False
    
    def test_empty_config_defaults_to_remote(self):
        """Empty config defaults to remote."""
        router = LLMRouter({})
        
        assert router.use_local == False
    
    def test_extracts_hf_settings(self):
        config = {
            "models": {
                "llm_model": "meta-llama/Llama-2-7b",
                "llm_device": "cuda",
                "llm_max_new_tokens": 512,
                "llm_temperature": 0.5
            }
        }
        
        router = LLMRouter(config)
        
        assert router.local_model_id == "meta-llama/Llama-2-7b"
        assert router.local_device == "cuda"
        assert router.local_max_tokens == 512
        assert router.local_temperature == 0.5
    
    def test_extracts_api_settings(self):
        config = {
            "llm": {
                "api_base": "https://api.example.com",
                "api_key": "sk-test",
                "model": "gpt-4",
                "temperature": 0.7,
                "max_tokens": 1024
            }
        }
        
        router = LLMRouter(config)
        
        assert router.api_base == "https://api.example.com"
        assert router.api_key == "sk-test"
        assert router.api_model == "gpt-4"
        assert router.api_temperature == 0.7
        assert router.api_max_tokens == 1024
```

#### 2. Backend Routing Tests

```python
class TestBackendRouting:
    
    @patch.object(LLMRouter, '_chat_hf')
    def test_routes_to_hf_when_local(self, mock_hf):
        mock_hf.return_value = "hf response"
        
        router = LLMRouter({"models": {"use_local": True, "llm_model": "test"}})
        result = router.chat([{"role": "user", "content": "test"}])
        
        mock_hf.assert_called_once()
        assert result == "hf response"
    
    @patch.object(LLMRouter, '_chat_openai')
    def test_routes_to_openai_when_remote(self, mock_openai):
        mock_openai.return_value = "openai response"
        
        router = LLMRouter({
            "models": {"use_local": False},
            "llm": {"api_base": "http://test", "api_key": "key", "model": "gpt-4"}
        })
        result = router.chat([{"role": "user", "content": "test"}])
        
        mock_openai.assert_called_once()
        assert result == "openai response"
```

#### 3. Message Conversion Tests

```python
class TestMessageConversion:
    
    def test_single_user_message(self):
        messages = [{"role": "user", "content": "Hello"}]
        
        prompt = LLMRouter._messages_to_prompt(messages)
        
        assert prompt == "USER: Hello\nASSISTANT:"
    
    def test_system_and_user_messages(self):
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"}
        ]
        
        prompt = LLMRouter._messages_to_prompt(messages)
        
        expected = "SYSTEM: You are helpful.\nUSER: Hello\nASSISTANT:"
        assert prompt == expected
    
    def test_multi_turn_conversation(self):
        messages = [
            {"role": "system", "content": "Be helpful."},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
            {"role": "user", "content": "How are you?"}
        ]
        
        prompt = LLMRouter._messages_to_prompt(messages)
        
        lines = prompt.split("\n")
        assert lines[0] == "SYSTEM: Be helpful."
        assert lines[1] == "USER: Hi"
        assert lines[2] == "ASSISTANT: Hello!"
        assert lines[3] == "USER: How are you?"
        assert lines[4] == "ASSISTANT:"
```

#### 4. Generate Method Tests

```python
class TestGenerateMethod:
    
    @patch.object(LLMRouter, 'chat')
    def test_generate_calls_chat(self, mock_chat):
        mock_chat.return_value = "response"
        
        router = LLMRouter({"api_base": "http://test", "api_key": "key", "model": "gpt-4"})
        result = router.generate("test prompt")
        
        mock_chat.assert_called_once()
        call_args = mock_chat.call_args
        messages = call_args[0][0]
        
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "test prompt"
    
    @patch.object(LLMRouter, 'chat')
    def test_generate_passes_overrides(self, mock_chat):
        mock_chat.return_value = "response"
        
        router = LLMRouter({"api_base": "http://test", "api_key": "key", "model": "gpt-4"})
        router.generate("test", max_tokens=100, temperature=0.5)
        
        call_kwargs = mock_chat.call_args[1]
        assert call_kwargs["max_tokens"] == 100
        assert call_kwargs["temperature"] == 0.5
```

#### 5. Lazy Loading Tests

```python
class TestLazyLoading:
    
    @patch('llm_router.OpenAI')
    def test_openai_client_lazy_loaded(self, mock_openai_class):
        router = LLMRouter({
            "api_base": "http://test",
            "api_key": "key",
            "model": "gpt-4"
        })
        
        # Client not created yet
        assert router._client is None
        mock_openai_class.assert_not_called()
        
        # Trigger loading
        router._load_openai()
        
        mock_openai_class.assert_called_once()
        assert router._client is not None
    
    @patch('llm_router.OpenAI')
    def test_openai_client_loaded_once(self, mock_openai_class):
        router = LLMRouter({
            "api_base": "http://test",
            "api_key": "key",
            "model": "gpt-4"
        })
        
        router._load_openai()
        router._load_openai()
        router._load_openai()
        
        # Only created once
        mock_openai_class.assert_called_once()
```

#### 6. Error Handling Tests

```python
class TestErrorHandling:
    
    def test_missing_hf_model_raises(self):
        router = LLMRouter({"models": {"use_local": True}})  # No llm_model
        
        with pytest.raises(ValueError, match="Missing models.llm_model"):
            router._load_hf()
    
    def test_missing_api_base_raises(self):
        router = LLMRouter({"api_key": "key", "model": "gpt-4"})  # No api_base
        
        with pytest.raises(ValueError, match="Missing llm.api_base"):
            router._load_openai()
    
    def test_missing_api_key_raises(self):
        router = LLMRouter({"api_base": "http://test", "model": "gpt-4"})  # No api_key
        
        with pytest.raises(ValueError, match="llm.api_key"):
            router._load_openai()
```

#### 7. Model Name Tests

```python
class TestModelName:
    
    def test_model_name_from_local(self):
        router = LLMRouter({
            "models": {"use_local": True, "llm_model": "meta-llama/Llama-2"}
        })
        
        assert router.model_name == "meta-llama/Llama-2"
    
    def test_model_name_from_api(self):
        router = LLMRouter({
            "api_base": "http://test",
            "api_key": "key",
            "model": "gpt-4o-mini"
        })
        
        assert router.model_name == "gpt-4o-mini"
    
    def test_model_name_fallback(self):
        router = LLMRouter({})
        
        assert router.model_name == "unknown-model"
```

### Test Commands

```bash
# Run all LLM router tests
pytest test_llm_router.py -v

# Run with coverage
pytest test_llm_router.py --cov=llm_router --cov-report=html

# Run specific test class
pytest test_llm_router.py::TestConfigurationParsing -v
```

---

## Recommendations and Improvements

### Critical Issues

#### 1. No Error Handling for API Calls

**Problem:** API calls can fail with various errors (network, rate limits, etc.).

**Recommendation:** Add comprehensive error handling:

```python
import time
from openai import APIError, RateLimitError

def _chat_openai(self, messages, **overrides):
    self._load_openai()
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            resp = self._client.chat.completions.create(...)
            return resp.choices[0].message.content.strip()
        except RateLimitError:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                raise
        except APIError as e:
            logger.error(f"OpenAI API error: {e}")
            raise
```

#### 2. No Response Validation

**Problem:** No validation that response contains expected data.

**Recommendation:** Add validation:

```python
def _chat_openai(self, messages, **overrides):
    # ... API call ...
    
    if not resp.choices:
        raise RuntimeError("No choices in API response")
    
    content = resp.choices[0].message.content
    if content is None:
        raise RuntimeError("Empty content in API response")
    
    return content.strip()
```

---

### High Priority Improvements

#### 3. Logging and Observability

**Problem:** No visibility into LLM operations.

**Recommendation:** Add structured logging:

```python
import logging
logger = logging.getLogger(__name__)

def chat(self, messages, **overrides):
    logger.info(
        "llm_request",
        extra={
            "backend": "local" if self.use_local else "api",
            "model": self.model_name,
            "num_messages": len(messages),
            "temperature": overrides.get("temperature"),
            "max_tokens": overrides.get("max_tokens"),
        }
    )
    
    start = time.time()
    try:
        result = self._chat_hf(messages, **overrides) if self.use_local else self._chat_openai(messages, **overrides)
        
        logger.info(
            "llm_response",
            extra={
                "elapsed_ms": (time.time() - start) * 1000,
                "response_length": len(result),
            }
        )
        return result
    except Exception as e:
        logger.error(f"llm_error: {e}")
        raise
```

#### 4. Streaming Support

**Problem:** No support for streaming responses.

**Recommendation:** Add streaming methods:

```python
def chat_stream(self, messages, **overrides):
    """Yield response chunks as they're generated."""
    if self.use_local:
        yield from self._chat_hf_stream(messages, **overrides)
    else:
        yield from self._chat_openai_stream(messages, **overrides)

def _chat_openai_stream(self, messages, **overrides):
    self._load_openai()
    
    stream = self._client.chat.completions.create(
        model=self.api_model,
        messages=messages,
        stream=True,
        **overrides
    )
    
    for chunk in stream:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content
```

#### 5. Token Counting

**Problem:** No way to count tokens before/after calls.

**Recommendation:** Add token utilities:

```python
def count_tokens(self, text: str) -> int:
    """Estimate token count for text."""
    if self.use_local and self._hf_tokenizer:
        return len(self._hf_tokenizer.encode(text))
    else:
        # Rough estimate for OpenAI models
        return len(text) // 4

def get_usage(self) -> dict:
    """Return token usage from last call."""
    return self._last_usage or {}
```

---

### Medium Priority Improvements

#### 6. Caching Support

**Problem:** Same prompts always hit the model.

**Recommendation:** Add optional caching:

```python
from functools import lru_cache
import hashlib

class LLMRouter:
    def __init__(self, config, enable_cache: bool = False):
        self.enable_cache = enable_cache
        self._cache = {}
    
    def chat(self, messages, **overrides):
        if self.enable_cache:
            cache_key = self._cache_key(messages, overrides)
            if cache_key in self._cache:
                return self._cache[cache_key]
        
        result = self._do_chat(messages, **overrides)
        
        if self.enable_cache:
            self._cache[cache_key] = result
        
        return result
```

#### 7. Model Fallback

**Problem:** No fallback if primary model fails.

**Recommendation:** Add fallback support:

```python
class LLMRouter:
    def __init__(self, config):
        self.fallback_model = config.get("llm", {}).get("fallback_model")
    
    def chat(self, messages, **overrides):
        try:
            return self._do_chat(messages, **overrides)
        except Exception as e:
            if self.fallback_model:
                logger.warning(f"Primary model failed, using fallback: {e}")
                return self._chat_with_model(self.fallback_model, messages, **overrides)
            raise
```

#### 8. Async Support

**Problem:** Synchronous only.

**Recommendation:** Add async methods:

```python
from openai import AsyncOpenAI

class LLMRouter:
    async def chat_async(self, messages, **overrides):
        if self.use_local:
            # Run in thread pool for HF
            return await asyncio.get_event_loop().run_in_executor(
                None, lambda: self._chat_hf(messages, **overrides)
            )
        else:
            return await self._chat_openai_async(messages, **overrides)
    
    async def _chat_openai_async(self, messages, **overrides):
        if self._async_client is None:
            self._async_client = AsyncOpenAI(base_url=self.api_base, api_key=self.api_key)
        
        resp = await self._async_client.chat.completions.create(...)
        return resp.choices[0].message.content.strip()
```

---

### Low Priority / Future Enhancements

#### 9. Multiple Model Support

**Recommendation:** Support multiple models:

```python
class LLMRouter:
    def chat(self, messages, model: str = None, **overrides):
        """Allow per-call model selection."""
        model = model or self.default_model
        # Route based on model
```

#### 10. Cost Tracking

**Recommendation:** Track API costs:

```python
class LLMRouter:
    COST_PER_1K_TOKENS = {
        "gpt-4": {"input": 0.03, "output": 0.06},
        "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    }
    
    def get_estimated_cost(self) -> float:
        """Return estimated cost from token usage."""
        usage = self._last_usage or {}
        rates = self.COST_PER_1K_TOKENS.get(self.api_model, {})
        
        input_cost = (usage.get("prompt_tokens", 0) / 1000) * rates.get("input", 0)
        output_cost = (usage.get("completion_tokens", 0) / 1000) * rates.get("output", 0)
        
        return input_cost + output_cost
```

#### 11. Model Capability Checking

**Recommendation:** Check model capabilities:

```python
class LLMRouter:
    def supports_function_calling(self) -> bool:
        """Check if model supports function calling."""
        return self.api_model in ["gpt-4", "gpt-4o-mini", "gpt-3.5-turbo"]
    
    def max_context_length(self) -> int:
        """Return model's max context length."""
        return self.MODEL_CONTEXT_LENGTHS.get(self.api_model, 4096)
```

---

## Usage Examples

### Basic Usage

```python
from llm_router import LLMRouter

# With OpenAI-compatible API
config = {
    "llm": {
        "api_base": "https://api.openai.com/v1",
        "api_key": "sk-...",
        "model": "gpt-4o-mini",
        "temperature": 0.2,
        "max_tokens": 256
    }
}

router = LLMRouter(config)

# Chat-style interaction
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"}
]

response = router.chat(messages)
print(response)  # "Paris is the capital of France."

# Simple text generation
response = router.generate(
    prompt="Explain quantum computing in one sentence.",
    max_tokens=50,
    temperature=0.3
)
```

### With Local HuggingFace Model

```python
config = {
    "models": {
        "use_local": True,
        "llm_model": "meta-llama/Llama-2-7b-chat-hf",
        "llm_device": "cuda",
        "llm_max_new_tokens": 256,
        "llm_temperature": 0.3
    }
}

router = LLMRouter(config)

# Same interface as API
response = router.chat([
    {"role": "user", "content": "Hello, how are you?"}
])
```

### With Parameter Overrides

```python
router = LLMRouter(config)

# Override defaults per call
response = router.chat(
    messages=[{"role": "user", "content": "Write a haiku."}],
    temperature=0.9,  # More creative
    max_tokens=50     # Shorter response
)
```

### Agent Integration

```python
class LLMQEAgent(QEAgent):
    def __init__(self, config: dict):
        self.router = LLMRouter(config)
        self.temperature = config.get("retrieval", {}).get("qe_temperature", 0.2)
    
    def expand(self, inp: QEInput) -> QEOutput:
        messages = [
            {"role": "system", "content": "Generate query paraphrases."},
            {"role": "user", "content": f"Query: {inp.query}"}
        ]
        
        raw = self.router.chat(messages, temperature=self.temperature)
        
        # Parse and return variants
        variants = [line.strip() for line in raw.splitlines() if line.strip()]
        return QEOutput(expanded_queries=variants)
```

---

## Appendix

### Glossary

| Term | Definition |
|------|------------|
| **HF** | HuggingFace |
| **Chat Messages** | List of role/content dicts in OpenAI format |
| **Lazy Loading** | Loading resources on first use |
| **Cross-Encoder** | Model that processes query+document together |

### Configuration Reference

| Config Path | Type | Default | Description |
|-------------|------|---------|-------------|
| `models.use_local` | bool | True (full) / False (bare) | Use local HF |
| `models.llm_model` | str | None | HF model ID |
| `models.llm_device` | str | Auto | cuda/cpu |
| `models.llm_max_new_tokens` | int | 256 | Max tokens (HF) |
| `models.llm_temperature` | float | 0.3 | Temperature (HF) |
| `llm.api_base` | str | None | API endpoint |
| `llm.api_key` | str | None | API key |
| `llm.model` | str | None | API model name |
| `llm.temperature` | float | 0.2 | Temperature (API) |
| `llm.max_tokens` | int | 256 | Max tokens (API) |

### Message Format

```python
# Standard OpenAI chat format
messages = [
    {"role": "system", "content": "System instructions"},
    {"role": "user", "content": "User message"},
    {"role": "assistant", "content": "Assistant response"},
    {"role": "user", "content": "Follow-up question"},
]
```

### Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | Initial | Dual backend support (HF + OpenAI) |

### References

- Radiant Repository: https://github.com/dshipley71/radiant
- Related Files: `generator_llm_agent.py`, `qe_llm_agent.py`, `rewrite_llm_agent.py`
- OpenAI API: https://platform.openai.com/docs/api-reference
- HuggingFace: https://huggingface.co/docs/transformers

---

*Document generated for the Radiant RAG Pipeline project. For questions or contributions, please refer to the project repository.*
