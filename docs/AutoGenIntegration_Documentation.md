# AutoGen Integration Documentation

Technical reference for integrating Radiant with Microsoft AutoGen.

---

## Overview

The AutoGen integration provides a thin wrapper layer that exposes Radiant's Agentic RAG pipeline as an AutoGen-compatible tool. This allows Radiant to be used within AutoGen's multi-agent orchestration framework.

**Module Locations:**
- `radiant_autogen_wrappers.py` - Core wrapper and tool function
- `radiant_manager_agent.py` - AutoGen AssistantAgent subclass
- `test_radiant_autogen.py` - Test script and usage examples

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      AutoGen Framework                       │
│  ┌───────────────────────────────────────────────────────┐  │
│  │              RadiantManagerAgent                       │  │
│  │                 (AssistantAgent)                       │  │
│  └───────────────────────┬───────────────────────────────┘  │
│                          │                                   │
│                          ▼                                   │
│  ┌───────────────────────────────────────────────────────┐  │
│  │                   radiant_tool()                       │  │
│  │              (AutoGen FunctionTool)                    │  │
│  └───────────────────────┬───────────────────────────────┘  │
└──────────────────────────┼──────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│                    Radiant Pipeline                          │
│  ┌────────────────────────────────────────────────────────┐ │
│  │              agentic_once_with_metadata()              │ │
│  │                                                        │ │
│  │  Router → Planner → Retriever → Generator → Critic    │ │
│  └────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### Basic Usage (Without AutoGen)

```python
from radiant_autogen_wrappers import radiant_tool

# Single query
result = radiant_tool(query="What is hierarchical RAG?")
print(result["answer_text"])

# With conversation history
history = [
    {"role": "user", "content": "What is RAG?"},
    {"role": "assistant", "content": "RAG stands for Retrieval-Augmented Generation..."}
]
result = radiant_tool(query="How does it differ from fine-tuning?", history=history)
```

### With AutoGen Agent

```python
import asyncio
from autogen_ext.models.openai import OpenAIChatCompletionClient
from radiant_manager_agent import RadiantManagerAgent

async def main():
    model_client = OpenAIChatCompletionClient(
        model="gpt-4o-mini",
        api_key="sk-..."
    )
    
    agent = RadiantManagerAgent(model_client=model_client)
    result = await agent.run(task="What is hierarchical RAG?")
    print(result.messages[-1].content)

asyncio.run(main())
```

---

## Installation Requirements

### Python Dependencies

```bash
pip install autogen-agentchat autogen-ext openai tiktoken pydantic pyyaml
```

### Radiant Dependencies

Ensure all Radiant dependencies are installed (haystack, chromadb, sentence-transformers, etc.)

---

## Configuration

### Auto-Detection

The wrapper automatically detects:

1. **Radiant Package Location** (searched in order):
   - `RADIANT_PATH` environment variable
   - Same directory as the wrapper file
   - `./radiant/` subdirectory
   - `/content/radiant/` (Colab default)
   - Current working directory

2. **Config File Location** (searched in order):
   - `AGENTIC_RAG_CONFIG` environment variable
   - `./config.fast.yaml`
   - `../config.fast.yaml`
   - Same directory as module
   - Inside detected radiant directory

### Manual Configuration

```python
import os

# Set paths explicitly
os.environ["RADIANT_PATH"] = "/path/to/radiant"
os.environ["AGENTIC_RAG_CONFIG"] = "/path/to/config.fast.yaml"

# Or pass config_path directly
result = radiant_tool(query="...", config_path="/path/to/config.fast.yaml")
```

### LLM Configuration

The wrapper uses Radiant's LLM configuration from `config.fast.yaml`:

```yaml
llm:
  model:        "gpt-4o-mini"
  api_base:     "https://api.openai.com/v1"
  api_key:      "sk-your-key-here"
  temperature:  0.2
  max_tokens:   512
```

Or override via environment variables:

```python
os.environ["AGENTIC_LLM_MODEL"] = "gpt-4o-mini"
os.environ["AGENTIC_LLM_API_BASE"] = "https://api.openai.com/v1"
os.environ["AGENTIC_LLM_API_KEY"] = "sk-..."
```

---

## API Reference

### radiant_tool()

```python
def radiant_tool(
    query: str,
    config_path: Optional[str] = None,
    history: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, Any]:
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `query` | `str` | User query for the RAG pipeline |
| `config_path` | `Optional[str]` | Override path to config.fast.yaml |
| `history` | `Optional[List[Dict]]` | Conversation history for context |

**Returns:**

```python
{
    "answer_text": str | None,  # Final natural-language answer
    "meta": {                    # Serialized pipeline metadata
        "ctx": {...},            # Request context
        "router": {...},         # Router output
        "plan": {...},           # Execution plan
        "retrieval_results": [...],
        "answer": {...},
        "citations": [...],
        "critic": {...},
        "iterations": [...],
        ...
    }
}
```

### RadiantManagerAgent

```python
class RadiantManagerAgent(AssistantAgent):
    def __init__(
        self,
        model_client: OpenAIChatCompletionClient,
        *,
        name: str = "radiant_manager",
        model_client_stream: bool = True,
    ) -> None:
```

**Behavior:**
- Automatically calls `radiant_tool` for each user query
- Uses `answer_text` as the core reply
- Handles errors and guardrail rejections gracefully
- Supports conversation history for multi-turn interactions

---

## Error Handling

### Common Errors

| Error | Cause | Solution |
|-------|-------|----------|
| `ModuleNotFoundError: No module named 'core'` | Radiant not in path | Set `RADIANT_PATH` or place wrapper in radiant directory |
| `ValueError: Missing llm.api_base or llm.api_key` | LLM not configured | Set credentials in config.fast.yaml or env vars |
| `FileNotFoundError: config.fast.yaml` | Config not found | Set `AGENTIC_RAG_CONFIG` environment variable |

### Debugging

```python
# Check detected paths
from radiant_autogen_wrappers import _find_config_path, _setup_radiant_path
import sys

# View sys.path after setup
print(sys.path[:5])

# Check config path
print(_find_config_path())
```

---

## Multi-Turn Conversations

The wrapper supports conversation history for resolving references:

```python
history = []

# First turn
result1 = radiant_tool(query="What is RAG?")
history.append({"role": "user", "content": "What is RAG?"})
history.append({"role": "assistant", "content": result1["answer_text"]})

# Second turn (uses history to resolve "it")
result2 = radiant_tool(query="How does it handle context?", history=history)
```

---

## Extending the Integration

### Adding More Tools

You can expose finer-grained Radiant operations as separate tools:

```python
from core.orchestrator import REGISTRY

def router_tool(query: str) -> Dict[str, Any]:
    """Classify query type only."""
    router = REGISTRY.get("router")
    # ... implement

def retriever_tool(query: str, top_k: int = 10) -> List[Dict]:
    """Retrieve documents only."""
    retriever = REGISTRY.get("retriever")
    # ... implement
```

### Custom Agent Behaviors

```python
class CustomRadiantAgent(RadiantManagerAgent):
    def __init__(self, model_client, **kwargs):
        super().__init__(model_client, **kwargs)
        # Add custom tools
        self._tools.append(my_custom_tool)
```

---

## Testing

### Run Test Script

```bash
# With AutoGen (requires OPENAI_API_KEY)
export OPENAI_API_KEY="sk-..."
python test_radiant_autogen.py

# Direct test without AutoGen
python test_radiant_autogen.py --direct
```

### Unit Test Example

```python
def test_radiant_tool_basic():
    result = radiant_tool(query="What is hierarchical RAG?")
    assert "answer_text" in result
    assert "meta" in result
    assert result["answer_text"] is not None
```

---

## Related Documentation

- [Orchestrator_Documentation.md](Orchestrator_Documentation.md) - Pipeline internals
- [CoreSchemas_Documentation.md](CoreSchemas_Documentation.md) - Data models
- [LLMGeneratorAgent_Documentation.md](LLMGeneratorAgent_Documentation.md) - Answer generation
