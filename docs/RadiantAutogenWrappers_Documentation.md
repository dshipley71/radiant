# RadiantAutogenWrappers Documentation

Technical reference for the Radiant-AutoGen wrapper module.

---

## Overview

The `radiant_autogen_wrappers.py` module provides a thin integration layer between Radiant's Agentic RAG pipeline and Microsoft AutoGen. It handles path resolution, initialization, and serialization to expose Radiant as an AutoGen-compatible tool.

**Module Location:** `radiant_autogen_wrappers.py` (project root)

---

## Module Components

### Path Resolution

```python
def _setup_radiant_path() -> None:
    """
    Find and add Radiant package directory to sys.path.
    
    Search order:
      1. RADIANT_PATH environment variable
      2. Same directory as this file
      3. ./radiant/ subdirectory
      4. /content/radiant/ (Colab)
      5. Current working directory
    """
```

**Detection Logic:**
- Looks for directories containing both `core/` and `agents/` subdirectories
- Adds found path to `sys.path[0]` for import priority
- Raises `ImportError` with helpful message if not found

### Config Resolution

```python
def _find_config_path() -> Optional[str]:
    """
    Locate config.fast.yaml automatically.
    
    Search order:
      1. AGENTIC_RAG_CONFIG environment variable
      2. ./config.fast.yaml (current directory)
      3. ../config.fast.yaml (parent directory)
      4. Same directory as module
      5. Inside radiant package directory
    """
```

### Initialization

```python
_RADIANT_INITIALIZED: bool = False

def _ensure_radiant_initialized(config_path: Optional[str] = None) -> None:
    """
    Initialize Radiant's agent registry (singleton pattern).
    
    - Only runs once per process
    - Auto-detects config if not provided
    - Sets AGENTIC_RAG_CONFIG env var for orchestrator
    """
```

### Serialization

```python
def _to_serializable(obj: Any) -> Any:
    """
    Convert Pydantic models and nested structures to JSON-serializable dicts.
    
    Handles:
      - Pydantic v1 (.dict()) and v2 (.model_dump())
      - Nested dicts, lists, tuples, sets
      - Primitive types (passthrough)
    """
```

---

## Main Tool Function

### radiant_tool()

```python
def radiant_tool(
    query: str,
    config_path: Optional[str] = None,
    history: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, Any]:
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | `str` | required | User query for RAG pipeline |
| `config_path` | `Optional[str]` | `None` | Override config file path |
| `history` | `Optional[List[Dict]]` | `None` | Conversation history |

**History Format:**
```python
[
    {"role": "user", "content": "First question"},
    {"role": "assistant", "content": "First answer"},
    {"role": "user", "content": "Follow-up question"},
    {"role": "assistant", "content": "Follow-up answer"},
]
```

**Return Value:**
```python
{
    "answer_text": str | None,
    "meta": Dict[str, Any]
}
```

**Answer Text Resolution:**
1. Try `meta["postprocess"].final_text`
2. Fallback to `meta["answer"].text`
3. Return `None` if both fail

---

## Usage Examples

### Basic Query

```python
from radiant_autogen_wrappers import radiant_tool

result = radiant_tool(query="What is hierarchical RAG?")

if result["answer_text"]:
    print(result["answer_text"])
else:
    print("No answer generated")
```

### With Explicit Config

```python
result = radiant_tool(
    query="Explain MCP protocol",
    config_path="/path/to/my/config.fast.yaml"
)
```

### Multi-Turn Conversation

```python
history = []

# Turn 1
q1 = "What is RAG?"
r1 = radiant_tool(query=q1)
history.append({"role": "user", "content": q1})
history.append({"role": "assistant", "content": r1["answer_text"] or ""})

# Turn 2 (with context)
q2 = "What are its limitations?"
r2 = radiant_tool(query=q2, history=history)
```

### Accessing Metadata

```python
result = radiant_tool(query="What is RAG?")
meta = result["meta"]

# Router classification
print(f"Query type: {meta['router']['router_profile']['query_type']}")

# Retrieval results
for doc in meta["retrieval_results"][:3]:
    print(f"- {doc['doc_id']}: {doc['snippets'][0]['text'][:100]}...")

# Citations
for cite in meta["citations"]:
    print(f"Source: {cite['doc_title']}")
```

---

## Environment Variables

| Variable | Purpose | Example |
|----------|---------|---------|
| `RADIANT_PATH` | Radiant package directory | `/home/user/radiant` |
| `AGENTIC_RAG_CONFIG` | Config file path | `/home/user/radiant/config.fast.yaml` |
| `AGENTIC_LLM_MODEL` | Override LLM model | `gpt-4o-mini` |
| `AGENTIC_LLM_API_BASE` | Override LLM API base | `https://api.openai.com/v1` |
| `AGENTIC_LLM_API_KEY` | Override LLM API key | `sk-...` |

---

## AutoGen Compatibility

The `radiant_tool` function is designed to work as an AutoGen `FunctionTool`:

```python
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient

model_client = OpenAIChatCompletionClient(model="gpt-4o-mini", api_key="...")

agent = AssistantAgent(
    name="radiant_agent",
    model_client=model_client,
    tools=[radiant_tool],  # Direct function reference
    system_message="Use radiant_tool to answer questions."
)
```

**Generated Tool Schema:**
```json
{
  "name": "radiant_tool",
  "parameters": {
    "type": "object",
    "properties": {
      "query": {"type": "string"},
      "config_path": {"type": "string", "nullable": true},
      "history": {
        "type": "array",
        "items": {"type": "object"},
        "nullable": true
      }
    },
    "required": ["query"]
  }
}
```

---

## Error Handling

### Import Errors

```python
# If Radiant not found
ImportError: Could not find Radiant package directory (containing 'core/' and 'agents/').
Either:
  1. Place this file inside the radiant directory, or
  2. Set RADIANT_PATH environment variable, or
  3. Ensure ./radiant/ exists in current directory
```

**Solution:** Set `RADIANT_PATH` or move wrapper file.

### Config Errors

```python
# If config not found
ValueError: Missing llm.api_base or llm.api_key for OpenAI-compatible mode.
```

**Solution:** Set `AGENTIC_RAG_CONFIG` or configure LLM in config.fast.yaml.

### Runtime Errors

The tool catches and returns errors gracefully:

```python
result = radiant_tool(query="...")
if result["answer_text"] is None:
    # Check meta for error information
    guardrail = result["meta"].get("guardrail", {})
    if guardrail.get("blocked"):
        print(f"Query blocked: {guardrail.get('reason')}")
```

---

## Internals

### Initialization Flow

```
radiant_tool() called
        │
        ▼
_ensure_radiant_initialized()
        │
        ├─► Already initialized? → Skip
        │
        ▼
_find_config_path()
        │
        ▼
register_default_agents(config_path)
        │
        ▼
_RADIANT_INITIALIZED = True
```

### Execution Flow

```
radiant_tool(query, config_path, history)
        │
        ▼
_ensure_radiant_initialized(config_path)
        │
        ▼
agentic_once_with_metadata(query, history)
        │
        ▼
Extract answer_text from postprocess/answer
        │
        ▼
_to_serializable(meta)
        │
        ▼
Return {"answer_text": ..., "meta": ...}
```

---

## Related Documentation

- [AutoGenIntegration_Documentation.md](AutoGenIntegration_Documentation.md) - Integration overview
- [RadiantManagerAgent_Documentation.md](RadiantManagerAgent_Documentation.md) - AutoGen agent class
- [Orchestrator_Documentation.md](Orchestrator_Documentation.md) - Pipeline internals
