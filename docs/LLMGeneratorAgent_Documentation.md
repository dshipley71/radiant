# LLMGeneratorAgent Documentation

## Technical Reference for the Radiant RAG Pipeline Response Generation

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture Context](#architecture-context)
3. [Class Structure](#class-structure)
4. [Core Functionality](#core-functionality)
5. [Configuration System](#configuration-system)
6. [Prompt Engineering](#prompt-engineering)
7. [Data Flow](#data-flow)
8. [Testing Strategies](#testing-strategies)
9. [Recommendations and Improvements](#recommendations-and-improvements)
10. [Usage Examples](#usage-examples)

---

## Overview

### Purpose

The `LLMGeneratorAgent` is the response generation component within the Radiant RAG pipeline. It takes retrieved and optionally reranked documents, constructs a context-aware prompt, and calls an LLM to generate the final answer for the user's query.

### Key Responsibilities

- Build cited context from retrieved documents with `[S1], [S2], ...` markers
- Construct RAG prompts with query, context, and optional query expansions
- Route LLM calls through the unified `LLMRouter` (supports OpenAI-compatible APIs and local models)
- Store answers and citations in pipeline state
- Emit telemetry events for monitoring and debugging

### Design Philosophy

The agent follows a **stateful execution model** where it reads from and writes to a shared state dictionary. This enables flexible pipeline composition while maintaining compatibility with both dict-based and object-based orchestration patterns.

---

## Architecture Context

### Position in the Radiant Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    Pipeline State                               │
│  query | reranked_documents | qe_variants | telemetry          │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    LLMGeneratorAgent                            │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  1. Extract Inputs                                      │   │
│  │     └─ query, docs, qe_variants from state              │   │
│  ├─────────────────────────────────────────────────────────┤   │
│  │  2. Build Cited Context                                 │   │
│  │     └─ [S1] snippet1, [S2] snippet2, ... (max chars)    │   │
│  ├─────────────────────────────────────────────────────────┤   │
│  │  3. Construct Prompt                                    │   │
│  │     └─ System instructions + query + context            │   │
│  ├─────────────────────────────────────────────────────────┤   │
│  │  4. LLM Generation                                      │   │
│  │     └─ LLMRouter.generate() → answer_text               │   │
│  ├─────────────────────────────────────────────────────────┤   │
│  │  5. Output & Telemetry                                  │   │
│  │     └─ Write answer, citations to state; emit events    │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Updated State                                │
│  + answer_text | answer (Answer object) | citations            │
└─────────────────────────────────────────────────────────────────┘
```

### Related Components

| Component | Relationship |
|-----------|--------------|
| `LLMRouter` | LLM routing abstraction (from `core.llm_router`) |
| `Answer` | Pydantic model for answer storage (from `core.schemas`) |
| `Document` | Haystack document type for retrieved content |
| `RAGGeneratorOutput` | Internal DTO for generation results |

---

## Class Structure

### Main Class: LLMGeneratorAgent

```python
class LLMGeneratorAgent:
    """LLM-backed RAG generator using LLMRouter."""
```

### Class Attributes

| Attribute | Type | Value | Description |
|-----------|------|-------|-------------|
| `role` | `str` | `"generator"` | Agent role identifier |

### Instance Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `name` | `str` | Agent instance name |
| `llm_cfg` | `Dict[str, Any]` | LLM configuration dictionary |

### Constructor

```python
def __init__(
    self,
    config_path: Optional[str] = "config/config.fast.yaml",
    config: Optional[Any] = None,
    name: str = "LLMGeneratorAgent",
) -> None
```

**Parameters:**
- `config_path`: Path to YAML configuration file
- `config`: Optional config object with `.llm` attribute
- `name`: Instance name for logging/telemetry

**Configuration Priority:**
1. `config.llm` (if provided as dict)
2. `_load_llm_config(config_path)` (fallback)

### Methods

| Method | Visibility | Purpose |
|--------|------------|---------|
| `describe()` | Public | Returns agent description |
| `run(state)` | Public | Main execution method (state-based) |
| `generate(input_obj)` | Public | Compatibility wrapper for orchestrator |

### Supporting Classes

#### RAGGeneratorOutput

```python
@dataclass
class RAGGeneratorOutput:
    answer_text: str                      # Generated answer
    refs: List[Tuple[int, str, str]]      # (n, human_title, snippet)
```

---

## Core Functionality

### The `run()` Method

Primary execution method that processes state and generates answers.

**Signature:**
```python
def run(self, state: Dict[str, Any]) -> Dict[str, Any]
```

**Processing Steps:**

1. **Extract Query**
   - Check `state["query"]` or `state["user_query"]`
   - Return early with empty answer if no query

2. **Get Documents**
   - Prefer `state["reranked_documents"]`
   - Fall back to `state["retrieved_documents"]`
   - Limit to top 6 documents

3. **Get Query Expansions**
   - Optional `state["qe_variants"]` for enhanced context

4. **Apply Config Overrides**
   - Merge `state["llm_config"]` or `state["llm_cfg"]`

5. **Generate Answer**
   - Call `_rag_answer()` with prepared inputs
   - Track generation time

6. **Update State**
   - Set `answer_text` (string)
   - Set `answer` (Answer object)
   - Set `citations` (list of refs)

7. **Emit Telemetry**
   - Record `generator.output` event with timing

### The `generate()` Method

Compatibility wrapper for different input formats.

**Input Handling:**

| Input Type | Processing |
|------------|------------|
| `dict` | Pass directly to `run()` |
| `dataclass` | Convert via `asdict()` |
| Object with attributes | Extract known attributes |
| Other | Try `__dict__` extraction |

**Output:** `SimpleNamespace` with all state attributes

---

## Configuration System

### Configuration File: `config.fast.yaml`

```yaml
llm:
  model: gpt-4o-mini
  api_base: https://api.openai.com/v1
  api_key: ${OPENAI_API_KEY}
  temperature: 0.2
  max_tokens: 512

retrieval:
  context_max_chars: 4000
```

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `str` | `None` | LLM model identifier |
| `api_base` | `str` | `None` | API endpoint URL |
| `api_key` | `str` | `None` | API authentication key |
| `temperature` | `float` | `0.2` | Generation temperature |
| `max_tokens` | `int` | `512` | Maximum output tokens |
| `context_max_chars` | `int` | `4000` | Max context length |

### Environment Variable Overrides

| Environment Variable | Config Key |
|---------------------|------------|
| `AGENTIC_LLM_MODEL` | `model` |
| `AGENTIC_LLM_API_BASE` | `api_base` |
| `AGENTIC_LLM_API_KEY` | `api_key` |
| `AGENTIC_LLM_TEMPERATURE` | `temperature` |

### Configuration Resolution Order

```
1. config.llm (passed object)
         │
         ▼
2. config.fast.yaml file
         │
         ▼
3. Environment variables (override)
         │
         ▼
4. Default values
```

---

## Prompt Engineering

### Prompt Structure

```
┌─────────────────────────────────────────────────────────────────┐
│ SYSTEM INSTRUCTIONS                                             │
│ - Use ONLY provided context                                     │
│ - Say explicitly if context insufficient                        │
│ - Do NOT include citation tags [S1], [S2] in answer             │
│ - Answer in natural language                                    │
├─────────────────────────────────────────────────────────────────┤
│ QUESTION                                                        │
│ {query}                                                         │
├─────────────────────────────────────────────────────────────────┤
│ QUERY EXPANSIONS (optional)                                     │
│ - expansion_1                                                   │
│ - expansion_2                                                   │
├─────────────────────────────────────────────────────────────────┤
│ CONTEXT                                                         │
│ [S1] First document snippet...                                  │
│                                                                 │
│ [S2] Second document snippet...                                 │
│                                                                 │
│ [S3] Third document snippet...                                  │
├─────────────────────────────────────────────────────────────────┤
│ Answer:                                                         │
└─────────────────────────────────────────────────────────────────┘
```

### Context Building Algorithm

```python
def _build_cited_context(docs: List[Document], max_chars: int = 4000):
    """
    1. For each document:
       a. Format title with page info
       b. Extract snippet content
       c. Tag as [S{n}] {snippet}
    2. Accumulate until max_chars reached
    3. Return context string + reference list
    """
```

**Truncation Behavior:**
- Stops adding documents when `max_chars` would be exceeded
- Never truncates mid-document
- Always includes at least one document (if available)

### Document Title Formatting

| Metadata | Output |
|----------|--------|
| `doc_title="Report"`, `page=5` | `"Report (p. 5)"` |
| `file_name="data.pdf"`, `page=[1,3]` | `"data.pdf (pp. 1–3)"` |
| No metadata | `"Document {n}"` |

---

## Data Flow

### Input State Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `query` / `user_query` | `str` | Yes | User's question |
| `reranked_documents` | `List[Document]` | Preferred | Reranked results |
| `retrieved_documents` | `List[Document]` | Fallback | Raw retrieval results |
| `qe_variants` | `List[str]` | No | Query expansions |
| `llm_config` / `llm_cfg` | `Dict` | No | Runtime config overrides |
| `telemetry` | `object` | No | Telemetry recorder |

### Output State Fields

| Field | Type | Description |
|-------|------|-------------|
| `answer_text` | `str` | Plain text answer |
| `answer` | `Answer` | Pydantic Answer model |
| `citations` | `List[Tuple]` | `(n, title, snippet)` references |

### Citation Format

```python
citations = [
    (1, "Document Title (p. 5)", "First 200 chars of snippet..."),
    (2, "Another Doc (pp. 1-3)", "First 200 chars..."),
    ...
]
```

---

## Testing Strategies

### Unit Tests

#### 1. Configuration Loading Tests

```python
import pytest
import os
import tempfile
from generator_llm_agent import _load_llm_config, _load_raw_config

class TestConfigLoading:
    
    def test_load_yaml_config(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
llm:
  model: gpt-4
  temperature: 0.5
  max_tokens: 1024
retrieval:
  context_max_chars: 8000
""")
        cfg = _load_llm_config(str(config_file))
        
        assert cfg["model"] == "gpt-4"
        assert cfg["temperature"] == 0.5
        assert cfg["max_tokens"] == 1024
        assert cfg["context_max_chars"] == 8000
    
    def test_missing_config_uses_defaults(self):
        cfg = _load_llm_config("/nonexistent/config.yaml")
        
        assert cfg["temperature"] == 0.2
        assert cfg["max_tokens"] == 512
        assert cfg["context_max_chars"] == 4000
    
    def test_env_var_override(self, monkeypatch):
        monkeypatch.setenv("AGENTIC_LLM_MODEL", "claude-3")
        monkeypatch.setenv("AGENTIC_LLM_TEMPERATURE", "0.7")
        
        cfg = _load_llm_config(None)
        
        assert cfg["model"] == "claude-3"
        assert cfg["temperature"] == 0.7
    
    def test_invalid_temperature_env_var(self, monkeypatch):
        monkeypatch.setenv("AGENTIC_LLM_TEMPERATURE", "invalid")
        
        cfg = _load_llm_config(None)
        
        # Should use default, not crash
        assert cfg["temperature"] == 0.2
```

#### 2. Context Building Tests

```python
from generator_llm_agent import _build_cited_context, _format_pages, _format_doc_title
from haystack import Document

class TestContextBuilding:
    
    def test_basic_context_building(self):
        docs = [
            Document(id="1", content="First snippet", meta={}),
            Document(id="2", content="Second snippet", meta={}),
        ]
        context, refs = _build_cited_context(docs, max_chars=1000)
        
        assert "[S1] First snippet" in context
        assert "[S2] Second snippet" in context
        assert len(refs) == 2
        assert refs[0][0] == 1
        assert refs[1][0] == 2
    
    def test_context_respects_max_chars(self):
        docs = [
            Document(id="1", content="A" * 500, meta={}),
            Document(id="2", content="B" * 500, meta={}),
            Document(id="3", content="C" * 500, meta={}),
        ]
        context, refs = _build_cited_context(docs, max_chars=600)
        
        # Should only include first doc
        assert "[S1]" in context
        assert "[S2]" not in context
        assert len(refs) == 1
    
    def test_empty_snippets_skipped(self):
        docs = [
            Document(id="1", content="", meta={}),
            Document(id="2", content="Valid content", meta={}),
        ]
        context, refs = _build_cited_context(docs, max_chars=1000)
        
        assert "[S1] Valid content" in context
        assert len(refs) == 1
    
    def test_format_pages_single(self):
        doc = Document(id="1", content="", meta={"page": 5})
        assert _format_pages(doc) == "p. 5"
    
    def test_format_pages_range(self):
        doc = Document(id="1", content="", meta={"page": [1, 5]})
        assert _format_pages(doc) == "pp. 1–5"
    
    def test_format_pages_none(self):
        doc = Document(id="1", content="", meta={})
        assert _format_pages(doc) == ""
    
    def test_format_doc_title_with_title(self):
        doc = Document(id="1", content="", meta={"doc_title": "Report", "page": 3})
        assert _format_doc_title(doc, 0) == "Report (p. 3)"
    
    def test_format_doc_title_fallback(self):
        doc = Document(id="1", content="", meta={})
        assert _format_doc_title(doc, 0) == "Document 1"
```

#### 3. Generator Agent Tests

```python
from generator_llm_agent import LLMGeneratorAgent, RAGGeneratorOutput
from unittest.mock import Mock, patch
from haystack import Document

class TestLLMGeneratorAgent:
    
    @pytest.fixture
    def agent(self):
        return LLMGeneratorAgent(config_path=None, name="TestGenerator")
    
    def test_empty_query_returns_empty_answer(self, agent):
        state = {"query": ""}
        result = agent.run(state)
        
        assert result["answer_text"] == ""
        assert result["answer"].text == ""
        assert result["citations"] == []
    
    def test_no_query_key_returns_empty_answer(self, agent):
        state = {}
        result = agent.run(state)
        
        assert result["answer_text"] == ""
    
    @patch("generator_llm_agent._resolve_llm")
    def test_successful_generation(self, mock_resolve, agent):
        mock_llm = Mock()
        mock_llm.generate.return_value = Mock(text="Generated answer")
        mock_llm.model_name = "test-model"
        mock_resolve.return_value = mock_llm
        
        state = {
            "query": "What is RAG?",
            "reranked_documents": [
                Document(id="1", content="RAG is retrieval augmented generation", meta={})
            ]
        }
        result = agent.run(state)
        
        assert result["answer_text"] == "Generated answer"
        assert result["answer"].text == "Generated answer"
        assert len(result["citations"]) == 1
    
    @patch("generator_llm_agent._resolve_llm")
    def test_uses_reranked_over_retrieved(self, mock_resolve, agent):
        mock_llm = Mock()
        mock_llm.generate.return_value = Mock(text="Answer")
        mock_llm.model_name = "test"
        mock_resolve.return_value = mock_llm
        
        state = {
            "query": "Test",
            "retrieved_documents": [Document(id="r1", content="Retrieved", meta={})],
            "reranked_documents": [Document(id="rr1", content="Reranked", meta={})],
        }
        agent.run(state)
        
        # Check the prompt passed to generate
        call_args = mock_llm.generate.call_args
        prompt = call_args.kwargs.get("prompt") or call_args[1].get("prompt", "")
        assert "Reranked" in prompt
    
    @patch("generator_llm_agent._resolve_llm")
    def test_limits_to_six_documents(self, mock_resolve, agent):
        mock_llm = Mock()
        mock_llm.generate.return_value = Mock(text="Answer")
        mock_llm.model_name = "test"
        mock_resolve.return_value = mock_llm
        
        state = {
            "query": "Test",
            "reranked_documents": [
                Document(id=f"d{i}", content=f"Doc {i}", meta={})
                for i in range(10)
            ]
        }
        agent.run(state)
        
        # Verify only 6 docs used
        call_args = mock_llm.generate.call_args
        prompt = call_args.kwargs.get("prompt") or call_args[1].get("prompt", "")
        assert "[S6]" in prompt
        assert "[S7]" not in prompt
    
    @patch("generator_llm_agent._resolve_llm")
    def test_config_override_at_runtime(self, mock_resolve, agent):
        mock_llm = Mock()
        mock_llm.generate.return_value = Mock(text="Answer")
        mock_llm.model_name = "test"
        mock_resolve.return_value = mock_llm
        
        state = {
            "query": "Test",
            "reranked_documents": [],
            "llm_config": {"temperature": 0.9}
        }
        agent.run(state)
        
        # Config should be merged
        call_args = mock_llm.generate.call_args
        temp = call_args.kwargs.get("temperature")
        assert temp == 0.9
    
    @patch("generator_llm_agent._resolve_llm")
    def test_telemetry_recording(self, mock_resolve, agent):
        mock_llm = Mock()
        mock_llm.generate.return_value = Mock(text="Answer")
        mock_llm.model_name = "test-model"
        mock_resolve.return_value = mock_llm
        
        mock_telemetry = Mock()
        state = {
            "query": "Test",
            "reranked_documents": [],
            "telemetry": mock_telemetry
        }
        agent.run(state)
        
        mock_telemetry.record_event.assert_called()
        call_kwargs = mock_telemetry.record_event.call_args.kwargs
        assert call_kwargs["agent"] == "TestGenerator"
        assert call_kwargs["event"] == "generator.output"
```

#### 4. Generate Method Compatibility Tests

```python
from dataclasses import dataclass
from types import SimpleNamespace

class TestGenerateCompatibility:
    
    @pytest.fixture
    def agent(self):
        return LLMGeneratorAgent(config_path=None)
    
    @patch("generator_llm_agent._resolve_llm")
    def test_dict_input(self, mock_resolve, agent):
        mock_llm = Mock()
        mock_llm.generate.return_value = Mock(text="Answer")
        mock_llm.model_name = "test"
        mock_resolve.return_value = mock_llm
        
        result = agent.generate({"query": "Test", "reranked_documents": []})
        
        assert isinstance(result, dict)
        assert "answer_text" in result
    
    @patch("generator_llm_agent._resolve_llm")
    def test_dataclass_input(self, mock_resolve, agent):
        mock_llm = Mock()
        mock_llm.generate.return_value = Mock(text="Answer")
        mock_llm.model_name = "test"
        mock_resolve.return_value = mock_llm
        
        @dataclass
        class GeneratorInput:
            query: str
            reranked_documents: list
        
        inp = GeneratorInput(query="Test", reranked_documents=[])
        result = agent.generate(inp)
        
        assert hasattr(result, "answer_text")
        assert result.answer_text == "Answer"
    
    @patch("generator_llm_agent._resolve_llm")
    def test_object_input(self, mock_resolve, agent):
        mock_llm = Mock()
        mock_llm.generate.return_value = Mock(text="Answer")
        mock_llm.model_name = "test"
        mock_resolve.return_value = mock_llm
        
        inp = SimpleNamespace(query="Test", reranked_documents=[])
        result = agent.generate(inp)
        
        assert hasattr(result, "answer_text")
```

### Test Commands

```bash
# Run all generator tests
pytest test_generator_llm_agent.py -v

# Run with coverage
pytest test_generator_llm_agent.py --cov=generator_llm_agent --cov-report=html

# Run specific test class
pytest test_generator_llm_agent.py::TestContextBuilding -v
```

---

## Recommendations and Improvements

### Critical Issues

#### 1. No Error Handling for LLM Failures

**Problem:** LLM generation failures will crash the pipeline.

**Recommendation:** Add error handling:

```python
def _rag_answer(query: str, docs: List[Document], llm_cfg: Dict[str, Any], ...) -> RAGGeneratorOutput:
    try:
        route = _resolve_llm(llm_cfg)
        completion = route.generate(...)
    except Exception as e:
        logger.error(f"LLM generation failed: {e}")
        return RAGGeneratorOutput(
            answer_text="I apologize, but I encountered an error generating a response. Please try again.",
            refs=refs,
        )
```

#### 2. Hardcoded Document Limit

**Problem:** Top 6 documents limit is hardcoded.

**Current:**
```python
docs[:6]  # top docs for generation
```

**Recommendation:** Make configurable:

```yaml
llm:
  generation_top_k: 6
```

```python
top_k = int(llm_cfg.get("generation_top_k", 6))
output = _rag_answer(query=query, docs=docs[:top_k], ...)
```

---

### High Priority Improvements

#### 3. Streaming Support

**Problem:** No support for streaming responses.

**Recommendation:** Add streaming capability:

```python
def run_streaming(self, state: Dict[str, Any]):
    """Generator that yields answer chunks."""
    # ... setup ...
    
    route = _resolve_llm(llm_cfg)
    for chunk in route.generate_stream(prompt=prompt, ...):
        yield chunk
    
    # Final state update after complete
    state["answer_text"] = full_answer
```

#### 4. Token Counting and Budget Management

**Problem:** No visibility into token usage or budget management.

**Recommendation:** Add token tracking:

```python
@dataclass
class RAGGeneratorOutput:
    answer_text: str
    refs: List[Tuple[int, str, str]]
    token_usage: Optional[Dict[str, int]] = None  # prompt_tokens, completion_tokens

def _rag_answer(...) -> RAGGeneratorOutput:
    # ... generation ...
    
    token_usage = None
    if hasattr(completion, "usage"):
        token_usage = {
            "prompt_tokens": completion.usage.prompt_tokens,
            "completion_tokens": completion.usage.completion_tokens,
        }
    
    return RAGGeneratorOutput(
        answer_text=answer_text,
        refs=refs,
        token_usage=token_usage,
    )
```

#### 5. Logging and Observability

**Problem:** Limited logging for debugging.

**Recommendation:** Add structured logging:

```python
import logging
logger = logging.getLogger(__name__)

def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
    logger.info(
        "generator_start",
        extra={
            "query_length": len(query),
            "num_docs": len(docs),
            "has_qe_variants": bool(qe_variants),
        }
    )
    
    # ... generation ...
    
    logger.info(
        "generator_complete",
        extra={
            "answer_length": len(output.answer_text),
            "num_citations": len(output.refs),
            "elapsed_ms": elapsed_ms,
        }
    )
```

#### 6. Prompt Template System

**Problem:** Prompt is hardcoded, difficult to customize.

**Recommendation:** Use configurable templates:

```yaml
llm:
  prompt_template: |
    You are a helpful AI assistant. Use ONLY the provided context...
    
    Question: {query}
    
    {qe_block}
    
    Context:
    {context}
    
    Answer:
```

```python
def _build_prompt(query: str, context: str, qe_block: str, template: str) -> str:
    return template.format(
        query=query,
        context=context,
        qe_block=qe_block,
    )
```

---

### Medium Priority Improvements

#### 7. Response Validation

**Problem:** No validation of LLM output quality.

**Recommendation:** Add response validation:

```python
def _validate_response(answer: str, refs: List, query: str) -> Tuple[bool, str]:
    """Validate response quality."""
    issues = []
    
    # Check minimum length
    if len(answer) < 10:
        issues.append("Response too short")
    
    # Check for citation tags that shouldn't be there
    if "[S" in answer and "]" in answer:
        issues.append("Response contains citation tags")
    
    # Check for refusal patterns
    refusal_patterns = ["I cannot", "I'm unable", "I don't have"]
    if any(p in answer for p in refusal_patterns):
        issues.append("Response may be a refusal")
    
    return len(issues) == 0, "; ".join(issues)
```

#### 8. Retry Logic

**Problem:** No retry on transient LLM failures.

**Recommendation:** Add retry with backoff:

```python
import time
from functools import wraps

def retry_with_backoff(max_retries: int = 3, backoff_factor: float = 2.0):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        wait = backoff_factor ** attempt
                        time.sleep(wait)
            raise last_exception
        return wrapper
    return decorator

@retry_with_backoff(max_retries=3)
def _rag_answer(...):
    # ... existing logic
```

#### 9. Answer Caching

**Problem:** Identical queries regenerate answers.

**Recommendation:** Add answer caching:

```python
from functools import lru_cache
import hashlib

def _make_cache_key(query: str, docs: List[Document]) -> str:
    doc_ids = sorted([d.id for d in docs])
    content = f"{query}|{','.join(doc_ids)}"
    return hashlib.md5(content.encode()).hexdigest()

class LLMGeneratorAgent:
    def __init__(self, ...):
        self._cache: Dict[str, RAGGeneratorOutput] = {}
    
    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        cache_key = _make_cache_key(query, docs)
        if cache_key in self._cache:
            output = self._cache[cache_key]
        else:
            output = _rag_answer(...)
            self._cache[cache_key] = output
        # ...
```

---

### Low Priority / Future Enhancements

#### 10. Multi-Turn Conversation Support

**Recommendation:** Add conversation history handling:

```python
def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
    conversation_history = state.get("conversation_history", [])
    
    # Include history in prompt
    history_block = self._format_history(conversation_history)
    
    # ... generation with history context
    
    # Update history
    state["conversation_history"] = conversation_history + [
        {"role": "user", "content": query},
        {"role": "assistant", "content": output.answer_text},
    ]
```

#### 11. Answer Quality Scoring

**Recommendation:** Add self-evaluation:

```python
def _score_answer(query: str, answer: str, context: str) -> float:
    """Use LLM to score answer quality."""
    scoring_prompt = f"""
    Rate the following answer on a scale of 1-10:
    
    Question: {query}
    Answer: {answer}
    
    Criteria:
    - Relevance to question
    - Accuracy based on context
    - Completeness
    
    Score (1-10):
    """
    # ... LLM call for scoring
```

#### 12. Citation Verification

**Recommendation:** Verify citations match content:

```python
def _verify_citations(answer: str, refs: List[Tuple]) -> List[str]:
    """Check that answer content aligns with cited sources."""
    issues = []
    # ... verification logic
    return issues
```

---

## Usage Examples

### Basic Usage

```python
from generator_llm_agent import LLMGeneratorAgent
from haystack import Document

# Initialize agent
agent = LLMGeneratorAgent(
    config_path="config/config.fast.yaml",
    name="MainGenerator"
)

# Prepare state
state = {
    "query": "What are the benefits of RAG?",
    "reranked_documents": [
        Document(
            id="1",
            content="RAG improves accuracy by grounding responses in retrieved documents.",
            meta={"doc_title": "RAG Overview", "page": 5}
        ),
        Document(
            id="2", 
            content="RAG reduces hallucinations by providing factual context.",
            meta={"doc_title": "RAG Benefits", "page": 12}
        ),
    ],
    "qe_variants": ["RAG advantages", "retrieval augmented generation benefits"]
}

# Generate
result = agent.run(state)

print(f"Answer: {result['answer_text']}")
print(f"Citations: {result['citations']}")
```

### Pipeline Integration

```python
class RAGPipeline:
    def __init__(self, config_path: str):
        self.retriever = HybridRetrievalAgent(config_path=config_path)
        self.reranker = RerankAgent()
        self.generator = LLMGeneratorAgent(config_path=config_path)
    
    def process(self, query: str, plan: Plan) -> Dict[str, Any]:
        # Step 1: Retrieve
        retriever_output = self.retriever.retrieve(RetrieverInput(
            query=query,
            plan=plan
        ))
        
        # Step 2: Flatten and rerank
        all_docs = []
        for result in retriever_output.results:
            for snippet in result.snippets:
                all_docs.append(Document(
                    id=snippet.chunk_id,
                    content=snippet.text,
                    score=snippet.score,
                    meta={}
                ))
        
        reranked = self.reranker.rerank(query, all_docs, top_k=plan.rerank_top_k)
        
        # Step 3: Generate
        state = {
            "query": query,
            "reranked_documents": reranked,
        }
        result = self.generator.run(state)
        
        return {
            "answer": result["answer_text"],
            "citations": result["citations"],
        }
```

### With Telemetry

```python
from telemetry_basic_agent import TelemetryAgent

# Create telemetry
telemetry = TelemetryAgent()

# Add to state
state = {
    "query": "What is machine learning?",
    "reranked_documents": documents,
    "telemetry": telemetry,
}

# Run with telemetry
result = agent.run(state)

# View metrics
print(telemetry.get_summary())
```

### Runtime Configuration Override

```python
state = {
    "query": "Explain quantum computing",
    "reranked_documents": documents,
    "llm_config": {
        "temperature": 0.8,      # More creative
        "max_tokens": 1024,      # Longer response
        "model": "gpt-4"         # Different model
    }
}

result = agent.run(state)
```

---

## Appendix

### Glossary

| Term | Definition |
|------|------------|
| **RAG** | Retrieval-Augmented Generation |
| **Context Window** | Maximum tokens LLM can process |
| **Citation** | Reference to source document |
| **QE Variants** | Query expansion alternatives |
| **Telemetry** | Performance monitoring data |

### Configuration Reference

| Config Path | Type | Default | Description |
|-------------|------|---------|-------------|
| `llm.model` | str | None | LLM model identifier |
| `llm.api_base` | str | None | API endpoint URL |
| `llm.api_key` | str | None | API authentication |
| `llm.temperature` | float | 0.2 | Generation randomness |
| `llm.max_tokens` | int | 512 | Max output tokens |
| `retrieval.context_max_chars` | int | 4000 | Max context length |

### State Field Reference

| Field | Direction | Type | Description |
|-------|-----------|------|-------------|
| `query` | Input | str | User query |
| `reranked_documents` | Input | List[Document] | Preferred docs |
| `retrieved_documents` | Input | List[Document] | Fallback docs |
| `qe_variants` | Input | List[str] | Query expansions |
| `llm_config` | Input | Dict | Runtime overrides |
| `telemetry` | Input | object | Telemetry recorder |
| `answer_text` | Output | str | Generated answer |
| `answer` | Output | Answer | Answer object |
| `citations` | Output | List[Tuple] | Source references |

### Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | Initial | Basic LLM generation with citations |

### References

- Radiant Repository: https://github.com/dshipley71/radiant
- Related Files: `core/llm_router.py`, `core/schemas.py`, `config.fast.yaml`

---

*Document generated for the Radiant RAG Pipeline project. For questions or contributions, please refer to the project repository.*
