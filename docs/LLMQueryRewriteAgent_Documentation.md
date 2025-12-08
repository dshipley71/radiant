# LLMQueryRewriteAgent Documentation

## Technical Reference for the Radiant RAG Pipeline Query Refinement

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture Context](#architecture-context)
3. [Class Structure](#class-structure)
4. [Core Functionality](#core-functionality)
5. [Prompt Engineering](#prompt-engineering)
6. [Configuration System](#configuration-system)
7. [Data Flow](#data-flow)
8. [Testing Strategies](#testing-strategies)
9. [Recommendations and Improvements](#recommendations-and-improvements)
10. [Usage Examples](#usage-examples)

---

## Overview

### Purpose

The `LLMQueryRewriteAgent` is the query refinement component within the Radiant RAG pipeline. It uses an LLM to rewrite queries based on critic feedback, improving retrieval coverage and reducing hallucination risk in subsequent iterations.

### Key Responsibilities

- Analyze critic feedback (missing topics, ambiguities, notes)
- Construct context-aware rewrite prompts
- Generate improved query versions via LLM
- Preserve original query intent while addressing feedback

### Design Philosophy

The agent implements a **feedback-driven refinement** approach where critic insights directly inform query modifications. This creates a closed-loop improvement cycle within the RAG pipeline, allowing the system to learn from its own evaluation and self-correct.

---

## Architecture Context

### Position in the Radiant Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    Query Rewrite Loop                           │
└─────────────────────────────────────────────────────────────────┘

     Original Query
          │
          ▼
    ┌───────────┐
    │ Retriever │ ─────────────────────────────────────┐
    └─────┬─────┘                                      │
          │                                            │
          ▼                                            │
    ┌───────────┐                                      │
    │ Generator │                                      │
    └─────┬─────┘                                      │
          │                                            │
          ▼                                            │
    ┌───────────┐                                      │
    │  Critic   │ ─────────┐                           │
    └─────┬─────┘          │                           │
          │                │                           │
          ▼                ▼                           │
    ┌───────────┐    ┌─────────────┐                   │
    │  Policy   │───▶│ LLMQueryRe- │                   │
    │ (REWRITE) │    │ writeAgent  │                   │
    └───────────┘    └──────┬──────┘                   │
                           │                           │
                           │ rewritten_query           │
                           │                           │
                           └───────────────────────────┘
                                    (back to Retriever)
```

### Query Evolution Flow

```
┌─────────────────────────────────────────────────────────────────┐
│ Iteration 0                                                     │
│   Query: "ML best practices"                                    │
│   Critic: "Low coverage, missing topics: model selection"       │
├─────────────────────────────────────────────────────────────────┤
│ Iteration 1 (After Rewrite)                                     │
│   Query: "machine learning best practices for model selection   │
│           and training optimization"                            │
│   Critic: "Improved coverage, some ambiguity in 'optimization'" │
├─────────────────────────────────────────────────────────────────┤
│ Iteration 2 (After Rewrite)                                     │
│   Query: "machine learning best practices including model       │
│           selection criteria and hyperparameter tuning"         │
│   Critic: "Coverage sufficient, risk acceptable"                │
│   Policy: FINALIZE                                              │
└─────────────────────────────────────────────────────────────────┘
```

### Related Components

| Component | Relationship |
|-----------|--------------|
| `QueryRewriteAgent` | Abstract base class (from `core.interfaces`) |
| `QueryRewriteInput` | Input schema with queries and critic feedback |
| `QueryRewriteOutput` | Output schema with rewritten query |
| `LLMRouter` | LLM backend abstraction |
| `CriticOutput` | Feedback from BasicCriticAgent |
| `BasicPolicyAgent` | Triggers rewrite via REWRITE decision |

---

## Class Structure

### Inheritance

```python
class LLMQueryRewriteAgent(QueryRewriteAgent):
    """LLM-backed QueryRewriteAgent that uses critic feedback to refine queries."""
```

### Class Attributes

| Attribute | Type | Value | Description |
|-----------|------|-------|-------------|
| `role` | `str` | `"rewrite"` | Agent role identifier |

### Instance Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `cfg` | `dict` | Full configuration dictionary |
| `router` | `LLMRouter` | LLM backend router |
| `temperature` | `float` | LLM temperature setting |
| `max_new_tokens` | `int` | Max tokens for generation |

### Constructor

```python
def __init__(self, config: dict)
```

**Initialization:**
1. Store configuration dictionary
2. Create LLMRouter instance
3. Extract rewrite-specific settings from `config.retrieval.query_rewrite`

### Methods

| Method | Visibility | Purpose |
|--------|------------|---------|
| `name` | Property | Returns agent name |
| `describe()` | Public | Returns agent description |
| `rewrite(inp)` | Public | Main rewrite method |

---

## Core Functionality

### The `rewrite()` Method

Primary method that generates improved query versions.

**Signature:**
```python
def rewrite(self, inp: QueryRewriteInput) -> QueryRewriteOutput
```

**Parameters:**
- `inp` (`QueryRewriteInput`): Contains original query, current query, and critic feedback

**Returns:**
- `QueryRewriteOutput`: Contains rewritten query and notes

**Processing Steps:**

1. **Extract Critic Feedback**
   - Get critic feedback object from input

2. **Build Critic Summary**
   - Collect missing topics (if any)
   - Collect ambiguities (if any)
   - Collect general notes (if any)
   - Join into summary string

3. **Construct Prompt**
   - System prompt with rewrite instructions
   - User prompt with original query, current query, feedback

4. **Call LLM**
   - Use `router.chat()` with configured parameters
   - Strip whitespace from response

5. **Return Output**
   - Package rewritten query
   - Include critic summary in notes

### Critic Feedback Processing

```python
# Build feedback summary from critic output
notes_parts: List[str] = []

if cf.missing_topics:
    notes_parts.append("Missing topics: " + "; ".join(cf.missing_topics))

if cf.ambiguities:
    notes_parts.append("Ambiguities: " + "; ".join(cf.ambiguities))

if cf.notes:
    notes_parts.append("Critic notes: " + "; ".join(cf.notes))

critic_summary = "\n".join(notes_parts) if notes_parts else "No specific notes."
```

### Feedback Types and Their Impact

| Feedback Type | Example | Rewrite Impact |
|---------------|---------|----------------|
| `missing_topics` | `["model selection", "hyperparameters"]` | Add specific terms |
| `ambiguities` | `["optimization unclear"]` | Clarify intent |
| `notes` | `["Answer very short"]` | Broaden scope |

---

## Prompt Engineering

### System Prompt

```
You are a query rewriting assistant for a retrieval-augmented system.
Your goal is to rewrite the user's query so that retrieval gets better coverage
and reduces hallucination risk, based on the critic feedback.
Preserve the original intent and constraints; do NOT introduce new facts.
```

**Key Instructions:**
1. **Better coverage** - Improve retrieval recall
2. **Reduce hallucination risk** - More specific queries
3. **Preserve intent** - No semantic drift
4. **No new facts** - Only reformulate, don't add assumptions

### User Prompt

```
Original query: {original_query}
Current query: {current_query}
Critic feedback summary:
{critic_summary}

Rewrite the query to better target the user's intent and address the issues above.
Return ONLY the rewritten query, with no explanation or commentary.
```

### Dual Query Context

The prompt includes both:
- **Original query**: The user's initial question (preserved intent)
- **Current query**: The most recent version (may differ after previous rewrites)

This allows the LLM to:
1. Understand the original intent
2. See how the query has evolved
3. Avoid reverting beneficial changes

### Example Transformation

**Input:**
```
Original query: ML best practices
Current query: ML best practices
Critic feedback summary:
Missing topics: model selection; training optimization
Critic notes: Low coverage of available context
```

**Output:**
```
machine learning best practices for model selection and training optimization techniques
```

---

## Configuration System

### Configuration Structure

```yaml
retrieval:
  query_rewrite:
    temperature: 0.2
    max_new_tokens: 128
```

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `temperature` | float | 0.2 | LLM creativity (0-1) |
| `max_new_tokens` | int | 128 | Max generation length |

### LLM Configuration (via LLMRouter)

The agent inherits LLM settings from the global config:

```yaml
llm:
  model: gpt-4o-mini
  api_base: https://api.openai.com/v1
  api_key: ${OPENAI_API_KEY}
```

### Temperature Guidance

| Temperature | Effect | Recommendation |
|-------------|--------|----------------|
| 0.0-0.2 | Minimal changes, focused | Best for query rewrite |
| 0.3-0.5 | Moderate creativity | May drift from intent |
| 0.6+ | High variance | Not recommended |

**Note:** Low temperature is preferred for query rewriting to minimize the risk of introducing unintended semantic changes.

---

## Data Flow

### Input Schema: `QueryRewriteInput`

```python
@dataclass
class QueryRewriteInput:
    original_query: str       # User's initial query
    current_query: str        # Query from previous iteration
    critic_feedback: CriticOutput  # Feedback from critic
```

### CriticOutput Fields Used

| Field | Type | Purpose |
|-------|------|---------|
| `missing_topics` | `List[str]` | Topics not covered |
| `ambiguities` | `List[str]` | Unclear aspects |
| `notes` | `List[str]` | General observations |

### Output Schema: `QueryRewriteOutput`

```python
@dataclass
class QueryRewriteOutput:
    rewritten_query: str    # Improved query
    notes: List[str]        # Processing notes (critic summary)
```

### Example Flow

**Input:**
```python
QueryRewriteInput(
    original_query="python async programming",
    current_query="python async programming concurrency",
    critic_feedback=CriticOutput(
        coverage_score=0.3,
        hallucination_risk=0.7,
        missing_topics=["asyncio", "event loops"],
        ambiguities=["concurrency type unclear"],
        notes=["Low coverage of available context"]
    )
)
```

**Output:**
```python
QueryRewriteOutput(
    rewritten_query="python asyncio asynchronous programming with event loops and concurrent execution",
    notes=["Missing topics: asyncio; event loops\nAmbiguities: concurrency type unclear\nCritic notes: Low coverage of available context"]
)
```

---

## Testing Strategies

### Unit Tests

#### 1. Basic Rewrite Tests

```python
import pytest
from unittest.mock import Mock, patch
from rewrite_llm_agent import LLMQueryRewriteAgent
from core.schemas import QueryRewriteInput, CriticOutput

@pytest.fixture
def agent():
    config = {
        "retrieval": {
            "query_rewrite": {
                "temperature": 0.2,
                "max_new_tokens": 128
            }
        }
    }
    with patch.object(LLMQueryRewriteAgent, '__init__', lambda self, cfg: None):
        agent = LLMQueryRewriteAgent.__new__(LLMQueryRewriteAgent)
        agent.cfg = config
        agent.router = Mock()
        agent.temperature = 0.2
        agent.max_new_tokens = 128
        return agent

@pytest.fixture
def make_input():
    def _make(
        original: str = "original query",
        current: str = "current query",
        missing_topics: list = None,
        ambiguities: list = None,
        notes: list = None
    ):
        critic_feedback = CriticOutput(
            coverage_score=0.5,
            hallucination_risk=0.5,
            missing_topics=missing_topics or [],
            ambiguities=ambiguities or [],
            unsupported_claims=[],
            notes=notes or []
        )
        return QueryRewriteInput(
            original_query=original,
            current_query=current,
            critic_feedback=critic_feedback
        )
    return _make

class TestBasicRewrite:
    
    def test_basic_rewrite(self, agent, make_input):
        agent.router.chat.return_value = "improved query version"
        
        inp = make_input()
        output = agent.rewrite(inp)
        
        assert output.rewritten_query == "improved query version"
        agent.router.chat.assert_called_once()
    
    def test_strips_whitespace(self, agent, make_input):
        agent.router.chat.return_value = "  rewritten query  \n"
        
        inp = make_input()
        output = agent.rewrite(inp)
        
        assert output.rewritten_query == "rewritten query"
    
    def test_passes_both_queries(self, agent, make_input):
        agent.router.chat.return_value = "result"
        
        inp = make_input(original="first query", current="modified query")
        agent.rewrite(inp)
        
        call_args = agent.router.chat.call_args
        messages = call_args[0][0]
        user_message = next(m for m in messages if m["role"] == "user")
        
        assert "first query" in user_message["content"]
        assert "modified query" in user_message["content"]
```

#### 2. Critic Feedback Processing Tests

```python
class TestCriticFeedbackProcessing:
    
    def test_includes_missing_topics(self, agent, make_input):
        agent.router.chat.return_value = "result"
        
        inp = make_input(missing_topics=["topic1", "topic2"])
        agent.rewrite(inp)
        
        call_args = agent.router.chat.call_args
        messages = call_args[0][0]
        user_message = next(m for m in messages if m["role"] == "user")
        
        assert "Missing topics: topic1; topic2" in user_message["content"]
    
    def test_includes_ambiguities(self, agent, make_input):
        agent.router.chat.return_value = "result"
        
        inp = make_input(ambiguities=["unclear term", "vague reference"])
        agent.rewrite(inp)
        
        call_args = agent.router.chat.call_args
        messages = call_args[0][0]
        user_message = next(m for m in messages if m["role"] == "user")
        
        assert "Ambiguities: unclear term; vague reference" in user_message["content"]
    
    def test_includes_notes(self, agent, make_input):
        agent.router.chat.return_value = "result"
        
        inp = make_input(notes=["Low coverage", "Short answer"])
        agent.rewrite(inp)
        
        call_args = agent.router.chat.call_args
        messages = call_args[0][0]
        user_message = next(m for m in messages if m["role"] == "user")
        
        assert "Critic notes: Low coverage; Short answer" in user_message["content"]
    
    def test_no_feedback_shows_no_specific_notes(self, agent, make_input):
        agent.router.chat.return_value = "result"
        
        inp = make_input()  # No feedback
        agent.rewrite(inp)
        
        call_args = agent.router.chat.call_args
        messages = call_args[0][0]
        user_message = next(m for m in messages if m["role"] == "user")
        
        assert "No specific notes" in user_message["content"]
    
    def test_combined_feedback(self, agent, make_input):
        agent.router.chat.return_value = "result"
        
        inp = make_input(
            missing_topics=["databases"],
            ambiguities=["SQL vs NoSQL"],
            notes=["Coverage low"]
        )
        output = agent.rewrite(inp)
        
        # Notes should contain the summary
        assert len(output.notes) == 1
        assert "databases" in output.notes[0]
        assert "SQL vs NoSQL" in output.notes[0]
        assert "Coverage low" in output.notes[0]
```

#### 3. LLM Interaction Tests

```python
class TestLLMInteraction:
    
    def test_uses_configured_temperature(self, agent, make_input):
        agent.temperature = 0.5
        agent.router.chat.return_value = "result"
        
        inp = make_input()
        agent.rewrite(inp)
        
        call_kwargs = agent.router.chat.call_args[1]
        assert call_kwargs["temperature"] == 0.5
    
    def test_uses_configured_max_tokens(self, agent, make_input):
        agent.max_new_tokens = 256
        agent.router.chat.return_value = "result"
        
        inp = make_input()
        agent.rewrite(inp)
        
        call_kwargs = agent.router.chat.call_args[1]
        assert call_kwargs["max_tokens"] == 256
    
    def test_system_prompt_content(self, agent, make_input):
        agent.router.chat.return_value = "result"
        
        inp = make_input()
        agent.rewrite(inp)
        
        call_args = agent.router.chat.call_args
        messages = call_args[0][0]
        system_message = next(m for m in messages if m["role"] == "system")
        
        assert "query rewriting assistant" in system_message["content"]
        assert "Preserve the original intent" in system_message["content"]
```

#### 4. Output Structure Tests

```python
class TestOutputStructure:
    
    def test_output_has_required_fields(self, agent, make_input):
        agent.router.chat.return_value = "rewritten"
        
        inp = make_input()
        output = agent.rewrite(inp)
        
        assert hasattr(output, 'rewritten_query')
        assert hasattr(output, 'notes')
    
    def test_notes_contains_summary(self, agent, make_input):
        agent.router.chat.return_value = "result"
        
        inp = make_input(notes=["Test note"])
        output = agent.rewrite(inp)
        
        assert len(output.notes) == 1
        assert "Test note" in output.notes[0]
    
    def test_empty_feedback_still_has_notes(self, agent, make_input):
        agent.router.chat.return_value = "result"
        
        inp = make_input()
        output = agent.rewrite(inp)
        
        # Should have "No specific notes" in notes
        assert len(output.notes) == 1
```

#### 5. Edge Case Tests

```python
class TestEdgeCases:
    
    def test_empty_query(self, agent, make_input):
        agent.router.chat.return_value = "clarified query"
        
        inp = make_input(original="", current="")
        output = agent.rewrite(inp)
        
        assert output.rewritten_query == "clarified query"
    
    def test_very_long_query(self, agent, make_input):
        agent.router.chat.return_value = "shortened query"
        
        long_query = "a " * 1000
        inp = make_input(original=long_query, current=long_query)
        output = agent.rewrite(inp)
        
        assert output.rewritten_query is not None
    
    def test_unicode_query(self, agent, make_input):
        agent.router.chat.return_value = "日本語クエリの書き換え"
        
        inp = make_input(original="日本語クエリ", current="日本語クエリ")
        output = agent.rewrite(inp)
        
        assert "日本語" in output.rewritten_query
    
    def test_special_characters_in_feedback(self, agent, make_input):
        agent.router.chat.return_value = "result"
        
        inp = make_input(notes=["Contains <html> & 'quotes'"])
        output = agent.rewrite(inp)
        
        assert "<html>" in output.notes[0]
    
    def test_same_original_and_current(self, agent, make_input):
        agent.router.chat.return_value = "result"
        
        inp = make_input(original="same query", current="same query")
        agent.rewrite(inp)
        
        # Should still process normally
        agent.router.chat.assert_called_once()
```

#### 6. Agent Interface Tests

```python
class TestAgentInterface:
    
    def test_name_property(self, agent):
        assert agent.name == "LLMQueryRewriteAgent"
    
    def test_describe_method(self, agent):
        description = agent.describe()
        assert isinstance(description, str)
        assert "rewrite" in description.lower() or "query" in description.lower()
    
    def test_role_attribute(self, agent):
        assert agent.role == "rewrite"
```

### Test Commands

```bash
# Run all rewrite tests
pytest test_rewrite_llm_agent.py -v

# Run with coverage
pytest test_rewrite_llm_agent.py --cov=rewrite_llm_agent --cov-report=html

# Run specific test class
pytest test_rewrite_llm_agent.py::TestCriticFeedbackProcessing -v
```

---

## Recommendations and Improvements

### Critical Issues

#### 1. No Error Handling for LLM Failures

**Problem:** LLM call failures will crash the pipeline.

**Recommendation:** Add error handling:

```python
def rewrite(self, inp: QueryRewriteInput) -> QueryRewriteOutput:
    try:
        rewritten = self.router.chat(messages, ...).strip()
    except Exception as e:
        logger.error(f"Query rewrite LLM call failed: {e}")
        # Fall back to current query
        return QueryRewriteOutput(
            rewritten_query=inp.current_query,
            notes=[f"Rewrite failed: {str(e)}"]
        )
```

#### 2. No Validation of Rewritten Query

**Problem:** LLM might return empty or invalid responses.

**Recommendation:** Add validation:

```python
def rewrite(self, inp: QueryRewriteInput) -> QueryRewriteOutput:
    rewritten = self.router.chat(messages, ...).strip()
    
    # Validate result
    if not rewritten or len(rewritten) < 3:
        logger.warning("Rewritten query too short, using current query")
        rewritten = inp.current_query
    
    # Check for obvious non-query responses
    if rewritten.lower().startswith(("i ", "here is", "the rewritten")):
        logger.warning("LLM returned explanation instead of query")
        rewritten = inp.current_query
    
    return QueryRewriteOutput(rewritten_query=rewritten, ...)
```

---

### High Priority Improvements

#### 3. Logging and Observability

**Problem:** No visibility into rewrite operations.

**Recommendation:** Add structured logging:

```python
import logging
logger = logging.getLogger(__name__)

def rewrite(self, inp: QueryRewriteInput) -> QueryRewriteOutput:
    logger.info(
        "rewrite_start",
        extra={
            "original_query": inp.original_query,
            "current_query": inp.current_query,
            "has_missing_topics": bool(inp.critic_feedback.missing_topics),
            "has_ambiguities": bool(inp.critic_feedback.ambiguities),
            "has_notes": bool(inp.critic_feedback.notes),
        }
    )
    
    # ... rewrite logic ...
    
    logger.info(
        "rewrite_complete",
        extra={
            "original_length": len(inp.current_query),
            "rewritten_length": len(rewritten),
            "changed": inp.current_query != rewritten,
        }
    )
```

#### 4. Caching Rewrite Results

**Problem:** Same query+feedback always calls LLM.

**Recommendation:** Add caching:

```python
from functools import lru_cache
import hashlib

def _cache_key(self, original: str, current: str, feedback_summary: str) -> str:
    content = f"{original}|{current}|{feedback_summary}"
    return hashlib.md5(content.encode()).hexdigest()

@lru_cache(maxsize=128)
def _rewrite_cached(self, cache_key: str, original: str, current: str, critic_summary: str) -> str:
    # ... LLM call ...
    return rewritten
```

#### 5. Rewrite Quality Metrics

**Problem:** No way to measure rewrite effectiveness.

**Recommendation:** Track metrics:

```python
@dataclass
class QueryRewriteOutput:
    rewritten_query: str
    notes: List[str]
    metrics: Optional[Dict[str, Any]] = None

def rewrite(self, inp: QueryRewriteInput) -> QueryRewriteOutput:
    # ... rewrite ...
    
    metrics = {
        "length_change": len(rewritten) - len(inp.current_query),
        "word_count_change": len(rewritten.split()) - len(inp.current_query.split()),
        "addressed_topics": self._count_addressed_topics(
            rewritten, 
            inp.critic_feedback.missing_topics
        ),
    }
    
    return QueryRewriteOutput(
        rewritten_query=rewritten,
        notes=notes,
        metrics=metrics
    )
```

---

### Medium Priority Improvements

#### 6. Configurable Prompt Templates

**Problem:** Prompts are hardcoded.

**Recommendation:** Make configurable:

```yaml
retrieval:
  query_rewrite:
    system_prompt: |
      You are a query rewriting assistant...
    user_prompt_template: |
      Original query: {original_query}
      Current query: {current_query}
      Feedback: {critic_summary}
      ...
```

#### 7. Rewrite Strategy Selection

**Problem:** Single rewrite strategy for all cases.

**Recommendation:** Support multiple strategies:

```python
class RewriteStrategy(Enum):
    EXPAND = "expand"        # Add terms for missing topics
    CLARIFY = "clarify"      # Resolve ambiguities
    FOCUS = "focus"          # Narrow scope
    REPHRASE = "rephrase"    # Different wording

def rewrite(self, inp: QueryRewriteInput, strategy: RewriteStrategy = None):
    # Auto-select strategy based on feedback
    if strategy is None:
        if inp.critic_feedback.missing_topics:
            strategy = RewriteStrategy.EXPAND
        elif inp.critic_feedback.ambiguities:
            strategy = RewriteStrategy.CLARIFY
        else:
            strategy = RewriteStrategy.REPHRASE
    
    # Use strategy-specific prompt
    system_prompt = self._get_strategy_prompt(strategy)
```

#### 8. Semantic Similarity Check

**Problem:** No guarantee rewrite preserves meaning.

**Recommendation:** Add similarity check:

```python
def rewrite(self, inp: QueryRewriteInput) -> QueryRewriteOutput:
    rewritten = self.router.chat(messages, ...).strip()
    
    # Check semantic similarity
    similarity = self._compute_similarity(inp.original_query, rewritten)
    
    if similarity < 0.5:
        logger.warning(f"Rewrite may have drifted (similarity={similarity:.2f})")
        # Optionally: retry or use original
    
    return QueryRewriteOutput(
        rewritten_query=rewritten,
        notes=[..., f"Semantic similarity: {similarity:.2f}"]
    )
```

---

### Low Priority / Future Enhancements

#### 9. Multi-Language Support

**Recommendation:** Language-aware rewriting:

```python
def rewrite(self, inp: QueryRewriteInput) -> QueryRewriteOutput:
    # Detect query language
    lang = self._detect_language(inp.original_query)
    
    # Use language-specific prompt
    system_prompt = self._get_localized_prompt(lang)
```

#### 10. Rewrite History Tracking

**Recommendation:** Track rewrite chain:

```python
@dataclass
class QueryRewriteOutput:
    rewritten_query: str
    notes: List[str]
    rewrite_history: List[str] = None  # Chain of rewrites

def rewrite(self, inp: QueryRewriteInput) -> QueryRewriteOutput:
    history = getattr(inp, 'rewrite_history', []) or []
    history.append(inp.current_query)
    
    # ... rewrite ...
    
    return QueryRewriteOutput(
        rewritten_query=rewritten,
        notes=notes,
        rewrite_history=history
    )
```

#### 11. A/B Testing Support

**Recommendation:** Support multiple rewrite variants:

```python
def rewrite_variants(self, inp: QueryRewriteInput, n: int = 3) -> List[QueryRewriteOutput]:
    """Generate multiple rewrite options for testing."""
    variants = []
    for _ in range(n):
        output = self.rewrite(inp)
        variants.append(output)
    return variants
```

---

## Usage Examples

### Basic Usage

```python
from rewrite_llm_agent import LLMQueryRewriteAgent
from core.schemas import QueryRewriteInput, CriticOutput

# Initialize with config
config = {
    "llm": {
        "model": "gpt-4o-mini",
        "api_base": "https://api.openai.com/v1",
        "api_key": "..."
    },
    "retrieval": {
        "query_rewrite": {
            "temperature": 0.2,
            "max_new_tokens": 128
        }
    }
}

agent = LLMQueryRewriteAgent(config)

# Create input with critic feedback
critic_feedback = CriticOutput(
    coverage_score=0.3,
    hallucination_risk=0.7,
    missing_topics=["model selection", "hyperparameter tuning"],
    ambiguities=[],
    unsupported_claims=[],
    notes=["Low coverage of available context"]
)

inp = QueryRewriteInput(
    original_query="ML best practices",
    current_query="ML best practices",
    critic_feedback=critic_feedback
)

# Rewrite
output = agent.rewrite(inp)

print(f"Original: {inp.original_query}")
print(f"Rewritten: {output.rewritten_query}")
print(f"Notes: {output.notes}")

# Output:
# Original: ML best practices
# Rewritten: machine learning best practices for model selection and hyperparameter tuning
# Notes: ['Missing topics: model selection; hyperparameter tuning\nCritic notes: Low coverage of available context']
```

### Pipeline Integration

```python
class RAGPipeline:
    def __init__(self, config: dict):
        self.retriever = HybridRetrievalAgent(...)
        self.generator = LLMGeneratorAgent(...)
        self.critic = BasicCriticAgent()
        self.policy = BasicPolicyAgent()
        self.rewriter = LLMQueryRewriteAgent(config)
    
    def process(self, query: str, plan: Plan) -> Dict[str, Any]:
        current_query = query
        
        for iteration in range(plan.iterations.max_iters):
            # Retrieve and generate
            results = self.retriever.retrieve(...)
            answer = self.generator.run(...)
            
            # Evaluate
            critic_output = self.critic.evaluate(...)
            
            # Decide
            policy_output = self.policy.decide(PolicyInput(
                critic_feedback=critic_output,
                plan=plan,
                iteration=iteration
            ))
            
            if policy_output.decision == DecisionEnum.FINALIZE:
                return {"answer": answer, "iterations": iteration + 1}
            
            elif policy_output.decision == DecisionEnum.REWRITE:
                # Rewrite query
                rewrite_output = self.rewriter.rewrite(QueryRewriteInput(
                    original_query=query,
                    current_query=current_query,
                    critic_feedback=critic_output
                ))
                current_query = rewrite_output.rewritten_query
            
            # CONTINUE uses same query
        
        return {"answer": answer, "iterations": plan.iterations.max_iters}
```

### With Custom Feedback

```python
# Manual feedback for testing
critic_feedback = CriticOutput(
    coverage_score=0.4,
    hallucination_risk=0.6,
    missing_topics=["async/await", "coroutines"],
    ambiguities=["concurrency model unclear"],
    unsupported_claims=["claim about performance"],
    notes=["Consider adding specific framework"]
)

inp = QueryRewriteInput(
    original_query="python async",
    current_query="python async programming",
    critic_feedback=critic_feedback
)

output = agent.rewrite(inp)
# Rewritten might be: "python asyncio async/await programming with coroutines for concurrent execution"
```

---

## Appendix

### Glossary

| Term | Definition |
|------|------------|
| **Query Rewrite** | Modifying a query to improve retrieval |
| **Critic Feedback** | Quality assessment from critic agent |
| **Missing Topics** | Concepts not covered in the answer |
| **Ambiguities** | Unclear aspects of the query |
| **Semantic Drift** | Unintended change in query meaning |

### Configuration Reference

| Config Path | Type | Default | Description |
|-------------|------|---------|-------------|
| `retrieval.query_rewrite.temperature` | float | 0.2 | LLM temperature |
| `retrieval.query_rewrite.max_new_tokens` | int | 128 | Max generation length |

### Feedback Types

| Type | Purpose | Example |
|------|---------|---------|
| `missing_topics` | Topics to add | `["databases", "indexing"]` |
| `ambiguities` | Issues to clarify | `["SQL vs NoSQL unclear"]` |
| `notes` | General observations | `["Coverage low"]` |

### Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | Initial | Basic LLM-based query rewriting |

### References

- Radiant Repository: https://github.com/dshipley71/radiant
- Related Files: `critic_basic_agent.py`, `policy_basic_agent.py`, `core/llm_router.py`

---

*Document generated for the Radiant RAG Pipeline project. For questions or contributions, please refer to the project repository.*
