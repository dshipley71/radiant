# LLMQEAgent Documentation

## Technical Reference for the Radiant RAG Pipeline Query Expansion

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

The `LLMQEAgent` implements Query Expansion (QE), a retrieval enhancement technique within the Radiant RAG pipeline. It uses an LLM to generate diverse paraphrases of user queries, improving recall by capturing synonyms, alternative phrasings, and complementary aspects that the original query may not express.

### Key Responsibilities

- Generate semantically equivalent query variants using LLM
- Respect plan-level enable/disable controls (`use_qe`)
- Honor configurable variant counts and generation parameters
- Clean and normalize LLM output into usable query list

### Design Philosophy

The agent follows a **controlled expansion** approach where the LLM is explicitly instructed to preserve meaning without introducing new constraints. This balances the benefit of vocabulary expansion with the risk of query drift.

---

## Architecture Context

### Position in the Radiant Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    Original Query                               │
│               "best practices for ML models"                    │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                       LLMQEAgent                                │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  1. Check plan.use_qe                                   │   │
│  │     └─ If False → return empty list                     │   │
│  ├─────────────────────────────────────────────────────────┤   │
│  │  2. Build Prompt                                        │   │
│  │     └─ System: Query expansion instructions             │   │
│  │     └─ User: Query + num_variants                       │   │
│  ├─────────────────────────────────────────────────────────┤   │
│  │  3. LLM Generation                                      │   │
│  │     └─ LLMRouter.chat() → raw text                      │   │
│  ├─────────────────────────────────────────────────────────┤   │
│  │  4. Output Parsing                                      │   │
│  │     └─ Split lines, strip bullets/numbers               │   │
│  │     └─ Limit to num_variants                            │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Expanded Queries                             │
│  1. "machine learning model best practices"                     │
│  2. "recommended approaches for ML systems"                     │
│  3. "optimal techniques for machine learning"                   │
│  4. "guidelines for building ML models"                         │
│  5. "effective strategies for ML development"                   │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
                    Dense Retrieval (multi-query)
```

### QE vs PRF Comparison

```
┌─────────────────────────────────────────────────────────────────┐
│                    Query Enhancement Methods                    │
├─────────────────────┬───────────────────────────────────────────┤
│       LLMQEAgent    │           BasicPRFAgent                   │
├─────────────────────┼───────────────────────────────────────────┤
│ Technique: LLM      │ Technique: BM25 + frequency analysis      │
│ paraphrasing        │                                           │
├─────────────────────┼───────────────────────────────────────────┤
│ When: Before        │ When: After initial retrieval             │
│ retrieval           │                                           │
├─────────────────────┼───────────────────────────────────────────┤
│ Output: Full query  │ Output: Term list + augmented query       │
│ variants            │                                           │
├─────────────────────┼───────────────────────────────────────────┤
│ Cost: LLM API call  │ Cost: Local BM25 computation              │
├─────────────────────┼───────────────────────────────────────────┤
│ Strength: Semantic  │ Strength: Corpus-specific vocabulary      │
│ understanding       │                                           │
└─────────────────────┴───────────────────────────────────────────┘
```

### Related Components

| Component | Relationship |
|-----------|--------------|
| `QEAgent` | Abstract base class (from `core.interfaces`) |
| `QEInput` | Input schema with query and plan |
| `QEOutput` | Output schema with expanded queries |
| `LLMRouter` | LLM backend abstraction (HF or OpenAI) |
| `Plan` | Execution plan with `use_qe` flag |

---

## Class Structure

### Inheritance

```python
class LLMQEAgent(QEAgent):
    """QEAgent that uses LLMRouter for query expansion."""
```

### Class Attributes

| Attribute | Type | Value | Description |
|-----------|------|-------|-------------|
| `role` | `str` | `"qe"` | Agent role identifier |

### Instance Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `cfg` | `dict` | Full configuration dictionary |
| `router` | `LLMRouter` | LLM backend router |
| `default_num_variants` | `int` | Default expansion count |
| `temperature` | `float` | LLM temperature setting |
| `max_new_tokens` | `int` | Max tokens for generation |

### Constructor

```python
def __init__(self, config: dict)
```

**Initialization:**
1. Store configuration dictionary
2. Create LLMRouter instance
3. Extract QE-specific settings from `config.retrieval.query_expansion`

### Methods

| Method | Visibility | Purpose |
|--------|------------|---------|
| `name` | Property | Returns agent name |
| `describe()` | Public | Returns agent description |
| `expand(inp)` | Public | Main expansion method |

---

## Core Functionality

### The `expand()` Method

Primary method that generates query variants.

**Signature:**
```python
def expand(self, inp: QEInput) -> QEOutput
```

**Parameters:**
- `inp` (`QEInput`): Contains query and execution plan

**Returns:**
- `QEOutput`: Contains list of expanded queries

**Processing Steps:**

1. **Check Enable Flag**
   - If `inp.plan.use_qe` is False, return empty list
   - Early exit saves LLM costs

2. **Determine Variant Count**
   - Priority: `plan.max_qe_variants` → `config.num_variants` → 5

3. **Build Prompt Messages**
   - System prompt with expansion instructions
   - User prompt with query and requested count

4. **Call LLM**
   - Use `router.chat()` with configured parameters
   - Returns raw text response

5. **Parse Output**
   - Split into lines
   - Strip whitespace, bullets, numbers
   - Filter empty lines
   - Limit to requested count

6. **Return Output**
   - Package as `QEOutput`

### Variant Count Resolution

```
┌─────────────────────────────────────────────────────────────────┐
│                    Variant Count Priority                       │
├─────────────────────────────────────────────────────────────────┤
│  1. inp.plan.max_qe_variants (if present and truthy)           │
│     │                                                           │
│     ▼                                                           │
│  2. config.retrieval.query_expansion.num_variants (if present) │
│     │                                                           │
│     ▼                                                           │
│  3. Default: 5                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Output Parsing Logic

```python
# Raw LLM output might include:
"""
1. machine learning model best practices
2. recommended approaches for ML systems
- optimal techniques for machine learning
• guidelines for building ML models
  effective strategies for ML development
"""

# Parsing process:
for line in raw.splitlines():
    line = line.strip()                    # Remove whitespace
    if not line:
        continue                            # Skip empty
    line = line.lstrip("-•*0123456789. ")  # Remove bullets/numbers
    line = line.strip()                    # Clean again
    if line:
        variants.append(line)

# Result:
["machine learning model best practices",
 "recommended approaches for ML systems",
 "optimal techniques for machine learning",
 "guidelines for building ML models",
 "effective strategies for ML development"]
```

---

## Prompt Engineering

### System Prompt

```
You are a query expansion assistant for a retrieval system.
Given a user query, generate several diverse paraphrases that preserve the
original meaning but use different wording or focus on complementary aspects.
Do NOT introduce new facts, constraints, or assumptions.
Return ONLY the paraphrased queries, one per line, with no bullets or numbering.
```

**Key Instructions:**
1. **Diverse paraphrases** - Different wording for vocabulary coverage
2. **Preserve meaning** - No semantic drift
3. **Complementary aspects** - Cover different angles
4. **No new constraints** - Prevent query narrowing
5. **Clean output** - One per line, no formatting

### User Prompt

```
Original query:
{query}

Number of paraphrases: {num_variants}
```

### Example Expansion

**Input Query:** "how to optimize database performance"

**Expected Output:**
```
improving database query efficiency
database performance tuning techniques
methods for faster database operations
optimizing SQL query execution speed
best practices for database optimization
```

---

## Configuration System

### Configuration Structure

```yaml
retrieval:
  query_expansion:
    num_variants: 5
    temperature: 0.2
    max_new_tokens: 64
```

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_variants` | int | 5 | Number of query variants |
| `temperature` | float | 0.2 | LLM creativity (0-1) |
| `max_new_tokens` | int | 64 | Max generation length |

### LLM Configuration (via LLMRouter)

The agent inherits LLM settings from the global config:

```yaml
llm:
  model: gpt-4o-mini
  api_base: https://api.openai.com/v1
  api_key: ${OPENAI_API_KEY}
```

### Temperature Guidance

| Temperature | Effect | Use Case |
|-------------|--------|----------|
| 0.0-0.2 | Conservative, focused | Precise paraphrasing |
| 0.3-0.5 | Balanced diversity | General QE |
| 0.6-0.8 | Creative variations | Exploratory queries |
| 0.9-1.0 | High variance | Not recommended |

---

## Data Flow

### Input Schema: `QEInput`

```python
@dataclass
class QEInput:
    query: str       # Original user query
    plan: Plan       # Execution plan with use_qe flag
```

### Plan Fields Used

| Field | Type | Purpose |
|-------|------|---------|
| `use_qe` | bool | Enable/disable expansion |
| `max_qe_variants` | int | Override variant count |

### Output Schema: `QEOutput`

```python
@dataclass
class QEOutput:
    expanded_queries: List[str]  # List of query variants
```

### Example Flow

**Input:**
```python
QEInput(
    query="python web frameworks comparison",
    plan=Plan(use_qe=True, max_qe_variants=3, ...)
)
```

**Output:**
```python
QEOutput(
    expanded_queries=[
        "comparing python web development frameworks",
        "python framework comparison for web applications",
        "best python frameworks for web development compared"
    ]
)
```

---

## Testing Strategies

### Unit Tests

#### 1. Enable/Disable Tests

```python
import pytest
from unittest.mock import Mock, patch
from qe_llm_agent import LLMQEAgent
from core.schemas import QEInput, Plan

@pytest.fixture
def agent():
    config = {
        "retrieval": {
            "query_expansion": {
                "num_variants": 5,
                "temperature": 0.2,
                "max_new_tokens": 64
            }
        }
    }
    with patch.object(LLMQEAgent, '__init__', lambda self, cfg: None):
        agent = LLMQEAgent.__new__(LLMQEAgent)
        agent.cfg = config
        agent.router = Mock()
        agent.default_num_variants = 5
        agent.temperature = 0.2
        agent.max_new_tokens = 64
        return agent

class TestEnableDisable:
    
    def test_disabled_returns_empty(self, agent):
        plan = Mock()
        plan.use_qe = False
        
        inp = QEInput(query="test query", plan=plan)
        output = agent.expand(inp)
        
        assert output.expanded_queries == []
        agent.router.chat.assert_not_called()
    
    def test_enabled_calls_llm(self, agent):
        plan = Mock()
        plan.use_qe = True
        plan.max_qe_variants = None
        
        agent.router.chat.return_value = "variant 1\nvariant 2"
        
        inp = QEInput(query="test query", plan=plan)
        output = agent.expand(inp)
        
        agent.router.chat.assert_called_once()
        assert len(output.expanded_queries) == 2
    
    def test_missing_use_qe_defaults_false(self, agent):
        plan = Mock(spec=[])  # No use_qe attribute
        
        inp = QEInput(query="test query", plan=plan)
        output = agent.expand(inp)
        
        assert output.expanded_queries == []
```

#### 2. Variant Count Tests

```python
class TestVariantCount:
    
    def test_plan_variants_override(self, agent):
        plan = Mock()
        plan.use_qe = True
        plan.max_qe_variants = 3
        
        agent.router.chat.return_value = "v1\nv2\nv3\nv4\nv5"
        
        inp = QEInput(query="test", plan=plan)
        output = agent.expand(inp)
        
        # Should be limited to 3
        assert len(output.expanded_queries) == 3
    
    def test_config_variants_used_when_no_plan_override(self, agent):
        plan = Mock()
        plan.use_qe = True
        plan.max_qe_variants = None
        
        agent.default_num_variants = 4
        agent.router.chat.return_value = "v1\nv2\nv3\nv4\nv5"
        
        inp = QEInput(query="test", plan=plan)
        output = agent.expand(inp)
        
        assert len(output.expanded_queries) == 4
    
    def test_default_five_variants(self, agent):
        plan = Mock()
        plan.use_qe = True
        plan.max_qe_variants = None
        
        agent.default_num_variants = None  # Force default
        agent.router.chat.return_value = "\n".join([f"v{i}" for i in range(10)])
        
        inp = QEInput(query="test", plan=plan)
        output = agent.expand(inp)
        
        assert len(output.expanded_queries) == 5
```

#### 3. Output Parsing Tests

```python
class TestOutputParsing:
    
    def test_strips_whitespace(self, agent):
        plan = Mock(use_qe=True, max_qe_variants=5)
        agent.router.chat.return_value = "  variant 1  \n  variant 2  "
        
        inp = QEInput(query="test", plan=plan)
        output = agent.expand(inp)
        
        assert output.expanded_queries[0] == "variant 1"
        assert output.expanded_queries[1] == "variant 2"
    
    def test_strips_numbered_lists(self, agent):
        plan = Mock(use_qe=True, max_qe_variants=5)
        agent.router.chat.return_value = "1. variant one\n2. variant two\n3. variant three"
        
        inp = QEInput(query="test", plan=plan)
        output = agent.expand(inp)
        
        assert "1." not in output.expanded_queries[0]
        assert output.expanded_queries[0] == "variant one"
    
    def test_strips_bullet_points(self, agent):
        plan = Mock(use_qe=True, max_qe_variants=5)
        agent.router.chat.return_value = "- variant a\n• variant b\n* variant c"
        
        inp = QEInput(query="test", plan=plan)
        output = agent.expand(inp)
        
        assert output.expanded_queries[0] == "variant a"
        assert output.expanded_queries[1] == "variant b"
        assert output.expanded_queries[2] == "variant c"
    
    def test_filters_empty_lines(self, agent):
        plan = Mock(use_qe=True, max_qe_variants=5)
        agent.router.chat.return_value = "variant 1\n\n\nvariant 2\n   \nvariant 3"
        
        inp = QEInput(query="test", plan=plan)
        output = agent.expand(inp)
        
        assert len(output.expanded_queries) == 3
        assert "" not in output.expanded_queries
    
    def test_handles_mixed_formatting(self, agent):
        plan = Mock(use_qe=True, max_qe_variants=10)
        agent.router.chat.return_value = """
1. first variant
- second variant
  third variant
• fourth variant
5) fifth variant
"""
        inp = QEInput(query="test", plan=plan)
        output = agent.expand(inp)
        
        # All should be cleaned
        for variant in output.expanded_queries:
            assert not variant.startswith(("-", "•", "*"))
            assert not variant[0].isdigit()
```

#### 4. LLM Interaction Tests

```python
class TestLLMInteraction:
    
    def test_prompt_contains_query(self, agent):
        plan = Mock(use_qe=True, max_qe_variants=5)
        agent.router.chat.return_value = "variant"
        
        inp = QEInput(query="my specific query", plan=plan)
        agent.expand(inp)
        
        call_args = agent.router.chat.call_args
        messages = call_args[0][0]
        user_message = next(m for m in messages if m["role"] == "user")
        
        assert "my specific query" in user_message["content"]
    
    def test_prompt_contains_num_variants(self, agent):
        plan = Mock(use_qe=True, max_qe_variants=7)
        agent.router.chat.return_value = "variant"
        
        inp = QEInput(query="test", plan=plan)
        agent.expand(inp)
        
        call_args = agent.router.chat.call_args
        messages = call_args[0][0]
        user_message = next(m for m in messages if m["role"] == "user")
        
        assert "7" in user_message["content"]
    
    def test_uses_configured_temperature(self, agent):
        plan = Mock(use_qe=True, max_qe_variants=5)
        agent.temperature = 0.7
        agent.router.chat.return_value = "variant"
        
        inp = QEInput(query="test", plan=plan)
        agent.expand(inp)
        
        call_kwargs = agent.router.chat.call_args[1]
        assert call_kwargs["temperature"] == 0.7
    
    def test_uses_configured_max_tokens(self, agent):
        plan = Mock(use_qe=True, max_qe_variants=5)
        agent.max_new_tokens = 128
        agent.router.chat.return_value = "variant"
        
        inp = QEInput(query="test", plan=plan)
        agent.expand(inp)
        
        call_kwargs = agent.router.chat.call_args[1]
        assert call_kwargs["max_tokens"] == 128
```

#### 5. Edge Case Tests

```python
class TestEdgeCases:
    
    def test_empty_llm_response(self, agent):
        plan = Mock(use_qe=True, max_qe_variants=5)
        agent.router.chat.return_value = ""
        
        inp = QEInput(query="test", plan=plan)
        output = agent.expand(inp)
        
        assert output.expanded_queries == []
    
    def test_llm_returns_only_whitespace(self, agent):
        plan = Mock(use_qe=True, max_qe_variants=5)
        agent.router.chat.return_value = "   \n\n   \n"
        
        inp = QEInput(query="test", plan=plan)
        output = agent.expand(inp)
        
        assert output.expanded_queries == []
    
    def test_very_long_query(self, agent):
        plan = Mock(use_qe=True, max_qe_variants=3)
        long_query = "a" * 1000
        agent.router.chat.return_value = "v1\nv2\nv3"
        
        inp = QEInput(query=long_query, plan=plan)
        output = agent.expand(inp)
        
        assert len(output.expanded_queries) == 3
    
    def test_unicode_query(self, agent):
        plan = Mock(use_qe=True, max_qe_variants=3)
        agent.router.chat.return_value = "変形1\n变体2\nвариант3"
        
        inp = QEInput(query="日本語クエリ", plan=plan)
        output = agent.expand(inp)
        
        assert len(output.expanded_queries) == 3
        assert "変形1" in output.expanded_queries
```

#### 6. Agent Interface Tests

```python
class TestAgentInterface:
    
    def test_name_property(self, agent):
        assert agent.name == "LLMQEAgent"
    
    def test_describe_method(self, agent):
        description = agent.describe()
        assert isinstance(description, str)
        assert "Query Expansion" in description or "QE" in description
    
    def test_role_attribute(self, agent):
        assert agent.role == "qe"
```

### Test Commands

```bash
# Run all QE tests
pytest test_qe_llm_agent.py -v

# Run with coverage
pytest test_qe_llm_agent.py --cov=qe_llm_agent --cov-report=html

# Run specific test class
pytest test_qe_llm_agent.py::TestOutputParsing -v
```

---

## Recommendations and Improvements

### Critical Issues

#### 1. No Error Handling for LLM Failures

**Problem:** LLM call failures will crash the pipeline.

**Recommendation:** Add error handling:

```python
def expand(self, inp: QEInput) -> QEOutput:
    # ... setup ...
    
    try:
        raw = self.router.chat(messages, max_tokens=..., temperature=...)
    except Exception as e:
        logger.warning(f"QE LLM call failed: {e}, returning empty expansion")
        return QEOutput(expanded_queries=[])
    
    # ... parsing ...
```

#### 2. No Deduplication of Variants

**Problem:** LLM might return duplicate or near-duplicate variants.

**Recommendation:** Add deduplication:

```python
def expand(self, inp: QEInput) -> QEOutput:
    # ... LLM call and parsing ...
    
    # Exact deduplication
    seen = set()
    unique_variants = []
    for v in variants:
        v_lower = v.lower()
        if v_lower not in seen:
            seen.add(v_lower)
            unique_variants.append(v)
    
    # Optional: fuzzy deduplication
    # Remove variants too similar to each other
    
    return QEOutput(expanded_queries=unique_variants[:num_variants])
```

---

### High Priority Improvements

#### 3. Query Included in Output

**Problem:** Original query not included in expanded_queries (might be needed by retriever).

**Recommendation:** Option to include original:

```python
@dataclass
class QEConfig:
    include_original: bool = True

def expand(self, inp: QEInput) -> QEOutput:
    # ... expansion ...
    
    if self.include_original:
        variants = [inp.query] + variants
    
    return QEOutput(expanded_queries=variants)
```

#### 4. Logging and Observability

**Problem:** No visibility into QE operations.

**Recommendation:** Add structured logging:

```python
import logging
logger = logging.getLogger(__name__)

def expand(self, inp: QEInput) -> QEOutput:
    logger.info(
        "qe_start",
        extra={
            "query": inp.query,
            "use_qe": getattr(inp.plan, "use_qe", False),
            "num_variants": num_variants,
        }
    )
    
    # ... expansion ...
    
    logger.info(
        "qe_complete",
        extra={
            "query": inp.query,
            "num_generated": len(variants),
            "variants": variants,
        }
    )
```

#### 5. Caching Expansion Results

**Problem:** Same query always calls LLM.

**Recommendation:** Add caching:

```python
from functools import lru_cache

class LLMQEAgent:
    @lru_cache(maxsize=256)
    def _expand_cached(self, query: str, num_variants: int) -> tuple:
        # ... LLM call ...
        return tuple(variants)
    
    def expand(self, inp: QEInput) -> QEOutput:
        if not getattr(inp.plan, "use_qe", False):
            return QEOutput(expanded_queries=[])
        
        variants = list(self._expand_cached(inp.query, num_variants))
        return QEOutput(expanded_queries=variants)
```

#### 6. Configurable Prompt Templates

**Problem:** Prompts are hardcoded.

**Recommendation:** Make prompts configurable:

```yaml
retrieval:
  query_expansion:
    system_prompt: |
      You are a query expansion assistant...
    user_prompt_template: |
      Original query: {query}
      Number of paraphrases: {num_variants}
```

```python
def expand(self, inp: QEInput) -> QEOutput:
    system_prompt = self.qe_cfg.get("system_prompt", DEFAULT_SYSTEM_PROMPT)
    user_template = self.qe_cfg.get("user_prompt_template", DEFAULT_USER_TEMPLATE)
    
    user_prompt = user_template.format(
        query=inp.query,
        num_variants=num_variants
    )
```

---

### Medium Priority Improvements

#### 7. Quality Filtering

**Problem:** No validation that variants are actually useful.

**Recommendation:** Add quality checks:

```python
def _is_valid_variant(self, original: str, variant: str) -> bool:
    """Check if variant is a valid expansion."""
    # Not too short
    if len(variant) < 5:
        return False
    
    # Not identical to original
    if variant.lower() == original.lower():
        return False
    
    # Not just the original with minor changes
    if self._similarity(original, variant) > 0.95:
        return False
    
    return True
```

#### 8. Async Support

**Problem:** LLM call is blocking.

**Recommendation:** Add async method:

```python
async def expand_async(self, inp: QEInput) -> QEOutput:
    if not getattr(inp.plan, "use_qe", False):
        return QEOutput(expanded_queries=[])
    
    raw = await self.router.chat_async(messages, ...)
    # ... parsing ...
```

#### 9. Retry Logic

**Problem:** No retry on transient LLM failures.

**Recommendation:** Add retry with backoff:

```python
import time

def expand(self, inp: QEInput) -> QEOutput:
    max_retries = 3
    
    for attempt in range(max_retries):
        try:
            raw = self.router.chat(messages, ...)
            break
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                logger.error(f"QE failed after {max_retries} attempts")
                return QEOutput(expanded_queries=[])
```

---

### Low Priority / Future Enhancements

#### 10. Multi-Language Support

**Recommendation:** Language-aware expansion:

```python
def expand(self, inp: QEInput) -> QEOutput:
    # Detect query language
    lang = self._detect_language(inp.query)
    
    # Use language-specific prompt
    system_prompt = self.prompts.get(lang, self.prompts["en"])
```

#### 11. Domain-Specific Expansion

**Recommendation:** Add domain context:

```yaml
retrieval:
  query_expansion:
    domain: "medical"
    domain_context: "Focus on clinical terminology and medical concepts."
```

#### 12. Expansion Strategy Selection

**Recommendation:** Support different expansion strategies:

```python
class ExpansionStrategy(Enum):
    PARAPHRASE = "paraphrase"      # Semantic equivalents
    BROADEN = "broaden"            # More general queries
    NARROW = "narrow"              # More specific queries
    ASPECT = "aspect"              # Different aspects of same topic

def expand(self, inp: QEInput, strategy: ExpansionStrategy = ExpansionStrategy.PARAPHRASE):
    # Adjust prompt based on strategy
```

---

## Usage Examples

### Basic Usage

```python
from qe_llm_agent import LLMQEAgent
from core.schemas import QEInput, Plan

# Initialize with config
config = {
    "llm": {
        "model": "gpt-4o-mini",
        "api_base": "https://api.openai.com/v1",
        "api_key": "..."
    },
    "retrieval": {
        "query_expansion": {
            "num_variants": 5,
            "temperature": 0.3
        }
    }
}

agent = LLMQEAgent(config)

# Create input
plan = Plan(use_qe=True, ...)
inp = QEInput(query="best python web frameworks", plan=plan)

# Expand
output = agent.expand(inp)

print("Original:", inp.query)
print("Variants:")
for v in output.expanded_queries:
    print(f"  - {v}")

# Output:
# Original: best python web frameworks
# Variants:
#   - top python frameworks for web development
#   - recommended python web application frameworks
#   - python web framework comparison and recommendations
#   - most popular python frameworks for building websites
#   - which python framework is best for web apps
```

### Pipeline Integration

```python
class RAGPipeline:
    def __init__(self, config: dict):
        self.qe_agent = LLMQEAgent(config)
        self.retriever = HybridRetrievalAgent(config_path=config["config_path"])
    
    def process(self, query: str, plan: Plan) -> RetrieverOutput:
        # Step 1: Expand query if enabled
        expanded_queries = []
        if plan.use_qe:
            qe_output = self.qe_agent.expand(QEInput(query=query, plan=plan))
            expanded_queries = qe_output.expanded_queries
        
        # Step 2: Retrieve with all queries
        retriever_input = RetrieverInput(
            query=query,
            expanded_queries=expanded_queries,
            plan=plan
        )
        
        return self.retriever.retrieve(retriever_input)
```

### With Plan Override

```python
# Override variant count via plan
plan = Plan(
    use_qe=True,
    max_qe_variants=3,  # Only 3 variants instead of config default
    ...
)

inp = QEInput(query="machine learning tutorials", plan=plan)
output = agent.expand(inp)

assert len(output.expanded_queries) <= 3
```

### Disabled QE

```python
# Disable QE via plan
plan = Plan(use_qe=False, ...)

inp = QEInput(query="test query", plan=plan)
output = agent.expand(inp)

assert output.expanded_queries == []  # No LLM call made
```

---

## Appendix

### Glossary

| Term | Definition |
|------|------------|
| **QE** | Query Expansion |
| **Paraphrase** | Semantically equivalent reformulation |
| **Vocabulary Mismatch** | When query terms don't match document terms |
| **Query Drift** | When expansion changes query meaning |

### Configuration Reference

| Config Path | Type | Default | Description |
|-------------|------|---------|-------------|
| `retrieval.query_expansion.num_variants` | int | 5 | Default variant count |
| `retrieval.query_expansion.temperature` | float | 0.2 | LLM temperature |
| `retrieval.query_expansion.max_new_tokens` | int | 64 | Max generation length |

### Plan Fields

| Field | Type | Purpose |
|-------|------|---------|
| `use_qe` | bool | Enable/disable QE |
| `max_qe_variants` | int | Override variant count |

### Output Parsing Characters

| Character | Stripped |
|-----------|----------|
| `-` | Yes |
| `•` | Yes |
| `*` | Yes |
| `0-9` | Yes (at start) |
| `.` | Yes (after numbers) |
| `)` | No |

### Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | Initial | Basic LLM-based query expansion |

### References

- Radiant Repository: https://github.com/dshipley71/radiant
- Related Files: `prf_basic_agent.py`, `retriever_haystack_agent.py`, `core/llm_router.py`

---

*Document generated for the Radiant RAG Pipeline project. For questions or contributions, please refer to the project repository.*
