# BasicRouterAgent Documentation

## Technical Reference for the Radiant RAG Pipeline Router

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture Context](#architecture-context)
3. [Class Structure](#class-structure)
4. [Core Functionality](#core-functionality)
5. [Configurable Parameters](#configurable-parameters)
6. [Heuristic Methods](#heuristic-methods)
7. [Toggle Decision Logic](#toggle-decision-logic)
8. [Data Flow](#data-flow)
9. [Testing Strategies](#testing-strategies)
10. [Recommendations and Improvements](#recommendations-and-improvements)
11. [Usage Examples](#usage-examples)

---

## Overview

### Purpose

The `BasicRouterAgent` is a heuristic-based query routing component within the Radiant RAG (Retrieval-Augmented Generation) pipeline. Its primary responsibility is to analyze incoming user queries and determine the optimal retrieval and processing strategy by classifying queries and toggling various pipeline features.

### Key Responsibilities

- Classify user queries into semantic types (comparison, list, explanation, lookup, other)
- Determine expected answer styles (short, paragraph, multi_section)
- Assess query complexity (low, medium, high)
- Detect conversational follow-up queries
- Enable/disable Query Expansion (QE), Pseudo-Relevance Feedback (PRF), and reranking

### Design Philosophy

The router employs a **rule-based heuristic approach** rather than machine learning classification. This provides predictability, transparency, and low latency at the cost of nuanced understanding of edge cases.

---

## Architecture Context

### Position in the Radiant Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Query                               │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    BasicRouterAgent                             │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  • Query Type Classification                            │   │
│  │  • Answer Style Inference                               │   │
│  │  • Complexity Assessment                                │   │
│  │  • Follow-up Detection                                  │   │
│  │  • Feature Toggle Decisions                             │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
                      RouterProfile Output
                               │
              ┌────────────────┼────────────────┐
              ▼                ▼                ▼
        ┌──────────┐    ┌──────────┐    ┌──────────┐
        │    QE    │    │   PRF    │    │  Rerank  │
        │  Agent   │    │  Agent   │    │  Agent   │
        └──────────┘    └──────────┘    └──────────┘
              │                │                │
              └────────────────┼────────────────┘
                               ▼
                    Downstream Processing
                  (Retriever, Generator, etc.)
```

### Related Components

| Component | Relationship |
|-----------|--------------|
| `RouterAgent` | Abstract base class (from `core.interfaces`) |
| `RouterInput` | Input schema containing query, config, and history |
| `RouterOutput` | Output schema wrapping the RouterProfile |
| `RouterProfile` | Data structure containing routing decisions |
| `RouterConfig` | Configuration object with tunable parameters |

---

## Class Structure

### Inheritance

```python
class BasicRouterAgent(RouterAgent):
    """Heuristic router for query type and high-level retrieval toggles."""
```

The class inherits from `RouterAgent`, implementing the required interface contract.

### Class Attributes

| Attribute | Type | Value | Description |
|-----------|------|-------|-------------|
| `role` | `str` | `"router"` | Identifies the agent's role in the pipeline |

### Properties

| Property | Return Type | Description |
|----------|-------------|-------------|
| `name` | `str` | Returns `"BasicRouterAgent"` |

### Methods

| Method | Visibility | Purpose |
|--------|------------|---------|
| `describe()` | Public | Returns human-readable description |
| `route(inp)` | Public | Main entry point for routing logic |
| `_infer_query_type(q)` | Private | Classifies query type |
| `_infer_answer_style(q, is_followup)` | Private | Determines expected answer format |
| `_infer_complexity(q)` | Private | Assesses query complexity |
| `_is_followup(history, q_lower)` | Private | Detects follow-up queries |
| `_decide_toggles(...)` | Private | Determines QE/PRF/rerank flags |

---

## Core Functionality

### The `route()` Method

This is the primary entry point that orchestrates all routing logic.

**Signature:**
```python
def route(self, inp: RouterInput) -> RouterOutput
```

**Parameters:**
- `inp` (`RouterInput`): Contains user query, configuration, and conversation history

**Returns:**
- `RouterOutput`: Contains the `RouterProfile` with all routing decisions

**Processing Steps:**

1. **Extract and normalize query**: Strip whitespace and convert to lowercase
2. **Truncate history**: Apply `max_hist_turns` limit from configuration
3. **Classify query type**: Run heuristic classification
4. **Apply default fallback**: Use `default_query_type` if classified as "other"
5. **Detect follow-up**: Check if query is a conversational continuation
6. **Infer answer style**: Determine expected response format
7. **Assess complexity**: Evaluate query complexity level
8. **Decide toggles**: Set QE, PRF, and rerank flags
9. **Build and return profile**: Package decisions into output schema

---

## Configurable Parameters

### RouterConfig Options

The router respects the following configuration parameters:

| Parameter | Type | Purpose | Impact |
|-----------|------|---------|--------|
| `max_hist_turns` | `int` | Maximum conversation turns to consider | Limits history for follow-up detection |
| `default_query_type` | `str` | Fallback query type | Overrides "other" classification |

### Implicit Configuration

These aspects are currently hardcoded but could be made configurable:

| Aspect | Current Value | Location |
|--------|---------------|----------|
| Short query threshold | 8 tokens | `_infer_answer_style()`, `_infer_complexity()` |
| Medium complexity threshold | 20 tokens | `_infer_complexity()` |
| Follow-up max tokens | 5 tokens | `_is_followup()` |
| Rerank toggle | Always `True` | `_decide_toggles()` |

---

## Heuristic Methods

### Query Type Classification (`_infer_query_type`)

**Classification Logic:**

| Query Type | Trigger Conditions |
|------------|--------------------|
| `comparison` | Contains " vs ", " versus ", or "difference between" |
| `list` | Starts with "list " OR contains " list of " OR contains "top " |
| `explanation` | Starts with "how " or "why " OR contains "explain" |
| `lookup` | Starts with "what " OR ends with "?" |
| `other` | Default fallback |

**Priority Order:** comparison → list → explanation → lookup → other

### Answer Style Inference (`_infer_answer_style`)

| Style | Trigger Conditions |
|-------|-------------------|
| `multi_section` | Query contains "overview", "detailed", or "guide" |
| `short` | Token count < 8 AND not a follow-up |
| `paragraph` | Token count < 8 AND is follow-up, OR token count ≥ 8 |

### Complexity Assessment (`_infer_complexity`)

| Complexity | Token Count |
|------------|-------------|
| `low` | < 8 tokens |
| `medium` | 8-19 tokens |
| `high` | ≥ 20 tokens |

### Follow-up Detection (`_is_followup`)

A query is classified as a follow-up when **all** conditions are met:
1. Conversation history exists
2. Query has ≤ 5 tokens
3. Query does NOT start with: what, how, why, who, where, when

---

## Toggle Decision Logic

### Query Expansion (QE)

**Enabled when:**
- Complexity is "medium" or "high", OR
- Query is detected as a follow-up

**Rationale:** Longer, complex queries benefit from synonym expansion. Follow-ups may reference prior context and need broader recall.

### Pseudo-Relevance Feedback (PRF)

**Enabled when:**
- Query type is "comparison" or "list", AND
- Complexity is NOT "low"

**Rationale:** PRF helps gather diverse documents for multi-faceted queries but adds overhead for simple lookups.

### Reranking

**Status:** Always enabled (`True`)

**Rationale:** Cross-encoder reranking consistently improves result quality with acceptable latency trade-off.

### Toggle Decision Matrix

| Query Type | Complexity | Follow-up | QE | PRF | Rerank |
|------------|------------|-----------|-----|-----|--------|
| comparison | low | No | ❌ | ❌ | ✅ |
| comparison | medium | No | ✅ | ✅ | ✅ |
| comparison | high | No | ✅ | ✅ | ✅ |
| list | low | No | ❌ | ❌ | ✅ |
| list | medium | No | ✅ | ✅ | ✅ |
| explanation | medium | No | ✅ | ❌ | ✅ |
| lookup | low | Yes | ✅ | ❌ | ✅ |
| other | low | No | ❌ | ❌ | ✅ |

---

## Data Flow

### Input Schema: `RouterInput`

```python
@dataclass
class RouterInput:
    user_query: str          # Raw user query text
    config: RouterConfig     # Configuration parameters
    history: List[...]       # Conversation history
```

### Output Schema: `RouterOutput`

```python
@dataclass
class RouterOutput:
    router_profile: RouterProfile
```

### Profile Schema: `RouterProfile`

```python
@dataclass
class RouterProfile:
    query_type: str              # comparison|list|explanation|lookup|other
    use_qe: bool                 # Query Expansion toggle
    use_prf: bool                # Pseudo-Relevance Feedback toggle
    use_rerank: bool             # Reranking toggle
    expected_answer_style: str   # short|paragraph|multi_section
    complexity_hint: str         # low|medium|high
```

---

## Testing Strategies

### Unit Tests

#### 1. Query Type Classification Tests

```python
import pytest
from router_basic_agent import BasicRouterAgent
from core.schemas import RouterInput, RouterConfig

@pytest.fixture
def router():
    return BasicRouterAgent()

@pytest.fixture
def default_config():
    return RouterConfig(max_hist_turns=5, default_query_type=None)

class TestQueryTypeClassification:
    
    def test_comparison_vs(self, router):
        assert router._infer_query_type("python vs javascript") == "comparison"
    
    def test_comparison_versus(self, router):
        assert router._infer_query_type("cats versus dogs") == "comparison"
    
    def test_comparison_difference(self, router):
        assert router._infer_query_type("difference between sql and nosql") == "comparison"
    
    def test_list_prefix(self, router):
        assert router._infer_query_type("list all programming languages") == "list"
    
    def test_list_contains(self, router):
        assert router._infer_query_type("give me a list of countries") == "list"
    
    def test_list_top(self, router):
        assert router._infer_query_type("top 10 movies of 2024") == "list"
    
    def test_explanation_how(self, router):
        assert router._infer_query_type("how does photosynthesis work") == "explanation"
    
    def test_explanation_why(self, router):
        assert router._infer_query_type("why is the sky blue") == "explanation"
    
    def test_explanation_explain(self, router):
        assert router._infer_query_type("explain quantum computing") == "explanation"
    
    def test_lookup_what(self, router):
        assert router._infer_query_type("what is machine learning") == "lookup"
    
    def test_lookup_question_mark(self, router):
        assert router._infer_query_type("capital of france?") == "lookup"
    
    def test_other_fallback(self, router):
        assert router._infer_query_type("hello there") == "other"
```

#### 2. Complexity Assessment Tests

```python
class TestComplexityAssessment:
    
    def test_low_complexity(self, router):
        assert router._infer_complexity("short query") == "low"
    
    def test_medium_complexity(self, router):
        query = "this is a medium length query with several words"
        assert router._infer_complexity(query) == "medium"
    
    def test_high_complexity(self, router):
        query = " ".join(["word"] * 25)
        assert router._infer_complexity(query) == "high"
    
    def test_boundary_low_medium(self, router):
        query = " ".join(["word"] * 7)  # 7 tokens
        assert router._infer_complexity(query) == "low"
        
        query = " ".join(["word"] * 8)  # 8 tokens
        assert router._infer_complexity(query) == "medium"
```

#### 3. Follow-up Detection Tests

```python
class TestFollowupDetection:
    
    def test_no_history_not_followup(self, router):
        assert router._is_followup([], "and pricing") == False
    
    def test_short_query_with_history(self, router):
        history = [{"role": "user", "content": "tell me about product x"}]
        assert router._is_followup(history, "and pricing") == True
    
    def test_long_query_not_followup(self, router):
        history = [{"role": "user", "content": "previous query"}]
        assert router._is_followup(history, "this is a longer follow up question") == False
    
    def test_new_question_not_followup(self, router):
        history = [{"role": "user", "content": "previous query"}]
        assert router._is_followup(history, "what is x") == False
        assert router._is_followup(history, "how does it") == False
```

#### 4. Toggle Decision Tests

```python
class TestToggleDecisions:
    
    def test_qe_enabled_medium_complexity(self, router):
        qe, prf, rerank = router._decide_toggles("lookup", "medium", False)
        assert qe == True
    
    def test_qe_enabled_followup(self, router):
        qe, prf, rerank = router._decide_toggles("lookup", "low", True)
        assert qe == True
    
    def test_prf_enabled_comparison_medium(self, router):
        qe, prf, rerank = router._decide_toggles("comparison", "medium", False)
        assert prf == True
    
    def test_prf_disabled_comparison_low(self, router):
        qe, prf, rerank = router._decide_toggles("comparison", "low", False)
        assert prf == False
    
    def test_rerank_always_enabled(self, router):
        for query_type in ["comparison", "list", "explanation", "lookup", "other"]:
            for complexity in ["low", "medium", "high"]:
                _, _, rerank = router._decide_toggles(query_type, complexity, False)
                assert rerank == True
```

#### 5. Integration Tests

```python
class TestRouteIntegration:
    
    def test_full_route_comparison(self, router, default_config):
        inp = RouterInput(
            user_query="Python vs JavaScript for web development",
            config=default_config,
            history=[]
        )
        output = router.route(inp)
        profile = output.router_profile
        
        assert profile.query_type == "comparison"
        assert profile.use_rerank == True
    
    def test_default_query_type_override(self, router):
        config = RouterConfig(max_hist_turns=5, default_query_type="lookup")
        inp = RouterInput(
            user_query="hello",
            config=config,
            history=[]
        )
        output = router.route(inp)
        
        assert output.router_profile.query_type == "lookup"
    
    def test_history_truncation(self, router):
        config = RouterConfig(max_hist_turns=2, default_query_type=None)
        long_history = [{"content": f"msg {i}"} for i in range(10)]
        inp = RouterInput(
            user_query="follow up",
            config=config,
            history=long_history
        )
        # The router should only consider the last 2 turns
        output = router.route(inp)
        assert output.router_profile is not None
```

### Performance Tests

```python
import time

class TestPerformance:
    
    def test_route_latency(self, router, default_config):
        """Router should complete in < 1ms for typical queries."""
        inp = RouterInput(
            user_query="What is the difference between SQL and NoSQL databases?",
            config=default_config,
            history=[]
        )
        
        start = time.perf_counter()
        for _ in range(1000):
            router.route(inp)
        elapsed = time.perf_counter() - start
        
        avg_latency_ms = (elapsed / 1000) * 1000
        assert avg_latency_ms < 1.0, f"Average latency {avg_latency_ms}ms exceeds 1ms"
```

### Test Commands

```bash
# Run all router tests
pytest test_router_basic_agent.py -v

# Run with coverage
pytest test_router_basic_agent.py --cov=router_basic_agent --cov-report=html

# Run specific test class
pytest test_router_basic_agent.py::TestQueryTypeClassification -v

# Run performance tests only
pytest test_router_basic_agent.py::TestPerformance -v
```

---

## Recommendations and Improvements

### Critical Issues

#### 1. Case Sensitivity in Query Type Detection

**Problem:** The `_infer_query_type` method uses lowercase matching, but certain patterns like "VS" (uppercase) would fail.

**Current:**
```python
if " vs " in q or " versus " in q:
```

**Recommendation:** Already addressed by converting to lowercase before classification, but document this dependency clearly.

---

### High Priority Improvements

#### 2. Configurable Thresholds

**Problem:** Magic numbers are hardcoded throughout the heuristics.

**Recommendation:** Extract to configuration:

```python
@dataclass
class RouterConfig:
    max_hist_turns: int = 5
    default_query_type: str = None
    # NEW: Configurable thresholds
    short_query_threshold: int = 8
    medium_complexity_threshold: int = 20
    followup_max_tokens: int = 5
    enable_rerank_default: bool = True
```

#### 3. Enhanced Query Type Detection

**Problem:** Simple substring matching misses semantic nuances.

**Example failures:**
- "What's the difference in pricing?" → classified as "lookup" (misses "difference")
- "Compare these options" → classified as "other" (misses comparison intent)

**Recommendation:** Expand keyword lists and add regex patterns:

```python
COMPARISON_PATTERNS = [
    r'\bvs\.?\b', r'\bversus\b', r'difference between',
    r'\bcompare\b', r'\bcomparison\b', r'better than',
    r'\bor\b.*\bwhich\b'
]

def _infer_query_type(self, q: str) -> str:
    for pattern in COMPARISON_PATTERNS:
        if re.search(pattern, q):
            return "comparison"
    # ... rest of classification
```

#### 4. Logging and Observability

**Problem:** No visibility into routing decisions for debugging or analytics.

**Recommendation:** Add structured logging:

```python
import logging
from dataclasses import asdict

logger = logging.getLogger(__name__)

def route(self, inp: RouterInput) -> RouterOutput:
    # ... routing logic ...
    
    logger.info(
        "routing_decision",
        extra={
            "query_hash": hash(inp.user_query),
            "query_length": len(inp.user_query.split()),
            "profile": asdict(profile),
            "is_followup": is_followup,
        }
    )
    return RouterOutput(router_profile=profile)
```

---

### Medium Priority Improvements

#### 5. Multi-language Support

**Problem:** Heuristics are English-only.

**Recommendation:** Add language detection and language-specific keyword sets:

```python
from langdetect import detect

KEYWORDS = {
    "en": {"comparison": ["vs", "versus", "difference between"]},
    "es": {"comparison": ["vs", "versus", "diferencia entre"]},
    # ...
}

def _infer_query_type(self, q: str) -> str:
    lang = detect(q)
    keywords = KEYWORDS.get(lang, KEYWORDS["en"])
    # ... use language-specific keywords
```

#### 6. Confidence Scores

**Problem:** Binary classification provides no indication of certainty.

**Recommendation:** Return confidence scores with classifications:

```python
@dataclass
class RouterProfile:
    query_type: str
    query_type_confidence: float  # 0.0 - 1.0
    # ... other fields
```

#### 7. LLM-Based Router Option

**Problem:** Heuristics cannot capture complex semantic intent.

**Recommendation:** Implement an `LLMRouterAgent` as an alternative (note: `llm_router.py` may already exist in the codebase):

```python
class LLMRouterAgent(RouterAgent):
    """LLM-based router for complex query classification."""
    
    def route(self, inp: RouterInput) -> RouterOutput:
        prompt = f"""Classify this query:
        Query: {inp.user_query}
        
        Return JSON with:
        - query_type: comparison|list|explanation|lookup|other
        - complexity: low|medium|high
        - expected_answer_style: short|paragraph|multi_section
        """
        # ... LLM call and parsing
```

---

### Low Priority / Future Enhancements

#### 8. A/B Testing Infrastructure

**Recommendation:** Add experiment support for comparing routing strategies:

```python
def route(self, inp: RouterInput) -> RouterOutput:
    if inp.config.experiment_group == "llm_router":
        return self._route_with_llm(inp)
    return self._route_with_heuristics(inp)
```

#### 9. Query Type Expansion

**Recommendation:** Add more granular query types:

- `factoid` - Simple fact questions
- `procedural` - "How to" instructions
- `opinion` - Subjective questions
- `temporal` - Time-based queries
- `aggregation` - Questions requiring data synthesis

#### 10. Adaptive Thresholds

**Recommendation:** Learn optimal thresholds from user feedback:

```python
class AdaptiveRouterAgent(BasicRouterAgent):
    def __init__(self, feedback_store):
        self.feedback_store = feedback_store
    
    def _infer_complexity(self, q: str) -> str:
        # Use learned thresholds instead of hardcoded values
        thresholds = self.feedback_store.get_optimal_thresholds()
        # ...
```

---

## Usage Examples

### Basic Usage

```python
from router_basic_agent import BasicRouterAgent
from core.schemas import RouterInput, RouterConfig

# Initialize router
router = BasicRouterAgent()

# Create configuration
config = RouterConfig(
    max_hist_turns=5,
    default_query_type="lookup"
)

# Route a query
inp = RouterInput(
    user_query="What is the difference between REST and GraphQL?",
    config=config,
    history=[]
)

output = router.route(inp)
profile = output.router_profile

print(f"Query Type: {profile.query_type}")        # comparison
print(f"Use QE: {profile.use_qe}")                 # True (medium complexity)
print(f"Use PRF: {profile.use_prf}")               # True (comparison + medium)
print(f"Use Rerank: {profile.use_rerank}")         # True (always on)
print(f"Answer Style: {profile.expected_answer_style}")  # paragraph
print(f"Complexity: {profile.complexity_hint}")    # medium
```

### With Conversation History

```python
# Simulate a conversation
history = [
    {"role": "user", "content": "Tell me about AWS services"},
    {"role": "assistant", "content": "AWS offers many cloud services..."}
]

# Follow-up query
inp = RouterInput(
    user_query="and pricing?",
    config=config,
    history=history
)

output = router.route(inp)
# is_followup will be True
# use_qe will be True (follow-up triggers QE)
# expected_answer_style will be "paragraph" (follow-up bias)
```

### Pipeline Integration

```python
class RAGPipeline:
    def __init__(self):
        self.router = BasicRouterAgent()
        self.qe_agent = QueryExpansionAgent()
        self.prf_agent = PRFAgent()
        self.rerank_agent = RerankAgent()
        self.retriever = RetrieverAgent()
        self.generator = GeneratorAgent()
    
    def process(self, query: str, history: list, config: RouterConfig):
        # Step 1: Route the query
        router_input = RouterInput(
            user_query=query,
            config=config,
            history=history
        )
        router_output = self.router.route(router_input)
        profile = router_output.router_profile
        
        # Step 2: Apply enabled features
        processed_query = query
        
        if profile.use_qe:
            processed_query = self.qe_agent.expand(processed_query)
        
        # Step 3: Retrieve
        documents = self.retriever.retrieve(processed_query)
        
        if profile.use_prf:
            documents = self.prf_agent.refine(documents, processed_query)
        
        if profile.use_rerank:
            documents = self.rerank_agent.rerank(documents, query)
        
        # Step 4: Generate
        response = self.generator.generate(
            query=query,
            documents=documents,
            style=profile.expected_answer_style
        )
        
        return response
```

---

## Appendix

### Glossary

| Term | Definition |
|------|------------|
| **QE (Query Expansion)** | Technique to broaden search queries with synonyms and related terms |
| **PRF (Pseudo-Relevance Feedback)** | Uses initial search results to refine and improve the query |
| **Reranking** | Re-orders retrieved documents using a more sophisticated model (typically cross-encoder) |
| **Cross-encoder** | Neural model that processes query-document pairs jointly for relevance scoring |
| **RAG** | Retrieval-Augmented Generation - combining retrieval with LLM generation |

### Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | Initial | Basic heuristic routing implementation |

### References

- Radiant Repository: https://github.com/dshipley71/radiant
- Related Files: `llm_router.py`, `orchestrator.py`, `agents_interfaces.py`, `agents_schemas.py`

---

*Document generated for the Radiant RAG Pipeline project. For questions or contributions, please refer to the project repository.*
