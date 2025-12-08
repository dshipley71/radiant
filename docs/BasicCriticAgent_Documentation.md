# BasicCriticAgent Documentation

## Technical Reference for the Radiant RAG Pipeline Answer Evaluation

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture Context](#architecture-context)
3. [Class Structure](#class-structure)
4. [Core Functionality](#core-functionality)
5. [Evaluation Metrics](#evaluation-metrics)
6. [Data Flow](#data-flow)
7. [Testing Strategies](#testing-strategies)
8. [Recommendations and Improvements](#recommendations-and-improvements)
9. [Usage Examples](#usage-examples)

---

## Overview

### Purpose

The `BasicCriticAgent` is a quality evaluation component within the Radiant RAG pipeline. It assesses generated answers for potential issues such as hallucination risk, insufficient coverage, and answer quality problems using lightweight heuristic methods.

### Key Responsibilities

- Calculate coverage score based on context utilization
- Estimate hallucination risk inversely from coverage
- Detect answer quality issues (too short, no context, low coverage)
- Generate human-readable notes about potential problems

### Design Philosophy

The agent implements a **fast heuristic approach** rather than LLM-based evaluation. This enables real-time feedback without additional API calls, suitable for production systems where latency matters. The trade-off is less nuanced analysis compared to semantic evaluation methods.

---

## Architecture Context

### Position in the Radiant Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    CriticInput                                  │
│  answer | context_snippets | plan                               │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    BasicCriticAgent                             │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Evaluation Metrics:                                    │   │
│  │  • Coverage Score = snippets / top_k                    │   │
│  │  • Hallucination Risk = 1 - coverage                    │   │
│  ├─────────────────────────────────────────────────────────┤   │
│  │  Quality Checks:                                        │   │
│  │  • No context? → hallucination warning                  │   │
│  │  • Short answer? → elaboration suggestion               │   │
│  │  • Low coverage? → relevance warning                    │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    CriticOutput                                 │
│  hallucination_risk | coverage_score | notes                    │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
              Pipeline Decision (retry, flag, accept)
```

### Related Components

| Component | Relationship |
|-----------|--------------|
| `CriticAgent` | Abstract base class (from `core.interfaces`) |
| `CriticInput` | Input schema with answer, context, and plan |
| `CriticOutput` | Output schema with scores and notes |
| `Answer` | Answer object with text property |
| `Plan` | Execution plan with top_k setting |

---

## Class Structure

### Inheritance

```python
class BasicCriticAgent(CriticAgent):
    """Heuristic critic for coverage and hallucination risk."""
```

### Class Attributes

| Attribute | Type | Value | Description |
|-----------|------|-------|-------------|
| `role` | `str` | `"critic"` | Agent role identifier |

### Properties

| Property | Return Type | Description |
|----------|-------------|-------------|
| `name` | `str` | Returns `"BasicCriticAgent"` |

### Methods

| Method | Visibility | Purpose |
|--------|------------|---------|
| `describe()` | Public | Returns agent description |
| `evaluate(inp)` | Public | Main evaluation method |

---

## Core Functionality

### The `evaluate()` Method

Primary method that assesses answer quality.

**Signature:**
```python
def evaluate(self, inp: CriticInput) -> CriticOutput
```

**Parameters:**
- `inp` (`CriticInput`): Contains answer, context snippets, and plan

**Returns:**
- `CriticOutput`: Contains scores, issues, and notes

**Processing Steps:**

1. **Extract Inputs**
   - Get context snippets (default to empty list)
   - Get answer text (default to empty string)
   - Count snippets used

2. **Calculate Coverage**
   - `max_k = max(1, plan.top_k)` (avoid division by zero)
   - `coverage = min(1.0, num_snippets / max_k)`
   - Capped at 1.0 to handle cases where snippets > top_k

3. **Calculate Hallucination Risk**
   - `halluc_risk = 1.0 - coverage`
   - Simple inverse relationship

4. **Run Quality Checks**
   - No context → hallucination warning
   - Short answer (< 10 words) → elaboration suggestion
   - Low coverage (< 30%) → relevance warning

5. **Build Output**
   - Package scores and notes
   - Return empty lists for unused fields

---

## Evaluation Metrics

### Coverage Score

**Formula:**
```
coverage = min(1.0, num_snippets / max(1, top_k))
```

**Interpretation:**

| Coverage | Meaning |
|----------|---------|
| 1.0 | Full coverage - used all available context |
| 0.7-0.99 | Good coverage - most context utilized |
| 0.3-0.69 | Moderate coverage - some context gaps |
| 0.0-0.29 | Low coverage - minimal context used |
| 0.0 | No coverage - no context available |

**Edge Cases:**

| Scenario | num_snippets | top_k | Coverage |
|----------|--------------|-------|----------|
| Normal | 5 | 10 | 0.5 |
| Full utilization | 10 | 10 | 1.0 |
| Over-retrieval | 15 | 10 | 1.0 (capped) |
| No context | 0 | 10 | 0.0 |
| Zero top_k | 5 | 0 | 5.0 → capped logic needed |

### Hallucination Risk

**Formula:**
```
hallucination_risk = 1.0 - coverage
```

**Interpretation:**

| Risk | Meaning |
|------|---------|
| 0.0 | Low risk - answer well-grounded in context |
| 0.3-0.5 | Moderate risk - some unsupported content possible |
| 0.7-1.0 | High risk - answer may be largely hallucinated |
| 1.0 | Maximum risk - no supporting context |

**Assumptions:**
- More context = more grounded answer
- Linear relationship between coverage and risk
- Does NOT check if answer actually uses the context

### Quality Check Thresholds

| Check | Threshold | Message |
|-------|-----------|---------|
| No context | `len(ctx) == 0` | "No retrieval context available; answer may be hallucinated." |
| Short answer | `words < 10` | "Answer is very short; consider elaborating if user needs detail." |
| Low coverage | `coverage < 0.3` | "Low coverage of available context (few relevant snippets)." |

---

## Data Flow

### Input Schema: `CriticInput`

```python
@dataclass
class CriticInput:
    answer: Answer                    # Generated answer with .text
    context_snippets: List[Snippet]   # Retrieved/used snippets
    plan: Plan                        # Plan with .top_k
```

### Output Schema: `CriticOutput`

```python
@dataclass
class CriticOutput:
    hallucination_risk: float         # 0.0 - 1.0
    coverage_score: float             # 0.0 - 1.0
    missing_topics: List[str]         # (unused - always [])
    ambiguities: List[str]            # (unused - always [])
    unsupported_claims: List[str]     # (unused - always [])
    notes: List[str]                  # Human-readable warnings
```

### Notes Field Values

The `notes` field can contain any combination of:

```python
[
    "No retrieval context available; answer may be hallucinated.",
    "Answer is very short; consider elaborating if user needs detail.",
    "Low coverage of available context (few relevant snippets).",
]
```

---

## Testing Strategies

### Unit Tests

#### 1. Coverage Calculation Tests

```python
import pytest
from unittest.mock import Mock
from critic_basic_agent import BasicCriticAgent
from core.schemas import CriticInput, Answer

@pytest.fixture
def agent():
    return BasicCriticAgent()

@pytest.fixture
def make_input():
    def _make(num_snippets: int, top_k: int, answer_text: str = "A normal answer with enough words."):
        answer = Mock()
        answer.text = answer_text
        
        plan = Mock()
        plan.top_k = top_k
        
        snippets = [Mock() for _ in range(num_snippets)]
        
        return CriticInput(
            answer=answer,
            context_snippets=snippets,
            plan=plan
        )
    return _make

class TestCoverageCalculation:
    
    def test_full_coverage(self, agent, make_input):
        inp = make_input(num_snippets=10, top_k=10)
        output = agent.evaluate(inp)
        
        assert output.coverage_score == 1.0
        assert output.hallucination_risk == 0.0
    
    def test_half_coverage(self, agent, make_input):
        inp = make_input(num_snippets=5, top_k=10)
        output = agent.evaluate(inp)
        
        assert output.coverage_score == 0.5
        assert output.hallucination_risk == 0.5
    
    def test_no_coverage(self, agent, make_input):
        inp = make_input(num_snippets=0, top_k=10)
        output = agent.evaluate(inp)
        
        assert output.coverage_score == 0.0
        assert output.hallucination_risk == 1.0
    
    def test_over_coverage_capped(self, agent, make_input):
        inp = make_input(num_snippets=15, top_k=10)
        output = agent.evaluate(inp)
        
        assert output.coverage_score == 1.0  # Capped at 1.0
        assert output.hallucination_risk == 0.0
    
    def test_zero_top_k_handled(self, agent, make_input):
        inp = make_input(num_snippets=5, top_k=0)
        output = agent.evaluate(inp)
        
        # max(1, 0) = 1, so coverage = 5/1 = 5, capped to 1.0
        assert output.coverage_score == 1.0
    
    def test_low_coverage_threshold(self, agent, make_input):
        inp = make_input(num_snippets=2, top_k=10)
        output = agent.evaluate(inp)
        
        assert output.coverage_score == 0.2
        assert "Low coverage" in output.notes[0] or any("Low coverage" in n for n in output.notes)
```

#### 2. Quality Check Tests

```python
class TestQualityChecks:
    
    def test_no_context_warning(self, agent, make_input):
        inp = make_input(num_snippets=0, top_k=10)
        output = agent.evaluate(inp)
        
        assert any("No retrieval context" in note for note in output.notes)
    
    def test_short_answer_warning(self, agent, make_input):
        inp = make_input(num_snippets=5, top_k=10, answer_text="Too short")
        output = agent.evaluate(inp)
        
        assert any("very short" in note for note in output.notes)
    
    def test_short_answer_threshold(self, agent, make_input):
        # Exactly 10 words should NOT trigger warning
        inp = make_input(
            num_snippets=5, 
            top_k=10, 
            answer_text="one two three four five six seven eight nine ten"
        )
        output = agent.evaluate(inp)
        
        assert not any("very short" in note for note in output.notes)
    
    def test_nine_words_triggers_warning(self, agent, make_input):
        inp = make_input(
            num_snippets=5, 
            top_k=10, 
            answer_text="one two three four five six seven eight nine"
        )
        output = agent.evaluate(inp)
        
        assert any("very short" in note for note in output.notes)
    
    def test_low_coverage_warning(self, agent, make_input):
        inp = make_input(num_snippets=2, top_k=10)
        output = agent.evaluate(inp)
        
        assert any("Low coverage" in note for note in output.notes)
    
    def test_coverage_at_threshold_no_warning(self, agent, make_input):
        # 30% coverage should NOT trigger warning
        inp = make_input(num_snippets=3, top_k=10)
        output = agent.evaluate(inp)
        
        # 3/10 = 0.3, which is NOT < 0.3
        assert not any("Low coverage" in note for note in output.notes)
    
    def test_multiple_warnings(self, agent, make_input):
        inp = make_input(num_snippets=0, top_k=10, answer_text="Short")
        output = agent.evaluate(inp)
        
        # Should have both no context and short answer warnings
        assert len(output.notes) >= 2
```

#### 3. Edge Case Tests

```python
class TestEdgeCases:
    
    def test_none_context_snippets(self, agent):
        answer = Mock()
        answer.text = "A normal answer with enough words here."
        
        plan = Mock()
        plan.top_k = 10
        
        inp = CriticInput(
            answer=answer,
            context_snippets=None,  # None instead of list
            plan=plan
        )
        output = agent.evaluate(inp)
        
        # Should handle None gracefully
        assert output.coverage_score == 0.0
    
    def test_none_answer_text(self, agent, make_input):
        answer = Mock()
        answer.text = None
        
        plan = Mock()
        plan.top_k = 10
        
        inp = CriticInput(
            answer=answer,
            context_snippets=[Mock()],
            plan=plan
        )
        output = agent.evaluate(inp)
        
        # Should handle None text gracefully
        assert output.hallucination_risk >= 0
    
    def test_empty_answer_text(self, agent, make_input):
        inp = make_input(num_snippets=5, top_k=10, answer_text="")
        output = agent.evaluate(inp)
        
        # Empty string has 0 words when split
        assert any("very short" in note for note in output.notes)
    
    def test_negative_top_k(self, agent, make_input):
        inp = make_input(num_snippets=5, top_k=-5)
        output = agent.evaluate(inp)
        
        # max(1, -5) = 1
        assert output.coverage_score == 1.0
```

#### 4. Output Structure Tests

```python
class TestOutputStructure:
    
    def test_output_fields_present(self, agent, make_input):
        inp = make_input(num_snippets=5, top_k=10)
        output = agent.evaluate(inp)
        
        assert hasattr(output, 'hallucination_risk')
        assert hasattr(output, 'coverage_score')
        assert hasattr(output, 'missing_topics')
        assert hasattr(output, 'ambiguities')
        assert hasattr(output, 'unsupported_claims')
        assert hasattr(output, 'notes')
    
    def test_unused_fields_empty(self, agent, make_input):
        inp = make_input(num_snippets=5, top_k=10)
        output = agent.evaluate(inp)
        
        assert output.missing_topics == []
        assert output.ambiguities == []
        assert output.unsupported_claims == []
    
    def test_scores_in_valid_range(self, agent, make_input):
        for num_snippets in [0, 1, 5, 10, 20]:
            inp = make_input(num_snippets=num_snippets, top_k=10)
            output = agent.evaluate(inp)
            
            assert 0.0 <= output.coverage_score <= 1.0
            assert 0.0 <= output.hallucination_risk <= 1.0
```

#### 5. Agent Interface Tests

```python
class TestAgentInterface:
    
    def test_name_property(self, agent):
        assert agent.name == "BasicCriticAgent"
    
    def test_describe_method(self, agent):
        description = agent.describe()
        assert isinstance(description, str)
        assert "heuristic" in description.lower() or "critic" in description.lower()
    
    def test_role_attribute(self, agent):
        assert agent.role == "critic"
```

### Test Commands

```bash
# Run all critic tests
pytest test_critic_basic_agent.py -v

# Run with coverage
pytest test_critic_basic_agent.py --cov=critic_basic_agent --cov-report=html

# Run specific test class
pytest test_critic_basic_agent.py::TestCoverageCalculation -v
```

---

## Recommendations and Improvements

### Critical Issues

#### 1. Coverage Metric Doesn't Measure Actual Usage

**Problem:** Coverage only measures how many snippets were available, not whether the answer actually uses them.

**Current Logic:**
```python
coverage = num_snippets / top_k  # Just counts snippets
```

**Recommendation:** Add semantic overlap check:

```python
def _calculate_actual_coverage(self, answer_text: str, snippets: List) -> float:
    """Check if answer content overlaps with snippet content."""
    if not snippets or not answer_text:
        return 0.0
    
    answer_words = set(answer_text.lower().split())
    snippet_words = set()
    for s in snippets:
        text = getattr(s, 'text', '') or ''
        snippet_words.update(text.lower().split())
    
    if not snippet_words:
        return 0.0
    
    overlap = len(answer_words & snippet_words)
    return overlap / len(snippet_words)
```

#### 2. Unused Output Fields

**Problem:** `missing_topics`, `ambiguities`, and `unsupported_claims` are always empty.

**Recommendation:** Either implement or remove:

```python
# Option A: Implement basic detection
def _detect_unsupported_claims(self, answer: str, snippets: List) -> List[str]:
    """Find sentences in answer not supported by any snippet."""
    # ... implementation
    
# Option B: Remove from output schema if not needed
@dataclass
class CriticOutput:
    hallucination_risk: float
    coverage_score: float
    notes: List[str]
    # Remove unused fields
```

---

### High Priority Improvements

#### 3. Configurable Thresholds

**Problem:** Thresholds are hardcoded (10 words, 0.3 coverage).

**Recommendation:** Make configurable:

```python
@dataclass
class CriticConfig:
    min_answer_words: int = 10
    low_coverage_threshold: float = 0.3
    high_risk_threshold: float = 0.7

class BasicCriticAgent(CriticAgent):
    def __init__(self, config: CriticConfig = None):
        self.config = config or CriticConfig()
    
    def evaluate(self, inp: CriticInput) -> CriticOutput:
        # Use self.config.min_answer_words instead of 10
        if len(answer_text.split()) < self.config.min_answer_words:
            notes.append("Answer is very short...")
```

#### 4. Severity Levels for Notes

**Problem:** All notes have equal weight.

**Recommendation:** Add severity classification:

```python
from enum import Enum

class NoteSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"

@dataclass
class CriticNote:
    message: str
    severity: NoteSeverity
    metric: str  # e.g., "coverage", "length"

# Usage
notes.append(CriticNote(
    message="No retrieval context available",
    severity=NoteSeverity.ERROR,
    metric="coverage"
))
```

#### 5. Logging and Observability

**Problem:** No visibility into evaluation decisions.

**Recommendation:** Add structured logging:

```python
import logging
logger = logging.getLogger(__name__)

def evaluate(self, inp: CriticInput) -> CriticOutput:
    # ... calculations ...
    
    logger.info(
        "critic_evaluation",
        extra={
            "num_snippets": num_snips,
            "top_k": inp.plan.top_k,
            "coverage_score": coverage,
            "hallucination_risk": halluc_risk,
            "answer_word_count": len(answer_text.split()),
            "num_notes": len(notes),
        }
    )
    
    return output
```

---

### Medium Priority Improvements

#### 6. Query-Answer Relevance Check

**Problem:** No check if answer is relevant to the query.

**Recommendation:** Add relevance scoring:

```python
def _check_query_relevance(self, query: str, answer: str) -> float:
    """Check semantic similarity between query and answer."""
    query_words = set(query.lower().split())
    answer_words = set(answer.lower().split())
    
    # Remove stopwords
    stopwords = {"the", "a", "an", "is", "are", "was", "were", ...}
    query_words -= stopwords
    answer_words -= stopwords
    
    if not query_words:
        return 1.0  # Can't evaluate
    
    overlap = len(query_words & answer_words)
    return overlap / len(query_words)
```

#### 7. Answer Quality Metrics

**Problem:** Only checks word count, not other quality aspects.

**Recommendation:** Add more quality checks:

```python
def _check_answer_quality(self, answer: str) -> List[str]:
    notes = []
    
    # Check for question marks (answer shouldn't be a question)
    if answer.strip().endswith("?"):
        notes.append("Answer ends with a question; may be incomplete.")
    
    # Check for uncertainty markers
    uncertainty_phrases = ["I'm not sure", "I don't know", "possibly", "maybe"]
    if any(phrase in answer.lower() for phrase in uncertainty_phrases):
        notes.append("Answer contains uncertainty markers.")
    
    # Check for repetition
    sentences = answer.split(".")
    if len(sentences) != len(set(sentences)):
        notes.append("Answer may contain repeated content.")
    
    return notes
```

#### 8. Confidence Score

**Problem:** No overall confidence score combining all metrics.

**Recommendation:** Add composite confidence:

```python
def _calculate_confidence(
    self,
    coverage: float,
    answer_length: int,
    num_notes: int
) -> float:
    """Calculate overall confidence score."""
    # Base confidence from coverage
    confidence = coverage * 0.5
    
    # Bonus for longer answers (up to 0.3)
    length_score = min(0.3, answer_length / 100 * 0.3)
    confidence += length_score
    
    # Penalty for each note
    confidence -= num_notes * 0.1
    
    return max(0.0, min(1.0, confidence))
```

---

### Low Priority / Future Enhancements

#### 9. LLM-Based Critic Option

**Recommendation:** Add optional LLM evaluation:

```python
class LLMCriticAgent(CriticAgent):
    """LLM-based critic for deeper analysis."""
    
    def evaluate(self, inp: CriticInput) -> CriticOutput:
        prompt = f"""
        Evaluate this answer for:
        1. Hallucination risk (0-1)
        2. Coverage of the question
        3. Unsupported claims
        
        Question: {inp.query}
        Answer: {inp.answer.text}
        Context: {inp.context_snippets}
        
        Return JSON with your evaluation.
        """
        # ... LLM call
```

#### 10. Claim Extraction and Verification

**Recommendation:** Extract and verify individual claims:

```python
def _extract_claims(self, answer: str) -> List[str]:
    """Extract factual claims from answer."""
    # Use sentence splitting and claim detection
    pass

def _verify_claim(self, claim: str, snippets: List) -> bool:
    """Check if claim is supported by any snippet."""
    pass
```

#### 11. Historical Comparison

**Recommendation:** Compare with previous evaluations:

```python
class BasicCriticAgent(CriticAgent):
    def __init__(self):
        self._history: List[CriticOutput] = []
    
    def evaluate(self, inp: CriticInput) -> CriticOutput:
        output = self._evaluate_internal(inp)
        
        # Compare with history
        if self._history:
            avg_coverage = sum(h.coverage_score for h in self._history) / len(self._history)
            if output.coverage_score < avg_coverage * 0.5:
                output.notes.append("Coverage significantly below historical average.")
        
        self._history.append(output)
        return output
```

---

## Usage Examples

### Basic Usage

```python
from critic_basic_agent import BasicCriticAgent
from core.schemas import CriticInput, Answer, Plan, Snippet

# Initialize agent
agent = BasicCriticAgent()

# Create input
answer = Answer(text="RAG combines retrieval with generation for better accuracy.")
snippets = [
    Snippet(chunk_id="1", text="RAG retrieves relevant documents...", score=0.9),
    Snippet(chunk_id="2", text="Generation is improved with context...", score=0.8),
]
plan = Plan(top_k=10, ...)

inp = CriticInput(
    answer=answer,
    context_snippets=snippets,
    plan=plan
)

# Evaluate
output = agent.evaluate(inp)

print(f"Coverage: {output.coverage_score:.2f}")  # 0.20
print(f"Hallucination Risk: {output.hallucination_risk:.2f}")  # 0.80
print(f"Notes: {output.notes}")  # ["Low coverage..."]
```

### Pipeline Integration

```python
class RAGPipeline:
    def __init__(self):
        self.retriever = HybridRetrievalAgent()
        self.generator = LLMGeneratorAgent()
        self.critic = BasicCriticAgent()
    
    def process(self, query: str, plan: Plan) -> Dict[str, Any]:
        # Retrieve and generate
        retriever_output = self.retriever.retrieve(...)
        generator_output = self.generator.run(...)
        
        # Evaluate
        critic_input = CriticInput(
            answer=generator_output["answer"],
            context_snippets=self._flatten_snippets(retriever_output),
            plan=plan
        )
        critic_output = self.critic.evaluate(critic_input)
        
        # Check quality thresholds
        if critic_output.hallucination_risk > 0.7:
            logger.warning("High hallucination risk detected")
        
        return {
            "answer": generator_output["answer_text"],
            "quality": {
                "coverage": critic_output.coverage_score,
                "hallucination_risk": critic_output.hallucination_risk,
                "notes": critic_output.notes,
            }
        }
```

### Conditional Retry Based on Evaluation

```python
class QualityAwareRAGPipeline:
    MAX_RETRIES = 3
    MIN_COVERAGE = 0.3
    
    def process(self, query: str, plan: Plan) -> Dict[str, Any]:
        for attempt in range(self.MAX_RETRIES):
            # Generate answer
            answer = self._generate(query, plan)
            
            # Evaluate
            critic_output = self.critic.evaluate(CriticInput(
                answer=answer,
                context_snippets=self.snippets,
                plan=plan
            ))
            
            # Check if acceptable
            if critic_output.coverage_score >= self.MIN_COVERAGE:
                return {"answer": answer, "quality": critic_output}
            
            # Adjust plan for retry
            plan.top_k = int(plan.top_k * 1.5)
            logger.info(f"Retry {attempt + 1}: Increasing top_k to {plan.top_k}")
        
        # Return best effort with warning
        return {
            "answer": answer,
            "quality": critic_output,
            "warning": "Could not achieve minimum coverage"
        }
```

### Quality Dashboard Integration

```python
def collect_quality_metrics(critic_outputs: List[CriticOutput]) -> Dict[str, Any]:
    """Aggregate quality metrics for monitoring."""
    return {
        "avg_coverage": sum(o.coverage_score for o in critic_outputs) / len(critic_outputs),
        "avg_hallucination_risk": sum(o.hallucination_risk for o in critic_outputs) / len(critic_outputs),
        "high_risk_count": sum(1 for o in critic_outputs if o.hallucination_risk > 0.7),
        "low_coverage_count": sum(1 for o in critic_outputs if o.coverage_score < 0.3),
        "total_notes": sum(len(o.notes) for o in critic_outputs),
    }
```

---

## Appendix

### Glossary

| Term | Definition |
|------|------------|
| **Coverage** | Proportion of available context actually used |
| **Hallucination** | Generated content not supported by sources |
| **Heuristic** | Rule-based approach without ML |
| **top_k** | Number of documents to retrieve |

### Metric Formulas

| Metric | Formula | Range |
|--------|---------|-------|
| Coverage | `min(1.0, num_snippets / max(1, top_k))` | 0.0 - 1.0 |
| Hallucination Risk | `1.0 - coverage` | 0.0 - 1.0 |

### Threshold Reference

| Check | Threshold | Condition |
|-------|-----------|-----------|
| Short answer | 10 words | `len(answer.split()) < 10` |
| Low coverage | 0.3 | `coverage < 0.3` |
| No context | 0 snippets | `len(ctx) == 0` |

### Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | Initial | Basic heuristic evaluation |

### References

- Radiant Repository: https://github.com/dshipley71/radiant
- Related Files: `core/schemas.py`, `core/interfaces.py`, `orchestrator.py`

---

*Document generated for the Radiant RAG Pipeline project. For questions or contributions, please refer to the project repository.*
