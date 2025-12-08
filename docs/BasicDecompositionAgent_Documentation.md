# BasicDecompositionAgent Documentation

## Technical Reference for the Radiant RAG Pipeline Query Decomposition

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture Context](#architecture-context)
3. [Class Structure](#class-structure)
4. [Core Functionality](#core-functionality)
5. [Decomposition Algorithms](#decomposition-algorithms)
6. [Data Flow](#data-flow)
7. [Testing Strategies](#testing-strategies)
8. [Recommendations and Improvements](#recommendations-and-improvements)
9. [Usage Examples](#usage-examples)

---

## Overview

### Purpose

The `BasicDecompositionAgent` is a query analysis component within the Radiant RAG pipeline responsible for breaking down complex, multi-part queries into simpler, independently retrievable subqueries. It also identifies comparison queries that require information about multiple entities.

### Key Responsibilities

- Detect and extract comparison pairs from "X vs Y" style queries
- Split conjunctive queries (using "and" / "&") into individual subqueries
- Flag queries as multi-part when decomposition is applicable
- Generate structured decomposition output for downstream processing

### Design Philosophy

The agent employs a **lightweight heuristic approach** using string pattern matching rather than NLP or LLM-based decomposition. This provides fast, deterministic results suitable for common query patterns at the cost of handling more complex linguistic structures.

---

## Architecture Context

### Position in the Radiant Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Query                               │
│        "Compare Python vs JavaScript and their frameworks"      │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                  BasicDecompositionAgent                        │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Detection:                                             │   │
│  │  • " vs " pattern → Comparison pair extraction          │   │
│  │  • " and " / " & " pattern → Subquery splitting         │   │
│  ├─────────────────────────────────────────────────────────┤   │
│  │  Output:                                                │   │
│  │  • is_multi_part: true                                  │   │
│  │  • comparison_pairs: [{left: "Python", right: "JS"}]    │   │
│  │  • subqueries: [{id: "sub-1", text: "..."}]            │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Decomposition Output                         │
│  Used by: Retriever (parallel retrieval), Generator (synthesis) │
└─────────────────────────────────────────────────────────────────┘
```

### Related Components

| Component | Relationship |
|-----------|--------------|
| `DecompositionAgent` | Abstract base class (from `core.interfaces`) |
| `DecompositionInput` | Input schema containing user query |
| `DecompositionOutput` | Output schema wrapping the Decomposition |
| `Decomposition` | Data structure with decomposition results |
| `Subquery` | Individual subquery with ID and text |
| `ComparisonPair` | Pair of entities to compare |

---

## Class Structure

### Inheritance

```python
class BasicDecompositionAgent(DecompositionAgent):
    """Heuristic decomposition for multi-part and comparison queries."""
```

### Class Attributes

| Attribute | Type | Value | Description |
|-----------|------|-------|-------------|
| `role` | `str` | `"decomposition"` | Identifies the agent's role in the pipeline |

### Properties

| Property | Return Type | Description |
|----------|-------------|-------------|
| `name` | `str` | Returns `"BasicDecompositionAgent"` |

### Methods

| Method | Visibility | Purpose |
|--------|------------|---------|
| `describe()` | Public | Returns human-readable description |
| `decompose(inp)` | Public | Main entry point for query decomposition |

---

## Core Functionality

### The `decompose()` Method

This is the primary entry point that orchestrates all decomposition logic.

**Signature:**
```python
def decompose(self, inp: DecompositionInput) -> DecompositionOutput
```

**Parameters:**
- `inp` (`DecompositionInput`): Contains the user query to decompose

**Returns:**
- `DecompositionOutput`: Contains the `Decomposition` object with results

**Processing Steps:**

1. **Normalize Query**
   - Strip leading/trailing whitespace
   - Create lowercase version for pattern matching

2. **Comparison Detection**
   - Check for " vs " pattern (case-insensitive)
   - Split into left and right entities
   - Create `ComparisonPair` object

3. **Subquery Extraction**
   - Replace " & " with " and " for normalization
   - Split on " and "
   - Filter empty parts
   - Create `Subquery` objects with sequential IDs

4. **Multi-part Flag**
   - Set `is_multi_part = True` if any subqueries OR comparison pairs exist

5. **Output Assembly**
   - Package results into `Decomposition` and `DecompositionOutput`

---

## Decomposition Algorithms

### Comparison Pair Extraction

**Pattern:** ` vs ` (case-insensitive, space-padded)

**Algorithm:**
```
Input: "Python vs JavaScript"
       ─────────┬─────────
              split
       ┌───────┴───────┐
       ▼               ▼
    "Python"    "JavaScript"
       │               │
       └───────┬───────┘
               ▼
    ComparisonPair(left="Python", right="JavaScript")
```

**Edge Cases:**

| Input | Result |
|-------|--------|
| `"Python vs JavaScript"` | `ComparisonPair("Python", "JavaScript")` |
| `"A vs B vs C"` | `ComparisonPair("A", "B vs C")` ⚠️ |
| `"versus"` | No match (requires " vs ") |
| `"PythonvsJavaScript"` | No match (requires spaces) |

### Subquery Splitting

**Patterns:** ` and `, ` & ` (both normalized to ` and `)

**Algorithm:**
```
Input: "pricing & features and support"
              │
              ▼ (normalize)
       "pricing and features and support"
              │
              ▼ (split on " and ")
       ┌──────┼──────┐
       ▼      ▼      ▼
   "pricing" "features" "support"
       │      │         │
       ▼      ▼         ▼
    sub-1   sub-2     sub-3
```

**Edge Cases:**

| Input | Result |
|-------|--------|
| `"A and B"` | 2 subqueries |
| `"A & B"` | 2 subqueries |
| `"A and B and C"` | 3 subqueries |
| `"A"` | 0 subqueries (single part) |
| `"A and "` | 1 subquery ("A") |
| `"brand new"` | 0 subqueries (no split) |

### Combined Detection Matrix

| Query Pattern | is_multi_part | comparison_pairs | subqueries |
|---------------|---------------|------------------|------------|
| Simple query | `False` | `[]` | `[]` |
| `"A vs B"` | `True` | `[{A, B}]` | `[]` |
| `"A and B"` | `True` | `[]` | `[sub-1, sub-2]` |
| `"A vs B and C"` | `True` | `[{A vs B, C}]`* | `[sub-1, sub-2]` |
| `"A and B vs C"` | `True` | `[{A and B, C}]`* | `[sub-1, sub-2]` |

*Note: Current implementation has order-dependent behavior - see recommendations.

---

## Data Flow

### Input Schema: `DecompositionInput`

```python
@dataclass
class DecompositionInput:
    user_query: str  # Raw user query text
```

### Output Schema: `DecompositionOutput`

```python
@dataclass
class DecompositionOutput:
    decomposition: Decomposition
```

### Decomposition Schema

```python
@dataclass
class Decomposition:
    is_multi_part: bool              # True if query was decomposed
    subqueries: List[Subquery]       # List of extracted subqueries
    comparison_pairs: List[ComparisonPair]  # List of comparison pairs
```

### Subquery Schema

```python
@dataclass
class Subquery:
    id: str    # Sequential identifier (e.g., "sub-1")
    text: str  # Subquery text content
```

### ComparisonPair Schema

```python
@dataclass
class ComparisonPair:
    left: str   # Left entity in comparison
    right: str  # Right entity in comparison
```

---

## Testing Strategies

### Unit Tests

#### 1. Basic Decomposition Tests

```python
import pytest
from decomposition_basic_agent import BasicDecompositionAgent
from core.schemas import DecompositionInput

@pytest.fixture
def agent():
    return BasicDecompositionAgent()

class TestBasicDecomposition:
    
    def test_simple_query_no_decomposition(self, agent):
        inp = DecompositionInput(user_query="What is machine learning?")
        output = agent.decompose(inp)
        
        assert output.decomposition.is_multi_part == False
        assert output.decomposition.subqueries == []
        assert output.decomposition.comparison_pairs == []
    
    def test_whitespace_handling(self, agent):
        inp = DecompositionInput(user_query="  spaced query  ")
        output = agent.decompose(inp)
        
        assert output.decomposition.is_multi_part == False
```

#### 2. Comparison Detection Tests

```python
class TestComparisonDetection:
    
    def test_basic_vs_comparison(self, agent):
        inp = DecompositionInput(user_query="Python vs JavaScript")
        output = agent.decompose(inp)
        
        assert output.decomposition.is_multi_part == True
        assert len(output.decomposition.comparison_pairs) == 1
        
        pair = output.decomposition.comparison_pairs[0]
        assert pair.left == "Python"
        assert pair.right == "JavaScript"
    
    def test_vs_case_insensitive(self, agent):
        inp = DecompositionInput(user_query="Python VS JavaScript")
        output = agent.decompose(inp)
        
        assert len(output.decomposition.comparison_pairs) == 1
    
    def test_vs_with_context(self, agent):
        inp = DecompositionInput(
            user_query="Compare Python vs JavaScript for web development"
        )
        output = agent.decompose(inp)
        
        pair = output.decomposition.comparison_pairs[0]
        assert pair.left == "Compare Python"
        assert pair.right == "JavaScript for web development"
    
    def test_multiple_vs_only_first_split(self, agent):
        inp = DecompositionInput(user_query="A vs B vs C")
        output = agent.decompose(inp)
        
        pair = output.decomposition.comparison_pairs[0]
        assert pair.left == "A"
        assert pair.right == "B vs C"
    
    def test_versus_spelled_out_not_matched(self, agent):
        inp = DecompositionInput(user_query="Python versus JavaScript")
        output = agent.decompose(inp)
        
        # Current implementation only matches " vs "
        assert len(output.decomposition.comparison_pairs) == 0
    
    def test_vs_without_spaces_not_matched(self, agent):
        inp = DecompositionInput(user_query="PythonvsJavaScript")
        output = agent.decompose(inp)
        
        assert len(output.decomposition.comparison_pairs) == 0
```

#### 3. Subquery Splitting Tests

```python
class TestSubquerySplitting:
    
    def test_and_splitting(self, agent):
        inp = DecompositionInput(user_query="pricing and features")
        output = agent.decompose(inp)
        
        assert output.decomposition.is_multi_part == True
        assert len(output.decomposition.subqueries) == 2
        assert output.decomposition.subqueries[0].id == "sub-1"
        assert output.decomposition.subqueries[0].text == "pricing"
        assert output.decomposition.subqueries[1].id == "sub-2"
        assert output.decomposition.subqueries[1].text == "features"
    
    def test_ampersand_splitting(self, agent):
        inp = DecompositionInput(user_query="pricing & features")
        output = agent.decompose(inp)
        
        assert len(output.decomposition.subqueries) == 2
    
    def test_multiple_and_parts(self, agent):
        inp = DecompositionInput(user_query="A and B and C and D")
        output = agent.decompose(inp)
        
        assert len(output.decomposition.subqueries) == 4
        for i, sq in enumerate(output.decomposition.subqueries, start=1):
            assert sq.id == f"sub-{i}"
    
    def test_single_part_no_subqueries(self, agent):
        inp = DecompositionInput(user_query="single query")
        output = agent.decompose(inp)
        
        assert len(output.decomposition.subqueries) == 0
    
    def test_empty_parts_filtered(self, agent):
        inp = DecompositionInput(user_query="A and  and B")
        output = agent.decompose(inp)
        
        # Empty middle part should be filtered
        texts = [sq.text for sq in output.decomposition.subqueries]
        assert "" not in texts
    
    def test_and_in_word_not_split(self, agent):
        # "brand" contains "and" but shouldn't split
        inp = DecompositionInput(user_query="brand new product")
        output = agent.decompose(inp)
        
        # Should NOT split because " and " (with spaces) not present
        assert len(output.decomposition.subqueries) == 0
    
    def test_mixed_and_ampersand(self, agent):
        inp = DecompositionInput(user_query="A & B and C")
        output = agent.decompose(inp)
        
        assert len(output.decomposition.subqueries) == 3
```

#### 4. Combined Pattern Tests

```python
class TestCombinedPatterns:
    
    def test_vs_and_and_together(self, agent):
        inp = DecompositionInput(user_query="Python vs JavaScript and their ecosystems")
        output = agent.decompose(inp)
        
        assert output.decomposition.is_multi_part == True
        assert len(output.decomposition.comparison_pairs) == 1
        assert len(output.decomposition.subqueries) == 2
    
    def test_comparison_preserves_original_case(self, agent):
        inp = DecompositionInput(user_query="Python vs JavaScript")
        output = agent.decompose(inp)
        
        pair = output.decomposition.comparison_pairs[0]
        # Original case should be preserved in output
        assert pair.left == "Python"  # Not "python"
        assert pair.right == "JavaScript"  # Not "javascript"
```

#### 5. Edge Case Tests

```python
class TestEdgeCases:
    
    def test_empty_query(self, agent):
        inp = DecompositionInput(user_query="")
        output = agent.decompose(inp)
        
        assert output.decomposition.is_multi_part == False
    
    def test_only_whitespace(self, agent):
        inp = DecompositionInput(user_query="   ")
        output = agent.decompose(inp)
        
        assert output.decomposition.is_multi_part == False
    
    def test_only_vs(self, agent):
        inp = DecompositionInput(user_query=" vs ")
        output = agent.decompose(inp)
        
        # Should create comparison pair with empty strings
        assert len(output.decomposition.comparison_pairs) == 1
    
    def test_only_and(self, agent):
        inp = DecompositionInput(user_query=" and ")
        output = agent.decompose(inp)
        
        # Empty parts should be filtered
        assert len(output.decomposition.subqueries) == 0
    
    def test_unicode_query(self, agent):
        inp = DecompositionInput(user_query="日本語 vs 中文")
        output = agent.decompose(inp)
        
        assert len(output.decomposition.comparison_pairs) == 1
        assert output.decomposition.comparison_pairs[0].left == "日本語"
```

#### 6. Agent Interface Tests

```python
class TestAgentInterface:
    
    def test_name_property(self, agent):
        assert agent.name == "BasicDecompositionAgent"
    
    def test_describe_method(self, agent):
        description = agent.describe()
        assert isinstance(description, str)
        assert len(description) > 0
    
    def test_role_attribute(self, agent):
        assert agent.role == "decomposition"
```

### Performance Tests

```python
import time

class TestPerformance:
    
    def test_decomposition_latency(self, agent):
        """Decomposition should complete in < 0.1ms for typical queries."""
        inp = DecompositionInput(
            user_query="Compare Python vs JavaScript and Ruby and Go for web development"
        )
        
        start = time.perf_counter()
        for _ in range(10000):
            agent.decompose(inp)
        elapsed = time.perf_counter() - start
        
        avg_latency_ms = (elapsed / 10000) * 1000
        assert avg_latency_ms < 0.1, f"Average latency {avg_latency_ms}ms exceeds 0.1ms"
```

### Test Commands

```bash
# Run all decomposition tests
pytest test_decomposition_basic_agent.py -v

# Run with coverage
pytest test_decomposition_basic_agent.py --cov=decomposition_basic_agent --cov-report=html

# Run specific test class
pytest test_decomposition_basic_agent.py::TestComparisonDetection -v
```

---

## Recommendations and Improvements

### Critical Issues

#### 1. Order-Dependent Behavior with Combined Patterns

**Problem:** When both " vs " and " and " appear in a query, the order of detection affects results inconsistently.

**Example:**
```python
# "Python vs JavaScript and frameworks"
# Comparison: ("Python", "JavaScript and frameworks")
# Subqueries: ["Python", "JavaScript", "frameworks"]  # Overlapping!
```

**Recommendation:** Process patterns in a defined order and prevent overlap:

```python
def decompose(self, inp: DecompositionInput) -> DecompositionOutput:
    q = inp.user_query.strip()
    comparison_pairs = []
    subqueries = []
    
    # Step 1: Extract comparison first
    remaining = q
    if " vs " in q.lower():
        # Find the vs position and extract entities
        vs_idx = q.lower().index(" vs ")
        left = q[:vs_idx].strip()
        right = q[vs_idx + 4:].strip()
        comparison_pairs.append(ComparisonPair(left=left, right=right))
        
        # Don't split the comparison entities further
        remaining = None  # Skip subquery splitting
    
    # Step 2: Only split if no comparison found
    if remaining:
        parts = [p.strip() for p in remaining.replace(" & ", " and ").split(" and ") if p.strip()]
        if len(parts) > 1:
            for i, p in enumerate(parts, start=1):
                subqueries.append(Subquery(id=f"sub-{i}", text=p))
    
    # ...
```

---

### High Priority Improvements

#### 2. Support for "versus" Spelling

**Problem:** Only " vs " is detected, not "versus".

**Recommendation:** Expand comparison detection:

```python
COMPARISON_PATTERNS = [" vs ", " versus ", " compared to ", " or "]

def _extract_comparison(self, q: str) -> Optional[ComparisonPair]:
    lower = q.lower()
    for pattern in COMPARISON_PATTERNS:
        if pattern in lower:
            idx = lower.index(pattern)
            left = q[:idx].strip()
            right = q[idx + len(pattern):].strip()
            return ComparisonPair(left=left, right=right)
    return None
```

#### 3. Multiple Comparison Pairs Support

**Problem:** Only the first " vs " is processed; "A vs B vs C" yields incomplete results.

**Recommendation:** Support multiple comparison pairs:

```python
def _extract_all_comparisons(self, q: str) -> List[ComparisonPair]:
    pairs = []
    lower = q.lower()
    
    # For "A vs B vs C", create pairs: (A, B), (B, C)
    if " vs " in lower:
        parts = q.split(" vs ")
        parts = [p.strip() for p in parts]
        for i in range(len(parts) - 1):
            pairs.append(ComparisonPair(left=parts[i], right=parts[i + 1]))
    
    return pairs
```

#### 4. Smarter "and" Detection

**Problem:** "and" splitting is too aggressive - may split inside entity names.

**Example:** "Ben and Jerry's vs Häagen-Dazs" incorrectly splits "Ben and Jerry's"

**Recommendation:** Use NLP or pattern-based entity preservation:

```python
# Option 1: Preserve quoted strings
def _split_subqueries(self, q: str) -> List[str]:
    # Protect quoted strings
    protected = re.sub(r'"[^"]*"', lambda m: m.group().replace(" and ", "§AND§"), q)
    parts = protected.replace(" & ", " and ").split(" and ")
    return [p.strip().replace("§AND§", " and ") for p in parts if p.strip()]

# Option 2: Use spaCy for noun phrase preservation
import spacy
nlp = spacy.load("en_core_web_sm")

def _split_subqueries_nlp(self, q: str) -> List[str]:
    doc = nlp(q)
    # Preserve noun phrases containing "and"
    # ...
```

#### 5. Logging and Observability

**Problem:** No visibility into decomposition decisions.

**Recommendation:** Add structured logging:

```python
import logging
logger = logging.getLogger(__name__)

def decompose(self, inp: DecompositionInput) -> DecompositionOutput:
    q = inp.user_query.strip()
    
    # ... decomposition logic ...
    
    logger.info(
        "query_decomposed",
        extra={
            "original_query": q,
            "is_multi_part": dec.is_multi_part,
            "num_subqueries": len(subqueries),
            "num_comparisons": len(comparison_pairs),
            "subquery_texts": [sq.text for sq in subqueries],
            "comparison_entities": [(cp.left, cp.right) for cp in comparison_pairs],
        }
    )
    
    return DecompositionOutput(decomposition=dec)
```

---

### Medium Priority Improvements

#### 6. Configurable Patterns

**Problem:** Detection patterns are hardcoded.

**Recommendation:** Make patterns configurable:

```python
from dataclasses import dataclass, field
from typing import List

@dataclass
class DecompositionConfig:
    comparison_patterns: List[str] = field(
        default_factory=lambda: [" vs ", " versus "]
    )
    conjunction_patterns: List[str] = field(
        default_factory=lambda: [" and ", " & "]
    )
    min_subquery_length: int = 2

class BasicDecompositionAgent(DecompositionAgent):
    def __init__(self, config: DecompositionConfig = None):
        self.config = config or DecompositionConfig()
```

#### 7. Hierarchical Decomposition

**Problem:** Only single-level decomposition is supported.

**Recommendation:** Support nested decomposition:

```python
@dataclass
class Subquery:
    id: str
    text: str
    children: List[Subquery] = field(default_factory=list)

def decompose(self, inp: DecompositionInput) -> DecompositionOutput:
    # First level: major conjunctions
    # Second level: within each part, look for more patterns
    pass
```

#### 8. Confidence Scores

**Problem:** No indication of decomposition quality.

**Recommendation:** Return confidence scores:

```python
@dataclass
class Decomposition:
    is_multi_part: bool
    confidence: float  # 0.0 - 1.0
    subqueries: List[Subquery]
    comparison_pairs: List[ComparisonPair]

def decompose(self, inp: DecompositionInput) -> DecompositionOutput:
    # Higher confidence for clear patterns
    confidence = 1.0 if " vs " in q.lower() else 0.8
    # ...
```

---

### Low Priority / Future Enhancements

#### 9. LLM-Based Decomposition Option

**Recommendation:** Add an LLM-powered alternative for complex queries:

```python
class LLMDecompositionAgent(DecompositionAgent):
    def decompose(self, inp: DecompositionInput) -> DecompositionOutput:
        prompt = f"""Decompose this query into subqueries:
        Query: {inp.user_query}
        
        Return JSON with:
        - subqueries: list of independent questions
        - comparison_pairs: list of {{left, right}} for comparisons
        """
        # LLM call...
```

#### 10. Question Type Awareness

**Recommendation:** Adjust decomposition based on query type:

```python
def decompose(self, inp: DecompositionInput, query_type: str = None) -> DecompositionOutput:
    if query_type == "list":
        # List queries might have implicit subqueries
        pass
    elif query_type == "comparison":
        # Prioritize comparison extraction
        pass
```

#### 11. Coreference Resolution

**Problem:** Pronouns in subqueries lose context.

**Example:** "Python and its frameworks" → subquery "its frameworks" loses meaning

**Recommendation:** Resolve coreferences:

```python
def _resolve_coreferences(self, main_query: str, subquery: str) -> str:
    # Replace pronouns with their referents
    # "its" → "Python's"
    pass
```

---

## Usage Examples

### Basic Usage

```python
from decomposition_basic_agent import BasicDecompositionAgent
from core.schemas import DecompositionInput

# Initialize agent
agent = BasicDecompositionAgent()

# Simple query - no decomposition
inp = DecompositionInput(user_query="What is machine learning?")
output = agent.decompose(inp)
print(f"Multi-part: {output.decomposition.is_multi_part}")  # False

# Comparison query
inp = DecompositionInput(user_query="Python vs JavaScript")
output = agent.decompose(inp)
print(f"Multi-part: {output.decomposition.is_multi_part}")  # True
print(f"Comparisons: {output.decomposition.comparison_pairs}")
# [ComparisonPair(left='Python', right='JavaScript')]

# Conjunctive query
inp = DecompositionInput(user_query="pricing and features and support")
output = agent.decompose(inp)
print(f"Subqueries: {[(sq.id, sq.text) for sq in output.decomposition.subqueries]}")
# [('sub-1', 'pricing'), ('sub-2', 'features'), ('sub-3', 'support')]
```

### Pipeline Integration

```python
class RAGPipeline:
    def __init__(self):
        self.decomposer = BasicDecompositionAgent()
        self.retriever = RetrieverAgent()
        self.generator = GeneratorAgent()
    
    def process(self, query: str):
        # Step 1: Decompose query
        decomp_input = DecompositionInput(user_query=query)
        decomp_output = self.decomposer.decompose(decomp_input)
        decomp = decomp_output.decomposition
        
        # Step 2: Retrieve for each subquery
        all_documents = []
        
        if decomp.is_multi_part:
            # Handle subqueries
            for subquery in decomp.subqueries:
                docs = self.retriever.retrieve(subquery.text)
                all_documents.extend(docs)
            
            # Handle comparisons
            for pair in decomp.comparison_pairs:
                left_docs = self.retriever.retrieve(pair.left)
                right_docs = self.retriever.retrieve(pair.right)
                all_documents.extend(left_docs + right_docs)
        else:
            all_documents = self.retriever.retrieve(query)
        
        # Step 3: Deduplicate
        unique_documents = self._deduplicate(all_documents)
        
        # Step 4: Generate response
        return self.generator.generate(
            query=query,
            documents=unique_documents,
            decomposition=decomp  # Pass decomposition for structured response
        )
    
    def _deduplicate(self, documents):
        seen = set()
        unique = []
        for doc in documents:
            if doc.id not in seen:
                seen.add(doc.id)
                unique.append(doc)
        return unique
```

### Parallel Retrieval with Decomposition

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

class ParallelRAGPipeline:
    def __init__(self):
        self.decomposer = BasicDecompositionAgent()
        self.retriever = RetrieverAgent()
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    async def process(self, query: str):
        # Decompose
        decomp_input = DecompositionInput(user_query=query)
        decomp = self.decomposer.decompose(decomp_input).decomposition
        
        if not decomp.is_multi_part:
            return await self._retrieve_async(query)
        
        # Build retrieval tasks
        tasks = []
        
        for subquery in decomp.subqueries:
            tasks.append(self._retrieve_async(subquery.text))
        
        for pair in decomp.comparison_pairs:
            tasks.append(self._retrieve_async(pair.left))
            tasks.append(self._retrieve_async(pair.right))
        
        # Execute in parallel
        results = await asyncio.gather(*tasks)
        
        # Flatten and deduplicate
        all_docs = [doc for result in results for doc in result]
        return self._deduplicate(all_docs)
    
    async def _retrieve_async(self, query: str):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor, 
            self.retriever.retrieve, 
            query
        )
```

---

## Appendix

### Glossary

| Term | Definition |
|------|------------|
| **Decomposition** | Breaking a complex query into simpler sub-parts |
| **Subquery** | An independent query extracted from a multi-part query |
| **ComparisonPair** | Two entities that need to be compared |
| **Conjunction** | A word that joins parts of a query (and, &) |
| **Coreference** | When pronouns refer to previously mentioned entities |

### Pattern Reference

| Pattern | Detection | Example |
|---------|-----------|---------|
| ` vs ` | Comparison | "A vs B" |
| ` versus ` | Not detected* | "A versus B" |
| ` and ` | Conjunction | "A and B" |
| ` & ` | Conjunction | "A & B" |
| ` compared to ` | Not detected* | "A compared to B" |
| ` or ` | Not detected* | "A or B" |

*Marked for future implementation

### Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | Initial | Basic vs/and decomposition |

### References

- Radiant Repository: https://github.com/dshipley71/radiant
- Related Files: `orchestrator.py`, `core/schemas.py`, `core/interfaces.py`

---

*Document generated for the Radiant RAG Pipeline project. For questions or contributions, please refer to the project repository.*
