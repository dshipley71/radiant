# BasicPolicyAgent Documentation

## Technical Reference for the Radiant RAG Pipeline Decision Control

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture Context](#architecture-context)
3. [Class Structure](#class-structure)
4. [Core Functionality](#core-functionality)
5. [Decision Logic](#decision-logic)
6. [Data Flow](#data-flow)
7. [Testing Strategies](#testing-strategies)
8. [Recommendations and Improvements](#recommendations-and-improvements)
9. [Usage Examples](#usage-examples)

---

## Overview

### Purpose

The `BasicPolicyAgent` is the decision control component within the Radiant RAG pipeline. It determines whether the current answer is acceptable (finalize), needs query reformulation (rewrite), or should proceed with another retrieval iteration (continue) based on critic feedback and iteration budgets.

### Key Responsibilities

- Evaluate critic feedback (coverage, hallucination risk) against thresholds
- Manage iteration and rewrite budgets
- Make pipeline flow decisions (FINALIZE / REWRITE / CONTINUE)
- Provide human-readable reasons for decisions

### Design Philosophy

The agent implements a **budget-aware decision tree** that balances quality thresholds with resource constraints. It prioritizes quality when budget allows, but gracefully degrades to finalization when iterations are exhausted.

---

## Architecture Context

### Position in the Radiant Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    PolicyInput                                  │
│  critic_feedback | plan | iteration                             │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    BasicPolicyAgent                             │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Decision Tree:                                         │   │
│  │                                                         │   │
│  │  1. Quality Check                                       │   │
│  │     coverage ≥ 0.6 AND risk ≤ 0.4 → FINALIZE           │   │
│  │                                                         │   │
│  │  2. Rewrite Budget Check                                │   │
│  │     iteration < max_rewrites → REWRITE                  │   │
│  │                                                         │   │
│  │  3. Iteration Budget Check                              │   │
│  │     iteration + 1 < max_iters → CONTINUE                │   │
│  │                                                         │   │
│  │  4. Budget Exhausted                                    │   │
│  │     → FINALIZE (forced)                                 │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    PolicyOutput                                 │
│  decision: FINALIZE | REWRITE | CONTINUE                        │
│  reason: "..."                                                  │
│  adjustments: {}                                                │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
              Pipeline Control (Orchestrator)
```

### Pipeline Flow Control

```
                    ┌──────────────┐
                    │    Start     │
                    └──────┬───────┘
                           │
                           ▼
              ┌────────────────────────┐
              │  Retrieve & Generate   │
              └────────────┬───────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │    Critic Evaluate     │
              └────────────┬───────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │    Policy Decide       │
              └────────────┬───────────┘
                           │
           ┌───────────────┼───────────────┐
           │               │               │
           ▼               ▼               ▼
      ┌─────────┐    ┌──────────┐    ┌──────────┐
      │FINALIZE │    │ REWRITE  │    │ CONTINUE │
      └────┬────┘    └────┬─────┘    └────┬─────┘
           │              │               │
           ▼              │               │
      ┌─────────┐         │               │
      │  Done   │         ▼               ▼
      └─────────┘    ┌─────────┐    ┌──────────────┐
                     │ Rewrite │    │ Next Iter    │
                     │  Query  │    │ (no rewrite) │
                     └────┬────┘    └──────┬───────┘
                          │                │
                          └────────┬───────┘
                                   │
                                   ▼
                          Back to Retrieve
```

### Related Components

| Component | Relationship |
|-----------|--------------|
| `PolicyAgent` | Abstract base class (from `core.interfaces`) |
| `PolicyInput` | Input schema with critic feedback, plan, iteration |
| `PolicyOutput` | Output schema with decision, reason, adjustments |
| `DecisionEnum` | Enumeration of possible decisions |
| `CriticOutput` | Feedback from BasicCriticAgent |
| `PlanIterations` | Iteration budget configuration |

---

## Class Structure

### Inheritance

```python
class BasicPolicyAgent(PolicyAgent):
    """Heuristic policy for finalize / rewrite / continue decisions."""
```

### Class Attributes

| Attribute | Type | Value | Description |
|-----------|------|-------|-------------|
| `role` | `str` | `"policy"` | Agent role identifier |

### Properties

| Property | Return Type | Description |
|----------|-------------|-------------|
| `name` | `str` | Returns `"BasicPolicyAgent"` |

### Methods

| Method | Visibility | Purpose |
|--------|------------|---------|
| `describe()` | Public | Returns agent description |
| `decide(inp)` | Public | Main decision method |

---

## Core Functionality

### The `decide()` Method

Primary method that determines pipeline action.

**Signature:**
```python
def decide(self, inp: PolicyInput) -> PolicyOutput
```

**Parameters:**
- `inp` (`PolicyInput`): Contains critic feedback, plan, and current iteration

**Returns:**
- `PolicyOutput`: Contains decision, reason, and optional adjustments

**Processing Steps:**

1. **Extract Metrics**
   - `cov = inp.critic_feedback.coverage_score`
   - `risk = inp.critic_feedback.hallucination_risk`

2. **Extract Budgets**
   - `max_rewrites = max(0, inp.plan.iterations.max_rewrites)`
   - `max_iters = max(1, inp.plan.iterations.max_iters)`

3. **Quality Check**
   - If coverage ≥ 0.6 AND risk ≤ 0.4 → FINALIZE

4. **Rewrite Budget Check**
   - If iteration < max_rewrites → REWRITE

5. **Iteration Budget Check**
   - If iteration + 1 < max_iters → CONTINUE

6. **Default (Budget Exhausted)**
   - → FINALIZE

---

## Decision Logic

### Decision Tree Diagram

```
                         Start
                           │
                           ▼
              ┌────────────────────────┐
              │  coverage ≥ 0.6 AND    │
              │  risk ≤ 0.4 ?          │
              └────────────┬───────────┘
                      YES  │  NO
              ┌────────────┴───────────┐
              ▼                        ▼
         FINALIZE          ┌────────────────────┐
         (quality ok)      │ iteration <        │
                           │ max_rewrites ?     │
                           └─────────┬──────────┘
                                YES  │  NO
                           ┌─────────┴──────────┐
                           ▼                    ▼
                       REWRITE      ┌────────────────────┐
                       (improve)    │ iteration + 1 <    │
                                    │ max_iters ?        │
                                    └─────────┬──────────┘
                                         YES  │  NO
                                    ┌─────────┴──────────┐
                                    ▼                    ▼
                                CONTINUE             FINALIZE
                                (retry)              (exhausted)
```

### Decision Outcomes

| Decision | Meaning | Pipeline Action |
|----------|---------|-----------------|
| `FINALIZE` | Accept current answer | Return to user |
| `REWRITE` | Query needs reformulation | Rewrite query, re-retrieve |
| `CONTINUE` | Try again without rewrite | Re-retrieve with same query |

### Quality Thresholds

| Metric | Threshold | Condition for FINALIZE |
|--------|-----------|------------------------|
| Coverage | ≥ 0.6 | At least 60% context coverage |
| Hallucination Risk | ≤ 0.4 | At most 40% hallucination risk |

**Both conditions must be met** for quality-based finalization.

### Budget Constraints

| Budget | Condition | Effect |
|--------|-----------|--------|
| `max_rewrites` | `iteration < max_rewrites` | Allows REWRITE decision |
| `max_iters` | `iteration + 1 < max_iters` | Allows CONTINUE decision |

**Budget Enforcement:**
- `max_rewrites` is floored at 0 (no negative rewrites)
- `max_iters` is floored at 1 (at least one iteration)

### Decision Priority Order

1. **Quality First**: If quality is acceptable, always finalize (saves resources)
2. **Rewrite Preferred**: If quality is poor and rewrites available, prefer rewrite
3. **Continue as Fallback**: If no rewrites but iterations remain, continue
4. **Forced Finalization**: If all budgets exhausted, accept current answer

### Scenario Analysis

| coverage | risk | iteration | max_rewrites | max_iters | Decision |
|----------|------|-----------|--------------|-----------|----------|
| 0.7 | 0.3 | 0 | 2 | 3 | FINALIZE (quality ok) |
| 0.5 | 0.5 | 0 | 2 | 3 | REWRITE (poor quality, budget available) |
| 0.5 | 0.5 | 2 | 2 | 3 | CONTINUE (no rewrites left, iters available) |
| 0.5 | 0.5 | 2 | 2 | 3 | FINALIZE (next iter = 3, not < 3) |
| 0.3 | 0.7 | 0 | 0 | 1 | FINALIZE (no budget) |
| 0.6 | 0.4 | 0 | 2 | 3 | FINALIZE (exactly at thresholds) |
| 0.59 | 0.4 | 0 | 2 | 3 | REWRITE (coverage just below) |
| 0.6 | 0.41 | 0 | 2 | 3 | REWRITE (risk just above) |

---

## Data Flow

### Input Schema: `PolicyInput`

```python
@dataclass
class PolicyInput:
    critic_feedback: CriticOutput    # From BasicCriticAgent
    plan: Plan                       # With iterations budget
    iteration: int                   # Current iteration (0-indexed)
```

### Output Schema: `PolicyOutput`

```python
@dataclass
class PolicyOutput:
    decision: DecisionEnum           # FINALIZE | REWRITE | CONTINUE
    reason: str                      # Human-readable explanation
    adjustments: Dict[str, Any]      # Optional parameter adjustments
```

### DecisionEnum Values

```python
class DecisionEnum(Enum):
    FINALIZE = "finalize"
    REWRITE = "rewrite"
    CONTINUE = "continue"
```

### Reason Messages

| Decision | Reason |
|----------|--------|
| FINALIZE (quality) | "Coverage sufficient and hallucination risk acceptable." |
| REWRITE | "Coverage low or hallucination risk high; attempting rewrite." |
| CONTINUE | "Continuing without rewrite (iteration budget remains)." |
| FINALIZE (exhausted) | "Iteration budget exhausted." |

---

## Testing Strategies

### Unit Tests

#### 1. Quality Threshold Tests

```python
import pytest
from unittest.mock import Mock
from policy_basic_agent import BasicPolicyAgent
from core.schemas import PolicyInput, DecisionEnum

@pytest.fixture
def agent():
    return BasicPolicyAgent()

@pytest.fixture
def make_input():
    def _make(coverage: float, risk: float, iteration: int, max_rewrites: int, max_iters: int):
        critic_feedback = Mock()
        critic_feedback.coverage_score = coverage
        critic_feedback.hallucination_risk = risk
        
        iterations = Mock()
        iterations.max_rewrites = max_rewrites
        iterations.max_iters = max_iters
        
        plan = Mock()
        plan.iterations = iterations
        
        return PolicyInput(
            critic_feedback=critic_feedback,
            plan=plan,
            iteration=iteration
        )
    return _make

class TestQualityThresholds:
    
    def test_good_quality_finalizes(self, agent, make_input):
        inp = make_input(coverage=0.7, risk=0.3, iteration=0, max_rewrites=2, max_iters=3)
        output = agent.decide(inp)
        
        assert output.decision == DecisionEnum.FINALIZE
        assert "sufficient" in output.reason.lower()
    
    def test_exact_threshold_finalizes(self, agent, make_input):
        inp = make_input(coverage=0.6, risk=0.4, iteration=0, max_rewrites=2, max_iters=3)
        output = agent.decide(inp)
        
        assert output.decision == DecisionEnum.FINALIZE
    
    def test_coverage_just_below_threshold(self, agent, make_input):
        inp = make_input(coverage=0.59, risk=0.4, iteration=0, max_rewrites=2, max_iters=3)
        output = agent.decide(inp)
        
        assert output.decision == DecisionEnum.REWRITE
    
    def test_risk_just_above_threshold(self, agent, make_input):
        inp = make_input(coverage=0.6, risk=0.41, iteration=0, max_rewrites=2, max_iters=3)
        output = agent.decide(inp)
        
        assert output.decision == DecisionEnum.REWRITE
    
    def test_both_thresholds_failed(self, agent, make_input):
        inp = make_input(coverage=0.3, risk=0.7, iteration=0, max_rewrites=2, max_iters=3)
        output = agent.decide(inp)
        
        assert output.decision == DecisionEnum.REWRITE
```

#### 2. Budget Management Tests

```python
class TestBudgetManagement:
    
    def test_rewrite_when_budget_available(self, agent, make_input):
        inp = make_input(coverage=0.3, risk=0.7, iteration=0, max_rewrites=2, max_iters=3)
        output = agent.decide(inp)
        
        assert output.decision == DecisionEnum.REWRITE
    
    def test_rewrite_at_last_rewrite(self, agent, make_input):
        inp = make_input(coverage=0.3, risk=0.7, iteration=1, max_rewrites=2, max_iters=5)
        output = agent.decide(inp)
        
        assert output.decision == DecisionEnum.REWRITE
    
    def test_continue_when_rewrites_exhausted(self, agent, make_input):
        inp = make_input(coverage=0.3, risk=0.7, iteration=2, max_rewrites=2, max_iters=5)
        output = agent.decide(inp)
        
        assert output.decision == DecisionEnum.CONTINUE
    
    def test_finalize_when_all_budgets_exhausted(self, agent, make_input):
        inp = make_input(coverage=0.3, risk=0.7, iteration=2, max_rewrites=2, max_iters=3)
        output = agent.decide(inp)
        
        assert output.decision == DecisionEnum.FINALIZE
        assert "exhausted" in output.reason.lower()
    
    def test_zero_rewrites_skips_to_continue(self, agent, make_input):
        inp = make_input(coverage=0.3, risk=0.7, iteration=0, max_rewrites=0, max_iters=3)
        output = agent.decide(inp)
        
        assert output.decision == DecisionEnum.CONTINUE
    
    def test_single_iteration_finalizes_immediately(self, agent, make_input):
        inp = make_input(coverage=0.3, risk=0.7, iteration=0, max_rewrites=0, max_iters=1)
        output = agent.decide(inp)
        
        assert output.decision == DecisionEnum.FINALIZE
```

#### 3. Edge Case Tests

```python
class TestEdgeCases:
    
    def test_negative_max_rewrites_treated_as_zero(self, agent, make_input):
        inp = make_input(coverage=0.3, risk=0.7, iteration=0, max_rewrites=-5, max_iters=3)
        output = agent.decide(inp)
        
        # Should skip rewrite (0 rewrites available)
        assert output.decision == DecisionEnum.CONTINUE
    
    def test_zero_max_iters_treated_as_one(self, agent, make_input):
        inp = make_input(coverage=0.3, risk=0.7, iteration=0, max_rewrites=0, max_iters=0)
        output = agent.decide(inp)
        
        # Should finalize (1 iter, iteration 0, so 0+1 not < 1)
        assert output.decision == DecisionEnum.FINALIZE
    
    def test_negative_max_iters_treated_as_one(self, agent, make_input):
        inp = make_input(coverage=0.3, risk=0.7, iteration=0, max_rewrites=0, max_iters=-5)
        output = agent.decide(inp)
        
        assert output.decision == DecisionEnum.FINALIZE
    
    def test_coverage_zero(self, agent, make_input):
        inp = make_input(coverage=0.0, risk=1.0, iteration=0, max_rewrites=2, max_iters=3)
        output = agent.decide(inp)
        
        assert output.decision == DecisionEnum.REWRITE
    
    def test_perfect_scores(self, agent, make_input):
        inp = make_input(coverage=1.0, risk=0.0, iteration=0, max_rewrites=2, max_iters=3)
        output = agent.decide(inp)
        
        assert output.decision == DecisionEnum.FINALIZE
    
    def test_iteration_equals_max_iters(self, agent, make_input):
        # Edge case: iteration is AT max_iters (shouldn't happen but handle it)
        inp = make_input(coverage=0.3, risk=0.7, iteration=3, max_rewrites=0, max_iters=3)
        output = agent.decide(inp)
        
        assert output.decision == DecisionEnum.FINALIZE
```

#### 4. Output Structure Tests

```python
class TestOutputStructure:
    
    def test_output_has_required_fields(self, agent, make_input):
        inp = make_input(coverage=0.5, risk=0.5, iteration=0, max_rewrites=2, max_iters=3)
        output = agent.decide(inp)
        
        assert hasattr(output, 'decision')
        assert hasattr(output, 'reason')
        assert hasattr(output, 'adjustments')
    
    def test_adjustments_always_empty(self, agent, make_input):
        # Test various scenarios
        scenarios = [
            (0.7, 0.3, 0, 2, 3),  # FINALIZE quality
            (0.3, 0.7, 0, 2, 3),  # REWRITE
            (0.3, 0.7, 2, 2, 5),  # CONTINUE
            (0.3, 0.7, 4, 2, 5),  # FINALIZE exhausted
        ]
        
        for cov, risk, iter, rew, iters in scenarios:
            inp = make_input(cov, risk, iter, rew, iters)
            output = agent.decide(inp)
            assert output.adjustments == {}
    
    def test_reason_is_nonempty_string(self, agent, make_input):
        inp = make_input(coverage=0.5, risk=0.5, iteration=0, max_rewrites=2, max_iters=3)
        output = agent.decide(inp)
        
        assert isinstance(output.reason, str)
        assert len(output.reason) > 0
```

#### 5. Decision Sequence Tests

```python
class TestDecisionSequence:
    """Test typical iteration sequences."""
    
    def test_typical_improvement_sequence(self, agent, make_input):
        """Simulate improving quality over iterations."""
        # Iteration 0: Poor quality, rewrite
        inp = make_input(coverage=0.2, risk=0.8, iteration=0, max_rewrites=2, max_iters=5)
        assert agent.decide(inp).decision == DecisionEnum.REWRITE
        
        # Iteration 1: Slightly better, rewrite again
        inp = make_input(coverage=0.4, risk=0.6, iteration=1, max_rewrites=2, max_iters=5)
        assert agent.decide(inp).decision == DecisionEnum.REWRITE
        
        # Iteration 2: Good quality, finalize
        inp = make_input(coverage=0.7, risk=0.3, iteration=2, max_rewrites=2, max_iters=5)
        assert agent.decide(inp).decision == DecisionEnum.FINALIZE
    
    def test_exhaustion_sequence(self, agent, make_input):
        """Simulate exhausting all budgets without quality improvement."""
        # Iteration 0: Rewrite
        inp = make_input(coverage=0.3, risk=0.7, iteration=0, max_rewrites=1, max_iters=3)
        assert agent.decide(inp).decision == DecisionEnum.REWRITE
        
        # Iteration 1: Out of rewrites, continue
        inp = make_input(coverage=0.3, risk=0.7, iteration=1, max_rewrites=1, max_iters=3)
        assert agent.decide(inp).decision == DecisionEnum.CONTINUE
        
        # Iteration 2: Out of everything, finalize
        inp = make_input(coverage=0.3, risk=0.7, iteration=2, max_rewrites=1, max_iters=3)
        assert agent.decide(inp).decision == DecisionEnum.FINALIZE
```

#### 6. Agent Interface Tests

```python
class TestAgentInterface:
    
    def test_name_property(self, agent):
        assert agent.name == "BasicPolicyAgent"
    
    def test_describe_method(self, agent):
        description = agent.describe()
        assert isinstance(description, str)
        assert "policy" in description.lower()
    
    def test_role_attribute(self, agent):
        assert agent.role == "policy"
```

### Test Commands

```bash
# Run all policy tests
pytest test_policy_basic_agent.py -v

# Run with coverage
pytest test_policy_basic_agent.py --cov=policy_basic_agent --cov-report=html

# Run specific test class
pytest test_policy_basic_agent.py::TestDecisionSequence -v
```

---

## Recommendations and Improvements

### Critical Issues

#### 1. Adjustments Field Never Used

**Problem:** The `adjustments` field is always empty, representing dead functionality.

**Recommendation:** Either implement or document as reserved:

```python
# Option A: Implement adjustments
def decide(self, inp: PolicyInput) -> PolicyOutput:
    # ... decision logic ...
    
    adjustments = {}
    if decision == DecisionEnum.REWRITE:
        # Suggest query modifications
        if cov < 0.3:
            adjustments["expand_query"] = True
        if risk > 0.8:
            adjustments["increase_top_k"] = min(inp.plan.top_k * 1.5, 100)
    
    return PolicyOutput(decision=decision, reason=reason, adjustments=adjustments)
```

#### 2. Hardcoded Thresholds

**Problem:** Quality thresholds (0.6, 0.4) are hardcoded.

**Recommendation:** Make configurable:

```python
@dataclass
class PolicyConfig:
    min_coverage_threshold: float = 0.6
    max_risk_threshold: float = 0.4

class BasicPolicyAgent(PolicyAgent):
    def __init__(self, config: PolicyConfig = None):
        self.config = config or PolicyConfig()
    
    def decide(self, inp: PolicyInput) -> PolicyOutput:
        if cov >= self.config.min_coverage_threshold and risk <= self.config.max_risk_threshold:
            return PolicyOutput(decision=DecisionEnum.FINALIZE, ...)
```

---

### High Priority Improvements

#### 3. More Granular Decision Logic

**Problem:** Only two quality metrics considered; no context about query or answer.

**Recommendation:** Expand decision criteria:

```python
def decide(self, inp: PolicyInput) -> PolicyOutput:
    feedback = inp.critic_feedback
    
    # Consider additional signals
    has_notes = bool(feedback.notes)
    is_short_answer = "very short" in " ".join(feedback.notes or [])
    has_unsupported_claims = bool(feedback.unsupported_claims)
    
    # Quality score combining multiple factors
    quality_score = self._calculate_quality_score(
        coverage=feedback.coverage_score,
        risk=feedback.hallucination_risk,
        has_notes=has_notes,
        is_short=is_short_answer,
    )
    
    if quality_score >= self.config.quality_threshold:
        return PolicyOutput(decision=DecisionEnum.FINALIZE, ...)
```

#### 4. Logging and Observability

**Problem:** No visibility into decision process.

**Recommendation:** Add structured logging:

```python
import logging
logger = logging.getLogger(__name__)

def decide(self, inp: PolicyInput) -> PolicyOutput:
    # ... decision logic ...
    
    logger.info(
        "policy_decision",
        extra={
            "coverage": cov,
            "risk": risk,
            "iteration": inp.iteration,
            "max_rewrites": max_rewrites,
            "max_iters": max_iters,
            "decision": output.decision.value,
            "reason": output.reason,
        }
    )
    
    return output
```

#### 5. Dynamic Threshold Adjustment

**Problem:** Fixed thresholds don't adapt to query complexity.

**Recommendation:** Adjust thresholds based on context:

```python
def _get_dynamic_thresholds(self, inp: PolicyInput) -> Tuple[float, float]:
    """Adjust thresholds based on query complexity."""
    base_coverage = 0.6
    base_risk = 0.4
    
    # Relax thresholds for complex queries
    complexity = getattr(inp, 'complexity_hint', 'medium')
    if complexity == 'high':
        return base_coverage - 0.1, base_risk + 0.1  # 0.5, 0.5
    elif complexity == 'low':
        return base_coverage + 0.1, base_risk - 0.1  # 0.7, 0.3
    
    return base_coverage, base_risk
```

---

### Medium Priority Improvements

#### 6. Rewrite Strategy Selection

**Problem:** REWRITE decision doesn't specify what kind of rewrite.

**Recommendation:** Add rewrite strategy to adjustments:

```python
def decide(self, inp: PolicyInput) -> PolicyOutput:
    if decision == DecisionEnum.REWRITE:
        strategy = self._select_rewrite_strategy(feedback, inp.iteration)
        adjustments = {"rewrite_strategy": strategy}
        
        # Strategies:
        # - "expand": Add synonyms and related terms
        # - "simplify": Break into sub-queries
        # - "rephrase": Different wording, same intent
        # - "specialize": Add domain-specific terms

class RewriteStrategy(Enum):
    EXPAND = "expand"
    SIMPLIFY = "simplify"
    REPHRASE = "rephrase"
    SPECIALIZE = "specialize"
```

#### 7. Confidence in Decision

**Problem:** No indication of decision confidence.

**Recommendation:** Add confidence score:

```python
@dataclass
class PolicyOutput:
    decision: DecisionEnum
    reason: str
    adjustments: Dict[str, Any]
    confidence: float = 1.0  # 0.0 - 1.0

def decide(self, inp: PolicyInput) -> PolicyOutput:
    # High confidence when clearly meeting/missing thresholds
    if cov >= 0.8 and risk <= 0.2:
        confidence = 0.95  # Clearly good
    elif cov >= 0.6 and risk <= 0.4:
        confidence = 0.75  # Borderline
    else:
        confidence = 0.6   # Needs improvement
```

#### 8. Budget Utilization Feedback

**Problem:** No feedback on budget usage efficiency.

**Recommendation:** Track and report budget metrics:

```python
@dataclass
class PolicyOutput:
    decision: DecisionEnum
    reason: str
    adjustments: Dict[str, Any]
    budget_info: Dict[str, Any] = None

def decide(self, inp: PolicyInput) -> PolicyOutput:
    budget_info = {
        "rewrites_used": inp.iteration,
        "rewrites_remaining": max_rewrites - inp.iteration,
        "iterations_remaining": max_iters - inp.iteration - 1,
        "budget_utilization": inp.iteration / max_iters,
    }
    
    return PolicyOutput(
        decision=decision,
        reason=reason,
        adjustments={},
        budget_info=budget_info
    )
```

---

### Low Priority / Future Enhancements

#### 9. Learning from History

**Recommendation:** Track decision effectiveness:

```python
class BasicPolicyAgent(PolicyAgent):
    def __init__(self):
        self._decision_history: List[Dict] = []
    
    def record_outcome(self, decision: DecisionEnum, final_quality: float):
        """Record whether a decision led to quality improvement."""
        self._decision_history.append({
            "decision": decision,
            "outcome_quality": final_quality,
            "improved": final_quality > self._last_quality,
        })
    
    def get_decision_effectiveness(self) -> Dict[str, float]:
        """Calculate success rate per decision type."""
        # ... analysis
```

#### 10. Multi-Objective Optimization

**Recommendation:** Balance quality vs. cost:

```python
def decide(self, inp: PolicyInput) -> PolicyOutput:
    # Estimate cost of each option
    finalize_cost = 0  # No additional cost
    rewrite_cost = self._estimate_rewrite_cost(inp)
    continue_cost = self._estimate_continue_cost(inp)
    
    # Expected quality improvement
    rewrite_quality_gain = self._estimate_rewrite_gain(feedback)
    continue_quality_gain = self._estimate_continue_gain(feedback)
    
    # Choose option with best value
    # value = quality_gain / cost
```

#### 11. Explanation Generation

**Recommendation:** Provide detailed explanations:

```python
def decide(self, inp: PolicyInput) -> PolicyOutput:
    explanation_parts = []
    
    explanation_parts.append(f"Coverage: {cov:.2f} (threshold: 0.6)")
    explanation_parts.append(f"Risk: {risk:.2f} (threshold: 0.4)")
    explanation_parts.append(f"Iteration: {inp.iteration}/{max_iters}")
    
    if decision == DecisionEnum.REWRITE:
        explanation_parts.append(f"Rewrites remaining: {max_rewrites - inp.iteration}")
    
    return PolicyOutput(
        decision=decision,
        reason=reason,
        adjustments={},
        explanation="\n".join(explanation_parts)
    )
```

---

## Usage Examples

### Basic Usage

```python
from policy_basic_agent import BasicPolicyAgent
from core.schemas import PolicyInput, CriticOutput, Plan, PlanIterations

# Initialize agent
agent = BasicPolicyAgent()

# Create critic feedback
critic_feedback = CriticOutput(
    coverage_score=0.5,
    hallucination_risk=0.5,
    notes=["Low coverage"],
    missing_topics=[],
    ambiguities=[],
    unsupported_claims=[]
)

# Create plan with budgets
iterations = PlanIterations(max_iters=3, max_rewrites=2)
plan = Plan(iterations=iterations, ...)

# Create input
inp = PolicyInput(
    critic_feedback=critic_feedback,
    plan=plan,
    iteration=0
)

# Get decision
output = agent.decide(inp)

print(f"Decision: {output.decision}")  # REWRITE
print(f"Reason: {output.reason}")      # "Coverage low or hallucination risk high..."
```

### Pipeline Integration

```python
class RAGPipeline:
    def __init__(self):
        self.retriever = HybridRetrievalAgent()
        self.generator = LLMGeneratorAgent()
        self.critic = BasicCriticAgent()
        self.policy = BasicPolicyAgent()
    
    def process(self, query: str, plan: Plan) -> Dict[str, Any]:
        iteration = 0
        current_query = query
        
        while True:
            # Retrieve and generate
            retriever_output = self.retriever.retrieve(current_query, plan)
            generator_output = self.generator.run(current_query, retriever_output)
            
            # Evaluate
            critic_output = self.critic.evaluate(CriticInput(
                answer=generator_output["answer"],
                context_snippets=retriever_output.snippets,
                plan=plan
            ))
            
            # Decide next action
            policy_output = self.policy.decide(PolicyInput(
                critic_feedback=critic_output,
                plan=plan,
                iteration=iteration
            ))
            
            if policy_output.decision == DecisionEnum.FINALIZE:
                return {
                    "answer": generator_output["answer_text"],
                    "iterations": iteration + 1,
                    "final_coverage": critic_output.coverage_score,
                }
            
            elif policy_output.decision == DecisionEnum.REWRITE:
                current_query = self.rewriter.rewrite(query, critic_output)
            
            # CONTINUE uses same query
            iteration += 1
```

### Custom Thresholds

```python
# Strict quality requirements
strict_config = PolicyConfig(
    min_coverage_threshold=0.8,
    max_risk_threshold=0.2
)
strict_policy = BasicPolicyAgent(config=strict_config)

# Lenient quality requirements
lenient_config = PolicyConfig(
    min_coverage_threshold=0.4,
    max_risk_threshold=0.6
)
lenient_policy = BasicPolicyAgent(config=lenient_config)
```

### Decision Logging

```python
def process_with_logging(query: str, plan: Plan):
    decisions = []
    
    for iteration in range(plan.iterations.max_iters):
        # ... retrieval and generation ...
        
        policy_output = policy.decide(PolicyInput(...))
        
        decisions.append({
            "iteration": iteration,
            "decision": policy_output.decision.value,
            "reason": policy_output.reason,
            "coverage": critic_output.coverage_score,
            "risk": critic_output.hallucination_risk,
        })
        
        if policy_output.decision == DecisionEnum.FINALIZE:
            break
    
    # Log decision history
    for d in decisions:
        logger.info(f"Iteration {d['iteration']}: {d['decision']} - {d['reason']}")
    
    return result, decisions
```

---

## Appendix

### Glossary

| Term | Definition |
|------|------------|
| **FINALIZE** | Accept current answer and return to user |
| **REWRITE** | Reformulate query and retry retrieval |
| **CONTINUE** | Retry with same query (different retrieval) |
| **Coverage** | Proportion of context used in answer |
| **Hallucination Risk** | Likelihood of unsupported content |

### Threshold Reference

| Metric | Threshold | For FINALIZE |
|--------|-----------|--------------|
| Coverage | ≥ 0.6 | Required |
| Hallucination Risk | ≤ 0.4 | Required |

### Decision Priority

| Priority | Condition | Decision |
|----------|-----------|----------|
| 1 | Quality OK | FINALIZE |
| 2 | Rewrites available | REWRITE |
| 3 | Iterations available | CONTINUE |
| 4 | Budget exhausted | FINALIZE |

### Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | Initial | Basic threshold-based policy |

### References

- Radiant Repository: https://github.com/dshipley71/radiant
- Related Files: `critic_basic_agent.py`, `orchestrator.py`, `core/schemas.py`

---

*Document generated for the Radiant RAG Pipeline project. For questions or contributions, please refer to the project repository.*
