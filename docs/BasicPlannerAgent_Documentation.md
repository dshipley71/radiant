# BasicPlannerAgent Documentation

## Technical Reference for the Radiant RAG Pipeline Planner

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture Context](#architecture-context)
3. [Class Structure](#class-structure)
4. [Core Functionality](#core-functionality)
5. [Configuration System](#configuration-system)
6. [Scaling Algorithms](#scaling-algorithms)
7. [Data Flow](#data-flow)
8. [Testing Strategies](#testing-strategies)
9. [Recommendations and Improvements](#recommendations-and-improvements)
10. [Usage Examples](#usage-examples)

---

## Overview

### Purpose

The `BasicPlannerAgent` is a strategic planning component within the Radiant RAG pipeline. It transforms high-level configuration settings and router hints into a concrete execution plan that governs how the retrieval and generation pipeline operates for a given query.

### Key Responsibilities

- Determine retrieval mode (dual-index vs. leaf-only)
- Combine global configuration with router profile to set feature toggles (QE, PRF, Rerank)
- Scale iteration budgets based on query complexity
- Adjust top_k retrieval parameters
- Assemble a complete `Plan` object for downstream pipeline execution

### Design Philosophy

The planner acts as a **bridge between configuration and execution**, applying intelligent scaling heuristics based on the router's complexity assessment. It follows a "configuration-first" approach where explicit config values take precedence, with router hints providing dynamic adjustments.

---

## Architecture Context

### Position in the Radiant Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    RouterProfile (from Router)                  │
│  • query_type    • complexity_hint    • use_qe/prf/rerank      │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                      BasicPlannerAgent                          │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Inputs:                                                │   │
│  │  • GlobalConfig (from config.fast.yaml)                 │   │
│  │  • RouterProfile (from BasicRouterAgent)                │   │
│  │  • Runtime Context                                      │   │
│  ├─────────────────────────────────────────────────────────┤   │
│  │  Processing:                                            │   │
│  │  • Retrieval mode selection                             │   │
│  │  • Feature toggle AND-ing                               │   │
│  │  • Iteration budget scaling                             │   │
│  │  • top_k parameter adjustment                           │   │
│  ├─────────────────────────────────────────────────────────┤   │
│  │  Output:                                                │   │
│  │  • Plan object with all execution parameters            │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                        Plan Object                              │
│  retrieval_mode | use_qe | use_prf | use_rerank | iterations   │
│  top_k | rerank_top_k | language | allow_online_tools | backend│
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
              Downstream Agents (Retriever, Rewriter, Generator)
```

### Related Components

| Component | Relationship |
|-----------|--------------|
| `PlannerAgent` | Abstract base class (from `core.interfaces`) |
| `PlannerInput` | Input schema with global_config, router_profile, ctx |
| `PlannerOutput` | Output schema wrapping the Plan |
| `Plan` | Complete execution plan data structure |
| `PlanIterations` | Iteration budget sub-structure |
| `RetrievalModeEnum` | Enumeration for retrieval strategies |
| `GlobalConfig` | System-wide configuration object |
| `RouterProfile` | Query analysis results from router |

---

## Class Structure

### Inheritance

```python
class BasicPlannerAgent(PlannerAgent):
    """Planner that converts high-level config + router hints into a concrete Plan."""
```

### Class Attributes

| Attribute | Type | Value | Description |
|-----------|------|-------|-------------|
| `role` | `str` | `"planner"` | Identifies the agent's role in the pipeline |

### Instance Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `_cfg_raw` | `Dict[str, Any]` | Raw configuration loaded from YAML/JSON |

### Constructor

```python
def __init__(self, config_path: Optional[str] = None) -> None
```

**Parameters:**
- `config_path`: Optional path to configuration file. Falls back to:
  1. `AGENTIC_RAG_CONFIG` environment variable
  2. `config.fast.yaml` in the same directory as the module

### Properties

| Property | Return Type | Description |
|----------|-------------|-------------|
| `name` | `str` | Returns `"BasicPlannerAgent"` |

### Methods

| Method | Visibility | Purpose |
|--------|------------|---------|
| `describe()` | Public | Returns human-readable description |
| `plan(inp)` | Public | Main entry point - generates execution plan |
| `_decide_retrieval_mode(cfg)` | Private | Determines retrieval strategy |
| `_scale_iterations(...)` | Private | Scales iteration budgets by complexity |
| `_scale_top_k(...)` | Private | Validates and constrains top_k values |
| `_load_config(config_path)` | Static | Loads YAML/JSON configuration file |

---

## Core Functionality

### The `plan()` Method

This is the primary entry point that orchestrates all planning logic.

**Signature:**
```python
def plan(self, inp: PlannerInput) -> PlannerOutput
```

**Parameters:**
- `inp` (`PlannerInput`): Contains global_config, router_profile, and runtime context

**Returns:**
- `PlannerOutput`: Contains the complete `Plan` object

**Processing Steps:**

1. **Retrieval Mode Selection**
   - Check `config.fast.yaml` for `retrieval.leaf_only` setting
   - Fall back to `GlobalConfig.default_retrieval_mode` if not specified

2. **Feature Toggle Resolution**
   - AND global config enables with router profile recommendations
   - `use_qe = cfg.enable_qe AND rp.use_qe`
   - `use_prf = cfg.enable_prf AND rp.use_prf`
   - `use_rerank = cfg.enable_rerank AND rp.use_rerank`

3. **Iteration Budget Scaling**
   - Apply complexity-based scaling to `max_iters` and `max_rewrites`
   - Apply query-type bonuses for comparison/list queries

4. **top_k Parameter Validation**
   - Enforce minimum values and constraints
   - Ensure `rerank_top_k <= top_k`

5. **Plan Assembly**
   - Combine all decisions into a `Plan` object
   - Include runtime context (backend, language, online tools)

---

## Configuration System

### Configuration File: `config.fast.yaml`

The planner reads from a YAML configuration file with the following structure:

```yaml
# Retrieval configuration
retrieval:
  leaf_only: false  # true = LEAF_ONLY, false = DUAL_INDEX

# Agentic planner settings (read via GlobalConfig)
agentic:
  planner:
    max_iters: 3
    max_rewrites: 2
    top_k: 10
    rerank_top_k: 5

# Feature toggles (in GlobalConfig)
enable_qe: true
enable_prf: true
enable_rerank: true

# Other settings
language: "en"
allow_online_tools: false
```

### Configuration Resolution Order

```
1. Explicit config_path parameter
         │
         ▼
2. AGENTIC_RAG_CONFIG environment variable
         │
         ▼
3. config.fast.yaml in module directory
         │
         ▼
4. Empty dict {} (graceful fallback)
```

### Retrieval Mode Logic

| `retrieval.leaf_only` | Result |
|----------------------|--------|
| `true` | `RetrievalModeEnum.LEAF_ONLY` |
| `false` | `RetrievalModeEnum.DUAL_INDEX` |
| Not specified | `GlobalConfig.default_retrieval_mode` |

### Feature Toggle AND Logic

The planner implements a **two-level toggle system**:

```
Final Toggle = GlobalConfig.enable_X AND RouterProfile.use_X
```

| GlobalConfig | RouterProfile | Result |
|--------------|---------------|--------|
| `enable_qe=True` | `use_qe=True` | ✅ QE enabled |
| `enable_qe=True` | `use_qe=False` | ❌ QE disabled |
| `enable_qe=False` | `use_qe=True` | ❌ QE disabled |
| `enable_qe=False` | `use_qe=False` | ❌ QE disabled |

This allows:
- **Global disable**: Turn off features system-wide via config
- **Dynamic disable**: Router can disable features for specific query types

---

## Scaling Algorithms

### Iteration Budget Scaling (`_scale_iterations`)

The planner scales iteration budgets based on query complexity and type.

#### Complexity-Based Scaling

| Complexity | max_iters | max_rewrites |
|------------|-----------|--------------|
| `low` | `base // 2` (min 1) | `base // 2` (min 0) |
| `medium` | `base` | `base` |
| `high` | `base + 1` | `base + 1` |

#### Query Type Bonus

For `comparison` and `list` queries at `medium` or `high` complexity:
- `max_rewrites += 1`

**Rationale:** These query types often benefit from additional refinement passes to gather comprehensive information.

#### Scaling Matrix

| Complexity | Query Type | Base Iters | Base Rewrites | Final Iters | Final Rewrites |
|------------|------------|------------|---------------|-------------|----------------|
| low | any | 3 | 2 | 1 | 1 |
| medium | lookup | 3 | 2 | 3 | 2 |
| medium | comparison | 3 | 2 | 3 | 3 |
| high | explanation | 3 | 2 | 4 | 3 |
| high | list | 3 | 2 | 4 | 4 |

### top_k Validation (`_scale_top_k`)

The current implementation **disables dynamic scaling** and only enforces constraints:

```python
# Constraints applied:
top_k >= 1
rerank_top_k >= 1
rerank_top_k <= top_k
```

**Note:** The method signature includes `complexity_hint` and `query_type` parameters, suggesting dynamic scaling was previously implemented or planned for future use.

---

## Data Flow

### Input Schema: `PlannerInput`

```python
@dataclass
class PlannerInput:
    global_config: GlobalConfig    # System-wide configuration
    router_profile: RouterProfile  # Query analysis from router
    ctx: Context                   # Runtime context
```

### Output Schema: `PlannerOutput`

```python
@dataclass
class PlannerOutput:
    plan: Plan
```

### Plan Schema

```python
@dataclass
class Plan:
    retrieval_mode: RetrievalModeEnum  # DUAL_INDEX | LEAF_ONLY
    use_qe: bool                       # Query Expansion enabled
    use_prf: bool                      # Pseudo-Relevance Feedback enabled
    use_rerank: bool                   # Reranking enabled
    iterations: PlanIterations         # Iteration budgets
    top_k: int                         # Documents to retrieve
    rerank_top_k: int                  # Documents after reranking
    language: str                      # Language code
    allow_online_tools: bool           # External tool access
    backend: str                       # Runtime backend identifier
```

### PlanIterations Schema

```python
@dataclass
class PlanIterations:
    max_iters: int      # Maximum retrieval iterations
    max_rewrites: int   # Maximum query rewrites
```

---

## Testing Strategies

### Unit Tests

#### 1. Configuration Loading Tests

```python
import pytest
import tempfile
import os
from pathlib import Path
from planner_basic_agent import BasicPlannerAgent

class TestConfigLoading:
    
    def test_load_yaml_config(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
retrieval:
  leaf_only: true
""")
        planner = BasicPlannerAgent(config_path=str(config_file))
        assert planner._cfg_raw["retrieval"]["leaf_only"] == True
    
    def test_load_json_config(self, tmp_path):
        config_file = tmp_path / "config.json"
        config_file.write_text('{"retrieval": {"leaf_only": false}}')
        planner = BasicPlannerAgent(config_path=str(config_file))
        assert planner._cfg_raw["retrieval"]["leaf_only"] == False
    
    def test_missing_config_returns_empty_dict(self):
        planner = BasicPlannerAgent(config_path="/nonexistent/path.yaml")
        assert planner._cfg_raw == {}
    
    def test_env_variable_config_path(self, tmp_path, monkeypatch):
        config_file = tmp_path / "env_config.yaml"
        config_file.write_text("retrieval:\n  leaf_only: true")
        monkeypatch.setenv("AGENTIC_RAG_CONFIG", str(config_file))
        
        planner = BasicPlannerAgent(config_path=None)
        assert planner._cfg_raw.get("retrieval", {}).get("leaf_only") == True
    
    def test_malformed_yaml_returns_empty_dict(self, tmp_path):
        config_file = tmp_path / "bad.yaml"
        config_file.write_text("invalid: yaml: content: [")
        planner = BasicPlannerAgent(config_path=str(config_file))
        assert planner._cfg_raw == {}
```

#### 2. Retrieval Mode Tests

```python
from unittest.mock import Mock
from core.schemas import RetrievalModeEnum

class TestRetrievalModeSelection:
    
    @pytest.fixture
    def mock_global_config(self):
        cfg = Mock()
        cfg.default_retrieval_mode = RetrievalModeEnum.DUAL_INDEX
        return cfg
    
    def test_leaf_only_true(self, mock_global_config, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text("retrieval:\n  leaf_only: true")
        planner = BasicPlannerAgent(config_path=str(config_file))
        
        mode = planner._decide_retrieval_mode(mock_global_config)
        assert mode == RetrievalModeEnum.LEAF_ONLY
    
    def test_leaf_only_false(self, mock_global_config, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text("retrieval:\n  leaf_only: false")
        planner = BasicPlannerAgent(config_path=str(config_file))
        
        mode = planner._decide_retrieval_mode(mock_global_config)
        assert mode == RetrievalModeEnum.DUAL_INDEX
    
    def test_fallback_to_global_config(self, mock_global_config):
        planner = BasicPlannerAgent(config_path="/nonexistent.yaml")
        mode = planner._decide_retrieval_mode(mock_global_config)
        assert mode == RetrievalModeEnum.DUAL_INDEX
```

#### 3. Feature Toggle Tests

```python
from core.schemas import PlannerInput, RouterProfile

class TestFeatureToggles:
    
    @pytest.fixture
    def planner(self):
        return BasicPlannerAgent(config_path=None)
    
    @pytest.fixture
    def make_input(self):
        def _make(cfg_qe, cfg_prf, cfg_rerank, rp_qe, rp_prf, rp_rerank):
            global_config = Mock()
            global_config.enable_qe = cfg_qe
            global_config.enable_prf = cfg_prf
            global_config.enable_rerank = cfg_rerank
            global_config.default_retrieval_mode = RetrievalModeEnum.DUAL_INDEX
            global_config.max_iters = 3
            global_config.max_rewrites = 2
            global_config.top_k = 10
            global_config.rerank_top_k = 5
            global_config.language = "en"
            global_config.allow_online_tools = False
            
            router_profile = Mock()
            router_profile.use_qe = rp_qe
            router_profile.use_prf = rp_prf
            router_profile.use_rerank = rp_rerank
            router_profile.complexity_hint = "medium"
            router_profile.query_type = "lookup"
            
            ctx = Mock()
            ctx.runtime = Mock()
            ctx.runtime.backend = "default"
            
            return PlannerInput(
                global_config=global_config,
                router_profile=router_profile,
                ctx=ctx
            )
        return _make
    
    def test_all_enabled(self, planner, make_input):
        inp = make_input(True, True, True, True, True, True)
        output = planner.plan(inp)
        assert output.plan.use_qe == True
        assert output.plan.use_prf == True
        assert output.plan.use_rerank == True
    
    def test_global_disabled_router_enabled(self, planner, make_input):
        inp = make_input(False, False, False, True, True, True)
        output = planner.plan(inp)
        assert output.plan.use_qe == False
        assert output.plan.use_prf == False
        assert output.plan.use_rerank == False
    
    def test_global_enabled_router_disabled(self, planner, make_input):
        inp = make_input(True, True, True, False, False, False)
        output = planner.plan(inp)
        assert output.plan.use_qe == False
        assert output.plan.use_prf == False
        assert output.plan.use_rerank == False
```

#### 4. Iteration Scaling Tests

```python
class TestIterationScaling:
    
    @pytest.fixture
    def planner(self):
        return BasicPlannerAgent(config_path=None)
    
    def test_low_complexity_halves_iterations(self, planner):
        result = planner._scale_iterations(
            base_max_iters=4,
            base_max_rewrites=4,
            complexity_hint="low",
            query_type="lookup"
        )
        assert result.max_iters == 2
        assert result.max_rewrites == 2
    
    def test_low_complexity_minimum_one_iter(self, planner):
        result = planner._scale_iterations(
            base_max_iters=1,
            base_max_rewrites=1,
            complexity_hint="low",
            query_type="lookup"
        )
        assert result.max_iters == 1
        assert result.max_rewrites == 0
    
    def test_medium_complexity_unchanged(self, planner):
        result = planner._scale_iterations(
            base_max_iters=3,
            base_max_rewrites=2,
            complexity_hint="medium",
            query_type="lookup"
        )
        assert result.max_iters == 3
        assert result.max_rewrites == 2
    
    def test_high_complexity_increases(self, planner):
        result = planner._scale_iterations(
            base_max_iters=3,
            base_max_rewrites=2,
            complexity_hint="high",
            query_type="lookup"
        )
        assert result.max_iters == 4
        assert result.max_rewrites == 3
    
    def test_comparison_query_bonus(self, planner):
        result = planner._scale_iterations(
            base_max_iters=3,
            base_max_rewrites=2,
            complexity_hint="medium",
            query_type="comparison"
        )
        assert result.max_rewrites == 3  # +1 bonus
    
    def test_list_query_bonus_high_complexity(self, planner):
        result = planner._scale_iterations(
            base_max_iters=3,
            base_max_rewrites=2,
            complexity_hint="high",
            query_type="list"
        )
        assert result.max_iters == 4    # high complexity +1
        assert result.max_rewrites == 4  # high +1, list bonus +1
    
    def test_defensive_zero_base_iters(self, planner):
        result = planner._scale_iterations(
            base_max_iters=0,
            base_max_rewrites=-1,
            complexity_hint="medium",
            query_type="lookup"
        )
        assert result.max_iters >= 1
        assert result.max_rewrites >= 0
```

#### 5. top_k Validation Tests

```python
class TestTopKValidation:
    
    @pytest.fixture
    def planner(self):
        return BasicPlannerAgent(config_path=None)
    
    def test_valid_values_unchanged(self, planner):
        top_k, rerank_top_k = planner._scale_top_k(
            base_top_k=10,
            base_rerank_top_k=5,
            complexity_hint="medium",
            query_type="lookup"
        )
        assert top_k == 10
        assert rerank_top_k == 5
    
    def test_zero_top_k_becomes_one(self, planner):
        top_k, rerank_top_k = planner._scale_top_k(
            base_top_k=0,
            base_rerank_top_k=5,
            complexity_hint="medium",
            query_type="lookup"
        )
        assert top_k == 1
    
    def test_rerank_top_k_capped_at_top_k(self, planner):
        top_k, rerank_top_k = planner._scale_top_k(
            base_top_k=5,
            base_rerank_top_k=10,
            complexity_hint="medium",
            query_type="lookup"
        )
        assert rerank_top_k == 5
    
    def test_zero_rerank_top_k_uses_top_k(self, planner):
        top_k, rerank_top_k = planner._scale_top_k(
            base_top_k=10,
            base_rerank_top_k=0,
            complexity_hint="medium",
            query_type="lookup"
        )
        assert rerank_top_k == 10
```

#### 6. Integration Tests

```python
class TestPlanIntegration:
    
    def test_full_plan_generation(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
retrieval:
  leaf_only: false
""")
        planner = BasicPlannerAgent(config_path=str(config_file))
        
        global_config = Mock()
        global_config.enable_qe = True
        global_config.enable_prf = True
        global_config.enable_rerank = True
        global_config.default_retrieval_mode = RetrievalModeEnum.DUAL_INDEX
        global_config.max_iters = 3
        global_config.max_rewrites = 2
        global_config.top_k = 10
        global_config.rerank_top_k = 5
        global_config.language = "en"
        global_config.allow_online_tools = True
        
        router_profile = Mock()
        router_profile.use_qe = True
        router_profile.use_prf = False
        router_profile.use_rerank = True
        router_profile.complexity_hint = "high"
        router_profile.query_type = "comparison"
        
        ctx = Mock()
        ctx.runtime = Mock()
        ctx.runtime.backend = "haystack"
        
        inp = PlannerInput(
            global_config=global_config,
            router_profile=router_profile,
            ctx=ctx
        )
        
        output = planner.plan(inp)
        plan = output.plan
        
        assert plan.retrieval_mode == RetrievalModeEnum.DUAL_INDEX
        assert plan.use_qe == True
        assert plan.use_prf == False  # Router disabled
        assert plan.use_rerank == True
        assert plan.iterations.max_iters == 4  # high +1
        assert plan.iterations.max_rewrites == 4  # high +1, comparison +1
        assert plan.top_k == 10
        assert plan.rerank_top_k == 5
        assert plan.language == "en"
        assert plan.allow_online_tools == True
        assert plan.backend == "haystack"
```

### Test Commands

```bash
# Run all planner tests
pytest test_planner_basic_agent.py -v

# Run with coverage
pytest test_planner_basic_agent.py --cov=planner_basic_agent --cov-report=html

# Run specific test class
pytest test_planner_basic_agent.py::TestIterationScaling -v
```

---

## Recommendations and Improvements

### Critical Issues

#### 1. Silent Configuration Failures

**Problem:** Configuration loading silently returns an empty dict on any exception, making debugging difficult.

**Current:**
```python
except Exception:
    return {}
```

**Recommendation:** Add logging for configuration failures:

```python
import logging
logger = logging.getLogger(__name__)

@staticmethod
def _load_config(config_path: Optional[str]) -> Dict[str, Any]:
    # ... file loading logic ...
    except Exception as e:
        logger.warning(
            f"Failed to load config from {config_path}: {e}. Using defaults."
        )
        return {}
```

---

### High Priority Improvements

#### 2. Configuration Validation Schema

**Problem:** No validation of configuration structure or types.

**Recommendation:** Add Pydantic or dataclass validation:

```python
from pydantic import BaseModel, validator
from typing import Optional

class RetrievalConfig(BaseModel):
    leaf_only: bool = False

class PlannerRawConfig(BaseModel):
    retrieval: Optional[RetrievalConfig] = None
    
    @validator('retrieval', pre=True, always=True)
    def default_retrieval(cls, v):
        return v or RetrievalConfig()

def _load_config(config_path: Optional[str]) -> PlannerRawConfig:
    raw = _load_raw_yaml(config_path)
    return PlannerRawConfig(**raw)
```

#### 3. Re-enable Dynamic top_k Scaling

**Problem:** The `_scale_top_k` method has complexity/query_type parameters but doesn't use them.

**Recommendation:** Implement or remove unused parameters:

```python
def _scale_top_k(
    self,
    base_top_k: int,
    base_rerank_top_k: int,
    complexity_hint: Optional[str],
    query_type: Optional[str],
) -> Tuple[int, int]:
    """Scale top_k based on complexity and query type."""
    ch = (complexity_hint or "medium").lower()
    qt = (query_type or "other").lower()
    
    # Scale top_k by complexity
    if ch == "low":
        top_k = max(1, base_top_k // 2)
    elif ch == "high":
        top_k = base_top_k + 5
    else:
        top_k = base_top_k
    
    # Comparison/list queries benefit from more documents
    if qt in ("comparison", "list"):
        top_k = int(top_k * 1.5)
    
    rerank_top_k = min(base_rerank_top_k, top_k)
    return max(1, top_k), max(1, rerank_top_k)
```

#### 4. Logging and Observability

**Problem:** No visibility into planning decisions.

**Recommendation:** Add structured logging:

```python
import logging
from dataclasses import asdict

logger = logging.getLogger(__name__)

def plan(self, inp: PlannerInput) -> PlannerOutput:
    # ... planning logic ...
    
    logger.info(
        "plan_generated",
        extra={
            "retrieval_mode": plan.retrieval_mode.value,
            "toggles": {
                "qe": plan.use_qe,
                "prf": plan.use_prf,
                "rerank": plan.use_rerank
            },
            "iterations": asdict(plan.iterations),
            "top_k": plan.top_k,
            "complexity": rp.complexity_hint,
            "query_type": rp.query_type,
        }
    )
    return PlannerOutput(plan=plan)
```

---

### Medium Priority Improvements

#### 5. Configurable Scaling Factors

**Problem:** Scaling factors are hardcoded.

**Recommendation:** Make scaling configurable:

```yaml
# config.fast.yaml
scaling:
  complexity:
    low:
      iter_factor: 0.5
      rewrite_factor: 0.5
    high:
      iter_bonus: 1
      rewrite_bonus: 1
  query_type:
    comparison:
      rewrite_bonus: 1
    list:
      rewrite_bonus: 1
```

```python
def _scale_iterations(self, ...):
    scaling = self._cfg_raw.get("scaling", {})
    complexity_cfg = scaling.get("complexity", {}).get(ch, {})
    
    if "iter_factor" in complexity_cfg:
        max_iters = int(base_max_iters * complexity_cfg["iter_factor"])
    elif "iter_bonus" in complexity_cfg:
        max_iters = base_max_iters + complexity_cfg["iter_bonus"]
    # ...
```

#### 6. Plan Caching

**Problem:** Same inputs always recompute the plan.

**Recommendation:** Add deterministic caching:

```python
from functools import lru_cache

def _make_plan_key(self, inp: PlannerInput) -> tuple:
    return (
        inp.global_config.enable_qe,
        inp.global_config.enable_prf,
        inp.router_profile.complexity_hint,
        inp.router_profile.query_type,
        # ... other relevant fields
    )

@lru_cache(maxsize=128)
def _cached_plan(self, key: tuple) -> Plan:
    # ... planning logic
```

#### 7. Plan Explanation

**Problem:** No way to understand why a plan was generated.

**Recommendation:** Add explanation capability:

```python
@dataclass
class PlanExplanation:
    retrieval_mode_reason: str
    qe_reason: str
    prf_reason: str
    rerank_reason: str
    iteration_reason: str

def plan_with_explanation(self, inp: PlannerInput) -> Tuple[PlannerOutput, PlanExplanation]:
    # ... generate plan with reasons
    explanation = PlanExplanation(
        retrieval_mode_reason=f"Config leaf_only={leaf_only}",
        qe_reason=f"Global={cfg.enable_qe}, Router={rp.use_qe}",
        iteration_reason=f"Complexity={ch}, scaled {base_iters}->{max_iters}",
        # ...
    )
    return output, explanation
```

---

### Low Priority / Future Enhancements

#### 8. Multi-Strategy Planning

**Recommendation:** Support multiple retrieval strategies per query:

```python
@dataclass
class Plan:
    strategies: List[RetrievalStrategy]  # Try multiple approaches
    fallback_strategy: RetrievalStrategy
```

#### 9. Cost-Aware Planning

**Recommendation:** Factor in computational cost:

```python
def plan(self, inp: PlannerInput) -> PlannerOutput:
    estimated_cost = self._estimate_cost(inp)
    if estimated_cost > inp.global_config.max_cost_budget:
        return self._generate_budget_plan(inp)
    return self._generate_optimal_plan(inp)
```

#### 10. A/B Testing Support

**Recommendation:** Enable experimentation:

```python
def plan(self, inp: PlannerInput) -> PlannerOutput:
    experiment = inp.ctx.get_experiment("planner_v2")
    if experiment.is_treatment:
        return self._plan_v2(inp)
    return self._plan_v1(inp)
```

---

## Usage Examples

### Basic Usage

```python
from planner_basic_agent import BasicPlannerAgent
from core.schemas import PlannerInput, GlobalConfig, RouterProfile

# Initialize planner with config file
planner = BasicPlannerAgent(config_path="config.fast.yaml")

# Or use environment variable
import os
os.environ["AGENTIC_RAG_CONFIG"] = "/path/to/config.yaml"
planner = BasicPlannerAgent()

# Create input
inp = PlannerInput(
    global_config=global_config,
    router_profile=router_profile,
    ctx=runtime_context
)

# Generate plan
output = planner.plan(inp)
plan = output.plan

print(f"Retrieval Mode: {plan.retrieval_mode}")
print(f"Max Iterations: {plan.iterations.max_iters}")
print(f"Features: QE={plan.use_qe}, PRF={plan.use_prf}, Rerank={plan.use_rerank}")
```

### Pipeline Integration

```python
class RAGPipeline:
    def __init__(self, config_path: str):
        self.router = BasicRouterAgent()
        self.planner = BasicPlannerAgent(config_path=config_path)
        self.retriever = RetrieverAgent()
        self.generator = GeneratorAgent()
    
    def process(self, query: str, global_config: GlobalConfig, ctx: Context):
        # Step 1: Route the query
        router_input = RouterInput(
            user_query=query,
            config=router_config,
            history=[]
        )
        router_output = self.router.route(router_input)
        
        # Step 2: Generate plan
        planner_input = PlannerInput(
            global_config=global_config,
            router_profile=router_output.router_profile,
            ctx=ctx
        )
        planner_output = self.planner.plan(planner_input)
        plan = planner_output.plan
        
        # Step 3: Execute plan
        for iteration in range(plan.iterations.max_iters):
            documents = self.retriever.retrieve(
                query=query,
                top_k=plan.top_k,
                mode=plan.retrieval_mode
            )
            
            if plan.use_rerank:
                documents = self.reranker.rerank(
                    documents, 
                    top_k=plan.rerank_top_k
                )
            
            # Check if sufficient results
            if self._is_sufficient(documents):
                break
        
        # Step 4: Generate response
        return self.generator.generate(query, documents)
```

### Custom Configuration

```python
# Create a custom config file programmatically
import yaml

config = {
    "retrieval": {
        "leaf_only": True  # Use leaf-only retrieval
    },
    "scaling": {
        "complexity": {
            "high": {
                "iter_bonus": 2,
                "rewrite_bonus": 2
            }
        }
    }
}

with open("custom_config.yaml", "w") as f:
    yaml.dump(config, f)

planner = BasicPlannerAgent(config_path="custom_config.yaml")
```

---

## Appendix

### Glossary

| Term | Definition |
|------|------------|
| **DUAL_INDEX** | Retrieval mode using both hierarchical and leaf indices |
| **LEAF_ONLY** | Retrieval mode using only leaf-level document chunks |
| **GlobalConfig** | System-wide configuration parameters |
| **RouterProfile** | Query analysis results from the router agent |
| **Plan** | Complete execution plan for the RAG pipeline |
| **PlanIterations** | Iteration budget configuration |

### Configuration Reference

| Config Path | Type | Default | Description |
|-------------|------|---------|-------------|
| `retrieval.leaf_only` | bool | false | Use leaf-only retrieval mode |
| `enable_qe` | bool | true | Enable query expansion globally |
| `enable_prf` | bool | true | Enable pseudo-relevance feedback globally |
| `enable_rerank` | bool | true | Enable reranking globally |
| `max_iters` | int | 3 | Base maximum iterations |
| `max_rewrites` | int | 2 | Base maximum query rewrites |
| `top_k` | int | 10 | Base documents to retrieve |
| `rerank_top_k` | int | 5 | Base documents after reranking |

### Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | Initial | Basic planning with complexity scaling |

### References

- Radiant Repository: https://github.com/dshipley71/radiant
- Related Files: `config.fast.yaml`, `orchestrator.py`, `core/schemas.py`, `core/interfaces.py`

---

*Document generated for the Radiant RAG Pipeline project. For questions or contributions, please refer to the project repository.*
