# BasicTelemetryAgent Documentation

## Technical Reference for the Radiant RAG Pipeline Telemetry Collection

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture Context](#architecture-context)
3. [Class Structure](#class-structure)
4. [Core Functionality](#core-functionality)
5. [Telemetry Design](#telemetry-design)
6. [Data Flow](#data-flow)
7. [Testing Strategies](#testing-strategies)
8. [Recommendations and Improvements](#recommendations-and-improvements)
9. [Usage Examples](#usage-examples)

---

## Overview

### Purpose

The `BasicTelemetryAgent` is the event logging component within the Radiant RAG pipeline. It provides a minimal acknowledgment interface for telemetry events, designed to cooperate with the orchestrator's centralized event logging system.

### Key Responsibilities

- Acknowledge telemetry events from the orchestrator
- Return standardized telemetry output with status and trace ID
- Maintain API compatibility with the TelemetryAgent interface
- Avoid duplicate logging (orchestrator handles event storage)

### Design Philosophy

The agent follows a **passive acknowledgment** pattern where the actual event storage is handled by the orchestrator, and the agent simply confirms receipt. This prevents double-logging and keeps telemetry output clean for structured rendering by the reporting layer.

---

## Architecture Context

### Position in the Radiant Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                         Orchestrator                            │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  _log_telemetry_with_elapsed()                          │   │
│  │    └─ Creates TelemetryEvent                            │   │
│  │    └─ Appends to TELEMETRY_EVENTS global list          │   │
│  │    └─ Calls BasicTelemetryAgent.log_event()            │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                   BasicTelemetryAgent                           │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  log_event(event)                                       │   │
│  │    └─ Returns TelemetryOutput(status="logged")          │   │
│  │    └─ Does NOT store event (already done by orch)       │   │
│  │    └─ Does NOT print (reporting layer handles)          │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Reporting Layer                              │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Consumes TELEMETRY_EVENTS                              │   │
│  │    └─ Renders structured table                          │   │
│  │    └─ Generates metrics summary                         │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### Telemetry Flow

```
Pipeline Execution
       │
       ├─── Agent A runs ─────────────────┐
       │                                   │
       ├─── Agent B runs ─────────────────┤
       │                                   │
       ├─── Agent C runs ─────────────────┤
       │                                   ▼
       │                    ┌─────────────────────────┐
       │                    │  TELEMETRY_EVENTS       │
       │                    │  (Global Buffer)        │
       │                    │  ┌───────────────────┐  │
       │                    │  │ Event 1: Agent A  │  │
       │                    │  │ Event 2: Agent B  │  │
       │                    │  │ Event 3: Agent C  │  │
       │                    │  └───────────────────┘  │
       │                    └─────────────────────────┘
       │                                   │
       ▼                                   ▼
  Pipeline End            Reporting Layer Renders Table
```

### Related Components

| Component | Relationship |
|-----------|--------------|
| `TelemetryAgent` | Abstract base class (from `core.interfaces`) |
| `TelemetryEvent` | Event schema (from `core.schemas`) |
| `TelemetryOutput` | Output schema with status and trace |
| `TELEMETRY_EVENTS` | Global event buffer (in orchestrator) |
| `orchestrator._log_telemetry_with_elapsed` | Event creation and storage |

---

## Class Structure

### Inheritance

```python
class BasicTelemetryAgent(TelemetryAgent):
    """Telemetry agent that cooperates with orchestrator telemetry logging."""
```

### Class Attributes

| Attribute | Type | Value | Description |
|-----------|------|-------|-------------|
| `role` | `str` | `"telemetry"` | Agent role identifier |

### Instance Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `_sink` | `Optional[List[TelemetryEvent]]` | Optional experimental sink (unused in main path) |

### Constructor

```python
def __init__(self, events_sink: Optional[List[TelemetryEvent]] = None) -> None
```

**Parameters:**
- `events_sink`: Optional list for experimental event collection (API compatibility)

**Note:** The sink is kept for API compatibility but is NOT used for the main reporting path. The orchestrator handles event storage directly.

### Methods

| Method | Visibility | Purpose |
|--------|------------|---------|
| `name` | Property | Returns agent name |
| `log_event(event)` | Public | Acknowledge telemetry event |

---

## Core Functionality

### The `log_event()` Method

Primary method that acknowledges telemetry events.

**Signature:**
```python
def log_event(self, event: TelemetryEvent) -> TelemetryOutput
```

**Parameters:**
- `event` (`TelemetryEvent`): Event to acknowledge

**Returns:**
- `TelemetryOutput`: Acknowledgment with status and trace ID

**Behavior:**
1. Does NOT append to TELEMETRY_EVENTS (already done by orchestrator)
2. Does NOT print to stdout (reporting layer handles rendering)
3. Returns acknowledgment with:
   - `status`: "logged"
   - `trace_id`: From event context's request_id
   - `sink`: "memory"

### Why No Storage or Printing?

```
┌─────────────────────────────────────────────────────────────────┐
│                    Design Decision                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Problem: Double-logging and noisy output                       │
│                                                                 │
│  If agent stored events:                                        │
│    ┌──────────────┐    ┌──────────────┐                        │
│    │ Orchestrator │───▶│ TELEMETRY_   │  Event stored once     │
│    │              │    │ EVENTS       │                        │
│    └──────────────┘    └──────────────┘                        │
│           │                   ▲                                 │
│           ▼                   │                                 │
│    ┌──────────────┐          │                                 │
│    │ Telemetry    │──────────┘  Event stored AGAIN = DUPLICATE │
│    │ Agent        │                                            │
│    └──────────────┘                                            │
│                                                                 │
│  Solution: Orchestrator stores, agent only acknowledges         │
│                                                                 │
│    ┌──────────────┐    ┌──────────────┐                        │
│    │ Orchestrator │───▶│ TELEMETRY_   │  Event stored once     │
│    │              │    │ EVENTS       │                        │
│    └──────┬───────┘    └──────────────┘                        │
│           │                                                     │
│           ▼                                                     │
│    ┌──────────────┐                                            │
│    │ Telemetry    │  Returns ack only, no storage              │
│    │ Agent        │                                            │
│    └──────────────┘                                            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Telemetry Design

### Event Structure

```python
@dataclass
class TelemetryEvent:
    agent: str              # Agent name (e.g., "BasicRouterAgent")
    event: str              # Event type (e.g., "router.classify")
    phase: str              # Pipeline phase (e.g., "QUERY")
    elapsed_ms: float       # Execution time in milliseconds
    ctx: TelemetryContext   # Request context with request_id
    model: Optional[str]    # Model used (for LLM calls)
    extra: Optional[Dict]   # Additional metadata
```

### Output Structure

```python
@dataclass
class TelemetryOutput:
    status: str       # "logged" | "error" | ...
    trace_id: str     # Request ID for correlation
    sink: str         # Storage destination ("memory" | "file" | ...)
```

### Event Types in Pipeline

| Agent | Event | Description |
|-------|-------|-------------|
| BasicRouterAgent | `router.classify` | Query classification |
| BasicPlannerAgent | `planner.plan` | Plan creation |
| HybridRetrievalAgent | `retriever.retrieve` | Document retrieval |
| LLMGeneratorAgent | `generator.output` | Answer generation |
| BasicCriticAgent | `critic.evaluate` | Answer evaluation |
| BasicPolicyAgent | `policy.decide` | Decision making |
| LLMQueryRewriteAgent | `rewriter.rewrite` | Query refinement |

### Telemetry Phases

| Phase | When |
|-------|------|
| `INIT` | Pipeline initialization |
| `QUERY` | Query processing |
| `ITERATION` | Loop iterations |
| `FINALIZE` | Final processing |

---

## Data Flow

### Input Schema: `TelemetryEvent`

```python
@dataclass
class TelemetryEvent:
    agent: str                    # Source agent
    event: str                    # Event type
    phase: str                    # Pipeline phase
    elapsed_ms: float             # Execution time
    ctx: TelemetryContext         # Request context
    model: Optional[str] = None   # Model info
    extra: Optional[Dict] = None  # Extra data
```

### Context Schema: `TelemetryContext`

```python
@dataclass
class TelemetryContext:
    request_id: str   # Unique request identifier
    # ... other context fields
```

### Output Schema: `TelemetryOutput`

```python
@dataclass
class TelemetryOutput:
    status: str      # Acknowledgment status
    trace_id: str    # Request trace ID
    sink: str        # Storage destination
```

### Example Event Flow

**Orchestrator creates event:**
```python
event = TelemetryEvent(
    agent="BasicRouterAgent",
    event="router.classify",
    phase="QUERY",
    elapsed_ms=15.3,
    ctx=TelemetryContext(request_id="req-12345"),
    model=None,
    extra={"query_type": "comparison"}
)

# Orchestrator appends to global buffer
TELEMETRY_EVENTS.append(event)

# Orchestrator calls agent
output = telemetry_agent.log_event(event)
```

**Agent returns acknowledgment:**
```python
TelemetryOutput(
    status="logged",
    trace_id="req-12345",
    sink="memory"
)
```

---

## Testing Strategies

### Unit Tests

#### 1. Basic Functionality Tests

```python
import pytest
from telemetry_basic_agent import BasicTelemetryAgent
from core.schemas import TelemetryEvent, TelemetryContext, TelemetryOutput

@pytest.fixture
def agent():
    return BasicTelemetryAgent()

@pytest.fixture
def sample_event():
    ctx = TelemetryContext(request_id="test-req-123")
    return TelemetryEvent(
        agent="TestAgent",
        event="test.event",
        phase="QUERY",
        elapsed_ms=10.5,
        ctx=ctx,
        model=None,
        extra=None
    )

class TestBasicFunctionality:
    
    def test_log_event_returns_output(self, agent, sample_event):
        output = agent.log_event(sample_event)
        
        assert isinstance(output, TelemetryOutput)
    
    def test_output_status_is_logged(self, agent, sample_event):
        output = agent.log_event(sample_event)
        
        assert output.status == "logged"
    
    def test_output_trace_id_from_event(self, agent, sample_event):
        output = agent.log_event(sample_event)
        
        assert output.trace_id == "test-req-123"
    
    def test_output_sink_is_memory(self, agent, sample_event):
        output = agent.log_event(sample_event)
        
        assert output.sink == "memory"
```

#### 2. Initialization Tests

```python
class TestInitialization:
    
    def test_init_without_sink(self):
        agent = BasicTelemetryAgent()
        
        assert agent._sink is None
    
    def test_init_with_sink(self):
        sink = []
        agent = BasicTelemetryAgent(events_sink=sink)
        
        assert agent._sink is sink
    
    def test_sink_not_used_in_log_event(self):
        sink = []
        agent = BasicTelemetryAgent(events_sink=sink)
        
        ctx = TelemetryContext(request_id="test")
        event = TelemetryEvent(
            agent="Test",
            event="test",
            phase="QUERY",
            elapsed_ms=1.0,
            ctx=ctx
        )
        
        agent.log_event(event)
        
        # Sink should NOT have event (agent doesn't store)
        assert len(sink) == 0
```

#### 3. No Side Effects Tests

```python
class TestNoSideEffects:
    
    def test_does_not_print(self, agent, sample_event, capsys):
        agent.log_event(sample_event)
        
        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err == ""
    
    def test_does_not_modify_event(self, agent, sample_event):
        original_agent = sample_event.agent
        original_event = sample_event.event
        
        agent.log_event(sample_event)
        
        assert sample_event.agent == original_agent
        assert sample_event.event == original_event
    
    def test_multiple_calls_independent(self, agent):
        ctx1 = TelemetryContext(request_id="req-1")
        ctx2 = TelemetryContext(request_id="req-2")
        
        event1 = TelemetryEvent(
            agent="Agent1", event="e1", phase="QUERY",
            elapsed_ms=1.0, ctx=ctx1
        )
        event2 = TelemetryEvent(
            agent="Agent2", event="e2", phase="QUERY",
            elapsed_ms=2.0, ctx=ctx2
        )
        
        output1 = agent.log_event(event1)
        output2 = agent.log_event(event2)
        
        assert output1.trace_id == "req-1"
        assert output2.trace_id == "req-2"
```

#### 4. Edge Case Tests

```python
class TestEdgeCases:
    
    def test_empty_request_id(self, agent):
        ctx = TelemetryContext(request_id="")
        event = TelemetryEvent(
            agent="Test", event="test", phase="QUERY",
            elapsed_ms=1.0, ctx=ctx
        )
        
        output = agent.log_event(event)
        
        assert output.trace_id == ""
    
    def test_long_request_id(self, agent):
        long_id = "x" * 1000
        ctx = TelemetryContext(request_id=long_id)
        event = TelemetryEvent(
            agent="Test", event="test", phase="QUERY",
            elapsed_ms=1.0, ctx=ctx
        )
        
        output = agent.log_event(event)
        
        assert output.trace_id == long_id
    
    def test_unicode_in_event(self, agent):
        ctx = TelemetryContext(request_id="テスト-123")
        event = TelemetryEvent(
            agent="日本語Agent",
            event="テスト.イベント",
            phase="QUERY",
            elapsed_ms=1.0,
            ctx=ctx
        )
        
        output = agent.log_event(event)
        
        assert output.trace_id == "テスト-123"
    
    def test_zero_elapsed_ms(self, agent):
        ctx = TelemetryContext(request_id="test")
        event = TelemetryEvent(
            agent="Test", event="test", phase="QUERY",
            elapsed_ms=0.0, ctx=ctx
        )
        
        output = agent.log_event(event)
        
        assert output.status == "logged"
    
    def test_negative_elapsed_ms(self, agent):
        ctx = TelemetryContext(request_id="test")
        event = TelemetryEvent(
            agent="Test", event="test", phase="QUERY",
            elapsed_ms=-1.0, ctx=ctx
        )
        
        output = agent.log_event(event)
        
        # Should still log (validation is caller's responsibility)
        assert output.status == "logged"
```

#### 5. Agent Interface Tests

```python
class TestAgentInterface:
    
    def test_name_property(self, agent):
        assert agent.name == "BasicTelemetryAgent"
    
    def test_role_attribute(self, agent):
        assert agent.role == "telemetry"
    
    def test_inherits_from_telemetry_agent(self, agent):
        from core.interfaces import TelemetryAgent
        assert isinstance(agent, TelemetryAgent)
```

#### 6. Concurrency Tests

```python
import threading

class TestConcurrency:
    
    def test_thread_safe_logging(self, agent):
        results = []
        errors = []
        
        def log_event(event_id):
            try:
                ctx = TelemetryContext(request_id=f"req-{event_id}")
                event = TelemetryEvent(
                    agent="Test", event="test", phase="QUERY",
                    elapsed_ms=1.0, ctx=ctx
                )
                output = agent.log_event(event)
                results.append(output.trace_id)
            except Exception as e:
                errors.append(e)
        
        threads = [threading.Thread(target=log_event, args=(i,)) for i in range(100)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(errors) == 0
        assert len(results) == 100
```

### Test Commands

```bash
# Run all telemetry tests
pytest test_telemetry_basic_agent.py -v

# Run with coverage
pytest test_telemetry_basic_agent.py --cov=telemetry_basic_agent --cov-report=html

# Run specific test class
pytest test_telemetry_basic_agent.py::TestNoSideEffects -v
```

---

## Recommendations and Improvements

### Critical Issues

#### 1. No Error Handling

**Problem:** If event.ctx or event.ctx.request_id is None, will raise AttributeError.

**Recommendation:** Add defensive checks:

```python
def log_event(self, event: TelemetryEvent) -> TelemetryOutput:
    try:
        trace_id = str(event.ctx.request_id) if event.ctx else "unknown"
    except AttributeError:
        trace_id = "unknown"
    
    return TelemetryOutput(
        status="logged",
        trace_id=trace_id,
        sink="memory"
    )
```

#### 2. No Validation of Event

**Problem:** Invalid events are accepted without validation.

**Recommendation:** Add validation:

```python
def log_event(self, event: TelemetryEvent) -> TelemetryOutput:
    # Validate required fields
    if not event.agent or not event.event:
        return TelemetryOutput(
            status="error",
            trace_id=self._safe_trace_id(event),
            sink="none"
        )
    
    return TelemetryOutput(status="logged", ...)
```

---

### High Priority Improvements

#### 3. Describe Method Missing

**Problem:** Agent doesn't implement `describe()` method.

**Recommendation:** Add describe method:

```python
def describe(self) -> str:
    return (
        "Telemetry agent that acknowledges events logged by the orchestrator. "
        "Events are stored in the global TELEMETRY_EVENTS buffer and rendered "
        "by the reporting layer."
    )
```

#### 4. Logging and Observability

**Problem:** No internal logging for debugging.

**Recommendation:** Add optional debug logging:

```python
import logging
logger = logging.getLogger(__name__)

def log_event(self, event: TelemetryEvent) -> TelemetryOutput:
    logger.debug(
        "telemetry_ack",
        extra={
            "agent": event.agent,
            "event": event.event,
            "elapsed_ms": event.elapsed_ms,
            "request_id": event.ctx.request_id if event.ctx else None,
        }
    )
    
    return TelemetryOutput(...)
```

#### 5. Optional Sink Support

**Problem:** The `_sink` parameter is unused.

**Recommendation:** Support experimental sink:

```python
def log_event(self, event: TelemetryEvent) -> TelemetryOutput:
    # Optionally write to experimental sink
    if self._sink is not None:
        self._sink.append(event)
    
    return TelemetryOutput(
        status="logged",
        trace_id=str(event.ctx.request_id),
        sink="memory" if self._sink is None else "experimental"
    )
```

---

### Medium Priority Improvements

#### 6. Metrics Aggregation

**Problem:** No aggregation capabilities.

**Recommendation:** Add basic metrics:

```python
class BasicTelemetryAgent(TelemetryAgent):
    def __init__(self, ...):
        self._event_count = 0
        self._total_elapsed_ms = 0.0
        self._agent_counts: Dict[str, int] = {}
    
    def log_event(self, event: TelemetryEvent) -> TelemetryOutput:
        self._event_count += 1
        self._total_elapsed_ms += event.elapsed_ms
        self._agent_counts[event.agent] = self._agent_counts.get(event.agent, 0) + 1
        
        return TelemetryOutput(...)
    
    def get_summary(self) -> Dict[str, Any]:
        return {
            "total_events": self._event_count,
            "total_elapsed_ms": self._total_elapsed_ms,
            "events_by_agent": self._agent_counts,
        }
```

#### 7. Event Filtering

**Problem:** All events treated equally.

**Recommendation:** Add filtering support:

```python
class BasicTelemetryAgent(TelemetryAgent):
    def __init__(self, event_filter: Optional[Callable[[TelemetryEvent], bool]] = None):
        self._filter = event_filter
    
    def log_event(self, event: TelemetryEvent) -> TelemetryOutput:
        if self._filter and not self._filter(event):
            return TelemetryOutput(status="filtered", ...)
        
        return TelemetryOutput(status="logged", ...)
```

#### 8. Async Support

**Problem:** Synchronous only.

**Recommendation:** Add async method:

```python
async def log_event_async(self, event: TelemetryEvent) -> TelemetryOutput:
    # For async pipelines
    return TelemetryOutput(...)
```

---

### Low Priority / Future Enhancements

#### 9. Multiple Sinks Support

**Recommendation:** Support multiple output destinations:

```python
class BasicTelemetryAgent(TelemetryAgent):
    def __init__(self, sinks: List[TelemetrySink] = None):
        self._sinks = sinks or []
    
    def log_event(self, event: TelemetryEvent) -> TelemetryOutput:
        sink_names = []
        for sink in self._sinks:
            sink.write(event)
            sink_names.append(sink.name)
        
        return TelemetryOutput(
            status="logged",
            trace_id=...,
            sink=",".join(sink_names) or "memory"
        )
```

#### 10. Export Capabilities

**Recommendation:** Add export methods:

```python
def export_to_json(self, path: str) -> None:
    """Export collected events to JSON file."""
    pass

def export_to_csv(self, path: str) -> None:
    """Export collected events to CSV file."""
    pass
```

#### 11. OpenTelemetry Integration

**Recommendation:** Support OTEL format:

```python
def to_otel_span(self, event: TelemetryEvent):
    """Convert event to OpenTelemetry span."""
    pass
```

---

## Usage Examples

### Basic Usage

```python
from telemetry_basic_agent import BasicTelemetryAgent
from core.schemas import TelemetryEvent, TelemetryContext

# Initialize agent
agent = BasicTelemetryAgent()

# Create event
ctx = TelemetryContext(request_id="req-12345")
event = TelemetryEvent(
    agent="BasicRouterAgent",
    event="router.classify",
    phase="QUERY",
    elapsed_ms=15.3,
    ctx=ctx,
    model=None,
    extra={"query_type": "comparison"}
)

# Log event (acknowledge)
output = agent.log_event(event)

print(f"Status: {output.status}")      # "logged"
print(f"Trace ID: {output.trace_id}")  # "req-12345"
print(f"Sink: {output.sink}")          # "memory"
```

### Orchestrator Integration

```python
# In orchestrator.py

TELEMETRY_EVENTS: List[TelemetryEvent] = []

def _log_telemetry_with_elapsed(
    agent_name: str,
    event_type: str,
    phase: str,
    elapsed_ms: float,
    ctx: TelemetryContext,
    telemetry_agent: BasicTelemetryAgent,
    **extra
) -> None:
    """Log telemetry event with elapsed time."""
    
    # Create event
    event = TelemetryEvent(
        agent=agent_name,
        event=event_type,
        phase=phase,
        elapsed_ms=elapsed_ms,
        ctx=ctx,
        extra=extra or None
    )
    
    # Append to global buffer (primary storage)
    TELEMETRY_EVENTS.append(event)
    
    # Acknowledge via agent
    output = telemetry_agent.log_event(event)
    
    # Optional: log acknowledgment
    logger.debug(f"Telemetry logged: {output.trace_id}")
```

### With Experimental Sink

```python
# Collect events separately for analysis
experimental_events = []
agent = BasicTelemetryAgent(events_sink=experimental_events)

# Log events...
for event in events:
    agent.log_event(event)

# Note: With current implementation, sink is NOT populated
# This is a placeholder for future enhancement
```

### Reporting Layer Integration

```python
def render_telemetry_table(events: List[TelemetryEvent]) -> str:
    """Render telemetry events as formatted table."""
    
    headers = ["Agent", "Event", "Phase", "Elapsed (ms)"]
    rows = [
        [e.agent, e.event, e.phase, f"{e.elapsed_ms:.2f}"]
        for e in events
    ]
    
    # Format as table...
    return formatted_table

# In reporting/smoke test:
from orchestrator import TELEMETRY_EVENTS

table = render_telemetry_table(TELEMETRY_EVENTS)
print(table)

# Output:
# ┌─────────────────────┬──────────────────┬───────┬─────────────┐
# │ Agent               │ Event            │ Phase │ Elapsed (ms)│
# ├─────────────────────┼──────────────────┼───────┼─────────────┤
# │ BasicRouterAgent    │ router.classify  │ QUERY │ 15.30       │
# │ BasicPlannerAgent   │ planner.plan     │ QUERY │ 2.10        │
# │ HybridRetrievalAgent│ retriever.retrieve│QUERY │ 245.80      │
# └─────────────────────┴──────────────────┴───────┴─────────────┘
```

---

## Appendix

### Glossary

| Term | Definition |
|------|------------|
| **Telemetry** | Collection of execution metrics and events |
| **Trace ID** | Unique identifier for request correlation |
| **Sink** | Destination for telemetry data |
| **Acknowledgment** | Confirmation of event receipt |

### Event Schema Reference

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `agent` | str | Yes | Source agent name |
| `event` | str | Yes | Event type identifier |
| `phase` | str | Yes | Pipeline phase |
| `elapsed_ms` | float | Yes | Execution time |
| `ctx` | TelemetryContext | Yes | Request context |
| `model` | str | No | Model used (LLM calls) |
| `extra` | Dict | No | Additional metadata |

### Output Schema Reference

| Field | Type | Value | Description |
|-------|------|-------|-------------|
| `status` | str | `"logged"` | Acknowledgment status |
| `trace_id` | str | From ctx | Request correlation ID |
| `sink` | str | `"memory"` | Storage destination |

### Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | Initial | Basic acknowledgment-only telemetry |

### References

- Radiant Repository: https://github.com/dshipley71/radiant
- Related Files: `orchestrator.py`, `core/schemas.py`, reporting layer

---

*Document generated for the Radiant RAG Pipeline project. For questions or contributions, please refer to the project repository.*
