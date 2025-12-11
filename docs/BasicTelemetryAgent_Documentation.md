# BasicTelemetryAgent Documentation

Technical reference for the Radiant RAG pipeline telemetry agent.

---

## Overview

The `BasicTelemetryAgent` acknowledges telemetry events. The orchestrator handles event construction and storage; this agent provides a passive acknowledgment interface.

**Module Location:** `agents/telemetry.py`

**Interface:** `TelemetryAgent` (from `core.interfaces`)

---

## Class Definition

```python
class BasicTelemetryAgent(TelemetryAgent):
    """Telemetry agent that cooperates with orchestrator._log_telemetry_with_elapsed."""
    
    role = "telemetry"
    
    def __init__(self, events_sink: Optional[List[TelemetryEvent]] = None) -> None:
        self._sink = events_sink
    
    @property
    def name(self) -> str:
        return "BasicTelemetryAgent"
    
    def log_event(self, event: TelemetryEvent) -> TelemetryOutput:
        ...
```

---

## Functionality

### Main Method: `log_event()`

**Input:** `TelemetryEvent`
- `ctx`: Request context
- `phase`: Pipeline phase
- `iteration`: Current iteration
- `event_type`: Event type string
- `agent`: Agent name
- `mode`: Runtime mode
- `backend`: LLM backend
- `timing`: Timing information
- `payload`: Additional data

**Output:** `TelemetryOutput`
- `status`: Always `"logged"`
- `trace_id`: Request ID as string
- `sink`: Always `"memory"`

---

## Implementation

```python
def log_event(self, event: TelemetryEvent) -> TelemetryOutput:
    """
    Acknowledge telemetry event. Events are stored by the orchestrator
    in the global TELEMETRY_EVENTS buffer.
    """
    return TelemetryOutput(
        status="logged",
        trace_id=str(event.ctx.request_id),
        sink="memory",
    )
```

---

## Design Notes

The orchestrator is responsible for:
- Constructing `TelemetryEvent` objects
- Appending events to the global `TELEMETRY_EVENTS` list
- Timing measurements

This agent:
- Does NOT print to stdout (reporting layer renders structured tables)
- Does NOT append events (avoids double-logging)
- Provides acknowledgment for interface compliance

---

## Related Documentation

- [CoreInterfaces_Documentation.md](CoreInterfaces_Documentation.md) - `TelemetryAgent` interface
- [CoreSchemas_Documentation.md](CoreSchemas_Documentation.md) - `TelemetryEvent`, `TelemetryOutput` schemas
- [Orchestrator_Documentation.md](Orchestrator_Documentation.md) - Telemetry collection
