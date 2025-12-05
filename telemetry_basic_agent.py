from __future__ import annotations

from typing import List, Optional

from agents_interfaces import TelemetryAgent
from agents_schemas import TelemetryEvent, TelemetryOutput


class BasicTelemetryAgent(TelemetryAgent):
    """
    Telemetry agent that cooperates with orchestrator._log_telemetry_with_elapsed.

    The orchestrator is responsible for:
      * Constructing TelemetryEvent objects
      * Appending them to the global TELEMETRY_EVENTS list

    This agent's job is simply to acknowledge the event. We intentionally do
    NOT print to stdout here, because a structured Telemetry table is rendered
    by the smoke-test / reporting layer and the raw print output would be
    redundant and noisy in Colab / CLI runs.
    """

    role = "telemetry"

    def __init__(self, events_sink: Optional[List[TelemetryEvent]] = None) -> None:
        # Kept for API compatibility, but not used for the global buffer;
        # orchestrator._log_telemetry_with_elapsed already appends events.
        self._sink = events_sink

    # ----- Required abstract interface implementations -----------------

    @property
    def name(self) -> str:
        """
        Concrete implementation of the abstract 'name' property required by
        the TelemetryAgent / Agent base class.
        """
        return "BasicTelemetryAgent"

    # ----- TelemetryAgent API ------------------------------------------

    def log_event(self, event: TelemetryEvent) -> TelemetryOutput:
        """
        Log a telemetry event.

        The event has already been appended to the global TELEMETRY_EVENTS
        buffer by the orchestrator. We do not write anything to stdout here;
        the reporting layer will consume TELEMETRY_EVENTS and render a table.
        """
        # Do NOT append to TELEMETRY_EVENTS here; avoid double-logging.
        # You could optionally use self._sink as a separate experimental sink,
        # but it is unused for the main reporting path.
        return TelemetryOutput(
            status="logged",
            trace_id=str(event.ctx.request_id),
            sink="memory",
        )
