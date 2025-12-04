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
        # events_sink is kept for API compatibility but is not used here,
        # because orchestrator._log_telemetry_with_elapsed already appends
        # events to the global TELEMETRY_EVENTS list.
        self._sink = events_sink or []

    # ----- Required abstract interface implementations -----------------

    @property
    def name(self) -> str:
        """
        Concrete implementation of the abstract 'name' property required by
        the TelemetryAgent / Agent base class.
        """
        return "BasicTelemetryAgent"

    # You may have an abstract 'description' property in your base class as well.
    # If so, uncomment and adjust this implementation:
    #
    # @property
    # def description(self) -> str:
    #     return "Basic telemetry sink that records events into in-memory buffers."

    # ----- TelemetryAgent API ------------------------------------------

    def log_event(self, event: TelemetryEvent) -> TelemetryOutput:
        """
        Log a telemetry event.

        The event has already been appended to the global TELEMETRY_EVENTS
        buffer by the orchestrator. We do not write anything to stdout here;
        the reporting layer will consume TELEMETRY_EVENTS and render a table.
        """
        # Optionally keep a local sink if you want, but it's not required
        # for the current reporting path.
        if self._sink is not None:
            self._sink.append(event)

        return TelemetryOutput(
            status="logged",
            trace_id=str(event.ctx.request_id),
            sink="memory",
        )
