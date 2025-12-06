from __future__ import annotations

from typing import List

from core.agents_interfaces import GuardrailAgent
from core.agents_schemas import GuardrailInput, GuardrailOutput


class BasicGuardrailAgent(GuardrailAgent):
    """Basic guardrails for Plan sanity (limits & normalization)."""

    role = "guardrail"

    @property
    def name(self) -> str:
        return "BasicGuardrailAgent"

    def describe(self) -> str:
        return "Basic guardrails for Plan sanity (limits & normalization)."

    def validate_plan(self, inp: GuardrailInput) -> GuardrailOutput:
        plan = inp.plan
        adjusted = plan.model_copy(deep=True)
        messages: List[str] = []
        status = "ok"

        if adjusted.top_k <= 0:
            adjusted.top_k = 5
            messages.append("top_k <= 0; set to 5.")
        if adjusted.top_k > 100:
            adjusted.top_k = 100
            messages.append("top_k > 100; capped to 100.")

        if adjusted.rerank_top_k <= 0:
            adjusted.rerank_top_k = adjusted.top_k
            messages.append("rerank_top_k <= 0; set to top_k.")
        if adjusted.rerank_top_k > 200:
            adjusted.rerank_top_k = 200
            messages.append("rerank_top_k > 200; capped to 200.")

        if adjusted.iterations.max_iters < 1:
            adjusted.iterations.max_iters = 1
            messages.append("max_iters < 1; set to 1.")

        if messages:
            status = "adjusted"

        return GuardrailOutput(
            status=status,
            plan=adjusted,
            messages=messages,
        )
