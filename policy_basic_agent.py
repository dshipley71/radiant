from __future__ import annotations

from agents_interfaces import PolicyAgent
from agents_schemas import PolicyInput, PolicyOutput, DecisionEnum


class BasicPolicyAgent(PolicyAgent):
    """Heuristic policy for finalize / rewrite / continue decisions."""

    role = "policy"

    @property
    def name(self) -> str:
        return "BasicPolicyAgent"

    def describe(self) -> str:
        return "Heuristic policy for finalize / rewrite / continue decisions."

    def decide(self, inp: PolicyInput) -> PolicyOutput:
        cov = inp.critic_feedback.coverage_score
        risk = inp.critic_feedback.hallucination_risk

        max_rewrites = max(0, inp.plan.iterations.max_rewrites)
        max_iters = max(1, inp.plan.iterations.max_iters)

        if cov >= 0.6 and risk <= 0.4:
            return PolicyOutput(
                decision=DecisionEnum.FINALIZE,
                reason="Coverage sufficient and hallucination risk acceptable.",
                adjustments={},
            )

        if inp.iteration < max_rewrites:
            return PolicyOutput(
                decision=DecisionEnum.REWRITE,
                reason="Coverage low or hallucination risk high; attempting rewrite.",
                adjustments={},
            )

        if inp.iteration + 1 < max_iters:
            return PolicyOutput(
                decision=DecisionEnum.CONTINUE,
                reason="Continuing without rewrite (iteration budget remains).",
                adjustments={},
            )

        return PolicyOutput(
            decision=DecisionEnum.FINALIZE,
            reason="Iteration budget exhausted.",
            adjustments={},
        )
