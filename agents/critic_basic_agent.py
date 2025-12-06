from __future__ import annotations

from typing import List

from core.agents_interfaces import CriticAgent
from core.agents_schemas import CriticInput, CriticOutput


class BasicCriticAgent(CriticAgent):
    """Heuristic critic for coverage and hallucination risk."""

    role = "critic"

    @property
    def name(self) -> str:
        return "BasicCriticAgent"

    def describe(self) -> str:
        return "Heuristic critic for coverage and hallucination risk."

    def evaluate(self, inp: CriticInput) -> CriticOutput:
        ctx = inp.context_snippets or []
        answer_text = inp.answer.text or ""

        num_snips = len(ctx)
        max_k = max(1, inp.plan.top_k)

        coverage = min(1.0, num_snips / max_k)
        halluc_risk = 1.0 - coverage

        notes: List[str] = []
        if not ctx:
            notes.append("No retrieval context available; answer may be hallucinated.")
        if len(answer_text.split()) < 10:
            notes.append("Answer is very short; consider elaborating if user needs detail.")
        if coverage < 0.3:
            notes.append("Low coverage of available context (few relevant snippets).")

        return CriticOutput(
            hallucination_risk=halluc_risk,
            coverage_score=coverage,
            missing_topics=[],
            ambiguities=[],
            unsupported_claims=[],
            notes=notes,
        )
