from __future__ import annotations

from typing import List

from agents_interfaces import DecompositionAgent
from agents_schemas import (
    DecompositionInput,
    DecompositionOutput,
    Decomposition,
    Subquery,
    ComparisonPair,
)


class BasicDecompositionAgent(DecompositionAgent):
    """Heuristic decomposition for multi-part and comparison queries."""

    role = "decomposition"

    @property
    def name(self) -> str:
        return "BasicDecompositionAgent"

    def describe(self) -> str:
        return "Heuristic decomposition for multi-part and comparison queries."

    def decompose(self, inp: DecompositionInput) -> DecompositionOutput:
        q = inp.user_query.strip()

        comparison_pairs: List[ComparisonPair] = []
        subqueries: List[Subquery] = []

        lower = q.lower()
        if " vs " in lower:
            left, right = q.split(" vs ", 1)
            comparison_pairs.append(ComparisonPair(left=left.strip(), right=right.strip()))

        parts = [p.strip() for p in q.replace(" & ", " and ").split(" and ") if p.strip()]
        if len(parts) > 1:
            for i, p in enumerate(parts, start=1):
                subqueries.append(Subquery(id=f"sub-{i}", text=p))

        is_multi_part = bool(subqueries or comparison_pairs)

        dec = Decomposition(
            is_multi_part=is_multi_part,
            subqueries=subqueries,
            comparison_pairs=comparison_pairs,
        )
        return DecompositionOutput(decomposition=dec)
