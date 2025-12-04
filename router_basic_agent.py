from __future__ import annotations

from typing import List

from agents_interfaces import RouterAgent
from agents_schemas import RouterInput, RouterOutput, RouterProfile


class BasicRouterAgent(RouterAgent):
    """Heuristic router for query type and high-level retrieval toggles."""

    role = "router"

    @property
    def name(self) -> str:
        return "BasicRouterAgent"

    def describe(self) -> str:
        return "Heuristic router for query type and high-level retrieval toggles."

    def route(self, inp: RouterInput) -> RouterOutput:
        q = inp.user_query.strip().lower()

        query_type = self._infer_query_type(q)
        expected_answer_style = self._infer_answer_style(q)
        complexity_hint = self._infer_complexity(q)

        use_qe = complexity_hint in ("medium", "high")
        use_prf = False
        use_rerank = True

        profile = RouterProfile(
            query_type=query_type,
            use_qe=use_qe,
            use_prf=use_prf,
            use_rerank=use_rerank,
            expected_answer_style=expected_answer_style,
            complexity_hint=complexity_hint,
        )
        return RouterOutput(router_profile=profile)

    def _infer_query_type(self, q: str) -> str:
        if " vs " in q or " versus " in q or "difference between" in q:
            return "comparison"
        if q.startswith("list ") or " list of " in q or "top " in q:
            return "list"
        if q.startswith("how ") or q.startswith("why ") or "explain" in q:
            return "explanation"
        if q.startswith("what ") or q.endswith("?"):
            return "lookup"
        return "other"

    def _infer_answer_style(self, q: str) -> str:
        if "overview" in q or "detailed" in q or "guide" in q:
            return "multi_section"
        if len(q.split()) < 8:
            return "short"
        return "paragraph"

    def _infer_complexity(self, q: str) -> str:
        tokens: List[str] = q.split()
        n = len(tokens)
        if n < 8:
            return "low"
        if n < 20:
            return "medium"
        return "high"
