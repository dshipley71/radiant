from __future__ import annotations

from typing import List

from core.interfaces import RouterAgent
from core.schemas import RouterInput, RouterOutput, RouterProfile


class BasicRouterAgent(RouterAgent):
    """Heuristic router for query type and high-level retrieval toggles."""

    role = "router"

    @property
    def name(self) -> str:
        return "BasicRouterAgent"

    def describe(self) -> str:
        return "Heuristic router for query type and high-level retrieval toggles."

    def route(self, inp: RouterInput) -> RouterOutput:
        """
        Use RouterConfig + (truncated) history + simple heuristics to decide:
          - query_type
          - expected_answer_style
          - complexity_hint
          - QE / PRF / rerank toggles

        Notes:
          - RouterConfig.default_query_type is used as a fallback if we classify as "other".
          - RouterConfig.max_hist_turns is used to truncate history and detect follow-ups.
        """
        cfg = inp.config

        raw_q = inp.user_query.strip()
        q = raw_q.lower()

        # Use at most the last N turns of history (if configured).
        if cfg.max_hist_turns and cfg.max_hist_turns > 0:
            history = inp.history[-cfg.max_hist_turns :]
        else:
            history = inp.history

        # Basic classification.
        query_type = self._infer_query_type(q)
        if query_type == "other" and cfg.default_query_type:
            # Allow config to override the "other" bucket if desired.
            query_type = cfg.default_query_type

        is_followup = self._is_followup(history, q)

        expected_answer_style = self._infer_answer_style(q, is_followup=is_followup)
        complexity_hint = self._infer_complexity(q)

        use_qe, use_prf, use_rerank = self._decide_toggles(
            query_type=query_type,
            complexity_hint=complexity_hint,
            is_followup=is_followup,
        )

        profile = RouterProfile(
            query_type=query_type,
            use_qe=use_qe,
            use_prf=use_prf,
            use_rerank=use_rerank,
            expected_answer_style=expected_answer_style,
            complexity_hint=complexity_hint,
        )
        return RouterOutput(router_profile=profile)

    # -------------------------------------------------------------------------
    # Heuristics
    # -------------------------------------------------------------------------

    def _infer_query_type(self, q: str) -> str:
        """Classify query into a coarse type.

        Keeps the existing set of types for compatibility:
          - comparison | list | explanation | lookup | other
        """
        if " vs " in q or " versus " in q or "difference between" in q:
            return "comparison"
        if q.startswith("list ") or " list of " in q or "top " in q:
            return "list"
        if q.startswith("how ") or q.startswith("why ") or "explain" in q:
            return "explanation"
        if q.startswith("what ") or q.endswith("?"):
            return "lookup"
        return "other"

    def _infer_answer_style(self, q: str, is_followup: bool = False) -> str:
        """Infer answer style: short | paragraph | multi_section.

        - "overview"/"detailed"/"guide" → multi_section
        - Very short queries → short
        - Follow-ups are biased toward paragraph so they don't get ultra-short answers.
        """
        if "overview" in q or "detailed" in q or "guide" in q:
            return "multi_section"

        token_count = len(q.split())
        if token_count < 8:
            # Short, but if it's a follow-up like "and pricing?" we prefer a bit more detail.
            return "paragraph" if is_followup else "short"

        return "paragraph"

    def _infer_complexity(self, q: str) -> str:
        """Infer complexity: low | medium | high based on token length."""
        tokens: List[str] = q.split()
        n = len(tokens)
        if n < 8:
            return "low"
        if n < 20:
            return "medium"
        return "high"

    def _is_followup(self, history, q_lower: str) -> bool:
        """Very lightweight follow-up detection.

        Heuristic:
          - There is some history, AND
          - The new query is short (<= 5 tokens), AND
          - It doesn't clearly start a brand new "what/how/why" question.
        """
        if not history:
            return False

        tokens = q_lower.split()
        if len(tokens) > 5:
            return False

        if q_lower.startswith(("what ", "how ", "why ", "who ", "where ", "when ")):
            return False

        return True

    def _decide_toggles(
        self,
        query_type: str,
        complexity_hint: str,
        is_followup: bool,
    ) -> tuple[bool, bool, bool]:
        """
        Decide QE / PRF / rerank flags.

        Current heuristic (still simple but more intentional):

          - QE:
              * On for medium/high complexity.
              * Also on for short follow-ups, so refinements can benefit from extra recall.
          - PRF:
              * On for (comparison | list) queries at medium/high complexity.
              * Off otherwise (keeps behavior conservative).
          - Rerank:
              * Always on (consistent with previous behavior).
        """
        # Query Expansion
        use_qe = complexity_hint in ("medium", "high") or is_followup

        # Pseudo-Relevance Feedback – only for richer, comparison/list-style queries
        use_prf = query_type in ("comparison", "list") and complexity_hint != "low"

        # Cross-encoder re-ranking – keep this always on for now
        use_rerank = True

        return use_qe, use_prf, use_rerank
