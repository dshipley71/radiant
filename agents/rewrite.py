from __future__ import annotations

from typing import List, Dict

from core.interfaces import QueryRewriteAgent
from core.schemas import QueryRewriteInput, QueryRewriteOutput
from llm_router import LLMRouter


class LLMQueryRewriteAgent(QueryRewriteAgent):
    """LLM-backed QueryRewriteAgent that uses critic feedback to refine queries."""

    role = "rewrite"

    def __init__(self, config: dict):
        self.cfg = config or {}
        self.router = LLMRouter(self.cfg)

        rw_cfg = (
            self.cfg
            .get("retrieval", {})
            .get("query_rewrite", {})
        )
        self.temperature = rw_cfg.get("temperature", 0.2)
        self.max_new_tokens = rw_cfg.get("max_new_tokens", 128)

    @property
    def name(self) -> str:
        return "LLMQueryRewriteAgent"

    def describe(self) -> str:
        return "QueryRewriteAgent that uses LLMRouter and critic feedback to refine the query."

    def rewrite(self, inp: QueryRewriteInput) -> QueryRewriteOutput:
        cf = inp.critic_feedback

        system_prompt = (
            "You are a query rewriting assistant for a retrieval-augmented system.\n"
            "Your goal is to rewrite the user's query so that retrieval gets better coverage\n"
            "and reduces hallucination risk, based on the critic feedback.\n"
            "Preserve the original intent and constraints; do NOT introduce new facts.\n"
        )

        notes_parts: List[str] = []
        if cf.missing_topics:
            notes_parts.append("Missing topics: " + "; ".join(cf.missing_topics))
        if cf.ambiguities:
            notes_parts.append("Ambiguities: " + "; ".join(cf.ambiguities))
        if cf.notes:
            notes_parts.append("Critic notes: " + "; ".join(cf.notes))

        critic_summary = "\n".join(notes_parts) if notes_parts else "No specific notes."

        user_prompt = (
            f"Original query: {inp.original_query}\n"
            f"Current query: {inp.current_query}\n"
            f"Critic feedback summary:\n{critic_summary}\n\n"
            "Rewrite the query to better target the user's intent and address the issues above.\n"
            "Return ONLY the rewritten query, with no explanation or commentary."
        )

        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        rewritten = self.router.chat(
            messages,
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
        ).strip()

        return QueryRewriteOutput(
            rewritten_query=rewritten,
            notes=[critic_summary] if critic_summary else [],
        )
