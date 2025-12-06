from __future__ import annotations

from typing import List

from core.agents_interfaces import PostProcessorAgent
from core.agents_schemas import (
    PostprocessInput,
    PostprocessOutput,
    PostprocessMetadata,
)


class BasicPostProcessorAgent(PostProcessorAgent):
    """Formats final answer with critic + language notes."""

    role = "postprocess"

    @property
    def name(self) -> str:
        return "BasicPostProcessorAgent"

    def describe(self) -> str:
        return "Formats final answer with critic + language notes."

    def format(self, inp: PostprocessInput) -> PostprocessOutput:
        base = inp.answer.text or ""
        lines: List[str] = [base]

        langs = sorted({cs.lang for cs in inp.context_snippets if cs.lang}) if inp.context_snippets else []

        if inp.preferences.include_critic_note and inp.critic_feedback.notes:
            lines.append("\n---\nCritic notes:")
            for n in inp.critic_feedback.notes:
                lines.append(f"- {n}")

        if inp.preferences.include_language_notes and langs:
            lines.append("\n---\nLanguages observed: " + ", ".join(langs))

        final_text = "\n".join(lines)
        meta = PostprocessMetadata(
            critic_summary="; ".join(inp.critic_feedback.notes) if inp.critic_feedback.notes else "",
            languages=langs,
        )
        return PostprocessOutput(final_text=final_text, metadata=meta)
