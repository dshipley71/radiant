from __future__ import annotations

from typing import List, Dict, Any, Optional, Union

from core.interfaces import QueryRewriteAgent
from core.schemas import QueryRewriteInput, QueryRewriteOutput, Message
from core.llm_router import LLMRouter


# Patterns that suggest a query needs rewriting (contains pronouns/references)
_AMBIGUOUS_PATTERNS = [
    "they", "them", "their", "it", "its", "this", "that", "these", "those",
    "he", "she", "his", "her", "the author", "the artist", "the painting",
    "the image", "the document", "the scene", "the year", "the same",
]


def _needs_rewriting(query: str) -> bool:
    """Check if a query contains ambiguous references that need resolution."""
    query_lower = query.lower()
    # Check for pronouns and ambiguous references
    for pattern in _AMBIGUOUS_PATTERNS:
        if pattern in query_lower:
            return True
    # Also check for very short queries that might be follow-ups
    words = query.split()
    if len(words) <= 4:
        return True
    return False


class LLMQueryRewriteAgent(QueryRewriteAgent):
    """LLM-backed QueryRewriteAgent that uses critic feedback or conversation history to refine queries."""

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

        # Read max_history from agentic.history config (number of Q&A pairs)
        hist_cfg = self.cfg.get("agentic", {}).get("history", {})
        self.max_history = hist_cfg.get("max_history", 10)

    @property
    def name(self) -> str:
        return "LLMQueryRewriteAgent"

    def describe(self) -> str:
        return "QueryRewriteAgent that uses LLMRouter and critic feedback or conversation history to refine the query."

    def rewrite(self, inp: QueryRewriteInput) -> QueryRewriteOutput:
        """Rewrite query based on critic feedback (original method)."""
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

    def rewrite_with_history(
        self,
        query: str,
        history: List[Union[Dict[str, str], Message, Any]],
    ) -> QueryRewriteOutput:
        """
        Rewrite an ambiguous query using conversation history to make it self-contained.
        
        This resolves pronouns and references like "they", "it", "the author" by
        looking at previous Q&A pairs in the conversation.
        
        Args:
            query: The current query that may contain ambiguous references
            history: List of previous Q&A pairs (as dicts with 'role'/'content' or Message objects)
            
        Returns:
            QueryRewriteOutput with rewritten_query (or original if no rewriting needed)
        """
        # Check if rewriting is needed
        if not history or self.max_history <= 0 or not _needs_rewriting(query):
            return QueryRewriteOutput(
                rewritten_query=query,
                notes=["No rewriting needed - query is self-contained"],
            )
        
        # Build history context using max_history from config (Q&A pairs = messages * 2)
        history_text = ""
        max_messages = self.max_history * 2
        for msg in history[-max_messages:]:
            if hasattr(msg, "role"):
                role, content = msg.role, msg.content
            elif isinstance(msg, dict):
                role = msg.get("role", "user")
                content = msg.get("content", "")
            else:
                continue
            
            if role == "user":
                history_text += f"Q: {content}\n"
            elif role == "assistant":
                history_text += f"A: {content}\n"
        
        if not history_text.strip():
            return QueryRewriteOutput(
                rewritten_query=query,
                notes=["No valid history found"],
            )
        
        # Build prompts for history-based rewriting
        system_prompt = (
            "You are a query rewriting assistant. Your task is to rewrite ambiguous "
            "questions to be self-contained by resolving pronouns and references "
            "using the conversation history.\n"
            "Rules:\n"
            "- Replace pronouns (they, it, he, she, etc.) with specific entities\n"
            "- Replace vague references (the author, the scene, etc.) with specific referents\n"
            "- Keep the question concise but unambiguous\n"
            "- Preserve the original intent\n"
            "- Return ONLY the rewritten question, nothing else"
        )

        user_prompt = (
            f"Conversation history:\n{history_text}\n"
            f"Current question: {query}\n\n"
            "Rewrite the question to be self-contained. Only output the rewritten question."
        )

        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            rewritten = self.router.chat(
                messages,
                max_tokens=self.max_new_tokens,
                temperature=0.1,  # Lower temperature for more deterministic rewriting
            ).strip()
            
            # Basic validation
            if rewritten and len(rewritten) >= 5 and len(rewritten) < 500:
                # Remove any quotes or prefixes the LLM might have added
                rewritten = rewritten.strip('"\'')
                if rewritten.lower().startswith("rewritten:"):
                    rewritten = rewritten[10:].strip()
                if rewritten.lower().startswith("question:"):
                    rewritten = rewritten[9:].strip()
                
                return QueryRewriteOutput(
                    rewritten_query=rewritten,
                    notes=[f"Resolved references using conversation history"],
                )
        except Exception as e:
            # Log but don't fail
            pass
        
        # Fallback to original query
        return QueryRewriteOutput(
            rewritten_query=query,
            notes=["Rewriting failed - using original query"],
        )
