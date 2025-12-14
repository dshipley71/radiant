from __future__ import annotations

from typing import List, Dict

from core.interfaces import QEAgent
from core.schemas import QEInput, QEOutput
from core.llm_router import LLMRouter


class LLMQEAgent(QEAgent):
    """
    QEAgent that uses LLMRouter for query expansion.

    Behavior:
      - If inp.plan.use_qe is False, returns QEOutput(expanded_queries=[]).
      - Otherwise, calls the LLM to generate N paraphrases of the query.
      - N is taken from config.retrieval.query_expansion.num_variants if present,
        otherwise from inp.plan.max_qe_variants if defined, otherwise defaults to 5.

    The LLM backend (HF vs OpenAI-compatible) is selected by LLMRouter based on
    the same config dict used by the rest of the system.
    """

    role = "qe"

    def __init__(self, config: dict):
        self.cfg = config or {}
        self.router = LLMRouter(self.cfg)

        qe_cfg = (
            self.cfg
            .get("retrieval", {})
            .get("query_expansion", {})
        )

        self.default_num_variants = qe_cfg.get("num_variants", 5)
        self.temperature = qe_cfg.get("temperature", 0.2)
        self.max_new_tokens = qe_cfg.get("max_new_tokens", 64)

        # Read max_history from agentic.history config (number of Q&A pairs)
        hist_cfg = self.cfg.get("agentic", {}).get("history", {})
        self.max_history = hist_cfg.get("max_history", 10)

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "LLMQEAgent"

    def describe(self) -> str:
        return "Query Expansion agent using LLMRouter (HF or OpenAI-compatible)."

    # ------------------------------------------------------------------
    # QEAgent interface
    # ------------------------------------------------------------------

    def expand(self, inp: QEInput) -> QEOutput:
        """
        Expand a query into paraphrases for retrieval.

        - Honors plan.use_qe.
        - Tries to respect plan.max_qe_variants if present; otherwise falls back
          to config.retrieval.query_expansion.num_variants; otherwise 5.
        - Uses conversation history (if provided) to generate context-aware paraphrases.
        """

        if not getattr(inp.plan, "use_qe", False):
            return QEOutput(expanded_queries=[])

        plan_variants = getattr(inp.plan, "max_qe_variants", None)
        num_variants = plan_variants or self.default_num_variants or 5

        # Build history context if available
        history_context = ""
        if inp.history:
            max_messages = self.max_history * 2
            history_lines = []
            for msg in inp.history[-max_messages:]:
                if hasattr(msg, "role"):
                    role, content = msg.role, msg.content
                elif isinstance(msg, dict):
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                else:
                    continue
                
                # Truncate long messages for context
                if len(content) > 300:
                    content = content[:300] + "..."
                
                if role == "user":
                    history_lines.append(f"Q: {content}")
                elif role == "assistant":
                    history_lines.append(f"A: {content}")
            
            if history_lines:
                history_context = "\n".join(history_lines[-6:])  # Use last 3 Q&A pairs for prompt

        if history_context:
            system_prompt = (
                "You are a query expansion assistant for a retrieval system.\n"
                "Given a user query and conversation context, generate several diverse paraphrases that:\n"
                "1. Preserve the original meaning but use different wording\n"
                "2. Focus on complementary aspects of the query\n"
                "3. Incorporate relevant context from the conversation when it helps clarify the query\n"
                "Do NOT introduce new facts, constraints, or assumptions beyond what's in the query and context.\n"
                "Return ONLY the paraphrased queries, one per line, with no bullets or numbering."
            )

            user_prompt = (
                f"Conversation context:\n{history_context}\n\n"
                f"Current query to expand:\n{inp.query}\n\n"
                f"Number of paraphrases: {num_variants}\n"
            )
        else:
            system_prompt = (
                "You are a query expansion assistant for a retrieval system.\n"
                "Given a user query, generate several diverse paraphrases that preserve the\n"
                "original meaning but use different wording or focus on complementary aspects.\n"
                "Do NOT introduce new facts, constraints, or assumptions.\n"
                "Return ONLY the paraphrased queries, one per line, with no bullets or numbering."
            )

            user_prompt = (
                f"Original query:\n{inp.query}\n\n"
                f"Number of paraphrases: {num_variants}\n"
            )

        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        raw = self.router.chat(
            messages,
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
        )

        variants: List[str] = []
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            line = line.lstrip("-•*0123456789. ").strip()
            if line:
                variants.append(line)

        variants = variants[:num_variants]

        return QEOutput(expanded_queries=variants)
