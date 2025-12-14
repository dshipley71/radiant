"""
radiant_manager_agent.py

Defines RadiantManagerAgent, an AutoGen AssistantAgent that wraps the Radiant
Agentic RAG pipeline via the `radiant_tool` function.

This is intentionally simple: one tool = one Radiant call.

You can later expand this into multi-step tools (router_tool, planner_tool, etc.)
if you want finer-grained AutoGen orchestration.
"""

from __future__ import annotations

from typing import Optional

from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient

from radiant_autogen_wrappers import radiant_tool


class RadiantManagerAgent(AssistantAgent):
    """
    Manager agent that uses a single Radiant tool to answer user queries.

    The LLM is instructed to:
      - Call `radiant_tool` once per user query
      - Use `answer_text` for the natural-language reply
      - Optionally include some lightweight debug info if helpful
    """

    def __init__(
        self,
        model_client: OpenAIChatCompletionClient,
        *,
        name: str = "radiant_manager",
        model_client_stream: bool = True,
    ) -> None:
        system_message = """
You are RadiantManagerAgent, a specialized agent that delegates question
answering to the Radiant Agentic RAG pipeline via the `radiant_tool` function.

- Always, for each user query, call the `radiant_tool(query, config_path?, history?)` tool.
- The tool returns a JSON object with:
    {
        "answer_text": <final answer string or null>,
        "meta": {...}  // detailed metadata: router, plan, retrieval, citations, etc.
    }

Behavior guidelines:
- If `answer_text` is non-null and non-empty, use it as the core of your reply.
- You MAY lightly rephrase or clean up the wording, but do not change the meaning.
- If `answer_text` is null or empty, inspect `meta` to see if:
    - there is an error or guardrail rejection you should explain to the user, or
    - there are context snippets you can summarize.
- Do NOT dump the entire `meta` object to the user. It is for internal debugging.
- You MAY include a small, human-friendly debug section (like "Sources" or a short summary of what was retrieved),
  but keep it concise and readable.
- If Radiant indicates the query is unsafe or cannot be answered, explain that clearly and politely.
- For follow-up questions, you can pass conversation history to help resolve pronouns and references.

Be concise and helpful in your final responses.
        """.strip()

        tools = [radiant_tool]

        super().__init__(
            name=name,
            model_client=model_client,
            tools=tools,
            system_message=system_message,
            model_client_stream=model_client_stream,
        )
