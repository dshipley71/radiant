"""
test_radiant_autogen.py

Simple test script to run RadiantManagerAgent with AutoGen
and issue a test query.

Usage:
    # Set your OpenAI API key
    export OPENAI_API_KEY="sk-..."
    
    # Run the test
    python test_radiant_autogen.py
    
    # Or run direct test without AutoGen
    python test_radiant_autogen.py --direct
"""

from __future__ import annotations

import asyncio
import os
from typing import Any, Dict

from autogen_ext.models.openai import OpenAIChatCompletionClient

from radiant_manager_agent import RadiantManagerAgent
from radiant_autogen_wrappers import radiant_tool


async def run_test() -> None:
    # ------------------------------------------------------------------
    # Configure your model client here.
    #
    # This assumes:
    #   - OPENAI_API_KEY is set in your environment (or compatible key)
    #   - You want to use gpt-4o-mini (adjust as desired)
    # ------------------------------------------------------------------
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Please set OPENAI_API_KEY in your environment.")

    model_client = OpenAIChatCompletionClient(
        model="gpt-4o-mini",
        api_key=api_key,
    )

    try:
        agent = RadiantManagerAgent(
            model_client=model_client,
            name="radiant_manager",
            model_client_stream=True,
        )

        user_query = "What is hierarchical RAG?"

        print(f"=== User query ===\n{user_query}\n")

        # Run the conversation with AutoGen; this will call radiant_tool under the hood.
        # AssistantAgent.run(...) returns a TaskResult.
        result = await agent.run(task=user_query)

        # The last assistant message should contain the final reply.
        if result.messages:
            final_msg = result.messages[-1]
            print("=== RadiantManagerAgent reply ===")
            print(final_msg.content)
        else:
            print("No messages returned from RadiantManagerAgent.")

        # Optionally: call radiant_tool directly (bypassing AutoGen) for debugging
        print("\n=== Raw radiant_tool result (direct call) ===")
        tool_result: Dict[str, Any] = radiant_tool(query=user_query)
        print("answer_text:", tool_result.get("answer_text"))
        # If you want to inspect more, you can pretty-print meta or selected fields
        # import pprint
        # pprint.pprint(tool_result["meta"])

    finally:
        await model_client.close()


def run_direct_test() -> None:
    """
    Run a direct test of radiant_tool without AutoGen.
    
    This is useful for testing the Radiant pipeline in isolation.
    """
    print("=== Direct radiant_tool test (no AutoGen) ===\n")
    
    queries = [
        "What is hierarchical RAG?",
        "How does it differ from traditional RAG?",  # Follow-up with history
    ]
    
    history = []
    
    for i, query in enumerate(queries, start=1):
        print(f"--- Query {i}: {query} ---")
        
        result = radiant_tool(
            query=query,
            history=history if history else None,
        )
        
        answer = result.get("answer_text") or "(no answer)"
        print(f"Answer: {answer}\n")
        
        # Build history for next query
        history.append({"role": "user", "content": query})
        history.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--direct":
        # Run direct test without AutoGen
        run_direct_test()
    else:
        # Run full AutoGen test
        asyncio.run(run_test())


# =============================================================================
# USAGE EXAMPLES (for documentation / interactive use only)
# =============================================================================
#
# from radiant_autogen_wrappers import radiant_tool
#
# # Single query
# result = radiant_tool(query="What is MCP?")
# print(result["answer_text"])
#
# # Multi-turn with history
# history = [
#     {"role": "user", "content": "What is RAG?"},
#     {"role": "assistant", "content": "RAG stands for..."}
# ]
# result = radiant_tool(query="How does it work?", history=history)
