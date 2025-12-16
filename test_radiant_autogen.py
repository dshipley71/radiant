%%writefile test_radiant_autogen.py
"""
Simple test to run the RadiantManagerAgent.

Usage:
    # Run the test (uses Ollama Cloud minimax-m2 by default)
    python test_radiant_autogen.py

    # Or test radiant_tool directly without AutoGen
    python test_radiant_autogen.py --direct

    # Override with environment variables if needed:
    export OLLAMA_API_KEY="your-key"
    export OLLAMA_API_BASE="https://ollama.com/v1"
    export OLLAMA_MODEL="minimax-m2:cloud"
"""

import asyncio
import os
import sys
import rich

# Ensure we can import from the radiant package
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Ollama Cloud configuration (from config.fast.yaml)
DEFAULT_MODEL = "minimax-m2:cloud"
DEFAULT_API_BASE = "https://ollama.com/v1"
DEFAULT_API_KEY = "ENTER_YOUR_OLLAM_CLOUD_KEY"


async def test_radiant_manager():
    """Simple test of RadiantManagerAgent using Ollama Cloud."""

    from autogen_ext.models.openai import OpenAIChatCompletionClient
    from radiant_manager_agent import RadiantManagerAgent

    # Get configuration from environment or use defaults
    api_key = os.getenv("OLLAMA_API_KEY", DEFAULT_API_KEY)
    api_base = os.getenv("OLLAMA_API_BASE", DEFAULT_API_BASE)
    model = os.getenv("OLLAMA_MODEL", DEFAULT_MODEL)

    print(f"Using model: {model}")
    print(f"API base: {api_base}")
    print("-" * 50)

    # Create model client for Ollama Cloud (OpenAI-compatible)
    # Need to provide model_info since it's not a recognized OpenAI model
    model_client = OpenAIChatCompletionClient(
        model=model,
        api_key=api_key,
        base_url=api_base,
        model_info={
            "vision": False,
            "function_calling": True,
            "json_output": True,
            "structured_output": True,
            "family": "unknown",
        },
    )

    try:
        # Create the RadiantManagerAgent
        agent = RadiantManagerAgent(
            model_client=model_client,
            name="radiant_manager",
            model_client_stream=True,
        )

        # Test query
        query = "What is RAG?"
        print(f"Query: {query}")
        print("-" * 50)

        # Run the agent
        result = await agent.run(task=query)

        # Print the response
        if result.messages:
            final_msg = result.messages[-1]
            rich.print(f"Response:\n{final_msg.content}")
        else:
            print("No response received.")

    finally:
        await model_client.close()


def test_radiant_tool_direct():
    """Test radiant_tool directly without AutoGen."""

    from radiant_autogen_wrappers import radiant_tool

    query = "What is RAG?"
    print(f"Query: {query}")
    print("-" * 50)

    result = radiant_tool(query=query)

    answer = result.get("answer_text")
    if answer:
        print(f"Answer:\n{answer}")
    else:
        print("No answer returned.")
        print(f"Meta: {result.get('meta', {})}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test RadiantManagerAgent")
    parser.add_argument(
        "--direct",
        action="store_true",
        help="Test radiant_tool directly without AutoGen"
    )
    args = parser.parse_args()

    if args.direct:
        test_radiant_tool_direct()
    else:
        asyncio.run(test_radiant_manager())
