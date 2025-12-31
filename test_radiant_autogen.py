#!/usr/bin/env python3
"""
Radiant RAG Interactive Example

A practical example demonstrating how to use the Radiant RAG system with AutoGen.
Supports interactive mode, single queries, and batch processing.

Usage:
    # Interactive mode - chat with your indexed documents
    python test_radiant_autogen.py

    # Single query mode
    python test_radiant_autogen.py -q "What are the key concepts in this document?"

    # Direct tool mode (bypasses AutoGen, useful for debugging)
    python test_radiant_autogen.py --direct -q "Explain the architecture"

    # Batch mode - process queries from a file
    python test_radiant_autogen.py --batch queries.txt

    # Show retrieved sources
    python test_radiant_autogen.py -q "What is RAG?" --show-sources

Environment Variables:
    OLLAMA_API_KEY    - API key for Ollama Cloud (or compatible endpoint)
    OLLAMA_API_BASE   - API base URL (default: https://ollama.com/v1)
    OLLAMA_MODEL      - Model to use (default: minimax-m2:cloud)
    RADIANT_CONFIG    - Path to config file (default: config.fast.yaml)

Examples:
    # Ask about specific topics in your corpus
    python test_radiant_autogen.py -q "How does semantic chunking work?"

    # Multi-turn conversation
    python test_radiant_autogen.py
    > What documents do you have about embeddings?
    > Can you explain the vector store architecture?
    > How do I configure hybrid search?
    > exit

    # Process a list of questions
    echo "What is RAG?" > questions.txt
    echo "How does reranking work?" >> questions.txt
    python test_radiant_autogen.py --batch questions.txt --output answers.txt
"""

import asyncio
import json
import os
import sys
from datetime import datetime
from typing import List, Optional, Dict, Any

# Ensure we can import from the radiant package
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Default configuration
DEFAULT_MODEL = "minimax-m2:cloud"
DEFAULT_API_BASE = "https://ollama.com/v1"
DEFAULT_CONFIG = "config.fast.yaml"


def print_header():
    """Print a nice header."""
    print("\n" + "=" * 60)
    print("  Radiant RAG - Interactive Document Q&A")
    print("=" * 60)


def print_sources(result: Dict[str, Any], max_sources: int = 5):
    """Print retrieved sources from a RAG result."""
    meta = result.get("meta", {})
    sources = meta.get("sources", [])
    
    if not sources:
        print("\n  [No sources retrieved]")
        return
    
    print(f"\n  Sources ({len(sources)} retrieved, showing top {min(len(sources), max_sources)}):")
    print("  " + "-" * 50)
    
    for i, src in enumerate(sources[:max_sources]):
        title = src.get("title") or src.get("filename") or "Unknown"
        page = src.get("page", "?")
        score = src.get("score", 0.0)
        print(f"  [{i+1}] {title} (page {page}, score: {score:.3f})")


def format_answer(result: Dict[str, Any], show_sources: bool = False) -> str:
    """Format the answer from a RAG result."""
    answer = result.get("answer_text", "")
    
    if not answer:
        return "[No answer generated]"
    
    output = answer
    
    if show_sources:
        meta = result.get("meta", {})
        sources = meta.get("sources", [])
        if sources:
            output += "\n\nSources:"
            for i, src in enumerate(sources[:3]):
                title = src.get("title") or src.get("filename") or "Unknown"
                page = src.get("page", "?")
                output += f"\n  [{i+1}] {title} (p.{page})"
    
    return output


# -----------------------------------------------------------------------------
# Direct Tool Mode (no AutoGen)
# -----------------------------------------------------------------------------

def run_direct_query(query: str, show_sources: bool = False) -> Dict[str, Any]:
    """Run a query directly using radiant_tool (bypasses AutoGen)."""
    from radiant_autogen_wrappers import radiant_tool
    
    result = radiant_tool(query=query)
    return result


def direct_mode(query: str, show_sources: bool = False):
    """Single query using direct tool access."""
    print(f"\nQuery: {query}")
    print("-" * 50)
    
    result = run_direct_query(query, show_sources)
    answer = result.get("answer_text", "[No answer]")
    
    print(f"\nAnswer:\n{answer}")
    
    if show_sources:
        print_sources(result)


def direct_interactive():
    """Interactive mode using direct tool access."""
    print_header()
    print("\nDirect mode (no AutoGen). Type 'exit' or 'quit' to stop.\n")
    
    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        
        if not query:
            continue
        if query.lower() in ("exit", "quit", "q"):
            print("Goodbye!")
            break
        
        result = run_direct_query(query)
        answer = result.get("answer_text", "[No answer]")
        print(f"\nAssistant: {answer}\n")


# -----------------------------------------------------------------------------
# AutoGen Mode
# -----------------------------------------------------------------------------

async def create_agent():
    """Create and configure the RadiantManagerAgent."""
    from autogen_ext.models.openai import OpenAIChatCompletionClient
    from radiant_manager_agent import RadiantManagerAgent
    
    # Get configuration from environment
    api_key = os.getenv("OLLAMA_API_KEY", "")
    api_base = os.getenv("OLLAMA_API_BASE", DEFAULT_API_BASE)
    model = os.getenv("OLLAMA_MODEL", DEFAULT_MODEL)
    
    if not api_key:
        print("Warning: OLLAMA_API_KEY not set. Set it or update DEFAULT_API_KEY.")
    
    # Create model client
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
    
    # Create agent
    agent = RadiantManagerAgent(
        model_client=model_client,
        name="radiant_assistant",
        model_client_stream=True,
    )
    
    return agent, model_client


async def run_single_query(query: str, show_sources: bool = False):
    """Run a single query through the AutoGen agent."""
    agent, client = await create_agent()
    
    try:
        print(f"\nQuery: {query}")
        print("-" * 50)
        
        result = await agent.run(task=query)
        
        if result.messages:
            final_msg = result.messages[-1]
            print(f"\nAnswer:\n{final_msg.content}")
        else:
            print("\n[No response received]")
            
    finally:
        await client.close()


async def run_interactive():
    """Run interactive conversation with the AutoGen agent."""
    print_header()
    print("\nType 'exit' or 'quit' to stop. 'clear' to reset conversation.\n")
    
    agent, client = await create_agent()
    
    try:
        while True:
            try:
                query = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break
            
            if not query:
                continue
            if query.lower() in ("exit", "quit", "q"):
                print("Goodbye!")
                break
            if query.lower() == "clear":
                print("[Conversation cleared]")
                continue
            if query.lower() == "help":
                print("\nCommands:")
                print("  exit, quit, q  - Exit the program")
                print("  clear          - Clear conversation history")
                print("  help           - Show this help message")
                print("\nJust type your question to query the RAG system.\n")
                continue
            
            try:
                result = await agent.run(task=query)
                
                if result.messages:
                    final_msg = result.messages[-1]
                    print(f"\nAssistant: {final_msg.content}\n")
                else:
                    print("\n[No response received]\n")
                    
            except Exception as e:
                print(f"\n[Error: {e}]\n")
                
    finally:
        await client.close()


async def run_batch(input_file: str, output_file: Optional[str] = None, show_sources: bool = False):
    """Process a batch of queries from a file."""
    
    # Read queries
    with open(input_file, "r") as f:
        queries = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    
    if not queries:
        print(f"No queries found in {input_file}")
        return
    
    print(f"\nProcessing {len(queries)} queries from {input_file}...")
    print("-" * 50)
    
    results = []
    
    for i, query in enumerate(queries, 1):
        print(f"\n[{i}/{len(queries)}] {query}")
        
        result = run_direct_query(query, show_sources)
        answer = result.get("answer_text", "[No answer]")
        
        results.append({
            "query": query,
            "answer": answer,
            "timestamp": datetime.now().isoformat(),
        })
        
        # Print abbreviated answer
        preview = answer[:200] + "..." if len(answer) > 200 else answer
        print(f"    → {preview}")
    
    # Save results
    if output_file:
        if output_file.endswith(".json"):
            with open(output_file, "w") as f:
                json.dump(results, f, indent=2)
        else:
            with open(output_file, "w") as f:
                for r in results:
                    f.write(f"Q: {r['query']}\n")
                    f.write(f"A: {r['answer']}\n")
                    f.write("-" * 50 + "\n")
        print(f"\nResults saved to {output_file}")
    
    print(f"\nCompleted {len(queries)} queries.")


# -----------------------------------------------------------------------------
# Main Entry Point
# -----------------------------------------------------------------------------

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Radiant RAG Interactive Example",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                              # Interactive mode with AutoGen
  %(prog)s -q "What is RAG?"            # Single query
  %(prog)s --direct                     # Interactive mode without AutoGen
  %(prog)s --direct -q "Explain X"      # Single query without AutoGen
  %(prog)s --batch queries.txt          # Process queries from file
  %(prog)s -q "Query" --show-sources    # Show retrieved sources
        """
    )
    
    parser.add_argument(
        "-q", "--query",
        type=str,
        help="Single query to run (omit for interactive mode)"
    )
    parser.add_argument(
        "--direct",
        action="store_true",
        help="Use direct tool access (bypasses AutoGen)"
    )
    parser.add_argument(
        "--batch",
        type=str,
        metavar="FILE",
        help="Process queries from a file (one per line)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        metavar="FILE",
        help="Output file for batch results (.json or .txt)"
    )
    parser.add_argument(
        "--show-sources",
        action="store_true",
        help="Show retrieved source documents"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG,
        help=f"Path to config file (default: {DEFAULT_CONFIG})"
    )
    
    args = parser.parse_args()
    
    # Set config path in environment for radiant to pick up
    if args.config:
        os.environ.setdefault("RADIANT_CONFIG", args.config)
    
    # Batch mode
    if args.batch:
        asyncio.run(run_batch(args.batch, args.output, args.show_sources))
        return
    
    # Direct mode
    if args.direct:
        if args.query:
            direct_mode(args.query, args.show_sources)
        else:
            direct_interactive()
        return
    
    # AutoGen mode
    if args.query:
        asyncio.run(run_single_query(args.query, args.show_sources))
    else:
        asyncio.run(run_interactive())


if __name__ == "__main__":
    main()
