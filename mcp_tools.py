from __future__ import annotations

from typing import Dict

from fastmcp import FastMCP, tool

from agents_schemas import (
    RouterInput,
    RouterOutput,
    DecompositionInput,
    DecompositionOutput,
    PlannerInput,
    PlannerOutput,
)
from agents_interfaces import RouterAgent, DecompositionAgent

# Global MCP app
mcp_app = FastMCP("agentic_rag")


# Simple in-memory registry for concrete implementations
AGENT_REGISTRY: Dict[str, object] = {}


def register_agent(agent: object) -> None:
    """Register a concrete agent implementation."""
    role = getattr(agent, "role", None)
    if not role:
        raise ValueError("Agent must define a 'role' attribute")
    AGENT_REGISTRY[role] = agent


def get_agent(role: str) -> object:
    agent = AGENT_REGISTRY.get(role)
    if agent is None:
        raise ValueError(f"No agent registered for role={role}")
    return agent


# ---------------------------------------------------------------------------
# MCP tools
# ---------------------------------------------------------------------------

@mcp_app.tool()
def router_route(input: RouterInput) -> RouterOutput:
    """MCP tool: router.route"""
    agent: RouterAgent = get_agent("router")  # type: ignore[assignment]
    return agent.route(input)


@mcp_app.tool()
def decomposition_decompose(input: DecompositionInput) -> DecompositionOutput:
    """MCP tool: decomposition.decompose"""
    agent: DecompositionAgent = get_agent("decomposition")  # type: ignore[assignment]
    return agent.decompose(input)


if __name__ == "__main__":
    # Entrypoint for running MCP server if desired
    mcp_app.run()
