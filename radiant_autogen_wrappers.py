"""
radiant_autogen_wrappers.py

Thin integration layer between Radiant and AutoGen.

This module:
  - Initializes Radiant's AgentRegistry and config
  - Exposes a single high-level function `radiant_tool` suitable for use as an
    AutoGen tool (FunctionTool) on an AssistantAgent.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Auto-detect and add Radiant package to sys.path
# ---------------------------------------------------------------------------

def _setup_radiant_path() -> None:
    """
    Find the Radiant package directory (containing 'core/' and 'agents/') 
    and add it to sys.path if not already present.
    
    Search order:
      1. Same directory as this file
      2. ./radiant/ subdirectory
      3. ../radiant/ parent's subdirectory
      4. RADIANT_PATH environment variable
    """
    def is_radiant_dir(p: Path) -> bool:
        """Check if path contains core/ and agents/ subdirectories."""
        return (p / "core").is_dir() and (p / "agents").is_dir()
    
    # Already importable?
    try:
        import core.orchestrator
        return  # Already in path
    except ImportError:
        pass
    
    candidates = []
    
    # 1. Same directory as this file
    this_dir = Path(__file__).parent.resolve()
    candidates.append(this_dir)
    
    # 2. ./radiant/ subdirectory of this file's directory
    candidates.append(this_dir / "radiant")
    
    # 3. Common Colab/notebook patterns
    candidates.append(Path("/content/radiant"))
    candidates.append(Path.cwd() / "radiant")
    candidates.append(Path.cwd())
    
    # 4. Environment variable
    env_path = os.getenv("RADIANT_PATH")
    if env_path:
        candidates.insert(0, Path(env_path))
    
    for candidate in candidates:
        if candidate.is_dir() and is_radiant_dir(candidate):
            path_str = str(candidate)
            if path_str not in sys.path:
                sys.path.insert(0, path_str)
            return
    
    # If we get here, we couldn't find it - let the import fail naturally
    # with a helpful error message
    raise ImportError(
        "Could not find Radiant package directory (containing 'core/' and 'agents/'). "
        "Either:\n"
        "  1. Place this file inside the radiant directory, or\n"
        "  2. Set RADIANT_PATH environment variable, or\n"
        "  3. Ensure ./radiant/ exists in current directory"
    )

_setup_radiant_path()

from pydantic import BaseModel

# Import Radiant orchestrator + schemas
from core.orchestrator import (
    register_default_agents,
    agentic_once_with_metadata,
    REGISTRY,
    CONFIG,
)


# ---------------------------------------------------------------------------
# Radiant initialization utilities
# ---------------------------------------------------------------------------

_RADIANT_INITIALIZED: bool = False


def _find_config_path() -> Optional[str]:
    """
    Attempt to locate config.fast.yaml in common locations.
    
    Search order:
      1. AGENTIC_RAG_CONFIG environment variable
      2. ./config.fast.yaml (current directory)
      3. ../config.fast.yaml (parent directory)
      4. Same directory as this module
      5. Inside the radiant package directory
    """
    # 1. Check environment variable first
    env_path = os.getenv("AGENTIC_RAG_CONFIG")
    if env_path and Path(env_path).is_file():
        return env_path
    
    # 2. Check current working directory
    cwd_config = Path("config.fast.yaml")
    if cwd_config.is_file():
        return str(cwd_config.resolve())
    
    # 3. Check parent directory (original default)
    parent_config = Path("../config.fast.yaml")
    if parent_config.is_file():
        return str(parent_config.resolve())
    
    # 4. Check same directory as this module
    module_dir = Path(__file__).parent
    module_config = module_dir / "config.fast.yaml"
    if module_config.is_file():
        return str(module_config.resolve())
    
    # 5. Check common radiant locations
    for candidate in [Path("/content/radiant"), Path.cwd() / "radiant"]:
        cfg = candidate / "config.fast.yaml"
        if cfg.is_file():
            return str(cfg.resolve())
    
    # Return None and let orchestrator handle the fallback
    return None


def _ensure_radiant_initialized(config_path: Optional[str] = None) -> None:
    """
    Ensure Radiant's agent registry and config are initialized.

    If config_path is None, attempts to auto-detect config.fast.yaml location.
    """
    global _RADIANT_INITIALIZED

    if _RADIANT_INITIALIZED:
        return

    # Auto-detect config path if not provided
    if config_path is None:
        config_path = _find_config_path()
    
    # Set env var so orchestrator uses the correct path
    if config_path is not None:
        os.environ["AGENTIC_RAG_CONFIG"] = config_path

    # This will populate REGISTRY, CONFIG, etc.
    register_default_agents(config_path=config_path)
    _RADIANT_INITIALIZED = True


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------

def _to_serializable(obj: Any) -> Any:
    """
    Convert Radiant's Pydantic models and other objects to JSON-serializable
    Python primitives (dicts, lists, scalars).

    This is friendly for AutoGen's tool-calling and logging.
    """
    # Pydantic v1/v2 BaseModel: has model_dump() in v2, dict() in v1
    if isinstance(obj, BaseModel):
        # Try Pydantic v2 API first
        if hasattr(obj, "model_dump"):
            return obj.model_dump()
        # Fallback to v1
        if hasattr(obj, "dict"):
            return obj.dict()

    # Generic containers
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_to_serializable(v) for v in obj]

    # Primitive or unknown types: return as-is and hope JSON can handle it
    return obj


# ---------------------------------------------------------------------------
# Main tool exposed to AutoGen
# ---------------------------------------------------------------------------

def radiant_tool(
    query: str,
    config_path: Optional[str] = None,
    history: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, Any]:
    """
    High-level Radiant pipeline tool for AutoGen.

    Parameters
    ----------
    query : str
        User query to run through the full Agentic RAG pipeline.
    config_path : Optional[str]
        Optional override for Radiant's config.fast.yaml path. If None, the
        default logic in core.orchestrator._load_config is used.
    history : Optional[List[Dict[str, str]]]
        Optional conversation history as list of {"role": "user"|"assistant", "content": "..."}
        Used for resolving pronouns and references in multi-turn conversations.

    Returns
    -------
    Dict[str, Any]
        JSON-serializable dict containing:
          - answer_text : final natural-language answer
          - meta        : structured metadata (router, plan, retrieval, citations, etc.)
    """
    _ensure_radiant_initialized(config_path=config_path)

    # Run the full Agentic RAG pipeline
    meta = agentic_once_with_metadata(query=query, history=history)

    # meta is a dict with mixed Pydantic objects, plain dicts, etc.
    # We provide both a friendly answer text and the full serialized meta.
    
    # Try to get answer from postprocess output first
    postprocess_obj = meta.get("postprocess")
    answer_text = None
    try:
        if postprocess_obj is not None and hasattr(postprocess_obj, "final_text"):
            answer_text = postprocess_obj.final_text
    except Exception:
        answer_text = None

    # Fallback to answer object
    if answer_text is None:
        answer_obj = meta.get("answer")
        if answer_obj is not None:
            if hasattr(answer_obj, "text"):
                answer_text = answer_obj.text
            elif isinstance(answer_obj, str):
                answer_text = answer_obj

    serialized_meta = _to_serializable(meta)

    return {
        "answer_text": answer_text,
        "meta": serialized_meta,
    }
