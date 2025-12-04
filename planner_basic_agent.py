from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from agents_interfaces import PlannerAgent
from agents_schemas import (
    PlannerInput,
    PlannerOutput,
    Plan,
    PlanIterations,
    RetrievalModeEnum,
)


class BasicPlannerAgent(PlannerAgent):
    """Planner that converts high-level config + router hints into a concrete Plan.

    Retrieval mode is controlled by config.fast.yaml:

        retrieval:
          leaf_only: false   # -> DUAL_INDEX
          # or
          leaf_only: true    # -> LEAF_ONLY

    If retrieval.leaf_only is missing, falls back to GlobalConfig.default_retrieval_mode.
    """

    role = "planner"

    def __init__(self, config_path: Optional[str] = None) -> None:
        self._cfg_raw: Dict[str, Any] = self._load_config(config_path)

    # ---- BaseAgent interface ----
    @property
    def name(self) -> str:
        return "BasicPlannerAgent"

    def describe(self) -> str:
        return (
            "Basic planner that sets retrieval_mode (dual-index vs leaf-only) "
            "from config.fast.yaml, toggles QE/PRF/Rerank from GlobalConfig "
            "and RouterProfile, and wires scalar knobs into a Plan."
        )

    # ---- PlannerAgent interface ----
    def plan(self, inp: PlannerInput) -> PlannerOutput:
        cfg = inp.global_config
        rp = inp.router_profile

        # --------------------------------------------------
        # 1) Retrieval mode selection (config.fast.yaml first)
        # --------------------------------------------------
        retrieval_mode = cfg.default_retrieval_mode  # fallback

        retr_cfg = (self._cfg_raw.get("retrieval") or {}) if self._cfg_raw else {}
        if "leaf_only" in retr_cfg:
            leaf_only = bool(retr_cfg.get("leaf_only"))
            if leaf_only:
                retrieval_mode = RetrievalModeEnum.LEAF_ONLY
            else:
                retrieval_mode = RetrievalModeEnum.DUAL_INDEX

        # --------------------------------------------------
        # 2) Feature toggles (QE / PRF / Rerank)
        # --------------------------------------------------
        use_qe = bool(cfg.enable_qe and rp.use_qe)
        use_prf = bool(cfg.enable_prf and rp.use_prf)
        use_rerank = bool(cfg.enable_rerank and rp.use_rerank)

        # --------------------------------------------------
        # 3) Iteration budget
        # --------------------------------------------------
        iterations = PlanIterations(
            max_iters=cfg.max_iters,
            max_rewrites=cfg.max_rewrites,
        )

        # --------------------------------------------------
        # 4) Assemble Plan
        # --------------------------------------------------
        plan = Plan(
            retrieval_mode=retrieval_mode,
            use_qe=use_qe,
            use_prf=use_prf,
            use_rerank=use_rerank,
            iterations=iterations,
            top_k=cfg.top_k,
            rerank_top_k=cfg.rerank_top_k,
            language=cfg.language,
            allow_online_tools=cfg.allow_online_tools,
            backend=inp.ctx.runtime.backend,
        )
        return PlannerOutput(plan=plan)

    # ---- Internal helpers ----
    @staticmethod
    def _load_config(config_path: Optional[str]) -> Dict[str, Any]:
        """Lightweight loader for config.fast.yaml (YAML or JSON)."""
        if config_path is None:
            # Let other parts of the system decide defaults; we only care when
            # an explicit path is provided or AGENTIC_RAG_CONFIG is set.
            import os
            env_path = os.getenv("AGENTIC_RAG_CONFIG")
            if env_path:
                config_path = env_path
            else:
                here = Path(__file__).resolve().parent
                config_path = str(here / "config.fast.yaml")

        cfg_file = Path(config_path)
        if not cfg_file.exists():
            return {}

        try:
            if cfg_file.suffix.lower() in {".yaml", ".yml"}:
                with cfg_file.open("r", encoding="utf-8") as f:
                    return yaml.safe_load(f) or {}
            else:
                with cfg_file.open("r", encoding="utf-8") as f:
                    return json.load(f) or {}
        except Exception:
            return {}
