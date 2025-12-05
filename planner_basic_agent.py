from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

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

    Additionally, this planner scales iteration budgets and top_k using RouterProfile:
      - RouterProfile.complexity_hint: "low" | "medium" | "high"
      - RouterProfile.query_type: "comparison" | "list" | "explanation" | "lookup" | "other"
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
            "and RouterProfile, and wires scalar knobs into a Plan. "
            "Iteration budgets and top_k are lightly scaled using RouterProfile "
            "complexity and query_type."
        )

    # ---- PlannerAgent interface ----
    def plan(self, inp: PlannerInput) -> PlannerOutput:
        cfg = inp.global_config
        rp = inp.router_profile

        # --------------------------------------------------
        # 1) Retrieval mode selection (config.fast.yaml first)
        # --------------------------------------------------
        retrieval_mode = self._decide_retrieval_mode(cfg)

        # --------------------------------------------------
        # 2) Feature toggles (QE / PRF / Rerank)
        # --------------------------------------------------
        use_qe = bool(cfg.enable_qe and rp.use_qe)
        use_prf = bool(cfg.enable_prf and rp.use_prf)
        use_rerank = bool(cfg.enable_rerank and rp.use_rerank)

        # --------------------------------------------------
        # 3) Iteration budget (scaled by complexity)
        # --------------------------------------------------
        iterations = self._scale_iterations(
            base_max_iters=cfg.max_iters,
            base_max_rewrites=cfg.max_rewrites,
            complexity_hint=rp.complexity_hint,
            query_type=rp.query_type,
        )

        # --------------------------------------------------
        # 4) top_k / rerank_top_k (scaled by complexity & query_type)
        # --------------------------------------------------
        top_k, rerank_top_k = self._scale_top_k(
            base_top_k=cfg.top_k,
            base_rerank_top_k=cfg.rerank_top_k,
            complexity_hint=rp.complexity_hint,
            query_type=rp.query_type,
        )

        # --------------------------------------------------
        # 5) Assemble Plan
        # --------------------------------------------------
        plan = Plan(
            retrieval_mode=retrieval_mode,
            use_qe=use_qe,
            use_prf=use_prf,
            use_rerank=use_rerank,
            iterations=iterations,
            top_k=top_k,
            rerank_top_k=rerank_top_k,
            language=cfg.language,
            allow_online_tools=cfg.allow_online_tools,
            backend=inp.ctx.runtime.backend,
        )
        return PlannerOutput(plan=plan)

    # ---- Internal helpers ----

    def _decide_retrieval_mode(self, cfg) -> RetrievalModeEnum:
        """Determine retrieval mode from config.fast.yaml, with GlobalConfig fallback."""
        retrieval_mode = cfg.default_retrieval_mode  # fallback

        retr_cfg = (self._cfg_raw.get("retrieval") or {}) if self._cfg_raw else {}
        if "leaf_only" in retr_cfg:
            leaf_only = bool(retr_cfg.get("leaf_only"))
            if leaf_only:
                retrieval_mode = RetrievalModeEnum.LEAF_ONLY
            else:
                retrieval_mode = RetrievalModeEnum.DUAL_INDEX

        return retrieval_mode

    def _scale_iterations(
        self,
        base_max_iters: int,
        base_max_rewrites: int,
        complexity_hint: Optional[str],
        query_type: Optional[str],
    ) -> PlanIterations:
        """Scale iteration budget using router hints.

        Heuristic:
          - low complexity: halve iterations/rewrites (but keep ≥1/0)
          - medium: use base config
          - high: +1 iteration, +1 rewrite
          - comparison/list: allow one extra rewrite on medium/high

        All scaling starts from config.fast.yaml (agentic.planner.*), so
        defaults remain intact but can be modulated by RouterProfile.
        """
        # Defensive fallbacks
        if base_max_iters <= 0:
            base_max_iters = 1
        if base_max_rewrites < 0:
            base_max_rewrites = 0

        ch = (complexity_hint or "medium").lower()
        qt = (query_type or "other").lower()

        max_iters = base_max_iters
        max_rewrites = base_max_rewrites

        if ch == "low":
            max_iters = max(1, base_max_iters // 2)
            max_rewrites = max(0, base_max_rewrites // 2)
        elif ch == "medium":
            max_iters = base_max_iters
            max_rewrites = base_max_rewrites
        else:  # "high" or anything else -> treat as high
            max_iters = base_max_iters + 1
            max_rewrites = base_max_rewrites + 1

        # For comparison / list queries, allow slightly more rewriting on medium/high.
        if qt in ("comparison", "list") and ch in ("medium", "high"):
            max_rewrites = max_rewrites + 1

        return PlanIterations(
            max_iters=max_iters,
            max_rewrites=max_rewrites,
        )

    def _scale_top_k(
        self,
        base_top_k: int,
        base_rerank_top_k: int,
        complexity_hint: Optional[str],
        query_type: Optional[str],
    ) -> Tuple[int, int]:
        """Scale top_k and rerank_top_k based on complexity and query type.

        Heuristic (multiplicative scaling around config defaults):

          - Start with multiplier = 1.0
          - Complexity:
              * low    → ×0.7 (but ≥1)
              * medium → ×1.0
              * high   → ×1.3
          - Query type:
              * comparison/list → ×1.5 (we want broader candidate sets)
          - Ensure:
              * top_k ≥ 1
              * rerank_top_k ≥ 1
              * rerank_top_k ≤ top_k

        This keeps config.fast.yaml as the source of truth but adds a small,
        interpretable adjustment layer.
        """
        if base_top_k <= 0:
            base_top_k = 1
        if base_rerank_top_k <= 0:
            base_rerank_top_k = base_top_k

        ch = (complexity_hint or "medium").lower()
        qt = (query_type or "other").lower()

        mult = 1.0

        # Complexity scaling
        if ch == "low":
            mult *= 0.7
        elif ch == "medium":
            mult *= 1.0
        else:  # "high" or unknown
            mult *= 1.3

        # Query-type scaling
        if qt in ("comparison", "list"):
            mult *= 1.5

        # Apply multiplier and clamp
        scaled_top_k = max(1, int(round(base_top_k * mult)))
        scaled_rerank_top_k = max(1, int(round(base_rerank_top_k * mult)))

        # Rerank can't exceed top_k
        if scaled_rerank_top_k > scaled_top_k:
            scaled_rerank_top_k = scaled_top_k

        return scaled_top_k, scaled_rerank_top_k

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
