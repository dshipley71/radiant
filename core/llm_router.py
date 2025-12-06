from __future__ import annotations

import os
from typing import List, Dict, Any, Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
)

from openai import OpenAI


class LLMRouter:
    """
    Unified LLM router supporting:
      - Local HF models (use_local=True)
      - OpenAI-compatible APIs (use_local=False)

    Dynamically loaded based on config passed into the constructor.

    Public interface:

        response = router.chat(messages=[...], **overrides)
        response = router.generate(prompt="...", max_tokens=..., temperature=...)

    """

    # ------------------------------------------------------------------
    # Init
    # ------------------------------------------------------------------

    def __init__(self, config: dict):
        self.cfg = config or {}

        # Support two usage modes:
        #  1) Full config.fast.yaml dict with top-level "models" and "llm" keys.
        #  2) A bare llm-config dict (as used by LLMGeneratorAgent), which contains
        #     keys like model/api_base/api_key/temperature/max_tokens directly.
        if "models" in self.cfg or "llm" in self.cfg:
            models_cfg = self.cfg.get("models", {}) or {}
            llm_cfg = self.cfg.get("llm", {}) or {}
            # In this mode, default to the old behavior (local unless overridden).
            use_local_default = True
        else:
            models_cfg = {}
            llm_cfg = self.cfg
            # In bare-LLM config mode, default to remote/OpenAI-compatible.
            use_local_default = False

        # Routing flags
        self.use_local = bool(models_cfg.get("use_local", use_local_default))

        # HF model settings (used when use_local=True)
        self.local_model_id = models_cfg.get(
            "llm_model"
        )
        self.local_device = models_cfg.get(
            "llm_device",
            "cuda" if torch.cuda.is_available() else "cpu",
        )
        self.local_max_tokens = models_cfg.get("llm_max_new_tokens", 256)
        self.local_temperature = models_cfg.get("llm_temperature", 0.3)

        # OpenAI-compatible settings (used when use_local=False)
        self.api_base = llm_cfg.get("api_base")
        self.api_key = llm_cfg.get("api_key")
        self.api_model = llm_cfg.get("model")
        self.api_temperature = llm_cfg.get("temperature", 0.2)
        self.api_max_tokens = llm_cfg.get("max_tokens", 256)

        # Lazily loaded HF components
        self._hf_model = None
        self._hf_tokenizer = None
        self._hf_pipe = None

        # Unified name used by telemetry / agents
        self.model_name = (
            self.local_model_id
            if (self.use_local and self.local_model_id)
            else (self.api_model or self.local_model_id or "unknown-model")
        )

        # OpenAI client (lazy)
        self._client = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def chat(self, messages: List[Dict[str, str]], **overrides) -> str:
        """
        Route to HF or OpenAI-compatible backend based on `use_local`.

        `messages` follow the OpenAI chat schema:
            [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
        """

        if self.use_local:
            return self._chat_hf(messages, **overrides)
        else:
            return self._chat_openai(messages, **overrides)

    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """
        Compatibility helper used by LLMGeneratorAgent.

        Exposes a simple text-generation interface:

            text = router.generate(
                prompt="...",
                max_tokens=512,
                temperature=0.2,
            )

        Internally this delegates to `chat(...)` with a single user message.
        """
        messages = [{"role": "user", "content": prompt}]
        overrides: Dict[str, Any] = {}
        if max_tokens is not None:
            overrides["max_tokens"] = max_tokens
        if temperature is not None:
            overrides["temperature"] = temperature
        return self.chat(messages, **overrides)

    # ------------------------------------------------------------------
    # HF backend
    # ------------------------------------------------------------------

    def _load_hf(self):
        if self._hf_pipe is not None:
            return

        if not self.local_model_id:
            raise ValueError("Missing models.llm_model for local HF mode.")

        self._hf_tokenizer = AutoTokenizer.from_pretrained(self.local_model_id)
        self._hf_model = AutoModelForCausalLM.from_pretrained(
            self.local_model_id,
            torch_dtype=torch.float16
            if "cuda" in self.local_device
            else torch.float32,
            device_map="auto" if "cuda" in self.local_device else None,
        )

        self._hf_pipe = pipeline(
            "text-generation",
            model=self._hf_model,
            tokenizer=self._hf_tokenizer,
            device=0 if "cuda" in self.local_device else -1,
        )

    def _chat_hf(self, messages: List[Dict[str, str]], **overrides) -> str:
        """
        HF backend: assemble prompt → text-generation → extract output.
        """
        self._load_hf()

        max_tokens = overrides.get("max_tokens", self.local_max_tokens)
        temperature = overrides.get("temperature", self.local_temperature)

        prompt = self._messages_to_prompt(messages)

        out = self._hf_pipe(
            prompt,
            max_new_tokens=max_tokens,
            do_sample=(temperature > 0),
            temperature=temperature,
        )

        if not out or "generated_text" not in out[0]:
            raise RuntimeError("HF text-generation returned no output.")

        full = out[0]["generated_text"]
        return full[len(prompt) :].strip()

    # ------------------------------------------------------------------
    # OpenAI-compatible backend
    # ------------------------------------------------------------------

    def _load_openai(self):
        if self._client is None:
            if not self.api_base or not self.api_key:
                raise ValueError(
                    "Missing llm.api_base or llm.api_key for OpenAI-compatible mode."
                )
            self._client = OpenAI(base_url=self.api_base, api_key=self.api_key)

    def _chat_openai(self, messages: List[Dict[str, str]], **overrides) -> str:
        self._load_openai()

        temperature = overrides.get("temperature", self.api_temperature)
        max_tokens = overrides.get("max_tokens", self.api_max_tokens)

        resp = self._client.chat.completions.create(
            model=self.api_model,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=messages,
        )

        return resp.choices[0].message.content.strip()

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _messages_to_prompt(messages: List[Dict[str, str]]) -> str:
        """
        Convert OpenAI Chat messages into a single prompt for HF text-generation.
        """
        out = []
        for msg in messages:
            role = msg["role"].upper()
            out.append(f"{role}: {msg['content']}")
        out.append("ASSISTANT:")
        return "\n".join(out)
