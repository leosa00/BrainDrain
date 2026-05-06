"""
attacks/reasoning_bomb.py — ReasoningBomb PI-DoS attack module.

Based on: "ReasoningBomb: A Stealthy Denial-of-Service Attack by Inducing
Pathologically Long Reasoning in Large Reasoning Models" (Liu et al., 2026)
https://arxiv.org/abs/2602.00154

Attack principle:
  Short, semantically coherent puzzle prompts (128/256/512-token budget tiers)
  are fed to a Large Reasoning Model to induce pathologically long reasoning
  traces (L_rp), maximising the amplification ratio A = (L_rp + L_out) / L_in.
  A 286.7x average amplification ratio is reported in the paper.

Usage:
  The attack loads prompts from a local sample_puzzles.json file whose expected
  schema is:
      {
          "128": ["puzzle text ...", ...],   # short-budget tier
          "256": ["puzzle text ...", ...],   # medium-budget tier
          "512": ["puzzle text ...", ...]    # long-budget tier
      }
  Download the file from:
      https://github.com/SaFo-Lab/ReasoningBomb/blob/main/data/sample_puzzles.json
  and save it alongside this module or pass an explicit path via
  ReasoningBombConfig.puzzle_file.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from core.base_attack import (
    APIFormat,
    AttackConfig,
    AttackResult,
    BaseAttack,
    TokenMetrics,
)

# ─────────────────────────────────────────────
# Budget Tiers
# ─────────────────────────────────────────────

BUDGET_TIERS = ("128", "256", "512")


# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────

@dataclass
class ReasoningBombConfig:
    """
    Extended configuration for ReasoningBomb.
    Passed alongside AttackConfig; holds attack-specific parameters.
    """
    # Path to the sample_puzzles.json file from the ReasoningBomb GitHub repo.
    puzzle_file: str = "../prompts/sample_puzzles.json"

    # Token-budget tier to pull puzzles from: "128", "256", or "512".
    # Shorter budgets yield higher amplification ratios (A = L_gen / L_in).
    budget_tier: str = "128"

    # Select a specific puzzle by index, or None for random selection.
    puzzle_index: Optional[int] = None

    # Optional system prompt prepended to every request.
    # Keeping it None preserves the naturalness / stealthiness of the attack.
    system_prompt: Optional[str] = None

    # When True the module prints per-chunk token deltas (useful for debugging).
    verbose_stream: bool = False

    def __post_init__(self) -> None:
        if self.budget_tier not in BUDGET_TIERS:
            raise ValueError(
                f"budget_tier must be one of {BUDGET_TIERS}, "
                f"got {self.budget_tier!r}"
            )


# ─────────────────────────────────────────────
# Puzzle Loader
# ─────────────────────────────────────────────

class PuzzleLoader:
    """
    Loads and indexes the ReasoningBomb sample_puzzles.json dataset.

    Expected schema
    ---------------
    {
        "128": ["puzzle ...", "puzzle ...", ...],
        "256": ["puzzle ...", ...],
        "512": ["puzzle ...", ...]
    }

    Flat-list and dict-of-dicts schemas are also handled gracefully.
    """

    def __init__(self, puzzle_file: str) -> None:
        self.path = Path(puzzle_file)
        self._data: dict[str, list[str]] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            raise FileNotFoundError(
                f"Puzzle file not found: {self.path}\n"
                "Download from: https://github.com/SaFo-Lab/ReasoningBomb"
                "/blob/main/data/sample_puzzles.json"
            )
        with self.path.open("r", encoding="utf-8") as fh:
            raw = json.load(fh)

        # Unwrap common top-level wrapper keys
        if isinstance(raw, dict):
            for wrapper_key in ("puzzles", "data", "samples", "dataset", "problems"):
                if wrapper_key in raw and isinstance(raw[wrapper_key], dict):
                    raw = raw[wrapper_key]
                    break

        if isinstance(raw, list):
            for entry in raw:
                if isinstance(entry, dict):
                    tier_key = str(entry.get("budget", entry.get("tier", "unknown")))
                    self._data.setdefault(tier_key, []).append(
                        self._extract_text(entry)
                    )
                elif isinstance(entry, str):
                    self._data.setdefault("default", []).append(entry)

        elif isinstance(raw, dict):
            for tier, items in raw.items():
                tier_key = str(tier)
                if not items:
                    self._data[tier_key] = []
                elif isinstance(items, str):
                    self._data[tier_key] = [items]
                elif isinstance(items, list):
                    if len(items) == 0:
                        self._data[tier_key] = []
                    elif isinstance(items[0], str):
                        self._data[tier_key] = items
                    elif isinstance(items[0], dict):
                        self._data[tier_key] = [
                            self._extract_text(i) for i in items
                        ]
                    else:
                        self._data[tier_key] = [str(i) for i in items]
                elif isinstance(items, dict):
                    self._data[tier_key] = [
                        self._extract_text(v) if isinstance(v, dict) else str(v)
                        for v in items.values()
                    ]
                else:
                    self._data[tier_key] = [str(items)]
        else:
            raise ValueError(
                f"Unrecognised puzzle file format in {self.path}. "
                f"Top-level type is {type(raw).__name__}, expected dict or list."
            )


    @staticmethod
    def _extract_text(item: dict) -> str:
        for key in ("prompt", "text", "question", "puzzle", "input"):
            if key in item:
                return str(item[key])
        # fallback: first string value in the dict
        for v in item.values():
            if isinstance(v, str):
                return v
        raise ValueError(f"Cannot extract puzzle text from entry: {item}")

    def get(self, tier: str, index: Optional[int] = None) -> str:
        """Return a puzzle for the given tier, randomly if index is None."""
        if tier not in self._data:
            available = list(self._data.keys())
            raise KeyError(
                f"Tier {tier!r} not found in puzzle file. "
                f"Available tiers: {available}"
            )
        pool = self._data[tier]
        if not pool:
            raise ValueError(f"No puzzles found for tier {tier!r}")
        if index is not None:
            return pool[index % len(pool)]
        return random.choice(pool)

    def count(self, tier: str) -> int:
        return len(self._data.get(tier, []))

    def all_tiers(self) -> list[str]:
        return sorted(self._data.keys())

    def __repr__(self) -> str:
        summary = {t: len(v) for t, v in self._data.items()}
        return f"<PuzzleLoader {self.path.name} tiers={summary}>"


# ─────────────────────────────────────────────
# Attack Class
# ─────────────────────────────────────────────

class ReasoningBombAttack(BaseAttack):
    """
    Implements the ReasoningBomb PI-DoS technique.

    Selects a puzzle from the configured budget tier and submits it to the
    target LRM, expecting a pathologically long reasoning trace in return.
    Tracks amplification ratio, reasoning tokens (when exposed by the API),
    TTFT, and per-token ITL via the base class streaming machinery.
    """

    def __init__(
        self,
        config: AttackConfig,
        rb_config: Optional[ReasoningBombConfig] = None,
    ) -> None:
        super().__init__(config)
        self.rb_config = rb_config or ReasoningBombConfig()
        self._loader = PuzzleLoader(self.rb_config.puzzle_file)
        self._selected_puzzle: Optional[str] = None

    # ── Identity ─────────────────────────────

    @property
    def attack_name(self) -> str:
        return f"reasoning_bomb_{self.rb_config.budget_tier}"

    # ── Payload ──────────────────────────────

    def build_payload(self) -> dict[str, Any]:
        """
        Construct the API request body.
        Supports OpenAI, Anthropic, and Ollama formats.
        """
        puzzle = self._loader.get(
            self.rb_config.budget_tier,
            self.rb_config.puzzle_index,
        )
        self._selected_puzzle = puzzle
        self.config.metadata = {
            "budget_tier": self.rb_config.budget_tier,
            "puzzle_preview": puzzle[:120],
        }

        if self.config.target.request_template:
            return self._template_payload(puzzle)

        fmt = self.config.target.api_format
        if fmt == APIFormat.OPENAI:
            return self._openai_payload(puzzle)
        elif fmt == APIFormat.ANTHROPIC:
            return self._anthropic_payload(puzzle)
        elif fmt == APIFormat.OLLAMA:
            return self._ollama_payload(puzzle)
        elif fmt == APIFormat.TEST:
            return self._test_payload()
        elif fmt == APIFormat.VERTEX:
            return self._vertex_payload(puzzle)
        else:
            return self._openai_payload(puzzle)

    def _effective_system_prompt(self) -> Optional[str]:
        """Attack-level system prompt takes precedence over target-level."""
        return self.rb_config.system_prompt or self.config.target.system_prompt

    def _build_messages(self, puzzle: str) -> list[dict]:
        messages = []
        sp = self._effective_system_prompt()
        if sp:
            messages.append({"role": "system", "content": sp})
        prefix = self.config.target.request_prefix
        content = f"{prefix}{puzzle}" if prefix else puzzle
        messages.append({"role": "user", "content": content})
        return messages

    def _template_payload(self, puzzle: str) -> dict[str, Any]:
        """Use request_template as the base body, injecting the puzzle prompt."""
        import copy
        payload = copy.deepcopy(self.config.target.request_template)
        payload_str = json.dumps(payload).replace('"__PROMPT__"', json.dumps(puzzle))
        payload = json.loads(payload_str)
        payload["stream"] = True
        payload["max_tokens"] = self.config.max_tokens
        # Strip extended-thinking params for preflight (max_tokens=64) so it
        # returns quickly instead of running a full reasoning trace.
        if self.config.max_tokens <= 64:
            payload.pop("reasoning", None)
            payload.pop("thinking", None)
        return payload

    def _openai_payload(self, puzzle: str) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model":       self.config.target.model,
            "messages":    self._build_messages(puzzle),
            "max_tokens":  self.config.max_tokens,
            "temperature": self.config.temperature,
            "stream":      self.config.stream,
        }
        if self.config.stream and self.config.target.supports_stream_options:
            payload["stream_options"] = {"include_usage": True}
        return payload

    def _anthropic_payload(self, puzzle: str) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model":      self.config.target.model,
            "messages":   [{"role": "user", "content": puzzle}],
            "max_tokens": self.config.max_tokens,
            "stream":     self.config.stream,
        }
        sp = self._effective_system_prompt()
        if sp:
            payload["system"] = sp
        return payload

    def _ollama_payload(self, puzzle: str) -> dict[str, Any]:
        return {
            "model":    self.config.target.model,
            "messages": self._build_messages(puzzle),
            "stream":   self.config.stream,
            "options": {
                "num_predict": self.config.max_tokens,
                "temperature": self.config.temperature,
            },
        }

    def _vertex_payload(self, puzzle: str) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "contents": [{"role": "user", "parts": [{"text": puzzle}]}],
            "generationConfig": {
                "maxOutputTokens": self.config.max_tokens,
                "temperature":     self.config.temperature,
            },
        }
        sp = self._effective_system_prompt()
        if sp:
            payload["systemInstruction"] = {"parts": [{"text": sp}]}
        return payload

    def _test_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model":       self.config.target.model,
            "messages":    [{"role": "user", "content": "Hey there"}],
            "max_tokens":  self.config.max_tokens,
            "temperature": self.config.temperature,
            "stream":      self.config.stream,
        }
        if self.config.stream:
            payload["stream_options"] = {"include_usage": True}
        return payload
    
    # ── Streaming Parser ─────────────────────

    def parse_stream_chunk(
        self, raw_chunk: str, result: AttackResult
    ) -> Optional[str]:
        """
        Parse one SSE data line.  Handles:
          - OpenAI chat completions (incl. usage chunk at end)
          - Anthropic message streaming
          - Ollama streaming
        Returns the text delta string, or None for non-content chunks.
        """
        try:
            data = json.loads(raw_chunk)
        except json.JSONDecodeError:
            return None

        fmt = self.config.target.api_format
        if fmt == APIFormat.OPENAI:
            return self._parse_openai_chunk(data, result)
        elif fmt == APIFormat.ANTHROPIC:
            return self._parse_anthropic_chunk(data, result)
        elif fmt == APIFormat.OLLAMA:
            return self._parse_ollama_chunk(data, result)
        elif fmt == APIFormat.VERTEX:
            return self._parse_vertex_chunk(data, result)
        elif fmt in (APIFormat.TEST, APIFormat.CUSTOM):
            return self._parse_custom_chunk(data, result)
        return None

    def _parse_custom_chunk(self, data: dict, result: AttackResult) -> Optional[str]:
    # Ollama sends usage in the final chunk alongside choices (not separately)
        usage = data.get("usage")
        if usage:
            result.token_metrics.prompt_tokens     = usage.get("prompt_tokens", 0)
            result.token_metrics.completion_tokens = usage.get("completion_tokens", 0)

        choices = data.get("choices", [])
        if not choices:
            return None

        delta = choices[0].get("delta", {})
        t_now     = result.metadata.get("_t_now", 0.0)
        t_start   = result.metadata.get("_t_start", 0.0)

        reasoning = delta.get("reasoning_content") or delta.get("reasoning")
        if reasoning:
            if result.latency_metrics.ttfrt_s == 0.0:
                result.latency_metrics.ttfrt_s = t_now - t_start
            result.metadata["_streaming_reasoning_tokens"] = (
                result.metadata.get("_streaming_reasoning_tokens", 0) + 1
            )
            if self.rb_config.verbose_stream:
                print(reasoning, end="", flush=True)
            return reasoning

        content = delta.get("content")
        if content:
            if result.latency_metrics.ttft_s == 0.0:
                result.latency_metrics.ttft_s = t_now - t_start
            if self.rb_config.verbose_stream:
                print(content, end="", flush=True)
            return content

        return None

    def _parse_openai_chunk(
        self, data: dict, result: AttackResult
    ) -> Optional[str]:
        # Capture usage whenever it appears — vLLM and OpenAI both send it in
        # the final content chunk (choices present, delta empty, finish_reason
        # set) as well as in a trailing usage-only chunk (choices absent/null).
        # The old guard `data.get("choices") is None` missed the vLLM case.
        usage = data.get("usage")
        if usage:
            result.token_metrics.prompt_tokens     = usage.get("prompt_tokens", 0) or result.token_metrics.prompt_tokens
            result.token_metrics.completion_tokens = usage.get("completion_tokens", 0) or result.token_metrics.completion_tokens
            details = usage.get("completion_tokens_details") or {}
            rt = details.get("reasoning_tokens", 0)
            if rt:
                result.token_metrics.reasoning_tokens = rt

        choices = data.get("choices") or []
        if not choices:
            return None

        delta = choices[0].get("delta", {})

        # ── Reasoning trace (thinking tokens) ────────────────────────────────
        # DeepSeek R1 / vLLM:  delta.reasoning_content
        # QwQ / Qwen3:         delta.reasoning_content  (same key)
        # Some providers:      delta.reasoning
        reasoning = delta.get("reasoning_content") or delta.get("reasoning")
        if reasoning:
            # Record time-to-first-reasoning-token on first chunk
            if result.latency_metrics.ttfrt_s == 0.0:
                t_now   = result.metadata.get("_t_now", 0.0)
                t_start = result.metadata.get("_t_start", 0.0)
                result.latency_metrics.ttfrt_s = t_now - t_start
            # Increment streaming counter used as fallback on timeout
            result.metadata["_streaming_reasoning_tokens"] = (
                result.metadata.get("_streaming_reasoning_tokens", 0) + 1
            )
            if self.rb_config.verbose_stream:
                print(reasoning, end="", flush=True)
            # Return None so _consume_stream doesn't count this as a
            # completion-content token in inter_token_latencies.
            return None

        # ── Regular content ───────────────────────────────────────────────────
        content = delta.get("content")
        if self.rb_config.verbose_stream and content:
            print(content, end="", flush=True)
        return content if content else None

    def _parse_anthropic_chunk(
        self, data: dict, result: AttackResult
    ) -> Optional[str]:
        event_type = data.get("type", "")
        if event_type == "content_block_delta":
            return data.get("delta", {}).get("text")
        if event_type == "message_delta":
            usage = data.get("usage", {})
            result.token_metrics.completion_tokens = usage.get("output_tokens", 0)
        if event_type == "message_start":
            usage = data.get("message", {}).get("usage", {})
            result.token_metrics.prompt_tokens = usage.get("input_tokens", 0)
        return None

    def _parse_ollama_chunk(
        self, data: dict, result: AttackResult
    ) -> Optional[str]:
        if data.get("done"):
            result.token_metrics.prompt_tokens     = data.get("prompt_eval_count", 0)
            result.token_metrics.completion_tokens = data.get("eval_count", 0)
            return None
        return data.get("message", {}).get("content")

    def _parse_vertex_chunk(
        self, data: dict, result: AttackResult
    ) -> Optional[str]:
        # usageMetadata appears in the final chunk
        usage = data.get("usageMetadata", {})
        if usage:
            result.token_metrics.prompt_tokens = (
                usage.get("promptTokenCount", 0) or result.token_metrics.prompt_tokens
            )
            result.token_metrics.completion_tokens = (
                usage.get("candidatesTokenCount", 0) or result.token_metrics.completion_tokens
            )
            result.token_metrics.reasoning_tokens = (
                usage.get("thoughtsTokenCount", 0) or result.token_metrics.reasoning_tokens
            )

        candidates = data.get("candidates", [])
        if not candidates:
            return None
        parts = candidates[0].get("content", {}).get("parts", [])
        for part in parts:
            text = part.get("text", "")
            if not text:
                continue
            if part.get("thought"):
                # Thinking token — track count but don't emit as content
                if result.latency_metrics.ttfrt_s == 0.0:
                    t_now   = result.metadata.get("_t_now", 0.0)
                    t_start = result.metadata.get("_t_start", 0.0)
                    result.latency_metrics.ttfrt_s = t_now - t_start
                result.metadata["_streaming_reasoning_tokens"] = (
                    result.metadata.get("_streaming_reasoning_tokens", 0) + 1
                )
                return None
            if self.rb_config.verbose_stream:
                print(text, end="", flush=True)
            return text
        return None

    # ── Non-streaming Parser ──────────────────

    def parse_full_response(
        self, data: dict[str, Any], result: AttackResult
    ) -> None:
        fmt = self.config.target.api_format

        if fmt in (APIFormat.OPENAI, APIFormat.CUSTOM, APIFormat.TEST):
            usage = data.get("usage", {})
            result.token_metrics.prompt_tokens     = usage.get("prompt_tokens", 0)
            result.token_metrics.completion_tokens = usage.get("completion_tokens", 0)
            details = usage.get("completion_tokens_details", {})
            result.token_metrics.reasoning_tokens  = details.get("reasoning_tokens", 0)
            choices = data.get("choices", [])
            if choices:
                msg = choices[0].get("message", {})
                result.raw_response = msg.get("content", "")
                # Reasoning trace stored in metadata for analysis
                result.metadata["reasoning_trace"] = msg.get("reasoning", "")
                result.token_metrics.reasoning_tokens = len(
                    msg.get("reasoning", "").split()   # word-level estimate
                )

        elif fmt == APIFormat.ANTHROPIC:
            usage = data.get("usage", {})
            result.token_metrics.prompt_tokens     = usage.get("input_tokens", 0)
            result.token_metrics.completion_tokens = usage.get("output_tokens", 0)
            contents = data.get("content", [])
            result.raw_response = " ".join(
                c.get("text", "") for c in contents if c.get("type") == "text"
            )

        elif fmt == APIFormat.OLLAMA:
            result.token_metrics.prompt_tokens     = data.get("prompt_eval_count", 0)
            result.token_metrics.completion_tokens = data.get("eval_count", 0)
            result.raw_response = data.get("message", {}).get("content", "")

        elif fmt == APIFormat.VERTEX:
            usage = data.get("usageMetadata", {})
            result.token_metrics.prompt_tokens     = usage.get("promptTokenCount", 0)
            result.token_metrics.completion_tokens = usage.get("candidatesTokenCount", 0)
            result.token_metrics.reasoning_tokens  = usage.get("thoughtsTokenCount", 0)
            candidates = data.get("candidates", [])
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                result.raw_response = "".join(
                    p.get("text", "") for p in parts if not p.get("thought")
                )


    # ── Accessors ────────────────────────────

    @property
    def selected_puzzle(self) -> Optional[str]:
        """The puzzle text used in the last build_payload() call."""
        return self._selected_puzzle

    @property
    def loader(self) -> PuzzleLoader:
        """Direct access to the PuzzleLoader (useful for listing puzzles)."""
        return self._loader
