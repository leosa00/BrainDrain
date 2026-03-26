#https://github.com/iliaishacked/sponge_examples

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

        fmt = self.config.target.api_format

        if fmt == APIFormat.OPENAI:
            return self._openai_payload(puzzle)
        elif fmt == APIFormat.ANTHROPIC:
            return self._anthropic_payload(puzzle)
        elif fmt == APIFormat.OLLAMA:
            return self._ollama_payload(puzzle)
        elif fmt == APIFormat.TEST:
            return self._test_payload()
        else:
            # CUSTOM: fall back to OpenAI schema
            return self._openai_payload(puzzle)

    def _build_messages(self, puzzle: str) -> list[dict]:
        messages = []
        if self.rb_config.system_prompt:
            messages.append({
                "role": "system",
                "content": self.rb_config.system_prompt,
            })
        messages.append({"role": "user", "content": puzzle})
        return messages

    def _openai_payload(self, puzzle: str) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model":       self.config.target.model,
            "messages":    self._build_messages(puzzle),
            "max_tokens":  self.config.max_tokens,
            "temperature": self.config.temperature,
            "stream":      self.config.stream,
        }
        if self.config.stream:
            # Request token-level usage in the final stream chunk (OpenAI >=1.x)
            payload["stream_options"] = {"include_usage": True}
        return payload

    def _anthropic_payload(self, puzzle: str) -> dict[str, Any]:
        messages = [{"role": "user", "content": puzzle}]
        payload: dict[str, Any] = {
            "model":      self.config.target.model,
            "messages":   messages,
            "max_tokens": self.config.max_tokens,
            "stream":     self.config.stream,
        }
        if self.rb_config.system_prompt:
            payload["system"] = self.rb_config.system_prompt
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

    def _test_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model":       self.config.target.model,
            "messages":    [{"role": "user", "content": "Hey there"}],
            "max_tokens":  self.config.max_tokens,
            "temperature": self.config.temperature,
            "stream":      self.config.stream,
        }
        if self.config.stream:
            # Request token-level usage in the final stream chunk (OpenAI >=1.x)
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
        elif fmt == APIFormat.TEST:
            return self._parse_custom_chunk(data, result)
        elif fmt == APIFormat.CUSTOM: 
            return self._parse_custom_chunk(data,result)
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

        # DeepSeek-R1 exposes the thinking trace under "reasoning" key
        reasoning = delta.get("reasoning")
        if reasoning:
            # Record TTFRT on first reasoning chunk
            if result.latency_metrics.ttfrt_s == 0.0:
                result.latency_metrics.ttfrt_s = t_now - t_start
            if self.rb_config.verbose_stream:
                print(reasoning, end="", flush=True)
            return None

        content = delta.get("content")
        if content:
        # Record TTFCT on first content chunk
            if result.latency_metrics.ttft_s == 0.0:
                result.latency_metrics.ttft_s = t_now - t_start
            if self.rb_config.verbose_stream:
                print(content, end="", flush=True)
            return content

        return None

    def _parse_openai_chunk(
        self, data: dict, result: AttackResult
    ) -> Optional[str]:
        # Final usage chunk (stream_options.include_usage=True)
        if "usage" in data and data.get("choices") is None:
            usage = data["usage"]
            result.token_metrics.prompt_tokens     = usage.get("prompt_tokens", 0)
            result.token_metrics.completion_tokens = usage.get("completion_tokens", 0)
            # reasoning_tokens is inside completion_tokens_details for o-series models
            details = usage.get("completion_tokens_details", {})
            result.token_metrics.reasoning_tokens  = details.get("reasoning_tokens", 0)
            return None

        choices = data.get("choices", [])
        if not choices:
            return None
        delta = choices[0].get("delta", {})

        content = (
            delta.get("reasoning")        # ← thinking trace
            or delta.get("content")
            or delta.get("reasoning_content")
        )
        # Some providers embed usage in the last content chunk
        if "usage" in data:
            usage = data["usage"]
            result.token_metrics.prompt_tokens     = usage.get("prompt_tokens", 0)
            result.token_metrics.completion_tokens = usage.get("completion_tokens", 0)

        content = delta.get("content") or delta.get("reasoning_content")
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
            result.raw_response = (
                data.get("message", {}).get("content", "")
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