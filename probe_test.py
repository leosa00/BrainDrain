"""
probe_test.py — Manual test harness for all ITLProbes methods.
             — Google free-tier rate-limited variant (10 RPM / 250 RPD)

Rate limiting strategy
──────────────────────
Gemini 2.5 Flash free tier (2026):
  • 10 RPM  → effective inter-request gap ≥ 7 s  (safety margin on 6 s hard floor)
  • 250 RPD → RPD budget tracker warns at 200, aborts at 248

Components
──────────
  TokenBucketRateLimiter  — sliding-window token bucket; acquire() blocks until a
                            slot is available, respecting the configured RPM ceiling
  RateLimitedITLProbes    — thin ITLProbes subclass that:
                              1. acquires a limiter slot before every HTTP request
                              2. retries with full-jitter exponential back-off on 429
  RPDTracker              — lightweight daily-request counter with warn / abort thresholds

Run modes
─────────
    python probe_test.py                  # all suites against TARGET
    python probe_test.py --suite single   # one suite only
    python probe_test.py --suite monitor  # monitor loop (10 iterations)
    python probe_test.py --rpm 8          # override effective RPM cap
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import aiohttp

from core.base_attack import APIFormat, TargetConfig
from probes.itl_probes import InfraState, ITLProbes, ProbeConfig, ProbeResult, BurstProbeResult

import secret


# ─────────────────────────────────────────────────────────────────────────────
# Targets
# ─────────────────────────────────────────────────────────────────────────────

google_free = TargetConfig(
    base_url="https://generativelanguage.googleapis.com/v1beta/openai",
    model="gemini-2.5-flash",
    api_format=APIFormat.CUSTOM,
    api_key=secret.google_api_paid,
    timeout=120.0,
)


google_paid = TargetConfig(
    base_url="https://generativelanguage.googleapis.com/v1beta/openai",
    model="gemini-2.5-flash-lite",
    api_format=APIFormat.CUSTOM,
    api_key=secret.google_api_paid,   # same AIza... key, billing just needs to
    timeout=120.0,                    # be enabled in GCP console for this project
    supports_stream_options=True,
)

runpod_vllm = TargetConfig(
    base_url="https://wxjz5hyynv2a6n-8000.proxy.runpod.net",
    model="deepseek-r1-7b",           # must match --served-model-name
    api_format=APIFormat.CUSTOM,       # vLLM is OpenAI-compatible
    api_key=secret.runpod_api_key,
    timeout=600.0,
)

local = TargetConfig(
    base_url="http://localhost:11434",
    model="deepseek-r1:8b",
    api_format=APIFormat.OLLAMA,
    api_key=None,
    timeout=120.0,
)

TARGET = runpod_vllm  # ← switch to `local` for unlimited local testing


# ─────────────────────────────────────────────────────────────────────────────
# Token bucket rate limiter
# ─────────────────────────────────────────────────────────────────────────────

class TokenBucketRateLimiter:
    """Async token-bucket rate limiter with single and batch acquire."""

    def __init__(self, rate_per_minute: float, burst_size: int = 2) -> None:
        self._rate        = rate_per_minute / 60.0
        self._capacity    = float(burst_size)
        self._tokens      = float(burst_size)
        self._last_refill = time.monotonic()
        self._lock        = asyncio.Lock()

    def _refill(self) -> None:
        now     = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens      = min(self._capacity, self._tokens + elapsed * self._rate)
        self._last_refill = now

    async def acquire(self) -> float:
        """Block until 1 token is available. Returns seconds waited."""
        return await self.acquire_many(1)

    async def acquire_many(self, n: int) -> float:
        t_start = time.monotonic()
        needed = float(n)
        while True:
            async with self._lock:
                self._refill()
                if self._tokens >= needed:
                    self._tokens -= needed
                    return time.monotonic() - t_start
                deficit = needed - self._tokens
                wait_s = deficit / max(self._rate, 1e-9)
            await asyncio.sleep(max(0.05, wait_s * 0.9))



# ─────────────────────────────────────────────────────────────────────────────
# RPD budget tracker
# ─────────────────────────────────────────────────────────────────────────────

class RPDTracker:
    """
    Tracks requests-per-day against a hard budget.

    Resets automatically after `window_s` seconds (default 86 400 = 24 h).
    """

    def __init__(
        self,
        daily_budget: int   = 250,
        warn_at:      int   = 200,
        abort_at:     int   = 248,
        window_s:     float = 86_400.0,
    ) -> None:
        self.daily_budget = daily_budget
        self.warn_at      = warn_at
        self.abort_at     = abort_at
        self.window_s     = window_s
        self._count       = 0
        self._window_start = time.monotonic()

    def _maybe_reset(self) -> None:
        if time.monotonic() - self._window_start >= self.window_s:
            self._count        = 0
            self._window_start = time.monotonic()

    def increment(self) -> None:
        """Call before every API request. Raises RuntimeError at abort threshold."""
        self._maybe_reset()
        self._count += 1
        remaining = self.daily_budget - self._count
        if self._count >= self.abort_at:
            raise RuntimeError(
                f"[RPD] Daily budget exhausted: {self._count}/{self.daily_budget} requests used. "
                "Aborting to protect free-tier quota."
            )
        if self._count >= self.warn_at:
            print(f"[RPD] WARNING: {self._count}/{self.daily_budget} daily requests used "
                f"({remaining} remaining)"
            )

    @property
    def used(self) -> int:
        self._maybe_reset()
        return self._count

    @property
    def remaining(self) -> int:
        return max(0, self.daily_budget - self.used)


# ─────────────────────────────────────────────────────────────────────────────
# Rate-limited ITLProbes subclass
# ─────────────────────────────────────────────────────────────────────────────

class RateLimitedITLProbes(ITLProbes):
    """
    ITLProbes subclass that enforces token-bucket rate limiting and retries
    every HTTP request with full-jitter exponential back-off on 429.

    Back-off schedule (full jitter):
        delay_n = random(0, min(cap, base × 2^n))
        base = 10 s, cap = 120 s, max_retries = 5
    """

    def __init__(
        self,
        config:      ProbeConfig,
        limiter:     TokenBucketRateLimiter,
        rpd_tracker: RPDTracker,
    ) -> None:
        super().__init__(config)
        self._limiter     = limiter
        self._rpd         = rpd_tracker
        self._retry_base  = 10.0    # seconds
        self._retry_cap   = 120.0   # seconds
        self._max_retries = 5
    
    # ── Override _stream_probe ────────────────────────────────────────────────
    async def burst_probe(
        self,
        n:          Optional[int] = None,
        prompt:     Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> "BurstProbeResult":
        """
        Pre-acquires all N rate-limit slots atomically BEFORE spawning tasks,
        so all N requests fire concurrently as intended rather than sequentially.
        """
        

        concurrency = n      or self.config.burst_concurrency
        p           = prompt or self._PROMPTS["fixed_50"]
        t           = max_tokens or self.config.token_budget

        # ── 1. Acquire all N tokens in one blocking call ──────────────────
        waited = await self._limiter.acquire_many(concurrency)
        if waited > 0.5:
            print(f"  ⏳ [rate-limit] waited {waited:.1f}s to acquire "
                  f"{concurrency} burst slots")

        # ── 2. Check daily budget for all N requests upfront ─────────────
        for _ in range(concurrency):
            try:
                self._rpd.increment()
            except RuntimeError as exc:
                print(f"  ✗ [RPD] {exc}")
                break

        # ── 3. Fire all N tasks concurrently (no per-task limiter call) ───
        # Call super()._stream_probe directly to skip the acquire in the
        # overridden _stream_probe — tokens already consumed above.
        tasks      = [
            ITLProbes._stream_probe(self, p, t, probe_type="burst")
            for _ in range(concurrency)
        ]
        individual = list(await asyncio.gather(*tasks))

        burst = BurstProbeResult(n_probes=concurrency, individual=individual)
        burst.compute()

        if burst.n_success > 0:
            all_itl_ms = [
                v * 1000.0
                for r in individual if r.success
                for v in r.itl_values
            ]
            burst.kv_usage_est = self.estimator.estimate(all_itl_ms)

        return burst

    async def _stream_probe(
        self,
        prompt:     str,
        max_tokens: int,
        probe_type: str = "single",
    ) -> ProbeResult:
        for attempt in range(self._max_retries + 1):
            # 1. Acquire rate-limiter slot (blocks if at RPM ceiling)
            waited = await self._limiter.acquire()
            if waited > 0.5:
                print(f"  ⏳ [rate-limit] waited {waited:.1f}s before firing {probe_type} probe")

            # 2. Check daily budget
            try:
                self._rpd.increment()
            except RuntimeError as exc:
                from probes.itl_probes import ProbeResult
                r = ProbeResult(probe_type=probe_type)
                r.success, r.error = False, str(exc)
                return r

            # 3. Fire the actual request via parent
            result = await super()._stream_probe(prompt, max_tokens, probe_type)

            # 4. If 429, back-off and retry
            if not result.success and result.error and "429" in result.error:
                if attempt >= self._max_retries:
                    print(f"  ✗ [retry] gave up after {self._max_retries} retries (429)")
                    return result
                delay = random.uniform(
                    0,
                    min(self._retry_cap, self._retry_base * (2 ** attempt)),
                )
                print(
                    f"  🔄 [retry] 429 received — attempt {attempt + 1}/{self._max_retries}, "
                    f"back-off {delay:.1f}s"
                )
                await asyncio.sleep(delay)
                continue

            return result

        return result   # unreachable, satisfies type checker

    # ── Override _nonstream_probe ─────────────────────────────────────────────

    async def _nonstream_probe(
        self,
        prompt:     str,
        max_tokens: int,
        probe_type: str = "ttft_ns",
    ) -> ProbeResult:
        for attempt in range(self._max_retries + 1):
            waited = await self._limiter.acquire()
            if waited > 0.5:
                print(f"  ⏳ [rate-limit] waited {waited:.1f}s before firing {probe_type} probe")

            try:
                self._rpd.increment()
            except RuntimeError as exc:
                from probes.itl_probes import ProbeResult
                r = ProbeResult(probe_type=probe_type)
                r.success, r.error = False, str(exc)
                return r

            result = await super()._nonstream_probe(prompt, max_tokens, probe_type)

            if not result.success and result.error and "429" in result.error:
                if attempt >= self._max_retries:
                    return result
                delay = random.uniform(
                    0,
                    min(self._retry_cap, self._retry_base * (2 ** attempt)),
                )
                print(
                    f"  🔄 [retry] 429 — attempt {attempt + 1}/{self._max_retries}, "
                    f"back-off {delay:.1f}s"
                )
                await asyncio.sleep(delay)
                continue

            return result

        return result


# ─────────────────────────────────────────────────────────────────────────────
# Pretty printer helpers
# ─────────────────────────────────────────────────────────────────────────────

SEP  = "─" * 60
SEP2 = "═" * 60

def header(title: str, rpd: Optional[RPDTracker] = None) -> None:
    suffix = f"  [{rpd.used}/{rpd.daily_budget} RPD used]" if rpd else ""
    print(f"\n{SEP2}")
    print(f"  {title}{suffix}")
    print(SEP2)

def section(title: str) -> None:
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)

def dump(d: dict, indent: int = 2) -> None:
    pad = " " * indent
    for k, v in d.items():
        print(f"{pad}{k:<32s}: {json.dumps(v) if isinstance(v, list) else v}")


# ─────────────────────────────────────────────────────────────────────────────
# Suite functions
# ─────────────────────────────────────────────────────────────────────────────

async def suite_baseline(probes: RateLimitedITLProbes, rpd: RPDTracker) -> None:
    header("SUITE: calibrate_baseline()", rpd)
    # Use 3 samples + 1 warm-up to keep RPD cost low on free tier
    result = await probes.calibrate_baseline(n_samples=8, warm_up=1, inter_delay_s=0.1)
    dump(result)


async def suite_single(probes: RateLimitedITLProbes, rpd: RPDTracker) -> None:
    header("SUITE: single_probe()", rpd)

    section("Default prompt (fixed_50)")
    r = await probes.single_probe()
    dump(r.to_dict())

    section("Custom prompt")
    r = await probes.single_probe(
        prompt="List the planets of the solar system.",
        max_tokens=80,
    )
    dump(r.to_dict())
    print(f"  raw_text: {repr(r.raw_text)}")

async def suite_ttft(probes: RateLimitedITLProbes, rpd: RPDTracker) -> None:
    header("SUITE: ttft_probe()", rpd)

    section("Streaming TTFT probe")
    r = await probes.ttft_probe(use_streaming=True)
    dump(r.to_dict())

    section("Non-streaming TTFT probe (round-trip approximation)")
    r = await probes.ttft_probe(use_streaming=False)
    dump(r.to_dict())


async def suite_burst(probes: RateLimitedITLProbes, rpd: RPDTracker) -> None:
    header("SUITE: burst_probe()", rpd)
    # Keep concurrency low on free tier — 3 concurrent uses 3 RPD slots at once
    section("3 concurrent probes")
    b = await probes.burst_probe(n=3)
    dump(b.to_dict())
    print(f"  {'individual results':<32s}:")
    for i, r in enumerate(b.individual):
        print(f"    [{i}] ttft={r.ttft_s:.4f}s  mean_itl={r.mean_itl_ms:.2f}ms  "
              f"tokens={r.tokens_out}  success={r.success}")


async def suite_preemption(probes: RateLimitedITLProbes, rpd: RPDTracker) -> None:
    header("SUITE: preemption_probe()", rpd)
    # Single long-running request — 1 RPD slot
    section("8-second preemption watch window")
    result = await probes.preemption_probe(
        duration_s=8.0,
        max_tokens=150,
        spike_threshold=5.0,
    )
    dump({k: v for k, v in result.items() if k != "spikes"})
    spikes = result.get("spikes", [])
    if spikes:
        print(f"  {'spikes':<32s}:")
        for s in spikes:
            print(f"    token={s['token_index']:>4d}  itl={s['itl_s']:.4f}s  "
                  f"ratio={s['ratio']:.1f}x  elapsed={s['elapsed_s']:.2f}s")
    else:
        print(f"  {'spikes':<32s}: none detected")


async def suite_hol(probes: RateLimitedITLProbes, rpd: RPDTracker) -> None:
    header("SUITE: hol_probe()", rpd)
    # n_fill=3 uses 4 RPD slots total (3 fill + 1 short)
    section("3 fill requests + 1 short probe")
    result = await probes.hol_probe(n_fill=3)
    dump(result)


async def suite_scan(probes: RateLimitedITLProbes, rpd: RPDTracker) -> None:
    header("SUITE: memory_pressure_scan()", rpd)
    # 5 probes × ~7s limiter gap ≈ 35s elapsed, 5 RPD slots
    section("5 probes — ITL ramp mapping")
    results = await probes.memory_pressure_scan(n_probes=5, interval_s=0.1)
    print(f"  {'#':<4s} {'mean_itl_ms':>12s} {'ttft_s':>8s} {'kv_est':>8s} {'ok':>5s}")
    print(f"  {'-'*4} {'-'*12} {'-'*8} {'-'*8} {'-'*5}")
    for i, r in enumerate(results):
        print(f"  {i:<4d} {r.mean_itl_ms:>12.3f} {r.ttft_s:>8.4f} "
              f"{r.kv_usage_est:>8.4f} {str(r.success):>5s}")


async def suite_kv(probes: RateLimitedITLProbes, rpd: RPDTracker) -> None:
    header("SUITE: estimate_kv_usage()", rpd)
    section("3 consecutive KV estimates")
    for i in range(3):
        est = await probes.estimate_kv_usage()
        print(f"  estimate [{i}]: Û_sys = {est:.4f}")


async def suite_tpot(probes: RateLimitedITLProbes, rpd: RPDTracker) -> None:
    header("SUITE: tpot_probe()", rpd)
    # 3 probes instead of 5 to save RPD budget
    section("3-probe TPOT measurement")
    result = await probes.tpot_probe(n_probes=3)
    dump(result)


async def suite_abort(probes: RateLimitedITLProbes, rpd: RPDTracker) -> None:
    header("SUITE: request_timeout_probe()", rpd)
    # 5 probes to stay well within RPD budget
    section("5 probes — abort / partial-response detection")
    result = await probes.request_timeout_probe(
        n_probes=5,
        hard_timeout=30.0,
        max_tokens=80,
    )
    dump(result)


async def suite_signal(probes: RateLimitedITLProbes, rpd: RPDTracker) -> None:
    header("SUITE: adaptive_fill_signal()", rpd)
    section("3 consecutive control-loop signals (c_sat=0.90)")
    for i in range(3):
        sig    = await probes.adaptive_fill_signal(c_sat=0.90)
        action = sig["action"].upper()
        kv     = sig["kv_usage_est"]
        delta  = sig["delta_mem"]
        state  = sig["state"]
        print(f"  [{i}] action={action:<8s}  Û_sys={kv:.4f}  Δ_mem={delta:+.4f}  state={state}")


async def suite_monitor(probes: RateLimitedITLProbes, rpd: RPDTracker) -> None:
    header("SUITE: monitor()  [max_iterations=6, interval=2s]", rpd)
    section("Live probe stream with state-change callbacks")
    # 6 iterations × 1 RPD each = 6 slots; interval=2s (limiter handles real gap)

    iterations: List[tuple] = []

    def on_probe(result: ProbeResult, state: InfraState) -> None:
        kv  = result.kv_usage_est
        itl = result.mean_itl_ms
        iterations.append((state, kv, itl))
        print(f"  probe  state={state.value:<12s}  "
              f"Û_sys={kv:.4f}  mean_itl={itl:.2f}ms  ttft={result.ttft_s:.4f}s  "
              f"[RPD {rpd.used}/{rpd.daily_budget}]")

    def on_state_change(old: InfraState, new: InfraState) -> None:
        print(f"  *** STATE CHANGE: {old.value} → {new.value} ***")

    stop = asyncio.Event()
    await probes.monitor(
        interval_s=2.0,
        on_probe=on_probe,
        on_state_change=on_state_change,
        stop_event=stop,
        max_iterations=6,
    )

    section("Monitor finished — summary")
    if iterations:
        states = [s.value for s, _, __ in iterations]
        kv_avg = sum(k for _, k, __ in iterations) / len(iterations)
        print(f"  total probes   : {len(iterations)}")
        print(f"  states seen    : {sorted(set(states))}")
        print(f"  mean Û_sys     : {kv_avg:.4f}")


async def suite_summary(probes: RateLimitedITLProbes, rpd: RPDTracker) -> None:
    header("SUITE: rolling_summary()", rpd)
    summary = probes.rolling_summary()
    dump(summary)


async def suite_classify(probes: RateLimitedITLProbes, rpd: RPDTracker) -> None:
    header("SUITE: classify_state()", rpd)

    section("State from latest history (no new request)")
    state = probes.classify_state()
    print(f"  current state: {state.value}")

    section("State from fresh single probe")
    r     = await probes.single_probe()
    state = probes.classify_state(probe=r)
    print(f"  current state  : {state.value}")
    print(f"  itl_ratio      : "
          f"{r.mean_itl_ms / max(probes.config.baseline_itl_ms, 1e-6):.3f}×")
    print(f"  ttft_ratio     : "
          f"{r.ttft_s / max(probes.config.baseline_ttft_s, 1e-6):.3f}×")


# ─────────────────────────────────────────────────────────────────────────────
# Suite registry + ordering
# ─────────────────────────────────────────────────────────────────────────────

SUITES: Dict[str, Callable] = {
    "baseline":   suite_baseline,
    "single":     suite_single,
    "ttft":       suite_ttft,
    "burst":      suite_burst,
    "preemption": suite_preemption,
    "hol":        suite_hol,
    "scan":       suite_scan,
    "kv":         suite_kv,
    "tpot":       suite_tpot,
    "abort":      suite_abort,
    "signal":     suite_signal,
    "monitor":    suite_monitor,
    "classify":   suite_classify,
    "summary":    suite_summary,
}

# "all" order — baseline first; summary last (needs full history)
ALL_ORDER = [
    "baseline",
    "single", "ttft", "burst",
    "tpot", "kv", "scan",
    "hol", "preemption", "abort",
    "signal", "monitor",
    "classify", "summary",
]

# Estimated RPD cost per suite (used for pre-flight budget check)
SUITE_RPD_COST: Dict[str, int] = {
    "baseline":   4,   # 1 warm-up + 3 samples
    "single":     2,
    "ttft":       2,
    "burst":      3,
    "preemption": 1,
    "hol":        4,   # 3 fill + 1 short
    "scan":       5,
    "kv":         3,
    "tpot":       3,
    "abort":      5,
    "signal":     3,
    "monitor":    6,
    "classify":   2,
    "summary":    0,   # no new requests
}


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

async def main(suite_name: str, rpm: float, burst_size: int,daily_budget: int,abort_at: int) -> None:
    # ── Rate-limiting primitives ──────────────────────────────────────────────
    limiter = TokenBucketRateLimiter(rate_per_minute=rpm, burst_size=burst_size)
    rpd     = RPDTracker(daily_budget=daily_budget, warn_at=int(daily_budget * 0.8), abort_at=abort_at)

    # ── Pre-flight RPD budget check ───────────────────────────────────────────
    suites_to_run = ALL_ORDER if suite_name == "all" else (
        (["baseline"] if suite_name != "baseline" else []) + [suite_name]
    )
    estimated_cost = sum(SUITE_RPD_COST.get(s, 2) for s in suites_to_run)
    if estimated_cost > rpd.remaining:
        print(
            f"⚠  Estimated RPD cost ({estimated_cost}) exceeds remaining daily budget "
            f"({rpd.remaining}). Aborting."
        )
        return

    print(f"[probe_test] Target   : {TARGET.base_url}  ({TARGET.model})")
    print(f"[probe_test] RPM cap  : {rpm}  (≥ {60/rpm:.1f}s between requests)")
    print(f"[probe_test] RPD budget: {rpd.remaining} remaining  "
          f"(estimated cost: {estimated_cost})")

    # ── Build probe engine ────────────────────────────────────────────────────
    cfg_free = ProbeConfig(
        target=TARGET,
        token_budget=50,
        temperature=0.0,
        baseline_window=3,
        burst_concurrency=3,
        rolling_window=20,
        itl_fill_threshold=1.5,
        itl_saturated_threshold=3.0,
        ttft_hol_threshold=5.0,
        itl_cv_thrash_threshold=0.5,
        regressor_backend="linear",
    )

    cfg = ProbeConfig(
        target=TARGET,
        token_budget=512,            # ← was 50
        temperature=0.0,
        baseline_window=3,
        burst_concurrency=3,
        rolling_window=20,
        itl_fill_threshold=2.0,
        itl_saturated_threshold=4.0,
        ttft_hol_threshold=2.0, #change to 5.0 for vLLM default
        itl_cv_thrash_threshold=1.2,
        regressor_backend="linear",
        # Gemini 2.5 Flash: disable thinking for probe requests so tokens
        # are not wasted on reasoning — probes need fast, cheap responses
        #probe_extra_body={"thinking": {"type": "disabled"}},
    )
    probes = RateLimitedITLProbes(cfg, limiter, rpd)

    try:
        if suite_name == "all":
            for name in ALL_ORDER:
                await SUITES[name](probes, rpd)
        else:
            if suite_name != "baseline":
                print("\n[probe_test] Calibrating baseline first...")
                await suite_baseline(probes, rpd)
            await SUITES[suite_name](probes, rpd)

        print(f"\n{SEP2}")
        print(f"  All probes completed.  RPD used: {rpd.used}/{rpd.daily_budget}")
        print(SEP2)

    finally:
        await probes.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ITLProbes test harness (rate-limited)")
    parser.add_argument(
        "--suite",
        choices=list(SUITES.keys()) + ["all"],
        default="all",
        help="Probe suite to run (default: all)",
    )
    parser.add_argument(
        "--rpm",
        type=float,
        default=8.0,
        help="Effective RPM cap (default: 8)",
    )
    parser.add_argument(
        "--burst-size",
        type=int,
        default=3,
        dest="burst_size",
        help="Token bucket burst capacity (default: 2)",
    )
    parser.add_argument(
        "--daily-budget",
        type=int,
        default=250,
        dest="daily_budget",
        help="Max requests per day before hard abort (default: 250)",
    )
    parser.add_argument(
        "--abort-at",
        type=int,
        default=248,
        dest="abort_at",
        help="RPD count at which to hard-abort (default: 248)",
    )
    args = parser.parse_args()

    asyncio.run(main(
        suite_name=args.suite,
        rpm=args.rpm,
        burst_size=args.burst_size,
        daily_budget=args.daily_budget,
        abort_at=args.abort_at,
    ))
