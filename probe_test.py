"""
probe_test.py — Test harness for all ITLProbes methods against a vLLM endpoint.

Run modes:
    python probe_test.py                   # run all suites
    python probe_test.py --suite single    # one suite only
    python probe_test.py --suite monitor   # continuous monitor (10 iterations)
"""

# Run all suites in order (baseline → single → ttft → burst → ... → summary)                                                                                                                           
  #python probe_test.py                                                                                                                                                                                 
                                                                                                                                                                                                         
  # Run just one suite (baseline calibration runs automatically first)                                                                                                                                   
  #python probe_test.py --suite single                                                                                                                                                                    
  #python probe_test.py --suite ttft                                                                                                                                                                      
  #python probe_test.py --suite burst                                                                                                                                                                     
  #python probe_test.py --suite preemption
  #python probe_test.py --suite scan                                                                                                                                                                      
  #python probe_test.py --suite kv
  #python probe_test.py --suite tpot                                                                                                                                                                      
  #python probe_test.py --suite abort                                                                                                                                                                     
  #python probe_test.py --suite monitor     # 10 iterations, 3s apart (~30s)
  #python probe_test.py --suite classify
  #python probe_test.py --suite lb          # load balancer detection via APC consistency VERY EXPENSIVE
  #python probe_test.py --suite summary     # reads rolling history, no new requests                                                                                                                      
                                                                                                                                                                                                         
  # Run baseline only (no auto-prepend)                                                                                                                                                                  
  # To force a fresh baseline (server changed, pod restarted etc.):                                                                                                                                        
  #rm probes/baseline_cache.json                                                                                                                                                                          
  #python probe_test.py --suite baseline

from __future__ import annotations

import argparse
import asyncio
import json
from typing import Callable, Dict, List

from core.base_attack import APIFormat, TargetConfig
from probes.itl_probes import InfraState, ITLProbes, ProbeConfig, ProbeResult

import secret

# ─────────────────────────────────────────────────────────────────────────────
# Targets
# ─────────────────────────────────────────────────────────────────────────────
google = TargetConfig(
    base_url="https://generativelanguage.googleapis.com/v1beta/openai",
    model="gemini-2.5-flash",
    api_format=APIFormat.CUSTOM,
    api_key=secret.google_api_paid,
    timeout=300.0,
)

runpod_vllm = TargetConfig(
    base_url="https://4bd8ygkrqvok94-8000.proxy.runpod.net",
    model="deepseek-r1-7b",
    api_format=APIFormat.CUSTOM,
    api_key=secret.runpod_api_key,
    timeout=600.0,
)

gcp = TargetConfig(
    base_url="http://34.6.205.17",
    model="qwen3-14b",
    api_format=APIFormat.CUSTOM,
    timeout=600.0,
    system_prompt="You are the big boss man",
)

local = TargetConfig(
    base_url="http://localhost:11434",
    model="deepseek-r1:8b",
    api_format=APIFormat.OLLAMA,
    api_key=None,
    timeout=120.0,
)

TARGET = gcp# ← switch to `local` for local testing


# ─────────────────────────────────────────────────────────────────────────────
# Pretty printer helpers
# ─────────────────────────────────────────────────────────────────────────────

SEP  = "─" * 60
SEP2 = "═" * 60

def header(title: str) -> None:
    print(f"\n{SEP2}\n  {title}\n{SEP2}")

def section(title: str) -> None:
    print(f"\n{SEP}\n  {title}\n{SEP}")

def dump(d: dict, indent: int = 2) -> None:
    pad = " " * indent
    for k, v in d.items():
        print(f"{pad}{k:<30s}: {json.dumps(v) if isinstance(v, list) else v}")


# ─────────────────────────────────────────────────────────────────────────────
# Suite functions
# ─────────────────────────────────────────────────────────────────────────────

async def suite_baseline(probes: ITLProbes) -> None:
    header("SUITE: calibrate_baseline()")
    result = await probes.calibrate_baseline(n_samples=5, warm_up=1, inter_delay_s=0.5)
    dump(result)


async def suite_single(probes: ITLProbes) -> None:
    header("SUITE: single_probe()")

    section("Default prompt (fixed_50)")
    r = await probes.single_probe()
    dump(r.to_dict())

    section("Custom prompt")
    r = await probes.single_probe(
        prompt="List the planets of the solar system.",
        max_tokens=80,
    )
    dump(r.to_dict())
    print(f"  raw_text preview: {repr(r.raw_text[:120])}")


async def suite_ttft(probes: ITLProbes) -> None:
    header("SUITE: ttft_probe()")

    section("Streaming TTFT probe")
    r = await probes.ttft_probe(use_streaming=True)
    dump(r.to_dict())

    section("Non-streaming TTFT probe (round-trip approximation)")
    r = await probes.ttft_probe(use_streaming=False)
    dump(r.to_dict())


async def suite_burst(probes: ITLProbes) -> None:
    header("SUITE: burst_probe()")

    section(f"Concurrent probes (n={probes.config.burst_concurrency})")
    b = await probes.burst_probe()
    dump(b.to_dict())
    print(f"\n  Individual results:")
    print(f"  {'#':<4s} {'ttft_s':>8s} {'mean_itl_ms':>12s} {'tokens':>7s} {'ok':>5s}")
    print(f"  {'─'*4} {'─'*8} {'─'*12} {'─'*7} {'─'*5}")
    for i, r in enumerate(b.individual):
        print(f"  {i:<4d} {r.ttft_s:>8.4f} {r.mean_itl_ms:>12.3f} "
              f"{r.tokens_out:>7d} {str(r.success):>5s}")


async def suite_preemption(probes: ITLProbes) -> None:
    header("SUITE: preemption_probe()")

    section("10-second preemption watch window")
    result = await probes.preemption_probe(
        duration_s=10.0,
        max_tokens=200,
        spike_threshold=5.0,
    )
    dump({k: v for k, v in result.items() if k != "spikes"})

    if result.get("tokens_received", 0) == 0:
        print(f"\n  No tokens received — request stuck in WAITING queue (HOL blocking)")
        print(f"  Preemption probe requires the request to reach RUNNING state first")
    else:
        spikes = result.get("spikes", [])
        if spikes:
            print(f"\n  Spikes detected ({len(spikes)}):")
            for s in spikes:
                print(f"    token={s['token_index']:>4d}  itl={s['itl_s']:.4f}s  "
                      f"ratio={s['ratio']:.1f}x  elapsed={s['elapsed_s']:.2f}s")
        else:
            print(f"  No spikes detected")


async def suite_scan(probes: ITLProbes) -> None:
    header("SUITE: memory_pressure_scan()")

    section(f"Sequential ITL time-series (8 probes, {probes.config.probe_timeout_s:.0f}s per-probe timeout)")
    results = await probes.memory_pressure_scan(n_probes=8, interval_s=1.0)
    print(f"  {'#':<4s} {'mean_itl_ms':>12s} {'ttft_s':>8s} {'kv_est':>8s} {'tokens':>7s} {'status':>18s}")
    print(f"  {'─'*4} {'─'*12} {'─'*8} {'─'*8} {'─'*7} {'─'*18}")
    for i, r in enumerate(results):
        status = r.error if not r.success else "ok"
        print(f"  {i:<4d} {r.mean_itl_ms:>12.3f} {r.ttft_s:>8.4f} "
              f"{r.kv_usage_est:>8.4f} {r.tokens_out:>7d} {str(status):>18s}")

    good = [r for r in results if r.success and r.mean_itl_ms > 0]
    n_blocked = sum(1 for r in results if not r.success)
    if n_blocked:
        print(f"\n  {n_blocked}/{len(results)} probes timed out (HOL blocking)")
    if len(good) >= 2:
        first, last = good[0].mean_itl_ms, good[-1].mean_itl_ms
        if n_blocked:
            verdict = "⚠ HOL blocking detected"
        elif last > first * 1.1:
            verdict = "↑ rising"
        elif last < first * 0.9:
            verdict = "↓ falling"
        else:
            verdict = "→ stable"
        print(f"  ITL trend: {verdict}  ({first:.1f} ms → {last:.1f} ms)")


async def suite_kv(probes: ITLProbes) -> None:
    header("SUITE: estimate_kv_usage()")

    section("5 consecutive KV usage estimates")
    estimates = []
    for i in range(5):
        est = await probes.estimate_kv_usage()
        estimates.append(est)
        print(f"  [{i}] Û_sys = {est:.4f}")

    if estimates:
        print(f"\n  mean={sum(estimates)/len(estimates):.4f}  "
              f"min={min(estimates):.4f}  max={max(estimates):.4f}")


async def suite_tpot(probes: ITLProbes) -> None:
    header("SUITE: tpot_probe()")

    section("5-probe TPOT measurement")
    result = await probes.tpot_probe(n_probes=5)
    dump(result)


async def suite_abort(probes: ITLProbes) -> None:
    header("SUITE: request_timeout_probe()")

    section("10 concurrent probes — failure/timeout rate")
    result = await probes.request_timeout_probe(
        n_probes=10,
        hard_timeout=60.0,
        max_tokens=100,
    )
    dump(result)


async def suite_monitor(probes: ITLProbes) -> None:
    header("SUITE: monitor()  [10 iterations, 3s interval]")

    section("Live probe stream with state-change callbacks")
    iterations: List[tuple] = []

    def on_probe(result: ProbeResult, state: InfraState) -> None:
        kv  = result.kv_usage_est
        itl = result.mean_itl_ms
        iterations.append((state, kv, itl))
        kv_str  = "HOL/timeout" if kv < 0 else f"{kv:.4f}"
        itl_str = "---" if itl == 0.0 and not result.success else f"{itl:.2f}ms"
        print(f"  probe  state={state.value:<12s}  "
              f"Û_sys={kv_str:<11s}  mean_itl={itl_str:<10s}  ttft={result.ttft_s:.4f}s")

    def on_state_change(old: InfraState, new: InfraState) -> None:
        print(f"  *** STATE CHANGE: {old.value} → {new.value} ***")

    await probes.monitor(
        interval_s=3.0,
        on_probe=on_probe,
        on_state_change=on_state_change,
        max_iterations=10,
    )

    section("Monitor summary")
    if iterations:
        states   = [r[0].value for r in iterations]
        kv_vals  = [r[1] for r in iterations if r[1] >= 0]
        kv_avg   = sum(kv_vals) / len(kv_vals) if kv_vals else -1.0
        itl_vals = [r[2] for r in iterations if r[2] > 0]
        itl_avg  = sum(itl_vals) / len(itl_vals) if itl_vals else 0.0
        print(f"  total probes  : {len(iterations)}")
        print(f"  states seen   : {sorted(set(states))}")
        print(f"  mean Û_sys    : {kv_avg:.4f}")
        print(f"  mean ITL (ms) : {itl_avg:.2f}")


async def suite_classify(probes: ITLProbes) -> None:
    header("SUITE: classify_state()")

    section("State from fresh single probe")
    r     = await probes.single_probe()
    state = probes.classify_state(probe=r)
    b_itl  = max(probes.config.baseline_itl_ms,  1e-6)
    b_ttft = max(probes.config.baseline_ttft_s,  1e-6)
    print(f"  state         : {state.value}")
    print(f"  itl_ratio     : {r.mean_itl_ms / b_itl:.3f}×  "
          f"(baseline {b_itl:.1f} ms → current {r.mean_itl_ms:.1f} ms)")
    print(f"  ttft_ratio    : {r.ttft_s / b_ttft:.3f}×  "
          f"(baseline {b_ttft:.4f} s → current {r.ttft_s:.4f} s)")


async def suite_lb(probes: ITLProbes) -> None:
    header("SUITE: load_balancer_probe()")

    n_warmup      = 3
    prompt_repeats = 12
    section(f"6-pair prefix-cache consistency test  "
            f"(warmup={n_warmup}, prompt_repeats={prompt_repeats} → ~{prompt_repeats * 400} tokens)")
    result = await probes.load_balancer_probe(
        n_pairs=6,
        inter_pair_delay_s=1.0,
        n_warmup=n_warmup,
        prompt_repeats=prompt_repeats,
    )

    if not result["success"]:
        print(f"  error: {result.get('error')}")
        return

    print(f"  same_backend      : {result['same_backend']}")
    print(f"  confidence        : {result['confidence']}")
    print(f"  verdict           : {result['verdict']}")
    print(f"  cache_hit_rate    : {result['cache_hit_rate']:.1%}  "
          f"({result['n_cache_hits']}/{result['n_pairs']} pairs)")
    print(f"  median_ttft_ratio : {result['median_ttft_ratio']:.3f}×  "
          f"(warm/cold — lower = stronger cache hit)")
    print()
    print(f"  {'pair':<5s} {'ttft_cold_s':>11s} {'ttft_warm_s':>11s} "
          f"{'ratio':>7s} {'hit':>5s}")
    print(f"  {'─'*5} {'─'*11} {'─'*11} {'─'*7} {'─'*5}")
    for p in result["pairs"]:
        print(f"  {p['pair']:<5d} {p['ttft_cold_s']:>11.4f} {p['ttft_warm_s']:>11.4f} "
              f"{p['ratio']:>7.3f} {'yes' if p['cache_hit'] else 'no':>5s}")


async def suite_summary(probes: ITLProbes) -> None:
    header("SUITE: rolling_summary()")
    dump(probes.rolling_summary())


# ─────────────────────────────────────────────────────────────────────────────
# Suite registry
# ─────────────────────────────────────────────────────────────────────────────

SUITES: Dict[str, Callable] = {
    "baseline":   suite_baseline,
    "single":     suite_single,
    "ttft":       suite_ttft,
    "burst":      suite_burst,
    "preemption": suite_preemption,
    "scan":       suite_scan,
    "kv":         suite_kv,
    "tpot":       suite_tpot,
    "abort":      suite_abort,
    "monitor":    suite_monitor,
    "classify":   suite_classify,
    "lb":         suite_lb,
    "summary":    suite_summary,
}

# baseline always runs first; summary always runs last
ALL_ORDER = [
    "baseline",
    "single", "ttft", "burst",
    "tpot", "kv", "scan",
    "preemption", "abort",
    "monitor",
    "classify", "lb", "summary",
]


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

BASELINE_CACHE  = "probes/results/baseline_cache.json"
HISTORY_FILE    = "probes/results/probe_history.jsonl"


async def main(suite_name: str) -> None:
    cfg = ProbeConfig(
        target=TARGET,
        max_probe_tokens=256,
        temperature=0.0,
        baseline_window=5,
        burst_concurrency=4,
        rolling_window=20,
        itl_fill_threshold=1.5,
        itl_saturated_threshold=3.0,
        ttft_hol_threshold=5.0,
        itl_cv_thrash_threshold=0.5,
        regressor_backend="linear",
        baseline_cache_path=BASELINE_CACHE,
        history_path=HISTORY_FILE,
    )
    probes = ITLProbes(cfg)

    print(f"[probe_test] Target : {TARGET.base_url}  ({TARGET.model})")

    try:
        # --suite baseline always forces fresh calibration (and clears history).
        # Otherwise auto-calibrate only when no cached baseline is available.
        if suite_name == "baseline" or not probes._baseline_calibrated:
            if suite_name != "baseline":
                print("[probe_test] No cached baseline found — running calibration...")
            await suite_baseline(probes)
        else:
            print(f"[probe_test] Baseline loaded from cache  "
                  f"(itl={cfg.baseline_itl_ms:.3f} ms, ttft={cfg.baseline_ttft_s:.4f} s)")

        if suite_name == "all":
            for name in ALL_ORDER:
                await SUITES[name](probes)
        elif suite_name != "baseline":
            await SUITES[suite_name](probes)

        print(f"\n{SEP2}\n  Done.\n{SEP2}")

    finally:
        await probes.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ITLProbes test harness")
    parser.add_argument(
        "--suite",
        choices=list(SUITES.keys()) + ["all"],
        default="all",
        help="Probe suite to run (default: all)",
    )
    args = parser.parse_args()
    asyncio.run(main(args.suite))
