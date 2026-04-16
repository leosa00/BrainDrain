"""
probes/itl_probes.py — Black-box ITL side-channel probing for LLM serving infra.

Measures server degradation as a black-box side-channel alongside attack
workloads. ITL (inter-token latency) correlates with GPU KV-cache utilisation
due to memory bandwidth contention, making it a useful proxy for real-time
scheduler state without privileged access.

Probe API:
  calibrate_baseline()    — establish idle-state reference metrics
  single_probe()          — core ITL + TTFT one-shot measurement
  burst_probe()           — N concurrent probes (queue depth / variance)
  ttft_probe()            — TTFT-only fast probe (queue wait indicator)
  preemption_probe()      — long-running probe detecting intra-request spikes
  memory_pressure_scan()  — sequential ITL time-series
  tpot_probe()            — Time-Per-Output-Token across N requests
  request_timeout_probe() — failure rate under load
  estimate_kv_usage()     — live KV cache pressure estimate ∈ [0, 1]
  classify_state()        — InfraState classifier from probe + rolling history
  rolling_summary()       — aggregate stats over recent probe window
  monitor()               — continuous async polling loop with callbacks
"""

from __future__ import annotations

import asyncio
import datetime
import json
import logging
import math
import statistics
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Deque, List, Optional

import aiohttp
import numpy as np

try:
    import lightgbm as lgb
    _HAS_LGB = True
except ImportError:
    _HAS_LGB = False

try:
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False

from core.base_attack import APIFormat, TargetConfig

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Infra State
# ─────────────────────────────────────────────

class InfraState(Enum):
    UNKNOWN     = "unknown"      # No baseline or insufficient data
    IDLE        = "idle"         # Near-baseline ITL/TTFT, KV usage low
    FILLING     = "filling"      # KV usage rising, ITL trending up (1.5–3×)
    SATURATED   = "saturated"    # KV near 100%, ITL >> baseline (3×+)
    HOL_BLOCKED = "hol_blocked"  # TTFT spiked — WAITING queue frozen
    THRASHING   = "thrashing"    # High ITL variance — preemption loop detected
    DEGRADED    = "degraded"     # Generic significant degradation


# ─────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────

@dataclass
class ProbeResult:
    probe_id:      str   = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp:     float = field(default_factory=time.time)
    probe_type:    str   = "unknown"
    raw_text:      str   = ""

    # Core latency
    ttft_s:        float       = 0.0
    total_s:       float       = 0.0
    itl_values:    List[float] = field(default_factory=list)   # per-token, seconds

    # Derived (populated by compute_derived())
    mean_itl_ms:   float = 0.0
    p95_itl_ms:    float = 0.0
    p99_itl_ms:    float = 0.0
    itl_stddev_ms: float = 0.0
    tpot_ms:       float = 0.0
    tokens_out:    int   = 0

    # KV estimate [-1 = not estimated]
    kv_usage_est:  float = -1.0

    success:       bool          = True
    error:         Optional[str] = None

    def compute_derived(self) -> None:
        if not self.itl_values:
            return
        ms = [v * 1000.0 for v in self.itl_values]
        n  = len(ms)
        self.chunk_count = n
        if n > 1 and len(self.raw_text.split()) > n:
            self.tokens_out = len(self.raw_text.split())
        else:
            self.tokens_out = n
        self.mean_itl_ms   = statistics.mean(ms)
        self.tpot_ms       = self.mean_itl_ms
        self.itl_stddev_ms = statistics.stdev(ms) if n > 1 else 0.0
        s = sorted(ms)
        self.p95_itl_ms = s[min(int(math.ceil(0.95 * n)) - 1, n - 1)]
        self.p99_itl_ms = s[min(int(math.ceil(0.99 * n)) - 1, n - 1)]

    def to_dict(self) -> dict:
        return {
            "probe_id":       self.probe_id,
            "timestamp":      self.timestamp,
            "probe_type":     self.probe_type,
            "ttft_s":         round(self.ttft_s, 4),
            "total_s":        round(self.total_s, 4),
            "mean_itl_ms":    round(self.mean_itl_ms, 4),
            "p95_itl_ms":     round(self.p95_itl_ms, 4),
            "p99_itl_ms":     round(self.p99_itl_ms, 4),
            "itl_stddev_ms":  round(self.itl_stddev_ms, 4),
            "tpot_ms":        round(self.tpot_ms, 4),
            "tokens_out":     self.tokens_out,
            "kv_usage_est":   round(self.kv_usage_est, 4),
            "success":        self.success,
            "error":          self.error,
        }


@dataclass
class BurstProbeResult:
    timestamp:      float             = field(default_factory=time.time)
    n_probes:       int               = 0
    n_success:      int               = 0
    individual:     List[ProbeResult] = field(default_factory=list)

    mean_ttft_s:    float = 0.0
    p95_ttft_s:     float = 0.0
    ttft_stddev_s:  float = 0.0
    mean_itl_ms:    float = 0.0
    p95_itl_ms:     float = 0.0
    itl_stddev_ms:  float = 0.0
    kv_usage_est:   float = -1.0

    def compute(self) -> None:
        good = [r for r in self.individual if r.success]
        self.n_success = len(good)
        if not good:
            return

        ttfts = sorted([r.ttft_s      for r in good])
        itls  = sorted([r.mean_itl_ms for r in good if r.mean_itl_ms > 0])
        n_t   = len(ttfts)
        n_i   = len(itls)

        self.mean_ttft_s   = statistics.mean(ttfts)
        self.ttft_stddev_s = statistics.stdev(ttfts) if n_t > 1 else 0.0
        self.mean_itl_ms   = statistics.mean(itls)   if itls else 0.0
        self.itl_stddev_ms = statistics.stdev(itls)  if n_i > 1 else 0.0

        if n_t > 0:
            self.p95_ttft_s = ttfts[min(int(math.ceil(0.95 * n_t)) - 1, n_t - 1)]
        if n_i > 0:
            self.p95_itl_ms = itls [min(int(math.ceil(0.95 * n_i)) - 1, n_i - 1)]

        kv = [r.kv_usage_est for r in good if r.kv_usage_est >= 0]
        if kv:
            self.kv_usage_est = statistics.mean(kv)

    def to_dict(self) -> dict:
        return {
            "n_probes":      self.n_probes,
            "n_success":     self.n_success,
            "mean_ttft_s":   round(self.mean_ttft_s, 4),
            "p95_ttft_s":    round(self.p95_ttft_s, 4),
            "ttft_stddev_s": round(self.ttft_stddev_s, 4),
            "mean_itl_ms":   round(self.mean_itl_ms, 4),
            "p95_itl_ms":    round(self.p95_itl_ms, 4),
            "itl_stddev_ms": round(self.itl_stddev_ms, 4),
            "kv_usage_est":  round(self.kv_usage_est, 4),
        }


# ─────────────────────────────────────────────
# Probe Config
# ─────────────────────────────────────────────

@dataclass
class ProbeConfig:
    target: TargetConfig

    # Probe request sizing (keep small to avoid adding to server load)
    max_probe_tokens:    int   = 512
    temperature:         float = 0.0
    probe_timeout_s:     float = 30.0  # per-probe hard deadline (HOL blocking guard)

    # Baseline calibration
    baseline_window:     int   = 5
    baseline_itl_ms:     float = 0.0   # populated after calibrate_baseline()
    baseline_ttft_s:     float = 0.0
    baseline_kv_est:      float = 0.0   # idle-state KV estimate (normalisation floor)
    baseline_kv_n:        int   = 3     # number of KV samples taken during calibration
    baseline_pure_itl_ms: float = 0.0   # idle-state mean ITL excluding TTFT (decode-only)

    # Burst probe
    burst_concurrency:   int   = 4

    # Rolling window depth for monitor / rolling_summary
    rolling_window:      int   = 20

    # State classification thresholds (multipliers over baseline)
    itl_fill_threshold:       float = 1.5   # ITL > 1.5× baseline → FILLING
    itl_saturated_threshold:  float = 3.0   # ITL > 3×  baseline → SATURATED
    ttft_hol_threshold:       float = 5.0   # TTFT > 5× baseline → HOL_BLOCKED
    itl_cv_thrash_threshold:  float = 0.5   # rolling CoV(ITL) > 0.5 → THRASHING

    # KV usage regressor backend: "lightgbm" | "linear" | "none"
    regressor_backend:   str  = "linear"
    probe_extra_body:    dict = field(default_factory=dict)

    # Path to persist baseline calibration across runs (None = no persistence)
    baseline_cache_path: Optional[str] = None

    # Path to a JSONL file for persistent probe history (None = no persistence).
    # Each successful single_probe() result is appended as one JSON line.
    # On startup the last `rolling_window` entries are loaded into _itl_history
    # so rolling_summary() and classify_state() work across sessions.
    history_path: Optional[str] = None


# ─────────────────────────────────────────────
# KV Usage Estimator
# ─────────────────────────────────────────────

class KVUsageEstimator:
    """
    Regressor: ITL feature vector → Û_sys ∈ [0, 1].

    ITL correlates with GPU KV cache utilisation due to memory bandwidth
    contention, enabling a black-box pressure estimate.

    Backends:
      "lightgbm" — LGBMRegressor (requires supervised training data)
      "linear"   — Ridge regression (requires supervised training data)
      "none"     — unsupervised relative-pressure fallback (always available)

    For supervised training, call fit() with (ITL_sequences, kv_usages) pairs
    collected from a calibration phase where vLLM /metrics is readable.
    Until fit() is called, estimate() uses the unsupervised fallback:
        Û_sys = clip((mean_ITL − baseline_ITL) / baseline_ITL, 0, 1)
    """

    def __init__(self, backend: str = "linear") -> None:
        self.backend  = backend
        self._model   = None
        self._scaler  = None
        self._baseline_itl_ms: float = 0.0
        self._trained: bool = False

    @staticmethod
    def extract_features(itl_ms: List[float]) -> np.ndarray:
        """11-dim feature vector from a variable-length ITL sequence."""
        if not itl_ms:
            return np.zeros(11)
        a = np.asarray(itl_ms, dtype=float)
        slope = float(np.polyfit(np.arange(len(a)), a, 1)[0]) if len(a) >= 3 else 0.0
        cov   = float(np.std(a) / (np.mean(a) + 1e-9))
        return np.array([
            float(np.mean(a)),
            float(np.std(a)),
            float(np.percentile(a, 50)),
            float(np.percentile(a, 75)),
            float(np.percentile(a, 90)),
            float(np.percentile(a, 95)),
            float(np.percentile(a, 99)),
            float(np.min(a)),
            float(np.max(a)),
            slope,
            cov,
        ])

    def fit(
        self,
        itl_sequences: List[List[float]],  # list of ITL-ms sequences
        kv_usages:     List[float],         # ground-truth Û_sys ∈ [0, 1]
    ) -> None:
        """
        Train on (ITL_sequence, kv_usage) pairs.

        Ground-truth labels come from vLLM's /metrics endpoint
        (vllm:gpu_cache_usage_perc) during a controlled calibration phase.
        """
        X = np.array([self.extract_features(seq) for seq in itl_sequences])
        y = np.asarray(kv_usages, dtype=float)

        if self.backend == "lightgbm" and _HAS_LGB:
            self._model = lgb.LGBMRegressor(
                n_estimators=100, max_depth=4,
                learning_rate=0.1, num_leaves=15, verbose=-1,
            )
            self._model.fit(X, y)
            self._trained = True

        elif self.backend == "linear" and _HAS_SKLEARN:
            self._scaler = StandardScaler()
            self._model  = Ridge(alpha=1.0)
            self._model.fit(self._scaler.fit_transform(X), y)
            self._trained = True

        else:
            logger.warning(
                "KVUsageEstimator: backend '%s' unavailable, using fallback.",
                self.backend,
            )

    def set_baseline(self, baseline_itl_ms: float) -> None:
        self._baseline_itl_ms = max(baseline_itl_ms, 1e-6)

    def estimate(self, itl_ms: List[float]) -> float:
        """Return Û_sys ∈ [0.0, 1.0]. Uses trained model if available, else fallback."""
        if not itl_ms:
            return 0.0

        if self._trained and self._model is not None:
            feats = self.extract_features(itl_ms).reshape(1, -1)
            if self._scaler is not None:
                feats = self._scaler.transform(feats)
            return float(np.clip(self._model.predict(feats)[0], 0.0, 1.0))

        # Unsupervised fallback: relative ITL pressure over calibrated baseline
        mean_itl = float(np.mean(itl_ms))
        if self._baseline_itl_ms <= 0:
            return 0.0
        raw  = (mean_itl - self._baseline_itl_ms) / self._baseline_itl_ms
        soft = raw / (1.0 + abs(raw))
        return float(np.clip(soft, 0.0, 1.0))


# ─────────────────────────────────────────────
# Core Probe Engine
# ─────────────────────────────────────────────

class ITLProbes:
    """
    Black-box ITL probing suite for measuring LLM server degradation.

    Intended to run alongside attack workloads to measure their effectiveness
    in real-time via ITL side-channel, without any privileged server access.

    Typical usage:
        probes = ITLProbes(ProbeConfig(target=my_target))
        await probes.calibrate_baseline()

        stop = asyncio.Event()
        monitor_task = asyncio.create_task(probes.monitor(
            interval_s=3.0,
            on_probe=lambda r, s: print(r.mean_itl_ms, s.value),
            stop_event=stop,
        ))
        # ... run attack ...
        stop.set()
        await monitor_task
        print(probes.rolling_summary())
    """

    # Long context used by load_balancer_probe. Placed after a unique UUID prefix
    # so each pair has a unique full prompt (no cross-pair APC pollution) while
    # still being long enough that prefill takes ~100-200 ms on a 7B model,
    # making cache hits clearly visible as a TTFT drop.
    _LB_CONTEXT: str = (
        "The following is a technical overview of distributed systems and GPU computing. "
        "Modern computing relies on hierarchical memory: L1 cache (4-5 cycles), L2 (12-15 cycles), "
        "L3 (40-60 cycles), and DRAM (200+ cycles). Cache coherence protocols keep multi-core "
        "state consistent. In distributed systems, the CAP theorem states that consistency, "
        "availability, and partition tolerance cannot all be guaranteed simultaneously. "
        "Databases navigate this via eventual consistency, Raft/Paxos consensus, and MVCC. "
        "Load balancers distribute traffic using round-robin, least-connections, or consistent "
        "hashing. Consistent hashing maps servers and keys onto a ring, minimising reshuffling "
        "during scale events. Session affinity pins a client to one backend, improving cache "
        "locality but risking hot spots. CDNs use anycast and geographic routing to cut latency. "
        "In GPU computing, the memory hierarchy spans registers, shared memory, L1/L2, and global "
        "DRAM. Transformer attention is memory-bound: each forward pass reads the full KV cache "
        "from DRAM, so bandwidth — not compute — is the bottleneck at long context lengths. "
        "Automatic Prefix Caching (APC) in systems like vLLM reuses computed KV pairs for "
        "identical prompt prefixes, skipping redundant prefill and reducing TTFT proportionally "
        "to the cached prefix length. Block-level caching groups tokens into fixed-size blocks "
        "and evicts via LRU under memory pressure. Speculative decoding uses a small draft model "
        "to propose multiple tokens, which the target model verifies in parallel, increasing "
        "throughput without changing output distribution. Continuous batching interleaves "
        "prefill and decode phases across requests, keeping GPU utilisation high. PagedAttention "
        "manages KV cache as virtual memory pages, eliminating fragmentation. Chunked prefill "
        "splits long-context prefills into fixed-size chunks to bound scheduling latency. "
        "These techniques together allow serving frameworks to sustain high throughput while "
        "keeping per-request latency predictable under load."
    )

    _PROMPTS: dict[str, str] = {
        "minimal":   "Reply with only the single word: ok",
        "short":     "Count from 1 to 10, each number on a new line.",
        "medium":    "List 20 common English words, one per line.",
        "fixed_50":  "Write a paragraph of at least 60 words about the ocean. Do not use bullet points.",
        "fixed_100": "Write two paragraphs totaling at least 120 words about climate change.",
        "repeat_50": "List every number from 1 to 50, each on its own line.",
    }

    def __init__(self, config: ProbeConfig) -> None:
        self.config    = config
        self._session: Optional[aiohttp.ClientSession] = None
        self.estimator = KVUsageEstimator(backend=config.regressor_backend)
        self._itl_history:   Deque[ProbeResult] = deque(maxlen=config.rolling_window)
        self._state_history: Deque[InfraState]  = deque(maxlen=50)
        self._baseline_calibrated = False

        # Auto-load persisted baseline if a cache path is configured
        if config.baseline_cache_path:
            self._load_baseline(config.baseline_cache_path)

        # Pre-populate rolling history from disk
        if config.history_path:
            self._load_history(config.history_path)

    # ── Baseline persistence ──────────────────────────────────────────────────

    def _save_baseline(self, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "baseline_itl_ms":      self.config.baseline_itl_ms,
            "baseline_ttft_s":      self.config.baseline_ttft_s,
            "baseline_kv_est":      self.config.baseline_kv_est,
            "baseline_pure_itl_ms": self.config.baseline_pure_itl_ms,
            "target_url":           self.config.target.base_url,
            "model":           self.config.target.model,
            "saved_at":        datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
        p.write_text(json.dumps(data, indent=2))
        logger.info("Baseline saved → %s", path)

    def _load_baseline(self, path: str) -> bool:
        """Load persisted baseline. Returns True if loaded successfully."""
        p = Path(path)
        if not p.exists():
            return False
        try:
            data = json.loads(p.read_text())
            self.config.baseline_itl_ms = float(data["baseline_itl_ms"])
            self.config.baseline_ttft_s = float(data["baseline_ttft_s"])
            self.config.baseline_kv_est      = float(data.get("baseline_kv_est", 0.0))
            self.config.baseline_pure_itl_ms = float(data.get("baseline_pure_itl_ms", 0.0))
            self.estimator.set_baseline(self.config.baseline_itl_ms)
            self._baseline_calibrated = True
            logger.info(
                "Baseline loaded from %s  (itl=%.3f ms, ttft=%.4f s, kv=%.4f, saved %s)",
                path, self.config.baseline_itl_ms, self.config.baseline_ttft_s,
                self.config.baseline_kv_est, data.get("saved_at", "?"),
            )
            return True
        except Exception as exc:
            logger.warning("Failed to load baseline from %s: %s", path, exc)
            return False

    # ── Probe history persistence ─────────────────────────────────────────────

    def _append_history(self, result: ProbeResult) -> None:
        """Append one probe result to the JSONL history file."""
        path = self.config.history_path
        if not path:
            return
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a") as f:
            f.write(json.dumps(result.to_dict()) + "\n")

    def _load_history(self, path: str) -> None:
        """Load last `rolling_window` entries from the JSONL history file."""
        p = Path(path)
        if not p.exists():
            return
        try:
            lines = p.read_text().splitlines()
            # Take the last rolling_window lines
            recent = lines[-self.config.rolling_window:]
            for line in recent:
                if not line.strip():
                    continue
                d = json.loads(line)
                r = ProbeResult(
                    probe_id    = d.get("probe_id", ""),
                    timestamp   = d.get("timestamp", 0.0),
                    probe_type  = d.get("probe_type", "single"),
                    ttft_s      = d.get("ttft_s", 0.0),
                    total_s     = d.get("total_s", 0.0),
                    mean_itl_ms = d.get("mean_itl_ms", 0.0),
                    p95_itl_ms  = d.get("p95_itl_ms", 0.0),
                    p99_itl_ms  = d.get("p99_itl_ms", 0.0),
                    itl_stddev_ms = d.get("itl_stddev_ms", 0.0),
                    tpot_ms     = d.get("tpot_ms", 0.0),
                    tokens_out  = d.get("tokens_out", 0),
                    kv_usage_est = d.get("kv_usage_est", -1.0),
                    success     = d.get("success", True),
                    error       = d.get("error"),
                )
                self._itl_history.append(r)
            if self._itl_history:
                logger.info(
                    "Probe history loaded from %s  (%d entries)",
                    path, len(self._itl_history),
                )
        except Exception as exc:
            logger.warning("Failed to load probe history from %s: %s", path, exc)

    # ── HTTP session ──────────────────────────────────────────────────────────

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            connector = aiohttp.TCPConnector(ssl=self.config.target.verify_ssl)
            timeout   = aiohttp.ClientTimeout(total=self.config.target.timeout)
            self._session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers={"Content-Type": "application/json"},
            )
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    # ── Payload construction ──────────────────────────────────────────────────

    def _build_payload(self, prompt: str, max_tokens: int, stream: bool = True) -> dict:
        fmt = self.config.target.api_format
        sp  = self.config.target.system_prompt
        msgs = []
        if sp and fmt != APIFormat.ANTHROPIC:
            msgs.append({"role": "system", "content": sp})
        msgs.append({"role": "user", "content": prompt})

        base: dict[str, Any] = {
            "model":       self.config.target.model,
            "temperature": self.config.temperature,
            "stream":      stream,
        }

        if fmt in (APIFormat.OPENAI, APIFormat.CUSTOM, APIFormat.TEST):
            base["messages"]   = msgs
            base["max_tokens"] = max_tokens
            if stream and self.config.target.supports_stream_options:
                base["stream_options"] = {"include_usage": True}

        elif fmt == APIFormat.ANTHROPIC:
            base["messages"]   = msgs
            base["max_tokens"] = max_tokens
            if sp:
                base["system"] = sp

        elif fmt == APIFormat.OLLAMA:
            base["messages"] = msgs
            base["options"]  = {
                "num_predict": max_tokens,
                "temperature": self.config.temperature,
            }

        if self.config.probe_extra_body:
            base.update(self.config.probe_extra_body)

        return base

    # ── SSE token extractor ───────────────────────────────────────────────────

    def _extract_token(self, payload_str: str) -> Optional[str]:
        try:
            data = json.loads(payload_str)
        except json.JSONDecodeError:
            return None

        fmt = self.config.target.api_format
        if fmt in (APIFormat.OPENAI, APIFormat.CUSTOM, APIFormat.TEST):
            choices = data.get("choices", [])
            if not choices:
                return None
            delta = choices[0].get("delta", {})
            content   = delta.get("content")
            reasoning = delta.get("reasoning_content")
            if content is not None:
                return content
            if reasoning is not None:
                return reasoning
            return None

        elif fmt == APIFormat.ANTHROPIC:
            if data.get("type") == "content_block_delta":
                return data.get("delta", {}).get("text")
            return None

        elif fmt == APIFormat.OLLAMA:
            return None if data.get("done") else data.get("message", {}).get("content")

        return None

    # ── Raw stream consumer ───────────────────────────────────────────────────

    async def _stream_probe(
        self,
        prompt:     str,
        max_tokens: int,
        probe_type: str = "single",
    ) -> ProbeResult:
        result  = ProbeResult(probe_type=probe_type)
        payload = self._build_payload(prompt, max_tokens, stream=True)
        session = await self._get_session()
        t_start = time.perf_counter()
        t_last  = t_start

        async def _do() -> None:
            nonlocal t_last
            async with session.post(
                self.config.target.endpoint,
                json=payload,
                headers=self.config.target.auth_headers,
            ) as resp:
                if resp.status >= 400:
                    result.success = False
                    result.error   = f"HTTP {resp.status}: {(await resp.text())[:200]}"
                    return

                buf  = b""
                done = False
                async for raw in resp.content:
                    if done:
                        break
                    buf += raw
                    while b"\n" in buf:
                        line_b, buf = buf.split(b"\n", 1)
                        line = line_b.decode("utf-8", errors="replace").strip()

                        if self.config.target.api_format == APIFormat.OLLAMA:
                            if not line:
                                continue
                            data_str = line
                        else:
                            if not line or not line.startswith("data:"):
                                continue
                            data_str = line[5:].strip()

                        if data_str == "[DONE]":
                            done = True
                            break

                        t_now = time.perf_counter()
                        token = self._extract_token(data_str)
                        if token is not None:
                            if not result.itl_values:
                                result.ttft_s = t_now - t_start
                            result.itl_values.append(t_now - t_last)
                            t_last = t_now
                            result.raw_text += token

        try:
            await asyncio.wait_for(_do(), timeout=self.config.probe_timeout_s)
        except asyncio.TimeoutError:
            result.success = False
            result.error   = f"timeout>{self.config.probe_timeout_s:.0f}s (HOL blocking)"
        except aiohttp.ClientError as exc:
            result.success, result.error = False, str(exc)

        result.total_s = time.perf_counter() - t_start
        result.compute_derived()
        return result

    async def _nonstream_probe(
        self,
        prompt:     str,
        max_tokens: int,
        probe_type: str = "ttft_ns",
    ) -> ProbeResult:
        """Non-streaming: total round-trip time ≈ TTFT for short outputs."""
        result  = ProbeResult(probe_type=probe_type)
        payload = self._build_payload(prompt, max_tokens, stream=False)
        session = await self._get_session()
        t_start = time.perf_counter()

        async def _do() -> None:
            async with session.post(
                self.config.target.endpoint,
                json=payload,
                headers=self.config.target.auth_headers,
            ) as resp:
                if resp.status >= 400:
                    result.success = False
                    result.error   = f"HTTP {resp.status}"
                else:
                    await resp.json()
                    result.ttft_s = time.perf_counter() - t_start

        try:
            await asyncio.wait_for(_do(), timeout=self.config.probe_timeout_s)
        except asyncio.TimeoutError:
            result.success = False
            result.error   = f"timeout>{self.config.probe_timeout_s:.0f}s (HOL blocking)"
        except Exception as exc:
            result.success, result.error = False, str(exc)

        result.total_s = time.perf_counter() - t_start
        return result

    # ═════════════════════════════════════════════════════════════════════════
    # 1. calibrate_baseline
    # ═════════════════════════════════════════════════════════════════════════

    async def calibrate_baseline(
        self,
        n_samples:     Optional[int] = None,
        warm_up:       int           = 1,
        inter_delay_s: float         = 0.5,
    ) -> dict:
        """
        Establish baseline ITL and TTFT under idle conditions.

        Call before any attack phase. Runs `warm_up` discarded probes to avoid
        cold-start effects, then collects `n_samples` measurements.
        """
        n = n_samples or self.config.baseline_window

        for _ in range(warm_up):
            await self.single_probe()
            await asyncio.sleep(inter_delay_s)

        samples: List[ProbeResult] = []
        for _ in range(n):
            r = await self.single_probe()
            if r.success:
                samples.append(r)
            await asyncio.sleep(inter_delay_s)

        if not samples:
            logger.warning("Baseline calibration failed — no successful probes")
            return {"calibrated": False}

        itl_vals  = [r.mean_itl_ms for r in samples]
        ttft_vals = [r.ttft_s for r in samples]

        self.config.baseline_itl_ms  = statistics.mean(itl_vals)
        self.config.baseline_ttft_s  = statistics.mean(ttft_vals)
        self.estimator.set_baseline(self.config.baseline_itl_ms)
        self._baseline_calibrated = True

        # ── Pure ITL baseline (decode-only, TTFT excluded) ────────────────────
        # itl_values[0] = TTFT; itl_values[1:] = true inter-token intervals.
        # We keep a separate baseline for these so _kv_estimate can compare
        # apples-to-apples when computing the unsupervised KV pressure signal.
        pure_itl_means: List[float] = []
        for r in samples:
            if r.success and len(r.itl_values) > 1:
                pure_ms = [v * 1000.0 for v in r.itl_values[1:]]
                pure_itl_means.append(statistics.mean(pure_ms))
        self.config.baseline_pure_itl_ms = (
            statistics.mean(pure_itl_means) if pure_itl_means
            else self.config.baseline_itl_ms  # fallback: use full mean
        )

        # ── KV baseline: measure idle-state KV estimate ──────────────────────
        # These probes run *after* baseline_itl_ms is set, so the estimator
        # uses it for normalisation.  The resulting values represent the
        # "floor" — the non-zero KV estimate that corresponds to an idle server.
        # All subsequent single_probe() calls subtract this floor so that
        # idle → 0.0 and fully saturated → 1.0.
        kv_raw_samples: List[float] = []
        for _ in range(self.config.baseline_kv_n):
            r = await self.single_probe()
            if r.success and r.kv_usage_est >= 0:
                kv_raw_samples.append(r.kv_usage_est)
            await asyncio.sleep(inter_delay_s)

        self.config.baseline_kv_est = (
            statistics.mean(kv_raw_samples) if kv_raw_samples else 0.0
        )

        result = {
            "calibrated":            True,
            "n_samples":             len(samples),
            "baseline_itl_ms":       round(self.config.baseline_itl_ms, 3),
            "baseline_pure_itl_ms":  round(self.config.baseline_pure_itl_ms, 3),
            "baseline_ttft_s":       round(self.config.baseline_ttft_s, 4),
            "baseline_kv_est":       round(self.config.baseline_kv_est, 4),
            "itl_stddev_ms":         round(statistics.stdev(itl_vals) if len(itl_vals) > 1 else 0, 3),
            "ttft_stddev_s":         round(statistics.stdev(ttft_vals) if len(ttft_vals) > 1 else 0, 4),
        }
        logger.info("Baseline calibrated: %s", result)
        if self.config.baseline_cache_path:
            self._save_baseline(self.config.baseline_cache_path)
        if self.config.history_path:
            # Old history was measured against a different baseline — discard it.
            p = Path(self.config.history_path)
            if p.exists():
                p.write_text("")
                logger.info("Probe history cleared (new baseline) → %s", self.config.history_path)
            self._itl_history.clear()
            self._state_history.clear()
        return result

    # ═════════════════════════════════════════════════════════════════════════
    # 2. single_probe
    # ═════════════════════════════════════════════════════════════════════════

    async def single_probe(
        self,
        prompt:     Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> ProbeResult:
        """One-shot streaming probe. Measures ITL and TTFT."""
        p = prompt     or self._PROMPTS["fixed_50"]
        t = max_tokens or self.config.max_probe_tokens
        result = await self._stream_probe(p, t, probe_type="single")
        if result.success:
            result.kv_usage_est = self._kv_estimate(
                [v * 1000.0 for v in result.itl_values]
            )
            self._itl_history.append(result)
            self._append_history(result)
        return result

    # ═════════════════════════════════════════════════════════════════════════
    # 3. burst_probe
    # ═════════════════════════════════════════════════════════════════════════

    async def burst_probe(
        self,
        n:          Optional[int] = None,
        prompt:     Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> BurstProbeResult:
        """
        Fire N concurrent probe requests simultaneously.

        High TTFT variance across concurrent probes indicates HOL blocking.
        Mean TTFT >> single_probe TTFT indicates a growing wait queue.
        """
        concurrency = n      or self.config.burst_concurrency
        p           = prompt or self._PROMPTS["fixed_50"]
        t           = max_tokens or self.config.max_probe_tokens

        tasks      = [self._stream_probe(p, t, probe_type="burst") for _ in range(concurrency)]
        individual = list(await asyncio.gather(*tasks))

        burst = BurstProbeResult(n_probes=concurrency, individual=individual)
        burst.compute()

        if burst.n_success > 0:
            all_itl_ms = [
                v * 1000.0
                for r in individual if r.success
                for v in r.itl_values
            ]
            burst.kv_usage_est = self._kv_estimate(all_itl_ms)

        return burst

    # ═════════════════════════════════════════════════════════════════════════
    # 4. ttft_probe
    # ═════════════════════════════════════════════════════════════════════════

    async def ttft_probe(
        self,
        prompt:        Optional[str] = None,
        use_streaming: bool          = True,
    ) -> ProbeResult:
        """
        Minimise output tokens so TTFT dominates the measurement.
        Rising TTFT indicates a growing wait queue or prefill recompute.
        """
        p = prompt or self._PROMPTS["minimal"]
        if use_streaming:
            return await self._stream_probe(p, max_tokens=5, probe_type="ttft")
        return await self._nonstream_probe(p, max_tokens=5, probe_type="ttft")

    # ═════════════════════════════════════════════════════════════════════════
    # 5. preemption_probe
    # ═════════════════════════════════════════════════════════════════════════

    async def preemption_probe(
        self,
        duration_s:      float        = 10.0,
        prompt:          Optional[str] = None,
        max_tokens:      int           = 200,
        spike_threshold: float         = 5.0,
    ) -> dict:
        """
        Long-running streaming probe that watches for intra-request ITL spikes.

        When vLLM preempts a running request, the victim sees a multi-second
        silence between tokens before resuming. This manifests as a sudden
        ITL spike >> local mean.

        Spike criterion: itl_i > spike_threshold × mean(preceding 10 ITLs)
        """
        p               = prompt or self._PROMPTS["fixed_100"]
        result          = ProbeResult(probe_type="preemption")
        spike_events:   List[dict] = []
        window:         Deque[float] = deque(maxlen=10)

        payload = self._build_payload(p, max_tokens, stream=True)
        session = await self._get_session()
        t_start    = time.perf_counter()
        t_last     = t_start

        async def _stream() -> None:
            async with session.post(
                self.config.target.endpoint,
                json=payload,
                headers=self.config.target.auth_headers,
            ) as resp:
                if resp.status >= 400:
                    result.success = False
                    result.error   = f"HTTP {resp.status}"
                    return

                async for raw in resp.content:
                    line = raw.decode("utf-8").strip()
                    if not line or not line.startswith("data:"):
                        continue
                    data_str = line[5:].strip()
                    if data_str == "[DONE]":
                        break
                    t_now = time.perf_counter()
                    token = self._extract_token(data_str)
                    if token is not None:
                        itl = t_now - t_last
                        if not result.itl_values:
                            result.ttft_s = t_now - t_start
                        result.itl_values.append(itl)
                        local_mean = statistics.mean(window) if window else itl
                        if window and itl > spike_threshold * local_mean:
                            spike_events.append({
                                "token_index":  len(result.itl_values) - 1,
                                "itl_s":        round(itl, 4),
                                "local_mean_s": round(local_mean, 4),
                                "ratio":        round(itl / (local_mean + 1e-9), 2),
                                "elapsed_s":    round(t_now - t_start, 3),
                            })
                        window.append(itl)

        try:
            # wait_for enforces duration_s even when no tokens arrive (WAITING queue)
            await asyncio.wait_for(_stream(), timeout=duration_s)
        except asyncio.TimeoutError:
            pass   # clean deadline expiry — not a failure, just the window closing
        except Exception as exc:
            result.success, result.error = False, str(exc)

        result.total_s = time.perf_counter() - t_start
        result.compute_derived()

        return {
            "success":             result.success,
            "ttft_s":              round(result.ttft_s, 4),
            "total_s":             round(result.total_s, 4),
            "tokens_received":     result.tokens_out,
            "mean_itl_ms":         round(result.mean_itl_ms, 4),
            "itl_stddev_ms":       round(result.itl_stddev_ms, 4),
            "spike_count":         len(spike_events),
            "spikes":              spike_events,
            "preemption_detected": len(spike_events) > 0,
        }

    # ═════════════════════════════════════════════════════════════════════════
    # 6. memory_pressure_scan
    # ═════════════════════════════════════════════════════════════════════════

    async def memory_pressure_scan(
        self,
        n_probes:   int           = 10,
        interval_s: float         = 1.0,
        prompt:     Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> List[ProbeResult]:
        """
        Run `n_probes` sequential single probes spaced `interval_s` apart.

        Each probe respects ProbeConfig.probe_timeout_s — if HOL blocking is
        active and a probe receives no tokens it is marked as timed-out
        (success=False, kv_usage_est=-1.0) and the scan continues.

        Useful for mapping ITL rise during an attack, detecting recovery after
        an attack ends, or collecting (ITL, kv_usage) pairs for offline
        regressor training.
        """
        p = prompt     or self._PROMPTS["fixed_50"]
        t = max_tokens or self.config.max_probe_tokens
        results: List[ProbeResult] = []

        for i in range(n_probes):
            r = await self._stream_probe(p, t, probe_type="pressure_scan")
            if r.success:
                r.kv_usage_est = self._kv_estimate(
                    [v * 1000.0 for v in r.itl_values]
                )
            results.append(r)
            if i < n_probes - 1:
                await asyncio.sleep(interval_s)

        return results

    # ═════════════════════════════════════════════════════════════════════════
    # 7. estimate_kv_usage
    # ═════════════════════════════════════════════════════════════════════════

    def _kv_estimate(self, itl_ms: List[float]) -> float:
        """
        Compute a normalised KV pressure estimate from an ITL sequence.

        itl_ms[0] is the TTFT (time-to-first-token), which rises under HOL
        blocking (request queuing) but is *not* a direct KV pressure signal.
        itl_ms[1:] are the true inter-token intervals from the decode phase,
        which rise proportionally to GPU memory-bandwidth contention.

        We compare the mean of the decode-only intervals (itl_ms[1:]) against
        baseline_pure_itl_ms (calibrated from idle-state decode intervals) so
        that TTFT spikes from queue buildup don't inflate the KV estimate.

        If a trained ML model is available we fall back to it using the full
        sequence (it was trained with TTFT-inclusive features).

        Normalisation floor: baseline_kv_est shifts idle → 0.0.
        """
        if not itl_ms:
            return 0.0

        if self.estimator._trained and self.estimator._model is not None:
            # Supervised model: use full sequence (trained on same feature set)
            raw = self.estimator.estimate(itl_ms)
        else:
            # Unsupervised: use decode-only ITL to avoid TTFT contamination
            pure = itl_ms[1:] if len(itl_ms) > 1 else itl_ms
            bpure = max(self.config.baseline_pure_itl_ms, 1e-6)
            mean_pure = float(np.mean(pure))
            ratio = (mean_pure - bpure) / bpure
            soft  = ratio / (1.0 + abs(ratio))
            raw   = float(np.clip(soft, 0.0, 1.0))

        bkv = self.config.baseline_kv_est
        if bkv > 0.0:
            return float(np.clip(
                (raw - bkv) / max(1.0 - bkv, 1e-6),
                0.0, 1.0,
            ))
        return raw

    async def estimate_kv_usage(self) -> float:
        """Single-probe KV cache utilisation estimate: Û_sys ∈ [0.0, 1.0]."""
        r = await self.single_probe()
        return r.kv_usage_est

    # ═════════════════════════════════════════════════════════════════════════
    # 8. classify_state
    # ═════════════════════════════════════════════════════════════════════════

    def classify_state(
        self,
        probe: Optional[ProbeResult] = None,
    ) -> InfraState:
        """
        Classify current infra state from probe data + rolling ITL history.

        Priority order:
          HOL_BLOCKED — probe failed/timed out (request never left WAITING queue)
          THRASHING   — rolling CoV(ITL) > threshold (preemption loop)
          HOL_BLOCKED — TTFT >> baseline (wait queue frozen)
          SATURATED   — smoothed ITL > 3× baseline
          FILLING     — smoothed ITL > 1.5× baseline
          IDLE        — near baseline
          UNKNOWN     — no baseline calibrated
        """
        if not self._baseline_calibrated:
            return InfraState.UNKNOWN

        ref = probe
        if ref is None and self._itl_history:
            ref = self._itl_history[-1]
        if ref is None:
            return InfraState.UNKNOWN

        # Failed probe means request never reached RUNNING state → HOL blocking
        if not ref.success or ref.tokens_out == 0:
            state = InfraState.HOL_BLOCKED
            self._state_history.append(state)
            return state

        baseline_itl  = max(self.config.baseline_itl_ms, 1e-6)
        baseline_ttft = max(self.config.baseline_ttft_s, 1e-6)

        # Thrashing: rolling CoV over history window
        if len(self._itl_history) >= 3:
            rolling = [r.mean_itl_ms for r in self._itl_history if r.success]
            if len(rolling) >= 3:
                cv = statistics.stdev(rolling) / (statistics.mean(rolling) + 1e-9)
                if cv > self.config.itl_cv_thrash_threshold:
                    state = InfraState.THRASHING
                    self._state_history.append(state)
                    return state

        # Smooth ITL over last 3 successful probes to reduce single-probe noise
        recent = [r.mean_itl_ms for r in list(self._itl_history)[-3:] if r.success and r.mean_itl_ms > 0]
        smoothed_itl = statistics.mean(recent) if recent else ref.mean_itl_ms

        itl_ratio  = smoothed_itl  / baseline_itl
        ttft_ratio = ref.ttft_s    / baseline_ttft

        if   ttft_ratio > self.config.ttft_hol_threshold:
            state = InfraState.HOL_BLOCKED
        elif itl_ratio  > self.config.itl_saturated_threshold:
            state = InfraState.SATURATED
        elif itl_ratio  > self.config.itl_fill_threshold:
            state = InfraState.FILLING
        else:
            state = InfraState.IDLE

        self._state_history.append(state)
        return state

    # ═════════════════════════════════════════════════════════════════════════
    # 9. tpot_probe
    # ═════════════════════════════════════════════════════════════════════════

    async def tpot_probe(
        self,
        n_probes:   int           = 5,
        prompt:     Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> dict:
        """
        Measure TPOT (Time Per Output Token = mean ITL) across n_probes requests.

        TPOT is a direct proxy for GPU memory bandwidth contention — it rises
        proportionally to KV cache occupancy as attention becomes memory-bound.
        """
        p = prompt     or self._PROMPTS["fixed_50"]
        t = max_tokens or self.config.max_probe_tokens

        results = []
        for _ in range(n_probes):
            results.append(await self._stream_probe(p, t, probe_type="tpot"))

        good = [r for r in results if r.success and r.tpot_ms > 0]
        if not good:
            return {"success": False, "error": "no successful probes", "n": 0}

        tpots        = sorted([r.tpot_ms for r in good])
        n            = len(tpots)
        baseline_itl = max(self.config.baseline_itl_ms, 1e-6)
        mean_tpot    = statistics.mean(tpots)
        p95_idx      = min(int(math.ceil(0.95 * n)) - 1, n - 1)

        return {
            "success":         True,
            "n":               n,
            "mean_tpot_ms":    round(mean_tpot, 3),
            "p95_tpot_ms":     round(tpots[p95_idx], 3),
            "tpot_stddev_ms":  round(statistics.stdev(tpots) if n > 1 else 0.0, 3),
            "tpot_ratio":      round(mean_tpot / baseline_itl, 3),
            "baseline_itl_ms": round(baseline_itl, 3),
            "degraded":        (mean_tpot / baseline_itl) > self.config.itl_fill_threshold,
        }

    # ═════════════════════════════════════════════════════════════════════════
    # 10. request_timeout_probe
    # ═════════════════════════════════════════════════════════════════════════

    async def request_timeout_probe(
        self,
        n_probes:     int           = 10,
        hard_timeout: float         = 30.0,
        prompt:       Optional[str] = None,
        max_tokens:   int           = 100,
    ) -> dict:
        """Measure failure/timeout rate under load — direct degradation signal."""
        p = prompt or self._PROMPTS["fixed_50"]

        async def _one() -> tuple:
            try:
                r = await asyncio.wait_for(
                    self._stream_probe(p, max_tokens, probe_type="abort_probe"),
                    timeout=hard_timeout,
                )
                suspiciously_fast = r.success and r.total_s < 0.3
                return r.success, r.tokens_out, ("partial" if suspiciously_fast else None)
            except asyncio.TimeoutError:
                return False, 0, "client_timeout"
            except Exception as exc:
                return False, 0, str(exc)

        raw    = list(await asyncio.gather(*[_one() for _ in range(n_probes)]))
        n_ok   = sum(1 for r in raw if r[0])
        n_part = sum(1 for r in raw if r[2] == "partial")
        n_fail = n_probes - n_ok
        errors = [r[2] for r in raw if r[2] and r[2] != "partial"]

        return {
            "n_probes":             n_probes,
            "n_success":            n_ok,
            "n_partial":            n_part,
            "n_failed":             n_fail,
            "abort_rate":           round(n_fail  / max(n_probes, 1), 4),
            "partial_rate":         round(n_part  / max(n_probes, 1), 4),
            "error_types":          list(set(errors)),
            "scheduler_overloaded": (n_fail / max(n_probes, 1)) > 0.3,
        }

    # ═════════════════════════════════════════════════════════════════════════
    # 11. load_balancer_probe
    # ═════════════════════════════════════════════════════════════════════════

    async def load_balancer_probe(
        self,
        n_pairs:             int   = 6,
        inter_pair_delay_s:  float = 1.0,
        cache_hit_threshold: float = 0.70,
        prompt_repeats:      int   = 25,
        n_warmup:            int   = 3,
    ) -> dict:
        """
        Detect load balancer presence via prefix-cache consistency.

        Sends N pairs of identical requests in rapid succession. On a single
        server with Automatic Prefix Caching (APC) enabled, the second request
        in each pair skips prefill and shows dramatically lower TTFT (cache hit).
        A load balancer that routes the second request to a different backend
        produces no cache hit — TTFT stays similar to the first.

        Prompt sizing: _LB_CONTEXT is repeated `prompt_repeats` times (default
        25 → ~10,000 tokens) so that prefill takes 80-330 ms depending on GPU,
        making cache hits visible above the network RTT baseline.

        Warm-up: `n_warmup` discarded requests stabilise the server before
        measurement (eliminates CUDA kernel compilation and first-request memory
        allocation overhead that would otherwise inflate pair-0 cold TTFT).

        Only max_tokens=5 are generated — only TTFT matters here.

        Requires temperature=0 (default) and APC enabled on the server.

        Estimated backend count: N_backends ≈ 1 / cache_hit_rate
        """
        context_body = "\n\n".join([self._LB_CONTEXT] * prompt_repeats)

        # ── Warm-up phase ──────────────────────────────────────────────────────
        # Use different UUIDs so warm-up requests don't pre-warm the test cache.
        logger.info("load_balancer_probe: running %d warm-up requests...", n_warmup)
        for _ in range(n_warmup):
            warmup_prompt = (
                f"[warmup:{uuid.uuid4().hex}]\n\n{context_body}\n\n"
                "Question: In one word, what is this text about?"
            )
            await self._stream_probe(warmup_prompt, max_tokens=5, probe_type="lb_warmup")

        pairs = []

        for i in range(n_pairs):
            # UUID prefix makes the full prompt unique per pair. vLLM's block
            # chain-hashing propagates the UUID through all subsequent blocks,
            # so the long shared context body cannot leak across pairs.
            prompt = (
                f"[probe-id:{uuid.uuid4().hex}]\n\n{context_body}\n\n"
                "Question: In one word, what type of system is described above?"
            )

            # max_tokens=5: only need first token for TTFT measurement
            r_cold = await self._stream_probe(prompt, max_tokens=5, probe_type="lb_cold")
            r_warm = await self._stream_probe(prompt, max_tokens=5, probe_type="lb_warm")

            if r_cold.success and r_warm.success and r_cold.ttft_s > 0:
                ratio = r_warm.ttft_s / r_cold.ttft_s
                pairs.append({
                    "pair":       i,
                    "ttft_cold_s": round(r_cold.ttft_s, 4),
                    "ttft_warm_s": round(r_warm.ttft_s, 4),
                    "ratio":      round(ratio, 3),
                    "cache_hit":  ratio < cache_hit_threshold,
                })

            if i < n_pairs - 1:
                await asyncio.sleep(inter_pair_delay_s)

        if not pairs:
            return {"success": False, "error": "no successful probe pairs"}

        n_hits   = sum(1 for p in pairs if p["cache_hit"])
        hit_rate = n_hits / len(pairs)
        ratios   = [p["ratio"] for p in pairs]
        median_ratio = sorted(ratios)[len(ratios) // 2]

        # Determine if probes are consistently hitting the same backend.
        # A cache hit means the second request in a pair found its KV prefix
        # already cached — only possible when both requests land on the same
        # server with APC enabled.  High hit rate → consistent same-backend routing.
        #
        # NOTE: a single server and a prefix-aware load balancer are
        # indistinguishable here — both produce consistent cache hits for
        # same-prefix pairs, because prefix-aware routing is designed to pin
        # same-prefix traffic to the same backend.
        if hit_rate >= 0.70:
            same_backend = True
            confidence   = "high" if hit_rate >= 0.85 else "medium"
            verdict      = "single server or prefix-aware routing (cannot distinguish)"
        elif hit_rate <= 0.30:
            same_backend = False
            confidence   = "high" if hit_rate <= 0.15 else "medium"
            verdict      = "load balanced — requests hit different backends"
        else:
            same_backend = False
            confidence   = "low"
            verdict      = "ambiguous — inconsistent routing detected"

        return {
            "success":           True,
            "n_pairs":           len(pairs),
            "n_cache_hits":      n_hits,
            "cache_hit_rate":    round(hit_rate, 3),
            "median_ttft_ratio": round(median_ratio, 3),
            "same_backend":      same_backend,
            "confidence":        confidence,
            "verdict":           verdict,
            "pairs":             pairs,
        }

    # ═════════════════════════════════════════════════════════════════════════
    # 12. monitor
    # ═════════════════════════════════════════════════════════════════════════

    async def monitor(
        self,
        interval_s:      float                                               = 2.0,
        on_probe:        Optional[Callable[[ProbeResult, InfraState], None]] = None,
        on_state_change: Optional[Callable[[InfraState, InfraState], None]]  = None,
        stop_event:      Optional[asyncio.Event]                             = None,
        max_iterations:  Optional[int]                                       = None,
    ) -> None:
        """
        Continuous async polling loop for real-time infra state tracking.

        Fires single_probe() every `interval_s` seconds, classifies InfraState,
        and invokes optional callbacks:
          on_probe(result, state)      — every probe cycle
          on_state_change(old, new)    — only on InfraState transitions

        Designed to run as a background task alongside attack workers:
            stop = asyncio.Event()
            task = asyncio.create_task(probes.monitor(stop_event=stop, ...))
            # ... run attack ...
            stop.set(); await task
        """
        prev_state = InfraState.UNKNOWN
        iteration  = 0
        _stop      = stop_event or asyncio.Event()

        while not _stop.is_set():
            if max_iterations is not None and iteration >= max_iterations:
                break

            t_cycle_start = time.perf_counter()
            result = await self.single_probe()
            state  = self.classify_state(probe=result)

            if on_probe is not None:
                try:
                    ret = on_probe(result, state)
                    if asyncio.iscoroutine(ret):
                        await ret
                except Exception as exc:
                    logger.warning("monitor on_probe callback raised: %s", exc)

            if state != prev_state and on_state_change is not None:
                try:
                    ret = on_state_change(prev_state, state)
                    if asyncio.iscoroutine(ret):
                        await ret
                except Exception as exc:
                    logger.warning("monitor on_state_change callback raised: %s", exc)

            prev_state = state
            iteration += 1

            elapsed = time.perf_counter() - t_cycle_start
            sleep_s = max(0.0, interval_s - elapsed)
            try:
                await asyncio.wait_for(_stop.wait(), timeout=sleep_s)
                break
            except asyncio.TimeoutError:
                pass

    # ═════════════════════════════════════════════════════════════════════════
    # 12. rolling_summary
    # ═════════════════════════════════════════════════════════════════════════

    def rolling_summary(self) -> dict:
        """Aggregate stats over the last `rolling_window` successful probes."""
        good = [r for r in self._itl_history if r.success and r.mean_itl_ms > 0]
        if not good:
            return {"n": 0, "calibrated": self._baseline_calibrated}

        itls  = [r.mean_itl_ms for r in good]
        ttfts = [r.ttft_s      for r in good]
        kvs   = [r.kv_usage_est for r in good if r.kv_usage_est >= 0]
        n     = len(good)

        slope        = float(np.polyfit(np.arange(n), itls, 1)[0]) if n >= 2 else 0.0
        baseline_itl = max(self.config.baseline_itl_ms, 1e-6)
        degradation  = statistics.mean(itls) / baseline_itl
        s            = sorted(itls)
        p95_idx      = min(int(math.ceil(0.95 * n)) - 1, n - 1)

        return {
            "n":                      n,
            "calibrated":             self._baseline_calibrated,
            "mean_itl_ms":            round(statistics.mean(itls), 3),
            "p95_itl_ms":             round(s[p95_idx], 3),
            "itl_stddev_ms":          round(statistics.stdev(itls) if n > 1 else 0.0, 3),
            "itl_slope_ms_per_probe": round(slope, 4),
            "mean_ttft_s":            round(statistics.mean(ttfts), 4),
            "mean_kv_est":            round(statistics.mean(kvs), 4) if kvs else -1.0,
            "degradation_ratio":      round(degradation, 3),
            "current_state":          self.classify_state().value,
            "baseline_itl_ms":        round(self.config.baseline_itl_ms, 3),
        }
