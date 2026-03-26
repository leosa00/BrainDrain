"""
probes/itl_probes.py — Black-box ITL side-channel probing for LLM serving infra.

Implements all probing mechanisms from:
  "Rethinking Latency Denial-of-Service: Attacking the LLM Serving Framework,
   Not the Model" (Wang et al., 2026) — Fill & Squeeze paper.

Core insight (§5.2.2): ITL of standard requests correlates linearly with
global GPU KV-cache usage due to physical memory bandwidth contention.
This makes ITL a black-box side-channel for estimating real-time scheduler
state without privileged access, enabling adaptive Fill & Squeeze control.

Probe taxonomy:
  1.  single_probe()          — core ITL + TTFT one-shot measurement
  2.  burst_probe()           — N concurrent probes (queue depth, variance)
  3.  ttft_probe()            — TTFT-only fast probe (WAITING queue indicator)
  4.  calibrate_baseline()    — establish idle-state reference metrics
  5.  preemption_probe()      — long-running probe detecting intra-request spikes
  6.  hol_probe()             — Head-of-Line blocking detection via TTFT divergence
  7.  memory_pressure_scan()  — sequential ITL time-series (maps Fill ramp)
  8.  estimate_kv_usage()     — live Û_sys ∈ [0, 1] for adaptive control loop
  9.  classify_state()        — InfraState classifier from probe + rolling history
  10. monitor()               — continuous async polling loop with callbacks
"""

from __future__ import annotations

import asyncio
import json
import logging
import statistics
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Deque, List, Optional


import math
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
    raw_text: str = ""

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

    # KV estimate [-1 = not yet estimated]
    kv_usage_est:  float = -1.0

    success:       bool            = True
    error:         Optional[str]   = None

    def compute_derived(self) -> None:
        if not self.itl_values:
            return
        ms = [v * 1000.0 for v in self.itl_values]
        n  = len(ms)
        self.chunk_count    = n
        if n > 1 and len(self.raw_text.split()) > n:
            # Hosted API — chunks contain multiple tokens, use word estimate
            self.tokens_out = len(self.raw_text.split())
        else:
            # vLLM / Ollama — one token per chunk, chunk count is accurate
            self.tokens_out = n
        self.mean_itl_ms   = statistics.mean(ms)
        self.tpot_ms       = self.mean_itl_ms
        self.itl_stddev_ms = statistics.stdev(ms) if n > 1 else 0.0
        s = sorted(ms)
        # correct: clamp to [0, n-1], never subtract 1 before clamping
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
    timestamp:      float            = field(default_factory=time.time)
    n_probes:       int              = 0
    n_success:      int              = 0
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
        itls  = sorted([r.mean_itl_ms for r in good if r.mean_itl_ms > 0])  # exclude zero-token probes
        n_t   = len(ttfts)
        n_i   = len(itls)

        self.mean_ttft_s   = statistics.mean(ttfts)
        self.ttft_stddev_s = statistics.stdev(ttfts) if n_t > 1 else 0.0
        self.mean_itl_ms   = statistics.mean(itls)   if itls else 0.0
        self.itl_stddev_ms = statistics.stdev(itls)  if n_i > 1 else 0.0

        # ── OLD (floor − 1 picks median for small n) ──────────────────────
        # self.p95_ttft_s = st[max(int(0.95 * n) - 1, 0)]
        # self.p95_itl_ms = si[max(int(0.95 * n) - 1, 0)]

        # ── NEW (ceil then clamp — always picks the highest sample for n<20)
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

    # Probe request sizing (keep minimal to avoid self-inflicted load)
    token_budget:        int   = 512    # max_tokens for standard probe
    temperature:         float = 0.0

    # Baseline calibration
    baseline_window:     int   = 5      # cold probes for baseline
    baseline_itl_ms:     float = 0.0   # populated after calibrate_baseline()
    baseline_ttft_s:     float = 0.0

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

# ─────────────────────────────────────────────
# KV Usage Estimator
# ─────────────────────────────────────────────

class KVUsageEstimator:
    """
    Lightweight regressor: ITL feature vector → Û_sys ∈ [0, 1].

    From Fill & Squeeze §5.2.2: ITL correlates strongly with global GPU
    memory usage due to physical memory bandwidth contention, enabling a
    black-box KV cache utilisation estimate without privileged access.

    Backends:
      "lightgbm" — LGBMRegressor (requires supervised training data)
      "linear"   — Ridge regression (requires supervised training data)
      "none"     — unsupervised relative-pressure fallback (always available)

    Unsupervised fallback (black-box default):
        Û_sys = clip((mean_ITL - baseline_ITL) / baseline_ITL, 0, 1)
    """

    def __init__(self, backend: str = "linear") -> None:
        self.backend  = backend
        self._model   = None
        self._scaler  = None
        self._baseline_itl_ms: float = 0.0
        self._trained: bool = False

    # ── Feature engineering ───────────────────────────────────────────────────

    @staticmethod
    def extract_features(itl_ms: List[float]) -> np.ndarray:
        """
        Fixed-length 11-dim feature vector from variable-length ITL sequence.
        [mean, std, p50, p75, p90, p95, p99, min, max, linear_slope, CoV]
        """
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

    # ── Supervised training (optional, requires ground-truth labels) ──────────

    def fit(
        self,
        itl_sequences: List[List[float]],  # list of ITL-ms sequences
        kv_usages:     List[float],         # ground-truth Û_sys ∈ [0, 1]
    ) -> None:
        """
        Train the regressor on (ITL_sequence, kv_usage) pairs.
        Obtain ground-truth labels from privileged vLLM /metrics endpoint
        during a controlled calibration phase (white-box pre-training).
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

    # ── Inference ─────────────────────────────────────────────────────────────

    def estimate(self, itl_ms: List[float]) -> float:
        """Return Û_sys ∈ [0.0, 1.0]. Uses model if trained, else fallback."""
        if not itl_ms:
            return 0.0

        if self._trained and self._model is not None:
            feats = self.extract_features(itl_ms).reshape(1, -1)
            if self._scaler is not None:
                feats = self._scaler.transform(feats)
            return float(np.clip(self._model.predict(feats)[0], 0.0, 1.0))

        # Unsupervised: relative ITL pressure over calibrated baseline
        mean_itl = float(np.mean(itl_ms))
        if self._baseline_itl_ms <= 0:
            return 0.0

        raw = (mean_itl - self._baseline_itl_ms) / self._baseline_itl_ms
        soft = raw / (1.0 + abs(raw))
        return float(np.clip(soft, 0.0, 1.0))

# ─────────────────────────────────────────────
# Core Probe Engine
# ─────────────────────────────────────────────

class ITLProbes:
    """
    Black-box side-channel probing suite for LLM serving infrastructure.

    Designed to feed the Fill & Squeeze adaptive control loop:
        Δ_mem = C_sat − estimate_kv_usage()
        Δ_mem large  →  dispatch P_High (Fill phase)
        Δ_mem → 0    →  dispatch P_Low  (Squeeze phase)
        Δ_mem < 0    →  back-off (avoid self-preemption)

    All probes are payload-agnostic and work in pure black-box settings
    against any OpenAI-compatible, Anthropic, or Ollama endpoint.
    """

    # Deliberately innocuous probe prompts — blend with organic traffic
    _PROMPTS: dict[str, str] = {
        "minimal":   "Reply with only the single word: ok",
        "short":     "Count from 1 to 10, each number on a new line.",
        "medium":    "List 20 common English words, one per line.",
        "fixed_50":  "Write a paragraph of at least 60 words about the ocean. Do not use bullet points.",
        "fixed_100": "Write two paragraphs totaling at least 120 words about climate change.",
        "repeat_50": "List every number from 1 to 50, each on its own line.",
        "paragraph": "Write a paragraph of about 50 words on the weather.",
    }

    def __init__(self, config: ProbeConfig) -> None:
        self.config    = config
        self._session: Optional[aiohttp.ClientSession] = None
        self.estimator = KVUsageEstimator(backend=config.regressor_backend)
        self._itl_history:   Deque[ProbeResult] = deque(maxlen=config.rolling_window)
        self._state_history: Deque[InfraState]  = deque(maxlen=50)
        self._baseline_calibrated = False

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
        fmt  = self.config.target.api_format
        msgs = [{"role": "user", "content": prompt}]
        base: dict[str, Any] = {
            "model":       self.config.target.model,
            "temperature": self.config.temperature,
            "stream":      stream,
        }

        if fmt in (APIFormat.OPENAI, APIFormat.CUSTOM, APIFormat.TEST):
            base["messages"]   = msgs
            base["max_tokens"] = max_tokens
            # Only include if the endpoint explicitly supports it
            if stream and self.config.target.supports_stream_options:
                base["stream_options"] = {"include_usage": True}

        elif fmt == APIFormat.ANTHROPIC:
            base["messages"]   = msgs
            base["max_tokens"] = max_tokens

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
            return delta.get("content") or delta.get("reasoning_content")

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
        t_first: Optional[float] = None

        try:
            async with session.post(
                self.config.target.endpoint,
                json=payload,
                headers=self.config.target.auth_headers,
            ) as resp:
                if resp.status >= 400:
                    result.success = False
                    result.error   = f"HTTP {resp.status}: {(await resp.text())[:200]}"
                    return result

                buf  = b""
                done = False
                async for raw in resp.content:
                    if done:
                        break
                    buf += raw
                    while b"\n" in buf:
                        line_b, buf = buf.split(b"\n", 1)
                        line = line_b.decode("utf-8", errors="replace").strip()

                        # ── Ollama native NDJSON (no "data:" prefix) ─────────
                        # APIFormat.OLLAMA sends raw JSON lines, not SSE.
                        # CUSTOM/OPENAI pointing at Ollama's /v1 endpoint sends SSE.
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
                            if t_first is None:
                                t_first       = t_now
                                result.ttft_s = t_now - t_start
                            result.itl_values.append(t_now - t_last)
                            t_last = t_now
                            result.raw_text += token   

        except asyncio.TimeoutError:
            result.success, result.error = False, "timeout"
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
        try:
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
                    result.total_s = time.perf_counter() - t_start
                    result.ttft_s  = result.total_s  # close approximation for tiny outputs
        except asyncio.TimeoutError:
            result.success, result.error = False, "timeout"
        except Exception as exc:
            result.success, result.error = False, str(exc)
        result.total_s = time.perf_counter() - t_start
        return result

    # ═════════════════════════════════════════════════════════════════════════
    # 1. single_probe — primary side-channel measurement unit
    # ═════════════════════════════════════════════════════════════════════════

    async def single_probe(
        self,
        prompt:     Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> ProbeResult:
        """
        One-shot streaming probe. Measures ITL and TTFT.

        This is the core measurement primitive used by the Fill & Squeeze
        control loop to estimate KV cache pressure via ITL side-channel.
        """
        p = prompt     or self._PROMPTS["fixed_50"]
        t = max_tokens or self.config.token_budget
        result = await self._stream_probe(p, t, probe_type="single")
        if result.success:
            result.kv_usage_est = self.estimator.estimate(
                [v * 1000.0 for v in result.itl_values]
            )
            self._itl_history.append(result)
        return result

    # ═════════════════════════════════════════════════════════════════════════
    # 2. burst_probe — concurrency / queue depth indicator
    # ═════════════════════════════════════════════════════════════════════════

    async def burst_probe(
        self,
        n:          Optional[int] = None,
        prompt:     Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> BurstProbeResult:
        """
        Fire N concurrent probe requests simultaneously.

        Diagnostics:
          • High TTFT variance across concurrent probes → HOL blocking
            (some requests slip through, others stall at free_block_queue check)
          • Mean TTFT >> single_probe TTFT → WAITING queue growing
          • High ITL CoV across requests → preemption evicting some runners
        """
        concurrency = n      or self.config.burst_concurrency
        p           = prompt or self._PROMPTS["fixed_50"]
        t           = max_tokens or self.config.token_budget

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
            burst.kv_usage_est = self.estimator.estimate(all_itl_ms)

        return burst

    # ═════════════════════════════════════════════════════════════════════════
    # 3. ttft_probe — WAITING queue depth indicator
    # ═════════════════════════════════════════════════════════════════════════

    async def ttft_probe(
        self,
        prompt:         Optional[str] = None,
        use_streaming:  bool          = True,
    ) -> ProbeResult:
        """
        Minimise output tokens so TTFT dominates the measurement.

        TTFT = prefill_time + queue_wait_time.
        A rising TTFT baseline indicates either:
          (a) growing WAITING queue (approaching HOL blocking), or
          (b) the system is preempting and recomputing prefills (vLLM v1).
        """
        p = prompt or self._PROMPTS["minimal"]
        if use_streaming:
            return await self._stream_probe(p, max_tokens=5, probe_type="ttft")
        return await self._nonstream_probe(p, max_tokens=5, probe_type="ttft")

    # ═════════════════════════════════════════════════════════════════════════
    # 4. calibrate_baseline — idle reference
    # ═════════════════════════════════════════════════════════════════════════

    async def calibrate_baseline(
        self,
        n_samples:      Optional[int] = None,
        warm_up:        int           = 1,
        inter_delay_s:  float         = 0.5,
    ) -> dict:
        """
        Establish baseline ITL and TTFT under (approximately) idle conditions.

        Should be called before any attack phase. Runs `warm_up` discarded
        probes to avoid cold-start KV cache / prefix cache effects, then
        collects `n_samples` measurements.

        Populates:
          config.baseline_itl_ms / config.baseline_ttft_s
          estimator._baseline_itl_ms
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

        result = {
            "calibrated":       True,
            "n_samples":        len(samples),
            "baseline_itl_ms":  round(self.config.baseline_itl_ms, 3),
            "baseline_ttft_s":  round(self.config.baseline_ttft_s, 4),
            "itl_stddev_ms":    round(statistics.stdev(itl_vals) if len(itl_vals) > 1 else 0, 3),
            "ttft_stddev_s":    round(statistics.stdev(ttft_vals) if len(ttft_vals) > 1 else 0, 4),
        }
        logger.info("Baseline calibrated: %s", result)
        return result

    # ═════════════════════════════════════════════════════════════════════════
    # 5. preemption_probe — intra-request ITL spike detection
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

        Preemption signature (vLLM scheduler):
          When the scheduler invokes _preempt() on a running request, the
          victim experiences a multi-second silence between tokens before
          resuming (swap I/O in v0, full recompute in v1). This manifests
          as a sudden ITL spike >> local mean.

        Spike criterion: itl_i > spike_threshold × mean(preceding 10 ITLs)

        Returns spike timestamps, count, severity, and preemption_detected flag.
        """
        p               = prompt or self._PROMPTS["fixed_100"]
        result          = ProbeResult(probe_type="preemption")
        spike_events:   List[dict] = []
        window:         Deque[float] = deque(maxlen=10)

        payload = self._build_payload(p, max_tokens, stream=True)
        session = await self._get_session()
        t_start   = time.perf_counter()
        t_deadline = t_start + duration_s
        t_last    = t_start
        t_first: Optional[float] = None

        try:
            async with session.post(
                self.config.target.endpoint,
                json=payload,
                headers=self.config.target.auth_headers,
            ) as resp:
                if resp.status >= 400:
                    return {"success": False, "error": f"HTTP {resp.status}"}

                async for raw in resp.content:
                    if time.perf_counter() > t_deadline:
                        break
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
                        if t_first is None:
                            t_first, result.ttft_s = t_now, t_now - t_start
                        result.itl_values.append(itl)
                        local_mean = statistics.mean(window) if window else itl
                        if window and itl > spike_threshold * local_mean:
                            spike_events.append({
                                "token_index": len(result.itl_values) - 1,
                                "itl_s":       round(itl, 4),
                                "local_mean_s": round(local_mean, 4),
                                "ratio":       round(itl / (local_mean + 1e-9), 2),
                                "elapsed_s":   round(t_now - t_start, 3),
                            })
                        window.append(itl)
                        t_last = t_now

        except asyncio.TimeoutError:
            result.success, result.error = False, "timeout"
        except Exception as exc:
            result.success, result.error = False, str(exc)

        result.total_s = time.perf_counter() - t_start
        result.compute_derived()

        return {
            "success":              result.success,
            "ttft_s":               round(result.ttft_s, 4),
            "total_s":              round(result.total_s, 4),
            "tokens_received":      result.tokens_out,
            "mean_itl_ms":          round(result.mean_itl_ms, 4),
            "itl_stddev_ms":        round(result.itl_stddev_ms, 4),
            "spike_count":          len(spike_events),
            "spikes":               spike_events,
            "preemption_detected":  len(spike_events) > 0,
        }

    # ═════════════════════════════════════════════════════════════════════════
    # 6. hol_probe — Head-of-Line blocking detection
    # ═════════════════════════════════════════════════════════════════════════

    async def hol_probe(
        self,
        n_fill:      int           = 4,
        long_prompt: Optional[str] = None,
        short_prompt: Optional[str] = None,
    ) -> dict:
        """
        Detect HOL blocking by comparing short-request TTFT under memory pressure.

        Method (from Fill & Squeeze §5.1):
          1. Send `n_fill` long-output requests concurrently (fill KV pressure)
          2. Immediately send one minimal short probe (50 ms stagger)
          3. HOL detected if short_ttft >> baseline_ttft

        When free_block_queue is exhausted and can_allocate() returns False,
        the FCFS WAITING queue freezes: the short probe's TTFT explodes even
        though computational slots are still available.

        HOL indicator: ttft_ratio = short_ttft / baseline_ttft > hol_threshold
        """
        long_p  = long_prompt  or self._PROMPTS["fixed_100"]
        short_p = short_prompt or self._PROMPTS["minimal"]

        fill_tasks  = [
            self._stream_probe(long_p, 150, probe_type="hol_fill")
            for _ in range(n_fill)
        ]
        # Small stagger so fill requests enter the scheduler first
        fill_coro   = asyncio.gather(*fill_tasks)
        await asyncio.sleep(0.05)
        short_result, fill_results = await asyncio.gather(
            self._stream_probe(short_p, 5, probe_type="hol_short"),
            fill_coro,
        )

        baseline   = max(self.config.baseline_ttft_s, 1e-6)
        short_ttft = short_result.ttft_s if short_result.success else -1.0
        ratio      = short_ttft / baseline if short_ttft > 0 else -1.0

        fill_ttfts = [r.ttft_s for r in fill_results if r.success]
        return {
            "hol_detected":      ratio > self.config.ttft_hol_threshold,
            "ttft_ratio":        round(ratio, 3),
            "short_ttft_s":      round(short_ttft, 4),
            "baseline_ttft_s":   round(baseline, 4),
            "threshold":         self.config.ttft_hol_threshold,
            "fill_n":            len(fill_results),
            "fill_mean_ttft_s":  round(statistics.mean(fill_ttfts), 4) if fill_ttfts else -1.0,
        }

    # ═════════════════════════════════════════════════════════════════════════
    # 7. memory_pressure_scan — ITL time-series (Fill ramp mapping)
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

        Use cases:
          • Map ITL rise during Fill phase to identify saturation point
          • Detect recovery (ITL decay post-attack)
          • Collect (ITL, kv_usage) pairs for offline regressor training
          • Validate that an attack is maintaining memory pressure
        """
        p = prompt     or self._PROMPTS["fixed_50"]
        t = max_tokens or self.config.token_budget
        results: List[ProbeResult] = []

        for i in range(n_probes):
            r = await self._stream_probe(p, t, probe_type="pressure_scan")
            if r.success:
                r.kv_usage_est = self.estimator.estimate(
                    [v * 1000.0 for v in r.itl_values]
                )
            results.append(r)
            if i < n_probes - 1:
                await asyncio.sleep(interval_s)

        return results

    # ═════════════════════════════════════════════════════════════════════════
    # 8. estimate_kv_usage — live Û_sys for adaptive control
    # ═════════════════════════════════════════════════════════════════════════

    async def estimate_kv_usage(self) -> float:
        """
        Single-probe KV cache utilisation estimate: Û_sys ∈ [0.0, 1.0].

        Primary feedback signal for the Fill & Squeeze control loop:
            Δ_mem = C_sat - estimate_kv_usage()
            Δ_mem large → dispatch P_High  (Fill phase: rapid VRAM occupation)
            Δ_mem → 0   → dispatch P_Low   (Squeeze phase: trigger preemption)
            Δ_mem < 0   → back-off         (avoid self-preemption)
        """
        r = await self.single_probe()
        return r.kv_usage_est

    # ═════════════════════════════════════════════════════════════════════════
    # 9. classify_state — InfraState classifier
    # ═════════════════════════════════════════════════════════════════════════

    def classify_state(
        self,
        probe: Optional[ProbeResult]      = None,
        burst: Optional[BurstProbeResult] = None,
    ) -> InfraState:
        """
        Classify current infra state from probe data + rolling ITL history.

        Decision tree (priority order):
          THRASHING   — rolling CoV(ITL) > threshold (preemption loop)
          HOL_BLOCKED — TTFT >> baseline (WAITING queue frozen)
          SATURATED   — mean ITL > 3× baseline (KV near 100%)
          FILLING     — mean ITL > 1.5× baseline (KV rising)
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

        baseline_itl  = self.config.baseline_itl_ms
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

        itl_ratio  = ref.mean_itl_ms  / (baseline_itl  + 1e-9)
        ttft_ratio = ref.ttft_s        / baseline_ttft

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
    # 10. monitor — continuous async polling loop
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
        and invokes optional async-or-sync callbacks:
        on_probe(result, state)      — every probe cycle
        on_state_change(old, new)    — only on InfraState transitions

        Designed to run as a background asyncio.Task alongside attack workers:

            stop = asyncio.Event()
            task = asyncio.create_task(probes.monitor(
                interval_s=2.0,
                on_state_change=lambda old, new: dispatch_fill_squeeze(new),
                stop_event=stop,
            ))
            # ... run attack ...
            stop.set()
            await task

        Parameters
        ──────────
        interval_s      — seconds between probe cycles
        on_probe        — callback(ProbeResult, InfraState) every cycle
        on_state_change — callback(old_state, new_state) on transitions only
        stop_event      — asyncio.Event; set() to terminate the loop cleanly
        max_iterations  — hard cap on probe count (None = run forever)
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

            # ── on_probe: fires every cycle ───────────────────────────────────
            if on_probe is not None:
                try:
                    ret = on_probe(result, state)
                    if asyncio.iscoroutine(ret):
                        await ret
                except Exception as exc:
                    logger.warning("monitor on_probe callback raised: %s", exc)

            # ── on_state_change: fires only on InfraState transitions ─────────
            if state != prev_state and on_state_change is not None:
                try:
                    ret = on_state_change(prev_state, state)
                    if asyncio.iscoroutine(ret):
                        await ret
                except Exception as exc:
                    logger.warning("monitor on_state_change callback raised: %s", exc)

            prev_state = state
            iteration += 1

            # ── Interval sleep — wakes early if stop_event is set ────────────
            elapsed = time.perf_counter() - t_cycle_start
            sleep_s = max(0.0, interval_s - elapsed)
            try:
                await asyncio.wait_for(_stop.wait(), timeout=sleep_s)
                break   # stop_event fired during sleep → clean exit
            except asyncio.TimeoutError:
                pass    # normal path: interval elapsed, continue loop


    async def tpot_probe(
        self,
        n_probes:   int           = 5,
        prompt:     Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> dict:
        """
        Measure TPOT (Time Per Output Token = mean ITL) across n_probes requests.

        TPOT isolates decode-phase throughput from queuing effects and is a
        direct proxy for GPU memory bandwidth contention.  Under Fill & Squeeze
        saturation it rises proportionally to KV cache occupancy because the
        attention matmuls become increasingly memory-bound.
        """
        p = prompt     or self._PROMPTS["fixed_50"]
        t = max_tokens or self.config.token_budget

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
        tpot_ratio   = mean_tpot / baseline_itl


        p95_idx = min(int(math.ceil(0.95 * n)) - 1, n - 1)

        return {
            "success":         True,
            "n":               n,
            "mean_tpot_ms":    round(mean_tpot, 3),
            "p95_tpot_ms":     round(tpots[p95_idx], 3),
            "tpot_stddev_ms":  round(statistics.stdev(tpots) if n > 1 else 0.0, 3),
            "tpot_ratio":      round(tpot_ratio, 3),
            "baseline_itl_ms": round(baseline_itl, 3),
            "degraded":        tpot_ratio > self.config.itl_fill_threshold,
        }


    async def request_timeout_probe(
        self,
        n_probes:     int           = 10,
        hard_timeout: float         = 30.0,
        prompt:       Optional[str] = None,
        max_tokens:   int           = 100,
    ) -> dict:
        p = prompt or self._PROMPTS["fixed_50"]

        async def _one() -> tuple:
            try:
                r = await asyncio.wait_for(
                    self._stream_probe(p, max_tokens, probe_type="abort_probe"),
                    timeout=hard_timeout,
                )
                # ── partial = completed suspiciously fast with zero content
                # Do NOT use chunk count — hosted APIs batch chunks arbitrarily
                suspiciously_fast = r.success and r.total_s < 0.3
                return r.success, r.tokens_out, ("partial" if suspiciously_fast else None)
            except asyncio.TimeoutError:
                return False, 0, "client_timeout"
            except Exception as exc:
                return False, 0, str(exc)

        raw    = list(await asyncio.gather(*[_one() for _ in range(n_probes)]))
        n_ok   = sum(1 for ok, _, __ in raw if ok)
        n_part = sum(1 for _, __, e  in raw if e == "partial")
        n_fail = n_probes - n_ok
        errors = [e for _, __, e in raw if e and e != "partial"]

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

    async def adaptive_fill_signal(
        self,
        c_sat: float = 0.90,
    ) -> dict:
        """
        Combined measurement + decision primitive for the Fill & Squeeze loop.

            action = "fill"    → Δ_mem large; dispatch high-load requests
            action = "squeeze" → Δ_mem ≈ 0;   dispatch long-output requests
            action = "backoff" → Δ_mem < 0;   reduce request rate
            action = "wait"    → baseline not calibrated or probe failed
        """
        result = await self.single_probe()
        if not result.success or not self._baseline_calibrated:
            return {
                "action":       "wait",
                "kv_usage_est": -1.0,
                "delta_mem":    -1.0,
                "state":        InfraState.UNKNOWN.value,
                "probe":        result.to_dict(),
            }

        kv_est = result.kv_usage_est
        delta  = c_sat - kv_est
        state  = self.classify_state(probe=result)

        if   state in (InfraState.THRASHING, InfraState.DEGRADED):
            action = "backoff"
        elif delta < 0:
            action = "backoff"
        elif delta < 0.05:
            action = "squeeze"
        elif state == InfraState.HOL_BLOCKED:
            action = "squeeze"
        else:
            action = "fill"

        return {
            "action":       action,
            "kv_usage_est": round(kv_est, 4),
            "delta_mem":    round(delta, 4),
            "c_sat":        c_sat,
            "state":        state.value,
            "probe":        result.to_dict(),
        }

    def rolling_summary(self) -> dict:
        """Aggregate stats over the last `rolling_window` successful probes."""
        good = [r for r in self._itl_history if r.success and r.mean_itl_ms > 0]
        if not good:
            return {"n": 0, "calibrated": self._baseline_calibrated}

        itls  = [r.mean_itl_ms for r in good]
        ttfts = [r.ttft_s      for r in good]
        kvs   = [r.kv_usage_est for r in good if r.kv_usage_est >= 0]
        n     = len(good)

        slope = float(np.polyfit(np.arange(n), itls, 1)[0]) if n >= 2 else 0.0

        baseline_itl = max(self.config.baseline_itl_ms, 1e-6)
        degradation  = statistics.mean(itls) / baseline_itl

        s = sorted(itls)
        p95_idx = min(int(math.ceil(0.95 * n)) - 1, n - 1)

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