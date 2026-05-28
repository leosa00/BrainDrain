"""
orchestration/result_collector.py

Thread-safe async result aggregation for DDoS runs.
ResultCollector runs as a background asyncio.Task, draining
AttackResults from a Queue and computing aggregate metrics on finalise().
"""
from __future__ import annotations

import asyncio
import math
import statistics
import time
from dataclasses import dataclass, field
from typing import List, Optional

from core.base_attack import AttackResult, AttackStatus
from probes.itl_probes import InfraState


# ─────────────────────────────────────────────
# Run Result
# ─────────────────────────────────────────────

@dataclass
class DDoSRunResult:
    """
    Aggregated output of a complete DDoS run.

    Sections:
    - Timing:        wall clock and per-request duration percentiles
    - Counts:        request outcomes
    - Token metrics: total tokens generated and amplification ratio
    - Throughput:    requests/s and tokens/s from the attacker's side
    - State:         InfraState timeline recorded by ITLProbes.monitor()
    """

    # ── Timing ───────────────────────────────
    wall_clock_s: float = 0.0
    start_time: float = field(default_factory=time.time)

    # ── Counts ───────────────────────────────
    total_requests: int = 0
    successful: int = 0
    failed: int = 0
    timed_out: int = 0
    cancelled: int = 0

    # ── Token metrics ─────────────────────────
    mean_amplification_ratio: float = 0.0
    p95_amplification_ratio: float = 0.0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_reasoning_tokens: int = 0

    # ── Attacker-side latency ─────────────────
    # (how long each request held open a connection)
    mean_total_duration_s: float = 0.0
    p95_total_duration_s: float = 0.0
    mean_ttft_s: float = 0.0
    p95_ttft_s: float = 0.0

    # ── Throughput ───────────────────────────
    requests_per_second: float = 0.0
    tokens_generated_per_second: float = 0.0

    # ── InfraState timeline ──────────────────
    # List of (wall_clock_offset_s, state_value_str) tuples
    # recorded by the probes.monitor() callback.
    state_timeline: List[tuple] = field(default_factory=list)

    # ── Raw results ──────────────────────────
    all_results: List[AttackResult] = field(default_factory=list)

    # ── Per-instance stats ────────────────────
    instance_stats: List[dict] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        return self.successful / max(self.total_requests, 1)

    @property
    def total_generated_tokens(self) -> int:
        return self.total_completion_tokens

    def to_dict(self) -> dict:
        return {
            "wall_clock_s": round(self.wall_clock_s, 2),
            "total_requests": self.total_requests,
            "successful": self.successful,
            "failed": self.failed,
            "timed_out": self.timed_out,
            "success_rate": round(self.success_rate, 4),
            "mean_amplification_ratio": round(self.mean_amplification_ratio, 2),
            "p95_amplification_ratio": round(self.p95_amplification_ratio, 2),
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_reasoning_tokens": self.total_reasoning_tokens,
            "total_generated_tokens": self.total_generated_tokens,
            "mean_total_duration_s": round(self.mean_total_duration_s, 3),
            "p95_total_duration_s": round(self.p95_total_duration_s, 3),
            "mean_ttft_s": round(self.mean_ttft_s, 3),
            "p95_ttft_s": round(self.p95_ttft_s, 3),
            "requests_per_second": round(self.requests_per_second, 4),
            "tokens_generated_per_second": round(self.tokens_generated_per_second, 2),
            "state_timeline": self.state_timeline,
            "instance_stats": self.instance_stats,
        }

    def summary(self) -> str:
        lines = [
            f"{'─' * 52}",
            f"  DDoS Run Summary — {self.wall_clock_s:.1f}s wall clock",
            f"{'─' * 52}",
            f"  Requests : {self.total_requests} total | "
            f"{self.successful} ok / {self.timed_out} timeout / {self.failed} failed",
            f"  Amp ratio: {self.mean_amplification_ratio:.1f}× mean | "
            f"{self.p95_amplification_ratio:.1f}× p95",
            f"  Tokens   : {self.total_generated_tokens:,} generated "
            f"({self.tokens_generated_per_second:.0f} tok/s)",
            f"  Duration : {self.mean_total_duration_s:.1f}s mean | "
            f"{self.p95_total_duration_s:.1f}s p95",
            f"  TTFT     : {self.mean_ttft_s:.2f}s mean | "
            f"{self.p95_ttft_s:.2f}s p95",
        ]
        if self.state_timeline:
            last_state = self.state_timeline[-1][1]
            n_transitions = len(set(s for _, s in self.state_timeline))
            lines.append(
                f"  Infra    : {n_transitions} distinct states | "
                f"final={last_state}"
            )
        lines.append(f"{'─' * 52}")
        return "\n".join(lines)


# ─────────────────────────────────────────────
# Collector
# ─────────────────────────────────────────────

class ResultCollector:
    """
    Async background consumer for AttackResult objects.

    Usage:
        collector = ResultCollector()
        stop = asyncio.Event()
        task = asyncio.create_task(collector.consume(queue, stop))

        # ... run attack ...
        stop.set()
        await task
        result = collector.finalise()

    record_state() is called by the orchestrator's on_state_change
    callback to build the InfraState timeline alongside result data.
    """

    def __init__(self) -> None:
        self._results: List[AttackResult] = []
        self._start_time: float = time.time()
        self._state_timeline: List[tuple] = []

    def record_state(self, state: InfraState) -> None:
        """Append (wall_clock_offset_s, state) to timeline. Call from on_state_change."""
        offset = round(time.time() - self._start_time, 2)
        self._state_timeline.append((offset, state.value))

    async def consume(
        self,
        result_queue: asyncio.Queue,
        stop_event: asyncio.Event,
    ) -> None:
        """
        Drain result_queue continuously.

        Does NOT exit immediately when stop_event fires — it keeps
        draining until the queue is empty so no results are lost
        when the orchestrator shuts down.
        """
        while not stop_event.is_set() or not result_queue.empty():
            try:
                result: AttackResult = await asyncio.wait_for(
                    result_queue.get(), timeout=0.25
                )
                self._results.append(result)
                result_queue.task_done()
            except asyncio.TimeoutError:
                continue  # poll again, check stop_event

    def finalise(
        self,
        instance_stats: Optional[List[dict]] = None,
    ) -> DDoSRunResult:
        """
        Compute aggregate metrics over all collected results and return
        a sealed DDoSRunResult. Safe to call after consume() completes.
        """
        r = DDoSRunResult()
        r.start_time = self._start_time
        r.wall_clock_s = time.time() - self._start_time
        r.all_results = list(self._results)
        r.state_timeline = list(self._state_timeline)
        r.instance_stats = instance_stats or []

        if not self._results:
            return r

        r.total_requests = len(self._results)
        r.successful = sum(
            1 for x in self._results if x.status == AttackStatus.SUCCESS
        )
        r.timed_out = sum(
            1 for x in self._results if x.status == AttackStatus.TIMEOUT
        )
        r.cancelled = sum(
            1 for x in self._results if x.status == AttackStatus.CANCELLED
        )
        r.failed = r.total_requests - r.successful - r.timed_out - r.cancelled

        # Token metrics — all requests (including failed) may have partial data
        r.total_prompt_tokens = sum(
            x.token_metrics.prompt_tokens for x in self._results
        )
        r.total_completion_tokens = sum(
            x.token_metrics.completion_tokens for x in self._results
        )
        r.total_reasoning_tokens = sum(
            x.token_metrics.reasoning_tokens for x in self._results
        )

        # Amplification — only requests with non-zero prompt tokens
        ratios = sorted(
            x.token_metrics.amplification_ratio
            for x in self._results
            if x.token_metrics.prompt_tokens > 0
        )
        if ratios:
            n_r = len(ratios)
            r.mean_amplification_ratio = statistics.mean(ratios)
            r.p95_amplification_ratio = ratios[
                min(math.ceil(0.95 * n_r) - 1, n_r - 1)
            ]

        # Latency — successful requests only
        good = [x for x in self._results if x.status == AttackStatus.SUCCESS]
        if good:
            durations = sorted(x.latency_metrics.total_duration_s for x in good)
            ttfts = sorted(
                x.latency_metrics.ttft_s
                for x in good
                if x.latency_metrics.ttft_s > 0
            )
            n_d, n_t = len(durations), len(ttfts)
            r.mean_total_duration_s = statistics.mean(durations)
            r.p95_total_duration_s = durations[
                min(math.ceil(0.95 * n_d) - 1, n_d - 1)
            ]
            if ttfts:
                r.mean_ttft_s = statistics.mean(ttfts)
                r.p95_ttft_s = ttfts[min(math.ceil(0.95 * n_t) - 1, n_t - 1)]

        # Throughput
        total_gen = r.total_completion_tokens + r.total_reasoning_tokens
        elapsed = max(r.wall_clock_s, 1e-6)
        r.requests_per_second = r.total_requests / elapsed
        r.tokens_generated_per_second = total_gen / elapsed

        return r