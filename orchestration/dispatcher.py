"""
orchestration/dispatcher.py

Semaphore-based attack dispatcher: maintains exactly N in-flight requests
at all times.  When any request completes (any status), its slot is
refilled immediately — no 2 s backoff between retries.
"""
from __future__ import annotations

import asyncio
import logging
import time

from core.base_attack import AttackStatus
from orchestration.attacker_instance import AttackerInstance

logger = logging.getLogger(__name__)


class AttackDispatcher:
    """
    Keeps exactly ``len(instances)`` concurrent in-flight requests.

    Each slot is backed by a dedicated :class:`AttackerInstance` so
    per-instance stats (runs / successes / timeouts) are preserved in the
    final JSON output.

    Behavioural differences vs the legacy ``run_loop`` approach:

    * Uses ``asyncio.wait(FIRST_COMPLETED)`` — a replacement fires the
      instant any request finishes, not on the next loop iteration.
    * Connection-level failures get a 0.1 s guard instead of 2 s, so
      slots refill fast when the server recovers from saturation.
    * The initial fill is staggered by ``stagger_s`` to phase completions.
    """

    CONN_RETRY_DELAY_S = 0.1

    # A streaming request that has produced no token for this many seconds is
    # treated as "stalled" — almost always a server-side preemption/swap under
    # KV pressure.  Generation ITL is tens of ms, so seconds of silence is a
    # clear signal, not normal jitter.
    STALL_THRESHOLD_S = 5.0

    def __init__(
        self,
        instances: list[AttackerInstance],
        result_queue: asyncio.Queue,
        stop_event: asyncio.Event,
        stagger_s: float = 0.0,
    ) -> None:
        self._instances  = instances
        self._queue      = result_queue
        self._stop       = stop_event
        self._stagger_s  = stagger_s
        # Maps asyncio.Task → AttackerInstance for currently in-flight requests.
        self._in_flight: dict[asyncio.Task, AttackerInstance] = {}

    # ── Observability ──────────────────────────────────────────────────────

    @property
    def active_count(self) -> int:
        """Number of requests with an open HTTP connection right now."""
        return len(self._in_flight)

    @property
    def phase_breakdown(self) -> dict[str, int]:
        """Counts of in-flight requests by lifecycle phase.

        Lets the caller distinguish requests the server is actively generating
        for ('streaming') from those that are connected but receiving no tokens.
        Phases that mean "dispatched but not actually running on the server":
          * ``connecting``     — not yet accepted by the server
          * ``awaiting_ttft``  — accepted but no first token (queued / not admitted)
          * ``stalled``        — was streaming, now silent (preempted / swapped)
        ``starting`` covers the brief window before the attack object exists.
        """
        now = time.perf_counter()
        counts: dict[str, int] = {}
        for inst in self._in_flight.values():
            atk = inst._current_attack
            if atk is None:
                phase = "starting"
            else:
                phase = getattr(atk, "phase", None) or "starting"
                last = getattr(atk, "_last_token_t", None)
                if (
                    phase == "streaming"
                    and last is not None
                    and (now - last) > self.STALL_THRESHOLD_S
                ):
                    phase = "stalled"
            counts[phase] = counts.get(phase, 0) + 1
        return counts

    @property
    def instances(self) -> list[AttackerInstance]:
        return self._instances

    # ── Internal ───────────────────────────────────────────────────────────

    def _launch(self, inst: AttackerInstance) -> None:
        task = asyncio.create_task(inst.run_once())
        self._in_flight[task] = inst

    # ── Main loop ──────────────────────────────────────────────────────────

    async def run(self) -> None:
        # Staggered initial fill.
        for i, inst in enumerate(self._instances):
            if self._stop.is_set():
                break
            if self._stagger_s > 0 and i > 0:
                await asyncio.sleep(self._stagger_s)
            self._launch(inst)

        try:
            while not self._stop.is_set():
                if not self._in_flight:
                    await asyncio.sleep(0.05)
                    continue

                done, _ = await asyncio.wait(
                    list(self._in_flight.keys()),
                    return_when=asyncio.FIRST_COMPLETED,
                    timeout=0.5,
                )

                freed: list[AttackerInstance] = []
                conn_fail = False

                for task in done:
                    inst = self._in_flight.pop(task)

                    if task.cancelled():
                        freed.append(inst)
                        continue

                    exc = task.exception()
                    if exc is not None:
                        logger.warning(
                            "[%s] run_once raised: %s", inst.instance_id, exc
                        )
                        freed.append(inst)
                        continue

                    result = task.result()
                    if result is not None:
                        await self._queue.put(result)
                        if (
                            result.status == AttackStatus.FAILED
                            and result.error
                            and not result.error.startswith("HTTP")
                        ):
                            conn_fail = True
                    freed.append(inst)

                # Brief guard against busy-spinning when the server is refusing
                # all new connections (every slot fails instantly).
                if conn_fail:
                    await asyncio.sleep(self.CONN_RETRY_DELAY_S)

                # Immediately refill every slot that just freed up.
                if not self._stop.is_set():
                    for inst in freed:
                        self._launch(inst)

        except asyncio.CancelledError:
            pass

        finally:
            # Cancel all in-flight connections and collect partial results.
            for task in list(self._in_flight.keys()):
                task.cancel()
            if self._in_flight:
                await asyncio.gather(
                    *self._in_flight.keys(), return_exceptions=True
                )
            for inst in self._instances:
                if inst._cancelled_result is not None:
                    await self._queue.put(inst._cancelled_result)
                    inst._cancelled_result = None
