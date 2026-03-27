"""
orchestration/attacker_instance.py

Single attacker lifecycle wrapper around BaseAttack.
Handles config factory pattern (fresh request_id per run), retries,
and both one-shot and continuous loop execution modes.
"""
from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from typing import Callable, Optional, Type

from core.base_attack import AttackConfig, AttackResult, AttackStatus, BaseAttack

logger = logging.getLogger(__name__)


@dataclass
class AttackerInstanceConfig:
    """
    Defines a single attacker: which attack class to use and how to
    produce a fresh AttackConfig on every run.

    attack_config_factory must return a new AttackConfig each call —
    this ensures every request gets a unique request_id and, for
    ReasoningBombAttack, a randomly selected puzzle (when puzzle_index=None).

    attack_extra_kwargs are passed as **kwargs to the attack class
    constructor alongside AttackConfig, e.g.:
        attack_cls=ReasoningBombAttack,
        attack_extra_kwargs={"rb_config": ReasoningBombConfig(budget_tier="128")}
    """
    attack_cls: Type[BaseAttack]
    attack_config_factory: Callable[[], AttackConfig]
    attack_extra_kwargs: dict = field(default_factory=dict)
    instance_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    # Retry only on FAILED — never on TIMEOUT (timeouts are a DoS success signal)
    max_retries: int = 0
    retry_delay_s: float = 1.0

    # Optional stagger before the very first request (used by wave launch)
    launch_delay_s: float = 0.0


class AttackerInstance:
    """
    Lifecycle wrapper around a single BaseAttack subclass.

    Two execution modes:
    - run_once()  → fires one request, returns AttackResult
    - run_loop()  → continuously re-fires until stop_event is set,
                    pushing each AttackResult into result_queue
    """

    def __init__(self, cfg: AttackerInstanceConfig) -> None:
        self.cfg = cfg
        self.instance_id = cfg.instance_id
        self._runs: int = 0
        self._successes: int = 0
        self._failures: int = 0
        self._timeouts: int = 0
        self._current_attack: Optional[BaseAttack] = None

    # ── Core execution ────────────────────────────────────────────────────

    async def run_once(self) -> AttackResult:
        """
        Build a fresh attack from the factory, execute it, return result.

        Retries on FAILED up to max_retries times.
        TIMEOUT results are returned immediately — a timed-out request is
        holding KV blocks on the server, which is exactly what we want.
        """
        config = self.cfg.attack_config_factory()
        attack = self.cfg.attack_cls(config, **self.cfg.attack_extra_kwargs)
        self._current_attack = attack

        result: Optional[AttackResult] = None
        attempt = 0

        while attempt <= self.cfg.max_retries:
            try:
                result = await attack.run()
            except asyncio.CancelledError:
                await attack.close()
                self._current_attack = None
                raise
            except Exception as exc:
                logger.warning(
                    "[%s] run_once unhandled exception (attempt %d): %s",
                    self.instance_id, attempt, exc,
                )
                result = AttackResult(
                    request_id=config.request_id,
                    attack_name=attack.attack_name,
                    status=AttackStatus.FAILED,
                    error=str(exc),
                )

            if result.status in (AttackStatus.SUCCESS, AttackStatus.TIMEOUT):
                break

            attempt += 1
            if attempt <= self.cfg.max_retries:
                await asyncio.sleep(self.cfg.retry_delay_s)

        # Bookkeeping
        self._runs += 1
        if result.status == AttackStatus.SUCCESS:
            self._successes += 1
        elif result.status == AttackStatus.TIMEOUT:
            self._timeouts += 1
        else:
            self._failures += 1

        await attack.close()
        self._current_attack = None
        return result

    async def run_loop(
        self,
        result_queue: asyncio.Queue,
        stop_event: asyncio.Event,
        inter_request_delay_s: float = 0.0,
    ) -> None:
        """
        Sustained mode: re-fire immediately after each completion.

        This is the key primitive for LLM DoS — a new fill request
        replaces the completing one before the scheduler can free
        blocks to legitimate traffic.

        inter_request_delay_s can introduce a small gap between
        requests if needed (e.g. to avoid self-preemption during
        the THRASHING backoff phase).
        """
        if self.cfg.launch_delay_s > 0:
            await asyncio.sleep(self.cfg.launch_delay_s)

        while not stop_event.is_set():
            try:
                result = await self.run_once()
                await result_queue.put(result)

                if inter_request_delay_s > 0 and not stop_event.is_set():
                    await asyncio.sleep(inter_request_delay_s)

            except asyncio.CancelledError:
                raise
            except Exception as exc:
                # Should not happen — run_once catches internally, but
                # guard here to keep the loop alive on unexpected errors.
                logger.error("[%s] run_loop unhandled: %s", self.instance_id, exc)
                await asyncio.sleep(0.5)

    # ── Dynamic control ───────────────────────────────────────────────────

    def set_inter_request_delay(self, delay_s: float) -> None:
        """
        Adjust delay between requests at runtime.
        Called by orchestrator during THRASHING backoff.
        Not thread-safe — only call from the same event loop.
        """
        # Stored for use by run_loop on the next iteration.
        # run_loop reads this via the closure-passed kwarg on each cycle,
        # so we expose it as an attribute the orchestrator can overwrite.
        self._dynamic_delay_s = delay_s

    # ── Observability ─────────────────────────────────────────────────────

    @property
    def stats(self) -> dict:
        return {
            "instance_id": self.instance_id,
            "runs": self._runs,
            "successes": self._successes,
            "failures": self._failures,
            "timeouts": self._timeouts,
            "success_rate": round(self._successes / max(self._runs, 1), 4),
        }

    def __repr__(self) -> str:
        return (
            f"<AttackerInstance id={self.instance_id} "
            f"cls={self.cfg.attack_cls.__name__} "
            f"runs={self._runs}>"
        )