"""
orchestration/registry.py

Attack class registry — maps string identifiers to BaseAttack subclasses.
Extend this as new attack modules (ThinkTrap, FillSqueeze) are implemented.
"""
from __future__ import annotations

from typing import Callable, Optional, Type

from core.base_attack import AttackConfig, BaseAttack, TargetConfig
from attacks.reasoning_bomb import ReasoningBombAttack, ReasoningBombConfig

ATTACK_REGISTRY: dict[str, Type[BaseAttack]] = {
    "reasoning_bomb": ReasoningBombAttack,
    # "fill_squeeze":   FillSqueezeAttack,    # TODO
    # "think_trap":     ThinkTrapAttack,      # TODO
}


def make_config_factory(
    target: TargetConfig,
    max_tokens: int = 8192,
    temperature: float = 0.0,
    stream: bool = True,
    tag: Optional[str] = None,
    **kwargs,
) -> Callable[[], AttackConfig]:
    """
    Returns a factory that produces a fresh AttackConfig (new request_id)
    on every call. Pass this directly to AttackerInstanceConfig.
    """
    def _factory() -> AttackConfig:
        return AttackConfig(
            target=target,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=stream,
            tag=tag,
            **kwargs,
        )
    return _factory