"""Exact quota helpers for mixed MIU/EIL GRPO training."""
from __future__ import annotations

import hashlib
import math


def eil_rollout_count(*, total_rollouts: int, eil_probability: float) -> int:
    """Return the nearest feasible EIL quota for a finite training budget."""
    if total_rollouts < 1:
        raise ValueError("total_rollouts must be positive")
    if not 0.0 < eil_probability < 1.0:
        raise ValueError("eil_probability must be strictly between zero and one")
    return math.floor(total_rollouts * eil_probability + 0.5)


def task_for_rollout(
    *, rollout_id: int, seed: int, eil_probability: float = 0.5, total_rollouts: int | None = None,
) -> str:
    """Return the task for one rollout.

    With a finite budget, use a seed-phased, stratified schedule. It has the
    exact requested EIL quota while spacing tasks as uniformly as possible;
    this prevents sampling noise from becoming a confound in ratio ablations.
    The legacy hash draw is retained for callers that do not have a finite
    rollout horizon.
    """
    if rollout_id < 0 or seed < 0:
        raise ValueError("rollout_id and seed must be non-negative")
    if not 0.0 < eil_probability < 1.0:
        raise ValueError("eil_probability must be strictly between zero and one")
    if total_rollouts is not None:
        eil_count = eil_rollout_count(total_rollouts=total_rollouts, eil_probability=eil_probability)
        if rollout_id >= total_rollouts:
            raise ValueError("rollout_id must be smaller than total_rollouts")
        phase = int.from_bytes(hashlib.sha256(f"{seed}:mixed-task-phase-v1".encode("ascii")).digest()[:8], "big") / 2**64
        previous = math.floor(rollout_id * eil_count / total_rollouts + phase)
        current = math.floor((rollout_id + 1) * eil_count / total_rollouts + phase)
        return "EIL" if current > previous else "MIU"
    material = f"{seed}:{rollout_id}:mixed-task-v1".encode("ascii")
    draw = int.from_bytes(hashlib.sha256(material).digest()[:8], "big") / 2**64
    return "EIL" if draw < eil_probability else "MIU"
