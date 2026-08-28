"""Exact-frequency, single-task batches for mixed MIU/EIL GRPO training.

Every rollout/GRPO batch contains one task only.  The experiment's EIL:MIU
ratio is controlled exclusively by the frequency of those single-task
batches over a finite horizon; it is never implemented by mixing task groups
within a GRPO update.
"""
from __future__ import annotations

import os
from typing import Any, Callable

from scripts.training.mixed_schedule import task_for_rollout
from scripts.training.rollout.miu import generate_rollout_async


def _eil_batch_fraction() -> float:
    """Read the explicit single-task batch frequency."""
    raw = os.getenv("LOYAL_MIXED_EIL_BATCH_FRACTION", os.getenv("LOYAL_MIXED_EIL_PROBABILITY", "0.5"))
    value = float(raw)
    if not 0.0 < value < 1.0:
        raise ValueError("LOYAL_MIXED_EIL_BATCH_FRACTION must be strictly between zero and one")
    return value


def _schedule_total_rollouts(args: Any) -> int:
    """Use the full experiment horizon for ratio scheduling across restarts."""
    raw = os.getenv("LOYAL_MIXED_SCHEDULE_TOTAL_ROLLOUTS")
    value = int(raw) if raw else int(getattr(args, "num_rollout", 0))
    if value < 1:
        raise ValueError("mixed schedule total rollouts must be positive")
    return value


def _label_task(group: list[Any]) -> str:
    label = getattr(group[0], "label", None)
    if not isinstance(label, str) or ":" not in label:
        raise ValueError("mixed training requires namespaced record labels")
    task, _ = label.split(":", 1)
    if task not in {"EIL", "MIU"} or any(getattr(item, "label", "").split(":", 1)[0] != task for item in group):
        raise ValueError("a prompt group must contain exactly one valid mixed task")
    return task


class _TaskSource:
    """Select one task's groups without discarding groups for the other task."""

    def __init__(self, source: Callable[[int], list[list[Any]]], task: str) -> None:
        self.source, self.task = source, task
        self.pending: dict[str, list[list[Any]]] = {"EIL": [], "MIU": []}

    def __call__(self, count: int) -> list[list[Any]]:
        selected = self.pending[self.task][:count]
        del self.pending[self.task][:count]
        while len(selected) < count:
            for group in self.source(max(8, count - len(selected))):
                target = _label_task(group)
                if target == self.task and len(selected) < count:
                    selected.append(group)
                else:
                    self.pending[target].append(group)
        return selected


def generate_rollout(args: Any, rollout_id: int, data_buffer: Any, evaluation: bool = False):
    """SLIME entrypoint: choose one task deterministically, then fill one batch."""
    from slime.rollout.base_types import RolloutFnTrainOutput
    from slime.rollout.sglang_rollout import generate_abortable_samples
    from slime.utils.async_utils import run

    if evaluation:
        output, aborted = generate_abortable_samples(args, rollout_id, data_buffer.get_samples, evaluation=True)
        data_buffer.add_samples(aborted)
        return output

    task = task_for_rollout(
        rollout_id=rollout_id,
        seed=int(getattr(args, "rollout_seed", 0)),
        eil_probability=_eil_batch_fraction(),
        total_rollouts=_schedule_total_rollouts(args),
    )
    args.current_mixed_task = task
    source = _TaskSource(data_buffer.get_samples, task)
    output, aborted = run(generate_rollout_async(args, rollout_id, source))
    data_buffer.add_samples(aborted)
    data_buffer.add_samples(source.pending["EIL"] + source.pending["MIU"])
    metrics = dict(output.metrics or {})
    metrics[f"rollout/mixed/task/{task.lower()}/batch"] = 1.0
    metrics[f"rollout/mixed/task/{task.lower()}/batch_fraction"] = 1.0
    metrics["rollout/mixed/task/eil/configured_batch_fraction"] = _eil_batch_fraction()
    return RolloutFnTrainOutput(samples=output.samples, metrics=metrics)
