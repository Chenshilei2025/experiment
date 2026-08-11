"""MIU rollout with bounded candidate replacement after group-aware scoring."""
from __future__ import annotations

import asyncio
import copy
import os
from typing import Any, Callable

from tqdm import tqdm

from slime.rollout.base_types import RolloutFnTrainOutput
from slime.rollout.rm_hub import batched_async_rm
from slime.rollout.sglang_rollout import (
    GenerateState,
    _MetricGatherer,
    _call_dynamic_filter,
    abort,
    generate_and_rm,
    generate_and_rm_group,
)
from slime.utils.misc import load_function
from slime.utils.types import Sample


def _fresh_sample(sample: Sample, reason: str, attempt: int) -> Sample:
    """Reuse the prompt slot but remove all generation and score state."""
    replacement = copy.deepcopy(sample)
    replacement.tokens = []
    replacement.response = ""
    replacement.response_length = 0
    replacement.reward = None
    replacement.loss_mask = None
    replacement.weight_versions = []
    replacement.rollout_log_probs = None
    replacement.rollout_routed_experts = None
    replacement.status = Sample.Status.PENDING
    replacement.spec_info = Sample.SpecInfo()
    replacement.metadata = dict(replacement.metadata)
    history = list(replacement.metadata.get("miu_resample_history", []))
    history.append({
        "attempt": attempt,
        "reason": reason,
        "reward_value": _audit_value(getattr(sample, "reward", None), "reward_value"),
        "reward_category": _audit_value(getattr(sample, "reward", None), "reward_category"),
    })
    replacement.metadata["miu_resample_history"] = history
    replacement.metadata["miu_candidate_resample_attempt"] = attempt
    replacement.metadata["miu_candidate_resample_reason"] = reason
    return replacement


def _audit_value(reward: Any, key: str) -> Any:
    """Copy only scalar reward evidence into a replacement audit history."""
    if not isinstance(reward, dict):
        return None
    value = reward.get(key)
    return value if isinstance(value, str | int | float | bool) or value is None else None


def _replacement_reason(sample: Sample) -> str | None:
    reward = sample.reward
    if not isinstance(reward, dict):
        return "invalid_reward"
    if not reward.get("training_eligible", False):
        return str(reward.get("reward_category", "ineligible"))
    # Output protocol errors are trainable floors in the base reward, but this
    # rollout favors obtaining usable group rankings before consuming a GRPO step.
    if reward.get("reward_category") != "scored":
        return str(reward.get("reward_category", "unscored"))
    return None


def _has_nonzero_std(group: list[Sample]) -> bool:
    values = [float(sample.reward["reward_value"]) for sample in group if isinstance(sample.reward, dict)]
    return len(values) == len(group) and len(set(values)) > 1


async def _score_replacements(args: Any, samples: list[Sample], sampling_params: dict[str, Any]) -> list[Sample]:
    """Generate and score replacements while retaining the prompt's GRPO group."""
    generated = await asyncio.gather(*(
        generate_and_rm(args, sample, sampling_params.copy(), evaluation=False) for sample in samples
    ))
    generated = list(generated)
    rewards = await batched_async_rm(args, generated)
    for sample, reward in zip(generated, rewards, strict=True):
        sample.reward = reward
    return generated


async def repair_group(args: Any, group: list[Sample], sampling_params: dict[str, Any]) -> list[Sample]:
    """Replace only invalid candidates; retry a flat group as a last resort."""
    candidate_attempts = int(os.getenv("LOYAL_MIU_CANDIDATE_RESAMPLE_ATTEMPTS", "2"))
    flat_group_attempts = int(os.getenv("LOYAL_MIU_ZERO_STD_GROUP_RESAMPLE_ATTEMPTS", "1"))
    if candidate_attempts < 0 or flat_group_attempts < 0:
        raise ValueError("MIU resample attempt counts must be non-negative")

    for attempt in range(1, candidate_attempts + 1):
        indexes = [(index, _replacement_reason(sample)) for index, sample in enumerate(group)]
        indexes = [(index, reason) for index, reason in indexes if reason is not None]
        if not indexes:
            break
        replacements = [_fresh_sample(group[index], reason, attempt) for index, reason in indexes]
        replacements = await _score_replacements(args, replacements, sampling_params)
        for (index, _), replacement in zip(indexes, replacements, strict=True):
            group[index] = replacement

    if any(_replacement_reason(sample) is not None for sample in group):
        return group

    # A group with identical rewards yields zero GRPO advantage. Redraw the
    # entire prompt group, but keep this bounded because flat rewards can be a
    # real property of the task rather than an infrastructure failure.
    for attempt in range(1, flat_group_attempts + 1):
        if _has_nonzero_std(group):
            break
        replacements = [_fresh_sample(sample, "zero_std", attempt) for sample in group]
        group = await _score_replacements(args, replacements, sampling_params)
        for sample in group:
            sample.metadata["miu_zero_std_group_resample_attempt"] = attempt
    return group


async def generate_rollout_async(
    args: Any, rollout_id: int, data_source: Callable[[int], list[list[Sample]]]
) -> tuple[RolloutFnTrainOutput, list[list[Sample]]]:
    """Mirror SLIME's rollout loop with MIU candidate-level repair."""
    assert args.rollout_global_dataset
    args.current_rollout_id = rollout_id
    state = GenerateState(args)
    dynamic_filter = load_function(args.dynamic_sampling_filter_path) if args.dynamic_sampling_filter_path else None
    metric_gatherer = _MetricGatherer()
    target = args.rollout_batch_size
    data: list[list[Sample]] = []
    pbar = tqdm(total=target * args.n_samples_per_prompt, desc="Rollout generation")

    while len(data) < target:
        while state.remaining_batch_size < target:
            state.submit_generate_tasks(data_source(args.over_sampling_batch_size))
        done, state.pendings = await asyncio.wait(state.pendings, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            group = await repair_group(args, task.result(), state.sampling_params)
            outcome = _call_dynamic_filter(dynamic_filter, args, group)
            if not outcome.keep:
                metric_gatherer.on_dynamic_filter_drop(reason=outcome.reason)
                state.remaining_batch_size -= 1
                continue
            if len(data) < target:
                data.append(group)
                pbar.update(args.n_samples_per_prompt)

    pbar.close()
    # ``GenerateState`` deliberately keeps several groups in flight to replace
    # filtered groups quickly.  Once the target is full, however, waiting for
    # every oversubscribed group can leave all rollout GPUs idle behind a slow
    # remote judge.  Cancel local coroutine wrappers before asking SGLang to
    # abort the corresponding requests; ``abort`` then drains only the short
    # cancellation/abort cleanup, rather than continuing to score surplus
    # groups that can no longer enter this GRPO update.
    for task in state.pendings:
        task.cancel()
    aborted = await abort(args, rollout_id)
    data.sort(key=lambda group: group[0].index)
    state.reset()
    return RolloutFnTrainOutput(samples=data, metrics=metric_gatherer.collect()), aborted


def generate_rollout(args: Any, rollout_id: int, data_buffer: Any, evaluation: bool = False):
    """SLIME rollout entrypoint; evaluation retains the framework default path."""
    from slime.rollout.sglang_rollout import generate_abortable_samples
    from slime.utils.async_utils import run

    if evaluation:
        output, aborted = generate_abortable_samples(args, rollout_id, data_buffer.get_samples, evaluation=True)
    else:
        output, aborted = run(generate_rollout_async(args, rollout_id, data_buffer.get_samples))
    data_buffer.add_samples(aborted)
    return output
