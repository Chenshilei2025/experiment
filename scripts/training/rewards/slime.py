"""SLIME reward adapters and metrics for the MIU and EIL training jobs."""
from __future__ import annotations

import asyncio
import json
import math
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any

from eil.reward import batch_eil_rewards
from miu.reward import compute_miu_reward_for_response
from scripts.common.api_client import ApiClient, ApiClientPool, ChatClient
from scripts.common.thinking import has_incomplete_explicit_thinking, strip_thinking

_ROLL_OUT_COUNTS: dict[int, int] = {}
_STORES: dict[str, "RecordStore"] = {}
_CLIENTS: dict[str, ChatClient] = {}
_JUDGE_POOLS: dict[tuple[str, str, tuple[str | None, ...]], ApiClientPool] = {}
_GROUP_LIMITERS: dict[tuple[str, int], asyncio.Semaphore] = {}


def _quantile(values: list[float], percentile: float) -> float:
    """Compute an interpolated percentile without adding a metrics dependency."""
    ordered = sorted(values)
    if not ordered:
        raise ValueError("quantile requires at least one value")
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _distribution_metrics(values: list[float]) -> dict[str, float]:
    """Return compact distribution statistics for a non-empty reward slice."""
    mean = sum(values) / len(values)
    return {
        "count": float(len(values)),
        "mean": mean,
        "std": math.sqrt(sum((value - mean) ** 2 for value in values) / len(values)),
        "p25": _quantile(values, 0.25),
        "p50": _quantile(values, 0.50),
        "p75": _quantile(values, 0.75),
    }


def _response_length_metrics(samples: list[Any]) -> tuple[list[float], list[float]]:
    """Separate full policy tokens from the post-think visible answer length."""
    generated_tokens: list[float] = []
    visible_tokens: list[float] = []
    for sample in samples:
        length = getattr(sample, "response_length", None)
        if isinstance(length, int | float) and length >= 0:
            generated_tokens.append(float(length))
        response = getattr(sample, "response", "")
        if isinstance(response, str):
            # ``response`` has already passed through the thinking rollout adapter.
            visible_tokens.append(float(len(response.split())))
    return generated_tokens, visible_tokens


def _log_adaptive_signal(args: Any, prefix: str, raw_rewards: list[float]) -> None:
    """Append reward-distribution statistics for the alternating coordinator."""
    destination = os.getenv("LOYAL_ADAPTIVE_SIGNAL_LOG")
    if not destination or not raw_rewards:
        return
    try:
        group_size = int(args.n_samples_per_prompt)
        groups = [raw_rewards[index:index + group_size] for index in range(0, len(raw_rewards), group_size)]
        complete_groups = [group for group in groups if len(group) == group_size]
        group_stds = [math.sqrt(sum((value - sum(group) / len(group)) ** 2 for value in group) / len(group))
                      for group in complete_groups]
        group_ranges = [max(group) - min(group) for group in complete_groups]
        mean = sum(raw_rewards) / len(raw_rewards)
        payload = {
            "mechanism": os.getenv("LOYAL_ADAPTIVE_SIGNAL_MECHANISM", prefix),
            "timestamp": time.time(),
            "reward_count": len(raw_rewards),
            "raw_reward_mean": mean,
            "raw_reward_std": math.sqrt(sum((value - mean) ** 2 for value in raw_rewards) / len(raw_rewards)),
            "group_reward_std_mean": sum(group_stds) / max(len(group_stds), 1),
            "group_reward_range_mean": sum(group_ranges) / max(len(group_ranges), 1),
        }
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        # A single append keeps concurrent rollout-worker records intact on the
        # shared local filesystem without making training wait on the coordinator.
        descriptor = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
        try:
            os.write(descriptor, (json.dumps(payload, sort_keys=True) + "\n").encode("utf-8"))
        finally:
            os.close(descriptor)
    except (OSError, TypeError, ValueError, ZeroDivisionError):
        # Adaptive scheduling telemetry must never invalidate an RL rollout.
        return


class RecordStore:
    """Read immutable JSONL reward records once per rollout worker."""

    def __init__(self, path: str, mechanism: str) -> None:
        sources = [Path(item) for item in path.split(":") if item]
        if not sources or any(not source.is_file() for source in sources):
            raise FileNotFoundError(f"reward record file does not exist: {path}")
        self.records: dict[str, dict[str, Any]] = {}
        for source in sources:
            for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
                record = json.loads(line)
                record_id = record.get("id")
                if record.get("mechanism") != mechanism or not isinstance(record_id, str) or record_id in self.records:
                    raise ValueError(f"{source}:{line_number} has an invalid or duplicate {mechanism} record")
                self.records[record_id] = record

    def get(self, record_id: object) -> dict[str, Any]:
        if not isinstance(record_id, str) or record_id not in self.records:
            raise KeyError(f"unknown reward record ID: {record_id!r}")
        return self.records[record_id]


def _store(mechanism: str) -> RecordStore:
    if mechanism not in _STORES:
        _STORES[mechanism] = RecordStore(os.environ[f"LOYAL_{mechanism}_RECORDS"], mechanism)
    return _STORES[mechanism]


def _client(name: str) -> ChatClient:
    if name not in _CLIENTS:
        # API-backed EIL adversaries may provide multiple independent keys.
        # Reuse the pool's bounded round-robin behavior used by judge roles.
        _CLIENTS[name] = ApiClientPool.from_env(name)
    return _CLIENTS[name]


def _judge_pool(name: str, fallback_name: str) -> ChatClient:
    if name not in _CLIENTS:
        candidate = ApiClientPool.from_env(name, fallback_prefix=fallback_name)
        # Roles with identical endpoint, model, and credential share one client.
        identity = (
            candidate.clients[0].base_url,
            candidate.model,
            tuple(client.api_key for client in candidate.clients),
        )
        _CLIENTS[name] = _JUDGE_POOLS.setdefault(identity, candidate)
    return _CLIENTS[name]


def _visible_response(sample: Any) -> tuple[str | None, str | None]:
    raw = getattr(sample, "response", "")
    if not isinstance(raw, str):
        return None, "invalid_response_type"
    status = getattr(sample, "status", None)
    if getattr(status, "value", status) == "truncated":
        return None, "truncated_rollout"
    if has_incomplete_explicit_thinking(raw):
        return None, "incomplete_thinking"
    visible = strip_thinking(raw)
    return (visible, None) if visible else ("", "empty_visible_response")


def _unavailable_reward(reason: str) -> dict[str, Any]:
    return {"reward_value": 0.0, "training_eligible": False, "reward_category": reason}


def _group_limiter(mechanism: str) -> asyncio.Semaphore:
    """Bound simultaneous groups so a scorer outage cannot create a request storm."""
    limit = int(os.getenv(f"LOYAL_{mechanism}_GROUP_RM_MAX_CONCURRENT", "2"))
    if limit < 1:
        raise ValueError(f"LOYAL_{mechanism}_GROUP_RM_MAX_CONCURRENT must be at least one")
    key = (mechanism, id(asyncio.get_running_loop()))
    return _GROUP_LIMITERS.setdefault(key, asyncio.Semaphore(limit))


async def _retry_failed_scores(
    score_pending, score_all, *, mechanism: str,
) -> list[dict[str, Any]]:
    """Retry only unavailable candidates before allowing a GRPO group to be replaced.

    Successful candidates are not rejudged.  Keeping the group limiter acquired
    for the complete retry sequence makes recovery of this group take precedence
    over submitting replacement groups.
    """
    attempts = int(os.getenv(f"LOYAL_{mechanism}_GROUP_RM_MAX_ATTEMPTS", "2"))
    backoff = float(os.getenv(f"LOYAL_{mechanism}_GROUP_RM_RETRY_INITIAL_BACKOFF_SECONDS", "2"))
    if attempts < 1 or backoff < 0:
        raise ValueError(f"LOYAL_{mechanism}_GROUP_RM_MAX_ATTEMPTS must be positive and retry backoff non-negative")
    scores = await score_all(score_pending)
    failed = [(pending, score) for pending, score in zip(score_pending, scores, strict=True) if score.get("reward") is None]
    for attempt in range(1, attempts):
        if not failed:
            break
        await asyncio.sleep(backoff * (2 ** (attempt - 1)))
        retried = await score_all([pending for pending, _ in failed])
        retry_by_index = {pending[0]: score for (pending, _), score in zip(failed, retried, strict=True)}
        scores = [retry_by_index.get(pending[0], score) for pending, score in zip(score_pending, scores, strict=True)]
        failed = [(pending, score) for pending, score in zip(score_pending, scores, strict=True) if score.get("reward") is None]
    return scores


def _pending_samples(samples: Any, mechanism: str) -> tuple[bool, list[dict[str, Any] | None], list[tuple[int, str, dict[str, Any]]]]:
    one_sample = not isinstance(samples, list)
    batch = [samples] if one_sample else samples
    results: list[dict[str, Any] | None] = [None] * len(batch)
    pending: list[tuple[int, str, dict[str, Any]]] = []
    store = _store(mechanism)
    for index, sample in enumerate(batch):
        response, problem = _visible_response(sample)
        if problem and problem != "empty_visible_response":
            results[index] = _unavailable_reward(problem)
        else:
            pending.append((index, response or "", store.get(getattr(sample, "label", None))))
    return one_sample, results, pending


async def miu_reward_func(args: Any, samples: Any, **kwargs: Any) -> Any:
    """SLIME entrypoint for Manipulated Information Use rewards."""
    one_sample, results, pending = _pending_samples(samples, "MIU")
    if pending:
        async def score_all(batch):
            return await asyncio.gather(*(
                compute_miu_reward_for_response(
                    response,
                    record,
                    _judge_pool("LOYAL_MIU_FAITHFULNESS_JUDGE", "LOYAL_MIU_JUDGE"),
                )
                for _, response, record in batch
            ))

        async with _group_limiter("MIU"):
            raw = await _retry_failed_scores(pending, score_all, mechanism="MIU")
        for (index, _, _), score in zip(pending, raw, strict=True):
            if score["reward"] is None:
                # Preserve the scorer diagnostics for the dynamic group filter.
                # They are required to distinguish service failures from output
                # protocol failures; neither category is a policy negative.
                results[index] = {
                    **score,
                    "reward_value": 0.0,
                    "training_eligible": False,
                    "reward_category": str(score.get("reward_unavailable_reason") or "miu_judge_failure"),
                }
            else:
                results[index] = {
                    "reward_value": float(score["reward"]),
                    "training_eligible": bool(score.get("training_eligible", True)),
                    "reward_category": "scored" if score.get("policy_output_valid") else "invalid_policy_output",
                    "decision_quality": float(score["decision_quality"]),
                    "reasoning_faithfulness": float(score["reasoning_faithfulness"]),
                    "policy_output_valid": float(bool(score.get("policy_output_valid"))),
                    "decision_scorer_failed": bool(score.get("decision_scorer_failed", False)),
                    "faithfulness_scorer_failed": bool(score.get("faithfulness_scorer_failed", False)),
                    "decision_scorer_error": score.get("decision_scorer_error"),
                    "faithfulness_scorer_error": score.get("faithfulness_scorer_error"),
                    "policy_output_error": score.get("policy_output_error"),
                    "faithfulness_judge_latency_seconds": score.get("faithfulness_judge_latency_seconds"),
                    "reasoning_hard_gate": score.get("reasoning_hard_gate"),
                    "selected_option_id": score.get("selected_option_id"),
                    "reference_option_id": score.get("reference_option_id"),
                }
    assert all(result is not None for result in results)
    return results[0] if one_sample else results


async def eil_reward_func(args: Any, samples: Any, **kwargs: Any) -> Any:
    """SLIME entrypoint for EIL training or fixed-ensemble evaluation rewards."""
    one_sample, results, pending = _pending_samples(samples, "EIL")
    if pending:
        async def score_all(batch):
            return await batch_eil_rewards(
                [item[1] for item in batch], [item[2] for item in batch],
                _client("LOYAL_EIL_ADVERSARY"),
                _judge_pool("LOYAL_EIL_LEAKAGE_JUDGE", "LOYAL_EIL_JUDGE"),
                _judge_pool("LOYAL_EIL_UTILITY_JUDGE", "LOYAL_EIL_JUDGE"),
                # Each evaluation item fans out to four adversary branches, so use
                # a separate conservative cap to avoid overwhelming scorer APIs.
                max_concurrent=int(os.getenv(
                    "LOYAL_EIL_EVAL_RM_MAX_CONCURRENT" if kwargs.get("evaluation") else "LOYAL_EIL_RM_MAX_CONCURRENT",
                    "2" if kwargs.get("evaluation") else "4",
                )),
                rollout_id=getattr(args, "current_rollout_id", getattr(args, "start_rollout_id", 0)),
                evaluation=bool(kwargs.get("evaluation")),
            )

        async with _group_limiter("EIL"):
            raw = await _retry_failed_scores(pending, score_all, mechanism="EIL")
        for (index, _, _), score in zip(pending, raw, strict=True):
            results[index] = _unavailable_reward("eil_evaluator_failure") if score.get("reward") is None else {
                "reward_value": float(score["reward"]),
                "training_eligible": True,
                "reward_category": "scored",
                "task_utility": float(score["task_utility"]),
                "leakage": float(score["leakage"]),
                "leakage_weighted_exposure": float(score.get("leakage_weighted_exposure", score["leakage"])),
                "leakage_complete_high_slot": float(score.get("leakage_complete_high_slot", 0.0)),
            }
    assert all(result is not None for result in results)
    return results[0] if one_sample else results


def _rollout_step(args: Any) -> int:
    from slime.utils.metric_utils import compute_rollout_step

    identity = id(args)
    if identity not in _ROLL_OUT_COUNTS:
        _ROLL_OUT_COUNTS[identity] = int(getattr(args, "start_rollout_id", 0) or 0)
    step = _ROLL_OUT_COUNTS[identity]
    _ROLL_OUT_COUNTS[identity] = step + 1
    return compute_rollout_step(args, step)


def _post_process(args: Any, samples: list[Any], prefix: str, scalar_keys: tuple[str, ...]):
    """Log reward components while preserving SLIME's GRPO normalization."""
    import torch

    from slime.utils import tracking_utils
    from slime.utils.metric_utils import compute_statistics

    reward_dicts = [sample.reward for sample in samples]
    raw_rewards = [float(item["reward_value"]) for item in reward_dicts]
    _log_adaptive_signal(args, prefix, raw_rewards)
    metrics: dict[str, float] = {}
    # GRPO normalizes within each prompt group.  Log that pre-normalization
    # distribution explicitly so W&B can show whether a rollout supplies a
    # useful learning signal rather than only its batch-wide reward mean.
    group_size = int(getattr(args, "n_samples_per_prompt", 1))
    groups = [raw_rewards[index:index + group_size] for index in range(0, len(raw_rewards), group_size)]
    complete_groups = [group for group in groups if len(group) == group_size]
    if complete_groups:
        group_means = [sum(group) / group_size for group in complete_groups]
        group_stds = [math.sqrt(sum((value - mean) ** 2 for value in group) / group_size)
                      for group, mean in zip(complete_groups, group_means, strict=True)]
        group_ranges = [max(group) - min(group) for group in complete_groups]
        for name, value in _distribution_metrics(group_means).items():
            metrics[f"rollout/{prefix}/group_reward_mean/{name}"] = value
        for name, value in _distribution_metrics(group_stds).items():
            metrics[f"rollout/{prefix}/group_reward_std/{name}"] = value
        for name, value in _distribution_metrics(group_ranges).items():
            metrics[f"rollout/{prefix}/group_reward_range/{name}"] = value
        metrics[f"rollout/{prefix}/group_reward_zero_std_rate"] = (
            sum(std == 0.0 for std in group_stds) / len(group_stds)
        )
    for key in scalar_keys:
        values = [float(item[key]) for item in reward_dicts if item.get(key) is not None]
        if values:
            metrics[f"rollout/{prefix}/{key}/mean"] = sum(values) / len(values)
            metrics[f"rollout/{prefix}/{key}/median"] = compute_statistics(values)["median"]
    for category, count in Counter(str(item.get("reward_category", "unknown")) for item in reward_dicts).items():
        metrics[f"rollout/{prefix}/reward_category/{category}"] = count / len(reward_dicts)
    if prefix == "miu":
        # Health metrics are separate from policy reward: a remote scorer
        # outage must never look like poor policy behaviour.
        for key in ("decision_scorer_failed", "faithfulness_scorer_failed", "policy_output_valid"):
            values = [float(bool(item.get(key))) for item in reward_dicts if item.get(key) is not None]
            if values:
                metrics[f"rollout/miu/{key}/rate"] = sum(values) / len(values)
        for key in ("reasoning_hard_gate",):
            for value, count in Counter(
                str(item.get(key)) for item in reward_dicts if item.get(key) is not None
            ).items():
                metrics[f"rollout/miu/{key}/{value}/rate"] = count / len(reward_dicts)
        # Family-level monitoring detects scorer or reward-scale skew before it
        # is strong enough to justify changing GRPO normalization or sampling.
        families: dict[str, list[dict[str, Any]]] = {}
        family_samples: dict[str, list[Any]] = {}
        store = _store("MIU")
        for sample, reward in zip(samples, reward_dicts, strict=True):
            family = str(store.get(getattr(sample, "label", None))["family_domain"])
            families.setdefault(family, []).append(reward)
            family_samples.setdefault(family, []).append(sample)
        generated_tokens, visible_tokens = _response_length_metrics(samples)
        for name, value in _distribution_metrics(generated_tokens).items():
            metrics[f"rollout/miu/response_tokens/{name}"] = value
        for name, value in _distribution_metrics(visible_tokens).items():
            metrics[f"rollout/miu/visible_response_tokens/{name}"] = value
        for family, family_rewards in families.items():
            metric_prefix = f"rollout/miu/family/{family}"
            values = [float(item["reward_value"]) for item in family_rewards]
            for name, value in _distribution_metrics(values).items():
                metrics[f"{metric_prefix}/reward/{name}"] = value
            for component in ("decision_quality", "reasoning_faithfulness"):
                component_values = [float(item[component]) for item in family_rewards if item.get(component) is not None]
                if component_values:
                    for name, value in _distribution_metrics(component_values).items():
                        metrics[f"{metric_prefix}/{component}/{name}"] = value
            eligible = [float(item["reward_value"]) for item in family_rewards if item.get("training_eligible", False)]
            metrics[f"{metric_prefix}/eligible_rate"] = len(eligible) / len(family_rewards)
            if eligible:
                for name, value in _distribution_metrics(eligible).items():
                    metrics[f"{metric_prefix}/eligible_reward/{name}"] = value
            for category, count in Counter(str(item.get("reward_category", "unknown")) for item in family_rewards).items():
                metrics[f"{metric_prefix}/reward_category/{category}"] = count / len(family_rewards)
            family_generated, family_visible = _response_length_metrics(family_samples[family])
            for name, value in _distribution_metrics(family_generated).items():
                metrics[f"{metric_prefix}/response_tokens/{name}"] = value
            for name, value in _distribution_metrics(family_visible).items():
                metrics[f"{metric_prefix}/visible_response_tokens/{name}"] = value
    elif prefix == "eil":
        # Keep EIL family skew visible in W&B: the aggregate reward can hide
        # a family that is improving through utility while regressing leakage.
        families: dict[str, list[dict[str, Any]]] = {}
        store = _store("EIL")
        for sample, reward in zip(samples, reward_dicts, strict=True):
            family = str(store.get(getattr(sample, "label", None))["family_domain"])
            families.setdefault(family, []).append(reward)
        for family, family_rewards in families.items():
            metric_prefix = f"rollout/eil/family_domain/{family}"
            metrics[f"{metric_prefix}/n"] = float(len(family_rewards))
            for component in (
                "reward_value", "task_utility", "leakage",
                "leakage_weighted_exposure", "leakage_complete_high_slot",
            ):
                values = [float(item[component]) for item in family_rewards if item.get(component) is not None]
                if values:
                    metrics[f"{metric_prefix}/{component}/mean"] = sum(values) / len(values)
                    for name, value in _distribution_metrics(values).items():
                        metrics[f"{metric_prefix}/{component}/{name}"] = value
            metrics[f"{metric_prefix}/eligible_rate"] = sum(
                bool(item.get("training_eligible", False)) for item in family_rewards
            ) / len(family_rewards)
    metrics["rollout/step"] = _rollout_step(args)
    tracking_utils.log(args, metrics, step_key="rollout/step")
    if args.advantage_estimator in ["grpo", "gspo", "reinforce_plus_plus_baseline"] and args.rewards_normalization:
        rewards = torch.tensor(raw_rewards, dtype=torch.float)
        if rewards.shape[-1] == args.n_samples_per_prompt * args.rollout_batch_size:
            rewards = rewards.reshape(-1, args.n_samples_per_prompt)
        else:
            rewards = rewards.view(-1, rewards.shape[-1])
        rewards = rewards - rewards.mean(dim=-1, keepdim=True)
        if args.advantage_estimator in ["grpo", "gspo"] and args.grpo_std_normalization:
            rewards = rewards / (rewards.std(dim=-1, keepdim=True) + 1e-6)
        return raw_rewards, rewards.flatten().tolist()
    return raw_rewards, raw_rewards


def eil_post_process_rewards(args: Any, samples: list[Any], **kwargs: Any):
    return _post_process(args, samples, "eil", (
        "reward_value", "task_utility", "leakage",
        "leakage_weighted_exposure", "leakage_complete_high_slot",
    ))


def miu_post_process_rewards(args: Any, samples: list[Any], **kwargs: Any):
    return _post_process(args, samples, "miu", (
        "reward_value", "decision_quality", "reasoning_faithfulness", "policy_output_valid",
        "faithfulness_judge_latency_seconds",
        "decision_scorer_failed", "faithfulness_scorer_failed",
    ))
