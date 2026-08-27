"""SLIME dynamic-sampling filters for scorer availability and GRPO variance."""
from __future__ import annotations

from collections import Counter
import json
import os
from pathlib import Path
import time

from slime.rollout.filter_hub.base_types import DynamicFilterOutput


_CONSECUTIVE_INFRASTRUCTURE_FAILURES = 0
_INFRASTRUCTURE_FAILURE_CATEGORIES = frozenset({"judge_failure", "eil_evaluator_failure"})


class JudgeCircuitOpen(RuntimeError):
    """Stop a rollout before a down scorer causes unbounded replacement sampling."""


def _diagnostics_path() -> Path:
    return Path(
        os.getenv(
            "LOYAL_REWARD_FAILURE_LOG",
            os.getenv(
                "LOYAL_MIU_FAILURE_LOG",
                str(Path(__file__).resolve().parents[3] / "artifacts" / "diagnostics" / "reward_groups.jsonl"),
            ),
        )
    )


def _log_group(samples, *, kept: bool, reason: str, values: list[float] | None = None) -> None:
    """Persist per-candidate reward evidence without retaining generated text."""
    entries = []
    candidates = []
    for sample in samples:
        reward = getattr(sample, "reward", None)
        if not isinstance(reward, dict):
            entries.append({"record_id": str(getattr(sample, "label", "unknown")), "category": "invalid_reward"})
            continue
        metadata = getattr(sample, "metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
        candidate = {
            "record_id": str(getattr(sample, "label", "unknown")),
            "reward_value": _finite_float(reward.get("reward_value")),
            "training_eligible": bool(reward.get("training_eligible", False)),
            "reward_category": str(reward.get("reward_category", "unknown")),
            "policy_output_valid": _finite_float(reward.get("policy_output_valid")),
            "decision_exact_match": _finite_float(reward.get("decision_exact_match")),
            "reasoning_faithfulness": _finite_float(reward.get("reasoning_faithfulness")),
            "faithfulness_latency_seconds": _finite_float(reward.get("faithfulness_judge_latency_seconds")),
            "reasoning_hard_gate": reward.get("reasoning_hard_gate"),
            "selected_option_id": reward.get("selected_option_id"),
            "reference_option_id": reward.get("reference_option_id"),
            "candidate_resample_attempt": metadata.get("miu_candidate_resample_attempt"),
            "candidate_resample_reason": metadata.get("miu_candidate_resample_reason"),
            "zero_std_group_resample_attempt": metadata.get("miu_zero_std_group_resample_attempt"),
            "resample_history": metadata.get("miu_resample_history", []),
        }
        if isinstance(reward, dict) and "adversary_temperature" in reward:
            # EIL-only, judge-grounded telemetry for the training diversity
            # analysis. It intentionally excludes policy and adversary text.
            candidate["adversary_temperature"] = _finite_float(reward.get("adversary_temperature"))
            slot_ids = reward.get("adversary_recovered_slot_ids", [])
            candidate["adversary_recovered_slot_ids"] = (
                sorted(set(slot_ids)) if isinstance(slot_ids, list) and all(isinstance(slot_id, str) for slot_id in slot_ids)
                else []
            )
        candidates.append(candidate)
        if not candidate["training_eligible"]:
            entries.append({
                "record_id": candidate["record_id"], "category": candidate["reward_category"],
                # Validation errors identify the failure mode without retaining
                # policy output, prompts, private thinking, judge content, or keys.
                "policy_error": str(reward.get("policy_output_error") or "")[:300],
                "decision_error": str(reward.get("decision_scorer_error") or "")[:300],
                "faithfulness_error": str(reward.get("faithfulness_scorer_error") or "")[:300],
            })
    payload = {
        "schema_version": 2, "timestamp": time.time(), "kept": kept, "reason": reason,
        "record_ids": [str(getattr(sample, "label", "unknown")) for sample in samples],
        "reward_values": values,
        "reward_std": (sum((value - sum(values) / len(values)) ** 2 for value in values) / len(values)) ** 0.5 if values else None,
        "reward_mean": sum(values) / len(values) if values else None,
        "candidates": candidates,
        "failures": entries,
    }
    try:
        path = _diagnostics_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
    except OSError:
        # Diagnostics must never affect the acceptance decision for a rollout.
        return


def _finite_float(value):
    """Keep audit JSON numeric and compact when an unavailable scorer returns None."""
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else None


def _miu_min_group_reward_std(samples) -> float:
    """Return the optional MIU-only information floor for a GRPO group.

    A merely non-zero difference is not necessarily informative: after GRPO
    standard-deviation normalization it can otherwise become a large update.
    Keep this opt-in so historical recipes retain their exact behavior.
    """
    labels = [str(getattr(sample, "label", "")) for sample in samples]
    is_miu = bool(labels) and all(label.startswith("MIU:") for label in labels)
    if not is_miu:
        return 0.0
    value = float(os.getenv("LOYAL_MIU_MIN_GROUP_REWARD_STD", "0"))
    if value < 0:
        raise ValueError("LOYAL_MIU_MIN_GROUP_REWARD_STD must be non-negative")
    return value


def keep_eligible_nonzero_std(args, samples, **kwargs):
    """Discard unusable groups and optionally retain zero-advantage GRPO groups.

    Retaining a zero-variance group is useful when a fixed-size rollout batch
    must be produced without replacement sampling.  Its centered GRPO
    advantages are all zero, so it contributes no policy-gradient signal but
    keeps the batch shape valid.  This is deliberately opt-in because the
    default recipe prioritises informative groups over rollout throughput.
    """
    rewards = [getattr(sample, "reward", None) for sample in samples]
    ineligible = [item for item in rewards if not isinstance(item, dict) or not item.get("training_eligible", False)]
    if ineligible:
        # Attribute a rejected GRPO group to its member failure categories. The
        # rollout metric then distinguishes output-format issues from judge or
        # truncation availability failures without accepting a contaminated group.
        categories = Counter(
            str(item.get("reward_category", "invalid_reward")) if isinstance(item, dict) else "invalid_reward"
            for item in ineligible
        )
        reason = "+".join(f"{category}_{count}" for category, count in sorted(categories.items()))
        _log_group(samples, kept=False, reason=f"ineligible_{reason}")
        global _CONSECUTIVE_INFRASTRUCTURE_FAILURES
        if len(ineligible) == len(samples) and all(
            isinstance(item, dict) and item.get("reward_category") in _INFRASTRUCTURE_FAILURE_CATEGORIES
            for item in ineligible
        ):
            _CONSECUTIVE_INFRASTRUCTURE_FAILURES += 1
            threshold = int(os.getenv("LOYAL_JUDGE_CIRCUIT_FAILURE_THRESHOLD", "3"))
            if threshold < 1:
                raise ValueError("LOYAL_JUDGE_CIRCUIT_FAILURE_THRESHOLD must be at least one")
            if _CONSECUTIVE_INFRASTRUCTURE_FAILURES >= threshold:
                raise JudgeCircuitOpen(
                    "judge circuit open after "
                    f"{_CONSECUTIVE_INFRASTRUCTURE_FAILURES} consecutive all-group infrastructure failures; "
                    "restore the scorer service before resuming training"
                )
        else:
            _CONSECUTIVE_INFRASTRUCTURE_FAILURES = 0
        return DynamicFilterOutput(keep=False, reason=f"ineligible_{reason}")
    values = [float(item["reward_value"]) for item in rewards]
    reward_mean = sum(values) / len(values)
    reward_std = (sum((value - reward_mean) ** 2 for value in values) / len(values)) ** 0.5
    min_std = _miu_min_group_reward_std(samples)
    if len(set(values)) < 2 or reward_std < min_std:
        _CONSECUTIVE_INFRASTRUCTURE_FAILURES = 0
        reason = f"zero_std_{round(values[0], 2)}" if len(set(values)) < 2 else f"low_std_{round(reward_std, 4)}"
        if os.getenv("LOYAL_RETAIN_ZERO_STD_GROUPS", "0") == "1":
            _log_group(samples, kept=True, reason=f"retained_{reason}", values=values)
            return DynamicFilterOutput(keep=True)
        _log_group(samples, kept=False, reason=reason, values=values)
        return DynamicFilterOutput(keep=False, reason=reason)
    _CONSECUTIVE_INFRASTRUCTURE_FAILURES = 0
    _log_group(samples, kept=True, reason="accepted", values=values)
    return DynamicFilterOutput(keep=True)
