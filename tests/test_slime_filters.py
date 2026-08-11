"""Centralized safety contracts for GRPO group filtering."""
from __future__ import annotations

from dataclasses import dataclass

import pytest

from scripts.training.rewards import filters as slime_filters


@dataclass
class _Sample:
    reward: dict[str, object]
    label: str = "record"


def _group(category: str, count: int = 8) -> list[_Sample]:
    return [
        _Sample({"reward_value": 0.0, "training_eligible": False, "reward_category": category})
        for _ in range(count)
    ]


def test_judge_failure_rejects_the_entire_group(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOYAL_JUDGE_CIRCUIT_FAILURE_THRESHOLD", "10")
    slime_filters._CONSECUTIVE_INFRASTRUCTURE_FAILURES = 0

    result = slime_filters.keep_eligible_nonzero_std(None, [*_group("judge_failure", 1), *_group("scored", 7)])

    assert result.keep is False
    assert result.reason == "ineligible_judge_failure_1+scored_7"


def test_consecutive_all_group_judge_failures_open_circuit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOYAL_JUDGE_CIRCUIT_FAILURE_THRESHOLD", "2")
    slime_filters._CONSECUTIVE_INFRASTRUCTURE_FAILURES = 0

    first = slime_filters.keep_eligible_nonzero_std(None, _group("judge_failure"))
    assert first.keep is False
    with pytest.raises(slime_filters.JudgeCircuitOpen, match="restore the scorer service"):
        slime_filters.keep_eligible_nonzero_std(None, _group("judge_failure"))


def test_policy_failure_resets_judge_circuit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOYAL_JUDGE_CIRCUIT_FAILURE_THRESHOLD", "2")
    slime_filters._CONSECUTIVE_INFRASTRUCTURE_FAILURES = 0

    slime_filters.keep_eligible_nonzero_std(None, _group("judge_failure"))
    slime_filters.keep_eligible_nonzero_std(None, _group("truncated_rollout"))
    result = slime_filters.keep_eligible_nonzero_std(None, _group("judge_failure"))

    assert result.keep is False
    assert slime_filters._CONSECUTIVE_INFRASTRUCTURE_FAILURES == 1
