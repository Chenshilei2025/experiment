"""Centralized safety contracts for GRPO group filtering."""
from __future__ import annotations

from dataclasses import dataclass
import json

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
    monkeypatch.delenv("LOYAL_JUDGE_CIRCUIT_ACTION", raising=False)
    slime_filters._CONSECUTIVE_INFRASTRUCTURE_FAILURES = 0

    first = slime_filters.keep_eligible_nonzero_std(None, _group("judge_failure"))
    assert first.keep is False
    with pytest.raises(slime_filters.JudgeCircuitOpen, match="restore the scorer service"):
        slime_filters.keep_eligible_nonzero_std(None, _group("judge_failure"))


def test_circuit_soft_keep_keeps_training_recoverable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOYAL_JUDGE_CIRCUIT_FAILURE_THRESHOLD", "2")
    monkeypatch.setenv("LOYAL_JUDGE_CIRCUIT_ACTION", "soft_keep")
    slime_filters._CONSECUTIVE_INFRASTRUCTURE_FAILURES = 0

    first = slime_filters.keep_eligible_nonzero_std(None, _group("eil_evaluator_failure"))
    second = slime_filters.keep_eligible_nonzero_std(None, _group("eil_evaluator_failure"))

    assert first.keep is False
    assert second.keep is True
    assert second.reason == "judge_circuit_open_soft_keep_2_consecutive_infrastructure_failures"


def test_policy_failure_resets_judge_circuit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOYAL_JUDGE_CIRCUIT_FAILURE_THRESHOLD", "2")
    slime_filters._CONSECUTIVE_INFRASTRUCTURE_FAILURES = 0

    slime_filters.keep_eligible_nonzero_std(None, _group("judge_failure"))
    slime_filters.keep_eligible_nonzero_std(None, _group("truncated_rollout"))
    result = slime_filters.keep_eligible_nonzero_std(None, _group("judge_failure"))

    assert result.keep is False
    assert slime_filters._CONSECUTIVE_INFRASTRUCTURE_FAILURES == 1


def test_consecutive_all_group_failures_soft_keep_when_configured(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setenv("LOYAL_JUDGE_CIRCUIT_FAILURE_THRESHOLD", "2")
    monkeypatch.setenv("LOYAL_JUDGE_CIRCUIT_ACTION", "soft_keep")
    monkeypatch.setenv("LOYAL_REWARD_FAILURE_LOG", str(tmp_path / "reward_groups.jsonl"))
    slime_filters._CONSECUTIVE_INFRASTRUCTURE_FAILURES = 0

    first = slime_filters.keep_eligible_nonzero_std(None, _group("eil_evaluator_failure"))
    second = slime_filters.keep_eligible_nonzero_std(None, _group("eil_evaluator_failure"))

    assert first.keep is False
    assert second.keep is True
    assert second.reason == "judge_circuit_open_soft_keep_2_consecutive_infrastructure_failures"
    assert slime_filters._CONSECUTIVE_INFRASTRUCTURE_FAILURES == 2

    lines = (tmp_path / "reward_groups.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    payload = json.loads(lines[-1])
    assert payload["kept"] is True
    assert payload["reason"] == "judge_circuit_open_soft_keep_2_consecutive_infrastructure_failures"
