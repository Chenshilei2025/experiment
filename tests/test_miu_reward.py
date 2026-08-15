"""Centralized tests for MIU reward scale and variation."""
from __future__ import annotations

import asyncio

from miu.reward import compute_miu_reward, compute_miu_reward_for_response


def test_low_quality_decisions_remain_negative_but_are_distinguished_by_faithfulness(monkeypatch) -> None:
    monkeypatch.delenv("LOYAL_MIU_FAITHFULNESS_ETA", raising=False)
    assert compute_miu_reward(0.0, 0.0) == -1.0
    assert compute_miu_reward(0.0, 1.0) == -0.5


def test_reward_remains_monotonic_in_exact_match_and_faithfulness(monkeypatch) -> None:
    monkeypatch.delenv("LOYAL_MIU_FAITHFULNESS_ETA", raising=False)
    assert compute_miu_reward(0.0, 0.0) < compute_miu_reward(0.0, 1.0)
    assert compute_miu_reward(0.0, 1.0) < compute_miu_reward(1.0, 0.0)
    assert compute_miu_reward(1.0, 0.0) < compute_miu_reward(1.0, 1.0)


def test_reward_rejects_out_of_range_components() -> None:
    for decision, faithfulness in ((-0.1, 0.5), (0.5, 0.5), (1.0, 1.1)):
        try:
            compute_miu_reward(decision, faithfulness)
        except ValueError:
            continue
        raise AssertionError("out-of-range reward components must be rejected")


def test_faithfulness_eta_controls_the_miu_shaping_weight(monkeypatch) -> None:
    monkeypatch.setenv("LOYAL_MIU_FAITHFULNESS_ETA", "0")
    assert compute_miu_reward(1.0, 0.0) == compute_miu_reward(1.0, 1.0) == 1.0
    monkeypatch.setenv("LOYAL_MIU_FAITHFULNESS_ETA", "1")
    assert compute_miu_reward(0.0, 1.0) == 0.0


def test_invalid_policy_output_is_a_trainable_floor_reward_without_judges() -> None:
    record = {
        "external_information": [{"content": "A visible fact."}],
        "decision_boundary": {"options": [{"id": "option_a", "label": "Proceed"}]},
    }

    result = asyncio.run(compute_miu_reward_for_response("not the required format", record, None))

    assert result["reward"] == -0.75
    assert result["training_eligible"] is True
    assert result["policy_output_valid"] is False
    assert result["decision_scorer_failed"] is False
    assert result["faithfulness_scorer_failed"] is False
