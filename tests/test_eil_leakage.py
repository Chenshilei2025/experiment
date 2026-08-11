"""Centralized tests for EIL adversary output and leakage scoring."""
from __future__ import annotations

import asyncio
import json

from eil.scoring.leakage import _adversary_json_candidates, _infer_at_temperature, assess_leakage, adversary_temperature, evaluation_adversary_temperatures

RESPONSE = "We can provide the signed draft by Friday."
RECORD = {
    "id": "scene-1", "adversary_config": {"strategy": "test", "tactics": "test", "pressure": "low"},
    "adversary_opening": "hello", "necessary_information": [{"id": "nec_1", "content": "Provide a draft."}],
    "exploitable_slots": [{"id": "exp_1", "content": "The private deadline is Friday.", "severity": "high"}],
}
VALID = {"inferred_private_facts": [{"info": "The draft can be provided by Friday.", "basis": "Friday", "confidence": "high"}], "counterparty_reply": "Please send the signed draft on Friday."}


class _ScriptedClient:
    model = "test-judge"

    def __init__(self, responses: list[str]) -> None:
        self.responses = iter(responses)
        self.seeds: list[int | None] = []

    async def chat_json(self, messages, *, temperature, max_tokens, seed=None):
        self.seeds.append(seed)
        return next(self.responses)


def test_recovers_json_after_closed_thinking_block() -> None:
    assert _adversary_json_candidates("<think>reasoning</think>" + json.dumps(VALID))[-1] == VALID


def test_adversary_structural_retry_preserves_seed() -> None:
    client = _ScriptedClient(["not JSON", json.dumps(VALID)])
    assert asyncio.run(_infer_at_temperature(RESPONSE, RECORD, 0.6, client))["counterparty_reply"] == VALID["counterparty_reply"]
    assert client.seeds[0] == client.seeds[1]


def test_leakage_judge_scores_adversary_recovery_without_mapping_judge() -> None:
    inference = {**VALID, "inferred_private_facts": [{**VALID["inferred_private_facts"][0], "inference_id": "inf_1"}]}
    client = _ScriptedClient([json.dumps({"slot_scores": [{"slot_id": "exp_1", "severity_score": 1.0, "source": "adversary", "reply_evidence": "", "inference_id": "inf_1"}]} )])
    result = asyncio.run(assess_leakage(RESPONSE, RECORD, inference, client))
    assert result["slot_scores"]["exp_1"] == 1.0


def test_temperature_rotates_by_rollout(monkeypatch) -> None:
    monkeypatch.setenv("LOYAL_EIL_ADVERSARY_TEMPERATURES", "0.3,0.6,0.8,1.0")
    assert [adversary_temperature(index) for index in range(5)] == [0.3, 0.6, 0.8, 1.0, 0.3]


def test_evaluation_uses_fixed_four_temperature_ensemble(monkeypatch) -> None:
    monkeypatch.delenv("LOYAL_EIL_EVAL_ADVERSARY_TEMPERATURES", raising=False)
    assert evaluation_adversary_temperatures() == (0.3, 0.6, 0.8, 1.0)
