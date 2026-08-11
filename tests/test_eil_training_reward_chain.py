"""Centralized contract test from a SLIME sample through EIL GRPO rewards."""
from __future__ import annotations

import asyncio
import json
import os
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path

from scripts.training.rewards import slime as slime_rewards


@dataclass
class _Sample:
    """Minimal SLIME Sample contract used by the reward adapter."""

    label: str
    response: str
    response_length: int
    status: str = "completed"
    reward: dict[str, object] | None = None


class _Client:
    def __init__(self, payload: dict[str, object]) -> None:
        self.model = "chain-test"
        self.payload = payload
        self.calls = 0

    async def chat_json(self, messages, *, temperature, max_tokens, seed=None):
        self.calls += 1
        return json.dumps(self.payload)


def test_eil_sample_to_grpo_reward_contract(monkeypatch) -> None:
    project = Path(__file__).resolve().parents[1]
    record = json.loads((project / "eil/data/dataset/EIL/train.jsonl").read_text(encoding="utf-8").splitlines()[0])
    reply = "I can provide employment verification and would like to schedule a viewing for a two-bedroom that permits a dog."
    samples = [_Sample(label=record["id"], response=reply, response_length=16) for _ in range(8)]
    os.environ["LOYAL_EIL_RECORDS"] = str(project / "eil/data/dataset/EIL/train.jsonl")
    os.environ["LOYAL_EIL_RM_MAX_CONCURRENT"] = "8"
    slime_rewards._STORES.clear()
    slime_rewards._CLIENTS.clear()
    slime_rewards._JUDGE_POOLS.clear()

    adversary = _Client({"inferred_private_facts": [], "counterparty_reply": "Please send the verification and we can arrange a viewing."})
    leakage = _Client({"slot_scores": [
        {"slot_id": slot["id"], "severity_score": 0, "source": "none", "reply_evidence": "", "inference_id": ""}
        for slot in record["exploitable_slots"]
    ]})
    utility = _Client({"utility_score": 0.67, "agent_reply_evidence": "provide employment verification", "counterparty_reply_evidence": "arrange a viewing"})

    monkeypatch.setattr(slime_rewards, "_client", lambda _: adversary)
    monkeypatch.setattr(slime_rewards, "_judge_pool", lambda name, _: leakage if "LEAKAGE" in name else utility)
    args = Namespace(current_rollout_id=0, start_rollout_id=0)
    rewards = asyncio.run(slime_rewards.eil_reward_func(args, samples))

    assert len(rewards) == 8
    assert all(set(item) == {"reward_value", "training_eligible", "reward_category", "task_utility", "leakage"} for item in rewards)
    assert all(item["training_eligible"] and item["reward_category"] == "scored" for item in rewards)
    assert all(item["task_utility"] == 0.67 and item["leakage"] == 0.0 for item in rewards)
    assert (adversary.calls, leakage.calls, utility.calls) == (8, 8, 8)
