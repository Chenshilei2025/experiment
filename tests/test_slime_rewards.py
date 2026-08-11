"""Centralized tests for group-atomic remote-scorer retries."""
from __future__ import annotations

import asyncio

from scripts.training.rewards import slime as slime_rewards


def test_group_retry_only_rescores_unavailable_candidates(monkeypatch) -> None:
    monkeypatch.setenv("LOYAL_MIU_GROUP_RM_MAX_ATTEMPTS", "2")
    monkeypatch.setenv("LOYAL_MIU_GROUP_RM_RETRY_INITIAL_BACKOFF_SECONDS", "0")
    calls: list[list[int]] = []
    pending = [(0, "one", {}), (1, "two", {})]

    async def score(batch):
        indexes = [item[0] for item in batch]
        calls.append(indexes)
        return [
            {"reward": None} if indexes == [0, 1] and index == 1 else {"reward": float(index)}
            for index in indexes
        ]

    scores = asyncio.run(slime_rewards._retry_failed_scores(pending, score, mechanism="MIU"))

    assert calls == [[0, 1], [1]]
    assert scores == [{"reward": 0.0}, {"reward": 1.0}]
