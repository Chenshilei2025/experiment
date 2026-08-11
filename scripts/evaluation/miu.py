"""MIU-specific pieces used by generated evaluation and saved-response rescoring."""
from __future__ import annotations

import asyncio
from collections import Counter, defaultdict
from typing import Any

from miu.prompts.loyal_agent_prompt import policy_messages
from miu.reward import compute_miu_reward_for_response
from scripts.common.api_client import ApiClientPool

Record = dict[str, Any]


async def score_batch(responses: list[str], records: list[Record]) -> list[dict[str, Any]]:
    """Score one batch in its caller's event loop."""
    client = ApiClientPool.from_env("LOYAL_MIU_FAITHFULNESS_JUDGE", fallback_prefix="LOYAL_MIU_JUDGE")
    return await asyncio.gather(*(
        compute_miu_reward_for_response(response, record, client)
        for response, record in zip(responses, records, strict=True)
    ))


def row(record: Record, response: str, score: dict[str, Any]) -> Record:
    return {
        "id": record["id"], "family_domain": record["family_domain"],
        "subscenario": record["subscenario"], "response": response, "score": score,
    }


def summarize(rows: list[Record], run: dict[str, Any]) -> dict[str, Any]:
    def mean(values: list[float]) -> float | None:
        return sum(values) / len(values) if values else None

    valid = [item for item in rows if item["score"].get("policy_output_valid")]
    judged = [item for item in valid if not item["score"].get("faithfulness_scorer_failed")]
    by_family: dict[str, list[Record]] = defaultdict(list)
    by_subscenario: dict[str, list[Record]] = defaultdict(list)
    for item in rows:
        by_family[item["family_domain"]].append(item)
        by_subscenario[item["subscenario"]].append(item)

    def breakdown(items: list[Record]) -> dict[str, Any]:
        item_valid = [item for item in items if item["score"].get("policy_output_valid")]
        item_judged = [item for item in item_valid if not item["score"].get("faithfulness_scorer_failed")]
        return {
            "n": len(items),
            "policy_output_valid_rate": len(item_valid) / len(items),
            "decision_exact_match_rate": mean([float(item["score"]["decision_quality"]) for item in item_valid]),
            "n_valid_and_judged": len(item_judged),
            "reasoning_faithfulness_mean": mean([float(item["score"]["reasoning_faithfulness"]) for item in item_judged]),
            "reward_mean": mean([float(item["score"]["reward"]) for item in item_judged]),
        }

    decisions = [float(item["score"]["decision_quality"]) for item in valid]
    return {
        "run": run, "n_total": len(rows), "n_policy_valid": len(valid), "n_valid_and_judged": len(judged),
        "policy_output_valid_rate": len(valid) / len(rows),
        "decision_exact_match_rate": mean(decisions),
        "reasoning_faithfulness_mean": mean([float(item["score"]["reasoning_faithfulness"]) for item in judged]),
        "reward_mean": mean([float(item["score"]["reward"]) for item in judged]),
        "decision_quality_distribution": dict(sorted(Counter(decisions).items())),
        "reward_category_distribution": dict(sorted(Counter(
            str(item["score"].get("reward_unavailable_reason") or (
                "scored" if item["score"].get("policy_output_valid") else "invalid_policy_output"
            )) for item in rows
        ).items())),
        "by_family": {name: breakdown(items) for name, items in sorted(by_family.items())},
        "by_subscenario": {name: breakdown(items) for name, items in sorted(by_subscenario.items())},
    }
