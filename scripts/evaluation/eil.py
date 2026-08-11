"""EIL-specific pieces used by generated evaluation and saved-response rescoring."""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Awaitable, Callable

from eil.prompts.loyal_agent_prompt import policy_messages
from eil.reward import batch_eil_rewards
from scripts.common.api_client import ApiClientPool

Record = dict[str, Any]


def score_batch(max_concurrent: int) -> Callable[[list[str], list[Record]], Awaitable[list[dict[str, Any]]]]:
    """Return a batch scorer with the requested adversary/judge concurrency."""
    async def score(responses: list[str], records: list[Record]) -> list[dict[str, Any]]:
        adversary = ApiClientPool.from_env("LOYAL_EIL_ADVERSARY")
        leakage = ApiClientPool.from_env("LOYAL_EIL_LEAKAGE_JUDGE", fallback_prefix="LOYAL_EIL_JUDGE")
        utility = ApiClientPool.from_env("LOYAL_EIL_UTILITY_JUDGE", fallback_prefix="LOYAL_EIL_JUDGE")
        return await batch_eil_rewards(
            responses, records, adversary, leakage, utility, max_concurrent=max_concurrent, evaluation=True,
        )
    return score


def row(record: Record, response: str, score: dict[str, Any]) -> Record:
    return {
        "id": record["id"], "family_domain": record["family_domain"],
        "response": response, "score": score,
    }


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def summarize(rows: list[Record], run: dict[str, Any]) -> dict[str, Any]:
    scored = [item for item in rows if item["score"].get("reward") is not None]
    by_family: dict[str, list[Record]] = defaultdict(list)
    for item in rows:
        by_family[item["family_domain"]].append(item)

    def breakdown(items: list[Record]) -> dict[str, Any]:
        completed = [item for item in items if item["score"].get("reward") is not None]
        return {
            "n": len(items), "n_scored": len(completed),
            "task_utility_mean": _mean([float(item["score"]["task_utility"]) for item in completed]),
            "leakage_mean": _mean([float(item["score"]["leakage"]) for item in completed]),
            "leakage_zero_rate": _mean([float(item["score"]["leakage"] == 0.0) for item in completed]),
            "reward_mean": _mean([float(item["score"]["reward"]) for item in completed]),
        }

    return {
        "run": run, "n_total": len(rows), "n_scored": len(scored), "n_failed": len(rows) - len(scored),
        "task_utility_mean": _mean([float(item["score"]["task_utility"]) for item in scored]),
        "task_utility_distribution": dict(sorted(Counter(float(item["score"]["task_utility"]) for item in scored).items())),
        "leakage_mean": _mean([float(item["score"]["leakage"]) for item in scored]),
        "leakage_distribution": dict(sorted(Counter(float(item["score"]["leakage"]) for item in scored).items())),
        "leakage_zero_rate": _mean([float(item["score"]["leakage"] == 0.0) for item in scored]),
        "reward_mean": _mean([float(item["score"]["reward"]) for item in scored]),
        "failure_types": dict(Counter(str(item["score"].get("evaluator_error", ""))[:300] for item in rows if item["score"].get("reward") is None)),
        "by_family_domain": {name: breakdown(items) for name, items in sorted(by_family.items())},
    }
