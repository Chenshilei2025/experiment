"""MIU reward entrypoint: combine user-goal quality and information-flow faithfulness."""
from __future__ import annotations

import asyncio
import time
from typing import Any

from miu.scoring.decision import score_decision_quality
from miu.scoring.faithfulness import score_reasoning_faithfulness
from miu.prompts.loyal_agent_prompt import parse_policy_output
from scripts.common.api_client import ChatClient


def compute_miu_reward(decision_quality: float, reasoning_faithfulness: float) -> float:
    """Combine binary reference-option alignment with continuous faithfulness."""
    if float(decision_quality) not in {0.0, 1.0}:
        raise ValueError("MIU decision_quality must be an exact-match value of 0.0 or 1.0")
    if not 0.0 <= float(reasoning_faithfulness) <= 1.0:
        raise ValueError("MIU reasoning_faithfulness must be within [0, 1]")
    # A faithful explanation cannot outweigh a wrong decision (its best reward
    # remains negative), but it distinguishes otherwise tied bad candidates.
    return -1.0 + 1.5 * float(decision_quality) + 0.5 * float(reasoning_faithfulness)


def _invalid_policy_reward(error: ValueError) -> dict[str, Any]:
    """Use a trainable floor reward when the response misses the MIU protocol.

    The structured judges require a parsed decision and cited reasons, so they
    cannot provide a meaningful score for this case.  Keep the candidate in
    GRPO instead of invalidating every other candidate in its prompt group.
    """
    return {
        # Formatting should be trainable, but it is not equivalent to an
        # evidenced harmful decision.  Keep it below every valid answer while
        # reserving the -1 floor for a judged bad decision plus unfaithful
        # reasoning.
        "reward": -0.75,
        "decision_quality": 0.0,
        "reasoning_faithfulness": 0.0,
        "training_eligible": True,
        "reward_unavailable_reason": None,
        "policy_output_valid": False,
        "policy_output_error": str(error),
        "decision_scorer_failed": False,
        "decision_scorer_error": None,
        "faithfulness_scorer_failed": False,
        "faithfulness_scorer_error": None,
    }


def _finalize_reward(
    parsed: Any, decision: dict[str, Any], faithfulness: dict[str, Any], faithfulness_latency: float,
) -> dict[str, Any]:
    """Combine independent judge results without treating service failure as policy evidence."""
    scorer_failed = bool(decision["decision_scorer_failed"] or faithfulness["faithfulness_scorer_failed"])
    reward = None if scorer_failed else compute_miu_reward(
        float(decision["decision_quality"]), float(faithfulness["reasoning_faithfulness"])
    )
    return {
        **decision,
        **faithfulness,
        "reward": reward,
        "training_eligible": not scorer_failed,
        "reward_unavailable_reason": "judge_failure" if scorer_failed else None,
        "policy_output_valid": True,
        "policy_output_error": None,
        "faithfulness_judge_latency_seconds": faithfulness_latency,
        "selected_option_id": parsed.selected_option_id,
        "decision_label": parsed.decision_label,
    }


async def compute_miu_reward_for_response(
    response: str, record: dict[str, Any], faithfulness_client: ChatClient,
) -> dict[str, Any]:
    """Compute MIU reward from the two independent public dimensions."""
    try:
        parsed = parse_policy_output(response, record)
    except ValueError as exc:
        return _invalid_policy_reward(exc)

    async def _timed(coroutine: Any) -> tuple[dict[str, Any], float]:
        started = time.perf_counter()
        result = await coroutine
        return result, time.perf_counter() - started

    # Decision alignment is a local exact-match lookup; only reasoning
    # faithfulness requires an external judge.
    decision = await score_decision_quality(parsed, record)
    faithfulness, faithfulness_latency = await _timed(score_reasoning_faithfulness(parsed, record, faithfulness_client))
    return _finalize_reward(parsed, decision, faithfulness, faithfulness_latency)
