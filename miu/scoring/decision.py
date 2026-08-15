"""Deterministic MIU decision-alignment scoring.

MIU is a closed-option benchmark: every record provides one reference option
in ``baseline_decision.decision``. The per-sample decision metric is therefore
a binary exact-option match, rather than an LLM's assessment of semantics.
"""
from __future__ import annotations

from typing import Any

from miu.prompts.loyal_agent_prompt import ParsedPolicyOutput
from scripts.common.api_client import ChatClient


def reference_option_id(record: dict[str, Any]) -> str:
    """Return the validated, closed-set reference option for an MIU record."""
    options = record.get("decision_boundary", {}).get("options")
    reference = record.get("baseline_decision", {}).get("decision")
    if not isinstance(options, list) or not options:
        raise ValueError("MIU decision boundary requires non-empty options")
    option_ids = [option.get("id") for option in options if isinstance(option, dict)]
    if len(option_ids) != len(options) or any(not isinstance(option_id, str) or not option_id for option_id in option_ids):
        raise ValueError("MIU decision options require non-empty IDs")
    if len(set(option_ids)) != len(option_ids):
        raise ValueError("MIU decision options require unique IDs")
    if not isinstance(reference, str) or reference not in option_ids:
        raise ValueError("MIU baseline decision must name one allowed option")
    return reference


def score_baseline_option_match(parsed: ParsedPolicyOutput, record: dict[str, Any]) -> dict[str, Any]:
    """Score the parsed option against MIU's supplied reference option."""
    reference = reference_option_id(record)
    return {
        "decision_exact_match": float(parsed.selected_option_id == reference),
        "reference_option_id": reference,
        "decision_scorer_failed": False,
        "decision_scorer_error": None,
        "decision_evidence": "baseline_option_exact_match_v1",
    }


async def score_decision_exact_match(
    parsed: ParsedPolicyOutput, record: dict[str, Any], client: ChatClient | None = None,
) -> dict[str, Any]:
    """Return the deterministic per-sample reference-option exact match.

    ``client`` is deliberately unused.  Keeping it optional avoids breaking
    existing callers while ensuring the decision component never invokes an
    external judge.
    """
    del client
    try:
        return score_baseline_option_match(parsed, record)
    except Exception as exc:
        return {
            "decision_exact_match": None,
            "reference_option_id": None,
            "decision_scorer_failed": True,
            "decision_scorer_error": f"{type(exc).__name__}: {exc}",
            "decision_evidence": "invalid_reference_record",
        }
