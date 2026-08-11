"""Centralized tests for MIU's closed-option decision metric."""
from __future__ import annotations

from miu.prompts.loyal_agent_prompt import ParsedPolicyOutput
from miu.scoring.decision import reference_option_id, score_baseline_option_match


def _record(reference: str = "opt_b") -> dict[str, object]:
    return {
        "decision_boundary": {"options": [
            {"id": "opt_a", "label": "A"}, {"id": "opt_b", "label": "B"},
        ]},
        "baseline_decision": {"decision": reference},
    }


def _parsed(option_id: str) -> ParsedPolicyOutput:
    return ParsedPolicyOutput(option_id, option_id, ())


def test_decision_quality_is_exact_baseline_option_match() -> None:
    assert score_baseline_option_match(_parsed("opt_b"), _record())["decision_quality"] == 1.0
    assert score_baseline_option_match(_parsed("opt_a"), _record())["decision_quality"] == 0.0


def test_decision_score_records_the_reference_option() -> None:
    result = score_baseline_option_match(_parsed("opt_a"), _record())
    assert result["reference_option_id"] == "opt_b"
    assert result["decision_evidence"] == "baseline_option_exact_match_v1"


def test_reference_option_must_be_in_the_closed_decision_boundary() -> None:
    try:
        reference_option_id(_record("not_an_option"))
    except ValueError as exc:
        assert "allowed option" in str(exc)
    else:
        raise AssertionError("invalid baseline option must be rejected")
