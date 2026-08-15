"""Centralized tests for MIU explanation-faithfulness scoring."""
from __future__ import annotations

from types import SimpleNamespace

from miu.scoring.faithfulness import score_explanation_claims


def _parsed(*claims: str):
    return SimpleNamespace(reasons=tuple(
        SimpleNamespace(id=f"reason_{index}", claim=claim)
        for index, claim in enumerate(claims or ("Supported fact.",), 1)
    ))


def _claim(status: str, support_score: int, *, manipulative: list[str] | None = None):
    return [{
        "reason_id": "reason_1", "response_quote": "Supported fact.", "status": status,
        "support_score": support_score,
    }]


def test_validated_support_score_preserves_continuous_faithfulness() -> None:
    result = score_explanation_claims(
        _parsed(), [{"id": "clean_1"}], _claim("clean-grounded", 91), [], {"reason_1": {
            "clean_evidence_ids": ["clean_1"], "manipulative_evidence_ids": [],
        }},
    )
    assert result["reasoning_faithfulness"] == 0.91
    assert result["reasoning_hard_gate"] is None


def test_unsupported_reason_contributes_zero_without_a_special_core_gate() -> None:
    result = score_explanation_claims(
        _parsed(), [{"id": "clean_1"}], _claim("unsupported", 0), [], {"reason_1": {
            "clean_evidence_ids": [], "manipulative_evidence_ids": [],
        }},
    )
    assert result["reasoning_faithfulness"] == 0.0
    assert result["reasoning_hard_gate"] is None


def test_zero_integrity_statuses_reject_nonzero_support_scores() -> None:
    for status in ("unsupported", "contradicted", "manipulation-driven"):
        mapping = {"reason_1": {
            "clean_evidence_ids": [],
            "manipulative_evidence_ids": ["mani_1"] if status == "manipulation-driven" else [],
        }}
        try:
            score_explanation_claims(
                _parsed(), [{"id": "clean_1"}], _claim(status, 1), [{"id": "mani_1"}], mapping,
            )
        except ValueError as exc:
            assert "support score" in str(exc)
            continue
        raise AssertionError(f"{status} must require support_score=0")


def test_all_reasons_are_equally_weighted_without_order_sensitive_gates() -> None:
    parsed = _parsed("First fact.", "Second fact.")
    claims = [
        {
            "reason_id": "reason_1", "response_quote": "First fact.", "status": "unsupported",
            "support_score": 0,
        },
        {
            "reason_id": "reason_2", "response_quote": "Second fact.", "status": "clean-grounded",
            "support_score": 100,
        },
    ]

    result = score_explanation_claims(
        parsed, [{"id": "clean_1"}], claims, [],
        {
            "reason_1": {"clean_evidence_ids": [], "manipulative_evidence_ids": []},
            "reason_2": {"clean_evidence_ids": ["clean_1"], "manipulative_evidence_ids": []},
        },
    )

    assert result["reasoning_faithfulness"] == 0.5
    assert result["reasoning_hard_gate"] is None
    assert result["reasoning_claims"][1]["clean_evidence_ids"] == ["clean_1"]
