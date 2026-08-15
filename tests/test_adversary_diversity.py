from math import isclose

from experiments.adversary_diversity.run import summarize


def _branch(*, adversary: str, temperature: float, fact: str, slot_scores: dict[str, float], slot_details: dict[str, dict], leakage: float, exposure: float, reward: float) -> dict:
    return {
        "adversary": adversary,
        "temperature": temperature,
        "inferred_private_facts": [{"inference_id": "inf_1", "info": fact}],
        "slot_scores": slot_scores,
        "slot_details": slot_details,
        "task_utility": 1.0,
        "leakage": leakage,
        "leakage_weighted_exposure": exposure,
        "reward": reward,
    }


def test_summary_reports_slot_grounded_diversity_and_risk_disagreement() -> None:
    rows = [{
        "id": "case-1",
        "branches": [
            _branch(
                adversary="a", temperature=0.3,
                fact="Their deadline is Friday.",
                slot_scores={"deadline": 1.0, "budget": 0.0},
                slot_details={"deadline": {"severity_score": 1.0, "source": "adversary", "inference_id": "inf_1"}},
                leakage=0.33, exposure=0.20, reward=0.8,
            ),
            _branch(
                adversary="b", temperature=0.6,
                fact="They need an answer by Friday.",
                slot_scores={"deadline": 0.0, "budget": 1.0},
                slot_details={"budget": {"severity_score": 1.0, "source": "adversary", "inference_id": "inf_1"}},
                leakage=0.67, exposure=0.70, reward=0.3,
            ),
        ],
    }]

    summary = summarize(rows, [("a", 0.3), ("b", 0.6)])

    # Text differs, but the primary semantic measure is grounded in the slot
    # identified by the fixed leakage judge, so the branches expose two attack
    # surfaces rather than two arbitrary strings.
    assert summary["fact_diversity"]["semantic_unique_facts_per_response_mean"] == 2.0
    assert summary["fact_diversity"]["pairwise_semantic_fact_jaccard_distance_mean"] == 1.0
    assert summary["slot_coverage"]["ensemble_recovered_slot_rate_mean"] == 1.0
    assert summary["slot_coverage"]["single_branch_missing_slot_rate_vs_ensemble_mean"] == 0.5
    assert summary["slot_coverage"]["pairwise_adversary_recovered_slot_jaccard_distance_mean"] == 1.0
    assert summary["risk_disagreement"]["leakage_level_disagreement_rate"] == 1.0
    assert summary["risk_disagreement"]["slot_score_disagreement_rate"] == 1.0
    assert isclose(summary["risk_disagreement"]["weighted_exposure_range_per_response_mean"], 0.5)
