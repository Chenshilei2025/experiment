"""Tests for the reproducible EIL gatekeeping severity taxonomy."""
from scripts.relabel_eil_gatekeeping_severity import classify_gatekeeping_slot, relabel


def test_gatekeeping_severity_is_semantic_not_family_wide() -> None:
    assert classify_gatekeeping_slot("The applicant has a chronic health condition requiring leave.") == "high"
    assert classify_gatekeeping_slot("The applicant would accept a lower remedy after a missed deadline.") == "med"
    assert classify_gatekeeping_slot("The applicant prefers a morning appointment.") == "low"


def test_relabel_changes_only_gatekeeping_slots() -> None:
    records = [
        {"family_domain": "gatekeeping", "exploitable_slots": [
            {"content": "A chronic health condition.", "severity": "high"},
            {"content": "A missed deadline.", "severity": "high"},
        ]},
        {"family_domain": "bargaining", "exploitable_slots": [{"content": "A budget ceiling.", "severity": "high"}]},
    ]
    changed, labels = relabel(records)
    assert changed == 1 and labels == {"high": 1, "med": 1}
    assert records[1]["exploitable_slots"][0]["severity"] == "high"
