"""Tests for task-aware derived MIU/EIL training data."""
from scripts.build_balanced_training_data import _rotate_commerce_options, has_direct_eil_conflict, _target_counts


def test_target_counts_are_exact() -> None:
    assert _target_counts(10, {"a": 0.4, "b": 0.35, "c": 0.25}) == {"a": 4, "b": 4, "c": 2}


def test_commerce_rotation_moves_reference_action() -> None:
    record = {"baseline_decision": {"decision": "b"}, "decision_boundary": {"options": [
        {"id": "a"}, {"id": "b"}, {"id": "c"},
    ]}}
    _rotate_commerce_options(record, 0)
    assert record["decision_boundary"]["options"][0]["id"] == "b"


def test_direct_eil_conflict_is_conservative() -> None:
    conflict = {"necessary_information": [{"content": "The client requests the exact maximum monthly rent of 2600 dollars."}],
                "exploitable_slots": [{"content": "The client's exact maximum monthly rent is 2600 dollars.", "severity": "high"}]}
    safe = {"necessary_information": [{"content": "The client requests a one-year lease."}],
            "exploitable_slots": [{"content": "The client has a private budget ceiling.", "severity": "med"}]}
    assert has_direct_eil_conflict(conflict)
    assert not has_direct_eil_conflict(safe)
