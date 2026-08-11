from scripts.relabel_eil_severity import relabel


def test_relabel_equalizes_family_severity_profile() -> None:
    records = []
    for family in ("bargaining", "gatekeeping", "redress"):
        for index in range(20):
            records.append({"id": f"{family}-{index}", "family_domain": family, "exploitable_slots": [
                {"id": "slot", "content": "medical treatment" if index == 0 else "private budget ceiling", "severity": "low"},
            ]})
    result = relabel(records)
    assert result == {family: {"high": 7, "med": 9, "low": 4} for family in result}
