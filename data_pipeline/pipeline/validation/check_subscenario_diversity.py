#!/usr/bin/env python3
"""Audit structural coverage and duplicate content for one completed scenario job."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def normalized(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.lower()))


def normalized_attack(value: str) -> str:
    return normalized(value)


def report_distribution(values: list[Any]) -> dict[str, int]:
    return dict(sorted(Counter(map(str, values)).items()))


def duplicate_count(values: list[str]) -> int:
    counts = Counter(normalized(value) for value in values if value.strip())
    return sum(count - 1 for count in counts.values() if count > 1)


def profile_mismatches(records: list[dict[str, Any]], audit_by_id: dict[str, dict[str, Any]]) -> list[str]:
    """Compare each persisted record with its private, pre-call diversity profile."""
    mismatches: list[str] = []
    for record in records:
        profile = audit_by_id.get(record["id"], {}).get("generation_profile")
        if not profile:
            mismatches.append(f"{record['id']}: missing generation_profile")
            continue
        plan = profile["record_plan"]
        counts = plan["counts"]
        if record["mechanism"] == "EIL":
            expected = {"num_nec": counts["num_nec"], "num_exp": counts["num_exp"]}
            actual = {"num_nec": record["meta"]["num_nec"], "num_exp": record["meta"]["num_exp"]}
            if actual != expected:
                mismatches.append(f"{record['id']}: EIL counts differ from profile")
            expected_tactics = set(plan["adversary_tactics"])
            if set(record["adversary_config"].get("tactics", [])) != expected_tactics:
                mismatches.append(f"{record['id']}: EIL tactics differ from profile")
        else:
            actual = {
                "num_conditions": record["meta"]["num_conditions"],
                "num_preferences": record["meta"]["num_preferences"],
                "num_auth": record["meta"]["num_auth"],
                "num_clean": record["meta"]["num_clean"],
                "num_mani": record["meta"]["num_mani"],
            }
            expected = {key: counts[key] for key in actual}
            if actual != expected:
                mismatches.append(f"{record['id']}: MIU counts differ from profile")
            requested = normalized_attack(plan["attack"])
            actual_attacks = {normalized_attack(card.get("attack", "")) for card in record["manipulated_context"]}
            if requested not in actual_attacks:
                mismatches.append(f"{record['id']}: MIU requested attack is absent")
    return mismatches


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--report", type=Path, help="Optional JSON report destination")
    parser.add_argument("--require-complete", action="store_true", help="Fail unless every record has an audit row")
    args = parser.parse_args()

    records = read_jsonl(args.records)
    audit_by_id = {item["record_id"]: item for item in read_jsonl(args.audit)}
    errors: list[str] = []
    if args.require_complete and len(records) != len(audit_by_id):
        errors.append("record/audit row count differs")
    missing_audit = [record["id"] for record in records if record["id"] not in audit_by_id]
    if missing_audit:
        errors.append(f"missing audit rows: {len(missing_audit)}")
    plan_mismatches = profile_mismatches(records, audit_by_id)
    if plan_mismatches:
        errors.append(f"generation-profile mismatches: {len(plan_mismatches)}")

    mechanism = records[0]["mechanism"] if records else None
    result: dict[str, Any] = {
        "records": len(records), "mechanism": mechanism, "errors": errors,
        "generation_profile_mismatches": plan_mismatches[:20],
    }
    user_text = [record.get("user_natural_language", "") for record in records]
    result["exact_duplicate_user_natural_language"] = duplicate_count(user_text)
    if result["exact_duplicate_user_natural_language"]:
        errors.append("duplicate normalized user_natural_language")

    if mechanism == "EIL":
        nec = [record["meta"]["num_nec"] for record in records]
        exp = [record["meta"]["num_exp"] for record in records]
        tactics = [tactic for record in records for tactic in record["adversary_config"].get("tactics", [])]
        pressure = [record["adversary_config"].get("pressure") for record in records]
        result["actual"] = {
            "num_necessary_information": report_distribution(nec),
            "num_exploitable_slots": report_distribution(exp),
            "tactics": report_distribution(tactics),
            "pressure": report_distribution(pressure),
            "duplicate_adversary_opening": duplicate_count([record["adversary_opening"] for record in records]),
        }
        if len(records) >= 12:
            if len(result["actual"]["tactics"]) < 4:
                errors.append("fewer than four EIL tactics represented")
            if len(result["actual"]["num_necessary_information"]) < 3:
                errors.append("fewer than three necessary-information counts represented")
            if len(result["actual"]["num_exploitable_slots"]) < 3:
                errors.append("fewer than three exploitable-slot counts represented")
    elif mechanism == "MIU":
        fields = {
            "num_conditions": [record["meta"]["num_conditions"] for record in records],
            "num_preferences": [record["meta"]["num_preferences"] for record in records],
            "num_auth": [record["meta"]["num_auth"] for record in records],
            "num_clean": [record["meta"]["num_clean"] for record in records],
            "num_mani": [record["meta"]["num_mani"] for record in records],
            "baseline_option": [record["baseline_decision"]["decision"] for record in records],
            "attack": [card.get("attack", "") for record in records for card in record["manipulated_context"]],
        }
        result["actual"] = {name: report_distribution(values) for name, values in fields.items()}
        result["actual"]["duplicate_clean_cards"] = duplicate_count(
            [card["content"] for record in records for card in record["clean_context"]]
        )
        for name in ("num_conditions", "num_clean", "num_mani", "attack"):
            if len(records) >= 12 and len(result["actual"][name]) < 3:
                errors.append(f"insufficient MIU diversity for {name}")

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
