#!/usr/bin/env python3
"""Quarantine objectively risky completed records before same-ID regeneration.

Only completed jobs are eligible. Removed rows are preserved in a private
quarantine JSONL before their job and final split copies are atomically updated.
The normal resumable runner then regenerates the missing IDs.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..generation import run_subscenario as runner
from ..generation import builder as pipeline


RECORD_LEVEL_WARNING_CODES = {
    "near_duplicate_user_request",
    "miu_attack_promotes_baseline",
    "eil_slot_near_duplicate",
}


def selected_quality_ids(report: dict[str, Any]) -> set[str]:
    """Return explicit per-record release-gate failures and triage warnings."""
    selected = {issue["record_id"] for issue in report["hard_errors"]}
    selected.update(
        issue["record_id"] for issue in report["warnings"]
        if issue["code"] in RECORD_LEVEL_WARNING_CODES
    )
    selected.discard("__scenario__")
    return selected


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    os.replace(temporary, path)


def lock_and_rewrite(path: Path, rows: list[dict[str, Any]]) -> None:
    lock_path = path.with_suffix(path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        write_jsonl(path, rows)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def baseline_excess(records: list[dict[str, Any]], block: pipeline.PromptBlock, target: int, seed: int) -> set[str]:
    """Choose only surplus MIU rows; retains each split's balanced capacity."""
    quotas = runner.baseline_quotas(block, target, seed)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(record["split"], record["baseline_decision"]["decision"])].append(record)
    excluded: set[str] = set()
    for (split, option), rows in grouped.items():
        excess = len(rows) - quotas[split][option]
        if excess <= 0:
            continue
        # Deterministic private selection avoids retaining a prefix correlated
        # with ordinal; candidate baselines remain model-derived on regeneration.
        rows.sort(key=lambda row: hashlib.sha256(f"{seed}:{row['id']}:quarantine".encode()).hexdigest())
        excluded.update(row["id"] for row in rows[:excess])
    return excluded


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-dir", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--apply", action="store_true", help="Perform the reversible quarantine update")
    args = parser.parse_args()
    job = args.job_dir.resolve()
    records_path, audit_path = job / "records.jsonl", job / "records.audit.jsonl"
    manifest = json.loads((job / "job_manifest.json").read_text(encoding="utf-8"))
    records = read_jsonl(records_path)
    if len(records) < manifest["target_records"]:
        raise SystemExit("refusing: only completed jobs may be quarantined")
    report_path = job / "semantic_quality_report.json"
    if not report_path.exists():
        raise SystemExit("refusing: semantic_quality_report.json is required")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    selected = selected_quality_ids(report)
    block = next(item for item in pipeline.load_prompt_blocks() if item.scenario == manifest["subscenario"])
    # Do not additionally cull healthy MIU records merely for quota balancing:
    # after a quality regeneration the model must derive its own baseline again.
    if manifest["mechanism"] == "MIU" and not selected:
        selected.update(baseline_excess(records, block, manifest["target_records"], manifest["seed"]))
    result = {
        "job_dir": str(job), "records": len(records), "selected": len(selected),
        "selected_ids": sorted(selected), "applied": args.apply,
        "reason_codes": sorted({issue["code"] for issue in report["hard_errors"] + report["warnings"] if issue["record_id"] in selected}),
    }
    if not args.apply:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    audit = read_jsonl(audit_path)
    by_id = {row["record_id"]: row for row in audit}
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    quarantine = job / "quarantine" / f"{stamp}.jsonl"
    quarantine.parent.mkdir(exist_ok=True)
    quarantined = [{"record": row, "audit": by_id.get(row["id"]), "reason": "quality_triage"} for row in records if row["id"] in selected]
    write_jsonl(quarantine, quarantined)
    retained = [row for row in records if row["id"] not in selected]
    retained_audit = [row for row in audit if row["record_id"] not in selected]
    lock_and_rewrite(records_path, retained)
    lock_and_rewrite(audit_path, retained_audit)
    # Safely remove the matching released copies under the same partition locks
    # used by the producer. Other scenario writers cannot lose their appends.
    for mechanism in ("EIL", "MIU"):
        for split in ("train", "val", "test"):
            path = args.dataset_dir / mechanism / f"{split}.jsonl"
            if path.exists():
                rows = [row for row in read_jsonl(path) if row.get("id") not in selected]
                lock_and_rewrite(path, rows)
    result["quarantine"] = str(quarantine)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
