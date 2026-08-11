#!/usr/bin/env python3
"""Rebuild private per-scenario records and source audits from final JSONL data.

This is intended only after loss of ``data/runs``.  It reconstructs the
deterministic source packet and diversity profile from each persisted ID and
seed.  It deliberately labels every audit row as reconstructed: historical API
attempt counts, timestamps, and the exact pre-loss prompt revision cannot be
recovered from final records.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..generation import builder as pipeline


def read_final_records(dataset_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for mechanism in ("EIL", "MIU"):
        for split in ("train", "val", "test"):
            path = dataset_dir / mechanism / f"{split}.jsonl"
            if path.exists():
                records.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    return records


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--runs-dir", type=Path, default=Path(__file__).resolve().parents[2] / "data" / "runs")
    parser.add_argument("--seed", type=int, default=20260801)
    args = parser.parse_args()

    blocks = {pipeline.slugify(block.scenario): block for block in pipeline.load_prompt_blocks()}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in read_final_records(args.dataset_dir):
        grouped[record["subscenario"]].append(record)

    recovered_at = datetime.now(timezone.utc).isoformat()
    summary: dict[str, Any] = {"run_id": args.run_id, "reconstructed_at": recovered_at, "subscenarios": {}}
    for slug, records in sorted(grouped.items()):
        block = blocks[slug]
        records.sort(key=lambda item: int(item["id"].rsplit("-", 1)[1]))
        job_dir = args.runs_dir / slug / args.run_id
        audits: list[dict[str, Any]] = []
        for record in records:
            ordinal = int(record["id"].rsplit("-", 1)[1])
            inputs = pipeline.build_inputs(block, ordinal, args.seed)
            audit: dict[str, Any] = {
                "record_id": record["id"],
                "scenario": block.scenario,
                "prompt_index": block.index,
                "ordinal": ordinal,
                "seed": inputs["seed"],
                "prompt_sha256": hashlib.sha256(block.text.encode()).hexdigest(),
                "source_packet": inputs["frozen_source_packet"],
                "generation_profile": inputs["generation_profile"],
                "audit_status": "reconstructed_after_runs_directory_loss",
                "reconstructed_at": recovered_at,
                "unrecoverable_original_fields": ["attempt", "original_prompt_sha256", "original_generation_timestamp"],
            }
            audits.append(audit)

        manifest = {
            "format": "loyal-agent-subscenario-job-v1",
            "reconstructed_after_runs_directory_loss": True,
            "reconstructed_at": recovered_at,
            "subscenario": block.scenario,
            "prompt_index": block.index,
            "mechanism": "EIL" if block.family == "delegated" else "MIU",
            "target_records": block.target_count,
            "recovered_records": len(records),
            "source_allowlist": list(block.sources),
            "seed": args.seed,
            "model": os.getenv("CLAUDE_MODEL", "claude-opus-4-5-20251101"),
            "current_prompt_sha256": hashlib.sha256(block.text.encode()).hexdigest(),
            "caveat": "Source packets and profiles are deterministically reconstructed; historical API attempts and pre-loss prompt hashes are unavailable.",
        }
        job_dir.mkdir(parents=True, exist_ok=True)
        write_jsonl(job_dir / "records.jsonl", records)
        write_jsonl(job_dir / "records.audit.jsonl", audits)
        (job_dir / "job_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (job_dir / "generation.log").write_text(
            f"{recovered_at} reconstructed records and source audit after runs-directory loss records={len(records)}\n",
            encoding="utf-8",
        )
        summary["subscenarios"][slug] = len(records)

    report = args.runs_dir / args.run_id / "audit_recovery_report.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
