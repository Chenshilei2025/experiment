#!/usr/bin/env python3
"""Validate the accepted final records currently present in six dataset splits."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from ..generation import builder as pipeline


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, help="Optional JSON status report path")
    args = parser.parse_args()

    blocks = {pipeline.slugify(block.scenario): block for block in pipeline.load_prompt_blocks()}
    seen: set[str] = set()
    errors: list[str] = []
    counts: Counter[str] = Counter()
    scenario_counts: Counter[str] = Counter()
    for mechanism in ("EIL", "MIU"):
        for split in ("train", "val", "test"):
            path = args.dataset_dir / mechanism / f"{split}.jsonl"
            if not path.exists():
                continue
            for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    record_id = record["id"]
                    if record_id in seen:
                        raise ValueError("duplicate ID across final partitions")
                    seen.add(record_id)
                    if record.get("mechanism") != mechanism or record.get("split") != split:
                        raise ValueError("record is in the wrong final partition")
                    pipeline.validate_final_record(record, blocks[record["subscenario"]])
                    counts[f"{mechanism}/{split}"] += 1
                    scenario_counts[record["subscenario"]] += 1
                except Exception as error:
                    errors.append(f"{path}:{line_number}: {error}")
    report = {
        "records": sum(counts.values()),
        "unique_ids": len(seen),
        "partition_counts": dict(sorted(counts.items())),
        "subscenario_counts": dict(sorted(scenario_counts.items())),
        "errors": errors,
    }
    output = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(output, encoding="utf-8")
    print(output, end="")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
