"""Merge six independently generated EIL evaluation shards into one report."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.evaluation import eil


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers-dir", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    records = {
        item["id"]: item
        for item in (json.loads(line) for line in args.records.read_text(encoding="utf-8").splitlines() if line)
    }
    rows = []
    for path in sorted(args.workers_dir.glob("gpu*/per_sample.jsonl")):
        rows.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line)
    if len(rows) != len(records) or len({row["id"] for row in rows}) != len(rows) or {row["id"] for row in rows} != set(records):
        raise RuntimeError("worker outputs do not contain exactly one row for every EIL test record")
    for row in rows:
        record = records[row["id"]]
        row["family_domain"] = record["family_domain"]
    summary = eil.summarize(rows, run={
        "workers_dir": str(args.workers_dir.resolve()),
        "records": str(args.records.resolve()),
        "aggregation": "six GPU shards merged after independent generation and scoring",
    })
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("FINAL_SUMMARY=" + json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
