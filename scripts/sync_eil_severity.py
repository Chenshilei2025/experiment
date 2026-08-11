"""将原始 EIL severity 标注同步到已平衡、可能重复采样的训练集。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    args = parser.parse_args()
    labels = {}
    for line in args.source.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        labels[record["id"]] = {slot["id"]: slot["severity"] for slot in record["exploitable_slots"]}
    records = []
    for line in args.target.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        source_labels = labels[record["id"]]
        for slot in record["exploitable_slots"]:
            slot["severity"] = source_labels[slot["id"]]
        records.append(record)
    temporary = args.target.with_name(f".{args.target.name}.tmp")
    temporary.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records), encoding="utf-8")
    temporary.replace(args.target)
    print(f"已同步 {len(records)} 条平衡训练记录")


if __name__ == "__main__":
    main()
