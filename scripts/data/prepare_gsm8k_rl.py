#!/usr/bin/env python3
"""Convert GSM8K train data to the prompt/label contract for math GRPO."""
from __future__ import annotations

import argparse
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


INSTRUCTION = (
    "Solve this GSM8K problem step by step. Put the final answer on its own last "
    'line in the exact form "Answer: <answer>".\n\n'
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    rows = pq.read_table(args.input, columns=["question", "answer"]).to_pylist()
    converted = []
    for row in rows:
        _, marker, answer = str(row["answer"]).rpartition("####")
        if not marker or not answer.strip():
            raise ValueError("GSM8K target lacks #### delimiter")
        converted.append({"prompt": INSTRUCTION + str(row["question"]), "label": answer.strip()})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(converted), args.output, compression="zstd")
    print({"path": str(args.output), "n_total": len(converted)})


if __name__ == "__main__":
    main()
