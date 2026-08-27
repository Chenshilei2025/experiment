"""Build a prompt-safe, namespaced union of MIU and EIL training records.

The file deliberately contains no reward-only record fields.  The reward
adapter reloads the canonical source datasets using the namespaced label.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eil.prompts.loyal_agent_prompt import policy_messages as eil_messages
from miu.prompts.loyal_agent_prompt import policy_messages as miu_messages


def _read(source: Path, mechanism: str) -> list[dict[str, Any]]:
    if not source.is_file():
        raise FileNotFoundError(source)
    records = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line]
    if not records or any(item.get("mechanism") != mechanism or not isinstance(item.get("id"), str) for item in records):
        raise ValueError(f"{source} is not a non-empty {mechanism} JSONL dataset")
    return records


def build_mixed_file(*, miu_source: Path, eil_source: Path, destination: Path, seed: int) -> dict[str, Any]:
    """Create a deterministic shuffled prompt pool suitable for SLIME.

    The pool's source-size ratio is deliberately *not* the training ratio.
    The mixed rollout adapter consumes it through an exact schedule of
    single-task EIL and MIU batches, so actual frequency is independent of
    unequal canonical dataset sizes.
    """
    if seed < 0:
        raise ValueError("seed must be non-negative")
    rows: list[dict[str, Any]] = []
    for mechanism, source, builder in (("MIU", miu_source, miu_messages), ("EIL", eil_source, eil_messages)):
        for record in _read(source, mechanism):
            rows.append({"mechanism": mechanism, "record_id": f"{mechanism}:{record['id']}", "messages": builder(record)})
    random.Random(seed).shuffle(rows)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {
        "seed": seed,
        "n_total": len(rows),
        "by_mechanism": {"EIL": sum(row["mechanism"] == "EIL" for row in rows), "MIU": sum(row["mechanism"] == "MIU" for row in rows)},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--miu-source", type=Path, required=True)
    parser.add_argument("--eil-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(build_mixed_file(miu_source=args.miu_source, eil_source=args.eil_source, destination=args.output, seed=args.seed)))


if __name__ == "__main__":
    main()
