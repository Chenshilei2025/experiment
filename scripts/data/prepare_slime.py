"""Convert private EIL/MIU records into prompt-safe SLIME JSONL datasets."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

# Support both direct execution and ``python -m scripts.data.prepare_slime``.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eil.prompts.loyal_agent_prompt import policy_messages as eil_messages
from miu.prompts.loyal_agent_prompt import policy_messages as miu_messages


PROMPT_BUILDERS: dict[str, Callable[[dict[str, Any]], list[dict[str, str]]]] = {
    "eil": eil_messages,
    "miu": miu_messages,
}


def convert_file(mechanism: str, source: Path, destination: Path) -> int:
    """Write messages and a record ID, keeping evaluator-only fields out of prompts."""
    builder = PROMPT_BUILDERS[mechanism]
    destination.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with source.open(encoding="utf-8") as inputs, destination.open("w", encoding="utf-8") as outputs:
        for line_number, line in enumerate(inputs, 1):
            record = json.loads(line)
            expected = mechanism.upper()
            if record.get("mechanism") != expected or not isinstance(record.get("id"), str):
                raise ValueError(f"{source}:{line_number} is not a valid {expected} record")
            outputs.write(
                json.dumps({"messages": builder(record), "record_id": record["id"]}, ensure_ascii=False) + "\n"
            )
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mechanism", choices=sorted(PROMPT_BUILDERS))
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(f"wrote {convert_file(args.mechanism, args.source, args.output)} rows to {args.output}")


if __name__ == "__main__":
    main()
