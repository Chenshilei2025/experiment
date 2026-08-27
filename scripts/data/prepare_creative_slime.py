"""Convert WritingPrompts/ROCStories records into SLIME SFT messages.

The output is a Parquet file with a single ``messages`` column.  Each row is a
complete user/assistant conversation, so ``slime.rollout.sft_rollout`` can
build an assistant-only loss mask from the chat template.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.parquet as pq


WRITING_PROMPTS_INSTRUCTION = (
    "Write a vivid, coherent short story inspired by the following prompt. "
    "Preserve the central premise, develop concrete scenes, and end with a satisfying resolution.\n\n"
)
ROCSTORIES_INSTRUCTION = (
    "Continue the following story setup with a coherent, natural ending. "
    "Keep the continuation grounded in the given setup.\n\n"
)


def _clean_text(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.replace("\x00", " ").split())


def _read_parquet_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return pq.read_table(path).to_pylist()


def _writingprompts_messages(row: dict[str, Any]) -> list[dict[str, str]] | None:
    prompt = _clean_text(row.get("prompt"))
    story = _clean_text(row.get("story"))
    if not prompt or not story:
        return None
    return [
        {"role": "user", "content": WRITING_PROMPTS_INSTRUCTION + prompt},
        {"role": "assistant", "content": story},
    ]


def _rocstories_messages(row: dict[str, Any]) -> list[dict[str, str]] | None:
    prompt = _clean_text(row.get("prompt"))
    continuation = _clean_text(row.get("continuation"))
    if not continuation:
        text = _clean_text(row.get("text"))
        if prompt and text.startswith(prompt):
            continuation = text[len(prompt):].strip()
        elif text:
            continuation = text
    if not prompt or not continuation:
        return None
    return [
        {"role": "user", "content": ROCSTORIES_INSTRUCTION + prompt},
        {"role": "assistant", "content": continuation},
    ]


def _limited(rows: Iterable[dict[str, Any]], limit: int | None) -> list[dict[str, Any]]:
    if limit is None:
        return list(rows)
    if limit < 1:
        raise ValueError("per-dataset limits must be positive")
    return list(rows)[:limit]


def build_creative_file(
    *,
    writingprompts: Path | None,
    rocstories: Path | None,
    output: Path,
    seed: int,
    writingprompts_limit: int | None = None,
    rocstories_limit: int | None = None,
) -> dict[str, Any]:
    if writingprompts is None and rocstories is None:
        raise ValueError("at least one source dataset is required")
    if seed < 0:
        raise ValueError("seed must be non-negative")
    rows: list[dict[str, Any]] = []
    skipped = {"writingprompts": 0, "rocstories": 0}
    counts = {"writingprompts": 0, "rocstories": 0}
    if writingprompts is not None:
        for row in _limited(_read_parquet_rows(writingprompts), writingprompts_limit):
            messages = _writingprompts_messages(row)
            if messages is None:
                skipped["writingprompts"] += 1
                continue
            rows.append({"dataset": "writingprompts", "messages": messages})
            counts["writingprompts"] += 1
    if rocstories is not None:
        for row in _limited(_read_parquet_rows(rocstories), rocstories_limit):
            messages = _rocstories_messages(row)
            if messages is None:
                skipped["rocstories"] += 1
                continue
            rows.append({"dataset": "rocstories", "messages": messages})
            counts["rocstories"] += 1
    if not rows:
        raise ValueError("no usable creative SFT rows were produced")
    random.Random(seed).shuffle(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), output, compression="zstd")
    return {"path": str(output), "n_total": len(rows), "by_dataset": counts, "skipped": skipped, "seed": seed}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--writingprompts", type=Path, help="Parquet file with prompt/story columns")
    parser.add_argument("--rocstories", type=Path, help="Parquet file with prompt/continuation columns")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--writingprompts-limit", type=int)
    parser.add_argument("--rocstories-limit", type=int)
    args = parser.parse_args()
    summary = build_creative_file(
        writingprompts=args.writingprompts,
        rocstories=args.rocstories,
        output=args.output,
        seed=args.seed,
        writingprompts_limit=args.writingprompts_limit,
        rocstories_limit=args.rocstories_limit,
    )
    print(summary)


if __name__ == "__main__":
    main()
