"""Evaluate a HF checkpoint on math-like or GPQA-style reasoning datasets."""
from __future__ import annotations

import argparse
import json
import string
from pathlib import Path
from typing import Any

from scripts.evaluation.common import generate_batch, load_model


MATH_INSTRUCTION = (
    "Solve this problem. Explain the reasoning briefly and put the final answer "
    'on its own last line as "Answer: <answer>".\n\n'
)
GPQA_INSTRUCTION = (
    "Answer this multiple-choice science question. Explain briefly, then put the final choice "
    'on its own last line as "Answer: <letter>".\n\n'
)


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq
        return pq.read_table(path).to_pylist()
    raise ValueError("benchmark data must be .jsonl or .parquet")


def _get(row: dict[str, Any], key: str | None) -> Any:
    if not key:
        return None
    value: Any = row
    for part in key.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


def _choices(row: dict[str, Any], choices_key: str | None) -> list[str]:
    raw = _get(row, choices_key)
    if isinstance(raw, dict):
        return [str(value) for _, value in sorted(raw.items())]
    if isinstance(raw, list):
        return [str(value) for value in raw]
    choices = []
    for letter in string.ascii_uppercase[:10]:
        value = row.get(letter) or row.get(f"choice_{letter.lower()}") or row.get(f"option_{letter.lower()}")
        if value is not None:
            choices.append(str(value))
    return choices


def _prompt_math(record: dict[str, Any]) -> list[dict[str, str]]:
    return [{"role": "user", "content": MATH_INSTRUCTION + record["question"]}]


def _prompt_gpqa(record: dict[str, Any]) -> list[dict[str, str]]:
    lines = [GPQA_INSTRUCTION + record["question"], ""]
    for letter, choice in zip(string.ascii_uppercase, record["choices"]):
        lines.append(f"{letter}. {choice}")
    return [{"role": "user", "content": "\n".join(lines)}]


def _normalize_records(args: argparse.Namespace) -> list[dict[str, Any]]:
    records = []
    for index, row in enumerate(_read_rows(args.data), 1):
        question = _get(row, args.question_key)
        label = _get(row, args.answer_key)
        if question is None or label is None:
            continue
        record = {
            "id": str(_get(row, args.id_key) or index),
            "question": str(question),
            "label": str(label),
            "raw": row,
        }
        if args.task == "gpqa":
            choices = _choices(row, args.choices_key)
            if not choices:
                continue
            record["choices"] = choices
            record["metadata"] = {
                "choices": choices,
                "correct_letter": str(label).strip().upper() if len(str(label).strip()) == 1 else None,
                "correct_answer": str(label),
            }
        records.append(record)
        if args.limit is not None and len(records) >= args.limit:
            break
    if not records:
        raise ValueError("no usable benchmark records selected")
    return records


def _score_math(response: str, label: str) -> bool:
    from math_verify import parse, verify
    prediction = parse(response)
    target = parse(label)
    return bool(prediction and target and verify(target, prediction))


def _score_gpqa(response: str, record: dict[str, Any]) -> bool:
    from slime.rollout.rm_hub.gpqa import compute_gpqa_reward
    return bool(compute_gpqa_reward(response, record["label"], metadata=record["metadata"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("math", "gpqa"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--question-key", default="problem")
    parser.add_argument("--answer-key", default="answer")
    parser.add_argument("--id-key", default="unique_id")
    parser.add_argument("--choices-key")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.batch_size < 1 or args.max_new_tokens < 1:
        parser.error("batch-size and max-new-tokens must be positive")
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    records = _normalize_records(args)
    model, tokenizer, device = load_model(args.checkpoint, args.device)
    prompt_builder = _prompt_gpqa if args.task == "gpqa" else _prompt_math
    rows = []
    args.output_dir.mkdir(parents=True)
    with (args.output_dir / "per_sample.jsonl").open("x", encoding="utf-8") as output:
        for start in range(0, len(records), args.batch_size):
            batch = records[start:start + args.batch_size]
            responses = generate_batch(
                model, tokenizer, device, batch, prompt_builder, args.max_new_tokens, enable_thinking=False,
            )
            for record, response in zip(batch, responses, strict=True):
                correct = _score_gpqa(response, record) if args.task == "gpqa" else _score_math(response, record["label"])
                row = {
                    "id": record["id"],
                    "response": response,
                    "target_answer": record["label"],
                    "correct": correct,
                }
                rows.append(row)
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
            output.flush()
    n_correct = sum(bool(row["correct"]) for row in rows)
    summary = {
        "task": args.task,
        "checkpoint": str(args.checkpoint.resolve()),
        "data": str(args.data.resolve()),
        "n_total": len(rows),
        "n_correct": n_correct,
        "accuracy": n_correct / len(rows),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
