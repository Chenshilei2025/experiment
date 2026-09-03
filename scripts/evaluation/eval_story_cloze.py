#!/usr/bin/env python3
"""Evaluate ROCStories with the official Story Cloze two-choice protocol.

The official CSV has eight columns: story id, four context sentences, two
candidate endings, and the 1-based index of the correct ending.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch


def read_rows(path: Path) -> list[dict[str, object]]:
    rows = []
    if path.suffix == ".parquet":
        import pyarrow.parquet as parquet
        table = parquet.read_table(path)
        required = ["story_id", "input_sentence_1", "input_sentence_2", "input_sentence_3", "input_sentence_4", "sentence_quiz1", "sentence_quiz2", "answer_right_ending"]
        if any(name not in table.column_names for name in required):
            raise ValueError(f"official Story Cloze parquet is missing required columns: {required}")
        for index, row in enumerate(table.select(required).to_pylist()):
            rows.append({"id": row["story_id"] or str(index), "context": [row[f"input_sentence_{i}"] for i in range(1, 5)], "endings": [row["sentence_quiz1"], row["sentence_quiz2"]], "gold": int(row["answer_right_ending"])})
        if not rows:
            raise ValueError(f"no valid official Story Cloze rows in {path}")
        return rows
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle, quotechar='"', delimiter=",", quoting=csv.QUOTE_ALL, skipinitialspace=True)
        header = next(reader, None)
        for index, row in enumerate(reader):
            if len(row) != 8:
                continue
            rows.append({"id": row[0] or str(index), "context": row[1:5], "endings": row[5:7], "gold": int(row[7])})
    if not rows:
        raise ValueError(f"no valid official Story Cloze rows in {path}")
    return rows


def score_candidate(model, tokenizer, device: str, context: list[str], ending: str) -> float:
    context_text = "\n".join(context)
    messages = [{"role": "user", "content": (
        "Choose the more plausible ending for this four-sentence story.\n\n"
        + context_text + "\n\nEnding:\n"
    )}]
    prefix = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prefix_ids = tokenizer(prefix, add_special_tokens=False, return_tensors="pt").input_ids[0]
    ending_ids = tokenizer(ending, add_special_tokens=False, return_tensors="pt").input_ids[0]
    ids = torch.cat([prefix_ids, ending_ids]).unsqueeze(0).to(device)
    with torch.inference_mode():
        logits = model(input_ids=ids).logits[0, :-1]
    labels = ids[0, 1:]
    start = max(0, len(prefix_ids) - 1)
    return float(torch.log_softmax(logits[start:], dim=-1).gather(1, labels[start:, None]).sum().item())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True, help="Official Story Cloze CSV")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    rows = read_rows(args.data)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    model = AutoModelForCausalLM.from_pretrained(args.checkpoint, torch_dtype=torch.bfloat16).to(args.device).eval()
    results = []
    for row in rows:
        scores = [score_candidate(model, tokenizer, args.device, row["context"], ending) for ending in row["endings"]]
        prediction = 1 if scores[0] >= scores[1] else 2
        results.append({"id": row["id"], "gold": row["gold"], "prediction": prediction, "scores": scores, "correct": prediction == row["gold"]})
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "per_sample.jsonl").write_text("".join(json.dumps(item) + "\n" for item in results), encoding="utf-8")
    summary = {"task": "official_story_cloze", "protocol": "two_choice_conditional_log_likelihood", "checkpoint": str(args.checkpoint.resolve()), "data": str(args.data.resolve()), "n_total": len(results), "n_correct": sum(item["correct"] for item in results), "accuracy": sum(item["correct"] for item in results) / len(results)}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
