#!/usr/bin/env python3
"""Sample a checkpoint on AIME and report per-sample accuracy and pass@k."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def read_records(path: Path) -> list[dict[str, str]]:
    import pyarrow.parquet as parquet

    rows = parquet.read_table(path, columns=["question", "answer"]).to_pylist()
    return [{"question": str(row["question"]), "answer": str(row["answer"])} for row in rows]


def answer_text(text: str) -> str:
    matches = re.findall(r"(?:Answer:|####)\s*([^\n]+)", text, flags=re.IGNORECASE)
    if matches:
        text = matches[-1]
    boxed = re.findall(r"\\boxed\{([^{}]+)\}", text)
    if boxed:
        text = boxed[-1]
    return "".join(text.replace("$", "").replace(",", "").lower().split()).rstrip(".")


def render(tokenizer: Any, question: str) -> str:
    messages = [{"role": "user", "content": (
        "Solve the following mathematics problem. Explain the reasoning and put the final "
        'answer on its own last line as "Answer: <answer>".\n\n' + question
    )}]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except (TypeError, ValueError):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--test-data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=16384)
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    records = read_records(args.test_data)
    args.output_dir.mkdir(parents=True)
    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.checkpoint, torch_dtype=torch.bfloat16).to(args.device).eval()
    total_correct = pass_at_k = 0
    with (args.output_dir / "per_sample.jsonl").open("x", encoding="utf-8") as output:
        for index, row in enumerate(records):
            encoded = tokenizer(render(tokenizer, row["question"]), return_tensors="pt", add_special_tokens=False).to(args.device)
            with torch.inference_mode():
                generated = model.generate(**encoded, do_sample=True, temperature=args.temperature, top_p=args.top_p,
                    num_return_sequences=args.num_samples, max_new_tokens=args.max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
            prompt_len = encoded["input_ids"].shape[1]
            responses = tokenizer.batch_decode(generated[:, prompt_len:], skip_special_tokens=True)
            target = answer_text(row["answer"])
            correct = [answer_text(response) == target for response in responses]
            total_correct += sum(correct)
            pass_at_k += any(correct)
            output.write(json.dumps({"index": index, "target_answer": target, "correct": correct, "pass_at_k": any(correct)}, ensure_ascii=False) + "\n")
            output.flush()
            print(f"{index + 1}/{len(records)} pass={pass_at_k} sample_correct={total_correct}", flush=True)
    summary = {"checkpoint": str(args.checkpoint.resolve()), "test_data": str(args.test_data.resolve()),
        "n_questions": len(records), "n_total": len(records), "num_samples": args.num_samples,
        "sample_accuracy": total_correct / (len(records) * args.num_samples),
        "pass_at_k": pass_at_k / len(records), "n_pass_at_k": pass_at_k,
        "temperature": args.temperature, "top_p": args.top_p, "max_new_tokens": args.max_new_tokens}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
