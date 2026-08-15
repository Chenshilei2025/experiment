"""Run one model/mechanism/prompt-condition cell of the loyalty-CoT baseline.

Each run reuses production EIL/MIU scoring. It supports local Hugging Face
models and OpenAI-compatible hosted models.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Callable

from experiments.loyalty_cot_baseline.prompts import apply_condition
from scripts.common.api_client import ApiClient
from scripts.common.experiment_logging import capture_run_output, write_run_provenance
from scripts.evaluation import eil as eil_evaluation
from scripts.evaluation import miu as miu_evaluation


PROJECT_ROOT = Path(__file__).resolve().parents[2]
Record = dict[str, Any]
PromptBuilder = Callable[[Record], list[dict[str, str]]]


def _read_records(path: Path, mechanism: str) -> list[Record]:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not records or any(record.get("mechanism") != mechanism.upper() for record in records):
        raise ValueError(f"{path} is not a non-empty {mechanism.upper()} JSONL file")
    return records


def _seed(record: Record, condition: str) -> int:
    material = f"{record['id']}|{condition}|policy-generation".encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:4], "big") & 0x7FFFFFFF


def _policy_builder(mechanism: str) -> PromptBuilder:
    if mechanism == "miu":
        return miu_evaluation.policy_messages
    if mechanism == "eil":
        return eil_evaluation.policy_messages
    raise ValueError(f"unsupported mechanism {mechanism!r}")


async def _api_responses(
    records: list[Record], prompt_builder: PromptBuilder, condition: str,
    client: ApiClient, max_new_tokens: int, concurrency: int,
) -> list[str]:
    if client.json_mode:
        raise ValueError("LOYAL_EXPERIMENT_MODEL_JSON_MODE must be disabled for policy generation")
    # Plain is the explicit no-thinking control.  The CoT condition leaves
    # thinking enabled at the provider default while its system prompt asks for
    # the specified internal loyalty reasoning.
    client.disable_thinking = condition == "plain"
    semaphore = asyncio.Semaphore(concurrency)

    async def generate(record: Record) -> str:
        async with semaphore:
            return await client.chat_json(
                apply_condition(prompt_builder(record), condition),
                temperature=0.0, max_tokens=max_new_tokens, seed=_seed(record, condition),
            )

    return await asyncio.gather(*(generate(record) for record in records))


async def _score(mechanism: str, responses: list[str], records: list[Record], score_concurrency: int) -> list[dict[str, Any]]:
    if mechanism == "miu":
        return await miu_evaluation.score_batch(responses, records)
    return await eil_evaluation.score_batch(score_concurrency)(responses, records)


def _progress(summary: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "n_scored", "n_policy_valid", "n_valid_and_judged", "reward_mean",
        "task_utility_mean", "leakage_mean", "decision_exact_match_rate",
        "reasoning_faithfulness_mean", "policy_output_valid_rate",
    )
    return {key: summary[key] for key in keys if key in summary}


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _resume_rows(output_dir: Path, records: list[Record]) -> list[Record]:
    """Load a complete prefix of an interrupted run without duplicating IDs."""
    path = output_dir / "per_sample.jsonl"
    if not path.is_file():
        return []
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if len(rows) > len(records) or any(row.get("id") != record["id"] for row, record in zip(rows, records, strict=False)):
        raise ValueError("existing per_sample.jsonl is not an ordered prefix of the requested records")
    return rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.batch_size < 1 or args.generation_concurrency < 1 or args.score_concurrency < 1 or args.max_new_tokens < 1:
        raise ValueError("batch size, concurrencies, and max-new-tokens must be positive")
    if args.output_dir.exists() and not args.resume:
        allowed_new_files = {"run.log", "command.json", "environment.json"}
        existing = {path.name for path in args.output_dir.iterdir()}
        if existing - allowed_new_files:
            raise FileExistsError(f"refusing to overwrite experiment output {args.output_dir}")
    prompt_builder = _policy_builder(args.mechanism)
    records = _read_records(args.records, args.mechanism)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = _resume_rows(args.output_dir, records) if args.resume else []
    manifest: dict[str, Any] = {
        "experiment": "loyalty_cot_baseline_v1",
        "model_name": args.model_name,
        "backend": args.backend,
        "mechanism": args.mechanism.upper(),
        "condition": args.condition,
        "records": str(args.records.resolve()),
        "n_records": len(records),
        "decoding": {"temperature": 0.0, "do_sample": False, "max_new_tokens": args.max_new_tokens},
        "started_at_unix": time.time(),
        "resumed_from_completed": len(rows),
    }
    thinking_enabled = args.condition == "loyalty-cot"
    if args.backend == "hf":
        manifest.update({"checkpoint": str(args.checkpoint.resolve()), "device": args.device, "thinking_enabled": thinking_enabled})
        from scripts.evaluation.common import generate_batch, load_model
        model, tokenizer, torch_device = load_model(args.checkpoint, args.device)

        def generate(records_batch: list[Record]) -> list[str]:
            return generate_batch(
                model, tokenizer, torch_device, records_batch,
                lambda record: apply_condition(prompt_builder(record), args.condition),
                args.max_new_tokens, enable_thinking=thinking_enabled,
            )
    else:
        client = ApiClient.from_env("LOYAL_EXPERIMENT_MODEL")
        manifest.update({
            "api_model": client.model,
            "generation_concurrency": args.generation_concurrency,
            "thinking_enabled": thinking_enabled,
        })
        def generate(records_batch: list[Record]) -> list[str]:
            return asyncio.run(_api_responses(
                records_batch, prompt_builder, args.condition, client, args.max_new_tokens, args.generation_concurrency,
            ))
    _write_json(args.output_dir / "manifest.json", manifest)

    row_builder = miu_evaluation.row if args.mechanism == "miu" else eil_evaluation.row
    summarizer = miu_evaluation.summarize if args.mechanism == "miu" else eil_evaluation.summarize
    mode = "a" if rows else "x"
    with (args.output_dir / "per_sample.jsonl").open(mode, encoding="utf-8") as output:
        for start in range(len(rows), len(records), args.batch_size):
            batch_records = records[start:start + args.batch_size]
            # Persist after each generation-and-scoring batch. This makes long
            # CoT runs observable and lets --resume continue safely after an
            # interruption without regenerating completed responses.
            batch_responses = generate(batch_records)
            scores = asyncio.run(_score(args.mechanism, batch_responses, batch_records, args.score_concurrency))
            batch_rows = [row_builder(record, response, score) for record, response, score in zip(batch_records, batch_responses, scores, strict=True)]
            rows.extend(batch_rows)
            for row in batch_rows:
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
            output.flush()
            os.fsync(output.fileno())
            print(json.dumps({"event": "progress", "completed": len(rows), "total": len(records), **_progress(summarizer(rows, run={}))}, ensure_ascii=False), flush=True)

    manifest["finished_at_unix"] = time.time()
    _write_json(args.output_dir / "manifest.json", manifest)
    summary = summarizer(rows, run=manifest)
    _write_json(args.output_dir / "summary.json", summary)
    print("FINAL_SUMMARY=" + json.dumps(summary, ensure_ascii=False), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("hf", "api"), required=True)
    parser.add_argument("--model-name", required=True, help="model label recorded in manifest.json")
    parser.add_argument("--mechanism", choices=("eil", "miu"), required=True)
    parser.add_argument("--condition", choices=("plain", "loyalty-cot"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--records", type=Path)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--score-concurrency", type=int, default=2)
    parser.add_argument("--generation-concurrency", type=int, default=4)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true", help="continue an interrupted run from its complete JSONL prefix")
    args = parser.parse_args()
    if args.records is None:
        args.records = PROJECT_ROOT / args.mechanism / "data" / "dataset" / args.mechanism.upper() / "test.jsonl"
    if args.max_new_tokens is None:
        # CoT-enabled Qwen spends part of this budget on private reasoning
        # before emitting MIU's short structured answer.
        args.max_new_tokens = 2048 if (args.mechanism == "eil" or args.condition == "loyalty-cot") else 384
    if args.backend == "hf" and args.checkpoint is None:
        parser.error("--checkpoint is required for --backend hf")
    if args.backend == "api" and args.checkpoint is not None:
        parser.error("--checkpoint applies only to --backend hf")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_run_provenance(args.output_dir)
    with capture_run_output(args.output_dir):
        run(args)


if __name__ == "__main__":
    main()
