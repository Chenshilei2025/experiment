"""Unified commands for MIU/EIL test generation and rescoring.

Examples:
  python -m scripts.evaluation.cli miu --output-dir artifacts/evaluations/miu_run
  python -m scripts.evaluation.cli eil --output-dir artifacts/evaluations/eil_run
  python -m scripts.evaluation.cli rescore-eil --source-jsonl old/per_sample.jsonl --output-dir new_run
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
from typing import Any, Callable

from miu.prompts.loyal_agent_prompt import parse_policy_output
from scripts.evaluation import eil, miu
from scripts.evaluation.common import PROJECT_ROOT, read_records, run_async
from scripts.evaluation.runner import generated_test_run, saved_response_rescore


def _checkpoint() -> Path:
    return Path(os.environ.get("LOYAL_QWEN3_4B_HF_CHECKPOINT", "/ssd/shilei/models/Qwen3-4B"))


def _add_generation_args(parser: argparse.ArgumentParser, mechanism: str, max_new_tokens: int) -> None:
    parser.add_argument("--records", type=Path, default=PROJECT_ROOT / f"{mechanism.lower()}/data/dataset/{mechanism}/test.jsonl")
    parser.add_argument("--checkpoint", type=Path, default=_checkpoint())
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=max_new_tokens)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--disable-thinking", action="store_true", help="render Qwen prompts with thinking disabled")


def _shard_records(args: argparse.Namespace, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard-index must be in [0, num-shards), and num-shards must be positive")
    return records[args.shard_index::args.num_shards]


def _run_miu(args: argparse.Namespace) -> None:
    if args.batch_size < 1 or args.max_new_tokens < 1:
        raise ValueError("batch-size and max-new-tokens must be positive")
    generated_test_run(
        records=_shard_records(args, read_records(args.records, "MIU")), checkpoint=args.checkpoint, device_name=args.device,
        output_dir=args.output_dir, batch_size=args.batch_size, max_new_tokens=args.max_new_tokens,
        prompt_builder=miu.policy_messages, enable_thinking=False, score_batch=miu.score_batch, row_builder=miu.row,
        summarize=miu.summarize,
        run={
            "model_checkpoint": str(args.checkpoint.resolve()), "records": str(args.records.resolve()), "device": args.device,
            "decoding": {"temperature": 0.0, "do_sample": False, "max_new_tokens": args.max_new_tokens},
            "decision_metric": "baseline_option_exact_match_v1", "faithfulness_metric": "gpt_information_flow_v2",
            "shard_index": args.shard_index, "num_shards": args.num_shards,
        },
    )


def _run_eil(args: argparse.Namespace) -> None:
    if args.batch_size < 1 or args.max_new_tokens < 1 or args.score_concurrency < 1:
        raise ValueError("batch-size, max-new-tokens, and score-concurrency must be positive")
    generated_test_run(
        records=_shard_records(args, read_records(args.records, "EIL")), checkpoint=args.checkpoint, device_name=args.device,
        output_dir=args.output_dir, batch_size=args.batch_size, max_new_tokens=args.max_new_tokens,
        # EIL evaluation uses direct answers, matching the accelerated training
        # configuration; private Qwen thinking is disabled in the template.
        prompt_builder=eil.policy_messages, enable_thinking=False,
        score_batch=eil.score_batch(args.score_concurrency), row_builder=eil.row,
        summarize=eil.summarize,
        run={
            "model_checkpoint": str(args.checkpoint.resolve()), "records": str(args.records.resolve()), "device": args.device,
            "decoding": {"temperature": 0.0, "do_sample": False, "max_new_tokens": args.max_new_tokens},
            "policy_prompt": "v1",
            "thinking_enabled": False,
            "evaluation_adversary_temperatures": [0.3, 0.6, 0.8, 1.0],
            "shard_index": args.shard_index, "num_shards": args.num_shards,
        },
    )


def _run_eil_rescore(args: argparse.Namespace) -> None:
    if args.batch_size < 1 or args.score_concurrency < 1:
        raise ValueError("batch-size and score-concurrency must be positive")
    source_rows = [json.loads(line) for line in args.source_jsonl.read_text(encoding="utf-8").splitlines() if line]
    records = read_records(args.records, "EIL")
    records_by_id = {record["id"]: record for record in records}
    if not source_rows or len({item.get("id") for item in source_rows}) != len(source_rows) or any(item.get("id") not in records_by_id for item in source_rows):
        raise ValueError("source replies must have unique IDs from the supplied EIL records")
    saved_response_rescore(
        source_rows=source_rows, records_by_id=records_by_id, output_dir=args.output_dir,
        batch_size=args.batch_size, score_batch=eil.score_batch(args.score_concurrency), row_builder=eil.row, summarize=eil.summarize,
        run={
            "source_replies": str(args.source_jsonl.resolve()), "records": str(args.records.resolve()),
            "n_source_replies": len(source_rows), "decision": "responses reused; no Qwen generation performed",
            "leakage_aggregation": "equal_weight_mean_over_fixed_temperatures",
            "evaluation_adversary_temperatures": [0.3, 0.6, 0.8, 1.0],
        },
    )


def _run_miu_rescore(args: argparse.Namespace) -> None:
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    source = args.evaluation_dir / "per_sample.jsonl"
    if not source.is_file():
        raise ValueError("per_sample.jsonl must exist")
    original = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line]
    records = {record["id"]: record for record in read_records(args.records, "MIU")}
    if len(original) != len(records) or {row["id"] for row in original} != set(records):
        raise ValueError("evaluation responses and records must have the same unique IDs")
    targets = original
    if args.only_newly_valid:
        targets = []
        for row in original:
            if row["score"].get("policy_output_valid"):
                continue
            try:
                parse_policy_output(row["response"], records[row["id"]])
            except ValueError:
                continue
            targets.append(row)
    replacements: dict[str, dict[str, Any]] = {}
    for start in range(0, len(targets), args.batch_size):
        batch_rows = targets[start:start + args.batch_size]
        scores = run_async(miu.score_batch([row["response"] for row in batch_rows], [records[row["id"]] for row in batch_rows]))
        replacements.update({row["id"]: {**row, "score": score} for row, score in zip(batch_rows, scores, strict=True)})
        print(json.dumps({"event": "rescore_progress", "completed": len(replacements), "total": len(targets)}), flush=True)
    recovered = [replacements.get(row["id"], row) for row in original]
    archived = args.evaluation_dir / f"per_sample_event_loop_failure_{int(time.time())}.jsonl"
    replacement = args.evaluation_dir / "per_sample.recovered.jsonl"
    replacement.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in recovered), encoding="utf-8")
    source.rename(archived)
    replacement.rename(source)
    summary_path = args.evaluation_dir / "summary.json"
    prior = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
    run = dict(prior.get("run", {}))
    run["faithfulness_rescored_at_unix"] = time.time()
    run["faithfulness_rescore_note"] = (
        "Recovered after fixing cross-event-loop client reuse; model responses unchanged."
        if not args.only_newly_valid else
        "Re-scored responses newly valid under equivalent plain E# citation syntax; model responses unchanged."
    )
    summary_path.write_text(json.dumps(miu.summarize(recovered, run=run), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("FINAL_SUMMARY=" + summary_path.read_text(encoding="utf-8").strip(), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    miu_parser = commands.add_parser("miu", help="generate and score the MIU test set")
    _add_generation_args(miu_parser, "MIU", 384)
    miu_parser.set_defaults(handler=_run_miu)
    eil_parser = commands.add_parser("eil", help="generate and score the EIL test set")
    _add_generation_args(eil_parser, "EIL", 2048)
    eil_parser.add_argument("--score-concurrency", type=int, default=4)
    eil_parser.set_defaults(handler=_run_eil)
    miu_rescore = commands.add_parser("rescore-miu", help="rescore saved MIU replies in place")
    miu_rescore.add_argument("--evaluation-dir", type=Path, required=True)
    miu_rescore.add_argument("--records", type=Path, default=PROJECT_ROOT / "miu/data/dataset/MIU/test.jsonl")
    miu_rescore.add_argument("--batch-size", type=int, default=16)
    miu_rescore.add_argument("--only-newly-valid", action="store_true")
    miu_rescore.set_defaults(handler=_run_miu_rescore)
    eil_rescore = commands.add_parser("rescore-eil", help="rescore saved EIL replies into a new directory")
    eil_rescore.add_argument("--source-jsonl", type=Path, required=True)
    eil_rescore.add_argument("--records", type=Path, default=PROJECT_ROOT / "eil/data/dataset/EIL/test.jsonl")
    eil_rescore.add_argument("--output-dir", type=Path, required=True)
    eil_rescore.add_argument("--batch-size", type=int, default=8)
    eil_rescore.add_argument("--score-concurrency", type=int, default=4)
    eil_rescore.set_defaults(handler=_run_eil_rescore)
    args = parser.parse_args()
    handler: Callable[[argparse.Namespace], None] = args.handler
    handler(args)


if __name__ == "__main__":
    main()
