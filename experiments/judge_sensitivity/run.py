"""Rescore fixed MIU/EIL responses with each requested judge model.

This mirrors :mod:`experiments.adversary_diversity`: policy responses are held
fixed, so differences are evaluator differences rather than policy variation.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import defaultdict
import json
from pathlib import Path
import re
import time
from typing import Any

from eil.reward import aggregate_eil_metrics, severity_risk
from eil.scoring.leakage import adversary_turn, assess_leakage, evaluation_adversary_temperatures
from eil.scoring.task_utility import assess_utility
from miu.reward import compute_miu_reward_for_response
from scripts.common.api_client import ApiClientPool, ChatClient
from scripts.common.experiment_logging import capture_run_output, write_run_provenance
from scripts.evaluation import eil as eil_evaluation
from scripts.evaluation import miu as miu_evaluation


PROJECT_ROOT = Path(__file__).resolve().parents[2]
Record = dict[str, Any]


def _read_jsonl(path: Path) -> list[Record]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _parse_judge(value: str) -> tuple[str, str]:
    name, separator, prefix = value.partition("=")
    if not separator or not name or not prefix or not re.fullmatch(r"[A-Z][A-Z0-9_]*", prefix):
        raise ValueError("--judge must use NAME=OPENAI_COMPATIBLE_ENV_PREFIX")
    return name, prefix


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


async def _score_miu(responses: list[str], records: list[Record], judge: ChatClient) -> list[dict[str, Any]]:
    return await asyncio.gather(*(compute_miu_reward_for_response(response, record, judge) for response, record in zip(responses, records, strict=True)))


async def _score_eil_all_judges(
    responses: list[str], records: list[Record], judges: dict[str, ChatClient], adversary: ChatClient, concurrency: int,
) -> dict[str, list[dict[str, Any]]]:
    """Hold each four-temperature adversary inference fixed across judges."""
    semaphore = asyncio.Semaphore(concurrency)

    async def score_one(response: str, record: Record) -> dict[str, dict[str, Any]]:
        async with semaphore:
            temperatures = evaluation_adversary_temperatures()
            inferences = await asyncio.gather(*(
                adversary_turn(response, record, adversary, temperature=temperature) for temperature in temperatures
            ))
            per_judge: dict[str, dict[str, Any]] = {}
            for name, judge in judges.items():
                async def score_temperature(inference: dict[str, Any]) -> tuple[float, float]:
                    leakage, utility = await asyncio.gather(
                        assess_leakage(response, record, inference, judge),
                        assess_utility(response, record, inference, judge),
                    )
                    return severity_risk(leakage["slot_scores"], record), float(utility["task_utility"])

                branches = await asyncio.gather(*(score_temperature(inference) for inference in inferences))
                per_judge[name] = aggregate_eil_metrics(
                    [utility for _, utility in branches], [leakage for leakage, _ in branches],
                )
            return per_judge

    per_response = await asyncio.gather(*(score_one(response, record) for response, record in zip(responses, records, strict=True)))
    return {name: [scores[name] for scores in per_response] for name in judges}


def _summary(mechanism: str, rows: list[Record], judge_name: str) -> dict[str, Any]:
    if mechanism == "miu":
        return miu_evaluation.summarize(rows, {"judge": judge_name})
    return eil_evaluation.summarize(rows, {"judge": judge_name})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mechanism", choices=("miu", "eil"), required=True)
    parser.add_argument("--source-jsonl", type=Path, required=True)
    parser.add_argument("--records", type=Path)
    parser.add_argument("--judge", action="append", required=True, metavar="NAME=PREFIX")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--score-concurrency", type=int, default=2)
    args = parser.parse_args()
    if args.score_concurrency < 1:
        parser.error("--score-concurrency must be positive")
    if args.records is None:
        args.records = PROJECT_ROOT / args.mechanism / "data/dataset" / args.mechanism.upper() / "test.jsonl"
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    judges = dict(_parse_judge(item) for item in args.judge)
    if len(judges) != len(args.judge):
        parser.error("judge names must be unique")
    source = _read_jsonl(args.source_jsonl)
    records = {record["id"]: record for record in _read_jsonl(args.records)}
    if not source or any(not isinstance(row.get("response"), str) or row.get("id") not in records for row in source):
        parser.error("source rows must contain responses with IDs from --records")
    if len({row["id"] for row in source}) != len(source):
        parser.error("source rows must have unique IDs")

    args.output_dir.mkdir(parents=True)
    write_run_provenance(args.output_dir)
    with capture_run_output(args.output_dir):
        responses = [row["response"] for row in source]
        ordered_records = [records[row["id"]] for row in source]
        adversary = ApiClientPool.from_env("LOYAL_EIL_ADVERSARY") if args.mechanism == "eil" else None
        judge_clients = {name: ApiClientPool.from_env(prefix) for name, prefix in judges.items()}
        if args.mechanism == "miu":
            scores_by_judge = {name: asyncio.run(_score_miu(responses, ordered_records, judge)) for name, judge in judge_clients.items()}
        else:
            scores_by_judge = asyncio.run(_score_eil_all_judges(responses, ordered_records, judge_clients, adversary, args.score_concurrency))  # type: ignore[arg-type]
        row_builder = miu_evaluation.row if args.mechanism == "miu" else eil_evaluation.row
        results = {
            name: [row_builder(record, response, score) for record, response, score in zip(ordered_records, responses, scores, strict=True)]
            for name, scores in scores_by_judge.items()
        }

        with (args.output_dir / "per_response.jsonl").open("x", encoding="utf-8") as output:
            for index, source_row in enumerate(source):
                output.write(json.dumps({"id": source_row["id"], "response": source_row["response"], "judges": {
                    name: rows[index]["score"] for name, rows in results.items()
                }}, ensure_ascii=False) + "\n")
        summaries = {name: _summary(args.mechanism, rows, name) for name, rows in results.items()}
        metric_names = ("reasoning_faithfulness_mean", "decision_exact_match_rate") if args.mechanism == "miu" else ("task_utility_mean", "leakage_mean", "reward_mean")
        summary = {
            "experiment": "judge_sensitivity_v1", "mechanism": args.mechanism.upper(),
            "source_jsonl": str(args.source_jsonl.resolve()), "records": str(args.records.resolve()),
            "n_responses": len(source), "judge_summaries": summaries,
            "judge_ranges": {metric: {"min": min(values), "max": max(values), "range": max(values) - min(values)} for metric in metric_names
                             if (values := [float(summary[metric]) for summary in summaries.values() if summary.get(metric) is not None])},
            "eil_adversary": getattr(adversary, "model", None), "started_and_finished_at_unix": time.time(),
        }
        (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print("FINAL_SUMMARY=" + json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
