"""Mechanism-agnostic loops for generated and saved-response test evaluation."""
from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any, Awaitable, Callable

from scripts.evaluation.common import append_rows, generate_batch, load_model, run_async

Record = dict[str, Any]
ScoreBatch = Callable[[list[str], list[Record]], Awaitable[list[dict[str, Any]]]]
PromptBuilder = Callable[[Record], list[dict[str, str]]]
RowBuilder = Callable[[Record, str, dict[str, Any]], Record]
Summarizer = Callable[[list[Record], dict[str, Any]], dict[str, Any]]


def _progress_metrics(rows: list[Record], summarize: Summarizer) -> dict[str, Any]:
    """Return the current aggregate metrics without pretending a run is final."""
    summary = summarize(rows, run={})
    keys = (
        "n_scored", "n_policy_valid", "n_valid_and_judged", "reward_mean",
        "task_utility_mean", "leakage_mean", "decision_exact_match_rate",
        "reasoning_faithfulness_mean", "policy_output_valid_rate",
    )
    return {key: summary[key] for key in keys if key in summary}


def generated_test_run(
    *, records: list[Record], checkpoint: Path, device_name: str, output_dir: Path,
    batch_size: int, max_new_tokens: int, prompt_builder: PromptBuilder,
    enable_thinking: bool | None, score_batch: ScoreBatch, row_builder: RowBuilder,
    summarize: Summarizer, run: dict[str, Any],
) -> dict[str, Any]:
    """Generate, score, and checkpoint each test batch into a new directory."""
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing evaluation directory: {output_dir}")
    output_dir.mkdir(parents=True)
    run = {**run, "started_at_unix": time.time()}
    print(json.dumps({"event": "loading_model", **run}, ensure_ascii=False), flush=True)
    model, tokenizer, device = load_model(checkpoint, device_name)
    rows: list[Record] = []
    with (output_dir / "per_sample.jsonl").open("x", encoding="utf-8") as output:
        for start in range(0, len(records), batch_size):
            batch = records[start:start + batch_size]
            responses = generate_batch(
                model, tokenizer, device, batch, prompt_builder, max_new_tokens, enable_thinking=enable_thinking,
            )
            scores = run_async(score_batch(responses, batch))
            batch_rows = [row_builder(record, response, score) for record, response, score in zip(batch, responses, scores, strict=True)]
            rows.extend(batch_rows)
            append_rows(output, batch_rows)
            print(json.dumps({
                "event": "progress", "completed": len(rows), "total": len(records),
                **_progress_metrics(rows, summarize),
            }, ensure_ascii=False), flush=True)
    run["finished_at_unix"] = time.time()
    summary = summarize(rows, run)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("FINAL_SUMMARY=" + json.dumps(summary, ensure_ascii=False), flush=True)
    return summary


def saved_response_rescore(
    *, source_rows: list[Record], records_by_id: dict[str, Record], output_dir: Path,
    batch_size: int, score_batch: ScoreBatch, row_builder: RowBuilder,
    summarize: Summarizer, run: dict[str, Any],
) -> dict[str, Any]:
    """Re-score saved answers with the active evaluator without model generation."""
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing evaluation directory: {output_dir}")
    output_dir.mkdir(parents=True)
    run = {**run, "started_at_unix": time.time()}
    rows: list[Record] = []
    with (output_dir / "per_sample.jsonl").open("x", encoding="utf-8") as output:
        for start in range(0, len(source_rows), batch_size):
            prior = source_rows[start:start + batch_size]
            batch = [records_by_id[row["id"]] for row in prior]
            scores = run_async(score_batch([row["response"] for row in prior], batch))
            batch_rows = [row_builder(record, row["response"], score) for row, record, score in zip(prior, batch, scores, strict=True)]
            rows.extend(batch_rows)
            append_rows(output, batch_rows)
            print(json.dumps({
                "event": "rescore_progress", "completed": len(rows), "total": len(source_rows),
                **_progress_metrics(rows, summarize),
            }, ensure_ascii=False), flush=True)
    run["finished_at_unix"] = time.time()
    summary = summarize(rows, run)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("FINAL_SUMMARY=" + json.dumps(summary, ensure_ascii=False), flush=True)
    return summary
