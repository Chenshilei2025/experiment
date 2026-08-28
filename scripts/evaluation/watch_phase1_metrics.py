"""Watch EIL/MIU checkpoint metrics and write trend snapshots."""
from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _metric(payload: dict[str, Any], name: str) -> float | None:
    value = payload.get(name)
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if result == result else None


def _row(root: Path, step: int) -> dict[str, Any] | None:
    step_dir = root / f"step{step}"
    miu = _load_json(step_dir / "miu_final" / "summary.json")
    eil = _load_json(step_dir / "eil_final" / "summary.json")
    if miu is None or eil is None:
        return None
    miu_decision = _metric(miu, "decision_exact_match_rate")
    miu_faithfulness = _metric(miu, "reasoning_faithfulness_mean")
    eil_utility = _metric(eil, "task_utility_mean")
    eil_leakage = _metric(eil, "leakage_mean")
    required = [miu_decision, miu_faithfulness, eil_utility, eil_leakage]
    if any(value is None for value in required):
        return None
    miu_valid = _metric(miu, "policy_output_valid_rate") or 0.0
    eil_total = int(eil.get("n_total", 0) or 0)
    eil_failed = int(eil.get("n_failed", eil_total) or 0)
    eil_failure_rate = eil_failed / eil_total if eil_total else 1.0
    score = (
        float(miu_decision)
        + float(miu_faithfulness)
        + float(eil_utility)
        + (1.0 - float(eil_leakage))
    ) / 4.0
    eligible = miu_valid >= 0.95 and eil_failure_rate <= 0.05
    return {
        "step": step,
        "score": score,
        "eligible": eligible,
        "miu_policy_output_valid_rate": miu_valid,
        "miu_decision_exact_match_rate": miu_decision,
        "miu_reasoning_faithfulness_mean": miu_faithfulness,
        "miu_reward_mean": _metric(miu, "reward_mean"),
        "eil_task_utility_mean": eil_utility,
        "eil_leakage_mean": eil_leakage,
        "eil_low_leakage": 1.0 - float(eil_leakage),
        "eil_leakage_zero_rate": _metric(eil, "leakage_zero_rate"),
        "eil_reward_mean": _metric(eil, "reward_mean"),
        "eil_failure_rate": eil_failure_rate,
        "miu_n_total": int(miu.get("n_total", 0) or 0),
        "eil_n_total": eil_total,
    }


def _with_deltas(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    previous: dict[str, Any] | None = None
    output: list[dict[str, Any]] = []
    fields = [
        "score",
        "miu_reward_mean",
        "eil_reward_mean",
        "miu_decision_exact_match_rate",
        "miu_reasoning_faithfulness_mean",
        "eil_task_utility_mean",
        "eil_low_leakage",
        "eil_leakage_mean",
    ]
    for row in rows:
        item = dict(row)
        if previous is not None:
            for field in fields:
                current = item.get(field)
                old = previous.get(field)
                if current is None or old is None:
                    continue
                delta = float(current) - float(old)
                if field == "eil_leakage_mean":
                    delta = -delta
                item[f"delta_{field}"] = delta
        output.append(item)
        previous = row
    return output


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fields = [
        "step",
        "score",
        "eligible",
        "miu_policy_output_valid_rate",
        "miu_decision_exact_match_rate",
        "miu_reasoning_faithfulness_mean",
        "eil_task_utility_mean",
        "eil_leakage_mean",
        "eil_low_leakage",
        "miu_reward_mean",
        "eil_reward_mean",
        "eil_failure_rate",
        "delta_score",
        "delta_miu_reward_mean",
        "delta_eil_reward_mean",
        "delta_miu_decision_exact_match_rate",
        "delta_miu_reasoning_faithfulness_mean",
        "delta_eil_task_utility_mean",
        "delta_eil_low_leakage",
        "delta_eil_leakage_mean",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def snapshot(post_root: Path, steps: list[int]) -> dict[str, Any]:
    eval_root = post_root / "checkpoint_eval"
    rows = [row for step in steps if (row := _row(eval_root, step)) is not None]
    rows = _with_deltas(rows)
    eligible = [row for row in rows if row["eligible"]]
    best = max(eligible, key=lambda row: row["score"]) if eligible else None
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "post_root": str(post_root),
        "completed_steps": [row["step"] for row in rows],
        "n_completed": len(rows),
        "best_so_far": best,
        "rows": rows,
    }
    (post_root / "metrics_trend.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_csv(rows, post_root / "metrics_trend.csv")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--post-root", type=Path, required=True)
    parser.add_argument("--steps", type=int, nargs="+", required=True)
    parser.add_argument("--interval", type=int, default=300)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    args.post_root.mkdir(parents=True, exist_ok=True)
    while True:
        payload = snapshot(args.post_root, args.steps)
        print(json.dumps({
            "event": "metrics_snapshot",
            "n_completed": payload["n_completed"],
            "completed_steps": payload["completed_steps"],
            "best_step": None if payload["best_so_far"] is None else payload["best_so_far"]["step"],
            "best_score": None if payload["best_so_far"] is None else payload["best_so_far"]["score"],
        }, ensure_ascii=False), flush=True)
        if args.once:
            return
        acceptance = _load_json(args.post_root / "acceptance.json")
        if acceptance is not None and acceptance.get("status") in {"passed", "failed"}:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
