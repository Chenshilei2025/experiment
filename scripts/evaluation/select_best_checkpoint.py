"""Select the best checkpoint from a sequence of EIL/MIU evaluation runs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_summary(path: Path) -> dict[str, Any]:
    summary = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(summary, dict):
        raise ValueError(f"summary must be an object: {path}")
    return summary


def _score(
    summary_dir: Path,
    *,
    min_miu_valid_rate: float,
    max_eil_failure_rate: float,
    expected_miu_total: int,
    expected_eil_total: int,
) -> tuple[float, dict[str, Any], bool]:
    eil = _load_summary(summary_dir / "eil_final" / "summary.json")
    miu = _load_summary(summary_dir / "miu_final" / "summary.json")
    miu_total = int(miu.get("n_total", -1))
    eil_total = int(eil.get("n_total", -1))
    if miu_total != expected_miu_total:
        raise ValueError(f"{summary_dir}: MIU n_total={miu_total}, expected {expected_miu_total}")
    if eil_total != expected_eil_total:
        raise ValueError(f"{summary_dir}: EIL n_total={eil_total}, expected {expected_eil_total}")
    eil_reward = float(eil.get("reward_mean", float("nan")))
    miu_reward = float(miu.get("reward_mean", float("nan")))
    if eil_reward != eil_reward or miu_reward != miu_reward:  # NaN check
        raise ValueError(f"missing reward_mean in {summary_dir}")
    miu_valid = float(miu.get("policy_output_valid_rate", 0.0))
    eil_failed = int(eil.get("n_failed", eil_total))
    eil_failure_rate = eil_failed / eil_total if eil_total else 1.0
    eligible = miu_valid >= min_miu_valid_rate and eil_failure_rate <= max_eil_failure_rate
    return (eil_reward + miu_reward) / 2.0, {"eil": eil, "miu": miu}, eligible


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="evaluation root containing per-step subdirectories")
    parser.add_argument("--steps", type=int, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pattern", default="step{step}", help="subdirectory pattern under --root")
    parser.add_argument("--min-miu-valid-rate", type=float, default=0.95)
    parser.add_argument("--max-eil-failure-rate", type=float, default=0.05)
    parser.add_argument("--expected-miu-total", type=int, default=385)
    parser.add_argument("--expected-eil-total", type=int, default=656)
    args = parser.parse_args()
    rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for step in args.steps:
        summary_dir = args.root / args.pattern.format(step=step)
        score, metrics, eligible = _score(
            summary_dir,
            min_miu_valid_rate=args.min_miu_valid_rate,
            max_eil_failure_rate=args.max_eil_failure_rate,
            expected_miu_total=args.expected_miu_total,
            expected_eil_total=args.expected_eil_total,
        )
        row = {"step": step, "summary_dir": str(summary_dir), "score": score, "eligible": eligible, "metrics": metrics}
        rows.append(row)
        if eligible and (best is None or score > best["score"]):
            best = row
    if best is None:
        raise RuntimeError("no eligible checkpoint satisfied the configured EIL/MIU quality gates")
    payload = {"root": str(args.root), "rows": rows, "best": best}
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
