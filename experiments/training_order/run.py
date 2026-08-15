"""Compatibility CLI for one training-order condition.

New conditions should be committed as JSON under ``configs/`` and run through
``scripts.experiment_runner``.  This wrapper keeps the paper command line
working while delegating all launch, manifest, and provenance mechanics to the
shared runner.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from scripts.experiment_runner import run_config


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--order", choices=("miu-eil", "eil-miu"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--miu-rollouts", type=int, required=True)
    parser.add_argument("--eil-rollouts", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.seed < 0 or args.miu_rollouts < 1 or args.eil_rollouts < 1:
        parser.error("seed and rollout counts must be non-negative; run-name must be simple")
    stages = args.order.split("-")
    config = {
        "version": 1,
        "experiment": "training_order",
        "context": {"order": args.order},
        "checkpoint_template": "training-order_{run_name}_{order}_seed{seed}",
        "base_model": "qwen3-4b",
        "seed": args.seed,
        "stages": [
            {"mechanism": mechanism, "rollouts": args.miu_rollouts if mechanism == "miu" else args.eil_rollouts}
            for mechanism in stages
        ],
    }
    run_config(config, output_dir=args.output_dir, run_name=args.run_name)


if __name__ == "__main__":
    main()
