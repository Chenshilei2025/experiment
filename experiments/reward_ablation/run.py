"""Compatibility CLI for one fixed-coefficient reward-ablation condition.

The shared JSON runner now owns training execution and provenance.  This file
only translates the legacy flags into the same declarative condition format.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from scripts.experiment_runner import run_config


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EIL_LAMBDAS = (0.0, 0.5, 1.0, 2.0)
MIU_ETAS = (0.0, 0.5, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mechanism", choices=("eil", "miu"), required=True)
    parser.add_argument("--lambda", dest="leakage_lambda", type=float, choices=EIL_LAMBDAS)
    parser.add_argument("--eta", type=float, choices=MIU_ETAS)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--rollouts", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.rollouts < 1:
        parser.error("--rollouts must be positive")
    if args.mechanism == "eil" and (args.leakage_lambda is None or args.eta is not None):
        parser.error("EIL ablation requires --lambda and does not accept --eta")
    if args.mechanism == "miu" and (args.eta is None or args.leakage_lambda is not None):
        parser.error("MIU ablation requires --eta and does not accept --lambda")
    condition = f"{args.mechanism}_{'lambda' if args.mechanism == 'eil' else 'eta'}_{args.leakage_lambda if args.mechanism == 'eil' else args.eta:g}".replace(".", "_")
    environment = (
        {"LOYAL_EIL_LEAKAGE_LAMBDA": args.leakage_lambda}
        if args.mechanism == "eil" else {"LOYAL_MIU_FAITHFULNESS_ETA": args.eta}
    )
    config = {
        "version": 1,
        "experiment": "reward_ablation",
        "context": {"condition": condition},
        "checkpoint_template": "reward-ablation_{run_name}_{condition}_seed{seed}",
        "base_model": "qwen3-4b",
        "seed": args.seed,
        "environment": environment,
        "stages": [{"mechanism": args.mechanism, "rollouts": args.rollouts}],
    }
    run_config(config, output_dir=args.output_dir, run_name=args.run_name)


if __name__ == "__main__":
    main()
