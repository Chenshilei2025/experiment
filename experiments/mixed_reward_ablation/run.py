"""Run one protocol-v2 mixed reward or batch-ratio condition."""
from __future__ import annotations

import argparse
from pathlib import Path

from scripts.experiment_runner import load_config, run_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run_config(load_config(args.config), output_dir=args.output_dir, run_name=args.run_name, config_path=args.config)


if __name__ == "__main__":
    main()
