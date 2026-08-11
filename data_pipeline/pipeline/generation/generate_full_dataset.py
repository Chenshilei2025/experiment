#!/usr/bin/env python3
"""Run all 42 full targets sequentially, with bounded per-scenario concurrency.

The wrapper starts the next subscenario only after the current one has finished.
Within one subscenario, --workers independently makes one full-record model call
per ordinal; run_subscenario.py serializes all accepted writes and preserves its
private audit trail.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from . import builder as pipeline


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True, help="Private run-directory name shared by the 42 scenario jobs")
    parser.add_argument("--dataset-dir", type=Path, default=DATA / "dataset", help="Final six-partition storage root")
    parser.add_argument("--workers", type=int, default=2, help="Independent record workers within the active subscenario")
    parser.add_argument("--request-timeout", type=float, default=90, help="Per-request API timeout passed to each worker")
    parser.add_argument(
        "--record-retry-delay",
        type=int,
        default=60,
        help="Seconds to wait before retrying one unfinished record",
    )
    parser.add_argument("--max-tokens", type=int, default=1800, help="Maximum tokens for each complete-record model response")
    parser.add_argument("--temperature", type=float, default=0.7, help="Model temperature for each complete-record response")
    parser.add_argument("--api-key-env", help="Credential variable assigned to this scenario shard")
    parser.add_argument("--allow-prompt-update", action="store_true", help="Pass intentional prompt-contract updates to resumable jobs")
    parser.add_argument("--start-index", type=int, default=1, help="First 1-based prompt index in this shard")
    parser.add_argument("--end-index", type=int, default=42, help="Last inclusive 1-based prompt index in this shard")
    parser.add_argument("--step", type=int, default=1, help="Prompt-index stride for this shard")
    parser.add_argument("--model", help="Optional model override for all scenario jobs")
    parser.add_argument("--quality-report", type=Path, help="Write refreshed quality status after every finished subscenario")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    if not 1 <= args.start_index <= 42:
        parser.error("--start-index must be between 1 and 42")
    if not args.start_index <= args.end_index <= 42:
        parser.error("--end-index must be between --start-index and 42")
    if args.step < 1:
        parser.error("--step must be positive")
    if args.record_retry_delay < 1:
        parser.error("--record-retry-delay must be positive")

    for block in pipeline.load_prompt_blocks():
        if block.index < args.start_index or block.index > args.end_index or (block.index - args.start_index) % args.step:
            continue
        command = [
            sys.executable, "-m", "pipeline.generation.run_subscenario",
            "--subscenario", block.scenario,
            "--job-dir", str(DATA / "runs" / pipeline.slugify(block.scenario) / args.run_id),
            "--dataset-dir", str(args.dataset_dir),
            "--workers", str(args.workers),
            "--request-timeout", str(args.request_timeout),
            "--record-retry-delay", str(args.record_retry_delay),
            "--max-tokens", str(args.max_tokens),
            "--temperature", str(args.temperature),
        ]
        if args.model:
            command.extend(["--model", args.model])
        if args.api_key_env:
            command.extend(["--api-key-env", args.api_key_env])
        if args.allow_prompt_update:
            command.append("--allow-prompt-update")
        print(f"[{block.index:02d}/42] {block.scenario}: target={block.target_count}, workers={args.workers}", flush=True)
        subprocess.run(command, cwd=ROOT, check=True)
        diversity_command = [
            sys.executable, "-m", "pipeline.validation.check_subscenario_diversity",
            "--records", str(DATA / "runs" / pipeline.slugify(block.scenario) / args.run_id / "records.jsonl"),
            "--audit", str(DATA / "runs" / pipeline.slugify(block.scenario) / args.run_id / "records.audit.jsonl"),
            "--require-complete",
            "--report", str(DATA / "runs" / pipeline.slugify(block.scenario) / args.run_id / "diversity_report.json"),
        ]
        subprocess.run(diversity_command, cwd=ROOT, check=True)
        semantic_command = [
            sys.executable, "-m", "pipeline.validation.check_semantic_quality",
            "--records", str(DATA / "runs" / pipeline.slugify(block.scenario) / args.run_id / "records.jsonl"),
            "--report", str(DATA / "runs" / pipeline.slugify(block.scenario) / args.run_id / "semantic_quality_report.json"),
        ]
        subprocess.run(semantic_command, cwd=ROOT, check=True)
        quality_command = [sys.executable, "-m", "pipeline.validation.check_generated_dataset", "--dataset-dir", str(args.dataset_dir)]
        if args.quality_report:
            quality_command.extend(["--report", str(args.quality_report)])
        subprocess.run(quality_command, cwd=ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
