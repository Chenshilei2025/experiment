#!/usr/bin/env python3
"""Run one Loyal Agent subscenario as one resumable server job.

This is the only production entry point. One job directory contains one exact
prompt.md subscenario, its scenario-specific source policy, its final-schema
records, and private audit data. The runner never moves to another subscenario.
"""

from __future__ import annotations

import argparse
from collections import Counter, deque
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
import fcntl
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from . import builder as pipeline


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
DEFAULT_MODEL = pipeline.DEFAULT_MODEL


class RetryableRecordError(RuntimeError):
    """A validated record generation failed and can be retried later."""


def load_block(subscenario: str) -> pipeline.PromptBlock:
    matches = [block for block in pipeline.load_prompt_blocks() if block.scenario == subscenario]
    if not matches:
        available = ", ".join(block.scenario for block in pipeline.load_prompt_blocks())
        raise ValueError(f"Unknown subscenario {subscenario!r}. Available: {available}")
    return matches[0]


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_env_file(path: Path) -> None:
    """Load the explicitly selected local server configuration without logging it."""
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"'")
        if key:
            os.environ[key] = value


def append_dataset_record(dataset_dir: Path, record: dict, seen_ids: set[str]) -> None:
    """Append a final record once, safely even if another job is finishing."""
    record_id = record["id"]
    if record_id in seen_ids:
        return
    destination = dataset_dir / record["mechanism"] / f"{record['split']}.jsonl"
    lock_path = destination.with_suffix(destination.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            existing_ids = pipeline.completed_ids(destination)
            if record_id not in existing_ids:
                pipeline.append_jsonl(destination, record)
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    seen_ids.add(record_id)


def dataset_ids(dataset_dir: Path) -> set[str]:
    """Read existing final IDs so resumed scenario jobs cannot duplicate a split."""
    ids = set()
    for mechanism in ("EIL", "MIU"):
        for split in ("train", "val", "test"):
            ids.update(pipeline.completed_ids(dataset_dir / mechanism / f"{split}.jsonl"))
    return ids


def baseline_quotas(block: pipeline.PromptBlock, count: int, seed: int, ordinal_start: int = 1) -> dict[str, dict[str, int]]:
    """Set private per-split capacity limits without assigning any record an option.

    The model derives every MIU baseline from its clean evidence.  These caps are
    checked only after generation, so no ordinal-to-option pattern enters a
    model prompt or the released record content.
    """
    quotas = {split: {option: 0 for option in ("opt_1", "opt_2", "opt_3")} for split in ("train", "val", "test")}
    split_totals = Counter(
        pipeline.split_for(f"{pipeline.slugify(block.scenario)}-{ordinal:05d}")
        for ordinal in range(ordinal_start, ordinal_start + count)
    )
    for split, total in split_totals.items():
        base, remainder = divmod(total, 3)
        for option in quotas[split]:
            quotas[split][option] = base
        # Only which capacity gets the unavoidable remainder is randomized;
        # no record is paired with a particular option.
        ordered = sorted(
            quotas[split],
            key=lambda option: pipeline.stable_seed(seed, f"{block.scenario}:{split}:{option}", 0),
        )
        for option in ordered[:remainder]:
            quotas[split][option] += 1
    return quotas


def baseline_counts(records_path: Path) -> dict[str, Counter[str]]:
    counts = {split: Counter() for split in ("train", "val", "test")}
    if not records_path.exists():
        return counts
    for line in records_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("mechanism") == "MIU":
            counts[record["split"]][record["baseline_decision"]["decision"]] += 1
    return counts


def request_shingles(record: dict) -> set[str]:
    words = pipeline.user_request_tokens(record.get("user_natural_language", ""))
    return {" ".join(words[index:index + 5]) for index in range(max(0, len(words) - 4))}


def too_similar_to_accepted(final: dict, accepted_requests: dict[str, set[str]]) -> bool:
    """Prevent a newly generated request from reusing an accepted surface shell.

    Comparison is within one subscenario job only.  It is deliberately a
    narrow, high-confidence gate: duplicate 5-word openings or a 5-gram
    Jaccard similarity above .48 trigger a whole-record retry.
    """
    words = pipeline.user_request_tokens(final["user_natural_language"])
    prefix = " ".join(words[:5])
    if prefix and prefix in accepted_requests["prefixes"]:
        return True
    shingles = request_shingles(final)
    for prior in accepted_requests["shingles"]:
        union = shingles | prior
        if union and len(shingles & prior) / len(union) >= 0.48:
            return True
    return False


def accepted_request_index(records_path: Path) -> dict[str, set]:
    result: dict[str, set] = {"prefixes": set(), "shingles": set()}
    if not records_path.exists():
        return result
    for line in records_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        words = pipeline.user_request_tokens(record.get("user_natural_language", ""))
        if len(words) >= 5:
            result["prefixes"].add(" ".join(words[:5]))
        shingles = request_shingles(record)
        if shingles:
            result["shingles"].add(frozenset(shingles))
    return result


def job_manifest(block: pipeline.PromptBlock, count: int, args: argparse.Namespace) -> dict[str, object]:
    return {
        "format": "loyal-agent-subscenario-job-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "subscenario": block.scenario,
        "prompt_index": block.index,
        "mechanism": "EIL" if block.family == "delegated" else "MIU",
        "family_domain": pipeline.family_domain(block),
        "target_records": count,
        "ordinal_start": args.ordinal_start,
        "source_allowlist": list(block.sources),
        "source_policy": "Only this subscenario allowlist is extracted locally; empty means None/controlled synthesis.",
        "prompt_sha256": hashlib.sha256(block.text.encode()).hexdigest(),
        "model": args.model,
        "seed": args.seed,
        "model_calls_per_record": 1,
        "prompt_field_order": (
            ["necessary_information", "exploitable_slots", "user_natural_language", "adversary_opening", "adversary_config"]
            if block.family == "delegated"
            else ["user_constraints", "user_preferences", "authorized_information", "decision_boundary", "user_natural_language", "clean_context", "baseline_decision", "manipulated_context"]
        ),
        "final_storage": (
            "records.jsonl conforms to final.md and is private/resumable; each completed "
            "record is also appended to dataset_dir/{EIL|MIU}/{train|val|test}.jsonl. "
            "records.audit.jsonl is private and never released."
        ),
    }


def append_log(path: Path, message: str) -> None:
    stamp = datetime.now(timezone.utc).isoformat()
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"{stamp} {message}\n")


def is_permanent_api_error(error: Exception) -> bool:
    """Do not spin forever on invalid credentials, requests, or model access."""
    text = str(error).lower()
    status = getattr(error, "status_code", None)
    return "error code: 401" in text or "invalid token" in text or type(error).__name__ in {
        "AuthenticationError", "PermissionDeniedError", "BadRequestError", "NotFoundError",
    } or (
        isinstance(status, int) and 400 <= status < 500 and status != 429
    )


def make_anthropic_client(timeout: float) -> Any:
    """Import the optional generation SDK only when a model call is requested."""
    try:
        from anthropic import Anthropic
    except ImportError as exc:
        raise RuntimeError(
            "Generation requires the optional 'anthropic' package. Install "
            "data_pipeline/requirements.txt before running without --dry-run."
        ) from exc
    return Anthropic(timeout=timeout, max_retries=0)


def generate_record(client: Any, block: pipeline.PromptBlock, ordinal: int, args: argparse.Namespace) -> tuple[dict, dict]:
    """Generate one ordinal once so invalid records do not monopolize a worker."""
    record_id = f"{pipeline.slugify(block.scenario)}-{ordinal:05d}"
    try:
        return pipeline.generate_one(client, block, ordinal, args)
    except Exception as error:
        if is_permanent_api_error(error):
            raise RuntimeError(f"Cannot generate {record_id}: non-retryable API error: {error}") from error
        raise RetryableRecordError(str(error)) from error


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subscenario", required=True, help="One exact prompt.md subscenario name")
    parser.add_argument("--job-dir", type=Path, required=True, help="Dedicated private run directory for this one subscenario")
    parser.add_argument("--dataset-dir", type=Path, required=True, help="Final storage root containing EIL/{train,val,test}.jsonl and MIU/{train,val,test}.jsonl")
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env", help="Local API configuration; this explicit file takes precedence")
    parser.add_argument("--api-key-env", help="Environment variable holding the API key for this isolated scenario worker")
    parser.add_argument("--allow-prompt-update", action="store_true", help="Continue a resumable job after an intentional prompt-contract update")
    parser.add_argument(
        "--allow-request-template-repeat", action="store_true",
        help="Allow repeated request wording when it will be rewritten separately.",
    )
    parser.add_argument("--count", type=int, help="Records to produce; defaults to this prompt's stated target")
    parser.add_argument(
        "--ordinal-start", type=int, default=1,
        help="First positive ordinal. Values after the prompt target create a separate extension dataset.",
    )
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--model", help="Model name; defaults to CLAUDE_MODEL after loading --env-file")
    parser.add_argument("--max-tokens", type=int, default=1800)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-attempts", type=int, default=3, help="Complete one-call record retries after an invalid/API response")
    parser.add_argument("--record-retry-delay", type=int, default=60, help="Seconds before retrying an unfinished record")
    parser.add_argument("--request-timeout", type=float, default=90, help="Per-request API timeout in seconds")
    parser.add_argument("--workers", type=int, default=1, help="Independent record calls within this one subscenario")
    parser.add_argument("--dry-run", action="store_true", help="Write frozen inputs only; never call Claude")
    args = parser.parse_args()
    load_env_file(args.env_file)
    if args.api_key_env:
        selected_key = os.getenv(args.api_key_env)
        if not selected_key:
            parser.error(f"--api-key-env {args.api_key_env!r} is unset or empty")
        # Keep one client credential per scenario worker; do not expose it in
        # manifests, logs, or model prompts.
        os.environ["ANTHROPIC_API_KEY"] = selected_key
    if args.model is None:
        args.model = os.getenv("CLAUDE_MODEL", DEFAULT_MODEL)
    if not 0 <= args.temperature <= 1:
        parser.error("--temperature must be between 0 and 1")
    if args.record_retry_delay < 1:
        parser.error("--record-retry-delay must be positive")
    if args.request_timeout <= 0:
        parser.error("--request-timeout must be positive")
    if args.workers < 1:
        parser.error("--workers must be positive")

    block = load_block(args.subscenario)
    count = args.count or block.target_count
    if args.ordinal_start < 1:
        parser.error("--ordinal-start must be positive")
    if count < 1:
        parser.error("--count must be positive")
    if args.ordinal_start == 1 and count > block.target_count:
        parser.error(
            f"--count may exceed the prompt target ({block.target_count}) only with an explicit --ordinal-start "
            "for a separate extension dataset"
        )
    job_dir = args.job_dir.resolve()
    job_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = job_dir / "job_manifest.json"
    expected_manifest = job_manifest(block, count, args)
    if manifest_path.exists():
        prior = json.loads(manifest_path.read_text(encoding="utf-8"))
        for key in ("subscenario", "target_records", "ordinal_start", "seed", "model"):
            if prior.get(key) != expected_manifest[key]:
                parser.error(f"Existing job manifest differs at {key}; use a new --job-dir")
        if prior.get("prompt_sha256") != expected_manifest["prompt_sha256"]:
            if not args.allow_prompt_update:
                parser.error("Existing job prompt differs; use --allow-prompt-update for an intentional contract repair")
            prior["prompt_sha256"] = expected_manifest["prompt_sha256"]
            prior["prompt_updated_at"] = datetime.now(timezone.utc).isoformat()
            write_json(manifest_path, prior)
    else:
        write_json(manifest_path, expected_manifest)

    records_path = job_dir / "records.jsonl"
    audit_path = job_dir / "records.audit.jsonl"
    log_path = job_dir / "generation.log"
    args.generation_log = str(log_path)
    if manifest_path.exists() and args.allow_prompt_update:
        append_log(log_path, "resumed with intentional prompt-contract update")
    done = pipeline.completed_ids(records_path)
    accepted_requests = accepted_request_index(records_path)
    final_ids = dataset_ids(args.dataset_dir.resolve())
    # Repair a previous interruption between job and final-split appends.
    if records_path.exists() and not args.dry_run:
        for line in records_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                append_dataset_record(args.dataset_dir.resolve(), json.loads(line), final_ids)
    # A stalled upstream request must become a retryable record failure, never
    # block the active subscenario indefinitely.
    client = None if args.dry_run else make_anthropic_client(args.request_timeout)

    pending = [
        ordinal for ordinal in range(args.ordinal_start, args.ordinal_start + count)
        if f"{pipeline.slugify(block.scenario)}-{ordinal:05d}" not in done
    ]
    if args.dry_run:
        for ordinal in pending:
            record_id = f"{pipeline.slugify(block.scenario)}-{ordinal:05d}"
            frozen = pipeline.build_inputs(block, ordinal, args.seed)
            pipeline.append_jsonl(records_path, {
                "_generation": {"record_id": record_id, "subscenario": block.scenario, "dry_run": True},
                "input": frozen,
            })
            append_log(log_path, f"completed record_id={record_id} dry_run=true")
            print(f"completed {record_id}", flush=True)
    else:
        # Worker calls are independent. Commit each accepted result immediately:
        # a retried early ordinal must not stall unrelated validated replacements.
        # Stable IDs, partition locks, and audit rows provide resumability.
        queued = deque(pending)
        deferred: deque[int] = deque()
        retry_cycles: Counter[int] = Counter()
        futures: dict[Future[tuple[dict, dict]], int] = {}
        with ThreadPoolExecutor(max_workers=min(args.workers, len(pending) or 1)) as executor:
            def submit_ordinal(ordinal: int) -> None:
                futures[executor.submit(
                    generate_record, make_anthropic_client(args.request_timeout), block, ordinal, args,
                )] = ordinal

            def submit_next() -> bool:
                if queued:
                    ordinal = queued.popleft()
                elif deferred:
                    ordinal = deferred.popleft()
                else:
                    return False
                submit_ordinal(ordinal)
                return True

            for _ in range(min(args.workers, len(pending))):
                submit_next()
            while futures:
                ready, _ = wait(futures, return_when=FIRST_COMPLETED)
                for future in ready:
                    ordinal = futures.pop(future)
                    try:
                        final, audit = future.result()
                    except RetryableRecordError as error:
                        retry_cycles[ordinal] += 1
                        record_id = f"{pipeline.slugify(block.scenario)}-{ordinal:05d}"
                        append_log(
                            log_path,
                            f"deferred record_id={record_id} retry={retry_cycles[ordinal]} error={error}",
                        )
                        print(f"deferred {record_id} retry={retry_cycles[ordinal]}", flush=True)
                        # Give unseen ordinals priority; only then revisit failures.
                        deferred.append(ordinal)
                        submit_next()
                        continue
                    except Exception as error:
                        append_log(log_path, f"failed record_id={pipeline.slugify(block.scenario)}-{ordinal:05d} error={error}")
                        for remaining in futures:
                            remaining.cancel()
                        raise
                    submit_next()
                    record_id = final["id"]
                    if not args.allow_request_template_repeat and too_similar_to_accepted(final, accepted_requests):
                        append_log(log_path, f"retry record_id={record_id} reason=user_request_template")
                        submit_ordinal(ordinal)
                        continue
                    pipeline.append_jsonl(records_path, final)
                    pipeline.append_jsonl(audit_path, audit)
                    append_dataset_record(args.dataset_dir.resolve(), final, final_ids)
                    words = pipeline.user_request_tokens(final["user_natural_language"])
                    if len(words) >= 5:
                        accepted_requests["prefixes"].add(" ".join(words[:5]))
                    accepted_requests["shingles"].add(frozenset(request_shingles(final)))
                    append_log(log_path, f"completed record_id={record_id}")
                    print(f"completed {record_id}", flush=True)
    append_log(log_path, f"completed subscenario={block.scenario} records={count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
