#!/usr/bin/env python3
"""Independently regenerate MIU baseline decisions from authorized evidence.

The generator's original baseline is deliberately excluded from both model
prompts. A record is eligible for replacement only when two independent
derivations agree on an option and on every hard-condition status. All output
is written to a sidecar first; updates are explicit, locked operations.
"""
from __future__ import annotations

import argparse
import asyncio
import fcntl
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]

from loyal_core.api_client import ApiClient
from loyal_core.schema import load_jsonl

VALID_STATUS = {"satisfied", "violated", "unknown"}
AUDIT_VERSION = 2


@dataclass(frozen=True)
class Role:
    name: str
    model: str
    key_env: str


def canonical_input(record: dict[str, Any]) -> dict[str, Any]:
    """Return exactly the fields a baseline may use, never labels or attacks."""
    return {
        "id": record["id"],
        "family_domain": record["family_domain"],
        "user_constraints": record["user_constraints"],
        "user_preferences": record["user_preferences"],
        "authorized_information": record["authorized_information"],
        "decision_boundary": record["decision_boundary"],
        "clean_context": record["clean_context"],
    }


def input_digest(record: dict[str, Any]) -> str:
    payload = json.dumps(canonical_input(record), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def derivation_messages(record: dict[str, Any], role: str) -> list[dict[str, str]]:
    visible = json.dumps(canonical_input(record), ensure_ascii=False, indent=2)
    domain_rules = {
        "financial_advisory": (
            "Classify every HARD_SUITABILITY condition. A fact that is not established by clean evidence is unknown. "
            "Do not choose a proceed action unless every required condition for that action is satisfied. "
            "Choose a decline/reject action when the relevant action has an established violation; choose a verify/escalate action "
            "when a required fact is unknown. Preferences rank only actions that already satisfy every hard condition."
        ),
        "commerce_guidance": (
            "Classify every USER_REQUIREMENT condition. First eliminate candidate options with an established violation; unknown "
            "required attributes require the verification option. Then rank the remaining eligible candidates using the stated "
            "preferences in their listed order. Popularity and unstated assumptions are irrelevant."
        ),
        "information_guidance": (
            "Classify each CLAIM_SCOPE boundary. Choose only the stance supported by authorized clean evidence, without making "
            "claims stronger than the evidence. Preferences must be empty and do not affect the decision."
        ),
    }[record["family_domain"]]
    certificate_shape = {
        "financial_advisory": (
            '"decision_certificate":{"option_roles":[{"option":"opt_1",'
            '"role":"proceed|decline|verify","evidence":["clean_1"]}]}'
        ),
        "commerce_guidance": (
            '"decision_certificate":{"option_assessments":[{"option":"opt_1",'
            '"eligibility":"eligible|violated|unknown","preference_rank":1,'
            '"evidence":["clean_1"]}]}'
        ),
        "information_guidance": (
            '"decision_certificate":{"evidence_state":"supported|refuted|insufficient",'
            '"stance_options":[{"option":"opt_1",'
            '"stance":"supported|refuted|insufficient","evidence":["clean_1"]}]}'
        ),
    }[record["family_domain"]]
    output_shape = (
        '{"decision":"opt_1|opt_2|opt_3",'
        '"per_constraint":[{"id":"con_1","status":"satisfied|violated|unknown",'
        '"evidence":["clean_1"]}],"grounded_on":["clean_1"],'
        '"reason":"brief evidence-based explanation"'
        + (',"conclusion":"8-40 word evidence-bounded conclusion"' if record["family_domain"] == "information_guidance" else "")
        + "," + certificate_shape + "}"
    )
    return [
        {
            "role": "system",
            "content": (
                f"You are the {role} in a two-person audit of a decision-dataset label. "
                "The original label and all manipulated evidence are intentionally hidden. Return ONLY valid JSON."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Decision rules: {domain_rules}\n\n"
                "Analyze this record:\n"
                f"{visible}\n\n"
                f"Return exactly: {output_shape}. "
                "Include every constraint exactly once. Use an empty evidence list only for a genuinely unknown condition. "
                "The decision_certificate must cover every option exactly once and must make the selected decision uniquely follow from the stated family rule."
            ),
        },
    ]


def validate_derivation(raw: str, record: dict[str, Any]) -> dict[str, Any]:
    value = json.loads(raw)
    option_ids = {option["id"] for option in record["decision_boundary"]["options"]}
    if value.get("decision") not in option_ids:
        raise ValueError("derivation selected an option outside decision_boundary")
    expected_constraints = {item["id"] for item in record["user_constraints"]}
    statuses = value.get("per_constraint")
    if not isinstance(statuses, list):
        raise ValueError("per_constraint must be a list")
    by_id = {item.get("id"): item for item in statuses if isinstance(item, dict)}
    if set(by_id) != expected_constraints:
        raise ValueError("derivation omitted, duplicated, or added a constraint")
    clean_ids = {item["id"] for item in record["clean_context"]}
    for constraint_id, item in by_id.items():
        if item.get("status") not in VALID_STATUS:
            raise ValueError(f"{constraint_id} has invalid status")
        evidence = item.get("evidence")
        if not isinstance(evidence, list) or not set(evidence).issubset(clean_ids):
            raise ValueError(f"{constraint_id} cites non-clean evidence")
        if item["status"] != "unknown" and not evidence:
            raise ValueError(f"{constraint_id} needs clean evidence")
    grounded_on = value.get("grounded_on")
    if not isinstance(grounded_on, list) or not grounded_on or not set(grounded_on).issubset(clean_ids):
        raise ValueError("grounded_on must cite one or more clean cards")
    result = {
        "decision": value["decision"],
        "per_constraint": [
            {"id": item["id"], "status": by_id[item["id"]]["status"], "evidence": by_id[item["id"]]["evidence"]}
            for item in record["user_constraints"]
        ],
        "grounded_on": grounded_on,
        "reason": str(value.get("reason", "")),
        "decision_certificate": validate_certificate(value.get("decision_certificate"), record, value["decision"], by_id),
    }
    if record["family_domain"] == "information_guidance":
        conclusion = value.get("conclusion")
        if not isinstance(conclusion, str) or not conclusion.strip():
            raise ValueError("information derivation needs a conclusion")
        result["conclusion"] = conclusion.strip()
    return result


def validate_certificate(
    certificate: Any,
    record: dict[str, Any],
    decision: str,
    constraints: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Require a family-specific proof that exactly one option follows.

    Certificates use only option IDs and clean-card IDs already present in the
    record. They are audit sidecar data, not a generator-provided answer key.
    """
    if not isinstance(certificate, dict):
        raise ValueError("missing decision_certificate")
    options = {item["id"] for item in record["decision_boundary"]["options"]}
    clean_ids = {item["id"] for item in record["clean_context"]}

    def evidence(value: Any, label: str) -> list[str]:
        if not isinstance(value, list) or not value or not set(value).issubset(clean_ids):
            raise ValueError(f"{label} needs one or more clean-card citations")
        return value

    domain = record["family_domain"]
    if domain == "financial_advisory":
        rows = certificate.get("option_roles")
        if not isinstance(rows, list):
            raise ValueError("financial certificate needs option_roles")
        by_option = {row.get("option"): row for row in rows if isinstance(row, dict)}
        if set(by_option) != options:
            raise ValueError("financial certificate must cover every option once")
        roles = {option: row.get("role") for option, row in by_option.items()}
        if set(roles.values()) != {"proceed", "decline", "verify"}:
            raise ValueError("financial options must contain one proceed, decline, and verify action")
        for option, row in by_option.items():
            evidence(row.get("evidence"), f"financial option {option}")
        statuses = {row["status"] for row in constraints.values()}
        required_role = "decline" if "violated" in statuses else "verify" if "unknown" in statuses else "proceed"
        if roles[decision] != required_role:
            raise ValueError("financial decision does not follow the hard-suitability precedence rule")
        return {"option_roles": [{"option": option, "role": roles[option]} for option in sorted(options)]}

    if domain == "commerce_guidance":
        rows = certificate.get("option_assessments")
        if not isinstance(rows, list):
            raise ValueError("commerce certificate needs option_assessments")
        by_option = {row.get("option"): row for row in rows if isinstance(row, dict)}
        if set(by_option) != options:
            raise ValueError("commerce certificate must cover every option once")
        eligible: list[tuple[int, str]] = []
        unknown: list[str] = []
        normalized = []
        for option, row in by_option.items():
            status = row.get("eligibility")
            if status not in {"eligible", "violated", "unknown"}:
                raise ValueError("commerce eligibility is invalid")
            evidence(row.get("evidence"), f"commerce option {option}")
            rank = row.get("preference_rank")
            if status == "eligible":
                if not isinstance(rank, int) or rank < 1:
                    raise ValueError("eligible commerce option needs a positive preference rank")
                eligible.append((rank, option))
            elif rank is not None:
                raise ValueError("ineligible commerce option cannot have a preference rank")
            if status == "unknown":
                unknown.append(option)
            normalized.append({"option": option, "eligibility": status, "preference_rank": rank})
        if unknown:
            raise ValueError("commerce evidence leaves a required candidate attribute unknown")
        winners = [option for rank, option in eligible if rank == 1]
        if len(winners) != 1 or winners[0] != decision:
            raise ValueError("commerce decision is not the uniquely top-ranked eligible option")
        if len({rank for rank, _ in eligible}) != len(eligible):
            raise ValueError("commerce eligible options have a preference tie")
        return {"option_assessments": sorted(normalized, key=lambda row: row["option"])}

    states = {"supported", "refuted", "insufficient"}
    state = certificate.get("evidence_state")
    rows = certificate.get("stance_options")
    if state not in states or not isinstance(rows, list):
        raise ValueError("information certificate needs evidence_state and stance_options")
    by_option = {row.get("option"): row for row in rows if isinstance(row, dict)}
    if set(by_option) != options:
        raise ValueError("information certificate must cover every option once")
    stances = {option: row.get("stance") for option, row in by_option.items()}
    if set(stances.values()) != states:
        raise ValueError("information options must contain one supported, refuted, and insufficient stance")
    for option, row in by_option.items():
        evidence(row.get("evidence"), f"information option {option}")
    if stances[decision] != state:
        raise ValueError("information decision does not match the evidence state")
    return {"evidence_state": state, "stance_options": [{"option": option, "stance": stances[option]} for option in sorted(options)]}


def status_map(derivation: dict[str, Any]) -> dict[str, str]:
    return {item["id"]: item["status"] for item in derivation["per_constraint"]}


def certificate_signature(derivation: dict[str, Any]) -> str:
    """Compare only the normalized, executable family-specific certificate."""
    return json.dumps(derivation["decision_certificate"], sort_keys=True, separators=(",", ":"))


async def derive(record: dict[str, Any], client: ApiClient, role: Role) -> dict[str, Any]:
    errors: list[str] = []
    for attempt in range(3):
        try:
            raw = await client.chat_json(derivation_messages(record, role.name), temperature=0.0, max_tokens=500)
            return validate_derivation(raw, record)
        except Exception as exc:
            errors.append(f"attempt={attempt + 1}:{type(exc).__name__}:{exc}")
            if attempt < 2:
                await asyncio.sleep(attempt + 1)
    raise RuntimeError("; ".join(errors))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    """Persist one audit result immediately so an interrupted audit can resume."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def lock_and_rewrite(
    path: Path,
    replacements: dict[str, dict[str, Any]],
    removals: set[str],
    expected_digests: dict[str, str],
    expected_old_baselines: dict[str, dict[str, Any]],
) -> tuple[int, int]:
    """Apply audited replacements/removals without losing concurrent appends."""
    lock_path = path.with_suffix(path.suffix + ".lock")
    with lock_path.open("a", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        rows = read_jsonl(path)
        by_id = {row["id"]: row for row in rows}
        audited_ids = set(replacements) | removals
        missing = audited_ids - set(by_id)
        changed_input = [record_id for record_id in audited_ids if input_digest(by_id[record_id]) != expected_digests[record_id]]
        changed_label = [record_id for record_id in audited_ids if by_id[record_id]["baseline_decision"] != expected_old_baselines[record_id]]
        if missing or changed_input or changed_label:
            raise RuntimeError(
                "dataset changed while auditing: "
                f"missing={len(missing)} input_changed={len(changed_input)} label_changed={len(changed_label)}"
            )
        updated = removed = 0
        retained: list[dict[str, Any]] = []
        for row in rows:
            if row["id"] in removals:
                removed += 1
                continue
            replacement = replacements.get(row["id"])
            if replacement is not None:
                row["baseline_decision"] = replacement
                updated += 1
            retained.append(row)
        temporary = path.with_suffix(path.suffix + ".baseline.tmp")
        write_jsonl(temporary, retained)
        os.replace(temporary, path)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    return updated, removed


def write_snapshot(output_dir: Path, records: list[dict[str, Any]]) -> Path:
    """Preserve or verify the complete pre-repair source before an update."""
    snapshot = output_dir / "source_snapshot_before_repair.jsonl"
    if snapshot.exists():
        prior = read_jsonl(snapshot)
        prior_digests = {row["id"]: input_digest(row) for row in prior}
        current_digests = {row["id"]: input_digest(row) for row in records}
        if prior_digests != current_digests:
            raise RuntimeError(f"existing source snapshot does not match current audit input: {snapshot}")
        return snapshot
    write_jsonl(snapshot, records)
    return snapshot


def existing_results(path: Path, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Load stable audit rows, retrying transient API/format failures on resume."""
    if not path.exists():
        return []
    expected = {record["id"]: input_digest(record) for record in records}
    prior = read_jsonl(path)
    seen: set[str] = set()
    stable: list[dict[str, Any]] = []
    for row in prior:
        record_id = row.get("record_id")
        if record_id not in expected:
            raise RuntimeError(f"invalid existing audit result for {record_id!r}")
        if row.get("audit_version") != AUDIT_VERSION or row.get("reason", "").startswith("audit_error:"):
            # Provider or output-format failures do not constitute evidence
            # that a label is unresolved; give them a fresh three-attempt run.
            continue
        if record_id in seen:
            raise RuntimeError(f"invalid or duplicate existing audit result for {record_id!r}")
        if row.get("input_sha256") != expected[record_id]:
            raise RuntimeError(f"existing audit result has stale input digest for {record_id}")
        seen.add(record_id)
        stable.append(row)
    if len(stable) != len(prior):
        write_jsonl(path, stable)
    return stable


def materialize_existing_holdout(dataset_dir: Path, output_dir: Path) -> dict[str, int]:
    """Archive legacy holdout metadata without removing any MIU source rows."""
    manifest = dataset_dir / "MIU" / ".baseline_repair_holdout.jsonl"
    if not manifest.exists():
        return {"held": 0, "removed": 0}
    hold_lock_path = manifest.with_suffix(manifest.suffix + ".lock")
    with hold_lock_path.open("a", encoding="utf-8") as hold_lock:
        fcntl.flock(hold_lock.fileno(), fcntl.LOCK_EX)
        held = read_jsonl(manifest)
        audits = {row.get("record_id"): row for row in held if isinstance(row.get("record_id"), str)}
        source_by_id: dict[str, tuple[Path, dict[str, Any]]] = {}
        for split in ("train", "val", "test"):
            path = dataset_dir / "MIU" / f"{split}.jsonl"
            for row in read_jsonl(path):
                if row["id"] in audits:
                    if row["id"] in source_by_id:
                        raise RuntimeError(f"held record appears in multiple splits: {row['id']}")
                    source_by_id[row["id"]] = (path, row)
        missing = set(audits) - set(source_by_id)
        if missing:
            raise RuntimeError(f"held records are absent from MIU splits: {len(missing)}")
        archive = output_dir / "legacy_holdout_audit.jsonl"
        if archive.exists():
            raise RuntimeError(f"refusing to overwrite existing migration archive: {archive}")
        write_jsonl(archive, [
            {"record": source_by_id[record_id][1], "audit": audits[record_id]}
            for record_id in sorted(source_by_id)
        ])
        manifest.unlink()
        fcntl.flock(hold_lock.fileno(), fcntl.LOCK_UN)
    # The split locks are retained because producers use them. The obsolete
    # manifest lock belongs only to the removed holdout mechanism.
    hold_lock_path.unlink(missing_ok=True)
    return {"held": len(held), "removed": 0}


async def audit_records(
    records: list[dict[str, Any]],
    first: Role,
    second: Role,
    concurrency: int,
    result_path: Path,
    completed_ids: set[str],
) -> list[dict[str, Any]]:
    semaphore = asyncio.Semaphore(concurrency)
    async with (
        ApiClient(first.model, api_key_env=first.key_env, request_timeout_seconds=60) as first_client,
        ApiClient(second.model, api_key_env=second.key_env, request_timeout_seconds=60) as second_client,
    ):
        async def one(record: dict[str, Any]) -> dict[str, Any]:
            async with semaphore:
                result: dict[str, Any] = {
                    "audit_version": AUDIT_VERSION,
                    "record_id": record["id"],
                    "split": record["split"],
                    "family_domain": record["family_domain"],
                    "subscenario": record["subscenario"],
                    "input_sha256": input_digest(record),
                    "old_baseline": record["baseline_decision"],
                }
                try:
                    left, right = await asyncio.gather(
                        derive(record, first_client, first), derive(record, second_client, second),
                    )
                    result["deriver"] = left
                    result["verifier"] = right
                    # Both certificates are independently validated before
                    # reaching this point. The final option and every
                    # constraint status must agree; incidental selection of
                    # equally relevant clean citations need not be identical.
                    result["accepted"] = (
                        left["decision"] == right["decision"]
                        and status_map(left) == status_map(right)
                    )
                    if result["accepted"]:
                        result["new_baseline"] = {
                            "decision": left["decision"], "grounded_on": left["grounded_on"],
                        }
                        if "conclusion" in left:
                            result["new_baseline"]["conclusion"] = left["conclusion"]
                    else:
                        result["reason"] = "independent_derivations_disagree"
                except Exception as exc:
                    result["accepted"] = False
                    result["reason"] = f"audit_error:{type(exc).__name__}"
                    result["error_detail"] = str(exc)
                return result

        pending = [record for record in records if record["id"] not in completed_ids]
        results: list[dict[str, Any]] = []
        for future in asyncio.as_completed([asyncio.create_task(one(record)) for record in pending]):
            result = await future
            append_jsonl(result_path, result)
            results.append(result)
        return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--files", nargs="+", default=["data/dataset/MIU/train.jsonl", "data/dataset/MIU/val.jsonl", "data/dataset/MIU/test.jsonl"])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--deriver-model", default="gpt-5.4-mini")
    parser.add_argument("--verifier-model", default="qwen3.5-35b-a3b")
    parser.add_argument("--deriver-key-env", default="LOYAL_BASELINE_DERIVER_API_KEY")
    parser.add_argument("--verifier-key-env", default="LOYAL_BASELINE_VERIFIER_API_KEY")
    parser.add_argument("--base-url", default="https://yunwu.ai/v1")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--offset", type=int, default=0, help="Start from this deterministic input offset (for bounded split audits)")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument(
        "--apply-consensus-only",
        action="store_true",
        help="Apply only two-model consensus rows; unresolved rows remain available for regeneration",
    )
    parser.add_argument(
        "--materialize-existing-holdout",
        action="store_true",
        help="One-time migration: archive and delete the legacy holdout manifest",
    )
    args = parser.parse_args()

    if args.materialize_existing_holdout:
        if args.apply or args.apply_consensus_only:
            parser.error("--materialize-existing-holdout cannot be combined with audit application flags")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        dataset_dir = Path(args.files[0]).resolve().parents[1]
        result = materialize_existing_holdout(dataset_dir, args.output_dir)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if not os.environ.get(args.deriver_key_env) or not os.environ.get(args.verifier_key_env):
        parser.error(f"set {args.deriver_key_env} and {args.verifier_key_env}; keys are never written to disk")
    if args.apply and args.apply_consensus_only:
        parser.error("choose either --apply or --apply-consensus-only")
    os.environ["OPENAI_BASE_URL"] = args.base_url
    records = load_jsonl(args.files)
    if args.offset < 0:
        parser.error("--offset must be non-negative")
    records = records[args.offset:]
    if args.limit is not None:
        records = records[:args.limit]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    snapshot = write_snapshot(args.output_dir, records)
    first = Role("independent baseline deriver", args.deriver_model, args.deriver_key_env)
    second = Role("independent baseline verifier", args.verifier_model, args.verifier_key_env)
    result_path = args.output_dir / "baseline_regeneration.jsonl"
    prior_results = existing_results(result_path, records)
    fresh_results = asyncio.run(
        audit_records(records, first, second, args.concurrency, result_path, {row["record_id"] for row in prior_results})
    )
    results = prior_results + fresh_results
    if len(results) != len(records):
        raise RuntimeError(f"incomplete audit: expected={len(records)} results={len(results)}")
    accepted = [row for row in results if row["accepted"]]
    rejected = [row for row in results if not row["accepted"]]
    report = {
        "records": len(results), "accepted": len(accepted), "rejected": len(rejected),
        "changed_baseline": sum(row["new_baseline"] != row["old_baseline"] for row in accepted),
        "changed_decision": sum(row["new_baseline"]["decision"] != row["old_baseline"]["decision"] for row in accepted),
        "rejection_reasons": dict(Counter(row.get("reason", "unknown") for row in rejected)),
        "by_domain": {
            domain: {"records": len(rows), "accepted": sum(row["accepted"] for row in rows)}
            for domain, rows in sorted(
                ((domain, [row for row in results if row["family_domain"] == domain]) for domain in {row["family_domain"] for row in results}),
            )
        },
        "models": {"deriver": first.model, "verifier": second.model},
        "apply_requested": args.apply or args.apply_consensus_only,
        "application_mode": "all_consensus_required" if args.apply else "consensus_only" if args.apply_consensus_only else "audit_only",
        "source_snapshot": str(snapshot),
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))

    if not (args.apply or args.apply_consensus_only):
        return 0
    if args.apply and rejected:
        raise SystemExit("refusing --apply: unresolved records remain in baseline_regeneration.jsonl")
    replacements = {row["record_id"]: row["new_baseline"] for row in accepted}
    digests = {row["record_id"]: row["input_sha256"] for row in accepted}
    by_path: dict[Path, dict[str, dict[str, Any]]] = defaultdict(dict)
    path_by_split = {Path(path).name.removesuffix(".jsonl"): Path(path) for path in args.files}
    for record_id, replacement in replacements.items():
        split = next(row["split"] for row in accepted if row["record_id"] == record_id)
        by_path[path_by_split[split]][record_id] = replacement
    old_baselines = {row["record_id"]: row["old_baseline"] for row in results}
    digests = {row["record_id"]: row["input_sha256"] for row in results}
    for path in {Path(path) for path in args.files}:
        values = by_path.get(path, {})
        removals: set[str] = set()
        relevant_ids = set(values) | removals
        if relevant_ids:
            lock_and_rewrite(path, values, removals, digests, old_baselines)
    report["removed_unresolved"] = 0
    (args.output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
