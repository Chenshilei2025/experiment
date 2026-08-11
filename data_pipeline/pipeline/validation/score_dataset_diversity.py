#!/usr/bin/env python3
"""Score observable diversity and prompt-profile adherence for each job."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

from ..generation import builder as pipeline


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def normalized(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.lower()))


def axis_score(values: list[Any], possible: list[Any]) -> tuple[float, float, float]:
    """Coverage, distance-from-uniform evenness, and their mean."""
    if not values:
        return 0.0, 0.0, 0.0
    counts = Counter(values)
    coverage = len(counts) / len(possible)
    uniform = 1 / len(possible)
    distance = sum(abs(counts[item] / len(values) - uniform) for item in possible) / 2
    evenness = 1 - distance
    return coverage, evenness, (coverage + evenness) / 2


def shingle_set(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {" ".join(words[index:index + 5]) for index in range(max(len(words) - 4, 0))}


def near_duplicate_pairs(records: list[dict[str, Any]]) -> int:
    buckets: dict[str, list[tuple[str, set[str]]]] = defaultdict(list)
    for record in records:
        shingles = shingle_set(record.get("user_natural_language", ""))
        for shingle in shingles:
            buckets[shingle].append((record["id"], shingles))
    pairs: set[tuple[str, str]] = set()
    for candidates in buckets.values():
        for (left_id, left), (right_id, right) in combinations(candidates, 2):
            pair = tuple(sorted((left_id, right_id)))
            if pair in pairs:
                continue
            union = left | right
            if union and len(left & right) / len(union) >= 0.55:
                pairs.add(pair)
    return len(pairs)


def surface_template_metrics(records: list[dict[str, Any]]) -> dict[str, float]:
    """Detect recurring request shells even when slots are substituted."""
    texts = [record["user_natural_language"] for record in records]
    prefixes = Counter(" ".join(re.findall(r"[a-z0-9]+", text.lower())[:5]) for text in texts)
    trigrams = Counter(
        " ".join(words[index:index + 3])
        for text in texts
        for words in [re.findall(r"[a-z0-9]+", text.lower())]
        for index in range(max(0, len(words) - 2))
    )
    total_trigrams = sum(trigrams.values())
    repeated_share = sum(count for count in trigrams.values() if count >= 3) / total_trigrams if total_trigrams else 0.0
    max_prefix_share = max(prefixes.values(), default=0) / max(1, len(texts))
    # A score of 1 means no repeated three-gram or recurring opening shell.
    return {
        "top_prefix_share": max_prefix_share,
        "repeated_trigram_share": repeated_share,
        "surface_variation": max(0.0, 1 - 0.55 * repeated_share - 0.45 * max_prefix_share),
    }


def profile_fidelity(records: list[dict[str, Any]], audits: dict[str, dict[str, Any]]) -> tuple[float, int]:
    matched = checked = 0
    for record in records:
        plan = audits.get(record["id"], {}).get("generation_profile", {}).get("record_plan")
        if not plan:
            continue
        checked += 1
        if record["mechanism"] == "EIL":
            good = record["meta"]["num_nec"] == plan["counts"]["num_nec"]
            good = good and record["meta"]["num_exp"] == plan["counts"]["num_exp"]
            good = good and set(record["adversary_config"].get("tactics", [])) == set(plan["adversary_tactics"])
        else:
            good = all(record["meta"][name] == plan["counts"][name] for name in (
                "num_conditions", "num_preferences", "num_auth", "num_clean", "num_mani"
            ))
        matched += int(good)
    return (matched / checked if checked else 0.0), checked


def meaningful_tokens(text: str) -> set[str]:
    stop = {"about", "after", "also", "and", "are", "based", "can", "for", "from", "have", "into", "must", "not", "only", "that", "the", "this", "with", "would", "your"}
    return {token for token in re.findall(r"[a-z0-9]+", text.lower()) if len(token) >= 4 and token not in stop}


def user_field_texts(record: dict[str, Any]) -> list[str]:
    """Only fields that express the user's task, criteria, or private briefing."""
    if record["mechanism"] == "EIL":
        return [item["content"] for item in record["necessary_information"] + record["exploitable_slots"]]
    values = [item["content"] for item in record["user_constraints"] + record["user_preferences"]]
    values.extend(item["name"] for item in record["authorized_information"])
    values.extend(item["label"] for item in record["decision_boundary"]["options"])
    return values


def user_signature(record: dict[str, Any]) -> str:
    return " || ".join(sorted(normalized(value) for value in user_field_texts(record)))


def user_content_metrics(records: list[dict[str, Any]]) -> tuple[float, float, float, float]:
    """Combination uniqueness, vocabulary diversity, request uniqueness, alignment."""
    signatures = [user_signature(record) for record in records]
    signature_unique = len(set(signatures)) / max(1, len(signatures))
    all_tokens = [token for record in records for value in user_field_texts(record) for token in meaningful_tokens(value)]
    vocabulary = min(1.0, len(set(all_tokens)) / max(30, len(records) * 3))
    exact = sum(count - 1 for count in Counter(normalized(record["user_natural_language"]) for record in records).values() if count > 1)
    near = near_duplicate_pairs(records)
    request_unique = max(0.0, 1 - (exact + near) / max(1, len(records)))
    alignments = []
    for record in records:
        narrative = meaningful_tokens(record["user_natural_language"])
        fields = [meaningful_tokens(value) for value in user_field_texts(record)]
        # A field can be paraphrased; requiring one meaningful shared token is
        # intentionally conservative and reports a warning signal, not truth.
        alignments.append(sum(bool(field & narrative) for field in fields) / max(1, len(fields)))
    return signature_unique, vocabulary, request_unique, sum(alignments) / max(1, len(alignments))


def planned_dimension_metrics(
    records: list[dict[str, Any]], audits: dict[str, dict[str, Any]], block: pipeline.PromptBlock,
) -> dict[str, Any]:
    """Report coverage of the prompt-derived user dimensions in private plans.

    This intentionally reports plan coverage, not a claim that a language model
    has semantically realised every dimension.  The latter needs a separate
    semantic judge and must not be inferred from an ordinal or an attack label.
    """
    possible = pipeline.dimensions_for(block.scenario)
    assigned: list[str] = []
    pairs: list[tuple[str, str]] = []
    audited = 0
    for record in records:
        profile = audits.get(record["id"], {}).get("generation_profile", {}).get("user_diversity", {})
        selected = profile.get("user_dimensions", [])
        if len(selected) != 2:
            continue
        audited += 1
        assigned.extend(selected)
        pairs.append(tuple(sorted(selected)))
    if not audited:
        return {
            "status": "unavailable_for_legacy_records",
            "audited_records": 0,
            "possible_dimensions": possible,
        }
    coverage, evenness, _ = axis_score(assigned, possible)
    possible_pairs = len(possible) * (len(possible) - 1) // 2
    return {
        "status": "planned_only_not_semantic_proof",
        "audited_records": audited,
        "possible_dimensions": possible,
        "coverage": coverage,
        "evenness": evenness,
        "pair_coverage": len(set(pairs)) / possible_pairs,
        "distribution": dict(sorted(Counter(assigned).items())),
    }


def score_job(records: list[dict[str, Any]], audits: dict[str, dict[str, Any]], block: pipeline.PromptBlock) -> dict[str, Any]:
    profile, audited = profile_fidelity(records, audits)
    planned_dimensions = planned_dimension_metrics(records, audits, block)
    signature_unique, vocabulary, request_unique, alignment = user_content_metrics(records)
    surface = surface_template_metrics(records)
    exact = sum(count - 1 for count in Counter(normalized(record["user_natural_language"]) for record in records).values() if count > 1)
    near = near_duplicate_pairs(records)
    if block.family == "delegated":
        fields = {"num_nec": [1, 2, 3, 4, 5], "num_exp": [2, 3, 4, 5]}
        axes = {name: axis_score([record["meta"][name] for record in records], values) for name, values in fields.items()}
        structural = sum(value[2] for value in axes.values()) / len(axes)
        components = {"user_profile_fidelity": profile, "user_structure": structural, "user_combination_uniqueness": signature_unique, "user_vocabulary": vocabulary, "request_uniqueness": request_unique, "field_to_request_alignment": alignment, "surface_variation": surface["surface_variation"]}
        # Surface templating is a user-side risk, so it receives a substantial
        # weight instead of being hidden by mechanically perfect count quotas.
        score = 100 * (0.15 * profile + 0.20 * structural + 0.15 * signature_unique + 0.05 * vocabulary + 0.10 * request_unique + 0.10 * alignment + 0.25 * surface["surface_variation"])
        details = {"axes": axes}
    else:
        fields = {
            "num_conditions": [1, 2, 3, 4], "num_preferences": [1, 2, 3, 4],
            "num_auth": [1, 2, 3], "num_clean": [2, 3, 4, 5], "num_mani": [2, 3, 4, 5],
        }
        axes = {name: axis_score([record["meta"][name] for record in records], values) for name, values in fields.items()}
        structural = sum(value[2] for value in axes.values()) / len(axes)
        components = {"user_profile_fidelity": profile, "user_structure": structural, "user_combination_uniqueness": signature_unique, "user_vocabulary": vocabulary, "request_uniqueness": request_unique, "field_to_request_alignment": alignment, "surface_variation": surface["surface_variation"]}
        score = 100 * (0.15 * profile + 0.20 * structural + 0.15 * signature_unique + 0.05 * vocabulary + 0.10 * request_unique + 0.10 * alignment + 0.25 * surface["surface_variation"])
        details = {"axes": axes}
    return {
        "subscenario": block.scenario, "mechanism": "EIL" if block.family == "delegated" else "MIU",
        "records": len(records), "target": block.target_count,
        "status": "stable" if len(records) >= min(60, block.target_count) else "provisional",
        "score_100": round(score, 1), "components_100": {name: round(value * 100, 1) for name, value in components.items()},
        "audited_profile_rows": audited, "exact_duplicate_requests": exact, "near_duplicate_request_pairs": near,
        "details": {**details, "surface_template": surface, "prompt_user_dimensions": planned_dimensions},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=Path("data/runs"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    blocks = {pipeline.slugify(block.scenario): block for block in pipeline.load_prompt_blocks()}
    jobs = []
    for job in sorted(args.runs_dir.glob(f"*/{args.run_id}")):
        block = blocks.get(job.parent.name)
        records = read_jsonl(job / "records.jsonl")
        if not block or not records:
            continue
        audits = {item["record_id"]: item for item in read_jsonl(job / "records.audit.jsonl")}
        jobs.append(score_job(records, audits, block))
    total = sum(job["records"] for job in jobs)
    overall = sum(job["score_100"] * job["records"] for job in jobs) / total if total else 0.0
    result = {
        "metric_version": "v3-user-side", "overall_score_100": round(overall, 1), "records_scored": total,
        "scope": "The score measures observable user-side structure and surface variation. prompt_user_dimensions reports private planned coverage only; semantic realisation needs an independent judge and is not inferred from attacks or record ordinals.",
        "jobs": jobs,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
