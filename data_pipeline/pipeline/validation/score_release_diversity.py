#!/usr/bin/env python3
"""Compute stratified, release-facing diversity statistics for Loyal Agent.

Unlike the legacy generation-profile score, this script measures only fields
that are present in a released JSONL dataset.  It deliberately emits a metric
vector rather than an opaque weighted total: EIL and MIU have different valid
supports, and a single scalar can conceal template concentration or a missing
attack mode behind otherwise balanced count fields.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable


SPLITS = ("train", "val", "test")
TOKEN = re.compile(r"[a-z0-9]+")
NUMBER = re.compile(r"\b(?:\$?\d[\d,]*(?:\.\d+)?%?|one|two|three|four|five|six|seven|eight|nine|ten)\b", re.I)


def read_jsonl(paths: Iterable[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    return rows


def normalized(value: str) -> str:
    return " ".join(TOKEN.findall(value.lower()))


def request_tokens(row: dict[str, Any]) -> list[str]:
    return TOKEN.findall(row.get("user_natural_language", "").lower())


def request_template(row: dict[str, Any]) -> str:
    """A light delexicalisation that exposes repeated numeric-slot templates."""
    return normalized(NUMBER.sub(" <num> ", row.get("user_natural_language", "")))


def shingle_set(tokens: list[str], size: int = 5) -> set[str]:
    return {" ".join(tokens[index:index + size]) for index in range(max(0, len(tokens) - size + 1))}


def duplicate_excess(values: list[str]) -> int:
    return sum(count - 1 for count in Counter(values).values() if count > 1)


def categorical(values: list[Any]) -> dict[str, Any]:
    """Distribution profile with entropy normalized over its observed support.

    Normalizing by observed support avoids penalizing domains that correctly
    have a singleton axis (e.g., MIU information-guidance preferences).
    """
    counts = Counter(map(str, values))
    total = sum(counts.values())
    if not total:
        return {"observations": 0, "support": 0, "effective_support": 0.0, "normalized_entropy": None, "dominance": None, "distribution": {}}
    probabilities = [count / total for count in counts.values()]
    entropy = -sum(probability * math.log(probability) for probability in probabilities)
    support = len(counts)
    return {
        "observations": total,
        "support": support,
        "effective_support": round(math.exp(entropy), 3),
        "normalized_entropy": round(entropy / math.log(support), 4) if support > 1 else 1.0,
        "dominance": round(max(probabilities), 4),
        "distribution": dict(sorted(counts.items())),
    }


def lexical(rows: list[dict[str, Any]]) -> dict[str, Any]:
    streams = [request_tokens(row) for row in rows]
    tokens = [token for stream in streams for token in stream]
    bigrams = [" ".join(stream[index:index + 2]) for stream in streams for index in range(max(0, len(stream) - 1))]
    lengths = [len(stream) for stream in streams]
    return {
        "mean_request_tokens": round(mean(lengths), 2) if lengths else 0.0,
        "median_request_tokens": median(lengths) if lengths else 0,
        "distinct_1": round(len(set(tokens)) / len(tokens), 4) if tokens else 0.0,
        "distinct_2": round(len(set(bigrams)) / len(bigrams), 4) if bigrams else 0.0,
    }


def user_signature(row: dict[str, Any]) -> str:
    if row["mechanism"] == "EIL":
        fields = [item["content"] for item in row["necessary_information"] + row["exploitable_slots"]]
    else:
        fields = [item["content"] for item in row["user_constraints"] + row["user_preferences"]]
        fields += [item["name"] for item in row["authorized_information"]]
        fields += [item["label"] for item in row["decision_boundary"]["options"]]
    return " || ".join(sorted(normalized(value) for value in fields))


def near_duplicate_pairs(rows: list[dict[str, Any]], threshold: float = 0.55) -> int:
    """Count within-subscenario request pairs with 5-gram Jaccard >= threshold."""
    grouped: dict[str, list[tuple[str, set[str]]]] = defaultdict(list)
    for row in rows:
        grouped[row["subscenario"]].append((row["id"], shingle_set(request_tokens(row))))
    count = 0
    for examples in grouped.values():
        buckets: dict[str, list[tuple[str, set[str]]]] = defaultdict(list)
        for identifier, shingles in examples:
            for shingle in shingles:
                buckets[shingle].append((identifier, shingles))
        checked: set[tuple[str, str]] = set()
        for candidates in buckets.values():
            for left_index, (left_id, left) in enumerate(candidates):
                for right_id, right in candidates[left_index + 1:]:
                    pair = tuple(sorted((left_id, right_id)))
                    if pair in checked:
                        continue
                    checked.add(pair)
                    union = left | right
                    if union and len(left & right) / len(union) >= threshold:
                        count += 1
    return count


def surface(rows: list[dict[str, Any]]) -> dict[str, Any]:
    requests = [normalized(row.get("user_natural_language", "")) for row in rows]
    templates = [request_template(row) for row in rows]
    prefixes = [" ".join(request_tokens(row)[:5]) for row in rows]
    prefix_counts = Counter(prefix for prefix in prefixes if prefix)
    return {
        "exact_request_duplicate_excess": duplicate_excess(requests),
        "exact_request_duplicate_rate": round(duplicate_excess(requests) / len(rows), 4) if rows else 0.0,
        "delexicalized_template_duplicate_excess": duplicate_excess(templates),
        "delexicalized_template_duplicate_rate": round(duplicate_excess(templates) / len(rows), 4) if rows else 0.0,
        "near_duplicate_request_pairs_5gram_jaccard_ge_0_55": near_duplicate_pairs(rows),
        "repeated_five_token_prefix_excess": duplicate_excess(prefixes),
        "largest_five_token_prefix_share": round(max(prefix_counts.values(), default=0) / len(rows), 4) if rows else 0.0,
        "user_field_signature_unique_rate": round(len(set(user_signature(row) for row in rows)) / len(rows), 4) if rows else 0.0,
    }


def all_card_text(rows: list[dict[str, Any]]) -> list[str]:
    cards: list[str] = []
    for row in rows:
        if row["mechanism"] == "EIL":
            cards.extend(item["content"] for item in row["necessary_information"] + row["exploitable_slots"])
        else:
            cards.extend(item["content"] for item in row["clean_context"] + row["manipulated_context"])
    return [normalized(card) for card in cards]


def structure(rows: list[dict[str, Any]], mechanism: str) -> dict[str, Any]:
    if mechanism == "EIL":
        return {
            "num_necessary_information": categorical([row["meta"]["num_nec"] for row in rows]),
            "num_exploitable_slots": categorical([row["meta"]["num_exp"] for row in rows]),
            "pressure": categorical([row["adversary_config"]["pressure"] for row in rows]),
            "tactic": categorical([tactic for row in rows for tactic in row["adversary_config"].get("tactics", [])]),
            "slot_severity": categorical([item["severity"] for row in rows for item in row["exploitable_slots"]]),
        }
    return {
        "family_domain": categorical([row["family_domain"] for row in rows]),
        "num_constraints": categorical([row["meta"]["num_conditions"] for row in rows]),
        "num_preferences": categorical([row["meta"]["num_preferences"] for row in rows]),
        "num_authorized_sources": categorical([row["meta"]["num_auth"] for row in rows]),
        "num_clean_cards": categorical([row["meta"]["num_clean"] for row in rows]),
        "num_manipulated_cards": categorical([row["meta"]["num_mani"] for row in rows]),
        "baseline_action_type": categorical([
            next(option.get("action_type", "unlabelled") for option in row["decision_boundary"]["options"] if option["id"] == row["baseline_decision"]["decision"])
            for row in rows
        ]),
        "manipulation_tactic": categorical([item["attack"] for row in rows for item in row["manipulated_context"]]),
    }


def summarize(rows: list[dict[str, Any]], mechanism: str) -> dict[str, Any]:
    cards = all_card_text(rows)
    by_domain = {
        domain: {
            "records": len(group),
            "structural_axes": structure(group, mechanism),
            "lexical": lexical(group),
            "surface_and_combination": surface(group),
        }
        for domain, group in sorted(
            ((domain, [row for row in rows if row["family_domain"] == domain]) for domain in {row["family_domain"] for row in rows}),
        )
    }
    return {
        "records": len(rows),
        "splits": dict(sorted(Counter(row["split"] for row in rows).items())),
        "subscenarios": categorical([row["subscenario"] for row in rows]),
        "structural_axes": structure(rows, mechanism),
        "lexical": lexical(rows),
        "surface_and_combination": surface(rows),
        "card_content": {
            "cards": len(cards),
            "exact_duplicate_excess": duplicate_excess(cards),
            "exact_duplicate_rate": round(duplicate_excess(cards) / len(cards), 4) if cards else 0.0,
        },
        "family_domain_strata": by_domain,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eil-dir", type=Path, required=True, help="Directory containing EIL train/val/test JSONL files")
    parser.add_argument("--miu-dir", type=Path, required=True, help="Directory containing MIU train/val/test JSONL files")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    eil = read_jsonl([args.eil_dir / f"{split}.jsonl" for split in SPLITS])
    miu = read_jsonl([args.miu_dir / f"{split}.jsonl" for split in SPLITS])
    if any(row.get("mechanism") != "EIL" for row in eil) or any(row.get("mechanism") != "MIU" for row in miu):
        raise ValueError("input directory contains a record with the wrong mechanism")
    ids = [row["id"] for row in eil + miu]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate IDs across EIL and MIU inputs")
    result = {
        "metric_version": "release-diversity-v1",
        "methodology": {
            "unit_of_analysis": "released records; near-duplicate comparisons are restricted to a subscenario",
            "categorical_balance": "normalized Shannon entropy over observed support, plus effective support and dominance",
            "near_duplicate_rule": "5-gram Jaccard similarity >= 0.55",
            "template_rule": "exact equality after lowercasing, punctuation removal, and numeric-token replacement",
            "interpretation": "Report the vector by mechanism; no cross-mechanism composite is calculated.",
        },
        "EIL": summarize(eil, "EIL"),
        "MIU": summarize(miu, "MIU"),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
