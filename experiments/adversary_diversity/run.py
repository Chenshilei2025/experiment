"""Measure EIL adversary-temperature and adversary-model diversity.

This experiment never generates a new policy response. It rescoring saved EIL
``per_sample.jsonl`` rows under every requested adversary model and temperature,
then reuses the production leakage and utility judges for every branch.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import defaultdict
from itertools import combinations
import json
from pathlib import Path
import re
import time
from typing import Any

from eil.reward import aggregate_eil_metrics, severity_risk_details
from eil.scoring.leakage import adversary_turn, assess_leakage
from eil.scoring.task_utility import assess_utility
from scripts.common.api_client import ApiClientPool, ChatClient
from scripts.common.experiment_logging import capture_run_output, write_run_provenance


PROJECT_ROOT = Path(__file__).resolve().parents[2]
Record = dict[str, Any]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _parse_adversary(value: str) -> tuple[str, str]:
    """Parse ``display-name=ENV_PREFIX`` without putting endpoint data in files."""
    name, separator, prefix = value.partition("=")
    if not separator or not name or not prefix or not re.fullmatch(r"[A-Z][A-Z0-9_]*", prefix):
        raise ValueError("--adversary must use NAME=OPENAI_COMPATIBLE_ENV_PREFIX")
    return name, prefix


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _rank(values: list[float]) -> list[float]:
    """Average-tie ranks, sufficient for a dependency-free Spearman summary."""
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + 1 + end) / 2.0
        for index in order[start:end]:
            ranks[index] = rank
        start = end
    return ranks


def _pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) < 2 or len(left) != len(right):
        return None
    left_mean, right_mean = sum(left) / len(left), sum(right) / len(right)
    left_delta = [item - left_mean for item in left]
    right_delta = [item - right_mean for item in right]
    denominator = (sum(item * item for item in left_delta) * sum(item * item for item in right_delta)) ** 0.5
    return None if denominator == 0 else sum(a * b for a, b in zip(left_delta, right_delta, strict=True)) / denominator


def _spearman(left: list[float], right: list[float]) -> float | None:
    return _pearson(_rank(left), _rank(right))


def _fact_key(value: str) -> str:
    return " ".join(re.findall(r"\w+", value.casefold()))


def _jaccard_distance(left: set[str], right: set[str]) -> float:
    """Distance is zero for two empty sets and one for disjoint non-empty sets."""
    union = left | right
    return 0.0 if not union else 1.0 - len(left & right) / len(union)


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return (sum((value - mean) ** 2 for value in values) / len(values)) ** 0.5


def _semantic_fact_keys(branch: dict[str, Any]) -> set[str]:
    """Return auditable fact identities for one adversary branch.

    A fact cited by the leakage judge is identified by the protected slots it
    recovered, rather than its surface wording.  This makes paraphrases such
    as "their deadline is Friday" and "they need it by Friday" one fact when
    the judge maps both to the same protected slot.  Facts not linked to a
    protected slot remain visible under a normalized-text fallback instead of
    being silently discarded.
    """
    inference_slots: dict[str, set[str]] = defaultdict(set)
    for slot_id, detail in (branch.get("slot_details") or {}).items():
        if not isinstance(detail, dict):
            continue
        if float(detail.get("severity_score", 0.0)) <= 0.0:
            continue
        if detail.get("source") not in {"adversary", "both"}:
            continue
        inference_id = detail.get("inference_id")
        if isinstance(inference_id, str) and inference_id:
            inference_slots[inference_id].add(str(slot_id))

    keys: set[str] = set()
    for fact in branch.get("inferred_private_facts", []):
        if not isinstance(fact, dict) or not isinstance(fact.get("info"), str):
            continue
        inference_id = fact.get("inference_id")
        slots = inference_slots.get(inference_id) if isinstance(inference_id, str) else None
        if slots:
            # The protected slot is the semantic identity: if one adversary
            # recovers two slots in one sentence while another recovers them
            # separately, they still overlap on each recovered secret.
            keys.update("slot:" + slot_id for slot_id in slots)
        elif normalized := _fact_key(fact["info"]):
            keys.add("text:" + normalized)
    return keys


def _recovered_slots(branch: dict[str, Any], *, adversary_only: bool = False) -> set[str]:
    """Return protected slots with non-zero recovery in one branch."""
    if adversary_only:
        return {
            str(slot_id)
            for slot_id, detail in (branch.get("slot_details") or {}).items()
            if isinstance(detail, dict)
            and float(detail.get("severity_score", 0.0)) > 0.0
            and detail.get("source") in {"adversary", "both"}
        }
    return {
        str(slot_id)
        for slot_id, score in (branch.get("slot_scores") or {}).items()
        if isinstance(score, (int, float)) and not isinstance(score, bool) and float(score) > 0.0
    }


async def _branch(
    response: str, record: Record, name: str, client: ChatClient, temperature: float,
    leakage_judge: ChatClient, utility_judge: ChatClient,
) -> dict[str, Any]:
    try:
        inference = await adversary_turn(response, record, client, temperature=temperature)
        leakage, utility = await asyncio.gather(
            assess_leakage(response, record, inference, leakage_judge),
            assess_utility(response, record, inference, utility_judge),
        )
        leakage_value, exposure = severity_risk_details(leakage["slot_scores"], record)
        metrics = aggregate_eil_metrics([utility["task_utility"]], [leakage_value])
        return {
            "adversary": name,
            "temperature": temperature,
            "inferred_private_facts": inference["inferred_private_facts"],
            "counterparty_reply": inference["counterparty_reply"],
            "slot_scores": leakage["slot_scores"],
            "slot_details": leakage["slot_details"],
            "task_utility": utility["task_utility"],
            "leakage": leakage_value,
            "reward": metrics["reward"],
            "leakage_weighted_exposure": exposure,
            "evaluator_error": None,
        }
    except Exception as exc:
        return {
            "adversary": name, "temperature": temperature,
            "task_utility": None, "leakage": None, "reward": None,
            "evaluator_error": f"{type(exc).__name__}: {exc}",
        }


def summarize(rows: list[dict[str, Any]], branches: list[tuple[str, float]]) -> dict[str, Any]:
    """Summarize branch performance and adversary-specific attack surfaces."""
    complete = [row for row in rows if all(item.get("reward") is not None for item in row["branches"])]
    by_branch: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        for branch in row["branches"]:
            by_branch[(branch["adversary"], float(branch["temperature"]))].append(branch)

    branch_summary: dict[str, Any] = {}
    for name, temperature in branches:
        values = [item for item in by_branch[(name, temperature)] if item.get("reward") is not None]
        branch_summary[f"{name}@{temperature:g}"] = {
            "n_scored": len(values),
            "task_utility_mean": _mean([float(item["task_utility"]) for item in values]),
            "leakage_mean": _mean([float(item["leakage"]) for item in values]),
            "reward_mean": _mean([float(item["reward"]) for item in values]),
        }

    # The following per-response measures deliberately use only complete rows,
    # so every branch has equal opportunity to recover an item.
    legacy_unique_facts = []
    semantic_unique_facts = []
    fact_jaccard_distances = []
    fact_jaccard_same_model_different_temperature = []
    fact_jaccard_same_temperature_different_model = []
    branch_unique_fact_rates = []
    branch_specific_fact_counts = []
    union_slot_counts = []
    union_slot_rates = []
    branch_missing_slot_rates = []
    branch_specific_slot_counts = []
    slot_jaccard_distances = []
    slot_jaccard_same_model_different_temperature = []
    slot_jaccard_same_temperature_different_model = []
    adversary_slot_jaccard_distances = []
    adversary_slot_jaccard_same_model_different_temperature = []
    adversary_slot_jaccard_same_temperature_different_model = []
    leakage_level_disagreements = []
    exposure_stds = []
    exposure_ranges = []
    slot_score_disagreements = []
    ensemble_rewards: dict[str, float] = {}
    for row in complete:
        row_branches = row["branches"]
        legacy_fact_set = {
            _fact_key(fact["info"])
            for item in row_branches for fact in item.get("inferred_private_facts", [])
            if isinstance(fact, dict) and isinstance(fact.get("info"), str) and _fact_key(fact["info"])
        }
        legacy_unique_facts.append(len(legacy_fact_set))
        fact_sets = [_semantic_fact_keys(item) for item in row_branches]
        semantic_union = set().union(*fact_sets)
        semantic_unique_facts.append(len(semantic_union))
        for left_index, right_index in combinations(range(len(row_branches)), 2):
            left, right = fact_sets[left_index], fact_sets[right_index]
            distance = _jaccard_distance(left, right)
            fact_jaccard_distances.append(distance)
            left_branch, right_branch = row_branches[left_index], row_branches[right_index]
            if left_branch["adversary"] == right_branch["adversary"] and left_branch["temperature"] != right_branch["temperature"]:
                fact_jaccard_same_model_different_temperature.append(distance)
            if left_branch["adversary"] != right_branch["adversary"] and left_branch["temperature"] == right_branch["temperature"]:
                fact_jaccard_same_temperature_different_model.append(distance)
        fact_frequencies = defaultdict(int)
        for fact_set in fact_sets:
            for fact in fact_set:
                fact_frequencies[fact] += 1
        for fact_set in fact_sets:
            specific = sum(fact_frequencies[fact] == 1 for fact in fact_set)
            branch_specific_fact_counts.append(float(specific))
            if fact_set:
                branch_unique_fact_rates.append(specific / len(fact_set))

        slot_sets = [_recovered_slots(item) for item in row_branches]
        adversary_slot_sets = [_recovered_slots(item, adversary_only=True) for item in row_branches]
        union_slots = set().union(*slot_sets)
        all_slots = set().union(*(set((item.get("slot_scores") or {})) for item in row_branches))
        union_slot_counts.append(float(len(union_slots)))
        union_slot_rates.append(len(union_slots) / len(all_slots) if all_slots else 0.0)
        for left_index, right_index in combinations(range(len(row_branches)), 2):
            slot_distance = _jaccard_distance(slot_sets[left_index], slot_sets[right_index])
            adversary_slot_distance = _jaccard_distance(
                adversary_slot_sets[left_index], adversary_slot_sets[right_index],
            )
            slot_jaccard_distances.append(slot_distance)
            adversary_slot_jaccard_distances.append(adversary_slot_distance)
            left_branch, right_branch = row_branches[left_index], row_branches[right_index]
            if left_branch["adversary"] == right_branch["adversary"] and left_branch["temperature"] != right_branch["temperature"]:
                slot_jaccard_same_model_different_temperature.append(slot_distance)
                adversary_slot_jaccard_same_model_different_temperature.append(adversary_slot_distance)
            if left_branch["adversary"] != right_branch["adversary"] and left_branch["temperature"] == right_branch["temperature"]:
                slot_jaccard_same_temperature_different_model.append(slot_distance)
                adversary_slot_jaccard_same_temperature_different_model.append(adversary_slot_distance)
        slot_frequencies = defaultdict(int)
        for slot_set in slot_sets:
            for slot in slot_set:
                slot_frequencies[slot] += 1
        for slot_set in slot_sets:
            branch_specific_slot_counts.append(float(sum(slot_frequencies[slot] == 1 for slot in slot_set)))
            if union_slots:
                branch_missing_slot_rates.append(len(union_slots - slot_set) / len(union_slots))

        leakage_values = [float(item["leakage"]) for item in row_branches]
        exposures = [float(item["leakage_weighted_exposure"]) for item in row_branches]
        leakage_level_disagreements.append(float(len(set(leakage_values)) > 1))
        exposure_stds.append(_std(exposures))
        exposure_ranges.append(max(exposures) - min(exposures))
        score_maps = [tuple(sorted((item.get("slot_scores") or {}).items())) for item in row_branches]
        slot_score_disagreements.append(float(len(set(score_maps)) > 1))
        ensemble_rewards[row["id"]] = sum(float(item["reward"]) for item in row_branches) / len(row_branches)

    rank_correlation: dict[str, float | None] = {}
    ordered_ids = sorted(ensemble_rewards)
    for name, temperature in branches:
        per_id = {
            row["id"]: next(item["reward"] for item in row["branches"] if item["adversary"] == name and item["temperature"] == temperature)
            for row in complete
        }
        rank_correlation[f"{name}@{temperature:g}"] = _spearman(
            [ensemble_rewards[item] for item in ordered_ids], [float(per_id[item]) for item in ordered_ids],
        )
    ensemble = {
        "n_complete": len(complete),
        # Kept for compatibility with v1 result consumers.  New analyses
        # should use fact_diversity, whose primary keys are judge-grounded.
        "unique_inferred_facts_per_response_mean": _mean([float(item) for item in legacy_unique_facts]),
        "reward_mean": _mean(list(ensemble_rewards.values())),
    }
    return {
        "n_responses": len(rows), "n_branch_conditions": len(branches),
        "branch_metrics": branch_summary,
        "fact_diversity": {
            "deduplication": "judge-grounded protected-slot key; normalized-text fallback for unlinked facts",
            "semantic_unique_facts_per_response_mean": _mean([float(item) for item in semantic_unique_facts]),
            "pairwise_semantic_fact_jaccard_distance_mean": _mean(fact_jaccard_distances),
            "same_model_different_temperature_fact_jaccard_distance_mean": _mean(fact_jaccard_same_model_different_temperature),
            "same_temperature_different_model_fact_jaccard_distance_mean": _mean(fact_jaccard_same_temperature_different_model),
            "branch_specific_semantic_facts_per_response_mean": _mean(branch_specific_fact_counts),
            "branch_unique_semantic_fact_rate_mean": _mean(branch_unique_fact_rates),
        },
        "slot_coverage": {
            "ensemble_recovered_slots_per_response_mean": _mean(union_slot_counts),
            "ensemble_recovered_slot_rate_mean": _mean(union_slot_rates),
            "single_branch_missing_slot_rate_vs_ensemble_mean": _mean(branch_missing_slot_rates),
            "branch_specific_recovered_slots_per_response_mean": _mean(branch_specific_slot_counts),
            "pairwise_recovered_slot_jaccard_distance_mean": _mean(slot_jaccard_distances),
            "same_model_different_temperature_slot_jaccard_distance_mean": _mean(slot_jaccard_same_model_different_temperature),
            "same_temperature_different_model_slot_jaccard_distance_mean": _mean(slot_jaccard_same_temperature_different_model),
            "pairwise_adversary_recovered_slot_jaccard_distance_mean": _mean(adversary_slot_jaccard_distances),
            "same_model_different_temperature_adversary_slot_jaccard_distance_mean": _mean(adversary_slot_jaccard_same_model_different_temperature),
            "same_temperature_different_model_adversary_slot_jaccard_distance_mean": _mean(adversary_slot_jaccard_same_temperature_different_model),
        },
        "risk_disagreement": {
            "leakage_level_disagreement_rate": _mean(leakage_level_disagreements),
            "slot_score_disagreement_rate": _mean(slot_score_disagreements),
            "weighted_exposure_std_per_response_mean": _mean(exposure_stds),
            "weighted_exposure_range_per_response_mean": _mean(exposure_ranges),
        },
        "ensemble": ensemble,
        "single_branch_reward_spearman_with_ensemble": rank_correlation,
    }


async def _run_all(
    source_rows: list[dict[str, Any]], records: dict[str, Record], adversaries: dict[str, ChatClient],
    temperatures: tuple[float, ...], leakage_judge: ChatClient, utility_judge: ChatClient, concurrency: int,
) -> list[dict[str, Any]]:
    semaphore = asyncio.Semaphore(concurrency)

    async def bounded_branch(
        response: str, record: Record, name: str, client: ChatClient, temperature: float,
    ) -> dict[str, Any]:
        async with semaphore:
            return await _branch(response, record, name, client, temperature, leakage_judge, utility_judge)

    async def score(row: dict[str, Any]) -> dict[str, Any]:
        work = [
            bounded_branch(row["response"], records[row["id"]], name, client, temperature)
            for name, client in adversaries.items() for temperature in temperatures
        ]
        return {"id": row["id"], "response": row["response"], "branches": await asyncio.gather(*work)}

    return await asyncio.gather(*(score(row) for row in source_rows))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-jsonl", type=Path, required=True, help="saved EIL per_sample.jsonl with id and response")
    parser.add_argument("--records", type=Path, default=PROJECT_ROOT / "eil/data/dataset/EIL/test.jsonl")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--adversary", action="append", required=True, metavar="NAME=PREFIX")
    parser.add_argument("--temperatures", default="0.0,0.3,0.6,0.8,1.0")
    parser.add_argument("--concurrency", type=int, default=2)
    args = parser.parse_args()
    if args.concurrency < 1:
        parser.error("--concurrency must be positive")
    try:
        temperatures = tuple(float(item.strip()) for item in args.temperatures.split(",") if item.strip())
    except ValueError:
        parser.error("--temperatures must be comma-separated numbers")
    if not temperatures or any(not 0.0 <= item <= 2.0 for item in temperatures):
        parser.error("temperatures must be within [0, 2]")
    parsed_adversaries = dict(_parse_adversary(item) for item in args.adversary)
    if len(parsed_adversaries) != len(args.adversary):
        parser.error("adversary names must be unique")
    source_rows = _read_jsonl(args.source_jsonl)
    records = {item["id"]: item for item in _read_jsonl(args.records)}
    if not source_rows or len({item.get("id") for item in source_rows}) != len(source_rows) or any(
        not isinstance(item.get("response"), str) or item["id"] not in records for item in source_rows
    ):
        parser.error("source must contain unique EIL record IDs and non-empty response strings")
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    write_run_provenance(args.output_dir)
    with capture_run_output(args.output_dir):
        adversaries = {name: ApiClientPool.from_env(prefix) for name, prefix in parsed_adversaries.items()}
        leakage_judge = ApiClientPool.from_env("LOYAL_EIL_LEAKAGE_JUDGE", fallback_prefix="LOYAL_EIL_JUDGE")
        utility_judge = ApiClientPool.from_env("LOYAL_EIL_UTILITY_JUDGE", fallback_prefix="LOYAL_EIL_JUDGE")
        started = time.time()
        rows = asyncio.run(_run_all(source_rows, records, adversaries, temperatures, leakage_judge, utility_judge, args.concurrency))
        with (args.output_dir / "per_response.jsonl").open("x", encoding="utf-8") as output:
            for row in rows:
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
        branches = [(name, temperature) for name in parsed_adversaries for temperature in temperatures]
        summary = summarize(rows, branches)
        summary["run"] = {
            "experiment": "adversary_diversity_v1", "source_jsonl": str(args.source_jsonl.resolve()),
            "records": str(args.records.resolve()), "adversaries": parsed_adversaries,
            "temperatures": list(temperatures), "started_at_unix": started, "finished_at_unix": time.time(),
        }
        (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print("FINAL_SUMMARY=" + json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
