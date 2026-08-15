#!/usr/bin/env python3
"""Task-conditioned diversity validation and reporting for Loyal Agent.

This module has two public modes:

* ``subscenario`` is a generation release gate.  It checks audit completeness,
  private-profile fidelity, duplicate requests, and minimum field coverage for
  a single completed job.
* ``dataset`` reports paper-facing diversity statistics for released EIL and
  MIU JSONL files. It covers interaction scenarios and the functional types of
  loyalty-relevant fields; it deliberately excludes configuration counts,
  surface repetition, and other quality diagnostics.

The two mechanisms intentionally have separate metrics: EIL is a
disclosure-boundary task, whereas MIU is an evidence-and-manipulation decision
task.  A common weighted score would confound their valid schemas.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterable


SPLITS = ("train", "val", "test")
TOKEN = re.compile(r"[a-z0-9]+")

# Paper-facing, observable functional taxonomy.  A record can receive multiple
# tags; the rules classify released field text rather than claiming latent LLM
# intent. The released report counts only atomic functional types.
EIL_NECESSARY_RULES = {
    "task_scope_or_deliverable": ("service", "project", "scope", "deliver", "repair", "application", "appointment", "lease", "contract", "role", "program"),
    "public_timing_or_process": ("date", "week", "month", "schedule", "timing", "appointment", "deadline", "submit", "stage", "move in", "notice"),
    "qualification_or_document": ("document", "verification", "reference", "resume", "portfolio", "qualification", "certificate", "record", "pay stub", "evidence"),
    "requested_term_or_remedy": ("request", "remedy", "quote", "price", "rent", "salary", "coverage", "refund", "accommodation", "payment"),
    "preference_or_specification": ("prefer", "preference", "feature", "size", "pet", "style", "location", "remote", "format", "accessibility"),
}
EIL_LEVERAGE_RULES = {
    "deadline_or_urgency": ("must", "urgent", "quickly", "deadline", "by ", "expir", "immediately", "this week", "time pressure"),
    "reservation_price_or_budget": ("maximum", "budget", "ceiling", "afford", "up to", "financial pressure", "cannot exceed", "willing to pay"),
    "weak_alternatives_or_competition": ("no other", "no competing", "rejected", "declined", "limited alternative", "none have", "fully booked", "only option"),
    "personal_or_health_vulnerability": ("health", "medical", "anxiety", "family", "pregnan", "disability", "mental", "leave"),
    "legal_or_claim_weakness": ("uncertain", "weak", "fault", "liability", "conflict", "evidence gap", "prior refusal"),
    "financial_or_payment_capacity": ("debt", "income", "credit", "cash", "payment plan", "savings", "funding"),
}
MIU_USER_RULES = {
    "affordability_or_cost": ("fee", "cost", "budget", "premium", "price", "annual fee", "afford", "payment"),
    "risk_or_suitability": ("risk", "loss", "volatility", "suitability", "concentration", "exposure", "safety"),
    "liquidity_or_flexibility": ("liquid", "liquidity", "redemption", "withdraw", "flexib", "cancellation"),
    "eligibility_or_compliance": ("eligible", "eligibility", "require", "must", "compliance", "qualify", "verification"),
    "compatibility_or_function": ("compatible", "size", "material", "feature", "function", "durability", "maintenance"),
    "time_access_or_logistics": ("time", "schedule", "duration", "distance", "delivery", "accessibility", "travel", "appointment"),
    "evidence_scope_or_uncertainty": ("evidence", "claim", "support", "refut", "uncertain", "scope", "conclusion"),
}
# Shared with quality.py's clean-evidence authorization check. This taxonomy is
# intentionally not emitted by the diversity report.
MIU_AUTHORIZATION_RULES = {
    "official_or_regulatory": ("official", "regulat", "compliance", "policy", "issuer", "government", "municipal"),
    "product_or_provider_documentation": ("product", "manufacturer", "prospectus", "disclosure", "specification", "provider", "retailer"),
    "financial_or_market_data": ("financial", "portfolio", "market", "exchange", "liquidity", "credit", "fund"),
    "research_or_expert_evidence": ("research", "study", "clinical", "peer reviewed", "assessment", "analyst"),
    "inspection_or_operational_record": ("inspection", "health", "record", "report", "weather", "forecast"),
}
MIU_ATTACK_RULES = {
    "authority_or_expert_pressure": ("expert", "analyst", "manager", "influencer", "endorsement", "star", "pseudo"),
    "scarcity_or_urgency": ("scarcity", "limited", "countdown", "bonus", "window", "offer", "queue", "fomo"),
    "social_proof_or_ranking": ("review", "ranking", "community", "testimonial", "consensus", "sponsored", "placement"),
    "financial_incentive_or_price_frame": ("cashback", "discount", "commission", "fee", "points", "promotion", "return", "profit"),
    "false_or_selective_evidence": ("fake", "fabricated", "misleading", "cherry", "misquoted", "false", "screenshot", "selective"),
    "fear_or_emotional_pressure": ("fear", "anxiety", "identity", "emotional", "guilt", "relationship", "fear marketing"),
}
EIL_OPENING_RULES = {
    "timing_or_availability_probe": ("when", "date", "schedule", "availability", "move", "deadline", "soon"),
    "budget_or_payment_probe": ("budget", "pay", "payment", "deposit", "fee", "price", "afford"),
    "document_or_qualification_probe": ("document", "proof", "verification", "reference", "qualif", "record"),
    "alternative_or_commitment_probe": ("other option", "alternative", "flexible", "commit", "competing", "fallback"),
    "scope_or_process_probe": ("scope", "service", "process", "request", "need", "details", "confirm"),
}
EIL_ADVERSE_MOVE_RULES = {
    "price_or_fee_worsening": ("price", "fee", "rent", "cost", "payment", "deposit"),
    "term_or_scope_worsening": ("term", "scope", "service", "condition", "contract", "coverage"),
    "delay_or_procedural_burden": ("delay", "process", "review", "document", "require", "hurdle"),
    "access_or_remedy_restriction": ("access", "reject", "decline", "remedy", "availability", "eligib"),
    "commitment_or_choice_pressure": ("commit", "pressure", "accelerate", "deadline", "choice", "settlement"),
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_many(paths: Iterable[Path]) -> list[dict[str, Any]]:
    return [row for path in paths for row in read_jsonl(path)]


def normalized(text: str) -> str:
    return " ".join(TOKEN.findall(text.lower()))


def tags(text: str, rules: dict[str, tuple[str, ...]]) -> tuple[str, ...]:
    lowered = text.lower()
    result = tuple(name for name, needles in rules.items() if any(needle in lowered for needle in needles))
    return result or ("other",)


def tagset(values: Iterable[tuple[str, ...]]) -> str:
    """Return a deterministic multi-label profile for quality diagnostics."""
    return "|".join(sorted({tag for value in values for tag in value})) or "other"


def taxonomy_report(rows: list[dict[str, Any]], extract: Callable[[dict[str, Any]], list[tuple[str, ...]]], name: str) -> dict[str, Any]:
    """Report a text-derived functional taxonomy and its coverage limits.

    ``other`` is deliberately retained as an audit outcome: the rules are an
    interpretable measurement instrument, not a gold semantic annotation.
    """
    def summarize(group: list[dict[str, Any]]) -> dict[str, Any]:
        incidences = []
        unmapped_fields = total_fields = 0
        for row in group:
            field_tags = extract(row)
            total_fields += len(field_tags)
            unmapped_fields += sum(value == ("other",) for value in field_tags)
            incidences.extend(tag for field in field_tags for tag in field)
        return {
            "taxonomy_measurement": {
                "tagging_basis": "deterministic released-text rules",
                "fields": total_fields,
                "unmapped_other_fields": unmapped_fields,
                "mapped_field_rate": round((total_fields - unmapped_fields) / total_fields, 4) if total_fields else None,
            },
            "field_type_incidence": distribution(incidences),
        }
    by_family: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_family.setdefault(row["family_domain"], []).append(row)
    return {
        "definition": name,
        **summarize(rows),
        "by_family_domain": {family: summarize(group) for family, group in sorted(by_family.items())},
    }


def duplicate_excess(values: list[str]) -> int:
    return sum(count - 1 for count in Counter(values).values() if count > 1)


def distribution(values: list[Any]) -> dict[str, Any]:
    """Support, balance and dominance over the valid *observed* support."""
    counts = Counter(map(str, values))
    total = sum(counts.values())
    if not total:
        return {"observations": 0, "support": 0, "effective_support": 0.0, "normalized_entropy": None, "dominance": None, "distribution": {}}
    entropy = -sum((count / total) * math.log(count / total) for count in counts.values())
    support = len(counts)
    return {
        "observations": total,
        "support": support,
        "effective_support": round(math.exp(entropy), 3),
        "normalized_entropy": round(entropy / math.log(support), 4) if support > 1 else 1.0,
        "dominance": round(max(counts.values()) / total, 4),
        "distribution": dict(sorted(counts.items())),
    }


def eil_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "field_diversity": {
            "necessary_information_types": taxonomy_report(rows, lambda row: [tags(item["content"], EIL_NECESSARY_RULES) for item in row["necessary_information"]], "functional types of information the user can safely disclose"),
            "private_leverage_types": taxonomy_report(rows, lambda row: [tags(item["content"], EIL_LEVERAGE_RULES) for item in row["exploitable_slots"]], "functional types of user facts whose disclosure creates counterparty leverage"),
            "opening_probe_types": taxonomy_report(rows, lambda row: [tags(row["adversary_opening"], EIL_OPENING_RULES)], "functional types of first-turn counterparty probes"),
            "conditional_adverse_move_types": taxonomy_report(rows, lambda row: [tags(row["adversary_config"]["strategy"], EIL_ADVERSE_MOVE_RULES)], "types of worse terms or process actions described in adversary strategy"),
        },
    }


def miu_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "field_diversity": {
            "constraint_and_preference_types": taxonomy_report(rows, lambda row: [tags(item["content"], MIU_USER_RULES) for item in row["user_constraints"] + row["user_preferences"]], "functional types of user requirements and ranked preferences"),
            "manipulation_mechanism_types": taxonomy_report(rows, lambda row: [tags(card["attack"], MIU_ATTACK_RULES) for card in row["manipulated_context"]], "functional mechanisms represented by manipulation labels"),
        },
    }


def profile_mismatches(records: list[dict[str, Any]], audits: dict[str, dict[str, Any]]) -> list[str]:
    mismatches = []
    for record in records:
        plan = audits.get(record["id"], {}).get("generation_profile", {}).get("record_plan")
        if not plan:
            mismatches.append(f"{record['id']}: missing generation_profile")
            continue
        counts = plan["counts"]
        if record["mechanism"] == "EIL":
            if record["meta"]["num_nec"] != counts["num_nec"] or record["meta"]["num_exp"] != counts["num_exp"]:
                mismatches.append(f"{record['id']}: EIL field counts differ from profile")
            if set(record["adversary_config"].get("tactics", [])) != set(plan["adversary_tactics"]):
                mismatches.append(f"{record['id']}: EIL tactic differs from profile")
        else:
            keys = {"num_conditions": "num_conditions", "num_preferences": "num_preferences", "num_auth": "num_auth", "num_clean": "num_clean", "num_mani": "num_mani"}
            if any(record["meta"][actual] != counts[expected] for actual, expected in keys.items()):
                mismatches.append(f"{record['id']}: MIU field counts differ from profile")
            attacks = {normalized(card.get("attack", "")) for card in record["manipulated_context"]}
            if normalized(plan["attack"]) not in attacks:
                mismatches.append(f"{record['id']}: requested MIU attack absent")
    return mismatches


def subscenario_gate(records: list[dict[str, Any]], audits: dict[str, dict[str, Any]], require_complete: bool) -> dict[str, Any]:
    errors: list[str] = []
    if require_complete and len(records) != len(audits):
        errors.append("record/audit row count differs")
    missing = [record["id"] for record in records if record["id"] not in audits]
    if missing:
        errors.append(f"missing audit rows: {len(missing)}")
    mismatches = profile_mismatches(records, audits)
    if mismatches:
        errors.append(f"generation-profile mismatches: {len(mismatches)}")
    mechanism = records[0]["mechanism"] if records else None
    requests = [normalized(record.get("user_natural_language", "")) for record in records]
    duplicates = duplicate_excess(requests)
    if duplicates:
        errors.append("duplicate normalized user_natural_language")
    return {"metric_version": "task-conditioned-diversity-v6", "records": len(records), "mechanism": mechanism, "errors": errors, "generation_profile_mismatches": mismatches[:20], "exact_duplicate_user_requests": duplicates, "release_gate": "pass" if not errors else "fail"}


def dataset_report(eil_rows: list[dict[str, Any]], miu_rows: list[dict[str, Any]]) -> dict[str, Any]:
    ids = [row["id"] for row in eil_rows + miu_rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate IDs across EIL and MIU")
    return {
        "metric_version": "task-conditioned-diversity-v6",
        "methodology": {
            "primary_claim": "Diversity is interaction-scenario coverage and variation in loyalty-relevant field types, not lexical variation or field-count configurations.",
            "primary_metrics": ["family-domain and subscenario coverage", "EIL necessary-information and protected-leverage types", "EIL adversary-profile types", "MIU user-profile types", "MIU manipulation-mechanism types"],
            "no_composite_score": "A weighted total can hide a missing scenario or loyalty-relevant field type behind unrelated high values.",
        },
        "corpus": {
            "EIL": {"records": len(eil_rows), "splits": dict(sorted(Counter(row["split"] for row in eil_rows).items())), "family_domain": distribution([row["family_domain"] for row in eil_rows]), "subscenario": distribution([row["subscenario"] for row in eil_rows]), "family_domain_to_subscenarios": {domain: sorted({row["subscenario"] for row in eil_rows if row["family_domain"] == domain}) for domain in sorted({row["family_domain"] for row in eil_rows})}},
            "MIU": {"records": len(miu_rows), "splits": dict(sorted(Counter(row["split"] for row in miu_rows).items())), "family_domain": distribution([row["family_domain"] for row in miu_rows]), "subscenario": distribution([row["subscenario"] for row in miu_rows]), "family_domain_to_subscenarios": {domain: sorted({row["subscenario"] for row in miu_rows if row["family_domain"] == domain}) for domain in sorted({row["family_domain"] for row in miu_rows})}},
        },
        "EIL": eil_metrics(eil_rows), "MIU": miu_metrics(miu_rows),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_subparsers(dest="mode", required=True)
    job = mode.add_parser("subscenario", help="run the completed-job diversity release gate")
    job.add_argument("--records", type=Path, required=True); job.add_argument("--audit", type=Path, required=True)
    job.add_argument("--report", type=Path); job.add_argument("--require-complete", action="store_true")
    dataset = mode.add_parser("dataset", help="report release-facing diversity for EIL and MIU")
    dataset.add_argument("--eil-dir", type=Path, required=True); dataset.add_argument("--miu-dir", type=Path, required=True)
    dataset.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "subscenario":
        records, audits = read_jsonl(args.records), {row["record_id"]: row for row in read_jsonl(args.audit)}
        result = subscenario_gate(records, audits, args.require_complete)
        code = 0 if not result["errors"] else 1
    else:
        eil_rows = read_many([args.eil_dir / f"{split}.jsonl" for split in SPLITS])
        miu_rows = read_many([args.miu_dir / f"{split}.jsonl" for split in SPLITS])
        if any(row.get("mechanism") != "EIL" for row in eil_rows) or any(row.get("mechanism") != "MIU" for row in miu_rows):
            raise ValueError("input directories contain a record with the wrong mechanism")
        result, code = dataset_report(eil_rows, miu_rows), 0
    output = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True); args.report.write_text(output, encoding="utf-8")
    print(output, end="")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
