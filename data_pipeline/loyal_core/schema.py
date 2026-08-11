"""Final-schema records shared by training and evaluation."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Literal, TypedDict

Mechanism = Literal["EIL", "MIU"]

EIL_REQUIRED = {
    "id", "mechanism", "family_domain", "subscenario", "split",
    "user_natural_language", "adversary_opening", "necessary_information",
    "exploitable_slots", "adversary_config", "meta",
}
MIU_REQUIRED = {
    "id", "mechanism", "family_domain", "subscenario", "split",
    "user_natural_language", "user_constraints", "user_preferences",
    "authorized_information", "decision_boundary", "clean_context",
    "manipulated_context", "external_information", "baseline_decision",
    "schema_notes", "meta",
}


def validate_record(record: dict[str, Any]) -> dict[str, Any]:
    """Validate the persisted final schema without leaking evaluator fields."""
    mechanism = record.get("mechanism")
    expected = EIL_REQUIRED if mechanism == "EIL" else MIU_REQUIRED if mechanism == "MIU" else None
    if expected is None:
        raise ValueError(f"{record.get('id', '<unknown>')}: invalid mechanism {mechanism!r}")
    missing = expected - set(record)
    extra = set(record) - expected
    if missing or extra:
        raise ValueError(
            f"{record.get('id', '<unknown>')}: final-schema mismatch; "
            f"missing={sorted(missing)}, extra={sorted(extra)}"
        )
    if mechanism == "EIL":
        if not record["necessary_information"] or not record["exploitable_slots"]:
            raise ValueError(f"{record['id']}: EIL needs necessary_information and exploitable_slots")
    else:
        options = record["decision_boundary"].get("options", [])
        option_ids = {option["id"] for option in options}
        if [option.get("id") for option in options] != ["opt_1", "opt_2", "opt_3"]:
            raise ValueError(f"{record['id']}: MIU decision options must be canonical and ordered")
        labels = [" ".join(str(option.get("label", "")).casefold().split()) for option in options]
        if len(set(labels)) != 3 or any(not label for label in labels):
            raise ValueError(f"{record['id']}: MIU decision options must have distinct labels")
        if record["baseline_decision"].get("decision") not in option_ids:
            raise ValueError(f"{record['id']}: baseline decision is outside decision_boundary")
        clean_ids = {card["id"] for card in record["clean_context"]}
        if not set(record["baseline_decision"].get("grounded_on", [])).issubset(clean_ids):
            raise ValueError(f"{record['id']}: baseline cites a non-clean evidence card")
        authorized_ids = {source["id"] for source in record["authorized_information"]}
        authorized_names = {" ".join(str(source.get("name", "")).casefold().split()) for source in record["authorized_information"]}
        if len(authorized_names) != len(authorized_ids) or "" in authorized_names:
            raise ValueError(f"{record['id']}: MIU authorization names must be distinct")
        if any(card.get("authorized_by") not in authorized_ids for card in record["clean_context"]):
            raise ValueError(f"{record['id']}: clean evidence has an unknown authorization")
        if any(set(card) != {"id", "content", "authorized_by"} for card in record["clean_context"]):
            raise ValueError(f"{record['id']}: clean_context has unused provenance metadata")
        if any(set(card) != {"id", "content", "attack"} for card in record["manipulated_context"]):
            raise ValueError(f"{record['id']}: manipulated_context has unused provenance metadata")
        visible_cards = [card.get("content") for card in record["external_information"]]
        hidden_cards = [card.get("content") for card in record["clean_context"] + record["manipulated_context"]]
        if any(set(card) != {"content"} for card in record["external_information"]):
            raise ValueError(f"{record['id']}: external information must be content-only")
        if len(visible_cards) != len(set(visible_cards)):
            raise ValueError(f"{record['id']}: external_information has duplicate cards")
        if sorted(visible_cards) != sorted(hidden_cards):
            raise ValueError(f"{record['id']}: external_information is not the exact clean/manipulated union")
    return record


def is_eil(record: dict[str, Any]) -> bool:
    return record["mechanism"] == "EIL"


def option_label(record: dict[str, Any], option_id: str) -> str:
    """Return a stable human-readable label for a decision option id."""
    for option in record["decision_boundary"]["options"]:
        if option["id"] == option_id:
            return option["label"]
    return option_id


def render_items(items: list[dict[str, Any]], fields: tuple[str, ...]) -> str:
    """Render final-schema cards without exposing fields outside the caller's set."""
    lines: list[str] = []
    for item in items:
        body = "; ".join(f"{field}={item[field]}" for field in fields if field in item)
        lines.append(f"- {body}")
    return "\n".join(lines) or "- none"


def load_jsonl(paths: str | Path | Iterable[str | Path]) -> list[dict[str, Any]]:
    """Load one or more final-schema JSONL files."""
    if isinstance(paths, (str, Path)):
        paths = [paths]
    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for path_like in paths:
        path = Path(path_like)
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                record = validate_record(json.loads(line))
                if record["id"] in seen_ids:
                    raise ValueError(f"duplicate record id {record['id']} in {path}:{line_number}")
                seen_ids.add(record["id"])
                records.append(record)
    return records
