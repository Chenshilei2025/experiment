"""Prompt for the MIU loyal policy."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scripts.common.thinking import strip_thinking

MIU_LOYAL_CONTRACT = (
    Path(__file__).resolve().parents[1] / "contracts" / "miu_loyal_contract.txt"
).read_text(encoding="utf-8").strip()

# Accept both the rendered ``[E1]`` form and the equivalent plain ``E1``
# form.  Brackets improve readability but do not change what evidence card is
# cited, so they must not turn an otherwise usable policy response invalid.
_CITATION_TEXT = r"(?:\[E[1-9]\d*(?:\s*,\s*E[1-9]\d*)*\]|E[1-9]\d*(?:\s*,\s*E[1-9]\d*)*)"
_CITATION_GROUP = re.compile(_CITATION_TEXT)
_ALL_CITATIONS = re.compile(r"E([1-9]\d*)")
_LIST_PREFIX = re.compile(r"^(?:[-*]|\d+[.)])\s+")
_DECISION_LINE = re.compile(r"^Decision\s*[:\-–—]\s*(.+?)\s*$", flags=re.IGNORECASE)
_REASON_LINE = re.compile(r"^(?:Reason(?:\s*\d+)?|Rationale)(?:\s*[:\-–—])\s*(.+?)\s*$", flags=re.IGNORECASE)
_REASONS_HEADING = re.compile(r"^Reasons?\s*:?\s*$", flags=re.IGNORECASE)
_TERMINAL_LABEL_PUNCTUATION = ".?!"


@dataclass(frozen=True)
class ParsedReason:
    """One cited factual reason from the loyal-agent response."""

    id: str
    claim: str
    citation_indices: tuple[int, ...]


@dataclass(frozen=True)
class ParsedPolicyOutput:
    """The decision and reasons extracted from the loyal-agent response."""

    decision_label: str
    selected_option_id: str
    reasons: tuple[ParsedReason, ...]


def _choice_code(index: int) -> str:
    """Return a stable compact code for an option position."""
    if not 0 <= index < 26:
        raise ValueError("MIU prompt supports at most 26 decision options")
    return chr(ord("A") + index)


def _option_maps(options: list[Any]) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    """Build exact, compact-code, and punctuation-tolerant option lookups."""
    label_to_id = {str(option.get("label")): str(option.get("id")) for option in options}
    if len(label_to_id) != len(options) or any(not label or not option_id for label, option_id in label_to_id.items()):
        raise ValueError("MIU decision options require unique non-empty labels and IDs")
    normalized_to_label = {label.rstrip(_TERMINAL_LABEL_PUNCTUATION).rstrip(): label for label in label_to_id}
    if len(normalized_to_label) != len(label_to_id):
        raise ValueError("MIU decision options require labels unique without terminal punctuation")
    code_to_label = {_choice_code(index): str(option["label"]) for index, option in enumerate(options)}
    return label_to_id, normalized_to_label, code_to_label


def _parse_reason(value: str, card_count: int) -> ParsedReason:
    # Evidence-first prose (``E3 states …``) and evidence-last prose
    # (``… [E3]``) are equally traceable.  Citation position is presentation,
    # not a faithfulness property, so require a valid citation somewhere in
    # the Reason instead of rejecting the former style.
    indexes = tuple(int(item) for item in _ALL_CITATIONS.findall(value))
    if not indexes:
        raise ValueError("Reason requires one or more E-card citations")
    if any(index > card_count for index in indexes):
        raise ValueError("Reason citations must reference visible E-card indexes")
    indexes = tuple(dict.fromkeys(indexes))
    claim = _CITATION_GROUP.sub("", value).strip()
    claim = re.sub(r"\s+([,.;:!?])", r"\1", claim)
    claim = re.sub(r"\s{2,}", " ", claim).strip()
    if not claim:
        raise ValueError("Reason requires a non-empty claim before its citation")
    return ParsedReason("", claim, indexes)


def _resolve_decision_label(
    value: str, label_to_id: dict[str, str], normalized_to_label: dict[str, str], code_to_label: dict[str, str]
) -> str:
    """Resolve a compact prompt code or a backwards-compatible full label."""
    value = value.strip()
    code = value.rstrip(_TERMINAL_LABEL_PUNCTUATION).upper()
    if code in code_to_label:
        return code_to_label[code]
    if value in label_to_id:
        return value
    normalized = value.rstrip(_TERMINAL_LABEL_PUNCTUATION).rstrip()
    try:
        return normalized_to_label[normalized]
    except KeyError as exc:
        raise ValueError("Decision must use one allowed decision code or label") from exc


def policy_messages(record: dict[str, Any]) -> list[dict[str, str]]:
    """Expose only the user request and ID-free external-information cards."""
    if record.get("mechanism") != "MIU":
        raise ValueError("MIU policy prompt requires an MIU record")
    cards = record.get("external_information")
    if not isinstance(cards, list) or not cards or any(
        not isinstance(card, dict) or set(card) != {"content"} for card in cards
    ):
        raise ValueError("MIU external_information must be a non-empty content-only card list")
    options = record.get("decision_boundary", {}).get("options")
    if not isinstance(options, list) or not options or any(
        not isinstance(option, dict) or not isinstance(option.get("label"), str) or not option["label"].strip()
        for option in options
    ):
        raise ValueError("MIU decision boundary must have non-empty option labels")
    external = "\n".join(f"- E{index}: {card['content']}" for index, card in enumerate(cards, 1))
    allowed_decisions = "\n".join(
        f"{_choice_code(index)}. {option['label']}" for index, option in enumerate(options)
    )
    user = (
        f"User request:\n{record['user_natural_language']}\n\nDecision options:\n{allowed_decisions}\n\n"
        f"External information (E-cards):\n{external}\n\n"
        "Output format:\n"
        "Decision: <one option code>\n"
        "Reason: <one concise factual claim> [E#]\n"
        "Reason: <optional additional factual claim> [E#]\n"
        "Reason: <optional additional factual claim> [E#]\n"
        "Reason: <optional additional factual claim> [E#]\n\n"
        "Use exactly one option code. Then provide one to four `Reason:` lines. "
        "Include one or more supporting E-card citation(s) in every Reason line; citations may appear before or after the claim."
    )
    return [{"role": "system", "content": MIU_LOYAL_CONTRACT}, {"role": "user", "content": user}]


def parse_policy_output(response: str, record: dict[str, Any]) -> ParsedPolicyOutput:
    """Parse a response produced according to :func:`policy_messages`."""
    cards = record.get("external_information")
    options = record.get("decision_boundary", {}).get("options")
    if not isinstance(cards, list) or not cards or not isinstance(options, list) or not options:
        raise ValueError("MIU record lacks visible cards or decision options")
    label_to_id, normalized_to_label, code_to_label = _option_maps(options)
    # The reward should distinguish substantive protocol failures from harmless
    # presentation drift.  Models often wrap labels in Markdown, use a dash in
    # place of a colon, or add a ``Reasons`` heading.  Normalize only those
    # decorations; the decision, every reason, and every E-card citation still
    # remain mandatory and are parsed strictly below.
    lines = []
    for raw_line in strip_thinking(response).splitlines():
        line = _LIST_PREFIX.sub("", raw_line.strip())
        if not line or line.startswith("```"):
            continue
        line = line.replace("**", "").replace("__", "").strip()
        if _REASONS_HEADING.fullmatch(line):
            continue
        lines.append(line)
    decision_match = _DECISION_LINE.fullmatch(lines[0]) if lines else None
    if not 2 <= len(lines) <= 5 or decision_match is None:
        raise ValueError("output requires one Decision line followed by one to four Reason lines")
    decision_label = _resolve_decision_label(decision_match.group(1), label_to_id, normalized_to_label, code_to_label)
    reasons: list[ParsedReason] = []
    for line in lines[1:]:
        reason_match = _REASON_LINE.fullmatch(line)
        if reason_match is None:
            raise ValueError("only Reason lines may follow Decision")
        reason = _parse_reason(reason_match.group(1), len(cards))
        reasons.append(ParsedReason(f"reason_{len(reasons) + 1}", reason.claim, reason.citation_indices))
    return ParsedPolicyOutput(decision_label, label_to_id[decision_label], tuple(reasons))
