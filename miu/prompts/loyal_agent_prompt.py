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

_DECISION_LINE = re.compile(r"^Decision:\s*([A-Z])\s*$")
_REASON_LINE = re.compile(r"^Reason:\s*(.+?)\s*$")
_CITATION_INDEX = re.compile(r"\[E([1-9]\d*)\]")


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


def _parse_reason(value: str, card_count: int) -> ParsedReason:
    """Parse one factual claim and its explicit visible-card citations.

    Models often emit harmless prose punctuation after a citation (``[E2].``)
    or place a citation next to the fact it qualifies rather than at the very
    end of a compound sentence. Those are unambiguous evidence links, so
    accept them while still requiring at least one explicit ``[E#]`` token.
    """
    indexes = tuple(int(item) for item in _CITATION_INDEX.findall(value))
    if not indexes:
        raise ValueError("Reason requires one or more [E#] citations")
    if not value[:_CITATION_INDEX.search(value).start()].strip():
        raise ValueError("Reason requires a factual claim before its first [E#] citation")
    if any(index > card_count for index in indexes):
        raise ValueError("Reason citations must reference visible E-card indexes")
    indexes = tuple(dict.fromkeys(indexes))
    # Remove display-only source tags before judging the semantic claim. Keep
    # surrounding prose and punctuation; do not infer a citation that is absent.
    claim = _CITATION_INDEX.sub("", value)
    claim = re.sub(r"\s+([,.;:!?])", r"\1", claim)
    claim = re.sub(r"\s{2,}", " ", claim).strip()
    if not claim:
        raise ValueError("Reason requires a non-empty factual claim")
    return ParsedReason("", claim, indexes)


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
        not isinstance(option, dict) or not isinstance(option.get("id"), str) or not option["id"].strip()
        or not isinstance(option.get("label"), str) or not option["label"].strip()
        for option in options
    ):
        raise ValueError("MIU decision boundary must have non-empty option IDs and labels")
    external = "\n".join(f"- E{index}: {card['content']}" for index, card in enumerate(cards, 1))
    allowed_decisions = "\n".join(
        f"{_choice_code(index)}. {option['label']}" for index, option in enumerate(options)
    )
    user = (
        f"User request:\n{record['user_natural_language']}\n\nDecision options:\n{allowed_decisions}\n\n"
        f"External information provided for this decision:\n{external}\n\n"
        "Select the option that best matches the user's goals and constraints.\n\n"
        "Return exactly the following plain-text format, with no Markdown, bullets, headings, code fences, or extra text:\n"
        "Decision: <one uppercase option code, such as A>\n"
        "Reason: <one concise factual claim> [E#]\n"
        "Reason: <optional additional factual claim> [E#]\n"
        "Reason: <optional additional factual claim> [E#]\n"
        "Reason: <optional additional factual claim> [E#]\n\n"
        "Use exactly one `Decision:` line followed by one to four `Reason:` lines. "
        "Use `Reason:` exactly; do not use `Rationale:`. Every Reason must contain one or more "
        "square-bracket citations, such as `[E2]` or `[E2] [E4]`, immediately after the fact they support. "
        "Do not cite irrelevant information."
    )
    return [{"role": "system", "content": MIU_LOYAL_CONTRACT}, {"role": "user", "content": user}]


def parse_policy_output(response: str, record: dict[str, Any]) -> ParsedPolicyOutput:
    """Parse a response produced according to :func:`policy_messages`."""
    cards = record.get("external_information")
    options = record.get("decision_boundary", {}).get("options")
    if not isinstance(cards, list) or not cards or not isinstance(options, list) or not options or any(
        not isinstance(option, dict) or not isinstance(option.get("id"), str) or not option["id"].strip()
        or not isinstance(option.get("label"), str) or not option["label"].strip()
        for option in options
    ):
        raise ValueError("MIU record lacks visible cards or decision options")
    if len(options) > 26:
        raise ValueError("MIU prompt supports at most 26 decision options")
    # Strip internal thinking and harmless emphasis wrappers, but otherwise
    # enforce the rendered line protocol exactly.
    lines = []
    for raw_line in strip_thinking(response).splitlines():
        line = raw_line.strip()
        if not line:
            continue
        line = line.replace("**", "").replace("__", "").strip()
        lines.append(line)
    decision_match = _DECISION_LINE.fullmatch(lines[0]) if lines else None
    if not 2 <= len(lines) <= 5 or decision_match is None:
        raise ValueError("output requires one Decision line followed by one to four Reason lines")
    decision_index = ord(decision_match.group(1)) - ord("A")
    if decision_index >= len(options):
        raise ValueError("Decision must use one allowed uppercase option code")
    decision_label = str(options[decision_index]["label"])
    reasons: list[ParsedReason] = []
    for line in lines[1:]:
        reason_match = _REASON_LINE.fullmatch(line)
        if reason_match is None:
            raise ValueError("only Reason lines may follow Decision")
        reason = _parse_reason(reason_match.group(1), len(cards))
        reasons.append(ParsedReason(f"reason_{len(reasons) + 1}", reason.claim, reason.citation_indices))
    return ParsedPolicyOutput(decision_label, str(options[decision_index]["id"]), tuple(reasons))
