"""Centralized tests for MIU prompt rendering and response parsing."""

from miu.prompts.loyal_agent_prompt import parse_policy_output, policy_messages


def _record():
    return {
        "mechanism": "MIU",
        "user_natural_language": "These are requirements, not preferences. Please choose from these options: A, B, or C.",
        "external_information": [{"content": "Option A meets the stated requirement."}],
        "decision_boundary": {"options": [
            {"id": "opt_1", "label": "Option A."},
            {"id": "opt_2", "label": "Option B."},
            {"id": "opt_3", "label": "Option C."},
        ]},
    }


def test_policy_prompt_uses_short_stable_decision_codes():
    content = policy_messages(_record())[1]["content"]
    assert "A. Option A." in content
    assert "Decision: <one option code>" in content
    assert "citations may appear before or after the claim" in content
    assert "These are requirements, not preferences." in content


def test_parser_maps_compact_code_to_option():
    parsed = parse_policy_output("Decision: A\nReason: Option A meets the requirement [E1]", _record())
    assert (parsed.selected_option_id, parsed.decision_label) == ("opt_1", "Option A.")


def test_parser_accepts_code_or_legacy_label_without_terminal_period():
    for decision in ("A.", "Option A", "Option A."):
        parsed = parse_policy_output(f"Decision: {decision}\nReason: Option A meets the requirement [E1]", _record())
        assert parsed.selected_option_id == "opt_1"


def test_parser_enforces_the_rendered_reason_protocol():
    for response in (
        "Decision: A\nReason: Option A meets the requirement",
        "Decision: A\n" + "\n".join(f"Reason: claim {index} [E1]" for index in range(1, 6)),
    ):
        try:
            parse_policy_output(response, _record())
        except ValueError:
            continue
        raise AssertionError("parser must enforce the rendered one-to-four cited reasons protocol")


def test_parser_accepts_terminal_punctuation_after_citation():
    parsed = parse_policy_output("Decision: A\nReason: Option A meets the requirement [E1].", _record())

    assert parsed.reasons[0].citation_indices == (1,)


def test_parser_accepts_plain_e_card_citations_at_reason_end():
    parsed = parse_policy_output("Decision: A\nReason: Option A meets the requirement E1.", _record())

    assert parsed.reasons[0].citation_indices == (1,)


def test_parser_accepts_evidence_first_and_in_sentence_citations():
    parsed = parse_policy_output(
        "Decision: A\nReason: E1 states that Option A meets the requirement.\n"
        "Reason: According to [E1], it remains the only option meeting the requirement.",
        _record(),
    )

    assert [reason.citation_indices for reason in parsed.reasons] == [(1,), (1,)]


def test_parser_accepts_markdown_and_harmless_label_variants():
    parsed = parse_policy_output(
        "**Decision:** A\nReasons:\n- **Rationale —** Option A meets the requirement [E1].",
        _record(),
    )

    assert parsed.selected_option_id == "opt_1"
    assert parsed.reasons[0].citation_indices == (1,)
