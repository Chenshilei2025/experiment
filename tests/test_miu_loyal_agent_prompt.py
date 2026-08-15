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
    assert "Decision: <one uppercase option code, such as A>" in content
    assert "no Markdown, bullets, headings, code fences, or extra text" in content
    assert "Every Reason must contain one or more" in content
    assert "These are requirements, not preferences." in content


def test_parser_maps_compact_code_to_option():
    parsed = parse_policy_output("Decision: A\nReason: Option A meets the requirement [E1]", _record())
    assert (parsed.selected_option_id, parsed.decision_label) == ("opt_1", "Option A.")


def test_parser_rejects_noncanonical_decision_forms():
    for decision in ("A.", "a", "Option A", "Option A."):
        try:
            parse_policy_output(f"Decision: {decision}\nReason: Option A meets the requirement [E1]", _record())
        except ValueError:
            continue
        raise AssertionError("parser must require an uppercase option code")


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


def test_parser_requires_explicit_square_bracket_citations():
    for response in (
        "Decision: A\nReason: Option A meets the requirement E1",
        "Decision: A\nReason: [E1] states that Option A meets the requirement.",
    ):
        try:
            parse_policy_output(response, _record())
        except ValueError:
            continue
        raise AssertionError("parser must require an explicit [E#] citation")


def test_parser_accepts_punctuation_after_and_inline_citations():
    parsed = parse_policy_output(
        "Decision: A\nReason: Option A meets the requirement [E1], so it is appropriate.",
        _record(),
    )
    assert parsed.reasons[0].claim == "Option A meets the requirement, so it is appropriate."
    assert parsed.reasons[0].citation_indices == (1,)

    punctuated = parse_policy_output(
        "Decision: A\nReason: Option A meets the requirement [E1].",
        _record(),
    )
    assert punctuated.reasons[0].citation_indices == (1,)


def test_parser_accepts_emphasis_wrappers_but_not_other_format_drift():
    parsed = parse_policy_output(
        "**Decision:** A\n**Reason:** Option A meets the requirement [E1]",
        _record(),
    )

    assert parsed.selected_option_id == "opt_1"
    assert parsed.reasons[0].citation_indices == (1,)

    for response in (
        "Decision: A\nReasons:\nReason: Option A meets the requirement [E1]",
        "Decision: A\n- Reason: Option A meets the requirement [E1]",
        "Decision: A\nRationale: Option A meets the requirement [E1]",
    ):
        try:
            parse_policy_output(response, _record())
        except ValueError:
            continue
        raise AssertionError("parser must reject noncanonical Reason formatting")
