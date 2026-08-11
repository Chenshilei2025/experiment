#!/usr/bin/env python3
"""Write a human-readable inventory of the 42 isolated generation pipelines."""

from __future__ import annotations

from pathlib import Path

from ..generation import builder as pipeline


DATA = Path(__file__).resolve().parents[2] / "data"
DOCS = DATA / "docs"


def main() -> int:
    lines = [
        "# Per-Subscenario Data Pipelines",
        "",
        "Every job activates exactly one row. The runner reads only that row's source allowlist, extracts a small per-record anchor packet, and sends the original matching `prompt.md` block plus that packet in one model call. `none` means the packet has no external facts and the prompt permits controlled synthesis.",
        "",
        "| # | Subscenario | Mechanism | Local source allowlist | Anchor handling |",
        "|---:|---|---|---|---|",
    ]
    for block in pipeline.load_prompt_blocks():
        sources = ", ".join(block.sources) if block.sources else "none"
        handling = "controlled synthesis" if not block.sources else "approved factual paraphrase anchors; provenance stays private"
        mechanism = "EIL" if block.family == "delegated" else "MIU"
        lines.append(f"| {block.index} | {block.scenario} | {mechanism} | {sources} | {handling} |")
    lines.extend([
        "",
        "## Per-record controls",
        "",
        "- `generation_profile` is sampled deterministically from the active subscenario and ordinal. It fixes counts and non-factual diversity dimensions, never the MIU baseline or an evidence graph.",
        "- The model must construct fields in the order specified by the active scenario prompt but returns one complete JSON object.",
        "- The driver rejects wrong counts, EIL tactic mismatch, duplicate MIU attacks, source-anchor copying, invalid final fields, and final partition/ID errors. The audit sidecar retains the seed, profile, source anchor provenance, and retry count.",
    ])
    DOCS.mkdir(exist_ok=True)
    (DOCS / "PIPELINE_INVENTORY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
