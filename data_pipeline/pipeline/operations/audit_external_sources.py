#!/usr/bin/env python3
"""Audit source availability and extractor readiness for all 42 scenarios.

The audit distinguishes absent files from a present source whose optional
runtime dependency is unavailable. It never downloads or mutates source data.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from ..generation import builder


def audit() -> dict[str, Any]:
    blocks = builder.load_prompt_blocks()
    sources = sorted({source for block in blocks for source in block.sources})
    source_status = {source: {"extractor": source in builder.EXTRACTORS, "scenarios": []} for source in sources}
    scenario_status: list[dict[str, Any]] = []
    for block in blocks:
        status, error, fact_count = "controlled", None, 0
        if block.sources:
            for source in block.sources:
                source_status[source]["scenarios"].append(block.scenario)
            try:
                facts = builder.extract_source_facts(block, random.Random(20260801 + block.index))
                status, fact_count = "ready", len(facts)
            except ModuleNotFoundError as exc:
                status, error, fact_count = "present_missing_runtime_dependency", str(exc), None
            except Exception as exc:
                status, error, fact_count = "extractor_error", f"{type(exc).__name__}: {exc}", None
        scenario_status.append({
            "index": block.index, "subscenario": block.scenario,
            "family": "EIL" if block.family == "delegated" else "MIU",
            "family_domain": builder.family_domain(block), "sources": list(block.sources),
            "status": status, "fact_count": fact_count, "error": error,
        })
    return {
        "audit_version": "external-sources-v1",
        "source_root": str(builder.DATA / "external_benchmark"),
        "scenario_count": len(blocks), "referenced_source_count": len(sources),
        "source_status": source_status, "scenarios": scenario_status,
        "summary": {
            "controlled_scenarios": sum(item["status"] == "controlled" for item in scenario_status),
            "ready_scenarios": sum(item["status"] == "ready" for item in scenario_status),
            "dependency_blocked_scenarios": sum(item["status"] == "present_missing_runtime_dependency" for item in scenario_status),
            "extractor_error_scenarios": sum(item["status"] == "extractor_error" for item in scenario_status),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, help="optional JSON report path")
    args = parser.parse_args()
    result = audit()
    output = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(output, encoding="utf-8")
    print(output, end="")
    return 0 if result["summary"]["extractor_error_scenarios"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
