"""Choose EIL or MIU chunks from within-task reward-distribution signals.

Raw rewards are deliberately not compared across mechanisms: their objectives
have different scales.  Instead, this controller compares each mechanism's
recent reward location and group-level GRPO spread against its own history.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

MECHANISMS = ("eil", "miu")
METRICS = ("raw_reward_mean", "group_reward_std_mean")


def _load_json(path: Path, default: dict[str, Any]) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else default
    except (OSError, json.JSONDecodeError):
        return default


def _events(path: Path) -> list[dict[str, float | str]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    result = []
    for line in lines:
        try:
            event = json.loads(line)
            if event.get("mechanism") in MECHANISMS and all(isinstance(event.get(key), (int, float)) for key in METRICS):
                result.append(event)
        except json.JSONDecodeError:
            continue
    return result


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _std(values: list[float]) -> float:
    mean = _mean(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))


def _need(history: list[dict[str, float | str]], recent_size: int) -> float | None:
    if len(history) < recent_size * 2:
        return None
    recent, baseline = history[-recent_size:], history[:-recent_size]
    recent_reward = _mean([float(item["raw_reward_mean"]) for item in recent])
    baseline_rewards = [float(item["raw_reward_mean"]) for item in baseline]
    baseline_reward = _mean(baseline_rewards)
    # Lower recent quality means this task needs attention; normalize only
    # against its own history so EIL and MIU reward scales never mix.
    deficit = max(-2.0, min(2.0, (baseline_reward - recent_reward) / (_std(baseline_rewards) + 1e-6)))
    recent_spread = _mean([float(item["group_reward_std_mean"]) for item in recent])
    baseline_spread = _mean([float(item["group_reward_std_mean"]) for item in baseline])
    # High group spread provides useful relative-ranking signal for GRPO.
    signal = max(0.5, min(1.5, recent_spread / (baseline_spread + 1e-6)))
    return 0.5 * deficit + 0.5 * (signal - 1.0)


def choose(args: argparse.Namespace) -> str:
    default: dict[str, Any] = {"event_count": 0, "chosen": {"eil": 0, "miu": 0}, "history": {"eil": [], "miu": []}}
    state = _load_json(args.state, default)
    state.setdefault("event_count", 0)
    state.setdefault("chosen", {"eil": 0, "miu": 0})
    state.setdefault("history", {"eil": [], "miu": []})
    all_events = _events(args.signals)
    for event in all_events[int(state["event_count"]):]:
        mechanism = str(event["mechanism"])
        state["history"].setdefault(mechanism, []).append(event)
        state["history"][mechanism] = state["history"][mechanism][-args.history_size:]
    state["event_count"] = len(all_events)

    eil_need = _need(state["history"].get("eil", []), args.recent_size)
    miu_need = _need(state["history"].get("miu", []), args.recent_size)
    probability = args.initial_eil_probability
    if eil_need is not None and miu_need is not None:
        probability += args.adaptation_strength * (eil_need - miu_need)
    probability = max(args.min_eil_probability, min(args.max_eil_probability, probability))

    chosen = state["chosen"]
    total = int(chosen.get("eil", 0)) + int(chosen.get("miu", 0))
    mechanism = "eil" if int(chosen.get("eil", 0)) < probability * (total + 1) else "miu"
    chosen[mechanism] = int(chosen.get(mechanism, 0)) + 1
    state["last_eil_probability"] = probability
    state["last_need"] = {"eil": eil_need, "miu": miu_need}
    args.state.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.state.with_name(f".{args.state.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, args.state)
    return mechanism


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signals", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--initial-eil-probability", type=float, default=1 / 3)
    parser.add_argument("--min-eil-probability", type=float, default=0.25)
    parser.add_argument("--max-eil-probability", type=float, default=0.60)
    parser.add_argument("--adaptation-strength", type=float, default=0.15)
    parser.add_argument("--recent-size", type=int, default=10)
    parser.add_argument("--history-size", type=int, default=200)
    args = parser.parse_args()
    if not 0 <= args.min_eil_probability <= args.initial_eil_probability <= args.max_eil_probability <= 1:
        parser.error("EIL probabilities must satisfy 0 <= min <= initial <= max <= 1")
    if args.recent_size < 1 or args.history_size < args.recent_size * 2:
        parser.error("history-size must be at least twice recent-size")
    print(choose(args))


if __name__ == "__main__":
    main()
