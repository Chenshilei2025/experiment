"""Batch-compatible exact-match reward for Experiment G DAPO-Math GRPO."""

from __future__ import annotations

import re
from typing import Any


def _normalize(text: str) -> str:
    text = "".join(str(text).lower().split())
    return text.replace("\\boxed{", "").replace("}", "")


def _matches_label(predicted: str, ground_truth: str) -> bool:
    """Compare normal answers, plus the dataset's 0--4 MC option encoding.

    DAPO-Math stores a small subset of multiple-choice targets as zero-based
    option indices, while a model naturally emits ``Answer: A`` through
    ``Answer: E``.  Restrict this conversion to a standalone option letter and
    a standalone 0--4 label so numerical math answers retain exact matching.
    """
    predicted_norm = _normalize(predicted).rstrip(".,;:")
    ground_truth_norm = _normalize(ground_truth).rstrip(".,;:")
    if predicted_norm == ground_truth_norm:
        return True
    option = re.fullmatch(r"([a-e])(?:\)|\])?", predicted_norm)
    if option and re.fullmatch(r"[0-4]", ground_truth_norm):
        return ord(option.group(1)) - ord("a") == int(ground_truth_norm)
    return False


def _score(sample: Any) -> dict[str, Any]:
    response = getattr(sample, "response", "").strip()
    ground_truth = str(getattr(sample, "label", ""))
    matches = re.findall(r"Answer:\s*([^\n]+)", response, flags=re.IGNORECASE)
    if not matches:
        return {"reward_value": 0.0, "reward_cat": "no_answer_marker"}
    predicted = matches[-1].strip()
    correct = _matches_label(predicted, ground_truth)
    return {"reward_value": float(correct), "reward_cat": "correct" if correct else "incorrect"}


async def math_reward_func(args: Any, samples: Any, **kwargs: Any) -> Any:
    if isinstance(samples, list):
        return [_score(sample) for sample in samples]
    return _score(samples)
