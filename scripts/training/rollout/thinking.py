"""SLIME rollout wrapper that hides Qwen thinking text after token generation."""
from __future__ import annotations

from scripts.common.thinking import has_incomplete_explicit_thinking, strip_thinking
from slime.rollout.sglang_rollout import generate as sglang_generate
from slime.utils.types import Sample


async def generate(args, sample: Sample, sampling_params: dict) -> Sample:
    """Preserve generated tokens for loss, but expose only the final answer onward.

    SLIME training consumes ``tokens`` and ``response_length`` rather than the
    response string, so replacing the string does not remove thinking tokens
    from the policy loss.  It prevents a trace from reaching reward clients,
    counterparties, evaluation output, or ordinary SLIME rollout logging.
    """
    sample = await sglang_generate(args, sample, sampling_params)
    raw = sample.response
    if has_incomplete_explicit_thinking(raw):
        # Do not allow a partial private trace to masquerade as an answer.
        sample.response = ""
        sample.status = Sample.Status.TRUNCATED
        return sample
    sample.response = strip_thinking(raw)
    return sample
