"""Keep Qwen thinking traces out of user-facing and reward-facing text."""
from __future__ import annotations

import re

_THINK_BLOCK = re.compile(r"<think>.*?</think>\s*", flags=re.DOTALL | re.IGNORECASE)
_THINK_CLOSE = re.compile(r"</think>\s*", flags=re.IGNORECASE)
_THINK_OPEN = re.compile(r"<think>", flags=re.IGNORECASE)
# SGLang can return Qwen chat-template stop markers in the generated text.
# They are transport delimiters, not part of the policy's visible answer.
_TRAILING_CHAT_STOP = re.compile(r"(?:\s*<\|(im_end|endoftext)\|>)+\s*$")


def strip_thinking(response: str) -> str:
    """Return only the policy answer visible after the final thinking block.

    Qwen chat templates can put ``<think>`` in the prompt, so SGLang may
    return only the reasoning content followed by ``</think>``.  In that case
    everything before the closing tag is private reasoning as well.
    """
    if not isinstance(response, str):
        raise TypeError("response must be a string")
    if _THINK_CLOSE.search(response):
        response = _THINK_CLOSE.split(response)[-1]
    else:
        response = _THINK_BLOCK.sub("", response)
    return _TRAILING_CHAT_STOP.sub("", response).strip()


def has_incomplete_explicit_thinking(response: str) -> bool:
    """Identify a visible but unclosed think block.

    Truncation without an opening tag is detected from ``Sample.status`` by
    the SLIME adapters, because Qwen may receive its opening tag in the prompt.
    """
    return bool(_THINK_OPEN.search(response)) and not bool(_THINK_CLOSE.search(response))
