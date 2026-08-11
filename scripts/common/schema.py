"""Deterministic rendering for structured evidence included in judge prompts."""
from __future__ import annotations

from typing import Iterable


def render_items(items: Iterable[dict], fields: tuple[str, ...]) -> str:
    """Render selected fields without exposing unrequested record fields."""
    rows = []
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("rendered item must be a mapping")
        rows.append("; ".join(f"{field}={item.get(field)!r}" for field in fields))
    return "\n".join(f"- {row}" for row in rows)
