"""Pytest bootstrap for local repository imports."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SLIME_ROOT = PROJECT_ROOT / "slime"

for path in (PROJECT_ROOT, SLIME_ROOT):
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)
