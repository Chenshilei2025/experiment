#!/usr/bin/env bash
# Apply a tiny Megatron runtime compatibility patch so strict checkpoint
# resume keeps optimizer moments and preserves the loaded step counter when
# older checkpoints omit per-parameter "step" tensors.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MEGATRON_ROOT="${LOYAL_MEGATRON_ROOT:-/root/experiment_g_runtime/Megatron-LM}"
PYTHON="${LOYAL_PYTHON:-/root/experiment_g_runtime/conda/env/bin/python3}"
TARGET_FILE="${MEGATRON_ROOT}/megatron/core/optimizer/distrib_optimizer.py"

[[ -f "${TARGET_FILE}" ]] || { echo "missing Megatron runtime file: ${TARGET_FILE}" >&2; exit 2; }

"${PYTHON}" - "${TARGET_FILE}" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8")

old = """            dst_tensors = {\"param\": main_param, **optim_state}\n            for key in dst_tensors:\n                dst_tensors[key].copy_(tensors[key])\n"""
new = """            dst_tensors = {\"param\": main_param, **optim_state}\n            for key in dst_tensors:\n                if key not in tensors:\n                    if key == \"step\":\n                        continue\n                    raise KeyError(key)\n                dst_tensors[key].copy_(tensors[key])\n"""

if old in text:
    text = text.replace(old, new, 1)
elif new not in text:
    raise SystemExit("target optimizer resume snippet not found")

path.write_text(text, encoding="utf-8")
PY

