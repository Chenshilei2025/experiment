#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PKG_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
source "${SCRIPT_DIR}/common.sh"
PROJECT_ROOT="$(resolve_project_root)"
CEPH_ROOT="${LOYAL_CEPHFS_ROOT:-/cephfs/shared/experiment_g/cephfs_eil_miu_v1}"
MODEL_ROOT="${LOYAL_MODEL_ROOT:-/cephfs/shared/experiment_g/assets/models}"
MODEL_NAME="${LOYAL_MODEL_NAME:-Olmo-3-7B-Instruct}"
MODEL_HF_DIR="${LOYAL_MODEL_HF_CHECKPOINT:-${MODEL_ROOT}/${MODEL_NAME}}"
MODEL_TD_DIR="${LOYAL_MODEL_REF_LOAD:-${MODEL_ROOT}/${MODEL_NAME}_torch_dist}"
MIU_DATA_ROOT="${LOYAL_MIU_DATA_ROOT:-${PROJECT_ROOT}/miu/data/dataset/MIU-v2}"
EIL_DATA_ROOT="${LOYAL_EIL_DATA_ROOT:-${PROJECT_ROOT}/eil/data/dataset/EIL-v2}"
PYTHON="${LOYAL_PYTHON:-python3}"

cd "${PROJECT_ROOT}"

"${PYTHON}" -m py_compile \
  scripts/experiment_runner.py \
  scripts/training/preflight.py \
  scripts/training/rollout/mixed.py \
  scripts/training/rewards/slime.py

"${PYTHON}" -m scripts.experiment_runner \
  --config "${PKG_ROOT}/configs/e2m1_cephfs_rollout200.json" \
  --run-name "${LOYAL_RUN_NAME:-phase1}" \
  --validate-only

"${PYTHON}" - "${PKG_ROOT}/configs/e2m1_cephfs_rollout200.json" <<'PY'
import json
import sys
from pathlib import Path

config = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
env = config["environment"]
assert config["mixed_ablation"]["ratio"] == "E2M1"
assert abs(float(env["LOYAL_MIXED_EIL_BATCH_FRACTION"]) - (2 / 3)) < 1e-9
assert int(env["LOYAL_MIXED_GLOBAL_BATCH_SIZE"]) == int(env["LOYAL_MIXED_ROLLOUT_BATCH_SIZE"]) * int(env["LOYAL_MIXED_SAMPLES_PER_PROMPT"])
assert int(env["LOYAL_MIXED_TRAIN_GPU_COUNT"]) + int(env["LOYAL_MIXED_ROLLOUT_GPU_COUNT"]) == int(env["LOYAL_MIXED_RAY_NUM_GPUS"])
assert env["LOYAL_MIXED_LEARNING_RATE"] == "2e-6"
assert int(env["LOYAL_MIXED_SCHEDULE_TOTAL_ROLLOUTS"]) == 200
assert env["LOYAL_REFUSE_CEPH_ACTIVE_PATHS"] == "0"
assert config.get("base_model") == "olmo3-7b-instruct"
assert config.get("target_model") == "allenai/Olmo-3-7B-Instruct"
assert config.get("target_model_local_dir") == "/cephfs/shared/experiment_g/assets/models/Olmo-3-7B-Instruct"
print("cephfs_eil_miu_v1 config checks passed")
PY

for path in \
  "${MIU_DATA_ROOT}/train.jsonl" \
  "${MIU_DATA_ROOT}/val.jsonl" \
  "${EIL_DATA_ROOT}/train.jsonl" \
  "${EIL_DATA_ROOT}/val.jsonl" \
  "${MODEL_HF_DIR}" \
  "${MODEL_TD_DIR}" \
  "${CEPH_ROOT}"
do
  [[ -e "${path}" ]] || { echo "missing required path: ${path}" >&2; exit 4; }
done

echo "cephfs_eil_miu_v1 preflight passed"
