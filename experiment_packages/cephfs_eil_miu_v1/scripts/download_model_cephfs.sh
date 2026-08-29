#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"
PROJECT_ROOT="$(resolve_project_root)"
CEPH_ROOT="${LOYAL_CEPHFS_ROOT:-/cephfs/shared/experiment_g/assets/models}"
MODEL_NAME="${LOYAL_MODEL_NAME:-Olmo-3-7B-Instruct}"
MODEL_REPO="${LOYAL_MODEL_REPO:-allenai/Olmo-3-7B-Instruct}"
PYTHON="${LOYAL_PYTHON:-python3}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

mkdir -p "${CEPH_ROOT}"
HF_DIR="${CEPH_ROOT}/${MODEL_NAME}"

if [[ ! -d "${HF_DIR}" ]]; then
  huggingface-cli download "${MODEL_REPO}" --local-dir "${HF_DIR}"
fi

TD_DIR="${CEPH_ROOT}/${MODEL_NAME}_torch_dist"
if [[ ! -e "${TD_DIR}" && "${LOYAL_PREPARE_TORCH_DIST:-1}" == "1" ]]; then
  LOYAL_MODEL_ROOT="${CEPH_ROOT}" \
  LOYAL_BASE_MODEL="${LOYAL_MODEL_KEY:-olmo3-7b-instruct}" \
  LOYAL_MODEL_HF_CHECKPOINT="${HF_DIR}" \
  LOYAL_MODEL_REF_LOAD="${TD_DIR}" \
  bash "${PROJECT_ROOT}/scripts/launch/prepare_model_checkpoint.sh" "${LOYAL_MODEL_KEY:-olmo3-7b-instruct}"
fi

cat <<EOF
model assets prepared: ${HF_DIR}
torch_dist checkpoint: ${TD_DIR}
EOF
