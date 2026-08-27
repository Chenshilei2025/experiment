#!/usr/bin/env bash
# Convert a SLIME torch_dist checkpoint into a testable HF model on the host.
set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  echo "usage: $0 [checkpoint-name] [iteration]"
  exit 0
fi
if [[ $# -gt 2 ]]; then
  echo "usage: $0 [checkpoint-name] [iteration]" >&2
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
CONDA_SH="${LOYAL_CONDA_SH:-/root/miniforge3/etc/profile.d/conda.sh}"
CONDA_ENV="${LOYAL_CONDA_ENV:-/root/experiment_g_runtime/conda/env}"
set +u
# shellcheck disable=SC1090
source "${CONDA_SH}"
conda activate "${CONDA_ENV}"
set -u
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/launch/env.sh"

: "${LOYAL_MODEL_ROOT:?set LOYAL_MODEL_ROOT in .env to the directory containing model checkpoints}"
: "${LOYAL_BASE_MODEL:=qwen3-4b}"
CHECKPOINT_NAME="${1:-${LOYAL_SHARED_CHECKPOINT_NAME:-Qwen3-4B_loyal}}"
[[ "${CHECKPOINT_NAME}" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "checkpoint name must be a simple directory name" >&2; exit 2; }
CHECKPOINT_ROOT="${LOYAL_CHECKPOINT_HOST_DIR:-${PROJECT_ROOT}/artifacts/checkpoints/${CHECKPOINT_NAME}}"
[[ -d "${CHECKPOINT_ROOT}" ]] || { echo "checkpoint root does not exist: ${CHECKPOINT_ROOT}" >&2; exit 1; }
ITERATION="${2:-$(<"${CHECKPOINT_ROOT}/latest_checkpointed_iteration.txt")}"
[[ "${ITERATION}" =~ ^[0-9]+$ ]] || { echo "checkpoint iteration must be numeric" >&2; exit 2; }
INPUT_DIR="${CHECKPOINT_ROOT}/iter_$(printf '%07d' "${ITERATION}")"
[[ -f "${INPUT_DIR}/common.pt" && -f "${INPUT_DIR}/.metadata" ]] || { echo "not a complete torch_dist checkpoint: ${INPUT_DIR}" >&2; exit 1; }

EXPORT_ROOT="${LOYAL_EXPORT_ROOT:-${PROJECT_ROOT}/artifacts/exported_models}"
OUTPUT_DIR="${EXPORT_ROOT}/${CHECKPOINT_NAME}/iter_$(printf '%07d' "${ITERATION}")"
[[ ! -e "${OUTPUT_DIR}" ]] || { echo "refusing to overwrite existing exported model: ${OUTPUT_DIR}" >&2; exit 1; }
mkdir -p "$(dirname -- "${OUTPUT_DIR}")"

case "${LOYAL_BASE_MODEL}" in
  qwen3-4b)
    export LOYAL_MODEL_HF_CHECKPOINT="${LOYAL_MODEL_HF_CHECKPOINT:-${LOYAL_MODEL_ROOT}/Qwen3-4B}"
    ;;
  glm-z1-9b)
    export LOYAL_MODEL_HF_CHECKPOINT="${LOYAL_MODEL_HF_CHECKPOINT:-${LOYAL_MODEL_ROOT}/GLM-Z1-9B-0414}"
    ;;
  llama3.1-8b-instruct)
    export LOYAL_MODEL_HF_CHECKPOINT="${LOYAL_MODEL_HF_CHECKPOINT:-${LOYAL_LLAMA3_1_8B_MODEL_ROOT:-/ssd/models}/Llama-3.1-8B-Instruct}"
    ;;
  *)
    echo "unsupported LOYAL_BASE_MODEL=${LOYAL_BASE_MODEL}" >&2
    exit 2
    ;;
esac

SLIME_ROOT="${PROJECT_ROOT}/slime"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/launch/model_profiles.sh"
export LOYAL_MEGATRON_ROOT="${LOYAL_MEGATRON_ROOT:-/root/experiment_g_runtime/Megatron-LM}"
export PYTHONPATH="${PROJECT_ROOT}:${SLIME_ROOT}:${LOYAL_MEGATRON_ROOT}:${PYTHONPATH:-}"
python3 "${SLIME_ROOT}/tools/convert_torch_dist_to_hf.py" \
  --input-dir "${INPUT_DIR}" --output-dir "${OUTPUT_DIR}" --origin-hf-dir "${LOYAL_MODEL_HF_CHECKPOINT}" \
  --vocab-size "${LOYAL_MODEL_VOCAB_SIZE}" --force

test -f "${OUTPUT_DIR}/config.json"
test -f "${OUTPUT_DIR}/model.safetensors.index.json"
printf 'Exported final HF model: %s\n' "${OUTPUT_DIR}"
