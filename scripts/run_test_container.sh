#!/usr/bin/env bash
# Run a baseline or exported-final MIU/EIL test entirely inside the SLIME image.
set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  echo "usage: $0 {miu|eil} {baseline|final} <run-name> [checkpoint-iteration]"
  exit 0
fi
if [[ $# -lt 3 || $# -gt 4 || ( "$1" != "miu" && "$1" != "eil" ) || ( "$2" != "baseline" && "$2" != "final" ) ]]; then
  echo "usage: $0 {miu|eil} {baseline|final} <run-name> [checkpoint-iteration]" >&2
  exit 2
fi

MECHANISM="$1"
MODEL_KIND="$2"
RUN_NAME="$3"
if [[ ! "${RUN_NAME}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "run name must be a simple directory name" >&2
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/launch/env.sh"
require_fixed_training_evaluators

: "${LOYAL_BASE_MODEL:=qwen3-4b}"

if [[ "${MODEL_KIND}" == "baseline" ]]; then
  if [[ $# -eq 4 ]]; then
    echo "checkpoint iteration applies only to final models" >&2
    exit 2
  fi
  : "${LOYAL_MODEL_ROOT:?set LOYAL_MODEL_ROOT in .env to the directory containing model profiles and torch_dist checkpoints}"
  case "${LOYAL_BASE_MODEL}" in
    qwen3-4b|glm-z1-9b|olmo3-7b-instruct) MODEL_MOUNT_ROOT="${LOYAL_MODEL_ROOT}" ;;
    llama3.1-8b-instruct) MODEL_MOUNT_ROOT="${LOYAL_LLAMA3_1_8B_MODEL_ROOT}" ;;
    *) echo "unsupported LOYAL_BASE_MODEL=${LOYAL_BASE_MODEL}" >&2; exit 2 ;;
  esac
  if [[ ! -d "${MODEL_MOUNT_ROOT}" ]]; then
    echo "model root does not exist: ${MODEL_MOUNT_ROOT}" >&2
    exit 1
  fi
  # Baseline evaluation sources the same profile used by training, so the
  # selected HF checkpoint cannot drift from the experiment's base model.
  SLIME_ROOT="${PROJECT_ROOT}/slime"
  # shellcheck disable=SC1091
  source "${SCRIPT_DIR}/launch/model_profiles.sh"
  MODEL_PATH="${LOYAL_MODEL_HF_CHECKPOINT}"
  MODEL_MOUNT=( -v "${MODEL_MOUNT_ROOT}:/models:ro" )
else
  CHECKPOINT_NAME="${LOYAL_SHARED_CHECKPOINT_NAME:-Qwen3-4B_loyal}"
  CHECKPOINT_ROOT="${PROJECT_ROOT}/artifacts/checkpoints/${CHECKPOINT_NAME}"
  ITERATION="${4:-$(<"${CHECKPOINT_ROOT}/latest_checkpointed_iteration.txt")}"
  if [[ ! "${ITERATION}" =~ ^[0-9]+$ ]]; then
    echo "checkpoint iteration must be numeric" >&2
    exit 2
  fi
  HOST_MODEL_PATH="${PROJECT_ROOT}/artifacts/exported_models/${CHECKPOINT_NAME}/iter_$(printf '%07d' "${ITERATION}")"
  if [[ ! -f "${HOST_MODEL_PATH}/config.json" || ! -f "${HOST_MODEL_PATH}/model.safetensors.index.json" ]]; then
    echo "final model is not exported; run scripts/export_final_checkpoint.sh ${CHECKPOINT_NAME} ${ITERATION}" >&2
    exit 1
  fi
  MODEL_PATH="/final-model"
  MODEL_MOUNT=( -v "${HOST_MODEL_PATH}:/final-model:ro" )
fi

OUTPUT_ROOT="${PROJECT_ROOT}/artifacts/evaluations"
OUTPUT_DIR="${OUTPUT_ROOT}/${MECHANISM}_${MODEL_KIND}_${RUN_NAME}"
if [[ -e "${OUTPUT_DIR}" ]]; then
  echo "refusing to overwrite existing evaluation output: ${OUTPUT_DIR}" >&2
  exit 1
fi
mkdir -p "${OUTPUT_ROOT}"

: "${LOYAL_SLIME_IMAGE:=slimerl/slime:nightly-dev-202511127a}"
TEST_GPU_DEVICES="${LOYAL_TEST_GPU_DEVICES:-0}"
docker run --rm --gpus "device=${TEST_GPU_DEVICES}" --network host --ipc host --shm-size=16g --entrypoint bash \
  --env-file "${PROJECT_ROOT}/.env" \
  -e "LOYAL_BASE_MODEL=${LOYAL_BASE_MODEL}" \
  -e "LOYAL_MODEL_HF_CHECKPOINT=${MODEL_PATH}" \
  -v "${PROJECT_ROOT}:/workspace/loyal_agent:ro" \
  -v "${OUTPUT_ROOT}:/outputs" \
  "${MODEL_MOUNT[@]}" \
  -w /workspace/loyal_agent \
  "${LOYAL_SLIME_IMAGE}" \
  -lc "export PYTHONPATH=/workspace/loyal_agent:\${PYTHONPATH:-}; source scripts/launch/env.sh && python3 -m scripts.evaluation.cli ${MECHANISM} --checkpoint ${MODEL_PATH} --output-dir /outputs/${MECHANISM}_${MODEL_KIND}_${RUN_NAME} --device cuda:0 --shard-index \${LOYAL_TEST_SHARD_INDEX:-0} --num-shards \${LOYAL_TEST_NUM_SHARDS:-1}"

printf 'Evaluation output: %s\n' "${OUTPUT_DIR}"
