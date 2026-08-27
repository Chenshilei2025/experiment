#!/usr/bin/env bash
# Run a baseline or exported-final MIU/EIL test on the prepared host runtime.
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
[[ "${RUN_NAME}" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "run name must be a simple directory name" >&2; exit 2; }

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
require_fixed_training_evaluators

: "${LOYAL_BASE_MODEL:=qwen3-4b}"
if [[ "${MODEL_KIND}" == "baseline" ]]; then
  [[ $# -eq 3 ]] || { echo "checkpoint iteration applies only to final models" >&2; exit 2; }
  : "${LOYAL_MODEL_ROOT:?set LOYAL_MODEL_ROOT in .env to the directory containing model checkpoints}"
  case "${LOYAL_BASE_MODEL}" in
    qwen3-4b) MODEL_PATH="${LOYAL_MODEL_HF_CHECKPOINT:-${LOYAL_MODEL_ROOT}/Qwen3-4B}" ;;
    glm-z1-9b) MODEL_PATH="${LOYAL_MODEL_HF_CHECKPOINT:-${LOYAL_MODEL_ROOT}/GLM-Z1-9B-0414}" ;;
    llama3.1-8b-instruct) MODEL_PATH="${LOYAL_MODEL_HF_CHECKPOINT:-${LOYAL_LLAMA3_1_8B_MODEL_ROOT:-/ssd/models}/Llama-3.1-8B-Instruct}" ;;
    *) echo "unsupported LOYAL_BASE_MODEL=${LOYAL_BASE_MODEL}" >&2; exit 2 ;;
  esac
else
  CHECKPOINT_NAME="${LOYAL_SHARED_CHECKPOINT_NAME:-Qwen3-4B_loyal}"
  CHECKPOINT_ROOT="${LOYAL_CHECKPOINT_HOST_DIR:-${PROJECT_ROOT}/artifacts/checkpoints/${CHECKPOINT_NAME}}"
  ITERATION="${4:-$(<"${CHECKPOINT_ROOT}/latest_checkpointed_iteration.txt")}"
  [[ "${ITERATION}" =~ ^[0-9]+$ ]] || { echo "checkpoint iteration must be numeric" >&2; exit 2; }
  MODEL_PATH="${LOYAL_EXPORT_ROOT:-${PROJECT_ROOT}/artifacts/exported_models}/${CHECKPOINT_NAME}/iter_$(printf '%07d' "${ITERATION}")"
fi
[[ -f "${MODEL_PATH}/config.json" ]] || { echo "model checkpoint does not exist: ${MODEL_PATH}" >&2; exit 1; }

OUTPUT_ROOT="${PROJECT_ROOT}/artifacts/evaluations"
OUTPUT_DIR="${OUTPUT_ROOT}/${MECHANISM}_${MODEL_KIND}_${RUN_NAME}"
[[ ! -e "${OUTPUT_DIR}" ]] || { echo "refusing to overwrite existing evaluation output: ${OUTPUT_DIR}" >&2; exit 1; }
mkdir -p "${OUTPUT_ROOT}"

export LOYAL_MODEL_HF_CHECKPOINT="${MODEL_PATH}"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/slime:${LOYAL_MEGATRON_ROOT:-/root/experiment_g_runtime/Megatron-LM}:${PYTHONPATH:-}"
EXTRA_ARGS=()
if [[ "${MECHANISM}" == "eil" ]]; then
  EXTRA_ARGS+=(--score-concurrency "${LOYAL_EIL_TEST_SCORE_CONCURRENCY:-8}")
fi
python3 -m scripts.evaluation.cli "${MECHANISM}" \
  --checkpoint "${MODEL_PATH}" --output-dir "${OUTPUT_DIR}" --device cuda:0 \
  --shard-index "${LOYAL_TEST_SHARD_INDEX:-0}" --num-shards "${LOYAL_TEST_NUM_SHARDS:-1}" \
  "${EXTRA_ARGS[@]}"

printf 'Evaluation output: %s\n' "${OUTPUT_DIR}"
