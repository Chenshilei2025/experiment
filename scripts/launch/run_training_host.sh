#!/usr/bin/env bash
# Run one loyal-agent SLIME task directly on a prepared host conda runtime.
set -euo pipefail

if [[ $# -ne 1 || ( "$1" != "miu" && "$1" != "eil" && "$1" != "mixed" && "$1" != "creative" ) ]]; then
  echo "usage: $0 {miu|eil|mixed|creative}" >&2
  exit 2
fi

MECHANISM="$1"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

CONDA_SH="${LOYAL_CONDA_SH:-/root/miniforge3/etc/profile.d/conda.sh}"
CONDA_ENV="${LOYAL_CONDA_ENV:-/root/experiment_g_runtime/conda/env}"
if [[ ! -f "${CONDA_SH}" ]]; then
  echo "conda activation script does not exist: ${CONDA_SH}" >&2
  exit 1
fi
if [[ ! -d "${CONDA_ENV}" ]]; then
  echo "conda environment does not exist: ${CONDA_ENV}" >&2
  exit 1
fi

# Some conda activation hooks read unset variables, which conflicts with
# nounset. Restore strict mode immediately after activation.
set +u
# shellcheck disable=SC1090
source "${CONDA_SH}"
conda activate "${CONDA_ENV}"
set -u

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/env.sh"
: "${LOYAL_MODEL_ROOT:?set LOYAL_MODEL_ROOT in .env to the directory containing model checkpoints}"

case "${LOYAL_BASE_MODEL:-qwen3-4b}" in
  qwen3-4b)
    export LOYAL_MODEL_HF_CHECKPOINT="${LOYAL_MODEL_HF_CHECKPOINT:-${LOYAL_MODEL_ROOT}/Qwen3-4B}"
    export LOYAL_MODEL_REF_LOAD="${LOYAL_MODEL_REF_LOAD:-${LOYAL_MODEL_ROOT}/Qwen3-4B_torch_dist}"
    ;;
  glm-z1-9b)
    export LOYAL_MODEL_HF_CHECKPOINT="${LOYAL_MODEL_HF_CHECKPOINT:-${LOYAL_MODEL_ROOT}/GLM-Z1-9B-0414}"
    export LOYAL_MODEL_REF_LOAD="${LOYAL_MODEL_REF_LOAD:-${LOYAL_MODEL_ROOT}/GLM-Z1-9B-0414_torch_dist}"
    ;;
  llama3.1-8b-instruct)
    MODEL_ROOT="${LOYAL_LLAMA3_1_8B_MODEL_ROOT:-/ssd/models}"
    export LOYAL_MODEL_HF_CHECKPOINT="${LOYAL_MODEL_HF_CHECKPOINT:-${MODEL_ROOT}/Llama-3.1-8B-Instruct}"
    export LOYAL_MODEL_REF_LOAD="${LOYAL_MODEL_REF_LOAD:-${MODEL_ROOT}/Llama-3.1-8B-Instruct_torch_dist}"
    ;;
  olmo3-7b-instruct)
    export LOYAL_MODEL_HF_CHECKPOINT="${LOYAL_MODEL_HF_CHECKPOINT:-${LOYAL_MODEL_ROOT}/Olmo-3-7B-Instruct}"
    export LOYAL_MODEL_REF_LOAD="${LOYAL_MODEL_REF_LOAD:-${LOYAL_MODEL_ROOT}/Olmo-3-7B-Instruct_torch_dist}"
    ;;
  *)
    echo "unsupported LOYAL_BASE_MODEL=${LOYAL_BASE_MODEL}" >&2
    exit 2
    ;;
esac

CHECKPOINT_NAME="${LOYAL_SHARED_CHECKPOINT_NAME:-${LOYAL_BASE_MODEL}_${MECHANISM}_slime}"
if [[ ! "${CHECKPOINT_NAME}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "LOYAL_SHARED_CHECKPOINT_NAME must be a simple checkpoint directory name" >&2
  exit 1
fi
CHECKPOINT_DIR="${LOYAL_CHECKPOINT_HOST_DIR:-${PROJECT_ROOT}/artifacts/checkpoints/${CHECKPOINT_NAME}}"
mkdir -p "${CHECKPOINT_DIR}"
LOAD_VAR="LOYAL_${MECHANISM^^}_LOAD"
SAVE_VAR="LOYAL_${MECHANISM^^}_SAVE"
export "${LOAD_VAR}=${!LOAD_VAR:-${CHECKPOINT_DIR}}"
export "${SAVE_VAR}=${!SAVE_VAR:-${CHECKPOINT_DIR}}"

export LOYAL_MEGATRON_ROOT="${LOYAL_MEGATRON_ROOT:-/root/experiment_g_runtime/Megatron-LM}"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/slime:${LOYAL_MEGATRON_ROOT}:${PYTHONPATH:-}"

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  case "${MECHANISM}" in
    miu)
      [[ -z "${LOYAL_MIU_GPU_DEVICES:-}" ]] || export CUDA_VISIBLE_DEVICES="${LOYAL_MIU_GPU_DEVICES}"
      ;;
    eil)
      [[ -z "${LOYAL_EIL_TRAIN_GPU_DEVICES:-}" ]] || export CUDA_VISIBLE_DEVICES="${LOYAL_EIL_TRAIN_GPU_DEVICES}"
      ;;
    mixed)
      [[ -z "${LOYAL_MIXED_TRAIN_GPU_DEVICES:-}" ]] || export CUDA_VISIBLE_DEVICES="${LOYAL_MIXED_TRAIN_GPU_DEVICES}"
      ;;
    creative)
      [[ -z "${LOYAL_CREATIVE_TRAIN_GPU_DEVICES:-}" ]] || export CUDA_VISIBLE_DEVICES="${LOYAL_CREATIVE_TRAIN_GPU_DEVICES}"
      ;;
  esac
fi

cd "${PROJECT_ROOT}"
exec bash "${SCRIPT_DIR}/run-${MECHANISM}.sh"
