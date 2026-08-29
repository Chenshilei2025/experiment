#!/usr/bin/env bash
# Convert one downloaded HF model profile to SLIME/Megatron torch_dist format.
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 {qwen3-4b|glm-z1-9b|llama3.1-8b-instruct|olmo3-7b-instruct}" >&2
  exit 2
fi

MODEL_KEY="$1"
case "${MODEL_KEY}" in qwen3-4b|glm-z1-9b|llama3.1-8b-instruct|olmo3-7b-instruct) ;; *)
  echo "unsupported model profile: ${MODEL_KEY}" >&2; exit 2 ;; esac

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/env.sh"
: "${LOYAL_MODEL_ROOT:?set LOYAL_MODEL_ROOT in .env}"
: "${LOYAL_SLIME_IMAGE:=slimerl/slime:nightly-dev-202511127a}"

case "${MODEL_KEY}" in
  qwen3-4b|glm-z1-9b|olmo3-7b-instruct) MODEL_MOUNT_ROOT="${LOYAL_MODEL_ROOT}" ;;
  llama3.1-8b-instruct) MODEL_MOUNT_ROOT="${LOYAL_LLAMA3_1_8B_MODEL_ROOT}" ;;
esac
if [[ ! -d "${MODEL_MOUNT_ROOT}" ]]; then
  echo "model root does not exist: ${MODEL_MOUNT_ROOT}" >&2
  exit 1
fi

run_host_conversion() {
  local conda_sh="${LOYAL_CONDA_SH:-/root/miniforge3/etc/profile.d/conda.sh}"
  local conda_env="${LOYAL_CONDA_ENV:-/root/experiment_g_runtime/conda/env}"
  local megatron_root="${LOYAL_MEGATRON_ROOT:-/root/experiment_g_runtime/Megatron-LM}"
  if [[ ! -f "${conda_sh}" ]]; then
    echo "conda activation script does not exist: ${conda_sh}" >&2
    return 1
  fi
  if [[ ! -d "${conda_env}" ]]; then
    echo "conda environment does not exist: ${conda_env}" >&2
    return 1
  fi
  set +u
  # shellcheck disable=SC1090
  source "${conda_sh}"
  conda activate "${conda_env}"
  set -u
  SLIME_ROOT="${PROJECT_ROOT}/slime"
  source "${SCRIPT_DIR}/model_profiles.sh"
  test -d "${LOYAL_MODEL_HF_CHECKPOINT}"
  if [[ -e "${LOYAL_MODEL_REF_LOAD}" ]]; then
    echo "refusing to overwrite existing torch_dist checkpoint: ${LOYAL_MODEL_REF_LOAD}" >&2
    return 1
  fi
  PYTHONPATH="${PROJECT_ROOT}:${SLIME_ROOT}:${megatron_root}:${PYTHONPATH:-}" \
    MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}" \
    MASTER_PORT="${MASTER_PORT:-29591}" \
    WORLD_SIZE="${WORLD_SIZE:-1}" \
    RANK="${RANK:-0}" \
    LOCAL_RANK="${LOCAL_RANK:-0}" \
    python3 "${SLIME_ROOT}/tools/convert_hf_to_torch_dist.py" \
      "${MODEL_ARGS[@]}" \
      --no-gradient-accumulation-fusion \
      --hf-checkpoint "${LOYAL_MODEL_HF_CHECKPOINT}" \
      --save "${LOYAL_MODEL_REF_LOAD}"
}

if [[ "${LOYAL_PREPARE_MODEL_MODE:-auto}" == "host" ]] || { [[ "${LOYAL_PREPARE_MODEL_MODE:-auto}" == "auto" ]] && ! command -v docker >/dev/null 2>&1; }; then
  run_host_conversion
  exit $?
fi

docker run --rm --gpus all --network host --ipc host --shm-size=16g --entrypoint bash \
  -e "LOYAL_BASE_MODEL=${MODEL_KEY}" \
  -v "${PROJECT_ROOT}:/workspace/loyal_agent:ro" \
  -v "${MODEL_MOUNT_ROOT}:/models" \
  -w /workspace/loyal_agent \
  "${LOYAL_SLIME_IMAGE}" \
  -lc 'set -euo pipefail
    SLIME_ROOT=/root/slime
    source scripts/launch/model_profiles.sh
    test -d "${LOYAL_MODEL_HF_CHECKPOINT}"
    if [[ -e "${LOYAL_MODEL_REF_LOAD}" ]]; then
      echo "refusing to overwrite existing torch_dist checkpoint: ${LOYAL_MODEL_REF_LOAD}" >&2
      exit 1
    fi
    PYTHONPATH=/root/Megatron-LM python3 tools/convert_hf_to_torch_dist.py \
      "${MODEL_ARGS[@]}" \
      --no-gradient-accumulation-fusion \
      --hf-checkpoint "${LOYAL_MODEL_HF_CHECKPOINT}" \
      --save "${LOYAL_MODEL_REF_LOAD}"'
