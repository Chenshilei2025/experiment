#!/usr/bin/env bash
# Run Docker-only baseline/final tests around a fresh MIU -> EIL training chain.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/env.sh"

RUN_NAME="${1:-$(date -u +%Y%m%dT%H%M%SZ)_gpu0-5}"
if [[ ! "${RUN_NAME}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "run name must be a simple directory name" >&2
  exit 2
fi

# Both mechanisms use only GPUs 0--5. MIU reserves two for actors and four
# for TP=1 rollout engines; EIL uses the same 2+4 layout after MIU completes.
export LOYAL_MIU_GPU_DEVICES=0,1,2,3,4,5
export LOYAL_MIU_TRAIN_GPU_COUNT=2
export LOYAL_MIU_ROLLOUT_GPU_COUNT=4
export LOYAL_MIU_RAY_NUM_GPUS=6
export LOYAL_EIL_TRAIN_GPU_DEVICES=0,1,2,3,4,5
export LOYAL_EIL_TRAIN_GPU_COUNT=2
export LOYAL_EIL_ROLLOUT_GPU_COUNT=4
export LOYAL_EIL_RAY_NUM_GPUS=6
export LOYAL_TEST_GPU_DEVICES=0

LOG_DIR="${PROJECT_ROOT}/artifacts/logs/${RUN_NAME}"
mkdir -p "${LOG_DIR}"

run() {
  local label="$1"
  shift
  printf '\n[%s] %s\n' "$(date -u +%FT%TZ)" "${label}" | tee -a "${LOG_DIR}/workflow.log"
  "$@" 2>&1 | tee "${LOG_DIR}/${label}.log"
}

run preflight_miu python3 "${PROJECT_ROOT}/scripts/training/preflight.py" miu
run preflight_eil python3 "${PROJECT_ROOT}/scripts/training/preflight.py" eil
run baseline_miu "${PROJECT_ROOT}/scripts/run_test_container.sh" miu baseline "${RUN_NAME}"
run baseline_eil "${PROJECT_ROOT}/scripts/run_test_container.sh" eil baseline "${RUN_NAME}"
run train_miu bash "${SCRIPT_DIR}/run_training_container.sh" miu
run train_eil bash "${SCRIPT_DIR}/run_training_container.sh" eil
run export_final bash "${PROJECT_ROOT}/scripts/export_final_checkpoint.sh" "${LOYAL_SHARED_CHECKPOINT_NAME}"
run final_miu "${PROJECT_ROOT}/scripts/run_test_container.sh" miu final "${RUN_NAME}"
run final_eil "${PROJECT_ROOT}/scripts/run_test_container.sh" eil final "${RUN_NAME}"

printf '\n[%s] workflow completed: %s\n' "$(date -u +%FT%TZ)" "${RUN_NAME}" | tee -a "${LOG_DIR}/workflow.log"
