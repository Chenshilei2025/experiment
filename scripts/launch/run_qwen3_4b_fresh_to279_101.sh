#!/usr/bin/env bash
# Start a new Qwen3-4B E2M1 run only after the lost step179 state is confirmed.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
RUN_ID="${LOYAL_FRESH_RUN_ID:-$(date +%Y%m%d-%H%M%S)}"

export LOYAL_T1_SOURCE_STEP=0
export LOYAL_T1_SOURCE_ROOT=""
export LOYAL_T1_CHECKPOINT_NAME="mixed-v2-qwen3-4b-e2m1-lambda050-fresh-to279-${RUN_ID}"
export LOYAL_T1_SAVE_ROOT="${LOYAL_T1_SAVE_ROOT:-/cephfs/huangzimeng/experiment_g/checkpoints/${LOYAL_T1_CHECKPOINT_NAME}}"
export LOYAL_T1_POST_ROOT="${LOYAL_T1_POST_ROOT:-/tmp/experiment_g_longtask_101/evaluations/${LOYAL_T1_CHECKPOINT_NAME}}"

exec bash "${PROJECT_ROOT}/scripts/launch/run_t1_step179_to279_101.sh"
