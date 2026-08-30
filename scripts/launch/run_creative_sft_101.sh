#!/usr/bin/env bash
# Detachable creative SFT launcher for the 10.220.5.101 host.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
HOST_TAG="${LOYAL_HOST_TAG:-101}"
LOCAL_ROOT="${LOYAL_LOCAL_RUN_ROOT:-/tmp/experiment_g_longtask_${HOST_TAG}}"
CONDITION="${LOYAL_CREATIVE_CONDITION:-phase1-lambda050-e2m1-rollout200}"
RUN_NAME="${LOYAL_PHASE1_RUN_NAME:-phase1}"
CHECKPOINT_NAME="${LOYAL_SHARED_CHECKPOINT_NAME:-mixed-v2-${CONDITION}-${RUN_NAME}-seed1234-creative-sft}"
CHECKPOINT_ROOT="${LOYAL_CHECKPOINT_HOST_DIR:-${LOCAL_ROOT}/checkpoints/${CHECKPOINT_NAME}}"
POST_ROOT="${LOYAL_PHASE1_POST_ROOT:-${LOCAL_ROOT}/evaluations/${CONDITION}_posttrain}"
TRAIN_LOG="${POST_ROOT}/creative_sft_100.log"
TRAIN_PID="${POST_ROOT}/creative_sft_100.pid"
WATCH_LOG="${POST_ROOT}/creative_sft_100_watch.log"
WATCH_PID="${POST_ROOT}/creative_sft_100_watch.pid"
LAUNCH_JSON="${POST_ROOT}/creative_sft_launch_env.json"
SOURCE_STEP="${LOYAL_CREATIVE_SOURCE_STEP:-479}"
TARGET_STEP="${LOYAL_CREATIVE_TARGET_STEP:-579}"
TARGET_ROLLOUT="${LOYAL_CREATIVE_NUM_ROLLOUT:-580}"
TRAIN_RECORDS="${LOYAL_CREATIVE_TRAIN_RECORDS:-/cephfs/shared/experiment_g/artifacts/slime/CREATIVE/train.parquet}"
WATCH_SCRIPT="${PROJECT_ROOT}/scripts/evaluation/watch_creative_sft_101.sh"
RESUME_STEP=""

mkdir -p "${POST_ROOT}" "${CHECKPOINT_ROOT}"
exec >>"${POST_ROOT}/creative_sft_launcher.log" 2>&1

log() {
  printf '%s %s\n' "$(date -Is)" "$*"
}

checkpoint_complete() {
  local step="$1"
  local dir="${CHECKPOINT_ROOT}/iter_$(printf '%07d' "${step}")"
  [[ -s "${dir}/common.pt" && -f "${dir}/.metadata" ]]
}

pid_alive() {
  local pid_file="$1"
  [[ -f "${pid_file}" ]] || return 1
  local pid
  pid="$(tr -d '[:space:]' <"${pid_file}")"
  [[ "${pid}" =~ ^[0-9]+$ ]] || return 1
  kill -0 "${pid}" 2>/dev/null
}

write_launch_env() {
  python3 - "${LAUNCH_JSON}" "${CHECKPOINT_NAME}" "${CHECKPOINT_ROOT}" "${POST_ROOT}" "${SOURCE_STEP}" "${RESUME_STEP}" "${TARGET_STEP}" "${TARGET_ROLLOUT}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = {
    "checkpoint_name": sys.argv[2],
    "checkpoint_root": sys.argv[3],
    "post_root": sys.argv[4],
    "source_step": int(sys.argv[5]),
    "resume_step": int(sys.argv[6]),
    "target_step": int(sys.argv[7]),
    "target_rollout": int(sys.argv[8]),
    "strict_resume": True,
    "loads_optimizer": True,
    "loads_rng": True,
    "uses_checkpoint_opt_param_scheduler": True,
}
path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY
}

ensure_source_checkpoint() {
  local latest_file="${CHECKPOINT_ROOT}/latest_checkpointed_iteration.txt"
  if [[ ! -s "${latest_file}" ]]; then
    echo "missing source checkpoint tracker: ${latest_file}" >&2
    exit 7
  fi
  local latest
  latest="$(tr -d '[:space:]' <"${latest_file}")"
  if [[ ! "${latest}" =~ ^[0-9]+$ || "${latest}" -lt "${SOURCE_STEP}" ]]; then
    echo "source checkpoint latest is before ${SOURCE_STEP}: ${latest}" >&2
    exit 7
  fi
  if [[ "${latest}" -gt "${TARGET_STEP}" ]]; then
    echo "source checkpoint latest is beyond target ${TARGET_STEP}: ${latest}" >&2
    exit 7
  fi
  RESUME_STEP="${latest}"
  if ! checkpoint_complete "${RESUME_STEP}"; then
    echo "resume checkpoint iter_$(printf '%07d' "${RESUME_STEP}") is incomplete under ${CHECKPOINT_ROOT}" >&2
    exit 7
  fi
  if [[ ! -f "${TRAIN_RECORDS}" ]]; then
    echo "missing creative SFT records: ${TRAIN_RECORDS}" >&2
    exit 7
  fi
}

start_training() {
  if pid_alive "${TRAIN_PID}"; then
    log "training_already_running pid=$(cat "${TRAIN_PID}")"
    return 0
  fi
  if checkpoint_complete "${TARGET_STEP}"; then
    log "training_already_complete step=${TARGET_STEP}"
    return 0
  fi
  log "training_start source_step=${SOURCE_STEP} resume_step=${RESUME_STEP} target_rollout=${TARGET_ROLLOUT} checkpoint_root=${CHECKPOINT_ROOT}"
  nohup env \
    LOYAL_BASE_MODEL="${LOYAL_BASE_MODEL:-qwen3-4b}" \
    LOYAL_MODEL_ROOT="${LOYAL_MODEL_ROOT:-/cephfs/shared/experiment_g/assets/models}" \
    LOYAL_ASSET_ROOT="${LOYAL_ASSET_ROOT:-/cephfs/shared/experiment_g/assets}" \
    LOYAL_PYTHON="${LOYAL_PYTHON:-/root/experiment_g_runtime/conda/env/bin/python3}" \
    LOYAL_CONDA_SH="${LOYAL_CONDA_SH:-/root/miniforge3/etc/profile.d/conda.sh}" \
    LOYAL_CONDA_ENV="${LOYAL_CONDA_ENV:-/root/experiment_g_runtime/conda/env}" \
    LOYAL_MEGATRON_ROOT="${LOYAL_MEGATRON_ROOT:-/root/experiment_g_runtime/Megatron-LM}" \
    LOYAL_SHARED_CHECKPOINT_NAME="${CHECKPOINT_NAME}" \
    LOYAL_CHECKPOINT_HOST_DIR="${CHECKPOINT_ROOT}" \
    LOYAL_CREATIVE_LOAD="${CHECKPOINT_ROOT}" \
    LOYAL_CREATIVE_SAVE="${CHECKPOINT_ROOT}" \
    LOYAL_CREATIVE_CKPT_STEP="${RESUME_STEP}" \
    LOYAL_CREATIVE_NUM_ROLLOUT="${TARGET_ROLLOUT}" \
    LOYAL_CREATIVE_TRAIN_RECORDS="${TRAIN_RECORDS}" \
    LOYAL_CREATIVE_TRAIN_GPU_COUNT="${LOYAL_CREATIVE_TRAIN_GPU_COUNT:-2}" \
    LOYAL_CREATIVE_ROLLOUT_GPU_COUNT=0 \
    LOYAL_CREATIVE_RAY_NUM_GPUS="${LOYAL_CREATIVE_RAY_NUM_GPUS:-2}" \
    LOYAL_CREATIVE_TRAIN_GPU_DEVICES="${LOYAL_CREATIVE_TRAIN_GPU_DEVICES:-0,1}" \
    LOYAL_CREATIVE_ROLLOUT_BATCH_SIZE="${LOYAL_CREATIVE_ROLLOUT_BATCH_SIZE:-8}" \
    LOYAL_CREATIVE_GLOBAL_BATCH_SIZE="${LOYAL_CREATIVE_GLOBAL_BATCH_SIZE:-8}" \
    LOYAL_CREATIVE_SAVE_INTERVAL="${LOYAL_CREATIVE_SAVE_INTERVAL:-20}" \
    LOYAL_CREATIVE_SAVE_RETAIN_INTERVAL="${LOYAL_CREATIVE_SAVE_RETAIN_INTERVAL:-1000000}" \
    LOYAL_CREATIVE_LEARNING_RATE="${LOYAL_CREATIVE_LEARNING_RATE:-1e-5}" \
    LOYAL_CREATIVE_MIN_LR="${LOYAL_CREATIVE_MIN_LR:-1e-6}" \
    LOYAL_CREATIVE_LR_WARMUP_FRACTION="${LOYAL_CREATIVE_LR_WARMUP_FRACTION:-0.1}" \
    LOYAL_CREATIVE_MAX_TOKENS_PER_GPU="${LOYAL_CREATIVE_MAX_TOKENS_PER_GPU:-8192}" \
    LOYAL_CREATIVE_STRICT_RESUME=1 \
    LOYAL_CREATIVE_NO_LOAD_OPTIM=0 \
    LOYAL_CREATIVE_NO_LOAD_RNG=0 \
    LOYAL_CREATIVE_USE_CHECKPOINT_OPT_PARAM_SCHEDULER=1 \
    LOYAL_USE_WANDB="${LOYAL_USE_WANDB:-0}" \
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}" \
    bash "${PROJECT_ROOT}/scripts/launch/run_training_host.sh" creative \
    >"${TRAIN_LOG}" 2>&1 < /dev/null &
  echo "$!" >"${TRAIN_PID}"
  log "training_pid=$(cat "${TRAIN_PID}") log=${TRAIN_LOG}"
}

start_watcher() {
  if pid_alive "${WATCH_PID}"; then
    log "watcher_already_running pid=$(cat "${WATCH_PID}")"
    return 0
  fi
  log "watcher_start target_step=${TARGET_STEP}"
  nohup env \
    LOYAL_CONTINUE479_ROOT="${CHECKPOINT_ROOT}" \
    LOYAL_CONTINUE479_EVAL_STEP="${TARGET_STEP}" \
    LOYAL_CONTINUE479_WATCH_INTERVAL="${LOYAL_CONTINUE479_WATCH_INTERVAL:-180}" \
    LOYAL_PHASE1_POST_ROOT="${POST_ROOT}" \
    LOYAL_DIRECT_EVAL_PROJECT_ROOT="${LOYAL_DIRECT_EVAL_PROJECT_ROOT:-${PROJECT_ROOT}}" \
    LOYAL_DIRECT_EVAL_CHECKPOINT_NAME="${CHECKPOINT_NAME}" \
    LOYAL_DIRECT_EVAL_CHECKPOINT_ROOT="${CHECKPOINT_ROOT}" \
    bash "${WATCH_SCRIPT}" \
    >"${WATCH_LOG}" 2>&1 < /dev/null &
  echo "$!" >"${WATCH_PID}"
  log "watcher_pid=$(cat "${WATCH_PID}") log=${WATCH_LOG}"
}

main() {
  ensure_source_checkpoint
  write_launch_env
  if checkpoint_complete "${TARGET_STEP}" && \
    [[ -f "${POST_ROOT}/creative_eval/step${TARGET_STEP}/miu_final/summary.json" && -f "${POST_ROOT}/creative_eval/step${TARGET_STEP}/eil_final/summary.json" ]]; then
    log "creative_task_already_complete target_step=${TARGET_STEP}"
    exit 0
  fi
  start_training
  start_watcher
  log "creative_launch_ready checkpoint_root=${CHECKPOINT_ROOT} source_step=${SOURCE_STEP} resume_step=${RESUME_STEP} target_step=${TARGET_STEP}"
}

main "$@"
