#!/usr/bin/env bash
set -euo pipefail

ROOT=/tmp/experiment_g_longtask_101/checkpoints/mixed-v2-phase1-lambda050-e2m1-rollout200-phase1-seed1234-continue479-plus100
POST=/tmp/experiment_g_longtask_101/evaluations/phase1-lambda050-e2m1-rollout200_posttrain
TARGET_STEP=579
PROJECT_ROOT=/tmp/loyal_agent_docker
CHECKPOINT_NAME=mixed-v2-phase1-lambda050-e2m1-rollout200-phase1-seed1234-continue479-plus100
LOG=${POST}/watch_checkpoint579_eval.log

mkdir -p "${POST}"
exec >>"${LOG}" 2>&1

log() {
  printf '%s %s\n' "$(date -Is)" "$*"
}

checkpoint_complete() {
  local step="$1"
  local dir="${ROOT}/iter_$(printf '%07d' "${step}")"
  [[ -s "${dir}/common.pt" && -f "${dir}/.metadata" ]]
}

training_alive() {
  pgrep -af "/tmp/loyal_agent_docker/slime/train.py|/tmp/loyal_agent_docker/scripts/launch/run-mixed.sh" >/dev/null
}

run_eval() {
  local latest_step
  latest_step="$(<"${ROOT}/latest_checkpointed_iteration.txt")"
  if [[ ! "${latest_step}" =~ ^[0-9]+$ || "${latest_step}" -lt "${TARGET_STEP}" ]]; then
    log "refuse_eval_latest_mismatch latest=${latest_step} target=${TARGET_STEP}"
    exit 3
  fi

  log "eval_start step=${TARGET_STEP}"
  LOYAL_DIRECT_EVAL_PROJECT_ROOT="${PROJECT_ROOT}" \
  LOYAL_DIRECT_EVAL_CHECKPOINT_NAME="${CHECKPOINT_NAME}" \
  LOYAL_DIRECT_EVAL_CHECKPOINT_ROOT="${ROOT}" \
  LOYAL_DIRECT_EVAL_STEPS="${TARGET_STEP}" \
  LOYAL_DIRECT_EVAL_EXPORT_ROOT="${POST}/exported_models" \
  LOYAL_DIRECT_EVAL_OUTPUT_ROOT="${POST}/checkpoint_eval" \
  LOYAL_DIRECT_EVAL_LOG_FILE="${POST}/direct_checkpoint_eval_step${TARGET_STEP}.log" \
    bash "${PROJECT_ROOT}/scripts/evaluation/run_direct_checkpoint_eval.sh"
  log "eval_done step=${TARGET_STEP}"
}

log "watch_start step=${TARGET_STEP} root=${ROOT}"
while true; do
  latest=""
  if [[ -f "${ROOT}/latest_checkpointed_iteration.txt" ]]; then
    latest="$(tr -d '[:space:]' <"${ROOT}/latest_checkpointed_iteration.txt")"
  fi
  log "poll latest=${latest:-none}"
  if [[ "${latest}" =~ ^[0-9]+$ && "${latest}" -ge "${TARGET_STEP}" ]] && checkpoint_complete "${TARGET_STEP}"; then
    if training_alive; then
      log "waiting_training_exit"
    else
      run_eval
      log "watch_complete"
      exit 0
    fi
  fi
  sleep 180
done
