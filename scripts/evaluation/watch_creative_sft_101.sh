#!/usr/bin/env bash
# Watch the creative SFT step479->579 continuation on 101 and start full
# EIL/MIU evaluation only after iter_0000579 is complete and training exits.
set -euo pipefail

ROOT="${LOYAL_CONTINUE479_ROOT:-/tmp/experiment_g_longtask_101/checkpoints/mixed-v2-phase1-lambda050-e2m1-rollout200-phase1-seed1234-creative-sft}"
POST="${LOYAL_PHASE1_POST_ROOT:-/tmp/experiment_g_longtask_101/evaluations/phase1-lambda050-e2m1-rollout200_posttrain}"
TARGET_STEP="${LOYAL_CONTINUE479_EVAL_STEP:-579}"
PROJECT_ROOT="${LOYAL_DIRECT_EVAL_PROJECT_ROOT:-/tmp/loyal_agent_docker}"
CHECKPOINT_NAME="${LOYAL_DIRECT_EVAL_CHECKPOINT_NAME:-mixed-v2-phase1-lambda050-e2m1-rollout200-phase1-seed1234-creative-sft}"
LOG="${POST}/watch_creative_sft_${TARGET_STEP}.log"

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
  pgrep -af "/tmp/loyal_agent_docker/slime/train.py" >/dev/null
}

eval_complete() {
  [[ -f "${POST}/creative_eval/step${TARGET_STEP}/miu_final/summary.json" ]] || return 1
  [[ -f "${POST}/creative_eval/step${TARGET_STEP}/miu_final/per_sample.jsonl" ]] || return 1
  [[ -f "${POST}/creative_eval/step${TARGET_STEP}/eil_final/summary.json" ]] || return 1
  [[ -f "${POST}/creative_eval/step${TARGET_STEP}/eil_final/per_sample.jsonl" ]] || return 1
}

run_eval() {
  if eval_complete; then
    log "eval_skip_complete step=${TARGET_STEP}"
    return 0
  fi
  log "eval_start step=${TARGET_STEP}"
  LOYAL_DIRECT_EVAL_PROJECT_ROOT="${PROJECT_ROOT}" \
  LOYAL_DIRECT_EVAL_CHECKPOINT_NAME="${CHECKPOINT_NAME}" \
  LOYAL_DIRECT_EVAL_CHECKPOINT_ROOT="${ROOT}" \
  LOYAL_DIRECT_EVAL_STEPS="${TARGET_STEP}" \
  LOYAL_DIRECT_EVAL_EXPORT_ROOT="${POST}/creative_export" \
  LOYAL_DIRECT_EVAL_OUTPUT_ROOT="${POST}/creative_eval" \
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
  sleep "${LOYAL_CONTINUE479_WATCH_INTERVAL:-180}"
done
