#!/usr/bin/env bash
# Stage-by-stage manager for rollout200.  It trains one 20-rollout segment,
# immediately evaluates the resulting checkpoint, updates best_checkpoint.json,
# and cleans old non-best checkpoints before continuing.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
PYTHON="${LOYAL_PYTHON:-/root/experiment_g_runtime/conda/env/bin/python3}"
CONDITION="phase1-lambda050-e2m1-rollout200"
RUN_NAME="${LOYAL_PHASE1_RUN_NAME:-phase1}"
CHECKPOINT_NAME="${LOYAL_PHASE1_CHECKPOINT_NAME:-mixed-v2-${CONDITION}-${RUN_NAME}-seed1234}"
CHECKPOINT_ROOT="${LOYAL_PHASE1_CHECKPOINT_ROOT:-${LOYAL_CHECKPOINT_HOST_DIR:?set checkpoint dir}}"
RUN_DIR="${LOYAL_PHASE1_RUN_DIR:?set run dir}"
POST_ROOT="${LOYAL_PHASE1_POST_ROOT:?set post root}"
STEPS=(19 39 59 79 99 119 139 159 179 199)
KEEP_RECENT="${LOYAL_CHECKPOINT_KEEP_RECENT:-2}"
CHECKPOINT_WAIT_SECONDS="${LOYAL_CHECKPOINT_WAIT_SECONDS:-60}"
CHECKPOINT_WAIT_MAX_SECONDS="${LOYAL_CHECKPOINT_WAIT_MAX_SECONDS:-7200}"

mkdir -p "${RUN_DIR}" "${POST_ROOT}/checkpoint_eval" "${POST_ROOT}/exported_models"
exec >>"${POST_ROOT}/streaming_manager.log" 2>&1

log() {
  printf '%s %s\n' "$(date -Is)" "$*"
}

iter_dir() {
  printf '%s/iter_%07d' "${CHECKPOINT_ROOT}" "$1"
}

checkpoint_complete() {
  local step="$1"
  local dir
  dir="$(iter_dir "${step}")"
  [[ -s "${dir}/common.pt" && -f "${dir}/.metadata" ]]
}

active_mixed_lr() {
  local override_file="${LOYAL_MIXED_LEARNING_RATE_FILE:-}"
  local override_value=""
  if [[ -n "${override_file}" && -f "${override_file}" ]]; then
    override_value="$(tr -d '[:space:]' <"${override_file}")"
  fi
  if [[ -n "${override_value}" ]]; then
    printf '%s' "${override_value}"
  else
    printf '%s' "${LOYAL_MIXED_LEARNING_RATE:-unset}"
  fi
}

assert_resume_safe() {
  local step="$1"
  local previous=$((step - 20))
  local latest_file="${CHECKPOINT_ROOT}/latest_checkpointed_iteration.txt"
  local latest
  if [[ "${LOYAL_MIXED_NO_LOAD_OPTIM:-0}" == "1" || "${LOYAL_MIXED_NO_LOAD_RNG:-0}" == "1" ]]; then
    log "ERROR unsafe_resume_state step=${step} reason=no_load_optimizer_or_rng"
    exit 7
  fi
  if checkpoint_complete "${step}"; then
    log "resume_guard_skip_completed step=${step}"
    return 1
  fi
  if [[ "${previous}" -lt 0 ]]; then
    return 0
  fi
  if [[ ! -f "${latest_file}" ]]; then
    log "ERROR unsafe_resume_state step=${step} reason=missing_latest_checkpoint latest_file=${latest_file}"
    exit 7
  fi
  latest="$(<"${latest_file}")"
  if [[ ! "${latest}" =~ ^[0-9]+$ ]]; then
    log "ERROR unsafe_resume_state step=${step} reason=invalid_latest_checkpoint latest=${latest}"
    exit 7
  fi
  if [[ "${latest}" -lt "${previous}" ]]; then
    log "ERROR unsafe_resume_state step=${step} reason=latest_before_previous latest=${latest} expected_at_least=${previous}"
    exit 7
  fi
  if ! checkpoint_complete "${previous}"; then
    log "ERROR unsafe_resume_state step=${step} reason=previous_checkpoint_incomplete previous=${previous}"
    exit 7
  fi
  log "resume_guard_ok step=${step} latest=${latest} previous=${previous}"
  return 0
}

stop_ray() {
  "${PYTHON%/python3}/ray" stop --force >/tmp/ray_stop_${CONDITION}.log 2>&1 || ray stop --force >/tmp/ray_stop_${CONDITION}.log 2>&1 || true
}

wait_for_checkpoint() {
  local step="$1"
  local started now elapsed
  started="$(date +%s)"
  while ! checkpoint_complete "${step}"; do
    now="$(date +%s)"
    elapsed=$((now - started))
    if [[ "${CHECKPOINT_WAIT_MAX_SECONDS}" -gt 0 && "${elapsed}" -ge "${CHECKPOINT_WAIT_MAX_SECONDS}" ]]; then
      log "ERROR checkpoint_timeout step=${step} elapsed=${elapsed}"
      exit 5
    fi
    log "waiting_checkpoint step=${step} elapsed=${elapsed}"
    sleep "${CHECKPOINT_WAIT_SECONDS}"
  done
}

train_to_step() {
  local step="$1"
  local target=$((step + 1))
  local log_file="${RUN_DIR}/train_to_${step}.log"
  if ! assert_resume_safe "${step}"; then
    return 0
  fi
  log "train_start target_rollout=${target} checkpoint=${CHECKPOINT_ROOT} lr=$(active_mixed_lr)"
  LOYAL_MIXED_NUM_ROLLOUT="${target}" \
  LOYAL_EXPERIMENT_RESUME=1 \
    bash "${PROJECT_ROOT}/scripts/launch/run_training_host.sh" mixed \
    >"${log_file}" 2>&1 || {
      log "ERROR train_failed step=${step}; tail follows"
      tail -n 160 "${log_file}" || true
      exit 6
    }
  wait_for_checkpoint "${step}"
  stop_ray
  log "train_done step=${step}"
}

eval_step() {
  local step="$1"
  local final_dir="${POST_ROOT}/checkpoint_eval/step${step}"
  if [[ -f "${final_dir}/miu_final/summary.json" && -f "${final_dir}/eil_final/summary.json" ]]; then
    log "eval_skip step=${step}"
    return 0
  fi
  log "eval_start step=${step}"
  LOYAL_DIRECT_EVAL_PROJECT_ROOT="${PROJECT_ROOT}" \
  LOYAL_DIRECT_EVAL_CHECKPOINT_NAME="${CHECKPOINT_NAME}" \
  LOYAL_DIRECT_EVAL_CHECKPOINT_ROOT="${CHECKPOINT_ROOT}" \
  LOYAL_DIRECT_EVAL_STEPS="${step}" \
  LOYAL_DIRECT_EVAL_EXPORT_ROOT="${POST_ROOT}/exported_models" \
  LOYAL_DIRECT_EVAL_OUTPUT_ROOT="${POST_ROOT}/checkpoint_eval" \
  LOYAL_DIRECT_EVAL_LOG_FILE="${POST_ROOT}/direct_checkpoint_eval_step${step}.log" \
  LOYAL_ALLOW_TRAINING_PROCESS_DURING_DIRECT_EVAL=1 \
    bash "${PROJECT_ROOT}/scripts/evaluation/run_direct_checkpoint_eval.sh"
  log "eval_done step=${step}"
}

select_best_so_far() {
  local completed=()
  local step
  for step in "${STEPS[@]}"; do
    if [[ -f "${POST_ROOT}/checkpoint_eval/step${step}/miu_final/summary.json" && -f "${POST_ROOT}/checkpoint_eval/step${step}/eil_final/summary.json" ]]; then
      completed+=("${step}")
    fi
  done
  if [[ "${#completed[@]}" -eq 0 ]]; then
    return 0
  fi
  "${PYTHON}" -m scripts.evaluation.select_best_checkpoint \
    --root "${POST_ROOT}/checkpoint_eval" \
    --steps "${completed[@]}" \
    --output "${POST_ROOT}/best_checkpoint.json"
  log "best_updated steps=${completed[*]}"
}

best_step() {
  if [[ ! -f "${POST_ROOT}/best_checkpoint.json" ]]; then
    echo ""
    return 0
  fi
  "${PYTHON}" - "${POST_ROOT}/best_checkpoint.json" <<'PY'
import json
import sys
from pathlib import Path
print(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))["best"]["step"])
PY
}

cleanup_checkpoints() {
  if [[ "${LOYAL_CLEAN_EVALUATED_CHECKPOINTS:-1}" != "1" ]]; then
    return 0
  fi
  local current="$1"
  local best
  best="$(best_step)"
  local keep_floor=$((current - KEEP_RECENT * 20))
  local step dir
  for step in "${STEPS[@]}"; do
    [[ "${step}" -lt "${keep_floor}" ]] || continue
    [[ "${step}" != "${best}" ]] || continue
    [[ -f "${POST_ROOT}/checkpoint_eval/step${step}/miu_final/summary.json" ]] || continue
    [[ -f "${POST_ROOT}/checkpoint_eval/step${step}/miu_final/per_sample.jsonl" ]] || continue
    [[ -f "${POST_ROOT}/checkpoint_eval/step${step}/eil_final/summary.json" ]] || continue
    [[ -f "${POST_ROOT}/checkpoint_eval/step${step}/eil_final/per_sample.jsonl" ]] || continue
    dir="$(iter_dir "${step}")"
    if [[ -d "${dir}" ]]; then
      rm -rf "${dir}"
      log "checkpoint_removed step=${step} dir=${dir} best=${best} current=${current}"
    fi
  done
}

final_reasoning_and_creative() {
  log "final_pipeline_start"
  LOYAL_PHASE1_CHECKPOINT_NAME="${CHECKPOINT_NAME}" \
  LOYAL_PHASE1_CHECKPOINT_ROOT="${CHECKPOINT_ROOT}" \
  LOYAL_PHASE1_RUN_DIR="" \
  LOYAL_PHASE1_POST_ROOT="${POST_ROOT}" \
  LOYAL_PHASE1_EVAL_STEPS="19 39 59 79 99 119 139 159 179 199" \
  LOYAL_PHASE1_FINAL_STEP=199 \
  LOYAL_PHASE1_RUN_REASONING="${LOYAL_PHASE1_RUN_REASONING:-1}" \
  LOYAL_PHASE1_RUN_CREATIVE="${LOYAL_PHASE1_RUN_CREATIVE:-1}" \
  LOYAL_PHASE1_POST_LOG_FILE="${POST_ROOT}/posttrain_pipeline_final.log" \
    bash "${PROJECT_ROOT}/scripts/evaluation/run_phase1_posttrain_pipeline.sh"
  log "final_pipeline_done"
}

main() {
  log "streaming_manager_start checkpoint=${CHECKPOINT_ROOT} run_dir=${RUN_DIR}"
  log "runtime_patch_start path=${LOYAL_MEGATRON_ROOT:-/root/experiment_g_runtime/Megatron-LM}"
  bash "${PROJECT_ROOT}/scripts/launch/patch_megatron_strict_resume.sh"
  log "runtime_patch_done"
  if [[ ! -f "${RUN_DIR}/mixed_train.jsonl" ]]; then
    log "prepare_mixed_data_start"
    "${PYTHON}" "${PROJECT_ROOT}/scripts/data/prepare_mixed_slime.py" \
      --miu-source "${PROJECT_ROOT}/miu/data/dataset/MIU-v2/train.jsonl" \
      --eil-source "${PROJECT_ROOT}/eil/data/dataset/EIL-v2/train.jsonl" \
      --output "${RUN_DIR}/mixed_train.jsonl" \
      --seed 1234 >"${RUN_DIR}/mixed_training_data.json"
    log "prepare_mixed_data_done path=${RUN_DIR}/mixed_train.jsonl"
  fi
  export LOYAL_MIXED_TRAIN_RECORDS="${RUN_DIR}/mixed_train.jsonl"
  export LOYAL_MIU_RECORDS="${PROJECT_ROOT}/miu/data/dataset/MIU-v2/train.jsonl:${PROJECT_ROOT}/miu/data/dataset/MIU-v2/val.jsonl"
  export LOYAL_EIL_RECORDS="${PROJECT_ROOT}/eil/data/dataset/EIL-v2/train.jsonl:${PROJECT_ROOT}/eil/data/dataset/EIL-v2/val.jsonl"
  for step in "${STEPS[@]}"; do
    train_to_step "${step}"
    eval_step "${step}"
    select_best_so_far
    cleanup_checkpoints "${step}"
  done
  final_reasoning_and_creative
  log "streaming_manager_complete acceptance=${POST_ROOT}/acceptance.json"
}

main "$@"
