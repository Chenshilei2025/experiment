#!/usr/bin/env bash
# Wait for phase-1 rollout160 training, then evaluate intermediate checkpoints.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

CHECKPOINT_NAME="${LOYAL_PHASE1_CHECKPOINT_NAME:-mixed-v2-phase1-lambda050-e1m1-rollout160-phase1-seed1234}"
CHECKPOINT_ROOT="${LOYAL_PHASE1_CHECKPOINT_ROOT:-${PROJECT_ROOT}/artifacts/checkpoints/${CHECKPOINT_NAME}}"
RUN_DIR="${LOYAL_PHASE1_RUN_DIR:-}"
STEPS="${LOYAL_PHASE1_EVAL_STEPS:-19 39 59 79 99 119 139 159}"
FINAL_STEP="${LOYAL_PHASE1_FINAL_STEP:-159}"
POST_ROOT="${LOYAL_PHASE1_POST_ROOT:-${PROJECT_ROOT}/artifacts/evaluations/phase1_rollout160_posttrain}"
WAIT_SECONDS="${LOYAL_PHASE1_WAIT_SECONDS:-300}"
MAX_WAIT_SECONDS="${LOYAL_PHASE1_MAX_WAIT_SECONDS:-0}"
LOG_FILE="${LOYAL_PHASE1_POST_LOG_FILE:-${POST_ROOT}/posttrain_pipeline.log}"

mkdir -p "${POST_ROOT}" "$(dirname -- "${LOG_FILE}")"
exec >>"${LOG_FILE}" 2>&1

log() {
  printf '%s %s\n' "$(date -Is)" "$*"
}

training_is_active() {
  ps -eo args= | grep -E 'slime/train.py|scripts.experiment_runner|scripts/launch/run-mixed.sh' | grep -v grep >/dev/null
}

final_checkpoint_ready() {
  local iter_dir="${CHECKPOINT_ROOT}/iter_$(printf '%07d' "${FINAL_STEP}")"
  [[ -f "${iter_dir}/common.pt" && -f "${iter_dir}/.metadata" ]]
}

manifest_completed() {
  [[ -n "${RUN_DIR}" ]] || return 0
  python3 - "${RUN_DIR}/manifest.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(1)
payload = json.loads(path.read_text(encoding="utf-8"))
raise SystemExit(0 if payload.get("status") == "completed" else 1)
PY
}

wait_for_training() {
  local started now elapsed
  started="$(date +%s)"
  while true; do
    if final_checkpoint_ready && manifest_completed && ! training_is_active; then
      log "phase1_ready checkpoint=${CHECKPOINT_NAME} final_step=${FINAL_STEP}"
      return 0
    fi
    now="$(date +%s)"
    elapsed=$((now - started))
    if [[ "${MAX_WAIT_SECONDS}" -gt 0 && "${elapsed}" -ge "${MAX_WAIT_SECONDS}" ]]; then
      log "ERROR timeout_waiting_for_phase1 elapsed=${elapsed}"
      exit 2
    fi
    log "waiting_for_phase1 elapsed=${elapsed} checkpoint_ready=$(final_checkpoint_ready && echo 1 || echo 0) training_active=$(training_is_active && echo 1 || echo 0)"
    sleep "${WAIT_SECONDS}"
  done
}

run_checkpoint_eval() {
  log "checkpoint_eval_start steps=${STEPS}"
  LOYAL_DIRECT_EVAL_PROJECT_ROOT="${PROJECT_ROOT}" \
  LOYAL_DIRECT_EVAL_CHECKPOINT_NAME="${CHECKPOINT_NAME}" \
  LOYAL_DIRECT_EVAL_CHECKPOINT_ROOT="${CHECKPOINT_ROOT}" \
  LOYAL_DIRECT_EVAL_STEPS="${STEPS}" \
  LOYAL_DIRECT_EVAL_EXPORT_ROOT="${POST_ROOT}/exported_models" \
  LOYAL_DIRECT_EVAL_OUTPUT_ROOT="${POST_ROOT}/checkpoint_eval" \
  LOYAL_DIRECT_EVAL_LOG_FILE="${POST_ROOT}/direct_checkpoint_eval.log" \
    bash "${SCRIPT_DIR}/run_direct_checkpoint_eval.sh"
  log "checkpoint_eval_done output=${POST_ROOT}/checkpoint_eval"
}

select_best() {
  log "select_best_start"
  python3 -m scripts.evaluation.select_best_checkpoint \
    --root "${POST_ROOT}/checkpoint_eval" \
    --steps ${STEPS} \
    --output "${POST_ROOT}/best_checkpoint.json"
  log "select_best_done output=${POST_ROOT}/best_checkpoint.json"
}

best_checkpoint_path() {
  python3 - "${POST_ROOT}/best_checkpoint.json" "${POST_ROOT}/exported_models" "${CHECKPOINT_NAME}" <<'PY'
import json
import sys
from pathlib import Path

best = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))["best"]
step = int(best["step"])
root = Path(sys.argv[2])
checkpoint_name = sys.argv[3]
print(root / checkpoint_name / f"iter_{step:07d}")
PY
}

maybe_run_reasoning() {
  if [[ "${LOYAL_PHASE1_RUN_REASONING:-1}" != "1" ]]; then
    log "reasoning_skipped disabled"
    return 0
  fi
  if [[ -z "${LOYAL_MATH_DATA:-}" || -z "${LOYAL_UGMATH_DATA:-}" || -z "${LOYAL_GPQA_DATA:-}" ]]; then
    log "reasoning_skipped missing one of LOYAL_MATH_DATA, LOYAL_UGMATH_DATA, LOYAL_GPQA_DATA"
    return 0
  fi
  local best_path
  best_path="$(best_checkpoint_path)"
  log "reasoning_start checkpoint=${best_path}"
  LOYAL_REASONING_OUTPUT_ROOT="${POST_ROOT}/reasoning" \
    bash "${SCRIPT_DIR}/run_reasoning_benchmarks.sh" "${best_path}"
  log "reasoning_done output=${POST_ROOT}/reasoning"
}

main() {
  log "posttrain_pipeline_start checkpoint=${CHECKPOINT_NAME} checkpoint_root=${CHECKPOINT_ROOT}"
  wait_for_training
  run_checkpoint_eval
  select_best
  maybe_run_reasoning
  log "posttrain_pipeline_complete"
}

main "$@"
