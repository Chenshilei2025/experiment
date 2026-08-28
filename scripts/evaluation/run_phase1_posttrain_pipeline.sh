#!/usr/bin/env bash
# Evaluate phase-1 intermediate checkpoints, select the best, then run
# reasoning and creative-generation follow-up tests.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
if [[ -d "${PROJECT_ROOT}/assets" ]]; then
  ASSET_ROOT="${LOYAL_ASSET_ROOT:-${PROJECT_ROOT}/assets}"
else
  ASSET_ROOT="${LOYAL_ASSET_ROOT:-$(cd -- "${PROJECT_ROOT}/.." && pwd)/assets}"
fi

CHECKPOINT_NAME="${LOYAL_PHASE1_CHECKPOINT_NAME:-mixed-v2-phase1-lambda050-e2m1-rollout200-phase1-seed1234}"
CHECKPOINT_HOST_ROOT="${LOYAL_CHECKPOINT_HOST_ROOT:-${PROJECT_ROOT}/artifacts/checkpoints}"
CHECKPOINT_ROOT="${LOYAL_PHASE1_CHECKPOINT_ROOT:-${CHECKPOINT_HOST_ROOT}/${CHECKPOINT_NAME}}"
RUN_DIR="${LOYAL_PHASE1_RUN_DIR:-}"
STEPS="${LOYAL_PHASE1_EVAL_STEPS:-19 39 59 79 99 119 139 159 179 199}"
FINAL_STEP="${LOYAL_PHASE1_FINAL_STEP:-199}"
POST_ROOT="${LOYAL_PHASE1_POST_ROOT:-${PROJECT_ROOT}/artifacts/evaluations/phase1_rollout200_posttrain}"
WAIT_SECONDS="${LOYAL_PHASE1_WAIT_SECONDS:-300}"
MAX_WAIT_SECONDS="${LOYAL_PHASE1_MAX_WAIT_SECONDS:-0}"
LOG_FILE="${LOYAL_PHASE1_POST_LOG_FILE:-${POST_ROOT}/posttrain_pipeline.log}"
PYTHON="${LOYAL_PYTHON:-/root/experiment_g_runtime/conda/env/bin/python3}"
REASONING_DATA_ROOT="${LOYAL_REASONING_DATA_ROOT:-${ASSET_ROOT}/datasets}"
CREATIVE_CHECKPOINT_NAME="${LOYAL_PHASE1_CREATIVE_CHECKPOINT_NAME:-${CHECKPOINT_NAME}-creative-sft}"
CREATIVE_CHECKPOINT_ROOT="${LOYAL_PHASE1_CREATIVE_CHECKPOINT_ROOT:-${CHECKPOINT_HOST_ROOT}/${CREATIVE_CHECKPOINT_NAME}}"
CREATIVE_EVAL_RUN_NAME="${LOYAL_PHASE1_CREATIVE_EVAL_RUN_NAME:-phase1-best-creative-sft}"

mkdir -p "${POST_ROOT}" "$(dirname -- "${LOG_FILE}")"
exec >>"${LOG_FILE}" 2>&1

log() {
  printf '%s %s\n' "$(date -Is)" "$*"
}

training_is_active() {
  ps -eo args= | grep -E 'slime/train.py|scripts.experiment_runner|scripts/launch/run-mixed.sh' | grep -v grep >/dev/null
}

training_failed() {
  [[ -n "${RUN_DIR}" ]] || return 1
  "${PYTHON}" - "${RUN_DIR}/manifest.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(1)
try:
    payload = json.loads(path.read_text(encoding="utf-8"))
except json.JSONDecodeError:
    raise SystemExit(1)
if payload.get("status") in {"completed"}:
    raise SystemExit(1)
raise SystemExit(0)
PY
}

final_checkpoint_ready() {
  local iter_dir="${CHECKPOINT_ROOT}/iter_$(printf '%07d' "${FINAL_STEP}")"
  [[ -f "${iter_dir}/common.pt" && -f "${iter_dir}/.metadata" ]]
}

creative_checkpoint_ready() {
  local latest="${CREATIVE_CHECKPOINT_ROOT}/latest_checkpointed_iteration.txt"
  [[ -f "${latest}" ]] || return 1
  local iteration
  iteration="$(<"${latest}")"
  [[ "${iteration}" =~ ^[0-9]+$ ]] || return 1
  local iter_dir="${CREATIVE_CHECKPOINT_ROOT}/iter_$(printf '%07d' "${iteration}")"
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
    if [[ -n "${RUN_DIR}" ]] && ! training_is_active && training_failed; then
      log "ERROR phase1_training_stopped_before_completion run_dir=${RUN_DIR} checkpoint_ready=$(final_checkpoint_ready && echo 1 || echo 0)"
      tail -n 120 "${RUN_DIR}/run.log" 2>/dev/null || true
      exit 2
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
  "${PYTHON}" -m scripts.evaluation.select_best_checkpoint \
    --root "${POST_ROOT}/checkpoint_eval" \
    --steps ${STEPS} \
    --output "${POST_ROOT}/best_checkpoint.json"
  log "select_best_done output=${POST_ROOT}/best_checkpoint.json"
}

best_checkpoint_path() {
  "${PYTHON}" - "${POST_ROOT}/best_checkpoint.json" "${POST_ROOT}/exported_models" "${CHECKPOINT_NAME}" <<'PY'
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

best_checkpoint_step() {
  "${PYTHON}" - "${POST_ROOT}/best_checkpoint.json" <<'PY'
import json
import sys
from pathlib import Path

print(int(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))["best"]["step"]))
PY
}

bootstrap_reasoning_data() {
  if [[ "${LOYAL_PHASE1_BOOTSTRAP_REASONING:-1}" != "1" ]]; then
    return 0
  fi
  local math_source="${LOYAL_MATH_DATA:-${ASSET_ROOT}/datasets/math500/test.jsonl}"
  if [[ ! -f "${math_source}" ]]; then
    log "reasoning_bootstrap_skip missing_math_source=${math_source}"
    if [[ "${LOYAL_REASONING_REQUIRE_ALL:-1}" == "1" ]]; then
      log "ERROR missing required MATH source for reasoning bootstrap"
      exit 3
    fi
    return 0
  fi
  log "reasoning_bootstrap_start output_root=${REASONING_DATA_ROOT}"
  local args=(
    -m scripts.data.bootstrap_reasoning_benchmarks
    --output-root "${REASONING_DATA_ROOT}"
    --cache-root "${PROJECT_ROOT}/artifacts/cache/reasoning_sources"
    --math-source "${math_source}"
    --seed "${LOYAL_REASONING_BOOTSTRAP_SEED:-42}"
  )
  [[ -z "${LOYAL_UGMATH_LIMIT:-}" ]] || args+=(--umath-limit "${LOYAL_UGMATH_LIMIT}")
  [[ -z "${LOYAL_GPQA_LIMIT:-}" ]] || args+=(--gpqa-limit "${LOYAL_GPQA_LIMIT}")
  "${PYTHON}" "${args[@]}"
  log "reasoning_bootstrap_done output_root=${REASONING_DATA_ROOT}"
}

maybe_run_reasoning() {
  if [[ "${LOYAL_PHASE1_RUN_REASONING:-1}" != "1" ]]; then
    log "reasoning_skipped disabled"
    return 0
  fi
  bootstrap_reasoning_data
  local math_data="${LOYAL_MATH_DATA:-${REASONING_DATA_ROOT}/math500/test.jsonl}"
  local ugmath_data="${LOYAL_UGMATH_DATA:-${REASONING_DATA_ROOT}/ugmath/test.jsonl}"
  local gpqa_data="${LOYAL_GPQA_DATA:-${REASONING_DATA_ROOT}/gpqa/diamond.jsonl}"
  local best_path
  best_path="$(best_checkpoint_path)"
  log "reasoning_start checkpoint=${best_path}"
  LOYAL_MATH_DATA="${math_data}" \
  LOYAL_UGMATH_DATA="${ugmath_data}" \
  LOYAL_GPQA_DATA="${gpqa_data}" \
  LOYAL_REASONING_REQUIRE_ALL="${LOYAL_REASONING_REQUIRE_ALL:-1}" \
  LOYAL_REASONING_OUTPUT_ROOT="${POST_ROOT}/reasoning" \
    bash "${SCRIPT_DIR}/run_reasoning_benchmarks.sh" "${best_path}"
  log "reasoning_done output=${POST_ROOT}/reasoning"
}

run_creative_sft() {
  if [[ "${LOYAL_PHASE1_RUN_CREATIVE:-1}" != "1" ]]; then
    log "creative_skipped disabled"
    return 0
  fi
  local best_step
  best_step="$(best_checkpoint_step)"
  if creative_checkpoint_ready; then
    log "creative_sft_skip checkpoint_root=${CREATIVE_CHECKPOINT_ROOT}"
    return 0
  fi
  if [[ -e "${CREATIVE_CHECKPOINT_ROOT}" ]]; then
    local archived="${CREATIVE_CHECKPOINT_ROOT}.incomplete.$(date +%s)"
    log "creative_sft_archive_incomplete from=${CREATIVE_CHECKPOINT_ROOT} to=${archived}"
    mv "${CREATIVE_CHECKPOINT_ROOT}" "${archived}"
  fi
  log "creative_sft_start source_checkpoint=${CHECKPOINT_ROOT} source_step=${best_step} save=${CREATIVE_CHECKPOINT_ROOT}"
  LOYAL_USE_WANDB="${LOYAL_CREATIVE_USE_WANDB:-0}" \
  LOYAL_BASE_MODEL="${LOYAL_BASE_MODEL:-qwen3-4b}" \
  LOYAL_SHARED_CHECKPOINT_NAME="${CREATIVE_CHECKPOINT_NAME}" \
  LOYAL_CHECKPOINT_HOST_DIR="${CREATIVE_CHECKPOINT_ROOT}" \
  LOYAL_CREATIVE_LOAD="${CHECKPOINT_ROOT}" \
  LOYAL_CREATIVE_CKPT_STEP="${best_step}" \
  LOYAL_CREATIVE_SAVE="${CREATIVE_CHECKPOINT_ROOT}" \
  LOYAL_CREATIVE_STRICT_RESUME=1 \
  LOYAL_CREATIVE_NO_LOAD_OPTIM=0 \
  LOYAL_CREATIVE_NO_LOAD_RNG=0 \
  LOYAL_CREATIVE_TRAIN_GPU_DEVICES="${LOYAL_PHASE1_CREATIVE_GPUS:-0,1}" \
  LOYAL_CREATIVE_TRAIN_GPU_COUNT="${LOYAL_PHASE1_CREATIVE_TRAIN_GPU_COUNT:-2}" \
  LOYAL_CREATIVE_RAY_NUM_GPUS="${LOYAL_PHASE1_CREATIVE_TRAIN_GPU_COUNT:-2}" \
    bash "${PROJECT_ROOT}/scripts/launch/run_training_host.sh" creative
  log "creative_sft_done checkpoint_root=${CREATIVE_CHECKPOINT_ROOT}"
}

export_creative_checkpoint() {
  if [[ "${LOYAL_PHASE1_RUN_CREATIVE:-1}" != "1" ]]; then
    return 0
  fi
  local iteration
  iteration="$(<"${CREATIVE_CHECKPOINT_ROOT}/latest_checkpointed_iteration.txt")"
  local export_dir="${POST_ROOT}/creative_export/${CREATIVE_CHECKPOINT_NAME}/iter_$(printf '%07d' "${iteration}")"
  if [[ -f "${export_dir}/config.json" && -f "${export_dir}/model.safetensors.index.json" ]]; then
    log "creative_export_skip dir=${export_dir}"
    return 0
  fi
  if [[ -e "${export_dir}" ]]; then
    local archived="${export_dir}.incomplete.$(date +%s)"
    log "creative_export_archive_incomplete from=${export_dir} to=${archived}"
    mv "${export_dir}" "${archived}"
  fi
  log "creative_export_start iteration=${iteration}"
  LOYAL_SHARED_CHECKPOINT_NAME="${CREATIVE_CHECKPOINT_NAME}" \
  LOYAL_CHECKPOINT_HOST_DIR="${CREATIVE_CHECKPOINT_ROOT}" \
  LOYAL_EXPORT_ROOT="${POST_ROOT}/creative_export" \
    bash "${PROJECT_ROOT}/scripts/export_final_checkpoint_host.sh" "${CREATIVE_CHECKPOINT_NAME}" "${iteration}"
  log "creative_export_done dir=${export_dir}"
}

run_creative_eil_miu_eval() {
  if [[ "${LOYAL_PHASE1_RUN_CREATIVE:-1}" != "1" ]]; then
    return 0
  fi
  local iteration
  iteration="$(<"${CREATIVE_CHECKPOINT_ROOT}/latest_checkpointed_iteration.txt")"
  log "creative_eval_start iteration=${iteration}"
  for mechanism in miu eil; do
    local final_dir="${POST_ROOT}/creative_eval/${mechanism}_final"
    if [[ -f "${final_dir}/summary.json" && -f "${final_dir}/per_sample.jsonl" ]]; then
      log "creative_eval_skip mechanism=${mechanism}"
      continue
    fi
    if [[ -e "${final_dir}" ]]; then
      local archived="${final_dir}.incomplete.$(date +%s)"
      log "creative_eval_archive_incomplete mechanism=${mechanism} from=${final_dir} to=${archived}"
      mv "${final_dir}" "${archived}"
    fi
    local output="${PROJECT_ROOT}/artifacts/evaluations/${mechanism}_final_${CREATIVE_EVAL_RUN_NAME}-${mechanism}"
    if [[ -e "${output}" ]]; then
      local archived_output="${output}.incomplete.$(date +%s)"
      log "creative_eval_archive_existing_output mechanism=${mechanism} from=${output} to=${archived_output}"
      mv "${output}" "${archived_output}"
    fi
    LOYAL_SHARED_CHECKPOINT_NAME="${CREATIVE_CHECKPOINT_NAME}" \
    LOYAL_CHECKPOINT_HOST_DIR="${CREATIVE_CHECKPOINT_ROOT}" \
    LOYAL_EXPORT_ROOT="${POST_ROOT}/creative_export" \
    LOYAL_TEST_SHARD_INDEX=0 \
    LOYAL_TEST_NUM_SHARDS=1 \
      bash "${PROJECT_ROOT}/scripts/run_test_host.sh" "${mechanism}" final "${CREATIVE_EVAL_RUN_NAME}-${mechanism}" "${iteration}"
    mkdir -p "$(dirname -- "${final_dir}")"
    mv "${output}" "${final_dir}"
  done
  log "creative_eval_done output=${POST_ROOT}/creative_eval"
}

write_acceptance() {
  log "acceptance_start"
  "${PYTHON}" - "${POST_ROOT}" "${STEPS}" "${CREATIVE_CHECKPOINT_ROOT}" "${LOYAL_PHASE1_RUN_CREATIVE:-1}" <<'PY'
from __future__ import annotations
import json
import sys
from pathlib import Path

post_root = Path(sys.argv[1])
steps = [int(item) for item in sys.argv[2].split()]
creative_root = Path(sys.argv[3])
creative_enabled = sys.argv[4] == "1"
errors: list[str] = []

for step in steps:
    for mechanism, expected in (("miu", 385), ("eil", 656)):
        summary_path = post_root / "checkpoint_eval" / f"step{step}" / f"{mechanism}_final" / "summary.json"
        sample_path = summary_path.with_name("per_sample.jsonl")
        if not summary_path.is_file() or not sample_path.is_file():
            errors.append(f"missing {mechanism} checkpoint eval for step {step}")
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if int(summary.get("n_total", -1)) != expected:
            errors.append(f"{mechanism} step {step} n_total={summary.get('n_total')} expected={expected}")

best_path = post_root / "best_checkpoint.json"
if not best_path.is_file():
    errors.append("missing best_checkpoint.json")
else:
    best = json.loads(best_path.read_text(encoding="utf-8")).get("best", {})
    if not best.get("eligible"):
        errors.append("best checkpoint is not eligible")
    if int(best.get("step", -1)) not in steps:
        errors.append(f"best checkpoint step is outside requested steps: {best.get('step')}")

for name in ("math", "ugmath", "gpqa"):
    summary_path = post_root / "reasoning" / name / "summary.json"
    if not summary_path.is_file():
        errors.append(f"missing reasoning summary: {name}")
        continue
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if int(summary.get("n_total", 0)) <= 0:
        errors.append(f"empty reasoning summary: {name}")

if creative_enabled:
    latest = creative_root / "latest_checkpointed_iteration.txt"
    if not latest.is_file():
        errors.append("missing creative latest checkpoint")
    for mechanism in ("miu", "eil"):
        summary_path = post_root / "creative_eval" / f"{mechanism}_final" / "summary.json"
        sample_path = summary_path.with_name("per_sample.jsonl")
        if not summary_path.is_file() or not sample_path.is_file():
            errors.append(f"missing creative {mechanism} evaluation")
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if int(summary.get("n_total", 0)) <= 0:
            errors.append(f"empty creative {mechanism} evaluation")

payload = {
    "status": "passed" if not errors else "failed",
    "post_root": str(post_root),
    "steps": steps,
    "errors": errors,
}
(post_root / "acceptance.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(payload, ensure_ascii=False), flush=True)
raise SystemExit(0 if not errors else 5)
PY
  log "acceptance_done output=${POST_ROOT}/acceptance.json"
}

main() {
  log "posttrain_pipeline_start checkpoint=${CHECKPOINT_NAME} checkpoint_root=${CHECKPOINT_ROOT}"
  wait_for_training
  run_checkpoint_eval
  select_best
  maybe_run_reasoning
  run_creative_sft
  export_creative_checkpoint
  run_creative_eil_miu_eval
  write_acceptance
  log "posttrain_pipeline_complete"
}

main "$@"
