#!/usr/bin/env bash
# T1 supervisor: wait for a genuinely idle 101 host, then run the selected
# continuation protocol and launch evaluations at every checkpoint boundary.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
HOST_TAG="${LOYAL_HOST_TAG:-101}"
ROOT="${LOYAL_T1_ROOT:-/tmp/experiment_g_longtask_${HOST_TAG}}"
SOURCE_ROOT="${LOYAL_T1_SOURCE_ROOT:-${ROOT}/checkpoints/mixed-v2-phase1-lambda050-e2m1-rollout200-phase1-seed1234}"
SOURCE_STEP="${LOYAL_T1_SOURCE_STEP:-179}"
TARGET_STEP=279
CHECKPOINT_NAME="${LOYAL_T1_CHECKPOINT_NAME:-mixed-v2-phase1-lambda050-e2m1-rollout200-t1-step179-to279-seed1234}"
SAVE_ROOT="${LOYAL_T1_SAVE_ROOT:-/cephfs/huangzimeng/experiment_g/checkpoints/${CHECKPOINT_NAME}}"
POST_ROOT="${LOYAL_T1_POST_ROOT:-${ROOT}/evaluations/t1_step179_to279}"
TRAIN_LOG="${POST_ROOT}/training.log"
SUPERVISOR_LOG="${POST_ROOT}/supervisor.log"
PYTHON="${LOYAL_PYTHON:-/root/experiment_g_runtime/conda/env/bin/python3}"
POLL="${LOYAL_T1_POLL_SECONDS:-60}"
EVAL_GPU="${LOYAL_T1_EVAL_GPU:-0}"
if [[ "${SOURCE_STEP}" == "0" ]]; then
  EVAL_STEPS=(19 39 59 79 99 119 139 159 179 199 219 239 259 279)
else
  EVAL_STEPS=(199 219 239 259 279)
fi

# Resolve all data/checkpoint locations before starting the detached process.
# The host launcher inherits these values; relying on its defaults caused the
# remote 101 launch to look for MIU/EIL records in a different checkout.
MIU_RECORDS="${LOYAL_T1_MIU_RECORDS:-${PROJECT_ROOT}/miu/data/dataset/MIU-v2/train.jsonl:${PROJECT_ROOT}/miu/data/dataset/MIU-v2/val.jsonl}"
EIL_RECORDS="${LOYAL_T1_EIL_RECORDS:-${PROJECT_ROOT}/eil/data/dataset/EIL-v2/train.jsonl:${PROJECT_ROOT}/eil/data/dataset/EIL-v2/val.jsonl}"
MODEL_ROOT="${LOYAL_MODEL_ROOT:-/cephfs/shared/experiment_g/assets/models}"
ASSET_ROOT="${LOYAL_ASSET_ROOT:-/cephfs/shared/experiment_g/assets}"
CONDA_SH="${LOYAL_CONDA_SH:-/root/miniforge3/etc/profile.d/conda.sh}"
CONDA_ENV="${LOYAL_CONDA_ENV:-/root/experiment_g_runtime/conda/env}"
MEGATRON_ROOT="${LOYAL_MEGATRON_ROOT:-/root/experiment_g_runtime/Megatron-LM}"
RAY_TEMP_DIR="${LOYAL_T1_RAY_TEMP_DIR:-/tmp/experiment_g_t1_ray_${HOST_TAG}}"

mkdir -p "${POST_ROOT}" "${SAVE_ROOT}"
exec >>"${SUPERVISOR_LOG}" 2>&1
log() { printf '%s %s\n' "$(date -Is)" "$*"; }
iter_dir() { printf '%s/iter_%07d' "$1" "$2"; }
complete() { local d; d="$(iter_dir "$1" "$2")"; [[ -s "$d/common.pt" && -s "$d/.metadata" ]]; }
latest() { [[ -f "$1/latest_checkpointed_iteration.txt" ]] && tr -d '[:space:]' <"$1/latest_checkpointed_iteration.txt"; }

wait_idle() {
  while true; do
    local active
    active="$(ps -eo args= | grep -E 'slime/train.py|ray::SGLangEngine|ray::MegatronTrainRayActor' | grep -v grep || true)"
    if [[ -z "${active}" ]]; then log "host_idle"; return 0; fi
    log "waiting_host_idle"; sleep "${POLL}"
  done
}

assert_source() {
  if [[ "${SOURCE_STEP}" == "0" ]]; then
    [[ ! -e "${SAVE_ROOT}/iter_0000279" ]] || { log "ERROR target_already_exists path=${SAVE_ROOT}"; exit 7; }
    return 0
  fi
  [[ -f "${SOURCE_ROOT}/latest_checkpointed_iteration.txt" ]] || { log "ERROR missing_source_tracker"; exit 7; }
  [[ "$(latest "${SOURCE_ROOT}")" == "199" || "$(latest "${SOURCE_ROOT}")" == "179" ]] || { log "ERROR source_tracker_must_be_179_or_199 actual=$(latest "${SOURCE_ROOT}")"; exit 7; }
  complete "${SOURCE_ROOT}" 179 || { log "ERROR incomplete_step179 path=$(iter_dir "${SOURCE_ROOT}" 179)"; exit 7; }
  local shard_count
  shard_count="$(find "$(iter_dir "${SOURCE_ROOT}" 179)" -maxdepth 1 -type f -name '*.distcp' | wc -l)"
  [[ "${shard_count}" -ge 4 ]] || { log "ERROR source_step179_missing_rank_shards count=${shard_count}"; exit 7; }
  [[ ! -e "${SAVE_ROOT}/iter_0000279" ]] || { log "ERROR target_already_exists path=${SAVE_ROOT}"; exit 7; }
}

assert_strict_resume_log() {
  local log_file="$1" from_step="$2"
  [[ -s "${log_file}" ]] || { log "ERROR missing_training_log path=${log_file}"; return 1; }
  if rg -qi -- '--no-load-optim|--no-load-rng|--finetune' "${log_file}"; then
    log "ERROR strict_resume_forbidden_flag path=${log_file}"
    return 1
  fi
  if ! rg -qi "(load.*${from_step}|iteration.*${from_step}|checkpoint.*${from_step})" "${log_file}"; then
    log "ERROR strict_resume_source_not_observed source_step=${from_step} log=${log_file}"
    return 1
  fi
}

assert_runtime_paths() {
  [[ -x "${PYTHON}" ]] || { log "ERROR python_not_executable path=${PYTHON}"; exit 7; }
  [[ -f "${CONDA_SH}" ]] || { log "ERROR conda_script_missing path=${CONDA_SH}"; exit 7; }
  [[ -d "${CONDA_ENV}" ]] || { log "ERROR conda_env_missing path=${CONDA_ENV}"; exit 7; }
  [[ -d "${MODEL_ROOT}" ]] || { log "ERROR model_root_missing path=${MODEL_ROOT}"; exit 7; }
  [[ -d "${MEGATRON_ROOT}" ]] || { log "ERROR megatron_root_missing path=${MEGATRON_ROOT}"; exit 7; }
  local record path
  for record in "${MIU_RECORDS//:/ }" "${EIL_RECORDS//:/ }"; do
    for path in ${record}; do
      [[ -s "${path}" ]] || { log "ERROR reward_record_missing path=${path}"; exit 7; }
    done
  done
  [[ "${LOYAL_MIXED_RAY_NUM_GPUS:-4}" -eq "$(( ${LOYAL_T1_TRAIN_GPU_COUNT:-2} + ${LOYAL_T1_ROLLOUT_GPU_COUNT:-2} ))" ]] || {
    log "ERROR invalid_gpu_topology ray=${LOYAL_MIXED_RAY_NUM_GPUS:-4} train=${LOYAL_T1_TRAIN_GPU_COUNT:-2} rollout=${LOYAL_T1_ROLLOUT_GPU_COUNT:-2}"
    exit 7
  }
}

start_training() {
  local load_root="$1" current_step="$2" target_step="$3"
  log "training_start load=${load_root} current_step=${current_step} target=${target_step} save=${SAVE_ROOT}"
  export LOYAL_BASE_MODEL=qwen3-4b
  export LOYAL_MODEL_ROOT="${MODEL_ROOT}"
  export LOYAL_ASSET_ROOT="${ASSET_ROOT}"
  export LOYAL_PYTHON="${PYTHON}"
  export LOYAL_CONDA_SH="${CONDA_SH}"
  export LOYAL_CONDA_ENV="${CONDA_ENV}"
  export LOYAL_MEGATRON_ROOT="${MEGATRON_ROOT}"
  export LOYAL_SHARED_CHECKPOINT_NAME="${CHECKPOINT_NAME}"
  export LOYAL_CHECKPOINT_HOST_DIR="${SAVE_ROOT}"
  export LOYAL_MIXED_LOAD="${load_root}"
  export LOYAL_MIXED_SAVE="${SAVE_ROOT}"
  if [[ "${current_step}" == "0" ]]; then
    unset LOYAL_MIXED_CKPT_STEP
  else
    export LOYAL_MIXED_CKPT_STEP="${current_step}"
  fi
  # SLIME's rollout budget is one-based while checkpoint iterations are zero-based.
  export LOYAL_MIXED_NUM_ROLLOUT="$((target_step + 1))"
  export LOYAL_MIXED_SAVE_INTERVAL=20
  # Preserve the source checkpoint's 2-train + 2-rollout topology by default.
  export LOYAL_MIXED_TRAIN_GPU_COUNT="${LOYAL_T1_TRAIN_GPU_COUNT:-2}"
  export LOYAL_MIXED_ROLLOUT_GPU_COUNT="${LOYAL_T1_ROLLOUT_GPU_COUNT:-2}"
  export LOYAL_MIXED_RAY_NUM_GPUS="${LOYAL_T1_RAY_NUM_GPUS:-$((LOYAL_MIXED_TRAIN_GPU_COUNT + LOYAL_MIXED_ROLLOUT_GPU_COUNT))}"
  export CUDA_VISIBLE_DEVICES="${LOYAL_T1_TRAIN_ROLLOUT_GPUS:-0,1,2,3}"
  export LOYAL_MIXED_GLOBAL_BATCH_SIZE=512
  export LOYAL_MIXED_ROLLOUT_BATCH_SIZE=64
  export LOYAL_MIXED_EIL_BATCH_FRACTION=0.6666666666666666
  export LOYAL_MIXED_NO_LOAD_OPTIM=0
  export LOYAL_MIXED_NO_LOAD_RNG=0
  export LOYAL_USE_CHECKPOINT_OPT_PARAM_SCHEDULER=1
  export LOYAL_MIXED_ENABLE_EVAL=0
  export LOYAL_MIU_RECORDS="${MIU_RECORDS}"
  export LOYAL_EIL_RECORDS="${EIL_RECORDS}"
  export LOYAL_MIXED_TRAIN_RECORDS="${LOYAL_T1_MIXED_TRAIN_RECORDS:-${POST_ROOT}/mixed_train.jsonl}"
  export LOYAL_T1_MIU_RECORDS="${MIU_RECORDS}"
  export LOYAL_T1_EIL_RECORDS="${EIL_RECORDS}"
  if [[ ! -s "${LOYAL_MIXED_TRAIN_RECORDS}" ]]; then
    log "prepare_mixed_data_start path=${LOYAL_MIXED_TRAIN_RECORDS}"
    "${PYTHON}" "${PROJECT_ROOT}/scripts/data/prepare_mixed_slime.py" \
      --miu-source "${MIU_RECORDS%%:*}" \
      --eil-source "${EIL_RECORDS%%:*}" \
      --output "${LOYAL_MIXED_TRAIN_RECORDS}" --seed 1234 \
      >"${POST_ROOT}/mixed_training_data.json"
    log "prepare_mixed_data_done"
  fi
  export LOYAL_RAY_TEMP_DIR="${RAY_TEMP_DIR}"
  export LOYAL_RAY_DIRECT_DRIVER="${LOYAL_T1_RAY_DIRECT_DRIVER:-1}"
  export LOYAL_RAY_STOP_BEFORE_START="${LOYAL_T1_RAY_STOP_BEFORE_START:-1}"
  export LOYAL_RAY_STOP_AFTER_EXIT="${LOYAL_T1_RAY_STOP_AFTER_EXIT:-0}"
  nohup env \
    LOYAL_MODEL_ROOT="${MODEL_ROOT}" \
    LOYAL_ASSET_ROOT="${ASSET_ROOT}" \
    LOYAL_CONDA_SH="${CONDA_SH}" \
    LOYAL_CONDA_ENV="${CONDA_ENV}" \
    LOYAL_MEGATRON_ROOT="${MEGATRON_ROOT}" \
    LOYAL_SHARED_CHECKPOINT_NAME="${CHECKPOINT_NAME}" \
    LOYAL_CHECKPOINT_HOST_DIR="${SAVE_ROOT}" \
    LOYAL_MIXED_LOAD="${load_root}" \
    LOYAL_MIXED_SAVE="${SAVE_ROOT}" \
    LOYAL_MIXED_NUM_ROLLOUT="${LOYAL_MIXED_NUM_ROLLOUT}" \
    LOYAL_MIXED_TRAIN_GPU_COUNT="${LOYAL_MIXED_TRAIN_GPU_COUNT}" \
    LOYAL_MIXED_ROLLOUT_GPU_COUNT="${LOYAL_MIXED_ROLLOUT_GPU_COUNT}" \
    LOYAL_MIXED_RAY_NUM_GPUS="${LOYAL_MIXED_RAY_NUM_GPUS}" \
    LOYAL_MIU_RECORDS="${MIU_RECORDS}" \
    LOYAL_EIL_RECORDS="${EIL_RECORDS}" \
    LOYAL_MIXED_TRAIN_RECORDS="${LOYAL_MIXED_TRAIN_RECORDS}" \
    LOYAL_RAY_TEMP_DIR="${RAY_TEMP_DIR}" \
    LOYAL_RAY_DIRECT_DRIVER="${LOYAL_RAY_DIRECT_DRIVER}" \
    bash "${PROJECT_ROOT}/scripts/launch/run_training_host.sh" mixed \
    >"${POST_ROOT}/train_${current_step}_to_${target_step}.log" 2>&1 &
  echo "$!" >"${POST_ROOT}/training.pid"
}

stop_training_runtime() {
  "${PYTHON%/python3}/ray" stop --force >/dev/null 2>&1 || ray stop --force >/dev/null 2>&1 || true
}

export_checkpoint() {
  local step="$1"; local native_root="${SAVE_ROOT}"; local export_root="${POST_ROOT}/exported_models"
  local export_dir="${export_root}/${CHECKPOINT_NAME}/iter_$(printf '%07d' "${step}")"
  [[ -f "${export_dir}/model.safetensors.index.json" ]] && { printf '%s' "${export_dir}"; return 0; }
  [[ ! -e "${export_dir}" ]] || { log "ERROR incomplete_export_exists path=${export_dir}"; return 1; }
  LOYAL_BASE_MODEL=qwen3-4b \
  LOYAL_MODEL_ROOT="${MODEL_ROOT}" LOYAL_ASSET_ROOT="${ASSET_ROOT}" \
  LOYAL_PYTHON="${PYTHON}" LOYAL_CHECKPOINT_HOST_DIR="${native_root}" \
  LOYAL_SHARED_CHECKPOINT_NAME="${CHECKPOINT_NAME}" LOYAL_EXPORT_ROOT="${export_root}" \
    bash "${PROJECT_ROOT}/scripts/export_final_checkpoint_host.sh" "${CHECKPOINT_NAME}" "${step}" >"${POST_ROOT}/export_step${step}.log" 2>&1
  [[ -f "${export_dir}/model.safetensors.index.json" ]] || { log "ERROR export_incomplete step=${step}"; return 1; }
  printf '%s' "${export_dir}"
}

launch_eval() {
  local step="$1"; local iter; iter="$(iter_dir "${SAVE_ROOT}" "${step}")"
  local out="${POST_ROOT}/step${step}"
  [[ -f "${out}/suite_done" ]] && return 0
  if ! complete "${SAVE_ROOT}" "${step}"; then return 0; fi
  log "evaluation_start step=${step}"
  local hf_checkpoint
  hf_checkpoint="$(export_checkpoint "${step}")" || { log "ERROR checkpoint_export_failed step=${step}"; return 1; }
  CUDA_VISIBLE_DEVICES="${EVAL_GPU}" LOYAL_T1_EVAL_DEVICE=cuda:0 \
    bash "${PROJECT_ROOT}/scripts/evaluation/run_t1_checkpoint_suite_101.sh" "${hf_checkpoint}" "${out}/benchmarks" >"${out}.log" 2>&1 || { log "ERROR benchmark_eval_failed step=${step}"; return 1; }
  touch "${out}/suite_done"
  log "evaluation_done step=${step}"
}

main() {
  log "supervisor_start source=${SOURCE_ROOT} save=${SAVE_ROOT} target=${TARGET_STEP} eval_steps=${EVAL_STEPS[*]}"
  wait_idle
  assert_runtime_paths
  assert_source
  local load_root="${SOURCE_ROOT}" current_step="${SOURCE_STEP}" target_step pid
  for target_step in "${EVAL_STEPS[@]}"; do
    if ! complete "${SAVE_ROOT}" "${target_step}"; then
      start_training "${load_root}" "${current_step}" "${target_step}"
      pid="$(<"${POST_ROOT}/training.pid")"
      while ! complete "${SAVE_ROOT}" "${target_step}"; do
        if ! kill -0 "${pid}" 2>/dev/null; then
          log "ERROR training_exited_before_checkpoint current=${current_step} target=${target_step}"
          exit 8
        fi
        log "waiting_checkpoint current=${current_step} target=${target_step}"
        sleep "${POLL}"
      done
      wait "${pid}" || { log "ERROR training_failed current=${current_step} target=${target_step}"; exit 8; }
      if [[ "${current_step}" != "0" ]]; then
        assert_strict_resume_log "${POST_ROOT}/train_${current_step}_to_${target_step}.log" "${current_step}" || exit 8
      fi
      stop_training_runtime
      log "training_done current=${current_step} target=${target_step}"
    fi
    launch_eval "${target_step}" || exit 9
    load_root="${SAVE_ROOT}"
    current_step="${target_step}"
  done
  log "supervisor_complete"
}
main "$@"
