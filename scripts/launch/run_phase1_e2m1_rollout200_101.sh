#!/usr/bin/env bash
# One-command long run for the 10.220.5.101 4xA100 host.
#
# This package intentionally keeps checkpoints, Ray temp files, exports, and
# post-train outputs on the host overlay disk by default.  The shared CephFS is
# used only as an optional source for .env/model/data assets, not as the active
# checkpoint filesystem.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

CONDITION="phase1-lambda050-e2m1-rollout200"
RUN_NAME="${LOYAL_PHASE1_RUN_NAME:-phase1}"
CHECKPOINT_NAME="mixed-v2-${CONDITION}-${RUN_NAME}-seed1234"
HOST_TAG="${LOYAL_HOST_TAG:-101}"
LOCAL_ROOT="${LOYAL_LOCAL_RUN_ROOT:-/tmp/experiment_g_longtask_${HOST_TAG}}"
OUTPUT_ROOT="${LOYAL_PHASE1_OUTPUT_ROOT:-${LOCAL_ROOT}/experiments/mixed_reward_ablation_phase1_parallel}"
RUN_DIR="${OUTPUT_ROOT}/${CONDITION}-${RUN_NAME}"
LOG_DIR="${OUTPUT_ROOT}/launcher_logs"
LOG_FILE="${LOG_DIR}/${CONDITION}.log"
POST_ROOT="${LOYAL_PHASE1_POST_ROOT:-${LOCAL_ROOT}/evaluations/${CONDITION}_posttrain}"
CHECKPOINT_ROOT="${LOYAL_CHECKPOINT_HOST_ROOT:-${LOCAL_ROOT}/checkpoints}"
CHECKPOINT_DIR="${CHECKPOINT_ROOT}/${CHECKPOINT_NAME}"
# Keep this short: Ray creates AF_UNIX sockets below the session directory and
# Linux limits the full socket path to 107 bytes.
RAY_TEMP_DIR="${LOYAL_RAY_TEMP_DIR:-/tmp/r101}"
DATA_ROOT="${LOYAL_DATA_ROOT:-${LOCAL_ROOT}/slime_data/${CONDITION}}"
WATCHER_LOG_FILE="${POST_ROOT}/metrics_watcher.log"
ASSET_ROOT="${LOYAL_ASSET_ROOT:-/cephfs/shared/experiment_g/assets}"
MODEL_ROOT="${LOYAL_MODEL_ROOT:-${ASSET_ROOT}/models}"
PYTHON="${LOYAL_PYTHON:-/root/experiment_g_runtime/conda/env/bin/python3}"

ENV_SOURCE="${LOYAL_ENV_SOURCE:-}"
if [[ -z "${ENV_SOURCE}" ]]; then
  for candidate in \
    "${PROJECT_ROOT}/.env" \
    "/cephfs/shared/experiment_g/loyal_agent_docker_github/.env" \
    "/cephfs/shared/experiment_g/loyal_agent_mixed_reward_ablation_v2/.env" \
    "/cephfs/shared/experiment_g/loyal_agent_docker/.env"
  do
    if [[ -s "${candidate}" ]]; then
      ENV_SOURCE="${candidate}"
      break
    fi
  done
fi
if [[ ! -s "${PROJECT_ROOT}/.env" ]]; then
  [[ -n "${ENV_SOURCE}" && -s "${ENV_SOURCE}" ]] || { echo "missing .env; set LOYAL_ENV_SOURCE to a readable env file" >&2; exit 2; }
  cp "${ENV_SOURCE}" "${PROJECT_ROOT}/.env"
  chmod 600 "${PROJECT_ROOT}/.env" || true
fi

mkdir -p "${LOG_DIR}" "${POST_ROOT}" "${CHECKPOINT_ROOT}" "${RAY_TEMP_DIR}" "${DATA_ROOT}" "${LOCAL_ROOT}/cache"

if [[ "${LOYAL_FORCE_RESTART:-0}" == "1" ]]; then
  for path in "${RUN_DIR}" "${CHECKPOINT_DIR}" "${POST_ROOT}"; do
    if [[ -e "${path}" ]]; then
      mv "${path}" "${path}.restart.$(date +%Y%m%d-%H%M%S)"
    fi
  done
  rm -rf "${RAY_TEMP_DIR}"
  mkdir -p "${POST_ROOT}" "${CHECKPOINT_ROOT}" "${RAY_TEMP_DIR}"
fi

WORKER_PORTS=""
for port in $(seq 27400 27599); do
  WORKER_PORTS="${WORKER_PORTS}${WORKER_PORTS:+,}${port}"
done

export LOYAL_BASE_MODEL=qwen3-4b
export LOYAL_MODEL_ROOT="${MODEL_ROOT}"
export LOYAL_ASSET_ROOT="${ASSET_ROOT}"
export LOYAL_PYTHON="${PYTHON}"
export LOYAL_CONDA_SH="${LOYAL_CONDA_SH:-/root/miniforge3/etc/profile.d/conda.sh}"
export LOYAL_CONDA_ENV="${LOYAL_CONDA_ENV:-/root/experiment_g_runtime/conda/env}"
export LOYAL_MEGATRON_ROOT="${LOYAL_MEGATRON_ROOT:-/root/experiment_g_runtime/Megatron-LM}"
export LOYAL_TRAINING_LAUNCHER=scripts/launch/run_training_host.sh
export LOYAL_TEST_LAUNCHER=scripts/run_test_host.sh
export LOYAL_EXPORT_LAUNCHER=scripts/export_final_checkpoint_host.sh
export CUDA_VISIBLE_DEVICES="${LOYAL_PHASE1_GPUS:-0,1,2,3}"
export LOYAL_MIXED_TRAIN_GPU_DEVICES="${CUDA_VISIBLE_DEVICES}"
export LOYAL_MIXED_TRAIN_GPU_COUNT=2
export LOYAL_MIXED_ROLLOUT_GPU_COUNT=2
export LOYAL_MIXED_RAY_NUM_GPUS=4
export LOYAL_MIXED_EIL_BATCH_FRACTION=0.6666666666666666
export LOYAL_EIL_LEAKAGE_LAMBDA=0.5
export LOYAL_MIU_FAITHFULNESS_ETA=0.5
export LOYAL_MIXED_GLOBAL_BATCH_SIZE=512
export LOYAL_MIXED_ROLLOUT_BATCH_SIZE=64
export LOYAL_MIXED_SAMPLES_PER_PROMPT=8
export LOYAL_MIXED_MAX_RESPONSE_LEN=1024
export LOYAL_MIXED_MAX_TOKENS_PER_GPU=4096
export LOYAL_MIXED_LEARNING_RATE=2e-7
export LOYAL_MIXED_LEARNING_RATE_FILE="${LOYAL_PHASE1_LR_FILE:-${POST_ROOT}/phase1_next_lr.txt}"
export LOYAL_MIXED_KL_LOSS_COEF=0.05
export LOYAL_MIXED_ENTROPY_COEF=0.002
export LOYAL_MIXED_OPTIMIZER_CPU_OFFLOAD=0
export LOYAL_MIXED_ENABLE_EVAL=0
export LOYAL_MIXED_SAVE_INTERVAL=20
export LOYAL_MIXED_SCHEDULE_TOTAL_ROLLOUTS=200
export LOYAL_MIXED_SGLANG_MEM_FRACTION_STATIC="${LOYAL_MIXED_SGLANG_MEM_FRACTION_STATIC:-0.78}"
export LOYAL_MIXED_SGLANG_SERVER_CONCURRENCY="${LOYAL_MIXED_SGLANG_SERVER_CONCURRENCY:-64}"
export LOYAL_EIL_GROUP_RM_MAX_CONCURRENT="${LOYAL_EIL_GROUP_RM_MAX_CONCURRENT:-4}"
export LOYAL_EIL_RM_MAX_CONCURRENT="${LOYAL_EIL_RM_MAX_CONCURRENT:-32}"
export LOYAL_EIL_GROUP_RM_MAX_ATTEMPTS="${LOYAL_EIL_GROUP_RM_MAX_ATTEMPTS:-3}"
export LOYAL_EIL_GROUP_RM_RETRY_INITIAL_BACKOFF_SECONDS="${LOYAL_EIL_GROUP_RM_RETRY_INITIAL_BACKOFF_SECONDS:-2}"
export LOYAL_JUDGE_CIRCUIT_ACTION="${LOYAL_JUDGE_CIRCUIT_ACTION:-soft_keep}"
export LOYAL_JUDGE_CIRCUIT_FAILURE_THRESHOLD="${LOYAL_JUDGE_CIRCUIT_FAILURE_THRESHOLD:-100}"
export LOYAL_RETAIN_ZERO_STD_GROUPS="${LOYAL_RETAIN_ZERO_STD_GROUPS:-1}"
export MASTER_ADDR="${LOYAL_RAY_NODE_IP:-127.0.0.1}"
export LOYAL_RAY_STOP_BEFORE_START=1
export LOYAL_RAY_STOP_AFTER_EXIT=0
export LOYAL_RAY_DIRECT_DRIVER=1
export LOYAL_RAY_HEAD_PORT="${LOYAL_RAY_HEAD_PORT:-27379}"
export LOYAL_RAY_DASHBOARD_PORT="${LOYAL_RAY_DASHBOARD_PORT:-27380}"
export LOYAL_RAY_CLIENT_PORT="${LOYAL_RAY_CLIENT_PORT:-27381}"
export LOYAL_RAY_DASHBOARD_AGENT_HTTP_PORT="${LOYAL_RAY_DASHBOARD_AGENT_HTTP_PORT:-27382}"
export LOYAL_RAY_DASHBOARD_AGENT_GRPC_PORT="${LOYAL_RAY_DASHBOARD_AGENT_GRPC_PORT:-27383}"
export LOYAL_RAY_RUNTIME_ENV_AGENT_PORT="${LOYAL_RAY_RUNTIME_ENV_AGENT_PORT:-27384}"
export LOYAL_RAY_METRICS_PORT="${LOYAL_RAY_METRICS_PORT:-27385}"
export LOYAL_RAY_OBJECT_MANAGER_PORT="${LOYAL_RAY_OBJECT_MANAGER_PORT:-27386}"
export LOYAL_RAY_NODE_MANAGER_PORT="${LOYAL_RAY_NODE_MANAGER_PORT:-27387}"
export LOYAL_RAY_WORKER_PORT_LIST="${LOYAL_RAY_WORKER_PORT_LIST:-${WORKER_PORTS}}"
export LOYAL_RAY_TEMP_DIR="${RAY_TEMP_DIR}"
export LOYAL_DATA_ROOT="${DATA_ROOT}"
export LOYAL_CHECKPOINT_HOST_ROOT="${CHECKPOINT_ROOT}"
export LOYAL_CHECKPOINT_HOST_DIR="${CHECKPOINT_DIR}"
export LOYAL_SHARED_CHECKPOINT_NAME="${CHECKPOINT_NAME}"
export LOYAL_MIXED_LOAD="${CHECKPOINT_DIR}"
export LOYAL_MIXED_SAVE="${CHECKPOINT_DIR}"
export LOYAL_TRAINING_SEED=1234
export LOYAL_ROLLOUT_SEED=1234
export LOYAL_PHASE1_CHECKPOINT_NAME="${CHECKPOINT_NAME}"
export LOYAL_PHASE1_CHECKPOINT_ROOT="${CHECKPOINT_DIR}"
export LOYAL_PHASE1_RUN_DIR="${RUN_DIR}"
export LOYAL_PHASE1_POST_ROOT="${POST_ROOT}"
export LOYAL_PHASE1_POST_LOG_FILE="${POST_ROOT}/posttrain_pipeline.log"
export LOYAL_PHASE1_EVAL_STEPS="19 39 59 79 99 119 139 159 179 199"
export LOYAL_PHASE1_FINAL_STEP=199
export LOYAL_DIRECT_EVAL_STEPS="${LOYAL_PHASE1_EVAL_STEPS}"
export LOYAL_REASONING_DATA_ROOT="${LOYAL_REASONING_DATA_ROOT:-${ASSET_ROOT}/datasets}"
export LOYAL_PHASE1_RUN_REASONING="${LOYAL_PHASE1_RUN_REASONING:-1}"
export LOYAL_PHASE1_RUN_CREATIVE="${LOYAL_PHASE1_RUN_CREATIVE:-1}"
export LOYAL_PHASE1_CREATIVE_CHECKPOINT_NAME="${CHECKPOINT_NAME}-creative-sft"
export LOYAL_PHASE1_CREATIVE_CHECKPOINT_ROOT="${CHECKPOINT_ROOT}/${CHECKPOINT_NAME}-creative-sft"
export LOYAL_PHASE1_CREATIVE_GPUS="${LOYAL_PHASE1_CREATIVE_GPUS:-0,1}"
export LOYAL_PHASE1_CREATIVE_TRAIN_GPU_COUNT="${LOYAL_PHASE1_CREATIVE_TRAIN_GPU_COUNT:-2}"
export LOYAL_CREATIVE_WRITINGPROMPTS_LIMIT="${LOYAL_CREATIVE_WRITINGPROMPTS_LIMIT:-512}"
export LOYAL_CREATIVE_ROCSTORIES_LIMIT="${LOYAL_CREATIVE_ROCSTORIES_LIMIT:-512}"
export LOYAL_CREATIVE_USE_WANDB="${LOYAL_CREATIVE_USE_WANDB:-0}"
export LOYAL_EXPERIMENT_RESUME="${LOYAL_EXPERIMENT_RESUME:-0}"

if [[ "${LOYAL_REFUSE_CEPH_ACTIVE_PATHS:-1}" == "1" ]]; then
  for value in "${CHECKPOINT_ROOT}" "${CHECKPOINT_DIR}" "${POST_ROOT}" "${RAY_TEMP_DIR}" "${DATA_ROOT}"; do
    [[ "${value}" != /cephfs/* ]] || { echo "refusing active CephFS path: ${value}" >&2; exit 3; }
  done
fi

for path in "${MODEL_ROOT}/Qwen3-4B" "${MODEL_ROOT}/Qwen3-4B_torch_dist"; do
  [[ -e "${path}" ]] || { echo "missing model path: ${path}" >&2; exit 4; }
done

cd "${PROJECT_ROOT}"
"${PYTHON}" -m py_compile scripts/experiment_runner.py scripts/data/bootstrap_creative_slime.py \
  scripts/data/bootstrap_reasoning_benchmarks.py scripts/evaluation/select_best_checkpoint.py \
  scripts/evaluation/watch_phase1_metrics.py
bash -n scripts/launch/run_training_host.sh scripts/launch/run-mixed.sh scripts/launch/run-creative.sh \
  scripts/evaluation/run_phase1_posttrain_pipeline.sh scripts/evaluation/run_direct_checkpoint_eval.sh scripts/evaluation/run_reasoning_benchmarks.sh
"${PYTHON}" -m scripts.experiment_runner \
  --config "experiments/mixed_reward_ablation/configs/${CONDITION}.json" \
  --run-name "${RUN_NAME}" \
  --validate-only >/tmp/${CONDITION}.validate.json

cat >"${POST_ROOT}/launch_env.json" <<EOF
{
  "condition": "${CONDITION}",
  "checkpoint_name": "${CHECKPOINT_NAME}",
  "project_root": "${PROJECT_ROOT}",
  "run_dir": "${RUN_DIR}",
  "checkpoint_dir": "${CHECKPOINT_DIR}",
  "post_root": "${POST_ROOT}",
  "ray_temp_dir": "${RAY_TEMP_DIR}",
  "data_root": "${DATA_ROOT}",
  "model_root": "${MODEL_ROOT}",
  "eval_steps": "${LOYAL_PHASE1_EVAL_STEPS}",
  "gpu_layout": "2 train + 2 rollout on ${CUDA_VISIBLE_DEVICES}"
}
EOF

if [[ "${LOYAL_LONGTASK_DRY_RUN:-0}" == "1" ]]; then
  cat "${POST_ROOT}/launch_env.json"
  exit 0
fi

nohup bash "${PROJECT_ROOT}/scripts/launch/run_phase1_e2m1_rollout200_streaming_101.sh" \
  >"${LOG_FILE}" 2>&1 &
echo "$!" >"${LOG_DIR}/${CONDITION}.pid"
nohup "${PYTHON}" -m scripts.evaluation.watch_phase1_metrics \
  --post-root "${POST_ROOT}" \
  --steps 19 39 59 79 99 119 139 159 179 199 \
  --interval "${LOYAL_METRICS_WATCH_INTERVAL:-300}" \
  >"${WATCHER_LOG_FILE}" 2>&1 &
echo "$!" >"${LOG_DIR}/${CONDITION}.metrics_watcher.pid"

echo "started condition=${CONDITION}"
echo "manager_pid=$(cat "${LOG_DIR}/${CONDITION}.pid" 2>/dev/null || true)"
echo "metrics_watcher_pid=$(cat "${LOG_DIR}/${CONDITION}.metrics_watcher.pid" 2>/dev/null || true)"
echo "checkpoint_dir=${CHECKPOINT_DIR}"
echo "run_dir=${RUN_DIR}"
echo "post_root=${POST_ROOT}"
echo "acceptance=${POST_ROOT}/acceptance.json"
