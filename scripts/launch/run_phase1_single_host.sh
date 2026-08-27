#!/usr/bin/env bash
# Launch one phase-1 reward-ablation condition on a single 4-GPU host.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

CONDITION="${1:-${LOYAL_PHASE1_CONDITION:-}}"
if [[ -z "${CONDITION}" ]]; then
  echo "usage: $0 phase1-lambda025-e1m1|phase1-lambda050-e1m1|phase1-lambda050-e1m1-rollout160|phase1-lambda075-e1m1" >&2
  exit 2
fi
case "${CONDITION}" in
  phase1-lambda025-e1m1|phase1-lambda050-e1m1|phase1-lambda050-e1m1-rollout160|phase1-lambda075-e1m1) ;;
  *) echo "unsupported phase1 condition: ${CONDITION}" >&2; exit 2 ;;
esac

GPUS="${LOYAL_PHASE1_GPUS:-0,1,2,3}"
PORT_BASE="${LOYAL_PHASE1_RAY_PORT_BASE:-26379}"
P1_OUT="${LOYAL_PHASE1_OUTPUT_ROOT:-${PROJECT_ROOT}/artifacts/experiments/mixed_reward_ablation_phase1_parallel}"
OUTPUT_DIR="${P1_OUT}/${CONDITION}-phase1"
LOG_DIR="${P1_OUT}/launcher_logs"
LOG_PATH="${LOG_DIR}/${CONDITION}.log"
WORKER_PORTS=""

mkdir -p "${LOG_DIR}" "$(dirname -- "${OUTPUT_DIR}")"
for port in $(seq "$((PORT_BASE + 20))" "$((PORT_BASE + 219))"); do
  WORKER_PORTS="${WORKER_PORTS}${WORKER_PORTS:+,}${port}"
done

cd "${PROJECT_ROOT}"
export CUDA_VISIBLE_DEVICES="${GPUS}"
export LOYAL_MIXED_TRAIN_GPU_DEVICES="${GPUS}"
export LOYAL_TRAINING_LAUNCHER="${LOYAL_TRAINING_LAUNCHER:-scripts/launch/run_training_host.sh}"
export LOYAL_TEST_LAUNCHER="${LOYAL_TEST_LAUNCHER:-scripts/run_test_host.sh}"
export LOYAL_EXPORT_LAUNCHER="${LOYAL_EXPORT_LAUNCHER:-scripts/export_final_checkpoint_host.sh}"
export MASTER_ADDR="${LOYAL_RAY_NODE_IP:-127.0.0.1}"
export LOYAL_RAY_STOP_BEFORE_START="${LOYAL_RAY_STOP_BEFORE_START:-1}"
export LOYAL_RAY_STOP_AFTER_EXIT="${LOYAL_RAY_STOP_AFTER_EXIT:-1}"
export LOYAL_RAY_DIRECT_DRIVER=1
export LOYAL_JUDGE_CIRCUIT_ACTION="${LOYAL_JUDGE_CIRCUIT_ACTION:-soft_keep}"
export LOYAL_JUDGE_CIRCUIT_FAILURE_THRESHOLD="${LOYAL_JUDGE_CIRCUIT_FAILURE_THRESHOLD:-100}"
export LOYAL_RAY_HEAD_PORT="${PORT_BASE}"
export LOYAL_RAY_DASHBOARD_PORT="$((PORT_BASE + 1))"
export LOYAL_RAY_CLIENT_PORT="$((PORT_BASE + 2))"
export LOYAL_RAY_DASHBOARD_AGENT_HTTP_PORT="$((PORT_BASE + 3))"
export LOYAL_RAY_DASHBOARD_AGENT_GRPC_PORT="$((PORT_BASE + 4))"
export LOYAL_RAY_RUNTIME_ENV_AGENT_PORT="$((PORT_BASE + 5))"
export LOYAL_RAY_METRICS_PORT="$((PORT_BASE + 6))"
export LOYAL_RAY_OBJECT_MANAGER_PORT="$((PORT_BASE + 7))"
export LOYAL_RAY_NODE_MANAGER_PORT="$((PORT_BASE + 8))"
export LOYAL_RAY_WORKER_PORT_LIST="${WORKER_PORTS}"
export LOYAL_RAY_TEMP_DIR="${LOYAL_RAY_TEMP_DIR:-/tmp/ray-${CONDITION}}"
export LOYAL_DATA_ROOT="${OUTPUT_DIR}/slime_data"
export LOYAL_MIU_GROUP_RM_MAX_ATTEMPTS="${LOYAL_MIU_GROUP_RM_MAX_ATTEMPTS:-3}"
export LOYAL_MIU_GROUP_RM_RETRY_INITIAL_BACKOFF_SECONDS="${LOYAL_MIU_GROUP_RM_RETRY_INITIAL_BACKOFF_SECONDS:-3}"
export LOYAL_EIL_GROUP_RM_MAX_ATTEMPTS="${LOYAL_EIL_GROUP_RM_MAX_ATTEMPTS:-4}"
export LOYAL_EIL_GROUP_RM_RETRY_INITIAL_BACKOFF_SECONDS="${LOYAL_EIL_GROUP_RM_RETRY_INITIAL_BACKOFF_SECONDS:-5}"

EXTRA_ARGS=()
if [[ "${LOYAL_PHASE1_SKIP_BASELINE_EVAL:-1}" == "1" ]]; then
  EXTRA_ARGS+=(--set evaluation.baseline=false)
fi
if [[ "${LOYAL_PHASE1_SKIP_RUNNER_EVAL:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--set evaluation.baseline=false --set evaluation.after_each_stage=false)
fi

echo "${CONDITION} gpus=${GPUS} ray_port=${PORT_BASE} log=${LOG_PATH}"
python3 -m scripts.experiment_runner \
  --config "experiments/mixed_reward_ablation/configs/${CONDITION}.json" \
  --run-name phase1 \
  --output-dir "${OUTPUT_DIR}" \
  "${EXTRA_ARGS[@]}" >"${LOG_PATH}" 2>&1
