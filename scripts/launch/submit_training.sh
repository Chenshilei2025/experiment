#!/usr/bin/env bash
# Shared container-side Ray/SLIME submission. Recipe scripts define the model,
# data, reward function, and hyperparameters, then source this file.
set -euo pipefail

: "${MECHANISM:?recipe must set MECHANISM}"
: "${SLIME_ROOT:?recipe must set SLIME_ROOT}"
: "${TRAIN_GPU_COUNT:?recipe must set TRAIN_GPU_COUNT}"
: "${ROLLOUT_GPU_COUNT:?recipe must set ROLLOUT_GPU_COUNT}"
: "${RAY_GPU_COUNT:?recipe must set RAY_GPU_COUNT}"

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((TRAIN_GPU_COUNT + ROLLOUT_GPU_COUNT - 1)))"
fi
export CUDA_VISIBLE_DEVICES PYTHONUNBUFFERED=1
MEGATRON_ROOT="${LOYAL_MEGATRON_ROOT:-/root/Megatron-LM}"
# Ray workers inherit the environment of the raylet.  Set project imports
# before starting Ray so direct-driver launches have the same import context
# as `ray job submit` runtime environments.
export PYTHONPATH="${PROJECT_ROOT}:${SLIME_ROOT}:${MEGATRON_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

RUNTIME_ENV_JSON="{\"env_vars\":{\"PYTHONPATH\":\"${PROJECT_ROOT}:${SLIME_ROOT}:${MEGATRON_ROOT}\",\"CUDA_DEVICE_MAX_CONNECTIONS\":\"1\"${RUNTIME_EXTRA:-},\"NO_PROXY\":\"${NO_PROXY:-}\",\"no_proxy\":\"${no_proxy:-}\"}}"
TRAIN_COMMAND=(python3 "${SLIME_ROOT}/train.py" \
  --actor-num-nodes 1 --actor-num-gpus-per-node "${TRAIN_GPU_COUNT}" --rollout-num-gpus "${ROLLOUT_GPU_COUNT}" \
  "${MODEL_ARGS[@]}" "${CKPT_ARGS[@]}" "${ROLLOUT_ARGS[@]}" "${RM_ARGS[@]}" "${OPTIMIZER_ARGS[@]}" \
  "${GRPO_ARGS[@]}" --seed "${LOYAL_TRAINING_SEED:-1234}" "${WANDB_ARGS[@]}" "${PERF_ARGS[@]}" \
  "${EVAL_ARGS[@]}" "${SGLANG_ARGS[@]}" "${MISC_ARGS[@]}")

if [[ "${LOYAL_SUBMIT_DRY_RUN:-0}" == "1" ]]; then
  printf 'DRY_RUN CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES}"
  printf 'DRY_RUN RAY_GPU_COUNT=%s TRAIN_GPU_COUNT=%s ROLLOUT_GPU_COUNT=%s\n' "${RAY_GPU_COUNT}" "${TRAIN_GPU_COUNT}" "${ROLLOUT_GPU_COUNT}"
  printf 'DRY_RUN RUNTIME_ENV_JSON=%s\n' "${RUNTIME_ENV_JSON}"
  printf 'DRY_RUN TRAIN_COMMAND='
  printf '%q ' "${TRAIN_COMMAND[@]}"
  printf '\n'
  exit 0
fi

if [[ "${LOYAL_RAY_STOP_BEFORE_START:-1}" == "1" ]]; then
  ray stop --force || true
fi
RAY_NODE_IP="${MASTER_ADDR:-$(hostname -I | awk '{print $1}') }"
RAY_NODE_IP="${RAY_NODE_IP% }"
RAY_HEAD_PORT="${LOYAL_RAY_HEAD_PORT:-6379}"
RAY_DASHBOARD_PORT="${LOYAL_RAY_DASHBOARD_PORT:-8265}"
RAY_PORT_ARGS=(
  --ray-client-server-port "${LOYAL_RAY_CLIENT_PORT:-10001}"
  --dashboard-agent-listen-port "${LOYAL_RAY_DASHBOARD_AGENT_HTTP_PORT:-52365}"
  --dashboard-agent-grpc-port "${LOYAL_RAY_DASHBOARD_AGENT_GRPC_PORT:-40469}"
  --runtime-env-agent-port "${LOYAL_RAY_RUNTIME_ENV_AGENT_PORT:-62319}"
  --metrics-export-port "${LOYAL_RAY_METRICS_PORT:-61177}"
)
if [[ -n "${LOYAL_RAY_OBJECT_MANAGER_PORT:-}" ]]; then
  RAY_PORT_ARGS+=(--object-manager-port "${LOYAL_RAY_OBJECT_MANAGER_PORT}")
fi
if [[ -n "${LOYAL_RAY_NODE_MANAGER_PORT:-}" ]]; then
  RAY_PORT_ARGS+=(--node-manager-port "${LOYAL_RAY_NODE_MANAGER_PORT}")
fi
if [[ -n "${LOYAL_RAY_WORKER_PORT_LIST:-}" ]]; then
  RAY_PORT_ARGS+=(--worker-port-list "${LOYAL_RAY_WORKER_PORT_LIST}")
fi
RAY_TEMP_ARGS=()
if [[ -n "${LOYAL_RAY_TEMP_DIR:-}" ]]; then
  mkdir -p "${LOYAL_RAY_TEMP_DIR}"
  RAY_TEMP_ARGS=(--temp-dir "${LOYAL_RAY_TEMP_DIR}")
fi
if [[ "${LOYAL_RAY_STOP_AFTER_EXIT:-0}" == "1" ]]; then
  cleanup_ray_on_exit() {
    ray stop --force || true
  }
  trap cleanup_ray_on_exit EXIT
fi
LOCAL_NO_PROXY="127.0.0.1,localhost,${RAY_NODE_IP}"
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}${LOCAL_NO_PROXY}"
export no_proxy="${no_proxy:+${no_proxy},}${LOCAL_NO_PROXY}"
ray start --head --node-ip-address "${RAY_NODE_IP}" --port "${RAY_HEAD_PORT}" --num-gpus "${RAY_GPU_COUNT}" --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port="${RAY_DASHBOARD_PORT}" "${RAY_PORT_ARGS[@]}" "${RAY_TEMP_ARGS[@]}"

cd "${SLIME_ROOT}"
if [[ "${LOYAL_RAY_DIRECT_DRIVER:-0}" == "1" ]]; then
  # A second host-network container needs its own Ray control plane.  Running
  # the local driver directly avoids the dashboard Job API, which otherwise
  # can route the submit request to the first container's control plane.
  export RAY_ADDRESS="${RAY_NODE_IP}:${RAY_HEAD_PORT}"
  "${TRAIN_COMMAND[@]}"
else
  ray job submit --address="http://127.0.0.1:${RAY_DASHBOARD_PORT}" --runtime-env-json="${RUNTIME_ENV_JSON}" -- "${TRAIN_COMMAND[@]}"
fi
