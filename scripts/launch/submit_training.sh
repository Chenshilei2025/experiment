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

ray stop --force || true
RAY_NODE_IP="${MASTER_ADDR:-$(hostname -I | awk '{print $1}') }"
RAY_NODE_IP="${RAY_NODE_IP% }"
LOCAL_NO_PROXY="127.0.0.1,localhost,${RAY_NODE_IP}"
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}${LOCAL_NO_PROXY}"
export no_proxy="${no_proxy:+${no_proxy},}${LOCAL_NO_PROXY}"
ray start --head --node-ip-address "${RAY_NODE_IP}" --num-gpus "${RAY_GPU_COUNT}" --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

RUNTIME_ENV_JSON="{\"env_vars\":{\"PYTHONPATH\":\"${PROJECT_ROOT}:/root/Megatron-LM\",\"CUDA_DEVICE_MAX_CONNECTIONS\":\"1\"${RUNTIME_EXTRA:-},\"NO_PROXY\":\"${NO_PROXY}\",\"no_proxy\":\"${no_proxy}\"}}"
cd "${SLIME_ROOT}"
ray job submit --address="http://127.0.0.1:8265" --runtime-env-json="${RUNTIME_ENV_JSON}" -- python3 "${SLIME_ROOT}/train.py" \
  --actor-num-nodes 1 --actor-num-gpus-per-node "${TRAIN_GPU_COUNT}" --rollout-num-gpus "${ROLLOUT_GPU_COUNT}" \
  "${MODEL_ARGS[@]}" "${CKPT_ARGS[@]}" "${ROLLOUT_ARGS[@]}" "${RM_ARGS[@]}" "${OPTIMIZER_ARGS[@]}" \
  "${GRPO_ARGS[@]}" --seed "${LOYAL_TRAINING_SEED:-1234}" "${WANDB_ARGS[@]}" "${PERF_ARGS[@]}" \
  "${EVAL_ARGS[@]}" "${SGLANG_ARGS[@]}" "${MISC_ARGS[@]}"
