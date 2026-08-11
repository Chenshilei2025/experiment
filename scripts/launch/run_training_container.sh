#!/usr/bin/env bash
# Run one task in SLIME's prebuilt GPU image with checkpoints persisted on the host.
set -euo pipefail

if [[ $# -ne 1 || ( "$1" != "miu" && "$1" != "eil" ) ]]; then
  echo "usage: $0 {miu|eil}" >&2
  exit 2
fi

MECHANISM="$1"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/env.sh"

# The GPU container cannot reach external judge hosts directly, while the host
# can. Forward the container's TLS connection through a loopback-only host
# relay without terminating TLS or exposing judge credentials to the relay.
if [[ "${MECHANISM}" == "miu" && "${LOYAL_MIU_JUDGE_BASE_URL:-}" == https://new.pumpkinai.vip:18443/v1 ]]; then
  bash "${SCRIPT_DIR}/start_judge_tcp_proxy.sh" new.pumpkinai.vip
  JUDGE_HOST_ARGS=(--add-host "new.pumpkinai.vip:127.0.0.1")
else
  JUDGE_HOST_ARGS=()
fi

: "${LOYAL_MODEL_ROOT:?set LOYAL_MODEL_ROOT in .env to the directory containing Qwen3-4B and Qwen3-4B_torch_dist}"
# Do not silently move to ``latest``: this is the image matched to SLIME v0.2.0.
: "${LOYAL_SLIME_IMAGE:=slimerl/slime:nightly-dev-202511127a}"
if ! docker image inspect "${LOYAL_SLIME_IMAGE}" >/dev/null 2>&1; then
  echo "missing ${LOYAL_SLIME_IMAGE}; run: docker pull ${LOYAL_SLIME_IMAGE}" >&2
  exit 1
fi
if [[ ! -d "${LOYAL_MODEL_ROOT}/Qwen3-4B" || ! -d "${LOYAL_MODEL_ROOT}/Qwen3-4B_torch_dist" ]]; then
  echo "LOYAL_MODEL_ROOT must contain Qwen3-4B and Qwen3-4B_torch_dist" >&2
  exit 1
fi

mkdir -p "${PROJECT_ROOT}/artifacts/checkpoints"
CHECKPOINT_NAME="${LOYAL_SHARED_CHECKPOINT_NAME:-Qwen3-4B_${MECHANISM}_slime}"
if [[ ! "${CHECKPOINT_NAME}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "LOYAL_SHARED_CHECKPOINT_NAME must be a simple checkpoint directory name" >&2
  exit 1
fi
CHECKPOINT_DIR="/workspace/loyal_agent/artifacts/checkpoints/${CHECKPOINT_NAME}"
GPU_ARGS=(--gpus all)
if [[ "${MECHANISM}" == "eil" ]]; then
  # The EIL adversary is API-backed; this selects the six GPUs used by Ray.
  # Docker's GPU parser requires quoted comma-separated device IDs.
  GPU_ARGS=(--gpus "\"device=${LOYAL_EIL_TRAIN_GPU_DEVICES}\"")
elif [[ "${MECHANISM}" == "miu" && -n "${LOYAL_MIU_GPU_DEVICES:-}" ]]; then
  # Permit MIU to use a non-contiguous set when other host workloads occupy
  # some GPUs. Docker renumbers the selected devices inside the container.
  GPU_ARGS=(--gpus "\"device=${LOYAL_MIU_GPU_DEVICES}\"")
fi

DOCKER_RUN_ARGS=()
if [[ "${LOYAL_DOCKER_DETACH:-0}" == "1" ]]; then
  # Keep Ray and its submitted training job alive after this launcher returns.
  DOCKER_RUN_ARGS+=(--detach)
fi
if [[ "${LOYAL_DOCKER_KEEP_CONTAINER:-0}" != "1" ]]; then
  DOCKER_RUN_ARGS+=(--rm)
fi

# These are deliberately forwarded only when set in the host launcher.  They
# make short smoke runs reproducible without editing the credential-bearing
# .env file, while normal launches continue to use its defaults.
TRAINING_OVERRIDE_NAMES=(
  LOYAL_SHARED_CHECKPOINT_NAME
  LOYAL_MIU_NUM_ROLLOUT LOYAL_EIL_NUM_ROLLOUT
  LOYAL_MIU_ROLLOUT_BATCH_SIZE LOYAL_EIL_ROLLOUT_BATCH_SIZE
  LOYAL_MIU_SAMPLES_PER_PROMPT LOYAL_EIL_SAMPLES_PER_PROMPT
  LOYAL_MIU_GLOBAL_BATCH_SIZE LOYAL_EIL_GLOBAL_BATCH_SIZE
  LOYAL_MIU_MAX_RESPONSE_LEN LOYAL_MIU_MAX_TOKENS_PER_GPU
  LOYAL_MIU_SAVE_INTERVAL LOYAL_EIL_SAVE_INTERVAL
  LOYAL_MIU_SAVE_RETAIN_INTERVAL LOYAL_EIL_SAVE_RETAIN_INTERVAL
  LOYAL_MIU_DISABLE_EVAL LOYAL_EIL_EVAL_INTERVAL
  LOYAL_USE_WANDB LOYAL_WANDB_PROJECT LOYAL_WANDB_GROUP LOYAL_WANDB_MODE
  LOYAL_EIL_MAX_RESPONSE_LEN LOYAL_EIL_MAX_TOKENS_PER_GPU
  LOYAL_EIL_SGLANG_MEM_FRACTION_STATIC LOYAL_EIL_RM_MAX_CONCURRENT LOYAL_EIL_GROUP_RM_MAX_CONCURRENT
  LOYAL_MIU_GPU_DEVICES LOYAL_MIU_TRAIN_GPU_COUNT
  LOYAL_MIU_ROLLOUT_GPU_COUNT LOYAL_MIU_RAY_NUM_GPUS
  LOYAL_EIL_TRAIN_GPU_DEVICES LOYAL_EIL_TRAIN_GPU_COUNT
  LOYAL_EIL_ROLLOUT_GPU_COUNT LOYAL_EIL_RAY_NUM_GPUS
)
TRAINING_OVERRIDE_ARGS=()
for name in "${TRAINING_OVERRIDE_NAMES[@]}"; do
  if [[ -v "${name}" ]]; then
    TRAINING_OVERRIDE_ARGS+=(-e "${name}=${!name}")
  fi
done

docker run --name "${LOYAL_DOCKER_CONTAINER_NAME:-loyal-${MECHANISM}-next}" "${DOCKER_RUN_ARGS[@]}" "${GPU_ARGS[@]}" "${JUDGE_HOST_ARGS[@]}" --network host --ipc host --shm-size=16g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --env-file "${PROJECT_ROOT}/.env" \
  "${TRAINING_OVERRIDE_ARGS[@]}" \
  -e "LOYAL_CONTAINER_QWEN3_4B_HF_CHECKPOINT=/models/Qwen3-4B" \
  -e "LOYAL_CONTAINER_QWEN3_4B_REF_LOAD=/models/Qwen3-4B_torch_dist" \
  -e "LOYAL_${MECHANISM^^}_TRAIN_RECORDS=/workspace/loyal_agent/${MECHANISM}/data/dataset/${MECHANISM^^}/train.jsonl" \
  -e "LOYAL_${MECHANISM^^}_VAL_RECORDS=/workspace/loyal_agent/${MECHANISM}/data/dataset/${MECHANISM^^}/val.jsonl" \
  -e "LOYAL_${MECHANISM^^}_LOAD=${CHECKPOINT_DIR}" \
  -e "LOYAL_${MECHANISM^^}_SAVE=${CHECKPOINT_DIR}" \
  -v "${PROJECT_ROOT}:/workspace/loyal_agent" \
  -v "${LOYAL_MODEL_ROOT}:/models:ro" \
  -w /workspace/loyal_agent \
  "${LOYAL_SLIME_IMAGE}" \
  bash -lc "source scripts/launch/env.sh && python3 scripts/training/preflight.py ${MECHANISM} --runtime && bash scripts/launch/run-qwen3-4B-${MECHANISM}.sh"
