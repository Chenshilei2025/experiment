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

: "${LOYAL_MODEL_ROOT:?set LOYAL_MODEL_ROOT in .env to the directory containing model profiles and torch_dist checkpoints}"
: "${LOYAL_BASE_MODEL:=qwen3-4b}"
case "${LOYAL_BASE_MODEL}" in
  qwen3-4b|glm-z1-9b|olmo3-7b-instruct) MODEL_MOUNT_ROOT="${LOYAL_MODEL_ROOT}" ;;
  llama3.1-8b-instruct) MODEL_MOUNT_ROOT="${LOYAL_LLAMA3_1_8B_MODEL_ROOT:-/ssd/models}" ;;
  *) echo "unsupported LOYAL_BASE_MODEL=${LOYAL_BASE_MODEL}" >&2; exit 2 ;;
esac
if [[ ! -d "${MODEL_MOUNT_ROOT}" ]]; then
  echo "model root does not exist: ${MODEL_MOUNT_ROOT}" >&2
  exit 1
fi
# Do not silently move to ``latest``: this is the image matched to SLIME v0.2.0.
: "${LOYAL_SLIME_IMAGE:=slimerl/slime:nightly-dev-202511127a}"
if ! docker image inspect "${LOYAL_SLIME_IMAGE}" >/dev/null 2>&1; then
  echo "missing ${LOYAL_SLIME_IMAGE}; run: docker pull ${LOYAL_SLIME_IMAGE}" >&2
  exit 1
fi
mkdir -p "${PROJECT_ROOT}/artifacts/checkpoints"
CHECKPOINT_NAME="${LOYAL_SHARED_CHECKPOINT_NAME:-${LOYAL_BASE_MODEL}_${MECHANISM}_slime}"
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

# Forward every exported LOYAL_* value so a versioned experiment config can
# select any supported training knob without changing this wrapper.
TRAINING_OVERRIDE_ARGS=()
while IFS= read -r name; do
  [[ "${name}" =~ ^LOYAL_[A-Z0-9_]+$ ]] || continue
  [[ "${name}" =~ (_API_KEY|_API_KEYS|_BASE_URL)$ ]] && continue
  case "${name}" in
    LOYAL_MIU_JUDGE_MODEL|LOYAL_EIL_JUDGE_MODEL|LOYAL_EIL_ADVERSARY_MODEL) continue ;;
  esac
  TRAINING_OVERRIDE_ARGS+=(-e "${name}=${!name}")
done < <(compgen -e)

# Canonical project data stays inside the repository mount. Conditions can
# change preparation parameters, but changing host data mounts is separate
# infrastructure rather than a JSON experiment knob.
docker run --name "${LOYAL_DOCKER_CONTAINER_NAME:-loyal-${MECHANISM}-next}" "${DOCKER_RUN_ARGS[@]}" "${GPU_ARGS[@]}" "${JUDGE_HOST_ARGS[@]}" --network host --ipc host --shm-size=16g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --env-file "${PROJECT_ROOT}/.env" \
  "${TRAINING_OVERRIDE_ARGS[@]}" \
  -e "LOYAL_BASE_MODEL=${LOYAL_BASE_MODEL}" \
  -e "LOYAL_${MECHANISM^^}_TRAIN_RECORDS=/workspace/loyal_agent/${MECHANISM}/data/dataset/${MECHANISM^^}/train.jsonl" \
  -e "LOYAL_${MECHANISM^^}_VAL_RECORDS=/workspace/loyal_agent/${MECHANISM}/data/dataset/${MECHANISM^^}/val.jsonl" \
  -e "LOYAL_${MECHANISM^^}_LOAD=${CHECKPOINT_DIR}" \
  -e "LOYAL_${MECHANISM^^}_SAVE=${CHECKPOINT_DIR}" \
  -v "${PROJECT_ROOT}:/workspace/loyal_agent" \
  -v "${MODEL_MOUNT_ROOT}:/models:ro" \
  -w /workspace/loyal_agent \
  "${LOYAL_SLIME_IMAGE}" \
  bash -lc "source scripts/launch/env.sh && bash scripts/launch/run-${MECHANISM}.sh"
