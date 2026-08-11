#!/usr/bin/env bash
# Run the baseline EIL test set across GPUs 0--5, then start the EIL GRPO job
# on the same six GPUs (two actor/train GPUs and four single-GPU rollout engines).
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/env.sh"

RUN_NAME="${1:-eil_6gpu_$(date -u +%Y%m%dT%H%M%SZ)}"
if [[ ! "${RUN_NAME}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "run name must be a simple directory name" >&2
  exit 2
fi

: "${LOYAL_MODEL_ROOT:?set LOYAL_MODEL_ROOT in .env}"
: "${LOYAL_SLIME_IMAGE:=slimerl/slime:nightly-dev-202511127a}"
if ! docker image inspect "${LOYAL_SLIME_IMAGE}" >/dev/null 2>&1; then
  echo "missing ${LOYAL_SLIME_IMAGE}" >&2
  exit 1
fi
if [[ ! -d "${LOYAL_MODEL_ROOT}/Qwen3-4B" ]]; then
  echo "missing ${LOYAL_MODEL_ROOT}/Qwen3-4B" >&2
  exit 1
fi

export LOYAL_EIL_TRAIN_GPU_DEVICES=0,1,2,3,4,5
export LOYAL_EIL_TRAIN_GPU_COUNT=2
export LOYAL_EIL_ROLLOUT_GPU_COUNT=4
export LOYAL_EIL_RAY_NUM_GPUS=6
export LOYAL_USE_WANDB=1
export LOYAL_WANDB_GROUP="eil-qwen3-4b-grpo-${RUN_NAME}"
# Throughput-oriented EIL run: direct answers, a 1,024-token output cap, and
# a large KV-cache pool on the four dedicated 80 GB rollout GPUs.
export LOYAL_EIL_MAX_RESPONSE_LEN=1024
export LOYAL_EIL_SGLANG_MEM_FRACTION_STATIC=0.70
# Every sample makes two concurrent judge calls (leakage and utility). With
# four rollout workers, one active group per worker and four scored samples
# per group exactly uses the 32-request aggregate judge limit: 4 * 1 * 4 * 2.
export LOYAL_EIL_GROUP_RM_MAX_CONCURRENT=1
export LOYAL_EIL_RM_MAX_CONCURRENT=4
export LOYAL_EIL_EVAL_INTERVAL=100
export LOYAL_EIL_SAVE_INTERVAL=50
# Test workers are independent processes.  Two simultaneously scored samples
# per worker mean at most 24 judge calls globally (6 workers * 2 samples * 2
# judge roles), below the 32-call provider limit.  Eight prompts avoids one
# unusually long response holding an entire 32-sample evaluation batch open.
export LOYAL_EIL_TEST_BATCH_SIZE=8
export LOYAL_EIL_TEST_SCORE_CONCURRENCY=8
export LOYAL_EIL_TEST_MAX_NEW_TOKENS=1024

EVALUATION_ROOT="${PROJECT_ROOT}/artifacts/evaluations"
OUTPUT_NAME="eil_baseline_${RUN_NAME}"
OUTPUT_DIR="${EVALUATION_ROOT}/${OUTPUT_NAME}"
LOG_DIR="${PROJECT_ROOT}/artifacts/logs/${RUN_NAME}"
mkdir -p "${EVALUATION_ROOT}" "${LOG_DIR}"
if [[ -e "${OUTPUT_DIR}" ]]; then
  echo "refusing to overwrite existing evaluation output: ${OUTPUT_DIR}" >&2
  exit 1
fi
mkdir -p "${OUTPUT_DIR}/shards" "${OUTPUT_DIR}/workers"

python3 "${PROJECT_ROOT}/scripts/training/preflight.py" eil
for gpu in 0 1 2 3 4 5; do
  awk -v worker="${gpu}" '((NR - 1) % 6) == worker { print }' \
    "${PROJECT_ROOT}/eil/data/dataset/EIL/test.jsonl" > "${OUTPUT_DIR}/shards/gpu${gpu}.jsonl"
done

echo "[$(date -u +%FT%TZ)] starting six-way EIL baseline evaluation: ${OUTPUT_DIR}"
pids=()
for gpu in 0 1 2 3 4 5; do
  container="loyal-eil-test-${RUN_NAME}-gpu${gpu}"
  # Each scored sample produces one leakage and one utility judge request.
  # Give three workers three slots and three workers two: (3+3+3+2+2+2)
  # * 2 judge roles = 30 aggregate calls, safely below the 32-call limit.
  if [[ "${gpu}" -lt 3 ]]; then
    judge_slots=3
  else
    judge_slots=2
  fi
  docker run --rm --name "${container}" --gpus "device=${gpu}" --network host --ipc host --shm-size=16g --entrypoint bash \
    --env-file "${PROJECT_ROOT}/.env" \
    -e "LOYAL_QWEN3_4B_HF_CHECKPOINT=/models/Qwen3-4B" \
    -v "${PROJECT_ROOT}:/workspace/loyal_agent:ro" \
    -v "${EVALUATION_ROOT}:/outputs" \
    -v "${LOYAL_MODEL_ROOT}:/models:ro" \
    -w /workspace/loyal_agent \
    "${LOYAL_SLIME_IMAGE}" \
    -lc "export PYTHONPATH=/workspace/loyal_agent:\${PYTHONPATH:-}; source scripts/launch/env.sh && export LOYAL_EIL_LEAKAGE_JUDGE_MAX_CONCURRENT=${judge_slots} LOYAL_EIL_UTILITY_JUDGE_MAX_CONCURRENT=${judge_slots}; python3 -m scripts.evaluation.cli eil --checkpoint /models/Qwen3-4B --records /outputs/${OUTPUT_NAME}/shards/gpu${gpu}.jsonl --output-dir /outputs/${OUTPUT_NAME}/workers/gpu${gpu} --device cuda:0 --batch-size ${LOYAL_EIL_TEST_BATCH_SIZE} --score-concurrency ${LOYAL_EIL_TEST_SCORE_CONCURRENCY} --max-new-tokens ${LOYAL_EIL_TEST_MAX_NEW_TOKENS}" \
    > "${LOG_DIR}/eil_test_gpu${gpu}.log" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done
if [[ "${failed}" -ne 0 ]]; then
  echo "[$(date -u +%FT%TZ)] one or more EIL test workers failed; training was not started" >&2
  exit 1
fi

for gpu in 0 1 2 3 4 5; do
  test -f "${OUTPUT_DIR}/workers/gpu${gpu}/summary.json"
done
python3 -m scripts.evaluation.merge_eil_shards \
  --workers-dir "${OUTPUT_DIR}/workers" \
  --records "${PROJECT_ROOT}/eil/data/dataset/EIL/test.jsonl" \
  --output "${OUTPUT_DIR}/summary.json" \
  | tee "${LOG_DIR}/eil_test_summary.log"
echo "[$(date -u +%FT%TZ)] six-way EIL baseline evaluation completed"

export LOYAL_DOCKER_DETACH=1
export LOYAL_DOCKER_KEEP_CONTAINER=1
export LOYAL_DOCKER_CONTAINER_NAME="loyal-eil-train-${RUN_NAME}"
python3 -m scripts.evaluation.append_eil_report \
  --summary "${OUTPUT_DIR}/summary.json" \
  --report "${PROJECT_ROOT}/report.md" \
  --training-container "${LOYAL_DOCKER_CONTAINER_NAME}"
echo "[$(date -u +%FT%TZ)] starting EIL training: train=2 rollout=4 wandb_group=${LOYAL_WANDB_GROUP}"
bash "${SCRIPT_DIR}/run_training_container.sh" eil
echo "[$(date -u +%FT%TZ)] EIL training container started: ${LOYAL_DOCKER_CONTAINER_NAME}"
