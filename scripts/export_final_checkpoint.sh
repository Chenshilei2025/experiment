#!/usr/bin/env bash
# Convert the latest complete SLIME torch_dist checkpoint into a testable HF model.
set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  echo "usage: $0 [checkpoint-name] [iteration]"
  exit 0
fi
if [[ $# -gt 2 ]]; then
  echo "usage: $0 [checkpoint-name] [iteration]" >&2
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/launch/env.sh"

CHECKPOINT_NAME="${1:-${LOYAL_SHARED_CHECKPOINT_NAME:-Qwen3-4B_loyal}}"
if [[ ! "${CHECKPOINT_NAME}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "checkpoint name must be a simple directory name" >&2
  exit 2
fi
CHECKPOINT_ROOT="${PROJECT_ROOT}/artifacts/checkpoints/${CHECKPOINT_NAME}"
if [[ $# -eq 2 ]]; then
  ITERATION="$2"
else
  ITERATION="$(<"${CHECKPOINT_ROOT}/latest_checkpointed_iteration.txt")"
fi
if [[ ! "${ITERATION}" =~ ^[0-9]+$ ]]; then
  echo "checkpoint iteration must be numeric" >&2
  exit 2
fi
INPUT_DIR="${CHECKPOINT_ROOT}/iter_$(printf '%07d' "${ITERATION}")"
if [[ ! -f "${INPUT_DIR}/common.pt" || ! -f "${INPUT_DIR}/.metadata" ]]; then
  echo "not a complete torch_dist checkpoint: ${INPUT_DIR}" >&2
  exit 1
fi

OUTPUT_DIR="${PROJECT_ROOT}/artifacts/exported_models/${CHECKPOINT_NAME}/iter_$(printf '%07d' "${ITERATION}")"
if [[ -e "${OUTPUT_DIR}" ]]; then
  echo "refusing to overwrite existing exported model: ${OUTPUT_DIR}" >&2
  exit 1
fi
mkdir -p "$(dirname -- "${OUTPUT_DIR}")"

: "${LOYAL_SLIME_IMAGE:=slimerl/slime:nightly-dev-202511127a}"
docker run --rm --network none --entrypoint bash \
  -v "${PROJECT_ROOT}:/workspace/loyal_agent:ro" \
  -v "${LOYAL_MODEL_ROOT}:/models:ro" \
  -v "${OUTPUT_DIR}:/output" \
  -w /workspace/loyal_agent \
  "${LOYAL_SLIME_IMAGE}" \
  -lc 'PYTHONPATH=/root/Megatron-LM python3 slime/tools/convert_torch_dist_to_hf.py \
    --input-dir "$1" --output-dir /output --origin-hf-dir /models/Qwen3-4B --vocab-size 151936 --force' \
  bash "/workspace/loyal_agent/${INPUT_DIR#"${PROJECT_ROOT}/"}"

test -f "${OUTPUT_DIR}/config.json"
test -f "${OUTPUT_DIR}/model.safetensors.index.json"
printf 'Exported final HF model: %s\n' "${OUTPUT_DIR}"
