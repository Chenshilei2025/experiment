#!/usr/bin/env bash
# Creative-generation SFT recipe.  This is a checkpoint-producing pre-stage
# for later EIL/MIU evaluation, not an EIL/MIU reward-training run.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
if [[ -d "${PROJECT_ROOT}/assets" ]]; then
  ASSET_ROOT="${LOYAL_ASSET_ROOT:-${PROJECT_ROOT}/assets}"
else
  ASSET_ROOT="${LOYAL_ASSET_ROOT:-$(cd -- "${PROJECT_ROOT}/.." && pwd)/assets}"
fi
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/env.sh"
MECHANISM=creative
SLIME_ROOT="${SLIME_ROOT:-${PROJECT_ROOT}/slime}"
DATA_ROOT="${LOYAL_DATA_ROOT:-${PROJECT_ROOT}/artifacts/slime/CREATIVE}"
PYTHON="${LOYAL_PYTHON:-python3}"
source "${SCRIPT_DIR}/model_profiles.sh"

: "${LOYAL_CREATIVE_TRAIN_RECORDS:=}"
: "${LOYAL_CREATIVE_TRAIN_GPU_COUNT:=2}"
: "${LOYAL_CREATIVE_ROLLOUT_GPU_COUNT:=0}"
: "${LOYAL_CREATIVE_RAY_NUM_GPUS:=${LOYAL_CREATIVE_TRAIN_GPU_COUNT}}"
: "${LOYAL_CREATIVE_ROLLOUT_BATCH_SIZE:=32}"
: "${LOYAL_CREATIVE_GLOBAL_BATCH_SIZE:=${LOYAL_CREATIVE_ROLLOUT_BATCH_SIZE}}"
: "${LOYAL_CREATIVE_NUM_EPOCH:=1}"
: "${LOYAL_CREATIVE_SAVE_INTERVAL:=20}"
: "${LOYAL_CREATIVE_SAVE_RETAIN_INTERVAL:=1000000}"
: "${LOYAL_CREATIVE_LEARNING_RATE:=1e-5}"
: "${LOYAL_CREATIVE_MIN_LR:=1e-6}"
: "${LOYAL_CREATIVE_LR_WARMUP_FRACTION:=0.1}"
: "${LOYAL_CREATIVE_MAX_TOKENS_PER_GPU:=8192}"
: "${LOYAL_USE_WANDB:=1}"
: "${LOYAL_WANDB_PROJECT:=loyal-agent}"
: "${LOYAL_WANDB_GROUP:=creative-${LOYAL_BASE_MODEL}-sft}"
: "${LOYAL_WANDB_MODE:=online}"
: "${LOYAL_CREATIVE_BOOTSTRAP:=1}"
if [[ -z "${LOYAL_CREATIVE_TRAIN_RECORDS:-}" ]]; then
  for candidate in \
    "${ASSET_ROOT}/slime/CREATIVE/train.parquet" \
    "${DATA_ROOT}/train.parquet" \
    "${PROJECT_ROOT}/artifacts/slime/CREATIVE/train.parquet" \
    "${PROJECT_ROOT}/artifacts/slime/creative/train.parquet"
  do
    if [[ -f "${candidate}" ]]; then
      export LOYAL_CREATIVE_TRAIN_RECORDS="${candidate}"
      break
    fi
  done
fi
if [[ -z "${LOYAL_CREATIVE_TRAIN_RECORDS:-}" && "${LOYAL_CREATIVE_BOOTSTRAP}" == "1" ]]; then
  mkdir -p "${DATA_ROOT}"
  "${PYTHON}" "${PROJECT_ROOT}/scripts/data/bootstrap_creative_slime.py" \
    --output "${DATA_ROOT}/train.parquet" \
    --cache-root "${PROJECT_ROOT}/artifacts/cache/creative_sources" \
    --seed "${LOYAL_CREATIVE_SEED:-42}" \
    --writingprompts-limit "${LOYAL_CREATIVE_WRITINGPROMPTS_LIMIT:-512}" \
    --rocstories-limit "${LOYAL_CREATIVE_ROCSTORIES_LIMIT:-512}"
  export LOYAL_CREATIVE_TRAIN_RECORDS="${DATA_ROOT}/train.parquet"
fi
[[ "${LOYAL_CREATIVE_TRAIN_GPU_COUNT}" -gt 0 ]] || { echo 'creative train GPU count must be positive' >&2; exit 2; }
[[ "${LOYAL_CREATIVE_ROLLOUT_GPU_COUNT}" -eq 0 ]] || { echo 'creative SFT must not allocate rollout GPUs' >&2; exit 2; }
[[ "${LOYAL_CREATIVE_RAY_NUM_GPUS}" -eq "${LOYAL_CREATIVE_TRAIN_GPU_COUNT}" ]] || { echo 'creative Ray GPU count must equal train GPU count' >&2; exit 2; }
[[ -f "${LOYAL_CREATIVE_TRAIN_RECORDS}" ]] || { echo "creative train records do not exist: ${LOYAL_CREATIVE_TRAIN_RECORDS}" >&2; exit 2; }

export LOYAL_CREATIVE_TRAIN_GPU_COUNT LOYAL_CREATIVE_ROLLOUT_GPU_COUNT LOYAL_CREATIVE_RAY_NUM_GPUS
export LOYAL_CREATIVE_ROLLOUT_BATCH_SIZE LOYAL_CREATIVE_GLOBAL_BATCH_SIZE LOYAL_CREATIVE_NUM_EPOCH
export LOYAL_CREATIVE_SAVE_INTERVAL LOYAL_CREATIVE_SAVE_RETAIN_INTERVAL LOYAL_CREATIVE_LEARNING_RATE
export LOYAL_CREATIVE_MIN_LR LOYAL_CREATIVE_LR_WARMUP_FRACTION LOYAL_CREATIVE_MAX_TOKENS_PER_GPU
export LOYAL_USE_WANDB LOYAL_WANDB_PROJECT LOYAL_WANDB_GROUP LOYAL_WANDB_MODE
"${PYTHON}" "${PROJECT_ROOT}/scripts/training/preflight.py" creative --runtime

CKPT_ARGS=(
  --hf-checkpoint "${LOYAL_MODEL_HF_CHECKPOINT}" --ref-load "${LOYAL_MODEL_REF_LOAD}"
  --load "${LOYAL_CREATIVE_LOAD:-/root/${LOYAL_BASE_MODEL}_creative_slime}"
  --save "${LOYAL_CREATIVE_SAVE:-/root/${LOYAL_BASE_MODEL}_creative_slime}"
  --save-interval "${LOYAL_CREATIVE_SAVE_INTERVAL}"
  --save-retain-interval "${LOYAL_CREATIVE_SAVE_RETAIN_INTERVAL}"
)
ROLLOUT_ARGS=(
  --rollout-function-path slime.rollout.sft_rollout.generate_rollout
  --prompt-data "${LOYAL_CREATIVE_TRAIN_RECORDS}" --input-key messages --rollout-shuffle
  --rollout-batch-size "${LOYAL_CREATIVE_ROLLOUT_BATCH_SIZE}" --global-batch-size "${LOYAL_CREATIVE_GLOBAL_BATCH_SIZE}"
  --loss-type sft_loss --calculate-per-token-loss --disable-compute-advantages-and-returns --debug-train-only
)
[[ -z "${LOYAL_CREATIVE_NUM_ROLLOUT:-}" ]] && ROLLOUT_ARGS+=(--num-epoch "${LOYAL_CREATIVE_NUM_EPOCH}") || ROLLOUT_ARGS+=(--num-rollout "${LOYAL_CREATIVE_NUM_ROLLOUT}")
RM_ARGS=()
OPTIMIZER_ARGS=(--optimizer adam --lr "${LOYAL_CREATIVE_LEARNING_RATE}" --lr-decay-style cosine --min-lr "${LOYAL_CREATIVE_MIN_LR}" --lr-warmup-fraction "${LOYAL_CREATIVE_LR_WARMUP_FRACTION}" --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.95 --clip-grad 1.0)
GRPO_ARGS=()
PERF_ARGS=(--tensor-model-parallel-size 1 --recompute-granularity full --recompute-method uniform --recompute-num-layers 1 --use-dynamic-batch-size --max-tokens-per-gpu "${LOYAL_CREATIVE_MAX_TOKENS_PER_GPU}")
SGLANG_ARGS=()
EVAL_ARGS=()
WANDB_ARGS=()
if [[ "${LOYAL_USE_WANDB:-0}" == "1" ]]; then
  [[ -z "${WANDB_API_KEY:-}" ]] || export WANDB_API_KEY
  WANDB_ARGS=(--use-wandb --wandb-project "${LOYAL_WANDB_PROJECT}" --wandb-group "${LOYAL_WANDB_GROUP}" --wandb-mode "${LOYAL_WANDB_MODE}")
fi
MISC_ARGS=(--attention-dropout 0.0 --hidden-dropout 0.0 --no-gradient-accumulation-fusion --no-masked-softmax-fusion --accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32 --attention-backend flash --no-rope-fusion)
if [[ -n "${LOYAL_CREATIVE_CKPT_STEP:-}" ]]; then
  MISC_ARGS+=(--ckpt-step "${LOYAL_CREATIVE_CKPT_STEP}")
fi
if [[ "${LOYAL_CREATIVE_NO_LOAD_OPTIM:-1}" == "1" ]]; then
  MISC_ARGS+=(--no-load-optim --no-load-rng --finetune)
fi
TRAIN_GPU_COUNT="${LOYAL_CREATIVE_TRAIN_GPU_COUNT}"; ROLLOUT_GPU_COUNT="${LOYAL_CREATIVE_ROLLOUT_GPU_COUNT}"; RAY_GPU_COUNT="${LOYAL_CREATIVE_RAY_NUM_GPUS}"
RUNTIME_EXTRA=',"PYTORCH_CUDA_ALLOC_CONF":"expandable_segments:True"'
source "${SCRIPT_DIR}/submit_training.sh"
