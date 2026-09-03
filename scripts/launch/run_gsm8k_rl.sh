#!/usr/bin/env bash
# Independent GSM8K-only GRPO condition. Do not use DAPO-Math prompts here.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
source "${SCRIPT_DIR}/env.sh"
MECHANISM=gsm8k_rl
SLIME_ROOT="${SLIME_ROOT:-${PROJECT_ROOT}/slime}"
DATA_ROOT="${LOYAL_DATA_ROOT:-${PROJECT_ROOT}/artifacts/slime/GSM8K_RL}"
source "${SCRIPT_DIR}/model_profiles.sh"

: "${LOYAL_GSM8K_RL_NUM_ROLLOUT:=200}"
: "${LOYAL_GSM8K_RL_TRAIN_GPU_COUNT:=2}"
: "${LOYAL_GSM8K_RL_ROLLOUT_GPU_COUNT:=2}"
: "${LOYAL_GSM8K_RL_RAY_NUM_GPUS:=4}"
: "${LOYAL_GSM8K_RL_ROLLOUT_BATCH_SIZE:=32}"
: "${LOYAL_GSM8K_RL_GLOBAL_BATCH_SIZE:=256}"
: "${LOYAL_GSM8K_RL_SAMPLES_PER_PROMPT:=8}"
: "${LOYAL_GSM8K_RL_MAX_RESPONSE_LEN:=2048}"
: "${LOYAL_GSM8K_RL_MAX_TOKENS_PER_GPU:=8192}"
: "${LOYAL_GSM8K_RL_LEARNING_RATE:=1e-6}"
: "${LOYAL_GSM8K_RL_SAVE:?set an empty independent checkpoint directory}"

INPUT="${LOYAL_GSM8K_RL_INPUT:-${PROJECT_ROOT}/assets/datasets/gsm8k/main/train-00000-of-00001.parquet}"
PROMPTS="${LOYAL_GSM8K_RL_PROMPTS:-${DATA_ROOT}/train.parquet}"
if [[ ! -f "${PROMPTS}" ]]; then
  python3 "${PROJECT_ROOT}/scripts/data/prepare_gsm8k_rl.py" --input "${INPUT}" --output "${PROMPTS}"
fi
[[ ! -e "${LOYAL_GSM8K_RL_SAVE}" ]] || { echo "refusing to overwrite ${LOYAL_GSM8K_RL_SAVE}" >&2; exit 2; }
[[ "${LOYAL_GSM8K_RL_RAY_NUM_GPUS}" -eq $((LOYAL_GSM8K_RL_TRAIN_GPU_COUNT + LOYAL_GSM8K_RL_ROLLOUT_GPU_COUNT)) ]] || exit 2

BASE_LOAD="${LOYAL_GSM8K_RL_LOAD:-${LOYAL_MODEL_REF_LOAD}}"
[[ -e "${BASE_LOAD}" ]] || { echo "missing GSM8K RL base checkpoint: ${BASE_LOAD}" >&2; exit 2; }
CKPT_ARGS=(--hf-checkpoint "${LOYAL_MODEL_HF_CHECKPOINT}" --ref-load "${BASE_LOAD}" --load "${BASE_LOAD}" --save "${LOYAL_GSM8K_RL_SAVE}" --save-interval 20 --no-load-optim --no-load-rng --finetune)
ROLLOUT_ARGS=(--prompt-data "${PROMPTS}" --input-key prompt --label-key label --apply-chat-template --apply-chat-template-kwargs "${LOYAL_MODEL_CHAT_TEMPLATE_KWARGS}" --rollout-function-path slime.rollout.sglang_rollout.generate_rollout --num-rollout "${LOYAL_GSM8K_RL_NUM_ROLLOUT}" --rollout-batch-size "${LOYAL_GSM8K_RL_ROLLOUT_BATCH_SIZE}" --n-samples-per-prompt "${LOYAL_GSM8K_RL_SAMPLES_PER_PROMPT}" --rollout-max-response-len "${LOYAL_GSM8K_RL_MAX_RESPONSE_LEN}" --rollout-temperature 0.8 --global-batch-size "${LOYAL_GSM8K_RL_GLOBAL_BATCH_SIZE}")
RM_ARGS=(--custom-rm-path scripts.training.rewards.math_reward.math_reward_func --reward-key reward_value --log-reward-category reward_cat)
OPTIMIZER_ARGS=(--optimizer adam --lr "${LOYAL_GSM8K_RL_LEARNING_RATE}" --lr-decay-style cosine --min-lr 1e-7 --weight-decay 0.01 --adam-beta1 0.9 --adam-beta2 0.95 --clip-grad 1.0)
GRPO_ARGS=(--advantage-estimator grpo --eps-clip 0.2 --eps-clip-high 0.28)
PERF_ARGS=(--tensor-model-parallel-size 1 --recompute-granularity full --recompute-method uniform --recompute-num-layers 1 --use-dynamic-batch-size --max-tokens-per-gpu "${LOYAL_GSM8K_RL_MAX_TOKENS_PER_GPU}")
SGLANG_ARGS=(--rollout-num-gpus-per-engine 1 --sglang-mem-fraction-static 0.75 --sglang-server-concurrency 32)
MISC_ARGS=(--attention-dropout 0.0 --hidden-dropout 0.0 --no-gradient-accumulation-fusion --no-masked-softmax-fusion --accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32 --attention-backend flash --no-rope-fusion)
TRAIN_GPU_COUNT="${LOYAL_GSM8K_RL_TRAIN_GPU_COUNT}"; ROLLOUT_GPU_COUNT="${LOYAL_GSM8K_RL_ROLLOUT_GPU_COUNT}"; RAY_GPU_COUNT="${LOYAL_GSM8K_RL_RAY_NUM_GPUS}"
source "${SCRIPT_DIR}/submit_training.sh"
