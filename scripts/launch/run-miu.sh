#!/usr/bin/env bash
# MIU training recipe. Keep data, reward paths, and hyperparameters together
# here, in the same array-based style as SLIME's scripts/run-glm4-9B.sh.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/env.sh"
require_fixed_training_evaluators
MECHANISM=miu
SLIME_ROOT="${SLIME_ROOT:-${PROJECT_ROOT}/slime}"
DATA_ROOT="${LOYAL_DATA_ROOT:-${PROJECT_ROOT}/artifacts/slime/MIU}"
source "${SCRIPT_DIR}/model_profiles.sh"

# MIU recipe defaults.  A versioned experiment config may override these
# values, but they do not live in .env: that file is reserved for services and
# host-machine allocation.
# Four-GPU host topology: two actor/train GPUs and two single-GPU rollout
# engines. Docker renumbers the selected host devices to 0--3 in the container.
: "${LOYAL_MIU_TRAIN_GPU_COUNT:=2}"
: "${LOYAL_MIU_ROLLOUT_GPU_COUNT:=2}"
: "${LOYAL_MIU_RAY_NUM_GPUS:=4}"
: "${LOYAL_MIU_SGLANG_MEM_FRACTION_STATIC:=0.85}"
# Two rollout engines serve this four-GPU recipe.  Cap the router at 64
# requests per engine-equivalent so remote MIU judging cannot accumulate an
# unbounded backlog while an earlier group is being repaired.
: "${LOYAL_MIU_SGLANG_SERVER_CONCURRENCY:=64}"
: "${LOYAL_MIU_SAMPLES_PER_PROMPT:=8}"
: "${LOYAL_MIU_ROLLOUT_BATCH_SIZE:=64}"
: "${LOYAL_MIU_GLOBAL_BATCH_SIZE:=512}"
: "${LOYAL_MIU_LEARNING_RATE:=5e-7}"
: "${LOYAL_MIU_KL_LOSS_COEF:=0.01}"
: "${LOYAL_MIU_ENTROPY_COEF:=0.001}"
: "${LOYAL_MIU_CLIP_GRAD:=1.0}"
: "${LOYAL_MIU_EPS_CLIP:=0.2}"
: "${LOYAL_MIU_EPS_CLIP_HIGH:=0.28}"
: "${LOYAL_MIU_MAX_RESPONSE_LEN:=1024}"
: "${LOYAL_MIU_MAX_TOKENS_PER_GPU:=4096}"
: "${LOYAL_MIU_SAVE_INTERVAL:=1}"
: "${LOYAL_MIU_NUM_EPOCH:=4}"
: "${LOYAL_MIU_EVAL_INTERVAL:=20}"
: "${LOYAL_MIU_SAVE_RETAIN_INTERVAL:=1000000}"
: "${LOYAL_MIU_CANDIDATE_RESAMPLE_ATTEMPTS:=0}"
: "${LOYAL_MIU_ZERO_STD_GROUP_RESAMPLE_ATTEMPTS:=1}"
# At most two 8-candidate groups are judged at once: 16 MIU faithfulness
# requests, safely below the service limit and matched to the two engines.
: "${LOYAL_MIU_GROUP_RM_MAX_ATTEMPTS:=1}"
: "${LOYAL_MIU_GROUP_RM_MAX_CONCURRENT:=2}"
: "${LOYAL_MIU_FAILURE_LOG:=${PROJECT_ROOT}/artifacts/diagnostics/miu_groups_gpt54.jsonl}"
: "${LOYAL_USE_WANDB:=1}"
: "${LOYAL_WANDB_PROJECT:=loyal-agent}"
: "${LOYAL_WANDB_GROUP:=miu-${LOYAL_BASE_MODEL}-grpo}"
: "${LOYAL_WANDB_MODE:=online}"
export LOYAL_MIU_TRAIN_GPU_COUNT LOYAL_MIU_ROLLOUT_GPU_COUNT LOYAL_MIU_RAY_NUM_GPUS
export LOYAL_MIU_SGLANG_MEM_FRACTION_STATIC LOYAL_MIU_SGLANG_SERVER_CONCURRENCY
export LOYAL_MIU_SAMPLES_PER_PROMPT LOYAL_MIU_ROLLOUT_BATCH_SIZE LOYAL_MIU_GLOBAL_BATCH_SIZE
export LOYAL_MIU_LEARNING_RATE LOYAL_MIU_KL_LOSS_COEF LOYAL_MIU_ENTROPY_COEF LOYAL_MIU_CLIP_GRAD
export LOYAL_MIU_EPS_CLIP LOYAL_MIU_EPS_CLIP_HIGH LOYAL_MIU_MAX_RESPONSE_LEN LOYAL_MIU_MAX_TOKENS_PER_GPU
export LOYAL_MIU_SAVE_INTERVAL LOYAL_MIU_NUM_EPOCH LOYAL_MIU_EVAL_INTERVAL LOYAL_MIU_SAVE_RETAIN_INTERVAL
export LOYAL_MIU_CANDIDATE_RESAMPLE_ATTEMPTS LOYAL_MIU_ZERO_STD_GROUP_RESAMPLE_ATTEMPTS
export LOYAL_MIU_GROUP_RM_MAX_ATTEMPTS LOYAL_MIU_GROUP_RM_MAX_CONCURRENT LOYAL_MIU_FAILURE_LOG
export LOYAL_USE_WANDB LOYAL_WANDB_PROJECT LOYAL_WANDB_GROUP LOYAL_WANDB_MODE

: "${LOYAL_MIU_JUDGE_BASE_URL:?set the MIU judge endpoint in .env}"
: "${LOYAL_MIU_JUDGE_MODEL:?set the MIU judge model in .env}"
export LOYAL_MIU_JUDGE_BASE_URL LOYAL_MIU_JUDGE_MODEL
[[ -z "${LOYAL_MIU_JUDGE_API_KEY:-}" ]] || export LOYAL_MIU_JUDGE_API_KEY
export LOYAL_MIU_TRAIN_RECORDS="${LOYAL_MIU_TRAIN_RECORDS:-${PROJECT_ROOT}/miu/data/dataset/MIU/train.jsonl}"
export LOYAL_MIU_VAL_RECORDS="${LOYAL_MIU_VAL_RECORDS:-${PROJECT_ROOT}/miu/data/dataset/MIU/val.jsonl}"
export LOYAL_MIU_RECORDS="${LOYAL_MIU_RECORDS:-${LOYAL_MIU_TRAIN_RECORDS}:${LOYAL_MIU_VAL_RECORDS}}"
python3 "${PROJECT_ROOT}/scripts/training/preflight.py" miu --runtime
python3 "${PROJECT_ROOT}/scripts/data/prepare_slime.py" miu --source "${LOYAL_MIU_TRAIN_RECORDS}" --output "${DATA_ROOT}/train.jsonl"
python3 "${PROJECT_ROOT}/scripts/data/prepare_slime.py" miu --source "${LOYAL_MIU_VAL_RECORDS}" --output "${DATA_ROOT}/val.jsonl"

CKPT_ARGS=(
  --hf-checkpoint "${LOYAL_MODEL_HF_CHECKPOINT}" --ref-load "${LOYAL_MODEL_REF_LOAD}"
  --load "${LOYAL_MIU_LOAD:-/root/${LOYAL_BASE_MODEL}_miu_slime}" --save "${LOYAL_MIU_SAVE:-/root/${LOYAL_BASE_MODEL}_miu_slime}"
  --save-interval "${LOYAL_MIU_SAVE_INTERVAL}" --save-retain-interval "${LOYAL_MIU_SAVE_RETAIN_INTERVAL}"
)
ROLLOUT_ARGS=(
  --prompt-data "${DATA_ROOT}/train.jsonl" --input-key messages --label-key record_id --apply-chat-template
  --apply-chat-template-kwargs "${LOYAL_MODEL_CHAT_TEMPLATE_KWARGS}" --rollout-function-path scripts.training.rollout.miu.generate_rollout --rollout-shuffle
  --rollout-batch-size "${LOYAL_MIU_ROLLOUT_BATCH_SIZE}" --n-samples-per-prompt "${LOYAL_MIU_SAMPLES_PER_PROMPT}"
  --rollout-max-response-len "${LOYAL_MIU_MAX_RESPONSE_LEN}" --rollout-temperature 0.8 --global-batch-size "${LOYAL_MIU_GLOBAL_BATCH_SIZE}"
  --balance-data --rollout-seed "${LOYAL_ROLLOUT_SEED:-42}"
)
[[ -z "${LOYAL_MIU_NUM_ROLLOUT:-}" ]] && ROLLOUT_ARGS+=(--num-epoch "${LOYAL_MIU_NUM_EPOCH}") || ROLLOUT_ARGS+=(--num-rollout "${LOYAL_MIU_NUM_ROLLOUT}")
RM_ARGS=(
  --custom-rm-path scripts.training.rewards.slime.miu_reward_func --custom-reward-post-process-path scripts.training.rewards.slime.miu_post_process_rewards
  --reward-key reward_value --eval-reward-key reward_value --group-rm
  --dynamic-sampling-filter-path scripts.training.rewards.filters.keep_eligible_nonzero_std --log-reward-category reward_category
)
OPTIMIZER_ARGS=(--optimizer adam --lr "${LOYAL_MIU_LEARNING_RATE}" --lr-decay-style constant --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.98 --clip-grad "${LOYAL_MIU_CLIP_GRAD}")
GRPO_ARGS=(--advantage-estimator grpo --use-kl-loss --kl-loss-coef "${LOYAL_MIU_KL_LOSS_COEF}" --kl-loss-type low_var_kl --entropy-coef "${LOYAL_MIU_ENTROPY_COEF}" --eps-clip "${LOYAL_MIU_EPS_CLIP}" --eps-clip-high "${LOYAL_MIU_EPS_CLIP_HIGH}")
PERF_ARGS=(--tensor-model-parallel-size 1 --recompute-granularity full --recompute-method uniform --recompute-num-layers 1 --use-dynamic-batch-size --max-tokens-per-gpu "${LOYAL_MIU_MAX_TOKENS_PER_GPU}")
SGLANG_ARGS=(--rollout-num-gpus-per-engine 1 --sglang-mem-fraction-static "${LOYAL_MIU_SGLANG_MEM_FRACTION_STATIC}" --sglang-server-concurrency "${LOYAL_MIU_SGLANG_SERVER_CONCURRENCY}")
EVAL_ARGS=()
if [[ "${LOYAL_MIU_DISABLE_EVAL:-0}" != "1" ]]; then
  EVAL_ARGS=(--eval-interval "${LOYAL_MIU_EVAL_INTERVAL}" --eval-prompt-data miu_val "${DATA_ROOT}/val.jsonl" --eval-input-key messages --eval-label-key record_id --n-samples-per-eval-prompt 1 --eval-temperature 0.0 --eval-max-response-len "${LOYAL_MIU_MAX_RESPONSE_LEN}")
  [[ "${LOYAL_MIU_SKIP_INITIAL_EVAL:-1}" != "1" ]] || EVAL_ARGS+=(--skip-initial-eval)
fi
WANDB_ARGS=()
if [[ "${LOYAL_USE_WANDB:-0}" == "1" ]]; then [[ -z "${WANDB_API_KEY:-}" ]] || export WANDB_API_KEY; WANDB_ARGS=(--use-wandb --wandb-project "${LOYAL_WANDB_PROJECT:-loyal-agent}" --wandb-group "${LOYAL_WANDB_GROUP:-miu-${LOYAL_BASE_MODEL}-grpo}" --wandb-mode "${LOYAL_WANDB_MODE:-online}"); fi
MISC_ARGS=(--attention-dropout 0.0 --hidden-dropout 0.0 --accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32 --attention-backend flash)
TRAIN_GPU_COUNT="${LOYAL_MIU_TRAIN_GPU_COUNT}"; ROLLOUT_GPU_COUNT="${LOYAL_MIU_ROLLOUT_GPU_COUNT}"; RAY_GPU_COUNT="${LOYAL_MIU_RAY_NUM_GPUS}"
RUNTIME_EXTRA=',"PYTORCH_CUDA_ALLOC_CONF":"expandable_segments:True"'
export LOYAL_REWARD_FAILURE_LOG="${LOYAL_MIU_FAILURE_LOG}"
[[ -z "${LOYAL_ADAPTIVE_SIGNAL_LOG:-}" ]] || export LOYAL_ADAPTIVE_SIGNAL_LOG LOYAL_ADAPTIVE_SIGNAL_MECHANISM=miu
source "${SCRIPT_DIR}/submit_training.sh"
