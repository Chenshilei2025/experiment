#!/usr/bin/env bash
# EIL training recipe. Keep data, reward paths, and hyperparameters together
# here, in the same array-based style as SLIME's scripts/run-glm4-9B.sh.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/env.sh"
require_fixed_training_evaluators
MECHANISM=eil
SLIME_ROOT="${SLIME_ROOT:-${PROJECT_ROOT}/slime}"
DATA_ROOT="${LOYAL_DATA_ROOT:-${PROJECT_ROOT}/artifacts/slime/EIL}"
source "${SCRIPT_DIR}/model_profiles.sh"

# EIL recipe defaults. Endpoint, evaluator, and credential settings remain in
# .env; this block is the reproducible training/reward recipe.
: "${LOYAL_EIL_TRAIN_GPU_COUNT:=2}"
# Four-GPU host topology: two actor/train GPUs and two single-GPU rollout
# engines. Docker renumbers the selected host devices to 0--3 in the container.
: "${LOYAL_EIL_ROLLOUT_GPU_COUNT:=2}"
: "${LOYAL_EIL_RAY_NUM_GPUS:=4}"
# EIL produces direct answers on dedicated rollout GPUs.  Keep a large KV
# cache, as in the prior throughput recipe, while retaining room for graphs.
: "${LOYAL_EIL_SGLANG_MEM_FRACTION_STATIC:=0.70}"
# Rotate one temperature per rollout group; all candidates in that GRPO group
# see the same adversary strength.
: "${LOYAL_EIL_ADVERSARY_TEMPERATURES:=0.3,0.6,0.8,1.0}"
: "${LOYAL_EIL_RM_MAX_CONCURRENT:=4}"
: "${LOYAL_EIL_EVAL_RM_MAX_CONCURRENT:=1}"
: "${LOYAL_EIL_GROUP_RM_MAX_CONCURRENT:=2}"
: "${LOYAL_EIL_GROUP_RM_MAX_ATTEMPTS:=2}"
# A direct EIL answer does not require a private trace.  1,024 tokens bounds
# KV-cache use and external adversary/judge latency on two rollout engines.
: "${LOYAL_EIL_MAX_RESPONSE_LEN:=1024}"
: "${LOYAL_EIL_MAX_TOKENS_PER_GPU:=12288}"
: "${LOYAL_EIL_SAMPLES_PER_PROMPT:=8}"
# Two rollout engines generate 32 prompt groups × 8 candidates = 256 scored
# replies per update.  This avoids queuing the old four-engine 512-reply
# workload behind remote adversary and judge calls.
: "${LOYAL_EIL_ROLLOUT_BATCH_SIZE:=32}"
: "${LOYAL_EIL_GLOBAL_BATCH_SIZE:=256}"
: "${LOYAL_EIL_SAVE_INTERVAL:=20}"
: "${LOYAL_EIL_NUM_EPOCH:=10}"
: "${LOYAL_EIL_EVAL_INTERVAL:=20}"
: "${LOYAL_EIL_FAILURE_LOG:=${PROJECT_ROOT}/artifacts/diagnostics/eil_groups.jsonl}"
: "${LOYAL_USE_WANDB:=1}"
: "${LOYAL_WANDB_PROJECT:=loyal-agent}"
: "${LOYAL_WANDB_GROUP:=eil-${LOYAL_BASE_MODEL}-grpo}"
: "${LOYAL_WANDB_MODE:=online}"
export LOYAL_EIL_TRAIN_GPU_COUNT LOYAL_EIL_ROLLOUT_GPU_COUNT LOYAL_EIL_RAY_NUM_GPUS
export LOYAL_EIL_SGLANG_MEM_FRACTION_STATIC LOYAL_EIL_ADVERSARY_TEMPERATURES LOYAL_EIL_RM_MAX_CONCURRENT LOYAL_EIL_EVAL_RM_MAX_CONCURRENT
export LOYAL_EIL_GROUP_RM_MAX_CONCURRENT LOYAL_EIL_GROUP_RM_MAX_ATTEMPTS
export LOYAL_EIL_MAX_RESPONSE_LEN LOYAL_EIL_MAX_TOKENS_PER_GPU LOYAL_EIL_SAMPLES_PER_PROMPT LOYAL_EIL_ROLLOUT_BATCH_SIZE LOYAL_EIL_GLOBAL_BATCH_SIZE
export LOYAL_EIL_SAVE_INTERVAL LOYAL_EIL_NUM_EPOCH LOYAL_EIL_EVAL_INTERVAL LOYAL_EIL_FAILURE_LOG
export LOYAL_USE_WANDB LOYAL_WANDB_PROJECT LOYAL_WANDB_GROUP LOYAL_WANDB_MODE

: "${LOYAL_EIL_ADVERSARY_BASE_URL:?set the EIL adversary endpoint in .env}"
: "${LOYAL_EIL_ADVERSARY_MODEL:?set the EIL adversary model in .env}"
: "${LOYAL_EIL_JUDGE_BASE_URL:?set the EIL judge endpoint in .env}"
: "${LOYAL_EIL_JUDGE_MODEL:?set the EIL judge model in .env}"
export LOYAL_EIL_ADVERSARY_BASE_URL LOYAL_EIL_ADVERSARY_MODEL LOYAL_EIL_JUDGE_BASE_URL LOYAL_EIL_JUDGE_MODEL
for secret in LOYAL_EIL_ADVERSARY_API_KEY LOYAL_EIL_ADVERSARY_API_KEYS LOYAL_EIL_JUDGE_API_KEY LOYAL_EIL_JUDGE_API_KEYS; do [[ -z "${!secret:-}" ]] || export "${secret}"; done
export LOYAL_EIL_TRAIN_RECORDS="${LOYAL_EIL_TRAIN_RECORDS:-${PROJECT_ROOT}/eil/data/dataset/EIL/train.jsonl}"
export LOYAL_EIL_VAL_RECORDS="${LOYAL_EIL_VAL_RECORDS:-${PROJECT_ROOT}/eil/data/dataset/EIL/val.jsonl}"
export LOYAL_EIL_RECORDS="${LOYAL_EIL_RECORDS:-${LOYAL_EIL_TRAIN_RECORDS}:${LOYAL_EIL_VAL_RECORDS}}"
python3 "${PROJECT_ROOT}/scripts/training/preflight.py" eil --runtime
export LOYAL_EIL_COVERAGE_CACHE="${LOYAL_EIL_COVERAGE_CACHE:-${PROJECT_ROOT}/artifacts/cache/eil_coverage}"
export LOYAL_EIL_UTILITY_CACHE="${LOYAL_EIL_UTILITY_CACHE:-${PROJECT_ROOT}/artifacts/cache/eil_utility}"
python3 "${PROJECT_ROOT}/scripts/data/prepare_slime.py" eil --source "${LOYAL_EIL_TRAIN_RECORDS}" --output "${DATA_ROOT}/train.jsonl"
python3 "${PROJECT_ROOT}/scripts/data/prepare_slime.py" eil --source "${LOYAL_EIL_VAL_RECORDS}" --output "${DATA_ROOT}/val.jsonl"

CKPT_ARGS=(--hf-checkpoint "${LOYAL_MODEL_HF_CHECKPOINT}" --ref-load "${LOYAL_MODEL_REF_LOAD}" --load "${LOYAL_EIL_LOAD:-/root/${LOYAL_BASE_MODEL}_eil_slime}" --save "${LOYAL_EIL_SAVE:-/root/${LOYAL_BASE_MODEL}_eil_slime}" --save-interval "${LOYAL_EIL_SAVE_INTERVAL}")
ROLLOUT_ARGS=(
  --prompt-data "${DATA_ROOT}/train.jsonl" --input-key messages --label-key record_id --apply-chat-template --apply-chat-template-kwargs "${LOYAL_MODEL_CHAT_TEMPLATE_KWARGS}" --rollout-shuffle
  --rollout-batch-size "${LOYAL_EIL_ROLLOUT_BATCH_SIZE}" --n-samples-per-prompt "${LOYAL_EIL_SAMPLES_PER_PROMPT}" --rollout-max-response-len "${LOYAL_EIL_MAX_RESPONSE_LEN}" --rollout-temperature 0.8 --global-batch-size "${LOYAL_EIL_GLOBAL_BATCH_SIZE}" --balance-data --rollout-seed "${LOYAL_ROLLOUT_SEED:-42}"
)
[[ -z "${LOYAL_EIL_NUM_ROLLOUT:-}" ]] && ROLLOUT_ARGS+=(--num-epoch "${LOYAL_EIL_NUM_EPOCH}") || ROLLOUT_ARGS+=(--num-rollout "${LOYAL_EIL_NUM_ROLLOUT}")
RM_ARGS=(--custom-rm-path scripts.training.rewards.slime.eil_reward_func --custom-reward-post-process-path scripts.training.rewards.slime.eil_post_process_rewards --reward-key reward_value --eval-reward-key reward_value --group-rm --dynamic-sampling-filter-path scripts.training.rewards.filters.keep_eligible_nonzero_std --log-reward-category reward_category)
OPTIMIZER_ARGS=(--optimizer adam --lr 1e-6 --lr-decay-style constant --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.98)
GRPO_ARGS=(--advantage-estimator grpo --use-kl-loss --kl-loss-coef 0.00 --kl-loss-type low_var_kl --entropy-coef 0.00 --eps-clip 0.2 --eps-clip-high 0.28)
PERF_ARGS=(--tensor-model-parallel-size 1 --recompute-granularity full --recompute-method uniform --recompute-num-layers 1 --use-dynamic-batch-size --max-tokens-per-gpu "${LOYAL_EIL_MAX_TOKENS_PER_GPU}")
# With two single-GPU engines, cap router fan-out so long EIL generations do
# not monopolize the server while the external reward path is pending.
SGLANG_ARGS=(--rollout-num-gpus-per-engine 1 --sglang-mem-fraction-static "${LOYAL_EIL_SGLANG_MEM_FRACTION_STATIC}" --sglang-server-concurrency 64)
EVAL_ARGS=()
EVAL_ARGS=(--eval-interval "${LOYAL_EIL_EVAL_INTERVAL}" --eval-prompt-data eil_val "${DATA_ROOT}/val.jsonl" --eval-input-key messages --eval-label-key record_id --n-samples-per-eval-prompt 1 --eval-temperature 0.0 --eval-max-response-len "${LOYAL_EIL_MAX_RESPONSE_LEN}")
WANDB_ARGS=()
if [[ "${LOYAL_USE_WANDB:-0}" == "1" ]]; then [[ -z "${WANDB_API_KEY:-}" ]] || export WANDB_API_KEY; WANDB_ARGS=(--use-wandb --wandb-project "${LOYAL_WANDB_PROJECT:-loyal-agent}" --wandb-group "${LOYAL_WANDB_GROUP:-eil-${LOYAL_BASE_MODEL}-grpo}" --wandb-mode "${LOYAL_WANDB_MODE:-online}"); fi
MISC_ARGS=(--attention-dropout 0.0 --hidden-dropout 0.0 --accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32 --attention-backend flash)
TRAIN_GPU_COUNT="${LOYAL_EIL_TRAIN_GPU_COUNT}"; ROLLOUT_GPU_COUNT="${LOYAL_EIL_ROLLOUT_GPU_COUNT}"; RAY_GPU_COUNT="${LOYAL_EIL_RAY_NUM_GPUS}"
RUNTIME_EXTRA=''
export LOYAL_REWARD_FAILURE_LOG="${LOYAL_EIL_FAILURE_LOG}"
[[ -z "${LOYAL_ADAPTIVE_SIGNAL_LOG:-}" ]] || export LOYAL_ADAPTIVE_SIGNAL_LOG LOYAL_ADAPTIVE_SIGNAL_MECHANISM=eil
source "${SCRIPT_DIR}/submit_training.sh"
