#!/usr/bin/env bash
# Joint MIU/EIL GRPO recipe: quota-balanced single-task prompt batches.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
source "${SCRIPT_DIR}/env.sh"
if [[ "${LOYAL_ALLOW_EXPERIMENTAL_ADVERSARY:-0}" == 1 ]]; then
  : "${LOYAL_EXPERIMENTAL_ADVERSARY_PROFILE:?set a profile name such as ADV_QWEN8B}"
  [[ "${LOYAL_EXPERIMENTAL_ADVERSARY_PROFILE}" =~ ^[A-Z][A-Z0-9_]*$ ]] || { echo 'invalid experimental adversary profile' >&2; exit 2; }
  profile="${LOYAL_EXPERIMENTAL_ADVERSARY_PROFILE}"
  model_var="LOYAL_${profile}_MODEL"; url_var="LOYAL_${profile}_BASE_URL"; key_var="LOYAL_${profile}_API_KEY"
  : "${!model_var:?set ${model_var} in .env}"
  : "${!url_var:?set ${url_var} in .env}"
  export LOYAL_EIL_ADVERSARY_MODEL="${!model_var}"
  export LOYAL_EIL_ADVERSARY_BASE_URL="${!url_var}"
  [[ -z "${!key_var:-}" ]] || export LOYAL_EIL_ADVERSARY_API_KEY="${!key_var}"
fi
require_fixed_training_evaluators
MECHANISM=mixed
SLIME_ROOT="${SLIME_ROOT:-${PROJECT_ROOT}/slime}"
DATA_ROOT="${LOYAL_DATA_ROOT:-${PROJECT_ROOT}/artifacts/slime/MIXED}"
source "${SCRIPT_DIR}/model_profiles.sh"

: "${LOYAL_MIXED_TRAIN_GPU_COUNT:=2}"
: "${LOYAL_MIXED_ROLLOUT_GPU_COUNT:=2}"
: "${LOYAL_MIXED_RAY_NUM_GPUS:=4}"
: "${LOYAL_MIXED_SGLANG_MEM_FRACTION_STATIC:=0.70}"
: "${LOYAL_MIXED_SGLANG_SERVER_CONCURRENCY:=64}"
: "${LOYAL_MIXED_EIL_BATCH_FRACTION:=${LOYAL_MIXED_EIL_PROBABILITY:-0.5}}"
: "${LOYAL_MIXED_SAMPLES_PER_PROMPT:=8}"
: "${LOYAL_MIXED_ROLLOUT_BATCH_SIZE:=8}"
: "${LOYAL_MIXED_GLOBAL_BATCH_SIZE:=64}"
: "${LOYAL_MIXED_MAX_RESPONSE_LEN:=1024}"
: "${LOYAL_MIXED_MAX_TOKENS_PER_GPU:=4096}"
: "${LOYAL_MIXED_SAVE_INTERVAL:=20}"
: "${LOYAL_MIXED_EVAL_INTERVAL:=20}"
: "${LOYAL_MIXED_NUM_ROLLOUT:?set the total mixed rollout budget}"
: "${LOYAL_MIXED_TRAIN_RECORDS:?set the generated mixed train records path}"
: "${LOYAL_MIU_RECORDS:?set MIU reward records}"
: "${LOYAL_EIL_RECORDS:?set EIL reward records}"
: "${LOYAL_MIXED_LOAD:=/root/${LOYAL_BASE_MODEL}_mixed_slime}"
: "${LOYAL_MIXED_SAVE:=/root/${LOYAL_BASE_MODEL}_mixed_slime}"
[[ "${LOYAL_MIXED_TRAIN_GPU_COUNT}" -gt 0 && "${LOYAL_MIXED_ROLLOUT_GPU_COUNT}" -gt 0 ]] || { echo 'mixed GPU counts must be positive' >&2; exit 2; }
[[ $((LOYAL_MIXED_TRAIN_GPU_COUNT + LOYAL_MIXED_ROLLOUT_GPU_COUNT)) -eq "${LOYAL_MIXED_RAY_NUM_GPUS}" ]] || { echo 'mixed Ray GPU count must equal train plus rollout counts' >&2; exit 2; }
# If supplied by the host launcher, CUDA_VISIBLE_DEVICES already puts the
# training GPUs first.  submit_training.sh preserves that order for Ray.

for required in LOYAL_MIU_JUDGE_BASE_URL LOYAL_MIU_JUDGE_MODEL LOYAL_EIL_ADVERSARY_BASE_URL LOYAL_EIL_ADVERSARY_MODEL LOYAL_EIL_JUDGE_BASE_URL LOYAL_EIL_JUDGE_MODEL; do
  [[ -n "${!required:-}" ]] || { echo "missing ${required}" >&2; exit 2; }
done
export LOYAL_MIU_JUDGE_BASE_URL LOYAL_MIU_JUDGE_MODEL LOYAL_EIL_ADVERSARY_BASE_URL LOYAL_EIL_ADVERSARY_MODEL LOYAL_EIL_JUDGE_BASE_URL LOYAL_EIL_JUDGE_MODEL
for secret in LOYAL_MIU_JUDGE_API_KEY LOYAL_EIL_ADVERSARY_API_KEY LOYAL_EIL_ADVERSARY_API_KEYS LOYAL_EIL_JUDGE_API_KEY LOYAL_EIL_JUDGE_API_KEYS; do [[ -z "${!secret:-}" ]] || export "${secret}"; done
export LOYAL_JUDGE_CIRCUIT_ACTION="${LOYAL_JUDGE_CIRCUIT_ACTION:-soft_keep}"
export LOYAL_JUDGE_CIRCUIT_FAILURE_THRESHOLD="${LOYAL_JUDGE_CIRCUIT_FAILURE_THRESHOLD:-100}"
export LOYAL_MIU_GROUP_RM_MAX_ATTEMPTS="${LOYAL_MIU_GROUP_RM_MAX_ATTEMPTS:-3}"
export LOYAL_MIU_GROUP_RM_RETRY_INITIAL_BACKOFF_SECONDS="${LOYAL_MIU_GROUP_RM_RETRY_INITIAL_BACKOFF_SECONDS:-3}"
export LOYAL_MIU_GROUP_RM_MAX_CONCURRENT="${LOYAL_MIU_GROUP_RM_MAX_CONCURRENT:-2}"
export LOYAL_MIU_ZERO_STD_GROUP_RESAMPLE_ATTEMPTS="${LOYAL_MIU_ZERO_STD_GROUP_RESAMPLE_ATTEMPTS:-2}"
export LOYAL_EIL_GROUP_RM_MAX_ATTEMPTS="${LOYAL_EIL_GROUP_RM_MAX_ATTEMPTS:-4}"
export LOYAL_EIL_GROUP_RM_RETRY_INITIAL_BACKOFF_SECONDS="${LOYAL_EIL_GROUP_RM_RETRY_INITIAL_BACKOFF_SECONDS:-5}"
export LOYAL_EIL_GROUP_RM_MAX_CONCURRENT="${LOYAL_EIL_GROUP_RM_MAX_CONCURRENT:-1}"
export LOYAL_EIL_RM_MAX_CONCURRENT="${LOYAL_EIL_RM_MAX_CONCURRENT:-2}"
export LOYAL_EIL_EVAL_RM_MAX_CONCURRENT="${LOYAL_EIL_EVAL_RM_MAX_CONCURRENT:-1}"
export LOYAL_EIL_ADVERSARY_TEMPERATURES="${LOYAL_EIL_ADVERSARY_TEMPERATURES:-0.3,0.6,0.8,1.0}"
export LOYAL_REWARD_FAILURE_LOG="${LOYAL_MIXED_FAILURE_LOG:-${PROJECT_ROOT}/artifacts/diagnostics/mixed_groups.jsonl}"
export LOYAL_MIXED_EIL_BATCH_FRACTION

MIXED_EVAL_ROOT="${DATA_ROOT}/eval"
if [[ "${LOYAL_MIXED_ENABLE_EVAL:-0}" == "1" ]]; then
  mkdir -p "${MIXED_EVAL_ROOT}"
  python3 "${PROJECT_ROOT}/scripts/data/prepare_slime.py" miu \
    --source "${PROJECT_ROOT}/miu/data/dataset/MIU-v2/val.jsonl" \
    --output "${MIXED_EVAL_ROOT}/miu_val.jsonl" --namespace-record-id
  python3 "${PROJECT_ROOT}/scripts/data/prepare_slime.py" eil \
    --source "${PROJECT_ROOT}/eil/data/dataset/EIL-v2/val.jsonl" \
    --output "${MIXED_EVAL_ROOT}/eil_val.jsonl" --namespace-record-id
fi

CKPT_ARGS=(--hf-checkpoint "${LOYAL_MODEL_HF_CHECKPOINT}" --ref-load "${LOYAL_MODEL_REF_LOAD}" --load "${LOYAL_MIXED_LOAD}" --save "${LOYAL_MIXED_SAVE}" --save-interval "${LOYAL_MIXED_SAVE_INTERVAL}")
ROLLOUT_ARGS=(--prompt-data "${LOYAL_MIXED_TRAIN_RECORDS}" --input-key messages --label-key record_id --apply-chat-template --apply-chat-template-kwargs "${LOYAL_MODEL_CHAT_TEMPLATE_KWARGS}" --rollout-function-path scripts.training.rollout.mixed.generate_rollout --rollout-shuffle --rollout-batch-size "${LOYAL_MIXED_ROLLOUT_BATCH_SIZE}" --n-samples-per-prompt "${LOYAL_MIXED_SAMPLES_PER_PROMPT}" --rollout-max-response-len "${LOYAL_MIXED_MAX_RESPONSE_LEN}" --rollout-temperature 0.8 --global-batch-size "${LOYAL_MIXED_GLOBAL_BATCH_SIZE}" --balance-data --rollout-seed "${LOYAL_ROLLOUT_SEED:-42}" --num-rollout "${LOYAL_MIXED_NUM_ROLLOUT}")
RM_ARGS=(--custom-rm-path scripts.training.rewards.slime.mixed_reward_func --custom-reward-post-process-path scripts.training.rewards.slime.mixed_post_process_rewards --reward-key reward_value --eval-reward-key reward_value --group-rm --dynamic-sampling-filter-path scripts.training.rewards.filters.keep_eligible_nonzero_std --log-reward-category reward_category)
OPTIMIZER_ARGS=(--optimizer adam --lr "${LOYAL_MIXED_LEARNING_RATE:-7.5e-7}" --lr-decay-style constant --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.98 --clip-grad 1.0)
if [[ "${LOYAL_MIXED_OPTIMIZER_CPU_OFFLOAD:-0}" == "1" ]]; then
  # A single training GPU cannot hold Adam states, model buffers, and the
  # initial Megatron→SGLang weight-transfer workspace simultaneously.
  OPTIMIZER_ARGS+=(--optimizer-cpu-offload --use-precision-aware-optimizer)
fi
GRPO_ARGS=(--advantage-estimator grpo --use-kl-loss --kl-loss-coef "${LOYAL_MIXED_KL_LOSS_COEF:-0.01}" --kl-loss-type low_var_kl --entropy-coef "${LOYAL_MIXED_ENTROPY_COEF:-0.001}" --eps-clip 0.2 --eps-clip-high 0.28)
# Per-group standard-deviation normalization can turn a very small reward
# difference into a full-strength GRPO update.  It is configurable for reward
# ablations; MIU's near-tied groups use the conservative setting below.
if [[ "${LOYAL_MIXED_GRPO_STD_NORMALIZATION:-1}" != "1" ]]; then
  GRPO_ARGS+=(--disable-grpo-std-normalization)
fi
PERF_ARGS=(--tensor-model-parallel-size 1 --recompute-granularity full --recompute-method uniform --recompute-num-layers 1 --use-dynamic-batch-size --max-tokens-per-gpu "${LOYAL_MIXED_MAX_TOKENS_PER_GPU}" --update-weight-buffer-size "${LOYAL_MIXED_UPDATE_WEIGHT_BUFFER_SIZE:-536870912}")
SGLANG_ARGS=(--rollout-num-gpus-per-engine 1 --sglang-mem-fraction-static "${LOYAL_MIXED_SGLANG_MEM_FRACTION_STATIC}" --sglang-server-concurrency "${LOYAL_MIXED_SGLANG_SERVER_CONCURRENCY}")
EVAL_ARGS=()
# Optional held-out validation for mixed runs.  This uses the same mixed
# rollout/reward path on the MIU and EIL validation prompt sets; metrics are
# emitted by SLIME under eval/* and therefore appear alongside train/* in
# W&B.  It is opt-in because validation adds judge/API latency.
if [[ "${LOYAL_MIXED_ENABLE_EVAL:-0}" == "1" ]]; then
  : "${LOYAL_MIXED_EVAL_INTERVAL:=20}"
  : "${LOYAL_MIXED_EVAL_MAX_RESPONSE_LEN:=${LOYAL_MIXED_MAX_RESPONSE_LEN}}"
  EVAL_ARGS=(
    --eval-interval "${LOYAL_MIXED_EVAL_INTERVAL}"
    # argparse treats --eval-prompt-data as one nargs='+' option; passing the
    # flag twice makes the later dataset replace the earlier one.
    --eval-prompt-data
      miu_val "${MIXED_EVAL_ROOT}/miu_val.jsonl"
      eil_val "${MIXED_EVAL_ROOT}/eil_val.jsonl"
    --eval-input-key messages --eval-label-key record_id
    --n-samples-per-eval-prompt 1 --eval-temperature 0.0
    --eval-max-response-len "${LOYAL_MIXED_EVAL_MAX_RESPONSE_LEN}"
  )
  [[ "${LOYAL_MIXED_SKIP_INITIAL_EVAL:-1}" != "1" ]] || EVAL_ARGS+=(--skip-initial-eval)
fi
WANDB_ARGS=()
if [[ "${LOYAL_USE_WANDB:-1}" == 1 ]]; then
  [[ -z "${WANDB_API_KEY:-}" ]] || export WANDB_API_KEY
  WANDB_ARGS=(--use-wandb --wandb-project "${LOYAL_WANDB_PROJECT:-loyal-agent}" --wandb-group "${LOYAL_WANDB_GROUP:-mixed-${LOYAL_BASE_MODEL}-grpo}" --wandb-mode "${LOYAL_WANDB_MODE:-online}")
  # The run ID is an argparse value, not merely a W&B environment variable.
  # Passing it explicitly makes both the primary and Ray secondary writers
  # attach to the historical curve after a checkpoint resume.
  if [[ -n "${LOYAL_WANDB_RUN_ID:-}" ]]; then
    WANDB_ARGS+=(--wandb-run-id "${LOYAL_WANDB_RUN_ID}")
  fi
fi
MISC_ARGS=(--attention-dropout 0.0 --hidden-dropout 0.0 --no-gradient-accumulation-fusion --no-masked-softmax-fusion --accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32 --attention-backend flash --no-rope-fusion)
# Make scheduler replacement explicit on the command line when a checkpoint
# resume intentionally changes LR.  The environment fallback in the backend
# protects programmatic callers; this flag makes the training invocation
# auditable and lets Megatron accept the new scheduler value while retaining
# optimizer moments.
if [[ "${LOYAL_OVERRIDE_OPT_PARAM_SCHEDULER:-0}" == "1" ]]; then
  MISC_ARGS+=(--override-opt-param-scheduler)
fi
# Force an explicit checkpoint iteration when resuming a batch-size-change
# experiment.  This avoids relying on a stale/incorrect tracker read.
if [[ -n "${LOYAL_MIXED_CKPT_STEP:-}" ]]; then
  MISC_ARGS+=(--ckpt-step "${LOYAL_MIXED_CKPT_STEP}")
fi
if [[ "${LOYAL_MIXED_NO_LOAD_OPTIM:-0}" == "1" ]]; then
  MISC_ARGS+=(--no-load-optim --no-load-rng)
fi
TRAIN_GPU_COUNT="${LOYAL_MIXED_TRAIN_GPU_COUNT}"; ROLLOUT_GPU_COUNT="${LOYAL_MIXED_ROLLOUT_GPU_COUNT}"; RAY_GPU_COUNT="${LOYAL_MIXED_RAY_NUM_GPUS}"
RUNTIME_EXTRA=',"PYTORCH_CUDA_ALLOC_CONF":"expandable_segments:True","LOYAL_JUDGE_CIRCUIT_ACTION":"'"${LOYAL_JUDGE_CIRCUIT_ACTION}"'","LOYAL_JUDGE_CIRCUIT_FAILURE_THRESHOLD":"'"${LOYAL_JUDGE_CIRCUIT_FAILURE_THRESHOLD}"'"'
source "${SCRIPT_DIR}/submit_training.sh"
