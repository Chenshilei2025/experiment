#!/usr/bin/env bash
set -euo pipefail

# EIL and MIU must run as independent jobs with independent checkpoints.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/env.sh"
SLIME_ROOT="${SLIME_ROOT:-${PROJECT_ROOT}/slime}"
DATA_ROOT="${LOYAL_DATA_ROOT:-${PROJECT_ROOT}/artifacts/slime/EIL}"

: "${LOYAL_EIL_ADVERSARY_BASE_URL:?set the EIL adversary endpoint}"
: "${LOYAL_EIL_ADVERSARY_MODEL:?set the EIL adversary model}"
: "${LOYAL_EIL_JUDGE_BASE_URL:?set the EIL judge endpoint}"
: "${LOYAL_EIL_JUDGE_MODEL:?set the EIL judge model}"
python3 "${PROJECT_ROOT}/scripts/training/preflight.py" eil --runtime

# Start a fresh Ray daemon after these exports so API credentials are inherited
# by workers without being included in the submitted command or runtime JSON.
export LOYAL_EIL_ADVERSARY_BASE_URL LOYAL_EIL_ADVERSARY_MODEL LOYAL_EIL_JUDGE_BASE_URL LOYAL_EIL_JUDGE_MODEL
if [[ -n "${LOYAL_EIL_ADVERSARY_API_KEY:-}" ]]; then export LOYAL_EIL_ADVERSARY_API_KEY; fi
if [[ -n "${LOYAL_EIL_ADVERSARY_API_KEYS:-}" ]]; then export LOYAL_EIL_ADVERSARY_API_KEYS; fi
if [[ -n "${LOYAL_EIL_JUDGE_API_KEY:-}" ]]; then export LOYAL_EIL_JUDGE_API_KEY; fi
if [[ -n "${LOYAL_EIL_JUDGE_API_KEYS:-}" ]]; then export LOYAL_EIL_JUDGE_API_KEYS; fi

export PYTHONUNBUFFERED=1
# Docker renumbers the selected host GPUs.  Derive the visible device list
# from the active 2+2 (or larger) EIL topology rather than assuming six GPUs.
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  EIL_TOTAL_GPU_COUNT=$((LOYAL_EIL_TRAIN_GPU_COUNT + LOYAL_EIL_ROLLOUT_GPU_COUNT))
  CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((EIL_TOTAL_GPU_COUNT - 1)))"
fi
export CUDA_VISIBLE_DEVICES
# Keep source records immutable and train from the reproducible, balanced
# derivative.  It gives bargaining/redress/gatekeeping 40/35/25 exposure,
# equalizes subscenarios within each family, and excludes only high-confidence
# necessary/protected direct-conflict candidates pending human review.
EIL_SOURCE_TRAIN_RECORDS="${LOYAL_EIL_SOURCE_TRAIN_RECORDS:-${PROJECT_ROOT}/eil/data/dataset/EIL/train.jsonl}"
export LOYAL_EIL_TRAIN_RECORDS="${LOYAL_EIL_BALANCED_TRAIN_RECORDS:-${PROJECT_ROOT}/artifacts/training_data/EIL/train.jsonl}"
export LOYAL_EIL_VAL_RECORDS="${LOYAL_EIL_VAL_RECORDS:-${PROJECT_ROOT}/eil/data/dataset/EIL/val.jsonl}"
# The reward worker must recognize both train and in-process validation IDs.
export LOYAL_EIL_RECORDS="${LOYAL_EIL_RECORDS:-${LOYAL_EIL_TRAIN_RECORDS}:${LOYAL_EIL_VAL_RECORDS}}"
export LOYAL_EIL_COVERAGE_CACHE="${LOYAL_EIL_COVERAGE_CACHE:-${PROJECT_ROOT}/artifacts/cache/eil_coverage}"
export LOYAL_EIL_UTILITY_CACHE="${LOYAL_EIL_UTILITY_CACHE:-${PROJECT_ROOT}/artifacts/cache/eil_utility}"

echo "Building balanced EIL training records"
if [[ -f "${PROJECT_ROOT}/scripts/build_balanced_training_data.py" ]]; then
  python3 "${PROJECT_ROOT}/scripts/build_balanced_training_data.py" eil \
    --source "${EIL_SOURCE_TRAIN_RECORDS}" --output "${LOYAL_EIL_TRAIN_RECORDS}"
else
  # The persisted balanced derivative is valid; synchronize the newly
  # calibrated slot severities without changing its family/subscenario mix.
  test -f "${LOYAL_EIL_TRAIN_RECORDS}"
  python3 "${PROJECT_ROOT}/scripts/sync_eil_severity.py" \
    --source "${EIL_SOURCE_TRAIN_RECORDS}" --target "${LOYAL_EIL_TRAIN_RECORDS}"
fi
python3 "${PROJECT_ROOT}/scripts/data/prepare_slime.py" eil \
  --source "${LOYAL_EIL_TRAIN_RECORDS}" --output "${DATA_ROOT}/train.jsonl"
python3 "${PROJECT_ROOT}/scripts/data/prepare_slime.py" eil \
  --source "${LOYAL_EIL_VAL_RECORDS}" \
  --output "${DATA_ROOT}/val.jsonl"

source "${SLIME_ROOT}/scripts/models/qwen3-4B.sh"

CKPT_ARGS=(
  --hf-checkpoint "${LOYAL_QWEN3_4B_HF_CHECKPOINT:-/root/Qwen3-4B}"
  --ref-load "${LOYAL_QWEN3_4B_REF_LOAD:-/root/Qwen3-4B_torch_dist}"
  --load "${LOYAL_EIL_LOAD:-/root/Qwen3-4B_eil_slime}"
  --save "${LOYAL_EIL_SAVE:-/root/Qwen3-4B_eil_slime}"
  # EIL checkpoints and full validation are expensive because each answer is judged externally.
  --save-interval "${LOYAL_EIL_SAVE_INTERVAL:-20}"
)

ROLLOUT_ARGS=(
  --prompt-data "${DATA_ROOT}/train.jsonl"
  --input-key messages
  --label-key record_id
  --apply-chat-template
  # EIL's accelerated run trains on direct answers, not private reasoning traces.
  --apply-chat-template-kwargs '{"enable_thinking": false}'
  --rollout-shuffle
  # An alternating coordinator may override epoch scheduling with a finite
  # chunk of rollout updates, then hand the shared checkpoint to MIU.
  # 64 prompt groups x 8 candidates = 512 scored replies per rollout update.
  --rollout-batch-size "${LOYAL_EIL_ROLLOUT_BATCH_SIZE:-64}"
  --n-samples-per-prompt "${LOYAL_EIL_SAMPLES_PER_PROMPT:-8}"
  --rollout-max-response-len "${LOYAL_EIL_MAX_RESPONSE_LEN:-2048}"
  --rollout-temperature 0.8
  --global-batch-size "${LOYAL_EIL_GLOBAL_BATCH_SIZE:-512}"
  --balance-data
)
if [[ -n "${LOYAL_EIL_NUM_ROLLOUT:-}" ]]; then
  ROLLOUT_ARGS+=(--num-rollout "${LOYAL_EIL_NUM_ROLLOUT}")
else
  # Epoch-driven scheduling covers the full 5,337-record training set.
  ROLLOUT_ARGS+=(--num-epoch "${LOYAL_EIL_NUM_EPOCH:-10}")
fi

RM_ARGS=(
  --custom-rm-path scripts.training.rewards.slime.eil_reward_func
  --custom-reward-post-process-path scripts.training.rewards.slime.eil_post_process_rewards
  --reward-key reward_value
  --eval-reward-key reward_value
  # Score and retry all candidates for a prompt atomically; partial groups are
  # never allowed into GRPO after a scorer outage.
  --group-rm
  --dynamic-sampling-filter-path scripts.training.rewards.filters.keep_eligible_nonzero_std
  --log-reward-category reward_category
)
if [[ -n "${LOYAL_ADAPTIVE_SIGNAL_LOG:-}" ]]; then
  export LOYAL_ADAPTIVE_SIGNAL_LOG LOYAL_ADAPTIVE_SIGNAL_MECHANISM=eil
fi
export LOYAL_REWARD_FAILURE_LOG="${LOYAL_EIL_FAILURE_LOG:-${PROJECT_ROOT}/artifacts/diagnostics/eil_groups.jsonl}"

WANDB_ARGS=()
if [[ "${LOYAL_USE_WANDB:-0}" == "1" ]]; then
  if [[ -n "${WANDB_API_KEY:-}" ]]; then export WANDB_API_KEY; fi
  WANDB_ARGS=(
    --use-wandb
    --wandb-project "${LOYAL_WANDB_PROJECT:-loyal-agent}"
    --wandb-group "${LOYAL_WANDB_GROUP:-eil-qwen3-4b-grpo}"
    --wandb-mode "${LOYAL_WANDB_MODE:-online}"
  )
fi

EVAL_ARGS=()
# A complete EIL validation pass runs all 707 scenes through the external
# adversary and judges, so keep it opt-in instead of blocking the first GRPO step.
if [[ -n "${LOYAL_EIL_EVAL_INTERVAL:-}" ]]; then
  EVAL_ARGS=(
    --eval-interval "${LOYAL_EIL_EVAL_INTERVAL}"
    --eval-prompt-data eil_val "${DATA_ROOT}/val.jsonl"
    --eval-input-key messages
    --eval-label-key record_id
    --n-samples-per-eval-prompt 1
    --eval-temperature 0.0
    --eval-max-response-len "${LOYAL_EIL_MAX_RESPONSE_LEN:-2048}"
  )
fi

PERF_ARGS=(
  # Two actor GPUs with TP=1 form a DP=2 training replica group.
  --tensor-model-parallel-size 1
  --recompute-granularity full
  --recompute-method uniform
  --recompute-num-layers 1
  --use-dynamic-batch-size
  --max-tokens-per-gpu "${LOYAL_EIL_MAX_TOKENS_PER_GPU:-12288}"
)

GRPO_ARGS=(
  --advantage-estimator grpo
  --use-kl-loss
  --kl-loss-coef 0.00
  --kl-loss-type low_var_kl
  --entropy-coef 0.00
  --eps-clip 0.2
  --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(--optimizer adam --lr 1e-6 --lr-decay-style constant --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.98)
# Keep rollout engines single-GPU; this setting does not affect actor DP, but
# avoids a separate TP=2 inference topology.
SGLANG_ARGS=(--rollout-num-gpus-per-engine 1 --sglang-mem-fraction-static "${LOYAL_EIL_SGLANG_MEM_FRACTION_STATIC:-0.28}")
MISC_ARGS=(--attention-dropout 0.0 --hidden-dropout 0.0 --accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32 --attention-backend flash)

ray stop --force || true
RAY_NODE_IP="${MASTER_ADDR:-$(hostname -I | awk '{print $1}') }"
RAY_NODE_IP="${RAY_NODE_IP% }"
# Keep SGLang router-to-engine traffic on the local network, not the HTTP proxy.
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}${RAY_NODE_IP}"
export no_proxy="${no_proxy:+${no_proxy},}${RAY_NODE_IP}"
ray start --head --node-ip-address "${RAY_NODE_IP}" --num-gpus "${LOYAL_EIL_RAY_NUM_GPUS}" --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

RUNTIME_ENV_JSON="{\"env_vars\":{\"PYTHONPATH\":\"${PROJECT_ROOT}:/root/Megatron-LM\",\"CUDA_DEVICE_MAX_CONNECTIONS\":\"1\",\"NO_PROXY\":\"${NO_PROXY}\",\"no_proxy\":\"${no_proxy}\"}}"
cd "${SLIME_ROOT}"
ray job submit --address="http://127.0.0.1:8265" --runtime-env-json="${RUNTIME_ENV_JSON}" -- python3 "${SLIME_ROOT}/train.py" \
  --actor-num-nodes 1 --actor-num-gpus-per-node "${LOYAL_EIL_TRAIN_GPU_COUNT}" --rollout-num-gpus "${LOYAL_EIL_ROLLOUT_GPU_COUNT}" \
  "${MODEL_ARGS[@]}" "${CKPT_ARGS[@]}" "${ROLLOUT_ARGS[@]}" "${RM_ARGS[@]}" "${OPTIMIZER_ARGS[@]}" \
  "${GRPO_ARGS[@]}" "${WANDB_ARGS[@]}" "${PERF_ARGS[@]}" "${EVAL_ARGS[@]}" "${SGLANG_ARGS[@]}" "${MISC_ARGS[@]}"
