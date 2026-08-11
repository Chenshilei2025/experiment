  #!/usr/bin/env bash
set -euo pipefail

# MIU is intentionally separate from EIL: its prompt protocol and reward differ.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/env.sh"
SLIME_ROOT="${SLIME_ROOT:-${PROJECT_ROOT}/slime}"
DATA_ROOT="${LOYAL_DATA_ROOT:-${PROJECT_ROOT}/artifacts/slime/MIU}"

: "${LOYAL_MIU_JUDGE_BASE_URL:?set the MIU judge endpoint}"
: "${LOYAL_MIU_JUDGE_MODEL:?set the MIU judge model}"
python3 "${PROJECT_ROOT}/scripts/training/preflight.py" miu --runtime

# The fresh Ray daemon inherits these values; credentials never enter CLI args.
export LOYAL_MIU_JUDGE_BASE_URL LOYAL_MIU_JUDGE_MODEL
if [[ -n "${LOYAL_MIU_JUDGE_API_KEY:-}" ]]; then export LOYAL_MIU_JUDGE_API_KEY; fi

export PYTHONUNBUFFERED=1
# Docker renumbers the selected host GPUs to contiguous container IDs.  Derive
# the default from the assigned training and rollout counts so reduced-GPU
# launches do not advertise more Ray GPUs than are visible in the container.
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  MIU_TOTAL_GPU_COUNT=$((LOYAL_MIU_TRAIN_GPU_COUNT + LOYAL_MIU_ROLLOUT_GPU_COUNT))
  CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((MIU_TOTAL_GPU_COUNT - 1)))"
fi
export CUDA_VISIBLE_DEVICES
# MIU training records live with the MIU dataset.  The commerce reference
# option positions are balanced in this source file, so the prompt builder and
# reward store consume one canonical, version-controlled training set.
export LOYAL_MIU_TRAIN_RECORDS="${LOYAL_MIU_TRAIN_RECORDS:-${PROJECT_ROOT}/miu/data/dataset/MIU/train.jsonl}"
export LOYAL_MIU_VAL_RECORDS="${LOYAL_MIU_VAL_RECORDS:-${PROJECT_ROOT}/miu/data/dataset/MIU/val.jsonl}"
export LOYAL_MIU_RECORDS="${LOYAL_MIU_RECORDS:-${LOYAL_MIU_TRAIN_RECORDS}:${LOYAL_MIU_VAL_RECORDS}}"

echo "Rebuilding MIU prompt data with loyal_agent_prompt.py"
python3 "${PROJECT_ROOT}/scripts/data/prepare_slime.py" miu \
  --source "${LOYAL_MIU_TRAIN_RECORDS}" --output "${DATA_ROOT}/train.jsonl"
python3 "${PROJECT_ROOT}/scripts/data/prepare_slime.py" miu \
  --source "${LOYAL_MIU_VAL_RECORDS}" \
  --output "${DATA_ROOT}/val.jsonl"

source "${SLIME_ROOT}/scripts/models/qwen3-4B.sh"

CKPT_ARGS=(
  --hf-checkpoint "${LOYAL_QWEN3_4B_HF_CHECKPOINT:-/root/Qwen3-4B}"
  --ref-load "${LOYAL_QWEN3_4B_REF_LOAD:-/root/Qwen3-4B_torch_dist}"
  --load "${LOYAL_MIU_LOAD:-/root/Qwen3-4B_miu_slime}"
  --save "${LOYAL_MIU_SAVE:-/root/Qwen3-4B_miu_slime}"
  --save-interval "${LOYAL_MIU_SAVE_INTERVAL:-50}"
  # Megatron always retains the newest checkpoint and additionally retains
  # checkpoints at this interval. Its large default keeps only the newest
  # complete checkpoint in ordinary runs; a 4B checkpoint is about 53 GB.
  --save-retain-interval "${LOYAL_MIU_SAVE_RETAIN_INTERVAL:-1000000}"
)

ROLLOUT_ARGS=(
  --prompt-data "${DATA_ROOT}/train.jsonl"
  --input-key messages
  --label-key record_id
  --apply-chat-template
  # MIU scores a concise cited answer, not a hidden chain of thought. Disable
  # Qwen3 thinking in the chat template so these tokens neither truncate nor
  # enter the policy loss.
  --apply-chat-template-kwargs '{"enable_thinking": false}'
  --rollout-function-path scripts.training.rollout.miu.generate_rollout
  --rollout-shuffle
  # An alternating coordinator may override epoch scheduling with a finite
  # chunk of rollout updates, then hand the shared checkpoint to EIL.
  # Keep the scored-reply budget fixed while eight candidates provide GRPO
  # variation within each prompt group.
  --rollout-batch-size "${LOYAL_MIU_ROLLOUT_BATCH_SIZE:-128}"
  --n-samples-per-prompt "${LOYAL_MIU_SAMPLES_PER_PROMPT:-8}"
  --rollout-max-response-len "${LOYAL_MIU_MAX_RESPONSE_LEN:-2048}"
  --rollout-temperature 0.8
  --global-batch-size "${LOYAL_MIU_GLOBAL_BATCH_SIZE:-512}"
  --balance-data
)
if [[ -n "${LOYAL_MIU_NUM_ROLLOUT:-}" ]]; then
  ROLLOUT_ARGS+=(--num-rollout "${LOYAL_MIU_NUM_ROLLOUT}")
else
  # Epoch-driven scheduling covers the full 2,824-record training set.
  # With 128 prompts per rollout, 30 epochs gives about 660 optimizer updates.
  ROLLOUT_ARGS+=(--num-epoch "${LOYAL_MIU_NUM_EPOCH:-30}")
fi

RM_ARGS=(
  --custom-rm-path scripts.training.rewards.slime.miu_reward_func
  --custom-reward-post-process-path scripts.training.rewards.slime.miu_post_process_rewards
  --reward-key reward_value
  --eval-reward-key reward_value
  # Score and retry all candidates for a prompt atomically; partial groups are
  # never allowed into GRPO after a scorer outage.
  --group-rm
  --dynamic-sampling-filter-path scripts.training.rewards.filters.keep_eligible_nonzero_std
  --log-reward-category reward_category
)
if [[ -n "${LOYAL_ADAPTIVE_SIGNAL_LOG:-}" ]]; then
  export LOYAL_ADAPTIVE_SIGNAL_LOG LOYAL_ADAPTIVE_SIGNAL_MECHANISM=miu
fi
export LOYAL_REWARD_FAILURE_LOG="${LOYAL_MIU_FAILURE_LOG:-${PROJECT_ROOT}/artifacts/diagnostics/miu_groups.jsonl}"

WANDB_ARGS=()
if [[ "${LOYAL_USE_WANDB:-0}" == "1" ]]; then
  if [[ -n "${WANDB_API_KEY:-}" ]]; then export WANDB_API_KEY; fi
  WANDB_ARGS=(
    --use-wandb
    --wandb-project "${LOYAL_WANDB_PROJECT:-loyal-agent}"
    --wandb-group "${LOYAL_WANDB_GROUP:-miu-qwen3-4b-grpo}"
    --wandb-mode "${LOYAL_WANDB_MODE:-online}"
  )
fi

EVAL_ARGS=()
if [[ "${LOYAL_MIU_DISABLE_EVAL:-0}" != "1" ]]; then
  EVAL_ARGS=(
    --eval-interval "${LOYAL_MIU_EVAL_INTERVAL:-10}"
    --eval-prompt-data miu_val "${DATA_ROOT}/val.jsonl"
    --eval-input-key messages
    --eval-label-key record_id
    --n-samples-per-eval-prompt 1
    --eval-temperature 0.0
    --eval-max-response-len "${LOYAL_MIU_MAX_RESPONSE_LEN:-2048}"
  )
  if [[ "${LOYAL_MIU_SKIP_INITIAL_EVAL:-1}" == "1" ]]; then
    EVAL_ARGS+=(--skip-initial-eval)
  fi
fi

PERF_ARGS=(
  # Two actor GPUs with TP=1 form a DP=2 training replica group.
  --tensor-model-parallel-size 1
  --recompute-granularity full
  --recompute-method uniform
  --recompute-num-layers 1
  --use-dynamic-batch-size
  --max-tokens-per-gpu "${LOYAL_MIU_MAX_TOKENS_PER_GPU:-12288}"
)

GRPO_ARGS=(
  --advantage-estimator grpo
  --use-kl-loss
  --kl-loss-coef "${LOYAL_MIU_KL_LOSS_COEF:-0.01}"
  --kl-loss-type low_var_kl
  --entropy-coef "${LOYAL_MIU_ENTROPY_COEF:-0.001}"
  --eps-clip "${LOYAL_MIU_EPS_CLIP:-0.2}"
  --eps-clip-high "${LOYAL_MIU_EPS_CLIP_HIGH:-0.28}"
)
OPTIMIZER_ARGS=(
  --optimizer adam
  --lr "${LOYAL_MIU_LEARNING_RATE:-5e-7}"
  --lr-decay-style constant
  --weight-decay 0.1
  --adam-beta1 0.9
  --adam-beta2 0.98
  --clip-grad "${LOYAL_MIU_CLIP_GRAD:-1.0}"
)
# Each rollout engine uses one of the two dedicated rollout GPUs.
SGLANG_ARGS=(
  --rollout-num-gpus-per-engine 1
  --sglang-mem-fraction-static "${LOYAL_MIU_SGLANG_MEM_FRACTION_STATIC:-0.7}"
  # Bound queued requests so the router does not open its circuit breaker
  # during long thinking rollouts.
  --sglang-server-concurrency "${LOYAL_MIU_SGLANG_SERVER_CONCURRENCY:-128}"
)
MISC_ARGS=(--attention-dropout 0.0 --hidden-dropout 0.0 --accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32 --attention-backend flash)

ray stop --force || true
RAY_NODE_IP="${MASTER_ADDR:-$(hostname -I | awk '{print $1}') }"
RAY_NODE_IP="${RAY_NODE_IP% }"
# Keep Ray's dashboard and SGLang router-to-engine traffic off the HTTP proxy.
LOCAL_NO_PROXY="127.0.0.1,localhost,${RAY_NODE_IP}"
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}${LOCAL_NO_PROXY}"
export no_proxy="${no_proxy:+${no_proxy},}${LOCAL_NO_PROXY}"
ray start --head --node-ip-address "${RAY_NODE_IP}" --num-gpus "${LOYAL_MIU_RAY_NUM_GPUS}" --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

RUNTIME_ENV_JSON="{\"env_vars\":{\"PYTHONPATH\":\"${PROJECT_ROOT}:/root/Megatron-LM\",\"CUDA_DEVICE_MAX_CONNECTIONS\":\"1\",\"PYTORCH_CUDA_ALLOC_CONF\":\"expandable_segments:True\",\"NO_PROXY\":\"${NO_PROXY}\",\"no_proxy\":\"${no_proxy}\"}}"
cd "${SLIME_ROOT}"
ray job submit --address="http://127.0.0.1:8265" --runtime-env-json="${RUNTIME_ENV_JSON}" -- python3 "${SLIME_ROOT}/train.py" \
  --actor-num-nodes 1 --actor-num-gpus-per-node "${LOYAL_MIU_TRAIN_GPU_COUNT}" --rollout-num-gpus "${LOYAL_MIU_ROLLOUT_GPU_COUNT}" \
  "${MODEL_ARGS[@]}" "${CKPT_ARGS[@]}" "${ROLLOUT_ARGS[@]}" "${RM_ARGS[@]}" "${OPTIMIZER_ARGS[@]}" \
  "${GRPO_ARGS[@]}" "${WANDB_ARGS[@]}" "${PERF_ARGS[@]}" "${EVAL_ARGS[@]}" "${SGLANG_ARGS[@]}" "${MISC_ARGS[@]}"
