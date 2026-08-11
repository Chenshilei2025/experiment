#!/usr/bin/env bash
# Run one tiny MIU -> EIL GRPO chain through a single shared policy checkpoint.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

# Keep the test small but preserve GRPO's within-prompt relative comparison.
# The number of generated samples equals the global batch size for one update.
export LOYAL_SHARED_CHECKPOINT_NAME="${LOYAL_SHARED_CHECKPOINT_NAME:-Qwen3-4B_loyal_smoke}"
export LOYAL_MIU_NUM_ROLLOUT="${LOYAL_MIU_NUM_ROLLOUT:-1}"
# SLIME resumes from the saved global rollout ID.  The first MIU update saves
# ID 0, so EIL must run through total ID 1 to perform its one new update.
export LOYAL_EIL_NUM_ROLLOUT="${LOYAL_EIL_NUM_ROLLOUT:-2}"
export LOYAL_MIU_ROLLOUT_BATCH_SIZE="${LOYAL_MIU_ROLLOUT_BATCH_SIZE:-1}"
export LOYAL_EIL_ROLLOUT_BATCH_SIZE="${LOYAL_EIL_ROLLOUT_BATCH_SIZE:-1}"
export LOYAL_MIU_SAMPLES_PER_PROMPT="${LOYAL_MIU_SAMPLES_PER_PROMPT:-4}"
export LOYAL_EIL_SAMPLES_PER_PROMPT="${LOYAL_EIL_SAMPLES_PER_PROMPT:-4}"
export LOYAL_MIU_GLOBAL_BATCH_SIZE="${LOYAL_MIU_GLOBAL_BATCH_SIZE:-4}"
export LOYAL_EIL_GLOBAL_BATCH_SIZE="${LOYAL_EIL_GLOBAL_BATCH_SIZE:-4}"
export LOYAL_MIU_SAVE_INTERVAL="${LOYAL_MIU_SAVE_INTERVAL:-1}"
export LOYAL_EIL_SAVE_INTERVAL="${LOYAL_EIL_SAVE_INTERVAL:-1}"
# Evaluation is a separate full remote-scoring pass, not a smoke-test check.
export LOYAL_MIU_DISABLE_EVAL=1
export LOYAL_EIL_EVAL_INTERVAL=""

echo "Shared checkpoint: artifacts/checkpoints/${LOYAL_SHARED_CHECKPOINT_NAME}"
echo "Stage 1/2: MIU (${LOYAL_MIU_NUM_ROLLOUT} rollout)"
bash "${SCRIPT_DIR}/run_training_container.sh" miu
echo "Stage 2/2: EIL (${LOYAL_EIL_NUM_ROLLOUT} rollout), resuming the MIU checkpoint"
bash "${SCRIPT_DIR}/run_training_container.sh" eil
echo "Smoke chain completed: artifacts/checkpoints/${LOYAL_SHARED_CHECKPOINT_NAME}"
