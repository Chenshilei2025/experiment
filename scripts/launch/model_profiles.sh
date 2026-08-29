#!/usr/bin/env bash
# Select a dense model family supported by the pinned SLIME image.
#
# This file is sourced inside the training container after /models is mounted.
# It exports generic paths and sources SLIME's matching Megatron model args.
# Do not set these paths from untrusted user input: the caller validates the
# model key before constructing the container invocation.

set -euo pipefail

: "${SLIME_ROOT:?set SLIME_ROOT before sourcing model_profiles.sh}"
MODEL_KEY="${LOYAL_BASE_MODEL:-qwen3-4b}"

case "${MODEL_KEY}" in
  qwen3-4b)
    LOYAL_MODEL_HF_CHECKPOINT="${LOYAL_MODEL_HF_CHECKPOINT:-/models/Qwen3-4B}"
    LOYAL_MODEL_REF_LOAD="${LOYAL_MODEL_REF_LOAD:-/models/Qwen3-4B_torch_dist}"
    LOYAL_MODEL_VOCAB_SIZE=151936
    LOYAL_MODEL_CHAT_TEMPLATE_KWARGS='{"enable_thinking": false}'
    source "${SLIME_ROOT}/scripts/models/qwen3-4B.sh"
    ;;
  glm-z1-9b)
    # This is the current 9B GLM checkpoint exercised by SLIME's GLM4 quick
    # start, rather than an unverified similarly named GLM release.
    LOYAL_MODEL_HF_CHECKPOINT="${LOYAL_MODEL_HF_CHECKPOINT:-/models/GLM-Z1-9B-0414}"
    LOYAL_MODEL_REF_LOAD="${LOYAL_MODEL_REF_LOAD:-/models/GLM-Z1-9B-0414_torch_dist}"
    LOYAL_MODEL_VOCAB_SIZE=151552
    LOYAL_MODEL_CHAT_TEMPLATE_KWARGS='{}'
    source "${SLIME_ROOT}/scripts/models/glm4-9B.sh"
    ;;
  llama3.1-8b-instruct)
    LOYAL_MODEL_HF_CHECKPOINT="${LOYAL_MODEL_HF_CHECKPOINT:-/models/Llama-3.1-8B-Instruct}"
    LOYAL_MODEL_REF_LOAD="${LOYAL_MODEL_REF_LOAD:-/models/Llama-3.1-8B-Instruct_torch_dist}"
    LOYAL_MODEL_VOCAB_SIZE=128256
    LOYAL_MODEL_CHAT_TEMPLATE_KWARGS='{}'
    source "${SLIME_ROOT}/scripts/models/llama3.1-8B-Instruct.sh"
    ;;
  olmo3-7b-instruct)
    LOYAL_MODEL_HF_CHECKPOINT="${LOYAL_MODEL_HF_CHECKPOINT:-/models/Olmo-3-7B-Instruct}"
    LOYAL_MODEL_REF_LOAD="${LOYAL_MODEL_REF_LOAD:-/models/Olmo-3-7B-Instruct_torch_dist}"
    LOYAL_MODEL_VOCAB_SIZE=100278
    LOYAL_MODEL_CHAT_TEMPLATE_KWARGS='{"add_generation_prompt":true}'
    source "${SLIME_ROOT}/scripts/models/olmo3-7B-Instruct.sh"
    ;;
  *)
    echo "unsupported LOYAL_BASE_MODEL=${MODEL_KEY}; choose qwen3-4b, glm-z1-9b, llama3.1-8b-instruct, or olmo3-7b-instruct" >&2
    return 2
    ;;
esac

export LOYAL_BASE_MODEL LOYAL_MODEL_HF_CHECKPOINT LOYAL_MODEL_REF_LOAD LOYAL_MODEL_VOCAB_SIZE LOYAL_MODEL_CHAT_TEMPLATE_KWARGS
