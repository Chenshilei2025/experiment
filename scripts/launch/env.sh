#!/usr/bin/env bash
# Load local credentials and bridge the original shared scorer names to training names.
set -euo pipefail

TRAINING_ENV_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
# Keep every explicit LOYAL_* override after loading .env.  Experiments are
# declarative configs, so a new hyperparameter must not require editing a
# second, error-prone allowlist in both the host and container launchers.
declare -A TRAINING_ENV_OVERRIDES=()
while IFS= read -r name; do
  [[ "${name}" =~ ^LOYAL_[A-Z0-9_]+$ ]] || continue
  # Service identity and credentials are intentionally read only from .env.
  # Versioned experiment configs may tune training, not silently replace an
  # evaluator or endpoint.
  [[ "${name}" =~ (_API_KEY|_API_KEYS|_BASE_URL)$ ]] && continue
  case "${name}" in
    LOYAL_MIU_JUDGE_MODEL|LOYAL_EIL_JUDGE_MODEL|LOYAL_EIL_ADVERSARY_MODEL) continue ;;
  esac
  if [[ -v "${name}" ]]; then
    TRAINING_ENV_OVERRIDES["${name}"]="${!name}"
  fi
done < <(compgen -v)
if [[ -f "${TRAINING_ENV_ROOT}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${TRAINING_ENV_ROOT}/.env"
  set +a
fi
for name in "${!TRAINING_ENV_OVERRIDES[@]}"; do
  export "${name}=${TRAINING_ENV_OVERRIDES[${name}]}"
done

# Role-specific judge models fall back to the shared endpoint and credential.
export LOYAL_MIU_JUDGE_BASE_URL="${LOYAL_MIU_JUDGE_BASE_URL:-${LOYAL_JUDGE_BASE_URL:-}}"
export LOYAL_MIU_JUDGE_MODEL="${LOYAL_MIU_JUDGE_MODEL:-${LOYAL_JUDGE_MODEL:-}}"
export LOYAL_MIU_JUDGE_API_KEY="${LOYAL_MIU_JUDGE_API_KEY:-${LOYAL_JUDGE_API_KEY:-}}"
export LOYAL_EIL_JUDGE_BASE_URL="${LOYAL_EIL_JUDGE_BASE_URL:-${LOYAL_JUDGE_BASE_URL:-}}"
export LOYAL_EIL_JUDGE_MODEL="${LOYAL_EIL_JUDGE_MODEL:-${LOYAL_JUDGE_MODEL:-}}"
export LOYAL_EIL_JUDGE_API_KEY="${LOYAL_EIL_JUDGE_API_KEY:-${LOYAL_JUDGE_API_KEY:-}}"
export LOYAL_EIL_LEAKAGE_JUDGE_BASE_URL="${LOYAL_EIL_LEAKAGE_JUDGE_BASE_URL:-${LOYAL_EIL_JUDGE_BASE_URL}}"
export LOYAL_EIL_LEAKAGE_JUDGE_MODEL="${LOYAL_EIL_LEAKAGE_JUDGE_MODEL:-${LOYAL_EIL_JUDGE_MODEL}}"
export LOYAL_EIL_UTILITY_JUDGE_BASE_URL="${LOYAL_EIL_UTILITY_JUDGE_BASE_URL:-${LOYAL_EIL_JUDGE_BASE_URL}}"
export LOYAL_EIL_UTILITY_JUDGE_MODEL="${LOYAL_EIL_UTILITY_JUDGE_MODEL:-${LOYAL_EIL_JUDGE_MODEL}}"
export LOYAL_MIU_FAITHFULNESS_JUDGE_BASE_URL="${LOYAL_MIU_FAITHFULNESS_JUDGE_BASE_URL:-${LOYAL_MIU_JUDGE_BASE_URL}}"
export LOYAL_MIU_FAITHFULNESS_JUDGE_MODEL="${LOYAL_MIU_FAITHFULNESS_JUDGE_MODEL:-${LOYAL_MIU_JUDGE_MODEL}}"
export LOYAL_EIL_ADVERSARY_BASE_URL="${LOYAL_EIL_ADVERSARY_BASE_URL:-${LOYAL_ADVERSARY_BASE_URL:-}}"
export LOYAL_EIL_ADVERSARY_MODEL="${LOYAL_EIL_ADVERSARY_MODEL:-${LOYAL_ADVERSARY_MODEL:-qwen3.5-35b-a3b}}"
export LOYAL_EIL_ADVERSARY_API_KEY="${LOYAL_EIL_ADVERSARY_API_KEY:-${LOYAL_ADVERSARY_API_KEY:-}}"
# The Llama checkpoint is installed in the shared model store rather than the
# project-specific one used by Qwen and GLM.  Launchers select this mount only
# when LOYAL_BASE_MODEL=llama3.1-8b-instruct.
export LOYAL_LLAMA3_1_8B_MODEL_ROOT="${LOYAL_LLAMA3_1_8B_MODEL_ROOT:-/ssd/models}"

# The experimental training protocol fixes the evaluator families: Qwen is
# the EIL adversary and DeepSeek scores EIL utility/leakage and MIU reasoning.
# Model variants remain configurable in .env, but a training run must not
# silently change evaluator family between conditions.
require_fixed_training_evaluators() {
  local role model
  for role in LOYAL_MIU_FAITHFULNESS_JUDGE_MODEL LOYAL_EIL_LEAKAGE_JUDGE_MODEL LOYAL_EIL_UTILITY_JUDGE_MODEL; do
    model="${!role:-}"
    if [[ "${model,,}" != *deepseek* ]]; then
      echo "${role} must name a DeepSeek judge for the fixed training protocol; got ${model:-<unset>}" >&2
      return 1
    fi
  done
  model="${LOYAL_EIL_ADVERSARY_MODEL:-}"
  if [[ "${model,,}" != *qwen* ]]; then
    echo "LOYAL_EIL_ADVERSARY_MODEL must name a Qwen adversary for the fixed training protocol; got ${model:-<unset>}" >&2
    return 1
  fi
}
