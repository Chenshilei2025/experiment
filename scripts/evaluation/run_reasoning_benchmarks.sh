#!/usr/bin/env bash
# Evaluate one exported HF checkpoint on MATH, UGMATH, and GPQA.
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <hf-checkpoint-path>" >&2
  exit 2
fi

checkpoint="$1"
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${project_root}/scripts/launch/env.sh"

asset_root="${LOYAL_ASSET_ROOT:-}"
if [[ -z "${asset_root}" ]]; then
  if [[ -d "${project_root}/assets" ]]; then
    asset_root="${project_root}/assets"
  else
    asset_root="$(cd -- "${project_root}/.." && pwd)/assets"
  fi
fi
: "${LOYAL_MATH_DATA:=${asset_root}/datasets/math500/test.jsonl}"
: "${LOYAL_REASONING_OUTPUT_ROOT:=${project_root}/artifacts/evaluations/reasoning}"
ugmath_data="${LOYAL_UGMATH_DATA:-}"
gpqa_data="${LOYAL_GPQA_DATA:-}"

mkdir -p "${LOYAL_REASONING_OUTPUT_ROOT}"

run_eval() {
  local name="$1"
  shift
  local data_path="$1"
  shift
  if [[ ! -f "${data_path}" ]]; then
    printf 'SKIP %s missing %s\n' "${name}" "${data_path}" >&2
    return 0
  fi
  python3 -m scripts.evaluation.eval_reasoning_benchmark "$@"
}

run_eval math "${LOYAL_MATH_DATA}" \
  --task math --checkpoint "${checkpoint}" --data "${LOYAL_MATH_DATA}" \
  --output-dir "${LOYAL_REASONING_OUTPUT_ROOT}/math" \
  --question-key "${LOYAL_MATH_QUESTION_KEY:-problem}" \
  --answer-key "${LOYAL_MATH_ANSWER_KEY:-answer}" \
  --id-key "${LOYAL_MATH_ID_KEY:-unique_id}"

run_eval ugmath "${ugmath_data}" \
  --task math --checkpoint "${checkpoint}" --data "${ugmath_data}" \
  --output-dir "${LOYAL_REASONING_OUTPUT_ROOT}/ugmath" \
  --question-key "${LOYAL_UGMATH_QUESTION_KEY:-question}" \
  --answer-key "${LOYAL_UGMATH_ANSWER_KEY:-answer}" \
  --id-key "${LOYAL_UGMATH_ID_KEY:-id}"

run_eval gpqa "${gpqa_data}" \
  --task gpqa --checkpoint "${checkpoint}" --data "${gpqa_data}" \
  --output-dir "${LOYAL_REASONING_OUTPUT_ROOT}/gpqa" \
  --question-key "${LOYAL_GPQA_QUESTION_KEY:-question}" \
  --answer-key "${LOYAL_GPQA_ANSWER_KEY:-answer}" \
  --choices-key "${LOYAL_GPQA_CHOICES_KEY:-choices}" \
  --id-key "${LOYAL_GPQA_ID_KEY:-id}"
