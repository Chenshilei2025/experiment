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

: "${LOYAL_MATH_DATA:?set LOYAL_MATH_DATA to the MATH jsonl/parquet}"
: "${LOYAL_UGMATH_DATA:?set LOYAL_UGMATH_DATA to the UGMATH jsonl/parquet}"
: "${LOYAL_GPQA_DATA:?set LOYAL_GPQA_DATA to the GPQA jsonl/parquet}"
: "${LOYAL_REASONING_OUTPUT_ROOT:=${project_root}/artifacts/evaluations/reasoning}"

mkdir -p "${LOYAL_REASONING_OUTPUT_ROOT}"

python3 -m scripts.evaluation.eval_reasoning_benchmark \
  --task math --checkpoint "${checkpoint}" --data "${LOYAL_MATH_DATA}" \
  --output-dir "${LOYAL_REASONING_OUTPUT_ROOT}/math" \
  --question-key "${LOYAL_MATH_QUESTION_KEY:-problem}" \
  --answer-key "${LOYAL_MATH_ANSWER_KEY:-answer}" \
  --id-key "${LOYAL_MATH_ID_KEY:-unique_id}"

python3 -m scripts.evaluation.eval_reasoning_benchmark \
  --task math --checkpoint "${checkpoint}" --data "${LOYAL_UGMATH_DATA}" \
  --output-dir "${LOYAL_REASONING_OUTPUT_ROOT}/ugmath" \
  --question-key "${LOYAL_UGMATH_QUESTION_KEY:-question}" \
  --answer-key "${LOYAL_UGMATH_ANSWER_KEY:-answer}" \
  --id-key "${LOYAL_UGMATH_ID_KEY:-id}"

python3 -m scripts.evaluation.eval_reasoning_benchmark \
  --task gpqa --checkpoint "${checkpoint}" --data "${LOYAL_GPQA_DATA}" \
  --output-dir "${LOYAL_REASONING_OUTPUT_ROOT}/gpqa" \
  --question-key "${LOYAL_GPQA_QUESTION_KEY:-question}" \
  --answer-key "${LOYAL_GPQA_ANSWER_KEY:-answer}" \
  --choices-key "${LOYAL_GPQA_CHOICES_KEY:-choices}" \
  --id-key "${LOYAL_GPQA_ID_KEY:-id}"
