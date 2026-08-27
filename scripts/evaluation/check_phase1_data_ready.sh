#!/usr/bin/env bash
# Report whether the known follow-up datasets are present.
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"

paths=(
  "LOYAL_MATH_DATA:${LOYAL_MATH_DATA:-}"
  "LOYAL_UGMATH_DATA:${LOYAL_UGMATH_DATA:-}"
  "LOYAL_GPQA_DATA:${LOYAL_GPQA_DATA:-}"
  "LOYAL_CREATIVE_WRITINGPROMPTS:${LOYAL_CREATIVE_WRITINGPROMPTS:-}"
  "LOYAL_CREATIVE_ROCSTORIES:${LOYAL_CREATIVE_ROCSTORIES:-}"
  "default_math500:${PROJECT_ROOT}/assets/datasets/math500/test.jsonl"
)

missing=0
for item in "${paths[@]}"; do
  name="${item%%:*}"
  value="${item#*:}"
  if [[ -z "${value}" ]]; then
    printf '%s MISSING\n' "${name}"
    missing=1
    continue
  fi
  if [[ -f "${value}" ]]; then
    printf '%s OK %s\n' "${name}" "${value}"
  else
    printf '%s MISSING %s\n' "${name}" "${value}"
    missing=1
  fi
done

exit "${missing}"
