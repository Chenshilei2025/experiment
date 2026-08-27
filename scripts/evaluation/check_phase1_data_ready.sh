#!/usr/bin/env bash
# Report whether the known follow-up datasets are present.
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ -d "${PROJECT_ROOT}/assets" ]]; then
  ASSET_ROOT="${LOYAL_ASSET_ROOT:-${PROJECT_ROOT}/assets}"
else
  ASSET_ROOT="${LOYAL_ASSET_ROOT:-$(cd -- "${PROJECT_ROOT}/.." && pwd)/assets}"
fi

paths=(
  "LOYAL_MATH_DATA:${LOYAL_MATH_DATA:-${ASSET_ROOT}/datasets/math500/test.jsonl}"
  "LOYAL_UGMATH_DATA:${LOYAL_UGMATH_DATA:-}"
  "LOYAL_GPQA_DATA:${LOYAL_GPQA_DATA:-}"
  "LOYAL_CREATIVE_WRITINGPROMPTS:${LOYAL_CREATIVE_WRITINGPROMPTS:-}"
  "LOYAL_CREATIVE_ROCSTORIES:${LOYAL_CREATIVE_ROCSTORIES:-}"
  "LOYAL_CREATIVE_TRAIN_RECORDS:${LOYAL_CREATIVE_TRAIN_RECORDS:-${PROJECT_ROOT}/artifacts/slime/CREATIVE/train.parquet}"
  "default_math500:${ASSET_ROOT}/datasets/math500/test.jsonl"
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
    size="$(wc -c <"${value}")"
    if [[ "${value}" == *.jsonl ]]; then
      lines="$(grep -cve '^[[:space:]]*$' "${value}")"
      printf '%s OK %s bytes=%s nonempty_lines=%s\n' "${name}" "${value}" "${size}" "${lines}"
    else
      printf '%s OK %s bytes=%s\n' "${name}" "${value}" "${size}"
    fi
  else
    printf '%s MISSING %s\n' "${name}" "${value}"
    missing=1
  fi
done

exit "${missing}"
