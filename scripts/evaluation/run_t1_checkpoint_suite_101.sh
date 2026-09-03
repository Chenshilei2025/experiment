#!/usr/bin/env bash
# Run the complete T1 evaluation suite for one exported checkpoint.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
CHECKPOINT="${1:?usage: $0 <hf-checkpoint> <output-root> }"
OUTPUT_ROOT="${2:?usage: $0 <hf-checkpoint> <output-root> }"
PYTHON="${LOYAL_PYTHON:-/root/experiment_g_runtime/conda/env/bin/python3}"
ASSET_ROOT="${LOYAL_ASSET_ROOT:-/cephfs/shared/experiment_g/assets}"
GSM8K="${LOYAL_GSM8K_TEST_DATA:-${ASSET_ROOT}/datasets/gsm8k/main/test-00000-of-00001.parquet}"
MATH500="${LOYAL_MATH500_TEST_DATA:-${ASSET_ROOT}/datasets/math500/test.jsonl}"
AIME="${LOYAL_AIME_TEST_DATA:-/cephfs/huangzimeng/experiment_g/assets/datasets/aime_2024/aime-2024-unique-gsmformat.parquet}"
STORY_CLOZE="${LOYAL_STORY_CLOZE_TEST_DATA:-/cephfs/shared/experiment_g/assets/datasets/rocstories_story_cloze_2016/test.parquet}"
EIL_RECORDS="${LOYAL_EIL_TEST_RECORDS:-${PROJECT_ROOT}/eil/data/dataset/EIL/test.jsonl}"
MIU_RECORDS="${LOYAL_MIU_TEST_RECORDS:-${PROJECT_ROOT}/miu/data/dataset/MIU/test.jsonl}"

[[ -d "${CHECKPOINT}" ]] || { echo "missing checkpoint: ${CHECKPOINT}" >&2; exit 2; }
for path in "${GSM8K}" "${MATH500}" "${AIME}" "${STORY_CLOZE}" "${EIL_RECORDS}" "${MIU_RECORDS}"; do
  [[ -f "${path}" ]] || { echo "missing evaluation data: ${path}" >&2; exit 2; }
done
[[ ! -e "${OUTPUT_ROOT}" ]] || { echo "refusing to overwrite ${OUTPUT_ROOT}" >&2; exit 2; }
mkdir -p "${OUTPUT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/slime:${PYTHONPATH:-}"

"${PYTHON}" -m scripts.evaluation.cli miu \
  --checkpoint "${CHECKPOINT}" --records "${MIU_RECORDS}" --output-dir "${OUTPUT_ROOT}/miu" \
  --device "${LOYAL_T1_EVAL_DEVICE:-cuda:0}" --batch-size 8 --max-new-tokens 384 --disable-thinking
"${PYTHON}" -m scripts.evaluation.cli eil \
  --checkpoint "${CHECKPOINT}" --records "${EIL_RECORDS}" --output-dir "${OUTPUT_ROOT}/eil" \
  --device "${LOYAL_T1_EVAL_DEVICE:-cuda:0}" --batch-size 2 --max-new-tokens 2048 --score-concurrency 1 --disable-thinking

"${PYTHON}" "${PROJECT_ROOT}/scripts/evaluation/eval_gsm8k.py" \
  --checkpoint "${CHECKPOINT}" --test-data "${GSM8K}" --output-dir "${OUTPUT_ROOT}/gsm8k" --device "${LOYAL_T1_EVAL_DEVICE:-cuda:0}"
"${PYTHON}" "${PROJECT_ROOT}/scripts/evaluation/eval_reasoning_benchmark.py" \
  --task math --checkpoint "${CHECKPOINT}" --data "${MATH500}" --output-dir "${OUTPUT_ROOT}/math500" \
  --question-key problem --answer-key answer --id-key unique_id --device "${LOYAL_T1_EVAL_DEVICE:-cuda:0}"
"${PYTHON}" "${PROJECT_ROOT}/scripts/evaluation/eval_aime_passk.py" \
  --checkpoint "${CHECKPOINT}" --test-data "${AIME}" --output-dir "${OUTPUT_ROOT}/aime_pass16" \
  --device "${LOYAL_T1_EVAL_DEVICE:-cuda:0}" --num-samples 16
"${PYTHON}" "${PROJECT_ROOT}/scripts/evaluation/eval_story_cloze.py" \
  --checkpoint "${CHECKPOINT}" --data "${STORY_CLOZE}" --output-dir "${OUTPUT_ROOT}/story_cloze" \
  --device "${LOYAL_T1_EVAL_DEVICE:-cuda:0}"

"${PYTHON}" - "${OUTPUT_ROOT}" <<'PY'
import json
import sys
from pathlib import Path
root = Path(sys.argv[1])
expected = {"miu": 385, "eil": 656, "story_cloze": 1871}
for name, count in expected.items():
    summary = json.loads((root / name / "summary.json").read_text(encoding="utf-8"))
    actual = summary.get("n_total", summary.get("n_questions"))
    if actual != count:
        raise SystemExit(f"{name}: expected {count} rows, got {actual}")
print("checkpoint_suite_acceptance_ok")
PY

echo "checkpoint_suite_complete checkpoint=${CHECKPOINT} output=${OUTPUT_ROOT}"
