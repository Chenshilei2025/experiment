#!/usr/bin/env bash
# Independent T2 evaluation of the historical DAPO-Math step999 checkpoint.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
PYTHON="${LOYAL_PYTHON:-/root/experiment_g_runtime/conda/env/bin/python3}"
NATIVE="${LOYAL_T2_NATIVE_CHECKPOINT:-/cephfs/huangzimeng/experiment_g/checkpoints/dapo_math_grpo/iter_0000999}"
CHECKPOINT="${LOYAL_T2_HF_CHECKPOINT:-/cephfs/huangzimeng/experiment_g/artifacts/evaluations/grpo_iter_0000999_hf}"
OUTPUT_ROOT="${LOYAL_T2_OUTPUT_ROOT:-/cephfs/huangzimeng/experiment_g/artifacts/evaluations/five_followups/T2_dapo_math_step999}"
ASSET_ROOT="${LOYAL_ASSET_ROOT:-/cephfs/shared/experiment_g/assets}"
GSM8K="${LOYAL_GSM8K_TEST_DATA:-${ASSET_ROOT}/datasets/gsm8k/main/test-00000-of-00001.parquet}"
MATH500="${LOYAL_MATH500_TEST_DATA:-${ASSET_ROOT}/datasets/math500/test.jsonl}"
AIME="${LOYAL_AIME_TEST_DATA:-/cephfs/huangzimeng/experiment_g/assets/datasets/aime_2024/aime-2024-unique-gsmformat.parquet}"

[[ -s "${NATIVE}/common.pt" && -s "${NATIVE}/.metadata" ]] || { echo "incomplete native step999: ${NATIVE}" >&2; exit 2; }
[[ "$(find "${NATIVE}" -maxdepth 1 -type f -name '*.distcp' | wc -l)" -eq 8 ]] || { echo "native step999 shard count is not 8" >&2; exit 2; }
[[ -f "${CHECKPOINT}/config.json" && -f "${CHECKPOINT}/model.safetensors.index.json" ]] || { echo "incomplete HF export: ${CHECKPOINT}" >&2; exit 2; }
for path in "${GSM8K}" "${MATH500}" "${AIME}"; do [[ -f "${path}" ]] || { echo "missing data: ${path}" >&2; exit 2; }; done

# shellcheck disable=SC1091
source "${PROJECT_ROOT}/scripts/launch/env.sh"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/slime:${PYTHONPATH:-}"
if [[ -f "${OUTPUT_ROOT}/suite_done" ]]; then echo "T2 already complete: ${OUTPUT_ROOT}"; exit 0; fi
mkdir -p "${OUTPUT_ROOT}/logs"

run() { local name="$1" gpu="$2"; shift 2; [[ -e "${OUTPUT_ROOT}/${name}" ]] && rm -rf "${OUTPUT_ROOT:?}/${name}"; CUDA_VISIBLE_DEVICES="${gpu}" "$@" >"${OUTPUT_ROOT}/logs/${name}.log" 2>&1; }

run miu 0 "${PYTHON}" -m scripts.evaluation.cli miu --checkpoint "${CHECKPOINT}" --output-dir "${OUTPUT_ROOT}/miu" --device cuda:0 --batch-size 8 --max-new-tokens 384 --disable-thinking & p_miu=$!
run eil 1 "${PYTHON}" -m scripts.evaluation.cli eil --checkpoint "${CHECKPOINT}" --output-dir "${OUTPUT_ROOT}/eil" --device cuda:0 --batch-size 2 --max-new-tokens 2048 --score-concurrency 1 --disable-thinking & p_eil=$!
run gsm8k 2 "${PYTHON}" "${PROJECT_ROOT}/scripts/evaluation/eval_gsm8k.py" --checkpoint "${CHECKPOINT}" --test-data "${GSM8K}" --output-dir "${OUTPUT_ROOT}/gsm8k" --device cuda:0 & p_gsm=$!
run math500 3 "${PYTHON}" "${PROJECT_ROOT}/scripts/evaluation/eval_reasoning_benchmark.py" --task math --checkpoint "${CHECKPOINT}" --data "${MATH500}" --output-dir "${OUTPUT_ROOT}/math500" --question-key problem --answer-key answer --id-key unique_id --device cuda:0 & p_math=$!
wait "${p_miu}"; wait "${p_eil}"; wait "${p_math}"; wait "${p_gsm}"
run aime_pass16 2 "${PYTHON}" "${PROJECT_ROOT}/scripts/evaluation/eval_aime_passk.py" --checkpoint "${CHECKPOINT}" --test-data "${AIME}" --output-dir "${OUTPUT_ROOT}/aime_pass16" --device cuda:0 --num-samples 16

"${PYTHON}" - "${OUTPUT_ROOT}" "${NATIVE}" "${CHECKPOINT}" <<'PY'
import json
import sys
from pathlib import Path
root, native, checkpoint = map(Path, sys.argv[1:])
expected = {"miu": 385, "eil": 656, "gsm8k": 1319, "math500": 500, "aime_pass16": 30}
summaries = {}
for name, count in expected.items():
    summary = json.loads((root / name / "summary.json").read_text(encoding="utf-8"))
    actual = summary.get("n_total", summary.get("n_questions"))
    if actual != count:
        raise SystemExit(f"{name}: expected {count}, got {actual}")
    summaries[name] = summary
(root / "acceptance.json").write_text(json.dumps({"condition": "T2", "native_checkpoint": str(native), "hf_checkpoint": str(checkpoint), "expected_counts": expected, "summaries": summaries}, ensure_ascii=False, indent=2) + "\n")
PY
touch "${OUTPUT_ROOT}/suite_done"
echo "T2 complete: ${OUTPUT_ROOT}"
