#!/usr/bin/env bash
# Gate independent T3 GSM8K GRPO on the complete T2 evaluation contract.
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
T2_ROOT="${LOYAL_T2_OUTPUT_ROOT:-/cephfs/huangzimeng/experiment_g/artifacts/evaluations/five_followups/T2_dapo_math_step999}"
T3_ROOT="${LOYAL_T3_SAVE_ROOT:-/cephfs/huangzimeng/experiment_g/checkpoints/followup_t3_gsm8k_rl_200}"
T3_LOG_ROOT="${LOYAL_T3_LOG_ROOT:-/cephfs/huangzimeng/experiment_g/artifacts/experiments/five_followups/T3_gsm8k_rl_200}"
PYTHON="${LOYAL_PYTHON:-/root/experiment_g_runtime/conda/env/bin/python3}"

mkdir -p "${T3_LOG_ROOT}"
exec >>"${T3_LOG_ROOT}/supervisor.log" 2>&1
log() { printf '%s %s\n' "$(date -Is)" "$*"; }

validate_t2() {
  "${PYTHON}" - "${T2_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
expected = {"miu": 385, "eil": 656, "gsm8k": 1319, "math500": 500, "aime2026_sample16": 30}
for name, count in expected.items():
    path = root / name / "summary.json"
    if not path.is_file():
        raise SystemExit(1)
    summary = json.loads(path.read_text(encoding="utf-8"))
    if summary.get("n_total", summary.get("n_questions")) != count:
        raise SystemExit(1)
PY
}

while ! validate_t2; do
  log "waiting_t2_acceptance"
  sleep 120
done

while ps -eo args= | grep -E 'slime/train.py|ray::SGLangEngine|ray::MegatronTrainRayActor' | grep -v grep >/dev/null; do
  log "waiting_training_runtime_idle"
  sleep 60
done

[[ ! -e "${T3_ROOT}" ]] || { log "ERROR t3_output_exists path=${T3_ROOT}"; exit 2; }
log "t2_accepted_starting_t3"
export LOYAL_BASE_MODEL=qwen3-4b
export LOYAL_MODEL_ROOT="${LOYAL_MODEL_ROOT:-/cephfs/shared/experiment_g/assets/models}"
export LOYAL_GSM8K_RL_LOAD="${LOYAL_GSM8K_RL_LOAD:-${LOYAL_MODEL_ROOT}/Qwen3-4B_torch_dist/release}"
export LOYAL_GSM8K_RL_SAVE="${T3_ROOT}"
export LOYAL_GSM8K_RL_NUM_ROLLOUT=200
export LOYAL_GSM8K_RL_TRAIN_GPU_COUNT=2
export LOYAL_GSM8K_RL_ROLLOUT_GPU_COUNT=2
export LOYAL_GSM8K_RL_RAY_NUM_GPUS=4
export LOYAL_GSM8K_RL_GLOBAL_BATCH_SIZE=256
export LOYAL_GSM8K_RL_ROLLOUT_BATCH_SIZE=32
export LOYAL_GSM8K_RL_MAX_TOKENS_PER_GPU=4096
export LOYAL_GSM8K_RL_MAX_RESPONSE_LEN=2048
export LOYAL_GSM8K_RL_LEARNING_RATE=1e-6
export LOYAL_DATA_ROOT="${T3_LOG_ROOT}/slime_data"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export LOYAL_CONDA_SH="${LOYAL_CONDA_SH:-/root/miniforge3/etc/profile.d/conda.sh}"
export LOYAL_CONDA_ENV="${LOYAL_CONDA_ENV:-/root/experiment_g_runtime/conda/env}"
export LOYAL_MEGATRON_ROOT="${LOYAL_MEGATRON_ROOT:-/root/experiment_g_runtime/Megatron-LM}"
export LOYAL_RAY_TEMP_DIR="${LOYAL_RAY_TEMP_DIR:-/tmp/t3ray}"
export LOYAL_RAY_DIRECT_DRIVER=1
export LOYAL_RAY_STOP_BEFORE_START=1
export LOYAL_RAY_STOP_AFTER_EXIT=0
bash "${PROJECT_ROOT}/scripts/launch/run_training_host.sh" gsm8k_rl
log "t3_training_exit"
