#!/usr/bin/env bash
# Evaluate selected mixed-reward checkpoints with 4-way MIU/EIL sharding.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

PROJECT_ROOT="${LOYAL_DIRECT_EVAL_PROJECT_ROOT:-${REPO_ROOT}}"
CHECKPOINT_NAME="${LOYAL_DIRECT_EVAL_CHECKPOINT_NAME:-mixed-v2-phase1-lambda075-e1m1-phase1-seed1234}"
CHECKPOINT_ROOT="${LOYAL_DIRECT_EVAL_CHECKPOINT_ROOT:-${PROJECT_ROOT}/artifacts/checkpoints/${CHECKPOINT_NAME}}"
EXPORT_ROOT="${LOYAL_DIRECT_EVAL_EXPORT_ROOT:-${PROJECT_ROOT}/artifacts/exported_models/direct_checkpoint_eval}"
EVAL_ROOT="${LOYAL_DIRECT_EVAL_OUTPUT_ROOT:-${PROJECT_ROOT}/artifacts/evaluations/direct_checkpoint_eval}"
PYTHON="${LOYAL_PYTHON:-/root/experiment_g_runtime/conda/env/bin/python3}"
STEPS="${LOYAL_DIRECT_EVAL_STEPS:-19 39 59 79 99 119 139}"
LOG_FILE="${LOYAL_DIRECT_EVAL_LOG_FILE:-/tmp/direct_checkpoint_eval.log}"

exec >>"${LOG_FILE}" 2>&1

log() {
  printf '%s %s\n' "$(date -Is)" "$*"
}

load_env() {
  cd "${PROJECT_ROOT}"
  export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
  set +u
  # shellcheck disable=SC1091
  source "${PROJECT_ROOT}/scripts/launch/env.sh"
  set -u
}

assert_no_training() {
  if ps -eo pid=,ppid=,args= | grep -E 'slime/train.py|scripts/launch/run-mixed.sh|watch_phase1_lambda075_to159' | grep -v grep >/tmp/direct_eval_train_guard.txt; then
    log "ERROR training_process_detected; refusing_to_start_eval"
    cat /tmp/direct_eval_train_guard.txt
    exit 2
  fi
}

export_step() {
  local step="$1"
  local iter
  iter="iter_$(printf '%07d' "${step}")"
  local export_dir="${EXPORT_ROOT}/${CHECKPOINT_NAME}/${iter}"
  if [[ -f "${export_dir}/config.json" && -f "${export_dir}/model.safetensors.index.json" ]]; then
    log "export_skip step=${step} dir=${export_dir}"
    return 0
  fi
  log "export_start step=${step}"
  LOYAL_CHECKPOINT_HOST_DIR="${CHECKPOINT_ROOT}" \
  LOYAL_SHARED_CHECKPOINT_NAME="${CHECKPOINT_NAME}" \
  LOYAL_EXPORT_ROOT="${EXPORT_ROOT}" \
    bash "${PROJECT_ROOT}/scripts/export_final_checkpoint_host.sh" "${CHECKPOINT_NAME}" "${step}"
  [[ -f "${export_dir}/config.json" ]] || { log "ERROR export_missing_config step=${step}"; exit 3; }
  log "export_done step=${step} dir=${export_dir}"
}

aggregate_mechanism() {
  local mechanism="$1"
  local step="$2"
  local expected=385
  [[ "${mechanism}" == "eil" ]] && expected=656
  "${PYTHON}" - "${mechanism}" "${step}" "${EVAL_ROOT}/step${step}/${mechanism}" "${EVAL_ROOT}/step${step}/${mechanism}_final" "${expected}" <<'PY'
from __future__ import annotations
import json
import sys
from pathlib import Path
from scripts.evaluation import eil, miu

mechanism = sys.argv[1]
step = sys.argv[2]
shard_root = Path(sys.argv[3])
final_dir = Path(sys.argv[4])
expected = int(sys.argv[5])
module = miu if mechanism == "miu" else eil
rows_by_id = {}
sources = []
for shard in range(4):
    path = shard_root / f"shard_{shard}" / "per_sample.jsonl"
    if not path.is_file():
        raise RuntimeError(f"missing shard output: {path}")
    sources.append(str(path))
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        row = json.loads(line)
        row_id = row["id"]
        if row_id in rows_by_id:
            raise RuntimeError(f"duplicate id {row_id} from {path}")
        rows_by_id[row_id] = row
if len(rows_by_id) != expected:
    raise RuntimeError(f"{mechanism} step {step}: expected {expected} unique rows, got {len(rows_by_id)}")
rows = [rows_by_id[row_id] for row_id in sorted(rows_by_id)]
final_dir.mkdir(parents=True, exist_ok=True)
(final_dir / "per_sample.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
summary = module.summarize(rows, run={"step": step, "source_shards": sources, "aggregation": "4 shards merged"})
(final_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"event": "aggregate_done", "step": step, "mechanism": mechanism, "n_total": summary.get("n_total"), "reward_mean": summary.get("reward_mean")}, ensure_ascii=False), flush=True)
PY
}

rewrite_metrics_table() {
  "${PYTHON}" - "${EVAL_ROOT}" <<'PY'
from __future__ import annotations
import csv
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
fields = [
    "step",
    "miu_reward_mean",
    "miu_policy_output_valid_rate",
    "miu_decision_exact_match_rate",
    "miu_reasoning_faithfulness_mean",
    "miu_n_total",
    "eil_reward_mean",
    "eil_task_utility_mean",
    "eil_leakage_mean",
    "eil_leakage_zero_rate",
    "eil_n_total",
    "eil_n_failed",
]
rows = []
for step in [19, 39, 59, 79, 99, 119, 139]:
    step_dir = root / f"step{step}"
    miu_path = step_dir / "miu_final" / "summary.json"
    eil_path = step_dir / "eil_final" / "summary.json"
    if not (miu_path.is_file() and eil_path.is_file()):
        continue
    miu = json.loads(miu_path.read_text(encoding="utf-8"))
    eil = json.loads(eil_path.read_text(encoding="utf-8"))
    rows.append({
        "step": step,
        "miu_reward_mean": miu.get("reward_mean"),
        "miu_policy_output_valid_rate": miu.get("policy_output_valid_rate"),
        "miu_decision_exact_match_rate": miu.get("decision_exact_match_rate"),
        "miu_reasoning_faithfulness_mean": miu.get("reasoning_faithfulness_mean"),
        "miu_n_total": miu.get("n_total"),
        "eil_reward_mean": eil.get("reward_mean"),
        "eil_task_utility_mean": eil.get("task_utility_mean"),
        "eil_leakage_mean": eil.get("leakage_mean"),
        "eil_leakage_zero_rate": eil.get("leakage_zero_rate"),
        "eil_n_total": eil.get("n_total"),
        "eil_n_failed": eil.get("n_failed"),
    })
with (root / "checkpoint_metrics.csv").open("w", newline="", encoding="utf-8") as fh:
    writer = csv.DictWriter(fh, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
(root / "checkpoint_metrics.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"event": "metrics_table_written", "rows": len(rows), "csv": str(root / "checkpoint_metrics.csv")}, ensure_ascii=False), flush=True)
PY
}

run_mechanism() {
  local mechanism="$1"
  local step="$2"
  local iter
  iter="iter_$(printf '%07d' "${step}")"
  local export_dir="${EXPORT_ROOT}/${CHECKPOINT_NAME}/${iter}"
  local out_root="${EVAL_ROOT}/step${step}/${mechanism}"
  local final_dir="${EVAL_ROOT}/step${step}/${mechanism}_final"
  local batch_size=8
  local max_new_tokens=384
  [[ "${mechanism}" == "eil" ]] && batch_size=2 && max_new_tokens=2048
  if [[ -f "${final_dir}/summary.json" && -f "${final_dir}/per_sample.jsonl" ]]; then
    log "${mechanism}_final_skip step=${step}"
    return 0
  fi
  mkdir -p "${out_root}"
  local pids=()
  for shard in 0 1 2 3; do
    local shard_dir="${out_root}/shard_${shard}"
    if [[ -f "${shard_dir}/summary.json" && -f "${shard_dir}/per_sample.jsonl" ]]; then
      log "${mechanism}_shard_skip step=${step} shard=${shard}"
      continue
    fi
    log "${mechanism}_shard_start step=${step} shard=${shard} gpu=${shard}"
    if [[ "${mechanism}" == "miu" ]]; then
      CUDA_VISIBLE_DEVICES="${shard}" LOYAL_GPU_MEMORY_FRACTION=0.25 "${PYTHON}" -m scripts.evaluation.cli miu \
        --checkpoint "${export_dir}" --output-dir "${shard_dir}" --device cuda:0 \
        --batch-size "${batch_size}" --max-new-tokens "${max_new_tokens}" \
        --shard-index "${shard}" --num-shards 4 >"${out_root}/shard_${shard}.log" 2>&1 &
    else
      CUDA_VISIBLE_DEVICES="${shard}" LOYAL_GPU_MEMORY_FRACTION=0.25 "${PYTHON}" -m scripts.evaluation.cli eil \
        --checkpoint "${export_dir}" --output-dir "${shard_dir}" --device cuda:0 \
        --batch-size "${batch_size}" --max-new-tokens "${max_new_tokens}" --score-concurrency 1 \
        --shard-index "${shard}" --num-shards 4 >"${out_root}/shard_${shard}.log" 2>&1 &
    fi
    pids+=("$!")
  done
  local failed=0
  for pid in "${pids[@]}"; do
    wait "${pid}" || failed=1
  done
  if [[ "${failed}" -ne 0 ]]; then
    log "ERROR ${mechanism}_shards_failed step=${step}"
    for shard in 0 1 2 3; do
      log "tail ${mechanism} step=${step} shard=${shard}"
      tail -n 80 "${out_root}/shard_${shard}.log" 2>/dev/null || true
    done
    exit 4
  fi
  aggregate_mechanism "${mechanism}" "${step}"
}

main() {
  log "direct_checkpoint_eval_started pid=$$"
  assert_no_training
  load_env
  mkdir -p "${EXPORT_ROOT}" "${EVAL_ROOT}"
  for step in ${STEPS}; do
    assert_no_training
    log "step_start step=${step}"
    export_step "${step}"
    run_mechanism miu "${step}"
    run_mechanism eil "${step}"
    rewrite_metrics_table
    log "step_done step=${step}"
  done
  log "direct_checkpoint_eval_complete"
}

main "$@"
