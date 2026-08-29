#!/usr/bin/env bash
set -euo pipefail

RUN_NAME="${LOYAL_RUN_NAME:-phase1}"
CONDITION="olmo3_e2m1_cephfs_rollout200"
CEPH_ROOT="${LOYAL_CEPHFS_ROOT:-/cephfs/shared/experiment_g/cephfs_eil_miu_v1}"
CHECKPOINT_NAME="cephfs-e2m1-${CONDITION}-${RUN_NAME}-seed1234"
CHECKPOINT_DIR="${LOYAL_CHECKPOINT_HOST_DIR:-${CEPH_ROOT}/checkpoints/${CHECKPOINT_NAME}}"
POST_ROOT="${LOYAL_POST_ROOT:-${CEPH_ROOT}/evaluations/${CONDITION}_posttrain}"
STEPS=(19 39 59 79 99 119 139 159 179 199)

test -s "${CHECKPOINT_DIR}/iter_0000019/common.pt"
test -f "${CHECKPOINT_DIR}/iter_0000019/.metadata"

missing=0
for step in "${STEPS[@]}"; do
  iter="$(printf "iter_%07d" "${step}")"
  [[ -s "${CHECKPOINT_DIR}/${iter}/common.pt" && -f "${CHECKPOINT_DIR}/${iter}/.metadata" ]] || missing=1
  [[ -f "${POST_ROOT}/checkpoint_eval/step${step}/miu_final/summary.json" ]] || missing=1
  [[ -f "${POST_ROOT}/checkpoint_eval/step${step}/eil_final/summary.json" ]] || missing=1
done

if [[ "${missing}" == "1" ]]; then
  echo "partial: first checkpoint exists, but full checkpoint/eval matrix is incomplete"
  exit 10
fi

echo "acceptance passed: all checkpoints and EIL/MIU summaries exist"
