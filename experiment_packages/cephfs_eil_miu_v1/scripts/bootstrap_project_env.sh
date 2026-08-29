#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PKG_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
source "${SCRIPT_DIR}/common.sh"
PROJECT_ROOT="$(resolve_project_root)"
TARGET="${PROJECT_ROOT}/.env"

if [[ -e "${TARGET}" ]]; then
  echo "env already exists: ${TARGET}"
  exit 0
fi

cp "${PKG_ROOT}/env.example" "${TARGET}"
chmod 600 "${TARGET}" || true
echo "created ${TARGET}; fill evaluator endpoints and API keys before training"
