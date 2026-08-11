#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="${LOYAL_JUDGE_PROXY_PID_FILE:-/tmp/loyal-judge-tcp-proxy.pid}"
LOG_FILE="${LOYAL_JUDGE_PROXY_LOG_FILE:-/tmp/loyal-judge-tcp-proxy.log}"
UPSTREAM_HOST="${1:?usage: $0 upstream-host}"

if [[ -r "${PID_FILE}" ]] && kill -0 "$(<"${PID_FILE}")" 2>/dev/null; then
  exit 0
fi

# Detach from the launcher so a short-lived shell does not terminate the relay.
setsid python3 "${SCRIPT_DIR}/judge_tcp_proxy.py" --upstream-host "${UPSTREAM_HOST}" \
  >>"${LOG_FILE}" 2>&1 < /dev/null &
echo $! >"${PID_FILE}"
