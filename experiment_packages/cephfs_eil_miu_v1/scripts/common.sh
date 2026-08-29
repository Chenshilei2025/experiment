#!/usr/bin/env bash

resolve_project_root() {
  if [[ -n "${LOYAL_PROJECT_ROOT:-}" ]]; then
    printf '%s\n' "${LOYAL_PROJECT_ROOT}"
    return 0
  fi
  local script_dir pkg_root candidate
  script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
  pkg_root="$(cd -- "${script_dir}/.." && pwd)"
  for candidate in \
    "${pkg_root}/../.." \
    "${pkg_root}/../../loyal_agent_docker" \
    "${pkg_root}/../loyal_agent_docker" \
    "${pkg_root}/../../../loyal_agent_docker" \
    "${pkg_root}/../../../../loyal_agent_docker" \
    "/root/loyal_agent_docker" \
    "/root/experiment/loyal_agent_docker" \
    "/home/sherry/experiment_g/loyal_agent_docker"
  do
    if [[ -f "${candidate}/scripts/experiment_runner.py" ]]; then
      cd "${candidate}"
      pwd
      return 0
    fi
  done
  echo "cannot find loyal_agent_docker; set LOYAL_PROJECT_ROOT" >&2
  return 2
}
