#!/usr/bin/env bash

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# Simple, dependency-free orchestrator to run tests for all packages.
# Usage:
#   scripts/run_tests.sh [ -v|--verbose ] [ --install | --no-install ] [ --with-cython | --skip-cython ] [ --with-examples | --skip-examples ] [ --with-ptds ]
#   scripts/run_tests.sh [ flags ]                   # pathfinder -> bindings -> core
#   scripts/run_tests.sh [ flags ] core              # only core
#   scripts/run_tests.sh [ flags ] bindings          # only bindings
#   scripts/run_tests.sh [ flags ] pathfinder        # only pathfinder
#   scripts/run_tests.sh [ flags ] smoke             # meta-level import smoke tests

repo_root=$(cd "$(dirname "$0")/.." && pwd)
cd "${repo_root}"


print_help() {
  cat <<'USAGE'
Usage: scripts/run_tests.sh [options] [target]

Targets:
  all (default)   Run pathfinder → bindings → core
  core            Run cuda_core tests
  bindings        Run cuda_bindings tests
  pathfinder      Run cuda_pathfinder tests
  smoke           Run meta-level smoke tests (tests/integration)

Options:
  -v, --verbose       Verbose pytest output (-ra -s -v)
      --install       Force editable install with [test] extras
      --no-install    Skip install checks (assume environment is ready)
      --with-cython   Build and run cython tests (needs CUDA_HOME for core)
      --skip-cython   Skip cython tests (default)
      --with-examples Run examples where applicable (e.g., cuda_bindings/examples)
      --skip-examples Skip running examples (default)
      --with-ptds     Re-run cuda_bindings tests with PTDS (CUDA_PYTHON_CUDA_PER_THREAD_DEFAULT_STREAM=1)
  -h, --help          Show this help and exit

Examples:
  scripts/run_tests.sh --install
  scripts/run_tests.sh --no-install core
  scripts/run_tests.sh -v --with-cython bindings
  scripts/run_tests.sh smoke
USAGE
}

# Parse optional flags
VERBOSE=0
RUN_CYTHON=0
RUN_EXAMPLES=1
RUN_PTDS=1
INSTALL_MODE=auto  # auto|force|skip
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      print_help
      exit 0
      ;;
    -v|--verbose)
      VERBOSE=1
      shift
      ;;
    --install)
      INSTALL_MODE=force
      shift
      ;;
    --no-install)
      INSTALL_MODE=skip
      shift
      ;;
    --with-cython)
      RUN_CYTHON=1
      shift
      ;;
    --skip-cython)
      RUN_CYTHON=0
      shift
      ;;
    --with-examples)
      RUN_EXAMPLES=1
      shift
      ;;
    --skip-examples)
      RUN_EXAMPLES=0
      shift
      ;;
    --with-ptds)
      RUN_PTDS=1
      shift
      ;;
    *)
      break
      ;;
  esac
done

target=${1:-all}

if [[ ${VERBOSE} -eq 1 ]]; then
  PYTEST_FLAGS=( -ra -s -v )
else
  # Very quiet: show failures/errors summary only
  PYTEST_FLAGS=( -qq )
fi

declare -A RESULTS
ORDERED_RESULTS=()

add_result() {
  local name="$1"; shift
  local rc="$1"; shift
  RESULTS["${name}"]="${rc}"
  ORDERED_RESULTS+=("${name}")
}

status_from_rc() {
  local rc="$1"
  case "${rc}" in
    0) echo "PASS" ;;
    5) echo "SKIP(no-tests)" ;;
    1) echo "FAIL" ;;
    2) echo "INTERRUPTED" ;;
    3) echo "ERROR" ;;
    4) echo "USAGE" ;;
    *) echo "RC=${rc}" ;;
  esac
}

run_pytest() {
  # Run pytest safely under set -e and return its exit code
  set +e
  python -m pytest "${PYTEST_FLAGS[@]}" "$@"
  local rc=$?
  set -e
  return ${rc}
}

run_pytest_ptds() {
  # Run pytest with PTDS env set; safely return its exit code
  set +e
  CUDA_PYTHON_CUDA_PER_THREAD_DEFAULT_STREAM=1 python -m pytest "${PYTEST_FLAGS[@]}" "$@"
  local rc=$?
  set -e
  return ${rc}
}

ensure_installed() {
  # Args: module.import.name repo_subdir
  local mod_name="$1"; shift
  local subdir_name="$1"; shift

  if [[ "${INSTALL_MODE}" == "skip" ]]; then
    return 0
  fi

  if [[ "${INSTALL_MODE}" == "force" ]]; then
    pip install -e .[test]
    return 0
  fi

  # auto-detect: if module imports from this repo, assume installed; otherwise install
  python - <<PY 2>/dev/null
import importlib, sys, pathlib
mod = "${mod_name}"
try:
    m = importlib.import_module(mod)
except Exception:
    sys.exit(2)
p = pathlib.Path(getattr(m, "__file__", "")).resolve()
root = pathlib.Path(r"${repo_root}").resolve()
sub = pathlib.Path(r"${repo_root}/${subdir_name}").resolve()
sys.exit(0 if str(p).startswith(str(sub)) else 3)
PY
  rc=$?
  if [[ $rc -ne 0 ]]; then
    pip install -e .[test]
  fi
}

run_pathfinder() {
  echo "[tests] cuda_pathfinder"
  cd "${repo_root}/cuda_pathfinder"
  ensure_installed "cuda.pathfinder" "cuda_pathfinder"
  run_pytest tests/
  local rc=$?
  add_result "pathfinder" "${rc}"
}

run_bindings() {
  echo "[tests] cuda_bindings"
  cd "${repo_root}/cuda_bindings"
  ensure_installed "cuda.bindings" "cuda_bindings"
  run_pytest tests/
  local rc=$?
  add_result "bindings" "${rc}"
  if [ ${RUN_PTDS} -eq 1 ]; then
    echo "[tests] cuda_bindings (PTDS)"
    run_pytest_ptds tests/
    local rc_ptds=$?
    add_result "bindings-ptds" "${rc_ptds}"
  fi
  if [ ${RUN_EXAMPLES} -eq 1 ] && [ -d examples ]; then
    # Bindings examples are pytest-based (contain their own pytest.ini)
    echo "[examples] cuda_bindings/examples"
    run_pytest examples/
    local rc_ex=$?
    add_result "bindings-examples" "${rc_ex}"
  fi
  if [ ${RUN_CYTHON} -eq 1 ] && [ -d tests/cython ]; then
    if [ -x tests/cython/build_tests.sh ]; then
      echo "[build] cuda_bindings cython tests"
      ( cd tests/cython && ./build_tests.sh ) || true
    fi
    run_pytest tests/cython
    local rc_cy=$?
    add_result "bindings-cython" "${rc_cy}"
  fi
}

run_core() {
  echo "[tests] cuda_core"
  cd "${repo_root}/cuda_core"
  ensure_installed "cuda.core" "cuda_core"
  run_pytest tests/
  local rc=$?
  add_result "core" "${rc}"
  if [ ${RUN_EXAMPLES} -eq 1 ] && [ -d examples ] && [ -f examples/pytest.ini ]; then
    # Only run examples under pytest if they are configured as tests
    echo "[examples] cuda_core/examples"
    run_pytest examples/
    local rc_ex=$?
    add_result "core-examples" "${rc_ex}"
  fi
  if [ ${RUN_CYTHON} -eq 1 ] && [ -d tests/cython ]; then
    if [ -x tests/cython/build_tests.sh ]; then
      echo "[build] cuda_core cython tests"
      if [ -z "${CUDA_HOME-}" ]; then
        echo "[skip] CUDA_HOME not set; skipping cython tests"
      else
        ( cd tests/cython && ./build_tests.sh ) || true
      fi
    fi
    run_pytest tests/cython
    local rc_cy=$?
    add_result "core-cython" "${rc_cy}"
  fi
}

run_smoke() {
  echo "[tests] meta-level smoke"
  cd "${repo_root}"
  python - <<PY 2>/dev/null || pip install pytest>=6.2.4
import pytest
PY
  run_pytest tests/integration
  local rc=$?
  add_result "smoke" "${rc}"
}

case "${target}" in
  all)
    run_pathfinder
    run_bindings
    run_core
    ;;
  core)
    run_core ;;
  bindings)
    run_bindings ;;
  pathfinder)
    run_pathfinder ;;
  smoke)
    run_smoke ;;
  *)
    echo "Unknown target: ${target}" >&2
    exit 1
    ;;
esac

# Print summary
echo
echo "==================== Test Summary ===================="
overall_rc=0
if [ -t 1 ]; then
  GREEN=$(printf '\033[32m')
  RED=$(printf '\033[31m')
  RESET=$(printf '\033[0m')
else
  GREEN=""; RED=""; RESET=""
fi
for name in "${ORDERED_RESULTS[@]}"; do
  rc="${RESULTS[$name]}"
  status=$(status_from_rc "${rc}")
  color=""
  case "${status}" in
    PASS) color="${GREEN}" ;;
    FAIL|ERROR|INTERRUPTED|USAGE|RC=*) color="${RED}" ;;
    *) color="" ;;
  esac
  printf "%-18s : %s%s%s\n" "${name}" "${color}" "${status}" "${RESET}"
  if [[ "${rc}" -ne 0 && "${rc}" -ne 5 ]]; then
    overall_rc=1
  fi
done
echo "======================================================"
exit ${overall_rc}
