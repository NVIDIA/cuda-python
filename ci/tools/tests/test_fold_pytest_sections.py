# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import io
import os
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fold_pytest_sections import fold

SCRIPT = os.path.join(os.path.dirname(__file__), "..", "fold_pytest_sections.py")

AGENT = pytest.mark.agent_authored(model="claude-fable-5-1")


def run_fold(text: str) -> str:
    out = io.BytesIO()
    fold(io.BytesIO(text.encode()), out)
    return out.getvalue().decode()


def strip_groups(text: str) -> str:
    return "".join(
        line for line in text.splitlines(keepends=True) if not line.startswith(("::group::", "::endgroup::"))
    )


# The tail of a failed run under ``-rsxXfE -v --durations=20 --parallel-threads=N``.
FAILED_RUN = """\
tests/test_a.py::test_ok PARALLEL PASSED [ 50%]
tests/test_a.py::test_bad PARALLEL FAILED [100%]

==================================== ERRORS ====================================
____________________________ ERROR at call of test_bad _________________________
E       assert False
tests/test_a.py:7: AssertionError
=============================== warnings summary ===============================
tests/test_a.py::test_ok
  DeprecationWarning: old
============================= slowest 20 durations =============================
0.01s call     tests/test_a.py::test_ok
========================== pytest-run-parallel report ==========================
tests/test_a.py::test_ok was not run in parallel: uses monkeypatch
=========================== short test summary info ============================
SKIPPED [2] tests/test_b.py:10: no second GPU
XFAIL tests/test_c.py::test_known - known bug
FAILED ([thread-unsafe]: patches globals) tests/test_d.py::test_unsafe
PARALLEL FAILED tests/test_a.py::test_bad - assert False
ERROR tests/test_e.py::test_setup - ValueError: fixture
===== 1 failed, 1 passed, 2 skipped, 1 xfailed, 1 warning, 2 errors in 1.23s =====
"""

FAILED_RUN_FOLDED = """\
tests/test_a.py::test_ok PARALLEL PASSED [ 50%]
tests/test_a.py::test_bad PARALLEL FAILED [100%]

==================================== ERRORS ====================================
____________________________ ERROR at call of test_bad _________________________
E       assert False
tests/test_a.py:7: AssertionError
::group::warnings summary
=============================== warnings summary ===============================
tests/test_a.py::test_ok
  DeprecationWarning: old
::endgroup::
::group::slowest 20 durations
============================= slowest 20 durations =============================
0.01s call     tests/test_a.py::test_ok
::endgroup::
::group::pytest-run-parallel report
========================== pytest-run-parallel report ==========================
tests/test_a.py::test_ok was not run in parallel: uses monkeypatch
::endgroup::
::group::short test summary info
=========================== short test summary info ============================
SKIPPED [2] tests/test_b.py:10: no second GPU
XFAIL tests/test_c.py::test_known - known bug
::endgroup::
FAILED ([thread-unsafe]: patches globals) tests/test_d.py::test_unsafe
PARALLEL FAILED tests/test_a.py::test_bad - assert False
ERROR tests/test_e.py::test_setup - ValueError: fixture
===== 1 failed, 1 passed, 2 skipped, 1 xfailed, 1 warning, 2 errors in 1.23s =====
"""


@AGENT
def test_failed_run_keeps_tracebacks_and_failure_lines_outside_the_folds():
    assert run_fold(FAILED_RUN) == FAILED_RUN_FOLDED


@AGENT
def test_green_run_closes_the_skip_list_at_the_final_count():
    log = (
        "=========================== short test summary info ============================\n"
        "SKIPPED [1] tests/test_b.py:10: no second GPU\n"
        "================== 10 passed, 1 skipped in 0.50s ==================\n"
    )
    assert run_fold(log) == (
        "::group::short test summary info\n"
        "=========================== short test summary info ============================\n"
        "SKIPPED [1] tests/test_b.py:10: no second GPU\n"
        "::endgroup::\n"
        "================== 10 passed, 1 skipped in 0.50s ==================\n"
    )


@AGENT
def test_sections_outside_the_fold_list_end_an_open_group_and_stay_visible():
    log = (
        "=============================== warnings summary ===============================\n"
        "  DeprecationWarning: old\n"
        "========================= cuda_core OOM diagnostics =========================\n"
        "first CUDA OOM at tests/test_x.py::test_y\n"
        "=========================== short test summary info ============================\n"
        "FAILED tests/test_x.py::test_y - CUDA_ERROR_OUT_OF_MEMORY\n"
        "============================== 1 failed in 2.00s ===============================\n"
    )
    assert run_fold(log) == (
        "::group::warnings summary\n"
        "=============================== warnings summary ===============================\n"
        "  DeprecationWarning: old\n"
        "::endgroup::\n"
        "========================= cuda_core OOM diagnostics =========================\n"
        "first CUDA OOM at tests/test_x.py::test_y\n"
        "::group::short test summary info\n"
        "=========================== short test summary info ============================\n"
        "::endgroup::\n"
        "FAILED tests/test_x.py::test_y - CUDA_ERROR_OUT_OF_MEMORY\n"
        "============================== 1 failed in 2.00s ===============================\n"
    )


@AGENT
def test_colored_banners_are_recognized_and_written_back_unchanged():
    banner = "\x1b[33m=============================== warnings summary ===============================\x1b[0m\n"
    log = banner + "  DeprecationWarning: old\n"
    assert run_fold(log) == "::group::warnings summary\n" + log + "::endgroup::\n"


@AGENT
def test_final_warnings_section_is_folded():
    log = "=========================== warnings summary (final) ===========================\n  ResourceWarning: x\n"
    assert run_fold(log) == "::group::warnings summary (final)\n" + log + "::endgroup::\n"


@AGENT
def test_truncated_log_closes_the_open_group():
    log = "============================= slowest 20 durations =============================\n0.01s call test\n"
    assert run_fold(log).endswith("0.01s call test\n::endgroup::\n")


@AGENT
@pytest.mark.parametrize("newline", ["\n", "\r\n"], ids=["lf", "crlf"])
def test_only_group_markers_are_added(newline):
    log = FAILED_RUN.replace("\n", newline)
    folded = run_fold(log)
    assert strip_groups(folded) == log
    assert folded.count("::group::") == folded.count("::endgroup::") == 4


@AGENT
def test_cli_filters_stdin_to_stdout():
    result = subprocess.run(  # noqa: S603 - invokes the repository script under test
        [sys.executable, SCRIPT],
        input=FAILED_RUN.encode(),
        capture_output=True,
        check=True,
    )
    assert result.stdout.decode() == FAILED_RUN_FOLDED
    assert result.stderr == b""
