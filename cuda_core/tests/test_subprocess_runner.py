# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the shared isolated-subprocess test helper.

Every snippet here is plain stdlib code, so no CUDA device is needed. These
pin the helper's own contract only; they do not exercise the call sites that
use it.
"""

import os
import subprocess
import textwrap

import pytest
from cuda_python_test_helpers.subprocess_runner import run_python_snippet


@pytest.mark.agent_authored(model="claude-opus-5")
def test_returns_captured_text_output(tmp_path):
    result = run_python_snippet("print('hello from child')", cwd=tmp_path)
    assert result.returncode == 0
    assert result.stdout.strip() == "hello from child"
    assert isinstance(result.stdout, str)


@pytest.mark.agent_authored(model="claude-opus-5")
def test_runs_in_the_requested_directory(tmp_path):
    result = run_python_snippet("import os; print(os.getcwd())", cwd=tmp_path)
    assert os.path.realpath(result.stdout.strip()) == os.path.realpath(tmp_path)


@pytest.mark.agent_authored(model="claude-opus-5")
def test_extra_env_reaches_the_child(tmp_path):
    result = run_python_snippet(
        "import os; print(os.environ['CUDA_PYTHON_TEST_SENTINEL'])",
        cwd=tmp_path,
        extra_env={"CUDA_PYTHON_TEST_SENTINEL": "set-by-caller"},
    )
    assert result.stdout.strip() == "set-by-caller"


@pytest.mark.agent_authored(model="claude-opus-5")
def test_unset_env_removes_an_inherited_variable(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_PYTHON_TEST_SENTINEL", "inherited")
    code = "import os; print(os.environ.get('CUDA_PYTHON_TEST_SENTINEL', '<absent>'))"

    inherited = run_python_snippet(code, cwd=tmp_path)
    assert inherited.stdout.strip() == "inherited"

    removed = run_python_snippet(code, cwd=tmp_path, unset_env=("CUDA_PYTHON_TEST_SENTINEL",))
    assert removed.stdout.strip() == "<absent>"


@pytest.mark.agent_authored(model="claude-opus-5")
def test_unset_env_is_applied_before_extra_env(tmp_path, monkeypatch):
    """A name in both must end up with the ``extra_env`` value, not removed."""
    monkeypatch.setenv("CUDA_PYTHON_TEST_SENTINEL", "inherited")
    result = run_python_snippet(
        "import os; print(os.environ['CUDA_PYTHON_TEST_SENTINEL'])",
        cwd=tmp_path,
        unset_env=("CUDA_PYTHON_TEST_SENTINEL",),
        extra_env={"CUDA_PYTHON_TEST_SENTINEL": "set-by-caller"},
    )
    assert result.stdout.strip() == "set-by-caller"


@pytest.mark.agent_authored(model="claude-opus-5")
def test_nonzero_exit_fails_and_reports_both_streams(tmp_path):
    code = textwrap.dedent(
        """
        import sys
        print("stdout marker")
        print("stderr marker", file=sys.stderr)
        sys.exit(3)
        """
    )
    with pytest.raises(AssertionError) as excinfo:
        run_python_snippet(code, cwd=tmp_path)
    message = str(excinfo.value)
    assert "exited with code 3" in message
    assert "stdout marker" in message
    assert "stderr marker" in message


@pytest.mark.agent_authored(model="claude-opus-5")
def test_check_false_returns_the_failed_process(tmp_path):
    result = run_python_snippet("raise SystemExit(4)", cwd=tmp_path, check=False)
    assert result.returncode == 4


@pytest.mark.agent_authored(model="claude-opus-5")
def test_timeout_is_enforced(tmp_path):
    with pytest.raises(subprocess.TimeoutExpired):
        run_python_snippet("import time; time.sleep(30)", cwd=tmp_path, timeout=1)


@pytest.mark.agent_authored(model="claude-opus-5")
def test_stdin_is_closed(tmp_path):
    result = run_python_snippet("import sys; print(repr(sys.stdin.read()))", cwd=tmp_path)
    assert result.stdout.strip() == "''"


@pytest.mark.agent_authored(model="claude-opus-5")
def test_empty_cwd_is_rejected():
    with pytest.raises(AssertionError):
        run_python_snippet("pass", cwd="")
