# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TEMPORARY: deliberate failures that exercise the CI failure-visibility changes.

REMOVE THIS FILE BEFORE MERGE.

Each test fails in a different way, so one CI run shows how the folded log, the
reordered short test summary, and the GitHub annotations render:

* a plain assertion in a parallel test (pytest-run-parallel reports it as ERROR)
* two failing cases of a parametrized test, with a passing case between them
* a thread-unsafe test that raises (the FAILED path, outside the parallel runner)
* an assertion with a multi-line diff
* a fixture error (ERROR at setup; the annotate plugin covers only the call phase)
* a failure preceded by a warning (the warning must not become an annotation)
* an exception raised inside a helper (two traceback frames)
"""

import warnings

import pytest


@pytest.mark.agent_authored(model="claude-fable-5-1")
def test_injected_plain_assert():
    assert 1 + 1 == 3


@pytest.mark.agent_authored(model="claude-fable-5-1")
@pytest.mark.parametrize("value", [1, 2, 3])
def test_injected_parametrized(value):
    assert value % 2 == 0, f"value {value} is odd"


@pytest.mark.agent_authored(model="claude-fable-5-1")
@pytest.mark.thread_unsafe(reason="injected: exercises the non-parallel FAILED path")
def test_injected_thread_unsafe():
    raise RuntimeError("injected failure outside the parallel runner")


@pytest.mark.agent_authored(model="claude-fable-5-1")
def test_injected_dict_diff():
    expected = {"a": 1, "b": 2, "c": 3}
    actual = {"a": 1, "b": 20, "d": 4}
    assert actual == expected


@pytest.fixture
def broken_fixture():
    raise ValueError("injected fixture error")


@pytest.mark.agent_authored(model="claude-fable-5-1")
def test_injected_setup_error(broken_fixture):
    pass


@pytest.mark.agent_authored(model="claude-fable-5-1")
@pytest.mark.thread_unsafe(reason="records process-global warnings")
def test_injected_warning_then_fail():
    warnings.warn("injected warning: must not become an annotation", UserWarning, stacklevel=1)
    raise AssertionError("injected failure after a warning")


@pytest.mark.agent_authored(model="claude-fable-5-1")
def test_injected_exception_in_helper():
    def helper(n):
        return 10 // n

    assert helper(0) == 1
