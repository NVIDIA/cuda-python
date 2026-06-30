# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
import threading
import types
from contextlib import contextmanager
from pathlib import Path

import pytest

_TEST_HELPERS_ROOT = Path(__file__).resolve().parents[3] / "cuda_python_test_helpers"
sys.path.insert(0, str(_TEST_HELPERS_ROOT))

from cuda_python_test_helpers.pytest_run_parallel import (
    install_run_parallel_worker_context_patch,
    mark_item_for_worker_context,
)


def _install_fake_pytest_run_parallel(monkeypatch):
    package = types.ModuleType("pytest_run_parallel")
    package.__path__ = []
    plugin = types.ModuleType("pytest_run_parallel.plugin")

    def wrap_function_parallel(fn, n_workers, n_iterations):
        raise AssertionError("unpatched fake wrapper should not be called")

    plugin.wrap_function_parallel = wrap_function_parallel
    monkeypatch.setitem(sys.modules, "pytest_run_parallel", package)
    monkeypatch.setitem(sys.modules, "pytest_run_parallel.plugin", plugin)
    return plugin


@pytest.mark.agent_authored(model="gpt-5")
def test_install_run_parallel_worker_context_patch_is_idempotent(monkeypatch):
    plugin = _install_fake_pytest_run_parallel(monkeypatch)

    assert install_run_parallel_worker_context_patch() is True
    patched = plugin.wrap_function_parallel
    assert patched._cuda_python_patched_run_parallel_worker_context

    assert install_run_parallel_worker_context_patch() is True
    assert plugin.wrap_function_parallel is patched


@pytest.mark.agent_authored(model="gpt-5")
def test_patched_wrapper_runs_context_with_isolated_kwargs(monkeypatch):
    plugin = _install_fake_pytest_run_parallel(monkeypatch)
    install_run_parallel_worker_context_patch()

    lock = threading.Lock()
    context_events = []
    calls = []

    @contextmanager
    def worker_context(*, thread_index, iteration_index, kwargs):
        token = object()
        kwargs["token"] = (thread_index, iteration_index, id(token))
        kwargs["context_kwargs_id"] = id(kwargs)
        with lock:
            context_events.append(("enter", thread_index, iteration_index, id(kwargs)))
        try:
            yield
        finally:
            with lock:
                context_events.append(("exit", thread_index, iteration_index, id(kwargs)))

    def test_body(*, thread_index, iteration_index, token, context_kwargs_id, static_value):
        with lock:
            calls.append(
                {
                    "thread_index": thread_index,
                    "iteration_index": iteration_index,
                    "token": token,
                    "context_kwargs_id": context_kwargs_id,
                    "static_value": static_value,
                }
            )

    item = types.SimpleNamespace(obj=test_body)
    assert mark_item_for_worker_context(item, worker_context) is True

    wrapped = plugin.wrap_function_parallel(item.obj, n_workers=3, n_iterations=2)
    wrapped(thread_index=-1, iteration_index=-1, static_value="fixture-value")

    expected_pairs = {(thread_index, iteration_index) for thread_index in range(3) for iteration_index in range(2)}
    actual_pairs = {(call["thread_index"], call["iteration_index"]) for call in calls}
    assert actual_pairs == expected_pairs
    assert {call["token"][:2] for call in calls} == expected_pairs
    assert {call["static_value"] for call in calls} == {"fixture-value"}

    kwargs_ids = {call["context_kwargs_id"] for call in calls}
    assert len(kwargs_ids) == 6
    assert len(context_events) == 12
    assert {event[3] for event in context_events} == kwargs_ids


@pytest.mark.agent_authored(model="gpt-5")
def test_mark_item_for_worker_context_wraps_callables_without_attrs():
    class CallableWithoutAttrs:
        __slots__ = ("calls",)

        def __init__(self):
            self.calls = []

        def __call__(self, **kwargs):
            self.calls.append(kwargs)

    @contextmanager
    def worker_context(*, thread_index, iteration_index, kwargs):
        kwargs["patched"] = True
        yield

    original = CallableWithoutAttrs()
    item = types.SimpleNamespace(obj=original)

    assert mark_item_for_worker_context(item, worker_context) is True
    assert item.obj is not original
    item.obj()
    assert original.calls == [{}]
