# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the ``init_cuda`` teardown gc.collect() gate and its warnings.

``conftest.py`` snapshots the live owned-resource counts (owned memory pools and
VA reservations) at ``init_cuda`` setup and, at teardown, collects only when a
count differs from that snapshot. It also warns when a test leaves a resource
alive without ``init_cuda`` (``UnreleasedMemoryResourceWarning``, raised at
setup) or releases a resource that existed before it started
(``BaselineMemoryResourceReleasedWarning``, raised at teardown).

These behaviors live in the fixture's setup/teardown, so they cannot be observed
from an ordinary test body. Each test drives the *real* ``init_cuda`` generator
by hand with a minimal fake ``request``, and scripts ``conftest._live_resource_counts``
so the gate's decision is exercised deterministically without allocating driver
resources (that the counts move correctly is covered by
``test_resource_live_count.py``). The fixture function and its module globals are
reached through pytest's fixture registry (``FixtureDef.func``) so the
monkeypatched globals are the ones the running code reads. Tests are
``thread_unsafe`` because they patch process-global state (``gc.collect`` and the
conftest module globals).
"""

import gc
import inspect

import pytest
from helpers.memory import (
    BaselineMemoryResourceReleasedWarning,
    UnreleasedMemoryResourceWarning,
)


class _ScriptedCounts:
    """Callable stand-in for ``conftest._live_resource_counts``.

    Returns the queued ``(owned_pools, va_reservations)`` tuples in order and
    sticks on the last one, so the gate may read the counts more times than
    scripted without raising.
    """

    def __init__(self, *values):
        self._values = list(values)
        self._i = 0

    def __call__(self):
        value = self._values[min(self._i, len(self._values) - 1)]
        self._i += 1
        return value


_THREAD_UNSAFE_MARKER = pytest.mark.thread_unsafe(
    reason="patches gc.collect and conftest module globals (the resource-count hook and snapshots)"
)


class _FakeOption:
    randomly_seed = None


class _FakeConfig:
    def __init__(self):
        self.option = _FakeOption()


class _FakeNode:
    def __init__(self, nodeid):
        self.nodeid = nodeid
        self.funcargs = {}


class _FakeRequest:
    """Minimal stand-in for the pytest request that ``init_cuda`` reads."""

    def __init__(self, nodeid="manual::init_cuda_gate_probe"):
        self.node = _FakeNode(nodeid)
        self.config = _FakeConfig()


def _init_cuda_func(request):
    """Return the raw ``init_cuda`` generator function from the fixture registry.

    ``FixtureDef.func`` is the undecorated fixture function, and its
    ``__globals__`` is the live conftest namespace, so monkeypatching keys there
    affects the code these tests drive. Both are pytest internals; the asserts
    below fail loudly if a pytest change breaks the assumption.
    """
    fixturedefs = request._fixturemanager.getfixturedefs("init_cuda", request.node)
    assert fixturedefs, "init_cuda fixture not found in the registry"
    func = fixturedefs[-1].func
    assert inspect.isgeneratorfunction(func), "init_cuda is expected to be a generator fixture"
    return func


def _spy_on_gc_collect(monkeypatch):
    """Patch ``gc.collect`` to record each explicit call; returns the call list.

    Automatic generational collections do not route through ``gc.collect``, so
    the list counts only the fixture's explicit collects.
    """
    calls = []
    real_collect = gc.collect

    def spy(*args, **kwargs):
        calls.append(1)
        return real_collect(*args, **kwargs)

    monkeypatch.setattr(gc, "collect", spy)
    return calls


def _finish_teardown(gen):
    with pytest.raises(StopIteration):
        next(gen)


# --- 2. the collect gate -------------------------------------------------------


@_THREAD_UNSAFE_MARKER
@pytest.mark.agent_authored(model="claude-opus-4.8")
def test_teardown_skips_collect_when_counts_unchanged(request, monkeypatch):
    func = _init_cuda_func(request)
    g = func.__globals__
    # Disable the leftover check so only the teardown gate can call gc.collect,
    # and script equal setup/teardown counts.
    monkeypatch.setitem(g, "_mr_expected_live", None)
    monkeypatch.setitem(g, "_live_resource_counts", _ScriptedCounts((2, 1)))
    collects = _spy_on_gc_collect(monkeypatch)

    gen = func(_FakeRequest())
    next(gen)  # setup: live_at_setup == (2, 1)
    before = len(collects)
    _finish_teardown(gen)  # teardown: live_after == (2, 1) -> no collect

    assert len(collects) == before, "gate collected despite unchanged counts"


@_THREAD_UNSAFE_MARKER
@pytest.mark.agent_authored(model="claude-opus-4.8")
def test_teardown_collects_when_a_count_grows(request, monkeypatch):
    func = _init_cuda_func(request)
    g = func.__globals__
    monkeypatch.setitem(g, "_mr_expected_live", None)
    # Owned-pool count grows from 2 to 3 across the test.
    monkeypatch.setitem(g, "_live_resource_counts", _ScriptedCounts((2, 1), (3, 1)))
    collects = _spy_on_gc_collect(monkeypatch)

    gen = func(_FakeRequest())
    next(gen)  # setup: live_at_setup == (2, 1)
    before = len(collects)
    _finish_teardown(gen)  # teardown: live_after == (3, 1) -> collect

    assert len(collects) == before + 1, "gate skipped the collect despite a grown count"


# --- 3. the two warnings -------------------------------------------------------


@_THREAD_UNSAFE_MARKER
@pytest.mark.agent_authored(model="claude-opus-4.8")
def test_setup_warns_when_resources_left_alive_since_last_init_cuda(request, monkeypatch):
    func = _init_cuda_func(request)
    g = func.__globals__
    # Force the leftover-check precondition: the item started with one more owned
    # pool than the baseline recorded after the previous init_cuda test.
    monkeypatch.setitem(g, "_mr_expected_live", (0, 0))
    monkeypatch.setitem(g, "_mr_live_at_item_start", (1, 0))
    monkeypatch.setitem(g, "_mr_last_init_cuda_nodeid", "tests/prev::test_prev")
    monkeypatch.setitem(g, "_live_resource_counts", _ScriptedCounts((1, 0)))

    gen = func(_FakeRequest())
    with pytest.warns(UnreleasedMemoryResourceWarning, match="left alive"):
        next(gen)  # setup emits the leftover warning
    _finish_teardown(gen)  # live_after == live_at_setup -> no baseline warning


@_THREAD_UNSAFE_MARKER
@pytest.mark.agent_authored(model="claude-opus-4.8")
def test_teardown_warns_on_baseline_release(request, monkeypatch):
    func = _init_cuda_func(request)
    g = func.__globals__
    monkeypatch.setitem(g, "_mr_expected_live", None)  # disable the leftover check
    # Script a snapshot of (1, 0) at setup and (0, 0) at teardown: an owned pool
    # that existed at setup was released during the test.
    monkeypatch.setitem(g, "_live_resource_counts", _ScriptedCounts((1, 0), (0, 0)))

    gen = func(_FakeRequest())
    next(gen)  # setup: live_at_setup == (1, 0)
    with (
        pytest.warns(BaselineMemoryResourceReleasedWarning, match="existed before it started"),
        pytest.raises(StopIteration),
    ):
        next(gen)  # teardown: live_after == (0, 0) -> baseline-release warning
