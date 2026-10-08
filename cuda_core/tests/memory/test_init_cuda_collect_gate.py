# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the ``init_cuda`` teardown gc.collect() gate and its warnings.

``conftest.py`` snapshots the live ``MemoryResource`` count at ``init_cuda``
setup and, at teardown, collects only when the count differs from that
snapshot. It also warns when a test leaves a resource alive without
``init_cuda`` (``UnreleasedMemoryResourceWarning``, raised at setup) or releases
a resource that existed before it started (``BaselineMemoryResourceReleasedWarning``,
raised at teardown).

These behaviors live in the fixture's setup/teardown, so they cannot be observed
from an ordinary test body. Each test here drives the *real* ``init_cuda``
generator by hand with a minimal fake ``request`` and (where needed) a
``gc.collect`` spy, so the collect decision and both warnings can be asserted
directly. The fixture function and its module globals are reached through
pytest's fixture registry (``FixtureDef.func``) so the monkeypatched globals are
the ones the running code reads. Tests are ``thread_unsafe`` because they patch
process-global state (``gc.collect`` and the conftest module globals) and assert
on the process-global resource count; each cleans up every resource it creates
so the real session's leftover tracking is not perturbed.
"""

import gc
import inspect

import pytest
from helpers.memory import (
    BaselineMemoryResourceReleasedWarning,
    UnreleasedMemoryResourceWarning,
)

from cuda.core import MemoryResource
from cuda.core._memory._buffer import _live_memory_resource_count


class _LocalMR(MemoryResource):
    """A trivial MemoryResource subclass that holds no CUDA pool.

    Construction/destruction only moves the live counter, so it is a clean way
    to make the gate's snapshot comparison grow or shrink without allocating.
    """


_THREAD_UNSAFE_MARKER = pytest.mark.thread_unsafe(
    reason="patches gc.collect and conftest module globals, and asserts on the process-global resource count"
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


# --- 2. the collect gate -------------------------------------------------------


@_THREAD_UNSAFE_MARKER
@pytest.mark.agent_authored(model="claude-opus-4.8")
def test_teardown_skips_collect_when_mr_count_unchanged(request, monkeypatch):
    func = _init_cuda_func(request)
    # Disable the leftover check so only the teardown gate can call gc.collect.
    monkeypatch.setitem(func.__globals__, "_mr_expected_live", None)
    collects = _spy_on_gc_collect(monkeypatch)

    gen = func(_FakeRequest())
    next(gen)  # setup: records the snapshot
    before = len(collects)
    # A resource created and freed by refcount leaves the count at the snapshot.
    mr = _LocalMR()
    del mr
    with pytest.raises(StopIteration):
        next(gen)  # teardown runs the gate

    assert len(collects) == before, "gate collected despite an unchanged count"


@_THREAD_UNSAFE_MARKER
@pytest.mark.agent_authored(model="claude-opus-4.8")
def test_teardown_collects_when_mr_count_grows(request, monkeypatch):
    func = _init_cuda_func(request)
    monkeypatch.setitem(func.__globals__, "_mr_expected_live", None)
    collects = _spy_on_gc_collect(monkeypatch)

    gen = func(_FakeRequest())
    next(gen)  # setup: records the snapshot
    before = len(collects)
    # A resource still alive at teardown lifts the count above the snapshot.
    mr = _LocalMR()
    try:
        with pytest.raises(StopIteration):
            next(gen)  # teardown runs the gate
        assert len(collects) == before + 1, "gate skipped the collect despite a grown count"
    finally:
        # Restore the baseline so the session's leftover tracking is unperturbed.
        del mr
        gc.collect()


# --- 3. the two warnings -------------------------------------------------------


@_THREAD_UNSAFE_MARKER
@pytest.mark.agent_authored(model="claude-opus-4.8")
def test_setup_warns_when_resources_left_alive_since_last_init_cuda(request, monkeypatch):
    func = _init_cuda_func(request)
    g = func.__globals__
    # Force the leftover-check precondition: the count at item start exceeds the
    # baseline recorded after the previous init_cuda test, i.e. something in
    # between left a resource alive. Keep a real resource referenced so the
    # "still alive after a collect" count is nonzero.
    base = _live_memory_resource_count()
    leaked = _LocalMR()
    monkeypatch.setitem(g, "_mr_expected_live", base)
    monkeypatch.setitem(g, "_mr_live_at_item_start", base + 1)
    monkeypatch.setitem(g, "_mr_last_init_cuda_nodeid", "tests/prev::test_prev")

    gen = func(_FakeRequest())
    try:
        with pytest.warns(UnreleasedMemoryResourceWarning, match="left alive by"):
            next(gen)  # setup emits the leftover warning
        with pytest.raises(StopIteration):
            next(gen)  # finish teardown
    finally:
        del leaked
        gc.collect()


@_THREAD_UNSAFE_MARKER
@pytest.mark.agent_authored(model="claude-opus-4.8")
def test_teardown_warns_on_baseline_release(request, monkeypatch):
    func = _init_cuda_func(request)
    # Disable the leftover check; this test exercises the teardown-side warning.
    monkeypatch.setitem(func.__globals__, "_mr_expected_live", None)

    # A resource that exists before setup is part of the snapshot; releasing it
    # during the "test" drops the count below the snapshot at teardown.
    baseline_mr = _LocalMR()
    gen = func(_FakeRequest())
    next(gen)  # setup snapshots the count, including baseline_mr
    del baseline_mr  # release the pre-existing resource mid-"test"

    with (
        pytest.warns(BaselineMemoryResourceReleasedWarning, match="existed before it started"),
        pytest.raises(StopIteration),
    ):
        next(gen)  # teardown sees a count below the snapshot
