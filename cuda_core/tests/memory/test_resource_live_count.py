# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the process-wide live ``MemoryResource`` counter.

``conftest.py`` uses the counter to decide whether the ``init_cuda`` teardown
needs a ``gc.collect()`` and to detect resources left alive by other tests.
Tests compare against a count read at their start, because resources retained
from earlier tests make it nonzero. Each test requests ``init_cuda`` and frees
what it constructs. They are ``thread_unsafe`` because the count is
process-global.
"""

import gc
import threading

import pytest

from cuda.core import DeviceMemoryResource, MemoryResource
from cuda.core._memory._buffer import _live_memory_resource_count


class _CountedMR(MemoryResource):
    pass


class _CountedMRWithArgs(MemoryResource):
    def __init__(self, a, b):
        self.a = a
        self.b = b


_THREADSAFE_MARKER = pytest.mark.thread_unsafe(reason="asserts on the process-global resource count")


@_THREADSAFE_MARKER
@pytest.mark.agent_authored(model="glm-5.2")
def test_new_resource_is_counted_until_freed(init_cuda):
    before = _live_memory_resource_count()
    mr = _CountedMR()
    assert _live_memory_resource_count() == before + 1
    del mr
    assert _live_memory_resource_count() == before


@_THREADSAFE_MARKER
@pytest.mark.agent_authored(model="glm-5.2")
def test_resource_in_cycle_stays_counted_until_collect(init_cuda):
    before = _live_memory_resource_count()
    mr = _CountedMR()
    mr.self_ref = mr  # create a reference cycle
    del mr
    # The cycle keeps the resource alive, so it is still counted.
    assert _live_memory_resource_count() == before + 1
    # Disable automatic collection so a background gc cannot free the cycle
    # before the assertion is checked.
    enabled = gc.isenabled()
    gc.disable()
    try:
        gc.collect()
        assert _live_memory_resource_count() == before
    finally:
        if enabled:
            gc.enable()


@_THREADSAFE_MARKER
@pytest.mark.agent_authored(model="glm-5.2")
def test_subclass_with_init_args_is_counted(init_cuda):
    # Guards the no-argument base __cinit__: a Python subclass whose __init__
    # takes extra arguments still constructs (and is counted) via the base.
    before = _live_memory_resource_count()
    mr = _CountedMRWithArgs(1, 2)
    assert _live_memory_resource_count() == before + 1
    assert mr.a == 1 and mr.b == 2
    del mr
    assert _live_memory_resource_count() == before


@_THREADSAFE_MARKER
@pytest.mark.agent_authored(model="glm-5.2")
def test_cython_subclass_wrapper_is_counted(init_cuda):
    # Covers a Cython subclass with its own __cinit__: a non-owning
    # DeviceMemoryResource(device) wrapper (no options) wraps the default pool
    # and is counted and uncounted like any resource.
    before = _live_memory_resource_count()
    mr = DeviceMemoryResource(init_cuda)
    assert _live_memory_resource_count() == before + 1
    del mr
    assert _live_memory_resource_count() == before


@_THREADSAFE_MARKER
@pytest.mark.agent_authored(model="glm-5.2")
def test_concurrent_construct_and_drop_returns_to_start(init_cuda):
    # On a free-threaded build this guards against a non-atomic counter; on a
    # GIL build it passes trivially. Every thread constructs and drops its own
    # resources; after join the count is back at the starting value.
    before = _live_memory_resource_count()
    n_threads = 8
    n_per_thread = 200

    def worker():
        for _ in range(n_per_thread):
            mr = _CountedMR()
            del mr

    threads = [threading.Thread(target=worker) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    # Let any deferred deallocation settle.
    gc.collect()
    assert _live_memory_resource_count() == before
