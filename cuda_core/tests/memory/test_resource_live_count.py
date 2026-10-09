# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the process-wide live owned-resource counters.

The counting lives in the C++ RAII layer: ``_live_owned_mempool_count`` tracks
memory pools this layer owns (created with ``cuMemPoolCreate`` or imported), and
``_live_va_reservation_count`` tracks VA reservations (``cuMemAddressReserve``).
``conftest.py`` reads them to decide whether the ``init_cuda`` teardown needs a
``gc.collect()``. Borrowed pool references (the device default pool) and
``MemoryResource`` subclasses that hold no pool or reservation move neither
counter, which these tests pin down.

Tests compare against a count read at their start, because resources retained
from earlier tests make it nonzero. Each requests ``init_cuda`` and frees what
it constructs. They are ``thread_unsafe`` because the counters are process-global.
"""

import gc
import platform
import threading

import pytest
from helpers.constants import POOL_SIZE

from cuda.core import (
    DeviceMemoryResource,
    DeviceMemoryResourceOptions,
    VirtualMemoryResource,
    VirtualMemoryResourceOptions,
)
from cuda.core._memory._buffer import _live_owned_mempool_count, _live_va_reservation_count

# Matches the VMM handle type helper in test_memory.py.
VMM_HANDLE_TYPE = "win32_kmt" if platform.system() == "Windows" else "posix_fd"

_THREAD_UNSAFE_MARKER = pytest.mark.thread_unsafe(reason="asserts on the process-global owned-resource counts")


def _skip_without_mempools(device):
    if not device.properties.memory_pools_supported:
        pytest.skip("Device does not support memory pools")


def _owned_pool(device):
    """An owning DeviceMemoryResource: passing options creates its own pool."""
    return DeviceMemoryResource(device, DeviceMemoryResourceOptions(max_size=POOL_SIZE))


@_THREAD_UNSAFE_MARKER
@pytest.mark.agent_authored(model="claude-opus-4.8")
def test_owned_pool_is_counted_until_freed(init_cuda):
    _skip_without_mempools(init_cuda)
    before = _live_owned_mempool_count()
    mr = _owned_pool(init_cuda)
    assert _live_owned_mempool_count() == before + 1
    del mr
    assert _live_owned_mempool_count() == before


@_THREAD_UNSAFE_MARKER
@pytest.mark.agent_authored(model="claude-opus-4.8")
def test_borrowed_default_pool_is_not_counted(init_cuda):
    # A no-options DeviceMemoryResource borrows the device default pool
    # (cuDeviceGetMemPool), which this layer does not own or destroy, so it must
    # not move the owned-pool counter.
    _skip_without_mempools(init_cuda)
    before = _live_owned_mempool_count()
    mr = DeviceMemoryResource(init_cuda)
    assert _live_owned_mempool_count() == before
    del mr
    assert _live_owned_mempool_count() == before


@_THREAD_UNSAFE_MARKER
@pytest.mark.agent_authored(model="claude-opus-4.8")
def test_owned_pool_in_cycle_stays_counted_until_collect(init_cuda):
    _skip_without_mempools(init_cuda)
    before = _live_owned_mempool_count()
    mr = _owned_pool(init_cuda)
    # A dict cycle holds the only reference to mr (the Cython resource cannot
    # hold a self-attribute), so refcounting alone will not free it.
    holder = {"mr": mr}
    holder["self"] = holder
    del mr, holder
    assert _live_owned_mempool_count() == before + 1
    # Disable automatic collection so a background gc cannot free the cycle
    # before the assertion is checked.
    enabled = gc.isenabled()
    gc.disable()
    try:
        gc.collect()
        assert _live_owned_mempool_count() == before
    finally:
        if enabled:
            gc.enable()


@_THREAD_UNSAFE_MARKER
@pytest.mark.agent_authored(model="claude-opus-4.8")
def test_va_reservation_is_counted_until_freed(init_cuda):
    if not init_cuda.properties.virtual_memory_management_supported:
        pytest.skip("Device does not support virtual memory management")
    vmm = VirtualMemoryResource(init_cuda, config=VirtualMemoryResourceOptions(handle_type=VMM_HANDLE_TYPE))
    before = _live_va_reservation_count()
    try:
        buf = vmm.allocate(4096)
    except NotImplementedError:
        pytest.skip("VMM handle type not implemented on this platform")
    # An allocation reserves at least one address range.
    assert _live_va_reservation_count() >= before + 1
    buf.close()
    assert _live_va_reservation_count() == before


@_THREAD_UNSAFE_MARKER
@pytest.mark.agent_authored(model="claude-opus-4.8")
def test_concurrent_owned_pool_construct_and_drop_returns_to_start(init_cuda):
    # On a free-threaded build this guards against a non-atomic counter; on a
    # GIL build it passes trivially. Every thread sets the device current, then
    # creates and drops owning pools; after join the count is back at the start.
    _skip_without_mempools(init_cuda)
    before = _live_owned_mempool_count()
    n_threads = 4
    n_per_thread = 10

    def worker():
        init_cuda.set_current()
        for _ in range(n_per_thread):
            mr = _owned_pool(init_cuda)
            del mr

    threads = [threading.Thread(target=worker) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    # Let any deferred deallocation settle.
    gc.collect()
    assert _live_owned_mempool_count() == before
