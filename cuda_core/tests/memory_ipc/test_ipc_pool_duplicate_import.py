# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Imports of one memory pool in one process share ownership (#3510).

The driver returns the same ``CUmemoryPool`` for every import of a pool in a
process, and the pool's access settings are shared by all of them. Releasing
one import used to revoke device access for the pool (the nvbug 5698116
workaround), which broke the imports that were still open; for a pinned pool
the next device access failed with ``CUDA_ERROR_INVALID_VALUE``. All imports
now share one owner, which revokes access and destroys the pool only when the
last of them is released.
"""

import contextlib
import multiprocessing as mp
import threading

import pytest
from helpers.child_processes import run_child
from helpers.contexts import assert_no_cuda_warning
from helpers.memory import import_pool_resource, is_pool_resource, pool_access_flags

from cuda.bindings import driver
from cuda.core import Buffer, Device
from cuda.core._memory._buffer import _live_owned_mempool_count

NBYTES = 64
THREADS = 4
ITERATIONS = 20

# these tests spawn new processes and files which fails for very many threads;
# child processes and IPC fixtures are shared across workers (#2784); cap at two threads
pytestmark = pytest.mark.parallel_threads_limit(2)


def _import_twice(mr_type, device, alloc_handle):
    # A raw file descriptor carries no UUID, so the registry cannot dedupe.
    first = import_pool_resource(mr_type, device, int(alloc_handle))
    second = import_pool_resource(mr_type, device, int(alloc_handle))
    assert first is not second
    assert int(first.handle) == int(second.handle)
    return first, second


def child_close_one_import(mr_type, alloc_handle, descriptor):
    device = Device()
    device.set_current()
    stream = device.default_stream
    baseline = _live_owned_mempool_count()

    first, second = _import_twice(mr_type, device, alloc_handle)
    assert _live_owned_mempool_count() == baseline + 1

    first.close()
    assert pool_access_flags(second, device) == driver.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE

    buffer = Buffer.from_ipc_descriptor(second, descriptor, stream=stream)
    buffer.fill(7, stream=stream)
    stream.sync()
    buffer.close()
    second.close()
    assert _live_owned_mempool_count() == baseline


def child_close_both_imports_with_outstanding_buffer(mr_type, alloc_handle, descriptor):
    device = Device()
    device.set_current()
    stream = device.default_stream
    baseline = _live_owned_mempool_count()

    with assert_no_cuda_warning():
        first, second = _import_twice(mr_type, device, alloc_handle)
        # The buffer keeps the pool alive after both resources are closed.
        outstanding = Buffer.from_ipc_descriptor(first, descriptor, stream=stream)
        first.close()
        buffer = Buffer.from_ipc_descriptor(second, descriptor, stream=stream)
        buffer.fill(7, stream=stream)
        stream.sync()
        buffer.close()
        second.close()
        assert _live_owned_mempool_count() == baseline + 1

        outstanding.fill(9, stream=stream)
        stream.sync()
        outstanding.close()
        stream.sync()
    assert _live_owned_mempool_count() == baseline


def _import_concurrently(mr_type, device, alloc_handle):
    barrier = threading.Barrier(THREADS)
    results = [None] * THREADS
    errors = []

    def importer(index):
        try:
            Device().set_current()
            barrier.wait()
            results[index] = import_pool_resource(mr_type, device, alloc_handle)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=importer, args=(i,)) for i in range(THREADS)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if errors:
        raise errors[0]
    return results


def child_concurrent_uuid_imports(mr_type, alloc_handle):
    device = Device()
    device.set_current()
    assert alloc_handle.uuid is not None
    baseline = _live_owned_mempool_count()

    for _ in range(ITERATIONS):
        results = _import_concurrently(mr_type, device, alloc_handle)
        # Every import of the UUID returns the registered resource, whichever
        # thread registered it.
        assert all(mr is results[0] for mr in results)
        assert _live_owned_mempool_count() == baseline + 1
        results[0].close()
        del results
        assert _live_owned_mempool_count() == baseline


class TestIpcPoolDuplicateImport:
    """Repeat imports of one pool in a process share the pool and its access settings."""

    @pytest.fixture(autouse=True)
    def _set_start_method(self):
        # Ensure spawn is used for multiprocessing
        with contextlib.suppress(RuntimeError):
            mp.set_start_method("spawn", force=True)

    @pytest.fixture
    def pool_resource(self, ipc_memory_resource):
        if not is_pool_resource(ipc_memory_resource):
            pytest.skip("a VirtualMemoryResource has no memory pool to import")
        return ipc_memory_resource

    @pytest.fixture
    def exported_buffer(self, ipc_device, pool_resource):
        ipc_device.set_current()
        stream = ipc_device.default_stream
        buffer = pool_resource.allocate(NBYTES, stream=stream)
        stream.sync()
        yield buffer
        buffer.close(stream)

    @pytest.mark.agent_authored(model="claude-opus-5-5")
    def test_closing_one_import_keeps_the_other_usable(self, pool_resource, exported_buffer):
        run_child(
            child_close_one_import, type(pool_resource), pool_resource.allocation_handle, exported_buffer.ipc_descriptor
        )

    @pytest.mark.agent_authored(model="claude-opus-5-5")
    def test_closing_both_imports_with_an_outstanding_buffer(self, pool_resource, exported_buffer):
        run_child(
            child_close_both_imports_with_outstanding_buffer,
            type(pool_resource),
            pool_resource.allocation_handle,
            exported_buffer.ipc_descriptor,
        )

    @pytest.mark.agent_authored(model="claude-opus-5-5")
    def test_concurrent_imports_of_one_uuid_share_a_resource(self, ipc_device, pool_resource):
        ipc_device.set_current()
        run_child(child_concurrent_uuid_imports, type(pool_resource), pool_resource.allocation_handle)
