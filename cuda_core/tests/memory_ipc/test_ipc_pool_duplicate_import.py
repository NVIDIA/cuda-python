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
from helpers.child_processes import child_timeout_sec, kill_subprocesses

from cuda.bindings import driver
from cuda.core import Buffer, Device, DeviceMemoryResource, PinnedMemoryResource, VirtualMemoryResource
from cuda.core._memory._buffer import _live_owned_mempool_count
from cuda.core._utils.cuda_utils import handle_return

CHILD_TIMEOUT_SEC = child_timeout_sec()
NBYTES = 64
THREADS = 4
ITERATIONS = 20

# these tests spawn new processes and files which fails for very many threads;
# child processes and IPC fixtures are shared across workers (#2784); cap at two threads
pytestmark = pytest.mark.parallel_threads_limit(2)


def _import(mr_type, device, alloc_handle):
    if mr_type is DeviceMemoryResource:
        return DeviceMemoryResource.from_allocation_handle(device, alloc_handle)
    return PinnedMemoryResource.from_allocation_handle(alloc_handle)


def _location(device):
    location = driver.CUmemLocation()
    location.type = driver.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
    location.id = device.device_id
    return location


def _grant_device_access(mr, device):
    # An imported pinned pool is not device accessible until access is granted;
    # an imported device pool is always accessible from its own device, which
    # the driver reports as CUDA_ERROR_INVALID_DEVICE.
    desc = driver.CUmemAccessDesc()
    desc.location = _location(device)
    desc.flags = driver.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
    (err,) = driver.cuMemPoolSetAccess(driver.CUmemoryPool(int(mr.handle)), [desc], 1)
    assert err in (driver.CUresult.CUDA_SUCCESS, driver.CUresult.CUDA_ERROR_INVALID_DEVICE), err


def _device_access(mr, device):
    return handle_return(driver.cuMemPoolGetAccess(driver.CUmemoryPool(int(mr.handle)), _location(device)))


def child_close_one_import(queue):
    device = Device()
    device.set_current()
    stream = device.default_stream
    mr_type, alloc_handle, descriptor = queue.get()
    baseline = _live_owned_mempool_count()

    # A raw file descriptor carries no UUID, so the registry cannot dedupe.
    first = _import(mr_type, device, int(alloc_handle))
    second = _import(mr_type, device, int(alloc_handle))
    assert first is not second
    assert int(first.handle) == int(second.handle)
    assert _live_owned_mempool_count() == baseline + 1

    _grant_device_access(second, device)
    first.close()
    assert _device_access(second, device) == driver.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE

    buffer = Buffer.from_ipc_descriptor(second, descriptor, stream=stream)
    buffer.fill(7, stream=stream)
    stream.sync()
    buffer.close()
    second.close()
    assert _live_owned_mempool_count() == baseline


def _import_concurrently(mr_type, device, alloc_handle):
    barrier = threading.Barrier(THREADS)
    results = [None] * THREADS
    errors = []

    def importer(index):
        try:
            Device().set_current()
            barrier.wait()
            results[index] = _import(mr_type, device, alloc_handle)
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


def child_concurrent_uuid_imports(queue):
    device = Device()
    device.set_current()
    mr_type, alloc_handle = queue.get()
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


def _run_child(target, *items):
    queue = mp.Queue()
    process = mp.Process(target=target, args=(queue,))
    process.start()
    queue.put(items)
    process.join(timeout=CHILD_TIMEOUT_SEC)
    survivors = kill_subprocesses(process)
    assert not survivors, "child did not exit within timeout"
    assert process.exitcode == 0, f"child process failed with exit code {process.exitcode}"


class TestIpcPoolDuplicateImport:
    """Repeat imports of one pool in a process share the pool and its access settings."""

    @pytest.fixture(autouse=True)
    def _set_start_method(self):
        # Ensure spawn is used for multiprocessing
        with contextlib.suppress(RuntimeError):
            mp.set_start_method("spawn", force=True)

    @pytest.fixture
    def pool_resource(self, ipc_memory_resource):
        if isinstance(ipc_memory_resource, VirtualMemoryResource):
            pytest.skip("a VirtualMemoryResource has no memory pool to import")
        return ipc_memory_resource

    @pytest.mark.agent_authored(model="claude-opus-5-5")
    def test_closing_one_import_keeps_the_other_usable(self, ipc_device, pool_resource):
        ipc_device.set_current()
        stream = ipc_device.default_stream
        buffer = pool_resource.allocate(NBYTES, stream=stream)
        stream.sync()
        try:
            _run_child(
                child_close_one_import, type(pool_resource), pool_resource.allocation_handle, buffer.ipc_descriptor
            )
        finally:
            buffer.close(stream)

    @pytest.mark.agent_authored(model="claude-opus-5-5")
    def test_concurrent_imports_of_one_uuid_share_a_resource(self, ipc_device, pool_resource):
        ipc_device.set_current()
        _run_child(child_concurrent_uuid_imports, type(pool_resource), pool_resource.allocation_handle)
