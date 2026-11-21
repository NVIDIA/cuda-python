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
import gc
import multiprocessing as mp
import threading
import time
import uuid

import pytest
from helpers.child_processes import run_child
from helpers.contexts import assert_no_cuda_warning
from helpers.memory import import_pool_resource, is_pool_resource, pool_access_flags

from cuda.bindings import driver
from cuda.core import Buffer, Device, DeviceMemoryResource
from cuda.core._memory._buffer import _live_owned_mempool_count
from cuda.core._rt import _set_ipc_pool_release_delay_for_testing

NBYTES = 64
THREADS = 4
ITERATIONS = 20
# The owner's release sleeps this long in the race test, so the re-import overlaps it.
RELEASE_DELAY_US = 50_000
RACE_ITERATIONS = 3
REENTRY_ITERATIONS = 20

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
        assert _live_owned_mempool_count() == baseline + 1
        # The buffer keeps the pool alive after both resources are closed.
        outstanding = Buffer.from_ipc_descriptor(first, descriptor, stream=stream)
        first.close()
        assert pool_access_flags(second, device) == driver.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
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


def _race_reimport_with_last_release(mr_type, device, alloc_handle, descriptor):
    # The buffer is freed on another thread, which a per-thread default stream
    # would report as a cross-thread free.
    stream = device.create_stream()
    # The buffer holds the last reference to the pool once its resource is
    # closed, so closing the buffer releases the pool.
    mr = import_pool_resource(mr_type, device, int(alloc_handle))
    outstanding = Buffer.from_ipc_descriptor(mr, descriptor, stream=stream)
    mr.close()
    del mr

    barrier = threading.Barrier(2)
    reimported = []
    errors = []

    def release():
        try:
            barrier.wait()
            outstanding.close()
        except Exception as e:
            errors.append(e)
            barrier.abort()

    def reimport():
        try:
            Device().set_current()
            barrier.wait()
            # Start once the release is under way, inside its delay.
            time.sleep(RELEASE_DELAY_US / 4 / 1e6)
            # A raw file descriptor carries no UUID, so this is a real import,
            # through an object the releasing thread never sees.
            mr = import_pool_resource(mr_type, device, int(alloc_handle))
            reimported.append((mr, Buffer.from_ipc_descriptor(mr, descriptor, stream=stream)))
        except Exception as e:
            errors.append(e)
            barrier.abort()

    threads = [threading.Thread(target=release), threading.Thread(target=reimport)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if errors:
        raise errors[0]

    # The release has finished: its cleanup must have left the new import alone.
    ((mr, buffer),) = reimported
    assert pool_access_flags(mr, device) == driver.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
    buffer.fill(7, stream=stream)
    stream.sync()
    buffer.close()
    mr.close()
    stream.sync()
    stream.close()


def child_reimport_races_last_release(mr_type, alloc_handle, descriptor):
    device = Device()
    device.set_current()
    baseline = _live_owned_mempool_count()

    _set_ipc_pool_release_delay_for_testing(RELEASE_DELAY_US)
    try:
        with assert_no_cuda_warning():
            for _ in range(RACE_ITERATIONS):
                _race_reimport_with_last_release(mr_type, device, alloc_handle, descriptor)
    finally:
        _set_ipc_pool_release_delay_for_testing(0)
    assert _live_owned_mempool_count() == baseline


def child_register_reentered_by_gc(mr_type, alloc_handle):
    device = Device()
    device.set_current()
    baseline = _live_owned_mempool_count()

    nested = import_pool_resource(mr_type, device, int(alloc_handle))
    nested_uuid = uuid.uuid4()
    nested.register(nested_uuid)

    # Registering inserts a weak reference into the registry, and allocating
    # it can start a collection. A callback that registers again then runs on
    # the thread that holds the registry lock.
    def register_during_collection(phase, info):
        if phase == "start":
            nested.register(nested_uuid)

    imports = []
    threshold = gc.get_threshold()
    gc.callbacks.append(register_during_collection)
    try:
        gc.set_threshold(1)
        for _ in range(REENTRY_ITERATIONS):
            # A raw file descriptor carries no UUID, so each import registers a new one.
            mr = import_pool_resource(mr_type, device, int(alloc_handle))
            imports.append(mr)
            assert mr.register(uuid.uuid4()) is mr
    finally:
        gc.set_threshold(*threshold)
        gc.callbacks.remove(register_during_collection)

    for mr in imports:
        mr.close()
    nested.close()
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
            barrier.abort()

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
        # thread registered it, with its device set before it was registered.
        assert all(mr is results[0] for mr in results)
        if mr_type is DeviceMemoryResource:
            assert results[0].device_id == device.device_id
        assert _live_owned_mempool_count() == baseline + 1
        # Dropping the losing imports must not revoke the winner's access.
        assert pool_access_flags(results[0], device) == driver.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
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
    def test_reimport_racing_the_last_release_keeps_access(self, pool_resource, exported_buffer):
        run_child(
            child_reimport_races_last_release,
            type(pool_resource),
            pool_resource.allocation_handle,
            exported_buffer.ipc_descriptor,
        )

    @pytest.mark.agent_authored(model="claude-opus-5-5")
    def test_registration_reentered_from_garbage_collection(self, ipc_device, pool_resource):
        ipc_device.set_current()
        run_child(child_register_reentered_by_gc, type(pool_resource), pool_resource.allocation_handle)

    @pytest.mark.agent_authored(model="claude-opus-5-5")
    def test_concurrent_imports_of_one_uuid_share_a_resource(self, ipc_device, pool_resource):
        ipc_device.set_current()
        run_child(child_concurrent_uuid_imports, type(pool_resource), pool_resource.allocation_handle)
