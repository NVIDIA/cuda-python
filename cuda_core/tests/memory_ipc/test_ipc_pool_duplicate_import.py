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
REIMPORT_ROUNDS = 3

# these tests spawn new processes and files which fails for very many threads;
# child processes and IPC fixtures are shared across workers (#2784); cap at two threads
pytestmark = pytest.mark.parallel_threads_limit(2)


def _enter_import(stack, mr_type, device, alloc_handle):
    """Import the pool for ``alloc_handle`` and register it to close on ``stack``."""
    return stack.enter_context(contextlib.closing(import_pool_resource(mr_type, device, alloc_handle)))


def _run_threads(threads):
    """Start every thread, then join every thread."""
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()


@contextlib.contextmanager
def _child_device():
    """Set the child's current device, and assert the owned-pool count returns to its baseline."""
    device = Device()
    device.set_current()
    baseline = _live_owned_mempool_count()
    yield device, baseline
    assert _live_owned_mempool_count() == baseline


def _import_twice(stack, mr_type, device, alloc_handle):
    # A raw file descriptor carries no UUID, so the registry cannot dedupe.
    first = _enter_import(stack, mr_type, device, int(alloc_handle))
    second = _enter_import(stack, mr_type, device, int(alloc_handle))
    assert first is not second
    assert int(first.handle) == int(second.handle)
    return first, second


def child_close_one_import(mr_type, alloc_handle, descriptor):
    # The stack closes the buffer, then both resources; the first is closed
    # early on purpose, and closing it again is a no-op.
    with _child_device() as (device, baseline), contextlib.ExitStack() as stack:
        stream = device.default_stream
        first, second = _import_twice(stack, mr_type, device, alloc_handle)
        assert _live_owned_mempool_count() == baseline + 1

        first.close()
        assert pool_access_flags(second, device) == driver.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE

        buffer = stack.enter_context(Buffer.from_ipc_descriptor(second, descriptor, stream=stream))
        buffer.fill(7, stream=stream)
        stream.sync()


def child_close_both_imports_with_outstanding_buffer(mr_type, alloc_handle, descriptor):
    # The closes below are the sequence under test; the stack repeats them as
    # no-ops, and releases whatever a failed assertion leaves open.
    with _child_device() as (device, baseline), assert_no_cuda_warning(), contextlib.ExitStack() as stack:
        stream = device.default_stream
        first, second = _import_twice(stack, mr_type, device, alloc_handle)
        assert _live_owned_mempool_count() == baseline + 1
        # The buffer keeps the pool alive after both resources are closed.
        outstanding = stack.enter_context(Buffer.from_ipc_descriptor(first, descriptor, stream=stream))
        first.close()
        assert pool_access_flags(second, device) == driver.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
        buffer = stack.enter_context(Buffer.from_ipc_descriptor(second, descriptor, stream=stream))
        buffer.fill(7, stream=stream)
        stream.sync()
        buffer.close()
        second.close()
        assert _live_owned_mempool_count() == baseline + 1

        outstanding.fill(9, stream=stream)
        stream.sync()
        outstanding.close()
        stream.sync()


def _race_reimport_with_last_release(mr_type, device, alloc_handle, descriptor):
    # The buffer is freed on another thread, which a per-thread default stream
    # would report as a cross-thread free.
    stream = device.create_stream()
    # The stack closes in reverse order: the new buffer and resource, then the
    # stream after it has synchronized the pending frees.
    with contextlib.ExitStack() as stack:
        stack.enter_context(contextlib.closing(stream))
        stack.callback(stream.sync)

        # The buffer holds the last reference to the pool once its resource is
        # closed, so closing the buffer releases the pool.
        with contextlib.closing(import_pool_resource(mr_type, device, int(alloc_handle))) as mr:
            outstanding = stack.enter_context(Buffer.from_ipc_descriptor(mr, descriptor, stream=stream))
        del mr

        barrier = threading.Barrier(2)
        reimported = []
        reimported_buffers = []
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
                reimported.append(mr)
                reimported_buffers.append(Buffer.from_ipc_descriptor(mr, descriptor, stream=stream))
            except Exception as e:
                errors.append(e)
                barrier.abort()

        _run_threads([threading.Thread(target=release), threading.Thread(target=reimport)])
        # Own whatever the thread made before looking at its errors.
        for mr in reimported:
            stack.enter_context(contextlib.closing(mr))
        for buffer in reimported_buffers:
            stack.enter_context(buffer)
        if errors:
            raise errors[0]

        # The release has finished: its cleanup must have left the new import alone.
        (mr,) = reimported
        (buffer,) = reimported_buffers
        assert pool_access_flags(mr, device) == driver.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
        buffer.fill(7, stream=stream)
        stream.sync()


def child_reimport_races_last_release(mr_type, alloc_handle, descriptor):
    with _child_device() as (device, _baseline):
        _set_ipc_pool_release_delay_for_testing(RELEASE_DELAY_US)
        try:
            with assert_no_cuda_warning():
                for _ in range(RACE_ITERATIONS):
                    _race_reimport_with_last_release(mr_type, device, alloc_handle, descriptor)
        finally:
            _set_ipc_pool_release_delay_for_testing(0)


def child_register_reentered_by_gc(mr_type, alloc_handle):
    # The stack closes the imports, then the nested resource.
    with _child_device() as (device, _baseline), contextlib.ExitStack() as stack:
        nested = _enter_import(stack, mr_type, device, int(alloc_handle))
        nested_uuid = uuid.uuid4()
        nested.register(nested_uuid)

        # Registering inserts a weak reference into the registry, and allocating
        # it can start a collection. A callback that registers again then runs on
        # the thread that holds the registry lock.
        def register_during_collection(phase, info):
            if phase == "start":
                nested.register(nested_uuid)

        threshold = gc.get_threshold()
        gc.callbacks.append(register_during_collection)
        try:
            gc.set_threshold(1)
            for _ in range(REENTRY_ITERATIONS):
                # A raw file descriptor carries no UUID, so each import registers a new one.
                mr = _enter_import(stack, mr_type, device, int(alloc_handle))
                assert mr.register(uuid.uuid4()) is mr
        finally:
            gc.set_threshold(*threshold)
            gc.callbacks.remove(register_during_collection)


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

    _run_threads([threading.Thread(target=importer, args=(i,)) for i in range(THREADS)])
    if errors:
        # The imports that succeeded would otherwise stay open.
        for mr in results:
            if mr is not None:
                mr.close()
        raise errors[0]
    return results


def child_concurrent_uuid_imports(mr_type, alloc_handle):
    with _child_device() as (device, baseline):
        assert alloc_handle.uuid is not None
        for _ in range(ITERATIONS):
            with contextlib.ExitStack() as stack:
                results = _import_by_path(stack, "concurrent_uuid", mr_type, device, alloc_handle)
                # Every import of the UUID returns the registered resource, whichever
                # thread registered it, with its device set before it was registered.
                assert all(mr is results[0] for mr in results)
                if mr_type is DeviceMemoryResource:
                    assert results[0].device_id == device.device_id
                assert _live_owned_mempool_count() == baseline + 1
                # Dropping the losing imports must not revoke the winner's access.
                assert (
                    pool_access_flags(results[0], device) == driver.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
                )
            del results
            assert _live_owned_mempool_count() == baseline


def _import_by_path(stack, import_path, mr_type, device, alloc_handle):
    """Import the pool as ``import_path`` names; ``stack`` closes every resource the import made."""
    if import_path == "single":
        return [_enter_import(stack, mr_type, device, alloc_handle)]
    if import_path == "raw_handle":
        # A raw file descriptor carries no UUID, so each import is a real import.
        return [_enter_import(stack, mr_type, device, int(alloc_handle)) for _ in range(THREADS)]
    assert import_path == "concurrent_uuid", import_path
    return [stack.enter_context(contextlib.closing(mr)) for mr in _import_concurrently(mr_type, device, alloc_handle)]


def _import_fill_and_release_unsynchronized(import_path, mr_type, device, alloc_handle, descriptor, stream):
    # The stack closes in reverse order: the buffers, then the resources.
    # Closing a buffer enqueues its cuMemFreeAsync without waiting for it, so
    # the frees can still be pending when the last resource releases the pool.
    with contextlib.ExitStack() as stack:
        resources = _import_by_path(stack, import_path, mr_type, device, alloc_handle)
        buffers = [stack.enter_context(Buffer.from_ipc_descriptor(mr, descriptor, stream=stream)) for mr in resources]
        for buffer in buffers:
            buffer.fill(7, stream=stream)
        stream.sync()


def _reimport_and_use(import_path, mr_type, device, alloc_handle, descriptor, stream):
    with contextlib.ExitStack() as stack:
        resources = _import_by_path(stack, import_path, mr_type, device, alloc_handle)
        for mr in resources:
            assert pool_access_flags(mr, device) == driver.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
        buffer = stack.enter_context(Buffer.from_ipc_descriptor(resources[0], descriptor, stream=stream))
        buffer.fill(9, stream=stream)
        stream.sync()


def child_reimport_after_release_with_pending_free(mr_type, alloc_handle, descriptor, import_path):
    # A failed grant is also reported as a warning by the cleanup that follows it.
    with _child_device() as (device, _baseline), assert_no_cuda_warning():
        stream = device.default_stream
        for _ in range(REIMPORT_ROUNDS):
            # The helpers keep no reference to any resource once they return, so
            # the registry entry is gone and the next import is a real one.
            _import_fill_and_release_unsynchronized(import_path, mr_type, device, alloc_handle, descriptor, stream)
            _reimport_and_use(import_path, mr_type, device, alloc_handle, descriptor, stream)
            stream.sync()


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

    # Releasing the last import of a pool with unsynchronized frees pending
    # leaves the driver holding the pool (#3514); the next import must still
    # grant device access.
    @pytest.mark.parametrize("import_path", ["single", "raw_handle", "concurrent_uuid"])
    @pytest.mark.agent_authored(model="claude-sonnet-5-5")
    def test_reimport_after_release_with_pending_free(self, pool_resource, exported_buffer, import_path):
        run_child(
            child_reimport_after_release_with_pending_free,
            type(pool_resource),
            pool_resource.allocation_handle,
            exported_buffer.ipc_descriptor,
            import_path,
        )
