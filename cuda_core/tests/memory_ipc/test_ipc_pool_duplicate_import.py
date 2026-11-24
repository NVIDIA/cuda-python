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
import functools
import gc
import multiprocessing as mp
import threading
import uuid

import helpers
import pytest
from helpers.child_processes import child_timeout_sec, run_child
from helpers.contexts import assert_no_cuda_warning
from helpers.latch import LatchKernel
from helpers.memory import (
    drain_on_exit,
    enter_imported_pool,
    import_pool_resource,
    is_pool_resource,
    pool_access_flags,
)
from helpers.misc import run_in_threads

from cuda.bindings import driver
from cuda.core import Buffer, Device, DeviceMemoryResource
from cuda.core._memory._buffer import _live_owned_mempool_count
from cuda.core._utils._testing import (
    arm_pool_release_gate,
    disarm_pool_release_gate,
    install_driver_fn_wrappers,
    open_pool_release_gate,
    wait_pool_release_gate_entered,
)

NBYTES = 64
THREADS = 4
ITERATIONS = 20
# How long the race test holds the release after the re-import's driver call.
# A correct registry makes the re-import wait for the release, so the hold
# always runs out; it only bounds how fast a broken registry's re-import must
# finish to be caught.
RELEASE_GRACE_SEC = 0.25
REENTRY_ITERATIONS = 20
# How long the race test holds the release after the re-import's driver call.
# A correct registry makes the re-import wait for the release, so the hold
# always runs out; it only bounds how fast a broken registry's re-import must
# finish to be caught.
RELEASE_GRACE_SEC = 0.25

# these tests spawn new processes and files which fails for very many threads;
# child processes and IPC fixtures are shared across workers (#2784); cap at two threads
pytestmark = pytest.mark.parallel_threads_limit(2)


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
    first = enter_imported_pool(stack, mr_type, device, int(alloc_handle))
    second = enter_imported_pool(stack, mr_type, device, int(alloc_handle))
    assert first is not second
    assert int(first.handle) == int(second.handle)
    return first, second


def child_close_one_import(mr_type, alloc_handle, descriptor):
    # The stack closes the buffer, then both resources; the first is closed
    # early on purpose, and closing it again is a no-op.
    with _child_device() as (device, baseline), contextlib.ExitStack() as stack:
        stream = device.default_stream
        drain_on_exit(stack, stream)
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
        drain_on_exit(stack, stream)
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
    # Only a hang reaches a gate timeout; it ends before run_child's timeout, so
    # that a hang fails here with a message.
    gate_timeout = child_timeout_sec() / 4
    # The stack closes in reverse order: the new buffer and resource, then the
    # stream after it has synchronized the pending frees.
    with contextlib.ExitStack() as stack:
        stack.enter_context(contextlib.closing(stream))
        drain_on_exit(stack, stream)

        # The buffer holds the last reference to the pool once its resource is
        # closed, so closing the buffer releases the pool.
        with contextlib.closing(import_pool_resource(mr_type, device, int(alloc_handle))) as mr:
            pool = int(mr.handle)
            outstanding = stack.enter_context(Buffer.from_ipc_descriptor(mr, descriptor, stream=stream))
        del mr

        barrier = threading.Barrier(2)
        reimported = []
        reimported_buffers = []

        def release():
            barrier.wait()
            # The gate holds this release after its last reference is gone and
            # before it revokes access, while the re-import below runs.
            outstanding.close()

        def reimport():
            Device().set_current()
            barrier.wait()
            # Start once the release is held, past the point where it could be joined.
            assert wait_pool_release_gate_entered(gate_timeout), "the release never reached its access revocation"
            # A raw file descriptor carries no UUID, so this is a real import,
            # through an object the releasing thread never sees.
            try:
                mr = import_pool_resource(mr_type, device, int(alloc_handle))
            finally:
                # Only an import that did not wait for the release gets here first.
                open_pool_release_gate()
            reimported.append(mr)
            reimported_buffers.append(Buffer.from_ipc_descriptor(mr, descriptor, stream=stream))

        arm_pool_release_gate(pool, gate_timeout, RELEASE_GRACE_SEC)
        try:
            run_in_threads([release, reimport], barrier=barrier)
        finally:
            overlapped = disarm_pool_release_gate()
            # Own whatever the thread made, also when a thread failed.
            for mr in reimported:
                stack.enter_context(contextlib.closing(mr))
            for buffer in reimported_buffers:
                stack.enter_context(buffer)

        # Without the overlap the race was not exercised, and a pass would mean nothing.
        assert overlapped, "the re-import did not run while the release was held; the gate timed out"

        # The release has finished: its cleanup must have left the new import alone.
        (mr,) = reimported
        (buffer,) = reimported_buffers
        assert pool_access_flags(mr, device) == driver.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
        buffer.fill(7, stream=stream)
        stream.sync()


def child_reimport_races_last_release(mr_type, alloc_handle, descriptor):
    # A spawned child does not run conftest.py, which installs the wrappers.
    install_driver_fn_wrappers()
    with _child_device() as (device, _baseline), assert_no_cuda_warning():
        _race_reimport_with_last_release(mr_type, device, alloc_handle, descriptor)


def child_register_reentered_by_gc(mr_type, alloc_handle):
    # The stack closes the imports, then the nested resource.
    with _child_device() as (device, _baseline), contextlib.ExitStack() as stack:
        nested = enter_imported_pool(stack, mr_type, device, int(alloc_handle))
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
                mr = enter_imported_pool(stack, mr_type, device, int(alloc_handle))
                assert mr.register(uuid.uuid4()) is mr
        finally:
            gc.set_threshold(*threshold)
            gc.callbacks.remove(register_during_collection)


def _import_concurrently(mr_type, device, alloc_handle):
    barrier = threading.Barrier(THREADS)
    results = [None] * THREADS

    def importer(index):
        Device().set_current()
        barrier.wait()
        results[index] = import_pool_resource(mr_type, device, alloc_handle)

    try:
        run_in_threads([functools.partial(importer, i) for i in range(THREADS)], barrier=barrier)
    except BaseException:
        # The imports that succeeded would otherwise stay open.
        for mr in results:
            if mr is not None:
                mr.close()
        raise
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
        return [enter_imported_pool(stack, mr_type, device, alloc_handle)]
    if import_path == "raw_handle":
        # A raw file descriptor carries no UUID, so each import is a real import.
        return [enter_imported_pool(stack, mr_type, device, int(alloc_handle)) for _ in range(THREADS)]
    assert import_path == "concurrent_uuid", import_path
    return [stack.enter_context(contextlib.closing(mr)) for mr in _import_concurrently(mr_type, device, alloc_handle)]


def _release_import_holding_free(import_path, mr_type, device, alloc_handle, descriptor, stream, latch):
    """Import, fill, and release the pool with the import's free held pending behind ``latch``.

    The frees that closing the buffers enqueues sit behind ``latch`` on
    ``stream`` until it is released, so the imported pointer stays mapped and
    the pool stays alive when the next import grants device access. That is the
    state in which the grant fails with ``CUDA_ERROR_ALREADY_MAPPED`` and the
    retry under test recovers; a free that retired first would clear it and
    hide a regression.
    """
    # The stack closes the buffers, then the resources. Returning drops the
    # ``resources`` and ``buffers`` references, so the registry entry clears and
    # the next import is a real one, while the frees stay pending behind latch.
    with contextlib.ExitStack() as stack:
        resources = _import_by_path(stack, import_path, mr_type, device, alloc_handle)
        buffers = [stack.enter_context(Buffer.from_ipc_descriptor(mr, descriptor, stream=stream)) for mr in resources]
        for buffer in buffers:
            buffer.fill(7, stream=stream)
        stream.sync()
        latch.launch(stream)


def _reimport_and_check_grant(import_path, mr_type, device, alloc_handle, descriptor, stream, latch):
    """Re-import the pool while a free is pending and assert the grant that #3514 fails on."""
    with contextlib.ExitStack() as stack:
        resources = _import_by_path(stack, import_path, mr_type, device, alloc_handle)
        for mr in resources:
            assert pool_access_flags(mr, device) == driver.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
        # The grant was checked with the free still pending. Release the stream
        # before importing a pointer, which waits for a pending free of it.
        latch.release()
        buffer = stack.enter_context(Buffer.from_ipc_descriptor(resources[0], descriptor, stream=stream))
        buffer.fill(9, stream=stream)
        stream.sync()


def child_reimport_after_release_with_pending_free(mr_type, alloc_handle, descriptor, import_path):
    # A failed grant is also reported as a warning by the cleanup that follows it.
    with _child_device() as (device, _baseline), assert_no_cuda_warning(), contextlib.ExitStack() as stack:
        stream = stack.enter_context(contextlib.closing(device.create_stream()))
        # The timeout outlasts run_child's, so a stuck child is ended before the
        # latch can release itself and let the free retire unobserved.
        latch = LatchKernel(device, timeout_sec=4 * child_timeout_sec())
        stack.callback(latch.close)
        drain_on_exit(stack, stream)
        # Registered after the drain so that it runs first: a held latch would stall the drain.
        stack.callback(latch.release)
        _release_import_holding_free(import_path, mr_type, device, alloc_handle, descriptor, stream, latch)
        _reimport_and_check_grant(import_path, mr_type, device, alloc_handle, descriptor, stream, latch)
        assert not latch.timed_out, "the latch timed out, so the free may have retired before the grant was checked"


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
    def test_reimport_after_release_with_pending_free(self, ipc_device, pool_resource, exported_buffer, import_path):
        # The child holds the free with a LatchKernel. Check what it needs here:
        # a skip raised in the child would surface as a child failure.
        if helpers.CUDA_INCLUDE_PATH is None:
            pytest.skip("LatchKernel needs the CUDA headers")
        if ipc_device.compute_capability.major < 7:
            pytest.skip("__nanosleep is only available starting Volta (sm70)")
        run_child(
            child_reimport_after_release_with_pending_free,
            type(pool_resource),
            pool_resource.allocation_handle,
            exported_buffer.ipc_descriptor,
            import_path,
        )
