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

import helpers
import pytest
from helpers.child_processes import child_timeout_sec, run_child
from helpers.contexts import assert_no_cuda_warning
from helpers.latch import LatchKernel
from helpers.memory import import_pool_resource, is_pool_resource, pool_access_flags, pool_supported_devices

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
REENTRY_ITERATIONS = 100

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
    # early on purpose, and closing it again is a no-op. It drains the stream
    # last, so the imported free completes before the exporter frees its buffer.
    with _child_device() as (device, baseline), contextlib.ExitStack() as stack:
        stream = device.default_stream
        stack.callback(stream.sync)
        first, second = _import_twice(stack, mr_type, device, alloc_handle)
        assert _live_owned_mempool_count() == baseline + 1

        first.close()
        assert pool_access_flags(second, device) == driver.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE

        buffer = stack.enter_context(Buffer.from_ipc_descriptor(second, descriptor, stream=stream))
        buffer.fill(7, stream=stream)
        stream.sync()


def child_close_both_imports_with_outstanding_buffer(mr_type, alloc_handle, descriptor):
    # The closes below are the sequence under test; the stack repeats them as
    # no-ops, and releases whatever a failed assertion leaves open. It drains the
    # stream last, so a failure partway still completes the imported frees.
    with _child_device() as (device, baseline), assert_no_cuda_warning(), contextlib.ExitStack() as stack:
        stream = device.default_stream
        stack.callback(stream.sync)
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
        # The final release revokes access one GPU at a time, so check every GPU the
        # pool grants access to, not only the default one.
        (mr,) = reimported
        (buffer,) = reimported_buffers
        for access_device in pool_supported_devices(mr_type, device):
            assert pool_access_flags(mr, access_device) == driver.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
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

        _register_one_uuid_during_collections(stack, mr_type, device, alloc_handle)


def _register_one_uuid_during_collections(stack, mr_type, device, alloc_handle):
    # Two imports register one UUID, and every collection that starts while the
    # first registers makes the second register it too. However the two
    # interleave, both calls must return the one resource the registry holds.
    registering = []
    nested_results = []

    def register_during_collection(phase, info):
        if phase == "start" and registering:
            key, rival = registering
            nested_results.append((key, rival.register(key)))

    threshold = gc.get_threshold()
    gc.callbacks.append(register_during_collection)
    try:
        gc.set_threshold(1)
        for _ in range(REENTRY_ITERATIONS):
            mr = _enter_import(stack, mr_type, device, int(alloc_handle))
            rival = _enter_import(stack, mr_type, device, int(alloc_handle))
            key = uuid.uuid4()
            registering[:] = [key, rival]
            try:
                registered = mr.register(key)
            finally:
                registering.clear()
            assert registered is mr or registered is rival
            assert mr_type.from_registry(key) is registered
            assert all(result is registered for k, result in nested_results if k == key)
    finally:
        gc.set_threshold(*threshold)
        gc.callbacks.remove(register_during_collection)


class _TripwireUUID(uuid.UUID):
    """A UUID that calls ``tripwire`` whenever Python code reads its value.

    ``__hash__``, ``__eq__`` and ``bytes`` are how code that keeps a UUID in a
    registry reaches the value. Each is a point where a garbage-collection
    callback or finalizer could run in the middle of a registration, so a test
    can inject code at every such point without a hook in the implementation.
    """

    def __init__(self, tripwire):
        super().__init__(int=uuid.uuid4().int)
        object.__setattr__(self, "_tripwire", tripwire)

    def __hash__(self):
        self._tripwire()
        return super().__hash__()

    def __eq__(self, other):
        self._tripwire()
        return super().__eq__(other)

    @property
    def bytes(self):
        self._tripwire()
        return super().bytes


def _register_interrupted_at(index, register, action):
    """Call ``register(key)``, running ``action(key)`` at the ``index``-th use of ``key`` inside it.

    Returns ``(key, fired, outcome)``: whether ``action`` ran, and what the call
    returned or raised.
    """
    state = {"armed": True, "remaining": index, "fired": False}

    def tripwire():
        if not state["armed"] or state["fired"]:
            return
        state["remaining"] -= 1
        if state["remaining"] == 0:
            state["fired"] = True
            action(key)

    key = _TripwireUUID(tripwire)
    try:
        outcome = register(key)
    except Exception as e:  # the outcome under test
        outcome = e
    finally:
        state["armed"] = False
    return key, state["fired"], outcome


def _run_interrupted_registration(mr_type, device, alloc_handle, case, shared_handle, index):
    """Register one UUID with an import that is interrupted at the ``index``-th use of the key.

    Returns whether the interruption happened. ``case`` is what the
    interruption does: ``keep`` registers the same UUID with another import,
    ``close`` does that and then closes the winner, and ``fail`` raises.
    """
    with contextlib.ExitStack() as stack:
        first = _enter_import(stack, mr_type, device, int(alloc_handle))
        # Two imports share a handle only when one is built from the other's
        # handle, which carries no UUID until one of them registers.
        second_source = first.allocation_handle if shared_handle else int(alloc_handle)
        second = _enter_import(stack, mr_type, device, second_source)
        assert (second.allocation_handle is first.allocation_handle) == shared_handle
        nested = []

        def interrupt(key):
            if case == "fail":
                raise MemoryError("simulated failure at a use of the key")
            nested.append(second.register(key))
            if case == "close":
                second.close()

        key, fired, outcome = _register_interrupted_at(index, first.register, interrupt)

        if not fired:
            assert outcome is first
            return False
        if case == "fail":
            assert isinstance(outcome, MemoryError)
            # The failed call must leave the resource as it was.
            assert first.uuid is None
            with pytest.raises(RuntimeError, match="was not found"):
                mr_type.from_registry(key)
            other = uuid.uuid4()
            assert first.register(other) is first
            assert first.uuid == other
            return True
        (winner,) = nested
        if case == "keep":
            assert winner.uuid == key
            assert outcome is winner
            assert mr_type.from_registry(key) is winner
        else:
            # The call found the closed winner, as it would for any closed hit.
            assert outcome is winner or isinstance(outcome, RuntimeError)
        return True


def child_register_interrupted_at_every_key_use(mr_type, alloc_handle, case, shared_handle):
    with _child_device() as (device, _baseline):
        index = 1
        while _run_interrupted_registration(mr_type, device, alloc_handle, case, shared_handle, index):
            index += 1
        # Without a first interruption the sweep would prove nothing.
        assert index > 1, "the registration never used its key"


def child_dead_registry_entry_is_removed(mr_type, alloc_handle):
    with _child_device() as (device, _baseline), contextlib.ExitStack() as stack:
        key = uuid.uuid4()
        dead = import_pool_resource(mr_type, device, int(alloc_handle))
        assert dead.register(key) is dead
        dead.close()
        del dead
        with pytest.raises(RuntimeError, match="was not found"):
            mr_type.from_registry(key)

        # The UUID is free again for another import.
        successor = _enter_import(stack, mr_type, device, int(alloc_handle))
        assert successor.register(key) is successor
        assert mr_type.from_registry(key) is successor


def child_shared_handle_keeps_its_uuid_object(mr_type, alloc_handle):
    with _child_device() as (device, _baseline), contextlib.ExitStack() as stack:
        first = import_pool_resource(mr_type, device, int(alloc_handle))
        # Built from the first import's handle while it carries no UUID, so the
        # two imports share that handle.
        second = _enter_import(stack, mr_type, device, first.allocation_handle)
        assert second.allocation_handle is first.allocation_handle
        key = uuid.uuid4()
        assert first.register(key) is first
        first.close()
        del first
        with pytest.raises(RuntimeError, match="was not found"):
            mr_type.from_registry(key)

        # The handle outlived its first owner and still has the UUID. Registering
        # the second import under an equal UUID must not replace that object:
        # the handle's UUID is read without a lock.
        equal = uuid.UUID(bytes=key.bytes)
        assert second.register(equal) is second
        assert second.uuid is key
        assert mr_type.from_registry(equal) is second


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
        # On every path, release the latch and then drain the stream: the
        # imported frees must complete before the exporter frees its allocation,
        # and a held latch would stall the drain.
        stack.callback(stream.sync)
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
    @pytest.mark.parametrize("shared_handle", [False, True], ids=["separate_handles", "shared_handle"])
    @pytest.mark.parametrize("case", ["keep", "close", "fail"])
    def test_registration_interrupted_at_every_use_of_its_key(self, ipc_device, pool_resource, case, shared_handle):
        ipc_device.set_current()
        run_child(
            child_register_interrupted_at_every_key_use,
            type(pool_resource),
            pool_resource.allocation_handle,
            case,
            shared_handle,
        )

    @pytest.mark.agent_authored(model="claude-opus-5-5")
    def test_registry_entry_of_a_dropped_resource_is_removed(self, ipc_device, pool_resource):
        ipc_device.set_current()
        run_child(child_dead_registry_entry_is_removed, type(pool_resource), pool_resource.allocation_handle)

    @pytest.mark.agent_authored(model="claude-fable-5-2")
    def test_shared_handle_keeps_its_uuid_object_after_its_first_owner(self, ipc_device, pool_resource):
        ipc_device.set_current()
        run_child(child_shared_handle_keeps_its_uuid_object, type(pool_resource), pool_resource.allocation_handle)

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
