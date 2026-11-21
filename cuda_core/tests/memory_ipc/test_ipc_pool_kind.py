# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Importing a memory pool with the class of the other pool kind raises (#3502).

``DeviceMemoryResource.from_allocation_handle`` used to accept the handle of a
pinned (host) pool, and ``PinnedMemoryResource.from_allocation_handle`` the
handle of a device pool, unless the resource was already registered in the
importing process. The import now checks the pool's location before the
resource is registered. The check needs a CUDA 13 build of cuda.core and a
13.2 or later driver; with either missing, imports behave as before.
"""

import contextlib
import multiprocessing as mp

import pytest
from helpers.child_processes import run_child
from helpers.constants import POOL_SIZE
from helpers.contexts import assert_no_cuda_warning
from helpers.memory import (
    drain_on_exit,
    enter_imported_pool,
    import_pool_resource,
    is_pool_resource,
    pool_access_flags,
    skip_if_host_numa_pinned_memory_unsupported,
    skip_if_pool_location_query_unsupported,
)

from cuda.bindings import driver
from cuda.core import (
    Buffer,
    Device,
    DeviceMemoryResource,
    PinnedMemoryResource,
    PinnedMemoryResourceOptions,
)
from cuda.core._memory import _memory_pool
from cuda.core._memory._buffer import _live_owned_mempool_count
from cuda.core._memory._pinned_memory_resource import _deep_reduce_pinned_memory_resource

NBYTES = 64

# these tests spawn new processes and files which fails for very many threads;
# child processes and IPC fixtures are shared across workers (#2784); cap at two threads
pytestmark = pytest.mark.parallel_threads_limit(2)


def _nonzero_numa_node_with_memory():
    """The first host NUMA node other than 0 that has memory, or None."""
    try:
        with open("/sys/devices/system/node/has_memory") as f:
            node_list = f.read().strip()
    except FileNotFoundError:
        return None
    for node_range in node_list.split(","):
        first, _, last = node_range.partition("-")
        for node in range(int(first), int(last or first) + 1):
            if node > 0:
                return node
    return None


def _other_class(mr_type):
    return DeviceMemoryResource if mr_type is PinnedMemoryResource else PinnedMemoryResource


def _assert_import_raises(mr_type, device, alloc_handle):
    # Releasing the rejected import cannot raise; a failed cuMemPoolDestroy
    # surfaces only as a CUDAWarning.
    with assert_no_cuda_warning(), pytest.raises(TypeError):
        import_pool_resource(_other_class(mr_type), device, alloc_handle)


def _access_by_device(mr):
    return {dev.device_id: pool_access_flags(mr, dev) for dev in Device.get_all_devices()}


def child_mismatched_import_raises(mr_type, alloc_handle, descriptor):
    device = Device()
    device.set_current()
    stream = device.default_stream
    baseline = _live_owned_mempool_count()

    # Neither handle has a registered resource in this process: a raw file
    # descriptor carries no UUID, and nothing has imported the UUID yet.
    _assert_import_raises(mr_type, device, int(alloc_handle))
    _assert_import_raises(mr_type, device, alloc_handle)
    assert _live_owned_mempool_count() == baseline

    # The rejected imports did not register the UUID, so the matching class imports it.
    with contextlib.ExitStack() as stack:
        drain_on_exit(stack, stream)
        mr = enter_imported_pool(stack, mr_type, device, alloc_handle)
        buffer = stack.enter_context(Buffer.from_ipc_descriptor(mr, descriptor, stream=stream))
        buffer.fill(7, stream=stream)
    assert _live_owned_mempool_count() == baseline


def child_rejected_import_keeps_existing_import(mr_type, alloc_handle, descriptor):
    device = Device()
    device.set_current()
    stream = device.default_stream

    # The driver returns the same pool for every import in a process, so the
    # rejected import must not destroy the pool or revoke any device's access
    # to it, including peer access and while a buffer from it is alive.
    with contextlib.ExitStack() as stack:
        drain_on_exit(stack, stream)
        mr = enter_imported_pool(stack, mr_type, device, int(alloc_handle))
        if mr_type is DeviceMemoryResource:
            mr.peer_accessible_by = [
                dev
                for dev in Device.get_all_devices()
                if dev.device_id != device.device_id and dev.can_access_peer(device)
            ]
        buffer = stack.enter_context(Buffer.from_ipc_descriptor(mr, descriptor, stream=stream))
        access = _access_by_device(mr)
        assert access[device.device_id] == driver.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE

        _assert_import_raises(mr_type, device, int(alloc_handle))
        assert _access_by_device(mr) == access

        buffer.fill(7, stream=stream)


def child_imported_numa_id(alloc_handle, numa_id):
    Device().set_current()
    mr = PinnedMemoryResource.from_allocation_handle(alloc_handle)
    assert mr.numa_id == numa_id
    mr.close()


def child_unpickled_numa_id(mr, numa_id):
    # The resource was unpickled while the child's arguments were loaded.
    assert mr.numa_id == numa_id
    mr.close()


def _without_location_query():
    # Stand in for a 13.0 or 13.1 driver (see child_import_without_location_query).
    _memory_pool.driver_version = lambda: (13, 1, 0)


# A node that the tests cannot confuse with the default of 0. Without a location
# query nothing checks it against the pool.
CLAIMED_NUMA_ID = 7


def child_import_with_numa_id_without_location_query(alloc_handle):
    _without_location_query()
    Device().set_current()

    # A raw file descriptor carries no UUID, so each import is a separate resource.
    omitted = PinnedMemoryResource.from_allocation_handle(int(alloc_handle))
    assert omitted.numa_id == 0
    given = PinnedMemoryResource.from_allocation_handle(int(alloc_handle), numa_id=CLAIMED_NUMA_ID)
    assert given.numa_id == CLAIMED_NUMA_ID
    given.close()
    omitted.close()


def child_import_reduced_args_without_location_query(reduced_args):
    _without_location_query()
    Device().set_current()
    mr = PinnedMemoryResource.from_allocation_handle(*reduced_args)
    assert mr.numa_id == reduced_args[1]
    mr.close()


def child_reduce_imported_resource(alloc_handle):
    _without_location_query()
    Device().set_current()

    # The node of this import is unknown, so a third process must not be told 0.
    unknown = PinnedMemoryResource.from_allocation_handle(int(alloc_handle))
    assert unknown.numa_id == 0
    assert _deep_reduce_pinned_memory_resource(unknown)[1][1] is None

    known = PinnedMemoryResource.from_allocation_handle(int(alloc_handle), numa_id=CLAIMED_NUMA_ID)
    assert _deep_reduce_pinned_memory_resource(known)[1][1] == CLAIMED_NUMA_ID
    known.close()
    unknown.close()


def child_registered_import_rejects_conflicting_numa_id(alloc_handle):
    _without_location_query()
    Device().set_current()
    baseline = _live_owned_mempool_count()

    mr = PinnedMemoryResource.from_allocation_handle(alloc_handle, numa_id=CLAIMED_NUMA_ID)
    # The handle's UUID is now registered, so the resource is returned as is.
    assert PinnedMemoryResource.from_allocation_handle(alloc_handle, numa_id=CLAIMED_NUMA_ID) is mr
    with pytest.raises(ValueError, match="numa_id"):
        PinnedMemoryResource.from_allocation_handle(alloc_handle, numa_id=CLAIMED_NUMA_ID + 1)
    assert PinnedMemoryResource.from_allocation_handle(alloc_handle) is mr
    assert mr.numa_id == CLAIMED_NUMA_ID

    mr.close()
    assert _live_owned_mempool_count() == baseline


def child_numa_id_is_checked_against_driver(alloc_handle):
    Device().set_current()
    baseline = _live_owned_mempool_count()

    mr = PinnedMemoryResource.from_allocation_handle(int(alloc_handle))
    node = mr.numa_id
    # Releasing the rejected import cannot raise; a failed cuMemPoolDestroy
    # surfaces only as a CUDAWarning.
    with assert_no_cuda_warning(), pytest.raises(ValueError, match="numa_id"):
        PinnedMemoryResource.from_allocation_handle(int(alloc_handle), numa_id=node + 1)

    matching = PinnedMemoryResource.from_allocation_handle(int(alloc_handle), numa_id=node)
    assert matching.numa_id == node
    matching.close()
    mr.close()
    assert _live_owned_mempool_count() == baseline


def child_matching_import(mr_type, alloc_handle, descriptor):
    device = Device()
    device.set_current()
    stream = device.default_stream

    with contextlib.ExitStack() as stack:
        drain_on_exit(stack, stream)
        # The raw file descriptor and the IPCAllocationHandle are the two import paths.
        from_fd = enter_imported_pool(stack, mr_type, device, int(alloc_handle))
        from_handle = enter_imported_pool(stack, mr_type, device, alloc_handle)
        assert isinstance(from_fd, mr_type)
        assert isinstance(from_handle, mr_type)

        buffer = stack.enter_context(Buffer.from_ipc_descriptor(from_fd, descriptor, stream=stream))
        buffer.fill(7, stream=stream)


def child_import_without_location_query(mr_type, alloc_handle, descriptor):
    # Stand in for a 13.0 or 13.1 driver, as test_program_compile_cache.py does
    # for _program: MP_location_query_available reads driver_version from the
    # _memory_pool module on every import. The monkeypatch fixture cannot reach a
    # spawned child, and this process exits after the test.
    _memory_pool.driver_version = lambda: (13, 1, 0)
    device = Device()
    device.set_current()
    stream = device.default_stream

    # Importing with the exporter's class must not need the location query. A
    # wrong-class import is not tested here: without the pool kind check it
    # runs the other class's initialization on the wrong kind of pool, which
    # can fail on some topologies (a pinned import of a device pool grants
    # every device access, which needs peer access; #3540).
    with contextlib.ExitStack() as stack:
        drain_on_exit(stack, stream)
        mr = enter_imported_pool(stack, mr_type, device, int(alloc_handle))
        if mr_type is PinnedMemoryResource:
            # Without the location query the pool's node is unknown (see numa_id).
            assert mr.numa_id == 0
        buffer = stack.enter_context(Buffer.from_ipc_descriptor(mr, descriptor, stream=stream))
        buffer.fill(7, stream=stream)


class _IpcPoolFixtures:
    @pytest.fixture(autouse=True)
    def _set_start_method(self):
        # Ensure spawn is used for multiprocessing
        with contextlib.suppress(RuntimeError):
            mp.set_start_method("spawn", force=True)

    @pytest.fixture
    def pinned_ipc_resource(self, ipc_device):
        skip_if_host_numa_pinned_memory_unsupported(ipc_device)
        ipc_device.set_current()
        mr = PinnedMemoryResource(PinnedMemoryResourceOptions(max_size=POOL_SIZE, ipc_enabled=True))
        yield mr
        mr.close()
        ipc_device.sync()

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


class TestIpcPoolMatchingImport(_IpcPoolFixtures):
    """Importing a pool with the exporter's class works whether or not the pool's kind can be checked."""

    @pytest.mark.agent_authored(model="claude-opus-5-5")
    def test_matching_import_works(self, pool_resource, exported_buffer):
        run_child(
            child_matching_import,
            type(pool_resource),
            pool_resource.allocation_handle,
            exported_buffer.ipc_descriptor,
        )


class TestIpcPoolImportWithoutLocationQuery(_IpcPoolFixtures):
    """With a driver before 13.2, importing with the exporter's class works without the pool kind check."""

    @pytest.mark.agent_authored(model="claude-opus-5-5")
    def test_import_without_location_query(self, pool_resource, exported_buffer):
        run_child(
            child_import_without_location_query,
            type(pool_resource),
            pool_resource.allocation_handle,
            exported_buffer.ipc_descriptor,
        )


class TestIpcPoolKind(_IpcPoolFixtures):
    """An import checks that the pool's kind matches the requested class."""

    @pytest.fixture(autouse=True)
    def _require_pool_location_query(self):
        skip_if_pool_location_query_unsupported()

    @pytest.mark.agent_authored(model="claude-opus-5-5")
    def test_mismatched_import_raises(self, pool_resource, exported_buffer):
        run_child(
            child_mismatched_import_raises,
            type(pool_resource),
            pool_resource.allocation_handle,
            exported_buffer.ipc_descriptor,
        )

    @pytest.mark.agent_authored(model="claude-opus-5-5")
    def test_rejected_import_keeps_existing_import(self, pool_resource, exported_buffer):
        run_child(
            child_rejected_import_keeps_existing_import,
            type(pool_resource),
            pool_resource.allocation_handle,
            exported_buffer.ipc_descriptor,
        )

    @pytest.mark.agent_authored(model="claude-opus-5-5")
    def test_imported_pinned_resource_reports_numa_id(self, ipc_device):
        skip_if_host_numa_pinned_memory_unsupported(ipc_device)
        # An imported resource used to report 0, so node 0 cannot tell the two apart.
        numa_id = _nonzero_numa_node_with_memory()
        if numa_id is None:
            pytest.skip("needs a host NUMA node other than 0 with memory")
        ipc_device.set_current()
        mr = PinnedMemoryResource(PinnedMemoryResourceOptions(max_size=POOL_SIZE, ipc_enabled=True, numa_id=numa_id))
        try:
            assert mr.numa_id == numa_id
            run_child(child_imported_numa_id, mr.allocation_handle, numa_id)
            # Unpickling the resource is the other import path.
            run_child(child_unpickled_numa_id, mr, numa_id)
        finally:
            mr.close()

    @pytest.mark.agent_authored(model="claude-sonnet-5-5")
    def test_numa_id_is_checked_against_driver(self, pinned_ipc_resource):
        run_child(child_numa_id_is_checked_against_driver, pinned_ipc_resource.allocation_handle)


class TestPinnedImportNumaId(_IpcPoolFixtures):
    """Without a location query, an imported pinned pool's node comes from the caller or the pickle (#3508)."""

    @pytest.mark.agent_authored(model="claude-sonnet-5-5")
    def test_import_uses_given_numa_id(self, pinned_ipc_resource):
        run_child(child_import_with_numa_id_without_location_query, pinned_ipc_resource.allocation_handle)

    @pytest.mark.agent_authored(model="claude-sonnet-5-5")
    def test_pickle_carries_numa_id(self, pinned_ipc_resource):
        _, reduced_args = _deep_reduce_pinned_memory_resource(pinned_ipc_resource)
        assert reduced_args[0] is pinned_ipc_resource.allocation_handle
        assert reduced_args[1] == pinned_ipc_resource.numa_id
        run_child(child_import_reduced_args_without_location_query, reduced_args)

    @pytest.mark.agent_authored(model="claude-sonnet-5-5")
    def test_pickle_omits_unknown_numa_id(self, pinned_ipc_resource):
        run_child(child_reduce_imported_resource, pinned_ipc_resource.allocation_handle)

    @pytest.mark.agent_authored(model="claude-sonnet-5-5")
    def test_registered_import_rejects_conflicting_numa_id(self, pinned_ipc_resource):
        run_child(child_registered_import_rejects_conflicting_numa_id, pinned_ipc_resource.allocation_handle)

    @pytest.mark.agent_authored(model="claude-sonnet-5-5")
    @pytest.mark.parametrize("numa_id", [-1, -5])
    def test_negative_numa_id_raises(self, numa_id):
        # Rejected before the handle is used or CUDA is initialized.
        with pytest.raises(ValueError, match="numa_id"):
            PinnedMemoryResource.from_allocation_handle(0, numa_id=numa_id)
