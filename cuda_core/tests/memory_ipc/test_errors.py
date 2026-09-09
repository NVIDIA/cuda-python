# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import multiprocessing
import os
import pickle
import platform
import re
import uuid

import pytest
from helpers.child_processes import child_timeout_sec, kill_subprocesses
from helpers.constants import POOL_SIZE

from cuda.core import (
    Buffer,
    Device,
    DeviceMemoryResource,
    DeviceMemoryResourceOptions,
    PinnedMemoryResource,
)
from cuda.core._memory._ipc import IPCAllocationHandle, IPCBufferDescriptor
from cuda.core._utils.cuda_utils import CUDAError

CHILD_TIMEOUT_SEC = child_timeout_sec()
NBYTES = 64


# these tests spawn new processes and files which fails for very many threads
pytestmark = pytest.mark.parallel_threads_limit(4)


def test_outer_timeout_marker_is_applied(request):
    """Verify that memory_ipc/conftest.py applies the outer pytest-timeout marker.

    If this test fails, the per-directory conftest is not being loaded, or its
    pytest_collection_modifyitems hook is not adding the marker. Without this
    marker, the only thing protecting the GHA runner from a wedged IPC test is
    the in-test cleanup -- which we want to keep as defense in depth, not as
    the sole guard.
    """
    expected = child_timeout_sec() + 30
    marker = request.node.get_closest_marker("timeout")
    assert marker is not None, "memory_ipc/conftest.py did not apply a timeout marker"
    assert marker.args == (expected,), f"unexpected timeout value: {marker.args!r}"


@pytest.mark.agent_authored(model="gpt-5.6-sol")
def test_ipc_types_cannot_be_constructed_directly():
    """Factory-only IPC types reject direct construction."""
    with pytest.raises(RuntimeError, match=r"^IPCBufferDescriptor objects cannot be instantiated directly\."):
        IPCBufferDescriptor()
    with pytest.raises(RuntimeError, match=r"^IPCAllocationHandle objects cannot be instantiated directly\."):
        IPCAllocationHandle()


def test_import_truncated_buffer_descriptor(ipc_device, ipc_memory_resource):
    """Truncated IPC buffer descriptor payload is rejected before driver import."""
    desc = IPCBufferDescriptor._init(b"\x00" * 8, NBYTES)
    with pytest.raises(ValueError, match=r"payload is 8 bytes; expected at least 64"):
        Buffer.from_ipc_descriptor(ipc_memory_resource, desc, stream=ipc_device.default_stream)


def test_ipc_allocation_handle_rejects_negative_fd():
    """Negative fds are rejected even when CPython runs with -O (Glasswing V3.2)."""
    with pytest.raises(ValueError, match=r"Invalid allocation handle \(fd\) -1: must be non-negative"):
        IPCAllocationHandle._init(-1, None)


@pytest.mark.human_authored
def test_register_rejects_non_ipc_memory_resource(mempool_device):
    """register() on a resource without IPC enabled raises instead of dereferencing None."""
    mr = DeviceMemoryResource(mempool_device)
    assert not mr.is_ipc_enabled

    key = uuid.uuid4()
    with pytest.raises(RuntimeError, match="Memory resource is not IPC-enabled"):
        mr.register(key)

    # The rejected registration must not leave the resource in the registry.
    with pytest.raises(RuntimeError, match=r"Memory resource [a-z0-9-]+ was not found"):
        DeviceMemoryResource.from_registry(key)


@pytest.mark.skipif(os.name == "nt", reason="IPC allocation handles are not supported on Windows")
@pytest.mark.agent_authored(model="gpt-5.6")
def test_ipc_allocation_handle_state_tracks_close():
    read_fd, write_fd = os.pipe()
    handle = IPCAllocationHandle._init(read_fd, None)
    try:
        assert not handle.is_closed
        handle.close()
        assert handle.is_closed
        assert bool(handle) is True  # Preserve backward-compatible truthiness after close.
        with pytest.raises(ValueError, match="is closed"):
            int(handle)
    finally:
        handle.close()
        os.close(write_fd)


@pytest.mark.agent_authored(model="gpt-5.6")
def test_closed_ipc_allocation_handle_rejected_before_registry_hit(ipc_device, ipc_memory_resource):
    mr = ipc_memory_resource
    handle = IPCAllocationHandle._init(os.dup(int(mr.allocation_handle)), mr.uuid)
    assert mr.register(mr.uuid) is mr
    handle.close()

    with pytest.raises(RuntimeError, match="IPCAllocationHandle has been closed"):
        if isinstance(mr, DeviceMemoryResource):
            DeviceMemoryResource.from_allocation_handle(ipc_device, handle)
        else:
            assert isinstance(mr, PinnedMemoryResource)
            PinnedMemoryResource.from_allocation_handle(handle)


class ChildErrorHarness:
    """Test harness for checking errors in child processes. Subclasses override
    PARENT_ACTION, CHILD_ACTION, and ASSERT (see below for examples)."""

    @pytest.mark.flaky(reruns=2)
    def test_main(self, ipc_device, ipc_memory_resource):
        """Parent process that checks child errors."""
        # Attach fixtures to this object for convenience. These can be accessed
        # from PARENT_ACTION.
        self.device = ipc_device
        self.mr = ipc_memory_resource
        self._extra_mrs = []

        try:
            # Start a child process to generate error info.
            pipe = [multiprocessing.Queue() for _ in range(2)]
            process = multiprocessing.Process(target=self.child_main, args=(pipe, self.device, self.mr))
            process.start()

            # Interact.
            self.PARENT_ACTION(pipe[0])

            # Check the error.
            exc_type, exc_msg = pipe[1].get(timeout=CHILD_TIMEOUT_SEC)
            self.ASSERT(exc_type, exc_msg)

            # Wait for the child process.
            process.join(timeout=CHILD_TIMEOUT_SEC)
            survivors = kill_subprocesses(process)
            assert not survivors, "child did not exit within timeout"
            assert process.exitcode == 0
        finally:
            for mr in self._extra_mrs:
                mr.close()

    def child_main(self, pipe, device, mr):
        """Child process that pushes IPC errors to a shared pipe for testing."""
        self.device = device
        self.device.set_current()
        self.mr = mr
        try:
            self.CHILD_ACTION(pipe[0])
        except Exception as e:
            exc_info = type(e), str(e)
        else:
            exc_info = None, None
        pipe[1].put(exc_info)


class TestImportOversizedBufferDescriptorSize(ChildErrorHarness):
    """Reject peer-supplied sizes larger than the mapped allocation extent."""

    def PARENT_ACTION(self, queue):
        stream = self.device.default_stream
        self.buffer = self.mr.allocate(NBYTES, stream=stream)
        payload, _ = self.buffer.ipc_descriptor.__reduce__()[1]
        oversized = IPCBufferDescriptor._init(payload, NBYTES * 100)
        stream.sync()
        queue.put(oversized)

    def CHILD_ACTION(self, queue):
        oversized = queue.get(timeout=CHILD_TIMEOUT_SEC)
        Buffer.from_ipc_descriptor(self.mr, oversized, stream=self.device.default_stream)

    def ASSERT(self, exc_type, exc_msg):
        assert exc_type is ValueError
        assert "exceeds" in exc_msg
        assert "mapped allocation extent" in exc_msg


class TestAllocFromImportedMr(ChildErrorHarness):
    """Error when attempting to allocate from an import memory resource."""

    def PARENT_ACTION(self, queue):
        queue.put(self.mr)

    def CHILD_ACTION(self, queue):
        mr = queue.get(timeout=CHILD_TIMEOUT_SEC)
        mr.allocate(NBYTES, stream=self.device.default_stream)

    def ASSERT(self, exc_type, exc_msg):
        assert exc_type is TypeError
        assert exc_msg == "Cannot allocate from a mapped IPC-enabled memory resource"


class TestImportWrongMR(ChildErrorHarness):
    """Error when importing a buffer from the wrong memory resource."""

    def PARENT_ACTION(self, queue):
        options = DeviceMemoryResourceOptions(max_size=POOL_SIZE, ipc_enabled=True)
        mr2 = DeviceMemoryResource(self.device, options=options)
        self._extra_mrs.append(mr2)
        stream = self.device.default_stream
        self.buffer = mr2.allocate(NBYTES, stream=stream)
        stream.sync()
        queue.put([self.mr, self.buffer.ipc_descriptor])  # Note: mr does not own this buffer

    def CHILD_ACTION(self, queue):
        mr, buffer_desc = queue.get(timeout=CHILD_TIMEOUT_SEC)
        Buffer.from_ipc_descriptor(mr, buffer_desc, stream=self.device.default_stream)

    def ASSERT(self, exc_type, exc_msg):
        assert exc_type is CUDAError
        assert "CUDA_ERROR_INVALID_VALUE" in exc_msg


class TestImportBuffer(ChildErrorHarness):
    """Error when using a buffer as a buffer descriptor."""

    def PARENT_ACTION(self, queue):
        # Note: if the buffer is not attached to something to prolong its life,
        # CUDA_ERROR_INVALID_CONTEXT is raised from Buffer.__del__
        stream = self.device.default_stream
        self.buffer = self.mr.allocate(NBYTES, stream=stream)
        stream.sync()
        queue.put(self.buffer)

    def CHILD_ACTION(self, queue):
        buffer = queue.get(timeout=CHILD_TIMEOUT_SEC)
        Buffer.from_ipc_descriptor(self.mr, buffer, stream=self.device.default_stream)

    def ASSERT(self, exc_type, exc_msg):
        assert exc_type is TypeError
        assert exc_msg.startswith("Argument 'ipc_descriptor' has incorrect type")


class TestDanglingBuffer(ChildErrorHarness):
    """
    Error when importing a buffer object without registering its memory
    resource.
    """

    def PARENT_ACTION(self, queue):
        options = DeviceMemoryResourceOptions(max_size=POOL_SIZE, ipc_enabled=True)
        mr2 = DeviceMemoryResource(self.device, options=options)
        self._extra_mrs.append(mr2)
        stream = self.device.default_stream
        self.buffer = mr2.allocate(NBYTES, stream=stream)
        buffer_s = pickle.dumps(self.buffer)
        stream.sync()
        queue.put(buffer_s)  # Note: mr2 not sent

    def CHILD_ACTION(self, queue):
        Device().set_current()
        buffer_s = queue.get(timeout=CHILD_TIMEOUT_SEC)
        pickle.loads(buffer_s)  # noqa: S301

    def ASSERT(self, exc_type, exc_msg):
        assert exc_type is RuntimeError
        assert re.match(r"Memory resource [a-z0-9-]+ was not found", exc_msg)


@pytest.mark.skipif(platform.system() != "Linux", reason="CUDA mempool IPC is Linux-only")
@pytest.mark.agent_authored(model="gpt-5.6-sol")
def test_from_allocation_handle_raw_fd_imports_mapped_pool(ipc_device):
    """from_allocation_handle accepts a raw int fd and constructs an unregistered mapped MR."""
    from helpers.buffers import PatternGen

    device = ipc_device
    stream = device.default_stream
    exporter = DeviceMemoryResource(device, DeviceMemoryResourceOptions(max_size=POOL_SIZE, ipc_enabled=True))
    try:
        dup_fd = os.dup(exporter.allocation_handle.handle)
        try:
            imported = DeviceMemoryResource.from_allocation_handle(device, dup_fd)
        finally:
            # The int overload dups the fd internally, so the imported pool must
            # outlive the caller's copy: everything below runs without it.
            os.close(dup_fd)
        try:
            assert imported.is_mapped
            assert imported.is_ipc_enabled
            assert imported.device_id == device.device_id
            # No uuid was supplied, so the pool never entered the registry.
            assert imported.uuid is None
            # Mapped pools cannot allocate; import a peer buffer instead.
            with exporter.allocate(NBYTES, stream=stream) as exported:
                descriptor = exported.ipc_descriptor
                with Buffer.from_ipc_descriptor(imported, descriptor, stream=stream) as mapped_buf:
                    pgen = PatternGen(device, NBYTES, stream=stream)
                    pgen.fill_buffer(mapped_buf, seed=1)
                    pgen.verify_buffer(exported, seed=1)
        finally:
            imported.close()
    finally:
        exporter.close()


@pytest.mark.skipif(platform.system() != "Linux", reason="CUDA mempool IPC is Linux-only")
@pytest.mark.agent_authored(model="gpt-5.6-sol")
def test_allocation_handle_forking_pickler_roundtrip(ipc_device):
    """ForkingPickler transfers an IPCAllocationHandle by duplicating its fd."""
    from multiprocessing.reduction import ForkingPickler

    device = ipc_device
    mr = DeviceMemoryResource(device, DeviceMemoryResourceOptions(max_size=POOL_SIZE, ipc_enabled=True))
    try:
        handle = mr.allocation_handle
        restored = ForkingPickler.loads(ForkingPickler.dumps(handle))
        try:
            assert isinstance(restored, IPCAllocationHandle)
            # DupFd must hand back a real duplicate: a shared fd number would mean
            # restored.close() also clobbers the exporter's handle. The number
            # itself is unpredictable, so only check validity and distinctness.
            assert restored.handle > 0
            assert restored.handle != handle.handle
            assert restored.uuid == handle.uuid
        finally:
            restored.close()
    finally:
        mr.close()


@pytest.mark.skipif(platform.system() != "Linux", reason="CUDA mempool IPC is Linux-only")
@pytest.mark.agent_authored(model="gpt-5.6-sol")
def test_ipc_registry_dedups_repeated_imports(ipc_device):
    """from_allocation_handle registers the mapped pool; later imports hit the cache."""
    device = ipc_device
    exporter = DeviceMemoryResource(device, DeviceMemoryResourceOptions(max_size=POOL_SIZE, ipc_enabled=True))
    mapped = None
    try:
        key = exporter.uuid
        mapped = DeviceMemoryResource.from_allocation_handle(device, exporter.allocation_handle)
        assert mapped.is_mapped
        assert DeviceMemoryResource.from_registry(key) is mapped
        mapped2 = DeviceMemoryResource.from_allocation_handle(device, exporter.allocation_handle)
        assert mapped2 is mapped
        # Registering under a key that is already taken hands back the existing
        # entry, so the exporter itself never enters the registry.
        assert exporter.register(key) is mapped
    finally:
        if mapped is not None:
            mapped.close()
        exporter.close()
