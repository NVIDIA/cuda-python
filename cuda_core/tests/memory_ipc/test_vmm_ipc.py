# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sharing VirtualMemoryResource buffers between processes (issue #2980)."""

import ctypes
import gc
import multiprocessing as mp
import os
import pickle

import pytest
from helpers import IS_WINDOWS, IS_WSL
from helpers.child_processes import child_timeout_sec, kill_subprocesses, track_child_processes

from cuda.bindings import driver
from cuda.core import Buffer, Device, MemoryResource, VirtualMemoryResource, VirtualMemoryResourceOptions
from cuda.core._memory._ipc import IPCBufferDescriptor
from cuda.core._memory._virtual_memory_resource import VirtualMemoryBuffer
from cuda.core._utils.cuda_utils import handle_return

CHILD_TIMEOUT_SEC = child_timeout_sec()

# These tests spawn processes and pass file descriptors, which fails for very many threads.
pytestmark = pytest.mark.parallel_threads_limit(4)


@pytest.fixture
def vmm_ipc_device(init_cuda):
    """A device that can export VMM allocations as POSIX file descriptors, or skip."""
    device = init_cuda
    props = device.properties
    if not props.virtual_memory_management_supported:
        pytest.skip("Virtual memory management is not supported on this device")
    if IS_WINDOWS or IS_WSL or not props.handle_type_posix_file_descriptor_supported:
        pytest.skip("Sharing virtual memory needs POSIX file descriptor handles")
    with track_child_processes():
        yield device


def _resource(device, **options):
    options.setdefault("handle_type", "posix_fd")
    return VirtualMemoryResource(device, config=VirtualMemoryResourceOptions(**options))


def _fill(buf, value, *, offset=0, size=None):
    """Write ``value`` to ``size`` bytes of ``buf`` at ``offset`` and wait for the write."""
    size = buf.size - offset if size is None else size
    handle_return(driver.cuMemsetD8(int(buf.handle) + offset, value, size))
    handle_return(driver.cuCtxSynchronize())


def _read(buf, offset, size):
    """Copy ``size`` bytes of ``buf`` at ``offset`` to the host."""
    host = (ctypes.c_ubyte * size)()
    handle_return(driver.cuMemcpyDtoH(ctypes.addressof(host), int(buf.handle) + offset, size))
    return bytes(host)


def _is_mapped(ptr):
    """Whether the driver still has a mapping at ``ptr``."""
    status, handle = driver.cuMemRetainAllocationHandle(ptr)
    if status == driver.CUresult.CUDA_SUCCESS:
        handle_return(driver.cuMemRelease(handle))
        return True
    return False


class _SharedStub(MemoryResource):
    """Reports IPC support without being a pool, to reach the base-class export path."""

    def __init__(self, device):
        self._device_id = device.device_id

    def allocate(self, size, *, stream=None):
        raise NotImplementedError

    def deallocate(self, ptr, size, *, stream=None):
        pass

    @property
    def is_device_accessible(self):
        return True

    @property
    def is_host_accessible(self):
        return False

    @property
    def device_id(self):
        return self._device_id

    @property
    def is_ipc_enabled(self):
        return True


def _run_child(target, *args):
    """Run ``target(*args)`` in a spawned child and fail if it does not exit cleanly."""
    process = mp.Process(target=target, args=args)
    process.start()
    process.join(timeout=CHILD_TIMEOUT_SEC)
    survivors = kill_subprocesses(process)
    assert not survivors, "child did not exit within timeout"
    assert process.exitcode == 0, f"child exited with {process.exitcode}"


class TestVmmIpcDescriptor:
    @pytest.mark.agent_authored(model="claude-fable-5-1")
    def test_descriptor_round_trip(self, vmm_ipc_device):
        """A child imports the descriptor, reads the parent's data, writes its own, and releases."""
        device = vmm_ipc_device
        mr = _resource(device)
        assert mr.is_ipc_enabled
        buf = mr.allocate(1)
        _fill(buf, 0xAB)
        desc = buf.ipc_descriptor
        assert isinstance(desc, IPCBufferDescriptor)
        assert desc.size == buf.size
        assert buf.ipc_descriptor is desc  # cached
        assert not buf.is_mapped

        queue = mp.Queue()
        _run_child(self.child_main, device.device_id, desc, queue)
        reads = queue.get(timeout=CHILD_TIMEOUT_SEC)
        assert reads["is_mapped"] is True
        assert reads["size"] == buf.size
        assert reads["first_bytes"] == b"\xab" * 16
        assert reads["released"] is True
        assert _read(buf, 0, 16) == b"\x5a" * 16
        buf.close()

    @staticmethod
    def child_main(device_id, desc, queue):
        device = Device(device_id)
        device.set_current()
        mr = _resource(device)
        imported = Buffer.from_ipc_descriptor(mr, desc, stream=device.default_stream)
        assert isinstance(imported, VirtualMemoryBuffer)
        result = {
            "is_mapped": imported.is_mapped,
            "size": imported.size,
            "first_bytes": _read(imported, 0, 16),
        }
        _fill(imported, 0x5A)
        ptr = int(imported.handle)
        imported.close()
        result["released"] = not _is_mapped(ptr)
        queue.put(result)

    @pytest.mark.agent_authored(model="claude-fable-5-1")
    def test_buffer_pickles_through_queue(self, vmm_ipc_device):
        """A buffer sent directly pickles as (resource, descriptor) and imports on arrival."""
        device = vmm_ipc_device
        mr = _resource(device)
        buf = mr.allocate(1)
        _fill(buf, 0x11)
        queue = mp.Queue()
        queue.put(buf)
        _run_child(self.child_buffer, device.device_id, queue)
        assert _read(buf, 0, 16) == b"\x22" * 16
        buf.close()

    @staticmethod
    def child_buffer(device_id, queue):
        Device(device_id).set_current()
        imported = queue.get(timeout=CHILD_TIMEOUT_SEC)
        assert isinstance(imported, VirtualMemoryBuffer)
        assert imported.is_mapped
        assert isinstance(imported.memory_resource, VirtualMemoryResource)
        assert imported.memory_resource.device_id == device_id
        assert _read(imported, 0, 16) == b"\x11" * 16
        _fill(imported, 0x22)
        imported.close()

    @pytest.mark.agent_authored(model="claude-fable-5-1")
    def test_grown_buffer_imports_as_one_range(self, vmm_ipc_device):
        """A buffer with two physical allocations exports both and imports contiguously."""
        device = vmm_ipc_device
        mr = _resource(device)
        first = mr.allocate(1)
        gran = first.size
        grown = mr.modify_allocation(first, 2 * gran)
        assert grown.size == 2 * gran
        _fill(grown, 0x11, offset=0, size=gran)
        _fill(grown, 0x22, offset=gran, size=gran)
        desc = grown.ipc_descriptor
        assert desc.size == 2 * gran
        assert len(desc._chunk_sizes) == 2

        queue = mp.Queue()
        _run_child(self.child_grown, device.device_id, desc, gran, queue)
        assert queue.get(timeout=CHILD_TIMEOUT_SEC) == {"low": b"\x11" * 16, "high": b"\x22" * 16}
        first.close()
        grown.close()

    @staticmethod
    def child_grown(device_id, desc, gran, queue):
        device = Device(device_id)
        device.set_current()
        imported = Buffer.from_ipc_descriptor(_resource(device), desc, stream=device.default_stream)
        assert imported.size == 2 * gran
        queue.put({"low": _read(imported, 0, 16), "high": _read(imported, gran, 16)})
        imported.close()

    @pytest.mark.agent_authored(model="claude-fable-5-1")
    def test_import_in_same_process_aliases_memory(self, vmm_ipc_device):
        """Importing a descriptor where it was exported maps the same memory at a new address."""
        device = vmm_ipc_device
        mr = _resource(device)
        buf = mr.allocate(1)
        alias = Buffer.from_ipc_descriptor(mr, buf.ipc_descriptor, stream=device.default_stream)
        assert alias.is_mapped
        assert int(alias.handle) != int(buf.handle)
        _fill(buf, 0x33)
        assert _read(alias, 0, 16) == b"\x33" * 16
        _fill(alias, 0x44)
        assert _read(buf, 0, 16) == b"\x44" * 16
        ptr_buf, ptr_alias = int(buf.handle), int(alias.handle)
        buf.close()
        assert _is_mapped(ptr_alias)  # the import keeps the memory alive
        alias.close()
        assert not _is_mapped(ptr_alias)
        assert not _is_mapped(ptr_buf)

    @pytest.mark.agent_authored(model="claude-fable-5-1")
    def test_descriptor_owns_its_file_descriptors(self, vmm_ipc_device):
        """The exported file descriptors close when the descriptor is released."""
        device = vmm_ipc_device
        mr = _resource(device)
        buf = mr.allocate(1)
        desc = buf.ipc_descriptor
        fd = int(desc._handles[0])
        os.fstat(fd)  # open
        with pytest.raises(TypeError):
            pickle.dumps(desc)  # file descriptors need multiprocessing
        buf.close()  # drops the cached descriptor
        del desc
        gc.collect()
        with pytest.raises(OSError):
            os.fstat(fd)

    @pytest.mark.agent_authored(model="claude-fable-5-1")
    def test_errors(self, vmm_ipc_device):
        device = vmm_ipc_device
        stream = device.default_stream
        shared = _resource(device)
        private = _resource(device, handle_type=None)
        assert not private.is_ipc_enabled
        with private.allocate(1) as buf, pytest.raises(RuntimeError, match="not IPC-enabled"):
            _ = buf.ipc_descriptor
        with shared.allocate(1) as buf:
            desc = buf.ipc_descriptor
            with pytest.raises(RuntimeError, match="not IPC-enabled"):
                Buffer.from_ipc_descriptor(private, desc, stream=stream)
            with pytest.raises(TypeError, match="VirtualMemoryResource"):
                Buffer.from_ipc_descriptor(device.memory_resource, desc, stream=stream)
            pool_desc = IPCBufferDescriptor._init(b"\x00" * 64, 64)
            with pytest.raises(TypeError, match="memory pool"):
                Buffer.from_ipc_descriptor(shared, pool_desc, stream=stream)
        # The base-class export path serves pool buffers only: a Buffer that a
        # non-pool resource did not allocate is refused before any driver call.
        with device.memory_resource.allocate(64, stream=stream) as backing:
            wrapped = Buffer.from_handle(int(backing.handle), 64, mr=_SharedStub(device))
            with pytest.raises(TypeError, match="did not allocate"):
                _ = wrapped.ipc_descriptor
            wrapped.close()
