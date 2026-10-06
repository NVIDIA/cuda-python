# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sharing VirtualMemoryResource buffers between processes (issue #2980).

The scenarios that pool-backed and VMM buffers have in common run in the rest
of this directory through the ``VirtualMR`` parameter of the
``ipc_memory_resource`` fixture. This module covers what is specific to
virtual memory: ranges made of several physical allocations, which device the
memory belongs to, the validation of a descriptor from an untrusted peer, and
the limits of re-export.
"""

import ctypes
import gc
import json
import multiprocessing as mp
import multiprocessing.queues
import os
import pickle
import resource
import socket
import threading

import pytest
from helpers.child_processes import child_timeout_sec, kill_subprocesses
from helpers.contexts import assert_no_cuda_warning

from cuda.bindings import driver
from cuda.core import Buffer, Device, MemoryResource, VirtualMemoryResource, VirtualMemoryResourceOptions
from cuda.core._dlpack import DLDeviceType
from cuda.core._memory._ipc import IPCAllocationHandle, IPCBufferDescriptor, VirtualMemoryIPCBufferDescriptor
from cuda.core._memory._virtual_memory_resource import VirtualMemoryBuffer
from cuda.core._utils.cuda_utils import CUDAError, handle_return
from cuda.core.utils import StridedMemoryView

CHILD_TIMEOUT_SEC = child_timeout_sec()

# These tests spawn processes and pass file descriptors, which fails for very many threads.
pytestmark = pytest.mark.parallel_threads_limit(4)


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


def _access_flags(buf, device_id):
    """The access a device has to ``buf``: 0 for none, 3 for read-write."""
    location = driver.CUmemLocation()
    location.type = driver.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
    location.id = device_id
    return int(handle_return(driver.cuMemGetAccess(location, int(buf.handle))))


def _reserve_at(addr, size):
    """Reserve ``size`` bytes at ``addr``. Return the reservation, or None if the driver placed it elsewhere."""
    ptr = handle_return(driver.cuMemAddressReserve(size, 0, addr, 0))
    if int(ptr) != addr:
        handle_return(driver.cuMemAddressFree(ptr, size))
        return None
    return ptr


def _open_fds():
    return len(os.listdir("/proc/self/fd"))


def _has_note(exc, text):
    """Whether ``text`` is in one of the exception's notes, or in its message on Python 3.10."""
    return any(text in note for note in getattr(exc, "__notes__", ())) or text in str(exc)


def _forge(desc, **fields):
    """A copy of ``desc`` with some fields replaced, as a hostile or buggy peer could send."""
    fields.setdefault("handle_type", desc._handle_type)
    fields.setdefault("chunk_sizes", desc._chunk_sizes)
    fields.setdefault("handles", desc._handles)
    fields.setdefault("size", desc.size)
    return VirtualMemoryIPCBufferDescriptor._from_exports(
        fields["handle_type"], fields["chunk_sizes"], fields["handles"], fields["size"]
    )


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


class _CustomResource(VirtualMemoryResource):
    """A subclass, to check that pickling preserves the type."""


def _run_child(target, *args):
    """Run ``target(*args)`` in a spawned child and fail if it does not exit cleanly."""
    process = mp.Process(target=target, args=args)
    process.start()
    process.join(timeout=CHILD_TIMEOUT_SEC)
    survivors = kill_subprocesses(process)
    assert not survivors, "child did not exit within timeout"
    assert process.exitcode == 0, f"child exited with {process.exitcode}"


def _noop(*args):
    pass


def _close_argument(buffer):
    buffer.close()


_REEXPORT_MESSAGE = "imported from another process"
_FEEDER_ERRORS: list = []
_FEEDER_EVENT = threading.Event()


class _ReportingQueue(multiprocessing.queues.Queue):
    """A Queue whose feeder-thread errors are recorded, not only printed to stderr."""

    def __init__(self):
        super().__init__(ctx=mp.get_context())

    @staticmethod
    def _on_queue_feeder_error(e, _obj):
        _FEEDER_ERRORS.append((type(e), str(e)))
        _FEEDER_EVENT.set()


@pytest.mark.agent_authored(model="claude-fable-5-1")
@pytest.mark.parametrize(
    ("options", "expected"),
    [
        pytest.param({"handle_type": "posix_fd"}, True, id="posix_fd"),
        pytest.param({"handle_type": None}, False, id="none"),
        pytest.param({"handle_type": "fabric"}, False, id="fabric"),
        pytest.param({"handle_type": "win32_kmt"}, False, id="win32_kmt"),
        pytest.param({"location_type": "host", "handle_type": None}, False, id="host"),
        pytest.param({"location_type": "host_numa", "handle_type": "posix_fd"}, True, id="host_numa-posix_fd"),
    ],
)
def test_is_ipc_enabled(vmm_ipc_device, options, expected):
    """Only POSIX file descriptors have a transport; the location does not matter."""
    mr = VirtualMemoryResource(vmm_ipc_device, config=VirtualMemoryResourceOptions(**options))
    assert mr.is_ipc_enabled is expected


@pytest.mark.agent_authored(model="claude-fable-5-1")
def test_resource_pickles_as_its_own_type(vmm_ipc_device):
    """A VirtualMemoryResource pickles as (device, options) and a subclass comes back as the subclass."""
    mr = _CustomResource(vmm_ipc_device, config=VirtualMemoryResourceOptions(handle_type="posix_fd", peers=()))
    copy = pickle.loads(pickle.dumps(mr))  # noqa: S301  our own bytes; the resource carries no handle
    assert type(copy) is _CustomResource
    assert copy.device_id == mr.device_id
    assert copy.config == mr.config


class TestManyChunkBuffer:
    @pytest.mark.agent_authored(model="claude-fable-5-1")
    @pytest.mark.thread_unsafe(reason="counts process-global file descriptors and records process-global warnings")
    def test_main(self, vmm_ipc_device):
        """A range of three physical allocations, the last placed by a relocating grow, exports and imports whole.

        Each allocation costs one file descriptor while the descriptor lives;
        the child reads every chunk, writes one back, and tears the import
        down without a warning.
        """
        device = vmm_ipc_device
        mr = _resource(device)
        queue = mp.Queue()  # before the descriptor count: the queue's pipe is not part of it
        buffers = []
        try:
            buf = mr.allocate(1)
            buffers.append(buf)
            gran = buf.size
            buf = mr.modify_allocation(buf, 2 * gran)  # extends in place or moves
            buffers.append(buf)
            # Take the range after the buffer so that this grow must remap the
            # existing chunks at a new address.
            decoy = _reserve_at(int(buf.handle) + buf.size, gran)
            if decoy is None:
                pytest.skip("the driver did not grant a reservation right after the buffer")
            try:
                moved = mr.modify_allocation(buf, 3 * gran)
            finally:
                handle_return(driver.cuMemAddressFree(decoy, gran))
            buffers.append(moved)
            assert int(moved.handle) != int(buf.handle)
            assert moved.size == 3 * gran
            values = (0x11, 0x22, 0x33)
            for i, value in enumerate(values):
                _fill(moved, value, offset=i * gran, size=gran)

            fds_before = _open_fds()
            desc = moved.ipc_descriptor
            assert desc.size == 3 * gran
            assert desc._chunk_sizes == (gran, gran, gran)
            assert _open_fds() == fds_before + 3
            fds = [int(handle) for handle in desc._handles]

            _run_child(self.child_main, device.device_id, desc, gran, queue)
            result = queue.get(timeout=CHILD_TIMEOUT_SEC)
            assert result["chunks"] == [bytes([value]) * 16 for value in values]
            assert result["released"] is True
            assert _read(moved, 2 * gran, 16) == b"\x44" * 16
        finally:
            for buffer in buffers:
                buffer.close()
        # The descriptor is the only holder of the file descriptors; the buffer caches nothing.
        del desc
        gc.collect()
        for fd in fds:
            with pytest.raises(OSError):
                os.fstat(fd)

    @staticmethod
    def child_main(device_id, desc, gran, queue):
        device = Device(device_id)
        device.set_current()
        mr = _resource(device)
        with assert_no_cuda_warning():
            imported = Buffer.from_ipc_descriptor(mr, desc, stream=device.default_stream)
            assert isinstance(imported, VirtualMemoryBuffer)
            assert imported.is_mapped
            assert imported.size == 3 * gran
            chunks = [_read(imported, i * gran, 16) for i in range(3)]
            _fill(imported, 0x44, offset=2 * gran, size=gran)
            ptr = int(imported.handle)
            imported.close()
            released = not _is_mapped(ptr)
        queue.put({"chunks": chunks, "released": released})


class TestImportOnSecondDevice:
    """The memory belongs to the exporter's device; the importer maps it for that device and may add peers."""

    @pytest.mark.agent_authored(model="claude-fable-5-1")
    def test_main(self, vmm_ipc_device_x2):
        dev0, dev1 = vmm_ipc_device_x2
        dev1.set_current()
        mr = _resource(dev1)
        buf = mr.allocate(1)
        _fill(buf, 0x5A)
        desc = buf.ipc_descriptor

        # In this process: a resource for the other device refuses the memory before mapping anything.
        with pytest.raises(ValueError, match=f"memory on device {dev1.device_id}, but this resource maps memory on"):
            Buffer.from_ipc_descriptor(_resource(dev0), desc, stream=dev0.default_stream)

        peers_possible = dev1.can_access_peer(dev0)
        queue = mp.Queue()
        _run_child(self.child_main, dev0.device_id, dev1.device_id, desc, peers_possible, queue)
        result = queue.get(timeout=CHILD_TIMEOUT_SEC)
        assert f"device {dev0.device_id}" in result["wrong_device"]
        assert result["device_id"] == dev1.device_id
        assert result["access_without_peers"] == (3, 0)
        assert result["data"] == b"\x5a" * 16
        if peers_possible:
            assert result["access_with_peers"] == (3, 3)
            assert result["peer_read"] == b"\x5a" * 16
        else:
            assert "access_with_peers" not in result

        buf.close()
        dev1.sync()

    @staticmethod
    def child_main(id0, id1, desc, peers_possible, queue):
        dev0, dev1 = Device(id0), Device(id1)
        dev1.set_current()
        result = {}

        # A resource for the wrong device: refused with the device ids named.
        with pytest.raises(ValueError) as info:
            Buffer.from_ipc_descriptor(_resource(dev0), desc, stream=dev0.default_stream)
        result["wrong_device"] = str(info.value)

        # The owning device, without peers: only that device can reach the memory.
        imported = Buffer.from_ipc_descriptor(_resource(dev1), desc, stream=dev1.default_stream)
        result["device_id"] = imported.device_id
        result["access_without_peers"] = (_access_flags(imported, id1), _access_flags(imported, id0))
        result["data"] = _read(imported, 0, 16)
        imported.close()

        # With the other device as a peer, it can read the import directly.
        if peers_possible:
            imported = Buffer.from_ipc_descriptor(_resource(dev1, peers=[id0]), desc, stream=dev1.default_stream)
            result["access_with_peers"] = (_access_flags(imported, id1), _access_flags(imported, id0))
            dev0.set_current()
            result["peer_read"] = _read(imported, 0, 16)
            dev1.set_current()
            imported.close()
        queue.put(result)


class TestDescriptorValidation:
    """A descriptor is data from another process; each malformed field fails before memory is mapped."""

    @pytest.mark.agent_authored(model="claude-fable-5-1")
    def test_rejected_before_import(self, vmm_ipc_device):
        device = vmm_ipc_device
        stream = device.default_stream
        mr = _resource(device)
        with mr.allocate(1) as buf:
            gran = buf.size
            desc = buf.ipc_descriptor
            fabric = int(driver.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC)
            cases = [
                (_forge(desc, handle_type=fabric), ValueError, "handle type does not match"),
                (_forge(desc, chunk_sizes=(gran, gran)), ValueError, "malformed"),
                (_forge(desc, chunk_sizes=(0,), size=0), ValueError, "positive multiple"),
                (_forge(desc, chunk_sizes=(gran + 1,)), ValueError, "positive multiple"),
                (_forge(desc, size=2 * gran), ValueError, "exceeds the exported allocations"),
                (_forge(desc, chunk_sizes=(), handles=(), size=gran), ValueError, "no exported allocations"),
            ]
            for forged, exc_type, text in cases:
                with pytest.raises(exc_type, match=text):
                    Buffer.from_ipc_descriptor(mr, forged, stream=stream)

            # A handle that was closed on this side.
            closed = IPCAllocationHandle._init(os.dup(int(desc._handles[0])), None)
            closed.close()
            with pytest.raises(ValueError, match="closed"):
                Buffer.from_ipc_descriptor(mr, _forge(desc, handles=(closed,)), stream=stream)

            # A file descriptor that is not a CUDA allocation: the driver refuses it, and the error says where.
            read_end, write_end = os.pipe()
            os.close(write_end)
            not_cuda = IPCAllocationHandle._init(read_end, None)  # owns and closes read_end
            with pytest.raises(CUDAError) as info:
                Buffer.from_ipc_descriptor(mr, _forge(desc, handles=(not_cuda,)), stream=stream)
            assert _has_note(info.value, "while importing chunk 1 of 1")
            assert _has_note(info.value, "handle_type='posix_fd'")

            # The buffer is untouched and the genuine descriptor still imports.
            _fill(buf, 0x66)
            alias = Buffer.from_ipc_descriptor(mr, desc, stream=stream)
            assert _read(alias, 0, 16) == b"\x66" * 16
            alias.close()

    @pytest.mark.agent_authored(model="claude-fable-5-1")
    @pytest.mark.thread_unsafe(reason="checks mapping state by address; another thread can reuse a freed address")
    def test_wrong_chunk_size_fails_at_map_and_rolls_back(self, vmm_ipc_device):
        """A chunk size that does not match the allocation fails in the driver's map; earlier chunks are undone.

        The importer is given an address hint so that the rollback is
        observable: after the failure nothing is mapped at that address.
        """
        device = vmm_ipc_device
        stream = device.default_stream
        mr = _resource(device)
        first = mr.allocate(1)
        gran = first.size
        buf = mr.modify_allocation(first, 2 * gran)
        desc = buf.ipc_descriptor
        assert desc._chunk_sizes == (gran, gran)

        # An address the importer will use for a three-chunk range.
        probe = int(handle_return(driver.cuMemAddressReserve(3 * gran, 0, 0, 0)))
        handle_return(driver.cuMemAddressFree(probe, 3 * gran))
        importer = _resource(device, addr_hint=probe)
        alias = Buffer.from_ipc_descriptor(importer, desc, stream=stream)
        honored = int(alias.handle) == probe
        alias.close()
        if not honored:
            first.close()
            buf.close()
            pytest.skip("the driver did not honor the address hint")

        forged = _forge(desc, chunk_sizes=(gran, 2 * gran), size=3 * gran)
        with pytest.raises(CUDAError) as info:
            Buffer.from_ipc_descriptor(importer, forged, stream=stream)
        assert _has_note(info.value, f"while mapping chunk 2 of 2 ({2 * gran} bytes)")
        assert not _is_mapped(probe)  # chunk 1 was mapped there and has been undone
        reservation = _reserve_at(probe, 3 * gran)
        assert reservation is not None, "the importer's address reservation was not released"
        handle_return(driver.cuMemAddressFree(reservation, 3 * gran))

        # The exporter's memory is intact and the genuine descriptor still imports.
        _fill(buf, 0x77, offset=gran, size=gran)
        alias = Buffer.from_ipc_descriptor(importer, desc, stream=stream)
        assert _read(alias, gran, 16) == b"\x77" * 16
        alias.close()
        first.close()
        buf.close()


class TestReexport:
    """Memory imported from another process cannot be exported again; the error is clear and early."""

    @staticmethod
    def _derived(mr, imported, gran):
        """Buffers modify_allocation derives from an import: an alias, a grow, and (if the driver allows) a move."""
        derived = {
            "alias": mr.modify_allocation(imported, imported.size),
            "grow": mr.modify_allocation(imported, 2 * gran),
        }
        base = derived["grow"]
        decoy = _reserve_at(int(base.handle) + base.size, gran)
        if decoy is not None:
            try:
                derived["move"] = mr.modify_allocation(base, base.size + gran)
            finally:
                handle_return(driver.cuMemAddressFree(decoy, gran))
            assert int(derived["move"].handle) != int(base.handle)
        return derived

    @pytest.mark.agent_authored(model="claude-fable-5-1")
    def test_export_and_pickling_raise_before_any_driver_call(self, vmm_ipc_device):
        device = vmm_ipc_device
        stream = device.default_stream
        mr = _resource(device)
        buf = mr.allocate(1)
        gran = buf.size
        desc = buf.ipc_descriptor
        imported = Buffer.from_ipc_descriptor(mr, desc, stream=stream)
        assert imported.is_mapped
        buffers = {"import": imported, **self._derived(mr, imported, gran)}
        try:
            for name, buffer in buffers.items():
                with pytest.raises(RuntimeError, match=_REEXPORT_MESSAGE) as info:
                    _ = buffer.ipc_descriptor
                message = str(info.value)
                assert "cannot be exported again" in message, name
                assert "Forward the descriptor" in message and "copy the data" in message, name
                with pytest.raises(RuntimeError, match=_REEXPORT_MESSAGE):
                    mp.reduction.ForkingPickler.dumps(buffer)
            # The transports that pickle in the caller's thread raise to the caller.
            parent_conn, child_conn = mp.Pipe()
            with parent_conn, child_conn, pytest.raises(RuntimeError, match=_REEXPORT_MESSAGE):
                parent_conn.send(imported)
            with pytest.raises(RuntimeError, match=_REEXPORT_MESSAGE):
                mp.Process(target=_noop, args=(imported,)).start()
            # The original descriptor still serves new importers, and the data is intact.
            _fill(buf, 0x3C)
            alias = Buffer.from_ipc_descriptor(mr, desc, stream=stream)
            assert _read(alias, 0, 16) == b"\x3c" * 16
            alias.close()
        finally:
            for buffer in reversed(list(buffers.values())):
                buffer.close()
            buf.close()

    @pytest.mark.agent_authored(model="claude-fable-5-1")
    @pytest.mark.thread_unsafe(reason="records feeder-thread errors in module state")
    def test_queue_feeder_reports_the_error_and_stays_usable(self, vmm_ipc_device):
        """A Queue pickles in its feeder thread: the error is reported there, the item is dropped, and the queue works on."""
        device = vmm_ipc_device
        stream = device.default_stream
        mr = _resource(device)
        buf = mr.allocate(1)
        imported = Buffer.from_ipc_descriptor(mr, buf.ipc_descriptor, stream=stream)
        _FEEDER_ERRORS.clear()
        _FEEDER_EVENT.clear()
        queue = _ReportingQueue()
        try:
            queue.put(imported)
            assert _FEEDER_EVENT.wait(timeout=CHILD_TIMEOUT_SEC), "the feeder thread did not report the error"
            ((exc_type, message),) = _FEEDER_ERRORS
            assert exc_type is RuntimeError
            assert _REEXPORT_MESSAGE in message
            assert "cannot be exported again" in message
            # A consumer is not left waiting: the next item arrives.
            queue.put("next item")
            assert queue.get(timeout=CHILD_TIMEOUT_SEC) == "next item"
        finally:
            queue.close()
            queue.join_thread()
            imported.close()
            buf.close()

    @pytest.mark.agent_authored(model="claude-fable-5-1")
    def test_grow_cannot_change_handle_type(self, vmm_ipc_device):
        """Every chunk of a buffer must be exportable the same way."""
        device = vmm_ipc_device
        mr = _resource(device)
        with (
            mr.allocate(1) as buf,
            pytest.raises(ValueError, match="handle_type"),
        ):
            mr.modify_allocation(buf, 2 * buf.size, config=VirtualMemoryResourceOptions(handle_type=None))
        private = _resource(device, handle_type=None)
        with (
            private.allocate(1) as buf,
            pytest.raises(ValueError, match="handle_type"),
        ):
            private.modify_allocation(buf, 2 * buf.size, config=VirtualMemoryResourceOptions())


class TestFileDescriptorLifetime:
    """A descriptor is the only holder of file descriptors, on both sides."""

    @pytest.mark.agent_authored(model="claude-fable-5-1")
    @pytest.mark.thread_unsafe(reason="counts process-global file descriptors")
    def test_exporter_holds_none_without_a_descriptor(self, vmm_ipc_device):
        """Each access exports anew; the file descriptors close with the descriptor, not with the buffer."""
        device = vmm_ipc_device
        mr = _resource(device)
        first = mr.allocate(1)
        gran = first.size
        buf = mr.modify_allocation(first, 2 * gran)
        fds0 = _open_fds()
        desc = buf.ipc_descriptor
        assert desc.sizes == (gran, gran)
        assert _open_fds() == fds0 + 2
        again = buf.ipc_descriptor
        assert again is not desc
        assert set(again.fds).isdisjoint(desc.fds)
        assert _open_fds() == fds0 + 4
        del again
        gc.collect()
        assert _open_fds() == fds0 + 2
        numbers = desc.fds
        del desc
        gc.collect()
        assert _open_fds() == fds0, "the buffer alone must hold no file descriptors"
        for fd in numbers:
            with pytest.raises(OSError):
                os.fstat(fd)
        buf.close()
        first.close()

    @pytest.mark.agent_authored(model="claude-fable-5-1")
    @pytest.mark.thread_unsafe(reason="counts process-global file descriptors")
    def test_process_argument_releases_with_the_process_object(self, vmm_ipc_device):
        """A buffer passed to a spawned Process imports there; its descriptor goes when the Process object does."""
        device = vmm_ipc_device
        mr = _resource(device)
        buf = mr.allocate(1)
        fds0 = _open_fds()
        process = mp.Process(target=_close_argument, args=(buf,))
        process.start()
        process.join(timeout=CHILD_TIMEOUT_SEC)
        survivors = kill_subprocesses(process)
        assert not survivors, "child did not exit within timeout"
        assert process.exitcode == 0, f"child exited with {process.exitcode}"
        del process
        gc.collect()
        assert _open_fds() == fds0
        buf.close()

    @pytest.mark.agent_authored(model="claude-fable-5-1")
    def test_importer_holds_none_after_the_import(self, vmm_ipc_device):
        """The import adds no file descriptors, and releasing the received descriptor closes its own."""
        device = vmm_ipc_device
        mr = _resource(device)
        first = mr.allocate(1)
        gran = first.size
        buf = mr.modify_allocation(first, 2 * gran)
        _fill(buf, 0x5B)
        to_child, from_child = mp.Queue(), mp.Queue()
        process = mp.Process(target=self.child_main, args=(device.device_id, to_child, from_child))
        process.start()
        to_child.put(buf.ipc_descriptor)  # transient: nothing is kept on this side
        counts = from_child.get(timeout=CHILD_TIMEOUT_SEC)
        process.join(timeout=CHILD_TIMEOUT_SEC)
        survivors = kill_subprocesses(process)
        assert not survivors, "child did not exit within timeout"
        assert process.exitcode == 0, f"child exited with {process.exitcode}"
        assert counts["data"] == b"\x5b" * 16
        assert counts["after_import"] == counts["with_descriptor"]
        assert counts["after_release"] == counts["with_descriptor"] - 2
        assert counts["after_close"] == counts["after_release"]
        buf.close()
        first.close()

    @staticmethod
    def child_main(device_id, to_child, from_child):
        device = Device(device_id)
        device.set_current()
        mr = _resource(device)
        desc = to_child.get(timeout=CHILD_TIMEOUT_SEC)
        # A first import lets the driver finish any lazy initialization before counting.
        Buffer.from_ipc_descriptor(mr, desc, stream=device.default_stream).close()
        counts = {"with_descriptor": _open_fds()}
        imported = Buffer.from_ipc_descriptor(mr, desc, stream=device.default_stream)
        counts["after_import"] = _open_fds()
        del desc
        gc.collect()
        counts["after_release"] = _open_fds()
        counts["data"] = _read(imported, 0, 16)  # the import, not the descriptor, holds the memory now
        imported.close()
        counts["after_close"] = _open_fds()
        from_child.put(counts)


class TestFileDescriptorLimit:
    @pytest.mark.agent_authored(model="claude-fable-5-1")
    @pytest.mark.thread_unsafe(reason="changes the process-wide file descriptor limit")
    def test_export_near_the_limit(self, vmm_ipc_device):
        """Under a lowered RLIMIT_NOFILE an export either succeeds or fails with a readable error and no leak."""
        device = vmm_ipc_device
        stream = device.default_stream
        mr = _resource(device)
        first = mr.allocate(1)
        gran = first.size
        second = mr.modify_allocation(first, 2 * gran)
        buf = mr.modify_allocation(second, 3 * gran)
        values = (0x61, 0x62, 0x63)
        for i, value in enumerate(values):
            _fill(buf, value, offset=i * gran, size=gran)
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        fillers = []
        imported = desc = None
        try:
            # Fill the holes in the descriptor table so that the limit alone decides how many more fit.
            highest = max(int(name) for name in os.listdir("/proc/self/fd"))
            while True:
                fd = os.open(os.devnull, os.O_RDONLY)
                if fd > highest:
                    os.close(fd)
                    break
                fillers.append(fd)
            baseline = _open_fds()
            # Room for two more descriptors: a three-allocation export cannot complete.
            resource.setrlimit(resource.RLIMIT_NOFILE, (highest + 1 + 2, hard))
            failure = None
            try:
                desc = buf.ipc_descriptor
            except CUDAError as exc:
                failure = exc
            if failure is not None:
                assert _has_note(failure, "file descriptor limit")
                assert _open_fds() == baseline, "a failed export must close the descriptors it already took"
            else:
                # Acceptable only if the driver did not need one descriptor per allocation.
                assert len(desc.fds) == 3
                desc = None
                gc.collect()
            # Room for the three descriptors and the listing: export and import both succeed.
            resource.setrlimit(resource.RLIMIT_NOFILE, (highest + 1 + 6, hard))
            desc = buf.ipc_descriptor
            assert len(desc.fds) == 3
            imported = Buffer.from_ipc_descriptor(mr, desc, stream=stream)
            assert _open_fds() == baseline + 3, "the import must not take file descriptors of its own"
        finally:
            resource.setrlimit(resource.RLIMIT_NOFILE, (soft, hard))
            for fd in fillers:
                os.close(fd)
        assert [_read(imported, i * gran, 16) for i in range(3)] == [bytes([value]) * 16 for value in values]
        del desc
        gc.collect()
        for buffer in (imported, buf, second, first):
            buffer.close()


class TestRawFileDescriptorTransport:
    """The descriptor's fds, sizes, handle_type, and size travel over any file-descriptor-passing transport."""

    @pytest.mark.agent_authored(model="claude-fable-5-1")
    @pytest.mark.thread_unsafe(reason="counts process-global file descriptors")
    def test_round_trip_with_os_dup(self, vmm_ipc_device):
        device = vmm_ipc_device
        stream = device.default_stream
        mr = _resource(device)
        first = mr.allocate(1)
        gran = first.size
        buf = mr.modify_allocation(first, 2 * gran)
        _fill(buf, 0x71, size=gran)
        _fill(buf, 0x72, offset=gran, size=gran)
        fds0 = _open_fds()
        desc = buf.ipc_descriptor
        assert desc.handle_type == "posix_fd"
        assert desc.sizes == (gran, gran)
        assert desc.size == 2 * gran
        assert desc.fds == tuple(int(handle) for handle in desc.handles)
        # A transport of our own: duplicate the integers, then let the descriptor go.
        wire = {
            "fds": [os.dup(fd) for fd in desc.fds],
            "sizes": list(desc.sizes),
            "handle_type": str(desc.handle_type),
            "size": desc.size,
        }
        del desc
        gc.collect()
        assert _open_fds() == fds0 + 2
        rebuilt = VirtualMemoryIPCBufferDescriptor.from_fds(
            wire["fds"], wire["sizes"], handle_type=wire["handle_type"], size=wire["size"]
        )
        assert _open_fds() == fds0 + 4, "from_fds duplicates the integers it is given"
        for fd in wire["fds"]:
            os.close(fd)  # the caller keeps ownership of its own
        assert _open_fds() == fds0 + 2
        assert rebuilt.handle_type == "posix_fd"
        assert rebuilt.sizes == (gran, gran)
        assert rebuilt.size == 2 * gran
        imported = Buffer.from_ipc_descriptor(mr, rebuilt, stream=stream)
        assert _read(imported, 0, 16) == b"\x71" * 16
        assert _read(imported, gran, 16) == b"\x72" * 16
        del rebuilt
        gc.collect()
        assert _open_fds() == fds0
        imported.close()
        buf.close()
        first.close()

    @pytest.mark.agent_authored(model="claude-fable-5-1")
    def test_from_fds_shares_handle_objects_and_rejects_bad_input(self, vmm_ipc_device):
        device = vmm_ipc_device
        stream = device.default_stream
        mr = _resource(device)
        buf = mr.allocate(1)
        gran = buf.size
        desc = buf.ipc_descriptor
        from_fds = VirtualMemoryIPCBufferDescriptor.from_fds

        shared = from_fds(desc.handles, desc.sizes)
        assert shared.handles[0] is desc.handles[0]  # an IPCAllocationHandle is shared, not duplicated
        assert shared.size == gran
        Buffer.from_ipc_descriptor(mr, shared, stream=stream).close()

        with pytest.raises(ValueError, match="one size per handle"):
            from_fds(desc.fds, ())
        with pytest.raises(ValueError, match="positive"):
            from_fds(desc.fds, (0,))
        with pytest.raises(ValueError, match="sum of the allocation sizes"):
            from_fds(desc.fds, desc.sizes, size=gran + 1)
        with pytest.raises(ValueError):
            from_fds(desc.fds, desc.sizes, handle_type="win32")
        with pytest.raises(ValueError, match="handle_type=None"):
            from_fds(desc.fds, desc.sizes, handle_type=None)
        with pytest.raises(OSError):
            from_fds([1 << 20], desc.sizes)
        closed = IPCAllocationHandle._init(os.dup(desc.fds[0]), None)
        closed.close()
        with pytest.raises(RuntimeError, match="closed"):
            from_fds([closed], desc.sizes)
        # A handle type other than the resource's is refused at import, before any driver call.
        other = from_fds(desc.fds, desc.sizes, handle_type="fabric")
        with pytest.raises(ValueError, match="handle type does not match"):
            Buffer.from_ipc_descriptor(mr, other, stream=stream)
        buf.close()

    @pytest.mark.agent_authored(model="claude-fable-5-1")
    def test_round_trip_through_a_unix_socket(self, vmm_ipc_device):
        """A child receives the file descriptors with SCM_RIGHTS and rebuilds the descriptor itself."""
        device = vmm_ipc_device
        mr = _resource(device)
        first = mr.allocate(1)
        gran = first.size
        buf = mr.modify_allocation(first, 2 * gran)
        _fill(buf, 0x73, size=gran)
        _fill(buf, 0x74, offset=gran, size=gran)
        parent_sock, child_sock = socket.socketpair()
        queue = mp.Queue()
        process = mp.Process(target=self.child_main, args=(device.device_id, child_sock, queue))
        process.start()
        child_sock.close()
        desc = buf.ipc_descriptor
        header = json.dumps({"sizes": list(desc.sizes), "handle_type": str(desc.handle_type), "size": desc.size})
        socket.send_fds(parent_sock, [header.encode()], list(desc.fds))
        del desc  # the kernel duplicated the descriptors into the message
        gc.collect()
        result = queue.get(timeout=CHILD_TIMEOUT_SEC)
        process.join(timeout=CHILD_TIMEOUT_SEC)
        survivors = kill_subprocesses(process)
        assert not survivors, "child did not exit within timeout"
        assert process.exitcode == 0, f"child exited with {process.exitcode}"
        parent_sock.close()
        assert result == {"low": b"\x73" * 16, "high": b"\x74" * 16, "sizes": [gran, gran], "size": 2 * gran}
        assert _read(buf, 0, 16) == b"\x75" * 16
        buf.close()
        first.close()

    @staticmethod
    def child_main(device_id, sock, queue):
        device = Device(device_id)
        device.set_current()
        with sock:
            msg, fds, _flags, _addr = socket.recv_fds(sock, 4096, 16)
        header = json.loads(msg)
        desc = VirtualMemoryIPCBufferDescriptor.from_fds(
            fds, header["sizes"], handle_type=header["handle_type"], size=header["size"]
        )
        for fd in fds:
            os.close(fd)  # from_fds duplicated them; the received ones are ours to close
        imported = Buffer.from_ipc_descriptor(_resource(device), desc, stream=device.default_stream)
        gran = header["sizes"][0]
        result = {
            "low": _read(imported, 0, 16),
            "high": _read(imported, gran, 16),
            "sizes": list(desc.sizes),
            "size": desc.size,
        }
        _fill(imported, 0x75, size=gran)
        imported.close()
        queue.put(result)


class TestEmptyBuffer:
    @pytest.mark.agent_authored(model="claude-fable-5-1")
    def test_round_trip(self, vmm_ipc_device):
        """An empty buffer exports no allocations and imports as an empty mapped buffer."""
        device = vmm_ipc_device
        mr = _resource(device)
        empty = mr.allocate(0)
        desc = empty.ipc_descriptor
        assert desc.size == 0
        assert desc.sizes == ()
        assert desc.handles == ()
        assert desc.fds == ()
        alias = Buffer.from_ipc_descriptor(mr, desc, stream=device.default_stream)
        assert alias.size == 0
        assert alias.is_mapped
        alias.close()
        queue = mp.Queue()
        _run_child(self.child_main, device.device_id, desc, queue)
        assert queue.get(timeout=CHILD_TIMEOUT_SEC) == {"size": 0, "is_mapped": True}
        empty.close()

    @staticmethod
    def child_main(device_id, desc, queue):
        device = Device(device_id)
        device.set_current()
        imported = Buffer.from_ipc_descriptor(_resource(device), desc, stream=device.default_stream)
        queue.put({"size": imported.size, "is_mapped": imported.is_mapped})
        imported.close()


class TestDLPackConsumer:
    @pytest.mark.agent_authored(model="claude-fable-5-1")
    def test_imported_buffer_feeds_a_consumer(self, vmm_ipc_device):
        """An imported buffer, received as a Process argument, is a complete DLPack producer."""
        device = vmm_ipc_device
        mr = _resource(device)
        buf = mr.allocate(1)
        _fill(buf, 0x7E)
        queue = mp.Queue()
        _run_child(self.child_main, device.device_id, buf, queue)
        result = queue.get(timeout=CHILD_TIMEOUT_SEC)
        assert result["dl_device"] == (int(DLDeviceType.kDLCUDA), device.device_id)
        assert result["shape"] == (buf.size,)
        assert result["device_id"] == device.device_id
        assert result["is_device_accessible"] is True
        assert result["ptr_matches"] is True
        assert result["first_bytes"] == b"\x7e" * 16
        buf.close()

    @staticmethod
    def child_main(device_id, imported, queue):
        Device(device_id).set_current()
        assert isinstance(imported, VirtualMemoryBuffer)
        assert imported.is_mapped
        view = StridedMemoryView.from_dlpack(imported, stream_ptr=-1)
        result = {
            "dl_device": tuple(imported.__dlpack_device__()),
            "shape": tuple(view.shape),
            "device_id": view.device_id,
            "is_device_accessible": view.is_device_accessible,
            "ptr_matches": view.ptr == int(imported.handle),
            "first_bytes": _read(imported, 0, 16),
        }
        del view
        imported.close()
        queue.put(result)


class TestSameProcess:
    @pytest.mark.agent_authored(model="claude-fable-5-1")
    @pytest.mark.thread_unsafe(reason="checks mapping state by address; another thread can reuse a freed address")
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
    @pytest.mark.thread_unsafe(
        reason="file descriptor numbers are process-global; another thread can reuse a closed one"
    )
    def test_descriptor_owns_its_file_descriptors(self, vmm_ipc_device):
        """The exported file descriptors close when the descriptor is released, and plain pickle refuses them."""
        device = vmm_ipc_device
        mr = _resource(device)
        buf = mr.allocate(1)
        desc = buf.ipc_descriptor
        fd = desc.fds[0]
        os.fstat(fd)  # open
        with pytest.raises(TypeError):
            pickle.dumps(desc)  # file descriptors need multiprocessing
        with pytest.raises(TypeError):
            pickle.dumps(buf)
        del desc  # the buffer keeps no descriptor, so this closes the file descriptor
        gc.collect()
        with pytest.raises(OSError):
            os.fstat(fd)
        buf.close()

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
