# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Single-node VMM sharing with plain fabric handles (issue #2980)."""

import ctypes
import gc
import multiprocessing as mp
import os
import pickle
from contextlib import ExitStack, closing

import pytest
from helpers.child_processes import child_timeout_sec, kill_subprocesses, warmup_multiprocessing

from cuda.bindings import driver
from cuda.core import Buffer, Device, VirtualMemoryResource, VirtualMemoryResourceOptions
from cuda.core._device import _device_from_uuid
from cuda.core._memory._ipc import VirtualMemoryFabricBufferDescriptor
from cuda.core._utils.cuda_utils import CUDAError, handle_return
from cuda_python_test_helpers import IS_WINDOWS, IS_WSL

CHILD_TIMEOUT_SEC = child_timeout_sec()
pytestmark = pytest.mark.parallel_threads_limit(4)


def _resource(device, **options):
    options.setdefault("handle_type", "fabric")
    return VirtualMemoryResource(device, VirtualMemoryResourceOptions(**options))


def _fill(buf, value):
    handle_return(driver.cuMemsetD8(int(buf.handle), value, buf.size))
    handle_return(driver.cuCtxSynchronize())


def _read(buf):
    host = (ctypes.c_ubyte * 16)()
    handle_return(driver.cuMemcpyDtoH(ctypes.addressof(host), int(buf.handle), 16))
    return bytes(host)


def _notes(exc):
    # Python 3.10 appends the note to the message instead of using PEP 678.
    return " ".join((str(exc), *getattr(exc, "__notes__", ())))


def _check_import_error(exc):
    assert str(exc).startswith("CUDA_ERROR_NOT_SUPPORTED")
    for detail in ("stale", "exporting buffer was closed", "forged", "different IMEX channel"):
        assert detail in _notes(exc)


def _exchange(owner_uuid, payload):
    device = _device_from_uuid(owner_uuid)
    device.set_current()
    buf = (
        Buffer.from_ipc_descriptor(_resource(device), payload, stream=device.default_stream)
        if isinstance(payload, VirtualMemoryFabricBufferDescriptor)
        else payload
    )
    with buf:
        seen = _read(buf)
        _fill(buf, 0x62)
    return seen


def _child_exchange(owner_uuid, mode, incoming, outgoing):
    if mode == "queue":
        payload = incoming.get(timeout=CHILD_TIMEOUT_SEC)
    elif mode == "pipe":
        assert incoming.poll(CHILD_TIMEOUT_SEC), "no payload from parent"
        payload = incoming.recv()
    else:
        payload = incoming
    outgoing.send(_exchange(owner_uuid, payload))
    outgoing.close()


def _round_trip(mode, owner_uuid, payload):
    if mode == "pool":
        with mp.Pool(1) as pool:
            return pool.apply_async(_exchange, (owner_uuid, payload)).get(timeout=CHILD_TIMEOUT_SEC)
    with ExitStack() as stack:
        receive, send = mp.Pipe(duplex=False)
        stack.enter_context(receive)
        stack.enter_context(send)
        if mode == "queue":
            incoming = mp.Queue()
            stack.callback(incoming.join_thread)
            stack.callback(incoming.close)
            child_input = incoming
        elif mode == "pipe":
            child_input, incoming = mp.Pipe(duplex=False)
            stack.enter_context(child_input)
            stack.enter_context(incoming)
        else:
            child_input = payload
        process = stack.enter_context(
            closing(mp.Process(target=_child_exchange, args=(owner_uuid, mode, child_input, send)))
        )
        try:
            process.start()
            send.close()
            if mode == "queue":
                incoming.put(payload)
            elif mode == "pipe":
                child_input.close()
                incoming.send(payload)
            assert receive.poll(CHILD_TIMEOUT_SEC), "child did not respond"
            result = receive.recv()
            process.join(timeout=CHILD_TIMEOUT_SEC)
        finally:
            survivors = kill_subprocesses(process)
        assert not survivors, "child did not exit within timeout"
        assert process.exitcode == 0
        return result


@pytest.mark.agent_authored(model="gpt-6.1-sol")
def test_export_shape(vmm_fabric_device):
    device = vmm_fabric_device
    mr = _resource(device)
    with mr.allocate(1) as first, mr.modify_allocation(first, 2 * first.size) as grown:
        desc = grown.ipc_descriptor
        assert desc.handle_type == "fabric"
        assert desc.owner_uuid == device.uuid
        assert desc.sizes == (first.size, first.size)
        assert desc.size == grown.size == sum(desc.sizes)
        assert all(isinstance(handle, bytes) and len(handle) == 64 for handle in desc.handles)
        assert len(desc.handles) == 2
        assert desc.handles[0] != desc.handles[1]
        assert desc == grown.ipc_descriptor
        prefix = VirtualMemoryFabricBufferDescriptor.from_handles(desc.handles, desc.sizes, size=1)
        with Buffer.from_ipc_descriptor(mr, prefix, stream=device.default_stream) as alias:
            assert alias.size == 1 < sum(prefix.sizes)
        assert not any(hasattr(desc, name) for name in ("fds", "close", "is_closed"))
    with mr.allocate(0) as empty:
        desc = empty.ipc_descriptor
        assert (desc.handles, desc.sizes, desc.size) == ((), (), 0)
        with Buffer.from_ipc_descriptor(mr, desc, stream=device.default_stream) as alias:
            assert alias.is_mapped and alias.size == 0


@pytest.mark.agent_authored(model="gpt-6.1-sol")
def test_descriptor_pickle_and_validation():
    descriptor = VirtualMemoryFabricBufferDescriptor
    with pytest.raises(RuntimeError, match="cannot be instantiated directly"):
        descriptor()
    desc = descriptor.from_handles([bytes(64)], [4096], size=2048, owner_uuid="owner")
    rebuilt = pickle.loads(pickle.dumps(desc))  # noqa: S301  our own plain data
    assert rebuilt == desc and rebuilt is not desc
    assert hash(rebuilt) == hash(desc)
    assert rebuilt.owner_uuid == "owner"
    assert desc == descriptor.from_handles(desc.handles, desc.sizes, size=desc.size, owner_uuid="other")
    assert desc != descriptor.from_handles(desc.handles, desc.sizes)
    assert desc != descriptor.from_handles([b"x" * 64], desc.sizes, size=desc.size)
    assert desc != descriptor.from_handles(desc.handles, [8192], size=desc.size)
    assert desc != object()
    assert descriptor.from_handles(desc.handles, desc.sizes).size == 4096
    for handles, sizes, size in (
        ([bytes(63)], [4096], None),
        ([bytes(65)], [4096], None),
        ([bytearray(64)], [4096], None),
        ([bytes(64)], [], None),
        ([bytes(64)], [0], None),
        ([bytes(64)], [-1], None),
        ([bytes(64)], [1.5], None),
        ([bytes(64)], [4096], 4097),
        ([bytes(64)], [4096], -1),
        ([bytes(64)], [4096], 1.5),
    ):
        with pytest.raises(ValueError):
            descriptor.from_handles(handles, sizes, size=size)


@pytest.mark.parametrize("mode", ["queue", "pipe", "argument", "pool"])
@pytest.mark.parametrize("send_buffer", [False, True], ids=["descriptor", "buffer"])
@pytest.mark.thread_unsafe(reason="counts process-global file descriptors")
@pytest.mark.agent_authored(model="gpt-6.1-sol")
def test_cross_process(vmm_fabric_device, mode, send_buffer):
    warmup_multiprocessing()
    device = vmm_fabric_device
    with _resource(device).allocate(1) as buf:
        _fill(buf, 0x51)
        gc.collect()
        before = len(os.listdir("/proc/self/fd"))
        payload = buf if send_buffer else buf.ipc_descriptor
        assert len(os.listdir("/proc/self/fd")) == before
        assert _round_trip(mode, device.uuid, payload) == b"\x51" * 16
        assert _read(buf) == b"\x62" * 16
        gc.collect()
        assert len(os.listdir("/proc/self/fd")) == before


@pytest.mark.agent_authored(model="gpt-6.1-sol")
def test_second_gpu_and_wrong_owner(vmm_fabric_device_x2):
    owner, peer = vmm_fabric_device_x2
    if not peer.can_access_peer(owner):
        pytest.skip("Second device cannot access the owner's memory")
    mr = _resource(owner, peers=[peer.device_id])
    with mr.allocate(1) as buf:
        _fill(buf, 0x53)
        desc = buf.ipc_descriptor
        with pytest.raises(ValueError) as info:
            Buffer.from_ipc_descriptor(_resource(peer), desc, stream=peer.default_stream)
        assert f"memory belongs to the device with UUID {owner.uuid}" in str(info.value)
        assert f"which is device {owner.device_id} in this process" in str(info.value)
        hidden_owner = VirtualMemoryFabricBufferDescriptor.from_handles(
            desc.handles, desc.sizes, size=desc.size, owner_uuid="missing-owner"
        )
        with pytest.raises(ValueError, match="UUID missing-owner, which is not visible in this process"):
            Buffer.from_ipc_descriptor(_resource(peer), hidden_owner, stream=peer.default_stream)
        with Buffer.from_ipc_descriptor(mr, desc, stream=owner.default_stream) as alias:
            # set_current replaces the top of the context stack, so switch back the same way.
            peer.set_current()
            try:
                assert _read(alias) == b"\x53" * 16
                _fill(alias, 0x54)
            finally:
                owner.set_current()
            assert _read(buf) == b"\x54" * 16
            buf.close()
            assert _read(alias) == b"\x54" * 16  # the imported allocation owns a reference


@pytest.mark.agent_authored(model="gpt-6.1-sol")
def test_handle_type_mismatch(vmm_fabric_device):
    device = vmm_fabric_device
    fabric, fd = _resource(device), _resource(device, handle_type="posix_fd")
    # Empty descriptors would need no import call; even these must reject a mismatch.
    for exporting, importing in ((fabric, fd), (fd, fabric)):
        with exporting.allocate(0) as buf, pytest.raises(ValueError, match="configured like the exporter's"):
            Buffer.from_ipc_descriptor(importing, buf.ipc_descriptor, stream=device.default_stream)


@pytest.mark.parametrize("handle", [bytes(64), os.urandom(64)], ids=["zero", "random"])
@pytest.mark.agent_authored(model="gpt-6.1-sol")
def test_garbage(vmm_fabric_device, handle):
    device = vmm_fabric_device
    mr = _resource(device)
    with mr.allocate(1) as buf:
        desc = VirtualMemoryFabricBufferDescriptor.from_handles([handle], [buf.size])
        with pytest.raises(CUDAError) as info:
            Buffer.from_ipc_descriptor(mr, desc, stream=device.default_stream)
        _check_import_error(info.value)


@pytest.mark.agent_authored(model="gpt-6.1-sol")
def test_stale_descriptor(vmm_fabric_device):
    device = vmm_fabric_device
    mr = _resource(device)
    with mr.allocate(1) as buf:
        desc = buf.ipc_descriptor
    with pytest.raises(CUDAError) as info:
        Buffer.from_ipc_descriptor(mr, desc, stream=device.default_stream)
    _check_import_error(info.value)


@pytest.mark.agent_authored(model="gpt-6.1-sol")
def test_reexport_imported_derivatives(vmm_fabric_device):
    device = vmm_fabric_device
    mr = _resource(device)
    with ExitStack() as stack:
        buf = stack.enter_context(mr.allocate(1))
        alias = stack.enter_context(Buffer.from_ipc_descriptor(mr, buf.ipc_descriptor, stream=device.default_stream))
        same = stack.enter_context(mr.modify_allocation(alias, alias.size))
        smaller_request = stack.enter_context(mr.modify_allocation(alias, 1))
        grown = stack.enter_context(mr.modify_allocation(alias, 2 * alias.size))
        for imported in (alias, same, smaller_request, grown):
            with pytest.raises(RuntimeError, match="imported from another process"):
                _ = imported.ipc_descriptor
            with pytest.raises(RuntimeError, match="imported from another process"):
                pickle.dumps(imported)


@pytest.mark.agent_authored(model="gpt-6.1-sol")
def test_configuration_and_buffer_pickle(vmm_fabric_device):
    device = vmm_fabric_device
    mr = _resource(device)
    assert mr.is_ipc_enabled
    for location in ("host", "host_numa", "host_numa_current"):
        host = _resource(device, location_type=location, handle_type=None if location == "host" else "fabric")
        assert not host.is_ipc_enabled
    with mr.allocate(1) as buf:
        _fill(buf, 0x55)
        with pickle.loads(pickle.dumps(buf)) as alias:  # noqa: S301  our own live buffer
            assert alias.is_mapped
            assert _read(alias) == b"\x55" * 16
            _fill(alias, 0x56)
        assert _read(buf) == b"\x56" * 16


def _child_stale_queue(queue, ready, outgoing):
    assert ready.wait(CHILD_TIMEOUT_SEC), "parent did not release the export"
    try:
        buf = queue.get(timeout=CHILD_TIMEOUT_SEC)
    except CUDAError as exc:
        _check_import_error(exc)
        outgoing.send("stale")
    else:
        buf.close()
        raise AssertionError("queued fabric buffer pinned the allocation")
    assert queue.get(timeout=CHILD_TIMEOUT_SEC) == "next item"
    outgoing.close()


@pytest.mark.agent_authored(model="gpt-6.1-sol")
def test_queued_buffer_does_not_pin(vmm_fabric_device):
    device = vmm_fabric_device
    mr = _resource(device)
    with ExitStack() as stack:
        buf = stack.enter_context(mr.allocate(1))
        unaffected = stack.enter_context(mr.allocate(1))
        _fill(unaffected, 0x57)
        queue = mp.Queue()
        stack.callback(queue.join_thread)
        stack.callback(queue.close)
        ready = mp.Event()
        receive, send = mp.Pipe(duplex=False)
        stack.enter_context(receive)
        stack.enter_context(send)
        process = stack.enter_context(closing(mp.Process(target=_child_stale_queue, args=(queue, ready, send))))
        try:
            process.start()
            send.close()
            queue.put(buf)
            queue.put("next item")
            # Finish pickling before close, but hold the child before unpickling.
            queue.close()
            queue.join_thread()
            buf.close()
            ready.set()
            assert receive.poll(CHILD_TIMEOUT_SEC), "child did not report the stale import"
            assert receive.recv() == "stale"
            process.join(timeout=CHILD_TIMEOUT_SEC)
        finally:
            survivors = kill_subprocesses(process)
        assert not survivors, "child did not exit within timeout"
        assert process.exitcode == 0
        assert _read(unaffected) == b"\x57" * 16
        with mr.allocate(1) as fresh:
            _fill(fresh, 0x58)
            assert _read(fresh) == b"\x58" * 16


@pytest.mark.skipif(IS_WINDOWS or IS_WSL, reason="Fabric IPC requires Linux")
@pytest.mark.agent_authored(model="gpt-6.1-sol")
def test_unsupported_fabric_construction(init_cuda):
    unsupported = [
        device
        for device in Device.get_all_devices()
        if device.properties.virtual_memory_management_supported and not device.properties.handle_type_fabric_supported
    ]
    if not unsupported:
        pytest.skip("No visible VMM device without fabric handle support")
    for device in unsupported:
        with pytest.raises(RuntimeError) as info:
            _resource(device)
        message = str(info.value)
        assert "NVLink fabric" in message
        assert "accessible IMEX channel (/dev/nvidia-caps-imex-channels/channel*)" in message
        assert f"device {device.device_id} reports no fabric handle support" in message
