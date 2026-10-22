# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Devices and resources sent to another process keep their devices when the ordinals differ.

The child process sees the parent's two devices in reversed order, so each
ordinal names the other device there. A ``Device`` must arrive as the device
with the same UUID, the child must rebuild a resource for the device that
owns the memory, and the peers of a ``VirtualMemoryResource`` must name the
same devices as in the parent (issue #3097).
"""

import contextlib
import ctypes
import multiprocessing as mp
import os

import pytest
from helpers.child_processes import child_timeout_sec, kill_subprocesses
from helpers.constants import POOL_SIZE

from cuda.bindings import driver
from cuda.core import (
    Buffer,
    Device,
    DeviceMemoryResource,
    DeviceMemoryResourceOptions,
    VirtualMemoryResource,
    VirtualMemoryResourceOptions,
)
from cuda.core._utils.cuda_utils import handle_return

CHILD_TIMEOUT_SEC = child_timeout_sec()
NBYTES = 64
PATTERN = 0x5A


@contextlib.contextmanager
def _visible_devices(*devices):
    """Make only ``devices``, in this order, visible to processes started in the block.

    The spawn start method copies ``os.environ`` when ``Process.start()`` runs.
    The devices are named by UUID, so the order does not depend on how this
    process numbers them.
    """
    saved = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(f"GPU-{device.uuid}" for device in devices)
    try:
        yield
    finally:
        if saved is None:
            del os.environ["CUDA_VISIBLE_DEVICES"]
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = saved


def _send_to_child(obj, visible):
    """Send ``obj`` to a child that sees only ``visible``, and return the child's report."""
    inbox, outbox = mp.Queue(), mp.Queue()
    process = mp.Process(target=_child_main, args=(inbox, outbox))
    with _visible_devices(*visible):
        process.start()
    inbox.put(obj)
    try:
        return outbox.get(timeout=CHILD_TIMEOUT_SEC)
    finally:
        process.join(timeout=CHILD_TIMEOUT_SEC)
        survivors = kill_subprocesses(process)
        assert not survivors, "child did not exit within timeout"
        assert process.exitcode == 0, f"child exited with {process.exitcode}"


def _child_main(inbox, outbox):
    try:
        report = _receive(inbox.get(timeout=CHILD_TIMEOUT_SEC))
    except Exception as exc:  # report every failure to the parent, which asserts on it
        report = {"error": type(exc).__name__, "message": str(exc)}
    outbox.put(report)


def _receive(obj):
    """Describe the received object in terms of device UUIDs, which do not depend on the ordinals."""
    if isinstance(obj, Device):
        return {"device_id": obj.device_id, "device_uuid": obj.uuid}
    if not isinstance(obj, Buffer):
        return {"received": type(obj).__name__}
    mr = obj.memory_resource
    report = {"device_id": obj.device_id, "device_uuid": Device(obj.device_id).uuid}
    host = (ctypes.c_ubyte * NBYTES)()
    handle_return(driver.cuMemcpyDtoH(ctypes.addressof(host), int(obj.handle), NBYTES))
    report["data"] = bytes(host)
    if isinstance(mr, VirtualMemoryResource):
        report["peer_uuids"] = [Device(peer).uuid for peer in mr.config.peers]
        report["peer_access"] = [_access_flags(obj, peer) for peer in mr.config.peers]
    obj.close()
    return report


def _access_flags(buf, device_id):
    """The access a device has to ``buf``: 0 for none, 3 for read-write."""
    location = driver.CUmemLocation()
    location.type = driver.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
    location.id = device_id
    return int(handle_return(driver.cuMemGetAccess(location, int(buf.handle))))


def _fill(device, buf):
    device.set_current()
    handle_return(driver.cuMemsetD8(int(buf.handle), PATTERN, NBYTES))
    device.sync()


def _vmm_resource(device, peers=()):
    return VirtualMemoryResource(device, config=VirtualMemoryResourceOptions(handle_type="posix_fd", peers=peers))


def _check_report(report, owner):
    assert "error" not in report, f"the child could not receive the buffer: {report}"
    # The child numbers the devices the other way round; otherwise this test proves nothing.
    assert report["device_id"] != owner.device_id, "CUDA_VISIBLE_DEVICES did not reorder the devices in the child"
    assert report["device_uuid"] == owner.uuid
    assert report["data"] == bytes([PATTERN]) * NBYTES


@pytest.mark.agent_authored(model="claude-opus-5-5")
@pytest.mark.thread_unsafe(reason="sets CUDA_VISIBLE_DEVICES, which is process-global")
def test_device(device_x2):
    """A Device arrives as the device with the same UUID, not the device with the same ordinal."""
    dev0, dev1 = device_x2
    report = _send_to_child(dev0, visible=(dev1, dev0))
    assert "error" not in report, f"the child could not receive the device: {report}"
    assert report["device_id"] != dev0.device_id, "CUDA_VISIBLE_DEVICES did not reorder the devices in the child"
    assert report["device_uuid"] == dev0.uuid


@pytest.mark.agent_authored(model="claude-opus-5-5")
@pytest.mark.thread_unsafe(reason="sets CUDA_VISIBLE_DEVICES, which is process-global")
def test_device_not_visible(device_x2):
    """A child that cannot see the device refuses it and names the missing UUID."""
    dev0, dev1 = device_x2
    report = _send_to_child(dev0, visible=(dev1,))
    assert report["error"] == "RuntimeError", report
    assert f"no CUDA device visible to this process has UUID {dev0.uuid}" in report["message"]


@pytest.mark.agent_authored(model="claude-opus-5-5")
@pytest.mark.thread_unsafe(reason="sets CUDA_VISIBLE_DEVICES, which is process-global")
def test_device_memory_resource_buffer(ipc_mempool_device_x2):
    """A pool-backed buffer comes back on the device that owns it, not on the device with the same ordinal."""
    dev0, dev1 = ipc_mempool_device_x2
    mr = DeviceMemoryResource(dev0, options=DeviceMemoryResourceOptions(max_size=POOL_SIZE, ipc_enabled=True))
    buf = mr.allocate(NBYTES, stream=dev0.default_stream)
    _fill(dev0, buf)

    report = _send_to_child(buf, visible=(dev1, dev0))
    _check_report(report, dev0)

    buf.close()
    dev0.sync()
    mr.close()


@pytest.mark.agent_authored(model="claude-opus-5-5")
@pytest.mark.thread_unsafe(reason="sets CUDA_VISIBLE_DEVICES, which is process-global")
def test_virtual_memory_resource_buffer(vmm_ipc_device_x2):
    """A VirtualMemoryResource buffer comes back on the device that owns it."""
    dev0, dev1 = vmm_ipc_device_x2
    mr = _vmm_resource(dev0)
    buf = mr.allocate(NBYTES)
    _fill(dev0, buf)

    report = _send_to_child(buf, visible=(dev1, dev0))
    _check_report(report, dev0)
    assert report["peer_uuids"] == []

    buf.close()
    dev0.sync()


@pytest.mark.agent_authored(model="claude-opus-5-5")
@pytest.mark.thread_unsafe(reason="sets CUDA_VISIBLE_DEVICES, which is process-global")
def test_virtual_memory_resource_peers(vmm_ipc_device_x2):
    """The peers of a VirtualMemoryResource name the same devices in the child, and those devices get access."""
    dev0, dev1 = vmm_ipc_device_x2
    if not dev1.can_access_peer(dev0):
        pytest.skip("Test requires peer access from the second device to the first")
    mr = _vmm_resource(dev0, peers=(dev1.device_id,))
    buf = mr.allocate(NBYTES)
    _fill(dev0, buf)

    report = _send_to_child(buf, visible=(dev1, dev0))
    _check_report(report, dev0)
    assert report["peer_uuids"] == [dev1.uuid]
    assert report["peer_access"] == [3]

    buf.close()
    dev0.sync()


@pytest.mark.agent_authored(model="claude-opus-5-5")
@pytest.mark.thread_unsafe(reason="sets CUDA_VISIBLE_DEVICES, which is process-global")
def test_virtual_memory_resource_peers_from_generator(vmm_ipc_device_x2):
    """Peers given as a generator reach every allocation and the pickled resource, not only the first allocation."""
    dev0, dev1 = vmm_ipc_device_x2
    if not dev1.can_access_peer(dev0):
        pytest.skip("Test requires peer access from the second device to the first")
    mr = _vmm_resource(dev0, peers=(peer for peer in (dev1.device_id,)))
    first = mr.allocate(NBYTES)
    buf = mr.allocate(NBYTES)
    _fill(dev0, buf)
    assert _access_flags(first, dev1.device_id) == 3
    assert _access_flags(buf, dev1.device_id) == 3

    report = _send_to_child(buf, visible=(dev1, dev0))
    _check_report(report, dev0)
    assert report["peer_uuids"] == [dev1.uuid]
    assert report["peer_access"] == [3]

    first.close()
    buf.close()
    dev0.sync()


@pytest.mark.agent_authored(model="claude-opus-5-5")
@pytest.mark.thread_unsafe(reason="sets CUDA_VISIBLE_DEVICES, which is process-global")
@pytest.mark.parametrize("kind", ["device", "vmm"])
def test_owning_device_not_visible(request, kind):
    """A child that cannot see the owning device refuses the resource and names the missing UUID."""
    if kind == "device":
        dev0, dev1 = request.getfixturevalue("ipc_mempool_device_x2")
        mr = DeviceMemoryResource(dev0, options=DeviceMemoryResourceOptions(max_size=POOL_SIZE, ipc_enabled=True))
    else:
        dev0, dev1 = request.getfixturevalue("vmm_ipc_device_x2")
        mr = _vmm_resource(dev0)

    # The child receives the resource itself: it fails before it needs any buffer's handles.
    report = _send_to_child(mr, visible=(dev1,))
    assert report["error"] == "RuntimeError", report
    assert f"no CUDA device visible to this process has UUID {dev0.uuid}" in report["message"]
    assert dev1.uuid in report["message"]

    if kind == "device":
        mr.close()
