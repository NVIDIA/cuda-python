# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import ctypes
import struct
from contextlib import closing

import numpy as np
import pytest

from cuda.core import HOST_COPY_FN, Buffer, Device, LegacyPinnedMemoryResource, host_launch, pack_host_copy_args

N = 4096


@pytest.mark.agent_authored(model="claude-opus-5-5")
def test_host_copy_fn_has_cuhostfn_prototype():
    assert HOST_COPY_FN.restype is None
    assert tuple(HOST_COPY_FN.argtypes) == (ctypes.c_void_p,)


@pytest.mark.agent_authored(model="claude-opus-5-5")
def test_host_copy_orders_between_prior_and_later_work(init_cuda):
    dev = Device(0)
    dev.set_current()
    dst = np.zeros(N, dtype=np.uint8)
    with (
        closing(dev.create_stream()) as stream,
        dev.allocate(N, stream=stream) as device_buffer,
        LegacyPinnedMemoryResource().allocate(N) as pinned,
    ):
        device_buffer.fill(0x11, stream=stream)
        device_buffer.copy_to(pinned, stream=stream)
        host_launch(stream, HOST_COPY_FN, user_data=pack_host_copy_args(dst, pinned, N))
        device_buffer.fill(0x22, stream=stream)
        device_buffer.copy_to(pinned, stream=stream)
        stream.sync()
        assert (dst == 0x11).all()


@pytest.mark.agent_authored(model="claude-opus-5-5")
def test_host_copy_int_user_data_points_to_a_packed_record(init_cuda):
    dev = Device(0)
    dev.set_current()
    src = np.arange(N, dtype=np.uint8)
    dst = np.zeros(N, dtype=np.uint8)
    record = ctypes.create_string_buffer(struct.pack("@PPN", dst.ctypes.data, src.ctypes.data, N), 24)
    with closing(dev.create_stream()) as stream:
        host_launch(stream, HOST_COPY_FN, user_data=ctypes.addressof(record))
        stream.sync()
    assert (dst == src).all()


@pytest.mark.agent_authored(model="claude-opus-5-5")
def test_host_copy_replays_in_graph(init_cuda):
    dev = Device(0)
    dev.set_current()
    src = np.full(N, 0x5A, dtype=np.uint8)
    dst = np.zeros(N, dtype=np.uint8)
    with closing(dev.create_stream()) as stream:
        gb = stream.create_graph_builder().begin_building()
        host_launch(gb, HOST_COPY_FN, user_data=pack_host_copy_args(dst, src, N))
        gb.end_building()
        with closing(gb.complete()) as graph:
            graph.launch(stream)
            stream.sync()
            assert (dst == 0x5A).all()
            src[:] = 0x6B
            graph.launch(stream)
            stream.sync()
            assert (dst == 0x6B).all()


@pytest.mark.agent_authored(model="claude-opus-5-5")
def test_host_copy_zero_bytes_does_nothing(init_cuda):
    dev = Device(0)
    dev.set_current()
    with closing(dev.create_stream()) as stream:
        host_launch(stream, HOST_COPY_FN, user_data=pack_host_copy_args(0, 0, 0))
        stream.sync()


@pytest.mark.parametrize("user_data", [None, 0, b"", b"\x00" * 23])
@pytest.mark.agent_authored(model="claude-opus-5-5")
def test_host_launch_rejects_host_copy_without_a_record(init_cuda, user_data):
    dev = Device(0)
    dev.set_current()
    with closing(dev.create_stream()) as stream, pytest.raises(ValueError, match="HOST_COPY_FN"):
        host_launch(stream, HOST_COPY_FN, user_data=user_data)


@pytest.mark.agent_authored(model="claude-opus-5-5")
def test_pack_host_copy_args_layout():
    data = pack_host_copy_args(0x1000, 0x2000, 16)
    assert data == struct.pack("@PPN", 0x1000, 0x2000, 16)
    assert struct.unpack("@PPN", data) == (0x1000, 0x2000, 16)


@pytest.mark.agent_authored(model="claude-opus-5-5")
def test_pack_host_copy_args_takes_addresses_from_objects(init_cuda):
    src = np.arange(N, dtype=np.uint8)
    dst = bytearray(N)
    with LegacyPinnedMemoryResource().allocate(N) as pinned:
        assert struct.unpack("@PPN", pack_host_copy_args(pinned, src, N)) == (int(pinned.handle), src.ctypes.data, N)
        dst_address = ctypes.addressof((ctypes.c_char * N).from_buffer(dst))
        assert struct.unpack("@PPN", pack_host_copy_args(dst, pinned, N)) == (dst_address, int(pinned.handle), N)


@pytest.mark.agent_authored(model="claude-opus-5-5")
def test_pack_host_copy_args_accepts_buffer_without_memory_resource():
    src = np.arange(N, dtype=np.uint8)
    dst = np.zeros(N, dtype=np.uint8)
    buffer = Buffer.from_handle(dst.ctypes.data, N, owner=dst)
    assert struct.unpack("@PPN", pack_host_copy_args(buffer, src, N)) == (dst.ctypes.data, src.ctypes.data, N)


@pytest.mark.agent_authored(model="claude-opus-5-5")
def test_pack_host_copy_args_checks_bounds(init_cuda):
    small = np.zeros(N - 1, dtype=np.uint8)
    large = np.zeros(N, dtype=np.uint8)
    with pytest.raises(ValueError, match="exceeds the size of dst"):
        pack_host_copy_args(small, large, N)
    with pytest.raises(ValueError, match="exceeds the size of src"):
        pack_host_copy_args(large, small, N)
    with LegacyPinnedMemoryResource().allocate(N - 1) as pinned, pytest.raises(ValueError, match="exceeds the size"):
        pack_host_copy_args(large, pinned, N)


@pytest.mark.agent_authored(model="claude-opus-5-5")
def test_pack_host_copy_args_rejects_device_buffer(init_cuda):
    dev = Device(0)
    dev.set_current()
    host = np.zeros(N, dtype=np.uint8)
    with closing(dev.create_stream()) as stream, dev.allocate(N, stream=stream) as device_buffer:
        with pytest.raises(ValueError, match="not host-accessible"):
            pack_host_copy_args(device_buffer, host, N)
        with pytest.raises(ValueError, match="not host-accessible"):
            pack_host_copy_args(host, device_buffer, N)


@pytest.mark.agent_authored(model="claude-opus-5-5")
def test_pack_host_copy_args_rejects_closed_buffer(init_cuda):
    pinned = LegacyPinnedMemoryResource().allocate(N)
    pinned.close()
    with pytest.raises(RuntimeError, match="closed"):
        pack_host_copy_args(np.zeros(N, dtype=np.uint8), pinned, N)


@pytest.mark.agent_authored(model="claude-opus-5-5")
def test_pack_host_copy_args_rejects_read_only_dst():
    with pytest.raises(BufferError):
        pack_host_copy_args(bytes(N), bytearray(N), N)


@pytest.mark.agent_authored(model="claude-opus-5-5")
def test_pack_host_copy_args_rejects_unsupported_type():
    with pytest.raises(TypeError, match="dst must be"):
        pack_host_copy_args(1.0, bytearray(N), N)


@pytest.mark.parametrize("address", [True, np.uintp(0x1000), ctypes.c_void_p(0x1000)])
@pytest.mark.agent_authored(model="claude-opus-5-5")
def test_pack_host_copy_args_rejects_scalar_address(address):
    with pytest.raises(TypeError, match=r"int\(addr\)"):
        pack_host_copy_args(bytearray(8), address, 1)


@pytest.mark.agent_authored(model="claude-opus-5-5")
def test_pack_host_copy_args_rejects_overlap():
    data = bytearray(N)
    view = memoryview(data)
    with pytest.raises(ValueError, match="overlap"):
        pack_host_copy_args(view[1:], view, N - 1)
    with pytest.raises(ValueError, match="overlap"):
        pack_host_copy_args(view, view[1:], N - 1)
    pack_host_copy_args(view[N // 2 :], view, N // 2)
    pack_host_copy_args(view, view[N // 2 :], N // 2)


@pytest.mark.agent_authored(model="claude-opus-5-5")
def test_pack_host_copy_args_rejects_null_address():
    with pytest.raises(ValueError, match="null"):
        pack_host_copy_args(0, bytearray(N), N)


@pytest.mark.parametrize(
    ("dst", "nbytes"),
    [(0x1000, -1), (0x1000, 1 << 64), (-1, 1), (1 << 64, 1)],
)
@pytest.mark.agent_authored(model="claude-opus-5-5")
def test_pack_host_copy_args_rejects_out_of_range_int(dst, nbytes):
    with pytest.raises(OverflowError):
        pack_host_copy_args(dst, 0x2000, nbytes)
