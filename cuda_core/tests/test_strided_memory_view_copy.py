# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0


import ctypes
import logging
import random
from contextlib import contextmanager
from enum import Enum
from io import StringIO

import cuda.core.experimental as ccx
import numpy as np
import pytest
from cuda.core.experimental import Buffer
from cuda.core.experimental import (
    system as ccx_system,
)
from cuda.core.experimental._strided_copy._copy import _with_logger
from cuda.core.experimental.utils import (
    CopyAllocatorOptions,
    StridedLayout,
    StridedMemoryView,
)

try:
    import cupy as cp
except ImportError:
    cp = None

from helpers.layout import (
    NamedParam,
    dtype_from_itemsize,
    inv_permutation,
    permuted,
    pretty_name,
)


class CopyDirection(Enum):
    D2D = "d2d"
    D2H = "d2h"
    H2D = "h2d"


class SrcFormat(Enum):
    C_CONTIGUOUS = "c_contiguous"  # the src is contigious in some order
    F_CONTIGUOUS = "f_contiguous"  # the src is contigious in some order
    SLICED = "sliced"  # the src is sliced (has gaps)


class Transpose(Enum):
    SAME_ORDER = False  # the src and dst have the same stride order
    INVERSE = "inverse"  # the src and dst have the inverse stride order (e.g. F <-> C)
    PERMUTATION = "permutation"  # the src and dst have permuted stride orders


class DstFormat(Enum):
    DENSE = "dense"  # the dst is contigious in some order
    SLICED = "sliced"  # the dst is sliced (has gaps)


class Broadcast(Enum):
    NO = False
    RIGHT = "right"
    LEFT = "left"
    RIGHT_LEFT = "right_left"


class CustomAllocator(Enum):
    HOST = "host"
    DEVICE = "device"
    BOTH = "host_device"


_ITEMSIZES = [1, 2, 4, 8, 16]

py_rng = random.Random(43)
num_devices = ccx_system.num_devices


def get_ptr(array):
    if isinstance(array, np.ndarray):
        return array.ctypes.data
    elif isinstance(array, cp.ndarray):
        return array.data.ptr
    else:
        raise ValueError(f"Invalid array: {type(array)}")


@contextmanager
def with_ccx_device(device_id):
    current_dev = ccx.Device()
    current_dev_id = current_dev.device_id
    dev = ccx.Device(device_id)
    try:
        dev.set_current()
        yield dev
    finally:
        if current_dev_id != device_id:
            current_dev.set_current()


_log_stream = StringIO()
_logger = logging.Logger("smv_copy_test", level=logging.DEBUG)
_logger.addHandler(logging.StreamHandler(_log_stream))
_logger.setLevel(logging.DEBUG)


@contextmanager
def with_logger():
    try:
        with _with_logger(_logger):
            yield _log_stream
    finally:
        _log_stream.truncate(0)


def get_src_order(src_format):
    assert isinstance(src_format, SrcFormat)
    if src_format == SrcFormat.F_CONTIGUOUS:
        return "F"
    return "C"


def get_dst_order(rng, src_format, transpose, shape):
    assert isinstance(transpose, Transpose)
    src_order = get_src_order(src_format)
    if transpose == Transpose.SAME_ORDER:
        return src_order
    elif transpose == Transpose.INVERSE:
        return "F" if src_order == "C" else "C"
    else:
        assert transpose == Transpose.PERMUTATION
        perm = list(range(len(shape)))
        rng.shuffle(perm)
        return tuple(perm)


def get_src_shape(base_src_shape, broadcast):
    assert isinstance(broadcast, Broadcast)
    match broadcast:
        case Broadcast.NO | Broadcast.LEFT:
            return base_src_shape
        case Broadcast.RIGHT | Broadcast.RIGHT_LEFT:
            return base_src_shape + (1,)
    raise ValueError(f"Invalid broadcast: {broadcast}")


def get_dst_shape(rng, base_src_shape, broadcast):
    assert isinstance(broadcast, Broadcast)
    match broadcast:
        case Broadcast.NO:
            return base_src_shape
        case Broadcast.RIGHT:
            return base_src_shape + (rng.randint(2, 5),)
        case Broadcast.RIGHT_LEFT:
            return (rng.randint(2, 5),) + base_src_shape + (rng.randint(2, 5),)
        case Broadcast.LEFT:
            return (rng.randint(2, 5),) + base_src_shape
    raise ValueError(f"Invalid broadcast: {broadcast}")


def is_h2h_needed(direction, src_shape, src_format, dst_shape, dst_format, broadcast):
    if direction == CopyDirection.D2D:
        return False
    elif direction == CopyDirection.H2D:
        return len(src_shape) > 1 and (src_format == SrcFormat.SLICED) or broadcast != Broadcast.NO
    else:
        assert direction == CopyDirection.D2H
        return len(dst_shape) > 1 and (dst_format == DstFormat.SLICED)


def get_src(is_host, src_format, src_order, implicit_c, shape, itemsize):
    assert isinstance(src_format, SrcFormat)
    assert isinstance(implicit_c, bool)
    mod = np if is_host else cp
    dtype = dtype_from_itemsize(itemsize)
    slices = None
    if src_format == SrcFormat.SLICED:
        shape = tuple(e + 2 for e in shape)
        slices = tuple(slice(1, -1) for _ in shape)
    array = mod.arange(np.prod(shape), dtype=dtype).reshape(shape, order=src_order)
    smv = StridedMemoryView.from_dlpack(array, -1)
    # enforce implicit C strides
    if implicit_c and smv.layout.is_contiguous_c:
        layout = StridedLayout(smv.shape, None, itemsize)
        smv = smv.view(layout=layout)
    if slices is not None:
        base_ptr = smv.ptr
        smv = smv.view(layout=smv.layout[slices])
        assert smv.layout.slice_offset != 0
        assert smv.ptr == base_ptr + smv.layout.slice_offset_in_bytes
        assert smv.shape != shape
        assert smv.layout.volume > 0
    return smv


def get_dst(
    is_host,
    src_order,
    transpose,
    dst_order,
    implicit_c,
    dst_format,
    dst_shape,
    itemsize,
):
    assert isinstance(dst_format, DstFormat)
    assert isinstance(implicit_c, bool)
    mod = np if is_host else cp
    dtype = dtype_from_itemsize(itemsize)
    slices = None
    if dst_format == DstFormat.SLICED:
        dst_shape = tuple(e + 2 for e in dst_shape)
        slices = tuple(slice(1, -1) for _ in dst_shape)
    if transpose == Transpose.SAME_ORDER:
        assert src_order in "CF"
        assert dst_order == src_order
    elif transpose == Transpose.INVERSE:
        assert dst_order in "CF"
        assert dst_order != src_order
    else:
        assert transpose == Transpose.PERMUTATION
        assert isinstance(dst_order, tuple)
    if transpose == Transpose.PERMUTATION:
        array = mod.arange(np.prod(dst_shape), dtype=dtype)
        array = array.reshape(permuted(dst_shape, dst_order)).transpose(inv_permutation(dst_order))
    else:
        array = mod.arange(np.prod(dst_shape), dtype=dtype).reshape(dst_shape, order=dst_order)
    smv = StridedMemoryView.from_dlpack(array, -1)
    # enforce implicit C strides
    if implicit_c and smv.layout.is_contiguous_c:
        layout = StridedLayout(smv.shape, None, itemsize)
        smv = smv.view(layout=layout)
    if slices is not None:
        base_ptr = smv.ptr
        smv = smv.view(layout=smv.layout[slices])
        assert smv.layout.slice_offset != 0
        assert smv.ptr == base_ptr + smv.layout.slice_offset_in_bytes
        assert smv.shape != dst_shape
        assert smv.layout.volume > 0
    return smv


def as_array(device_id, smv):
    min_offset, max_offset = smv.layout.offset_bounds
    size = (max_offset - min_offset + 1) * smv.layout.itemsize
    dtype = smv.dtype
    if dtype is None:
        dtype = dtype_from_itemsize(smv.layout.itemsize)
    if device_id is None:
        c_mem = memoryview((ctypes.c_char * size).from_address(smv.ptr))
        np_array = np.frombuffer(c_mem, dtype=dtype)
        if smv.layout.strides_in_bytes is None:
            return np_array.reshape(smv.shape, order="C")
        else:
            return np.lib.stride_tricks.as_strided(
                np_array,
                shape=smv.shape,
                strides=smv.layout.strides_in_bytes,
            )
    else:
        assert smv.is_device_accessible
        um = cp.cuda.UnownedMemory(smv.ptr, size, smv, device_id)
        mem = cp.cuda.MemoryPointer(um, 0)
        cp_array = cp.ndarray(
            shape=smv.shape,
            strides=smv.layout.strides_in_bytes,
            dtype=dtype,
            memptr=mem,
        )
        return cp_array


@pytest.mark.parametrize(
    (
        "src_shape",
        "dst_shape",
        "direction",
        "src_format",
        "transpose",
        "dst_format",
        "implicit_c",
        "broadcast",
        "itemsize",
        "device_id",
        "default_stream",
        "src_order",
        "dst_order",
        "copy_from",
        "blocking",
    ),
    [
        (
            NamedParam("src_shape", src_shape),
            NamedParam("dst_shape", dst_shape),
            direction,
            src_format,
            transpose,
            dst_format,
            NamedParam("implicit_c", implicit_c),
            broadcast,
            NamedParam("itemsize", py_rng.choice(_ITEMSIZES)),
            NamedParam(
                "device_id",
                py_rng.randint(0, num_devices - 1) if num_devices >= 0 else None,
            ),
            NamedParam("default_stream", py_rng.choice([True, False])),
            NamedParam("src_order", get_src_order(src_format)),
            NamedParam("dst_order", get_dst_order(py_rng, src_format, transpose, dst_shape)),
            NamedParam("copy_from", py_rng.choice([True, False])),
            NamedParam("blocking", py_rng.choice([True, False])),
        )
        for base_src_shape in [
            tuple(),
            (11,),
            (16,),
            (16, 12),
            (1, 16, 1, 12, 1),
            (13, 12, 8),
            (4, 13, 15),
        ]
        for direction in list(CopyDirection)
        for src_format in list(SrcFormat)
        for transpose in list(Transpose)
        for dst_format in list(DstFormat)
        for implicit_c in [True, False]
        if not implicit_c or src_format == SrcFormat.C_CONTIGUOUS
        for broadcast in [py_rng.choice([Broadcast.NO, py_rng.choice(list(Broadcast)[1:])])]
        for src_shape in [get_src_shape(base_src_shape, broadcast)]
        for dst_shape in [get_dst_shape(py_rng, base_src_shape, broadcast)]
        if src_format != SrcFormat.SLICED or len(src_shape) > 0
        if transpose == Transpose.SAME_ORDER or len(dst_shape) > 1
        if dst_format != DstFormat.SLICED or len(dst_shape) > 0
    ],
    ids=pretty_name,
)
def test_strided_memory_view_copy(
    src_shape,
    dst_shape,
    direction,
    src_format,
    transpose,
    dst_format,
    implicit_c,
    broadcast,
    itemsize,
    device_id,
    default_stream,
    src_order,
    dst_order,
    copy_from,
    blocking,
):
    device_id = device_id.value
    src_shape = src_shape.value
    dst_shape = dst_shape.value
    implicit_c = implicit_c.value
    itemsize = itemsize.value
    default_stream = default_stream.value
    src_order = src_order.value
    dst_order = dst_order.value
    copy_from = copy_from.value
    blocking = blocking.value

    if device_id is None:
        pytest.skip("No devices available")
    if cp is None:
        pytest.skip("cupy is not installed")

    assert isinstance(direction, CopyDirection)
    is_src_host = direction == CopyDirection.H2D
    is_dst_host = direction == CopyDirection.D2H

    cp_stream = None
    stream = None
    try:
        with cp.cuda.Device(device_id):
            if default_stream:
                stream = ccx.Device(device_id).default_stream
                cp_stream = cp.cuda.ExternalStream(int(stream.handle), device_id)
            else:
                cp_stream = cp.cuda.Stream(non_blocking=True)
                stream = ccx.Stream.from_handle(cp_stream.ptr)

            with cp_stream:
                src = get_src(is_src_host, src_format, src_order, implicit_c, src_shape, itemsize)
                dst = get_dst(
                    is_dst_host,
                    src_order,
                    transpose,
                    dst_order,
                    implicit_c,
                    dst_format,
                    dst_shape,
                    itemsize,
                )

        if not is_src_host:
            assert src.device_id == device_id
            assert src.is_device_accessible
        if not is_dst_host:
            assert dst.is_device_accessible
            assert dst.device_id == device_id

        if broadcast != Broadcast.NO:
            assert src.shape != dst.shape
        else:
            assert src.shape == dst.shape

        with with_ccx_device(device_id), with_logger() as log_stream:
            if copy_from:
                dst.copy_from(src, stream, blocking=blocking)
            else:
                src.copy_to(dst, stream, blocking=blocking)
            debug_log = log_stream.getvalue()

        if blocking or is_h2h_needed(direction, src_shape, src_format, dst_shape, dst_format, broadcast):
            assert f"Syncing stream {int(stream.handle)}" in debug_log
        else:
            # if no extra H2H is needed we should respect non-blocking flag
            assert "Syncing stream" not in debug_log

        src_array = as_array(None if is_src_host else device_id, src)
        dst_array = as_array(None if is_dst_host else device_id, dst)
        assert src_array.shape == src.shape == src_shape
        assert dst_array.shape == dst.shape == dst_shape
        if src.layout.strides_in_bytes is None:
            dense_strides = StridedLayout.dense(src.shape, src.layout.itemsize).strides_in_bytes
            assert src_array.strides == dense_strides
        else:
            assert src_array.strides == src.layout.strides_in_bytes
        if dst.layout.strides_in_bytes is None:
            dense_strides = StridedLayout.dense(dst.shape, dst.layout.itemsize).strides_in_bytes
            assert dst_array.strides == dense_strides
        else:
            assert dst_array.strides == dst.layout.strides_in_bytes

        with cp.cuda.Device(device_id):
            if not blocking:
                stream.sync()
            if not is_src_host:
                src_array = cp.asnumpy(src_array)
            if not is_dst_host:
                dst_array = cp.asnumpy(dst_array)

        if broadcast != Broadcast.NO:
            src_array = np.broadcast_to(src_array, dst_array.shape)
        np.testing.assert_equal(src_array, dst_array)

    finally:
        if not default_stream and stream is not None:
            stream.close()


class CustomDeviceAllocator(ccx.MemoryResource):
    def __init__(self, device: ccx.Device):
        self.device = device
        self._mr = device.memory_resource
        self._recorded = []

    def allocate(self, size, stream=None):
        self._recorded.append((size, stream))
        return self._mr.allocate(size, stream)

    def deallocate(self, ptr, size, stream=None):
        self._mr.deallocate(ptr, size, stream)

    @property
    def is_device_accessible(self) -> bool:
        return True

    @property
    def is_host_accessible(self) -> bool:
        return False

    @property
    def device_id(self) -> int:
        return self.device.device_id


class CustomHostAllocator(ccx.LegacyPinnedMemoryResource):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._recorded = []

    def allocate(self, size, stream=None):
        self._recorded.append((size, stream))
        return super().allocate(size, stream)


@pytest.mark.parametrize(
    ("direction", "custom_allocator", "use_dataclass", "dtype", "device_id"),
    [
        (
            direction,
            custom_allocator,
            use_dataclass,
            py_rng.choice([np.float16, np.float32, np.float64, np.complex64]),
            NamedParam(
                "device_id",
                py_rng.randint(0, num_devices - 1) if num_devices >= 0 else None,
            ),
        )
        for direction in [CopyDirection.H2D, CopyDirection.D2H]
        for custom_allocator in [
            CustomAllocator.HOST,
            CustomAllocator.DEVICE,
            CustomAllocator.BOTH,
        ]
        for use_dataclass in [True, False]
    ],
    ids=pretty_name,
)
def test_custom_allocator(direction, custom_allocator, use_dataclass, dtype, device_id):
    if cp is None:
        pytest.skip("cupy is not installed")

    device_id = device_id.value

    assert isinstance(direction, CopyDirection)
    is_src_host = direction == CopyDirection.H2D
    is_dst_host = direction == CopyDirection.D2H
    src_mod = np if is_src_host else cp
    dst_mod = np if is_dst_host else cp

    shape = (111, 122)
    with cp.cuda.Device(device_id):
        a = src_mod.arange(np.prod(shape), dtype=dtype).reshape(shape, order="C")
        a = a[:, ::-2]
        b = dst_mod.arange(np.prod(shape), dtype=dtype).reshape(shape, order="F")
        b = b[::-1, : shape[1] // 2]

    nbytes = a.nbytes
    src_buf = Buffer.from_handle(get_ptr(a), nbytes, owner=a)
    dst_buf = Buffer.from_handle(get_ptr(b), nbytes, owner=b)
    src_layout = StridedLayout(a.shape, a.strides, a.itemsize, divide_strides=True)
    dst_layout = StridedLayout(b.shape, b.strides, b.itemsize, divide_strides=True)
    src = StridedMemoryView.from_buffer(src_buf, src_layout)
    dst = StridedMemoryView.from_buffer(dst_buf, dst_layout)

    host_allocator = None
    device_allocator = None
    if custom_allocator in [CustomAllocator.HOST, CustomAllocator.BOTH]:
        host_allocator = CustomHostAllocator()
    if custom_allocator in [CustomAllocator.DEVICE, CustomAllocator.BOTH]:
        device_allocator = CustomDeviceAllocator(ccx.Device(device_id))

    if use_dataclass:
        allocator = CopyAllocatorOptions(host=host_allocator, device=device_allocator)
    else:
        allocator = {k: v for k, v in [("host", host_allocator), ("device", device_allocator)] if v is not None}

    with with_ccx_device(device_id) as dev:
        stream = dev.default_stream
        dst.copy_from(src, stream, blocking=True, allocator=allocator)

    if host_allocator is not None:
        assert host_allocator._recorded == [(nbytes, stream)]
    if device_allocator is not None:
        assert device_allocator._recorded == [(nbytes, stream)]

    src_array = as_array(None if is_src_host else device_id, src)
    dst_array = as_array(None if is_dst_host else device_id, dst)
    with cp.cuda.Device(device_id):
        if not is_src_host:
            src_array = cp.asnumpy(src_array)
        if not is_dst_host:
            dst_array = cp.asnumpy(dst_array)
    np.testing.assert_equal(src_array, dst_array)


def test_wrong_shape():
    a = np.arange(10).reshape((2, 5))
    d = ccx.Device(0)
    d.set_current()
    layout = StridedLayout.dense((2, 6), a.itemsize)
    buf = d.memory_resource.allocate(layout.required_size_in_bytes())
    a_view = StridedMemoryView.from_dlpack(a, -1)
    b_view = StridedMemoryView.from_buffer(buf, layout)
    with pytest.raises(ValueError, match="cannot be broadcast together"):
        b_view.copy_from(a_view, d.default_stream, blocking=True)
    with pytest.raises(ValueError, match="cannot be broadcast together"):
        a_view.copy_from(b_view, d.default_stream, blocking=True)
    with pytest.raises(ValueError, match="cannot be broadcast together"):
        a_view.copy_to(b_view, d.default_stream, blocking=True)
    with pytest.raises(ValueError, match="cannot be broadcast together"):
        b_view.copy_to(a_view, d.default_stream, blocking=True)


def test_wrong_dtype():
    a = np.arange(10, dtype=np.int32).reshape((2, 5))
    d = ccx.Device(0)
    d.set_current()
    layout = StridedLayout.dense((2, 5), a.itemsize)
    buf = d.memory_resource.allocate(layout.required_size_in_bytes())
    a_view = StridedMemoryView.from_dlpack(a, -1)
    b_view = StridedMemoryView.from_buffer(buf, layout, dtype=np.float32)
    with pytest.raises(ValueError, match="destination and source dtypes"):
        b_view.copy_from(a_view, d.default_stream, blocking=True)
    with pytest.raises(ValueError, match="destination and source dtypes"):
        a_view.copy_to(b_view, d.default_stream, blocking=True)


def test_wrong_itemsize():
    a = np.arange(10, dtype=np.int32).reshape((2, 5))
    d = ccx.Device(0)
    d.set_current()
    layout = StridedLayout.dense((2, 5), 8)
    buf = d.memory_resource.allocate(layout.required_size_in_bytes())
    a_view = StridedMemoryView.from_dlpack(a, -1)
    b_view = StridedMemoryView.from_buffer(buf, layout)
    with pytest.raises(ValueError, match="itemsize"):
        b_view.copy_from(a_view, d.default_stream, blocking=True)
    with pytest.raises(ValueError, match="itemsize"):
        a_view.copy_to(b_view, d.default_stream, blocking=True)


def test_overlapping_dst():
    a = np.arange(10)
    a = np.lib.stride_tricks.sliding_window_view(a, 3, -1)
    # do this manually, as through dlpack numpy marks the view readonly
    buf = Buffer.from_handle(a.ctypes.data, a.nbytes, owner=a)
    layout = StridedLayout(a.shape, a.strides, a.itemsize, divide_strides=True)
    host_overlapping_view = StridedMemoryView.from_buffer(buf, layout)
    layout = host_overlapping_view.layout
    d = ccx.Device(0)
    d.set_current()
    buf = d.memory_resource.allocate(10 * a.itemsize)
    dev_overlapping_view = StridedMemoryView.from_buffer(buf, host_overlapping_view.layout)
    dense_layout = layout.to_dense()
    dense_buf = d.memory_resource.allocate(dense_layout.required_size_in_bytes())
    dense_view = StridedMemoryView.from_buffer(dense_buf, dense_layout)
    with pytest.raises(ValueError, match="destination layout is non-unique"):
        dense_view.copy_to(host_overlapping_view, d.default_stream, blocking=True)
    with pytest.raises(ValueError, match="destination layout is non-unique"):
        dense_view.copy_to(dev_overlapping_view, d.default_stream, blocking=True)
