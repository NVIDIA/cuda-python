# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0


import ctypes
import random
from enum import Enum

import cuda.core.experimental as ccx
import numpy as np
import pytest
from cuda.core.experimental import (
    system as ccx_system,
)
from cuda.core.experimental.utils import StridedLayout, StridedMemoryView

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
    BROADCAST = "broadcast"  # the src is broadcasted


class Transpose(Enum):
    SAME_ORDER = False  # the src and dst have the same stride order
    INVERSE = "inverse"  # the src and dst have the inverse stride order (e.g. F <-> C)
    PERMUTATION = "permutation"  # the src and dst have permuted stride orders


class DstFormat(Enum):
    DENSE = "dense"  # the dst is contigious in some order
    SLICED = "sliced"  # the dst is sliced (has gaps)


_ITEMSIZES = [1, 2, 4, 8, 16]

py_rng = random.Random(43)
num_devices = ccx_system.num_devices


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


def get_dst_shape(rng, src_format, src_shape):
    assert isinstance(src_format, SrcFormat)
    if src_format == SrcFormat.BROADCAST:
        # enforce implicit broadcasting when the shapes differ
        return (rng.randint(3, 5),) + src_shape
    return src_shape


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
    if device_id is None:
        c_mem = memoryview((ctypes.c_char * size).from_address(smv.ptr))
        np_array = np.frombuffer(c_mem, dtype=smv.dtype)
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
            dtype=smv.dtype,
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
        for src_shape in [
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
        if src_format != SrcFormat.SLICED or len(src_shape) > 0
        for dst_shape in [get_dst_shape(py_rng, src_format, src_shape)]
        for transpose in list(Transpose)
        if transpose == Transpose.SAME_ORDER or len(dst_shape) > 1
        for dst_format in list(DstFormat)
        if dst_format != DstFormat.SLICED or len(dst_shape) > 0
        for implicit_c in [True, False]
        if not implicit_c or src_format == SrcFormat.C_CONTIGUOUS
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
                cp_stream = cp.cuda.Stream(stream)
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

        if src_format == SrcFormat.BROADCAST:
            assert src.shape != dst.shape
        else:
            assert src.shape == dst.shape

        # default stream does not carry device information
        current_dev = ccx.Device()
        current_dev_id = current_dev.device_id
        dev = ccx.Device(device_id)
        try:
            dev.set_current()
            if copy_from:
                dst.copy_from(src, stream, blocking=blocking)
            else:
                src.copy_to(dst, stream, blocking=blocking)
        finally:
            if current_dev_id != device_id:
                current_dev.set_current()

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

        if src_format == SrcFormat.BROADCAST:
            src_array = np.broadcast_to(src_array, dst_array.shape)
        np.testing.assert_equal(src_array, dst_array)

    finally:
        if not default_stream and stream is not None:
            stream.close()
