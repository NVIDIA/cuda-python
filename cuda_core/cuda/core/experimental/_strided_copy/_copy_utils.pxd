# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

cimport cython

cimport cpython
from cpython.memoryview cimport PyMemoryView_FromMemory

from libc.stdint cimport intptr_t, int64_t

from cuda.core.experimental._memory._buffer cimport Buffer
from cuda.core.experimental._layout cimport (
    StridedLayout, BaseLayout, axis_vec_t, axes_mask_t,
    init_base_layout, _overflow_checked_mul, base_equal_shapes
)
from cuda.core.experimental._utils.cuda_utils cimport HANDLE_RETURN

from cuda.bindings cimport cydriver
from cuda.bindings.cydriver cimport CUstream, CUdeviceptr

import numpy as _numpy


cdef inline int64_t div_ceil(int64_t a, int64_t b) except? -1 nogil:
    return (a + b - 1) // b


cdef inline intptr_t get_data_ptr(Buffer buffer, StridedLayout layout) except? 0:
    return <intptr_t>(int(buffer.handle)) + layout.get_slice_offset_in_bytes()


cdef inline object _np_dtype(int itemsize):
    if itemsize == 1:
        return _numpy.uint8
    elif itemsize == 2:
        return _numpy.uint16
    elif itemsize == 4:
        return _numpy.uint32
    elif itemsize == 8:
        return _numpy.uint64
    elif itemsize == 16:
        return _numpy.complex128
    else:
        raise ValueError(f"Unsupported itemsize: {itemsize}")


cdef inline _view_as_strided(object array, StridedLayout layout):
    """
    Array must be a 1d numpy array with dtype.itemsize == layout.itemsize.
    """
    cdef tuple strides = layout.get_strides_in_bytes_tuple()
    if strides is None:
        return array.reshape(layout.get_shape_tuple(), order='C')
    else:
        return _numpy.lib.stride_tricks.as_strided(
            array,
            shape=layout.get_shape_tuple(),
            strides=strides
        )


cdef inline object _view_as_numpy(intptr_t data_ptr, int64_t size, StridedLayout layout):
    """
    Note the returned array is non-owning, it's caller responsibility to keep buffer alive.
    """
    cdef object buf = PyMemoryView_FromMemory(<char*>data_ptr, size, cpython.PyBUF_WRITE)
    cdef object array = _numpy.frombuffer(buf, dtype=_np_dtype(layout.itemsize))
    return _view_as_strided(array, layout)


cdef inline int flatten_together(StridedLayout a, StridedLayout b) except -1 nogil:
    cdef axes_mask_t axis_mask = a.get_flattened_axis_mask() & b.get_flattened_axis_mask()
    a.flatten_into(a, axis_mask)
    b.flatten_into(b, axis_mask)
    return 0


cdef inline bint vectorize_together(intptr_t dst_ptr, StridedLayout dst, intptr_t src_ptr, StridedLayout src) except -1 nogil:
    cdef int max_itemsize = 8
    cdef int itemsize = dst.itemsize
    if itemsize >= max_itemsize:
        return False
    cdef int new_itemsize = dst.get_max_compatible_itemsize(max_itemsize, dst_ptr)
    if itemsize >= new_itemsize:
        return False
    new_itemsize = src.get_max_compatible_itemsize(new_itemsize, src_ptr)
    if itemsize >= new_itemsize:
        return False
    dst.pack_into(dst, new_itemsize, 0, keep_dim=False)
    src.pack_into(src, new_itemsize, 0, keep_dim=False)
    return True


cdef inline int64_t volume_in_bytes(StridedLayout layout) except? -1 nogil:
    return _overflow_checked_mul(layout.get_volume(), <int64_t>layout.itemsize)


cdef inline int check_itemsize(StridedLayout dst, StridedLayout src) except -1 nogil:
    if dst.itemsize != src.itemsize:
        raise ValueError(
            f"The itemsize of the destination and source layouts must match. "
            f"Got dst itemsize:{dst.itemsize} and src itemsize:{src.itemsize}"
        )
    return 0


cdef inline StridedLayout maybe_broadcast_src(StridedLayout dst, StridedLayout src):
    if base_equal_shapes(dst.base, src.base):
        return src
    # If the shapes differ, try broadcasting the source layout to the destination layout.
    cdef StridedLayout new_src = StridedLayout.__new__(StridedLayout)
    cdef BaseLayout new_src_base
    cdef int dst_ndim = dst.base.ndim
    init_base_layout(new_src_base, dst_ndim)
    for i in range(dst_ndim):
        new_src_base.shape[i] = dst.base.shape[i]
    src.broadcast_into(new_src, new_src_base)
    return new_src


cdef inline int logging_axis_order(str msg, logger, axis_vec_t& fst) except -1 nogil:
    with cython.gil:
        logger.debug(msg.format(fst=fst))
    return 0


cdef inline int logging_memcopy(
    str msg, object logger, str kind,
    intptr_t dst_ptr, StridedLayout dst,
    intptr_t src_ptr, StridedLayout src,
    bint blocking, int64_t size, intptr_t stream_ptr
) except -1 nogil:
    with cython.gil:
        logger.debug(msg.format(msg=(
            f"Launching {kind} {'blocking' if blocking else 'non-blocking'} memcpy of "
            f"{size} bytes on stream {stream_ptr}.\n"
            f"Dst: {dst_ptr}, {dst} <- src: {src_ptr}, {src}"
        )))
    return 0


cdef inline int logging_py_objs(str msg, logger, fst=None, snd=None, third=None) except -1 nogil:
    with cython.gil:
        logger.debug(msg.format(fst=fst, snd=snd, third=third))
    return 0


cdef inline int memcpy_async(intptr_t dst_ptr, intptr_t src_ptr, size_t size, intptr_t stream_ptr) except -1 nogil:
    HANDLE_RETURN(
        cydriver.cuMemcpyAsync(
            <CUdeviceptr>dst_ptr,
            <CUdeviceptr>src_ptr,
            size,
            <CUstream>stream_ptr
        )
    )
    return 0


cdef inline int maybe_sync(intptr_t stream_ptr, bint blocking, object logger) except -1 nogil:
    if blocking:
        if logger is not None:
            with cython.gil:
                logger.debug(f"Syncing stream {stream_ptr}.")
        HANDLE_RETURN(cydriver.cuStreamSynchronize(<CUstream>stream_ptr))
    return 0
