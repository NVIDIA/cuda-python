# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0


cimport cython

from libc.stdint cimport intptr_t, int64_t

from cuda.core.experimental._layout cimport axis_vec_t, ORDER_C
from cuda.core.experimental._strided_copy._copy_utils cimport (
    logging_axis_order, logging_memcopy, logging_py_objs, memcpy_async,
    volume_in_bytes, maybe_sync, get_data_ptr,
    flatten_together, maybe_broadcast_src,
    check_itemsize, _view_as_numpy, _view_as_strided, _np_dtype
)
from cuda.core.experimental._strided_copy._cuda_kernel cimport cuda_kernel_copy

import contextlib
import threading
from dataclasses import dataclass
from cuda.core.experimental._device import Device
from cuda.core.experimental._memory import MemoryResource, DeviceMemoryResource
import numpy as _numpy


_thread_local = threading.local()
_current_logger = None


@contextlib.contextmanager
def _with_logger(logger):
    # Utility meant for debuggin and testing purposes.
    global _current_logger
    _current_logger = logger
    yield
    _current_logger = None


@dataclass
class CopyAllocatorOptions:
    host : MemoryResource | None = None
    device : DeviceMemoryResource | None = None


cdef inline alocator_options(allocator : CopyAllocatorOptions | dict[str, MemoryResource] | None):
    if allocator is None or type(allocator) is CopyAllocatorOptions:
        return allocator
    return CopyAllocatorOptions(**allocator)


cdef inline object _numpy_empty(
    Buffer host_alloc,
    StridedLayout layout,
    object logger,
):
    """
    The layout must be contigious in some order (layout.get_is_contiguous_any() is True)
    If host_alloc is not None, returns a numpy array being a view on the host_alloc with
    shape, itemsize and strides as in the layout.
    Otherwise, returns a new numpy array with shape, itemsize and strides as in the layout.
    """
    if host_alloc is not None:
        return _view_as_numpy(int(host_alloc.handle), volume_in_bytes(layout), layout)
    cdef object a =_numpy.empty(layout.get_volume(), dtype=_np_dtype(layout.itemsize))
    return _view_as_strided(a, layout)


cdef inline object _numpy_ascontiguousarray(
    Buffer host_alloc,
    intptr_t data_ptr,
    int64_t size,
    StridedLayout layout,
    object logger,
):
    """
    Returns a numpy array with the same shape and itemsize as the layout,
    but C-contiguous strides. The data_ptr must be a valid host pointer to
    a tensor described with the layout.
    The layout is modified in place so that it is C-contigious and dense.
    If host_alloc is provied, copies the data there and returns a view on it.
    Otherwise, returns a new numpy array.
    """
    cdef object a = _view_as_numpy(data_ptr, size, layout)
    if logger is not None:
        logger.debug(
            f"({layout}) is not contiguous, coalescing H2H copy is needed."
        )
    layout.make_dense(ORDER_C, NULL)
    if host_alloc is None:
        return _numpy.ascontiguousarray(a)
    cdef object b = _numpy_empty(host_alloc, layout, logger)
    _numpy.copyto(b, a)
    return b


cdef inline get_device_default_mr(int device_id):
    if not hasattr(_thread_local, "device_default_mrs"):
        _thread_local.device_default_mrs = {}
    cdef dict device_default_mrs = _thread_local.device_default_mrs
    cdef mr = device_default_mrs.get(device_id)
    if mr is not None:
        return mr

    # We're accessing the device's default mr for the first time
    # We need to make sure the device's context has ever been set
    cdef object current_dev = Device()
    cdef int current_dev_id = current_dev.device_id
    cdef object dev = Device(device_id)
    try:
        dev.set_current()
        mr = dev.memory_resource
        device_default_mrs[device_id] = mr
        return mr
    finally:
        if current_dev_id != device_id:
            current_dev.set_current()


cdef inline Buffer _device_allocate(
    device_allocator : DeviceMemoryResource | None,
    int64_t size,
    int device_id,
    Stream stream,
):
    if device_allocator is None:
        device_allocator = get_device_default_mr(device_id)
    return device_allocator.allocate(size, stream)


cdef inline int _copy_into_d2d(
    intptr_t dst_ptr,
    StridedLayout dst_layout,
    intptr_t src_ptr,
    StridedLayout src_layout,
    int device_id,
    intptr_t stream_ptr,
    bint blocking,
    object logger,
) except -1 nogil:
    # Note: this function assumes that layouts were squeezed
    # already and can be modified in place (i.e. they are not
    # referenced elsewhere, e.g. by StridedMemoryView).

    cdef int64_t size = volume_in_bytes(dst_layout)
    if size == 0:
        return 0

    # Normalize the layouts:
    # 1. permute the layouts so that the dst layouts order is C-like.
    # 2. remove all extents equal to 1, as their strides are irrelevant
    # 3. flatten extents that are mergable in both layouts
    cdef axis_vec_t axis_order
    dst_layout.get_stride_order(axis_order)
    dst_layout.permute_into(dst_layout, axis_order)
    src_layout.permute_into(src_layout, axis_order)
    flatten_together(dst_layout, src_layout)
    if logger is not None:
        logging_axis_order("The dst_order is {fst}", logger, axis_order)
        logging_py_objs("Normalized layouts: dst {fst}, src {snd}", logger, dst_layout, src_layout)

    # Can we just memcpy?
    # We permuted the extents to C-order, so if layouts are dense, they are C-contiguous.
    # The precondition is that the shapes are equal, thus the c-contiguity implies equal strides.
    if dst_layout.get_is_contiguous_c() and src_layout.get_is_contiguous_c():
        if logger is not None:
            logging_memcopy(
                "The layouts are contiguous and have the same stride order, we can memcopy and return.\n{msg}",
                logger, "D2D",
                dst_ptr, dst_layout,
                src_ptr, src_layout,
                blocking, size, stream_ptr
            )
        memcpy_async(dst_ptr, src_ptr, size, stream_ptr)
        maybe_sync(stream_ptr, blocking, logger)
        return 0

    if not dst_layout.get_is_unique():
        raise ValueError(
            f"The destination layout is non-unique, i.e. some elements "
            f"may overlap in memory: {dst_layout}"
        )
    cuda_kernel_copy(dst_ptr, dst_layout, src_ptr, src_layout, device_id, stream_ptr, logger)
    maybe_sync(stream_ptr, blocking, logger)
    return 0


cdef inline int _copy_into_h2d(
    intptr_t dst_data_ptr,
    StridedLayout dst_layout,
    intptr_t src_data_ptr,
    StridedLayout src_layout,
    int device_id,
    Stream stream,
    bint blocking,
    host_allocator : MemoryResource | None,
    device_allocator : DeviceMemoryResource | None,
    object logger,
) except -1:
    """
    Copies data from host to device, rearranging the data if needed.
    In a simplest case of contiguous layouts with matching stride orders,
    only H2D memcpy is needed.
    Otherwise, up to 2 extra copies (H2H and D2D) may be needed
    for 3 different reasons:
    * if src is non-contigious we need a H2H 'coalescing' copy
    * if dst is non-contigious we need a D2D 'scattering' copy
    * if the dst and src's stride orders differ, we need
      a either H2H or D2D copy to transpose the data.
    Moving data around in the device memory should be faster,
    so we prefer D2D for transpose-copy, unless coalescing
    H2H is needed anyway while D2D for scattering can be avoided.
    """

    # Note, this function assumes that layouts were squeezed
    # already and can be modified in place (i.e. they are not
    # referenced elsewhere, e.g. by StridedMemoryView).

    cdef intptr_t stream_ptr = int(stream.handle)
    cdef int64_t size = volume_in_bytes(dst_layout)
    if size == 0:
        return 0

    # Permute the layouts so that src has C-like strides
    # (increasing from the right to the left)
    cdef axis_vec_t axis_order
    src_layout.get_stride_order(axis_order)
    dst_layout.permute_into(dst_layout, axis_order)
    src_layout.permute_into(src_layout, axis_order)

    # First, make sure the src is contiguous, so that we can memcpy from it.
    cdef Buffer host_alloc = None
    if host_allocator is not None:
        host_alloc = host_allocator.allocate(size, stream)
    # A numpy array (either view on host_alloc or a new regular numpy array)
    cdef object src_tmp = None
    if not src_layout.get_is_contiguous_c():
        if dst_layout.get_is_contiguous_any():
            # We cannot avoid H2H copy, but we can avoid D2D copy
            # if we also make sure that the src and dst stride orders match.
            # We permute layouts to dst order, so that dst is C-contig
            # and the src will be made C-contig by the H2H copy.
            dst_layout.get_stride_order(axis_order)
            dst_layout.permute_into(dst_layout, axis_order)
            src_layout.permute_into(src_layout, axis_order)
        # the host allocation may not be stream-aware (in particular, the default numpy is not!)
        # so we need to block at least until the H2D is complete to avoid deallocating too early.
        blocking = True
        src_tmp = _numpy_ascontiguousarray(host_alloc, src_data_ptr, size, src_layout, logger)
        src_data_ptr = src_tmp.ctypes.data

    if dst_layout.get_is_contiguous_c():
        # We made sure that src layout is C-contig too, we can just memcopy.
        if logger is not None:
            logging_memcopy(
                "The layouts are contiguous and have the same stride order, we can memcopy and return.\n{msg}",
                logger, "H2D",
                dst_data_ptr, dst_layout,
                src_data_ptr, src_layout,
                blocking, size, stream_ptr
            )
        with cython.nogil:
            memcpy_async(dst_data_ptr, src_data_ptr, size, stream_ptr)
            maybe_sync(stream_ptr, blocking, logger)
        return 0

    # Otherwise, either dst is not contigious or src has a different stride order than dst.
    # In either case, src is contigious is some order, so we can just memcopy
    # it to a temporary buffer and then perform a D2D transpose/scatter copy to dst.
    cdef Buffer dev_tmp = _device_allocate(device_allocator, size, device_id, stream)
    cdef intptr_t dev_tmp_data_ptr = int(dev_tmp.handle)
    if logger is not None:
        logging_memcopy(
            f"First, memcpy into a temporary device buffer:\n{{msg}}\n"
            f"Followed by scatter/transpose D2D copy:\n"
            f"({dst_data_ptr}, {dst_layout} <- {dev_tmp_data_ptr}, {src_layout})",
            logger, "H2D",
            dev_tmp_data_ptr, src_layout,
            src_data_ptr, src_layout,
            blocking, size, stream_ptr
        )
    with cython.nogil:
        memcpy_async(dev_tmp_data_ptr, src_data_ptr, size, stream_ptr)
        _copy_into_d2d(
            dst_data_ptr, dst_layout,
            dev_tmp_data_ptr, src_layout,
            device_id, stream_ptr,
            blocking, logger
        )
    return 0


cdef inline int _copy_into_d2h(
    intptr_t dst_data_ptr,
    StridedLayout dst_layout,
    intptr_t src_data_ptr,
    StridedLayout src_layout,
    int device_id,
    Stream stream,
    bint blocking,
    host_allocator : MemoryResource | None,
    device_allocator : DeviceMemoryResource | None,
    object logger,
) except -1:
    """
    Copies data from device to host, rearranging the data if needed.
    In a simplest case of contiguous layouts with matching stride orders,
    only D2H memcpy is needed.
    Otherwise,up to 2 extra copies (D2D and H2H) may be needed:
    * if src is non-contigious we need a D2D 'coalescing' copy
    * if dst is non-contigious we need a H2H 'scattering' copy
    * if the dst and src's stride orders differ, we need
      a either D2D or H2H copy to transpose the data.
    Moving data around in the device memory should be faster,
    so we prefer D2D for transpose-copy, unless H2H is needed anyway
    for scattering while D2D can be avoided.
    """

    # Note, this function assumes that layouts were squeezed
    # already and can be modified in place (i.e. they are not
    # referenced elsewhere, e.g. by StridedMemoryView).

    cdef intptr_t stream_ptr = int(stream.handle)
    cdef int64_t size = volume_in_bytes(dst_layout)
    if size == 0:
        return 0

    if not dst_layout.get_is_unique():
        raise ValueError(
            f"The destination layout is non-unique, i.e. some elements "
            f"may overlap in memory: {dst_layout}"
        )

    # Permute the layouts so that dst has C-like strides
    # (increasing from the right to the left)
    cdef axis_vec_t axis_order
    dst_layout.get_stride_order(axis_order)
    dst_layout.permute_into(dst_layout, axis_order)
    src_layout.permute_into(src_layout, axis_order)

    # If, after the permutation to C-like order, dst is still not C-contig,
    # we need to H2H scatter the data.
    cdef bint is_dst_contig = dst_layout.get_is_contiguous_c()

    cdef Buffer src_tmp = None
    cdef intptr_t src_tmp_data_ptr
    cdef StridedLayout src_tmp_layout
    if (
        # if dst does not require scattering H2H copy,
        # run D2D whenever src needs transposing or coalescing
        (is_dst_contig and not src_layout.get_is_contiguous_c())
        # otherwise, as H2H is needed anyway,
        # run D2D only if src requires coalescing
        or not src_layout.get_is_contiguous_any()
    ):
        # After the copy, the src will be coalesced and
        # transposed to dst order.
        src_tmp_layout = src_layout.to_dense("C")
        src_tmp = _device_allocate(device_allocator, size, device_id, stream)
        src_tmp_data_ptr = int(src_tmp.handle)
        if logger is not None:
            logger.debug(
                f"We need to coalesce or transpose the data, "
                f"running D2D copy into a temporary device buffer:\n"
                f"({src_tmp_data_ptr}, {src_tmp_layout} <- {src_data_ptr}, {src_layout})"
            )
        _copy_into_d2d(
            src_tmp_data_ptr,
            # pass a copy of the dense layout, we'll
            # need it later and d2d can modify it in place
            src_tmp_layout.copy(),
            src_data_ptr, src_layout,
            device_id, stream_ptr,
            False, logger
        )
        src_data_ptr = src_tmp_data_ptr
        src_layout = src_tmp_layout

    if is_dst_contig:
        # The dst is c-contig. If we run a D2D copy, it made src C-contig too.
        # If we didn't run a D2D copy, src must have been C-contig already.
        if logger is not None:
            logging_memcopy(
                "The layouts are contiguous and have the same stride order, we can memcopy and return.\n{msg}",
                logger, "D2H",
                dst_data_ptr, dst_layout,
                src_data_ptr, src_layout,
                blocking, size, stream_ptr
            )
        with cython.nogil:
            memcpy_async(dst_data_ptr, src_data_ptr, size, stream_ptr)
            maybe_sync(stream_ptr, blocking, logger)
        return 0

    # Otherwise, we need to D2H copy into a temp host buffer and run
    # a H2H copy to scatter or transpose the data.
    cdef Buffer host_alloc = None
    if host_allocator is not None:
        host_alloc = host_allocator.allocate(size, stream)
    # A numpy array (either view on host_alloc or a new regular numpy array)
    cdef object dst_tmp = _numpy_empty(host_alloc, src_layout, logger)
    cdef intptr_t dst_tmp_data_ptr = dst_tmp.ctypes.data
    if logger is not None:
        logging_memcopy(
            f"First memcpy into a temporary host buffer:\n{{msg}}\n"
            f"Followed by scatter/transpose H2H copy:\n"
            f"({dst_data_ptr}, {dst_layout} <- {dst_tmp_data_ptr}, {src_layout})",
            logger, "D2H",
            dst_data_ptr, src_layout,
            src_data_ptr, src_layout,
            blocking, size, stream_ptr
        )
    with cython.nogil:
        memcpy_async(dst_tmp_data_ptr, src_data_ptr, size, stream_ptr)
        maybe_sync(stream_ptr, True, logger)
    _numpy.copyto(_view_as_numpy(dst_data_ptr, size, dst_layout), dst_tmp)
    return 0


cdef int copy_into_d2d(
    Buffer dst_buffer,
    StridedLayout dst_layout,
    Buffer src_buffer,
    StridedLayout src_layout,
    int device_id,
    Stream stream,
    bint blocking,
) except -1:
    check_itemsize(dst_layout, src_layout)
    src_layout = maybe_broadcast_src(dst_layout, src_layout)

    cdef intptr_t stream_ptr = int(stream.handle)
    cdef intptr_t dst_data_ptr = get_data_ptr(dst_buffer, dst_layout)
    cdef intptr_t src_data_ptr = get_data_ptr(src_buffer, src_layout)

    # Get rid of all 1-extents, as their strides are irrelevant.
    # Make sure the copy of the layouts is passed, as the _copy_into_h2d
    # may modify those
    cdef StridedLayout squeezed_dst = StridedLayout.__new__(StridedLayout)
    cdef StridedLayout squeezed_src = StridedLayout.__new__(StridedLayout)
    dst_layout.squeeze_into(squeezed_dst)
    src_layout.squeeze_into(squeezed_src)

    return _copy_into_d2d(
        dst_data_ptr, squeezed_dst,
        src_data_ptr, squeezed_src,
        device_id, stream_ptr,
        blocking, _current_logger
    )


cdef int copy_into_h2d(
    Buffer dst_buffer,
    StridedLayout dst_layout,
    Buffer src_buffer,
    StridedLayout src_layout,
    int device_id,
    Stream stream,
    allocator : CopyAllocatorOptions | dict[str, MemoryResource] | None,
    bint blocking,
) except -1:
    check_itemsize(dst_layout, src_layout)
    src_layout = maybe_broadcast_src(dst_layout, src_layout)

    cdef intptr_t dst_data_ptr = get_data_ptr(dst_buffer, dst_layout)
    cdef intptr_t src_data_ptr = get_data_ptr(src_buffer, src_layout)
    cdef host_allocator = None
    cdef device_allocator = None
    allocator = alocator_options(allocator)
    if allocator is not None:
        host_allocator = allocator.host
        device_allocator = allocator.device

    # Get rid of all 1-extents, as their strides are irrelevant.
    # Make sure the copy of the layouts is passed, as the _copy_into_h2d
    # may modify those
    cdef StridedLayout squeezed_dst = StridedLayout.__new__(StridedLayout)
    cdef StridedLayout squeezed_src = StridedLayout.__new__(StridedLayout)
    dst_layout.squeeze_into(squeezed_dst)
    src_layout.squeeze_into(squeezed_src)

    return _copy_into_h2d(
        dst_data_ptr, squeezed_dst,
        src_data_ptr, squeezed_src,
        device_id, stream,
        blocking, host_allocator, device_allocator,
        _current_logger,
    )


cdef int copy_into_d2h(
    Buffer dst_buffer,
    StridedLayout dst_layout,
    Buffer src_buffer,
    StridedLayout src_layout,
    int device_id,
    Stream stream,
    allocator : CopyAllocatorOptions | dict[str, MemoryResource] | None,
    bint blocking,
) except -1:
    check_itemsize(dst_layout, src_layout)
    src_layout = maybe_broadcast_src(dst_layout, src_layout)

    cdef intptr_t dst_data_ptr = get_data_ptr(dst_buffer, dst_layout)
    cdef intptr_t src_data_ptr = get_data_ptr(src_buffer, src_layout)
    cdef host_allocator = None
    cdef device_allocator = None
    allocator = alocator_options(allocator)
    if allocator is not None:
        host_allocator = allocator.host
        device_allocator = allocator.device

    # Get rid of all 1-extents, as their strides are irrelevant.
    # Make sure the copy of the layouts is passed, as the _copy_into_h2d
    # may modify those
    cdef StridedLayout squeezed_dst = StridedLayout.__new__(StridedLayout)
    cdef StridedLayout squeezed_src = StridedLayout.__new__(StridedLayout)
    dst_layout.squeeze_into(squeezed_dst)
    src_layout.squeeze_into(squeezed_src)

    return _copy_into_d2h(
        dst_data_ptr, squeezed_dst,
        src_data_ptr, squeezed_src,
        device_id, stream,
        blocking,
        host_allocator, device_allocator,
        _current_logger
    )
