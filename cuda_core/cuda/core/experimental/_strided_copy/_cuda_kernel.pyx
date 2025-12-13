# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

cimport cython

from libc.stdint cimport intptr_t, int64_t
from libcpp.memory cimport unique_ptr
from libcpp.functional cimport function

from cuda.bindings cimport cydriver
from cuda.core.experimental._utils.cuda_utils cimport HANDLE_RETURN
from cuda.core.experimental._layout cimport StridedLayout, axes_mask_t, flatten_all_axes_mask, get_strides_ptr
from cuda.core.experimental._strided_copy._jit cimport get_kernel
from cuda.core.experimental._strided_copy._copy_utils cimport logging_py_objs, div_ceil, vectorize_together

ctypedef unique_ptr[void, function[void(void*)]] opaque_args_t


cdef extern from "limits.h":
    cdef int INT_MAX
    cdef int INT_MIN


cdef extern from "include/strided_copy_utils.hpp":
    void _get_strided_copy_args(
        opaque_args_t& args,
        void *dst_ptr, const void *src_ptr,
        int dst_ndim, int src_ndim,
        int64_t* dst_shape, int64_t* src_shape,
        int64_t* dst_strides, int64_t* src_strides,
        int64_t grid_arg) except + nogil


cdef inline int get_kernel_args(
    opaque_args_t& args,
    intptr_t dst_ptr, StridedLayout dst,
    intptr_t src_ptr, StridedLayout src,
    int64_t grid_arg
) except-1 nogil:
    _get_strided_copy_args(
        args,
        <void*>dst_ptr, <const void*>src_ptr,
        dst.base.ndim, src.base.ndim,
        dst.base.shape, src.base.shape,
        get_strides_ptr(dst.base), get_strides_ptr(src.base),
        grid_arg
    )
    return 0


cdef inline bint needs_wide_strides(int64_t grid_volume, StridedLayout dst, StridedLayout src) except?-1 nogil:
    # grid_volume, i.e the block_size * num_blocks
    if grid_volume > INT_MAX:
        return True
    cdef int64_t dst_min_offset = 0
    cdef int64_t dst_max_offset = 0
    cdef int64_t src_min_offset = 0
    cdef int64_t src_max_offset = 0
    dst.get_offset_bounds(dst_min_offset, dst_max_offset)
    src.get_offset_bounds(src_min_offset, src_max_offset)
    cdef int64_t min_offset = min(dst_min_offset, src_min_offset)
    cdef int64_t max_offset = max(dst_max_offset, src_max_offset)
    # forbid INT_MIN too so that:
    # 1. abs() is safe
    # 2. the INT_MIN can be used as special-value/sentinel in the kernel
    return min_offset <= INT_MIN or max_offset > INT_MAX


cdef str emit_elementwise_kernel_code(StridedLayout dst, StridedLayout src, bint has_wide_strides, bint has_grid_stride_loop):
    cdef str stride_t_str = "int64_t" if has_wide_strides else "int32_t"
    cdef str has_grid_stride_loop_str = "true" if has_grid_stride_loop else "false"
    kernel_code = f"""
    #include "elementwise.h"
    ELEMENTWISE_KERNEL({stride_t_str}, {dst.base.ndim}, {src.base.ndim}, {dst.itemsize}, {has_grid_stride_loop_str})
    """
    return kernel_code


cdef inline intptr_t get_elementwise_copy_kernel(
    StridedLayout dst, StridedLayout src,
    bint has_wide_strides, bint has_grid_stride_loop,
    int device_id, object logger
) except? 0:
    cdef str kernel_code = emit_elementwise_kernel_code(dst, src, has_wide_strides, has_grid_stride_loop)
    cdef intptr_t kernel_ptr = get_kernel(kernel_code, device_id, logger)
    return kernel_ptr


cdef inline int adjust_layouts_for_elementwise_copy(StridedLayout dst, StridedLayout src, object logger) except -1 nogil:
    # We want the layouts to keep the same shapes, so that, in cuda kernel,
    # we have to unravel flat element index only once.
    # The exception is if one of the layouts is flattened to 1D,
    # as those don't require unraveling.
    cdef int ndim = dst.base.ndim
    if ndim == 1:
        return 0
    cdef axes_mask_t all_extents = flatten_all_axes_mask(ndim)
    cdef axes_mask_t dst_mask = dst.get_flattened_axis_mask()
    cdef axes_mask_t src_mask = src.get_flattened_axis_mask()
    if dst_mask == all_extents or src_mask == all_extents:
        if dst_mask == all_extents:
            dst.flatten_into(dst, dst_mask)
        if src_mask == all_extents:
            src.flatten_into(src, src_mask)
        if logger is not None:
            logging_py_objs("At least one of the layouts is flattened to 1D: dst {fst}, src {snd}", logger, dst, src)
    return 0


cdef int launch_elementwise_copy(
    intptr_t dst_ptr, StridedLayout dst,
    intptr_t src_ptr, StridedLayout src,
    int block_size, int device_id,
    intptr_t stream_ptr, object logger
) except -1 nogil:
    cdef int64_t volume = dst.get_volume()
    cdef int64_t num_logical_blocks = div_ceil(volume, block_size)
    cdef int64_t cuda_num_blocks = min(num_logical_blocks, INT_MAX)
    cdef bint has_grid_stride_loop = cuda_num_blocks != num_logical_blocks
    cdef bint has_wide_strides = needs_wide_strides(num_logical_blocks * block_size, dst, src)
    cdef opaque_args_t args
    get_kernel_args(args, dst_ptr, dst, src_ptr, src, volume)
    cdef void* args_ptr = args.get()
    cdef intptr_t kernel_fn_ptr
    with cython.gil:
        kernel_fn_ptr = get_elementwise_copy_kernel(dst, src, has_wide_strides, has_grid_stride_loop, device_id, logger)
        if logger is not None:
            logger.debug(
                f"Launching elementwise copy kernel {kernel_fn_ptr} "
                f"with grid.x={cuda_num_blocks}, block.x={block_size}."
            )
    HANDLE_RETURN(cydriver.cuLaunchKernel(
        <cydriver.CUfunction>kernel_fn_ptr,
        cuda_num_blocks, 1, 1,
        block_size, 1, 1,
        0,  # shared_mem_size
        <cydriver.CUstream>stream_ptr,
        &args_ptr,
        NULL
    ))
    return 0


cdef inline int elementwise_copy(
    intptr_t dst_ptr, StridedLayout dst,
    intptr_t src_ptr, StridedLayout src,
    int device_id, intptr_t stream_ptr, object logger
) except -1 nogil:
    cdef int block_size = 128
    adjust_layouts_for_elementwise_copy(dst, src, logger)
    launch_elementwise_copy(dst_ptr, dst, src_ptr, src, block_size, device_id, stream_ptr, logger)
    return 0


cdef int cuda_kernel_copy(
    intptr_t dst_ptr,
    StridedLayout dst,
    intptr_t src_ptr,
    StridedLayout src,
    int device_id,
    intptr_t stream_ptr,
    object logger,
) except -1 nogil:
    # the dst and layouts must be already validated and normalized, i.e.:
    # * the shapes must be equal
    # * the dst stride order must be C-like
    # * implicit C-strides are not allowed (i.e. the strides must not be NULL)
    # * the volume should be >= 2
    # * there should not be any extents equal to 1
    # * the layouts should be flattened together
    if vectorize_together(dst_ptr, dst, src_ptr, src) and logger is not None:
        logging_py_objs("Vectorized the layouts: dst {fst}, src {snd}", logger, dst, src)
    elementwise_copy(dst_ptr, dst, src_ptr, src, device_id, stream_ptr, logger)
    return 0
