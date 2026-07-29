# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from libc.stdint cimport int64_t, uint8_t, uint16_t
from libc.stddef cimport size_t

# Shared with ``_tensor_map.pyx``, which must cimport the typedef only:
# cimporting the getter makes Cython import ``_tensor_map_cccl`` when
# ``_tensor_map`` loads, which is exactly what the capsule lookup avoids.
ctypedef int (*make_tma_descriptor_tiled_t)(
    void* out_tensor_map,
    void* data,
    int device_type,
    int device_id,
    int ndim,
    const int64_t* shape,
    const int64_t* strides,
    uint8_t dtype_code,
    uint8_t dtype_bits,
    uint16_t dtype_lanes,
    const int* box_sizes,
    const int* elem_strides,
    int interleave_layout,
    int swizzle,
    int l2_fetch_size,
    int oob_fill,
    char* err,
    size_t err_cap) noexcept nogil

# Exported via ``__pyx_capi__`` for soft-linking from ``_tensor_map.pyx``.
# Returns the CCCL implementation, or NULL when unavailable at C++ compile time.
cdef make_tma_descriptor_tiled_t get_make_tma_descriptor_tiled() noexcept nogil
