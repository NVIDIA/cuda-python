# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

cimport cython

from libc.stdint cimport intptr_t

from cuda.core.experimental._layout cimport StridedLayout


cdef int cuda_kernel_copy(
    intptr_t dst_ptr,
    StridedLayout dst,
    intptr_t src_ptr,
    StridedLayout src,
    int device_id,
    intptr_t stream_ptr,
    object logger,
) except -1 nogil
