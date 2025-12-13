# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from cuda.core.experimental._stream cimport Stream
from cuda.core.experimental._layout cimport StridedLayout
from cuda.core.experimental._memory._buffer cimport Buffer

from cuda.core.experimental._memory import MemoryResource


cdef int copy_into_d2d(
    Buffer dst_buffer,
    StridedLayout dst_layout,
    Buffer src_buffer,
    StridedLayout src_layout,
    int device_id,
    Stream stream,
    bint blocking,
) except -1


cdef int copy_into_d2h(
    Buffer dst_buffer,
    StridedLayout dst_layout,
    Buffer src_buffer,
    StridedLayout src_layout,
    int device_id,
    Stream stream,
    allocator : CopyAllocatorOptions | dict[str, MemoryResource] | None,
    bint blocking,
) except -1


cdef int copy_into_h2d(
    Buffer dst_buffer,
    StridedLayout dst_layout,
    Buffer src_buffer,
    StridedLayout src_layout,
    int device_id,
    Stream stream,
    allocator : CopyAllocatorOptions | dict[str, MemoryResource] | None,
    bint blocking,
) except -1
