# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from libc.stdint cimport intptr_t

from cuda.core._dlpack cimport DLTensor
from cuda.core._layout cimport _StridedLayout


cdef class StridedMemoryView:
    cdef readonly:
        intptr_t ptr
        int device_id
        bint is_device_accessible
        bint readonly
        object exporting_obj

    cdef:
        object metadata
        DLTensor* dl_tensor
        _StridedLayout _layout
        object _buffer
        object _dtype

    cdef inline _StridedLayout get_layout(self)
    cdef inline object get_buffer(self)
    cdef inline object get_dtype(self)
