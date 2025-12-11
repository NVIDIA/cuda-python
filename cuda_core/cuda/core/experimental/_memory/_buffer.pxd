# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from libc.stdint cimport uintptr_t

from cuda.core.experimental._resource_handles cimport DevicePtrHandle
from cuda.core.experimental._stream cimport Stream


cdef struct _MemAttrs:
    int device_id
    bint is_device_accessible
    bint is_host_accessible


cdef class Buffer:
    cdef:
        DevicePtrHandle _h_ptr
        size_t          _size
        MemoryResource  _memory_resource
        object          _ipc_data


cdef class MemoryResource:
    pass


# Helper function to create a Buffer from a DevicePtrHandle
cdef Buffer Buffer_from_deviceptr_handle(
    DevicePtrHandle h_ptr,
    size_t size,
    MemoryResource mr,
    object ipc_descriptor = *
)
