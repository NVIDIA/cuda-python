# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from libc.stdint cimport uintptr_t

from cuda.core._resource_handles cimport DevicePtrHandle
from cuda.core._stream cimport Stream


cdef struct _MemAttrs:
    int device_id
    bint is_device_accessible
    bint is_host_accessible
    bint is_managed


cdef class Buffer:
    cdef:
        DevicePtrHandle _h_ptr
        MemoryResource  _memory_resource
        object          _ipc_data
        object          _owner
        _MemAttrs       _mem_attrs
        bint            _mem_attrs_inited
        object          __weakref__
    cdef public:
        # Python code in _memory/_virtual_memory_resource.py needs to update
        # this value, though it is technically private.
        size_t          _size


cdef class MemoryResource:
    pass


# Helper function to create a Buffer from a DevicePtrHandle
cdef Buffer Buffer_from_deviceptr_handle(
    DevicePtrHandle h_ptr,
    size_t size,
    MemoryResource mr,
    object ipc_descriptor = *
)
