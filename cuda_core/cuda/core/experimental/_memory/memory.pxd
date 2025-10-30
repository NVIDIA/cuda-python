# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from libc.stdint cimport uintptr_t, intptr_t
from cuda.bindings cimport cydriver

from cuda.core.experimental._memory.ipc cimport IPCAllocationHandle
from cuda.core.experimental._stream cimport Stream as _cyStream


cdef class Buffer:
    cdef:
        intptr_t _ptr
        size_t _size
        MemoryResource _mr
        object _ptr_obj
        _cyStream _alloc_stream

    cpdef close(self, stream=*)


cdef class MemoryResource:
    cdef void _deallocate(self, intptr_t ptr, size_t size, _cyStream stream) noexcept


cdef class DeviceMemoryResource(MemoryResource):
    cdef:
        int _dev_id
        cydriver.CUmemoryPool _mempool_handle
        object _attributes
        cydriver.CUmemAllocationHandleType _ipc_handle_type
        bint _mempool_owned
        bint _is_mapped
        object _uuid
        IPCAllocationHandle _alloc_handle
        object __weakref__

    cpdef close(self)
    cdef Buffer _allocate(self, size_t size, _cyStream stream)
    cdef void _deallocate(self, intptr_t ptr, size_t size, _cyStream stream) noexcept
    cpdef deallocate(self, ptr, size_t size, stream=*)
