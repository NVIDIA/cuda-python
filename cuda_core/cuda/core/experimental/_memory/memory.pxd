# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from libc.stdint cimport uintptr_t, intptr_t
from cuda.bindings cimport cydriver

from cuda.core.experimental._stream cimport Stream as _cyStream


cdef class _cyBuffer:
    cdef:
        intptr_t _ptr
        size_t _size
        _cyMemoryResource _mr
        object _ptr_obj
        _cyStream _alloc_stream


cdef class _cyMemoryResource:
    cdef Buffer _allocate(self, size_t size, _cyStream stream)
    cdef void _deallocate(self, intptr_t ptr, size_t size, _cyStream stream) noexcept


cdef class Buffer(_cyBuffer):
    cpdef close(self, stream=*)


cdef class MemoryResource(_cyMemoryResource):
    cdef void _deallocate(self, intptr_t ptr, size_t size, _cyStream stream) noexcept


cdef class IPCBufferDescriptor:
    cdef:
        bytes _reserved
        size_t _size


cdef class IPCAllocationHandle:
    cdef:
        int _handle
        object _uuid

    cpdef close(self)


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
    cpdef IPCAllocationHandle get_allocation_handle(self)
    cdef Buffer _allocate(self, size_t size, _cyStream stream)
    cdef void _deallocate(self, intptr_t ptr, size_t size, _cyStream stream) noexcept
    cpdef deallocate(self, ptr, size_t size, stream=*)
