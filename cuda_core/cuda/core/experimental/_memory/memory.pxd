# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from libc.stdint cimport uintptr_t, intptr_t
from cuda.core.experimental._stream cimport Stream as cyStream

from cuda.core.experimental._stream import Stream


cdef class _cyBuffer:
    """
    Internal only. Responsible for offering fast C method access.
    """
    cdef:
        intptr_t _ptr
        size_t _size
        _cyMemoryResource _mr
        object _ptr_obj
        cyStream _alloc_stream


cdef class Buffer(_cyBuffer):
    cpdef close(self, stream: Stream=*)


cdef class _cyMemoryResource:
    """
    Internal only. Responsible for offering fast C method access.
    """
    cdef Buffer _allocate(self, size_t size, cyStream stream)
    cdef void _deallocate(self, intptr_t ptr, size_t size, cyStream stream) noexcept


cdef class MemoryResource(_cyMemoryResource):
    cdef void _deallocate(self, intptr_t ptr, size_t size, cyStream stream) noexcept
