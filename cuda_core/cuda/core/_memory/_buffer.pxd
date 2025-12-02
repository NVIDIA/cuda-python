# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from libc.stdint cimport uintptr_t

from cuda.core._stream cimport Stream


cdef class Buffer:
    cdef:
        uintptr_t      _ptr
        size_t         _size
        MemoryResource _memory_resource
        object         _ptr_obj
        Stream         _alloc_stream


cdef class MemoryResource:
    pass
