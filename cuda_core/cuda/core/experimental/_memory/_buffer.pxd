# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from libc.stdint cimport intptr_t

from cuda.core.experimental._stream cimport Stream as _cyStream


cdef class Buffer:
    cdef:
        intptr_t _ptr
        size_t _size
        MemoryResource _mr
        object _ptr_obj
        _cyStream _alloc_stream


cdef class MemoryResource:
    pass


