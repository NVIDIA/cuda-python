# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0


cdef class IPCBufferDescriptor:
    cdef:
        bytes _reserved
        size_t _size


cdef class IPCAllocationHandle:
    cdef:
        int _handle
        object _uuid

    cpdef close(self)

