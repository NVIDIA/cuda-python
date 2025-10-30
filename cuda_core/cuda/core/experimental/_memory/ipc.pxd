# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0


# Holds DeviceMemoryResource objects imported by this process.  This enables
# buffer serialization, as buffers can reduce to a pair comprising the memory
# resource UUID (the key into this registry) and the serialized buffer
# descriptor.
cdef object registry


cdef class IPCBufferDescriptor:
    cdef:
        bytes _reserved
        size_t _size


cdef class IPCAllocationHandle:
    cdef:
        int _handle
        object _uuid

    cpdef close(self)


