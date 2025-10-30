# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from cuda.bindings cimport cydriver
from cuda.core.experimental._memory.memory cimport DeviceMemoryResource


# Holds DeviceMemoryResource objects imported by this process.  This enables
# buffer serialization, as buffers can reduce to a pair comprising the memory
# resource UUID (the key into this registry) and the serialized buffer
# descriptor.
cdef object registry

# IPC is currently only supported on Linux. On other platforms, the IPC handle
# type is set equal to the no-IPC handle type.
cdef cydriver.CUmemAllocationHandleType IPC_HANDLE_TYPE


cdef class IPCBufferDescriptor:
    cdef:
        bytes _reserved
        size_t _size


cdef class IPCAllocationHandle:
    cdef:
        int _handle
        object _uuid

    cpdef close(self)


# DeviceMemoryResource IPC Implementation
# ------
cpdef IPCAllocationHandle DMR_get_allocation_handle(DeviceMemoryResource)
cpdef DeviceMemoryResource DMR_from_allocation_handle(cls, device_id, alloc_handle)
cpdef DeviceMemoryResource DMR_register(DeviceMemoryResource, uuid)
cpdef DeviceMemoryResource DMR_from_registry(uuid)
