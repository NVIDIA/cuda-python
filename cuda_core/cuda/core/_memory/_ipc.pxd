# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from cuda.bindings cimport cydriver
from cuda.core._memory._buffer cimport Buffer
from cuda.core._memory._device_memory_resource cimport DeviceMemoryResource


# Holds DeviceMemoryResource objects imported by this process.  This enables
# buffer serialization, as buffers can reduce to a pair comprising the memory
# resource UUID (the key into this registry) and the serialized buffer
# descriptor.
cdef object registry


# The IPC handle type for this platform.  IPC is currently only supported on
# Linux. On other platforms, the IPC handle type is set equal to the no-IPC
# handle type.
cdef cydriver.CUmemAllocationHandleType IPC_HANDLE_TYPE


# Whether IPC is supported on this platform.
cdef is_supported()


cdef class IPCData:
    cdef:
        IPCAllocationHandle _alloc_handle
        bint                _is_mapped


cdef class IPCBufferDescriptor:
    cdef:
        bytes  _payload
        size_t _size


cdef class IPCAllocationHandle:
    cdef:
        int    _handle
        object _uuid

    cpdef close(self)


# Buffer IPC Implementation
# -------------------------
cdef IPCBufferDescriptor Buffer_get_ipc_descriptor(Buffer)
cdef Buffer Buffer_from_ipc_descriptor(cls, DeviceMemoryResource, IPCBufferDescriptor, stream)


# DeviceMemoryResource IPC Implementation
# ---------------------------------------
cdef DeviceMemoryResource DMR_from_allocation_handle(cls, device_id, alloc_handle)
cdef DeviceMemoryResource DMR_from_registry(uuid)
cdef DeviceMemoryResource DMR_register(DeviceMemoryResource, uuid)
cdef IPCAllocationHandle DMR_export_mempool(DeviceMemoryResource)
