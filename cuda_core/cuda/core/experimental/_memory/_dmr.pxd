# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from cuda.bindings cimport cydriver

from cuda.core.experimental._memory._buffer cimport MemoryResource
from cuda.core.experimental._memory._ipc cimport IPCAllocationHandle, IPCData


cdef class DeviceMemoryResource(MemoryResource):
    cdef:
        object __weakref__
        int _dev_id
        cydriver.CUmemoryPool _handle
        object _attributes
        bint _mempool_owned
        IPCData _ipc_data

