# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from cuda.bindings cimport cydriver
from cuda.core._memory._buffer cimport MemoryResource
from cuda.core._memory._ipc cimport IPCData


cdef class DeviceMemoryResource(MemoryResource):
    cdef:
        int                   _dev_id
        cydriver.CUmemoryPool _handle
        bint                  _mempool_owned
        IPCData               _ipc_data
        object                _attributes
        object                _peer_accessible_by
        object                __weakref__


cpdef DMR_mempool_get_access(DeviceMemoryResource, int)
