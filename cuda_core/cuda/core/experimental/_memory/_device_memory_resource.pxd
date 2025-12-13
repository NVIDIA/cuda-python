# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from cuda.bindings cimport cydriver
from cuda.core.experimental._memory._buffer cimport MemoryResource
from cuda.core.experimental._memory._ipc cimport IPCDataForMR
from cuda.core.experimental._resource_handles cimport MemoryPoolHandle


cdef class DeviceMemoryResource(MemoryResource):
    cdef:
        MemoryPoolHandle _h_pool
        int              _device_id
        bint             _pool_owned
        IPCDataForMR     _ipc_data
        object           _attributes
        object           _peer_accessible_by
        object           __weakref__


cpdef DMR_mempool_get_access(DeviceMemoryResource, int)
