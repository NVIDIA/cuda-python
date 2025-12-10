# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from cuda.bindings cimport cydriver
from cuda.core.experimental._memory._buffer cimport MemoryResource
from cuda.core.experimental._memory._ipc cimport IPCDataForMR


cdef class DeviceMemoryResource(MemoryResource):
    cdef:
        int                   _device_id
        cydriver.CUmemoryPool _handle
        bint                  _mempool_owned
        IPCDataForMR          _ipc_data
        object                _attributes
        object                _peer_accessible_by
        object                __weakref__


cpdef DMR_mempool_get_access(DeviceMemoryResource, int)
