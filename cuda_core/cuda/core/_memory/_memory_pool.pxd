# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from cuda.bindings cimport cydriver
from cuda.core._memory._buffer cimport MemoryResource
from cuda.core._memory._ipc cimport IPCDataForMR
from cuda.core._resource_handles cimport MemoryPoolHandle


cdef class _MemPool(MemoryResource):
    cdef:
        int                   _dev_id
        MemoryPoolHandle      _h_pool
        bint                  _mempool_owned
        IPCDataForMR          _ipc_data
        object                _attributes
        object                _peer_accessible_by
        object                __weakref__


cdef class _MemPoolAttributes:
    cdef:
        MemoryPoolHandle _h_pool

    @staticmethod
    cdef _MemPoolAttributes _init(MemoryPoolHandle h_pool)

    cdef int _getattribute(self, cydriver.CUmemPool_attribute attr_enum, void* value) except? -1


cdef class _MemPoolOptions:

    cdef:
        bint _ipc_enabled
        size_t _max_size
        cydriver.CUmemLocationType _location
        cydriver.CUmemAllocationType _type
        bint _use_current
