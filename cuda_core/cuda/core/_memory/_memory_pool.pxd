# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from cuda.bindings cimport cydriver
from cuda.core._memory._buffer cimport MemoryResource
from cuda.core._memory._ipc cimport IPCDataForMR
from cuda.core._resource_handles cimport MemoryPoolHandle


cdef class _MemPool(MemoryResource):
    cdef:
        MemoryPoolHandle      _h_pool
        bint                  _mempool_owned
        IPCDataForMR          _ipc_data
        object                _attributes
        object                __weakref__


cdef int MP_init_create_pool(
    _MemPool self,
    cydriver.CUmemLocationType loc_type,
    int loc_id,
    cydriver.CUmemAllocationType alloc_type,
    bint ipc_enabled,
    size_t max_size,
) except? -1

cdef int MP_init_current_pool(
    _MemPool self,
    cydriver.CUmemLocationType loc_type,
    int loc_id,
    cydriver.CUmemAllocationType alloc_type,
) except? -1

cdef int MP_raise_release_threshold(_MemPool self) except? -1


cdef class _MemPoolAttributes:
    cdef:
        MemoryPoolHandle _h_pool

    @staticmethod
    cdef _MemPoolAttributes _init(MemoryPoolHandle h_pool)

    cdef int _getattribute(self, cydriver.CUmemPool_attribute attr_enum, void* value) except? -1
