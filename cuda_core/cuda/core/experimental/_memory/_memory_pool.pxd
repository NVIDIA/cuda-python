# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from cuda.bindings cimport cydriver
from cuda.core.experimental._memory._buffer cimport MemoryResource
from cuda.core.experimental._memory._ipc cimport IPCDataForMR


cdef class _MemPool(MemoryResource):
    cdef:
        int                   _dev_id
        cydriver.CUmemoryPool _handle
        bint                  _mempool_owned
        IPCDataForMR          _ipc_data
        object                _attributes
        object                _peer_accessible_by
        object                __weakref__


cdef class _MemPoolOptions:

    cdef:
        bint _ipc_enabled
        size_t _max_size
        cydriver.CUmemLocationType _location
        cydriver.CUmemAllocationType _type
