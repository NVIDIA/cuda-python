# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from cuda.bindings cimport cydriver

from cuda.core.experimental._memory.buffer cimport MemoryResource
from cuda.core.experimental._memory.ipc cimport IPCAllocationHandle


cdef class DeviceMemoryResource(MemoryResource):
    cdef:
        int _dev_id
        cydriver.CUmemoryPool _mempool_handle
        object _attributes
        bint _mempool_owned
        object __weakref__

        cydriver.CUmemAllocationHandleType _ipc_handle_type
        bint _is_mapped
        object _uuid
        IPCAllocationHandle _alloc_handle

