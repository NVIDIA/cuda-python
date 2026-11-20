# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from cuda.bindings cimport cydriver
from cuda.core._memory._buffer cimport Buffer, MemoryResource
from cuda.core._memory._ipc cimport IPCDataForMR
from cuda.core._rt cimport MemoryPoolHandle
from cuda.core._stream cimport Stream
from cuda.core._utils.version cimport cy_driver_version  # no-cython-lint


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

cdef int MP_set_access(
    _MemPool self, const cydriver.CUmemAccessDesc* desc, size_t count,
    bint retry_already_mapped=*) except? -1

IF CUDA_CORE_BUILD_MAJOR >= 13:
    cdef inline bint MP_location_query_available():
        # CU_MEMPOOL_ATTR_LOCATION_TYPE and _ID are 13.2 driver attributes. Every
        # cuda-bindings at or above the CUDA 13 floor defines them, so only the
        # driver can lack them; an older driver rejects them as invalid.
        return cy_driver_version() >= (13, 2, 0)
ELSE:
    cdef inline bint MP_location_query_available():
        return False

# Requires MP_location_query_available().
cdef int MP_get_location(_MemPool self, cydriver.CUmemLocation* loc) except? -1


cdef inline int MP_check_open(_MemPool self) except -1:
    if not self._h_pool:
        raise RuntimeError(f"{self.__class__.__name__} has been closed")
    return 0


# Allocate from this pool, returning an instance of `cls` (defaulting to
# Buffer). Subclasses (e.g. ManagedMemoryResource) pass their own buffer
# subclass so their `allocate` returns the typed object.
cdef Buffer _MP_allocate(_MemPool self, size_t size, Stream stream, type cls = *)
