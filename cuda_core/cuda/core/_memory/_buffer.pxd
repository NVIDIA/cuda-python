# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from libc.stddef cimport size_t

from libcpp cimport bool as cpp_bool
from libcpp.atomic cimport atomic as std_atomic

from cuda.bindings cimport cydriver
from cuda.core._resource_handles cimport DevicePtrHandle
from cuda.core._stream cimport Stream


cdef struct _MemAttrs:
    int device_id
    bint is_device_accessible
    bint is_host_accessible
    bint is_managed


cdef class Buffer:
    cdef:
        DevicePtrHandle       _h_ptr
        MemoryResource        _memory_resource
        object                _ipc_data
        object                _owner
        _MemAttrs             _mem_attrs
        std_atomic[cpp_bool]  _mem_attrs_inited
        object                __weakref__
    cdef public:
        # Python code in _memory/_virtual_memory_resource.py needs to update
        # this value, though it is technically private.
        size_t          _size


cdef class MemoryResource:
    pass


# Helper function to create a Buffer from a DevicePtrHandle.
# `cls` lets callers materialize Buffer subclasses (e.g. ManagedBuffer for
# managed-memory allocations); defaults to Buffer.
cdef Buffer Buffer_from_deviceptr_handle(
    DevicePtrHandle h_ptr,
    size_t size,
    MemoryResource mr,
    object ipc_descriptor = *,
    type cls = *,
)


# Wrap a raw device pointer with MR-owned teardown and record the stream.
cdef DevicePtrHandle deviceptr_create_owned_by_mr(
    cydriver.CUdeviceptr ptr,
    size_t size,
    object mr,
    Stream stream,
) except *


# Shared argument coercion for the batched free functions (copy_batch,
# prefetch_batch, discard_batch, discard_prefetch_batch). `single_hint`
# names the per-buffer API to use instead when a bare Buffer is passed.
cdef tuple Buffer_coerce_batch(object buffers, str what, str single_hint)
