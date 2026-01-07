# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from libc.stddef cimport size_t
from libc.stdint cimport intptr_t, uint32_t
from libcpp.memory cimport shared_ptr

from cpython.pycapsule cimport PyCapsule_Import

from cuda.bindings cimport cydriver

# Declare the C++ namespace and types (inline helpers live in the header).
cdef extern from "_cpp/resource_handles.hpp" namespace "cuda_core":
    ctypedef shared_ptr[const cydriver.CUcontext] ContextHandle
    ctypedef shared_ptr[const cydriver.CUstream] StreamHandle
    ctypedef shared_ptr[const cydriver.CUevent] EventHandle
    ctypedef shared_ptr[const cydriver.CUmemoryPool] MemoryPoolHandle
    ctypedef shared_ptr[const cydriver.CUdeviceptr] DevicePtrHandle

    # cu() - extract the raw CUDA handle (inline C++)
    cydriver.CUcontext cu(ContextHandle h) nogil
    cydriver.CUstream cu(StreamHandle h) nogil
    cydriver.CUevent cu(EventHandle h) nogil
    cydriver.CUmemoryPool cu(MemoryPoolHandle h) nogil
    cydriver.CUdeviceptr cu(DevicePtrHandle h) nogil

    # intptr() - extract handle as intptr_t for Python interop (inline C++)
    intptr_t intptr(ContextHandle h) nogil
    intptr_t intptr(StreamHandle h) nogil
    intptr_t intptr(EventHandle h) nogil
    intptr_t intptr(MemoryPoolHandle h) nogil
    intptr_t intptr(DevicePtrHandle h) nogil

    # py() - convert handle to Python driver wrapper object (inline C++; requires GIL)
    object py(ContextHandle h)
    object py(StreamHandle h)
    object py(EventHandle h)
    object py(MemoryPoolHandle h)
    object py(DevicePtrHandle h)


# The resource handles API table is exported from `cuda.core._resource_handles`
# as a PyCapsule named:
#
#   "cuda.core._resource_handles._CXX_API"
#
# Consumers dispatch through this table to avoid relying on RTLD_GLOBAL and to
# ensure a single owner of correctness-critical static/thread_local state.
cdef extern from "_cpp/resource_handles_cxx_api.hpp" namespace "cuda_core":
    cdef struct ResourceHandlesCxxApiV1:
        uint32_t abi_version
        uint32_t struct_size

        # Thread-local error handling
        cydriver.CUresult (*get_last_error)() nogil
        cydriver.CUresult (*peek_last_error)() nogil
        void (*clear_last_error)() nogil

        # Context handles
        ContextHandle (*create_context_handle_ref)(cydriver.CUcontext ctx) nogil
        ContextHandle (*get_primary_context)(int device_id) nogil
        ContextHandle (*get_current_context)() nogil

        # Stream handles
        StreamHandle (*create_stream_handle)(ContextHandle h_ctx, unsigned int flags, int priority) nogil
        StreamHandle (*create_stream_handle_ref)(cydriver.CUstream stream) nogil
        StreamHandle (*create_stream_handle_with_owner)(cydriver.CUstream stream, object owner)
        StreamHandle (*get_legacy_stream)() nogil
        StreamHandle (*get_per_thread_stream)() nogil

        # Event handles
        EventHandle (*create_event_handle)(ContextHandle h_ctx, unsigned int flags) nogil
        EventHandle (*create_event_handle_noctx)(unsigned int flags) nogil
        EventHandle (*create_event_handle_ipc)(const cydriver.CUipcEventHandle& ipc_handle) nogil

        # Memory pool handles
        MemoryPoolHandle (*create_mempool_handle)(const cydriver.CUmemPoolProps& props) nogil
        MemoryPoolHandle (*create_mempool_handle_ref)(cydriver.CUmemoryPool pool) nogil
        MemoryPoolHandle (*get_device_mempool)(int device_id) nogil
        MemoryPoolHandle (*create_mempool_handle_ipc)(int fd, cydriver.CUmemAllocationHandleType handle_type) nogil

        # Device pointer handles
        DevicePtrHandle (*deviceptr_alloc_from_pool)(
            size_t size,
            MemoryPoolHandle h_pool,
            StreamHandle h_stream) nogil
        DevicePtrHandle (*deviceptr_alloc_async)(size_t size, StreamHandle h_stream) nogil
        DevicePtrHandle (*deviceptr_alloc)(size_t size) nogil
        DevicePtrHandle (*deviceptr_alloc_host)(size_t size) nogil
        DevicePtrHandle (*deviceptr_create_ref)(cydriver.CUdeviceptr ptr) nogil
        DevicePtrHandle (*deviceptr_create_with_owner)(cydriver.CUdeviceptr ptr, object owner)
        DevicePtrHandle (*deviceptr_import_ipc)(
            MemoryPoolHandle h_pool,
            const void* export_data,
            StreamHandle h_stream) nogil
        StreamHandle (*deallocation_stream)(const DevicePtrHandle& h) nogil
        void (*set_deallocation_stream)(const DevicePtrHandle& h, StreamHandle h_stream) nogil

    const ResourceHandlesCxxApiV1* get_resource_handles_cxx_api_v1() nogil


cdef const ResourceHandlesCxxApiV1* _handles_table = NULL


cdef inline const ResourceHandlesCxxApiV1* _get_handles_table() except NULL nogil:
    global _handles_table
    if _handles_table == NULL:
        with gil:
            if _handles_table == NULL:
                _handles_table = <const ResourceHandlesCxxApiV1*>PyCapsule_Import(
                    b"cuda.core._resource_handles._CXX_API", 0
                )
                if _handles_table == NULL:
                    raise ImportError("Failed to import cuda.core._resource_handles._CXX_API capsule")
                if _handles_table.abi_version != 1:
                    raise ImportError("Unsupported resource handles C++ API version")
                if _handles_table.struct_size < sizeof(ResourceHandlesCxxApiV1):
                    raise ImportError("Resource handles C++ API table is too small")
    return _handles_table


# -----------------------------------------------------------------------------
# Dispatch wrappers
#
# These wrappers assume _handles_table has been initialized. Consumers must call
# _init_handles_table() at module level before using these functions in nogil blocks.
# -----------------------------------------------------------------------------

cdef inline void _init_handles_table() except *:
    """Initialize the handles table. Call at module level before using wrappers."""
    _get_handles_table()


cdef inline cydriver.CUresult get_last_error() noexcept nogil:
    return _handles_table.get_last_error()


cdef inline cydriver.CUresult peek_last_error() noexcept nogil:
    return _handles_table.peek_last_error()


cdef inline void clear_last_error() noexcept nogil:
    _handles_table.clear_last_error()


cdef inline ContextHandle create_context_handle_ref(cydriver.CUcontext ctx) noexcept nogil:
    return _handles_table.create_context_handle_ref(ctx)


cdef inline ContextHandle get_primary_context(int device_id) noexcept nogil:
    return _handles_table.get_primary_context(device_id)


cdef inline ContextHandle get_current_context() noexcept nogil:
    return _handles_table.get_current_context()


cdef inline StreamHandle create_stream_handle(ContextHandle h_ctx, unsigned int flags, int priority) noexcept nogil:
    return _handles_table.create_stream_handle(h_ctx, flags, priority)


cdef inline StreamHandle create_stream_handle_ref(cydriver.CUstream stream) noexcept nogil:
    return _handles_table.create_stream_handle_ref(stream)


cdef inline StreamHandle create_stream_handle_with_owner(cydriver.CUstream stream, object owner):
    return _handles_table.create_stream_handle_with_owner(stream, owner)


cdef inline StreamHandle get_legacy_stream() noexcept nogil:
    return _handles_table.get_legacy_stream()


cdef inline StreamHandle get_per_thread_stream() noexcept nogil:
    return _handles_table.get_per_thread_stream()


cdef inline EventHandle create_event_handle(ContextHandle h_ctx, unsigned int flags) noexcept nogil:
    return _handles_table.create_event_handle(h_ctx, flags)


cdef inline EventHandle create_event_handle_noctx(unsigned int flags) noexcept nogil:
    return _handles_table.create_event_handle_noctx(flags)


cdef inline EventHandle create_event_handle_ipc(const cydriver.CUipcEventHandle& ipc_handle) noexcept nogil:
    return _handles_table.create_event_handle_ipc(ipc_handle)


cdef inline MemoryPoolHandle create_mempool_handle(const cydriver.CUmemPoolProps& props) noexcept nogil:
    return _handles_table.create_mempool_handle(props)


cdef inline MemoryPoolHandle create_mempool_handle_ref(cydriver.CUmemoryPool pool) noexcept nogil:
    return _handles_table.create_mempool_handle_ref(pool)


cdef inline MemoryPoolHandle get_device_mempool(int device_id) noexcept nogil:
    return _handles_table.get_device_mempool(device_id)


cdef inline MemoryPoolHandle create_mempool_handle_ipc(int fd, cydriver.CUmemAllocationHandleType handle_type) noexcept nogil:
    return _handles_table.create_mempool_handle_ipc(fd, handle_type)


cdef inline DevicePtrHandle deviceptr_alloc_from_pool(
    size_t size,
    MemoryPoolHandle h_pool,
    StreamHandle h_stream) noexcept nogil:
    return _handles_table.deviceptr_alloc_from_pool(size, h_pool, h_stream)


cdef inline DevicePtrHandle deviceptr_alloc_async(size_t size, StreamHandle h_stream) noexcept nogil:
    return _handles_table.deviceptr_alloc_async(size, h_stream)


cdef inline DevicePtrHandle deviceptr_alloc(size_t size) noexcept nogil:
    return _handles_table.deviceptr_alloc(size)


cdef inline DevicePtrHandle deviceptr_alloc_host(size_t size) noexcept nogil:
    return _handles_table.deviceptr_alloc_host(size)


cdef inline DevicePtrHandle deviceptr_create_ref(cydriver.CUdeviceptr ptr) noexcept nogil:
    return _handles_table.deviceptr_create_ref(ptr)


cdef inline DevicePtrHandle deviceptr_create_with_owner(cydriver.CUdeviceptr ptr, object owner):
    return _handles_table.deviceptr_create_with_owner(ptr, owner)


cdef inline DevicePtrHandle deviceptr_import_ipc(
    MemoryPoolHandle h_pool,
    const void* export_data,
    StreamHandle h_stream) noexcept nogil:
    return _handles_table.deviceptr_import_ipc(h_pool, export_data, h_stream)


cdef inline StreamHandle deallocation_stream(const DevicePtrHandle& h) noexcept nogil:
    return _handles_table.deallocation_stream(h)


cdef inline void set_deallocation_stream(const DevicePtrHandle& h, StreamHandle h_stream) noexcept nogil:
    _handles_table.set_deallocation_stream(h, h_stream)
