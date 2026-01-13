# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from libc.stddef cimport size_t
from libc.stdint cimport intptr_t

from libcpp.memory cimport shared_ptr

from cuda.bindings cimport cydriver


# =============================================================================
# Handle type aliases and inline helpers (declared from C++ header)
# =============================================================================

cdef extern from "_cpp/resource_handles.hpp" namespace "cuda_core":
    # Handle types
    ctypedef shared_ptr[const cydriver.CUcontext] ContextHandle
    ctypedef shared_ptr[const cydriver.CUstream] StreamHandle
    ctypedef shared_ptr[const cydriver.CUevent] EventHandle
    ctypedef shared_ptr[const cydriver.CUmemoryPool] MemoryPoolHandle
    ctypedef shared_ptr[const cydriver.CUdeviceptr] DevicePtrHandle

    # as_cu() - extract the raw CUDA handle (inline C++)
    cydriver.CUcontext as_cu(ContextHandle h) noexcept nogil
    cydriver.CUstream as_cu(StreamHandle h) noexcept nogil
    cydriver.CUevent as_cu(EventHandle h) noexcept nogil
    cydriver.CUmemoryPool as_cu(MemoryPoolHandle h) noexcept nogil
    cydriver.CUdeviceptr as_cu(DevicePtrHandle h) noexcept nogil

    # as_intptr() - extract handle as intptr_t for Python interop (inline C++)
    intptr_t as_intptr(ContextHandle h) noexcept nogil
    intptr_t as_intptr(StreamHandle h) noexcept nogil
    intptr_t as_intptr(EventHandle h) noexcept nogil
    intptr_t as_intptr(MemoryPoolHandle h) noexcept nogil
    intptr_t as_intptr(DevicePtrHandle h) noexcept nogil

    # as_py() - convert handle to Python driver wrapper object (inline C++; requires GIL)
    object as_py(ContextHandle h)
    object as_py(StreamHandle h)
    object as_py(EventHandle h)
    object as_py(MemoryPoolHandle h)
    object as_py(DevicePtrHandle h)


# =============================================================================
# Wrapper function declarations (implemented in _resource_handles.pyx)
#
# Consumer modules cimport these. Calls go through _resource_handles.so.
# =============================================================================

# Thread-local error handling
cdef cydriver.CUresult get_last_error() noexcept nogil
cdef cydriver.CUresult peek_last_error() noexcept nogil
cdef void clear_last_error() noexcept nogil

# Context handles
cdef ContextHandle create_context_handle_ref(cydriver.CUcontext ctx) noexcept nogil
cdef ContextHandle get_primary_context(int device_id) noexcept nogil
cdef ContextHandle get_current_context() noexcept nogil

# Stream handles
cdef StreamHandle create_stream_handle(
    ContextHandle h_ctx, unsigned int flags, int priority) noexcept nogil
cdef StreamHandle create_stream_handle_ref(cydriver.CUstream stream) noexcept nogil
cdef StreamHandle create_stream_handle_with_owner(cydriver.CUstream stream, object owner) noexcept nogil
cdef StreamHandle get_legacy_stream() noexcept nogil
cdef StreamHandle get_per_thread_stream() noexcept nogil

# Event handles
cdef EventHandle create_event_handle(ContextHandle h_ctx, unsigned int flags) noexcept nogil
cdef EventHandle create_event_handle_noctx(unsigned int flags) noexcept nogil
cdef EventHandle create_event_handle_ipc(
    const cydriver.CUipcEventHandle& ipc_handle) noexcept nogil

# Memory pool handles
cdef MemoryPoolHandle create_mempool_handle(
    const cydriver.CUmemPoolProps& props) noexcept nogil
cdef MemoryPoolHandle create_mempool_handle_ref(cydriver.CUmemoryPool pool) noexcept nogil
cdef MemoryPoolHandle get_device_mempool(int device_id) noexcept nogil
cdef MemoryPoolHandle create_mempool_handle_ipc(
    int fd, cydriver.CUmemAllocationHandleType handle_type) noexcept nogil

# Device pointer handles
cdef DevicePtrHandle deviceptr_alloc_from_pool(
    size_t size, MemoryPoolHandle h_pool, StreamHandle h_stream) noexcept nogil
cdef DevicePtrHandle deviceptr_alloc_async(size_t size, StreamHandle h_stream) noexcept nogil
cdef DevicePtrHandle deviceptr_alloc(size_t size) noexcept nogil
cdef DevicePtrHandle deviceptr_alloc_host(size_t size) noexcept nogil
cdef DevicePtrHandle deviceptr_create_ref(cydriver.CUdeviceptr ptr) noexcept nogil
cdef DevicePtrHandle deviceptr_create_with_owner(cydriver.CUdeviceptr ptr, object owner) noexcept nogil
cdef DevicePtrHandle deviceptr_import_ipc(
    MemoryPoolHandle h_pool, const void* export_data, StreamHandle h_stream) noexcept nogil
cdef StreamHandle deallocation_stream(const DevicePtrHandle& h) noexcept nogil
cdef void set_deallocation_stream(const DevicePtrHandle& h, StreamHandle h_stream) noexcept nogil
