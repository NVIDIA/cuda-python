# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from libc.stddef cimport size_t
from libc.stdint cimport intptr_t

from libcpp.memory cimport shared_ptr

from cuda.bindings cimport cydriver
from cuda.bindings cimport cynvrtc
from cuda.bindings cimport cynvvm


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
    ctypedef shared_ptr[const cydriver.CUlibrary] LibraryHandle
    ctypedef shared_ptr[const cydriver.CUkernel] KernelHandle
    ctypedef shared_ptr[const cydriver.CUgraphicsResource] GraphicsResourceHandle
    ctypedef shared_ptr[const cynvrtc.nvrtcProgram] NvrtcProgramHandle
    ctypedef shared_ptr[const cynvvm.nvvmProgram] NvvmProgramHandle

    # as_cu() - extract the raw CUDA handle (inline C++)
    cydriver.CUcontext as_cu(ContextHandle h) noexcept nogil
    cydriver.CUstream as_cu(StreamHandle h) noexcept nogil
    cydriver.CUevent as_cu(EventHandle h) noexcept nogil
    cydriver.CUmemoryPool as_cu(MemoryPoolHandle h) noexcept nogil
    cydriver.CUdeviceptr as_cu(DevicePtrHandle h) noexcept nogil
    cydriver.CUlibrary as_cu(LibraryHandle h) noexcept nogil
    cydriver.CUkernel as_cu(KernelHandle h) noexcept nogil
    cydriver.CUgraphicsResource as_cu(GraphicsResourceHandle h) noexcept nogil
    cynvrtc.nvrtcProgram as_cu(NvrtcProgramHandle h) noexcept nogil
    cynvvm.nvvmProgram as_cu(NvvmProgramHandle h) noexcept nogil\

    # as_intptr() - extract handle as intptr_t for Python interop (inline C++)
    intptr_t as_intptr(ContextHandle h) noexcept nogil
    intptr_t as_intptr(StreamHandle h) noexcept nogil
    intptr_t as_intptr(EventHandle h) noexcept nogil
    intptr_t as_intptr(MemoryPoolHandle h) noexcept nogil
    intptr_t as_intptr(DevicePtrHandle h) noexcept nogil
    intptr_t as_intptr(LibraryHandle h) noexcept nogil
    intptr_t as_intptr(KernelHandle h) noexcept nogil
    intptr_t as_intptr(GraphicsResourceHandle h) noexcept nogil
    intptr_t as_intptr(NvrtcProgramHandle h) noexcept nogil
    intptr_t as_intptr(NvvmProgramHandle h) noexcept nogil

    # as_py() - convert handle to Python wrapper object (inline C++; requires GIL)
    object as_py(ContextHandle h)
    object as_py(StreamHandle h)
    object as_py(EventHandle h)
    object as_py(MemoryPoolHandle h)
    object as_py(DevicePtrHandle h)
    object as_py(LibraryHandle h)
    object as_py(KernelHandle h)
    object as_py(GraphicsResourceHandle h)
    object as_py(NvrtcProgramHandle h)
    object as_py(NvvmProgramHandle h)


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
cdef ContextHandle create_context_handle_ref(cydriver.CUcontext ctx) except+ nogil
cdef ContextHandle get_primary_context(int device_id) except+ nogil
cdef ContextHandle get_current_context() except+ nogil

# Stream handles
cdef StreamHandle create_stream_handle(
    const ContextHandle& h_ctx, unsigned int flags, int priority) except+ nogil
cdef StreamHandle create_stream_handle_ref(cydriver.CUstream stream) except+ nogil
cdef StreamHandle create_stream_handle_with_owner(cydriver.CUstream stream, object owner) except+ nogil
cdef StreamHandle get_legacy_stream() except+ nogil
cdef StreamHandle get_per_thread_stream() except+ nogil

# Event handles
cdef EventHandle create_event_handle(const ContextHandle& h_ctx, unsigned int flags) except+ nogil
cdef EventHandle create_event_handle_noctx(unsigned int flags) except+ nogil
cdef EventHandle create_event_handle_ipc(
    const cydriver.CUipcEventHandle& ipc_handle) except+ nogil

# Memory pool handles
cdef MemoryPoolHandle create_mempool_handle(
    const cydriver.CUmemPoolProps& props) except+ nogil
cdef MemoryPoolHandle create_mempool_handle_ref(cydriver.CUmemoryPool pool) except+ nogil
cdef MemoryPoolHandle get_device_mempool(int device_id) except+ nogil
cdef MemoryPoolHandle create_mempool_handle_ipc(
    int fd, cydriver.CUmemAllocationHandleType handle_type) except+ nogil

# Device pointer handles
cdef DevicePtrHandle deviceptr_alloc_from_pool(
    size_t size, const MemoryPoolHandle& h_pool, const StreamHandle& h_stream) except+ nogil
cdef DevicePtrHandle deviceptr_alloc_async(size_t size, const StreamHandle& h_stream) except+ nogil
cdef DevicePtrHandle deviceptr_alloc(size_t size) except+ nogil
cdef DevicePtrHandle deviceptr_alloc_host(size_t size) except+ nogil
cdef DevicePtrHandle deviceptr_create_ref(cydriver.CUdeviceptr ptr) except+ nogil
cdef DevicePtrHandle deviceptr_create_with_owner(cydriver.CUdeviceptr ptr, object owner) except+ nogil
cdef DevicePtrHandle deviceptr_create_with_mr(
    cydriver.CUdeviceptr ptr, size_t size, object mr) except+ nogil

# MR deallocation callback type and registration
ctypedef void (*MRDeallocCallback)(
    object mr, cydriver.CUdeviceptr ptr, size_t size,
    const StreamHandle& stream) noexcept
cdef void register_mr_dealloc_callback(MRDeallocCallback cb) noexcept

cdef DevicePtrHandle deviceptr_import_ipc(
    const MemoryPoolHandle& h_pool, const void* export_data, const StreamHandle& h_stream) except+ nogil
cdef StreamHandle deallocation_stream(const DevicePtrHandle& h) noexcept nogil
cdef void set_deallocation_stream(const DevicePtrHandle& h, const StreamHandle& h_stream) noexcept nogil

# Library handles
cdef LibraryHandle create_library_handle_from_file(const char* path) except+ nogil
cdef LibraryHandle create_library_handle_from_data(const void* data) except+ nogil
cdef LibraryHandle create_library_handle_ref(cydriver.CUlibrary library) except+ nogil

# Kernel handles
cdef KernelHandle create_kernel_handle(const LibraryHandle& h_library, const char* name) except+ nogil
cdef KernelHandle create_kernel_handle_ref(
    cydriver.CUkernel kernel, const LibraryHandle& h_library) except+ nogil

# Graphics resource handles
cdef GraphicsResourceHandle create_graphics_resource_handle(
    cydriver.CUgraphicsResource resource) except+ nogil

# NVRTC Program handles
cdef NvrtcProgramHandle create_nvrtc_program_handle(cynvrtc.nvrtcProgram prog) except+ nogil
cdef NvrtcProgramHandle create_nvrtc_program_handle_ref(cynvrtc.nvrtcProgram prog) except+ nogil

# NVVM Program handles
cdef NvvmProgramHandle create_nvvm_program_handle(cynvvm.nvvmProgram prog) except+ nogil
cdef NvvmProgramHandle create_nvvm_program_handle_ref(cynvvm.nvvmProgram prog) except+ nogil
