# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# This module compiles _cpp/resource_handles.cpp into a shared library.
# Consumer modules cimport the functions declared in _resource_handles.pxd.
# Since there is only one copy of the C++ code (in this .so), all static and
# thread-local state is shared correctly across all consumer modules.
#
# The cdef extern from declarations below satisfy the .pxd declarations directly,
# without needing separate wrapper functions.

from cpython.pycapsule cimport PyCapsule_GetName, PyCapsule_GetPointer
from libc.stddef cimport size_t

from cuda.bindings cimport cydriver
from cuda.bindings cimport cynvrtc
from cuda.bindings cimport cynvvm

from ._resource_handles cimport (
    ContextHandle,
    StreamHandle,
    EventHandle,
    MemoryPoolHandle,
    DevicePtrHandle,
    LibraryHandle,
    KernelHandle,
    GraphicsResourceHandle,
    NvrtcProgramHandle,
    NvvmProgramHandle,
)

import cuda.bindings.cydriver as cydriver
import cuda.bindings.cynvrtc as cynvrtc
import cuda.bindings.cynvvm as cynvvm

# =============================================================================
# C++ function declarations (non-inline, implemented in resource_handles.cpp)
#
# These declarations satisfy the cdef function declarations in _resource_handles.pxd.
# Consumer modules cimport these functions and calls go through this .so.
# =============================================================================

cdef extern from "_cpp/resource_handles.hpp" namespace "cuda_core":
    # Thread-local error handling
    cydriver.CUresult get_last_error "cuda_core::get_last_error" () noexcept nogil
    cydriver.CUresult peek_last_error "cuda_core::peek_last_error" () noexcept nogil
    void clear_last_error "cuda_core::clear_last_error" () noexcept nogil

    # Context handles
    ContextHandle create_context_handle_ref "cuda_core::create_context_handle_ref" (
        cydriver.CUcontext ctx) except+ nogil
    ContextHandle get_primary_context "cuda_core::get_primary_context" (
        int device_id) except+ nogil
    ContextHandle get_current_context "cuda_core::get_current_context" () except+ nogil

    # Stream handles
    StreamHandle create_stream_handle "cuda_core::create_stream_handle" (
        const ContextHandle& h_ctx, unsigned int flags, int priority) except+ nogil
    StreamHandle create_stream_handle_ref "cuda_core::create_stream_handle_ref" (
        cydriver.CUstream stream) except+ nogil
    StreamHandle create_stream_handle_with_owner "cuda_core::create_stream_handle_with_owner" (
        cydriver.CUstream stream, object owner) except+ nogil
    StreamHandle get_legacy_stream "cuda_core::get_legacy_stream" () except+ nogil
    StreamHandle get_per_thread_stream "cuda_core::get_per_thread_stream" () except+ nogil

    # Event handles (note: _create_event_handle* are internal due to C++ overloading)
    EventHandle create_event_handle "cuda_core::create_event_handle" (
        const ContextHandle& h_ctx, unsigned int flags) except+ nogil
    EventHandle create_event_handle_noctx "cuda_core::create_event_handle_noctx" (
        unsigned int flags) except+ nogil
    EventHandle create_event_handle_ipc "cuda_core::create_event_handle_ipc" (
        const cydriver.CUipcEventHandle& ipc_handle) except+ nogil

    # Memory pool handles
    MemoryPoolHandle create_mempool_handle "cuda_core::create_mempool_handle" (
        const cydriver.CUmemPoolProps& props) except+ nogil
    MemoryPoolHandle create_mempool_handle_ref "cuda_core::create_mempool_handle_ref" (
        cydriver.CUmemoryPool pool) except+ nogil
    MemoryPoolHandle get_device_mempool "cuda_core::get_device_mempool" (
        int device_id) except+ nogil
    MemoryPoolHandle create_mempool_handle_ipc "cuda_core::create_mempool_handle_ipc" (
        int fd, cydriver.CUmemAllocationHandleType handle_type) except+ nogil

    # Device pointer handles
    DevicePtrHandle deviceptr_alloc_from_pool "cuda_core::deviceptr_alloc_from_pool" (
        size_t size, const MemoryPoolHandle& h_pool, const StreamHandle& h_stream) except+ nogil
    DevicePtrHandle deviceptr_alloc_async "cuda_core::deviceptr_alloc_async" (
        size_t size, const StreamHandle& h_stream) except+ nogil
    DevicePtrHandle deviceptr_alloc "cuda_core::deviceptr_alloc" (size_t size) except+ nogil
    DevicePtrHandle deviceptr_alloc_host "cuda_core::deviceptr_alloc_host" (size_t size) except+ nogil
    DevicePtrHandle deviceptr_create_ref "cuda_core::deviceptr_create_ref" (
        cydriver.CUdeviceptr ptr) except+ nogil
    DevicePtrHandle deviceptr_create_with_owner "cuda_core::deviceptr_create_with_owner" (
        cydriver.CUdeviceptr ptr, object owner) except+ nogil

    # MR deallocation callback
    ctypedef void (*MRDeallocCallback)(
        object mr, cydriver.CUdeviceptr ptr, size_t size,
        const StreamHandle& stream) noexcept
    void register_mr_dealloc_callback "cuda_core::register_mr_dealloc_callback" (
        MRDeallocCallback cb) noexcept
    DevicePtrHandle deviceptr_create_with_mr "cuda_core::deviceptr_create_with_mr" (
        cydriver.CUdeviceptr ptr, size_t size, object mr) except+ nogil

    DevicePtrHandle deviceptr_import_ipc "cuda_core::deviceptr_import_ipc" (
        const MemoryPoolHandle& h_pool, const void* export_data, const StreamHandle& h_stream) except+ nogil
    StreamHandle deallocation_stream "cuda_core::deallocation_stream" (
        const DevicePtrHandle& h) noexcept nogil
    void set_deallocation_stream "cuda_core::set_deallocation_stream" (
        const DevicePtrHandle& h, const StreamHandle& h_stream) noexcept nogil

    # Library handles
    LibraryHandle create_library_handle_from_file "cuda_core::create_library_handle_from_file" (
        const char* path) except+ nogil
    LibraryHandle create_library_handle_from_data "cuda_core::create_library_handle_from_data" (
        const void* data) except+ nogil
    LibraryHandle create_library_handle_ref "cuda_core::create_library_handle_ref" (
        cydriver.CUlibrary library) except+ nogil

    # Kernel handles
    KernelHandle create_kernel_handle "cuda_core::create_kernel_handle" (
        const LibraryHandle& h_library, const char* name) except+ nogil
    KernelHandle create_kernel_handle_ref "cuda_core::create_kernel_handle_ref" (
        cydriver.CUkernel kernel, const LibraryHandle& h_library) except+ nogil

    # Graphics resource handles
    GraphicsResourceHandle create_graphics_resource_handle "cuda_core::create_graphics_resource_handle" (
        cydriver.CUgraphicsResource resource) except+ nogil

    # NVRTC Program handles
    NvrtcProgramHandle create_nvrtc_program_handle "cuda_core::create_nvrtc_program_handle" (
        cynvrtc.nvrtcProgram prog) except+ nogil
    NvrtcProgramHandle create_nvrtc_program_handle_ref "cuda_core::create_nvrtc_program_handle_ref" (
        cynvrtc.nvrtcProgram prog) except+ nogil

    # NVVM Program handles
    NvvmProgramHandle create_nvvm_program_handle "cuda_core::create_nvvm_program_handle" (
        cynvvm.nvvmProgram prog) except+ nogil
    NvvmProgramHandle create_nvvm_program_handle_ref "cuda_core::create_nvvm_program_handle_ref" (
        cynvvm.nvvmProgram prog) except+ nogil


# =============================================================================
# CUDA Driver API capsule
#
# This provides resolved CUDA driver function pointers to the C++ code.
# =============================================================================

cdef const char* _CUDA_DRIVER_API_V1_NAME = b"cuda.core._resource_handles._CUDA_DRIVER_API_V1"


# =============================================================================
# CUDA driver function pointer initialization
#
# The C++ code declares extern function pointers (p_cuXxx) that need to be
# populated before any handle creation functions are called. We extract these
# from cuda.bindings.cydriver.__pyx_capi__ at module import time.
#
# The Cython string substitution (e.g., "reinterpret_cast<void*&>(...)")
# allows us to assign void* values to typed function pointer variables.
# =============================================================================

# Declare extern variables with reinterpret_cast to allow void* assignment
cdef extern from "_cpp/resource_handles.hpp" namespace "cuda_core":
    # Context
    void* p_cuDevicePrimaryCtxRetain "reinterpret_cast<void*&>(cuda_core::p_cuDevicePrimaryCtxRetain)"
    void* p_cuDevicePrimaryCtxRelease "reinterpret_cast<void*&>(cuda_core::p_cuDevicePrimaryCtxRelease)"
    void* p_cuCtxGetCurrent "reinterpret_cast<void*&>(cuda_core::p_cuCtxGetCurrent)"

    # Stream
    void* p_cuStreamCreateWithPriority "reinterpret_cast<void*&>(cuda_core::p_cuStreamCreateWithPriority)"
    void* p_cuStreamDestroy "reinterpret_cast<void*&>(cuda_core::p_cuStreamDestroy)"

    # Event
    void* p_cuEventCreate "reinterpret_cast<void*&>(cuda_core::p_cuEventCreate)"
    void* p_cuEventDestroy "reinterpret_cast<void*&>(cuda_core::p_cuEventDestroy)"
    void* p_cuIpcOpenEventHandle "reinterpret_cast<void*&>(cuda_core::p_cuIpcOpenEventHandle)"

    # Device
    void* p_cuDeviceGetCount "reinterpret_cast<void*&>(cuda_core::p_cuDeviceGetCount)"

    # Memory pool
    void* p_cuMemPoolSetAccess "reinterpret_cast<void*&>(cuda_core::p_cuMemPoolSetAccess)"
    void* p_cuMemPoolDestroy "reinterpret_cast<void*&>(cuda_core::p_cuMemPoolDestroy)"
    void* p_cuMemPoolCreate "reinterpret_cast<void*&>(cuda_core::p_cuMemPoolCreate)"
    void* p_cuDeviceGetMemPool "reinterpret_cast<void*&>(cuda_core::p_cuDeviceGetMemPool)"
    void* p_cuMemPoolImportFromShareableHandle "reinterpret_cast<void*&>(cuda_core::p_cuMemPoolImportFromShareableHandle)"

    # Memory allocation
    void* p_cuMemAllocFromPoolAsync "reinterpret_cast<void*&>(cuda_core::p_cuMemAllocFromPoolAsync)"
    void* p_cuMemAllocAsync "reinterpret_cast<void*&>(cuda_core::p_cuMemAllocAsync)"
    void* p_cuMemAlloc "reinterpret_cast<void*&>(cuda_core::p_cuMemAlloc)"
    void* p_cuMemAllocHost "reinterpret_cast<void*&>(cuda_core::p_cuMemAllocHost)"

    # Memory deallocation
    void* p_cuMemFreeAsync "reinterpret_cast<void*&>(cuda_core::p_cuMemFreeAsync)"
    void* p_cuMemFree "reinterpret_cast<void*&>(cuda_core::p_cuMemFree)"
    void* p_cuMemFreeHost "reinterpret_cast<void*&>(cuda_core::p_cuMemFreeHost)"

    # IPC
    void* p_cuMemPoolImportPointer "reinterpret_cast<void*&>(cuda_core::p_cuMemPoolImportPointer)"

    # Library
    void* p_cuLibraryLoadFromFile "reinterpret_cast<void*&>(cuda_core::p_cuLibraryLoadFromFile)"
    void* p_cuLibraryLoadData "reinterpret_cast<void*&>(cuda_core::p_cuLibraryLoadData)"
    void* p_cuLibraryUnload "reinterpret_cast<void*&>(cuda_core::p_cuLibraryUnload)"
    void* p_cuLibraryGetKernel "reinterpret_cast<void*&>(cuda_core::p_cuLibraryGetKernel)"

    # Graphics interop
    void* p_cuGraphicsUnregisterResource "reinterpret_cast<void*&>(cuda_core::p_cuGraphicsUnregisterResource)"

    # NVRTC
    void* p_nvrtcDestroyProgram "reinterpret_cast<void*&>(cuda_core::p_nvrtcDestroyProgram)"

    # NVVM
    void* p_nvvmDestroyProgram "reinterpret_cast<void*&>(cuda_core::p_nvvmDestroyProgram)"


# Initialize driver function pointers from cydriver.__pyx_capi__ at module load
cdef void* _get_driver_fn(str name):
    capsule = cydriver.__pyx_capi__[name]
    return PyCapsule_GetPointer(capsule, PyCapsule_GetName(capsule))

# Context
p_cuDevicePrimaryCtxRetain = _get_driver_fn("cuDevicePrimaryCtxRetain")
p_cuDevicePrimaryCtxRelease = _get_driver_fn("cuDevicePrimaryCtxRelease")
p_cuCtxGetCurrent = _get_driver_fn("cuCtxGetCurrent")

# Stream
p_cuStreamCreateWithPriority = _get_driver_fn("cuStreamCreateWithPriority")
p_cuStreamDestroy = _get_driver_fn("cuStreamDestroy")

# Event
p_cuEventCreate = _get_driver_fn("cuEventCreate")
p_cuEventDestroy = _get_driver_fn("cuEventDestroy")
p_cuIpcOpenEventHandle = _get_driver_fn("cuIpcOpenEventHandle")

# Device
p_cuDeviceGetCount = _get_driver_fn("cuDeviceGetCount")

# Memory pool
p_cuMemPoolSetAccess = _get_driver_fn("cuMemPoolSetAccess")
p_cuMemPoolDestroy = _get_driver_fn("cuMemPoolDestroy")
p_cuMemPoolCreate = _get_driver_fn("cuMemPoolCreate")
p_cuDeviceGetMemPool = _get_driver_fn("cuDeviceGetMemPool")
p_cuMemPoolImportFromShareableHandle = _get_driver_fn("cuMemPoolImportFromShareableHandle")

# Memory allocation
p_cuMemAllocFromPoolAsync = _get_driver_fn("cuMemAllocFromPoolAsync")
p_cuMemAllocAsync = _get_driver_fn("cuMemAllocAsync")
p_cuMemAlloc = _get_driver_fn("cuMemAlloc")
p_cuMemAllocHost = _get_driver_fn("cuMemAllocHost")

# Memory deallocation
p_cuMemFreeAsync = _get_driver_fn("cuMemFreeAsync")
p_cuMemFree = _get_driver_fn("cuMemFree")
p_cuMemFreeHost = _get_driver_fn("cuMemFreeHost")

# IPC
p_cuMemPoolImportPointer = _get_driver_fn("cuMemPoolImportPointer")

# Library
p_cuLibraryLoadFromFile = _get_driver_fn("cuLibraryLoadFromFile")
p_cuLibraryLoadData = _get_driver_fn("cuLibraryLoadData")
p_cuLibraryUnload = _get_driver_fn("cuLibraryUnload")
p_cuLibraryGetKernel = _get_driver_fn("cuLibraryGetKernel")

# Graphics interop
p_cuGraphicsUnregisterResource = _get_driver_fn("cuGraphicsUnregisterResource")

# =============================================================================
# NVRTC function pointer initialization
# =============================================================================

cdef void* _get_nvrtc_fn(str name):
    capsule = cynvrtc.__pyx_capi__[name]
    return PyCapsule_GetPointer(capsule, PyCapsule_GetName(capsule))

p_nvrtcDestroyProgram = _get_nvrtc_fn("nvrtcDestroyProgram")

# =============================================================================
# NVVM function pointer initialization
#
# NVVM may not be available at runtime, so we handle missing function pointers
# gracefully. The C++ deleter checks for null before calling.
# =============================================================================

cdef void* _get_nvvm_fn(str name):
    capsule = cynvvm.__pyx_capi__[name]
    return PyCapsule_GetPointer(capsule, PyCapsule_GetName(capsule))

p_nvvmDestroyProgram = _get_nvvm_fn("nvvmDestroyProgram")
