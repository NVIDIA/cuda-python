# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# This module compiles _cpp/resource_handles.cpp into a shared library.
# Consumer modules cimport the functions declared in _resource_handles.pxd.
# Since there is only one copy of the C++ code (in this .so), all static and
# thread-local state is shared correctly across all consumer modules.
#
# The cdef extern from declarations below satisfy the .pxd declarations directly,
# without needing separate wrapper functions.

from cpython.pycapsule cimport PyCapsule_New
from libc.stddef cimport size_t
from libc.stdint cimport uint32_t, uint64_t, uintptr_t

from cuda.bindings cimport cydriver

from ._resource_handles cimport (
    ContextHandle,
    StreamHandle,
    EventHandle,
    MemoryPoolHandle,
    DevicePtrHandle,
)

import cython


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
        cydriver.CUcontext ctx) noexcept nogil
    ContextHandle get_primary_context "cuda_core::get_primary_context" (
        int device_id) noexcept nogil
    ContextHandle get_current_context "cuda_core::get_current_context" () noexcept nogil

    # Stream handles
    StreamHandle create_stream_handle "cuda_core::create_stream_handle" (
        ContextHandle h_ctx, unsigned int flags, int priority) noexcept nogil
    StreamHandle create_stream_handle_ref "cuda_core::create_stream_handle_ref" (
        cydriver.CUstream stream) noexcept nogil
    StreamHandle create_stream_handle_with_owner "cuda_core::create_stream_handle_with_owner" (
        cydriver.CUstream stream, object owner) noexcept nogil
    StreamHandle get_legacy_stream "cuda_core::get_legacy_stream" () noexcept nogil
    StreamHandle get_per_thread_stream "cuda_core::get_per_thread_stream" () noexcept nogil

    # Event handles (note: _create_event_handle* are internal due to C++ overloading)
    EventHandle create_event_handle "cuda_core::create_event_handle" (
        ContextHandle h_ctx, unsigned int flags) noexcept nogil
    EventHandle create_event_handle_noctx "cuda_core::create_event_handle_noctx" (
        unsigned int flags) noexcept nogil
    EventHandle create_event_handle_ipc "cuda_core::create_event_handle_ipc" (
        const cydriver.CUipcEventHandle& ipc_handle) noexcept nogil

    # Memory pool handles
    MemoryPoolHandle create_mempool_handle "cuda_core::create_mempool_handle" (
        const cydriver.CUmemPoolProps& props) noexcept nogil
    MemoryPoolHandle create_mempool_handle_ref "cuda_core::create_mempool_handle_ref" (
        cydriver.CUmemoryPool pool) noexcept nogil
    MemoryPoolHandle get_device_mempool "cuda_core::get_device_mempool" (
        int device_id) noexcept nogil
    MemoryPoolHandle create_mempool_handle_ipc "cuda_core::create_mempool_handle_ipc" (
        int fd, cydriver.CUmemAllocationHandleType handle_type) noexcept nogil

    # Device pointer handles
    DevicePtrHandle deviceptr_alloc_from_pool "cuda_core::deviceptr_alloc_from_pool" (
        size_t size, MemoryPoolHandle h_pool, StreamHandle h_stream) noexcept nogil
    DevicePtrHandle deviceptr_alloc_async "cuda_core::deviceptr_alloc_async" (
        size_t size, StreamHandle h_stream) noexcept nogil
    DevicePtrHandle deviceptr_alloc "cuda_core::deviceptr_alloc" (size_t size) noexcept nogil
    DevicePtrHandle deviceptr_alloc_host "cuda_core::deviceptr_alloc_host" (size_t size) noexcept nogil
    DevicePtrHandle deviceptr_create_ref "cuda_core::deviceptr_create_ref" (
        cydriver.CUdeviceptr ptr) noexcept nogil
    DevicePtrHandle deviceptr_create_with_owner "cuda_core::deviceptr_create_with_owner" (
        cydriver.CUdeviceptr ptr, object owner) noexcept nogil
    DevicePtrHandle deviceptr_import_ipc "cuda_core::deviceptr_import_ipc" (
        MemoryPoolHandle h_pool, const void* export_data, StreamHandle h_stream) noexcept nogil
    StreamHandle deallocation_stream "cuda_core::deallocation_stream" (
        const DevicePtrHandle& h) noexcept nogil
    void set_deallocation_stream "cuda_core::set_deallocation_stream" (
        const DevicePtrHandle& h, StreamHandle h_stream) noexcept nogil


# =============================================================================
# CUDA Driver API capsule
#
# This provides resolved CUDA driver function pointers to the C++ code.
# =============================================================================

cdef const char* _CUDA_DRIVER_API_V1_NAME = b"cuda.core._resource_handles._CUDA_DRIVER_API_V1"


cdef struct CudaDriverApiV1:
    uint32_t abi_version
    uint32_t struct_size

    uintptr_t cuDevicePrimaryCtxRetain
    uintptr_t cuDevicePrimaryCtxRelease
    uintptr_t cuCtxGetCurrent

    uintptr_t cuStreamCreateWithPriority
    uintptr_t cuStreamDestroy

    uintptr_t cuEventCreate
    uintptr_t cuEventDestroy
    uintptr_t cuIpcOpenEventHandle

    uintptr_t cuDeviceGetCount

    uintptr_t cuMemPoolSetAccess
    uintptr_t cuMemPoolDestroy
    uintptr_t cuMemPoolCreate
    uintptr_t cuDeviceGetMemPool
    uintptr_t cuMemPoolImportFromShareableHandle

    uintptr_t cuMemAllocFromPoolAsync
    uintptr_t cuMemAllocAsync
    uintptr_t cuMemAlloc
    uintptr_t cuMemAllocHost

    uintptr_t cuMemFreeAsync
    uintptr_t cuMemFree
    uintptr_t cuMemFreeHost

    uintptr_t cuMemPoolImportPointer


cdef CudaDriverApiV1 _cuda_driver_api_v1
cdef bint _cuda_driver_api_v1_inited = False


cdef inline uintptr_t _as_addr(object pfn) except 0:
    return <uintptr_t>int(pfn)


cdef inline uintptr_t _resolve(object d, int driver_ver, uint64_t flags, bytes sym) except 0:
    err, pfn, status = d.cuGetProcAddress(sym, driver_ver, flags)
    if int(err) != 0 or pfn is None:
        raise RuntimeError(f"cuGetProcAddress failed for {sym!r}, err={err}, status={status}")
    return _as_addr(pfn)


def _get_cuda_driver_api_v1_capsule():
    """Return a PyCapsule containing cached CUDA driver entrypoints.

    This is evaluated lazily on first use so cuda-core remains importable on
    CPU-only machines.
    """
    global _cuda_driver_api_v1_inited, _cuda_driver_api_v1
    if not _cuda_driver_api_v1_inited:
        import cuda.bindings.driver as d

        err, ver = d.cuDriverGetVersion()
        if int(err) != 0:
            raise RuntimeError(f"cuDriverGetVersion failed: {err}")
        driver_ver = int(ver)

        flags = 0  # CU_GET_PROC_ADDRESS_DEFAULT

        _cuda_driver_api_v1.cuDevicePrimaryCtxRetain = _resolve(d, driver_ver, flags, b"cuDevicePrimaryCtxRetain")
        _cuda_driver_api_v1.cuDevicePrimaryCtxRelease = _resolve(d, driver_ver, flags, b"cuDevicePrimaryCtxRelease")
        _cuda_driver_api_v1.cuCtxGetCurrent = _resolve(d, driver_ver, flags, b"cuCtxGetCurrent")

        _cuda_driver_api_v1.cuStreamCreateWithPriority = _resolve(d, driver_ver, flags, b"cuStreamCreateWithPriority")
        _cuda_driver_api_v1.cuStreamDestroy = _resolve(d, driver_ver, flags, b"cuStreamDestroy")

        _cuda_driver_api_v1.cuEventCreate = _resolve(d, driver_ver, flags, b"cuEventCreate")
        _cuda_driver_api_v1.cuEventDestroy = _resolve(d, driver_ver, flags, b"cuEventDestroy")
        _cuda_driver_api_v1.cuIpcOpenEventHandle = _resolve(d, driver_ver, flags, b"cuIpcOpenEventHandle")

        _cuda_driver_api_v1.cuDeviceGetCount = _resolve(d, driver_ver, flags, b"cuDeviceGetCount")

        _cuda_driver_api_v1.cuMemPoolSetAccess = _resolve(d, driver_ver, flags, b"cuMemPoolSetAccess")
        _cuda_driver_api_v1.cuMemPoolDestroy = _resolve(d, driver_ver, flags, b"cuMemPoolDestroy")
        _cuda_driver_api_v1.cuMemPoolCreate = _resolve(d, driver_ver, flags, b"cuMemPoolCreate")
        _cuda_driver_api_v1.cuDeviceGetMemPool = _resolve(d, driver_ver, flags, b"cuDeviceGetMemPool")
        _cuda_driver_api_v1.cuMemPoolImportFromShareableHandle = _resolve(
            d, driver_ver, flags, b"cuMemPoolImportFromShareableHandle"
        )

        _cuda_driver_api_v1.cuMemAllocFromPoolAsync = _resolve(d, driver_ver, flags, b"cuMemAllocFromPoolAsync")
        _cuda_driver_api_v1.cuMemAllocAsync = _resolve(d, driver_ver, flags, b"cuMemAllocAsync")
        _cuda_driver_api_v1.cuMemAlloc = _resolve(d, driver_ver, flags, b"cuMemAlloc")
        _cuda_driver_api_v1.cuMemAllocHost = _resolve(d, driver_ver, flags, b"cuMemAllocHost")

        _cuda_driver_api_v1.cuMemFreeAsync = _resolve(d, driver_ver, flags, b"cuMemFreeAsync")
        _cuda_driver_api_v1.cuMemFree = _resolve(d, driver_ver, flags, b"cuMemFree")
        _cuda_driver_api_v1.cuMemFreeHost = _resolve(d, driver_ver, flags, b"cuMemFreeHost")

        _cuda_driver_api_v1.cuMemPoolImportPointer = _resolve(d, driver_ver, flags, b"cuMemPoolImportPointer")

        _cuda_driver_api_v1.abi_version = 1
        _cuda_driver_api_v1.struct_size = cython.sizeof(CudaDriverApiV1)
        _cuda_driver_api_v1_inited = True

    return <object>PyCapsule_New(<void*>&_cuda_driver_api_v1, _CUDA_DRIVER_API_V1_NAME, NULL)
