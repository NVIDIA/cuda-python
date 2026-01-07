# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# This module exists to compile _cpp/resource_handles.cpp into a shared library.
# The helper functions (cu, intptr, py) are implemented as inline C++ functions
# in _cpp/resource_handles.hpp and declared as extern in _resource_handles.pxd.

from cpython.pycapsule cimport PyCapsule_New
from libc.stdint cimport uint32_t, uint64_t, uintptr_t

from ._resource_handles_cxx_api cimport (
    ResourceHandlesCxxApiV1,
    get_resource_handles_cxx_api_v1,
)

import cython


cdef const char* _CXX_API_NAME = b"cuda.core._resource_handles._CXX_API"
cdef const char* _CUDA_DRIVER_API_V1_NAME = b"cuda.core._resource_handles._CUDA_DRIVER_API_V1"

# Export the C++ handles dispatch table as a PyCapsule.
# Consumers use PyCapsule_Import(_CXX_API_NAME, 0) to retrieve it.
cdef const ResourceHandlesCxxApiV1* _handles_table = get_resource_handles_cxx_api_v1()
if _handles_table == NULL:
    raise RuntimeError("Failed to initialize resource handles C++ API table")

_CXX_API = <object>PyCapsule_New(<void*>_handles_table, _CXX_API_NAME, NULL)
if _CXX_API is None:
    raise RuntimeError("Failed to create _CXX_API capsule")


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
