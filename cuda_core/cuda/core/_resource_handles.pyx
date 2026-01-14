# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# This module compiles _cpp/resource_handles.cpp into a shared library.
# At import time, it populates the C++ driver function pointers using
# capsules from cuda.bindings.cydriver.__pyx_capi__.

from cpython.pycapsule cimport PyCapsule_GetName, PyCapsule_GetPointer, PyCapsule_New

from ._resource_handles_cxx_api cimport (
    ResourceHandlesCxxApiV1,
    get_resource_handles_cxx_api_v1,
)

import cuda.bindings.cydriver as cydriver

cdef const char* _CXX_API_NAME = b"cuda.core._resource_handles._CXX_API"

# Export the C++ handles dispatch table as a PyCapsule.
# Consumers use PyCapsule_Import(_CXX_API_NAME, 0) to retrieve it.
cdef const ResourceHandlesCxxApiV1* _handles_table = get_resource_handles_cxx_api_v1()
if _handles_table == NULL:
    raise RuntimeError("Failed to initialize resource handles C++ API table")

_CXX_API = <object>PyCapsule_New(<void*>_handles_table, _CXX_API_NAME, NULL)
if _CXX_API is None:
    raise RuntimeError("Failed to create _CXX_API capsule")


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
