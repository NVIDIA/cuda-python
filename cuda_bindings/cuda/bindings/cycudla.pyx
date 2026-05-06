# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

# This code was automatically generated with version 1.5.0, generator version 0.3.1.dev1465+gc5c5c8652. Do not modify it directly.

from ._internal cimport cudla as _cudla




###############################################################################
# Wrapper functions
###############################################################################

cdef cudlaStatus cudlaGetVersion(uint64_t* const version) except?_CUDLASTATUS_INTERNAL_LOADING_ERROR nogil:
    return _cudla._cudlaGetVersion(version)


cdef cudlaStatus cudlaDeviceGetCount(uint64_t* const pNumDevices) except?_CUDLASTATUS_INTERNAL_LOADING_ERROR nogil:
    return _cudla._cudlaDeviceGetCount(pNumDevices)


cdef cudlaStatus cudlaCreateDevice(const uint64_t device, cudlaDevHandle* const devHandle, const uint32_t flags) except?_CUDLASTATUS_INTERNAL_LOADING_ERROR nogil:
    return _cudla._cudlaCreateDevice(device, devHandle, flags)


cdef cudlaStatus cudlaMemRegister(const cudlaDevHandle devHandle, const uint64_t* const ptr, const size_t size, uint64_t** const devPtr, const uint32_t flags) except?_CUDLASTATUS_INTERNAL_LOADING_ERROR nogil:
    return _cudla._cudlaMemRegister(devHandle, ptr, size, devPtr, flags)


cdef cudlaStatus cudlaModuleLoadFromMemory(const cudlaDevHandle devHandle, const uint8_t* const pModule, const size_t moduleSize, cudlaModule* const hModule, const uint32_t flags) except?_CUDLASTATUS_INTERNAL_LOADING_ERROR nogil:
    return _cudla._cudlaModuleLoadFromMemory(devHandle, pModule, moduleSize, hModule, flags)


cdef cudlaStatus cudlaModuleGetAttributes(const cudlaModule hModule, const cudlaModuleAttributeType attrType, cudlaModuleAttribute* const attribute) except?_CUDLASTATUS_INTERNAL_LOADING_ERROR nogil:
    return _cudla._cudlaModuleGetAttributes(hModule, attrType, attribute)


cdef cudlaStatus cudlaModuleUnload(const cudlaModule hModule, const uint32_t flags) except?_CUDLASTATUS_INTERNAL_LOADING_ERROR nogil:
    return _cudla._cudlaModuleUnload(hModule, flags)


cdef cudlaStatus cudlaSubmitTask(const cudlaDevHandle devHandle, const cudlaTask* const ptrToTasks, const uint32_t numTasks, void* const stream, const uint32_t flags) except?_CUDLASTATUS_INTERNAL_LOADING_ERROR nogil:
    return _cudla._cudlaSubmitTask(devHandle, ptrToTasks, numTasks, stream, flags)


cdef cudlaStatus cudlaDeviceGetAttribute(const cudlaDevHandle devHandle, const cudlaDevAttributeType attrib, cudlaDevAttribute* const pAttribute) except?_CUDLASTATUS_INTERNAL_LOADING_ERROR nogil:
    return _cudla._cudlaDeviceGetAttribute(devHandle, attrib, pAttribute)


cdef cudlaStatus cudlaMemUnregister(const cudlaDevHandle devHandle, const uint64_t* const devPtr) except?_CUDLASTATUS_INTERNAL_LOADING_ERROR nogil:
    return _cudla._cudlaMemUnregister(devHandle, devPtr)


cdef cudlaStatus cudlaGetLastError(const cudlaDevHandle devHandle) except?_CUDLASTATUS_INTERNAL_LOADING_ERROR nogil:
    return _cudla._cudlaGetLastError(devHandle)


cdef cudlaStatus cudlaDestroyDevice(const cudlaDevHandle devHandle) except?_CUDLASTATUS_INTERNAL_LOADING_ERROR nogil:
    return _cudla._cudlaDestroyDevice(devHandle)


cdef cudlaStatus cudlaSetTaskTimeoutInMs(const cudlaDevHandle devHandle, const uint32_t timeout) except?_CUDLASTATUS_INTERNAL_LOADING_ERROR nogil:
    return _cudla._cudlaSetTaskTimeoutInMs(devHandle, timeout)
