# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

# This code was automatically generated with version 1.5.0, generator version 0.3.1.dev1465+gc5c5c8652. Do not modify it directly.

from ..cycudla cimport *


###############################################################################
# Wrapper functions
###############################################################################

cdef cudlaStatus _cudlaGetVersion(uint64_t* const version) except?_CUDLASTATUS_INTERNAL_LOADING_ERROR nogil
cdef cudlaStatus _cudlaDeviceGetCount(uint64_t* const pNumDevices) except?_CUDLASTATUS_INTERNAL_LOADING_ERROR nogil
cdef cudlaStatus _cudlaCreateDevice(const uint64_t device, cudlaDevHandle* const devHandle, const uint32_t flags) except?_CUDLASTATUS_INTERNAL_LOADING_ERROR nogil
cdef cudlaStatus _cudlaMemRegister(const cudlaDevHandle devHandle, const uint64_t* const ptr, const size_t size, uint64_t** const devPtr, const uint32_t flags) except?_CUDLASTATUS_INTERNAL_LOADING_ERROR nogil
cdef cudlaStatus _cudlaModuleLoadFromMemory(const cudlaDevHandle devHandle, const uint8_t* const pModule, const size_t moduleSize, cudlaModule* const hModule, const uint32_t flags) except?_CUDLASTATUS_INTERNAL_LOADING_ERROR nogil
cdef cudlaStatus _cudlaModuleGetAttributes(const cudlaModule hModule, const cudlaModuleAttributeType attrType, cudlaModuleAttribute* const attribute) except?_CUDLASTATUS_INTERNAL_LOADING_ERROR nogil
cdef cudlaStatus _cudlaModuleUnload(const cudlaModule hModule, const uint32_t flags) except?_CUDLASTATUS_INTERNAL_LOADING_ERROR nogil
cdef cudlaStatus _cudlaSubmitTask(const cudlaDevHandle devHandle, const cudlaTask* const ptrToTasks, const uint32_t numTasks, void* const stream, const uint32_t flags) except?_CUDLASTATUS_INTERNAL_LOADING_ERROR nogil
cdef cudlaStatus _cudlaDeviceGetAttribute(const cudlaDevHandle devHandle, const cudlaDevAttributeType attrib, cudlaDevAttribute* const pAttribute) except?_CUDLASTATUS_INTERNAL_LOADING_ERROR nogil
cdef cudlaStatus _cudlaMemUnregister(const cudlaDevHandle devHandle, const uint64_t* const devPtr) except?_CUDLASTATUS_INTERNAL_LOADING_ERROR nogil
cdef cudlaStatus _cudlaGetLastError(const cudlaDevHandle devHandle) except?_CUDLASTATUS_INTERNAL_LOADING_ERROR nogil
cdef cudlaStatus _cudlaDestroyDevice(const cudlaDevHandle devHandle) except?_CUDLASTATUS_INTERNAL_LOADING_ERROR nogil
cdef cudlaStatus _cudlaSetTaskTimeoutInMs(const cudlaDevHandle devHandle, const uint32_t timeout) except?_CUDLASTATUS_INTERNAL_LOADING_ERROR nogil
