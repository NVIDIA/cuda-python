# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

# This code was automatically generated with version 1.5.0, generator version 0.3.1.dev1465+gc5c5c8652. Do not modify it directly.
# This layer exposes the C header to Cython as-is.

from libc.stdint cimport int8_t, int16_t, int32_t, int64_t
from libc.stdint cimport uint8_t, uint16_t, uint32_t, uint64_t
from libc.stdint cimport intptr_t, uintptr_t
from libc.stddef cimport size_t




###############################################################################
# Types (structs, enums, ...)
###############################################################################

# enums
ctypedef enum cudlaStatus "cudlaStatus":
    cudlaSuccess "cudlaSuccess" = 0
    cudlaErrorInvalidParam "cudlaErrorInvalidParam" = 1
    cudlaErrorOutOfResources "cudlaErrorOutOfResources" = 2
    cudlaErrorCreationFailed "cudlaErrorCreationFailed" = 3
    cudlaErrorInvalidAddress "cudlaErrorInvalidAddress" = 4
    cudlaErrorOs "cudlaErrorOs" = 5
    cudlaErrorCuda "cudlaErrorCuda" = 6
    cudlaErrorUmd "cudlaErrorUmd" = 7
    cudlaErrorInvalidDevice "cudlaErrorInvalidDevice" = 8
    cudlaErrorInvalidAttribute "cudlaErrorInvalidAttribute" = 9
    cudlaErrorIncompatibleDlaSWVersion "cudlaErrorIncompatibleDlaSWVersion" = 10
    cudlaErrorMemoryRegistered "cudlaErrorMemoryRegistered" = 11
    cudlaErrorInvalidModule "cudlaErrorInvalidModule" = 12
    cudlaErrorUnsupportedOperation "cudlaErrorUnsupportedOperation" = 13
    cudlaErrorNvSci "cudlaErrorNvSci" = 14
    cudlaErrorDlaErrInvalidInput "cudlaErrorDlaErrInvalidInput" = 0x40000001
    cudlaErrorDlaErrInvalidPreAction "cudlaErrorDlaErrInvalidPreAction" = 0x40000002
    cudlaErrorDlaErrNoMem "cudlaErrorDlaErrNoMem" = 0x40000003
    cudlaErrorDlaErrProcessorBusy "cudlaErrorDlaErrProcessorBusy" = 0x40000004
    cudlaErrorDlaErrTaskStatusMismatch "cudlaErrorDlaErrTaskStatusMismatch" = 0x40000005
    cudlaErrorDlaErrEngineTimeout "cudlaErrorDlaErrEngineTimeout" = 0x40000006
    cudlaErrorDlaErrDataMismatch "cudlaErrorDlaErrDataMismatch" = 0x40000007
    cudlaErrorUnknown "cudlaErrorUnknown" = 0x7fffffff
    _CUDLASTATUS_INTERNAL_LOADING_ERROR "_CUDLASTATUS_INTERNAL_LOADING_ERROR" = -42

ctypedef enum cudlaMode "cudlaMode":
    CUDLA_CUDA_DLA "CUDLA_CUDA_DLA" = 0
    CUDLA_STANDALONE "CUDLA_STANDALONE" = 1

ctypedef enum cudlaModuleAttributeType "cudlaModuleAttributeType":
    CUDLA_NUM_INPUT_TENSORS "CUDLA_NUM_INPUT_TENSORS" = 0
    CUDLA_NUM_OUTPUT_TENSORS "CUDLA_NUM_OUTPUT_TENSORS" = 1
    CUDLA_INPUT_TENSOR_DESCRIPTORS "CUDLA_INPUT_TENSOR_DESCRIPTORS" = 2
    CUDLA_OUTPUT_TENSOR_DESCRIPTORS "CUDLA_OUTPUT_TENSOR_DESCRIPTORS" = 3
    CUDLA_NUM_OUTPUT_TASK_STATISTICS "CUDLA_NUM_OUTPUT_TASK_STATISTICS" = 4
    CUDLA_OUTPUT_TASK_STATISTICS_DESCRIPTORS "CUDLA_OUTPUT_TASK_STATISTICS_DESCRIPTORS" = 5

ctypedef enum cudlaFenceType "cudlaFenceType":
    CUDLA_NVSCISYNC_FENCE "CUDLA_NVSCISYNC_FENCE" = 1
    CUDLA_NVSCISYNC_FENCE_SOF "CUDLA_NVSCISYNC_FENCE_SOF" = 2

ctypedef enum cudlaModuleLoadFlags "cudlaModuleLoadFlags":
    CUDLA_MODULE_DEFAULT "CUDLA_MODULE_DEFAULT" = 0
    CUDLA_MODULE_ENABLE_FAULT_DIAGNOSTICS "CUDLA_MODULE_ENABLE_FAULT_DIAGNOSTICS" = 1

ctypedef enum cudlaSubmissionFlags "cudlaSubmissionFlags":
    CUDLA_SUBMIT_NOOP "CUDLA_SUBMIT_NOOP" = 1
    CUDLA_SUBMIT_SKIP_LOCK_ACQUIRE "CUDLA_SUBMIT_SKIP_LOCK_ACQUIRE" = (1 << 1)
    CUDLA_SUBMIT_DIAGNOSTICS_TASK "CUDLA_SUBMIT_DIAGNOSTICS_TASK" = (1 << 2)

ctypedef enum cudlaAccessPermissionFlags "cudlaAccessPermissionFlags":
    CUDLA_READ_WRITE_PERM "CUDLA_READ_WRITE_PERM" = 0
    CUDLA_READ_ONLY_PERM "CUDLA_READ_ONLY_PERM" = 1
    CUDLA_TASK_STATISTICS "CUDLA_TASK_STATISTICS" = (1 << 1)

ctypedef enum cudlaDevAttributeType "cudlaDevAttributeType":
    CUDLA_UNIFIED_ADDRESSING "CUDLA_UNIFIED_ADDRESSING" = 0
    CUDLA_DEVICE_VERSION "CUDLA_DEVICE_VERSION" = 1

# types
ctypedef void* cudlaDevHandle 'cudlaDevHandle'

ctypedef void* cudlaModule 'cudlaModule'

ctypedef struct cudlaExternalMemoryHandleDesc_t 'cudlaExternalMemoryHandleDesc_t':
    void* extBufObject
    unsigned long long size

ctypedef struct cudlaExternalSemaphoreHandleDesc_t 'cudlaExternalSemaphoreHandleDesc_t':
    void* extSyncObject

ctypedef struct cudlaModuleTensorDescriptor 'cudlaModuleTensorDescriptor':
    char name[(80U + 1)]
    uint64_t size
    uint64_t n
    uint64_t c
    uint64_t h
    uint64_t w
    uint8_t dataFormat
    uint8_t dataType
    uint8_t dataCategory
    uint8_t pixelFormat
    uint8_t pixelMapping
    uint32_t stride[8U]

ctypedef struct CudlaFence 'CudlaFence':
    void* fence
    cudlaFenceType type

ctypedef union cudlaDevAttribute 'cudlaDevAttribute':
    uint8_t unifiedAddressingSupported
    uint32_t deviceVersion

ctypedef union cudlaModuleAttribute 'cudlaModuleAttribute':
    uint32_t numInputTensors
    uint32_t numOutputTensors
    cudlaModuleTensorDescriptor* inputTensorDesc
    cudlaModuleTensorDescriptor* outputTensorDesc

ctypedef struct cudlaWaitEvents 'cudlaWaitEvents':
    CudlaFence* preFences
    uint32_t numEvents

ctypedef struct cudlaSignalEvents 'cudlaSignalEvents':
    uint64_t** devPtrs
    CudlaFence* eofFences
    uint32_t numEvents

ctypedef struct cudlaTask 'cudlaTask':
    cudlaModule moduleHandle
    uint64_t** outputTensor
    uint32_t numOutputTensors
    uint32_t numInputTensors
    uint64_t** inputTensor
    cudlaWaitEvents* waitEvents
    cudlaSignalEvents* signalEvents

# Typedef aliases for struct types (struct has _t, typedef doesn't)
ctypedef cudlaExternalMemoryHandleDesc_t cudlaExternalMemoryHandleDesc 'cudlaExternalMemoryHandleDesc'
ctypedef cudlaExternalSemaphoreHandleDesc_t cudlaExternalSemaphoreHandleDesc 'cudlaExternalSemaphoreHandleDesc'


###############################################################################
# Functions
###############################################################################

cdef cudlaStatus cudlaGetVersion(uint64_t* const version) except?_CUDLASTATUS_INTERNAL_LOADING_ERROR nogil
cdef cudlaStatus cudlaDeviceGetCount(uint64_t* const pNumDevices) except?_CUDLASTATUS_INTERNAL_LOADING_ERROR nogil
cdef cudlaStatus cudlaCreateDevice(const uint64_t device, cudlaDevHandle* const devHandle, const uint32_t flags) except?_CUDLASTATUS_INTERNAL_LOADING_ERROR nogil
cdef cudlaStatus cudlaMemRegister(const cudlaDevHandle devHandle, const uint64_t* const ptr, const size_t size, uint64_t** const devPtr, const uint32_t flags) except?_CUDLASTATUS_INTERNAL_LOADING_ERROR nogil
cdef cudlaStatus cudlaModuleLoadFromMemory(const cudlaDevHandle devHandle, const uint8_t* const pModule, const size_t moduleSize, cudlaModule* const hModule, const uint32_t flags) except?_CUDLASTATUS_INTERNAL_LOADING_ERROR nogil
cdef cudlaStatus cudlaModuleGetAttributes(const cudlaModule hModule, const cudlaModuleAttributeType attrType, cudlaModuleAttribute* const attribute) except?_CUDLASTATUS_INTERNAL_LOADING_ERROR nogil
cdef cudlaStatus cudlaModuleUnload(const cudlaModule hModule, const uint32_t flags) except?_CUDLASTATUS_INTERNAL_LOADING_ERROR nogil
cdef cudlaStatus cudlaSubmitTask(const cudlaDevHandle devHandle, const cudlaTask* const ptrToTasks, const uint32_t numTasks, void* const stream, const uint32_t flags) except?_CUDLASTATUS_INTERNAL_LOADING_ERROR nogil
cdef cudlaStatus cudlaDeviceGetAttribute(const cudlaDevHandle devHandle, const cudlaDevAttributeType attrib, cudlaDevAttribute* const pAttribute) except?_CUDLASTATUS_INTERNAL_LOADING_ERROR nogil
cdef cudlaStatus cudlaMemUnregister(const cudlaDevHandle devHandle, const uint64_t* const devPtr) except?_CUDLASTATUS_INTERNAL_LOADING_ERROR nogil
cdef cudlaStatus cudlaGetLastError(const cudlaDevHandle devHandle) except?_CUDLASTATUS_INTERNAL_LOADING_ERROR nogil
cdef cudlaStatus cudlaDestroyDevice(const cudlaDevHandle devHandle) except?_CUDLASTATUS_INTERNAL_LOADING_ERROR nogil
cdef cudlaStatus cudlaSetTaskTimeoutInMs(const cudlaDevHandle devHandle, const uint32_t timeout) except?_CUDLASTATUS_INTERNAL_LOADING_ERROR nogil
