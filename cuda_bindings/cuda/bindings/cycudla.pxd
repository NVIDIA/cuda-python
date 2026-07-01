# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

# This code was automatically generated across versions from 1.5.0 to 13.3.0, generator version 0.3.1.dev1779+ga8cc71818.d20260626. Do not modify it directly.
# This layer exposes the C header to Cython as-is.

from libc.stdint cimport int8_t, int16_t, int32_t, int64_t
from libc.stdint cimport uint8_t, uint16_t, uint32_t, uint64_t
from libc.stdint cimport intptr_t, uintptr_t
from libc.stddef cimport size_t




###############################################################################
# Types (structs, enums, ...)
###############################################################################

# enums
cdef extern from 'cudla.h':
    ctypedef enum cudlaStatus:
        cudlaSuccess
        cudlaErrorInvalidParam
        cudlaErrorOutOfResources
        cudlaErrorCreationFailed
        cudlaErrorInvalidAddress
        cudlaErrorOs
        cudlaErrorCuda
        cudlaErrorUmd
        cudlaErrorInvalidDevice
        cudlaErrorInvalidAttribute
        cudlaErrorIncompatibleDlaSWVersion
        cudlaErrorMemoryRegistered
        cudlaErrorInvalidModule
        cudlaErrorUnsupportedOperation
        cudlaErrorNvSci
        cudlaErrorDriverNotFound
        cudlaErrorDlaErrInvalidInput
        cudlaErrorDlaErrInvalidPreAction
        cudlaErrorDlaErrNoMem
        cudlaErrorDlaErrProcessorBusy
        cudlaErrorDlaErrTaskStatusMismatch
        cudlaErrorDlaErrEngineTimeout
        cudlaErrorDlaErrDataMismatch
        cudlaErrorUnknown

cdef extern from 'cudla.h':
    ctypedef enum cudlaMode:
        CUDLA_CUDA_DLA
        CUDLA_STANDALONE

cdef extern from 'cudla.h':
    ctypedef enum cudlaModuleAttributeType:
        CUDLA_NUM_INPUT_TENSORS
        CUDLA_NUM_OUTPUT_TENSORS
        CUDLA_INPUT_TENSOR_DESCRIPTORS
        CUDLA_OUTPUT_TENSOR_DESCRIPTORS
        CUDLA_NUM_OUTPUT_TASK_STATISTICS
        CUDLA_OUTPUT_TASK_STATISTICS_DESCRIPTORS

cdef extern from 'cudla.h':
    ctypedef enum cudlaFenceType:
        CUDLA_NVSCISYNC_FENCE
        CUDLA_NVSCISYNC_FENCE_SOF

cdef extern from 'cudla.h':
    ctypedef enum cudlaModuleLoadFlags:
        CUDLA_MODULE_DEFAULT
        CUDLA_MODULE_ENABLE_FAULT_DIAGNOSTICS

cdef extern from 'cudla.h':
    ctypedef enum cudlaSubmissionFlags:
        CUDLA_SUBMIT_NOOP
        CUDLA_SUBMIT_SKIP_LOCK_ACQUIRE
        CUDLA_SUBMIT_DIAGNOSTICS_TASK

cdef extern from 'cudla.h':
    ctypedef enum cudlaAccessPermissionFlags:
        CUDLA_READ_WRITE_PERM
        CUDLA_READ_ONLY_PERM
        CUDLA_TASK_STATISTICS

cdef extern from 'cudla.h':
    ctypedef enum cudlaDevAttributeType:
        CUDLA_UNIFIED_ADDRESSING
        CUDLA_DEVICE_VERSION
cdef enum: _CUDLASTATUS_INTERNAL_LOADING_ERROR = -42

# types
cdef extern from 'cudla.h':
    ctypedef void* cudlaDevHandle 'cudlaDevHandle'


cdef extern from 'cudla.h':
    ctypedef void* cudlaModule 'cudlaModule'


cdef extern from 'cudla.h':
    ctypedef struct cudlaExternalMemoryHandleDesc_t 'cudlaExternalMemoryHandleDesc_t':
        void* extBufObject
        unsigned long long size

cdef extern from 'cudla.h':
    ctypedef struct cudlaExternalSemaphoreHandleDesc_t 'cudlaExternalSemaphoreHandleDesc_t':
        void* extSyncObject

cdef extern from 'cudla.h':
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

cdef extern from 'cudla.h':
    ctypedef struct CudlaFence 'CudlaFence':
        void* fence
        cudlaFenceType type

cdef extern from 'cudla.h':
    ctypedef union cudlaDevAttribute 'cudlaDevAttribute':
        uint8_t unifiedAddressingSupported
        uint32_t deviceVersion

cdef extern from 'cudla.h':
    ctypedef union cudlaModuleAttribute 'cudlaModuleAttribute':
        uint32_t numInputTensors
        uint32_t numOutputTensors
        cudlaModuleTensorDescriptor* inputTensorDesc
        cudlaModuleTensorDescriptor* outputTensorDesc

cdef extern from 'cudla.h':
    ctypedef struct cudlaWaitEvents 'cudlaWaitEvents':
        CudlaFence* preFences
        uint32_t numEvents

cdef extern from 'cudla.h':
    ctypedef struct cudlaSignalEvents 'cudlaSignalEvents':
        uint64_t** devPtrs
        CudlaFence* eofFences
        uint32_t numEvents

cdef extern from 'cudla.h':
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

cdef cudlaStatus cudlaGetVersion(uint64_t* const version) except?<cudlaStatus>_CUDLASTATUS_INTERNAL_LOADING_ERROR nogil
cdef cudlaStatus cudlaDeviceGetCount(uint64_t* const pNumDevices) except?<cudlaStatus>_CUDLASTATUS_INTERNAL_LOADING_ERROR nogil
cdef cudlaStatus cudlaCreateDevice(const uint64_t device, cudlaDevHandle* const devHandle, const uint32_t flags) except?<cudlaStatus>_CUDLASTATUS_INTERNAL_LOADING_ERROR nogil
cdef cudlaStatus cudlaMemRegister(const cudlaDevHandle devHandle, const uint64_t* const ptr, const size_t size, uint64_t** const devPtr, const uint32_t flags) except?<cudlaStatus>_CUDLASTATUS_INTERNAL_LOADING_ERROR nogil
cdef cudlaStatus cudlaModuleLoadFromMemory(const cudlaDevHandle devHandle, const uint8_t* const pModule, const size_t moduleSize, cudlaModule* const hModule, const uint32_t flags) except?<cudlaStatus>_CUDLASTATUS_INTERNAL_LOADING_ERROR nogil
cdef cudlaStatus cudlaModuleGetAttributes(const cudlaModule hModule, const cudlaModuleAttributeType attrType, cudlaModuleAttribute* const attribute) except?<cudlaStatus>_CUDLASTATUS_INTERNAL_LOADING_ERROR nogil
cdef cudlaStatus cudlaModuleUnload(const cudlaModule hModule, const uint32_t flags) except?<cudlaStatus>_CUDLASTATUS_INTERNAL_LOADING_ERROR nogil
cdef cudlaStatus cudlaSubmitTask(const cudlaDevHandle devHandle, const cudlaTask* const ptrToTasks, const uint32_t numTasks, void* const stream, const uint32_t flags) except?<cudlaStatus>_CUDLASTATUS_INTERNAL_LOADING_ERROR nogil
cdef cudlaStatus cudlaDeviceGetAttribute(const cudlaDevHandle devHandle, const cudlaDevAttributeType attrib, cudlaDevAttribute* const pAttribute) except?<cudlaStatus>_CUDLASTATUS_INTERNAL_LOADING_ERROR nogil
cdef cudlaStatus cudlaMemUnregister(const cudlaDevHandle devHandle, const uint64_t* const devPtr) except?<cudlaStatus>_CUDLASTATUS_INTERNAL_LOADING_ERROR nogil
cdef cudlaStatus cudlaGetLastError(const cudlaDevHandle devHandle) except?<cudlaStatus>_CUDLASTATUS_INTERNAL_LOADING_ERROR nogil
cdef cudlaStatus cudlaDestroyDevice(const cudlaDevHandle devHandle) except?<cudlaStatus>_CUDLASTATUS_INTERNAL_LOADING_ERROR nogil
cdef cudlaStatus cudlaSetTaskTimeoutInMs(const cudlaDevHandle devHandle, const uint32_t timeout) except?<cudlaStatus>_CUDLASTATUS_INTERNAL_LOADING_ERROR nogil
