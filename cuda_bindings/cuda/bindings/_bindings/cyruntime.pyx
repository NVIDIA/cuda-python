# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

# This code was automatically generated with version 13.3.0, generator version 0.3.1.dev1630+gadce055ea.d20260422. Do not modify it directly.
include "../cyruntime_functions.pxi"

import os
cimport cuda.bindings._bindings.cyruntime_ptds as ptds
cimport cython

cdef bint __cudaPythonInit = False
cdef bint __usePTDS = False
cdef int _cudaPythonInit() except -1 nogil:
        global __cudaPythonInit
        global __usePTDS

        with gil:
            __usePTDS = bool(int(os.getenv('CUDA_PYTHON_CUDA_PER_THREAD_DEFAULT_STREAM', default=0)))
        __cudaPythonInit = True
        return __usePTDS

# Create a very small function to check whether we are init'ed, so the C
# compiler can inline it.
cdef inline int cudaPythonInit() except -1 nogil:
    if __cudaPythonInit:
        return __usePTDS
    return _cudaPythonInit()

cdef cudaError_t _cudaDeviceReset() except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaDeviceReset()
    return cudaDeviceReset()

cdef cudaError_t _cudaDeviceSynchronize() except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaDeviceSynchronize()
    return cudaDeviceSynchronize()

cdef cudaError_t _cudaDeviceSetLimit(cudaLimit limit, size_t value) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaDeviceSetLimit(limit, value)
    return cudaDeviceSetLimit(limit, value)

cdef cudaError_t _cudaDeviceGetLimit(size_t* pValue, cudaLimit limit) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaDeviceGetLimit(pValue, limit)
    return cudaDeviceGetLimit(pValue, limit)

cdef cudaError_t _cudaDeviceGetTexture1DLinearMaxWidth(size_t* maxWidthInElements, const cudaChannelFormatDesc* fmtDesc, int device) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaDeviceGetTexture1DLinearMaxWidth(maxWidthInElements, fmtDesc, device)
    return cudaDeviceGetTexture1DLinearMaxWidth(maxWidthInElements, fmtDesc, device)

cdef cudaError_t _cudaDeviceGetCacheConfig(cudaFuncCache* pCacheConfig) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaDeviceGetCacheConfig(pCacheConfig)
    return cudaDeviceGetCacheConfig(pCacheConfig)

cdef cudaError_t _cudaDeviceGetStreamPriorityRange(int* leastPriority, int* greatestPriority) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaDeviceGetStreamPriorityRange(leastPriority, greatestPriority)
    return cudaDeviceGetStreamPriorityRange(leastPriority, greatestPriority)

cdef cudaError_t _cudaDeviceSetCacheConfig(cudaFuncCache cacheConfig) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaDeviceSetCacheConfig(cacheConfig)
    return cudaDeviceSetCacheConfig(cacheConfig)

cdef cudaError_t _cudaDeviceGetByPCIBusId(int* device, const char* pciBusId) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaDeviceGetByPCIBusId(device, pciBusId)
    return cudaDeviceGetByPCIBusId(device, pciBusId)

cdef cudaError_t _cudaDeviceGetPCIBusId(char* pciBusId, int length, int device) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaDeviceGetPCIBusId(pciBusId, length, device)
    return cudaDeviceGetPCIBusId(pciBusId, length, device)

cdef cudaError_t _cudaIpcGetEventHandle(cudaIpcEventHandle_t* handle, cudaEvent_t event) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaIpcGetEventHandle(handle, event)
    return cudaIpcGetEventHandle(handle, event)

cdef cudaError_t _cudaIpcOpenEventHandle(cudaEvent_t* event, cudaIpcEventHandle_t handle) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaIpcOpenEventHandle(event, handle)
    return cudaIpcOpenEventHandle(event, handle)

cdef cudaError_t _cudaIpcGetMemHandle(cudaIpcMemHandle_t* handle, void* devPtr) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaIpcGetMemHandle(handle, devPtr)
    return cudaIpcGetMemHandle(handle, devPtr)

cdef cudaError_t _cudaIpcOpenMemHandle(void** devPtr, cudaIpcMemHandle_t handle, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaIpcOpenMemHandle(devPtr, handle, flags)
    return cudaIpcOpenMemHandle(devPtr, handle, flags)

cdef cudaError_t _cudaIpcCloseMemHandle(void* devPtr) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaIpcCloseMemHandle(devPtr)
    return cudaIpcCloseMemHandle(devPtr)

cdef cudaError_t _cudaDeviceFlushGPUDirectRDMAWrites(cudaFlushGPUDirectRDMAWritesTarget target, cudaFlushGPUDirectRDMAWritesScope scope) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaDeviceFlushGPUDirectRDMAWrites(target, scope)
    return cudaDeviceFlushGPUDirectRDMAWrites(target, scope)

cdef cudaError_t _cudaDeviceRegisterAsyncNotification(int device, cudaAsyncCallback callbackFunc, void* userData, cudaAsyncCallbackHandle_t* callback) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaDeviceRegisterAsyncNotification(device, callbackFunc, userData, callback)
    return cudaDeviceRegisterAsyncNotification(device, callbackFunc, userData, callback)

cdef cudaError_t _cudaDeviceUnregisterAsyncNotification(int device, cudaAsyncCallbackHandle_t callback) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaDeviceUnregisterAsyncNotification(device, callback)
    return cudaDeviceUnregisterAsyncNotification(device, callback)

cdef cudaError_t _cudaDeviceGetSharedMemConfig(cudaSharedMemConfig* pConfig) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaDeviceGetSharedMemConfig(pConfig)
    return cudaDeviceGetSharedMemConfig(pConfig)

cdef cudaError_t _cudaDeviceSetSharedMemConfig(cudaSharedMemConfig config) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaDeviceSetSharedMemConfig(config)
    return cudaDeviceSetSharedMemConfig(config)

cdef cudaError_t _cudaGetLastError() except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaGetLastError()
    return cudaGetLastError()

cdef cudaError_t _cudaPeekAtLastError() except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaPeekAtLastError()
    return cudaPeekAtLastError()

cdef const char* _cudaGetErrorName(cudaError_t error) except ?NULL nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaGetErrorName(error)
    return cudaGetErrorName(error)

cdef const char* _cudaGetErrorString(cudaError_t error) except ?NULL nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaGetErrorString(error)
    return cudaGetErrorString(error)

cdef cudaError_t _cudaGetDeviceCount(int* count) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaGetDeviceCount(count)
    return cudaGetDeviceCount(count)

cdef cudaError_t _cudaGetDeviceProperties(cudaDeviceProp* prop, int device) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaGetDeviceProperties(prop, device)
    return cudaGetDeviceProperties(prop, device)

cdef cudaError_t _cudaDeviceGetAttribute(int* value, cudaDeviceAttr attr, int device) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaDeviceGetAttribute(value, attr, device)
    return cudaDeviceGetAttribute(value, attr, device)

cdef cudaError_t _cudaDeviceGetHostAtomicCapabilities(unsigned int* capabilities, const cudaAtomicOperation* operations, unsigned int count, int device) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaDeviceGetHostAtomicCapabilities(capabilities, operations, count, device)
    return cudaDeviceGetHostAtomicCapabilities(capabilities, operations, count, device)

cdef cudaError_t _cudaDeviceGetDefaultMemPool(cudaMemPool_t* memPool, int device) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaDeviceGetDefaultMemPool(memPool, device)
    return cudaDeviceGetDefaultMemPool(memPool, device)

cdef cudaError_t _cudaDeviceSetMemPool(int device, cudaMemPool_t memPool) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaDeviceSetMemPool(device, memPool)
    return cudaDeviceSetMemPool(device, memPool)

cdef cudaError_t _cudaDeviceGetMemPool(cudaMemPool_t* memPool, int device) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaDeviceGetMemPool(memPool, device)
    return cudaDeviceGetMemPool(memPool, device)

cdef cudaError_t _cudaDeviceGetNvSciSyncAttributes(void* nvSciSyncAttrList, int device, int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaDeviceGetNvSciSyncAttributes(nvSciSyncAttrList, device, flags)
    return cudaDeviceGetNvSciSyncAttributes(nvSciSyncAttrList, device, flags)

cdef cudaError_t _cudaDeviceGetP2PAttribute(int* value, cudaDeviceP2PAttr attr, int srcDevice, int dstDevice) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaDeviceGetP2PAttribute(value, attr, srcDevice, dstDevice)
    return cudaDeviceGetP2PAttribute(value, attr, srcDevice, dstDevice)

cdef cudaError_t _cudaDeviceGetP2PAtomicCapabilities(unsigned int* capabilities, const cudaAtomicOperation* operations, unsigned int count, int srcDevice, int dstDevice) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaDeviceGetP2PAtomicCapabilities(capabilities, operations, count, srcDevice, dstDevice)
    return cudaDeviceGetP2PAtomicCapabilities(capabilities, operations, count, srcDevice, dstDevice)

cdef cudaError_t _cudaChooseDevice(int* device, const cudaDeviceProp* prop) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaChooseDevice(device, prop)
    return cudaChooseDevice(device, prop)

cdef cudaError_t _cudaInitDevice(int device, unsigned int deviceFlags, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaInitDevice(device, deviceFlags, flags)
    return cudaInitDevice(device, deviceFlags, flags)

cdef cudaError_t _cudaSetDevice(int device) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaSetDevice(device)
    return cudaSetDevice(device)

cdef cudaError_t _cudaGetDevice(int* device) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaGetDevice(device)
    return cudaGetDevice(device)

cdef cudaError_t _cudaSetDeviceFlags(unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaSetDeviceFlags(flags)
    return cudaSetDeviceFlags(flags)

cdef cudaError_t _cudaGetDeviceFlags(unsigned int* flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaGetDeviceFlags(flags)
    return cudaGetDeviceFlags(flags)

cdef cudaError_t _cudaStreamCreate(cudaStream_t* pStream) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaStreamCreate(pStream)
    return cudaStreamCreate(pStream)

cdef cudaError_t _cudaStreamCreateWithFlags(cudaStream_t* pStream, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaStreamCreateWithFlags(pStream, flags)
    return cudaStreamCreateWithFlags(pStream, flags)

cdef cudaError_t _cudaStreamCreateWithPriority(cudaStream_t* pStream, unsigned int flags, int priority) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaStreamCreateWithPriority(pStream, flags, priority)
    return cudaStreamCreateWithPriority(pStream, flags, priority)

cdef cudaError_t _cudaStreamGetPriority(cudaStream_t hStream, int* priority) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaStreamGetPriority(hStream, priority)
    return cudaStreamGetPriority(hStream, priority)

cdef cudaError_t _cudaStreamGetFlags(cudaStream_t hStream, unsigned int* flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaStreamGetFlags(hStream, flags)
    return cudaStreamGetFlags(hStream, flags)

cdef cudaError_t _cudaStreamGetId(cudaStream_t hStream, unsigned long long* streamId) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaStreamGetId(hStream, streamId)
    return cudaStreamGetId(hStream, streamId)

cdef cudaError_t _cudaStreamGetDevice(cudaStream_t hStream, int* device) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaStreamGetDevice(hStream, device)
    return cudaStreamGetDevice(hStream, device)

cdef cudaError_t _cudaCtxResetPersistingL2Cache() except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaCtxResetPersistingL2Cache()
    return cudaCtxResetPersistingL2Cache()

cdef cudaError_t _cudaStreamCopyAttributes(cudaStream_t dst, cudaStream_t src) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaStreamCopyAttributes(dst, src)
    return cudaStreamCopyAttributes(dst, src)

cdef cudaError_t _cudaStreamGetAttribute(cudaStream_t hStream, cudaStreamAttrID attr, cudaStreamAttrValue* value_out) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaStreamGetAttribute(hStream, attr, value_out)
    return cudaStreamGetAttribute(hStream, attr, value_out)

cdef cudaError_t _cudaStreamSetAttribute(cudaStream_t hStream, cudaStreamAttrID attr, const cudaStreamAttrValue* value) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaStreamSetAttribute(hStream, attr, value)
    return cudaStreamSetAttribute(hStream, attr, value)

cdef cudaError_t _cudaStreamDestroy(cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaStreamDestroy(stream)
    return cudaStreamDestroy(stream)

cdef cudaError_t _cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaStreamWaitEvent(stream, event, flags)
    return cudaStreamWaitEvent(stream, event, flags)

cdef cudaError_t _cudaStreamAddCallback(cudaStream_t stream, cudaStreamCallback_t callback, void* userData, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaStreamAddCallback(stream, callback, userData, flags)
    return cudaStreamAddCallback(stream, callback, userData, flags)

cdef cudaError_t _cudaStreamSynchronize(cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaStreamSynchronize(stream)
    return cudaStreamSynchronize(stream)

cdef cudaError_t _cudaStreamQuery(cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaStreamQuery(stream)
    return cudaStreamQuery(stream)

cdef cudaError_t _cudaStreamAttachMemAsync(cudaStream_t stream, void* devPtr, size_t length, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaStreamAttachMemAsync(stream, devPtr, length, flags)
    return cudaStreamAttachMemAsync(stream, devPtr, length, flags)

cdef cudaError_t _cudaStreamBeginCapture(cudaStream_t stream, cudaStreamCaptureMode mode) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaStreamBeginCapture(stream, mode)
    return cudaStreamBeginCapture(stream, mode)

cdef cudaError_t _cudaStreamBeginRecaptureToGraph(cudaStream_t stream, cudaStreamCaptureMode mode, cudaGraph_t graph, cudaGraphRecaptureCallbackData* callbackData) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaStreamBeginRecaptureToGraph(stream, mode, graph, callbackData)
    return cudaStreamBeginRecaptureToGraph(stream, mode, graph, callbackData)

cdef cudaError_t _cudaStreamBeginCaptureToGraph(cudaStream_t stream, cudaGraph_t graph, const cudaGraphNode_t* dependencies, const cudaGraphEdgeData* dependencyData, size_t numDependencies, cudaStreamCaptureMode mode) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaStreamBeginCaptureToGraph(stream, graph, dependencies, dependencyData, numDependencies, mode)
    return cudaStreamBeginCaptureToGraph(stream, graph, dependencies, dependencyData, numDependencies, mode)

cdef cudaError_t _cudaThreadExchangeStreamCaptureMode(cudaStreamCaptureMode* mode) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaThreadExchangeStreamCaptureMode(mode)
    return cudaThreadExchangeStreamCaptureMode(mode)

cdef cudaError_t _cudaStreamEndCapture(cudaStream_t stream, cudaGraph_t* pGraph) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaStreamEndCapture(stream, pGraph)
    return cudaStreamEndCapture(stream, pGraph)

cdef cudaError_t _cudaStreamIsCapturing(cudaStream_t stream, cudaStreamCaptureStatus* pCaptureStatus) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaStreamIsCapturing(stream, pCaptureStatus)
    return cudaStreamIsCapturing(stream, pCaptureStatus)

cdef cudaError_t _cudaStreamGetCaptureInfo(cudaStream_t stream, cudaStreamCaptureStatus* captureStatus_out, unsigned long long* id_out, cudaGraph_t* graph_out, const cudaGraphNode_t** dependencies_out, const cudaGraphEdgeData** edgeData_out, size_t* numDependencies_out) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaStreamGetCaptureInfo(stream, captureStatus_out, id_out, graph_out, dependencies_out, edgeData_out, numDependencies_out)
    return cudaStreamGetCaptureInfo(stream, captureStatus_out, id_out, graph_out, dependencies_out, edgeData_out, numDependencies_out)

cdef cudaError_t _cudaStreamUpdateCaptureDependencies(cudaStream_t stream, cudaGraphNode_t* dependencies, const cudaGraphEdgeData* dependencyData, size_t numDependencies, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaStreamUpdateCaptureDependencies(stream, dependencies, dependencyData, numDependencies, flags)
    return cudaStreamUpdateCaptureDependencies(stream, dependencies, dependencyData, numDependencies, flags)

cdef cudaError_t _cudaEventCreate(cudaEvent_t* event) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaEventCreate(event)
    return cudaEventCreate(event)

cdef cudaError_t _cudaEventCreateWithFlags(cudaEvent_t* event, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaEventCreateWithFlags(event, flags)
    return cudaEventCreateWithFlags(event, flags)

cdef cudaError_t _cudaEventRecord(cudaEvent_t event, cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaEventRecord(event, stream)
    return cudaEventRecord(event, stream)

cdef cudaError_t _cudaEventRecordWithFlags(cudaEvent_t event, cudaStream_t stream, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaEventRecordWithFlags(event, stream, flags)
    return cudaEventRecordWithFlags(event, stream, flags)

cdef cudaError_t _cudaEventQuery(cudaEvent_t event) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaEventQuery(event)
    return cudaEventQuery(event)

cdef cudaError_t _cudaEventSynchronize(cudaEvent_t event) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaEventSynchronize(event)
    return cudaEventSynchronize(event)

cdef cudaError_t _cudaEventDestroy(cudaEvent_t event) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaEventDestroy(event)
    return cudaEventDestroy(event)

cdef cudaError_t _cudaEventElapsedTime(float* ms, cudaEvent_t start, cudaEvent_t end) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaEventElapsedTime(ms, start, end)
    return cudaEventElapsedTime(ms, start, end)

cdef cudaError_t _cudaImportExternalMemory(cudaExternalMemory_t* extMem_out, const cudaExternalMemoryHandleDesc* memHandleDesc) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaImportExternalMemory(extMem_out, memHandleDesc)
    return cudaImportExternalMemory(extMem_out, memHandleDesc)

cdef cudaError_t _cudaExternalMemoryGetMappedBuffer(void** devPtr, cudaExternalMemory_t extMem, const cudaExternalMemoryBufferDesc* bufferDesc) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaExternalMemoryGetMappedBuffer(devPtr, extMem, bufferDesc)
    return cudaExternalMemoryGetMappedBuffer(devPtr, extMem, bufferDesc)

cdef cudaError_t _cudaExternalMemoryGetMappedMipmappedArray(cudaMipmappedArray_t* mipmap, cudaExternalMemory_t extMem, const cudaExternalMemoryMipmappedArrayDesc* mipmapDesc) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaExternalMemoryGetMappedMipmappedArray(mipmap, extMem, mipmapDesc)
    return cudaExternalMemoryGetMappedMipmappedArray(mipmap, extMem, mipmapDesc)

cdef cudaError_t _cudaDestroyExternalMemory(cudaExternalMemory_t extMem) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaDestroyExternalMemory(extMem)
    return cudaDestroyExternalMemory(extMem)

cdef cudaError_t _cudaImportExternalSemaphore(cudaExternalSemaphore_t* extSem_out, const cudaExternalSemaphoreHandleDesc* semHandleDesc) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaImportExternalSemaphore(extSem_out, semHandleDesc)
    return cudaImportExternalSemaphore(extSem_out, semHandleDesc)

cdef cudaError_t _cudaSignalExternalSemaphoresAsync(const cudaExternalSemaphore_t* extSemArray, const cudaExternalSemaphoreSignalParams* paramsArray, unsigned int numExtSems, cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaSignalExternalSemaphoresAsync(extSemArray, paramsArray, numExtSems, stream)
    return cudaSignalExternalSemaphoresAsync(extSemArray, paramsArray, numExtSems, stream)

cdef cudaError_t _cudaWaitExternalSemaphoresAsync(const cudaExternalSemaphore_t* extSemArray, const cudaExternalSemaphoreWaitParams* paramsArray, unsigned int numExtSems, cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaWaitExternalSemaphoresAsync(extSemArray, paramsArray, numExtSems, stream)
    return cudaWaitExternalSemaphoresAsync(extSemArray, paramsArray, numExtSems, stream)

cdef cudaError_t _cudaDestroyExternalSemaphore(cudaExternalSemaphore_t extSem) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaDestroyExternalSemaphore(extSem)
    return cudaDestroyExternalSemaphore(extSem)

cdef cudaError_t _cudaFuncSetCacheConfig(const void* func, cudaFuncCache cacheConfig) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaFuncSetCacheConfig(func, cacheConfig)
    return cudaFuncSetCacheConfig(func, cacheConfig)

cdef cudaError_t _cudaFuncGetAttributes(cudaFuncAttributes* attr, const void* func) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaFuncGetAttributes(attr, func)
    return cudaFuncGetAttributes(attr, func)

cdef cudaError_t _cudaFuncSetAttribute(const void* func, cudaFuncAttribute attr, int value) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaFuncSetAttribute(func, attr, value)
    return cudaFuncSetAttribute(func, attr, value)

cdef cudaError_t _cudaFuncGetParamCount(const void* func, size_t* paramCount) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaFuncGetParamCount(func, paramCount)
    return cudaFuncGetParamCount(func, paramCount)

cdef cudaError_t _cudaLaunchHostFunc(cudaStream_t stream, cudaHostFn_t fn, void* userData) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaLaunchHostFunc(stream, fn, userData)
    return cudaLaunchHostFunc(stream, fn, userData)

cdef cudaError_t _cudaLaunchHostFunc_v2(cudaStream_t stream, cudaHostFn_t fn, void* userData, unsigned int syncMode) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaLaunchHostFunc_v2(stream, fn, userData, syncMode)
    return cudaLaunchHostFunc_v2(stream, fn, userData, syncMode)

cdef cudaError_t _cudaFuncSetSharedMemConfig(const void* func, cudaSharedMemConfig config) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaFuncSetSharedMemConfig(func, config)
    return cudaFuncSetSharedMemConfig(func, config)

cdef cudaError_t _cudaOccupancyMaxActiveBlocksPerMultiprocessor(int* numBlocks, const void* func, int blockSize, size_t dynamicSMemSize) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaOccupancyMaxActiveBlocksPerMultiprocessor(numBlocks, func, blockSize, dynamicSMemSize)
    return cudaOccupancyMaxActiveBlocksPerMultiprocessor(numBlocks, func, blockSize, dynamicSMemSize)

cdef cudaError_t _cudaOccupancyAvailableDynamicSMemPerBlock(size_t* dynamicSmemSize, const void* func, int numBlocks, int blockSize) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaOccupancyAvailableDynamicSMemPerBlock(dynamicSmemSize, func, numBlocks, blockSize)
    return cudaOccupancyAvailableDynamicSMemPerBlock(dynamicSmemSize, func, numBlocks, blockSize)

cdef cudaError_t _cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int* numBlocks, const void* func, int blockSize, size_t dynamicSMemSize, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(numBlocks, func, blockSize, dynamicSMemSize, flags)
    return cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(numBlocks, func, blockSize, dynamicSMemSize, flags)

cdef cudaError_t _cudaMallocManaged(void** devPtr, size_t size, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaMallocManaged(devPtr, size, flags)
    return cudaMallocManaged(devPtr, size, flags)

cdef cudaError_t _cudaMalloc(void** devPtr, size_t size) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaMalloc(devPtr, size)
    return cudaMalloc(devPtr, size)

cdef cudaError_t _cudaMallocHost(void** ptr, size_t size) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaMallocHost(ptr, size)
    return cudaMallocHost(ptr, size)

cdef cudaError_t _cudaMallocPitch(void** devPtr, size_t* pitch, size_t width, size_t height) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaMallocPitch(devPtr, pitch, width, height)
    return cudaMallocPitch(devPtr, pitch, width, height)

cdef cudaError_t _cudaMallocArray(cudaArray_t* array, const cudaChannelFormatDesc* desc, size_t width, size_t height, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaMallocArray(array, desc, width, height, flags)
    return cudaMallocArray(array, desc, width, height, flags)

cdef cudaError_t _cudaFree(void* devPtr) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaFree(devPtr)
    return cudaFree(devPtr)

cdef cudaError_t _cudaFreeHost(void* ptr) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaFreeHost(ptr)
    return cudaFreeHost(ptr)

cdef cudaError_t _cudaFreeArray(cudaArray_t array) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaFreeArray(array)
    return cudaFreeArray(array)

cdef cudaError_t _cudaFreeMipmappedArray(cudaMipmappedArray_t mipmappedArray) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaFreeMipmappedArray(mipmappedArray)
    return cudaFreeMipmappedArray(mipmappedArray)

cdef cudaError_t _cudaHostAlloc(void** pHost, size_t size, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaHostAlloc(pHost, size, flags)
    return cudaHostAlloc(pHost, size, flags)

cdef cudaError_t _cudaHostRegister(void* ptr, size_t size, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaHostRegister(ptr, size, flags)
    return cudaHostRegister(ptr, size, flags)

cdef cudaError_t _cudaHostUnregister(void* ptr) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaHostUnregister(ptr)
    return cudaHostUnregister(ptr)

cdef cudaError_t _cudaHostGetDevicePointer(void** pDevice, void* pHost, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaHostGetDevicePointer(pDevice, pHost, flags)
    return cudaHostGetDevicePointer(pDevice, pHost, flags)

cdef cudaError_t _cudaHostGetFlags(unsigned int* pFlags, void* pHost) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaHostGetFlags(pFlags, pHost)
    return cudaHostGetFlags(pFlags, pHost)

cdef cudaError_t _cudaMalloc3D(cudaPitchedPtr* pitchedDevPtr, cudaExtent extent) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaMalloc3D(pitchedDevPtr, extent)
    return cudaMalloc3D(pitchedDevPtr, extent)

cdef cudaError_t _cudaMalloc3DArray(cudaArray_t* array, const cudaChannelFormatDesc* desc, cudaExtent extent, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaMalloc3DArray(array, desc, extent, flags)
    return cudaMalloc3DArray(array, desc, extent, flags)

cdef cudaError_t _cudaMallocMipmappedArray(cudaMipmappedArray_t* mipmappedArray, const cudaChannelFormatDesc* desc, cudaExtent extent, unsigned int numLevels, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaMallocMipmappedArray(mipmappedArray, desc, extent, numLevels, flags)
    return cudaMallocMipmappedArray(mipmappedArray, desc, extent, numLevels, flags)

cdef cudaError_t _cudaGetMipmappedArrayLevel(cudaArray_t* levelArray, cudaMipmappedArray_const_t mipmappedArray, unsigned int level) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaGetMipmappedArrayLevel(levelArray, mipmappedArray, level)
    return cudaGetMipmappedArrayLevel(levelArray, mipmappedArray, level)

cdef cudaError_t _cudaMemcpy3D(const cudaMemcpy3DParms* p) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaMemcpy3D(p)
    return cudaMemcpy3D(p)

cdef cudaError_t _cudaMemcpy3DPeer(const cudaMemcpy3DPeerParms* p) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaMemcpy3DPeer(p)
    return cudaMemcpy3DPeer(p)

cdef cudaError_t _cudaMemcpy3DAsync(const cudaMemcpy3DParms* p, cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaMemcpy3DAsync(p, stream)
    return cudaMemcpy3DAsync(p, stream)

cdef cudaError_t _cudaMemcpy3DPeerAsync(const cudaMemcpy3DPeerParms* p, cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaMemcpy3DPeerAsync(p, stream)
    return cudaMemcpy3DPeerAsync(p, stream)

cdef cudaError_t _cudaMemGetInfo(size_t* free, size_t* total) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaMemGetInfo(free, total)
    return cudaMemGetInfo(free, total)

cdef cudaError_t _cudaArrayGetInfo(cudaChannelFormatDesc* desc, cudaExtent* extent, unsigned int* flags, cudaArray_t array) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaArrayGetInfo(desc, extent, flags, array)
    return cudaArrayGetInfo(desc, extent, flags, array)

cdef cudaError_t _cudaArrayGetPlane(cudaArray_t* pPlaneArray, cudaArray_t hArray, unsigned int planeIdx) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaArrayGetPlane(pPlaneArray, hArray, planeIdx)
    return cudaArrayGetPlane(pPlaneArray, hArray, planeIdx)

cdef cudaError_t _cudaArrayGetMemoryRequirements(cudaArrayMemoryRequirements* memoryRequirements, cudaArray_t array, int device) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaArrayGetMemoryRequirements(memoryRequirements, array, device)
    return cudaArrayGetMemoryRequirements(memoryRequirements, array, device)

cdef cudaError_t _cudaMipmappedArrayGetMemoryRequirements(cudaArrayMemoryRequirements* memoryRequirements, cudaMipmappedArray_t mipmap, int device) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaMipmappedArrayGetMemoryRequirements(memoryRequirements, mipmap, device)
    return cudaMipmappedArrayGetMemoryRequirements(memoryRequirements, mipmap, device)

cdef cudaError_t _cudaArrayGetSparseProperties(cudaArraySparseProperties* sparseProperties, cudaArray_t array) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaArrayGetSparseProperties(sparseProperties, array)
    return cudaArrayGetSparseProperties(sparseProperties, array)

cdef cudaError_t _cudaMipmappedArrayGetSparseProperties(cudaArraySparseProperties* sparseProperties, cudaMipmappedArray_t mipmap) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaMipmappedArrayGetSparseProperties(sparseProperties, mipmap)
    return cudaMipmappedArrayGetSparseProperties(sparseProperties, mipmap)

cdef cudaError_t _cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaMemcpy(dst, src, count, kind)
    return cudaMemcpy(dst, src, count, kind)

cdef cudaError_t _cudaMemcpyPeer(void* dst, int dstDevice, const void* src, int srcDevice, size_t count) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaMemcpyPeer(dst, dstDevice, src, srcDevice, count)
    return cudaMemcpyPeer(dst, dstDevice, src, srcDevice, count)

cdef cudaError_t _cudaMemcpy2D(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaMemcpy2D(dst, dpitch, src, spitch, width, height, kind)
    return cudaMemcpy2D(dst, dpitch, src, spitch, width, height, kind)

cdef cudaError_t _cudaMemcpy2DToArray(cudaArray_t dst, size_t wOffset, size_t hOffset, const void* src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaMemcpy2DToArray(dst, wOffset, hOffset, src, spitch, width, height, kind)
    return cudaMemcpy2DToArray(dst, wOffset, hOffset, src, spitch, width, height, kind)

cdef cudaError_t _cudaMemcpy2DFromArray(void* dst, size_t dpitch, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t width, size_t height, cudaMemcpyKind kind) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaMemcpy2DFromArray(dst, dpitch, src, wOffset, hOffset, width, height, kind)
    return cudaMemcpy2DFromArray(dst, dpitch, src, wOffset, hOffset, width, height, kind)

cdef cudaError_t _cudaMemcpy2DArrayToArray(cudaArray_t dst, size_t wOffsetDst, size_t hOffsetDst, cudaArray_const_t src, size_t wOffsetSrc, size_t hOffsetSrc, size_t width, size_t height, cudaMemcpyKind kind) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaMemcpy2DArrayToArray(dst, wOffsetDst, hOffsetDst, src, wOffsetSrc, hOffsetSrc, width, height, kind)
    return cudaMemcpy2DArrayToArray(dst, wOffsetDst, hOffsetDst, src, wOffsetSrc, hOffsetSrc, width, height, kind)

cdef cudaError_t _cudaMemcpyAsync(void* dst, const void* src, size_t count, cudaMemcpyKind kind, cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaMemcpyAsync(dst, src, count, kind, stream)
    return cudaMemcpyAsync(dst, src, count, kind, stream)

cdef cudaError_t _cudaMemcpyPeerAsync(void* dst, int dstDevice, const void* src, int srcDevice, size_t count, cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaMemcpyPeerAsync(dst, dstDevice, src, srcDevice, count, stream)
    return cudaMemcpyPeerAsync(dst, dstDevice, src, srcDevice, count, stream)

cdef cudaError_t _cudaMemcpyBatchAsync(const void** dsts, const void** srcs, const size_t* sizes, size_t count, cudaMemcpyAttributes* attrs, size_t* attrsIdxs, size_t numAttrs, cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaMemcpyBatchAsync(dsts, srcs, sizes, count, attrs, attrsIdxs, numAttrs, stream)
    return cudaMemcpyBatchAsync(dsts, srcs, sizes, count, attrs, attrsIdxs, numAttrs, stream)

cdef cudaError_t _cudaMemcpy3DBatchAsync(size_t numOps, cudaMemcpy3DBatchOp* opList, unsigned long long flags, cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaMemcpy3DBatchAsync(numOps, opList, flags, stream)
    return cudaMemcpy3DBatchAsync(numOps, opList, flags, stream)

cdef cudaError_t _cudaMemcpyWithAttributesAsync(void* dst, const void* src, size_t size, cudaMemcpyAttributes* attr, cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaMemcpyWithAttributesAsync(dst, src, size, attr, stream)
    return cudaMemcpyWithAttributesAsync(dst, src, size, attr, stream)

cdef cudaError_t _cudaMemcpy3DWithAttributesAsync(cudaMemcpy3DBatchOp* op, unsigned long long flags, cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaMemcpy3DWithAttributesAsync(op, flags, stream)
    return cudaMemcpy3DWithAttributesAsync(op, flags, stream)

cdef cudaError_t _cudaMemcpy2DAsync(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaMemcpy2DAsync(dst, dpitch, src, spitch, width, height, kind, stream)
    return cudaMemcpy2DAsync(dst, dpitch, src, spitch, width, height, kind, stream)

cdef cudaError_t _cudaMemcpy2DToArrayAsync(cudaArray_t dst, size_t wOffset, size_t hOffset, const void* src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaMemcpy2DToArrayAsync(dst, wOffset, hOffset, src, spitch, width, height, kind, stream)
    return cudaMemcpy2DToArrayAsync(dst, wOffset, hOffset, src, spitch, width, height, kind, stream)

cdef cudaError_t _cudaMemcpy2DFromArrayAsync(void* dst, size_t dpitch, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaMemcpy2DFromArrayAsync(dst, dpitch, src, wOffset, hOffset, width, height, kind, stream)
    return cudaMemcpy2DFromArrayAsync(dst, dpitch, src, wOffset, hOffset, width, height, kind, stream)

cdef cudaError_t _cudaMemset(void* devPtr, int value, size_t count) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaMemset(devPtr, value, count)
    return cudaMemset(devPtr, value, count)

cdef cudaError_t _cudaMemset2D(void* devPtr, size_t pitch, int value, size_t width, size_t height) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaMemset2D(devPtr, pitch, value, width, height)
    return cudaMemset2D(devPtr, pitch, value, width, height)

cdef cudaError_t _cudaMemset3D(cudaPitchedPtr pitchedDevPtr, int value, cudaExtent extent) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaMemset3D(pitchedDevPtr, value, extent)
    return cudaMemset3D(pitchedDevPtr, value, extent)

cdef cudaError_t _cudaMemsetAsync(void* devPtr, int value, size_t count, cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaMemsetAsync(devPtr, value, count, stream)
    return cudaMemsetAsync(devPtr, value, count, stream)

cdef cudaError_t _cudaMemset2DAsync(void* devPtr, size_t pitch, int value, size_t width, size_t height, cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaMemset2DAsync(devPtr, pitch, value, width, height, stream)
    return cudaMemset2DAsync(devPtr, pitch, value, width, height, stream)

cdef cudaError_t _cudaMemset3DAsync(cudaPitchedPtr pitchedDevPtr, int value, cudaExtent extent, cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaMemset3DAsync(pitchedDevPtr, value, extent, stream)
    return cudaMemset3DAsync(pitchedDevPtr, value, extent, stream)

cdef cudaError_t _cudaMemPrefetchAsync(const void* devPtr, size_t count, cudaMemLocation location, unsigned int flags, cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaMemPrefetchAsync(devPtr, count, location, flags, stream)
    return cudaMemPrefetchAsync(devPtr, count, location, flags, stream)

cdef cudaError_t _cudaMemPrefetchBatchAsync(void** dptrs, size_t* sizes, size_t count, cudaMemLocation* prefetchLocs, size_t* prefetchLocIdxs, size_t numPrefetchLocs, unsigned long long flags, cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaMemPrefetchBatchAsync(dptrs, sizes, count, prefetchLocs, prefetchLocIdxs, numPrefetchLocs, flags, stream)
    return cudaMemPrefetchBatchAsync(dptrs, sizes, count, prefetchLocs, prefetchLocIdxs, numPrefetchLocs, flags, stream)

cdef cudaError_t _cudaMemDiscardBatchAsync(void** dptrs, size_t* sizes, size_t count, unsigned long long flags, cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaMemDiscardBatchAsync(dptrs, sizes, count, flags, stream)
    return cudaMemDiscardBatchAsync(dptrs, sizes, count, flags, stream)

cdef cudaError_t _cudaMemDiscardAndPrefetchBatchAsync(void** dptrs, size_t* sizes, size_t count, cudaMemLocation* prefetchLocs, size_t* prefetchLocIdxs, size_t numPrefetchLocs, unsigned long long flags, cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaMemDiscardAndPrefetchBatchAsync(dptrs, sizes, count, prefetchLocs, prefetchLocIdxs, numPrefetchLocs, flags, stream)
    return cudaMemDiscardAndPrefetchBatchAsync(dptrs, sizes, count, prefetchLocs, prefetchLocIdxs, numPrefetchLocs, flags, stream)

cdef cudaError_t _cudaMemAdvise(const void* devPtr, size_t count, cudaMemoryAdvise advice, cudaMemLocation location) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaMemAdvise(devPtr, count, advice, location)
    return cudaMemAdvise(devPtr, count, advice, location)

cdef cudaError_t _cudaMemRangeGetAttribute(void* data, size_t dataSize, cudaMemRangeAttribute attribute, const void* devPtr, size_t count) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaMemRangeGetAttribute(data, dataSize, attribute, devPtr, count)
    return cudaMemRangeGetAttribute(data, dataSize, attribute, devPtr, count)

cdef cudaError_t _cudaMemRangeGetAttributes(void** data, size_t* dataSizes, cudaMemRangeAttribute* attributes, size_t numAttributes, const void* devPtr, size_t count) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaMemRangeGetAttributes(data, dataSizes, attributes, numAttributes, devPtr, count)
    return cudaMemRangeGetAttributes(data, dataSizes, attributes, numAttributes, devPtr, count)

cdef cudaError_t _cudaMemcpyToArray(cudaArray_t dst, size_t wOffset, size_t hOffset, const void* src, size_t count, cudaMemcpyKind kind) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaMemcpyToArray(dst, wOffset, hOffset, src, count, kind)
    return cudaMemcpyToArray(dst, wOffset, hOffset, src, count, kind)

cdef cudaError_t _cudaMemcpyFromArray(void* dst, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t count, cudaMemcpyKind kind) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaMemcpyFromArray(dst, src, wOffset, hOffset, count, kind)
    return cudaMemcpyFromArray(dst, src, wOffset, hOffset, count, kind)

cdef cudaError_t _cudaMemcpyArrayToArray(cudaArray_t dst, size_t wOffsetDst, size_t hOffsetDst, cudaArray_const_t src, size_t wOffsetSrc, size_t hOffsetSrc, size_t count, cudaMemcpyKind kind) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaMemcpyArrayToArray(dst, wOffsetDst, hOffsetDst, src, wOffsetSrc, hOffsetSrc, count, kind)
    return cudaMemcpyArrayToArray(dst, wOffsetDst, hOffsetDst, src, wOffsetSrc, hOffsetSrc, count, kind)

cdef cudaError_t _cudaMemcpyToArrayAsync(cudaArray_t dst, size_t wOffset, size_t hOffset, const void* src, size_t count, cudaMemcpyKind kind, cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaMemcpyToArrayAsync(dst, wOffset, hOffset, src, count, kind, stream)
    return cudaMemcpyToArrayAsync(dst, wOffset, hOffset, src, count, kind, stream)

cdef cudaError_t _cudaMemcpyFromArrayAsync(void* dst, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t count, cudaMemcpyKind kind, cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaMemcpyFromArrayAsync(dst, src, wOffset, hOffset, count, kind, stream)
    return cudaMemcpyFromArrayAsync(dst, src, wOffset, hOffset, count, kind, stream)

cdef cudaError_t _cudaMallocAsync(void** devPtr, size_t size, cudaStream_t hStream) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaMallocAsync(devPtr, size, hStream)
    return cudaMallocAsync(devPtr, size, hStream)

cdef cudaError_t _cudaFreeAsync(void* devPtr, cudaStream_t hStream) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaFreeAsync(devPtr, hStream)
    return cudaFreeAsync(devPtr, hStream)

cdef cudaError_t _cudaMemPoolTrimTo(cudaMemPool_t memPool, size_t minBytesToKeep) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaMemPoolTrimTo(memPool, minBytesToKeep)
    return cudaMemPoolTrimTo(memPool, minBytesToKeep)

cdef cudaError_t _cudaMemPoolSetAttribute(cudaMemPool_t memPool, cudaMemPoolAttr attr, void* value) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaMemPoolSetAttribute(memPool, attr, value)
    return cudaMemPoolSetAttribute(memPool, attr, value)

cdef cudaError_t _cudaMemPoolGetAttribute(cudaMemPool_t memPool, cudaMemPoolAttr attr, void* value) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaMemPoolGetAttribute(memPool, attr, value)
    return cudaMemPoolGetAttribute(memPool, attr, value)

cdef cudaError_t _cudaMemPoolSetAccess(cudaMemPool_t memPool, const cudaMemAccessDesc* descList, size_t count) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaMemPoolSetAccess(memPool, descList, count)
    return cudaMemPoolSetAccess(memPool, descList, count)

cdef cudaError_t _cudaMemPoolGetAccess(cudaMemAccessFlags* flags, cudaMemPool_t memPool, cudaMemLocation* location) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaMemPoolGetAccess(flags, memPool, location)
    return cudaMemPoolGetAccess(flags, memPool, location)

cdef cudaError_t _cudaMemPoolCreate(cudaMemPool_t* memPool, const cudaMemPoolProps* poolProps) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaMemPoolCreate(memPool, poolProps)
    return cudaMemPoolCreate(memPool, poolProps)

cdef cudaError_t _cudaMemPoolDestroy(cudaMemPool_t memPool) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaMemPoolDestroy(memPool)
    return cudaMemPoolDestroy(memPool)

cdef cudaError_t _cudaMemGetDefaultMemPool(cudaMemPool_t* memPool, cudaMemLocation* location, cudaMemAllocationType typename) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaMemGetDefaultMemPool(memPool, location, typename)
    return cudaMemGetDefaultMemPool(memPool, location, typename)

cdef cudaError_t _cudaMemGetMemPool(cudaMemPool_t* memPool, cudaMemLocation* location, cudaMemAllocationType typename) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaMemGetMemPool(memPool, location, typename)
    return cudaMemGetMemPool(memPool, location, typename)

cdef cudaError_t _cudaMemSetMemPool(cudaMemLocation* location, cudaMemAllocationType typename, cudaMemPool_t memPool) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaMemSetMemPool(location, typename, memPool)
    return cudaMemSetMemPool(location, typename, memPool)

cdef cudaError_t _cudaMallocFromPoolAsync(void** ptr, size_t size, cudaMemPool_t memPool, cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaMallocFromPoolAsync(ptr, size, memPool, stream)
    return cudaMallocFromPoolAsync(ptr, size, memPool, stream)

cdef cudaError_t _cudaMemPoolExportToShareableHandle(void* shareableHandle, cudaMemPool_t memPool, cudaMemAllocationHandleType handleType, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaMemPoolExportToShareableHandle(shareableHandle, memPool, handleType, flags)
    return cudaMemPoolExportToShareableHandle(shareableHandle, memPool, handleType, flags)

cdef cudaError_t _cudaMemPoolImportFromShareableHandle(cudaMemPool_t* memPool, void* shareableHandle, cudaMemAllocationHandleType handleType, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaMemPoolImportFromShareableHandle(memPool, shareableHandle, handleType, flags)
    return cudaMemPoolImportFromShareableHandle(memPool, shareableHandle, handleType, flags)

cdef cudaError_t _cudaMemPoolExportPointer(cudaMemPoolPtrExportData* exportData, void* ptr) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaMemPoolExportPointer(exportData, ptr)
    return cudaMemPoolExportPointer(exportData, ptr)

cdef cudaError_t _cudaMemPoolImportPointer(void** ptr, cudaMemPool_t memPool, cudaMemPoolPtrExportData* exportData) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaMemPoolImportPointer(ptr, memPool, exportData)
    return cudaMemPoolImportPointer(ptr, memPool, exportData)

cdef cudaError_t _cudaPointerGetAttributes(cudaPointerAttributes* attributes, const void* ptr) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaPointerGetAttributes(attributes, ptr)
    return cudaPointerGetAttributes(attributes, ptr)

cdef cudaError_t _cudaDeviceCanAccessPeer(int* canAccessPeer, int device, int peerDevice) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaDeviceCanAccessPeer(canAccessPeer, device, peerDevice)
    return cudaDeviceCanAccessPeer(canAccessPeer, device, peerDevice)

cdef cudaError_t _cudaDeviceEnablePeerAccess(int peerDevice, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaDeviceEnablePeerAccess(peerDevice, flags)
    return cudaDeviceEnablePeerAccess(peerDevice, flags)

cdef cudaError_t _cudaDeviceDisablePeerAccess(int peerDevice) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaDeviceDisablePeerAccess(peerDevice)
    return cudaDeviceDisablePeerAccess(peerDevice)

cdef cudaError_t _cudaGraphicsUnregisterResource(cudaGraphicsResource_t resource) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaGraphicsUnregisterResource(resource)
    return cudaGraphicsUnregisterResource(resource)

cdef cudaError_t _cudaGraphicsResourceSetMapFlags(cudaGraphicsResource_t resource, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaGraphicsResourceSetMapFlags(resource, flags)
    return cudaGraphicsResourceSetMapFlags(resource, flags)

cdef cudaError_t _cudaGraphicsMapResources(int count, cudaGraphicsResource_t* resources, cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaGraphicsMapResources(count, resources, stream)
    return cudaGraphicsMapResources(count, resources, stream)

cdef cudaError_t _cudaGraphicsUnmapResources(int count, cudaGraphicsResource_t* resources, cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaGraphicsUnmapResources(count, resources, stream)
    return cudaGraphicsUnmapResources(count, resources, stream)

cdef cudaError_t _cudaGraphicsResourceGetMappedPointer(void** devPtr, size_t* size, cudaGraphicsResource_t resource) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaGraphicsResourceGetMappedPointer(devPtr, size, resource)
    return cudaGraphicsResourceGetMappedPointer(devPtr, size, resource)

cdef cudaError_t _cudaGraphicsSubResourceGetMappedArray(cudaArray_t* array, cudaGraphicsResource_t resource, unsigned int arrayIndex, unsigned int mipLevel) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaGraphicsSubResourceGetMappedArray(array, resource, arrayIndex, mipLevel)
    return cudaGraphicsSubResourceGetMappedArray(array, resource, arrayIndex, mipLevel)

cdef cudaError_t _cudaGraphicsResourceGetMappedMipmappedArray(cudaMipmappedArray_t* mipmappedArray, cudaGraphicsResource_t resource) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaGraphicsResourceGetMappedMipmappedArray(mipmappedArray, resource)
    return cudaGraphicsResourceGetMappedMipmappedArray(mipmappedArray, resource)

cdef cudaError_t _cudaGetChannelDesc(cudaChannelFormatDesc* desc, cudaArray_const_t array) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaGetChannelDesc(desc, array)
    return cudaGetChannelDesc(desc, array)
@cython.show_performance_hints(False)
cdef cudaChannelFormatDesc _cudaCreateChannelDesc(int x, int y, int z, int w, cudaChannelFormatKind f) except* nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaCreateChannelDesc(x, y, z, w, f)
    return cudaCreateChannelDesc(x, y, z, w, f)

cdef cudaError_t _cudaCreateTextureObject(cudaTextureObject_t* pTexObject, const cudaResourceDesc* pResDesc, const cudaTextureDesc* pTexDesc, const cudaResourceViewDesc* pResViewDesc) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaCreateTextureObject(pTexObject, pResDesc, pTexDesc, pResViewDesc)
    return cudaCreateTextureObject(pTexObject, pResDesc, pTexDesc, pResViewDesc)

cdef cudaError_t _cudaDestroyTextureObject(cudaTextureObject_t texObject) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaDestroyTextureObject(texObject)
    return cudaDestroyTextureObject(texObject)

cdef cudaError_t _cudaGetTextureObjectResourceDesc(cudaResourceDesc* pResDesc, cudaTextureObject_t texObject) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaGetTextureObjectResourceDesc(pResDesc, texObject)
    return cudaGetTextureObjectResourceDesc(pResDesc, texObject)

cdef cudaError_t _cudaGetTextureObjectTextureDesc(cudaTextureDesc* pTexDesc, cudaTextureObject_t texObject) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaGetTextureObjectTextureDesc(pTexDesc, texObject)
    return cudaGetTextureObjectTextureDesc(pTexDesc, texObject)

cdef cudaError_t _cudaGetTextureObjectResourceViewDesc(cudaResourceViewDesc* pResViewDesc, cudaTextureObject_t texObject) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaGetTextureObjectResourceViewDesc(pResViewDesc, texObject)
    return cudaGetTextureObjectResourceViewDesc(pResViewDesc, texObject)

cdef cudaError_t _cudaCreateSurfaceObject(cudaSurfaceObject_t* pSurfObject, const cudaResourceDesc* pResDesc) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaCreateSurfaceObject(pSurfObject, pResDesc)
    return cudaCreateSurfaceObject(pSurfObject, pResDesc)

cdef cudaError_t _cudaDestroySurfaceObject(cudaSurfaceObject_t surfObject) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaDestroySurfaceObject(surfObject)
    return cudaDestroySurfaceObject(surfObject)

cdef cudaError_t _cudaGetSurfaceObjectResourceDesc(cudaResourceDesc* pResDesc, cudaSurfaceObject_t surfObject) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaGetSurfaceObjectResourceDesc(pResDesc, surfObject)
    return cudaGetSurfaceObjectResourceDesc(pResDesc, surfObject)

cdef cudaError_t _cudaDriverGetVersion(int* driverVersion) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaDriverGetVersion(driverVersion)
    return cudaDriverGetVersion(driverVersion)

cdef cudaError_t _cudaRuntimeGetVersion(int* runtimeVersion) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaRuntimeGetVersion(runtimeVersion)
    return cudaRuntimeGetVersion(runtimeVersion)

cdef cudaError_t _cudaLogsRegisterCallback(cudaLogsCallback_t callbackFunc, void* userData, cudaLogsCallbackHandle* callback_out) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaLogsRegisterCallback(callbackFunc, userData, callback_out)
    return cudaLogsRegisterCallback(callbackFunc, userData, callback_out)

cdef cudaError_t _cudaLogsUnregisterCallback(cudaLogsCallbackHandle callback) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaLogsUnregisterCallback(callback)
    return cudaLogsUnregisterCallback(callback)

cdef cudaError_t _cudaLogsCurrent(cudaLogIterator* iterator_out, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaLogsCurrent(iterator_out, flags)
    return cudaLogsCurrent(iterator_out, flags)

cdef cudaError_t _cudaLogsDumpToFile(cudaLogIterator* iterator, const char* pathToFile, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaLogsDumpToFile(iterator, pathToFile, flags)
    return cudaLogsDumpToFile(iterator, pathToFile, flags)

cdef cudaError_t _cudaLogsDumpToMemory(cudaLogIterator* iterator, char* buffer, size_t* size, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaLogsDumpToMemory(iterator, buffer, size, flags)
    return cudaLogsDumpToMemory(iterator, buffer, size, flags)

cdef cudaError_t _cudaGraphCreate(cudaGraph_t* pGraph, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaGraphCreate(pGraph, flags)
    return cudaGraphCreate(pGraph, flags)

cdef cudaError_t _cudaGraphAddKernelNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, const cudaKernelNodeParams* pNodeParams) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaGraphAddKernelNode(pGraphNode, graph, pDependencies, numDependencies, pNodeParams)
    return cudaGraphAddKernelNode(pGraphNode, graph, pDependencies, numDependencies, pNodeParams)

cdef cudaError_t _cudaGraphKernelNodeGetParams(cudaGraphNode_t node, cudaKernelNodeParams* pNodeParams) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaGraphKernelNodeGetParams(node, pNodeParams)
    return cudaGraphKernelNodeGetParams(node, pNodeParams)

cdef cudaError_t _cudaGraphKernelNodeSetParams(cudaGraphNode_t node, const cudaKernelNodeParams* pNodeParams) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaGraphKernelNodeSetParams(node, pNodeParams)
    return cudaGraphKernelNodeSetParams(node, pNodeParams)

cdef cudaError_t _cudaGraphKernelNodeCopyAttributes(cudaGraphNode_t hDst, cudaGraphNode_t hSrc) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaGraphKernelNodeCopyAttributes(hDst, hSrc)
    return cudaGraphKernelNodeCopyAttributes(hDst, hSrc)

cdef cudaError_t _cudaGraphKernelNodeGetAttribute(cudaGraphNode_t hNode, cudaKernelNodeAttrID attr, cudaKernelNodeAttrValue* value_out) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaGraphKernelNodeGetAttribute(hNode, attr, value_out)
    return cudaGraphKernelNodeGetAttribute(hNode, attr, value_out)

cdef cudaError_t _cudaGraphKernelNodeSetAttribute(cudaGraphNode_t hNode, cudaKernelNodeAttrID attr, const cudaKernelNodeAttrValue* value) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaGraphKernelNodeSetAttribute(hNode, attr, value)
    return cudaGraphKernelNodeSetAttribute(hNode, attr, value)

cdef cudaError_t _cudaGraphAddMemcpyNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, const cudaMemcpy3DParms* pCopyParams) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaGraphAddMemcpyNode(pGraphNode, graph, pDependencies, numDependencies, pCopyParams)
    return cudaGraphAddMemcpyNode(pGraphNode, graph, pDependencies, numDependencies, pCopyParams)

cdef cudaError_t _cudaGraphAddMemcpyNode1D(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, void* dst, const void* src, size_t count, cudaMemcpyKind kind) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaGraphAddMemcpyNode1D(pGraphNode, graph, pDependencies, numDependencies, dst, src, count, kind)
    return cudaGraphAddMemcpyNode1D(pGraphNode, graph, pDependencies, numDependencies, dst, src, count, kind)

cdef cudaError_t _cudaGraphMemcpyNodeGetParams(cudaGraphNode_t node, cudaMemcpy3DParms* pNodeParams) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaGraphMemcpyNodeGetParams(node, pNodeParams)
    return cudaGraphMemcpyNodeGetParams(node, pNodeParams)

cdef cudaError_t _cudaGraphMemcpyNodeSetParams(cudaGraphNode_t node, const cudaMemcpy3DParms* pNodeParams) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaGraphMemcpyNodeSetParams(node, pNodeParams)
    return cudaGraphMemcpyNodeSetParams(node, pNodeParams)

cdef cudaError_t _cudaGraphMemcpyNodeSetParams1D(cudaGraphNode_t node, void* dst, const void* src, size_t count, cudaMemcpyKind kind) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaGraphMemcpyNodeSetParams1D(node, dst, src, count, kind)
    return cudaGraphMemcpyNodeSetParams1D(node, dst, src, count, kind)

cdef cudaError_t _cudaGraphAddMemsetNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, const cudaMemsetParams* pMemsetParams) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaGraphAddMemsetNode(pGraphNode, graph, pDependencies, numDependencies, pMemsetParams)
    return cudaGraphAddMemsetNode(pGraphNode, graph, pDependencies, numDependencies, pMemsetParams)

cdef cudaError_t _cudaGraphMemsetNodeGetParams(cudaGraphNode_t node, cudaMemsetParams* pNodeParams) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaGraphMemsetNodeGetParams(node, pNodeParams)
    return cudaGraphMemsetNodeGetParams(node, pNodeParams)

cdef cudaError_t _cudaGraphMemsetNodeSetParams(cudaGraphNode_t node, const cudaMemsetParams* pNodeParams) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaGraphMemsetNodeSetParams(node, pNodeParams)
    return cudaGraphMemsetNodeSetParams(node, pNodeParams)

cdef cudaError_t _cudaGraphAddHostNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, const cudaHostNodeParams* pNodeParams) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaGraphAddHostNode(pGraphNode, graph, pDependencies, numDependencies, pNodeParams)
    return cudaGraphAddHostNode(pGraphNode, graph, pDependencies, numDependencies, pNodeParams)

cdef cudaError_t _cudaGraphHostNodeGetParams(cudaGraphNode_t node, cudaHostNodeParams* pNodeParams) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaGraphHostNodeGetParams(node, pNodeParams)
    return cudaGraphHostNodeGetParams(node, pNodeParams)

cdef cudaError_t _cudaGraphHostNodeSetParams(cudaGraphNode_t node, const cudaHostNodeParams* pNodeParams) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaGraphHostNodeSetParams(node, pNodeParams)
    return cudaGraphHostNodeSetParams(node, pNodeParams)

cdef cudaError_t _cudaGraphAddChildGraphNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, cudaGraph_t childGraph) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaGraphAddChildGraphNode(pGraphNode, graph, pDependencies, numDependencies, childGraph)
    return cudaGraphAddChildGraphNode(pGraphNode, graph, pDependencies, numDependencies, childGraph)

cdef cudaError_t _cudaGraphChildGraphNodeGetGraph(cudaGraphNode_t node, cudaGraph_t* pGraph) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaGraphChildGraphNodeGetGraph(node, pGraph)
    return cudaGraphChildGraphNodeGetGraph(node, pGraph)

cdef cudaError_t _cudaGraphAddEmptyNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaGraphAddEmptyNode(pGraphNode, graph, pDependencies, numDependencies)
    return cudaGraphAddEmptyNode(pGraphNode, graph, pDependencies, numDependencies)

cdef cudaError_t _cudaGraphAddEventRecordNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, cudaEvent_t event) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaGraphAddEventRecordNode(pGraphNode, graph, pDependencies, numDependencies, event)
    return cudaGraphAddEventRecordNode(pGraphNode, graph, pDependencies, numDependencies, event)

cdef cudaError_t _cudaGraphEventRecordNodeGetEvent(cudaGraphNode_t node, cudaEvent_t* event_out) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaGraphEventRecordNodeGetEvent(node, event_out)
    return cudaGraphEventRecordNodeGetEvent(node, event_out)

cdef cudaError_t _cudaGraphEventRecordNodeSetEvent(cudaGraphNode_t node, cudaEvent_t event) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaGraphEventRecordNodeSetEvent(node, event)
    return cudaGraphEventRecordNodeSetEvent(node, event)

cdef cudaError_t _cudaGraphAddEventWaitNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, cudaEvent_t event) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaGraphAddEventWaitNode(pGraphNode, graph, pDependencies, numDependencies, event)
    return cudaGraphAddEventWaitNode(pGraphNode, graph, pDependencies, numDependencies, event)

cdef cudaError_t _cudaGraphEventWaitNodeGetEvent(cudaGraphNode_t node, cudaEvent_t* event_out) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaGraphEventWaitNodeGetEvent(node, event_out)
    return cudaGraphEventWaitNodeGetEvent(node, event_out)

cdef cudaError_t _cudaGraphEventWaitNodeSetEvent(cudaGraphNode_t node, cudaEvent_t event) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaGraphEventWaitNodeSetEvent(node, event)
    return cudaGraphEventWaitNodeSetEvent(node, event)

cdef cudaError_t _cudaGraphAddExternalSemaphoresSignalNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, const cudaExternalSemaphoreSignalNodeParams* nodeParams) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaGraphAddExternalSemaphoresSignalNode(pGraphNode, graph, pDependencies, numDependencies, nodeParams)
    return cudaGraphAddExternalSemaphoresSignalNode(pGraphNode, graph, pDependencies, numDependencies, nodeParams)

cdef cudaError_t _cudaGraphExternalSemaphoresSignalNodeGetParams(cudaGraphNode_t hNode, cudaExternalSemaphoreSignalNodeParams* params_out) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaGraphExternalSemaphoresSignalNodeGetParams(hNode, params_out)
    return cudaGraphExternalSemaphoresSignalNodeGetParams(hNode, params_out)

cdef cudaError_t _cudaGraphExternalSemaphoresSignalNodeSetParams(cudaGraphNode_t hNode, const cudaExternalSemaphoreSignalNodeParams* nodeParams) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaGraphExternalSemaphoresSignalNodeSetParams(hNode, nodeParams)
    return cudaGraphExternalSemaphoresSignalNodeSetParams(hNode, nodeParams)

cdef cudaError_t _cudaGraphAddExternalSemaphoresWaitNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, const cudaExternalSemaphoreWaitNodeParams* nodeParams) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaGraphAddExternalSemaphoresWaitNode(pGraphNode, graph, pDependencies, numDependencies, nodeParams)
    return cudaGraphAddExternalSemaphoresWaitNode(pGraphNode, graph, pDependencies, numDependencies, nodeParams)

cdef cudaError_t _cudaGraphExternalSemaphoresWaitNodeGetParams(cudaGraphNode_t hNode, cudaExternalSemaphoreWaitNodeParams* params_out) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaGraphExternalSemaphoresWaitNodeGetParams(hNode, params_out)
    return cudaGraphExternalSemaphoresWaitNodeGetParams(hNode, params_out)

cdef cudaError_t _cudaGraphExternalSemaphoresWaitNodeSetParams(cudaGraphNode_t hNode, const cudaExternalSemaphoreWaitNodeParams* nodeParams) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaGraphExternalSemaphoresWaitNodeSetParams(hNode, nodeParams)
    return cudaGraphExternalSemaphoresWaitNodeSetParams(hNode, nodeParams)

cdef cudaError_t _cudaGraphAddMemAllocNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, cudaMemAllocNodeParams* nodeParams) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaGraphAddMemAllocNode(pGraphNode, graph, pDependencies, numDependencies, nodeParams)
    return cudaGraphAddMemAllocNode(pGraphNode, graph, pDependencies, numDependencies, nodeParams)

cdef cudaError_t _cudaGraphMemAllocNodeGetParams(cudaGraphNode_t node, cudaMemAllocNodeParams* params_out) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaGraphMemAllocNodeGetParams(node, params_out)
    return cudaGraphMemAllocNodeGetParams(node, params_out)

cdef cudaError_t _cudaGraphAddMemFreeNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, void* dptr) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaGraphAddMemFreeNode(pGraphNode, graph, pDependencies, numDependencies, dptr)
    return cudaGraphAddMemFreeNode(pGraphNode, graph, pDependencies, numDependencies, dptr)

cdef cudaError_t _cudaGraphMemFreeNodeGetParams(cudaGraphNode_t node, void* dptr_out) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaGraphMemFreeNodeGetParams(node, dptr_out)
    return cudaGraphMemFreeNodeGetParams(node, dptr_out)

cdef cudaError_t _cudaDeviceGraphMemTrim(int device) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaDeviceGraphMemTrim(device)
    return cudaDeviceGraphMemTrim(device)

cdef cudaError_t _cudaDeviceGetGraphMemAttribute(int device, cudaGraphMemAttributeType attr, void* value) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaDeviceGetGraphMemAttribute(device, attr, value)
    return cudaDeviceGetGraphMemAttribute(device, attr, value)

cdef cudaError_t _cudaDeviceSetGraphMemAttribute(int device, cudaGraphMemAttributeType attr, void* value) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaDeviceSetGraphMemAttribute(device, attr, value)
    return cudaDeviceSetGraphMemAttribute(device, attr, value)

cdef cudaError_t _cudaGraphClone(cudaGraph_t* pGraphClone, cudaGraph_t originalGraph) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaGraphClone(pGraphClone, originalGraph)
    return cudaGraphClone(pGraphClone, originalGraph)

cdef cudaError_t _cudaGraphNodeFindInClone(cudaGraphNode_t* pNode, cudaGraphNode_t originalNode, cudaGraph_t clonedGraph) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaGraphNodeFindInClone(pNode, originalNode, clonedGraph)
    return cudaGraphNodeFindInClone(pNode, originalNode, clonedGraph)

cdef cudaError_t _cudaGraphNodeGetType(cudaGraphNode_t node, cudaGraphNodeType* pType) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaGraphNodeGetType(node, pType)
    return cudaGraphNodeGetType(node, pType)

cdef cudaError_t _cudaGraphNodeGetContainingGraph(cudaGraphNode_t hNode, cudaGraph_t* phGraph) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaGraphNodeGetContainingGraph(hNode, phGraph)
    return cudaGraphNodeGetContainingGraph(hNode, phGraph)

cdef cudaError_t _cudaGraphNodeGetLocalId(cudaGraphNode_t hNode, unsigned int* nodeId) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaGraphNodeGetLocalId(hNode, nodeId)
    return cudaGraphNodeGetLocalId(hNode, nodeId)

cdef cudaError_t _cudaGraphNodeGetToolsId(cudaGraphNode_t hNode, unsigned long long* toolsNodeId) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaGraphNodeGetToolsId(hNode, toolsNodeId)
    return cudaGraphNodeGetToolsId(hNode, toolsNodeId)

cdef cudaError_t _cudaGraphGetId(cudaGraph_t hGraph, unsigned int* graphID) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaGraphGetId(hGraph, graphID)
    return cudaGraphGetId(hGraph, graphID)

cdef cudaError_t _cudaGraphExecGetId(cudaGraphExec_t hGraphExec, unsigned int* graphID) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaGraphExecGetId(hGraphExec, graphID)
    return cudaGraphExecGetId(hGraphExec, graphID)

cdef cudaError_t _cudaGraphGetNodes(cudaGraph_t graph, cudaGraphNode_t* nodes, size_t* numNodes) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaGraphGetNodes(graph, nodes, numNodes)
    return cudaGraphGetNodes(graph, nodes, numNodes)

cdef cudaError_t _cudaGraphGetRootNodes(cudaGraph_t graph, cudaGraphNode_t* pRootNodes, size_t* pNumRootNodes) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaGraphGetRootNodes(graph, pRootNodes, pNumRootNodes)
    return cudaGraphGetRootNodes(graph, pRootNodes, pNumRootNodes)

cdef cudaError_t _cudaGraphGetEdges(cudaGraph_t graph, cudaGraphNode_t* from_, cudaGraphNode_t* to, cudaGraphEdgeData* edgeData, size_t* numEdges) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaGraphGetEdges(graph, from_, to, edgeData, numEdges)
    return cudaGraphGetEdges(graph, from_, to, edgeData, numEdges)

cdef cudaError_t _cudaGraphNodeGetDependencies(cudaGraphNode_t node, cudaGraphNode_t* pDependencies, cudaGraphEdgeData* edgeData, size_t* pNumDependencies) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaGraphNodeGetDependencies(node, pDependencies, edgeData, pNumDependencies)
    return cudaGraphNodeGetDependencies(node, pDependencies, edgeData, pNumDependencies)

cdef cudaError_t _cudaGraphNodeGetDependentNodes(cudaGraphNode_t node, cudaGraphNode_t* pDependentNodes, cudaGraphEdgeData* edgeData, size_t* pNumDependentNodes) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaGraphNodeGetDependentNodes(node, pDependentNodes, edgeData, pNumDependentNodes)
    return cudaGraphNodeGetDependentNodes(node, pDependentNodes, edgeData, pNumDependentNodes)

cdef cudaError_t _cudaGraphAddDependencies(cudaGraph_t graph, const cudaGraphNode_t* from_, const cudaGraphNode_t* to, const cudaGraphEdgeData* edgeData, size_t numDependencies) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaGraphAddDependencies(graph, from_, to, edgeData, numDependencies)
    return cudaGraphAddDependencies(graph, from_, to, edgeData, numDependencies)

cdef cudaError_t _cudaGraphRemoveDependencies(cudaGraph_t graph, const cudaGraphNode_t* from_, const cudaGraphNode_t* to, const cudaGraphEdgeData* edgeData, size_t numDependencies) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaGraphRemoveDependencies(graph, from_, to, edgeData, numDependencies)
    return cudaGraphRemoveDependencies(graph, from_, to, edgeData, numDependencies)

cdef cudaError_t _cudaGraphDestroyNode(cudaGraphNode_t node) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaGraphDestroyNode(node)
    return cudaGraphDestroyNode(node)

cdef cudaError_t _cudaGraphInstantiate(cudaGraphExec_t* pGraphExec, cudaGraph_t graph, unsigned long long flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaGraphInstantiate(pGraphExec, graph, flags)
    return cudaGraphInstantiate(pGraphExec, graph, flags)

cdef cudaError_t _cudaGraphInstantiateWithFlags(cudaGraphExec_t* pGraphExec, cudaGraph_t graph, unsigned long long flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaGraphInstantiateWithFlags(pGraphExec, graph, flags)
    return cudaGraphInstantiateWithFlags(pGraphExec, graph, flags)

cdef cudaError_t _cudaGraphInstantiateWithParams(cudaGraphExec_t* pGraphExec, cudaGraph_t graph, cudaGraphInstantiateParams* instantiateParams) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaGraphInstantiateWithParams(pGraphExec, graph, instantiateParams)
    return cudaGraphInstantiateWithParams(pGraphExec, graph, instantiateParams)

cdef cudaError_t _cudaGraphExecGetFlags(cudaGraphExec_t graphExec, unsigned long long* flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaGraphExecGetFlags(graphExec, flags)
    return cudaGraphExecGetFlags(graphExec, flags)

cdef cudaError_t _cudaGraphExecKernelNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, const cudaKernelNodeParams* pNodeParams) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaGraphExecKernelNodeSetParams(hGraphExec, node, pNodeParams)
    return cudaGraphExecKernelNodeSetParams(hGraphExec, node, pNodeParams)

cdef cudaError_t _cudaGraphExecMemcpyNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, const cudaMemcpy3DParms* pNodeParams) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaGraphExecMemcpyNodeSetParams(hGraphExec, node, pNodeParams)
    return cudaGraphExecMemcpyNodeSetParams(hGraphExec, node, pNodeParams)

cdef cudaError_t _cudaGraphExecMemcpyNodeSetParams1D(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, void* dst, const void* src, size_t count, cudaMemcpyKind kind) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaGraphExecMemcpyNodeSetParams1D(hGraphExec, node, dst, src, count, kind)
    return cudaGraphExecMemcpyNodeSetParams1D(hGraphExec, node, dst, src, count, kind)

cdef cudaError_t _cudaGraphExecMemsetNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, const cudaMemsetParams* pNodeParams) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaGraphExecMemsetNodeSetParams(hGraphExec, node, pNodeParams)
    return cudaGraphExecMemsetNodeSetParams(hGraphExec, node, pNodeParams)

cdef cudaError_t _cudaGraphExecHostNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, const cudaHostNodeParams* pNodeParams) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaGraphExecHostNodeSetParams(hGraphExec, node, pNodeParams)
    return cudaGraphExecHostNodeSetParams(hGraphExec, node, pNodeParams)

cdef cudaError_t _cudaGraphExecChildGraphNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, cudaGraph_t childGraph) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaGraphExecChildGraphNodeSetParams(hGraphExec, node, childGraph)
    return cudaGraphExecChildGraphNodeSetParams(hGraphExec, node, childGraph)

cdef cudaError_t _cudaGraphExecEventRecordNodeSetEvent(cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, cudaEvent_t event) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaGraphExecEventRecordNodeSetEvent(hGraphExec, hNode, event)
    return cudaGraphExecEventRecordNodeSetEvent(hGraphExec, hNode, event)

cdef cudaError_t _cudaGraphExecEventWaitNodeSetEvent(cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, cudaEvent_t event) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaGraphExecEventWaitNodeSetEvent(hGraphExec, hNode, event)
    return cudaGraphExecEventWaitNodeSetEvent(hGraphExec, hNode, event)

cdef cudaError_t _cudaGraphExecExternalSemaphoresSignalNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, const cudaExternalSemaphoreSignalNodeParams* nodeParams) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaGraphExecExternalSemaphoresSignalNodeSetParams(hGraphExec, hNode, nodeParams)
    return cudaGraphExecExternalSemaphoresSignalNodeSetParams(hGraphExec, hNode, nodeParams)

cdef cudaError_t _cudaGraphExecExternalSemaphoresWaitNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, const cudaExternalSemaphoreWaitNodeParams* nodeParams) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaGraphExecExternalSemaphoresWaitNodeSetParams(hGraphExec, hNode, nodeParams)
    return cudaGraphExecExternalSemaphoresWaitNodeSetParams(hGraphExec, hNode, nodeParams)

cdef cudaError_t _cudaGraphNodeSetEnabled(cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, unsigned int isEnabled) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaGraphNodeSetEnabled(hGraphExec, hNode, isEnabled)
    return cudaGraphNodeSetEnabled(hGraphExec, hNode, isEnabled)

cdef cudaError_t _cudaGraphNodeGetEnabled(cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, unsigned int* isEnabled) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaGraphNodeGetEnabled(hGraphExec, hNode, isEnabled)
    return cudaGraphNodeGetEnabled(hGraphExec, hNode, isEnabled)

cdef cudaError_t _cudaGraphExecUpdate(cudaGraphExec_t hGraphExec, cudaGraph_t hGraph, cudaGraphExecUpdateResultInfo* resultInfo) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaGraphExecUpdate(hGraphExec, hGraph, resultInfo)
    return cudaGraphExecUpdate(hGraphExec, hGraph, resultInfo)

cdef cudaError_t _cudaGraphUpload(cudaGraphExec_t graphExec, cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaGraphUpload(graphExec, stream)
    return cudaGraphUpload(graphExec, stream)

cdef cudaError_t _cudaGraphLaunch(cudaGraphExec_t graphExec, cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaGraphLaunch(graphExec, stream)
    return cudaGraphLaunch(graphExec, stream)

cdef cudaError_t _cudaGraphExecDestroy(cudaGraphExec_t graphExec) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaGraphExecDestroy(graphExec)
    return cudaGraphExecDestroy(graphExec)

cdef cudaError_t _cudaGraphDestroy(cudaGraph_t graph) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaGraphDestroy(graph)
    return cudaGraphDestroy(graph)

cdef cudaError_t _cudaGraphDebugDotPrint(cudaGraph_t graph, const char* path, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaGraphDebugDotPrint(graph, path, flags)
    return cudaGraphDebugDotPrint(graph, path, flags)

cdef cudaError_t _cudaUserObjectCreate(cudaUserObject_t* object_out, void* ptr, cudaHostFn_t destroy, unsigned int initialRefcount, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaUserObjectCreate(object_out, ptr, destroy, initialRefcount, flags)
    return cudaUserObjectCreate(object_out, ptr, destroy, initialRefcount, flags)

cdef cudaError_t _cudaUserObjectRetain(cudaUserObject_t object, unsigned int count) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaUserObjectRetain(object, count)
    return cudaUserObjectRetain(object, count)

cdef cudaError_t _cudaUserObjectRelease(cudaUserObject_t object, unsigned int count) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaUserObjectRelease(object, count)
    return cudaUserObjectRelease(object, count)

cdef cudaError_t _cudaGraphRetainUserObject(cudaGraph_t graph, cudaUserObject_t object, unsigned int count, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaGraphRetainUserObject(graph, object, count, flags)
    return cudaGraphRetainUserObject(graph, object, count, flags)

cdef cudaError_t _cudaGraphReleaseUserObject(cudaGraph_t graph, cudaUserObject_t object, unsigned int count) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaGraphReleaseUserObject(graph, object, count)
    return cudaGraphReleaseUserObject(graph, object, count)

cdef cudaError_t _cudaGraphAddNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, const cudaGraphEdgeData* dependencyData, size_t numDependencies, cudaGraphNodeParams* nodeParams) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaGraphAddNode(pGraphNode, graph, pDependencies, dependencyData, numDependencies, nodeParams)
    return cudaGraphAddNode(pGraphNode, graph, pDependencies, dependencyData, numDependencies, nodeParams)

cdef cudaError_t _cudaGraphNodeSetParams(cudaGraphNode_t node, cudaGraphNodeParams* nodeParams) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaGraphNodeSetParams(node, nodeParams)
    return cudaGraphNodeSetParams(node, nodeParams)

cdef cudaError_t _cudaGraphNodeGetParams(cudaGraphNode_t node, cudaGraphNodeParams* nodeParams) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaGraphNodeGetParams(node, nodeParams)
    return cudaGraphNodeGetParams(node, nodeParams)

cdef cudaError_t _cudaGraphExecNodeSetParams(cudaGraphExec_t graphExec, cudaGraphNode_t node, cudaGraphNodeParams* nodeParams) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaGraphExecNodeSetParams(graphExec, node, nodeParams)
    return cudaGraphExecNodeSetParams(graphExec, node, nodeParams)

cdef cudaError_t _cudaGraphConditionalHandleCreate(cudaGraphConditionalHandle* pHandle_out, cudaGraph_t graph, unsigned int defaultLaunchValue, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaGraphConditionalHandleCreate(pHandle_out, graph, defaultLaunchValue, flags)
    return cudaGraphConditionalHandleCreate(pHandle_out, graph, defaultLaunchValue, flags)

cdef cudaError_t _cudaGraphConditionalHandleCreate_v2(cudaGraphConditionalHandle* pHandle_out, cudaGraph_t graph, cudaExecutionContext_t ctx, unsigned int defaultLaunchValue, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaGraphConditionalHandleCreate_v2(pHandle_out, graph, ctx, defaultLaunchValue, flags)
    return cudaGraphConditionalHandleCreate_v2(pHandle_out, graph, ctx, defaultLaunchValue, flags)

cdef cudaError_t _cudaGetDriverEntryPoint(const char* symbol, void** funcPtr, unsigned long long flags, cudaDriverEntryPointQueryResult* driverStatus) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaGetDriverEntryPoint(symbol, funcPtr, flags, driverStatus)
    return cudaGetDriverEntryPoint(symbol, funcPtr, flags, driverStatus)

cdef cudaError_t _cudaGetDriverEntryPointByVersion(const char* symbol, void** funcPtr, unsigned int cudaVersion, unsigned long long flags, cudaDriverEntryPointQueryResult* driverStatus) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaGetDriverEntryPointByVersion(symbol, funcPtr, cudaVersion, flags, driverStatus)
    return cudaGetDriverEntryPointByVersion(symbol, funcPtr, cudaVersion, flags, driverStatus)

cdef cudaError_t _cudaLibraryLoadData(cudaLibrary_t* library, const void* code, cudaJitOption* jitOptions, void** jitOptionsValues, unsigned int numJitOptions, cudaLibraryOption* libraryOptions, void** libraryOptionValues, unsigned int numLibraryOptions) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaLibraryLoadData(library, code, jitOptions, jitOptionsValues, numJitOptions, libraryOptions, libraryOptionValues, numLibraryOptions)
    return cudaLibraryLoadData(library, code, jitOptions, jitOptionsValues, numJitOptions, libraryOptions, libraryOptionValues, numLibraryOptions)

cdef cudaError_t _cudaLibraryLoadFromFile(cudaLibrary_t* library, const char* fileName, cudaJitOption* jitOptions, void** jitOptionsValues, unsigned int numJitOptions, cudaLibraryOption* libraryOptions, void** libraryOptionValues, unsigned int numLibraryOptions) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaLibraryLoadFromFile(library, fileName, jitOptions, jitOptionsValues, numJitOptions, libraryOptions, libraryOptionValues, numLibraryOptions)
    return cudaLibraryLoadFromFile(library, fileName, jitOptions, jitOptionsValues, numJitOptions, libraryOptions, libraryOptionValues, numLibraryOptions)

cdef cudaError_t _cudaLibraryUnload(cudaLibrary_t library) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaLibraryUnload(library)
    return cudaLibraryUnload(library)

cdef cudaError_t _cudaLibraryGetKernel(cudaKernel_t* pKernel, cudaLibrary_t library, const char* name) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaLibraryGetKernel(pKernel, library, name)
    return cudaLibraryGetKernel(pKernel, library, name)

cdef cudaError_t _cudaLibraryGetGlobal(void** dptr, size_t* numbytes, cudaLibrary_t library, const char* name) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaLibraryGetGlobal(dptr, numbytes, library, name)
    return cudaLibraryGetGlobal(dptr, numbytes, library, name)

cdef cudaError_t _cudaLibraryGetManaged(void** dptr, size_t* numbytes, cudaLibrary_t library, const char* name) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaLibraryGetManaged(dptr, numbytes, library, name)
    return cudaLibraryGetManaged(dptr, numbytes, library, name)

cdef cudaError_t _cudaLibraryGetUnifiedFunction(void** fptr, cudaLibrary_t library, const char* symbol) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaLibraryGetUnifiedFunction(fptr, library, symbol)
    return cudaLibraryGetUnifiedFunction(fptr, library, symbol)

cdef cudaError_t _cudaLibraryGetKernelCount(unsigned int* count, cudaLibrary_t lib) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaLibraryGetKernelCount(count, lib)
    return cudaLibraryGetKernelCount(count, lib)

cdef cudaError_t _cudaLibraryEnumerateKernels(cudaKernel_t* kernels, unsigned int numKernels, cudaLibrary_t lib) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaLibraryEnumerateKernels(kernels, numKernels, lib)
    return cudaLibraryEnumerateKernels(kernels, numKernels, lib)

cdef cudaError_t _cudaKernelSetAttributeForDevice(cudaKernel_t kernel, cudaFuncAttribute attr, int value, int device) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaKernelSetAttributeForDevice(kernel, attr, value, device)
    return cudaKernelSetAttributeForDevice(kernel, attr, value, device)

cdef cudaError_t _cudaDeviceGetDevResource(int device, cudaDevResource* resource, cudaDevResourceType typename) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaDeviceGetDevResource(device, resource, typename)
    return cudaDeviceGetDevResource(device, resource, typename)

cdef cudaError_t _cudaDevSmResourceSplitByCount(cudaDevResource* result, unsigned int* nbGroups, const cudaDevResource* input, cudaDevResource* remaining, unsigned int flags, unsigned int minCount) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaDevSmResourceSplitByCount(result, nbGroups, input, remaining, flags, minCount)
    return cudaDevSmResourceSplitByCount(result, nbGroups, input, remaining, flags, minCount)

cdef cudaError_t _cudaDevSmResourceSplit(cudaDevResource* result, unsigned int nbGroups, const cudaDevResource* input, cudaDevResource* remainder, unsigned int flags, cudaDevSmResourceGroupParams* groupParams) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaDevSmResourceSplit(result, nbGroups, input, remainder, flags, groupParams)
    return cudaDevSmResourceSplit(result, nbGroups, input, remainder, flags, groupParams)

cdef cudaError_t _cudaDevResourceGenerateDesc(cudaDevResourceDesc_t* phDesc, cudaDevResource* resources, unsigned int nbResources) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaDevResourceGenerateDesc(phDesc, resources, nbResources)
    return cudaDevResourceGenerateDesc(phDesc, resources, nbResources)

cdef cudaError_t _cudaGreenCtxCreate(cudaExecutionContext_t* phCtx, cudaDevResourceDesc_t desc, int device, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaGreenCtxCreate(phCtx, desc, device, flags)
    return cudaGreenCtxCreate(phCtx, desc, device, flags)

cdef cudaError_t _cudaExecutionCtxDestroy(cudaExecutionContext_t ctx) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaExecutionCtxDestroy(ctx)
    return cudaExecutionCtxDestroy(ctx)

cdef cudaError_t _cudaExecutionCtxGetDevResource(cudaExecutionContext_t ctx, cudaDevResource* resource, cudaDevResourceType typename) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaExecutionCtxGetDevResource(ctx, resource, typename)
    return cudaExecutionCtxGetDevResource(ctx, resource, typename)

cdef cudaError_t _cudaExecutionCtxGetDevice(int* device, cudaExecutionContext_t ctx) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaExecutionCtxGetDevice(device, ctx)
    return cudaExecutionCtxGetDevice(device, ctx)

cdef cudaError_t _cudaExecutionCtxGetId(cudaExecutionContext_t ctx, unsigned long long* ctxId) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaExecutionCtxGetId(ctx, ctxId)
    return cudaExecutionCtxGetId(ctx, ctxId)

cdef cudaError_t _cudaExecutionCtxStreamCreate(cudaStream_t* phStream, cudaExecutionContext_t ctx, unsigned int flags, int priority) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaExecutionCtxStreamCreate(phStream, ctx, flags, priority)
    return cudaExecutionCtxStreamCreate(phStream, ctx, flags, priority)

cdef cudaError_t _cudaExecutionCtxSynchronize(cudaExecutionContext_t ctx) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaExecutionCtxSynchronize(ctx)
    return cudaExecutionCtxSynchronize(ctx)

cdef cudaError_t _cudaStreamGetDevResource(cudaStream_t hStream, cudaDevResource* resource, cudaDevResourceType typename) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaStreamGetDevResource(hStream, resource, typename)
    return cudaStreamGetDevResource(hStream, resource, typename)

cdef cudaError_t _cudaExecutionCtxRecordEvent(cudaExecutionContext_t ctx, cudaEvent_t event) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaExecutionCtxRecordEvent(ctx, event)
    return cudaExecutionCtxRecordEvent(ctx, event)

cdef cudaError_t _cudaExecutionCtxWaitEvent(cudaExecutionContext_t ctx, cudaEvent_t event) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaExecutionCtxWaitEvent(ctx, event)
    return cudaExecutionCtxWaitEvent(ctx, event)

cdef cudaError_t _cudaDeviceGetExecutionCtx(cudaExecutionContext_t* ctx, int device) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaDeviceGetExecutionCtx(ctx, device)
    return cudaDeviceGetExecutionCtx(ctx, device)

cdef cudaError_t _cudaGetExportTable(const void** ppExportTable, const cudaUUID_t* pExportTableId) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaGetExportTable(ppExportTable, pExportTableId)
    return cudaGetExportTable(ppExportTable, pExportTableId)

cdef cudaError_t _cudaGetKernel(cudaKernel_t* kernelPtr, const void* entryFuncAddr) except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaGetKernel(kernelPtr, entryFuncAddr)
    return cudaGetKernel(kernelPtr, entryFuncAddr)
@cython.show_performance_hints(False)
cdef cudaPitchedPtr _make_cudaPitchedPtr(void* d, size_t p, size_t xsz, size_t ysz) except* nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._make_cudaPitchedPtr(d, p, xsz, ysz)
    return make_cudaPitchedPtr(d, p, xsz, ysz)
@cython.show_performance_hints(False)
cdef cudaPos _make_cudaPos(size_t x, size_t y, size_t z) except* nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._make_cudaPos(x, y, z)
    return make_cudaPos(x, y, z)
@cython.show_performance_hints(False)
cdef cudaExtent _make_cudaExtent(size_t w, size_t h, size_t d) except* nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._make_cudaExtent(w, h, d)
    return make_cudaExtent(w, h, d)

cdef cudaError_t _cudaProfilerStart() except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaProfilerStart()
    return cudaProfilerStart()

cdef cudaError_t _cudaProfilerStop() except ?cudaErrorCallRequiresNewerDriver nogil:
    cdef bint usePTDS = cudaPythonInit()
    if usePTDS:
        return ptds._cudaProfilerStop()
    return cudaProfilerStop()


include "../_lib/cyruntime/cyruntime.pxi"
