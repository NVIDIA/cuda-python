# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

# This code was automatically generated with version 13.3.0, generator version 0.3.1.dev1630+gadce055ea.d20260422. Do not modify it directly.
cdef extern from "":
    """
    #define CUDA_API_PER_THREAD_DEFAULT_STREAM
    """

include "../cyruntime_functions.pxi"

cimport cython

cdef cudaError_t _cudaDeviceReset() except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaDeviceReset()

cdef cudaError_t _cudaDeviceSynchronize() except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaDeviceSynchronize()

cdef cudaError_t _cudaDeviceSetLimit(cudaLimit limit, size_t value) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaDeviceSetLimit(limit, value)

cdef cudaError_t _cudaDeviceGetLimit(size_t* pValue, cudaLimit limit) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaDeviceGetLimit(pValue, limit)

cdef cudaError_t _cudaDeviceGetTexture1DLinearMaxWidth(size_t* maxWidthInElements, const cudaChannelFormatDesc* fmtDesc, int device) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaDeviceGetTexture1DLinearMaxWidth(maxWidthInElements, fmtDesc, device)

cdef cudaError_t _cudaDeviceGetCacheConfig(cudaFuncCache* pCacheConfig) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaDeviceGetCacheConfig(pCacheConfig)

cdef cudaError_t _cudaDeviceGetStreamPriorityRange(int* leastPriority, int* greatestPriority) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaDeviceGetStreamPriorityRange(leastPriority, greatestPriority)

cdef cudaError_t _cudaDeviceSetCacheConfig(cudaFuncCache cacheConfig) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaDeviceSetCacheConfig(cacheConfig)

cdef cudaError_t _cudaDeviceGetByPCIBusId(int* device, const char* pciBusId) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaDeviceGetByPCIBusId(device, pciBusId)

cdef cudaError_t _cudaDeviceGetPCIBusId(char* pciBusId, int length, int device) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaDeviceGetPCIBusId(pciBusId, length, device)

cdef cudaError_t _cudaIpcGetEventHandle(cudaIpcEventHandle_t* handle, cudaEvent_t event) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaIpcGetEventHandle(handle, event)

cdef cudaError_t _cudaIpcOpenEventHandle(cudaEvent_t* event, cudaIpcEventHandle_t handle) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaIpcOpenEventHandle(event, handle)

cdef cudaError_t _cudaIpcGetMemHandle(cudaIpcMemHandle_t* handle, void* devPtr) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaIpcGetMemHandle(handle, devPtr)

cdef cudaError_t _cudaIpcOpenMemHandle(void** devPtr, cudaIpcMemHandle_t handle, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaIpcOpenMemHandle(devPtr, handle, flags)

cdef cudaError_t _cudaIpcCloseMemHandle(void* devPtr) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaIpcCloseMemHandle(devPtr)

cdef cudaError_t _cudaDeviceFlushGPUDirectRDMAWrites(cudaFlushGPUDirectRDMAWritesTarget target, cudaFlushGPUDirectRDMAWritesScope scope) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaDeviceFlushGPUDirectRDMAWrites(target, scope)

cdef cudaError_t _cudaDeviceRegisterAsyncNotification(int device, cudaAsyncCallback callbackFunc, void* userData, cudaAsyncCallbackHandle_t* callback) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaDeviceRegisterAsyncNotification(device, callbackFunc, userData, callback)

cdef cudaError_t _cudaDeviceUnregisterAsyncNotification(int device, cudaAsyncCallbackHandle_t callback) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaDeviceUnregisterAsyncNotification(device, callback)

cdef cudaError_t _cudaDeviceGetSharedMemConfig(cudaSharedMemConfig* pConfig) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaDeviceGetSharedMemConfig(pConfig)

cdef cudaError_t _cudaDeviceSetSharedMemConfig(cudaSharedMemConfig config) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaDeviceSetSharedMemConfig(config)

cdef cudaError_t _cudaGetLastError() except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaGetLastError()

cdef cudaError_t _cudaPeekAtLastError() except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaPeekAtLastError()

cdef const char* _cudaGetErrorName(cudaError_t error) except ?NULL nogil:
    return cudaGetErrorName(error)

cdef const char* _cudaGetErrorString(cudaError_t error) except ?NULL nogil:
    return cudaGetErrorString(error)

cdef cudaError_t _cudaGetDeviceCount(int* count) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaGetDeviceCount(count)

cdef cudaError_t _cudaGetDeviceProperties(cudaDeviceProp* prop, int device) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaGetDeviceProperties(prop, device)

cdef cudaError_t _cudaDeviceGetAttribute(int* value, cudaDeviceAttr attr, int device) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaDeviceGetAttribute(value, attr, device)

cdef cudaError_t _cudaDeviceGetHostAtomicCapabilities(unsigned int* capabilities, const cudaAtomicOperation* operations, unsigned int count, int device) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaDeviceGetHostAtomicCapabilities(capabilities, operations, count, device)

cdef cudaError_t _cudaDeviceGetDefaultMemPool(cudaMemPool_t* memPool, int device) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaDeviceGetDefaultMemPool(memPool, device)

cdef cudaError_t _cudaDeviceSetMemPool(int device, cudaMemPool_t memPool) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaDeviceSetMemPool(device, memPool)

cdef cudaError_t _cudaDeviceGetMemPool(cudaMemPool_t* memPool, int device) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaDeviceGetMemPool(memPool, device)

cdef cudaError_t _cudaDeviceGetNvSciSyncAttributes(void* nvSciSyncAttrList, int device, int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaDeviceGetNvSciSyncAttributes(nvSciSyncAttrList, device, flags)

cdef cudaError_t _cudaDeviceGetP2PAttribute(int* value, cudaDeviceP2PAttr attr, int srcDevice, int dstDevice) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaDeviceGetP2PAttribute(value, attr, srcDevice, dstDevice)

cdef cudaError_t _cudaDeviceGetP2PAtomicCapabilities(unsigned int* capabilities, const cudaAtomicOperation* operations, unsigned int count, int srcDevice, int dstDevice) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaDeviceGetP2PAtomicCapabilities(capabilities, operations, count, srcDevice, dstDevice)

cdef cudaError_t _cudaChooseDevice(int* device, const cudaDeviceProp* prop) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaChooseDevice(device, prop)

cdef cudaError_t _cudaInitDevice(int device, unsigned int deviceFlags, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaInitDevice(device, deviceFlags, flags)

cdef cudaError_t _cudaSetDevice(int device) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaSetDevice(device)

cdef cudaError_t _cudaGetDevice(int* device) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaGetDevice(device)

cdef cudaError_t _cudaSetDeviceFlags(unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaSetDeviceFlags(flags)

cdef cudaError_t _cudaGetDeviceFlags(unsigned int* flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaGetDeviceFlags(flags)

cdef cudaError_t _cudaStreamCreate(cudaStream_t* pStream) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaStreamCreate(pStream)

cdef cudaError_t _cudaStreamCreateWithFlags(cudaStream_t* pStream, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaStreamCreateWithFlags(pStream, flags)

cdef cudaError_t _cudaStreamCreateWithPriority(cudaStream_t* pStream, unsigned int flags, int priority) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaStreamCreateWithPriority(pStream, flags, priority)

cdef cudaError_t _cudaStreamGetPriority(cudaStream_t hStream, int* priority) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaStreamGetPriority(hStream, priority)

cdef cudaError_t _cudaStreamGetFlags(cudaStream_t hStream, unsigned int* flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaStreamGetFlags(hStream, flags)

cdef cudaError_t _cudaStreamGetId(cudaStream_t hStream, unsigned long long* streamId) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaStreamGetId(hStream, streamId)

cdef cudaError_t _cudaStreamGetDevice(cudaStream_t hStream, int* device) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaStreamGetDevice(hStream, device)

cdef cudaError_t _cudaCtxResetPersistingL2Cache() except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaCtxResetPersistingL2Cache()

cdef cudaError_t _cudaStreamCopyAttributes(cudaStream_t dst, cudaStream_t src) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaStreamCopyAttributes(dst, src)

cdef cudaError_t _cudaStreamGetAttribute(cudaStream_t hStream, cudaStreamAttrID attr, cudaStreamAttrValue* value_out) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaStreamGetAttribute(hStream, attr, value_out)

cdef cudaError_t _cudaStreamSetAttribute(cudaStream_t hStream, cudaStreamAttrID attr, const cudaStreamAttrValue* value) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaStreamSetAttribute(hStream, attr, value)

cdef cudaError_t _cudaStreamDestroy(cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaStreamDestroy(stream)

cdef cudaError_t _cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaStreamWaitEvent(stream, event, flags)

cdef cudaError_t _cudaStreamAddCallback(cudaStream_t stream, cudaStreamCallback_t callback, void* userData, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaStreamAddCallback(stream, callback, userData, flags)

cdef cudaError_t _cudaStreamSynchronize(cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaStreamSynchronize(stream)

cdef cudaError_t _cudaStreamQuery(cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaStreamQuery(stream)

cdef cudaError_t _cudaStreamAttachMemAsync(cudaStream_t stream, void* devPtr, size_t length, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaStreamAttachMemAsync(stream, devPtr, length, flags)

cdef cudaError_t _cudaStreamBeginCapture(cudaStream_t stream, cudaStreamCaptureMode mode) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaStreamBeginCapture(stream, mode)

cdef cudaError_t _cudaStreamBeginRecaptureToGraph(cudaStream_t stream, cudaStreamCaptureMode mode, cudaGraph_t graph, cudaGraphRecaptureCallbackData* callbackData) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaStreamBeginRecaptureToGraph(stream, mode, graph, callbackData)

cdef cudaError_t _cudaStreamBeginCaptureToGraph(cudaStream_t stream, cudaGraph_t graph, const cudaGraphNode_t* dependencies, const cudaGraphEdgeData* dependencyData, size_t numDependencies, cudaStreamCaptureMode mode) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaStreamBeginCaptureToGraph(stream, graph, dependencies, dependencyData, numDependencies, mode)

cdef cudaError_t _cudaThreadExchangeStreamCaptureMode(cudaStreamCaptureMode* mode) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaThreadExchangeStreamCaptureMode(mode)

cdef cudaError_t _cudaStreamEndCapture(cudaStream_t stream, cudaGraph_t* pGraph) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaStreamEndCapture(stream, pGraph)

cdef cudaError_t _cudaStreamIsCapturing(cudaStream_t stream, cudaStreamCaptureStatus* pCaptureStatus) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaStreamIsCapturing(stream, pCaptureStatus)

cdef cudaError_t _cudaStreamGetCaptureInfo(cudaStream_t stream, cudaStreamCaptureStatus* captureStatus_out, unsigned long long* id_out, cudaGraph_t* graph_out, const cudaGraphNode_t** dependencies_out, const cudaGraphEdgeData** edgeData_out, size_t* numDependencies_out) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaStreamGetCaptureInfo(stream, captureStatus_out, id_out, graph_out, dependencies_out, edgeData_out, numDependencies_out)

cdef cudaError_t _cudaStreamUpdateCaptureDependencies(cudaStream_t stream, cudaGraphNode_t* dependencies, const cudaGraphEdgeData* dependencyData, size_t numDependencies, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaStreamUpdateCaptureDependencies(stream, dependencies, dependencyData, numDependencies, flags)

cdef cudaError_t _cudaEventCreate(cudaEvent_t* event) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaEventCreate(event)

cdef cudaError_t _cudaEventCreateWithFlags(cudaEvent_t* event, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaEventCreateWithFlags(event, flags)

cdef cudaError_t _cudaEventRecord(cudaEvent_t event, cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaEventRecord(event, stream)

cdef cudaError_t _cudaEventRecordWithFlags(cudaEvent_t event, cudaStream_t stream, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaEventRecordWithFlags(event, stream, flags)

cdef cudaError_t _cudaEventQuery(cudaEvent_t event) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaEventQuery(event)

cdef cudaError_t _cudaEventSynchronize(cudaEvent_t event) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaEventSynchronize(event)

cdef cudaError_t _cudaEventDestroy(cudaEvent_t event) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaEventDestroy(event)

cdef cudaError_t _cudaEventElapsedTime(float* ms, cudaEvent_t start, cudaEvent_t end) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaEventElapsedTime(ms, start, end)

cdef cudaError_t _cudaImportExternalMemory(cudaExternalMemory_t* extMem_out, const cudaExternalMemoryHandleDesc* memHandleDesc) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaImportExternalMemory(extMem_out, memHandleDesc)

cdef cudaError_t _cudaExternalMemoryGetMappedBuffer(void** devPtr, cudaExternalMemory_t extMem, const cudaExternalMemoryBufferDesc* bufferDesc) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaExternalMemoryGetMappedBuffer(devPtr, extMem, bufferDesc)

cdef cudaError_t _cudaExternalMemoryGetMappedMipmappedArray(cudaMipmappedArray_t* mipmap, cudaExternalMemory_t extMem, const cudaExternalMemoryMipmappedArrayDesc* mipmapDesc) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaExternalMemoryGetMappedMipmappedArray(mipmap, extMem, mipmapDesc)

cdef cudaError_t _cudaDestroyExternalMemory(cudaExternalMemory_t extMem) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaDestroyExternalMemory(extMem)

cdef cudaError_t _cudaImportExternalSemaphore(cudaExternalSemaphore_t* extSem_out, const cudaExternalSemaphoreHandleDesc* semHandleDesc) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaImportExternalSemaphore(extSem_out, semHandleDesc)

cdef cudaError_t _cudaSignalExternalSemaphoresAsync(const cudaExternalSemaphore_t* extSemArray, const cudaExternalSemaphoreSignalParams* paramsArray, unsigned int numExtSems, cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaSignalExternalSemaphoresAsync(extSemArray, paramsArray, numExtSems, stream)

cdef cudaError_t _cudaWaitExternalSemaphoresAsync(const cudaExternalSemaphore_t* extSemArray, const cudaExternalSemaphoreWaitParams* paramsArray, unsigned int numExtSems, cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaWaitExternalSemaphoresAsync(extSemArray, paramsArray, numExtSems, stream)

cdef cudaError_t _cudaDestroyExternalSemaphore(cudaExternalSemaphore_t extSem) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaDestroyExternalSemaphore(extSem)

cdef cudaError_t _cudaFuncSetCacheConfig(const void* func, cudaFuncCache cacheConfig) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaFuncSetCacheConfig(func, cacheConfig)

cdef cudaError_t _cudaFuncGetAttributes(cudaFuncAttributes* attr, const void* func) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaFuncGetAttributes(attr, func)

cdef cudaError_t _cudaFuncSetAttribute(const void* func, cudaFuncAttribute attr, int value) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaFuncSetAttribute(func, attr, value)

cdef cudaError_t _cudaFuncGetParamCount(const void* func, size_t* paramCount) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaFuncGetParamCount(func, paramCount)

cdef cudaError_t _cudaLaunchHostFunc(cudaStream_t stream, cudaHostFn_t fn, void* userData) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaLaunchHostFunc(stream, fn, userData)

cdef cudaError_t _cudaLaunchHostFunc_v2(cudaStream_t stream, cudaHostFn_t fn, void* userData, unsigned int syncMode) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaLaunchHostFunc_v2(stream, fn, userData, syncMode)

cdef cudaError_t _cudaFuncSetSharedMemConfig(const void* func, cudaSharedMemConfig config) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaFuncSetSharedMemConfig(func, config)

cdef cudaError_t _cudaOccupancyMaxActiveBlocksPerMultiprocessor(int* numBlocks, const void* func, int blockSize, size_t dynamicSMemSize) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaOccupancyMaxActiveBlocksPerMultiprocessor(numBlocks, func, blockSize, dynamicSMemSize)

cdef cudaError_t _cudaOccupancyAvailableDynamicSMemPerBlock(size_t* dynamicSmemSize, const void* func, int numBlocks, int blockSize) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaOccupancyAvailableDynamicSMemPerBlock(dynamicSmemSize, func, numBlocks, blockSize)

cdef cudaError_t _cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int* numBlocks, const void* func, int blockSize, size_t dynamicSMemSize, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(numBlocks, func, blockSize, dynamicSMemSize, flags)

cdef cudaError_t _cudaMallocManaged(void** devPtr, size_t size, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaMallocManaged(devPtr, size, flags)

cdef cudaError_t _cudaMalloc(void** devPtr, size_t size) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaMalloc(devPtr, size)

cdef cudaError_t _cudaMallocHost(void** ptr, size_t size) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaMallocHost(ptr, size)

cdef cudaError_t _cudaMallocPitch(void** devPtr, size_t* pitch, size_t width, size_t height) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaMallocPitch(devPtr, pitch, width, height)

cdef cudaError_t _cudaMallocArray(cudaArray_t* array, const cudaChannelFormatDesc* desc, size_t width, size_t height, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaMallocArray(array, desc, width, height, flags)

cdef cudaError_t _cudaFree(void* devPtr) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaFree(devPtr)

cdef cudaError_t _cudaFreeHost(void* ptr) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaFreeHost(ptr)

cdef cudaError_t _cudaFreeArray(cudaArray_t array) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaFreeArray(array)

cdef cudaError_t _cudaFreeMipmappedArray(cudaMipmappedArray_t mipmappedArray) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaFreeMipmappedArray(mipmappedArray)

cdef cudaError_t _cudaHostAlloc(void** pHost, size_t size, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaHostAlloc(pHost, size, flags)

cdef cudaError_t _cudaHostRegister(void* ptr, size_t size, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaHostRegister(ptr, size, flags)

cdef cudaError_t _cudaHostUnregister(void* ptr) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaHostUnregister(ptr)

cdef cudaError_t _cudaHostGetDevicePointer(void** pDevice, void* pHost, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaHostGetDevicePointer(pDevice, pHost, flags)

cdef cudaError_t _cudaHostGetFlags(unsigned int* pFlags, void* pHost) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaHostGetFlags(pFlags, pHost)

cdef cudaError_t _cudaMalloc3D(cudaPitchedPtr* pitchedDevPtr, cudaExtent extent) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaMalloc3D(pitchedDevPtr, extent)

cdef cudaError_t _cudaMalloc3DArray(cudaArray_t* array, const cudaChannelFormatDesc* desc, cudaExtent extent, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaMalloc3DArray(array, desc, extent, flags)

cdef cudaError_t _cudaMallocMipmappedArray(cudaMipmappedArray_t* mipmappedArray, const cudaChannelFormatDesc* desc, cudaExtent extent, unsigned int numLevels, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaMallocMipmappedArray(mipmappedArray, desc, extent, numLevels, flags)

cdef cudaError_t _cudaGetMipmappedArrayLevel(cudaArray_t* levelArray, cudaMipmappedArray_const_t mipmappedArray, unsigned int level) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaGetMipmappedArrayLevel(levelArray, mipmappedArray, level)

cdef cudaError_t _cudaMemcpy3D(const cudaMemcpy3DParms* p) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaMemcpy3D(p)

cdef cudaError_t _cudaMemcpy3DPeer(const cudaMemcpy3DPeerParms* p) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaMemcpy3DPeer(p)

cdef cudaError_t _cudaMemcpy3DAsync(const cudaMemcpy3DParms* p, cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaMemcpy3DAsync(p, stream)

cdef cudaError_t _cudaMemcpy3DPeerAsync(const cudaMemcpy3DPeerParms* p, cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaMemcpy3DPeerAsync(p, stream)

cdef cudaError_t _cudaMemGetInfo(size_t* free, size_t* total) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaMemGetInfo(free, total)

cdef cudaError_t _cudaArrayGetInfo(cudaChannelFormatDesc* desc, cudaExtent* extent, unsigned int* flags, cudaArray_t array) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaArrayGetInfo(desc, extent, flags, array)

cdef cudaError_t _cudaArrayGetPlane(cudaArray_t* pPlaneArray, cudaArray_t hArray, unsigned int planeIdx) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaArrayGetPlane(pPlaneArray, hArray, planeIdx)

cdef cudaError_t _cudaArrayGetMemoryRequirements(cudaArrayMemoryRequirements* memoryRequirements, cudaArray_t array, int device) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaArrayGetMemoryRequirements(memoryRequirements, array, device)

cdef cudaError_t _cudaMipmappedArrayGetMemoryRequirements(cudaArrayMemoryRequirements* memoryRequirements, cudaMipmappedArray_t mipmap, int device) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaMipmappedArrayGetMemoryRequirements(memoryRequirements, mipmap, device)

cdef cudaError_t _cudaArrayGetSparseProperties(cudaArraySparseProperties* sparseProperties, cudaArray_t array) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaArrayGetSparseProperties(sparseProperties, array)

cdef cudaError_t _cudaMipmappedArrayGetSparseProperties(cudaArraySparseProperties* sparseProperties, cudaMipmappedArray_t mipmap) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaMipmappedArrayGetSparseProperties(sparseProperties, mipmap)

cdef cudaError_t _cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaMemcpy(dst, src, count, kind)

cdef cudaError_t _cudaMemcpyPeer(void* dst, int dstDevice, const void* src, int srcDevice, size_t count) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaMemcpyPeer(dst, dstDevice, src, srcDevice, count)

cdef cudaError_t _cudaMemcpy2D(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaMemcpy2D(dst, dpitch, src, spitch, width, height, kind)

cdef cudaError_t _cudaMemcpy2DToArray(cudaArray_t dst, size_t wOffset, size_t hOffset, const void* src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaMemcpy2DToArray(dst, wOffset, hOffset, src, spitch, width, height, kind)

cdef cudaError_t _cudaMemcpy2DFromArray(void* dst, size_t dpitch, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t width, size_t height, cudaMemcpyKind kind) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaMemcpy2DFromArray(dst, dpitch, src, wOffset, hOffset, width, height, kind)

cdef cudaError_t _cudaMemcpy2DArrayToArray(cudaArray_t dst, size_t wOffsetDst, size_t hOffsetDst, cudaArray_const_t src, size_t wOffsetSrc, size_t hOffsetSrc, size_t width, size_t height, cudaMemcpyKind kind) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaMemcpy2DArrayToArray(dst, wOffsetDst, hOffsetDst, src, wOffsetSrc, hOffsetSrc, width, height, kind)

cdef cudaError_t _cudaMemcpyAsync(void* dst, const void* src, size_t count, cudaMemcpyKind kind, cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaMemcpyAsync(dst, src, count, kind, stream)

cdef cudaError_t _cudaMemcpyPeerAsync(void* dst, int dstDevice, const void* src, int srcDevice, size_t count, cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaMemcpyPeerAsync(dst, dstDevice, src, srcDevice, count, stream)

cdef cudaError_t _cudaMemcpyBatchAsync(const void** dsts, const void** srcs, const size_t* sizes, size_t count, cudaMemcpyAttributes* attrs, size_t* attrsIdxs, size_t numAttrs, cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaMemcpyBatchAsync(dsts, srcs, sizes, count, attrs, attrsIdxs, numAttrs, stream)

cdef cudaError_t _cudaMemcpy3DBatchAsync(size_t numOps, cudaMemcpy3DBatchOp* opList, unsigned long long flags, cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaMemcpy3DBatchAsync(numOps, opList, flags, stream)

cdef cudaError_t _cudaMemcpyWithAttributesAsync(void* dst, const void* src, size_t size, cudaMemcpyAttributes* attr, cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaMemcpyWithAttributesAsync(dst, src, size, attr, stream)

cdef cudaError_t _cudaMemcpy3DWithAttributesAsync(cudaMemcpy3DBatchOp* op, unsigned long long flags, cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaMemcpy3DWithAttributesAsync(op, flags, stream)

cdef cudaError_t _cudaMemcpy2DAsync(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaMemcpy2DAsync(dst, dpitch, src, spitch, width, height, kind, stream)

cdef cudaError_t _cudaMemcpy2DToArrayAsync(cudaArray_t dst, size_t wOffset, size_t hOffset, const void* src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaMemcpy2DToArrayAsync(dst, wOffset, hOffset, src, spitch, width, height, kind, stream)

cdef cudaError_t _cudaMemcpy2DFromArrayAsync(void* dst, size_t dpitch, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaMemcpy2DFromArrayAsync(dst, dpitch, src, wOffset, hOffset, width, height, kind, stream)

cdef cudaError_t _cudaMemset(void* devPtr, int value, size_t count) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaMemset(devPtr, value, count)

cdef cudaError_t _cudaMemset2D(void* devPtr, size_t pitch, int value, size_t width, size_t height) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaMemset2D(devPtr, pitch, value, width, height)

cdef cudaError_t _cudaMemset3D(cudaPitchedPtr pitchedDevPtr, int value, cudaExtent extent) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaMemset3D(pitchedDevPtr, value, extent)

cdef cudaError_t _cudaMemsetAsync(void* devPtr, int value, size_t count, cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaMemsetAsync(devPtr, value, count, stream)

cdef cudaError_t _cudaMemset2DAsync(void* devPtr, size_t pitch, int value, size_t width, size_t height, cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaMemset2DAsync(devPtr, pitch, value, width, height, stream)

cdef cudaError_t _cudaMemset3DAsync(cudaPitchedPtr pitchedDevPtr, int value, cudaExtent extent, cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaMemset3DAsync(pitchedDevPtr, value, extent, stream)

cdef cudaError_t _cudaMemPrefetchAsync(const void* devPtr, size_t count, cudaMemLocation location, unsigned int flags, cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaMemPrefetchAsync(devPtr, count, location, flags, stream)

cdef cudaError_t _cudaMemPrefetchBatchAsync(void** dptrs, size_t* sizes, size_t count, cudaMemLocation* prefetchLocs, size_t* prefetchLocIdxs, size_t numPrefetchLocs, unsigned long long flags, cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaMemPrefetchBatchAsync(dptrs, sizes, count, prefetchLocs, prefetchLocIdxs, numPrefetchLocs, flags, stream)

cdef cudaError_t _cudaMemDiscardBatchAsync(void** dptrs, size_t* sizes, size_t count, unsigned long long flags, cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaMemDiscardBatchAsync(dptrs, sizes, count, flags, stream)

cdef cudaError_t _cudaMemDiscardAndPrefetchBatchAsync(void** dptrs, size_t* sizes, size_t count, cudaMemLocation* prefetchLocs, size_t* prefetchLocIdxs, size_t numPrefetchLocs, unsigned long long flags, cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaMemDiscardAndPrefetchBatchAsync(dptrs, sizes, count, prefetchLocs, prefetchLocIdxs, numPrefetchLocs, flags, stream)

cdef cudaError_t _cudaMemAdvise(const void* devPtr, size_t count, cudaMemoryAdvise advice, cudaMemLocation location) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaMemAdvise(devPtr, count, advice, location)

cdef cudaError_t _cudaMemRangeGetAttribute(void* data, size_t dataSize, cudaMemRangeAttribute attribute, const void* devPtr, size_t count) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaMemRangeGetAttribute(data, dataSize, attribute, devPtr, count)

cdef cudaError_t _cudaMemRangeGetAttributes(void** data, size_t* dataSizes, cudaMemRangeAttribute* attributes, size_t numAttributes, const void* devPtr, size_t count) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaMemRangeGetAttributes(data, dataSizes, attributes, numAttributes, devPtr, count)

cdef cudaError_t _cudaMemcpyToArray(cudaArray_t dst, size_t wOffset, size_t hOffset, const void* src, size_t count, cudaMemcpyKind kind) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaMemcpyToArray(dst, wOffset, hOffset, src, count, kind)

cdef cudaError_t _cudaMemcpyFromArray(void* dst, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t count, cudaMemcpyKind kind) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaMemcpyFromArray(dst, src, wOffset, hOffset, count, kind)

cdef cudaError_t _cudaMemcpyArrayToArray(cudaArray_t dst, size_t wOffsetDst, size_t hOffsetDst, cudaArray_const_t src, size_t wOffsetSrc, size_t hOffsetSrc, size_t count, cudaMemcpyKind kind) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaMemcpyArrayToArray(dst, wOffsetDst, hOffsetDst, src, wOffsetSrc, hOffsetSrc, count, kind)

cdef cudaError_t _cudaMemcpyToArrayAsync(cudaArray_t dst, size_t wOffset, size_t hOffset, const void* src, size_t count, cudaMemcpyKind kind, cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaMemcpyToArrayAsync(dst, wOffset, hOffset, src, count, kind, stream)

cdef cudaError_t _cudaMemcpyFromArrayAsync(void* dst, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t count, cudaMemcpyKind kind, cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaMemcpyFromArrayAsync(dst, src, wOffset, hOffset, count, kind, stream)

cdef cudaError_t _cudaMallocAsync(void** devPtr, size_t size, cudaStream_t hStream) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaMallocAsync(devPtr, size, hStream)

cdef cudaError_t _cudaFreeAsync(void* devPtr, cudaStream_t hStream) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaFreeAsync(devPtr, hStream)

cdef cudaError_t _cudaMemPoolTrimTo(cudaMemPool_t memPool, size_t minBytesToKeep) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaMemPoolTrimTo(memPool, minBytesToKeep)

cdef cudaError_t _cudaMemPoolSetAttribute(cudaMemPool_t memPool, cudaMemPoolAttr attr, void* value) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaMemPoolSetAttribute(memPool, attr, value)

cdef cudaError_t _cudaMemPoolGetAttribute(cudaMemPool_t memPool, cudaMemPoolAttr attr, void* value) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaMemPoolGetAttribute(memPool, attr, value)

cdef cudaError_t _cudaMemPoolSetAccess(cudaMemPool_t memPool, const cudaMemAccessDesc* descList, size_t count) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaMemPoolSetAccess(memPool, descList, count)

cdef cudaError_t _cudaMemPoolGetAccess(cudaMemAccessFlags* flags, cudaMemPool_t memPool, cudaMemLocation* location) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaMemPoolGetAccess(flags, memPool, location)

cdef cudaError_t _cudaMemPoolCreate(cudaMemPool_t* memPool, const cudaMemPoolProps* poolProps) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaMemPoolCreate(memPool, poolProps)

cdef cudaError_t _cudaMemPoolDestroy(cudaMemPool_t memPool) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaMemPoolDestroy(memPool)

cdef cudaError_t _cudaMemGetDefaultMemPool(cudaMemPool_t* memPool, cudaMemLocation* location, cudaMemAllocationType typename) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaMemGetDefaultMemPool(memPool, location, typename)

cdef cudaError_t _cudaMemGetMemPool(cudaMemPool_t* memPool, cudaMemLocation* location, cudaMemAllocationType typename) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaMemGetMemPool(memPool, location, typename)

cdef cudaError_t _cudaMemSetMemPool(cudaMemLocation* location, cudaMemAllocationType typename, cudaMemPool_t memPool) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaMemSetMemPool(location, typename, memPool)

cdef cudaError_t _cudaMallocFromPoolAsync(void** ptr, size_t size, cudaMemPool_t memPool, cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaMallocFromPoolAsync(ptr, size, memPool, stream)

cdef cudaError_t _cudaMemPoolExportToShareableHandle(void* shareableHandle, cudaMemPool_t memPool, cudaMemAllocationHandleType handleType, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaMemPoolExportToShareableHandle(shareableHandle, memPool, handleType, flags)

cdef cudaError_t _cudaMemPoolImportFromShareableHandle(cudaMemPool_t* memPool, void* shareableHandle, cudaMemAllocationHandleType handleType, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaMemPoolImportFromShareableHandle(memPool, shareableHandle, handleType, flags)

cdef cudaError_t _cudaMemPoolExportPointer(cudaMemPoolPtrExportData* exportData, void* ptr) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaMemPoolExportPointer(exportData, ptr)

cdef cudaError_t _cudaMemPoolImportPointer(void** ptr, cudaMemPool_t memPool, cudaMemPoolPtrExportData* exportData) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaMemPoolImportPointer(ptr, memPool, exportData)

cdef cudaError_t _cudaPointerGetAttributes(cudaPointerAttributes* attributes, const void* ptr) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaPointerGetAttributes(attributes, ptr)

cdef cudaError_t _cudaDeviceCanAccessPeer(int* canAccessPeer, int device, int peerDevice) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaDeviceCanAccessPeer(canAccessPeer, device, peerDevice)

cdef cudaError_t _cudaDeviceEnablePeerAccess(int peerDevice, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaDeviceEnablePeerAccess(peerDevice, flags)

cdef cudaError_t _cudaDeviceDisablePeerAccess(int peerDevice) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaDeviceDisablePeerAccess(peerDevice)

cdef cudaError_t _cudaGraphicsUnregisterResource(cudaGraphicsResource_t resource) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaGraphicsUnregisterResource(resource)

cdef cudaError_t _cudaGraphicsResourceSetMapFlags(cudaGraphicsResource_t resource, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaGraphicsResourceSetMapFlags(resource, flags)

cdef cudaError_t _cudaGraphicsMapResources(int count, cudaGraphicsResource_t* resources, cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaGraphicsMapResources(count, resources, stream)

cdef cudaError_t _cudaGraphicsUnmapResources(int count, cudaGraphicsResource_t* resources, cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaGraphicsUnmapResources(count, resources, stream)

cdef cudaError_t _cudaGraphicsResourceGetMappedPointer(void** devPtr, size_t* size, cudaGraphicsResource_t resource) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaGraphicsResourceGetMappedPointer(devPtr, size, resource)

cdef cudaError_t _cudaGraphicsSubResourceGetMappedArray(cudaArray_t* array, cudaGraphicsResource_t resource, unsigned int arrayIndex, unsigned int mipLevel) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaGraphicsSubResourceGetMappedArray(array, resource, arrayIndex, mipLevel)

cdef cudaError_t _cudaGraphicsResourceGetMappedMipmappedArray(cudaMipmappedArray_t* mipmappedArray, cudaGraphicsResource_t resource) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaGraphicsResourceGetMappedMipmappedArray(mipmappedArray, resource)

cdef cudaError_t _cudaGetChannelDesc(cudaChannelFormatDesc* desc, cudaArray_const_t array) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaGetChannelDesc(desc, array)
@cython.show_performance_hints(False)
cdef cudaChannelFormatDesc _cudaCreateChannelDesc(int x, int y, int z, int w, cudaChannelFormatKind f) except* nogil:
    return cudaCreateChannelDesc(x, y, z, w, f)

cdef cudaError_t _cudaCreateTextureObject(cudaTextureObject_t* pTexObject, const cudaResourceDesc* pResDesc, const cudaTextureDesc* pTexDesc, const cudaResourceViewDesc* pResViewDesc) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaCreateTextureObject(pTexObject, pResDesc, pTexDesc, pResViewDesc)

cdef cudaError_t _cudaDestroyTextureObject(cudaTextureObject_t texObject) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaDestroyTextureObject(texObject)

cdef cudaError_t _cudaGetTextureObjectResourceDesc(cudaResourceDesc* pResDesc, cudaTextureObject_t texObject) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaGetTextureObjectResourceDesc(pResDesc, texObject)

cdef cudaError_t _cudaGetTextureObjectTextureDesc(cudaTextureDesc* pTexDesc, cudaTextureObject_t texObject) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaGetTextureObjectTextureDesc(pTexDesc, texObject)

cdef cudaError_t _cudaGetTextureObjectResourceViewDesc(cudaResourceViewDesc* pResViewDesc, cudaTextureObject_t texObject) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaGetTextureObjectResourceViewDesc(pResViewDesc, texObject)

cdef cudaError_t _cudaCreateSurfaceObject(cudaSurfaceObject_t* pSurfObject, const cudaResourceDesc* pResDesc) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaCreateSurfaceObject(pSurfObject, pResDesc)

cdef cudaError_t _cudaDestroySurfaceObject(cudaSurfaceObject_t surfObject) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaDestroySurfaceObject(surfObject)

cdef cudaError_t _cudaGetSurfaceObjectResourceDesc(cudaResourceDesc* pResDesc, cudaSurfaceObject_t surfObject) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaGetSurfaceObjectResourceDesc(pResDesc, surfObject)

cdef cudaError_t _cudaDriverGetVersion(int* driverVersion) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaDriverGetVersion(driverVersion)

cdef cudaError_t _cudaRuntimeGetVersion(int* runtimeVersion) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaRuntimeGetVersion(runtimeVersion)

cdef cudaError_t _cudaLogsRegisterCallback(cudaLogsCallback_t callbackFunc, void* userData, cudaLogsCallbackHandle* callback_out) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaLogsRegisterCallback(callbackFunc, userData, callback_out)

cdef cudaError_t _cudaLogsUnregisterCallback(cudaLogsCallbackHandle callback) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaLogsUnregisterCallback(callback)

cdef cudaError_t _cudaLogsCurrent(cudaLogIterator* iterator_out, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaLogsCurrent(iterator_out, flags)

cdef cudaError_t _cudaLogsDumpToFile(cudaLogIterator* iterator, const char* pathToFile, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaLogsDumpToFile(iterator, pathToFile, flags)

cdef cudaError_t _cudaLogsDumpToMemory(cudaLogIterator* iterator, char* buffer, size_t* size, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaLogsDumpToMemory(iterator, buffer, size, flags)

cdef cudaError_t _cudaGraphCreate(cudaGraph_t* pGraph, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaGraphCreate(pGraph, flags)

cdef cudaError_t _cudaGraphAddKernelNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, const cudaKernelNodeParams* pNodeParams) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaGraphAddKernelNode(pGraphNode, graph, pDependencies, numDependencies, pNodeParams)

cdef cudaError_t _cudaGraphKernelNodeGetParams(cudaGraphNode_t node, cudaKernelNodeParams* pNodeParams) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaGraphKernelNodeGetParams(node, pNodeParams)

cdef cudaError_t _cudaGraphKernelNodeSetParams(cudaGraphNode_t node, const cudaKernelNodeParams* pNodeParams) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaGraphKernelNodeSetParams(node, pNodeParams)

cdef cudaError_t _cudaGraphKernelNodeCopyAttributes(cudaGraphNode_t hDst, cudaGraphNode_t hSrc) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaGraphKernelNodeCopyAttributes(hDst, hSrc)

cdef cudaError_t _cudaGraphKernelNodeGetAttribute(cudaGraphNode_t hNode, cudaKernelNodeAttrID attr, cudaKernelNodeAttrValue* value_out) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaGraphKernelNodeGetAttribute(hNode, attr, value_out)

cdef cudaError_t _cudaGraphKernelNodeSetAttribute(cudaGraphNode_t hNode, cudaKernelNodeAttrID attr, const cudaKernelNodeAttrValue* value) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaGraphKernelNodeSetAttribute(hNode, attr, value)

cdef cudaError_t _cudaGraphAddMemcpyNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, const cudaMemcpy3DParms* pCopyParams) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaGraphAddMemcpyNode(pGraphNode, graph, pDependencies, numDependencies, pCopyParams)

cdef cudaError_t _cudaGraphAddMemcpyNode1D(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, void* dst, const void* src, size_t count, cudaMemcpyKind kind) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaGraphAddMemcpyNode1D(pGraphNode, graph, pDependencies, numDependencies, dst, src, count, kind)

cdef cudaError_t _cudaGraphMemcpyNodeGetParams(cudaGraphNode_t node, cudaMemcpy3DParms* pNodeParams) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaGraphMemcpyNodeGetParams(node, pNodeParams)

cdef cudaError_t _cudaGraphMemcpyNodeSetParams(cudaGraphNode_t node, const cudaMemcpy3DParms* pNodeParams) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaGraphMemcpyNodeSetParams(node, pNodeParams)

cdef cudaError_t _cudaGraphMemcpyNodeSetParams1D(cudaGraphNode_t node, void* dst, const void* src, size_t count, cudaMemcpyKind kind) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaGraphMemcpyNodeSetParams1D(node, dst, src, count, kind)

cdef cudaError_t _cudaGraphAddMemsetNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, const cudaMemsetParams* pMemsetParams) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaGraphAddMemsetNode(pGraphNode, graph, pDependencies, numDependencies, pMemsetParams)

cdef cudaError_t _cudaGraphMemsetNodeGetParams(cudaGraphNode_t node, cudaMemsetParams* pNodeParams) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaGraphMemsetNodeGetParams(node, pNodeParams)

cdef cudaError_t _cudaGraphMemsetNodeSetParams(cudaGraphNode_t node, const cudaMemsetParams* pNodeParams) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaGraphMemsetNodeSetParams(node, pNodeParams)

cdef cudaError_t _cudaGraphAddHostNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, const cudaHostNodeParams* pNodeParams) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaGraphAddHostNode(pGraphNode, graph, pDependencies, numDependencies, pNodeParams)

cdef cudaError_t _cudaGraphHostNodeGetParams(cudaGraphNode_t node, cudaHostNodeParams* pNodeParams) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaGraphHostNodeGetParams(node, pNodeParams)

cdef cudaError_t _cudaGraphHostNodeSetParams(cudaGraphNode_t node, const cudaHostNodeParams* pNodeParams) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaGraphHostNodeSetParams(node, pNodeParams)

cdef cudaError_t _cudaGraphAddChildGraphNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, cudaGraph_t childGraph) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaGraphAddChildGraphNode(pGraphNode, graph, pDependencies, numDependencies, childGraph)

cdef cudaError_t _cudaGraphChildGraphNodeGetGraph(cudaGraphNode_t node, cudaGraph_t* pGraph) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaGraphChildGraphNodeGetGraph(node, pGraph)

cdef cudaError_t _cudaGraphAddEmptyNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaGraphAddEmptyNode(pGraphNode, graph, pDependencies, numDependencies)

cdef cudaError_t _cudaGraphAddEventRecordNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, cudaEvent_t event) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaGraphAddEventRecordNode(pGraphNode, graph, pDependencies, numDependencies, event)

cdef cudaError_t _cudaGraphEventRecordNodeGetEvent(cudaGraphNode_t node, cudaEvent_t* event_out) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaGraphEventRecordNodeGetEvent(node, event_out)

cdef cudaError_t _cudaGraphEventRecordNodeSetEvent(cudaGraphNode_t node, cudaEvent_t event) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaGraphEventRecordNodeSetEvent(node, event)

cdef cudaError_t _cudaGraphAddEventWaitNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, cudaEvent_t event) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaGraphAddEventWaitNode(pGraphNode, graph, pDependencies, numDependencies, event)

cdef cudaError_t _cudaGraphEventWaitNodeGetEvent(cudaGraphNode_t node, cudaEvent_t* event_out) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaGraphEventWaitNodeGetEvent(node, event_out)

cdef cudaError_t _cudaGraphEventWaitNodeSetEvent(cudaGraphNode_t node, cudaEvent_t event) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaGraphEventWaitNodeSetEvent(node, event)

cdef cudaError_t _cudaGraphAddExternalSemaphoresSignalNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, const cudaExternalSemaphoreSignalNodeParams* nodeParams) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaGraphAddExternalSemaphoresSignalNode(pGraphNode, graph, pDependencies, numDependencies, nodeParams)

cdef cudaError_t _cudaGraphExternalSemaphoresSignalNodeGetParams(cudaGraphNode_t hNode, cudaExternalSemaphoreSignalNodeParams* params_out) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaGraphExternalSemaphoresSignalNodeGetParams(hNode, params_out)

cdef cudaError_t _cudaGraphExternalSemaphoresSignalNodeSetParams(cudaGraphNode_t hNode, const cudaExternalSemaphoreSignalNodeParams* nodeParams) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaGraphExternalSemaphoresSignalNodeSetParams(hNode, nodeParams)

cdef cudaError_t _cudaGraphAddExternalSemaphoresWaitNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, const cudaExternalSemaphoreWaitNodeParams* nodeParams) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaGraphAddExternalSemaphoresWaitNode(pGraphNode, graph, pDependencies, numDependencies, nodeParams)

cdef cudaError_t _cudaGraphExternalSemaphoresWaitNodeGetParams(cudaGraphNode_t hNode, cudaExternalSemaphoreWaitNodeParams* params_out) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaGraphExternalSemaphoresWaitNodeGetParams(hNode, params_out)

cdef cudaError_t _cudaGraphExternalSemaphoresWaitNodeSetParams(cudaGraphNode_t hNode, const cudaExternalSemaphoreWaitNodeParams* nodeParams) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaGraphExternalSemaphoresWaitNodeSetParams(hNode, nodeParams)

cdef cudaError_t _cudaGraphAddMemAllocNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, cudaMemAllocNodeParams* nodeParams) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaGraphAddMemAllocNode(pGraphNode, graph, pDependencies, numDependencies, nodeParams)

cdef cudaError_t _cudaGraphMemAllocNodeGetParams(cudaGraphNode_t node, cudaMemAllocNodeParams* params_out) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaGraphMemAllocNodeGetParams(node, params_out)

cdef cudaError_t _cudaGraphAddMemFreeNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, void* dptr) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaGraphAddMemFreeNode(pGraphNode, graph, pDependencies, numDependencies, dptr)

cdef cudaError_t _cudaGraphMemFreeNodeGetParams(cudaGraphNode_t node, void* dptr_out) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaGraphMemFreeNodeGetParams(node, dptr_out)

cdef cudaError_t _cudaDeviceGraphMemTrim(int device) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaDeviceGraphMemTrim(device)

cdef cudaError_t _cudaDeviceGetGraphMemAttribute(int device, cudaGraphMemAttributeType attr, void* value) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaDeviceGetGraphMemAttribute(device, attr, value)

cdef cudaError_t _cudaDeviceSetGraphMemAttribute(int device, cudaGraphMemAttributeType attr, void* value) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaDeviceSetGraphMemAttribute(device, attr, value)

cdef cudaError_t _cudaGraphClone(cudaGraph_t* pGraphClone, cudaGraph_t originalGraph) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaGraphClone(pGraphClone, originalGraph)

cdef cudaError_t _cudaGraphNodeFindInClone(cudaGraphNode_t* pNode, cudaGraphNode_t originalNode, cudaGraph_t clonedGraph) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaGraphNodeFindInClone(pNode, originalNode, clonedGraph)

cdef cudaError_t _cudaGraphNodeGetType(cudaGraphNode_t node, cudaGraphNodeType* pType) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaGraphNodeGetType(node, pType)

cdef cudaError_t _cudaGraphNodeGetContainingGraph(cudaGraphNode_t hNode, cudaGraph_t* phGraph) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaGraphNodeGetContainingGraph(hNode, phGraph)

cdef cudaError_t _cudaGraphNodeGetLocalId(cudaGraphNode_t hNode, unsigned int* nodeId) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaGraphNodeGetLocalId(hNode, nodeId)

cdef cudaError_t _cudaGraphNodeGetToolsId(cudaGraphNode_t hNode, unsigned long long* toolsNodeId) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaGraphNodeGetToolsId(hNode, toolsNodeId)

cdef cudaError_t _cudaGraphGetId(cudaGraph_t hGraph, unsigned int* graphID) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaGraphGetId(hGraph, graphID)

cdef cudaError_t _cudaGraphExecGetId(cudaGraphExec_t hGraphExec, unsigned int* graphID) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaGraphExecGetId(hGraphExec, graphID)

cdef cudaError_t _cudaGraphGetNodes(cudaGraph_t graph, cudaGraphNode_t* nodes, size_t* numNodes) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaGraphGetNodes(graph, nodes, numNodes)

cdef cudaError_t _cudaGraphGetRootNodes(cudaGraph_t graph, cudaGraphNode_t* pRootNodes, size_t* pNumRootNodes) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaGraphGetRootNodes(graph, pRootNodes, pNumRootNodes)

cdef cudaError_t _cudaGraphGetEdges(cudaGraph_t graph, cudaGraphNode_t* from_, cudaGraphNode_t* to, cudaGraphEdgeData* edgeData, size_t* numEdges) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaGraphGetEdges(graph, from_, to, edgeData, numEdges)

cdef cudaError_t _cudaGraphNodeGetDependencies(cudaGraphNode_t node, cudaGraphNode_t* pDependencies, cudaGraphEdgeData* edgeData, size_t* pNumDependencies) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaGraphNodeGetDependencies(node, pDependencies, edgeData, pNumDependencies)

cdef cudaError_t _cudaGraphNodeGetDependentNodes(cudaGraphNode_t node, cudaGraphNode_t* pDependentNodes, cudaGraphEdgeData* edgeData, size_t* pNumDependentNodes) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaGraphNodeGetDependentNodes(node, pDependentNodes, edgeData, pNumDependentNodes)

cdef cudaError_t _cudaGraphAddDependencies(cudaGraph_t graph, const cudaGraphNode_t* from_, const cudaGraphNode_t* to, const cudaGraphEdgeData* edgeData, size_t numDependencies) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaGraphAddDependencies(graph, from_, to, edgeData, numDependencies)

cdef cudaError_t _cudaGraphRemoveDependencies(cudaGraph_t graph, const cudaGraphNode_t* from_, const cudaGraphNode_t* to, const cudaGraphEdgeData* edgeData, size_t numDependencies) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaGraphRemoveDependencies(graph, from_, to, edgeData, numDependencies)

cdef cudaError_t _cudaGraphDestroyNode(cudaGraphNode_t node) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaGraphDestroyNode(node)

cdef cudaError_t _cudaGraphInstantiate(cudaGraphExec_t* pGraphExec, cudaGraph_t graph, unsigned long long flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaGraphInstantiate(pGraphExec, graph, flags)

cdef cudaError_t _cudaGraphInstantiateWithFlags(cudaGraphExec_t* pGraphExec, cudaGraph_t graph, unsigned long long flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaGraphInstantiateWithFlags(pGraphExec, graph, flags)

cdef cudaError_t _cudaGraphInstantiateWithParams(cudaGraphExec_t* pGraphExec, cudaGraph_t graph, cudaGraphInstantiateParams* instantiateParams) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaGraphInstantiateWithParams(pGraphExec, graph, instantiateParams)

cdef cudaError_t _cudaGraphExecGetFlags(cudaGraphExec_t graphExec, unsigned long long* flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaGraphExecGetFlags(graphExec, flags)

cdef cudaError_t _cudaGraphExecKernelNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, const cudaKernelNodeParams* pNodeParams) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaGraphExecKernelNodeSetParams(hGraphExec, node, pNodeParams)

cdef cudaError_t _cudaGraphExecMemcpyNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, const cudaMemcpy3DParms* pNodeParams) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaGraphExecMemcpyNodeSetParams(hGraphExec, node, pNodeParams)

cdef cudaError_t _cudaGraphExecMemcpyNodeSetParams1D(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, void* dst, const void* src, size_t count, cudaMemcpyKind kind) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaGraphExecMemcpyNodeSetParams1D(hGraphExec, node, dst, src, count, kind)

cdef cudaError_t _cudaGraphExecMemsetNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, const cudaMemsetParams* pNodeParams) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaGraphExecMemsetNodeSetParams(hGraphExec, node, pNodeParams)

cdef cudaError_t _cudaGraphExecHostNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, const cudaHostNodeParams* pNodeParams) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaGraphExecHostNodeSetParams(hGraphExec, node, pNodeParams)

cdef cudaError_t _cudaGraphExecChildGraphNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, cudaGraph_t childGraph) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaGraphExecChildGraphNodeSetParams(hGraphExec, node, childGraph)

cdef cudaError_t _cudaGraphExecEventRecordNodeSetEvent(cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, cudaEvent_t event) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaGraphExecEventRecordNodeSetEvent(hGraphExec, hNode, event)

cdef cudaError_t _cudaGraphExecEventWaitNodeSetEvent(cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, cudaEvent_t event) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaGraphExecEventWaitNodeSetEvent(hGraphExec, hNode, event)

cdef cudaError_t _cudaGraphExecExternalSemaphoresSignalNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, const cudaExternalSemaphoreSignalNodeParams* nodeParams) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaGraphExecExternalSemaphoresSignalNodeSetParams(hGraphExec, hNode, nodeParams)

cdef cudaError_t _cudaGraphExecExternalSemaphoresWaitNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, const cudaExternalSemaphoreWaitNodeParams* nodeParams) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaGraphExecExternalSemaphoresWaitNodeSetParams(hGraphExec, hNode, nodeParams)

cdef cudaError_t _cudaGraphNodeSetEnabled(cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, unsigned int isEnabled) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaGraphNodeSetEnabled(hGraphExec, hNode, isEnabled)

cdef cudaError_t _cudaGraphNodeGetEnabled(cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, unsigned int* isEnabled) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaGraphNodeGetEnabled(hGraphExec, hNode, isEnabled)

cdef cudaError_t _cudaGraphExecUpdate(cudaGraphExec_t hGraphExec, cudaGraph_t hGraph, cudaGraphExecUpdateResultInfo* resultInfo) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaGraphExecUpdate(hGraphExec, hGraph, resultInfo)

cdef cudaError_t _cudaGraphUpload(cudaGraphExec_t graphExec, cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaGraphUpload(graphExec, stream)

cdef cudaError_t _cudaGraphLaunch(cudaGraphExec_t graphExec, cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaGraphLaunch(graphExec, stream)

cdef cudaError_t _cudaGraphExecDestroy(cudaGraphExec_t graphExec) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaGraphExecDestroy(graphExec)

cdef cudaError_t _cudaGraphDestroy(cudaGraph_t graph) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaGraphDestroy(graph)

cdef cudaError_t _cudaGraphDebugDotPrint(cudaGraph_t graph, const char* path, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaGraphDebugDotPrint(graph, path, flags)

cdef cudaError_t _cudaUserObjectCreate(cudaUserObject_t* object_out, void* ptr, cudaHostFn_t destroy, unsigned int initialRefcount, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaUserObjectCreate(object_out, ptr, destroy, initialRefcount, flags)

cdef cudaError_t _cudaUserObjectRetain(cudaUserObject_t object, unsigned int count) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaUserObjectRetain(object, count)

cdef cudaError_t _cudaUserObjectRelease(cudaUserObject_t object, unsigned int count) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaUserObjectRelease(object, count)

cdef cudaError_t _cudaGraphRetainUserObject(cudaGraph_t graph, cudaUserObject_t object, unsigned int count, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaGraphRetainUserObject(graph, object, count, flags)

cdef cudaError_t _cudaGraphReleaseUserObject(cudaGraph_t graph, cudaUserObject_t object, unsigned int count) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaGraphReleaseUserObject(graph, object, count)

cdef cudaError_t _cudaGraphAddNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, const cudaGraphEdgeData* dependencyData, size_t numDependencies, cudaGraphNodeParams* nodeParams) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaGraphAddNode(pGraphNode, graph, pDependencies, dependencyData, numDependencies, nodeParams)

cdef cudaError_t _cudaGraphNodeSetParams(cudaGraphNode_t node, cudaGraphNodeParams* nodeParams) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaGraphNodeSetParams(node, nodeParams)

cdef cudaError_t _cudaGraphNodeGetParams(cudaGraphNode_t node, cudaGraphNodeParams* nodeParams) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaGraphNodeGetParams(node, nodeParams)

cdef cudaError_t _cudaGraphExecNodeSetParams(cudaGraphExec_t graphExec, cudaGraphNode_t node, cudaGraphNodeParams* nodeParams) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaGraphExecNodeSetParams(graphExec, node, nodeParams)

cdef cudaError_t _cudaGraphConditionalHandleCreate(cudaGraphConditionalHandle* pHandle_out, cudaGraph_t graph, unsigned int defaultLaunchValue, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaGraphConditionalHandleCreate(pHandle_out, graph, defaultLaunchValue, flags)

cdef cudaError_t _cudaGraphConditionalHandleCreate_v2(cudaGraphConditionalHandle* pHandle_out, cudaGraph_t graph, cudaExecutionContext_t ctx, unsigned int defaultLaunchValue, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaGraphConditionalHandleCreate_v2(pHandle_out, graph, ctx, defaultLaunchValue, flags)

cdef cudaError_t _cudaGetDriverEntryPoint(const char* symbol, void** funcPtr, unsigned long long flags, cudaDriverEntryPointQueryResult* driverStatus) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaGetDriverEntryPoint(symbol, funcPtr, flags, driverStatus)

cdef cudaError_t _cudaGetDriverEntryPointByVersion(const char* symbol, void** funcPtr, unsigned int cudaVersion, unsigned long long flags, cudaDriverEntryPointQueryResult* driverStatus) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaGetDriverEntryPointByVersion(symbol, funcPtr, cudaVersion, flags, driverStatus)

cdef cudaError_t _cudaLibraryLoadData(cudaLibrary_t* library, const void* code, cudaJitOption* jitOptions, void** jitOptionsValues, unsigned int numJitOptions, cudaLibraryOption* libraryOptions, void** libraryOptionValues, unsigned int numLibraryOptions) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaLibraryLoadData(library, code, jitOptions, jitOptionsValues, numJitOptions, libraryOptions, libraryOptionValues, numLibraryOptions)

cdef cudaError_t _cudaLibraryLoadFromFile(cudaLibrary_t* library, const char* fileName, cudaJitOption* jitOptions, void** jitOptionsValues, unsigned int numJitOptions, cudaLibraryOption* libraryOptions, void** libraryOptionValues, unsigned int numLibraryOptions) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaLibraryLoadFromFile(library, fileName, jitOptions, jitOptionsValues, numJitOptions, libraryOptions, libraryOptionValues, numLibraryOptions)

cdef cudaError_t _cudaLibraryUnload(cudaLibrary_t library) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaLibraryUnload(library)

cdef cudaError_t _cudaLibraryGetKernel(cudaKernel_t* pKernel, cudaLibrary_t library, const char* name) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaLibraryGetKernel(pKernel, library, name)

cdef cudaError_t _cudaLibraryGetGlobal(void** dptr, size_t* numbytes, cudaLibrary_t library, const char* name) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaLibraryGetGlobal(dptr, numbytes, library, name)

cdef cudaError_t _cudaLibraryGetManaged(void** dptr, size_t* numbytes, cudaLibrary_t library, const char* name) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaLibraryGetManaged(dptr, numbytes, library, name)

cdef cudaError_t _cudaLibraryGetUnifiedFunction(void** fptr, cudaLibrary_t library, const char* symbol) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaLibraryGetUnifiedFunction(fptr, library, symbol)

cdef cudaError_t _cudaLibraryGetKernelCount(unsigned int* count, cudaLibrary_t lib) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaLibraryGetKernelCount(count, lib)

cdef cudaError_t _cudaLibraryEnumerateKernels(cudaKernel_t* kernels, unsigned int numKernels, cudaLibrary_t lib) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaLibraryEnumerateKernels(kernels, numKernels, lib)

cdef cudaError_t _cudaKernelSetAttributeForDevice(cudaKernel_t kernel, cudaFuncAttribute attr, int value, int device) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaKernelSetAttributeForDevice(kernel, attr, value, device)

cdef cudaError_t _cudaDeviceGetDevResource(int device, cudaDevResource* resource, cudaDevResourceType typename) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaDeviceGetDevResource(device, resource, typename)

cdef cudaError_t _cudaDevSmResourceSplitByCount(cudaDevResource* result, unsigned int* nbGroups, const cudaDevResource* input, cudaDevResource* remaining, unsigned int flags, unsigned int minCount) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaDevSmResourceSplitByCount(result, nbGroups, input, remaining, flags, minCount)

cdef cudaError_t _cudaDevSmResourceSplit(cudaDevResource* result, unsigned int nbGroups, const cudaDevResource* input, cudaDevResource* remainder, unsigned int flags, cudaDevSmResourceGroupParams* groupParams) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaDevSmResourceSplit(result, nbGroups, input, remainder, flags, groupParams)

cdef cudaError_t _cudaDevResourceGenerateDesc(cudaDevResourceDesc_t* phDesc, cudaDevResource* resources, unsigned int nbResources) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaDevResourceGenerateDesc(phDesc, resources, nbResources)

cdef cudaError_t _cudaGreenCtxCreate(cudaExecutionContext_t* phCtx, cudaDevResourceDesc_t desc, int device, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaGreenCtxCreate(phCtx, desc, device, flags)

cdef cudaError_t _cudaExecutionCtxDestroy(cudaExecutionContext_t ctx) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaExecutionCtxDestroy(ctx)

cdef cudaError_t _cudaExecutionCtxGetDevResource(cudaExecutionContext_t ctx, cudaDevResource* resource, cudaDevResourceType typename) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaExecutionCtxGetDevResource(ctx, resource, typename)

cdef cudaError_t _cudaExecutionCtxGetDevice(int* device, cudaExecutionContext_t ctx) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaExecutionCtxGetDevice(device, ctx)

cdef cudaError_t _cudaExecutionCtxGetId(cudaExecutionContext_t ctx, unsigned long long* ctxId) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaExecutionCtxGetId(ctx, ctxId)

cdef cudaError_t _cudaExecutionCtxStreamCreate(cudaStream_t* phStream, cudaExecutionContext_t ctx, unsigned int flags, int priority) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaExecutionCtxStreamCreate(phStream, ctx, flags, priority)

cdef cudaError_t _cudaExecutionCtxSynchronize(cudaExecutionContext_t ctx) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaExecutionCtxSynchronize(ctx)

cdef cudaError_t _cudaStreamGetDevResource(cudaStream_t hStream, cudaDevResource* resource, cudaDevResourceType typename) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaStreamGetDevResource(hStream, resource, typename)

cdef cudaError_t _cudaExecutionCtxRecordEvent(cudaExecutionContext_t ctx, cudaEvent_t event) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaExecutionCtxRecordEvent(ctx, event)

cdef cudaError_t _cudaExecutionCtxWaitEvent(cudaExecutionContext_t ctx, cudaEvent_t event) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaExecutionCtxWaitEvent(ctx, event)

cdef cudaError_t _cudaDeviceGetExecutionCtx(cudaExecutionContext_t* ctx, int device) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaDeviceGetExecutionCtx(ctx, device)

cdef cudaError_t _cudaGetExportTable(const void** ppExportTable, const cudaUUID_t* pExportTableId) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaGetExportTable(ppExportTable, pExportTableId)

cdef cudaError_t _cudaGetKernel(cudaKernel_t* kernelPtr, const void* entryFuncAddr) except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaGetKernel(kernelPtr, entryFuncAddr)
@cython.show_performance_hints(False)
cdef cudaPitchedPtr _make_cudaPitchedPtr(void* d, size_t p, size_t xsz, size_t ysz) except* nogil:
    return make_cudaPitchedPtr(d, p, xsz, ysz)
@cython.show_performance_hints(False)
cdef cudaPos _make_cudaPos(size_t x, size_t y, size_t z) except* nogil:
    return make_cudaPos(x, y, z)
@cython.show_performance_hints(False)
cdef cudaExtent _make_cudaExtent(size_t w, size_t h, size_t d) except* nogil:
    return make_cudaExtent(w, h, d)

cdef cudaError_t _cudaProfilerStart() except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaProfilerStart()

cdef cudaError_t _cudaProfilerStop() except ?cudaErrorCallRequiresNewerDriver nogil:
    return cudaProfilerStop()
