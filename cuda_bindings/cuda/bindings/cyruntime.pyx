# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
#
# This code was automatically generated across versions from 12.9.0 to 13.3.0. Do not modify it directly.

# CYTHON-BINDINGS-GENERATED-DO-NOT-MODIFY-THIS-FILE: format=1; content-sha256=034f6c5c936d547d3106aa249f8f85b67389eac8b90ab569aa74815f08d699af
from ._internal cimport runtime as _runtime

cdef cudaError_t cudaDeviceReset() except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaDeviceReset()


cdef cudaError_t cudaDeviceSynchronize() except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaDeviceSynchronize()


cdef cudaError_t cudaDeviceSetLimit(cudaLimit limit, size_t value) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaDeviceSetLimit(limit, value)


cdef cudaError_t cudaDeviceGetLimit(size_t* pValue, cudaLimit limit) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaDeviceGetLimit(pValue, limit)


cdef cudaError_t cudaDeviceGetTexture1DLinearMaxWidth(size_t* maxWidthInElements, const cudaChannelFormatDesc* fmtDesc, int device) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaDeviceGetTexture1DLinearMaxWidth(maxWidthInElements, fmtDesc, device)


cdef cudaError_t cudaDeviceGetCacheConfig(cudaFuncCache* pCacheConfig) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaDeviceGetCacheConfig(pCacheConfig)


cdef cudaError_t cudaDeviceGetStreamPriorityRange(int* leastPriority, int* greatestPriority) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaDeviceGetStreamPriorityRange(leastPriority, greatestPriority)


cdef cudaError_t cudaDeviceSetCacheConfig(cudaFuncCache cacheConfig) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaDeviceSetCacheConfig(cacheConfig)


cdef cudaError_t cudaDeviceGetByPCIBusId(int* device, const char* pciBusId) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaDeviceGetByPCIBusId(device, pciBusId)


cdef cudaError_t cudaDeviceGetPCIBusId(char* pciBusId, int len, int device) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaDeviceGetPCIBusId(pciBusId, len, device)


cdef cudaError_t cudaIpcGetEventHandle(cudaIpcEventHandle_t* handle, cudaEvent_t event) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaIpcGetEventHandle(handle, event)


cdef cudaError_t cudaIpcOpenEventHandle(cudaEvent_t* event, cudaIpcEventHandle_t handle) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaIpcOpenEventHandle(event, handle)


cdef cudaError_t cudaIpcGetMemHandle(cudaIpcMemHandle_t* handle, void* devPtr) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaIpcGetMemHandle(handle, devPtr)


cdef cudaError_t cudaIpcOpenMemHandle(void** devPtr, cudaIpcMemHandle_t handle, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaIpcOpenMemHandle(devPtr, handle, flags)


cdef cudaError_t cudaIpcCloseMemHandle(void* devPtr) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaIpcCloseMemHandle(devPtr)


cdef cudaError_t cudaDeviceFlushGPUDirectRDMAWrites(cudaFlushGPUDirectRDMAWritesTarget target, cudaFlushGPUDirectRDMAWritesScope scope) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaDeviceFlushGPUDirectRDMAWrites(target, scope)


cdef cudaError_t cudaDeviceRegisterAsyncNotification(int device, cudaAsyncCallback callbackFunc, void* userData, cudaAsyncCallbackHandle_t* callback) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaDeviceRegisterAsyncNotification(device, callbackFunc, userData, callback)


cdef cudaError_t cudaDeviceUnregisterAsyncNotification(int device, cudaAsyncCallbackHandle_t callback) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaDeviceUnregisterAsyncNotification(device, callback)


cdef cudaError_t cudaDeviceGetSharedMemConfig(cudaSharedMemConfig* pConfig) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaDeviceGetSharedMemConfig(pConfig)


cdef cudaError_t cudaDeviceSetSharedMemConfig(cudaSharedMemConfig config) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaDeviceSetSharedMemConfig(config)


cdef cudaError_t cudaGetLastError() except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaGetLastError()


cdef cudaError_t cudaPeekAtLastError() except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaPeekAtLastError()


cdef const char* cudaGetErrorName(cudaError_t error) except?NULL nogil:
    return _runtime._cudaGetErrorName(error)


cdef const char* cudaGetErrorString(cudaError_t error) except?NULL nogil:
    return _runtime._cudaGetErrorString(error)


cdef cudaError_t cudaGetDeviceCount(int* count) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaGetDeviceCount(count)


cdef cudaError_t cudaDeviceGetAttribute(int* value, cudaDeviceAttr attr, int device) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaDeviceGetAttribute(value, attr, device)


cdef cudaError_t cudaDeviceGetDefaultMemPool(cudaMemPool_t* memPool, int device) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaDeviceGetDefaultMemPool(memPool, device)


cdef cudaError_t cudaDeviceSetMemPool(int device, cudaMemPool_t memPool) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaDeviceSetMemPool(device, memPool)


cdef cudaError_t cudaDeviceGetMemPool(cudaMemPool_t* memPool, int device) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaDeviceGetMemPool(memPool, device)


cdef cudaError_t cudaDeviceGetNvSciSyncAttributes(void* nvSciSyncAttrList, int device, int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaDeviceGetNvSciSyncAttributes(nvSciSyncAttrList, device, flags)


cdef cudaError_t cudaDeviceGetP2PAttribute(int* value, cudaDeviceP2PAttr attr, int srcDevice, int dstDevice) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaDeviceGetP2PAttribute(value, attr, srcDevice, dstDevice)


cdef cudaError_t cudaChooseDevice(int* device, const cudaDeviceProp* prop) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaChooseDevice(device, prop)


cdef cudaError_t cudaInitDevice(int device, unsigned int deviceFlags, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaInitDevice(device, deviceFlags, flags)


cdef cudaError_t cudaSetDevice(int device) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaSetDevice(device)


cdef cudaError_t cudaGetDevice(int* device) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaGetDevice(device)


cdef cudaError_t cudaSetDeviceFlags(unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaSetDeviceFlags(flags)


cdef cudaError_t cudaGetDeviceFlags(unsigned int* flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaGetDeviceFlags(flags)


cdef cudaError_t cudaStreamCreate(cudaStream_t* pStream) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaStreamCreate(pStream)


cdef cudaError_t cudaStreamCreateWithFlags(cudaStream_t* pStream, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaStreamCreateWithFlags(pStream, flags)


cdef cudaError_t cudaStreamCreateWithPriority(cudaStream_t* pStream, unsigned int flags, int priority) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaStreamCreateWithPriority(pStream, flags, priority)


cdef cudaError_t cudaStreamGetPriority(cudaStream_t hStream, int* priority) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaStreamGetPriority(hStream, priority)


cdef cudaError_t cudaStreamGetFlags(cudaStream_t hStream, unsigned int* flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaStreamGetFlags(hStream, flags)


cdef cudaError_t cudaStreamGetId(cudaStream_t hStream, unsigned long long* streamId) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaStreamGetId(hStream, streamId)


cdef cudaError_t cudaStreamGetDevice(cudaStream_t hStream, int* device) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaStreamGetDevice(hStream, device)


cdef cudaError_t cudaCtxResetPersistingL2Cache() except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaCtxResetPersistingL2Cache()


cdef cudaError_t cudaStreamCopyAttributes(cudaStream_t dst, cudaStream_t src) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaStreamCopyAttributes(dst, src)


cdef cudaError_t cudaStreamGetAttribute(cudaStream_t hStream, cudaStreamAttrID attr, cudaStreamAttrValue* value_out) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaStreamGetAttribute(hStream, attr, value_out)


cdef cudaError_t cudaStreamSetAttribute(cudaStream_t hStream, cudaStreamAttrID attr, const cudaStreamAttrValue* value) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaStreamSetAttribute(hStream, attr, value)


cdef cudaError_t cudaStreamDestroy(cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaStreamDestroy(stream)


cdef cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaStreamWaitEvent(stream, event, flags)


cdef cudaError_t cudaStreamAddCallback(cudaStream_t stream, cudaStreamCallback_t callback, void* userData, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaStreamAddCallback(stream, callback, userData, flags)


cdef cudaError_t cudaStreamSynchronize(cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaStreamSynchronize(stream)


cdef cudaError_t cudaStreamQuery(cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaStreamQuery(stream)


cdef cudaError_t cudaStreamAttachMemAsync(cudaStream_t stream, void* devPtr, size_t length, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaStreamAttachMemAsync(stream, devPtr, length, flags)


cdef cudaError_t cudaStreamBeginCapture(cudaStream_t stream, cudaStreamCaptureMode mode) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaStreamBeginCapture(stream, mode)


cdef cudaError_t cudaStreamBeginCaptureToGraph(cudaStream_t stream, cudaGraph_t graph, const cudaGraphNode_t* dependencies, const cudaGraphEdgeData* dependencyData, size_t numDependencies, cudaStreamCaptureMode mode) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaStreamBeginCaptureToGraph(stream, graph, dependencies, dependencyData, numDependencies, mode)


cdef cudaError_t cudaThreadExchangeStreamCaptureMode(cudaStreamCaptureMode* mode) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaThreadExchangeStreamCaptureMode(mode)


cdef cudaError_t cudaStreamEndCapture(cudaStream_t stream, cudaGraph_t* pGraph) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaStreamEndCapture(stream, pGraph)


cdef cudaError_t cudaStreamIsCapturing(cudaStream_t stream, cudaStreamCaptureStatus* pCaptureStatus) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaStreamIsCapturing(stream, pCaptureStatus)


cdef cudaError_t cudaStreamUpdateCaptureDependencies(cudaStream_t stream, cudaGraphNode_t* dependencies, const cudaGraphEdgeData* dependencyData, size_t numDependencies, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaStreamUpdateCaptureDependencies(stream, dependencies, dependencyData, numDependencies, flags)


cdef cudaError_t cudaEventCreate(cudaEvent_t* event) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaEventCreate(event)


cdef cudaError_t cudaEventCreateWithFlags(cudaEvent_t* event, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaEventCreateWithFlags(event, flags)


cdef cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaEventRecord(event, stream)


cdef cudaError_t cudaEventRecordWithFlags(cudaEvent_t event, cudaStream_t stream, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaEventRecordWithFlags(event, stream, flags)


cdef cudaError_t cudaEventQuery(cudaEvent_t event) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaEventQuery(event)


cdef cudaError_t cudaEventSynchronize(cudaEvent_t event) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaEventSynchronize(event)


cdef cudaError_t cudaEventDestroy(cudaEvent_t event) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaEventDestroy(event)


cdef cudaError_t cudaEventElapsedTime(float* ms, cudaEvent_t start, cudaEvent_t end) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaEventElapsedTime(ms, start, end)


cdef cudaError_t cudaImportExternalMemory(cudaExternalMemory_t* extMem_out, const cudaExternalMemoryHandleDesc* memHandleDesc) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaImportExternalMemory(extMem_out, memHandleDesc)


cdef cudaError_t cudaExternalMemoryGetMappedBuffer(void** devPtr, cudaExternalMemory_t extMem, const cudaExternalMemoryBufferDesc* bufferDesc) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaExternalMemoryGetMappedBuffer(devPtr, extMem, bufferDesc)


cdef cudaError_t cudaExternalMemoryGetMappedMipmappedArray(cudaMipmappedArray_t* mipmap, cudaExternalMemory_t extMem, const cudaExternalMemoryMipmappedArrayDesc* mipmapDesc) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaExternalMemoryGetMappedMipmappedArray(mipmap, extMem, mipmapDesc)


cdef cudaError_t cudaDestroyExternalMemory(cudaExternalMemory_t extMem) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaDestroyExternalMemory(extMem)


cdef cudaError_t cudaImportExternalSemaphore(cudaExternalSemaphore_t* extSem_out, const cudaExternalSemaphoreHandleDesc* semHandleDesc) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaImportExternalSemaphore(extSem_out, semHandleDesc)


cdef cudaError_t cudaDestroyExternalSemaphore(cudaExternalSemaphore_t extSem) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaDestroyExternalSemaphore(extSem)


cdef cudaError_t cudaFuncSetCacheConfig(const void* func, cudaFuncCache cacheConfig) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaFuncSetCacheConfig(func, cacheConfig)


cdef cudaError_t cudaFuncGetAttributes(cudaFuncAttributes* attr, const void* func) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaFuncGetAttributes(attr, func)


cdef cudaError_t cudaFuncSetAttribute(const void* func, cudaFuncAttribute attr, int value) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaFuncSetAttribute(func, attr, value)


cdef cudaError_t cudaFuncGetName(const char** name, const void* func) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaFuncGetName(name, func)


cdef cudaError_t cudaFuncGetParamInfo(const void* func, size_t paramIndex, size_t* paramOffset, size_t* paramSize) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaFuncGetParamInfo(func, paramIndex, paramOffset, paramSize)


cdef cudaError_t cudaLaunchHostFunc(cudaStream_t stream, cudaHostFn_t fn, void* userData) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaLaunchHostFunc(stream, fn, userData)


cdef cudaError_t cudaFuncSetSharedMemConfig(const void* func, cudaSharedMemConfig config) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaFuncSetSharedMemConfig(func, config)


cdef cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(int* numBlocks, const void* func, int blockSize, size_t dynamicSMemSize) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaOccupancyMaxActiveBlocksPerMultiprocessor(numBlocks, func, blockSize, dynamicSMemSize)


cdef cudaError_t cudaOccupancyAvailableDynamicSMemPerBlock(size_t* dynamicSmemSize, const void* func, int numBlocks, int blockSize) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaOccupancyAvailableDynamicSMemPerBlock(dynamicSmemSize, func, numBlocks, blockSize)


cdef cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int* numBlocks, const void* func, int blockSize, size_t dynamicSMemSize, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(numBlocks, func, blockSize, dynamicSMemSize, flags)


cdef cudaError_t cudaMallocManaged(void** devPtr, size_t size, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaMallocManaged(devPtr, size, flags)


cdef cudaError_t cudaMalloc(void** devPtr, size_t size) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaMalloc(devPtr, size)


cdef cudaError_t cudaMallocHost(void** ptr, size_t size) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaMallocHost(ptr, size)


cdef cudaError_t cudaMallocPitch(void** devPtr, size_t* pitch, size_t width, size_t height) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaMallocPitch(devPtr, pitch, width, height)


cdef cudaError_t cudaMallocArray(cudaArray_t* array, const cudaChannelFormatDesc* desc, size_t width, size_t height, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaMallocArray(array, desc, width, height, flags)


cdef cudaError_t cudaFree(void* devPtr) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaFree(devPtr)


cdef cudaError_t cudaFreeHost(void* ptr) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaFreeHost(ptr)


cdef cudaError_t cudaFreeArray(cudaArray_t array) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaFreeArray(array)


cdef cudaError_t cudaFreeMipmappedArray(cudaMipmappedArray_t mipmappedArray) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaFreeMipmappedArray(mipmappedArray)


cdef cudaError_t cudaHostAlloc(void** pHost, size_t size, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaHostAlloc(pHost, size, flags)


cdef cudaError_t cudaHostRegister(void* ptr, size_t size, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaHostRegister(ptr, size, flags)


cdef cudaError_t cudaHostUnregister(void* ptr) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaHostUnregister(ptr)


cdef cudaError_t cudaHostGetDevicePointer(void** pDevice, void* pHost, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaHostGetDevicePointer(pDevice, pHost, flags)


cdef cudaError_t cudaHostGetFlags(unsigned int* pFlags, void* pHost) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaHostGetFlags(pFlags, pHost)


cdef cudaError_t cudaMalloc3D(cudaPitchedPtr* pitchedDevPtr, cudaExtent extent) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaMalloc3D(pitchedDevPtr, extent)


cdef cudaError_t cudaMalloc3DArray(cudaArray_t* array, const cudaChannelFormatDesc* desc, cudaExtent extent, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaMalloc3DArray(array, desc, extent, flags)


cdef cudaError_t cudaMallocMipmappedArray(cudaMipmappedArray_t* mipmappedArray, const cudaChannelFormatDesc* desc, cudaExtent extent, unsigned int numLevels, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaMallocMipmappedArray(mipmappedArray, desc, extent, numLevels, flags)


cdef cudaError_t cudaGetMipmappedArrayLevel(cudaArray_t* levelArray, cudaMipmappedArray_const_t mipmappedArray, unsigned int level) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaGetMipmappedArrayLevel(levelArray, mipmappedArray, level)


cdef cudaError_t cudaMemcpy3D(const cudaMemcpy3DParms* p) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaMemcpy3D(p)


cdef cudaError_t cudaMemcpy3DPeer(const cudaMemcpy3DPeerParms* p) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaMemcpy3DPeer(p)


cdef cudaError_t cudaMemcpy3DAsync(const cudaMemcpy3DParms* p, cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaMemcpy3DAsync(p, stream)


cdef cudaError_t cudaMemcpy3DPeerAsync(const cudaMemcpy3DPeerParms* p, cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaMemcpy3DPeerAsync(p, stream)


cdef cudaError_t cudaMemGetInfo(size_t* free, size_t* total) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaMemGetInfo(free, total)


cdef cudaError_t cudaArrayGetInfo(cudaChannelFormatDesc* desc, cudaExtent* extent, unsigned int* flags, cudaArray_t array) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaArrayGetInfo(desc, extent, flags, array)


cdef cudaError_t cudaArrayGetPlane(cudaArray_t* pPlaneArray, cudaArray_t hArray, unsigned int planeIdx) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaArrayGetPlane(pPlaneArray, hArray, planeIdx)


cdef cudaError_t cudaArrayGetMemoryRequirements(cudaArrayMemoryRequirements* memoryRequirements, cudaArray_t array, int device) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaArrayGetMemoryRequirements(memoryRequirements, array, device)


cdef cudaError_t cudaMipmappedArrayGetMemoryRequirements(cudaArrayMemoryRequirements* memoryRequirements, cudaMipmappedArray_t mipmap, int device) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaMipmappedArrayGetMemoryRequirements(memoryRequirements, mipmap, device)


cdef cudaError_t cudaArrayGetSparseProperties(cudaArraySparseProperties* sparseProperties, cudaArray_t array) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaArrayGetSparseProperties(sparseProperties, array)


cdef cudaError_t cudaMipmappedArrayGetSparseProperties(cudaArraySparseProperties* sparseProperties, cudaMipmappedArray_t mipmap) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaMipmappedArrayGetSparseProperties(sparseProperties, mipmap)


cdef cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaMemcpy(dst, src, count, kind)


cdef cudaError_t cudaMemcpyPeer(void* dst, int dstDevice, const void* src, int srcDevice, size_t count) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaMemcpyPeer(dst, dstDevice, src, srcDevice, count)


cdef cudaError_t cudaMemcpy2D(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaMemcpy2D(dst, dpitch, src, spitch, width, height, kind)


cdef cudaError_t cudaMemcpy2DToArray(cudaArray_t dst, size_t wOffset, size_t hOffset, const void* src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaMemcpy2DToArray(dst, wOffset, hOffset, src, spitch, width, height, kind)


cdef cudaError_t cudaMemcpy2DFromArray(void* dst, size_t dpitch, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t width, size_t height, cudaMemcpyKind kind) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaMemcpy2DFromArray(dst, dpitch, src, wOffset, hOffset, width, height, kind)


cdef cudaError_t cudaMemcpy2DArrayToArray(cudaArray_t dst, size_t wOffsetDst, size_t hOffsetDst, cudaArray_const_t src, size_t wOffsetSrc, size_t hOffsetSrc, size_t width, size_t height, cudaMemcpyKind kind) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaMemcpy2DArrayToArray(dst, wOffsetDst, hOffsetDst, src, wOffsetSrc, hOffsetSrc, width, height, kind)


cdef cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t count, cudaMemcpyKind kind, cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaMemcpyAsync(dst, src, count, kind, stream)


cdef cudaError_t cudaMemcpyPeerAsync(void* dst, int dstDevice, const void* src, int srcDevice, size_t count, cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaMemcpyPeerAsync(dst, dstDevice, src, srcDevice, count, stream)


cdef cudaError_t cudaMemcpyBatchAsync(const void** dsts, const void** srcs, const size_t* sizes, size_t count, cudaMemcpyAttributes* attrs, size_t* attrsIdxs, size_t numAttrs, cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaMemcpyBatchAsync(dsts, srcs, sizes, count, attrs, attrsIdxs, numAttrs, stream)


cdef cudaError_t cudaMemcpy3DBatchAsync(size_t numOps, cudaMemcpy3DBatchOp* opList, unsigned long long flags, cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaMemcpy3DBatchAsync(numOps, opList, flags, stream)


cdef cudaError_t cudaMemcpy2DAsync(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaMemcpy2DAsync(dst, dpitch, src, spitch, width, height, kind, stream)


cdef cudaError_t cudaMemcpy2DToArrayAsync(cudaArray_t dst, size_t wOffset, size_t hOffset, const void* src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaMemcpy2DToArrayAsync(dst, wOffset, hOffset, src, spitch, width, height, kind, stream)


cdef cudaError_t cudaMemcpy2DFromArrayAsync(void* dst, size_t dpitch, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaMemcpy2DFromArrayAsync(dst, dpitch, src, wOffset, hOffset, width, height, kind, stream)


cdef cudaError_t cudaMemset(void* devPtr, int value, size_t count) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaMemset(devPtr, value, count)


cdef cudaError_t cudaMemset2D(void* devPtr, size_t pitch, int value, size_t width, size_t height) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaMemset2D(devPtr, pitch, value, width, height)


cdef cudaError_t cudaMemset3D(cudaPitchedPtr pitchedDevPtr, int value, cudaExtent extent) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaMemset3D(pitchedDevPtr, value, extent)


cdef cudaError_t cudaMemsetAsync(void* devPtr, int value, size_t count, cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaMemsetAsync(devPtr, value, count, stream)


cdef cudaError_t cudaMemset2DAsync(void* devPtr, size_t pitch, int value, size_t width, size_t height, cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaMemset2DAsync(devPtr, pitch, value, width, height, stream)


cdef cudaError_t cudaMemset3DAsync(cudaPitchedPtr pitchedDevPtr, int value, cudaExtent extent, cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaMemset3DAsync(pitchedDevPtr, value, extent, stream)


cdef cudaError_t cudaMemPrefetchAsync(const void* devPtr, size_t count, cudaMemLocation location, unsigned int flags, cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaMemPrefetchAsync(devPtr, count, location, flags, stream)


cdef cudaError_t cudaMemAdvise(const void* devPtr, size_t count, cudaMemoryAdvise advice, cudaMemLocation location) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaMemAdvise(devPtr, count, advice, location)


cdef cudaError_t cudaMemRangeGetAttribute(void* data, size_t dataSize, cudaMemRangeAttribute attribute, const void* devPtr, size_t count) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaMemRangeGetAttribute(data, dataSize, attribute, devPtr, count)


cdef cudaError_t cudaMemRangeGetAttributes(void** data, size_t* dataSizes, cudaMemRangeAttribute* attributes, size_t numAttributes, const void* devPtr, size_t count) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaMemRangeGetAttributes(data, dataSizes, attributes, numAttributes, devPtr, count)


cdef cudaError_t cudaMemcpyToArray(cudaArray_t dst, size_t wOffset, size_t hOffset, const void* src, size_t count, cudaMemcpyKind kind) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaMemcpyToArray(dst, wOffset, hOffset, src, count, kind)


cdef cudaError_t cudaMemcpyFromArray(void* dst, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t count, cudaMemcpyKind kind) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaMemcpyFromArray(dst, src, wOffset, hOffset, count, kind)


cdef cudaError_t cudaMemcpyArrayToArray(cudaArray_t dst, size_t wOffsetDst, size_t hOffsetDst, cudaArray_const_t src, size_t wOffsetSrc, size_t hOffsetSrc, size_t count, cudaMemcpyKind kind) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaMemcpyArrayToArray(dst, wOffsetDst, hOffsetDst, src, wOffsetSrc, hOffsetSrc, count, kind)


cdef cudaError_t cudaMemcpyToArrayAsync(cudaArray_t dst, size_t wOffset, size_t hOffset, const void* src, size_t count, cudaMemcpyKind kind, cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaMemcpyToArrayAsync(dst, wOffset, hOffset, src, count, kind, stream)


cdef cudaError_t cudaMemcpyFromArrayAsync(void* dst, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t count, cudaMemcpyKind kind, cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaMemcpyFromArrayAsync(dst, src, wOffset, hOffset, count, kind, stream)


cdef cudaError_t cudaMallocAsync(void** devPtr, size_t size, cudaStream_t hStream) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaMallocAsync(devPtr, size, hStream)


cdef cudaError_t cudaFreeAsync(void* devPtr, cudaStream_t hStream) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaFreeAsync(devPtr, hStream)


cdef cudaError_t cudaMemPoolTrimTo(cudaMemPool_t memPool, size_t minBytesToKeep) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaMemPoolTrimTo(memPool, minBytesToKeep)


cdef cudaError_t cudaMemPoolSetAttribute(cudaMemPool_t memPool, cudaMemPoolAttr attr, void* value) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaMemPoolSetAttribute(memPool, attr, value)


cdef cudaError_t cudaMemPoolGetAttribute(cudaMemPool_t memPool, cudaMemPoolAttr attr, void* value) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaMemPoolGetAttribute(memPool, attr, value)


cdef cudaError_t cudaMemPoolSetAccess(cudaMemPool_t memPool, const cudaMemAccessDesc* descList, size_t count) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaMemPoolSetAccess(memPool, descList, count)


cdef cudaError_t cudaMemPoolGetAccess(cudaMemAccessFlags* flags, cudaMemPool_t memPool, cudaMemLocation* location) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaMemPoolGetAccess(flags, memPool, location)


cdef cudaError_t cudaMemPoolCreate(cudaMemPool_t* memPool, const cudaMemPoolProps* poolProps) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaMemPoolCreate(memPool, poolProps)


cdef cudaError_t cudaMemPoolDestroy(cudaMemPool_t memPool) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaMemPoolDestroy(memPool)


cdef cudaError_t cudaMallocFromPoolAsync(void** ptr, size_t size, cudaMemPool_t memPool, cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaMallocFromPoolAsync(ptr, size, memPool, stream)


cdef cudaError_t cudaMemPoolExportToShareableHandle(void* shareableHandle, cudaMemPool_t memPool, cudaMemAllocationHandleType handleType, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaMemPoolExportToShareableHandle(shareableHandle, memPool, handleType, flags)


cdef cudaError_t cudaMemPoolImportFromShareableHandle(cudaMemPool_t* memPool, void* shareableHandle, cudaMemAllocationHandleType handleType, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaMemPoolImportFromShareableHandle(memPool, shareableHandle, handleType, flags)


cdef cudaError_t cudaMemPoolExportPointer(cudaMemPoolPtrExportData* exportData, void* ptr) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaMemPoolExportPointer(exportData, ptr)


cdef cudaError_t cudaMemPoolImportPointer(void** ptr, cudaMemPool_t memPool, cudaMemPoolPtrExportData* exportData) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaMemPoolImportPointer(ptr, memPool, exportData)


cdef cudaError_t cudaPointerGetAttributes(cudaPointerAttributes* attributes, const void* ptr) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaPointerGetAttributes(attributes, ptr)


cdef cudaError_t cudaDeviceCanAccessPeer(int* canAccessPeer, int device, int peerDevice) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaDeviceCanAccessPeer(canAccessPeer, device, peerDevice)


cdef cudaError_t cudaDeviceEnablePeerAccess(int peerDevice, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaDeviceEnablePeerAccess(peerDevice, flags)


cdef cudaError_t cudaDeviceDisablePeerAccess(int peerDevice) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaDeviceDisablePeerAccess(peerDevice)


cdef cudaError_t cudaGraphicsUnregisterResource(cudaGraphicsResource_t resource) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaGraphicsUnregisterResource(resource)


cdef cudaError_t cudaGraphicsResourceSetMapFlags(cudaGraphicsResource_t resource, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaGraphicsResourceSetMapFlags(resource, flags)


cdef cudaError_t cudaGraphicsMapResources(int count, cudaGraphicsResource_t* resources, cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaGraphicsMapResources(count, resources, stream)


cdef cudaError_t cudaGraphicsUnmapResources(int count, cudaGraphicsResource_t* resources, cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaGraphicsUnmapResources(count, resources, stream)


cdef cudaError_t cudaGraphicsResourceGetMappedPointer(void** devPtr, size_t* size, cudaGraphicsResource_t resource) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaGraphicsResourceGetMappedPointer(devPtr, size, resource)


cdef cudaError_t cudaGraphicsSubResourceGetMappedArray(cudaArray_t* array, cudaGraphicsResource_t resource, unsigned int arrayIndex, unsigned int mipLevel) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaGraphicsSubResourceGetMappedArray(array, resource, arrayIndex, mipLevel)


cdef cudaError_t cudaGraphicsResourceGetMappedMipmappedArray(cudaMipmappedArray_t* mipmappedArray, cudaGraphicsResource_t resource) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaGraphicsResourceGetMappedMipmappedArray(mipmappedArray, resource)


cdef cudaError_t cudaGetChannelDesc(cudaChannelFormatDesc* desc, cudaArray_const_t array) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaGetChannelDesc(desc, array)


cdef cudaChannelFormatDesc cudaCreateChannelDesc(int x, int y, int z, int w, cudaChannelFormatKind f) except* nogil:
    return _runtime._cudaCreateChannelDesc(x, y, z, w, f)


cdef cudaError_t cudaCreateTextureObject(cudaTextureObject_t* pTexObject, const cudaResourceDesc* pResDesc, const cudaTextureDesc* pTexDesc, const cudaResourceViewDesc* pResViewDesc) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaCreateTextureObject(pTexObject, pResDesc, pTexDesc, pResViewDesc)


cdef cudaError_t cudaDestroyTextureObject(cudaTextureObject_t texObject) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaDestroyTextureObject(texObject)


cdef cudaError_t cudaGetTextureObjectResourceDesc(cudaResourceDesc* pResDesc, cudaTextureObject_t texObject) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaGetTextureObjectResourceDesc(pResDesc, texObject)


cdef cudaError_t cudaGetTextureObjectTextureDesc(cudaTextureDesc* pTexDesc, cudaTextureObject_t texObject) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaGetTextureObjectTextureDesc(pTexDesc, texObject)


cdef cudaError_t cudaGetTextureObjectResourceViewDesc(cudaResourceViewDesc* pResViewDesc, cudaTextureObject_t texObject) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaGetTextureObjectResourceViewDesc(pResViewDesc, texObject)


cdef cudaError_t cudaCreateSurfaceObject(cudaSurfaceObject_t* pSurfObject, const cudaResourceDesc* pResDesc) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaCreateSurfaceObject(pSurfObject, pResDesc)


cdef cudaError_t cudaDestroySurfaceObject(cudaSurfaceObject_t surfObject) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaDestroySurfaceObject(surfObject)


cdef cudaError_t cudaGetSurfaceObjectResourceDesc(cudaResourceDesc* pResDesc, cudaSurfaceObject_t surfObject) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaGetSurfaceObjectResourceDesc(pResDesc, surfObject)


cdef cudaError_t cudaDriverGetVersion(int* driverVersion) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaDriverGetVersion(driverVersion)


cdef cudaError_t cudaRuntimeGetVersion(int* runtimeVersion) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaRuntimeGetVersion(runtimeVersion)


cdef cudaError_t cudaGraphCreate(cudaGraph_t* pGraph, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaGraphCreate(pGraph, flags)


cdef cudaError_t cudaGraphAddKernelNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, const cudaKernelNodeParams* pNodeParams) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaGraphAddKernelNode(pGraphNode, graph, pDependencies, numDependencies, pNodeParams)


cdef cudaError_t cudaGraphKernelNodeGetParams(cudaGraphNode_t node, cudaKernelNodeParams* pNodeParams) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaGraphKernelNodeGetParams(node, pNodeParams)


cdef cudaError_t cudaGraphKernelNodeSetParams(cudaGraphNode_t node, const cudaKernelNodeParams* pNodeParams) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaGraphKernelNodeSetParams(node, pNodeParams)


cdef cudaError_t cudaGraphKernelNodeCopyAttributes(cudaGraphNode_t hDst, cudaGraphNode_t hSrc) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaGraphKernelNodeCopyAttributes(hDst, hSrc)


cdef cudaError_t cudaGraphKernelNodeGetAttribute(cudaGraphNode_t hNode, cudaKernelNodeAttrID attr, cudaKernelNodeAttrValue* value_out) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaGraphKernelNodeGetAttribute(hNode, attr, value_out)


cdef cudaError_t cudaGraphKernelNodeSetAttribute(cudaGraphNode_t hNode, cudaKernelNodeAttrID attr, const cudaKernelNodeAttrValue* value) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaGraphKernelNodeSetAttribute(hNode, attr, value)


cdef cudaError_t cudaGraphAddMemcpyNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, const cudaMemcpy3DParms* pCopyParams) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaGraphAddMemcpyNode(pGraphNode, graph, pDependencies, numDependencies, pCopyParams)


cdef cudaError_t cudaGraphAddMemcpyNode1D(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, void* dst, const void* src, size_t count, cudaMemcpyKind kind) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaGraphAddMemcpyNode1D(pGraphNode, graph, pDependencies, numDependencies, dst, src, count, kind)


cdef cudaError_t cudaGraphMemcpyNodeGetParams(cudaGraphNode_t node, cudaMemcpy3DParms* pNodeParams) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaGraphMemcpyNodeGetParams(node, pNodeParams)


cdef cudaError_t cudaGraphMemcpyNodeSetParams(cudaGraphNode_t node, const cudaMemcpy3DParms* pNodeParams) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaGraphMemcpyNodeSetParams(node, pNodeParams)


cdef cudaError_t cudaGraphMemcpyNodeSetParams1D(cudaGraphNode_t node, void* dst, const void* src, size_t count, cudaMemcpyKind kind) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaGraphMemcpyNodeSetParams1D(node, dst, src, count, kind)


cdef cudaError_t cudaGraphAddMemsetNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, const cudaMemsetParams* pMemsetParams) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaGraphAddMemsetNode(pGraphNode, graph, pDependencies, numDependencies, pMemsetParams)


cdef cudaError_t cudaGraphMemsetNodeGetParams(cudaGraphNode_t node, cudaMemsetParams* pNodeParams) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaGraphMemsetNodeGetParams(node, pNodeParams)


cdef cudaError_t cudaGraphMemsetNodeSetParams(cudaGraphNode_t node, const cudaMemsetParams* pNodeParams) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaGraphMemsetNodeSetParams(node, pNodeParams)


cdef cudaError_t cudaGraphAddHostNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, const cudaHostNodeParams* pNodeParams) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaGraphAddHostNode(pGraphNode, graph, pDependencies, numDependencies, pNodeParams)


cdef cudaError_t cudaGraphHostNodeGetParams(cudaGraphNode_t node, cudaHostNodeParams* pNodeParams) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaGraphHostNodeGetParams(node, pNodeParams)


cdef cudaError_t cudaGraphHostNodeSetParams(cudaGraphNode_t node, const cudaHostNodeParams* pNodeParams) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaGraphHostNodeSetParams(node, pNodeParams)


cdef cudaError_t cudaGraphAddChildGraphNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, cudaGraph_t childGraph) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaGraphAddChildGraphNode(pGraphNode, graph, pDependencies, numDependencies, childGraph)


cdef cudaError_t cudaGraphChildGraphNodeGetGraph(cudaGraphNode_t node, cudaGraph_t* pGraph) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaGraphChildGraphNodeGetGraph(node, pGraph)


cdef cudaError_t cudaGraphAddEmptyNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaGraphAddEmptyNode(pGraphNode, graph, pDependencies, numDependencies)


cdef cudaError_t cudaGraphAddEventRecordNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, cudaEvent_t event) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaGraphAddEventRecordNode(pGraphNode, graph, pDependencies, numDependencies, event)


cdef cudaError_t cudaGraphEventRecordNodeGetEvent(cudaGraphNode_t node, cudaEvent_t* event_out) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaGraphEventRecordNodeGetEvent(node, event_out)


cdef cudaError_t cudaGraphEventRecordNodeSetEvent(cudaGraphNode_t node, cudaEvent_t event) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaGraphEventRecordNodeSetEvent(node, event)


cdef cudaError_t cudaGraphAddEventWaitNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, cudaEvent_t event) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaGraphAddEventWaitNode(pGraphNode, graph, pDependencies, numDependencies, event)


cdef cudaError_t cudaGraphEventWaitNodeGetEvent(cudaGraphNode_t node, cudaEvent_t* event_out) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaGraphEventWaitNodeGetEvent(node, event_out)


cdef cudaError_t cudaGraphEventWaitNodeSetEvent(cudaGraphNode_t node, cudaEvent_t event) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaGraphEventWaitNodeSetEvent(node, event)


cdef cudaError_t cudaGraphAddExternalSemaphoresSignalNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, const cudaExternalSemaphoreSignalNodeParams* nodeParams) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaGraphAddExternalSemaphoresSignalNode(pGraphNode, graph, pDependencies, numDependencies, nodeParams)


cdef cudaError_t cudaGraphExternalSemaphoresSignalNodeGetParams(cudaGraphNode_t hNode, cudaExternalSemaphoreSignalNodeParams* params_out) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaGraphExternalSemaphoresSignalNodeGetParams(hNode, params_out)


cdef cudaError_t cudaGraphExternalSemaphoresSignalNodeSetParams(cudaGraphNode_t hNode, const cudaExternalSemaphoreSignalNodeParams* nodeParams) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaGraphExternalSemaphoresSignalNodeSetParams(hNode, nodeParams)


cdef cudaError_t cudaGraphAddExternalSemaphoresWaitNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, const cudaExternalSemaphoreWaitNodeParams* nodeParams) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaGraphAddExternalSemaphoresWaitNode(pGraphNode, graph, pDependencies, numDependencies, nodeParams)


cdef cudaError_t cudaGraphExternalSemaphoresWaitNodeGetParams(cudaGraphNode_t hNode, cudaExternalSemaphoreWaitNodeParams* params_out) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaGraphExternalSemaphoresWaitNodeGetParams(hNode, params_out)


cdef cudaError_t cudaGraphExternalSemaphoresWaitNodeSetParams(cudaGraphNode_t hNode, const cudaExternalSemaphoreWaitNodeParams* nodeParams) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaGraphExternalSemaphoresWaitNodeSetParams(hNode, nodeParams)


cdef cudaError_t cudaGraphAddMemAllocNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, cudaMemAllocNodeParams* nodeParams) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaGraphAddMemAllocNode(pGraphNode, graph, pDependencies, numDependencies, nodeParams)


cdef cudaError_t cudaGraphMemAllocNodeGetParams(cudaGraphNode_t node, cudaMemAllocNodeParams* params_out) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaGraphMemAllocNodeGetParams(node, params_out)


cdef cudaError_t cudaGraphAddMemFreeNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, void* dptr) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaGraphAddMemFreeNode(pGraphNode, graph, pDependencies, numDependencies, dptr)


cdef cudaError_t cudaGraphMemFreeNodeGetParams(cudaGraphNode_t node, void* dptr_out) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaGraphMemFreeNodeGetParams(node, dptr_out)


cdef cudaError_t cudaDeviceGraphMemTrim(int device) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaDeviceGraphMemTrim(device)


cdef cudaError_t cudaDeviceGetGraphMemAttribute(int device, cudaGraphMemAttributeType attr, void* value) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaDeviceGetGraphMemAttribute(device, attr, value)


cdef cudaError_t cudaDeviceSetGraphMemAttribute(int device, cudaGraphMemAttributeType attr, void* value) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaDeviceSetGraphMemAttribute(device, attr, value)


cdef cudaError_t cudaGraphClone(cudaGraph_t* pGraphClone, cudaGraph_t originalGraph) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaGraphClone(pGraphClone, originalGraph)


cdef cudaError_t cudaGraphNodeFindInClone(cudaGraphNode_t* pNode, cudaGraphNode_t originalNode, cudaGraph_t clonedGraph) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaGraphNodeFindInClone(pNode, originalNode, clonedGraph)


cdef cudaError_t cudaGraphNodeGetType(cudaGraphNode_t node, cudaGraphNodeType* pType) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaGraphNodeGetType(node, pType)


cdef cudaError_t cudaGraphGetNodes(cudaGraph_t graph, cudaGraphNode_t* nodes, size_t* numNodes) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaGraphGetNodes(graph, nodes, numNodes)


cdef cudaError_t cudaGraphGetRootNodes(cudaGraph_t graph, cudaGraphNode_t* pRootNodes, size_t* pNumRootNodes) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaGraphGetRootNodes(graph, pRootNodes, pNumRootNodes)


cdef cudaError_t cudaGraphGetEdges(cudaGraph_t graph, cudaGraphNode_t* from_, cudaGraphNode_t* to, cudaGraphEdgeData* edgeData, size_t* numEdges) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaGraphGetEdges(graph, from_, to, edgeData, numEdges)


cdef cudaError_t cudaGraphNodeGetDependencies(cudaGraphNode_t node, cudaGraphNode_t* pDependencies, cudaGraphEdgeData* edgeData, size_t* pNumDependencies) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaGraphNodeGetDependencies(node, pDependencies, edgeData, pNumDependencies)


cdef cudaError_t cudaGraphNodeGetDependentNodes(cudaGraphNode_t node, cudaGraphNode_t* pDependentNodes, cudaGraphEdgeData* edgeData, size_t* pNumDependentNodes) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaGraphNodeGetDependentNodes(node, pDependentNodes, edgeData, pNumDependentNodes)


cdef cudaError_t cudaGraphAddDependencies(cudaGraph_t graph, const cudaGraphNode_t* from_, const cudaGraphNode_t* to, const cudaGraphEdgeData* edgeData, size_t numDependencies) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaGraphAddDependencies(graph, from_, to, edgeData, numDependencies)


cdef cudaError_t cudaGraphRemoveDependencies(cudaGraph_t graph, const cudaGraphNode_t* from_, const cudaGraphNode_t* to, const cudaGraphEdgeData* edgeData, size_t numDependencies) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaGraphRemoveDependencies(graph, from_, to, edgeData, numDependencies)


cdef cudaError_t cudaGraphDestroyNode(cudaGraphNode_t node) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaGraphDestroyNode(node)


cdef cudaError_t cudaGraphInstantiate(cudaGraphExec_t* pGraphExec, cudaGraph_t graph, unsigned long long flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaGraphInstantiate(pGraphExec, graph, flags)


cdef cudaError_t cudaGraphInstantiateWithFlags(cudaGraphExec_t* pGraphExec, cudaGraph_t graph, unsigned long long flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaGraphInstantiateWithFlags(pGraphExec, graph, flags)


cdef cudaError_t cudaGraphInstantiateWithParams(cudaGraphExec_t* pGraphExec, cudaGraph_t graph, cudaGraphInstantiateParams* instantiateParams) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaGraphInstantiateWithParams(pGraphExec, graph, instantiateParams)


cdef cudaError_t cudaGraphExecGetFlags(cudaGraphExec_t graphExec, unsigned long long* flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaGraphExecGetFlags(graphExec, flags)


cdef cudaError_t cudaGraphExecKernelNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, const cudaKernelNodeParams* pNodeParams) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaGraphExecKernelNodeSetParams(hGraphExec, node, pNodeParams)


cdef cudaError_t cudaGraphExecMemcpyNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, const cudaMemcpy3DParms* pNodeParams) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaGraphExecMemcpyNodeSetParams(hGraphExec, node, pNodeParams)


cdef cudaError_t cudaGraphExecMemcpyNodeSetParams1D(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, void* dst, const void* src, size_t count, cudaMemcpyKind kind) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaGraphExecMemcpyNodeSetParams1D(hGraphExec, node, dst, src, count, kind)


cdef cudaError_t cudaGraphExecMemsetNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, const cudaMemsetParams* pNodeParams) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaGraphExecMemsetNodeSetParams(hGraphExec, node, pNodeParams)


cdef cudaError_t cudaGraphExecHostNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, const cudaHostNodeParams* pNodeParams) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaGraphExecHostNodeSetParams(hGraphExec, node, pNodeParams)


cdef cudaError_t cudaGraphExecChildGraphNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, cudaGraph_t childGraph) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaGraphExecChildGraphNodeSetParams(hGraphExec, node, childGraph)


cdef cudaError_t cudaGraphExecEventRecordNodeSetEvent(cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, cudaEvent_t event) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaGraphExecEventRecordNodeSetEvent(hGraphExec, hNode, event)


cdef cudaError_t cudaGraphExecEventWaitNodeSetEvent(cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, cudaEvent_t event) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaGraphExecEventWaitNodeSetEvent(hGraphExec, hNode, event)


cdef cudaError_t cudaGraphExecExternalSemaphoresSignalNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, const cudaExternalSemaphoreSignalNodeParams* nodeParams) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaGraphExecExternalSemaphoresSignalNodeSetParams(hGraphExec, hNode, nodeParams)


cdef cudaError_t cudaGraphExecExternalSemaphoresWaitNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, const cudaExternalSemaphoreWaitNodeParams* nodeParams) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaGraphExecExternalSemaphoresWaitNodeSetParams(hGraphExec, hNode, nodeParams)


cdef cudaError_t cudaGraphNodeSetEnabled(cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, unsigned int isEnabled) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaGraphNodeSetEnabled(hGraphExec, hNode, isEnabled)


cdef cudaError_t cudaGraphNodeGetEnabled(cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, unsigned int* isEnabled) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaGraphNodeGetEnabled(hGraphExec, hNode, isEnabled)


cdef cudaError_t cudaGraphExecUpdate(cudaGraphExec_t hGraphExec, cudaGraph_t hGraph, cudaGraphExecUpdateResultInfo* resultInfo) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaGraphExecUpdate(hGraphExec, hGraph, resultInfo)


cdef cudaError_t cudaGraphUpload(cudaGraphExec_t graphExec, cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaGraphUpload(graphExec, stream)


cdef cudaError_t cudaGraphLaunch(cudaGraphExec_t graphExec, cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaGraphLaunch(graphExec, stream)


cdef cudaError_t cudaGraphExecDestroy(cudaGraphExec_t graphExec) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaGraphExecDestroy(graphExec)


cdef cudaError_t cudaGraphDestroy(cudaGraph_t graph) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaGraphDestroy(graph)


cdef cudaError_t cudaGraphDebugDotPrint(cudaGraph_t graph, const char* path, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaGraphDebugDotPrint(graph, path, flags)


cdef cudaError_t cudaUserObjectCreate(cudaUserObject_t* object_out, void* ptr, cudaHostFn_t destroy, unsigned int initialRefcount, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaUserObjectCreate(object_out, ptr, destroy, initialRefcount, flags)


cdef cudaError_t cudaUserObjectRetain(cudaUserObject_t object, unsigned int count) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaUserObjectRetain(object, count)


cdef cudaError_t cudaUserObjectRelease(cudaUserObject_t object, unsigned int count) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaUserObjectRelease(object, count)


cdef cudaError_t cudaGraphRetainUserObject(cudaGraph_t graph, cudaUserObject_t object, unsigned int count, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaGraphRetainUserObject(graph, object, count, flags)


cdef cudaError_t cudaGraphReleaseUserObject(cudaGraph_t graph, cudaUserObject_t object, unsigned int count) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaGraphReleaseUserObject(graph, object, count)


cdef cudaError_t cudaGraphAddNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, const cudaGraphEdgeData* dependencyData, size_t numDependencies, cudaGraphNodeParams* nodeParams) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaGraphAddNode(pGraphNode, graph, pDependencies, dependencyData, numDependencies, nodeParams)


cdef cudaError_t cudaGraphNodeSetParams(cudaGraphNode_t node, cudaGraphNodeParams* nodeParams) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaGraphNodeSetParams(node, nodeParams)


cdef cudaError_t cudaGraphExecNodeSetParams(cudaGraphExec_t graphExec, cudaGraphNode_t node, cudaGraphNodeParams* nodeParams) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaGraphExecNodeSetParams(graphExec, node, nodeParams)


cdef cudaError_t cudaGraphConditionalHandleCreate(cudaGraphConditionalHandle* pHandle_out, cudaGraph_t graph, unsigned int defaultLaunchValue, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaGraphConditionalHandleCreate(pHandle_out, graph, defaultLaunchValue, flags)


cdef cudaError_t cudaGetDriverEntryPoint(const char* symbol, void** funcPtr, unsigned long long flags, cudaDriverEntryPointQueryResult* driverStatus) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaGetDriverEntryPoint(symbol, funcPtr, flags, driverStatus)


cdef cudaError_t cudaGetDriverEntryPointByVersion(const char* symbol, void** funcPtr, unsigned int cudaVersion, unsigned long long flags, cudaDriverEntryPointQueryResult* driverStatus) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaGetDriverEntryPointByVersion(symbol, funcPtr, cudaVersion, flags, driverStatus)


cdef cudaError_t cudaLibraryLoadData(cudaLibrary_t* library, const void* code, cudaJitOption* jitOptions, void** jitOptionsValues, unsigned int numJitOptions, cudaLibraryOption* libraryOptions, void** libraryOptionValues, unsigned int numLibraryOptions) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaLibraryLoadData(library, code, jitOptions, jitOptionsValues, numJitOptions, libraryOptions, libraryOptionValues, numLibraryOptions)


cdef cudaError_t cudaLibraryLoadFromFile(cudaLibrary_t* library, const char* fileName, cudaJitOption* jitOptions, void** jitOptionsValues, unsigned int numJitOptions, cudaLibraryOption* libraryOptions, void** libraryOptionValues, unsigned int numLibraryOptions) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaLibraryLoadFromFile(library, fileName, jitOptions, jitOptionsValues, numJitOptions, libraryOptions, libraryOptionValues, numLibraryOptions)


cdef cudaError_t cudaLibraryUnload(cudaLibrary_t library) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaLibraryUnload(library)


cdef cudaError_t cudaLibraryGetKernel(cudaKernel_t* pKernel, cudaLibrary_t library, const char* name) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaLibraryGetKernel(pKernel, library, name)


cdef cudaError_t cudaLibraryGetGlobal(void** dptr, size_t* bytes, cudaLibrary_t library, const char* name) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaLibraryGetGlobal(dptr, bytes, library, name)


cdef cudaError_t cudaLibraryGetManaged(void** dptr, size_t* bytes, cudaLibrary_t library, const char* name) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaLibraryGetManaged(dptr, bytes, library, name)


cdef cudaError_t cudaLibraryGetUnifiedFunction(void** fptr, cudaLibrary_t library, const char* symbol) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaLibraryGetUnifiedFunction(fptr, library, symbol)


cdef cudaError_t cudaLibraryGetKernelCount(unsigned int* count, cudaLibrary_t lib) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaLibraryGetKernelCount(count, lib)


cdef cudaError_t cudaLibraryEnumerateKernels(cudaKernel_t* kernels, unsigned int numKernels, cudaLibrary_t lib) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaLibraryEnumerateKernels(kernels, numKernels, lib)


cdef cudaError_t cudaKernelSetAttributeForDevice(cudaKernel_t kernel, cudaFuncAttribute attr, int value, int device) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaKernelSetAttributeForDevice(kernel, attr, value, device)


cdef cudaError_t cudaGetExportTable(const void** ppExportTable, const cudaUUID_t* pExportTableId) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaGetExportTable(ppExportTable, pExportTableId)


cdef cudaError_t cudaGetKernel(cudaKernel_t* kernelPtr, const void* entryFuncAddr) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaGetKernel(kernelPtr, entryFuncAddr)


cdef cudaError_t cudaGraphicsEGLRegisterImage(cudaGraphicsResource** pCudaResource, EGLImageKHR image, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaGraphicsEGLRegisterImage(pCudaResource, image, flags)


cdef cudaError_t cudaEGLStreamConsumerConnect(cudaEglStreamConnection* conn, EGLStreamKHR eglStream) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaEGLStreamConsumerConnect(conn, eglStream)


cdef cudaError_t cudaEGLStreamConsumerConnectWithFlags(cudaEglStreamConnection* conn, EGLStreamKHR eglStream, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaEGLStreamConsumerConnectWithFlags(conn, eglStream, flags)


cdef cudaError_t cudaEGLStreamConsumerDisconnect(cudaEglStreamConnection* conn) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaEGLStreamConsumerDisconnect(conn)


cdef cudaError_t cudaEGLStreamConsumerAcquireFrame(cudaEglStreamConnection* conn, cudaGraphicsResource_t* pCudaResource, cudaStream_t* pStream, unsigned int timeout) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaEGLStreamConsumerAcquireFrame(conn, pCudaResource, pStream, timeout)


cdef cudaError_t cudaEGLStreamConsumerReleaseFrame(cudaEglStreamConnection* conn, cudaGraphicsResource_t pCudaResource, cudaStream_t* pStream) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaEGLStreamConsumerReleaseFrame(conn, pCudaResource, pStream)


cdef cudaError_t cudaEGLStreamProducerConnect(cudaEglStreamConnection* conn, EGLStreamKHR eglStream, EGLint width, EGLint height) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaEGLStreamProducerConnect(conn, eglStream, width, height)


cdef cudaError_t cudaEGLStreamProducerDisconnect(cudaEglStreamConnection* conn) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaEGLStreamProducerDisconnect(conn)


cdef cudaError_t cudaEGLStreamProducerPresentFrame(cudaEglStreamConnection* conn, cudaEglFrame eglframe, cudaStream_t* pStream) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaEGLStreamProducerPresentFrame(conn, eglframe, pStream)


cdef cudaError_t cudaEGLStreamProducerReturnFrame(cudaEglStreamConnection* conn, cudaEglFrame* eglframe, cudaStream_t* pStream) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaEGLStreamProducerReturnFrame(conn, eglframe, pStream)


cdef cudaError_t cudaGraphicsResourceGetMappedEglFrame(cudaEglFrame* eglFrame, cudaGraphicsResource_t resource, unsigned int index, unsigned int mipLevel) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaGraphicsResourceGetMappedEglFrame(eglFrame, resource, index, mipLevel)


cdef cudaError_t cudaEventCreateFromEGLSync(cudaEvent_t* phEvent, EGLSyncKHR eglSync, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaEventCreateFromEGLSync(phEvent, eglSync, flags)


cdef cudaError_t cudaProfilerStart() except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaProfilerStart()


cdef cudaError_t cudaProfilerStop() except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaProfilerStop()


cdef cudaError_t cudaGLGetDevices(unsigned int* pCudaDeviceCount, int* pCudaDevices, unsigned int cudaDeviceCount, cudaGLDeviceList deviceList) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaGLGetDevices(pCudaDeviceCount, pCudaDevices, cudaDeviceCount, deviceList)


cdef cudaError_t cudaGraphicsGLRegisterImage(cudaGraphicsResource** resource, GLuint image, GLenum target, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaGraphicsGLRegisterImage(resource, image, target, flags)


cdef cudaError_t cudaGraphicsGLRegisterBuffer(cudaGraphicsResource** resource, GLuint buffer, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaGraphicsGLRegisterBuffer(resource, buffer, flags)


cdef cudaError_t cudaVDPAUGetDevice(int* device, VdpDevice vdpDevice, VdpGetProcAddress* vdpGetProcAddress) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaVDPAUGetDevice(device, vdpDevice, vdpGetProcAddress)


cdef cudaError_t cudaVDPAUSetVDPAUDevice(int device, VdpDevice vdpDevice, VdpGetProcAddress* vdpGetProcAddress) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaVDPAUSetVDPAUDevice(device, vdpDevice, vdpGetProcAddress)


cdef cudaError_t cudaGraphicsVDPAURegisterVideoSurface(cudaGraphicsResource** resource, VdpVideoSurface vdpSurface, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaGraphicsVDPAURegisterVideoSurface(resource, vdpSurface, flags)


cdef cudaError_t cudaGraphicsVDPAURegisterOutputSurface(cudaGraphicsResource** resource, VdpOutputSurface vdpSurface, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaGraphicsVDPAURegisterOutputSurface(resource, vdpSurface, flags)


cdef cudaError_t cudaGetDeviceProperties(cudaDeviceProp* prop, int device) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaGetDeviceProperties(prop, device)


cdef cudaError_t cudaDeviceGetHostAtomicCapabilities(unsigned int* capabilities, const cudaAtomicOperation* operations, unsigned int count, int device) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaDeviceGetHostAtomicCapabilities(capabilities, operations, count, device)


cdef cudaError_t cudaDeviceGetP2PAtomicCapabilities(unsigned int* capabilities, const cudaAtomicOperation* operations, unsigned int count, int srcDevice, int dstDevice) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaDeviceGetP2PAtomicCapabilities(capabilities, operations, count, srcDevice, dstDevice)


cdef cudaError_t cudaStreamGetCaptureInfo(cudaStream_t stream, cudaStreamCaptureStatus* captureStatus_out, unsigned long long* id_out, cudaGraph_t* graph_out, const cudaGraphNode_t** dependencies_out, const cudaGraphEdgeData** edgeData_out, size_t* numDependencies_out) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaStreamGetCaptureInfo(stream, captureStatus_out, id_out, graph_out, dependencies_out, edgeData_out, numDependencies_out)


cdef cudaError_t cudaSignalExternalSemaphoresAsync(const cudaExternalSemaphore_t* extSemArray, const cudaExternalSemaphoreSignalParams* paramsArray, unsigned int numExtSems, cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaSignalExternalSemaphoresAsync(extSemArray, paramsArray, numExtSems, stream)


cdef cudaError_t cudaWaitExternalSemaphoresAsync(const cudaExternalSemaphore_t* extSemArray, const cudaExternalSemaphoreWaitParams* paramsArray, unsigned int numExtSems, cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaWaitExternalSemaphoresAsync(extSemArray, paramsArray, numExtSems, stream)


cdef cudaError_t cudaMemPrefetchBatchAsync(void** dptrs, size_t* sizes, size_t count, cudaMemLocation* prefetchLocs, size_t* prefetchLocIdxs, size_t numPrefetchLocs, unsigned long long flags, cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaMemPrefetchBatchAsync(dptrs, sizes, count, prefetchLocs, prefetchLocIdxs, numPrefetchLocs, flags, stream)


cdef cudaError_t cudaMemDiscardBatchAsync(void** dptrs, size_t* sizes, size_t count, unsigned long long flags, cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaMemDiscardBatchAsync(dptrs, sizes, count, flags, stream)


cdef cudaError_t cudaMemDiscardAndPrefetchBatchAsync(void** dptrs, size_t* sizes, size_t count, cudaMemLocation* prefetchLocs, size_t* prefetchLocIdxs, size_t numPrefetchLocs, unsigned long long flags, cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaMemDiscardAndPrefetchBatchAsync(dptrs, sizes, count, prefetchLocs, prefetchLocIdxs, numPrefetchLocs, flags, stream)


cdef cudaError_t cudaMemGetDefaultMemPool(cudaMemPool_t* memPool, cudaMemLocation* location, cudaMemAllocationType type) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaMemGetDefaultMemPool(memPool, location, type)


cdef cudaError_t cudaMemGetMemPool(cudaMemPool_t* memPool, cudaMemLocation* location, cudaMemAllocationType type) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaMemGetMemPool(memPool, location, type)


cdef cudaError_t cudaMemSetMemPool(cudaMemLocation* location, cudaMemAllocationType type, cudaMemPool_t memPool) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaMemSetMemPool(location, type, memPool)


cdef cudaError_t cudaLogsRegisterCallback(cudaLogsCallback_t callbackFunc, void* userData, cudaLogsCallbackHandle* callback_out) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaLogsRegisterCallback(callbackFunc, userData, callback_out)


cdef cudaError_t cudaLogsUnregisterCallback(cudaLogsCallbackHandle callback) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaLogsUnregisterCallback(callback)


cdef cudaError_t cudaLogsCurrent(cudaLogIterator* iterator_out, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaLogsCurrent(iterator_out, flags)


cdef cudaError_t cudaLogsDumpToFile(cudaLogIterator* iterator, const char* pathToFile, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaLogsDumpToFile(iterator, pathToFile, flags)


cdef cudaError_t cudaLogsDumpToMemory(cudaLogIterator* iterator, char* buffer, size_t* size, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaLogsDumpToMemory(iterator, buffer, size, flags)


cdef cudaError_t cudaGraphNodeGetContainingGraph(cudaGraphNode_t hNode, cudaGraph_t* phGraph) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaGraphNodeGetContainingGraph(hNode, phGraph)


cdef cudaError_t cudaGraphNodeGetLocalId(cudaGraphNode_t hNode, unsigned int* nodeId) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaGraphNodeGetLocalId(hNode, nodeId)


cdef cudaError_t cudaGraphNodeGetToolsId(cudaGraphNode_t hNode, unsigned long long* toolsNodeId) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaGraphNodeGetToolsId(hNode, toolsNodeId)


cdef cudaError_t cudaGraphGetId(cudaGraph_t hGraph, unsigned int* graphID) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaGraphGetId(hGraph, graphID)


cdef cudaError_t cudaGraphExecGetId(cudaGraphExec_t hGraphExec, unsigned int* graphID) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaGraphExecGetId(hGraphExec, graphID)


cdef cudaError_t cudaGraphConditionalHandleCreate_v2(cudaGraphConditionalHandle* pHandle_out, cudaGraph_t graph, cudaExecutionContext_t ctx, unsigned int defaultLaunchValue, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaGraphConditionalHandleCreate_v2(pHandle_out, graph, ctx, defaultLaunchValue, flags)


cdef cudaError_t cudaDeviceGetDevResource(int device, cudaDevResource* resource, cudaDevResourceType type) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaDeviceGetDevResource(device, resource, type)


cdef cudaError_t cudaDevSmResourceSplitByCount(cudaDevResource* result, unsigned int* nbGroups, const cudaDevResource* input, cudaDevResource* remaining, unsigned int flags, unsigned int minCount) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaDevSmResourceSplitByCount(result, nbGroups, input, remaining, flags, minCount)


cdef cudaError_t cudaDevSmResourceSplit(cudaDevResource* result, unsigned int nbGroups, const cudaDevResource* input, cudaDevResource* remainder, unsigned int flags, cudaDevSmResourceGroupParams* groupParams) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaDevSmResourceSplit(result, nbGroups, input, remainder, flags, groupParams)


cdef cudaError_t cudaDevResourceGenerateDesc(cudaDevResourceDesc_t* phDesc, cudaDevResource* resources, unsigned int nbResources) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaDevResourceGenerateDesc(phDesc, resources, nbResources)


cdef cudaError_t cudaGreenCtxCreate(cudaExecutionContext_t* phCtx, cudaDevResourceDesc_t desc, int device, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaGreenCtxCreate(phCtx, desc, device, flags)


cdef cudaError_t cudaExecutionCtxDestroy(cudaExecutionContext_t ctx) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaExecutionCtxDestroy(ctx)


cdef cudaError_t cudaExecutionCtxGetDevResource(cudaExecutionContext_t ctx, cudaDevResource* resource, cudaDevResourceType type) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaExecutionCtxGetDevResource(ctx, resource, type)


cdef cudaError_t cudaExecutionCtxGetDevice(int* device, cudaExecutionContext_t ctx) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaExecutionCtxGetDevice(device, ctx)


cdef cudaError_t cudaExecutionCtxGetId(cudaExecutionContext_t ctx, unsigned long long* ctxId) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaExecutionCtxGetId(ctx, ctxId)


cdef cudaError_t cudaExecutionCtxStreamCreate(cudaStream_t* phStream, cudaExecutionContext_t ctx, unsigned int flags, int priority) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaExecutionCtxStreamCreate(phStream, ctx, flags, priority)


cdef cudaError_t cudaExecutionCtxSynchronize(cudaExecutionContext_t ctx) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaExecutionCtxSynchronize(ctx)


cdef cudaError_t cudaStreamGetDevResource(cudaStream_t hStream, cudaDevResource* resource, cudaDevResourceType type) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaStreamGetDevResource(hStream, resource, type)


cdef cudaError_t cudaExecutionCtxRecordEvent(cudaExecutionContext_t ctx, cudaEvent_t event) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaExecutionCtxRecordEvent(ctx, event)


cdef cudaError_t cudaExecutionCtxWaitEvent(cudaExecutionContext_t ctx, cudaEvent_t event) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaExecutionCtxWaitEvent(ctx, event)


cdef cudaError_t cudaDeviceGetExecutionCtx(cudaExecutionContext_t* ctx, int device) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaDeviceGetExecutionCtx(ctx, device)


cdef cudaError_t cudaFuncGetParamCount(const void* func, size_t* paramCount) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaFuncGetParamCount(func, paramCount)


cdef cudaError_t cudaLaunchHostFunc_v2(cudaStream_t stream, cudaHostFn_t fn, void* userData, unsigned int syncMode) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaLaunchHostFunc_v2(stream, fn, userData, syncMode)


cdef cudaError_t cudaMemcpyWithAttributesAsync(void* dst, const void* src, size_t size, cudaMemcpyAttributes* attr, cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaMemcpyWithAttributesAsync(dst, src, size, attr, stream)


cdef cudaError_t cudaMemcpy3DWithAttributesAsync(cudaMemcpy3DBatchOp* op, unsigned long long flags, cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaMemcpy3DWithAttributesAsync(op, flags, stream)


cdef cudaError_t cudaGraphNodeGetParams(cudaGraphNode_t node, cudaGraphNodeParams* nodeParams) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaGraphNodeGetParams(node, nodeParams)


cdef cudaError_t cudaStreamBeginRecaptureToGraph(cudaStream_t stream, cudaStreamCaptureMode mode, cudaGraph_t graph, cudaGraphRecaptureCallbackData* callbackData) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._cudaStreamBeginRecaptureToGraph(stream, mode, graph, callbackData)


###############################################################################
# Static inline helpers from driver_functions.h
###############################################################################

cdef extern from 'driver_functions.h':
    cudaPitchedPtr _c_make_cudaPitchedPtr "make_cudaPitchedPtr"(void* d, size_t p, size_t xsz, size_t ysz) nogil
    cudaPos _c_make_cudaPos "make_cudaPos"(size_t x, size_t y, size_t z) nogil
    cudaExtent _c_make_cudaExtent "make_cudaExtent"(size_t w, size_t h, size_t d) nogil


cdef cudaPitchedPtr make_cudaPitchedPtr(void* d, size_t p, size_t xsz, size_t ysz) except* nogil:
    return _c_make_cudaPitchedPtr(d, p, xsz, ysz)


cdef cudaPos make_cudaPos(size_t x, size_t y, size_t z) except* nogil:
    return _c_make_cudaPos(x, y, z)


cdef cudaExtent make_cudaExtent(size_t w, size_t h, size_t d) except* nogil:
    return _c_make_cudaExtent(w, h, d)


###############################################################################
# getLocalRuntimeVersion
###############################################################################

cdef cudaError_t getLocalRuntimeVersion(int* runtimeVersion) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _runtime._getLocalRuntimeVersion(runtimeVersion)
