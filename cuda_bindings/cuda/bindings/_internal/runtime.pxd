# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
#
# This code was automatically generated across versions from 12.9.0 to 13.3.0. Do not modify it directly.

# CYTHON-BINDINGS-GENERATED-DO-NOT-MODIFY-THIS-FILE: format=1; content-sha256=afc5b9003f3efc6f826b46b17b5294f581a00a97f82beaad39634b5037b7669a
from ..cyruntime cimport *

# EGL/GL/VDPAU helper declarations (implementations included in runtime_linux/windows.pyx)
include "../_lib/cyruntime/cyruntime.pxd"

cdef cudaError_t _getLocalRuntimeVersion(int* runtimeVersion) except ?cudaErrorCallRequiresNewerDriver nogil


###############################################################################
# Wrapper functions
###############################################################################

cdef cudaError_t _cudaDeviceReset() except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaDeviceSynchronize() except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaDeviceSetLimit(cudaLimit limit, size_t value) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaDeviceGetLimit(size_t* pValue, cudaLimit limit) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaDeviceGetTexture1DLinearMaxWidth(size_t* maxWidthInElements, const cudaChannelFormatDesc* fmtDesc, int device) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaDeviceGetCacheConfig(cudaFuncCache* pCacheConfig) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaDeviceGetStreamPriorityRange(int* leastPriority, int* greatestPriority) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaDeviceSetCacheConfig(cudaFuncCache cacheConfig) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaDeviceGetByPCIBusId(int* device, const char* pciBusId) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaDeviceGetPCIBusId(char* pciBusId, int len, int device) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaIpcGetEventHandle(cudaIpcEventHandle_t* handle, cudaEvent_t event) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaIpcOpenEventHandle(cudaEvent_t* event, cudaIpcEventHandle_t handle) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaIpcGetMemHandle(cudaIpcMemHandle_t* handle, void* devPtr) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaIpcOpenMemHandle(void** devPtr, cudaIpcMemHandle_t handle, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaIpcCloseMemHandle(void* devPtr) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaDeviceFlushGPUDirectRDMAWrites(cudaFlushGPUDirectRDMAWritesTarget target, cudaFlushGPUDirectRDMAWritesScope scope) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaDeviceRegisterAsyncNotification(int device, cudaAsyncCallback callbackFunc, void* userData, cudaAsyncCallbackHandle_t* callback) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaDeviceUnregisterAsyncNotification(int device, cudaAsyncCallbackHandle_t callback) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaDeviceGetSharedMemConfig(cudaSharedMemConfig* pConfig) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaDeviceSetSharedMemConfig(cudaSharedMemConfig config) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaGetLastError() except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaPeekAtLastError() except ?cudaErrorCallRequiresNewerDriver nogil
cdef const char* _cudaGetErrorName(cudaError_t error) except?NULL nogil
cdef const char* _cudaGetErrorString(cudaError_t error) except?NULL nogil
cdef cudaError_t _cudaGetDeviceCount(int* count) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaDeviceGetAttribute(int* value, cudaDeviceAttr attr, int device) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaDeviceGetDefaultMemPool(cudaMemPool_t* memPool, int device) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaDeviceSetMemPool(int device, cudaMemPool_t memPool) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaDeviceGetMemPool(cudaMemPool_t* memPool, int device) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaDeviceGetNvSciSyncAttributes(void* nvSciSyncAttrList, int device, int flags) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaDeviceGetP2PAttribute(int* value, cudaDeviceP2PAttr attr, int srcDevice, int dstDevice) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaChooseDevice(int* device, const cudaDeviceProp* prop) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaInitDevice(int device, unsigned int deviceFlags, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaSetDevice(int device) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaGetDevice(int* device) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaSetDeviceFlags(unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaGetDeviceFlags(unsigned int* flags) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaStreamCreate(cudaStream_t* pStream) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaStreamCreateWithFlags(cudaStream_t* pStream, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaStreamCreateWithPriority(cudaStream_t* pStream, unsigned int flags, int priority) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaStreamGetPriority(cudaStream_t hStream, int* priority) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaStreamGetFlags(cudaStream_t hStream, unsigned int* flags) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaStreamGetId(cudaStream_t hStream, unsigned long long* streamId) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaStreamGetDevice(cudaStream_t hStream, int* device) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaCtxResetPersistingL2Cache() except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaStreamCopyAttributes(cudaStream_t dst, cudaStream_t src) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaStreamGetAttribute(cudaStream_t hStream, cudaStreamAttrID attr, cudaStreamAttrValue* value_out) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaStreamSetAttribute(cudaStream_t hStream, cudaStreamAttrID attr, const cudaStreamAttrValue* value) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaStreamDestroy(cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaStreamAddCallback(cudaStream_t stream, cudaStreamCallback_t callback, void* userData, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaStreamSynchronize(cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaStreamQuery(cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaStreamAttachMemAsync(cudaStream_t stream, void* devPtr, size_t length, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaStreamBeginCapture(cudaStream_t stream, cudaStreamCaptureMode mode) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaStreamBeginCaptureToGraph(cudaStream_t stream, cudaGraph_t graph, const cudaGraphNode_t* dependencies, const cudaGraphEdgeData* dependencyData, size_t numDependencies, cudaStreamCaptureMode mode) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaThreadExchangeStreamCaptureMode(cudaStreamCaptureMode* mode) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaStreamEndCapture(cudaStream_t stream, cudaGraph_t* pGraph) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaStreamIsCapturing(cudaStream_t stream, cudaStreamCaptureStatus* pCaptureStatus) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaStreamUpdateCaptureDependencies(cudaStream_t stream, cudaGraphNode_t* dependencies, const cudaGraphEdgeData* dependencyData, size_t numDependencies, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaEventCreate(cudaEvent_t* event) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaEventCreateWithFlags(cudaEvent_t* event, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaEventRecord(cudaEvent_t event, cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaEventRecordWithFlags(cudaEvent_t event, cudaStream_t stream, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaEventQuery(cudaEvent_t event) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaEventSynchronize(cudaEvent_t event) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaEventDestroy(cudaEvent_t event) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaEventElapsedTime(float* ms, cudaEvent_t start, cudaEvent_t end) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaImportExternalMemory(cudaExternalMemory_t* extMem_out, const cudaExternalMemoryHandleDesc* memHandleDesc) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaExternalMemoryGetMappedBuffer(void** devPtr, cudaExternalMemory_t extMem, const cudaExternalMemoryBufferDesc* bufferDesc) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaExternalMemoryGetMappedMipmappedArray(cudaMipmappedArray_t* mipmap, cudaExternalMemory_t extMem, const cudaExternalMemoryMipmappedArrayDesc* mipmapDesc) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaDestroyExternalMemory(cudaExternalMemory_t extMem) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaImportExternalSemaphore(cudaExternalSemaphore_t* extSem_out, const cudaExternalSemaphoreHandleDesc* semHandleDesc) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaDestroyExternalSemaphore(cudaExternalSemaphore_t extSem) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaFuncSetCacheConfig(const void* func, cudaFuncCache cacheConfig) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaFuncGetAttributes(cudaFuncAttributes* attr, const void* func) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaFuncSetAttribute(const void* func, cudaFuncAttribute attr, int value) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaFuncGetName(const char** name, const void* func) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaFuncGetParamInfo(const void* func, size_t paramIndex, size_t* paramOffset, size_t* paramSize) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaLaunchHostFunc(cudaStream_t stream, cudaHostFn_t fn, void* userData) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaFuncSetSharedMemConfig(const void* func, cudaSharedMemConfig config) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaOccupancyMaxActiveBlocksPerMultiprocessor(int* numBlocks, const void* func, int blockSize, size_t dynamicSMemSize) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaOccupancyAvailableDynamicSMemPerBlock(size_t* dynamicSmemSize, const void* func, int numBlocks, int blockSize) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int* numBlocks, const void* func, int blockSize, size_t dynamicSMemSize, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaMallocManaged(void** devPtr, size_t size, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaMalloc(void** devPtr, size_t size) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaMallocHost(void** ptr, size_t size) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaMallocPitch(void** devPtr, size_t* pitch, size_t width, size_t height) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaMallocArray(cudaArray_t* array, const cudaChannelFormatDesc* desc, size_t width, size_t height, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaFree(void* devPtr) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaFreeHost(void* ptr) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaFreeArray(cudaArray_t array) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaFreeMipmappedArray(cudaMipmappedArray_t mipmappedArray) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaHostAlloc(void** pHost, size_t size, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaHostRegister(void* ptr, size_t size, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaHostUnregister(void* ptr) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaHostGetDevicePointer(void** pDevice, void* pHost, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaHostGetFlags(unsigned int* pFlags, void* pHost) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaMalloc3D(cudaPitchedPtr* pitchedDevPtr, cudaExtent extent) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaMalloc3DArray(cudaArray_t* array, const cudaChannelFormatDesc* desc, cudaExtent extent, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaMallocMipmappedArray(cudaMipmappedArray_t* mipmappedArray, const cudaChannelFormatDesc* desc, cudaExtent extent, unsigned int numLevels, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaGetMipmappedArrayLevel(cudaArray_t* levelArray, cudaMipmappedArray_const_t mipmappedArray, unsigned int level) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaMemcpy3D(const cudaMemcpy3DParms* p) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaMemcpy3DPeer(const cudaMemcpy3DPeerParms* p) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaMemcpy3DAsync(const cudaMemcpy3DParms* p, cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaMemcpy3DPeerAsync(const cudaMemcpy3DPeerParms* p, cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaMemGetInfo(size_t* free, size_t* total) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaArrayGetInfo(cudaChannelFormatDesc* desc, cudaExtent* extent, unsigned int* flags, cudaArray_t array) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaArrayGetPlane(cudaArray_t* pPlaneArray, cudaArray_t hArray, unsigned int planeIdx) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaArrayGetMemoryRequirements(cudaArrayMemoryRequirements* memoryRequirements, cudaArray_t array, int device) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaMipmappedArrayGetMemoryRequirements(cudaArrayMemoryRequirements* memoryRequirements, cudaMipmappedArray_t mipmap, int device) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaArrayGetSparseProperties(cudaArraySparseProperties* sparseProperties, cudaArray_t array) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaMipmappedArrayGetSparseProperties(cudaArraySparseProperties* sparseProperties, cudaMipmappedArray_t mipmap) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaMemcpyPeer(void* dst, int dstDevice, const void* src, int srcDevice, size_t count) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaMemcpy2D(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaMemcpy2DToArray(cudaArray_t dst, size_t wOffset, size_t hOffset, const void* src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaMemcpy2DFromArray(void* dst, size_t dpitch, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t width, size_t height, cudaMemcpyKind kind) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaMemcpy2DArrayToArray(cudaArray_t dst, size_t wOffsetDst, size_t hOffsetDst, cudaArray_const_t src, size_t wOffsetSrc, size_t hOffsetSrc, size_t width, size_t height, cudaMemcpyKind kind) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaMemcpyAsync(void* dst, const void* src, size_t count, cudaMemcpyKind kind, cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaMemcpyPeerAsync(void* dst, int dstDevice, const void* src, int srcDevice, size_t count, cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaMemcpyBatchAsync(const void** dsts, const void** srcs, const size_t* sizes, size_t count, cudaMemcpyAttributes* attrs, size_t* attrsIdxs, size_t numAttrs, cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaMemcpy3DBatchAsync(size_t numOps, cudaMemcpy3DBatchOp* opList, unsigned long long flags, cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaMemcpy2DAsync(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaMemcpy2DToArrayAsync(cudaArray_t dst, size_t wOffset, size_t hOffset, const void* src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaMemcpy2DFromArrayAsync(void* dst, size_t dpitch, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaMemset(void* devPtr, int value, size_t count) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaMemset2D(void* devPtr, size_t pitch, int value, size_t width, size_t height) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaMemset3D(cudaPitchedPtr pitchedDevPtr, int value, cudaExtent extent) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaMemsetAsync(void* devPtr, int value, size_t count, cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaMemset2DAsync(void* devPtr, size_t pitch, int value, size_t width, size_t height, cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaMemset3DAsync(cudaPitchedPtr pitchedDevPtr, int value, cudaExtent extent, cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaMemPrefetchAsync(const void* devPtr, size_t count, cudaMemLocation location, unsigned int flags, cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaMemAdvise(const void* devPtr, size_t count, cudaMemoryAdvise advice, cudaMemLocation location) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaMemRangeGetAttribute(void* data, size_t dataSize, cudaMemRangeAttribute attribute, const void* devPtr, size_t count) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaMemRangeGetAttributes(void** data, size_t* dataSizes, cudaMemRangeAttribute* attributes, size_t numAttributes, const void* devPtr, size_t count) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaMemcpyToArray(cudaArray_t dst, size_t wOffset, size_t hOffset, const void* src, size_t count, cudaMemcpyKind kind) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaMemcpyFromArray(void* dst, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t count, cudaMemcpyKind kind) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaMemcpyArrayToArray(cudaArray_t dst, size_t wOffsetDst, size_t hOffsetDst, cudaArray_const_t src, size_t wOffsetSrc, size_t hOffsetSrc, size_t count, cudaMemcpyKind kind) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaMemcpyToArrayAsync(cudaArray_t dst, size_t wOffset, size_t hOffset, const void* src, size_t count, cudaMemcpyKind kind, cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaMemcpyFromArrayAsync(void* dst, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t count, cudaMemcpyKind kind, cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaMallocAsync(void** devPtr, size_t size, cudaStream_t hStream) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaFreeAsync(void* devPtr, cudaStream_t hStream) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaMemPoolTrimTo(cudaMemPool_t memPool, size_t minBytesToKeep) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaMemPoolSetAttribute(cudaMemPool_t memPool, cudaMemPoolAttr attr, void* value) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaMemPoolGetAttribute(cudaMemPool_t memPool, cudaMemPoolAttr attr, void* value) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaMemPoolSetAccess(cudaMemPool_t memPool, const cudaMemAccessDesc* descList, size_t count) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaMemPoolGetAccess(cudaMemAccessFlags* flags, cudaMemPool_t memPool, cudaMemLocation* location) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaMemPoolCreate(cudaMemPool_t* memPool, const cudaMemPoolProps* poolProps) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaMemPoolDestroy(cudaMemPool_t memPool) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaMallocFromPoolAsync(void** ptr, size_t size, cudaMemPool_t memPool, cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaMemPoolExportToShareableHandle(void* shareableHandle, cudaMemPool_t memPool, cudaMemAllocationHandleType handleType, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaMemPoolImportFromShareableHandle(cudaMemPool_t* memPool, void* shareableHandle, cudaMemAllocationHandleType handleType, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaMemPoolExportPointer(cudaMemPoolPtrExportData* exportData, void* ptr) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaMemPoolImportPointer(void** ptr, cudaMemPool_t memPool, cudaMemPoolPtrExportData* exportData) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaPointerGetAttributes(cudaPointerAttributes* attributes, const void* ptr) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaDeviceCanAccessPeer(int* canAccessPeer, int device, int peerDevice) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaDeviceEnablePeerAccess(int peerDevice, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaDeviceDisablePeerAccess(int peerDevice) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaGraphicsUnregisterResource(cudaGraphicsResource_t resource) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaGraphicsResourceSetMapFlags(cudaGraphicsResource_t resource, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaGraphicsMapResources(int count, cudaGraphicsResource_t* resources, cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaGraphicsUnmapResources(int count, cudaGraphicsResource_t* resources, cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaGraphicsResourceGetMappedPointer(void** devPtr, size_t* size, cudaGraphicsResource_t resource) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaGraphicsSubResourceGetMappedArray(cudaArray_t* array, cudaGraphicsResource_t resource, unsigned int arrayIndex, unsigned int mipLevel) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaGraphicsResourceGetMappedMipmappedArray(cudaMipmappedArray_t* mipmappedArray, cudaGraphicsResource_t resource) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaGetChannelDesc(cudaChannelFormatDesc* desc, cudaArray_const_t array) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaChannelFormatDesc _cudaCreateChannelDesc(int x, int y, int z, int w, cudaChannelFormatKind f) except* nogil
cdef cudaError_t _cudaCreateTextureObject(cudaTextureObject_t* pTexObject, const cudaResourceDesc* pResDesc, const cudaTextureDesc* pTexDesc, const cudaResourceViewDesc* pResViewDesc) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaDestroyTextureObject(cudaTextureObject_t texObject) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaGetTextureObjectResourceDesc(cudaResourceDesc* pResDesc, cudaTextureObject_t texObject) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaGetTextureObjectTextureDesc(cudaTextureDesc* pTexDesc, cudaTextureObject_t texObject) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaGetTextureObjectResourceViewDesc(cudaResourceViewDesc* pResViewDesc, cudaTextureObject_t texObject) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaCreateSurfaceObject(cudaSurfaceObject_t* pSurfObject, const cudaResourceDesc* pResDesc) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaDestroySurfaceObject(cudaSurfaceObject_t surfObject) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaGetSurfaceObjectResourceDesc(cudaResourceDesc* pResDesc, cudaSurfaceObject_t surfObject) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaDriverGetVersion(int* driverVersion) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaRuntimeGetVersion(int* runtimeVersion) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaGraphCreate(cudaGraph_t* pGraph, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaGraphAddKernelNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, const cudaKernelNodeParams* pNodeParams) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaGraphKernelNodeGetParams(cudaGraphNode_t node, cudaKernelNodeParams* pNodeParams) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaGraphKernelNodeSetParams(cudaGraphNode_t node, const cudaKernelNodeParams* pNodeParams) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaGraphKernelNodeCopyAttributes(cudaGraphNode_t hDst, cudaGraphNode_t hSrc) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaGraphKernelNodeGetAttribute(cudaGraphNode_t hNode, cudaKernelNodeAttrID attr, cudaKernelNodeAttrValue* value_out) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaGraphKernelNodeSetAttribute(cudaGraphNode_t hNode, cudaKernelNodeAttrID attr, const cudaKernelNodeAttrValue* value) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaGraphAddMemcpyNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, const cudaMemcpy3DParms* pCopyParams) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaGraphAddMemcpyNode1D(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, void* dst, const void* src, size_t count, cudaMemcpyKind kind) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaGraphMemcpyNodeGetParams(cudaGraphNode_t node, cudaMemcpy3DParms* pNodeParams) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaGraphMemcpyNodeSetParams(cudaGraphNode_t node, const cudaMemcpy3DParms* pNodeParams) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaGraphMemcpyNodeSetParams1D(cudaGraphNode_t node, void* dst, const void* src, size_t count, cudaMemcpyKind kind) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaGraphAddMemsetNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, const cudaMemsetParams* pMemsetParams) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaGraphMemsetNodeGetParams(cudaGraphNode_t node, cudaMemsetParams* pNodeParams) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaGraphMemsetNodeSetParams(cudaGraphNode_t node, const cudaMemsetParams* pNodeParams) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaGraphAddHostNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, const cudaHostNodeParams* pNodeParams) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaGraphHostNodeGetParams(cudaGraphNode_t node, cudaHostNodeParams* pNodeParams) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaGraphHostNodeSetParams(cudaGraphNode_t node, const cudaHostNodeParams* pNodeParams) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaGraphAddChildGraphNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, cudaGraph_t childGraph) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaGraphChildGraphNodeGetGraph(cudaGraphNode_t node, cudaGraph_t* pGraph) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaGraphAddEmptyNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaGraphAddEventRecordNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, cudaEvent_t event) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaGraphEventRecordNodeGetEvent(cudaGraphNode_t node, cudaEvent_t* event_out) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaGraphEventRecordNodeSetEvent(cudaGraphNode_t node, cudaEvent_t event) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaGraphAddEventWaitNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, cudaEvent_t event) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaGraphEventWaitNodeGetEvent(cudaGraphNode_t node, cudaEvent_t* event_out) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaGraphEventWaitNodeSetEvent(cudaGraphNode_t node, cudaEvent_t event) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaGraphAddExternalSemaphoresSignalNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, const cudaExternalSemaphoreSignalNodeParams* nodeParams) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaGraphExternalSemaphoresSignalNodeGetParams(cudaGraphNode_t hNode, cudaExternalSemaphoreSignalNodeParams* params_out) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaGraphExternalSemaphoresSignalNodeSetParams(cudaGraphNode_t hNode, const cudaExternalSemaphoreSignalNodeParams* nodeParams) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaGraphAddExternalSemaphoresWaitNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, const cudaExternalSemaphoreWaitNodeParams* nodeParams) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaGraphExternalSemaphoresWaitNodeGetParams(cudaGraphNode_t hNode, cudaExternalSemaphoreWaitNodeParams* params_out) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaGraphExternalSemaphoresWaitNodeSetParams(cudaGraphNode_t hNode, const cudaExternalSemaphoreWaitNodeParams* nodeParams) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaGraphAddMemAllocNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, cudaMemAllocNodeParams* nodeParams) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaGraphMemAllocNodeGetParams(cudaGraphNode_t node, cudaMemAllocNodeParams* params_out) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaGraphAddMemFreeNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, void* dptr) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaGraphMemFreeNodeGetParams(cudaGraphNode_t node, void* dptr_out) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaDeviceGraphMemTrim(int device) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaDeviceGetGraphMemAttribute(int device, cudaGraphMemAttributeType attr, void* value) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaDeviceSetGraphMemAttribute(int device, cudaGraphMemAttributeType attr, void* value) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaGraphClone(cudaGraph_t* pGraphClone, cudaGraph_t originalGraph) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaGraphNodeFindInClone(cudaGraphNode_t* pNode, cudaGraphNode_t originalNode, cudaGraph_t clonedGraph) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaGraphNodeGetType(cudaGraphNode_t node, cudaGraphNodeType* pType) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaGraphGetNodes(cudaGraph_t graph, cudaGraphNode_t* nodes, size_t* numNodes) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaGraphGetRootNodes(cudaGraph_t graph, cudaGraphNode_t* pRootNodes, size_t* pNumRootNodes) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaGraphGetEdges(cudaGraph_t graph, cudaGraphNode_t* from_, cudaGraphNode_t* to, cudaGraphEdgeData* edgeData, size_t* numEdges) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaGraphNodeGetDependencies(cudaGraphNode_t node, cudaGraphNode_t* pDependencies, cudaGraphEdgeData* edgeData, size_t* pNumDependencies) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaGraphNodeGetDependentNodes(cudaGraphNode_t node, cudaGraphNode_t* pDependentNodes, cudaGraphEdgeData* edgeData, size_t* pNumDependentNodes) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaGraphAddDependencies(cudaGraph_t graph, const cudaGraphNode_t* from_, const cudaGraphNode_t* to, const cudaGraphEdgeData* edgeData, size_t numDependencies) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaGraphRemoveDependencies(cudaGraph_t graph, const cudaGraphNode_t* from_, const cudaGraphNode_t* to, const cudaGraphEdgeData* edgeData, size_t numDependencies) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaGraphDestroyNode(cudaGraphNode_t node) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaGraphInstantiate(cudaGraphExec_t* pGraphExec, cudaGraph_t graph, unsigned long long flags) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaGraphInstantiateWithFlags(cudaGraphExec_t* pGraphExec, cudaGraph_t graph, unsigned long long flags) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaGraphInstantiateWithParams(cudaGraphExec_t* pGraphExec, cudaGraph_t graph, cudaGraphInstantiateParams* instantiateParams) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaGraphExecGetFlags(cudaGraphExec_t graphExec, unsigned long long* flags) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaGraphExecKernelNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, const cudaKernelNodeParams* pNodeParams) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaGraphExecMemcpyNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, const cudaMemcpy3DParms* pNodeParams) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaGraphExecMemcpyNodeSetParams1D(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, void* dst, const void* src, size_t count, cudaMemcpyKind kind) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaGraphExecMemsetNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, const cudaMemsetParams* pNodeParams) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaGraphExecHostNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, const cudaHostNodeParams* pNodeParams) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaGraphExecChildGraphNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, cudaGraph_t childGraph) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaGraphExecEventRecordNodeSetEvent(cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, cudaEvent_t event) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaGraphExecEventWaitNodeSetEvent(cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, cudaEvent_t event) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaGraphExecExternalSemaphoresSignalNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, const cudaExternalSemaphoreSignalNodeParams* nodeParams) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaGraphExecExternalSemaphoresWaitNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, const cudaExternalSemaphoreWaitNodeParams* nodeParams) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaGraphNodeSetEnabled(cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, unsigned int isEnabled) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaGraphNodeGetEnabled(cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, unsigned int* isEnabled) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaGraphExecUpdate(cudaGraphExec_t hGraphExec, cudaGraph_t hGraph, cudaGraphExecUpdateResultInfo* resultInfo) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaGraphUpload(cudaGraphExec_t graphExec, cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaGraphLaunch(cudaGraphExec_t graphExec, cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaGraphExecDestroy(cudaGraphExec_t graphExec) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaGraphDestroy(cudaGraph_t graph) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaGraphDebugDotPrint(cudaGraph_t graph, const char* path, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaUserObjectCreate(cudaUserObject_t* object_out, void* ptr, cudaHostFn_t destroy, unsigned int initialRefcount, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaUserObjectRetain(cudaUserObject_t object, unsigned int count) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaUserObjectRelease(cudaUserObject_t object, unsigned int count) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaGraphRetainUserObject(cudaGraph_t graph, cudaUserObject_t object, unsigned int count, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaGraphReleaseUserObject(cudaGraph_t graph, cudaUserObject_t object, unsigned int count) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaGraphAddNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, const cudaGraphEdgeData* dependencyData, size_t numDependencies, cudaGraphNodeParams* nodeParams) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaGraphNodeSetParams(cudaGraphNode_t node, cudaGraphNodeParams* nodeParams) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaGraphExecNodeSetParams(cudaGraphExec_t graphExec, cudaGraphNode_t node, cudaGraphNodeParams* nodeParams) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaGraphConditionalHandleCreate(cudaGraphConditionalHandle* pHandle_out, cudaGraph_t graph, unsigned int defaultLaunchValue, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaGetDriverEntryPoint(const char* symbol, void** funcPtr, unsigned long long flags, cudaDriverEntryPointQueryResult* driverStatus) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaGetDriverEntryPointByVersion(const char* symbol, void** funcPtr, unsigned int cudaVersion, unsigned long long flags, cudaDriverEntryPointQueryResult* driverStatus) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaLibraryLoadData(cudaLibrary_t* library, const void* code, cudaJitOption* jitOptions, void** jitOptionsValues, unsigned int numJitOptions, cudaLibraryOption* libraryOptions, void** libraryOptionValues, unsigned int numLibraryOptions) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaLibraryLoadFromFile(cudaLibrary_t* library, const char* fileName, cudaJitOption* jitOptions, void** jitOptionsValues, unsigned int numJitOptions, cudaLibraryOption* libraryOptions, void** libraryOptionValues, unsigned int numLibraryOptions) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaLibraryUnload(cudaLibrary_t library) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaLibraryGetKernel(cudaKernel_t* pKernel, cudaLibrary_t library, const char* name) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaLibraryGetGlobal(void** dptr, size_t* bytes, cudaLibrary_t library, const char* name) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaLibraryGetManaged(void** dptr, size_t* bytes, cudaLibrary_t library, const char* name) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaLibraryGetUnifiedFunction(void** fptr, cudaLibrary_t library, const char* symbol) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaLibraryGetKernelCount(unsigned int* count, cudaLibrary_t lib) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaLibraryEnumerateKernels(cudaKernel_t* kernels, unsigned int numKernels, cudaLibrary_t lib) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaKernelSetAttributeForDevice(cudaKernel_t kernel, cudaFuncAttribute attr, int value, int device) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaGetExportTable(const void** ppExportTable, const cudaUUID_t* pExportTableId) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaGetKernel(cudaKernel_t* kernelPtr, const void* entryFuncAddr) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaProfilerStart() except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaProfilerStop() except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaGetDeviceProperties(cudaDeviceProp* prop, int device) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaDeviceGetHostAtomicCapabilities(unsigned int* capabilities, const cudaAtomicOperation* operations, unsigned int count, int device) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaDeviceGetP2PAtomicCapabilities(unsigned int* capabilities, const cudaAtomicOperation* operations, unsigned int count, int srcDevice, int dstDevice) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaStreamGetCaptureInfo(cudaStream_t stream, cudaStreamCaptureStatus* captureStatus_out, unsigned long long* id_out, cudaGraph_t* graph_out, const cudaGraphNode_t** dependencies_out, const cudaGraphEdgeData** edgeData_out, size_t* numDependencies_out) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaSignalExternalSemaphoresAsync(const cudaExternalSemaphore_t* extSemArray, const cudaExternalSemaphoreSignalParams* paramsArray, unsigned int numExtSems, cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaWaitExternalSemaphoresAsync(const cudaExternalSemaphore_t* extSemArray, const cudaExternalSemaphoreWaitParams* paramsArray, unsigned int numExtSems, cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaMemPrefetchBatchAsync(void** dptrs, size_t* sizes, size_t count, cudaMemLocation* prefetchLocs, size_t* prefetchLocIdxs, size_t numPrefetchLocs, unsigned long long flags, cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaMemDiscardBatchAsync(void** dptrs, size_t* sizes, size_t count, unsigned long long flags, cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaMemDiscardAndPrefetchBatchAsync(void** dptrs, size_t* sizes, size_t count, cudaMemLocation* prefetchLocs, size_t* prefetchLocIdxs, size_t numPrefetchLocs, unsigned long long flags, cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaMemGetDefaultMemPool(cudaMemPool_t* memPool, cudaMemLocation* location, cudaMemAllocationType type) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaMemGetMemPool(cudaMemPool_t* memPool, cudaMemLocation* location, cudaMemAllocationType type) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaMemSetMemPool(cudaMemLocation* location, cudaMemAllocationType type, cudaMemPool_t memPool) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaLogsRegisterCallback(cudaLogsCallback_t callbackFunc, void* userData, cudaLogsCallbackHandle* callback_out) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaLogsUnregisterCallback(cudaLogsCallbackHandle callback) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaLogsCurrent(cudaLogIterator* iterator_out, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaLogsDumpToFile(cudaLogIterator* iterator, const char* pathToFile, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaLogsDumpToMemory(cudaLogIterator* iterator, char* buffer, size_t* size, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaGraphNodeGetContainingGraph(cudaGraphNode_t hNode, cudaGraph_t* phGraph) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaGraphNodeGetLocalId(cudaGraphNode_t hNode, unsigned int* nodeId) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaGraphNodeGetToolsId(cudaGraphNode_t hNode, unsigned long long* toolsNodeId) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaGraphGetId(cudaGraph_t hGraph, unsigned int* graphID) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaGraphExecGetId(cudaGraphExec_t hGraphExec, unsigned int* graphID) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaGraphConditionalHandleCreate_v2(cudaGraphConditionalHandle* pHandle_out, cudaGraph_t graph, cudaExecutionContext_t ctx, unsigned int defaultLaunchValue, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaDeviceGetDevResource(int device, cudaDevResource* resource, cudaDevResourceType type) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaDevSmResourceSplitByCount(cudaDevResource* result, unsigned int* nbGroups, const cudaDevResource* input, cudaDevResource* remaining, unsigned int flags, unsigned int minCount) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaDevSmResourceSplit(cudaDevResource* result, unsigned int nbGroups, const cudaDevResource* input, cudaDevResource* remainder, unsigned int flags, cudaDevSmResourceGroupParams* groupParams) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaDevResourceGenerateDesc(cudaDevResourceDesc_t* phDesc, cudaDevResource* resources, unsigned int nbResources) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaGreenCtxCreate(cudaExecutionContext_t* phCtx, cudaDevResourceDesc_t desc, int device, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaExecutionCtxDestroy(cudaExecutionContext_t ctx) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaExecutionCtxGetDevResource(cudaExecutionContext_t ctx, cudaDevResource* resource, cudaDevResourceType type) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaExecutionCtxGetDevice(int* device, cudaExecutionContext_t ctx) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaExecutionCtxGetId(cudaExecutionContext_t ctx, unsigned long long* ctxId) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaExecutionCtxStreamCreate(cudaStream_t* phStream, cudaExecutionContext_t ctx, unsigned int flags, int priority) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaExecutionCtxSynchronize(cudaExecutionContext_t ctx) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaStreamGetDevResource(cudaStream_t hStream, cudaDevResource* resource, cudaDevResourceType type) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaExecutionCtxRecordEvent(cudaExecutionContext_t ctx, cudaEvent_t event) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaExecutionCtxWaitEvent(cudaExecutionContext_t ctx, cudaEvent_t event) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaDeviceGetExecutionCtx(cudaExecutionContext_t* ctx, int device) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaFuncGetParamCount(const void* func, size_t* paramCount) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaLaunchHostFunc_v2(cudaStream_t stream, cudaHostFn_t fn, void* userData, unsigned int syncMode) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaMemcpyWithAttributesAsync(void* dst, const void* src, size_t size, cudaMemcpyAttributes* attr, cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaMemcpy3DWithAttributesAsync(cudaMemcpy3DBatchOp* op, unsigned long long flags, cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaGraphNodeGetParams(cudaGraphNode_t node, cudaGraphNodeParams* nodeParams) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t _cudaStreamBeginRecaptureToGraph(cudaStream_t stream, cudaStreamCaptureMode mode, cudaGraph_t graph, cudaGraphRecaptureCallbackData* callbackData) except ?cudaErrorCallRequiresNewerDriver nogil
