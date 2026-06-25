# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

# This code was automatically generated with version 13.3.0, generator version 0.3.1.dev1779+ga8cc71818.d20260623. Do not modify it directly.
cdef extern from "cuda_runtime_api.h":



    cudaError_t cudaDeviceReset() nogil




    cudaError_t cudaDeviceSynchronize() nogil




    cudaError_t cudaDeviceSetLimit(cudaLimit limit, size_t value) nogil




    cudaError_t cudaDeviceGetLimit(size_t* pValue, cudaLimit limit) nogil




    cudaError_t cudaDeviceGetTexture1DLinearMaxWidth(size_t* maxWidthInElements, const cudaChannelFormatDesc* fmtDesc, int device) nogil




    cudaError_t cudaDeviceGetCacheConfig(cudaFuncCache* pCacheConfig) nogil




    cudaError_t cudaDeviceGetStreamPriorityRange(int* leastPriority, int* greatestPriority) nogil




    cudaError_t cudaDeviceSetCacheConfig(cudaFuncCache cacheConfig) nogil




    cudaError_t cudaDeviceGetByPCIBusId(int* device, const char* pciBusId) nogil




    cudaError_t cudaDeviceGetPCIBusId(char* pciBusId, int length, int device) nogil




    cudaError_t cudaIpcGetEventHandle(cudaIpcEventHandle_t* handle, cudaEvent_t event) nogil




    cudaError_t cudaIpcOpenEventHandle(cudaEvent_t* event, cudaIpcEventHandle_t handle) nogil




    cudaError_t cudaIpcGetMemHandle(cudaIpcMemHandle_t* handle, void* devPtr) nogil




    cudaError_t cudaIpcOpenMemHandle(void** devPtr, cudaIpcMemHandle_t handle, unsigned int flags) nogil




    cudaError_t cudaIpcCloseMemHandle(void* devPtr) nogil




    cudaError_t cudaDeviceFlushGPUDirectRDMAWrites(cudaFlushGPUDirectRDMAWritesTarget target, cudaFlushGPUDirectRDMAWritesScope scope) nogil




    cudaError_t cudaDeviceRegisterAsyncNotification(int device, cudaAsyncCallback callbackFunc, void* userData, cudaAsyncCallbackHandle_t* callback) nogil




    cudaError_t cudaDeviceUnregisterAsyncNotification(int device, cudaAsyncCallbackHandle_t callback) nogil




    cudaError_t cudaDeviceGetSharedMemConfig(cudaSharedMemConfig* pConfig) nogil




    cudaError_t cudaDeviceSetSharedMemConfig(cudaSharedMemConfig config) nogil




    cudaError_t cudaGetLastError() nogil




    cudaError_t cudaPeekAtLastError() nogil




    const char* cudaGetErrorName(cudaError_t error) nogil




    const char* cudaGetErrorString(cudaError_t error) nogil




    cudaError_t cudaGetDeviceCount(int* count) nogil




    cudaError_t cudaGetDeviceProperties(cudaDeviceProp* prop, int device) nogil




    cudaError_t cudaDeviceGetAttribute(int* value, cudaDeviceAttr attr, int device) nogil




    cudaError_t cudaDeviceGetHostAtomicCapabilities(unsigned int* capabilities, const cudaAtomicOperation* operations, unsigned int count, int device) nogil




    cudaError_t cudaDeviceGetDefaultMemPool(cudaMemPool_t* memPool, int device) nogil




    cudaError_t cudaDeviceSetMemPool(int device, cudaMemPool_t memPool) nogil




    cudaError_t cudaDeviceGetMemPool(cudaMemPool_t* memPool, int device) nogil




    cudaError_t cudaDeviceGetNvSciSyncAttributes(void* nvSciSyncAttrList, int device, int flags) nogil




    cudaError_t cudaDeviceGetP2PAttribute(int* value, cudaDeviceP2PAttr attr, int srcDevice, int dstDevice) nogil




    cudaError_t cudaDeviceGetP2PAtomicCapabilities(unsigned int* capabilities, const cudaAtomicOperation* operations, unsigned int count, int srcDevice, int dstDevice) nogil




    cudaError_t cudaChooseDevice(int* device, const cudaDeviceProp* prop) nogil




    cudaError_t cudaInitDevice(int device, unsigned int deviceFlags, unsigned int flags) nogil




    cudaError_t cudaSetDevice(int device) nogil




    cudaError_t cudaGetDevice(int* device) nogil




    cudaError_t cudaSetDeviceFlags(unsigned int flags) nogil




    cudaError_t cudaGetDeviceFlags(unsigned int* flags) nogil




    cudaError_t cudaStreamCreate(cudaStream_t* pStream) nogil




    cudaError_t cudaStreamCreateWithFlags(cudaStream_t* pStream, unsigned int flags) nogil




    cudaError_t cudaStreamCreateWithPriority(cudaStream_t* pStream, unsigned int flags, int priority) nogil




    cudaError_t cudaStreamGetPriority(cudaStream_t hStream, int* priority) nogil




    cudaError_t cudaStreamGetFlags(cudaStream_t hStream, unsigned int* flags) nogil




    cudaError_t cudaStreamGetId(cudaStream_t hStream, unsigned long long* streamId) nogil




    cudaError_t cudaStreamGetDevice(cudaStream_t hStream, int* device) nogil




    cudaError_t cudaCtxResetPersistingL2Cache() nogil




    cudaError_t cudaStreamCopyAttributes(cudaStream_t dst, cudaStream_t src) nogil




    cudaError_t cudaStreamGetAttribute(cudaStream_t hStream, cudaStreamAttrID attr, cudaStreamAttrValue* value_out) nogil




    cudaError_t cudaStreamSetAttribute(cudaStream_t hStream, cudaStreamAttrID attr, const cudaStreamAttrValue* value) nogil




    cudaError_t cudaStreamDestroy(cudaStream_t stream) nogil




    cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned int flags) nogil




    cudaError_t cudaStreamAddCallback(cudaStream_t stream, cudaStreamCallback_t callback, void* userData, unsigned int flags) nogil




    cudaError_t cudaStreamSynchronize(cudaStream_t stream) nogil




    cudaError_t cudaStreamQuery(cudaStream_t stream) nogil




    cudaError_t cudaStreamAttachMemAsync(cudaStream_t stream, void* devPtr, size_t length, unsigned int flags) nogil




    cudaError_t cudaStreamBeginCapture(cudaStream_t stream, cudaStreamCaptureMode mode) nogil




    cudaError_t cudaStreamBeginRecaptureToGraph(cudaStream_t stream, cudaStreamCaptureMode mode, cudaGraph_t graph, cudaGraphRecaptureCallbackData* callbackData) nogil




    cudaError_t cudaStreamBeginCaptureToGraph(cudaStream_t stream, cudaGraph_t graph, const cudaGraphNode_t* dependencies, const cudaGraphEdgeData* dependencyData, size_t numDependencies, cudaStreamCaptureMode mode) nogil




    cudaError_t cudaThreadExchangeStreamCaptureMode(cudaStreamCaptureMode* mode) nogil




    cudaError_t cudaStreamEndCapture(cudaStream_t stream, cudaGraph_t* pGraph) nogil




    cudaError_t cudaStreamIsCapturing(cudaStream_t stream, cudaStreamCaptureStatus* pCaptureStatus) nogil




    cudaError_t cudaStreamGetCaptureInfo(cudaStream_t stream, cudaStreamCaptureStatus* captureStatus_out, unsigned long long* id_out, cudaGraph_t* graph_out, const cudaGraphNode_t** dependencies_out, const cudaGraphEdgeData** edgeData_out, size_t* numDependencies_out) nogil




    cudaError_t cudaStreamUpdateCaptureDependencies(cudaStream_t stream, cudaGraphNode_t* dependencies, const cudaGraphEdgeData* dependencyData, size_t numDependencies, unsigned int flags) nogil




    cudaError_t cudaEventCreate(cudaEvent_t* event) nogil




    cudaError_t cudaEventCreateWithFlags(cudaEvent_t* event, unsigned int flags) nogil




    cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream) nogil




    cudaError_t cudaEventRecordWithFlags(cudaEvent_t event, cudaStream_t stream, unsigned int flags) nogil




    cudaError_t cudaEventQuery(cudaEvent_t event) nogil




    cudaError_t cudaEventSynchronize(cudaEvent_t event) nogil




    cudaError_t cudaEventDestroy(cudaEvent_t event) nogil




    cudaError_t cudaEventElapsedTime(float* ms, cudaEvent_t start, cudaEvent_t end) nogil




    cudaError_t cudaImportExternalMemory(cudaExternalMemory_t* extMem_out, const cudaExternalMemoryHandleDesc* memHandleDesc) nogil




    cudaError_t cudaExternalMemoryGetMappedBuffer(void** devPtr, cudaExternalMemory_t extMem, const cudaExternalMemoryBufferDesc* bufferDesc) nogil




    cudaError_t cudaExternalMemoryGetMappedMipmappedArray(cudaMipmappedArray_t* mipmap, cudaExternalMemory_t extMem, const cudaExternalMemoryMipmappedArrayDesc* mipmapDesc) nogil




    cudaError_t cudaDestroyExternalMemory(cudaExternalMemory_t extMem) nogil




    cudaError_t cudaImportExternalSemaphore(cudaExternalSemaphore_t* extSem_out, const cudaExternalSemaphoreHandleDesc* semHandleDesc) nogil




    cudaError_t cudaSignalExternalSemaphoresAsync(const cudaExternalSemaphore_t* extSemArray, const cudaExternalSemaphoreSignalParams* paramsArray, unsigned int numExtSems, cudaStream_t stream) nogil




    cudaError_t cudaWaitExternalSemaphoresAsync(const cudaExternalSemaphore_t* extSemArray, const cudaExternalSemaphoreWaitParams* paramsArray, unsigned int numExtSems, cudaStream_t stream) nogil




    cudaError_t cudaDestroyExternalSemaphore(cudaExternalSemaphore_t extSem) nogil




    cudaError_t cudaFuncSetCacheConfig(const void* func, cudaFuncCache cacheConfig) nogil




    cudaError_t cudaFuncGetAttributes(cudaFuncAttributes* attr, const void* func) nogil




    cudaError_t cudaFuncSetAttribute(const void* func, cudaFuncAttribute attr, int value) nogil




    cudaError_t cudaFuncGetParamCount(const void* func, size_t* paramCount) nogil




    cudaError_t cudaLaunchHostFunc(cudaStream_t stream, cudaHostFn_t fn, void* userData) nogil




    cudaError_t cudaLaunchHostFunc_v2(cudaStream_t stream, cudaHostFn_t fn, void* userData, unsigned int syncMode) nogil




    cudaError_t cudaFuncSetSharedMemConfig(const void* func, cudaSharedMemConfig config) nogil




    cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(int* numBlocks, const void* func, int blockSize, size_t dynamicSMemSize) nogil




    cudaError_t cudaOccupancyAvailableDynamicSMemPerBlock(size_t* dynamicSmemSize, const void* func, int numBlocks, int blockSize) nogil




    cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int* numBlocks, const void* func, int blockSize, size_t dynamicSMemSize, unsigned int flags) nogil




    cudaError_t cudaMallocManaged(void** devPtr, size_t size, unsigned int flags) nogil




    cudaError_t cudaMalloc(void** devPtr, size_t size) nogil




    cudaError_t cudaMallocHost(void** ptr, size_t size) nogil




    cudaError_t cudaMallocPitch(void** devPtr, size_t* pitch, size_t width, size_t height) nogil




    cudaError_t cudaMallocArray(cudaArray_t* array, const cudaChannelFormatDesc* desc, size_t width, size_t height, unsigned int flags) nogil




    cudaError_t cudaFree(void* devPtr) nogil




    cudaError_t cudaFreeHost(void* ptr) nogil




    cudaError_t cudaFreeArray(cudaArray_t array) nogil




    cudaError_t cudaFreeMipmappedArray(cudaMipmappedArray_t mipmappedArray) nogil




    cudaError_t cudaHostAlloc(void** pHost, size_t size, unsigned int flags) nogil




    cudaError_t cudaHostRegister(void* ptr, size_t size, unsigned int flags) nogil




    cudaError_t cudaHostUnregister(void* ptr) nogil




    cudaError_t cudaHostGetDevicePointer(void** pDevice, void* pHost, unsigned int flags) nogil




    cudaError_t cudaHostGetFlags(unsigned int* pFlags, void* pHost) nogil




    cudaError_t cudaMalloc3D(cudaPitchedPtr* pitchedDevPtr, cudaExtent extent) nogil




    cudaError_t cudaMalloc3DArray(cudaArray_t* array, const cudaChannelFormatDesc* desc, cudaExtent extent, unsigned int flags) nogil




    cudaError_t cudaMallocMipmappedArray(cudaMipmappedArray_t* mipmappedArray, const cudaChannelFormatDesc* desc, cudaExtent extent, unsigned int numLevels, unsigned int flags) nogil




    cudaError_t cudaGetMipmappedArrayLevel(cudaArray_t* levelArray, cudaMipmappedArray_const_t mipmappedArray, unsigned int level) nogil




    cudaError_t cudaMemcpy3D(const cudaMemcpy3DParms* p) nogil




    cudaError_t cudaMemcpy3DPeer(const cudaMemcpy3DPeerParms* p) nogil




    cudaError_t cudaMemcpy3DAsync(const cudaMemcpy3DParms* p, cudaStream_t stream) nogil




    cudaError_t cudaMemcpy3DPeerAsync(const cudaMemcpy3DPeerParms* p, cudaStream_t stream) nogil




    cudaError_t cudaMemGetInfo(size_t* free, size_t* total) nogil




    cudaError_t cudaArrayGetInfo(cudaChannelFormatDesc* desc, cudaExtent* extent, unsigned int* flags, cudaArray_t array) nogil




    cudaError_t cudaArrayGetPlane(cudaArray_t* pPlaneArray, cudaArray_t hArray, unsigned int planeIdx) nogil




    cudaError_t cudaArrayGetMemoryRequirements(cudaArrayMemoryRequirements* memoryRequirements, cudaArray_t array, int device) nogil




    cudaError_t cudaMipmappedArrayGetMemoryRequirements(cudaArrayMemoryRequirements* memoryRequirements, cudaMipmappedArray_t mipmap, int device) nogil




    cudaError_t cudaArrayGetSparseProperties(cudaArraySparseProperties* sparseProperties, cudaArray_t array) nogil




    cudaError_t cudaMipmappedArrayGetSparseProperties(cudaArraySparseProperties* sparseProperties, cudaMipmappedArray_t mipmap) nogil




    cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind) nogil




    cudaError_t cudaMemcpyPeer(void* dst, int dstDevice, const void* src, int srcDevice, size_t count) nogil




    cudaError_t cudaMemcpy2D(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind) nogil




    cudaError_t cudaMemcpy2DToArray(cudaArray_t dst, size_t wOffset, size_t hOffset, const void* src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind) nogil




    cudaError_t cudaMemcpy2DFromArray(void* dst, size_t dpitch, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t width, size_t height, cudaMemcpyKind kind) nogil




    cudaError_t cudaMemcpy2DArrayToArray(cudaArray_t dst, size_t wOffsetDst, size_t hOffsetDst, cudaArray_const_t src, size_t wOffsetSrc, size_t hOffsetSrc, size_t width, size_t height, cudaMemcpyKind kind) nogil




    cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t count, cudaMemcpyKind kind, cudaStream_t stream) nogil




    cudaError_t cudaMemcpyPeerAsync(void* dst, int dstDevice, const void* src, int srcDevice, size_t count, cudaStream_t stream) nogil




    cudaError_t cudaMemcpyBatchAsync(const void** dsts, const void** srcs, const size_t* sizes, size_t count, cudaMemcpyAttributes* attrs, size_t* attrsIdxs, size_t numAttrs, cudaStream_t stream) nogil




    cudaError_t cudaMemcpy3DBatchAsync(size_t numOps, cudaMemcpy3DBatchOp* opList, unsigned long long flags, cudaStream_t stream) nogil




    cudaError_t cudaMemcpyWithAttributesAsync(void* dst, const void* src, size_t size, cudaMemcpyAttributes* attr, cudaStream_t stream) nogil




    cudaError_t cudaMemcpy3DWithAttributesAsync(cudaMemcpy3DBatchOp* op, unsigned long long flags, cudaStream_t stream) nogil




    cudaError_t cudaMemcpy2DAsync(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream) nogil




    cudaError_t cudaMemcpy2DToArrayAsync(cudaArray_t dst, size_t wOffset, size_t hOffset, const void* src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream) nogil




    cudaError_t cudaMemcpy2DFromArrayAsync(void* dst, size_t dpitch, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream) nogil




    cudaError_t cudaMemset(void* devPtr, int value, size_t count) nogil




    cudaError_t cudaMemset2D(void* devPtr, size_t pitch, int value, size_t width, size_t height) nogil




    cudaError_t cudaMemset3D(cudaPitchedPtr pitchedDevPtr, int value, cudaExtent extent) nogil




    cudaError_t cudaMemsetAsync(void* devPtr, int value, size_t count, cudaStream_t stream) nogil




    cudaError_t cudaMemset2DAsync(void* devPtr, size_t pitch, int value, size_t width, size_t height, cudaStream_t stream) nogil




    cudaError_t cudaMemset3DAsync(cudaPitchedPtr pitchedDevPtr, int value, cudaExtent extent, cudaStream_t stream) nogil




    cudaError_t cudaMemPrefetchAsync(const void* devPtr, size_t count, cudaMemLocation location, unsigned int flags, cudaStream_t stream) nogil




    cudaError_t cudaMemPrefetchBatchAsync(void** dptrs, size_t* sizes, size_t count, cudaMemLocation* prefetchLocs, size_t* prefetchLocIdxs, size_t numPrefetchLocs, unsigned long long flags, cudaStream_t stream) nogil




    cudaError_t cudaMemDiscardBatchAsync(void** dptrs, size_t* sizes, size_t count, unsigned long long flags, cudaStream_t stream) nogil




    cudaError_t cudaMemDiscardAndPrefetchBatchAsync(void** dptrs, size_t* sizes, size_t count, cudaMemLocation* prefetchLocs, size_t* prefetchLocIdxs, size_t numPrefetchLocs, unsigned long long flags, cudaStream_t stream) nogil




    cudaError_t cudaMemAdvise(const void* devPtr, size_t count, cudaMemoryAdvise advice, cudaMemLocation location) nogil




    cudaError_t cudaMemRangeGetAttribute(void* data, size_t dataSize, cudaMemRangeAttribute attribute, const void* devPtr, size_t count) nogil




    cudaError_t cudaMemRangeGetAttributes(void** data, size_t* dataSizes, cudaMemRangeAttribute* attributes, size_t numAttributes, const void* devPtr, size_t count) nogil




    cudaError_t cudaMemcpyToArray(cudaArray_t dst, size_t wOffset, size_t hOffset, const void* src, size_t count, cudaMemcpyKind kind) nogil




    cudaError_t cudaMemcpyFromArray(void* dst, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t count, cudaMemcpyKind kind) nogil




    cudaError_t cudaMemcpyArrayToArray(cudaArray_t dst, size_t wOffsetDst, size_t hOffsetDst, cudaArray_const_t src, size_t wOffsetSrc, size_t hOffsetSrc, size_t count, cudaMemcpyKind kind) nogil




    cudaError_t cudaMemcpyToArrayAsync(cudaArray_t dst, size_t wOffset, size_t hOffset, const void* src, size_t count, cudaMemcpyKind kind, cudaStream_t stream) nogil




    cudaError_t cudaMemcpyFromArrayAsync(void* dst, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t count, cudaMemcpyKind kind, cudaStream_t stream) nogil




    cudaError_t cudaMallocAsync(void** devPtr, size_t size, cudaStream_t hStream) nogil




    cudaError_t cudaFreeAsync(void* devPtr, cudaStream_t hStream) nogil




    cudaError_t cudaMemPoolTrimTo(cudaMemPool_t memPool, size_t minBytesToKeep) nogil




    cudaError_t cudaMemPoolSetAttribute(cudaMemPool_t memPool, cudaMemPoolAttr attr, void* value) nogil




    cudaError_t cudaMemPoolGetAttribute(cudaMemPool_t memPool, cudaMemPoolAttr attr, void* value) nogil




    cudaError_t cudaMemPoolSetAccess(cudaMemPool_t memPool, const cudaMemAccessDesc* descList, size_t count) nogil




    cudaError_t cudaMemPoolGetAccess(cudaMemAccessFlags* flags, cudaMemPool_t memPool, cudaMemLocation* location) nogil




    cudaError_t cudaMemPoolCreate(cudaMemPool_t* memPool, const cudaMemPoolProps* poolProps) nogil




    cudaError_t cudaMemPoolDestroy(cudaMemPool_t memPool) nogil




    cudaError_t cudaMemGetDefaultMemPool(cudaMemPool_t* memPool, cudaMemLocation* location, cudaMemAllocationType typename) nogil




    cudaError_t cudaMemGetMemPool(cudaMemPool_t* memPool, cudaMemLocation* location, cudaMemAllocationType typename) nogil




    cudaError_t cudaMemSetMemPool(cudaMemLocation* location, cudaMemAllocationType typename, cudaMemPool_t memPool) nogil




    cudaError_t cudaMallocFromPoolAsync(void** ptr, size_t size, cudaMemPool_t memPool, cudaStream_t stream) nogil




    cudaError_t cudaMemPoolExportToShareableHandle(void* shareableHandle, cudaMemPool_t memPool, cudaMemAllocationHandleType handleType, unsigned int flags) nogil




    cudaError_t cudaMemPoolImportFromShareableHandle(cudaMemPool_t* memPool, void* shareableHandle, cudaMemAllocationHandleType handleType, unsigned int flags) nogil




    cudaError_t cudaMemPoolExportPointer(cudaMemPoolPtrExportData* exportData, void* ptr) nogil




    cudaError_t cudaMemPoolImportPointer(void** ptr, cudaMemPool_t memPool, cudaMemPoolPtrExportData* exportData) nogil




    cudaError_t cudaPointerGetAttributes(cudaPointerAttributes* attributes, const void* ptr) nogil




    cudaError_t cudaDeviceCanAccessPeer(int* canAccessPeer, int device, int peerDevice) nogil




    cudaError_t cudaDeviceEnablePeerAccess(int peerDevice, unsigned int flags) nogil




    cudaError_t cudaDeviceDisablePeerAccess(int peerDevice) nogil




    cudaError_t cudaGraphicsUnregisterResource(cudaGraphicsResource_t resource) nogil




    cudaError_t cudaGraphicsResourceSetMapFlags(cudaGraphicsResource_t resource, unsigned int flags) nogil




    cudaError_t cudaGraphicsMapResources(int count, cudaGraphicsResource_t* resources, cudaStream_t stream) nogil




    cudaError_t cudaGraphicsUnmapResources(int count, cudaGraphicsResource_t* resources, cudaStream_t stream) nogil




    cudaError_t cudaGraphicsResourceGetMappedPointer(void** devPtr, size_t* size, cudaGraphicsResource_t resource) nogil




    cudaError_t cudaGraphicsSubResourceGetMappedArray(cudaArray_t* array, cudaGraphicsResource_t resource, unsigned int arrayIndex, unsigned int mipLevel) nogil




    cudaError_t cudaGraphicsResourceGetMappedMipmappedArray(cudaMipmappedArray_t* mipmappedArray, cudaGraphicsResource_t resource) nogil




    cudaError_t cudaGetChannelDesc(cudaChannelFormatDesc* desc, cudaArray_const_t array) nogil




    cudaChannelFormatDesc cudaCreateChannelDesc(int x, int y, int z, int w, cudaChannelFormatKind f) nogil




    cudaError_t cudaCreateTextureObject(cudaTextureObject_t* pTexObject, const cudaResourceDesc* pResDesc, const cudaTextureDesc* pTexDesc, const cudaResourceViewDesc* pResViewDesc) nogil




    cudaError_t cudaDestroyTextureObject(cudaTextureObject_t texObject) nogil




    cudaError_t cudaGetTextureObjectResourceDesc(cudaResourceDesc* pResDesc, cudaTextureObject_t texObject) nogil




    cudaError_t cudaGetTextureObjectTextureDesc(cudaTextureDesc* pTexDesc, cudaTextureObject_t texObject) nogil




    cudaError_t cudaGetTextureObjectResourceViewDesc(cudaResourceViewDesc* pResViewDesc, cudaTextureObject_t texObject) nogil




    cudaError_t cudaCreateSurfaceObject(cudaSurfaceObject_t* pSurfObject, const cudaResourceDesc* pResDesc) nogil




    cudaError_t cudaDestroySurfaceObject(cudaSurfaceObject_t surfObject) nogil




    cudaError_t cudaGetSurfaceObjectResourceDesc(cudaResourceDesc* pResDesc, cudaSurfaceObject_t surfObject) nogil




    cudaError_t cudaDriverGetVersion(int* driverVersion) nogil




    cudaError_t cudaRuntimeGetVersion(int* runtimeVersion) nogil




    cudaError_t cudaLogsRegisterCallback(cudaLogsCallback_t callbackFunc, void* userData, cudaLogsCallbackHandle* callback_out) nogil




    cudaError_t cudaLogsUnregisterCallback(cudaLogsCallbackHandle callback) nogil




    cudaError_t cudaLogsCurrent(cudaLogIterator* iterator_out, unsigned int flags) nogil




    cudaError_t cudaLogsDumpToFile(cudaLogIterator* iterator, const char* pathToFile, unsigned int flags) nogil




    cudaError_t cudaLogsDumpToMemory(cudaLogIterator* iterator, char* buffer, size_t* size, unsigned int flags) nogil




    cudaError_t cudaGraphCreate(cudaGraph_t* pGraph, unsigned int flags) nogil




    cudaError_t cudaGraphAddKernelNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, const cudaKernelNodeParams* pNodeParams) nogil




    cudaError_t cudaGraphKernelNodeGetParams(cudaGraphNode_t node, cudaKernelNodeParams* pNodeParams) nogil




    cudaError_t cudaGraphKernelNodeSetParams(cudaGraphNode_t node, const cudaKernelNodeParams* pNodeParams) nogil




    cudaError_t cudaGraphKernelNodeCopyAttributes(cudaGraphNode_t hDst, cudaGraphNode_t hSrc) nogil




    cudaError_t cudaGraphKernelNodeGetAttribute(cudaGraphNode_t hNode, cudaKernelNodeAttrID attr, cudaKernelNodeAttrValue* value_out) nogil




    cudaError_t cudaGraphKernelNodeSetAttribute(cudaGraphNode_t hNode, cudaKernelNodeAttrID attr, const cudaKernelNodeAttrValue* value) nogil




    cudaError_t cudaGraphAddMemcpyNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, const cudaMemcpy3DParms* pCopyParams) nogil




    cudaError_t cudaGraphAddMemcpyNode1D(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, void* dst, const void* src, size_t count, cudaMemcpyKind kind) nogil




    cudaError_t cudaGraphMemcpyNodeGetParams(cudaGraphNode_t node, cudaMemcpy3DParms* pNodeParams) nogil




    cudaError_t cudaGraphMemcpyNodeSetParams(cudaGraphNode_t node, const cudaMemcpy3DParms* pNodeParams) nogil




    cudaError_t cudaGraphMemcpyNodeSetParams1D(cudaGraphNode_t node, void* dst, const void* src, size_t count, cudaMemcpyKind kind) nogil




    cudaError_t cudaGraphAddMemsetNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, const cudaMemsetParams* pMemsetParams) nogil




    cudaError_t cudaGraphMemsetNodeGetParams(cudaGraphNode_t node, cudaMemsetParams* pNodeParams) nogil




    cudaError_t cudaGraphMemsetNodeSetParams(cudaGraphNode_t node, const cudaMemsetParams* pNodeParams) nogil




    cudaError_t cudaGraphAddHostNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, const cudaHostNodeParams* pNodeParams) nogil




    cudaError_t cudaGraphHostNodeGetParams(cudaGraphNode_t node, cudaHostNodeParams* pNodeParams) nogil




    cudaError_t cudaGraphHostNodeSetParams(cudaGraphNode_t node, const cudaHostNodeParams* pNodeParams) nogil




    cudaError_t cudaGraphAddChildGraphNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, cudaGraph_t childGraph) nogil




    cudaError_t cudaGraphChildGraphNodeGetGraph(cudaGraphNode_t node, cudaGraph_t* pGraph) nogil




    cudaError_t cudaGraphAddEmptyNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies) nogil




    cudaError_t cudaGraphAddEventRecordNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, cudaEvent_t event) nogil




    cudaError_t cudaGraphEventRecordNodeGetEvent(cudaGraphNode_t node, cudaEvent_t* event_out) nogil




    cudaError_t cudaGraphEventRecordNodeSetEvent(cudaGraphNode_t node, cudaEvent_t event) nogil




    cudaError_t cudaGraphAddEventWaitNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, cudaEvent_t event) nogil




    cudaError_t cudaGraphEventWaitNodeGetEvent(cudaGraphNode_t node, cudaEvent_t* event_out) nogil




    cudaError_t cudaGraphEventWaitNodeSetEvent(cudaGraphNode_t node, cudaEvent_t event) nogil




    cudaError_t cudaGraphAddExternalSemaphoresSignalNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, const cudaExternalSemaphoreSignalNodeParams* nodeParams) nogil




    cudaError_t cudaGraphExternalSemaphoresSignalNodeGetParams(cudaGraphNode_t hNode, cudaExternalSemaphoreSignalNodeParams* params_out) nogil




    cudaError_t cudaGraphExternalSemaphoresSignalNodeSetParams(cudaGraphNode_t hNode, const cudaExternalSemaphoreSignalNodeParams* nodeParams) nogil




    cudaError_t cudaGraphAddExternalSemaphoresWaitNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, const cudaExternalSemaphoreWaitNodeParams* nodeParams) nogil




    cudaError_t cudaGraphExternalSemaphoresWaitNodeGetParams(cudaGraphNode_t hNode, cudaExternalSemaphoreWaitNodeParams* params_out) nogil




    cudaError_t cudaGraphExternalSemaphoresWaitNodeSetParams(cudaGraphNode_t hNode, const cudaExternalSemaphoreWaitNodeParams* nodeParams) nogil




    cudaError_t cudaGraphAddMemAllocNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, cudaMemAllocNodeParams* nodeParams) nogil




    cudaError_t cudaGraphMemAllocNodeGetParams(cudaGraphNode_t node, cudaMemAllocNodeParams* params_out) nogil




    cudaError_t cudaGraphAddMemFreeNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, void* dptr) nogil




    cudaError_t cudaGraphMemFreeNodeGetParams(cudaGraphNode_t node, void* dptr_out) nogil




    cudaError_t cudaDeviceGraphMemTrim(int device) nogil




    cudaError_t cudaDeviceGetGraphMemAttribute(int device, cudaGraphMemAttributeType attr, void* value) nogil




    cudaError_t cudaDeviceSetGraphMemAttribute(int device, cudaGraphMemAttributeType attr, void* value) nogil




    cudaError_t cudaGraphClone(cudaGraph_t* pGraphClone, cudaGraph_t originalGraph) nogil




    cudaError_t cudaGraphNodeFindInClone(cudaGraphNode_t* pNode, cudaGraphNode_t originalNode, cudaGraph_t clonedGraph) nogil




    cudaError_t cudaGraphNodeGetType(cudaGraphNode_t node, cudaGraphNodeType* pType) nogil




    cudaError_t cudaGraphNodeGetContainingGraph(cudaGraphNode_t hNode, cudaGraph_t* phGraph) nogil




    cudaError_t cudaGraphNodeGetLocalId(cudaGraphNode_t hNode, unsigned int* nodeId) nogil




    cudaError_t cudaGraphNodeGetToolsId(cudaGraphNode_t hNode, unsigned long long* toolsNodeId) nogil




    cudaError_t cudaGraphGetId(cudaGraph_t hGraph, unsigned int* graphID) nogil




    cudaError_t cudaGraphExecGetId(cudaGraphExec_t hGraphExec, unsigned int* graphID) nogil




    cudaError_t cudaGraphGetNodes(cudaGraph_t graph, cudaGraphNode_t* nodes, size_t* numNodes) nogil




    cudaError_t cudaGraphGetRootNodes(cudaGraph_t graph, cudaGraphNode_t* pRootNodes, size_t* pNumRootNodes) nogil




    cudaError_t cudaGraphGetEdges(cudaGraph_t graph, cudaGraphNode_t* from_, cudaGraphNode_t* to, cudaGraphEdgeData* edgeData, size_t* numEdges) nogil




    cudaError_t cudaGraphNodeGetDependencies(cudaGraphNode_t node, cudaGraphNode_t* pDependencies, cudaGraphEdgeData* edgeData, size_t* pNumDependencies) nogil




    cudaError_t cudaGraphNodeGetDependentNodes(cudaGraphNode_t node, cudaGraphNode_t* pDependentNodes, cudaGraphEdgeData* edgeData, size_t* pNumDependentNodes) nogil




    cudaError_t cudaGraphAddDependencies(cudaGraph_t graph, const cudaGraphNode_t* from_, const cudaGraphNode_t* to, const cudaGraphEdgeData* edgeData, size_t numDependencies) nogil




    cudaError_t cudaGraphRemoveDependencies(cudaGraph_t graph, const cudaGraphNode_t* from_, const cudaGraphNode_t* to, const cudaGraphEdgeData* edgeData, size_t numDependencies) nogil




    cudaError_t cudaGraphDestroyNode(cudaGraphNode_t node) nogil




    cudaError_t cudaGraphInstantiate(cudaGraphExec_t* pGraphExec, cudaGraph_t graph, unsigned long long flags) nogil




    cudaError_t cudaGraphInstantiateWithFlags(cudaGraphExec_t* pGraphExec, cudaGraph_t graph, unsigned long long flags) nogil




    cudaError_t cudaGraphInstantiateWithParams(cudaGraphExec_t* pGraphExec, cudaGraph_t graph, cudaGraphInstantiateParams* instantiateParams) nogil




    cudaError_t cudaGraphExecGetFlags(cudaGraphExec_t graphExec, unsigned long long* flags) nogil




    cudaError_t cudaGraphExecKernelNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, const cudaKernelNodeParams* pNodeParams) nogil




    cudaError_t cudaGraphExecMemcpyNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, const cudaMemcpy3DParms* pNodeParams) nogil




    cudaError_t cudaGraphExecMemcpyNodeSetParams1D(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, void* dst, const void* src, size_t count, cudaMemcpyKind kind) nogil




    cudaError_t cudaGraphExecMemsetNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, const cudaMemsetParams* pNodeParams) nogil




    cudaError_t cudaGraphExecHostNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, const cudaHostNodeParams* pNodeParams) nogil




    cudaError_t cudaGraphExecChildGraphNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, cudaGraph_t childGraph) nogil




    cudaError_t cudaGraphExecEventRecordNodeSetEvent(cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, cudaEvent_t event) nogil




    cudaError_t cudaGraphExecEventWaitNodeSetEvent(cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, cudaEvent_t event) nogil




    cudaError_t cudaGraphExecExternalSemaphoresSignalNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, const cudaExternalSemaphoreSignalNodeParams* nodeParams) nogil




    cudaError_t cudaGraphExecExternalSemaphoresWaitNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, const cudaExternalSemaphoreWaitNodeParams* nodeParams) nogil




    cudaError_t cudaGraphNodeSetEnabled(cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, unsigned int isEnabled) nogil




    cudaError_t cudaGraphNodeGetEnabled(cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, unsigned int* isEnabled) nogil




    cudaError_t cudaGraphExecUpdate(cudaGraphExec_t hGraphExec, cudaGraph_t hGraph, cudaGraphExecUpdateResultInfo* resultInfo) nogil




    cudaError_t cudaGraphUpload(cudaGraphExec_t graphExec, cudaStream_t stream) nogil




    cudaError_t cudaGraphLaunch(cudaGraphExec_t graphExec, cudaStream_t stream) nogil




    cudaError_t cudaGraphExecDestroy(cudaGraphExec_t graphExec) nogil




    cudaError_t cudaGraphDestroy(cudaGraph_t graph) nogil




    cudaError_t cudaGraphDebugDotPrint(cudaGraph_t graph, const char* path, unsigned int flags) nogil




    cudaError_t cudaUserObjectCreate(cudaUserObject_t* object_out, void* ptr, cudaHostFn_t destroy, unsigned int initialRefcount, unsigned int flags) nogil




    cudaError_t cudaUserObjectRetain(cudaUserObject_t object, unsigned int count) nogil




    cudaError_t cudaUserObjectRelease(cudaUserObject_t object, unsigned int count) nogil




    cudaError_t cudaGraphRetainUserObject(cudaGraph_t graph, cudaUserObject_t object, unsigned int count, unsigned int flags) nogil




    cudaError_t cudaGraphReleaseUserObject(cudaGraph_t graph, cudaUserObject_t object, unsigned int count) nogil




    cudaError_t cudaGraphAddNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, const cudaGraphEdgeData* dependencyData, size_t numDependencies, cudaGraphNodeParams* nodeParams) nogil




    cudaError_t cudaGraphNodeSetParams(cudaGraphNode_t node, cudaGraphNodeParams* nodeParams) nogil




    cudaError_t cudaGraphNodeGetParams(cudaGraphNode_t node, cudaGraphNodeParams* nodeParams) nogil




    cudaError_t cudaGraphExecNodeSetParams(cudaGraphExec_t graphExec, cudaGraphNode_t node, cudaGraphNodeParams* nodeParams) nogil




    cudaError_t cudaGraphConditionalHandleCreate(cudaGraphConditionalHandle* pHandle_out, cudaGraph_t graph, unsigned int defaultLaunchValue, unsigned int flags) nogil




    cudaError_t cudaGraphConditionalHandleCreate_v2(cudaGraphConditionalHandle* pHandle_out, cudaGraph_t graph, cudaExecutionContext_t ctx, unsigned int defaultLaunchValue, unsigned int flags) nogil




    cudaError_t cudaGetDriverEntryPoint(const char* symbol, void** funcPtr, unsigned long long flags, cudaDriverEntryPointQueryResult* driverStatus) nogil




    cudaError_t cudaGetDriverEntryPointByVersion(const char* symbol, void** funcPtr, unsigned int cudaVersion, unsigned long long flags, cudaDriverEntryPointQueryResult* driverStatus) nogil




    cudaError_t cudaLibraryLoadData(cudaLibrary_t* library, const void* code, cudaJitOption* jitOptions, void** jitOptionsValues, unsigned int numJitOptions, cudaLibraryOption* libraryOptions, void** libraryOptionValues, unsigned int numLibraryOptions) nogil




    cudaError_t cudaLibraryLoadFromFile(cudaLibrary_t* library, const char* fileName, cudaJitOption* jitOptions, void** jitOptionsValues, unsigned int numJitOptions, cudaLibraryOption* libraryOptions, void** libraryOptionValues, unsigned int numLibraryOptions) nogil




    cudaError_t cudaLibraryUnload(cudaLibrary_t library) nogil




    cudaError_t cudaLibraryGetKernel(cudaKernel_t* pKernel, cudaLibrary_t library, const char* name) nogil




    cudaError_t cudaLibraryGetGlobal(void** dptr, size_t* numbytes, cudaLibrary_t library, const char* name) nogil




    cudaError_t cudaLibraryGetManaged(void** dptr, size_t* numbytes, cudaLibrary_t library, const char* name) nogil




    cudaError_t cudaLibraryGetUnifiedFunction(void** fptr, cudaLibrary_t library, const char* symbol) nogil




    cudaError_t cudaLibraryGetKernelCount(unsigned int* count, cudaLibrary_t lib) nogil




    cudaError_t cudaLibraryEnumerateKernels(cudaKernel_t* kernels, unsigned int numKernels, cudaLibrary_t lib) nogil




    cudaError_t cudaKernelSetAttributeForDevice(cudaKernel_t kernel, cudaFuncAttribute attr, int value, int device) nogil




    cudaError_t cudaDeviceGetDevResource(int device, cudaDevResource* resource, cudaDevResourceType typename) nogil




    cudaError_t cudaDevSmResourceSplitByCount(cudaDevResource* result, unsigned int* nbGroups, const cudaDevResource* input, cudaDevResource* remaining, unsigned int flags, unsigned int minCount) nogil




    cudaError_t cudaDevSmResourceSplit(cudaDevResource* result, unsigned int nbGroups, const cudaDevResource* input, cudaDevResource* remainder, unsigned int flags, cudaDevSmResourceGroupParams* groupParams) nogil




    cudaError_t cudaDevResourceGenerateDesc(cudaDevResourceDesc_t* phDesc, cudaDevResource* resources, unsigned int nbResources) nogil




    cudaError_t cudaGreenCtxCreate(cudaExecutionContext_t* phCtx, cudaDevResourceDesc_t desc, int device, unsigned int flags) nogil




    cudaError_t cudaExecutionCtxDestroy(cudaExecutionContext_t ctx) nogil




    cudaError_t cudaExecutionCtxGetDevResource(cudaExecutionContext_t ctx, cudaDevResource* resource, cudaDevResourceType typename) nogil




    cudaError_t cudaExecutionCtxGetDevice(int* device, cudaExecutionContext_t ctx) nogil




    cudaError_t cudaExecutionCtxGetId(cudaExecutionContext_t ctx, unsigned long long* ctxId) nogil




    cudaError_t cudaExecutionCtxStreamCreate(cudaStream_t* phStream, cudaExecutionContext_t ctx, unsigned int flags, int priority) nogil




    cudaError_t cudaExecutionCtxSynchronize(cudaExecutionContext_t ctx) nogil




    cudaError_t cudaStreamGetDevResource(cudaStream_t hStream, cudaDevResource* resource, cudaDevResourceType typename) nogil




    cudaError_t cudaExecutionCtxRecordEvent(cudaExecutionContext_t ctx, cudaEvent_t event) nogil




    cudaError_t cudaExecutionCtxWaitEvent(cudaExecutionContext_t ctx, cudaEvent_t event) nogil




    cudaError_t cudaDeviceGetExecutionCtx(cudaExecutionContext_t* ctx, int device) nogil




    cudaError_t cudaGetExportTable(const void** ppExportTable, const cudaUUID_t* pExportTableId) nogil




    cudaError_t cudaGetKernel(cudaKernel_t* kernelPtr, const void* entryFuncAddr) nogil



cdef extern from "cuda_runtime.h":



    cudaPitchedPtr make_cudaPitchedPtr(void* d, size_t p, size_t xsz, size_t ysz) nogil




    cudaPos make_cudaPos(size_t x, size_t y, size_t z) nogil




    cudaExtent make_cudaExtent(size_t w, size_t h, size_t d) nogil



cdef extern from "cuda_profiler_api.h":



    cudaError_t cudaProfilerStart() nogil




    cudaError_t cudaProfilerStop() nogil
