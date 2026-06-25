# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE
#
# This code was automatically generated across versions from 12.9.0 to 13.3.0, generator version 0.3.1.dev1779+ga8cc71818.d20260623. Do not modify it directly.

cdef extern from "":
    """
    #define CUDA_API_PER_THREAD_DEFAULT_STREAM
    """


###############################################################################
# C function declarations for static cudart with per-thread default stream
# (CUDA_API_PER_THREAD_DEFAULT_STREAM causes cudaXxx to resolve to the _ptsz
# per-thread-stream symbol variants)
###############################################################################

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaDeviceReset "cudaDeviceReset" () noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaDeviceSynchronize "cudaDeviceSynchronize" () noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaDeviceSetLimit "cudaDeviceSetLimit" (cudaLimit limit, size_t value) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaDeviceGetLimit "cudaDeviceGetLimit" (size_t* pValue, cudaLimit limit) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaDeviceGetTexture1DLinearMaxWidth "cudaDeviceGetTexture1DLinearMaxWidth" (size_t* maxWidthInElements, const cudaChannelFormatDesc* fmtDesc, int device) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaDeviceGetCacheConfig "cudaDeviceGetCacheConfig" (cudaFuncCache* pCacheConfig) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaDeviceGetStreamPriorityRange "cudaDeviceGetStreamPriorityRange" (int* leastPriority, int* greatestPriority) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaDeviceSetCacheConfig "cudaDeviceSetCacheConfig" (cudaFuncCache cacheConfig) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaDeviceGetByPCIBusId "cudaDeviceGetByPCIBusId" (int* device, const char* pciBusId) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaDeviceGetPCIBusId "cudaDeviceGetPCIBusId" (char* pciBusId, int len, int device) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaIpcGetEventHandle "cudaIpcGetEventHandle" (cudaIpcEventHandle_t* handle, cudaEvent_t event) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaIpcOpenEventHandle "cudaIpcOpenEventHandle" (cudaEvent_t* event, cudaIpcEventHandle_t handle) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaIpcGetMemHandle "cudaIpcGetMemHandle" (cudaIpcMemHandle_t* handle, void* devPtr) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaIpcOpenMemHandle "cudaIpcOpenMemHandle" (void** devPtr, cudaIpcMemHandle_t handle, unsigned int flags) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaIpcCloseMemHandle "cudaIpcCloseMemHandle" (void* devPtr) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaDeviceFlushGPUDirectRDMAWrites "cudaDeviceFlushGPUDirectRDMAWrites" (cudaFlushGPUDirectRDMAWritesTarget target, cudaFlushGPUDirectRDMAWritesScope scope) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaDeviceRegisterAsyncNotification "cudaDeviceRegisterAsyncNotification" (int device, cudaAsyncCallback callbackFunc, void* userData, cudaAsyncCallbackHandle_t* callback) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaDeviceUnregisterAsyncNotification "cudaDeviceUnregisterAsyncNotification" (int device, cudaAsyncCallbackHandle_t callback) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaDeviceGetSharedMemConfig "cudaDeviceGetSharedMemConfig" (cudaSharedMemConfig* pConfig) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaDeviceSetSharedMemConfig "cudaDeviceSetSharedMemConfig" (cudaSharedMemConfig config) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaGetLastError "cudaGetLastError" () noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaPeekAtLastError "cudaPeekAtLastError" () noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    const char* _static_cudaGetErrorName "cudaGetErrorName" (cudaError_t error) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    const char* _static_cudaGetErrorString "cudaGetErrorString" (cudaError_t error) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaGetDeviceCount "cudaGetDeviceCount" (int* count) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaDeviceGetAttribute "cudaDeviceGetAttribute" (int* value, cudaDeviceAttr attr, int device) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaDeviceGetDefaultMemPool "cudaDeviceGetDefaultMemPool" (cudaMemPool_t* memPool, int device) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaDeviceSetMemPool "cudaDeviceSetMemPool" (int device, cudaMemPool_t memPool) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaDeviceGetMemPool "cudaDeviceGetMemPool" (cudaMemPool_t* memPool, int device) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaDeviceGetNvSciSyncAttributes "cudaDeviceGetNvSciSyncAttributes" (void* nvSciSyncAttrList, int device, int flags) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaDeviceGetP2PAttribute "cudaDeviceGetP2PAttribute" (int* value, cudaDeviceP2PAttr attr, int srcDevice, int dstDevice) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaChooseDevice "cudaChooseDevice" (int* device, const cudaDeviceProp* prop) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaInitDevice "cudaInitDevice" (int device, unsigned int deviceFlags, unsigned int flags) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaSetDevice "cudaSetDevice" (int device) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaGetDevice "cudaGetDevice" (int* device) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaSetDeviceFlags "cudaSetDeviceFlags" (unsigned int flags) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaGetDeviceFlags "cudaGetDeviceFlags" (unsigned int* flags) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaStreamCreate "cudaStreamCreate" (cudaStream_t* pStream) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaStreamCreateWithFlags "cudaStreamCreateWithFlags" (cudaStream_t* pStream, unsigned int flags) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaStreamCreateWithPriority "cudaStreamCreateWithPriority" (cudaStream_t* pStream, unsigned int flags, int priority) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaStreamGetPriority "cudaStreamGetPriority" (cudaStream_t hStream, int* priority) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaStreamGetFlags "cudaStreamGetFlags" (cudaStream_t hStream, unsigned int* flags) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaStreamGetId "cudaStreamGetId" (cudaStream_t hStream, unsigned long long* streamId) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaStreamGetDevice "cudaStreamGetDevice" (cudaStream_t hStream, int* device) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaCtxResetPersistingL2Cache "cudaCtxResetPersistingL2Cache" () noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaStreamCopyAttributes "cudaStreamCopyAttributes" (cudaStream_t dst, cudaStream_t src) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaStreamGetAttribute "cudaStreamGetAttribute" (cudaStream_t hStream, cudaLaunchAttributeID attr, cudaLaunchAttributeValue* value_out) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaStreamSetAttribute "cudaStreamSetAttribute" (cudaStream_t hStream, cudaLaunchAttributeID attr, const cudaLaunchAttributeValue* value) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaStreamDestroy "cudaStreamDestroy" (cudaStream_t stream) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaStreamWaitEvent "cudaStreamWaitEvent" (cudaStream_t stream, cudaEvent_t event, unsigned int flags) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaStreamAddCallback "cudaStreamAddCallback" (cudaStream_t stream, cudaStreamCallback_t callback, void* userData, unsigned int flags) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaStreamSynchronize "cudaStreamSynchronize" (cudaStream_t stream) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaStreamQuery "cudaStreamQuery" (cudaStream_t stream) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaStreamAttachMemAsync "cudaStreamAttachMemAsync" (cudaStream_t stream, void* devPtr, size_t length, unsigned int flags) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaStreamBeginCapture "cudaStreamBeginCapture" (cudaStream_t stream, cudaStreamCaptureMode mode) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaStreamBeginCaptureToGraph "cudaStreamBeginCaptureToGraph" (cudaStream_t stream, cudaGraph_t graph, const cudaGraphNode_t* dependencies, const cudaGraphEdgeData* dependencyData, size_t numDependencies, cudaStreamCaptureMode mode) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaThreadExchangeStreamCaptureMode "cudaThreadExchangeStreamCaptureMode" (cudaStreamCaptureMode* mode) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaStreamEndCapture "cudaStreamEndCapture" (cudaStream_t stream, cudaGraph_t* pGraph) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaStreamIsCapturing "cudaStreamIsCapturing" (cudaStream_t stream, cudaStreamCaptureStatus* pCaptureStatus) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaStreamUpdateCaptureDependencies "cudaStreamUpdateCaptureDependencies" (cudaStream_t stream, cudaGraphNode_t* dependencies, const cudaGraphEdgeData* dependencyData, size_t numDependencies, unsigned int flags) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaEventCreate "cudaEventCreate" (cudaEvent_t* event) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaEventCreateWithFlags "cudaEventCreateWithFlags" (cudaEvent_t* event, unsigned int flags) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaEventRecord "cudaEventRecord" (cudaEvent_t event, cudaStream_t stream) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaEventRecordWithFlags "cudaEventRecordWithFlags" (cudaEvent_t event, cudaStream_t stream, unsigned int flags) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaEventQuery "cudaEventQuery" (cudaEvent_t event) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaEventSynchronize "cudaEventSynchronize" (cudaEvent_t event) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaEventDestroy "cudaEventDestroy" (cudaEvent_t event) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaEventElapsedTime "cudaEventElapsedTime" (float* ms, cudaEvent_t start, cudaEvent_t end) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaImportExternalMemory "cudaImportExternalMemory" (cudaExternalMemory_t* extMem_out, const cudaExternalMemoryHandleDesc* memHandleDesc) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaExternalMemoryGetMappedBuffer "cudaExternalMemoryGetMappedBuffer" (void** devPtr, cudaExternalMemory_t extMem, const cudaExternalMemoryBufferDesc* bufferDesc) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaExternalMemoryGetMappedMipmappedArray "cudaExternalMemoryGetMappedMipmappedArray" (cudaMipmappedArray_t* mipmap, cudaExternalMemory_t extMem, const cudaExternalMemoryMipmappedArrayDesc* mipmapDesc) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaDestroyExternalMemory "cudaDestroyExternalMemory" (cudaExternalMemory_t extMem) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaImportExternalSemaphore "cudaImportExternalSemaphore" (cudaExternalSemaphore_t* extSem_out, const cudaExternalSemaphoreHandleDesc* semHandleDesc) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaDestroyExternalSemaphore "cudaDestroyExternalSemaphore" (cudaExternalSemaphore_t extSem) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaFuncSetCacheConfig "cudaFuncSetCacheConfig" (const void* func, cudaFuncCache cacheConfig) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaFuncGetAttributes "cudaFuncGetAttributes" (cudaFuncAttributes* attr, const void* func) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaFuncSetAttribute "cudaFuncSetAttribute" (const void* func, cudaFuncAttribute attr, int value) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaFuncGetName "cudaFuncGetName" (const char** name, const void* func) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaFuncGetParamInfo "cudaFuncGetParamInfo" (const void* func, size_t paramIndex, size_t* paramOffset, size_t* paramSize) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaLaunchHostFunc "cudaLaunchHostFunc" (cudaStream_t stream, cudaHostFn_t fn, void* userData) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaFuncSetSharedMemConfig "cudaFuncSetSharedMemConfig" (const void* func, cudaSharedMemConfig config) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaOccupancyMaxActiveBlocksPerMultiprocessor "cudaOccupancyMaxActiveBlocksPerMultiprocessor" (int* numBlocks, const void* func, int blockSize, size_t dynamicSMemSize) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaOccupancyAvailableDynamicSMemPerBlock "cudaOccupancyAvailableDynamicSMemPerBlock" (size_t* dynamicSmemSize, const void* func, int numBlocks, int blockSize) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags "cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags" (int* numBlocks, const void* func, int blockSize, size_t dynamicSMemSize, unsigned int flags) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaMallocManaged "cudaMallocManaged" (void** devPtr, size_t size, unsigned int flags) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaMalloc "cudaMalloc" (void** devPtr, size_t size) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaMallocHost "cudaMallocHost" (void** ptr, size_t size) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaMallocPitch "cudaMallocPitch" (void** devPtr, size_t* pitch, size_t width, size_t height) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaMallocArray "cudaMallocArray" (cudaArray_t* array, const cudaChannelFormatDesc* desc, size_t width, size_t height, unsigned int flags) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaFree "cudaFree" (void* devPtr) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaFreeHost "cudaFreeHost" (void* ptr) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaFreeArray "cudaFreeArray" (cudaArray_t array) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaFreeMipmappedArray "cudaFreeMipmappedArray" (cudaMipmappedArray_t mipmappedArray) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaHostAlloc "cudaHostAlloc" (void** pHost, size_t size, unsigned int flags) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaHostRegister "cudaHostRegister" (void* ptr, size_t size, unsigned int flags) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaHostUnregister "cudaHostUnregister" (void* ptr) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaHostGetDevicePointer "cudaHostGetDevicePointer" (void** pDevice, void* pHost, unsigned int flags) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaHostGetFlags "cudaHostGetFlags" (unsigned int* pFlags, void* pHost) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaMalloc3D "cudaMalloc3D" (cudaPitchedPtr* pitchedDevPtr, cudaExtent extent) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaMalloc3DArray "cudaMalloc3DArray" (cudaArray_t* array, const cudaChannelFormatDesc* desc, cudaExtent extent, unsigned int flags) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaMallocMipmappedArray "cudaMallocMipmappedArray" (cudaMipmappedArray_t* mipmappedArray, const cudaChannelFormatDesc* desc, cudaExtent extent, unsigned int numLevels, unsigned int flags) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaGetMipmappedArrayLevel "cudaGetMipmappedArrayLevel" (cudaArray_t* levelArray, cudaMipmappedArray_const_t mipmappedArray, unsigned int level) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaMemcpy3D "cudaMemcpy3D" (const cudaMemcpy3DParms* p) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaMemcpy3DPeer "cudaMemcpy3DPeer" (const cudaMemcpy3DPeerParms* p) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaMemcpy3DAsync "cudaMemcpy3DAsync" (const cudaMemcpy3DParms* p, cudaStream_t stream) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaMemcpy3DPeerAsync "cudaMemcpy3DPeerAsync" (const cudaMemcpy3DPeerParms* p, cudaStream_t stream) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaMemGetInfo "cudaMemGetInfo" (size_t* free, size_t* total) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaArrayGetInfo "cudaArrayGetInfo" (cudaChannelFormatDesc* desc, cudaExtent* extent, unsigned int* flags, cudaArray_t array) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaArrayGetPlane "cudaArrayGetPlane" (cudaArray_t* pPlaneArray, cudaArray_t hArray, unsigned int planeIdx) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaArrayGetMemoryRequirements "cudaArrayGetMemoryRequirements" (cudaArrayMemoryRequirements* memoryRequirements, cudaArray_t array, int device) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaMipmappedArrayGetMemoryRequirements "cudaMipmappedArrayGetMemoryRequirements" (cudaArrayMemoryRequirements* memoryRequirements, cudaMipmappedArray_t mipmap, int device) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaArrayGetSparseProperties "cudaArrayGetSparseProperties" (cudaArraySparseProperties* sparseProperties, cudaArray_t array) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaMipmappedArrayGetSparseProperties "cudaMipmappedArrayGetSparseProperties" (cudaArraySparseProperties* sparseProperties, cudaMipmappedArray_t mipmap) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaMemcpy "cudaMemcpy" (void* dst, const void* src, size_t count, cudaMemcpyKind kind) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaMemcpyPeer "cudaMemcpyPeer" (void* dst, int dstDevice, const void* src, int srcDevice, size_t count) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaMemcpy2D "cudaMemcpy2D" (void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaMemcpy2DToArray "cudaMemcpy2DToArray" (cudaArray_t dst, size_t wOffset, size_t hOffset, const void* src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaMemcpy2DFromArray "cudaMemcpy2DFromArray" (void* dst, size_t dpitch, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t width, size_t height, cudaMemcpyKind kind) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaMemcpy2DArrayToArray "cudaMemcpy2DArrayToArray" (cudaArray_t dst, size_t wOffsetDst, size_t hOffsetDst, cudaArray_const_t src, size_t wOffsetSrc, size_t hOffsetSrc, size_t width, size_t height, cudaMemcpyKind kind) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaMemcpyAsync "cudaMemcpyAsync" (void* dst, const void* src, size_t count, cudaMemcpyKind kind, cudaStream_t stream) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaMemcpyPeerAsync "cudaMemcpyPeerAsync" (void* dst, int dstDevice, const void* src, int srcDevice, size_t count, cudaStream_t stream) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaMemcpyBatchAsync "cudaMemcpyBatchAsync" (void** dsts, const void** srcs, const size_t* sizes, size_t count, cudaMemcpyAttributes* attrs, size_t* attrsIdxs, size_t numAttrs, cudaStream_t stream) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaMemcpy3DBatchAsync "cudaMemcpy3DBatchAsync" (size_t numOps, cudaMemcpy3DBatchOp* opList, unsigned long long flags, cudaStream_t stream) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaMemcpy2DAsync "cudaMemcpy2DAsync" (void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaMemcpy2DToArrayAsync "cudaMemcpy2DToArrayAsync" (cudaArray_t dst, size_t wOffset, size_t hOffset, const void* src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaMemcpy2DFromArrayAsync "cudaMemcpy2DFromArrayAsync" (void* dst, size_t dpitch, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaMemset "cudaMemset" (void* devPtr, int value, size_t count) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaMemset2D "cudaMemset2D" (void* devPtr, size_t pitch, int value, size_t width, size_t height) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaMemset3D "cudaMemset3D" (cudaPitchedPtr pitchedDevPtr, int value, cudaExtent extent) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaMemsetAsync "cudaMemsetAsync" (void* devPtr, int value, size_t count, cudaStream_t stream) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaMemset2DAsync "cudaMemset2DAsync" (void* devPtr, size_t pitch, int value, size_t width, size_t height, cudaStream_t stream) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaMemset3DAsync "cudaMemset3DAsync" (cudaPitchedPtr pitchedDevPtr, int value, cudaExtent extent, cudaStream_t stream) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaMemPrefetchAsync "cudaMemPrefetchAsync" (const void* devPtr, size_t count, cudaMemLocation location, unsigned int flags, cudaStream_t stream) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaMemAdvise "cudaMemAdvise" (const void* devPtr, size_t count, cudaMemoryAdvise advice, cudaMemLocation location) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaMemRangeGetAttribute "cudaMemRangeGetAttribute" (void* data, size_t dataSize, cudaMemRangeAttribute attribute, const void* devPtr, size_t count) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaMemRangeGetAttributes "cudaMemRangeGetAttributes" (void** data, size_t* dataSizes, cudaMemRangeAttribute* attributes, size_t numAttributes, const void* devPtr, size_t count) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaMemcpyToArray "cudaMemcpyToArray" (cudaArray_t dst, size_t wOffset, size_t hOffset, const void* src, size_t count, cudaMemcpyKind kind) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaMemcpyFromArray "cudaMemcpyFromArray" (void* dst, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t count, cudaMemcpyKind kind) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaMemcpyArrayToArray "cudaMemcpyArrayToArray" (cudaArray_t dst, size_t wOffsetDst, size_t hOffsetDst, cudaArray_const_t src, size_t wOffsetSrc, size_t hOffsetSrc, size_t count, cudaMemcpyKind kind) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaMemcpyToArrayAsync "cudaMemcpyToArrayAsync" (cudaArray_t dst, size_t wOffset, size_t hOffset, const void* src, size_t count, cudaMemcpyKind kind, cudaStream_t stream) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaMemcpyFromArrayAsync "cudaMemcpyFromArrayAsync" (void* dst, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t count, cudaMemcpyKind kind, cudaStream_t stream) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaMallocAsync "cudaMallocAsync" (void** devPtr, size_t size, cudaStream_t hStream) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaFreeAsync "cudaFreeAsync" (void* devPtr, cudaStream_t hStream) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaMemPoolTrimTo "cudaMemPoolTrimTo" (cudaMemPool_t memPool, size_t minBytesToKeep) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaMemPoolSetAttribute "cudaMemPoolSetAttribute" (cudaMemPool_t memPool, cudaMemPoolAttr attr, void* value) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaMemPoolGetAttribute "cudaMemPoolGetAttribute" (cudaMemPool_t memPool, cudaMemPoolAttr attr, void* value) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaMemPoolSetAccess "cudaMemPoolSetAccess" (cudaMemPool_t memPool, const cudaMemAccessDesc* descList, size_t count) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaMemPoolGetAccess "cudaMemPoolGetAccess" (cudaMemAccessFlags* flags, cudaMemPool_t memPool, cudaMemLocation* location) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaMemPoolCreate "cudaMemPoolCreate" (cudaMemPool_t* memPool, const cudaMemPoolProps* poolProps) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaMemPoolDestroy "cudaMemPoolDestroy" (cudaMemPool_t memPool) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaMallocFromPoolAsync "cudaMallocFromPoolAsync" (void** ptr, size_t size, cudaMemPool_t memPool, cudaStream_t stream) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaMemPoolExportToShareableHandle "cudaMemPoolExportToShareableHandle" (void* shareableHandle, cudaMemPool_t memPool, cudaMemAllocationHandleType handleType, unsigned int flags) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaMemPoolImportFromShareableHandle "cudaMemPoolImportFromShareableHandle" (cudaMemPool_t* memPool, void* shareableHandle, cudaMemAllocationHandleType handleType, unsigned int flags) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaMemPoolExportPointer "cudaMemPoolExportPointer" (cudaMemPoolPtrExportData* exportData, void* ptr) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaMemPoolImportPointer "cudaMemPoolImportPointer" (void** ptr, cudaMemPool_t memPool, cudaMemPoolPtrExportData* exportData) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaPointerGetAttributes "cudaPointerGetAttributes" (cudaPointerAttributes* attributes, const void* ptr) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaDeviceCanAccessPeer "cudaDeviceCanAccessPeer" (int* canAccessPeer, int device, int peerDevice) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaDeviceEnablePeerAccess "cudaDeviceEnablePeerAccess" (int peerDevice, unsigned int flags) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaDeviceDisablePeerAccess "cudaDeviceDisablePeerAccess" (int peerDevice) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaGraphicsUnregisterResource "cudaGraphicsUnregisterResource" (cudaGraphicsResource_t resource) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaGraphicsResourceSetMapFlags "cudaGraphicsResourceSetMapFlags" (cudaGraphicsResource_t resource, unsigned int flags) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaGraphicsMapResources "cudaGraphicsMapResources" (int count, cudaGraphicsResource_t* resources, cudaStream_t stream) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaGraphicsUnmapResources "cudaGraphicsUnmapResources" (int count, cudaGraphicsResource_t* resources, cudaStream_t stream) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaGraphicsResourceGetMappedPointer "cudaGraphicsResourceGetMappedPointer" (void** devPtr, size_t* size, cudaGraphicsResource_t resource) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaGraphicsSubResourceGetMappedArray "cudaGraphicsSubResourceGetMappedArray" (cudaArray_t* array, cudaGraphicsResource_t resource, unsigned int arrayIndex, unsigned int mipLevel) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaGraphicsResourceGetMappedMipmappedArray "cudaGraphicsResourceGetMappedMipmappedArray" (cudaMipmappedArray_t* mipmappedArray, cudaGraphicsResource_t resource) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaGetChannelDesc "cudaGetChannelDesc" (cudaChannelFormatDesc* desc, cudaArray_const_t array) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaChannelFormatDesc _static_cudaCreateChannelDesc "cudaCreateChannelDesc" (int x, int y, int z, int w, cudaChannelFormatKind f) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaCreateTextureObject "cudaCreateTextureObject" (cudaTextureObject_t* pTexObject, const cudaResourceDesc* pResDesc, const cudaTextureDesc* pTexDesc, const cudaResourceViewDesc* pResViewDesc) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaDestroyTextureObject "cudaDestroyTextureObject" (cudaTextureObject_t texObject) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaGetTextureObjectResourceDesc "cudaGetTextureObjectResourceDesc" (cudaResourceDesc* pResDesc, cudaTextureObject_t texObject) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaGetTextureObjectTextureDesc "cudaGetTextureObjectTextureDesc" (cudaTextureDesc* pTexDesc, cudaTextureObject_t texObject) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaGetTextureObjectResourceViewDesc "cudaGetTextureObjectResourceViewDesc" (cudaResourceViewDesc* pResViewDesc, cudaTextureObject_t texObject) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaCreateSurfaceObject "cudaCreateSurfaceObject" (cudaSurfaceObject_t* pSurfObject, const cudaResourceDesc* pResDesc) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaDestroySurfaceObject "cudaDestroySurfaceObject" (cudaSurfaceObject_t surfObject) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaGetSurfaceObjectResourceDesc "cudaGetSurfaceObjectResourceDesc" (cudaResourceDesc* pResDesc, cudaSurfaceObject_t surfObject) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaDriverGetVersion "cudaDriverGetVersion" (int* driverVersion) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaRuntimeGetVersion "cudaRuntimeGetVersion" (int* runtimeVersion) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaGraphCreate "cudaGraphCreate" (cudaGraph_t* pGraph, unsigned int flags) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaGraphAddKernelNode "cudaGraphAddKernelNode" (cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, const cudaKernelNodeParams* pNodeParams) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaGraphKernelNodeGetParams "cudaGraphKernelNodeGetParams" (cudaGraphNode_t node, cudaKernelNodeParams* pNodeParams) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaGraphKernelNodeSetParams "cudaGraphKernelNodeSetParams" (cudaGraphNode_t node, const cudaKernelNodeParams* pNodeParams) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaGraphKernelNodeCopyAttributes "cudaGraphKernelNodeCopyAttributes" (cudaGraphNode_t hDst, cudaGraphNode_t hSrc) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaGraphKernelNodeGetAttribute "cudaGraphKernelNodeGetAttribute" (cudaGraphNode_t hNode, cudaLaunchAttributeID attr, cudaLaunchAttributeValue* value_out) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaGraphKernelNodeSetAttribute "cudaGraphKernelNodeSetAttribute" (cudaGraphNode_t hNode, cudaLaunchAttributeID attr, const cudaLaunchAttributeValue* value) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaGraphAddMemcpyNode "cudaGraphAddMemcpyNode" (cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, const cudaMemcpy3DParms* pCopyParams) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaGraphAddMemcpyNode1D "cudaGraphAddMemcpyNode1D" (cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, void* dst, const void* src, size_t count, cudaMemcpyKind kind) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaGraphMemcpyNodeGetParams "cudaGraphMemcpyNodeGetParams" (cudaGraphNode_t node, cudaMemcpy3DParms* pNodeParams) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaGraphMemcpyNodeSetParams "cudaGraphMemcpyNodeSetParams" (cudaGraphNode_t node, const cudaMemcpy3DParms* pNodeParams) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaGraphMemcpyNodeSetParams1D "cudaGraphMemcpyNodeSetParams1D" (cudaGraphNode_t node, void* dst, const void* src, size_t count, cudaMemcpyKind kind) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaGraphAddMemsetNode "cudaGraphAddMemsetNode" (cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, const cudaMemsetParams* pMemsetParams) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaGraphMemsetNodeGetParams "cudaGraphMemsetNodeGetParams" (cudaGraphNode_t node, cudaMemsetParams* pNodeParams) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaGraphMemsetNodeSetParams "cudaGraphMemsetNodeSetParams" (cudaGraphNode_t node, const cudaMemsetParams* pNodeParams) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaGraphAddHostNode "cudaGraphAddHostNode" (cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, const cudaHostNodeParams* pNodeParams) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaGraphHostNodeGetParams "cudaGraphHostNodeGetParams" (cudaGraphNode_t node, cudaHostNodeParams* pNodeParams) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaGraphHostNodeSetParams "cudaGraphHostNodeSetParams" (cudaGraphNode_t node, const cudaHostNodeParams* pNodeParams) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaGraphAddChildGraphNode "cudaGraphAddChildGraphNode" (cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, cudaGraph_t childGraph) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaGraphChildGraphNodeGetGraph "cudaGraphChildGraphNodeGetGraph" (cudaGraphNode_t node, cudaGraph_t* pGraph) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaGraphAddEmptyNode "cudaGraphAddEmptyNode" (cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaGraphAddEventRecordNode "cudaGraphAddEventRecordNode" (cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, cudaEvent_t event) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaGraphEventRecordNodeGetEvent "cudaGraphEventRecordNodeGetEvent" (cudaGraphNode_t node, cudaEvent_t* event_out) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaGraphEventRecordNodeSetEvent "cudaGraphEventRecordNodeSetEvent" (cudaGraphNode_t node, cudaEvent_t event) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaGraphAddEventWaitNode "cudaGraphAddEventWaitNode" (cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, cudaEvent_t event) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaGraphEventWaitNodeGetEvent "cudaGraphEventWaitNodeGetEvent" (cudaGraphNode_t node, cudaEvent_t* event_out) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaGraphEventWaitNodeSetEvent "cudaGraphEventWaitNodeSetEvent" (cudaGraphNode_t node, cudaEvent_t event) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaGraphAddExternalSemaphoresSignalNode "cudaGraphAddExternalSemaphoresSignalNode" (cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, const cudaExternalSemaphoreSignalNodeParams* nodeParams) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaGraphExternalSemaphoresSignalNodeGetParams "cudaGraphExternalSemaphoresSignalNodeGetParams" (cudaGraphNode_t hNode, cudaExternalSemaphoreSignalNodeParams* params_out) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaGraphExternalSemaphoresSignalNodeSetParams "cudaGraphExternalSemaphoresSignalNodeSetParams" (cudaGraphNode_t hNode, const cudaExternalSemaphoreSignalNodeParams* nodeParams) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaGraphAddExternalSemaphoresWaitNode "cudaGraphAddExternalSemaphoresWaitNode" (cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, const cudaExternalSemaphoreWaitNodeParams* nodeParams) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaGraphExternalSemaphoresWaitNodeGetParams "cudaGraphExternalSemaphoresWaitNodeGetParams" (cudaGraphNode_t hNode, cudaExternalSemaphoreWaitNodeParams* params_out) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaGraphExternalSemaphoresWaitNodeSetParams "cudaGraphExternalSemaphoresWaitNodeSetParams" (cudaGraphNode_t hNode, const cudaExternalSemaphoreWaitNodeParams* nodeParams) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaGraphAddMemAllocNode "cudaGraphAddMemAllocNode" (cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, cudaMemAllocNodeParams* nodeParams) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaGraphMemAllocNodeGetParams "cudaGraphMemAllocNodeGetParams" (cudaGraphNode_t node, cudaMemAllocNodeParams* params_out) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaGraphAddMemFreeNode "cudaGraphAddMemFreeNode" (cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, void* dptr) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaGraphMemFreeNodeGetParams "cudaGraphMemFreeNodeGetParams" (cudaGraphNode_t node, void* dptr_out) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaDeviceGraphMemTrim "cudaDeviceGraphMemTrim" (int device) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaDeviceGetGraphMemAttribute "cudaDeviceGetGraphMemAttribute" (int device, cudaGraphMemAttributeType attr, void* value) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaDeviceSetGraphMemAttribute "cudaDeviceSetGraphMemAttribute" (int device, cudaGraphMemAttributeType attr, void* value) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaGraphClone "cudaGraphClone" (cudaGraph_t* pGraphClone, cudaGraph_t originalGraph) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaGraphNodeFindInClone "cudaGraphNodeFindInClone" (cudaGraphNode_t* pNode, cudaGraphNode_t originalNode, cudaGraph_t clonedGraph) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaGraphNodeGetType "cudaGraphNodeGetType" (cudaGraphNode_t node, cudaGraphNodeType* pType) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaGraphGetNodes "cudaGraphGetNodes" (cudaGraph_t graph, cudaGraphNode_t* nodes, size_t* numNodes) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaGraphGetRootNodes "cudaGraphGetRootNodes" (cudaGraph_t graph, cudaGraphNode_t* pRootNodes, size_t* pNumRootNodes) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaGraphGetEdges "cudaGraphGetEdges" (cudaGraph_t graph, cudaGraphNode_t* from_, cudaGraphNode_t* to, cudaGraphEdgeData* edgeData, size_t* numEdges) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaGraphNodeGetDependencies "cudaGraphNodeGetDependencies" (cudaGraphNode_t node, cudaGraphNode_t* pDependencies, cudaGraphEdgeData* edgeData, size_t* pNumDependencies) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaGraphNodeGetDependentNodes "cudaGraphNodeGetDependentNodes" (cudaGraphNode_t node, cudaGraphNode_t* pDependentNodes, cudaGraphEdgeData* edgeData, size_t* pNumDependentNodes) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaGraphAddDependencies "cudaGraphAddDependencies" (cudaGraph_t graph, const cudaGraphNode_t* from_, const cudaGraphNode_t* to, const cudaGraphEdgeData* edgeData, size_t numDependencies) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaGraphRemoveDependencies "cudaGraphRemoveDependencies" (cudaGraph_t graph, const cudaGraphNode_t* from_, const cudaGraphNode_t* to, const cudaGraphEdgeData* edgeData, size_t numDependencies) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaGraphDestroyNode "cudaGraphDestroyNode" (cudaGraphNode_t node) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaGraphInstantiate "cudaGraphInstantiate" (cudaGraphExec_t* pGraphExec, cudaGraph_t graph, unsigned long long flags) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaGraphInstantiateWithFlags "cudaGraphInstantiateWithFlags" (cudaGraphExec_t* pGraphExec, cudaGraph_t graph, unsigned long long flags) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaGraphInstantiateWithParams "cudaGraphInstantiateWithParams" (cudaGraphExec_t* pGraphExec, cudaGraph_t graph, cudaGraphInstantiateParams* instantiateParams) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaGraphExecGetFlags "cudaGraphExecGetFlags" (cudaGraphExec_t graphExec, unsigned long long* flags) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaGraphExecKernelNodeSetParams "cudaGraphExecKernelNodeSetParams" (cudaGraphExec_t hGraphExec, cudaGraphNode_t node, const cudaKernelNodeParams* pNodeParams) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaGraphExecMemcpyNodeSetParams "cudaGraphExecMemcpyNodeSetParams" (cudaGraphExec_t hGraphExec, cudaGraphNode_t node, const cudaMemcpy3DParms* pNodeParams) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaGraphExecMemcpyNodeSetParams1D "cudaGraphExecMemcpyNodeSetParams1D" (cudaGraphExec_t hGraphExec, cudaGraphNode_t node, void* dst, const void* src, size_t count, cudaMemcpyKind kind) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaGraphExecMemsetNodeSetParams "cudaGraphExecMemsetNodeSetParams" (cudaGraphExec_t hGraphExec, cudaGraphNode_t node, const cudaMemsetParams* pNodeParams) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaGraphExecHostNodeSetParams "cudaGraphExecHostNodeSetParams" (cudaGraphExec_t hGraphExec, cudaGraphNode_t node, const cudaHostNodeParams* pNodeParams) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaGraphExecChildGraphNodeSetParams "cudaGraphExecChildGraphNodeSetParams" (cudaGraphExec_t hGraphExec, cudaGraphNode_t node, cudaGraph_t childGraph) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaGraphExecEventRecordNodeSetEvent "cudaGraphExecEventRecordNodeSetEvent" (cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, cudaEvent_t event) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaGraphExecEventWaitNodeSetEvent "cudaGraphExecEventWaitNodeSetEvent" (cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, cudaEvent_t event) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaGraphExecExternalSemaphoresSignalNodeSetParams "cudaGraphExecExternalSemaphoresSignalNodeSetParams" (cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, const cudaExternalSemaphoreSignalNodeParams* nodeParams) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaGraphExecExternalSemaphoresWaitNodeSetParams "cudaGraphExecExternalSemaphoresWaitNodeSetParams" (cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, const cudaExternalSemaphoreWaitNodeParams* nodeParams) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaGraphNodeSetEnabled "cudaGraphNodeSetEnabled" (cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, unsigned int isEnabled) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaGraphNodeGetEnabled "cudaGraphNodeGetEnabled" (cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, unsigned int* isEnabled) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaGraphExecUpdate "cudaGraphExecUpdate" (cudaGraphExec_t hGraphExec, cudaGraph_t hGraph, cudaGraphExecUpdateResultInfo* resultInfo) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaGraphUpload "cudaGraphUpload" (cudaGraphExec_t graphExec, cudaStream_t stream) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaGraphLaunch "cudaGraphLaunch" (cudaGraphExec_t graphExec, cudaStream_t stream) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaGraphExecDestroy "cudaGraphExecDestroy" (cudaGraphExec_t graphExec) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaGraphDestroy "cudaGraphDestroy" (cudaGraph_t graph) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaGraphDebugDotPrint "cudaGraphDebugDotPrint" (cudaGraph_t graph, const char* path, unsigned int flags) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaUserObjectCreate "cudaUserObjectCreate" (cudaUserObject_t* object_out, void* ptr, cudaHostFn_t destroy, unsigned int initialRefcount, unsigned int flags) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaUserObjectRetain "cudaUserObjectRetain" (cudaUserObject_t object, unsigned int count) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaUserObjectRelease "cudaUserObjectRelease" (cudaUserObject_t object, unsigned int count) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaGraphRetainUserObject "cudaGraphRetainUserObject" (cudaGraph_t graph, cudaUserObject_t object, unsigned int count, unsigned int flags) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaGraphReleaseUserObject "cudaGraphReleaseUserObject" (cudaGraph_t graph, cudaUserObject_t object, unsigned int count) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaGraphAddNode "cudaGraphAddNode" (cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, const cudaGraphEdgeData* dependencyData, size_t numDependencies, cudaGraphNodeParams* nodeParams) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaGraphNodeSetParams "cudaGraphNodeSetParams" (cudaGraphNode_t node, cudaGraphNodeParams* nodeParams) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaGraphExecNodeSetParams "cudaGraphExecNodeSetParams" (cudaGraphExec_t graphExec, cudaGraphNode_t node, cudaGraphNodeParams* nodeParams) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaGraphConditionalHandleCreate "cudaGraphConditionalHandleCreate" (cudaGraphConditionalHandle* pHandle_out, cudaGraph_t graph, unsigned int defaultLaunchValue, unsigned int flags) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaGetDriverEntryPoint "cudaGetDriverEntryPoint" (const char* symbol, void** funcPtr, unsigned long long flags, cudaDriverEntryPointQueryResult* driverStatus) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaGetDriverEntryPointByVersion "cudaGetDriverEntryPointByVersion" (const char* symbol, void** funcPtr, unsigned int cudaVersion, unsigned long long flags, cudaDriverEntryPointQueryResult* driverStatus) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaLibraryLoadData "cudaLibraryLoadData" (cudaLibrary_t* library, const void* code, cudaJitOption* jitOptions, void** jitOptionsValues, unsigned int numJitOptions, cudaLibraryOption* libraryOptions, void** libraryOptionValues, unsigned int numLibraryOptions) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaLibraryLoadFromFile "cudaLibraryLoadFromFile" (cudaLibrary_t* library, const char* fileName, cudaJitOption* jitOptions, void** jitOptionsValues, unsigned int numJitOptions, cudaLibraryOption* libraryOptions, void** libraryOptionValues, unsigned int numLibraryOptions) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaLibraryUnload "cudaLibraryUnload" (cudaLibrary_t library) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaLibraryGetKernel "cudaLibraryGetKernel" (cudaKernel_t* pKernel, cudaLibrary_t library, const char* name) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaLibraryGetGlobal "cudaLibraryGetGlobal" (void** dptr, size_t* bytes, cudaLibrary_t library, const char* name) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaLibraryGetManaged "cudaLibraryGetManaged" (void** dptr, size_t* bytes, cudaLibrary_t library, const char* name) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaLibraryGetUnifiedFunction "cudaLibraryGetUnifiedFunction" (void** fptr, cudaLibrary_t library, const char* symbol) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaLibraryGetKernelCount "cudaLibraryGetKernelCount" (unsigned int* count, cudaLibrary_t lib) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaLibraryEnumerateKernels "cudaLibraryEnumerateKernels" (cudaKernel_t* kernels, unsigned int numKernels, cudaLibrary_t lib) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaKernelSetAttributeForDevice "cudaKernelSetAttributeForDevice" (cudaKernel_t kernel, cudaFuncAttribute attr, int value, int device) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaGetExportTable "cudaGetExportTable" (const void** ppExportTable, const cudaUUID_t* pExportTableId) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaGetKernel "cudaGetKernel" (cudaKernel_t* kernelPtr, const void* entryFuncAddr) noexcept

cdef extern from 'cuda_profiler_api.h' nogil:
    cudaError_t _static_cudaProfilerStart "cudaProfilerStart" () noexcept

cdef extern from 'cuda_profiler_api.h' nogil:
    cudaError_t _static_cudaProfilerStop "cudaProfilerStop" () noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaGetDeviceProperties "cudaGetDeviceProperties" (cudaDeviceProp* prop, int device) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaDeviceGetHostAtomicCapabilities "cudaDeviceGetHostAtomicCapabilities" (unsigned int* capabilities, const cudaAtomicOperation* operations, unsigned int count, int device) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaDeviceGetP2PAtomicCapabilities "cudaDeviceGetP2PAtomicCapabilities" (unsigned int* capabilities, const cudaAtomicOperation* operations, unsigned int count, int srcDevice, int dstDevice) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaStreamGetCaptureInfo "cudaStreamGetCaptureInfo" (cudaStream_t stream, cudaStreamCaptureStatus* captureStatus_out, unsigned long long* id_out, cudaGraph_t* graph_out, const cudaGraphNode_t** dependencies_out, const cudaGraphEdgeData** edgeData_out, size_t* numDependencies_out) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaSignalExternalSemaphoresAsync "cudaSignalExternalSemaphoresAsync" (const cudaExternalSemaphore_t* extSemArray, const cudaExternalSemaphoreSignalParams* paramsArray, unsigned int numExtSems, cudaStream_t stream) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaWaitExternalSemaphoresAsync "cudaWaitExternalSemaphoresAsync" (const cudaExternalSemaphore_t* extSemArray, const cudaExternalSemaphoreWaitParams* paramsArray, unsigned int numExtSems, cudaStream_t stream) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaMemPrefetchBatchAsync "cudaMemPrefetchBatchAsync" (void** dptrs, size_t* sizes, size_t count, cudaMemLocation* prefetchLocs, size_t* prefetchLocIdxs, size_t numPrefetchLocs, unsigned long long flags, cudaStream_t stream) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaMemDiscardBatchAsync "cudaMemDiscardBatchAsync" (void** dptrs, size_t* sizes, size_t count, unsigned long long flags, cudaStream_t stream) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaMemDiscardAndPrefetchBatchAsync "cudaMemDiscardAndPrefetchBatchAsync" (void** dptrs, size_t* sizes, size_t count, cudaMemLocation* prefetchLocs, size_t* prefetchLocIdxs, size_t numPrefetchLocs, unsigned long long flags, cudaStream_t stream) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaMemGetDefaultMemPool "cudaMemGetDefaultMemPool" (cudaMemPool_t* memPool, cudaMemLocation* location, cudaMemAllocationType type) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaMemGetMemPool "cudaMemGetMemPool" (cudaMemPool_t* memPool, cudaMemLocation* location, cudaMemAllocationType type) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaMemSetMemPool "cudaMemSetMemPool" (cudaMemLocation* location, cudaMemAllocationType type, cudaMemPool_t memPool) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaLogsRegisterCallback "cudaLogsRegisterCallback" (cudaLogsCallback_t callbackFunc, void* userData, cudaLogsCallbackHandle* callback_out) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaLogsUnregisterCallback "cudaLogsUnregisterCallback" (cudaLogsCallbackHandle callback) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaLogsCurrent "cudaLogsCurrent" (cudaLogIterator* iterator_out, unsigned int flags) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaLogsDumpToFile "cudaLogsDumpToFile" (cudaLogIterator* iterator, const char* pathToFile, unsigned int flags) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaLogsDumpToMemory "cudaLogsDumpToMemory" (cudaLogIterator* iterator, char* buffer, size_t* size, unsigned int flags) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaGraphNodeGetContainingGraph "cudaGraphNodeGetContainingGraph" (cudaGraphNode_t hNode, cudaGraph_t* phGraph) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaGraphNodeGetLocalId "cudaGraphNodeGetLocalId" (cudaGraphNode_t hNode, unsigned int* nodeId) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaGraphNodeGetToolsId "cudaGraphNodeGetToolsId" (cudaGraphNode_t hNode, unsigned long long* toolsNodeId) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaGraphGetId "cudaGraphGetId" (cudaGraph_t hGraph, unsigned int* graphID) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaGraphExecGetId "cudaGraphExecGetId" (cudaGraphExec_t hGraphExec, unsigned int* graphID) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaGraphConditionalHandleCreate_v2 "cudaGraphConditionalHandleCreate_v2" (cudaGraphConditionalHandle* pHandle_out, cudaGraph_t graph, cudaExecutionContext_t ctx, unsigned int defaultLaunchValue, unsigned int flags) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaDeviceGetDevResource "cudaDeviceGetDevResource" (int device, cudaDevResource* resource, cudaDevResourceType type) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaDevSmResourceSplitByCount "cudaDevSmResourceSplitByCount" (cudaDevResource* result, unsigned int* nbGroups, const cudaDevResource* input, cudaDevResource* remaining, unsigned int flags, unsigned int minCount) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaDevSmResourceSplit "cudaDevSmResourceSplit" (cudaDevResource* result, unsigned int nbGroups, const cudaDevResource* input, cudaDevResource* remainder, unsigned int flags, cudaDevSmResourceGroupParams* groupParams) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaDevResourceGenerateDesc "cudaDevResourceGenerateDesc" (cudaDevResourceDesc_t* phDesc, cudaDevResource* resources, unsigned int nbResources) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaGreenCtxCreate "cudaGreenCtxCreate" (cudaExecutionContext_t* phCtx, cudaDevResourceDesc_t desc, int device, unsigned int flags) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaExecutionCtxDestroy "cudaExecutionCtxDestroy" (cudaExecutionContext_t ctx) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaExecutionCtxGetDevResource "cudaExecutionCtxGetDevResource" (cudaExecutionContext_t ctx, cudaDevResource* resource, cudaDevResourceType type) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaExecutionCtxGetDevice "cudaExecutionCtxGetDevice" (int* device, cudaExecutionContext_t ctx) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaExecutionCtxGetId "cudaExecutionCtxGetId" (cudaExecutionContext_t ctx, unsigned long long* ctxId) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaExecutionCtxStreamCreate "cudaExecutionCtxStreamCreate" (cudaStream_t* phStream, cudaExecutionContext_t ctx, unsigned int flags, int priority) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaExecutionCtxSynchronize "cudaExecutionCtxSynchronize" (cudaExecutionContext_t ctx) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaStreamGetDevResource "cudaStreamGetDevResource" (cudaStream_t hStream, cudaDevResource* resource, cudaDevResourceType type) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaExecutionCtxRecordEvent "cudaExecutionCtxRecordEvent" (cudaExecutionContext_t ctx, cudaEvent_t event) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaExecutionCtxWaitEvent "cudaExecutionCtxWaitEvent" (cudaExecutionContext_t ctx, cudaEvent_t event) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaDeviceGetExecutionCtx "cudaDeviceGetExecutionCtx" (cudaExecutionContext_t* ctx, int device) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaFuncGetParamCount "cudaFuncGetParamCount" (const void* func, size_t* paramCount) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaLaunchHostFunc_v2 "cudaLaunchHostFunc_v2" (cudaStream_t stream, cudaHostFn_t fn, void* userData, unsigned int syncMode) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaMemcpyWithAttributesAsync "cudaMemcpyWithAttributesAsync" (void* dst, const void* src, size_t size, cudaMemcpyAttributes* attr, cudaStream_t stream) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaMemcpy3DWithAttributesAsync "cudaMemcpy3DWithAttributesAsync" (cudaMemcpy3DBatchOp* op, unsigned long long flags, cudaStream_t stream) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaGraphNodeGetParams "cudaGraphNodeGetParams" (cudaGraphNode_t node, cudaGraphNodeParams* nodeParams) noexcept

cdef extern from 'cuda_runtime_api.h' nogil:
    cudaError_t _static_cudaStreamBeginRecaptureToGraph "cudaStreamBeginRecaptureToGraph" (cudaStream_t stream, cudaStreamCaptureMode mode, cudaGraph_t graph, cudaGraphRecaptureCallbackData* callbackData) noexcept


###############################################################################
# Wrapper functions
###############################################################################

cdef cudaError_t _cudaDeviceReset() except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaDeviceReset()


cdef cudaError_t _cudaDeviceSynchronize() except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaDeviceSynchronize()


cdef cudaError_t _cudaDeviceSetLimit(cudaLimit limit, size_t value) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaDeviceSetLimit(limit, value)


cdef cudaError_t _cudaDeviceGetLimit(size_t* pValue, cudaLimit limit) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaDeviceGetLimit(pValue, limit)


cdef cudaError_t _cudaDeviceGetTexture1DLinearMaxWidth(size_t* maxWidthInElements, const cudaChannelFormatDesc* fmtDesc, int device) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaDeviceGetTexture1DLinearMaxWidth(maxWidthInElements, fmtDesc, device)


cdef cudaError_t _cudaDeviceGetCacheConfig(cudaFuncCache* pCacheConfig) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaDeviceGetCacheConfig(pCacheConfig)


cdef cudaError_t _cudaDeviceGetStreamPriorityRange(int* leastPriority, int* greatestPriority) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaDeviceGetStreamPriorityRange(leastPriority, greatestPriority)


cdef cudaError_t _cudaDeviceSetCacheConfig(cudaFuncCache cacheConfig) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaDeviceSetCacheConfig(cacheConfig)


cdef cudaError_t _cudaDeviceGetByPCIBusId(int* device, const char* pciBusId) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaDeviceGetByPCIBusId(device, pciBusId)


cdef cudaError_t _cudaDeviceGetPCIBusId(char* pciBusId, int len, int device) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaDeviceGetPCIBusId(pciBusId, len, device)


cdef cudaError_t _cudaIpcGetEventHandle(cudaIpcEventHandle_t* handle, cudaEvent_t event) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaIpcGetEventHandle(handle, event)


cdef cudaError_t _cudaIpcOpenEventHandle(cudaEvent_t* event, cudaIpcEventHandle_t handle) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaIpcOpenEventHandle(event, handle)


cdef cudaError_t _cudaIpcGetMemHandle(cudaIpcMemHandle_t* handle, void* devPtr) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaIpcGetMemHandle(handle, devPtr)


cdef cudaError_t _cudaIpcOpenMemHandle(void** devPtr, cudaIpcMemHandle_t handle, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaIpcOpenMemHandle(devPtr, handle, flags)


cdef cudaError_t _cudaIpcCloseMemHandle(void* devPtr) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaIpcCloseMemHandle(devPtr)


cdef cudaError_t _cudaDeviceFlushGPUDirectRDMAWrites(cudaFlushGPUDirectRDMAWritesTarget target, cudaFlushGPUDirectRDMAWritesScope scope) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaDeviceFlushGPUDirectRDMAWrites(target, scope)


cdef cudaError_t _cudaDeviceRegisterAsyncNotification(int device, cudaAsyncCallback callbackFunc, void* userData, cudaAsyncCallbackHandle_t* callback) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaDeviceRegisterAsyncNotification(device, callbackFunc, userData, callback)


cdef cudaError_t _cudaDeviceUnregisterAsyncNotification(int device, cudaAsyncCallbackHandle_t callback) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaDeviceUnregisterAsyncNotification(device, callback)


cdef cudaError_t _cudaDeviceGetSharedMemConfig(cudaSharedMemConfig* pConfig) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaDeviceGetSharedMemConfig(pConfig)


cdef cudaError_t _cudaDeviceSetSharedMemConfig(cudaSharedMemConfig config) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaDeviceSetSharedMemConfig(config)


cdef cudaError_t _cudaGetLastError() except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaGetLastError()


cdef cudaError_t _cudaPeekAtLastError() except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaPeekAtLastError()


cdef const char* _cudaGetErrorName(cudaError_t error) except?NULL nogil:
    return _static_cudaGetErrorName(error)


cdef const char* _cudaGetErrorString(cudaError_t error) except?NULL nogil:
    return _static_cudaGetErrorString(error)


cdef cudaError_t _cudaGetDeviceCount(int* count) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaGetDeviceCount(count)


cdef cudaError_t _cudaDeviceGetAttribute(int* value, cudaDeviceAttr attr, int device) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaDeviceGetAttribute(value, attr, device)


cdef cudaError_t _cudaDeviceGetDefaultMemPool(cudaMemPool_t* memPool, int device) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaDeviceGetDefaultMemPool(memPool, device)


cdef cudaError_t _cudaDeviceSetMemPool(int device, cudaMemPool_t memPool) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaDeviceSetMemPool(device, memPool)


cdef cudaError_t _cudaDeviceGetMemPool(cudaMemPool_t* memPool, int device) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaDeviceGetMemPool(memPool, device)


cdef cudaError_t _cudaDeviceGetNvSciSyncAttributes(void* nvSciSyncAttrList, int device, int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaDeviceGetNvSciSyncAttributes(nvSciSyncAttrList, device, flags)


cdef cudaError_t _cudaDeviceGetP2PAttribute(int* value, cudaDeviceP2PAttr attr, int srcDevice, int dstDevice) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaDeviceGetP2PAttribute(value, attr, srcDevice, dstDevice)


cdef cudaError_t _cudaChooseDevice(int* device, const cudaDeviceProp* prop) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaChooseDevice(device, prop)


cdef cudaError_t _cudaInitDevice(int device, unsigned int deviceFlags, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaInitDevice(device, deviceFlags, flags)


cdef cudaError_t _cudaSetDevice(int device) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaSetDevice(device)


cdef cudaError_t _cudaGetDevice(int* device) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaGetDevice(device)


cdef cudaError_t _cudaSetDeviceFlags(unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaSetDeviceFlags(flags)


cdef cudaError_t _cudaGetDeviceFlags(unsigned int* flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaGetDeviceFlags(flags)


cdef cudaError_t _cudaStreamCreate(cudaStream_t* pStream) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaStreamCreate(pStream)


cdef cudaError_t _cudaStreamCreateWithFlags(cudaStream_t* pStream, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaStreamCreateWithFlags(pStream, flags)


cdef cudaError_t _cudaStreamCreateWithPriority(cudaStream_t* pStream, unsigned int flags, int priority) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaStreamCreateWithPriority(pStream, flags, priority)


cdef cudaError_t _cudaStreamGetPriority(cudaStream_t hStream, int* priority) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaStreamGetPriority(hStream, priority)


cdef cudaError_t _cudaStreamGetFlags(cudaStream_t hStream, unsigned int* flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaStreamGetFlags(hStream, flags)


cdef cudaError_t _cudaStreamGetId(cudaStream_t hStream, unsigned long long* streamId) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaStreamGetId(hStream, streamId)


cdef cudaError_t _cudaStreamGetDevice(cudaStream_t hStream, int* device) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaStreamGetDevice(hStream, device)


cdef cudaError_t _cudaCtxResetPersistingL2Cache() except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaCtxResetPersistingL2Cache()


cdef cudaError_t _cudaStreamCopyAttributes(cudaStream_t dst, cudaStream_t src) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaStreamCopyAttributes(dst, src)


cdef cudaError_t _cudaStreamGetAttribute(cudaStream_t hStream, cudaLaunchAttributeID attr, cudaLaunchAttributeValue* value_out) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaStreamGetAttribute(hStream, attr, value_out)


cdef cudaError_t _cudaStreamSetAttribute(cudaStream_t hStream, cudaLaunchAttributeID attr, const cudaLaunchAttributeValue* value) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaStreamSetAttribute(hStream, attr, value)


cdef cudaError_t _cudaStreamDestroy(cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaStreamDestroy(stream)


cdef cudaError_t _cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaStreamWaitEvent(stream, event, flags)


cdef cudaError_t _cudaStreamAddCallback(cudaStream_t stream, cudaStreamCallback_t callback, void* userData, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaStreamAddCallback(stream, callback, userData, flags)


cdef cudaError_t _cudaStreamSynchronize(cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaStreamSynchronize(stream)


cdef cudaError_t _cudaStreamQuery(cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaStreamQuery(stream)


cdef cudaError_t _cudaStreamAttachMemAsync(cudaStream_t stream, void* devPtr, size_t length, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaStreamAttachMemAsync(stream, devPtr, length, flags)


cdef cudaError_t _cudaStreamBeginCapture(cudaStream_t stream, cudaStreamCaptureMode mode) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaStreamBeginCapture(stream, mode)


cdef cudaError_t _cudaStreamBeginCaptureToGraph(cudaStream_t stream, cudaGraph_t graph, const cudaGraphNode_t* dependencies, const cudaGraphEdgeData* dependencyData, size_t numDependencies, cudaStreamCaptureMode mode) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaStreamBeginCaptureToGraph(stream, graph, dependencies, dependencyData, numDependencies, mode)


cdef cudaError_t _cudaThreadExchangeStreamCaptureMode(cudaStreamCaptureMode* mode) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaThreadExchangeStreamCaptureMode(mode)


cdef cudaError_t _cudaStreamEndCapture(cudaStream_t stream, cudaGraph_t* pGraph) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaStreamEndCapture(stream, pGraph)


cdef cudaError_t _cudaStreamIsCapturing(cudaStream_t stream, cudaStreamCaptureStatus* pCaptureStatus) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaStreamIsCapturing(stream, pCaptureStatus)


cdef cudaError_t _cudaStreamUpdateCaptureDependencies(cudaStream_t stream, cudaGraphNode_t* dependencies, const cudaGraphEdgeData* dependencyData, size_t numDependencies, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaStreamUpdateCaptureDependencies(stream, dependencies, dependencyData, numDependencies, flags)


cdef cudaError_t _cudaEventCreate(cudaEvent_t* event) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaEventCreate(event)


cdef cudaError_t _cudaEventCreateWithFlags(cudaEvent_t* event, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaEventCreateWithFlags(event, flags)


cdef cudaError_t _cudaEventRecord(cudaEvent_t event, cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaEventRecord(event, stream)


cdef cudaError_t _cudaEventRecordWithFlags(cudaEvent_t event, cudaStream_t stream, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaEventRecordWithFlags(event, stream, flags)


cdef cudaError_t _cudaEventQuery(cudaEvent_t event) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaEventQuery(event)


cdef cudaError_t _cudaEventSynchronize(cudaEvent_t event) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaEventSynchronize(event)


cdef cudaError_t _cudaEventDestroy(cudaEvent_t event) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaEventDestroy(event)


cdef cudaError_t _cudaEventElapsedTime(float* ms, cudaEvent_t start, cudaEvent_t end) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaEventElapsedTime(ms, start, end)


cdef cudaError_t _cudaImportExternalMemory(cudaExternalMemory_t* extMem_out, const cudaExternalMemoryHandleDesc* memHandleDesc) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaImportExternalMemory(extMem_out, memHandleDesc)


cdef cudaError_t _cudaExternalMemoryGetMappedBuffer(void** devPtr, cudaExternalMemory_t extMem, const cudaExternalMemoryBufferDesc* bufferDesc) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaExternalMemoryGetMappedBuffer(devPtr, extMem, bufferDesc)


cdef cudaError_t _cudaExternalMemoryGetMappedMipmappedArray(cudaMipmappedArray_t* mipmap, cudaExternalMemory_t extMem, const cudaExternalMemoryMipmappedArrayDesc* mipmapDesc) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaExternalMemoryGetMappedMipmappedArray(mipmap, extMem, mipmapDesc)


cdef cudaError_t _cudaDestroyExternalMemory(cudaExternalMemory_t extMem) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaDestroyExternalMemory(extMem)


cdef cudaError_t _cudaImportExternalSemaphore(cudaExternalSemaphore_t* extSem_out, const cudaExternalSemaphoreHandleDesc* semHandleDesc) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaImportExternalSemaphore(extSem_out, semHandleDesc)


cdef cudaError_t _cudaDestroyExternalSemaphore(cudaExternalSemaphore_t extSem) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaDestroyExternalSemaphore(extSem)


cdef cudaError_t _cudaFuncSetCacheConfig(const void* func, cudaFuncCache cacheConfig) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaFuncSetCacheConfig(func, cacheConfig)


cdef cudaError_t _cudaFuncGetAttributes(cudaFuncAttributes* attr, const void* func) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaFuncGetAttributes(attr, func)


cdef cudaError_t _cudaFuncSetAttribute(const void* func, cudaFuncAttribute attr, int value) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaFuncSetAttribute(func, attr, value)


cdef cudaError_t _cudaFuncGetName(const char** name, const void* func) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaFuncGetName(name, func)


cdef cudaError_t _cudaFuncGetParamInfo(const void* func, size_t paramIndex, size_t* paramOffset, size_t* paramSize) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaFuncGetParamInfo(func, paramIndex, paramOffset, paramSize)


cdef cudaError_t _cudaLaunchHostFunc(cudaStream_t stream, cudaHostFn_t fn, void* userData) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaLaunchHostFunc(stream, fn, userData)


cdef cudaError_t _cudaFuncSetSharedMemConfig(const void* func, cudaSharedMemConfig config) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaFuncSetSharedMemConfig(func, config)


cdef cudaError_t _cudaOccupancyMaxActiveBlocksPerMultiprocessor(int* numBlocks, const void* func, int blockSize, size_t dynamicSMemSize) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaOccupancyMaxActiveBlocksPerMultiprocessor(numBlocks, func, blockSize, dynamicSMemSize)


cdef cudaError_t _cudaOccupancyAvailableDynamicSMemPerBlock(size_t* dynamicSmemSize, const void* func, int numBlocks, int blockSize) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaOccupancyAvailableDynamicSMemPerBlock(dynamicSmemSize, func, numBlocks, blockSize)


cdef cudaError_t _cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int* numBlocks, const void* func, int blockSize, size_t dynamicSMemSize, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(numBlocks, func, blockSize, dynamicSMemSize, flags)


cdef cudaError_t _cudaMallocManaged(void** devPtr, size_t size, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaMallocManaged(devPtr, size, flags)


cdef cudaError_t _cudaMalloc(void** devPtr, size_t size) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaMalloc(devPtr, size)


cdef cudaError_t _cudaMallocHost(void** ptr, size_t size) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaMallocHost(ptr, size)


cdef cudaError_t _cudaMallocPitch(void** devPtr, size_t* pitch, size_t width, size_t height) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaMallocPitch(devPtr, pitch, width, height)


cdef cudaError_t _cudaMallocArray(cudaArray_t* array, const cudaChannelFormatDesc* desc, size_t width, size_t height, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaMallocArray(array, desc, width, height, flags)


cdef cudaError_t _cudaFree(void* devPtr) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaFree(devPtr)


cdef cudaError_t _cudaFreeHost(void* ptr) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaFreeHost(ptr)


cdef cudaError_t _cudaFreeArray(cudaArray_t array) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaFreeArray(array)


cdef cudaError_t _cudaFreeMipmappedArray(cudaMipmappedArray_t mipmappedArray) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaFreeMipmappedArray(mipmappedArray)


cdef cudaError_t _cudaHostAlloc(void** pHost, size_t size, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaHostAlloc(pHost, size, flags)


cdef cudaError_t _cudaHostRegister(void* ptr, size_t size, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaHostRegister(ptr, size, flags)


cdef cudaError_t _cudaHostUnregister(void* ptr) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaHostUnregister(ptr)


cdef cudaError_t _cudaHostGetDevicePointer(void** pDevice, void* pHost, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaHostGetDevicePointer(pDevice, pHost, flags)


cdef cudaError_t _cudaHostGetFlags(unsigned int* pFlags, void* pHost) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaHostGetFlags(pFlags, pHost)


cdef cudaError_t _cudaMalloc3D(cudaPitchedPtr* pitchedDevPtr, cudaExtent extent) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaMalloc3D(pitchedDevPtr, extent)


cdef cudaError_t _cudaMalloc3DArray(cudaArray_t* array, const cudaChannelFormatDesc* desc, cudaExtent extent, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaMalloc3DArray(array, desc, extent, flags)


cdef cudaError_t _cudaMallocMipmappedArray(cudaMipmappedArray_t* mipmappedArray, const cudaChannelFormatDesc* desc, cudaExtent extent, unsigned int numLevels, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaMallocMipmappedArray(mipmappedArray, desc, extent, numLevels, flags)


cdef cudaError_t _cudaGetMipmappedArrayLevel(cudaArray_t* levelArray, cudaMipmappedArray_const_t mipmappedArray, unsigned int level) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaGetMipmappedArrayLevel(levelArray, mipmappedArray, level)


cdef cudaError_t _cudaMemcpy3D(const cudaMemcpy3DParms* p) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaMemcpy3D(p)


cdef cudaError_t _cudaMemcpy3DPeer(const cudaMemcpy3DPeerParms* p) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaMemcpy3DPeer(p)


cdef cudaError_t _cudaMemcpy3DAsync(const cudaMemcpy3DParms* p, cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaMemcpy3DAsync(p, stream)


cdef cudaError_t _cudaMemcpy3DPeerAsync(const cudaMemcpy3DPeerParms* p, cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaMemcpy3DPeerAsync(p, stream)


cdef cudaError_t _cudaMemGetInfo(size_t* free, size_t* total) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaMemGetInfo(free, total)


cdef cudaError_t _cudaArrayGetInfo(cudaChannelFormatDesc* desc, cudaExtent* extent, unsigned int* flags, cudaArray_t array) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaArrayGetInfo(desc, extent, flags, array)


cdef cudaError_t _cudaArrayGetPlane(cudaArray_t* pPlaneArray, cudaArray_t hArray, unsigned int planeIdx) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaArrayGetPlane(pPlaneArray, hArray, planeIdx)


cdef cudaError_t _cudaArrayGetMemoryRequirements(cudaArrayMemoryRequirements* memoryRequirements, cudaArray_t array, int device) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaArrayGetMemoryRequirements(memoryRequirements, array, device)


cdef cudaError_t _cudaMipmappedArrayGetMemoryRequirements(cudaArrayMemoryRequirements* memoryRequirements, cudaMipmappedArray_t mipmap, int device) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaMipmappedArrayGetMemoryRequirements(memoryRequirements, mipmap, device)


cdef cudaError_t _cudaArrayGetSparseProperties(cudaArraySparseProperties* sparseProperties, cudaArray_t array) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaArrayGetSparseProperties(sparseProperties, array)


cdef cudaError_t _cudaMipmappedArrayGetSparseProperties(cudaArraySparseProperties* sparseProperties, cudaMipmappedArray_t mipmap) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaMipmappedArrayGetSparseProperties(sparseProperties, mipmap)


cdef cudaError_t _cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaMemcpy(dst, src, count, kind)


cdef cudaError_t _cudaMemcpyPeer(void* dst, int dstDevice, const void* src, int srcDevice, size_t count) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaMemcpyPeer(dst, dstDevice, src, srcDevice, count)


cdef cudaError_t _cudaMemcpy2D(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaMemcpy2D(dst, dpitch, src, spitch, width, height, kind)


cdef cudaError_t _cudaMemcpy2DToArray(cudaArray_t dst, size_t wOffset, size_t hOffset, const void* src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaMemcpy2DToArray(dst, wOffset, hOffset, src, spitch, width, height, kind)


cdef cudaError_t _cudaMemcpy2DFromArray(void* dst, size_t dpitch, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t width, size_t height, cudaMemcpyKind kind) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaMemcpy2DFromArray(dst, dpitch, src, wOffset, hOffset, width, height, kind)


cdef cudaError_t _cudaMemcpy2DArrayToArray(cudaArray_t dst, size_t wOffsetDst, size_t hOffsetDst, cudaArray_const_t src, size_t wOffsetSrc, size_t hOffsetSrc, size_t width, size_t height, cudaMemcpyKind kind) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaMemcpy2DArrayToArray(dst, wOffsetDst, hOffsetDst, src, wOffsetSrc, hOffsetSrc, width, height, kind)


cdef cudaError_t _cudaMemcpyAsync(void* dst, const void* src, size_t count, cudaMemcpyKind kind, cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaMemcpyAsync(dst, src, count, kind, stream)


cdef cudaError_t _cudaMemcpyPeerAsync(void* dst, int dstDevice, const void* src, int srcDevice, size_t count, cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaMemcpyPeerAsync(dst, dstDevice, src, srcDevice, count, stream)


cdef cudaError_t _cudaMemcpyBatchAsync(void** dsts, const void** srcs, const size_t* sizes, size_t count, cudaMemcpyAttributes* attrs, size_t* attrsIdxs, size_t numAttrs, cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaMemcpyBatchAsync(dsts, srcs, sizes, count, attrs, attrsIdxs, numAttrs, stream)


cdef cudaError_t _cudaMemcpy3DBatchAsync(size_t numOps, cudaMemcpy3DBatchOp* opList, unsigned long long flags, cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaMemcpy3DBatchAsync(numOps, opList, flags, stream)


cdef cudaError_t _cudaMemcpy2DAsync(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaMemcpy2DAsync(dst, dpitch, src, spitch, width, height, kind, stream)


cdef cudaError_t _cudaMemcpy2DToArrayAsync(cudaArray_t dst, size_t wOffset, size_t hOffset, const void* src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaMemcpy2DToArrayAsync(dst, wOffset, hOffset, src, spitch, width, height, kind, stream)


cdef cudaError_t _cudaMemcpy2DFromArrayAsync(void* dst, size_t dpitch, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaMemcpy2DFromArrayAsync(dst, dpitch, src, wOffset, hOffset, width, height, kind, stream)


cdef cudaError_t _cudaMemset(void* devPtr, int value, size_t count) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaMemset(devPtr, value, count)


cdef cudaError_t _cudaMemset2D(void* devPtr, size_t pitch, int value, size_t width, size_t height) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaMemset2D(devPtr, pitch, value, width, height)


cdef cudaError_t _cudaMemset3D(cudaPitchedPtr pitchedDevPtr, int value, cudaExtent extent) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaMemset3D(pitchedDevPtr, value, extent)


cdef cudaError_t _cudaMemsetAsync(void* devPtr, int value, size_t count, cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaMemsetAsync(devPtr, value, count, stream)


cdef cudaError_t _cudaMemset2DAsync(void* devPtr, size_t pitch, int value, size_t width, size_t height, cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaMemset2DAsync(devPtr, pitch, value, width, height, stream)


cdef cudaError_t _cudaMemset3DAsync(cudaPitchedPtr pitchedDevPtr, int value, cudaExtent extent, cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaMemset3DAsync(pitchedDevPtr, value, extent, stream)


cdef cudaError_t _cudaMemPrefetchAsync(const void* devPtr, size_t count, cudaMemLocation location, unsigned int flags, cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaMemPrefetchAsync(devPtr, count, location, flags, stream)


cdef cudaError_t _cudaMemAdvise(const void* devPtr, size_t count, cudaMemoryAdvise advice, cudaMemLocation location) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaMemAdvise(devPtr, count, advice, location)


cdef cudaError_t _cudaMemRangeGetAttribute(void* data, size_t dataSize, cudaMemRangeAttribute attribute, const void* devPtr, size_t count) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaMemRangeGetAttribute(data, dataSize, attribute, devPtr, count)


cdef cudaError_t _cudaMemRangeGetAttributes(void** data, size_t* dataSizes, cudaMemRangeAttribute* attributes, size_t numAttributes, const void* devPtr, size_t count) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaMemRangeGetAttributes(data, dataSizes, attributes, numAttributes, devPtr, count)


cdef cudaError_t _cudaMemcpyToArray(cudaArray_t dst, size_t wOffset, size_t hOffset, const void* src, size_t count, cudaMemcpyKind kind) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaMemcpyToArray(dst, wOffset, hOffset, src, count, kind)


cdef cudaError_t _cudaMemcpyFromArray(void* dst, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t count, cudaMemcpyKind kind) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaMemcpyFromArray(dst, src, wOffset, hOffset, count, kind)


cdef cudaError_t _cudaMemcpyArrayToArray(cudaArray_t dst, size_t wOffsetDst, size_t hOffsetDst, cudaArray_const_t src, size_t wOffsetSrc, size_t hOffsetSrc, size_t count, cudaMemcpyKind kind) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaMemcpyArrayToArray(dst, wOffsetDst, hOffsetDst, src, wOffsetSrc, hOffsetSrc, count, kind)


cdef cudaError_t _cudaMemcpyToArrayAsync(cudaArray_t dst, size_t wOffset, size_t hOffset, const void* src, size_t count, cudaMemcpyKind kind, cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaMemcpyToArrayAsync(dst, wOffset, hOffset, src, count, kind, stream)


cdef cudaError_t _cudaMemcpyFromArrayAsync(void* dst, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t count, cudaMemcpyKind kind, cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaMemcpyFromArrayAsync(dst, src, wOffset, hOffset, count, kind, stream)


cdef cudaError_t _cudaMallocAsync(void** devPtr, size_t size, cudaStream_t hStream) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaMallocAsync(devPtr, size, hStream)


cdef cudaError_t _cudaFreeAsync(void* devPtr, cudaStream_t hStream) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaFreeAsync(devPtr, hStream)


cdef cudaError_t _cudaMemPoolTrimTo(cudaMemPool_t memPool, size_t minBytesToKeep) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaMemPoolTrimTo(memPool, minBytesToKeep)


cdef cudaError_t _cudaMemPoolSetAttribute(cudaMemPool_t memPool, cudaMemPoolAttr attr, void* value) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaMemPoolSetAttribute(memPool, attr, value)


cdef cudaError_t _cudaMemPoolGetAttribute(cudaMemPool_t memPool, cudaMemPoolAttr attr, void* value) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaMemPoolGetAttribute(memPool, attr, value)


cdef cudaError_t _cudaMemPoolSetAccess(cudaMemPool_t memPool, const cudaMemAccessDesc* descList, size_t count) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaMemPoolSetAccess(memPool, descList, count)


cdef cudaError_t _cudaMemPoolGetAccess(cudaMemAccessFlags* flags, cudaMemPool_t memPool, cudaMemLocation* location) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaMemPoolGetAccess(flags, memPool, location)


cdef cudaError_t _cudaMemPoolCreate(cudaMemPool_t* memPool, const cudaMemPoolProps* poolProps) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaMemPoolCreate(memPool, poolProps)


cdef cudaError_t _cudaMemPoolDestroy(cudaMemPool_t memPool) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaMemPoolDestroy(memPool)


cdef cudaError_t _cudaMallocFromPoolAsync(void** ptr, size_t size, cudaMemPool_t memPool, cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaMallocFromPoolAsync(ptr, size, memPool, stream)


cdef cudaError_t _cudaMemPoolExportToShareableHandle(void* shareableHandle, cudaMemPool_t memPool, cudaMemAllocationHandleType handleType, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaMemPoolExportToShareableHandle(shareableHandle, memPool, handleType, flags)


cdef cudaError_t _cudaMemPoolImportFromShareableHandle(cudaMemPool_t* memPool, void* shareableHandle, cudaMemAllocationHandleType handleType, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaMemPoolImportFromShareableHandle(memPool, shareableHandle, handleType, flags)


cdef cudaError_t _cudaMemPoolExportPointer(cudaMemPoolPtrExportData* exportData, void* ptr) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaMemPoolExportPointer(exportData, ptr)


cdef cudaError_t _cudaMemPoolImportPointer(void** ptr, cudaMemPool_t memPool, cudaMemPoolPtrExportData* exportData) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaMemPoolImportPointer(ptr, memPool, exportData)


cdef cudaError_t _cudaPointerGetAttributes(cudaPointerAttributes* attributes, const void* ptr) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaPointerGetAttributes(attributes, ptr)


cdef cudaError_t _cudaDeviceCanAccessPeer(int* canAccessPeer, int device, int peerDevice) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaDeviceCanAccessPeer(canAccessPeer, device, peerDevice)


cdef cudaError_t _cudaDeviceEnablePeerAccess(int peerDevice, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaDeviceEnablePeerAccess(peerDevice, flags)


cdef cudaError_t _cudaDeviceDisablePeerAccess(int peerDevice) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaDeviceDisablePeerAccess(peerDevice)


cdef cudaError_t _cudaGraphicsUnregisterResource(cudaGraphicsResource_t resource) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaGraphicsUnregisterResource(resource)


cdef cudaError_t _cudaGraphicsResourceSetMapFlags(cudaGraphicsResource_t resource, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaGraphicsResourceSetMapFlags(resource, flags)


cdef cudaError_t _cudaGraphicsMapResources(int count, cudaGraphicsResource_t* resources, cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaGraphicsMapResources(count, resources, stream)


cdef cudaError_t _cudaGraphicsUnmapResources(int count, cudaGraphicsResource_t* resources, cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaGraphicsUnmapResources(count, resources, stream)


cdef cudaError_t _cudaGraphicsResourceGetMappedPointer(void** devPtr, size_t* size, cudaGraphicsResource_t resource) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaGraphicsResourceGetMappedPointer(devPtr, size, resource)


cdef cudaError_t _cudaGraphicsSubResourceGetMappedArray(cudaArray_t* array, cudaGraphicsResource_t resource, unsigned int arrayIndex, unsigned int mipLevel) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaGraphicsSubResourceGetMappedArray(array, resource, arrayIndex, mipLevel)


cdef cudaError_t _cudaGraphicsResourceGetMappedMipmappedArray(cudaMipmappedArray_t* mipmappedArray, cudaGraphicsResource_t resource) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaGraphicsResourceGetMappedMipmappedArray(mipmappedArray, resource)


cdef cudaError_t _cudaGetChannelDesc(cudaChannelFormatDesc* desc, cudaArray_const_t array) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaGetChannelDesc(desc, array)


cdef cudaChannelFormatDesc _cudaCreateChannelDesc(int x, int y, int z, int w, cudaChannelFormatKind f) except* nogil:
    return _static_cudaCreateChannelDesc(x, y, z, w, f)


cdef cudaError_t _cudaCreateTextureObject(cudaTextureObject_t* pTexObject, const cudaResourceDesc* pResDesc, const cudaTextureDesc* pTexDesc, const cudaResourceViewDesc* pResViewDesc) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaCreateTextureObject(pTexObject, pResDesc, pTexDesc, pResViewDesc)


cdef cudaError_t _cudaDestroyTextureObject(cudaTextureObject_t texObject) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaDestroyTextureObject(texObject)


cdef cudaError_t _cudaGetTextureObjectResourceDesc(cudaResourceDesc* pResDesc, cudaTextureObject_t texObject) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaGetTextureObjectResourceDesc(pResDesc, texObject)


cdef cudaError_t _cudaGetTextureObjectTextureDesc(cudaTextureDesc* pTexDesc, cudaTextureObject_t texObject) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaGetTextureObjectTextureDesc(pTexDesc, texObject)


cdef cudaError_t _cudaGetTextureObjectResourceViewDesc(cudaResourceViewDesc* pResViewDesc, cudaTextureObject_t texObject) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaGetTextureObjectResourceViewDesc(pResViewDesc, texObject)


cdef cudaError_t _cudaCreateSurfaceObject(cudaSurfaceObject_t* pSurfObject, const cudaResourceDesc* pResDesc) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaCreateSurfaceObject(pSurfObject, pResDesc)


cdef cudaError_t _cudaDestroySurfaceObject(cudaSurfaceObject_t surfObject) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaDestroySurfaceObject(surfObject)


cdef cudaError_t _cudaGetSurfaceObjectResourceDesc(cudaResourceDesc* pResDesc, cudaSurfaceObject_t surfObject) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaGetSurfaceObjectResourceDesc(pResDesc, surfObject)


cdef cudaError_t _cudaDriverGetVersion(int* driverVersion) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaDriverGetVersion(driverVersion)


cdef cudaError_t _cudaRuntimeGetVersion(int* runtimeVersion) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaRuntimeGetVersion(runtimeVersion)


cdef cudaError_t _cudaGraphCreate(cudaGraph_t* pGraph, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaGraphCreate(pGraph, flags)


cdef cudaError_t _cudaGraphAddKernelNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, const cudaKernelNodeParams* pNodeParams) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaGraphAddKernelNode(pGraphNode, graph, pDependencies, numDependencies, pNodeParams)


cdef cudaError_t _cudaGraphKernelNodeGetParams(cudaGraphNode_t node, cudaKernelNodeParams* pNodeParams) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaGraphKernelNodeGetParams(node, pNodeParams)


cdef cudaError_t _cudaGraphKernelNodeSetParams(cudaGraphNode_t node, const cudaKernelNodeParams* pNodeParams) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaGraphKernelNodeSetParams(node, pNodeParams)


cdef cudaError_t _cudaGraphKernelNodeCopyAttributes(cudaGraphNode_t hDst, cudaGraphNode_t hSrc) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaGraphKernelNodeCopyAttributes(hDst, hSrc)


cdef cudaError_t _cudaGraphKernelNodeGetAttribute(cudaGraphNode_t hNode, cudaLaunchAttributeID attr, cudaLaunchAttributeValue* value_out) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaGraphKernelNodeGetAttribute(hNode, attr, value_out)


cdef cudaError_t _cudaGraphKernelNodeSetAttribute(cudaGraphNode_t hNode, cudaLaunchAttributeID attr, const cudaLaunchAttributeValue* value) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaGraphKernelNodeSetAttribute(hNode, attr, value)


cdef cudaError_t _cudaGraphAddMemcpyNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, const cudaMemcpy3DParms* pCopyParams) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaGraphAddMemcpyNode(pGraphNode, graph, pDependencies, numDependencies, pCopyParams)


cdef cudaError_t _cudaGraphAddMemcpyNode1D(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, void* dst, const void* src, size_t count, cudaMemcpyKind kind) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaGraphAddMemcpyNode1D(pGraphNode, graph, pDependencies, numDependencies, dst, src, count, kind)


cdef cudaError_t _cudaGraphMemcpyNodeGetParams(cudaGraphNode_t node, cudaMemcpy3DParms* pNodeParams) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaGraphMemcpyNodeGetParams(node, pNodeParams)


cdef cudaError_t _cudaGraphMemcpyNodeSetParams(cudaGraphNode_t node, const cudaMemcpy3DParms* pNodeParams) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaGraphMemcpyNodeSetParams(node, pNodeParams)


cdef cudaError_t _cudaGraphMemcpyNodeSetParams1D(cudaGraphNode_t node, void* dst, const void* src, size_t count, cudaMemcpyKind kind) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaGraphMemcpyNodeSetParams1D(node, dst, src, count, kind)


cdef cudaError_t _cudaGraphAddMemsetNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, const cudaMemsetParams* pMemsetParams) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaGraphAddMemsetNode(pGraphNode, graph, pDependencies, numDependencies, pMemsetParams)


cdef cudaError_t _cudaGraphMemsetNodeGetParams(cudaGraphNode_t node, cudaMemsetParams* pNodeParams) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaGraphMemsetNodeGetParams(node, pNodeParams)


cdef cudaError_t _cudaGraphMemsetNodeSetParams(cudaGraphNode_t node, const cudaMemsetParams* pNodeParams) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaGraphMemsetNodeSetParams(node, pNodeParams)


cdef cudaError_t _cudaGraphAddHostNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, const cudaHostNodeParams* pNodeParams) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaGraphAddHostNode(pGraphNode, graph, pDependencies, numDependencies, pNodeParams)


cdef cudaError_t _cudaGraphHostNodeGetParams(cudaGraphNode_t node, cudaHostNodeParams* pNodeParams) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaGraphHostNodeGetParams(node, pNodeParams)


cdef cudaError_t _cudaGraphHostNodeSetParams(cudaGraphNode_t node, const cudaHostNodeParams* pNodeParams) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaGraphHostNodeSetParams(node, pNodeParams)


cdef cudaError_t _cudaGraphAddChildGraphNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, cudaGraph_t childGraph) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaGraphAddChildGraphNode(pGraphNode, graph, pDependencies, numDependencies, childGraph)


cdef cudaError_t _cudaGraphChildGraphNodeGetGraph(cudaGraphNode_t node, cudaGraph_t* pGraph) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaGraphChildGraphNodeGetGraph(node, pGraph)


cdef cudaError_t _cudaGraphAddEmptyNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaGraphAddEmptyNode(pGraphNode, graph, pDependencies, numDependencies)


cdef cudaError_t _cudaGraphAddEventRecordNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, cudaEvent_t event) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaGraphAddEventRecordNode(pGraphNode, graph, pDependencies, numDependencies, event)


cdef cudaError_t _cudaGraphEventRecordNodeGetEvent(cudaGraphNode_t node, cudaEvent_t* event_out) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaGraphEventRecordNodeGetEvent(node, event_out)


cdef cudaError_t _cudaGraphEventRecordNodeSetEvent(cudaGraphNode_t node, cudaEvent_t event) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaGraphEventRecordNodeSetEvent(node, event)


cdef cudaError_t _cudaGraphAddEventWaitNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, cudaEvent_t event) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaGraphAddEventWaitNode(pGraphNode, graph, pDependencies, numDependencies, event)


cdef cudaError_t _cudaGraphEventWaitNodeGetEvent(cudaGraphNode_t node, cudaEvent_t* event_out) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaGraphEventWaitNodeGetEvent(node, event_out)


cdef cudaError_t _cudaGraphEventWaitNodeSetEvent(cudaGraphNode_t node, cudaEvent_t event) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaGraphEventWaitNodeSetEvent(node, event)


cdef cudaError_t _cudaGraphAddExternalSemaphoresSignalNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, const cudaExternalSemaphoreSignalNodeParams* nodeParams) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaGraphAddExternalSemaphoresSignalNode(pGraphNode, graph, pDependencies, numDependencies, nodeParams)


cdef cudaError_t _cudaGraphExternalSemaphoresSignalNodeGetParams(cudaGraphNode_t hNode, cudaExternalSemaphoreSignalNodeParams* params_out) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaGraphExternalSemaphoresSignalNodeGetParams(hNode, params_out)


cdef cudaError_t _cudaGraphExternalSemaphoresSignalNodeSetParams(cudaGraphNode_t hNode, const cudaExternalSemaphoreSignalNodeParams* nodeParams) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaGraphExternalSemaphoresSignalNodeSetParams(hNode, nodeParams)


cdef cudaError_t _cudaGraphAddExternalSemaphoresWaitNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, const cudaExternalSemaphoreWaitNodeParams* nodeParams) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaGraphAddExternalSemaphoresWaitNode(pGraphNode, graph, pDependencies, numDependencies, nodeParams)


cdef cudaError_t _cudaGraphExternalSemaphoresWaitNodeGetParams(cudaGraphNode_t hNode, cudaExternalSemaphoreWaitNodeParams* params_out) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaGraphExternalSemaphoresWaitNodeGetParams(hNode, params_out)


cdef cudaError_t _cudaGraphExternalSemaphoresWaitNodeSetParams(cudaGraphNode_t hNode, const cudaExternalSemaphoreWaitNodeParams* nodeParams) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaGraphExternalSemaphoresWaitNodeSetParams(hNode, nodeParams)


cdef cudaError_t _cudaGraphAddMemAllocNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, cudaMemAllocNodeParams* nodeParams) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaGraphAddMemAllocNode(pGraphNode, graph, pDependencies, numDependencies, nodeParams)


cdef cudaError_t _cudaGraphMemAllocNodeGetParams(cudaGraphNode_t node, cudaMemAllocNodeParams* params_out) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaGraphMemAllocNodeGetParams(node, params_out)


cdef cudaError_t _cudaGraphAddMemFreeNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, void* dptr) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaGraphAddMemFreeNode(pGraphNode, graph, pDependencies, numDependencies, dptr)


cdef cudaError_t _cudaGraphMemFreeNodeGetParams(cudaGraphNode_t node, void* dptr_out) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaGraphMemFreeNodeGetParams(node, dptr_out)


cdef cudaError_t _cudaDeviceGraphMemTrim(int device) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaDeviceGraphMemTrim(device)


cdef cudaError_t _cudaDeviceGetGraphMemAttribute(int device, cudaGraphMemAttributeType attr, void* value) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaDeviceGetGraphMemAttribute(device, attr, value)


cdef cudaError_t _cudaDeviceSetGraphMemAttribute(int device, cudaGraphMemAttributeType attr, void* value) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaDeviceSetGraphMemAttribute(device, attr, value)


cdef cudaError_t _cudaGraphClone(cudaGraph_t* pGraphClone, cudaGraph_t originalGraph) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaGraphClone(pGraphClone, originalGraph)


cdef cudaError_t _cudaGraphNodeFindInClone(cudaGraphNode_t* pNode, cudaGraphNode_t originalNode, cudaGraph_t clonedGraph) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaGraphNodeFindInClone(pNode, originalNode, clonedGraph)


cdef cudaError_t _cudaGraphNodeGetType(cudaGraphNode_t node, cudaGraphNodeType* pType) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaGraphNodeGetType(node, pType)


cdef cudaError_t _cudaGraphGetNodes(cudaGraph_t graph, cudaGraphNode_t* nodes, size_t* numNodes) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaGraphGetNodes(graph, nodes, numNodes)


cdef cudaError_t _cudaGraphGetRootNodes(cudaGraph_t graph, cudaGraphNode_t* pRootNodes, size_t* pNumRootNodes) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaGraphGetRootNodes(graph, pRootNodes, pNumRootNodes)


cdef cudaError_t _cudaGraphGetEdges(cudaGraph_t graph, cudaGraphNode_t* from_, cudaGraphNode_t* to, cudaGraphEdgeData* edgeData, size_t* numEdges) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaGraphGetEdges(graph, from_, to, edgeData, numEdges)


cdef cudaError_t _cudaGraphNodeGetDependencies(cudaGraphNode_t node, cudaGraphNode_t* pDependencies, cudaGraphEdgeData* edgeData, size_t* pNumDependencies) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaGraphNodeGetDependencies(node, pDependencies, edgeData, pNumDependencies)


cdef cudaError_t _cudaGraphNodeGetDependentNodes(cudaGraphNode_t node, cudaGraphNode_t* pDependentNodes, cudaGraphEdgeData* edgeData, size_t* pNumDependentNodes) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaGraphNodeGetDependentNodes(node, pDependentNodes, edgeData, pNumDependentNodes)


cdef cudaError_t _cudaGraphAddDependencies(cudaGraph_t graph, const cudaGraphNode_t* from_, const cudaGraphNode_t* to, const cudaGraphEdgeData* edgeData, size_t numDependencies) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaGraphAddDependencies(graph, from_, to, edgeData, numDependencies)


cdef cudaError_t _cudaGraphRemoveDependencies(cudaGraph_t graph, const cudaGraphNode_t* from_, const cudaGraphNode_t* to, const cudaGraphEdgeData* edgeData, size_t numDependencies) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaGraphRemoveDependencies(graph, from_, to, edgeData, numDependencies)


cdef cudaError_t _cudaGraphDestroyNode(cudaGraphNode_t node) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaGraphDestroyNode(node)


cdef cudaError_t _cudaGraphInstantiate(cudaGraphExec_t* pGraphExec, cudaGraph_t graph, unsigned long long flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaGraphInstantiate(pGraphExec, graph, flags)


cdef cudaError_t _cudaGraphInstantiateWithFlags(cudaGraphExec_t* pGraphExec, cudaGraph_t graph, unsigned long long flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaGraphInstantiateWithFlags(pGraphExec, graph, flags)


cdef cudaError_t _cudaGraphInstantiateWithParams(cudaGraphExec_t* pGraphExec, cudaGraph_t graph, cudaGraphInstantiateParams* instantiateParams) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaGraphInstantiateWithParams(pGraphExec, graph, instantiateParams)


cdef cudaError_t _cudaGraphExecGetFlags(cudaGraphExec_t graphExec, unsigned long long* flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaGraphExecGetFlags(graphExec, flags)


cdef cudaError_t _cudaGraphExecKernelNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, const cudaKernelNodeParams* pNodeParams) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaGraphExecKernelNodeSetParams(hGraphExec, node, pNodeParams)


cdef cudaError_t _cudaGraphExecMemcpyNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, const cudaMemcpy3DParms* pNodeParams) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaGraphExecMemcpyNodeSetParams(hGraphExec, node, pNodeParams)


cdef cudaError_t _cudaGraphExecMemcpyNodeSetParams1D(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, void* dst, const void* src, size_t count, cudaMemcpyKind kind) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaGraphExecMemcpyNodeSetParams1D(hGraphExec, node, dst, src, count, kind)


cdef cudaError_t _cudaGraphExecMemsetNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, const cudaMemsetParams* pNodeParams) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaGraphExecMemsetNodeSetParams(hGraphExec, node, pNodeParams)


cdef cudaError_t _cudaGraphExecHostNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, const cudaHostNodeParams* pNodeParams) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaGraphExecHostNodeSetParams(hGraphExec, node, pNodeParams)


cdef cudaError_t _cudaGraphExecChildGraphNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, cudaGraph_t childGraph) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaGraphExecChildGraphNodeSetParams(hGraphExec, node, childGraph)


cdef cudaError_t _cudaGraphExecEventRecordNodeSetEvent(cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, cudaEvent_t event) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaGraphExecEventRecordNodeSetEvent(hGraphExec, hNode, event)


cdef cudaError_t _cudaGraphExecEventWaitNodeSetEvent(cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, cudaEvent_t event) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaGraphExecEventWaitNodeSetEvent(hGraphExec, hNode, event)


cdef cudaError_t _cudaGraphExecExternalSemaphoresSignalNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, const cudaExternalSemaphoreSignalNodeParams* nodeParams) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaGraphExecExternalSemaphoresSignalNodeSetParams(hGraphExec, hNode, nodeParams)


cdef cudaError_t _cudaGraphExecExternalSemaphoresWaitNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, const cudaExternalSemaphoreWaitNodeParams* nodeParams) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaGraphExecExternalSemaphoresWaitNodeSetParams(hGraphExec, hNode, nodeParams)


cdef cudaError_t _cudaGraphNodeSetEnabled(cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, unsigned int isEnabled) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaGraphNodeSetEnabled(hGraphExec, hNode, isEnabled)


cdef cudaError_t _cudaGraphNodeGetEnabled(cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, unsigned int* isEnabled) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaGraphNodeGetEnabled(hGraphExec, hNode, isEnabled)


cdef cudaError_t _cudaGraphExecUpdate(cudaGraphExec_t hGraphExec, cudaGraph_t hGraph, cudaGraphExecUpdateResultInfo* resultInfo) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaGraphExecUpdate(hGraphExec, hGraph, resultInfo)


cdef cudaError_t _cudaGraphUpload(cudaGraphExec_t graphExec, cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaGraphUpload(graphExec, stream)


cdef cudaError_t _cudaGraphLaunch(cudaGraphExec_t graphExec, cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaGraphLaunch(graphExec, stream)


cdef cudaError_t _cudaGraphExecDestroy(cudaGraphExec_t graphExec) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaGraphExecDestroy(graphExec)


cdef cudaError_t _cudaGraphDestroy(cudaGraph_t graph) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaGraphDestroy(graph)


cdef cudaError_t _cudaGraphDebugDotPrint(cudaGraph_t graph, const char* path, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaGraphDebugDotPrint(graph, path, flags)


cdef cudaError_t _cudaUserObjectCreate(cudaUserObject_t* object_out, void* ptr, cudaHostFn_t destroy, unsigned int initialRefcount, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaUserObjectCreate(object_out, ptr, destroy, initialRefcount, flags)


cdef cudaError_t _cudaUserObjectRetain(cudaUserObject_t object, unsigned int count) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaUserObjectRetain(object, count)


cdef cudaError_t _cudaUserObjectRelease(cudaUserObject_t object, unsigned int count) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaUserObjectRelease(object, count)


cdef cudaError_t _cudaGraphRetainUserObject(cudaGraph_t graph, cudaUserObject_t object, unsigned int count, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaGraphRetainUserObject(graph, object, count, flags)


cdef cudaError_t _cudaGraphReleaseUserObject(cudaGraph_t graph, cudaUserObject_t object, unsigned int count) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaGraphReleaseUserObject(graph, object, count)


cdef cudaError_t _cudaGraphAddNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, const cudaGraphEdgeData* dependencyData, size_t numDependencies, cudaGraphNodeParams* nodeParams) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaGraphAddNode(pGraphNode, graph, pDependencies, dependencyData, numDependencies, nodeParams)


cdef cudaError_t _cudaGraphNodeSetParams(cudaGraphNode_t node, cudaGraphNodeParams* nodeParams) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaGraphNodeSetParams(node, nodeParams)


cdef cudaError_t _cudaGraphExecNodeSetParams(cudaGraphExec_t graphExec, cudaGraphNode_t node, cudaGraphNodeParams* nodeParams) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaGraphExecNodeSetParams(graphExec, node, nodeParams)


cdef cudaError_t _cudaGraphConditionalHandleCreate(cudaGraphConditionalHandle* pHandle_out, cudaGraph_t graph, unsigned int defaultLaunchValue, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaGraphConditionalHandleCreate(pHandle_out, graph, defaultLaunchValue, flags)


cdef cudaError_t _cudaGetDriverEntryPoint(const char* symbol, void** funcPtr, unsigned long long flags, cudaDriverEntryPointQueryResult* driverStatus) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaGetDriverEntryPoint(symbol, funcPtr, flags, driverStatus)


cdef cudaError_t _cudaGetDriverEntryPointByVersion(const char* symbol, void** funcPtr, unsigned int cudaVersion, unsigned long long flags, cudaDriverEntryPointQueryResult* driverStatus) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaGetDriverEntryPointByVersion(symbol, funcPtr, cudaVersion, flags, driverStatus)


cdef cudaError_t _cudaLibraryLoadData(cudaLibrary_t* library, const void* code, cudaJitOption* jitOptions, void** jitOptionsValues, unsigned int numJitOptions, cudaLibraryOption* libraryOptions, void** libraryOptionValues, unsigned int numLibraryOptions) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaLibraryLoadData(library, code, jitOptions, jitOptionsValues, numJitOptions, libraryOptions, libraryOptionValues, numLibraryOptions)


cdef cudaError_t _cudaLibraryLoadFromFile(cudaLibrary_t* library, const char* fileName, cudaJitOption* jitOptions, void** jitOptionsValues, unsigned int numJitOptions, cudaLibraryOption* libraryOptions, void** libraryOptionValues, unsigned int numLibraryOptions) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaLibraryLoadFromFile(library, fileName, jitOptions, jitOptionsValues, numJitOptions, libraryOptions, libraryOptionValues, numLibraryOptions)


cdef cudaError_t _cudaLibraryUnload(cudaLibrary_t library) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaLibraryUnload(library)


cdef cudaError_t _cudaLibraryGetKernel(cudaKernel_t* pKernel, cudaLibrary_t library, const char* name) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaLibraryGetKernel(pKernel, library, name)


cdef cudaError_t _cudaLibraryGetGlobal(void** dptr, size_t* bytes, cudaLibrary_t library, const char* name) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaLibraryGetGlobal(dptr, bytes, library, name)


cdef cudaError_t _cudaLibraryGetManaged(void** dptr, size_t* bytes, cudaLibrary_t library, const char* name) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaLibraryGetManaged(dptr, bytes, library, name)


cdef cudaError_t _cudaLibraryGetUnifiedFunction(void** fptr, cudaLibrary_t library, const char* symbol) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaLibraryGetUnifiedFunction(fptr, library, symbol)


cdef cudaError_t _cudaLibraryGetKernelCount(unsigned int* count, cudaLibrary_t lib) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaLibraryGetKernelCount(count, lib)


cdef cudaError_t _cudaLibraryEnumerateKernels(cudaKernel_t* kernels, unsigned int numKernels, cudaLibrary_t lib) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaLibraryEnumerateKernels(kernels, numKernels, lib)


cdef cudaError_t _cudaKernelSetAttributeForDevice(cudaKernel_t kernel, cudaFuncAttribute attr, int value, int device) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaKernelSetAttributeForDevice(kernel, attr, value, device)


cdef cudaError_t _cudaGetExportTable(const void** ppExportTable, const cudaUUID_t* pExportTableId) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaGetExportTable(ppExportTable, pExportTableId)


cdef cudaError_t _cudaGetKernel(cudaKernel_t* kernelPtr, const void* entryFuncAddr) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaGetKernel(kernelPtr, entryFuncAddr)


cdef cudaError_t _cudaProfilerStart() except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaProfilerStart()


cdef cudaError_t _cudaProfilerStop() except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaProfilerStop()


cdef cudaError_t _cudaGetDeviceProperties(cudaDeviceProp* prop, int device) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaGetDeviceProperties(prop, device)


cdef cudaError_t _cudaDeviceGetHostAtomicCapabilities(unsigned int* capabilities, const cudaAtomicOperation* operations, unsigned int count, int device) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaDeviceGetHostAtomicCapabilities(capabilities, operations, count, device)


cdef cudaError_t _cudaDeviceGetP2PAtomicCapabilities(unsigned int* capabilities, const cudaAtomicOperation* operations, unsigned int count, int srcDevice, int dstDevice) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaDeviceGetP2PAtomicCapabilities(capabilities, operations, count, srcDevice, dstDevice)


cdef cudaError_t _cudaStreamGetCaptureInfo(cudaStream_t stream, cudaStreamCaptureStatus* captureStatus_out, unsigned long long* id_out, cudaGraph_t* graph_out, const cudaGraphNode_t** dependencies_out, const cudaGraphEdgeData** edgeData_out, size_t* numDependencies_out) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaStreamGetCaptureInfo(stream, captureStatus_out, id_out, graph_out, dependencies_out, edgeData_out, numDependencies_out)


cdef cudaError_t _cudaSignalExternalSemaphoresAsync(const cudaExternalSemaphore_t* extSemArray, const cudaExternalSemaphoreSignalParams* paramsArray, unsigned int numExtSems, cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaSignalExternalSemaphoresAsync(extSemArray, paramsArray, numExtSems, stream)


cdef cudaError_t _cudaWaitExternalSemaphoresAsync(const cudaExternalSemaphore_t* extSemArray, const cudaExternalSemaphoreWaitParams* paramsArray, unsigned int numExtSems, cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaWaitExternalSemaphoresAsync(extSemArray, paramsArray, numExtSems, stream)


cdef cudaError_t _cudaMemPrefetchBatchAsync(void** dptrs, size_t* sizes, size_t count, cudaMemLocation* prefetchLocs, size_t* prefetchLocIdxs, size_t numPrefetchLocs, unsigned long long flags, cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaMemPrefetchBatchAsync(dptrs, sizes, count, prefetchLocs, prefetchLocIdxs, numPrefetchLocs, flags, stream)


cdef cudaError_t _cudaMemDiscardBatchAsync(void** dptrs, size_t* sizes, size_t count, unsigned long long flags, cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaMemDiscardBatchAsync(dptrs, sizes, count, flags, stream)


cdef cudaError_t _cudaMemDiscardAndPrefetchBatchAsync(void** dptrs, size_t* sizes, size_t count, cudaMemLocation* prefetchLocs, size_t* prefetchLocIdxs, size_t numPrefetchLocs, unsigned long long flags, cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaMemDiscardAndPrefetchBatchAsync(dptrs, sizes, count, prefetchLocs, prefetchLocIdxs, numPrefetchLocs, flags, stream)


cdef cudaError_t _cudaMemGetDefaultMemPool(cudaMemPool_t* memPool, cudaMemLocation* location, cudaMemAllocationType type) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaMemGetDefaultMemPool(memPool, location, type)


cdef cudaError_t _cudaMemGetMemPool(cudaMemPool_t* memPool, cudaMemLocation* location, cudaMemAllocationType type) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaMemGetMemPool(memPool, location, type)


cdef cudaError_t _cudaMemSetMemPool(cudaMemLocation* location, cudaMemAllocationType type, cudaMemPool_t memPool) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaMemSetMemPool(location, type, memPool)


cdef cudaError_t _cudaLogsRegisterCallback(cudaLogsCallback_t callbackFunc, void* userData, cudaLogsCallbackHandle* callback_out) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaLogsRegisterCallback(callbackFunc, userData, callback_out)


cdef cudaError_t _cudaLogsUnregisterCallback(cudaLogsCallbackHandle callback) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaLogsUnregisterCallback(callback)


cdef cudaError_t _cudaLogsCurrent(cudaLogIterator* iterator_out, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaLogsCurrent(iterator_out, flags)


cdef cudaError_t _cudaLogsDumpToFile(cudaLogIterator* iterator, const char* pathToFile, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaLogsDumpToFile(iterator, pathToFile, flags)


cdef cudaError_t _cudaLogsDumpToMemory(cudaLogIterator* iterator, char* buffer, size_t* size, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaLogsDumpToMemory(iterator, buffer, size, flags)


cdef cudaError_t _cudaGraphNodeGetContainingGraph(cudaGraphNode_t hNode, cudaGraph_t* phGraph) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaGraphNodeGetContainingGraph(hNode, phGraph)


cdef cudaError_t _cudaGraphNodeGetLocalId(cudaGraphNode_t hNode, unsigned int* nodeId) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaGraphNodeGetLocalId(hNode, nodeId)


cdef cudaError_t _cudaGraphNodeGetToolsId(cudaGraphNode_t hNode, unsigned long long* toolsNodeId) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaGraphNodeGetToolsId(hNode, toolsNodeId)


cdef cudaError_t _cudaGraphGetId(cudaGraph_t hGraph, unsigned int* graphID) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaGraphGetId(hGraph, graphID)


cdef cudaError_t _cudaGraphExecGetId(cudaGraphExec_t hGraphExec, unsigned int* graphID) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaGraphExecGetId(hGraphExec, graphID)


cdef cudaError_t _cudaGraphConditionalHandleCreate_v2(cudaGraphConditionalHandle* pHandle_out, cudaGraph_t graph, cudaExecutionContext_t ctx, unsigned int defaultLaunchValue, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaGraphConditionalHandleCreate_v2(pHandle_out, graph, ctx, defaultLaunchValue, flags)


cdef cudaError_t _cudaDeviceGetDevResource(int device, cudaDevResource* resource, cudaDevResourceType type) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaDeviceGetDevResource(device, resource, type)


cdef cudaError_t _cudaDevSmResourceSplitByCount(cudaDevResource* result, unsigned int* nbGroups, const cudaDevResource* input, cudaDevResource* remaining, unsigned int flags, unsigned int minCount) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaDevSmResourceSplitByCount(result, nbGroups, input, remaining, flags, minCount)


cdef cudaError_t _cudaDevSmResourceSplit(cudaDevResource* result, unsigned int nbGroups, const cudaDevResource* input, cudaDevResource* remainder, unsigned int flags, cudaDevSmResourceGroupParams* groupParams) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaDevSmResourceSplit(result, nbGroups, input, remainder, flags, groupParams)


cdef cudaError_t _cudaDevResourceGenerateDesc(cudaDevResourceDesc_t* phDesc, cudaDevResource* resources, unsigned int nbResources) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaDevResourceGenerateDesc(phDesc, resources, nbResources)


cdef cudaError_t _cudaGreenCtxCreate(cudaExecutionContext_t* phCtx, cudaDevResourceDesc_t desc, int device, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaGreenCtxCreate(phCtx, desc, device, flags)


cdef cudaError_t _cudaExecutionCtxDestroy(cudaExecutionContext_t ctx) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaExecutionCtxDestroy(ctx)


cdef cudaError_t _cudaExecutionCtxGetDevResource(cudaExecutionContext_t ctx, cudaDevResource* resource, cudaDevResourceType type) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaExecutionCtxGetDevResource(ctx, resource, type)


cdef cudaError_t _cudaExecutionCtxGetDevice(int* device, cudaExecutionContext_t ctx) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaExecutionCtxGetDevice(device, ctx)


cdef cudaError_t _cudaExecutionCtxGetId(cudaExecutionContext_t ctx, unsigned long long* ctxId) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaExecutionCtxGetId(ctx, ctxId)


cdef cudaError_t _cudaExecutionCtxStreamCreate(cudaStream_t* phStream, cudaExecutionContext_t ctx, unsigned int flags, int priority) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaExecutionCtxStreamCreate(phStream, ctx, flags, priority)


cdef cudaError_t _cudaExecutionCtxSynchronize(cudaExecutionContext_t ctx) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaExecutionCtxSynchronize(ctx)


cdef cudaError_t _cudaStreamGetDevResource(cudaStream_t hStream, cudaDevResource* resource, cudaDevResourceType type) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaStreamGetDevResource(hStream, resource, type)


cdef cudaError_t _cudaExecutionCtxRecordEvent(cudaExecutionContext_t ctx, cudaEvent_t event) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaExecutionCtxRecordEvent(ctx, event)


cdef cudaError_t _cudaExecutionCtxWaitEvent(cudaExecutionContext_t ctx, cudaEvent_t event) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaExecutionCtxWaitEvent(ctx, event)


cdef cudaError_t _cudaDeviceGetExecutionCtx(cudaExecutionContext_t* ctx, int device) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaDeviceGetExecutionCtx(ctx, device)


cdef cudaError_t _cudaFuncGetParamCount(const void* func, size_t* paramCount) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaFuncGetParamCount(func, paramCount)


cdef cudaError_t _cudaLaunchHostFunc_v2(cudaStream_t stream, cudaHostFn_t fn, void* userData, unsigned int syncMode) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaLaunchHostFunc_v2(stream, fn, userData, syncMode)


cdef cudaError_t _cudaMemcpyWithAttributesAsync(void* dst, const void* src, size_t size, cudaMemcpyAttributes* attr, cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaMemcpyWithAttributesAsync(dst, src, size, attr, stream)


cdef cudaError_t _cudaMemcpy3DWithAttributesAsync(cudaMemcpy3DBatchOp* op, unsigned long long flags, cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaMemcpy3DWithAttributesAsync(op, flags, stream)


cdef cudaError_t _cudaGraphNodeGetParams(cudaGraphNode_t node, cudaGraphNodeParams* nodeParams) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaGraphNodeGetParams(node, nodeParams)


cdef cudaError_t _cudaStreamBeginRecaptureToGraph(cudaStream_t stream, cudaStreamCaptureMode mode, cudaGraph_t graph, cudaGraphRecaptureCallbackData* callbackData) except ?cudaErrorCallRequiresNewerDriver nogil:
    return _static_cudaStreamBeginRecaptureToGraph(stream, mode, graph, callbackData)
