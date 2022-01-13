# Copyright 2021-2022 NVIDIA Corporation.  All rights reserved.
#
# Please refer to the NVIDIA end user license agreement (EULA) associated
# with this source code for terms and conditions that govern your use of
# this software. Any use, reproduction, disclosure, or distribution of
# this software and related documentation outside the terms of the EULA
# is strictly prohibited.
cimport cuda.ccuda as ccuda
from cuda.ccudart cimport *
from libc.stdlib cimport malloc, free, calloc
from libc.string cimport memset, memcpy, strncmp
from libcpp cimport bool

cdef cudaError_t _cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaStreamCreate(cudaStream_t* pStream) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaEventCreate(cudaEvent_t* event) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaEventQuery(cudaEvent_t event) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaChannelFormatDesc _cudaCreateChannelDesc(int x, int y, int z, int w, cudaChannelFormatKind f) nogil
cdef cudaError_t _cudaDriverGetVersion(int* driverVersion) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaRuntimeGetVersion(int* runtimeVersion) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaDeviceGetTexture1DLinearMaxWidth(size_t* maxWidthInElements, const cudaChannelFormatDesc* fmtDesc, int device) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaMallocHost(void** ptr, size_t size) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaMallocPitch(void** devPtr, size_t* pitch, size_t width, size_t height) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaMallocMipmappedArray(cudaMipmappedArray_t* mipmappedArray, const cudaChannelFormatDesc* desc, cudaExtent extent, unsigned int numLevels, unsigned int flags) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaMemcpy2D(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaMemcpy2DAsync(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaMemcpyAsync(void* dst, const void* src, size_t count, cudaMemcpyKind kind, cudaStream_t stream) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaGraphAddMemcpyNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, const cudaMemcpy3DParms* pCopyParams) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaGraphAddMemcpyNode1D(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, void* dst, const void* src, size_t count, cudaMemcpyKind kind) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaGraphMemcpyNodeSetParams1D(cudaGraphNode_t node, void* dst, const void* src, size_t count, cudaMemcpyKind kind) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaGraphExecMemcpyNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, const cudaMemcpy3DParms* pNodeParams) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaGraphExecMemcpyNodeSetParams1D(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, void* dst, const void* src, size_t count, cudaMemcpyKind kind) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaGetDriverEntryPoint(const char* symbol, void** funcPtr, unsigned long long flags) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaGraphAddMemsetNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, const cudaMemsetParams* pMemsetParams) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaGraphExecMemsetNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, const cudaMemsetParams* pNodeParams) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaGraphMemcpyNodeSetParams(cudaGraphNode_t node, const cudaMemcpy3DParms* pNodeParams) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaGraphMemcpyNodeGetParams(cudaGraphNode_t node, cudaMemcpy3DParms* p) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaFuncGetAttributes(cudaFuncAttributes* attr, const void* func) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaMallocArray(cudaArray_t* arrayPtr, const cudaChannelFormatDesc* desc, size_t width, size_t height, unsigned int flags) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaMalloc3D(cudaPitchedPtr* pitchedDevPtr, cudaExtent extent) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaMalloc3DArray(cudaArray_t* arrayPtr, const cudaChannelFormatDesc* desc, cudaExtent extent, unsigned int flags) nogil except ?cudaErrorCallRequiresNewerDriver
cdef const char* _cudaGetErrorString(cudaError_t error) nogil except ?NULL
cdef cudaError_t _cudaStreamAddCallback(cudaStream_t stream, cudaStreamCallback_t callback, void* userData, unsigned int flags) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaStreamGetCaptureInfo(cudaStream_t stream, cudaStreamCaptureStatus* captureStatus_out, unsigned long long* id_out) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaStreamGetCaptureInfo_v2(cudaStream_t stream, cudaStreamCaptureStatus* captureStatus_out, unsigned long long* id_out, cudaGraph_t* graph_out, const cudaGraphNode_t** dependencies_out, size_t* numDependencies_out) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaImportExternalSemaphore(cudaExternalSemaphore_t* extSem_out, const cudaExternalSemaphoreHandleDesc* semHandleDesc) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaSignalExternalSemaphoresAsync(const cudaExternalSemaphore_t* extSemArray, const cudaExternalSemaphoreSignalParams* paramsArray, unsigned int numExtSems, cudaStream_t stream) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaWaitExternalSemaphoresAsync(const cudaExternalSemaphore_t* extSemArray, const cudaExternalSemaphoreWaitParams* paramsArray, unsigned int numExtSems, cudaStream_t stream) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaArrayGetInfo(cudaChannelFormatDesc* desc, cudaExtent* extent, unsigned int* flags, cudaArray_t array) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaMemcpy2DToArray(cudaArray_t dst, size_t wOffset, size_t hOffset, const void* src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaMemcpy2DFromArray(void* dst, size_t dpitch, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t width, size_t height, cudaMemcpyKind kind) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaMemcpy2DArrayToArray(cudaArray_t dst, size_t wOffsetDst, size_t hOffsetDst, cudaArray_const_t src, size_t wOffsetSrc, size_t hOffsetSrc, size_t width, size_t height, cudaMemcpyKind kind) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaMemcpy2DToArrayAsync(cudaArray_t dst, size_t wOffset, size_t hOffset, const void* src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaMemcpy2DFromArrayAsync(void* dst, size_t dpitch, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaMemset3D(cudaPitchedPtr pitchedDevPtr, int value, cudaExtent extent) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaMemset3DAsync(cudaPitchedPtr pitchedDevPtr, int value, cudaExtent extent, cudaStream_t stream) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaMemcpyToArray(cudaArray_t dst, size_t wOffset, size_t hOffset, const void* src, size_t count, cudaMemcpyKind kind) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaMemcpyFromArray(void* dst, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t count, cudaMemcpyKind kind) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaMemcpyToArrayAsync(cudaArray_t dst, size_t wOffset, size_t hOffset, const void* src, size_t count, cudaMemcpyKind kind, cudaStream_t stream) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaMemcpyFromArrayAsync(void* dst, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t count, cudaMemcpyKind kind, cudaStream_t stream) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaPointerGetAttributes(cudaPointerAttributes* attributes, const void* ptr) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaGetDeviceFlags(unsigned int* flags) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaMemcpy3D(const cudaMemcpy3DParms* p) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaMemcpy3DAsync(const cudaMemcpy3DParms* p, cudaStream_t stream) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaMemPoolSetAccess(cudaMemPool_t memPool, const cudaMemAccessDesc* descList, size_t count) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaDeviceReset() nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaThreadExit() nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaGetLastError() nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaPeekAtLastError() nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaGetDevice(int* device) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaSetDevice(int device) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaGetDeviceProperties(cudaDeviceProp* prop, int device) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaChooseDevice(int* device, const cudaDeviceProp* prop) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaMemcpyArrayToArray(cudaArray_t dst, size_t wOffsetDst, size_t hOffsetDst, cudaArray_const_t src, size_t wOffsetSrc, size_t hOffsetSrc, size_t count, cudaMemcpyKind kind) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaGetChannelDesc(cudaChannelFormatDesc* desc, cudaArray_const_t array) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaCreateTextureObject(cudaTextureObject_t* pTexObject, const cudaResourceDesc* pResDesc, const cudaTextureDesc* pTexDesc, const cudaResourceViewDesc* pResViewDesc) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaGetTextureObjectTextureDesc(cudaTextureDesc* pTexDesc, cudaTextureObject_t texObject) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaGetTextureObjectResourceViewDesc(cudaResourceViewDesc* pResViewDesc, cudaTextureObject_t texObject) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaGetExportTable(const void** ppExportTable, cudaUUID_t* pExportTableId) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaMemcpy3DPeer(const cudaMemcpy3DPeerParms* p) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaMemcpy3DPeerAsync(const cudaMemcpy3DPeerParms* p, cudaStream_t stream) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaPitchedPtr _make_cudaPitchedPtr(void* d, size_t p, size_t xsz, size_t ysz) nogil
cdef cudaPos _make_cudaPos(size_t x, size_t y, size_t z) nogil
cdef cudaExtent _make_cudaExtent(size_t w, size_t h, size_t d) nogil
cdef cudaError_t _cudaSetDoubleForDevice(double* d) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaSetDoubleForHost(double* d) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaSetDeviceFlags(unsigned int flags) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaGraphAddMemAllocNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, cudaMemAllocNodeParams* nodeParams) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaGraphMemAllocNodeGetParams(cudaGraphNode_t node, cudaMemAllocNodeParams* params_out) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaGraphMemFreeNodeGetParams(cudaGraphNode_t node, void* dptr_out) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaMemAdvise(const void* devPtr, size_t count, cudaMemoryAdvise advice, int device) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaMemRangeGetAttribute(void* data, size_t dataSize, cudaMemRangeAttribute attribute, const void* devPtr, size_t count) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaMemRangeGetAttributes(void** data, size_t* dataSizes, cudaMemRangeAttribute* attributes, size_t numAttributes, const void* devPtr, size_t count) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaGetDeviceCount(int* count) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaDeviceGetAttribute(int* value, cudaDeviceAttr attr, int device) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaDeviceSetSharedMemConfig(cudaSharedMemConfig config) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaDeviceGetByPCIBusId(int* device, const char* pciBusId) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaDeviceGetPCIBusId(char* pciBusId, int length, int device) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaDeviceGetP2PAttribute(int* value, cudaDeviceP2PAttr attr, int srcDevice, int dstDevice) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaArrayGetSparseProperties(cudaArraySparseProperties* sparseProperties, cudaArray_t array) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaMipmappedArrayGetSparseProperties(cudaArraySparseProperties* sparseProperties, cudaMipmappedArray_t mipmap) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaDeviceCanAccessPeer(int* canAccessPeer, int device, int peerDevice) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaMemcpyPeer(void* dst, int dstDevice, const void* src, int srcDevice, size_t count) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaMemcpyPeerAsync(void* dst, int dstDevice, const void* src, int srcDevice, size_t count, cudaStream_t stream) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaDeviceEnablePeerAccess(int peerDevice, unsigned int flags) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaDeviceDisablePeerAccess(int peerDevice) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaExternalMemoryGetMappedMipmappedArray(cudaMipmappedArray_t* mipmap, cudaExternalMemory_t extMem, const cudaExternalMemoryMipmappedArrayDesc* mipmapDesc) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaGetSurfaceObjectResourceDesc(cudaResourceDesc* pResDesc, cudaSurfaceObject_t surfObject) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaGraphKernelNodeGetParams(cudaGraphNode_t node, cudaKernelNodeParams* pNodeParams) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaExternalMemoryGetMappedBuffer(void** devPtr, cudaExternalMemory_t extMem, const cudaExternalMemoryBufferDesc* bufferDesc) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaImportExternalMemory(cudaExternalMemory_t* extMem_out, const cudaExternalMemoryHandleDesc* memHandleDesc) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaCreateSurfaceObject(cudaSurfaceObject_t* pSurfObject, const cudaResourceDesc* pResDesc) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaGetTextureObjectResourceDesc(cudaResourceDesc* pResDesc, cudaTextureObject_t texObject) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaProfilerInitialize(const char* configFile, const char* outputFile, cudaOutputMode_t outputMode) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaGraphicsEGLRegisterImage(cudaGraphicsResource_t* pCudaResource, EGLImageKHR image, unsigned int flags) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaEGLStreamProducerPresentFrame(cudaEglStreamConnection* conn, cudaEglFrame eglframe, cudaStream_t* pStream) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaEGLStreamProducerReturnFrame(cudaEglStreamConnection* conn, cudaEglFrame* eglframe, cudaStream_t* pStream) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaGraphicsResourceGetMappedEglFrame(cudaEglFrame* eglFrame, cudaGraphicsResource_t resource, unsigned int index, unsigned int mipLevel) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaVDPAUSetVDPAUDevice(int device, VdpDevice vdpDevice, VdpGetProcAddress* vdpGetProcAddress) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaArrayGetMemoryRequirements(cudaArrayMemoryRequirements* memoryRequirements, cudaArray_t array, int device) nogil except ?cudaErrorCallRequiresNewerDriver
cdef cudaError_t _cudaMipmappedArrayGetMemoryRequirements(cudaArrayMemoryRequirements* memoryRequirements, cudaMipmappedArray_t mipmap, int device) nogil except ?cudaErrorCallRequiresNewerDriver
