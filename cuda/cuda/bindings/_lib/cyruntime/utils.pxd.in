# Copyright 2021-2024 NVIDIA Corporation.  All rights reserved.
#
# Please refer to the NVIDIA end user license agreement (EULA) associated
# with this source code for terms and conditions that govern your use of
# this software. Any use, reproduction, disclosure, or distribution of
# this software and related documentation outside the terms of the EULA
# is strictly prohibited.
from cuda.bindings.cyruntime cimport *
from libc.stdlib cimport malloc, free, calloc
from libc.string cimport memset, memcpy, strncmp
from libcpp cimport bool
cimport cuda.bindings._bindings.cydriver as cydriver

cdef struct cudaPythonDevice:
    cydriver.CUdevice driverDevice
    cydriver.CUcontext primaryContext
    bool primaryContextRetained
    int deviceOrdinal
    cudaDeviceProp deviceProperties

cdef class cudaPythonGlobal:
    cdef bint _lazyInitDriver
    cdef int _numDevices
    cdef cudaPythonDevice* _deviceList
    cdef cudaError_t _lastError
    cdef int _CUDART_VERSION

    cdef cudaError_t lazyInitDriver(self) except ?cudaErrorCallRequiresNewerDriver nogil
    cdef cudaError_t lazyInitContextState(self) except ?cudaErrorCallRequiresNewerDriver nogil
    cdef cudaPythonDevice* getDevice(self, int deviceOrdinal) noexcept nogil
    cdef cudaPythonDevice* getDeviceFromDriver(self, cydriver.CUdevice driverDevice) noexcept nogil
    cdef cudaPythonDevice* getDeviceFromPrimaryCtx(self, cydriver.CUcontext context) noexcept nogil

cdef cudaError_t initPrimaryContext(cudaPythonDevice *device) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t resetPrimaryContext(cudaPythonDevice* device) except ?cudaErrorCallRequiresNewerDriver nogil

cdef cudaPythonGlobal globalGetInstance()
cdef cudaError_t _setLastError(cudaError_t err) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t getDescInfo(const cudaChannelFormatDesc* d, int *numberOfChannels, cydriver.CUarray_format *format) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t streamAddCallbackCommon(cudaStream_t stream, cudaStreamCallback_t callback, void *userData, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t streamGetCaptureInfoCommon(cudaStream_t stream, cudaStreamCaptureStatus* captureStatus_out, unsigned long long *id_out, cudaGraph_t *graph_out, const cudaGraphNode_t **dependencies_out, size_t *numDependencies_out) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t getChannelFormatDescFromDriverDesc(cudaChannelFormatDesc* pRuntimeDesc, size_t* pDepth, size_t* pHeight, size_t* pWidth, const cydriver.CUDA_ARRAY3D_DESCRIPTOR_v2* pDriverDesc) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t copyFromHost2D(cudaArray_const_t thisArray, size_t hOffset, size_t wOffset, const char *src, size_t spitch, size_t width, size_t height, cydriver.CUstream stream, bool async) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t copyFromDevice2D(cydriver.CUmemorytype type, cudaArray_const_t thisArray, size_t hOffset, size_t wOffset, const char *src, size_t srcOffset,
                                  size_t spitch, size_t width, size_t height, cydriver.CUstream stream, bool async) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t copyToHost2D(cudaArray_const_t thisArray, size_t hOffset, size_t wOffset, char *dst, size_t dpitch, size_t width,
                              size_t height, cydriver.CUstream stream, bool async) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t copyToDevice2D(cydriver.CUmemorytype type, cudaArray_const_t thisArray, size_t hOffset, size_t wOffset, const char *dst, size_t dstOffset, size_t dpitch,
                                size_t width, size_t height, cydriver.CUstream stream, bool async) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t copyToArray2D(cudaArray_const_t thisArray, size_t hOffsetSrc, size_t wOffsetSrc, cudaArray_t dst,
                               size_t hOffsetDst, size_t wOffsetDst, size_t width, size_t height) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t getChannelDesc(cudaArray_const_t thisArray, cudaChannelFormatDesc *outDesc) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t getDriverResDescFromResDesc(cydriver.CUDA_RESOURCE_DESC *rdDst, const cudaResourceDesc *rdSrc,
                                             cydriver.CUDA_TEXTURE_DESC *tdDst, const cudaTextureDesc_v2 *tdSrc,
                                             cydriver.CUDA_RESOURCE_VIEW_DESC *rvdDst, const cudaResourceViewDesc *rvdSrc) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t getResDescFromDriverResDesc(cudaResourceDesc *rdDst, const cydriver.CUDA_RESOURCE_DESC *rdSrc,
                                             cudaTextureDesc_v2 *tdDst, const cydriver.CUDA_TEXTURE_DESC *tdSrc,
                                             cudaResourceViewDesc *rvdDst, const cydriver.CUDA_RESOURCE_VIEW_DESC *rvdSrc) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t memsetPtr(char *mem, int c, size_t count, cudaStream_t sid, bool async) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t memset2DPtr(char *mem, size_t pitch, int c, size_t width, size_t height, cudaStream_t sid, bool async) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t copyFromHost(cudaArray_const_t thisArray, size_t hOffset, size_t wOffset, const char *src, size_t count,
                              cydriver.CUstream stream, bool async) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t copyFromDevice(cydriver.CUmemorytype type, cudaArray_const_t thisArray, size_t hOffset, size_t wOffset,
                                const char *src, size_t srcOffset, size_t count, cydriver.CUstream stream, bool async) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t copyToHost(cudaArray_const_t thisArray, size_t hOffset, size_t wOffset, char *dst, size_t count, cydriver.CUstream stream, bool async) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t copyToDevice(cydriver.CUmemorytype type, cudaArray_const_t thisArray, size_t hOffset, size_t wOffset,
                              const char *dst, size_t dstOffset, size_t count, cydriver.CUstream stream, bool async) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t copy1DConvertTo3DParams(void* dst, const void* src, size_t count, cudaMemcpyKind kind, cudaMemcpy3DParms *p) except ?cudaErrorCallRequiresNewerDriver nogil
cdef void toDriverMemsetNodeParams(const cudaMemsetParams *pRuntimeParams, cydriver.CUDA_MEMSET_NODE_PARAMS *pDriverParams) noexcept nogil
cdef cudaError_t toDriverMemCopy3DParams(const cudaMemcpy3DParms *p, cydriver.CUDA_MEMCPY3D *cd) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t mallocArray(cudaArray_t *arrayPtr, const cudaChannelFormatDesc *desc, size_t depth, size_t height,
                             size_t width, int corr2D, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t memcpy2DToArray(cudaArray_t dst, size_t hOffset, size_t wOffset, const char *src,
                                 size_t spitch, size_t width, size_t height, cudaMemcpyKind kind,
                                 cudaStream_t sid, bool async) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t memcpyDispatch(void *dst, const void *src, size_t size, cudaMemcpyKind kind) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t mallocHost(size_t size, void **mem, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t mallocPitch(size_t width, size_t height, size_t depth, void **mem, size_t *pitch) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t mallocMipmappedArray(cudaMipmappedArray_t *mipmappedArrayPtr, const cudaChannelFormatDesc *desc,
                                      size_t depth, size_t height, size_t width, unsigned int numLevels, unsigned int flags) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t memcpy2DPtr(char *dst, size_t dpitch, const char *src, size_t spitch, size_t width,
                             size_t height, cudaMemcpyKind kind,
                             cudaStream_t sid, bool async) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t memcpy3D(const cudaMemcpy3DParms *p, bool peer, int srcDevice, int dstDevice, cudaStream_t sid, bool async) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t memcpyAsyncDispatch(void *dst, const void *src, size_t size, cudaMemcpyKind kind, cudaStream_t stream) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t toCudartMemCopy3DParams(const cydriver.CUDA_MEMCPY3D_v2 *cd, cudaMemcpy3DParms *p) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t memcpy2DFromArray(char *dst, size_t dpitch, cudaArray_const_t src, size_t hOffset,
                                   size_t wOffset, size_t width, size_t height, cudaMemcpyKind kind,
                                   cudaStream_t sid, bool async) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t memcpy2DArrayToArray(cudaArray_t dst, size_t hOffsetDst, size_t wOffsetDst,
                                      cudaArray_const_t src, size_t hOffsetSrc, size_t wOffsetSrc,
                                      size_t width, size_t height, cudaMemcpyKind kind) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t memset3DPtr(cudaPitchedPtr p, int val, cudaExtent e, cudaStream_t sid, bool async) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t memcpyToArray(cudaArray_t dst, size_t hOffset, size_t wOffset, const char *src,
                               size_t count, cudaMemcpyKind kind,
                               cudaStream_t sid, bool async) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t memcpyFromArray(char *dst, cudaArray_const_t src, size_t hOffset, size_t wOffset,
                                 size_t count, cudaMemcpyKind kind,
                                 cudaStream_t sid, bool async) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t memcpyArrayToArray(cudaArray_t dst, size_t hOffsetDst, size_t wOffsetDst,
                                    cudaArray_const_t src, size_t hOffsetSrc, size_t wOffsetSrc,
                                    size_t count, cudaMemcpyKind kind) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t toDriverCudaResourceDesc(cydriver.CUDA_RESOURCE_DESC *_driver_pResDesc, const cudaResourceDesc *pResDesc) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t getDriverEglFrame(cydriver.CUeglFrame *cuEglFrame, cudaEglFrame eglFrame) except ?cudaErrorCallRequiresNewerDriver nogil
cdef cudaError_t getRuntimeEglFrame(cudaEglFrame *eglFrame, cydriver.CUeglFrame cueglFrame) except ?cudaErrorCallRequiresNewerDriver nogil
