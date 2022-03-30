# Copyright 2021-2022 NVIDIA Corporation.  All rights reserved.
#
# Please refer to the NVIDIA end user license agreement (EULA) associated
# with this source code for terms and conditions that govern your use of
# this software. Any use, reproduction, disclosure, or distribution of
# this software and related documentation outside the terms of the EULA
# is strictly prohibited.
from cuda.ccudart cimport *
from cuda._lib.ccudart.utils cimport *
from libc.stdlib cimport malloc, free, calloc
from libc.string cimport memset, memcpy, strncmp
from libcpp cimport bool
cimport cuda._cuda.ccuda as ccuda

cdef cudaPythonGlobal m_global = globalGetInstance()

cdef cudaError_t _cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = memcpyDispatch(dst, src, count, kind)
    if err != cudaSuccess:
        _setLastError(err)
    return err


cdef cudaError_t _cudaStreamCreate(cudaStream_t* pStream) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err

    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuStreamCreate(pStream, 0)
    if err != cudaSuccess:
        _setLastError(err)
    return err


cdef cudaError_t _cudaEventCreate(cudaEvent_t* event) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err

    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuEventCreate(event, ccuda.CUevent_flags_enum.CU_EVENT_DEFAULT)
    if err != cudaSuccess:
        _setLastError(err)
    return err


cdef cudaError_t _cudaEventQuery(cudaEvent_t event) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = <cudaError_t>ccuda._cuEventQuery(<ccuda.CUevent>event)
    if err != cudaSuccess:
        _setLastError(err)
    return err


cdef cudaChannelFormatDesc _cudaCreateChannelDesc(int x, int y, int z, int w, cudaChannelFormatKind f) nogil:
    cdef cudaChannelFormatDesc desc
    desc.x = x
    desc.y = y
    desc.z = z
    desc.w = w
    desc.f = f
    return desc


cdef cudaError_t _cudaDriverGetVersion(int* driverVersion) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = <cudaError_t>ccuda._cuDriverGetVersion(driverVersion)
    if err != cudaSuccess:
        _setLastError(err)
    return err


cdef cudaError_t _cudaRuntimeGetVersion(int* runtimeVersion) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    runtimeVersion[0] = m_global.CUDART_VERSION
    return cudaSuccess


cdef cudaError_t _cudaDeviceGetTexture1DLinearMaxWidth(size_t* maxWidthInElements, const cudaChannelFormatDesc* fmtDesc, int device) nogil except ?cudaErrorCallRequiresNewerDriver:
    if fmtDesc == NULL:
        _setLastError(cudaErrorInvalidValue)
        return cudaErrorInvalidValue

    cdef cudaError_t err
    cdef ccuda.CUarray_format fmt
    cdef int numChannels = 0

    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = getDescInfo(fmtDesc, &numChannels, &fmt)
    if err == cudaSuccess:
        _setLastError(err)
        return err
    err = <cudaError_t>ccuda._cuDeviceGetTexture1DLinearMaxWidth(maxWidthInElements, fmt, <unsigned>numChannels, device)
    if err != cudaSuccess:
        _setLastError(err)
    return err


cdef cudaError_t _cudaMallocHost(void** ptr, size_t size) nogil except ?cudaErrorCallRequiresNewerDriver:
    if ptr == NULL:
        _setLastError(cudaErrorInvalidValue)
        return cudaErrorInvalidValue

    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = mallocHost(size, ptr, 0)
    if err != cudaSuccess:
        _setLastError(err)
    return err


cdef cudaError_t _cudaMallocPitch(void** devPtr, size_t* pitch, size_t width, size_t height) nogil except ?cudaErrorCallRequiresNewerDriver:
    if devPtr == NULL or pitch == NULL:
        return cudaErrorInvalidValue

    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = mallocPitch(width, height, 1, devPtr, pitch)
    if err != cudaSuccess:
        _setLastError(err)
    return err


cdef cudaError_t _cudaMallocMipmappedArray(cudaMipmappedArray_t* mipmappedArray, const cudaChannelFormatDesc* desc, cudaExtent extent, unsigned int numLevels, unsigned int flags) nogil except ?cudaErrorCallRequiresNewerDriver:
    if mipmappedArray == NULL or desc == NULL:
        return cudaErrorInvalidValue

    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = mallocMipmappedArray(mipmappedArray, desc, extent.depth, extent.height, extent.width, numLevels, flags)
    if err != cudaSuccess:
        _setLastError(err)
    return err


cdef cudaError_t _cudaMemcpy2D(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err

    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = memcpy2DPtr(<char*>dst, dpitch, <const char*>src, spitch, width, height, kind, NULL, False)
    if err != cudaSuccess:
        _setLastError(err)
    return err


cdef cudaError_t _cudaMemcpy2DAsync(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err

    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = memcpy2DPtr(<char*>dst, dpitch, <const char*>src, spitch, width, height, kind, stream, True)
    if err != cudaSuccess:
        _setLastError(err)
    return err


cdef cudaError_t _cudaMemcpyAsync(void* dst, const void* src, size_t count, cudaMemcpyKind kind, cudaStream_t stream) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err

    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = memcpyAsyncDispatch(dst, src, count, kind, stream)
    if err != cudaSuccess:
        _setLastError(err)
    return err


cdef cudaError_t _cudaGraphAddMemcpyNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, const cudaMemcpy3DParms* pCopyParams) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef ccuda.CUcontext context
    cdef ccuda.CUDA_MEMCPY3D_v2 driverNodeParams
    cdef cudaError_t err

    if pCopyParams == NULL:
        _setLastError(cudaErrorInvalidValue)
        return cudaErrorInvalidValue

    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuCtxGetCurrent(&context)
    if err != cudaSuccess:
        _setLastError(err)
        return err

    err = toDriverMemCopy3DParams(pCopyParams, &driverNodeParams)
    if err != cudaSuccess:
        _setLastError(err)
        return err

    err = <cudaError_t>ccuda._cuGraphAddMemcpyNode(pGraphNode, graph, pDependencies, numDependencies, &driverNodeParams, context)
    if err != cudaSuccess:
        _setLastError(err)
    return err


cdef cudaError_t _cudaGraphAddMemcpyNode1D(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, void* dst, const void* src, size_t count, cudaMemcpyKind kind) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef ccuda.CUcontext context
    cdef ccuda.CUDA_MEMCPY3D_v2 driverNodeParams
    cdef cudaMemcpy3DParms copyParams
    cdef cudaError_t err

    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuCtxGetCurrent(&context)
    if err != cudaSuccess:
        _setLastError(err)
        return err

    copy1DConvertTo3DParams(dst, src, count, kind, &copyParams)

    err = toDriverMemCopy3DParams(&copyParams, &driverNodeParams)
    if err != cudaSuccess:
        _setLastError(err)
        return err

    err = <cudaError_t>ccuda._cuGraphAddMemcpyNode(pGraphNode, graph, pDependencies, numDependencies, &driverNodeParams, context)
    if err != cudaSuccess:
        _setLastError(err)
    return err


cdef cudaError_t _cudaGraphMemcpyNodeSetParams1D(cudaGraphNode_t node, void* dst, const void* src, size_t count, cudaMemcpyKind kind) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef ccuda.CUDA_MEMCPY3D_v2 driverNodeParams
    cdef cudaMemcpy3DParms copyParams
    cdef cudaError_t err

    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    copy1DConvertTo3DParams(dst, src, count, kind, &copyParams)

    err = toDriverMemCopy3DParams(&copyParams, &driverNodeParams)
    if err != cudaSuccess:
        _setLastError(err)
        return err

    err = <cudaError_t>ccuda._cuGraphMemcpyNodeSetParams(node, &driverNodeParams)
    if err != cudaSuccess:
        _setLastError(err)
    return err


cdef cudaError_t _cudaGraphExecMemcpyNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, const cudaMemcpy3DParms* pNodeParams) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef ccuda.CUcontext context
    cdef ccuda.CUDA_MEMCPY3D_v2 driverNodeParams
    cdef cudaError_t err

    if pNodeParams == NULL:
        _setLastError(cudaErrorInvalidValue)
        return cudaErrorInvalidValue

    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuCtxGetCurrent(&context)
    if err != cudaSuccess:
        _setLastError(err)
        return err

    err = toDriverMemCopy3DParams(pNodeParams, &driverNodeParams)
    if err != cudaSuccess:
        _setLastError(err)
        return err

    err = <cudaError_t>ccuda._cuGraphExecMemcpyNodeSetParams(hGraphExec, node, &driverNodeParams, context)
    if err != cudaSuccess:
        _setLastError(err)
    return err


cdef cudaError_t _cudaGraphExecMemcpyNodeSetParams1D(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, void* dst, const void* src, size_t count, cudaMemcpyKind kind) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef ccuda.CUcontext context
    cdef ccuda.CUDA_MEMCPY3D_v2 driverNodeParams
    cdef cudaMemcpy3DParms copyParams
    cdef cudaError_t err

    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuCtxGetCurrent(&context)
    if err != cudaSuccess:
        _setLastError(err)
        return err

    copy1DConvertTo3DParams(dst, src, count, kind, &copyParams)

    err = toDriverMemCopy3DParams(&copyParams, &driverNodeParams)
    if err != cudaSuccess:
        _setLastError(err)
        return err

    err = <cudaError_t>ccuda._cuGraphExecMemcpyNodeSetParams(hGraphExec, node, &driverNodeParams, context)
    if err != cudaSuccess:
        _setLastError(err)
    return err


cdef cudaError_t _cudaGetDriverEntryPoint(const char* symbol, void** funcPtr, unsigned long long flags) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = <cudaError_t>ccuda._cuGetProcAddress(symbol, funcPtr, m_global.CUDART_VERSION, flags)
    if err != cudaSuccess:
        _setLastError(err)
    return err


cdef cudaError_t _cudaGraphAddMemsetNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, const cudaMemsetParams* pMemsetParams) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef ccuda.CUcontext context
    cdef ccuda.CUDA_MEMSET_NODE_PARAMS driverParams
    cdef cudaError_t err

    if pMemsetParams == NULL:
        _setLastError(cudaErrorInvalidValue)
        return cudaErrorInvalidValue

    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuCtxGetCurrent(&context)
    if err != cudaSuccess:
        _setLastError(err)
        return err

    toDriverMemsetNodeParams(pMemsetParams, &driverParams)

    err = <cudaError_t>ccuda._cuGraphAddMemsetNode(pGraphNode, graph, pDependencies, numDependencies, &driverParams, context)
    if err != cudaSuccess:
        _setLastError(err)
    return err


cdef cudaError_t _cudaGraphExecMemsetNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, const cudaMemsetParams* pNodeParams) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef ccuda.CUcontext context
    cdef ccuda.CUDA_MEMSET_NODE_PARAMS driverParams
    cdef cudaError_t err

    if pNodeParams == NULL:
        _setLastError(cudaErrorInvalidValue)
        return cudaErrorInvalidValue

    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuCtxGetCurrent(&context)
    if err != cudaSuccess:
        _setLastError(err)
        return err

    toDriverMemsetNodeParams(pNodeParams, &driverParams)

    err = <cudaError_t>ccuda._cuGraphExecMemsetNodeSetParams(hGraphExec, node, &driverParams, context)
    if err != cudaSuccess:
        _setLastError(err)
    return err


cdef cudaError_t _cudaGraphMemcpyNodeSetParams(cudaGraphNode_t node, const cudaMemcpy3DParms* pNodeParams) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef ccuda.CUDA_MEMCPY3D_v2 driverNodeParams
    cdef cudaError_t err

    if pNodeParams == NULL:
        _setLastError(cudaErrorInvalidValue)
        return cudaErrorInvalidValue

    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = toDriverMemCopy3DParams(pNodeParams, &driverNodeParams)
    if err != cudaSuccess:
        _setLastError(err)
        return err
    err = <cudaError_t>ccuda._cuGraphMemcpyNodeSetParams(node, &driverNodeParams)
    if err != cudaSuccess:
        _setLastError(err)
    return err


cdef cudaError_t _cudaGraphMemcpyNodeGetParams(cudaGraphNode_t node, cudaMemcpy3DParms* p) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef ccuda.CUDA_MEMCPY3D_v2 driverNodeParams

    if p == NULL:
        _setLastError(cudaErrorInvalidValue)
        return cudaErrorInvalidValue
    
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuGraphMemcpyNodeGetParams(node, &driverNodeParams)
    if err != cudaSuccess:
        _setLastError(err)
        return err
    err = toCudartMemCopy3DParams(&driverNodeParams, p)
    if err != cudaSuccess:
        _setLastError(err)
    return err


cdef cudaError_t _cudaFuncGetAttributes(cudaFuncAttributes* attr, const void* func) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    if NULL == attr:
        _setLastError(cudaErrorInvalidValue)
        return cudaErrorInvalidValue
    cdef int bytes
    memset(attr, 0, sizeof(cudaFuncAttributes))
    err = <cudaError_t>ccuda._cuFuncGetAttribute(&attr[0].maxThreadsPerBlock,     ccuda.CUfunction_attribute_enum.CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, <ccuda.CUfunction>func)
    if err != cudaSuccess:
        _setLastError(err)
        return err
    err = <cudaError_t>ccuda._cuFuncGetAttribute(&attr[0].numRegs,                ccuda.CUfunction_attribute_enum.CU_FUNC_ATTRIBUTE_NUM_REGS, <ccuda.CUfunction>func)
    if err != cudaSuccess:
        _setLastError(err)
        return err
    err = <cudaError_t>ccuda._cuFuncGetAttribute(&attr[0].ptxVersion,             ccuda.CUfunction_attribute_enum.CU_FUNC_ATTRIBUTE_PTX_VERSION, <ccuda.CUfunction>func)
    if err != cudaSuccess:
        _setLastError(err)
        return err
    err = <cudaError_t>ccuda._cuFuncGetAttribute(&attr[0].binaryVersion,          ccuda.CUfunction_attribute_enum.CU_FUNC_ATTRIBUTE_BINARY_VERSION, <ccuda.CUfunction>func)
    if err != cudaSuccess:
        _setLastError(err)
        return err
    err = <cudaError_t>ccuda._cuFuncGetAttribute(&bytes,                          ccuda.CUfunction_attribute_enum.CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, <ccuda.CUfunction>func)
    if err != cudaSuccess:
        _setLastError(err)
        return err
    attr[0].sharedSizeBytes = <size_t>bytes
    err = <cudaError_t>ccuda._cuFuncGetAttribute(&bytes,                          ccuda.CUfunction_attribute_enum.CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES, <ccuda.CUfunction>func)
    if err != cudaSuccess:
        _setLastError(err)
        return err
    attr[0].constSizeBytes = <size_t>bytes
    err = <cudaError_t>ccuda._cuFuncGetAttribute(&bytes,                          ccuda.CUfunction_attribute_enum.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, <ccuda.CUfunction>func)
    if err != cudaSuccess:
        _setLastError(err)
        return err
    attr[0].localSizeBytes = <size_t>bytes
    err = <cudaError_t>ccuda._cuFuncGetAttribute(&attr[0].cacheModeCA,            ccuda.CUfunction_attribute_enum.CU_FUNC_ATTRIBUTE_CACHE_MODE_CA, <ccuda.CUfunction>func)
    if err != cudaSuccess:
        _setLastError(err)
        return err
    err = <cudaError_t>ccuda._cuFuncGetAttribute(&bytes,                          ccuda.CUfunction_attribute_enum.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, <ccuda.CUfunction>func)
    if err != cudaSuccess:
        _setLastError(err)
        return err
    err = <cudaError_t>ccuda._cuFuncGetAttribute(&attr[0].preferredShmemCarveout, ccuda.CUfunction_attribute_enum.CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT, <ccuda.CUfunction>func)
    if err != cudaSuccess:
        _setLastError(err)
        return err
    attr[0].maxDynamicSharedSizeBytes = <size_t>bytes
    if err != cudaSuccess:
        _setLastError(err)
    return err


cdef cudaError_t _cudaMallocArray(cudaArray_t* arrayPtr, const cudaChannelFormatDesc* desc, size_t width, size_t height, unsigned int flags) nogil except ?cudaErrorCallRequiresNewerDriver:
    if arrayPtr == NULL or desc == NULL:
        _setLastError(cudaErrorInvalidValue)
        return cudaErrorInvalidValue
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = mallocArray(arrayPtr, desc, 0, height, width, 0, flags)
    if err != cudaSuccess:
        _setLastError(err)
    return err


cdef cudaError_t _cudaMalloc3D(cudaPitchedPtr* pitchedDevPtr, cudaExtent extent) nogil except ?cudaErrorCallRequiresNewerDriver:
    if pitchedDevPtr == NULL:
        return cudaErrorInvalidValue

    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = mallocPitch(extent.width, extent.height, extent.depth, &pitchedDevPtr[0].ptr, &pitchedDevPtr[0].pitch)
    if err != cudaSuccess:
        _setLastError(err)
        return err
    pitchedDevPtr[0].xsize = extent.width
    pitchedDevPtr[0].ysize = extent.height
    return err


cdef cudaError_t _cudaMalloc3DArray(cudaArray_t* arrayPtr, const cudaChannelFormatDesc* desc, cudaExtent extent, unsigned int flags) nogil except ?cudaErrorCallRequiresNewerDriver:
    if arrayPtr == NULL or desc == NULL:
        _setLastError(cudaErrorInvalidValue)
        return cudaErrorInvalidValue

    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = mallocArray(arrayPtr, desc, extent.depth, extent.height, extent.width, 0, flags)
    if err != cudaSuccess:
        _setLastError(err)
    return err


cdef const char* _cudaGetErrorString(cudaError_t error) nogil except ?NULL:
    cdef const char* pStr = NULL
    cdef cudaError_t err

    err = <cudaError_t>ccuda._cuGetErrorString(<ccuda.CUresult>error, &pStr)
    if err != cudaSuccess:
        _setLastError(err)
    if err == <cudaError_t>cudaErrorInvalidValue:
        pStr = "unrecognized error code"
    return pStr


cdef cudaError_t _cudaStreamAddCallback(cudaStream_t stream, cudaStreamCallback_t callback, void* userData, unsigned int flags) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err

    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = streamAddCallbackCommon(stream, callback, userData, flags)
    if err != cudaSuccess:
        _setLastError(err)
    return err


cdef cudaError_t _cudaStreamGetCaptureInfo(cudaStream_t stream, cudaStreamCaptureStatus* captureStatus_out, unsigned long long* id_out) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err

    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = streamGetCaptureInfoCommon(stream, captureStatus_out, id_out, <cudaGraph_t *>0, <const cudaGraphNode_t **>0, <size_t *>0)
    if err != cudaSuccess:
        _setLastError(err)
    return err


cdef cudaError_t _cudaStreamGetCaptureInfo_v2(cudaStream_t stream, cudaStreamCaptureStatus* captureStatus_out, unsigned long long* id_out, cudaGraph_t* graph_out, const cudaGraphNode_t** dependencies_out, size_t* numDependencies_out) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err

    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = streamGetCaptureInfoCommon(stream, captureStatus_out, id_out, graph_out, dependencies_out, numDependencies_out)
    if err != cudaSuccess:
        _setLastError(err)
    return err


cdef cudaError_t _cudaImportExternalSemaphore(cudaExternalSemaphore_t* extSem_out, const cudaExternalSemaphoreHandleDesc* semHandleDesc) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    cdef ccuda.CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC driverSemHandleDesc

    if semHandleDesc == NULL:
        _setLastError(cudaErrorInvalidValue)
        return cudaErrorInvalidValue

    memset(&driverSemHandleDesc, 0, sizeof(driverSemHandleDesc))

    if semHandleDesc.type == cudaExternalSemaphoreHandleType.cudaExternalSemaphoreHandleTypeOpaqueFd:
        driverSemHandleDesc.type =  ccuda.CUexternalSemaphoreHandleType_enum.CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD
        driverSemHandleDesc.handle.fd = semHandleDesc.handle.fd
    elif semHandleDesc.type == cudaExternalSemaphoreHandleType.cudaExternalSemaphoreHandleTypeOpaqueWin32:
        driverSemHandleDesc.type =  ccuda.CUexternalSemaphoreHandleType_enum.CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32
        driverSemHandleDesc.handle.win32.handle = semHandleDesc.handle.win32.handle
        driverSemHandleDesc.handle.win32.name = semHandleDesc.handle.win32.name
    elif semHandleDesc.type == cudaExternalSemaphoreHandleType.cudaExternalSemaphoreHandleTypeOpaqueWin32Kmt:
        driverSemHandleDesc.type =  ccuda.CUexternalSemaphoreHandleType_enum.CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_KMT
        driverSemHandleDesc.handle.win32.handle = semHandleDesc.handle.win32.handle
        driverSemHandleDesc.handle.win32.name = semHandleDesc.handle.win32.name
    elif semHandleDesc.type == cudaExternalSemaphoreHandleType.cudaExternalSemaphoreHandleTypeD3D12Fence:
        driverSemHandleDesc.type =  ccuda.CUexternalSemaphoreHandleType_enum.CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D12_FENCE
        driverSemHandleDesc.handle.win32.handle = semHandleDesc.handle.win32.handle
        driverSemHandleDesc.handle.win32.name = semHandleDesc.handle.win32.name
    elif semHandleDesc.type == cudaExternalSemaphoreHandleType.cudaExternalSemaphoreHandleTypeD3D11Fence:
        driverSemHandleDesc.type =  ccuda.CUexternalSemaphoreHandleType_enum.CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D11_FENCE
        driverSemHandleDesc.handle.win32.handle = semHandleDesc.handle.win32.handle
        driverSemHandleDesc.handle.win32.name = semHandleDesc.handle.win32.name
    elif semHandleDesc.type == cudaExternalSemaphoreHandleType.cudaExternalSemaphoreHandleTypeNvSciSync:
        driverSemHandleDesc.type =  ccuda.CUexternalSemaphoreHandleType_enum.CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_NVSCISYNC
        driverSemHandleDesc.handle.nvSciSyncObj = semHandleDesc.handle.nvSciSyncObj
    elif semHandleDesc.type == cudaExternalSemaphoreHandleType.cudaExternalSemaphoreHandleTypeKeyedMutex:
        driverSemHandleDesc.type =  ccuda.CUexternalSemaphoreHandleType_enum.CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D11_KEYED_MUTEX
        driverSemHandleDesc.handle.win32.handle = semHandleDesc.handle.win32.handle
        driverSemHandleDesc.handle.win32.name = semHandleDesc.handle.win32.name
    elif semHandleDesc.type == cudaExternalSemaphoreHandleType.cudaExternalSemaphoreHandleTypeKeyedMutexKmt:
        driverSemHandleDesc.type =  ccuda.CUexternalSemaphoreHandleType_enum.CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D11_KEYED_MUTEX_KMT
        driverSemHandleDesc.handle.win32.handle = semHandleDesc.handle.win32.handle
        driverSemHandleDesc.handle.win32.name = semHandleDesc.handle.win32.name
    elif semHandleDesc.type == cudaExternalSemaphoreHandleType.cudaExternalSemaphoreHandleTypeTimelineSemaphoreFd:
        driverSemHandleDesc.type =  ccuda.CUexternalSemaphoreHandleType_enum.CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_TIMELINE_SEMAPHORE_FD
        driverSemHandleDesc.handle.fd = semHandleDesc.handle.fd
    elif semHandleDesc.type == cudaExternalSemaphoreHandleType.cudaExternalSemaphoreHandleTypeTimelineSemaphoreWin32:
        driverSemHandleDesc.type =  ccuda.CUexternalSemaphoreHandleType_enum.CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_TIMELINE_SEMAPHORE_WIN32
        driverSemHandleDesc.handle.win32.handle = semHandleDesc.handle.win32.handle
        driverSemHandleDesc.handle.win32.name = semHandleDesc.handle.win32.name
    driverSemHandleDesc.flags = semHandleDesc.flags

    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuImportExternalSemaphore(<ccuda.CUexternalSemaphore *>extSem_out, &driverSemHandleDesc)
    if err != <cudaError_t>cudaSuccess:
        _setLastError(err)
    return err


cdef cudaError_t _cudaSignalExternalSemaphoresAsync(const cudaExternalSemaphore_t* extSemArray, const cudaExternalSemaphoreSignalParams* paramsArray, unsigned int numExtSems, cudaStream_t stream) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuSignalExternalSemaphoresAsync(<const ccuda.CUexternalSemaphore *>extSemArray, <ccuda.CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS *>paramsArray, numExtSems, stream)
    if err != cudaSuccess:
        _setLastError(err)
        return err
    return cudaSuccess


cdef cudaError_t _cudaWaitExternalSemaphoresAsync(const cudaExternalSemaphore_t* extSemArray, const cudaExternalSemaphoreWaitParams* paramsArray, unsigned int numExtSems, cudaStream_t stream) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuWaitExternalSemaphoresAsync(<const ccuda.CUexternalSemaphore *>extSemArray, <ccuda.CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS *>paramsArray, numExtSems, stream)
    if err != cudaSuccess:
        _setLastError(err)
        return err
    return cudaSuccess


cdef cudaError_t _cudaArrayGetInfo(cudaChannelFormatDesc* desc, cudaExtent* extent, unsigned int* flags, cudaArray_t array) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    cdef ccuda.CUDA_ARRAY3D_DESCRIPTOR_v2 driverDesc
    cdef size_t width  = 0
    cdef size_t height = 0
    cdef size_t depth  = 0

    # Zero out parameters in case cuArray3DGetDescriptor fails
    if flags:
        flags[0] = 0

    if desc:
        memset(desc, 0, sizeof(desc[0]))


    if extent:
        memset(extent, 0, sizeof(extent[0]))

    err = <cudaError_t>ccuda._cuArray3DGetDescriptor_v2(&driverDesc, <ccuda.CUarray>array)
    if err != <cudaError_t>cudaSuccess:
        _setLastError(err)
        return err

    # Flags are copied directly from the driver API
    if flags:
        flags[0] = driverDesc.Flags

    # Convert from driver API types to runtime API types. extent.Depth = 0
    # indicates a 2D array.
    if desc:
        width  = 0
        height = 0
        depth  = 0

        err = getChannelFormatDescFromDriverDesc(desc, &depth, &height, &width, &driverDesc)
        if err != <cudaError_t>cudaSuccess:
            _setLastError(err)
            return err

    if extent:
        extent.width  = driverDesc.Width
        extent.height = driverDesc.Height
        extent.depth  = driverDesc.Depth

    return cudaSuccess


cdef cudaError_t _cudaMemcpy2DToArray(cudaArray_t dst, size_t wOffset, size_t hOffset, const void* src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err

    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = memcpy2DToArray(dst, hOffset, wOffset, <const char*>src, spitch, width, height, kind, NULL, False)
    if err != cudaSuccess:
        _setLastError(err)
    return err


cdef cudaError_t _cudaMemcpy2DFromArray(void* dst, size_t dpitch, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t width, size_t height, cudaMemcpyKind kind) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err

    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = memcpy2DFromArray(<char*>dst, dpitch, src, hOffset, wOffset, width, height, kind, NULL, False)
    if err != cudaSuccess:
        _setLastError(err)
    return err


cdef cudaError_t _cudaMemcpy2DArrayToArray(cudaArray_t dst, size_t wOffsetDst, size_t hOffsetDst, cudaArray_const_t src, size_t wOffsetSrc, size_t hOffsetSrc, size_t width, size_t height, cudaMemcpyKind kind) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err

    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = memcpy2DArrayToArray(dst, hOffsetDst, wOffsetDst, src, hOffsetSrc, wOffsetSrc, width, height, kind)
    if err != cudaSuccess:
        _setLastError(err)
    return err


cdef cudaError_t _cudaMemcpy2DToArrayAsync(cudaArray_t dst, size_t wOffset, size_t hOffset, const void* src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err

    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = memcpy2DToArray(dst, hOffset, wOffset, <const char*>src, spitch, width, height, kind, stream, True)
    if err != cudaSuccess:
        _setLastError(err)
    return err


cdef cudaError_t _cudaMemcpy2DFromArrayAsync(void* dst, size_t dpitch, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err

    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = memcpy2DFromArray(<char*>dst, dpitch, src, hOffset, wOffset, width, height, kind, stream, True)
    if err != cudaSuccess:
        _setLastError(err)
    return err


cdef cudaError_t _cudaMemset3D(cudaPitchedPtr pitchedDevPtr, int value, cudaExtent extent) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err

    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = memset3DPtr(pitchedDevPtr, value, extent, NULL, False)
    if err != cudaSuccess:
        _setLastError(err)
    return err


cdef cudaError_t _cudaMemset3DAsync(cudaPitchedPtr pitchedDevPtr, int value, cudaExtent extent, cudaStream_t stream) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err

    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = memset3DPtr(pitchedDevPtr, value, extent, stream, True)
    if err != cudaSuccess:
        _setLastError(err)
    return err


cdef cudaError_t _cudaMemcpyToArray(cudaArray_t dst, size_t wOffset, size_t hOffset, const void* src, size_t count, cudaMemcpyKind kind) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err

    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = memcpyToArray(dst, hOffset, wOffset, <const char*>src, count, kind, NULL, False)
    if err != cudaSuccess:
        _setLastError(err)
    return err


cdef cudaError_t _cudaMemcpyFromArray(void* dst, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t count, cudaMemcpyKind kind) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err

    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = memcpyFromArray(<char*>dst, src, hOffset, wOffset, count, kind, NULL, 0)
    if err != cudaSuccess:
        _setLastError(err)
    return err


cdef cudaError_t _cudaMemcpyToArrayAsync(cudaArray_t dst, size_t wOffset, size_t hOffset, const void* src, size_t count, cudaMemcpyKind kind, cudaStream_t stream) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err

    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = memcpyToArray(dst, hOffset, wOffset, <const char*>src, count, kind, stream, True)
    if err != cudaSuccess:
        _setLastError(err)
    return err


cdef cudaError_t _cudaMemcpyFromArrayAsync(void* dst, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t count, cudaMemcpyKind kind, cudaStream_t stream) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err

    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = memcpyFromArray(<char*>dst, src, hOffset, wOffset, count, kind, stream, True)
    if err != cudaSuccess:
        _setLastError(err)
    return err


cdef cudaError_t _cudaPointerGetAttributes(cudaPointerAttributes* attributes, const void* ptr) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    cdef cudaPointerAttributes attrib
    cdef ccuda.CUcontext driverContext = NULL
    cdef ccuda.CUmemorytype driverMemoryType
    cdef int isManaged
    cdef ccuda.CUpointer_attribute[6] query
    query[0] = ccuda.CUpointer_attribute_enum.CU_POINTER_ATTRIBUTE_CONTEXT
    query[1] = ccuda.CUpointer_attribute_enum.CU_POINTER_ATTRIBUTE_MEMORY_TYPE
    query[2] = ccuda.CUpointer_attribute_enum.CU_POINTER_ATTRIBUTE_DEVICE_POINTER
    query[3] = ccuda.CUpointer_attribute_enum.CU_POINTER_ATTRIBUTE_HOST_POINTER
    query[4] = ccuda.CUpointer_attribute_enum.CU_POINTER_ATTRIBUTE_IS_MANAGED
    query[5] = ccuda.CUpointer_attribute_enum.CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL

    cdef void** data = [
        &driverContext,
        &driverMemoryType,
        &attrib.devicePointer,
        &attrib.hostPointer,
        &isManaged,
        &attrib.device
    ]

    if attributes == NULL:
        _setLastError(cudaErrorInvalidValue)
        return cudaErrorInvalidValue

    # Get all the attributes we need
    err = <cudaError_t>ccuda._cuPointerGetAttributes(<unsigned int>(sizeof(query)/sizeof(query[0])), query, data, <ccuda.CUdeviceptr_v2>ptr)
    if err != cudaSuccess:
        if attributes != NULL:
            memset(attributes, 0, sizeof(attributes[0]))
            attributes[0].device = -1
        _setLastError(err)
        return err

    if driverMemoryType == ccuda.CUmemorytype_enum.CU_MEMORYTYPE_HOST:
        if isManaged:
            attrib.type = cudaMemoryTypeManaged
        else:
            attrib.type = cudaMemoryTypeHost
    elif driverMemoryType == ccuda.CUmemorytype_enum.CU_MEMORYTYPE_DEVICE:
        if isManaged:
            attrib.type = cudaMemoryTypeManaged
        else:
            attrib.type = cudaMemoryTypeDevice
    else:
         if driverMemoryType == 0:
            attrib.type = cudaMemoryTypeUnregistered
         else:
            if attributes != NULL:
                memset(attributes, 0, sizeof(attributes[0]))
                attributes[0].device = -1
            _setLastError(cudaErrorInvalidValue)
            return cudaErrorInvalidValue

    # copy to user structure
    attributes[0] = attrib

    return cudaSuccess


cdef cudaError_t _cudaGetDeviceFlags(unsigned int* flags) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err

    if flags == NULL:
        _setLastError(cudaErrorInvalidValue)
        return cudaErrorInvalidValue

    cdef ccuda.CUcontext driverContext
    err = <cudaError_t>ccuda._cuCtxGetCurrent(&driverContext)
    if err != cudaSuccess:
        _setLastError(err)
        return err

    # Get the flags from the current context
    if driverContext != NULL:
        err = <cudaError_t>ccuda._cuCtxGetFlags(flags)
        if err != cudaSuccess:
            _setLastError(err)
        return err

    # Assume first valid device and get its implicit flags
    cdef ccuda.CUdevice dev
    cdef unsigned int pcFlags
    cdef int pcActive
    err = <cudaError_t>ccuda._cuDeviceGet(&dev, 0)
    if err != cudaSuccess:
        _setLastError(err)
        return err
    err = <cudaError_t>ccuda._cuDevicePrimaryCtxGetState(dev, &pcFlags, &pcActive)
    if err != cudaSuccess:
        _setLastError(err)
        return err
    flags[0] = pcFlags | cudaDeviceMapHost
    return cudaSuccess


cdef cudaError_t _cudaMemcpy3D(const cudaMemcpy3DParms* p) nogil except ?cudaErrorCallRequiresNewerDriver:
    if p == NULL:
        return cudaErrorInvalidValue

    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = memcpy3D(p, False, 0, 0, NULL, False)
    if err != cudaSuccess:
        _setLastError(err)
    return err


cdef cudaError_t _cudaMemcpy3DAsync(const cudaMemcpy3DParms* p, cudaStream_t stream) nogil except ?cudaErrorCallRequiresNewerDriver:
    if p == NULL:
        return cudaErrorInvalidValue

    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = memcpy3D(p, False, 0, 0, stream, True)
    if err != cudaSuccess:
        _setLastError(err)
    return err


cdef cudaError_t _cudaMemPoolSetAccess(cudaMemPool_t memPool, const cudaMemAccessDesc* descList, size_t count) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    cdef size_t MAX_DEVICES = 32
    cdef ccuda.CUmemAccessDesc localList[32]
    cdef ccuda.CUmemAccessDesc *cuDescList
    cdef size_t i = 0

    if (count > MAX_DEVICES):
        cuDescList = <ccuda.CUmemAccessDesc*>calloc(sizeof(ccuda.CUmemAccessDesc), count)
    else:
        cuDescList = localList

    if cuDescList == NULL:
        _setLastError(cudaErrorMemoryAllocation)
        return cudaErrorMemoryAllocation

    while i < count:
        cuDescList[i].location.type = <ccuda.CUmemLocationType>descList[i].location.type
        cuDescList[i].location.id = descList[i].location.id
        cuDescList[i].flags = <ccuda.CUmemAccess_flags>descList[i].flags
        i += 1

    err = <cudaError_t>ccuda._cuMemPoolSetAccess(memPool, cuDescList, count)
    if err != cudaSuccess:
        _setLastError(err)
        return err

    if count > MAX_DEVICES:
        free(cuDescList)

    return cudaSuccess


cdef cudaError_t _cudaDeviceReset() nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef int deviceOrdinal = 0
    cdef cudaError_t err
    if not m_global._cudaPythonInit:
        return cudaSuccess
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    cdef ccuda.CUcontext context
    err = <cudaError_t>ccuda._cuCtxGetCurrent(&context)
    if err != cudaSuccess:
        _setLastError(err)
        return err
    for deviceOrdinal in range(m_global._numDevices):
        if m_global._driverContext[deviceOrdinal] == context:
            err = <cudaError_t>ccuda._cuDevicePrimaryCtxReset_v2(m_global._driverDevice[deviceOrdinal])
            break
    return err


cdef cudaError_t _cudaThreadExit() nogil except ?cudaErrorCallRequiresNewerDriver:
    return cudaDeviceReset()


cdef cudaError_t _cudaGetLastError() nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t last_err = m_global._lastError
    m_global._lastError = cudaSuccess
    return last_err


cdef cudaError_t _cudaPeekAtLastError() nogil except ?cudaErrorCallRequiresNewerDriver:
    return m_global._lastError


cdef cudaError_t _cudaGetDevice(int* device) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef int deviceOrdinal = 0
    cdef cudaError_t err
    cdef ccuda.CUresult err_driver
    cdef ccuda.CUcontext context
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err

    err_driver = ccuda._cuCtxGetCurrent(&context)
    if err_driver == ccuda.cudaError_enum.CUDA_ERROR_INVALID_CONTEXT or (err_driver == ccuda.cudaError_enum.CUDA_SUCCESS and context == NULL):
        err = cudaSetDevice(0)
        if err != cudaSuccess:
            _setLastError(err)
            return err
        err_driver = ccuda._cuCtxGetCurrent(&context)

    if err_driver != ccuda.cudaError_enum.CUDA_SUCCESS:
        _setLastError(err)
        return err

    found = False
    for deviceOrdinal in range(m_global._numDevices):
        if m_global._driverContext[deviceOrdinal] == context:
            found = True
            break
    device[0] = deviceOrdinal if found else 0
    return cudaSuccess


cdef cudaError_t _cudaSetDevice(int device) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInitGlobal()
    if err != cudaSuccess:
        return err
    if device < 0 or device >= m_global._numDevices:
        _setLastError(err)
        return cudaErrorInvalidDevice

    err = m_global.lazyInitDevice(device)
    if err != cudaSuccess:
        _setLastError(err)
        return err

    err = <cudaError_t>ccuda._cuCtxSetCurrent(m_global._driverContext[device])
    if err != cudaSuccess:
        _setLastError(err)
    return err


cdef cudaError_t _cudaGetDeviceProperties(cudaDeviceProp* prop, int device) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err

    err = <cudaError_t>ccuda._cuDeviceGetAttribute(&(prop[0].kernelExecTimeoutEnabled),  ccuda.CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT, <ccuda.CUdevice>device)
    if err != cudaSuccess:
        _setLastError(err)
        return err

    err = <cudaError_t>ccuda._cuDeviceGetAttribute(&(prop[0].computeMode),  ccuda.CU_DEVICE_ATTRIBUTE_COMPUTE_MODE, <ccuda.CUdevice>device)
    if err != cudaSuccess:
        _setLastError(err)
        return err

    err = <cudaError_t>ccuda._cuDeviceGetAttribute(&(prop[0].clockRate), ccuda.CU_DEVICE_ATTRIBUTE_CLOCK_RATE, <ccuda.CUdevice>device)
    if err != cudaSuccess:
        _setLastError(err)
        return err

    err = <cudaError_t>ccuda._cuDeviceGetAttribute(&(prop[0].memoryClockRate), ccuda.CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE, <ccuda.CUdevice>device)
    if err != cudaSuccess:
        _setLastError(err)
        return err

    err = <cudaError_t>ccuda._cuDeviceGetAttribute(&(prop[0].singleToDoublePrecisionPerfRatio), ccuda.CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO, <ccuda.CUdevice>device)
    if err != cudaSuccess:
        _setLastError(err)
        return err

    prop[0] = m_global._deviceProperties[device]

    return cudaSuccess


cdef cudaError_t _cudaChooseDevice(int* device, const cudaDeviceProp* prop) nogil except ?cudaErrorCallRequiresNewerDriver:
    if device == NULL or prop == NULL:
        _setLastError(cudaErrorInvalidValue)
        return cudaErrorInvalidValue

    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    cdef int best = -1
    cdef int maxrank = -1
    cdef int rank
    cdef char* dontCare_name = [b'\0']
    cdef int dontCare_major = -1
    cdef int dontCare_minor = -1
    cdef size_t dontCare_totalGlobalMem = 0
    cdef int deviceOrdinal = 0

    for deviceOrdinal in range(m_global._numDevices):
        rank = 0
        if (strncmp(prop[0].name, dontCare_name, sizeof(prop[0].name)) != 0):
            rank += strncmp(prop[0].name, m_global._deviceProperties[deviceOrdinal].name, sizeof(prop[0].name)) == 0
        if (prop[0].major != dontCare_major):
            rank += prop[0].major <= m_global._deviceProperties[deviceOrdinal].major
        if (prop[0].major == m_global._deviceProperties[deviceOrdinal].major and prop[0].minor != dontCare_minor):
            rank += prop[0].minor <= m_global._deviceProperties[deviceOrdinal].minor
        if (prop[0].totalGlobalMem != dontCare_totalGlobalMem):
            rank += prop[0].totalGlobalMem <= m_global._deviceProperties[deviceOrdinal].totalGlobalMem
        if (rank > maxrank):
            maxrank = rank
            best = deviceOrdinal

    device[0] = best
    return cudaSuccess


cdef cudaError_t _cudaMemcpyArrayToArray(cudaArray_t dst, size_t wOffsetDst, size_t hOffsetDst, cudaArray_const_t src, size_t wOffsetSrc, size_t hOffsetSrc, size_t count, cudaMemcpyKind kind) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err

    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = memcpyArrayToArray(dst, hOffsetDst, wOffsetDst, src, hOffsetSrc, wOffsetSrc, count, kind)
    if err != cudaSuccess:
        _setLastError(err)
    return err


cdef cudaError_t _cudaGetChannelDesc(cudaChannelFormatDesc* desc, cudaArray_const_t array) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err

    if desc == NULL:
        _setLastError(cudaErrorInvalidValue)
        return cudaErrorInvalidValue

    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = getChannelDesc(array, desc)
    if err != cudaSuccess:
        _setLastError(err)
    return err


cdef cudaError_t _cudaCreateTextureObject(cudaTextureObject_t* pTexObject, const cudaResourceDesc* pResDesc, const cudaTextureDesc* pTexDesc, const cudaResourceViewDesc* pResViewDesc) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err

    if pResDesc == NULL:
        _setLastError(cudaErrorInvalidValue)
        return cudaErrorInvalidValue

    cdef ccuda.CUDA_RESOURCE_DESC rd
    cdef ccuda.CUDA_TEXTURE_DESC td
    cdef ccuda.CUDA_RESOURCE_VIEW_DESC rvd

    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    if pResViewDesc:
        err = getDriverResDescFromResDesc(&rd, pResDesc, &td, pTexDesc, &rvd, pResViewDesc)
    else:
        err = getDriverResDescFromResDesc(&rd, pResDesc, &td, pTexDesc, NULL, pResViewDesc)
    if err != cudaSuccess:
        _setLastError(err)
        return err
    if pResViewDesc:
        err = <cudaError_t>ccuda._cuTexObjectCreate(pTexObject, &rd, &td, &rvd)
    else:
        err = <cudaError_t>ccuda._cuTexObjectCreate(pTexObject, &rd, &td, NULL)
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t _cudaGetTextureObjectTextureDesc(cudaTextureDesc* pTexDesc, cudaTextureObject_t texObject) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err

    cdef cudaResourceDesc resDesc
    cdef ccuda.CUDA_RESOURCE_DESC rd
    cdef ccuda.CUDA_TEXTURE_DESC td

    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuTexObjectGetResourceDesc(&rd, texObject)
    if err != cudaSuccess:
        _setLastError(err)
        return err
    err = <cudaError_t>ccuda._cuTexObjectGetTextureDesc(&td, texObject)
    if err != cudaSuccess:
        _setLastError(err)
        return err
    err = getResDescFromDriverResDesc(&resDesc, &rd, pTexDesc, &td, NULL, NULL)
    if err != cudaSuccess:
        _setLastError(err)
        return err

    return cudaSuccess


cdef cudaError_t _cudaGetTextureObjectResourceViewDesc(cudaResourceViewDesc* pResViewDesc, cudaTextureObject_t texObject) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    cdef cudaResourceDesc resDesc
    cdef ccuda.CUDA_RESOURCE_DESC rd
    cdef ccuda.CUDA_RESOURCE_VIEW_DESC rvd

    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err =  <cudaError_t>ccuda.cuTexObjectGetResourceDesc(&rd, texObject)
    if err != cudaSuccess:
        _setLastError(err)
        return err
    err =  <cudaError_t>ccuda.cuTexObjectGetResourceViewDesc(&rvd, texObject)
    if err != cudaSuccess:
        _setLastError(err)
        return err
    err = getResDescFromDriverResDesc(&resDesc, &rd, NULL, NULL, pResViewDesc, &rvd)
    if err != cudaSuccess:
        _setLastError(err)
        return err

    return cudaSuccess


cdef cudaError_t _cudaGetExportTable(const void** ppExportTable, cudaUUID_t* pExportTableId) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err

    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuGetExportTable(ppExportTable, <ccuda.CUuuid*>pExportTableId)
    if err != cudaSuccess:
        _setLastError(err)
    return err


cdef cudaError_t _cudaMemcpy3DPeer(const cudaMemcpy3DPeerParms* p) nogil except ?cudaErrorCallRequiresNewerDriver:
    if p == NULL:
        _setLastError(cudaErrorInvalidValue)
        return cudaErrorInvalidValue

    cdef cudaError_t err
    cdef cudaMemcpy3DParms cp
    memset(&cp, 0, sizeof(cp))

    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    cp.srcArray = p[0].srcArray
    cp.srcPos = p[0].srcPos
    cp.srcPtr = p[0].srcPtr
    cp.dstArray = p[0].dstArray
    cp.dstPos = p[0].dstPos
    cp.dstPtr = p[0].dstPtr
    cp.extent = p[0].extent
    cp.kind = cudaMemcpyKind.cudaMemcpyDeviceToDevice

    err = memcpy3D(&cp, True, p[0].srcDevice, p[0].dstDevice, NULL, False)
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t _cudaMemcpy3DPeerAsync(const cudaMemcpy3DPeerParms* p, cudaStream_t stream) nogil except ?cudaErrorCallRequiresNewerDriver:
    if p == NULL:
        _setLastError(cudaErrorInvalidValue)
        return cudaErrorInvalidValue

    cdef cudaError_t err
    cdef cudaMemcpy3DParms cp
    memset(&cp, 0, sizeof(cp))

    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    cp.srcArray = p[0].srcArray
    cp.srcPos = p[0].srcPos
    cp.srcPtr = p[0].srcPtr
    cp.dstArray = p[0].dstArray
    cp.dstPos = p[0].dstPos
    cp.dstPtr = p[0].dstPtr
    cp.extent = p[0].extent
    cp.kind = cudaMemcpyKind.cudaMemcpyDeviceToDevice

    err = memcpy3D(&cp, True, p[0].srcDevice, p[0].dstDevice, stream, True)
    if err != cudaSuccess:
        _setLastError(err)
    return err


cdef cudaPitchedPtr _make_cudaPitchedPtr(void* d, size_t p, size_t xsz, size_t ysz) nogil:
    cdef cudaPitchedPtr s
    s.ptr   = d
    s.pitch = p
    s.xsize = xsz
    s.ysize = ysz
    return s


cdef cudaPos _make_cudaPos(size_t x, size_t y, size_t z) nogil:
    cdef cudaPos p
    p.x = x
    p.y = y
    p.z = z
    return p


cdef cudaExtent _make_cudaExtent(size_t w, size_t h, size_t d) nogil:
    cdef cudaExtent e
    e.width  = w
    e.height = h
    e.depth  = d
    return e


cdef cudaError_t _cudaSetDoubleForDevice(double* d) nogil except ?cudaErrorCallRequiresNewerDriver:
    return cudaSuccess


cdef cudaError_t _cudaSetDoubleForHost(double* d) nogil except ?cudaErrorCallRequiresNewerDriver:
    return cudaSuccess


cdef cudaError_t _cudaSetDeviceFlags(unsigned int flags) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    cdef int deviceOrdinal = 0
    flags &= ~cudaDeviceMapHost
    if flags & ~cudaDeviceMask:
        _setLastError(cudaErrorInvalidValue)
        return cudaErrorInvalidValue
    cdef unsigned int scheduleFlags = flags & cudaDeviceScheduleMask
    if scheduleFlags and (scheduleFlags != cudaDeviceScheduleSpin and
                          scheduleFlags != cudaDeviceScheduleYield and
                          scheduleFlags != cudaDeviceScheduleBlockingSync):
        _setLastError(cudaErrorInvalidValue)
        return cudaErrorInvalidValue
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err

    cdef ccuda.CUcontext context
    err = <cudaError_t>ccuda._cuCtxGetCurrent(&context)
    if err != cudaSuccess:
        _setLastError(err)
        return err
    for deviceOrdinal in range(m_global._numDevices):
        if m_global._driverContext[deviceOrdinal] == context:
            break
    err = <cudaError_t>ccuda._cuDevicePrimaryCtxSetFlags_v2(m_global._driverDevice[deviceOrdinal], flags)
    if err != cudaSuccess:
        _setLastError(err)
    return err


cdef cudaError_t _cudaGraphAddMemAllocNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, cudaMemAllocNodeParams* nodeParams) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    if nodeParams == NULL:
        _setLastError(cudaErrorInvalidValue)
        return cudaErrorInvalidValue
    
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuGraphAddMemAllocNode(pGraphNode, graph, pDependencies, numDependencies, <ccuda.CUDA_MEM_ALLOC_NODE_PARAMS *>nodeParams)
    if err != cudaSuccess:
        _setLastError(err)
    return err


cdef cudaError_t _cudaGraphMemAllocNodeGetParams(cudaGraphNode_t node, cudaMemAllocNodeParams* params_out) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    if params_out == NULL:
        _setLastError(cudaErrorInvalidValue)
        return cudaErrorInvalidValue

    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuGraphMemAllocNodeGetParams(node, <ccuda.CUDA_MEM_ALLOC_NODE_PARAMS *>params_out)
    if err != cudaSuccess:
        _setLastError(err)
    return err


cdef cudaError_t _cudaGraphMemFreeNodeGetParams(cudaGraphNode_t node, void* dptr_out) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    if dptr_out == NULL:
        _setLastError(cudaErrorInvalidValue)
        return cudaErrorInvalidValue

    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuGraphMemFreeNodeGetParams(node, <ccuda.CUdeviceptr *>dptr_out)
    if err != cudaSuccess:
        _setLastError(err)
    return err


cdef cudaError_t _cudaMemAdvise(const void* devPtr, size_t count, cudaMemoryAdvise advice, int device) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = <cudaError_t>ccuda._cuMemAdvise(<ccuda.CUdeviceptr>devPtr, count, <ccuda.CUmem_advise>advice, <ccuda.CUdevice>device)
    if err != cudaSuccess:
        _setLastError(err)
    return err


cdef cudaError_t _cudaMemRangeGetAttribute(void* data, size_t dataSize, cudaMemRangeAttribute attribute, const void* devPtr, size_t count) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = <cudaError_t>ccuda._cuMemRangeGetAttribute(data, dataSize, <ccuda.CUmem_range_attribute>attribute, <ccuda.CUdeviceptr>devPtr, count)
    if err != cudaSuccess:
        _setLastError(err)
    return err


cdef cudaError_t _cudaMemRangeGetAttributes(void** data, size_t* dataSizes, cudaMemRangeAttribute* attributes, size_t numAttributes, const void* devPtr, size_t count) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = <cudaError_t>ccuda._cuMemRangeGetAttributes(data, dataSizes, <ccuda.CUmem_range_attribute*>attributes, numAttributes, <ccuda.CUdeviceptr>devPtr, count)
    if err != cudaSuccess:
        _setLastError(err)
    return err


cdef cudaError_t _cudaGetDeviceCount(int* count) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuDeviceGetCount(count)
    if err != cudaSuccess:
        _setLastError(err)
    return err


cdef cudaError_t _cudaDeviceGetAttribute(int* value, cudaDeviceAttr attr, int device) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = <cudaError_t>ccuda._cuDeviceGetAttribute(value, <ccuda.CUdevice_attribute>attr, <ccuda.CUdevice>device)
    if err != cudaSuccess:
        _setLastError(err)
    return err


cdef cudaError_t _cudaDeviceSetSharedMemConfig(cudaSharedMemConfig config) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuCtxSetSharedMemConfig(<ccuda.CUsharedconfig>config)
    if err != cudaSuccess:
        _setLastError(err)
    return err


cdef cudaError_t _cudaDeviceGetByPCIBusId(int* device, const char* pciBusId) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = <cudaError_t>ccuda._cuDeviceGetByPCIBusId(<ccuda.CUdevice*>device, pciBusId)
    if err != cudaSuccess:
        _setLastError(err)
    return err


cdef cudaError_t _cudaDeviceGetPCIBusId(char* pciBusId, int length, int device) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = <cudaError_t>ccuda._cuDeviceGetPCIBusId(pciBusId, length, <ccuda.CUdevice>device)
    if err != cudaSuccess:
        _setLastError(err)
    return err


cdef cudaError_t _cudaDeviceGetP2PAttribute(int* value, cudaDeviceP2PAttr attr, int srcDevice, int dstDevice) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = <cudaError_t>ccuda._cuDeviceGetP2PAttribute(value, <ccuda.CUdevice_P2PAttribute>attr, <ccuda.CUdevice>srcDevice, <ccuda.CUdevice>dstDevice)
    if err != cudaSuccess:
        _setLastError(err)
    return err


cdef cudaError_t _cudaArrayGetSparseProperties(cudaArraySparseProperties* sparseProperties, cudaArray_t array) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    cdef ccuda.CUDA_ARRAY_SPARSE_PROPERTIES _driver_sparseProperties
    if not sparseProperties:
        _setLastError(cudaErrorInvalidValue)
        return cudaError.cudaErrorInvalidValue
    memset(sparseProperties, 0, sizeof(cudaArraySparseProperties))

    err = <cudaError_t>ccuda._cuArrayGetSparseProperties(&_driver_sparseProperties, <ccuda.CUarray>array)
    if err == cudaSuccess:
        sparseProperties[0].miptailFirstLevel = _driver_sparseProperties.miptailFirstLevel
        sparseProperties[0].miptailSize       = _driver_sparseProperties.miptailSize
        sparseProperties[0].flags             = _driver_sparseProperties.flags
        sparseProperties[0].tileExtent.width  = _driver_sparseProperties.tileExtent.width
        sparseProperties[0].tileExtent.height = _driver_sparseProperties.tileExtent.height
        sparseProperties[0].tileExtent.depth  = _driver_sparseProperties.tileExtent.depth

    if err != cudaSuccess:
        _setLastError(err)
    return err


cdef cudaError_t _cudaMipmappedArrayGetSparseProperties(cudaArraySparseProperties* sparseProperties, cudaMipmappedArray_t mipmap) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    cdef ccuda.CUDA_ARRAY_SPARSE_PROPERTIES _driver_sparseProperties
    if not sparseProperties:
        _setLastError(cudaErrorInvalidValue)
        return cudaError.cudaErrorInvalidValue
    memset(sparseProperties, 0, sizeof(cudaArraySparseProperties))

    err = <cudaError_t>ccuda._cuMipmappedArrayGetSparseProperties(&_driver_sparseProperties, <ccuda.CUmipmappedArray>mipmap)
    if err == cudaSuccess:
        sparseProperties[0].miptailFirstLevel = _driver_sparseProperties.miptailFirstLevel
        sparseProperties[0].miptailSize       = _driver_sparseProperties.miptailSize
        sparseProperties[0].flags             = _driver_sparseProperties.flags
        sparseProperties[0].tileExtent.width  = _driver_sparseProperties.tileExtent.width
        sparseProperties[0].tileExtent.height = _driver_sparseProperties.tileExtent.height
        sparseProperties[0].tileExtent.depth  = _driver_sparseProperties.tileExtent.depth

    if err != cudaSuccess:
        _setLastError(err)
    return err


cdef cudaError_t _cudaDeviceCanAccessPeer(int* canAccessPeer, int device, int peerDevice) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuDeviceCanAccessPeer(canAccessPeer, <ccuda.CUdevice>device, <ccuda.CUdevice>peerDevice)
    if err != cudaSuccess:
        _setLastError(err)
    err = m_global.lazyInitDevice(peerDevice)
    if err != cudaSuccess:
        _setLastError(err)
        return err
    if device == peerDevice:
        canAccessPeer[0] = 0
    return err


cdef cudaError_t _cudaMemcpyPeer(void* dst, int dstDevice, const void* src, int srcDevice, size_t count) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    if dstDevice < 0 or dstDevice >= m_global._numDevices:
        return cudaErrorInvalidDevice
    if srcDevice < 0 or srcDevice >= m_global._numDevices:
        return cudaErrorInvalidDevice
    err = <cudaError_t>ccuda._cuMemcpyPeer(<ccuda.CUdeviceptr>dst, m_global._driverContext[dstDevice], <ccuda.CUdeviceptr>src, m_global._driverContext[srcDevice], count)
    if err != cudaSuccess:
        _setLastError(err)
    return err


cdef cudaError_t _cudaMemcpyPeerAsync(void* dst, int dstDevice, const void* src, int srcDevice, size_t count, cudaStream_t stream) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    if dstDevice < 0 or dstDevice >= m_global._numDevices:
        return cudaErrorInvalidDevice
    if srcDevice < 0 or srcDevice >= m_global._numDevices:
        return cudaErrorInvalidDevice
    err = <cudaError_t>ccuda._cuMemcpyPeerAsync(<ccuda.CUdeviceptr>dst, m_global._driverContext[dstDevice], <ccuda.CUdeviceptr>src, m_global._driverContext[srcDevice], count, <ccuda.CUstream>stream)
    if err != cudaSuccess:
        _setLastError(err)
    return err


cdef cudaError_t _cudaDeviceEnablePeerAccess(int peerDevice, unsigned int flags) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    if peerDevice < 0 or peerDevice >= m_global._numDevices:
        return cudaErrorInvalidDevice
    err = m_global.lazyInitDevice(peerDevice)
    if err != cudaSuccess:
        _setLastError(err)
        return err
    err = <cudaError_t>ccuda._cuCtxEnablePeerAccess(m_global._driverContext[peerDevice], flags)
    if err != cudaSuccess:
        _setLastError(err)
    return err


cdef cudaError_t _cudaDeviceDisablePeerAccess(int peerDevice) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    if peerDevice < 0 or peerDevice >= m_global._numDevices:
        return cudaErrorInvalidDevice
    err = m_global.lazyInitDevice(peerDevice)
    if err != cudaSuccess:
        _setLastError(err)
        return err
    err = <cudaError_t>ccuda._cuCtxDisablePeerAccess(m_global._driverContext[peerDevice])
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t _cudaExternalMemoryGetMappedMipmappedArray(cudaMipmappedArray_t* mipmap, cudaExternalMemory_t extMem, const cudaExternalMemoryMipmappedArrayDesc* mipmapDesc) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    cdef ccuda.CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC _driver_mipmapDesc
    memset(&_driver_mipmapDesc, 0, sizeof(_driver_mipmapDesc))
    _driver_mipmapDesc.offset = mipmapDesc[0].offset
    _driver_mipmapDesc.arrayDesc.Width = mipmapDesc[0].extent.width
    _driver_mipmapDesc.arrayDesc.Height = mipmapDesc[0].extent.height
    _driver_mipmapDesc.arrayDesc.Depth = mipmapDesc[0].extent.depth
    err_rt = getDescInfo(&mipmapDesc[0].formatDesc, <int *>&_driver_mipmapDesc.arrayDesc.NumChannels, &_driver_mipmapDesc.arrayDesc.Format)
    if err_rt != cudaError.cudaSuccess:
        _setLastError(err_rt)
        return err_rt
    _driver_mipmapDesc.arrayDesc.Flags = mipmapDesc[0].flags
    _driver_mipmapDesc.numLevels = mipmapDesc[0].numLevels

    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuExternalMemoryGetMappedMipmappedArray(<ccuda.CUmipmappedArray*>mipmap, <ccuda.CUexternalMemory>extMem, &_driver_mipmapDesc)
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t _cudaGetSurfaceObjectResourceDesc(cudaResourceDesc* pResDesc, cudaSurfaceObject_t surfObject) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    cdef ccuda.CUDA_RESOURCE_DESC _driver_pResDesc

    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuSurfObjectGetResourceDesc(&_driver_pResDesc, <ccuda.CUsurfObject>surfObject)
    memset(pResDesc, 0, sizeof(cudaResourceDesc))
    if _driver_pResDesc.resType == ccuda.CU_RESOURCE_TYPE_ARRAY:
        pResDesc[0].resType         = cudaResourceType.cudaResourceTypeArray
        pResDesc[0].res.array.array = <cudaArray_t>_driver_pResDesc.res.array.hArray
    elif _driver_pResDesc.resType == ccuda.CU_RESOURCE_TYPE_MIPMAPPED_ARRAY:
        pResDesc[0].resType = cudaResourceType.cudaResourceTypeMipmappedArray
        pResDesc[0].res.mipmap.mipmap = <cudaMipmappedArray_t>_driver_pResDesc.res.mipmap.hMipmappedArray
    elif _driver_pResDesc.resType == ccuda.CU_RESOURCE_TYPE_LINEAR:
        pResDesc[0].resType                = cudaResourceType.cudaResourceTypeLinear
        pResDesc[0].res.linear.devPtr      = <void *>_driver_pResDesc.res.linear.devPtr
        pResDesc[0].res.linear.sizeInBytes = _driver_pResDesc.res.linear.sizeInBytes
    elif _driver_pResDesc.resType == ccuda.CU_RESOURCE_TYPE_PITCH2D:
        pResDesc[0].resType                  = cudaResourceType.cudaResourceTypePitch2D
        pResDesc[0].res.pitch2D.devPtr       = <void *>_driver_pResDesc.res.pitch2D.devPtr
        pResDesc[0].res.pitch2D.pitchInBytes = _driver_pResDesc.res.pitch2D.pitchInBytes
        pResDesc[0].res.pitch2D.width        = _driver_pResDesc.res.pitch2D.width
        pResDesc[0].res.pitch2D.height       = _driver_pResDesc.res.pitch2D.height
    if _driver_pResDesc.resType == ccuda.CU_RESOURCE_TYPE_LINEAR or _driver_pResDesc.resType == ccuda.CU_RESOURCE_TYPE_PITCH2D:
        channel_size = 0
        if _driver_pResDesc.res.linear.format == ccuda.CU_AD_FORMAT_UNSIGNED_INT8:
            pResDesc[0].res.linear.desc.f = cudaChannelFormatKind.cudaChannelFormatKindUnsigned
            channel_size = 8
        elif _driver_pResDesc.res.linear.format == ccuda.CU_AD_FORMAT_UNSIGNED_INT16:
            pResDesc[0].res.linear.desc.f = cudaChannelFormatKind.cudaChannelFormatKindUnsigned
            channel_size = 16
        elif _driver_pResDesc.res.linear.format == ccuda.CU_AD_FORMAT_UNSIGNED_INT32:
            pResDesc[0].res.linear.desc.f = cudaChannelFormatKind.cudaChannelFormatKindUnsigned
            channel_size = 32
        elif _driver_pResDesc.res.linear.format == ccuda.CU_AD_FORMAT_SIGNED_INT8:
            pResDesc[0].res.linear.desc.f = cudaChannelFormatKind.cudaChannelFormatKindSigned
            channel_size = 8
        elif _driver_pResDesc.res.linear.format == ccuda.CU_AD_FORMAT_SIGNED_INT16:
            pResDesc[0].res.linear.desc.f = cudaChannelFormatKind.cudaChannelFormatKindSigned
            channel_size = 16
        elif _driver_pResDesc.res.linear.format == ccuda.CU_AD_FORMAT_SIGNED_INT32:
            pResDesc[0].res.linear.desc.f = cudaChannelFormatKind.cudaChannelFormatKindSigned
            channel_size = 32
        elif _driver_pResDesc.res.linear.format == ccuda.CU_AD_FORMAT_HALF:
            pResDesc[0].res.linear.desc.f = cudaChannelFormatKind.cudaChannelFormatKindFloat
            channel_size = 16
        elif _driver_pResDesc.res.linear.format == ccuda.CU_AD_FORMAT_FLOAT:
            pResDesc[0].res.linear.desc.f = cudaChannelFormatKind.cudaChannelFormatKindFloat
            channel_size = 32
        elif _driver_pResDesc.res.linear.format == ccuda.CU_AD_FORMAT_NV12:
            pResDesc[0].res.linear.desc.f = cudaChannelFormatKind.cudaChannelFormatKindNV12
            channel_size = 8
        else:
            _setLastError(cudaErrorInvalidChannelDescriptor)
            return cudaError.cudaErrorInvalidChannelDescriptor
        pResDesc[0].res.linear.desc.x = 0
        pResDesc[0].res.linear.desc.y = 0
        pResDesc[0].res.linear.desc.z = 0
        pResDesc[0].res.linear.desc.w = 0
        if _driver_pResDesc.res.linear.numChannels >= 4:
            pResDesc[0].res.linear.desc.w = channel_size
        if _driver_pResDesc.res.linear.numChannels >= 3:
            pResDesc[0].res.linear.desc.z = channel_size
        if _driver_pResDesc.res.linear.numChannels >= 2:
            pResDesc[0].res.linear.desc.y = channel_size
        if _driver_pResDesc.res.linear.numChannels >= 1:
            pResDesc[0].res.linear.desc.x = channel_size
        if _driver_pResDesc.res.linear.numChannels < 1 or _driver_pResDesc.res.linear.numChannels >= 5:
            _setLastError(cudaErrorInvalidChannelDescriptor)
            return cudaError.cudaErrorInvalidChannelDescriptor

    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t _cudaGraphKernelNodeGetParams(cudaGraphNode_t node, cudaKernelNodeParams* pNodeParams) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    cdef ccuda.CUDA_KERNEL_NODE_PARAMS _driver_pNodeParams

    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuGraphKernelNodeGetParams(<ccuda.CUgraphNode>node, &_driver_pNodeParams)
    pNodeParams[0].func = <void*>_driver_pNodeParams.func
    pNodeParams[0].gridDim.x = _driver_pNodeParams.gridDimX
    pNodeParams[0].gridDim.y = _driver_pNodeParams.gridDimY
    pNodeParams[0].gridDim.z = _driver_pNodeParams.gridDimZ
    pNodeParams[0].blockDim.x = _driver_pNodeParams.blockDimX
    pNodeParams[0].blockDim.y = _driver_pNodeParams.blockDimY
    pNodeParams[0].blockDim.z = _driver_pNodeParams.blockDimZ
    pNodeParams[0].sharedMemBytes = _driver_pNodeParams.sharedMemBytes
    pNodeParams[0].kernelParams = _driver_pNodeParams.kernelParams
    pNodeParams[0].extra = _driver_pNodeParams.extra

    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t _cudaExternalMemoryGetMappedBuffer(void** devPtr, cudaExternalMemory_t extMem, const cudaExternalMemoryBufferDesc* bufferDesc) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    cdef ccuda.CUDA_EXTERNAL_MEMORY_BUFFER_DESC _driver_bufferDesc
    memset(&_driver_bufferDesc, 0, sizeof(_driver_bufferDesc))
    _driver_bufferDesc.offset = bufferDesc[0].offset
    _driver_bufferDesc.size = bufferDesc[0].size
    _driver_bufferDesc.flags = bufferDesc[0].flags

    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuExternalMemoryGetMappedBuffer(<ccuda.CUdeviceptr*>devPtr, <ccuda.CUexternalMemory>extMem, &_driver_bufferDesc)
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t _cudaImportExternalMemory(cudaExternalMemory_t* extMem_out, const cudaExternalMemoryHandleDesc* memHandleDesc) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    cdef ccuda.CUDA_EXTERNAL_MEMORY_HANDLE_DESC _driver_memHandleDesc
    memset(&_driver_memHandleDesc, 0, sizeof(_driver_memHandleDesc))

    if memHandleDesc[0].type == cudaExternalMemoryHandleType.cudaExternalMemoryHandleTypeOpaqueFd:
        _driver_memHandleDesc.type = ccuda.CUexternalMemoryHandleType_enum.CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD
        _driver_memHandleDesc.handle.fd = memHandleDesc[0].handle.fd
    elif memHandleDesc[0].type == cudaExternalMemoryHandleType.cudaExternalMemoryHandleTypeOpaqueWin32:
        _driver_memHandleDesc.type = ccuda.CUexternalMemoryHandleType_enum.CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32
        _driver_memHandleDesc.handle.win32.handle = memHandleDesc[0].handle.win32.handle
        _driver_memHandleDesc.handle.win32.name = memHandleDesc[0].handle.win32.name
    elif memHandleDesc[0].type == cudaExternalMemoryHandleType.cudaExternalMemoryHandleTypeOpaqueWin32Kmt:
        _driver_memHandleDesc.type = ccuda.CUexternalMemoryHandleType_enum.CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT
        _driver_memHandleDesc.handle.win32.handle = memHandleDesc[0].handle.win32.handle
        _driver_memHandleDesc.handle.win32.name = memHandleDesc[0].handle.win32.name
    elif memHandleDesc[0].type == cudaExternalMemoryHandleType.cudaExternalMemoryHandleTypeD3D12Heap:
        _driver_memHandleDesc.type = ccuda.CUexternalMemoryHandleType_enum.CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_HEAP
        _driver_memHandleDesc.handle.win32.handle = memHandleDesc[0].handle.win32.handle
        _driver_memHandleDesc.handle.win32.name = memHandleDesc[0].handle.win32.name
    elif memHandleDesc[0].type == cudaExternalMemoryHandleType.cudaExternalMemoryHandleTypeD3D12Resource:
        _driver_memHandleDesc.type = ccuda.CUexternalMemoryHandleType_enum.CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE
        _driver_memHandleDesc.handle.win32.handle = memHandleDesc[0].handle.win32.handle
        _driver_memHandleDesc.handle.win32.name = memHandleDesc[0].handle.win32.name
    elif memHandleDesc[0].type == cudaExternalMemoryHandleType.cudaExternalMemoryHandleTypeD3D11Resource:
        _driver_memHandleDesc.type = ccuda.CUexternalMemoryHandleType_enum.CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_RESOURCE
        _driver_memHandleDesc.handle.win32.handle = memHandleDesc[0].handle.win32.handle
        _driver_memHandleDesc.handle.win32.name = memHandleDesc[0].handle.win32.name
    elif memHandleDesc[0].type == cudaExternalMemoryHandleType.cudaExternalMemoryHandleTypeD3D11ResourceKmt:
        _driver_memHandleDesc.type = ccuda.CUexternalMemoryHandleType_enum.CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_RESOURCE_KMT
        _driver_memHandleDesc.handle.win32.handle = memHandleDesc[0].handle.win32.handle
        _driver_memHandleDesc.handle.win32.name = memHandleDesc[0].handle.win32.name
    elif memHandleDesc[0].type == cudaExternalMemoryHandleType.cudaExternalMemoryHandleTypeNvSciBuf:
        _driver_memHandleDesc.type = ccuda.CUexternalMemoryHandleType_enum.CU_EXTERNAL_MEMORY_HANDLE_TYPE_NVSCIBUF
        _driver_memHandleDesc.handle.nvSciBufObject = memHandleDesc[0].handle.nvSciBufObject
    _driver_memHandleDesc.size = memHandleDesc[0].size
    _driver_memHandleDesc.flags = memHandleDesc[0].flags

    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuImportExternalMemory(<ccuda.CUexternalMemory*>extMem_out, &_driver_memHandleDesc)
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t _cudaCreateSurfaceObject(cudaSurfaceObject_t* pSurfObject, const cudaResourceDesc* pResDesc) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    cdef ccuda.CUDA_RESOURCE_DESC _driver_pResDesc
    memset(&_driver_pResDesc, 0, sizeof(_driver_pResDesc))
    err = toDriverCudaResourceDesc(&_driver_pResDesc, pResDesc)
    if err != cudaSuccess:
        _setLastError(err)
        return err

    err = <cudaError_t>ccuda._cuSurfObjectCreate(<ccuda.CUsurfObject*>pSurfObject, &_driver_pResDesc)
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t _cudaGetTextureObjectResourceDesc(cudaResourceDesc* pResDesc, cudaTextureObject_t texObject) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    cdef ccuda.CUDA_RESOURCE_DESC _driver_pResDesc
    memset(&_driver_pResDesc, 0, sizeof(_driver_pResDesc))
    err = toDriverCudaResourceDesc(&_driver_pResDesc, pResDesc)
    if err != cudaSuccess:
        _setLastError(err)
        return err

    err = <cudaError_t>ccuda._cuTexObjectGetResourceDesc(&_driver_pResDesc, <ccuda.CUtexObject>texObject)
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t _cudaProfilerInitialize(const char* configFile, const char* outputFile, cudaOutputMode_t outputMode) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuProfilerInitialize(configFile, outputFile, <ccuda.CUoutput_mode>outputMode)
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t _cudaGraphicsEGLRegisterImage(cudaGraphicsResource_t* pCudaResource, EGLImageKHR image, unsigned int flags) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuGraphicsEGLRegisterImage(<ccuda.CUgraphicsResource*>pCudaResource, image, flags)
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t _cudaEGLStreamProducerPresentFrame(cudaEglStreamConnection* conn, cudaEglFrame eglframe, cudaStream_t* pStream) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    cdef ccuda.CUeglFrame cueglFrame
    err = getDriverEglFrame(&cueglFrame, eglframe)
    if err != cudaSuccess:
        _setLastError(err)
        return err
    err = <cudaError_t>ccuda._cuEGLStreamProducerPresentFrame(<ccuda.CUeglStreamConnection*>conn, cueglFrame, pStream)
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t _cudaEGLStreamProducerReturnFrame(cudaEglStreamConnection* conn, cudaEglFrame* eglframe, cudaStream_t* pStream) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    if eglframe == NULL:
        err = cudaErrorInvalidResourceHandle
        _setLastError(err)
        return err
    cdef ccuda.CUeglFrame cueglFrame
    err = <cudaError_t>ccuda._cuEGLStreamProducerReturnFrame(<ccuda.CUeglStreamConnection*>conn, &cueglFrame, pStream)
    if err != cudaSuccess:
        _setLastError(err)
        return err
    err = getRuntimeEglFrame(eglframe, cueglFrame)
    if err != cudaSuccess:
        _setLastError(err)
        return err
    return err

cdef cudaError_t _cudaGraphicsResourceGetMappedEglFrame(cudaEglFrame* eglFrame, cudaGraphicsResource_t resource, unsigned int index, unsigned int mipLevel) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    cdef ccuda.CUeglFrame cueglFrame
    memset(&cueglFrame, 0, sizeof(cueglFrame))
    err = <cudaError_t>ccuda._cuGraphicsResourceGetMappedEglFrame(&cueglFrame, <ccuda.CUgraphicsResource>resource, index, mipLevel)
    if err != cudaSuccess:
        _setLastError(err)
        return err
    err = getRuntimeEglFrame(eglFrame, cueglFrame)
    if err != cudaSuccess:
        _setLastError(err)
        return err
    return err

cdef cudaError_t _cudaVDPAUSetVDPAUDevice(int device, VdpDevice vdpDevice, VdpGetProcAddress* vdpGetProcAddress) nogil except ?cudaErrorCallRequiresNewerDriver:
    return cudaErrorNotSupported

cdef cudaError_t _cudaArrayGetMemoryRequirements(cudaArrayMemoryRequirements* memoryRequirements, cudaArray_t array, int device) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    cdef ccuda.CUDA_ARRAY_MEMORY_REQUIREMENTS driverMemoryRequirements
    if memoryRequirements == NULL:
        _setLastError(cudaErrorInvalidValue)
        return cudaErrorInvalidValue

    memset(memoryRequirements, 0, sizeof(memoryRequirements[0]))
    err = <cudaError_t>ccuda._cuArrayGetMemoryRequirements(&driverMemoryRequirements, <ccuda.CUarray>array, device)
    if err != cudaSuccess:
        _setLastError(err)
        return err

    memoryRequirements[0].size = driverMemoryRequirements.size
    memoryRequirements[0].alignment = driverMemoryRequirements.alignment
    return cudaSuccess

cdef cudaError_t _cudaMipmappedArrayGetMemoryRequirements(cudaArrayMemoryRequirements* memoryRequirements, cudaMipmappedArray_t mipmap, int device) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    cdef ccuda.CUDA_ARRAY_MEMORY_REQUIREMENTS driverMemoryRequirements
    if memoryRequirements == NULL:
        _setLastError(cudaErrorInvalidValue)
        return cudaErrorInvalidValue

    memset(memoryRequirements, 0, sizeof(memoryRequirements[0]))
    err = <cudaError_t>ccuda._cuMipmappedArrayGetMemoryRequirements(&driverMemoryRequirements, <ccuda.CUmipmappedArray>mipmap, device)
    if err != cudaSuccess:
        _setLastError(err)
        return err

    memoryRequirements[0].size = driverMemoryRequirements.size
    memoryRequirements[0].alignment = driverMemoryRequirements.alignment
    return cudaSuccess
