# Copyright 2021 NVIDIA Corporation.  All rights reserved.
#
# Please refer to the NVIDIA end user license agreement (EULA) associated
# with this source code for terms and conditions that govern your use of
# this software. Any use, reproduction, disclosure, or distribution of
# this software and related documentation outside the terms of the EULA
# is strictly prohibited.
cimport cudapython._cuda.ccuda as ccuda
from cudapython._lib.ccudart.ccudart cimport *
from cudapython._lib.ccudart.utils cimport *
from libc.stdlib cimport malloc, free, calloc
from libc.string cimport memset, memcpy, strncmp
from libcpp cimport bool

cdef cudaPythonGlobal m_global = globalGetInstance()

cdef cudaError_t cudaDeviceReset() nogil except ?cudaErrorCallRequiresNewerDriver:
    return _cudaDeviceReset()

cdef cudaError_t cudaDeviceSynchronize() nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUresult err
    err = ccuda._cuCtxSynchronize()
    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaDeviceSetLimit(cudaLimit limit, size_t value) nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUresult err
    err = ccuda._cuCtxSetLimit(<ccuda.CUlimit>limit, value)
    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaDeviceGetLimit(size_t* pValue, cudaLimit limit) nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUresult err
    err = ccuda._cuCtxGetLimit(pValue, <ccuda.CUlimit>limit)
    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaDeviceGetTexture1DLinearMaxWidth(size_t* maxWidthInElements, const cudaChannelFormatDesc* fmtDesc, int device) nogil except ?cudaErrorCallRequiresNewerDriver:
    return _cudaDeviceGetTexture1DLinearMaxWidth(maxWidthInElements, fmtDesc, device)

cdef cudaError_t cudaDeviceGetCacheConfig(cudaFuncCache* pCacheConfig) nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUresult err
    err = ccuda._cuCtxGetCacheConfig(<ccuda.CUfunc_cache*>pCacheConfig)
    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaDeviceGetStreamPriorityRange(int* leastPriority, int* greatestPriority) nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUresult err
    err = ccuda._cuCtxGetStreamPriorityRange(leastPriority, greatestPriority)
    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaDeviceSetCacheConfig(cudaFuncCache cacheConfig) nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUresult err
    err = ccuda._cuCtxSetCacheConfig(<ccuda.CUfunc_cache>cacheConfig)
    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaDeviceGetSharedMemConfig(cudaSharedMemConfig* pConfig) nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUresult err
    err = ccuda._cuCtxGetSharedMemConfig(<ccuda.CUsharedconfig*>pConfig)
    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaDeviceSetSharedMemConfig(cudaSharedMemConfig config) nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUresult err
    err = ccuda._cuCtxSetSharedMemConfig(<ccuda.CUsharedconfig>config)
    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaDeviceGetByPCIBusId(int* device, const char* pciBusId) nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUresult err
    err = ccuda._cuDeviceGetByPCIBusId(<ccuda.CUdevice*>device, pciBusId)
    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaDeviceGetPCIBusId(char* pciBusId, int length, int device) nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUresult err
    err = ccuda._cuDeviceGetPCIBusId(pciBusId, length, <ccuda.CUdevice>device)
    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaIpcGetEventHandle(cudaIpcEventHandle_t* handle, cudaEvent_t event) nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUresult err
    err = ccuda._cuIpcGetEventHandle(<ccuda.CUipcEventHandle*>handle, <ccuda.CUevent>event)
    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaIpcOpenEventHandle(cudaEvent_t* event, cudaIpcEventHandle_t handle) nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUipcEventHandle _driver_handle
    memcpy(&_driver_handle, &handle, sizeof(_driver_handle))
    cdef ccuda.CUresult err
    err = ccuda._cuIpcOpenEventHandle(<ccuda.CUevent*>event, _driver_handle)
    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaIpcGetMemHandle(cudaIpcMemHandle_t* handle, void* devPtr) nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUresult err
    err = ccuda._cuIpcGetMemHandle(<ccuda.CUipcMemHandle*>handle, <ccuda.CUdeviceptr>devPtr)
    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaIpcOpenMemHandle(void** devPtr, cudaIpcMemHandle_t handle, unsigned int flags) nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUipcMemHandle _driver_handle
    memcpy(&_driver_handle, &handle, sizeof(_driver_handle))
    cdef ccuda.CUresult err
    err = ccuda._cuIpcOpenMemHandle_v2(<ccuda.CUdeviceptr*>devPtr, _driver_handle, flags)
    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaIpcCloseMemHandle(void* devPtr) nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUresult err
    err = ccuda._cuIpcCloseMemHandle(<ccuda.CUdeviceptr>devPtr)
    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaDeviceFlushGPUDirectRDMAWrites(cudaFlushGPUDirectRDMAWritesTarget target, cudaFlushGPUDirectRDMAWritesScope scope) nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUresult err
    err = ccuda._cuFlushGPUDirectRDMAWrites(<ccuda.CUflushGPUDirectRDMAWritesTarget>target, <ccuda.CUflushGPUDirectRDMAWritesScope>scope)
    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaThreadExit() nogil except ?cudaErrorCallRequiresNewerDriver:
    return _cudaThreadExit()

cdef cudaError_t cudaThreadSynchronize() nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUresult err
    err = ccuda._cuCtxSynchronize()
    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaThreadSetLimit(cudaLimit limit, size_t value) nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUresult err
    err = ccuda._cuCtxSetLimit(<ccuda.CUlimit>limit, value)
    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaThreadGetLimit(size_t* pValue, cudaLimit limit) nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUresult err
    err = ccuda._cuCtxGetLimit(pValue, <ccuda.CUlimit>limit)
    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaThreadGetCacheConfig(cudaFuncCache* pCacheConfig) nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUresult err
    err = ccuda._cuCtxGetCacheConfig(<ccuda.CUfunc_cache*>pCacheConfig)
    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaThreadSetCacheConfig(cudaFuncCache cacheConfig) nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUresult err
    err = ccuda._cuCtxSetCacheConfig(<ccuda.CUfunc_cache>cacheConfig)
    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaGetLastError() nogil except ?cudaErrorCallRequiresNewerDriver:
    return _cudaGetLastError()

cdef cudaError_t cudaPeekAtLastError() nogil except ?cudaErrorCallRequiresNewerDriver:
    return _cudaPeekAtLastError()

cdef const char* cudaGetErrorName(cudaError_t error) nogil except ?NULL:
    return _cudaGetErrorName(error)

cdef const char* cudaGetErrorString(cudaError_t error) nogil except ?NULL:
    return _cudaGetErrorString(error)

cdef cudaError_t cudaGetDeviceCount(int* count) nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUresult err
    err = ccuda._cuDeviceGetCount(count)
    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaGetDeviceProperties(cudaDeviceProp* prop, int device) nogil except ?cudaErrorCallRequiresNewerDriver:
    return _cudaGetDeviceProperties(prop, device)

cdef cudaError_t cudaDeviceGetAttribute(int* value, cudaDeviceAttr attr, int device) nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUresult err
    err = ccuda._cuDeviceGetAttribute(value, <ccuda.CUdevice_attribute>attr, <ccuda.CUdevice>device)
    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaDeviceGetDefaultMemPool(cudaMemPool_t* memPool, int device) nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUresult err
    err = ccuda._cuDeviceGetDefaultMemPool(<ccuda.CUmemoryPool*>memPool, <ccuda.CUdevice>device)
    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaDeviceSetMemPool(int device, cudaMemPool_t memPool) nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUresult err
    err = ccuda._cuDeviceSetMemPool(<ccuda.CUdevice>device, <ccuda.CUmemoryPool>memPool)
    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaDeviceGetMemPool(cudaMemPool_t* memPool, int device) nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUresult err
    err = ccuda._cuDeviceGetMemPool(<ccuda.CUmemoryPool*>memPool, <ccuda.CUdevice>device)
    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaDeviceGetNvSciSyncAttributes(void* nvSciSyncAttrList, int device, int flags) nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUresult err
    err = ccuda._cuDeviceGetNvSciSyncAttributes(nvSciSyncAttrList, <ccuda.CUdevice>device, flags)
    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaDeviceGetP2PAttribute(int* value, cudaDeviceP2PAttr attr, int srcDevice, int dstDevice) nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUresult err
    err = ccuda._cuDeviceGetP2PAttribute(value, <ccuda.CUdevice_P2PAttribute>attr, <ccuda.CUdevice>srcDevice, <ccuda.CUdevice>dstDevice)
    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaChooseDevice(int* device, const cudaDeviceProp* prop) nogil except ?cudaErrorCallRequiresNewerDriver:
    return _cudaChooseDevice(device, prop)

cdef cudaError_t cudaSetDevice(int device) nogil except ?cudaErrorCallRequiresNewerDriver:
    return _cudaSetDevice(device)

cdef cudaError_t cudaGetDevice(int* device) nogil except ?cudaErrorCallRequiresNewerDriver:
    return _cudaGetDevice(device)

cdef cudaError_t cudaSetDeviceFlags(unsigned int flags) nogil except ?cudaErrorCallRequiresNewerDriver:
    return _cudaSetDeviceFlags(flags)

cdef cudaError_t cudaGetDeviceFlags(unsigned int* flags) nogil except ?cudaErrorCallRequiresNewerDriver:
    return _cudaGetDeviceFlags(flags)

cdef cudaError_t cudaStreamCreate(cudaStream_t* pStream) nogil except ?cudaErrorCallRequiresNewerDriver:
    return _cudaStreamCreate(pStream)

cdef cudaError_t cudaStreamCreateWithFlags(cudaStream_t* pStream, unsigned int flags) nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUresult err
    err = ccuda._cuStreamCreate(<ccuda.CUstream*>pStream, flags)
    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaStreamCreateWithPriority(cudaStream_t* pStream, unsigned int flags, int priority) nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUresult err
    err = ccuda._cuStreamCreateWithPriority(<ccuda.CUstream*>pStream, flags, priority)
    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaStreamGetPriority(cudaStream_t hStream, int* priority) nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUresult err
    err = ccuda._cuStreamGetPriority(<ccuda.CUstream>hStream, priority)
    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaStreamGetFlags(cudaStream_t hStream, unsigned int* flags) nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUresult err
    err = ccuda._cuStreamGetFlags(<ccuda.CUstream>hStream, flags)
    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaCtxResetPersistingL2Cache() nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUresult err
    err = ccuda._cuCtxResetPersistingL2Cache()
    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaStreamCopyAttributes(cudaStream_t dst, cudaStream_t src) nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUresult err
    err = ccuda._cuStreamCopyAttributes(<ccuda.CUstream>dst, <ccuda.CUstream>src)
    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaStreamGetAttribute(cudaStream_t hStream, cudaStreamAttrID attr, cudaStreamAttrValue* value_out) nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUresult err
    err = ccuda._cuStreamGetAttribute(<ccuda.CUstream>hStream, <ccuda.CUstreamAttrID>attr, <ccuda.CUstreamAttrValue*>value_out)
    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaStreamSetAttribute(cudaStream_t hStream, cudaStreamAttrID attr, const cudaStreamAttrValue* value) nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUresult err
    err = ccuda._cuStreamSetAttribute(<ccuda.CUstream>hStream, <ccuda.CUstreamAttrID>attr, <ccuda.CUstreamAttrValue*>value)
    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaStreamDestroy(cudaStream_t stream) nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUresult err
    err = ccuda._cuStreamDestroy_v2(<ccuda.CUstream>stream)
    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned int flags) nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUresult err
    err = ccuda._cuStreamWaitEvent(<ccuda.CUstream>stream, <ccuda.CUevent>event, flags)
    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaStreamAddCallback(cudaStream_t stream, cudaStreamCallback_t callback, void* userData, unsigned int flags) nogil except ?cudaErrorCallRequiresNewerDriver:
    return _cudaStreamAddCallback(stream, callback, userData, flags)

cdef cudaError_t cudaStreamSynchronize(cudaStream_t stream) nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUresult err
    err = ccuda._cuStreamSynchronize(<ccuda.CUstream>stream)
    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaStreamQuery(cudaStream_t stream) nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUresult err
    err = ccuda._cuStreamQuery(<ccuda.CUstream>stream)
    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaStreamAttachMemAsync(cudaStream_t stream, void* devPtr, size_t length, unsigned int flags) nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUresult err
    err = ccuda._cuStreamAttachMemAsync(<ccuda.CUstream>stream, <ccuda.CUdeviceptr>devPtr, length, flags)
    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaStreamBeginCapture(cudaStream_t stream, cudaStreamCaptureMode mode) nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUresult err
    err = ccuda._cuStreamBeginCapture_v2(<ccuda.CUstream>stream, <ccuda.CUstreamCaptureMode>mode)
    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaThreadExchangeStreamCaptureMode(cudaStreamCaptureMode* mode) nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUresult err
    err = ccuda._cuThreadExchangeStreamCaptureMode(<ccuda.CUstreamCaptureMode*>mode)
    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaStreamEndCapture(cudaStream_t stream, cudaGraph_t* pGraph) nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUresult err
    err = ccuda._cuStreamEndCapture(<ccuda.CUstream>stream, <ccuda.CUgraph*>pGraph)
    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaStreamIsCapturing(cudaStream_t stream, cudaStreamCaptureStatus* pCaptureStatus) nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUresult err
    err = ccuda._cuStreamIsCapturing(<ccuda.CUstream>stream, <ccuda.CUstreamCaptureStatus*>pCaptureStatus)
    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaStreamGetCaptureInfo(cudaStream_t stream, cudaStreamCaptureStatus* pCaptureStatus, unsigned long long* pId) nogil except ?cudaErrorCallRequiresNewerDriver:
    return _cudaStreamGetCaptureInfo(stream, pCaptureStatus, pId)

cdef cudaError_t cudaStreamGetCaptureInfo_v2(cudaStream_t stream, cudaStreamCaptureStatus* captureStatus_out, unsigned long long* id_out, cudaGraph_t* graph_out, const cudaGraphNode_t** dependencies_out, size_t* numDependencies_out) nogil except ?cudaErrorCallRequiresNewerDriver:
    return _cudaStreamGetCaptureInfo_v2(stream, captureStatus_out, id_out, graph_out, dependencies_out, numDependencies_out)

cdef cudaError_t cudaStreamUpdateCaptureDependencies(cudaStream_t stream, cudaGraphNode_t* dependencies, size_t numDependencies, unsigned int flags) nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUresult err
    err = ccuda._cuStreamUpdateCaptureDependencies(<ccuda.CUstream>stream, <ccuda.CUgraphNode*>dependencies, numDependencies, flags)
    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaEventCreate(cudaEvent_t* event) nogil except ?cudaErrorCallRequiresNewerDriver:
    return _cudaEventCreate(event)

cdef cudaError_t cudaEventCreateWithFlags(cudaEvent_t* event, unsigned int flags) nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUresult err
    err = ccuda._cuEventCreate(<ccuda.CUevent*>event, flags)
    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream) nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUresult err
    err = ccuda._cuEventRecord(<ccuda.CUevent>event, <ccuda.CUstream>stream)
    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaEventRecordWithFlags(cudaEvent_t event, cudaStream_t stream, unsigned int flags) nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUresult err
    err = ccuda._cuEventRecordWithFlags(<ccuda.CUevent>event, <ccuda.CUstream>stream, flags)
    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaEventQuery(cudaEvent_t event) nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUresult err
    err = ccuda._cuEventQuery(<ccuda.CUevent>event)
    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaEventSynchronize(cudaEvent_t event) nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUresult err
    err = ccuda._cuEventSynchronize(<ccuda.CUevent>event)
    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaEventDestroy(cudaEvent_t event) nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUresult err
    err = ccuda._cuEventDestroy_v2(<ccuda.CUevent>event)
    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaEventElapsedTime(float* ms, cudaEvent_t start, cudaEvent_t end) nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUresult err
    err = ccuda._cuEventElapsedTime(ms, <ccuda.CUevent>start, <ccuda.CUevent>end)
    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaImportExternalMemory(cudaExternalMemory_t* extMem_out, const cudaExternalMemoryHandleDesc* memHandleDesc) nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
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

    cdef ccuda.CUresult err
    err = ccuda._cuImportExternalMemory(<ccuda.CUexternalMemory*>extMem_out, &_driver_memHandleDesc)
    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaExternalMemoryGetMappedBuffer(void** devPtr, cudaExternalMemory_t extMem, const cudaExternalMemoryBufferDesc* bufferDesc) nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUDA_EXTERNAL_MEMORY_BUFFER_DESC _driver_bufferDesc
    memset(&_driver_bufferDesc, 0, sizeof(_driver_bufferDesc))
    _driver_bufferDesc.offset = bufferDesc[0].offset
    _driver_bufferDesc.size = bufferDesc[0].size
    _driver_bufferDesc.flags = bufferDesc[0].flags

    cdef ccuda.CUresult err
    err = ccuda._cuExternalMemoryGetMappedBuffer(<ccuda.CUdeviceptr*>devPtr, <ccuda.CUexternalMemory>extMem, &_driver_bufferDesc)
    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaExternalMemoryGetMappedMipmappedArray(cudaMipmappedArray_t* mipmap, cudaExternalMemory_t extMem, const cudaExternalMemoryMipmappedArrayDesc* mipmapDesc) nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
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

    cdef ccuda.CUresult err
    err = ccuda._cuExternalMemoryGetMappedMipmappedArray(<ccuda.CUmipmappedArray*>mipmap, <ccuda.CUexternalMemory>extMem, &_driver_mipmapDesc)
    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaDestroyExternalMemory(cudaExternalMemory_t extMem) nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUresult err
    err = ccuda._cuDestroyExternalMemory(<ccuda.CUexternalMemory>extMem)
    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaImportExternalSemaphore(cudaExternalSemaphore_t* extSem_out, const cudaExternalSemaphoreHandleDesc* semHandleDesc) nogil except ?cudaErrorCallRequiresNewerDriver:
    return _cudaImportExternalSemaphore(extSem_out, semHandleDesc)

cdef cudaError_t cudaSignalExternalSemaphoresAsync(const cudaExternalSemaphore_t* extSemArray, const cudaExternalSemaphoreSignalParams* paramsArray, unsigned int numExtSems, cudaStream_t stream) nogil except ?cudaErrorCallRequiresNewerDriver:
    return _cudaSignalExternalSemaphoresAsync(extSemArray, paramsArray, numExtSems, stream)

cdef cudaError_t cudaWaitExternalSemaphoresAsync(const cudaExternalSemaphore_t* extSemArray, const cudaExternalSemaphoreWaitParams* paramsArray, unsigned int numExtSems, cudaStream_t stream) nogil except ?cudaErrorCallRequiresNewerDriver:
    return _cudaWaitExternalSemaphoresAsync(extSemArray, paramsArray, numExtSems, stream)

cdef cudaError_t cudaDestroyExternalSemaphore(cudaExternalSemaphore_t extSem) nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUresult err
    err = ccuda._cuDestroyExternalSemaphore(<ccuda.CUexternalSemaphore>extSem)
    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaFuncSetCacheConfig(const void* func, cudaFuncCache cacheConfig) nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUresult err
    err = ccuda._cuFuncSetCacheConfig(<ccuda.CUfunction>func, <ccuda.CUfunc_cache>cacheConfig)
    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaFuncSetSharedMemConfig(const void* func, cudaSharedMemConfig config) nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUresult err
    err = ccuda._cuFuncSetSharedMemConfig(<ccuda.CUfunction>func, <ccuda.CUsharedconfig>config)
    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaFuncGetAttributes(cudaFuncAttributes* attr, const void* func) nogil except ?cudaErrorCallRequiresNewerDriver:
    return _cudaFuncGetAttributes(attr, func)

cdef cudaError_t cudaFuncSetAttribute(const void* func, cudaFuncAttribute attr, int value) nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUresult err
    err = ccuda._cuFuncSetAttribute(<ccuda.CUfunction>func, <ccuda.CUfunction_attribute>attr, value)
    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaSetDoubleForDevice(double* d) nogil except ?cudaErrorCallRequiresNewerDriver:
    return _cudaSetDoubleForDevice(d)

cdef cudaError_t cudaSetDoubleForHost(double* d) nogil except ?cudaErrorCallRequiresNewerDriver:
    return _cudaSetDoubleForHost(d)

cdef cudaError_t cudaLaunchHostFunc(cudaStream_t stream, cudaHostFn_t fn, void* userData) nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUresult err
    err = ccuda._cuLaunchHostFunc(<ccuda.CUstream>stream, <ccuda.CUhostFn>fn, userData)
    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(int* numBlocks, const void* func, int blockSize, size_t dynamicSMemSize) nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUresult err
    err = ccuda._cuOccupancyMaxActiveBlocksPerMultiprocessor(numBlocks, <ccuda.CUfunction>func, blockSize, dynamicSMemSize)
    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaOccupancyAvailableDynamicSMemPerBlock(size_t* dynamicSmemSize, const void* func, int numBlocks, int blockSize) nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUresult err
    err = ccuda._cuOccupancyAvailableDynamicSMemPerBlock(dynamicSmemSize, <ccuda.CUfunction>func, numBlocks, blockSize)
    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int* numBlocks, const void* func, int blockSize, size_t dynamicSMemSize, unsigned int flags) nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUresult err
    err = ccuda._cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(numBlocks, <ccuda.CUfunction>func, blockSize, dynamicSMemSize, flags)
    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaMallocManaged(void** devPtr, size_t size, unsigned int flags) nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUresult err
    err = ccuda._cuMemAllocManaged(<ccuda.CUdeviceptr*>devPtr, size, flags)
    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaMalloc(void** devPtr, size_t size) nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUresult err
    err = ccuda._cuMemAlloc_v2(<ccuda.CUdeviceptr*>devPtr, size)
    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaMallocHost(void** ptr, size_t size) nogil except ?cudaErrorCallRequiresNewerDriver:
    return _cudaMallocHost(ptr, size)

cdef cudaError_t cudaMallocPitch(void** devPtr, size_t* pitch, size_t width, size_t height) nogil except ?cudaErrorCallRequiresNewerDriver:
    return _cudaMallocPitch(devPtr, pitch, width, height)

cdef cudaError_t cudaMallocArray(cudaArray_t* array, const cudaChannelFormatDesc* desc, size_t width, size_t height, unsigned int flags) nogil except ?cudaErrorCallRequiresNewerDriver:
    return _cudaMallocArray(array, desc, width, height, flags)

cdef cudaError_t cudaFree(void* devPtr) nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUresult err
    err = ccuda._cuMemFree_v2(<ccuda.CUdeviceptr>devPtr)
    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaFreeHost(void* ptr) nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUresult err
    err = ccuda._cuMemFreeHost(ptr)
    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaFreeArray(cudaArray_t array) nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUresult err
    err = ccuda._cuArrayDestroy(<ccuda.CUarray>array)
    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaFreeMipmappedArray(cudaMipmappedArray_t mipmappedArray) nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUresult err
    err = ccuda._cuMipmappedArrayDestroy(<ccuda.CUmipmappedArray>mipmappedArray)
    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaHostAlloc(void** pHost, size_t size, unsigned int flags) nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUresult err
    err = ccuda._cuMemHostAlloc(pHost, size, flags)
    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaHostRegister(void* ptr, size_t size, unsigned int flags) nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUresult err
    err = ccuda._cuMemHostRegister_v2(ptr, size, flags)
    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaHostUnregister(void* ptr) nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUresult err
    err = ccuda._cuMemHostUnregister(ptr)
    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaHostGetDevicePointer(void** pDevice, void* pHost, unsigned int flags) nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUresult err
    err = ccuda._cuMemHostGetDevicePointer_v2(<ccuda.CUdeviceptr*>pDevice, pHost, flags)
    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaHostGetFlags(unsigned int* pFlags, void* pHost) nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUresult err
    err = ccuda._cuMemHostGetFlags(pFlags, pHost)
    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaMalloc3D(cudaPitchedPtr* pitchedDevPtr, cudaExtent extent) nogil except ?cudaErrorCallRequiresNewerDriver:
    return _cudaMalloc3D(pitchedDevPtr, extent)

cdef cudaError_t cudaMalloc3DArray(cudaArray_t* array, const cudaChannelFormatDesc* desc, cudaExtent extent, unsigned int flags) nogil except ?cudaErrorCallRequiresNewerDriver:
    return _cudaMalloc3DArray(array, desc, extent, flags)

cdef cudaError_t cudaMallocMipmappedArray(cudaMipmappedArray_t* mipmappedArray, const cudaChannelFormatDesc* desc, cudaExtent extent, unsigned int numLevels, unsigned int flags) nogil except ?cudaErrorCallRequiresNewerDriver:
    return _cudaMallocMipmappedArray(mipmappedArray, desc, extent, numLevels, flags)

cdef cudaError_t cudaGetMipmappedArrayLevel(cudaArray_t* levelArray, cudaMipmappedArray_const_t mipmappedArray, unsigned int level) nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUresult err
    err = ccuda._cuMipmappedArrayGetLevel(<ccuda.CUarray*>levelArray, <ccuda.CUmipmappedArray>mipmappedArray, level)
    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaMemcpy3D(const cudaMemcpy3DParms* p) nogil except ?cudaErrorCallRequiresNewerDriver:
    return _cudaMemcpy3D(p)

cdef cudaError_t cudaMemcpy3DPeer(const cudaMemcpy3DPeerParms* p) nogil except ?cudaErrorCallRequiresNewerDriver:
    return _cudaMemcpy3DPeer(p)

cdef cudaError_t cudaMemcpy3DAsync(const cudaMemcpy3DParms* p, cudaStream_t stream) nogil except ?cudaErrorCallRequiresNewerDriver:
    return _cudaMemcpy3DAsync(p, stream)

cdef cudaError_t cudaMemcpy3DPeerAsync(const cudaMemcpy3DPeerParms* p, cudaStream_t stream) nogil except ?cudaErrorCallRequiresNewerDriver:
    return _cudaMemcpy3DPeerAsync(p, stream)

cdef cudaError_t cudaMemGetInfo(size_t* free, size_t* total) nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUresult err
    err = ccuda._cuMemGetInfo_v2(free, total)
    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaArrayGetInfo(cudaChannelFormatDesc* desc, cudaExtent* extent, unsigned int* flags, cudaArray_t array) nogil except ?cudaErrorCallRequiresNewerDriver:
    return _cudaArrayGetInfo(desc, extent, flags, array)

cdef cudaError_t cudaArrayGetPlane(cudaArray_t* pPlaneArray, cudaArray_t hArray, unsigned int planeIdx) nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUresult err
    err = ccuda._cuArrayGetPlane(<ccuda.CUarray*>pPlaneArray, <ccuda.CUarray>hArray, planeIdx)
    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaArrayGetSparseProperties(cudaArraySparseProperties* sparseProperties, cudaArray_t array) nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUDA_ARRAY_SPARSE_PROPERTIES _driver_sparseProperties
    if not sparseProperties:
        _setLastError(cudaErrorInvalidValue)
        return cudaError.cudaErrorInvalidValue
    memset(sparseProperties, 0, sizeof(cudaArraySparseProperties))

    cdef ccuda.CUresult err
    err = ccuda._cuArrayGetSparseProperties(&_driver_sparseProperties, <ccuda.CUarray>array)
    if err == ccuda.cudaError_enum.CUDA_SUCCESS:
        sparseProperties[0].miptailFirstLevel = _driver_sparseProperties.miptailFirstLevel
        sparseProperties[0].miptailSize       = _driver_sparseProperties.miptailSize
        sparseProperties[0].flags             = _driver_sparseProperties.flags
        sparseProperties[0].tileExtent.width  = _driver_sparseProperties.tileExtent.width
        sparseProperties[0].tileExtent.height = _driver_sparseProperties.tileExtent.height
        sparseProperties[0].tileExtent.depth  = _driver_sparseProperties.tileExtent.depth

    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaMipmappedArrayGetSparseProperties(cudaArraySparseProperties* sparseProperties, cudaMipmappedArray_t mipmap) nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUDA_ARRAY_SPARSE_PROPERTIES _driver_sparseProperties
    if not sparseProperties:
        _setLastError(cudaErrorInvalidValue)
        return cudaError.cudaErrorInvalidValue
    memset(sparseProperties, 0, sizeof(cudaArraySparseProperties))

    cdef ccuda.CUresult err
    err = ccuda._cuMipmappedArrayGetSparseProperties(&_driver_sparseProperties, <ccuda.CUmipmappedArray>mipmap)
    if err == ccuda.cudaError_enum.CUDA_SUCCESS:
        sparseProperties[0].miptailFirstLevel = _driver_sparseProperties.miptailFirstLevel
        sparseProperties[0].miptailSize       = _driver_sparseProperties.miptailSize
        sparseProperties[0].flags             = _driver_sparseProperties.flags
        sparseProperties[0].tileExtent.width  = _driver_sparseProperties.tileExtent.width
        sparseProperties[0].tileExtent.height = _driver_sparseProperties.tileExtent.height
        sparseProperties[0].tileExtent.depth  = _driver_sparseProperties.tileExtent.depth

    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind) nogil except ?cudaErrorCallRequiresNewerDriver:
    return _cudaMemcpy(dst, src, count, kind)

cdef cudaError_t cudaMemcpyPeer(void* dst, int dstDevice, const void* src, int srcDevice, size_t count) nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUresult err
    err = ccuda._cuMemcpyPeer(<ccuda.CUdeviceptr>dst, m_global._driverContext[dstDevice], <ccuda.CUdeviceptr>src, m_global._driverContext[srcDevice], count)
    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaMemcpy2D(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind) nogil except ?cudaErrorCallRequiresNewerDriver:
    return _cudaMemcpy2D(dst, dpitch, src, spitch, width, height, kind)

cdef cudaError_t cudaMemcpy2DToArray(cudaArray_t dst, size_t wOffset, size_t hOffset, const void* src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind) nogil except ?cudaErrorCallRequiresNewerDriver:
    return _cudaMemcpy2DToArray(dst, wOffset, hOffset, src, spitch, width, height, kind)

cdef cudaError_t cudaMemcpy2DFromArray(void* dst, size_t dpitch, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t width, size_t height, cudaMemcpyKind kind) nogil except ?cudaErrorCallRequiresNewerDriver:
    return _cudaMemcpy2DFromArray(dst, dpitch, src, wOffset, hOffset, width, height, kind)

cdef cudaError_t cudaMemcpy2DArrayToArray(cudaArray_t dst, size_t wOffsetDst, size_t hOffsetDst, cudaArray_const_t src, size_t wOffsetSrc, size_t hOffsetSrc, size_t width, size_t height, cudaMemcpyKind kind) nogil except ?cudaErrorCallRequiresNewerDriver:
    return _cudaMemcpy2DArrayToArray(dst, wOffsetDst, hOffsetDst, src, wOffsetSrc, hOffsetSrc, width, height, kind)

cdef cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t count, cudaMemcpyKind kind, cudaStream_t stream) nogil except ?cudaErrorCallRequiresNewerDriver:
    return _cudaMemcpyAsync(dst, src, count, kind, stream)

cdef cudaError_t cudaMemcpyPeerAsync(void* dst, int dstDevice, const void* src, int srcDevice, size_t count, cudaStream_t stream) nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUresult err
    err = ccuda._cuMemcpyPeerAsync(<ccuda.CUdeviceptr>dst, m_global._driverContext[dstDevice], <ccuda.CUdeviceptr>src, m_global._driverContext[srcDevice], count, <ccuda.CUstream>stream)
    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaMemcpy2DAsync(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream) nogil except ?cudaErrorCallRequiresNewerDriver:
    return _cudaMemcpy2DAsync(dst, dpitch, src, spitch, width, height, kind, stream)

cdef cudaError_t cudaMemcpy2DToArrayAsync(cudaArray_t dst, size_t wOffset, size_t hOffset, const void* src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream) nogil except ?cudaErrorCallRequiresNewerDriver:
    return _cudaMemcpy2DToArrayAsync(dst, wOffset, hOffset, src, spitch, width, height, kind, stream)

cdef cudaError_t cudaMemcpy2DFromArrayAsync(void* dst, size_t dpitch, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream) nogil except ?cudaErrorCallRequiresNewerDriver:
    return _cudaMemcpy2DFromArrayAsync(dst, dpitch, src, wOffset, hOffset, width, height, kind, stream)

cdef cudaError_t cudaMemset(void* devPtr, int value, size_t count) nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUresult err
    err = ccuda._cuMemsetD8_v2(<ccuda.CUdeviceptr>devPtr, <unsigned char>value, count)
    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaMemset2D(void* devPtr, size_t pitch, int value, size_t width, size_t height) nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUresult err
    err = ccuda._cuMemsetD2D8_v2(<ccuda.CUdeviceptr>devPtr, pitch, <unsigned char>value, width, height)
    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaMemset3D(cudaPitchedPtr pitchedDevPtr, int value, cudaExtent extent) nogil except ?cudaErrorCallRequiresNewerDriver:
    return _cudaMemset3D(pitchedDevPtr, value, extent)

cdef cudaError_t cudaMemsetAsync(void* devPtr, int value, size_t count, cudaStream_t stream) nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUresult err
    err = ccuda._cuMemsetD8Async(<ccuda.CUdeviceptr>devPtr, <unsigned char>value, count, <ccuda.CUstream>stream)
    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaMemset2DAsync(void* devPtr, size_t pitch, int value, size_t width, size_t height, cudaStream_t stream) nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUresult err
    err = ccuda._cuMemsetD2D8Async(<ccuda.CUdeviceptr>devPtr, pitch, <unsigned char>value, width, height, <ccuda.CUstream>stream)
    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaMemset3DAsync(cudaPitchedPtr pitchedDevPtr, int value, cudaExtent extent, cudaStream_t stream) nogil except ?cudaErrorCallRequiresNewerDriver:
    return _cudaMemset3DAsync(pitchedDevPtr, value, extent, stream)

cdef cudaError_t cudaMemPrefetchAsync(const void* devPtr, size_t count, int dstDevice, cudaStream_t stream) nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUresult err
    err = ccuda._cuMemPrefetchAsync(<ccuda.CUdeviceptr>devPtr, count, <ccuda.CUdevice>dstDevice, <ccuda.CUstream>stream)
    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaMemAdvise(const void* devPtr, size_t count, cudaMemoryAdvise advice, int device) nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUresult err
    err = ccuda._cuMemAdvise(<ccuda.CUdeviceptr>devPtr, count, <ccuda.CUmem_advise>advice, <ccuda.CUdevice>device)
    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaMemRangeGetAttribute(void* data, size_t dataSize, cudaMemRangeAttribute attribute, const void* devPtr, size_t count) nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUresult err
    err = ccuda._cuMemRangeGetAttribute(data, dataSize, <ccuda.CUmem_range_attribute>attribute, <ccuda.CUdeviceptr>devPtr, count)
    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaMemRangeGetAttributes(void** data, size_t* dataSizes, cudaMemRangeAttribute* attributes, size_t numAttributes, const void* devPtr, size_t count) nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUresult err
    err = ccuda._cuMemRangeGetAttributes(data, dataSizes, <ccuda.CUmem_range_attribute*>attributes, numAttributes, <ccuda.CUdeviceptr>devPtr, count)
    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaMemcpyToArray(cudaArray_t dst, size_t wOffset, size_t hOffset, const void* src, size_t count, cudaMemcpyKind kind) nogil except ?cudaErrorCallRequiresNewerDriver:
    return _cudaMemcpyToArray(dst, wOffset, hOffset, src, count, kind)

cdef cudaError_t cudaMemcpyFromArray(void* dst, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t count, cudaMemcpyKind kind) nogil except ?cudaErrorCallRequiresNewerDriver:
    return _cudaMemcpyFromArray(dst, src, wOffset, hOffset, count, kind)

cdef cudaError_t cudaMemcpyArrayToArray(cudaArray_t dst, size_t wOffsetDst, size_t hOffsetDst, cudaArray_const_t src, size_t wOffsetSrc, size_t hOffsetSrc, size_t count, cudaMemcpyKind kind) nogil except ?cudaErrorCallRequiresNewerDriver:
    return _cudaMemcpyArrayToArray(dst, wOffsetDst, hOffsetDst, src, wOffsetSrc, hOffsetSrc, count, kind)

cdef cudaError_t cudaMemcpyToArrayAsync(cudaArray_t dst, size_t wOffset, size_t hOffset, const void* src, size_t count, cudaMemcpyKind kind, cudaStream_t stream) nogil except ?cudaErrorCallRequiresNewerDriver:
    return _cudaMemcpyToArrayAsync(dst, wOffset, hOffset, src, count, kind, stream)

cdef cudaError_t cudaMemcpyFromArrayAsync(void* dst, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t count, cudaMemcpyKind kind, cudaStream_t stream) nogil except ?cudaErrorCallRequiresNewerDriver:
    return _cudaMemcpyFromArrayAsync(dst, src, wOffset, hOffset, count, kind, stream)

cdef cudaError_t cudaMallocAsync(void** devPtr, size_t size, cudaStream_t hStream) nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUresult err
    err = ccuda._cuMemAllocAsync(<ccuda.CUdeviceptr*>devPtr, size, <ccuda.CUstream>hStream)
    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaFreeAsync(void* devPtr, cudaStream_t hStream) nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUresult err
    err = ccuda._cuMemFreeAsync(<ccuda.CUdeviceptr>devPtr, <ccuda.CUstream>hStream)
    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaMemPoolTrimTo(cudaMemPool_t memPool, size_t minBytesToKeep) nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUresult err
    err = ccuda._cuMemPoolTrimTo(<ccuda.CUmemoryPool>memPool, minBytesToKeep)
    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaMemPoolSetAttribute(cudaMemPool_t memPool, cudaMemPoolAttr attr, void* value) nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUresult err
    err = ccuda._cuMemPoolSetAttribute(<ccuda.CUmemoryPool>memPool, <ccuda.CUmemPool_attribute>attr, value)
    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaMemPoolGetAttribute(cudaMemPool_t memPool, cudaMemPoolAttr attr, void* value) nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUresult err
    err = ccuda._cuMemPoolGetAttribute(<ccuda.CUmemoryPool>memPool, <ccuda.CUmemPool_attribute>attr, value)
    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaMemPoolSetAccess(cudaMemPool_t memPool, const cudaMemAccessDesc* descList, size_t count) nogil except ?cudaErrorCallRequiresNewerDriver:
    return _cudaMemPoolSetAccess(memPool, descList, count)

cdef cudaError_t cudaMemPoolGetAccess(cudaMemAccessFlags* flags, cudaMemPool_t memPool, cudaMemLocation* location) nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUresult err
    err = ccuda._cuMemPoolGetAccess(<ccuda.CUmemAccess_flags*>flags, <ccuda.CUmemoryPool>memPool, <ccuda.CUmemLocation*>location)
    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaMemPoolCreate(cudaMemPool_t* memPool, const cudaMemPoolProps* poolProps) nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUresult err
    err = ccuda._cuMemPoolCreate(<ccuda.CUmemoryPool*>memPool, <ccuda.CUmemPoolProps*>poolProps)
    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaMemPoolDestroy(cudaMemPool_t memPool) nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUresult err
    err = ccuda._cuMemPoolDestroy(<ccuda.CUmemoryPool>memPool)
    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaMallocFromPoolAsync(void** ptr, size_t size, cudaMemPool_t memPool, cudaStream_t stream) nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUresult err
    err = ccuda._cuMemAllocFromPoolAsync(<ccuda.CUdeviceptr*>ptr, size, <ccuda.CUmemoryPool>memPool, <ccuda.CUstream>stream)
    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaMemPoolExportToShareableHandle(void* shareableHandle, cudaMemPool_t memPool, cudaMemAllocationHandleType handleType, unsigned int flags) nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUresult err
    err = ccuda._cuMemPoolExportToShareableHandle(shareableHandle, <ccuda.CUmemoryPool>memPool, <ccuda.CUmemAllocationHandleType>handleType, <unsigned long long>flags)
    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaMemPoolImportFromShareableHandle(cudaMemPool_t* memPool, void* shareableHandle, cudaMemAllocationHandleType handleType, unsigned int flags) nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUresult err
    err = ccuda._cuMemPoolImportFromShareableHandle(<ccuda.CUmemoryPool*>memPool, shareableHandle, <ccuda.CUmemAllocationHandleType>handleType, <unsigned long long>flags)
    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaMemPoolExportPointer(cudaMemPoolPtrExportData* exportData, void* ptr) nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUresult err
    err = ccuda._cuMemPoolExportPointer(<ccuda.CUmemPoolPtrExportData*>exportData, <ccuda.CUdeviceptr>ptr)
    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaMemPoolImportPointer(void** ptr, cudaMemPool_t memPool, cudaMemPoolPtrExportData* exportData) nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUresult err
    err = ccuda._cuMemPoolImportPointer(<ccuda.CUdeviceptr*>ptr, <ccuda.CUmemoryPool>memPool, <ccuda.CUmemPoolPtrExportData*>exportData)
    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaPointerGetAttributes(cudaPointerAttributes* attributes, const void* ptr) nogil except ?cudaErrorCallRequiresNewerDriver:
    return _cudaPointerGetAttributes(attributes, ptr)

cdef cudaError_t cudaDeviceCanAccessPeer(int* canAccessPeer, int device, int peerDevice) nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUresult err
    err = ccuda._cuDeviceCanAccessPeer(canAccessPeer, <ccuda.CUdevice>device, <ccuda.CUdevice>peerDevice)
    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaDeviceEnablePeerAccess(int peerDevice, unsigned int flags) nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUresult err
    err = ccuda._cuCtxEnablePeerAccess(m_global._driverContext[peerDevice], flags)
    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaDeviceDisablePeerAccess(int peerDevice) nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUresult err
    err = ccuda._cuCtxDisablePeerAccess(m_global._driverContext[peerDevice])
    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaGraphicsUnregisterResource(cudaGraphicsResource_t resource) nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUresult err
    err = ccuda._cuGraphicsUnregisterResource(<ccuda.CUgraphicsResource>resource)
    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaGraphicsResourceSetMapFlags(cudaGraphicsResource_t resource, unsigned int flags) nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUresult err
    err = ccuda._cuGraphicsResourceSetMapFlags_v2(<ccuda.CUgraphicsResource>resource, flags)
    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaGraphicsMapResources(int count, cudaGraphicsResource_t* resources, cudaStream_t stream) nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUresult err
    err = ccuda._cuGraphicsMapResources(<unsigned int>count, <ccuda.CUgraphicsResource*>resources, <ccuda.CUstream>stream)
    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaGraphicsUnmapResources(int count, cudaGraphicsResource_t* resources, cudaStream_t stream) nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUresult err
    err = ccuda._cuGraphicsUnmapResources(<unsigned int>count, <ccuda.CUgraphicsResource*>resources, <ccuda.CUstream>stream)
    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaGraphicsResourceGetMappedPointer(void** devPtr, size_t* size, cudaGraphicsResource_t resource) nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUresult err
    err = ccuda._cuGraphicsResourceGetMappedPointer_v2(<ccuda.CUdeviceptr*>devPtr, size, <ccuda.CUgraphicsResource>resource)
    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaGraphicsSubResourceGetMappedArray(cudaArray_t* array, cudaGraphicsResource_t resource, unsigned int arrayIndex, unsigned int mipLevel) nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUresult err
    err = ccuda._cuGraphicsSubResourceGetMappedArray(<ccuda.CUarray*>array, <ccuda.CUgraphicsResource>resource, arrayIndex, mipLevel)
    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaGraphicsResourceGetMappedMipmappedArray(cudaMipmappedArray_t* mipmappedArray, cudaGraphicsResource_t resource) nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUresult err
    err = ccuda._cuGraphicsResourceGetMappedMipmappedArray(<ccuda.CUmipmappedArray*>mipmappedArray, <ccuda.CUgraphicsResource>resource)
    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaGetChannelDesc(cudaChannelFormatDesc* desc, cudaArray_const_t array) nogil except ?cudaErrorCallRequiresNewerDriver:
    return _cudaGetChannelDesc(desc, array)

cdef cudaChannelFormatDesc cudaCreateChannelDesc(int x, int y, int z, int w, cudaChannelFormatKind f) nogil:
    return _cudaCreateChannelDesc(x, y, z, w, f)

cdef cudaError_t cudaCreateTextureObject(cudaTextureObject_t* pTexObject, const cudaResourceDesc* pResDesc, const cudaTextureDesc* pTexDesc, const cudaResourceViewDesc* pResViewDesc) nogil except ?cudaErrorCallRequiresNewerDriver:
    return _cudaCreateTextureObject(pTexObject, pResDesc, pTexDesc, pResViewDesc)

cdef cudaError_t cudaDestroyTextureObject(cudaTextureObject_t texObject) nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUresult err
    err = ccuda._cuTexObjectDestroy(<ccuda.CUtexObject>texObject)
    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaGetTextureObjectResourceDesc(cudaResourceDesc* pResDesc, cudaTextureObject_t texObject) nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUDA_RESOURCE_DESC _driver_pResDesc
    cdef int numChannels
    cdef ccuda.CUarray_format format
    memset(&_driver_pResDesc, 0, sizeof(_driver_pResDesc))
    if pResDesc[0].resType == cudaResourceType.cudaResourceTypeArray:
        _driver_pResDesc.resType          = ccuda.CUresourcetype_enum.CU_RESOURCE_TYPE_ARRAY
        _driver_pResDesc.res.array.hArray = <ccuda.CUarray>pResDesc[0].res.array.array
    elif pResDesc[0].resType == cudaResourceType.cudaResourceTypeMipmappedArray:
        _driver_pResDesc.resType                    = ccuda.CUresourcetype_enum.CU_RESOURCE_TYPE_MIPMAPPED_ARRAY
        _driver_pResDesc.res.mipmap.hMipmappedArray = <ccuda.CUmipmappedArray>pResDesc[0].res.mipmap.mipmap
    elif pResDesc[0].resType == cudaResourceType.cudaResourceTypeLinear:
        _driver_pResDesc.resType                = ccuda.CUresourcetype_enum.CU_RESOURCE_TYPE_LINEAR
        _driver_pResDesc.res.linear.devPtr      = <ccuda.CUdeviceptr>pResDesc[0].res.linear.devPtr
        _driver_pResDesc.res.linear.sizeInBytes = pResDesc[0].res.linear.sizeInBytes
        err_rt = getDescInfo(&pResDesc[0].res.linear.desc, &numChannels, &format)
        if err_rt != cudaError.cudaSuccess:
            _setLastError(err_rt)
            return err_rt
        _driver_pResDesc.res.linear.format      = format
        _driver_pResDesc.res.linear.numChannels = numChannels
    elif pResDesc[0].resType == cudaResourceType.cudaResourceTypePitch2D:
        _driver_pResDesc.resType                  = ccuda.CUresourcetype_enum.CU_RESOURCE_TYPE_PITCH2D
        _driver_pResDesc.res.pitch2D.devPtr       = <ccuda.CUdeviceptr>pResDesc[0].res.pitch2D.devPtr
        _driver_pResDesc.res.pitch2D.pitchInBytes = pResDesc[0].res.pitch2D.pitchInBytes
        _driver_pResDesc.res.pitch2D.width        = pResDesc[0].res.pitch2D.width
        _driver_pResDesc.res.pitch2D.height       = pResDesc[0].res.pitch2D.height
        err_rt = getDescInfo(&pResDesc[0].res.linear.desc, &numChannels, &format)
        if err_rt != cudaError.cudaSuccess:
            _setLastError(err_rt)
            return err_rt
        _driver_pResDesc.res.pitch2D.format       = format
        _driver_pResDesc.res.pitch2D.numChannels  = numChannels
    else:
        _setLastError(cudaError.cudaErrorInvalidValue)
        return cudaError.cudaErrorInvalidValue
    _driver_pResDesc.flags = 0

    cdef ccuda.CUresult err
    err = ccuda._cuTexObjectGetResourceDesc(&_driver_pResDesc, <ccuda.CUtexObject>texObject)
    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaGetTextureObjectTextureDesc(cudaTextureDesc* pTexDesc, cudaTextureObject_t texObject) nogil except ?cudaErrorCallRequiresNewerDriver:
    return _cudaGetTextureObjectTextureDesc(pTexDesc, texObject)

cdef cudaError_t cudaGetTextureObjectResourceViewDesc(cudaResourceViewDesc* pResViewDesc, cudaTextureObject_t texObject) nogil except ?cudaErrorCallRequiresNewerDriver:
    return _cudaGetTextureObjectResourceViewDesc(pResViewDesc, texObject)

cdef cudaError_t cudaCreateSurfaceObject(cudaSurfaceObject_t* pSurfObject, const cudaResourceDesc* pResDesc) nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUDA_RESOURCE_DESC _driver_pResDesc
    cdef int numChannels
    cdef ccuda.CUarray_format format
    memset(&_driver_pResDesc, 0, sizeof(_driver_pResDesc))
    if pResDesc[0].resType == cudaResourceType.cudaResourceTypeArray:
        _driver_pResDesc.resType          = ccuda.CUresourcetype_enum.CU_RESOURCE_TYPE_ARRAY
        _driver_pResDesc.res.array.hArray = <ccuda.CUarray>pResDesc[0].res.array.array
    elif pResDesc[0].resType == cudaResourceType.cudaResourceTypeMipmappedArray:
        _driver_pResDesc.resType                    = ccuda.CUresourcetype_enum.CU_RESOURCE_TYPE_MIPMAPPED_ARRAY
        _driver_pResDesc.res.mipmap.hMipmappedArray = <ccuda.CUmipmappedArray>pResDesc[0].res.mipmap.mipmap
    elif pResDesc[0].resType == cudaResourceType.cudaResourceTypeLinear:
        _driver_pResDesc.resType                = ccuda.CUresourcetype_enum.CU_RESOURCE_TYPE_LINEAR
        _driver_pResDesc.res.linear.devPtr      = <ccuda.CUdeviceptr>pResDesc[0].res.linear.devPtr
        _driver_pResDesc.res.linear.sizeInBytes = pResDesc[0].res.linear.sizeInBytes
        err_rt = getDescInfo(&pResDesc[0].res.linear.desc, &numChannels, &format)
        if err_rt != cudaError.cudaSuccess:
            _setLastError(err_rt)
            return err_rt
        _driver_pResDesc.res.linear.format      = format
        _driver_pResDesc.res.linear.numChannels = numChannels
    elif pResDesc[0].resType == cudaResourceType.cudaResourceTypePitch2D:
        _driver_pResDesc.resType                  = ccuda.CUresourcetype_enum.CU_RESOURCE_TYPE_PITCH2D
        _driver_pResDesc.res.pitch2D.devPtr       = <ccuda.CUdeviceptr>pResDesc[0].res.pitch2D.devPtr
        _driver_pResDesc.res.pitch2D.pitchInBytes = pResDesc[0].res.pitch2D.pitchInBytes
        _driver_pResDesc.res.pitch2D.width        = pResDesc[0].res.pitch2D.width
        _driver_pResDesc.res.pitch2D.height       = pResDesc[0].res.pitch2D.height
        err_rt = getDescInfo(&pResDesc[0].res.linear.desc, &numChannels, &format)
        if err_rt != cudaError.cudaSuccess:
            _setLastError(err_rt)
            return err_rt
        _driver_pResDesc.res.pitch2D.format       = format
        _driver_pResDesc.res.pitch2D.numChannels  = numChannels
    else:
        _setLastError(cudaError.cudaErrorInvalidValue)
        return cudaError.cudaErrorInvalidValue
    _driver_pResDesc.flags = 0

    cdef ccuda.CUresult err
    err = ccuda._cuSurfObjectCreate(<ccuda.CUsurfObject*>pSurfObject, &_driver_pResDesc)
    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaDestroySurfaceObject(cudaSurfaceObject_t surfObject) nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUresult err
    err = ccuda._cuSurfObjectDestroy(<ccuda.CUsurfObject>surfObject)
    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaGetSurfaceObjectResourceDesc(cudaResourceDesc* pResDesc, cudaSurfaceObject_t surfObject) nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUDA_RESOURCE_DESC _driver_pResDesc

    cdef ccuda.CUresult err
    err = ccuda._cuSurfObjectGetResourceDesc(&_driver_pResDesc, <ccuda.CUsurfObject>surfObject)
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

    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaDriverGetVersion(int* driverVersion) nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUresult err
    err = ccuda._cuDriverGetVersion(driverVersion)
    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaRuntimeGetVersion(int* runtimeVersion) nogil except ?cudaErrorCallRequiresNewerDriver:
    return _cudaRuntimeGetVersion(runtimeVersion)

cdef cudaError_t cudaGraphCreate(cudaGraph_t* pGraph, unsigned int flags) nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUresult err
    err = ccuda._cuGraphCreate(<ccuda.CUgraph*>pGraph, flags)
    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaGraphAddKernelNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, const cudaKernelNodeParams* pNodeParams) nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUDA_KERNEL_NODE_PARAMS _driver_pNodeParams
    _driver_pNodeParams.func = <ccuda.CUfunction>pNodeParams[0].func
    _driver_pNodeParams.gridDimX = pNodeParams[0].gridDim.x
    _driver_pNodeParams.gridDimY = pNodeParams[0].gridDim.y
    _driver_pNodeParams.gridDimZ = pNodeParams[0].gridDim.z
    _driver_pNodeParams.blockDimX = pNodeParams[0].blockDim.x
    _driver_pNodeParams.blockDimY = pNodeParams[0].blockDim.y
    _driver_pNodeParams.blockDimZ = pNodeParams[0].blockDim.z
    _driver_pNodeParams.sharedMemBytes = pNodeParams[0].sharedMemBytes
    _driver_pNodeParams.kernelParams = pNodeParams[0].kernelParams
    _driver_pNodeParams.extra = pNodeParams[0].extra

    cdef ccuda.CUresult err
    err = ccuda._cuGraphAddKernelNode(<ccuda.CUgraphNode*>pGraphNode, <ccuda.CUgraph>graph, <ccuda.CUgraphNode*>pDependencies, numDependencies, &_driver_pNodeParams)
    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaGraphKernelNodeGetParams(cudaGraphNode_t node, cudaKernelNodeParams* pNodeParams) nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUDA_KERNEL_NODE_PARAMS _driver_pNodeParams

    cdef ccuda.CUresult err
    err = ccuda._cuGraphKernelNodeGetParams(<ccuda.CUgraphNode>node, &_driver_pNodeParams)
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

    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaGraphKernelNodeSetParams(cudaGraphNode_t node, const cudaKernelNodeParams* pNodeParams) nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUDA_KERNEL_NODE_PARAMS _driver_pNodeParams
    _driver_pNodeParams.func = <ccuda.CUfunction>pNodeParams[0].func
    _driver_pNodeParams.gridDimX = pNodeParams[0].gridDim.x
    _driver_pNodeParams.gridDimY = pNodeParams[0].gridDim.y
    _driver_pNodeParams.gridDimZ = pNodeParams[0].gridDim.z
    _driver_pNodeParams.blockDimX = pNodeParams[0].blockDim.x
    _driver_pNodeParams.blockDimY = pNodeParams[0].blockDim.y
    _driver_pNodeParams.blockDimZ = pNodeParams[0].blockDim.z
    _driver_pNodeParams.sharedMemBytes = pNodeParams[0].sharedMemBytes
    _driver_pNodeParams.kernelParams = pNodeParams[0].kernelParams
    _driver_pNodeParams.extra = pNodeParams[0].extra

    cdef ccuda.CUresult err
    err = ccuda._cuGraphKernelNodeSetParams(<ccuda.CUgraphNode>node, &_driver_pNodeParams)
    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaGraphKernelNodeCopyAttributes(cudaGraphNode_t hSrc, cudaGraphNode_t hDst) nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUresult err
    err = ccuda._cuGraphKernelNodeCopyAttributes(<ccuda.CUgraphNode>hSrc, <ccuda.CUgraphNode>hDst)
    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaGraphKernelNodeGetAttribute(cudaGraphNode_t hNode, cudaKernelNodeAttrID attr, cudaKernelNodeAttrValue* value_out) nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUresult err
    err = ccuda._cuGraphKernelNodeGetAttribute(<ccuda.CUgraphNode>hNode, <ccuda.CUkernelNodeAttrID>attr, <ccuda.CUkernelNodeAttrValue*>value_out)
    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaGraphKernelNodeSetAttribute(cudaGraphNode_t hNode, cudaKernelNodeAttrID attr, const cudaKernelNodeAttrValue* value) nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUresult err
    err = ccuda._cuGraphKernelNodeSetAttribute(<ccuda.CUgraphNode>hNode, <ccuda.CUkernelNodeAttrID>attr, <ccuda.CUkernelNodeAttrValue*>value)
    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaGraphAddMemcpyNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, const cudaMemcpy3DParms* pCopyParams) nogil except ?cudaErrorCallRequiresNewerDriver:
    return _cudaGraphAddMemcpyNode(pGraphNode, graph, pDependencies, numDependencies, pCopyParams)

cdef cudaError_t cudaGraphAddMemcpyNode1D(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, void* dst, const void* src, size_t count, cudaMemcpyKind kind) nogil except ?cudaErrorCallRequiresNewerDriver:
    return _cudaGraphAddMemcpyNode1D(pGraphNode, graph, pDependencies, numDependencies, dst, src, count, kind)

cdef cudaError_t cudaGraphMemcpyNodeGetParams(cudaGraphNode_t node, cudaMemcpy3DParms* pNodeParams) nogil except ?cudaErrorCallRequiresNewerDriver:
    return _cudaGraphMemcpyNodeGetParams(node, pNodeParams)

cdef cudaError_t cudaGraphMemcpyNodeSetParams(cudaGraphNode_t node, const cudaMemcpy3DParms* pNodeParams) nogil except ?cudaErrorCallRequiresNewerDriver:
    return _cudaGraphMemcpyNodeSetParams(node, pNodeParams)

cdef cudaError_t cudaGraphMemcpyNodeSetParams1D(cudaGraphNode_t node, void* dst, const void* src, size_t count, cudaMemcpyKind kind) nogil except ?cudaErrorCallRequiresNewerDriver:
    return _cudaGraphMemcpyNodeSetParams1D(node, dst, src, count, kind)

cdef cudaError_t cudaGraphAddMemsetNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, const cudaMemsetParams* pMemsetParams) nogil except ?cudaErrorCallRequiresNewerDriver:
    return _cudaGraphAddMemsetNode(pGraphNode, graph, pDependencies, numDependencies, pMemsetParams)

cdef cudaError_t cudaGraphMemsetNodeGetParams(cudaGraphNode_t node, cudaMemsetParams* pNodeParams) nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUresult err
    err = ccuda._cuGraphMemsetNodeGetParams(<ccuda.CUgraphNode>node, <ccuda.CUDA_MEMSET_NODE_PARAMS*>pNodeParams)
    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaGraphMemsetNodeSetParams(cudaGraphNode_t node, const cudaMemsetParams* pNodeParams) nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUresult err
    err = ccuda._cuGraphMemsetNodeSetParams(<ccuda.CUgraphNode>node, <ccuda.CUDA_MEMSET_NODE_PARAMS*>pNodeParams)
    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaGraphAddHostNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, const cudaHostNodeParams* pNodeParams) nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUresult err
    err = ccuda._cuGraphAddHostNode(<ccuda.CUgraphNode*>pGraphNode, <ccuda.CUgraph>graph, <ccuda.CUgraphNode*>pDependencies, numDependencies, <ccuda.CUDA_HOST_NODE_PARAMS*>pNodeParams)
    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaGraphHostNodeGetParams(cudaGraphNode_t node, cudaHostNodeParams* pNodeParams) nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUresult err
    err = ccuda._cuGraphHostNodeGetParams(<ccuda.CUgraphNode>node, <ccuda.CUDA_HOST_NODE_PARAMS*>pNodeParams)
    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaGraphHostNodeSetParams(cudaGraphNode_t node, const cudaHostNodeParams* pNodeParams) nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUresult err
    err = ccuda._cuGraphHostNodeSetParams(<ccuda.CUgraphNode>node, <ccuda.CUDA_HOST_NODE_PARAMS*>pNodeParams)
    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaGraphAddChildGraphNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, cudaGraph_t childGraph) nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUresult err
    err = ccuda._cuGraphAddChildGraphNode(<ccuda.CUgraphNode*>pGraphNode, <ccuda.CUgraph>graph, <ccuda.CUgraphNode*>pDependencies, numDependencies, <ccuda.CUgraph>childGraph)
    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaGraphChildGraphNodeGetGraph(cudaGraphNode_t node, cudaGraph_t* pGraph) nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUresult err
    err = ccuda._cuGraphChildGraphNodeGetGraph(<ccuda.CUgraphNode>node, <ccuda.CUgraph*>pGraph)
    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaGraphAddEmptyNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies) nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUresult err
    err = ccuda._cuGraphAddEmptyNode(<ccuda.CUgraphNode*>pGraphNode, <ccuda.CUgraph>graph, <ccuda.CUgraphNode*>pDependencies, numDependencies)
    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaGraphAddEventRecordNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, cudaEvent_t event) nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUresult err
    err = ccuda._cuGraphAddEventRecordNode(<ccuda.CUgraphNode*>pGraphNode, <ccuda.CUgraph>graph, <ccuda.CUgraphNode*>pDependencies, numDependencies, <ccuda.CUevent>event)
    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaGraphEventRecordNodeGetEvent(cudaGraphNode_t node, cudaEvent_t* event_out) nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUresult err
    err = ccuda._cuGraphEventRecordNodeGetEvent(<ccuda.CUgraphNode>node, <ccuda.CUevent*>event_out)
    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaGraphEventRecordNodeSetEvent(cudaGraphNode_t node, cudaEvent_t event) nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUresult err
    err = ccuda._cuGraphEventRecordNodeSetEvent(<ccuda.CUgraphNode>node, <ccuda.CUevent>event)
    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaGraphAddEventWaitNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, cudaEvent_t event) nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUresult err
    err = ccuda._cuGraphAddEventWaitNode(<ccuda.CUgraphNode*>pGraphNode, <ccuda.CUgraph>graph, <ccuda.CUgraphNode*>pDependencies, numDependencies, <ccuda.CUevent>event)
    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaGraphEventWaitNodeGetEvent(cudaGraphNode_t node, cudaEvent_t* event_out) nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUresult err
    err = ccuda._cuGraphEventWaitNodeGetEvent(<ccuda.CUgraphNode>node, <ccuda.CUevent*>event_out)
    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaGraphEventWaitNodeSetEvent(cudaGraphNode_t node, cudaEvent_t event) nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUresult err
    err = ccuda._cuGraphEventWaitNodeSetEvent(<ccuda.CUgraphNode>node, <ccuda.CUevent>event)
    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaGraphAddExternalSemaphoresSignalNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, const cudaExternalSemaphoreSignalNodeParams* nodeParams) nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUresult err
    err = ccuda._cuGraphAddExternalSemaphoresSignalNode(<ccuda.CUgraphNode*>pGraphNode, <ccuda.CUgraph>graph, <ccuda.CUgraphNode*>pDependencies, numDependencies, <ccuda.CUDA_EXT_SEM_SIGNAL_NODE_PARAMS*>nodeParams)
    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaGraphExternalSemaphoresSignalNodeGetParams(cudaGraphNode_t hNode, cudaExternalSemaphoreSignalNodeParams* params_out) nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUresult err
    err = ccuda._cuGraphExternalSemaphoresSignalNodeGetParams(<ccuda.CUgraphNode>hNode, <ccuda.CUDA_EXT_SEM_SIGNAL_NODE_PARAMS*>params_out)
    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaGraphExternalSemaphoresSignalNodeSetParams(cudaGraphNode_t hNode, const cudaExternalSemaphoreSignalNodeParams* nodeParams) nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUresult err
    err = ccuda._cuGraphExternalSemaphoresSignalNodeSetParams(<ccuda.CUgraphNode>hNode, <ccuda.CUDA_EXT_SEM_SIGNAL_NODE_PARAMS*>nodeParams)
    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaGraphAddExternalSemaphoresWaitNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, const cudaExternalSemaphoreWaitNodeParams* nodeParams) nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUresult err
    err = ccuda._cuGraphAddExternalSemaphoresWaitNode(<ccuda.CUgraphNode*>pGraphNode, <ccuda.CUgraph>graph, <ccuda.CUgraphNode*>pDependencies, numDependencies, <ccuda.CUDA_EXT_SEM_WAIT_NODE_PARAMS*>nodeParams)
    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaGraphExternalSemaphoresWaitNodeGetParams(cudaGraphNode_t hNode, cudaExternalSemaphoreWaitNodeParams* params_out) nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUresult err
    err = ccuda._cuGraphExternalSemaphoresWaitNodeGetParams(<ccuda.CUgraphNode>hNode, <ccuda.CUDA_EXT_SEM_WAIT_NODE_PARAMS*>params_out)
    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaGraphExternalSemaphoresWaitNodeSetParams(cudaGraphNode_t hNode, const cudaExternalSemaphoreWaitNodeParams* nodeParams) nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUresult err
    err = ccuda._cuGraphExternalSemaphoresWaitNodeSetParams(<ccuda.CUgraphNode>hNode, <ccuda.CUDA_EXT_SEM_WAIT_NODE_PARAMS*>nodeParams)
    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaGraphAddMemAllocNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, cudaMemAllocNodeParams* nodeParams) nogil except ?cudaErrorCallRequiresNewerDriver:
    return _cudaGraphAddMemAllocNode(pGraphNode, graph, pDependencies, numDependencies, nodeParams)

cdef cudaError_t cudaGraphMemAllocNodeGetParams(cudaGraphNode_t node, cudaMemAllocNodeParams* params_out) nogil except ?cudaErrorCallRequiresNewerDriver:
    return _cudaGraphMemAllocNodeGetParams(node, params_out)

cdef cudaError_t cudaGraphAddMemFreeNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, void* dptr) nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUresult err
    err = ccuda._cuGraphAddMemFreeNode(<ccuda.CUgraphNode*>pGraphNode, <ccuda.CUgraph>graph, <ccuda.CUgraphNode*>pDependencies, numDependencies, <ccuda.CUdeviceptr>dptr)
    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaGraphMemFreeNodeGetParams(cudaGraphNode_t node, void* dptr_out) nogil except ?cudaErrorCallRequiresNewerDriver:
    return _cudaGraphMemFreeNodeGetParams(node, dptr_out)

cdef cudaError_t cudaDeviceGraphMemTrim(int device) nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUresult err
    err = ccuda._cuDeviceGraphMemTrim(<ccuda.CUdevice>device)
    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaDeviceGetGraphMemAttribute(int device, cudaGraphMemAttributeType attr, void* value) nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUresult err
    err = ccuda._cuDeviceGetGraphMemAttribute(<ccuda.CUdevice>device, <ccuda.CUgraphMem_attribute>attr, value)
    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaDeviceSetGraphMemAttribute(int device, cudaGraphMemAttributeType attr, void* value) nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUresult err
    err = ccuda._cuDeviceSetGraphMemAttribute(<ccuda.CUdevice>device, <ccuda.CUgraphMem_attribute>attr, value)
    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaGraphClone(cudaGraph_t* pGraphClone, cudaGraph_t originalGraph) nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUresult err
    err = ccuda._cuGraphClone(<ccuda.CUgraph*>pGraphClone, <ccuda.CUgraph>originalGraph)
    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaGraphNodeFindInClone(cudaGraphNode_t* pNode, cudaGraphNode_t originalNode, cudaGraph_t clonedGraph) nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUresult err
    err = ccuda._cuGraphNodeFindInClone(<ccuda.CUgraphNode*>pNode, <ccuda.CUgraphNode>originalNode, <ccuda.CUgraph>clonedGraph)
    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaGraphNodeGetType(cudaGraphNode_t node, cudaGraphNodeType* pType) nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUresult err
    err = ccuda._cuGraphNodeGetType(<ccuda.CUgraphNode>node, <ccuda.CUgraphNodeType*>pType)
    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaGraphGetNodes(cudaGraph_t graph, cudaGraphNode_t* nodes, size_t* numNodes) nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUresult err
    err = ccuda._cuGraphGetNodes(<ccuda.CUgraph>graph, <ccuda.CUgraphNode*>nodes, numNodes)
    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaGraphGetRootNodes(cudaGraph_t graph, cudaGraphNode_t* pRootNodes, size_t* pNumRootNodes) nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUresult err
    err = ccuda._cuGraphGetRootNodes(<ccuda.CUgraph>graph, <ccuda.CUgraphNode*>pRootNodes, pNumRootNodes)
    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaGraphGetEdges(cudaGraph_t graph, cudaGraphNode_t* from_, cudaGraphNode_t* to, size_t* numEdges) nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUresult err
    err = ccuda._cuGraphGetEdges(<ccuda.CUgraph>graph, <ccuda.CUgraphNode*>from_, <ccuda.CUgraphNode*>to, numEdges)
    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaGraphNodeGetDependencies(cudaGraphNode_t node, cudaGraphNode_t* pDependencies, size_t* pNumDependencies) nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUresult err
    err = ccuda._cuGraphNodeGetDependencies(<ccuda.CUgraphNode>node, <ccuda.CUgraphNode*>pDependencies, pNumDependencies)
    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaGraphNodeGetDependentNodes(cudaGraphNode_t node, cudaGraphNode_t* pDependentNodes, size_t* pNumDependentNodes) nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUresult err
    err = ccuda._cuGraphNodeGetDependentNodes(<ccuda.CUgraphNode>node, <ccuda.CUgraphNode*>pDependentNodes, pNumDependentNodes)
    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaGraphAddDependencies(cudaGraph_t graph, const cudaGraphNode_t* from_, const cudaGraphNode_t* to, size_t numDependencies) nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUresult err
    err = ccuda._cuGraphAddDependencies(<ccuda.CUgraph>graph, <ccuda.CUgraphNode*>from_, <ccuda.CUgraphNode*>to, numDependencies)
    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaGraphRemoveDependencies(cudaGraph_t graph, const cudaGraphNode_t* from_, const cudaGraphNode_t* to, size_t numDependencies) nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUresult err
    err = ccuda._cuGraphRemoveDependencies(<ccuda.CUgraph>graph, <ccuda.CUgraphNode*>from_, <ccuda.CUgraphNode*>to, numDependencies)
    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaGraphDestroyNode(cudaGraphNode_t node) nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUresult err
    err = ccuda._cuGraphDestroyNode(<ccuda.CUgraphNode>node)
    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaGraphInstantiate(cudaGraphExec_t* pGraphExec, cudaGraph_t graph, cudaGraphNode_t* pErrorNode, char* pLogBuffer, size_t bufferSize) nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUresult err
    err = ccuda._cuGraphInstantiate_v2(<ccuda.CUgraphExec*>pGraphExec, <ccuda.CUgraph>graph, <ccuda.CUgraphNode*>pErrorNode, pLogBuffer, bufferSize)
    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaGraphInstantiateWithFlags(cudaGraphExec_t* pGraphExec, cudaGraph_t graph, unsigned long long flags) nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUresult err
    err = ccuda._cuGraphInstantiateWithFlags(<ccuda.CUgraphExec*>pGraphExec, <ccuda.CUgraph>graph, flags)
    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaGraphExecKernelNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, const cudaKernelNodeParams* pNodeParams) nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUDA_KERNEL_NODE_PARAMS _driver_pNodeParams
    _driver_pNodeParams.func = <ccuda.CUfunction>pNodeParams[0].func
    _driver_pNodeParams.gridDimX = pNodeParams[0].gridDim.x
    _driver_pNodeParams.gridDimY = pNodeParams[0].gridDim.y
    _driver_pNodeParams.gridDimZ = pNodeParams[0].gridDim.z
    _driver_pNodeParams.blockDimX = pNodeParams[0].blockDim.x
    _driver_pNodeParams.blockDimY = pNodeParams[0].blockDim.y
    _driver_pNodeParams.blockDimZ = pNodeParams[0].blockDim.z
    _driver_pNodeParams.sharedMemBytes = pNodeParams[0].sharedMemBytes
    _driver_pNodeParams.kernelParams = pNodeParams[0].kernelParams
    _driver_pNodeParams.extra = pNodeParams[0].extra

    cdef ccuda.CUresult err
    err = ccuda._cuGraphExecKernelNodeSetParams(<ccuda.CUgraphExec>hGraphExec, <ccuda.CUgraphNode>node, &_driver_pNodeParams)
    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaGraphExecMemcpyNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, const cudaMemcpy3DParms* pNodeParams) nogil except ?cudaErrorCallRequiresNewerDriver:
    return _cudaGraphExecMemcpyNodeSetParams(hGraphExec, node, pNodeParams)

cdef cudaError_t cudaGraphExecMemcpyNodeSetParams1D(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, void* dst, const void* src, size_t count, cudaMemcpyKind kind) nogil except ?cudaErrorCallRequiresNewerDriver:
    return _cudaGraphExecMemcpyNodeSetParams1D(hGraphExec, node, dst, src, count, kind)

cdef cudaError_t cudaGraphExecMemsetNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, const cudaMemsetParams* pNodeParams) nogil except ?cudaErrorCallRequiresNewerDriver:
    return _cudaGraphExecMemsetNodeSetParams(hGraphExec, node, pNodeParams)

cdef cudaError_t cudaGraphExecHostNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, const cudaHostNodeParams* pNodeParams) nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUresult err
    err = ccuda._cuGraphExecHostNodeSetParams(<ccuda.CUgraphExec>hGraphExec, <ccuda.CUgraphNode>node, <ccuda.CUDA_HOST_NODE_PARAMS*>pNodeParams)
    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaGraphExecChildGraphNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, cudaGraph_t childGraph) nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUresult err
    err = ccuda._cuGraphExecChildGraphNodeSetParams(<ccuda.CUgraphExec>hGraphExec, <ccuda.CUgraphNode>node, <ccuda.CUgraph>childGraph)
    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaGraphExecEventRecordNodeSetEvent(cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, cudaEvent_t event) nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUresult err
    err = ccuda._cuGraphExecEventRecordNodeSetEvent(<ccuda.CUgraphExec>hGraphExec, <ccuda.CUgraphNode>hNode, <ccuda.CUevent>event)
    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaGraphExecEventWaitNodeSetEvent(cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, cudaEvent_t event) nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUresult err
    err = ccuda._cuGraphExecEventWaitNodeSetEvent(<ccuda.CUgraphExec>hGraphExec, <ccuda.CUgraphNode>hNode, <ccuda.CUevent>event)
    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaGraphExecExternalSemaphoresSignalNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, const cudaExternalSemaphoreSignalNodeParams* nodeParams) nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUresult err
    err = ccuda._cuGraphExecExternalSemaphoresSignalNodeSetParams(<ccuda.CUgraphExec>hGraphExec, <ccuda.CUgraphNode>hNode, <ccuda.CUDA_EXT_SEM_SIGNAL_NODE_PARAMS*>nodeParams)
    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaGraphExecExternalSemaphoresWaitNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, const cudaExternalSemaphoreWaitNodeParams* nodeParams) nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUresult err
    err = ccuda._cuGraphExecExternalSemaphoresWaitNodeSetParams(<ccuda.CUgraphExec>hGraphExec, <ccuda.CUgraphNode>hNode, <ccuda.CUDA_EXT_SEM_WAIT_NODE_PARAMS*>nodeParams)
    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaGraphExecUpdate(cudaGraphExec_t hGraphExec, cudaGraph_t hGraph, cudaGraphNode_t* hErrorNode_out, cudaGraphExecUpdateResult* updateResult_out) nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUresult err
    err = ccuda._cuGraphExecUpdate(<ccuda.CUgraphExec>hGraphExec, <ccuda.CUgraph>hGraph, <ccuda.CUgraphNode*>hErrorNode_out, <ccuda.CUgraphExecUpdateResult*>updateResult_out)
    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaGraphUpload(cudaGraphExec_t graphExec, cudaStream_t stream) nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUresult err
    err = ccuda._cuGraphUpload(<ccuda.CUgraphExec>graphExec, <ccuda.CUstream>stream)
    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaGraphLaunch(cudaGraphExec_t graphExec, cudaStream_t stream) nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUresult err
    err = ccuda._cuGraphLaunch(<ccuda.CUgraphExec>graphExec, <ccuda.CUstream>stream)
    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaGraphExecDestroy(cudaGraphExec_t graphExec) nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUresult err
    err = ccuda._cuGraphExecDestroy(<ccuda.CUgraphExec>graphExec)
    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaGraphDestroy(cudaGraph_t graph) nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUresult err
    err = ccuda._cuGraphDestroy(<ccuda.CUgraph>graph)
    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaGraphDebugDotPrint(cudaGraph_t graph, const char* path, unsigned int flags) nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUresult err
    err = ccuda._cuGraphDebugDotPrint(<ccuda.CUgraph>graph, path, flags)
    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaUserObjectCreate(cudaUserObject_t* object_out, void* ptr, cudaHostFn_t destroy, unsigned int initialRefcount, unsigned int flags) nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUresult err
    err = ccuda._cuUserObjectCreate(<ccuda.CUuserObject*>object_out, ptr, <ccuda.CUhostFn>destroy, initialRefcount, flags)
    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaUserObjectRetain(cudaUserObject_t object, unsigned int count) nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUresult err
    err = ccuda._cuUserObjectRetain(<ccuda.CUuserObject>object, count)
    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaUserObjectRelease(cudaUserObject_t object, unsigned int count) nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUresult err
    err = ccuda._cuUserObjectRelease(<ccuda.CUuserObject>object, count)
    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaGraphRetainUserObject(cudaGraph_t graph, cudaUserObject_t object, unsigned int count, unsigned int flags) nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUresult err
    err = ccuda._cuGraphRetainUserObject(<ccuda.CUgraph>graph, <ccuda.CUuserObject>object, count, flags)
    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaGraphReleaseUserObject(cudaGraph_t graph, cudaUserObject_t object, unsigned int count) nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUresult err
    err = ccuda._cuGraphReleaseUserObject(<ccuda.CUgraph>graph, <ccuda.CUuserObject>object, count)
    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaError_t cudaGetDriverEntryPoint(const char* symbol, void** funcPtr, unsigned long long flags) nogil except ?cudaErrorCallRequiresNewerDriver:
    return _cudaGetDriverEntryPoint(symbol, funcPtr, flags)

cdef cudaError_t cudaGetExportTable(const void** ppExportTable, const cudaUUID_t* pExportTableId) nogil except ?cudaErrorCallRequiresNewerDriver:
    m_global.lazyInit()
    cdef ccuda.CUresult err
    err = ccuda._cuGetExportTable(ppExportTable, <ccuda.CUuuid*>pExportTableId)
    _setLastError(<cudaError_t>(err))
    return <cudaError_t>(err)

cdef cudaPitchedPtr make_cudaPitchedPtr(void* d, size_t p, size_t xsz, size_t ysz) nogil:
    return _make_cudaPitchedPtr(d, p, xsz, ysz)

cdef cudaPos make_cudaPos(size_t x, size_t y, size_t z) nogil:
    return _make_cudaPos(x, y, z)

cdef cudaExtent make_cudaExtent(size_t w, size_t h, size_t d) nogil:
    return _make_cudaExtent(w, h, d)
