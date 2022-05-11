# Copyright 2021-2022 NVIDIA Corporation.  All rights reserved.
#
# Please refer to the NVIDIA end user license agreement (EULA) associated
# with this source code for terms and conditions that govern your use of
# this software. Any use, reproduction, disclosure, or distribution of
# this software and related documentation outside the terms of the EULA
# is strictly prohibited.
cimport cuda._cuda.ccuda as ccuda
from cuda._lib.ccudart.ccudart cimport *
from cuda._lib.ccudart.utils cimport *
from libc.stdlib cimport malloc, free, calloc
from libc.string cimport memset, memcpy, strncmp
from libcpp cimport bool

cdef cudaPythonGlobal m_global = globalGetInstance()

cdef cudaError_t cudaDeviceReset() nogil except ?cudaErrorCallRequiresNewerDriver:
    return _cudaDeviceReset()

cdef cudaError_t cudaDeviceSynchronize() nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuCtxSynchronize()
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t cudaDeviceSetLimit(cudaLimit limit, size_t value) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuCtxSetLimit(<ccuda.CUlimit>limit, value)
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t cudaDeviceGetLimit(size_t* pValue, cudaLimit limit) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuCtxGetLimit(pValue, <ccuda.CUlimit>limit)
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t cudaDeviceGetTexture1DLinearMaxWidth(size_t* maxWidthInElements, const cudaChannelFormatDesc* fmtDesc, int device) nogil except ?cudaErrorCallRequiresNewerDriver:
    return _cudaDeviceGetTexture1DLinearMaxWidth(maxWidthInElements, fmtDesc, device)

cdef cudaError_t cudaDeviceGetCacheConfig(cudaFuncCache* pCacheConfig) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuCtxGetCacheConfig(<ccuda.CUfunc_cache*>pCacheConfig)
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t cudaDeviceGetStreamPriorityRange(int* leastPriority, int* greatestPriority) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuCtxGetStreamPriorityRange(leastPriority, greatestPriority)
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t cudaDeviceSetCacheConfig(cudaFuncCache cacheConfig) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuCtxSetCacheConfig(<ccuda.CUfunc_cache>cacheConfig)
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t cudaDeviceGetSharedMemConfig(cudaSharedMemConfig* pConfig) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuCtxGetSharedMemConfig(<ccuda.CUsharedconfig*>pConfig)
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t cudaDeviceSetSharedMemConfig(cudaSharedMemConfig config) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuCtxSetSharedMemConfig(<ccuda.CUsharedconfig>config)
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t cudaDeviceGetByPCIBusId(int* device, const char* pciBusId) nogil except ?cudaErrorCallRequiresNewerDriver:
    return _cudaDeviceGetByPCIBusId(device, pciBusId)

cdef cudaError_t cudaDeviceGetPCIBusId(char* pciBusId, int length, int device) nogil except ?cudaErrorCallRequiresNewerDriver:
    return _cudaDeviceGetPCIBusId(pciBusId, length, device)

cdef cudaError_t cudaIpcGetEventHandle(cudaIpcEventHandle_t* handle, cudaEvent_t event) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuIpcGetEventHandle(<ccuda.CUipcEventHandle*>handle, <ccuda.CUevent>event)
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t cudaIpcOpenEventHandle(cudaEvent_t* event, cudaIpcEventHandle_t handle) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    cdef ccuda.CUipcEventHandle _driver_handle
    memcpy(&_driver_handle, &handle, sizeof(_driver_handle))
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuIpcOpenEventHandle(<ccuda.CUevent*>event, _driver_handle)
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t cudaIpcGetMemHandle(cudaIpcMemHandle_t* handle, void* devPtr) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuIpcGetMemHandle(<ccuda.CUipcMemHandle*>handle, <ccuda.CUdeviceptr>devPtr)
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t cudaIpcOpenMemHandle(void** devPtr, cudaIpcMemHandle_t handle, unsigned int flags) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    cdef ccuda.CUipcMemHandle _driver_handle
    memcpy(&_driver_handle, &handle, sizeof(_driver_handle))
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuIpcOpenMemHandle_v2(<ccuda.CUdeviceptr*>devPtr, _driver_handle, flags)
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t cudaIpcCloseMemHandle(void* devPtr) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuIpcCloseMemHandle(<ccuda.CUdeviceptr>devPtr)
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t cudaDeviceFlushGPUDirectRDMAWrites(cudaFlushGPUDirectRDMAWritesTarget target, cudaFlushGPUDirectRDMAWritesScope scope) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuFlushGPUDirectRDMAWrites(<ccuda.CUflushGPUDirectRDMAWritesTarget>target, <ccuda.CUflushGPUDirectRDMAWritesScope>scope)
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t cudaThreadExit() nogil except ?cudaErrorCallRequiresNewerDriver:
    return _cudaThreadExit()

cdef cudaError_t cudaThreadSynchronize() nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuCtxSynchronize()
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t cudaThreadSetLimit(cudaLimit limit, size_t value) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuCtxSetLimit(<ccuda.CUlimit>limit, value)
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t cudaThreadGetLimit(size_t* pValue, cudaLimit limit) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuCtxGetLimit(pValue, <ccuda.CUlimit>limit)
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t cudaGetLastError() nogil except ?cudaErrorCallRequiresNewerDriver:
    return _cudaGetLastError()

cdef cudaError_t cudaPeekAtLastError() nogil except ?cudaErrorCallRequiresNewerDriver:
    return _cudaPeekAtLastError()

cdef const char* cudaGetErrorName(cudaError_t error) nogil except ?NULL:
    cdef const char* pStr = "unrecognized error code"
    if error == cudaSuccess:
        return "cudaSuccess"
    if error == cudaErrorInvalidValue:
        return "cudaErrorInvalidValue"
    if error == cudaErrorMemoryAllocation:
        return "cudaErrorMemoryAllocation"
    if error == cudaErrorInitializationError:
        return "cudaErrorInitializationError"
    if error == cudaErrorCudartUnloading:
        return "cudaErrorCudartUnloading"
    if error == cudaErrorProfilerDisabled:
        return "cudaErrorProfilerDisabled"
    if error == cudaErrorProfilerNotInitialized:
        return "cudaErrorProfilerNotInitialized"
    if error == cudaErrorProfilerAlreadyStarted:
        return "cudaErrorProfilerAlreadyStarted"
    if error == cudaErrorProfilerAlreadyStopped:
        return "cudaErrorProfilerAlreadyStopped"
    if error == cudaErrorInvalidConfiguration:
        return "cudaErrorInvalidConfiguration"
    if error == cudaErrorInvalidPitchValue:
        return "cudaErrorInvalidPitchValue"
    if error == cudaErrorInvalidSymbol:
        return "cudaErrorInvalidSymbol"
    if error == cudaErrorInvalidHostPointer:
        return "cudaErrorInvalidHostPointer"
    if error == cudaErrorInvalidDevicePointer:
        return "cudaErrorInvalidDevicePointer"
    if error == cudaErrorInvalidTexture:
        return "cudaErrorInvalidTexture"
    if error == cudaErrorInvalidTextureBinding:
        return "cudaErrorInvalidTextureBinding"
    if error == cudaErrorInvalidChannelDescriptor:
        return "cudaErrorInvalidChannelDescriptor"
    if error == cudaErrorInvalidMemcpyDirection:
        return "cudaErrorInvalidMemcpyDirection"
    if error == cudaErrorAddressOfConstant:
        return "cudaErrorAddressOfConstant"
    if error == cudaErrorTextureFetchFailed:
        return "cudaErrorTextureFetchFailed"
    if error == cudaErrorTextureNotBound:
        return "cudaErrorTextureNotBound"
    if error == cudaErrorSynchronizationError:
        return "cudaErrorSynchronizationError"
    if error == cudaErrorInvalidFilterSetting:
        return "cudaErrorInvalidFilterSetting"
    if error == cudaErrorInvalidNormSetting:
        return "cudaErrorInvalidNormSetting"
    if error == cudaErrorMixedDeviceExecution:
        return "cudaErrorMixedDeviceExecution"
    if error == cudaErrorNotYetImplemented:
        return "cudaErrorNotYetImplemented"
    if error == cudaErrorMemoryValueTooLarge:
        return "cudaErrorMemoryValueTooLarge"
    if error == cudaErrorStubLibrary:
        return "cudaErrorStubLibrary"
    if error == cudaErrorInsufficientDriver:
        return "cudaErrorInsufficientDriver"
    if error == cudaErrorCallRequiresNewerDriver:
        return "cudaErrorCallRequiresNewerDriver"
    if error == cudaErrorInvalidSurface:
        return "cudaErrorInvalidSurface"
    if error == cudaErrorDuplicateVariableName:
        return "cudaErrorDuplicateVariableName"
    if error == cudaErrorDuplicateTextureName:
        return "cudaErrorDuplicateTextureName"
    if error == cudaErrorDuplicateSurfaceName:
        return "cudaErrorDuplicateSurfaceName"
    if error == cudaErrorDevicesUnavailable:
        return "cudaErrorDevicesUnavailable"
    if error == cudaErrorIncompatibleDriverContext:
        return "cudaErrorIncompatibleDriverContext"
    if error == cudaErrorMissingConfiguration:
        return "cudaErrorMissingConfiguration"
    if error == cudaErrorPriorLaunchFailure:
        return "cudaErrorPriorLaunchFailure"
    if error == cudaErrorLaunchMaxDepthExceeded:
        return "cudaErrorLaunchMaxDepthExceeded"
    if error == cudaErrorLaunchFileScopedTex:
        return "cudaErrorLaunchFileScopedTex"
    if error == cudaErrorLaunchFileScopedSurf:
        return "cudaErrorLaunchFileScopedSurf"
    if error == cudaErrorSyncDepthExceeded:
        return "cudaErrorSyncDepthExceeded"
    if error == cudaErrorLaunchPendingCountExceeded:
        return "cudaErrorLaunchPendingCountExceeded"
    if error == cudaErrorInvalidDeviceFunction:
        return "cudaErrorInvalidDeviceFunction"
    if error == cudaErrorNoDevice:
        return "cudaErrorNoDevice"
    if error == cudaErrorInvalidDevice:
        return "cudaErrorInvalidDevice"
    if error == cudaErrorDeviceNotLicensed:
        return "cudaErrorDeviceNotLicensed"
    if error == cudaErrorSoftwareValidityNotEstablished:
        return "cudaErrorSoftwareValidityNotEstablished"
    if error == cudaErrorStartupFailure:
        return "cudaErrorStartupFailure"
    if error == cudaErrorInvalidKernelImage:
        return "cudaErrorInvalidKernelImage"
    if error == cudaErrorDeviceUninitialized:
        return "cudaErrorDeviceUninitialized"
    if error == cudaErrorMapBufferObjectFailed:
        return "cudaErrorMapBufferObjectFailed"
    if error == cudaErrorUnmapBufferObjectFailed:
        return "cudaErrorUnmapBufferObjectFailed"
    if error == cudaErrorArrayIsMapped:
        return "cudaErrorArrayIsMapped"
    if error == cudaErrorAlreadyMapped:
        return "cudaErrorAlreadyMapped"
    if error == cudaErrorNoKernelImageForDevice:
        return "cudaErrorNoKernelImageForDevice"
    if error == cudaErrorAlreadyAcquired:
        return "cudaErrorAlreadyAcquired"
    if error == cudaErrorNotMapped:
        return "cudaErrorNotMapped"
    if error == cudaErrorNotMappedAsArray:
        return "cudaErrorNotMappedAsArray"
    if error == cudaErrorNotMappedAsPointer:
        return "cudaErrorNotMappedAsPointer"
    if error == cudaErrorECCUncorrectable:
        return "cudaErrorECCUncorrectable"
    if error == cudaErrorUnsupportedLimit:
        return "cudaErrorUnsupportedLimit"
    if error == cudaErrorDeviceAlreadyInUse:
        return "cudaErrorDeviceAlreadyInUse"
    if error == cudaErrorPeerAccessUnsupported:
        return "cudaErrorPeerAccessUnsupported"
    if error == cudaErrorInvalidPtx:
        return "cudaErrorInvalidPtx"
    if error == cudaErrorInvalidGraphicsContext:
        return "cudaErrorInvalidGraphicsContext"
    if error == cudaErrorNvlinkUncorrectable:
        return "cudaErrorNvlinkUncorrectable"
    if error == cudaErrorJitCompilerNotFound:
        return "cudaErrorJitCompilerNotFound"
    if error == cudaErrorUnsupportedPtxVersion:
        return "cudaErrorUnsupportedPtxVersion"
    if error == cudaErrorJitCompilationDisabled:
        return "cudaErrorJitCompilationDisabled"
    if error == cudaErrorUnsupportedExecAffinity:
        return "cudaErrorUnsupportedExecAffinity"
    if error == cudaErrorInvalidSource:
        return "cudaErrorInvalidSource"
    if error == cudaErrorFileNotFound:
        return "cudaErrorFileNotFound"
    if error == cudaErrorSharedObjectSymbolNotFound:
        return "cudaErrorSharedObjectSymbolNotFound"
    if error == cudaErrorSharedObjectInitFailed:
        return "cudaErrorSharedObjectInitFailed"
    if error == cudaErrorOperatingSystem:
        return "cudaErrorOperatingSystem"
    if error == cudaErrorInvalidResourceHandle:
        return "cudaErrorInvalidResourceHandle"
    if error == cudaErrorIllegalState:
        return "cudaErrorIllegalState"
    if error == cudaErrorSymbolNotFound:
        return "cudaErrorSymbolNotFound"
    if error == cudaErrorNotReady:
        return "cudaErrorNotReady"
    if error == cudaErrorIllegalAddress:
        return "cudaErrorIllegalAddress"
    if error == cudaErrorLaunchOutOfResources:
        return "cudaErrorLaunchOutOfResources"
    if error == cudaErrorLaunchTimeout:
        return "cudaErrorLaunchTimeout"
    if error == cudaErrorLaunchIncompatibleTexturing:
        return "cudaErrorLaunchIncompatibleTexturing"
    if error == cudaErrorPeerAccessAlreadyEnabled:
        return "cudaErrorPeerAccessAlreadyEnabled"
    if error == cudaErrorPeerAccessNotEnabled:
        return "cudaErrorPeerAccessNotEnabled"
    if error == cudaErrorSetOnActiveProcess:
        return "cudaErrorSetOnActiveProcess"
    if error == cudaErrorContextIsDestroyed:
        return "cudaErrorContextIsDestroyed"
    if error == cudaErrorAssert:
        return "cudaErrorAssert"
    if error == cudaErrorTooManyPeers:
        return "cudaErrorTooManyPeers"
    if error == cudaErrorHostMemoryAlreadyRegistered:
        return "cudaErrorHostMemoryAlreadyRegistered"
    if error == cudaErrorHostMemoryNotRegistered:
        return "cudaErrorHostMemoryNotRegistered"
    if error == cudaErrorHardwareStackError:
        return "cudaErrorHardwareStackError"
    if error == cudaErrorIllegalInstruction:
        return "cudaErrorIllegalInstruction"
    if error == cudaErrorMisalignedAddress:
        return "cudaErrorMisalignedAddress"
    if error == cudaErrorInvalidAddressSpace:
        return "cudaErrorInvalidAddressSpace"
    if error == cudaErrorInvalidPc:
        return "cudaErrorInvalidPc"
    if error == cudaErrorLaunchFailure:
        return "cudaErrorLaunchFailure"
    if error == cudaErrorCooperativeLaunchTooLarge:
        return "cudaErrorCooperativeLaunchTooLarge"
    if error == cudaErrorNotPermitted:
        return "cudaErrorNotPermitted"
    if error == cudaErrorNotSupported:
        return "cudaErrorNotSupported"
    if error == cudaErrorSystemNotReady:
        return "cudaErrorSystemNotReady"
    if error == cudaErrorSystemDriverMismatch:
        return "cudaErrorSystemDriverMismatch"
    if error == cudaErrorCompatNotSupportedOnDevice:
        return "cudaErrorCompatNotSupportedOnDevice"
    if error == cudaErrorMpsConnectionFailed:
        return "cudaErrorMpsConnectionFailed"
    if error == cudaErrorMpsRpcFailure:
        return "cudaErrorMpsRpcFailure"
    if error == cudaErrorMpsServerNotReady:
        return "cudaErrorMpsServerNotReady"
    if error == cudaErrorMpsMaxClientsReached:
        return "cudaErrorMpsMaxClientsReached"
    if error == cudaErrorMpsMaxConnectionsReached:
        return "cudaErrorMpsMaxConnectionsReached"
    if error == cudaErrorStreamCaptureUnsupported:
        return "cudaErrorStreamCaptureUnsupported"
    if error == cudaErrorStreamCaptureInvalidated:
        return "cudaErrorStreamCaptureInvalidated"
    if error == cudaErrorStreamCaptureMerge:
        return "cudaErrorStreamCaptureMerge"
    if error == cudaErrorStreamCaptureUnmatched:
        return "cudaErrorStreamCaptureUnmatched"
    if error == cudaErrorStreamCaptureUnjoined:
        return "cudaErrorStreamCaptureUnjoined"
    if error == cudaErrorStreamCaptureIsolation:
        return "cudaErrorStreamCaptureIsolation"
    if error == cudaErrorStreamCaptureImplicit:
        return "cudaErrorStreamCaptureImplicit"
    if error == cudaErrorCapturedEvent:
        return "cudaErrorCapturedEvent"
    if error == cudaErrorStreamCaptureWrongThread:
        return "cudaErrorStreamCaptureWrongThread"
    if error == cudaErrorTimeout:
        return "cudaErrorTimeout"
    if error == cudaErrorGraphExecUpdateFailure:
        return "cudaErrorGraphExecUpdateFailure"
    if error == cudaErrorExternalDevice:
        return "cudaErrorExternalDevice"
    if error == cudaErrorUnknown:
        return "cudaErrorUnknown"
    if error == cudaErrorApiFailureBase:
        return "cudaErrorApiFailureBase"
    return pStr

cdef const char* cudaGetErrorString(cudaError_t error) nogil except ?NULL:
    return _cudaGetErrorString(error)

cdef cudaError_t cudaGetDeviceCount(int* count) nogil except ?cudaErrorCallRequiresNewerDriver:
    return _cudaGetDeviceCount(count)

cdef cudaError_t cudaGetDeviceProperties(cudaDeviceProp* prop, int device) nogil except ?cudaErrorCallRequiresNewerDriver:
    return _cudaGetDeviceProperties(prop, device)

cdef cudaError_t cudaDeviceGetAttribute(int* value, cudaDeviceAttr attr, int device) nogil except ?cudaErrorCallRequiresNewerDriver:
    return _cudaDeviceGetAttribute(value, attr, device)

cdef cudaError_t cudaDeviceGetDefaultMemPool(cudaMemPool_t* memPool, int device) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuDeviceGetDefaultMemPool(<ccuda.CUmemoryPool*>memPool, <ccuda.CUdevice>device)
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t cudaDeviceSetMemPool(int device, cudaMemPool_t memPool) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuDeviceSetMemPool(<ccuda.CUdevice>device, <ccuda.CUmemoryPool>memPool)
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t cudaDeviceGetMemPool(cudaMemPool_t* memPool, int device) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuDeviceGetMemPool(<ccuda.CUmemoryPool*>memPool, <ccuda.CUdevice>device)
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t cudaDeviceGetNvSciSyncAttributes(void* nvSciSyncAttrList, int device, int flags) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuDeviceGetNvSciSyncAttributes(nvSciSyncAttrList, <ccuda.CUdevice>device, flags)
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t cudaDeviceGetP2PAttribute(int* value, cudaDeviceP2PAttr attr, int srcDevice, int dstDevice) nogil except ?cudaErrorCallRequiresNewerDriver:
    return _cudaDeviceGetP2PAttribute(value, attr, srcDevice, dstDevice)

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
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuStreamCreate(<ccuda.CUstream*>pStream, flags)
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t cudaStreamCreateWithPriority(cudaStream_t* pStream, unsigned int flags, int priority) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuStreamCreateWithPriority(<ccuda.CUstream*>pStream, flags, priority)
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t cudaStreamGetPriority(cudaStream_t hStream, int* priority) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuStreamGetPriority(<ccuda.CUstream>hStream, priority)
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t cudaStreamGetFlags(cudaStream_t hStream, unsigned int* flags) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuStreamGetFlags(<ccuda.CUstream>hStream, flags)
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t cudaCtxResetPersistingL2Cache() nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuCtxResetPersistingL2Cache()
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t cudaStreamCopyAttributes(cudaStream_t dst, cudaStream_t src) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuStreamCopyAttributes(<ccuda.CUstream>dst, <ccuda.CUstream>src)
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t cudaStreamGetAttribute(cudaStream_t hStream, cudaStreamAttrID attr, cudaStreamAttrValue* value_out) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuStreamGetAttribute(<ccuda.CUstream>hStream, <ccuda.CUstreamAttrID>attr, <ccuda.CUstreamAttrValue*>value_out)
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t cudaStreamSetAttribute(cudaStream_t hStream, cudaStreamAttrID attr, const cudaStreamAttrValue* value) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuStreamSetAttribute(<ccuda.CUstream>hStream, <ccuda.CUstreamAttrID>attr, <ccuda.CUstreamAttrValue*>value)
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t cudaStreamDestroy(cudaStream_t stream) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuStreamDestroy_v2(<ccuda.CUstream>stream)
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned int flags) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuStreamWaitEvent(<ccuda.CUstream>stream, <ccuda.CUevent>event, flags)
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t cudaStreamAddCallback(cudaStream_t stream, cudaStreamCallback_t callback, void* userData, unsigned int flags) nogil except ?cudaErrorCallRequiresNewerDriver:
    return _cudaStreamAddCallback(stream, callback, userData, flags)

cdef cudaError_t cudaStreamSynchronize(cudaStream_t stream) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuStreamSynchronize(<ccuda.CUstream>stream)
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t cudaStreamQuery(cudaStream_t stream) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuStreamQuery(<ccuda.CUstream>stream)
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t cudaStreamAttachMemAsync(cudaStream_t stream, void* devPtr, size_t length, unsigned int flags) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuStreamAttachMemAsync(<ccuda.CUstream>stream, <ccuda.CUdeviceptr>devPtr, length, flags)
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t cudaStreamBeginCapture(cudaStream_t stream, cudaStreamCaptureMode mode) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuStreamBeginCapture_v2(<ccuda.CUstream>stream, <ccuda.CUstreamCaptureMode>mode)
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t cudaThreadExchangeStreamCaptureMode(cudaStreamCaptureMode* mode) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuThreadExchangeStreamCaptureMode(<ccuda.CUstreamCaptureMode*>mode)
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t cudaStreamEndCapture(cudaStream_t stream, cudaGraph_t* pGraph) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuStreamEndCapture(<ccuda.CUstream>stream, <ccuda.CUgraph*>pGraph)
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t cudaStreamIsCapturing(cudaStream_t stream, cudaStreamCaptureStatus* pCaptureStatus) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuStreamIsCapturing(<ccuda.CUstream>stream, <ccuda.CUstreamCaptureStatus*>pCaptureStatus)
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t cudaStreamGetCaptureInfo(cudaStream_t stream, cudaStreamCaptureStatus* pCaptureStatus, unsigned long long* pId) nogil except ?cudaErrorCallRequiresNewerDriver:
    return _cudaStreamGetCaptureInfo(stream, pCaptureStatus, pId)

cdef cudaError_t cudaStreamGetCaptureInfo_v2(cudaStream_t stream, cudaStreamCaptureStatus* captureStatus_out, unsigned long long* id_out, cudaGraph_t* graph_out, const cudaGraphNode_t** dependencies_out, size_t* numDependencies_out) nogil except ?cudaErrorCallRequiresNewerDriver:
    return _cudaStreamGetCaptureInfo_v2(stream, captureStatus_out, id_out, graph_out, dependencies_out, numDependencies_out)

cdef cudaError_t cudaStreamUpdateCaptureDependencies(cudaStream_t stream, cudaGraphNode_t* dependencies, size_t numDependencies, unsigned int flags) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuStreamUpdateCaptureDependencies(<ccuda.CUstream>stream, <ccuda.CUgraphNode*>dependencies, numDependencies, flags)
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t cudaEventCreate(cudaEvent_t* event) nogil except ?cudaErrorCallRequiresNewerDriver:
    return _cudaEventCreate(event)

cdef cudaError_t cudaEventCreateWithFlags(cudaEvent_t* event, unsigned int flags) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuEventCreate(<ccuda.CUevent*>event, flags)
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuEventRecord(<ccuda.CUevent>event, <ccuda.CUstream>stream)
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t cudaEventRecordWithFlags(cudaEvent_t event, cudaStream_t stream, unsigned int flags) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuEventRecordWithFlags(<ccuda.CUevent>event, <ccuda.CUstream>stream, flags)
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t cudaEventQuery(cudaEvent_t event) nogil except ?cudaErrorCallRequiresNewerDriver:
    return _cudaEventQuery(event)

cdef cudaError_t cudaEventSynchronize(cudaEvent_t event) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuEventSynchronize(<ccuda.CUevent>event)
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t cudaEventDestroy(cudaEvent_t event) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuEventDestroy_v2(<ccuda.CUevent>event)
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t cudaEventElapsedTime(float* ms, cudaEvent_t start, cudaEvent_t end) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuEventElapsedTime(ms, <ccuda.CUevent>start, <ccuda.CUevent>end)
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t cudaImportExternalMemory(cudaExternalMemory_t* extMem_out, const cudaExternalMemoryHandleDesc* memHandleDesc) nogil except ?cudaErrorCallRequiresNewerDriver:
    return _cudaImportExternalMemory(extMem_out, memHandleDesc)

cdef cudaError_t cudaExternalMemoryGetMappedBuffer(void** devPtr, cudaExternalMemory_t extMem, const cudaExternalMemoryBufferDesc* bufferDesc) nogil except ?cudaErrorCallRequiresNewerDriver:
    return _cudaExternalMemoryGetMappedBuffer(devPtr, extMem, bufferDesc)

cdef cudaError_t cudaExternalMemoryGetMappedMipmappedArray(cudaMipmappedArray_t* mipmap, cudaExternalMemory_t extMem, const cudaExternalMemoryMipmappedArrayDesc* mipmapDesc) nogil except ?cudaErrorCallRequiresNewerDriver:
    return _cudaExternalMemoryGetMappedMipmappedArray(mipmap, extMem, mipmapDesc)

cdef cudaError_t cudaDestroyExternalMemory(cudaExternalMemory_t extMem) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuDestroyExternalMemory(<ccuda.CUexternalMemory>extMem)
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t cudaImportExternalSemaphore(cudaExternalSemaphore_t* extSem_out, const cudaExternalSemaphoreHandleDesc* semHandleDesc) nogil except ?cudaErrorCallRequiresNewerDriver:
    return _cudaImportExternalSemaphore(extSem_out, semHandleDesc)

cdef cudaError_t cudaSignalExternalSemaphoresAsync(const cudaExternalSemaphore_t* extSemArray, const cudaExternalSemaphoreSignalParams* paramsArray, unsigned int numExtSems, cudaStream_t stream) nogil except ?cudaErrorCallRequiresNewerDriver:
    return _cudaSignalExternalSemaphoresAsync(extSemArray, paramsArray, numExtSems, stream)

cdef cudaError_t cudaWaitExternalSemaphoresAsync(const cudaExternalSemaphore_t* extSemArray, const cudaExternalSemaphoreWaitParams* paramsArray, unsigned int numExtSems, cudaStream_t stream) nogil except ?cudaErrorCallRequiresNewerDriver:
    return _cudaWaitExternalSemaphoresAsync(extSemArray, paramsArray, numExtSems, stream)

cdef cudaError_t cudaDestroyExternalSemaphore(cudaExternalSemaphore_t extSem) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuDestroyExternalSemaphore(<ccuda.CUexternalSemaphore>extSem)
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t cudaFuncSetCacheConfig(const void* func, cudaFuncCache cacheConfig) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuFuncSetCacheConfig(<ccuda.CUfunction>func, <ccuda.CUfunc_cache>cacheConfig)
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t cudaFuncSetSharedMemConfig(const void* func, cudaSharedMemConfig config) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuFuncSetSharedMemConfig(<ccuda.CUfunction>func, <ccuda.CUsharedconfig>config)
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t cudaFuncGetAttributes(cudaFuncAttributes* attr, const void* func) nogil except ?cudaErrorCallRequiresNewerDriver:
    return _cudaFuncGetAttributes(attr, func)

cdef cudaError_t cudaFuncSetAttribute(const void* func, cudaFuncAttribute attr, int value) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuFuncSetAttribute(<ccuda.CUfunction>func, <ccuda.CUfunction_attribute>attr, value)
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t cudaSetDoubleForDevice(double* d) nogil except ?cudaErrorCallRequiresNewerDriver:
    return _cudaSetDoubleForDevice(d)

cdef cudaError_t cudaSetDoubleForHost(double* d) nogil except ?cudaErrorCallRequiresNewerDriver:
    return _cudaSetDoubleForHost(d)

cdef cudaError_t cudaLaunchHostFunc(cudaStream_t stream, cudaHostFn_t fn, void* userData) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuLaunchHostFunc(<ccuda.CUstream>stream, <ccuda.CUhostFn>fn, userData)
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(int* numBlocks, const void* func, int blockSize, size_t dynamicSMemSize) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuOccupancyMaxActiveBlocksPerMultiprocessor(numBlocks, <ccuda.CUfunction>func, blockSize, dynamicSMemSize)
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t cudaOccupancyAvailableDynamicSMemPerBlock(size_t* dynamicSmemSize, const void* func, int numBlocks, int blockSize) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuOccupancyAvailableDynamicSMemPerBlock(dynamicSmemSize, <ccuda.CUfunction>func, numBlocks, blockSize)
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int* numBlocks, const void* func, int blockSize, size_t dynamicSMemSize, unsigned int flags) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(numBlocks, <ccuda.CUfunction>func, blockSize, dynamicSMemSize, flags)
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t cudaMallocManaged(void** devPtr, size_t size, unsigned int flags) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuMemAllocManaged(<ccuda.CUdeviceptr*>devPtr, size, flags)
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t cudaMalloc(void** devPtr, size_t size) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuMemAlloc_v2(<ccuda.CUdeviceptr*>devPtr, size)
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t cudaMallocHost(void** ptr, size_t size) nogil except ?cudaErrorCallRequiresNewerDriver:
    return _cudaMallocHost(ptr, size)

cdef cudaError_t cudaMallocPitch(void** devPtr, size_t* pitch, size_t width, size_t height) nogil except ?cudaErrorCallRequiresNewerDriver:
    return _cudaMallocPitch(devPtr, pitch, width, height)

cdef cudaError_t cudaMallocArray(cudaArray_t* array, const cudaChannelFormatDesc* desc, size_t width, size_t height, unsigned int flags) nogil except ?cudaErrorCallRequiresNewerDriver:
    return _cudaMallocArray(array, desc, width, height, flags)

cdef cudaError_t cudaFree(void* devPtr) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuMemFree_v2(<ccuda.CUdeviceptr>devPtr)
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t cudaFreeHost(void* ptr) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuMemFreeHost(ptr)
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t cudaFreeArray(cudaArray_t array) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuArrayDestroy(<ccuda.CUarray>array)
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t cudaFreeMipmappedArray(cudaMipmappedArray_t mipmappedArray) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuMipmappedArrayDestroy(<ccuda.CUmipmappedArray>mipmappedArray)
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t cudaHostAlloc(void** pHost, size_t size, unsigned int flags) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuMemHostAlloc(pHost, size, flags)
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t cudaHostRegister(void* ptr, size_t size, unsigned int flags) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuMemHostRegister_v2(ptr, size, flags)
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t cudaHostUnregister(void* ptr) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuMemHostUnregister(ptr)
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t cudaHostGetDevicePointer(void** pDevice, void* pHost, unsigned int flags) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuMemHostGetDevicePointer_v2(<ccuda.CUdeviceptr*>pDevice, pHost, flags)
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t cudaHostGetFlags(unsigned int* pFlags, void* pHost) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuMemHostGetFlags(pFlags, pHost)
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t cudaMalloc3D(cudaPitchedPtr* pitchedDevPtr, cudaExtent extent) nogil except ?cudaErrorCallRequiresNewerDriver:
    return _cudaMalloc3D(pitchedDevPtr, extent)

cdef cudaError_t cudaMalloc3DArray(cudaArray_t* array, const cudaChannelFormatDesc* desc, cudaExtent extent, unsigned int flags) nogil except ?cudaErrorCallRequiresNewerDriver:
    return _cudaMalloc3DArray(array, desc, extent, flags)

cdef cudaError_t cudaMallocMipmappedArray(cudaMipmappedArray_t* mipmappedArray, const cudaChannelFormatDesc* desc, cudaExtent extent, unsigned int numLevels, unsigned int flags) nogil except ?cudaErrorCallRequiresNewerDriver:
    return _cudaMallocMipmappedArray(mipmappedArray, desc, extent, numLevels, flags)

cdef cudaError_t cudaGetMipmappedArrayLevel(cudaArray_t* levelArray, cudaMipmappedArray_const_t mipmappedArray, unsigned int level) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuMipmappedArrayGetLevel(<ccuda.CUarray*>levelArray, <ccuda.CUmipmappedArray>mipmappedArray, level)
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t cudaMemcpy3D(const cudaMemcpy3DParms* p) nogil except ?cudaErrorCallRequiresNewerDriver:
    return _cudaMemcpy3D(p)

cdef cudaError_t cudaMemcpy3DPeer(const cudaMemcpy3DPeerParms* p) nogil except ?cudaErrorCallRequiresNewerDriver:
    return _cudaMemcpy3DPeer(p)

cdef cudaError_t cudaMemcpy3DAsync(const cudaMemcpy3DParms* p, cudaStream_t stream) nogil except ?cudaErrorCallRequiresNewerDriver:
    return _cudaMemcpy3DAsync(p, stream)

cdef cudaError_t cudaMemcpy3DPeerAsync(const cudaMemcpy3DPeerParms* p, cudaStream_t stream) nogil except ?cudaErrorCallRequiresNewerDriver:
    return _cudaMemcpy3DPeerAsync(p, stream)

cdef cudaError_t cudaMemGetInfo(size_t* free, size_t* total) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuMemGetInfo_v2(free, total)
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t cudaArrayGetInfo(cudaChannelFormatDesc* desc, cudaExtent* extent, unsigned int* flags, cudaArray_t array) nogil except ?cudaErrorCallRequiresNewerDriver:
    return _cudaArrayGetInfo(desc, extent, flags, array)

cdef cudaError_t cudaArrayGetPlane(cudaArray_t* pPlaneArray, cudaArray_t hArray, unsigned int planeIdx) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuArrayGetPlane(<ccuda.CUarray*>pPlaneArray, <ccuda.CUarray>hArray, planeIdx)
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t cudaArrayGetMemoryRequirements(cudaArrayMemoryRequirements* memoryRequirements, cudaArray_t array, int device) nogil except ?cudaErrorCallRequiresNewerDriver:
    return _cudaArrayGetMemoryRequirements(memoryRequirements, array, device)

cdef cudaError_t cudaMipmappedArrayGetMemoryRequirements(cudaArrayMemoryRequirements* memoryRequirements, cudaMipmappedArray_t mipmap, int device) nogil except ?cudaErrorCallRequiresNewerDriver:
    return _cudaMipmappedArrayGetMemoryRequirements(memoryRequirements, mipmap, device)

cdef cudaError_t cudaArrayGetSparseProperties(cudaArraySparseProperties* sparseProperties, cudaArray_t array) nogil except ?cudaErrorCallRequiresNewerDriver:
    return _cudaArrayGetSparseProperties(sparseProperties, array)

cdef cudaError_t cudaMipmappedArrayGetSparseProperties(cudaArraySparseProperties* sparseProperties, cudaMipmappedArray_t mipmap) nogil except ?cudaErrorCallRequiresNewerDriver:
    return _cudaMipmappedArrayGetSparseProperties(sparseProperties, mipmap)

cdef cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind) nogil except ?cudaErrorCallRequiresNewerDriver:
    return _cudaMemcpy(dst, src, count, kind)

cdef cudaError_t cudaMemcpyPeer(void* dst, int dstDevice, const void* src, int srcDevice, size_t count) nogil except ?cudaErrorCallRequiresNewerDriver:
    return _cudaMemcpyPeer(dst, dstDevice, src, srcDevice, count)

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
    return _cudaMemcpyPeerAsync(dst, dstDevice, src, srcDevice, count, stream)

cdef cudaError_t cudaMemcpy2DAsync(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream) nogil except ?cudaErrorCallRequiresNewerDriver:
    return _cudaMemcpy2DAsync(dst, dpitch, src, spitch, width, height, kind, stream)

cdef cudaError_t cudaMemcpy2DToArrayAsync(cudaArray_t dst, size_t wOffset, size_t hOffset, const void* src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream) nogil except ?cudaErrorCallRequiresNewerDriver:
    return _cudaMemcpy2DToArrayAsync(dst, wOffset, hOffset, src, spitch, width, height, kind, stream)

cdef cudaError_t cudaMemcpy2DFromArrayAsync(void* dst, size_t dpitch, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream) nogil except ?cudaErrorCallRequiresNewerDriver:
    return _cudaMemcpy2DFromArrayAsync(dst, dpitch, src, wOffset, hOffset, width, height, kind, stream)

cdef cudaError_t cudaMemset(void* devPtr, int value, size_t count) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuMemsetD8_v2(<ccuda.CUdeviceptr>devPtr, <unsigned char>value, count)
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t cudaMemset2D(void* devPtr, size_t pitch, int value, size_t width, size_t height) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuMemsetD2D8_v2(<ccuda.CUdeviceptr>devPtr, pitch, <unsigned char>value, width, height)
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t cudaMemset3D(cudaPitchedPtr pitchedDevPtr, int value, cudaExtent extent) nogil except ?cudaErrorCallRequiresNewerDriver:
    return _cudaMemset3D(pitchedDevPtr, value, extent)

cdef cudaError_t cudaMemsetAsync(void* devPtr, int value, size_t count, cudaStream_t stream) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuMemsetD8Async(<ccuda.CUdeviceptr>devPtr, <unsigned char>value, count, <ccuda.CUstream>stream)
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t cudaMemset2DAsync(void* devPtr, size_t pitch, int value, size_t width, size_t height, cudaStream_t stream) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuMemsetD2D8Async(<ccuda.CUdeviceptr>devPtr, pitch, <unsigned char>value, width, height, <ccuda.CUstream>stream)
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t cudaMemset3DAsync(cudaPitchedPtr pitchedDevPtr, int value, cudaExtent extent, cudaStream_t stream) nogil except ?cudaErrorCallRequiresNewerDriver:
    return _cudaMemset3DAsync(pitchedDevPtr, value, extent, stream)

cdef cudaError_t cudaMemPrefetchAsync(const void* devPtr, size_t count, int dstDevice, cudaStream_t stream) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuMemPrefetchAsync(<ccuda.CUdeviceptr>devPtr, count, <ccuda.CUdevice>dstDevice, <ccuda.CUstream>stream)
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t cudaMemAdvise(const void* devPtr, size_t count, cudaMemoryAdvise advice, int device) nogil except ?cudaErrorCallRequiresNewerDriver:
    return _cudaMemAdvise(devPtr, count, advice, device)

cdef cudaError_t cudaMemRangeGetAttribute(void* data, size_t dataSize, cudaMemRangeAttribute attribute, const void* devPtr, size_t count) nogil except ?cudaErrorCallRequiresNewerDriver:
    return _cudaMemRangeGetAttribute(data, dataSize, attribute, devPtr, count)

cdef cudaError_t cudaMemRangeGetAttributes(void** data, size_t* dataSizes, cudaMemRangeAttribute* attributes, size_t numAttributes, const void* devPtr, size_t count) nogil except ?cudaErrorCallRequiresNewerDriver:
    return _cudaMemRangeGetAttributes(data, dataSizes, attributes, numAttributes, devPtr, count)

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
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuMemAllocAsync(<ccuda.CUdeviceptr*>devPtr, size, <ccuda.CUstream>hStream)
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t cudaFreeAsync(void* devPtr, cudaStream_t hStream) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuMemFreeAsync(<ccuda.CUdeviceptr>devPtr, <ccuda.CUstream>hStream)
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t cudaMemPoolTrimTo(cudaMemPool_t memPool, size_t minBytesToKeep) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuMemPoolTrimTo(<ccuda.CUmemoryPool>memPool, minBytesToKeep)
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t cudaMemPoolSetAttribute(cudaMemPool_t memPool, cudaMemPoolAttr attr, void* value) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuMemPoolSetAttribute(<ccuda.CUmemoryPool>memPool, <ccuda.CUmemPool_attribute>attr, value)
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t cudaMemPoolGetAttribute(cudaMemPool_t memPool, cudaMemPoolAttr attr, void* value) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuMemPoolGetAttribute(<ccuda.CUmemoryPool>memPool, <ccuda.CUmemPool_attribute>attr, value)
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t cudaMemPoolSetAccess(cudaMemPool_t memPool, const cudaMemAccessDesc* descList, size_t count) nogil except ?cudaErrorCallRequiresNewerDriver:
    return _cudaMemPoolSetAccess(memPool, descList, count)

cdef cudaError_t cudaMemPoolGetAccess(cudaMemAccessFlags* flags, cudaMemPool_t memPool, cudaMemLocation* location) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuMemPoolGetAccess(<ccuda.CUmemAccess_flags*>flags, <ccuda.CUmemoryPool>memPool, <ccuda.CUmemLocation*>location)
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t cudaMemPoolCreate(cudaMemPool_t* memPool, const cudaMemPoolProps* poolProps) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuMemPoolCreate(<ccuda.CUmemoryPool*>memPool, <ccuda.CUmemPoolProps*>poolProps)
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t cudaMemPoolDestroy(cudaMemPool_t memPool) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuMemPoolDestroy(<ccuda.CUmemoryPool>memPool)
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t cudaMallocFromPoolAsync(void** ptr, size_t size, cudaMemPool_t memPool, cudaStream_t stream) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuMemAllocFromPoolAsync(<ccuda.CUdeviceptr*>ptr, size, <ccuda.CUmemoryPool>memPool, <ccuda.CUstream>stream)
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t cudaMemPoolExportToShareableHandle(void* shareableHandle, cudaMemPool_t memPool, cudaMemAllocationHandleType handleType, unsigned int flags) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuMemPoolExportToShareableHandle(shareableHandle, <ccuda.CUmemoryPool>memPool, <ccuda.CUmemAllocationHandleType>handleType, <unsigned long long>flags)
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t cudaMemPoolImportFromShareableHandle(cudaMemPool_t* memPool, void* shareableHandle, cudaMemAllocationHandleType handleType, unsigned int flags) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuMemPoolImportFromShareableHandle(<ccuda.CUmemoryPool*>memPool, shareableHandle, <ccuda.CUmemAllocationHandleType>handleType, <unsigned long long>flags)
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t cudaMemPoolExportPointer(cudaMemPoolPtrExportData* exportData, void* ptr) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuMemPoolExportPointer(<ccuda.CUmemPoolPtrExportData*>exportData, <ccuda.CUdeviceptr>ptr)
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t cudaMemPoolImportPointer(void** ptr, cudaMemPool_t memPool, cudaMemPoolPtrExportData* exportData) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuMemPoolImportPointer(<ccuda.CUdeviceptr*>ptr, <ccuda.CUmemoryPool>memPool, <ccuda.CUmemPoolPtrExportData*>exportData)
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t cudaPointerGetAttributes(cudaPointerAttributes* attributes, const void* ptr) nogil except ?cudaErrorCallRequiresNewerDriver:
    return _cudaPointerGetAttributes(attributes, ptr)

cdef cudaError_t cudaDeviceCanAccessPeer(int* canAccessPeer, int device, int peerDevice) nogil except ?cudaErrorCallRequiresNewerDriver:
    return _cudaDeviceCanAccessPeer(canAccessPeer, device, peerDevice)

cdef cudaError_t cudaDeviceEnablePeerAccess(int peerDevice, unsigned int flags) nogil except ?cudaErrorCallRequiresNewerDriver:
    return _cudaDeviceEnablePeerAccess(peerDevice, flags)

cdef cudaError_t cudaDeviceDisablePeerAccess(int peerDevice) nogil except ?cudaErrorCallRequiresNewerDriver:
    return _cudaDeviceDisablePeerAccess(peerDevice)

cdef cudaError_t cudaGraphicsUnregisterResource(cudaGraphicsResource_t resource) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuGraphicsUnregisterResource(<ccuda.CUgraphicsResource>resource)
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t cudaGraphicsResourceSetMapFlags(cudaGraphicsResource_t resource, unsigned int flags) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuGraphicsResourceSetMapFlags_v2(<ccuda.CUgraphicsResource>resource, flags)
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t cudaGraphicsMapResources(int count, cudaGraphicsResource_t* resources, cudaStream_t stream) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuGraphicsMapResources(<unsigned int>count, <ccuda.CUgraphicsResource*>resources, <ccuda.CUstream>stream)
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t cudaGraphicsUnmapResources(int count, cudaGraphicsResource_t* resources, cudaStream_t stream) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuGraphicsUnmapResources(<unsigned int>count, <ccuda.CUgraphicsResource*>resources, <ccuda.CUstream>stream)
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t cudaGraphicsResourceGetMappedPointer(void** devPtr, size_t* size, cudaGraphicsResource_t resource) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuGraphicsResourceGetMappedPointer_v2(<ccuda.CUdeviceptr*>devPtr, size, <ccuda.CUgraphicsResource>resource)
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t cudaGraphicsSubResourceGetMappedArray(cudaArray_t* array, cudaGraphicsResource_t resource, unsigned int arrayIndex, unsigned int mipLevel) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuGraphicsSubResourceGetMappedArray(<ccuda.CUarray*>array, <ccuda.CUgraphicsResource>resource, arrayIndex, mipLevel)
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t cudaGraphicsResourceGetMappedMipmappedArray(cudaMipmappedArray_t* mipmappedArray, cudaGraphicsResource_t resource) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuGraphicsResourceGetMappedMipmappedArray(<ccuda.CUmipmappedArray*>mipmappedArray, <ccuda.CUgraphicsResource>resource)
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t cudaGetChannelDesc(cudaChannelFormatDesc* desc, cudaArray_const_t array) nogil except ?cudaErrorCallRequiresNewerDriver:
    return _cudaGetChannelDesc(desc, array)

cdef cudaChannelFormatDesc cudaCreateChannelDesc(int x, int y, int z, int w, cudaChannelFormatKind f) nogil:
    return _cudaCreateChannelDesc(x, y, z, w, f)

cdef cudaError_t cudaCreateTextureObject(cudaTextureObject_t* pTexObject, const cudaResourceDesc* pResDesc, const cudaTextureDesc* pTexDesc, const cudaResourceViewDesc* pResViewDesc) nogil except ?cudaErrorCallRequiresNewerDriver:
    return _cudaCreateTextureObject(pTexObject, pResDesc, pTexDesc, pResViewDesc)

cdef cudaError_t cudaDestroyTextureObject(cudaTextureObject_t texObject) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuTexObjectDestroy(<ccuda.CUtexObject>texObject)
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t cudaGetTextureObjectResourceDesc(cudaResourceDesc* pResDesc, cudaTextureObject_t texObject) nogil except ?cudaErrorCallRequiresNewerDriver:
    return _cudaGetTextureObjectResourceDesc(pResDesc, texObject)

cdef cudaError_t cudaGetTextureObjectTextureDesc(cudaTextureDesc* pTexDesc, cudaTextureObject_t texObject) nogil except ?cudaErrorCallRequiresNewerDriver:
    return _cudaGetTextureObjectTextureDesc(pTexDesc, texObject)

cdef cudaError_t cudaGetTextureObjectResourceViewDesc(cudaResourceViewDesc* pResViewDesc, cudaTextureObject_t texObject) nogil except ?cudaErrorCallRequiresNewerDriver:
    return _cudaGetTextureObjectResourceViewDesc(pResViewDesc, texObject)

cdef cudaError_t cudaCreateSurfaceObject(cudaSurfaceObject_t* pSurfObject, const cudaResourceDesc* pResDesc) nogil except ?cudaErrorCallRequiresNewerDriver:
    return _cudaCreateSurfaceObject(pSurfObject, pResDesc)

cdef cudaError_t cudaDestroySurfaceObject(cudaSurfaceObject_t surfObject) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuSurfObjectDestroy(<ccuda.CUsurfObject>surfObject)
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t cudaGetSurfaceObjectResourceDesc(cudaResourceDesc* pResDesc, cudaSurfaceObject_t surfObject) nogil except ?cudaErrorCallRequiresNewerDriver:
    return _cudaGetSurfaceObjectResourceDesc(pResDesc, surfObject)

cdef cudaError_t cudaDriverGetVersion(int* driverVersion) nogil except ?cudaErrorCallRequiresNewerDriver:
    return _cudaDriverGetVersion(driverVersion)

cdef cudaError_t cudaRuntimeGetVersion(int* runtimeVersion) nogil except ?cudaErrorCallRequiresNewerDriver:
    return _cudaRuntimeGetVersion(runtimeVersion)

cdef cudaError_t cudaGraphCreate(cudaGraph_t* pGraph, unsigned int flags) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuGraphCreate(<ccuda.CUgraph*>pGraph, flags)
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t cudaGraphAddKernelNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, const cudaKernelNodeParams* pNodeParams) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
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

    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuGraphAddKernelNode(<ccuda.CUgraphNode*>pGraphNode, <ccuda.CUgraph>graph, <ccuda.CUgraphNode*>pDependencies, numDependencies, &_driver_pNodeParams)
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t cudaGraphKernelNodeGetParams(cudaGraphNode_t node, cudaKernelNodeParams* pNodeParams) nogil except ?cudaErrorCallRequiresNewerDriver:
    return _cudaGraphKernelNodeGetParams(node, pNodeParams)

cdef cudaError_t cudaGraphKernelNodeSetParams(cudaGraphNode_t node, const cudaKernelNodeParams* pNodeParams) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
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

    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuGraphKernelNodeSetParams(<ccuda.CUgraphNode>node, &_driver_pNodeParams)
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t cudaGraphKernelNodeCopyAttributes(cudaGraphNode_t hSrc, cudaGraphNode_t hDst) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuGraphKernelNodeCopyAttributes(<ccuda.CUgraphNode>hSrc, <ccuda.CUgraphNode>hDst)
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t cudaGraphKernelNodeGetAttribute(cudaGraphNode_t hNode, cudaKernelNodeAttrID attr, cudaKernelNodeAttrValue* value_out) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuGraphKernelNodeGetAttribute(<ccuda.CUgraphNode>hNode, <ccuda.CUkernelNodeAttrID>attr, <ccuda.CUkernelNodeAttrValue*>value_out)
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t cudaGraphKernelNodeSetAttribute(cudaGraphNode_t hNode, cudaKernelNodeAttrID attr, const cudaKernelNodeAttrValue* value) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuGraphKernelNodeSetAttribute(<ccuda.CUgraphNode>hNode, <ccuda.CUkernelNodeAttrID>attr, <ccuda.CUkernelNodeAttrValue*>value)
    if err != cudaSuccess:
        _setLastError(err)
    return err

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
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuGraphMemsetNodeGetParams(<ccuda.CUgraphNode>node, <ccuda.CUDA_MEMSET_NODE_PARAMS*>pNodeParams)
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t cudaGraphMemsetNodeSetParams(cudaGraphNode_t node, const cudaMemsetParams* pNodeParams) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuGraphMemsetNodeSetParams(<ccuda.CUgraphNode>node, <ccuda.CUDA_MEMSET_NODE_PARAMS*>pNodeParams)
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t cudaGraphAddHostNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, const cudaHostNodeParams* pNodeParams) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuGraphAddHostNode(<ccuda.CUgraphNode*>pGraphNode, <ccuda.CUgraph>graph, <ccuda.CUgraphNode*>pDependencies, numDependencies, <ccuda.CUDA_HOST_NODE_PARAMS*>pNodeParams)
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t cudaGraphHostNodeGetParams(cudaGraphNode_t node, cudaHostNodeParams* pNodeParams) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuGraphHostNodeGetParams(<ccuda.CUgraphNode>node, <ccuda.CUDA_HOST_NODE_PARAMS*>pNodeParams)
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t cudaGraphHostNodeSetParams(cudaGraphNode_t node, const cudaHostNodeParams* pNodeParams) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuGraphHostNodeSetParams(<ccuda.CUgraphNode>node, <ccuda.CUDA_HOST_NODE_PARAMS*>pNodeParams)
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t cudaGraphAddChildGraphNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, cudaGraph_t childGraph) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuGraphAddChildGraphNode(<ccuda.CUgraphNode*>pGraphNode, <ccuda.CUgraph>graph, <ccuda.CUgraphNode*>pDependencies, numDependencies, <ccuda.CUgraph>childGraph)
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t cudaGraphChildGraphNodeGetGraph(cudaGraphNode_t node, cudaGraph_t* pGraph) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuGraphChildGraphNodeGetGraph(<ccuda.CUgraphNode>node, <ccuda.CUgraph*>pGraph)
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t cudaGraphAddEmptyNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuGraphAddEmptyNode(<ccuda.CUgraphNode*>pGraphNode, <ccuda.CUgraph>graph, <ccuda.CUgraphNode*>pDependencies, numDependencies)
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t cudaGraphAddEventRecordNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, cudaEvent_t event) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuGraphAddEventRecordNode(<ccuda.CUgraphNode*>pGraphNode, <ccuda.CUgraph>graph, <ccuda.CUgraphNode*>pDependencies, numDependencies, <ccuda.CUevent>event)
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t cudaGraphEventRecordNodeGetEvent(cudaGraphNode_t node, cudaEvent_t* event_out) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuGraphEventRecordNodeGetEvent(<ccuda.CUgraphNode>node, <ccuda.CUevent*>event_out)
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t cudaGraphEventRecordNodeSetEvent(cudaGraphNode_t node, cudaEvent_t event) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuGraphEventRecordNodeSetEvent(<ccuda.CUgraphNode>node, <ccuda.CUevent>event)
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t cudaGraphAddEventWaitNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, cudaEvent_t event) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuGraphAddEventWaitNode(<ccuda.CUgraphNode*>pGraphNode, <ccuda.CUgraph>graph, <ccuda.CUgraphNode*>pDependencies, numDependencies, <ccuda.CUevent>event)
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t cudaGraphEventWaitNodeGetEvent(cudaGraphNode_t node, cudaEvent_t* event_out) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuGraphEventWaitNodeGetEvent(<ccuda.CUgraphNode>node, <ccuda.CUevent*>event_out)
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t cudaGraphEventWaitNodeSetEvent(cudaGraphNode_t node, cudaEvent_t event) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuGraphEventWaitNodeSetEvent(<ccuda.CUgraphNode>node, <ccuda.CUevent>event)
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t cudaGraphAddExternalSemaphoresSignalNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, const cudaExternalSemaphoreSignalNodeParams* nodeParams) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuGraphAddExternalSemaphoresSignalNode(<ccuda.CUgraphNode*>pGraphNode, <ccuda.CUgraph>graph, <ccuda.CUgraphNode*>pDependencies, numDependencies, <ccuda.CUDA_EXT_SEM_SIGNAL_NODE_PARAMS*>nodeParams)
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t cudaGraphExternalSemaphoresSignalNodeGetParams(cudaGraphNode_t hNode, cudaExternalSemaphoreSignalNodeParams* params_out) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuGraphExternalSemaphoresSignalNodeGetParams(<ccuda.CUgraphNode>hNode, <ccuda.CUDA_EXT_SEM_SIGNAL_NODE_PARAMS*>params_out)
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t cudaGraphExternalSemaphoresSignalNodeSetParams(cudaGraphNode_t hNode, const cudaExternalSemaphoreSignalNodeParams* nodeParams) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuGraphExternalSemaphoresSignalNodeSetParams(<ccuda.CUgraphNode>hNode, <ccuda.CUDA_EXT_SEM_SIGNAL_NODE_PARAMS*>nodeParams)
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t cudaGraphAddExternalSemaphoresWaitNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, const cudaExternalSemaphoreWaitNodeParams* nodeParams) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuGraphAddExternalSemaphoresWaitNode(<ccuda.CUgraphNode*>pGraphNode, <ccuda.CUgraph>graph, <ccuda.CUgraphNode*>pDependencies, numDependencies, <ccuda.CUDA_EXT_SEM_WAIT_NODE_PARAMS*>nodeParams)
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t cudaGraphExternalSemaphoresWaitNodeGetParams(cudaGraphNode_t hNode, cudaExternalSemaphoreWaitNodeParams* params_out) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuGraphExternalSemaphoresWaitNodeGetParams(<ccuda.CUgraphNode>hNode, <ccuda.CUDA_EXT_SEM_WAIT_NODE_PARAMS*>params_out)
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t cudaGraphExternalSemaphoresWaitNodeSetParams(cudaGraphNode_t hNode, const cudaExternalSemaphoreWaitNodeParams* nodeParams) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuGraphExternalSemaphoresWaitNodeSetParams(<ccuda.CUgraphNode>hNode, <ccuda.CUDA_EXT_SEM_WAIT_NODE_PARAMS*>nodeParams)
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t cudaGraphAddMemAllocNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, cudaMemAllocNodeParams* nodeParams) nogil except ?cudaErrorCallRequiresNewerDriver:
    return _cudaGraphAddMemAllocNode(pGraphNode, graph, pDependencies, numDependencies, nodeParams)

cdef cudaError_t cudaGraphMemAllocNodeGetParams(cudaGraphNode_t node, cudaMemAllocNodeParams* params_out) nogil except ?cudaErrorCallRequiresNewerDriver:
    return _cudaGraphMemAllocNodeGetParams(node, params_out)

cdef cudaError_t cudaGraphAddMemFreeNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, void* dptr) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuGraphAddMemFreeNode(<ccuda.CUgraphNode*>pGraphNode, <ccuda.CUgraph>graph, <ccuda.CUgraphNode*>pDependencies, numDependencies, <ccuda.CUdeviceptr>dptr)
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t cudaGraphMemFreeNodeGetParams(cudaGraphNode_t node, void* dptr_out) nogil except ?cudaErrorCallRequiresNewerDriver:
    return _cudaGraphMemFreeNodeGetParams(node, dptr_out)

cdef cudaError_t cudaDeviceGraphMemTrim(int device) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuDeviceGraphMemTrim(<ccuda.CUdevice>device)
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t cudaDeviceGetGraphMemAttribute(int device, cudaGraphMemAttributeType attr, void* value) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuDeviceGetGraphMemAttribute(<ccuda.CUdevice>device, <ccuda.CUgraphMem_attribute>attr, value)
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t cudaDeviceSetGraphMemAttribute(int device, cudaGraphMemAttributeType attr, void* value) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuDeviceSetGraphMemAttribute(<ccuda.CUdevice>device, <ccuda.CUgraphMem_attribute>attr, value)
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t cudaGraphClone(cudaGraph_t* pGraphClone, cudaGraph_t originalGraph) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuGraphClone(<ccuda.CUgraph*>pGraphClone, <ccuda.CUgraph>originalGraph)
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t cudaGraphNodeFindInClone(cudaGraphNode_t* pNode, cudaGraphNode_t originalNode, cudaGraph_t clonedGraph) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuGraphNodeFindInClone(<ccuda.CUgraphNode*>pNode, <ccuda.CUgraphNode>originalNode, <ccuda.CUgraph>clonedGraph)
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t cudaGraphNodeGetType(cudaGraphNode_t node, cudaGraphNodeType* pType) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuGraphNodeGetType(<ccuda.CUgraphNode>node, <ccuda.CUgraphNodeType*>pType)
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t cudaGraphGetNodes(cudaGraph_t graph, cudaGraphNode_t* nodes, size_t* numNodes) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuGraphGetNodes(<ccuda.CUgraph>graph, <ccuda.CUgraphNode*>nodes, numNodes)
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t cudaGraphGetRootNodes(cudaGraph_t graph, cudaGraphNode_t* pRootNodes, size_t* pNumRootNodes) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuGraphGetRootNodes(<ccuda.CUgraph>graph, <ccuda.CUgraphNode*>pRootNodes, pNumRootNodes)
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t cudaGraphGetEdges(cudaGraph_t graph, cudaGraphNode_t* from_, cudaGraphNode_t* to, size_t* numEdges) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuGraphGetEdges(<ccuda.CUgraph>graph, <ccuda.CUgraphNode*>from_, <ccuda.CUgraphNode*>to, numEdges)
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t cudaGraphNodeGetDependencies(cudaGraphNode_t node, cudaGraphNode_t* pDependencies, size_t* pNumDependencies) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuGraphNodeGetDependencies(<ccuda.CUgraphNode>node, <ccuda.CUgraphNode*>pDependencies, pNumDependencies)
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t cudaGraphNodeGetDependentNodes(cudaGraphNode_t node, cudaGraphNode_t* pDependentNodes, size_t* pNumDependentNodes) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuGraphNodeGetDependentNodes(<ccuda.CUgraphNode>node, <ccuda.CUgraphNode*>pDependentNodes, pNumDependentNodes)
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t cudaGraphAddDependencies(cudaGraph_t graph, const cudaGraphNode_t* from_, const cudaGraphNode_t* to, size_t numDependencies) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuGraphAddDependencies(<ccuda.CUgraph>graph, <ccuda.CUgraphNode*>from_, <ccuda.CUgraphNode*>to, numDependencies)
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t cudaGraphRemoveDependencies(cudaGraph_t graph, const cudaGraphNode_t* from_, const cudaGraphNode_t* to, size_t numDependencies) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuGraphRemoveDependencies(<ccuda.CUgraph>graph, <ccuda.CUgraphNode*>from_, <ccuda.CUgraphNode*>to, numDependencies)
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t cudaGraphDestroyNode(cudaGraphNode_t node) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuGraphDestroyNode(<ccuda.CUgraphNode>node)
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t cudaGraphInstantiate(cudaGraphExec_t* pGraphExec, cudaGraph_t graph, cudaGraphNode_t* pErrorNode, char* pLogBuffer, size_t bufferSize) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuGraphInstantiate_v2(<ccuda.CUgraphExec*>pGraphExec, <ccuda.CUgraph>graph, <ccuda.CUgraphNode*>pErrorNode, pLogBuffer, bufferSize)
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t cudaGraphInstantiateWithFlags(cudaGraphExec_t* pGraphExec, cudaGraph_t graph, unsigned long long flags) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuGraphInstantiateWithFlags(<ccuda.CUgraphExec*>pGraphExec, <ccuda.CUgraph>graph, flags)
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t cudaGraphExecKernelNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, const cudaKernelNodeParams* pNodeParams) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
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

    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuGraphExecKernelNodeSetParams(<ccuda.CUgraphExec>hGraphExec, <ccuda.CUgraphNode>node, &_driver_pNodeParams)
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t cudaGraphExecMemcpyNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, const cudaMemcpy3DParms* pNodeParams) nogil except ?cudaErrorCallRequiresNewerDriver:
    return _cudaGraphExecMemcpyNodeSetParams(hGraphExec, node, pNodeParams)

cdef cudaError_t cudaGraphExecMemcpyNodeSetParams1D(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, void* dst, const void* src, size_t count, cudaMemcpyKind kind) nogil except ?cudaErrorCallRequiresNewerDriver:
    return _cudaGraphExecMemcpyNodeSetParams1D(hGraphExec, node, dst, src, count, kind)

cdef cudaError_t cudaGraphExecMemsetNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, const cudaMemsetParams* pNodeParams) nogil except ?cudaErrorCallRequiresNewerDriver:
    return _cudaGraphExecMemsetNodeSetParams(hGraphExec, node, pNodeParams)

cdef cudaError_t cudaGraphExecHostNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, const cudaHostNodeParams* pNodeParams) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuGraphExecHostNodeSetParams(<ccuda.CUgraphExec>hGraphExec, <ccuda.CUgraphNode>node, <ccuda.CUDA_HOST_NODE_PARAMS*>pNodeParams)
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t cudaGraphExecChildGraphNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, cudaGraph_t childGraph) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuGraphExecChildGraphNodeSetParams(<ccuda.CUgraphExec>hGraphExec, <ccuda.CUgraphNode>node, <ccuda.CUgraph>childGraph)
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t cudaGraphExecEventRecordNodeSetEvent(cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, cudaEvent_t event) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuGraphExecEventRecordNodeSetEvent(<ccuda.CUgraphExec>hGraphExec, <ccuda.CUgraphNode>hNode, <ccuda.CUevent>event)
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t cudaGraphExecEventWaitNodeSetEvent(cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, cudaEvent_t event) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuGraphExecEventWaitNodeSetEvent(<ccuda.CUgraphExec>hGraphExec, <ccuda.CUgraphNode>hNode, <ccuda.CUevent>event)
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t cudaGraphExecExternalSemaphoresSignalNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, const cudaExternalSemaphoreSignalNodeParams* nodeParams) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuGraphExecExternalSemaphoresSignalNodeSetParams(<ccuda.CUgraphExec>hGraphExec, <ccuda.CUgraphNode>hNode, <ccuda.CUDA_EXT_SEM_SIGNAL_NODE_PARAMS*>nodeParams)
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t cudaGraphExecExternalSemaphoresWaitNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, const cudaExternalSemaphoreWaitNodeParams* nodeParams) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuGraphExecExternalSemaphoresWaitNodeSetParams(<ccuda.CUgraphExec>hGraphExec, <ccuda.CUgraphNode>hNode, <ccuda.CUDA_EXT_SEM_WAIT_NODE_PARAMS*>nodeParams)
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t cudaGraphNodeSetEnabled(cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, unsigned int isEnabled) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuGraphNodeSetEnabled(<ccuda.CUgraphExec>hGraphExec, <ccuda.CUgraphNode>hNode, isEnabled)
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t cudaGraphNodeGetEnabled(cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, unsigned int* isEnabled) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuGraphNodeGetEnabled(<ccuda.CUgraphExec>hGraphExec, <ccuda.CUgraphNode>hNode, isEnabled)
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t cudaGraphExecUpdate(cudaGraphExec_t hGraphExec, cudaGraph_t hGraph, cudaGraphNode_t* hErrorNode_out, cudaGraphExecUpdateResult* updateResult_out) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuGraphExecUpdate(<ccuda.CUgraphExec>hGraphExec, <ccuda.CUgraph>hGraph, <ccuda.CUgraphNode*>hErrorNode_out, <ccuda.CUgraphExecUpdateResult*>updateResult_out)
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t cudaGraphUpload(cudaGraphExec_t graphExec, cudaStream_t stream) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuGraphUpload(<ccuda.CUgraphExec>graphExec, <ccuda.CUstream>stream)
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t cudaGraphLaunch(cudaGraphExec_t graphExec, cudaStream_t stream) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuGraphLaunch(<ccuda.CUgraphExec>graphExec, <ccuda.CUstream>stream)
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t cudaGraphExecDestroy(cudaGraphExec_t graphExec) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuGraphExecDestroy(<ccuda.CUgraphExec>graphExec)
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t cudaGraphDestroy(cudaGraph_t graph) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuGraphDestroy(<ccuda.CUgraph>graph)
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t cudaGraphDebugDotPrint(cudaGraph_t graph, const char* path, unsigned int flags) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuGraphDebugDotPrint(<ccuda.CUgraph>graph, path, flags)
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t cudaUserObjectCreate(cudaUserObject_t* object_out, void* ptr, cudaHostFn_t destroy, unsigned int initialRefcount, unsigned int flags) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuUserObjectCreate(<ccuda.CUuserObject*>object_out, ptr, <ccuda.CUhostFn>destroy, initialRefcount, flags)
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t cudaUserObjectRetain(cudaUserObject_t object, unsigned int count) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuUserObjectRetain(<ccuda.CUuserObject>object, count)
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t cudaUserObjectRelease(cudaUserObject_t object, unsigned int count) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuUserObjectRelease(<ccuda.CUuserObject>object, count)
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t cudaGraphRetainUserObject(cudaGraph_t graph, cudaUserObject_t object, unsigned int count, unsigned int flags) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuGraphRetainUserObject(<ccuda.CUgraph>graph, <ccuda.CUuserObject>object, count, flags)
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t cudaGraphReleaseUserObject(cudaGraph_t graph, cudaUserObject_t object, unsigned int count) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuGraphReleaseUserObject(<ccuda.CUgraph>graph, <ccuda.CUuserObject>object, count)
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t cudaGetDriverEntryPoint(const char* symbol, void** funcPtr, unsigned long long flags) nogil except ?cudaErrorCallRequiresNewerDriver:
    return _cudaGetDriverEntryPoint(symbol, funcPtr, flags)

cdef cudaError_t cudaGetExportTable(const void** ppExportTable, const cudaUUID_t* pExportTableId) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuGetExportTable(ppExportTable, <ccuda.CUuuid*>pExportTableId)
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaPitchedPtr make_cudaPitchedPtr(void* d, size_t p, size_t xsz, size_t ysz) nogil:
    return _make_cudaPitchedPtr(d, p, xsz, ysz)

cdef cudaPos make_cudaPos(size_t x, size_t y, size_t z) nogil:
    return _make_cudaPos(x, y, z)

cdef cudaExtent make_cudaExtent(size_t w, size_t h, size_t d) nogil:
    return _make_cudaExtent(w, h, d)

cdef cudaError_t cudaProfilerInitialize(const char* configFile, const char* outputFile, cudaOutputMode_t outputMode) nogil except ?cudaErrorCallRequiresNewerDriver:
    return _cudaProfilerInitialize(configFile, outputFile, outputMode)

cdef cudaError_t cudaProfilerStart() nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuProfilerStart()
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t cudaProfilerStop() nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuProfilerStop()
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t cudaVDPAUGetDevice(int* device, VdpDevice vdpDevice, VdpGetProcAddress* vdpGetProcAddress) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuVDPAUGetDevice(<ccuda.CUdevice*>device, vdpDevice, vdpGetProcAddress)
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t cudaVDPAUSetVDPAUDevice(int device, VdpDevice vdpDevice, VdpGetProcAddress* vdpGetProcAddress) nogil except ?cudaErrorCallRequiresNewerDriver:
    return _cudaVDPAUSetVDPAUDevice(device, vdpDevice, vdpGetProcAddress)

cdef cudaError_t cudaGraphicsVDPAURegisterVideoSurface(cudaGraphicsResource** resource, VdpVideoSurface vdpSurface, unsigned int flags) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuGraphicsVDPAURegisterVideoSurface(<ccuda.CUgraphicsResource*>resource, vdpSurface, flags)
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t cudaGraphicsVDPAURegisterOutputSurface(cudaGraphicsResource** resource, VdpOutputSurface vdpSurface, unsigned int flags) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuGraphicsVDPAURegisterOutputSurface(<ccuda.CUgraphicsResource*>resource, vdpSurface, flags)
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t cudaGLGetDevices(unsigned int* pCudaDeviceCount, int* pCudaDevices, unsigned int cudaDeviceCount, cudaGLDeviceList deviceList) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuGLGetDevices_v2(pCudaDeviceCount, <ccuda.CUdevice*>pCudaDevices, cudaDeviceCount, <ccuda.CUGLDeviceList>deviceList)
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t cudaGraphicsGLRegisterImage(cudaGraphicsResource** resource, GLuint image, GLenum target, unsigned int flags) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuGraphicsGLRegisterImage(<ccuda.CUgraphicsResource*>resource, image, target, flags)
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t cudaGraphicsGLRegisterBuffer(cudaGraphicsResource** resource, GLuint buffer, unsigned int flags) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuGraphicsGLRegisterBuffer(<ccuda.CUgraphicsResource*>resource, buffer, flags)
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t cudaGraphicsEGLRegisterImage(cudaGraphicsResource_t* pCudaResource, EGLImageKHR image, unsigned int flags) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuGraphicsEGLRegisterImage(<ccuda.CUgraphicsResource*>pCudaResource, image, flags)
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t cudaEGLStreamConsumerConnect(cudaEglStreamConnection* conn, EGLStreamKHR eglStream) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuEGLStreamConsumerConnect(<ccuda.CUeglStreamConnection*>conn, eglStream)
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t cudaEGLStreamConsumerConnectWithFlags(cudaEglStreamConnection* conn, EGLStreamKHR eglStream, unsigned int flags) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuEGLStreamConsumerConnectWithFlags(<ccuda.CUeglStreamConnection*>conn, eglStream, flags)
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t cudaEGLStreamConsumerDisconnect(cudaEglStreamConnection* conn) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuEGLStreamConsumerDisconnect(<ccuda.CUeglStreamConnection*>conn)
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t cudaEGLStreamConsumerAcquireFrame(cudaEglStreamConnection* conn, cudaGraphicsResource_t* pCudaResource, cudaStream_t* pStream, unsigned int timeout) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuEGLStreamConsumerAcquireFrame(<ccuda.CUeglStreamConnection*>conn, <ccuda.CUgraphicsResource*>pCudaResource, <ccuda.CUstream*>pStream, timeout)
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t cudaEGLStreamConsumerReleaseFrame(cudaEglStreamConnection* conn, cudaGraphicsResource_t pCudaResource, cudaStream_t* pStream) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuEGLStreamConsumerReleaseFrame(<ccuda.CUeglStreamConnection*>conn, <ccuda.CUgraphicsResource>pCudaResource, <ccuda.CUstream*>pStream)
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t cudaEGLStreamProducerConnect(cudaEglStreamConnection* conn, EGLStreamKHR eglStream, EGLint width, EGLint height) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuEGLStreamProducerConnect(<ccuda.CUeglStreamConnection*>conn, eglStream, width, height)
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t cudaEGLStreamProducerDisconnect(cudaEglStreamConnection* conn) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuEGLStreamProducerDisconnect(<ccuda.CUeglStreamConnection*>conn)
    if err != cudaSuccess:
        _setLastError(err)
    return err

cdef cudaError_t cudaEGLStreamProducerPresentFrame(cudaEglStreamConnection* conn, cudaEglFrame eglframe, cudaStream_t* pStream) nogil except ?cudaErrorCallRequiresNewerDriver:
    return _cudaEGLStreamProducerPresentFrame(conn, eglframe, pStream)

cdef cudaError_t cudaEGLStreamProducerReturnFrame(cudaEglStreamConnection* conn, cudaEglFrame* eglframe, cudaStream_t* pStream) nogil except ?cudaErrorCallRequiresNewerDriver:
    return _cudaEGLStreamProducerReturnFrame(conn, eglframe, pStream)

cdef cudaError_t cudaGraphicsResourceGetMappedEglFrame(cudaEglFrame* eglFrame, cudaGraphicsResource_t resource, unsigned int index, unsigned int mipLevel) nogil except ?cudaErrorCallRequiresNewerDriver:
    return _cudaGraphicsResourceGetMappedEglFrame(eglFrame, resource, index, mipLevel)

cdef cudaError_t cudaEventCreateFromEGLSync(cudaEvent_t* phEvent, EGLSyncKHR eglSync, unsigned int flags) nogil except ?cudaErrorCallRequiresNewerDriver:
    cdef cudaError_t err
    err = m_global.lazyInit()
    if err != cudaSuccess:
        return err
    err = <cudaError_t>ccuda._cuEventCreateFromEGLSync(<ccuda.CUevent*>phEvent, eglSync, flags)
    if err != cudaSuccess:
        _setLastError(err)
    return err
