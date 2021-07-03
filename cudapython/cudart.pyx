# Copyright 2021 NVIDIA Corporation.  All rights reserved.
#
# Please refer to the NVIDIA end user license agreement (EULA) associated
# with this source code for terms and conditions that govern your use of
# this software. Any use, reproduction, disclosure, or distribution of
# this software and related documentation outside the terms of the EULA
# is strictly prohibited.
from typing import List, Tuple, Any
from enum import Enum
import cython
import ctypes
from libc.stdlib cimport calloc, free
from libc.string cimport memcpy
from libc.stdint cimport int32_t, uint32_t, int64_t, uint64_t
from libc.stddef cimport wchar_t
from libcpp.vector cimport vector
from cpython.buffer cimport PyObject_CheckBuffer, PyObject_GetBuffer, PyBuffer_Release, PyBUF_SIMPLE, PyBUF_ANY_CONTIGUOUS
import cudapython.cuda

ctypedef unsigned long long signed_char_ptr
ctypedef unsigned long long unsigned_char_ptr
ctypedef unsigned long long char_ptr
ctypedef unsigned long long short_ptr
ctypedef unsigned long long unsigned_short_ptr
ctypedef unsigned long long int_ptr
ctypedef unsigned long long long_int_ptr
ctypedef unsigned long long long_long_int_ptr
ctypedef unsigned long long unsigned_int_ptr
ctypedef unsigned long long unsigned_long_int_ptr
ctypedef unsigned long long unsigned_long_long_int_ptr
ctypedef unsigned long long uint32_t_ptr
ctypedef unsigned long long uint64_t_ptr
ctypedef unsigned long long int32_t_ptr
ctypedef unsigned long long int64_t_ptr
ctypedef unsigned long long unsigned_ptr
ctypedef unsigned long long unsigned_long_long_ptr
ctypedef unsigned long long size_t_ptr
ctypedef unsigned long long float_ptr
ctypedef unsigned long long double_ptr
ctypedef unsigned long long void_ptr

cudaHostAllocDefault = ccudart.cudaHostAllocDefault
cudaHostAllocPortable = ccudart.cudaHostAllocPortable
cudaHostAllocMapped = ccudart.cudaHostAllocMapped
cudaHostAllocWriteCombined = ccudart.cudaHostAllocWriteCombined
cudaHostRegisterDefault = ccudart.cudaHostRegisterDefault
cudaHostRegisterPortable = ccudart.cudaHostRegisterPortable
cudaHostRegisterMapped = ccudart.cudaHostRegisterMapped
cudaHostRegisterIoMemory = ccudart.cudaHostRegisterIoMemory
cudaHostRegisterReadOnly = ccudart.cudaHostRegisterReadOnly
cudaPeerAccessDefault = ccudart.cudaPeerAccessDefault
cudaStreamDefault = ccudart.cudaStreamDefault
cudaStreamNonBlocking = ccudart.cudaStreamNonBlocking
cudaStreamLegacy = ccudart.cudaStreamLegacy
cudaStreamPerThread = ccudart.cudaStreamPerThread
cudaEventDefault = ccudart.cudaEventDefault
cudaEventBlockingSync = ccudart.cudaEventBlockingSync
cudaEventDisableTiming = ccudart.cudaEventDisableTiming
cudaEventInterprocess = ccudart.cudaEventInterprocess
cudaEventRecordDefault = ccudart.cudaEventRecordDefault
cudaEventRecordExternal = ccudart.cudaEventRecordExternal
cudaEventWaitDefault = ccudart.cudaEventWaitDefault
cudaEventWaitExternal = ccudart.cudaEventWaitExternal
cudaDeviceScheduleAuto = ccudart.cudaDeviceScheduleAuto
cudaDeviceScheduleSpin = ccudart.cudaDeviceScheduleSpin
cudaDeviceScheduleYield = ccudart.cudaDeviceScheduleYield
cudaDeviceScheduleBlockingSync = ccudart.cudaDeviceScheduleBlockingSync
cudaDeviceBlockingSync = ccudart.cudaDeviceBlockingSync
cudaDeviceScheduleMask = ccudart.cudaDeviceScheduleMask
cudaDeviceMapHost = ccudart.cudaDeviceMapHost
cudaDeviceLmemResizeToMax = ccudart.cudaDeviceLmemResizeToMax
cudaDeviceMask = ccudart.cudaDeviceMask
cudaArrayDefault = ccudart.cudaArrayDefault
cudaArrayLayered = ccudart.cudaArrayLayered
cudaArraySurfaceLoadStore = ccudart.cudaArraySurfaceLoadStore
cudaArrayCubemap = ccudart.cudaArrayCubemap
cudaArrayTextureGather = ccudart.cudaArrayTextureGather
cudaArrayColorAttachment = ccudart.cudaArrayColorAttachment
cudaArraySparse = ccudart.cudaArraySparse
cudaIpcMemLazyEnablePeerAccess = ccudart.cudaIpcMemLazyEnablePeerAccess
cudaMemAttachGlobal = ccudart.cudaMemAttachGlobal
cudaMemAttachHost = ccudart.cudaMemAttachHost
cudaMemAttachSingle = ccudart.cudaMemAttachSingle
cudaOccupancyDefault = ccudart.cudaOccupancyDefault
cudaOccupancyDisableCachingOverride = ccudart.cudaOccupancyDisableCachingOverride
cudaCpuDeviceId = ccudart.cudaCpuDeviceId
cudaInvalidDeviceId = ccudart.cudaInvalidDeviceId
cudaCooperativeLaunchMultiDeviceNoPreSync = ccudart.cudaCooperativeLaunchMultiDeviceNoPreSync
cudaCooperativeLaunchMultiDeviceNoPostSync = ccudart.cudaCooperativeLaunchMultiDeviceNoPostSync
cudaArraySparsePropertiesSingleMipTail = ccudart.cudaArraySparsePropertiesSingleMipTail
CUDA_IPC_HANDLE_SIZE = ccudart.CUDA_IPC_HANDLE_SIZE
cudaExternalMemoryDedicated = ccudart.cudaExternalMemoryDedicated
cudaExternalSemaphoreSignalSkipNvSciBufMemSync = ccudart.cudaExternalSemaphoreSignalSkipNvSciBufMemSync
cudaExternalSemaphoreWaitSkipNvSciBufMemSync = ccudart.cudaExternalSemaphoreWaitSkipNvSciBufMemSync
cudaNvSciSyncAttrSignal = ccudart.cudaNvSciSyncAttrSignal
cudaNvSciSyncAttrWait = ccudart.cudaNvSciSyncAttrWait
cudaSurfaceType1D = ccudart.cudaSurfaceType1D
cudaSurfaceType2D = ccudart.cudaSurfaceType2D
cudaSurfaceType3D = ccudart.cudaSurfaceType3D
cudaSurfaceTypeCubemap = ccudart.cudaSurfaceTypeCubemap
cudaSurfaceType1DLayered = ccudart.cudaSurfaceType1DLayered
cudaSurfaceType2DLayered = ccudart.cudaSurfaceType2DLayered
cudaSurfaceTypeCubemapLayered = ccudart.cudaSurfaceTypeCubemapLayered
cudaTextureType1D = ccudart.cudaTextureType1D
cudaTextureType2D = ccudart.cudaTextureType2D
cudaTextureType3D = ccudart.cudaTextureType3D
cudaTextureTypeCubemap = ccudart.cudaTextureTypeCubemap
cudaTextureType1DLayered = ccudart.cudaTextureType1DLayered
cudaTextureType2DLayered = ccudart.cudaTextureType2DLayered
cudaTextureTypeCubemapLayered = ccudart.cudaTextureTypeCubemapLayered
CUDART_VERSION = ccudart.CUDART_VERSION

class cudaRoundMode(Enum):
    cudaRoundNearest = ccudart.cudaRoundMode.cudaRoundNearest
    cudaRoundZero = ccudart.cudaRoundMode.cudaRoundZero
    cudaRoundPosInf = ccudart.cudaRoundMode.cudaRoundPosInf
    cudaRoundMinInf = ccudart.cudaRoundMode.cudaRoundMinInf

class cudaError_t(Enum):
    cudaSuccess = ccudart.cudaError.cudaSuccess
    cudaErrorInvalidValue = ccudart.cudaError.cudaErrorInvalidValue
    cudaErrorMemoryAllocation = ccudart.cudaError.cudaErrorMemoryAllocation
    cudaErrorInitializationError = ccudart.cudaError.cudaErrorInitializationError
    cudaErrorCudartUnloading = ccudart.cudaError.cudaErrorCudartUnloading
    cudaErrorProfilerDisabled = ccudart.cudaError.cudaErrorProfilerDisabled
    cudaErrorProfilerNotInitialized = ccudart.cudaError.cudaErrorProfilerNotInitialized
    cudaErrorProfilerAlreadyStarted = ccudart.cudaError.cudaErrorProfilerAlreadyStarted
    cudaErrorProfilerAlreadyStopped = ccudart.cudaError.cudaErrorProfilerAlreadyStopped
    cudaErrorInvalidConfiguration = ccudart.cudaError.cudaErrorInvalidConfiguration
    cudaErrorInvalidPitchValue = ccudart.cudaError.cudaErrorInvalidPitchValue
    cudaErrorInvalidSymbol = ccudart.cudaError.cudaErrorInvalidSymbol
    cudaErrorInvalidHostPointer = ccudart.cudaError.cudaErrorInvalidHostPointer
    cudaErrorInvalidDevicePointer = ccudart.cudaError.cudaErrorInvalidDevicePointer
    cudaErrorInvalidTexture = ccudart.cudaError.cudaErrorInvalidTexture
    cudaErrorInvalidTextureBinding = ccudart.cudaError.cudaErrorInvalidTextureBinding
    cudaErrorInvalidChannelDescriptor = ccudart.cudaError.cudaErrorInvalidChannelDescriptor
    cudaErrorInvalidMemcpyDirection = ccudart.cudaError.cudaErrorInvalidMemcpyDirection
    cudaErrorAddressOfConstant = ccudart.cudaError.cudaErrorAddressOfConstant
    cudaErrorTextureFetchFailed = ccudart.cudaError.cudaErrorTextureFetchFailed
    cudaErrorTextureNotBound = ccudart.cudaError.cudaErrorTextureNotBound
    cudaErrorSynchronizationError = ccudart.cudaError.cudaErrorSynchronizationError
    cudaErrorInvalidFilterSetting = ccudart.cudaError.cudaErrorInvalidFilterSetting
    cudaErrorInvalidNormSetting = ccudart.cudaError.cudaErrorInvalidNormSetting
    cudaErrorMixedDeviceExecution = ccudart.cudaError.cudaErrorMixedDeviceExecution
    cudaErrorNotYetImplemented = ccudart.cudaError.cudaErrorNotYetImplemented
    cudaErrorMemoryValueTooLarge = ccudart.cudaError.cudaErrorMemoryValueTooLarge
    cudaErrorStubLibrary = ccudart.cudaError.cudaErrorStubLibrary
    cudaErrorInsufficientDriver = ccudart.cudaError.cudaErrorInsufficientDriver
    cudaErrorCallRequiresNewerDriver = ccudart.cudaError.cudaErrorCallRequiresNewerDriver
    cudaErrorInvalidSurface = ccudart.cudaError.cudaErrorInvalidSurface
    cudaErrorDuplicateVariableName = ccudart.cudaError.cudaErrorDuplicateVariableName
    cudaErrorDuplicateTextureName = ccudart.cudaError.cudaErrorDuplicateTextureName
    cudaErrorDuplicateSurfaceName = ccudart.cudaError.cudaErrorDuplicateSurfaceName
    cudaErrorDevicesUnavailable = ccudart.cudaError.cudaErrorDevicesUnavailable
    cudaErrorIncompatibleDriverContext = ccudart.cudaError.cudaErrorIncompatibleDriverContext
    cudaErrorMissingConfiguration = ccudart.cudaError.cudaErrorMissingConfiguration
    cudaErrorPriorLaunchFailure = ccudart.cudaError.cudaErrorPriorLaunchFailure
    cudaErrorLaunchMaxDepthExceeded = ccudart.cudaError.cudaErrorLaunchMaxDepthExceeded
    cudaErrorLaunchFileScopedTex = ccudart.cudaError.cudaErrorLaunchFileScopedTex
    cudaErrorLaunchFileScopedSurf = ccudart.cudaError.cudaErrorLaunchFileScopedSurf
    cudaErrorSyncDepthExceeded = ccudart.cudaError.cudaErrorSyncDepthExceeded
    cudaErrorLaunchPendingCountExceeded = ccudart.cudaError.cudaErrorLaunchPendingCountExceeded
    cudaErrorInvalidDeviceFunction = ccudart.cudaError.cudaErrorInvalidDeviceFunction
    cudaErrorNoDevice = ccudart.cudaError.cudaErrorNoDevice
    cudaErrorInvalidDevice = ccudart.cudaError.cudaErrorInvalidDevice
    cudaErrorDeviceNotLicensed = ccudart.cudaError.cudaErrorDeviceNotLicensed
    cudaErrorSoftwareValidityNotEstablished = ccudart.cudaError.cudaErrorSoftwareValidityNotEstablished
    cudaErrorStartupFailure = ccudart.cudaError.cudaErrorStartupFailure
    cudaErrorInvalidKernelImage = ccudart.cudaError.cudaErrorInvalidKernelImage
    cudaErrorDeviceUninitialized = ccudart.cudaError.cudaErrorDeviceUninitialized
    cudaErrorMapBufferObjectFailed = ccudart.cudaError.cudaErrorMapBufferObjectFailed
    cudaErrorUnmapBufferObjectFailed = ccudart.cudaError.cudaErrorUnmapBufferObjectFailed
    cudaErrorArrayIsMapped = ccudart.cudaError.cudaErrorArrayIsMapped
    cudaErrorAlreadyMapped = ccudart.cudaError.cudaErrorAlreadyMapped
    cudaErrorNoKernelImageForDevice = ccudart.cudaError.cudaErrorNoKernelImageForDevice
    cudaErrorAlreadyAcquired = ccudart.cudaError.cudaErrorAlreadyAcquired
    cudaErrorNotMapped = ccudart.cudaError.cudaErrorNotMapped
    cudaErrorNotMappedAsArray = ccudart.cudaError.cudaErrorNotMappedAsArray
    cudaErrorNotMappedAsPointer = ccudart.cudaError.cudaErrorNotMappedAsPointer
    cudaErrorECCUncorrectable = ccudart.cudaError.cudaErrorECCUncorrectable
    cudaErrorUnsupportedLimit = ccudart.cudaError.cudaErrorUnsupportedLimit
    cudaErrorDeviceAlreadyInUse = ccudart.cudaError.cudaErrorDeviceAlreadyInUse
    cudaErrorPeerAccessUnsupported = ccudart.cudaError.cudaErrorPeerAccessUnsupported
    cudaErrorInvalidPtx = ccudart.cudaError.cudaErrorInvalidPtx
    cudaErrorInvalidGraphicsContext = ccudart.cudaError.cudaErrorInvalidGraphicsContext
    cudaErrorNvlinkUncorrectable = ccudart.cudaError.cudaErrorNvlinkUncorrectable
    cudaErrorJitCompilerNotFound = ccudart.cudaError.cudaErrorJitCompilerNotFound
    cudaErrorUnsupportedPtxVersion = ccudart.cudaError.cudaErrorUnsupportedPtxVersion
    cudaErrorJitCompilationDisabled = ccudart.cudaError.cudaErrorJitCompilationDisabled
    cudaErrorUnsupportedExecAffinity = ccudart.cudaError.cudaErrorUnsupportedExecAffinity
    cudaErrorInvalidSource = ccudart.cudaError.cudaErrorInvalidSource
    cudaErrorFileNotFound = ccudart.cudaError.cudaErrorFileNotFound
    cudaErrorSharedObjectSymbolNotFound = ccudart.cudaError.cudaErrorSharedObjectSymbolNotFound
    cudaErrorSharedObjectInitFailed = ccudart.cudaError.cudaErrorSharedObjectInitFailed
    cudaErrorOperatingSystem = ccudart.cudaError.cudaErrorOperatingSystem
    cudaErrorInvalidResourceHandle = ccudart.cudaError.cudaErrorInvalidResourceHandle
    cudaErrorIllegalState = ccudart.cudaError.cudaErrorIllegalState
    cudaErrorSymbolNotFound = ccudart.cudaError.cudaErrorSymbolNotFound
    cudaErrorNotReady = ccudart.cudaError.cudaErrorNotReady
    cudaErrorIllegalAddress = ccudart.cudaError.cudaErrorIllegalAddress
    cudaErrorLaunchOutOfResources = ccudart.cudaError.cudaErrorLaunchOutOfResources
    cudaErrorLaunchTimeout = ccudart.cudaError.cudaErrorLaunchTimeout
    cudaErrorLaunchIncompatibleTexturing = ccudart.cudaError.cudaErrorLaunchIncompatibleTexturing
    cudaErrorPeerAccessAlreadyEnabled = ccudart.cudaError.cudaErrorPeerAccessAlreadyEnabled
    cudaErrorPeerAccessNotEnabled = ccudart.cudaError.cudaErrorPeerAccessNotEnabled
    cudaErrorSetOnActiveProcess = ccudart.cudaError.cudaErrorSetOnActiveProcess
    cudaErrorContextIsDestroyed = ccudart.cudaError.cudaErrorContextIsDestroyed
    cudaErrorAssert = ccudart.cudaError.cudaErrorAssert
    cudaErrorTooManyPeers = ccudart.cudaError.cudaErrorTooManyPeers
    cudaErrorHostMemoryAlreadyRegistered = ccudart.cudaError.cudaErrorHostMemoryAlreadyRegistered
    cudaErrorHostMemoryNotRegistered = ccudart.cudaError.cudaErrorHostMemoryNotRegistered
    cudaErrorHardwareStackError = ccudart.cudaError.cudaErrorHardwareStackError
    cudaErrorIllegalInstruction = ccudart.cudaError.cudaErrorIllegalInstruction
    cudaErrorMisalignedAddress = ccudart.cudaError.cudaErrorMisalignedAddress
    cudaErrorInvalidAddressSpace = ccudart.cudaError.cudaErrorInvalidAddressSpace
    cudaErrorInvalidPc = ccudart.cudaError.cudaErrorInvalidPc
    cudaErrorLaunchFailure = ccudart.cudaError.cudaErrorLaunchFailure
    cudaErrorCooperativeLaunchTooLarge = ccudart.cudaError.cudaErrorCooperativeLaunchTooLarge
    cudaErrorNotPermitted = ccudart.cudaError.cudaErrorNotPermitted
    cudaErrorNotSupported = ccudart.cudaError.cudaErrorNotSupported
    cudaErrorSystemNotReady = ccudart.cudaError.cudaErrorSystemNotReady
    cudaErrorSystemDriverMismatch = ccudart.cudaError.cudaErrorSystemDriverMismatch
    cudaErrorCompatNotSupportedOnDevice = ccudart.cudaError.cudaErrorCompatNotSupportedOnDevice
    cudaErrorMpsConnectionFailed = ccudart.cudaError.cudaErrorMpsConnectionFailed
    cudaErrorMpsRpcFailure = ccudart.cudaError.cudaErrorMpsRpcFailure
    cudaErrorMpsServerNotReady = ccudart.cudaError.cudaErrorMpsServerNotReady
    cudaErrorMpsMaxClientsReached = ccudart.cudaError.cudaErrorMpsMaxClientsReached
    cudaErrorMpsMaxConnectionsReached = ccudart.cudaError.cudaErrorMpsMaxConnectionsReached
    cudaErrorStreamCaptureUnsupported = ccudart.cudaError.cudaErrorStreamCaptureUnsupported
    cudaErrorStreamCaptureInvalidated = ccudart.cudaError.cudaErrorStreamCaptureInvalidated
    cudaErrorStreamCaptureMerge = ccudart.cudaError.cudaErrorStreamCaptureMerge
    cudaErrorStreamCaptureUnmatched = ccudart.cudaError.cudaErrorStreamCaptureUnmatched
    cudaErrorStreamCaptureUnjoined = ccudart.cudaError.cudaErrorStreamCaptureUnjoined
    cudaErrorStreamCaptureIsolation = ccudart.cudaError.cudaErrorStreamCaptureIsolation
    cudaErrorStreamCaptureImplicit = ccudart.cudaError.cudaErrorStreamCaptureImplicit
    cudaErrorCapturedEvent = ccudart.cudaError.cudaErrorCapturedEvent
    cudaErrorStreamCaptureWrongThread = ccudart.cudaError.cudaErrorStreamCaptureWrongThread
    cudaErrorTimeout = ccudart.cudaError.cudaErrorTimeout
    cudaErrorGraphExecUpdateFailure = ccudart.cudaError.cudaErrorGraphExecUpdateFailure
    cudaErrorUnknown = ccudart.cudaError.cudaErrorUnknown
    cudaErrorApiFailureBase = ccudart.cudaError.cudaErrorApiFailureBase

class cudaChannelFormatKind(Enum):
    cudaChannelFormatKindSigned = ccudart.cudaChannelFormatKind.cudaChannelFormatKindSigned
    cudaChannelFormatKindUnsigned = ccudart.cudaChannelFormatKind.cudaChannelFormatKindUnsigned
    cudaChannelFormatKindFloat = ccudart.cudaChannelFormatKind.cudaChannelFormatKindFloat
    cudaChannelFormatKindNone = ccudart.cudaChannelFormatKind.cudaChannelFormatKindNone
    cudaChannelFormatKindNV12 = ccudart.cudaChannelFormatKind.cudaChannelFormatKindNV12

class cudaMemoryType(Enum):
    cudaMemoryTypeUnregistered = ccudart.cudaMemoryType.cudaMemoryTypeUnregistered
    cudaMemoryTypeHost = ccudart.cudaMemoryType.cudaMemoryTypeHost
    cudaMemoryTypeDevice = ccudart.cudaMemoryType.cudaMemoryTypeDevice
    cudaMemoryTypeManaged = ccudart.cudaMemoryType.cudaMemoryTypeManaged

class cudaMemcpyKind(Enum):
    cudaMemcpyHostToHost = ccudart.cudaMemcpyKind.cudaMemcpyHostToHost
    cudaMemcpyHostToDevice = ccudart.cudaMemcpyKind.cudaMemcpyHostToDevice
    cudaMemcpyDeviceToHost = ccudart.cudaMemcpyKind.cudaMemcpyDeviceToHost
    cudaMemcpyDeviceToDevice = ccudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice
    cudaMemcpyDefault = ccudart.cudaMemcpyKind.cudaMemcpyDefault

class cudaAccessProperty(Enum):
    cudaAccessPropertyNormal = ccudart.cudaAccessProperty.cudaAccessPropertyNormal
    cudaAccessPropertyStreaming = ccudart.cudaAccessProperty.cudaAccessPropertyStreaming
    cudaAccessPropertyPersisting = ccudart.cudaAccessProperty.cudaAccessPropertyPersisting

class cudaStreamCaptureStatus(Enum):
    cudaStreamCaptureStatusNone = ccudart.cudaStreamCaptureStatus.cudaStreamCaptureStatusNone
    cudaStreamCaptureStatusActive = ccudart.cudaStreamCaptureStatus.cudaStreamCaptureStatusActive
    cudaStreamCaptureStatusInvalidated = ccudart.cudaStreamCaptureStatus.cudaStreamCaptureStatusInvalidated

class cudaStreamCaptureMode(Enum):
    cudaStreamCaptureModeGlobal = ccudart.cudaStreamCaptureMode.cudaStreamCaptureModeGlobal
    cudaStreamCaptureModeThreadLocal = ccudart.cudaStreamCaptureMode.cudaStreamCaptureModeThreadLocal
    cudaStreamCaptureModeRelaxed = ccudart.cudaStreamCaptureMode.cudaStreamCaptureModeRelaxed

class cudaSynchronizationPolicy(Enum):
    cudaSyncPolicyAuto = ccudart.cudaSynchronizationPolicy.cudaSyncPolicyAuto
    cudaSyncPolicySpin = ccudart.cudaSynchronizationPolicy.cudaSyncPolicySpin
    cudaSyncPolicyYield = ccudart.cudaSynchronizationPolicy.cudaSyncPolicyYield
    cudaSyncPolicyBlockingSync = ccudart.cudaSynchronizationPolicy.cudaSyncPolicyBlockingSync

class cudaStreamAttrID(Enum):
    cudaStreamAttributeAccessPolicyWindow = ccudart.cudaStreamAttrID.cudaStreamAttributeAccessPolicyWindow
    cudaStreamAttributeSynchronizationPolicy = ccudart.cudaStreamAttrID.cudaStreamAttributeSynchronizationPolicy

class cudaStreamUpdateCaptureDependenciesFlags(Enum):
    cudaStreamAddCaptureDependencies = ccudart.cudaStreamUpdateCaptureDependenciesFlags.cudaStreamAddCaptureDependencies
    cudaStreamSetCaptureDependencies = ccudart.cudaStreamUpdateCaptureDependenciesFlags.cudaStreamSetCaptureDependencies

class cudaUserObjectFlags(Enum):
    cudaUserObjectNoDestructorSync = ccudart.cudaUserObjectFlags.cudaUserObjectNoDestructorSync

class cudaUserObjectRetainFlags(Enum):
    cudaGraphUserObjectMove = ccudart.cudaUserObjectRetainFlags.cudaGraphUserObjectMove

class cudaGraphicsRegisterFlags(Enum):
    cudaGraphicsRegisterFlagsNone = ccudart.cudaGraphicsRegisterFlags.cudaGraphicsRegisterFlagsNone
    cudaGraphicsRegisterFlagsReadOnly = ccudart.cudaGraphicsRegisterFlags.cudaGraphicsRegisterFlagsReadOnly
    cudaGraphicsRegisterFlagsWriteDiscard = ccudart.cudaGraphicsRegisterFlags.cudaGraphicsRegisterFlagsWriteDiscard
    cudaGraphicsRegisterFlagsSurfaceLoadStore = ccudart.cudaGraphicsRegisterFlags.cudaGraphicsRegisterFlagsSurfaceLoadStore
    cudaGraphicsRegisterFlagsTextureGather = ccudart.cudaGraphicsRegisterFlags.cudaGraphicsRegisterFlagsTextureGather

class cudaGraphicsMapFlags(Enum):
    cudaGraphicsMapFlagsNone = ccudart.cudaGraphicsMapFlags.cudaGraphicsMapFlagsNone
    cudaGraphicsMapFlagsReadOnly = ccudart.cudaGraphicsMapFlags.cudaGraphicsMapFlagsReadOnly
    cudaGraphicsMapFlagsWriteDiscard = ccudart.cudaGraphicsMapFlags.cudaGraphicsMapFlagsWriteDiscard

class cudaGraphicsCubeFace(Enum):
    cudaGraphicsCubeFacePositiveX = ccudart.cudaGraphicsCubeFace.cudaGraphicsCubeFacePositiveX
    cudaGraphicsCubeFaceNegativeX = ccudart.cudaGraphicsCubeFace.cudaGraphicsCubeFaceNegativeX
    cudaGraphicsCubeFacePositiveY = ccudart.cudaGraphicsCubeFace.cudaGraphicsCubeFacePositiveY
    cudaGraphicsCubeFaceNegativeY = ccudart.cudaGraphicsCubeFace.cudaGraphicsCubeFaceNegativeY
    cudaGraphicsCubeFacePositiveZ = ccudart.cudaGraphicsCubeFace.cudaGraphicsCubeFacePositiveZ
    cudaGraphicsCubeFaceNegativeZ = ccudart.cudaGraphicsCubeFace.cudaGraphicsCubeFaceNegativeZ

class cudaKernelNodeAttrID(Enum):
    cudaKernelNodeAttributeAccessPolicyWindow = ccudart.cudaKernelNodeAttrID.cudaKernelNodeAttributeAccessPolicyWindow
    cudaKernelNodeAttributeCooperative = ccudart.cudaKernelNodeAttrID.cudaKernelNodeAttributeCooperative

class cudaResourceType(Enum):
    cudaResourceTypeArray = ccudart.cudaResourceType.cudaResourceTypeArray
    cudaResourceTypeMipmappedArray = ccudart.cudaResourceType.cudaResourceTypeMipmappedArray
    cudaResourceTypeLinear = ccudart.cudaResourceType.cudaResourceTypeLinear
    cudaResourceTypePitch2D = ccudart.cudaResourceType.cudaResourceTypePitch2D

class cudaResourceViewFormat(Enum):
    cudaResViewFormatNone = ccudart.cudaResourceViewFormat.cudaResViewFormatNone
    cudaResViewFormatUnsignedChar1 = ccudart.cudaResourceViewFormat.cudaResViewFormatUnsignedChar1
    cudaResViewFormatUnsignedChar2 = ccudart.cudaResourceViewFormat.cudaResViewFormatUnsignedChar2
    cudaResViewFormatUnsignedChar4 = ccudart.cudaResourceViewFormat.cudaResViewFormatUnsignedChar4
    cudaResViewFormatSignedChar1 = ccudart.cudaResourceViewFormat.cudaResViewFormatSignedChar1
    cudaResViewFormatSignedChar2 = ccudart.cudaResourceViewFormat.cudaResViewFormatSignedChar2
    cudaResViewFormatSignedChar4 = ccudart.cudaResourceViewFormat.cudaResViewFormatSignedChar4
    cudaResViewFormatUnsignedShort1 = ccudart.cudaResourceViewFormat.cudaResViewFormatUnsignedShort1
    cudaResViewFormatUnsignedShort2 = ccudart.cudaResourceViewFormat.cudaResViewFormatUnsignedShort2
    cudaResViewFormatUnsignedShort4 = ccudart.cudaResourceViewFormat.cudaResViewFormatUnsignedShort4
    cudaResViewFormatSignedShort1 = ccudart.cudaResourceViewFormat.cudaResViewFormatSignedShort1
    cudaResViewFormatSignedShort2 = ccudart.cudaResourceViewFormat.cudaResViewFormatSignedShort2
    cudaResViewFormatSignedShort4 = ccudart.cudaResourceViewFormat.cudaResViewFormatSignedShort4
    cudaResViewFormatUnsignedInt1 = ccudart.cudaResourceViewFormat.cudaResViewFormatUnsignedInt1
    cudaResViewFormatUnsignedInt2 = ccudart.cudaResourceViewFormat.cudaResViewFormatUnsignedInt2
    cudaResViewFormatUnsignedInt4 = ccudart.cudaResourceViewFormat.cudaResViewFormatUnsignedInt4
    cudaResViewFormatSignedInt1 = ccudart.cudaResourceViewFormat.cudaResViewFormatSignedInt1
    cudaResViewFormatSignedInt2 = ccudart.cudaResourceViewFormat.cudaResViewFormatSignedInt2
    cudaResViewFormatSignedInt4 = ccudart.cudaResourceViewFormat.cudaResViewFormatSignedInt4
    cudaResViewFormatHalf1 = ccudart.cudaResourceViewFormat.cudaResViewFormatHalf1
    cudaResViewFormatHalf2 = ccudart.cudaResourceViewFormat.cudaResViewFormatHalf2
    cudaResViewFormatHalf4 = ccudart.cudaResourceViewFormat.cudaResViewFormatHalf4
    cudaResViewFormatFloat1 = ccudart.cudaResourceViewFormat.cudaResViewFormatFloat1
    cudaResViewFormatFloat2 = ccudart.cudaResourceViewFormat.cudaResViewFormatFloat2
    cudaResViewFormatFloat4 = ccudart.cudaResourceViewFormat.cudaResViewFormatFloat4
    cudaResViewFormatUnsignedBlockCompressed1 = ccudart.cudaResourceViewFormat.cudaResViewFormatUnsignedBlockCompressed1
    cudaResViewFormatUnsignedBlockCompressed2 = ccudart.cudaResourceViewFormat.cudaResViewFormatUnsignedBlockCompressed2
    cudaResViewFormatUnsignedBlockCompressed3 = ccudart.cudaResourceViewFormat.cudaResViewFormatUnsignedBlockCompressed3
    cudaResViewFormatUnsignedBlockCompressed4 = ccudart.cudaResourceViewFormat.cudaResViewFormatUnsignedBlockCompressed4
    cudaResViewFormatSignedBlockCompressed4 = ccudart.cudaResourceViewFormat.cudaResViewFormatSignedBlockCompressed4
    cudaResViewFormatUnsignedBlockCompressed5 = ccudart.cudaResourceViewFormat.cudaResViewFormatUnsignedBlockCompressed5
    cudaResViewFormatSignedBlockCompressed5 = ccudart.cudaResourceViewFormat.cudaResViewFormatSignedBlockCompressed5
    cudaResViewFormatUnsignedBlockCompressed6H = ccudart.cudaResourceViewFormat.cudaResViewFormatUnsignedBlockCompressed6H
    cudaResViewFormatSignedBlockCompressed6H = ccudart.cudaResourceViewFormat.cudaResViewFormatSignedBlockCompressed6H
    cudaResViewFormatUnsignedBlockCompressed7 = ccudart.cudaResourceViewFormat.cudaResViewFormatUnsignedBlockCompressed7

class cudaFuncAttribute(Enum):
    cudaFuncAttributeMaxDynamicSharedMemorySize = ccudart.cudaFuncAttribute.cudaFuncAttributeMaxDynamicSharedMemorySize
    cudaFuncAttributePreferredSharedMemoryCarveout = ccudart.cudaFuncAttribute.cudaFuncAttributePreferredSharedMemoryCarveout
    cudaFuncAttributeMax = ccudart.cudaFuncAttribute.cudaFuncAttributeMax

class cudaFuncCache(Enum):
    cudaFuncCachePreferNone = ccudart.cudaFuncCache.cudaFuncCachePreferNone
    cudaFuncCachePreferShared = ccudart.cudaFuncCache.cudaFuncCachePreferShared
    cudaFuncCachePreferL1 = ccudart.cudaFuncCache.cudaFuncCachePreferL1
    cudaFuncCachePreferEqual = ccudart.cudaFuncCache.cudaFuncCachePreferEqual

class cudaSharedMemConfig(Enum):
    cudaSharedMemBankSizeDefault = ccudart.cudaSharedMemConfig.cudaSharedMemBankSizeDefault
    cudaSharedMemBankSizeFourByte = ccudart.cudaSharedMemConfig.cudaSharedMemBankSizeFourByte
    cudaSharedMemBankSizeEightByte = ccudart.cudaSharedMemConfig.cudaSharedMemBankSizeEightByte

class cudaSharedCarveout(Enum):
    cudaSharedmemCarveoutDefault = ccudart.cudaSharedCarveout.cudaSharedmemCarveoutDefault
    cudaSharedmemCarveoutMaxShared = ccudart.cudaSharedCarveout.cudaSharedmemCarveoutMaxShared
    cudaSharedmemCarveoutMaxL1 = ccudart.cudaSharedCarveout.cudaSharedmemCarveoutMaxL1

class cudaComputeMode(Enum):
    cudaComputeModeDefault = ccudart.cudaComputeMode.cudaComputeModeDefault
    cudaComputeModeExclusive = ccudart.cudaComputeMode.cudaComputeModeExclusive
    cudaComputeModeProhibited = ccudart.cudaComputeMode.cudaComputeModeProhibited
    cudaComputeModeExclusiveProcess = ccudart.cudaComputeMode.cudaComputeModeExclusiveProcess

class cudaLimit(Enum):
    cudaLimitStackSize = ccudart.cudaLimit.cudaLimitStackSize
    cudaLimitPrintfFifoSize = ccudart.cudaLimit.cudaLimitPrintfFifoSize
    cudaLimitMallocHeapSize = ccudart.cudaLimit.cudaLimitMallocHeapSize
    cudaLimitDevRuntimeSyncDepth = ccudart.cudaLimit.cudaLimitDevRuntimeSyncDepth
    cudaLimitDevRuntimePendingLaunchCount = ccudart.cudaLimit.cudaLimitDevRuntimePendingLaunchCount
    cudaLimitMaxL2FetchGranularity = ccudart.cudaLimit.cudaLimitMaxL2FetchGranularity
    cudaLimitPersistingL2CacheSize = ccudart.cudaLimit.cudaLimitPersistingL2CacheSize

class cudaMemoryAdvise(Enum):
    cudaMemAdviseSetReadMostly = ccudart.cudaMemoryAdvise.cudaMemAdviseSetReadMostly
    cudaMemAdviseUnsetReadMostly = ccudart.cudaMemoryAdvise.cudaMemAdviseUnsetReadMostly
    cudaMemAdviseSetPreferredLocation = ccudart.cudaMemoryAdvise.cudaMemAdviseSetPreferredLocation
    cudaMemAdviseUnsetPreferredLocation = ccudart.cudaMemoryAdvise.cudaMemAdviseUnsetPreferredLocation
    cudaMemAdviseSetAccessedBy = ccudart.cudaMemoryAdvise.cudaMemAdviseSetAccessedBy
    cudaMemAdviseUnsetAccessedBy = ccudart.cudaMemoryAdvise.cudaMemAdviseUnsetAccessedBy

class cudaMemRangeAttribute(Enum):
    cudaMemRangeAttributeReadMostly = ccudart.cudaMemRangeAttribute.cudaMemRangeAttributeReadMostly
    cudaMemRangeAttributePreferredLocation = ccudart.cudaMemRangeAttribute.cudaMemRangeAttributePreferredLocation
    cudaMemRangeAttributeAccessedBy = ccudart.cudaMemRangeAttribute.cudaMemRangeAttributeAccessedBy
    cudaMemRangeAttributeLastPrefetchLocation = ccudart.cudaMemRangeAttribute.cudaMemRangeAttributeLastPrefetchLocation

class cudaOutputMode_t(Enum):
    cudaKeyValuePair = ccudart.cudaOutputMode.cudaKeyValuePair
    cudaCSV = ccudart.cudaOutputMode.cudaCSV

class cudaFlushGPUDirectRDMAWritesOptions(Enum):
    cudaFlushGPUDirectRDMAWritesOptionHost = ccudart.cudaFlushGPUDirectRDMAWritesOptions.cudaFlushGPUDirectRDMAWritesOptionHost
    cudaFlushGPUDirectRDMAWritesOptionMemOps = ccudart.cudaFlushGPUDirectRDMAWritesOptions.cudaFlushGPUDirectRDMAWritesOptionMemOps

class cudaGPUDirectRDMAWritesOrdering(Enum):
    cudaGPUDirectRDMAWritesOrderingNone = ccudart.cudaGPUDirectRDMAWritesOrdering.cudaGPUDirectRDMAWritesOrderingNone
    cudaGPUDirectRDMAWritesOrderingOwner = ccudart.cudaGPUDirectRDMAWritesOrdering.cudaGPUDirectRDMAWritesOrderingOwner
    cudaGPUDirectRDMAWritesOrderingAllDevices = ccudart.cudaGPUDirectRDMAWritesOrdering.cudaGPUDirectRDMAWritesOrderingAllDevices

class cudaFlushGPUDirectRDMAWritesScope(Enum):
    cudaFlushGPUDirectRDMAWritesToOwner = ccudart.cudaFlushGPUDirectRDMAWritesScope.cudaFlushGPUDirectRDMAWritesToOwner
    cudaFlushGPUDirectRDMAWritesToAllDevices = ccudart.cudaFlushGPUDirectRDMAWritesScope.cudaFlushGPUDirectRDMAWritesToAllDevices

class cudaFlushGPUDirectRDMAWritesTarget(Enum):
    cudaFlushGPUDirectRDMAWritesTargetCurrentDevice = ccudart.cudaFlushGPUDirectRDMAWritesTarget.cudaFlushGPUDirectRDMAWritesTargetCurrentDevice

class cudaDeviceAttr(Enum):
    cudaDevAttrMaxThreadsPerBlock = ccudart.cudaDeviceAttr.cudaDevAttrMaxThreadsPerBlock
    cudaDevAttrMaxBlockDimX = ccudart.cudaDeviceAttr.cudaDevAttrMaxBlockDimX
    cudaDevAttrMaxBlockDimY = ccudart.cudaDeviceAttr.cudaDevAttrMaxBlockDimY
    cudaDevAttrMaxBlockDimZ = ccudart.cudaDeviceAttr.cudaDevAttrMaxBlockDimZ
    cudaDevAttrMaxGridDimX = ccudart.cudaDeviceAttr.cudaDevAttrMaxGridDimX
    cudaDevAttrMaxGridDimY = ccudart.cudaDeviceAttr.cudaDevAttrMaxGridDimY
    cudaDevAttrMaxGridDimZ = ccudart.cudaDeviceAttr.cudaDevAttrMaxGridDimZ
    cudaDevAttrMaxSharedMemoryPerBlock = ccudart.cudaDeviceAttr.cudaDevAttrMaxSharedMemoryPerBlock
    cudaDevAttrTotalConstantMemory = ccudart.cudaDeviceAttr.cudaDevAttrTotalConstantMemory
    cudaDevAttrWarpSize = ccudart.cudaDeviceAttr.cudaDevAttrWarpSize
    cudaDevAttrMaxPitch = ccudart.cudaDeviceAttr.cudaDevAttrMaxPitch
    cudaDevAttrMaxRegistersPerBlock = ccudart.cudaDeviceAttr.cudaDevAttrMaxRegistersPerBlock
    cudaDevAttrClockRate = ccudart.cudaDeviceAttr.cudaDevAttrClockRate
    cudaDevAttrTextureAlignment = ccudart.cudaDeviceAttr.cudaDevAttrTextureAlignment
    cudaDevAttrGpuOverlap = ccudart.cudaDeviceAttr.cudaDevAttrGpuOverlap
    cudaDevAttrMultiProcessorCount = ccudart.cudaDeviceAttr.cudaDevAttrMultiProcessorCount
    cudaDevAttrKernelExecTimeout = ccudart.cudaDeviceAttr.cudaDevAttrKernelExecTimeout
    cudaDevAttrIntegrated = ccudart.cudaDeviceAttr.cudaDevAttrIntegrated
    cudaDevAttrCanMapHostMemory = ccudart.cudaDeviceAttr.cudaDevAttrCanMapHostMemory
    cudaDevAttrComputeMode = ccudart.cudaDeviceAttr.cudaDevAttrComputeMode
    cudaDevAttrMaxTexture1DWidth = ccudart.cudaDeviceAttr.cudaDevAttrMaxTexture1DWidth
    cudaDevAttrMaxTexture2DWidth = ccudart.cudaDeviceAttr.cudaDevAttrMaxTexture2DWidth
    cudaDevAttrMaxTexture2DHeight = ccudart.cudaDeviceAttr.cudaDevAttrMaxTexture2DHeight
    cudaDevAttrMaxTexture3DWidth = ccudart.cudaDeviceAttr.cudaDevAttrMaxTexture3DWidth
    cudaDevAttrMaxTexture3DHeight = ccudart.cudaDeviceAttr.cudaDevAttrMaxTexture3DHeight
    cudaDevAttrMaxTexture3DDepth = ccudart.cudaDeviceAttr.cudaDevAttrMaxTexture3DDepth
    cudaDevAttrMaxTexture2DLayeredWidth = ccudart.cudaDeviceAttr.cudaDevAttrMaxTexture2DLayeredWidth
    cudaDevAttrMaxTexture2DLayeredHeight = ccudart.cudaDeviceAttr.cudaDevAttrMaxTexture2DLayeredHeight
    cudaDevAttrMaxTexture2DLayeredLayers = ccudart.cudaDeviceAttr.cudaDevAttrMaxTexture2DLayeredLayers
    cudaDevAttrSurfaceAlignment = ccudart.cudaDeviceAttr.cudaDevAttrSurfaceAlignment
    cudaDevAttrConcurrentKernels = ccudart.cudaDeviceAttr.cudaDevAttrConcurrentKernels
    cudaDevAttrEccEnabled = ccudart.cudaDeviceAttr.cudaDevAttrEccEnabled
    cudaDevAttrPciBusId = ccudart.cudaDeviceAttr.cudaDevAttrPciBusId
    cudaDevAttrPciDeviceId = ccudart.cudaDeviceAttr.cudaDevAttrPciDeviceId
    cudaDevAttrTccDriver = ccudart.cudaDeviceAttr.cudaDevAttrTccDriver
    cudaDevAttrMemoryClockRate = ccudart.cudaDeviceAttr.cudaDevAttrMemoryClockRate
    cudaDevAttrGlobalMemoryBusWidth = ccudart.cudaDeviceAttr.cudaDevAttrGlobalMemoryBusWidth
    cudaDevAttrL2CacheSize = ccudart.cudaDeviceAttr.cudaDevAttrL2CacheSize
    cudaDevAttrMaxThreadsPerMultiProcessor = ccudart.cudaDeviceAttr.cudaDevAttrMaxThreadsPerMultiProcessor
    cudaDevAttrAsyncEngineCount = ccudart.cudaDeviceAttr.cudaDevAttrAsyncEngineCount
    cudaDevAttrUnifiedAddressing = ccudart.cudaDeviceAttr.cudaDevAttrUnifiedAddressing
    cudaDevAttrMaxTexture1DLayeredWidth = ccudart.cudaDeviceAttr.cudaDevAttrMaxTexture1DLayeredWidth
    cudaDevAttrMaxTexture1DLayeredLayers = ccudart.cudaDeviceAttr.cudaDevAttrMaxTexture1DLayeredLayers
    cudaDevAttrMaxTexture2DGatherWidth = ccudart.cudaDeviceAttr.cudaDevAttrMaxTexture2DGatherWidth
    cudaDevAttrMaxTexture2DGatherHeight = ccudart.cudaDeviceAttr.cudaDevAttrMaxTexture2DGatherHeight
    cudaDevAttrMaxTexture3DWidthAlt = ccudart.cudaDeviceAttr.cudaDevAttrMaxTexture3DWidthAlt
    cudaDevAttrMaxTexture3DHeightAlt = ccudart.cudaDeviceAttr.cudaDevAttrMaxTexture3DHeightAlt
    cudaDevAttrMaxTexture3DDepthAlt = ccudart.cudaDeviceAttr.cudaDevAttrMaxTexture3DDepthAlt
    cudaDevAttrPciDomainId = ccudart.cudaDeviceAttr.cudaDevAttrPciDomainId
    cudaDevAttrTexturePitchAlignment = ccudart.cudaDeviceAttr.cudaDevAttrTexturePitchAlignment
    cudaDevAttrMaxTextureCubemapWidth = ccudart.cudaDeviceAttr.cudaDevAttrMaxTextureCubemapWidth
    cudaDevAttrMaxTextureCubemapLayeredWidth = ccudart.cudaDeviceAttr.cudaDevAttrMaxTextureCubemapLayeredWidth
    cudaDevAttrMaxTextureCubemapLayeredLayers = ccudart.cudaDeviceAttr.cudaDevAttrMaxTextureCubemapLayeredLayers
    cudaDevAttrMaxSurface1DWidth = ccudart.cudaDeviceAttr.cudaDevAttrMaxSurface1DWidth
    cudaDevAttrMaxSurface2DWidth = ccudart.cudaDeviceAttr.cudaDevAttrMaxSurface2DWidth
    cudaDevAttrMaxSurface2DHeight = ccudart.cudaDeviceAttr.cudaDevAttrMaxSurface2DHeight
    cudaDevAttrMaxSurface3DWidth = ccudart.cudaDeviceAttr.cudaDevAttrMaxSurface3DWidth
    cudaDevAttrMaxSurface3DHeight = ccudart.cudaDeviceAttr.cudaDevAttrMaxSurface3DHeight
    cudaDevAttrMaxSurface3DDepth = ccudart.cudaDeviceAttr.cudaDevAttrMaxSurface3DDepth
    cudaDevAttrMaxSurface1DLayeredWidth = ccudart.cudaDeviceAttr.cudaDevAttrMaxSurface1DLayeredWidth
    cudaDevAttrMaxSurface1DLayeredLayers = ccudart.cudaDeviceAttr.cudaDevAttrMaxSurface1DLayeredLayers
    cudaDevAttrMaxSurface2DLayeredWidth = ccudart.cudaDeviceAttr.cudaDevAttrMaxSurface2DLayeredWidth
    cudaDevAttrMaxSurface2DLayeredHeight = ccudart.cudaDeviceAttr.cudaDevAttrMaxSurface2DLayeredHeight
    cudaDevAttrMaxSurface2DLayeredLayers = ccudart.cudaDeviceAttr.cudaDevAttrMaxSurface2DLayeredLayers
    cudaDevAttrMaxSurfaceCubemapWidth = ccudart.cudaDeviceAttr.cudaDevAttrMaxSurfaceCubemapWidth
    cudaDevAttrMaxSurfaceCubemapLayeredWidth = ccudart.cudaDeviceAttr.cudaDevAttrMaxSurfaceCubemapLayeredWidth
    cudaDevAttrMaxSurfaceCubemapLayeredLayers = ccudart.cudaDeviceAttr.cudaDevAttrMaxSurfaceCubemapLayeredLayers
    cudaDevAttrMaxTexture1DLinearWidth = ccudart.cudaDeviceAttr.cudaDevAttrMaxTexture1DLinearWidth
    cudaDevAttrMaxTexture2DLinearWidth = ccudart.cudaDeviceAttr.cudaDevAttrMaxTexture2DLinearWidth
    cudaDevAttrMaxTexture2DLinearHeight = ccudart.cudaDeviceAttr.cudaDevAttrMaxTexture2DLinearHeight
    cudaDevAttrMaxTexture2DLinearPitch = ccudart.cudaDeviceAttr.cudaDevAttrMaxTexture2DLinearPitch
    cudaDevAttrMaxTexture2DMipmappedWidth = ccudart.cudaDeviceAttr.cudaDevAttrMaxTexture2DMipmappedWidth
    cudaDevAttrMaxTexture2DMipmappedHeight = ccudart.cudaDeviceAttr.cudaDevAttrMaxTexture2DMipmappedHeight
    cudaDevAttrComputeCapabilityMajor = ccudart.cudaDeviceAttr.cudaDevAttrComputeCapabilityMajor
    cudaDevAttrComputeCapabilityMinor = ccudart.cudaDeviceAttr.cudaDevAttrComputeCapabilityMinor
    cudaDevAttrMaxTexture1DMipmappedWidth = ccudart.cudaDeviceAttr.cudaDevAttrMaxTexture1DMipmappedWidth
    cudaDevAttrStreamPrioritiesSupported = ccudart.cudaDeviceAttr.cudaDevAttrStreamPrioritiesSupported
    cudaDevAttrGlobalL1CacheSupported = ccudart.cudaDeviceAttr.cudaDevAttrGlobalL1CacheSupported
    cudaDevAttrLocalL1CacheSupported = ccudart.cudaDeviceAttr.cudaDevAttrLocalL1CacheSupported
    cudaDevAttrMaxSharedMemoryPerMultiprocessor = ccudart.cudaDeviceAttr.cudaDevAttrMaxSharedMemoryPerMultiprocessor
    cudaDevAttrMaxRegistersPerMultiprocessor = ccudart.cudaDeviceAttr.cudaDevAttrMaxRegistersPerMultiprocessor
    cudaDevAttrManagedMemory = ccudart.cudaDeviceAttr.cudaDevAttrManagedMemory
    cudaDevAttrIsMultiGpuBoard = ccudart.cudaDeviceAttr.cudaDevAttrIsMultiGpuBoard
    cudaDevAttrMultiGpuBoardGroupID = ccudart.cudaDeviceAttr.cudaDevAttrMultiGpuBoardGroupID
    cudaDevAttrHostNativeAtomicSupported = ccudart.cudaDeviceAttr.cudaDevAttrHostNativeAtomicSupported
    cudaDevAttrSingleToDoublePrecisionPerfRatio = ccudart.cudaDeviceAttr.cudaDevAttrSingleToDoublePrecisionPerfRatio
    cudaDevAttrPageableMemoryAccess = ccudart.cudaDeviceAttr.cudaDevAttrPageableMemoryAccess
    cudaDevAttrConcurrentManagedAccess = ccudart.cudaDeviceAttr.cudaDevAttrConcurrentManagedAccess
    cudaDevAttrComputePreemptionSupported = ccudart.cudaDeviceAttr.cudaDevAttrComputePreemptionSupported
    cudaDevAttrCanUseHostPointerForRegisteredMem = ccudart.cudaDeviceAttr.cudaDevAttrCanUseHostPointerForRegisteredMem
    cudaDevAttrReserved92 = ccudart.cudaDeviceAttr.cudaDevAttrReserved92
    cudaDevAttrReserved93 = ccudart.cudaDeviceAttr.cudaDevAttrReserved93
    cudaDevAttrReserved94 = ccudart.cudaDeviceAttr.cudaDevAttrReserved94
    cudaDevAttrCooperativeLaunch = ccudart.cudaDeviceAttr.cudaDevAttrCooperativeLaunch
    cudaDevAttrCooperativeMultiDeviceLaunch = ccudart.cudaDeviceAttr.cudaDevAttrCooperativeMultiDeviceLaunch
    cudaDevAttrMaxSharedMemoryPerBlockOptin = ccudart.cudaDeviceAttr.cudaDevAttrMaxSharedMemoryPerBlockOptin
    cudaDevAttrCanFlushRemoteWrites = ccudart.cudaDeviceAttr.cudaDevAttrCanFlushRemoteWrites
    cudaDevAttrHostRegisterSupported = ccudart.cudaDeviceAttr.cudaDevAttrHostRegisterSupported
    cudaDevAttrPageableMemoryAccessUsesHostPageTables = ccudart.cudaDeviceAttr.cudaDevAttrPageableMemoryAccessUsesHostPageTables
    cudaDevAttrDirectManagedMemAccessFromHost = ccudart.cudaDeviceAttr.cudaDevAttrDirectManagedMemAccessFromHost
    cudaDevAttrMaxBlocksPerMultiprocessor = ccudart.cudaDeviceAttr.cudaDevAttrMaxBlocksPerMultiprocessor
    cudaDevAttrMaxPersistingL2CacheSize = ccudart.cudaDeviceAttr.cudaDevAttrMaxPersistingL2CacheSize
    cudaDevAttrMaxAccessPolicyWindowSize = ccudart.cudaDeviceAttr.cudaDevAttrMaxAccessPolicyWindowSize
    cudaDevAttrReservedSharedMemoryPerBlock = ccudart.cudaDeviceAttr.cudaDevAttrReservedSharedMemoryPerBlock
    cudaDevAttrSparseCudaArraySupported = ccudart.cudaDeviceAttr.cudaDevAttrSparseCudaArraySupported
    cudaDevAttrHostRegisterReadOnlySupported = ccudart.cudaDeviceAttr.cudaDevAttrHostRegisterReadOnlySupported
    cudaDevAttrMaxTimelineSemaphoreInteropSupported = ccudart.cudaDeviceAttr.cudaDevAttrMaxTimelineSemaphoreInteropSupported
    cudaDevAttrMemoryPoolsSupported = ccudart.cudaDeviceAttr.cudaDevAttrMemoryPoolsSupported
    cudaDevAttrGPUDirectRDMASupported = ccudart.cudaDeviceAttr.cudaDevAttrGPUDirectRDMASupported
    cudaDevAttrGPUDirectRDMAFlushWritesOptions = ccudart.cudaDeviceAttr.cudaDevAttrGPUDirectRDMAFlushWritesOptions
    cudaDevAttrGPUDirectRDMAWritesOrdering = ccudart.cudaDeviceAttr.cudaDevAttrGPUDirectRDMAWritesOrdering
    cudaDevAttrMemoryPoolSupportedHandleTypes = ccudart.cudaDeviceAttr.cudaDevAttrMemoryPoolSupportedHandleTypes
    cudaDevAttrMax = ccudart.cudaDeviceAttr.cudaDevAttrMax

class cudaMemPoolAttr(Enum):
    cudaMemPoolReuseFollowEventDependencies = ccudart.cudaMemPoolAttr.cudaMemPoolReuseFollowEventDependencies
    cudaMemPoolReuseAllowOpportunistic = ccudart.cudaMemPoolAttr.cudaMemPoolReuseAllowOpportunistic
    cudaMemPoolReuseAllowInternalDependencies = ccudart.cudaMemPoolAttr.cudaMemPoolReuseAllowInternalDependencies
    cudaMemPoolAttrReleaseThreshold = ccudart.cudaMemPoolAttr.cudaMemPoolAttrReleaseThreshold
    cudaMemPoolAttrReservedMemCurrent = ccudart.cudaMemPoolAttr.cudaMemPoolAttrReservedMemCurrent
    cudaMemPoolAttrReservedMemHigh = ccudart.cudaMemPoolAttr.cudaMemPoolAttrReservedMemHigh
    cudaMemPoolAttrUsedMemCurrent = ccudart.cudaMemPoolAttr.cudaMemPoolAttrUsedMemCurrent
    cudaMemPoolAttrUsedMemHigh = ccudart.cudaMemPoolAttr.cudaMemPoolAttrUsedMemHigh

class cudaMemLocationType(Enum):
    cudaMemLocationTypeInvalid = ccudart.cudaMemLocationType.cudaMemLocationTypeInvalid
    cudaMemLocationTypeDevice = ccudart.cudaMemLocationType.cudaMemLocationTypeDevice

class cudaMemAccessFlags(Enum):
    cudaMemAccessFlagsProtNone = ccudart.cudaMemAccessFlags.cudaMemAccessFlagsProtNone
    cudaMemAccessFlagsProtRead = ccudart.cudaMemAccessFlags.cudaMemAccessFlagsProtRead
    cudaMemAccessFlagsProtReadWrite = ccudart.cudaMemAccessFlags.cudaMemAccessFlagsProtReadWrite

class cudaMemAllocationType(Enum):
    cudaMemAllocationTypeInvalid = ccudart.cudaMemAllocationType.cudaMemAllocationTypeInvalid
    cudaMemAllocationTypePinned = ccudart.cudaMemAllocationType.cudaMemAllocationTypePinned
    cudaMemAllocationTypeMax = ccudart.cudaMemAllocationType.cudaMemAllocationTypeMax

class cudaMemAllocationHandleType(Enum):
    cudaMemHandleTypeNone = ccudart.cudaMemAllocationHandleType.cudaMemHandleTypeNone
    cudaMemHandleTypePosixFileDescriptor = ccudart.cudaMemAllocationHandleType.cudaMemHandleTypePosixFileDescriptor
    cudaMemHandleTypeWin32 = ccudart.cudaMemAllocationHandleType.cudaMemHandleTypeWin32
    cudaMemHandleTypeWin32Kmt = ccudart.cudaMemAllocationHandleType.cudaMemHandleTypeWin32Kmt

class cudaGraphMemAttributeType(Enum):
    cudaGraphMemAttrUsedMemCurrent = ccudart.cudaGraphMemAttributeType.cudaGraphMemAttrUsedMemCurrent
    cudaGraphMemAttrUsedMemHigh = ccudart.cudaGraphMemAttributeType.cudaGraphMemAttrUsedMemHigh
    cudaGraphMemAttrReservedMemCurrent = ccudart.cudaGraphMemAttributeType.cudaGraphMemAttrReservedMemCurrent
    cudaGraphMemAttrReservedMemHigh = ccudart.cudaGraphMemAttributeType.cudaGraphMemAttrReservedMemHigh

class cudaDeviceP2PAttr(Enum):
    cudaDevP2PAttrPerformanceRank = ccudart.cudaDeviceP2PAttr.cudaDevP2PAttrPerformanceRank
    cudaDevP2PAttrAccessSupported = ccudart.cudaDeviceP2PAttr.cudaDevP2PAttrAccessSupported
    cudaDevP2PAttrNativeAtomicSupported = ccudart.cudaDeviceP2PAttr.cudaDevP2PAttrNativeAtomicSupported
    cudaDevP2PAttrCudaArrayAccessSupported = ccudart.cudaDeviceP2PAttr.cudaDevP2PAttrCudaArrayAccessSupported

class cudaExternalMemoryHandleType(Enum):
    cudaExternalMemoryHandleTypeOpaqueFd = ccudart.cudaExternalMemoryHandleType.cudaExternalMemoryHandleTypeOpaqueFd
    cudaExternalMemoryHandleTypeOpaqueWin32 = ccudart.cudaExternalMemoryHandleType.cudaExternalMemoryHandleTypeOpaqueWin32
    cudaExternalMemoryHandleTypeOpaqueWin32Kmt = ccudart.cudaExternalMemoryHandleType.cudaExternalMemoryHandleTypeOpaqueWin32Kmt
    cudaExternalMemoryHandleTypeD3D12Heap = ccudart.cudaExternalMemoryHandleType.cudaExternalMemoryHandleTypeD3D12Heap
    cudaExternalMemoryHandleTypeD3D12Resource = ccudart.cudaExternalMemoryHandleType.cudaExternalMemoryHandleTypeD3D12Resource
    cudaExternalMemoryHandleTypeD3D11Resource = ccudart.cudaExternalMemoryHandleType.cudaExternalMemoryHandleTypeD3D11Resource
    cudaExternalMemoryHandleTypeD3D11ResourceKmt = ccudart.cudaExternalMemoryHandleType.cudaExternalMemoryHandleTypeD3D11ResourceKmt
    cudaExternalMemoryHandleTypeNvSciBuf = ccudart.cudaExternalMemoryHandleType.cudaExternalMemoryHandleTypeNvSciBuf

class cudaExternalSemaphoreHandleType(Enum):
    cudaExternalSemaphoreHandleTypeOpaqueFd = ccudart.cudaExternalSemaphoreHandleType.cudaExternalSemaphoreHandleTypeOpaqueFd
    cudaExternalSemaphoreHandleTypeOpaqueWin32 = ccudart.cudaExternalSemaphoreHandleType.cudaExternalSemaphoreHandleTypeOpaqueWin32
    cudaExternalSemaphoreHandleTypeOpaqueWin32Kmt = ccudart.cudaExternalSemaphoreHandleType.cudaExternalSemaphoreHandleTypeOpaqueWin32Kmt
    cudaExternalSemaphoreHandleTypeD3D12Fence = ccudart.cudaExternalSemaphoreHandleType.cudaExternalSemaphoreHandleTypeD3D12Fence
    cudaExternalSemaphoreHandleTypeD3D11Fence = ccudart.cudaExternalSemaphoreHandleType.cudaExternalSemaphoreHandleTypeD3D11Fence
    cudaExternalSemaphoreHandleTypeNvSciSync = ccudart.cudaExternalSemaphoreHandleType.cudaExternalSemaphoreHandleTypeNvSciSync
    cudaExternalSemaphoreHandleTypeKeyedMutex = ccudart.cudaExternalSemaphoreHandleType.cudaExternalSemaphoreHandleTypeKeyedMutex
    cudaExternalSemaphoreHandleTypeKeyedMutexKmt = ccudart.cudaExternalSemaphoreHandleType.cudaExternalSemaphoreHandleTypeKeyedMutexKmt
    cudaExternalSemaphoreHandleTypeTimelineSemaphoreFd = ccudart.cudaExternalSemaphoreHandleType.cudaExternalSemaphoreHandleTypeTimelineSemaphoreFd
    cudaExternalSemaphoreHandleTypeTimelineSemaphoreWin32 = ccudart.cudaExternalSemaphoreHandleType.cudaExternalSemaphoreHandleTypeTimelineSemaphoreWin32

class cudaCGScope(Enum):
    cudaCGScopeInvalid = ccudart.cudaCGScope.cudaCGScopeInvalid
    cudaCGScopeGrid = ccudart.cudaCGScope.cudaCGScopeGrid
    cudaCGScopeMultiGrid = ccudart.cudaCGScope.cudaCGScopeMultiGrid

class cudaGraphNodeType(Enum):
    cudaGraphNodeTypeKernel = ccudart.cudaGraphNodeType.cudaGraphNodeTypeKernel
    cudaGraphNodeTypeMemcpy = ccudart.cudaGraphNodeType.cudaGraphNodeTypeMemcpy
    cudaGraphNodeTypeMemset = ccudart.cudaGraphNodeType.cudaGraphNodeTypeMemset
    cudaGraphNodeTypeHost = ccudart.cudaGraphNodeType.cudaGraphNodeTypeHost
    cudaGraphNodeTypeGraph = ccudart.cudaGraphNodeType.cudaGraphNodeTypeGraph
    cudaGraphNodeTypeEmpty = ccudart.cudaGraphNodeType.cudaGraphNodeTypeEmpty
    cudaGraphNodeTypeWaitEvent = ccudart.cudaGraphNodeType.cudaGraphNodeTypeWaitEvent
    cudaGraphNodeTypeEventRecord = ccudart.cudaGraphNodeType.cudaGraphNodeTypeEventRecord
    cudaGraphNodeTypeExtSemaphoreSignal = ccudart.cudaGraphNodeType.cudaGraphNodeTypeExtSemaphoreSignal
    cudaGraphNodeTypeExtSemaphoreWait = ccudart.cudaGraphNodeType.cudaGraphNodeTypeExtSemaphoreWait
    cudaGraphNodeTypeMemAlloc = ccudart.cudaGraphNodeType.cudaGraphNodeTypeMemAlloc
    cudaGraphNodeTypeMemFree = ccudart.cudaGraphNodeType.cudaGraphNodeTypeMemFree
    cudaGraphNodeTypeCount = ccudart.cudaGraphNodeType.cudaGraphNodeTypeCount

class cudaGraphExecUpdateResult(Enum):
    cudaGraphExecUpdateSuccess = ccudart.cudaGraphExecUpdateResult.cudaGraphExecUpdateSuccess
    cudaGraphExecUpdateError = ccudart.cudaGraphExecUpdateResult.cudaGraphExecUpdateError
    cudaGraphExecUpdateErrorTopologyChanged = ccudart.cudaGraphExecUpdateResult.cudaGraphExecUpdateErrorTopologyChanged
    cudaGraphExecUpdateErrorNodeTypeChanged = ccudart.cudaGraphExecUpdateResult.cudaGraphExecUpdateErrorNodeTypeChanged
    cudaGraphExecUpdateErrorFunctionChanged = ccudart.cudaGraphExecUpdateResult.cudaGraphExecUpdateErrorFunctionChanged
    cudaGraphExecUpdateErrorParametersChanged = ccudart.cudaGraphExecUpdateResult.cudaGraphExecUpdateErrorParametersChanged
    cudaGraphExecUpdateErrorNotSupported = ccudart.cudaGraphExecUpdateResult.cudaGraphExecUpdateErrorNotSupported
    cudaGraphExecUpdateErrorUnsupportedFunctionChange = ccudart.cudaGraphExecUpdateResult.cudaGraphExecUpdateErrorUnsupportedFunctionChange

class cudaGetDriverEntryPointFlags(Enum):
    cudaEnableDefault = ccudart.cudaGetDriverEntryPointFlags.cudaEnableDefault
    cudaEnableLegacyStream = ccudart.cudaGetDriverEntryPointFlags.cudaEnableLegacyStream
    cudaEnablePerThreadDefaultStream = ccudart.cudaGetDriverEntryPointFlags.cudaEnablePerThreadDefaultStream

class cudaGraphDebugDotFlags(Enum):
    cudaGraphDebugDotFlagsVerbose = ccudart.cudaGraphDebugDotFlags.cudaGraphDebugDotFlagsVerbose
    cudaGraphDebugDotFlagsKernelNodeParams = ccudart.cudaGraphDebugDotFlags.cudaGraphDebugDotFlagsKernelNodeParams
    cudaGraphDebugDotFlagsMemcpyNodeParams = ccudart.cudaGraphDebugDotFlags.cudaGraphDebugDotFlagsMemcpyNodeParams
    cudaGraphDebugDotFlagsMemsetNodeParams = ccudart.cudaGraphDebugDotFlags.cudaGraphDebugDotFlagsMemsetNodeParams
    cudaGraphDebugDotFlagsHostNodeParams = ccudart.cudaGraphDebugDotFlags.cudaGraphDebugDotFlagsHostNodeParams
    cudaGraphDebugDotFlagsEventNodeParams = ccudart.cudaGraphDebugDotFlags.cudaGraphDebugDotFlagsEventNodeParams
    cudaGraphDebugDotFlagsExtSemasSignalNodeParams = ccudart.cudaGraphDebugDotFlags.cudaGraphDebugDotFlagsExtSemasSignalNodeParams
    cudaGraphDebugDotFlagsExtSemasWaitNodeParams = ccudart.cudaGraphDebugDotFlags.cudaGraphDebugDotFlagsExtSemasWaitNodeParams
    cudaGraphDebugDotFlagsKernelNodeAttributes = ccudart.cudaGraphDebugDotFlags.cudaGraphDebugDotFlagsKernelNodeAttributes
    cudaGraphDebugDotFlagsHandles = ccudart.cudaGraphDebugDotFlags.cudaGraphDebugDotFlagsHandles

class cudaGraphInstantiateFlags(Enum):
    cudaGraphInstantiateFlagAutoFreeOnLaunch = ccudart.cudaGraphInstantiateFlags.cudaGraphInstantiateFlagAutoFreeOnLaunch

class cudaSurfaceBoundaryMode(Enum):
    cudaBoundaryModeZero = ccudart.cudaSurfaceBoundaryMode.cudaBoundaryModeZero
    cudaBoundaryModeClamp = ccudart.cudaSurfaceBoundaryMode.cudaBoundaryModeClamp
    cudaBoundaryModeTrap = ccudart.cudaSurfaceBoundaryMode.cudaBoundaryModeTrap

class cudaSurfaceFormatMode(Enum):
    cudaFormatModeForced = ccudart.cudaSurfaceFormatMode.cudaFormatModeForced
    cudaFormatModeAuto = ccudart.cudaSurfaceFormatMode.cudaFormatModeAuto

class cudaTextureAddressMode(Enum):
    cudaAddressModeWrap = ccudart.cudaTextureAddressMode.cudaAddressModeWrap
    cudaAddressModeClamp = ccudart.cudaTextureAddressMode.cudaAddressModeClamp
    cudaAddressModeMirror = ccudart.cudaTextureAddressMode.cudaAddressModeMirror
    cudaAddressModeBorder = ccudart.cudaTextureAddressMode.cudaAddressModeBorder

class cudaTextureFilterMode(Enum):
    cudaFilterModePoint = ccudart.cudaTextureFilterMode.cudaFilterModePoint
    cudaFilterModeLinear = ccudart.cudaTextureFilterMode.cudaFilterModeLinear

class cudaTextureReadMode(Enum):
    cudaReadModeElementType = ccudart.cudaTextureReadMode.cudaReadModeElementType
    cudaReadModeNormalizedFloat = ccudart.cudaTextureReadMode.cudaReadModeNormalizedFloat

class cudaDataType(Enum):
    CUDA_R_16F = ccudart.cudaDataType_t.CUDA_R_16F
    CUDA_C_16F = ccudart.cudaDataType_t.CUDA_C_16F
    CUDA_R_16BF = ccudart.cudaDataType_t.CUDA_R_16BF
    CUDA_C_16BF = ccudart.cudaDataType_t.CUDA_C_16BF
    CUDA_R_32F = ccudart.cudaDataType_t.CUDA_R_32F
    CUDA_C_32F = ccudart.cudaDataType_t.CUDA_C_32F
    CUDA_R_64F = ccudart.cudaDataType_t.CUDA_R_64F
    CUDA_C_64F = ccudart.cudaDataType_t.CUDA_C_64F
    CUDA_R_4I = ccudart.cudaDataType_t.CUDA_R_4I
    CUDA_C_4I = ccudart.cudaDataType_t.CUDA_C_4I
    CUDA_R_4U = ccudart.cudaDataType_t.CUDA_R_4U
    CUDA_C_4U = ccudart.cudaDataType_t.CUDA_C_4U
    CUDA_R_8I = ccudart.cudaDataType_t.CUDA_R_8I
    CUDA_C_8I = ccudart.cudaDataType_t.CUDA_C_8I
    CUDA_R_8U = ccudart.cudaDataType_t.CUDA_R_8U
    CUDA_C_8U = ccudart.cudaDataType_t.CUDA_C_8U
    CUDA_R_16I = ccudart.cudaDataType_t.CUDA_R_16I
    CUDA_C_16I = ccudart.cudaDataType_t.CUDA_C_16I
    CUDA_R_16U = ccudart.cudaDataType_t.CUDA_R_16U
    CUDA_C_16U = ccudart.cudaDataType_t.CUDA_C_16U
    CUDA_R_32I = ccudart.cudaDataType_t.CUDA_R_32I
    CUDA_C_32I = ccudart.cudaDataType_t.CUDA_C_32I
    CUDA_R_32U = ccudart.cudaDataType_t.CUDA_R_32U
    CUDA_C_32U = ccudart.cudaDataType_t.CUDA_C_32U
    CUDA_R_64I = ccudart.cudaDataType_t.CUDA_R_64I
    CUDA_C_64I = ccudart.cudaDataType_t.CUDA_C_64I
    CUDA_R_64U = ccudart.cudaDataType_t.CUDA_R_64U
    CUDA_C_64U = ccudart.cudaDataType_t.CUDA_C_64U

class libraryPropertyType(Enum):
    MAJOR_VERSION = ccudart.libraryPropertyType_t.MAJOR_VERSION
    MINOR_VERSION = ccudart.libraryPropertyType_t.MINOR_VERSION
    PATCH_LEVEL = ccudart.libraryPropertyType_t.PATCH_LEVEL


cdef class cudaArray_t:
    def __cinit__(self, void_ptr init_value = 0, void_ptr _ptr = 0):
        if _ptr == 0:
            self._ptr_owner = True
            self._ptr = <ccudart.cudaArray_t *>calloc(1, sizeof(ccudart.cudaArray_t))
            if self._ptr is NULL:
                raise MemoryError('Failed to allocate length x size memory: 1x' + str(sizeof(ccudart.cudaArray_t)))
            self._ptr[0] = <ccudart.cudaArray_t>init_value
        else:
            self._ptr_owner = False
            self._ptr = <ccudart.cudaArray_t *>_ptr
    def __dealloc__(self):
        if self._ptr_owner is True and self._ptr is not NULL:
            free(self._ptr)
    def __repr__(self):
        return '<cudaArray_t ' + str(hex(self.__int__())) + '>'
    def __index__(self):
        return self.__int__()
    def __int__(self):
        return <void_ptr>self._ptr[0]
    def getPtr(self):
        return <void_ptr>self._ptr

cdef class cudaArray_const_t:
    def __cinit__(self, void_ptr init_value = 0, void_ptr _ptr = 0):
        if _ptr == 0:
            self._ptr_owner = True
            self._ptr = <ccudart.cudaArray_const_t *>calloc(1, sizeof(ccudart.cudaArray_const_t))
            if self._ptr is NULL:
                raise MemoryError('Failed to allocate length x size memory: 1x' + str(sizeof(ccudart.cudaArray_const_t)))
            self._ptr[0] = <ccudart.cudaArray_const_t>init_value
        else:
            self._ptr_owner = False
            self._ptr = <ccudart.cudaArray_const_t *>_ptr
    def __dealloc__(self):
        if self._ptr_owner is True and self._ptr is not NULL:
            free(self._ptr)
    def __repr__(self):
        return '<cudaArray_const_t ' + str(hex(self.__int__())) + '>'
    def __index__(self):
        return self.__int__()
    def __int__(self):
        return <void_ptr>self._ptr[0]
    def getPtr(self):
        return <void_ptr>self._ptr

cdef class cudaMipmappedArray_t:
    def __cinit__(self, void_ptr init_value = 0, void_ptr _ptr = 0):
        if _ptr == 0:
            self._ptr_owner = True
            self._ptr = <ccudart.cudaMipmappedArray_t *>calloc(1, sizeof(ccudart.cudaMipmappedArray_t))
            if self._ptr is NULL:
                raise MemoryError('Failed to allocate length x size memory: 1x' + str(sizeof(ccudart.cudaMipmappedArray_t)))
            self._ptr[0] = <ccudart.cudaMipmappedArray_t>init_value
        else:
            self._ptr_owner = False
            self._ptr = <ccudart.cudaMipmappedArray_t *>_ptr
    def __dealloc__(self):
        if self._ptr_owner is True and self._ptr is not NULL:
            free(self._ptr)
    def __repr__(self):
        return '<cudaMipmappedArray_t ' + str(hex(self.__int__())) + '>'
    def __index__(self):
        return self.__int__()
    def __int__(self):
        return <void_ptr>self._ptr[0]
    def getPtr(self):
        return <void_ptr>self._ptr

cdef class cudaMipmappedArray_const_t:
    def __cinit__(self, void_ptr init_value = 0, void_ptr _ptr = 0):
        if _ptr == 0:
            self._ptr_owner = True
            self._ptr = <ccudart.cudaMipmappedArray_const_t *>calloc(1, sizeof(ccudart.cudaMipmappedArray_const_t))
            if self._ptr is NULL:
                raise MemoryError('Failed to allocate length x size memory: 1x' + str(sizeof(ccudart.cudaMipmappedArray_const_t)))
            self._ptr[0] = <ccudart.cudaMipmappedArray_const_t>init_value
        else:
            self._ptr_owner = False
            self._ptr = <ccudart.cudaMipmappedArray_const_t *>_ptr
    def __dealloc__(self):
        if self._ptr_owner is True and self._ptr is not NULL:
            free(self._ptr)
    def __repr__(self):
        return '<cudaMipmappedArray_const_t ' + str(hex(self.__int__())) + '>'
    def __index__(self):
        return self.__int__()
    def __int__(self):
        return <void_ptr>self._ptr[0]
    def getPtr(self):
        return <void_ptr>self._ptr

cdef class cudaGraphicsResource_t:
    def __cinit__(self, void_ptr init_value = 0, void_ptr _ptr = 0):
        if _ptr == 0:
            self._ptr_owner = True
            self._ptr = <ccudart.cudaGraphicsResource_t *>calloc(1, sizeof(ccudart.cudaGraphicsResource_t))
            if self._ptr is NULL:
                raise MemoryError('Failed to allocate length x size memory: 1x' + str(sizeof(ccudart.cudaGraphicsResource_t)))
            self._ptr[0] = <ccudart.cudaGraphicsResource_t>init_value
        else:
            self._ptr_owner = False
            self._ptr = <ccudart.cudaGraphicsResource_t *>_ptr
    def __dealloc__(self):
        if self._ptr_owner is True and self._ptr is not NULL:
            free(self._ptr)
    def __repr__(self):
        return '<cudaGraphicsResource_t ' + str(hex(self.__int__())) + '>'
    def __index__(self):
        return self.__int__()
    def __int__(self):
        return <void_ptr>self._ptr[0]
    def getPtr(self):
        return <void_ptr>self._ptr

cdef class cudaExternalMemory_t:
    def __cinit__(self, void_ptr init_value = 0, void_ptr _ptr = 0):
        if _ptr == 0:
            self._ptr_owner = True
            self._ptr = <ccudart.cudaExternalMemory_t *>calloc(1, sizeof(ccudart.cudaExternalMemory_t))
            if self._ptr is NULL:
                raise MemoryError('Failed to allocate length x size memory: 1x' + str(sizeof(ccudart.cudaExternalMemory_t)))
            self._ptr[0] = <ccudart.cudaExternalMemory_t>init_value
        else:
            self._ptr_owner = False
            self._ptr = <ccudart.cudaExternalMemory_t *>_ptr
    def __dealloc__(self):
        if self._ptr_owner is True and self._ptr is not NULL:
            free(self._ptr)
    def __repr__(self):
        return '<cudaExternalMemory_t ' + str(hex(self.__int__())) + '>'
    def __index__(self):
        return self.__int__()
    def __int__(self):
        return <void_ptr>self._ptr[0]
    def getPtr(self):
        return <void_ptr>self._ptr

cdef class cudaExternalSemaphore_t:
    def __cinit__(self, void_ptr init_value = 0, void_ptr _ptr = 0):
        if _ptr == 0:
            self._ptr_owner = True
            self._ptr = <ccudart.cudaExternalSemaphore_t *>calloc(1, sizeof(ccudart.cudaExternalSemaphore_t))
            if self._ptr is NULL:
                raise MemoryError('Failed to allocate length x size memory: 1x' + str(sizeof(ccudart.cudaExternalSemaphore_t)))
            self._ptr[0] = <ccudart.cudaExternalSemaphore_t>init_value
        else:
            self._ptr_owner = False
            self._ptr = <ccudart.cudaExternalSemaphore_t *>_ptr
    def __dealloc__(self):
        if self._ptr_owner is True and self._ptr is not NULL:
            free(self._ptr)
    def __repr__(self):
        return '<cudaExternalSemaphore_t ' + str(hex(self.__int__())) + '>'
    def __index__(self):
        return self.__int__()
    def __int__(self):
        return <void_ptr>self._ptr[0]
    def getPtr(self):
        return <void_ptr>self._ptr

cdef class cudaHostFn_t:
    def __cinit__(self, void_ptr init_value = 0, void_ptr _ptr = 0):
        if _ptr == 0:
            self._ptr_owner = True
            self._ptr = <ccudart.cudaHostFn_t *>calloc(1, sizeof(ccudart.cudaHostFn_t))
            if self._ptr is NULL:
                raise MemoryError('Failed to allocate length x size memory: 1x' + str(sizeof(ccudart.cudaHostFn_t)))
            self._ptr[0] = <ccudart.cudaHostFn_t>init_value
        else:
            self._ptr_owner = False
            self._ptr = <ccudart.cudaHostFn_t *>_ptr
    def __dealloc__(self):
        if self._ptr_owner is True and self._ptr is not NULL:
            free(self._ptr)
    def __repr__(self):
        return '<cudaHostFn_t ' + str(hex(self.__int__())) + '>'
    def __index__(self):
        return self.__int__()
    def __int__(self):
        return <void_ptr>self._ptr[0]
    def getPtr(self):
        return <void_ptr>self._ptr

cdef class cudaStreamCallback_t:
    def __cinit__(self, void_ptr init_value = 0, void_ptr _ptr = 0):
        if _ptr == 0:
            self._ptr_owner = True
            self._ptr = <ccudart.cudaStreamCallback_t *>calloc(1, sizeof(ccudart.cudaStreamCallback_t))
            if self._ptr is NULL:
                raise MemoryError('Failed to allocate length x size memory: 1x' + str(sizeof(ccudart.cudaStreamCallback_t)))
            self._ptr[0] = <ccudart.cudaStreamCallback_t>init_value
        else:
            self._ptr_owner = False
            self._ptr = <ccudart.cudaStreamCallback_t *>_ptr
    def __dealloc__(self):
        if self._ptr_owner is True and self._ptr is not NULL:
            free(self._ptr)
    def __repr__(self):
        return '<cudaStreamCallback_t ' + str(hex(self.__int__())) + '>'
    def __index__(self):
        return self.__int__()
    def __int__(self):
        return <void_ptr>self._ptr[0]
    def getPtr(self):
        return <void_ptr>self._ptr

cdef class cudaSurfaceObject_t:
    def __cinit__(self, unsigned long long init_value = 0, void_ptr _ptr = 0):
        if _ptr == 0:
            self._ptr_owner = True
            self._ptr = <ccudart.cudaSurfaceObject_t *>calloc(1, sizeof(ccudart.cudaSurfaceObject_t))
            if self._ptr is NULL:
                raise MemoryError('Failed to allocate length x size memory: 1x' + str(sizeof(ccudart.cudaSurfaceObject_t)))
        else:
            self._ptr_owner = False
            self._ptr = <ccudart.cudaSurfaceObject_t *>_ptr
        if init_value:
            self._ptr[0] = init_value
    def __dealloc__(self):
        if self._ptr_owner is True and self._ptr is not NULL:
            free(self._ptr)
    def __repr__(self):
        return '<cudaSurfaceObject_t ' + str(self.__int__()) + '>'
    def __int__(self):
        return <unsigned long long>self._ptr[0]
    def getPtr(self):
        return <void_ptr>self._ptr

cdef class cudaTextureObject_t:
    def __cinit__(self, unsigned long long init_value = 0, void_ptr _ptr = 0):
        if _ptr == 0:
            self._ptr_owner = True
            self._ptr = <ccudart.cudaTextureObject_t *>calloc(1, sizeof(ccudart.cudaTextureObject_t))
            if self._ptr is NULL:
                raise MemoryError('Failed to allocate length x size memory: 1x' + str(sizeof(ccudart.cudaTextureObject_t)))
        else:
            self._ptr_owner = False
            self._ptr = <ccudart.cudaTextureObject_t *>_ptr
        if init_value:
            self._ptr[0] = init_value
    def __dealloc__(self):
        if self._ptr_owner is True and self._ptr is not NULL:
            free(self._ptr)
    def __repr__(self):
        return '<cudaTextureObject_t ' + str(self.__int__()) + '>'
    def __int__(self):
        return <unsigned long long>self._ptr[0]
    def getPtr(self):
        return <void_ptr>self._ptr

cdef class dim3:
    def __cinit__(self, void_ptr _ptr = 0):
        if _ptr == 0:
            self._ptr_owner = True
            self._ptr = <ccudart.dim3 *>calloc(1, sizeof(ccudart.dim3))
            if self._ptr is NULL:
                raise MemoryError('Failed to allocate length x size memory: 1x' + str(sizeof(ccudart.dim3)))
        else:
            self._ptr_owner = False
            self._ptr = <ccudart.dim3 *>_ptr
    def __init__(self, void_ptr _ptr = 0):
        pass
    def __dealloc__(self):
        if self._ptr_owner is True and self._ptr is not NULL:
            free(self._ptr)
        pass
    def getPtr(self):
        return <void_ptr>self._ptr
    def __repr__(self):
        if self._ptr is not NULL:
            str_list = []
            str_list += ['x : ' + str(self.x)]
            str_list += ['y : ' + str(self.y)]
            str_list += ['z : ' + str(self.z)]
            return '\n'.join(str_list)
        else:
            return ''

    @property
    def x(self):
        return self._ptr[0].x
    @x.setter
    def x(self, unsigned int x):
        pass
        self._ptr[0].x = x
    @property
    def y(self):
        return self._ptr[0].y
    @y.setter
    def y(self, unsigned int y):
        pass
        self._ptr[0].y = y
    @property
    def z(self):
        return self._ptr[0].z
    @z.setter
    def z(self, unsigned int z):
        pass
        self._ptr[0].z = z

cdef class cudaChannelFormatDesc:
    def __cinit__(self, void_ptr _ptr = 0):
        if _ptr == 0:
            self._ptr_owner = True
            self._ptr = <ccudart.cudaChannelFormatDesc *>calloc(1, sizeof(ccudart.cudaChannelFormatDesc))
            if self._ptr is NULL:
                raise MemoryError('Failed to allocate length x size memory: 1x' + str(sizeof(ccudart.cudaChannelFormatDesc)))
        else:
            self._ptr_owner = False
            self._ptr = <ccudart.cudaChannelFormatDesc *>_ptr
    def __init__(self, void_ptr _ptr = 0):
        pass
    def __dealloc__(self):
        if self._ptr_owner is True and self._ptr is not NULL:
            free(self._ptr)
        pass
    def getPtr(self):
        return <void_ptr>self._ptr
    def __repr__(self):
        if self._ptr is not NULL:
            str_list = []
            str_list += ['x : ' + str(self.x)]
            str_list += ['y : ' + str(self.y)]
            str_list += ['z : ' + str(self.z)]
            str_list += ['w : ' + str(self.w)]
            str_list += ['f : ' + str(self.f)]
            return '\n'.join(str_list)
        else:
            return ''

    @property
    def x(self):
        return self._ptr[0].x
    @x.setter
    def x(self, int x):
        pass
        self._ptr[0].x = x
    @property
    def y(self):
        return self._ptr[0].y
    @y.setter
    def y(self, int y):
        pass
        self._ptr[0].y = y
    @property
    def z(self):
        return self._ptr[0].z
    @z.setter
    def z(self, int z):
        pass
        self._ptr[0].z = z
    @property
    def w(self):
        return self._ptr[0].w
    @w.setter
    def w(self, int w):
        pass
        self._ptr[0].w = w
    @property
    def f(self):
        return cudaChannelFormatKind(self._ptr[0].f)
    @f.setter
    def f(self, f not None : cudaChannelFormatKind):
        pass
        self._ptr[0].f = f.value

cdef class _cudaArraySparseProperties_tileExtent_s:
    def __cinit__(self, void_ptr _ptr):
        self._ptr = <ccudart.cudaArraySparseProperties *>_ptr
    def __init__(self, void_ptr _ptr):
        pass
    def getPtr(self):
        return <void_ptr>self._ptr
    def __repr__(self):
        if self._ptr is not NULL:
            str_list = []
            str_list += ['width : ' + str(self.width)]
            str_list += ['height : ' + str(self.height)]
            str_list += ['depth : ' + str(self.depth)]
            return '\n'.join(str_list)
        else:
            return ''

    @property
    def width(self):
        return self._ptr[0].tileExtent.width
    @width.setter
    def width(self, unsigned int width):
        pass
        self._ptr[0].tileExtent.width = width
    @property
    def height(self):
        return self._ptr[0].tileExtent.height
    @height.setter
    def height(self, unsigned int height):
        pass
        self._ptr[0].tileExtent.height = height
    @property
    def depth(self):
        return self._ptr[0].tileExtent.depth
    @depth.setter
    def depth(self, unsigned int depth):
        pass
        self._ptr[0].tileExtent.depth = depth

cdef class cudaArraySparseProperties:
    def __cinit__(self, void_ptr _ptr = 0):
        if _ptr == 0:
            self._ptr_owner = True
            self._ptr = <ccudart.cudaArraySparseProperties *>calloc(1, sizeof(ccudart.cudaArraySparseProperties))
            if self._ptr is NULL:
                raise MemoryError('Failed to allocate length x size memory: 1x' + str(sizeof(ccudart.cudaArraySparseProperties)))
        else:
            self._ptr_owner = False
            self._ptr = <ccudart.cudaArraySparseProperties *>_ptr
    def __init__(self, void_ptr _ptr = 0):
        self._tileExtent = _cudaArraySparseProperties_tileExtent_s(_ptr=<void_ptr>self._ptr)
    def __dealloc__(self):
        if self._ptr_owner is True and self._ptr is not NULL:
            free(self._ptr)
        pass
    def getPtr(self):
        return <void_ptr>self._ptr
    def __repr__(self):
        if self._ptr is not NULL:
            str_list = []
            str_list += ['tileExtent :\n' + '\n'.join(['    ' + line for line in str(self.tileExtent).splitlines()])]
            str_list += ['miptailFirstLevel : ' + str(self.miptailFirstLevel)]
            str_list += ['miptailSize : ' + str(self.miptailSize)]
            str_list += ['flags : ' + str(self.flags)]
            str_list += ['reserved : ' + str(self.reserved)]
            return '\n'.join(str_list)
        else:
            return ''

    @property
    def tileExtent(self):
        return self._tileExtent
    @tileExtent.setter
    def tileExtent(self, tileExtent not None : _cudaArraySparseProperties_tileExtent_s):
        pass
        for _attr in dir(tileExtent):
            if _attr == 'getPtr':
                continue
            if not _attr.startswith('_'):
                setattr(self._tileExtent, _attr, getattr(tileExtent, _attr))
    @property
    def miptailFirstLevel(self):
        return self._ptr[0].miptailFirstLevel
    @miptailFirstLevel.setter
    def miptailFirstLevel(self, unsigned int miptailFirstLevel):
        pass
        self._ptr[0].miptailFirstLevel = miptailFirstLevel
    @property
    def miptailSize(self):
        return self._ptr[0].miptailSize
    @miptailSize.setter
    def miptailSize(self, unsigned long long miptailSize):
        pass
        self._ptr[0].miptailSize = miptailSize
    @property
    def flags(self):
        return self._ptr[0].flags
    @flags.setter
    def flags(self, unsigned int flags):
        pass
        self._ptr[0].flags = flags
    @property
    def reserved(self):
        return self._ptr[0].reserved
    @reserved.setter
    def reserved(self, reserved):
        pass
        self._ptr[0].reserved = reserved

cdef class cudaPitchedPtr:
    def __cinit__(self, void_ptr _ptr = 0):
        if _ptr == 0:
            self._ptr_owner = True
            self._ptr = <ccudart.cudaPitchedPtr *>calloc(1, sizeof(ccudart.cudaPitchedPtr))
            if self._ptr is NULL:
                raise MemoryError('Failed to allocate length x size memory: 1x' + str(sizeof(ccudart.cudaPitchedPtr)))
        else:
            self._ptr_owner = False
            self._ptr = <ccudart.cudaPitchedPtr *>_ptr
    def __init__(self, void_ptr _ptr = 0):
        pass
    def __dealloc__(self):
        if self._ptr_owner is True and self._ptr is not NULL:
            free(self._ptr)
        pass
    def getPtr(self):
        return <void_ptr>self._ptr
    def __repr__(self):
        if self._ptr is not NULL:
            str_list = []
            str_list += ['ptr : ' + hex(self.ptr)]
            str_list += ['pitch : ' + str(self.pitch)]
            str_list += ['xsize : ' + str(self.xsize)]
            str_list += ['ysize : ' + str(self.ysize)]
            return '\n'.join(str_list)
        else:
            return ''

    @property
    def ptr(self):
        return <void_ptr>self._ptr[0].ptr
    @ptr.setter
    def ptr(self, ptr):
        _cptr = utils.HelperInputVoidPtr(ptr)
        self._ptr[0].ptr = <void*><void_ptr>_cptr.cptr
    @property
    def pitch(self):
        return self._ptr[0].pitch
    @pitch.setter
    def pitch(self, size_t pitch):
        pass
        self._ptr[0].pitch = pitch
    @property
    def xsize(self):
        return self._ptr[0].xsize
    @xsize.setter
    def xsize(self, size_t xsize):
        pass
        self._ptr[0].xsize = xsize
    @property
    def ysize(self):
        return self._ptr[0].ysize
    @ysize.setter
    def ysize(self, size_t ysize):
        pass
        self._ptr[0].ysize = ysize

cdef class cudaExtent:
    def __cinit__(self, void_ptr _ptr = 0):
        if _ptr == 0:
            self._ptr_owner = True
            self._ptr = <ccudart.cudaExtent *>calloc(1, sizeof(ccudart.cudaExtent))
            if self._ptr is NULL:
                raise MemoryError('Failed to allocate length x size memory: 1x' + str(sizeof(ccudart.cudaExtent)))
        else:
            self._ptr_owner = False
            self._ptr = <ccudart.cudaExtent *>_ptr
    def __init__(self, void_ptr _ptr = 0):
        pass
    def __dealloc__(self):
        if self._ptr_owner is True and self._ptr is not NULL:
            free(self._ptr)
        pass
    def getPtr(self):
        return <void_ptr>self._ptr
    def __repr__(self):
        if self._ptr is not NULL:
            str_list = []
            str_list += ['width : ' + str(self.width)]
            str_list += ['height : ' + str(self.height)]
            str_list += ['depth : ' + str(self.depth)]
            return '\n'.join(str_list)
        else:
            return ''

    @property
    def width(self):
        return self._ptr[0].width
    @width.setter
    def width(self, size_t width):
        pass
        self._ptr[0].width = width
    @property
    def height(self):
        return self._ptr[0].height
    @height.setter
    def height(self, size_t height):
        pass
        self._ptr[0].height = height
    @property
    def depth(self):
        return self._ptr[0].depth
    @depth.setter
    def depth(self, size_t depth):
        pass
        self._ptr[0].depth = depth

cdef class cudaPos:
    def __cinit__(self, void_ptr _ptr = 0):
        if _ptr == 0:
            self._ptr_owner = True
            self._ptr = <ccudart.cudaPos *>calloc(1, sizeof(ccudart.cudaPos))
            if self._ptr is NULL:
                raise MemoryError('Failed to allocate length x size memory: 1x' + str(sizeof(ccudart.cudaPos)))
        else:
            self._ptr_owner = False
            self._ptr = <ccudart.cudaPos *>_ptr
    def __init__(self, void_ptr _ptr = 0):
        pass
    def __dealloc__(self):
        if self._ptr_owner is True and self._ptr is not NULL:
            free(self._ptr)
        pass
    def getPtr(self):
        return <void_ptr>self._ptr
    def __repr__(self):
        if self._ptr is not NULL:
            str_list = []
            str_list += ['x : ' + str(self.x)]
            str_list += ['y : ' + str(self.y)]
            str_list += ['z : ' + str(self.z)]
            return '\n'.join(str_list)
        else:
            return ''

    @property
    def x(self):
        return self._ptr[0].x
    @x.setter
    def x(self, size_t x):
        pass
        self._ptr[0].x = x
    @property
    def y(self):
        return self._ptr[0].y
    @y.setter
    def y(self, size_t y):
        pass
        self._ptr[0].y = y
    @property
    def z(self):
        return self._ptr[0].z
    @z.setter
    def z(self, size_t z):
        pass
        self._ptr[0].z = z

cdef class cudaMemcpy3DParms:
    def __cinit__(self, void_ptr _ptr = 0):
        if _ptr == 0:
            self._ptr_owner = True
            self._ptr = <ccudart.cudaMemcpy3DParms *>calloc(1, sizeof(ccudart.cudaMemcpy3DParms))
            if self._ptr is NULL:
                raise MemoryError('Failed to allocate length x size memory: 1x' + str(sizeof(ccudart.cudaMemcpy3DParms)))
        else:
            self._ptr_owner = False
            self._ptr = <ccudart.cudaMemcpy3DParms *>_ptr
    def __init__(self, void_ptr _ptr = 0):
        self._srcArray = cudaArray_t(_ptr=<void_ptr>&self._ptr[0].srcArray)
        self._srcPos = cudaPos(_ptr=<void_ptr>&self._ptr[0].srcPos)
        self._srcPtr = cudaPitchedPtr(_ptr=<void_ptr>&self._ptr[0].srcPtr)
        self._dstArray = cudaArray_t(_ptr=<void_ptr>&self._ptr[0].dstArray)
        self._dstPos = cudaPos(_ptr=<void_ptr>&self._ptr[0].dstPos)
        self._dstPtr = cudaPitchedPtr(_ptr=<void_ptr>&self._ptr[0].dstPtr)
        self._extent = cudaExtent(_ptr=<void_ptr>&self._ptr[0].extent)
    def __dealloc__(self):
        if self._ptr_owner is True and self._ptr is not NULL:
            free(self._ptr)
        pass
    def getPtr(self):
        return <void_ptr>self._ptr
    def __repr__(self):
        if self._ptr is not NULL:
            str_list = []
            str_list += ['srcArray : ' + str(self.srcArray)]
            str_list += ['srcPos :\n' + '\n'.join(['    ' + line for line in str(self.srcPos).splitlines()])]
            str_list += ['srcPtr :\n' + '\n'.join(['    ' + line for line in str(self.srcPtr).splitlines()])]
            str_list += ['dstArray : ' + str(self.dstArray)]
            str_list += ['dstPos :\n' + '\n'.join(['    ' + line for line in str(self.dstPos).splitlines()])]
            str_list += ['dstPtr :\n' + '\n'.join(['    ' + line for line in str(self.dstPtr).splitlines()])]
            str_list += ['extent :\n' + '\n'.join(['    ' + line for line in str(self.extent).splitlines()])]
            str_list += ['kind : ' + str(self.kind)]
            return '\n'.join(str_list)
        else:
            return ''

    @property
    def srcArray(self):
        return self._srcArray
    @srcArray.setter
    def srcArray(self, srcArray : cudaArray_t):
        pass
        self._srcArray._ptr[0] = <ccudart.cudaArray_t> NULL if srcArray == None else (<cudaArray_t>srcArray)._ptr[0]
    @property
    def srcPos(self):
        return self._srcPos
    @srcPos.setter
    def srcPos(self, srcPos not None : cudaPos):
        pass
        for _attr in dir(srcPos):
            if _attr == 'getPtr':
                continue
            if not _attr.startswith('_'):
                setattr(self._srcPos, _attr, getattr(srcPos, _attr))
    @property
    def srcPtr(self):
        return self._srcPtr
    @srcPtr.setter
    def srcPtr(self, srcPtr not None : cudaPitchedPtr):
        pass
        for _attr in dir(srcPtr):
            if _attr == 'getPtr':
                continue
            if not _attr.startswith('_'):
                setattr(self._srcPtr, _attr, getattr(srcPtr, _attr))
    @property
    def dstArray(self):
        return self._dstArray
    @dstArray.setter
    def dstArray(self, dstArray : cudaArray_t):
        pass
        self._dstArray._ptr[0] = <ccudart.cudaArray_t> NULL if dstArray == None else (<cudaArray_t>dstArray)._ptr[0]
    @property
    def dstPos(self):
        return self._dstPos
    @dstPos.setter
    def dstPos(self, dstPos not None : cudaPos):
        pass
        for _attr in dir(dstPos):
            if _attr == 'getPtr':
                continue
            if not _attr.startswith('_'):
                setattr(self._dstPos, _attr, getattr(dstPos, _attr))
    @property
    def dstPtr(self):
        return self._dstPtr
    @dstPtr.setter
    def dstPtr(self, dstPtr not None : cudaPitchedPtr):
        pass
        for _attr in dir(dstPtr):
            if _attr == 'getPtr':
                continue
            if not _attr.startswith('_'):
                setattr(self._dstPtr, _attr, getattr(dstPtr, _attr))
    @property
    def extent(self):
        return self._extent
    @extent.setter
    def extent(self, extent not None : cudaExtent):
        pass
        for _attr in dir(extent):
            if _attr == 'getPtr':
                continue
            if not _attr.startswith('_'):
                setattr(self._extent, _attr, getattr(extent, _attr))
    @property
    def kind(self):
        return cudaMemcpyKind(self._ptr[0].kind)
    @kind.setter
    def kind(self, kind not None : cudaMemcpyKind):
        pass
        self._ptr[0].kind = kind.value

cdef class cudaMemcpy3DPeerParms:
    def __cinit__(self, void_ptr _ptr = 0):
        if _ptr == 0:
            self._ptr_owner = True
            self._ptr = <ccudart.cudaMemcpy3DPeerParms *>calloc(1, sizeof(ccudart.cudaMemcpy3DPeerParms))
            if self._ptr is NULL:
                raise MemoryError('Failed to allocate length x size memory: 1x' + str(sizeof(ccudart.cudaMemcpy3DPeerParms)))
        else:
            self._ptr_owner = False
            self._ptr = <ccudart.cudaMemcpy3DPeerParms *>_ptr
    def __init__(self, void_ptr _ptr = 0):
        self._srcArray = cudaArray_t(_ptr=<void_ptr>&self._ptr[0].srcArray)
        self._srcPos = cudaPos(_ptr=<void_ptr>&self._ptr[0].srcPos)
        self._srcPtr = cudaPitchedPtr(_ptr=<void_ptr>&self._ptr[0].srcPtr)
        self._dstArray = cudaArray_t(_ptr=<void_ptr>&self._ptr[0].dstArray)
        self._dstPos = cudaPos(_ptr=<void_ptr>&self._ptr[0].dstPos)
        self._dstPtr = cudaPitchedPtr(_ptr=<void_ptr>&self._ptr[0].dstPtr)
        self._extent = cudaExtent(_ptr=<void_ptr>&self._ptr[0].extent)
    def __dealloc__(self):
        if self._ptr_owner is True and self._ptr is not NULL:
            free(self._ptr)
        pass
    def getPtr(self):
        return <void_ptr>self._ptr
    def __repr__(self):
        if self._ptr is not NULL:
            str_list = []
            str_list += ['srcArray : ' + str(self.srcArray)]
            str_list += ['srcPos :\n' + '\n'.join(['    ' + line for line in str(self.srcPos).splitlines()])]
            str_list += ['srcPtr :\n' + '\n'.join(['    ' + line for line in str(self.srcPtr).splitlines()])]
            str_list += ['srcDevice : ' + str(self.srcDevice)]
            str_list += ['dstArray : ' + str(self.dstArray)]
            str_list += ['dstPos :\n' + '\n'.join(['    ' + line for line in str(self.dstPos).splitlines()])]
            str_list += ['dstPtr :\n' + '\n'.join(['    ' + line for line in str(self.dstPtr).splitlines()])]
            str_list += ['dstDevice : ' + str(self.dstDevice)]
            str_list += ['extent :\n' + '\n'.join(['    ' + line for line in str(self.extent).splitlines()])]
            return '\n'.join(str_list)
        else:
            return ''

    @property
    def srcArray(self):
        return self._srcArray
    @srcArray.setter
    def srcArray(self, srcArray : cudaArray_t):
        pass
        self._srcArray._ptr[0] = <ccudart.cudaArray_t> NULL if srcArray == None else (<cudaArray_t>srcArray)._ptr[0]
    @property
    def srcPos(self):
        return self._srcPos
    @srcPos.setter
    def srcPos(self, srcPos not None : cudaPos):
        pass
        for _attr in dir(srcPos):
            if _attr == 'getPtr':
                continue
            if not _attr.startswith('_'):
                setattr(self._srcPos, _attr, getattr(srcPos, _attr))
    @property
    def srcPtr(self):
        return self._srcPtr
    @srcPtr.setter
    def srcPtr(self, srcPtr not None : cudaPitchedPtr):
        pass
        for _attr in dir(srcPtr):
            if _attr == 'getPtr':
                continue
            if not _attr.startswith('_'):
                setattr(self._srcPtr, _attr, getattr(srcPtr, _attr))
    @property
    def srcDevice(self):
        return self._ptr[0].srcDevice
    @srcDevice.setter
    def srcDevice(self, int srcDevice):
        pass
        self._ptr[0].srcDevice = srcDevice
    @property
    def dstArray(self):
        return self._dstArray
    @dstArray.setter
    def dstArray(self, dstArray : cudaArray_t):
        pass
        self._dstArray._ptr[0] = <ccudart.cudaArray_t> NULL if dstArray == None else (<cudaArray_t>dstArray)._ptr[0]
    @property
    def dstPos(self):
        return self._dstPos
    @dstPos.setter
    def dstPos(self, dstPos not None : cudaPos):
        pass
        for _attr in dir(dstPos):
            if _attr == 'getPtr':
                continue
            if not _attr.startswith('_'):
                setattr(self._dstPos, _attr, getattr(dstPos, _attr))
    @property
    def dstPtr(self):
        return self._dstPtr
    @dstPtr.setter
    def dstPtr(self, dstPtr not None : cudaPitchedPtr):
        pass
        for _attr in dir(dstPtr):
            if _attr == 'getPtr':
                continue
            if not _attr.startswith('_'):
                setattr(self._dstPtr, _attr, getattr(dstPtr, _attr))
    @property
    def dstDevice(self):
        return self._ptr[0].dstDevice
    @dstDevice.setter
    def dstDevice(self, int dstDevice):
        pass
        self._ptr[0].dstDevice = dstDevice
    @property
    def extent(self):
        return self._extent
    @extent.setter
    def extent(self, extent not None : cudaExtent):
        pass
        for _attr in dir(extent):
            if _attr == 'getPtr':
                continue
            if not _attr.startswith('_'):
                setattr(self._extent, _attr, getattr(extent, _attr))

cdef class cudaMemsetParams:
    def __cinit__(self, void_ptr _ptr = 0):
        if _ptr == 0:
            self._ptr_owner = True
            self._ptr = <ccudart.cudaMemsetParams *>calloc(1, sizeof(ccudart.cudaMemsetParams))
            if self._ptr is NULL:
                raise MemoryError('Failed to allocate length x size memory: 1x' + str(sizeof(ccudart.cudaMemsetParams)))
        else:
            self._ptr_owner = False
            self._ptr = <ccudart.cudaMemsetParams *>_ptr
    def __init__(self, void_ptr _ptr = 0):
        pass
    def __dealloc__(self):
        if self._ptr_owner is True and self._ptr is not NULL:
            free(self._ptr)
        pass
    def getPtr(self):
        return <void_ptr>self._ptr
    def __repr__(self):
        if self._ptr is not NULL:
            str_list = []
            str_list += ['dst : ' + hex(self.dst)]
            str_list += ['pitch : ' + str(self.pitch)]
            str_list += ['value : ' + str(self.value)]
            str_list += ['elementSize : ' + str(self.elementSize)]
            str_list += ['width : ' + str(self.width)]
            str_list += ['height : ' + str(self.height)]
            return '\n'.join(str_list)
        else:
            return ''

    @property
    def dst(self):
        return <void_ptr>self._ptr[0].dst
    @dst.setter
    def dst(self, dst):
        _cdst = utils.HelperInputVoidPtr(dst)
        self._ptr[0].dst = <void*><void_ptr>_cdst.cptr
    @property
    def pitch(self):
        return self._ptr[0].pitch
    @pitch.setter
    def pitch(self, size_t pitch):
        pass
        self._ptr[0].pitch = pitch
    @property
    def value(self):
        return self._ptr[0].value
    @value.setter
    def value(self, unsigned int value):
        pass
        self._ptr[0].value = value
    @property
    def elementSize(self):
        return self._ptr[0].elementSize
    @elementSize.setter
    def elementSize(self, unsigned int elementSize):
        pass
        self._ptr[0].elementSize = elementSize
    @property
    def width(self):
        return self._ptr[0].width
    @width.setter
    def width(self, size_t width):
        pass
        self._ptr[0].width = width
    @property
    def height(self):
        return self._ptr[0].height
    @height.setter
    def height(self, size_t height):
        pass
        self._ptr[0].height = height

cdef class cudaAccessPolicyWindow:
    def __cinit__(self, void_ptr _ptr = 0):
        if _ptr == 0:
            self._ptr_owner = True
            self._ptr = <ccudart.cudaAccessPolicyWindow *>calloc(1, sizeof(ccudart.cudaAccessPolicyWindow))
            if self._ptr is NULL:
                raise MemoryError('Failed to allocate length x size memory: 1x' + str(sizeof(ccudart.cudaAccessPolicyWindow)))
        else:
            self._ptr_owner = False
            self._ptr = <ccudart.cudaAccessPolicyWindow *>_ptr
    def __init__(self, void_ptr _ptr = 0):
        pass
    def __dealloc__(self):
        if self._ptr_owner is True and self._ptr is not NULL:
            free(self._ptr)
        pass
    def getPtr(self):
        return <void_ptr>self._ptr
    def __repr__(self):
        if self._ptr is not NULL:
            str_list = []
            str_list += ['base_ptr : ' + hex(self.base_ptr)]
            str_list += ['num_bytes : ' + str(self.num_bytes)]
            str_list += ['hitRatio : ' + str(self.hitRatio)]
            str_list += ['hitProp : ' + str(self.hitProp)]
            str_list += ['missProp : ' + str(self.missProp)]
            return '\n'.join(str_list)
        else:
            return ''

    @property
    def base_ptr(self):
        return <void_ptr>self._ptr[0].base_ptr
    @base_ptr.setter
    def base_ptr(self, base_ptr):
        _cbase_ptr = utils.HelperInputVoidPtr(base_ptr)
        self._ptr[0].base_ptr = <void*><void_ptr>_cbase_ptr.cptr
    @property
    def num_bytes(self):
        return self._ptr[0].num_bytes
    @num_bytes.setter
    def num_bytes(self, size_t num_bytes):
        pass
        self._ptr[0].num_bytes = num_bytes
    @property
    def hitRatio(self):
        return self._ptr[0].hitRatio
    @hitRatio.setter
    def hitRatio(self, float hitRatio):
        pass
        self._ptr[0].hitRatio = hitRatio
    @property
    def hitProp(self):
        return cudaAccessProperty(self._ptr[0].hitProp)
    @hitProp.setter
    def hitProp(self, hitProp not None : cudaAccessProperty):
        pass
        self._ptr[0].hitProp = hitProp.value
    @property
    def missProp(self):
        return cudaAccessProperty(self._ptr[0].missProp)
    @missProp.setter
    def missProp(self, missProp not None : cudaAccessProperty):
        pass
        self._ptr[0].missProp = missProp.value

cdef class cudaHostNodeParams:
    def __cinit__(self, void_ptr _ptr = 0):
        if _ptr == 0:
            self._ptr_owner = True
            self._ptr = <ccudart.cudaHostNodeParams *>calloc(1, sizeof(ccudart.cudaHostNodeParams))
            if self._ptr is NULL:
                raise MemoryError('Failed to allocate length x size memory: 1x' + str(sizeof(ccudart.cudaHostNodeParams)))
        else:
            self._ptr_owner = False
            self._ptr = <ccudart.cudaHostNodeParams *>_ptr
    def __init__(self, void_ptr _ptr = 0):
        self._fn = cudaHostFn_t(_ptr=<void_ptr>&self._ptr[0].fn)
    def __dealloc__(self):
        if self._ptr_owner is True and self._ptr is not NULL:
            free(self._ptr)
        pass
    def getPtr(self):
        return <void_ptr>self._ptr
    def __repr__(self):
        if self._ptr is not NULL:
            str_list = []
            str_list += ['fn : ' + str(self.fn)]
            str_list += ['userData : ' + hex(self.userData)]
            return '\n'.join(str_list)
        else:
            return ''

    @property
    def fn(self):
        return self._fn
    @fn.setter
    def fn(self, fn : cudaHostFn_t):
        pass
        self._fn._ptr[0] = <ccudart.cudaHostFn_t> NULL if fn == None else (<cudaHostFn_t>fn)._ptr[0]
    @property
    def userData(self):
        return <void_ptr>self._ptr[0].userData
    @userData.setter
    def userData(self, userData):
        _cuserData = utils.HelperInputVoidPtr(userData)
        self._ptr[0].userData = <void*><void_ptr>_cuserData.cptr

cdef class cudaStreamAttrValue:
    def __cinit__(self, void_ptr _ptr = 0):
        if _ptr == 0:
            self._ptr_owner = True
            self._ptr = <ccudart.cudaStreamAttrValue *>calloc(1, sizeof(ccudart.cudaStreamAttrValue))
            if self._ptr is NULL:
                raise MemoryError('Failed to allocate length x size memory: 1x' + str(sizeof(ccudart.cudaStreamAttrValue)))
        else:
            self._ptr_owner = False
            self._ptr = <ccudart.cudaStreamAttrValue *>_ptr
    def __init__(self, void_ptr _ptr = 0):
        self._accessPolicyWindow = cudaAccessPolicyWindow(_ptr=<void_ptr>&self._ptr[0].accessPolicyWindow)
    def __dealloc__(self):
        if self._ptr_owner is True and self._ptr is not NULL:
            free(self._ptr)
        pass
    def getPtr(self):
        return <void_ptr>self._ptr
    def __repr__(self):
        if self._ptr is not NULL:
            str_list = []
            str_list += ['accessPolicyWindow :\n' + '\n'.join(['    ' + line for line in str(self.accessPolicyWindow).splitlines()])]
            str_list += ['syncPolicy : ' + str(self.syncPolicy)]
            return '\n'.join(str_list)
        else:
            return ''

    @property
    def accessPolicyWindow(self):
        return self._accessPolicyWindow
    @accessPolicyWindow.setter
    def accessPolicyWindow(self, accessPolicyWindow not None : cudaAccessPolicyWindow):
        pass
        for _attr in dir(accessPolicyWindow):
            if _attr == 'getPtr':
                continue
            if not _attr.startswith('_'):
                setattr(self._accessPolicyWindow, _attr, getattr(accessPolicyWindow, _attr))
    @property
    def syncPolicy(self):
        return cudaSynchronizationPolicy(self._ptr[0].syncPolicy)
    @syncPolicy.setter
    def syncPolicy(self, syncPolicy not None : cudaSynchronizationPolicy):
        pass
        self._ptr[0].syncPolicy = syncPolicy.value

cdef class cudaKernelNodeAttrValue:
    def __cinit__(self, void_ptr _ptr = 0):
        if _ptr == 0:
            self._ptr_owner = True
            self._ptr = <ccudart.cudaKernelNodeAttrValue *>calloc(1, sizeof(ccudart.cudaKernelNodeAttrValue))
            if self._ptr is NULL:
                raise MemoryError('Failed to allocate length x size memory: 1x' + str(sizeof(ccudart.cudaKernelNodeAttrValue)))
        else:
            self._ptr_owner = False
            self._ptr = <ccudart.cudaKernelNodeAttrValue *>_ptr
    def __init__(self, void_ptr _ptr = 0):
        self._accessPolicyWindow = cudaAccessPolicyWindow(_ptr=<void_ptr>&self._ptr[0].accessPolicyWindow)
    def __dealloc__(self):
        if self._ptr_owner is True and self._ptr is not NULL:
            free(self._ptr)
        pass
    def getPtr(self):
        return <void_ptr>self._ptr
    def __repr__(self):
        if self._ptr is not NULL:
            str_list = []
            str_list += ['accessPolicyWindow :\n' + '\n'.join(['    ' + line for line in str(self.accessPolicyWindow).splitlines()])]
            str_list += ['cooperative : ' + str(self.cooperative)]
            return '\n'.join(str_list)
        else:
            return ''

    @property
    def accessPolicyWindow(self):
        return self._accessPolicyWindow
    @accessPolicyWindow.setter
    def accessPolicyWindow(self, accessPolicyWindow not None : cudaAccessPolicyWindow):
        pass
        for _attr in dir(accessPolicyWindow):
            if _attr == 'getPtr':
                continue
            if not _attr.startswith('_'):
                setattr(self._accessPolicyWindow, _attr, getattr(accessPolicyWindow, _attr))
    @property
    def cooperative(self):
        return self._ptr[0].cooperative
    @cooperative.setter
    def cooperative(self, int cooperative):
        pass
        self._ptr[0].cooperative = cooperative

cdef class _cudaResourceDesc_res_res_array_s:
    def __cinit__(self, void_ptr _ptr):
        self._ptr = <ccudart.cudaResourceDesc *>_ptr
    def __init__(self, void_ptr _ptr):
        self._array = cudaArray_t(_ptr=<void_ptr>&self._ptr[0].res.array.array)
    def getPtr(self):
        return <void_ptr>self._ptr
    def __repr__(self):
        if self._ptr is not NULL:
            str_list = []
            str_list += ['array : ' + str(self.array)]
            return '\n'.join(str_list)
        else:
            return ''

    @property
    def array(self):
        return self._array
    @array.setter
    def array(self, array : cudaArray_t):
        pass
        self._array._ptr[0] = <ccudart.cudaArray_t> NULL if array == None else (<cudaArray_t>array)._ptr[0]

cdef class _cudaResourceDesc_res_res_mipmap_s:
    def __cinit__(self, void_ptr _ptr):
        self._ptr = <ccudart.cudaResourceDesc *>_ptr
    def __init__(self, void_ptr _ptr):
        self._mipmap = cudaMipmappedArray_t(_ptr=<void_ptr>&self._ptr[0].res.mipmap.mipmap)
    def getPtr(self):
        return <void_ptr>self._ptr
    def __repr__(self):
        if self._ptr is not NULL:
            str_list = []
            str_list += ['mipmap : ' + str(self.mipmap)]
            return '\n'.join(str_list)
        else:
            return ''

    @property
    def mipmap(self):
        return self._mipmap
    @mipmap.setter
    def mipmap(self, mipmap : cudaMipmappedArray_t):
        pass
        self._mipmap._ptr[0] = <ccudart.cudaMipmappedArray_t> NULL if mipmap == None else (<cudaMipmappedArray_t>mipmap)._ptr[0]

cdef class _cudaResourceDesc_res_res_linear_s:
    def __cinit__(self, void_ptr _ptr):
        self._ptr = <ccudart.cudaResourceDesc *>_ptr
    def __init__(self, void_ptr _ptr):
        self._desc = cudaChannelFormatDesc(_ptr=<void_ptr>&self._ptr[0].res.linear.desc)
    def getPtr(self):
        return <void_ptr>self._ptr
    def __repr__(self):
        if self._ptr is not NULL:
            str_list = []
            str_list += ['devPtr : ' + hex(self.devPtr)]
            str_list += ['desc :\n' + '\n'.join(['    ' + line for line in str(self.desc).splitlines()])]
            str_list += ['sizeInBytes : ' + str(self.sizeInBytes)]
            return '\n'.join(str_list)
        else:
            return ''

    @property
    def devPtr(self):
        return <void_ptr>self._ptr[0].res.linear.devPtr
    @devPtr.setter
    def devPtr(self, devPtr):
        _cdevPtr = utils.HelperInputVoidPtr(devPtr)
        self._ptr[0].res.linear.devPtr = <void*><void_ptr>_cdevPtr.cptr
    @property
    def desc(self):
        return self._desc
    @desc.setter
    def desc(self, desc not None : cudaChannelFormatDesc):
        pass
        for _attr in dir(desc):
            if _attr == 'getPtr':
                continue
            if not _attr.startswith('_'):
                setattr(self._desc, _attr, getattr(desc, _attr))
    @property
    def sizeInBytes(self):
        return self._ptr[0].res.linear.sizeInBytes
    @sizeInBytes.setter
    def sizeInBytes(self, size_t sizeInBytes):
        pass
        self._ptr[0].res.linear.sizeInBytes = sizeInBytes

cdef class _cudaResourceDesc_res_res_pitch2D_s:
    def __cinit__(self, void_ptr _ptr):
        self._ptr = <ccudart.cudaResourceDesc *>_ptr
    def __init__(self, void_ptr _ptr):
        self._desc = cudaChannelFormatDesc(_ptr=<void_ptr>&self._ptr[0].res.pitch2D.desc)
    def getPtr(self):
        return <void_ptr>self._ptr
    def __repr__(self):
        if self._ptr is not NULL:
            str_list = []
            str_list += ['devPtr : ' + hex(self.devPtr)]
            str_list += ['desc :\n' + '\n'.join(['    ' + line for line in str(self.desc).splitlines()])]
            str_list += ['width : ' + str(self.width)]
            str_list += ['height : ' + str(self.height)]
            str_list += ['pitchInBytes : ' + str(self.pitchInBytes)]
            return '\n'.join(str_list)
        else:
            return ''

    @property
    def devPtr(self):
        return <void_ptr>self._ptr[0].res.pitch2D.devPtr
    @devPtr.setter
    def devPtr(self, devPtr):
        _cdevPtr = utils.HelperInputVoidPtr(devPtr)
        self._ptr[0].res.pitch2D.devPtr = <void*><void_ptr>_cdevPtr.cptr
    @property
    def desc(self):
        return self._desc
    @desc.setter
    def desc(self, desc not None : cudaChannelFormatDesc):
        pass
        for _attr in dir(desc):
            if _attr == 'getPtr':
                continue
            if not _attr.startswith('_'):
                setattr(self._desc, _attr, getattr(desc, _attr))
    @property
    def width(self):
        return self._ptr[0].res.pitch2D.width
    @width.setter
    def width(self, size_t width):
        pass
        self._ptr[0].res.pitch2D.width = width
    @property
    def height(self):
        return self._ptr[0].res.pitch2D.height
    @height.setter
    def height(self, size_t height):
        pass
        self._ptr[0].res.pitch2D.height = height
    @property
    def pitchInBytes(self):
        return self._ptr[0].res.pitch2D.pitchInBytes
    @pitchInBytes.setter
    def pitchInBytes(self, size_t pitchInBytes):
        pass
        self._ptr[0].res.pitch2D.pitchInBytes = pitchInBytes

cdef class _cudaResourceDesc_res_u:
    def __cinit__(self, void_ptr _ptr):
        self._ptr = <ccudart.cudaResourceDesc *>_ptr
    def __init__(self, void_ptr _ptr):
        self._array = _cudaResourceDesc_res_res_array_s(_ptr=<void_ptr>self._ptr)
        self._mipmap = _cudaResourceDesc_res_res_mipmap_s(_ptr=<void_ptr>self._ptr)
        self._linear = _cudaResourceDesc_res_res_linear_s(_ptr=<void_ptr>self._ptr)
        self._pitch2D = _cudaResourceDesc_res_res_pitch2D_s(_ptr=<void_ptr>self._ptr)
    def getPtr(self):
        return <void_ptr>self._ptr
    def __repr__(self):
        if self._ptr is not NULL:
            str_list = []
            str_list += ['array :\n' + '\n'.join(['    ' + line for line in str(self.array).splitlines()])]
            str_list += ['mipmap :\n' + '\n'.join(['    ' + line for line in str(self.mipmap).splitlines()])]
            str_list += ['linear :\n' + '\n'.join(['    ' + line for line in str(self.linear).splitlines()])]
            str_list += ['pitch2D :\n' + '\n'.join(['    ' + line for line in str(self.pitch2D).splitlines()])]
            return '\n'.join(str_list)
        else:
            return ''

    @property
    def array(self):
        return self._array
    @array.setter
    def array(self, array not None : _cudaResourceDesc_res_res_array_s):
        pass
        for _attr in dir(array):
            if _attr == 'getPtr':
                continue
            if not _attr.startswith('_'):
                setattr(self._array, _attr, getattr(array, _attr))
    @property
    def mipmap(self):
        return self._mipmap
    @mipmap.setter
    def mipmap(self, mipmap not None : _cudaResourceDesc_res_res_mipmap_s):
        pass
        for _attr in dir(mipmap):
            if _attr == 'getPtr':
                continue
            if not _attr.startswith('_'):
                setattr(self._mipmap, _attr, getattr(mipmap, _attr))
    @property
    def linear(self):
        return self._linear
    @linear.setter
    def linear(self, linear not None : _cudaResourceDesc_res_res_linear_s):
        pass
        for _attr in dir(linear):
            if _attr == 'getPtr':
                continue
            if not _attr.startswith('_'):
                setattr(self._linear, _attr, getattr(linear, _attr))
    @property
    def pitch2D(self):
        return self._pitch2D
    @pitch2D.setter
    def pitch2D(self, pitch2D not None : _cudaResourceDesc_res_res_pitch2D_s):
        pass
        for _attr in dir(pitch2D):
            if _attr == 'getPtr':
                continue
            if not _attr.startswith('_'):
                setattr(self._pitch2D, _attr, getattr(pitch2D, _attr))

cdef class cudaResourceDesc:
    def __cinit__(self, void_ptr _ptr = 0):
        if _ptr == 0:
            self._ptr_owner = True
            self._ptr = <ccudart.cudaResourceDesc *>calloc(1, sizeof(ccudart.cudaResourceDesc))
            if self._ptr is NULL:
                raise MemoryError('Failed to allocate length x size memory: 1x' + str(sizeof(ccudart.cudaResourceDesc)))
        else:
            self._ptr_owner = False
            self._ptr = <ccudart.cudaResourceDesc *>_ptr
    def __init__(self, void_ptr _ptr = 0):
        self._res = _cudaResourceDesc_res_u(_ptr=<void_ptr>self._ptr)
    def __dealloc__(self):
        if self._ptr_owner is True and self._ptr is not NULL:
            free(self._ptr)
        pass
    def getPtr(self):
        return <void_ptr>self._ptr
    def __repr__(self):
        if self._ptr is not NULL:
            str_list = []
            str_list += ['resType : ' + str(self.resType)]
            str_list += ['res :\n' + '\n'.join(['    ' + line for line in str(self.res).splitlines()])]
            return '\n'.join(str_list)
        else:
            return ''

    @property
    def resType(self):
        return cudaResourceType(self._ptr[0].resType)
    @resType.setter
    def resType(self, resType not None : cudaResourceType):
        pass
        self._ptr[0].resType = resType.value
    @property
    def res(self):
        return self._res
    @res.setter
    def res(self, res not None : _cudaResourceDesc_res_u):
        pass
        for _attr in dir(res):
            if _attr == 'getPtr':
                continue
            if not _attr.startswith('_'):
                setattr(self._res, _attr, getattr(res, _attr))

cdef class cudaResourceViewDesc:
    def __cinit__(self, void_ptr _ptr = 0):
        if _ptr == 0:
            self._ptr_owner = True
            self._ptr = <ccudart.cudaResourceViewDesc *>calloc(1, sizeof(ccudart.cudaResourceViewDesc))
            if self._ptr is NULL:
                raise MemoryError('Failed to allocate length x size memory: 1x' + str(sizeof(ccudart.cudaResourceViewDesc)))
        else:
            self._ptr_owner = False
            self._ptr = <ccudart.cudaResourceViewDesc *>_ptr
    def __init__(self, void_ptr _ptr = 0):
        pass
    def __dealloc__(self):
        if self._ptr_owner is True and self._ptr is not NULL:
            free(self._ptr)
        pass
    def getPtr(self):
        return <void_ptr>self._ptr
    def __repr__(self):
        if self._ptr is not NULL:
            str_list = []
            str_list += ['format : ' + str(self.format)]
            str_list += ['width : ' + str(self.width)]
            str_list += ['height : ' + str(self.height)]
            str_list += ['depth : ' + str(self.depth)]
            str_list += ['firstMipmapLevel : ' + str(self.firstMipmapLevel)]
            str_list += ['lastMipmapLevel : ' + str(self.lastMipmapLevel)]
            str_list += ['firstLayer : ' + str(self.firstLayer)]
            str_list += ['lastLayer : ' + str(self.lastLayer)]
            return '\n'.join(str_list)
        else:
            return ''

    @property
    def format(self):
        return cudaResourceViewFormat(self._ptr[0].format)
    @format.setter
    def format(self, format not None : cudaResourceViewFormat):
        pass
        self._ptr[0].format = format.value
    @property
    def width(self):
        return self._ptr[0].width
    @width.setter
    def width(self, size_t width):
        pass
        self._ptr[0].width = width
    @property
    def height(self):
        return self._ptr[0].height
    @height.setter
    def height(self, size_t height):
        pass
        self._ptr[0].height = height
    @property
    def depth(self):
        return self._ptr[0].depth
    @depth.setter
    def depth(self, size_t depth):
        pass
        self._ptr[0].depth = depth
    @property
    def firstMipmapLevel(self):
        return self._ptr[0].firstMipmapLevel
    @firstMipmapLevel.setter
    def firstMipmapLevel(self, unsigned int firstMipmapLevel):
        pass
        self._ptr[0].firstMipmapLevel = firstMipmapLevel
    @property
    def lastMipmapLevel(self):
        return self._ptr[0].lastMipmapLevel
    @lastMipmapLevel.setter
    def lastMipmapLevel(self, unsigned int lastMipmapLevel):
        pass
        self._ptr[0].lastMipmapLevel = lastMipmapLevel
    @property
    def firstLayer(self):
        return self._ptr[0].firstLayer
    @firstLayer.setter
    def firstLayer(self, unsigned int firstLayer):
        pass
        self._ptr[0].firstLayer = firstLayer
    @property
    def lastLayer(self):
        return self._ptr[0].lastLayer
    @lastLayer.setter
    def lastLayer(self, unsigned int lastLayer):
        pass
        self._ptr[0].lastLayer = lastLayer

cdef class cudaPointerAttributes:
    def __cinit__(self, void_ptr _ptr = 0):
        if _ptr == 0:
            self._ptr_owner = True
            self._ptr = <ccudart.cudaPointerAttributes *>calloc(1, sizeof(ccudart.cudaPointerAttributes))
            if self._ptr is NULL:
                raise MemoryError('Failed to allocate length x size memory: 1x' + str(sizeof(ccudart.cudaPointerAttributes)))
        else:
            self._ptr_owner = False
            self._ptr = <ccudart.cudaPointerAttributes *>_ptr
    def __init__(self, void_ptr _ptr = 0):
        pass
    def __dealloc__(self):
        if self._ptr_owner is True and self._ptr is not NULL:
            free(self._ptr)
        pass
    def getPtr(self):
        return <void_ptr>self._ptr
    def __repr__(self):
        if self._ptr is not NULL:
            str_list = []
            str_list += ['type : ' + str(self.type)]
            str_list += ['device : ' + str(self.device)]
            str_list += ['devicePointer : ' + hex(self.devicePointer)]
            str_list += ['hostPointer : ' + hex(self.hostPointer)]
            return '\n'.join(str_list)
        else:
            return ''

    @property
    def type(self):
        return cudaMemoryType(self._ptr[0].type)
    @type.setter
    def type(self, type not None : cudaMemoryType):
        pass
        self._ptr[0].type = type.value
    @property
    def device(self):
        return self._ptr[0].device
    @device.setter
    def device(self, int device):
        pass
        self._ptr[0].device = device
    @property
    def devicePointer(self):
        return <void_ptr>self._ptr[0].devicePointer
    @devicePointer.setter
    def devicePointer(self, devicePointer):
        _cdevicePointer = utils.HelperInputVoidPtr(devicePointer)
        self._ptr[0].devicePointer = <void*><void_ptr>_cdevicePointer.cptr
    @property
    def hostPointer(self):
        return <void_ptr>self._ptr[0].hostPointer
    @hostPointer.setter
    def hostPointer(self, hostPointer):
        _chostPointer = utils.HelperInputVoidPtr(hostPointer)
        self._ptr[0].hostPointer = <void*><void_ptr>_chostPointer.cptr

cdef class cudaFuncAttributes:
    def __cinit__(self, void_ptr _ptr = 0):
        if _ptr == 0:
            self._ptr_owner = True
            self._ptr = <ccudart.cudaFuncAttributes *>calloc(1, sizeof(ccudart.cudaFuncAttributes))
            if self._ptr is NULL:
                raise MemoryError('Failed to allocate length x size memory: 1x' + str(sizeof(ccudart.cudaFuncAttributes)))
        else:
            self._ptr_owner = False
            self._ptr = <ccudart.cudaFuncAttributes *>_ptr
    def __init__(self, void_ptr _ptr = 0):
        pass
    def __dealloc__(self):
        if self._ptr_owner is True and self._ptr is not NULL:
            free(self._ptr)
        pass
    def getPtr(self):
        return <void_ptr>self._ptr
    def __repr__(self):
        if self._ptr is not NULL:
            str_list = []
            str_list += ['sharedSizeBytes : ' + str(self.sharedSizeBytes)]
            str_list += ['constSizeBytes : ' + str(self.constSizeBytes)]
            str_list += ['localSizeBytes : ' + str(self.localSizeBytes)]
            str_list += ['maxThreadsPerBlock : ' + str(self.maxThreadsPerBlock)]
            str_list += ['numRegs : ' + str(self.numRegs)]
            str_list += ['ptxVersion : ' + str(self.ptxVersion)]
            str_list += ['binaryVersion : ' + str(self.binaryVersion)]
            str_list += ['cacheModeCA : ' + str(self.cacheModeCA)]
            str_list += ['maxDynamicSharedSizeBytes : ' + str(self.maxDynamicSharedSizeBytes)]
            str_list += ['preferredShmemCarveout : ' + str(self.preferredShmemCarveout)]
            return '\n'.join(str_list)
        else:
            return ''

    @property
    def sharedSizeBytes(self):
        return self._ptr[0].sharedSizeBytes
    @sharedSizeBytes.setter
    def sharedSizeBytes(self, size_t sharedSizeBytes):
        pass
        self._ptr[0].sharedSizeBytes = sharedSizeBytes
    @property
    def constSizeBytes(self):
        return self._ptr[0].constSizeBytes
    @constSizeBytes.setter
    def constSizeBytes(self, size_t constSizeBytes):
        pass
        self._ptr[0].constSizeBytes = constSizeBytes
    @property
    def localSizeBytes(self):
        return self._ptr[0].localSizeBytes
    @localSizeBytes.setter
    def localSizeBytes(self, size_t localSizeBytes):
        pass
        self._ptr[0].localSizeBytes = localSizeBytes
    @property
    def maxThreadsPerBlock(self):
        return self._ptr[0].maxThreadsPerBlock
    @maxThreadsPerBlock.setter
    def maxThreadsPerBlock(self, int maxThreadsPerBlock):
        pass
        self._ptr[0].maxThreadsPerBlock = maxThreadsPerBlock
    @property
    def numRegs(self):
        return self._ptr[0].numRegs
    @numRegs.setter
    def numRegs(self, int numRegs):
        pass
        self._ptr[0].numRegs = numRegs
    @property
    def ptxVersion(self):
        return self._ptr[0].ptxVersion
    @ptxVersion.setter
    def ptxVersion(self, int ptxVersion):
        pass
        self._ptr[0].ptxVersion = ptxVersion
    @property
    def binaryVersion(self):
        return self._ptr[0].binaryVersion
    @binaryVersion.setter
    def binaryVersion(self, int binaryVersion):
        pass
        self._ptr[0].binaryVersion = binaryVersion
    @property
    def cacheModeCA(self):
        return self._ptr[0].cacheModeCA
    @cacheModeCA.setter
    def cacheModeCA(self, int cacheModeCA):
        pass
        self._ptr[0].cacheModeCA = cacheModeCA
    @property
    def maxDynamicSharedSizeBytes(self):
        return self._ptr[0].maxDynamicSharedSizeBytes
    @maxDynamicSharedSizeBytes.setter
    def maxDynamicSharedSizeBytes(self, int maxDynamicSharedSizeBytes):
        pass
        self._ptr[0].maxDynamicSharedSizeBytes = maxDynamicSharedSizeBytes
    @property
    def preferredShmemCarveout(self):
        return self._ptr[0].preferredShmemCarveout
    @preferredShmemCarveout.setter
    def preferredShmemCarveout(self, int preferredShmemCarveout):
        pass
        self._ptr[0].preferredShmemCarveout = preferredShmemCarveout

cdef class cudaMemLocation:
    def __cinit__(self, void_ptr _ptr = 0):
        if _ptr == 0:
            self._ptr_owner = True
            self._ptr = <ccudart.cudaMemLocation *>calloc(1, sizeof(ccudart.cudaMemLocation))
            if self._ptr is NULL:
                raise MemoryError('Failed to allocate length x size memory: 1x' + str(sizeof(ccudart.cudaMemLocation)))
        else:
            self._ptr_owner = False
            self._ptr = <ccudart.cudaMemLocation *>_ptr
    def __init__(self, void_ptr _ptr = 0):
        pass
    def __dealloc__(self):
        if self._ptr_owner is True and self._ptr is not NULL:
            free(self._ptr)
        pass
    def getPtr(self):
        return <void_ptr>self._ptr
    def __repr__(self):
        if self._ptr is not NULL:
            str_list = []
            str_list += ['type : ' + str(self.type)]
            str_list += ['id : ' + str(self.id)]
            return '\n'.join(str_list)
        else:
            return ''

    @property
    def type(self):
        return cudaMemLocationType(self._ptr[0].type)
    @type.setter
    def type(self, type not None : cudaMemLocationType):
        pass
        self._ptr[0].type = type.value
    @property
    def id(self):
        return self._ptr[0].id
    @id.setter
    def id(self, int id):
        pass
        self._ptr[0].id = id

cdef class cudaMemAccessDesc:
    def __cinit__(self, void_ptr _ptr = 0):
        if _ptr == 0:
            self._ptr_owner = True
            self._ptr = <ccudart.cudaMemAccessDesc *>calloc(1, sizeof(ccudart.cudaMemAccessDesc))
            if self._ptr is NULL:
                raise MemoryError('Failed to allocate length x size memory: 1x' + str(sizeof(ccudart.cudaMemAccessDesc)))
        else:
            self._ptr_owner = False
            self._ptr = <ccudart.cudaMemAccessDesc *>_ptr
    def __init__(self, void_ptr _ptr = 0):
        self._location = cudaMemLocation(_ptr=<void_ptr>&self._ptr[0].location)
    def __dealloc__(self):
        if self._ptr_owner is True and self._ptr is not NULL:
            free(self._ptr)
        pass
    def getPtr(self):
        return <void_ptr>self._ptr
    def __repr__(self):
        if self._ptr is not NULL:
            str_list = []
            str_list += ['location :\n' + '\n'.join(['    ' + line for line in str(self.location).splitlines()])]
            str_list += ['flags : ' + str(self.flags)]
            return '\n'.join(str_list)
        else:
            return ''

    @property
    def location(self):
        return self._location
    @location.setter
    def location(self, location not None : cudaMemLocation):
        pass
        for _attr in dir(location):
            if _attr == 'getPtr':
                continue
            if not _attr.startswith('_'):
                setattr(self._location, _attr, getattr(location, _attr))
    @property
    def flags(self):
        return cudaMemAccessFlags(self._ptr[0].flags)
    @flags.setter
    def flags(self, flags not None : cudaMemAccessFlags):
        pass
        self._ptr[0].flags = flags.value

cdef class cudaMemPoolProps:
    def __cinit__(self, void_ptr _ptr = 0):
        if _ptr == 0:
            self._ptr_owner = True
            self._ptr = <ccudart.cudaMemPoolProps *>calloc(1, sizeof(ccudart.cudaMemPoolProps))
            if self._ptr is NULL:
                raise MemoryError('Failed to allocate length x size memory: 1x' + str(sizeof(ccudart.cudaMemPoolProps)))
        else:
            self._ptr_owner = False
            self._ptr = <ccudart.cudaMemPoolProps *>_ptr
    def __init__(self, void_ptr _ptr = 0):
        self._location = cudaMemLocation(_ptr=<void_ptr>&self._ptr[0].location)
    def __dealloc__(self):
        if self._ptr_owner is True and self._ptr is not NULL:
            free(self._ptr)
        pass
    def getPtr(self):
        return <void_ptr>self._ptr
    def __repr__(self):
        if self._ptr is not NULL:
            str_list = []
            str_list += ['allocType : ' + str(self.allocType)]
            str_list += ['handleTypes : ' + str(self.handleTypes)]
            str_list += ['location :\n' + '\n'.join(['    ' + line for line in str(self.location).splitlines()])]
            str_list += ['win32SecurityAttributes : ' + hex(self.win32SecurityAttributes)]
            str_list += ['reserved : ' + str(self.reserved)]
            return '\n'.join(str_list)
        else:
            return ''

    @property
    def allocType(self):
        return cudaMemAllocationType(self._ptr[0].allocType)
    @allocType.setter
    def allocType(self, allocType not None : cudaMemAllocationType):
        pass
        self._ptr[0].allocType = allocType.value
    @property
    def handleTypes(self):
        return cudaMemAllocationHandleType(self._ptr[0].handleTypes)
    @handleTypes.setter
    def handleTypes(self, handleTypes not None : cudaMemAllocationHandleType):
        pass
        self._ptr[0].handleTypes = handleTypes.value
    @property
    def location(self):
        return self._location
    @location.setter
    def location(self, location not None : cudaMemLocation):
        pass
        for _attr in dir(location):
            if _attr == 'getPtr':
                continue
            if not _attr.startswith('_'):
                setattr(self._location, _attr, getattr(location, _attr))
    @property
    def win32SecurityAttributes(self):
        return <void_ptr>self._ptr[0].win32SecurityAttributes
    @win32SecurityAttributes.setter
    def win32SecurityAttributes(self, win32SecurityAttributes):
        _cwin32SecurityAttributes = utils.HelperInputVoidPtr(win32SecurityAttributes)
        self._ptr[0].win32SecurityAttributes = <void*><void_ptr>_cwin32SecurityAttributes.cptr
    @property
    def reserved(self):
        return self._ptr[0].reserved
    @reserved.setter
    def reserved(self, reserved):
        pass
        self._ptr[0].reserved = reserved

cdef class cudaMemPoolPtrExportData:
    def __cinit__(self, void_ptr _ptr = 0):
        if _ptr == 0:
            self._ptr_owner = True
            self._ptr = <ccudart.cudaMemPoolPtrExportData *>calloc(1, sizeof(ccudart.cudaMemPoolPtrExportData))
            if self._ptr is NULL:
                raise MemoryError('Failed to allocate length x size memory: 1x' + str(sizeof(ccudart.cudaMemPoolPtrExportData)))
        else:
            self._ptr_owner = False
            self._ptr = <ccudart.cudaMemPoolPtrExportData *>_ptr
    def __init__(self, void_ptr _ptr = 0):
        pass
    def __dealloc__(self):
        if self._ptr_owner is True and self._ptr is not NULL:
            free(self._ptr)
        pass
    def getPtr(self):
        return <void_ptr>self._ptr
    def __repr__(self):
        if self._ptr is not NULL:
            str_list = []
            str_list += ['reserved : ' + str(self.reserved)]
            return '\n'.join(str_list)
        else:
            return ''

    @property
    def reserved(self):
        return self._ptr[0].reserved
    @reserved.setter
    def reserved(self, reserved):
        pass
        self._ptr[0].reserved = reserved

cdef class cudaMemAllocNodeParams:
    def __cinit__(self, void_ptr _ptr = 0):
        if _ptr == 0:
            self._ptr_owner = True
            self._ptr = <ccudart.cudaMemAllocNodeParams *>calloc(1, sizeof(ccudart.cudaMemAllocNodeParams))
            if self._ptr is NULL:
                raise MemoryError('Failed to allocate length x size memory: 1x' + str(sizeof(ccudart.cudaMemAllocNodeParams)))
        else:
            self._ptr_owner = False
            self._ptr = <ccudart.cudaMemAllocNodeParams *>_ptr
    def __init__(self, void_ptr _ptr = 0):
        self._poolProps = cudaMemPoolProps(_ptr=<void_ptr>&self._ptr[0].poolProps)
    def __dealloc__(self):
        if self._ptr_owner is True and self._ptr is not NULL:
            free(self._ptr)
        if self._accessDescs is not NULL:
            free(self._accessDescs)
    def getPtr(self):
        return <void_ptr>self._ptr
    def __repr__(self):
        if self._ptr is not NULL:
            str_list = []
            str_list += ['poolProps :\n' + '\n'.join(['    ' + line for line in str(self.poolProps).splitlines()])]
            str_list += ['accessDescs : ' + str(self.accessDescs)]
            str_list += ['accessDescCount : ' + str(self.accessDescCount)]
            str_list += ['bytesize : ' + str(self.bytesize)]
            str_list += ['dptr : ' + hex(self.dptr)]
            return '\n'.join(str_list)
        else:
            return ''

    @property
    def poolProps(self):
        return self._poolProps
    @poolProps.setter
    def poolProps(self, poolProps not None : cudaMemPoolProps):
        pass
        for _attr in dir(poolProps):
            if _attr == 'getPtr':
                continue
            if not _attr.startswith('_'):
                setattr(self._poolProps, _attr, getattr(poolProps, _attr))
    @property
    def accessDescs(self):
        arrs = [<void_ptr>self._ptr[0].accessDescs + x*sizeof(ccudart.cudaMemAccessDesc) for x in range(self._accessDescs_length)]
        return [cudaMemAccessDesc(_ptr=arr) for arr in arrs]
    @accessDescs.setter
    def accessDescs(self, val):
        if len(val) == 0:
            free(self._accessDescs)
            self._accessDescs_length = 0
            self._ptr[0].accessDescs = NULL
        else:
            if self._accessDescs_length != <size_t>len(val):
                free(self._accessDescs)
                self._accessDescs = <ccudart.cudaMemAccessDesc*> calloc(len(val), sizeof(ccudart.cudaMemAccessDesc))
                if self._accessDescs is NULL:
                    raise MemoryError('Failed to allocate length x size memory: ' + str(len(val)) + 'x' + str(sizeof(ccudart.cudaMemAccessDesc)))
                self._accessDescs_length = <size_t>len(val)
                self._ptr[0].accessDescs = self._accessDescs
            for idx in range(len(val)):
                memcpy(&self._accessDescs[idx], (<cudaMemAccessDesc>val[idx])._ptr, sizeof(ccudart.cudaMemAccessDesc))
    @property
    def accessDescCount(self):
        return self._ptr[0].accessDescCount
    @accessDescCount.setter
    def accessDescCount(self, size_t accessDescCount):
        pass
        self._ptr[0].accessDescCount = accessDescCount
    @property
    def bytesize(self):
        return self._ptr[0].bytesize
    @bytesize.setter
    def bytesize(self, size_t bytesize):
        pass
        self._ptr[0].bytesize = bytesize
    @property
    def dptr(self):
        return <void_ptr>self._ptr[0].dptr
    @dptr.setter
    def dptr(self, dptr):
        _cdptr = utils.HelperInputVoidPtr(dptr)
        self._ptr[0].dptr = <void*><void_ptr>_cdptr.cptr

cdef class CUuuid_st:
    def __cinit__(self, void_ptr _ptr = 0):
        if _ptr == 0:
            self._ptr_owner = True
            self._ptr = <ccudart.CUuuid_st *>calloc(1, sizeof(ccudart.CUuuid_st))
            if self._ptr is NULL:
                raise MemoryError('Failed to allocate length x size memory: 1x' + str(sizeof(ccudart.CUuuid_st)))
        else:
            self._ptr_owner = False
            self._ptr = <ccudart.CUuuid_st *>_ptr
    def __init__(self, void_ptr _ptr = 0):
        pass
    def __dealloc__(self):
        if self._ptr_owner is True and self._ptr is not NULL:
            free(self._ptr)
        pass
    def getPtr(self):
        return <void_ptr>self._ptr
    def __repr__(self):
        if self._ptr is not NULL:
            str_list = []
            str_list += ['bytes : ' + str(self.bytes.hex())]
            return '\n'.join(str_list)
        else:
            return ''

    @property
    def bytes(self):
        return self._ptr[0].bytes

cdef class cudaDeviceProp:
    def __cinit__(self, void_ptr _ptr = 0):
        if _ptr == 0:
            self._ptr_owner = True
            self._ptr = <ccudart.cudaDeviceProp *>calloc(1, sizeof(ccudart.cudaDeviceProp))
            if self._ptr is NULL:
                raise MemoryError('Failed to allocate length x size memory: 1x' + str(sizeof(ccudart.cudaDeviceProp)))
        else:
            self._ptr_owner = False
            self._ptr = <ccudart.cudaDeviceProp *>_ptr
    def __init__(self, void_ptr _ptr = 0):
        self._uuid = cudaUUID_t(_ptr=<void_ptr>&self._ptr[0].uuid)
    def __dealloc__(self):
        if self._ptr_owner is True and self._ptr is not NULL:
            free(self._ptr)
        pass
    def getPtr(self):
        return <void_ptr>self._ptr
    def __repr__(self):
        if self._ptr is not NULL:
            str_list = []
            str_list += ['name : ' + self.name.decode('utf-8')]
            str_list += ['uuid :\n' + '\n'.join(['    ' + line for line in str(self.uuid).splitlines()])]
            str_list += ['luid : ' + self.luid.hex()]
            str_list += ['luidDeviceNodeMask : ' + str(self.luidDeviceNodeMask)]
            str_list += ['totalGlobalMem : ' + str(self.totalGlobalMem)]
            str_list += ['sharedMemPerBlock : ' + str(self.sharedMemPerBlock)]
            str_list += ['regsPerBlock : ' + str(self.regsPerBlock)]
            str_list += ['warpSize : ' + str(self.warpSize)]
            str_list += ['memPitch : ' + str(self.memPitch)]
            str_list += ['maxThreadsPerBlock : ' + str(self.maxThreadsPerBlock)]
            str_list += ['maxThreadsDim : ' + str(self.maxThreadsDim)]
            str_list += ['maxGridSize : ' + str(self.maxGridSize)]
            str_list += ['clockRate : ' + str(self.clockRate)]
            str_list += ['totalConstMem : ' + str(self.totalConstMem)]
            str_list += ['major : ' + str(self.major)]
            str_list += ['minor : ' + str(self.minor)]
            str_list += ['textureAlignment : ' + str(self.textureAlignment)]
            str_list += ['texturePitchAlignment : ' + str(self.texturePitchAlignment)]
            str_list += ['deviceOverlap : ' + str(self.deviceOverlap)]
            str_list += ['multiProcessorCount : ' + str(self.multiProcessorCount)]
            str_list += ['kernelExecTimeoutEnabled : ' + str(self.kernelExecTimeoutEnabled)]
            str_list += ['integrated : ' + str(self.integrated)]
            str_list += ['canMapHostMemory : ' + str(self.canMapHostMemory)]
            str_list += ['computeMode : ' + str(self.computeMode)]
            str_list += ['maxTexture1D : ' + str(self.maxTexture1D)]
            str_list += ['maxTexture1DMipmap : ' + str(self.maxTexture1DMipmap)]
            str_list += ['maxTexture1DLinear : ' + str(self.maxTexture1DLinear)]
            str_list += ['maxTexture2D : ' + str(self.maxTexture2D)]
            str_list += ['maxTexture2DMipmap : ' + str(self.maxTexture2DMipmap)]
            str_list += ['maxTexture2DLinear : ' + str(self.maxTexture2DLinear)]
            str_list += ['maxTexture2DGather : ' + str(self.maxTexture2DGather)]
            str_list += ['maxTexture3D : ' + str(self.maxTexture3D)]
            str_list += ['maxTexture3DAlt : ' + str(self.maxTexture3DAlt)]
            str_list += ['maxTextureCubemap : ' + str(self.maxTextureCubemap)]
            str_list += ['maxTexture1DLayered : ' + str(self.maxTexture1DLayered)]
            str_list += ['maxTexture2DLayered : ' + str(self.maxTexture2DLayered)]
            str_list += ['maxTextureCubemapLayered : ' + str(self.maxTextureCubemapLayered)]
            str_list += ['maxSurface1D : ' + str(self.maxSurface1D)]
            str_list += ['maxSurface2D : ' + str(self.maxSurface2D)]
            str_list += ['maxSurface3D : ' + str(self.maxSurface3D)]
            str_list += ['maxSurface1DLayered : ' + str(self.maxSurface1DLayered)]
            str_list += ['maxSurface2DLayered : ' + str(self.maxSurface2DLayered)]
            str_list += ['maxSurfaceCubemap : ' + str(self.maxSurfaceCubemap)]
            str_list += ['maxSurfaceCubemapLayered : ' + str(self.maxSurfaceCubemapLayered)]
            str_list += ['surfaceAlignment : ' + str(self.surfaceAlignment)]
            str_list += ['concurrentKernels : ' + str(self.concurrentKernels)]
            str_list += ['ECCEnabled : ' + str(self.ECCEnabled)]
            str_list += ['pciBusID : ' + str(self.pciBusID)]
            str_list += ['pciDeviceID : ' + str(self.pciDeviceID)]
            str_list += ['pciDomainID : ' + str(self.pciDomainID)]
            str_list += ['tccDriver : ' + str(self.tccDriver)]
            str_list += ['asyncEngineCount : ' + str(self.asyncEngineCount)]
            str_list += ['unifiedAddressing : ' + str(self.unifiedAddressing)]
            str_list += ['memoryClockRate : ' + str(self.memoryClockRate)]
            str_list += ['memoryBusWidth : ' + str(self.memoryBusWidth)]
            str_list += ['l2CacheSize : ' + str(self.l2CacheSize)]
            str_list += ['persistingL2CacheMaxSize : ' + str(self.persistingL2CacheMaxSize)]
            str_list += ['maxThreadsPerMultiProcessor : ' + str(self.maxThreadsPerMultiProcessor)]
            str_list += ['streamPrioritiesSupported : ' + str(self.streamPrioritiesSupported)]
            str_list += ['globalL1CacheSupported : ' + str(self.globalL1CacheSupported)]
            str_list += ['localL1CacheSupported : ' + str(self.localL1CacheSupported)]
            str_list += ['sharedMemPerMultiprocessor : ' + str(self.sharedMemPerMultiprocessor)]
            str_list += ['regsPerMultiprocessor : ' + str(self.regsPerMultiprocessor)]
            str_list += ['managedMemory : ' + str(self.managedMemory)]
            str_list += ['isMultiGpuBoard : ' + str(self.isMultiGpuBoard)]
            str_list += ['multiGpuBoardGroupID : ' + str(self.multiGpuBoardGroupID)]
            str_list += ['hostNativeAtomicSupported : ' + str(self.hostNativeAtomicSupported)]
            str_list += ['singleToDoublePrecisionPerfRatio : ' + str(self.singleToDoublePrecisionPerfRatio)]
            str_list += ['pageableMemoryAccess : ' + str(self.pageableMemoryAccess)]
            str_list += ['concurrentManagedAccess : ' + str(self.concurrentManagedAccess)]
            str_list += ['computePreemptionSupported : ' + str(self.computePreemptionSupported)]
            str_list += ['canUseHostPointerForRegisteredMem : ' + str(self.canUseHostPointerForRegisteredMem)]
            str_list += ['cooperativeLaunch : ' + str(self.cooperativeLaunch)]
            str_list += ['cooperativeMultiDeviceLaunch : ' + str(self.cooperativeMultiDeviceLaunch)]
            str_list += ['sharedMemPerBlockOptin : ' + str(self.sharedMemPerBlockOptin)]
            str_list += ['pageableMemoryAccessUsesHostPageTables : ' + str(self.pageableMemoryAccessUsesHostPageTables)]
            str_list += ['directManagedMemAccessFromHost : ' + str(self.directManagedMemAccessFromHost)]
            str_list += ['maxBlocksPerMultiProcessor : ' + str(self.maxBlocksPerMultiProcessor)]
            str_list += ['accessPolicyMaxWindowSize : ' + str(self.accessPolicyMaxWindowSize)]
            str_list += ['reservedSharedMemPerBlock : ' + str(self.reservedSharedMemPerBlock)]
            return '\n'.join(str_list)
        else:
            return ''

    @property
    def name(self):
        return self._ptr[0].name
    @name.setter
    def name(self, name):
        pass
        self._ptr[0].name = name
    @property
    def uuid(self):
        return self._uuid
    @uuid.setter
    def uuid(self, uuid not None : cudaUUID_t):
        pass
        for _attr in dir(uuid):
            if _attr == 'getPtr':
                continue
            if not _attr.startswith('_'):
                setattr(self._uuid, _attr, getattr(uuid, _attr))
    @property
    def luid(self):
        return self._ptr[0].luid
    @luid.setter
    def luid(self, luid):
        pass
        self._ptr[0].luid = luid
    @property
    def luidDeviceNodeMask(self):
        return self._ptr[0].luidDeviceNodeMask
    @luidDeviceNodeMask.setter
    def luidDeviceNodeMask(self, unsigned int luidDeviceNodeMask):
        pass
        self._ptr[0].luidDeviceNodeMask = luidDeviceNodeMask
    @property
    def totalGlobalMem(self):
        return self._ptr[0].totalGlobalMem
    @totalGlobalMem.setter
    def totalGlobalMem(self, size_t totalGlobalMem):
        pass
        self._ptr[0].totalGlobalMem = totalGlobalMem
    @property
    def sharedMemPerBlock(self):
        return self._ptr[0].sharedMemPerBlock
    @sharedMemPerBlock.setter
    def sharedMemPerBlock(self, size_t sharedMemPerBlock):
        pass
        self._ptr[0].sharedMemPerBlock = sharedMemPerBlock
    @property
    def regsPerBlock(self):
        return self._ptr[0].regsPerBlock
    @regsPerBlock.setter
    def regsPerBlock(self, int regsPerBlock):
        pass
        self._ptr[0].regsPerBlock = regsPerBlock
    @property
    def warpSize(self):
        return self._ptr[0].warpSize
    @warpSize.setter
    def warpSize(self, int warpSize):
        pass
        self._ptr[0].warpSize = warpSize
    @property
    def memPitch(self):
        return self._ptr[0].memPitch
    @memPitch.setter
    def memPitch(self, size_t memPitch):
        pass
        self._ptr[0].memPitch = memPitch
    @property
    def maxThreadsPerBlock(self):
        return self._ptr[0].maxThreadsPerBlock
    @maxThreadsPerBlock.setter
    def maxThreadsPerBlock(self, int maxThreadsPerBlock):
        pass
        self._ptr[0].maxThreadsPerBlock = maxThreadsPerBlock
    @property
    def maxThreadsDim(self):
        return self._ptr[0].maxThreadsDim
    @maxThreadsDim.setter
    def maxThreadsDim(self, maxThreadsDim):
        pass
        self._ptr[0].maxThreadsDim = maxThreadsDim
    @property
    def maxGridSize(self):
        return self._ptr[0].maxGridSize
    @maxGridSize.setter
    def maxGridSize(self, maxGridSize):
        pass
        self._ptr[0].maxGridSize = maxGridSize
    @property
    def clockRate(self):
        return self._ptr[0].clockRate
    @clockRate.setter
    def clockRate(self, int clockRate):
        pass
        self._ptr[0].clockRate = clockRate
    @property
    def totalConstMem(self):
        return self._ptr[0].totalConstMem
    @totalConstMem.setter
    def totalConstMem(self, size_t totalConstMem):
        pass
        self._ptr[0].totalConstMem = totalConstMem
    @property
    def major(self):
        return self._ptr[0].major
    @major.setter
    def major(self, int major):
        pass
        self._ptr[0].major = major
    @property
    def minor(self):
        return self._ptr[0].minor
    @minor.setter
    def minor(self, int minor):
        pass
        self._ptr[0].minor = minor
    @property
    def textureAlignment(self):
        return self._ptr[0].textureAlignment
    @textureAlignment.setter
    def textureAlignment(self, size_t textureAlignment):
        pass
        self._ptr[0].textureAlignment = textureAlignment
    @property
    def texturePitchAlignment(self):
        return self._ptr[0].texturePitchAlignment
    @texturePitchAlignment.setter
    def texturePitchAlignment(self, size_t texturePitchAlignment):
        pass
        self._ptr[0].texturePitchAlignment = texturePitchAlignment
    @property
    def deviceOverlap(self):
        return self._ptr[0].deviceOverlap
    @deviceOverlap.setter
    def deviceOverlap(self, int deviceOverlap):
        pass
        self._ptr[0].deviceOverlap = deviceOverlap
    @property
    def multiProcessorCount(self):
        return self._ptr[0].multiProcessorCount
    @multiProcessorCount.setter
    def multiProcessorCount(self, int multiProcessorCount):
        pass
        self._ptr[0].multiProcessorCount = multiProcessorCount
    @property
    def kernelExecTimeoutEnabled(self):
        return self._ptr[0].kernelExecTimeoutEnabled
    @kernelExecTimeoutEnabled.setter
    def kernelExecTimeoutEnabled(self, int kernelExecTimeoutEnabled):
        pass
        self._ptr[0].kernelExecTimeoutEnabled = kernelExecTimeoutEnabled
    @property
    def integrated(self):
        return self._ptr[0].integrated
    @integrated.setter
    def integrated(self, int integrated):
        pass
        self._ptr[0].integrated = integrated
    @property
    def canMapHostMemory(self):
        return self._ptr[0].canMapHostMemory
    @canMapHostMemory.setter
    def canMapHostMemory(self, int canMapHostMemory):
        pass
        self._ptr[0].canMapHostMemory = canMapHostMemory
    @property
    def computeMode(self):
        return self._ptr[0].computeMode
    @computeMode.setter
    def computeMode(self, int computeMode):
        pass
        self._ptr[0].computeMode = computeMode
    @property
    def maxTexture1D(self):
        return self._ptr[0].maxTexture1D
    @maxTexture1D.setter
    def maxTexture1D(self, int maxTexture1D):
        pass
        self._ptr[0].maxTexture1D = maxTexture1D
    @property
    def maxTexture1DMipmap(self):
        return self._ptr[0].maxTexture1DMipmap
    @maxTexture1DMipmap.setter
    def maxTexture1DMipmap(self, int maxTexture1DMipmap):
        pass
        self._ptr[0].maxTexture1DMipmap = maxTexture1DMipmap
    @property
    def maxTexture1DLinear(self):
        return self._ptr[0].maxTexture1DLinear
    @maxTexture1DLinear.setter
    def maxTexture1DLinear(self, int maxTexture1DLinear):
        pass
        self._ptr[0].maxTexture1DLinear = maxTexture1DLinear
    @property
    def maxTexture2D(self):
        return self._ptr[0].maxTexture2D
    @maxTexture2D.setter
    def maxTexture2D(self, maxTexture2D):
        pass
        self._ptr[0].maxTexture2D = maxTexture2D
    @property
    def maxTexture2DMipmap(self):
        return self._ptr[0].maxTexture2DMipmap
    @maxTexture2DMipmap.setter
    def maxTexture2DMipmap(self, maxTexture2DMipmap):
        pass
        self._ptr[0].maxTexture2DMipmap = maxTexture2DMipmap
    @property
    def maxTexture2DLinear(self):
        return self._ptr[0].maxTexture2DLinear
    @maxTexture2DLinear.setter
    def maxTexture2DLinear(self, maxTexture2DLinear):
        pass
        self._ptr[0].maxTexture2DLinear = maxTexture2DLinear
    @property
    def maxTexture2DGather(self):
        return self._ptr[0].maxTexture2DGather
    @maxTexture2DGather.setter
    def maxTexture2DGather(self, maxTexture2DGather):
        pass
        self._ptr[0].maxTexture2DGather = maxTexture2DGather
    @property
    def maxTexture3D(self):
        return self._ptr[0].maxTexture3D
    @maxTexture3D.setter
    def maxTexture3D(self, maxTexture3D):
        pass
        self._ptr[0].maxTexture3D = maxTexture3D
    @property
    def maxTexture3DAlt(self):
        return self._ptr[0].maxTexture3DAlt
    @maxTexture3DAlt.setter
    def maxTexture3DAlt(self, maxTexture3DAlt):
        pass
        self._ptr[0].maxTexture3DAlt = maxTexture3DAlt
    @property
    def maxTextureCubemap(self):
        return self._ptr[0].maxTextureCubemap
    @maxTextureCubemap.setter
    def maxTextureCubemap(self, int maxTextureCubemap):
        pass
        self._ptr[0].maxTextureCubemap = maxTextureCubemap
    @property
    def maxTexture1DLayered(self):
        return self._ptr[0].maxTexture1DLayered
    @maxTexture1DLayered.setter
    def maxTexture1DLayered(self, maxTexture1DLayered):
        pass
        self._ptr[0].maxTexture1DLayered = maxTexture1DLayered
    @property
    def maxTexture2DLayered(self):
        return self._ptr[0].maxTexture2DLayered
    @maxTexture2DLayered.setter
    def maxTexture2DLayered(self, maxTexture2DLayered):
        pass
        self._ptr[0].maxTexture2DLayered = maxTexture2DLayered
    @property
    def maxTextureCubemapLayered(self):
        return self._ptr[0].maxTextureCubemapLayered
    @maxTextureCubemapLayered.setter
    def maxTextureCubemapLayered(self, maxTextureCubemapLayered):
        pass
        self._ptr[0].maxTextureCubemapLayered = maxTextureCubemapLayered
    @property
    def maxSurface1D(self):
        return self._ptr[0].maxSurface1D
    @maxSurface1D.setter
    def maxSurface1D(self, int maxSurface1D):
        pass
        self._ptr[0].maxSurface1D = maxSurface1D
    @property
    def maxSurface2D(self):
        return self._ptr[0].maxSurface2D
    @maxSurface2D.setter
    def maxSurface2D(self, maxSurface2D):
        pass
        self._ptr[0].maxSurface2D = maxSurface2D
    @property
    def maxSurface3D(self):
        return self._ptr[0].maxSurface3D
    @maxSurface3D.setter
    def maxSurface3D(self, maxSurface3D):
        pass
        self._ptr[0].maxSurface3D = maxSurface3D
    @property
    def maxSurface1DLayered(self):
        return self._ptr[0].maxSurface1DLayered
    @maxSurface1DLayered.setter
    def maxSurface1DLayered(self, maxSurface1DLayered):
        pass
        self._ptr[0].maxSurface1DLayered = maxSurface1DLayered
    @property
    def maxSurface2DLayered(self):
        return self._ptr[0].maxSurface2DLayered
    @maxSurface2DLayered.setter
    def maxSurface2DLayered(self, maxSurface2DLayered):
        pass
        self._ptr[0].maxSurface2DLayered = maxSurface2DLayered
    @property
    def maxSurfaceCubemap(self):
        return self._ptr[0].maxSurfaceCubemap
    @maxSurfaceCubemap.setter
    def maxSurfaceCubemap(self, int maxSurfaceCubemap):
        pass
        self._ptr[0].maxSurfaceCubemap = maxSurfaceCubemap
    @property
    def maxSurfaceCubemapLayered(self):
        return self._ptr[0].maxSurfaceCubemapLayered
    @maxSurfaceCubemapLayered.setter
    def maxSurfaceCubemapLayered(self, maxSurfaceCubemapLayered):
        pass
        self._ptr[0].maxSurfaceCubemapLayered = maxSurfaceCubemapLayered
    @property
    def surfaceAlignment(self):
        return self._ptr[0].surfaceAlignment
    @surfaceAlignment.setter
    def surfaceAlignment(self, size_t surfaceAlignment):
        pass
        self._ptr[0].surfaceAlignment = surfaceAlignment
    @property
    def concurrentKernels(self):
        return self._ptr[0].concurrentKernels
    @concurrentKernels.setter
    def concurrentKernels(self, int concurrentKernels):
        pass
        self._ptr[0].concurrentKernels = concurrentKernels
    @property
    def ECCEnabled(self):
        return self._ptr[0].ECCEnabled
    @ECCEnabled.setter
    def ECCEnabled(self, int ECCEnabled):
        pass
        self._ptr[0].ECCEnabled = ECCEnabled
    @property
    def pciBusID(self):
        return self._ptr[0].pciBusID
    @pciBusID.setter
    def pciBusID(self, int pciBusID):
        pass
        self._ptr[0].pciBusID = pciBusID
    @property
    def pciDeviceID(self):
        return self._ptr[0].pciDeviceID
    @pciDeviceID.setter
    def pciDeviceID(self, int pciDeviceID):
        pass
        self._ptr[0].pciDeviceID = pciDeviceID
    @property
    def pciDomainID(self):
        return self._ptr[0].pciDomainID
    @pciDomainID.setter
    def pciDomainID(self, int pciDomainID):
        pass
        self._ptr[0].pciDomainID = pciDomainID
    @property
    def tccDriver(self):
        return self._ptr[0].tccDriver
    @tccDriver.setter
    def tccDriver(self, int tccDriver):
        pass
        self._ptr[0].tccDriver = tccDriver
    @property
    def asyncEngineCount(self):
        return self._ptr[0].asyncEngineCount
    @asyncEngineCount.setter
    def asyncEngineCount(self, int asyncEngineCount):
        pass
        self._ptr[0].asyncEngineCount = asyncEngineCount
    @property
    def unifiedAddressing(self):
        return self._ptr[0].unifiedAddressing
    @unifiedAddressing.setter
    def unifiedAddressing(self, int unifiedAddressing):
        pass
        self._ptr[0].unifiedAddressing = unifiedAddressing
    @property
    def memoryClockRate(self):
        return self._ptr[0].memoryClockRate
    @memoryClockRate.setter
    def memoryClockRate(self, int memoryClockRate):
        pass
        self._ptr[0].memoryClockRate = memoryClockRate
    @property
    def memoryBusWidth(self):
        return self._ptr[0].memoryBusWidth
    @memoryBusWidth.setter
    def memoryBusWidth(self, int memoryBusWidth):
        pass
        self._ptr[0].memoryBusWidth = memoryBusWidth
    @property
    def l2CacheSize(self):
        return self._ptr[0].l2CacheSize
    @l2CacheSize.setter
    def l2CacheSize(self, int l2CacheSize):
        pass
        self._ptr[0].l2CacheSize = l2CacheSize
    @property
    def persistingL2CacheMaxSize(self):
        return self._ptr[0].persistingL2CacheMaxSize
    @persistingL2CacheMaxSize.setter
    def persistingL2CacheMaxSize(self, int persistingL2CacheMaxSize):
        pass
        self._ptr[0].persistingL2CacheMaxSize = persistingL2CacheMaxSize
    @property
    def maxThreadsPerMultiProcessor(self):
        return self._ptr[0].maxThreadsPerMultiProcessor
    @maxThreadsPerMultiProcessor.setter
    def maxThreadsPerMultiProcessor(self, int maxThreadsPerMultiProcessor):
        pass
        self._ptr[0].maxThreadsPerMultiProcessor = maxThreadsPerMultiProcessor
    @property
    def streamPrioritiesSupported(self):
        return self._ptr[0].streamPrioritiesSupported
    @streamPrioritiesSupported.setter
    def streamPrioritiesSupported(self, int streamPrioritiesSupported):
        pass
        self._ptr[0].streamPrioritiesSupported = streamPrioritiesSupported
    @property
    def globalL1CacheSupported(self):
        return self._ptr[0].globalL1CacheSupported
    @globalL1CacheSupported.setter
    def globalL1CacheSupported(self, int globalL1CacheSupported):
        pass
        self._ptr[0].globalL1CacheSupported = globalL1CacheSupported
    @property
    def localL1CacheSupported(self):
        return self._ptr[0].localL1CacheSupported
    @localL1CacheSupported.setter
    def localL1CacheSupported(self, int localL1CacheSupported):
        pass
        self._ptr[0].localL1CacheSupported = localL1CacheSupported
    @property
    def sharedMemPerMultiprocessor(self):
        return self._ptr[0].sharedMemPerMultiprocessor
    @sharedMemPerMultiprocessor.setter
    def sharedMemPerMultiprocessor(self, size_t sharedMemPerMultiprocessor):
        pass
        self._ptr[0].sharedMemPerMultiprocessor = sharedMemPerMultiprocessor
    @property
    def regsPerMultiprocessor(self):
        return self._ptr[0].regsPerMultiprocessor
    @regsPerMultiprocessor.setter
    def regsPerMultiprocessor(self, int regsPerMultiprocessor):
        pass
        self._ptr[0].regsPerMultiprocessor = regsPerMultiprocessor
    @property
    def managedMemory(self):
        return self._ptr[0].managedMemory
    @managedMemory.setter
    def managedMemory(self, int managedMemory):
        pass
        self._ptr[0].managedMemory = managedMemory
    @property
    def isMultiGpuBoard(self):
        return self._ptr[0].isMultiGpuBoard
    @isMultiGpuBoard.setter
    def isMultiGpuBoard(self, int isMultiGpuBoard):
        pass
        self._ptr[0].isMultiGpuBoard = isMultiGpuBoard
    @property
    def multiGpuBoardGroupID(self):
        return self._ptr[0].multiGpuBoardGroupID
    @multiGpuBoardGroupID.setter
    def multiGpuBoardGroupID(self, int multiGpuBoardGroupID):
        pass
        self._ptr[0].multiGpuBoardGroupID = multiGpuBoardGroupID
    @property
    def hostNativeAtomicSupported(self):
        return self._ptr[0].hostNativeAtomicSupported
    @hostNativeAtomicSupported.setter
    def hostNativeAtomicSupported(self, int hostNativeAtomicSupported):
        pass
        self._ptr[0].hostNativeAtomicSupported = hostNativeAtomicSupported
    @property
    def singleToDoublePrecisionPerfRatio(self):
        return self._ptr[0].singleToDoublePrecisionPerfRatio
    @singleToDoublePrecisionPerfRatio.setter
    def singleToDoublePrecisionPerfRatio(self, int singleToDoublePrecisionPerfRatio):
        pass
        self._ptr[0].singleToDoublePrecisionPerfRatio = singleToDoublePrecisionPerfRatio
    @property
    def pageableMemoryAccess(self):
        return self._ptr[0].pageableMemoryAccess
    @pageableMemoryAccess.setter
    def pageableMemoryAccess(self, int pageableMemoryAccess):
        pass
        self._ptr[0].pageableMemoryAccess = pageableMemoryAccess
    @property
    def concurrentManagedAccess(self):
        return self._ptr[0].concurrentManagedAccess
    @concurrentManagedAccess.setter
    def concurrentManagedAccess(self, int concurrentManagedAccess):
        pass
        self._ptr[0].concurrentManagedAccess = concurrentManagedAccess
    @property
    def computePreemptionSupported(self):
        return self._ptr[0].computePreemptionSupported
    @computePreemptionSupported.setter
    def computePreemptionSupported(self, int computePreemptionSupported):
        pass
        self._ptr[0].computePreemptionSupported = computePreemptionSupported
    @property
    def canUseHostPointerForRegisteredMem(self):
        return self._ptr[0].canUseHostPointerForRegisteredMem
    @canUseHostPointerForRegisteredMem.setter
    def canUseHostPointerForRegisteredMem(self, int canUseHostPointerForRegisteredMem):
        pass
        self._ptr[0].canUseHostPointerForRegisteredMem = canUseHostPointerForRegisteredMem
    @property
    def cooperativeLaunch(self):
        return self._ptr[0].cooperativeLaunch
    @cooperativeLaunch.setter
    def cooperativeLaunch(self, int cooperativeLaunch):
        pass
        self._ptr[0].cooperativeLaunch = cooperativeLaunch
    @property
    def cooperativeMultiDeviceLaunch(self):
        return self._ptr[0].cooperativeMultiDeviceLaunch
    @cooperativeMultiDeviceLaunch.setter
    def cooperativeMultiDeviceLaunch(self, int cooperativeMultiDeviceLaunch):
        pass
        self._ptr[0].cooperativeMultiDeviceLaunch = cooperativeMultiDeviceLaunch
    @property
    def sharedMemPerBlockOptin(self):
        return self._ptr[0].sharedMemPerBlockOptin
    @sharedMemPerBlockOptin.setter
    def sharedMemPerBlockOptin(self, size_t sharedMemPerBlockOptin):
        pass
        self._ptr[0].sharedMemPerBlockOptin = sharedMemPerBlockOptin
    @property
    def pageableMemoryAccessUsesHostPageTables(self):
        return self._ptr[0].pageableMemoryAccessUsesHostPageTables
    @pageableMemoryAccessUsesHostPageTables.setter
    def pageableMemoryAccessUsesHostPageTables(self, int pageableMemoryAccessUsesHostPageTables):
        pass
        self._ptr[0].pageableMemoryAccessUsesHostPageTables = pageableMemoryAccessUsesHostPageTables
    @property
    def directManagedMemAccessFromHost(self):
        return self._ptr[0].directManagedMemAccessFromHost
    @directManagedMemAccessFromHost.setter
    def directManagedMemAccessFromHost(self, int directManagedMemAccessFromHost):
        pass
        self._ptr[0].directManagedMemAccessFromHost = directManagedMemAccessFromHost
    @property
    def maxBlocksPerMultiProcessor(self):
        return self._ptr[0].maxBlocksPerMultiProcessor
    @maxBlocksPerMultiProcessor.setter
    def maxBlocksPerMultiProcessor(self, int maxBlocksPerMultiProcessor):
        pass
        self._ptr[0].maxBlocksPerMultiProcessor = maxBlocksPerMultiProcessor
    @property
    def accessPolicyMaxWindowSize(self):
        return self._ptr[0].accessPolicyMaxWindowSize
    @accessPolicyMaxWindowSize.setter
    def accessPolicyMaxWindowSize(self, int accessPolicyMaxWindowSize):
        pass
        self._ptr[0].accessPolicyMaxWindowSize = accessPolicyMaxWindowSize
    @property
    def reservedSharedMemPerBlock(self):
        return self._ptr[0].reservedSharedMemPerBlock
    @reservedSharedMemPerBlock.setter
    def reservedSharedMemPerBlock(self, size_t reservedSharedMemPerBlock):
        pass
        self._ptr[0].reservedSharedMemPerBlock = reservedSharedMemPerBlock

cdef class cudaIpcEventHandle_st:
    def __cinit__(self, void_ptr _ptr = 0):
        if _ptr == 0:
            self._ptr_owner = True
            self._ptr = <ccudart.cudaIpcEventHandle_st *>calloc(1, sizeof(ccudart.cudaIpcEventHandle_st))
            if self._ptr is NULL:
                raise MemoryError('Failed to allocate length x size memory: 1x' + str(sizeof(ccudart.cudaIpcEventHandle_st)))
        else:
            self._ptr_owner = False
            self._ptr = <ccudart.cudaIpcEventHandle_st *>_ptr
    def __init__(self, void_ptr _ptr = 0):
        pass
    def __dealloc__(self):
        if self._ptr_owner is True and self._ptr is not NULL:
            free(self._ptr)
        pass
    def getPtr(self):
        return <void_ptr>self._ptr
    def __repr__(self):
        if self._ptr is not NULL:
            str_list = []
            str_list += ['reserved : ' + str(self.reserved)]
            return '\n'.join(str_list)
        else:
            return ''

    @property
    def reserved(self):
        return self._ptr[0].reserved
    @reserved.setter
    def reserved(self, reserved):
        pass
        self._ptr[0].reserved = reserved

cdef class cudaIpcMemHandle_st:
    def __cinit__(self, void_ptr _ptr = 0):
        if _ptr == 0:
            self._ptr_owner = True
            self._ptr = <ccudart.cudaIpcMemHandle_st *>calloc(1, sizeof(ccudart.cudaIpcMemHandle_st))
            if self._ptr is NULL:
                raise MemoryError('Failed to allocate length x size memory: 1x' + str(sizeof(ccudart.cudaIpcMemHandle_st)))
        else:
            self._ptr_owner = False
            self._ptr = <ccudart.cudaIpcMemHandle_st *>_ptr
    def __init__(self, void_ptr _ptr = 0):
        pass
    def __dealloc__(self):
        if self._ptr_owner is True and self._ptr is not NULL:
            free(self._ptr)
        pass
    def getPtr(self):
        return <void_ptr>self._ptr
    def __repr__(self):
        if self._ptr is not NULL:
            str_list = []
            str_list += ['reserved : ' + str(self.reserved)]
            return '\n'.join(str_list)
        else:
            return ''

    @property
    def reserved(self):
        return self._ptr[0].reserved
    @reserved.setter
    def reserved(self, reserved):
        pass
        self._ptr[0].reserved = reserved

cdef class _cudaExternalMemoryHandleDesc_handle_handle_win32_s:
    def __cinit__(self, void_ptr _ptr):
        self._ptr = <ccudart.cudaExternalMemoryHandleDesc *>_ptr
    def __init__(self, void_ptr _ptr):
        pass
    def getPtr(self):
        return <void_ptr>self._ptr
    def __repr__(self):
        if self._ptr is not NULL:
            str_list = []
            str_list += ['handle : ' + hex(self.handle)]
            str_list += ['name : ' + hex(self.name)]
            return '\n'.join(str_list)
        else:
            return ''

    @property
    def handle(self):
        return <void_ptr>self._ptr[0].handle.win32.handle
    @handle.setter
    def handle(self, handle):
        _chandle = utils.HelperInputVoidPtr(handle)
        self._ptr[0].handle.win32.handle = <void*><void_ptr>_chandle.cptr
    @property
    def name(self):
        return <void_ptr>self._ptr[0].handle.win32.name
    @name.setter
    def name(self, name):
        _cname = utils.HelperInputVoidPtr(name)
        self._ptr[0].handle.win32.name = <void*><void_ptr>_cname.cptr

cdef class _cudaExternalMemoryHandleDesc_handle_u:
    def __cinit__(self, void_ptr _ptr):
        self._ptr = <ccudart.cudaExternalMemoryHandleDesc *>_ptr
    def __init__(self, void_ptr _ptr):
        self._win32 = _cudaExternalMemoryHandleDesc_handle_handle_win32_s(_ptr=<void_ptr>self._ptr)
    def getPtr(self):
        return <void_ptr>self._ptr
    def __repr__(self):
        if self._ptr is not NULL:
            str_list = []
            str_list += ['fd : ' + str(self.fd)]
            str_list += ['win32 :\n' + '\n'.join(['    ' + line for line in str(self.win32).splitlines()])]
            str_list += ['nvSciBufObject : ' + hex(self.nvSciBufObject)]
            return '\n'.join(str_list)
        else:
            return ''

    @property
    def fd(self):
        return self._ptr[0].handle.fd
    @fd.setter
    def fd(self, int fd):
        pass
        self._ptr[0].handle.fd = fd
    @property
    def win32(self):
        return self._win32
    @win32.setter
    def win32(self, win32 not None : _cudaExternalMemoryHandleDesc_handle_handle_win32_s):
        pass
        for _attr in dir(win32):
            if _attr == 'getPtr':
                continue
            if not _attr.startswith('_'):
                setattr(self._win32, _attr, getattr(win32, _attr))
    @property
    def nvSciBufObject(self):
        return <void_ptr>self._ptr[0].handle.nvSciBufObject
    @nvSciBufObject.setter
    def nvSciBufObject(self, nvSciBufObject):
        _cnvSciBufObject = utils.HelperInputVoidPtr(nvSciBufObject)
        self._ptr[0].handle.nvSciBufObject = <void*><void_ptr>_cnvSciBufObject.cptr

cdef class cudaExternalMemoryHandleDesc:
    def __cinit__(self, void_ptr _ptr = 0):
        if _ptr == 0:
            self._ptr_owner = True
            self._ptr = <ccudart.cudaExternalMemoryHandleDesc *>calloc(1, sizeof(ccudart.cudaExternalMemoryHandleDesc))
            if self._ptr is NULL:
                raise MemoryError('Failed to allocate length x size memory: 1x' + str(sizeof(ccudart.cudaExternalMemoryHandleDesc)))
        else:
            self._ptr_owner = False
            self._ptr = <ccudart.cudaExternalMemoryHandleDesc *>_ptr
    def __init__(self, void_ptr _ptr = 0):
        self._handle = _cudaExternalMemoryHandleDesc_handle_u(_ptr=<void_ptr>self._ptr)
    def __dealloc__(self):
        if self._ptr_owner is True and self._ptr is not NULL:
            free(self._ptr)
        pass
    def getPtr(self):
        return <void_ptr>self._ptr
    def __repr__(self):
        if self._ptr is not NULL:
            str_list = []
            str_list += ['type : ' + str(self.type)]
            str_list += ['handle :\n' + '\n'.join(['    ' + line for line in str(self.handle).splitlines()])]
            str_list += ['size : ' + str(self.size)]
            str_list += ['flags : ' + str(self.flags)]
            return '\n'.join(str_list)
        else:
            return ''

    @property
    def type(self):
        return cudaExternalMemoryHandleType(self._ptr[0].type)
    @type.setter
    def type(self, type not None : cudaExternalMemoryHandleType):
        pass
        self._ptr[0].type = type.value
    @property
    def handle(self):
        return self._handle
    @handle.setter
    def handle(self, handle not None : _cudaExternalMemoryHandleDesc_handle_u):
        pass
        for _attr in dir(handle):
            if _attr == 'getPtr':
                continue
            if not _attr.startswith('_'):
                setattr(self._handle, _attr, getattr(handle, _attr))
    @property
    def size(self):
        return self._ptr[0].size
    @size.setter
    def size(self, unsigned long long size):
        pass
        self._ptr[0].size = size
    @property
    def flags(self):
        return self._ptr[0].flags
    @flags.setter
    def flags(self, unsigned int flags):
        pass
        self._ptr[0].flags = flags

cdef class cudaExternalMemoryBufferDesc:
    def __cinit__(self, void_ptr _ptr = 0):
        if _ptr == 0:
            self._ptr_owner = True
            self._ptr = <ccudart.cudaExternalMemoryBufferDesc *>calloc(1, sizeof(ccudart.cudaExternalMemoryBufferDesc))
            if self._ptr is NULL:
                raise MemoryError('Failed to allocate length x size memory: 1x' + str(sizeof(ccudart.cudaExternalMemoryBufferDesc)))
        else:
            self._ptr_owner = False
            self._ptr = <ccudart.cudaExternalMemoryBufferDesc *>_ptr
    def __init__(self, void_ptr _ptr = 0):
        pass
    def __dealloc__(self):
        if self._ptr_owner is True and self._ptr is not NULL:
            free(self._ptr)
        pass
    def getPtr(self):
        return <void_ptr>self._ptr
    def __repr__(self):
        if self._ptr is not NULL:
            str_list = []
            str_list += ['offset : ' + str(self.offset)]
            str_list += ['size : ' + str(self.size)]
            str_list += ['flags : ' + str(self.flags)]
            return '\n'.join(str_list)
        else:
            return ''

    @property
    def offset(self):
        return self._ptr[0].offset
    @offset.setter
    def offset(self, unsigned long long offset):
        pass
        self._ptr[0].offset = offset
    @property
    def size(self):
        return self._ptr[0].size
    @size.setter
    def size(self, unsigned long long size):
        pass
        self._ptr[0].size = size
    @property
    def flags(self):
        return self._ptr[0].flags
    @flags.setter
    def flags(self, unsigned int flags):
        pass
        self._ptr[0].flags = flags

cdef class cudaExternalMemoryMipmappedArrayDesc:
    def __cinit__(self, void_ptr _ptr = 0):
        if _ptr == 0:
            self._ptr_owner = True
            self._ptr = <ccudart.cudaExternalMemoryMipmappedArrayDesc *>calloc(1, sizeof(ccudart.cudaExternalMemoryMipmappedArrayDesc))
            if self._ptr is NULL:
                raise MemoryError('Failed to allocate length x size memory: 1x' + str(sizeof(ccudart.cudaExternalMemoryMipmappedArrayDesc)))
        else:
            self._ptr_owner = False
            self._ptr = <ccudart.cudaExternalMemoryMipmappedArrayDesc *>_ptr
    def __init__(self, void_ptr _ptr = 0):
        self._formatDesc = cudaChannelFormatDesc(_ptr=<void_ptr>&self._ptr[0].formatDesc)
        self._extent = cudaExtent(_ptr=<void_ptr>&self._ptr[0].extent)
    def __dealloc__(self):
        if self._ptr_owner is True and self._ptr is not NULL:
            free(self._ptr)
        pass
    def getPtr(self):
        return <void_ptr>self._ptr
    def __repr__(self):
        if self._ptr is not NULL:
            str_list = []
            str_list += ['offset : ' + str(self.offset)]
            str_list += ['formatDesc :\n' + '\n'.join(['    ' + line for line in str(self.formatDesc).splitlines()])]
            str_list += ['extent :\n' + '\n'.join(['    ' + line for line in str(self.extent).splitlines()])]
            str_list += ['flags : ' + str(self.flags)]
            str_list += ['numLevels : ' + str(self.numLevels)]
            return '\n'.join(str_list)
        else:
            return ''

    @property
    def offset(self):
        return self._ptr[0].offset
    @offset.setter
    def offset(self, unsigned long long offset):
        pass
        self._ptr[0].offset = offset
    @property
    def formatDesc(self):
        return self._formatDesc
    @formatDesc.setter
    def formatDesc(self, formatDesc not None : cudaChannelFormatDesc):
        pass
        for _attr in dir(formatDesc):
            if _attr == 'getPtr':
                continue
            if not _attr.startswith('_'):
                setattr(self._formatDesc, _attr, getattr(formatDesc, _attr))
    @property
    def extent(self):
        return self._extent
    @extent.setter
    def extent(self, extent not None : cudaExtent):
        pass
        for _attr in dir(extent):
            if _attr == 'getPtr':
                continue
            if not _attr.startswith('_'):
                setattr(self._extent, _attr, getattr(extent, _attr))
    @property
    def flags(self):
        return self._ptr[0].flags
    @flags.setter
    def flags(self, unsigned int flags):
        pass
        self._ptr[0].flags = flags
    @property
    def numLevels(self):
        return self._ptr[0].numLevels
    @numLevels.setter
    def numLevels(self, unsigned int numLevels):
        pass
        self._ptr[0].numLevels = numLevels

cdef class _cudaExternalSemaphoreHandleDesc_handle_handle_win32_s:
    def __cinit__(self, void_ptr _ptr):
        self._ptr = <ccudart.cudaExternalSemaphoreHandleDesc *>_ptr
    def __init__(self, void_ptr _ptr):
        pass
    def getPtr(self):
        return <void_ptr>self._ptr
    def __repr__(self):
        if self._ptr is not NULL:
            str_list = []
            str_list += ['handle : ' + hex(self.handle)]
            str_list += ['name : ' + hex(self.name)]
            return '\n'.join(str_list)
        else:
            return ''

    @property
    def handle(self):
        return <void_ptr>self._ptr[0].handle.win32.handle
    @handle.setter
    def handle(self, handle):
        _chandle = utils.HelperInputVoidPtr(handle)
        self._ptr[0].handle.win32.handle = <void*><void_ptr>_chandle.cptr
    @property
    def name(self):
        return <void_ptr>self._ptr[0].handle.win32.name
    @name.setter
    def name(self, name):
        _cname = utils.HelperInputVoidPtr(name)
        self._ptr[0].handle.win32.name = <void*><void_ptr>_cname.cptr

cdef class _cudaExternalSemaphoreHandleDesc_handle_u:
    def __cinit__(self, void_ptr _ptr):
        self._ptr = <ccudart.cudaExternalSemaphoreHandleDesc *>_ptr
    def __init__(self, void_ptr _ptr):
        self._win32 = _cudaExternalSemaphoreHandleDesc_handle_handle_win32_s(_ptr=<void_ptr>self._ptr)
    def getPtr(self):
        return <void_ptr>self._ptr
    def __repr__(self):
        if self._ptr is not NULL:
            str_list = []
            str_list += ['fd : ' + str(self.fd)]
            str_list += ['win32 :\n' + '\n'.join(['    ' + line for line in str(self.win32).splitlines()])]
            str_list += ['nvSciSyncObj : ' + hex(self.nvSciSyncObj)]
            return '\n'.join(str_list)
        else:
            return ''

    @property
    def fd(self):
        return self._ptr[0].handle.fd
    @fd.setter
    def fd(self, int fd):
        pass
        self._ptr[0].handle.fd = fd
    @property
    def win32(self):
        return self._win32
    @win32.setter
    def win32(self, win32 not None : _cudaExternalSemaphoreHandleDesc_handle_handle_win32_s):
        pass
        for _attr in dir(win32):
            if _attr == 'getPtr':
                continue
            if not _attr.startswith('_'):
                setattr(self._win32, _attr, getattr(win32, _attr))
    @property
    def nvSciSyncObj(self):
        return <void_ptr>self._ptr[0].handle.nvSciSyncObj
    @nvSciSyncObj.setter
    def nvSciSyncObj(self, nvSciSyncObj):
        _cnvSciSyncObj = utils.HelperInputVoidPtr(nvSciSyncObj)
        self._ptr[0].handle.nvSciSyncObj = <void*><void_ptr>_cnvSciSyncObj.cptr

cdef class cudaExternalSemaphoreHandleDesc:
    def __cinit__(self, void_ptr _ptr = 0):
        if _ptr == 0:
            self._ptr_owner = True
            self._ptr = <ccudart.cudaExternalSemaphoreHandleDesc *>calloc(1, sizeof(ccudart.cudaExternalSemaphoreHandleDesc))
            if self._ptr is NULL:
                raise MemoryError('Failed to allocate length x size memory: 1x' + str(sizeof(ccudart.cudaExternalSemaphoreHandleDesc)))
        else:
            self._ptr_owner = False
            self._ptr = <ccudart.cudaExternalSemaphoreHandleDesc *>_ptr
    def __init__(self, void_ptr _ptr = 0):
        self._handle = _cudaExternalSemaphoreHandleDesc_handle_u(_ptr=<void_ptr>self._ptr)
    def __dealloc__(self):
        if self._ptr_owner is True and self._ptr is not NULL:
            free(self._ptr)
        pass
    def getPtr(self):
        return <void_ptr>self._ptr
    def __repr__(self):
        if self._ptr is not NULL:
            str_list = []
            str_list += ['type : ' + str(self.type)]
            str_list += ['handle :\n' + '\n'.join(['    ' + line for line in str(self.handle).splitlines()])]
            str_list += ['flags : ' + str(self.flags)]
            return '\n'.join(str_list)
        else:
            return ''

    @property
    def type(self):
        return cudaExternalSemaphoreHandleType(self._ptr[0].type)
    @type.setter
    def type(self, type not None : cudaExternalSemaphoreHandleType):
        pass
        self._ptr[0].type = type.value
    @property
    def handle(self):
        return self._handle
    @handle.setter
    def handle(self, handle not None : _cudaExternalSemaphoreHandleDesc_handle_u):
        pass
        for _attr in dir(handle):
            if _attr == 'getPtr':
                continue
            if not _attr.startswith('_'):
                setattr(self._handle, _attr, getattr(handle, _attr))
    @property
    def flags(self):
        return self._ptr[0].flags
    @flags.setter
    def flags(self, unsigned int flags):
        pass
        self._ptr[0].flags = flags

cdef class _cudaExternalSemaphoreSignalParams_params_params_fence_s:
    def __cinit__(self, void_ptr _ptr):
        self._ptr = <ccudart.cudaExternalSemaphoreSignalParams *>_ptr
    def __init__(self, void_ptr _ptr):
        pass
    def getPtr(self):
        return <void_ptr>self._ptr
    def __repr__(self):
        if self._ptr is not NULL:
            str_list = []
            str_list += ['value : ' + str(self.value)]
            return '\n'.join(str_list)
        else:
            return ''

    @property
    def value(self):
        return self._ptr[0].params.fence.value
    @value.setter
    def value(self, unsigned long long value):
        pass
        self._ptr[0].params.fence.value = value

cdef class _cudaExternalSemaphoreSignalParams_params_params_nvSciSync_u:
    def __cinit__(self, void_ptr _ptr):
        self._ptr = <ccudart.cudaExternalSemaphoreSignalParams *>_ptr
    def __init__(self, void_ptr _ptr):
        pass
    def getPtr(self):
        return <void_ptr>self._ptr
    def __repr__(self):
        if self._ptr is not NULL:
            str_list = []
            str_list += ['fence : ' + hex(self.fence)]
            str_list += ['reserved : ' + str(self.reserved)]
            return '\n'.join(str_list)
        else:
            return ''

    @property
    def fence(self):
        return <void_ptr>self._ptr[0].params.nvSciSync.fence
    @fence.setter
    def fence(self, fence):
        _cfence = utils.HelperInputVoidPtr(fence)
        self._ptr[0].params.nvSciSync.fence = <void*><void_ptr>_cfence.cptr
    @property
    def reserved(self):
        return self._ptr[0].params.nvSciSync.reserved
    @reserved.setter
    def reserved(self, unsigned long long reserved):
        pass
        self._ptr[0].params.nvSciSync.reserved = reserved

cdef class _cudaExternalSemaphoreSignalParams_params_params_keyedMutex_s:
    def __cinit__(self, void_ptr _ptr):
        self._ptr = <ccudart.cudaExternalSemaphoreSignalParams *>_ptr
    def __init__(self, void_ptr _ptr):
        pass
    def getPtr(self):
        return <void_ptr>self._ptr
    def __repr__(self):
        if self._ptr is not NULL:
            str_list = []
            str_list += ['key : ' + str(self.key)]
            return '\n'.join(str_list)
        else:
            return ''

    @property
    def key(self):
        return self._ptr[0].params.keyedMutex.key
    @key.setter
    def key(self, unsigned long long key):
        pass
        self._ptr[0].params.keyedMutex.key = key

cdef class _cudaExternalSemaphoreSignalParams_params_s:
    def __cinit__(self, void_ptr _ptr):
        self._ptr = <ccudart.cudaExternalSemaphoreSignalParams *>_ptr
    def __init__(self, void_ptr _ptr):
        self._fence = _cudaExternalSemaphoreSignalParams_params_params_fence_s(_ptr=<void_ptr>self._ptr)
        self._nvSciSync = _cudaExternalSemaphoreSignalParams_params_params_nvSciSync_u(_ptr=<void_ptr>self._ptr)
        self._keyedMutex = _cudaExternalSemaphoreSignalParams_params_params_keyedMutex_s(_ptr=<void_ptr>self._ptr)
    def getPtr(self):
        return <void_ptr>self._ptr
    def __repr__(self):
        if self._ptr is not NULL:
            str_list = []
            str_list += ['fence :\n' + '\n'.join(['    ' + line for line in str(self.fence).splitlines()])]
            str_list += ['nvSciSync :\n' + '\n'.join(['    ' + line for line in str(self.nvSciSync).splitlines()])]
            str_list += ['keyedMutex :\n' + '\n'.join(['    ' + line for line in str(self.keyedMutex).splitlines()])]
            str_list += ['reserved : ' + str(self.reserved)]
            return '\n'.join(str_list)
        else:
            return ''

    @property
    def fence(self):
        return self._fence
    @fence.setter
    def fence(self, fence not None : _cudaExternalSemaphoreSignalParams_params_params_fence_s):
        pass
        for _attr in dir(fence):
            if _attr == 'getPtr':
                continue
            if not _attr.startswith('_'):
                setattr(self._fence, _attr, getattr(fence, _attr))
    @property
    def nvSciSync(self):
        return self._nvSciSync
    @nvSciSync.setter
    def nvSciSync(self, nvSciSync not None : _cudaExternalSemaphoreSignalParams_params_params_nvSciSync_u):
        pass
        for _attr in dir(nvSciSync):
            if _attr == 'getPtr':
                continue
            if not _attr.startswith('_'):
                setattr(self._nvSciSync, _attr, getattr(nvSciSync, _attr))
    @property
    def keyedMutex(self):
        return self._keyedMutex
    @keyedMutex.setter
    def keyedMutex(self, keyedMutex not None : _cudaExternalSemaphoreSignalParams_params_params_keyedMutex_s):
        pass
        for _attr in dir(keyedMutex):
            if _attr == 'getPtr':
                continue
            if not _attr.startswith('_'):
                setattr(self._keyedMutex, _attr, getattr(keyedMutex, _attr))
    @property
    def reserved(self):
        return self._ptr[0].params.reserved
    @reserved.setter
    def reserved(self, reserved):
        pass
        self._ptr[0].params.reserved = reserved

cdef class cudaExternalSemaphoreSignalParams:
    def __cinit__(self, void_ptr _ptr = 0):
        if _ptr == 0:
            self._ptr_owner = True
            self._ptr = <ccudart.cudaExternalSemaphoreSignalParams *>calloc(1, sizeof(ccudart.cudaExternalSemaphoreSignalParams))
            if self._ptr is NULL:
                raise MemoryError('Failed to allocate length x size memory: 1x' + str(sizeof(ccudart.cudaExternalSemaphoreSignalParams)))
        else:
            self._ptr_owner = False
            self._ptr = <ccudart.cudaExternalSemaphoreSignalParams *>_ptr
    def __init__(self, void_ptr _ptr = 0):
        self._params = _cudaExternalSemaphoreSignalParams_params_s(_ptr=<void_ptr>self._ptr)
    def __dealloc__(self):
        if self._ptr_owner is True and self._ptr is not NULL:
            free(self._ptr)
        pass
    def getPtr(self):
        return <void_ptr>self._ptr
    def __repr__(self):
        if self._ptr is not NULL:
            str_list = []
            str_list += ['params :\n' + '\n'.join(['    ' + line for line in str(self.params).splitlines()])]
            str_list += ['flags : ' + str(self.flags)]
            str_list += ['reserved : ' + str(self.reserved)]
            return '\n'.join(str_list)
        else:
            return ''

    @property
    def params(self):
        return self._params
    @params.setter
    def params(self, params not None : _cudaExternalSemaphoreSignalParams_params_s):
        pass
        for _attr in dir(params):
            if _attr == 'getPtr':
                continue
            if not _attr.startswith('_'):
                setattr(self._params, _attr, getattr(params, _attr))
    @property
    def flags(self):
        return self._ptr[0].flags
    @flags.setter
    def flags(self, unsigned int flags):
        pass
        self._ptr[0].flags = flags
    @property
    def reserved(self):
        return self._ptr[0].reserved
    @reserved.setter
    def reserved(self, reserved):
        pass
        self._ptr[0].reserved = reserved

cdef class _cudaExternalSemaphoreWaitParams_params_params_fence_s:
    def __cinit__(self, void_ptr _ptr):
        self._ptr = <ccudart.cudaExternalSemaphoreWaitParams *>_ptr
    def __init__(self, void_ptr _ptr):
        pass
    def getPtr(self):
        return <void_ptr>self._ptr
    def __repr__(self):
        if self._ptr is not NULL:
            str_list = []
            str_list += ['value : ' + str(self.value)]
            return '\n'.join(str_list)
        else:
            return ''

    @property
    def value(self):
        return self._ptr[0].params.fence.value
    @value.setter
    def value(self, unsigned long long value):
        pass
        self._ptr[0].params.fence.value = value

cdef class _cudaExternalSemaphoreWaitParams_params_params_nvSciSync_u:
    def __cinit__(self, void_ptr _ptr):
        self._ptr = <ccudart.cudaExternalSemaphoreWaitParams *>_ptr
    def __init__(self, void_ptr _ptr):
        pass
    def getPtr(self):
        return <void_ptr>self._ptr
    def __repr__(self):
        if self._ptr is not NULL:
            str_list = []
            str_list += ['fence : ' + hex(self.fence)]
            str_list += ['reserved : ' + str(self.reserved)]
            return '\n'.join(str_list)
        else:
            return ''

    @property
    def fence(self):
        return <void_ptr>self._ptr[0].params.nvSciSync.fence
    @fence.setter
    def fence(self, fence):
        _cfence = utils.HelperInputVoidPtr(fence)
        self._ptr[0].params.nvSciSync.fence = <void*><void_ptr>_cfence.cptr
    @property
    def reserved(self):
        return self._ptr[0].params.nvSciSync.reserved
    @reserved.setter
    def reserved(self, unsigned long long reserved):
        pass
        self._ptr[0].params.nvSciSync.reserved = reserved

cdef class _cudaExternalSemaphoreWaitParams_params_params_keyedMutex_s:
    def __cinit__(self, void_ptr _ptr):
        self._ptr = <ccudart.cudaExternalSemaphoreWaitParams *>_ptr
    def __init__(self, void_ptr _ptr):
        pass
    def getPtr(self):
        return <void_ptr>self._ptr
    def __repr__(self):
        if self._ptr is not NULL:
            str_list = []
            str_list += ['key : ' + str(self.key)]
            str_list += ['timeoutMs : ' + str(self.timeoutMs)]
            return '\n'.join(str_list)
        else:
            return ''

    @property
    def key(self):
        return self._ptr[0].params.keyedMutex.key
    @key.setter
    def key(self, unsigned long long key):
        pass
        self._ptr[0].params.keyedMutex.key = key
    @property
    def timeoutMs(self):
        return self._ptr[0].params.keyedMutex.timeoutMs
    @timeoutMs.setter
    def timeoutMs(self, unsigned int timeoutMs):
        pass
        self._ptr[0].params.keyedMutex.timeoutMs = timeoutMs

cdef class _cudaExternalSemaphoreWaitParams_params_s:
    def __cinit__(self, void_ptr _ptr):
        self._ptr = <ccudart.cudaExternalSemaphoreWaitParams *>_ptr
    def __init__(self, void_ptr _ptr):
        self._fence = _cudaExternalSemaphoreWaitParams_params_params_fence_s(_ptr=<void_ptr>self._ptr)
        self._nvSciSync = _cudaExternalSemaphoreWaitParams_params_params_nvSciSync_u(_ptr=<void_ptr>self._ptr)
        self._keyedMutex = _cudaExternalSemaphoreWaitParams_params_params_keyedMutex_s(_ptr=<void_ptr>self._ptr)
    def getPtr(self):
        return <void_ptr>self._ptr
    def __repr__(self):
        if self._ptr is not NULL:
            str_list = []
            str_list += ['fence :\n' + '\n'.join(['    ' + line for line in str(self.fence).splitlines()])]
            str_list += ['nvSciSync :\n' + '\n'.join(['    ' + line for line in str(self.nvSciSync).splitlines()])]
            str_list += ['keyedMutex :\n' + '\n'.join(['    ' + line for line in str(self.keyedMutex).splitlines()])]
            str_list += ['reserved : ' + str(self.reserved)]
            return '\n'.join(str_list)
        else:
            return ''

    @property
    def fence(self):
        return self._fence
    @fence.setter
    def fence(self, fence not None : _cudaExternalSemaphoreWaitParams_params_params_fence_s):
        pass
        for _attr in dir(fence):
            if _attr == 'getPtr':
                continue
            if not _attr.startswith('_'):
                setattr(self._fence, _attr, getattr(fence, _attr))
    @property
    def nvSciSync(self):
        return self._nvSciSync
    @nvSciSync.setter
    def nvSciSync(self, nvSciSync not None : _cudaExternalSemaphoreWaitParams_params_params_nvSciSync_u):
        pass
        for _attr in dir(nvSciSync):
            if _attr == 'getPtr':
                continue
            if not _attr.startswith('_'):
                setattr(self._nvSciSync, _attr, getattr(nvSciSync, _attr))
    @property
    def keyedMutex(self):
        return self._keyedMutex
    @keyedMutex.setter
    def keyedMutex(self, keyedMutex not None : _cudaExternalSemaphoreWaitParams_params_params_keyedMutex_s):
        pass
        for _attr in dir(keyedMutex):
            if _attr == 'getPtr':
                continue
            if not _attr.startswith('_'):
                setattr(self._keyedMutex, _attr, getattr(keyedMutex, _attr))
    @property
    def reserved(self):
        return self._ptr[0].params.reserved
    @reserved.setter
    def reserved(self, reserved):
        pass
        self._ptr[0].params.reserved = reserved

cdef class cudaExternalSemaphoreWaitParams:
    def __cinit__(self, void_ptr _ptr = 0):
        if _ptr == 0:
            self._ptr_owner = True
            self._ptr = <ccudart.cudaExternalSemaphoreWaitParams *>calloc(1, sizeof(ccudart.cudaExternalSemaphoreWaitParams))
            if self._ptr is NULL:
                raise MemoryError('Failed to allocate length x size memory: 1x' + str(sizeof(ccudart.cudaExternalSemaphoreWaitParams)))
        else:
            self._ptr_owner = False
            self._ptr = <ccudart.cudaExternalSemaphoreWaitParams *>_ptr
    def __init__(self, void_ptr _ptr = 0):
        self._params = _cudaExternalSemaphoreWaitParams_params_s(_ptr=<void_ptr>self._ptr)
    def __dealloc__(self):
        if self._ptr_owner is True and self._ptr is not NULL:
            free(self._ptr)
        pass
    def getPtr(self):
        return <void_ptr>self._ptr
    def __repr__(self):
        if self._ptr is not NULL:
            str_list = []
            str_list += ['params :\n' + '\n'.join(['    ' + line for line in str(self.params).splitlines()])]
            str_list += ['flags : ' + str(self.flags)]
            str_list += ['reserved : ' + str(self.reserved)]
            return '\n'.join(str_list)
        else:
            return ''

    @property
    def params(self):
        return self._params
    @params.setter
    def params(self, params not None : _cudaExternalSemaphoreWaitParams_params_s):
        pass
        for _attr in dir(params):
            if _attr == 'getPtr':
                continue
            if not _attr.startswith('_'):
                setattr(self._params, _attr, getattr(params, _attr))
    @property
    def flags(self):
        return self._ptr[0].flags
    @flags.setter
    def flags(self, unsigned int flags):
        pass
        self._ptr[0].flags = flags
    @property
    def reserved(self):
        return self._ptr[0].reserved
    @reserved.setter
    def reserved(self, reserved):
        pass
        self._ptr[0].reserved = reserved

cdef class cudaKernelNodeParams:
    def __cinit__(self, void_ptr _ptr = 0):
        if _ptr == 0:
            self._ptr_owner = True
            self._ptr = <ccudart.cudaKernelNodeParams *>calloc(1, sizeof(ccudart.cudaKernelNodeParams))
            if self._ptr is NULL:
                raise MemoryError('Failed to allocate length x size memory: 1x' + str(sizeof(ccudart.cudaKernelNodeParams)))
        else:
            self._ptr_owner = False
            self._ptr = <ccudart.cudaKernelNodeParams *>_ptr
    def __init__(self, void_ptr _ptr = 0):
        self._gridDim = dim3(_ptr=<void_ptr>&self._ptr[0].gridDim)
        self._blockDim = dim3(_ptr=<void_ptr>&self._ptr[0].blockDim)
    def __dealloc__(self):
        if self._ptr_owner is True and self._ptr is not NULL:
            free(self._ptr)
        pass
    def getPtr(self):
        return <void_ptr>self._ptr
    def __repr__(self):
        if self._ptr is not NULL:
            str_list = []
            str_list += ['func : ' + hex(self.func)]
            str_list += ['gridDim :\n' + '\n'.join(['    ' + line for line in str(self.gridDim).splitlines()])]
            str_list += ['blockDim :\n' + '\n'.join(['    ' + line for line in str(self.blockDim).splitlines()])]
            str_list += ['sharedMemBytes : ' + str(self.sharedMemBytes)]
            str_list += ['kernelParams : ' + str(self.kernelParams)]
            str_list += ['extra : ' + str(self.extra)]
            return '\n'.join(str_list)
        else:
            return ''

    @property
    def func(self):
        return <void_ptr>self._ptr[0].func
    @func.setter
    def func(self, func):
        _cfunc = utils.HelperInputVoidPtr(func)
        self._ptr[0].func = <void*><void_ptr>_cfunc.cptr
    @property
    def gridDim(self):
        return self._gridDim
    @gridDim.setter
    def gridDim(self, gridDim not None : dim3):
        pass
        for _attr in dir(gridDim):
            if _attr == 'getPtr':
                continue
            if not _attr.startswith('_'):
                setattr(self._gridDim, _attr, getattr(gridDim, _attr))
    @property
    def blockDim(self):
        return self._blockDim
    @blockDim.setter
    def blockDim(self, blockDim not None : dim3):
        pass
        for _attr in dir(blockDim):
            if _attr == 'getPtr':
                continue
            if not _attr.startswith('_'):
                setattr(self._blockDim, _attr, getattr(blockDim, _attr))
    @property
    def sharedMemBytes(self):
        return self._ptr[0].sharedMemBytes
    @sharedMemBytes.setter
    def sharedMemBytes(self, unsigned int sharedMemBytes):
        pass
        self._ptr[0].sharedMemBytes = sharedMemBytes
    @property
    def kernelParams(self):
        return <void_ptr>self._ptr[0].kernelParams
    @kernelParams.setter
    def kernelParams(self, kernelParams):
        self._ckernelParams = utils.HelperKernelParams(kernelParams)
        self._ptr[0].kernelParams = <void**><void_ptr>self._ckernelParams.ckernelParams
    @property
    def extra(self):
        return <void_ptr>self._ptr[0].extra
    @extra.setter
    def extra(self, void_ptr extra):
        self._ptr[0].extra = <void**>extra

cdef class cudaExternalSemaphoreSignalNodeParams:
    def __cinit__(self, void_ptr _ptr = 0):
        if _ptr == 0:
            self._ptr_owner = True
            self._ptr = <ccudart.cudaExternalSemaphoreSignalNodeParams *>calloc(1, sizeof(ccudart.cudaExternalSemaphoreSignalNodeParams))
            if self._ptr is NULL:
                raise MemoryError('Failed to allocate length x size memory: 1x' + str(sizeof(ccudart.cudaExternalSemaphoreSignalNodeParams)))
        else:
            self._ptr_owner = False
            self._ptr = <ccudart.cudaExternalSemaphoreSignalNodeParams *>_ptr
    def __init__(self, void_ptr _ptr = 0):
        pass
    def __dealloc__(self):
        if self._ptr_owner is True and self._ptr is not NULL:
            free(self._ptr)
        if self._extSemArray is not NULL:
            free(self._extSemArray)
        if self._paramsArray is not NULL:
            free(self._paramsArray)
    def getPtr(self):
        return <void_ptr>self._ptr
    def __repr__(self):
        if self._ptr is not NULL:
            str_list = []
            str_list += ['extSemArray : ' + str(self.extSemArray)]
            str_list += ['paramsArray : ' + str(self.paramsArray)]
            str_list += ['numExtSems : ' + str(self.numExtSems)]
            return '\n'.join(str_list)
        else:
            return ''

    @property
    def extSemArray(self):
        arrs = [<void_ptr>self._ptr[0].extSemArray + x*sizeof(ccudart.cudaExternalSemaphore_t) for x in range(self._extSemArray_length)]
        return [cudaExternalSemaphore_t(_ptr=arr) for arr in arrs]
    @extSemArray.setter
    def extSemArray(self, val):
        if len(val) == 0:
            free(self._extSemArray)
            self._extSemArray_length = 0
            self._ptr[0].extSemArray = NULL
        else:
            if self._extSemArray_length != <size_t>len(val):
                free(self._extSemArray)
                self._extSemArray = <ccudart.cudaExternalSemaphore_t*> calloc(len(val), sizeof(ccudart.cudaExternalSemaphore_t))
                if self._extSemArray is NULL:
                    raise MemoryError('Failed to allocate length x size memory: ' + str(len(val)) + 'x' + str(sizeof(ccudart.cudaExternalSemaphore_t)))
                self._extSemArray_length = <size_t>len(val)
                self._ptr[0].extSemArray = self._extSemArray
            for idx in range(len(val)):
                self._extSemArray[idx] = (<cudaExternalSemaphore_t>val[idx])._ptr[0]
    @property
    def paramsArray(self):
        arrs = [<void_ptr>self._ptr[0].paramsArray + x*sizeof(ccudart.cudaExternalSemaphoreSignalParams) for x in range(self._paramsArray_length)]
        return [cudaExternalSemaphoreSignalParams(_ptr=arr) for arr in arrs]
    @paramsArray.setter
    def paramsArray(self, val):
        if len(val) == 0:
            free(self._paramsArray)
            self._paramsArray_length = 0
            self._ptr[0].paramsArray = NULL
        else:
            if self._paramsArray_length != <size_t>len(val):
                free(self._paramsArray)
                self._paramsArray = <ccudart.cudaExternalSemaphoreSignalParams*> calloc(len(val), sizeof(ccudart.cudaExternalSemaphoreSignalParams))
                if self._paramsArray is NULL:
                    raise MemoryError('Failed to allocate length x size memory: ' + str(len(val)) + 'x' + str(sizeof(ccudart.cudaExternalSemaphoreSignalParams)))
                self._paramsArray_length = <size_t>len(val)
                self._ptr[0].paramsArray = self._paramsArray
            for idx in range(len(val)):
                memcpy(&self._paramsArray[idx], (<cudaExternalSemaphoreSignalParams>val[idx])._ptr, sizeof(ccudart.cudaExternalSemaphoreSignalParams))
    @property
    def numExtSems(self):
        return self._ptr[0].numExtSems
    @numExtSems.setter
    def numExtSems(self, unsigned int numExtSems):
        pass
        self._ptr[0].numExtSems = numExtSems

cdef class cudaExternalSemaphoreWaitNodeParams:
    def __cinit__(self, void_ptr _ptr = 0):
        if _ptr == 0:
            self._ptr_owner = True
            self._ptr = <ccudart.cudaExternalSemaphoreWaitNodeParams *>calloc(1, sizeof(ccudart.cudaExternalSemaphoreWaitNodeParams))
            if self._ptr is NULL:
                raise MemoryError('Failed to allocate length x size memory: 1x' + str(sizeof(ccudart.cudaExternalSemaphoreWaitNodeParams)))
        else:
            self._ptr_owner = False
            self._ptr = <ccudart.cudaExternalSemaphoreWaitNodeParams *>_ptr
    def __init__(self, void_ptr _ptr = 0):
        pass
    def __dealloc__(self):
        if self._ptr_owner is True and self._ptr is not NULL:
            free(self._ptr)
        if self._extSemArray is not NULL:
            free(self._extSemArray)
        if self._paramsArray is not NULL:
            free(self._paramsArray)
    def getPtr(self):
        return <void_ptr>self._ptr
    def __repr__(self):
        if self._ptr is not NULL:
            str_list = []
            str_list += ['extSemArray : ' + str(self.extSemArray)]
            str_list += ['paramsArray : ' + str(self.paramsArray)]
            str_list += ['numExtSems : ' + str(self.numExtSems)]
            return '\n'.join(str_list)
        else:
            return ''

    @property
    def extSemArray(self):
        arrs = [<void_ptr>self._ptr[0].extSemArray + x*sizeof(ccudart.cudaExternalSemaphore_t) for x in range(self._extSemArray_length)]
        return [cudaExternalSemaphore_t(_ptr=arr) for arr in arrs]
    @extSemArray.setter
    def extSemArray(self, val):
        if len(val) == 0:
            free(self._extSemArray)
            self._extSemArray_length = 0
            self._ptr[0].extSemArray = NULL
        else:
            if self._extSemArray_length != <size_t>len(val):
                free(self._extSemArray)
                self._extSemArray = <ccudart.cudaExternalSemaphore_t*> calloc(len(val), sizeof(ccudart.cudaExternalSemaphore_t))
                if self._extSemArray is NULL:
                    raise MemoryError('Failed to allocate length x size memory: ' + str(len(val)) + 'x' + str(sizeof(ccudart.cudaExternalSemaphore_t)))
                self._extSemArray_length = <size_t>len(val)
                self._ptr[0].extSemArray = self._extSemArray
            for idx in range(len(val)):
                self._extSemArray[idx] = (<cudaExternalSemaphore_t>val[idx])._ptr[0]
    @property
    def paramsArray(self):
        arrs = [<void_ptr>self._ptr[0].paramsArray + x*sizeof(ccudart.cudaExternalSemaphoreWaitParams) for x in range(self._paramsArray_length)]
        return [cudaExternalSemaphoreWaitParams(_ptr=arr) for arr in arrs]
    @paramsArray.setter
    def paramsArray(self, val):
        if len(val) == 0:
            free(self._paramsArray)
            self._paramsArray_length = 0
            self._ptr[0].paramsArray = NULL
        else:
            if self._paramsArray_length != <size_t>len(val):
                free(self._paramsArray)
                self._paramsArray = <ccudart.cudaExternalSemaphoreWaitParams*> calloc(len(val), sizeof(ccudart.cudaExternalSemaphoreWaitParams))
                if self._paramsArray is NULL:
                    raise MemoryError('Failed to allocate length x size memory: ' + str(len(val)) + 'x' + str(sizeof(ccudart.cudaExternalSemaphoreWaitParams)))
                self._paramsArray_length = <size_t>len(val)
                self._ptr[0].paramsArray = self._paramsArray
            for idx in range(len(val)):
                memcpy(&self._paramsArray[idx], (<cudaExternalSemaphoreWaitParams>val[idx])._ptr, sizeof(ccudart.cudaExternalSemaphoreWaitParams))
    @property
    def numExtSems(self):
        return self._ptr[0].numExtSems
    @numExtSems.setter
    def numExtSems(self, unsigned int numExtSems):
        pass
        self._ptr[0].numExtSems = numExtSems

cdef class cudaTextureDesc:
    def __cinit__(self, void_ptr _ptr = 0):
        if _ptr == 0:
            self._ptr_owner = True
            self._ptr = <ccudart.cudaTextureDesc *>calloc(1, sizeof(ccudart.cudaTextureDesc))
            if self._ptr is NULL:
                raise MemoryError('Failed to allocate length x size memory: 1x' + str(sizeof(ccudart.cudaTextureDesc)))
        else:
            self._ptr_owner = False
            self._ptr = <ccudart.cudaTextureDesc *>_ptr
    def __init__(self, void_ptr _ptr = 0):
        pass
    def __dealloc__(self):
        if self._ptr_owner is True and self._ptr is not NULL:
            free(self._ptr)
        pass
    def getPtr(self):
        return <void_ptr>self._ptr
    def __repr__(self):
        if self._ptr is not NULL:
            str_list = []
            str_list += ['addressMode : ' + str(self.addressMode)]
            str_list += ['filterMode : ' + str(self.filterMode)]
            str_list += ['readMode : ' + str(self.readMode)]
            str_list += ['sRGB : ' + str(self.sRGB)]
            str_list += ['borderColor : ' + str(self.borderColor)]
            str_list += ['normalizedCoords : ' + str(self.normalizedCoords)]
            str_list += ['maxAnisotropy : ' + str(self.maxAnisotropy)]
            str_list += ['mipmapFilterMode : ' + str(self.mipmapFilterMode)]
            str_list += ['mipmapLevelBias : ' + str(self.mipmapLevelBias)]
            str_list += ['minMipmapLevelClamp : ' + str(self.minMipmapLevelClamp)]
            str_list += ['maxMipmapLevelClamp : ' + str(self.maxMipmapLevelClamp)]
            str_list += ['disableTrilinearOptimization : ' + str(self.disableTrilinearOptimization)]
            return '\n'.join(str_list)
        else:
            return ''

    @property
    def addressMode(self):
        return [cudaTextureAddressMode(_x) for _x in list(self._ptr[0].addressMode)]
    @addressMode.setter
    def addressMode(self, addressMode):
        self._ptr[0].addressMode = [_x.value for _x in addressMode]
    @property
    def filterMode(self):
        return cudaTextureFilterMode(self._ptr[0].filterMode)
    @filterMode.setter
    def filterMode(self, filterMode not None : cudaTextureFilterMode):
        pass
        self._ptr[0].filterMode = filterMode.value
    @property
    def readMode(self):
        return cudaTextureReadMode(self._ptr[0].readMode)
    @readMode.setter
    def readMode(self, readMode not None : cudaTextureReadMode):
        pass
        self._ptr[0].readMode = readMode.value
    @property
    def sRGB(self):
        return self._ptr[0].sRGB
    @sRGB.setter
    def sRGB(self, int sRGB):
        pass
        self._ptr[0].sRGB = sRGB
    @property
    def borderColor(self):
        return self._ptr[0].borderColor
    @borderColor.setter
    def borderColor(self, borderColor):
        pass
        self._ptr[0].borderColor = borderColor
    @property
    def normalizedCoords(self):
        return self._ptr[0].normalizedCoords
    @normalizedCoords.setter
    def normalizedCoords(self, int normalizedCoords):
        pass
        self._ptr[0].normalizedCoords = normalizedCoords
    @property
    def maxAnisotropy(self):
        return self._ptr[0].maxAnisotropy
    @maxAnisotropy.setter
    def maxAnisotropy(self, unsigned int maxAnisotropy):
        pass
        self._ptr[0].maxAnisotropy = maxAnisotropy
    @property
    def mipmapFilterMode(self):
        return cudaTextureFilterMode(self._ptr[0].mipmapFilterMode)
    @mipmapFilterMode.setter
    def mipmapFilterMode(self, mipmapFilterMode not None : cudaTextureFilterMode):
        pass
        self._ptr[0].mipmapFilterMode = mipmapFilterMode.value
    @property
    def mipmapLevelBias(self):
        return self._ptr[0].mipmapLevelBias
    @mipmapLevelBias.setter
    def mipmapLevelBias(self, float mipmapLevelBias):
        pass
        self._ptr[0].mipmapLevelBias = mipmapLevelBias
    @property
    def minMipmapLevelClamp(self):
        return self._ptr[0].minMipmapLevelClamp
    @minMipmapLevelClamp.setter
    def minMipmapLevelClamp(self, float minMipmapLevelClamp):
        pass
        self._ptr[0].minMipmapLevelClamp = minMipmapLevelClamp
    @property
    def maxMipmapLevelClamp(self):
        return self._ptr[0].maxMipmapLevelClamp
    @maxMipmapLevelClamp.setter
    def maxMipmapLevelClamp(self, float maxMipmapLevelClamp):
        pass
        self._ptr[0].maxMipmapLevelClamp = maxMipmapLevelClamp
    @property
    def disableTrilinearOptimization(self):
        return self._ptr[0].disableTrilinearOptimization
    @disableTrilinearOptimization.setter
    def disableTrilinearOptimization(self, int disableTrilinearOptimization):
        pass
        self._ptr[0].disableTrilinearOptimization = disableTrilinearOptimization

@cython.embedsignature(True)
def cudaDeviceReset():
    err = ccudart.cudaDeviceReset()
    return (cudaError_t(err),)

@cython.embedsignature(True)
def cudaDeviceSynchronize():
    with nogil:
        err = ccudart.cudaDeviceSynchronize()
    return (cudaError_t(err),)

@cython.embedsignature(True)
def cudaDeviceSetLimit(limit not None : cudaLimit, size_t value):
    cdef ccudart.cudaLimit climit = limit.value
    err = ccudart.cudaDeviceSetLimit(climit, value)
    return (cudaError_t(err),)

@cython.embedsignature(True)
def cudaDeviceGetLimit(limit not None : cudaLimit):
    cdef size_t pValue = 0
    cdef ccudart.cudaLimit climit = limit.value
    err = ccudart.cudaDeviceGetLimit(&pValue, climit)
    return (cudaError_t(err), pValue)

@cython.embedsignature(True)
def cudaDeviceGetTexture1DLinearMaxWidth(fmtDesc : cudaChannelFormatDesc, int device):
    cdef size_t maxWidthInElements = 0
    cdef ccudart.cudaChannelFormatDesc* cfmtDesc_ptr = fmtDesc._ptr if fmtDesc != None else NULL
    err = ccudart.cudaDeviceGetTexture1DLinearMaxWidth(&maxWidthInElements, cfmtDesc_ptr, device)
    return (cudaError_t(err), maxWidthInElements)

@cython.embedsignature(True)
def cudaDeviceGetCacheConfig():
    cdef ccudart.cudaFuncCache pCacheConfig
    err = ccudart.cudaDeviceGetCacheConfig(&pCacheConfig)
    return (cudaError_t(err), cudaFuncCache(pCacheConfig))

@cython.embedsignature(True)
def cudaDeviceGetStreamPriorityRange():
    cdef int leastPriority = 0
    cdef int greatestPriority = 0
    err = ccudart.cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority)
    return (cudaError_t(err), leastPriority, greatestPriority)

@cython.embedsignature(True)
def cudaDeviceSetCacheConfig(cacheConfig not None : cudaFuncCache):
    cdef ccudart.cudaFuncCache ccacheConfig = cacheConfig.value
    err = ccudart.cudaDeviceSetCacheConfig(ccacheConfig)
    return (cudaError_t(err),)

@cython.embedsignature(True)
def cudaDeviceGetSharedMemConfig():
    cdef ccudart.cudaSharedMemConfig pConfig
    err = ccudart.cudaDeviceGetSharedMemConfig(&pConfig)
    return (cudaError_t(err), cudaSharedMemConfig(pConfig))

@cython.embedsignature(True)
def cudaDeviceSetSharedMemConfig(config not None : cudaSharedMemConfig):
    cdef ccudart.cudaSharedMemConfig cconfig = config.value
    err = ccudart.cudaDeviceSetSharedMemConfig(cconfig)
    return (cudaError_t(err),)

@cython.embedsignature(True)
def cudaDeviceGetByPCIBusId(char* pciBusId):
    cdef int device = 0
    err = ccudart.cudaDeviceGetByPCIBusId(&device, pciBusId)
    return (cudaError_t(err), device)

@cython.embedsignature(True)
def cudaDeviceGetPCIBusId(int length, int device):
    cdef char* pciBusId = <char*>calloc(1, length)
    err = ccudart.cudaDeviceGetPCIBusId(pciBusId, length, device)
    return (cudaError_t(err), <bytes>pciBusId)

@cython.embedsignature(True)
def cudaIpcGetEventHandle(event):
    if not isinstance(event, (cudaEvent_t, cuda.CUevent)):
        raise TypeError("Argument 'event' is not instance of type (expected <class 'cudapython.cudart.cudaEvent_t, cuda.CUevent'>, found " + str(type(event)))
    cdef cudaIpcEventHandle_t handle = cudaIpcEventHandle_t()
    err = ccudart.cudaIpcGetEventHandle(handle._ptr, <ccudart.cudaEvent_t>(<cudaEvent_t>event)._ptr[0])
    return (cudaError_t(err), handle)

@cython.embedsignature(True)
def cudaIpcOpenEventHandle(handle not None : cudaIpcEventHandle_t):
    cdef cudaEvent_t event = cudaEvent_t()
    err = ccudart.cudaIpcOpenEventHandle(<ccudart.cudaEvent_t*>event._ptr, handle._ptr[0])
    return (cudaError_t(err), event)

@cython.embedsignature(True)
def cudaIpcGetMemHandle(devPtr):
    cdef cudaIpcMemHandle_t handle = cudaIpcMemHandle_t()
    cdevPtr = utils.HelperInputVoidPtr(devPtr)
    cdef void* cdevPtr_ptr = <void*><void_ptr>cdevPtr.cptr
    err = ccudart.cudaIpcGetMemHandle(handle._ptr, cdevPtr_ptr)
    return (cudaError_t(err), handle)

@cython.embedsignature(True)
def cudaIpcOpenMemHandle(handle not None : cudaIpcMemHandle_t, unsigned int flags):
    cdef void_ptr devPtr = 0
    err = ccudart.cudaIpcOpenMemHandle(<void**>&devPtr, handle._ptr[0], flags)
    return (cudaError_t(err), devPtr)

@cython.embedsignature(True)
def cudaIpcCloseMemHandle(devPtr):
    cdevPtr = utils.HelperInputVoidPtr(devPtr)
    cdef void* cdevPtr_ptr = <void*><void_ptr>cdevPtr.cptr
    err = ccudart.cudaIpcCloseMemHandle(cdevPtr_ptr)
    return (cudaError_t(err),)

@cython.embedsignature(True)
def cudaDeviceFlushGPUDirectRDMAWrites(target not None : cudaFlushGPUDirectRDMAWritesTarget, scope not None : cudaFlushGPUDirectRDMAWritesScope):
    cdef ccudart.cudaFlushGPUDirectRDMAWritesTarget ctarget = target.value
    cdef ccudart.cudaFlushGPUDirectRDMAWritesScope cscope = scope.value
    err = ccudart.cudaDeviceFlushGPUDirectRDMAWrites(ctarget, cscope)
    return (cudaError_t(err),)

@cython.embedsignature(True)
def cudaThreadExit():
    err = ccudart.cudaThreadExit()
    return (cudaError_t(err),)

@cython.embedsignature(True)
def cudaThreadSynchronize():
    err = ccudart.cudaThreadSynchronize()
    return (cudaError_t(err),)

@cython.embedsignature(True)
def cudaThreadSetLimit(limit not None : cudaLimit, size_t value):
    cdef ccudart.cudaLimit climit = limit.value
    err = ccudart.cudaThreadSetLimit(climit, value)
    return (cudaError_t(err),)

@cython.embedsignature(True)
def cudaThreadGetLimit(limit not None : cudaLimit):
    cdef size_t pValue = 0
    cdef ccudart.cudaLimit climit = limit.value
    err = ccudart.cudaThreadGetLimit(&pValue, climit)
    return (cudaError_t(err), pValue)

@cython.embedsignature(True)
def cudaThreadGetCacheConfig():
    cdef ccudart.cudaFuncCache pCacheConfig
    err = ccudart.cudaThreadGetCacheConfig(&pCacheConfig)
    return (cudaError_t(err), cudaFuncCache(pCacheConfig))

@cython.embedsignature(True)
def cudaThreadSetCacheConfig(cacheConfig not None : cudaFuncCache):
    cdef ccudart.cudaFuncCache ccacheConfig = cacheConfig.value
    err = ccudart.cudaThreadSetCacheConfig(ccacheConfig)
    return (cudaError_t(err),)

@cython.embedsignature(True)
def cudaGetLastError():
    err = ccudart.cudaGetLastError()
    return (cudaError_t(err),)

@cython.embedsignature(True)
def cudaPeekAtLastError():
    err = ccudart.cudaPeekAtLastError()
    return (cudaError_t(err),)

@cython.embedsignature(True)
def cudaGetErrorName(error not None : cudaError_t):
    cdef ccudart.cudaError_t cerror = error.value
    err = ccudart.cudaGetErrorName(cerror)
    return (cudaError_t.cudaSuccess, err)

@cython.embedsignature(True)
def cudaGetErrorString(error not None : cudaError_t):
    cdef ccudart.cudaError_t cerror = error.value
    err = ccudart.cudaGetErrorString(cerror)
    return (cudaError_t.cudaSuccess, err)

@cython.embedsignature(True)
def cudaGetDeviceCount():
    cdef int count = 0
    err = ccudart.cudaGetDeviceCount(&count)
    return (cudaError_t(err), count)

@cython.embedsignature(True)
def cudaGetDeviceProperties(int device):
    cdef cudaDeviceProp prop = cudaDeviceProp()
    err = ccudart.cudaGetDeviceProperties(prop._ptr, device)
    return (cudaError_t(err), prop)

@cython.embedsignature(True)
def cudaDeviceGetAttribute(attr not None : cudaDeviceAttr, int device):
    cdef int value = 0
    cdef ccudart.cudaDeviceAttr cattr = attr.value
    err = ccudart.cudaDeviceGetAttribute(&value, cattr, device)
    return (cudaError_t(err), value)

@cython.embedsignature(True)
def cudaDeviceGetDefaultMemPool(int device):
    cdef cudaMemPool_t memPool = cudaMemPool_t()
    with nogil:
        err = ccudart.cudaDeviceGetDefaultMemPool(<ccudart.cudaMemPool_t*>memPool._ptr, device)
    return (cudaError_t(err), memPool)

@cython.embedsignature(True)
def cudaDeviceSetMemPool(int device, memPool):
    if not isinstance(memPool, (cudaMemPool_t, cuda.CUmemoryPool)):
        raise TypeError("Argument 'memPool' is not instance of type (expected <class 'cudapython.cudart.cudaMemPool_t, cuda.CUmemoryPool'>, found " + str(type(memPool)))
    with nogil:
        err = ccudart.cudaDeviceSetMemPool(device, <ccudart.cudaMemPool_t>(<cudaMemPool_t>memPool)._ptr[0])
    return (cudaError_t(err),)

@cython.embedsignature(True)
def cudaDeviceGetMemPool(int device):
    cdef cudaMemPool_t memPool = cudaMemPool_t()
    with nogil:
        err = ccudart.cudaDeviceGetMemPool(<ccudart.cudaMemPool_t*>memPool._ptr, device)
    return (cudaError_t(err), memPool)

@cython.embedsignature(True)
def cudaDeviceGetNvSciSyncAttributes(int device, int flags):
    cdef void_ptr nvSciSyncAttrList = 0
    cdef void* cnvSciSyncAttrList_ptr = <void*>nvSciSyncAttrList
    err = ccudart.cudaDeviceGetNvSciSyncAttributes(cnvSciSyncAttrList_ptr, device, flags)
    return (cudaError_t(err), nvSciSyncAttrList)

@cython.embedsignature(True)
def cudaDeviceGetP2PAttribute(attr not None : cudaDeviceP2PAttr, int srcDevice, int dstDevice):
    cdef int value = 0
    cdef ccudart.cudaDeviceP2PAttr cattr = attr.value
    err = ccudart.cudaDeviceGetP2PAttribute(&value, cattr, srcDevice, dstDevice)
    return (cudaError_t(err), value)

@cython.embedsignature(True)
def cudaChooseDevice(prop : cudaDeviceProp):
    cdef int device = 0
    cdef ccudart.cudaDeviceProp* cprop_ptr = prop._ptr if prop != None else NULL
    err = ccudart.cudaChooseDevice(&device, cprop_ptr)
    return (cudaError_t(err), device)

@cython.embedsignature(True)
def cudaSetDevice(int device):
    err = ccudart.cudaSetDevice(device)
    return (cudaError_t(err),)

@cython.embedsignature(True)
def cudaGetDevice():
    cdef int device = 0
    err = ccudart.cudaGetDevice(&device)
    return (cudaError_t(err), device)

@cython.embedsignature(True)
def cudaSetDeviceFlags(unsigned int flags):
    err = ccudart.cudaSetDeviceFlags(flags)
    return (cudaError_t(err),)

@cython.embedsignature(True)
def cudaGetDeviceFlags():
    cdef unsigned int flags = 0
    err = ccudart.cudaGetDeviceFlags(&flags)
    return (cudaError_t(err), flags)

@cython.embedsignature(True)
def cudaStreamCreate():
    cdef cudaStream_t pStream = cudaStream_t()
    err = ccudart.cudaStreamCreate(<ccudart.cudaStream_t*>pStream._ptr)
    return (cudaError_t(err), pStream)

@cython.embedsignature(True)
def cudaStreamCreateWithFlags(unsigned int flags):
    cdef cudaStream_t pStream = cudaStream_t()
    err = ccudart.cudaStreamCreateWithFlags(<ccudart.cudaStream_t*>pStream._ptr, flags)
    return (cudaError_t(err), pStream)

@cython.embedsignature(True)
def cudaStreamCreateWithPriority(unsigned int flags, int priority):
    cdef cudaStream_t pStream = cudaStream_t()
    err = ccudart.cudaStreamCreateWithPriority(<ccudart.cudaStream_t*>pStream._ptr, flags, priority)
    return (cudaError_t(err), pStream)

@cython.embedsignature(True)
def cudaStreamGetPriority(hStream):
    if not isinstance(hStream, (cudaStream_t, cuda.CUstream)):
        raise TypeError("Argument 'hStream' is not instance of type (expected <class 'cudapython.cudart.cudaStream_t, cuda.CUstream'>, found " + str(type(hStream)))
    cdef int priority = 0
    err = ccudart.cudaStreamGetPriority(<ccudart.cudaStream_t>(<cudaStream_t>hStream)._ptr[0], &priority)
    return (cudaError_t(err), priority)

@cython.embedsignature(True)
def cudaStreamGetFlags(hStream):
    if not isinstance(hStream, (cudaStream_t, cuda.CUstream)):
        raise TypeError("Argument 'hStream' is not instance of type (expected <class 'cudapython.cudart.cudaStream_t, cuda.CUstream'>, found " + str(type(hStream)))
    cdef unsigned int flags = 0
    err = ccudart.cudaStreamGetFlags(<ccudart.cudaStream_t>(<cudaStream_t>hStream)._ptr[0], &flags)
    return (cudaError_t(err), flags)

@cython.embedsignature(True)
def cudaCtxResetPersistingL2Cache():
    err = ccudart.cudaCtxResetPersistingL2Cache()
    return (cudaError_t(err),)

@cython.embedsignature(True)
def cudaStreamCopyAttributes(dst, src):
    if not isinstance(src, (cudaStream_t, cuda.CUstream)):
        raise TypeError("Argument 'src' is not instance of type (expected <class 'cudapython.cudart.cudaStream_t, cuda.CUstream'>, found " + str(type(src)))
    if not isinstance(dst, (cudaStream_t, cuda.CUstream)):
        raise TypeError("Argument 'dst' is not instance of type (expected <class 'cudapython.cudart.cudaStream_t, cuda.CUstream'>, found " + str(type(dst)))
    err = ccudart.cudaStreamCopyAttributes(<ccudart.cudaStream_t>(<cudaStream_t>dst)._ptr[0], <ccudart.cudaStream_t>(<cudaStream_t>src)._ptr[0])
    return (cudaError_t(err),)

@cython.embedsignature(True)
def cudaStreamGetAttribute(hStream, attr not None : cudaStreamAttrID):
    if not isinstance(hStream, (cudaStream_t, cuda.CUstream)):
        raise TypeError("Argument 'hStream' is not instance of type (expected <class 'cudapython.cudart.cudaStream_t, cuda.CUstream'>, found " + str(type(hStream)))
    cdef ccudart.cudaStreamAttrID cattr = attr.value
    cdef cudaStreamAttrValue value_out = cudaStreamAttrValue()
    err = ccudart.cudaStreamGetAttribute(<ccudart.cudaStream_t>(<cudaStream_t>hStream)._ptr[0], cattr, value_out._ptr)
    return (cudaError_t(err), value_out)

@cython.embedsignature(True)
def cudaStreamSetAttribute(hStream, attr not None : cudaStreamAttrID, value : cudaStreamAttrValue):
    if not isinstance(hStream, (cudaStream_t, cuda.CUstream)):
        raise TypeError("Argument 'hStream' is not instance of type (expected <class 'cudapython.cudart.cudaStream_t, cuda.CUstream'>, found " + str(type(hStream)))
    cdef ccudart.cudaStreamAttrID cattr = attr.value
    cdef ccudart.cudaStreamAttrValue* cvalue_ptr = value._ptr if value != None else NULL
    err = ccudart.cudaStreamSetAttribute(<ccudart.cudaStream_t>(<cudaStream_t>hStream)._ptr[0], cattr, cvalue_ptr)
    return (cudaError_t(err),)

@cython.embedsignature(True)
def cudaStreamDestroy(stream):
    if not isinstance(stream, (cudaStream_t, cuda.CUstream)):
        raise TypeError("Argument 'stream' is not instance of type (expected <class 'cudapython.cudart.cudaStream_t, cuda.CUstream'>, found " + str(type(stream)))
    err = ccudart.cudaStreamDestroy(<ccudart.cudaStream_t>(<cudaStream_t>stream)._ptr[0])
    return (cudaError_t(err),)

@cython.embedsignature(True)
def cudaStreamWaitEvent(stream, event, unsigned int flags):
    if not isinstance(event, (cudaEvent_t, cuda.CUevent)):
        raise TypeError("Argument 'event' is not instance of type (expected <class 'cudapython.cudart.cudaEvent_t, cuda.CUevent'>, found " + str(type(event)))
    if not isinstance(stream, (cudaStream_t, cuda.CUstream)):
        raise TypeError("Argument 'stream' is not instance of type (expected <class 'cudapython.cudart.cudaStream_t, cuda.CUstream'>, found " + str(type(stream)))
    with nogil:
        err = ccudart.cudaStreamWaitEvent(<ccudart.cudaStream_t>(<cudaStream_t>stream)._ptr[0], <ccudart.cudaEvent_t>(<cudaEvent_t>event)._ptr[0], flags)
    return (cudaError_t(err),)

@cython.embedsignature(True)
def cudaStreamAddCallback(stream, callback not None : cudaStreamCallback_t, userData, unsigned int flags):
    if not isinstance(stream, (cudaStream_t, cuda.CUstream)):
        raise TypeError("Argument 'stream' is not instance of type (expected <class 'cudapython.cudart.cudaStream_t, cuda.CUstream'>, found " + str(type(stream)))
    cuserData = utils.HelperInputVoidPtr(userData)
    cdef void* cuserData_ptr = <void*><void_ptr>cuserData.cptr
    with nogil:
        err = ccudart.cudaStreamAddCallback(<ccudart.cudaStream_t>(<cudaStream_t>stream)._ptr[0], callback._ptr[0], cuserData_ptr, flags)
    return (cudaError_t(err),)

@cython.embedsignature(True)
def cudaStreamSynchronize(stream):
    if not isinstance(stream, (cudaStream_t, cuda.CUstream)):
        raise TypeError("Argument 'stream' is not instance of type (expected <class 'cudapython.cudart.cudaStream_t, cuda.CUstream'>, found " + str(type(stream)))
    with nogil:
        err = ccudart.cudaStreamSynchronize(<ccudart.cudaStream_t>(<cudaStream_t>stream)._ptr[0])
    return (cudaError_t(err),)

@cython.embedsignature(True)
def cudaStreamQuery(stream):
    if not isinstance(stream, (cudaStream_t, cuda.CUstream)):
        raise TypeError("Argument 'stream' is not instance of type (expected <class 'cudapython.cudart.cudaStream_t, cuda.CUstream'>, found " + str(type(stream)))
    err = ccudart.cudaStreamQuery(<ccudart.cudaStream_t>(<cudaStream_t>stream)._ptr[0])
    return (cudaError_t(err),)

@cython.embedsignature(True)
def cudaStreamAttachMemAsync(stream, devPtr, size_t length, unsigned int flags):
    if not isinstance(stream, (cudaStream_t, cuda.CUstream)):
        raise TypeError("Argument 'stream' is not instance of type (expected <class 'cudapython.cudart.cudaStream_t, cuda.CUstream'>, found " + str(type(stream)))
    cdevPtr = utils.HelperInputVoidPtr(devPtr)
    cdef void* cdevPtr_ptr = <void*><void_ptr>cdevPtr.cptr
    err = ccudart.cudaStreamAttachMemAsync(<ccudart.cudaStream_t>(<cudaStream_t>stream)._ptr[0], cdevPtr_ptr, length, flags)
    return (cudaError_t(err),)

@cython.embedsignature(True)
def cudaStreamBeginCapture(stream, mode not None : cudaStreamCaptureMode):
    if not isinstance(stream, (cudaStream_t, cuda.CUstream)):
        raise TypeError("Argument 'stream' is not instance of type (expected <class 'cudapython.cudart.cudaStream_t, cuda.CUstream'>, found " + str(type(stream)))
    cdef ccudart.cudaStreamCaptureMode cmode = mode.value
    err = ccudart.cudaStreamBeginCapture(<ccudart.cudaStream_t>(<cudaStream_t>stream)._ptr[0], cmode)
    return (cudaError_t(err),)

@cython.embedsignature(True)
def cudaThreadExchangeStreamCaptureMode(mode not None : cudaStreamCaptureMode):
    cdef ccudart.cudaStreamCaptureMode cmode = mode.value
    err = ccudart.cudaThreadExchangeStreamCaptureMode(&cmode)
    return (cudaError_t(err), cudaStreamCaptureMode(cmode))

@cython.embedsignature(True)
def cudaStreamEndCapture(stream):
    if not isinstance(stream, (cudaStream_t, cuda.CUstream)):
        raise TypeError("Argument 'stream' is not instance of type (expected <class 'cudapython.cudart.cudaStream_t, cuda.CUstream'>, found " + str(type(stream)))
    cdef cudaGraph_t pGraph = cudaGraph_t()
    err = ccudart.cudaStreamEndCapture(<ccudart.cudaStream_t>(<cudaStream_t>stream)._ptr[0], <ccudart.cudaGraph_t*>pGraph._ptr)
    return (cudaError_t(err), pGraph)

@cython.embedsignature(True)
def cudaStreamIsCapturing(stream):
    if not isinstance(stream, (cudaStream_t, cuda.CUstream)):
        raise TypeError("Argument 'stream' is not instance of type (expected <class 'cudapython.cudart.cudaStream_t, cuda.CUstream'>, found " + str(type(stream)))
    cdef ccudart.cudaStreamCaptureStatus pCaptureStatus
    err = ccudart.cudaStreamIsCapturing(<ccudart.cudaStream_t>(<cudaStream_t>stream)._ptr[0], &pCaptureStatus)
    return (cudaError_t(err), cudaStreamCaptureStatus(pCaptureStatus))

@cython.embedsignature(True)
def cudaStreamGetCaptureInfo(stream):
    if not isinstance(stream, (cudaStream_t, cuda.CUstream)):
        raise TypeError("Argument 'stream' is not instance of type (expected <class 'cudapython.cudart.cudaStream_t, cuda.CUstream'>, found " + str(type(stream)))
    cdef ccudart.cudaStreamCaptureStatus pCaptureStatus
    cdef unsigned long long pId = 0
    err = ccudart.cudaStreamGetCaptureInfo(<ccudart.cudaStream_t>(<cudaStream_t>stream)._ptr[0], &pCaptureStatus, &pId)
    return (cudaError_t(err), cudaStreamCaptureStatus(pCaptureStatus), pId)

@cython.embedsignature(True)
def cudaStreamGetCaptureInfo_v2(stream):
    if not isinstance(stream, (cudaStream_t, cuda.CUstream)):
        raise TypeError("Argument 'stream' is not instance of type (expected <class 'cudapython.cudart.cudaStream_t, cuda.CUstream'>, found " + str(type(stream)))
    cdef ccudart.cudaStreamCaptureStatus captureStatus_out
    cdef unsigned long long id_out = 0
    cdef cudaGraph_t graph_out = cudaGraph_t()
    cdef const ccudart.cudaGraphNode_t* cdependencies_out = NULL
    pydependencies_out = []
    cdef size_t numDependencies_out = 0
    err = ccudart.cudaStreamGetCaptureInfo_v2(<ccudart.cudaStream_t>(<cudaStream_t>stream)._ptr[0], &captureStatus_out, &id_out, <ccudart.cudaGraph_t*>graph_out._ptr, &cdependencies_out, &numDependencies_out)
    if cudaError_t(err) == cudaError_t(0):
        pydependencies_out = [cudaGraphNode_t(init_value=<void_ptr>cdependencies_out[idx]) for idx in range(numDependencies_out)]
    return (cudaError_t(err), cudaStreamCaptureStatus(captureStatus_out), id_out, graph_out, pydependencies_out, numDependencies_out)

@cython.embedsignature(True)
def cudaStreamUpdateCaptureDependencies(stream, dependencies : List[cudaGraphNode_t], size_t numDependencies, unsigned int flags):
    dependencies = [] if dependencies is None else dependencies
    if not all(isinstance(_x, (cudaGraphNode_t, cuda.CUgraphNode)) for _x in dependencies):
        raise TypeError("Argument 'dependencies' is not instance of type (expected List[cudapython.ccudart.cudaGraphNode_t, cuda.CUgraphNode]")
    if not isinstance(stream, (cudaStream_t, cuda.CUstream)):
        raise TypeError("Argument 'stream' is not instance of type (expected <class 'cudapython.cudart.cudaStream_t, cuda.CUstream'>, found " + str(type(stream)))
    cdef ccudart.cudaGraphNode_t* cdependencies = NULL
    if len(dependencies) > 0:
        cdependencies = <ccudart.cudaGraphNode_t*> calloc(len(dependencies), sizeof(ccudart.cudaGraphNode_t))
        if cdependencies is NULL:
            raise MemoryError('Failed to allocate length x size memory: ' + str(len(dependencies)) + 'x' + str(sizeof(ccudart.cudaGraphNode_t)))
        else:
            for idx in range(len(dependencies)):
                cdependencies[idx] = <ccudart.cudaGraphNode_t>(<cudaGraphNode_t>dependencies[idx])._ptr[0]

    if numDependencies > <size_t>len(dependencies): raise RuntimeError("List is too small: " + str(len(dependencies)) + " < " + str(numDependencies))
    err = ccudart.cudaStreamUpdateCaptureDependencies(<ccudart.cudaStream_t>(<cudaStream_t>stream)._ptr[0], <ccudart.cudaGraphNode_t*>(<cudaGraphNode_t>dependencies[0])._ptr if len(dependencies) == 1 else cdependencies, numDependencies, flags)
    if cdependencies is not NULL:
        free(cdependencies)
    return (cudaError_t(err),)

@cython.embedsignature(True)
def cudaEventCreate():
    cdef cudaEvent_t event = cudaEvent_t()
    err = ccudart.cudaEventCreate(<ccudart.cudaEvent_t*>event._ptr)
    return (cudaError_t(err), event)

@cython.embedsignature(True)
def cudaEventCreateWithFlags(unsigned int flags):
    cdef cudaEvent_t event = cudaEvent_t()
    err = ccudart.cudaEventCreateWithFlags(<ccudart.cudaEvent_t*>event._ptr, flags)
    return (cudaError_t(err), event)

@cython.embedsignature(True)
def cudaEventRecord(event, stream):
    if not isinstance(stream, (cudaStream_t, cuda.CUstream)):
        raise TypeError("Argument 'stream' is not instance of type (expected <class 'cudapython.cudart.cudaStream_t, cuda.CUstream'>, found " + str(type(stream)))
    if not isinstance(event, (cudaEvent_t, cuda.CUevent)):
        raise TypeError("Argument 'event' is not instance of type (expected <class 'cudapython.cudart.cudaEvent_t, cuda.CUevent'>, found " + str(type(event)))
    err = ccudart.cudaEventRecord(<ccudart.cudaEvent_t>(<cudaEvent_t>event)._ptr[0], <ccudart.cudaStream_t>(<cudaStream_t>stream)._ptr[0])
    return (cudaError_t(err),)

@cython.embedsignature(True)
def cudaEventRecordWithFlags(event, stream, unsigned int flags):
    if not isinstance(stream, (cudaStream_t, cuda.CUstream)):
        raise TypeError("Argument 'stream' is not instance of type (expected <class 'cudapython.cudart.cudaStream_t, cuda.CUstream'>, found " + str(type(stream)))
    if not isinstance(event, (cudaEvent_t, cuda.CUevent)):
        raise TypeError("Argument 'event' is not instance of type (expected <class 'cudapython.cudart.cudaEvent_t, cuda.CUevent'>, found " + str(type(event)))
    err = ccudart.cudaEventRecordWithFlags(<ccudart.cudaEvent_t>(<cudaEvent_t>event)._ptr[0], <ccudart.cudaStream_t>(<cudaStream_t>stream)._ptr[0], flags)
    return (cudaError_t(err),)

@cython.embedsignature(True)
def cudaEventQuery(event):
    if not isinstance(event, (cudaEvent_t, cuda.CUevent)):
        raise TypeError("Argument 'event' is not instance of type (expected <class 'cudapython.cudart.cudaEvent_t, cuda.CUevent'>, found " + str(type(event)))
    err = ccudart.cudaEventQuery(<ccudart.cudaEvent_t>(<cudaEvent_t>event)._ptr[0])
    return (cudaError_t(err),)

@cython.embedsignature(True)
def cudaEventSynchronize(event):
    if not isinstance(event, (cudaEvent_t, cuda.CUevent)):
        raise TypeError("Argument 'event' is not instance of type (expected <class 'cudapython.cudart.cudaEvent_t, cuda.CUevent'>, found " + str(type(event)))
    err = ccudart.cudaEventSynchronize(<ccudart.cudaEvent_t>(<cudaEvent_t>event)._ptr[0])
    return (cudaError_t(err),)

@cython.embedsignature(True)
def cudaEventDestroy(event):
    if not isinstance(event, (cudaEvent_t, cuda.CUevent)):
        raise TypeError("Argument 'event' is not instance of type (expected <class 'cudapython.cudart.cudaEvent_t, cuda.CUevent'>, found " + str(type(event)))
    err = ccudart.cudaEventDestroy(<ccudart.cudaEvent_t>(<cudaEvent_t>event)._ptr[0])
    return (cudaError_t(err),)

@cython.embedsignature(True)
def cudaEventElapsedTime(start, end):
    if not isinstance(end, (cudaEvent_t, cuda.CUevent)):
        raise TypeError("Argument 'end' is not instance of type (expected <class 'cudapython.cudart.cudaEvent_t, cuda.CUevent'>, found " + str(type(end)))
    if not isinstance(start, (cudaEvent_t, cuda.CUevent)):
        raise TypeError("Argument 'start' is not instance of type (expected <class 'cudapython.cudart.cudaEvent_t, cuda.CUevent'>, found " + str(type(start)))
    cdef float ms = 0
    err = ccudart.cudaEventElapsedTime(&ms, <ccudart.cudaEvent_t>(<cudaEvent_t>start)._ptr[0], <ccudart.cudaEvent_t>(<cudaEvent_t>end)._ptr[0])
    return (cudaError_t(err), ms)

@cython.embedsignature(True)
def cudaImportExternalMemory(memHandleDesc : cudaExternalMemoryHandleDesc):
    cdef cudaExternalMemory_t extMem_out = cudaExternalMemory_t()
    cdef ccudart.cudaExternalMemoryHandleDesc* cmemHandleDesc_ptr = memHandleDesc._ptr if memHandleDesc != None else NULL
    err = ccudart.cudaImportExternalMemory(extMem_out._ptr, cmemHandleDesc_ptr)
    return (cudaError_t(err), extMem_out)

@cython.embedsignature(True)
def cudaExternalMemoryGetMappedBuffer(extMem not None : cudaExternalMemory_t, bufferDesc : cudaExternalMemoryBufferDesc):
    cdef void_ptr devPtr = 0
    cdef ccudart.cudaExternalMemoryBufferDesc* cbufferDesc_ptr = bufferDesc._ptr if bufferDesc != None else NULL
    err = ccudart.cudaExternalMemoryGetMappedBuffer(<void**>&devPtr, extMem._ptr[0], cbufferDesc_ptr)
    return (cudaError_t(err), devPtr)

@cython.embedsignature(True)
def cudaExternalMemoryGetMappedMipmappedArray(extMem not None : cudaExternalMemory_t, mipmapDesc : cudaExternalMemoryMipmappedArrayDesc):
    cdef cudaMipmappedArray_t mipmap = cudaMipmappedArray_t()
    cdef ccudart.cudaExternalMemoryMipmappedArrayDesc* cmipmapDesc_ptr = mipmapDesc._ptr if mipmapDesc != None else NULL
    err = ccudart.cudaExternalMemoryGetMappedMipmappedArray(mipmap._ptr, extMem._ptr[0], cmipmapDesc_ptr)
    return (cudaError_t(err), mipmap)

@cython.embedsignature(True)
def cudaDestroyExternalMemory(extMem not None : cudaExternalMemory_t):
    err = ccudart.cudaDestroyExternalMemory(extMem._ptr[0])
    return (cudaError_t(err),)

@cython.embedsignature(True)
def cudaImportExternalSemaphore(semHandleDesc : cudaExternalSemaphoreHandleDesc):
    cdef cudaExternalSemaphore_t extSem_out = cudaExternalSemaphore_t()
    cdef ccudart.cudaExternalSemaphoreHandleDesc* csemHandleDesc_ptr = semHandleDesc._ptr if semHandleDesc != None else NULL
    err = ccudart.cudaImportExternalSemaphore(extSem_out._ptr, csemHandleDesc_ptr)
    return (cudaError_t(err), extSem_out)

@cython.embedsignature(True)
def cudaSignalExternalSemaphoresAsync(extSemArray : List[cudaExternalSemaphore_t], paramsArray : List[cudaExternalSemaphoreSignalParams], unsigned int numExtSems, stream):
    if not isinstance(stream, (cudaStream_t, cuda.CUstream)):
        raise TypeError("Argument 'stream' is not instance of type (expected <class 'cudapython.cudart.cudaStream_t, cuda.CUstream'>, found " + str(type(stream)))
    paramsArray = [] if paramsArray is None else paramsArray
    if not all(isinstance(_x, (cudaExternalSemaphoreSignalParams)) for _x in paramsArray):
        raise TypeError("Argument 'paramsArray' is not instance of type (expected List[cudapython.ccudart.cudaExternalSemaphoreSignalParams]")
    extSemArray = [] if extSemArray is None else extSemArray
    if not all(isinstance(_x, (cudaExternalSemaphore_t)) for _x in extSemArray):
        raise TypeError("Argument 'extSemArray' is not instance of type (expected List[cudapython.ccudart.cudaExternalSemaphore_t]")
    cdef ccudart.cudaExternalSemaphore_t* cextSemArray = NULL
    if len(extSemArray) > 0:
        cextSemArray = <ccudart.cudaExternalSemaphore_t*> calloc(len(extSemArray), sizeof(ccudart.cudaExternalSemaphore_t))
        if cextSemArray is NULL:
            raise MemoryError('Failed to allocate length x size memory: ' + str(len(extSemArray)) + 'x' + str(sizeof(ccudart.cudaExternalSemaphore_t)))
        else:
            for idx in range(len(extSemArray)):
                cextSemArray[idx] = (<cudaExternalSemaphore_t>extSemArray[idx])._ptr[0]

    cdef ccudart.cudaExternalSemaphoreSignalParams* cparamsArray = NULL
    if len(paramsArray) > 0:
        cparamsArray = <ccudart.cudaExternalSemaphoreSignalParams*> calloc(len(paramsArray), sizeof(ccudart.cudaExternalSemaphoreSignalParams))
        if cparamsArray is NULL:
            raise MemoryError('Failed to allocate length x size memory: ' + str(len(paramsArray)) + 'x' + str(sizeof(ccudart.cudaExternalSemaphoreSignalParams)))
        for idx in range(len(paramsArray)):
            memcpy(&cparamsArray[idx], (<cudaExternalSemaphoreSignalParams>paramsArray[idx])._ptr, sizeof(ccudart.cudaExternalSemaphoreSignalParams))

    if numExtSems > len(extSemArray): raise RuntimeError("List is too small: " + str(len(extSemArray)) + " < " + str(numExtSems))
    if numExtSems > len(paramsArray): raise RuntimeError("List is too small: " + str(len(paramsArray)) + " < " + str(numExtSems))
    err = ccudart.cudaSignalExternalSemaphoresAsync((<cudaExternalSemaphore_t>extSemArray[0])._ptr if len(extSemArray) == 1 else cextSemArray, (<cudaExternalSemaphoreSignalParams>paramsArray[0])._ptr if len(paramsArray) == 1 else cparamsArray, numExtSems, <ccudart.cudaStream_t>(<cudaStream_t>stream)._ptr[0])
    if cextSemArray is not NULL:
        free(cextSemArray)
    if cparamsArray is not NULL:
        free(cparamsArray)
    return (cudaError_t(err),)

@cython.embedsignature(True)
def cudaWaitExternalSemaphoresAsync(extSemArray : List[cudaExternalSemaphore_t], paramsArray : List[cudaExternalSemaphoreWaitParams], unsigned int numExtSems, stream):
    if not isinstance(stream, (cudaStream_t, cuda.CUstream)):
        raise TypeError("Argument 'stream' is not instance of type (expected <class 'cudapython.cudart.cudaStream_t, cuda.CUstream'>, found " + str(type(stream)))
    paramsArray = [] if paramsArray is None else paramsArray
    if not all(isinstance(_x, (cudaExternalSemaphoreWaitParams)) for _x in paramsArray):
        raise TypeError("Argument 'paramsArray' is not instance of type (expected List[cudapython.ccudart.cudaExternalSemaphoreWaitParams]")
    extSemArray = [] if extSemArray is None else extSemArray
    if not all(isinstance(_x, (cudaExternalSemaphore_t)) for _x in extSemArray):
        raise TypeError("Argument 'extSemArray' is not instance of type (expected List[cudapython.ccudart.cudaExternalSemaphore_t]")
    cdef ccudart.cudaExternalSemaphore_t* cextSemArray = NULL
    if len(extSemArray) > 0:
        cextSemArray = <ccudart.cudaExternalSemaphore_t*> calloc(len(extSemArray), sizeof(ccudart.cudaExternalSemaphore_t))
        if cextSemArray is NULL:
            raise MemoryError('Failed to allocate length x size memory: ' + str(len(extSemArray)) + 'x' + str(sizeof(ccudart.cudaExternalSemaphore_t)))
        else:
            for idx in range(len(extSemArray)):
                cextSemArray[idx] = (<cudaExternalSemaphore_t>extSemArray[idx])._ptr[0]

    cdef ccudart.cudaExternalSemaphoreWaitParams* cparamsArray = NULL
    if len(paramsArray) > 0:
        cparamsArray = <ccudart.cudaExternalSemaphoreWaitParams*> calloc(len(paramsArray), sizeof(ccudart.cudaExternalSemaphoreWaitParams))
        if cparamsArray is NULL:
            raise MemoryError('Failed to allocate length x size memory: ' + str(len(paramsArray)) + 'x' + str(sizeof(ccudart.cudaExternalSemaphoreWaitParams)))
        for idx in range(len(paramsArray)):
            memcpy(&cparamsArray[idx], (<cudaExternalSemaphoreWaitParams>paramsArray[idx])._ptr, sizeof(ccudart.cudaExternalSemaphoreWaitParams))

    if numExtSems > len(extSemArray): raise RuntimeError("List is too small: " + str(len(extSemArray)) + " < " + str(numExtSems))
    if numExtSems > len(paramsArray): raise RuntimeError("List is too small: " + str(len(paramsArray)) + " < " + str(numExtSems))
    err = ccudart.cudaWaitExternalSemaphoresAsync((<cudaExternalSemaphore_t>extSemArray[0])._ptr if len(extSemArray) == 1 else cextSemArray, (<cudaExternalSemaphoreWaitParams>paramsArray[0])._ptr if len(paramsArray) == 1 else cparamsArray, numExtSems, <ccudart.cudaStream_t>(<cudaStream_t>stream)._ptr[0])
    if cextSemArray is not NULL:
        free(cextSemArray)
    if cparamsArray is not NULL:
        free(cparamsArray)
    return (cudaError_t(err),)

@cython.embedsignature(True)
def cudaDestroyExternalSemaphore(extSem not None : cudaExternalSemaphore_t):
    err = ccudart.cudaDestroyExternalSemaphore(extSem._ptr[0])
    return (cudaError_t(err),)

@cython.embedsignature(True)
def cudaFuncSetCacheConfig(func, cacheConfig not None : cudaFuncCache):
    cfunc = utils.HelperInputVoidPtr(func)
    cdef void* cfunc_ptr = <void*><void_ptr>cfunc.cptr
    cdef ccudart.cudaFuncCache ccacheConfig = cacheConfig.value
    err = ccudart.cudaFuncSetCacheConfig(cfunc_ptr, ccacheConfig)
    return (cudaError_t(err),)

@cython.embedsignature(True)
def cudaFuncSetSharedMemConfig(func, config not None : cudaSharedMemConfig):
    cfunc = utils.HelperInputVoidPtr(func)
    cdef void* cfunc_ptr = <void*><void_ptr>cfunc.cptr
    cdef ccudart.cudaSharedMemConfig cconfig = config.value
    err = ccudart.cudaFuncSetSharedMemConfig(cfunc_ptr, cconfig)
    return (cudaError_t(err),)

@cython.embedsignature(True)
def cudaFuncGetAttributes(func):
    cdef cudaFuncAttributes attr = cudaFuncAttributes()
    cfunc = utils.HelperInputVoidPtr(func)
    cdef void* cfunc_ptr = <void*><void_ptr>cfunc.cptr
    err = ccudart.cudaFuncGetAttributes(attr._ptr, cfunc_ptr)
    return (cudaError_t(err), attr)

@cython.embedsignature(True)
def cudaFuncSetAttribute(func, attr not None : cudaFuncAttribute, int value):
    cfunc = utils.HelperInputVoidPtr(func)
    cdef void* cfunc_ptr = <void*><void_ptr>cfunc.cptr
    cdef ccudart.cudaFuncAttribute cattr = attr.value
    err = ccudart.cudaFuncSetAttribute(cfunc_ptr, cattr, value)
    return (cudaError_t(err),)

@cython.embedsignature(True)
def cudaSetDoubleForDevice(double d):
    err = ccudart.cudaSetDoubleForDevice(&d)
    return (cudaError_t(err),)

@cython.embedsignature(True)
def cudaSetDoubleForHost(double d):
    err = ccudart.cudaSetDoubleForHost(&d)
    return (cudaError_t(err),)

@cython.embedsignature(True)
def cudaLaunchHostFunc(stream, fn not None : cudaHostFn_t, userData):
    if not isinstance(stream, (cudaStream_t, cuda.CUstream)):
        raise TypeError("Argument 'stream' is not instance of type (expected <class 'cudapython.cudart.cudaStream_t, cuda.CUstream'>, found " + str(type(stream)))
    cuserData = utils.HelperInputVoidPtr(userData)
    cdef void* cuserData_ptr = <void*><void_ptr>cuserData.cptr
    with nogil:
        err = ccudart.cudaLaunchHostFunc(<ccudart.cudaStream_t>(<cudaStream_t>stream)._ptr[0], fn._ptr[0], cuserData_ptr)
    return (cudaError_t(err),)

@cython.embedsignature(True)
def cudaOccupancyMaxActiveBlocksPerMultiprocessor(func, int blockSize, size_t dynamicSMemSize):
    cdef int numBlocks = 0
    cfunc = utils.HelperInputVoidPtr(func)
    cdef void* cfunc_ptr = <void*><void_ptr>cfunc.cptr
    err = ccudart.cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks, cfunc_ptr, blockSize, dynamicSMemSize)
    return (cudaError_t(err), numBlocks)

@cython.embedsignature(True)
def cudaOccupancyAvailableDynamicSMemPerBlock(func, int numBlocks, int blockSize):
    cdef size_t dynamicSmemSize = 0
    cfunc = utils.HelperInputVoidPtr(func)
    cdef void* cfunc_ptr = <void*><void_ptr>cfunc.cptr
    err = ccudart.cudaOccupancyAvailableDynamicSMemPerBlock(&dynamicSmemSize, cfunc_ptr, numBlocks, blockSize)
    return (cudaError_t(err), dynamicSmemSize)

@cython.embedsignature(True)
def cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(func, int blockSize, size_t dynamicSMemSize, unsigned int flags):
    cdef int numBlocks = 0
    cfunc = utils.HelperInputVoidPtr(func)
    cdef void* cfunc_ptr = <void*><void_ptr>cfunc.cptr
    err = ccudart.cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(&numBlocks, cfunc_ptr, blockSize, dynamicSMemSize, flags)
    return (cudaError_t(err), numBlocks)

@cython.embedsignature(True)
def cudaMallocManaged(size_t size, unsigned int flags):
    cdef void_ptr devPtr = 0
    with nogil:
        err = ccudart.cudaMallocManaged(<void**>&devPtr, size, flags)
    return (cudaError_t(err), devPtr)

@cython.embedsignature(True)
def cudaMalloc(size_t size):
    cdef void_ptr devPtr = 0
    with nogil:
        err = ccudart.cudaMalloc(<void**>&devPtr, size)
    return (cudaError_t(err), devPtr)

@cython.embedsignature(True)
def cudaMallocHost(size_t size):
    cdef void_ptr ptr = 0
    err = ccudart.cudaMallocHost(<void**>&ptr, size)
    return (cudaError_t(err), ptr)

@cython.embedsignature(True)
def cudaMallocPitch(size_t width, size_t height):
    cdef void_ptr devPtr = 0
    cdef size_t pitch = 0
    err = ccudart.cudaMallocPitch(<void**>&devPtr, &pitch, width, height)
    return (cudaError_t(err), devPtr, pitch)

@cython.embedsignature(True)
def cudaMallocArray(desc : cudaChannelFormatDesc, size_t width, size_t height, unsigned int flags):
    cdef cudaArray_t array = cudaArray_t()
    cdef ccudart.cudaChannelFormatDesc* cdesc_ptr = desc._ptr if desc != None else NULL
    with nogil:
        err = ccudart.cudaMallocArray(array._ptr, cdesc_ptr, width, height, flags)
    return (cudaError_t(err), array)

@cython.embedsignature(True)
def cudaFree(devPtr):
    cdevPtr = utils.HelperInputVoidPtr(devPtr)
    cdef void* cdevPtr_ptr = <void*><void_ptr>cdevPtr.cptr
    with nogil:
        err = ccudart.cudaFree(cdevPtr_ptr)
    return (cudaError_t(err),)

@cython.embedsignature(True)
def cudaFreeHost(ptr):
    cptr = utils.HelperInputVoidPtr(ptr)
    cdef void* cptr_ptr = <void*><void_ptr>cptr.cptr
    with nogil:
        err = ccudart.cudaFreeHost(cptr_ptr)
    return (cudaError_t(err),)

@cython.embedsignature(True)
def cudaFreeArray(array not None : cudaArray_t):
    with nogil:
        err = ccudart.cudaFreeArray(array._ptr[0])
    return (cudaError_t(err),)

@cython.embedsignature(True)
def cudaFreeMipmappedArray(mipmappedArray not None : cudaMipmappedArray_t):
    err = ccudart.cudaFreeMipmappedArray(mipmappedArray._ptr[0])
    return (cudaError_t(err),)

@cython.embedsignature(True)
def cudaHostAlloc(size_t size, unsigned int flags):
    cdef void_ptr pHost = 0
    with nogil:
        err = ccudart.cudaHostAlloc(<void**>&pHost, size, flags)
    return (cudaError_t(err), pHost)

@cython.embedsignature(True)
def cudaHostRegister(ptr, size_t size, unsigned int flags):
    cptr = utils.HelperInputVoidPtr(ptr)
    cdef void* cptr_ptr = <void*><void_ptr>cptr.cptr
    with nogil:
        err = ccudart.cudaHostRegister(cptr_ptr, size, flags)
    return (cudaError_t(err),)

@cython.embedsignature(True)
def cudaHostUnregister(ptr):
    cptr = utils.HelperInputVoidPtr(ptr)
    cdef void* cptr_ptr = <void*><void_ptr>cptr.cptr
    with nogil:
        err = ccudart.cudaHostUnregister(cptr_ptr)
    return (cudaError_t(err),)

@cython.embedsignature(True)
def cudaHostGetDevicePointer(pHost, unsigned int flags):
    cdef void_ptr pDevice = 0
    cpHost = utils.HelperInputVoidPtr(pHost)
    cdef void* cpHost_ptr = <void*><void_ptr>cpHost.cptr
    err = ccudart.cudaHostGetDevicePointer(<void**>&pDevice, cpHost_ptr, flags)
    return (cudaError_t(err), pDevice)

@cython.embedsignature(True)
def cudaHostGetFlags(pHost):
    cdef unsigned int pFlags = 0
    cpHost = utils.HelperInputVoidPtr(pHost)
    cdef void* cpHost_ptr = <void*><void_ptr>cpHost.cptr
    err = ccudart.cudaHostGetFlags(&pFlags, cpHost_ptr)
    return (cudaError_t(err), pFlags)

@cython.embedsignature(True)
def cudaMalloc3D(extent not None : cudaExtent):
    cdef cudaPitchedPtr pitchedDevPtr = cudaPitchedPtr()
    err = ccudart.cudaMalloc3D(pitchedDevPtr._ptr, extent._ptr[0])
    return (cudaError_t(err), pitchedDevPtr)

@cython.embedsignature(True)
def cudaMalloc3DArray(desc : cudaChannelFormatDesc, extent not None : cudaExtent, unsigned int flags):
    cdef cudaArray_t array = cudaArray_t()
    cdef ccudart.cudaChannelFormatDesc* cdesc_ptr = desc._ptr if desc != None else NULL
    with nogil:
        err = ccudart.cudaMalloc3DArray(array._ptr, cdesc_ptr, extent._ptr[0], flags)
    return (cudaError_t(err), array)

@cython.embedsignature(True)
def cudaMallocMipmappedArray(desc : cudaChannelFormatDesc, extent not None : cudaExtent, unsigned int numLevels, unsigned int flags):
    cdef cudaMipmappedArray_t mipmappedArray = cudaMipmappedArray_t()
    cdef ccudart.cudaChannelFormatDesc* cdesc_ptr = desc._ptr if desc != None else NULL
    err = ccudart.cudaMallocMipmappedArray(mipmappedArray._ptr, cdesc_ptr, extent._ptr[0], numLevels, flags)
    return (cudaError_t(err), mipmappedArray)

@cython.embedsignature(True)
def cudaGetMipmappedArrayLevel(mipmappedArray not None : cudaMipmappedArray_const_t, unsigned int level):
    cdef cudaArray_t levelArray = cudaArray_t()
    err = ccudart.cudaGetMipmappedArrayLevel(levelArray._ptr, mipmappedArray._ptr[0], level)
    return (cudaError_t(err), levelArray)

@cython.embedsignature(True)
def cudaMemcpy3D(p : cudaMemcpy3DParms):
    cdef ccudart.cudaMemcpy3DParms* cp_ptr = p._ptr if p != None else NULL
    with nogil:
        err = ccudart.cudaMemcpy3D(cp_ptr)
    return (cudaError_t(err),)

@cython.embedsignature(True)
def cudaMemcpy3DPeer(p : cudaMemcpy3DPeerParms):
    cdef ccudart.cudaMemcpy3DPeerParms* cp_ptr = p._ptr if p != None else NULL
    err = ccudart.cudaMemcpy3DPeer(cp_ptr)
    return (cudaError_t(err),)

@cython.embedsignature(True)
def cudaMemcpy3DAsync(p : cudaMemcpy3DParms, stream):
    if not isinstance(stream, (cudaStream_t, cuda.CUstream)):
        raise TypeError("Argument 'stream' is not instance of type (expected <class 'cudapython.cudart.cudaStream_t, cuda.CUstream'>, found " + str(type(stream)))
    cdef ccudart.cudaMemcpy3DParms* cp_ptr = p._ptr if p != None else NULL
    with nogil:
        err = ccudart.cudaMemcpy3DAsync(cp_ptr, <ccudart.cudaStream_t>(<cudaStream_t>stream)._ptr[0])
    return (cudaError_t(err),)

@cython.embedsignature(True)
def cudaMemcpy3DPeerAsync(p : cudaMemcpy3DPeerParms, stream):
    if not isinstance(stream, (cudaStream_t, cuda.CUstream)):
        raise TypeError("Argument 'stream' is not instance of type (expected <class 'cudapython.cudart.cudaStream_t, cuda.CUstream'>, found " + str(type(stream)))
    cdef ccudart.cudaMemcpy3DPeerParms* cp_ptr = p._ptr if p != None else NULL
    err = ccudart.cudaMemcpy3DPeerAsync(cp_ptr, <ccudart.cudaStream_t>(<cudaStream_t>stream)._ptr[0])
    return (cudaError_t(err),)

@cython.embedsignature(True)
def cudaMemGetInfo():
    cdef size_t free = 0
    cdef size_t total = 0
    err = ccudart.cudaMemGetInfo(&free, &total)
    return (cudaError_t(err), free, total)

@cython.embedsignature(True)
def cudaArrayGetInfo(array not None : cudaArray_t):
    cdef cudaChannelFormatDesc desc = cudaChannelFormatDesc()
    cdef cudaExtent extent = cudaExtent()
    cdef unsigned int flags = 0
    err = ccudart.cudaArrayGetInfo(desc._ptr, extent._ptr, &flags, array._ptr[0])
    return (cudaError_t(err), desc, extent, flags)

@cython.embedsignature(True)
def cudaArrayGetPlane(hArray not None : cudaArray_t, unsigned int planeIdx):
    cdef cudaArray_t pPlaneArray = cudaArray_t()
    err = ccudart.cudaArrayGetPlane(pPlaneArray._ptr, hArray._ptr[0], planeIdx)
    return (cudaError_t(err), pPlaneArray)

@cython.embedsignature(True)
def cudaArrayGetSparseProperties(array not None : cudaArray_t):
    cdef cudaArraySparseProperties sparseProperties = cudaArraySparseProperties()
    err = ccudart.cudaArrayGetSparseProperties(sparseProperties._ptr, array._ptr[0])
    return (cudaError_t(err), sparseProperties)

@cython.embedsignature(True)
def cudaMipmappedArrayGetSparseProperties(mipmap not None : cudaMipmappedArray_t):
    cdef cudaArraySparseProperties sparseProperties = cudaArraySparseProperties()
    err = ccudart.cudaMipmappedArrayGetSparseProperties(sparseProperties._ptr, mipmap._ptr[0])
    return (cudaError_t(err), sparseProperties)

@cython.embedsignature(True)
def cudaMemcpy(dst, src, size_t count, kind not None : cudaMemcpyKind):
    cdst = utils.HelperInputVoidPtr(dst)
    cdef void* cdst_ptr = <void*><void_ptr>cdst.cptr
    csrc = utils.HelperInputVoidPtr(src)
    cdef void* csrc_ptr = <void*><void_ptr>csrc.cptr
    cdef ccudart.cudaMemcpyKind ckind = kind.value
    with nogil:
        err = ccudart.cudaMemcpy(cdst_ptr, csrc_ptr, count, ckind)
    return (cudaError_t(err),)

@cython.embedsignature(True)
def cudaMemcpyPeer(dst, int dstDevice, src, int srcDevice, size_t count):
    cdst = utils.HelperInputVoidPtr(dst)
    cdef void* cdst_ptr = <void*><void_ptr>cdst.cptr
    csrc = utils.HelperInputVoidPtr(src)
    cdef void* csrc_ptr = <void*><void_ptr>csrc.cptr
    with nogil:
        err = ccudart.cudaMemcpyPeer(cdst_ptr, dstDevice, csrc_ptr, srcDevice, count)
    return (cudaError_t(err),)

@cython.embedsignature(True)
def cudaMemcpy2D(dst, size_t dpitch, src, size_t spitch, size_t width, size_t height, kind not None : cudaMemcpyKind):
    cdst = utils.HelperInputVoidPtr(dst)
    cdef void* cdst_ptr = <void*><void_ptr>cdst.cptr
    csrc = utils.HelperInputVoidPtr(src)
    cdef void* csrc_ptr = <void*><void_ptr>csrc.cptr
    cdef ccudart.cudaMemcpyKind ckind = kind.value
    with nogil:
        err = ccudart.cudaMemcpy2D(cdst_ptr, dpitch, csrc_ptr, spitch, width, height, ckind)
    return (cudaError_t(err),)

@cython.embedsignature(True)
def cudaMemcpy2DToArray(dst not None : cudaArray_t, size_t wOffset, size_t hOffset, src, size_t spitch, size_t width, size_t height, kind not None : cudaMemcpyKind):
    csrc = utils.HelperInputVoidPtr(src)
    cdef void* csrc_ptr = <void*><void_ptr>csrc.cptr
    cdef ccudart.cudaMemcpyKind ckind = kind.value
    with nogil:
        err = ccudart.cudaMemcpy2DToArray(dst._ptr[0], wOffset, hOffset, csrc_ptr, spitch, width, height, ckind)
    return (cudaError_t(err),)

@cython.embedsignature(True)
def cudaMemcpy2DFromArray(dst, size_t dpitch, src not None : cudaArray_const_t, size_t wOffset, size_t hOffset, size_t width, size_t height, kind not None : cudaMemcpyKind):
    cdst = utils.HelperInputVoidPtr(dst)
    cdef void* cdst_ptr = <void*><void_ptr>cdst.cptr
    cdef ccudart.cudaMemcpyKind ckind = kind.value
    with nogil:
        err = ccudart.cudaMemcpy2DFromArray(cdst_ptr, dpitch, src._ptr[0], wOffset, hOffset, width, height, ckind)
    return (cudaError_t(err),)

@cython.embedsignature(True)
def cudaMemcpy2DArrayToArray(dst not None : cudaArray_t, size_t wOffsetDst, size_t hOffsetDst, src not None : cudaArray_const_t, size_t wOffsetSrc, size_t hOffsetSrc, size_t width, size_t height, kind not None : cudaMemcpyKind):
    cdef ccudart.cudaMemcpyKind ckind = kind.value
    err = ccudart.cudaMemcpy2DArrayToArray(dst._ptr[0], wOffsetDst, hOffsetDst, src._ptr[0], wOffsetSrc, hOffsetSrc, width, height, ckind)
    return (cudaError_t(err),)

@cython.embedsignature(True)
def cudaMemcpyAsync(dst, src, size_t count, kind not None : cudaMemcpyKind, stream):
    if not isinstance(stream, (cudaStream_t, cuda.CUstream)):
        raise TypeError("Argument 'stream' is not instance of type (expected <class 'cudapython.cudart.cudaStream_t, cuda.CUstream'>, found " + str(type(stream)))
    cdst = utils.HelperInputVoidPtr(dst)
    cdef void* cdst_ptr = <void*><void_ptr>cdst.cptr
    csrc = utils.HelperInputVoidPtr(src)
    cdef void* csrc_ptr = <void*><void_ptr>csrc.cptr
    cdef ccudart.cudaMemcpyKind ckind = kind.value
    with nogil:
        err = ccudart.cudaMemcpyAsync(cdst_ptr, csrc_ptr, count, ckind, <ccudart.cudaStream_t>(<cudaStream_t>stream)._ptr[0])
    return (cudaError_t(err),)

@cython.embedsignature(True)
def cudaMemcpyPeerAsync(dst, int dstDevice, src, int srcDevice, size_t count, stream):
    if not isinstance(stream, (cudaStream_t, cuda.CUstream)):
        raise TypeError("Argument 'stream' is not instance of type (expected <class 'cudapython.cudart.cudaStream_t, cuda.CUstream'>, found " + str(type(stream)))
    cdst = utils.HelperInputVoidPtr(dst)
    cdef void* cdst_ptr = <void*><void_ptr>cdst.cptr
    csrc = utils.HelperInputVoidPtr(src)
    cdef void* csrc_ptr = <void*><void_ptr>csrc.cptr
    with nogil:
        err = ccudart.cudaMemcpyPeerAsync(cdst_ptr, dstDevice, csrc_ptr, srcDevice, count, <ccudart.cudaStream_t>(<cudaStream_t>stream)._ptr[0])
    return (cudaError_t(err),)

@cython.embedsignature(True)
def cudaMemcpy2DAsync(dst, size_t dpitch, src, size_t spitch, size_t width, size_t height, kind not None : cudaMemcpyKind, stream):
    if not isinstance(stream, (cudaStream_t, cuda.CUstream)):
        raise TypeError("Argument 'stream' is not instance of type (expected <class 'cudapython.cudart.cudaStream_t, cuda.CUstream'>, found " + str(type(stream)))
    cdst = utils.HelperInputVoidPtr(dst)
    cdef void* cdst_ptr = <void*><void_ptr>cdst.cptr
    csrc = utils.HelperInputVoidPtr(src)
    cdef void* csrc_ptr = <void*><void_ptr>csrc.cptr
    cdef ccudart.cudaMemcpyKind ckind = kind.value
    with nogil:
        err = ccudart.cudaMemcpy2DAsync(cdst_ptr, dpitch, csrc_ptr, spitch, width, height, ckind, <ccudart.cudaStream_t>(<cudaStream_t>stream)._ptr[0])
    return (cudaError_t(err),)

@cython.embedsignature(True)
def cudaMemcpy2DToArrayAsync(dst not None : cudaArray_t, size_t wOffset, size_t hOffset, src, size_t spitch, size_t width, size_t height, kind not None : cudaMemcpyKind, stream):
    if not isinstance(stream, (cudaStream_t, cuda.CUstream)):
        raise TypeError("Argument 'stream' is not instance of type (expected <class 'cudapython.cudart.cudaStream_t, cuda.CUstream'>, found " + str(type(stream)))
    csrc = utils.HelperInputVoidPtr(src)
    cdef void* csrc_ptr = <void*><void_ptr>csrc.cptr
    cdef ccudart.cudaMemcpyKind ckind = kind.value
    with nogil:
        err = ccudart.cudaMemcpy2DToArrayAsync(dst._ptr[0], wOffset, hOffset, csrc_ptr, spitch, width, height, ckind, <ccudart.cudaStream_t>(<cudaStream_t>stream)._ptr[0])
    return (cudaError_t(err),)

@cython.embedsignature(True)
def cudaMemcpy2DFromArrayAsync(dst, size_t dpitch, src not None : cudaArray_const_t, size_t wOffset, size_t hOffset, size_t width, size_t height, kind not None : cudaMemcpyKind, stream):
    if not isinstance(stream, (cudaStream_t, cuda.CUstream)):
        raise TypeError("Argument 'stream' is not instance of type (expected <class 'cudapython.cudart.cudaStream_t, cuda.CUstream'>, found " + str(type(stream)))
    cdst = utils.HelperInputVoidPtr(dst)
    cdef void* cdst_ptr = <void*><void_ptr>cdst.cptr
    cdef ccudart.cudaMemcpyKind ckind = kind.value
    with nogil:
        err = ccudart.cudaMemcpy2DFromArrayAsync(cdst_ptr, dpitch, src._ptr[0], wOffset, hOffset, width, height, ckind, <ccudart.cudaStream_t>(<cudaStream_t>stream)._ptr[0])
    return (cudaError_t(err),)

@cython.embedsignature(True)
def cudaMemset(devPtr, int value, size_t count):
    cdevPtr = utils.HelperInputVoidPtr(devPtr)
    cdef void* cdevPtr_ptr = <void*><void_ptr>cdevPtr.cptr
    with nogil:
        err = ccudart.cudaMemset(cdevPtr_ptr, value, count)
    return (cudaError_t(err),)

@cython.embedsignature(True)
def cudaMemset2D(devPtr, size_t pitch, int value, size_t width, size_t height):
    cdevPtr = utils.HelperInputVoidPtr(devPtr)
    cdef void* cdevPtr_ptr = <void*><void_ptr>cdevPtr.cptr
    err = ccudart.cudaMemset2D(cdevPtr_ptr, pitch, value, width, height)
    return (cudaError_t(err),)

@cython.embedsignature(True)
def cudaMemset3D(pitchedDevPtr not None : cudaPitchedPtr, int value, extent not None : cudaExtent):
    err = ccudart.cudaMemset3D(pitchedDevPtr._ptr[0], value, extent._ptr[0])
    return (cudaError_t(err),)

@cython.embedsignature(True)
def cudaMemsetAsync(devPtr, int value, size_t count, stream):
    if not isinstance(stream, (cudaStream_t, cuda.CUstream)):
        raise TypeError("Argument 'stream' is not instance of type (expected <class 'cudapython.cudart.cudaStream_t, cuda.CUstream'>, found " + str(type(stream)))
    cdevPtr = utils.HelperInputVoidPtr(devPtr)
    cdef void* cdevPtr_ptr = <void*><void_ptr>cdevPtr.cptr
    with nogil:
        err = ccudart.cudaMemsetAsync(cdevPtr_ptr, value, count, <ccudart.cudaStream_t>(<cudaStream_t>stream)._ptr[0])
    return (cudaError_t(err),)

@cython.embedsignature(True)
def cudaMemset2DAsync(devPtr, size_t pitch, int value, size_t width, size_t height, stream):
    if not isinstance(stream, (cudaStream_t, cuda.CUstream)):
        raise TypeError("Argument 'stream' is not instance of type (expected <class 'cudapython.cudart.cudaStream_t, cuda.CUstream'>, found " + str(type(stream)))
    cdevPtr = utils.HelperInputVoidPtr(devPtr)
    cdef void* cdevPtr_ptr = <void*><void_ptr>cdevPtr.cptr
    err = ccudart.cudaMemset2DAsync(cdevPtr_ptr, pitch, value, width, height, <ccudart.cudaStream_t>(<cudaStream_t>stream)._ptr[0])
    return (cudaError_t(err),)

@cython.embedsignature(True)
def cudaMemset3DAsync(pitchedDevPtr not None : cudaPitchedPtr, int value, extent not None : cudaExtent, stream):
    if not isinstance(stream, (cudaStream_t, cuda.CUstream)):
        raise TypeError("Argument 'stream' is not instance of type (expected <class 'cudapython.cudart.cudaStream_t, cuda.CUstream'>, found " + str(type(stream)))
    err = ccudart.cudaMemset3DAsync(pitchedDevPtr._ptr[0], value, extent._ptr[0], <ccudart.cudaStream_t>(<cudaStream_t>stream)._ptr[0])
    return (cudaError_t(err),)

@cython.embedsignature(True)
def cudaMemPrefetchAsync(devPtr, size_t count, int dstDevice, stream):
    if not isinstance(stream, (cudaStream_t, cuda.CUstream)):
        raise TypeError("Argument 'stream' is not instance of type (expected <class 'cudapython.cudart.cudaStream_t, cuda.CUstream'>, found " + str(type(stream)))
    cdevPtr = utils.HelperInputVoidPtr(devPtr)
    cdef void* cdevPtr_ptr = <void*><void_ptr>cdevPtr.cptr
    with nogil:
        err = ccudart.cudaMemPrefetchAsync(cdevPtr_ptr, count, dstDevice, <ccudart.cudaStream_t>(<cudaStream_t>stream)._ptr[0])
    return (cudaError_t(err),)

@cython.embedsignature(True)
def cudaMemAdvise(devPtr, size_t count, advice not None : cudaMemoryAdvise, int device):
    cdevPtr = utils.HelperInputVoidPtr(devPtr)
    cdef void* cdevPtr_ptr = <void*><void_ptr>cdevPtr.cptr
    cdef ccudart.cudaMemoryAdvise cadvice = advice.value
    with nogil:
        err = ccudart.cudaMemAdvise(cdevPtr_ptr, count, cadvice, device)
    return (cudaError_t(err),)

@cython.embedsignature(True)
def cudaMemRangeGetAttribute(size_t dataSize, attribute not None : cudaMemRangeAttribute, devPtr, size_t count):
    cdef utils.HelperCUmem_range_attribute cdata = utils.HelperCUmem_range_attribute(attribute, dataSize)
    cdef void* cdata_ptr = <void*><void_ptr>cdata.cptr
    cdef ccudart.cudaMemRangeAttribute cattribute = attribute.value
    cdevPtr = utils.HelperInputVoidPtr(devPtr)
    cdef void* cdevPtr_ptr = <void*><void_ptr>cdevPtr.cptr
    err = ccudart.cudaMemRangeGetAttribute(cdata_ptr, dataSize, cattribute, cdevPtr_ptr, count)
    return (cudaError_t(err), cdata.pyObj())

@cython.embedsignature(True)
def cudaMemRangeGetAttributes(dataSizes : List[int], attributes : List[cudaMemRangeAttribute], size_t numAttributes, devPtr, size_t count):
    attributes = [] if attributes is None else attributes
    if not all(isinstance(_x, (cudaMemRangeAttribute)) for _x in attributes):
        raise TypeError("Argument 'attributes' is not instance of type (expected List[cudapython.ccudart.cudaMemRangeAttribute]")
    if not all(isinstance(_x, (int)) for _x in dataSizes):
        raise TypeError("Argument 'dataSizes' is not instance of type (expected List[int]")
    pylist = [utils.HelperCUmem_range_attribute(pyattributes, pydataSizes) for (pyattributes, pydataSizes) in zip(attributes, dataSizes)]
    cdef utils.InputVoidPtrPtrHelper voidStarHelper = utils.InputVoidPtrPtrHelper(pylist)
    cdef void** cvoidStarHelper_ptr = <void**><void_ptr>voidStarHelper.cptr
    cdef vector[size_t] cdataSizes = dataSizes
    cdef vector[ccudart.cudaMemRangeAttribute] cattributes = [pyattributes.value for pyattributes in (attributes)]
    if numAttributes > <size_t>len(attributes): raise RuntimeError("List is too small: " + str(len(attributes)) + " < " + str(numAttributes))
    if numAttributes > <size_t>len(numAttributes): raise RuntimeError("List is too small: " + str(len(numAttributes)) + " < " + str(numAttributes))
    cdevPtr = utils.HelperInputVoidPtr(devPtr)
    cdef void* cdevPtr_ptr = <void*><void_ptr>cdevPtr.cptr
    err = ccudart.cudaMemRangeGetAttributes(cvoidStarHelper_ptr, cdataSizes.data(), cattributes.data(), numAttributes, cdevPtr_ptr, count)
    return (cudaError_t(err), [obj.pyObj() for obj in pylist])

@cython.embedsignature(True)
def cudaMemcpyToArray(dst not None : cudaArray_t, size_t wOffset, size_t hOffset, src, size_t count, kind not None : cudaMemcpyKind):
    csrc = utils.HelperInputVoidPtr(src)
    cdef void* csrc_ptr = <void*><void_ptr>csrc.cptr
    cdef ccudart.cudaMemcpyKind ckind = kind.value
    err = ccudart.cudaMemcpyToArray(dst._ptr[0], wOffset, hOffset, csrc_ptr, count, ckind)
    return (cudaError_t(err),)

@cython.embedsignature(True)
def cudaMemcpyFromArray(dst, src not None : cudaArray_const_t, size_t wOffset, size_t hOffset, size_t count, kind not None : cudaMemcpyKind):
    cdst = utils.HelperInputVoidPtr(dst)
    cdef void* cdst_ptr = <void*><void_ptr>cdst.cptr
    cdef ccudart.cudaMemcpyKind ckind = kind.value
    err = ccudart.cudaMemcpyFromArray(cdst_ptr, src._ptr[0], wOffset, hOffset, count, ckind)
    return (cudaError_t(err),)

@cython.embedsignature(True)
def cudaMemcpyArrayToArray(dst not None : cudaArray_t, size_t wOffsetDst, size_t hOffsetDst, src not None : cudaArray_const_t, size_t wOffsetSrc, size_t hOffsetSrc, size_t count, kind not None : cudaMemcpyKind):
    cdef ccudart.cudaMemcpyKind ckind = kind.value
    err = ccudart.cudaMemcpyArrayToArray(dst._ptr[0], wOffsetDst, hOffsetDst, src._ptr[0], wOffsetSrc, hOffsetSrc, count, ckind)
    return (cudaError_t(err),)

@cython.embedsignature(True)
def cudaMemcpyToArrayAsync(dst not None : cudaArray_t, size_t wOffset, size_t hOffset, src, size_t count, kind not None : cudaMemcpyKind, stream):
    if not isinstance(stream, (cudaStream_t, cuda.CUstream)):
        raise TypeError("Argument 'stream' is not instance of type (expected <class 'cudapython.cudart.cudaStream_t, cuda.CUstream'>, found " + str(type(stream)))
    csrc = utils.HelperInputVoidPtr(src)
    cdef void* csrc_ptr = <void*><void_ptr>csrc.cptr
    cdef ccudart.cudaMemcpyKind ckind = kind.value
    err = ccudart.cudaMemcpyToArrayAsync(dst._ptr[0], wOffset, hOffset, csrc_ptr, count, ckind, <ccudart.cudaStream_t>(<cudaStream_t>stream)._ptr[0])
    return (cudaError_t(err),)

@cython.embedsignature(True)
def cudaMemcpyFromArrayAsync(dst, src not None : cudaArray_const_t, size_t wOffset, size_t hOffset, size_t count, kind not None : cudaMemcpyKind, stream):
    if not isinstance(stream, (cudaStream_t, cuda.CUstream)):
        raise TypeError("Argument 'stream' is not instance of type (expected <class 'cudapython.cudart.cudaStream_t, cuda.CUstream'>, found " + str(type(stream)))
    cdst = utils.HelperInputVoidPtr(dst)
    cdef void* cdst_ptr = <void*><void_ptr>cdst.cptr
    cdef ccudart.cudaMemcpyKind ckind = kind.value
    err = ccudart.cudaMemcpyFromArrayAsync(cdst_ptr, src._ptr[0], wOffset, hOffset, count, ckind, <ccudart.cudaStream_t>(<cudaStream_t>stream)._ptr[0])
    return (cudaError_t(err),)

@cython.embedsignature(True)
def cudaMallocAsync(size_t size, hStream):
    if not isinstance(hStream, (cudaStream_t, cuda.CUstream)):
        raise TypeError("Argument 'hStream' is not instance of type (expected <class 'cudapython.cudart.cudaStream_t, cuda.CUstream'>, found " + str(type(hStream)))
    cdef void_ptr devPtr = 0
    with nogil:
        err = ccudart.cudaMallocAsync(<void**>&devPtr, size, <ccudart.cudaStream_t>(<cudaStream_t>hStream)._ptr[0])
    return (cudaError_t(err), devPtr)

@cython.embedsignature(True)
def cudaFreeAsync(devPtr, hStream):
    if not isinstance(hStream, (cudaStream_t, cuda.CUstream)):
        raise TypeError("Argument 'hStream' is not instance of type (expected <class 'cudapython.cudart.cudaStream_t, cuda.CUstream'>, found " + str(type(hStream)))
    cdevPtr = utils.HelperInputVoidPtr(devPtr)
    cdef void* cdevPtr_ptr = <void*><void_ptr>cdevPtr.cptr
    with nogil:
        err = ccudart.cudaFreeAsync(cdevPtr_ptr, <ccudart.cudaStream_t>(<cudaStream_t>hStream)._ptr[0])
    return (cudaError_t(err),)

@cython.embedsignature(True)
def cudaMemPoolTrimTo(memPool, size_t minBytesToKeep):
    if not isinstance(memPool, (cudaMemPool_t, cuda.CUmemoryPool)):
        raise TypeError("Argument 'memPool' is not instance of type (expected <class 'cudapython.cudart.cudaMemPool_t, cuda.CUmemoryPool'>, found " + str(type(memPool)))
    with nogil:
        err = ccudart.cudaMemPoolTrimTo(<ccudart.cudaMemPool_t>(<cudaMemPool_t>memPool)._ptr[0], minBytesToKeep)
    return (cudaError_t(err),)

@cython.embedsignature(True)
def cudaMemPoolSetAttribute(memPool, attr not None : cudaMemPoolAttr, value):
    if not isinstance(memPool, (cudaMemPool_t, cuda.CUmemoryPool)):
        raise TypeError("Argument 'memPool' is not instance of type (expected <class 'cudapython.cudart.cudaMemPool_t, cuda.CUmemoryPool'>, found " + str(type(memPool)))
    cdef ccudart.cudaMemPoolAttr cattr = attr.value
    cdef utils.HelperCUmemPool_attribute cvalue = utils.HelperCUmemPool_attribute(attr, value, is_getter=False)
    cdef void* cvalue_ptr = <void*><void_ptr>cvalue.cptr
    with nogil:
        err = ccudart.cudaMemPoolSetAttribute(<ccudart.cudaMemPool_t>(<cudaMemPool_t>memPool)._ptr[0], cattr, cvalue_ptr)
    return (cudaError_t(err),)

@cython.embedsignature(True)
def cudaMemPoolGetAttribute(memPool, attr not None : cudaMemPoolAttr):
    if not isinstance(memPool, (cudaMemPool_t, cuda.CUmemoryPool)):
        raise TypeError("Argument 'memPool' is not instance of type (expected <class 'cudapython.cudart.cudaMemPool_t, cuda.CUmemoryPool'>, found " + str(type(memPool)))
    cdef ccudart.cudaMemPoolAttr cattr = attr.value
    cdef utils.HelperCUmemPool_attribute cvalue = utils.HelperCUmemPool_attribute(attr, 0, is_getter=True)
    cdef void* cvalue_ptr = <void*><void_ptr>cvalue.cptr
    with nogil:
        err = ccudart.cudaMemPoolGetAttribute(<ccudart.cudaMemPool_t>(<cudaMemPool_t>memPool)._ptr[0], cattr, cvalue_ptr)
    return (cudaError_t(err), cvalue.pyObj())

@cython.embedsignature(True)
def cudaMemPoolSetAccess(memPool, descList : List[cudaMemAccessDesc], size_t count):
    descList = [] if descList is None else descList
    if not all(isinstance(_x, (cudaMemAccessDesc)) for _x in descList):
        raise TypeError("Argument 'descList' is not instance of type (expected List[cudapython.ccudart.cudaMemAccessDesc]")
    if not isinstance(memPool, (cudaMemPool_t, cuda.CUmemoryPool)):
        raise TypeError("Argument 'memPool' is not instance of type (expected <class 'cudapython.cudart.cudaMemPool_t, cuda.CUmemoryPool'>, found " + str(type(memPool)))
    cdef ccudart.cudaMemAccessDesc* cdescList = NULL
    if len(descList) > 0:
        cdescList = <ccudart.cudaMemAccessDesc*> calloc(len(descList), sizeof(ccudart.cudaMemAccessDesc))
        if cdescList is NULL:
            raise MemoryError('Failed to allocate length x size memory: ' + str(len(descList)) + 'x' + str(sizeof(ccudart.cudaMemAccessDesc)))
        for idx in range(len(descList)):
            memcpy(&cdescList[idx], (<cudaMemAccessDesc>descList[idx])._ptr, sizeof(ccudart.cudaMemAccessDesc))

    if count > <size_t>len(descList): raise RuntimeError("List is too small: " + str(len(descList)) + " < " + str(count))
    err = ccudart.cudaMemPoolSetAccess(<ccudart.cudaMemPool_t>(<cudaMemPool_t>memPool)._ptr[0], (<cudaMemAccessDesc>descList[0])._ptr if len(descList) == 1 else cdescList, count)
    if cdescList is not NULL:
        free(cdescList)
    return (cudaError_t(err),)

@cython.embedsignature(True)
def cudaMemPoolGetAccess(memPool, location : cudaMemLocation):
    if not isinstance(memPool, (cudaMemPool_t, cuda.CUmemoryPool)):
        raise TypeError("Argument 'memPool' is not instance of type (expected <class 'cudapython.cudart.cudaMemPool_t, cuda.CUmemoryPool'>, found " + str(type(memPool)))
    cdef ccudart.cudaMemAccessFlags flags
    cdef ccudart.cudaMemLocation* clocation_ptr = location._ptr if location != None else NULL
    err = ccudart.cudaMemPoolGetAccess(&flags, <ccudart.cudaMemPool_t>(<cudaMemPool_t>memPool)._ptr[0], clocation_ptr)
    return (cudaError_t(err), cudaMemAccessFlags(flags))

@cython.embedsignature(True)
def cudaMemPoolCreate(poolProps : cudaMemPoolProps):
    cdef cudaMemPool_t memPool = cudaMemPool_t()
    cdef ccudart.cudaMemPoolProps* cpoolProps_ptr = poolProps._ptr if poolProps != None else NULL
    err = ccudart.cudaMemPoolCreate(<ccudart.cudaMemPool_t*>memPool._ptr, cpoolProps_ptr)
    return (cudaError_t(err), memPool)

@cython.embedsignature(True)
def cudaMemPoolDestroy(memPool):
    if not isinstance(memPool, (cudaMemPool_t, cuda.CUmemoryPool)):
        raise TypeError("Argument 'memPool' is not instance of type (expected <class 'cudapython.cudart.cudaMemPool_t, cuda.CUmemoryPool'>, found " + str(type(memPool)))
    err = ccudart.cudaMemPoolDestroy(<ccudart.cudaMemPool_t>(<cudaMemPool_t>memPool)._ptr[0])
    return (cudaError_t(err),)

@cython.embedsignature(True)
def cudaMallocFromPoolAsync(size_t size, memPool, stream):
    if not isinstance(stream, (cudaStream_t, cuda.CUstream)):
        raise TypeError("Argument 'stream' is not instance of type (expected <class 'cudapython.cudart.cudaStream_t, cuda.CUstream'>, found " + str(type(stream)))
    if not isinstance(memPool, (cudaMemPool_t, cuda.CUmemoryPool)):
        raise TypeError("Argument 'memPool' is not instance of type (expected <class 'cudapython.cudart.cudaMemPool_t, cuda.CUmemoryPool'>, found " + str(type(memPool)))
    cdef void_ptr ptr = 0
    err = ccudart.cudaMallocFromPoolAsync(<void**>&ptr, size, <ccudart.cudaMemPool_t>(<cudaMemPool_t>memPool)._ptr[0], <ccudart.cudaStream_t>(<cudaStream_t>stream)._ptr[0])
    return (cudaError_t(err), ptr)

@cython.embedsignature(True)
def cudaMemPoolExportToShareableHandle(memPool, handleType not None : cudaMemAllocationHandleType, unsigned int flags):
    if not isinstance(memPool, (cudaMemPool_t, cuda.CUmemoryPool)):
        raise TypeError("Argument 'memPool' is not instance of type (expected <class 'cudapython.cudart.cudaMemPool_t, cuda.CUmemoryPool'>, found " + str(type(memPool)))
    cdef void_ptr shareableHandle = 0
    cdef void* cshareableHandle_ptr = <void*>shareableHandle
    cdef ccudart.cudaMemAllocationHandleType chandleType = handleType.value
    err = ccudart.cudaMemPoolExportToShareableHandle(cshareableHandle_ptr, <ccudart.cudaMemPool_t>(<cudaMemPool_t>memPool)._ptr[0], chandleType, flags)
    return (cudaError_t(err), shareableHandle)

@cython.embedsignature(True)
def cudaMemPoolImportFromShareableHandle(shareableHandle, handleType not None : cudaMemAllocationHandleType, unsigned int flags):
    cdef cudaMemPool_t memPool = cudaMemPool_t()
    cshareableHandle = utils.HelperInputVoidPtr(shareableHandle)
    cdef void* cshareableHandle_ptr = <void*><void_ptr>cshareableHandle.cptr
    cdef ccudart.cudaMemAllocationHandleType chandleType = handleType.value
    err = ccudart.cudaMemPoolImportFromShareableHandle(<ccudart.cudaMemPool_t*>memPool._ptr, cshareableHandle_ptr, chandleType, flags)
    return (cudaError_t(err), memPool)

@cython.embedsignature(True)
def cudaMemPoolExportPointer(ptr):
    cdef cudaMemPoolPtrExportData exportData = cudaMemPoolPtrExportData()
    cptr = utils.HelperInputVoidPtr(ptr)
    cdef void* cptr_ptr = <void*><void_ptr>cptr.cptr
    err = ccudart.cudaMemPoolExportPointer(exportData._ptr, cptr_ptr)
    return (cudaError_t(err), exportData)

@cython.embedsignature(True)
def cudaMemPoolImportPointer(memPool, exportData : cudaMemPoolPtrExportData):
    if not isinstance(memPool, (cudaMemPool_t, cuda.CUmemoryPool)):
        raise TypeError("Argument 'memPool' is not instance of type (expected <class 'cudapython.cudart.cudaMemPool_t, cuda.CUmemoryPool'>, found " + str(type(memPool)))
    cdef void_ptr ptr = 0
    cdef ccudart.cudaMemPoolPtrExportData* cexportData_ptr = exportData._ptr if exportData != None else NULL
    err = ccudart.cudaMemPoolImportPointer(<void**>&ptr, <ccudart.cudaMemPool_t>(<cudaMemPool_t>memPool)._ptr[0], cexportData_ptr)
    return (cudaError_t(err), ptr)

@cython.embedsignature(True)
def cudaPointerGetAttributes(ptr):
    cdef cudaPointerAttributes attributes = cudaPointerAttributes()
    cptr = utils.HelperInputVoidPtr(ptr)
    cdef void* cptr_ptr = <void*><void_ptr>cptr.cptr
    err = ccudart.cudaPointerGetAttributes(attributes._ptr, cptr_ptr)
    return (cudaError_t(err), attributes)

@cython.embedsignature(True)
def cudaDeviceCanAccessPeer(int device, int peerDevice):
    cdef int canAccessPeer = 0
    err = ccudart.cudaDeviceCanAccessPeer(&canAccessPeer, device, peerDevice)
    return (cudaError_t(err), canAccessPeer)

@cython.embedsignature(True)
def cudaDeviceEnablePeerAccess(int peerDevice, unsigned int flags):
    err = ccudart.cudaDeviceEnablePeerAccess(peerDevice, flags)
    return (cudaError_t(err),)

@cython.embedsignature(True)
def cudaDeviceDisablePeerAccess(int peerDevice):
    err = ccudart.cudaDeviceDisablePeerAccess(peerDevice)
    return (cudaError_t(err),)

@cython.embedsignature(True)
def cudaGraphicsUnregisterResource(resource not None : cudaGraphicsResource_t):
    err = ccudart.cudaGraphicsUnregisterResource(resource._ptr[0])
    return (cudaError_t(err),)

@cython.embedsignature(True)
def cudaGraphicsResourceSetMapFlags(resource not None : cudaGraphicsResource_t, unsigned int flags):
    err = ccudart.cudaGraphicsResourceSetMapFlags(resource._ptr[0], flags)
    return (cudaError_t(err),)

@cython.embedsignature(True)
def cudaGraphicsMapResources(int count, resources : cudaGraphicsResource_t, stream):
    if not isinstance(stream, (cudaStream_t, cuda.CUstream)):
        raise TypeError("Argument 'stream' is not instance of type (expected <class 'cudapython.cudart.cudaStream_t, cuda.CUstream'>, found " + str(type(stream)))
    cdef ccudart.cudaGraphicsResource_t* cresources_ptr = resources._ptr if resources != None else NULL
    err = ccudart.cudaGraphicsMapResources(count, cresources_ptr, <ccudart.cudaStream_t>(<cudaStream_t>stream)._ptr[0])
    return (cudaError_t(err),)

@cython.embedsignature(True)
def cudaGraphicsUnmapResources(int count, resources : cudaGraphicsResource_t, stream):
    if not isinstance(stream, (cudaStream_t, cuda.CUstream)):
        raise TypeError("Argument 'stream' is not instance of type (expected <class 'cudapython.cudart.cudaStream_t, cuda.CUstream'>, found " + str(type(stream)))
    cdef ccudart.cudaGraphicsResource_t* cresources_ptr = resources._ptr if resources != None else NULL
    err = ccudart.cudaGraphicsUnmapResources(count, cresources_ptr, <ccudart.cudaStream_t>(<cudaStream_t>stream)._ptr[0])
    return (cudaError_t(err),)

@cython.embedsignature(True)
def cudaGraphicsResourceGetMappedPointer(resource not None : cudaGraphicsResource_t):
    cdef void_ptr devPtr = 0
    cdef size_t size = 0
    err = ccudart.cudaGraphicsResourceGetMappedPointer(<void**>&devPtr, &size, resource._ptr[0])
    return (cudaError_t(err), devPtr, size)

@cython.embedsignature(True)
def cudaGraphicsSubResourceGetMappedArray(resource not None : cudaGraphicsResource_t, unsigned int arrayIndex, unsigned int mipLevel):
    cdef cudaArray_t array = cudaArray_t()
    err = ccudart.cudaGraphicsSubResourceGetMappedArray(array._ptr, resource._ptr[0], arrayIndex, mipLevel)
    return (cudaError_t(err), array)

@cython.embedsignature(True)
def cudaGraphicsResourceGetMappedMipmappedArray(resource not None : cudaGraphicsResource_t):
    cdef cudaMipmappedArray_t mipmappedArray = cudaMipmappedArray_t()
    err = ccudart.cudaGraphicsResourceGetMappedMipmappedArray(mipmappedArray._ptr, resource._ptr[0])
    return (cudaError_t(err), mipmappedArray)

@cython.embedsignature(True)
def cudaGetChannelDesc(array not None : cudaArray_const_t):
    cdef cudaChannelFormatDesc desc = cudaChannelFormatDesc()
    with nogil:
        err = ccudart.cudaGetChannelDesc(desc._ptr, array._ptr[0])
    return (cudaError_t(err), desc)

@cython.embedsignature(True)
def cudaCreateChannelDesc(int x, int y, int z, int w, f not None : cudaChannelFormatKind):
    cdef ccudart.cudaChannelFormatKind cf = f.value
    cdef ccudart.cudaChannelFormatDesc err
    err = ccudart.cudaCreateChannelDesc(x, y, z, w, cf)
    cdef cudaChannelFormatDesc wrapper = cudaChannelFormatDesc()
    wrapper._ptr[0] = err
    return (cudaError_t.cudaSuccess, wrapper)

@cython.embedsignature(True)
def cudaCreateTextureObject(pResDesc : cudaResourceDesc, pTexDesc : cudaTextureDesc, pResViewDesc : cudaResourceViewDesc):
    cdef cudaTextureObject_t pTexObject = cudaTextureObject_t()
    cdef ccudart.cudaResourceDesc* cpResDesc_ptr = pResDesc._ptr if pResDesc != None else NULL
    cdef ccudart.cudaTextureDesc* cpTexDesc_ptr = pTexDesc._ptr if pTexDesc != None else NULL
    cdef ccudart.cudaResourceViewDesc* cpResViewDesc_ptr = pResViewDesc._ptr if pResViewDesc != None else NULL
    err = ccudart.cudaCreateTextureObject(pTexObject._ptr, cpResDesc_ptr, cpTexDesc_ptr, cpResViewDesc_ptr)
    return (cudaError_t(err), pTexObject)

@cython.embedsignature(True)
def cudaDestroyTextureObject(texObject not None : cudaTextureObject_t):
    with nogil:
        err = ccudart.cudaDestroyTextureObject(texObject._ptr[0])
    return (cudaError_t(err),)

@cython.embedsignature(True)
def cudaGetTextureObjectResourceDesc(texObject not None : cudaTextureObject_t):
    cdef cudaResourceDesc pResDesc = cudaResourceDesc()
    with nogil:
        err = ccudart.cudaGetTextureObjectResourceDesc(pResDesc._ptr, texObject._ptr[0])
    return (cudaError_t(err), pResDesc)

@cython.embedsignature(True)
def cudaGetTextureObjectTextureDesc(texObject not None : cudaTextureObject_t):
    cdef cudaTextureDesc pTexDesc = cudaTextureDesc()
    with nogil:
        err = ccudart.cudaGetTextureObjectTextureDesc(pTexDesc._ptr, texObject._ptr[0])
    return (cudaError_t(err), pTexDesc)

@cython.embedsignature(True)
def cudaGetTextureObjectResourceViewDesc(texObject not None : cudaTextureObject_t):
    cdef cudaResourceViewDesc pResViewDesc = cudaResourceViewDesc()
    err = ccudart.cudaGetTextureObjectResourceViewDesc(pResViewDesc._ptr, texObject._ptr[0])
    return (cudaError_t(err), pResViewDesc)

@cython.embedsignature(True)
def cudaCreateSurfaceObject(pResDesc : cudaResourceDesc):
    cdef cudaSurfaceObject_t pSurfObject = cudaSurfaceObject_t()
    cdef ccudart.cudaResourceDesc* cpResDesc_ptr = pResDesc._ptr if pResDesc != None else NULL
    with nogil:
        err = ccudart.cudaCreateSurfaceObject(pSurfObject._ptr, cpResDesc_ptr)
    return (cudaError_t(err), pSurfObject)

@cython.embedsignature(True)
def cudaDestroySurfaceObject(surfObject not None : cudaSurfaceObject_t):
    with nogil:
        err = ccudart.cudaDestroySurfaceObject(surfObject._ptr[0])
    return (cudaError_t(err),)

@cython.embedsignature(True)
def cudaGetSurfaceObjectResourceDesc(surfObject not None : cudaSurfaceObject_t):
    cdef cudaResourceDesc pResDesc = cudaResourceDesc()
    err = ccudart.cudaGetSurfaceObjectResourceDesc(pResDesc._ptr, surfObject._ptr[0])
    return (cudaError_t(err), pResDesc)

@cython.embedsignature(True)
def cudaDriverGetVersion():
    cdef int driverVersion = 0
    err = ccudart.cudaDriverGetVersion(&driverVersion)
    return (cudaError_t(err), driverVersion)

@cython.embedsignature(True)
def cudaRuntimeGetVersion():
    cdef int runtimeVersion = 0
    err = ccudart.cudaRuntimeGetVersion(&runtimeVersion)
    return (cudaError_t(err), runtimeVersion)

@cython.embedsignature(True)
def cudaGraphCreate(unsigned int flags):
    cdef cudaGraph_t pGraph = cudaGraph_t()
    err = ccudart.cudaGraphCreate(<ccudart.cudaGraph_t*>pGraph._ptr, flags)
    return (cudaError_t(err), pGraph)

@cython.embedsignature(True)
def cudaGraphAddKernelNode(graph, pDependencies : List[cudaGraphNode_t], size_t numDependencies, pNodeParams : cudaKernelNodeParams):
    pDependencies = [] if pDependencies is None else pDependencies
    if not all(isinstance(_x, (cudaGraphNode_t, cuda.CUgraphNode)) for _x in pDependencies):
        raise TypeError("Argument 'pDependencies' is not instance of type (expected List[cudapython.ccudart.cudaGraphNode_t, cuda.CUgraphNode]")
    if not isinstance(graph, (cudaGraph_t, cuda.CUgraph)):
        raise TypeError("Argument 'graph' is not instance of type (expected <class 'cudapython.cudart.cudaGraph_t, cuda.CUgraph'>, found " + str(type(graph)))
    cdef cudaGraphNode_t pGraphNode = cudaGraphNode_t()
    cdef ccudart.cudaGraphNode_t* cpDependencies = NULL
    if len(pDependencies) > 0:
        cpDependencies = <ccudart.cudaGraphNode_t*> calloc(len(pDependencies), sizeof(ccudart.cudaGraphNode_t))
        if cpDependencies is NULL:
            raise MemoryError('Failed to allocate length x size memory: ' + str(len(pDependencies)) + 'x' + str(sizeof(ccudart.cudaGraphNode_t)))
        else:
            for idx in range(len(pDependencies)):
                cpDependencies[idx] = <ccudart.cudaGraphNode_t>(<cudaGraphNode_t>pDependencies[idx])._ptr[0]

    if numDependencies > <size_t>len(pDependencies): raise RuntimeError("List is too small: " + str(len(pDependencies)) + " < " + str(numDependencies))
    cdef ccudart.cudaKernelNodeParams* cpNodeParams_ptr = pNodeParams._ptr if pNodeParams != None else NULL
    err = ccudart.cudaGraphAddKernelNode(<ccudart.cudaGraphNode_t*>pGraphNode._ptr, <ccudart.cudaGraph_t>(<cudaGraph_t>graph)._ptr[0], <ccudart.cudaGraphNode_t*>(<cudaGraphNode_t>pDependencies[0])._ptr if len(pDependencies) == 1 else cpDependencies, numDependencies, cpNodeParams_ptr)
    if cpDependencies is not NULL:
        free(cpDependencies)
    return (cudaError_t(err), pGraphNode)

@cython.embedsignature(True)
def cudaGraphKernelNodeGetParams(node):
    if not isinstance(node, (cudaGraphNode_t, cuda.CUgraphNode)):
        raise TypeError("Argument 'node' is not instance of type (expected <class 'cudapython.cudart.cudaGraphNode_t, cuda.CUgraphNode'>, found " + str(type(node)))
    cdef cudaKernelNodeParams pNodeParams = cudaKernelNodeParams()
    err = ccudart.cudaGraphKernelNodeGetParams(<ccudart.cudaGraphNode_t>(<cudaGraphNode_t>node)._ptr[0], pNodeParams._ptr)
    return (cudaError_t(err), pNodeParams)

@cython.embedsignature(True)
def cudaGraphKernelNodeSetParams(node, pNodeParams : cudaKernelNodeParams):
    if not isinstance(node, (cudaGraphNode_t, cuda.CUgraphNode)):
        raise TypeError("Argument 'node' is not instance of type (expected <class 'cudapython.cudart.cudaGraphNode_t, cuda.CUgraphNode'>, found " + str(type(node)))
    cdef ccudart.cudaKernelNodeParams* cpNodeParams_ptr = pNodeParams._ptr if pNodeParams != None else NULL
    err = ccudart.cudaGraphKernelNodeSetParams(<ccudart.cudaGraphNode_t>(<cudaGraphNode_t>node)._ptr[0], cpNodeParams_ptr)
    return (cudaError_t(err),)

@cython.embedsignature(True)
def cudaGraphKernelNodeCopyAttributes(hSrc, hDst):
    if not isinstance(hDst, (cudaGraphNode_t, cuda.CUgraphNode)):
        raise TypeError("Argument 'hDst' is not instance of type (expected <class 'cudapython.cudart.cudaGraphNode_t, cuda.CUgraphNode'>, found " + str(type(hDst)))
    if not isinstance(hSrc, (cudaGraphNode_t, cuda.CUgraphNode)):
        raise TypeError("Argument 'hSrc' is not instance of type (expected <class 'cudapython.cudart.cudaGraphNode_t, cuda.CUgraphNode'>, found " + str(type(hSrc)))
    err = ccudart.cudaGraphKernelNodeCopyAttributes(<ccudart.cudaGraphNode_t>(<cudaGraphNode_t>hSrc)._ptr[0], <ccudart.cudaGraphNode_t>(<cudaGraphNode_t>hDst)._ptr[0])
    return (cudaError_t(err),)

@cython.embedsignature(True)
def cudaGraphKernelNodeGetAttribute(hNode, attr not None : cudaKernelNodeAttrID):
    if not isinstance(hNode, (cudaGraphNode_t, cuda.CUgraphNode)):
        raise TypeError("Argument 'hNode' is not instance of type (expected <class 'cudapython.cudart.cudaGraphNode_t, cuda.CUgraphNode'>, found " + str(type(hNode)))
    cdef ccudart.cudaKernelNodeAttrID cattr = attr.value
    cdef cudaKernelNodeAttrValue value_out = cudaKernelNodeAttrValue()
    err = ccudart.cudaGraphKernelNodeGetAttribute(<ccudart.cudaGraphNode_t>(<cudaGraphNode_t>hNode)._ptr[0], cattr, value_out._ptr)
    return (cudaError_t(err), value_out)

@cython.embedsignature(True)
def cudaGraphKernelNodeSetAttribute(hNode, attr not None : cudaKernelNodeAttrID, value : cudaKernelNodeAttrValue):
    if not isinstance(hNode, (cudaGraphNode_t, cuda.CUgraphNode)):
        raise TypeError("Argument 'hNode' is not instance of type (expected <class 'cudapython.cudart.cudaGraphNode_t, cuda.CUgraphNode'>, found " + str(type(hNode)))
    cdef ccudart.cudaKernelNodeAttrID cattr = attr.value
    cdef ccudart.cudaKernelNodeAttrValue* cvalue_ptr = value._ptr if value != None else NULL
    err = ccudart.cudaGraphKernelNodeSetAttribute(<ccudart.cudaGraphNode_t>(<cudaGraphNode_t>hNode)._ptr[0], cattr, cvalue_ptr)
    return (cudaError_t(err),)

@cython.embedsignature(True)
def cudaGraphAddMemcpyNode(graph, pDependencies : List[cudaGraphNode_t], size_t numDependencies, pCopyParams : cudaMemcpy3DParms):
    pDependencies = [] if pDependencies is None else pDependencies
    if not all(isinstance(_x, (cudaGraphNode_t, cuda.CUgraphNode)) for _x in pDependencies):
        raise TypeError("Argument 'pDependencies' is not instance of type (expected List[cudapython.ccudart.cudaGraphNode_t, cuda.CUgraphNode]")
    if not isinstance(graph, (cudaGraph_t, cuda.CUgraph)):
        raise TypeError("Argument 'graph' is not instance of type (expected <class 'cudapython.cudart.cudaGraph_t, cuda.CUgraph'>, found " + str(type(graph)))
    cdef cudaGraphNode_t pGraphNode = cudaGraphNode_t()
    cdef ccudart.cudaGraphNode_t* cpDependencies = NULL
    if len(pDependencies) > 0:
        cpDependencies = <ccudart.cudaGraphNode_t*> calloc(len(pDependencies), sizeof(ccudart.cudaGraphNode_t))
        if cpDependencies is NULL:
            raise MemoryError('Failed to allocate length x size memory: ' + str(len(pDependencies)) + 'x' + str(sizeof(ccudart.cudaGraphNode_t)))
        else:
            for idx in range(len(pDependencies)):
                cpDependencies[idx] = <ccudart.cudaGraphNode_t>(<cudaGraphNode_t>pDependencies[idx])._ptr[0]

    if numDependencies > <size_t>len(pDependencies): raise RuntimeError("List is too small: " + str(len(pDependencies)) + " < " + str(numDependencies))
    cdef ccudart.cudaMemcpy3DParms* cpCopyParams_ptr = pCopyParams._ptr if pCopyParams != None else NULL
    err = ccudart.cudaGraphAddMemcpyNode(<ccudart.cudaGraphNode_t*>pGraphNode._ptr, <ccudart.cudaGraph_t>(<cudaGraph_t>graph)._ptr[0], <ccudart.cudaGraphNode_t*>(<cudaGraphNode_t>pDependencies[0])._ptr if len(pDependencies) == 1 else cpDependencies, numDependencies, cpCopyParams_ptr)
    if cpDependencies is not NULL:
        free(cpDependencies)
    return (cudaError_t(err), pGraphNode)

@cython.embedsignature(True)
def cudaGraphAddMemcpyNode1D(graph, pDependencies : List[cudaGraphNode_t], size_t numDependencies, dst, src, size_t count, kind not None : cudaMemcpyKind):
    pDependencies = [] if pDependencies is None else pDependencies
    if not all(isinstance(_x, (cudaGraphNode_t, cuda.CUgraphNode)) for _x in pDependencies):
        raise TypeError("Argument 'pDependencies' is not instance of type (expected List[cudapython.ccudart.cudaGraphNode_t, cuda.CUgraphNode]")
    if not isinstance(graph, (cudaGraph_t, cuda.CUgraph)):
        raise TypeError("Argument 'graph' is not instance of type (expected <class 'cudapython.cudart.cudaGraph_t, cuda.CUgraph'>, found " + str(type(graph)))
    cdef cudaGraphNode_t pGraphNode = cudaGraphNode_t()
    cdef ccudart.cudaGraphNode_t* cpDependencies = NULL
    if len(pDependencies) > 0:
        cpDependencies = <ccudart.cudaGraphNode_t*> calloc(len(pDependencies), sizeof(ccudart.cudaGraphNode_t))
        if cpDependencies is NULL:
            raise MemoryError('Failed to allocate length x size memory: ' + str(len(pDependencies)) + 'x' + str(sizeof(ccudart.cudaGraphNode_t)))
        else:
            for idx in range(len(pDependencies)):
                cpDependencies[idx] = <ccudart.cudaGraphNode_t>(<cudaGraphNode_t>pDependencies[idx])._ptr[0]

    cdst = utils.HelperInputVoidPtr(dst)
    cdef void* cdst_ptr = <void*><void_ptr>cdst.cptr
    csrc = utils.HelperInputVoidPtr(src)
    cdef void* csrc_ptr = <void*><void_ptr>csrc.cptr
    cdef ccudart.cudaMemcpyKind ckind = kind.value
    err = ccudart.cudaGraphAddMemcpyNode1D(<ccudart.cudaGraphNode_t*>pGraphNode._ptr, <ccudart.cudaGraph_t>(<cudaGraph_t>graph)._ptr[0], <ccudart.cudaGraphNode_t*>(<cudaGraphNode_t>pDependencies[0])._ptr if len(pDependencies) == 1 else cpDependencies, numDependencies, cdst_ptr, csrc_ptr, count, ckind)
    if cpDependencies is not NULL:
        free(cpDependencies)
    return (cudaError_t(err), pGraphNode)

@cython.embedsignature(True)
def cudaGraphMemcpyNodeGetParams(node):
    if not isinstance(node, (cudaGraphNode_t, cuda.CUgraphNode)):
        raise TypeError("Argument 'node' is not instance of type (expected <class 'cudapython.cudart.cudaGraphNode_t, cuda.CUgraphNode'>, found " + str(type(node)))
    cdef cudaMemcpy3DParms pNodeParams = cudaMemcpy3DParms()
    err = ccudart.cudaGraphMemcpyNodeGetParams(<ccudart.cudaGraphNode_t>(<cudaGraphNode_t>node)._ptr[0], pNodeParams._ptr)
    return (cudaError_t(err), pNodeParams)

@cython.embedsignature(True)
def cudaGraphMemcpyNodeSetParams(node, pNodeParams : cudaMemcpy3DParms):
    if not isinstance(node, (cudaGraphNode_t, cuda.CUgraphNode)):
        raise TypeError("Argument 'node' is not instance of type (expected <class 'cudapython.cudart.cudaGraphNode_t, cuda.CUgraphNode'>, found " + str(type(node)))
    cdef ccudart.cudaMemcpy3DParms* cpNodeParams_ptr = pNodeParams._ptr if pNodeParams != None else NULL
    err = ccudart.cudaGraphMemcpyNodeSetParams(<ccudart.cudaGraphNode_t>(<cudaGraphNode_t>node)._ptr[0], cpNodeParams_ptr)
    return (cudaError_t(err),)

@cython.embedsignature(True)
def cudaGraphMemcpyNodeSetParams1D(node, dst, src, size_t count, kind not None : cudaMemcpyKind):
    if not isinstance(node, (cudaGraphNode_t, cuda.CUgraphNode)):
        raise TypeError("Argument 'node' is not instance of type (expected <class 'cudapython.cudart.cudaGraphNode_t, cuda.CUgraphNode'>, found " + str(type(node)))
    cdst = utils.HelperInputVoidPtr(dst)
    cdef void* cdst_ptr = <void*><void_ptr>cdst.cptr
    csrc = utils.HelperInputVoidPtr(src)
    cdef void* csrc_ptr = <void*><void_ptr>csrc.cptr
    cdef ccudart.cudaMemcpyKind ckind = kind.value
    err = ccudart.cudaGraphMemcpyNodeSetParams1D(<ccudart.cudaGraphNode_t>(<cudaGraphNode_t>node)._ptr[0], cdst_ptr, csrc_ptr, count, ckind)
    return (cudaError_t(err),)

@cython.embedsignature(True)
def cudaGraphAddMemsetNode(graph, pDependencies : List[cudaGraphNode_t], size_t numDependencies, pMemsetParams : cudaMemsetParams):
    pDependencies = [] if pDependencies is None else pDependencies
    if not all(isinstance(_x, (cudaGraphNode_t, cuda.CUgraphNode)) for _x in pDependencies):
        raise TypeError("Argument 'pDependencies' is not instance of type (expected List[cudapython.ccudart.cudaGraphNode_t, cuda.CUgraphNode]")
    if not isinstance(graph, (cudaGraph_t, cuda.CUgraph)):
        raise TypeError("Argument 'graph' is not instance of type (expected <class 'cudapython.cudart.cudaGraph_t, cuda.CUgraph'>, found " + str(type(graph)))
    cdef cudaGraphNode_t pGraphNode = cudaGraphNode_t()
    cdef ccudart.cudaGraphNode_t* cpDependencies = NULL
    if len(pDependencies) > 0:
        cpDependencies = <ccudart.cudaGraphNode_t*> calloc(len(pDependencies), sizeof(ccudart.cudaGraphNode_t))
        if cpDependencies is NULL:
            raise MemoryError('Failed to allocate length x size memory: ' + str(len(pDependencies)) + 'x' + str(sizeof(ccudart.cudaGraphNode_t)))
        else:
            for idx in range(len(pDependencies)):
                cpDependencies[idx] = <ccudart.cudaGraphNode_t>(<cudaGraphNode_t>pDependencies[idx])._ptr[0]

    if numDependencies > <size_t>len(pDependencies): raise RuntimeError("List is too small: " + str(len(pDependencies)) + " < " + str(numDependencies))
    cdef ccudart.cudaMemsetParams* cpMemsetParams_ptr = pMemsetParams._ptr if pMemsetParams != None else NULL
    err = ccudart.cudaGraphAddMemsetNode(<ccudart.cudaGraphNode_t*>pGraphNode._ptr, <ccudart.cudaGraph_t>(<cudaGraph_t>graph)._ptr[0], <ccudart.cudaGraphNode_t*>(<cudaGraphNode_t>pDependencies[0])._ptr if len(pDependencies) == 1 else cpDependencies, numDependencies, cpMemsetParams_ptr)
    if cpDependencies is not NULL:
        free(cpDependencies)
    return (cudaError_t(err), pGraphNode)

@cython.embedsignature(True)
def cudaGraphMemsetNodeGetParams(node):
    if not isinstance(node, (cudaGraphNode_t, cuda.CUgraphNode)):
        raise TypeError("Argument 'node' is not instance of type (expected <class 'cudapython.cudart.cudaGraphNode_t, cuda.CUgraphNode'>, found " + str(type(node)))
    cdef cudaMemsetParams pNodeParams = cudaMemsetParams()
    err = ccudart.cudaGraphMemsetNodeGetParams(<ccudart.cudaGraphNode_t>(<cudaGraphNode_t>node)._ptr[0], pNodeParams._ptr)
    return (cudaError_t(err), pNodeParams)

@cython.embedsignature(True)
def cudaGraphMemsetNodeSetParams(node, pNodeParams : cudaMemsetParams):
    if not isinstance(node, (cudaGraphNode_t, cuda.CUgraphNode)):
        raise TypeError("Argument 'node' is not instance of type (expected <class 'cudapython.cudart.cudaGraphNode_t, cuda.CUgraphNode'>, found " + str(type(node)))
    cdef ccudart.cudaMemsetParams* cpNodeParams_ptr = pNodeParams._ptr if pNodeParams != None else NULL
    err = ccudart.cudaGraphMemsetNodeSetParams(<ccudart.cudaGraphNode_t>(<cudaGraphNode_t>node)._ptr[0], cpNodeParams_ptr)
    return (cudaError_t(err),)

@cython.embedsignature(True)
def cudaGraphAddHostNode(graph, pDependencies : List[cudaGraphNode_t], size_t numDependencies, pNodeParams : cudaHostNodeParams):
    pDependencies = [] if pDependencies is None else pDependencies
    if not all(isinstance(_x, (cudaGraphNode_t, cuda.CUgraphNode)) for _x in pDependencies):
        raise TypeError("Argument 'pDependencies' is not instance of type (expected List[cudapython.ccudart.cudaGraphNode_t, cuda.CUgraphNode]")
    if not isinstance(graph, (cudaGraph_t, cuda.CUgraph)):
        raise TypeError("Argument 'graph' is not instance of type (expected <class 'cudapython.cudart.cudaGraph_t, cuda.CUgraph'>, found " + str(type(graph)))
    cdef cudaGraphNode_t pGraphNode = cudaGraphNode_t()
    cdef ccudart.cudaGraphNode_t* cpDependencies = NULL
    if len(pDependencies) > 0:
        cpDependencies = <ccudart.cudaGraphNode_t*> calloc(len(pDependencies), sizeof(ccudart.cudaGraphNode_t))
        if cpDependencies is NULL:
            raise MemoryError('Failed to allocate length x size memory: ' + str(len(pDependencies)) + 'x' + str(sizeof(ccudart.cudaGraphNode_t)))
        else:
            for idx in range(len(pDependencies)):
                cpDependencies[idx] = <ccudart.cudaGraphNode_t>(<cudaGraphNode_t>pDependencies[idx])._ptr[0]

    if numDependencies > <size_t>len(pDependencies): raise RuntimeError("List is too small: " + str(len(pDependencies)) + " < " + str(numDependencies))
    cdef ccudart.cudaHostNodeParams* cpNodeParams_ptr = pNodeParams._ptr if pNodeParams != None else NULL
    err = ccudart.cudaGraphAddHostNode(<ccudart.cudaGraphNode_t*>pGraphNode._ptr, <ccudart.cudaGraph_t>(<cudaGraph_t>graph)._ptr[0], <ccudart.cudaGraphNode_t*>(<cudaGraphNode_t>pDependencies[0])._ptr if len(pDependencies) == 1 else cpDependencies, numDependencies, cpNodeParams_ptr)
    if cpDependencies is not NULL:
        free(cpDependencies)
    return (cudaError_t(err), pGraphNode)

@cython.embedsignature(True)
def cudaGraphHostNodeGetParams(node):
    if not isinstance(node, (cudaGraphNode_t, cuda.CUgraphNode)):
        raise TypeError("Argument 'node' is not instance of type (expected <class 'cudapython.cudart.cudaGraphNode_t, cuda.CUgraphNode'>, found " + str(type(node)))
    cdef cudaHostNodeParams pNodeParams = cudaHostNodeParams()
    err = ccudart.cudaGraphHostNodeGetParams(<ccudart.cudaGraphNode_t>(<cudaGraphNode_t>node)._ptr[0], pNodeParams._ptr)
    return (cudaError_t(err), pNodeParams)

@cython.embedsignature(True)
def cudaGraphHostNodeSetParams(node, pNodeParams : cudaHostNodeParams):
    if not isinstance(node, (cudaGraphNode_t, cuda.CUgraphNode)):
        raise TypeError("Argument 'node' is not instance of type (expected <class 'cudapython.cudart.cudaGraphNode_t, cuda.CUgraphNode'>, found " + str(type(node)))
    cdef ccudart.cudaHostNodeParams* cpNodeParams_ptr = pNodeParams._ptr if pNodeParams != None else NULL
    err = ccudart.cudaGraphHostNodeSetParams(<ccudart.cudaGraphNode_t>(<cudaGraphNode_t>node)._ptr[0], cpNodeParams_ptr)
    return (cudaError_t(err),)

@cython.embedsignature(True)
def cudaGraphAddChildGraphNode(graph, pDependencies : List[cudaGraphNode_t], size_t numDependencies, childGraph):
    if not isinstance(childGraph, (cudaGraph_t, cuda.CUgraph)):
        raise TypeError("Argument 'childGraph' is not instance of type (expected <class 'cudapython.cudart.cudaGraph_t, cuda.CUgraph'>, found " + str(type(childGraph)))
    pDependencies = [] if pDependencies is None else pDependencies
    if not all(isinstance(_x, (cudaGraphNode_t, cuda.CUgraphNode)) for _x in pDependencies):
        raise TypeError("Argument 'pDependencies' is not instance of type (expected List[cudapython.ccudart.cudaGraphNode_t, cuda.CUgraphNode]")
    if not isinstance(graph, (cudaGraph_t, cuda.CUgraph)):
        raise TypeError("Argument 'graph' is not instance of type (expected <class 'cudapython.cudart.cudaGraph_t, cuda.CUgraph'>, found " + str(type(graph)))
    cdef cudaGraphNode_t pGraphNode = cudaGraphNode_t()
    cdef ccudart.cudaGraphNode_t* cpDependencies = NULL
    if len(pDependencies) > 0:
        cpDependencies = <ccudart.cudaGraphNode_t*> calloc(len(pDependencies), sizeof(ccudart.cudaGraphNode_t))
        if cpDependencies is NULL:
            raise MemoryError('Failed to allocate length x size memory: ' + str(len(pDependencies)) + 'x' + str(sizeof(ccudart.cudaGraphNode_t)))
        else:
            for idx in range(len(pDependencies)):
                cpDependencies[idx] = <ccudart.cudaGraphNode_t>(<cudaGraphNode_t>pDependencies[idx])._ptr[0]

    if numDependencies > <size_t>len(pDependencies): raise RuntimeError("List is too small: " + str(len(pDependencies)) + " < " + str(numDependencies))
    err = ccudart.cudaGraphAddChildGraphNode(<ccudart.cudaGraphNode_t*>pGraphNode._ptr, <ccudart.cudaGraph_t>(<cudaGraph_t>graph)._ptr[0], <ccudart.cudaGraphNode_t*>(<cudaGraphNode_t>pDependencies[0])._ptr if len(pDependencies) == 1 else cpDependencies, numDependencies, <ccudart.cudaGraph_t>(<cudaGraph_t>childGraph)._ptr[0])
    if cpDependencies is not NULL:
        free(cpDependencies)
    return (cudaError_t(err), pGraphNode)

@cython.embedsignature(True)
def cudaGraphChildGraphNodeGetGraph(node):
    if not isinstance(node, (cudaGraphNode_t, cuda.CUgraphNode)):
        raise TypeError("Argument 'node' is not instance of type (expected <class 'cudapython.cudart.cudaGraphNode_t, cuda.CUgraphNode'>, found " + str(type(node)))
    cdef cudaGraph_t pGraph = cudaGraph_t()
    err = ccudart.cudaGraphChildGraphNodeGetGraph(<ccudart.cudaGraphNode_t>(<cudaGraphNode_t>node)._ptr[0], <ccudart.cudaGraph_t*>pGraph._ptr)
    return (cudaError_t(err), pGraph)

@cython.embedsignature(True)
def cudaGraphAddEmptyNode(graph, pDependencies : List[cudaGraphNode_t], size_t numDependencies):
    pDependencies = [] if pDependencies is None else pDependencies
    if not all(isinstance(_x, (cudaGraphNode_t, cuda.CUgraphNode)) for _x in pDependencies):
        raise TypeError("Argument 'pDependencies' is not instance of type (expected List[cudapython.ccudart.cudaGraphNode_t, cuda.CUgraphNode]")
    if not isinstance(graph, (cudaGraph_t, cuda.CUgraph)):
        raise TypeError("Argument 'graph' is not instance of type (expected <class 'cudapython.cudart.cudaGraph_t, cuda.CUgraph'>, found " + str(type(graph)))
    cdef cudaGraphNode_t pGraphNode = cudaGraphNode_t()
    cdef ccudart.cudaGraphNode_t* cpDependencies = NULL
    if len(pDependencies) > 0:
        cpDependencies = <ccudart.cudaGraphNode_t*> calloc(len(pDependencies), sizeof(ccudart.cudaGraphNode_t))
        if cpDependencies is NULL:
            raise MemoryError('Failed to allocate length x size memory: ' + str(len(pDependencies)) + 'x' + str(sizeof(ccudart.cudaGraphNode_t)))
        else:
            for idx in range(len(pDependencies)):
                cpDependencies[idx] = <ccudart.cudaGraphNode_t>(<cudaGraphNode_t>pDependencies[idx])._ptr[0]

    if numDependencies > <size_t>len(pDependencies): raise RuntimeError("List is too small: " + str(len(pDependencies)) + " < " + str(numDependencies))
    err = ccudart.cudaGraphAddEmptyNode(<ccudart.cudaGraphNode_t*>pGraphNode._ptr, <ccudart.cudaGraph_t>(<cudaGraph_t>graph)._ptr[0], <ccudart.cudaGraphNode_t*>(<cudaGraphNode_t>pDependencies[0])._ptr if len(pDependencies) == 1 else cpDependencies, numDependencies)
    if cpDependencies is not NULL:
        free(cpDependencies)
    return (cudaError_t(err), pGraphNode)

@cython.embedsignature(True)
def cudaGraphAddEventRecordNode(graph, pDependencies : List[cudaGraphNode_t], size_t numDependencies, event):
    if not isinstance(event, (cudaEvent_t, cuda.CUevent)):
        raise TypeError("Argument 'event' is not instance of type (expected <class 'cudapython.cudart.cudaEvent_t, cuda.CUevent'>, found " + str(type(event)))
    pDependencies = [] if pDependencies is None else pDependencies
    if not all(isinstance(_x, (cudaGraphNode_t, cuda.CUgraphNode)) for _x in pDependencies):
        raise TypeError("Argument 'pDependencies' is not instance of type (expected List[cudapython.ccudart.cudaGraphNode_t, cuda.CUgraphNode]")
    if not isinstance(graph, (cudaGraph_t, cuda.CUgraph)):
        raise TypeError("Argument 'graph' is not instance of type (expected <class 'cudapython.cudart.cudaGraph_t, cuda.CUgraph'>, found " + str(type(graph)))
    cdef cudaGraphNode_t pGraphNode = cudaGraphNode_t()
    cdef ccudart.cudaGraphNode_t* cpDependencies = NULL
    if len(pDependencies) > 0:
        cpDependencies = <ccudart.cudaGraphNode_t*> calloc(len(pDependencies), sizeof(ccudart.cudaGraphNode_t))
        if cpDependencies is NULL:
            raise MemoryError('Failed to allocate length x size memory: ' + str(len(pDependencies)) + 'x' + str(sizeof(ccudart.cudaGraphNode_t)))
        else:
            for idx in range(len(pDependencies)):
                cpDependencies[idx] = <ccudart.cudaGraphNode_t>(<cudaGraphNode_t>pDependencies[idx])._ptr[0]

    if numDependencies > <size_t>len(pDependencies): raise RuntimeError("List is too small: " + str(len(pDependencies)) + " < " + str(numDependencies))
    err = ccudart.cudaGraphAddEventRecordNode(<ccudart.cudaGraphNode_t*>pGraphNode._ptr, <ccudart.cudaGraph_t>(<cudaGraph_t>graph)._ptr[0], <ccudart.cudaGraphNode_t*>(<cudaGraphNode_t>pDependencies[0])._ptr if len(pDependencies) == 1 else cpDependencies, numDependencies, <ccudart.cudaEvent_t>(<cudaEvent_t>event)._ptr[0])
    if cpDependencies is not NULL:
        free(cpDependencies)
    return (cudaError_t(err), pGraphNode)

@cython.embedsignature(True)
def cudaGraphEventRecordNodeGetEvent(node):
    if not isinstance(node, (cudaGraphNode_t, cuda.CUgraphNode)):
        raise TypeError("Argument 'node' is not instance of type (expected <class 'cudapython.cudart.cudaGraphNode_t, cuda.CUgraphNode'>, found " + str(type(node)))
    cdef cudaEvent_t event_out = cudaEvent_t()
    err = ccudart.cudaGraphEventRecordNodeGetEvent(<ccudart.cudaGraphNode_t>(<cudaGraphNode_t>node)._ptr[0], <ccudart.cudaEvent_t*>event_out._ptr)
    return (cudaError_t(err), event_out)

@cython.embedsignature(True)
def cudaGraphEventRecordNodeSetEvent(node, event):
    if not isinstance(event, (cudaEvent_t, cuda.CUevent)):
        raise TypeError("Argument 'event' is not instance of type (expected <class 'cudapython.cudart.cudaEvent_t, cuda.CUevent'>, found " + str(type(event)))
    if not isinstance(node, (cudaGraphNode_t, cuda.CUgraphNode)):
        raise TypeError("Argument 'node' is not instance of type (expected <class 'cudapython.cudart.cudaGraphNode_t, cuda.CUgraphNode'>, found " + str(type(node)))
    err = ccudart.cudaGraphEventRecordNodeSetEvent(<ccudart.cudaGraphNode_t>(<cudaGraphNode_t>node)._ptr[0], <ccudart.cudaEvent_t>(<cudaEvent_t>event)._ptr[0])
    return (cudaError_t(err),)

@cython.embedsignature(True)
def cudaGraphAddEventWaitNode(graph, pDependencies : List[cudaGraphNode_t], size_t numDependencies, event):
    if not isinstance(event, (cudaEvent_t, cuda.CUevent)):
        raise TypeError("Argument 'event' is not instance of type (expected <class 'cudapython.cudart.cudaEvent_t, cuda.CUevent'>, found " + str(type(event)))
    pDependencies = [] if pDependencies is None else pDependencies
    if not all(isinstance(_x, (cudaGraphNode_t, cuda.CUgraphNode)) for _x in pDependencies):
        raise TypeError("Argument 'pDependencies' is not instance of type (expected List[cudapython.ccudart.cudaGraphNode_t, cuda.CUgraphNode]")
    if not isinstance(graph, (cudaGraph_t, cuda.CUgraph)):
        raise TypeError("Argument 'graph' is not instance of type (expected <class 'cudapython.cudart.cudaGraph_t, cuda.CUgraph'>, found " + str(type(graph)))
    cdef cudaGraphNode_t pGraphNode = cudaGraphNode_t()
    cdef ccudart.cudaGraphNode_t* cpDependencies = NULL
    if len(pDependencies) > 0:
        cpDependencies = <ccudart.cudaGraphNode_t*> calloc(len(pDependencies), sizeof(ccudart.cudaGraphNode_t))
        if cpDependencies is NULL:
            raise MemoryError('Failed to allocate length x size memory: ' + str(len(pDependencies)) + 'x' + str(sizeof(ccudart.cudaGraphNode_t)))
        else:
            for idx in range(len(pDependencies)):
                cpDependencies[idx] = <ccudart.cudaGraphNode_t>(<cudaGraphNode_t>pDependencies[idx])._ptr[0]

    if numDependencies > <size_t>len(pDependencies): raise RuntimeError("List is too small: " + str(len(pDependencies)) + " < " + str(numDependencies))
    err = ccudart.cudaGraphAddEventWaitNode(<ccudart.cudaGraphNode_t*>pGraphNode._ptr, <ccudart.cudaGraph_t>(<cudaGraph_t>graph)._ptr[0], <ccudart.cudaGraphNode_t*>(<cudaGraphNode_t>pDependencies[0])._ptr if len(pDependencies) == 1 else cpDependencies, numDependencies, <ccudart.cudaEvent_t>(<cudaEvent_t>event)._ptr[0])
    if cpDependencies is not NULL:
        free(cpDependencies)
    return (cudaError_t(err), pGraphNode)

@cython.embedsignature(True)
def cudaGraphEventWaitNodeGetEvent(node):
    if not isinstance(node, (cudaGraphNode_t, cuda.CUgraphNode)):
        raise TypeError("Argument 'node' is not instance of type (expected <class 'cudapython.cudart.cudaGraphNode_t, cuda.CUgraphNode'>, found " + str(type(node)))
    cdef cudaEvent_t event_out = cudaEvent_t()
    err = ccudart.cudaGraphEventWaitNodeGetEvent(<ccudart.cudaGraphNode_t>(<cudaGraphNode_t>node)._ptr[0], <ccudart.cudaEvent_t*>event_out._ptr)
    return (cudaError_t(err), event_out)

@cython.embedsignature(True)
def cudaGraphEventWaitNodeSetEvent(node, event):
    if not isinstance(event, (cudaEvent_t, cuda.CUevent)):
        raise TypeError("Argument 'event' is not instance of type (expected <class 'cudapython.cudart.cudaEvent_t, cuda.CUevent'>, found " + str(type(event)))
    if not isinstance(node, (cudaGraphNode_t, cuda.CUgraphNode)):
        raise TypeError("Argument 'node' is not instance of type (expected <class 'cudapython.cudart.cudaGraphNode_t, cuda.CUgraphNode'>, found " + str(type(node)))
    err = ccudart.cudaGraphEventWaitNodeSetEvent(<ccudart.cudaGraphNode_t>(<cudaGraphNode_t>node)._ptr[0], <ccudart.cudaEvent_t>(<cudaEvent_t>event)._ptr[0])
    return (cudaError_t(err),)

@cython.embedsignature(True)
def cudaGraphAddExternalSemaphoresSignalNode(graph, pDependencies : List[cudaGraphNode_t], size_t numDependencies, nodeParams : cudaExternalSemaphoreSignalNodeParams):
    pDependencies = [] if pDependencies is None else pDependencies
    if not all(isinstance(_x, (cudaGraphNode_t, cuda.CUgraphNode)) for _x in pDependencies):
        raise TypeError("Argument 'pDependencies' is not instance of type (expected List[cudapython.ccudart.cudaGraphNode_t, cuda.CUgraphNode]")
    if not isinstance(graph, (cudaGraph_t, cuda.CUgraph)):
        raise TypeError("Argument 'graph' is not instance of type (expected <class 'cudapython.cudart.cudaGraph_t, cuda.CUgraph'>, found " + str(type(graph)))
    cdef cudaGraphNode_t pGraphNode = cudaGraphNode_t()
    cdef ccudart.cudaGraphNode_t* cpDependencies = NULL
    if len(pDependencies) > 0:
        cpDependencies = <ccudart.cudaGraphNode_t*> calloc(len(pDependencies), sizeof(ccudart.cudaGraphNode_t))
        if cpDependencies is NULL:
            raise MemoryError('Failed to allocate length x size memory: ' + str(len(pDependencies)) + 'x' + str(sizeof(ccudart.cudaGraphNode_t)))
        else:
            for idx in range(len(pDependencies)):
                cpDependencies[idx] = <ccudart.cudaGraphNode_t>(<cudaGraphNode_t>pDependencies[idx])._ptr[0]

    if numDependencies > <size_t>len(pDependencies): raise RuntimeError("List is too small: " + str(len(pDependencies)) + " < " + str(numDependencies))
    cdef ccudart.cudaExternalSemaphoreSignalNodeParams* cnodeParams_ptr = nodeParams._ptr if nodeParams != None else NULL
    err = ccudart.cudaGraphAddExternalSemaphoresSignalNode(<ccudart.cudaGraphNode_t*>pGraphNode._ptr, <ccudart.cudaGraph_t>(<cudaGraph_t>graph)._ptr[0], <ccudart.cudaGraphNode_t*>(<cudaGraphNode_t>pDependencies[0])._ptr if len(pDependencies) == 1 else cpDependencies, numDependencies, cnodeParams_ptr)
    if cpDependencies is not NULL:
        free(cpDependencies)
    return (cudaError_t(err), pGraphNode)

@cython.embedsignature(True)
def cudaGraphExternalSemaphoresSignalNodeGetParams(hNode):
    if not isinstance(hNode, (cudaGraphNode_t, cuda.CUgraphNode)):
        raise TypeError("Argument 'hNode' is not instance of type (expected <class 'cudapython.cudart.cudaGraphNode_t, cuda.CUgraphNode'>, found " + str(type(hNode)))
    cdef cudaExternalSemaphoreSignalNodeParams params_out = cudaExternalSemaphoreSignalNodeParams()
    err = ccudart.cudaGraphExternalSemaphoresSignalNodeGetParams(<ccudart.cudaGraphNode_t>(<cudaGraphNode_t>hNode)._ptr[0], params_out._ptr)
    return (cudaError_t(err), params_out)

@cython.embedsignature(True)
def cudaGraphExternalSemaphoresSignalNodeSetParams(hNode, nodeParams : cudaExternalSemaphoreSignalNodeParams):
    if not isinstance(hNode, (cudaGraphNode_t, cuda.CUgraphNode)):
        raise TypeError("Argument 'hNode' is not instance of type (expected <class 'cudapython.cudart.cudaGraphNode_t, cuda.CUgraphNode'>, found " + str(type(hNode)))
    cdef ccudart.cudaExternalSemaphoreSignalNodeParams* cnodeParams_ptr = nodeParams._ptr if nodeParams != None else NULL
    err = ccudart.cudaGraphExternalSemaphoresSignalNodeSetParams(<ccudart.cudaGraphNode_t>(<cudaGraphNode_t>hNode)._ptr[0], cnodeParams_ptr)
    return (cudaError_t(err),)

@cython.embedsignature(True)
def cudaGraphAddExternalSemaphoresWaitNode(graph, pDependencies : List[cudaGraphNode_t], size_t numDependencies, nodeParams : cudaExternalSemaphoreWaitNodeParams):
    pDependencies = [] if pDependencies is None else pDependencies
    if not all(isinstance(_x, (cudaGraphNode_t, cuda.CUgraphNode)) for _x in pDependencies):
        raise TypeError("Argument 'pDependencies' is not instance of type (expected List[cudapython.ccudart.cudaGraphNode_t, cuda.CUgraphNode]")
    if not isinstance(graph, (cudaGraph_t, cuda.CUgraph)):
        raise TypeError("Argument 'graph' is not instance of type (expected <class 'cudapython.cudart.cudaGraph_t, cuda.CUgraph'>, found " + str(type(graph)))
    cdef cudaGraphNode_t pGraphNode = cudaGraphNode_t()
    cdef ccudart.cudaGraphNode_t* cpDependencies = NULL
    if len(pDependencies) > 0:
        cpDependencies = <ccudart.cudaGraphNode_t*> calloc(len(pDependencies), sizeof(ccudart.cudaGraphNode_t))
        if cpDependencies is NULL:
            raise MemoryError('Failed to allocate length x size memory: ' + str(len(pDependencies)) + 'x' + str(sizeof(ccudart.cudaGraphNode_t)))
        else:
            for idx in range(len(pDependencies)):
                cpDependencies[idx] = <ccudart.cudaGraphNode_t>(<cudaGraphNode_t>pDependencies[idx])._ptr[0]

    if numDependencies > <size_t>len(pDependencies): raise RuntimeError("List is too small: " + str(len(pDependencies)) + " < " + str(numDependencies))
    cdef ccudart.cudaExternalSemaphoreWaitNodeParams* cnodeParams_ptr = nodeParams._ptr if nodeParams != None else NULL
    err = ccudart.cudaGraphAddExternalSemaphoresWaitNode(<ccudart.cudaGraphNode_t*>pGraphNode._ptr, <ccudart.cudaGraph_t>(<cudaGraph_t>graph)._ptr[0], <ccudart.cudaGraphNode_t*>(<cudaGraphNode_t>pDependencies[0])._ptr if len(pDependencies) == 1 else cpDependencies, numDependencies, cnodeParams_ptr)
    if cpDependencies is not NULL:
        free(cpDependencies)
    return (cudaError_t(err), pGraphNode)

@cython.embedsignature(True)
def cudaGraphExternalSemaphoresWaitNodeGetParams(hNode):
    if not isinstance(hNode, (cudaGraphNode_t, cuda.CUgraphNode)):
        raise TypeError("Argument 'hNode' is not instance of type (expected <class 'cudapython.cudart.cudaGraphNode_t, cuda.CUgraphNode'>, found " + str(type(hNode)))
    cdef cudaExternalSemaphoreWaitNodeParams params_out = cudaExternalSemaphoreWaitNodeParams()
    err = ccudart.cudaGraphExternalSemaphoresWaitNodeGetParams(<ccudart.cudaGraphNode_t>(<cudaGraphNode_t>hNode)._ptr[0], params_out._ptr)
    return (cudaError_t(err), params_out)

@cython.embedsignature(True)
def cudaGraphExternalSemaphoresWaitNodeSetParams(hNode, nodeParams : cudaExternalSemaphoreWaitNodeParams):
    if not isinstance(hNode, (cudaGraphNode_t, cuda.CUgraphNode)):
        raise TypeError("Argument 'hNode' is not instance of type (expected <class 'cudapython.cudart.cudaGraphNode_t, cuda.CUgraphNode'>, found " + str(type(hNode)))
    cdef ccudart.cudaExternalSemaphoreWaitNodeParams* cnodeParams_ptr = nodeParams._ptr if nodeParams != None else NULL
    err = ccudart.cudaGraphExternalSemaphoresWaitNodeSetParams(<ccudart.cudaGraphNode_t>(<cudaGraphNode_t>hNode)._ptr[0], cnodeParams_ptr)
    return (cudaError_t(err),)

@cython.embedsignature(True)
def cudaGraphAddMemAllocNode(graph, pDependencies : List[cudaGraphNode_t], size_t numDependencies, nodeParams : cudaMemAllocNodeParams):
    pDependencies = [] if pDependencies is None else pDependencies
    if not all(isinstance(_x, (cudaGraphNode_t, cuda.CUgraphNode)) for _x in pDependencies):
        raise TypeError("Argument 'pDependencies' is not instance of type (expected List[cudapython.ccudart.cudaGraphNode_t, cuda.CUgraphNode]")
    if not isinstance(graph, (cudaGraph_t, cuda.CUgraph)):
        raise TypeError("Argument 'graph' is not instance of type (expected <class 'cudapython.cudart.cudaGraph_t, cuda.CUgraph'>, found " + str(type(graph)))
    cdef cudaGraphNode_t pGraphNode = cudaGraphNode_t()
    cdef ccudart.cudaGraphNode_t* cpDependencies = NULL
    if len(pDependencies) > 0:
        cpDependencies = <ccudart.cudaGraphNode_t*> calloc(len(pDependencies), sizeof(ccudart.cudaGraphNode_t))
        if cpDependencies is NULL:
            raise MemoryError('Failed to allocate length x size memory: ' + str(len(pDependencies)) + 'x' + str(sizeof(ccudart.cudaGraphNode_t)))
        else:
            for idx in range(len(pDependencies)):
                cpDependencies[idx] = <ccudart.cudaGraphNode_t>(<cudaGraphNode_t>pDependencies[idx])._ptr[0]

    if numDependencies > <size_t>len(pDependencies): raise RuntimeError("List is too small: " + str(len(pDependencies)) + " < " + str(numDependencies))
    cdef ccudart.cudaMemAllocNodeParams* cnodeParams_ptr = nodeParams._ptr if nodeParams != None else NULL
    err = ccudart.cudaGraphAddMemAllocNode(<ccudart.cudaGraphNode_t*>pGraphNode._ptr, <ccudart.cudaGraph_t>(<cudaGraph_t>graph)._ptr[0], <ccudart.cudaGraphNode_t*>(<cudaGraphNode_t>pDependencies[0])._ptr if len(pDependencies) == 1 else cpDependencies, numDependencies, cnodeParams_ptr)
    if cpDependencies is not NULL:
        free(cpDependencies)
    return (cudaError_t(err), pGraphNode)

@cython.embedsignature(True)
def cudaGraphMemAllocNodeGetParams(node):
    if not isinstance(node, (cudaGraphNode_t, cuda.CUgraphNode)):
        raise TypeError("Argument 'node' is not instance of type (expected <class 'cudapython.cudart.cudaGraphNode_t, cuda.CUgraphNode'>, found " + str(type(node)))
    cdef cudaMemAllocNodeParams params_out = cudaMemAllocNodeParams()
    err = ccudart.cudaGraphMemAllocNodeGetParams(<ccudart.cudaGraphNode_t>(<cudaGraphNode_t>node)._ptr[0], params_out._ptr)
    return (cudaError_t(err), params_out)

@cython.embedsignature(True)
def cudaGraphAddMemFreeNode(graph, pDependencies : List[cudaGraphNode_t], size_t numDependencies, dptr):
    pDependencies = [] if pDependencies is None else pDependencies
    if not all(isinstance(_x, (cudaGraphNode_t, cuda.CUgraphNode)) for _x in pDependencies):
        raise TypeError("Argument 'pDependencies' is not instance of type (expected List[cudapython.ccudart.cudaGraphNode_t, cuda.CUgraphNode]")
    if not isinstance(graph, (cudaGraph_t, cuda.CUgraph)):
        raise TypeError("Argument 'graph' is not instance of type (expected <class 'cudapython.cudart.cudaGraph_t, cuda.CUgraph'>, found " + str(type(graph)))
    cdef cudaGraphNode_t pGraphNode = cudaGraphNode_t()
    cdef ccudart.cudaGraphNode_t* cpDependencies = NULL
    if len(pDependencies) > 0:
        cpDependencies = <ccudart.cudaGraphNode_t*> calloc(len(pDependencies), sizeof(ccudart.cudaGraphNode_t))
        if cpDependencies is NULL:
            raise MemoryError('Failed to allocate length x size memory: ' + str(len(pDependencies)) + 'x' + str(sizeof(ccudart.cudaGraphNode_t)))
        else:
            for idx in range(len(pDependencies)):
                cpDependencies[idx] = <ccudart.cudaGraphNode_t>(<cudaGraphNode_t>pDependencies[idx])._ptr[0]

    if numDependencies > <size_t>len(pDependencies): raise RuntimeError("List is too small: " + str(len(pDependencies)) + " < " + str(numDependencies))
    cdptr = utils.HelperInputVoidPtr(dptr)
    cdef void* cdptr_ptr = <void*><void_ptr>cdptr.cptr
    err = ccudart.cudaGraphAddMemFreeNode(<ccudart.cudaGraphNode_t*>pGraphNode._ptr, <ccudart.cudaGraph_t>(<cudaGraph_t>graph)._ptr[0], <ccudart.cudaGraphNode_t*>(<cudaGraphNode_t>pDependencies[0])._ptr if len(pDependencies) == 1 else cpDependencies, numDependencies, cdptr_ptr)
    if cpDependencies is not NULL:
        free(cpDependencies)
    return (cudaError_t(err), pGraphNode)

@cython.embedsignature(True)
def cudaGraphMemFreeNodeGetParams(node):
    if not isinstance(node, (cudaGraphNode_t, cuda.CUgraphNode)):
        raise TypeError("Argument 'node' is not instance of type (expected <class 'cudapython.cudart.cudaGraphNode_t, cuda.CUgraphNode'>, found " + str(type(node)))
    cdef void_ptr dptr_out = 0
    cdef void* cdptr_out_ptr = <void*>dptr_out
    err = ccudart.cudaGraphMemFreeNodeGetParams(<ccudart.cudaGraphNode_t>(<cudaGraphNode_t>node)._ptr[0], cdptr_out_ptr)
    return (cudaError_t(err), dptr_out)

@cython.embedsignature(True)
def cudaDeviceGraphMemTrim(int device):
    err = ccudart.cudaDeviceGraphMemTrim(device)
    return (cudaError_t(err),)

@cython.embedsignature(True)
def cudaDeviceGetGraphMemAttribute(int device, attr not None : cudaGraphMemAttributeType):
    cdef ccudart.cudaGraphMemAttributeType cattr = attr.value
    cdef utils.HelperCUgraphMem_attribute cvalue = utils.HelperCUgraphMem_attribute(attr, 0, is_getter=True)
    cdef void* cvalue_ptr = <void*><void_ptr>cvalue.cptr
    err = ccudart.cudaDeviceGetGraphMemAttribute(device, cattr, cvalue_ptr)
    return (cudaError_t(err), cvalue.pyObj())

@cython.embedsignature(True)
def cudaDeviceSetGraphMemAttribute(int device, attr not None : cudaGraphMemAttributeType, value):
    cdef ccudart.cudaGraphMemAttributeType cattr = attr.value
    cdef utils.HelperCUgraphMem_attribute cvalue = utils.HelperCUgraphMem_attribute(attr, value, is_getter=False)
    cdef void* cvalue_ptr = <void*><void_ptr>cvalue.cptr
    err = ccudart.cudaDeviceSetGraphMemAttribute(device, cattr, cvalue_ptr)
    return (cudaError_t(err),)

@cython.embedsignature(True)
def cudaGraphClone(originalGraph):
    if not isinstance(originalGraph, (cudaGraph_t, cuda.CUgraph)):
        raise TypeError("Argument 'originalGraph' is not instance of type (expected <class 'cudapython.cudart.cudaGraph_t, cuda.CUgraph'>, found " + str(type(originalGraph)))
    cdef cudaGraph_t pGraphClone = cudaGraph_t()
    err = ccudart.cudaGraphClone(<ccudart.cudaGraph_t*>pGraphClone._ptr, <ccudart.cudaGraph_t>(<cudaGraph_t>originalGraph)._ptr[0])
    return (cudaError_t(err), pGraphClone)

@cython.embedsignature(True)
def cudaGraphNodeFindInClone(originalNode, clonedGraph):
    if not isinstance(clonedGraph, (cudaGraph_t, cuda.CUgraph)):
        raise TypeError("Argument 'clonedGraph' is not instance of type (expected <class 'cudapython.cudart.cudaGraph_t, cuda.CUgraph'>, found " + str(type(clonedGraph)))
    if not isinstance(originalNode, (cudaGraphNode_t, cuda.CUgraphNode)):
        raise TypeError("Argument 'originalNode' is not instance of type (expected <class 'cudapython.cudart.cudaGraphNode_t, cuda.CUgraphNode'>, found " + str(type(originalNode)))
    cdef cudaGraphNode_t pNode = cudaGraphNode_t()
    err = ccudart.cudaGraphNodeFindInClone(<ccudart.cudaGraphNode_t*>pNode._ptr, <ccudart.cudaGraphNode_t>(<cudaGraphNode_t>originalNode)._ptr[0], <ccudart.cudaGraph_t>(<cudaGraph_t>clonedGraph)._ptr[0])
    return (cudaError_t(err), pNode)

@cython.embedsignature(True)
def cudaGraphNodeGetType(node):
    if not isinstance(node, (cudaGraphNode_t, cuda.CUgraphNode)):
        raise TypeError("Argument 'node' is not instance of type (expected <class 'cudapython.cudart.cudaGraphNode_t, cuda.CUgraphNode'>, found " + str(type(node)))
    cdef ccudart.cudaGraphNodeType pType
    err = ccudart.cudaGraphNodeGetType(<ccudart.cudaGraphNode_t>(<cudaGraphNode_t>node)._ptr[0], &pType)
    return (cudaError_t(err), cudaGraphNodeType(pType))

@cython.embedsignature(True)
def cudaGraphGetNodes(graph, size_t numNodes = 0):
    cdef size_t _graph_length = numNodes
    if not isinstance(graph, (cudaGraph_t, cuda.CUgraph)):
        raise TypeError("Argument 'graph' is not instance of type (expected <class 'cudapython.cudart.cudaGraph_t, cuda.CUgraph'>, found " + str(type(graph)))
    cdef ccudart.cudaGraphNode_t* cnodes = NULL
    pynodes = []
    if _graph_length != 0:
        cnodes = <ccudart.cudaGraphNode_t*>calloc(_graph_length, sizeof(ccudart.cudaGraphNode_t))
        if cnodes is NULL:
            raise MemoryError('Failed to allocate length x size memory: ' + str(_graph_length) + 'x' + str(sizeof(ccudart.cudaGraphNode_t)))
    err = ccudart.cudaGraphGetNodes(<ccudart.cudaGraph_t>(<cudaGraph_t>graph)._ptr[0], cnodes, &numNodes)
    if cudaError_t(err) == cudaError_t(0):
        pynodes = [cudaGraphNode_t(init_value=<void_ptr>cnodes[idx]) for idx in range(_graph_length)]
    if cnodes is not NULL:
        free(cnodes)
    return (cudaError_t(err), pynodes, numNodes)

@cython.embedsignature(True)
def cudaGraphGetRootNodes(graph, size_t pNumRootNodes = 0):
    cdef size_t _graph_length = pNumRootNodes
    if not isinstance(graph, (cudaGraph_t, cuda.CUgraph)):
        raise TypeError("Argument 'graph' is not instance of type (expected <class 'cudapython.cudart.cudaGraph_t, cuda.CUgraph'>, found " + str(type(graph)))
    cdef ccudart.cudaGraphNode_t* cpRootNodes = NULL
    pypRootNodes = []
    if _graph_length != 0:
        cpRootNodes = <ccudart.cudaGraphNode_t*>calloc(_graph_length, sizeof(ccudart.cudaGraphNode_t))
        if cpRootNodes is NULL:
            raise MemoryError('Failed to allocate length x size memory: ' + str(_graph_length) + 'x' + str(sizeof(ccudart.cudaGraphNode_t)))
    err = ccudart.cudaGraphGetRootNodes(<ccudart.cudaGraph_t>(<cudaGraph_t>graph)._ptr[0], cpRootNodes, &pNumRootNodes)
    if cudaError_t(err) == cudaError_t(0):
        pypRootNodes = [cudaGraphNode_t(init_value=<void_ptr>cpRootNodes[idx]) for idx in range(_graph_length)]
    if cpRootNodes is not NULL:
        free(cpRootNodes)
    return (cudaError_t(err), pypRootNodes, pNumRootNodes)

@cython.embedsignature(True)
def cudaGraphGetEdges(graph, size_t numEdges = 0):
    cdef size_t _graph_length = numEdges
    if not isinstance(graph, (cudaGraph_t, cuda.CUgraph)):
        raise TypeError("Argument 'graph' is not instance of type (expected <class 'cudapython.cudart.cudaGraph_t, cuda.CUgraph'>, found " + str(type(graph)))
    cdef ccudart.cudaGraphNode_t* cfrom_ = NULL
    pyfrom_ = []
    if _graph_length != 0:
        cfrom_ = <ccudart.cudaGraphNode_t*>calloc(_graph_length, sizeof(ccudart.cudaGraphNode_t))
        if cfrom_ is NULL:
            raise MemoryError('Failed to allocate length x size memory: ' + str(_graph_length) + 'x' + str(sizeof(ccudart.cudaGraphNode_t)))
    cdef ccudart.cudaGraphNode_t* cto = NULL
    pyto = []
    if _graph_length != 0:
        cto = <ccudart.cudaGraphNode_t*>calloc(_graph_length, sizeof(ccudart.cudaGraphNode_t))
        if cto is NULL:
            raise MemoryError('Failed to allocate length x size memory: ' + str(_graph_length) + 'x' + str(sizeof(ccudart.cudaGraphNode_t)))
    err = ccudart.cudaGraphGetEdges(<ccudart.cudaGraph_t>(<cudaGraph_t>graph)._ptr[0], cfrom_, cto, &numEdges)
    if cudaError_t(err) == cudaError_t(0):
        pyfrom_ = [cudaGraphNode_t(init_value=<void_ptr>cfrom_[idx]) for idx in range(_graph_length)]
    if cfrom_ is not NULL:
        free(cfrom_)
    if cudaError_t(err) == cudaError_t(0):
        pyto = [cudaGraphNode_t(init_value=<void_ptr>cto[idx]) for idx in range(_graph_length)]
    if cto is not NULL:
        free(cto)
    return (cudaError_t(err), pyfrom_, pyto, numEdges)

@cython.embedsignature(True)
def cudaGraphNodeGetDependencies(node, size_t pNumDependencies = 0):
    cdef size_t _graph_length = pNumDependencies
    if not isinstance(node, (cudaGraphNode_t, cuda.CUgraphNode)):
        raise TypeError("Argument 'node' is not instance of type (expected <class 'cudapython.cudart.cudaGraphNode_t, cuda.CUgraphNode'>, found " + str(type(node)))
    cdef ccudart.cudaGraphNode_t* cpDependencies = NULL
    pypDependencies = []
    if _graph_length != 0:
        cpDependencies = <ccudart.cudaGraphNode_t*>calloc(_graph_length, sizeof(ccudart.cudaGraphNode_t))
        if cpDependencies is NULL:
            raise MemoryError('Failed to allocate length x size memory: ' + str(_graph_length) + 'x' + str(sizeof(ccudart.cudaGraphNode_t)))
    err = ccudart.cudaGraphNodeGetDependencies(<ccudart.cudaGraphNode_t>(<cudaGraphNode_t>node)._ptr[0], cpDependencies, &pNumDependencies)
    if cudaError_t(err) == cudaError_t(0):
        pypDependencies = [cudaGraphNode_t(init_value=<void_ptr>cpDependencies[idx]) for idx in range(_graph_length)]
    if cpDependencies is not NULL:
        free(cpDependencies)
    return (cudaError_t(err), pypDependencies, pNumDependencies)

@cython.embedsignature(True)
def cudaGraphNodeGetDependentNodes(node, size_t pNumDependentNodes = 0):
    cdef size_t _graph_length = pNumDependentNodes
    if not isinstance(node, (cudaGraphNode_t, cuda.CUgraphNode)):
        raise TypeError("Argument 'node' is not instance of type (expected <class 'cudapython.cudart.cudaGraphNode_t, cuda.CUgraphNode'>, found " + str(type(node)))
    cdef ccudart.cudaGraphNode_t* cpDependentNodes = NULL
    pypDependentNodes = []
    if _graph_length != 0:
        cpDependentNodes = <ccudart.cudaGraphNode_t*>calloc(_graph_length, sizeof(ccudart.cudaGraphNode_t))
        if cpDependentNodes is NULL:
            raise MemoryError('Failed to allocate length x size memory: ' + str(_graph_length) + 'x' + str(sizeof(ccudart.cudaGraphNode_t)))
    err = ccudart.cudaGraphNodeGetDependentNodes(<ccudart.cudaGraphNode_t>(<cudaGraphNode_t>node)._ptr[0], cpDependentNodes, &pNumDependentNodes)
    if cudaError_t(err) == cudaError_t(0):
        pypDependentNodes = [cudaGraphNode_t(init_value=<void_ptr>cpDependentNodes[idx]) for idx in range(_graph_length)]
    if cpDependentNodes is not NULL:
        free(cpDependentNodes)
    return (cudaError_t(err), pypDependentNodes, pNumDependentNodes)

@cython.embedsignature(True)
def cudaGraphAddDependencies(graph, from_ : List[cudaGraphNode_t], to : List[cudaGraphNode_t], size_t numDependencies):
    to = [] if to is None else to
    if not all(isinstance(_x, (cudaGraphNode_t, cuda.CUgraphNode)) for _x in to):
        raise TypeError("Argument 'to' is not instance of type (expected List[cudapython.ccudart.cudaGraphNode_t, cuda.CUgraphNode]")
    from_ = [] if from_ is None else from_
    if not all(isinstance(_x, (cudaGraphNode_t, cuda.CUgraphNode)) for _x in from_):
        raise TypeError("Argument 'from_' is not instance of type (expected List[cudapython.ccudart.cudaGraphNode_t, cuda.CUgraphNode]")
    if not isinstance(graph, (cudaGraph_t, cuda.CUgraph)):
        raise TypeError("Argument 'graph' is not instance of type (expected <class 'cudapython.cudart.cudaGraph_t, cuda.CUgraph'>, found " + str(type(graph)))
    cdef ccudart.cudaGraphNode_t* cfrom_ = NULL
    if len(from_) > 0:
        cfrom_ = <ccudart.cudaGraphNode_t*> calloc(len(from_), sizeof(ccudart.cudaGraphNode_t))
        if cfrom_ is NULL:
            raise MemoryError('Failed to allocate length x size memory: ' + str(len(from_)) + 'x' + str(sizeof(ccudart.cudaGraphNode_t)))
        else:
            for idx in range(len(from_)):
                cfrom_[idx] = <ccudart.cudaGraphNode_t>(<cudaGraphNode_t>from_[idx])._ptr[0]

    cdef ccudart.cudaGraphNode_t* cto = NULL
    if len(to) > 0:
        cto = <ccudart.cudaGraphNode_t*> calloc(len(to), sizeof(ccudart.cudaGraphNode_t))
        if cto is NULL:
            raise MemoryError('Failed to allocate length x size memory: ' + str(len(to)) + 'x' + str(sizeof(ccudart.cudaGraphNode_t)))
        else:
            for idx in range(len(to)):
                cto[idx] = <ccudart.cudaGraphNode_t>(<cudaGraphNode_t>to[idx])._ptr[0]

    if numDependencies > <size_t>len(from_): raise RuntimeError("List is too small: " + str(len(from_)) + " < " + str(numDependencies))
    if numDependencies > <size_t>len(to): raise RuntimeError("List is too small: " + str(len(to)) + " < " + str(numDependencies))
    err = ccudart.cudaGraphAddDependencies(<ccudart.cudaGraph_t>(<cudaGraph_t>graph)._ptr[0], <ccudart.cudaGraphNode_t*>(<cudaGraphNode_t>from_[0])._ptr if len(from_) == 1 else cfrom_, <ccudart.cudaGraphNode_t*>(<cudaGraphNode_t>to[0])._ptr if len(to) == 1 else cto, numDependencies)
    if cfrom_ is not NULL:
        free(cfrom_)
    if cto is not NULL:
        free(cto)
    return (cudaError_t(err),)

@cython.embedsignature(True)
def cudaGraphRemoveDependencies(graph, from_ : List[cudaGraphNode_t], to : List[cudaGraphNode_t], size_t numDependencies):
    to = [] if to is None else to
    if not all(isinstance(_x, (cudaGraphNode_t, cuda.CUgraphNode)) for _x in to):
        raise TypeError("Argument 'to' is not instance of type (expected List[cudapython.ccudart.cudaGraphNode_t, cuda.CUgraphNode]")
    from_ = [] if from_ is None else from_
    if not all(isinstance(_x, (cudaGraphNode_t, cuda.CUgraphNode)) for _x in from_):
        raise TypeError("Argument 'from_' is not instance of type (expected List[cudapython.ccudart.cudaGraphNode_t, cuda.CUgraphNode]")
    if not isinstance(graph, (cudaGraph_t, cuda.CUgraph)):
        raise TypeError("Argument 'graph' is not instance of type (expected <class 'cudapython.cudart.cudaGraph_t, cuda.CUgraph'>, found " + str(type(graph)))
    cdef ccudart.cudaGraphNode_t* cfrom_ = NULL
    if len(from_) > 0:
        cfrom_ = <ccudart.cudaGraphNode_t*> calloc(len(from_), sizeof(ccudart.cudaGraphNode_t))
        if cfrom_ is NULL:
            raise MemoryError('Failed to allocate length x size memory: ' + str(len(from_)) + 'x' + str(sizeof(ccudart.cudaGraphNode_t)))
        else:
            for idx in range(len(from_)):
                cfrom_[idx] = <ccudart.cudaGraphNode_t>(<cudaGraphNode_t>from_[idx])._ptr[0]

    cdef ccudart.cudaGraphNode_t* cto = NULL
    if len(to) > 0:
        cto = <ccudart.cudaGraphNode_t*> calloc(len(to), sizeof(ccudart.cudaGraphNode_t))
        if cto is NULL:
            raise MemoryError('Failed to allocate length x size memory: ' + str(len(to)) + 'x' + str(sizeof(ccudart.cudaGraphNode_t)))
        else:
            for idx in range(len(to)):
                cto[idx] = <ccudart.cudaGraphNode_t>(<cudaGraphNode_t>to[idx])._ptr[0]

    if numDependencies > <size_t>len(from_): raise RuntimeError("List is too small: " + str(len(from_)) + " < " + str(numDependencies))
    if numDependencies > <size_t>len(to): raise RuntimeError("List is too small: " + str(len(to)) + " < " + str(numDependencies))
    err = ccudart.cudaGraphRemoveDependencies(<ccudart.cudaGraph_t>(<cudaGraph_t>graph)._ptr[0], <ccudart.cudaGraphNode_t*>(<cudaGraphNode_t>from_[0])._ptr if len(from_) == 1 else cfrom_, <ccudart.cudaGraphNode_t*>(<cudaGraphNode_t>to[0])._ptr if len(to) == 1 else cto, numDependencies)
    if cfrom_ is not NULL:
        free(cfrom_)
    if cto is not NULL:
        free(cto)
    return (cudaError_t(err),)

@cython.embedsignature(True)
def cudaGraphDestroyNode(node):
    if not isinstance(node, (cudaGraphNode_t, cuda.CUgraphNode)):
        raise TypeError("Argument 'node' is not instance of type (expected <class 'cudapython.cudart.cudaGraphNode_t, cuda.CUgraphNode'>, found " + str(type(node)))
    err = ccudart.cudaGraphDestroyNode(<ccudart.cudaGraphNode_t>(<cudaGraphNode_t>node)._ptr[0])
    return (cudaError_t(err),)

@cython.embedsignature(True)
def cudaGraphInstantiate(graph, char* pLogBuffer, size_t bufferSize):
    if not isinstance(graph, (cudaGraph_t, cuda.CUgraph)):
        raise TypeError("Argument 'graph' is not instance of type (expected <class 'cudapython.cudart.cudaGraph_t, cuda.CUgraph'>, found " + str(type(graph)))
    cdef cudaGraphExec_t pGraphExec = cudaGraphExec_t()
    cdef cudaGraphNode_t pErrorNode = cudaGraphNode_t()
    err = ccudart.cudaGraphInstantiate(<ccudart.cudaGraphExec_t*>pGraphExec._ptr, <ccudart.cudaGraph_t>(<cudaGraph_t>graph)._ptr[0], <ccudart.cudaGraphNode_t*>pErrorNode._ptr, pLogBuffer, bufferSize)
    return (cudaError_t(err), pGraphExec, pErrorNode)

@cython.embedsignature(True)
def cudaGraphInstantiateWithFlags(graph, unsigned long long flags):
    if not isinstance(graph, (cudaGraph_t, cuda.CUgraph)):
        raise TypeError("Argument 'graph' is not instance of type (expected <class 'cudapython.cudart.cudaGraph_t, cuda.CUgraph'>, found " + str(type(graph)))
    cdef cudaGraphExec_t pGraphExec = cudaGraphExec_t()
    err = ccudart.cudaGraphInstantiateWithFlags(<ccudart.cudaGraphExec_t*>pGraphExec._ptr, <ccudart.cudaGraph_t>(<cudaGraph_t>graph)._ptr[0], flags)
    return (cudaError_t(err), pGraphExec)

@cython.embedsignature(True)
def cudaGraphExecKernelNodeSetParams(hGraphExec, node, pNodeParams : cudaKernelNodeParams):
    if not isinstance(node, (cudaGraphNode_t, cuda.CUgraphNode)):
        raise TypeError("Argument 'node' is not instance of type (expected <class 'cudapython.cudart.cudaGraphNode_t, cuda.CUgraphNode'>, found " + str(type(node)))
    if not isinstance(hGraphExec, (cudaGraphExec_t, cuda.CUgraphExec)):
        raise TypeError("Argument 'hGraphExec' is not instance of type (expected <class 'cudapython.cudart.cudaGraphExec_t, cuda.CUgraphExec'>, found " + str(type(hGraphExec)))
    cdef ccudart.cudaKernelNodeParams* cpNodeParams_ptr = pNodeParams._ptr if pNodeParams != None else NULL
    err = ccudart.cudaGraphExecKernelNodeSetParams(<ccudart.cudaGraphExec_t>(<cudaGraphExec_t>hGraphExec)._ptr[0], <ccudart.cudaGraphNode_t>(<cudaGraphNode_t>node)._ptr[0], cpNodeParams_ptr)
    return (cudaError_t(err),)

@cython.embedsignature(True)
def cudaGraphExecMemcpyNodeSetParams(hGraphExec, node, pNodeParams : cudaMemcpy3DParms):
    if not isinstance(node, (cudaGraphNode_t, cuda.CUgraphNode)):
        raise TypeError("Argument 'node' is not instance of type (expected <class 'cudapython.cudart.cudaGraphNode_t, cuda.CUgraphNode'>, found " + str(type(node)))
    if not isinstance(hGraphExec, (cudaGraphExec_t, cuda.CUgraphExec)):
        raise TypeError("Argument 'hGraphExec' is not instance of type (expected <class 'cudapython.cudart.cudaGraphExec_t, cuda.CUgraphExec'>, found " + str(type(hGraphExec)))
    cdef ccudart.cudaMemcpy3DParms* cpNodeParams_ptr = pNodeParams._ptr if pNodeParams != None else NULL
    err = ccudart.cudaGraphExecMemcpyNodeSetParams(<ccudart.cudaGraphExec_t>(<cudaGraphExec_t>hGraphExec)._ptr[0], <ccudart.cudaGraphNode_t>(<cudaGraphNode_t>node)._ptr[0], cpNodeParams_ptr)
    return (cudaError_t(err),)

@cython.embedsignature(True)
def cudaGraphExecMemcpyNodeSetParams1D(hGraphExec, node, dst, src, size_t count, kind not None : cudaMemcpyKind):
    if not isinstance(node, (cudaGraphNode_t, cuda.CUgraphNode)):
        raise TypeError("Argument 'node' is not instance of type (expected <class 'cudapython.cudart.cudaGraphNode_t, cuda.CUgraphNode'>, found " + str(type(node)))
    if not isinstance(hGraphExec, (cudaGraphExec_t, cuda.CUgraphExec)):
        raise TypeError("Argument 'hGraphExec' is not instance of type (expected <class 'cudapython.cudart.cudaGraphExec_t, cuda.CUgraphExec'>, found " + str(type(hGraphExec)))
    cdst = utils.HelperInputVoidPtr(dst)
    cdef void* cdst_ptr = <void*><void_ptr>cdst.cptr
    csrc = utils.HelperInputVoidPtr(src)
    cdef void* csrc_ptr = <void*><void_ptr>csrc.cptr
    cdef ccudart.cudaMemcpyKind ckind = kind.value
    err = ccudart.cudaGraphExecMemcpyNodeSetParams1D(<ccudart.cudaGraphExec_t>(<cudaGraphExec_t>hGraphExec)._ptr[0], <ccudart.cudaGraphNode_t>(<cudaGraphNode_t>node)._ptr[0], cdst_ptr, csrc_ptr, count, ckind)
    return (cudaError_t(err),)

@cython.embedsignature(True)
def cudaGraphExecMemsetNodeSetParams(hGraphExec, node, pNodeParams : cudaMemsetParams):
    if not isinstance(node, (cudaGraphNode_t, cuda.CUgraphNode)):
        raise TypeError("Argument 'node' is not instance of type (expected <class 'cudapython.cudart.cudaGraphNode_t, cuda.CUgraphNode'>, found " + str(type(node)))
    if not isinstance(hGraphExec, (cudaGraphExec_t, cuda.CUgraphExec)):
        raise TypeError("Argument 'hGraphExec' is not instance of type (expected <class 'cudapython.cudart.cudaGraphExec_t, cuda.CUgraphExec'>, found " + str(type(hGraphExec)))
    cdef ccudart.cudaMemsetParams* cpNodeParams_ptr = pNodeParams._ptr if pNodeParams != None else NULL
    err = ccudart.cudaGraphExecMemsetNodeSetParams(<ccudart.cudaGraphExec_t>(<cudaGraphExec_t>hGraphExec)._ptr[0], <ccudart.cudaGraphNode_t>(<cudaGraphNode_t>node)._ptr[0], cpNodeParams_ptr)
    return (cudaError_t(err),)

@cython.embedsignature(True)
def cudaGraphExecHostNodeSetParams(hGraphExec, node, pNodeParams : cudaHostNodeParams):
    if not isinstance(node, (cudaGraphNode_t, cuda.CUgraphNode)):
        raise TypeError("Argument 'node' is not instance of type (expected <class 'cudapython.cudart.cudaGraphNode_t, cuda.CUgraphNode'>, found " + str(type(node)))
    if not isinstance(hGraphExec, (cudaGraphExec_t, cuda.CUgraphExec)):
        raise TypeError("Argument 'hGraphExec' is not instance of type (expected <class 'cudapython.cudart.cudaGraphExec_t, cuda.CUgraphExec'>, found " + str(type(hGraphExec)))
    cdef ccudart.cudaHostNodeParams* cpNodeParams_ptr = pNodeParams._ptr if pNodeParams != None else NULL
    err = ccudart.cudaGraphExecHostNodeSetParams(<ccudart.cudaGraphExec_t>(<cudaGraphExec_t>hGraphExec)._ptr[0], <ccudart.cudaGraphNode_t>(<cudaGraphNode_t>node)._ptr[0], cpNodeParams_ptr)
    return (cudaError_t(err),)

@cython.embedsignature(True)
def cudaGraphExecChildGraphNodeSetParams(hGraphExec, node, childGraph):
    if not isinstance(childGraph, (cudaGraph_t, cuda.CUgraph)):
        raise TypeError("Argument 'childGraph' is not instance of type (expected <class 'cudapython.cudart.cudaGraph_t, cuda.CUgraph'>, found " + str(type(childGraph)))
    if not isinstance(node, (cudaGraphNode_t, cuda.CUgraphNode)):
        raise TypeError("Argument 'node' is not instance of type (expected <class 'cudapython.cudart.cudaGraphNode_t, cuda.CUgraphNode'>, found " + str(type(node)))
    if not isinstance(hGraphExec, (cudaGraphExec_t, cuda.CUgraphExec)):
        raise TypeError("Argument 'hGraphExec' is not instance of type (expected <class 'cudapython.cudart.cudaGraphExec_t, cuda.CUgraphExec'>, found " + str(type(hGraphExec)))
    err = ccudart.cudaGraphExecChildGraphNodeSetParams(<ccudart.cudaGraphExec_t>(<cudaGraphExec_t>hGraphExec)._ptr[0], <ccudart.cudaGraphNode_t>(<cudaGraphNode_t>node)._ptr[0], <ccudart.cudaGraph_t>(<cudaGraph_t>childGraph)._ptr[0])
    return (cudaError_t(err),)

@cython.embedsignature(True)
def cudaGraphExecEventRecordNodeSetEvent(hGraphExec, hNode, event):
    if not isinstance(event, (cudaEvent_t, cuda.CUevent)):
        raise TypeError("Argument 'event' is not instance of type (expected <class 'cudapython.cudart.cudaEvent_t, cuda.CUevent'>, found " + str(type(event)))
    if not isinstance(hNode, (cudaGraphNode_t, cuda.CUgraphNode)):
        raise TypeError("Argument 'hNode' is not instance of type (expected <class 'cudapython.cudart.cudaGraphNode_t, cuda.CUgraphNode'>, found " + str(type(hNode)))
    if not isinstance(hGraphExec, (cudaGraphExec_t, cuda.CUgraphExec)):
        raise TypeError("Argument 'hGraphExec' is not instance of type (expected <class 'cudapython.cudart.cudaGraphExec_t, cuda.CUgraphExec'>, found " + str(type(hGraphExec)))
    err = ccudart.cudaGraphExecEventRecordNodeSetEvent(<ccudart.cudaGraphExec_t>(<cudaGraphExec_t>hGraphExec)._ptr[0], <ccudart.cudaGraphNode_t>(<cudaGraphNode_t>hNode)._ptr[0], <ccudart.cudaEvent_t>(<cudaEvent_t>event)._ptr[0])
    return (cudaError_t(err),)

@cython.embedsignature(True)
def cudaGraphExecEventWaitNodeSetEvent(hGraphExec, hNode, event):
    if not isinstance(event, (cudaEvent_t, cuda.CUevent)):
        raise TypeError("Argument 'event' is not instance of type (expected <class 'cudapython.cudart.cudaEvent_t, cuda.CUevent'>, found " + str(type(event)))
    if not isinstance(hNode, (cudaGraphNode_t, cuda.CUgraphNode)):
        raise TypeError("Argument 'hNode' is not instance of type (expected <class 'cudapython.cudart.cudaGraphNode_t, cuda.CUgraphNode'>, found " + str(type(hNode)))
    if not isinstance(hGraphExec, (cudaGraphExec_t, cuda.CUgraphExec)):
        raise TypeError("Argument 'hGraphExec' is not instance of type (expected <class 'cudapython.cudart.cudaGraphExec_t, cuda.CUgraphExec'>, found " + str(type(hGraphExec)))
    err = ccudart.cudaGraphExecEventWaitNodeSetEvent(<ccudart.cudaGraphExec_t>(<cudaGraphExec_t>hGraphExec)._ptr[0], <ccudart.cudaGraphNode_t>(<cudaGraphNode_t>hNode)._ptr[0], <ccudart.cudaEvent_t>(<cudaEvent_t>event)._ptr[0])
    return (cudaError_t(err),)

@cython.embedsignature(True)
def cudaGraphExecExternalSemaphoresSignalNodeSetParams(hGraphExec, hNode, nodeParams : cudaExternalSemaphoreSignalNodeParams):
    if not isinstance(hNode, (cudaGraphNode_t, cuda.CUgraphNode)):
        raise TypeError("Argument 'hNode' is not instance of type (expected <class 'cudapython.cudart.cudaGraphNode_t, cuda.CUgraphNode'>, found " + str(type(hNode)))
    if not isinstance(hGraphExec, (cudaGraphExec_t, cuda.CUgraphExec)):
        raise TypeError("Argument 'hGraphExec' is not instance of type (expected <class 'cudapython.cudart.cudaGraphExec_t, cuda.CUgraphExec'>, found " + str(type(hGraphExec)))
    cdef ccudart.cudaExternalSemaphoreSignalNodeParams* cnodeParams_ptr = nodeParams._ptr if nodeParams != None else NULL
    err = ccudart.cudaGraphExecExternalSemaphoresSignalNodeSetParams(<ccudart.cudaGraphExec_t>(<cudaGraphExec_t>hGraphExec)._ptr[0], <ccudart.cudaGraphNode_t>(<cudaGraphNode_t>hNode)._ptr[0], cnodeParams_ptr)
    return (cudaError_t(err),)

@cython.embedsignature(True)
def cudaGraphExecExternalSemaphoresWaitNodeSetParams(hGraphExec, hNode, nodeParams : cudaExternalSemaphoreWaitNodeParams):
    if not isinstance(hNode, (cudaGraphNode_t, cuda.CUgraphNode)):
        raise TypeError("Argument 'hNode' is not instance of type (expected <class 'cudapython.cudart.cudaGraphNode_t, cuda.CUgraphNode'>, found " + str(type(hNode)))
    if not isinstance(hGraphExec, (cudaGraphExec_t, cuda.CUgraphExec)):
        raise TypeError("Argument 'hGraphExec' is not instance of type (expected <class 'cudapython.cudart.cudaGraphExec_t, cuda.CUgraphExec'>, found " + str(type(hGraphExec)))
    cdef ccudart.cudaExternalSemaphoreWaitNodeParams* cnodeParams_ptr = nodeParams._ptr if nodeParams != None else NULL
    err = ccudart.cudaGraphExecExternalSemaphoresWaitNodeSetParams(<ccudart.cudaGraphExec_t>(<cudaGraphExec_t>hGraphExec)._ptr[0], <ccudart.cudaGraphNode_t>(<cudaGraphNode_t>hNode)._ptr[0], cnodeParams_ptr)
    return (cudaError_t(err),)

@cython.embedsignature(True)
def cudaGraphExecUpdate(hGraphExec, hGraph):
    if not isinstance(hGraph, (cudaGraph_t, cuda.CUgraph)):
        raise TypeError("Argument 'hGraph' is not instance of type (expected <class 'cudapython.cudart.cudaGraph_t, cuda.CUgraph'>, found " + str(type(hGraph)))
    if not isinstance(hGraphExec, (cudaGraphExec_t, cuda.CUgraphExec)):
        raise TypeError("Argument 'hGraphExec' is not instance of type (expected <class 'cudapython.cudart.cudaGraphExec_t, cuda.CUgraphExec'>, found " + str(type(hGraphExec)))
    cdef cudaGraphNode_t hErrorNode_out = cudaGraphNode_t()
    cdef ccudart.cudaGraphExecUpdateResult updateResult_out
    err = ccudart.cudaGraphExecUpdate(<ccudart.cudaGraphExec_t>(<cudaGraphExec_t>hGraphExec)._ptr[0], <ccudart.cudaGraph_t>(<cudaGraph_t>hGraph)._ptr[0], <ccudart.cudaGraphNode_t*>hErrorNode_out._ptr, &updateResult_out)
    return (cudaError_t(err), hErrorNode_out, cudaGraphExecUpdateResult(updateResult_out))

@cython.embedsignature(True)
def cudaGraphUpload(graphExec, stream):
    if not isinstance(stream, (cudaStream_t, cuda.CUstream)):
        raise TypeError("Argument 'stream' is not instance of type (expected <class 'cudapython.cudart.cudaStream_t, cuda.CUstream'>, found " + str(type(stream)))
    if not isinstance(graphExec, (cudaGraphExec_t, cuda.CUgraphExec)):
        raise TypeError("Argument 'graphExec' is not instance of type (expected <class 'cudapython.cudart.cudaGraphExec_t, cuda.CUgraphExec'>, found " + str(type(graphExec)))
    err = ccudart.cudaGraphUpload(<ccudart.cudaGraphExec_t>(<cudaGraphExec_t>graphExec)._ptr[0], <ccudart.cudaStream_t>(<cudaStream_t>stream)._ptr[0])
    return (cudaError_t(err),)

@cython.embedsignature(True)
def cudaGraphLaunch(graphExec, stream):
    if not isinstance(stream, (cudaStream_t, cuda.CUstream)):
        raise TypeError("Argument 'stream' is not instance of type (expected <class 'cudapython.cudart.cudaStream_t, cuda.CUstream'>, found " + str(type(stream)))
    if not isinstance(graphExec, (cudaGraphExec_t, cuda.CUgraphExec)):
        raise TypeError("Argument 'graphExec' is not instance of type (expected <class 'cudapython.cudart.cudaGraphExec_t, cuda.CUgraphExec'>, found " + str(type(graphExec)))
    err = ccudart.cudaGraphLaunch(<ccudart.cudaGraphExec_t>(<cudaGraphExec_t>graphExec)._ptr[0], <ccudart.cudaStream_t>(<cudaStream_t>stream)._ptr[0])
    return (cudaError_t(err),)

@cython.embedsignature(True)
def cudaGraphExecDestroy(graphExec):
    if not isinstance(graphExec, (cudaGraphExec_t, cuda.CUgraphExec)):
        raise TypeError("Argument 'graphExec' is not instance of type (expected <class 'cudapython.cudart.cudaGraphExec_t, cuda.CUgraphExec'>, found " + str(type(graphExec)))
    err = ccudart.cudaGraphExecDestroy(<ccudart.cudaGraphExec_t>(<cudaGraphExec_t>graphExec)._ptr[0])
    return (cudaError_t(err),)

@cython.embedsignature(True)
def cudaGraphDestroy(graph):
    if not isinstance(graph, (cudaGraph_t, cuda.CUgraph)):
        raise TypeError("Argument 'graph' is not instance of type (expected <class 'cudapython.cudart.cudaGraph_t, cuda.CUgraph'>, found " + str(type(graph)))
    err = ccudart.cudaGraphDestroy(<ccudart.cudaGraph_t>(<cudaGraph_t>graph)._ptr[0])
    return (cudaError_t(err),)

@cython.embedsignature(True)
def cudaGraphDebugDotPrint(graph, char* path, unsigned int flags):
    if not isinstance(graph, (cudaGraph_t, cuda.CUgraph)):
        raise TypeError("Argument 'graph' is not instance of type (expected <class 'cudapython.cudart.cudaGraph_t, cuda.CUgraph'>, found " + str(type(graph)))
    err = ccudart.cudaGraphDebugDotPrint(<ccudart.cudaGraph_t>(<cudaGraph_t>graph)._ptr[0], path, flags)
    return (cudaError_t(err),)

@cython.embedsignature(True)
def cudaUserObjectCreate(ptr, destroy not None : cudaHostFn_t, unsigned int initialRefcount, unsigned int flags):
    cdef cudaUserObject_t object_out = cudaUserObject_t()
    cptr = utils.HelperInputVoidPtr(ptr)
    cdef void* cptr_ptr = <void*><void_ptr>cptr.cptr
    err = ccudart.cudaUserObjectCreate(<ccudart.cudaUserObject_t*>object_out._ptr, cptr_ptr, destroy._ptr[0], initialRefcount, flags)
    return (cudaError_t(err), object_out)

@cython.embedsignature(True)
def cudaUserObjectRetain(object, unsigned int count):
    if not isinstance(object, (cudaUserObject_t, cuda.CUuserObject)):
        raise TypeError("Argument 'object' is not instance of type (expected <class 'cudapython.cudart.cudaUserObject_t, cuda.CUuserObject'>, found " + str(type(object)))
    err = ccudart.cudaUserObjectRetain(<ccudart.cudaUserObject_t>(<cudaUserObject_t>object)._ptr[0], count)
    return (cudaError_t(err),)

@cython.embedsignature(True)
def cudaUserObjectRelease(object, unsigned int count):
    if not isinstance(object, (cudaUserObject_t, cuda.CUuserObject)):
        raise TypeError("Argument 'object' is not instance of type (expected <class 'cudapython.cudart.cudaUserObject_t, cuda.CUuserObject'>, found " + str(type(object)))
    err = ccudart.cudaUserObjectRelease(<ccudart.cudaUserObject_t>(<cudaUserObject_t>object)._ptr[0], count)
    return (cudaError_t(err),)

@cython.embedsignature(True)
def cudaGraphRetainUserObject(graph, object, unsigned int count, unsigned int flags):
    if not isinstance(object, (cudaUserObject_t, cuda.CUuserObject)):
        raise TypeError("Argument 'object' is not instance of type (expected <class 'cudapython.cudart.cudaUserObject_t, cuda.CUuserObject'>, found " + str(type(object)))
    if not isinstance(graph, (cudaGraph_t, cuda.CUgraph)):
        raise TypeError("Argument 'graph' is not instance of type (expected <class 'cudapython.cudart.cudaGraph_t, cuda.CUgraph'>, found " + str(type(graph)))
    err = ccudart.cudaGraphRetainUserObject(<ccudart.cudaGraph_t>(<cudaGraph_t>graph)._ptr[0], <ccudart.cudaUserObject_t>(<cudaUserObject_t>object)._ptr[0], count, flags)
    return (cudaError_t(err),)

@cython.embedsignature(True)
def cudaGraphReleaseUserObject(graph, object, unsigned int count):
    if not isinstance(object, (cudaUserObject_t, cuda.CUuserObject)):
        raise TypeError("Argument 'object' is not instance of type (expected <class 'cudapython.cudart.cudaUserObject_t, cuda.CUuserObject'>, found " + str(type(object)))
    if not isinstance(graph, (cudaGraph_t, cuda.CUgraph)):
        raise TypeError("Argument 'graph' is not instance of type (expected <class 'cudapython.cudart.cudaGraph_t, cuda.CUgraph'>, found " + str(type(graph)))
    err = ccudart.cudaGraphReleaseUserObject(<ccudart.cudaGraph_t>(<cudaGraph_t>graph)._ptr[0], <ccudart.cudaUserObject_t>(<cudaUserObject_t>object)._ptr[0], count)
    return (cudaError_t(err),)

@cython.embedsignature(True)
def cudaGetDriverEntryPoint(char* symbol, unsigned long long flags):
    cdef void_ptr funcPtr = 0
    err = ccudart.cudaGetDriverEntryPoint(symbol, <void**>&funcPtr, flags)
    return (cudaError_t(err), funcPtr)

@cython.embedsignature(True)
def cudaGetExportTable(pExportTableId : cudaUUID_t):
    cdef void_ptr ppExportTable = 0
    cdef ccudart.cudaUUID_t* cpExportTableId_ptr = pExportTableId._ptr if pExportTableId != None else NULL
    err = ccudart.cudaGetExportTable(<const void**>&ppExportTable, cpExportTableId_ptr)
    return (cudaError_t(err), ppExportTable)

@cython.embedsignature(True)
def make_cudaPitchedPtr(d, size_t p, size_t xsz, size_t ysz):
    cd = utils.HelperInputVoidPtr(d)
    cdef void* cd_ptr = <void*><void_ptr>cd.cptr
    cdef ccudart.cudaPitchedPtr err
    err = ccudart.make_cudaPitchedPtr(cd_ptr, p, xsz, ysz)
    cdef cudaPitchedPtr wrapper = cudaPitchedPtr()
    wrapper._ptr[0] = err
    return wrapper

@cython.embedsignature(True)
def make_cudaPos(size_t x, size_t y, size_t z):
    cdef ccudart.cudaPos err
    err = ccudart.make_cudaPos(x, y, z)
    cdef cudaPos wrapper = cudaPos()
    wrapper._ptr[0] = err
    return wrapper

@cython.embedsignature(True)
def make_cudaExtent(size_t w, size_t h, size_t d):
    cdef ccudart.cudaExtent err
    err = ccudart.make_cudaExtent(w, h, d)
    cdef cudaExtent wrapper = cudaExtent()
    wrapper._ptr[0] = err
    return wrapper

@cython.embedsignature(True)
def sizeof(objType):
    if objType == dim3:
        return sizeof(ccudart.dim3)
    if objType == cudaChannelFormatDesc:
        return sizeof(ccudart.cudaChannelFormatDesc)
    if objType == cudaArraySparseProperties:
        return sizeof(ccudart.cudaArraySparseProperties)
    if objType == cudaPitchedPtr:
        return sizeof(ccudart.cudaPitchedPtr)
    if objType == cudaExtent:
        return sizeof(ccudart.cudaExtent)
    if objType == cudaPos:
        return sizeof(ccudart.cudaPos)
    if objType == cudaMemcpy3DParms:
        return sizeof(ccudart.cudaMemcpy3DParms)
    if objType == cudaMemcpy3DPeerParms:
        return sizeof(ccudart.cudaMemcpy3DPeerParms)
    if objType == cudaMemsetParams:
        return sizeof(ccudart.cudaMemsetParams)
    if objType == cudaAccessPolicyWindow:
        return sizeof(ccudart.cudaAccessPolicyWindow)
    if objType == cudaHostNodeParams:
        return sizeof(ccudart.cudaHostNodeParams)
    if objType == cudaStreamAttrValue:
        return sizeof(ccudart.cudaStreamAttrValue)
    if objType == cudaKernelNodeAttrValue:
        return sizeof(ccudart.cudaKernelNodeAttrValue)
    if objType == cudaResourceDesc:
        return sizeof(ccudart.cudaResourceDesc)
    if objType == cudaResourceViewDesc:
        return sizeof(ccudart.cudaResourceViewDesc)
    if objType == cudaPointerAttributes:
        return sizeof(ccudart.cudaPointerAttributes)
    if objType == cudaFuncAttributes:
        return sizeof(ccudart.cudaFuncAttributes)
    if objType == cudaMemLocation:
        return sizeof(ccudart.cudaMemLocation)
    if objType == cudaMemAccessDesc:
        return sizeof(ccudart.cudaMemAccessDesc)
    if objType == cudaMemPoolProps:
        return sizeof(ccudart.cudaMemPoolProps)
    if objType == cudaMemPoolPtrExportData:
        return sizeof(ccudart.cudaMemPoolPtrExportData)
    if objType == cudaMemAllocNodeParams:
        return sizeof(ccudart.cudaMemAllocNodeParams)
    if objType == CUuuid_st:
        return sizeof(ccudart.CUuuid_st)
    if objType == cudaDeviceProp:
        return sizeof(ccudart.cudaDeviceProp)
    if objType == cudaIpcEventHandle_st:
        return sizeof(ccudart.cudaIpcEventHandle_st)
    if objType == cudaIpcMemHandle_st:
        return sizeof(ccudart.cudaIpcMemHandle_st)
    if objType == cudaExternalMemoryHandleDesc:
        return sizeof(ccudart.cudaExternalMemoryHandleDesc)
    if objType == cudaExternalMemoryBufferDesc:
        return sizeof(ccudart.cudaExternalMemoryBufferDesc)
    if objType == cudaExternalMemoryMipmappedArrayDesc:
        return sizeof(ccudart.cudaExternalMemoryMipmappedArrayDesc)
    if objType == cudaExternalSemaphoreHandleDesc:
        return sizeof(ccudart.cudaExternalSemaphoreHandleDesc)
    if objType == cudaExternalSemaphoreSignalParams:
        return sizeof(ccudart.cudaExternalSemaphoreSignalParams)
    if objType == cudaExternalSemaphoreWaitParams:
        return sizeof(ccudart.cudaExternalSemaphoreWaitParams)
    if objType == cudaKernelNodeParams:
        return sizeof(ccudart.cudaKernelNodeParams)
    if objType == cudaExternalSemaphoreSignalNodeParams:
        return sizeof(ccudart.cudaExternalSemaphoreSignalNodeParams)
    if objType == cudaExternalSemaphoreWaitNodeParams:
        return sizeof(ccudart.cudaExternalSemaphoreWaitNodeParams)
    if objType == cudaTextureDesc:
        return sizeof(ccudart.cudaTextureDesc)
    if objType == cudaArray_t:
        return sizeof(ccudart.cudaArray_t)
    if objType == cudaArray_const_t:
        return sizeof(ccudart.cudaArray_const_t)
    if objType == cudaMipmappedArray_t:
        return sizeof(ccudart.cudaMipmappedArray_t)
    if objType == cudaMipmappedArray_const_t:
        return sizeof(ccudart.cudaMipmappedArray_const_t)
    if objType == cudaStream_t:
        return sizeof(ccudart.cudaStream_t)
    if objType == cudaEvent_t:
        return sizeof(ccudart.cudaEvent_t)
    if objType == cudaGraphicsResource_t:
        return sizeof(ccudart.cudaGraphicsResource_t)
    if objType == cudaExternalMemory_t:
        return sizeof(ccudart.cudaExternalMemory_t)
    if objType == cudaExternalSemaphore_t:
        return sizeof(ccudart.cudaExternalSemaphore_t)
    if objType == cudaGraph_t:
        return sizeof(ccudart.cudaGraph_t)
    if objType == cudaGraphNode_t:
        return sizeof(ccudart.cudaGraphNode_t)
    if objType == cudaUserObject_t:
        return sizeof(ccudart.cudaUserObject_t)
    if objType == cudaFunction_t:
        return sizeof(ccudart.cudaFunction_t)
    if objType == cudaMemPool_t:
        return sizeof(ccudart.cudaMemPool_t)
    if objType == cudaGraphExec_t:
        return sizeof(ccudart.cudaGraphExec_t)
    if objType == cudaHostFn_t:
        return sizeof(ccudart.cudaHostFn_t)
    if objType == cudaStreamCallback_t:
        return sizeof(ccudart.cudaStreamCallback_t)
    if objType == CUuuid:
        return sizeof(ccudart.CUuuid)
    if objType == cudaUUID_t:
        return sizeof(ccudart.cudaUUID_t)
    if objType == cudaIpcEventHandle_t:
        return sizeof(ccudart.cudaIpcEventHandle_t)
    if objType == cudaIpcMemHandle_t:
        return sizeof(ccudart.cudaIpcMemHandle_t)
    if objType == cudaSurfaceObject_t:
        return sizeof(ccudart.cudaSurfaceObject_t)
    if objType == cudaTextureObject_t:
        return sizeof(ccudart.cudaTextureObject_t)
    raise TypeError("Unknown type: " + str(objType))
