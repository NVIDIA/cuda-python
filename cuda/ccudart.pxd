# Copyright 2021 NVIDIA Corporation.  All rights reserved.
#
# Please refer to the NVIDIA end user license agreement (EULA) associated
# with this source code for terms and conditions that govern your use of
# this software. Any use, reproduction, disclosure, or distribution of
# this software and related documentation outside the terms of the EULA
# is strictly prohibited.

cdef enum cudaRoundMode:
    cudaRoundNearest
    cudaRoundZero
    cudaRoundPosInf
    cudaRoundMinInf

cdef struct dim3:
    unsigned int x
    unsigned int y
    unsigned int z

cdef enum cudaError:
    cudaSuccess = 0
    cudaErrorInvalidValue = 1
    cudaErrorMemoryAllocation = 2
    cudaErrorInitializationError = 3
    cudaErrorCudartUnloading = 4
    cudaErrorProfilerDisabled = 5
    cudaErrorProfilerNotInitialized = 6
    cudaErrorProfilerAlreadyStarted = 7
    cudaErrorProfilerAlreadyStopped = 8
    cudaErrorInvalidConfiguration = 9
    cudaErrorInvalidPitchValue = 12
    cudaErrorInvalidSymbol = 13
    cudaErrorInvalidHostPointer = 16
    cudaErrorInvalidDevicePointer = 17
    cudaErrorInvalidTexture = 18
    cudaErrorInvalidTextureBinding = 19
    cudaErrorInvalidChannelDescriptor = 20
    cudaErrorInvalidMemcpyDirection = 21
    cudaErrorAddressOfConstant = 22
    cudaErrorTextureFetchFailed = 23
    cudaErrorTextureNotBound = 24
    cudaErrorSynchronizationError = 25
    cudaErrorInvalidFilterSetting = 26
    cudaErrorInvalidNormSetting = 27
    cudaErrorMixedDeviceExecution = 28
    cudaErrorNotYetImplemented = 31
    cudaErrorMemoryValueTooLarge = 32
    cudaErrorStubLibrary = 34
    cudaErrorInsufficientDriver = 35
    cudaErrorCallRequiresNewerDriver = 36
    cudaErrorInvalidSurface = 37
    cudaErrorDuplicateVariableName = 43
    cudaErrorDuplicateTextureName = 44
    cudaErrorDuplicateSurfaceName = 45
    cudaErrorDevicesUnavailable = 46
    cudaErrorIncompatibleDriverContext = 49
    cudaErrorMissingConfiguration = 52
    cudaErrorPriorLaunchFailure = 53
    cudaErrorLaunchMaxDepthExceeded = 65
    cudaErrorLaunchFileScopedTex = 66
    cudaErrorLaunchFileScopedSurf = 67
    cudaErrorSyncDepthExceeded = 68
    cudaErrorLaunchPendingCountExceeded = 69
    cudaErrorInvalidDeviceFunction = 98
    cudaErrorNoDevice = 100
    cudaErrorInvalidDevice = 101
    cudaErrorDeviceNotLicensed = 102
    cudaErrorSoftwareValidityNotEstablished = 103
    cudaErrorStartupFailure = 127
    cudaErrorInvalidKernelImage = 200
    cudaErrorDeviceUninitialized = 201
    cudaErrorMapBufferObjectFailed = 205
    cudaErrorUnmapBufferObjectFailed = 206
    cudaErrorArrayIsMapped = 207
    cudaErrorAlreadyMapped = 208
    cudaErrorNoKernelImageForDevice = 209
    cudaErrorAlreadyAcquired = 210
    cudaErrorNotMapped = 211
    cudaErrorNotMappedAsArray = 212
    cudaErrorNotMappedAsPointer = 213
    cudaErrorECCUncorrectable = 214
    cudaErrorUnsupportedLimit = 215
    cudaErrorDeviceAlreadyInUse = 216
    cudaErrorPeerAccessUnsupported = 217
    cudaErrorInvalidPtx = 218
    cudaErrorInvalidGraphicsContext = 219
    cudaErrorNvlinkUncorrectable = 220
    cudaErrorJitCompilerNotFound = 221
    cudaErrorUnsupportedPtxVersion = 222
    cudaErrorJitCompilationDisabled = 223
    cudaErrorUnsupportedExecAffinity = 224
    cudaErrorInvalidSource = 300
    cudaErrorFileNotFound = 301
    cudaErrorSharedObjectSymbolNotFound = 302
    cudaErrorSharedObjectInitFailed = 303
    cudaErrorOperatingSystem = 304
    cudaErrorInvalidResourceHandle = 400
    cudaErrorIllegalState = 401
    cudaErrorSymbolNotFound = 500
    cudaErrorNotReady = 600
    cudaErrorIllegalAddress = 700
    cudaErrorLaunchOutOfResources = 701
    cudaErrorLaunchTimeout = 702
    cudaErrorLaunchIncompatibleTexturing = 703
    cudaErrorPeerAccessAlreadyEnabled = 704
    cudaErrorPeerAccessNotEnabled = 705
    cudaErrorSetOnActiveProcess = 708
    cudaErrorContextIsDestroyed = 709
    cudaErrorAssert = 710
    cudaErrorTooManyPeers = 711
    cudaErrorHostMemoryAlreadyRegistered = 712
    cudaErrorHostMemoryNotRegistered = 713
    cudaErrorHardwareStackError = 714
    cudaErrorIllegalInstruction = 715
    cudaErrorMisalignedAddress = 716
    cudaErrorInvalidAddressSpace = 717
    cudaErrorInvalidPc = 718
    cudaErrorLaunchFailure = 719
    cudaErrorCooperativeLaunchTooLarge = 720
    cudaErrorNotPermitted = 800
    cudaErrorNotSupported = 801
    cudaErrorSystemNotReady = 802
    cudaErrorSystemDriverMismatch = 803
    cudaErrorCompatNotSupportedOnDevice = 804
    cudaErrorMpsConnectionFailed = 805
    cudaErrorMpsRpcFailure = 806
    cudaErrorMpsServerNotReady = 807
    cudaErrorMpsMaxClientsReached = 808
    cudaErrorMpsMaxConnectionsReached = 809
    cudaErrorStreamCaptureUnsupported = 900
    cudaErrorStreamCaptureInvalidated = 901
    cudaErrorStreamCaptureMerge = 902
    cudaErrorStreamCaptureUnmatched = 903
    cudaErrorStreamCaptureUnjoined = 904
    cudaErrorStreamCaptureIsolation = 905
    cudaErrorStreamCaptureImplicit = 906
    cudaErrorCapturedEvent = 907
    cudaErrorStreamCaptureWrongThread = 908
    cudaErrorTimeout = 909
    cudaErrorGraphExecUpdateFailure = 910
    cudaErrorExternalDevice = 911
    cudaErrorUnknown = 999
    cudaErrorApiFailureBase = 10000

cdef enum cudaChannelFormatKind:
    cudaChannelFormatKindSigned = 0
    cudaChannelFormatKindUnsigned = 1
    cudaChannelFormatKindFloat = 2
    cudaChannelFormatKindNone = 3
    cudaChannelFormatKindNV12 = 4
    cudaChannelFormatKindUnsignedNormalized8X1 = 5
    cudaChannelFormatKindUnsignedNormalized8X2 = 6
    cudaChannelFormatKindUnsignedNormalized8X4 = 7
    cudaChannelFormatKindUnsignedNormalized16X1 = 8
    cudaChannelFormatKindUnsignedNormalized16X2 = 9
    cudaChannelFormatKindUnsignedNormalized16X4 = 10
    cudaChannelFormatKindSignedNormalized8X1 = 11
    cudaChannelFormatKindSignedNormalized8X2 = 12
    cudaChannelFormatKindSignedNormalized8X4 = 13
    cudaChannelFormatKindSignedNormalized16X1 = 14
    cudaChannelFormatKindSignedNormalized16X2 = 15
    cudaChannelFormatKindSignedNormalized16X4 = 16
    cudaChannelFormatKindUnsignedBlockCompressed1 = 17
    cudaChannelFormatKindUnsignedBlockCompressed1SRGB = 18
    cudaChannelFormatKindUnsignedBlockCompressed2 = 19
    cudaChannelFormatKindUnsignedBlockCompressed2SRGB = 20
    cudaChannelFormatKindUnsignedBlockCompressed3 = 21
    cudaChannelFormatKindUnsignedBlockCompressed3SRGB = 22
    cudaChannelFormatKindUnsignedBlockCompressed4 = 23
    cudaChannelFormatKindSignedBlockCompressed4 = 24
    cudaChannelFormatKindUnsignedBlockCompressed5 = 25
    cudaChannelFormatKindSignedBlockCompressed5 = 26
    cudaChannelFormatKindUnsignedBlockCompressed6H = 27
    cudaChannelFormatKindSignedBlockCompressed6H = 28
    cudaChannelFormatKindUnsignedBlockCompressed7 = 29
    cudaChannelFormatKindUnsignedBlockCompressed7SRGB = 30

cdef struct cudaChannelFormatDesc:
    int x
    int y
    int z
    int w
    cudaChannelFormatKind f

cdef extern from "":
    cdef struct cudaArray:
        pass
ctypedef cudaArray* cudaArray_t

cdef extern from "":
    cdef struct cudaArray:
        pass
ctypedef cudaArray* cudaArray_const_t


cdef extern from "":
    cdef struct cudaMipmappedArray:
        pass
ctypedef cudaMipmappedArray* cudaMipmappedArray_t

cdef extern from "":
    cdef struct cudaMipmappedArray:
        pass
ctypedef cudaMipmappedArray* cudaMipmappedArray_const_t


cdef struct _cudaArraySparseProperties_tileExtent_s:
    unsigned int width
    unsigned int height
    unsigned int depth

cdef struct cudaArraySparseProperties:
    _cudaArraySparseProperties_tileExtent_s tileExtent
    unsigned int miptailFirstLevel
    unsigned long long miptailSize
    unsigned int flags
    unsigned int reserved[4]

cdef enum cudaMemoryType:
    cudaMemoryTypeUnregistered = 0
    cudaMemoryTypeHost = 1
    cudaMemoryTypeDevice = 2
    cudaMemoryTypeManaged = 3

cdef enum cudaMemcpyKind:
    cudaMemcpyHostToHost = 0
    cudaMemcpyHostToDevice = 1
    cudaMemcpyDeviceToHost = 2
    cudaMemcpyDeviceToDevice = 3
    cudaMemcpyDefault = 4

cdef struct cudaPitchedPtr:
    void* ptr
    size_t pitch
    size_t xsize
    size_t ysize

cdef struct cudaExtent:
    size_t width
    size_t height
    size_t depth

cdef struct cudaPos:
    size_t x
    size_t y
    size_t z

cdef struct cudaMemcpy3DParms:
    cudaArray_t srcArray
    cudaPos srcPos
    cudaPitchedPtr srcPtr
    cudaArray_t dstArray
    cudaPos dstPos
    cudaPitchedPtr dstPtr
    cudaExtent extent
    cudaMemcpyKind kind

cdef struct cudaMemcpy3DPeerParms:
    cudaArray_t srcArray
    cudaPos srcPos
    cudaPitchedPtr srcPtr
    int srcDevice
    cudaArray_t dstArray
    cudaPos dstPos
    cudaPitchedPtr dstPtr
    int dstDevice
    cudaExtent extent

cdef struct cudaMemsetParams:
    void* dst
    size_t pitch
    unsigned int value
    unsigned int elementSize
    size_t width
    size_t height

cdef enum cudaAccessProperty:
    cudaAccessPropertyNormal = 0
    cudaAccessPropertyStreaming = 1
    cudaAccessPropertyPersisting = 2

cdef struct cudaAccessPolicyWindow:
    void* base_ptr
    size_t num_bytes
    float hitRatio
    cudaAccessProperty hitProp
    cudaAccessProperty missProp

ctypedef void (*cudaHostFn_t)(void* userData)

cdef struct cudaHostNodeParams:
    cudaHostFn_t fn
    void* userData

cdef enum cudaStreamCaptureStatus:
    cudaStreamCaptureStatusNone = 0
    cudaStreamCaptureStatusActive = 1
    cudaStreamCaptureStatusInvalidated = 2

cdef enum cudaStreamCaptureMode:
    cudaStreamCaptureModeGlobal = 0
    cudaStreamCaptureModeThreadLocal = 1
    cudaStreamCaptureModeRelaxed = 2

cdef enum cudaSynchronizationPolicy:
    cudaSyncPolicyAuto = 1
    cudaSyncPolicySpin = 2
    cudaSyncPolicyYield = 3
    cudaSyncPolicyBlockingSync = 4

cdef enum cudaStreamAttrID:
    cudaStreamAttributeAccessPolicyWindow = 1
    cudaStreamAttributeSynchronizationPolicy = 3

cdef union cudaStreamAttrValue:
    cudaAccessPolicyWindow accessPolicyWindow
    cudaSynchronizationPolicy syncPolicy

cdef enum cudaStreamUpdateCaptureDependenciesFlags:
    cudaStreamAddCaptureDependencies = 0x0
    cudaStreamSetCaptureDependencies = 0x1

cdef enum cudaUserObjectFlags:
    cudaUserObjectNoDestructorSync = 0x1

cdef enum cudaUserObjectRetainFlags:
    cudaGraphUserObjectMove = 0x1


cdef enum cudaGraphicsRegisterFlags:
    cudaGraphicsRegisterFlagsNone = 0
    cudaGraphicsRegisterFlagsReadOnly = 1
    cudaGraphicsRegisterFlagsWriteDiscard = 2
    cudaGraphicsRegisterFlagsSurfaceLoadStore = 4
    cudaGraphicsRegisterFlagsTextureGather = 8

cdef enum cudaGraphicsMapFlags:
    cudaGraphicsMapFlagsNone = 0
    cudaGraphicsMapFlagsReadOnly = 1
    cudaGraphicsMapFlagsWriteDiscard = 2

cdef enum cudaGraphicsCubeFace:
    cudaGraphicsCubeFacePositiveX = 0x00
    cudaGraphicsCubeFaceNegativeX = 0x01
    cudaGraphicsCubeFacePositiveY = 0x02
    cudaGraphicsCubeFaceNegativeY = 0x03
    cudaGraphicsCubeFacePositiveZ = 0x04
    cudaGraphicsCubeFaceNegativeZ = 0x05

cdef enum cudaKernelNodeAttrID:
    cudaKernelNodeAttributeAccessPolicyWindow = 1
    cudaKernelNodeAttributeCooperative = 2

cdef union cudaKernelNodeAttrValue:
    cudaAccessPolicyWindow accessPolicyWindow
    int cooperative

cdef enum cudaResourceType:
    cudaResourceTypeArray = 0x00
    cudaResourceTypeMipmappedArray = 0x01
    cudaResourceTypeLinear = 0x02
    cudaResourceTypePitch2D = 0x03

cdef enum cudaResourceViewFormat:
    cudaResViewFormatNone = 0x00
    cudaResViewFormatUnsignedChar1 = 0x01
    cudaResViewFormatUnsignedChar2 = 0x02
    cudaResViewFormatUnsignedChar4 = 0x03
    cudaResViewFormatSignedChar1 = 0x04
    cudaResViewFormatSignedChar2 = 0x05
    cudaResViewFormatSignedChar4 = 0x06
    cudaResViewFormatUnsignedShort1 = 0x07
    cudaResViewFormatUnsignedShort2 = 0x08
    cudaResViewFormatUnsignedShort4 = 0x09
    cudaResViewFormatSignedShort1 = 0x0a
    cudaResViewFormatSignedShort2 = 0x0b
    cudaResViewFormatSignedShort4 = 0x0c
    cudaResViewFormatUnsignedInt1 = 0x0d
    cudaResViewFormatUnsignedInt2 = 0x0e
    cudaResViewFormatUnsignedInt4 = 0x0f
    cudaResViewFormatSignedInt1 = 0x10
    cudaResViewFormatSignedInt2 = 0x11
    cudaResViewFormatSignedInt4 = 0x12
    cudaResViewFormatHalf1 = 0x13
    cudaResViewFormatHalf2 = 0x14
    cudaResViewFormatHalf4 = 0x15
    cudaResViewFormatFloat1 = 0x16
    cudaResViewFormatFloat2 = 0x17
    cudaResViewFormatFloat4 = 0x18
    cudaResViewFormatUnsignedBlockCompressed1 = 0x19
    cudaResViewFormatUnsignedBlockCompressed2 = 0x1a
    cudaResViewFormatUnsignedBlockCompressed3 = 0x1b
    cudaResViewFormatUnsignedBlockCompressed4 = 0x1c
    cudaResViewFormatSignedBlockCompressed4 = 0x1d
    cudaResViewFormatUnsignedBlockCompressed5 = 0x1e
    cudaResViewFormatSignedBlockCompressed5 = 0x1f
    cudaResViewFormatUnsignedBlockCompressed6H = 0x20
    cudaResViewFormatSignedBlockCompressed6H = 0x21
    cudaResViewFormatUnsignedBlockCompressed7 = 0x22

cdef struct _cudaResourceDesc_res_res_array_s:
    cudaArray_t array

cdef struct _cudaResourceDesc_res_res_mipmap_s:
    cudaMipmappedArray_t mipmap

cdef struct _cudaResourceDesc_res_res_linear_s:
    void* devPtr
    cudaChannelFormatDesc desc
    size_t sizeInBytes

cdef struct _cudaResourceDesc_res_res_pitch2D_s:
    void* devPtr
    cudaChannelFormatDesc desc
    size_t width
    size_t height
    size_t pitchInBytes

cdef union _cudaResourceDesc_res_u:
    _cudaResourceDesc_res_res_array_s array
    _cudaResourceDesc_res_res_mipmap_s mipmap
    _cudaResourceDesc_res_res_linear_s linear
    _cudaResourceDesc_res_res_pitch2D_s pitch2D

cdef struct cudaResourceDesc:
    cudaResourceType resType
    _cudaResourceDesc_res_u res

cdef struct cudaResourceViewDesc:
    cudaResourceViewFormat format
    size_t width
    size_t height
    size_t depth
    unsigned int firstMipmapLevel
    unsigned int lastMipmapLevel
    unsigned int firstLayer
    unsigned int lastLayer

cdef struct cudaPointerAttributes:
    cudaMemoryType type
    int device
    void* devicePointer
    void* hostPointer

cdef struct cudaFuncAttributes:
    size_t sharedSizeBytes
    size_t constSizeBytes
    size_t localSizeBytes
    int maxThreadsPerBlock
    int numRegs
    int ptxVersion
    int binaryVersion
    int cacheModeCA
    int maxDynamicSharedSizeBytes
    int preferredShmemCarveout

cdef enum cudaFuncAttribute:
    cudaFuncAttributeMaxDynamicSharedMemorySize = 8
    cudaFuncAttributePreferredSharedMemoryCarveout = 9
    cudaFuncAttributeMax

cdef enum cudaFuncCache:
    cudaFuncCachePreferNone = 0
    cudaFuncCachePreferShared = 1
    cudaFuncCachePreferL1 = 2
    cudaFuncCachePreferEqual = 3

cdef enum cudaSharedMemConfig:
    cudaSharedMemBankSizeDefault = 0
    cudaSharedMemBankSizeFourByte = 1
    cudaSharedMemBankSizeEightByte = 2

cdef enum cudaSharedCarveout:
    cudaSharedmemCarveoutDefault = -1
    cudaSharedmemCarveoutMaxShared = 100
    cudaSharedmemCarveoutMaxL1 = 0

cdef enum cudaComputeMode:
    cudaComputeModeDefault = 0
    cudaComputeModeExclusive = 1
    cudaComputeModeProhibited = 2
    cudaComputeModeExclusiveProcess = 3

cdef enum cudaLimit:
    cudaLimitStackSize = 0x00
    cudaLimitPrintfFifoSize = 0x01
    cudaLimitMallocHeapSize = 0x02
    cudaLimitDevRuntimeSyncDepth = 0x03
    cudaLimitDevRuntimePendingLaunchCount = 0x04
    cudaLimitMaxL2FetchGranularity = 0x05
    cudaLimitPersistingL2CacheSize = 0x06

cdef enum cudaMemoryAdvise:
    cudaMemAdviseSetReadMostly = 1
    cudaMemAdviseUnsetReadMostly = 2
    cudaMemAdviseSetPreferredLocation = 3
    cudaMemAdviseUnsetPreferredLocation = 4
    cudaMemAdviseSetAccessedBy = 5
    cudaMemAdviseUnsetAccessedBy = 6

cdef enum cudaMemRangeAttribute:
    cudaMemRangeAttributeReadMostly = 1
    cudaMemRangeAttributePreferredLocation = 2
    cudaMemRangeAttributeAccessedBy = 3
    cudaMemRangeAttributeLastPrefetchLocation = 4

cdef enum cudaOutputMode:
    cudaKeyValuePair = 0x00
    cudaCSV = 0x01

cdef enum cudaFlushGPUDirectRDMAWritesOptions:
    cudaFlushGPUDirectRDMAWritesOptionHost = 1<<0
    cudaFlushGPUDirectRDMAWritesOptionMemOps = 1<<1

cdef enum cudaGPUDirectRDMAWritesOrdering:
    cudaGPUDirectRDMAWritesOrderingNone = 0
    cudaGPUDirectRDMAWritesOrderingOwner = 100
    cudaGPUDirectRDMAWritesOrderingAllDevices = 200

cdef enum cudaFlushGPUDirectRDMAWritesScope:
    cudaFlushGPUDirectRDMAWritesToOwner = 100
    cudaFlushGPUDirectRDMAWritesToAllDevices = 200

cdef enum cudaFlushGPUDirectRDMAWritesTarget:
    cudaFlushGPUDirectRDMAWritesTargetCurrentDevice

cdef enum cudaDeviceAttr:
    cudaDevAttrMaxThreadsPerBlock = 1
    cudaDevAttrMaxBlockDimX = 2
    cudaDevAttrMaxBlockDimY = 3
    cudaDevAttrMaxBlockDimZ = 4
    cudaDevAttrMaxGridDimX = 5
    cudaDevAttrMaxGridDimY = 6
    cudaDevAttrMaxGridDimZ = 7
    cudaDevAttrMaxSharedMemoryPerBlock = 8
    cudaDevAttrTotalConstantMemory = 9
    cudaDevAttrWarpSize = 10
    cudaDevAttrMaxPitch = 11
    cudaDevAttrMaxRegistersPerBlock = 12
    cudaDevAttrClockRate = 13
    cudaDevAttrTextureAlignment = 14
    cudaDevAttrGpuOverlap = 15
    cudaDevAttrMultiProcessorCount = 16
    cudaDevAttrKernelExecTimeout = 17
    cudaDevAttrIntegrated = 18
    cudaDevAttrCanMapHostMemory = 19
    cudaDevAttrComputeMode = 20
    cudaDevAttrMaxTexture1DWidth = 21
    cudaDevAttrMaxTexture2DWidth = 22
    cudaDevAttrMaxTexture2DHeight = 23
    cudaDevAttrMaxTexture3DWidth = 24
    cudaDevAttrMaxTexture3DHeight = 25
    cudaDevAttrMaxTexture3DDepth = 26
    cudaDevAttrMaxTexture2DLayeredWidth = 27
    cudaDevAttrMaxTexture2DLayeredHeight = 28
    cudaDevAttrMaxTexture2DLayeredLayers = 29
    cudaDevAttrSurfaceAlignment = 30
    cudaDevAttrConcurrentKernels = 31
    cudaDevAttrEccEnabled = 32
    cudaDevAttrPciBusId = 33
    cudaDevAttrPciDeviceId = 34
    cudaDevAttrTccDriver = 35
    cudaDevAttrMemoryClockRate = 36
    cudaDevAttrGlobalMemoryBusWidth = 37
    cudaDevAttrL2CacheSize = 38
    cudaDevAttrMaxThreadsPerMultiProcessor = 39
    cudaDevAttrAsyncEngineCount = 40
    cudaDevAttrUnifiedAddressing = 41
    cudaDevAttrMaxTexture1DLayeredWidth = 42
    cudaDevAttrMaxTexture1DLayeredLayers = 43
    cudaDevAttrMaxTexture2DGatherWidth = 45
    cudaDevAttrMaxTexture2DGatherHeight = 46
    cudaDevAttrMaxTexture3DWidthAlt = 47
    cudaDevAttrMaxTexture3DHeightAlt = 48
    cudaDevAttrMaxTexture3DDepthAlt = 49
    cudaDevAttrPciDomainId = 50
    cudaDevAttrTexturePitchAlignment = 51
    cudaDevAttrMaxTextureCubemapWidth = 52
    cudaDevAttrMaxTextureCubemapLayeredWidth = 53
    cudaDevAttrMaxTextureCubemapLayeredLayers = 54
    cudaDevAttrMaxSurface1DWidth = 55
    cudaDevAttrMaxSurface2DWidth = 56
    cudaDevAttrMaxSurface2DHeight = 57
    cudaDevAttrMaxSurface3DWidth = 58
    cudaDevAttrMaxSurface3DHeight = 59
    cudaDevAttrMaxSurface3DDepth = 60
    cudaDevAttrMaxSurface1DLayeredWidth = 61
    cudaDevAttrMaxSurface1DLayeredLayers = 62
    cudaDevAttrMaxSurface2DLayeredWidth = 63
    cudaDevAttrMaxSurface2DLayeredHeight = 64
    cudaDevAttrMaxSurface2DLayeredLayers = 65
    cudaDevAttrMaxSurfaceCubemapWidth = 66
    cudaDevAttrMaxSurfaceCubemapLayeredWidth = 67
    cudaDevAttrMaxSurfaceCubemapLayeredLayers = 68
    cudaDevAttrMaxTexture1DLinearWidth = 69
    cudaDevAttrMaxTexture2DLinearWidth = 70
    cudaDevAttrMaxTexture2DLinearHeight = 71
    cudaDevAttrMaxTexture2DLinearPitch = 72
    cudaDevAttrMaxTexture2DMipmappedWidth = 73
    cudaDevAttrMaxTexture2DMipmappedHeight = 74
    cudaDevAttrComputeCapabilityMajor = 75
    cudaDevAttrComputeCapabilityMinor = 76
    cudaDevAttrMaxTexture1DMipmappedWidth = 77
    cudaDevAttrStreamPrioritiesSupported = 78
    cudaDevAttrGlobalL1CacheSupported = 79
    cudaDevAttrLocalL1CacheSupported = 80
    cudaDevAttrMaxSharedMemoryPerMultiprocessor = 81
    cudaDevAttrMaxRegistersPerMultiprocessor = 82
    cudaDevAttrManagedMemory = 83
    cudaDevAttrIsMultiGpuBoard = 84
    cudaDevAttrMultiGpuBoardGroupID = 85
    cudaDevAttrHostNativeAtomicSupported = 86
    cudaDevAttrSingleToDoublePrecisionPerfRatio = 87
    cudaDevAttrPageableMemoryAccess = 88
    cudaDevAttrConcurrentManagedAccess = 89
    cudaDevAttrComputePreemptionSupported = 90
    cudaDevAttrCanUseHostPointerForRegisteredMem = 91
    cudaDevAttrReserved92 = 92
    cudaDevAttrReserved93 = 93
    cudaDevAttrReserved94 = 94
    cudaDevAttrCooperativeLaunch = 95
    cudaDevAttrCooperativeMultiDeviceLaunch = 96
    cudaDevAttrMaxSharedMemoryPerBlockOptin = 97
    cudaDevAttrCanFlushRemoteWrites = 98
    cudaDevAttrHostRegisterSupported = 99
    cudaDevAttrPageableMemoryAccessUsesHostPageTables = 100
    cudaDevAttrDirectManagedMemAccessFromHost = 101
    cudaDevAttrMaxBlocksPerMultiprocessor = 106
    cudaDevAttrMaxPersistingL2CacheSize = 108
    cudaDevAttrMaxAccessPolicyWindowSize = 109
    cudaDevAttrReservedSharedMemoryPerBlock = 111
    cudaDevAttrSparseCudaArraySupported = 112
    cudaDevAttrHostRegisterReadOnlySupported = 113
    cudaDevAttrTimelineSemaphoreInteropSupported = 114
    cudaDevAttrMaxTimelineSemaphoreInteropSupported = 114
    cudaDevAttrMemoryPoolsSupported = 115
    cudaDevAttrGPUDirectRDMASupported = 116
    cudaDevAttrGPUDirectRDMAFlushWritesOptions = 117
    cudaDevAttrGPUDirectRDMAWritesOrdering = 118
    cudaDevAttrMemoryPoolSupportedHandleTypes = 119
    cudaDevAttrMax

cdef enum cudaMemPoolAttr:
    cudaMemPoolReuseFollowEventDependencies = 0x1
    cudaMemPoolReuseAllowOpportunistic = 0x2
    cudaMemPoolReuseAllowInternalDependencies = 0x3
    cudaMemPoolAttrReleaseThreshold = 0x4
    cudaMemPoolAttrReservedMemCurrent = 0x5
    cudaMemPoolAttrReservedMemHigh = 0x6
    cudaMemPoolAttrUsedMemCurrent = 0x7
    cudaMemPoolAttrUsedMemHigh = 0x8

cdef enum cudaMemLocationType:
    cudaMemLocationTypeInvalid = 0
    cudaMemLocationTypeDevice = 1

cdef struct cudaMemLocation:
    cudaMemLocationType type
    int id

cdef enum cudaMemAccessFlags:
    cudaMemAccessFlagsProtNone = 0
    cudaMemAccessFlagsProtRead = 1
    cudaMemAccessFlagsProtReadWrite = 3

cdef struct cudaMemAccessDesc:
    cudaMemLocation location
    cudaMemAccessFlags flags

cdef enum cudaMemAllocationType:
    cudaMemAllocationTypeInvalid = 0x0
    cudaMemAllocationTypePinned = 0x1
    cudaMemAllocationTypeMax = 0x7FFFFFFF

cdef enum cudaMemAllocationHandleType:
    cudaMemHandleTypeNone = 0x0
    cudaMemHandleTypePosixFileDescriptor = 0x1
    cudaMemHandleTypeWin32 = 0x2
    cudaMemHandleTypeWin32Kmt = 0x4

cdef struct cudaMemPoolProps:
    cudaMemAllocationType allocType
    cudaMemAllocationHandleType handleTypes
    cudaMemLocation location
    void* win32SecurityAttributes
    unsigned char reserved[64]

cdef struct cudaMemPoolPtrExportData:
    unsigned char reserved[64]

cdef struct cudaMemAllocNodeParams:
    cudaMemPoolProps poolProps
    const cudaMemAccessDesc* accessDescs
    size_t accessDescCount
    size_t bytesize
    void* dptr

cdef enum cudaGraphMemAttributeType:
    cudaGraphMemAttrUsedMemCurrent = 0x1
    cudaGraphMemAttrUsedMemHigh = 0x2
    cudaGraphMemAttrReservedMemCurrent = 0x3
    cudaGraphMemAttrReservedMemHigh = 0x4

cdef enum cudaDeviceP2PAttr:
    cudaDevP2PAttrPerformanceRank = 1
    cudaDevP2PAttrAccessSupported = 2
    cudaDevP2PAttrNativeAtomicSupported = 3
    cudaDevP2PAttrCudaArrayAccessSupported = 4

cdef struct CUuuid_st:
    char bytes[16]

ctypedef CUuuid_st CUuuid

ctypedef CUuuid_st cudaUUID_t

cdef struct cudaDeviceProp:
    char name[256]
    cudaUUID_t uuid
    char luid[8]
    unsigned int luidDeviceNodeMask
    size_t totalGlobalMem
    size_t sharedMemPerBlock
    int regsPerBlock
    int warpSize
    size_t memPitch
    int maxThreadsPerBlock
    int maxThreadsDim[3]
    int maxGridSize[3]
    int clockRate
    size_t totalConstMem
    int major
    int minor
    size_t textureAlignment
    size_t texturePitchAlignment
    int deviceOverlap
    int multiProcessorCount
    int kernelExecTimeoutEnabled
    int integrated
    int canMapHostMemory
    int computeMode
    int maxTexture1D
    int maxTexture1DMipmap
    int maxTexture1DLinear
    int maxTexture2D[2]
    int maxTexture2DMipmap[2]
    int maxTexture2DLinear[3]
    int maxTexture2DGather[2]
    int maxTexture3D[3]
    int maxTexture3DAlt[3]
    int maxTextureCubemap
    int maxTexture1DLayered[2]
    int maxTexture2DLayered[3]
    int maxTextureCubemapLayered[2]
    int maxSurface1D
    int maxSurface2D[2]
    int maxSurface3D[3]
    int maxSurface1DLayered[2]
    int maxSurface2DLayered[3]
    int maxSurfaceCubemap
    int maxSurfaceCubemapLayered[2]
    size_t surfaceAlignment
    int concurrentKernels
    int ECCEnabled
    int pciBusID
    int pciDeviceID
    int pciDomainID
    int tccDriver
    int asyncEngineCount
    int unifiedAddressing
    int memoryClockRate
    int memoryBusWidth
    int l2CacheSize
    int persistingL2CacheMaxSize
    int maxThreadsPerMultiProcessor
    int streamPrioritiesSupported
    int globalL1CacheSupported
    int localL1CacheSupported
    size_t sharedMemPerMultiprocessor
    int regsPerMultiprocessor
    int managedMemory
    int isMultiGpuBoard
    int multiGpuBoardGroupID
    int hostNativeAtomicSupported
    int singleToDoublePrecisionPerfRatio
    int pageableMemoryAccess
    int concurrentManagedAccess
    int computePreemptionSupported
    int canUseHostPointerForRegisteredMem
    int cooperativeLaunch
    int cooperativeMultiDeviceLaunch
    size_t sharedMemPerBlockOptin
    int pageableMemoryAccessUsesHostPageTables
    int directManagedMemAccessFromHost
    int maxBlocksPerMultiProcessor
    int accessPolicyMaxWindowSize
    size_t reservedSharedMemPerBlock

cdef struct cudaIpcEventHandle_st:
    char reserved[64]

ctypedef cudaIpcEventHandle_st cudaIpcEventHandle_t

cdef struct cudaIpcMemHandle_st:
    char reserved[64]

ctypedef cudaIpcMemHandle_st cudaIpcMemHandle_t

cdef enum cudaExternalMemoryHandleType:
    cudaExternalMemoryHandleTypeOpaqueFd = 1
    cudaExternalMemoryHandleTypeOpaqueWin32 = 2
    cudaExternalMemoryHandleTypeOpaqueWin32Kmt = 3
    cudaExternalMemoryHandleTypeD3D12Heap = 4
    cudaExternalMemoryHandleTypeD3D12Resource = 5
    cudaExternalMemoryHandleTypeD3D11Resource = 6
    cudaExternalMemoryHandleTypeD3D11ResourceKmt = 7
    cudaExternalMemoryHandleTypeNvSciBuf = 8

cdef struct _cudaExternalMemoryHandleDesc_handle_handle_win32_s:
    void* handle
    void* name

cdef union _cudaExternalMemoryHandleDesc_handle_u:
    int fd
    _cudaExternalMemoryHandleDesc_handle_handle_win32_s win32
    void* nvSciBufObject

cdef struct cudaExternalMemoryHandleDesc:
    cudaExternalMemoryHandleType type
    _cudaExternalMemoryHandleDesc_handle_u handle
    unsigned long long size
    unsigned int flags

cdef struct cudaExternalMemoryBufferDesc:
    unsigned long long offset
    unsigned long long size
    unsigned int flags

cdef struct cudaExternalMemoryMipmappedArrayDesc:
    unsigned long long offset
    cudaChannelFormatDesc formatDesc
    cudaExtent extent
    unsigned int flags
    unsigned int numLevels

cdef enum cudaExternalSemaphoreHandleType:
    cudaExternalSemaphoreHandleTypeOpaqueFd = 1
    cudaExternalSemaphoreHandleTypeOpaqueWin32 = 2
    cudaExternalSemaphoreHandleTypeOpaqueWin32Kmt = 3
    cudaExternalSemaphoreHandleTypeD3D12Fence = 4
    cudaExternalSemaphoreHandleTypeD3D11Fence = 5
    cudaExternalSemaphoreHandleTypeNvSciSync = 6
    cudaExternalSemaphoreHandleTypeKeyedMutex = 7
    cudaExternalSemaphoreHandleTypeKeyedMutexKmt = 8
    cudaExternalSemaphoreHandleTypeTimelineSemaphoreFd = 9
    cudaExternalSemaphoreHandleTypeTimelineSemaphoreWin32 = 10

cdef struct _cudaExternalSemaphoreHandleDesc_handle_handle_win32_s:
    void* handle
    void* name

cdef union _cudaExternalSemaphoreHandleDesc_handle_u:
    int fd
    _cudaExternalSemaphoreHandleDesc_handle_handle_win32_s win32
    void* nvSciSyncObj

cdef struct cudaExternalSemaphoreHandleDesc:
    cudaExternalSemaphoreHandleType type
    _cudaExternalSemaphoreHandleDesc_handle_u handle
    unsigned int flags

cdef struct _cudaExternalSemaphoreSignalParams_params_params_fence_s:
    unsigned long long value

cdef union _cudaExternalSemaphoreSignalParams_params_params_nvSciSync_u:
    void* fence
    unsigned long long reserved

cdef struct _cudaExternalSemaphoreSignalParams_params_params_keyedMutex_s:
    unsigned long long key

cdef struct _cudaExternalSemaphoreSignalParams_params_s:
    _cudaExternalSemaphoreSignalParams_params_params_fence_s fence
    _cudaExternalSemaphoreSignalParams_params_params_nvSciSync_u nvSciSync
    _cudaExternalSemaphoreSignalParams_params_params_keyedMutex_s keyedMutex
    unsigned int reserved[12]

cdef struct cudaExternalSemaphoreSignalParams:
    _cudaExternalSemaphoreSignalParams_params_s params
    unsigned int flags
    unsigned int reserved[16]

cdef struct _cudaExternalSemaphoreWaitParams_params_params_fence_s:
    unsigned long long value

cdef union _cudaExternalSemaphoreWaitParams_params_params_nvSciSync_u:
    void* fence
    unsigned long long reserved

cdef struct _cudaExternalSemaphoreWaitParams_params_params_keyedMutex_s:
    unsigned long long key
    unsigned int timeoutMs

cdef struct _cudaExternalSemaphoreWaitParams_params_s:
    _cudaExternalSemaphoreWaitParams_params_params_fence_s fence
    _cudaExternalSemaphoreWaitParams_params_params_nvSciSync_u nvSciSync
    _cudaExternalSemaphoreWaitParams_params_params_keyedMutex_s keyedMutex
    unsigned int reserved[10]

cdef struct cudaExternalSemaphoreWaitParams:
    _cudaExternalSemaphoreWaitParams_params_s params
    unsigned int flags
    unsigned int reserved[16]

ctypedef cudaError cudaError_t

cdef extern from "":
    cdef struct CUstream_st:
        pass
ctypedef CUstream_st* cudaStream_t

cdef extern from "":
    cdef struct CUevent_st:
        pass
ctypedef CUevent_st* cudaEvent_t

cdef extern from "":
    cdef struct cudaGraphicsResource:
        pass
ctypedef cudaGraphicsResource* cudaGraphicsResource_t

ctypedef cudaOutputMode cudaOutputMode_t

cdef extern from "":
    cdef struct CUexternalMemory_st:
        pass
ctypedef CUexternalMemory_st* cudaExternalMemory_t

cdef extern from "":
    cdef struct CUexternalSemaphore_st:
        pass
ctypedef CUexternalSemaphore_st* cudaExternalSemaphore_t

cdef extern from "":
    cdef struct CUgraph_st:
        pass
ctypedef CUgraph_st* cudaGraph_t

cdef extern from "":
    cdef struct CUgraphNode_st:
        pass
ctypedef CUgraphNode_st* cudaGraphNode_t

cdef extern from "":
    cdef struct CUuserObject_st:
        pass
ctypedef CUuserObject_st* cudaUserObject_t

cdef extern from "":
    cdef struct CUfunc_st:
        pass
ctypedef CUfunc_st* cudaFunction_t

cdef extern from "":
    cdef struct CUmemPoolHandle_st:
        pass
ctypedef CUmemPoolHandle_st* cudaMemPool_t

cdef enum cudaCGScope:
    cudaCGScopeInvalid = 0
    cudaCGScopeGrid = 1
    cudaCGScopeMultiGrid = 2

cdef struct cudaKernelNodeParams:
    void* func
    dim3 gridDim
    dim3 blockDim
    unsigned int sharedMemBytes
    void** kernelParams
    void** extra

cdef struct cudaExternalSemaphoreSignalNodeParams:
    cudaExternalSemaphore_t* extSemArray
    const cudaExternalSemaphoreSignalParams* paramsArray
    unsigned int numExtSems

cdef struct cudaExternalSemaphoreWaitNodeParams:
    cudaExternalSemaphore_t* extSemArray
    const cudaExternalSemaphoreWaitParams* paramsArray
    unsigned int numExtSems

cdef enum cudaGraphNodeType:
    cudaGraphNodeTypeKernel = 0x00
    cudaGraphNodeTypeMemcpy = 0x01
    cudaGraphNodeTypeMemset = 0x02
    cudaGraphNodeTypeHost = 0x03
    cudaGraphNodeTypeGraph = 0x04
    cudaGraphNodeTypeEmpty = 0x05
    cudaGraphNodeTypeWaitEvent = 0x06
    cudaGraphNodeTypeEventRecord = 0x07
    cudaGraphNodeTypeExtSemaphoreSignal = 0x08
    cudaGraphNodeTypeExtSemaphoreWait = 0x09
    cudaGraphNodeTypeMemAlloc = 0x0a
    cudaGraphNodeTypeMemFree = 0x0b
    cudaGraphNodeTypeCount

cdef extern from "":
    cdef struct CUgraphExec_st:
        pass
ctypedef CUgraphExec_st* cudaGraphExec_t

cdef enum cudaGraphExecUpdateResult:
    cudaGraphExecUpdateSuccess = 0x0
    cudaGraphExecUpdateError = 0x1
    cudaGraphExecUpdateErrorTopologyChanged = 0x2
    cudaGraphExecUpdateErrorNodeTypeChanged = 0x3
    cudaGraphExecUpdateErrorFunctionChanged = 0x4
    cudaGraphExecUpdateErrorParametersChanged = 0x5
    cudaGraphExecUpdateErrorNotSupported = 0x6
    cudaGraphExecUpdateErrorUnsupportedFunctionChange = 0x7

cdef enum cudaGetDriverEntryPointFlags:
    cudaEnableDefault = 0x0
    cudaEnableLegacyStream = 0x1
    cudaEnablePerThreadDefaultStream = 0x2

cdef enum cudaGraphDebugDotFlags:
    cudaGraphDebugDotFlagsVerbose = 1<<0
    cudaGraphDebugDotFlagsKernelNodeParams = 1<<2
    cudaGraphDebugDotFlagsMemcpyNodeParams = 1<<3
    cudaGraphDebugDotFlagsMemsetNodeParams = 1<<4
    cudaGraphDebugDotFlagsHostNodeParams = 1<<5
    cudaGraphDebugDotFlagsEventNodeParams = 1<<6
    cudaGraphDebugDotFlagsExtSemasSignalNodeParams = 1<<7
    cudaGraphDebugDotFlagsExtSemasWaitNodeParams = 1<<8
    cudaGraphDebugDotFlagsKernelNodeAttributes = 1<<9
    cudaGraphDebugDotFlagsHandles = 1<<10

cdef enum cudaGraphInstantiateFlags:
    cudaGraphInstantiateFlagAutoFreeOnLaunch = 1

cdef enum cudaSurfaceBoundaryMode:
    cudaBoundaryModeZero = 0
    cudaBoundaryModeClamp = 1
    cudaBoundaryModeTrap = 2

cdef enum cudaSurfaceFormatMode:
    cudaFormatModeForced = 0
    cudaFormatModeAuto = 1

ctypedef unsigned long long cudaSurfaceObject_t

cdef enum cudaTextureAddressMode:
    cudaAddressModeWrap = 0
    cudaAddressModeClamp = 1
    cudaAddressModeMirror = 2
    cudaAddressModeBorder = 3

cdef enum cudaTextureFilterMode:
    cudaFilterModePoint = 0
    cudaFilterModeLinear = 1

cdef enum cudaTextureReadMode:
    cudaReadModeElementType = 0
    cudaReadModeNormalizedFloat = 1

cdef struct cudaTextureDesc:
    cudaTextureAddressMode addressMode[3]
    cudaTextureFilterMode filterMode
    cudaTextureReadMode readMode
    int sRGB
    float borderColor[4]
    int normalizedCoords
    unsigned int maxAnisotropy
    cudaTextureFilterMode mipmapFilterMode
    float mipmapLevelBias
    float minMipmapLevelClamp
    float maxMipmapLevelClamp
    int disableTrilinearOptimization

ctypedef unsigned long long cudaTextureObject_t

cdef enum cudaDataType_t:
    CUDA_R_16F = 2
    CUDA_C_16F = 6
    CUDA_R_16BF = 14
    CUDA_C_16BF = 15
    CUDA_R_32F = 0
    CUDA_C_32F = 4
    CUDA_R_64F = 1
    CUDA_C_64F = 5
    CUDA_R_4I = 16
    CUDA_C_4I = 17
    CUDA_R_4U = 18
    CUDA_C_4U = 19
    CUDA_R_8I = 3
    CUDA_C_8I = 7
    CUDA_R_8U = 8
    CUDA_C_8U = 9
    CUDA_R_16I = 20
    CUDA_C_16I = 21
    CUDA_R_16U = 22
    CUDA_C_16U = 23
    CUDA_R_32I = 10
    CUDA_C_32I = 11
    CUDA_R_32U = 12
    CUDA_C_32U = 13
    CUDA_R_64I = 24
    CUDA_C_64I = 25
    CUDA_R_64U = 26
    CUDA_C_64U = 27

ctypedef cudaDataType_t cudaDataType

cdef enum libraryPropertyType_t:
    MAJOR_VERSION
    MINOR_VERSION
    PATCH_LEVEL

ctypedef libraryPropertyType_t libraryPropertyType

cdef cudaError_t cudaDeviceReset() nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaDeviceSynchronize() nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaDeviceSetLimit(cudaLimit limit, size_t value) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaDeviceGetLimit(size_t* pValue, cudaLimit limit) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaDeviceGetTexture1DLinearMaxWidth(size_t* maxWidthInElements, const cudaChannelFormatDesc* fmtDesc, int device) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaDeviceGetCacheConfig(cudaFuncCache* pCacheConfig) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaDeviceGetStreamPriorityRange(int* leastPriority, int* greatestPriority) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaDeviceSetCacheConfig(cudaFuncCache cacheConfig) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaDeviceGetSharedMemConfig(cudaSharedMemConfig* pConfig) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaDeviceSetSharedMemConfig(cudaSharedMemConfig config) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaDeviceGetByPCIBusId(int* device, const char* pciBusId) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaDeviceGetPCIBusId(char* pciBusId, int length, int device) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaIpcGetEventHandle(cudaIpcEventHandle_t* handle, cudaEvent_t event) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaIpcOpenEventHandle(cudaEvent_t* event, cudaIpcEventHandle_t handle) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaIpcGetMemHandle(cudaIpcMemHandle_t* handle, void* devPtr) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaIpcOpenMemHandle(void** devPtr, cudaIpcMemHandle_t handle, unsigned int flags) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaIpcCloseMemHandle(void* devPtr) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaDeviceFlushGPUDirectRDMAWrites(cudaFlushGPUDirectRDMAWritesTarget target, cudaFlushGPUDirectRDMAWritesScope scope) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaThreadExit() nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaThreadSynchronize() nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaThreadSetLimit(cudaLimit limit, size_t value) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaThreadGetLimit(size_t* pValue, cudaLimit limit) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaThreadGetCacheConfig(cudaFuncCache* pCacheConfig) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaThreadSetCacheConfig(cudaFuncCache cacheConfig) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaGetLastError() nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaPeekAtLastError() nogil except ?cudaErrorCallRequiresNewerDriver

cdef const char* cudaGetErrorName(cudaError_t error) nogil except ?NULL

cdef const char* cudaGetErrorString(cudaError_t error) nogil except ?NULL

cdef cudaError_t cudaGetDeviceCount(int* count) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaGetDeviceProperties(cudaDeviceProp* prop, int device) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaDeviceGetAttribute(int* value, cudaDeviceAttr attr, int device) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaDeviceGetDefaultMemPool(cudaMemPool_t* memPool, int device) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaDeviceSetMemPool(int device, cudaMemPool_t memPool) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaDeviceGetMemPool(cudaMemPool_t* memPool, int device) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaDeviceGetNvSciSyncAttributes(void* nvSciSyncAttrList, int device, int flags) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaDeviceGetP2PAttribute(int* value, cudaDeviceP2PAttr attr, int srcDevice, int dstDevice) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaChooseDevice(int* device, const cudaDeviceProp* prop) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaSetDevice(int device) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaGetDevice(int* device) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaSetDeviceFlags(unsigned int flags) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaGetDeviceFlags(unsigned int* flags) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaStreamCreate(cudaStream_t* pStream) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaStreamCreateWithFlags(cudaStream_t* pStream, unsigned int flags) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaStreamCreateWithPriority(cudaStream_t* pStream, unsigned int flags, int priority) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaStreamGetPriority(cudaStream_t hStream, int* priority) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaStreamGetFlags(cudaStream_t hStream, unsigned int* flags) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaCtxResetPersistingL2Cache() nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaStreamCopyAttributes(cudaStream_t dst, cudaStream_t src) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaStreamGetAttribute(cudaStream_t hStream, cudaStreamAttrID attr, cudaStreamAttrValue* value_out) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaStreamSetAttribute(cudaStream_t hStream, cudaStreamAttrID attr, const cudaStreamAttrValue* value) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaStreamDestroy(cudaStream_t stream) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned int flags) nogil except ?cudaErrorCallRequiresNewerDriver

ctypedef void (*cudaStreamCallback_t)(cudaStream_t stream, cudaError_t status, void* userData)

cdef cudaError_t cudaStreamAddCallback(cudaStream_t stream, cudaStreamCallback_t callback, void* userData, unsigned int flags) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaStreamSynchronize(cudaStream_t stream) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaStreamQuery(cudaStream_t stream) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaStreamAttachMemAsync(cudaStream_t stream, void* devPtr, size_t length, unsigned int flags) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaStreamBeginCapture(cudaStream_t stream, cudaStreamCaptureMode mode) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaThreadExchangeStreamCaptureMode(cudaStreamCaptureMode* mode) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaStreamEndCapture(cudaStream_t stream, cudaGraph_t* pGraph) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaStreamIsCapturing(cudaStream_t stream, cudaStreamCaptureStatus* pCaptureStatus) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaStreamGetCaptureInfo(cudaStream_t stream, cudaStreamCaptureStatus* pCaptureStatus, unsigned long long* pId) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaStreamGetCaptureInfo_v2(cudaStream_t stream, cudaStreamCaptureStatus* captureStatus_out, unsigned long long* id_out, cudaGraph_t* graph_out, const cudaGraphNode_t** dependencies_out, size_t* numDependencies_out) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaStreamUpdateCaptureDependencies(cudaStream_t stream, cudaGraphNode_t* dependencies, size_t numDependencies, unsigned int flags) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaEventCreate(cudaEvent_t* event) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaEventCreateWithFlags(cudaEvent_t* event, unsigned int flags) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaEventRecordWithFlags(cudaEvent_t event, cudaStream_t stream, unsigned int flags) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaEventQuery(cudaEvent_t event) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaEventSynchronize(cudaEvent_t event) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaEventDestroy(cudaEvent_t event) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaEventElapsedTime(float* ms, cudaEvent_t start, cudaEvent_t end) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaImportExternalMemory(cudaExternalMemory_t* extMem_out, const cudaExternalMemoryHandleDesc* memHandleDesc) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaExternalMemoryGetMappedBuffer(void** devPtr, cudaExternalMemory_t extMem, const cudaExternalMemoryBufferDesc* bufferDesc) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaExternalMemoryGetMappedMipmappedArray(cudaMipmappedArray_t* mipmap, cudaExternalMemory_t extMem, const cudaExternalMemoryMipmappedArrayDesc* mipmapDesc) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaDestroyExternalMemory(cudaExternalMemory_t extMem) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaImportExternalSemaphore(cudaExternalSemaphore_t* extSem_out, const cudaExternalSemaphoreHandleDesc* semHandleDesc) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaSignalExternalSemaphoresAsync(const cudaExternalSemaphore_t* extSemArray, const cudaExternalSemaphoreSignalParams* paramsArray, unsigned int numExtSems, cudaStream_t stream) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaWaitExternalSemaphoresAsync(const cudaExternalSemaphore_t* extSemArray, const cudaExternalSemaphoreWaitParams* paramsArray, unsigned int numExtSems, cudaStream_t stream) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaDestroyExternalSemaphore(cudaExternalSemaphore_t extSem) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaFuncSetCacheConfig(const void* func, cudaFuncCache cacheConfig) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaFuncSetSharedMemConfig(const void* func, cudaSharedMemConfig config) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaFuncGetAttributes(cudaFuncAttributes* attr, const void* func) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaFuncSetAttribute(const void* func, cudaFuncAttribute attr, int value) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaSetDoubleForDevice(double* d) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaSetDoubleForHost(double* d) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaLaunchHostFunc(cudaStream_t stream, cudaHostFn_t fn, void* userData) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(int* numBlocks, const void* func, int blockSize, size_t dynamicSMemSize) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaOccupancyAvailableDynamicSMemPerBlock(size_t* dynamicSmemSize, const void* func, int numBlocks, int blockSize) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int* numBlocks, const void* func, int blockSize, size_t dynamicSMemSize, unsigned int flags) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaMallocManaged(void** devPtr, size_t size, unsigned int flags) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaMalloc(void** devPtr, size_t size) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaMallocHost(void** ptr, size_t size) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaMallocPitch(void** devPtr, size_t* pitch, size_t width, size_t height) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaMallocArray(cudaArray_t* array, const cudaChannelFormatDesc* desc, size_t width, size_t height, unsigned int flags) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaFree(void* devPtr) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaFreeHost(void* ptr) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaFreeArray(cudaArray_t array) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaFreeMipmappedArray(cudaMipmappedArray_t mipmappedArray) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaHostAlloc(void** pHost, size_t size, unsigned int flags) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaHostRegister(void* ptr, size_t size, unsigned int flags) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaHostUnregister(void* ptr) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaHostGetDevicePointer(void** pDevice, void* pHost, unsigned int flags) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaHostGetFlags(unsigned int* pFlags, void* pHost) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaMalloc3D(cudaPitchedPtr* pitchedDevPtr, cudaExtent extent) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaMalloc3DArray(cudaArray_t* array, const cudaChannelFormatDesc* desc, cudaExtent extent, unsigned int flags) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaMallocMipmappedArray(cudaMipmappedArray_t* mipmappedArray, const cudaChannelFormatDesc* desc, cudaExtent extent, unsigned int numLevels, unsigned int flags) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaGetMipmappedArrayLevel(cudaArray_t* levelArray, cudaMipmappedArray_const_t mipmappedArray, unsigned int level) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaMemcpy3D(const cudaMemcpy3DParms* p) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaMemcpy3DPeer(const cudaMemcpy3DPeerParms* p) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaMemcpy3DAsync(const cudaMemcpy3DParms* p, cudaStream_t stream) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaMemcpy3DPeerAsync(const cudaMemcpy3DPeerParms* p, cudaStream_t stream) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaMemGetInfo(size_t* free, size_t* total) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaArrayGetInfo(cudaChannelFormatDesc* desc, cudaExtent* extent, unsigned int* flags, cudaArray_t array) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaArrayGetPlane(cudaArray_t* pPlaneArray, cudaArray_t hArray, unsigned int planeIdx) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaArrayGetSparseProperties(cudaArraySparseProperties* sparseProperties, cudaArray_t array) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaMipmappedArrayGetSparseProperties(cudaArraySparseProperties* sparseProperties, cudaMipmappedArray_t mipmap) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaMemcpyPeer(void* dst, int dstDevice, const void* src, int srcDevice, size_t count) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaMemcpy2D(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaMemcpy2DToArray(cudaArray_t dst, size_t wOffset, size_t hOffset, const void* src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaMemcpy2DFromArray(void* dst, size_t dpitch, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t width, size_t height, cudaMemcpyKind kind) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaMemcpy2DArrayToArray(cudaArray_t dst, size_t wOffsetDst, size_t hOffsetDst, cudaArray_const_t src, size_t wOffsetSrc, size_t hOffsetSrc, size_t width, size_t height, cudaMemcpyKind kind) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t count, cudaMemcpyKind kind, cudaStream_t stream) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaMemcpyPeerAsync(void* dst, int dstDevice, const void* src, int srcDevice, size_t count, cudaStream_t stream) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaMemcpy2DAsync(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaMemcpy2DToArrayAsync(cudaArray_t dst, size_t wOffset, size_t hOffset, const void* src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaMemcpy2DFromArrayAsync(void* dst, size_t dpitch, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaMemset(void* devPtr, int value, size_t count) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaMemset2D(void* devPtr, size_t pitch, int value, size_t width, size_t height) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaMemset3D(cudaPitchedPtr pitchedDevPtr, int value, cudaExtent extent) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaMemsetAsync(void* devPtr, int value, size_t count, cudaStream_t stream) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaMemset2DAsync(void* devPtr, size_t pitch, int value, size_t width, size_t height, cudaStream_t stream) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaMemset3DAsync(cudaPitchedPtr pitchedDevPtr, int value, cudaExtent extent, cudaStream_t stream) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaMemPrefetchAsync(const void* devPtr, size_t count, int dstDevice, cudaStream_t stream) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaMemAdvise(const void* devPtr, size_t count, cudaMemoryAdvise advice, int device) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaMemRangeGetAttribute(void* data, size_t dataSize, cudaMemRangeAttribute attribute, const void* devPtr, size_t count) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaMemRangeGetAttributes(void** data, size_t* dataSizes, cudaMemRangeAttribute* attributes, size_t numAttributes, const void* devPtr, size_t count) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaMemcpyToArray(cudaArray_t dst, size_t wOffset, size_t hOffset, const void* src, size_t count, cudaMemcpyKind kind) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaMemcpyFromArray(void* dst, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t count, cudaMemcpyKind kind) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaMemcpyArrayToArray(cudaArray_t dst, size_t wOffsetDst, size_t hOffsetDst, cudaArray_const_t src, size_t wOffsetSrc, size_t hOffsetSrc, size_t count, cudaMemcpyKind kind) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaMemcpyToArrayAsync(cudaArray_t dst, size_t wOffset, size_t hOffset, const void* src, size_t count, cudaMemcpyKind kind, cudaStream_t stream) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaMemcpyFromArrayAsync(void* dst, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t count, cudaMemcpyKind kind, cudaStream_t stream) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaMallocAsync(void** devPtr, size_t size, cudaStream_t hStream) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaFreeAsync(void* devPtr, cudaStream_t hStream) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaMemPoolTrimTo(cudaMemPool_t memPool, size_t minBytesToKeep) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaMemPoolSetAttribute(cudaMemPool_t memPool, cudaMemPoolAttr attr, void* value) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaMemPoolGetAttribute(cudaMemPool_t memPool, cudaMemPoolAttr attr, void* value) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaMemPoolSetAccess(cudaMemPool_t memPool, const cudaMemAccessDesc* descList, size_t count) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaMemPoolGetAccess(cudaMemAccessFlags* flags, cudaMemPool_t memPool, cudaMemLocation* location) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaMemPoolCreate(cudaMemPool_t* memPool, const cudaMemPoolProps* poolProps) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaMemPoolDestroy(cudaMemPool_t memPool) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaMallocFromPoolAsync(void** ptr, size_t size, cudaMemPool_t memPool, cudaStream_t stream) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaMemPoolExportToShareableHandle(void* shareableHandle, cudaMemPool_t memPool, cudaMemAllocationHandleType handleType, unsigned int flags) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaMemPoolImportFromShareableHandle(cudaMemPool_t* memPool, void* shareableHandle, cudaMemAllocationHandleType handleType, unsigned int flags) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaMemPoolExportPointer(cudaMemPoolPtrExportData* exportData, void* ptr) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaMemPoolImportPointer(void** ptr, cudaMemPool_t memPool, cudaMemPoolPtrExportData* exportData) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaPointerGetAttributes(cudaPointerAttributes* attributes, const void* ptr) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaDeviceCanAccessPeer(int* canAccessPeer, int device, int peerDevice) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaDeviceEnablePeerAccess(int peerDevice, unsigned int flags) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaDeviceDisablePeerAccess(int peerDevice) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaGraphicsUnregisterResource(cudaGraphicsResource_t resource) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaGraphicsResourceSetMapFlags(cudaGraphicsResource_t resource, unsigned int flags) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaGraphicsMapResources(int count, cudaGraphicsResource_t* resources, cudaStream_t stream) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaGraphicsUnmapResources(int count, cudaGraphicsResource_t* resources, cudaStream_t stream) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaGraphicsResourceGetMappedPointer(void** devPtr, size_t* size, cudaGraphicsResource_t resource) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaGraphicsSubResourceGetMappedArray(cudaArray_t* array, cudaGraphicsResource_t resource, unsigned int arrayIndex, unsigned int mipLevel) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaGraphicsResourceGetMappedMipmappedArray(cudaMipmappedArray_t* mipmappedArray, cudaGraphicsResource_t resource) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaGetChannelDesc(cudaChannelFormatDesc* desc, cudaArray_const_t array) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaChannelFormatDesc cudaCreateChannelDesc(int x, int y, int z, int w, cudaChannelFormatKind f) nogil

cdef cudaError_t cudaCreateTextureObject(cudaTextureObject_t* pTexObject, const cudaResourceDesc* pResDesc, const cudaTextureDesc* pTexDesc, const cudaResourceViewDesc* pResViewDesc) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaDestroyTextureObject(cudaTextureObject_t texObject) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaGetTextureObjectResourceDesc(cudaResourceDesc* pResDesc, cudaTextureObject_t texObject) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaGetTextureObjectTextureDesc(cudaTextureDesc* pTexDesc, cudaTextureObject_t texObject) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaGetTextureObjectResourceViewDesc(cudaResourceViewDesc* pResViewDesc, cudaTextureObject_t texObject) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaCreateSurfaceObject(cudaSurfaceObject_t* pSurfObject, const cudaResourceDesc* pResDesc) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaDestroySurfaceObject(cudaSurfaceObject_t surfObject) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaGetSurfaceObjectResourceDesc(cudaResourceDesc* pResDesc, cudaSurfaceObject_t surfObject) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaDriverGetVersion(int* driverVersion) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaRuntimeGetVersion(int* runtimeVersion) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaGraphCreate(cudaGraph_t* pGraph, unsigned int flags) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaGraphAddKernelNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, const cudaKernelNodeParams* pNodeParams) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaGraphKernelNodeGetParams(cudaGraphNode_t node, cudaKernelNodeParams* pNodeParams) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaGraphKernelNodeSetParams(cudaGraphNode_t node, const cudaKernelNodeParams* pNodeParams) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaGraphKernelNodeCopyAttributes(cudaGraphNode_t hSrc, cudaGraphNode_t hDst) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaGraphKernelNodeGetAttribute(cudaGraphNode_t hNode, cudaKernelNodeAttrID attr, cudaKernelNodeAttrValue* value_out) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaGraphKernelNodeSetAttribute(cudaGraphNode_t hNode, cudaKernelNodeAttrID attr, const cudaKernelNodeAttrValue* value) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaGraphAddMemcpyNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, const cudaMemcpy3DParms* pCopyParams) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaGraphAddMemcpyNode1D(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, void* dst, const void* src, size_t count, cudaMemcpyKind kind) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaGraphMemcpyNodeGetParams(cudaGraphNode_t node, cudaMemcpy3DParms* pNodeParams) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaGraphMemcpyNodeSetParams(cudaGraphNode_t node, const cudaMemcpy3DParms* pNodeParams) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaGraphMemcpyNodeSetParams1D(cudaGraphNode_t node, void* dst, const void* src, size_t count, cudaMemcpyKind kind) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaGraphAddMemsetNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, const cudaMemsetParams* pMemsetParams) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaGraphMemsetNodeGetParams(cudaGraphNode_t node, cudaMemsetParams* pNodeParams) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaGraphMemsetNodeSetParams(cudaGraphNode_t node, const cudaMemsetParams* pNodeParams) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaGraphAddHostNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, const cudaHostNodeParams* pNodeParams) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaGraphHostNodeGetParams(cudaGraphNode_t node, cudaHostNodeParams* pNodeParams) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaGraphHostNodeSetParams(cudaGraphNode_t node, const cudaHostNodeParams* pNodeParams) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaGraphAddChildGraphNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, cudaGraph_t childGraph) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaGraphChildGraphNodeGetGraph(cudaGraphNode_t node, cudaGraph_t* pGraph) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaGraphAddEmptyNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaGraphAddEventRecordNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, cudaEvent_t event) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaGraphEventRecordNodeGetEvent(cudaGraphNode_t node, cudaEvent_t* event_out) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaGraphEventRecordNodeSetEvent(cudaGraphNode_t node, cudaEvent_t event) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaGraphAddEventWaitNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, cudaEvent_t event) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaGraphEventWaitNodeGetEvent(cudaGraphNode_t node, cudaEvent_t* event_out) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaGraphEventWaitNodeSetEvent(cudaGraphNode_t node, cudaEvent_t event) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaGraphAddExternalSemaphoresSignalNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, const cudaExternalSemaphoreSignalNodeParams* nodeParams) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaGraphExternalSemaphoresSignalNodeGetParams(cudaGraphNode_t hNode, cudaExternalSemaphoreSignalNodeParams* params_out) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaGraphExternalSemaphoresSignalNodeSetParams(cudaGraphNode_t hNode, const cudaExternalSemaphoreSignalNodeParams* nodeParams) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaGraphAddExternalSemaphoresWaitNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, const cudaExternalSemaphoreWaitNodeParams* nodeParams) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaGraphExternalSemaphoresWaitNodeGetParams(cudaGraphNode_t hNode, cudaExternalSemaphoreWaitNodeParams* params_out) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaGraphExternalSemaphoresWaitNodeSetParams(cudaGraphNode_t hNode, const cudaExternalSemaphoreWaitNodeParams* nodeParams) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaGraphAddMemAllocNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, cudaMemAllocNodeParams* nodeParams) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaGraphMemAllocNodeGetParams(cudaGraphNode_t node, cudaMemAllocNodeParams* params_out) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaGraphAddMemFreeNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, void* dptr) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaGraphMemFreeNodeGetParams(cudaGraphNode_t node, void* dptr_out) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaDeviceGraphMemTrim(int device) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaDeviceGetGraphMemAttribute(int device, cudaGraphMemAttributeType attr, void* value) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaDeviceSetGraphMemAttribute(int device, cudaGraphMemAttributeType attr, void* value) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaGraphClone(cudaGraph_t* pGraphClone, cudaGraph_t originalGraph) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaGraphNodeFindInClone(cudaGraphNode_t* pNode, cudaGraphNode_t originalNode, cudaGraph_t clonedGraph) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaGraphNodeGetType(cudaGraphNode_t node, cudaGraphNodeType* pType) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaGraphGetNodes(cudaGraph_t graph, cudaGraphNode_t* nodes, size_t* numNodes) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaGraphGetRootNodes(cudaGraph_t graph, cudaGraphNode_t* pRootNodes, size_t* pNumRootNodes) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaGraphGetEdges(cudaGraph_t graph, cudaGraphNode_t* from_, cudaGraphNode_t* to, size_t* numEdges) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaGraphNodeGetDependencies(cudaGraphNode_t node, cudaGraphNode_t* pDependencies, size_t* pNumDependencies) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaGraphNodeGetDependentNodes(cudaGraphNode_t node, cudaGraphNode_t* pDependentNodes, size_t* pNumDependentNodes) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaGraphAddDependencies(cudaGraph_t graph, const cudaGraphNode_t* from_, const cudaGraphNode_t* to, size_t numDependencies) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaGraphRemoveDependencies(cudaGraph_t graph, const cudaGraphNode_t* from_, const cudaGraphNode_t* to, size_t numDependencies) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaGraphDestroyNode(cudaGraphNode_t node) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaGraphInstantiate(cudaGraphExec_t* pGraphExec, cudaGraph_t graph, cudaGraphNode_t* pErrorNode, char* pLogBuffer, size_t bufferSize) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaGraphInstantiateWithFlags(cudaGraphExec_t* pGraphExec, cudaGraph_t graph, unsigned long long flags) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaGraphExecKernelNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, const cudaKernelNodeParams* pNodeParams) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaGraphExecMemcpyNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, const cudaMemcpy3DParms* pNodeParams) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaGraphExecMemcpyNodeSetParams1D(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, void* dst, const void* src, size_t count, cudaMemcpyKind kind) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaGraphExecMemsetNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, const cudaMemsetParams* pNodeParams) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaGraphExecHostNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, const cudaHostNodeParams* pNodeParams) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaGraphExecChildGraphNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, cudaGraph_t childGraph) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaGraphExecEventRecordNodeSetEvent(cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, cudaEvent_t event) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaGraphExecEventWaitNodeSetEvent(cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, cudaEvent_t event) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaGraphExecExternalSemaphoresSignalNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, const cudaExternalSemaphoreSignalNodeParams* nodeParams) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaGraphExecExternalSemaphoresWaitNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, const cudaExternalSemaphoreWaitNodeParams* nodeParams) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaGraphExecUpdate(cudaGraphExec_t hGraphExec, cudaGraph_t hGraph, cudaGraphNode_t* hErrorNode_out, cudaGraphExecUpdateResult* updateResult_out) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaGraphUpload(cudaGraphExec_t graphExec, cudaStream_t stream) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaGraphLaunch(cudaGraphExec_t graphExec, cudaStream_t stream) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaGraphExecDestroy(cudaGraphExec_t graphExec) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaGraphDestroy(cudaGraph_t graph) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaGraphDebugDotPrint(cudaGraph_t graph, const char* path, unsigned int flags) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaUserObjectCreate(cudaUserObject_t* object_out, void* ptr, cudaHostFn_t destroy, unsigned int initialRefcount, unsigned int flags) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaUserObjectRetain(cudaUserObject_t object, unsigned int count) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaUserObjectRelease(cudaUserObject_t object, unsigned int count) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaGraphRetainUserObject(cudaGraph_t graph, cudaUserObject_t object, unsigned int count, unsigned int flags) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaGraphReleaseUserObject(cudaGraph_t graph, cudaUserObject_t object, unsigned int count) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaGetDriverEntryPoint(const char* symbol, void** funcPtr, unsigned long long flags) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaError_t cudaGetExportTable(const void** ppExportTable, const cudaUUID_t* pExportTableId) nogil except ?cudaErrorCallRequiresNewerDriver

cdef cudaPitchedPtr make_cudaPitchedPtr(void* d, size_t p, size_t xsz, size_t ysz) nogil

cdef cudaPos make_cudaPos(size_t x, size_t y, size_t z) nogil

cdef cudaExtent make_cudaExtent(size_t w, size_t h, size_t d) nogil

cdef enum: cudaHostAllocDefault = 0x00

cdef enum: cudaHostAllocPortable = 0x01

cdef enum: cudaHostAllocMapped = 0x02

cdef enum: cudaHostAllocWriteCombined = 0x04

cdef enum: cudaHostRegisterDefault = 0x00

cdef enum: cudaHostRegisterPortable = 0x01

cdef enum: cudaHostRegisterMapped = 0x02

cdef enum: cudaHostRegisterIoMemory = 0x04

cdef enum: cudaHostRegisterReadOnly = 0x08

cdef enum: cudaPeerAccessDefault = 0x00

cdef enum: cudaStreamDefault = 0x00

cdef enum: cudaStreamNonBlocking = 0x01

cdef enum: cudaStreamLegacy = 0x1

cdef enum: cudaStreamPerThread = 0x2

cdef enum: cudaEventDefault = 0x00

cdef enum: cudaEventBlockingSync = 0x01

cdef enum: cudaEventDisableTiming = 0x02

cdef enum: cudaEventInterprocess = 0x04

cdef enum: cudaEventRecordDefault = 0x00

cdef enum: cudaEventRecordExternal = 0x01

cdef enum: cudaEventWaitDefault = 0x00

cdef enum: cudaEventWaitExternal = 0x01

cdef enum: cudaDeviceScheduleAuto = 0x00

cdef enum: cudaDeviceScheduleSpin = 0x01

cdef enum: cudaDeviceScheduleYield = 0x02

cdef enum: cudaDeviceScheduleBlockingSync = 0x04

cdef enum: cudaDeviceBlockingSync = 0x04

cdef enum: cudaDeviceScheduleMask = 0x07

cdef enum: cudaDeviceMapHost = 0x08

cdef enum: cudaDeviceLmemResizeToMax = 0x10

cdef enum: cudaDeviceMask = 0x1f

cdef enum: cudaArrayDefault = 0x00

cdef enum: cudaArrayLayered = 0x01

cdef enum: cudaArraySurfaceLoadStore = 0x02

cdef enum: cudaArrayCubemap = 0x04

cdef enum: cudaArrayTextureGather = 0x08

cdef enum: cudaArrayColorAttachment = 0x20

cdef enum: cudaArraySparse = 0x40

cdef enum: cudaIpcMemLazyEnablePeerAccess = 0x01

cdef enum: cudaMemAttachGlobal = 0x01

cdef enum: cudaMemAttachHost = 0x02

cdef enum: cudaMemAttachSingle = 0x04

cdef enum: cudaOccupancyDefault = 0x00

cdef enum: cudaOccupancyDisableCachingOverride = 0x01

cdef enum: cudaCpuDeviceId = -1

cdef enum: cudaInvalidDeviceId = -2

cdef enum: cudaCooperativeLaunchMultiDeviceNoPreSync = 0x01

cdef enum: cudaCooperativeLaunchMultiDeviceNoPostSync = 0x02

cdef enum: cudaArraySparsePropertiesSingleMipTail = 0x1

cdef enum: CUDA_IPC_HANDLE_SIZE = 64

cdef enum: cudaExternalMemoryDedicated = 0x1

cdef enum: cudaExternalSemaphoreSignalSkipNvSciBufMemSync = 0x01

cdef enum: cudaExternalSemaphoreWaitSkipNvSciBufMemSync = 0x02

cdef enum: cudaNvSciSyncAttrSignal = 0x1

cdef enum: cudaNvSciSyncAttrWait = 0x2

cdef enum: cudaSurfaceType1D = 0x01

cdef enum: cudaSurfaceType2D = 0x02

cdef enum: cudaSurfaceType3D = 0x03

cdef enum: cudaSurfaceTypeCubemap = 0x0C

cdef enum: cudaSurfaceType1DLayered = 0xF1

cdef enum: cudaSurfaceType2DLayered = 0xF2

cdef enum: cudaSurfaceTypeCubemapLayered = 0xFC

cdef enum: cudaTextureType1D = 0x01

cdef enum: cudaTextureType2D = 0x02

cdef enum: cudaTextureType3D = 0x03

cdef enum: cudaTextureTypeCubemap = 0x0C

cdef enum: cudaTextureType1DLayered = 0xF1

cdef enum: cudaTextureType2DLayered = 0xF2

cdef enum: cudaTextureTypeCubemapLayered = 0xFC

cdef enum: CUDART_VERSION = 11050
