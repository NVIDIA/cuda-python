cdef extern from "cuda_runtime.h" nogil:

    cdef enum cudaRoundMode:
        cudaRoundNearest	"cudaRoundNearest"
        cudaRoundZero	"cudaRoundZero"
        cudaRoundPosInf	"cudaRoundPosInf"
        cudaRoundMinInf	"cudaRoundMinInf"

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
        cudaErrorUnknown = 999
        cudaErrorApiFailureBase = 10000

    cdef enum cudaChannelFormatKind:
        cudaChannelFormatKindSigned = 0
        cudaChannelFormatKindUnsigned = 1
        cudaChannelFormatKindFloat = 2
        cudaChannelFormatKindNone = 3
        cudaChannelFormatKindNV12 = 4

    cdef struct cudaChannelFormatDesc:
        int x
        int y
        int z
        int w
        cudaChannelFormatKind f

    ctypedef struct cudaArray:
        pass
    ctypedef cudaArray* cudaArray_t

    ctypedef struct cudaArray:
        pass
    ctypedef cudaArray* cudaArray_const_t

    ctypedef struct cudaArray:
        pass

    ctypedef struct cudaMipmappedArray:
        pass
    ctypedef cudaMipmappedArray* cudaMipmappedArray_t

    ctypedef struct cudaMipmappedArray:
        pass
    ctypedef cudaMipmappedArray* cudaMipmappedArray_const_t

    ctypedef struct cudaMipmappedArray:
        pass

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

    ctypedef struct cudaGraphicsResource:
        pass

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
        cudaFuncAttributeMax	"cudaFuncAttributeMax"

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
        cudaFlushGPUDirectRDMAWritesTargetCurrentDevice	"cudaFlushGPUDirectRDMAWritesTargetCurrentDevice"

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
        cudaDevAttrMaxTimelineSemaphoreInteropSupported = 114
        cudaDevAttrMemoryPoolsSupported = 115
        cudaDevAttrGPUDirectRDMASupported = 116
        cudaDevAttrGPUDirectRDMAFlushWritesOptions = 117
        cudaDevAttrGPUDirectRDMAWritesOrdering = 118
        cudaDevAttrMemoryPoolSupportedHandleTypes = 119
        cudaDevAttrMax	"cudaDevAttrMax"

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
        cudaMemAccessDesc* accessDescs
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

    ctypedef struct CUstream_st:
        pass
    ctypedef CUstream_st* cudaStream_t

    ctypedef struct CUevent_st:
        pass
    ctypedef CUevent_st* cudaEvent_t

    ctypedef struct cudaGraphicsResource:
        pass
    ctypedef cudaGraphicsResource* cudaGraphicsResource_t

    ctypedef cudaOutputMode cudaOutputMode_t

    ctypedef struct CUexternalMemory_st:
        pass
    ctypedef CUexternalMemory_st* cudaExternalMemory_t

    ctypedef struct CUexternalSemaphore_st:
        pass
    ctypedef CUexternalSemaphore_st* cudaExternalSemaphore_t

    ctypedef struct CUgraph_st:
        pass
    ctypedef CUgraph_st* cudaGraph_t

    ctypedef struct CUgraphNode_st:
        pass
    ctypedef CUgraphNode_st* cudaGraphNode_t

    ctypedef struct CUuserObject_st:
        pass
    ctypedef CUuserObject_st* cudaUserObject_t

    ctypedef struct CUfunc_st:
        pass
    ctypedef CUfunc_st* cudaFunction_t

    ctypedef struct CUmemPoolHandle_st:
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
        cudaExternalSemaphoreSignalParams* paramsArray
        unsigned int numExtSems

    cdef struct cudaExternalSemaphoreWaitNodeParams:
        cudaExternalSemaphore_t* extSemArray
        cudaExternalSemaphoreWaitParams* paramsArray
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
        cudaGraphNodeTypeCount	"cudaGraphNodeTypeCount"

    ctypedef struct CUgraphExec_st:
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
        CUDA_R_16F	"CUDA_R_16F"
        CUDA_C_16F	"CUDA_C_16F"
        CUDA_R_16BF	"CUDA_R_16BF"
        CUDA_C_16BF	"CUDA_C_16BF"
        CUDA_R_32F	"CUDA_R_32F"
        CUDA_C_32F	"CUDA_C_32F"
        CUDA_R_64F	"CUDA_R_64F"
        CUDA_C_64F	"CUDA_C_64F"
        CUDA_R_4I	"CUDA_R_4I"
        CUDA_C_4I	"CUDA_C_4I"
        CUDA_R_4U	"CUDA_R_4U"
        CUDA_C_4U	"CUDA_C_4U"
        CUDA_R_8I	"CUDA_R_8I"
        CUDA_C_8I	"CUDA_C_8I"
        CUDA_R_8U	"CUDA_R_8U"
        CUDA_C_8U	"CUDA_C_8U"
        CUDA_R_16I	"CUDA_R_16I"
        CUDA_C_16I	"CUDA_C_16I"
        CUDA_R_16U	"CUDA_R_16U"
        CUDA_C_16U	"CUDA_C_16U"
        CUDA_R_32I	"CUDA_R_32I"
        CUDA_C_32I	"CUDA_C_32I"
        CUDA_R_32U	"CUDA_R_32U"
        CUDA_C_32U	"CUDA_C_32U"
        CUDA_R_64I	"CUDA_R_64I"
        CUDA_C_64I	"CUDA_C_64I"
        CUDA_R_64U	"CUDA_R_64U"
        CUDA_C_64U	"CUDA_C_64U"

    ctypedef cudaDataType_t cudaDataType

    cdef enum libraryPropertyType_t:
        MAJOR_VERSION	"MAJOR_VERSION"
        MINOR_VERSION	"MINOR_VERSION"
        PATCH_LEVEL	"PATCH_LEVEL"

    ctypedef libraryPropertyType_t libraryPropertyType

    cudaError_t cudaDeviceReset()

    cudaError_t cudaDeviceSynchronize()

    cudaError_t cudaDeviceSetLimit(cudaLimit limit, size_t value)

    cudaError_t cudaDeviceGetLimit(size_t* pValue, cudaLimit limit)

    cudaError_t cudaDeviceGetTexture1DLinearMaxWidth(size_t* maxWidthInElements, cudaChannelFormatDesc* fmtDesc, int device)

    cudaError_t cudaDeviceGetCacheConfig(cudaFuncCache* pCacheConfig)

    cudaError_t cudaDeviceGetStreamPriorityRange(int* leastPriority, int* greatestPriority)

    cudaError_t cudaDeviceSetCacheConfig(cudaFuncCache cacheConfig)

    cudaError_t cudaDeviceGetSharedMemConfig(cudaSharedMemConfig* pConfig)

    cudaError_t cudaDeviceSetSharedMemConfig(cudaSharedMemConfig config)

    cudaError_t cudaDeviceGetByPCIBusId(int* device, char* pciBusId)

    cudaError_t cudaDeviceGetPCIBusId(char* pciBusId, int length, int device)

    cudaError_t cudaIpcGetEventHandle(cudaIpcEventHandle_t* handle, cudaEvent_t event)

    cudaError_t cudaIpcOpenEventHandle(cudaEvent_t* event, cudaIpcEventHandle_t handle)

    cudaError_t cudaIpcGetMemHandle(cudaIpcMemHandle_t* handle, void* devPtr)

    cudaError_t cudaIpcOpenMemHandle(void** devPtr, cudaIpcMemHandle_t handle, unsigned int flags)

    cudaError_t cudaIpcCloseMemHandle(void* devPtr)

    cudaError_t cudaDeviceFlushGPUDirectRDMAWrites(cudaFlushGPUDirectRDMAWritesTarget target, cudaFlushGPUDirectRDMAWritesScope scope)

    cudaError_t cudaThreadExit()

    cudaError_t cudaThreadSynchronize()

    cudaError_t cudaThreadSetLimit(cudaLimit limit, size_t value)

    cudaError_t cudaThreadGetLimit(size_t* pValue, cudaLimit limit)

    cudaError_t cudaThreadGetCacheConfig(cudaFuncCache* pCacheConfig)

    cudaError_t cudaThreadSetCacheConfig(cudaFuncCache cacheConfig)

    cudaError_t cudaGetLastError()

    cudaError_t cudaPeekAtLastError()

    const char* cudaGetErrorName(cudaError_t error)

    const char* cudaGetErrorString(cudaError_t error)

    cudaError_t cudaGetDeviceCount(int* count)

    cudaError_t cudaGetDeviceProperties(cudaDeviceProp* prop, int device)

    cudaError_t cudaDeviceGetAttribute(int* value, cudaDeviceAttr attr, int device)

    cudaError_t cudaDeviceGetDefaultMemPool(cudaMemPool_t* memPool, int device)

    cudaError_t cudaDeviceSetMemPool(int device, cudaMemPool_t memPool)

    cudaError_t cudaDeviceGetMemPool(cudaMemPool_t* memPool, int device)

    cudaError_t cudaDeviceGetNvSciSyncAttributes(void* nvSciSyncAttrList, int device, int flags)

    cudaError_t cudaDeviceGetP2PAttribute(int* value, cudaDeviceP2PAttr attr, int srcDevice, int dstDevice)

    cudaError_t cudaChooseDevice(int* device, cudaDeviceProp* prop)

    cudaError_t cudaSetDevice(int device)

    cudaError_t cudaGetDevice(int* device)

    cudaError_t cudaSetDeviceFlags(unsigned int flags)

    cudaError_t cudaGetDeviceFlags(unsigned int* flags)

    cudaError_t cudaStreamCreate(cudaStream_t* pStream)

    cudaError_t cudaStreamCreateWithFlags(cudaStream_t* pStream, unsigned int flags)

    cudaError_t cudaStreamCreateWithPriority(cudaStream_t* pStream, unsigned int flags, int priority)

    cudaError_t cudaStreamGetPriority(cudaStream_t hStream, int* priority)

    cudaError_t cudaStreamGetFlags(cudaStream_t hStream, unsigned int* flags)

    cudaError_t cudaCtxResetPersistingL2Cache()

    cudaError_t cudaStreamCopyAttributes(cudaStream_t dst, cudaStream_t src)

    cudaError_t cudaStreamGetAttribute(cudaStream_t hStream, cudaStreamAttrID attr, cudaStreamAttrValue* value_out)

    cudaError_t cudaStreamSetAttribute(cudaStream_t hStream, cudaStreamAttrID attr, cudaStreamAttrValue* value)

    cudaError_t cudaStreamDestroy(cudaStream_t stream)

    cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned int flags)

    ctypedef void (*cudaStreamCallback_t)(cudaStream_t stream, cudaError_t status, void* userData)

    cudaError_t cudaStreamAddCallback(cudaStream_t stream, cudaStreamCallback_t callback, void* userData, unsigned int flags)

    cudaError_t cudaStreamSynchronize(cudaStream_t stream)

    cudaError_t cudaStreamQuery(cudaStream_t stream)

    cudaError_t cudaStreamAttachMemAsync(cudaStream_t stream, void* devPtr, size_t length, unsigned int flags)

    cudaError_t cudaStreamBeginCapture(cudaStream_t stream, cudaStreamCaptureMode mode)

    cudaError_t cudaThreadExchangeStreamCaptureMode(cudaStreamCaptureMode* mode)

    cudaError_t cudaStreamEndCapture(cudaStream_t stream, cudaGraph_t* pGraph)

    cudaError_t cudaStreamIsCapturing(cudaStream_t stream, cudaStreamCaptureStatus* pCaptureStatus)

    cudaError_t cudaStreamGetCaptureInfo(cudaStream_t stream, cudaStreamCaptureStatus* pCaptureStatus, unsigned long long* pId)

    cudaError_t cudaStreamGetCaptureInfo_v2(cudaStream_t stream, cudaStreamCaptureStatus* captureStatus_out, unsigned long long* id_out, cudaGraph_t* graph_out, const cudaGraphNode_t** dependencies_out, size_t* numDependencies_out)

    cudaError_t cudaStreamUpdateCaptureDependencies(cudaStream_t stream, cudaGraphNode_t* dependencies, size_t numDependencies, unsigned int flags)

    cudaError_t cudaEventCreate(cudaEvent_t* event)

    cudaError_t cudaEventCreateWithFlags(cudaEvent_t* event, unsigned int flags)

    cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream)

    cudaError_t cudaEventRecordWithFlags(cudaEvent_t event, cudaStream_t stream, unsigned int flags)

    cudaError_t cudaEventQuery(cudaEvent_t event)

    cudaError_t cudaEventSynchronize(cudaEvent_t event)

    cudaError_t cudaEventDestroy(cudaEvent_t event)

    cudaError_t cudaEventElapsedTime(float* ms, cudaEvent_t start, cudaEvent_t end)

    cudaError_t cudaImportExternalMemory(cudaExternalMemory_t* extMem_out, cudaExternalMemoryHandleDesc* memHandleDesc)

    cudaError_t cudaExternalMemoryGetMappedBuffer(void** devPtr, cudaExternalMemory_t extMem, cudaExternalMemoryBufferDesc* bufferDesc)

    cudaError_t cudaExternalMemoryGetMappedMipmappedArray(cudaMipmappedArray_t* mipmap, cudaExternalMemory_t extMem, cudaExternalMemoryMipmappedArrayDesc* mipmapDesc)

    cudaError_t cudaDestroyExternalMemory(cudaExternalMemory_t extMem)

    cudaError_t cudaImportExternalSemaphore(cudaExternalSemaphore_t* extSem_out, cudaExternalSemaphoreHandleDesc* semHandleDesc)

    cudaError_t cudaSignalExternalSemaphoresAsync_v2(cudaExternalSemaphore_t* extSemArray, cudaExternalSemaphoreSignalParams* paramsArray, unsigned int numExtSems, cudaStream_t stream)

    cudaError_t cudaWaitExternalSemaphoresAsync_v2(cudaExternalSemaphore_t* extSemArray, cudaExternalSemaphoreWaitParams* paramsArray, unsigned int numExtSems, cudaStream_t stream)

    cudaError_t cudaDestroyExternalSemaphore(cudaExternalSemaphore_t extSem)

    cudaError_t cudaFuncSetCacheConfig(void* func, cudaFuncCache cacheConfig)

    cudaError_t cudaFuncSetSharedMemConfig(void* func, cudaSharedMemConfig config)

    cudaError_t cudaFuncGetAttributes(cudaFuncAttributes* attr, void* func)

    cudaError_t cudaFuncSetAttribute(void* func, cudaFuncAttribute attr, int value)

    cudaError_t cudaSetDoubleForDevice(double* d)

    cudaError_t cudaSetDoubleForHost(double* d)

    cudaError_t cudaLaunchHostFunc(cudaStream_t stream, cudaHostFn_t fn, void* userData)

    cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(int* numBlocks, void* func, int blockSize, size_t dynamicSMemSize)

    cudaError_t cudaOccupancyAvailableDynamicSMemPerBlock(size_t* dynamicSmemSize, void* func, int numBlocks, int blockSize)

    cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int* numBlocks, void* func, int blockSize, size_t dynamicSMemSize, unsigned int flags)

    cudaError_t cudaMallocManaged(void** devPtr, size_t size, unsigned int flags)

    cudaError_t cudaMalloc(void** devPtr, size_t size)

    cudaError_t cudaMallocHost(void** ptr, size_t size)

    cudaError_t cudaMallocPitch(void** devPtr, size_t* pitch, size_t width, size_t height)

    cudaError_t cudaMallocArray(cudaArray_t* array, cudaChannelFormatDesc* desc, size_t width, size_t height, unsigned int flags)

    cudaError_t cudaFree(void* devPtr)

    cudaError_t cudaFreeHost(void* ptr)

    cudaError_t cudaFreeArray(cudaArray_t array)

    cudaError_t cudaFreeMipmappedArray(cudaMipmappedArray_t mipmappedArray)

    cudaError_t cudaHostAlloc(void** pHost, size_t size, unsigned int flags)

    cudaError_t cudaHostRegister(void* ptr, size_t size, unsigned int flags)

    cudaError_t cudaHostUnregister(void* ptr)

    cudaError_t cudaHostGetDevicePointer(void** pDevice, void* pHost, unsigned int flags)

    cudaError_t cudaHostGetFlags(unsigned int* pFlags, void* pHost)

    cudaError_t cudaMalloc3D(cudaPitchedPtr* pitchedDevPtr, cudaExtent extent)

    cudaError_t cudaMalloc3DArray(cudaArray_t* array, cudaChannelFormatDesc* desc, cudaExtent extent, unsigned int flags)

    cudaError_t cudaMallocMipmappedArray(cudaMipmappedArray_t* mipmappedArray, cudaChannelFormatDesc* desc, cudaExtent extent, unsigned int numLevels, unsigned int flags)

    cudaError_t cudaGetMipmappedArrayLevel(cudaArray_t* levelArray, cudaMipmappedArray_const_t mipmappedArray, unsigned int level)

    cudaError_t cudaMemcpy3D(cudaMemcpy3DParms* p)

    cudaError_t cudaMemcpy3DPeer(cudaMemcpy3DPeerParms* p)

    cudaError_t cudaMemcpy3DAsync(cudaMemcpy3DParms* p, cudaStream_t stream)

    cudaError_t cudaMemcpy3DPeerAsync(cudaMemcpy3DPeerParms* p, cudaStream_t stream)

    cudaError_t cudaMemGetInfo(size_t* free, size_t* total)

    cudaError_t cudaArrayGetInfo(cudaChannelFormatDesc* desc, cudaExtent* extent, unsigned int* flags, cudaArray_t array)

    cudaError_t cudaArrayGetPlane(cudaArray_t* pPlaneArray, cudaArray_t hArray, unsigned int planeIdx)

    cudaError_t cudaArrayGetSparseProperties(cudaArraySparseProperties* sparseProperties, cudaArray_t array)

    cudaError_t cudaMipmappedArrayGetSparseProperties(cudaArraySparseProperties* sparseProperties, cudaMipmappedArray_t mipmap)

    cudaError_t cudaMemcpy(void* dst, void* src, size_t count, cudaMemcpyKind kind)

    cudaError_t cudaMemcpyPeer(void* dst, int dstDevice, void* src, int srcDevice, size_t count)

    cudaError_t cudaMemcpy2D(void* dst, size_t dpitch, void* src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind)

    cudaError_t cudaMemcpy2DToArray(cudaArray_t dst, size_t wOffset, size_t hOffset, void* src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind)

    cudaError_t cudaMemcpy2DFromArray(void* dst, size_t dpitch, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t width, size_t height, cudaMemcpyKind kind)

    cudaError_t cudaMemcpy2DArrayToArray(cudaArray_t dst, size_t wOffsetDst, size_t hOffsetDst, cudaArray_const_t src, size_t wOffsetSrc, size_t hOffsetSrc, size_t width, size_t height, cudaMemcpyKind kind)

    cudaError_t cudaMemcpyAsync(void* dst, void* src, size_t count, cudaMemcpyKind kind, cudaStream_t stream)

    cudaError_t cudaMemcpyPeerAsync(void* dst, int dstDevice, void* src, int srcDevice, size_t count, cudaStream_t stream)

    cudaError_t cudaMemcpy2DAsync(void* dst, size_t dpitch, void* src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream)

    cudaError_t cudaMemcpy2DToArrayAsync(cudaArray_t dst, size_t wOffset, size_t hOffset, void* src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream)

    cudaError_t cudaMemcpy2DFromArrayAsync(void* dst, size_t dpitch, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream)

    cudaError_t cudaMemset(void* devPtr, int value, size_t count)

    cudaError_t cudaMemset2D(void* devPtr, size_t pitch, int value, size_t width, size_t height)

    cudaError_t cudaMemset3D(cudaPitchedPtr pitchedDevPtr, int value, cudaExtent extent)

    cudaError_t cudaMemsetAsync(void* devPtr, int value, size_t count, cudaStream_t stream)

    cudaError_t cudaMemset2DAsync(void* devPtr, size_t pitch, int value, size_t width, size_t height, cudaStream_t stream)

    cudaError_t cudaMemset3DAsync(cudaPitchedPtr pitchedDevPtr, int value, cudaExtent extent, cudaStream_t stream)

    cudaError_t cudaMemPrefetchAsync(void* devPtr, size_t count, int dstDevice, cudaStream_t stream)

    cudaError_t cudaMemAdvise(void* devPtr, size_t count, cudaMemoryAdvise advice, int device)

    cudaError_t cudaMemRangeGetAttribute(void* data, size_t dataSize, cudaMemRangeAttribute attribute, void* devPtr, size_t count)

    cudaError_t cudaMemRangeGetAttributes(void** data, size_t* dataSizes, cudaMemRangeAttribute* attributes, size_t numAttributes, void* devPtr, size_t count)

    cudaError_t cudaMemcpyToArray(cudaArray_t dst, size_t wOffset, size_t hOffset, void* src, size_t count, cudaMemcpyKind kind)

    cudaError_t cudaMemcpyFromArray(void* dst, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t count, cudaMemcpyKind kind)

    cudaError_t cudaMemcpyArrayToArray(cudaArray_t dst, size_t wOffsetDst, size_t hOffsetDst, cudaArray_const_t src, size_t wOffsetSrc, size_t hOffsetSrc, size_t count, cudaMemcpyKind kind)

    cudaError_t cudaMemcpyToArrayAsync(cudaArray_t dst, size_t wOffset, size_t hOffset, void* src, size_t count, cudaMemcpyKind kind, cudaStream_t stream)

    cudaError_t cudaMemcpyFromArrayAsync(void* dst, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t count, cudaMemcpyKind kind, cudaStream_t stream)

    cudaError_t cudaMallocAsync(void** devPtr, size_t size, cudaStream_t hStream)

    cudaError_t cudaFreeAsync(void* devPtr, cudaStream_t hStream)

    cudaError_t cudaMemPoolTrimTo(cudaMemPool_t memPool, size_t minBytesToKeep)

    cudaError_t cudaMemPoolSetAttribute(cudaMemPool_t memPool, cudaMemPoolAttr attr, void* value)

    cudaError_t cudaMemPoolGetAttribute(cudaMemPool_t memPool, cudaMemPoolAttr attr, void* value)

    cudaError_t cudaMemPoolSetAccess(cudaMemPool_t memPool, cudaMemAccessDesc* descList, size_t count)

    cudaError_t cudaMemPoolGetAccess(cudaMemAccessFlags* flags, cudaMemPool_t memPool, cudaMemLocation* location)

    cudaError_t cudaMemPoolCreate(cudaMemPool_t* memPool, cudaMemPoolProps* poolProps)

    cudaError_t cudaMemPoolDestroy(cudaMemPool_t memPool)

    cudaError_t cudaMallocFromPoolAsync(void** ptr, size_t size, cudaMemPool_t memPool, cudaStream_t stream)

    cudaError_t cudaMemPoolExportToShareableHandle(void* shareableHandle, cudaMemPool_t memPool, cudaMemAllocationHandleType handleType, unsigned int flags)

    cudaError_t cudaMemPoolImportFromShareableHandle(cudaMemPool_t* memPool, void* shareableHandle, cudaMemAllocationHandleType handleType, unsigned int flags)

    cudaError_t cudaMemPoolExportPointer(cudaMemPoolPtrExportData* exportData, void* ptr)

    cudaError_t cudaMemPoolImportPointer(void** ptr, cudaMemPool_t memPool, cudaMemPoolPtrExportData* exportData)

    cudaError_t cudaPointerGetAttributes(cudaPointerAttributes* attributes, void* ptr)

    cudaError_t cudaDeviceCanAccessPeer(int* canAccessPeer, int device, int peerDevice)

    cudaError_t cudaDeviceEnablePeerAccess(int peerDevice, unsigned int flags)

    cudaError_t cudaDeviceDisablePeerAccess(int peerDevice)

    cudaError_t cudaGraphicsUnregisterResource(cudaGraphicsResource_t resource)

    cudaError_t cudaGraphicsResourceSetMapFlags(cudaGraphicsResource_t resource, unsigned int flags)

    cudaError_t cudaGraphicsMapResources(int count, cudaGraphicsResource_t* resources, cudaStream_t stream)

    cudaError_t cudaGraphicsUnmapResources(int count, cudaGraphicsResource_t* resources, cudaStream_t stream)

    cudaError_t cudaGraphicsResourceGetMappedPointer(void** devPtr, size_t* size, cudaGraphicsResource_t resource)

    cudaError_t cudaGraphicsSubResourceGetMappedArray(cudaArray_t* array, cudaGraphicsResource_t resource, unsigned int arrayIndex, unsigned int mipLevel)

    cudaError_t cudaGraphicsResourceGetMappedMipmappedArray(cudaMipmappedArray_t* mipmappedArray, cudaGraphicsResource_t resource)

    cudaError_t cudaGetChannelDesc(cudaChannelFormatDesc* desc, cudaArray_const_t array)

    cudaChannelFormatDesc cudaCreateChannelDesc(int x, int y, int z, int w, cudaChannelFormatKind f)

    cudaError_t cudaCreateTextureObject(cudaTextureObject_t* pTexObject, cudaResourceDesc* pResDesc, cudaTextureDesc* pTexDesc, cudaResourceViewDesc* pResViewDesc)

    cudaError_t cudaDestroyTextureObject(cudaTextureObject_t texObject)

    cudaError_t cudaGetTextureObjectResourceDesc(cudaResourceDesc* pResDesc, cudaTextureObject_t texObject)

    cudaError_t cudaGetTextureObjectTextureDesc(cudaTextureDesc* pTexDesc, cudaTextureObject_t texObject)

    cudaError_t cudaGetTextureObjectResourceViewDesc(cudaResourceViewDesc* pResViewDesc, cudaTextureObject_t texObject)

    cudaError_t cudaCreateSurfaceObject(cudaSurfaceObject_t* pSurfObject, cudaResourceDesc* pResDesc)

    cudaError_t cudaDestroySurfaceObject(cudaSurfaceObject_t surfObject)

    cudaError_t cudaGetSurfaceObjectResourceDesc(cudaResourceDesc* pResDesc, cudaSurfaceObject_t surfObject)

    cudaError_t cudaDriverGetVersion(int* driverVersion)

    cudaError_t cudaRuntimeGetVersion(int* runtimeVersion)

    cudaError_t cudaGraphCreate(cudaGraph_t* pGraph, unsigned int flags)

    cudaError_t cudaGraphAddKernelNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, cudaGraphNode_t* pDependencies, size_t numDependencies, cudaKernelNodeParams* pNodeParams)

    cudaError_t cudaGraphKernelNodeGetParams(cudaGraphNode_t node, cudaKernelNodeParams* pNodeParams)

    cudaError_t cudaGraphKernelNodeSetParams(cudaGraphNode_t node, cudaKernelNodeParams* pNodeParams)

    cudaError_t cudaGraphKernelNodeCopyAttributes(cudaGraphNode_t hSrc, cudaGraphNode_t hDst)

    cudaError_t cudaGraphKernelNodeGetAttribute(cudaGraphNode_t hNode, cudaKernelNodeAttrID attr, cudaKernelNodeAttrValue* value_out)

    cudaError_t cudaGraphKernelNodeSetAttribute(cudaGraphNode_t hNode, cudaKernelNodeAttrID attr, cudaKernelNodeAttrValue* value)

    cudaError_t cudaGraphAddMemcpyNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, cudaGraphNode_t* pDependencies, size_t numDependencies, cudaMemcpy3DParms* pCopyParams)

    cudaError_t cudaGraphAddMemcpyNode1D(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, cudaGraphNode_t* pDependencies, size_t numDependencies, void* dst, void* src, size_t count, cudaMemcpyKind kind)

    cudaError_t cudaGraphMemcpyNodeGetParams(cudaGraphNode_t node, cudaMemcpy3DParms* pNodeParams)

    cudaError_t cudaGraphMemcpyNodeSetParams(cudaGraphNode_t node, cudaMemcpy3DParms* pNodeParams)

    cudaError_t cudaGraphMemcpyNodeSetParams1D(cudaGraphNode_t node, void* dst, void* src, size_t count, cudaMemcpyKind kind)

    cudaError_t cudaGraphAddMemsetNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, cudaGraphNode_t* pDependencies, size_t numDependencies, cudaMemsetParams* pMemsetParams)

    cudaError_t cudaGraphMemsetNodeGetParams(cudaGraphNode_t node, cudaMemsetParams* pNodeParams)

    cudaError_t cudaGraphMemsetNodeSetParams(cudaGraphNode_t node, cudaMemsetParams* pNodeParams)

    cudaError_t cudaGraphAddHostNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, cudaGraphNode_t* pDependencies, size_t numDependencies, cudaHostNodeParams* pNodeParams)

    cudaError_t cudaGraphHostNodeGetParams(cudaGraphNode_t node, cudaHostNodeParams* pNodeParams)

    cudaError_t cudaGraphHostNodeSetParams(cudaGraphNode_t node, cudaHostNodeParams* pNodeParams)

    cudaError_t cudaGraphAddChildGraphNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, cudaGraphNode_t* pDependencies, size_t numDependencies, cudaGraph_t childGraph)

    cudaError_t cudaGraphChildGraphNodeGetGraph(cudaGraphNode_t node, cudaGraph_t* pGraph)

    cudaError_t cudaGraphAddEmptyNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, cudaGraphNode_t* pDependencies, size_t numDependencies)

    cudaError_t cudaGraphAddEventRecordNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, cudaGraphNode_t* pDependencies, size_t numDependencies, cudaEvent_t event)

    cudaError_t cudaGraphEventRecordNodeGetEvent(cudaGraphNode_t node, cudaEvent_t* event_out)

    cudaError_t cudaGraphEventRecordNodeSetEvent(cudaGraphNode_t node, cudaEvent_t event)

    cudaError_t cudaGraphAddEventWaitNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, cudaGraphNode_t* pDependencies, size_t numDependencies, cudaEvent_t event)

    cudaError_t cudaGraphEventWaitNodeGetEvent(cudaGraphNode_t node, cudaEvent_t* event_out)

    cudaError_t cudaGraphEventWaitNodeSetEvent(cudaGraphNode_t node, cudaEvent_t event)

    cudaError_t cudaGraphAddExternalSemaphoresSignalNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, cudaGraphNode_t* pDependencies, size_t numDependencies, cudaExternalSemaphoreSignalNodeParams* nodeParams)

    cudaError_t cudaGraphExternalSemaphoresSignalNodeGetParams(cudaGraphNode_t hNode, cudaExternalSemaphoreSignalNodeParams* params_out)

    cudaError_t cudaGraphExternalSemaphoresSignalNodeSetParams(cudaGraphNode_t hNode, cudaExternalSemaphoreSignalNodeParams* nodeParams)

    cudaError_t cudaGraphAddExternalSemaphoresWaitNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, cudaGraphNode_t* pDependencies, size_t numDependencies, cudaExternalSemaphoreWaitNodeParams* nodeParams)

    cudaError_t cudaGraphExternalSemaphoresWaitNodeGetParams(cudaGraphNode_t hNode, cudaExternalSemaphoreWaitNodeParams* params_out)

    cudaError_t cudaGraphExternalSemaphoresWaitNodeSetParams(cudaGraphNode_t hNode, cudaExternalSemaphoreWaitNodeParams* nodeParams)

    cudaError_t cudaGraphAddMemAllocNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, cudaGraphNode_t* pDependencies, size_t numDependencies, cudaMemAllocNodeParams* nodeParams)

    cudaError_t cudaGraphMemAllocNodeGetParams(cudaGraphNode_t node, cudaMemAllocNodeParams* params_out)

    cudaError_t cudaGraphAddMemFreeNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, cudaGraphNode_t* pDependencies, size_t numDependencies, void* dptr)

    cudaError_t cudaGraphMemFreeNodeGetParams(cudaGraphNode_t node, void* dptr_out)

    cudaError_t cudaDeviceGraphMemTrim(int device)

    cudaError_t cudaDeviceGetGraphMemAttribute(int device, cudaGraphMemAttributeType attr, void* value)

    cudaError_t cudaDeviceSetGraphMemAttribute(int device, cudaGraphMemAttributeType attr, void* value)

    cudaError_t cudaGraphClone(cudaGraph_t* pGraphClone, cudaGraph_t originalGraph)

    cudaError_t cudaGraphNodeFindInClone(cudaGraphNode_t* pNode, cudaGraphNode_t originalNode, cudaGraph_t clonedGraph)

    cudaError_t cudaGraphNodeGetType(cudaGraphNode_t node, cudaGraphNodeType* pType)

    cudaError_t cudaGraphGetNodes(cudaGraph_t graph, cudaGraphNode_t* nodes, size_t* numNodes)

    cudaError_t cudaGraphGetRootNodes(cudaGraph_t graph, cudaGraphNode_t* pRootNodes, size_t* pNumRootNodes)

    cudaError_t cudaGraphGetEdges(cudaGraph_t graph, cudaGraphNode_t* from_, cudaGraphNode_t* to, size_t* numEdges)

    cudaError_t cudaGraphNodeGetDependencies(cudaGraphNode_t node, cudaGraphNode_t* pDependencies, size_t* pNumDependencies)

    cudaError_t cudaGraphNodeGetDependentNodes(cudaGraphNode_t node, cudaGraphNode_t* pDependentNodes, size_t* pNumDependentNodes)

    cudaError_t cudaGraphAddDependencies(cudaGraph_t graph, cudaGraphNode_t* from_, cudaGraphNode_t* to, size_t numDependencies)

    cudaError_t cudaGraphRemoveDependencies(cudaGraph_t graph, cudaGraphNode_t* from_, cudaGraphNode_t* to, size_t numDependencies)

    cudaError_t cudaGraphDestroyNode(cudaGraphNode_t node)

    cudaError_t cudaGraphInstantiate(cudaGraphExec_t* pGraphExec, cudaGraph_t graph, cudaGraphNode_t* pErrorNode, char* pLogBuffer, size_t bufferSize)

    cudaError_t cudaGraphInstantiateWithFlags(cudaGraphExec_t* pGraphExec, cudaGraph_t graph, unsigned long long flags)

    cudaError_t cudaGraphExecKernelNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, cudaKernelNodeParams* pNodeParams)

    cudaError_t cudaGraphExecMemcpyNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, cudaMemcpy3DParms* pNodeParams)

    cudaError_t cudaGraphExecMemcpyNodeSetParams1D(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, void* dst, void* src, size_t count, cudaMemcpyKind kind)

    cudaError_t cudaGraphExecMemsetNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, cudaMemsetParams* pNodeParams)

    cudaError_t cudaGraphExecHostNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, cudaHostNodeParams* pNodeParams)

    cudaError_t cudaGraphExecChildGraphNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, cudaGraph_t childGraph)

    cudaError_t cudaGraphExecEventRecordNodeSetEvent(cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, cudaEvent_t event)

    cudaError_t cudaGraphExecEventWaitNodeSetEvent(cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, cudaEvent_t event)

    cudaError_t cudaGraphExecExternalSemaphoresSignalNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, cudaExternalSemaphoreSignalNodeParams* nodeParams)

    cudaError_t cudaGraphExecExternalSemaphoresWaitNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, cudaExternalSemaphoreWaitNodeParams* nodeParams)

    cudaError_t cudaGraphExecUpdate(cudaGraphExec_t hGraphExec, cudaGraph_t hGraph, cudaGraphNode_t* hErrorNode_out, cudaGraphExecUpdateResult* updateResult_out)

    cudaError_t cudaGraphUpload(cudaGraphExec_t graphExec, cudaStream_t stream)

    cudaError_t cudaGraphLaunch(cudaGraphExec_t graphExec, cudaStream_t stream)

    cudaError_t cudaGraphExecDestroy(cudaGraphExec_t graphExec)

    cudaError_t cudaGraphDestroy(cudaGraph_t graph)

    cudaError_t cudaGraphDebugDotPrint(cudaGraph_t graph, char* path, unsigned int flags)

    cudaError_t cudaUserObjectCreate(cudaUserObject_t* object_out, void* ptr, cudaHostFn_t destroy, unsigned int initialRefcount, unsigned int flags)

    cudaError_t cudaUserObjectRetain(cudaUserObject_t object, unsigned int count)

    cudaError_t cudaUserObjectRelease(cudaUserObject_t object, unsigned int count)

    cudaError_t cudaGraphRetainUserObject(cudaGraph_t graph, cudaUserObject_t object, unsigned int count, unsigned int flags)

    cudaError_t cudaGraphReleaseUserObject(cudaGraph_t graph, cudaUserObject_t object, unsigned int count)

    cudaError_t cudaGetDriverEntryPoint(char* symbol, void** funcPtr, unsigned long long flags)

    cudaError_t cudaGetExportTable(const void** ppExportTable, cudaUUID_t* pExportTableId)

    cudaPitchedPtr make_cudaPitchedPtr(void* d, size_t p, size_t xsz, size_t ysz)

    cudaPos make_cudaPos(size_t x, size_t y, size_t z)

    cudaExtent make_cudaExtent(size_t w, size_t h, size_t d)
