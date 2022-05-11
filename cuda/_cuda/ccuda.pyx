# Copyright 2021-2022 NVIDIA Corporation.  All rights reserved.
#
# Please refer to the NVIDIA end user license agreement (EULA) associated
# with this source code for terms and conditions that govern your use of
# this software. Any use, reproduction, disclosure, or distribution of
# this software and related documentation outside the terms of the EULA
# is strictly prohibited.
IF UNAME_SYSNAME == "Windows":
    import win32api
    import struct
    from pywintypes import error
ELSE:
    cimport cuda._lib.dlfcn as dlfcn
import os
import sys
cimport cuda._cuda.loader as loader
cdef bint __cuPythonInit = False
cdef void *__cuGetErrorString = NULL
cdef void *__cuGetErrorName = NULL
cdef void *__cuInit = NULL
cdef void *__cuDriverGetVersion = NULL
cdef void *__cuDeviceGet = NULL
cdef void *__cuDeviceGetCount = NULL
cdef void *__cuDeviceGetName = NULL
cdef void *__cuDeviceGetUuid = NULL
cdef void *__cuDeviceGetUuid_v2 = NULL
cdef void *__cuDeviceGetLuid = NULL
cdef void *__cuDeviceTotalMem_v2 = NULL
cdef void *__cuDeviceGetTexture1DLinearMaxWidth = NULL
cdef void *__cuDeviceGetAttribute = NULL
cdef void *__cuDeviceGetNvSciSyncAttributes = NULL
cdef void *__cuDeviceSetMemPool = NULL
cdef void *__cuDeviceGetMemPool = NULL
cdef void *__cuDeviceGetDefaultMemPool = NULL
cdef void *__cuFlushGPUDirectRDMAWrites = NULL
cdef void *__cuDeviceGetProperties = NULL
cdef void *__cuDeviceComputeCapability = NULL
cdef void *__cuDevicePrimaryCtxRetain = NULL
cdef void *__cuDevicePrimaryCtxRelease_v2 = NULL
cdef void *__cuDevicePrimaryCtxSetFlags_v2 = NULL
cdef void *__cuDevicePrimaryCtxGetState = NULL
cdef void *__cuDevicePrimaryCtxReset_v2 = NULL
cdef void *__cuDeviceGetExecAffinitySupport = NULL
cdef void *__cuCtxCreate_v2 = NULL
cdef void *__cuCtxCreate_v3 = NULL
cdef void *__cuCtxDestroy_v2 = NULL
cdef void *__cuCtxPushCurrent_v2 = NULL
cdef void *__cuCtxPopCurrent_v2 = NULL
cdef void *__cuCtxSetCurrent = NULL
cdef void *__cuCtxGetCurrent = NULL
cdef void *__cuCtxGetDevice = NULL
cdef void *__cuCtxGetFlags = NULL
cdef void *__cuCtxSynchronize = NULL
cdef void *__cuCtxSetLimit = NULL
cdef void *__cuCtxGetLimit = NULL
cdef void *__cuCtxGetCacheConfig = NULL
cdef void *__cuCtxSetCacheConfig = NULL
cdef void *__cuCtxGetSharedMemConfig = NULL
cdef void *__cuCtxSetSharedMemConfig = NULL
cdef void *__cuCtxGetApiVersion = NULL
cdef void *__cuCtxGetStreamPriorityRange = NULL
cdef void *__cuCtxResetPersistingL2Cache = NULL
cdef void *__cuCtxGetExecAffinity = NULL
cdef void *__cuCtxAttach = NULL
cdef void *__cuCtxDetach = NULL
cdef void *__cuModuleLoad = NULL
cdef void *__cuModuleLoadData = NULL
cdef void *__cuModuleLoadDataEx = NULL
cdef void *__cuModuleLoadFatBinary = NULL
cdef void *__cuModuleUnload = NULL
cdef void *__cuModuleGetFunction = NULL
cdef void *__cuModuleGetGlobal_v2 = NULL
cdef void *__cuModuleGetTexRef = NULL
cdef void *__cuModuleGetSurfRef = NULL
cdef void *__cuLinkCreate_v2 = NULL
cdef void *__cuLinkAddData_v2 = NULL
cdef void *__cuLinkAddFile_v2 = NULL
cdef void *__cuLinkComplete = NULL
cdef void *__cuLinkDestroy = NULL
cdef void *__cuMemGetInfo_v2 = NULL
cdef void *__cuMemAlloc_v2 = NULL
cdef void *__cuMemAllocPitch_v2 = NULL
cdef void *__cuMemFree_v2 = NULL
cdef void *__cuMemGetAddressRange_v2 = NULL
cdef void *__cuMemAllocHost_v2 = NULL
cdef void *__cuMemFreeHost = NULL
cdef void *__cuMemHostAlloc = NULL
cdef void *__cuMemHostGetDevicePointer_v2 = NULL
cdef void *__cuMemHostGetFlags = NULL
cdef void *__cuMemAllocManaged = NULL
cdef void *__cuDeviceGetByPCIBusId = NULL
cdef void *__cuDeviceGetPCIBusId = NULL
cdef void *__cuIpcGetEventHandle = NULL
cdef void *__cuIpcOpenEventHandle = NULL
cdef void *__cuIpcGetMemHandle = NULL
cdef void *__cuIpcOpenMemHandle_v2 = NULL
cdef void *__cuIpcCloseMemHandle = NULL
cdef void *__cuMemHostRegister_v2 = NULL
cdef void *__cuMemHostUnregister = NULL
cdef void *__cuMemcpy = NULL
cdef void *__cuMemcpyPeer = NULL
cdef void *__cuMemcpyHtoD_v2 = NULL
cdef void *__cuMemcpyDtoH_v2 = NULL
cdef void *__cuMemcpyDtoD_v2 = NULL
cdef void *__cuMemcpyDtoA_v2 = NULL
cdef void *__cuMemcpyAtoD_v2 = NULL
cdef void *__cuMemcpyHtoA_v2 = NULL
cdef void *__cuMemcpyAtoH_v2 = NULL
cdef void *__cuMemcpyAtoA_v2 = NULL
cdef void *__cuMemcpy2D_v2 = NULL
cdef void *__cuMemcpy2DUnaligned_v2 = NULL
cdef void *__cuMemcpy3D_v2 = NULL
cdef void *__cuMemcpy3DPeer = NULL
cdef void *__cuMemcpyAsync = NULL
cdef void *__cuMemcpyPeerAsync = NULL
cdef void *__cuMemcpyHtoDAsync_v2 = NULL
cdef void *__cuMemcpyDtoHAsync_v2 = NULL
cdef void *__cuMemcpyDtoDAsync_v2 = NULL
cdef void *__cuMemcpyHtoAAsync_v2 = NULL
cdef void *__cuMemcpyAtoHAsync_v2 = NULL
cdef void *__cuMemcpy2DAsync_v2 = NULL
cdef void *__cuMemcpy3DAsync_v2 = NULL
cdef void *__cuMemcpy3DPeerAsync = NULL
cdef void *__cuMemsetD8_v2 = NULL
cdef void *__cuMemsetD16_v2 = NULL
cdef void *__cuMemsetD32_v2 = NULL
cdef void *__cuMemsetD2D8_v2 = NULL
cdef void *__cuMemsetD2D16_v2 = NULL
cdef void *__cuMemsetD2D32_v2 = NULL
cdef void *__cuMemsetD8Async = NULL
cdef void *__cuMemsetD16Async = NULL
cdef void *__cuMemsetD32Async = NULL
cdef void *__cuMemsetD2D8Async = NULL
cdef void *__cuMemsetD2D16Async = NULL
cdef void *__cuMemsetD2D32Async = NULL
cdef void *__cuArrayCreate_v2 = NULL
cdef void *__cuArrayGetDescriptor_v2 = NULL
cdef void *__cuArrayGetSparseProperties = NULL
cdef void *__cuMipmappedArrayGetSparseProperties = NULL
cdef void *__cuArrayGetMemoryRequirements = NULL
cdef void *__cuMipmappedArrayGetMemoryRequirements = NULL
cdef void *__cuArrayGetPlane = NULL
cdef void *__cuArrayDestroy = NULL
cdef void *__cuArray3DCreate_v2 = NULL
cdef void *__cuArray3DGetDescriptor_v2 = NULL
cdef void *__cuMipmappedArrayCreate = NULL
cdef void *__cuMipmappedArrayGetLevel = NULL
cdef void *__cuMipmappedArrayDestroy = NULL
cdef void *__cuMemAddressReserve = NULL
cdef void *__cuMemAddressFree = NULL
cdef void *__cuMemCreate = NULL
cdef void *__cuMemRelease = NULL
cdef void *__cuMemMap = NULL
cdef void *__cuMemMapArrayAsync = NULL
cdef void *__cuMemUnmap = NULL
cdef void *__cuMemSetAccess = NULL
cdef void *__cuMemGetAccess = NULL
cdef void *__cuMemExportToShareableHandle = NULL
cdef void *__cuMemImportFromShareableHandle = NULL
cdef void *__cuMemGetAllocationGranularity = NULL
cdef void *__cuMemGetAllocationPropertiesFromHandle = NULL
cdef void *__cuMemRetainAllocationHandle = NULL
cdef void *__cuMemFreeAsync = NULL
cdef void *__cuMemAllocAsync = NULL
cdef void *__cuMemPoolTrimTo = NULL
cdef void *__cuMemPoolSetAttribute = NULL
cdef void *__cuMemPoolGetAttribute = NULL
cdef void *__cuMemPoolSetAccess = NULL
cdef void *__cuMemPoolGetAccess = NULL
cdef void *__cuMemPoolCreate = NULL
cdef void *__cuMemPoolDestroy = NULL
cdef void *__cuMemAllocFromPoolAsync = NULL
cdef void *__cuMemPoolExportToShareableHandle = NULL
cdef void *__cuMemPoolImportFromShareableHandle = NULL
cdef void *__cuMemPoolExportPointer = NULL
cdef void *__cuMemPoolImportPointer = NULL
cdef void *__cuPointerGetAttribute = NULL
cdef void *__cuMemPrefetchAsync = NULL
cdef void *__cuMemAdvise = NULL
cdef void *__cuMemRangeGetAttribute = NULL
cdef void *__cuMemRangeGetAttributes = NULL
cdef void *__cuPointerSetAttribute = NULL
cdef void *__cuPointerGetAttributes = NULL
cdef void *__cuStreamCreate = NULL
cdef void *__cuStreamCreateWithPriority = NULL
cdef void *__cuStreamGetPriority = NULL
cdef void *__cuStreamGetFlags = NULL
cdef void *__cuStreamGetCtx = NULL
cdef void *__cuStreamWaitEvent = NULL
cdef void *__cuStreamAddCallback = NULL
cdef void *__cuStreamBeginCapture_v2 = NULL
cdef void *__cuThreadExchangeStreamCaptureMode = NULL
cdef void *__cuStreamEndCapture = NULL
cdef void *__cuStreamIsCapturing = NULL
cdef void *__cuStreamGetCaptureInfo = NULL
cdef void *__cuStreamGetCaptureInfo_v2 = NULL
cdef void *__cuStreamUpdateCaptureDependencies = NULL
cdef void *__cuStreamAttachMemAsync = NULL
cdef void *__cuStreamQuery = NULL
cdef void *__cuStreamSynchronize = NULL
cdef void *__cuStreamDestroy_v2 = NULL
cdef void *__cuStreamCopyAttributes = NULL
cdef void *__cuStreamGetAttribute = NULL
cdef void *__cuStreamSetAttribute = NULL
cdef void *__cuEventCreate = NULL
cdef void *__cuEventRecord = NULL
cdef void *__cuEventRecordWithFlags = NULL
cdef void *__cuEventQuery = NULL
cdef void *__cuEventSynchronize = NULL
cdef void *__cuEventDestroy_v2 = NULL
cdef void *__cuEventElapsedTime = NULL
cdef void *__cuImportExternalMemory = NULL
cdef void *__cuExternalMemoryGetMappedBuffer = NULL
cdef void *__cuExternalMemoryGetMappedMipmappedArray = NULL
cdef void *__cuDestroyExternalMemory = NULL
cdef void *__cuImportExternalSemaphore = NULL
cdef void *__cuSignalExternalSemaphoresAsync = NULL
cdef void *__cuWaitExternalSemaphoresAsync = NULL
cdef void *__cuDestroyExternalSemaphore = NULL
cdef void *__cuStreamWaitValue32 = NULL
cdef void *__cuStreamWaitValue64 = NULL
cdef void *__cuStreamWriteValue32 = NULL
cdef void *__cuStreamWriteValue64 = NULL
cdef void *__cuStreamBatchMemOp = NULL
cdef void *__cuStreamWaitValue32_v2 = NULL
cdef void *__cuStreamWaitValue64_v2 = NULL
cdef void *__cuStreamWriteValue32_v2 = NULL
cdef void *__cuStreamWriteValue64_v2 = NULL
cdef void *__cuStreamBatchMemOp_v2 = NULL
cdef void *__cuFuncGetAttribute = NULL
cdef void *__cuFuncSetAttribute = NULL
cdef void *__cuFuncSetCacheConfig = NULL
cdef void *__cuFuncSetSharedMemConfig = NULL
cdef void *__cuFuncGetModule = NULL
cdef void *__cuLaunchKernel = NULL
cdef void *__cuLaunchCooperativeKernel = NULL
cdef void *__cuLaunchCooperativeKernelMultiDevice = NULL
cdef void *__cuLaunchHostFunc = NULL
cdef void *__cuFuncSetBlockShape = NULL
cdef void *__cuFuncSetSharedSize = NULL
cdef void *__cuParamSetSize = NULL
cdef void *__cuParamSeti = NULL
cdef void *__cuParamSetf = NULL
cdef void *__cuParamSetv = NULL
cdef void *__cuLaunch = NULL
cdef void *__cuLaunchGrid = NULL
cdef void *__cuLaunchGridAsync = NULL
cdef void *__cuParamSetTexRef = NULL
cdef void *__cuGraphCreate = NULL
cdef void *__cuGraphAddKernelNode = NULL
cdef void *__cuGraphKernelNodeGetParams = NULL
cdef void *__cuGraphKernelNodeSetParams = NULL
cdef void *__cuGraphAddMemcpyNode = NULL
cdef void *__cuGraphMemcpyNodeGetParams = NULL
cdef void *__cuGraphMemcpyNodeSetParams = NULL
cdef void *__cuGraphAddMemsetNode = NULL
cdef void *__cuGraphMemsetNodeGetParams = NULL
cdef void *__cuGraphMemsetNodeSetParams = NULL
cdef void *__cuGraphAddHostNode = NULL
cdef void *__cuGraphHostNodeGetParams = NULL
cdef void *__cuGraphHostNodeSetParams = NULL
cdef void *__cuGraphAddChildGraphNode = NULL
cdef void *__cuGraphChildGraphNodeGetGraph = NULL
cdef void *__cuGraphAddEmptyNode = NULL
cdef void *__cuGraphAddEventRecordNode = NULL
cdef void *__cuGraphEventRecordNodeGetEvent = NULL
cdef void *__cuGraphEventRecordNodeSetEvent = NULL
cdef void *__cuGraphAddEventWaitNode = NULL
cdef void *__cuGraphEventWaitNodeGetEvent = NULL
cdef void *__cuGraphEventWaitNodeSetEvent = NULL
cdef void *__cuGraphAddExternalSemaphoresSignalNode = NULL
cdef void *__cuGraphExternalSemaphoresSignalNodeGetParams = NULL
cdef void *__cuGraphExternalSemaphoresSignalNodeSetParams = NULL
cdef void *__cuGraphAddExternalSemaphoresWaitNode = NULL
cdef void *__cuGraphExternalSemaphoresWaitNodeGetParams = NULL
cdef void *__cuGraphExternalSemaphoresWaitNodeSetParams = NULL
cdef void *__cuGraphAddBatchMemOpNode = NULL
cdef void *__cuGraphBatchMemOpNodeGetParams = NULL
cdef void *__cuGraphBatchMemOpNodeSetParams = NULL
cdef void *__cuGraphExecBatchMemOpNodeSetParams = NULL
cdef void *__cuGraphAddMemAllocNode = NULL
cdef void *__cuGraphMemAllocNodeGetParams = NULL
cdef void *__cuGraphAddMemFreeNode = NULL
cdef void *__cuGraphMemFreeNodeGetParams = NULL
cdef void *__cuDeviceGraphMemTrim = NULL
cdef void *__cuDeviceGetGraphMemAttribute = NULL
cdef void *__cuDeviceSetGraphMemAttribute = NULL
cdef void *__cuGraphClone = NULL
cdef void *__cuGraphNodeFindInClone = NULL
cdef void *__cuGraphNodeGetType = NULL
cdef void *__cuGraphGetNodes = NULL
cdef void *__cuGraphGetRootNodes = NULL
cdef void *__cuGraphGetEdges = NULL
cdef void *__cuGraphNodeGetDependencies = NULL
cdef void *__cuGraphNodeGetDependentNodes = NULL
cdef void *__cuGraphAddDependencies = NULL
cdef void *__cuGraphRemoveDependencies = NULL
cdef void *__cuGraphDestroyNode = NULL
cdef void *__cuGraphInstantiate_v2 = NULL
cdef void *__cuGraphInstantiateWithFlags = NULL
cdef void *__cuGraphExecKernelNodeSetParams = NULL
cdef void *__cuGraphExecMemcpyNodeSetParams = NULL
cdef void *__cuGraphExecMemsetNodeSetParams = NULL
cdef void *__cuGraphExecHostNodeSetParams = NULL
cdef void *__cuGraphExecChildGraphNodeSetParams = NULL
cdef void *__cuGraphExecEventRecordNodeSetEvent = NULL
cdef void *__cuGraphExecEventWaitNodeSetEvent = NULL
cdef void *__cuGraphExecExternalSemaphoresSignalNodeSetParams = NULL
cdef void *__cuGraphExecExternalSemaphoresWaitNodeSetParams = NULL
cdef void *__cuGraphNodeSetEnabled = NULL
cdef void *__cuGraphNodeGetEnabled = NULL
cdef void *__cuGraphUpload = NULL
cdef void *__cuGraphLaunch = NULL
cdef void *__cuGraphExecDestroy = NULL
cdef void *__cuGraphDestroy = NULL
cdef void *__cuGraphExecUpdate = NULL
cdef void *__cuGraphKernelNodeCopyAttributes = NULL
cdef void *__cuGraphKernelNodeGetAttribute = NULL
cdef void *__cuGraphKernelNodeSetAttribute = NULL
cdef void *__cuGraphDebugDotPrint = NULL
cdef void *__cuUserObjectCreate = NULL
cdef void *__cuUserObjectRetain = NULL
cdef void *__cuUserObjectRelease = NULL
cdef void *__cuGraphRetainUserObject = NULL
cdef void *__cuGraphReleaseUserObject = NULL
cdef void *__cuOccupancyMaxActiveBlocksPerMultiprocessor = NULL
cdef void *__cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags = NULL
cdef void *__cuOccupancyMaxPotentialBlockSize = NULL
cdef void *__cuOccupancyMaxPotentialBlockSizeWithFlags = NULL
cdef void *__cuOccupancyAvailableDynamicSMemPerBlock = NULL
cdef void *__cuTexRefSetArray = NULL
cdef void *__cuTexRefSetMipmappedArray = NULL
cdef void *__cuTexRefSetAddress_v2 = NULL
cdef void *__cuTexRefSetAddress2D_v3 = NULL
cdef void *__cuTexRefSetFormat = NULL
cdef void *__cuTexRefSetAddressMode = NULL
cdef void *__cuTexRefSetFilterMode = NULL
cdef void *__cuTexRefSetMipmapFilterMode = NULL
cdef void *__cuTexRefSetMipmapLevelBias = NULL
cdef void *__cuTexRefSetMipmapLevelClamp = NULL
cdef void *__cuTexRefSetMaxAnisotropy = NULL
cdef void *__cuTexRefSetBorderColor = NULL
cdef void *__cuTexRefSetFlags = NULL
cdef void *__cuTexRefGetAddress_v2 = NULL
cdef void *__cuTexRefGetArray = NULL
cdef void *__cuTexRefGetMipmappedArray = NULL
cdef void *__cuTexRefGetAddressMode = NULL
cdef void *__cuTexRefGetFilterMode = NULL
cdef void *__cuTexRefGetFormat = NULL
cdef void *__cuTexRefGetMipmapFilterMode = NULL
cdef void *__cuTexRefGetMipmapLevelBias = NULL
cdef void *__cuTexRefGetMipmapLevelClamp = NULL
cdef void *__cuTexRefGetMaxAnisotropy = NULL
cdef void *__cuTexRefGetBorderColor = NULL
cdef void *__cuTexRefGetFlags = NULL
cdef void *__cuTexRefCreate = NULL
cdef void *__cuTexRefDestroy = NULL
cdef void *__cuSurfRefSetArray = NULL
cdef void *__cuSurfRefGetArray = NULL
cdef void *__cuTexObjectCreate = NULL
cdef void *__cuTexObjectDestroy = NULL
cdef void *__cuTexObjectGetResourceDesc = NULL
cdef void *__cuTexObjectGetTextureDesc = NULL
cdef void *__cuTexObjectGetResourceViewDesc = NULL
cdef void *__cuSurfObjectCreate = NULL
cdef void *__cuSurfObjectDestroy = NULL
cdef void *__cuSurfObjectGetResourceDesc = NULL
cdef void *__cuDeviceCanAccessPeer = NULL
cdef void *__cuCtxEnablePeerAccess = NULL
cdef void *__cuCtxDisablePeerAccess = NULL
cdef void *__cuDeviceGetP2PAttribute = NULL
cdef void *__cuGraphicsUnregisterResource = NULL
cdef void *__cuGraphicsSubResourceGetMappedArray = NULL
cdef void *__cuGraphicsResourceGetMappedMipmappedArray = NULL
cdef void *__cuGraphicsResourceGetMappedPointer_v2 = NULL
cdef void *__cuGraphicsResourceSetMapFlags_v2 = NULL
cdef void *__cuGraphicsMapResources = NULL
cdef void *__cuGraphicsUnmapResources = NULL
cdef void *__cuGetProcAddress = NULL
cdef void *__cuModuleGetLoadingMode = NULL
cdef void *__cuMemGetHandleForAddressRange = NULL
cdef void *__cuGetExportTable = NULL
cdef void *__cuProfilerInitialize = NULL
cdef void *__cuProfilerStart = NULL
cdef void *__cuProfilerStop = NULL
cdef void *__cuVDPAUGetDevice = NULL
cdef void *__cuVDPAUCtxCreate_v2 = NULL
cdef void *__cuGraphicsVDPAURegisterVideoSurface = NULL
cdef void *__cuGraphicsVDPAURegisterOutputSurface = NULL
cdef void *__cuGraphicsEGLRegisterImage = NULL
cdef void *__cuEGLStreamConsumerConnect = NULL
cdef void *__cuEGLStreamConsumerConnectWithFlags = NULL
cdef void *__cuEGLStreamConsumerDisconnect = NULL
cdef void *__cuEGLStreamConsumerAcquireFrame = NULL
cdef void *__cuEGLStreamConsumerReleaseFrame = NULL
cdef void *__cuEGLStreamProducerConnect = NULL
cdef void *__cuEGLStreamProducerDisconnect = NULL
cdef void *__cuEGLStreamProducerPresentFrame = NULL
cdef void *__cuEGLStreamProducerReturnFrame = NULL
cdef void *__cuGraphicsResourceGetMappedEglFrame = NULL
cdef void *__cuEventCreateFromEGLSync = NULL
cdef void *__cuGraphicsGLRegisterBuffer = NULL
cdef void *__cuGraphicsGLRegisterImage = NULL
cdef void *__cuGLGetDevices_v2 = NULL

cdef int cuPythonInit() nogil except -1:
    global __cuPythonInit
    cdef bint usePTDS
    if __cuPythonInit:
        return 0
    __cuPythonInit = True
    with gil:
        usePTDS = os.getenv('CUDA_PYTHON_CUDA_PER_THREAD_DEFAULT_STREAM', default=0)
    cdef char libPath[260]
    libPath[0] = 0
    with gil:
        status = loader.getCUDALibraryPath(libPath, sys.maxsize > 2**32)
        if status == 0 and len(libPath) != 0:
            path = libPath.decode('utf-8')
        else:
            IF UNAME_SYSNAME == "Windows":
                path = 'nvcuda.dll'
            ELSE:
                path = 'libcuda.so'

        IF UNAME_SYSNAME == "Windows":
            LOAD_LIBRARY_SEARCH_SYSTEM32 = 0x00000800
            try:
                handle = win32api.LoadLibraryEx(path, 0, LOAD_LIBRARY_SEARCH_SYSTEM32)
            except error as e:
                raise RuntimeError('Failed to LoadLibraryEx ' + path)
        ELSE:
            handle = dlfcn.dlopen(bytes(path, encoding='utf-8'), dlfcn.RTLD_NOW)
            if (handle == NULL):
                raise RuntimeError('Failed to dlopen libcuda.so')
    global __cuGetErrorString
    global __cuGetErrorName
    global __cuInit
    global __cuDriverGetVersion
    global __cuDeviceGet
    global __cuDeviceGetCount
    global __cuDeviceGetName
    global __cuDeviceGetUuid
    global __cuDeviceGetUuid_v2
    global __cuDeviceGetLuid
    global __cuDeviceTotalMem_v2
    global __cuDeviceGetTexture1DLinearMaxWidth
    global __cuDeviceGetAttribute
    global __cuDeviceGetNvSciSyncAttributes
    global __cuDeviceSetMemPool
    global __cuDeviceGetMemPool
    global __cuDeviceGetDefaultMemPool
    global __cuFlushGPUDirectRDMAWrites
    global __cuDeviceGetProperties
    global __cuDeviceComputeCapability
    global __cuDevicePrimaryCtxRetain
    global __cuDevicePrimaryCtxRelease_v2
    global __cuDevicePrimaryCtxSetFlags_v2
    global __cuDevicePrimaryCtxGetState
    global __cuDevicePrimaryCtxReset_v2
    global __cuDeviceGetExecAffinitySupport
    global __cuCtxCreate_v2
    global __cuCtxCreate_v3
    global __cuCtxDestroy_v2
    global __cuCtxPushCurrent_v2
    global __cuCtxPopCurrent_v2
    global __cuCtxSetCurrent
    global __cuCtxGetCurrent
    global __cuCtxGetDevice
    global __cuCtxGetFlags
    global __cuCtxSynchronize
    global __cuCtxSetLimit
    global __cuCtxGetLimit
    global __cuCtxGetCacheConfig
    global __cuCtxSetCacheConfig
    global __cuCtxGetSharedMemConfig
    global __cuCtxSetSharedMemConfig
    global __cuCtxGetApiVersion
    global __cuCtxGetStreamPriorityRange
    global __cuCtxResetPersistingL2Cache
    global __cuCtxGetExecAffinity
    global __cuCtxAttach
    global __cuCtxDetach
    global __cuModuleLoad
    global __cuModuleLoadData
    global __cuModuleLoadDataEx
    global __cuModuleLoadFatBinary
    global __cuModuleUnload
    global __cuModuleGetFunction
    global __cuModuleGetGlobal_v2
    global __cuModuleGetTexRef
    global __cuModuleGetSurfRef
    global __cuLinkCreate_v2
    global __cuLinkAddData_v2
    global __cuLinkAddFile_v2
    global __cuLinkComplete
    global __cuLinkDestroy
    global __cuMemGetInfo_v2
    global __cuMemAlloc_v2
    global __cuMemAllocPitch_v2
    global __cuMemFree_v2
    global __cuMemGetAddressRange_v2
    global __cuMemAllocHost_v2
    global __cuMemFreeHost
    global __cuMemHostAlloc
    global __cuMemHostGetDevicePointer_v2
    global __cuMemHostGetFlags
    global __cuMemAllocManaged
    global __cuDeviceGetByPCIBusId
    global __cuDeviceGetPCIBusId
    global __cuIpcGetEventHandle
    global __cuIpcOpenEventHandle
    global __cuIpcGetMemHandle
    global __cuIpcOpenMemHandle_v2
    global __cuIpcCloseMemHandle
    global __cuMemHostRegister_v2
    global __cuMemHostUnregister
    global __cuMemcpy
    global __cuMemcpyPeer
    global __cuMemcpyHtoD_v2
    global __cuMemcpyDtoH_v2
    global __cuMemcpyDtoD_v2
    global __cuMemcpyDtoA_v2
    global __cuMemcpyAtoD_v2
    global __cuMemcpyHtoA_v2
    global __cuMemcpyAtoH_v2
    global __cuMemcpyAtoA_v2
    global __cuMemcpy2D_v2
    global __cuMemcpy2DUnaligned_v2
    global __cuMemcpy3D_v2
    global __cuMemcpy3DPeer
    global __cuMemcpyAsync
    global __cuMemcpyPeerAsync
    global __cuMemcpyHtoDAsync_v2
    global __cuMemcpyDtoHAsync_v2
    global __cuMemcpyDtoDAsync_v2
    global __cuMemcpyHtoAAsync_v2
    global __cuMemcpyAtoHAsync_v2
    global __cuMemcpy2DAsync_v2
    global __cuMemcpy3DAsync_v2
    global __cuMemcpy3DPeerAsync
    global __cuMemsetD8_v2
    global __cuMemsetD16_v2
    global __cuMemsetD32_v2
    global __cuMemsetD2D8_v2
    global __cuMemsetD2D16_v2
    global __cuMemsetD2D32_v2
    global __cuMemsetD8Async
    global __cuMemsetD16Async
    global __cuMemsetD32Async
    global __cuMemsetD2D8Async
    global __cuMemsetD2D16Async
    global __cuMemsetD2D32Async
    global __cuArrayCreate_v2
    global __cuArrayGetDescriptor_v2
    global __cuArrayGetSparseProperties
    global __cuMipmappedArrayGetSparseProperties
    global __cuArrayGetMemoryRequirements
    global __cuMipmappedArrayGetMemoryRequirements
    global __cuArrayGetPlane
    global __cuArrayDestroy
    global __cuArray3DCreate_v2
    global __cuArray3DGetDescriptor_v2
    global __cuMipmappedArrayCreate
    global __cuMipmappedArrayGetLevel
    global __cuMipmappedArrayDestroy
    global __cuMemAddressReserve
    global __cuMemAddressFree
    global __cuMemCreate
    global __cuMemRelease
    global __cuMemMap
    global __cuMemMapArrayAsync
    global __cuMemUnmap
    global __cuMemSetAccess
    global __cuMemGetAccess
    global __cuMemExportToShareableHandle
    global __cuMemImportFromShareableHandle
    global __cuMemGetAllocationGranularity
    global __cuMemGetAllocationPropertiesFromHandle
    global __cuMemRetainAllocationHandle
    global __cuMemFreeAsync
    global __cuMemAllocAsync
    global __cuMemPoolTrimTo
    global __cuMemPoolSetAttribute
    global __cuMemPoolGetAttribute
    global __cuMemPoolSetAccess
    global __cuMemPoolGetAccess
    global __cuMemPoolCreate
    global __cuMemPoolDestroy
    global __cuMemAllocFromPoolAsync
    global __cuMemPoolExportToShareableHandle
    global __cuMemPoolImportFromShareableHandle
    global __cuMemPoolExportPointer
    global __cuMemPoolImportPointer
    global __cuPointerGetAttribute
    global __cuMemPrefetchAsync
    global __cuMemAdvise
    global __cuMemRangeGetAttribute
    global __cuMemRangeGetAttributes
    global __cuPointerSetAttribute
    global __cuPointerGetAttributes
    global __cuStreamCreate
    global __cuStreamCreateWithPriority
    global __cuStreamGetPriority
    global __cuStreamGetFlags
    global __cuStreamGetCtx
    global __cuStreamWaitEvent
    global __cuStreamAddCallback
    global __cuStreamBeginCapture_v2
    global __cuThreadExchangeStreamCaptureMode
    global __cuStreamEndCapture
    global __cuStreamIsCapturing
    global __cuStreamGetCaptureInfo
    global __cuStreamGetCaptureInfo_v2
    global __cuStreamUpdateCaptureDependencies
    global __cuStreamAttachMemAsync
    global __cuStreamQuery
    global __cuStreamSynchronize
    global __cuStreamDestroy_v2
    global __cuStreamCopyAttributes
    global __cuStreamGetAttribute
    global __cuStreamSetAttribute
    global __cuEventCreate
    global __cuEventRecord
    global __cuEventRecordWithFlags
    global __cuEventQuery
    global __cuEventSynchronize
    global __cuEventDestroy_v2
    global __cuEventElapsedTime
    global __cuImportExternalMemory
    global __cuExternalMemoryGetMappedBuffer
    global __cuExternalMemoryGetMappedMipmappedArray
    global __cuDestroyExternalMemory
    global __cuImportExternalSemaphore
    global __cuSignalExternalSemaphoresAsync
    global __cuWaitExternalSemaphoresAsync
    global __cuDestroyExternalSemaphore
    global __cuStreamWaitValue32
    global __cuStreamWaitValue64
    global __cuStreamWriteValue32
    global __cuStreamWriteValue64
    global __cuStreamBatchMemOp
    global __cuStreamWaitValue32_v2
    global __cuStreamWaitValue64_v2
    global __cuStreamWriteValue32_v2
    global __cuStreamWriteValue64_v2
    global __cuStreamBatchMemOp_v2
    global __cuFuncGetAttribute
    global __cuFuncSetAttribute
    global __cuFuncSetCacheConfig
    global __cuFuncSetSharedMemConfig
    global __cuFuncGetModule
    global __cuLaunchKernel
    global __cuLaunchCooperativeKernel
    global __cuLaunchCooperativeKernelMultiDevice
    global __cuLaunchHostFunc
    global __cuFuncSetBlockShape
    global __cuFuncSetSharedSize
    global __cuParamSetSize
    global __cuParamSeti
    global __cuParamSetf
    global __cuParamSetv
    global __cuLaunch
    global __cuLaunchGrid
    global __cuLaunchGridAsync
    global __cuParamSetTexRef
    global __cuGraphCreate
    global __cuGraphAddKernelNode
    global __cuGraphKernelNodeGetParams
    global __cuGraphKernelNodeSetParams
    global __cuGraphAddMemcpyNode
    global __cuGraphMemcpyNodeGetParams
    global __cuGraphMemcpyNodeSetParams
    global __cuGraphAddMemsetNode
    global __cuGraphMemsetNodeGetParams
    global __cuGraphMemsetNodeSetParams
    global __cuGraphAddHostNode
    global __cuGraphHostNodeGetParams
    global __cuGraphHostNodeSetParams
    global __cuGraphAddChildGraphNode
    global __cuGraphChildGraphNodeGetGraph
    global __cuGraphAddEmptyNode
    global __cuGraphAddEventRecordNode
    global __cuGraphEventRecordNodeGetEvent
    global __cuGraphEventRecordNodeSetEvent
    global __cuGraphAddEventWaitNode
    global __cuGraphEventWaitNodeGetEvent
    global __cuGraphEventWaitNodeSetEvent
    global __cuGraphAddExternalSemaphoresSignalNode
    global __cuGraphExternalSemaphoresSignalNodeGetParams
    global __cuGraphExternalSemaphoresSignalNodeSetParams
    global __cuGraphAddExternalSemaphoresWaitNode
    global __cuGraphExternalSemaphoresWaitNodeGetParams
    global __cuGraphExternalSemaphoresWaitNodeSetParams
    global __cuGraphAddBatchMemOpNode
    global __cuGraphBatchMemOpNodeGetParams
    global __cuGraphBatchMemOpNodeSetParams
    global __cuGraphExecBatchMemOpNodeSetParams
    global __cuGraphAddMemAllocNode
    global __cuGraphMemAllocNodeGetParams
    global __cuGraphAddMemFreeNode
    global __cuGraphMemFreeNodeGetParams
    global __cuDeviceGraphMemTrim
    global __cuDeviceGetGraphMemAttribute
    global __cuDeviceSetGraphMemAttribute
    global __cuGraphClone
    global __cuGraphNodeFindInClone
    global __cuGraphNodeGetType
    global __cuGraphGetNodes
    global __cuGraphGetRootNodes
    global __cuGraphGetEdges
    global __cuGraphNodeGetDependencies
    global __cuGraphNodeGetDependentNodes
    global __cuGraphAddDependencies
    global __cuGraphRemoveDependencies
    global __cuGraphDestroyNode
    global __cuGraphInstantiate_v2
    global __cuGraphInstantiateWithFlags
    global __cuGraphExecKernelNodeSetParams
    global __cuGraphExecMemcpyNodeSetParams
    global __cuGraphExecMemsetNodeSetParams
    global __cuGraphExecHostNodeSetParams
    global __cuGraphExecChildGraphNodeSetParams
    global __cuGraphExecEventRecordNodeSetEvent
    global __cuGraphExecEventWaitNodeSetEvent
    global __cuGraphExecExternalSemaphoresSignalNodeSetParams
    global __cuGraphExecExternalSemaphoresWaitNodeSetParams
    global __cuGraphNodeSetEnabled
    global __cuGraphNodeGetEnabled
    global __cuGraphUpload
    global __cuGraphLaunch
    global __cuGraphExecDestroy
    global __cuGraphDestroy
    global __cuGraphExecUpdate
    global __cuGraphKernelNodeCopyAttributes
    global __cuGraphKernelNodeGetAttribute
    global __cuGraphKernelNodeSetAttribute
    global __cuGraphDebugDotPrint
    global __cuUserObjectCreate
    global __cuUserObjectRetain
    global __cuUserObjectRelease
    global __cuGraphRetainUserObject
    global __cuGraphReleaseUserObject
    global __cuOccupancyMaxActiveBlocksPerMultiprocessor
    global __cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags
    global __cuOccupancyMaxPotentialBlockSize
    global __cuOccupancyMaxPotentialBlockSizeWithFlags
    global __cuOccupancyAvailableDynamicSMemPerBlock
    global __cuTexRefSetArray
    global __cuTexRefSetMipmappedArray
    global __cuTexRefSetAddress_v2
    global __cuTexRefSetAddress2D_v3
    global __cuTexRefSetFormat
    global __cuTexRefSetAddressMode
    global __cuTexRefSetFilterMode
    global __cuTexRefSetMipmapFilterMode
    global __cuTexRefSetMipmapLevelBias
    global __cuTexRefSetMipmapLevelClamp
    global __cuTexRefSetMaxAnisotropy
    global __cuTexRefSetBorderColor
    global __cuTexRefSetFlags
    global __cuTexRefGetAddress_v2
    global __cuTexRefGetArray
    global __cuTexRefGetMipmappedArray
    global __cuTexRefGetAddressMode
    global __cuTexRefGetFilterMode
    global __cuTexRefGetFormat
    global __cuTexRefGetMipmapFilterMode
    global __cuTexRefGetMipmapLevelBias
    global __cuTexRefGetMipmapLevelClamp
    global __cuTexRefGetMaxAnisotropy
    global __cuTexRefGetBorderColor
    global __cuTexRefGetFlags
    global __cuTexRefCreate
    global __cuTexRefDestroy
    global __cuSurfRefSetArray
    global __cuSurfRefGetArray
    global __cuTexObjectCreate
    global __cuTexObjectDestroy
    global __cuTexObjectGetResourceDesc
    global __cuTexObjectGetTextureDesc
    global __cuTexObjectGetResourceViewDesc
    global __cuSurfObjectCreate
    global __cuSurfObjectDestroy
    global __cuSurfObjectGetResourceDesc
    global __cuDeviceCanAccessPeer
    global __cuCtxEnablePeerAccess
    global __cuCtxDisablePeerAccess
    global __cuDeviceGetP2PAttribute
    global __cuGraphicsUnregisterResource
    global __cuGraphicsSubResourceGetMappedArray
    global __cuGraphicsResourceGetMappedMipmappedArray
    global __cuGraphicsResourceGetMappedPointer_v2
    global __cuGraphicsResourceSetMapFlags_v2
    global __cuGraphicsMapResources
    global __cuGraphicsUnmapResources
    global __cuGetProcAddress
    global __cuModuleGetLoadingMode
    global __cuMemGetHandleForAddressRange
    global __cuGetExportTable
    global __cuProfilerInitialize
    global __cuProfilerStart
    global __cuProfilerStop
    global __cuVDPAUGetDevice
    global __cuVDPAUCtxCreate_v2
    global __cuGraphicsVDPAURegisterVideoSurface
    global __cuGraphicsVDPAURegisterOutputSurface
    global __cuGraphicsEGLRegisterImage
    global __cuEGLStreamConsumerConnect
    global __cuEGLStreamConsumerConnectWithFlags
    global __cuEGLStreamConsumerDisconnect
    global __cuEGLStreamConsumerAcquireFrame
    global __cuEGLStreamConsumerReleaseFrame
    global __cuEGLStreamProducerConnect
    global __cuEGLStreamProducerDisconnect
    global __cuEGLStreamProducerPresentFrame
    global __cuEGLStreamProducerReturnFrame
    global __cuGraphicsResourceGetMappedEglFrame
    global __cuEventCreateFromEGLSync
    global __cuGraphicsGLRegisterBuffer
    global __cuGraphicsGLRegisterImage
    global __cuGLGetDevices_v2
    # Get latest __cuGetProcAddress
    IF UNAME_SYSNAME == "Windows":
        with gil:
            try:
                __cuGetProcAddress = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuGetProcAddress')
            except:
                pass
    ELSE:
        __cuGetProcAddress = dlfcn.dlsym(handle, 'cuGetProcAddress')

    if __cuGetProcAddress != NULL:
        if usePTDS:
            # Get all PTDS version of functions
            cuGetProcAddress('cuMemcpy', &__cuMemcpy, 7000, CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM)
            cuGetProcAddress('cuMemcpyPeer', &__cuMemcpyPeer, 7000, CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM)
            cuGetProcAddress('cuMemcpyHtoD', &__cuMemcpyHtoD_v2, 7000, CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM)
            cuGetProcAddress('cuMemcpyDtoH', &__cuMemcpyDtoH_v2, 7000, CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM)
            cuGetProcAddress('cuMemcpyDtoD', &__cuMemcpyDtoD_v2, 7000, CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM)
            cuGetProcAddress('cuMemcpyDtoA', &__cuMemcpyDtoA_v2, 7000, CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM)
            cuGetProcAddress('cuMemcpyAtoD', &__cuMemcpyAtoD_v2, 7000, CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM)
            cuGetProcAddress('cuMemcpyHtoA', &__cuMemcpyHtoA_v2, 7000, CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM)
            cuGetProcAddress('cuMemcpyAtoH', &__cuMemcpyAtoH_v2, 7000, CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM)
            cuGetProcAddress('cuMemcpyAtoA', &__cuMemcpyAtoA_v2, 7000, CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM)
            cuGetProcAddress('cuMemcpy2D', &__cuMemcpy2D_v2, 7000, CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM)
            cuGetProcAddress('cuMemcpy2DUnaligned', &__cuMemcpy2DUnaligned_v2, 7000, CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM)
            cuGetProcAddress('cuMemcpy3D', &__cuMemcpy3D_v2, 7000, CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM)
            cuGetProcAddress('cuMemcpy3DPeer', &__cuMemcpy3DPeer, 7000, CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM)
            cuGetProcAddress('cuMemcpyAsync', &__cuMemcpyAsync, 7000, CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM)
            cuGetProcAddress('cuMemcpyPeerAsync', &__cuMemcpyPeerAsync, 7000, CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM)
            cuGetProcAddress('cuMemcpyHtoDAsync', &__cuMemcpyHtoDAsync_v2, 7000, CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM)
            cuGetProcAddress('cuMemcpyDtoHAsync', &__cuMemcpyDtoHAsync_v2, 7000, CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM)
            cuGetProcAddress('cuMemcpyDtoDAsync', &__cuMemcpyDtoDAsync_v2, 7000, CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM)
            cuGetProcAddress('cuMemcpyHtoAAsync', &__cuMemcpyHtoAAsync_v2, 7000, CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM)
            cuGetProcAddress('cuMemcpyAtoHAsync', &__cuMemcpyAtoHAsync_v2, 7000, CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM)
            cuGetProcAddress('cuMemcpy2DAsync', &__cuMemcpy2DAsync_v2, 7000, CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM)
            cuGetProcAddress('cuMemcpy3DAsync', &__cuMemcpy3DAsync_v2, 7000, CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM)
            cuGetProcAddress('cuMemcpy3DPeerAsync', &__cuMemcpy3DPeerAsync, 7000, CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM)
            cuGetProcAddress('cuMemsetD8', &__cuMemsetD8_v2, 7000, CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM)
            cuGetProcAddress('cuMemsetD16', &__cuMemsetD16_v2, 7000, CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM)
            cuGetProcAddress('cuMemsetD32', &__cuMemsetD32_v2, 7000, CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM)
            cuGetProcAddress('cuMemsetD2D8', &__cuMemsetD2D8_v2, 7000, CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM)
            cuGetProcAddress('cuMemsetD2D16', &__cuMemsetD2D16_v2, 7000, CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM)
            cuGetProcAddress('cuMemsetD2D32', &__cuMemsetD2D32_v2, 7000, CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM)
            cuGetProcAddress('cuMemsetD8Async', &__cuMemsetD8Async, 7000, CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM)
            cuGetProcAddress('cuMemsetD16Async', &__cuMemsetD16Async, 7000, CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM)
            cuGetProcAddress('cuMemsetD32Async', &__cuMemsetD32Async, 7000, CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM)
            cuGetProcAddress('cuMemsetD2D8Async', &__cuMemsetD2D8Async, 7000, CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM)
            cuGetProcAddress('cuMemsetD2D16Async', &__cuMemsetD2D16Async, 7000, CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM)
            cuGetProcAddress('cuMemsetD2D32Async', &__cuMemsetD2D32Async, 7000, CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM)
            cuGetProcAddress('cuMemMapArrayAsync', &__cuMemMapArrayAsync, 11010, CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM)
            cuGetProcAddress('cuMemFreeAsync', &__cuMemFreeAsync, 11020, CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM)
            cuGetProcAddress('cuMemAllocAsync', &__cuMemAllocAsync, 11020, CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM)
            cuGetProcAddress('cuMemAllocFromPoolAsync', &__cuMemAllocFromPoolAsync, 11020, CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM)
            cuGetProcAddress('cuMemPrefetchAsync', &__cuMemPrefetchAsync, 8000, CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM)
            cuGetProcAddress('cuStreamGetPriority', &__cuStreamGetPriority, 7000, CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM)
            cuGetProcAddress('cuStreamGetFlags', &__cuStreamGetFlags, 7000, CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM)
            cuGetProcAddress('cuStreamGetCtx', &__cuStreamGetCtx, 9020, CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM)
            cuGetProcAddress('cuStreamWaitEvent', &__cuStreamWaitEvent, 7000, CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM)
            cuGetProcAddress('cuStreamAddCallback', &__cuStreamAddCallback, 7000, CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM)
            cuGetProcAddress('cuStreamBeginCapture', &__cuStreamBeginCapture_v2, 10010, CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM)
            cuGetProcAddress('cuStreamEndCapture', &__cuStreamEndCapture, 10000, CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM)
            cuGetProcAddress('cuStreamIsCapturing', &__cuStreamIsCapturing, 10000, CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM)
            cuGetProcAddress('cuStreamGetCaptureInfo', &__cuStreamGetCaptureInfo, 10010, CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM)
            cuGetProcAddress('cuStreamGetCaptureInfo', &__cuStreamGetCaptureInfo_v2, 11030, CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM)
            cuGetProcAddress('cuStreamUpdateCaptureDependencies', &__cuStreamUpdateCaptureDependencies, 11030, CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM)
            cuGetProcAddress('cuStreamAttachMemAsync', &__cuStreamAttachMemAsync, 7000, CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM)
            cuGetProcAddress('cuStreamQuery', &__cuStreamQuery, 7000, CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM)
            cuGetProcAddress('cuStreamSynchronize', &__cuStreamSynchronize, 7000, CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM)
            cuGetProcAddress('cuStreamCopyAttributes', &__cuStreamCopyAttributes, 11000, CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM)
            cuGetProcAddress('cuStreamGetAttribute', &__cuStreamGetAttribute, 11000, CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM)
            cuGetProcAddress('cuStreamSetAttribute', &__cuStreamSetAttribute, 11000, CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM)
            cuGetProcAddress('cuEventRecord', &__cuEventRecord, 7000, CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM)
            cuGetProcAddress('cuEventRecordWithFlags', &__cuEventRecordWithFlags, 11010, CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM)
            cuGetProcAddress('cuSignalExternalSemaphoresAsync', &__cuSignalExternalSemaphoresAsync, 10000, CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM)
            cuGetProcAddress('cuWaitExternalSemaphoresAsync', &__cuWaitExternalSemaphoresAsync, 10000, CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM)
            cuGetProcAddress('cuStreamWaitValue32', &__cuStreamWaitValue32, 8000, CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM)
            cuGetProcAddress('cuStreamWaitValue64', &__cuStreamWaitValue64, 9000, CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM)
            cuGetProcAddress('cuStreamWriteValue32', &__cuStreamWriteValue32, 8000, CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM)
            cuGetProcAddress('cuStreamWriteValue64', &__cuStreamWriteValue64, 9000, CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM)
            cuGetProcAddress('cuStreamBatchMemOp', &__cuStreamBatchMemOp, 8000, CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM)
            cuGetProcAddress('cuStreamWaitValue32', &__cuStreamWaitValue32_v2, 11070, CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM)
            cuGetProcAddress('cuStreamWaitValue64', &__cuStreamWaitValue64_v2, 11070, CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM)
            cuGetProcAddress('cuStreamWriteValue32', &__cuStreamWriteValue32_v2, 11070, CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM)
            cuGetProcAddress('cuStreamWriteValue64', &__cuStreamWriteValue64_v2, 11070, CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM)
            cuGetProcAddress('cuStreamBatchMemOp', &__cuStreamBatchMemOp_v2, 11070, CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM)
            cuGetProcAddress('cuLaunchKernel', &__cuLaunchKernel, 7000, CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM)
            cuGetProcAddress('cuLaunchCooperativeKernel', &__cuLaunchCooperativeKernel, 9000, CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM)
            cuGetProcAddress('cuLaunchHostFunc', &__cuLaunchHostFunc, 10000, CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM)
            cuGetProcAddress('cuGraphUpload', &__cuGraphUpload, 11010, CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM)
            cuGetProcAddress('cuGraphLaunch', &__cuGraphLaunch, 10000, CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM)
            cuGetProcAddress('cuGraphicsMapResources', &__cuGraphicsMapResources, 7000, CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM)
            cuGetProcAddress('cuGraphicsUnmapResources', &__cuGraphicsUnmapResources, 7000, CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM)
        else:
            # Else get the regular version
            cuGetProcAddress('cuMemcpy', &__cuMemcpy, 4000, CU_GET_PROC_ADDRESS_DEFAULT)
            cuGetProcAddress('cuMemcpyPeer', &__cuMemcpyPeer, 4000, CU_GET_PROC_ADDRESS_DEFAULT)
            cuGetProcAddress('cuMemcpyHtoD', &__cuMemcpyHtoD_v2, 3020, CU_GET_PROC_ADDRESS_DEFAULT)
            cuGetProcAddress('cuMemcpyDtoH', &__cuMemcpyDtoH_v2, 3020, CU_GET_PROC_ADDRESS_DEFAULT)
            cuGetProcAddress('cuMemcpyDtoD', &__cuMemcpyDtoD_v2, 3020, CU_GET_PROC_ADDRESS_DEFAULT)
            cuGetProcAddress('cuMemcpyDtoA', &__cuMemcpyDtoA_v2, 3020, CU_GET_PROC_ADDRESS_DEFAULT)
            cuGetProcAddress('cuMemcpyAtoD', &__cuMemcpyAtoD_v2, 3020, CU_GET_PROC_ADDRESS_DEFAULT)
            cuGetProcAddress('cuMemcpyHtoA', &__cuMemcpyHtoA_v2, 3020, CU_GET_PROC_ADDRESS_DEFAULT)
            cuGetProcAddress('cuMemcpyAtoH', &__cuMemcpyAtoH_v2, 3020, CU_GET_PROC_ADDRESS_DEFAULT)
            cuGetProcAddress('cuMemcpyAtoA', &__cuMemcpyAtoA_v2, 3020, CU_GET_PROC_ADDRESS_DEFAULT)
            cuGetProcAddress('cuMemcpy2D', &__cuMemcpy2D_v2, 3020, CU_GET_PROC_ADDRESS_DEFAULT)
            cuGetProcAddress('cuMemcpy2DUnaligned', &__cuMemcpy2DUnaligned_v2, 3020, CU_GET_PROC_ADDRESS_DEFAULT)
            cuGetProcAddress('cuMemcpy3D', &__cuMemcpy3D_v2, 3020, CU_GET_PROC_ADDRESS_DEFAULT)
            cuGetProcAddress('cuMemcpy3DPeer', &__cuMemcpy3DPeer, 4000, CU_GET_PROC_ADDRESS_DEFAULT)
            cuGetProcAddress('cuMemcpyAsync', &__cuMemcpyAsync, 4000, CU_GET_PROC_ADDRESS_DEFAULT)
            cuGetProcAddress('cuMemcpyPeerAsync', &__cuMemcpyPeerAsync, 4000, CU_GET_PROC_ADDRESS_DEFAULT)
            cuGetProcAddress('cuMemcpyHtoDAsync', &__cuMemcpyHtoDAsync_v2, 3020, CU_GET_PROC_ADDRESS_DEFAULT)
            cuGetProcAddress('cuMemcpyDtoHAsync', &__cuMemcpyDtoHAsync_v2, 3020, CU_GET_PROC_ADDRESS_DEFAULT)
            cuGetProcAddress('cuMemcpyDtoDAsync', &__cuMemcpyDtoDAsync_v2, 3020, CU_GET_PROC_ADDRESS_DEFAULT)
            cuGetProcAddress('cuMemcpyHtoAAsync', &__cuMemcpyHtoAAsync_v2, 3020, CU_GET_PROC_ADDRESS_DEFAULT)
            cuGetProcAddress('cuMemcpyAtoHAsync', &__cuMemcpyAtoHAsync_v2, 3020, CU_GET_PROC_ADDRESS_DEFAULT)
            cuGetProcAddress('cuMemcpy2DAsync', &__cuMemcpy2DAsync_v2, 3020, CU_GET_PROC_ADDRESS_DEFAULT)
            cuGetProcAddress('cuMemcpy3DAsync', &__cuMemcpy3DAsync_v2, 3020, CU_GET_PROC_ADDRESS_DEFAULT)
            cuGetProcAddress('cuMemcpy3DPeerAsync', &__cuMemcpy3DPeerAsync, 4000, CU_GET_PROC_ADDRESS_DEFAULT)
            cuGetProcAddress('cuMemsetD8', &__cuMemsetD8_v2, 3020, CU_GET_PROC_ADDRESS_DEFAULT)
            cuGetProcAddress('cuMemsetD16', &__cuMemsetD16_v2, 3020, CU_GET_PROC_ADDRESS_DEFAULT)
            cuGetProcAddress('cuMemsetD32', &__cuMemsetD32_v2, 3020, CU_GET_PROC_ADDRESS_DEFAULT)
            cuGetProcAddress('cuMemsetD2D8', &__cuMemsetD2D8_v2, 3020, CU_GET_PROC_ADDRESS_DEFAULT)
            cuGetProcAddress('cuMemsetD2D16', &__cuMemsetD2D16_v2, 3020, CU_GET_PROC_ADDRESS_DEFAULT)
            cuGetProcAddress('cuMemsetD2D32', &__cuMemsetD2D32_v2, 3020, CU_GET_PROC_ADDRESS_DEFAULT)
            cuGetProcAddress('cuMemsetD8Async', &__cuMemsetD8Async, 3020, CU_GET_PROC_ADDRESS_DEFAULT)
            cuGetProcAddress('cuMemsetD16Async', &__cuMemsetD16Async, 3020, CU_GET_PROC_ADDRESS_DEFAULT)
            cuGetProcAddress('cuMemsetD32Async', &__cuMemsetD32Async, 3020, CU_GET_PROC_ADDRESS_DEFAULT)
            cuGetProcAddress('cuMemsetD2D8Async', &__cuMemsetD2D8Async, 3020, CU_GET_PROC_ADDRESS_DEFAULT)
            cuGetProcAddress('cuMemsetD2D16Async', &__cuMemsetD2D16Async, 3020, CU_GET_PROC_ADDRESS_DEFAULT)
            cuGetProcAddress('cuMemsetD2D32Async', &__cuMemsetD2D32Async, 3020, CU_GET_PROC_ADDRESS_DEFAULT)
            cuGetProcAddress('cuMemMapArrayAsync', &__cuMemMapArrayAsync, 11010, CU_GET_PROC_ADDRESS_DEFAULT)
            cuGetProcAddress('cuMemFreeAsync', &__cuMemFreeAsync, 11020, CU_GET_PROC_ADDRESS_DEFAULT)
            cuGetProcAddress('cuMemAllocAsync', &__cuMemAllocAsync, 11020, CU_GET_PROC_ADDRESS_DEFAULT)
            cuGetProcAddress('cuMemAllocFromPoolAsync', &__cuMemAllocFromPoolAsync, 11020, CU_GET_PROC_ADDRESS_DEFAULT)
            cuGetProcAddress('cuMemPrefetchAsync', &__cuMemPrefetchAsync, 8000, CU_GET_PROC_ADDRESS_DEFAULT)
            cuGetProcAddress('cuStreamGetPriority', &__cuStreamGetPriority, 5050, CU_GET_PROC_ADDRESS_DEFAULT)
            cuGetProcAddress('cuStreamGetFlags', &__cuStreamGetFlags, 5050, CU_GET_PROC_ADDRESS_DEFAULT)
            cuGetProcAddress('cuStreamGetCtx', &__cuStreamGetCtx, 9020, CU_GET_PROC_ADDRESS_DEFAULT)
            cuGetProcAddress('cuStreamWaitEvent', &__cuStreamWaitEvent, 3020, CU_GET_PROC_ADDRESS_DEFAULT)
            cuGetProcAddress('cuStreamAddCallback', &__cuStreamAddCallback, 5000, CU_GET_PROC_ADDRESS_DEFAULT)
            cuGetProcAddress('cuStreamBeginCapture', &__cuStreamBeginCapture_v2, 10010, CU_GET_PROC_ADDRESS_DEFAULT)
            cuGetProcAddress('cuStreamEndCapture', &__cuStreamEndCapture, 10000, CU_GET_PROC_ADDRESS_DEFAULT)
            cuGetProcAddress('cuStreamIsCapturing', &__cuStreamIsCapturing, 10000, CU_GET_PROC_ADDRESS_DEFAULT)
            cuGetProcAddress('cuStreamGetCaptureInfo', &__cuStreamGetCaptureInfo, 10010, CU_GET_PROC_ADDRESS_DEFAULT)
            cuGetProcAddress('cuStreamGetCaptureInfo', &__cuStreamGetCaptureInfo_v2, 11030, CU_GET_PROC_ADDRESS_DEFAULT)
            cuGetProcAddress('cuStreamUpdateCaptureDependencies', &__cuStreamUpdateCaptureDependencies, 11030, CU_GET_PROC_ADDRESS_DEFAULT)
            cuGetProcAddress('cuStreamAttachMemAsync', &__cuStreamAttachMemAsync, 6000, CU_GET_PROC_ADDRESS_DEFAULT)
            cuGetProcAddress('cuStreamQuery', &__cuStreamQuery, 2000, CU_GET_PROC_ADDRESS_DEFAULT)
            cuGetProcAddress('cuStreamSynchronize', &__cuStreamSynchronize, 2000, CU_GET_PROC_ADDRESS_DEFAULT)
            cuGetProcAddress('cuStreamCopyAttributes', &__cuStreamCopyAttributes, 11000, CU_GET_PROC_ADDRESS_DEFAULT)
            cuGetProcAddress('cuStreamGetAttribute', &__cuStreamGetAttribute, 11000, CU_GET_PROC_ADDRESS_DEFAULT)
            cuGetProcAddress('cuStreamSetAttribute', &__cuStreamSetAttribute, 11000, CU_GET_PROC_ADDRESS_DEFAULT)
            cuGetProcAddress('cuEventRecord', &__cuEventRecord, 2000, CU_GET_PROC_ADDRESS_DEFAULT)
            cuGetProcAddress('cuEventRecordWithFlags', &__cuEventRecordWithFlags, 11010, CU_GET_PROC_ADDRESS_DEFAULT)
            cuGetProcAddress('cuSignalExternalSemaphoresAsync', &__cuSignalExternalSemaphoresAsync, 10000, CU_GET_PROC_ADDRESS_DEFAULT)
            cuGetProcAddress('cuWaitExternalSemaphoresAsync', &__cuWaitExternalSemaphoresAsync, 10000, CU_GET_PROC_ADDRESS_DEFAULT)
            cuGetProcAddress('cuStreamWaitValue32', &__cuStreamWaitValue32, 8000, CU_GET_PROC_ADDRESS_DEFAULT)
            cuGetProcAddress('cuStreamWaitValue64', &__cuStreamWaitValue64, 9000, CU_GET_PROC_ADDRESS_DEFAULT)
            cuGetProcAddress('cuStreamWriteValue32', &__cuStreamWriteValue32, 8000, CU_GET_PROC_ADDRESS_DEFAULT)
            cuGetProcAddress('cuStreamWriteValue64', &__cuStreamWriteValue64, 9000, CU_GET_PROC_ADDRESS_DEFAULT)
            cuGetProcAddress('cuStreamBatchMemOp', &__cuStreamBatchMemOp, 8000, CU_GET_PROC_ADDRESS_DEFAULT)
            cuGetProcAddress('cuStreamWaitValue32', &__cuStreamWaitValue32_v2, 11070, CU_GET_PROC_ADDRESS_DEFAULT)
            cuGetProcAddress('cuStreamWaitValue64', &__cuStreamWaitValue64_v2, 11070, CU_GET_PROC_ADDRESS_DEFAULT)
            cuGetProcAddress('cuStreamWriteValue32', &__cuStreamWriteValue32_v2, 11070, CU_GET_PROC_ADDRESS_DEFAULT)
            cuGetProcAddress('cuStreamWriteValue64', &__cuStreamWriteValue64_v2, 11070, CU_GET_PROC_ADDRESS_DEFAULT)
            cuGetProcAddress('cuStreamBatchMemOp', &__cuStreamBatchMemOp_v2, 11070, CU_GET_PROC_ADDRESS_DEFAULT)
            cuGetProcAddress('cuLaunchKernel', &__cuLaunchKernel, 4000, CU_GET_PROC_ADDRESS_DEFAULT)
            cuGetProcAddress('cuLaunchCooperativeKernel', &__cuLaunchCooperativeKernel, 9000, CU_GET_PROC_ADDRESS_DEFAULT)
            cuGetProcAddress('cuLaunchHostFunc', &__cuLaunchHostFunc, 10000, CU_GET_PROC_ADDRESS_DEFAULT)
            cuGetProcAddress('cuGraphUpload', &__cuGraphUpload, 11010, CU_GET_PROC_ADDRESS_DEFAULT)
            cuGetProcAddress('cuGraphLaunch', &__cuGraphLaunch, 10000, CU_GET_PROC_ADDRESS_DEFAULT)
            cuGetProcAddress('cuGraphicsMapResources', &__cuGraphicsMapResources, 3000, CU_GET_PROC_ADDRESS_DEFAULT)
            cuGetProcAddress('cuGraphicsUnmapResources', &__cuGraphicsUnmapResources, 3000, CU_GET_PROC_ADDRESS_DEFAULT)
        # Get remaining functions
        cuGetProcAddress('cuGetErrorString', &__cuGetErrorString, 6000, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuGetErrorName', &__cuGetErrorName, 6000, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuInit', &__cuInit, 2000, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuDriverGetVersion', &__cuDriverGetVersion, 2020, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuDeviceGet', &__cuDeviceGet, 2000, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuDeviceGetCount', &__cuDeviceGetCount, 2000, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuDeviceGetName', &__cuDeviceGetName, 2000, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuDeviceGetUuid', &__cuDeviceGetUuid, 9020, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuDeviceGetUuid', &__cuDeviceGetUuid_v2, 11040, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuDeviceGetLuid', &__cuDeviceGetLuid, 10000, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuDeviceTotalMem', &__cuDeviceTotalMem_v2, 3020, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuDeviceGetTexture1DLinearMaxWidth', &__cuDeviceGetTexture1DLinearMaxWidth, 11010, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuDeviceGetAttribute', &__cuDeviceGetAttribute, 2000, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuDeviceGetNvSciSyncAttributes', &__cuDeviceGetNvSciSyncAttributes, 10020, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuDeviceSetMemPool', &__cuDeviceSetMemPool, 11020, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuDeviceGetMemPool', &__cuDeviceGetMemPool, 11020, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuDeviceGetDefaultMemPool', &__cuDeviceGetDefaultMemPool, 11020, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuFlushGPUDirectRDMAWrites', &__cuFlushGPUDirectRDMAWrites, 11030, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuDeviceGetProperties', &__cuDeviceGetProperties, 2000, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuDeviceComputeCapability', &__cuDeviceComputeCapability, 2000, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuDevicePrimaryCtxRetain', &__cuDevicePrimaryCtxRetain, 7000, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuDevicePrimaryCtxRelease', &__cuDevicePrimaryCtxRelease_v2, 11000, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuDevicePrimaryCtxSetFlags', &__cuDevicePrimaryCtxSetFlags_v2, 11000, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuDevicePrimaryCtxGetState', &__cuDevicePrimaryCtxGetState, 7000, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuDevicePrimaryCtxReset', &__cuDevicePrimaryCtxReset_v2, 11000, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuDeviceGetExecAffinitySupport', &__cuDeviceGetExecAffinitySupport, 11040, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuCtxCreate', &__cuCtxCreate_v2, 3020, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuCtxCreate', &__cuCtxCreate_v3, 11040, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuCtxDestroy', &__cuCtxDestroy_v2, 4000, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuCtxPushCurrent', &__cuCtxPushCurrent_v2, 4000, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuCtxPopCurrent', &__cuCtxPopCurrent_v2, 4000, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuCtxSetCurrent', &__cuCtxSetCurrent, 4000, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuCtxGetCurrent', &__cuCtxGetCurrent, 4000, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuCtxGetDevice', &__cuCtxGetDevice, 2000, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuCtxGetFlags', &__cuCtxGetFlags, 7000, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuCtxSynchronize', &__cuCtxSynchronize, 2000, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuCtxSetLimit', &__cuCtxSetLimit, 3010, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuCtxGetLimit', &__cuCtxGetLimit, 3010, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuCtxGetCacheConfig', &__cuCtxGetCacheConfig, 3020, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuCtxSetCacheConfig', &__cuCtxSetCacheConfig, 3020, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuCtxGetSharedMemConfig', &__cuCtxGetSharedMemConfig, 4020, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuCtxSetSharedMemConfig', &__cuCtxSetSharedMemConfig, 4020, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuCtxGetApiVersion', &__cuCtxGetApiVersion, 3020, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuCtxGetStreamPriorityRange', &__cuCtxGetStreamPriorityRange, 5050, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuCtxResetPersistingL2Cache', &__cuCtxResetPersistingL2Cache, 11000, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuCtxGetExecAffinity', &__cuCtxGetExecAffinity, 11040, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuCtxAttach', &__cuCtxAttach, 2000, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuCtxDetach', &__cuCtxDetach, 2000, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuModuleLoad', &__cuModuleLoad, 2000, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuModuleLoadData', &__cuModuleLoadData, 2000, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuModuleLoadDataEx', &__cuModuleLoadDataEx, 2010, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuModuleLoadFatBinary', &__cuModuleLoadFatBinary, 2000, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuModuleUnload', &__cuModuleUnload, 2000, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuModuleGetFunction', &__cuModuleGetFunction, 2000, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuModuleGetGlobal', &__cuModuleGetGlobal_v2, 3020, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuModuleGetTexRef', &__cuModuleGetTexRef, 2000, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuModuleGetSurfRef', &__cuModuleGetSurfRef, 3000, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuLinkCreate', &__cuLinkCreate_v2, 6050, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuLinkAddData', &__cuLinkAddData_v2, 6050, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuLinkAddFile', &__cuLinkAddFile_v2, 6050, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuLinkComplete', &__cuLinkComplete, 5050, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuLinkDestroy', &__cuLinkDestroy, 5050, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuMemGetInfo', &__cuMemGetInfo_v2, 3020, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuMemAlloc', &__cuMemAlloc_v2, 3020, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuMemAllocPitch', &__cuMemAllocPitch_v2, 3020, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuMemFree', &__cuMemFree_v2, 3020, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuMemGetAddressRange', &__cuMemGetAddressRange_v2, 3020, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuMemAllocHost', &__cuMemAllocHost_v2, 3020, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuMemFreeHost', &__cuMemFreeHost, 2000, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuMemHostAlloc', &__cuMemHostAlloc, 2020, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuMemHostGetDevicePointer', &__cuMemHostGetDevicePointer_v2, 3020, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuMemHostGetFlags', &__cuMemHostGetFlags, 2030, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuMemAllocManaged', &__cuMemAllocManaged, 6000, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuDeviceGetByPCIBusId', &__cuDeviceGetByPCIBusId, 4010, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuDeviceGetPCIBusId', &__cuDeviceGetPCIBusId, 4010, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuIpcGetEventHandle', &__cuIpcGetEventHandle, 4010, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuIpcOpenEventHandle', &__cuIpcOpenEventHandle, 4010, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuIpcGetMemHandle', &__cuIpcGetMemHandle, 4010, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuIpcOpenMemHandle', &__cuIpcOpenMemHandle_v2, 11000, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuIpcCloseMemHandle', &__cuIpcCloseMemHandle, 4010, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuMemHostRegister', &__cuMemHostRegister_v2, 6050, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuMemHostUnregister', &__cuMemHostUnregister, 4000, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuArrayCreate', &__cuArrayCreate_v2, 3020, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuArrayGetDescriptor', &__cuArrayGetDescriptor_v2, 3020, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuArrayGetSparseProperties', &__cuArrayGetSparseProperties, 11010, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuMipmappedArrayGetSparseProperties', &__cuMipmappedArrayGetSparseProperties, 11010, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuArrayGetMemoryRequirements', &__cuArrayGetMemoryRequirements, 11060, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuMipmappedArrayGetMemoryRequirements', &__cuMipmappedArrayGetMemoryRequirements, 11060, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuArrayGetPlane', &__cuArrayGetPlane, 11020, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuArrayDestroy', &__cuArrayDestroy, 2000, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuArray3DCreate', &__cuArray3DCreate_v2, 3020, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuArray3DGetDescriptor', &__cuArray3DGetDescriptor_v2, 3020, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuMipmappedArrayCreate', &__cuMipmappedArrayCreate, 5000, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuMipmappedArrayGetLevel', &__cuMipmappedArrayGetLevel, 5000, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuMipmappedArrayDestroy', &__cuMipmappedArrayDestroy, 5000, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuMemAddressReserve', &__cuMemAddressReserve, 10020, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuMemAddressFree', &__cuMemAddressFree, 10020, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuMemCreate', &__cuMemCreate, 10020, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuMemRelease', &__cuMemRelease, 10020, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuMemMap', &__cuMemMap, 10020, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuMemUnmap', &__cuMemUnmap, 10020, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuMemSetAccess', &__cuMemSetAccess, 10020, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuMemGetAccess', &__cuMemGetAccess, 10020, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuMemExportToShareableHandle', &__cuMemExportToShareableHandle, 10020, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuMemImportFromShareableHandle', &__cuMemImportFromShareableHandle, 10020, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuMemGetAllocationGranularity', &__cuMemGetAllocationGranularity, 10020, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuMemGetAllocationPropertiesFromHandle', &__cuMemGetAllocationPropertiesFromHandle, 10020, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuMemRetainAllocationHandle', &__cuMemRetainAllocationHandle, 11000, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuMemPoolTrimTo', &__cuMemPoolTrimTo, 11020, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuMemPoolSetAttribute', &__cuMemPoolSetAttribute, 11020, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuMemPoolGetAttribute', &__cuMemPoolGetAttribute, 11020, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuMemPoolSetAccess', &__cuMemPoolSetAccess, 11020, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuMemPoolGetAccess', &__cuMemPoolGetAccess, 11020, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuMemPoolCreate', &__cuMemPoolCreate, 11020, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuMemPoolDestroy', &__cuMemPoolDestroy, 11020, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuMemPoolExportToShareableHandle', &__cuMemPoolExportToShareableHandle, 11020, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuMemPoolImportFromShareableHandle', &__cuMemPoolImportFromShareableHandle, 11020, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuMemPoolExportPointer', &__cuMemPoolExportPointer, 11020, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuMemPoolImportPointer', &__cuMemPoolImportPointer, 11020, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuPointerGetAttribute', &__cuPointerGetAttribute, 4000, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuMemAdvise', &__cuMemAdvise, 8000, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuMemRangeGetAttribute', &__cuMemRangeGetAttribute, 8000, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuMemRangeGetAttributes', &__cuMemRangeGetAttributes, 8000, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuPointerSetAttribute', &__cuPointerSetAttribute, 6000, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuPointerGetAttributes', &__cuPointerGetAttributes, 7000, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuStreamCreate', &__cuStreamCreate, 2000, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuStreamCreateWithPriority', &__cuStreamCreateWithPriority, 5050, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuThreadExchangeStreamCaptureMode', &__cuThreadExchangeStreamCaptureMode, 10010, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuStreamDestroy', &__cuStreamDestroy_v2, 4000, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuEventCreate', &__cuEventCreate, 2000, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuEventQuery', &__cuEventQuery, 2000, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuEventSynchronize', &__cuEventSynchronize, 2000, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuEventDestroy', &__cuEventDestroy_v2, 4000, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuEventElapsedTime', &__cuEventElapsedTime, 2000, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuImportExternalMemory', &__cuImportExternalMemory, 10000, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuExternalMemoryGetMappedBuffer', &__cuExternalMemoryGetMappedBuffer, 10000, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuExternalMemoryGetMappedMipmappedArray', &__cuExternalMemoryGetMappedMipmappedArray, 10000, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuDestroyExternalMemory', &__cuDestroyExternalMemory, 10000, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuImportExternalSemaphore', &__cuImportExternalSemaphore, 10000, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuDestroyExternalSemaphore', &__cuDestroyExternalSemaphore, 10000, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuFuncGetAttribute', &__cuFuncGetAttribute, 2020, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuFuncSetAttribute', &__cuFuncSetAttribute, 9000, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuFuncSetCacheConfig', &__cuFuncSetCacheConfig, 3000, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuFuncSetSharedMemConfig', &__cuFuncSetSharedMemConfig, 4020, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuFuncGetModule', &__cuFuncGetModule, 11000, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuLaunchCooperativeKernelMultiDevice', &__cuLaunchCooperativeKernelMultiDevice, 9000, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuFuncSetBlockShape', &__cuFuncSetBlockShape, 2000, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuFuncSetSharedSize', &__cuFuncSetSharedSize, 2000, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuParamSetSize', &__cuParamSetSize, 2000, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuParamSeti', &__cuParamSeti, 2000, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuParamSetf', &__cuParamSetf, 2000, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuParamSetv', &__cuParamSetv, 2000, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuLaunch', &__cuLaunch, 2000, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuLaunchGrid', &__cuLaunchGrid, 2000, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuLaunchGridAsync', &__cuLaunchGridAsync, 2000, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuParamSetTexRef', &__cuParamSetTexRef, 2000, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuGraphCreate', &__cuGraphCreate, 10000, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuGraphAddKernelNode', &__cuGraphAddKernelNode, 10000, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuGraphKernelNodeGetParams', &__cuGraphKernelNodeGetParams, 10000, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuGraphKernelNodeSetParams', &__cuGraphKernelNodeSetParams, 10000, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuGraphAddMemcpyNode', &__cuGraphAddMemcpyNode, 10000, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuGraphMemcpyNodeGetParams', &__cuGraphMemcpyNodeGetParams, 10000, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuGraphMemcpyNodeSetParams', &__cuGraphMemcpyNodeSetParams, 10000, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuGraphAddMemsetNode', &__cuGraphAddMemsetNode, 10000, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuGraphMemsetNodeGetParams', &__cuGraphMemsetNodeGetParams, 10000, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuGraphMemsetNodeSetParams', &__cuGraphMemsetNodeSetParams, 10000, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuGraphAddHostNode', &__cuGraphAddHostNode, 10000, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuGraphHostNodeGetParams', &__cuGraphHostNodeGetParams, 10000, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuGraphHostNodeSetParams', &__cuGraphHostNodeSetParams, 10000, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuGraphAddChildGraphNode', &__cuGraphAddChildGraphNode, 10000, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuGraphChildGraphNodeGetGraph', &__cuGraphChildGraphNodeGetGraph, 10000, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuGraphAddEmptyNode', &__cuGraphAddEmptyNode, 10000, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuGraphAddEventRecordNode', &__cuGraphAddEventRecordNode, 11010, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuGraphEventRecordNodeGetEvent', &__cuGraphEventRecordNodeGetEvent, 11010, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuGraphEventRecordNodeSetEvent', &__cuGraphEventRecordNodeSetEvent, 11010, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuGraphAddEventWaitNode', &__cuGraphAddEventWaitNode, 11010, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuGraphEventWaitNodeGetEvent', &__cuGraphEventWaitNodeGetEvent, 11010, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuGraphEventWaitNodeSetEvent', &__cuGraphEventWaitNodeSetEvent, 11010, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuGraphAddExternalSemaphoresSignalNode', &__cuGraphAddExternalSemaphoresSignalNode, 11020, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuGraphExternalSemaphoresSignalNodeGetParams', &__cuGraphExternalSemaphoresSignalNodeGetParams, 11020, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuGraphExternalSemaphoresSignalNodeSetParams', &__cuGraphExternalSemaphoresSignalNodeSetParams, 11020, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuGraphAddExternalSemaphoresWaitNode', &__cuGraphAddExternalSemaphoresWaitNode, 11020, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuGraphExternalSemaphoresWaitNodeGetParams', &__cuGraphExternalSemaphoresWaitNodeGetParams, 11020, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuGraphExternalSemaphoresWaitNodeSetParams', &__cuGraphExternalSemaphoresWaitNodeSetParams, 11020, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuGraphAddBatchMemOpNode', &__cuGraphAddBatchMemOpNode, 11070, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuGraphBatchMemOpNodeGetParams', &__cuGraphBatchMemOpNodeGetParams, 11070, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuGraphBatchMemOpNodeSetParams', &__cuGraphBatchMemOpNodeSetParams, 11070, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuGraphExecBatchMemOpNodeSetParams', &__cuGraphExecBatchMemOpNodeSetParams, 11070, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuGraphAddMemAllocNode', &__cuGraphAddMemAllocNode, 11040, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuGraphMemAllocNodeGetParams', &__cuGraphMemAllocNodeGetParams, 11040, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuGraphAddMemFreeNode', &__cuGraphAddMemFreeNode, 11040, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuGraphMemFreeNodeGetParams', &__cuGraphMemFreeNodeGetParams, 11040, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuDeviceGraphMemTrim', &__cuDeviceGraphMemTrim, 11040, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuDeviceGetGraphMemAttribute', &__cuDeviceGetGraphMemAttribute, 11040, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuDeviceSetGraphMemAttribute', &__cuDeviceSetGraphMemAttribute, 11040, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuGraphClone', &__cuGraphClone, 10000, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuGraphNodeFindInClone', &__cuGraphNodeFindInClone, 10000, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuGraphNodeGetType', &__cuGraphNodeGetType, 10000, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuGraphGetNodes', &__cuGraphGetNodes, 10000, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuGraphGetRootNodes', &__cuGraphGetRootNodes, 10000, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuGraphGetEdges', &__cuGraphGetEdges, 10000, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuGraphNodeGetDependencies', &__cuGraphNodeGetDependencies, 10000, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuGraphNodeGetDependentNodes', &__cuGraphNodeGetDependentNodes, 10000, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuGraphAddDependencies', &__cuGraphAddDependencies, 10000, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuGraphRemoveDependencies', &__cuGraphRemoveDependencies, 10000, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuGraphDestroyNode', &__cuGraphDestroyNode, 10000, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuGraphInstantiate', &__cuGraphInstantiate_v2, 11000, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuGraphInstantiateWithFlags', &__cuGraphInstantiateWithFlags, 11040, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuGraphExecKernelNodeSetParams', &__cuGraphExecKernelNodeSetParams, 10010, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuGraphExecMemcpyNodeSetParams', &__cuGraphExecMemcpyNodeSetParams, 10020, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuGraphExecMemsetNodeSetParams', &__cuGraphExecMemsetNodeSetParams, 10020, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuGraphExecHostNodeSetParams', &__cuGraphExecHostNodeSetParams, 10020, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuGraphExecChildGraphNodeSetParams', &__cuGraphExecChildGraphNodeSetParams, 11010, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuGraphExecEventRecordNodeSetEvent', &__cuGraphExecEventRecordNodeSetEvent, 11010, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuGraphExecEventWaitNodeSetEvent', &__cuGraphExecEventWaitNodeSetEvent, 11010, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuGraphExecExternalSemaphoresSignalNodeSetParams', &__cuGraphExecExternalSemaphoresSignalNodeSetParams, 11020, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuGraphExecExternalSemaphoresWaitNodeSetParams', &__cuGraphExecExternalSemaphoresWaitNodeSetParams, 11020, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuGraphNodeSetEnabled', &__cuGraphNodeSetEnabled, 11060, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuGraphNodeGetEnabled', &__cuGraphNodeGetEnabled, 11060, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuGraphExecDestroy', &__cuGraphExecDestroy, 10000, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuGraphDestroy', &__cuGraphDestroy, 10000, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuGraphExecUpdate', &__cuGraphExecUpdate, 10020, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuGraphKernelNodeCopyAttributes', &__cuGraphKernelNodeCopyAttributes, 11000, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuGraphKernelNodeGetAttribute', &__cuGraphKernelNodeGetAttribute, 11000, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuGraphKernelNodeSetAttribute', &__cuGraphKernelNodeSetAttribute, 11000, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuGraphDebugDotPrint', &__cuGraphDebugDotPrint, 11030, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuUserObjectCreate', &__cuUserObjectCreate, 11030, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuUserObjectRetain', &__cuUserObjectRetain, 11030, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuUserObjectRelease', &__cuUserObjectRelease, 11030, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuGraphRetainUserObject', &__cuGraphRetainUserObject, 11030, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuGraphReleaseUserObject', &__cuGraphReleaseUserObject, 11030, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuOccupancyMaxActiveBlocksPerMultiprocessor', &__cuOccupancyMaxActiveBlocksPerMultiprocessor, 6050, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags', &__cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags, 7000, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuOccupancyMaxPotentialBlockSize', &__cuOccupancyMaxPotentialBlockSize, 6050, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuOccupancyMaxPotentialBlockSizeWithFlags', &__cuOccupancyMaxPotentialBlockSizeWithFlags, 7000, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuOccupancyAvailableDynamicSMemPerBlock', &__cuOccupancyAvailableDynamicSMemPerBlock, 10020, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuTexRefSetArray', &__cuTexRefSetArray, 2000, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuTexRefSetMipmappedArray', &__cuTexRefSetMipmappedArray, 5000, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuTexRefSetAddress', &__cuTexRefSetAddress_v2, 3020, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuTexRefSetAddress2D', &__cuTexRefSetAddress2D_v3, 4010, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuTexRefSetFormat', &__cuTexRefSetFormat, 2000, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuTexRefSetAddressMode', &__cuTexRefSetAddressMode, 2000, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuTexRefSetFilterMode', &__cuTexRefSetFilterMode, 2000, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuTexRefSetMipmapFilterMode', &__cuTexRefSetMipmapFilterMode, 5000, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuTexRefSetMipmapLevelBias', &__cuTexRefSetMipmapLevelBias, 5000, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuTexRefSetMipmapLevelClamp', &__cuTexRefSetMipmapLevelClamp, 5000, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuTexRefSetMaxAnisotropy', &__cuTexRefSetMaxAnisotropy, 5000, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuTexRefSetBorderColor', &__cuTexRefSetBorderColor, 8000, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuTexRefSetFlags', &__cuTexRefSetFlags, 2000, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuTexRefGetAddress', &__cuTexRefGetAddress_v2, 3020, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuTexRefGetArray', &__cuTexRefGetArray, 2000, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuTexRefGetMipmappedArray', &__cuTexRefGetMipmappedArray, 5000, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuTexRefGetAddressMode', &__cuTexRefGetAddressMode, 2000, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuTexRefGetFilterMode', &__cuTexRefGetFilterMode, 2000, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuTexRefGetFormat', &__cuTexRefGetFormat, 2000, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuTexRefGetMipmapFilterMode', &__cuTexRefGetMipmapFilterMode, 5000, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuTexRefGetMipmapLevelBias', &__cuTexRefGetMipmapLevelBias, 5000, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuTexRefGetMipmapLevelClamp', &__cuTexRefGetMipmapLevelClamp, 5000, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuTexRefGetMaxAnisotropy', &__cuTexRefGetMaxAnisotropy, 5000, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuTexRefGetBorderColor', &__cuTexRefGetBorderColor, 8000, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuTexRefGetFlags', &__cuTexRefGetFlags, 2000, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuTexRefCreate', &__cuTexRefCreate, 2000, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuTexRefDestroy', &__cuTexRefDestroy, 2000, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuSurfRefSetArray', &__cuSurfRefSetArray, 3000, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuSurfRefGetArray', &__cuSurfRefGetArray, 3000, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuTexObjectCreate', &__cuTexObjectCreate, 5000, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuTexObjectDestroy', &__cuTexObjectDestroy, 5000, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuTexObjectGetResourceDesc', &__cuTexObjectGetResourceDesc, 5000, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuTexObjectGetTextureDesc', &__cuTexObjectGetTextureDesc, 5000, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuTexObjectGetResourceViewDesc', &__cuTexObjectGetResourceViewDesc, 5000, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuSurfObjectCreate', &__cuSurfObjectCreate, 5000, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuSurfObjectDestroy', &__cuSurfObjectDestroy, 5000, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuSurfObjectGetResourceDesc', &__cuSurfObjectGetResourceDesc, 5000, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuDeviceCanAccessPeer', &__cuDeviceCanAccessPeer, 4000, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuCtxEnablePeerAccess', &__cuCtxEnablePeerAccess, 4000, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuCtxDisablePeerAccess', &__cuCtxDisablePeerAccess, 4000, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuDeviceGetP2PAttribute', &__cuDeviceGetP2PAttribute, 8000, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuGraphicsUnregisterResource', &__cuGraphicsUnregisterResource, 3000, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuGraphicsSubResourceGetMappedArray', &__cuGraphicsSubResourceGetMappedArray, 3000, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuGraphicsResourceGetMappedMipmappedArray', &__cuGraphicsResourceGetMappedMipmappedArray, 5000, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuGraphicsResourceGetMappedPointer', &__cuGraphicsResourceGetMappedPointer_v2, 3020, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuGraphicsResourceSetMapFlags', &__cuGraphicsResourceSetMapFlags_v2, 6050, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuGetProcAddress', &__cuGetProcAddress, 11030, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuModuleGetLoadingMode', &__cuModuleGetLoadingMode, 11070, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuMemGetHandleForAddressRange', &__cuMemGetHandleForAddressRange, 11070, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuGetExportTable', &__cuGetExportTable, 3000, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuProfilerInitialize', &__cuProfilerInitialize, 4000, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuProfilerStart', &__cuProfilerStart, 4000, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuProfilerStop', &__cuProfilerStop, 4000, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuVDPAUGetDevice', &__cuVDPAUGetDevice, 3010, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuVDPAUCtxCreate', &__cuVDPAUCtxCreate_v2, 3020, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuGraphicsVDPAURegisterVideoSurface', &__cuGraphicsVDPAURegisterVideoSurface, 3010, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuGraphicsVDPAURegisterOutputSurface', &__cuGraphicsVDPAURegisterOutputSurface, 3010, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuGraphicsEGLRegisterImage', &__cuGraphicsEGLRegisterImage, 7000, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuEGLStreamConsumerConnect', &__cuEGLStreamConsumerConnect, 7000, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuEGLStreamConsumerConnectWithFlags', &__cuEGLStreamConsumerConnectWithFlags, 8000, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuEGLStreamConsumerDisconnect', &__cuEGLStreamConsumerDisconnect, 7000, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuEGLStreamConsumerAcquireFrame', &__cuEGLStreamConsumerAcquireFrame, 7000, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuEGLStreamConsumerReleaseFrame', &__cuEGLStreamConsumerReleaseFrame, 7000, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuEGLStreamProducerConnect', &__cuEGLStreamProducerConnect, 7000, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuEGLStreamProducerDisconnect', &__cuEGLStreamProducerDisconnect, 7000, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuEGLStreamProducerPresentFrame', &__cuEGLStreamProducerPresentFrame, 7000, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuEGLStreamProducerReturnFrame', &__cuEGLStreamProducerReturnFrame, 7000, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuGraphicsResourceGetMappedEglFrame', &__cuGraphicsResourceGetMappedEglFrame, 7000, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuEventCreateFromEGLSync', &__cuEventCreateFromEGLSync, 9000, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuGraphicsGLRegisterBuffer', &__cuGraphicsGLRegisterBuffer, 3000, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuGraphicsGLRegisterImage', &__cuGraphicsGLRegisterImage, 3000, CU_GET_PROC_ADDRESS_DEFAULT)
        cuGetProcAddress('cuGLGetDevices', &__cuGLGetDevices_v2, 6050, CU_GET_PROC_ADDRESS_DEFAULT)
        return 0
    # dlsym calls
    IF UNAME_SYSNAME == "Windows":
        with gil:
            if usePTDS:
                # Get all PTDS version of functions
                try:
                    __cuMemcpy = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuMemcpy_ptds')
                except:
                    pass
                try:
                    __cuMemcpyPeer = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuMemcpyPeer_ptds')
                except:
                    pass
                try:
                    __cuMemcpyHtoD_v2 = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuMemcpyHtoD_v2_ptds')
                except:
                    pass
                try:
                    __cuMemcpyDtoH_v2 = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuMemcpyDtoH_v2_ptds')
                except:
                    pass
                try:
                    __cuMemcpyDtoD_v2 = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuMemcpyDtoD_v2_ptds')
                except:
                    pass
                try:
                    __cuMemcpyDtoA_v2 = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuMemcpyDtoA_v2_ptds')
                except:
                    pass
                try:
                    __cuMemcpyAtoD_v2 = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuMemcpyAtoD_v2_ptds')
                except:
                    pass
                try:
                    __cuMemcpyHtoA_v2 = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuMemcpyHtoA_v2_ptds')
                except:
                    pass
                try:
                    __cuMemcpyAtoH_v2 = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuMemcpyAtoH_v2_ptds')
                except:
                    pass
                try:
                    __cuMemcpyAtoA_v2 = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuMemcpyAtoA_v2_ptds')
                except:
                    pass
                try:
                    __cuMemcpy2D_v2 = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuMemcpy2D_v2_ptds')
                except:
                    pass
                try:
                    __cuMemcpy2DUnaligned_v2 = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuMemcpy2DUnaligned_v2_ptds')
                except:
                    pass
                try:
                    __cuMemcpy3D_v2 = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuMemcpy3D_v2_ptds')
                except:
                    pass
                try:
                    __cuMemcpy3DPeer = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuMemcpy3DPeer_ptds')
                except:
                    pass
                try:
                    __cuMemcpyAsync = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuMemcpyAsync_ptsz')
                except:
                    pass
                try:
                    __cuMemcpyPeerAsync = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuMemcpyPeerAsync_ptsz')
                except:
                    pass
                try:
                    __cuMemcpyHtoDAsync_v2 = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuMemcpyHtoDAsync_v2_ptsz')
                except:
                    pass
                try:
                    __cuMemcpyDtoHAsync_v2 = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuMemcpyDtoHAsync_v2_ptsz')
                except:
                    pass
                try:
                    __cuMemcpyDtoDAsync_v2 = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuMemcpyDtoDAsync_v2_ptsz')
                except:
                    pass
                try:
                    __cuMemcpyHtoAAsync_v2 = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuMemcpyHtoAAsync_v2_ptsz')
                except:
                    pass
                try:
                    __cuMemcpyAtoHAsync_v2 = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuMemcpyAtoHAsync_v2_ptsz')
                except:
                    pass
                try:
                    __cuMemcpy2DAsync_v2 = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuMemcpy2DAsync_v2_ptsz')
                except:
                    pass
                try:
                    __cuMemcpy3DAsync_v2 = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuMemcpy3DAsync_v2_ptsz')
                except:
                    pass
                try:
                    __cuMemcpy3DPeerAsync = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuMemcpy3DPeerAsync_ptsz')
                except:
                    pass
                try:
                    __cuMemsetD8_v2 = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuMemsetD8_v2_ptds')
                except:
                    pass
                try:
                    __cuMemsetD16_v2 = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuMemsetD16_v2_ptds')
                except:
                    pass
                try:
                    __cuMemsetD32_v2 = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuMemsetD32_v2_ptds')
                except:
                    pass
                try:
                    __cuMemsetD2D8_v2 = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuMemsetD2D8_v2_ptds')
                except:
                    pass
                try:
                    __cuMemsetD2D16_v2 = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuMemsetD2D16_v2_ptds')
                except:
                    pass
                try:
                    __cuMemsetD2D32_v2 = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuMemsetD2D32_v2_ptds')
                except:
                    pass
                try:
                    __cuMemsetD8Async = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuMemsetD8Async_ptsz')
                except:
                    pass
                try:
                    __cuMemsetD16Async = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuMemsetD16Async_ptsz')
                except:
                    pass
                try:
                    __cuMemsetD32Async = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuMemsetD32Async_ptsz')
                except:
                    pass
                try:
                    __cuMemsetD2D8Async = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuMemsetD2D8Async_ptsz')
                except:
                    pass
                try:
                    __cuMemsetD2D16Async = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuMemsetD2D16Async_ptsz')
                except:
                    pass
                try:
                    __cuMemsetD2D32Async = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuMemsetD2D32Async_ptsz')
                except:
                    pass
                try:
                    __cuMemMapArrayAsync = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuMemMapArrayAsync_ptsz')
                except:
                    pass
                try:
                    __cuMemFreeAsync = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuMemFreeAsync_ptsz')
                except:
                    pass
                try:
                    __cuMemAllocAsync = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuMemAllocAsync_ptsz')
                except:
                    pass
                try:
                    __cuMemAllocFromPoolAsync = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuMemAllocFromPoolAsync_ptsz')
                except:
                    pass
                try:
                    __cuMemPrefetchAsync = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuMemPrefetchAsync_ptsz')
                except:
                    pass
                try:
                    __cuStreamGetPriority = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuStreamGetPriority_ptsz')
                except:
                    pass
                try:
                    __cuStreamGetFlags = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuStreamGetFlags_ptsz')
                except:
                    pass
                try:
                    __cuStreamGetCtx = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuStreamGetCtx_ptsz')
                except:
                    pass
                try:
                    __cuStreamWaitEvent = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuStreamWaitEvent_ptsz')
                except:
                    pass
                try:
                    __cuStreamAddCallback = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuStreamAddCallback_ptsz')
                except:
                    pass
                try:
                    __cuStreamBeginCapture_v2 = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuStreamBeginCapture_v2_ptsz')
                except:
                    pass
                try:
                    __cuStreamEndCapture = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuStreamEndCapture_ptsz')
                except:
                    pass
                try:
                    __cuStreamIsCapturing = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuStreamIsCapturing_ptsz')
                except:
                    pass
                try:
                    __cuStreamGetCaptureInfo = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuStreamGetCaptureInfo_ptsz')
                except:
                    pass
                try:
                    __cuStreamGetCaptureInfo_v2 = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuStreamGetCaptureInfo_v2_ptsz')
                except:
                    pass
                try:
                    __cuStreamUpdateCaptureDependencies = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuStreamUpdateCaptureDependencies_ptsz')
                except:
                    pass
                try:
                    __cuStreamAttachMemAsync = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuStreamAttachMemAsync_ptsz')
                except:
                    pass
                try:
                    __cuStreamQuery = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuStreamQuery_ptsz')
                except:
                    pass
                try:
                    __cuStreamSynchronize = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuStreamSynchronize_ptsz')
                except:
                    pass
                try:
                    __cuStreamCopyAttributes = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuStreamCopyAttributes_ptsz')
                except:
                    pass
                try:
                    __cuStreamGetAttribute = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuStreamGetAttribute_ptsz')
                except:
                    pass
                try:
                    __cuStreamSetAttribute = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuStreamSetAttribute_ptsz')
                except:
                    pass
                try:
                    __cuEventRecord = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuEventRecord_ptsz')
                except:
                    pass
                try:
                    __cuEventRecordWithFlags = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuEventRecordWithFlags_ptsz')
                except:
                    pass
                try:
                    __cuSignalExternalSemaphoresAsync = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuSignalExternalSemaphoresAsync_ptsz')
                except:
                    pass
                try:
                    __cuWaitExternalSemaphoresAsync = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuWaitExternalSemaphoresAsync_ptsz')
                except:
                    pass
                try:
                    __cuStreamWaitValue32 = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuStreamWaitValue32_ptsz')
                except:
                    pass
                try:
                    __cuStreamWaitValue64 = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuStreamWaitValue64_ptsz')
                except:
                    pass
                try:
                    __cuStreamWriteValue32 = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuStreamWriteValue32_ptsz')
                except:
                    pass
                try:
                    __cuStreamWriteValue64 = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuStreamWriteValue64_ptsz')
                except:
                    pass
                try:
                    __cuStreamBatchMemOp = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuStreamBatchMemOp_ptsz')
                except:
                    pass
                try:
                    __cuStreamWaitValue32_v2 = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuStreamWaitValue32_v2_ptsz')
                except:
                    pass
                try:
                    __cuStreamWaitValue64_v2 = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuStreamWaitValue64_v2_ptsz')
                except:
                    pass
                try:
                    __cuStreamWriteValue32_v2 = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuStreamWriteValue32_v2_ptsz')
                except:
                    pass
                try:
                    __cuStreamWriteValue64_v2 = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuStreamWriteValue64_v2_ptsz')
                except:
                    pass
                try:
                    __cuStreamBatchMemOp_v2 = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuStreamBatchMemOp_v2_ptsz')
                except:
                    pass
                try:
                    __cuLaunchKernel = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuLaunchKernel_ptsz')
                except:
                    pass
                try:
                    __cuLaunchCooperativeKernel = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuLaunchCooperativeKernel_ptsz')
                except:
                    pass
                try:
                    __cuLaunchHostFunc = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuLaunchHostFunc_ptsz')
                except:
                    pass
                try:
                    __cuGraphUpload = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuGraphUpload_ptsz')
                except:
                    pass
                try:
                    __cuGraphLaunch = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuGraphLaunch_ptsz')
                except:
                    pass
                try:
                    __cuGraphicsMapResources = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuGraphicsMapResources_ptsz')
                except:
                    pass
                try:
                    __cuGraphicsUnmapResources = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuGraphicsUnmapResources_ptsz')
                except:
                    pass
            else:
                # Else get the regular version
                try:
                    __cuMemcpy = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuMemcpy')
                except:
                    pass
                try:
                    __cuMemcpyPeer = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuMemcpyPeer')
                except:
                    pass
                try:
                    __cuMemcpyHtoD_v2 = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuMemcpyHtoD_v2')
                except:
                    pass
                try:
                    __cuMemcpyDtoH_v2 = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuMemcpyDtoH_v2')
                except:
                    pass
                try:
                    __cuMemcpyDtoD_v2 = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuMemcpyDtoD_v2')
                except:
                    pass
                try:
                    __cuMemcpyDtoA_v2 = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuMemcpyDtoA_v2')
                except:
                    pass
                try:
                    __cuMemcpyAtoD_v2 = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuMemcpyAtoD_v2')
                except:
                    pass
                try:
                    __cuMemcpyHtoA_v2 = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuMemcpyHtoA_v2')
                except:
                    pass
                try:
                    __cuMemcpyAtoH_v2 = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuMemcpyAtoH_v2')
                except:
                    pass
                try:
                    __cuMemcpyAtoA_v2 = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuMemcpyAtoA_v2')
                except:
                    pass
                try:
                    __cuMemcpy2D_v2 = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuMemcpy2D_v2')
                except:
                    pass
                try:
                    __cuMemcpy2DUnaligned_v2 = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuMemcpy2DUnaligned_v2')
                except:
                    pass
                try:
                    __cuMemcpy3D_v2 = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuMemcpy3D_v2')
                except:
                    pass
                try:
                    __cuMemcpy3DPeer = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuMemcpy3DPeer')
                except:
                    pass
                try:
                    __cuMemcpyAsync = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuMemcpyAsync')
                except:
                    pass
                try:
                    __cuMemcpyPeerAsync = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuMemcpyPeerAsync')
                except:
                    pass
                try:
                    __cuMemcpyHtoDAsync_v2 = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuMemcpyHtoDAsync_v2')
                except:
                    pass
                try:
                    __cuMemcpyDtoHAsync_v2 = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuMemcpyDtoHAsync_v2')
                except:
                    pass
                try:
                    __cuMemcpyDtoDAsync_v2 = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuMemcpyDtoDAsync_v2')
                except:
                    pass
                try:
                    __cuMemcpyHtoAAsync_v2 = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuMemcpyHtoAAsync_v2')
                except:
                    pass
                try:
                    __cuMemcpyAtoHAsync_v2 = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuMemcpyAtoHAsync_v2')
                except:
                    pass
                try:
                    __cuMemcpy2DAsync_v2 = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuMemcpy2DAsync_v2')
                except:
                    pass
                try:
                    __cuMemcpy3DAsync_v2 = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuMemcpy3DAsync_v2')
                except:
                    pass
                try:
                    __cuMemcpy3DPeerAsync = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuMemcpy3DPeerAsync')
                except:
                    pass
                try:
                    __cuMemsetD8_v2 = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuMemsetD8_v2')
                except:
                    pass
                try:
                    __cuMemsetD16_v2 = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuMemsetD16_v2')
                except:
                    pass
                try:
                    __cuMemsetD32_v2 = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuMemsetD32_v2')
                except:
                    pass
                try:
                    __cuMemsetD2D8_v2 = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuMemsetD2D8_v2')
                except:
                    pass
                try:
                    __cuMemsetD2D16_v2 = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuMemsetD2D16_v2')
                except:
                    pass
                try:
                    __cuMemsetD2D32_v2 = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuMemsetD2D32_v2')
                except:
                    pass
                try:
                    __cuMemsetD8Async = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuMemsetD8Async')
                except:
                    pass
                try:
                    __cuMemsetD16Async = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuMemsetD16Async')
                except:
                    pass
                try:
                    __cuMemsetD32Async = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuMemsetD32Async')
                except:
                    pass
                try:
                    __cuMemsetD2D8Async = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuMemsetD2D8Async')
                except:
                    pass
                try:
                    __cuMemsetD2D16Async = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuMemsetD2D16Async')
                except:
                    pass
                try:
                    __cuMemsetD2D32Async = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuMemsetD2D32Async')
                except:
                    pass
                try:
                    __cuMemMapArrayAsync = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuMemMapArrayAsync')
                except:
                    pass
                try:
                    __cuMemFreeAsync = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuMemFreeAsync')
                except:
                    pass
                try:
                    __cuMemAllocAsync = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuMemAllocAsync')
                except:
                    pass
                try:
                    __cuMemAllocFromPoolAsync = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuMemAllocFromPoolAsync')
                except:
                    pass
                try:
                    __cuMemPrefetchAsync = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuMemPrefetchAsync')
                except:
                    pass
                try:
                    __cuStreamGetPriority = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuStreamGetPriority')
                except:
                    pass
                try:
                    __cuStreamGetFlags = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuStreamGetFlags')
                except:
                    pass
                try:
                    __cuStreamGetCtx = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuStreamGetCtx')
                except:
                    pass
                try:
                    __cuStreamWaitEvent = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuStreamWaitEvent')
                except:
                    pass
                try:
                    __cuStreamAddCallback = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuStreamAddCallback')
                except:
                    pass
                try:
                    __cuStreamBeginCapture_v2 = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuStreamBeginCapture_v2')
                except:
                    pass
                try:
                    __cuStreamEndCapture = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuStreamEndCapture')
                except:
                    pass
                try:
                    __cuStreamIsCapturing = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuStreamIsCapturing')
                except:
                    pass
                try:
                    __cuStreamGetCaptureInfo = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuStreamGetCaptureInfo')
                except:
                    pass
                try:
                    __cuStreamGetCaptureInfo_v2 = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuStreamGetCaptureInfo_v2')
                except:
                    pass
                try:
                    __cuStreamUpdateCaptureDependencies = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuStreamUpdateCaptureDependencies')
                except:
                    pass
                try:
                    __cuStreamAttachMemAsync = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuStreamAttachMemAsync')
                except:
                    pass
                try:
                    __cuStreamQuery = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuStreamQuery')
                except:
                    pass
                try:
                    __cuStreamSynchronize = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuStreamSynchronize')
                except:
                    pass
                try:
                    __cuStreamCopyAttributes = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuStreamCopyAttributes')
                except:
                    pass
                try:
                    __cuStreamGetAttribute = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuStreamGetAttribute')
                except:
                    pass
                try:
                    __cuStreamSetAttribute = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuStreamSetAttribute')
                except:
                    pass
                try:
                    __cuEventRecord = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuEventRecord')
                except:
                    pass
                try:
                    __cuEventRecordWithFlags = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuEventRecordWithFlags')
                except:
                    pass
                try:
                    __cuSignalExternalSemaphoresAsync = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuSignalExternalSemaphoresAsync')
                except:
                    pass
                try:
                    __cuWaitExternalSemaphoresAsync = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuWaitExternalSemaphoresAsync')
                except:
                    pass
                try:
                    __cuStreamWaitValue32 = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuStreamWaitValue32')
                except:
                    pass
                try:
                    __cuStreamWaitValue64 = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuStreamWaitValue64')
                except:
                    pass
                try:
                    __cuStreamWriteValue32 = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuStreamWriteValue32')
                except:
                    pass
                try:
                    __cuStreamWriteValue64 = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuStreamWriteValue64')
                except:
                    pass
                try:
                    __cuStreamBatchMemOp = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuStreamBatchMemOp')
                except:
                    pass
                try:
                    __cuStreamWaitValue32_v2 = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuStreamWaitValue32_v2')
                except:
                    pass
                try:
                    __cuStreamWaitValue64_v2 = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuStreamWaitValue64_v2')
                except:
                    pass
                try:
                    __cuStreamWriteValue32_v2 = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuStreamWriteValue32_v2')
                except:
                    pass
                try:
                    __cuStreamWriteValue64_v2 = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuStreamWriteValue64_v2')
                except:
                    pass
                try:
                    __cuStreamBatchMemOp_v2 = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuStreamBatchMemOp_v2')
                except:
                    pass
                try:
                    __cuLaunchKernel = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuLaunchKernel')
                except:
                    pass
                try:
                    __cuLaunchCooperativeKernel = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuLaunchCooperativeKernel')
                except:
                    pass
                try:
                    __cuLaunchHostFunc = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuLaunchHostFunc')
                except:
                    pass
                try:
                    __cuGraphUpload = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuGraphUpload')
                except:
                    pass
                try:
                    __cuGraphLaunch = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuGraphLaunch')
                except:
                    pass
                try:
                    __cuGraphicsMapResources = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuGraphicsMapResources')
                except:
                    pass
                try:
                    __cuGraphicsUnmapResources = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuGraphicsUnmapResources')
                except:
                    pass
            # Get remaining functions
            try:
                __cuGetErrorString = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuGetErrorString')
            except:
                pass
            try:
                __cuGetErrorName = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuGetErrorName')
            except:
                pass
            try:
                __cuInit = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuInit')
            except:
                pass
            try:
                __cuDriverGetVersion = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuDriverGetVersion')
            except:
                pass
            try:
                __cuDeviceGet = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuDeviceGet')
            except:
                pass
            try:
                __cuDeviceGetCount = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuDeviceGetCount')
            except:
                pass
            try:
                __cuDeviceGetName = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuDeviceGetName')
            except:
                pass
            try:
                __cuDeviceGetUuid = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuDeviceGetUuid')
            except:
                pass
            try:
                __cuDeviceGetUuid_v2 = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuDeviceGetUuid_v2')
            except:
                pass
            try:
                __cuDeviceGetLuid = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuDeviceGetLuid')
            except:
                pass
            try:
                __cuDeviceTotalMem_v2 = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuDeviceTotalMem_v2')
            except:
                pass
            try:
                __cuDeviceGetTexture1DLinearMaxWidth = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuDeviceGetTexture1DLinearMaxWidth')
            except:
                pass
            try:
                __cuDeviceGetAttribute = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuDeviceGetAttribute')
            except:
                pass
            try:
                __cuDeviceGetNvSciSyncAttributes = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuDeviceGetNvSciSyncAttributes')
            except:
                pass
            try:
                __cuDeviceSetMemPool = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuDeviceSetMemPool')
            except:
                pass
            try:
                __cuDeviceGetMemPool = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuDeviceGetMemPool')
            except:
                pass
            try:
                __cuDeviceGetDefaultMemPool = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuDeviceGetDefaultMemPool')
            except:
                pass
            try:
                __cuFlushGPUDirectRDMAWrites = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuFlushGPUDirectRDMAWrites')
            except:
                pass
            try:
                __cuDeviceGetProperties = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuDeviceGetProperties')
            except:
                pass
            try:
                __cuDeviceComputeCapability = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuDeviceComputeCapability')
            except:
                pass
            try:
                __cuDevicePrimaryCtxRetain = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuDevicePrimaryCtxRetain')
            except:
                pass
            try:
                __cuDevicePrimaryCtxRelease_v2 = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuDevicePrimaryCtxRelease_v2')
            except:
                pass
            try:
                __cuDevicePrimaryCtxSetFlags_v2 = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuDevicePrimaryCtxSetFlags_v2')
            except:
                pass
            try:
                __cuDevicePrimaryCtxGetState = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuDevicePrimaryCtxGetState')
            except:
                pass
            try:
                __cuDevicePrimaryCtxReset_v2 = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuDevicePrimaryCtxReset_v2')
            except:
                pass
            try:
                __cuDeviceGetExecAffinitySupport = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuDeviceGetExecAffinitySupport')
            except:
                pass
            try:
                __cuCtxCreate_v2 = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuCtxCreate_v2')
            except:
                pass
            try:
                __cuCtxCreate_v3 = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuCtxCreate_v3')
            except:
                pass
            try:
                __cuCtxDestroy_v2 = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuCtxDestroy_v2')
            except:
                pass
            try:
                __cuCtxPushCurrent_v2 = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuCtxPushCurrent_v2')
            except:
                pass
            try:
                __cuCtxPopCurrent_v2 = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuCtxPopCurrent_v2')
            except:
                pass
            try:
                __cuCtxSetCurrent = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuCtxSetCurrent')
            except:
                pass
            try:
                __cuCtxGetCurrent = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuCtxGetCurrent')
            except:
                pass
            try:
                __cuCtxGetDevice = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuCtxGetDevice')
            except:
                pass
            try:
                __cuCtxGetFlags = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuCtxGetFlags')
            except:
                pass
            try:
                __cuCtxSynchronize = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuCtxSynchronize')
            except:
                pass
            try:
                __cuCtxSetLimit = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuCtxSetLimit')
            except:
                pass
            try:
                __cuCtxGetLimit = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuCtxGetLimit')
            except:
                pass
            try:
                __cuCtxGetCacheConfig = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuCtxGetCacheConfig')
            except:
                pass
            try:
                __cuCtxSetCacheConfig = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuCtxSetCacheConfig')
            except:
                pass
            try:
                __cuCtxGetSharedMemConfig = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuCtxGetSharedMemConfig')
            except:
                pass
            try:
                __cuCtxSetSharedMemConfig = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuCtxSetSharedMemConfig')
            except:
                pass
            try:
                __cuCtxGetApiVersion = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuCtxGetApiVersion')
            except:
                pass
            try:
                __cuCtxGetStreamPriorityRange = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuCtxGetStreamPriorityRange')
            except:
                pass
            try:
                __cuCtxResetPersistingL2Cache = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuCtxResetPersistingL2Cache')
            except:
                pass
            try:
                __cuCtxGetExecAffinity = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuCtxGetExecAffinity')
            except:
                pass
            try:
                __cuCtxAttach = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuCtxAttach')
            except:
                pass
            try:
                __cuCtxDetach = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuCtxDetach')
            except:
                pass
            try:
                __cuModuleLoad = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuModuleLoad')
            except:
                pass
            try:
                __cuModuleLoadData = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuModuleLoadData')
            except:
                pass
            try:
                __cuModuleLoadDataEx = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuModuleLoadDataEx')
            except:
                pass
            try:
                __cuModuleLoadFatBinary = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuModuleLoadFatBinary')
            except:
                pass
            try:
                __cuModuleUnload = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuModuleUnload')
            except:
                pass
            try:
                __cuModuleGetFunction = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuModuleGetFunction')
            except:
                pass
            try:
                __cuModuleGetGlobal_v2 = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuModuleGetGlobal_v2')
            except:
                pass
            try:
                __cuModuleGetTexRef = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuModuleGetTexRef')
            except:
                pass
            try:
                __cuModuleGetSurfRef = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuModuleGetSurfRef')
            except:
                pass
            try:
                __cuLinkCreate_v2 = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuLinkCreate_v2')
            except:
                pass
            try:
                __cuLinkAddData_v2 = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuLinkAddData_v2')
            except:
                pass
            try:
                __cuLinkAddFile_v2 = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuLinkAddFile_v2')
            except:
                pass
            try:
                __cuLinkComplete = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuLinkComplete')
            except:
                pass
            try:
                __cuLinkDestroy = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuLinkDestroy')
            except:
                pass
            try:
                __cuMemGetInfo_v2 = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuMemGetInfo_v2')
            except:
                pass
            try:
                __cuMemAlloc_v2 = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuMemAlloc_v2')
            except:
                pass
            try:
                __cuMemAllocPitch_v2 = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuMemAllocPitch_v2')
            except:
                pass
            try:
                __cuMemFree_v2 = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuMemFree_v2')
            except:
                pass
            try:
                __cuMemGetAddressRange_v2 = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuMemGetAddressRange_v2')
            except:
                pass
            try:
                __cuMemAllocHost_v2 = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuMemAllocHost_v2')
            except:
                pass
            try:
                __cuMemFreeHost = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuMemFreeHost')
            except:
                pass
            try:
                __cuMemHostAlloc = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuMemHostAlloc')
            except:
                pass
            try:
                __cuMemHostGetDevicePointer_v2 = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuMemHostGetDevicePointer_v2')
            except:
                pass
            try:
                __cuMemHostGetFlags = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuMemHostGetFlags')
            except:
                pass
            try:
                __cuMemAllocManaged = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuMemAllocManaged')
            except:
                pass
            try:
                __cuDeviceGetByPCIBusId = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuDeviceGetByPCIBusId')
            except:
                pass
            try:
                __cuDeviceGetPCIBusId = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuDeviceGetPCIBusId')
            except:
                pass
            try:
                __cuIpcGetEventHandle = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuIpcGetEventHandle')
            except:
                pass
            try:
                __cuIpcOpenEventHandle = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuIpcOpenEventHandle')
            except:
                pass
            try:
                __cuIpcGetMemHandle = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuIpcGetMemHandle')
            except:
                pass
            try:
                __cuIpcOpenMemHandle_v2 = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuIpcOpenMemHandle_v2')
            except:
                pass
            try:
                __cuIpcCloseMemHandle = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuIpcCloseMemHandle')
            except:
                pass
            try:
                __cuMemHostRegister_v2 = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuMemHostRegister_v2')
            except:
                pass
            try:
                __cuMemHostUnregister = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuMemHostUnregister')
            except:
                pass
            try:
                __cuArrayCreate_v2 = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuArrayCreate_v2')
            except:
                pass
            try:
                __cuArrayGetDescriptor_v2 = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuArrayGetDescriptor_v2')
            except:
                pass
            try:
                __cuArrayGetSparseProperties = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuArrayGetSparseProperties')
            except:
                pass
            try:
                __cuMipmappedArrayGetSparseProperties = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuMipmappedArrayGetSparseProperties')
            except:
                pass
            try:
                __cuArrayGetMemoryRequirements = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuArrayGetMemoryRequirements')
            except:
                pass
            try:
                __cuMipmappedArrayGetMemoryRequirements = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuMipmappedArrayGetMemoryRequirements')
            except:
                pass
            try:
                __cuArrayGetPlane = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuArrayGetPlane')
            except:
                pass
            try:
                __cuArrayDestroy = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuArrayDestroy')
            except:
                pass
            try:
                __cuArray3DCreate_v2 = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuArray3DCreate_v2')
            except:
                pass
            try:
                __cuArray3DGetDescriptor_v2 = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuArray3DGetDescriptor_v2')
            except:
                pass
            try:
                __cuMipmappedArrayCreate = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuMipmappedArrayCreate')
            except:
                pass
            try:
                __cuMipmappedArrayGetLevel = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuMipmappedArrayGetLevel')
            except:
                pass
            try:
                __cuMipmappedArrayDestroy = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuMipmappedArrayDestroy')
            except:
                pass
            try:
                __cuMemAddressReserve = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuMemAddressReserve')
            except:
                pass
            try:
                __cuMemAddressFree = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuMemAddressFree')
            except:
                pass
            try:
                __cuMemCreate = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuMemCreate')
            except:
                pass
            try:
                __cuMemRelease = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuMemRelease')
            except:
                pass
            try:
                __cuMemMap = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuMemMap')
            except:
                pass
            try:
                __cuMemUnmap = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuMemUnmap')
            except:
                pass
            try:
                __cuMemSetAccess = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuMemSetAccess')
            except:
                pass
            try:
                __cuMemGetAccess = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuMemGetAccess')
            except:
                pass
            try:
                __cuMemExportToShareableHandle = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuMemExportToShareableHandle')
            except:
                pass
            try:
                __cuMemImportFromShareableHandle = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuMemImportFromShareableHandle')
            except:
                pass
            try:
                __cuMemGetAllocationGranularity = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuMemGetAllocationGranularity')
            except:
                pass
            try:
                __cuMemGetAllocationPropertiesFromHandle = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuMemGetAllocationPropertiesFromHandle')
            except:
                pass
            try:
                __cuMemRetainAllocationHandle = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuMemRetainAllocationHandle')
            except:
                pass
            try:
                __cuMemPoolTrimTo = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuMemPoolTrimTo')
            except:
                pass
            try:
                __cuMemPoolSetAttribute = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuMemPoolSetAttribute')
            except:
                pass
            try:
                __cuMemPoolGetAttribute = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuMemPoolGetAttribute')
            except:
                pass
            try:
                __cuMemPoolSetAccess = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuMemPoolSetAccess')
            except:
                pass
            try:
                __cuMemPoolGetAccess = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuMemPoolGetAccess')
            except:
                pass
            try:
                __cuMemPoolCreate = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuMemPoolCreate')
            except:
                pass
            try:
                __cuMemPoolDestroy = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuMemPoolDestroy')
            except:
                pass
            try:
                __cuMemPoolExportToShareableHandle = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuMemPoolExportToShareableHandle')
            except:
                pass
            try:
                __cuMemPoolImportFromShareableHandle = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuMemPoolImportFromShareableHandle')
            except:
                pass
            try:
                __cuMemPoolExportPointer = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuMemPoolExportPointer')
            except:
                pass
            try:
                __cuMemPoolImportPointer = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuMemPoolImportPointer')
            except:
                pass
            try:
                __cuPointerGetAttribute = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuPointerGetAttribute')
            except:
                pass
            try:
                __cuMemAdvise = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuMemAdvise')
            except:
                pass
            try:
                __cuMemRangeGetAttribute = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuMemRangeGetAttribute')
            except:
                pass
            try:
                __cuMemRangeGetAttributes = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuMemRangeGetAttributes')
            except:
                pass
            try:
                __cuPointerSetAttribute = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuPointerSetAttribute')
            except:
                pass
            try:
                __cuPointerGetAttributes = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuPointerGetAttributes')
            except:
                pass
            try:
                __cuStreamCreate = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuStreamCreate')
            except:
                pass
            try:
                __cuStreamCreateWithPriority = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuStreamCreateWithPriority')
            except:
                pass
            try:
                __cuThreadExchangeStreamCaptureMode = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuThreadExchangeStreamCaptureMode')
            except:
                pass
            try:
                __cuStreamDestroy_v2 = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuStreamDestroy_v2')
            except:
                pass
            try:
                __cuEventCreate = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuEventCreate')
            except:
                pass
            try:
                __cuEventQuery = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuEventQuery')
            except:
                pass
            try:
                __cuEventSynchronize = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuEventSynchronize')
            except:
                pass
            try:
                __cuEventDestroy_v2 = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuEventDestroy_v2')
            except:
                pass
            try:
                __cuEventElapsedTime = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuEventElapsedTime')
            except:
                pass
            try:
                __cuImportExternalMemory = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuImportExternalMemory')
            except:
                pass
            try:
                __cuExternalMemoryGetMappedBuffer = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuExternalMemoryGetMappedBuffer')
            except:
                pass
            try:
                __cuExternalMemoryGetMappedMipmappedArray = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuExternalMemoryGetMappedMipmappedArray')
            except:
                pass
            try:
                __cuDestroyExternalMemory = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuDestroyExternalMemory')
            except:
                pass
            try:
                __cuImportExternalSemaphore = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuImportExternalSemaphore')
            except:
                pass
            try:
                __cuDestroyExternalSemaphore = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuDestroyExternalSemaphore')
            except:
                pass
            try:
                __cuFuncGetAttribute = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuFuncGetAttribute')
            except:
                pass
            try:
                __cuFuncSetAttribute = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuFuncSetAttribute')
            except:
                pass
            try:
                __cuFuncSetCacheConfig = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuFuncSetCacheConfig')
            except:
                pass
            try:
                __cuFuncSetSharedMemConfig = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuFuncSetSharedMemConfig')
            except:
                pass
            try:
                __cuFuncGetModule = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuFuncGetModule')
            except:
                pass
            try:
                __cuLaunchCooperativeKernelMultiDevice = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuLaunchCooperativeKernelMultiDevice')
            except:
                pass
            try:
                __cuFuncSetBlockShape = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuFuncSetBlockShape')
            except:
                pass
            try:
                __cuFuncSetSharedSize = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuFuncSetSharedSize')
            except:
                pass
            try:
                __cuParamSetSize = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuParamSetSize')
            except:
                pass
            try:
                __cuParamSeti = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuParamSeti')
            except:
                pass
            try:
                __cuParamSetf = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuParamSetf')
            except:
                pass
            try:
                __cuParamSetv = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuParamSetv')
            except:
                pass
            try:
                __cuLaunch = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuLaunch')
            except:
                pass
            try:
                __cuLaunchGrid = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuLaunchGrid')
            except:
                pass
            try:
                __cuLaunchGridAsync = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuLaunchGridAsync')
            except:
                pass
            try:
                __cuParamSetTexRef = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuParamSetTexRef')
            except:
                pass
            try:
                __cuGraphCreate = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuGraphCreate')
            except:
                pass
            try:
                __cuGraphAddKernelNode = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuGraphAddKernelNode')
            except:
                pass
            try:
                __cuGraphKernelNodeGetParams = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuGraphKernelNodeGetParams')
            except:
                pass
            try:
                __cuGraphKernelNodeSetParams = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuGraphKernelNodeSetParams')
            except:
                pass
            try:
                __cuGraphAddMemcpyNode = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuGraphAddMemcpyNode')
            except:
                pass
            try:
                __cuGraphMemcpyNodeGetParams = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuGraphMemcpyNodeGetParams')
            except:
                pass
            try:
                __cuGraphMemcpyNodeSetParams = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuGraphMemcpyNodeSetParams')
            except:
                pass
            try:
                __cuGraphAddMemsetNode = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuGraphAddMemsetNode')
            except:
                pass
            try:
                __cuGraphMemsetNodeGetParams = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuGraphMemsetNodeGetParams')
            except:
                pass
            try:
                __cuGraphMemsetNodeSetParams = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuGraphMemsetNodeSetParams')
            except:
                pass
            try:
                __cuGraphAddHostNode = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuGraphAddHostNode')
            except:
                pass
            try:
                __cuGraphHostNodeGetParams = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuGraphHostNodeGetParams')
            except:
                pass
            try:
                __cuGraphHostNodeSetParams = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuGraphHostNodeSetParams')
            except:
                pass
            try:
                __cuGraphAddChildGraphNode = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuGraphAddChildGraphNode')
            except:
                pass
            try:
                __cuGraphChildGraphNodeGetGraph = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuGraphChildGraphNodeGetGraph')
            except:
                pass
            try:
                __cuGraphAddEmptyNode = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuGraphAddEmptyNode')
            except:
                pass
            try:
                __cuGraphAddEventRecordNode = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuGraphAddEventRecordNode')
            except:
                pass
            try:
                __cuGraphEventRecordNodeGetEvent = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuGraphEventRecordNodeGetEvent')
            except:
                pass
            try:
                __cuGraphEventRecordNodeSetEvent = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuGraphEventRecordNodeSetEvent')
            except:
                pass
            try:
                __cuGraphAddEventWaitNode = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuGraphAddEventWaitNode')
            except:
                pass
            try:
                __cuGraphEventWaitNodeGetEvent = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuGraphEventWaitNodeGetEvent')
            except:
                pass
            try:
                __cuGraphEventWaitNodeSetEvent = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuGraphEventWaitNodeSetEvent')
            except:
                pass
            try:
                __cuGraphAddExternalSemaphoresSignalNode = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuGraphAddExternalSemaphoresSignalNode')
            except:
                pass
            try:
                __cuGraphExternalSemaphoresSignalNodeGetParams = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuGraphExternalSemaphoresSignalNodeGetParams')
            except:
                pass
            try:
                __cuGraphExternalSemaphoresSignalNodeSetParams = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuGraphExternalSemaphoresSignalNodeSetParams')
            except:
                pass
            try:
                __cuGraphAddExternalSemaphoresWaitNode = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuGraphAddExternalSemaphoresWaitNode')
            except:
                pass
            try:
                __cuGraphExternalSemaphoresWaitNodeGetParams = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuGraphExternalSemaphoresWaitNodeGetParams')
            except:
                pass
            try:
                __cuGraphExternalSemaphoresWaitNodeSetParams = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuGraphExternalSemaphoresWaitNodeSetParams')
            except:
                pass
            try:
                __cuGraphAddBatchMemOpNode = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuGraphAddBatchMemOpNode')
            except:
                pass
            try:
                __cuGraphBatchMemOpNodeGetParams = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuGraphBatchMemOpNodeGetParams')
            except:
                pass
            try:
                __cuGraphBatchMemOpNodeSetParams = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuGraphBatchMemOpNodeSetParams')
            except:
                pass
            try:
                __cuGraphExecBatchMemOpNodeSetParams = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuGraphExecBatchMemOpNodeSetParams')
            except:
                pass
            try:
                __cuGraphAddMemAllocNode = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuGraphAddMemAllocNode')
            except:
                pass
            try:
                __cuGraphMemAllocNodeGetParams = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuGraphMemAllocNodeGetParams')
            except:
                pass
            try:
                __cuGraphAddMemFreeNode = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuGraphAddMemFreeNode')
            except:
                pass
            try:
                __cuGraphMemFreeNodeGetParams = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuGraphMemFreeNodeGetParams')
            except:
                pass
            try:
                __cuDeviceGraphMemTrim = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuDeviceGraphMemTrim')
            except:
                pass
            try:
                __cuDeviceGetGraphMemAttribute = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuDeviceGetGraphMemAttribute')
            except:
                pass
            try:
                __cuDeviceSetGraphMemAttribute = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuDeviceSetGraphMemAttribute')
            except:
                pass
            try:
                __cuGraphClone = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuGraphClone')
            except:
                pass
            try:
                __cuGraphNodeFindInClone = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuGraphNodeFindInClone')
            except:
                pass
            try:
                __cuGraphNodeGetType = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuGraphNodeGetType')
            except:
                pass
            try:
                __cuGraphGetNodes = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuGraphGetNodes')
            except:
                pass
            try:
                __cuGraphGetRootNodes = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuGraphGetRootNodes')
            except:
                pass
            try:
                __cuGraphGetEdges = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuGraphGetEdges')
            except:
                pass
            try:
                __cuGraphNodeGetDependencies = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuGraphNodeGetDependencies')
            except:
                pass
            try:
                __cuGraphNodeGetDependentNodes = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuGraphNodeGetDependentNodes')
            except:
                pass
            try:
                __cuGraphAddDependencies = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuGraphAddDependencies')
            except:
                pass
            try:
                __cuGraphRemoveDependencies = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuGraphRemoveDependencies')
            except:
                pass
            try:
                __cuGraphDestroyNode = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuGraphDestroyNode')
            except:
                pass
            try:
                __cuGraphInstantiate_v2 = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuGraphInstantiate_v2')
            except:
                pass
            try:
                __cuGraphInstantiateWithFlags = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuGraphInstantiateWithFlags')
            except:
                pass
            try:
                __cuGraphExecKernelNodeSetParams = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuGraphExecKernelNodeSetParams')
            except:
                pass
            try:
                __cuGraphExecMemcpyNodeSetParams = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuGraphExecMemcpyNodeSetParams')
            except:
                pass
            try:
                __cuGraphExecMemsetNodeSetParams = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuGraphExecMemsetNodeSetParams')
            except:
                pass
            try:
                __cuGraphExecHostNodeSetParams = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuGraphExecHostNodeSetParams')
            except:
                pass
            try:
                __cuGraphExecChildGraphNodeSetParams = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuGraphExecChildGraphNodeSetParams')
            except:
                pass
            try:
                __cuGraphExecEventRecordNodeSetEvent = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuGraphExecEventRecordNodeSetEvent')
            except:
                pass
            try:
                __cuGraphExecEventWaitNodeSetEvent = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuGraphExecEventWaitNodeSetEvent')
            except:
                pass
            try:
                __cuGraphExecExternalSemaphoresSignalNodeSetParams = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuGraphExecExternalSemaphoresSignalNodeSetParams')
            except:
                pass
            try:
                __cuGraphExecExternalSemaphoresWaitNodeSetParams = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuGraphExecExternalSemaphoresWaitNodeSetParams')
            except:
                pass
            try:
                __cuGraphNodeSetEnabled = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuGraphNodeSetEnabled')
            except:
                pass
            try:
                __cuGraphNodeGetEnabled = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuGraphNodeGetEnabled')
            except:
                pass
            try:
                __cuGraphExecDestroy = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuGraphExecDestroy')
            except:
                pass
            try:
                __cuGraphDestroy = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuGraphDestroy')
            except:
                pass
            try:
                __cuGraphExecUpdate = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuGraphExecUpdate')
            except:
                pass
            try:
                __cuGraphKernelNodeCopyAttributes = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuGraphKernelNodeCopyAttributes')
            except:
                pass
            try:
                __cuGraphKernelNodeGetAttribute = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuGraphKernelNodeGetAttribute')
            except:
                pass
            try:
                __cuGraphKernelNodeSetAttribute = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuGraphKernelNodeSetAttribute')
            except:
                pass
            try:
                __cuGraphDebugDotPrint = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuGraphDebugDotPrint')
            except:
                pass
            try:
                __cuUserObjectCreate = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuUserObjectCreate')
            except:
                pass
            try:
                __cuUserObjectRetain = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuUserObjectRetain')
            except:
                pass
            try:
                __cuUserObjectRelease = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuUserObjectRelease')
            except:
                pass
            try:
                __cuGraphRetainUserObject = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuGraphRetainUserObject')
            except:
                pass
            try:
                __cuGraphReleaseUserObject = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuGraphReleaseUserObject')
            except:
                pass
            try:
                __cuOccupancyMaxActiveBlocksPerMultiprocessor = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuOccupancyMaxActiveBlocksPerMultiprocessor')
            except:
                pass
            try:
                __cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags')
            except:
                pass
            try:
                __cuOccupancyMaxPotentialBlockSize = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuOccupancyMaxPotentialBlockSize')
            except:
                pass
            try:
                __cuOccupancyMaxPotentialBlockSizeWithFlags = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuOccupancyMaxPotentialBlockSizeWithFlags')
            except:
                pass
            try:
                __cuOccupancyAvailableDynamicSMemPerBlock = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuOccupancyAvailableDynamicSMemPerBlock')
            except:
                pass
            try:
                __cuTexRefSetArray = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuTexRefSetArray')
            except:
                pass
            try:
                __cuTexRefSetMipmappedArray = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuTexRefSetMipmappedArray')
            except:
                pass
            try:
                __cuTexRefSetAddress_v2 = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuTexRefSetAddress_v2')
            except:
                pass
            try:
                __cuTexRefSetAddress2D_v3 = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuTexRefSetAddress2D_v3')
            except:
                pass
            try:
                __cuTexRefSetFormat = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuTexRefSetFormat')
            except:
                pass
            try:
                __cuTexRefSetAddressMode = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuTexRefSetAddressMode')
            except:
                pass
            try:
                __cuTexRefSetFilterMode = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuTexRefSetFilterMode')
            except:
                pass
            try:
                __cuTexRefSetMipmapFilterMode = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuTexRefSetMipmapFilterMode')
            except:
                pass
            try:
                __cuTexRefSetMipmapLevelBias = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuTexRefSetMipmapLevelBias')
            except:
                pass
            try:
                __cuTexRefSetMipmapLevelClamp = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuTexRefSetMipmapLevelClamp')
            except:
                pass
            try:
                __cuTexRefSetMaxAnisotropy = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuTexRefSetMaxAnisotropy')
            except:
                pass
            try:
                __cuTexRefSetBorderColor = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuTexRefSetBorderColor')
            except:
                pass
            try:
                __cuTexRefSetFlags = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuTexRefSetFlags')
            except:
                pass
            try:
                __cuTexRefGetAddress_v2 = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuTexRefGetAddress_v2')
            except:
                pass
            try:
                __cuTexRefGetArray = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuTexRefGetArray')
            except:
                pass
            try:
                __cuTexRefGetMipmappedArray = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuTexRefGetMipmappedArray')
            except:
                pass
            try:
                __cuTexRefGetAddressMode = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuTexRefGetAddressMode')
            except:
                pass
            try:
                __cuTexRefGetFilterMode = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuTexRefGetFilterMode')
            except:
                pass
            try:
                __cuTexRefGetFormat = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuTexRefGetFormat')
            except:
                pass
            try:
                __cuTexRefGetMipmapFilterMode = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuTexRefGetMipmapFilterMode')
            except:
                pass
            try:
                __cuTexRefGetMipmapLevelBias = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuTexRefGetMipmapLevelBias')
            except:
                pass
            try:
                __cuTexRefGetMipmapLevelClamp = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuTexRefGetMipmapLevelClamp')
            except:
                pass
            try:
                __cuTexRefGetMaxAnisotropy = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuTexRefGetMaxAnisotropy')
            except:
                pass
            try:
                __cuTexRefGetBorderColor = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuTexRefGetBorderColor')
            except:
                pass
            try:
                __cuTexRefGetFlags = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuTexRefGetFlags')
            except:
                pass
            try:
                __cuTexRefCreate = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuTexRefCreate')
            except:
                pass
            try:
                __cuTexRefDestroy = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuTexRefDestroy')
            except:
                pass
            try:
                __cuSurfRefSetArray = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuSurfRefSetArray')
            except:
                pass
            try:
                __cuSurfRefGetArray = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuSurfRefGetArray')
            except:
                pass
            try:
                __cuTexObjectCreate = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuTexObjectCreate')
            except:
                pass
            try:
                __cuTexObjectDestroy = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuTexObjectDestroy')
            except:
                pass
            try:
                __cuTexObjectGetResourceDesc = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuTexObjectGetResourceDesc')
            except:
                pass
            try:
                __cuTexObjectGetTextureDesc = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuTexObjectGetTextureDesc')
            except:
                pass
            try:
                __cuTexObjectGetResourceViewDesc = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuTexObjectGetResourceViewDesc')
            except:
                pass
            try:
                __cuSurfObjectCreate = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuSurfObjectCreate')
            except:
                pass
            try:
                __cuSurfObjectDestroy = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuSurfObjectDestroy')
            except:
                pass
            try:
                __cuSurfObjectGetResourceDesc = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuSurfObjectGetResourceDesc')
            except:
                pass
            try:
                __cuDeviceCanAccessPeer = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuDeviceCanAccessPeer')
            except:
                pass
            try:
                __cuCtxEnablePeerAccess = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuCtxEnablePeerAccess')
            except:
                pass
            try:
                __cuCtxDisablePeerAccess = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuCtxDisablePeerAccess')
            except:
                pass
            try:
                __cuDeviceGetP2PAttribute = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuDeviceGetP2PAttribute')
            except:
                pass
            try:
                __cuGraphicsUnregisterResource = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuGraphicsUnregisterResource')
            except:
                pass
            try:
                __cuGraphicsSubResourceGetMappedArray = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuGraphicsSubResourceGetMappedArray')
            except:
                pass
            try:
                __cuGraphicsResourceGetMappedMipmappedArray = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuGraphicsResourceGetMappedMipmappedArray')
            except:
                pass
            try:
                __cuGraphicsResourceGetMappedPointer_v2 = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuGraphicsResourceGetMappedPointer_v2')
            except:
                pass
            try:
                __cuGraphicsResourceSetMapFlags_v2 = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuGraphicsResourceSetMapFlags_v2')
            except:
                pass
            try:
                __cuGetProcAddress = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuGetProcAddress')
            except:
                pass
            try:
                __cuModuleGetLoadingMode = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuModuleGetLoadingMode')
            except:
                pass
            try:
                __cuMemGetHandleForAddressRange = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuMemGetHandleForAddressRange')
            except:
                pass
            try:
                __cuGetExportTable = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuGetExportTable')
            except:
                pass
            try:
                __cuProfilerInitialize = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuProfilerInitialize')
            except:
                pass
            try:
                __cuProfilerStart = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuProfilerStart')
            except:
                pass
            try:
                __cuProfilerStop = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuProfilerStop')
            except:
                pass
            try:
                __cuVDPAUGetDevice = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuVDPAUGetDevice')
            except:
                pass
            try:
                __cuVDPAUCtxCreate_v2 = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuVDPAUCtxCreate_v2')
            except:
                pass
            try:
                __cuGraphicsVDPAURegisterVideoSurface = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuGraphicsVDPAURegisterVideoSurface')
            except:
                pass
            try:
                __cuGraphicsVDPAURegisterOutputSurface = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuGraphicsVDPAURegisterOutputSurface')
            except:
                pass
            try:
                __cuGraphicsEGLRegisterImage = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuGraphicsEGLRegisterImage')
            except:
                pass
            try:
                __cuEGLStreamConsumerConnect = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuEGLStreamConsumerConnect')
            except:
                pass
            try:
                __cuEGLStreamConsumerConnectWithFlags = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuEGLStreamConsumerConnectWithFlags')
            except:
                pass
            try:
                __cuEGLStreamConsumerDisconnect = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuEGLStreamConsumerDisconnect')
            except:
                pass
            try:
                __cuEGLStreamConsumerAcquireFrame = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuEGLStreamConsumerAcquireFrame')
            except:
                pass
            try:
                __cuEGLStreamConsumerReleaseFrame = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuEGLStreamConsumerReleaseFrame')
            except:
                pass
            try:
                __cuEGLStreamProducerConnect = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuEGLStreamProducerConnect')
            except:
                pass
            try:
                __cuEGLStreamProducerDisconnect = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuEGLStreamProducerDisconnect')
            except:
                pass
            try:
                __cuEGLStreamProducerPresentFrame = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuEGLStreamProducerPresentFrame')
            except:
                pass
            try:
                __cuEGLStreamProducerReturnFrame = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuEGLStreamProducerReturnFrame')
            except:
                pass
            try:
                __cuGraphicsResourceGetMappedEglFrame = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuGraphicsResourceGetMappedEglFrame')
            except:
                pass
            try:
                __cuEventCreateFromEGLSync = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuEventCreateFromEGLSync')
            except:
                pass
            try:
                __cuGraphicsGLRegisterBuffer = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuGraphicsGLRegisterBuffer')
            except:
                pass
            try:
                __cuGraphicsGLRegisterImage = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuGraphicsGLRegisterImage')
            except:
                pass
            try:
                __cuGLGetDevices_v2 = <void*><unsigned long long>win32api.GetProcAddress(handle, 'cuGLGetDevices_v2')
            except:
                pass
    ELSE:
        if usePTDS:
            # Get all PTDS version of functions
            __cuMemcpy = dlfcn.dlsym(handle, 'cuMemcpy_ptds')
            __cuMemcpyPeer = dlfcn.dlsym(handle, 'cuMemcpyPeer_ptds')
            __cuMemcpyHtoD_v2 = dlfcn.dlsym(handle, 'cuMemcpyHtoD_v2_ptds')
            __cuMemcpyDtoH_v2 = dlfcn.dlsym(handle, 'cuMemcpyDtoH_v2_ptds')
            __cuMemcpyDtoD_v2 = dlfcn.dlsym(handle, 'cuMemcpyDtoD_v2_ptds')
            __cuMemcpyDtoA_v2 = dlfcn.dlsym(handle, 'cuMemcpyDtoA_v2_ptds')
            __cuMemcpyAtoD_v2 = dlfcn.dlsym(handle, 'cuMemcpyAtoD_v2_ptds')
            __cuMemcpyHtoA_v2 = dlfcn.dlsym(handle, 'cuMemcpyHtoA_v2_ptds')
            __cuMemcpyAtoH_v2 = dlfcn.dlsym(handle, 'cuMemcpyAtoH_v2_ptds')
            __cuMemcpyAtoA_v2 = dlfcn.dlsym(handle, 'cuMemcpyAtoA_v2_ptds')
            __cuMemcpy2D_v2 = dlfcn.dlsym(handle, 'cuMemcpy2D_v2_ptds')
            __cuMemcpy2DUnaligned_v2 = dlfcn.dlsym(handle, 'cuMemcpy2DUnaligned_v2_ptds')
            __cuMemcpy3D_v2 = dlfcn.dlsym(handle, 'cuMemcpy3D_v2_ptds')
            __cuMemcpy3DPeer = dlfcn.dlsym(handle, 'cuMemcpy3DPeer_ptds')
            __cuMemcpyAsync = dlfcn.dlsym(handle, 'cuMemcpyAsync_ptsz')
            __cuMemcpyPeerAsync = dlfcn.dlsym(handle, 'cuMemcpyPeerAsync_ptsz')
            __cuMemcpyHtoDAsync_v2 = dlfcn.dlsym(handle, 'cuMemcpyHtoDAsync_v2_ptsz')
            __cuMemcpyDtoHAsync_v2 = dlfcn.dlsym(handle, 'cuMemcpyDtoHAsync_v2_ptsz')
            __cuMemcpyDtoDAsync_v2 = dlfcn.dlsym(handle, 'cuMemcpyDtoDAsync_v2_ptsz')
            __cuMemcpyHtoAAsync_v2 = dlfcn.dlsym(handle, 'cuMemcpyHtoAAsync_v2_ptsz')
            __cuMemcpyAtoHAsync_v2 = dlfcn.dlsym(handle, 'cuMemcpyAtoHAsync_v2_ptsz')
            __cuMemcpy2DAsync_v2 = dlfcn.dlsym(handle, 'cuMemcpy2DAsync_v2_ptsz')
            __cuMemcpy3DAsync_v2 = dlfcn.dlsym(handle, 'cuMemcpy3DAsync_v2_ptsz')
            __cuMemcpy3DPeerAsync = dlfcn.dlsym(handle, 'cuMemcpy3DPeerAsync_ptsz')
            __cuMemsetD8_v2 = dlfcn.dlsym(handle, 'cuMemsetD8_v2_ptds')
            __cuMemsetD16_v2 = dlfcn.dlsym(handle, 'cuMemsetD16_v2_ptds')
            __cuMemsetD32_v2 = dlfcn.dlsym(handle, 'cuMemsetD32_v2_ptds')
            __cuMemsetD2D8_v2 = dlfcn.dlsym(handle, 'cuMemsetD2D8_v2_ptds')
            __cuMemsetD2D16_v2 = dlfcn.dlsym(handle, 'cuMemsetD2D16_v2_ptds')
            __cuMemsetD2D32_v2 = dlfcn.dlsym(handle, 'cuMemsetD2D32_v2_ptds')
            __cuMemsetD8Async = dlfcn.dlsym(handle, 'cuMemsetD8Async_ptsz')
            __cuMemsetD16Async = dlfcn.dlsym(handle, 'cuMemsetD16Async_ptsz')
            __cuMemsetD32Async = dlfcn.dlsym(handle, 'cuMemsetD32Async_ptsz')
            __cuMemsetD2D8Async = dlfcn.dlsym(handle, 'cuMemsetD2D8Async_ptsz')
            __cuMemsetD2D16Async = dlfcn.dlsym(handle, 'cuMemsetD2D16Async_ptsz')
            __cuMemsetD2D32Async = dlfcn.dlsym(handle, 'cuMemsetD2D32Async_ptsz')
            __cuMemMapArrayAsync = dlfcn.dlsym(handle, 'cuMemMapArrayAsync_ptsz')
            __cuMemFreeAsync = dlfcn.dlsym(handle, 'cuMemFreeAsync_ptsz')
            __cuMemAllocAsync = dlfcn.dlsym(handle, 'cuMemAllocAsync_ptsz')
            __cuMemAllocFromPoolAsync = dlfcn.dlsym(handle, 'cuMemAllocFromPoolAsync_ptsz')
            __cuMemPrefetchAsync = dlfcn.dlsym(handle, 'cuMemPrefetchAsync_ptsz')
            __cuStreamGetPriority = dlfcn.dlsym(handle, 'cuStreamGetPriority_ptsz')
            __cuStreamGetFlags = dlfcn.dlsym(handle, 'cuStreamGetFlags_ptsz')
            __cuStreamGetCtx = dlfcn.dlsym(handle, 'cuStreamGetCtx_ptsz')
            __cuStreamWaitEvent = dlfcn.dlsym(handle, 'cuStreamWaitEvent_ptsz')
            __cuStreamAddCallback = dlfcn.dlsym(handle, 'cuStreamAddCallback_ptsz')
            __cuStreamBeginCapture_v2 = dlfcn.dlsym(handle, 'cuStreamBeginCapture_v2_ptsz')
            __cuStreamEndCapture = dlfcn.dlsym(handle, 'cuStreamEndCapture_ptsz')
            __cuStreamIsCapturing = dlfcn.dlsym(handle, 'cuStreamIsCapturing_ptsz')
            __cuStreamGetCaptureInfo = dlfcn.dlsym(handle, 'cuStreamGetCaptureInfo_ptsz')
            __cuStreamGetCaptureInfo_v2 = dlfcn.dlsym(handle, 'cuStreamGetCaptureInfo_v2_ptsz')
            __cuStreamUpdateCaptureDependencies = dlfcn.dlsym(handle, 'cuStreamUpdateCaptureDependencies_ptsz')
            __cuStreamAttachMemAsync = dlfcn.dlsym(handle, 'cuStreamAttachMemAsync_ptsz')
            __cuStreamQuery = dlfcn.dlsym(handle, 'cuStreamQuery_ptsz')
            __cuStreamSynchronize = dlfcn.dlsym(handle, 'cuStreamSynchronize_ptsz')
            __cuStreamCopyAttributes = dlfcn.dlsym(handle, 'cuStreamCopyAttributes_ptsz')
            __cuStreamGetAttribute = dlfcn.dlsym(handle, 'cuStreamGetAttribute_ptsz')
            __cuStreamSetAttribute = dlfcn.dlsym(handle, 'cuStreamSetAttribute_ptsz')
            __cuEventRecord = dlfcn.dlsym(handle, 'cuEventRecord_ptsz')
            __cuEventRecordWithFlags = dlfcn.dlsym(handle, 'cuEventRecordWithFlags_ptsz')
            __cuSignalExternalSemaphoresAsync = dlfcn.dlsym(handle, 'cuSignalExternalSemaphoresAsync_ptsz')
            __cuWaitExternalSemaphoresAsync = dlfcn.dlsym(handle, 'cuWaitExternalSemaphoresAsync_ptsz')
            __cuStreamWaitValue32 = dlfcn.dlsym(handle, 'cuStreamWaitValue32_ptsz')
            __cuStreamWaitValue64 = dlfcn.dlsym(handle, 'cuStreamWaitValue64_ptsz')
            __cuStreamWriteValue32 = dlfcn.dlsym(handle, 'cuStreamWriteValue32_ptsz')
            __cuStreamWriteValue64 = dlfcn.dlsym(handle, 'cuStreamWriteValue64_ptsz')
            __cuStreamBatchMemOp = dlfcn.dlsym(handle, 'cuStreamBatchMemOp_ptsz')
            __cuStreamWaitValue32_v2 = dlfcn.dlsym(handle, 'cuStreamWaitValue32_v2_ptsz')
            __cuStreamWaitValue64_v2 = dlfcn.dlsym(handle, 'cuStreamWaitValue64_v2_ptsz')
            __cuStreamWriteValue32_v2 = dlfcn.dlsym(handle, 'cuStreamWriteValue32_v2_ptsz')
            __cuStreamWriteValue64_v2 = dlfcn.dlsym(handle, 'cuStreamWriteValue64_v2_ptsz')
            __cuStreamBatchMemOp_v2 = dlfcn.dlsym(handle, 'cuStreamBatchMemOp_v2_ptsz')
            __cuLaunchKernel = dlfcn.dlsym(handle, 'cuLaunchKernel_ptsz')
            __cuLaunchCooperativeKernel = dlfcn.dlsym(handle, 'cuLaunchCooperativeKernel_ptsz')
            __cuLaunchHostFunc = dlfcn.dlsym(handle, 'cuLaunchHostFunc_ptsz')
            __cuGraphUpload = dlfcn.dlsym(handle, 'cuGraphUpload_ptsz')
            __cuGraphLaunch = dlfcn.dlsym(handle, 'cuGraphLaunch_ptsz')
            __cuGraphicsMapResources = dlfcn.dlsym(handle, 'cuGraphicsMapResources_ptsz')
            __cuGraphicsUnmapResources = dlfcn.dlsym(handle, 'cuGraphicsUnmapResources_ptsz')
        else:
            # Else get the regular version
            __cuMemcpy = dlfcn.dlsym(handle, 'cuMemcpy')
            __cuMemcpyPeer = dlfcn.dlsym(handle, 'cuMemcpyPeer')
            __cuMemcpyHtoD_v2 = dlfcn.dlsym(handle, 'cuMemcpyHtoD_v2')
            __cuMemcpyDtoH_v2 = dlfcn.dlsym(handle, 'cuMemcpyDtoH_v2')
            __cuMemcpyDtoD_v2 = dlfcn.dlsym(handle, 'cuMemcpyDtoD_v2')
            __cuMemcpyDtoA_v2 = dlfcn.dlsym(handle, 'cuMemcpyDtoA_v2')
            __cuMemcpyAtoD_v2 = dlfcn.dlsym(handle, 'cuMemcpyAtoD_v2')
            __cuMemcpyHtoA_v2 = dlfcn.dlsym(handle, 'cuMemcpyHtoA_v2')
            __cuMemcpyAtoH_v2 = dlfcn.dlsym(handle, 'cuMemcpyAtoH_v2')
            __cuMemcpyAtoA_v2 = dlfcn.dlsym(handle, 'cuMemcpyAtoA_v2')
            __cuMemcpy2D_v2 = dlfcn.dlsym(handle, 'cuMemcpy2D_v2')
            __cuMemcpy2DUnaligned_v2 = dlfcn.dlsym(handle, 'cuMemcpy2DUnaligned_v2')
            __cuMemcpy3D_v2 = dlfcn.dlsym(handle, 'cuMemcpy3D_v2')
            __cuMemcpy3DPeer = dlfcn.dlsym(handle, 'cuMemcpy3DPeer')
            __cuMemcpyAsync = dlfcn.dlsym(handle, 'cuMemcpyAsync')
            __cuMemcpyPeerAsync = dlfcn.dlsym(handle, 'cuMemcpyPeerAsync')
            __cuMemcpyHtoDAsync_v2 = dlfcn.dlsym(handle, 'cuMemcpyHtoDAsync_v2')
            __cuMemcpyDtoHAsync_v2 = dlfcn.dlsym(handle, 'cuMemcpyDtoHAsync_v2')
            __cuMemcpyDtoDAsync_v2 = dlfcn.dlsym(handle, 'cuMemcpyDtoDAsync_v2')
            __cuMemcpyHtoAAsync_v2 = dlfcn.dlsym(handle, 'cuMemcpyHtoAAsync_v2')
            __cuMemcpyAtoHAsync_v2 = dlfcn.dlsym(handle, 'cuMemcpyAtoHAsync_v2')
            __cuMemcpy2DAsync_v2 = dlfcn.dlsym(handle, 'cuMemcpy2DAsync_v2')
            __cuMemcpy3DAsync_v2 = dlfcn.dlsym(handle, 'cuMemcpy3DAsync_v2')
            __cuMemcpy3DPeerAsync = dlfcn.dlsym(handle, 'cuMemcpy3DPeerAsync')
            __cuMemsetD8_v2 = dlfcn.dlsym(handle, 'cuMemsetD8_v2')
            __cuMemsetD16_v2 = dlfcn.dlsym(handle, 'cuMemsetD16_v2')
            __cuMemsetD32_v2 = dlfcn.dlsym(handle, 'cuMemsetD32_v2')
            __cuMemsetD2D8_v2 = dlfcn.dlsym(handle, 'cuMemsetD2D8_v2')
            __cuMemsetD2D16_v2 = dlfcn.dlsym(handle, 'cuMemsetD2D16_v2')
            __cuMemsetD2D32_v2 = dlfcn.dlsym(handle, 'cuMemsetD2D32_v2')
            __cuMemsetD8Async = dlfcn.dlsym(handle, 'cuMemsetD8Async')
            __cuMemsetD16Async = dlfcn.dlsym(handle, 'cuMemsetD16Async')
            __cuMemsetD32Async = dlfcn.dlsym(handle, 'cuMemsetD32Async')
            __cuMemsetD2D8Async = dlfcn.dlsym(handle, 'cuMemsetD2D8Async')
            __cuMemsetD2D16Async = dlfcn.dlsym(handle, 'cuMemsetD2D16Async')
            __cuMemsetD2D32Async = dlfcn.dlsym(handle, 'cuMemsetD2D32Async')
            __cuMemMapArrayAsync = dlfcn.dlsym(handle, 'cuMemMapArrayAsync')
            __cuMemFreeAsync = dlfcn.dlsym(handle, 'cuMemFreeAsync')
            __cuMemAllocAsync = dlfcn.dlsym(handle, 'cuMemAllocAsync')
            __cuMemAllocFromPoolAsync = dlfcn.dlsym(handle, 'cuMemAllocFromPoolAsync')
            __cuMemPrefetchAsync = dlfcn.dlsym(handle, 'cuMemPrefetchAsync')
            __cuStreamGetPriority = dlfcn.dlsym(handle, 'cuStreamGetPriority')
            __cuStreamGetFlags = dlfcn.dlsym(handle, 'cuStreamGetFlags')
            __cuStreamGetCtx = dlfcn.dlsym(handle, 'cuStreamGetCtx')
            __cuStreamWaitEvent = dlfcn.dlsym(handle, 'cuStreamWaitEvent')
            __cuStreamAddCallback = dlfcn.dlsym(handle, 'cuStreamAddCallback')
            __cuStreamBeginCapture_v2 = dlfcn.dlsym(handle, 'cuStreamBeginCapture_v2')
            __cuStreamEndCapture = dlfcn.dlsym(handle, 'cuStreamEndCapture')
            __cuStreamIsCapturing = dlfcn.dlsym(handle, 'cuStreamIsCapturing')
            __cuStreamGetCaptureInfo = dlfcn.dlsym(handle, 'cuStreamGetCaptureInfo')
            __cuStreamGetCaptureInfo_v2 = dlfcn.dlsym(handle, 'cuStreamGetCaptureInfo_v2')
            __cuStreamUpdateCaptureDependencies = dlfcn.dlsym(handle, 'cuStreamUpdateCaptureDependencies')
            __cuStreamAttachMemAsync = dlfcn.dlsym(handle, 'cuStreamAttachMemAsync')
            __cuStreamQuery = dlfcn.dlsym(handle, 'cuStreamQuery')
            __cuStreamSynchronize = dlfcn.dlsym(handle, 'cuStreamSynchronize')
            __cuStreamCopyAttributes = dlfcn.dlsym(handle, 'cuStreamCopyAttributes')
            __cuStreamGetAttribute = dlfcn.dlsym(handle, 'cuStreamGetAttribute')
            __cuStreamSetAttribute = dlfcn.dlsym(handle, 'cuStreamSetAttribute')
            __cuEventRecord = dlfcn.dlsym(handle, 'cuEventRecord')
            __cuEventRecordWithFlags = dlfcn.dlsym(handle, 'cuEventRecordWithFlags')
            __cuSignalExternalSemaphoresAsync = dlfcn.dlsym(handle, 'cuSignalExternalSemaphoresAsync')
            __cuWaitExternalSemaphoresAsync = dlfcn.dlsym(handle, 'cuWaitExternalSemaphoresAsync')
            __cuStreamWaitValue32 = dlfcn.dlsym(handle, 'cuStreamWaitValue32')
            __cuStreamWaitValue64 = dlfcn.dlsym(handle, 'cuStreamWaitValue64')
            __cuStreamWriteValue32 = dlfcn.dlsym(handle, 'cuStreamWriteValue32')
            __cuStreamWriteValue64 = dlfcn.dlsym(handle, 'cuStreamWriteValue64')
            __cuStreamBatchMemOp = dlfcn.dlsym(handle, 'cuStreamBatchMemOp')
            __cuStreamWaitValue32_v2 = dlfcn.dlsym(handle, 'cuStreamWaitValue32_v2')
            __cuStreamWaitValue64_v2 = dlfcn.dlsym(handle, 'cuStreamWaitValue64_v2')
            __cuStreamWriteValue32_v2 = dlfcn.dlsym(handle, 'cuStreamWriteValue32_v2')
            __cuStreamWriteValue64_v2 = dlfcn.dlsym(handle, 'cuStreamWriteValue64_v2')
            __cuStreamBatchMemOp_v2 = dlfcn.dlsym(handle, 'cuStreamBatchMemOp_v2')
            __cuLaunchKernel = dlfcn.dlsym(handle, 'cuLaunchKernel')
            __cuLaunchCooperativeKernel = dlfcn.dlsym(handle, 'cuLaunchCooperativeKernel')
            __cuLaunchHostFunc = dlfcn.dlsym(handle, 'cuLaunchHostFunc')
            __cuGraphUpload = dlfcn.dlsym(handle, 'cuGraphUpload')
            __cuGraphLaunch = dlfcn.dlsym(handle, 'cuGraphLaunch')
            __cuGraphicsMapResources = dlfcn.dlsym(handle, 'cuGraphicsMapResources')
            __cuGraphicsUnmapResources = dlfcn.dlsym(handle, 'cuGraphicsUnmapResources')
        # Get remaining functions
        __cuGetErrorString = dlfcn.dlsym(handle, 'cuGetErrorString')
        __cuGetErrorName = dlfcn.dlsym(handle, 'cuGetErrorName')
        __cuInit = dlfcn.dlsym(handle, 'cuInit')
        __cuDriverGetVersion = dlfcn.dlsym(handle, 'cuDriverGetVersion')
        __cuDeviceGet = dlfcn.dlsym(handle, 'cuDeviceGet')
        __cuDeviceGetCount = dlfcn.dlsym(handle, 'cuDeviceGetCount')
        __cuDeviceGetName = dlfcn.dlsym(handle, 'cuDeviceGetName')
        __cuDeviceGetUuid = dlfcn.dlsym(handle, 'cuDeviceGetUuid')
        __cuDeviceGetUuid_v2 = dlfcn.dlsym(handle, 'cuDeviceGetUuid_v2')
        __cuDeviceGetLuid = dlfcn.dlsym(handle, 'cuDeviceGetLuid')
        __cuDeviceTotalMem_v2 = dlfcn.dlsym(handle, 'cuDeviceTotalMem_v2')
        __cuDeviceGetTexture1DLinearMaxWidth = dlfcn.dlsym(handle, 'cuDeviceGetTexture1DLinearMaxWidth')
        __cuDeviceGetAttribute = dlfcn.dlsym(handle, 'cuDeviceGetAttribute')
        __cuDeviceGetNvSciSyncAttributes = dlfcn.dlsym(handle, 'cuDeviceGetNvSciSyncAttributes')
        __cuDeviceSetMemPool = dlfcn.dlsym(handle, 'cuDeviceSetMemPool')
        __cuDeviceGetMemPool = dlfcn.dlsym(handle, 'cuDeviceGetMemPool')
        __cuDeviceGetDefaultMemPool = dlfcn.dlsym(handle, 'cuDeviceGetDefaultMemPool')
        __cuFlushGPUDirectRDMAWrites = dlfcn.dlsym(handle, 'cuFlushGPUDirectRDMAWrites')
        __cuDeviceGetProperties = dlfcn.dlsym(handle, 'cuDeviceGetProperties')
        __cuDeviceComputeCapability = dlfcn.dlsym(handle, 'cuDeviceComputeCapability')
        __cuDevicePrimaryCtxRetain = dlfcn.dlsym(handle, 'cuDevicePrimaryCtxRetain')
        __cuDevicePrimaryCtxRelease_v2 = dlfcn.dlsym(handle, 'cuDevicePrimaryCtxRelease_v2')
        __cuDevicePrimaryCtxSetFlags_v2 = dlfcn.dlsym(handle, 'cuDevicePrimaryCtxSetFlags_v2')
        __cuDevicePrimaryCtxGetState = dlfcn.dlsym(handle, 'cuDevicePrimaryCtxGetState')
        __cuDevicePrimaryCtxReset_v2 = dlfcn.dlsym(handle, 'cuDevicePrimaryCtxReset_v2')
        __cuDeviceGetExecAffinitySupport = dlfcn.dlsym(handle, 'cuDeviceGetExecAffinitySupport')
        __cuCtxCreate_v2 = dlfcn.dlsym(handle, 'cuCtxCreate_v2')
        __cuCtxCreate_v3 = dlfcn.dlsym(handle, 'cuCtxCreate_v3')
        __cuCtxDestroy_v2 = dlfcn.dlsym(handle, 'cuCtxDestroy_v2')
        __cuCtxPushCurrent_v2 = dlfcn.dlsym(handle, 'cuCtxPushCurrent_v2')
        __cuCtxPopCurrent_v2 = dlfcn.dlsym(handle, 'cuCtxPopCurrent_v2')
        __cuCtxSetCurrent = dlfcn.dlsym(handle, 'cuCtxSetCurrent')
        __cuCtxGetCurrent = dlfcn.dlsym(handle, 'cuCtxGetCurrent')
        __cuCtxGetDevice = dlfcn.dlsym(handle, 'cuCtxGetDevice')
        __cuCtxGetFlags = dlfcn.dlsym(handle, 'cuCtxGetFlags')
        __cuCtxSynchronize = dlfcn.dlsym(handle, 'cuCtxSynchronize')
        __cuCtxSetLimit = dlfcn.dlsym(handle, 'cuCtxSetLimit')
        __cuCtxGetLimit = dlfcn.dlsym(handle, 'cuCtxGetLimit')
        __cuCtxGetCacheConfig = dlfcn.dlsym(handle, 'cuCtxGetCacheConfig')
        __cuCtxSetCacheConfig = dlfcn.dlsym(handle, 'cuCtxSetCacheConfig')
        __cuCtxGetSharedMemConfig = dlfcn.dlsym(handle, 'cuCtxGetSharedMemConfig')
        __cuCtxSetSharedMemConfig = dlfcn.dlsym(handle, 'cuCtxSetSharedMemConfig')
        __cuCtxGetApiVersion = dlfcn.dlsym(handle, 'cuCtxGetApiVersion')
        __cuCtxGetStreamPriorityRange = dlfcn.dlsym(handle, 'cuCtxGetStreamPriorityRange')
        __cuCtxResetPersistingL2Cache = dlfcn.dlsym(handle, 'cuCtxResetPersistingL2Cache')
        __cuCtxGetExecAffinity = dlfcn.dlsym(handle, 'cuCtxGetExecAffinity')
        __cuCtxAttach = dlfcn.dlsym(handle, 'cuCtxAttach')
        __cuCtxDetach = dlfcn.dlsym(handle, 'cuCtxDetach')
        __cuModuleLoad = dlfcn.dlsym(handle, 'cuModuleLoad')
        __cuModuleLoadData = dlfcn.dlsym(handle, 'cuModuleLoadData')
        __cuModuleLoadDataEx = dlfcn.dlsym(handle, 'cuModuleLoadDataEx')
        __cuModuleLoadFatBinary = dlfcn.dlsym(handle, 'cuModuleLoadFatBinary')
        __cuModuleUnload = dlfcn.dlsym(handle, 'cuModuleUnload')
        __cuModuleGetFunction = dlfcn.dlsym(handle, 'cuModuleGetFunction')
        __cuModuleGetGlobal_v2 = dlfcn.dlsym(handle, 'cuModuleGetGlobal_v2')
        __cuModuleGetTexRef = dlfcn.dlsym(handle, 'cuModuleGetTexRef')
        __cuModuleGetSurfRef = dlfcn.dlsym(handle, 'cuModuleGetSurfRef')
        __cuLinkCreate_v2 = dlfcn.dlsym(handle, 'cuLinkCreate_v2')
        __cuLinkAddData_v2 = dlfcn.dlsym(handle, 'cuLinkAddData_v2')
        __cuLinkAddFile_v2 = dlfcn.dlsym(handle, 'cuLinkAddFile_v2')
        __cuLinkComplete = dlfcn.dlsym(handle, 'cuLinkComplete')
        __cuLinkDestroy = dlfcn.dlsym(handle, 'cuLinkDestroy')
        __cuMemGetInfo_v2 = dlfcn.dlsym(handle, 'cuMemGetInfo_v2')
        __cuMemAlloc_v2 = dlfcn.dlsym(handle, 'cuMemAlloc_v2')
        __cuMemAllocPitch_v2 = dlfcn.dlsym(handle, 'cuMemAllocPitch_v2')
        __cuMemFree_v2 = dlfcn.dlsym(handle, 'cuMemFree_v2')
        __cuMemGetAddressRange_v2 = dlfcn.dlsym(handle, 'cuMemGetAddressRange_v2')
        __cuMemAllocHost_v2 = dlfcn.dlsym(handle, 'cuMemAllocHost_v2')
        __cuMemFreeHost = dlfcn.dlsym(handle, 'cuMemFreeHost')
        __cuMemHostAlloc = dlfcn.dlsym(handle, 'cuMemHostAlloc')
        __cuMemHostGetDevicePointer_v2 = dlfcn.dlsym(handle, 'cuMemHostGetDevicePointer_v2')
        __cuMemHostGetFlags = dlfcn.dlsym(handle, 'cuMemHostGetFlags')
        __cuMemAllocManaged = dlfcn.dlsym(handle, 'cuMemAllocManaged')
        __cuDeviceGetByPCIBusId = dlfcn.dlsym(handle, 'cuDeviceGetByPCIBusId')
        __cuDeviceGetPCIBusId = dlfcn.dlsym(handle, 'cuDeviceGetPCIBusId')
        __cuIpcGetEventHandle = dlfcn.dlsym(handle, 'cuIpcGetEventHandle')
        __cuIpcOpenEventHandle = dlfcn.dlsym(handle, 'cuIpcOpenEventHandle')
        __cuIpcGetMemHandle = dlfcn.dlsym(handle, 'cuIpcGetMemHandle')
        __cuIpcOpenMemHandle_v2 = dlfcn.dlsym(handle, 'cuIpcOpenMemHandle_v2')
        __cuIpcCloseMemHandle = dlfcn.dlsym(handle, 'cuIpcCloseMemHandle')
        __cuMemHostRegister_v2 = dlfcn.dlsym(handle, 'cuMemHostRegister_v2')
        __cuMemHostUnregister = dlfcn.dlsym(handle, 'cuMemHostUnregister')
        __cuArrayCreate_v2 = dlfcn.dlsym(handle, 'cuArrayCreate_v2')
        __cuArrayGetDescriptor_v2 = dlfcn.dlsym(handle, 'cuArrayGetDescriptor_v2')
        __cuArrayGetSparseProperties = dlfcn.dlsym(handle, 'cuArrayGetSparseProperties')
        __cuMipmappedArrayGetSparseProperties = dlfcn.dlsym(handle, 'cuMipmappedArrayGetSparseProperties')
        __cuArrayGetMemoryRequirements = dlfcn.dlsym(handle, 'cuArrayGetMemoryRequirements')
        __cuMipmappedArrayGetMemoryRequirements = dlfcn.dlsym(handle, 'cuMipmappedArrayGetMemoryRequirements')
        __cuArrayGetPlane = dlfcn.dlsym(handle, 'cuArrayGetPlane')
        __cuArrayDestroy = dlfcn.dlsym(handle, 'cuArrayDestroy')
        __cuArray3DCreate_v2 = dlfcn.dlsym(handle, 'cuArray3DCreate_v2')
        __cuArray3DGetDescriptor_v2 = dlfcn.dlsym(handle, 'cuArray3DGetDescriptor_v2')
        __cuMipmappedArrayCreate = dlfcn.dlsym(handle, 'cuMipmappedArrayCreate')
        __cuMipmappedArrayGetLevel = dlfcn.dlsym(handle, 'cuMipmappedArrayGetLevel')
        __cuMipmappedArrayDestroy = dlfcn.dlsym(handle, 'cuMipmappedArrayDestroy')
        __cuMemAddressReserve = dlfcn.dlsym(handle, 'cuMemAddressReserve')
        __cuMemAddressFree = dlfcn.dlsym(handle, 'cuMemAddressFree')
        __cuMemCreate = dlfcn.dlsym(handle, 'cuMemCreate')
        __cuMemRelease = dlfcn.dlsym(handle, 'cuMemRelease')
        __cuMemMap = dlfcn.dlsym(handle, 'cuMemMap')
        __cuMemUnmap = dlfcn.dlsym(handle, 'cuMemUnmap')
        __cuMemSetAccess = dlfcn.dlsym(handle, 'cuMemSetAccess')
        __cuMemGetAccess = dlfcn.dlsym(handle, 'cuMemGetAccess')
        __cuMemExportToShareableHandle = dlfcn.dlsym(handle, 'cuMemExportToShareableHandle')
        __cuMemImportFromShareableHandle = dlfcn.dlsym(handle, 'cuMemImportFromShareableHandle')
        __cuMemGetAllocationGranularity = dlfcn.dlsym(handle, 'cuMemGetAllocationGranularity')
        __cuMemGetAllocationPropertiesFromHandle = dlfcn.dlsym(handle, 'cuMemGetAllocationPropertiesFromHandle')
        __cuMemRetainAllocationHandle = dlfcn.dlsym(handle, 'cuMemRetainAllocationHandle')
        __cuMemPoolTrimTo = dlfcn.dlsym(handle, 'cuMemPoolTrimTo')
        __cuMemPoolSetAttribute = dlfcn.dlsym(handle, 'cuMemPoolSetAttribute')
        __cuMemPoolGetAttribute = dlfcn.dlsym(handle, 'cuMemPoolGetAttribute')
        __cuMemPoolSetAccess = dlfcn.dlsym(handle, 'cuMemPoolSetAccess')
        __cuMemPoolGetAccess = dlfcn.dlsym(handle, 'cuMemPoolGetAccess')
        __cuMemPoolCreate = dlfcn.dlsym(handle, 'cuMemPoolCreate')
        __cuMemPoolDestroy = dlfcn.dlsym(handle, 'cuMemPoolDestroy')
        __cuMemPoolExportToShareableHandle = dlfcn.dlsym(handle, 'cuMemPoolExportToShareableHandle')
        __cuMemPoolImportFromShareableHandle = dlfcn.dlsym(handle, 'cuMemPoolImportFromShareableHandle')
        __cuMemPoolExportPointer = dlfcn.dlsym(handle, 'cuMemPoolExportPointer')
        __cuMemPoolImportPointer = dlfcn.dlsym(handle, 'cuMemPoolImportPointer')
        __cuPointerGetAttribute = dlfcn.dlsym(handle, 'cuPointerGetAttribute')
        __cuMemAdvise = dlfcn.dlsym(handle, 'cuMemAdvise')
        __cuMemRangeGetAttribute = dlfcn.dlsym(handle, 'cuMemRangeGetAttribute')
        __cuMemRangeGetAttributes = dlfcn.dlsym(handle, 'cuMemRangeGetAttributes')
        __cuPointerSetAttribute = dlfcn.dlsym(handle, 'cuPointerSetAttribute')
        __cuPointerGetAttributes = dlfcn.dlsym(handle, 'cuPointerGetAttributes')
        __cuStreamCreate = dlfcn.dlsym(handle, 'cuStreamCreate')
        __cuStreamCreateWithPriority = dlfcn.dlsym(handle, 'cuStreamCreateWithPriority')
        __cuThreadExchangeStreamCaptureMode = dlfcn.dlsym(handle, 'cuThreadExchangeStreamCaptureMode')
        __cuStreamDestroy_v2 = dlfcn.dlsym(handle, 'cuStreamDestroy_v2')
        __cuEventCreate = dlfcn.dlsym(handle, 'cuEventCreate')
        __cuEventQuery = dlfcn.dlsym(handle, 'cuEventQuery')
        __cuEventSynchronize = dlfcn.dlsym(handle, 'cuEventSynchronize')
        __cuEventDestroy_v2 = dlfcn.dlsym(handle, 'cuEventDestroy_v2')
        __cuEventElapsedTime = dlfcn.dlsym(handle, 'cuEventElapsedTime')
        __cuImportExternalMemory = dlfcn.dlsym(handle, 'cuImportExternalMemory')
        __cuExternalMemoryGetMappedBuffer = dlfcn.dlsym(handle, 'cuExternalMemoryGetMappedBuffer')
        __cuExternalMemoryGetMappedMipmappedArray = dlfcn.dlsym(handle, 'cuExternalMemoryGetMappedMipmappedArray')
        __cuDestroyExternalMemory = dlfcn.dlsym(handle, 'cuDestroyExternalMemory')
        __cuImportExternalSemaphore = dlfcn.dlsym(handle, 'cuImportExternalSemaphore')
        __cuDestroyExternalSemaphore = dlfcn.dlsym(handle, 'cuDestroyExternalSemaphore')
        __cuFuncGetAttribute = dlfcn.dlsym(handle, 'cuFuncGetAttribute')
        __cuFuncSetAttribute = dlfcn.dlsym(handle, 'cuFuncSetAttribute')
        __cuFuncSetCacheConfig = dlfcn.dlsym(handle, 'cuFuncSetCacheConfig')
        __cuFuncSetSharedMemConfig = dlfcn.dlsym(handle, 'cuFuncSetSharedMemConfig')
        __cuFuncGetModule = dlfcn.dlsym(handle, 'cuFuncGetModule')
        __cuLaunchCooperativeKernelMultiDevice = dlfcn.dlsym(handle, 'cuLaunchCooperativeKernelMultiDevice')
        __cuFuncSetBlockShape = dlfcn.dlsym(handle, 'cuFuncSetBlockShape')
        __cuFuncSetSharedSize = dlfcn.dlsym(handle, 'cuFuncSetSharedSize')
        __cuParamSetSize = dlfcn.dlsym(handle, 'cuParamSetSize')
        __cuParamSeti = dlfcn.dlsym(handle, 'cuParamSeti')
        __cuParamSetf = dlfcn.dlsym(handle, 'cuParamSetf')
        __cuParamSetv = dlfcn.dlsym(handle, 'cuParamSetv')
        __cuLaunch = dlfcn.dlsym(handle, 'cuLaunch')
        __cuLaunchGrid = dlfcn.dlsym(handle, 'cuLaunchGrid')
        __cuLaunchGridAsync = dlfcn.dlsym(handle, 'cuLaunchGridAsync')
        __cuParamSetTexRef = dlfcn.dlsym(handle, 'cuParamSetTexRef')
        __cuGraphCreate = dlfcn.dlsym(handle, 'cuGraphCreate')
        __cuGraphAddKernelNode = dlfcn.dlsym(handle, 'cuGraphAddKernelNode')
        __cuGraphKernelNodeGetParams = dlfcn.dlsym(handle, 'cuGraphKernelNodeGetParams')
        __cuGraphKernelNodeSetParams = dlfcn.dlsym(handle, 'cuGraphKernelNodeSetParams')
        __cuGraphAddMemcpyNode = dlfcn.dlsym(handle, 'cuGraphAddMemcpyNode')
        __cuGraphMemcpyNodeGetParams = dlfcn.dlsym(handle, 'cuGraphMemcpyNodeGetParams')
        __cuGraphMemcpyNodeSetParams = dlfcn.dlsym(handle, 'cuGraphMemcpyNodeSetParams')
        __cuGraphAddMemsetNode = dlfcn.dlsym(handle, 'cuGraphAddMemsetNode')
        __cuGraphMemsetNodeGetParams = dlfcn.dlsym(handle, 'cuGraphMemsetNodeGetParams')
        __cuGraphMemsetNodeSetParams = dlfcn.dlsym(handle, 'cuGraphMemsetNodeSetParams')
        __cuGraphAddHostNode = dlfcn.dlsym(handle, 'cuGraphAddHostNode')
        __cuGraphHostNodeGetParams = dlfcn.dlsym(handle, 'cuGraphHostNodeGetParams')
        __cuGraphHostNodeSetParams = dlfcn.dlsym(handle, 'cuGraphHostNodeSetParams')
        __cuGraphAddChildGraphNode = dlfcn.dlsym(handle, 'cuGraphAddChildGraphNode')
        __cuGraphChildGraphNodeGetGraph = dlfcn.dlsym(handle, 'cuGraphChildGraphNodeGetGraph')
        __cuGraphAddEmptyNode = dlfcn.dlsym(handle, 'cuGraphAddEmptyNode')
        __cuGraphAddEventRecordNode = dlfcn.dlsym(handle, 'cuGraphAddEventRecordNode')
        __cuGraphEventRecordNodeGetEvent = dlfcn.dlsym(handle, 'cuGraphEventRecordNodeGetEvent')
        __cuGraphEventRecordNodeSetEvent = dlfcn.dlsym(handle, 'cuGraphEventRecordNodeSetEvent')
        __cuGraphAddEventWaitNode = dlfcn.dlsym(handle, 'cuGraphAddEventWaitNode')
        __cuGraphEventWaitNodeGetEvent = dlfcn.dlsym(handle, 'cuGraphEventWaitNodeGetEvent')
        __cuGraphEventWaitNodeSetEvent = dlfcn.dlsym(handle, 'cuGraphEventWaitNodeSetEvent')
        __cuGraphAddExternalSemaphoresSignalNode = dlfcn.dlsym(handle, 'cuGraphAddExternalSemaphoresSignalNode')
        __cuGraphExternalSemaphoresSignalNodeGetParams = dlfcn.dlsym(handle, 'cuGraphExternalSemaphoresSignalNodeGetParams')
        __cuGraphExternalSemaphoresSignalNodeSetParams = dlfcn.dlsym(handle, 'cuGraphExternalSemaphoresSignalNodeSetParams')
        __cuGraphAddExternalSemaphoresWaitNode = dlfcn.dlsym(handle, 'cuGraphAddExternalSemaphoresWaitNode')
        __cuGraphExternalSemaphoresWaitNodeGetParams = dlfcn.dlsym(handle, 'cuGraphExternalSemaphoresWaitNodeGetParams')
        __cuGraphExternalSemaphoresWaitNodeSetParams = dlfcn.dlsym(handle, 'cuGraphExternalSemaphoresWaitNodeSetParams')
        __cuGraphAddBatchMemOpNode = dlfcn.dlsym(handle, 'cuGraphAddBatchMemOpNode')
        __cuGraphBatchMemOpNodeGetParams = dlfcn.dlsym(handle, 'cuGraphBatchMemOpNodeGetParams')
        __cuGraphBatchMemOpNodeSetParams = dlfcn.dlsym(handle, 'cuGraphBatchMemOpNodeSetParams')
        __cuGraphExecBatchMemOpNodeSetParams = dlfcn.dlsym(handle, 'cuGraphExecBatchMemOpNodeSetParams')
        __cuGraphAddMemAllocNode = dlfcn.dlsym(handle, 'cuGraphAddMemAllocNode')
        __cuGraphMemAllocNodeGetParams = dlfcn.dlsym(handle, 'cuGraphMemAllocNodeGetParams')
        __cuGraphAddMemFreeNode = dlfcn.dlsym(handle, 'cuGraphAddMemFreeNode')
        __cuGraphMemFreeNodeGetParams = dlfcn.dlsym(handle, 'cuGraphMemFreeNodeGetParams')
        __cuDeviceGraphMemTrim = dlfcn.dlsym(handle, 'cuDeviceGraphMemTrim')
        __cuDeviceGetGraphMemAttribute = dlfcn.dlsym(handle, 'cuDeviceGetGraphMemAttribute')
        __cuDeviceSetGraphMemAttribute = dlfcn.dlsym(handle, 'cuDeviceSetGraphMemAttribute')
        __cuGraphClone = dlfcn.dlsym(handle, 'cuGraphClone')
        __cuGraphNodeFindInClone = dlfcn.dlsym(handle, 'cuGraphNodeFindInClone')
        __cuGraphNodeGetType = dlfcn.dlsym(handle, 'cuGraphNodeGetType')
        __cuGraphGetNodes = dlfcn.dlsym(handle, 'cuGraphGetNodes')
        __cuGraphGetRootNodes = dlfcn.dlsym(handle, 'cuGraphGetRootNodes')
        __cuGraphGetEdges = dlfcn.dlsym(handle, 'cuGraphGetEdges')
        __cuGraphNodeGetDependencies = dlfcn.dlsym(handle, 'cuGraphNodeGetDependencies')
        __cuGraphNodeGetDependentNodes = dlfcn.dlsym(handle, 'cuGraphNodeGetDependentNodes')
        __cuGraphAddDependencies = dlfcn.dlsym(handle, 'cuGraphAddDependencies')
        __cuGraphRemoveDependencies = dlfcn.dlsym(handle, 'cuGraphRemoveDependencies')
        __cuGraphDestroyNode = dlfcn.dlsym(handle, 'cuGraphDestroyNode')
        __cuGraphInstantiate_v2 = dlfcn.dlsym(handle, 'cuGraphInstantiate_v2')
        __cuGraphInstantiateWithFlags = dlfcn.dlsym(handle, 'cuGraphInstantiateWithFlags')
        __cuGraphExecKernelNodeSetParams = dlfcn.dlsym(handle, 'cuGraphExecKernelNodeSetParams')
        __cuGraphExecMemcpyNodeSetParams = dlfcn.dlsym(handle, 'cuGraphExecMemcpyNodeSetParams')
        __cuGraphExecMemsetNodeSetParams = dlfcn.dlsym(handle, 'cuGraphExecMemsetNodeSetParams')
        __cuGraphExecHostNodeSetParams = dlfcn.dlsym(handle, 'cuGraphExecHostNodeSetParams')
        __cuGraphExecChildGraphNodeSetParams = dlfcn.dlsym(handle, 'cuGraphExecChildGraphNodeSetParams')
        __cuGraphExecEventRecordNodeSetEvent = dlfcn.dlsym(handle, 'cuGraphExecEventRecordNodeSetEvent')
        __cuGraphExecEventWaitNodeSetEvent = dlfcn.dlsym(handle, 'cuGraphExecEventWaitNodeSetEvent')
        __cuGraphExecExternalSemaphoresSignalNodeSetParams = dlfcn.dlsym(handle, 'cuGraphExecExternalSemaphoresSignalNodeSetParams')
        __cuGraphExecExternalSemaphoresWaitNodeSetParams = dlfcn.dlsym(handle, 'cuGraphExecExternalSemaphoresWaitNodeSetParams')
        __cuGraphNodeSetEnabled = dlfcn.dlsym(handle, 'cuGraphNodeSetEnabled')
        __cuGraphNodeGetEnabled = dlfcn.dlsym(handle, 'cuGraphNodeGetEnabled')
        __cuGraphExecDestroy = dlfcn.dlsym(handle, 'cuGraphExecDestroy')
        __cuGraphDestroy = dlfcn.dlsym(handle, 'cuGraphDestroy')
        __cuGraphExecUpdate = dlfcn.dlsym(handle, 'cuGraphExecUpdate')
        __cuGraphKernelNodeCopyAttributes = dlfcn.dlsym(handle, 'cuGraphKernelNodeCopyAttributes')
        __cuGraphKernelNodeGetAttribute = dlfcn.dlsym(handle, 'cuGraphKernelNodeGetAttribute')
        __cuGraphKernelNodeSetAttribute = dlfcn.dlsym(handle, 'cuGraphKernelNodeSetAttribute')
        __cuGraphDebugDotPrint = dlfcn.dlsym(handle, 'cuGraphDebugDotPrint')
        __cuUserObjectCreate = dlfcn.dlsym(handle, 'cuUserObjectCreate')
        __cuUserObjectRetain = dlfcn.dlsym(handle, 'cuUserObjectRetain')
        __cuUserObjectRelease = dlfcn.dlsym(handle, 'cuUserObjectRelease')
        __cuGraphRetainUserObject = dlfcn.dlsym(handle, 'cuGraphRetainUserObject')
        __cuGraphReleaseUserObject = dlfcn.dlsym(handle, 'cuGraphReleaseUserObject')
        __cuOccupancyMaxActiveBlocksPerMultiprocessor = dlfcn.dlsym(handle, 'cuOccupancyMaxActiveBlocksPerMultiprocessor')
        __cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags = dlfcn.dlsym(handle, 'cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags')
        __cuOccupancyMaxPotentialBlockSize = dlfcn.dlsym(handle, 'cuOccupancyMaxPotentialBlockSize')
        __cuOccupancyMaxPotentialBlockSizeWithFlags = dlfcn.dlsym(handle, 'cuOccupancyMaxPotentialBlockSizeWithFlags')
        __cuOccupancyAvailableDynamicSMemPerBlock = dlfcn.dlsym(handle, 'cuOccupancyAvailableDynamicSMemPerBlock')
        __cuTexRefSetArray = dlfcn.dlsym(handle, 'cuTexRefSetArray')
        __cuTexRefSetMipmappedArray = dlfcn.dlsym(handle, 'cuTexRefSetMipmappedArray')
        __cuTexRefSetAddress_v2 = dlfcn.dlsym(handle, 'cuTexRefSetAddress_v2')
        __cuTexRefSetAddress2D_v3 = dlfcn.dlsym(handle, 'cuTexRefSetAddress2D_v3')
        __cuTexRefSetFormat = dlfcn.dlsym(handle, 'cuTexRefSetFormat')
        __cuTexRefSetAddressMode = dlfcn.dlsym(handle, 'cuTexRefSetAddressMode')
        __cuTexRefSetFilterMode = dlfcn.dlsym(handle, 'cuTexRefSetFilterMode')
        __cuTexRefSetMipmapFilterMode = dlfcn.dlsym(handle, 'cuTexRefSetMipmapFilterMode')
        __cuTexRefSetMipmapLevelBias = dlfcn.dlsym(handle, 'cuTexRefSetMipmapLevelBias')
        __cuTexRefSetMipmapLevelClamp = dlfcn.dlsym(handle, 'cuTexRefSetMipmapLevelClamp')
        __cuTexRefSetMaxAnisotropy = dlfcn.dlsym(handle, 'cuTexRefSetMaxAnisotropy')
        __cuTexRefSetBorderColor = dlfcn.dlsym(handle, 'cuTexRefSetBorderColor')
        __cuTexRefSetFlags = dlfcn.dlsym(handle, 'cuTexRefSetFlags')
        __cuTexRefGetAddress_v2 = dlfcn.dlsym(handle, 'cuTexRefGetAddress_v2')
        __cuTexRefGetArray = dlfcn.dlsym(handle, 'cuTexRefGetArray')
        __cuTexRefGetMipmappedArray = dlfcn.dlsym(handle, 'cuTexRefGetMipmappedArray')
        __cuTexRefGetAddressMode = dlfcn.dlsym(handle, 'cuTexRefGetAddressMode')
        __cuTexRefGetFilterMode = dlfcn.dlsym(handle, 'cuTexRefGetFilterMode')
        __cuTexRefGetFormat = dlfcn.dlsym(handle, 'cuTexRefGetFormat')
        __cuTexRefGetMipmapFilterMode = dlfcn.dlsym(handle, 'cuTexRefGetMipmapFilterMode')
        __cuTexRefGetMipmapLevelBias = dlfcn.dlsym(handle, 'cuTexRefGetMipmapLevelBias')
        __cuTexRefGetMipmapLevelClamp = dlfcn.dlsym(handle, 'cuTexRefGetMipmapLevelClamp')
        __cuTexRefGetMaxAnisotropy = dlfcn.dlsym(handle, 'cuTexRefGetMaxAnisotropy')
        __cuTexRefGetBorderColor = dlfcn.dlsym(handle, 'cuTexRefGetBorderColor')
        __cuTexRefGetFlags = dlfcn.dlsym(handle, 'cuTexRefGetFlags')
        __cuTexRefCreate = dlfcn.dlsym(handle, 'cuTexRefCreate')
        __cuTexRefDestroy = dlfcn.dlsym(handle, 'cuTexRefDestroy')
        __cuSurfRefSetArray = dlfcn.dlsym(handle, 'cuSurfRefSetArray')
        __cuSurfRefGetArray = dlfcn.dlsym(handle, 'cuSurfRefGetArray')
        __cuTexObjectCreate = dlfcn.dlsym(handle, 'cuTexObjectCreate')
        __cuTexObjectDestroy = dlfcn.dlsym(handle, 'cuTexObjectDestroy')
        __cuTexObjectGetResourceDesc = dlfcn.dlsym(handle, 'cuTexObjectGetResourceDesc')
        __cuTexObjectGetTextureDesc = dlfcn.dlsym(handle, 'cuTexObjectGetTextureDesc')
        __cuTexObjectGetResourceViewDesc = dlfcn.dlsym(handle, 'cuTexObjectGetResourceViewDesc')
        __cuSurfObjectCreate = dlfcn.dlsym(handle, 'cuSurfObjectCreate')
        __cuSurfObjectDestroy = dlfcn.dlsym(handle, 'cuSurfObjectDestroy')
        __cuSurfObjectGetResourceDesc = dlfcn.dlsym(handle, 'cuSurfObjectGetResourceDesc')
        __cuDeviceCanAccessPeer = dlfcn.dlsym(handle, 'cuDeviceCanAccessPeer')
        __cuCtxEnablePeerAccess = dlfcn.dlsym(handle, 'cuCtxEnablePeerAccess')
        __cuCtxDisablePeerAccess = dlfcn.dlsym(handle, 'cuCtxDisablePeerAccess')
        __cuDeviceGetP2PAttribute = dlfcn.dlsym(handle, 'cuDeviceGetP2PAttribute')
        __cuGraphicsUnregisterResource = dlfcn.dlsym(handle, 'cuGraphicsUnregisterResource')
        __cuGraphicsSubResourceGetMappedArray = dlfcn.dlsym(handle, 'cuGraphicsSubResourceGetMappedArray')
        __cuGraphicsResourceGetMappedMipmappedArray = dlfcn.dlsym(handle, 'cuGraphicsResourceGetMappedMipmappedArray')
        __cuGraphicsResourceGetMappedPointer_v2 = dlfcn.dlsym(handle, 'cuGraphicsResourceGetMappedPointer_v2')
        __cuGraphicsResourceSetMapFlags_v2 = dlfcn.dlsym(handle, 'cuGraphicsResourceSetMapFlags_v2')
        __cuGetProcAddress = dlfcn.dlsym(handle, 'cuGetProcAddress')
        __cuModuleGetLoadingMode = dlfcn.dlsym(handle, 'cuModuleGetLoadingMode')
        __cuMemGetHandleForAddressRange = dlfcn.dlsym(handle, 'cuMemGetHandleForAddressRange')
        __cuGetExportTable = dlfcn.dlsym(handle, 'cuGetExportTable')
        __cuProfilerInitialize = dlfcn.dlsym(handle, 'cuProfilerInitialize')
        __cuProfilerStart = dlfcn.dlsym(handle, 'cuProfilerStart')
        __cuProfilerStop = dlfcn.dlsym(handle, 'cuProfilerStop')
        __cuVDPAUGetDevice = dlfcn.dlsym(handle, 'cuVDPAUGetDevice')
        __cuVDPAUCtxCreate_v2 = dlfcn.dlsym(handle, 'cuVDPAUCtxCreate_v2')
        __cuGraphicsVDPAURegisterVideoSurface = dlfcn.dlsym(handle, 'cuGraphicsVDPAURegisterVideoSurface')
        __cuGraphicsVDPAURegisterOutputSurface = dlfcn.dlsym(handle, 'cuGraphicsVDPAURegisterOutputSurface')
        __cuGraphicsEGLRegisterImage = dlfcn.dlsym(handle, 'cuGraphicsEGLRegisterImage')
        __cuEGLStreamConsumerConnect = dlfcn.dlsym(handle, 'cuEGLStreamConsumerConnect')
        __cuEGLStreamConsumerConnectWithFlags = dlfcn.dlsym(handle, 'cuEGLStreamConsumerConnectWithFlags')
        __cuEGLStreamConsumerDisconnect = dlfcn.dlsym(handle, 'cuEGLStreamConsumerDisconnect')
        __cuEGLStreamConsumerAcquireFrame = dlfcn.dlsym(handle, 'cuEGLStreamConsumerAcquireFrame')
        __cuEGLStreamConsumerReleaseFrame = dlfcn.dlsym(handle, 'cuEGLStreamConsumerReleaseFrame')
        __cuEGLStreamProducerConnect = dlfcn.dlsym(handle, 'cuEGLStreamProducerConnect')
        __cuEGLStreamProducerDisconnect = dlfcn.dlsym(handle, 'cuEGLStreamProducerDisconnect')
        __cuEGLStreamProducerPresentFrame = dlfcn.dlsym(handle, 'cuEGLStreamProducerPresentFrame')
        __cuEGLStreamProducerReturnFrame = dlfcn.dlsym(handle, 'cuEGLStreamProducerReturnFrame')
        __cuGraphicsResourceGetMappedEglFrame = dlfcn.dlsym(handle, 'cuGraphicsResourceGetMappedEglFrame')
        __cuEventCreateFromEGLSync = dlfcn.dlsym(handle, 'cuEventCreateFromEGLSync')
        __cuGraphicsGLRegisterBuffer = dlfcn.dlsym(handle, 'cuGraphicsGLRegisterBuffer')
        __cuGraphicsGLRegisterImage = dlfcn.dlsym(handle, 'cuGraphicsGLRegisterImage')
        __cuGLGetDevices_v2 = dlfcn.dlsym(handle, 'cuGLGetDevices_v2')

cdef CUresult _cuGetErrorString(CUresult error, const char** pStr) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuGetErrorString
    cuPythonInit()
    if __cuGetErrorString == NULL:
        with gil:
            raise RuntimeError('Function "cuGetErrorString" not found')
    err = (<CUresult (*)(CUresult, const char**) nogil> __cuGetErrorString)(error, pStr)
    return err

cdef CUresult _cuGetErrorName(CUresult error, const char** pStr) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuGetErrorName
    cuPythonInit()
    if __cuGetErrorName == NULL:
        with gil:
            raise RuntimeError('Function "cuGetErrorName" not found')
    err = (<CUresult (*)(CUresult, const char**) nogil> __cuGetErrorName)(error, pStr)
    return err

cdef CUresult _cuInit(unsigned int Flags) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuInit
    cuPythonInit()
    if __cuInit == NULL:
        with gil:
            raise RuntimeError('Function "cuInit" not found')
    err = (<CUresult (*)(unsigned int) nogil> __cuInit)(Flags)
    return err

cdef CUresult _cuDriverGetVersion(int* driverVersion) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuDriverGetVersion
    cuPythonInit()
    if __cuDriverGetVersion == NULL:
        with gil:
            raise RuntimeError('Function "cuDriverGetVersion" not found')
    err = (<CUresult (*)(int*) nogil> __cuDriverGetVersion)(driverVersion)
    return err

cdef CUresult _cuDeviceGet(CUdevice* device, int ordinal) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuDeviceGet
    cuPythonInit()
    if __cuDeviceGet == NULL:
        with gil:
            raise RuntimeError('Function "cuDeviceGet" not found')
    err = (<CUresult (*)(CUdevice*, int) nogil> __cuDeviceGet)(device, ordinal)
    return err

cdef CUresult _cuDeviceGetCount(int* count) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuDeviceGetCount
    cuPythonInit()
    if __cuDeviceGetCount == NULL:
        with gil:
            raise RuntimeError('Function "cuDeviceGetCount" not found')
    err = (<CUresult (*)(int*) nogil> __cuDeviceGetCount)(count)
    return err

cdef CUresult _cuDeviceGetName(char* name, int length, CUdevice dev) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuDeviceGetName
    cuPythonInit()
    if __cuDeviceGetName == NULL:
        with gil:
            raise RuntimeError('Function "cuDeviceGetName" not found')
    err = (<CUresult (*)(char*, int, CUdevice) nogil> __cuDeviceGetName)(name, length, dev)
    return err

cdef CUresult _cuDeviceGetUuid(CUuuid* uuid, CUdevice dev) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuDeviceGetUuid
    cuPythonInit()
    if __cuDeviceGetUuid == NULL:
        with gil:
            raise RuntimeError('Function "cuDeviceGetUuid" not found')
    err = (<CUresult (*)(CUuuid*, CUdevice) nogil> __cuDeviceGetUuid)(uuid, dev)
    return err

cdef CUresult _cuDeviceGetUuid_v2(CUuuid* uuid, CUdevice dev) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuDeviceGetUuid_v2
    cuPythonInit()
    if __cuDeviceGetUuid_v2 == NULL:
        with gil:
            raise RuntimeError('Function "cuDeviceGetUuid_v2" not found')
    err = (<CUresult (*)(CUuuid*, CUdevice) nogil> __cuDeviceGetUuid_v2)(uuid, dev)
    return err

cdef CUresult _cuDeviceGetLuid(char* luid, unsigned int* deviceNodeMask, CUdevice dev) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuDeviceGetLuid
    cuPythonInit()
    if __cuDeviceGetLuid == NULL:
        with gil:
            raise RuntimeError('Function "cuDeviceGetLuid" not found')
    err = (<CUresult (*)(char*, unsigned int*, CUdevice) nogil> __cuDeviceGetLuid)(luid, deviceNodeMask, dev)
    return err

cdef CUresult _cuDeviceTotalMem_v2(size_t* numbytes, CUdevice dev) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuDeviceTotalMem_v2
    cuPythonInit()
    if __cuDeviceTotalMem_v2 == NULL:
        with gil:
            raise RuntimeError('Function "cuDeviceTotalMem_v2" not found')
    err = (<CUresult (*)(size_t*, CUdevice) nogil> __cuDeviceTotalMem_v2)(numbytes, dev)
    return err

cdef CUresult _cuDeviceGetTexture1DLinearMaxWidth(size_t* maxWidthInElements, CUarray_format pformat, unsigned numChannels, CUdevice dev) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuDeviceGetTexture1DLinearMaxWidth
    cuPythonInit()
    if __cuDeviceGetTexture1DLinearMaxWidth == NULL:
        with gil:
            raise RuntimeError('Function "cuDeviceGetTexture1DLinearMaxWidth" not found')
    err = (<CUresult (*)(size_t*, CUarray_format, unsigned, CUdevice) nogil> __cuDeviceGetTexture1DLinearMaxWidth)(maxWidthInElements, pformat, numChannels, dev)
    return err

cdef CUresult _cuDeviceGetAttribute(int* pi, CUdevice_attribute attrib, CUdevice dev) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuDeviceGetAttribute
    cuPythonInit()
    if __cuDeviceGetAttribute == NULL:
        with gil:
            raise RuntimeError('Function "cuDeviceGetAttribute" not found')
    err = (<CUresult (*)(int*, CUdevice_attribute, CUdevice) nogil> __cuDeviceGetAttribute)(pi, attrib, dev)
    return err

cdef CUresult _cuDeviceGetNvSciSyncAttributes(void* nvSciSyncAttrList, CUdevice dev, int flags) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuDeviceGetNvSciSyncAttributes
    cuPythonInit()
    if __cuDeviceGetNvSciSyncAttributes == NULL:
        with gil:
            raise RuntimeError('Function "cuDeviceGetNvSciSyncAttributes" not found')
    err = (<CUresult (*)(void*, CUdevice, int) nogil> __cuDeviceGetNvSciSyncAttributes)(nvSciSyncAttrList, dev, flags)
    return err

cdef CUresult _cuDeviceSetMemPool(CUdevice dev, CUmemoryPool pool) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuDeviceSetMemPool
    cuPythonInit()
    if __cuDeviceSetMemPool == NULL:
        with gil:
            raise RuntimeError('Function "cuDeviceSetMemPool" not found')
    err = (<CUresult (*)(CUdevice, CUmemoryPool) nogil> __cuDeviceSetMemPool)(dev, pool)
    return err

cdef CUresult _cuDeviceGetMemPool(CUmemoryPool* pool, CUdevice dev) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuDeviceGetMemPool
    cuPythonInit()
    if __cuDeviceGetMemPool == NULL:
        with gil:
            raise RuntimeError('Function "cuDeviceGetMemPool" not found')
    err = (<CUresult (*)(CUmemoryPool*, CUdevice) nogil> __cuDeviceGetMemPool)(pool, dev)
    return err

cdef CUresult _cuDeviceGetDefaultMemPool(CUmemoryPool* pool_out, CUdevice dev) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuDeviceGetDefaultMemPool
    cuPythonInit()
    if __cuDeviceGetDefaultMemPool == NULL:
        with gil:
            raise RuntimeError('Function "cuDeviceGetDefaultMemPool" not found')
    err = (<CUresult (*)(CUmemoryPool*, CUdevice) nogil> __cuDeviceGetDefaultMemPool)(pool_out, dev)
    return err

cdef CUresult _cuFlushGPUDirectRDMAWrites(CUflushGPUDirectRDMAWritesTarget target, CUflushGPUDirectRDMAWritesScope scope) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuFlushGPUDirectRDMAWrites
    cuPythonInit()
    if __cuFlushGPUDirectRDMAWrites == NULL:
        with gil:
            raise RuntimeError('Function "cuFlushGPUDirectRDMAWrites" not found')
    err = (<CUresult (*)(CUflushGPUDirectRDMAWritesTarget, CUflushGPUDirectRDMAWritesScope) nogil> __cuFlushGPUDirectRDMAWrites)(target, scope)
    return err

cdef CUresult _cuDeviceGetProperties(CUdevprop* prop, CUdevice dev) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuDeviceGetProperties
    cuPythonInit()
    if __cuDeviceGetProperties == NULL:
        with gil:
            raise RuntimeError('Function "cuDeviceGetProperties" not found')
    err = (<CUresult (*)(CUdevprop*, CUdevice) nogil> __cuDeviceGetProperties)(prop, dev)
    return err

cdef CUresult _cuDeviceComputeCapability(int* major, int* minor, CUdevice dev) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuDeviceComputeCapability
    cuPythonInit()
    if __cuDeviceComputeCapability == NULL:
        with gil:
            raise RuntimeError('Function "cuDeviceComputeCapability" not found')
    err = (<CUresult (*)(int*, int*, CUdevice) nogil> __cuDeviceComputeCapability)(major, minor, dev)
    return err

cdef CUresult _cuDevicePrimaryCtxRetain(CUcontext* pctx, CUdevice dev) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuDevicePrimaryCtxRetain
    cuPythonInit()
    if __cuDevicePrimaryCtxRetain == NULL:
        with gil:
            raise RuntimeError('Function "cuDevicePrimaryCtxRetain" not found')
    err = (<CUresult (*)(CUcontext*, CUdevice) nogil> __cuDevicePrimaryCtxRetain)(pctx, dev)
    return err

cdef CUresult _cuDevicePrimaryCtxRelease_v2(CUdevice dev) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuDevicePrimaryCtxRelease_v2
    cuPythonInit()
    if __cuDevicePrimaryCtxRelease_v2 == NULL:
        with gil:
            raise RuntimeError('Function "cuDevicePrimaryCtxRelease_v2" not found')
    err = (<CUresult (*)(CUdevice) nogil> __cuDevicePrimaryCtxRelease_v2)(dev)
    return err

cdef CUresult _cuDevicePrimaryCtxSetFlags_v2(CUdevice dev, unsigned int flags) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuDevicePrimaryCtxSetFlags_v2
    cuPythonInit()
    if __cuDevicePrimaryCtxSetFlags_v2 == NULL:
        with gil:
            raise RuntimeError('Function "cuDevicePrimaryCtxSetFlags_v2" not found')
    err = (<CUresult (*)(CUdevice, unsigned int) nogil> __cuDevicePrimaryCtxSetFlags_v2)(dev, flags)
    return err

cdef CUresult _cuDevicePrimaryCtxGetState(CUdevice dev, unsigned int* flags, int* active) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuDevicePrimaryCtxGetState
    cuPythonInit()
    if __cuDevicePrimaryCtxGetState == NULL:
        with gil:
            raise RuntimeError('Function "cuDevicePrimaryCtxGetState" not found')
    err = (<CUresult (*)(CUdevice, unsigned int*, int*) nogil> __cuDevicePrimaryCtxGetState)(dev, flags, active)
    return err

cdef CUresult _cuDevicePrimaryCtxReset_v2(CUdevice dev) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuDevicePrimaryCtxReset_v2
    cuPythonInit()
    if __cuDevicePrimaryCtxReset_v2 == NULL:
        with gil:
            raise RuntimeError('Function "cuDevicePrimaryCtxReset_v2" not found')
    err = (<CUresult (*)(CUdevice) nogil> __cuDevicePrimaryCtxReset_v2)(dev)
    return err

cdef CUresult _cuDeviceGetExecAffinitySupport(int* pi, CUexecAffinityType typename, CUdevice dev) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuDeviceGetExecAffinitySupport
    cuPythonInit()
    if __cuDeviceGetExecAffinitySupport == NULL:
        with gil:
            raise RuntimeError('Function "cuDeviceGetExecAffinitySupport" not found')
    err = (<CUresult (*)(int*, CUexecAffinityType, CUdevice) nogil> __cuDeviceGetExecAffinitySupport)(pi, typename, dev)
    return err

cdef CUresult _cuCtxCreate_v2(CUcontext* pctx, unsigned int flags, CUdevice dev) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuCtxCreate_v2
    cuPythonInit()
    if __cuCtxCreate_v2 == NULL:
        with gil:
            raise RuntimeError('Function "cuCtxCreate_v2" not found')
    err = (<CUresult (*)(CUcontext*, unsigned int, CUdevice) nogil> __cuCtxCreate_v2)(pctx, flags, dev)
    return err

cdef CUresult _cuCtxCreate_v3(CUcontext* pctx, CUexecAffinityParam* paramsArray, int numParams, unsigned int flags, CUdevice dev) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuCtxCreate_v3
    cuPythonInit()
    if __cuCtxCreate_v3 == NULL:
        with gil:
            raise RuntimeError('Function "cuCtxCreate_v3" not found')
    err = (<CUresult (*)(CUcontext*, CUexecAffinityParam*, int, unsigned int, CUdevice) nogil> __cuCtxCreate_v3)(pctx, paramsArray, numParams, flags, dev)
    return err

cdef CUresult _cuCtxDestroy_v2(CUcontext ctx) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuCtxDestroy_v2
    cuPythonInit()
    if __cuCtxDestroy_v2 == NULL:
        with gil:
            raise RuntimeError('Function "cuCtxDestroy_v2" not found')
    err = (<CUresult (*)(CUcontext) nogil> __cuCtxDestroy_v2)(ctx)
    return err

cdef CUresult _cuCtxPushCurrent_v2(CUcontext ctx) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuCtxPushCurrent_v2
    cuPythonInit()
    if __cuCtxPushCurrent_v2 == NULL:
        with gil:
            raise RuntimeError('Function "cuCtxPushCurrent_v2" not found')
    err = (<CUresult (*)(CUcontext) nogil> __cuCtxPushCurrent_v2)(ctx)
    return err

cdef CUresult _cuCtxPopCurrent_v2(CUcontext* pctx) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuCtxPopCurrent_v2
    cuPythonInit()
    if __cuCtxPopCurrent_v2 == NULL:
        with gil:
            raise RuntimeError('Function "cuCtxPopCurrent_v2" not found')
    err = (<CUresult (*)(CUcontext*) nogil> __cuCtxPopCurrent_v2)(pctx)
    return err

cdef CUresult _cuCtxSetCurrent(CUcontext ctx) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuCtxSetCurrent
    cuPythonInit()
    if __cuCtxSetCurrent == NULL:
        with gil:
            raise RuntimeError('Function "cuCtxSetCurrent" not found')
    err = (<CUresult (*)(CUcontext) nogil> __cuCtxSetCurrent)(ctx)
    return err

cdef CUresult _cuCtxGetCurrent(CUcontext* pctx) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuCtxGetCurrent
    cuPythonInit()
    if __cuCtxGetCurrent == NULL:
        with gil:
            raise RuntimeError('Function "cuCtxGetCurrent" not found')
    err = (<CUresult (*)(CUcontext*) nogil> __cuCtxGetCurrent)(pctx)
    return err

cdef CUresult _cuCtxGetDevice(CUdevice* device) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuCtxGetDevice
    cuPythonInit()
    if __cuCtxGetDevice == NULL:
        with gil:
            raise RuntimeError('Function "cuCtxGetDevice" not found')
    err = (<CUresult (*)(CUdevice*) nogil> __cuCtxGetDevice)(device)
    return err

cdef CUresult _cuCtxGetFlags(unsigned int* flags) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuCtxGetFlags
    cuPythonInit()
    if __cuCtxGetFlags == NULL:
        with gil:
            raise RuntimeError('Function "cuCtxGetFlags" not found')
    err = (<CUresult (*)(unsigned int*) nogil> __cuCtxGetFlags)(flags)
    return err

cdef CUresult _cuCtxSynchronize() nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuCtxSynchronize
    cuPythonInit()
    if __cuCtxSynchronize == NULL:
        with gil:
            raise RuntimeError('Function "cuCtxSynchronize" not found')
    err = (<CUresult (*)() nogil> __cuCtxSynchronize)()
    return err

cdef CUresult _cuCtxSetLimit(CUlimit limit, size_t value) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuCtxSetLimit
    cuPythonInit()
    if __cuCtxSetLimit == NULL:
        with gil:
            raise RuntimeError('Function "cuCtxSetLimit" not found')
    err = (<CUresult (*)(CUlimit, size_t) nogil> __cuCtxSetLimit)(limit, value)
    return err

cdef CUresult _cuCtxGetLimit(size_t* pvalue, CUlimit limit) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuCtxGetLimit
    cuPythonInit()
    if __cuCtxGetLimit == NULL:
        with gil:
            raise RuntimeError('Function "cuCtxGetLimit" not found')
    err = (<CUresult (*)(size_t*, CUlimit) nogil> __cuCtxGetLimit)(pvalue, limit)
    return err

cdef CUresult _cuCtxGetCacheConfig(CUfunc_cache* pconfig) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuCtxGetCacheConfig
    cuPythonInit()
    if __cuCtxGetCacheConfig == NULL:
        with gil:
            raise RuntimeError('Function "cuCtxGetCacheConfig" not found')
    err = (<CUresult (*)(CUfunc_cache*) nogil> __cuCtxGetCacheConfig)(pconfig)
    return err

cdef CUresult _cuCtxSetCacheConfig(CUfunc_cache config) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuCtxSetCacheConfig
    cuPythonInit()
    if __cuCtxSetCacheConfig == NULL:
        with gil:
            raise RuntimeError('Function "cuCtxSetCacheConfig" not found')
    err = (<CUresult (*)(CUfunc_cache) nogil> __cuCtxSetCacheConfig)(config)
    return err

cdef CUresult _cuCtxGetSharedMemConfig(CUsharedconfig* pConfig) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuCtxGetSharedMemConfig
    cuPythonInit()
    if __cuCtxGetSharedMemConfig == NULL:
        with gil:
            raise RuntimeError('Function "cuCtxGetSharedMemConfig" not found')
    err = (<CUresult (*)(CUsharedconfig*) nogil> __cuCtxGetSharedMemConfig)(pConfig)
    return err

cdef CUresult _cuCtxSetSharedMemConfig(CUsharedconfig config) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuCtxSetSharedMemConfig
    cuPythonInit()
    if __cuCtxSetSharedMemConfig == NULL:
        with gil:
            raise RuntimeError('Function "cuCtxSetSharedMemConfig" not found')
    err = (<CUresult (*)(CUsharedconfig) nogil> __cuCtxSetSharedMemConfig)(config)
    return err

cdef CUresult _cuCtxGetApiVersion(CUcontext ctx, unsigned int* version) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuCtxGetApiVersion
    cuPythonInit()
    if __cuCtxGetApiVersion == NULL:
        with gil:
            raise RuntimeError('Function "cuCtxGetApiVersion" not found')
    err = (<CUresult (*)(CUcontext, unsigned int*) nogil> __cuCtxGetApiVersion)(ctx, version)
    return err

cdef CUresult _cuCtxGetStreamPriorityRange(int* leastPriority, int* greatestPriority) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuCtxGetStreamPriorityRange
    cuPythonInit()
    if __cuCtxGetStreamPriorityRange == NULL:
        with gil:
            raise RuntimeError('Function "cuCtxGetStreamPriorityRange" not found')
    err = (<CUresult (*)(int*, int*) nogil> __cuCtxGetStreamPriorityRange)(leastPriority, greatestPriority)
    return err

cdef CUresult _cuCtxResetPersistingL2Cache() nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuCtxResetPersistingL2Cache
    cuPythonInit()
    if __cuCtxResetPersistingL2Cache == NULL:
        with gil:
            raise RuntimeError('Function "cuCtxResetPersistingL2Cache" not found')
    err = (<CUresult (*)() nogil> __cuCtxResetPersistingL2Cache)()
    return err

cdef CUresult _cuCtxGetExecAffinity(CUexecAffinityParam* pExecAffinity, CUexecAffinityType typename) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuCtxGetExecAffinity
    cuPythonInit()
    if __cuCtxGetExecAffinity == NULL:
        with gil:
            raise RuntimeError('Function "cuCtxGetExecAffinity" not found')
    err = (<CUresult (*)(CUexecAffinityParam*, CUexecAffinityType) nogil> __cuCtxGetExecAffinity)(pExecAffinity, typename)
    return err

cdef CUresult _cuCtxAttach(CUcontext* pctx, unsigned int flags) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuCtxAttach
    cuPythonInit()
    if __cuCtxAttach == NULL:
        with gil:
            raise RuntimeError('Function "cuCtxAttach" not found')
    err = (<CUresult (*)(CUcontext*, unsigned int) nogil> __cuCtxAttach)(pctx, flags)
    return err

cdef CUresult _cuCtxDetach(CUcontext ctx) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuCtxDetach
    cuPythonInit()
    if __cuCtxDetach == NULL:
        with gil:
            raise RuntimeError('Function "cuCtxDetach" not found')
    err = (<CUresult (*)(CUcontext) nogil> __cuCtxDetach)(ctx)
    return err

cdef CUresult _cuModuleLoad(CUmodule* module, const char* fname) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuModuleLoad
    cuPythonInit()
    if __cuModuleLoad == NULL:
        with gil:
            raise RuntimeError('Function "cuModuleLoad" not found')
    err = (<CUresult (*)(CUmodule*, const char*) nogil> __cuModuleLoad)(module, fname)
    return err

cdef CUresult _cuModuleLoadData(CUmodule* module, const void* image) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuModuleLoadData
    cuPythonInit()
    if __cuModuleLoadData == NULL:
        with gil:
            raise RuntimeError('Function "cuModuleLoadData" not found')
    err = (<CUresult (*)(CUmodule*, const void*) nogil> __cuModuleLoadData)(module, image)
    return err

cdef CUresult _cuModuleLoadDataEx(CUmodule* module, const void* image, unsigned int numOptions, CUjit_option* options, void** optionValues) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuModuleLoadDataEx
    cuPythonInit()
    if __cuModuleLoadDataEx == NULL:
        with gil:
            raise RuntimeError('Function "cuModuleLoadDataEx" not found')
    err = (<CUresult (*)(CUmodule*, const void*, unsigned int, CUjit_option*, void**) nogil> __cuModuleLoadDataEx)(module, image, numOptions, options, optionValues)
    return err

cdef CUresult _cuModuleLoadFatBinary(CUmodule* module, const void* fatCubin) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuModuleLoadFatBinary
    cuPythonInit()
    if __cuModuleLoadFatBinary == NULL:
        with gil:
            raise RuntimeError('Function "cuModuleLoadFatBinary" not found')
    err = (<CUresult (*)(CUmodule*, const void*) nogil> __cuModuleLoadFatBinary)(module, fatCubin)
    return err

cdef CUresult _cuModuleUnload(CUmodule hmod) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuModuleUnload
    cuPythonInit()
    if __cuModuleUnload == NULL:
        with gil:
            raise RuntimeError('Function "cuModuleUnload" not found')
    err = (<CUresult (*)(CUmodule) nogil> __cuModuleUnload)(hmod)
    return err

cdef CUresult _cuModuleGetFunction(CUfunction* hfunc, CUmodule hmod, const char* name) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuModuleGetFunction
    cuPythonInit()
    if __cuModuleGetFunction == NULL:
        with gil:
            raise RuntimeError('Function "cuModuleGetFunction" not found')
    err = (<CUresult (*)(CUfunction*, CUmodule, const char*) nogil> __cuModuleGetFunction)(hfunc, hmod, name)
    return err

cdef CUresult _cuModuleGetGlobal_v2(CUdeviceptr* dptr, size_t* numbytes, CUmodule hmod, const char* name) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuModuleGetGlobal_v2
    cuPythonInit()
    if __cuModuleGetGlobal_v2 == NULL:
        with gil:
            raise RuntimeError('Function "cuModuleGetGlobal_v2" not found')
    err = (<CUresult (*)(CUdeviceptr*, size_t*, CUmodule, const char*) nogil> __cuModuleGetGlobal_v2)(dptr, numbytes, hmod, name)
    return err

cdef CUresult _cuModuleGetTexRef(CUtexref* pTexRef, CUmodule hmod, const char* name) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuModuleGetTexRef
    cuPythonInit()
    if __cuModuleGetTexRef == NULL:
        with gil:
            raise RuntimeError('Function "cuModuleGetTexRef" not found')
    err = (<CUresult (*)(CUtexref*, CUmodule, const char*) nogil> __cuModuleGetTexRef)(pTexRef, hmod, name)
    return err

cdef CUresult _cuModuleGetSurfRef(CUsurfref* pSurfRef, CUmodule hmod, const char* name) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuModuleGetSurfRef
    cuPythonInit()
    if __cuModuleGetSurfRef == NULL:
        with gil:
            raise RuntimeError('Function "cuModuleGetSurfRef" not found')
    err = (<CUresult (*)(CUsurfref*, CUmodule, const char*) nogil> __cuModuleGetSurfRef)(pSurfRef, hmod, name)
    return err

cdef CUresult _cuLinkCreate_v2(unsigned int numOptions, CUjit_option* options, void** optionValues, CUlinkState* stateOut) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuLinkCreate_v2
    cuPythonInit()
    if __cuLinkCreate_v2 == NULL:
        with gil:
            raise RuntimeError('Function "cuLinkCreate_v2" not found')
    err = (<CUresult (*)(unsigned int, CUjit_option*, void**, CUlinkState*) nogil> __cuLinkCreate_v2)(numOptions, options, optionValues, stateOut)
    return err

cdef CUresult _cuLinkAddData_v2(CUlinkState state, CUjitInputType typename, void* data, size_t size, const char* name, unsigned int numOptions, CUjit_option* options, void** optionValues) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuLinkAddData_v2
    cuPythonInit()
    if __cuLinkAddData_v2 == NULL:
        with gil:
            raise RuntimeError('Function "cuLinkAddData_v2" not found')
    err = (<CUresult (*)(CUlinkState, CUjitInputType, void*, size_t, const char*, unsigned int, CUjit_option*, void**) nogil> __cuLinkAddData_v2)(state, typename, data, size, name, numOptions, options, optionValues)
    return err

cdef CUresult _cuLinkAddFile_v2(CUlinkState state, CUjitInputType typename, const char* path, unsigned int numOptions, CUjit_option* options, void** optionValues) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuLinkAddFile_v2
    cuPythonInit()
    if __cuLinkAddFile_v2 == NULL:
        with gil:
            raise RuntimeError('Function "cuLinkAddFile_v2" not found')
    err = (<CUresult (*)(CUlinkState, CUjitInputType, const char*, unsigned int, CUjit_option*, void**) nogil> __cuLinkAddFile_v2)(state, typename, path, numOptions, options, optionValues)
    return err

cdef CUresult _cuLinkComplete(CUlinkState state, void** cubinOut, size_t* sizeOut) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuLinkComplete
    cuPythonInit()
    if __cuLinkComplete == NULL:
        with gil:
            raise RuntimeError('Function "cuLinkComplete" not found')
    err = (<CUresult (*)(CUlinkState, void**, size_t*) nogil> __cuLinkComplete)(state, cubinOut, sizeOut)
    return err

cdef CUresult _cuLinkDestroy(CUlinkState state) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuLinkDestroy
    cuPythonInit()
    if __cuLinkDestroy == NULL:
        with gil:
            raise RuntimeError('Function "cuLinkDestroy" not found')
    err = (<CUresult (*)(CUlinkState) nogil> __cuLinkDestroy)(state)
    return err

cdef CUresult _cuMemGetInfo_v2(size_t* free, size_t* total) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuMemGetInfo_v2
    cuPythonInit()
    if __cuMemGetInfo_v2 == NULL:
        with gil:
            raise RuntimeError('Function "cuMemGetInfo_v2" not found')
    err = (<CUresult (*)(size_t*, size_t*) nogil> __cuMemGetInfo_v2)(free, total)
    return err

cdef CUresult _cuMemAlloc_v2(CUdeviceptr* dptr, size_t bytesize) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuMemAlloc_v2
    cuPythonInit()
    if __cuMemAlloc_v2 == NULL:
        with gil:
            raise RuntimeError('Function "cuMemAlloc_v2" not found')
    err = (<CUresult (*)(CUdeviceptr*, size_t) nogil> __cuMemAlloc_v2)(dptr, bytesize)
    return err

cdef CUresult _cuMemAllocPitch_v2(CUdeviceptr* dptr, size_t* pPitch, size_t WidthInBytes, size_t Height, unsigned int ElementSizeBytes) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuMemAllocPitch_v2
    cuPythonInit()
    if __cuMemAllocPitch_v2 == NULL:
        with gil:
            raise RuntimeError('Function "cuMemAllocPitch_v2" not found')
    err = (<CUresult (*)(CUdeviceptr*, size_t*, size_t, size_t, unsigned int) nogil> __cuMemAllocPitch_v2)(dptr, pPitch, WidthInBytes, Height, ElementSizeBytes)
    return err

cdef CUresult _cuMemFree_v2(CUdeviceptr dptr) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuMemFree_v2
    cuPythonInit()
    if __cuMemFree_v2 == NULL:
        with gil:
            raise RuntimeError('Function "cuMemFree_v2" not found')
    err = (<CUresult (*)(CUdeviceptr) nogil> __cuMemFree_v2)(dptr)
    return err

cdef CUresult _cuMemGetAddressRange_v2(CUdeviceptr* pbase, size_t* psize, CUdeviceptr dptr) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuMemGetAddressRange_v2
    cuPythonInit()
    if __cuMemGetAddressRange_v2 == NULL:
        with gil:
            raise RuntimeError('Function "cuMemGetAddressRange_v2" not found')
    err = (<CUresult (*)(CUdeviceptr*, size_t*, CUdeviceptr) nogil> __cuMemGetAddressRange_v2)(pbase, psize, dptr)
    return err

cdef CUresult _cuMemAllocHost_v2(void** pp, size_t bytesize) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuMemAllocHost_v2
    cuPythonInit()
    if __cuMemAllocHost_v2 == NULL:
        with gil:
            raise RuntimeError('Function "cuMemAllocHost_v2" not found')
    err = (<CUresult (*)(void**, size_t) nogil> __cuMemAllocHost_v2)(pp, bytesize)
    return err

cdef CUresult _cuMemFreeHost(void* p) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuMemFreeHost
    cuPythonInit()
    if __cuMemFreeHost == NULL:
        with gil:
            raise RuntimeError('Function "cuMemFreeHost" not found')
    err = (<CUresult (*)(void*) nogil> __cuMemFreeHost)(p)
    return err

cdef CUresult _cuMemHostAlloc(void** pp, size_t bytesize, unsigned int Flags) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuMemHostAlloc
    cuPythonInit()
    if __cuMemHostAlloc == NULL:
        with gil:
            raise RuntimeError('Function "cuMemHostAlloc" not found')
    err = (<CUresult (*)(void**, size_t, unsigned int) nogil> __cuMemHostAlloc)(pp, bytesize, Flags)
    return err

cdef CUresult _cuMemHostGetDevicePointer_v2(CUdeviceptr* pdptr, void* p, unsigned int Flags) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuMemHostGetDevicePointer_v2
    cuPythonInit()
    if __cuMemHostGetDevicePointer_v2 == NULL:
        with gil:
            raise RuntimeError('Function "cuMemHostGetDevicePointer_v2" not found')
    err = (<CUresult (*)(CUdeviceptr*, void*, unsigned int) nogil> __cuMemHostGetDevicePointer_v2)(pdptr, p, Flags)
    return err

cdef CUresult _cuMemHostGetFlags(unsigned int* pFlags, void* p) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuMemHostGetFlags
    cuPythonInit()
    if __cuMemHostGetFlags == NULL:
        with gil:
            raise RuntimeError('Function "cuMemHostGetFlags" not found')
    err = (<CUresult (*)(unsigned int*, void*) nogil> __cuMemHostGetFlags)(pFlags, p)
    return err

cdef CUresult _cuMemAllocManaged(CUdeviceptr* dptr, size_t bytesize, unsigned int flags) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuMemAllocManaged
    cuPythonInit()
    if __cuMemAllocManaged == NULL:
        with gil:
            raise RuntimeError('Function "cuMemAllocManaged" not found')
    err = (<CUresult (*)(CUdeviceptr*, size_t, unsigned int) nogil> __cuMemAllocManaged)(dptr, bytesize, flags)
    return err

cdef CUresult _cuDeviceGetByPCIBusId(CUdevice* dev, const char* pciBusId) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuDeviceGetByPCIBusId
    cuPythonInit()
    if __cuDeviceGetByPCIBusId == NULL:
        with gil:
            raise RuntimeError('Function "cuDeviceGetByPCIBusId" not found')
    err = (<CUresult (*)(CUdevice*, const char*) nogil> __cuDeviceGetByPCIBusId)(dev, pciBusId)
    return err

cdef CUresult _cuDeviceGetPCIBusId(char* pciBusId, int length, CUdevice dev) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuDeviceGetPCIBusId
    cuPythonInit()
    if __cuDeviceGetPCIBusId == NULL:
        with gil:
            raise RuntimeError('Function "cuDeviceGetPCIBusId" not found')
    err = (<CUresult (*)(char*, int, CUdevice) nogil> __cuDeviceGetPCIBusId)(pciBusId, length, dev)
    return err

cdef CUresult _cuIpcGetEventHandle(CUipcEventHandle* pHandle, CUevent event) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuIpcGetEventHandle
    cuPythonInit()
    if __cuIpcGetEventHandle == NULL:
        with gil:
            raise RuntimeError('Function "cuIpcGetEventHandle" not found')
    err = (<CUresult (*)(CUipcEventHandle*, CUevent) nogil> __cuIpcGetEventHandle)(pHandle, event)
    return err

cdef CUresult _cuIpcOpenEventHandle(CUevent* phEvent, CUipcEventHandle handle) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuIpcOpenEventHandle
    cuPythonInit()
    if __cuIpcOpenEventHandle == NULL:
        with gil:
            raise RuntimeError('Function "cuIpcOpenEventHandle" not found')
    err = (<CUresult (*)(CUevent*, CUipcEventHandle) nogil> __cuIpcOpenEventHandle)(phEvent, handle)
    return err

cdef CUresult _cuIpcGetMemHandle(CUipcMemHandle* pHandle, CUdeviceptr dptr) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuIpcGetMemHandle
    cuPythonInit()
    if __cuIpcGetMemHandle == NULL:
        with gil:
            raise RuntimeError('Function "cuIpcGetMemHandle" not found')
    err = (<CUresult (*)(CUipcMemHandle*, CUdeviceptr) nogil> __cuIpcGetMemHandle)(pHandle, dptr)
    return err

cdef CUresult _cuIpcOpenMemHandle_v2(CUdeviceptr* pdptr, CUipcMemHandle handle, unsigned int Flags) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuIpcOpenMemHandle_v2
    cuPythonInit()
    if __cuIpcOpenMemHandle_v2 == NULL:
        with gil:
            raise RuntimeError('Function "cuIpcOpenMemHandle_v2" not found')
    err = (<CUresult (*)(CUdeviceptr*, CUipcMemHandle, unsigned int) nogil> __cuIpcOpenMemHandle_v2)(pdptr, handle, Flags)
    return err

cdef CUresult _cuIpcCloseMemHandle(CUdeviceptr dptr) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuIpcCloseMemHandle
    cuPythonInit()
    if __cuIpcCloseMemHandle == NULL:
        with gil:
            raise RuntimeError('Function "cuIpcCloseMemHandle" not found')
    err = (<CUresult (*)(CUdeviceptr) nogil> __cuIpcCloseMemHandle)(dptr)
    return err

cdef CUresult _cuMemHostRegister_v2(void* p, size_t bytesize, unsigned int Flags) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuMemHostRegister_v2
    cuPythonInit()
    if __cuMemHostRegister_v2 == NULL:
        with gil:
            raise RuntimeError('Function "cuMemHostRegister_v2" not found')
    err = (<CUresult (*)(void*, size_t, unsigned int) nogil> __cuMemHostRegister_v2)(p, bytesize, Flags)
    return err

cdef CUresult _cuMemHostUnregister(void* p) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuMemHostUnregister
    cuPythonInit()
    if __cuMemHostUnregister == NULL:
        with gil:
            raise RuntimeError('Function "cuMemHostUnregister" not found')
    err = (<CUresult (*)(void*) nogil> __cuMemHostUnregister)(p)
    return err

cdef CUresult _cuMemcpy(CUdeviceptr dst, CUdeviceptr src, size_t ByteCount) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuMemcpy
    cuPythonInit()
    if __cuMemcpy == NULL:
        with gil:
            raise RuntimeError('Function "cuMemcpy" not found')
    err = (<CUresult (*)(CUdeviceptr, CUdeviceptr, size_t) nogil> __cuMemcpy)(dst, src, ByteCount)
    return err

cdef CUresult _cuMemcpyPeer(CUdeviceptr dstDevice, CUcontext dstContext, CUdeviceptr srcDevice, CUcontext srcContext, size_t ByteCount) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuMemcpyPeer
    cuPythonInit()
    if __cuMemcpyPeer == NULL:
        with gil:
            raise RuntimeError('Function "cuMemcpyPeer" not found')
    err = (<CUresult (*)(CUdeviceptr, CUcontext, CUdeviceptr, CUcontext, size_t) nogil> __cuMemcpyPeer)(dstDevice, dstContext, srcDevice, srcContext, ByteCount)
    return err

cdef CUresult _cuMemcpyHtoD_v2(CUdeviceptr dstDevice, const void* srcHost, size_t ByteCount) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuMemcpyHtoD_v2
    cuPythonInit()
    if __cuMemcpyHtoD_v2 == NULL:
        with gil:
            raise RuntimeError('Function "cuMemcpyHtoD_v2" not found')
    err = (<CUresult (*)(CUdeviceptr, const void*, size_t) nogil> __cuMemcpyHtoD_v2)(dstDevice, srcHost, ByteCount)
    return err

cdef CUresult _cuMemcpyDtoH_v2(void* dstHost, CUdeviceptr srcDevice, size_t ByteCount) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuMemcpyDtoH_v2
    cuPythonInit()
    if __cuMemcpyDtoH_v2 == NULL:
        with gil:
            raise RuntimeError('Function "cuMemcpyDtoH_v2" not found')
    err = (<CUresult (*)(void*, CUdeviceptr, size_t) nogil> __cuMemcpyDtoH_v2)(dstHost, srcDevice, ByteCount)
    return err

cdef CUresult _cuMemcpyDtoD_v2(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuMemcpyDtoD_v2
    cuPythonInit()
    if __cuMemcpyDtoD_v2 == NULL:
        with gil:
            raise RuntimeError('Function "cuMemcpyDtoD_v2" not found')
    err = (<CUresult (*)(CUdeviceptr, CUdeviceptr, size_t) nogil> __cuMemcpyDtoD_v2)(dstDevice, srcDevice, ByteCount)
    return err

cdef CUresult _cuMemcpyDtoA_v2(CUarray dstArray, size_t dstOffset, CUdeviceptr srcDevice, size_t ByteCount) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuMemcpyDtoA_v2
    cuPythonInit()
    if __cuMemcpyDtoA_v2 == NULL:
        with gil:
            raise RuntimeError('Function "cuMemcpyDtoA_v2" not found')
    err = (<CUresult (*)(CUarray, size_t, CUdeviceptr, size_t) nogil> __cuMemcpyDtoA_v2)(dstArray, dstOffset, srcDevice, ByteCount)
    return err

cdef CUresult _cuMemcpyAtoD_v2(CUdeviceptr dstDevice, CUarray srcArray, size_t srcOffset, size_t ByteCount) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuMemcpyAtoD_v2
    cuPythonInit()
    if __cuMemcpyAtoD_v2 == NULL:
        with gil:
            raise RuntimeError('Function "cuMemcpyAtoD_v2" not found')
    err = (<CUresult (*)(CUdeviceptr, CUarray, size_t, size_t) nogil> __cuMemcpyAtoD_v2)(dstDevice, srcArray, srcOffset, ByteCount)
    return err

cdef CUresult _cuMemcpyHtoA_v2(CUarray dstArray, size_t dstOffset, const void* srcHost, size_t ByteCount) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuMemcpyHtoA_v2
    cuPythonInit()
    if __cuMemcpyHtoA_v2 == NULL:
        with gil:
            raise RuntimeError('Function "cuMemcpyHtoA_v2" not found')
    err = (<CUresult (*)(CUarray, size_t, const void*, size_t) nogil> __cuMemcpyHtoA_v2)(dstArray, dstOffset, srcHost, ByteCount)
    return err

cdef CUresult _cuMemcpyAtoH_v2(void* dstHost, CUarray srcArray, size_t srcOffset, size_t ByteCount) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuMemcpyAtoH_v2
    cuPythonInit()
    if __cuMemcpyAtoH_v2 == NULL:
        with gil:
            raise RuntimeError('Function "cuMemcpyAtoH_v2" not found')
    err = (<CUresult (*)(void*, CUarray, size_t, size_t) nogil> __cuMemcpyAtoH_v2)(dstHost, srcArray, srcOffset, ByteCount)
    return err

cdef CUresult _cuMemcpyAtoA_v2(CUarray dstArray, size_t dstOffset, CUarray srcArray, size_t srcOffset, size_t ByteCount) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuMemcpyAtoA_v2
    cuPythonInit()
    if __cuMemcpyAtoA_v2 == NULL:
        with gil:
            raise RuntimeError('Function "cuMemcpyAtoA_v2" not found')
    err = (<CUresult (*)(CUarray, size_t, CUarray, size_t, size_t) nogil> __cuMemcpyAtoA_v2)(dstArray, dstOffset, srcArray, srcOffset, ByteCount)
    return err

cdef CUresult _cuMemcpy2D_v2(const CUDA_MEMCPY2D* pCopy) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuMemcpy2D_v2
    cuPythonInit()
    if __cuMemcpy2D_v2 == NULL:
        with gil:
            raise RuntimeError('Function "cuMemcpy2D_v2" not found')
    err = (<CUresult (*)(const CUDA_MEMCPY2D*) nogil> __cuMemcpy2D_v2)(pCopy)
    return err

cdef CUresult _cuMemcpy2DUnaligned_v2(const CUDA_MEMCPY2D* pCopy) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuMemcpy2DUnaligned_v2
    cuPythonInit()
    if __cuMemcpy2DUnaligned_v2 == NULL:
        with gil:
            raise RuntimeError('Function "cuMemcpy2DUnaligned_v2" not found')
    err = (<CUresult (*)(const CUDA_MEMCPY2D*) nogil> __cuMemcpy2DUnaligned_v2)(pCopy)
    return err

cdef CUresult _cuMemcpy3D_v2(const CUDA_MEMCPY3D* pCopy) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuMemcpy3D_v2
    cuPythonInit()
    if __cuMemcpy3D_v2 == NULL:
        with gil:
            raise RuntimeError('Function "cuMemcpy3D_v2" not found')
    err = (<CUresult (*)(const CUDA_MEMCPY3D*) nogil> __cuMemcpy3D_v2)(pCopy)
    return err

cdef CUresult _cuMemcpy3DPeer(const CUDA_MEMCPY3D_PEER* pCopy) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuMemcpy3DPeer
    cuPythonInit()
    if __cuMemcpy3DPeer == NULL:
        with gil:
            raise RuntimeError('Function "cuMemcpy3DPeer" not found')
    err = (<CUresult (*)(const CUDA_MEMCPY3D_PEER*) nogil> __cuMemcpy3DPeer)(pCopy)
    return err

cdef CUresult _cuMemcpyAsync(CUdeviceptr dst, CUdeviceptr src, size_t ByteCount, CUstream hStream) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuMemcpyAsync
    cuPythonInit()
    if __cuMemcpyAsync == NULL:
        with gil:
            raise RuntimeError('Function "cuMemcpyAsync" not found')
    err = (<CUresult (*)(CUdeviceptr, CUdeviceptr, size_t, CUstream) nogil> __cuMemcpyAsync)(dst, src, ByteCount, hStream)
    return err

cdef CUresult _cuMemcpyPeerAsync(CUdeviceptr dstDevice, CUcontext dstContext, CUdeviceptr srcDevice, CUcontext srcContext, size_t ByteCount, CUstream hStream) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuMemcpyPeerAsync
    cuPythonInit()
    if __cuMemcpyPeerAsync == NULL:
        with gil:
            raise RuntimeError('Function "cuMemcpyPeerAsync" not found')
    err = (<CUresult (*)(CUdeviceptr, CUcontext, CUdeviceptr, CUcontext, size_t, CUstream) nogil> __cuMemcpyPeerAsync)(dstDevice, dstContext, srcDevice, srcContext, ByteCount, hStream)
    return err

cdef CUresult _cuMemcpyHtoDAsync_v2(CUdeviceptr dstDevice, const void* srcHost, size_t ByteCount, CUstream hStream) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuMemcpyHtoDAsync_v2
    cuPythonInit()
    if __cuMemcpyHtoDAsync_v2 == NULL:
        with gil:
            raise RuntimeError('Function "cuMemcpyHtoDAsync_v2" not found')
    err = (<CUresult (*)(CUdeviceptr, const void*, size_t, CUstream) nogil> __cuMemcpyHtoDAsync_v2)(dstDevice, srcHost, ByteCount, hStream)
    return err

cdef CUresult _cuMemcpyDtoHAsync_v2(void* dstHost, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuMemcpyDtoHAsync_v2
    cuPythonInit()
    if __cuMemcpyDtoHAsync_v2 == NULL:
        with gil:
            raise RuntimeError('Function "cuMemcpyDtoHAsync_v2" not found')
    err = (<CUresult (*)(void*, CUdeviceptr, size_t, CUstream) nogil> __cuMemcpyDtoHAsync_v2)(dstHost, srcDevice, ByteCount, hStream)
    return err

cdef CUresult _cuMemcpyDtoDAsync_v2(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuMemcpyDtoDAsync_v2
    cuPythonInit()
    if __cuMemcpyDtoDAsync_v2 == NULL:
        with gil:
            raise RuntimeError('Function "cuMemcpyDtoDAsync_v2" not found')
    err = (<CUresult (*)(CUdeviceptr, CUdeviceptr, size_t, CUstream) nogil> __cuMemcpyDtoDAsync_v2)(dstDevice, srcDevice, ByteCount, hStream)
    return err

cdef CUresult _cuMemcpyHtoAAsync_v2(CUarray dstArray, size_t dstOffset, const void* srcHost, size_t ByteCount, CUstream hStream) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuMemcpyHtoAAsync_v2
    cuPythonInit()
    if __cuMemcpyHtoAAsync_v2 == NULL:
        with gil:
            raise RuntimeError('Function "cuMemcpyHtoAAsync_v2" not found')
    err = (<CUresult (*)(CUarray, size_t, const void*, size_t, CUstream) nogil> __cuMemcpyHtoAAsync_v2)(dstArray, dstOffset, srcHost, ByteCount, hStream)
    return err

cdef CUresult _cuMemcpyAtoHAsync_v2(void* dstHost, CUarray srcArray, size_t srcOffset, size_t ByteCount, CUstream hStream) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuMemcpyAtoHAsync_v2
    cuPythonInit()
    if __cuMemcpyAtoHAsync_v2 == NULL:
        with gil:
            raise RuntimeError('Function "cuMemcpyAtoHAsync_v2" not found')
    err = (<CUresult (*)(void*, CUarray, size_t, size_t, CUstream) nogil> __cuMemcpyAtoHAsync_v2)(dstHost, srcArray, srcOffset, ByteCount, hStream)
    return err

cdef CUresult _cuMemcpy2DAsync_v2(const CUDA_MEMCPY2D* pCopy, CUstream hStream) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuMemcpy2DAsync_v2
    cuPythonInit()
    if __cuMemcpy2DAsync_v2 == NULL:
        with gil:
            raise RuntimeError('Function "cuMemcpy2DAsync_v2" not found')
    err = (<CUresult (*)(const CUDA_MEMCPY2D*, CUstream) nogil> __cuMemcpy2DAsync_v2)(pCopy, hStream)
    return err

cdef CUresult _cuMemcpy3DAsync_v2(const CUDA_MEMCPY3D* pCopy, CUstream hStream) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuMemcpy3DAsync_v2
    cuPythonInit()
    if __cuMemcpy3DAsync_v2 == NULL:
        with gil:
            raise RuntimeError('Function "cuMemcpy3DAsync_v2" not found')
    err = (<CUresult (*)(const CUDA_MEMCPY3D*, CUstream) nogil> __cuMemcpy3DAsync_v2)(pCopy, hStream)
    return err

cdef CUresult _cuMemcpy3DPeerAsync(const CUDA_MEMCPY3D_PEER* pCopy, CUstream hStream) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuMemcpy3DPeerAsync
    cuPythonInit()
    if __cuMemcpy3DPeerAsync == NULL:
        with gil:
            raise RuntimeError('Function "cuMemcpy3DPeerAsync" not found')
    err = (<CUresult (*)(const CUDA_MEMCPY3D_PEER*, CUstream) nogil> __cuMemcpy3DPeerAsync)(pCopy, hStream)
    return err

cdef CUresult _cuMemsetD8_v2(CUdeviceptr dstDevice, unsigned char uc, size_t N) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuMemsetD8_v2
    cuPythonInit()
    if __cuMemsetD8_v2 == NULL:
        with gil:
            raise RuntimeError('Function "cuMemsetD8_v2" not found')
    err = (<CUresult (*)(CUdeviceptr, unsigned char, size_t) nogil> __cuMemsetD8_v2)(dstDevice, uc, N)
    return err

cdef CUresult _cuMemsetD16_v2(CUdeviceptr dstDevice, unsigned short us, size_t N) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuMemsetD16_v2
    cuPythonInit()
    if __cuMemsetD16_v2 == NULL:
        with gil:
            raise RuntimeError('Function "cuMemsetD16_v2" not found')
    err = (<CUresult (*)(CUdeviceptr, unsigned short, size_t) nogil> __cuMemsetD16_v2)(dstDevice, us, N)
    return err

cdef CUresult _cuMemsetD32_v2(CUdeviceptr dstDevice, unsigned int ui, size_t N) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuMemsetD32_v2
    cuPythonInit()
    if __cuMemsetD32_v2 == NULL:
        with gil:
            raise RuntimeError('Function "cuMemsetD32_v2" not found')
    err = (<CUresult (*)(CUdeviceptr, unsigned int, size_t) nogil> __cuMemsetD32_v2)(dstDevice, ui, N)
    return err

cdef CUresult _cuMemsetD2D8_v2(CUdeviceptr dstDevice, size_t dstPitch, unsigned char uc, size_t Width, size_t Height) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuMemsetD2D8_v2
    cuPythonInit()
    if __cuMemsetD2D8_v2 == NULL:
        with gil:
            raise RuntimeError('Function "cuMemsetD2D8_v2" not found')
    err = (<CUresult (*)(CUdeviceptr, size_t, unsigned char, size_t, size_t) nogil> __cuMemsetD2D8_v2)(dstDevice, dstPitch, uc, Width, Height)
    return err

cdef CUresult _cuMemsetD2D16_v2(CUdeviceptr dstDevice, size_t dstPitch, unsigned short us, size_t Width, size_t Height) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuMemsetD2D16_v2
    cuPythonInit()
    if __cuMemsetD2D16_v2 == NULL:
        with gil:
            raise RuntimeError('Function "cuMemsetD2D16_v2" not found')
    err = (<CUresult (*)(CUdeviceptr, size_t, unsigned short, size_t, size_t) nogil> __cuMemsetD2D16_v2)(dstDevice, dstPitch, us, Width, Height)
    return err

cdef CUresult _cuMemsetD2D32_v2(CUdeviceptr dstDevice, size_t dstPitch, unsigned int ui, size_t Width, size_t Height) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuMemsetD2D32_v2
    cuPythonInit()
    if __cuMemsetD2D32_v2 == NULL:
        with gil:
            raise RuntimeError('Function "cuMemsetD2D32_v2" not found')
    err = (<CUresult (*)(CUdeviceptr, size_t, unsigned int, size_t, size_t) nogil> __cuMemsetD2D32_v2)(dstDevice, dstPitch, ui, Width, Height)
    return err

cdef CUresult _cuMemsetD8Async(CUdeviceptr dstDevice, unsigned char uc, size_t N, CUstream hStream) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuMemsetD8Async
    cuPythonInit()
    if __cuMemsetD8Async == NULL:
        with gil:
            raise RuntimeError('Function "cuMemsetD8Async" not found')
    err = (<CUresult (*)(CUdeviceptr, unsigned char, size_t, CUstream) nogil> __cuMemsetD8Async)(dstDevice, uc, N, hStream)
    return err

cdef CUresult _cuMemsetD16Async(CUdeviceptr dstDevice, unsigned short us, size_t N, CUstream hStream) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuMemsetD16Async
    cuPythonInit()
    if __cuMemsetD16Async == NULL:
        with gil:
            raise RuntimeError('Function "cuMemsetD16Async" not found')
    err = (<CUresult (*)(CUdeviceptr, unsigned short, size_t, CUstream) nogil> __cuMemsetD16Async)(dstDevice, us, N, hStream)
    return err

cdef CUresult _cuMemsetD32Async(CUdeviceptr dstDevice, unsigned int ui, size_t N, CUstream hStream) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuMemsetD32Async
    cuPythonInit()
    if __cuMemsetD32Async == NULL:
        with gil:
            raise RuntimeError('Function "cuMemsetD32Async" not found')
    err = (<CUresult (*)(CUdeviceptr, unsigned int, size_t, CUstream) nogil> __cuMemsetD32Async)(dstDevice, ui, N, hStream)
    return err

cdef CUresult _cuMemsetD2D8Async(CUdeviceptr dstDevice, size_t dstPitch, unsigned char uc, size_t Width, size_t Height, CUstream hStream) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuMemsetD2D8Async
    cuPythonInit()
    if __cuMemsetD2D8Async == NULL:
        with gil:
            raise RuntimeError('Function "cuMemsetD2D8Async" not found')
    err = (<CUresult (*)(CUdeviceptr, size_t, unsigned char, size_t, size_t, CUstream) nogil> __cuMemsetD2D8Async)(dstDevice, dstPitch, uc, Width, Height, hStream)
    return err

cdef CUresult _cuMemsetD2D16Async(CUdeviceptr dstDevice, size_t dstPitch, unsigned short us, size_t Width, size_t Height, CUstream hStream) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuMemsetD2D16Async
    cuPythonInit()
    if __cuMemsetD2D16Async == NULL:
        with gil:
            raise RuntimeError('Function "cuMemsetD2D16Async" not found')
    err = (<CUresult (*)(CUdeviceptr, size_t, unsigned short, size_t, size_t, CUstream) nogil> __cuMemsetD2D16Async)(dstDevice, dstPitch, us, Width, Height, hStream)
    return err

cdef CUresult _cuMemsetD2D32Async(CUdeviceptr dstDevice, size_t dstPitch, unsigned int ui, size_t Width, size_t Height, CUstream hStream) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuMemsetD2D32Async
    cuPythonInit()
    if __cuMemsetD2D32Async == NULL:
        with gil:
            raise RuntimeError('Function "cuMemsetD2D32Async" not found')
    err = (<CUresult (*)(CUdeviceptr, size_t, unsigned int, size_t, size_t, CUstream) nogil> __cuMemsetD2D32Async)(dstDevice, dstPitch, ui, Width, Height, hStream)
    return err

cdef CUresult _cuArrayCreate_v2(CUarray* pHandle, const CUDA_ARRAY_DESCRIPTOR* pAllocateArray) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuArrayCreate_v2
    cuPythonInit()
    if __cuArrayCreate_v2 == NULL:
        with gil:
            raise RuntimeError('Function "cuArrayCreate_v2" not found')
    err = (<CUresult (*)(CUarray*, const CUDA_ARRAY_DESCRIPTOR*) nogil> __cuArrayCreate_v2)(pHandle, pAllocateArray)
    return err

cdef CUresult _cuArrayGetDescriptor_v2(CUDA_ARRAY_DESCRIPTOR* pArrayDescriptor, CUarray hArray) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuArrayGetDescriptor_v2
    cuPythonInit()
    if __cuArrayGetDescriptor_v2 == NULL:
        with gil:
            raise RuntimeError('Function "cuArrayGetDescriptor_v2" not found')
    err = (<CUresult (*)(CUDA_ARRAY_DESCRIPTOR*, CUarray) nogil> __cuArrayGetDescriptor_v2)(pArrayDescriptor, hArray)
    return err

cdef CUresult _cuArrayGetSparseProperties(CUDA_ARRAY_SPARSE_PROPERTIES* sparseProperties, CUarray array) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuArrayGetSparseProperties
    cuPythonInit()
    if __cuArrayGetSparseProperties == NULL:
        with gil:
            raise RuntimeError('Function "cuArrayGetSparseProperties" not found')
    err = (<CUresult (*)(CUDA_ARRAY_SPARSE_PROPERTIES*, CUarray) nogil> __cuArrayGetSparseProperties)(sparseProperties, array)
    return err

cdef CUresult _cuMipmappedArrayGetSparseProperties(CUDA_ARRAY_SPARSE_PROPERTIES* sparseProperties, CUmipmappedArray mipmap) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuMipmappedArrayGetSparseProperties
    cuPythonInit()
    if __cuMipmappedArrayGetSparseProperties == NULL:
        with gil:
            raise RuntimeError('Function "cuMipmappedArrayGetSparseProperties" not found')
    err = (<CUresult (*)(CUDA_ARRAY_SPARSE_PROPERTIES*, CUmipmappedArray) nogil> __cuMipmappedArrayGetSparseProperties)(sparseProperties, mipmap)
    return err

cdef CUresult _cuArrayGetMemoryRequirements(CUDA_ARRAY_MEMORY_REQUIREMENTS* memoryRequirements, CUarray array, CUdevice device) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuArrayGetMemoryRequirements
    cuPythonInit()
    if __cuArrayGetMemoryRequirements == NULL:
        with gil:
            raise RuntimeError('Function "cuArrayGetMemoryRequirements" not found')
    err = (<CUresult (*)(CUDA_ARRAY_MEMORY_REQUIREMENTS*, CUarray, CUdevice) nogil> __cuArrayGetMemoryRequirements)(memoryRequirements, array, device)
    return err

cdef CUresult _cuMipmappedArrayGetMemoryRequirements(CUDA_ARRAY_MEMORY_REQUIREMENTS* memoryRequirements, CUmipmappedArray mipmap, CUdevice device) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuMipmappedArrayGetMemoryRequirements
    cuPythonInit()
    if __cuMipmappedArrayGetMemoryRequirements == NULL:
        with gil:
            raise RuntimeError('Function "cuMipmappedArrayGetMemoryRequirements" not found')
    err = (<CUresult (*)(CUDA_ARRAY_MEMORY_REQUIREMENTS*, CUmipmappedArray, CUdevice) nogil> __cuMipmappedArrayGetMemoryRequirements)(memoryRequirements, mipmap, device)
    return err

cdef CUresult _cuArrayGetPlane(CUarray* pPlaneArray, CUarray hArray, unsigned int planeIdx) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuArrayGetPlane
    cuPythonInit()
    if __cuArrayGetPlane == NULL:
        with gil:
            raise RuntimeError('Function "cuArrayGetPlane" not found')
    err = (<CUresult (*)(CUarray*, CUarray, unsigned int) nogil> __cuArrayGetPlane)(pPlaneArray, hArray, planeIdx)
    return err

cdef CUresult _cuArrayDestroy(CUarray hArray) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuArrayDestroy
    cuPythonInit()
    if __cuArrayDestroy == NULL:
        with gil:
            raise RuntimeError('Function "cuArrayDestroy" not found')
    err = (<CUresult (*)(CUarray) nogil> __cuArrayDestroy)(hArray)
    return err

cdef CUresult _cuArray3DCreate_v2(CUarray* pHandle, const CUDA_ARRAY3D_DESCRIPTOR* pAllocateArray) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuArray3DCreate_v2
    cuPythonInit()
    if __cuArray3DCreate_v2 == NULL:
        with gil:
            raise RuntimeError('Function "cuArray3DCreate_v2" not found')
    err = (<CUresult (*)(CUarray*, const CUDA_ARRAY3D_DESCRIPTOR*) nogil> __cuArray3DCreate_v2)(pHandle, pAllocateArray)
    return err

cdef CUresult _cuArray3DGetDescriptor_v2(CUDA_ARRAY3D_DESCRIPTOR* pArrayDescriptor, CUarray hArray) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuArray3DGetDescriptor_v2
    cuPythonInit()
    if __cuArray3DGetDescriptor_v2 == NULL:
        with gil:
            raise RuntimeError('Function "cuArray3DGetDescriptor_v2" not found')
    err = (<CUresult (*)(CUDA_ARRAY3D_DESCRIPTOR*, CUarray) nogil> __cuArray3DGetDescriptor_v2)(pArrayDescriptor, hArray)
    return err

cdef CUresult _cuMipmappedArrayCreate(CUmipmappedArray* pHandle, const CUDA_ARRAY3D_DESCRIPTOR* pMipmappedArrayDesc, unsigned int numMipmapLevels) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuMipmappedArrayCreate
    cuPythonInit()
    if __cuMipmappedArrayCreate == NULL:
        with gil:
            raise RuntimeError('Function "cuMipmappedArrayCreate" not found')
    err = (<CUresult (*)(CUmipmappedArray*, const CUDA_ARRAY3D_DESCRIPTOR*, unsigned int) nogil> __cuMipmappedArrayCreate)(pHandle, pMipmappedArrayDesc, numMipmapLevels)
    return err

cdef CUresult _cuMipmappedArrayGetLevel(CUarray* pLevelArray, CUmipmappedArray hMipmappedArray, unsigned int level) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuMipmappedArrayGetLevel
    cuPythonInit()
    if __cuMipmappedArrayGetLevel == NULL:
        with gil:
            raise RuntimeError('Function "cuMipmappedArrayGetLevel" not found')
    err = (<CUresult (*)(CUarray*, CUmipmappedArray, unsigned int) nogil> __cuMipmappedArrayGetLevel)(pLevelArray, hMipmappedArray, level)
    return err

cdef CUresult _cuMipmappedArrayDestroy(CUmipmappedArray hMipmappedArray) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuMipmappedArrayDestroy
    cuPythonInit()
    if __cuMipmappedArrayDestroy == NULL:
        with gil:
            raise RuntimeError('Function "cuMipmappedArrayDestroy" not found')
    err = (<CUresult (*)(CUmipmappedArray) nogil> __cuMipmappedArrayDestroy)(hMipmappedArray)
    return err

cdef CUresult _cuMemAddressReserve(CUdeviceptr* ptr, size_t size, size_t alignment, CUdeviceptr addr, unsigned long long flags) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuMemAddressReserve
    cuPythonInit()
    if __cuMemAddressReserve == NULL:
        with gil:
            raise RuntimeError('Function "cuMemAddressReserve" not found')
    err = (<CUresult (*)(CUdeviceptr*, size_t, size_t, CUdeviceptr, unsigned long long) nogil> __cuMemAddressReserve)(ptr, size, alignment, addr, flags)
    return err

cdef CUresult _cuMemAddressFree(CUdeviceptr ptr, size_t size) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuMemAddressFree
    cuPythonInit()
    if __cuMemAddressFree == NULL:
        with gil:
            raise RuntimeError('Function "cuMemAddressFree" not found')
    err = (<CUresult (*)(CUdeviceptr, size_t) nogil> __cuMemAddressFree)(ptr, size)
    return err

cdef CUresult _cuMemCreate(CUmemGenericAllocationHandle* handle, size_t size, const CUmemAllocationProp* prop, unsigned long long flags) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuMemCreate
    cuPythonInit()
    if __cuMemCreate == NULL:
        with gil:
            raise RuntimeError('Function "cuMemCreate" not found')
    err = (<CUresult (*)(CUmemGenericAllocationHandle*, size_t, const CUmemAllocationProp*, unsigned long long) nogil> __cuMemCreate)(handle, size, prop, flags)
    return err

cdef CUresult _cuMemRelease(CUmemGenericAllocationHandle handle) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuMemRelease
    cuPythonInit()
    if __cuMemRelease == NULL:
        with gil:
            raise RuntimeError('Function "cuMemRelease" not found')
    err = (<CUresult (*)(CUmemGenericAllocationHandle) nogil> __cuMemRelease)(handle)
    return err

cdef CUresult _cuMemMap(CUdeviceptr ptr, size_t size, size_t offset, CUmemGenericAllocationHandle handle, unsigned long long flags) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuMemMap
    cuPythonInit()
    if __cuMemMap == NULL:
        with gil:
            raise RuntimeError('Function "cuMemMap" not found')
    err = (<CUresult (*)(CUdeviceptr, size_t, size_t, CUmemGenericAllocationHandle, unsigned long long) nogil> __cuMemMap)(ptr, size, offset, handle, flags)
    return err

cdef CUresult _cuMemMapArrayAsync(CUarrayMapInfo* mapInfoList, unsigned int count, CUstream hStream) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuMemMapArrayAsync
    cuPythonInit()
    if __cuMemMapArrayAsync == NULL:
        with gil:
            raise RuntimeError('Function "cuMemMapArrayAsync" not found')
    err = (<CUresult (*)(CUarrayMapInfo*, unsigned int, CUstream) nogil> __cuMemMapArrayAsync)(mapInfoList, count, hStream)
    return err

cdef CUresult _cuMemUnmap(CUdeviceptr ptr, size_t size) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuMemUnmap
    cuPythonInit()
    if __cuMemUnmap == NULL:
        with gil:
            raise RuntimeError('Function "cuMemUnmap" not found')
    err = (<CUresult (*)(CUdeviceptr, size_t) nogil> __cuMemUnmap)(ptr, size)
    return err

cdef CUresult _cuMemSetAccess(CUdeviceptr ptr, size_t size, const CUmemAccessDesc* desc, size_t count) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuMemSetAccess
    cuPythonInit()
    if __cuMemSetAccess == NULL:
        with gil:
            raise RuntimeError('Function "cuMemSetAccess" not found')
    err = (<CUresult (*)(CUdeviceptr, size_t, const CUmemAccessDesc*, size_t) nogil> __cuMemSetAccess)(ptr, size, desc, count)
    return err

cdef CUresult _cuMemGetAccess(unsigned long long* flags, const CUmemLocation* location, CUdeviceptr ptr) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuMemGetAccess
    cuPythonInit()
    if __cuMemGetAccess == NULL:
        with gil:
            raise RuntimeError('Function "cuMemGetAccess" not found')
    err = (<CUresult (*)(unsigned long long*, const CUmemLocation*, CUdeviceptr) nogil> __cuMemGetAccess)(flags, location, ptr)
    return err

cdef CUresult _cuMemExportToShareableHandle(void* shareableHandle, CUmemGenericAllocationHandle handle, CUmemAllocationHandleType handleType, unsigned long long flags) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuMemExportToShareableHandle
    cuPythonInit()
    if __cuMemExportToShareableHandle == NULL:
        with gil:
            raise RuntimeError('Function "cuMemExportToShareableHandle" not found')
    err = (<CUresult (*)(void*, CUmemGenericAllocationHandle, CUmemAllocationHandleType, unsigned long long) nogil> __cuMemExportToShareableHandle)(shareableHandle, handle, handleType, flags)
    return err

cdef CUresult _cuMemImportFromShareableHandle(CUmemGenericAllocationHandle* handle, void* osHandle, CUmemAllocationHandleType shHandleType) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuMemImportFromShareableHandle
    cuPythonInit()
    if __cuMemImportFromShareableHandle == NULL:
        with gil:
            raise RuntimeError('Function "cuMemImportFromShareableHandle" not found')
    err = (<CUresult (*)(CUmemGenericAllocationHandle*, void*, CUmemAllocationHandleType) nogil> __cuMemImportFromShareableHandle)(handle, osHandle, shHandleType)
    return err

cdef CUresult _cuMemGetAllocationGranularity(size_t* granularity, const CUmemAllocationProp* prop, CUmemAllocationGranularity_flags option) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuMemGetAllocationGranularity
    cuPythonInit()
    if __cuMemGetAllocationGranularity == NULL:
        with gil:
            raise RuntimeError('Function "cuMemGetAllocationGranularity" not found')
    err = (<CUresult (*)(size_t*, const CUmemAllocationProp*, CUmemAllocationGranularity_flags) nogil> __cuMemGetAllocationGranularity)(granularity, prop, option)
    return err

cdef CUresult _cuMemGetAllocationPropertiesFromHandle(CUmemAllocationProp* prop, CUmemGenericAllocationHandle handle) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuMemGetAllocationPropertiesFromHandle
    cuPythonInit()
    if __cuMemGetAllocationPropertiesFromHandle == NULL:
        with gil:
            raise RuntimeError('Function "cuMemGetAllocationPropertiesFromHandle" not found')
    err = (<CUresult (*)(CUmemAllocationProp*, CUmemGenericAllocationHandle) nogil> __cuMemGetAllocationPropertiesFromHandle)(prop, handle)
    return err

cdef CUresult _cuMemRetainAllocationHandle(CUmemGenericAllocationHandle* handle, void* addr) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuMemRetainAllocationHandle
    cuPythonInit()
    if __cuMemRetainAllocationHandle == NULL:
        with gil:
            raise RuntimeError('Function "cuMemRetainAllocationHandle" not found')
    err = (<CUresult (*)(CUmemGenericAllocationHandle*, void*) nogil> __cuMemRetainAllocationHandle)(handle, addr)
    return err

cdef CUresult _cuMemFreeAsync(CUdeviceptr dptr, CUstream hStream) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuMemFreeAsync
    cuPythonInit()
    if __cuMemFreeAsync == NULL:
        with gil:
            raise RuntimeError('Function "cuMemFreeAsync" not found')
    err = (<CUresult (*)(CUdeviceptr, CUstream) nogil> __cuMemFreeAsync)(dptr, hStream)
    return err

cdef CUresult _cuMemAllocAsync(CUdeviceptr* dptr, size_t bytesize, CUstream hStream) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuMemAllocAsync
    cuPythonInit()
    if __cuMemAllocAsync == NULL:
        with gil:
            raise RuntimeError('Function "cuMemAllocAsync" not found')
    err = (<CUresult (*)(CUdeviceptr*, size_t, CUstream) nogil> __cuMemAllocAsync)(dptr, bytesize, hStream)
    return err

cdef CUresult _cuMemPoolTrimTo(CUmemoryPool pool, size_t minBytesToKeep) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuMemPoolTrimTo
    cuPythonInit()
    if __cuMemPoolTrimTo == NULL:
        with gil:
            raise RuntimeError('Function "cuMemPoolTrimTo" not found')
    err = (<CUresult (*)(CUmemoryPool, size_t) nogil> __cuMemPoolTrimTo)(pool, minBytesToKeep)
    return err

cdef CUresult _cuMemPoolSetAttribute(CUmemoryPool pool, CUmemPool_attribute attr, void* value) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuMemPoolSetAttribute
    cuPythonInit()
    if __cuMemPoolSetAttribute == NULL:
        with gil:
            raise RuntimeError('Function "cuMemPoolSetAttribute" not found')
    err = (<CUresult (*)(CUmemoryPool, CUmemPool_attribute, void*) nogil> __cuMemPoolSetAttribute)(pool, attr, value)
    return err

cdef CUresult _cuMemPoolGetAttribute(CUmemoryPool pool, CUmemPool_attribute attr, void* value) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuMemPoolGetAttribute
    cuPythonInit()
    if __cuMemPoolGetAttribute == NULL:
        with gil:
            raise RuntimeError('Function "cuMemPoolGetAttribute" not found')
    err = (<CUresult (*)(CUmemoryPool, CUmemPool_attribute, void*) nogil> __cuMemPoolGetAttribute)(pool, attr, value)
    return err

cdef CUresult _cuMemPoolSetAccess(CUmemoryPool pool, const CUmemAccessDesc* map, size_t count) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuMemPoolSetAccess
    cuPythonInit()
    if __cuMemPoolSetAccess == NULL:
        with gil:
            raise RuntimeError('Function "cuMemPoolSetAccess" not found')
    err = (<CUresult (*)(CUmemoryPool, const CUmemAccessDesc*, size_t) nogil> __cuMemPoolSetAccess)(pool, map, count)
    return err

cdef CUresult _cuMemPoolGetAccess(CUmemAccess_flags* flags, CUmemoryPool memPool, CUmemLocation* location) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuMemPoolGetAccess
    cuPythonInit()
    if __cuMemPoolGetAccess == NULL:
        with gil:
            raise RuntimeError('Function "cuMemPoolGetAccess" not found')
    err = (<CUresult (*)(CUmemAccess_flags*, CUmemoryPool, CUmemLocation*) nogil> __cuMemPoolGetAccess)(flags, memPool, location)
    return err

cdef CUresult _cuMemPoolCreate(CUmemoryPool* pool, const CUmemPoolProps* poolProps) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuMemPoolCreate
    cuPythonInit()
    if __cuMemPoolCreate == NULL:
        with gil:
            raise RuntimeError('Function "cuMemPoolCreate" not found')
    err = (<CUresult (*)(CUmemoryPool*, const CUmemPoolProps*) nogil> __cuMemPoolCreate)(pool, poolProps)
    return err

cdef CUresult _cuMemPoolDestroy(CUmemoryPool pool) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuMemPoolDestroy
    cuPythonInit()
    if __cuMemPoolDestroy == NULL:
        with gil:
            raise RuntimeError('Function "cuMemPoolDestroy" not found')
    err = (<CUresult (*)(CUmemoryPool) nogil> __cuMemPoolDestroy)(pool)
    return err

cdef CUresult _cuMemAllocFromPoolAsync(CUdeviceptr* dptr, size_t bytesize, CUmemoryPool pool, CUstream hStream) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuMemAllocFromPoolAsync
    cuPythonInit()
    if __cuMemAllocFromPoolAsync == NULL:
        with gil:
            raise RuntimeError('Function "cuMemAllocFromPoolAsync" not found')
    err = (<CUresult (*)(CUdeviceptr*, size_t, CUmemoryPool, CUstream) nogil> __cuMemAllocFromPoolAsync)(dptr, bytesize, pool, hStream)
    return err

cdef CUresult _cuMemPoolExportToShareableHandle(void* handle_out, CUmemoryPool pool, CUmemAllocationHandleType handleType, unsigned long long flags) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuMemPoolExportToShareableHandle
    cuPythonInit()
    if __cuMemPoolExportToShareableHandle == NULL:
        with gil:
            raise RuntimeError('Function "cuMemPoolExportToShareableHandle" not found')
    err = (<CUresult (*)(void*, CUmemoryPool, CUmemAllocationHandleType, unsigned long long) nogil> __cuMemPoolExportToShareableHandle)(handle_out, pool, handleType, flags)
    return err

cdef CUresult _cuMemPoolImportFromShareableHandle(CUmemoryPool* pool_out, void* handle, CUmemAllocationHandleType handleType, unsigned long long flags) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuMemPoolImportFromShareableHandle
    cuPythonInit()
    if __cuMemPoolImportFromShareableHandle == NULL:
        with gil:
            raise RuntimeError('Function "cuMemPoolImportFromShareableHandle" not found')
    err = (<CUresult (*)(CUmemoryPool*, void*, CUmemAllocationHandleType, unsigned long long) nogil> __cuMemPoolImportFromShareableHandle)(pool_out, handle, handleType, flags)
    return err

cdef CUresult _cuMemPoolExportPointer(CUmemPoolPtrExportData* shareData_out, CUdeviceptr ptr) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuMemPoolExportPointer
    cuPythonInit()
    if __cuMemPoolExportPointer == NULL:
        with gil:
            raise RuntimeError('Function "cuMemPoolExportPointer" not found')
    err = (<CUresult (*)(CUmemPoolPtrExportData*, CUdeviceptr) nogil> __cuMemPoolExportPointer)(shareData_out, ptr)
    return err

cdef CUresult _cuMemPoolImportPointer(CUdeviceptr* ptr_out, CUmemoryPool pool, CUmemPoolPtrExportData* shareData) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuMemPoolImportPointer
    cuPythonInit()
    if __cuMemPoolImportPointer == NULL:
        with gil:
            raise RuntimeError('Function "cuMemPoolImportPointer" not found')
    err = (<CUresult (*)(CUdeviceptr*, CUmemoryPool, CUmemPoolPtrExportData*) nogil> __cuMemPoolImportPointer)(ptr_out, pool, shareData)
    return err

cdef CUresult _cuPointerGetAttribute(void* data, CUpointer_attribute attribute, CUdeviceptr ptr) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuPointerGetAttribute
    cuPythonInit()
    if __cuPointerGetAttribute == NULL:
        with gil:
            raise RuntimeError('Function "cuPointerGetAttribute" not found')
    err = (<CUresult (*)(void*, CUpointer_attribute, CUdeviceptr) nogil> __cuPointerGetAttribute)(data, attribute, ptr)
    return err

cdef CUresult _cuMemPrefetchAsync(CUdeviceptr devPtr, size_t count, CUdevice dstDevice, CUstream hStream) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuMemPrefetchAsync
    cuPythonInit()
    if __cuMemPrefetchAsync == NULL:
        with gil:
            raise RuntimeError('Function "cuMemPrefetchAsync" not found')
    err = (<CUresult (*)(CUdeviceptr, size_t, CUdevice, CUstream) nogil> __cuMemPrefetchAsync)(devPtr, count, dstDevice, hStream)
    return err

cdef CUresult _cuMemAdvise(CUdeviceptr devPtr, size_t count, CUmem_advise advice, CUdevice device) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuMemAdvise
    cuPythonInit()
    if __cuMemAdvise == NULL:
        with gil:
            raise RuntimeError('Function "cuMemAdvise" not found')
    err = (<CUresult (*)(CUdeviceptr, size_t, CUmem_advise, CUdevice) nogil> __cuMemAdvise)(devPtr, count, advice, device)
    return err

cdef CUresult _cuMemRangeGetAttribute(void* data, size_t dataSize, CUmem_range_attribute attribute, CUdeviceptr devPtr, size_t count) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuMemRangeGetAttribute
    cuPythonInit()
    if __cuMemRangeGetAttribute == NULL:
        with gil:
            raise RuntimeError('Function "cuMemRangeGetAttribute" not found')
    err = (<CUresult (*)(void*, size_t, CUmem_range_attribute, CUdeviceptr, size_t) nogil> __cuMemRangeGetAttribute)(data, dataSize, attribute, devPtr, count)
    return err

cdef CUresult _cuMemRangeGetAttributes(void** data, size_t* dataSizes, CUmem_range_attribute* attributes, size_t numAttributes, CUdeviceptr devPtr, size_t count) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuMemRangeGetAttributes
    cuPythonInit()
    if __cuMemRangeGetAttributes == NULL:
        with gil:
            raise RuntimeError('Function "cuMemRangeGetAttributes" not found')
    err = (<CUresult (*)(void**, size_t*, CUmem_range_attribute*, size_t, CUdeviceptr, size_t) nogil> __cuMemRangeGetAttributes)(data, dataSizes, attributes, numAttributes, devPtr, count)
    return err

cdef CUresult _cuPointerSetAttribute(const void* value, CUpointer_attribute attribute, CUdeviceptr ptr) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuPointerSetAttribute
    cuPythonInit()
    if __cuPointerSetAttribute == NULL:
        with gil:
            raise RuntimeError('Function "cuPointerSetAttribute" not found')
    err = (<CUresult (*)(const void*, CUpointer_attribute, CUdeviceptr) nogil> __cuPointerSetAttribute)(value, attribute, ptr)
    return err

cdef CUresult _cuPointerGetAttributes(unsigned int numAttributes, CUpointer_attribute* attributes, void** data, CUdeviceptr ptr) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuPointerGetAttributes
    cuPythonInit()
    if __cuPointerGetAttributes == NULL:
        with gil:
            raise RuntimeError('Function "cuPointerGetAttributes" not found')
    err = (<CUresult (*)(unsigned int, CUpointer_attribute*, void**, CUdeviceptr) nogil> __cuPointerGetAttributes)(numAttributes, attributes, data, ptr)
    return err

cdef CUresult _cuStreamCreate(CUstream* phStream, unsigned int Flags) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuStreamCreate
    cuPythonInit()
    if __cuStreamCreate == NULL:
        with gil:
            raise RuntimeError('Function "cuStreamCreate" not found')
    err = (<CUresult (*)(CUstream*, unsigned int) nogil> __cuStreamCreate)(phStream, Flags)
    return err

cdef CUresult _cuStreamCreateWithPriority(CUstream* phStream, unsigned int flags, int priority) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuStreamCreateWithPriority
    cuPythonInit()
    if __cuStreamCreateWithPriority == NULL:
        with gil:
            raise RuntimeError('Function "cuStreamCreateWithPriority" not found')
    err = (<CUresult (*)(CUstream*, unsigned int, int) nogil> __cuStreamCreateWithPriority)(phStream, flags, priority)
    return err

cdef CUresult _cuStreamGetPriority(CUstream hStream, int* priority) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuStreamGetPriority
    cuPythonInit()
    if __cuStreamGetPriority == NULL:
        with gil:
            raise RuntimeError('Function "cuStreamGetPriority" not found')
    err = (<CUresult (*)(CUstream, int*) nogil> __cuStreamGetPriority)(hStream, priority)
    return err

cdef CUresult _cuStreamGetFlags(CUstream hStream, unsigned int* flags) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuStreamGetFlags
    cuPythonInit()
    if __cuStreamGetFlags == NULL:
        with gil:
            raise RuntimeError('Function "cuStreamGetFlags" not found')
    err = (<CUresult (*)(CUstream, unsigned int*) nogil> __cuStreamGetFlags)(hStream, flags)
    return err

cdef CUresult _cuStreamGetCtx(CUstream hStream, CUcontext* pctx) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuStreamGetCtx
    cuPythonInit()
    if __cuStreamGetCtx == NULL:
        with gil:
            raise RuntimeError('Function "cuStreamGetCtx" not found')
    err = (<CUresult (*)(CUstream, CUcontext*) nogil> __cuStreamGetCtx)(hStream, pctx)
    return err

cdef CUresult _cuStreamWaitEvent(CUstream hStream, CUevent hEvent, unsigned int Flags) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuStreamWaitEvent
    cuPythonInit()
    if __cuStreamWaitEvent == NULL:
        with gil:
            raise RuntimeError('Function "cuStreamWaitEvent" not found')
    err = (<CUresult (*)(CUstream, CUevent, unsigned int) nogil> __cuStreamWaitEvent)(hStream, hEvent, Flags)
    return err

cdef CUresult _cuStreamAddCallback(CUstream hStream, CUstreamCallback callback, void* userData, unsigned int flags) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuStreamAddCallback
    cuPythonInit()
    if __cuStreamAddCallback == NULL:
        with gil:
            raise RuntimeError('Function "cuStreamAddCallback" not found')
    err = (<CUresult (*)(CUstream, CUstreamCallback, void*, unsigned int) nogil> __cuStreamAddCallback)(hStream, callback, userData, flags)
    return err

cdef CUresult _cuStreamBeginCapture_v2(CUstream hStream, CUstreamCaptureMode mode) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuStreamBeginCapture_v2
    cuPythonInit()
    if __cuStreamBeginCapture_v2 == NULL:
        with gil:
            raise RuntimeError('Function "cuStreamBeginCapture_v2" not found')
    err = (<CUresult (*)(CUstream, CUstreamCaptureMode) nogil> __cuStreamBeginCapture_v2)(hStream, mode)
    return err

cdef CUresult _cuThreadExchangeStreamCaptureMode(CUstreamCaptureMode* mode) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuThreadExchangeStreamCaptureMode
    cuPythonInit()
    if __cuThreadExchangeStreamCaptureMode == NULL:
        with gil:
            raise RuntimeError('Function "cuThreadExchangeStreamCaptureMode" not found')
    err = (<CUresult (*)(CUstreamCaptureMode*) nogil> __cuThreadExchangeStreamCaptureMode)(mode)
    return err

cdef CUresult _cuStreamEndCapture(CUstream hStream, CUgraph* phGraph) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuStreamEndCapture
    cuPythonInit()
    if __cuStreamEndCapture == NULL:
        with gil:
            raise RuntimeError('Function "cuStreamEndCapture" not found')
    err = (<CUresult (*)(CUstream, CUgraph*) nogil> __cuStreamEndCapture)(hStream, phGraph)
    return err

cdef CUresult _cuStreamIsCapturing(CUstream hStream, CUstreamCaptureStatus* captureStatus) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuStreamIsCapturing
    cuPythonInit()
    if __cuStreamIsCapturing == NULL:
        with gil:
            raise RuntimeError('Function "cuStreamIsCapturing" not found')
    err = (<CUresult (*)(CUstream, CUstreamCaptureStatus*) nogil> __cuStreamIsCapturing)(hStream, captureStatus)
    return err

cdef CUresult _cuStreamGetCaptureInfo(CUstream hStream, CUstreamCaptureStatus* captureStatus_out, cuuint64_t* id_out) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuStreamGetCaptureInfo
    cuPythonInit()
    if __cuStreamGetCaptureInfo == NULL:
        with gil:
            raise RuntimeError('Function "cuStreamGetCaptureInfo" not found')
    err = (<CUresult (*)(CUstream, CUstreamCaptureStatus*, cuuint64_t*) nogil> __cuStreamGetCaptureInfo)(hStream, captureStatus_out, id_out)
    return err

cdef CUresult _cuStreamGetCaptureInfo_v2(CUstream hStream, CUstreamCaptureStatus* captureStatus_out, cuuint64_t* id_out, CUgraph* graph_out, const CUgraphNode** dependencies_out, size_t* numDependencies_out) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuStreamGetCaptureInfo_v2
    cuPythonInit()
    if __cuStreamGetCaptureInfo_v2 == NULL:
        with gil:
            raise RuntimeError('Function "cuStreamGetCaptureInfo_v2" not found')
    err = (<CUresult (*)(CUstream, CUstreamCaptureStatus*, cuuint64_t*, CUgraph*, const CUgraphNode**, size_t*) nogil> __cuStreamGetCaptureInfo_v2)(hStream, captureStatus_out, id_out, graph_out, dependencies_out, numDependencies_out)
    return err

cdef CUresult _cuStreamUpdateCaptureDependencies(CUstream hStream, CUgraphNode* dependencies, size_t numDependencies, unsigned int flags) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuStreamUpdateCaptureDependencies
    cuPythonInit()
    if __cuStreamUpdateCaptureDependencies == NULL:
        with gil:
            raise RuntimeError('Function "cuStreamUpdateCaptureDependencies" not found')
    err = (<CUresult (*)(CUstream, CUgraphNode*, size_t, unsigned int) nogil> __cuStreamUpdateCaptureDependencies)(hStream, dependencies, numDependencies, flags)
    return err

cdef CUresult _cuStreamAttachMemAsync(CUstream hStream, CUdeviceptr dptr, size_t length, unsigned int flags) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuStreamAttachMemAsync
    cuPythonInit()
    if __cuStreamAttachMemAsync == NULL:
        with gil:
            raise RuntimeError('Function "cuStreamAttachMemAsync" not found')
    err = (<CUresult (*)(CUstream, CUdeviceptr, size_t, unsigned int) nogil> __cuStreamAttachMemAsync)(hStream, dptr, length, flags)
    return err

cdef CUresult _cuStreamQuery(CUstream hStream) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuStreamQuery
    cuPythonInit()
    if __cuStreamQuery == NULL:
        with gil:
            raise RuntimeError('Function "cuStreamQuery" not found')
    err = (<CUresult (*)(CUstream) nogil> __cuStreamQuery)(hStream)
    return err

cdef CUresult _cuStreamSynchronize(CUstream hStream) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuStreamSynchronize
    cuPythonInit()
    if __cuStreamSynchronize == NULL:
        with gil:
            raise RuntimeError('Function "cuStreamSynchronize" not found')
    err = (<CUresult (*)(CUstream) nogil> __cuStreamSynchronize)(hStream)
    return err

cdef CUresult _cuStreamDestroy_v2(CUstream hStream) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuStreamDestroy_v2
    cuPythonInit()
    if __cuStreamDestroy_v2 == NULL:
        with gil:
            raise RuntimeError('Function "cuStreamDestroy_v2" not found')
    err = (<CUresult (*)(CUstream) nogil> __cuStreamDestroy_v2)(hStream)
    return err

cdef CUresult _cuStreamCopyAttributes(CUstream dst, CUstream src) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuStreamCopyAttributes
    cuPythonInit()
    if __cuStreamCopyAttributes == NULL:
        with gil:
            raise RuntimeError('Function "cuStreamCopyAttributes" not found')
    err = (<CUresult (*)(CUstream, CUstream) nogil> __cuStreamCopyAttributes)(dst, src)
    return err

cdef CUresult _cuStreamGetAttribute(CUstream hStream, CUstreamAttrID attr, CUstreamAttrValue* value_out) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuStreamGetAttribute
    cuPythonInit()
    if __cuStreamGetAttribute == NULL:
        with gil:
            raise RuntimeError('Function "cuStreamGetAttribute" not found')
    err = (<CUresult (*)(CUstream, CUstreamAttrID, CUstreamAttrValue*) nogil> __cuStreamGetAttribute)(hStream, attr, value_out)
    return err

cdef CUresult _cuStreamSetAttribute(CUstream hStream, CUstreamAttrID attr, const CUstreamAttrValue* value) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuStreamSetAttribute
    cuPythonInit()
    if __cuStreamSetAttribute == NULL:
        with gil:
            raise RuntimeError('Function "cuStreamSetAttribute" not found')
    err = (<CUresult (*)(CUstream, CUstreamAttrID, const CUstreamAttrValue*) nogil> __cuStreamSetAttribute)(hStream, attr, value)
    return err

cdef CUresult _cuEventCreate(CUevent* phEvent, unsigned int Flags) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuEventCreate
    cuPythonInit()
    if __cuEventCreate == NULL:
        with gil:
            raise RuntimeError('Function "cuEventCreate" not found')
    err = (<CUresult (*)(CUevent*, unsigned int) nogil> __cuEventCreate)(phEvent, Flags)
    return err

cdef CUresult _cuEventRecord(CUevent hEvent, CUstream hStream) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuEventRecord
    cuPythonInit()
    if __cuEventRecord == NULL:
        with gil:
            raise RuntimeError('Function "cuEventRecord" not found')
    err = (<CUresult (*)(CUevent, CUstream) nogil> __cuEventRecord)(hEvent, hStream)
    return err

cdef CUresult _cuEventRecordWithFlags(CUevent hEvent, CUstream hStream, unsigned int flags) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuEventRecordWithFlags
    cuPythonInit()
    if __cuEventRecordWithFlags == NULL:
        with gil:
            raise RuntimeError('Function "cuEventRecordWithFlags" not found')
    err = (<CUresult (*)(CUevent, CUstream, unsigned int) nogil> __cuEventRecordWithFlags)(hEvent, hStream, flags)
    return err

cdef CUresult _cuEventQuery(CUevent hEvent) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuEventQuery
    cuPythonInit()
    if __cuEventQuery == NULL:
        with gil:
            raise RuntimeError('Function "cuEventQuery" not found')
    err = (<CUresult (*)(CUevent) nogil> __cuEventQuery)(hEvent)
    return err

cdef CUresult _cuEventSynchronize(CUevent hEvent) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuEventSynchronize
    cuPythonInit()
    if __cuEventSynchronize == NULL:
        with gil:
            raise RuntimeError('Function "cuEventSynchronize" not found')
    err = (<CUresult (*)(CUevent) nogil> __cuEventSynchronize)(hEvent)
    return err

cdef CUresult _cuEventDestroy_v2(CUevent hEvent) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuEventDestroy_v2
    cuPythonInit()
    if __cuEventDestroy_v2 == NULL:
        with gil:
            raise RuntimeError('Function "cuEventDestroy_v2" not found')
    err = (<CUresult (*)(CUevent) nogil> __cuEventDestroy_v2)(hEvent)
    return err

cdef CUresult _cuEventElapsedTime(float* pMilliseconds, CUevent hStart, CUevent hEnd) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuEventElapsedTime
    cuPythonInit()
    if __cuEventElapsedTime == NULL:
        with gil:
            raise RuntimeError('Function "cuEventElapsedTime" not found')
    err = (<CUresult (*)(float*, CUevent, CUevent) nogil> __cuEventElapsedTime)(pMilliseconds, hStart, hEnd)
    return err

cdef CUresult _cuImportExternalMemory(CUexternalMemory* extMem_out, const CUDA_EXTERNAL_MEMORY_HANDLE_DESC* memHandleDesc) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuImportExternalMemory
    cuPythonInit()
    if __cuImportExternalMemory == NULL:
        with gil:
            raise RuntimeError('Function "cuImportExternalMemory" not found')
    err = (<CUresult (*)(CUexternalMemory*, const CUDA_EXTERNAL_MEMORY_HANDLE_DESC*) nogil> __cuImportExternalMemory)(extMem_out, memHandleDesc)
    return err

cdef CUresult _cuExternalMemoryGetMappedBuffer(CUdeviceptr* devPtr, CUexternalMemory extMem, const CUDA_EXTERNAL_MEMORY_BUFFER_DESC* bufferDesc) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuExternalMemoryGetMappedBuffer
    cuPythonInit()
    if __cuExternalMemoryGetMappedBuffer == NULL:
        with gil:
            raise RuntimeError('Function "cuExternalMemoryGetMappedBuffer" not found')
    err = (<CUresult (*)(CUdeviceptr*, CUexternalMemory, const CUDA_EXTERNAL_MEMORY_BUFFER_DESC*) nogil> __cuExternalMemoryGetMappedBuffer)(devPtr, extMem, bufferDesc)
    return err

cdef CUresult _cuExternalMemoryGetMappedMipmappedArray(CUmipmappedArray* mipmap, CUexternalMemory extMem, const CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC* mipmapDesc) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuExternalMemoryGetMappedMipmappedArray
    cuPythonInit()
    if __cuExternalMemoryGetMappedMipmappedArray == NULL:
        with gil:
            raise RuntimeError('Function "cuExternalMemoryGetMappedMipmappedArray" not found')
    err = (<CUresult (*)(CUmipmappedArray*, CUexternalMemory, const CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC*) nogil> __cuExternalMemoryGetMappedMipmappedArray)(mipmap, extMem, mipmapDesc)
    return err

cdef CUresult _cuDestroyExternalMemory(CUexternalMemory extMem) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuDestroyExternalMemory
    cuPythonInit()
    if __cuDestroyExternalMemory == NULL:
        with gil:
            raise RuntimeError('Function "cuDestroyExternalMemory" not found')
    err = (<CUresult (*)(CUexternalMemory) nogil> __cuDestroyExternalMemory)(extMem)
    return err

cdef CUresult _cuImportExternalSemaphore(CUexternalSemaphore* extSem_out, const CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC* semHandleDesc) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuImportExternalSemaphore
    cuPythonInit()
    if __cuImportExternalSemaphore == NULL:
        with gil:
            raise RuntimeError('Function "cuImportExternalSemaphore" not found')
    err = (<CUresult (*)(CUexternalSemaphore*, const CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC*) nogil> __cuImportExternalSemaphore)(extSem_out, semHandleDesc)
    return err

cdef CUresult _cuSignalExternalSemaphoresAsync(const CUexternalSemaphore* extSemArray, const CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS* paramsArray, unsigned int numExtSems, CUstream stream) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuSignalExternalSemaphoresAsync
    cuPythonInit()
    if __cuSignalExternalSemaphoresAsync == NULL:
        with gil:
            raise RuntimeError('Function "cuSignalExternalSemaphoresAsync" not found')
    err = (<CUresult (*)(const CUexternalSemaphore*, const CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS*, unsigned int, CUstream) nogil> __cuSignalExternalSemaphoresAsync)(extSemArray, paramsArray, numExtSems, stream)
    return err

cdef CUresult _cuWaitExternalSemaphoresAsync(const CUexternalSemaphore* extSemArray, const CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS* paramsArray, unsigned int numExtSems, CUstream stream) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuWaitExternalSemaphoresAsync
    cuPythonInit()
    if __cuWaitExternalSemaphoresAsync == NULL:
        with gil:
            raise RuntimeError('Function "cuWaitExternalSemaphoresAsync" not found')
    err = (<CUresult (*)(const CUexternalSemaphore*, const CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS*, unsigned int, CUstream) nogil> __cuWaitExternalSemaphoresAsync)(extSemArray, paramsArray, numExtSems, stream)
    return err

cdef CUresult _cuDestroyExternalSemaphore(CUexternalSemaphore extSem) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuDestroyExternalSemaphore
    cuPythonInit()
    if __cuDestroyExternalSemaphore == NULL:
        with gil:
            raise RuntimeError('Function "cuDestroyExternalSemaphore" not found')
    err = (<CUresult (*)(CUexternalSemaphore) nogil> __cuDestroyExternalSemaphore)(extSem)
    return err

cdef CUresult _cuStreamWaitValue32(CUstream stream, CUdeviceptr addr, cuuint32_t value, unsigned int flags) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuStreamWaitValue32
    cuPythonInit()
    if __cuStreamWaitValue32 == NULL:
        with gil:
            raise RuntimeError('Function "cuStreamWaitValue32" not found')
    err = (<CUresult (*)(CUstream, CUdeviceptr, cuuint32_t, unsigned int) nogil> __cuStreamWaitValue32)(stream, addr, value, flags)
    return err

cdef CUresult _cuStreamWaitValue64(CUstream stream, CUdeviceptr addr, cuuint64_t value, unsigned int flags) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuStreamWaitValue64
    cuPythonInit()
    if __cuStreamWaitValue64 == NULL:
        with gil:
            raise RuntimeError('Function "cuStreamWaitValue64" not found')
    err = (<CUresult (*)(CUstream, CUdeviceptr, cuuint64_t, unsigned int) nogil> __cuStreamWaitValue64)(stream, addr, value, flags)
    return err

cdef CUresult _cuStreamWriteValue32(CUstream stream, CUdeviceptr addr, cuuint32_t value, unsigned int flags) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuStreamWriteValue32
    cuPythonInit()
    if __cuStreamWriteValue32 == NULL:
        with gil:
            raise RuntimeError('Function "cuStreamWriteValue32" not found')
    err = (<CUresult (*)(CUstream, CUdeviceptr, cuuint32_t, unsigned int) nogil> __cuStreamWriteValue32)(stream, addr, value, flags)
    return err

cdef CUresult _cuStreamWriteValue64(CUstream stream, CUdeviceptr addr, cuuint64_t value, unsigned int flags) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuStreamWriteValue64
    cuPythonInit()
    if __cuStreamWriteValue64 == NULL:
        with gil:
            raise RuntimeError('Function "cuStreamWriteValue64" not found')
    err = (<CUresult (*)(CUstream, CUdeviceptr, cuuint64_t, unsigned int) nogil> __cuStreamWriteValue64)(stream, addr, value, flags)
    return err

cdef CUresult _cuStreamBatchMemOp(CUstream stream, unsigned int count, CUstreamBatchMemOpParams* paramArray, unsigned int flags) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuStreamBatchMemOp
    cuPythonInit()
    if __cuStreamBatchMemOp == NULL:
        with gil:
            raise RuntimeError('Function "cuStreamBatchMemOp" not found')
    err = (<CUresult (*)(CUstream, unsigned int, CUstreamBatchMemOpParams*, unsigned int) nogil> __cuStreamBatchMemOp)(stream, count, paramArray, flags)
    return err

cdef CUresult _cuStreamWaitValue32_v2(CUstream stream, CUdeviceptr addr, cuuint32_t value, unsigned int flags) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuStreamWaitValue32_v2
    cuPythonInit()
    if __cuStreamWaitValue32_v2 == NULL:
        with gil:
            raise RuntimeError('Function "cuStreamWaitValue32_v2" not found')
    err = (<CUresult (*)(CUstream, CUdeviceptr, cuuint32_t, unsigned int) nogil> __cuStreamWaitValue32_v2)(stream, addr, value, flags)
    return err

cdef CUresult _cuStreamWaitValue64_v2(CUstream stream, CUdeviceptr addr, cuuint64_t value, unsigned int flags) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuStreamWaitValue64_v2
    cuPythonInit()
    if __cuStreamWaitValue64_v2 == NULL:
        with gil:
            raise RuntimeError('Function "cuStreamWaitValue64_v2" not found')
    err = (<CUresult (*)(CUstream, CUdeviceptr, cuuint64_t, unsigned int) nogil> __cuStreamWaitValue64_v2)(stream, addr, value, flags)
    return err

cdef CUresult _cuStreamWriteValue32_v2(CUstream stream, CUdeviceptr addr, cuuint32_t value, unsigned int flags) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuStreamWriteValue32_v2
    cuPythonInit()
    if __cuStreamWriteValue32_v2 == NULL:
        with gil:
            raise RuntimeError('Function "cuStreamWriteValue32_v2" not found')
    err = (<CUresult (*)(CUstream, CUdeviceptr, cuuint32_t, unsigned int) nogil> __cuStreamWriteValue32_v2)(stream, addr, value, flags)
    return err

cdef CUresult _cuStreamWriteValue64_v2(CUstream stream, CUdeviceptr addr, cuuint64_t value, unsigned int flags) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuStreamWriteValue64_v2
    cuPythonInit()
    if __cuStreamWriteValue64_v2 == NULL:
        with gil:
            raise RuntimeError('Function "cuStreamWriteValue64_v2" not found')
    err = (<CUresult (*)(CUstream, CUdeviceptr, cuuint64_t, unsigned int) nogil> __cuStreamWriteValue64_v2)(stream, addr, value, flags)
    return err

cdef CUresult _cuStreamBatchMemOp_v2(CUstream stream, unsigned int count, CUstreamBatchMemOpParams* paramArray, unsigned int flags) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuStreamBatchMemOp_v2
    cuPythonInit()
    if __cuStreamBatchMemOp_v2 == NULL:
        with gil:
            raise RuntimeError('Function "cuStreamBatchMemOp_v2" not found')
    err = (<CUresult (*)(CUstream, unsigned int, CUstreamBatchMemOpParams*, unsigned int) nogil> __cuStreamBatchMemOp_v2)(stream, count, paramArray, flags)
    return err

cdef CUresult _cuFuncGetAttribute(int* pi, CUfunction_attribute attrib, CUfunction hfunc) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuFuncGetAttribute
    cuPythonInit()
    if __cuFuncGetAttribute == NULL:
        with gil:
            raise RuntimeError('Function "cuFuncGetAttribute" not found')
    err = (<CUresult (*)(int*, CUfunction_attribute, CUfunction) nogil> __cuFuncGetAttribute)(pi, attrib, hfunc)
    return err

cdef CUresult _cuFuncSetAttribute(CUfunction hfunc, CUfunction_attribute attrib, int value) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuFuncSetAttribute
    cuPythonInit()
    if __cuFuncSetAttribute == NULL:
        with gil:
            raise RuntimeError('Function "cuFuncSetAttribute" not found')
    err = (<CUresult (*)(CUfunction, CUfunction_attribute, int) nogil> __cuFuncSetAttribute)(hfunc, attrib, value)
    return err

cdef CUresult _cuFuncSetCacheConfig(CUfunction hfunc, CUfunc_cache config) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuFuncSetCacheConfig
    cuPythonInit()
    if __cuFuncSetCacheConfig == NULL:
        with gil:
            raise RuntimeError('Function "cuFuncSetCacheConfig" not found')
    err = (<CUresult (*)(CUfunction, CUfunc_cache) nogil> __cuFuncSetCacheConfig)(hfunc, config)
    return err

cdef CUresult _cuFuncSetSharedMemConfig(CUfunction hfunc, CUsharedconfig config) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuFuncSetSharedMemConfig
    cuPythonInit()
    if __cuFuncSetSharedMemConfig == NULL:
        with gil:
            raise RuntimeError('Function "cuFuncSetSharedMemConfig" not found')
    err = (<CUresult (*)(CUfunction, CUsharedconfig) nogil> __cuFuncSetSharedMemConfig)(hfunc, config)
    return err

cdef CUresult _cuFuncGetModule(CUmodule* hmod, CUfunction hfunc) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuFuncGetModule
    cuPythonInit()
    if __cuFuncGetModule == NULL:
        with gil:
            raise RuntimeError('Function "cuFuncGetModule" not found')
    err = (<CUresult (*)(CUmodule*, CUfunction) nogil> __cuFuncGetModule)(hmod, hfunc)
    return err

cdef CUresult _cuLaunchKernel(CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, unsigned int sharedMemBytes, CUstream hStream, void** kernelParams, void** extra) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuLaunchKernel
    cuPythonInit()
    if __cuLaunchKernel == NULL:
        with gil:
            raise RuntimeError('Function "cuLaunchKernel" not found')
    err = (<CUresult (*)(CUfunction, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, CUstream, void**, void**) nogil> __cuLaunchKernel)(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, kernelParams, extra)
    return err

cdef CUresult _cuLaunchCooperativeKernel(CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, unsigned int sharedMemBytes, CUstream hStream, void** kernelParams) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuLaunchCooperativeKernel
    cuPythonInit()
    if __cuLaunchCooperativeKernel == NULL:
        with gil:
            raise RuntimeError('Function "cuLaunchCooperativeKernel" not found')
    err = (<CUresult (*)(CUfunction, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, CUstream, void**) nogil> __cuLaunchCooperativeKernel)(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, kernelParams)
    return err

cdef CUresult _cuLaunchCooperativeKernelMultiDevice(CUDA_LAUNCH_PARAMS* launchParamsList, unsigned int numDevices, unsigned int flags) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuLaunchCooperativeKernelMultiDevice
    cuPythonInit()
    if __cuLaunchCooperativeKernelMultiDevice == NULL:
        with gil:
            raise RuntimeError('Function "cuLaunchCooperativeKernelMultiDevice" not found')
    err = (<CUresult (*)(CUDA_LAUNCH_PARAMS*, unsigned int, unsigned int) nogil> __cuLaunchCooperativeKernelMultiDevice)(launchParamsList, numDevices, flags)
    return err

cdef CUresult _cuLaunchHostFunc(CUstream hStream, CUhostFn fn, void* userData) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuLaunchHostFunc
    cuPythonInit()
    if __cuLaunchHostFunc == NULL:
        with gil:
            raise RuntimeError('Function "cuLaunchHostFunc" not found')
    err = (<CUresult (*)(CUstream, CUhostFn, void*) nogil> __cuLaunchHostFunc)(hStream, fn, userData)
    return err

cdef CUresult _cuFuncSetBlockShape(CUfunction hfunc, int x, int y, int z) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuFuncSetBlockShape
    cuPythonInit()
    if __cuFuncSetBlockShape == NULL:
        with gil:
            raise RuntimeError('Function "cuFuncSetBlockShape" not found')
    err = (<CUresult (*)(CUfunction, int, int, int) nogil> __cuFuncSetBlockShape)(hfunc, x, y, z)
    return err

cdef CUresult _cuFuncSetSharedSize(CUfunction hfunc, unsigned int numbytes) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuFuncSetSharedSize
    cuPythonInit()
    if __cuFuncSetSharedSize == NULL:
        with gil:
            raise RuntimeError('Function "cuFuncSetSharedSize" not found')
    err = (<CUresult (*)(CUfunction, unsigned int) nogil> __cuFuncSetSharedSize)(hfunc, numbytes)
    return err

cdef CUresult _cuParamSetSize(CUfunction hfunc, unsigned int numbytes) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuParamSetSize
    cuPythonInit()
    if __cuParamSetSize == NULL:
        with gil:
            raise RuntimeError('Function "cuParamSetSize" not found')
    err = (<CUresult (*)(CUfunction, unsigned int) nogil> __cuParamSetSize)(hfunc, numbytes)
    return err

cdef CUresult _cuParamSeti(CUfunction hfunc, int offset, unsigned int value) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuParamSeti
    cuPythonInit()
    if __cuParamSeti == NULL:
        with gil:
            raise RuntimeError('Function "cuParamSeti" not found')
    err = (<CUresult (*)(CUfunction, int, unsigned int) nogil> __cuParamSeti)(hfunc, offset, value)
    return err

cdef CUresult _cuParamSetf(CUfunction hfunc, int offset, float value) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuParamSetf
    cuPythonInit()
    if __cuParamSetf == NULL:
        with gil:
            raise RuntimeError('Function "cuParamSetf" not found')
    err = (<CUresult (*)(CUfunction, int, float) nogil> __cuParamSetf)(hfunc, offset, value)
    return err

cdef CUresult _cuParamSetv(CUfunction hfunc, int offset, void* ptr, unsigned int numbytes) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuParamSetv
    cuPythonInit()
    if __cuParamSetv == NULL:
        with gil:
            raise RuntimeError('Function "cuParamSetv" not found')
    err = (<CUresult (*)(CUfunction, int, void*, unsigned int) nogil> __cuParamSetv)(hfunc, offset, ptr, numbytes)
    return err

cdef CUresult _cuLaunch(CUfunction f) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuLaunch
    cuPythonInit()
    if __cuLaunch == NULL:
        with gil:
            raise RuntimeError('Function "cuLaunch" not found')
    err = (<CUresult (*)(CUfunction) nogil> __cuLaunch)(f)
    return err

cdef CUresult _cuLaunchGrid(CUfunction f, int grid_width, int grid_height) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuLaunchGrid
    cuPythonInit()
    if __cuLaunchGrid == NULL:
        with gil:
            raise RuntimeError('Function "cuLaunchGrid" not found')
    err = (<CUresult (*)(CUfunction, int, int) nogil> __cuLaunchGrid)(f, grid_width, grid_height)
    return err

cdef CUresult _cuLaunchGridAsync(CUfunction f, int grid_width, int grid_height, CUstream hStream) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuLaunchGridAsync
    cuPythonInit()
    if __cuLaunchGridAsync == NULL:
        with gil:
            raise RuntimeError('Function "cuLaunchGridAsync" not found')
    err = (<CUresult (*)(CUfunction, int, int, CUstream) nogil> __cuLaunchGridAsync)(f, grid_width, grid_height, hStream)
    return err

cdef CUresult _cuParamSetTexRef(CUfunction hfunc, int texunit, CUtexref hTexRef) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuParamSetTexRef
    cuPythonInit()
    if __cuParamSetTexRef == NULL:
        with gil:
            raise RuntimeError('Function "cuParamSetTexRef" not found')
    err = (<CUresult (*)(CUfunction, int, CUtexref) nogil> __cuParamSetTexRef)(hfunc, texunit, hTexRef)
    return err

cdef CUresult _cuGraphCreate(CUgraph* phGraph, unsigned int flags) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuGraphCreate
    cuPythonInit()
    if __cuGraphCreate == NULL:
        with gil:
            raise RuntimeError('Function "cuGraphCreate" not found')
    err = (<CUresult (*)(CUgraph*, unsigned int) nogil> __cuGraphCreate)(phGraph, flags)
    return err

cdef CUresult _cuGraphAddKernelNode(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, size_t numDependencies, const CUDA_KERNEL_NODE_PARAMS* nodeParams) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuGraphAddKernelNode
    cuPythonInit()
    if __cuGraphAddKernelNode == NULL:
        with gil:
            raise RuntimeError('Function "cuGraphAddKernelNode" not found')
    err = (<CUresult (*)(CUgraphNode*, CUgraph, const CUgraphNode*, size_t, const CUDA_KERNEL_NODE_PARAMS*) nogil> __cuGraphAddKernelNode)(phGraphNode, hGraph, dependencies, numDependencies, nodeParams)
    return err

cdef CUresult _cuGraphKernelNodeGetParams(CUgraphNode hNode, CUDA_KERNEL_NODE_PARAMS* nodeParams) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuGraphKernelNodeGetParams
    cuPythonInit()
    if __cuGraphKernelNodeGetParams == NULL:
        with gil:
            raise RuntimeError('Function "cuGraphKernelNodeGetParams" not found')
    err = (<CUresult (*)(CUgraphNode, CUDA_KERNEL_NODE_PARAMS*) nogil> __cuGraphKernelNodeGetParams)(hNode, nodeParams)
    return err

cdef CUresult _cuGraphKernelNodeSetParams(CUgraphNode hNode, const CUDA_KERNEL_NODE_PARAMS* nodeParams) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuGraphKernelNodeSetParams
    cuPythonInit()
    if __cuGraphKernelNodeSetParams == NULL:
        with gil:
            raise RuntimeError('Function "cuGraphKernelNodeSetParams" not found')
    err = (<CUresult (*)(CUgraphNode, const CUDA_KERNEL_NODE_PARAMS*) nogil> __cuGraphKernelNodeSetParams)(hNode, nodeParams)
    return err

cdef CUresult _cuGraphAddMemcpyNode(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, size_t numDependencies, const CUDA_MEMCPY3D* copyParams, CUcontext ctx) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuGraphAddMemcpyNode
    cuPythonInit()
    if __cuGraphAddMemcpyNode == NULL:
        with gil:
            raise RuntimeError('Function "cuGraphAddMemcpyNode" not found')
    err = (<CUresult (*)(CUgraphNode*, CUgraph, const CUgraphNode*, size_t, const CUDA_MEMCPY3D*, CUcontext) nogil> __cuGraphAddMemcpyNode)(phGraphNode, hGraph, dependencies, numDependencies, copyParams, ctx)
    return err

cdef CUresult _cuGraphMemcpyNodeGetParams(CUgraphNode hNode, CUDA_MEMCPY3D* nodeParams) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuGraphMemcpyNodeGetParams
    cuPythonInit()
    if __cuGraphMemcpyNodeGetParams == NULL:
        with gil:
            raise RuntimeError('Function "cuGraphMemcpyNodeGetParams" not found')
    err = (<CUresult (*)(CUgraphNode, CUDA_MEMCPY3D*) nogil> __cuGraphMemcpyNodeGetParams)(hNode, nodeParams)
    return err

cdef CUresult _cuGraphMemcpyNodeSetParams(CUgraphNode hNode, const CUDA_MEMCPY3D* nodeParams) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuGraphMemcpyNodeSetParams
    cuPythonInit()
    if __cuGraphMemcpyNodeSetParams == NULL:
        with gil:
            raise RuntimeError('Function "cuGraphMemcpyNodeSetParams" not found')
    err = (<CUresult (*)(CUgraphNode, const CUDA_MEMCPY3D*) nogil> __cuGraphMemcpyNodeSetParams)(hNode, nodeParams)
    return err

cdef CUresult _cuGraphAddMemsetNode(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, size_t numDependencies, const CUDA_MEMSET_NODE_PARAMS* memsetParams, CUcontext ctx) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuGraphAddMemsetNode
    cuPythonInit()
    if __cuGraphAddMemsetNode == NULL:
        with gil:
            raise RuntimeError('Function "cuGraphAddMemsetNode" not found')
    err = (<CUresult (*)(CUgraphNode*, CUgraph, const CUgraphNode*, size_t, const CUDA_MEMSET_NODE_PARAMS*, CUcontext) nogil> __cuGraphAddMemsetNode)(phGraphNode, hGraph, dependencies, numDependencies, memsetParams, ctx)
    return err

cdef CUresult _cuGraphMemsetNodeGetParams(CUgraphNode hNode, CUDA_MEMSET_NODE_PARAMS* nodeParams) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuGraphMemsetNodeGetParams
    cuPythonInit()
    if __cuGraphMemsetNodeGetParams == NULL:
        with gil:
            raise RuntimeError('Function "cuGraphMemsetNodeGetParams" not found')
    err = (<CUresult (*)(CUgraphNode, CUDA_MEMSET_NODE_PARAMS*) nogil> __cuGraphMemsetNodeGetParams)(hNode, nodeParams)
    return err

cdef CUresult _cuGraphMemsetNodeSetParams(CUgraphNode hNode, const CUDA_MEMSET_NODE_PARAMS* nodeParams) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuGraphMemsetNodeSetParams
    cuPythonInit()
    if __cuGraphMemsetNodeSetParams == NULL:
        with gil:
            raise RuntimeError('Function "cuGraphMemsetNodeSetParams" not found')
    err = (<CUresult (*)(CUgraphNode, const CUDA_MEMSET_NODE_PARAMS*) nogil> __cuGraphMemsetNodeSetParams)(hNode, nodeParams)
    return err

cdef CUresult _cuGraphAddHostNode(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, size_t numDependencies, const CUDA_HOST_NODE_PARAMS* nodeParams) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuGraphAddHostNode
    cuPythonInit()
    if __cuGraphAddHostNode == NULL:
        with gil:
            raise RuntimeError('Function "cuGraphAddHostNode" not found')
    err = (<CUresult (*)(CUgraphNode*, CUgraph, const CUgraphNode*, size_t, const CUDA_HOST_NODE_PARAMS*) nogil> __cuGraphAddHostNode)(phGraphNode, hGraph, dependencies, numDependencies, nodeParams)
    return err

cdef CUresult _cuGraphHostNodeGetParams(CUgraphNode hNode, CUDA_HOST_NODE_PARAMS* nodeParams) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuGraphHostNodeGetParams
    cuPythonInit()
    if __cuGraphHostNodeGetParams == NULL:
        with gil:
            raise RuntimeError('Function "cuGraphHostNodeGetParams" not found')
    err = (<CUresult (*)(CUgraphNode, CUDA_HOST_NODE_PARAMS*) nogil> __cuGraphHostNodeGetParams)(hNode, nodeParams)
    return err

cdef CUresult _cuGraphHostNodeSetParams(CUgraphNode hNode, const CUDA_HOST_NODE_PARAMS* nodeParams) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuGraphHostNodeSetParams
    cuPythonInit()
    if __cuGraphHostNodeSetParams == NULL:
        with gil:
            raise RuntimeError('Function "cuGraphHostNodeSetParams" not found')
    err = (<CUresult (*)(CUgraphNode, const CUDA_HOST_NODE_PARAMS*) nogil> __cuGraphHostNodeSetParams)(hNode, nodeParams)
    return err

cdef CUresult _cuGraphAddChildGraphNode(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, size_t numDependencies, CUgraph childGraph) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuGraphAddChildGraphNode
    cuPythonInit()
    if __cuGraphAddChildGraphNode == NULL:
        with gil:
            raise RuntimeError('Function "cuGraphAddChildGraphNode" not found')
    err = (<CUresult (*)(CUgraphNode*, CUgraph, const CUgraphNode*, size_t, CUgraph) nogil> __cuGraphAddChildGraphNode)(phGraphNode, hGraph, dependencies, numDependencies, childGraph)
    return err

cdef CUresult _cuGraphChildGraphNodeGetGraph(CUgraphNode hNode, CUgraph* phGraph) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuGraphChildGraphNodeGetGraph
    cuPythonInit()
    if __cuGraphChildGraphNodeGetGraph == NULL:
        with gil:
            raise RuntimeError('Function "cuGraphChildGraphNodeGetGraph" not found')
    err = (<CUresult (*)(CUgraphNode, CUgraph*) nogil> __cuGraphChildGraphNodeGetGraph)(hNode, phGraph)
    return err

cdef CUresult _cuGraphAddEmptyNode(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, size_t numDependencies) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuGraphAddEmptyNode
    cuPythonInit()
    if __cuGraphAddEmptyNode == NULL:
        with gil:
            raise RuntimeError('Function "cuGraphAddEmptyNode" not found')
    err = (<CUresult (*)(CUgraphNode*, CUgraph, const CUgraphNode*, size_t) nogil> __cuGraphAddEmptyNode)(phGraphNode, hGraph, dependencies, numDependencies)
    return err

cdef CUresult _cuGraphAddEventRecordNode(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, size_t numDependencies, CUevent event) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuGraphAddEventRecordNode
    cuPythonInit()
    if __cuGraphAddEventRecordNode == NULL:
        with gil:
            raise RuntimeError('Function "cuGraphAddEventRecordNode" not found')
    err = (<CUresult (*)(CUgraphNode*, CUgraph, const CUgraphNode*, size_t, CUevent) nogil> __cuGraphAddEventRecordNode)(phGraphNode, hGraph, dependencies, numDependencies, event)
    return err

cdef CUresult _cuGraphEventRecordNodeGetEvent(CUgraphNode hNode, CUevent* event_out) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuGraphEventRecordNodeGetEvent
    cuPythonInit()
    if __cuGraphEventRecordNodeGetEvent == NULL:
        with gil:
            raise RuntimeError('Function "cuGraphEventRecordNodeGetEvent" not found')
    err = (<CUresult (*)(CUgraphNode, CUevent*) nogil> __cuGraphEventRecordNodeGetEvent)(hNode, event_out)
    return err

cdef CUresult _cuGraphEventRecordNodeSetEvent(CUgraphNode hNode, CUevent event) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuGraphEventRecordNodeSetEvent
    cuPythonInit()
    if __cuGraphEventRecordNodeSetEvent == NULL:
        with gil:
            raise RuntimeError('Function "cuGraphEventRecordNodeSetEvent" not found')
    err = (<CUresult (*)(CUgraphNode, CUevent) nogil> __cuGraphEventRecordNodeSetEvent)(hNode, event)
    return err

cdef CUresult _cuGraphAddEventWaitNode(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, size_t numDependencies, CUevent event) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuGraphAddEventWaitNode
    cuPythonInit()
    if __cuGraphAddEventWaitNode == NULL:
        with gil:
            raise RuntimeError('Function "cuGraphAddEventWaitNode" not found')
    err = (<CUresult (*)(CUgraphNode*, CUgraph, const CUgraphNode*, size_t, CUevent) nogil> __cuGraphAddEventWaitNode)(phGraphNode, hGraph, dependencies, numDependencies, event)
    return err

cdef CUresult _cuGraphEventWaitNodeGetEvent(CUgraphNode hNode, CUevent* event_out) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuGraphEventWaitNodeGetEvent
    cuPythonInit()
    if __cuGraphEventWaitNodeGetEvent == NULL:
        with gil:
            raise RuntimeError('Function "cuGraphEventWaitNodeGetEvent" not found')
    err = (<CUresult (*)(CUgraphNode, CUevent*) nogil> __cuGraphEventWaitNodeGetEvent)(hNode, event_out)
    return err

cdef CUresult _cuGraphEventWaitNodeSetEvent(CUgraphNode hNode, CUevent event) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuGraphEventWaitNodeSetEvent
    cuPythonInit()
    if __cuGraphEventWaitNodeSetEvent == NULL:
        with gil:
            raise RuntimeError('Function "cuGraphEventWaitNodeSetEvent" not found')
    err = (<CUresult (*)(CUgraphNode, CUevent) nogil> __cuGraphEventWaitNodeSetEvent)(hNode, event)
    return err

cdef CUresult _cuGraphAddExternalSemaphoresSignalNode(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, size_t numDependencies, const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS* nodeParams) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuGraphAddExternalSemaphoresSignalNode
    cuPythonInit()
    if __cuGraphAddExternalSemaphoresSignalNode == NULL:
        with gil:
            raise RuntimeError('Function "cuGraphAddExternalSemaphoresSignalNode" not found')
    err = (<CUresult (*)(CUgraphNode*, CUgraph, const CUgraphNode*, size_t, const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS*) nogil> __cuGraphAddExternalSemaphoresSignalNode)(phGraphNode, hGraph, dependencies, numDependencies, nodeParams)
    return err

cdef CUresult _cuGraphExternalSemaphoresSignalNodeGetParams(CUgraphNode hNode, CUDA_EXT_SEM_SIGNAL_NODE_PARAMS* params_out) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuGraphExternalSemaphoresSignalNodeGetParams
    cuPythonInit()
    if __cuGraphExternalSemaphoresSignalNodeGetParams == NULL:
        with gil:
            raise RuntimeError('Function "cuGraphExternalSemaphoresSignalNodeGetParams" not found')
    err = (<CUresult (*)(CUgraphNode, CUDA_EXT_SEM_SIGNAL_NODE_PARAMS*) nogil> __cuGraphExternalSemaphoresSignalNodeGetParams)(hNode, params_out)
    return err

cdef CUresult _cuGraphExternalSemaphoresSignalNodeSetParams(CUgraphNode hNode, const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS* nodeParams) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuGraphExternalSemaphoresSignalNodeSetParams
    cuPythonInit()
    if __cuGraphExternalSemaphoresSignalNodeSetParams == NULL:
        with gil:
            raise RuntimeError('Function "cuGraphExternalSemaphoresSignalNodeSetParams" not found')
    err = (<CUresult (*)(CUgraphNode, const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS*) nogil> __cuGraphExternalSemaphoresSignalNodeSetParams)(hNode, nodeParams)
    return err

cdef CUresult _cuGraphAddExternalSemaphoresWaitNode(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, size_t numDependencies, const CUDA_EXT_SEM_WAIT_NODE_PARAMS* nodeParams) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuGraphAddExternalSemaphoresWaitNode
    cuPythonInit()
    if __cuGraphAddExternalSemaphoresWaitNode == NULL:
        with gil:
            raise RuntimeError('Function "cuGraphAddExternalSemaphoresWaitNode" not found')
    err = (<CUresult (*)(CUgraphNode*, CUgraph, const CUgraphNode*, size_t, const CUDA_EXT_SEM_WAIT_NODE_PARAMS*) nogil> __cuGraphAddExternalSemaphoresWaitNode)(phGraphNode, hGraph, dependencies, numDependencies, nodeParams)
    return err

cdef CUresult _cuGraphExternalSemaphoresWaitNodeGetParams(CUgraphNode hNode, CUDA_EXT_SEM_WAIT_NODE_PARAMS* params_out) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuGraphExternalSemaphoresWaitNodeGetParams
    cuPythonInit()
    if __cuGraphExternalSemaphoresWaitNodeGetParams == NULL:
        with gil:
            raise RuntimeError('Function "cuGraphExternalSemaphoresWaitNodeGetParams" not found')
    err = (<CUresult (*)(CUgraphNode, CUDA_EXT_SEM_WAIT_NODE_PARAMS*) nogil> __cuGraphExternalSemaphoresWaitNodeGetParams)(hNode, params_out)
    return err

cdef CUresult _cuGraphExternalSemaphoresWaitNodeSetParams(CUgraphNode hNode, const CUDA_EXT_SEM_WAIT_NODE_PARAMS* nodeParams) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuGraphExternalSemaphoresWaitNodeSetParams
    cuPythonInit()
    if __cuGraphExternalSemaphoresWaitNodeSetParams == NULL:
        with gil:
            raise RuntimeError('Function "cuGraphExternalSemaphoresWaitNodeSetParams" not found')
    err = (<CUresult (*)(CUgraphNode, const CUDA_EXT_SEM_WAIT_NODE_PARAMS*) nogil> __cuGraphExternalSemaphoresWaitNodeSetParams)(hNode, nodeParams)
    return err

cdef CUresult _cuGraphAddBatchMemOpNode(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, size_t numDependencies, const CUDA_BATCH_MEM_OP_NODE_PARAMS* nodeParams) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuGraphAddBatchMemOpNode
    cuPythonInit()
    if __cuGraphAddBatchMemOpNode == NULL:
        with gil:
            raise RuntimeError('Function "cuGraphAddBatchMemOpNode" not found')
    err = (<CUresult (*)(CUgraphNode*, CUgraph, const CUgraphNode*, size_t, const CUDA_BATCH_MEM_OP_NODE_PARAMS*) nogil> __cuGraphAddBatchMemOpNode)(phGraphNode, hGraph, dependencies, numDependencies, nodeParams)
    return err

cdef CUresult _cuGraphBatchMemOpNodeGetParams(CUgraphNode hNode, CUDA_BATCH_MEM_OP_NODE_PARAMS* nodeParams_out) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuGraphBatchMemOpNodeGetParams
    cuPythonInit()
    if __cuGraphBatchMemOpNodeGetParams == NULL:
        with gil:
            raise RuntimeError('Function "cuGraphBatchMemOpNodeGetParams" not found')
    err = (<CUresult (*)(CUgraphNode, CUDA_BATCH_MEM_OP_NODE_PARAMS*) nogil> __cuGraphBatchMemOpNodeGetParams)(hNode, nodeParams_out)
    return err

cdef CUresult _cuGraphBatchMemOpNodeSetParams(CUgraphNode hNode, const CUDA_BATCH_MEM_OP_NODE_PARAMS* nodeParams) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuGraphBatchMemOpNodeSetParams
    cuPythonInit()
    if __cuGraphBatchMemOpNodeSetParams == NULL:
        with gil:
            raise RuntimeError('Function "cuGraphBatchMemOpNodeSetParams" not found')
    err = (<CUresult (*)(CUgraphNode, const CUDA_BATCH_MEM_OP_NODE_PARAMS*) nogil> __cuGraphBatchMemOpNodeSetParams)(hNode, nodeParams)
    return err

cdef CUresult _cuGraphExecBatchMemOpNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_BATCH_MEM_OP_NODE_PARAMS* nodeParams) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuGraphExecBatchMemOpNodeSetParams
    cuPythonInit()
    if __cuGraphExecBatchMemOpNodeSetParams == NULL:
        with gil:
            raise RuntimeError('Function "cuGraphExecBatchMemOpNodeSetParams" not found')
    err = (<CUresult (*)(CUgraphExec, CUgraphNode, const CUDA_BATCH_MEM_OP_NODE_PARAMS*) nogil> __cuGraphExecBatchMemOpNodeSetParams)(hGraphExec, hNode, nodeParams)
    return err

cdef CUresult _cuGraphAddMemAllocNode(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, size_t numDependencies, CUDA_MEM_ALLOC_NODE_PARAMS* nodeParams) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuGraphAddMemAllocNode
    cuPythonInit()
    if __cuGraphAddMemAllocNode == NULL:
        with gil:
            raise RuntimeError('Function "cuGraphAddMemAllocNode" not found')
    err = (<CUresult (*)(CUgraphNode*, CUgraph, const CUgraphNode*, size_t, CUDA_MEM_ALLOC_NODE_PARAMS*) nogil> __cuGraphAddMemAllocNode)(phGraphNode, hGraph, dependencies, numDependencies, nodeParams)
    return err

cdef CUresult _cuGraphMemAllocNodeGetParams(CUgraphNode hNode, CUDA_MEM_ALLOC_NODE_PARAMS* params_out) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuGraphMemAllocNodeGetParams
    cuPythonInit()
    if __cuGraphMemAllocNodeGetParams == NULL:
        with gil:
            raise RuntimeError('Function "cuGraphMemAllocNodeGetParams" not found')
    err = (<CUresult (*)(CUgraphNode, CUDA_MEM_ALLOC_NODE_PARAMS*) nogil> __cuGraphMemAllocNodeGetParams)(hNode, params_out)
    return err

cdef CUresult _cuGraphAddMemFreeNode(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, size_t numDependencies, CUdeviceptr dptr) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuGraphAddMemFreeNode
    cuPythonInit()
    if __cuGraphAddMemFreeNode == NULL:
        with gil:
            raise RuntimeError('Function "cuGraphAddMemFreeNode" not found')
    err = (<CUresult (*)(CUgraphNode*, CUgraph, const CUgraphNode*, size_t, CUdeviceptr) nogil> __cuGraphAddMemFreeNode)(phGraphNode, hGraph, dependencies, numDependencies, dptr)
    return err

cdef CUresult _cuGraphMemFreeNodeGetParams(CUgraphNode hNode, CUdeviceptr* dptr_out) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuGraphMemFreeNodeGetParams
    cuPythonInit()
    if __cuGraphMemFreeNodeGetParams == NULL:
        with gil:
            raise RuntimeError('Function "cuGraphMemFreeNodeGetParams" not found')
    err = (<CUresult (*)(CUgraphNode, CUdeviceptr*) nogil> __cuGraphMemFreeNodeGetParams)(hNode, dptr_out)
    return err

cdef CUresult _cuDeviceGraphMemTrim(CUdevice device) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuDeviceGraphMemTrim
    cuPythonInit()
    if __cuDeviceGraphMemTrim == NULL:
        with gil:
            raise RuntimeError('Function "cuDeviceGraphMemTrim" not found')
    err = (<CUresult (*)(CUdevice) nogil> __cuDeviceGraphMemTrim)(device)
    return err

cdef CUresult _cuDeviceGetGraphMemAttribute(CUdevice device, CUgraphMem_attribute attr, void* value) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuDeviceGetGraphMemAttribute
    cuPythonInit()
    if __cuDeviceGetGraphMemAttribute == NULL:
        with gil:
            raise RuntimeError('Function "cuDeviceGetGraphMemAttribute" not found')
    err = (<CUresult (*)(CUdevice, CUgraphMem_attribute, void*) nogil> __cuDeviceGetGraphMemAttribute)(device, attr, value)
    return err

cdef CUresult _cuDeviceSetGraphMemAttribute(CUdevice device, CUgraphMem_attribute attr, void* value) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuDeviceSetGraphMemAttribute
    cuPythonInit()
    if __cuDeviceSetGraphMemAttribute == NULL:
        with gil:
            raise RuntimeError('Function "cuDeviceSetGraphMemAttribute" not found')
    err = (<CUresult (*)(CUdevice, CUgraphMem_attribute, void*) nogil> __cuDeviceSetGraphMemAttribute)(device, attr, value)
    return err

cdef CUresult _cuGraphClone(CUgraph* phGraphClone, CUgraph originalGraph) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuGraphClone
    cuPythonInit()
    if __cuGraphClone == NULL:
        with gil:
            raise RuntimeError('Function "cuGraphClone" not found')
    err = (<CUresult (*)(CUgraph*, CUgraph) nogil> __cuGraphClone)(phGraphClone, originalGraph)
    return err

cdef CUresult _cuGraphNodeFindInClone(CUgraphNode* phNode, CUgraphNode hOriginalNode, CUgraph hClonedGraph) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuGraphNodeFindInClone
    cuPythonInit()
    if __cuGraphNodeFindInClone == NULL:
        with gil:
            raise RuntimeError('Function "cuGraphNodeFindInClone" not found')
    err = (<CUresult (*)(CUgraphNode*, CUgraphNode, CUgraph) nogil> __cuGraphNodeFindInClone)(phNode, hOriginalNode, hClonedGraph)
    return err

cdef CUresult _cuGraphNodeGetType(CUgraphNode hNode, CUgraphNodeType* typename) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuGraphNodeGetType
    cuPythonInit()
    if __cuGraphNodeGetType == NULL:
        with gil:
            raise RuntimeError('Function "cuGraphNodeGetType" not found')
    err = (<CUresult (*)(CUgraphNode, CUgraphNodeType*) nogil> __cuGraphNodeGetType)(hNode, typename)
    return err

cdef CUresult _cuGraphGetNodes(CUgraph hGraph, CUgraphNode* nodes, size_t* numNodes) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuGraphGetNodes
    cuPythonInit()
    if __cuGraphGetNodes == NULL:
        with gil:
            raise RuntimeError('Function "cuGraphGetNodes" not found')
    err = (<CUresult (*)(CUgraph, CUgraphNode*, size_t*) nogil> __cuGraphGetNodes)(hGraph, nodes, numNodes)
    return err

cdef CUresult _cuGraphGetRootNodes(CUgraph hGraph, CUgraphNode* rootNodes, size_t* numRootNodes) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuGraphGetRootNodes
    cuPythonInit()
    if __cuGraphGetRootNodes == NULL:
        with gil:
            raise RuntimeError('Function "cuGraphGetRootNodes" not found')
    err = (<CUresult (*)(CUgraph, CUgraphNode*, size_t*) nogil> __cuGraphGetRootNodes)(hGraph, rootNodes, numRootNodes)
    return err

cdef CUresult _cuGraphGetEdges(CUgraph hGraph, CUgraphNode* from_, CUgraphNode* to, size_t* numEdges) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuGraphGetEdges
    cuPythonInit()
    if __cuGraphGetEdges == NULL:
        with gil:
            raise RuntimeError('Function "cuGraphGetEdges" not found')
    err = (<CUresult (*)(CUgraph, CUgraphNode*, CUgraphNode*, size_t*) nogil> __cuGraphGetEdges)(hGraph, from_, to, numEdges)
    return err

cdef CUresult _cuGraphNodeGetDependencies(CUgraphNode hNode, CUgraphNode* dependencies, size_t* numDependencies) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuGraphNodeGetDependencies
    cuPythonInit()
    if __cuGraphNodeGetDependencies == NULL:
        with gil:
            raise RuntimeError('Function "cuGraphNodeGetDependencies" not found')
    err = (<CUresult (*)(CUgraphNode, CUgraphNode*, size_t*) nogil> __cuGraphNodeGetDependencies)(hNode, dependencies, numDependencies)
    return err

cdef CUresult _cuGraphNodeGetDependentNodes(CUgraphNode hNode, CUgraphNode* dependentNodes, size_t* numDependentNodes) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuGraphNodeGetDependentNodes
    cuPythonInit()
    if __cuGraphNodeGetDependentNodes == NULL:
        with gil:
            raise RuntimeError('Function "cuGraphNodeGetDependentNodes" not found')
    err = (<CUresult (*)(CUgraphNode, CUgraphNode*, size_t*) nogil> __cuGraphNodeGetDependentNodes)(hNode, dependentNodes, numDependentNodes)
    return err

cdef CUresult _cuGraphAddDependencies(CUgraph hGraph, const CUgraphNode* from_, const CUgraphNode* to, size_t numDependencies) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuGraphAddDependencies
    cuPythonInit()
    if __cuGraphAddDependencies == NULL:
        with gil:
            raise RuntimeError('Function "cuGraphAddDependencies" not found')
    err = (<CUresult (*)(CUgraph, const CUgraphNode*, const CUgraphNode*, size_t) nogil> __cuGraphAddDependencies)(hGraph, from_, to, numDependencies)
    return err

cdef CUresult _cuGraphRemoveDependencies(CUgraph hGraph, const CUgraphNode* from_, const CUgraphNode* to, size_t numDependencies) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuGraphRemoveDependencies
    cuPythonInit()
    if __cuGraphRemoveDependencies == NULL:
        with gil:
            raise RuntimeError('Function "cuGraphRemoveDependencies" not found')
    err = (<CUresult (*)(CUgraph, const CUgraphNode*, const CUgraphNode*, size_t) nogil> __cuGraphRemoveDependencies)(hGraph, from_, to, numDependencies)
    return err

cdef CUresult _cuGraphDestroyNode(CUgraphNode hNode) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuGraphDestroyNode
    cuPythonInit()
    if __cuGraphDestroyNode == NULL:
        with gil:
            raise RuntimeError('Function "cuGraphDestroyNode" not found')
    err = (<CUresult (*)(CUgraphNode) nogil> __cuGraphDestroyNode)(hNode)
    return err

cdef CUresult _cuGraphInstantiate_v2(CUgraphExec* phGraphExec, CUgraph hGraph, CUgraphNode* phErrorNode, char* logBuffer, size_t bufferSize) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuGraphInstantiate_v2
    cuPythonInit()
    if __cuGraphInstantiate_v2 == NULL:
        with gil:
            raise RuntimeError('Function "cuGraphInstantiate_v2" not found')
    err = (<CUresult (*)(CUgraphExec*, CUgraph, CUgraphNode*, char*, size_t) nogil> __cuGraphInstantiate_v2)(phGraphExec, hGraph, phErrorNode, logBuffer, bufferSize)
    return err

cdef CUresult _cuGraphInstantiateWithFlags(CUgraphExec* phGraphExec, CUgraph hGraph, unsigned long long flags) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuGraphInstantiateWithFlags
    cuPythonInit()
    if __cuGraphInstantiateWithFlags == NULL:
        with gil:
            raise RuntimeError('Function "cuGraphInstantiateWithFlags" not found')
    err = (<CUresult (*)(CUgraphExec*, CUgraph, unsigned long long) nogil> __cuGraphInstantiateWithFlags)(phGraphExec, hGraph, flags)
    return err

cdef CUresult _cuGraphExecKernelNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_KERNEL_NODE_PARAMS* nodeParams) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuGraphExecKernelNodeSetParams
    cuPythonInit()
    if __cuGraphExecKernelNodeSetParams == NULL:
        with gil:
            raise RuntimeError('Function "cuGraphExecKernelNodeSetParams" not found')
    err = (<CUresult (*)(CUgraphExec, CUgraphNode, const CUDA_KERNEL_NODE_PARAMS*) nogil> __cuGraphExecKernelNodeSetParams)(hGraphExec, hNode, nodeParams)
    return err

cdef CUresult _cuGraphExecMemcpyNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_MEMCPY3D* copyParams, CUcontext ctx) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuGraphExecMemcpyNodeSetParams
    cuPythonInit()
    if __cuGraphExecMemcpyNodeSetParams == NULL:
        with gil:
            raise RuntimeError('Function "cuGraphExecMemcpyNodeSetParams" not found')
    err = (<CUresult (*)(CUgraphExec, CUgraphNode, const CUDA_MEMCPY3D*, CUcontext) nogil> __cuGraphExecMemcpyNodeSetParams)(hGraphExec, hNode, copyParams, ctx)
    return err

cdef CUresult _cuGraphExecMemsetNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_MEMSET_NODE_PARAMS* memsetParams, CUcontext ctx) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuGraphExecMemsetNodeSetParams
    cuPythonInit()
    if __cuGraphExecMemsetNodeSetParams == NULL:
        with gil:
            raise RuntimeError('Function "cuGraphExecMemsetNodeSetParams" not found')
    err = (<CUresult (*)(CUgraphExec, CUgraphNode, const CUDA_MEMSET_NODE_PARAMS*, CUcontext) nogil> __cuGraphExecMemsetNodeSetParams)(hGraphExec, hNode, memsetParams, ctx)
    return err

cdef CUresult _cuGraphExecHostNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_HOST_NODE_PARAMS* nodeParams) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuGraphExecHostNodeSetParams
    cuPythonInit()
    if __cuGraphExecHostNodeSetParams == NULL:
        with gil:
            raise RuntimeError('Function "cuGraphExecHostNodeSetParams" not found')
    err = (<CUresult (*)(CUgraphExec, CUgraphNode, const CUDA_HOST_NODE_PARAMS*) nogil> __cuGraphExecHostNodeSetParams)(hGraphExec, hNode, nodeParams)
    return err

cdef CUresult _cuGraphExecChildGraphNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, CUgraph childGraph) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuGraphExecChildGraphNodeSetParams
    cuPythonInit()
    if __cuGraphExecChildGraphNodeSetParams == NULL:
        with gil:
            raise RuntimeError('Function "cuGraphExecChildGraphNodeSetParams" not found')
    err = (<CUresult (*)(CUgraphExec, CUgraphNode, CUgraph) nogil> __cuGraphExecChildGraphNodeSetParams)(hGraphExec, hNode, childGraph)
    return err

cdef CUresult _cuGraphExecEventRecordNodeSetEvent(CUgraphExec hGraphExec, CUgraphNode hNode, CUevent event) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuGraphExecEventRecordNodeSetEvent
    cuPythonInit()
    if __cuGraphExecEventRecordNodeSetEvent == NULL:
        with gil:
            raise RuntimeError('Function "cuGraphExecEventRecordNodeSetEvent" not found')
    err = (<CUresult (*)(CUgraphExec, CUgraphNode, CUevent) nogil> __cuGraphExecEventRecordNodeSetEvent)(hGraphExec, hNode, event)
    return err

cdef CUresult _cuGraphExecEventWaitNodeSetEvent(CUgraphExec hGraphExec, CUgraphNode hNode, CUevent event) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuGraphExecEventWaitNodeSetEvent
    cuPythonInit()
    if __cuGraphExecEventWaitNodeSetEvent == NULL:
        with gil:
            raise RuntimeError('Function "cuGraphExecEventWaitNodeSetEvent" not found')
    err = (<CUresult (*)(CUgraphExec, CUgraphNode, CUevent) nogil> __cuGraphExecEventWaitNodeSetEvent)(hGraphExec, hNode, event)
    return err

cdef CUresult _cuGraphExecExternalSemaphoresSignalNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS* nodeParams) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuGraphExecExternalSemaphoresSignalNodeSetParams
    cuPythonInit()
    if __cuGraphExecExternalSemaphoresSignalNodeSetParams == NULL:
        with gil:
            raise RuntimeError('Function "cuGraphExecExternalSemaphoresSignalNodeSetParams" not found')
    err = (<CUresult (*)(CUgraphExec, CUgraphNode, const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS*) nogil> __cuGraphExecExternalSemaphoresSignalNodeSetParams)(hGraphExec, hNode, nodeParams)
    return err

cdef CUresult _cuGraphExecExternalSemaphoresWaitNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_EXT_SEM_WAIT_NODE_PARAMS* nodeParams) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuGraphExecExternalSemaphoresWaitNodeSetParams
    cuPythonInit()
    if __cuGraphExecExternalSemaphoresWaitNodeSetParams == NULL:
        with gil:
            raise RuntimeError('Function "cuGraphExecExternalSemaphoresWaitNodeSetParams" not found')
    err = (<CUresult (*)(CUgraphExec, CUgraphNode, const CUDA_EXT_SEM_WAIT_NODE_PARAMS*) nogil> __cuGraphExecExternalSemaphoresWaitNodeSetParams)(hGraphExec, hNode, nodeParams)
    return err

cdef CUresult _cuGraphNodeSetEnabled(CUgraphExec hGraphExec, CUgraphNode hNode, unsigned int isEnabled) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuGraphNodeSetEnabled
    cuPythonInit()
    if __cuGraphNodeSetEnabled == NULL:
        with gil:
            raise RuntimeError('Function "cuGraphNodeSetEnabled" not found')
    err = (<CUresult (*)(CUgraphExec, CUgraphNode, unsigned int) nogil> __cuGraphNodeSetEnabled)(hGraphExec, hNode, isEnabled)
    return err

cdef CUresult _cuGraphNodeGetEnabled(CUgraphExec hGraphExec, CUgraphNode hNode, unsigned int* isEnabled) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuGraphNodeGetEnabled
    cuPythonInit()
    if __cuGraphNodeGetEnabled == NULL:
        with gil:
            raise RuntimeError('Function "cuGraphNodeGetEnabled" not found')
    err = (<CUresult (*)(CUgraphExec, CUgraphNode, unsigned int*) nogil> __cuGraphNodeGetEnabled)(hGraphExec, hNode, isEnabled)
    return err

cdef CUresult _cuGraphUpload(CUgraphExec hGraphExec, CUstream hStream) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuGraphUpload
    cuPythonInit()
    if __cuGraphUpload == NULL:
        with gil:
            raise RuntimeError('Function "cuGraphUpload" not found')
    err = (<CUresult (*)(CUgraphExec, CUstream) nogil> __cuGraphUpload)(hGraphExec, hStream)
    return err

cdef CUresult _cuGraphLaunch(CUgraphExec hGraphExec, CUstream hStream) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuGraphLaunch
    cuPythonInit()
    if __cuGraphLaunch == NULL:
        with gil:
            raise RuntimeError('Function "cuGraphLaunch" not found')
    err = (<CUresult (*)(CUgraphExec, CUstream) nogil> __cuGraphLaunch)(hGraphExec, hStream)
    return err

cdef CUresult _cuGraphExecDestroy(CUgraphExec hGraphExec) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuGraphExecDestroy
    cuPythonInit()
    if __cuGraphExecDestroy == NULL:
        with gil:
            raise RuntimeError('Function "cuGraphExecDestroy" not found')
    err = (<CUresult (*)(CUgraphExec) nogil> __cuGraphExecDestroy)(hGraphExec)
    return err

cdef CUresult _cuGraphDestroy(CUgraph hGraph) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuGraphDestroy
    cuPythonInit()
    if __cuGraphDestroy == NULL:
        with gil:
            raise RuntimeError('Function "cuGraphDestroy" not found')
    err = (<CUresult (*)(CUgraph) nogil> __cuGraphDestroy)(hGraph)
    return err

cdef CUresult _cuGraphExecUpdate(CUgraphExec hGraphExec, CUgraph hGraph, CUgraphNode* hErrorNode_out, CUgraphExecUpdateResult* updateResult_out) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuGraphExecUpdate
    cuPythonInit()
    if __cuGraphExecUpdate == NULL:
        with gil:
            raise RuntimeError('Function "cuGraphExecUpdate" not found')
    err = (<CUresult (*)(CUgraphExec, CUgraph, CUgraphNode*, CUgraphExecUpdateResult*) nogil> __cuGraphExecUpdate)(hGraphExec, hGraph, hErrorNode_out, updateResult_out)
    return err

cdef CUresult _cuGraphKernelNodeCopyAttributes(CUgraphNode dst, CUgraphNode src) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuGraphKernelNodeCopyAttributes
    cuPythonInit()
    if __cuGraphKernelNodeCopyAttributes == NULL:
        with gil:
            raise RuntimeError('Function "cuGraphKernelNodeCopyAttributes" not found')
    err = (<CUresult (*)(CUgraphNode, CUgraphNode) nogil> __cuGraphKernelNodeCopyAttributes)(dst, src)
    return err

cdef CUresult _cuGraphKernelNodeGetAttribute(CUgraphNode hNode, CUkernelNodeAttrID attr, CUkernelNodeAttrValue* value_out) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuGraphKernelNodeGetAttribute
    cuPythonInit()
    if __cuGraphKernelNodeGetAttribute == NULL:
        with gil:
            raise RuntimeError('Function "cuGraphKernelNodeGetAttribute" not found')
    err = (<CUresult (*)(CUgraphNode, CUkernelNodeAttrID, CUkernelNodeAttrValue*) nogil> __cuGraphKernelNodeGetAttribute)(hNode, attr, value_out)
    return err

cdef CUresult _cuGraphKernelNodeSetAttribute(CUgraphNode hNode, CUkernelNodeAttrID attr, const CUkernelNodeAttrValue* value) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuGraphKernelNodeSetAttribute
    cuPythonInit()
    if __cuGraphKernelNodeSetAttribute == NULL:
        with gil:
            raise RuntimeError('Function "cuGraphKernelNodeSetAttribute" not found')
    err = (<CUresult (*)(CUgraphNode, CUkernelNodeAttrID, const CUkernelNodeAttrValue*) nogil> __cuGraphKernelNodeSetAttribute)(hNode, attr, value)
    return err

cdef CUresult _cuGraphDebugDotPrint(CUgraph hGraph, const char* path, unsigned int flags) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuGraphDebugDotPrint
    cuPythonInit()
    if __cuGraphDebugDotPrint == NULL:
        with gil:
            raise RuntimeError('Function "cuGraphDebugDotPrint" not found')
    err = (<CUresult (*)(CUgraph, const char*, unsigned int) nogil> __cuGraphDebugDotPrint)(hGraph, path, flags)
    return err

cdef CUresult _cuUserObjectCreate(CUuserObject* object_out, void* ptr, CUhostFn destroy, unsigned int initialRefcount, unsigned int flags) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuUserObjectCreate
    cuPythonInit()
    if __cuUserObjectCreate == NULL:
        with gil:
            raise RuntimeError('Function "cuUserObjectCreate" not found')
    err = (<CUresult (*)(CUuserObject*, void*, CUhostFn, unsigned int, unsigned int) nogil> __cuUserObjectCreate)(object_out, ptr, destroy, initialRefcount, flags)
    return err

cdef CUresult _cuUserObjectRetain(CUuserObject object, unsigned int count) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuUserObjectRetain
    cuPythonInit()
    if __cuUserObjectRetain == NULL:
        with gil:
            raise RuntimeError('Function "cuUserObjectRetain" not found')
    err = (<CUresult (*)(CUuserObject, unsigned int) nogil> __cuUserObjectRetain)(object, count)
    return err

cdef CUresult _cuUserObjectRelease(CUuserObject object, unsigned int count) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuUserObjectRelease
    cuPythonInit()
    if __cuUserObjectRelease == NULL:
        with gil:
            raise RuntimeError('Function "cuUserObjectRelease" not found')
    err = (<CUresult (*)(CUuserObject, unsigned int) nogil> __cuUserObjectRelease)(object, count)
    return err

cdef CUresult _cuGraphRetainUserObject(CUgraph graph, CUuserObject object, unsigned int count, unsigned int flags) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuGraphRetainUserObject
    cuPythonInit()
    if __cuGraphRetainUserObject == NULL:
        with gil:
            raise RuntimeError('Function "cuGraphRetainUserObject" not found')
    err = (<CUresult (*)(CUgraph, CUuserObject, unsigned int, unsigned int) nogil> __cuGraphRetainUserObject)(graph, object, count, flags)
    return err

cdef CUresult _cuGraphReleaseUserObject(CUgraph graph, CUuserObject object, unsigned int count) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuGraphReleaseUserObject
    cuPythonInit()
    if __cuGraphReleaseUserObject == NULL:
        with gil:
            raise RuntimeError('Function "cuGraphReleaseUserObject" not found')
    err = (<CUresult (*)(CUgraph, CUuserObject, unsigned int) nogil> __cuGraphReleaseUserObject)(graph, object, count)
    return err

cdef CUresult _cuOccupancyMaxActiveBlocksPerMultiprocessor(int* numBlocks, CUfunction func, int blockSize, size_t dynamicSMemSize) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuOccupancyMaxActiveBlocksPerMultiprocessor
    cuPythonInit()
    if __cuOccupancyMaxActiveBlocksPerMultiprocessor == NULL:
        with gil:
            raise RuntimeError('Function "cuOccupancyMaxActiveBlocksPerMultiprocessor" not found')
    err = (<CUresult (*)(int*, CUfunction, int, size_t) nogil> __cuOccupancyMaxActiveBlocksPerMultiprocessor)(numBlocks, func, blockSize, dynamicSMemSize)
    return err

cdef CUresult _cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int* numBlocks, CUfunction func, int blockSize, size_t dynamicSMemSize, unsigned int flags) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags
    cuPythonInit()
    if __cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags == NULL:
        with gil:
            raise RuntimeError('Function "cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags" not found')
    err = (<CUresult (*)(int*, CUfunction, int, size_t, unsigned int) nogil> __cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags)(numBlocks, func, blockSize, dynamicSMemSize, flags)
    return err

cdef CUresult _cuOccupancyMaxPotentialBlockSize(int* minGridSize, int* blockSize, CUfunction func, CUoccupancyB2DSize blockSizeToDynamicSMemSize, size_t dynamicSMemSize, int blockSizeLimit) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuOccupancyMaxPotentialBlockSize
    cuPythonInit()
    if __cuOccupancyMaxPotentialBlockSize == NULL:
        with gil:
            raise RuntimeError('Function "cuOccupancyMaxPotentialBlockSize" not found')
    err = (<CUresult (*)(int*, int*, CUfunction, CUoccupancyB2DSize, size_t, int) nogil> __cuOccupancyMaxPotentialBlockSize)(minGridSize, blockSize, func, blockSizeToDynamicSMemSize, dynamicSMemSize, blockSizeLimit)
    return err

cdef CUresult _cuOccupancyMaxPotentialBlockSizeWithFlags(int* minGridSize, int* blockSize, CUfunction func, CUoccupancyB2DSize blockSizeToDynamicSMemSize, size_t dynamicSMemSize, int blockSizeLimit, unsigned int flags) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuOccupancyMaxPotentialBlockSizeWithFlags
    cuPythonInit()
    if __cuOccupancyMaxPotentialBlockSizeWithFlags == NULL:
        with gil:
            raise RuntimeError('Function "cuOccupancyMaxPotentialBlockSizeWithFlags" not found')
    err = (<CUresult (*)(int*, int*, CUfunction, CUoccupancyB2DSize, size_t, int, unsigned int) nogil> __cuOccupancyMaxPotentialBlockSizeWithFlags)(minGridSize, blockSize, func, blockSizeToDynamicSMemSize, dynamicSMemSize, blockSizeLimit, flags)
    return err

cdef CUresult _cuOccupancyAvailableDynamicSMemPerBlock(size_t* dynamicSmemSize, CUfunction func, int numBlocks, int blockSize) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuOccupancyAvailableDynamicSMemPerBlock
    cuPythonInit()
    if __cuOccupancyAvailableDynamicSMemPerBlock == NULL:
        with gil:
            raise RuntimeError('Function "cuOccupancyAvailableDynamicSMemPerBlock" not found')
    err = (<CUresult (*)(size_t*, CUfunction, int, int) nogil> __cuOccupancyAvailableDynamicSMemPerBlock)(dynamicSmemSize, func, numBlocks, blockSize)
    return err

cdef CUresult _cuTexRefSetArray(CUtexref hTexRef, CUarray hArray, unsigned int Flags) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuTexRefSetArray
    cuPythonInit()
    if __cuTexRefSetArray == NULL:
        with gil:
            raise RuntimeError('Function "cuTexRefSetArray" not found')
    err = (<CUresult (*)(CUtexref, CUarray, unsigned int) nogil> __cuTexRefSetArray)(hTexRef, hArray, Flags)
    return err

cdef CUresult _cuTexRefSetMipmappedArray(CUtexref hTexRef, CUmipmappedArray hMipmappedArray, unsigned int Flags) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuTexRefSetMipmappedArray
    cuPythonInit()
    if __cuTexRefSetMipmappedArray == NULL:
        with gil:
            raise RuntimeError('Function "cuTexRefSetMipmappedArray" not found')
    err = (<CUresult (*)(CUtexref, CUmipmappedArray, unsigned int) nogil> __cuTexRefSetMipmappedArray)(hTexRef, hMipmappedArray, Flags)
    return err

cdef CUresult _cuTexRefSetAddress_v2(size_t* ByteOffset, CUtexref hTexRef, CUdeviceptr dptr, size_t numbytes) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuTexRefSetAddress_v2
    cuPythonInit()
    if __cuTexRefSetAddress_v2 == NULL:
        with gil:
            raise RuntimeError('Function "cuTexRefSetAddress_v2" not found')
    err = (<CUresult (*)(size_t*, CUtexref, CUdeviceptr, size_t) nogil> __cuTexRefSetAddress_v2)(ByteOffset, hTexRef, dptr, numbytes)
    return err

cdef CUresult _cuTexRefSetAddress2D_v3(CUtexref hTexRef, const CUDA_ARRAY_DESCRIPTOR* desc, CUdeviceptr dptr, size_t Pitch) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuTexRefSetAddress2D_v3
    cuPythonInit()
    if __cuTexRefSetAddress2D_v3 == NULL:
        with gil:
            raise RuntimeError('Function "cuTexRefSetAddress2D_v3" not found')
    err = (<CUresult (*)(CUtexref, const CUDA_ARRAY_DESCRIPTOR*, CUdeviceptr, size_t) nogil> __cuTexRefSetAddress2D_v3)(hTexRef, desc, dptr, Pitch)
    return err

cdef CUresult _cuTexRefSetFormat(CUtexref hTexRef, CUarray_format fmt, int NumPackedComponents) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuTexRefSetFormat
    cuPythonInit()
    if __cuTexRefSetFormat == NULL:
        with gil:
            raise RuntimeError('Function "cuTexRefSetFormat" not found')
    err = (<CUresult (*)(CUtexref, CUarray_format, int) nogil> __cuTexRefSetFormat)(hTexRef, fmt, NumPackedComponents)
    return err

cdef CUresult _cuTexRefSetAddressMode(CUtexref hTexRef, int dim, CUaddress_mode am) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuTexRefSetAddressMode
    cuPythonInit()
    if __cuTexRefSetAddressMode == NULL:
        with gil:
            raise RuntimeError('Function "cuTexRefSetAddressMode" not found')
    err = (<CUresult (*)(CUtexref, int, CUaddress_mode) nogil> __cuTexRefSetAddressMode)(hTexRef, dim, am)
    return err

cdef CUresult _cuTexRefSetFilterMode(CUtexref hTexRef, CUfilter_mode fm) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuTexRefSetFilterMode
    cuPythonInit()
    if __cuTexRefSetFilterMode == NULL:
        with gil:
            raise RuntimeError('Function "cuTexRefSetFilterMode" not found')
    err = (<CUresult (*)(CUtexref, CUfilter_mode) nogil> __cuTexRefSetFilterMode)(hTexRef, fm)
    return err

cdef CUresult _cuTexRefSetMipmapFilterMode(CUtexref hTexRef, CUfilter_mode fm) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuTexRefSetMipmapFilterMode
    cuPythonInit()
    if __cuTexRefSetMipmapFilterMode == NULL:
        with gil:
            raise RuntimeError('Function "cuTexRefSetMipmapFilterMode" not found')
    err = (<CUresult (*)(CUtexref, CUfilter_mode) nogil> __cuTexRefSetMipmapFilterMode)(hTexRef, fm)
    return err

cdef CUresult _cuTexRefSetMipmapLevelBias(CUtexref hTexRef, float bias) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuTexRefSetMipmapLevelBias
    cuPythonInit()
    if __cuTexRefSetMipmapLevelBias == NULL:
        with gil:
            raise RuntimeError('Function "cuTexRefSetMipmapLevelBias" not found')
    err = (<CUresult (*)(CUtexref, float) nogil> __cuTexRefSetMipmapLevelBias)(hTexRef, bias)
    return err

cdef CUresult _cuTexRefSetMipmapLevelClamp(CUtexref hTexRef, float minMipmapLevelClamp, float maxMipmapLevelClamp) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuTexRefSetMipmapLevelClamp
    cuPythonInit()
    if __cuTexRefSetMipmapLevelClamp == NULL:
        with gil:
            raise RuntimeError('Function "cuTexRefSetMipmapLevelClamp" not found')
    err = (<CUresult (*)(CUtexref, float, float) nogil> __cuTexRefSetMipmapLevelClamp)(hTexRef, minMipmapLevelClamp, maxMipmapLevelClamp)
    return err

cdef CUresult _cuTexRefSetMaxAnisotropy(CUtexref hTexRef, unsigned int maxAniso) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuTexRefSetMaxAnisotropy
    cuPythonInit()
    if __cuTexRefSetMaxAnisotropy == NULL:
        with gil:
            raise RuntimeError('Function "cuTexRefSetMaxAnisotropy" not found')
    err = (<CUresult (*)(CUtexref, unsigned int) nogil> __cuTexRefSetMaxAnisotropy)(hTexRef, maxAniso)
    return err

cdef CUresult _cuTexRefSetBorderColor(CUtexref hTexRef, float* pBorderColor) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuTexRefSetBorderColor
    cuPythonInit()
    if __cuTexRefSetBorderColor == NULL:
        with gil:
            raise RuntimeError('Function "cuTexRefSetBorderColor" not found')
    err = (<CUresult (*)(CUtexref, float*) nogil> __cuTexRefSetBorderColor)(hTexRef, pBorderColor)
    return err

cdef CUresult _cuTexRefSetFlags(CUtexref hTexRef, unsigned int Flags) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuTexRefSetFlags
    cuPythonInit()
    if __cuTexRefSetFlags == NULL:
        with gil:
            raise RuntimeError('Function "cuTexRefSetFlags" not found')
    err = (<CUresult (*)(CUtexref, unsigned int) nogil> __cuTexRefSetFlags)(hTexRef, Flags)
    return err

cdef CUresult _cuTexRefGetAddress_v2(CUdeviceptr* pdptr, CUtexref hTexRef) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuTexRefGetAddress_v2
    cuPythonInit()
    if __cuTexRefGetAddress_v2 == NULL:
        with gil:
            raise RuntimeError('Function "cuTexRefGetAddress_v2" not found')
    err = (<CUresult (*)(CUdeviceptr*, CUtexref) nogil> __cuTexRefGetAddress_v2)(pdptr, hTexRef)
    return err

cdef CUresult _cuTexRefGetArray(CUarray* phArray, CUtexref hTexRef) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuTexRefGetArray
    cuPythonInit()
    if __cuTexRefGetArray == NULL:
        with gil:
            raise RuntimeError('Function "cuTexRefGetArray" not found')
    err = (<CUresult (*)(CUarray*, CUtexref) nogil> __cuTexRefGetArray)(phArray, hTexRef)
    return err

cdef CUresult _cuTexRefGetMipmappedArray(CUmipmappedArray* phMipmappedArray, CUtexref hTexRef) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuTexRefGetMipmappedArray
    cuPythonInit()
    if __cuTexRefGetMipmappedArray == NULL:
        with gil:
            raise RuntimeError('Function "cuTexRefGetMipmappedArray" not found')
    err = (<CUresult (*)(CUmipmappedArray*, CUtexref) nogil> __cuTexRefGetMipmappedArray)(phMipmappedArray, hTexRef)
    return err

cdef CUresult _cuTexRefGetAddressMode(CUaddress_mode* pam, CUtexref hTexRef, int dim) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuTexRefGetAddressMode
    cuPythonInit()
    if __cuTexRefGetAddressMode == NULL:
        with gil:
            raise RuntimeError('Function "cuTexRefGetAddressMode" not found')
    err = (<CUresult (*)(CUaddress_mode*, CUtexref, int) nogil> __cuTexRefGetAddressMode)(pam, hTexRef, dim)
    return err

cdef CUresult _cuTexRefGetFilterMode(CUfilter_mode* pfm, CUtexref hTexRef) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuTexRefGetFilterMode
    cuPythonInit()
    if __cuTexRefGetFilterMode == NULL:
        with gil:
            raise RuntimeError('Function "cuTexRefGetFilterMode" not found')
    err = (<CUresult (*)(CUfilter_mode*, CUtexref) nogil> __cuTexRefGetFilterMode)(pfm, hTexRef)
    return err

cdef CUresult _cuTexRefGetFormat(CUarray_format* pFormat, int* pNumChannels, CUtexref hTexRef) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuTexRefGetFormat
    cuPythonInit()
    if __cuTexRefGetFormat == NULL:
        with gil:
            raise RuntimeError('Function "cuTexRefGetFormat" not found')
    err = (<CUresult (*)(CUarray_format*, int*, CUtexref) nogil> __cuTexRefGetFormat)(pFormat, pNumChannels, hTexRef)
    return err

cdef CUresult _cuTexRefGetMipmapFilterMode(CUfilter_mode* pfm, CUtexref hTexRef) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuTexRefGetMipmapFilterMode
    cuPythonInit()
    if __cuTexRefGetMipmapFilterMode == NULL:
        with gil:
            raise RuntimeError('Function "cuTexRefGetMipmapFilterMode" not found')
    err = (<CUresult (*)(CUfilter_mode*, CUtexref) nogil> __cuTexRefGetMipmapFilterMode)(pfm, hTexRef)
    return err

cdef CUresult _cuTexRefGetMipmapLevelBias(float* pbias, CUtexref hTexRef) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuTexRefGetMipmapLevelBias
    cuPythonInit()
    if __cuTexRefGetMipmapLevelBias == NULL:
        with gil:
            raise RuntimeError('Function "cuTexRefGetMipmapLevelBias" not found')
    err = (<CUresult (*)(float*, CUtexref) nogil> __cuTexRefGetMipmapLevelBias)(pbias, hTexRef)
    return err

cdef CUresult _cuTexRefGetMipmapLevelClamp(float* pminMipmapLevelClamp, float* pmaxMipmapLevelClamp, CUtexref hTexRef) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuTexRefGetMipmapLevelClamp
    cuPythonInit()
    if __cuTexRefGetMipmapLevelClamp == NULL:
        with gil:
            raise RuntimeError('Function "cuTexRefGetMipmapLevelClamp" not found')
    err = (<CUresult (*)(float*, float*, CUtexref) nogil> __cuTexRefGetMipmapLevelClamp)(pminMipmapLevelClamp, pmaxMipmapLevelClamp, hTexRef)
    return err

cdef CUresult _cuTexRefGetMaxAnisotropy(int* pmaxAniso, CUtexref hTexRef) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuTexRefGetMaxAnisotropy
    cuPythonInit()
    if __cuTexRefGetMaxAnisotropy == NULL:
        with gil:
            raise RuntimeError('Function "cuTexRefGetMaxAnisotropy" not found')
    err = (<CUresult (*)(int*, CUtexref) nogil> __cuTexRefGetMaxAnisotropy)(pmaxAniso, hTexRef)
    return err

cdef CUresult _cuTexRefGetBorderColor(float* pBorderColor, CUtexref hTexRef) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuTexRefGetBorderColor
    cuPythonInit()
    if __cuTexRefGetBorderColor == NULL:
        with gil:
            raise RuntimeError('Function "cuTexRefGetBorderColor" not found')
    err = (<CUresult (*)(float*, CUtexref) nogil> __cuTexRefGetBorderColor)(pBorderColor, hTexRef)
    return err

cdef CUresult _cuTexRefGetFlags(unsigned int* pFlags, CUtexref hTexRef) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuTexRefGetFlags
    cuPythonInit()
    if __cuTexRefGetFlags == NULL:
        with gil:
            raise RuntimeError('Function "cuTexRefGetFlags" not found')
    err = (<CUresult (*)(unsigned int*, CUtexref) nogil> __cuTexRefGetFlags)(pFlags, hTexRef)
    return err

cdef CUresult _cuTexRefCreate(CUtexref* pTexRef) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuTexRefCreate
    cuPythonInit()
    if __cuTexRefCreate == NULL:
        with gil:
            raise RuntimeError('Function "cuTexRefCreate" not found')
    err = (<CUresult (*)(CUtexref*) nogil> __cuTexRefCreate)(pTexRef)
    return err

cdef CUresult _cuTexRefDestroy(CUtexref hTexRef) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuTexRefDestroy
    cuPythonInit()
    if __cuTexRefDestroy == NULL:
        with gil:
            raise RuntimeError('Function "cuTexRefDestroy" not found')
    err = (<CUresult (*)(CUtexref) nogil> __cuTexRefDestroy)(hTexRef)
    return err

cdef CUresult _cuSurfRefSetArray(CUsurfref hSurfRef, CUarray hArray, unsigned int Flags) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuSurfRefSetArray
    cuPythonInit()
    if __cuSurfRefSetArray == NULL:
        with gil:
            raise RuntimeError('Function "cuSurfRefSetArray" not found')
    err = (<CUresult (*)(CUsurfref, CUarray, unsigned int) nogil> __cuSurfRefSetArray)(hSurfRef, hArray, Flags)
    return err

cdef CUresult _cuSurfRefGetArray(CUarray* phArray, CUsurfref hSurfRef) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuSurfRefGetArray
    cuPythonInit()
    if __cuSurfRefGetArray == NULL:
        with gil:
            raise RuntimeError('Function "cuSurfRefGetArray" not found')
    err = (<CUresult (*)(CUarray*, CUsurfref) nogil> __cuSurfRefGetArray)(phArray, hSurfRef)
    return err

cdef CUresult _cuTexObjectCreate(CUtexObject* pTexObject, const CUDA_RESOURCE_DESC* pResDesc, const CUDA_TEXTURE_DESC* pTexDesc, const CUDA_RESOURCE_VIEW_DESC* pResViewDesc) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuTexObjectCreate
    cuPythonInit()
    if __cuTexObjectCreate == NULL:
        with gil:
            raise RuntimeError('Function "cuTexObjectCreate" not found')
    err = (<CUresult (*)(CUtexObject*, const CUDA_RESOURCE_DESC*, const CUDA_TEXTURE_DESC*, const CUDA_RESOURCE_VIEW_DESC*) nogil> __cuTexObjectCreate)(pTexObject, pResDesc, pTexDesc, pResViewDesc)
    return err

cdef CUresult _cuTexObjectDestroy(CUtexObject texObject) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuTexObjectDestroy
    cuPythonInit()
    if __cuTexObjectDestroy == NULL:
        with gil:
            raise RuntimeError('Function "cuTexObjectDestroy" not found')
    err = (<CUresult (*)(CUtexObject) nogil> __cuTexObjectDestroy)(texObject)
    return err

cdef CUresult _cuTexObjectGetResourceDesc(CUDA_RESOURCE_DESC* pResDesc, CUtexObject texObject) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuTexObjectGetResourceDesc
    cuPythonInit()
    if __cuTexObjectGetResourceDesc == NULL:
        with gil:
            raise RuntimeError('Function "cuTexObjectGetResourceDesc" not found')
    err = (<CUresult (*)(CUDA_RESOURCE_DESC*, CUtexObject) nogil> __cuTexObjectGetResourceDesc)(pResDesc, texObject)
    return err

cdef CUresult _cuTexObjectGetTextureDesc(CUDA_TEXTURE_DESC* pTexDesc, CUtexObject texObject) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuTexObjectGetTextureDesc
    cuPythonInit()
    if __cuTexObjectGetTextureDesc == NULL:
        with gil:
            raise RuntimeError('Function "cuTexObjectGetTextureDesc" not found')
    err = (<CUresult (*)(CUDA_TEXTURE_DESC*, CUtexObject) nogil> __cuTexObjectGetTextureDesc)(pTexDesc, texObject)
    return err

cdef CUresult _cuTexObjectGetResourceViewDesc(CUDA_RESOURCE_VIEW_DESC* pResViewDesc, CUtexObject texObject) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuTexObjectGetResourceViewDesc
    cuPythonInit()
    if __cuTexObjectGetResourceViewDesc == NULL:
        with gil:
            raise RuntimeError('Function "cuTexObjectGetResourceViewDesc" not found')
    err = (<CUresult (*)(CUDA_RESOURCE_VIEW_DESC*, CUtexObject) nogil> __cuTexObjectGetResourceViewDesc)(pResViewDesc, texObject)
    return err

cdef CUresult _cuSurfObjectCreate(CUsurfObject* pSurfObject, const CUDA_RESOURCE_DESC* pResDesc) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuSurfObjectCreate
    cuPythonInit()
    if __cuSurfObjectCreate == NULL:
        with gil:
            raise RuntimeError('Function "cuSurfObjectCreate" not found')
    err = (<CUresult (*)(CUsurfObject*, const CUDA_RESOURCE_DESC*) nogil> __cuSurfObjectCreate)(pSurfObject, pResDesc)
    return err

cdef CUresult _cuSurfObjectDestroy(CUsurfObject surfObject) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuSurfObjectDestroy
    cuPythonInit()
    if __cuSurfObjectDestroy == NULL:
        with gil:
            raise RuntimeError('Function "cuSurfObjectDestroy" not found')
    err = (<CUresult (*)(CUsurfObject) nogil> __cuSurfObjectDestroy)(surfObject)
    return err

cdef CUresult _cuSurfObjectGetResourceDesc(CUDA_RESOURCE_DESC* pResDesc, CUsurfObject surfObject) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuSurfObjectGetResourceDesc
    cuPythonInit()
    if __cuSurfObjectGetResourceDesc == NULL:
        with gil:
            raise RuntimeError('Function "cuSurfObjectGetResourceDesc" not found')
    err = (<CUresult (*)(CUDA_RESOURCE_DESC*, CUsurfObject) nogil> __cuSurfObjectGetResourceDesc)(pResDesc, surfObject)
    return err

cdef CUresult _cuDeviceCanAccessPeer(int* canAccessPeer, CUdevice dev, CUdevice peerDev) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuDeviceCanAccessPeer
    cuPythonInit()
    if __cuDeviceCanAccessPeer == NULL:
        with gil:
            raise RuntimeError('Function "cuDeviceCanAccessPeer" not found')
    err = (<CUresult (*)(int*, CUdevice, CUdevice) nogil> __cuDeviceCanAccessPeer)(canAccessPeer, dev, peerDev)
    return err

cdef CUresult _cuCtxEnablePeerAccess(CUcontext peerContext, unsigned int Flags) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuCtxEnablePeerAccess
    cuPythonInit()
    if __cuCtxEnablePeerAccess == NULL:
        with gil:
            raise RuntimeError('Function "cuCtxEnablePeerAccess" not found')
    err = (<CUresult (*)(CUcontext, unsigned int) nogil> __cuCtxEnablePeerAccess)(peerContext, Flags)
    return err

cdef CUresult _cuCtxDisablePeerAccess(CUcontext peerContext) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuCtxDisablePeerAccess
    cuPythonInit()
    if __cuCtxDisablePeerAccess == NULL:
        with gil:
            raise RuntimeError('Function "cuCtxDisablePeerAccess" not found')
    err = (<CUresult (*)(CUcontext) nogil> __cuCtxDisablePeerAccess)(peerContext)
    return err

cdef CUresult _cuDeviceGetP2PAttribute(int* value, CUdevice_P2PAttribute attrib, CUdevice srcDevice, CUdevice dstDevice) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuDeviceGetP2PAttribute
    cuPythonInit()
    if __cuDeviceGetP2PAttribute == NULL:
        with gil:
            raise RuntimeError('Function "cuDeviceGetP2PAttribute" not found')
    err = (<CUresult (*)(int*, CUdevice_P2PAttribute, CUdevice, CUdevice) nogil> __cuDeviceGetP2PAttribute)(value, attrib, srcDevice, dstDevice)
    return err

cdef CUresult _cuGraphicsUnregisterResource(CUgraphicsResource resource) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuGraphicsUnregisterResource
    cuPythonInit()
    if __cuGraphicsUnregisterResource == NULL:
        with gil:
            raise RuntimeError('Function "cuGraphicsUnregisterResource" not found')
    err = (<CUresult (*)(CUgraphicsResource) nogil> __cuGraphicsUnregisterResource)(resource)
    return err

cdef CUresult _cuGraphicsSubResourceGetMappedArray(CUarray* pArray, CUgraphicsResource resource, unsigned int arrayIndex, unsigned int mipLevel) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuGraphicsSubResourceGetMappedArray
    cuPythonInit()
    if __cuGraphicsSubResourceGetMappedArray == NULL:
        with gil:
            raise RuntimeError('Function "cuGraphicsSubResourceGetMappedArray" not found')
    err = (<CUresult (*)(CUarray*, CUgraphicsResource, unsigned int, unsigned int) nogil> __cuGraphicsSubResourceGetMappedArray)(pArray, resource, arrayIndex, mipLevel)
    return err

cdef CUresult _cuGraphicsResourceGetMappedMipmappedArray(CUmipmappedArray* pMipmappedArray, CUgraphicsResource resource) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuGraphicsResourceGetMappedMipmappedArray
    cuPythonInit()
    if __cuGraphicsResourceGetMappedMipmappedArray == NULL:
        with gil:
            raise RuntimeError('Function "cuGraphicsResourceGetMappedMipmappedArray" not found')
    err = (<CUresult (*)(CUmipmappedArray*, CUgraphicsResource) nogil> __cuGraphicsResourceGetMappedMipmappedArray)(pMipmappedArray, resource)
    return err

cdef CUresult _cuGraphicsResourceGetMappedPointer_v2(CUdeviceptr* pDevPtr, size_t* pSize, CUgraphicsResource resource) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuGraphicsResourceGetMappedPointer_v2
    cuPythonInit()
    if __cuGraphicsResourceGetMappedPointer_v2 == NULL:
        with gil:
            raise RuntimeError('Function "cuGraphicsResourceGetMappedPointer_v2" not found')
    err = (<CUresult (*)(CUdeviceptr*, size_t*, CUgraphicsResource) nogil> __cuGraphicsResourceGetMappedPointer_v2)(pDevPtr, pSize, resource)
    return err

cdef CUresult _cuGraphicsResourceSetMapFlags_v2(CUgraphicsResource resource, unsigned int flags) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuGraphicsResourceSetMapFlags_v2
    cuPythonInit()
    if __cuGraphicsResourceSetMapFlags_v2 == NULL:
        with gil:
            raise RuntimeError('Function "cuGraphicsResourceSetMapFlags_v2" not found')
    err = (<CUresult (*)(CUgraphicsResource, unsigned int) nogil> __cuGraphicsResourceSetMapFlags_v2)(resource, flags)
    return err

cdef CUresult _cuGraphicsMapResources(unsigned int count, CUgraphicsResource* resources, CUstream hStream) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuGraphicsMapResources
    cuPythonInit()
    if __cuGraphicsMapResources == NULL:
        with gil:
            raise RuntimeError('Function "cuGraphicsMapResources" not found')
    err = (<CUresult (*)(unsigned int, CUgraphicsResource*, CUstream) nogil> __cuGraphicsMapResources)(count, resources, hStream)
    return err

cdef CUresult _cuGraphicsUnmapResources(unsigned int count, CUgraphicsResource* resources, CUstream hStream) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuGraphicsUnmapResources
    cuPythonInit()
    if __cuGraphicsUnmapResources == NULL:
        with gil:
            raise RuntimeError('Function "cuGraphicsUnmapResources" not found')
    err = (<CUresult (*)(unsigned int, CUgraphicsResource*, CUstream) nogil> __cuGraphicsUnmapResources)(count, resources, hStream)
    return err

cdef CUresult _cuGetProcAddress(const char* symbol, void** pfn, int cudaVersion, cuuint64_t flags) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuGetProcAddress
    cuPythonInit()
    if __cuGetProcAddress == NULL:
        with gil:
            raise RuntimeError('Function "cuGetProcAddress" not found')
    err = (<CUresult (*)(const char*, void**, int, cuuint64_t) nogil> __cuGetProcAddress)(symbol, pfn, cudaVersion, flags)
    return err

cdef CUresult _cuModuleGetLoadingMode(CUmoduleLoadingMode* mode) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuModuleGetLoadingMode
    cuPythonInit()
    if __cuModuleGetLoadingMode == NULL:
        with gil:
            raise RuntimeError('Function "cuModuleGetLoadingMode" not found')
    err = (<CUresult (*)(CUmoduleLoadingMode*) nogil> __cuModuleGetLoadingMode)(mode)
    return err

cdef CUresult _cuMemGetHandleForAddressRange(void* handle, CUdeviceptr dptr, size_t size, CUmemRangeHandleType handleType, unsigned long long flags) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuMemGetHandleForAddressRange
    cuPythonInit()
    if __cuMemGetHandleForAddressRange == NULL:
        with gil:
            raise RuntimeError('Function "cuMemGetHandleForAddressRange" not found')
    err = (<CUresult (*)(void*, CUdeviceptr, size_t, CUmemRangeHandleType, unsigned long long) nogil> __cuMemGetHandleForAddressRange)(handle, dptr, size, handleType, flags)
    return err

cdef CUresult _cuGetExportTable(const void** ppExportTable, const CUuuid* pExportTableId) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuGetExportTable
    cuPythonInit()
    if __cuGetExportTable == NULL:
        with gil:
            raise RuntimeError('Function "cuGetExportTable" not found')
    err = (<CUresult (*)(const void**, const CUuuid*) nogil> __cuGetExportTable)(ppExportTable, pExportTableId)
    return err

cdef CUresult _cuProfilerInitialize(const char* configFile, const char* outputFile, CUoutput_mode outputMode) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuProfilerInitialize
    cuPythonInit()
    if __cuProfilerInitialize == NULL:
        with gil:
            raise RuntimeError('Function "cuProfilerInitialize" not found')
    err = (<CUresult (*)(const char*, const char*, CUoutput_mode) nogil> __cuProfilerInitialize)(configFile, outputFile, outputMode)
    return err

cdef CUresult _cuProfilerStart() nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuProfilerStart
    cuPythonInit()
    if __cuProfilerStart == NULL:
        with gil:
            raise RuntimeError('Function "cuProfilerStart" not found')
    err = (<CUresult (*)() nogil> __cuProfilerStart)()
    return err

cdef CUresult _cuProfilerStop() nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuProfilerStop
    cuPythonInit()
    if __cuProfilerStop == NULL:
        with gil:
            raise RuntimeError('Function "cuProfilerStop" not found')
    err = (<CUresult (*)() nogil> __cuProfilerStop)()
    return err

cdef CUresult _cuVDPAUGetDevice(CUdevice* pDevice, VdpDevice vdpDevice, VdpGetProcAddress* vdpGetProcAddress) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuVDPAUGetDevice
    cuPythonInit()
    if __cuVDPAUGetDevice == NULL:
        with gil:
            raise RuntimeError('Function "cuVDPAUGetDevice" not found')
    err = (<CUresult (*)(CUdevice*, VdpDevice, VdpGetProcAddress*) nogil> __cuVDPAUGetDevice)(pDevice, vdpDevice, vdpGetProcAddress)
    return err

cdef CUresult _cuVDPAUCtxCreate_v2(CUcontext* pCtx, unsigned int flags, CUdevice device, VdpDevice vdpDevice, VdpGetProcAddress* vdpGetProcAddress) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuVDPAUCtxCreate_v2
    cuPythonInit()
    if __cuVDPAUCtxCreate_v2 == NULL:
        with gil:
            raise RuntimeError('Function "cuVDPAUCtxCreate_v2" not found')
    err = (<CUresult (*)(CUcontext*, unsigned int, CUdevice, VdpDevice, VdpGetProcAddress*) nogil> __cuVDPAUCtxCreate_v2)(pCtx, flags, device, vdpDevice, vdpGetProcAddress)
    return err

cdef CUresult _cuGraphicsVDPAURegisterVideoSurface(CUgraphicsResource* pCudaResource, VdpVideoSurface vdpSurface, unsigned int flags) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuGraphicsVDPAURegisterVideoSurface
    cuPythonInit()
    if __cuGraphicsVDPAURegisterVideoSurface == NULL:
        with gil:
            raise RuntimeError('Function "cuGraphicsVDPAURegisterVideoSurface" not found')
    err = (<CUresult (*)(CUgraphicsResource*, VdpVideoSurface, unsigned int) nogil> __cuGraphicsVDPAURegisterVideoSurface)(pCudaResource, vdpSurface, flags)
    return err

cdef CUresult _cuGraphicsVDPAURegisterOutputSurface(CUgraphicsResource* pCudaResource, VdpOutputSurface vdpSurface, unsigned int flags) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuGraphicsVDPAURegisterOutputSurface
    cuPythonInit()
    if __cuGraphicsVDPAURegisterOutputSurface == NULL:
        with gil:
            raise RuntimeError('Function "cuGraphicsVDPAURegisterOutputSurface" not found')
    err = (<CUresult (*)(CUgraphicsResource*, VdpOutputSurface, unsigned int) nogil> __cuGraphicsVDPAURegisterOutputSurface)(pCudaResource, vdpSurface, flags)
    return err

cdef CUresult _cuGraphicsEGLRegisterImage(CUgraphicsResource* pCudaResource, EGLImageKHR image, unsigned int flags) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuGraphicsEGLRegisterImage
    cuPythonInit()
    if __cuGraphicsEGLRegisterImage == NULL:
        with gil:
            raise RuntimeError('Function "cuGraphicsEGLRegisterImage" not found')
    err = (<CUresult (*)(CUgraphicsResource*, EGLImageKHR, unsigned int) nogil> __cuGraphicsEGLRegisterImage)(pCudaResource, image, flags)
    return err

cdef CUresult _cuEGLStreamConsumerConnect(CUeglStreamConnection* conn, EGLStreamKHR stream) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuEGLStreamConsumerConnect
    cuPythonInit()
    if __cuEGLStreamConsumerConnect == NULL:
        with gil:
            raise RuntimeError('Function "cuEGLStreamConsumerConnect" not found')
    err = (<CUresult (*)(CUeglStreamConnection*, EGLStreamKHR) nogil> __cuEGLStreamConsumerConnect)(conn, stream)
    return err

cdef CUresult _cuEGLStreamConsumerConnectWithFlags(CUeglStreamConnection* conn, EGLStreamKHR stream, unsigned int flags) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuEGLStreamConsumerConnectWithFlags
    cuPythonInit()
    if __cuEGLStreamConsumerConnectWithFlags == NULL:
        with gil:
            raise RuntimeError('Function "cuEGLStreamConsumerConnectWithFlags" not found')
    err = (<CUresult (*)(CUeglStreamConnection*, EGLStreamKHR, unsigned int) nogil> __cuEGLStreamConsumerConnectWithFlags)(conn, stream, flags)
    return err

cdef CUresult _cuEGLStreamConsumerDisconnect(CUeglStreamConnection* conn) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuEGLStreamConsumerDisconnect
    cuPythonInit()
    if __cuEGLStreamConsumerDisconnect == NULL:
        with gil:
            raise RuntimeError('Function "cuEGLStreamConsumerDisconnect" not found')
    err = (<CUresult (*)(CUeglStreamConnection*) nogil> __cuEGLStreamConsumerDisconnect)(conn)
    return err

cdef CUresult _cuEGLStreamConsumerAcquireFrame(CUeglStreamConnection* conn, CUgraphicsResource* pCudaResource, CUstream* pStream, unsigned int timeout) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuEGLStreamConsumerAcquireFrame
    cuPythonInit()
    if __cuEGLStreamConsumerAcquireFrame == NULL:
        with gil:
            raise RuntimeError('Function "cuEGLStreamConsumerAcquireFrame" not found')
    err = (<CUresult (*)(CUeglStreamConnection*, CUgraphicsResource*, CUstream*, unsigned int) nogil> __cuEGLStreamConsumerAcquireFrame)(conn, pCudaResource, pStream, timeout)
    return err

cdef CUresult _cuEGLStreamConsumerReleaseFrame(CUeglStreamConnection* conn, CUgraphicsResource pCudaResource, CUstream* pStream) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuEGLStreamConsumerReleaseFrame
    cuPythonInit()
    if __cuEGLStreamConsumerReleaseFrame == NULL:
        with gil:
            raise RuntimeError('Function "cuEGLStreamConsumerReleaseFrame" not found')
    err = (<CUresult (*)(CUeglStreamConnection*, CUgraphicsResource, CUstream*) nogil> __cuEGLStreamConsumerReleaseFrame)(conn, pCudaResource, pStream)
    return err

cdef CUresult _cuEGLStreamProducerConnect(CUeglStreamConnection* conn, EGLStreamKHR stream, EGLint width, EGLint height) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuEGLStreamProducerConnect
    cuPythonInit()
    if __cuEGLStreamProducerConnect == NULL:
        with gil:
            raise RuntimeError('Function "cuEGLStreamProducerConnect" not found')
    err = (<CUresult (*)(CUeglStreamConnection*, EGLStreamKHR, EGLint, EGLint) nogil> __cuEGLStreamProducerConnect)(conn, stream, width, height)
    return err

cdef CUresult _cuEGLStreamProducerDisconnect(CUeglStreamConnection* conn) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuEGLStreamProducerDisconnect
    cuPythonInit()
    if __cuEGLStreamProducerDisconnect == NULL:
        with gil:
            raise RuntimeError('Function "cuEGLStreamProducerDisconnect" not found')
    err = (<CUresult (*)(CUeglStreamConnection*) nogil> __cuEGLStreamProducerDisconnect)(conn)
    return err

cdef CUresult _cuEGLStreamProducerPresentFrame(CUeglStreamConnection* conn, CUeglFrame eglframe, CUstream* pStream) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuEGLStreamProducerPresentFrame
    cuPythonInit()
    if __cuEGLStreamProducerPresentFrame == NULL:
        with gil:
            raise RuntimeError('Function "cuEGLStreamProducerPresentFrame" not found')
    err = (<CUresult (*)(CUeglStreamConnection*, CUeglFrame, CUstream*) nogil> __cuEGLStreamProducerPresentFrame)(conn, eglframe, pStream)
    return err

cdef CUresult _cuEGLStreamProducerReturnFrame(CUeglStreamConnection* conn, CUeglFrame* eglframe, CUstream* pStream) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuEGLStreamProducerReturnFrame
    cuPythonInit()
    if __cuEGLStreamProducerReturnFrame == NULL:
        with gil:
            raise RuntimeError('Function "cuEGLStreamProducerReturnFrame" not found')
    err = (<CUresult (*)(CUeglStreamConnection*, CUeglFrame*, CUstream*) nogil> __cuEGLStreamProducerReturnFrame)(conn, eglframe, pStream)
    return err

cdef CUresult _cuGraphicsResourceGetMappedEglFrame(CUeglFrame* eglFrame, CUgraphicsResource resource, unsigned int index, unsigned int mipLevel) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuGraphicsResourceGetMappedEglFrame
    cuPythonInit()
    if __cuGraphicsResourceGetMappedEglFrame == NULL:
        with gil:
            raise RuntimeError('Function "cuGraphicsResourceGetMappedEglFrame" not found')
    err = (<CUresult (*)(CUeglFrame*, CUgraphicsResource, unsigned int, unsigned int) nogil> __cuGraphicsResourceGetMappedEglFrame)(eglFrame, resource, index, mipLevel)
    return err

cdef CUresult _cuEventCreateFromEGLSync(CUevent* phEvent, EGLSyncKHR eglSync, unsigned int flags) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuEventCreateFromEGLSync
    cuPythonInit()
    if __cuEventCreateFromEGLSync == NULL:
        with gil:
            raise RuntimeError('Function "cuEventCreateFromEGLSync" not found')
    err = (<CUresult (*)(CUevent*, EGLSyncKHR, unsigned int) nogil> __cuEventCreateFromEGLSync)(phEvent, eglSync, flags)
    return err

cdef CUresult _cuGraphicsGLRegisterBuffer(CUgraphicsResource* pCudaResource, GLuint buffer, unsigned int Flags) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuGraphicsGLRegisterBuffer
    cuPythonInit()
    if __cuGraphicsGLRegisterBuffer == NULL:
        with gil:
            raise RuntimeError('Function "cuGraphicsGLRegisterBuffer" not found')
    err = (<CUresult (*)(CUgraphicsResource*, GLuint, unsigned int) nogil> __cuGraphicsGLRegisterBuffer)(pCudaResource, buffer, Flags)
    return err

cdef CUresult _cuGraphicsGLRegisterImage(CUgraphicsResource* pCudaResource, GLuint image, GLenum target, unsigned int Flags) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuGraphicsGLRegisterImage
    cuPythonInit()
    if __cuGraphicsGLRegisterImage == NULL:
        with gil:
            raise RuntimeError('Function "cuGraphicsGLRegisterImage" not found')
    err = (<CUresult (*)(CUgraphicsResource*, GLuint, GLenum, unsigned int) nogil> __cuGraphicsGLRegisterImage)(pCudaResource, image, target, Flags)
    return err

cdef CUresult _cuGLGetDevices_v2(unsigned int* pCudaDeviceCount, CUdevice* pCudaDevices, unsigned int cudaDeviceCount, CUGLDeviceList deviceList) nogil except ?CUDA_ERROR_NOT_FOUND:
    global __cuGLGetDevices_v2
    cuPythonInit()
    if __cuGLGetDevices_v2 == NULL:
        with gil:
            raise RuntimeError('Function "cuGLGetDevices_v2" not found')
    err = (<CUresult (*)(unsigned int*, CUdevice*, unsigned int, CUGLDeviceList) nogil> __cuGLGetDevices_v2)(pCudaDeviceCount, pCudaDevices, cudaDeviceCount, deviceList)
    return err
