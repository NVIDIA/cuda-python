# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE
#
# This code was automatically generated across versions from 12.9.0 to 13.2.0, generator version 0.3.1.dev1630+gadce055ea.d20260422. Do not modify it directly.

from libc.stdint cimport intptr_t, uintptr_t

import os
import threading
from .utils import FunctionNotFoundError, NotSupportedError

from cuda.pathfinder import load_nvidia_dynamic_lib


###############################################################################
# Extern
###############################################################################

# You must 'from .utils import NotSupportedError' before using this template

cdef extern from "<dlfcn.h>" nogil:
    void* dlopen(const char*, int)
    char* dlerror()
    void* dlsym(void*, const char*)
    int dlclose(void*)

    enum:
        RTLD_LAZY
        RTLD_NOW
        RTLD_GLOBAL
        RTLD_LOCAL

    const void* RTLD_DEFAULT 'RTLD_DEFAULT'

cdef int get_cuda_version():
    cdef void* handle = NULL
    cdef int err, driver_ver = 0

    # Load driver to check version
    handle = dlopen('libcuda.so.1', RTLD_NOW | RTLD_GLOBAL)
    if handle == NULL:
        err_msg = dlerror()
        raise NotSupportedError(f'CUDA driver is not found ({err_msg.decode()})')
    cuDriverGetVersion = dlsym(handle, "cuDriverGetVersion")
    if cuDriverGetVersion == NULL:
        raise RuntimeError('Did not find cuDriverGetVersion symbol in libcuda.so.1')
    err = (<int (*)(int*) noexcept nogil>cuDriverGetVersion)(&driver_ver)
    if err != 0:
        raise RuntimeError(f'cuDriverGetVersion returned error code {err}')

    return driver_ver



###############################################################################
# Wrapper init
###############################################################################

cdef object __symbol_lock = threading.Lock()
cdef bint __py_driver_init = False

cdef void* __cuGetErrorString = NULL
cdef void* __cuGetErrorName = NULL
cdef void* __cuInit = NULL
cdef void* __cuDriverGetVersion = NULL
cdef void* __cuDeviceGet = NULL
cdef void* __cuDeviceGetCount = NULL
cdef void* __cuDeviceGetName = NULL
cdef void* __cuDeviceGetUuid_v2 = NULL
cdef void* __cuDeviceGetLuid = NULL
cdef void* __cuDeviceTotalMem_v2 = NULL
cdef void* __cuDeviceGetTexture1DLinearMaxWidth = NULL
cdef void* __cuDeviceGetAttribute = NULL
cdef void* __cuDeviceGetNvSciSyncAttributes = NULL
cdef void* __cuDeviceSetMemPool = NULL
cdef void* __cuDeviceGetMemPool = NULL
cdef void* __cuDeviceGetDefaultMemPool = NULL
cdef void* __cuDeviceGetExecAffinitySupport = NULL
cdef void* __cuFlushGPUDirectRDMAWrites = NULL
cdef void* __cuDeviceGetProperties = NULL
cdef void* __cuDeviceComputeCapability = NULL
cdef void* __cuDevicePrimaryCtxRetain = NULL
cdef void* __cuDevicePrimaryCtxRelease_v2 = NULL
cdef void* __cuDevicePrimaryCtxSetFlags_v2 = NULL
cdef void* __cuDevicePrimaryCtxGetState = NULL
cdef void* __cuDevicePrimaryCtxReset_v2 = NULL
cdef void* __cuCtxCreate_v4 = NULL
cdef void* __cuCtxDestroy_v2 = NULL
cdef void* __cuCtxPushCurrent_v2 = NULL
cdef void* __cuCtxPopCurrent_v2 = NULL
cdef void* __cuCtxSetCurrent = NULL
cdef void* __cuCtxGetCurrent = NULL
cdef void* __cuCtxGetDevice = NULL
cdef void* __cuCtxGetFlags = NULL
cdef void* __cuCtxSetFlags = NULL
cdef void* __cuCtxGetId = NULL
cdef void* __cuCtxSynchronize = NULL
cdef void* __cuCtxSetLimit = NULL
cdef void* __cuCtxGetLimit = NULL
cdef void* __cuCtxGetCacheConfig = NULL
cdef void* __cuCtxSetCacheConfig = NULL
cdef void* __cuCtxGetApiVersion = NULL
cdef void* __cuCtxGetStreamPriorityRange = NULL
cdef void* __cuCtxResetPersistingL2Cache = NULL
cdef void* __cuCtxGetExecAffinity = NULL
cdef void* __cuCtxRecordEvent = NULL
cdef void* __cuCtxWaitEvent = NULL
cdef void* __cuCtxAttach = NULL
cdef void* __cuCtxDetach = NULL
cdef void* __cuCtxGetSharedMemConfig = NULL
cdef void* __cuCtxSetSharedMemConfig = NULL
cdef void* __cuModuleLoad = NULL
cdef void* __cuModuleLoadData = NULL
cdef void* __cuModuleLoadDataEx = NULL
cdef void* __cuModuleLoadFatBinary = NULL
cdef void* __cuModuleUnload = NULL
cdef void* __cuModuleGetLoadingMode = NULL
cdef void* __cuModuleGetFunction = NULL
cdef void* __cuModuleGetFunctionCount = NULL
cdef void* __cuModuleEnumerateFunctions = NULL
cdef void* __cuModuleGetGlobal_v2 = NULL
cdef void* __cuLinkCreate_v2 = NULL
cdef void* __cuLinkAddData_v2 = NULL
cdef void* __cuLinkAddFile_v2 = NULL
cdef void* __cuLinkComplete = NULL
cdef void* __cuLinkDestroy = NULL
cdef void* __cuModuleGetTexRef = NULL
cdef void* __cuModuleGetSurfRef = NULL
cdef void* __cuLibraryLoadData = NULL
cdef void* __cuLibraryLoadFromFile = NULL
cdef void* __cuLibraryUnload = NULL
cdef void* __cuLibraryGetKernel = NULL
cdef void* __cuLibraryGetKernelCount = NULL
cdef void* __cuLibraryEnumerateKernels = NULL
cdef void* __cuLibraryGetModule = NULL
cdef void* __cuKernelGetFunction = NULL
cdef void* __cuKernelGetLibrary = NULL
cdef void* __cuLibraryGetGlobal = NULL
cdef void* __cuLibraryGetManaged = NULL
cdef void* __cuLibraryGetUnifiedFunction = NULL
cdef void* __cuKernelGetAttribute = NULL
cdef void* __cuKernelSetAttribute = NULL
cdef void* __cuKernelSetCacheConfig = NULL
cdef void* __cuKernelGetName = NULL
cdef void* __cuKernelGetParamInfo = NULL
cdef void* __cuMemGetInfo_v2 = NULL
cdef void* __cuMemAlloc_v2 = NULL
cdef void* __cuMemAllocPitch_v2 = NULL
cdef void* __cuMemFree_v2 = NULL
cdef void* __cuMemGetAddressRange_v2 = NULL
cdef void* __cuMemAllocHost_v2 = NULL
cdef void* __cuMemFreeHost = NULL
cdef void* __cuMemHostAlloc = NULL
cdef void* __cuMemHostGetDevicePointer_v2 = NULL
cdef void* __cuMemHostGetFlags = NULL
cdef void* __cuMemAllocManaged = NULL
cdef void* __cuDeviceRegisterAsyncNotification = NULL
cdef void* __cuDeviceUnregisterAsyncNotification = NULL
cdef void* __cuDeviceGetByPCIBusId = NULL
cdef void* __cuDeviceGetPCIBusId = NULL
cdef void* __cuIpcGetEventHandle = NULL
cdef void* __cuIpcOpenEventHandle = NULL
cdef void* __cuIpcGetMemHandle = NULL
cdef void* __cuIpcOpenMemHandle_v2 = NULL
cdef void* __cuIpcCloseMemHandle = NULL
cdef void* __cuMemHostRegister_v2 = NULL
cdef void* __cuMemHostUnregister = NULL
cdef void* __cuMemcpy = NULL
cdef void* __cuMemcpyPeer = NULL
cdef void* __cuMemcpyHtoD_v2 = NULL
cdef void* __cuMemcpyDtoH_v2 = NULL
cdef void* __cuMemcpyDtoD_v2 = NULL
cdef void* __cuMemcpyDtoA_v2 = NULL
cdef void* __cuMemcpyAtoD_v2 = NULL
cdef void* __cuMemcpyHtoA_v2 = NULL
cdef void* __cuMemcpyAtoH_v2 = NULL
cdef void* __cuMemcpyAtoA_v2 = NULL
cdef void* __cuMemcpy2D_v2 = NULL
cdef void* __cuMemcpy2DUnaligned_v2 = NULL
cdef void* __cuMemcpy3D_v2 = NULL
cdef void* __cuMemcpy3DPeer = NULL
cdef void* __cuMemcpyAsync = NULL
cdef void* __cuMemcpyPeerAsync = NULL
cdef void* __cuMemcpyHtoDAsync_v2 = NULL
cdef void* __cuMemcpyDtoHAsync_v2 = NULL
cdef void* __cuMemcpyDtoDAsync_v2 = NULL
cdef void* __cuMemcpyHtoAAsync_v2 = NULL
cdef void* __cuMemcpyAtoHAsync_v2 = NULL
cdef void* __cuMemcpy2DAsync_v2 = NULL
cdef void* __cuMemcpy3DAsync_v2 = NULL
cdef void* __cuMemcpy3DPeerAsync = NULL
cdef void* __cuMemsetD8_v2 = NULL
cdef void* __cuMemsetD16_v2 = NULL
cdef void* __cuMemsetD32_v2 = NULL
cdef void* __cuMemsetD2D8_v2 = NULL
cdef void* __cuMemsetD2D16_v2 = NULL
cdef void* __cuMemsetD2D32_v2 = NULL
cdef void* __cuMemsetD8Async = NULL
cdef void* __cuMemsetD16Async = NULL
cdef void* __cuMemsetD32Async = NULL
cdef void* __cuMemsetD2D8Async = NULL
cdef void* __cuMemsetD2D16Async = NULL
cdef void* __cuMemsetD2D32Async = NULL
cdef void* __cuArrayCreate_v2 = NULL
cdef void* __cuArrayGetDescriptor_v2 = NULL
cdef void* __cuArrayGetSparseProperties = NULL
cdef void* __cuMipmappedArrayGetSparseProperties = NULL
cdef void* __cuArrayGetMemoryRequirements = NULL
cdef void* __cuMipmappedArrayGetMemoryRequirements = NULL
cdef void* __cuArrayGetPlane = NULL
cdef void* __cuArrayDestroy = NULL
cdef void* __cuArray3DCreate_v2 = NULL
cdef void* __cuArray3DGetDescriptor_v2 = NULL
cdef void* __cuMipmappedArrayCreate = NULL
cdef void* __cuMipmappedArrayGetLevel = NULL
cdef void* __cuMipmappedArrayDestroy = NULL
cdef void* __cuMemGetHandleForAddressRange = NULL
cdef void* __cuMemBatchDecompressAsync = NULL
cdef void* __cuMemAddressReserve = NULL
cdef void* __cuMemAddressFree = NULL
cdef void* __cuMemCreate = NULL
cdef void* __cuMemRelease = NULL
cdef void* __cuMemMap = NULL
cdef void* __cuMemMapArrayAsync = NULL
cdef void* __cuMemUnmap = NULL
cdef void* __cuMemSetAccess = NULL
cdef void* __cuMemGetAccess = NULL
cdef void* __cuMemExportToShareableHandle = NULL
cdef void* __cuMemImportFromShareableHandle = NULL
cdef void* __cuMemGetAllocationGranularity = NULL
cdef void* __cuMemGetAllocationPropertiesFromHandle = NULL
cdef void* __cuMemRetainAllocationHandle = NULL
cdef void* __cuMemFreeAsync = NULL
cdef void* __cuMemAllocAsync = NULL
cdef void* __cuMemPoolTrimTo = NULL
cdef void* __cuMemPoolSetAttribute = NULL
cdef void* __cuMemPoolGetAttribute = NULL
cdef void* __cuMemPoolSetAccess = NULL
cdef void* __cuMemPoolGetAccess = NULL
cdef void* __cuMemPoolCreate = NULL
cdef void* __cuMemPoolDestroy = NULL
cdef void* __cuMemAllocFromPoolAsync = NULL
cdef void* __cuMemPoolExportToShareableHandle = NULL
cdef void* __cuMemPoolImportFromShareableHandle = NULL
cdef void* __cuMemPoolExportPointer = NULL
cdef void* __cuMemPoolImportPointer = NULL
cdef void* __cuMulticastCreate = NULL
cdef void* __cuMulticastAddDevice = NULL
cdef void* __cuMulticastBindMem = NULL
cdef void* __cuMulticastBindAddr = NULL
cdef void* __cuMulticastUnbind = NULL
cdef void* __cuMulticastGetGranularity = NULL
cdef void* __cuPointerGetAttribute = NULL
cdef void* __cuMemPrefetchAsync_v2 = NULL
cdef void* __cuMemAdvise_v2 = NULL
cdef void* __cuMemRangeGetAttribute = NULL
cdef void* __cuMemRangeGetAttributes = NULL
cdef void* __cuPointerSetAttribute = NULL
cdef void* __cuPointerGetAttributes = NULL
cdef void* __cuStreamCreate = NULL
cdef void* __cuStreamCreateWithPriority = NULL
cdef void* __cuStreamGetPriority = NULL
cdef void* __cuStreamGetDevice = NULL
cdef void* __cuStreamGetFlags = NULL
cdef void* __cuStreamGetId = NULL
cdef void* __cuStreamGetCtx = NULL
cdef void* __cuStreamGetCtx_v2 = NULL
cdef void* __cuStreamWaitEvent = NULL
cdef void* __cuStreamAddCallback = NULL
cdef void* __cuStreamBeginCapture_v2 = NULL
cdef void* __cuStreamBeginCaptureToGraph = NULL
cdef void* __cuThreadExchangeStreamCaptureMode = NULL
cdef void* __cuStreamEndCapture = NULL
cdef void* __cuStreamIsCapturing = NULL
cdef void* __cuStreamGetCaptureInfo_v3 = NULL
cdef void* __cuStreamUpdateCaptureDependencies_v2 = NULL
cdef void* __cuStreamAttachMemAsync = NULL
cdef void* __cuStreamQuery = NULL
cdef void* __cuStreamSynchronize = NULL
cdef void* __cuStreamDestroy_v2 = NULL
cdef void* __cuStreamCopyAttributes = NULL
cdef void* __cuStreamGetAttribute = NULL
cdef void* __cuStreamSetAttribute = NULL
cdef void* __cuEventCreate = NULL
cdef void* __cuEventRecord = NULL
cdef void* __cuEventRecordWithFlags = NULL
cdef void* __cuEventQuery = NULL
cdef void* __cuEventSynchronize = NULL
cdef void* __cuEventDestroy_v2 = NULL
cdef void* __cuEventElapsedTime_v2 = NULL
cdef void* __cuImportExternalMemory = NULL
cdef void* __cuExternalMemoryGetMappedBuffer = NULL
cdef void* __cuExternalMemoryGetMappedMipmappedArray = NULL
cdef void* __cuDestroyExternalMemory = NULL
cdef void* __cuImportExternalSemaphore = NULL
cdef void* __cuSignalExternalSemaphoresAsync = NULL
cdef void* __cuWaitExternalSemaphoresAsync = NULL
cdef void* __cuDestroyExternalSemaphore = NULL
cdef void* __cuStreamWaitValue32_v2 = NULL
cdef void* __cuStreamWaitValue64_v2 = NULL
cdef void* __cuStreamWriteValue32_v2 = NULL
cdef void* __cuStreamWriteValue64_v2 = NULL
cdef void* __cuStreamBatchMemOp_v2 = NULL
cdef void* __cuFuncGetAttribute = NULL
cdef void* __cuFuncSetAttribute = NULL
cdef void* __cuFuncSetCacheConfig = NULL
cdef void* __cuFuncGetModule = NULL
cdef void* __cuFuncGetName = NULL
cdef void* __cuFuncGetParamInfo = NULL
cdef void* __cuFuncIsLoaded = NULL
cdef void* __cuFuncLoad = NULL
cdef void* __cuLaunchKernel = NULL
cdef void* __cuLaunchKernelEx = NULL
cdef void* __cuLaunchCooperativeKernel = NULL
cdef void* __cuLaunchCooperativeKernelMultiDevice = NULL
cdef void* __cuLaunchHostFunc = NULL
cdef void* __cuFuncSetBlockShape = NULL
cdef void* __cuFuncSetSharedSize = NULL
cdef void* __cuParamSetSize = NULL
cdef void* __cuParamSeti = NULL
cdef void* __cuParamSetf = NULL
cdef void* __cuParamSetv = NULL
cdef void* __cuLaunch = NULL
cdef void* __cuLaunchGrid = NULL
cdef void* __cuLaunchGridAsync = NULL
cdef void* __cuParamSetTexRef = NULL
cdef void* __cuFuncSetSharedMemConfig = NULL
cdef void* __cuGraphCreate = NULL
cdef void* __cuGraphAddKernelNode_v2 = NULL
cdef void* __cuGraphKernelNodeGetParams_v2 = NULL
cdef void* __cuGraphKernelNodeSetParams_v2 = NULL
cdef void* __cuGraphAddMemcpyNode = NULL
cdef void* __cuGraphMemcpyNodeGetParams = NULL
cdef void* __cuGraphMemcpyNodeSetParams = NULL
cdef void* __cuGraphAddMemsetNode = NULL
cdef void* __cuGraphMemsetNodeGetParams = NULL
cdef void* __cuGraphMemsetNodeSetParams = NULL
cdef void* __cuGraphAddHostNode = NULL
cdef void* __cuGraphHostNodeGetParams = NULL
cdef void* __cuGraphHostNodeSetParams = NULL
cdef void* __cuGraphAddChildGraphNode = NULL
cdef void* __cuGraphChildGraphNodeGetGraph = NULL
cdef void* __cuGraphAddEmptyNode = NULL
cdef void* __cuGraphAddEventRecordNode = NULL
cdef void* __cuGraphEventRecordNodeGetEvent = NULL
cdef void* __cuGraphEventRecordNodeSetEvent = NULL
cdef void* __cuGraphAddEventWaitNode = NULL
cdef void* __cuGraphEventWaitNodeGetEvent = NULL
cdef void* __cuGraphEventWaitNodeSetEvent = NULL
cdef void* __cuGraphAddExternalSemaphoresSignalNode = NULL
cdef void* __cuGraphExternalSemaphoresSignalNodeGetParams = NULL
cdef void* __cuGraphExternalSemaphoresSignalNodeSetParams = NULL
cdef void* __cuGraphAddExternalSemaphoresWaitNode = NULL
cdef void* __cuGraphExternalSemaphoresWaitNodeGetParams = NULL
cdef void* __cuGraphExternalSemaphoresWaitNodeSetParams = NULL
cdef void* __cuGraphAddBatchMemOpNode = NULL
cdef void* __cuGraphBatchMemOpNodeGetParams = NULL
cdef void* __cuGraphBatchMemOpNodeSetParams = NULL
cdef void* __cuGraphExecBatchMemOpNodeSetParams = NULL
cdef void* __cuGraphAddMemAllocNode = NULL
cdef void* __cuGraphMemAllocNodeGetParams = NULL
cdef void* __cuGraphAddMemFreeNode = NULL
cdef void* __cuGraphMemFreeNodeGetParams = NULL
cdef void* __cuDeviceGraphMemTrim = NULL
cdef void* __cuDeviceGetGraphMemAttribute = NULL
cdef void* __cuDeviceSetGraphMemAttribute = NULL
cdef void* __cuGraphClone = NULL
cdef void* __cuGraphNodeFindInClone = NULL
cdef void* __cuGraphNodeGetType = NULL
cdef void* __cuGraphGetNodes = NULL
cdef void* __cuGraphGetRootNodes = NULL
cdef void* __cuGraphGetEdges_v2 = NULL
cdef void* __cuGraphNodeGetDependencies_v2 = NULL
cdef void* __cuGraphNodeGetDependentNodes_v2 = NULL
cdef void* __cuGraphAddDependencies_v2 = NULL
cdef void* __cuGraphRemoveDependencies_v2 = NULL
cdef void* __cuGraphDestroyNode = NULL
cdef void* __cuGraphInstantiateWithFlags = NULL
cdef void* __cuGraphInstantiateWithParams = NULL
cdef void* __cuGraphExecGetFlags = NULL
cdef void* __cuGraphExecKernelNodeSetParams_v2 = NULL
cdef void* __cuGraphExecMemcpyNodeSetParams = NULL
cdef void* __cuGraphExecMemsetNodeSetParams = NULL
cdef void* __cuGraphExecHostNodeSetParams = NULL
cdef void* __cuGraphExecChildGraphNodeSetParams = NULL
cdef void* __cuGraphExecEventRecordNodeSetEvent = NULL
cdef void* __cuGraphExecEventWaitNodeSetEvent = NULL
cdef void* __cuGraphExecExternalSemaphoresSignalNodeSetParams = NULL
cdef void* __cuGraphExecExternalSemaphoresWaitNodeSetParams = NULL
cdef void* __cuGraphNodeSetEnabled = NULL
cdef void* __cuGraphNodeGetEnabled = NULL
cdef void* __cuGraphUpload = NULL
cdef void* __cuGraphLaunch = NULL
cdef void* __cuGraphExecDestroy = NULL
cdef void* __cuGraphDestroy = NULL
cdef void* __cuGraphExecUpdate_v2 = NULL
cdef void* __cuGraphKernelNodeCopyAttributes = NULL
cdef void* __cuGraphKernelNodeGetAttribute = NULL
cdef void* __cuGraphKernelNodeSetAttribute = NULL
cdef void* __cuGraphDebugDotPrint = NULL
cdef void* __cuUserObjectCreate = NULL
cdef void* __cuUserObjectRetain = NULL
cdef void* __cuUserObjectRelease = NULL
cdef void* __cuGraphRetainUserObject = NULL
cdef void* __cuGraphReleaseUserObject = NULL
cdef void* __cuGraphAddNode_v2 = NULL
cdef void* __cuGraphNodeSetParams = NULL
cdef void* __cuGraphExecNodeSetParams = NULL
cdef void* __cuGraphConditionalHandleCreate = NULL
cdef void* __cuOccupancyMaxActiveBlocksPerMultiprocessor = NULL
cdef void* __cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags = NULL
cdef void* __cuOccupancyMaxPotentialBlockSize = NULL
cdef void* __cuOccupancyMaxPotentialBlockSizeWithFlags = NULL
cdef void* __cuOccupancyAvailableDynamicSMemPerBlock = NULL
cdef void* __cuOccupancyMaxPotentialClusterSize = NULL
cdef void* __cuOccupancyMaxActiveClusters = NULL
cdef void* __cuTexRefSetArray = NULL
cdef void* __cuTexRefSetMipmappedArray = NULL
cdef void* __cuTexRefSetAddress_v2 = NULL
cdef void* __cuTexRefSetAddress2D_v3 = NULL
cdef void* __cuTexRefSetFormat = NULL
cdef void* __cuTexRefSetAddressMode = NULL
cdef void* __cuTexRefSetFilterMode = NULL
cdef void* __cuTexRefSetMipmapFilterMode = NULL
cdef void* __cuTexRefSetMipmapLevelBias = NULL
cdef void* __cuTexRefSetMipmapLevelClamp = NULL
cdef void* __cuTexRefSetMaxAnisotropy = NULL
cdef void* __cuTexRefSetBorderColor = NULL
cdef void* __cuTexRefSetFlags = NULL
cdef void* __cuTexRefGetAddress_v2 = NULL
cdef void* __cuTexRefGetArray = NULL
cdef void* __cuTexRefGetMipmappedArray = NULL
cdef void* __cuTexRefGetAddressMode = NULL
cdef void* __cuTexRefGetFilterMode = NULL
cdef void* __cuTexRefGetFormat = NULL
cdef void* __cuTexRefGetMipmapFilterMode = NULL
cdef void* __cuTexRefGetMipmapLevelBias = NULL
cdef void* __cuTexRefGetMipmapLevelClamp = NULL
cdef void* __cuTexRefGetMaxAnisotropy = NULL
cdef void* __cuTexRefGetBorderColor = NULL
cdef void* __cuTexRefGetFlags = NULL
cdef void* __cuTexRefCreate = NULL
cdef void* __cuTexRefDestroy = NULL
cdef void* __cuSurfRefSetArray = NULL
cdef void* __cuSurfRefGetArray = NULL
cdef void* __cuTexObjectCreate = NULL
cdef void* __cuTexObjectDestroy = NULL
cdef void* __cuTexObjectGetResourceDesc = NULL
cdef void* __cuTexObjectGetTextureDesc = NULL
cdef void* __cuTexObjectGetResourceViewDesc = NULL
cdef void* __cuSurfObjectCreate = NULL
cdef void* __cuSurfObjectDestroy = NULL
cdef void* __cuSurfObjectGetResourceDesc = NULL
cdef void* __cuTensorMapEncodeTiled = NULL
cdef void* __cuTensorMapEncodeIm2col = NULL
cdef void* __cuTensorMapEncodeIm2colWide = NULL
cdef void* __cuTensorMapReplaceAddress = NULL
cdef void* __cuDeviceCanAccessPeer = NULL
cdef void* __cuCtxEnablePeerAccess = NULL
cdef void* __cuCtxDisablePeerAccess = NULL
cdef void* __cuDeviceGetP2PAttribute = NULL
cdef void* __cuGraphicsUnregisterResource = NULL
cdef void* __cuGraphicsSubResourceGetMappedArray = NULL
cdef void* __cuGraphicsResourceGetMappedMipmappedArray = NULL
cdef void* __cuGraphicsResourceGetMappedPointer_v2 = NULL
cdef void* __cuGraphicsResourceSetMapFlags_v2 = NULL
cdef void* __cuGraphicsMapResources = NULL
cdef void* __cuGraphicsUnmapResources = NULL
cdef void* __cuGetProcAddress_v2 = NULL
cdef void* __cuCoredumpGetAttribute = NULL
cdef void* __cuCoredumpGetAttributeGlobal = NULL
cdef void* __cuCoredumpSetAttribute = NULL
cdef void* __cuCoredumpSetAttributeGlobal = NULL
cdef void* __cuGetExportTable = NULL
cdef void* __cuGreenCtxCreate = NULL
cdef void* __cuGreenCtxDestroy = NULL
cdef void* __cuCtxFromGreenCtx = NULL
cdef void* __cuDeviceGetDevResource = NULL
cdef void* __cuCtxGetDevResource = NULL
cdef void* __cuGreenCtxGetDevResource = NULL
cdef void* __cuDevSmResourceSplitByCount = NULL
cdef void* __cuDevResourceGenerateDesc = NULL
cdef void* __cuGreenCtxRecordEvent = NULL
cdef void* __cuGreenCtxWaitEvent = NULL
cdef void* __cuStreamGetGreenCtx = NULL
cdef void* __cuGreenCtxStreamCreate = NULL
cdef void* __cuLogsRegisterCallback = NULL
cdef void* __cuLogsUnregisterCallback = NULL
cdef void* __cuLogsCurrent = NULL
cdef void* __cuLogsDumpToFile = NULL
cdef void* __cuLogsDumpToMemory = NULL
cdef void* __cuCheckpointProcessGetRestoreThreadId = NULL
cdef void* __cuCheckpointProcessGetState = NULL
cdef void* __cuCheckpointProcessLock = NULL
cdef void* __cuCheckpointProcessCheckpoint = NULL
cdef void* __cuCheckpointProcessRestore = NULL
cdef void* __cuCheckpointProcessUnlock = NULL
cdef void* __cuGraphicsEGLRegisterImage = NULL
cdef void* __cuEGLStreamConsumerConnect = NULL
cdef void* __cuEGLStreamConsumerConnectWithFlags = NULL
cdef void* __cuEGLStreamConsumerDisconnect = NULL
cdef void* __cuEGLStreamConsumerAcquireFrame = NULL
cdef void* __cuEGLStreamConsumerReleaseFrame = NULL
cdef void* __cuEGLStreamProducerConnect = NULL
cdef void* __cuEGLStreamProducerDisconnect = NULL
cdef void* __cuEGLStreamProducerPresentFrame = NULL
cdef void* __cuEGLStreamProducerReturnFrame = NULL
cdef void* __cuGraphicsResourceGetMappedEglFrame = NULL
cdef void* __cuEventCreateFromEGLSync = NULL
cdef void* __cuGraphicsGLRegisterBuffer = NULL
cdef void* __cuGraphicsGLRegisterImage = NULL
cdef void* __cuGLGetDevices_v2 = NULL
cdef void* __cuGLCtxCreate_v2 = NULL
cdef void* __cuGLInit = NULL
cdef void* __cuGLRegisterBufferObject = NULL
cdef void* __cuGLMapBufferObject_v2 = NULL
cdef void* __cuGLUnmapBufferObject = NULL
cdef void* __cuGLUnregisterBufferObject = NULL
cdef void* __cuGLSetBufferObjectMapFlags = NULL
cdef void* __cuGLMapBufferObjectAsync_v2 = NULL
cdef void* __cuGLUnmapBufferObjectAsync = NULL
cdef void* __cuProfilerInitialize = NULL
cdef void* __cuProfilerStart = NULL
cdef void* __cuProfilerStop = NULL
cdef void* __cuVDPAUGetDevice = NULL
cdef void* __cuVDPAUCtxCreate_v2 = NULL
cdef void* __cuGraphicsVDPAURegisterVideoSurface = NULL
cdef void* __cuGraphicsVDPAURegisterOutputSurface = NULL
cdef void* __cuDeviceGetHostAtomicCapabilities = NULL
cdef void* __cuCtxGetDevice_v2 = NULL
cdef void* __cuCtxSynchronize_v2 = NULL
cdef void* __cuMemcpyBatchAsync_v2 = NULL
cdef void* __cuMemcpy3DBatchAsync_v2 = NULL
cdef void* __cuMemGetDefaultMemPool = NULL
cdef void* __cuMemGetMemPool = NULL
cdef void* __cuMemSetMemPool = NULL
cdef void* __cuMemPrefetchBatchAsync = NULL
cdef void* __cuMemDiscardBatchAsync = NULL
cdef void* __cuMemDiscardAndPrefetchBatchAsync = NULL
cdef void* __cuDeviceGetP2PAtomicCapabilities = NULL
cdef void* __cuGreenCtxGetId = NULL
cdef void* __cuMulticastBindMem_v2 = NULL
cdef void* __cuMulticastBindAddr_v2 = NULL
cdef void* __cuGraphNodeGetContainingGraph = NULL
cdef void* __cuGraphNodeGetLocalId = NULL
cdef void* __cuGraphNodeGetToolsId = NULL
cdef void* __cuGraphGetId = NULL
cdef void* __cuGraphExecGetId = NULL
cdef void* __cuDevSmResourceSplit = NULL
cdef void* __cuStreamGetDevResource = NULL
cdef void* __cuKernelGetParamCount = NULL
cdef void* __cuMemcpyWithAttributesAsync = NULL
cdef void* __cuMemcpy3DWithAttributesAsync = NULL
cdef void* __cuStreamBeginCaptureToCig = NULL
cdef void* __cuStreamEndCaptureToCig = NULL
cdef void* __cuFuncGetParamCount = NULL
cdef void* __cuLaunchHostFunc_v2 = NULL
cdef void* __cuGraphNodeGetParams = NULL
cdef void* __cuCoredumpRegisterStartCallback = NULL
cdef void* __cuCoredumpRegisterCompleteCallback = NULL
cdef void* __cuCoredumpDeregisterStartCallback = NULL
cdef void* __cuCoredumpDeregisterCompleteCallback = NULL



cdef void* load_library() except* with gil:
    cdef uintptr_t handle = load_nvidia_dynamic_lib("cuda")._handle_uint
    return <void*>handle


ctypedef CUresult (*__cuGetProcAddress_v2_T)(const char*, void**, int, cuuint64_t, CUdriverProcAddressQueryResult*) except?CUDA_ERROR_NOT_FOUND nogil
cdef __cuGetProcAddress_v2_T _F_cuGetProcAddress_v2 = NULL


cdef int _init_driver() except -1 nogil:
    global __py_driver_init

    cdef void* handle = NULL
    cdef int ptds_mode

    with gil, __symbol_lock:
        # Recheck the flag after obtaining the locks
        if __py_driver_init:
            return 0

        handle = load_library()
        if handle == NULL:
            raise RuntimeError('Failed to open cuda')

        # Get latest __cuGetProcAddress_v2
        global __cuGetProcAddress_v2
        __cuGetProcAddress_v2 = dlsym(handle, 'cuGetProcAddress_v2')
        if __cuGetProcAddress_v2 == NULL:
            raise RuntimeError("Failed to get __cuGetProcAddress_v2")
        _F_cuGetProcAddress_v2 = <__cuGetProcAddress_v2_T>__cuGetProcAddress_v2

        if os.getenv('CUDA_PYTHON_CUDA_PER_THREAD_DEFAULT_STREAM', default=0):
            ptds_mode = CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM
        else:
            ptds_mode = CU_GET_PROC_ADDRESS_DEFAULT

        # Load function
        global __cuGetErrorString
        _F_cuGetProcAddress_v2('cuGetErrorString', <void **>&__cuGetErrorString, 6000, ptds_mode, NULL)

        global __cuGetErrorName
        _F_cuGetProcAddress_v2('cuGetErrorName', <void **>&__cuGetErrorName, 6000, ptds_mode, NULL)

        global __cuInit
        _F_cuGetProcAddress_v2('cuInit', <void **>&__cuInit, 2000, ptds_mode, NULL)

        global __cuDriverGetVersion
        _F_cuGetProcAddress_v2('cuDriverGetVersion', <void **>&__cuDriverGetVersion, 2020, ptds_mode, NULL)

        global __cuDeviceGet
        _F_cuGetProcAddress_v2('cuDeviceGet', <void **>&__cuDeviceGet, 2000, ptds_mode, NULL)

        global __cuDeviceGetCount
        _F_cuGetProcAddress_v2('cuDeviceGetCount', <void **>&__cuDeviceGetCount, 2000, ptds_mode, NULL)

        global __cuDeviceGetName
        _F_cuGetProcAddress_v2('cuDeviceGetName', <void **>&__cuDeviceGetName, 2000, ptds_mode, NULL)

        global __cuDeviceGetUuid_v2
        _F_cuGetProcAddress_v2('cuDeviceGetUuid', <void **>&__cuDeviceGetUuid_v2, 11040, ptds_mode, NULL)

        global __cuDeviceGetLuid
        _F_cuGetProcAddress_v2('cuDeviceGetLuid', <void **>&__cuDeviceGetLuid, 10000, ptds_mode, NULL)

        global __cuDeviceTotalMem_v2
        _F_cuGetProcAddress_v2('cuDeviceTotalMem', <void **>&__cuDeviceTotalMem_v2, 3020, ptds_mode, NULL)

        global __cuDeviceGetTexture1DLinearMaxWidth
        _F_cuGetProcAddress_v2('cuDeviceGetTexture1DLinearMaxWidth', <void **>&__cuDeviceGetTexture1DLinearMaxWidth, 11010, ptds_mode, NULL)

        global __cuDeviceGetAttribute
        _F_cuGetProcAddress_v2('cuDeviceGetAttribute', <void **>&__cuDeviceGetAttribute, 2000, ptds_mode, NULL)

        global __cuDeviceGetNvSciSyncAttributes
        _F_cuGetProcAddress_v2('cuDeviceGetNvSciSyncAttributes', <void **>&__cuDeviceGetNvSciSyncAttributes, 10020, ptds_mode, NULL)

        global __cuDeviceSetMemPool
        _F_cuGetProcAddress_v2('cuDeviceSetMemPool', <void **>&__cuDeviceSetMemPool, 11020, ptds_mode, NULL)

        global __cuDeviceGetMemPool
        _F_cuGetProcAddress_v2('cuDeviceGetMemPool', <void **>&__cuDeviceGetMemPool, 11020, ptds_mode, NULL)

        global __cuDeviceGetDefaultMemPool
        _F_cuGetProcAddress_v2('cuDeviceGetDefaultMemPool', <void **>&__cuDeviceGetDefaultMemPool, 11020, ptds_mode, NULL)

        global __cuDeviceGetExecAffinitySupport
        _F_cuGetProcAddress_v2('cuDeviceGetExecAffinitySupport', <void **>&__cuDeviceGetExecAffinitySupport, 11040, ptds_mode, NULL)

        global __cuFlushGPUDirectRDMAWrites
        _F_cuGetProcAddress_v2('cuFlushGPUDirectRDMAWrites', <void **>&__cuFlushGPUDirectRDMAWrites, 11030, ptds_mode, NULL)

        global __cuDeviceGetProperties
        _F_cuGetProcAddress_v2('cuDeviceGetProperties', <void **>&__cuDeviceGetProperties, 2000, ptds_mode, NULL)

        global __cuDeviceComputeCapability
        _F_cuGetProcAddress_v2('cuDeviceComputeCapability', <void **>&__cuDeviceComputeCapability, 2000, ptds_mode, NULL)

        global __cuDevicePrimaryCtxRetain
        _F_cuGetProcAddress_v2('cuDevicePrimaryCtxRetain', <void **>&__cuDevicePrimaryCtxRetain, 7000, ptds_mode, NULL)

        global __cuDevicePrimaryCtxRelease_v2
        _F_cuGetProcAddress_v2('cuDevicePrimaryCtxRelease', <void **>&__cuDevicePrimaryCtxRelease_v2, 11000, ptds_mode, NULL)

        global __cuDevicePrimaryCtxSetFlags_v2
        _F_cuGetProcAddress_v2('cuDevicePrimaryCtxSetFlags', <void **>&__cuDevicePrimaryCtxSetFlags_v2, 11000, ptds_mode, NULL)

        global __cuDevicePrimaryCtxGetState
        _F_cuGetProcAddress_v2('cuDevicePrimaryCtxGetState', <void **>&__cuDevicePrimaryCtxGetState, 7000, ptds_mode, NULL)

        global __cuDevicePrimaryCtxReset_v2
        _F_cuGetProcAddress_v2('cuDevicePrimaryCtxReset', <void **>&__cuDevicePrimaryCtxReset_v2, 11000, ptds_mode, NULL)

        global __cuCtxCreate_v4
        _F_cuGetProcAddress_v2('cuCtxCreate', <void **>&__cuCtxCreate_v4, 12050, ptds_mode, NULL)

        global __cuCtxDestroy_v2
        _F_cuGetProcAddress_v2('cuCtxDestroy', <void **>&__cuCtxDestroy_v2, 4000, ptds_mode, NULL)

        global __cuCtxPushCurrent_v2
        _F_cuGetProcAddress_v2('cuCtxPushCurrent', <void **>&__cuCtxPushCurrent_v2, 4000, ptds_mode, NULL)

        global __cuCtxPopCurrent_v2
        _F_cuGetProcAddress_v2('cuCtxPopCurrent', <void **>&__cuCtxPopCurrent_v2, 4000, ptds_mode, NULL)

        global __cuCtxSetCurrent
        _F_cuGetProcAddress_v2('cuCtxSetCurrent', <void **>&__cuCtxSetCurrent, 4000, ptds_mode, NULL)

        global __cuCtxGetCurrent
        _F_cuGetProcAddress_v2('cuCtxGetCurrent', <void **>&__cuCtxGetCurrent, 4000, ptds_mode, NULL)

        global __cuCtxGetDevice
        _F_cuGetProcAddress_v2('cuCtxGetDevice', <void **>&__cuCtxGetDevice, 2000, ptds_mode, NULL)

        global __cuCtxGetFlags
        _F_cuGetProcAddress_v2('cuCtxGetFlags', <void **>&__cuCtxGetFlags, 7000, ptds_mode, NULL)

        global __cuCtxSetFlags
        _F_cuGetProcAddress_v2('cuCtxSetFlags', <void **>&__cuCtxSetFlags, 12010, ptds_mode, NULL)

        global __cuCtxGetId
        _F_cuGetProcAddress_v2('cuCtxGetId', <void **>&__cuCtxGetId, 12000, ptds_mode, NULL)

        global __cuCtxSynchronize
        _F_cuGetProcAddress_v2('cuCtxSynchronize', <void **>&__cuCtxSynchronize, 2000, ptds_mode, NULL)

        global __cuCtxSetLimit
        _F_cuGetProcAddress_v2('cuCtxSetLimit', <void **>&__cuCtxSetLimit, 3010, ptds_mode, NULL)

        global __cuCtxGetLimit
        _F_cuGetProcAddress_v2('cuCtxGetLimit', <void **>&__cuCtxGetLimit, 3010, ptds_mode, NULL)

        global __cuCtxGetCacheConfig
        _F_cuGetProcAddress_v2('cuCtxGetCacheConfig', <void **>&__cuCtxGetCacheConfig, 3020, ptds_mode, NULL)

        global __cuCtxSetCacheConfig
        _F_cuGetProcAddress_v2('cuCtxSetCacheConfig', <void **>&__cuCtxSetCacheConfig, 3020, ptds_mode, NULL)

        global __cuCtxGetApiVersion
        _F_cuGetProcAddress_v2('cuCtxGetApiVersion', <void **>&__cuCtxGetApiVersion, 3020, ptds_mode, NULL)

        global __cuCtxGetStreamPriorityRange
        _F_cuGetProcAddress_v2('cuCtxGetStreamPriorityRange', <void **>&__cuCtxGetStreamPriorityRange, 5050, ptds_mode, NULL)

        global __cuCtxResetPersistingL2Cache
        _F_cuGetProcAddress_v2('cuCtxResetPersistingL2Cache', <void **>&__cuCtxResetPersistingL2Cache, 11000, ptds_mode, NULL)

        global __cuCtxGetExecAffinity
        _F_cuGetProcAddress_v2('cuCtxGetExecAffinity', <void **>&__cuCtxGetExecAffinity, 11040, ptds_mode, NULL)

        global __cuCtxRecordEvent
        _F_cuGetProcAddress_v2('cuCtxRecordEvent', <void **>&__cuCtxRecordEvent, 12050, ptds_mode, NULL)

        global __cuCtxWaitEvent
        _F_cuGetProcAddress_v2('cuCtxWaitEvent', <void **>&__cuCtxWaitEvent, 12050, ptds_mode, NULL)

        global __cuCtxAttach
        _F_cuGetProcAddress_v2('cuCtxAttach', <void **>&__cuCtxAttach, 2000, ptds_mode, NULL)

        global __cuCtxDetach
        _F_cuGetProcAddress_v2('cuCtxDetach', <void **>&__cuCtxDetach, 2000, ptds_mode, NULL)

        global __cuCtxGetSharedMemConfig
        _F_cuGetProcAddress_v2('cuCtxGetSharedMemConfig', <void **>&__cuCtxGetSharedMemConfig, 4020, ptds_mode, NULL)

        global __cuCtxSetSharedMemConfig
        _F_cuGetProcAddress_v2('cuCtxSetSharedMemConfig', <void **>&__cuCtxSetSharedMemConfig, 4020, ptds_mode, NULL)

        global __cuModuleLoad
        _F_cuGetProcAddress_v2('cuModuleLoad', <void **>&__cuModuleLoad, 2000, ptds_mode, NULL)

        global __cuModuleLoadData
        _F_cuGetProcAddress_v2('cuModuleLoadData', <void **>&__cuModuleLoadData, 2000, ptds_mode, NULL)

        global __cuModuleLoadDataEx
        _F_cuGetProcAddress_v2('cuModuleLoadDataEx', <void **>&__cuModuleLoadDataEx, 2010, ptds_mode, NULL)

        global __cuModuleLoadFatBinary
        _F_cuGetProcAddress_v2('cuModuleLoadFatBinary', <void **>&__cuModuleLoadFatBinary, 2000, ptds_mode, NULL)

        global __cuModuleUnload
        _F_cuGetProcAddress_v2('cuModuleUnload', <void **>&__cuModuleUnload, 2000, ptds_mode, NULL)

        global __cuModuleGetLoadingMode
        _F_cuGetProcAddress_v2('cuModuleGetLoadingMode', <void **>&__cuModuleGetLoadingMode, 11070, ptds_mode, NULL)

        global __cuModuleGetFunction
        _F_cuGetProcAddress_v2('cuModuleGetFunction', <void **>&__cuModuleGetFunction, 2000, ptds_mode, NULL)

        global __cuModuleGetFunctionCount
        _F_cuGetProcAddress_v2('cuModuleGetFunctionCount', <void **>&__cuModuleGetFunctionCount, 12040, ptds_mode, NULL)

        global __cuModuleEnumerateFunctions
        _F_cuGetProcAddress_v2('cuModuleEnumerateFunctions', <void **>&__cuModuleEnumerateFunctions, 12040, ptds_mode, NULL)

        global __cuModuleGetGlobal_v2
        _F_cuGetProcAddress_v2('cuModuleGetGlobal', <void **>&__cuModuleGetGlobal_v2, 3020, ptds_mode, NULL)

        global __cuLinkCreate_v2
        _F_cuGetProcAddress_v2('cuLinkCreate', <void **>&__cuLinkCreate_v2, 6050, ptds_mode, NULL)

        global __cuLinkAddData_v2
        _F_cuGetProcAddress_v2('cuLinkAddData', <void **>&__cuLinkAddData_v2, 6050, ptds_mode, NULL)

        global __cuLinkAddFile_v2
        _F_cuGetProcAddress_v2('cuLinkAddFile', <void **>&__cuLinkAddFile_v2, 6050, ptds_mode, NULL)

        global __cuLinkComplete
        _F_cuGetProcAddress_v2('cuLinkComplete', <void **>&__cuLinkComplete, 5050, ptds_mode, NULL)

        global __cuLinkDestroy
        _F_cuGetProcAddress_v2('cuLinkDestroy', <void **>&__cuLinkDestroy, 5050, ptds_mode, NULL)

        global __cuModuleGetTexRef
        _F_cuGetProcAddress_v2('cuModuleGetTexRef', <void **>&__cuModuleGetTexRef, 2000, ptds_mode, NULL)

        global __cuModuleGetSurfRef
        _F_cuGetProcAddress_v2('cuModuleGetSurfRef', <void **>&__cuModuleGetSurfRef, 3000, ptds_mode, NULL)

        global __cuLibraryLoadData
        _F_cuGetProcAddress_v2('cuLibraryLoadData', <void **>&__cuLibraryLoadData, 12000, ptds_mode, NULL)

        global __cuLibraryLoadFromFile
        _F_cuGetProcAddress_v2('cuLibraryLoadFromFile', <void **>&__cuLibraryLoadFromFile, 12000, ptds_mode, NULL)

        global __cuLibraryUnload
        _F_cuGetProcAddress_v2('cuLibraryUnload', <void **>&__cuLibraryUnload, 12000, ptds_mode, NULL)

        global __cuLibraryGetKernel
        _F_cuGetProcAddress_v2('cuLibraryGetKernel', <void **>&__cuLibraryGetKernel, 12000, ptds_mode, NULL)

        global __cuLibraryGetKernelCount
        _F_cuGetProcAddress_v2('cuLibraryGetKernelCount', <void **>&__cuLibraryGetKernelCount, 12040, ptds_mode, NULL)

        global __cuLibraryEnumerateKernels
        _F_cuGetProcAddress_v2('cuLibraryEnumerateKernels', <void **>&__cuLibraryEnumerateKernels, 12040, ptds_mode, NULL)

        global __cuLibraryGetModule
        _F_cuGetProcAddress_v2('cuLibraryGetModule', <void **>&__cuLibraryGetModule, 12000, ptds_mode, NULL)

        global __cuKernelGetFunction
        _F_cuGetProcAddress_v2('cuKernelGetFunction', <void **>&__cuKernelGetFunction, 12000, ptds_mode, NULL)

        global __cuKernelGetLibrary
        _F_cuGetProcAddress_v2('cuKernelGetLibrary', <void **>&__cuKernelGetLibrary, 12050, ptds_mode, NULL)

        global __cuLibraryGetGlobal
        _F_cuGetProcAddress_v2('cuLibraryGetGlobal', <void **>&__cuLibraryGetGlobal, 12000, ptds_mode, NULL)

        global __cuLibraryGetManaged
        _F_cuGetProcAddress_v2('cuLibraryGetManaged', <void **>&__cuLibraryGetManaged, 12000, ptds_mode, NULL)

        global __cuLibraryGetUnifiedFunction
        _F_cuGetProcAddress_v2('cuLibraryGetUnifiedFunction', <void **>&__cuLibraryGetUnifiedFunction, 12000, ptds_mode, NULL)

        global __cuKernelGetAttribute
        _F_cuGetProcAddress_v2('cuKernelGetAttribute', <void **>&__cuKernelGetAttribute, 12000, ptds_mode, NULL)

        global __cuKernelSetAttribute
        _F_cuGetProcAddress_v2('cuKernelSetAttribute', <void **>&__cuKernelSetAttribute, 12000, ptds_mode, NULL)

        global __cuKernelSetCacheConfig
        _F_cuGetProcAddress_v2('cuKernelSetCacheConfig', <void **>&__cuKernelSetCacheConfig, 12000, ptds_mode, NULL)

        global __cuKernelGetName
        _F_cuGetProcAddress_v2('cuKernelGetName', <void **>&__cuKernelGetName, 12030, ptds_mode, NULL)

        global __cuKernelGetParamInfo
        _F_cuGetProcAddress_v2('cuKernelGetParamInfo', <void **>&__cuKernelGetParamInfo, 12040, ptds_mode, NULL)

        global __cuMemGetInfo_v2
        _F_cuGetProcAddress_v2('cuMemGetInfo', <void **>&__cuMemGetInfo_v2, 3020, ptds_mode, NULL)

        global __cuMemAlloc_v2
        _F_cuGetProcAddress_v2('cuMemAlloc', <void **>&__cuMemAlloc_v2, 3020, ptds_mode, NULL)

        global __cuMemAllocPitch_v2
        _F_cuGetProcAddress_v2('cuMemAllocPitch', <void **>&__cuMemAllocPitch_v2, 3020, ptds_mode, NULL)

        global __cuMemFree_v2
        _F_cuGetProcAddress_v2('cuMemFree', <void **>&__cuMemFree_v2, 3020, ptds_mode, NULL)

        global __cuMemGetAddressRange_v2
        _F_cuGetProcAddress_v2('cuMemGetAddressRange', <void **>&__cuMemGetAddressRange_v2, 3020, ptds_mode, NULL)

        global __cuMemAllocHost_v2
        _F_cuGetProcAddress_v2('cuMemAllocHost', <void **>&__cuMemAllocHost_v2, 3020, ptds_mode, NULL)

        global __cuMemFreeHost
        _F_cuGetProcAddress_v2('cuMemFreeHost', <void **>&__cuMemFreeHost, 2000, ptds_mode, NULL)

        global __cuMemHostAlloc
        _F_cuGetProcAddress_v2('cuMemHostAlloc', <void **>&__cuMemHostAlloc, 2020, ptds_mode, NULL)

        global __cuMemHostGetDevicePointer_v2
        _F_cuGetProcAddress_v2('cuMemHostGetDevicePointer', <void **>&__cuMemHostGetDevicePointer_v2, 3020, ptds_mode, NULL)

        global __cuMemHostGetFlags
        _F_cuGetProcAddress_v2('cuMemHostGetFlags', <void **>&__cuMemHostGetFlags, 2030, ptds_mode, NULL)

        global __cuMemAllocManaged
        _F_cuGetProcAddress_v2('cuMemAllocManaged', <void **>&__cuMemAllocManaged, 6000, ptds_mode, NULL)

        global __cuDeviceRegisterAsyncNotification
        _F_cuGetProcAddress_v2('cuDeviceRegisterAsyncNotification', <void **>&__cuDeviceRegisterAsyncNotification, 12040, ptds_mode, NULL)

        global __cuDeviceUnregisterAsyncNotification
        _F_cuGetProcAddress_v2('cuDeviceUnregisterAsyncNotification', <void **>&__cuDeviceUnregisterAsyncNotification, 12040, ptds_mode, NULL)

        global __cuDeviceGetByPCIBusId
        _F_cuGetProcAddress_v2('cuDeviceGetByPCIBusId', <void **>&__cuDeviceGetByPCIBusId, 4010, ptds_mode, NULL)

        global __cuDeviceGetPCIBusId
        _F_cuGetProcAddress_v2('cuDeviceGetPCIBusId', <void **>&__cuDeviceGetPCIBusId, 4010, ptds_mode, NULL)

        global __cuIpcGetEventHandle
        _F_cuGetProcAddress_v2('cuIpcGetEventHandle', <void **>&__cuIpcGetEventHandle, 4010, ptds_mode, NULL)

        global __cuIpcOpenEventHandle
        _F_cuGetProcAddress_v2('cuIpcOpenEventHandle', <void **>&__cuIpcOpenEventHandle, 4010, ptds_mode, NULL)

        global __cuIpcGetMemHandle
        _F_cuGetProcAddress_v2('cuIpcGetMemHandle', <void **>&__cuIpcGetMemHandle, 4010, ptds_mode, NULL)

        global __cuIpcOpenMemHandle_v2
        _F_cuGetProcAddress_v2('cuIpcOpenMemHandle', <void **>&__cuIpcOpenMemHandle_v2, 11000, ptds_mode, NULL)

        global __cuIpcCloseMemHandle
        _F_cuGetProcAddress_v2('cuIpcCloseMemHandle', <void **>&__cuIpcCloseMemHandle, 4010, ptds_mode, NULL)

        global __cuMemHostRegister_v2
        _F_cuGetProcAddress_v2('cuMemHostRegister', <void **>&__cuMemHostRegister_v2, 6050, ptds_mode, NULL)

        global __cuMemHostUnregister
        _F_cuGetProcAddress_v2('cuMemHostUnregister', <void **>&__cuMemHostUnregister, 4000, ptds_mode, NULL)

        global __cuMemcpy
        _F_cuGetProcAddress_v2('cuMemcpy', <void **>&__cuMemcpy, 4000, ptds_mode, NULL)

        global __cuMemcpyPeer
        _F_cuGetProcAddress_v2('cuMemcpyPeer', <void **>&__cuMemcpyPeer, 4000, ptds_mode, NULL)

        global __cuMemcpyHtoD_v2
        _F_cuGetProcAddress_v2('cuMemcpyHtoD', <void **>&__cuMemcpyHtoD_v2, 3020, ptds_mode, NULL)

        global __cuMemcpyDtoH_v2
        _F_cuGetProcAddress_v2('cuMemcpyDtoH', <void **>&__cuMemcpyDtoH_v2, 3020, ptds_mode, NULL)

        global __cuMemcpyDtoD_v2
        _F_cuGetProcAddress_v2('cuMemcpyDtoD', <void **>&__cuMemcpyDtoD_v2, 3020, ptds_mode, NULL)

        global __cuMemcpyDtoA_v2
        _F_cuGetProcAddress_v2('cuMemcpyDtoA', <void **>&__cuMemcpyDtoA_v2, 3020, ptds_mode, NULL)

        global __cuMemcpyAtoD_v2
        _F_cuGetProcAddress_v2('cuMemcpyAtoD', <void **>&__cuMemcpyAtoD_v2, 3020, ptds_mode, NULL)

        global __cuMemcpyHtoA_v2
        _F_cuGetProcAddress_v2('cuMemcpyHtoA', <void **>&__cuMemcpyHtoA_v2, 3020, ptds_mode, NULL)

        global __cuMemcpyAtoH_v2
        _F_cuGetProcAddress_v2('cuMemcpyAtoH', <void **>&__cuMemcpyAtoH_v2, 3020, ptds_mode, NULL)

        global __cuMemcpyAtoA_v2
        _F_cuGetProcAddress_v2('cuMemcpyAtoA', <void **>&__cuMemcpyAtoA_v2, 3020, ptds_mode, NULL)

        global __cuMemcpy2D_v2
        _F_cuGetProcAddress_v2('cuMemcpy2D', <void **>&__cuMemcpy2D_v2, 3020, ptds_mode, NULL)

        global __cuMemcpy2DUnaligned_v2
        _F_cuGetProcAddress_v2('cuMemcpy2DUnaligned', <void **>&__cuMemcpy2DUnaligned_v2, 3020, ptds_mode, NULL)

        global __cuMemcpy3D_v2
        _F_cuGetProcAddress_v2('cuMemcpy3D', <void **>&__cuMemcpy3D_v2, 3020, ptds_mode, NULL)

        global __cuMemcpy3DPeer
        _F_cuGetProcAddress_v2('cuMemcpy3DPeer', <void **>&__cuMemcpy3DPeer, 4000, ptds_mode, NULL)

        global __cuMemcpyAsync
        _F_cuGetProcAddress_v2('cuMemcpyAsync', <void **>&__cuMemcpyAsync, 4000, ptds_mode, NULL)

        global __cuMemcpyPeerAsync
        _F_cuGetProcAddress_v2('cuMemcpyPeerAsync', <void **>&__cuMemcpyPeerAsync, 4000, ptds_mode, NULL)

        global __cuMemcpyHtoDAsync_v2
        _F_cuGetProcAddress_v2('cuMemcpyHtoDAsync', <void **>&__cuMemcpyHtoDAsync_v2, 3020, ptds_mode, NULL)

        global __cuMemcpyDtoHAsync_v2
        _F_cuGetProcAddress_v2('cuMemcpyDtoHAsync', <void **>&__cuMemcpyDtoHAsync_v2, 3020, ptds_mode, NULL)

        global __cuMemcpyDtoDAsync_v2
        _F_cuGetProcAddress_v2('cuMemcpyDtoDAsync', <void **>&__cuMemcpyDtoDAsync_v2, 3020, ptds_mode, NULL)

        global __cuMemcpyHtoAAsync_v2
        _F_cuGetProcAddress_v2('cuMemcpyHtoAAsync', <void **>&__cuMemcpyHtoAAsync_v2, 3020, ptds_mode, NULL)

        global __cuMemcpyAtoHAsync_v2
        _F_cuGetProcAddress_v2('cuMemcpyAtoHAsync', <void **>&__cuMemcpyAtoHAsync_v2, 3020, ptds_mode, NULL)

        global __cuMemcpy2DAsync_v2
        _F_cuGetProcAddress_v2('cuMemcpy2DAsync', <void **>&__cuMemcpy2DAsync_v2, 3020, ptds_mode, NULL)

        global __cuMemcpy3DAsync_v2
        _F_cuGetProcAddress_v2('cuMemcpy3DAsync', <void **>&__cuMemcpy3DAsync_v2, 3020, ptds_mode, NULL)

        global __cuMemcpy3DPeerAsync
        _F_cuGetProcAddress_v2('cuMemcpy3DPeerAsync', <void **>&__cuMemcpy3DPeerAsync, 4000, ptds_mode, NULL)

        global __cuMemsetD8_v2
        _F_cuGetProcAddress_v2('cuMemsetD8', <void **>&__cuMemsetD8_v2, 3020, ptds_mode, NULL)

        global __cuMemsetD16_v2
        _F_cuGetProcAddress_v2('cuMemsetD16', <void **>&__cuMemsetD16_v2, 3020, ptds_mode, NULL)

        global __cuMemsetD32_v2
        _F_cuGetProcAddress_v2('cuMemsetD32', <void **>&__cuMemsetD32_v2, 3020, ptds_mode, NULL)

        global __cuMemsetD2D8_v2
        _F_cuGetProcAddress_v2('cuMemsetD2D8', <void **>&__cuMemsetD2D8_v2, 3020, ptds_mode, NULL)

        global __cuMemsetD2D16_v2
        _F_cuGetProcAddress_v2('cuMemsetD2D16', <void **>&__cuMemsetD2D16_v2, 3020, ptds_mode, NULL)

        global __cuMemsetD2D32_v2
        _F_cuGetProcAddress_v2('cuMemsetD2D32', <void **>&__cuMemsetD2D32_v2, 3020, ptds_mode, NULL)

        global __cuMemsetD8Async
        _F_cuGetProcAddress_v2('cuMemsetD8Async', <void **>&__cuMemsetD8Async, 3020, ptds_mode, NULL)

        global __cuMemsetD16Async
        _F_cuGetProcAddress_v2('cuMemsetD16Async', <void **>&__cuMemsetD16Async, 3020, ptds_mode, NULL)

        global __cuMemsetD32Async
        _F_cuGetProcAddress_v2('cuMemsetD32Async', <void **>&__cuMemsetD32Async, 3020, ptds_mode, NULL)

        global __cuMemsetD2D8Async
        _F_cuGetProcAddress_v2('cuMemsetD2D8Async', <void **>&__cuMemsetD2D8Async, 3020, ptds_mode, NULL)

        global __cuMemsetD2D16Async
        _F_cuGetProcAddress_v2('cuMemsetD2D16Async', <void **>&__cuMemsetD2D16Async, 3020, ptds_mode, NULL)

        global __cuMemsetD2D32Async
        _F_cuGetProcAddress_v2('cuMemsetD2D32Async', <void **>&__cuMemsetD2D32Async, 3020, ptds_mode, NULL)

        global __cuArrayCreate_v2
        _F_cuGetProcAddress_v2('cuArrayCreate', <void **>&__cuArrayCreate_v2, 3020, ptds_mode, NULL)

        global __cuArrayGetDescriptor_v2
        _F_cuGetProcAddress_v2('cuArrayGetDescriptor', <void **>&__cuArrayGetDescriptor_v2, 3020, ptds_mode, NULL)

        global __cuArrayGetSparseProperties
        _F_cuGetProcAddress_v2('cuArrayGetSparseProperties', <void **>&__cuArrayGetSparseProperties, 11010, ptds_mode, NULL)

        global __cuMipmappedArrayGetSparseProperties
        _F_cuGetProcAddress_v2('cuMipmappedArrayGetSparseProperties', <void **>&__cuMipmappedArrayGetSparseProperties, 11010, ptds_mode, NULL)

        global __cuArrayGetMemoryRequirements
        _F_cuGetProcAddress_v2('cuArrayGetMemoryRequirements', <void **>&__cuArrayGetMemoryRequirements, 11060, ptds_mode, NULL)

        global __cuMipmappedArrayGetMemoryRequirements
        _F_cuGetProcAddress_v2('cuMipmappedArrayGetMemoryRequirements', <void **>&__cuMipmappedArrayGetMemoryRequirements, 11060, ptds_mode, NULL)

        global __cuArrayGetPlane
        _F_cuGetProcAddress_v2('cuArrayGetPlane', <void **>&__cuArrayGetPlane, 11020, ptds_mode, NULL)

        global __cuArrayDestroy
        _F_cuGetProcAddress_v2('cuArrayDestroy', <void **>&__cuArrayDestroy, 2000, ptds_mode, NULL)

        global __cuArray3DCreate_v2
        _F_cuGetProcAddress_v2('cuArray3DCreate', <void **>&__cuArray3DCreate_v2, 3020, ptds_mode, NULL)

        global __cuArray3DGetDescriptor_v2
        _F_cuGetProcAddress_v2('cuArray3DGetDescriptor', <void **>&__cuArray3DGetDescriptor_v2, 3020, ptds_mode, NULL)

        global __cuMipmappedArrayCreate
        _F_cuGetProcAddress_v2('cuMipmappedArrayCreate', <void **>&__cuMipmappedArrayCreate, 5000, ptds_mode, NULL)

        global __cuMipmappedArrayGetLevel
        _F_cuGetProcAddress_v2('cuMipmappedArrayGetLevel', <void **>&__cuMipmappedArrayGetLevel, 5000, ptds_mode, NULL)

        global __cuMipmappedArrayDestroy
        _F_cuGetProcAddress_v2('cuMipmappedArrayDestroy', <void **>&__cuMipmappedArrayDestroy, 5000, ptds_mode, NULL)

        global __cuMemGetHandleForAddressRange
        _F_cuGetProcAddress_v2('cuMemGetHandleForAddressRange', <void **>&__cuMemGetHandleForAddressRange, 11070, ptds_mode, NULL)

        global __cuMemBatchDecompressAsync
        _F_cuGetProcAddress_v2('cuMemBatchDecompressAsync', <void **>&__cuMemBatchDecompressAsync, 12060, ptds_mode, NULL)

        global __cuMemAddressReserve
        _F_cuGetProcAddress_v2('cuMemAddressReserve', <void **>&__cuMemAddressReserve, 10020, ptds_mode, NULL)

        global __cuMemAddressFree
        _F_cuGetProcAddress_v2('cuMemAddressFree', <void **>&__cuMemAddressFree, 10020, ptds_mode, NULL)

        global __cuMemCreate
        _F_cuGetProcAddress_v2('cuMemCreate', <void **>&__cuMemCreate, 10020, ptds_mode, NULL)

        global __cuMemRelease
        _F_cuGetProcAddress_v2('cuMemRelease', <void **>&__cuMemRelease, 10020, ptds_mode, NULL)

        global __cuMemMap
        _F_cuGetProcAddress_v2('cuMemMap', <void **>&__cuMemMap, 10020, ptds_mode, NULL)

        global __cuMemMapArrayAsync
        _F_cuGetProcAddress_v2('cuMemMapArrayAsync', <void **>&__cuMemMapArrayAsync, 11010, ptds_mode, NULL)

        global __cuMemUnmap
        _F_cuGetProcAddress_v2('cuMemUnmap', <void **>&__cuMemUnmap, 10020, ptds_mode, NULL)

        global __cuMemSetAccess
        _F_cuGetProcAddress_v2('cuMemSetAccess', <void **>&__cuMemSetAccess, 10020, ptds_mode, NULL)

        global __cuMemGetAccess
        _F_cuGetProcAddress_v2('cuMemGetAccess', <void **>&__cuMemGetAccess, 10020, ptds_mode, NULL)

        global __cuMemExportToShareableHandle
        _F_cuGetProcAddress_v2('cuMemExportToShareableHandle', <void **>&__cuMemExportToShareableHandle, 10020, ptds_mode, NULL)

        global __cuMemImportFromShareableHandle
        _F_cuGetProcAddress_v2('cuMemImportFromShareableHandle', <void **>&__cuMemImportFromShareableHandle, 10020, ptds_mode, NULL)

        global __cuMemGetAllocationGranularity
        _F_cuGetProcAddress_v2('cuMemGetAllocationGranularity', <void **>&__cuMemGetAllocationGranularity, 10020, ptds_mode, NULL)

        global __cuMemGetAllocationPropertiesFromHandle
        _F_cuGetProcAddress_v2('cuMemGetAllocationPropertiesFromHandle', <void **>&__cuMemGetAllocationPropertiesFromHandle, 10020, ptds_mode, NULL)

        global __cuMemRetainAllocationHandle
        _F_cuGetProcAddress_v2('cuMemRetainAllocationHandle', <void **>&__cuMemRetainAllocationHandle, 11000, ptds_mode, NULL)

        global __cuMemFreeAsync
        _F_cuGetProcAddress_v2('cuMemFreeAsync', <void **>&__cuMemFreeAsync, 11020, ptds_mode, NULL)

        global __cuMemAllocAsync
        _F_cuGetProcAddress_v2('cuMemAllocAsync', <void **>&__cuMemAllocAsync, 11020, ptds_mode, NULL)

        global __cuMemPoolTrimTo
        _F_cuGetProcAddress_v2('cuMemPoolTrimTo', <void **>&__cuMemPoolTrimTo, 11020, ptds_mode, NULL)

        global __cuMemPoolSetAttribute
        _F_cuGetProcAddress_v2('cuMemPoolSetAttribute', <void **>&__cuMemPoolSetAttribute, 11020, ptds_mode, NULL)

        global __cuMemPoolGetAttribute
        _F_cuGetProcAddress_v2('cuMemPoolGetAttribute', <void **>&__cuMemPoolGetAttribute, 11020, ptds_mode, NULL)

        global __cuMemPoolSetAccess
        _F_cuGetProcAddress_v2('cuMemPoolSetAccess', <void **>&__cuMemPoolSetAccess, 11020, ptds_mode, NULL)

        global __cuMemPoolGetAccess
        _F_cuGetProcAddress_v2('cuMemPoolGetAccess', <void **>&__cuMemPoolGetAccess, 11020, ptds_mode, NULL)

        global __cuMemPoolCreate
        _F_cuGetProcAddress_v2('cuMemPoolCreate', <void **>&__cuMemPoolCreate, 11020, ptds_mode, NULL)

        global __cuMemPoolDestroy
        _F_cuGetProcAddress_v2('cuMemPoolDestroy', <void **>&__cuMemPoolDestroy, 11020, ptds_mode, NULL)

        global __cuMemAllocFromPoolAsync
        _F_cuGetProcAddress_v2('cuMemAllocFromPoolAsync', <void **>&__cuMemAllocFromPoolAsync, 11020, ptds_mode, NULL)

        global __cuMemPoolExportToShareableHandle
        _F_cuGetProcAddress_v2('cuMemPoolExportToShareableHandle', <void **>&__cuMemPoolExportToShareableHandle, 11020, ptds_mode, NULL)

        global __cuMemPoolImportFromShareableHandle
        _F_cuGetProcAddress_v2('cuMemPoolImportFromShareableHandle', <void **>&__cuMemPoolImportFromShareableHandle, 11020, ptds_mode, NULL)

        global __cuMemPoolExportPointer
        _F_cuGetProcAddress_v2('cuMemPoolExportPointer', <void **>&__cuMemPoolExportPointer, 11020, ptds_mode, NULL)

        global __cuMemPoolImportPointer
        _F_cuGetProcAddress_v2('cuMemPoolImportPointer', <void **>&__cuMemPoolImportPointer, 11020, ptds_mode, NULL)

        global __cuMulticastCreate
        _F_cuGetProcAddress_v2('cuMulticastCreate', <void **>&__cuMulticastCreate, 12010, ptds_mode, NULL)

        global __cuMulticastAddDevice
        _F_cuGetProcAddress_v2('cuMulticastAddDevice', <void **>&__cuMulticastAddDevice, 12010, ptds_mode, NULL)

        global __cuMulticastBindMem
        _F_cuGetProcAddress_v2('cuMulticastBindMem', <void **>&__cuMulticastBindMem, 12010, ptds_mode, NULL)

        global __cuMulticastBindAddr
        _F_cuGetProcAddress_v2('cuMulticastBindAddr', <void **>&__cuMulticastBindAddr, 12010, ptds_mode, NULL)

        global __cuMulticastUnbind
        _F_cuGetProcAddress_v2('cuMulticastUnbind', <void **>&__cuMulticastUnbind, 12010, ptds_mode, NULL)

        global __cuMulticastGetGranularity
        _F_cuGetProcAddress_v2('cuMulticastGetGranularity', <void **>&__cuMulticastGetGranularity, 12010, ptds_mode, NULL)

        global __cuPointerGetAttribute
        _F_cuGetProcAddress_v2('cuPointerGetAttribute', <void **>&__cuPointerGetAttribute, 4000, ptds_mode, NULL)

        global __cuMemPrefetchAsync_v2
        _F_cuGetProcAddress_v2('cuMemPrefetchAsync', <void **>&__cuMemPrefetchAsync_v2, 12020, ptds_mode, NULL)

        global __cuMemAdvise_v2
        _F_cuGetProcAddress_v2('cuMemAdvise', <void **>&__cuMemAdvise_v2, 12020, ptds_mode, NULL)

        global __cuMemRangeGetAttribute
        _F_cuGetProcAddress_v2('cuMemRangeGetAttribute', <void **>&__cuMemRangeGetAttribute, 8000, ptds_mode, NULL)

        global __cuMemRangeGetAttributes
        _F_cuGetProcAddress_v2('cuMemRangeGetAttributes', <void **>&__cuMemRangeGetAttributes, 8000, ptds_mode, NULL)

        global __cuPointerSetAttribute
        _F_cuGetProcAddress_v2('cuPointerSetAttribute', <void **>&__cuPointerSetAttribute, 6000, ptds_mode, NULL)

        global __cuPointerGetAttributes
        _F_cuGetProcAddress_v2('cuPointerGetAttributes', <void **>&__cuPointerGetAttributes, 7000, ptds_mode, NULL)

        global __cuStreamCreate
        _F_cuGetProcAddress_v2('cuStreamCreate', <void **>&__cuStreamCreate, 2000, ptds_mode, NULL)

        global __cuStreamCreateWithPriority
        _F_cuGetProcAddress_v2('cuStreamCreateWithPriority', <void **>&__cuStreamCreateWithPriority, 5050, ptds_mode, NULL)

        global __cuStreamGetPriority
        _F_cuGetProcAddress_v2('cuStreamGetPriority', <void **>&__cuStreamGetPriority, 5050, ptds_mode, NULL)

        global __cuStreamGetDevice
        _F_cuGetProcAddress_v2('cuStreamGetDevice', <void **>&__cuStreamGetDevice, 12080, ptds_mode, NULL)

        global __cuStreamGetFlags
        _F_cuGetProcAddress_v2('cuStreamGetFlags', <void **>&__cuStreamGetFlags, 5050, ptds_mode, NULL)

        global __cuStreamGetId
        _F_cuGetProcAddress_v2('cuStreamGetId', <void **>&__cuStreamGetId, 12000, ptds_mode, NULL)

        global __cuStreamGetCtx
        _F_cuGetProcAddress_v2('cuStreamGetCtx', <void **>&__cuStreamGetCtx, 9020, ptds_mode, NULL)

        global __cuStreamGetCtx_v2
        _F_cuGetProcAddress_v2('cuStreamGetCtx_v2', <void **>&__cuStreamGetCtx_v2, 12050, ptds_mode, NULL)

        global __cuStreamWaitEvent
        _F_cuGetProcAddress_v2('cuStreamWaitEvent', <void **>&__cuStreamWaitEvent, 3020, ptds_mode, NULL)

        global __cuStreamAddCallback
        _F_cuGetProcAddress_v2('cuStreamAddCallback', <void **>&__cuStreamAddCallback, 5000, ptds_mode, NULL)

        global __cuStreamBeginCapture_v2
        _F_cuGetProcAddress_v2('cuStreamBeginCapture', <void **>&__cuStreamBeginCapture_v2, 10010, ptds_mode, NULL)

        global __cuStreamBeginCaptureToGraph
        _F_cuGetProcAddress_v2('cuStreamBeginCaptureToGraph', <void **>&__cuStreamBeginCaptureToGraph, 12030, ptds_mode, NULL)

        global __cuThreadExchangeStreamCaptureMode
        _F_cuGetProcAddress_v2('cuThreadExchangeStreamCaptureMode', <void **>&__cuThreadExchangeStreamCaptureMode, 10010, ptds_mode, NULL)

        global __cuStreamEndCapture
        _F_cuGetProcAddress_v2('cuStreamEndCapture', <void **>&__cuStreamEndCapture, 10000, ptds_mode, NULL)

        global __cuStreamIsCapturing
        _F_cuGetProcAddress_v2('cuStreamIsCapturing', <void **>&__cuStreamIsCapturing, 10000, ptds_mode, NULL)

        global __cuStreamGetCaptureInfo_v3
        _F_cuGetProcAddress_v2('cuStreamGetCaptureInfo', <void **>&__cuStreamGetCaptureInfo_v3, 12030, ptds_mode, NULL)

        global __cuStreamUpdateCaptureDependencies_v2
        _F_cuGetProcAddress_v2('cuStreamUpdateCaptureDependencies', <void **>&__cuStreamUpdateCaptureDependencies_v2, 12030, ptds_mode, NULL)

        global __cuStreamAttachMemAsync
        _F_cuGetProcAddress_v2('cuStreamAttachMemAsync', <void **>&__cuStreamAttachMemAsync, 6000, ptds_mode, NULL)

        global __cuStreamQuery
        _F_cuGetProcAddress_v2('cuStreamQuery', <void **>&__cuStreamQuery, 2000, ptds_mode, NULL)

        global __cuStreamSynchronize
        _F_cuGetProcAddress_v2('cuStreamSynchronize', <void **>&__cuStreamSynchronize, 2000, ptds_mode, NULL)

        global __cuStreamDestroy_v2
        _F_cuGetProcAddress_v2('cuStreamDestroy', <void **>&__cuStreamDestroy_v2, 4000, ptds_mode, NULL)

        global __cuStreamCopyAttributes
        _F_cuGetProcAddress_v2('cuStreamCopyAttributes', <void **>&__cuStreamCopyAttributes, 11000, ptds_mode, NULL)

        global __cuStreamGetAttribute
        _F_cuGetProcAddress_v2('cuStreamGetAttribute', <void **>&__cuStreamGetAttribute, 11000, ptds_mode, NULL)

        global __cuStreamSetAttribute
        _F_cuGetProcAddress_v2('cuStreamSetAttribute', <void **>&__cuStreamSetAttribute, 11000, ptds_mode, NULL)

        global __cuEventCreate
        _F_cuGetProcAddress_v2('cuEventCreate', <void **>&__cuEventCreate, 2000, ptds_mode, NULL)

        global __cuEventRecord
        _F_cuGetProcAddress_v2('cuEventRecord', <void **>&__cuEventRecord, 2000, ptds_mode, NULL)

        global __cuEventRecordWithFlags
        _F_cuGetProcAddress_v2('cuEventRecordWithFlags', <void **>&__cuEventRecordWithFlags, 11010, ptds_mode, NULL)

        global __cuEventQuery
        _F_cuGetProcAddress_v2('cuEventQuery', <void **>&__cuEventQuery, 2000, ptds_mode, NULL)

        global __cuEventSynchronize
        _F_cuGetProcAddress_v2('cuEventSynchronize', <void **>&__cuEventSynchronize, 2000, ptds_mode, NULL)

        global __cuEventDestroy_v2
        _F_cuGetProcAddress_v2('cuEventDestroy', <void **>&__cuEventDestroy_v2, 4000, ptds_mode, NULL)

        global __cuEventElapsedTime_v2
        _F_cuGetProcAddress_v2('cuEventElapsedTime', <void **>&__cuEventElapsedTime_v2, 12080, ptds_mode, NULL)

        global __cuImportExternalMemory
        _F_cuGetProcAddress_v2('cuImportExternalMemory', <void **>&__cuImportExternalMemory, 10000, ptds_mode, NULL)

        global __cuExternalMemoryGetMappedBuffer
        _F_cuGetProcAddress_v2('cuExternalMemoryGetMappedBuffer', <void **>&__cuExternalMemoryGetMappedBuffer, 10000, ptds_mode, NULL)

        global __cuExternalMemoryGetMappedMipmappedArray
        _F_cuGetProcAddress_v2('cuExternalMemoryGetMappedMipmappedArray', <void **>&__cuExternalMemoryGetMappedMipmappedArray, 10000, ptds_mode, NULL)

        global __cuDestroyExternalMemory
        _F_cuGetProcAddress_v2('cuDestroyExternalMemory', <void **>&__cuDestroyExternalMemory, 10000, ptds_mode, NULL)

        global __cuImportExternalSemaphore
        _F_cuGetProcAddress_v2('cuImportExternalSemaphore', <void **>&__cuImportExternalSemaphore, 10000, ptds_mode, NULL)

        global __cuSignalExternalSemaphoresAsync
        _F_cuGetProcAddress_v2('cuSignalExternalSemaphoresAsync', <void **>&__cuSignalExternalSemaphoresAsync, 10000, ptds_mode, NULL)

        global __cuWaitExternalSemaphoresAsync
        _F_cuGetProcAddress_v2('cuWaitExternalSemaphoresAsync', <void **>&__cuWaitExternalSemaphoresAsync, 10000, ptds_mode, NULL)

        global __cuDestroyExternalSemaphore
        _F_cuGetProcAddress_v2('cuDestroyExternalSemaphore', <void **>&__cuDestroyExternalSemaphore, 10000, ptds_mode, NULL)

        global __cuStreamWaitValue32_v2
        _F_cuGetProcAddress_v2('cuStreamWaitValue32', <void **>&__cuStreamWaitValue32_v2, 11070, ptds_mode, NULL)

        global __cuStreamWaitValue64_v2
        _F_cuGetProcAddress_v2('cuStreamWaitValue64', <void **>&__cuStreamWaitValue64_v2, 11070, ptds_mode, NULL)

        global __cuStreamWriteValue32_v2
        _F_cuGetProcAddress_v2('cuStreamWriteValue32', <void **>&__cuStreamWriteValue32_v2, 11070, ptds_mode, NULL)

        global __cuStreamWriteValue64_v2
        _F_cuGetProcAddress_v2('cuStreamWriteValue64', <void **>&__cuStreamWriteValue64_v2, 11070, ptds_mode, NULL)

        global __cuStreamBatchMemOp_v2
        _F_cuGetProcAddress_v2('cuStreamBatchMemOp', <void **>&__cuStreamBatchMemOp_v2, 11070, ptds_mode, NULL)

        global __cuFuncGetAttribute
        _F_cuGetProcAddress_v2('cuFuncGetAttribute', <void **>&__cuFuncGetAttribute, 2020, ptds_mode, NULL)

        global __cuFuncSetAttribute
        _F_cuGetProcAddress_v2('cuFuncSetAttribute', <void **>&__cuFuncSetAttribute, 9000, ptds_mode, NULL)

        global __cuFuncSetCacheConfig
        _F_cuGetProcAddress_v2('cuFuncSetCacheConfig', <void **>&__cuFuncSetCacheConfig, 3000, ptds_mode, NULL)

        global __cuFuncGetModule
        _F_cuGetProcAddress_v2('cuFuncGetModule', <void **>&__cuFuncGetModule, 11000, ptds_mode, NULL)

        global __cuFuncGetName
        _F_cuGetProcAddress_v2('cuFuncGetName', <void **>&__cuFuncGetName, 12030, ptds_mode, NULL)

        global __cuFuncGetParamInfo
        _F_cuGetProcAddress_v2('cuFuncGetParamInfo', <void **>&__cuFuncGetParamInfo, 12040, ptds_mode, NULL)

        global __cuFuncIsLoaded
        _F_cuGetProcAddress_v2('cuFuncIsLoaded', <void **>&__cuFuncIsLoaded, 12040, ptds_mode, NULL)

        global __cuFuncLoad
        _F_cuGetProcAddress_v2('cuFuncLoad', <void **>&__cuFuncLoad, 12040, ptds_mode, NULL)

        global __cuLaunchKernel
        _F_cuGetProcAddress_v2('cuLaunchKernel', <void **>&__cuLaunchKernel, 4000, ptds_mode, NULL)

        global __cuLaunchKernelEx
        _F_cuGetProcAddress_v2('cuLaunchKernelEx', <void **>&__cuLaunchKernelEx, 11060, ptds_mode, NULL)

        global __cuLaunchCooperativeKernel
        _F_cuGetProcAddress_v2('cuLaunchCooperativeKernel', <void **>&__cuLaunchCooperativeKernel, 9000, ptds_mode, NULL)

        global __cuLaunchCooperativeKernelMultiDevice
        _F_cuGetProcAddress_v2('cuLaunchCooperativeKernelMultiDevice', <void **>&__cuLaunchCooperativeKernelMultiDevice, 9000, ptds_mode, NULL)

        global __cuLaunchHostFunc
        _F_cuGetProcAddress_v2('cuLaunchHostFunc', <void **>&__cuLaunchHostFunc, 10000, ptds_mode, NULL)

        global __cuFuncSetBlockShape
        _F_cuGetProcAddress_v2('cuFuncSetBlockShape', <void **>&__cuFuncSetBlockShape, 2000, ptds_mode, NULL)

        global __cuFuncSetSharedSize
        _F_cuGetProcAddress_v2('cuFuncSetSharedSize', <void **>&__cuFuncSetSharedSize, 2000, ptds_mode, NULL)

        global __cuParamSetSize
        _F_cuGetProcAddress_v2('cuParamSetSize', <void **>&__cuParamSetSize, 2000, ptds_mode, NULL)

        global __cuParamSeti
        _F_cuGetProcAddress_v2('cuParamSeti', <void **>&__cuParamSeti, 2000, ptds_mode, NULL)

        global __cuParamSetf
        _F_cuGetProcAddress_v2('cuParamSetf', <void **>&__cuParamSetf, 2000, ptds_mode, NULL)

        global __cuParamSetv
        _F_cuGetProcAddress_v2('cuParamSetv', <void **>&__cuParamSetv, 2000, ptds_mode, NULL)

        global __cuLaunch
        _F_cuGetProcAddress_v2('cuLaunch', <void **>&__cuLaunch, 2000, ptds_mode, NULL)

        global __cuLaunchGrid
        _F_cuGetProcAddress_v2('cuLaunchGrid', <void **>&__cuLaunchGrid, 2000, ptds_mode, NULL)

        global __cuLaunchGridAsync
        _F_cuGetProcAddress_v2('cuLaunchGridAsync', <void **>&__cuLaunchGridAsync, 2000, ptds_mode, NULL)

        global __cuParamSetTexRef
        _F_cuGetProcAddress_v2('cuParamSetTexRef', <void **>&__cuParamSetTexRef, 2000, ptds_mode, NULL)

        global __cuFuncSetSharedMemConfig
        _F_cuGetProcAddress_v2('cuFuncSetSharedMemConfig', <void **>&__cuFuncSetSharedMemConfig, 4020, ptds_mode, NULL)

        global __cuGraphCreate
        _F_cuGetProcAddress_v2('cuGraphCreate', <void **>&__cuGraphCreate, 10000, ptds_mode, NULL)

        global __cuGraphAddKernelNode_v2
        _F_cuGetProcAddress_v2('cuGraphAddKernelNode', <void **>&__cuGraphAddKernelNode_v2, 12000, ptds_mode, NULL)

        global __cuGraphKernelNodeGetParams_v2
        _F_cuGetProcAddress_v2('cuGraphKernelNodeGetParams', <void **>&__cuGraphKernelNodeGetParams_v2, 12000, ptds_mode, NULL)

        global __cuGraphKernelNodeSetParams_v2
        _F_cuGetProcAddress_v2('cuGraphKernelNodeSetParams', <void **>&__cuGraphKernelNodeSetParams_v2, 12000, ptds_mode, NULL)

        global __cuGraphAddMemcpyNode
        _F_cuGetProcAddress_v2('cuGraphAddMemcpyNode', <void **>&__cuGraphAddMemcpyNode, 10000, ptds_mode, NULL)

        global __cuGraphMemcpyNodeGetParams
        _F_cuGetProcAddress_v2('cuGraphMemcpyNodeGetParams', <void **>&__cuGraphMemcpyNodeGetParams, 10000, ptds_mode, NULL)

        global __cuGraphMemcpyNodeSetParams
        _F_cuGetProcAddress_v2('cuGraphMemcpyNodeSetParams', <void **>&__cuGraphMemcpyNodeSetParams, 10000, ptds_mode, NULL)

        global __cuGraphAddMemsetNode
        _F_cuGetProcAddress_v2('cuGraphAddMemsetNode', <void **>&__cuGraphAddMemsetNode, 10000, ptds_mode, NULL)

        global __cuGraphMemsetNodeGetParams
        _F_cuGetProcAddress_v2('cuGraphMemsetNodeGetParams', <void **>&__cuGraphMemsetNodeGetParams, 10000, ptds_mode, NULL)

        global __cuGraphMemsetNodeSetParams
        _F_cuGetProcAddress_v2('cuGraphMemsetNodeSetParams', <void **>&__cuGraphMemsetNodeSetParams, 10000, ptds_mode, NULL)

        global __cuGraphAddHostNode
        _F_cuGetProcAddress_v2('cuGraphAddHostNode', <void **>&__cuGraphAddHostNode, 10000, ptds_mode, NULL)

        global __cuGraphHostNodeGetParams
        _F_cuGetProcAddress_v2('cuGraphHostNodeGetParams', <void **>&__cuGraphHostNodeGetParams, 10000, ptds_mode, NULL)

        global __cuGraphHostNodeSetParams
        _F_cuGetProcAddress_v2('cuGraphHostNodeSetParams', <void **>&__cuGraphHostNodeSetParams, 10000, ptds_mode, NULL)

        global __cuGraphAddChildGraphNode
        _F_cuGetProcAddress_v2('cuGraphAddChildGraphNode', <void **>&__cuGraphAddChildGraphNode, 10000, ptds_mode, NULL)

        global __cuGraphChildGraphNodeGetGraph
        _F_cuGetProcAddress_v2('cuGraphChildGraphNodeGetGraph', <void **>&__cuGraphChildGraphNodeGetGraph, 10000, ptds_mode, NULL)

        global __cuGraphAddEmptyNode
        _F_cuGetProcAddress_v2('cuGraphAddEmptyNode', <void **>&__cuGraphAddEmptyNode, 10000, ptds_mode, NULL)

        global __cuGraphAddEventRecordNode
        _F_cuGetProcAddress_v2('cuGraphAddEventRecordNode', <void **>&__cuGraphAddEventRecordNode, 11010, ptds_mode, NULL)

        global __cuGraphEventRecordNodeGetEvent
        _F_cuGetProcAddress_v2('cuGraphEventRecordNodeGetEvent', <void **>&__cuGraphEventRecordNodeGetEvent, 11010, ptds_mode, NULL)

        global __cuGraphEventRecordNodeSetEvent
        _F_cuGetProcAddress_v2('cuGraphEventRecordNodeSetEvent', <void **>&__cuGraphEventRecordNodeSetEvent, 11010, ptds_mode, NULL)

        global __cuGraphAddEventWaitNode
        _F_cuGetProcAddress_v2('cuGraphAddEventWaitNode', <void **>&__cuGraphAddEventWaitNode, 11010, ptds_mode, NULL)

        global __cuGraphEventWaitNodeGetEvent
        _F_cuGetProcAddress_v2('cuGraphEventWaitNodeGetEvent', <void **>&__cuGraphEventWaitNodeGetEvent, 11010, ptds_mode, NULL)

        global __cuGraphEventWaitNodeSetEvent
        _F_cuGetProcAddress_v2('cuGraphEventWaitNodeSetEvent', <void **>&__cuGraphEventWaitNodeSetEvent, 11010, ptds_mode, NULL)

        global __cuGraphAddExternalSemaphoresSignalNode
        _F_cuGetProcAddress_v2('cuGraphAddExternalSemaphoresSignalNode', <void **>&__cuGraphAddExternalSemaphoresSignalNode, 11020, ptds_mode, NULL)

        global __cuGraphExternalSemaphoresSignalNodeGetParams
        _F_cuGetProcAddress_v2('cuGraphExternalSemaphoresSignalNodeGetParams', <void **>&__cuGraphExternalSemaphoresSignalNodeGetParams, 11020, ptds_mode, NULL)

        global __cuGraphExternalSemaphoresSignalNodeSetParams
        _F_cuGetProcAddress_v2('cuGraphExternalSemaphoresSignalNodeSetParams', <void **>&__cuGraphExternalSemaphoresSignalNodeSetParams, 11020, ptds_mode, NULL)

        global __cuGraphAddExternalSemaphoresWaitNode
        _F_cuGetProcAddress_v2('cuGraphAddExternalSemaphoresWaitNode', <void **>&__cuGraphAddExternalSemaphoresWaitNode, 11020, ptds_mode, NULL)

        global __cuGraphExternalSemaphoresWaitNodeGetParams
        _F_cuGetProcAddress_v2('cuGraphExternalSemaphoresWaitNodeGetParams', <void **>&__cuGraphExternalSemaphoresWaitNodeGetParams, 11020, ptds_mode, NULL)

        global __cuGraphExternalSemaphoresWaitNodeSetParams
        _F_cuGetProcAddress_v2('cuGraphExternalSemaphoresWaitNodeSetParams', <void **>&__cuGraphExternalSemaphoresWaitNodeSetParams, 11020, ptds_mode, NULL)

        global __cuGraphAddBatchMemOpNode
        _F_cuGetProcAddress_v2('cuGraphAddBatchMemOpNode', <void **>&__cuGraphAddBatchMemOpNode, 11070, ptds_mode, NULL)

        global __cuGraphBatchMemOpNodeGetParams
        _F_cuGetProcAddress_v2('cuGraphBatchMemOpNodeGetParams', <void **>&__cuGraphBatchMemOpNodeGetParams, 11070, ptds_mode, NULL)

        global __cuGraphBatchMemOpNodeSetParams
        _F_cuGetProcAddress_v2('cuGraphBatchMemOpNodeSetParams', <void **>&__cuGraphBatchMemOpNodeSetParams, 11070, ptds_mode, NULL)

        global __cuGraphExecBatchMemOpNodeSetParams
        _F_cuGetProcAddress_v2('cuGraphExecBatchMemOpNodeSetParams', <void **>&__cuGraphExecBatchMemOpNodeSetParams, 11070, ptds_mode, NULL)

        global __cuGraphAddMemAllocNode
        _F_cuGetProcAddress_v2('cuGraphAddMemAllocNode', <void **>&__cuGraphAddMemAllocNode, 11040, ptds_mode, NULL)

        global __cuGraphMemAllocNodeGetParams
        _F_cuGetProcAddress_v2('cuGraphMemAllocNodeGetParams', <void **>&__cuGraphMemAllocNodeGetParams, 11040, ptds_mode, NULL)

        global __cuGraphAddMemFreeNode
        _F_cuGetProcAddress_v2('cuGraphAddMemFreeNode', <void **>&__cuGraphAddMemFreeNode, 11040, ptds_mode, NULL)

        global __cuGraphMemFreeNodeGetParams
        _F_cuGetProcAddress_v2('cuGraphMemFreeNodeGetParams', <void **>&__cuGraphMemFreeNodeGetParams, 11040, ptds_mode, NULL)

        global __cuDeviceGraphMemTrim
        _F_cuGetProcAddress_v2('cuDeviceGraphMemTrim', <void **>&__cuDeviceGraphMemTrim, 11040, ptds_mode, NULL)

        global __cuDeviceGetGraphMemAttribute
        _F_cuGetProcAddress_v2('cuDeviceGetGraphMemAttribute', <void **>&__cuDeviceGetGraphMemAttribute, 11040, ptds_mode, NULL)

        global __cuDeviceSetGraphMemAttribute
        _F_cuGetProcAddress_v2('cuDeviceSetGraphMemAttribute', <void **>&__cuDeviceSetGraphMemAttribute, 11040, ptds_mode, NULL)

        global __cuGraphClone
        _F_cuGetProcAddress_v2('cuGraphClone', <void **>&__cuGraphClone, 10000, ptds_mode, NULL)

        global __cuGraphNodeFindInClone
        _F_cuGetProcAddress_v2('cuGraphNodeFindInClone', <void **>&__cuGraphNodeFindInClone, 10000, ptds_mode, NULL)

        global __cuGraphNodeGetType
        _F_cuGetProcAddress_v2('cuGraphNodeGetType', <void **>&__cuGraphNodeGetType, 10000, ptds_mode, NULL)

        global __cuGraphGetNodes
        _F_cuGetProcAddress_v2('cuGraphGetNodes', <void **>&__cuGraphGetNodes, 10000, ptds_mode, NULL)

        global __cuGraphGetRootNodes
        _F_cuGetProcAddress_v2('cuGraphGetRootNodes', <void **>&__cuGraphGetRootNodes, 10000, ptds_mode, NULL)

        global __cuGraphGetEdges_v2
        _F_cuGetProcAddress_v2('cuGraphGetEdges', <void **>&__cuGraphGetEdges_v2, 12030, ptds_mode, NULL)

        global __cuGraphNodeGetDependencies_v2
        _F_cuGetProcAddress_v2('cuGraphNodeGetDependencies', <void **>&__cuGraphNodeGetDependencies_v2, 12030, ptds_mode, NULL)

        global __cuGraphNodeGetDependentNodes_v2
        _F_cuGetProcAddress_v2('cuGraphNodeGetDependentNodes', <void **>&__cuGraphNodeGetDependentNodes_v2, 12030, ptds_mode, NULL)

        global __cuGraphAddDependencies_v2
        _F_cuGetProcAddress_v2('cuGraphAddDependencies', <void **>&__cuGraphAddDependencies_v2, 12030, ptds_mode, NULL)

        global __cuGraphRemoveDependencies_v2
        _F_cuGetProcAddress_v2('cuGraphRemoveDependencies', <void **>&__cuGraphRemoveDependencies_v2, 12030, ptds_mode, NULL)

        global __cuGraphDestroyNode
        _F_cuGetProcAddress_v2('cuGraphDestroyNode', <void **>&__cuGraphDestroyNode, 10000, ptds_mode, NULL)

        global __cuGraphInstantiateWithFlags
        _F_cuGetProcAddress_v2('cuGraphInstantiate', <void **>&__cuGraphInstantiateWithFlags, 11040, ptds_mode, NULL)

        global __cuGraphInstantiateWithParams
        _F_cuGetProcAddress_v2('cuGraphInstantiateWithParams', <void **>&__cuGraphInstantiateWithParams, 12000, ptds_mode, NULL)

        global __cuGraphExecGetFlags
        _F_cuGetProcAddress_v2('cuGraphExecGetFlags', <void **>&__cuGraphExecGetFlags, 12000, ptds_mode, NULL)

        global __cuGraphExecKernelNodeSetParams_v2
        _F_cuGetProcAddress_v2('cuGraphExecKernelNodeSetParams', <void **>&__cuGraphExecKernelNodeSetParams_v2, 12000, ptds_mode, NULL)

        global __cuGraphExecMemcpyNodeSetParams
        _F_cuGetProcAddress_v2('cuGraphExecMemcpyNodeSetParams', <void **>&__cuGraphExecMemcpyNodeSetParams, 10020, ptds_mode, NULL)

        global __cuGraphExecMemsetNodeSetParams
        _F_cuGetProcAddress_v2('cuGraphExecMemsetNodeSetParams', <void **>&__cuGraphExecMemsetNodeSetParams, 10020, ptds_mode, NULL)

        global __cuGraphExecHostNodeSetParams
        _F_cuGetProcAddress_v2('cuGraphExecHostNodeSetParams', <void **>&__cuGraphExecHostNodeSetParams, 10020, ptds_mode, NULL)

        global __cuGraphExecChildGraphNodeSetParams
        _F_cuGetProcAddress_v2('cuGraphExecChildGraphNodeSetParams', <void **>&__cuGraphExecChildGraphNodeSetParams, 11010, ptds_mode, NULL)

        global __cuGraphExecEventRecordNodeSetEvent
        _F_cuGetProcAddress_v2('cuGraphExecEventRecordNodeSetEvent', <void **>&__cuGraphExecEventRecordNodeSetEvent, 11010, ptds_mode, NULL)

        global __cuGraphExecEventWaitNodeSetEvent
        _F_cuGetProcAddress_v2('cuGraphExecEventWaitNodeSetEvent', <void **>&__cuGraphExecEventWaitNodeSetEvent, 11010, ptds_mode, NULL)

        global __cuGraphExecExternalSemaphoresSignalNodeSetParams
        _F_cuGetProcAddress_v2('cuGraphExecExternalSemaphoresSignalNodeSetParams', <void **>&__cuGraphExecExternalSemaphoresSignalNodeSetParams, 11020, ptds_mode, NULL)

        global __cuGraphExecExternalSemaphoresWaitNodeSetParams
        _F_cuGetProcAddress_v2('cuGraphExecExternalSemaphoresWaitNodeSetParams', <void **>&__cuGraphExecExternalSemaphoresWaitNodeSetParams, 11020, ptds_mode, NULL)

        global __cuGraphNodeSetEnabled
        _F_cuGetProcAddress_v2('cuGraphNodeSetEnabled', <void **>&__cuGraphNodeSetEnabled, 11060, ptds_mode, NULL)

        global __cuGraphNodeGetEnabled
        _F_cuGetProcAddress_v2('cuGraphNodeGetEnabled', <void **>&__cuGraphNodeGetEnabled, 11060, ptds_mode, NULL)

        global __cuGraphUpload
        _F_cuGetProcAddress_v2('cuGraphUpload', <void **>&__cuGraphUpload, 11010, ptds_mode, NULL)

        global __cuGraphLaunch
        _F_cuGetProcAddress_v2('cuGraphLaunch', <void **>&__cuGraphLaunch, 10000, ptds_mode, NULL)

        global __cuGraphExecDestroy
        _F_cuGetProcAddress_v2('cuGraphExecDestroy', <void **>&__cuGraphExecDestroy, 10000, ptds_mode, NULL)

        global __cuGraphDestroy
        _F_cuGetProcAddress_v2('cuGraphDestroy', <void **>&__cuGraphDestroy, 10000, ptds_mode, NULL)

        global __cuGraphExecUpdate_v2
        _F_cuGetProcAddress_v2('cuGraphExecUpdate', <void **>&__cuGraphExecUpdate_v2, 12000, ptds_mode, NULL)

        global __cuGraphKernelNodeCopyAttributes
        _F_cuGetProcAddress_v2('cuGraphKernelNodeCopyAttributes', <void **>&__cuGraphKernelNodeCopyAttributes, 11000, ptds_mode, NULL)

        global __cuGraphKernelNodeGetAttribute
        _F_cuGetProcAddress_v2('cuGraphKernelNodeGetAttribute', <void **>&__cuGraphKernelNodeGetAttribute, 11000, ptds_mode, NULL)

        global __cuGraphKernelNodeSetAttribute
        _F_cuGetProcAddress_v2('cuGraphKernelNodeSetAttribute', <void **>&__cuGraphKernelNodeSetAttribute, 11000, ptds_mode, NULL)

        global __cuGraphDebugDotPrint
        _F_cuGetProcAddress_v2('cuGraphDebugDotPrint', <void **>&__cuGraphDebugDotPrint, 11030, ptds_mode, NULL)

        global __cuUserObjectCreate
        _F_cuGetProcAddress_v2('cuUserObjectCreate', <void **>&__cuUserObjectCreate, 11030, ptds_mode, NULL)

        global __cuUserObjectRetain
        _F_cuGetProcAddress_v2('cuUserObjectRetain', <void **>&__cuUserObjectRetain, 11030, ptds_mode, NULL)

        global __cuUserObjectRelease
        _F_cuGetProcAddress_v2('cuUserObjectRelease', <void **>&__cuUserObjectRelease, 11030, ptds_mode, NULL)

        global __cuGraphRetainUserObject
        _F_cuGetProcAddress_v2('cuGraphRetainUserObject', <void **>&__cuGraphRetainUserObject, 11030, ptds_mode, NULL)

        global __cuGraphReleaseUserObject
        _F_cuGetProcAddress_v2('cuGraphReleaseUserObject', <void **>&__cuGraphReleaseUserObject, 11030, ptds_mode, NULL)

        global __cuGraphAddNode_v2
        _F_cuGetProcAddress_v2('cuGraphAddNode', <void **>&__cuGraphAddNode_v2, 12030, ptds_mode, NULL)

        global __cuGraphNodeSetParams
        _F_cuGetProcAddress_v2('cuGraphNodeSetParams', <void **>&__cuGraphNodeSetParams, 12020, ptds_mode, NULL)

        global __cuGraphExecNodeSetParams
        _F_cuGetProcAddress_v2('cuGraphExecNodeSetParams', <void **>&__cuGraphExecNodeSetParams, 12020, ptds_mode, NULL)

        global __cuGraphConditionalHandleCreate
        _F_cuGetProcAddress_v2('cuGraphConditionalHandleCreate', <void **>&__cuGraphConditionalHandleCreate, 12030, ptds_mode, NULL)

        global __cuOccupancyMaxActiveBlocksPerMultiprocessor
        _F_cuGetProcAddress_v2('cuOccupancyMaxActiveBlocksPerMultiprocessor', <void **>&__cuOccupancyMaxActiveBlocksPerMultiprocessor, 6050, ptds_mode, NULL)

        global __cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags
        _F_cuGetProcAddress_v2('cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags', <void **>&__cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags, 7000, ptds_mode, NULL)

        global __cuOccupancyMaxPotentialBlockSize
        _F_cuGetProcAddress_v2('cuOccupancyMaxPotentialBlockSize', <void **>&__cuOccupancyMaxPotentialBlockSize, 6050, ptds_mode, NULL)

        global __cuOccupancyMaxPotentialBlockSizeWithFlags
        _F_cuGetProcAddress_v2('cuOccupancyMaxPotentialBlockSizeWithFlags', <void **>&__cuOccupancyMaxPotentialBlockSizeWithFlags, 7000, ptds_mode, NULL)

        global __cuOccupancyAvailableDynamicSMemPerBlock
        _F_cuGetProcAddress_v2('cuOccupancyAvailableDynamicSMemPerBlock', <void **>&__cuOccupancyAvailableDynamicSMemPerBlock, 10020, ptds_mode, NULL)

        global __cuOccupancyMaxPotentialClusterSize
        _F_cuGetProcAddress_v2('cuOccupancyMaxPotentialClusterSize', <void **>&__cuOccupancyMaxPotentialClusterSize, 11070, ptds_mode, NULL)

        global __cuOccupancyMaxActiveClusters
        _F_cuGetProcAddress_v2('cuOccupancyMaxActiveClusters', <void **>&__cuOccupancyMaxActiveClusters, 11070, ptds_mode, NULL)

        global __cuTexRefSetArray
        _F_cuGetProcAddress_v2('cuTexRefSetArray', <void **>&__cuTexRefSetArray, 2000, ptds_mode, NULL)

        global __cuTexRefSetMipmappedArray
        _F_cuGetProcAddress_v2('cuTexRefSetMipmappedArray', <void **>&__cuTexRefSetMipmappedArray, 5000, ptds_mode, NULL)

        global __cuTexRefSetAddress_v2
        _F_cuGetProcAddress_v2('cuTexRefSetAddress', <void **>&__cuTexRefSetAddress_v2, 3020, ptds_mode, NULL)

        global __cuTexRefSetAddress2D_v3
        _F_cuGetProcAddress_v2('cuTexRefSetAddress2D', <void **>&__cuTexRefSetAddress2D_v3, 4010, ptds_mode, NULL)

        global __cuTexRefSetFormat
        _F_cuGetProcAddress_v2('cuTexRefSetFormat', <void **>&__cuTexRefSetFormat, 2000, ptds_mode, NULL)

        global __cuTexRefSetAddressMode
        _F_cuGetProcAddress_v2('cuTexRefSetAddressMode', <void **>&__cuTexRefSetAddressMode, 2000, ptds_mode, NULL)

        global __cuTexRefSetFilterMode
        _F_cuGetProcAddress_v2('cuTexRefSetFilterMode', <void **>&__cuTexRefSetFilterMode, 2000, ptds_mode, NULL)

        global __cuTexRefSetMipmapFilterMode
        _F_cuGetProcAddress_v2('cuTexRefSetMipmapFilterMode', <void **>&__cuTexRefSetMipmapFilterMode, 5000, ptds_mode, NULL)

        global __cuTexRefSetMipmapLevelBias
        _F_cuGetProcAddress_v2('cuTexRefSetMipmapLevelBias', <void **>&__cuTexRefSetMipmapLevelBias, 5000, ptds_mode, NULL)

        global __cuTexRefSetMipmapLevelClamp
        _F_cuGetProcAddress_v2('cuTexRefSetMipmapLevelClamp', <void **>&__cuTexRefSetMipmapLevelClamp, 5000, ptds_mode, NULL)

        global __cuTexRefSetMaxAnisotropy
        _F_cuGetProcAddress_v2('cuTexRefSetMaxAnisotropy', <void **>&__cuTexRefSetMaxAnisotropy, 5000, ptds_mode, NULL)

        global __cuTexRefSetBorderColor
        _F_cuGetProcAddress_v2('cuTexRefSetBorderColor', <void **>&__cuTexRefSetBorderColor, 8000, ptds_mode, NULL)

        global __cuTexRefSetFlags
        _F_cuGetProcAddress_v2('cuTexRefSetFlags', <void **>&__cuTexRefSetFlags, 2000, ptds_mode, NULL)

        global __cuTexRefGetAddress_v2
        _F_cuGetProcAddress_v2('cuTexRefGetAddress', <void **>&__cuTexRefGetAddress_v2, 3020, ptds_mode, NULL)

        global __cuTexRefGetArray
        _F_cuGetProcAddress_v2('cuTexRefGetArray', <void **>&__cuTexRefGetArray, 2000, ptds_mode, NULL)

        global __cuTexRefGetMipmappedArray
        _F_cuGetProcAddress_v2('cuTexRefGetMipmappedArray', <void **>&__cuTexRefGetMipmappedArray, 5000, ptds_mode, NULL)

        global __cuTexRefGetAddressMode
        _F_cuGetProcAddress_v2('cuTexRefGetAddressMode', <void **>&__cuTexRefGetAddressMode, 2000, ptds_mode, NULL)

        global __cuTexRefGetFilterMode
        _F_cuGetProcAddress_v2('cuTexRefGetFilterMode', <void **>&__cuTexRefGetFilterMode, 2000, ptds_mode, NULL)

        global __cuTexRefGetFormat
        _F_cuGetProcAddress_v2('cuTexRefGetFormat', <void **>&__cuTexRefGetFormat, 2000, ptds_mode, NULL)

        global __cuTexRefGetMipmapFilterMode
        _F_cuGetProcAddress_v2('cuTexRefGetMipmapFilterMode', <void **>&__cuTexRefGetMipmapFilterMode, 5000, ptds_mode, NULL)

        global __cuTexRefGetMipmapLevelBias
        _F_cuGetProcAddress_v2('cuTexRefGetMipmapLevelBias', <void **>&__cuTexRefGetMipmapLevelBias, 5000, ptds_mode, NULL)

        global __cuTexRefGetMipmapLevelClamp
        _F_cuGetProcAddress_v2('cuTexRefGetMipmapLevelClamp', <void **>&__cuTexRefGetMipmapLevelClamp, 5000, ptds_mode, NULL)

        global __cuTexRefGetMaxAnisotropy
        _F_cuGetProcAddress_v2('cuTexRefGetMaxAnisotropy', <void **>&__cuTexRefGetMaxAnisotropy, 5000, ptds_mode, NULL)

        global __cuTexRefGetBorderColor
        _F_cuGetProcAddress_v2('cuTexRefGetBorderColor', <void **>&__cuTexRefGetBorderColor, 8000, ptds_mode, NULL)

        global __cuTexRefGetFlags
        _F_cuGetProcAddress_v2('cuTexRefGetFlags', <void **>&__cuTexRefGetFlags, 2000, ptds_mode, NULL)

        global __cuTexRefCreate
        _F_cuGetProcAddress_v2('cuTexRefCreate', <void **>&__cuTexRefCreate, 2000, ptds_mode, NULL)

        global __cuTexRefDestroy
        _F_cuGetProcAddress_v2('cuTexRefDestroy', <void **>&__cuTexRefDestroy, 2000, ptds_mode, NULL)

        global __cuSurfRefSetArray
        _F_cuGetProcAddress_v2('cuSurfRefSetArray', <void **>&__cuSurfRefSetArray, 3000, ptds_mode, NULL)

        global __cuSurfRefGetArray
        _F_cuGetProcAddress_v2('cuSurfRefGetArray', <void **>&__cuSurfRefGetArray, 3000, ptds_mode, NULL)

        global __cuTexObjectCreate
        _F_cuGetProcAddress_v2('cuTexObjectCreate', <void **>&__cuTexObjectCreate, 5000, ptds_mode, NULL)

        global __cuTexObjectDestroy
        _F_cuGetProcAddress_v2('cuTexObjectDestroy', <void **>&__cuTexObjectDestroy, 5000, ptds_mode, NULL)

        global __cuTexObjectGetResourceDesc
        _F_cuGetProcAddress_v2('cuTexObjectGetResourceDesc', <void **>&__cuTexObjectGetResourceDesc, 5000, ptds_mode, NULL)

        global __cuTexObjectGetTextureDesc
        _F_cuGetProcAddress_v2('cuTexObjectGetTextureDesc', <void **>&__cuTexObjectGetTextureDesc, 5000, ptds_mode, NULL)

        global __cuTexObjectGetResourceViewDesc
        _F_cuGetProcAddress_v2('cuTexObjectGetResourceViewDesc', <void **>&__cuTexObjectGetResourceViewDesc, 5000, ptds_mode, NULL)

        global __cuSurfObjectCreate
        _F_cuGetProcAddress_v2('cuSurfObjectCreate', <void **>&__cuSurfObjectCreate, 5000, ptds_mode, NULL)

        global __cuSurfObjectDestroy
        _F_cuGetProcAddress_v2('cuSurfObjectDestroy', <void **>&__cuSurfObjectDestroy, 5000, ptds_mode, NULL)

        global __cuSurfObjectGetResourceDesc
        _F_cuGetProcAddress_v2('cuSurfObjectGetResourceDesc', <void **>&__cuSurfObjectGetResourceDesc, 5000, ptds_mode, NULL)

        global __cuTensorMapEncodeTiled
        _F_cuGetProcAddress_v2('cuTensorMapEncodeTiled', <void **>&__cuTensorMapEncodeTiled, 12000, ptds_mode, NULL)

        global __cuTensorMapEncodeIm2col
        _F_cuGetProcAddress_v2('cuTensorMapEncodeIm2col', <void **>&__cuTensorMapEncodeIm2col, 12000, ptds_mode, NULL)

        global __cuTensorMapEncodeIm2colWide
        _F_cuGetProcAddress_v2('cuTensorMapEncodeIm2colWide', <void **>&__cuTensorMapEncodeIm2colWide, 12080, ptds_mode, NULL)

        global __cuTensorMapReplaceAddress
        _F_cuGetProcAddress_v2('cuTensorMapReplaceAddress', <void **>&__cuTensorMapReplaceAddress, 12000, ptds_mode, NULL)

        global __cuDeviceCanAccessPeer
        _F_cuGetProcAddress_v2('cuDeviceCanAccessPeer', <void **>&__cuDeviceCanAccessPeer, 4000, ptds_mode, NULL)

        global __cuCtxEnablePeerAccess
        _F_cuGetProcAddress_v2('cuCtxEnablePeerAccess', <void **>&__cuCtxEnablePeerAccess, 4000, ptds_mode, NULL)

        global __cuCtxDisablePeerAccess
        _F_cuGetProcAddress_v2('cuCtxDisablePeerAccess', <void **>&__cuCtxDisablePeerAccess, 4000, ptds_mode, NULL)

        global __cuDeviceGetP2PAttribute
        _F_cuGetProcAddress_v2('cuDeviceGetP2PAttribute', <void **>&__cuDeviceGetP2PAttribute, 8000, ptds_mode, NULL)

        global __cuGraphicsUnregisterResource
        _F_cuGetProcAddress_v2('cuGraphicsUnregisterResource', <void **>&__cuGraphicsUnregisterResource, 3000, ptds_mode, NULL)

        global __cuGraphicsSubResourceGetMappedArray
        _F_cuGetProcAddress_v2('cuGraphicsSubResourceGetMappedArray', <void **>&__cuGraphicsSubResourceGetMappedArray, 3000, ptds_mode, NULL)

        global __cuGraphicsResourceGetMappedMipmappedArray
        _F_cuGetProcAddress_v2('cuGraphicsResourceGetMappedMipmappedArray', <void **>&__cuGraphicsResourceGetMappedMipmappedArray, 5000, ptds_mode, NULL)

        global __cuGraphicsResourceGetMappedPointer_v2
        _F_cuGetProcAddress_v2('cuGraphicsResourceGetMappedPointer', <void **>&__cuGraphicsResourceGetMappedPointer_v2, 3020, ptds_mode, NULL)

        global __cuGraphicsResourceSetMapFlags_v2
        _F_cuGetProcAddress_v2('cuGraphicsResourceSetMapFlags', <void **>&__cuGraphicsResourceSetMapFlags_v2, 6050, ptds_mode, NULL)

        global __cuGraphicsMapResources
        _F_cuGetProcAddress_v2('cuGraphicsMapResources', <void **>&__cuGraphicsMapResources, 3000, ptds_mode, NULL)

        global __cuGraphicsUnmapResources
        _F_cuGetProcAddress_v2('cuGraphicsUnmapResources', <void **>&__cuGraphicsUnmapResources, 3000, ptds_mode, NULL)

        global __cuGetProcAddress_v2
        _F_cuGetProcAddress_v2('cuGetProcAddress', <void **>&__cuGetProcAddress_v2, 12000, ptds_mode, NULL)

        global __cuCoredumpGetAttribute
        _F_cuGetProcAddress_v2('cuCoredumpGetAttribute', <void **>&__cuCoredumpGetAttribute, 12010, ptds_mode, NULL)

        global __cuCoredumpGetAttributeGlobal
        _F_cuGetProcAddress_v2('cuCoredumpGetAttributeGlobal', <void **>&__cuCoredumpGetAttributeGlobal, 12010, ptds_mode, NULL)

        global __cuCoredumpSetAttribute
        _F_cuGetProcAddress_v2('cuCoredumpSetAttribute', <void **>&__cuCoredumpSetAttribute, 12010, ptds_mode, NULL)

        global __cuCoredumpSetAttributeGlobal
        _F_cuGetProcAddress_v2('cuCoredumpSetAttributeGlobal', <void **>&__cuCoredumpSetAttributeGlobal, 12010, ptds_mode, NULL)

        global __cuGetExportTable
        _F_cuGetProcAddress_v2('cuGetExportTable', <void **>&__cuGetExportTable, 3000, ptds_mode, NULL)

        global __cuGreenCtxCreate
        _F_cuGetProcAddress_v2('cuGreenCtxCreate', <void **>&__cuGreenCtxCreate, 12040, ptds_mode, NULL)

        global __cuGreenCtxDestroy
        _F_cuGetProcAddress_v2('cuGreenCtxDestroy', <void **>&__cuGreenCtxDestroy, 12040, ptds_mode, NULL)

        global __cuCtxFromGreenCtx
        _F_cuGetProcAddress_v2('cuCtxFromGreenCtx', <void **>&__cuCtxFromGreenCtx, 12040, ptds_mode, NULL)

        global __cuDeviceGetDevResource
        _F_cuGetProcAddress_v2('cuDeviceGetDevResource', <void **>&__cuDeviceGetDevResource, 12040, ptds_mode, NULL)

        global __cuCtxGetDevResource
        _F_cuGetProcAddress_v2('cuCtxGetDevResource', <void **>&__cuCtxGetDevResource, 12040, ptds_mode, NULL)

        global __cuGreenCtxGetDevResource
        _F_cuGetProcAddress_v2('cuGreenCtxGetDevResource', <void **>&__cuGreenCtxGetDevResource, 12040, ptds_mode, NULL)

        global __cuDevSmResourceSplitByCount
        _F_cuGetProcAddress_v2('cuDevSmResourceSplitByCount', <void **>&__cuDevSmResourceSplitByCount, 12040, ptds_mode, NULL)

        global __cuDevResourceGenerateDesc
        _F_cuGetProcAddress_v2('cuDevResourceGenerateDesc', <void **>&__cuDevResourceGenerateDesc, 12040, ptds_mode, NULL)

        global __cuGreenCtxRecordEvent
        _F_cuGetProcAddress_v2('cuGreenCtxRecordEvent', <void **>&__cuGreenCtxRecordEvent, 12040, ptds_mode, NULL)

        global __cuGreenCtxWaitEvent
        _F_cuGetProcAddress_v2('cuGreenCtxWaitEvent', <void **>&__cuGreenCtxWaitEvent, 12040, ptds_mode, NULL)

        global __cuStreamGetGreenCtx
        _F_cuGetProcAddress_v2('cuStreamGetGreenCtx', <void **>&__cuStreamGetGreenCtx, 12040, ptds_mode, NULL)

        global __cuGreenCtxStreamCreate
        _F_cuGetProcAddress_v2('cuGreenCtxStreamCreate', <void **>&__cuGreenCtxStreamCreate, 12050, ptds_mode, NULL)

        global __cuLogsRegisterCallback
        _F_cuGetProcAddress_v2('cuLogsRegisterCallback', <void **>&__cuLogsRegisterCallback, 12080, ptds_mode, NULL)

        global __cuLogsUnregisterCallback
        _F_cuGetProcAddress_v2('cuLogsUnregisterCallback', <void **>&__cuLogsUnregisterCallback, 12080, ptds_mode, NULL)

        global __cuLogsCurrent
        _F_cuGetProcAddress_v2('cuLogsCurrent', <void **>&__cuLogsCurrent, 12080, ptds_mode, NULL)

        global __cuLogsDumpToFile
        _F_cuGetProcAddress_v2('cuLogsDumpToFile', <void **>&__cuLogsDumpToFile, 12080, ptds_mode, NULL)

        global __cuLogsDumpToMemory
        _F_cuGetProcAddress_v2('cuLogsDumpToMemory', <void **>&__cuLogsDumpToMemory, 12080, ptds_mode, NULL)

        global __cuCheckpointProcessGetRestoreThreadId
        _F_cuGetProcAddress_v2('cuCheckpointProcessGetRestoreThreadId', <void **>&__cuCheckpointProcessGetRestoreThreadId, 12080, ptds_mode, NULL)

        global __cuCheckpointProcessGetState
        _F_cuGetProcAddress_v2('cuCheckpointProcessGetState', <void **>&__cuCheckpointProcessGetState, 12080, ptds_mode, NULL)

        global __cuCheckpointProcessLock
        _F_cuGetProcAddress_v2('cuCheckpointProcessLock', <void **>&__cuCheckpointProcessLock, 12080, ptds_mode, NULL)

        global __cuCheckpointProcessCheckpoint
        _F_cuGetProcAddress_v2('cuCheckpointProcessCheckpoint', <void **>&__cuCheckpointProcessCheckpoint, 12080, ptds_mode, NULL)

        global __cuCheckpointProcessRestore
        _F_cuGetProcAddress_v2('cuCheckpointProcessRestore', <void **>&__cuCheckpointProcessRestore, 12080, ptds_mode, NULL)

        global __cuCheckpointProcessUnlock
        _F_cuGetProcAddress_v2('cuCheckpointProcessUnlock', <void **>&__cuCheckpointProcessUnlock, 12080, ptds_mode, NULL)

        global __cuGraphicsEGLRegisterImage
        _F_cuGetProcAddress_v2('cuGraphicsEGLRegisterImage', <void **>&__cuGraphicsEGLRegisterImage, 7000, ptds_mode, NULL)

        global __cuEGLStreamConsumerConnect
        _F_cuGetProcAddress_v2('cuEGLStreamConsumerConnect', <void **>&__cuEGLStreamConsumerConnect, 7000, ptds_mode, NULL)

        global __cuEGLStreamConsumerConnectWithFlags
        _F_cuGetProcAddress_v2('cuEGLStreamConsumerConnectWithFlags', <void **>&__cuEGLStreamConsumerConnectWithFlags, 8000, ptds_mode, NULL)

        global __cuEGLStreamConsumerDisconnect
        _F_cuGetProcAddress_v2('cuEGLStreamConsumerDisconnect', <void **>&__cuEGLStreamConsumerDisconnect, 7000, ptds_mode, NULL)

        global __cuEGLStreamConsumerAcquireFrame
        _F_cuGetProcAddress_v2('cuEGLStreamConsumerAcquireFrame', <void **>&__cuEGLStreamConsumerAcquireFrame, 7000, ptds_mode, NULL)

        global __cuEGLStreamConsumerReleaseFrame
        _F_cuGetProcAddress_v2('cuEGLStreamConsumerReleaseFrame', <void **>&__cuEGLStreamConsumerReleaseFrame, 7000, ptds_mode, NULL)

        global __cuEGLStreamProducerConnect
        _F_cuGetProcAddress_v2('cuEGLStreamProducerConnect', <void **>&__cuEGLStreamProducerConnect, 7000, ptds_mode, NULL)

        global __cuEGLStreamProducerDisconnect
        _F_cuGetProcAddress_v2('cuEGLStreamProducerDisconnect', <void **>&__cuEGLStreamProducerDisconnect, 7000, ptds_mode, NULL)

        global __cuEGLStreamProducerPresentFrame
        _F_cuGetProcAddress_v2('cuEGLStreamProducerPresentFrame', <void **>&__cuEGLStreamProducerPresentFrame, 7000, ptds_mode, NULL)

        global __cuEGLStreamProducerReturnFrame
        _F_cuGetProcAddress_v2('cuEGLStreamProducerReturnFrame', <void **>&__cuEGLStreamProducerReturnFrame, 7000, ptds_mode, NULL)

        global __cuGraphicsResourceGetMappedEglFrame
        _F_cuGetProcAddress_v2('cuGraphicsResourceGetMappedEglFrame', <void **>&__cuGraphicsResourceGetMappedEglFrame, 7000, ptds_mode, NULL)

        global __cuEventCreateFromEGLSync
        _F_cuGetProcAddress_v2('cuEventCreateFromEGLSync', <void **>&__cuEventCreateFromEGLSync, 9000, ptds_mode, NULL)

        global __cuGraphicsGLRegisterBuffer
        _F_cuGetProcAddress_v2('cuGraphicsGLRegisterBuffer', <void **>&__cuGraphicsGLRegisterBuffer, 3000, ptds_mode, NULL)

        global __cuGraphicsGLRegisterImage
        _F_cuGetProcAddress_v2('cuGraphicsGLRegisterImage', <void **>&__cuGraphicsGLRegisterImage, 3000, ptds_mode, NULL)

        global __cuGLGetDevices_v2
        _F_cuGetProcAddress_v2('cuGLGetDevices', <void **>&__cuGLGetDevices_v2, 6050, ptds_mode, NULL)

        global __cuGLCtxCreate_v2
        _F_cuGetProcAddress_v2('cuGLCtxCreate', <void **>&__cuGLCtxCreate_v2, 3020, ptds_mode, NULL)

        global __cuGLInit
        _F_cuGetProcAddress_v2('cuGLInit', <void **>&__cuGLInit, 2000, ptds_mode, NULL)

        global __cuGLRegisterBufferObject
        _F_cuGetProcAddress_v2('cuGLRegisterBufferObject', <void **>&__cuGLRegisterBufferObject, 2000, ptds_mode, NULL)

        global __cuGLMapBufferObject_v2
        _F_cuGetProcAddress_v2('cuGLMapBufferObject', <void **>&__cuGLMapBufferObject_v2, 3020, ptds_mode, NULL)

        global __cuGLUnmapBufferObject
        _F_cuGetProcAddress_v2('cuGLUnmapBufferObject', <void **>&__cuGLUnmapBufferObject, 2000, ptds_mode, NULL)

        global __cuGLUnregisterBufferObject
        _F_cuGetProcAddress_v2('cuGLUnregisterBufferObject', <void **>&__cuGLUnregisterBufferObject, 2000, ptds_mode, NULL)

        global __cuGLSetBufferObjectMapFlags
        _F_cuGetProcAddress_v2('cuGLSetBufferObjectMapFlags', <void **>&__cuGLSetBufferObjectMapFlags, 2030, ptds_mode, NULL)

        global __cuGLMapBufferObjectAsync_v2
        _F_cuGetProcAddress_v2('cuGLMapBufferObjectAsync', <void **>&__cuGLMapBufferObjectAsync_v2, 3020, ptds_mode, NULL)

        global __cuGLUnmapBufferObjectAsync
        _F_cuGetProcAddress_v2('cuGLUnmapBufferObjectAsync', <void **>&__cuGLUnmapBufferObjectAsync, 2030, ptds_mode, NULL)

        global __cuProfilerInitialize
        _F_cuGetProcAddress_v2('cuProfilerInitialize', <void **>&__cuProfilerInitialize, 4000, ptds_mode, NULL)

        global __cuProfilerStart
        _F_cuGetProcAddress_v2('cuProfilerStart', <void **>&__cuProfilerStart, 4000, ptds_mode, NULL)

        global __cuProfilerStop
        _F_cuGetProcAddress_v2('cuProfilerStop', <void **>&__cuProfilerStop, 4000, ptds_mode, NULL)

        global __cuVDPAUGetDevice
        _F_cuGetProcAddress_v2('cuVDPAUGetDevice', <void **>&__cuVDPAUGetDevice, 3010, ptds_mode, NULL)

        global __cuVDPAUCtxCreate_v2
        _F_cuGetProcAddress_v2('cuVDPAUCtxCreate', <void **>&__cuVDPAUCtxCreate_v2, 3020, ptds_mode, NULL)

        global __cuGraphicsVDPAURegisterVideoSurface
        _F_cuGetProcAddress_v2('cuGraphicsVDPAURegisterVideoSurface', <void **>&__cuGraphicsVDPAURegisterVideoSurface, 3010, ptds_mode, NULL)

        global __cuGraphicsVDPAURegisterOutputSurface
        _F_cuGetProcAddress_v2('cuGraphicsVDPAURegisterOutputSurface', <void **>&__cuGraphicsVDPAURegisterOutputSurface, 3010, ptds_mode, NULL)

        global __cuDeviceGetHostAtomicCapabilities
        _F_cuGetProcAddress_v2('cuDeviceGetHostAtomicCapabilities', <void **>&__cuDeviceGetHostAtomicCapabilities, 13000, ptds_mode, NULL)

        global __cuCtxGetDevice_v2
        _F_cuGetProcAddress_v2('cuCtxGetDevice_v2', <void **>&__cuCtxGetDevice_v2, 13000, ptds_mode, NULL)

        global __cuCtxSynchronize_v2
        _F_cuGetProcAddress_v2('cuCtxSynchronize_v2', <void **>&__cuCtxSynchronize_v2, 13000, ptds_mode, NULL)

        global __cuMemcpyBatchAsync_v2
        _F_cuGetProcAddress_v2('cuMemcpyBatchAsync', <void **>&__cuMemcpyBatchAsync_v2, 13000, ptds_mode, NULL)

        global __cuMemcpy3DBatchAsync_v2
        _F_cuGetProcAddress_v2('cuMemcpy3DBatchAsync', <void **>&__cuMemcpy3DBatchAsync_v2, 13000, ptds_mode, NULL)

        global __cuMemGetDefaultMemPool
        _F_cuGetProcAddress_v2('cuMemGetDefaultMemPool', <void **>&__cuMemGetDefaultMemPool, 13000, ptds_mode, NULL)

        global __cuMemGetMemPool
        _F_cuGetProcAddress_v2('cuMemGetMemPool', <void **>&__cuMemGetMemPool, 13000, ptds_mode, NULL)

        global __cuMemSetMemPool
        _F_cuGetProcAddress_v2('cuMemSetMemPool', <void **>&__cuMemSetMemPool, 13000, ptds_mode, NULL)

        global __cuMemPrefetchBatchAsync
        _F_cuGetProcAddress_v2('cuMemPrefetchBatchAsync', <void **>&__cuMemPrefetchBatchAsync, 13000, ptds_mode, NULL)

        global __cuMemDiscardBatchAsync
        _F_cuGetProcAddress_v2('cuMemDiscardBatchAsync', <void **>&__cuMemDiscardBatchAsync, 13000, ptds_mode, NULL)

        global __cuMemDiscardAndPrefetchBatchAsync
        _F_cuGetProcAddress_v2('cuMemDiscardAndPrefetchBatchAsync', <void **>&__cuMemDiscardAndPrefetchBatchAsync, 13000, ptds_mode, NULL)

        global __cuDeviceGetP2PAtomicCapabilities
        _F_cuGetProcAddress_v2('cuDeviceGetP2PAtomicCapabilities', <void **>&__cuDeviceGetP2PAtomicCapabilities, 13000, ptds_mode, NULL)

        global __cuGreenCtxGetId
        _F_cuGetProcAddress_v2('cuGreenCtxGetId', <void **>&__cuGreenCtxGetId, 13000, ptds_mode, NULL)

        global __cuMulticastBindMem_v2
        _F_cuGetProcAddress_v2('cuMulticastBindMem_v2', <void **>&__cuMulticastBindMem_v2, 13010, ptds_mode, NULL)

        global __cuMulticastBindAddr_v2
        _F_cuGetProcAddress_v2('cuMulticastBindAddr_v2', <void **>&__cuMulticastBindAddr_v2, 13010, ptds_mode, NULL)

        global __cuGraphNodeGetContainingGraph
        _F_cuGetProcAddress_v2('cuGraphNodeGetContainingGraph', <void **>&__cuGraphNodeGetContainingGraph, 13010, ptds_mode, NULL)

        global __cuGraphNodeGetLocalId
        _F_cuGetProcAddress_v2('cuGraphNodeGetLocalId', <void **>&__cuGraphNodeGetLocalId, 13010, ptds_mode, NULL)

        global __cuGraphNodeGetToolsId
        _F_cuGetProcAddress_v2('cuGraphNodeGetToolsId', <void **>&__cuGraphNodeGetToolsId, 13010, ptds_mode, NULL)

        global __cuGraphGetId
        _F_cuGetProcAddress_v2('cuGraphGetId', <void **>&__cuGraphGetId, 13010, ptds_mode, NULL)

        global __cuGraphExecGetId
        _F_cuGetProcAddress_v2('cuGraphExecGetId', <void **>&__cuGraphExecGetId, 13010, ptds_mode, NULL)

        global __cuDevSmResourceSplit
        _F_cuGetProcAddress_v2('cuDevSmResourceSplit', <void **>&__cuDevSmResourceSplit, 13010, ptds_mode, NULL)

        global __cuStreamGetDevResource
        _F_cuGetProcAddress_v2('cuStreamGetDevResource', <void **>&__cuStreamGetDevResource, 13010, ptds_mode, NULL)

        global __cuKernelGetParamCount
        _F_cuGetProcAddress_v2('cuKernelGetParamCount', <void **>&__cuKernelGetParamCount, 13020, ptds_mode, NULL)

        global __cuMemcpyWithAttributesAsync
        _F_cuGetProcAddress_v2('cuMemcpyWithAttributesAsync', <void **>&__cuMemcpyWithAttributesAsync, 13020, ptds_mode, NULL)

        global __cuMemcpy3DWithAttributesAsync
        _F_cuGetProcAddress_v2('cuMemcpy3DWithAttributesAsync', <void **>&__cuMemcpy3DWithAttributesAsync, 13020, ptds_mode, NULL)

        global __cuStreamBeginCaptureToCig
        _F_cuGetProcAddress_v2('cuStreamBeginCaptureToCig', <void **>&__cuStreamBeginCaptureToCig, 13020, ptds_mode, NULL)

        global __cuStreamEndCaptureToCig
        _F_cuGetProcAddress_v2('cuStreamEndCaptureToCig', <void **>&__cuStreamEndCaptureToCig, 13020, ptds_mode, NULL)

        global __cuFuncGetParamCount
        _F_cuGetProcAddress_v2('cuFuncGetParamCount', <void **>&__cuFuncGetParamCount, 13020, ptds_mode, NULL)

        global __cuLaunchHostFunc_v2
        _F_cuGetProcAddress_v2('cuLaunchHostFunc_v2', <void **>&__cuLaunchHostFunc_v2, 13020, ptds_mode, NULL)

        global __cuGraphNodeGetParams
        _F_cuGetProcAddress_v2('cuGraphNodeGetParams', <void **>&__cuGraphNodeGetParams, 13020, ptds_mode, NULL)

        global __cuCoredumpRegisterStartCallback
        _F_cuGetProcAddress_v2('cuCoredumpRegisterStartCallback', <void **>&__cuCoredumpRegisterStartCallback, 13020, ptds_mode, NULL)

        global __cuCoredumpRegisterCompleteCallback
        _F_cuGetProcAddress_v2('cuCoredumpRegisterCompleteCallback', <void **>&__cuCoredumpRegisterCompleteCallback, 13020, ptds_mode, NULL)

        global __cuCoredumpDeregisterStartCallback
        _F_cuGetProcAddress_v2('cuCoredumpDeregisterStartCallback', <void **>&__cuCoredumpDeregisterStartCallback, 13020, ptds_mode, NULL)

        global __cuCoredumpDeregisterCompleteCallback
        _F_cuGetProcAddress_v2('cuCoredumpDeregisterCompleteCallback', <void **>&__cuCoredumpDeregisterCompleteCallback, 13020, ptds_mode, NULL)

        __py_driver_init = True
        return 0


cdef inline int _check_or_init_driver() except -1 nogil:
    if __py_driver_init:
        return 0

    return _init_driver()

cdef dict func_ptrs = None


cpdef dict _inspect_function_pointers():
    global func_ptrs
    if func_ptrs is not None:
        return func_ptrs

    _check_or_init_driver()
    cdef dict data = {}

    global __cuGetErrorString
    data["__cuGetErrorString"] = <intptr_t>__cuGetErrorString

    global __cuGetErrorName
    data["__cuGetErrorName"] = <intptr_t>__cuGetErrorName

    global __cuInit
    data["__cuInit"] = <intptr_t>__cuInit

    global __cuDriverGetVersion
    data["__cuDriverGetVersion"] = <intptr_t>__cuDriverGetVersion

    global __cuDeviceGet
    data["__cuDeviceGet"] = <intptr_t>__cuDeviceGet

    global __cuDeviceGetCount
    data["__cuDeviceGetCount"] = <intptr_t>__cuDeviceGetCount

    global __cuDeviceGetName
    data["__cuDeviceGetName"] = <intptr_t>__cuDeviceGetName

    global __cuDeviceGetUuid_v2
    data["__cuDeviceGetUuid_v2"] = <intptr_t>__cuDeviceGetUuid_v2

    global __cuDeviceGetLuid
    data["__cuDeviceGetLuid"] = <intptr_t>__cuDeviceGetLuid

    global __cuDeviceTotalMem_v2
    data["__cuDeviceTotalMem_v2"] = <intptr_t>__cuDeviceTotalMem_v2

    global __cuDeviceGetTexture1DLinearMaxWidth
    data["__cuDeviceGetTexture1DLinearMaxWidth"] = <intptr_t>__cuDeviceGetTexture1DLinearMaxWidth

    global __cuDeviceGetAttribute
    data["__cuDeviceGetAttribute"] = <intptr_t>__cuDeviceGetAttribute

    global __cuDeviceGetNvSciSyncAttributes
    data["__cuDeviceGetNvSciSyncAttributes"] = <intptr_t>__cuDeviceGetNvSciSyncAttributes

    global __cuDeviceSetMemPool
    data["__cuDeviceSetMemPool"] = <intptr_t>__cuDeviceSetMemPool

    global __cuDeviceGetMemPool
    data["__cuDeviceGetMemPool"] = <intptr_t>__cuDeviceGetMemPool

    global __cuDeviceGetDefaultMemPool
    data["__cuDeviceGetDefaultMemPool"] = <intptr_t>__cuDeviceGetDefaultMemPool

    global __cuDeviceGetExecAffinitySupport
    data["__cuDeviceGetExecAffinitySupport"] = <intptr_t>__cuDeviceGetExecAffinitySupport

    global __cuFlushGPUDirectRDMAWrites
    data["__cuFlushGPUDirectRDMAWrites"] = <intptr_t>__cuFlushGPUDirectRDMAWrites

    global __cuDeviceGetProperties
    data["__cuDeviceGetProperties"] = <intptr_t>__cuDeviceGetProperties

    global __cuDeviceComputeCapability
    data["__cuDeviceComputeCapability"] = <intptr_t>__cuDeviceComputeCapability

    global __cuDevicePrimaryCtxRetain
    data["__cuDevicePrimaryCtxRetain"] = <intptr_t>__cuDevicePrimaryCtxRetain

    global __cuDevicePrimaryCtxRelease_v2
    data["__cuDevicePrimaryCtxRelease_v2"] = <intptr_t>__cuDevicePrimaryCtxRelease_v2

    global __cuDevicePrimaryCtxSetFlags_v2
    data["__cuDevicePrimaryCtxSetFlags_v2"] = <intptr_t>__cuDevicePrimaryCtxSetFlags_v2

    global __cuDevicePrimaryCtxGetState
    data["__cuDevicePrimaryCtxGetState"] = <intptr_t>__cuDevicePrimaryCtxGetState

    global __cuDevicePrimaryCtxReset_v2
    data["__cuDevicePrimaryCtxReset_v2"] = <intptr_t>__cuDevicePrimaryCtxReset_v2

    global __cuCtxCreate_v4
    data["__cuCtxCreate_v4"] = <intptr_t>__cuCtxCreate_v4

    global __cuCtxDestroy_v2
    data["__cuCtxDestroy_v2"] = <intptr_t>__cuCtxDestroy_v2

    global __cuCtxPushCurrent_v2
    data["__cuCtxPushCurrent_v2"] = <intptr_t>__cuCtxPushCurrent_v2

    global __cuCtxPopCurrent_v2
    data["__cuCtxPopCurrent_v2"] = <intptr_t>__cuCtxPopCurrent_v2

    global __cuCtxSetCurrent
    data["__cuCtxSetCurrent"] = <intptr_t>__cuCtxSetCurrent

    global __cuCtxGetCurrent
    data["__cuCtxGetCurrent"] = <intptr_t>__cuCtxGetCurrent

    global __cuCtxGetDevice
    data["__cuCtxGetDevice"] = <intptr_t>__cuCtxGetDevice

    global __cuCtxGetFlags
    data["__cuCtxGetFlags"] = <intptr_t>__cuCtxGetFlags

    global __cuCtxSetFlags
    data["__cuCtxSetFlags"] = <intptr_t>__cuCtxSetFlags

    global __cuCtxGetId
    data["__cuCtxGetId"] = <intptr_t>__cuCtxGetId

    global __cuCtxSynchronize
    data["__cuCtxSynchronize"] = <intptr_t>__cuCtxSynchronize

    global __cuCtxSetLimit
    data["__cuCtxSetLimit"] = <intptr_t>__cuCtxSetLimit

    global __cuCtxGetLimit
    data["__cuCtxGetLimit"] = <intptr_t>__cuCtxGetLimit

    global __cuCtxGetCacheConfig
    data["__cuCtxGetCacheConfig"] = <intptr_t>__cuCtxGetCacheConfig

    global __cuCtxSetCacheConfig
    data["__cuCtxSetCacheConfig"] = <intptr_t>__cuCtxSetCacheConfig

    global __cuCtxGetApiVersion
    data["__cuCtxGetApiVersion"] = <intptr_t>__cuCtxGetApiVersion

    global __cuCtxGetStreamPriorityRange
    data["__cuCtxGetStreamPriorityRange"] = <intptr_t>__cuCtxGetStreamPriorityRange

    global __cuCtxResetPersistingL2Cache
    data["__cuCtxResetPersistingL2Cache"] = <intptr_t>__cuCtxResetPersistingL2Cache

    global __cuCtxGetExecAffinity
    data["__cuCtxGetExecAffinity"] = <intptr_t>__cuCtxGetExecAffinity

    global __cuCtxRecordEvent
    data["__cuCtxRecordEvent"] = <intptr_t>__cuCtxRecordEvent

    global __cuCtxWaitEvent
    data["__cuCtxWaitEvent"] = <intptr_t>__cuCtxWaitEvent

    global __cuCtxAttach
    data["__cuCtxAttach"] = <intptr_t>__cuCtxAttach

    global __cuCtxDetach
    data["__cuCtxDetach"] = <intptr_t>__cuCtxDetach

    global __cuCtxGetSharedMemConfig
    data["__cuCtxGetSharedMemConfig"] = <intptr_t>__cuCtxGetSharedMemConfig

    global __cuCtxSetSharedMemConfig
    data["__cuCtxSetSharedMemConfig"] = <intptr_t>__cuCtxSetSharedMemConfig

    global __cuModuleLoad
    data["__cuModuleLoad"] = <intptr_t>__cuModuleLoad

    global __cuModuleLoadData
    data["__cuModuleLoadData"] = <intptr_t>__cuModuleLoadData

    global __cuModuleLoadDataEx
    data["__cuModuleLoadDataEx"] = <intptr_t>__cuModuleLoadDataEx

    global __cuModuleLoadFatBinary
    data["__cuModuleLoadFatBinary"] = <intptr_t>__cuModuleLoadFatBinary

    global __cuModuleUnload
    data["__cuModuleUnload"] = <intptr_t>__cuModuleUnload

    global __cuModuleGetLoadingMode
    data["__cuModuleGetLoadingMode"] = <intptr_t>__cuModuleGetLoadingMode

    global __cuModuleGetFunction
    data["__cuModuleGetFunction"] = <intptr_t>__cuModuleGetFunction

    global __cuModuleGetFunctionCount
    data["__cuModuleGetFunctionCount"] = <intptr_t>__cuModuleGetFunctionCount

    global __cuModuleEnumerateFunctions
    data["__cuModuleEnumerateFunctions"] = <intptr_t>__cuModuleEnumerateFunctions

    global __cuModuleGetGlobal_v2
    data["__cuModuleGetGlobal_v2"] = <intptr_t>__cuModuleGetGlobal_v2

    global __cuLinkCreate_v2
    data["__cuLinkCreate_v2"] = <intptr_t>__cuLinkCreate_v2

    global __cuLinkAddData_v2
    data["__cuLinkAddData_v2"] = <intptr_t>__cuLinkAddData_v2

    global __cuLinkAddFile_v2
    data["__cuLinkAddFile_v2"] = <intptr_t>__cuLinkAddFile_v2

    global __cuLinkComplete
    data["__cuLinkComplete"] = <intptr_t>__cuLinkComplete

    global __cuLinkDestroy
    data["__cuLinkDestroy"] = <intptr_t>__cuLinkDestroy

    global __cuModuleGetTexRef
    data["__cuModuleGetTexRef"] = <intptr_t>__cuModuleGetTexRef

    global __cuModuleGetSurfRef
    data["__cuModuleGetSurfRef"] = <intptr_t>__cuModuleGetSurfRef

    global __cuLibraryLoadData
    data["__cuLibraryLoadData"] = <intptr_t>__cuLibraryLoadData

    global __cuLibraryLoadFromFile
    data["__cuLibraryLoadFromFile"] = <intptr_t>__cuLibraryLoadFromFile

    global __cuLibraryUnload
    data["__cuLibraryUnload"] = <intptr_t>__cuLibraryUnload

    global __cuLibraryGetKernel
    data["__cuLibraryGetKernel"] = <intptr_t>__cuLibraryGetKernel

    global __cuLibraryGetKernelCount
    data["__cuLibraryGetKernelCount"] = <intptr_t>__cuLibraryGetKernelCount

    global __cuLibraryEnumerateKernels
    data["__cuLibraryEnumerateKernels"] = <intptr_t>__cuLibraryEnumerateKernels

    global __cuLibraryGetModule
    data["__cuLibraryGetModule"] = <intptr_t>__cuLibraryGetModule

    global __cuKernelGetFunction
    data["__cuKernelGetFunction"] = <intptr_t>__cuKernelGetFunction

    global __cuKernelGetLibrary
    data["__cuKernelGetLibrary"] = <intptr_t>__cuKernelGetLibrary

    global __cuLibraryGetGlobal
    data["__cuLibraryGetGlobal"] = <intptr_t>__cuLibraryGetGlobal

    global __cuLibraryGetManaged
    data["__cuLibraryGetManaged"] = <intptr_t>__cuLibraryGetManaged

    global __cuLibraryGetUnifiedFunction
    data["__cuLibraryGetUnifiedFunction"] = <intptr_t>__cuLibraryGetUnifiedFunction

    global __cuKernelGetAttribute
    data["__cuKernelGetAttribute"] = <intptr_t>__cuKernelGetAttribute

    global __cuKernelSetAttribute
    data["__cuKernelSetAttribute"] = <intptr_t>__cuKernelSetAttribute

    global __cuKernelSetCacheConfig
    data["__cuKernelSetCacheConfig"] = <intptr_t>__cuKernelSetCacheConfig

    global __cuKernelGetName
    data["__cuKernelGetName"] = <intptr_t>__cuKernelGetName

    global __cuKernelGetParamInfo
    data["__cuKernelGetParamInfo"] = <intptr_t>__cuKernelGetParamInfo

    global __cuMemGetInfo_v2
    data["__cuMemGetInfo_v2"] = <intptr_t>__cuMemGetInfo_v2

    global __cuMemAlloc_v2
    data["__cuMemAlloc_v2"] = <intptr_t>__cuMemAlloc_v2

    global __cuMemAllocPitch_v2
    data["__cuMemAllocPitch_v2"] = <intptr_t>__cuMemAllocPitch_v2

    global __cuMemFree_v2
    data["__cuMemFree_v2"] = <intptr_t>__cuMemFree_v2

    global __cuMemGetAddressRange_v2
    data["__cuMemGetAddressRange_v2"] = <intptr_t>__cuMemGetAddressRange_v2

    global __cuMemAllocHost_v2
    data["__cuMemAllocHost_v2"] = <intptr_t>__cuMemAllocHost_v2

    global __cuMemFreeHost
    data["__cuMemFreeHost"] = <intptr_t>__cuMemFreeHost

    global __cuMemHostAlloc
    data["__cuMemHostAlloc"] = <intptr_t>__cuMemHostAlloc

    global __cuMemHostGetDevicePointer_v2
    data["__cuMemHostGetDevicePointer_v2"] = <intptr_t>__cuMemHostGetDevicePointer_v2

    global __cuMemHostGetFlags
    data["__cuMemHostGetFlags"] = <intptr_t>__cuMemHostGetFlags

    global __cuMemAllocManaged
    data["__cuMemAllocManaged"] = <intptr_t>__cuMemAllocManaged

    global __cuDeviceRegisterAsyncNotification
    data["__cuDeviceRegisterAsyncNotification"] = <intptr_t>__cuDeviceRegisterAsyncNotification

    global __cuDeviceUnregisterAsyncNotification
    data["__cuDeviceUnregisterAsyncNotification"] = <intptr_t>__cuDeviceUnregisterAsyncNotification

    global __cuDeviceGetByPCIBusId
    data["__cuDeviceGetByPCIBusId"] = <intptr_t>__cuDeviceGetByPCIBusId

    global __cuDeviceGetPCIBusId
    data["__cuDeviceGetPCIBusId"] = <intptr_t>__cuDeviceGetPCIBusId

    global __cuIpcGetEventHandle
    data["__cuIpcGetEventHandle"] = <intptr_t>__cuIpcGetEventHandle

    global __cuIpcOpenEventHandle
    data["__cuIpcOpenEventHandle"] = <intptr_t>__cuIpcOpenEventHandle

    global __cuIpcGetMemHandle
    data["__cuIpcGetMemHandle"] = <intptr_t>__cuIpcGetMemHandle

    global __cuIpcOpenMemHandle_v2
    data["__cuIpcOpenMemHandle_v2"] = <intptr_t>__cuIpcOpenMemHandle_v2

    global __cuIpcCloseMemHandle
    data["__cuIpcCloseMemHandle"] = <intptr_t>__cuIpcCloseMemHandle

    global __cuMemHostRegister_v2
    data["__cuMemHostRegister_v2"] = <intptr_t>__cuMemHostRegister_v2

    global __cuMemHostUnregister
    data["__cuMemHostUnregister"] = <intptr_t>__cuMemHostUnregister

    global __cuMemcpy
    data["__cuMemcpy"] = <intptr_t>__cuMemcpy

    global __cuMemcpyPeer
    data["__cuMemcpyPeer"] = <intptr_t>__cuMemcpyPeer

    global __cuMemcpyHtoD_v2
    data["__cuMemcpyHtoD_v2"] = <intptr_t>__cuMemcpyHtoD_v2

    global __cuMemcpyDtoH_v2
    data["__cuMemcpyDtoH_v2"] = <intptr_t>__cuMemcpyDtoH_v2

    global __cuMemcpyDtoD_v2
    data["__cuMemcpyDtoD_v2"] = <intptr_t>__cuMemcpyDtoD_v2

    global __cuMemcpyDtoA_v2
    data["__cuMemcpyDtoA_v2"] = <intptr_t>__cuMemcpyDtoA_v2

    global __cuMemcpyAtoD_v2
    data["__cuMemcpyAtoD_v2"] = <intptr_t>__cuMemcpyAtoD_v2

    global __cuMemcpyHtoA_v2
    data["__cuMemcpyHtoA_v2"] = <intptr_t>__cuMemcpyHtoA_v2

    global __cuMemcpyAtoH_v2
    data["__cuMemcpyAtoH_v2"] = <intptr_t>__cuMemcpyAtoH_v2

    global __cuMemcpyAtoA_v2
    data["__cuMemcpyAtoA_v2"] = <intptr_t>__cuMemcpyAtoA_v2

    global __cuMemcpy2D_v2
    data["__cuMemcpy2D_v2"] = <intptr_t>__cuMemcpy2D_v2

    global __cuMemcpy2DUnaligned_v2
    data["__cuMemcpy2DUnaligned_v2"] = <intptr_t>__cuMemcpy2DUnaligned_v2

    global __cuMemcpy3D_v2
    data["__cuMemcpy3D_v2"] = <intptr_t>__cuMemcpy3D_v2

    global __cuMemcpy3DPeer
    data["__cuMemcpy3DPeer"] = <intptr_t>__cuMemcpy3DPeer

    global __cuMemcpyAsync
    data["__cuMemcpyAsync"] = <intptr_t>__cuMemcpyAsync

    global __cuMemcpyPeerAsync
    data["__cuMemcpyPeerAsync"] = <intptr_t>__cuMemcpyPeerAsync

    global __cuMemcpyHtoDAsync_v2
    data["__cuMemcpyHtoDAsync_v2"] = <intptr_t>__cuMemcpyHtoDAsync_v2

    global __cuMemcpyDtoHAsync_v2
    data["__cuMemcpyDtoHAsync_v2"] = <intptr_t>__cuMemcpyDtoHAsync_v2

    global __cuMemcpyDtoDAsync_v2
    data["__cuMemcpyDtoDAsync_v2"] = <intptr_t>__cuMemcpyDtoDAsync_v2

    global __cuMemcpyHtoAAsync_v2
    data["__cuMemcpyHtoAAsync_v2"] = <intptr_t>__cuMemcpyHtoAAsync_v2

    global __cuMemcpyAtoHAsync_v2
    data["__cuMemcpyAtoHAsync_v2"] = <intptr_t>__cuMemcpyAtoHAsync_v2

    global __cuMemcpy2DAsync_v2
    data["__cuMemcpy2DAsync_v2"] = <intptr_t>__cuMemcpy2DAsync_v2

    global __cuMemcpy3DAsync_v2
    data["__cuMemcpy3DAsync_v2"] = <intptr_t>__cuMemcpy3DAsync_v2

    global __cuMemcpy3DPeerAsync
    data["__cuMemcpy3DPeerAsync"] = <intptr_t>__cuMemcpy3DPeerAsync

    global __cuMemsetD8_v2
    data["__cuMemsetD8_v2"] = <intptr_t>__cuMemsetD8_v2

    global __cuMemsetD16_v2
    data["__cuMemsetD16_v2"] = <intptr_t>__cuMemsetD16_v2

    global __cuMemsetD32_v2
    data["__cuMemsetD32_v2"] = <intptr_t>__cuMemsetD32_v2

    global __cuMemsetD2D8_v2
    data["__cuMemsetD2D8_v2"] = <intptr_t>__cuMemsetD2D8_v2

    global __cuMemsetD2D16_v2
    data["__cuMemsetD2D16_v2"] = <intptr_t>__cuMemsetD2D16_v2

    global __cuMemsetD2D32_v2
    data["__cuMemsetD2D32_v2"] = <intptr_t>__cuMemsetD2D32_v2

    global __cuMemsetD8Async
    data["__cuMemsetD8Async"] = <intptr_t>__cuMemsetD8Async

    global __cuMemsetD16Async
    data["__cuMemsetD16Async"] = <intptr_t>__cuMemsetD16Async

    global __cuMemsetD32Async
    data["__cuMemsetD32Async"] = <intptr_t>__cuMemsetD32Async

    global __cuMemsetD2D8Async
    data["__cuMemsetD2D8Async"] = <intptr_t>__cuMemsetD2D8Async

    global __cuMemsetD2D16Async
    data["__cuMemsetD2D16Async"] = <intptr_t>__cuMemsetD2D16Async

    global __cuMemsetD2D32Async
    data["__cuMemsetD2D32Async"] = <intptr_t>__cuMemsetD2D32Async

    global __cuArrayCreate_v2
    data["__cuArrayCreate_v2"] = <intptr_t>__cuArrayCreate_v2

    global __cuArrayGetDescriptor_v2
    data["__cuArrayGetDescriptor_v2"] = <intptr_t>__cuArrayGetDescriptor_v2

    global __cuArrayGetSparseProperties
    data["__cuArrayGetSparseProperties"] = <intptr_t>__cuArrayGetSparseProperties

    global __cuMipmappedArrayGetSparseProperties
    data["__cuMipmappedArrayGetSparseProperties"] = <intptr_t>__cuMipmappedArrayGetSparseProperties

    global __cuArrayGetMemoryRequirements
    data["__cuArrayGetMemoryRequirements"] = <intptr_t>__cuArrayGetMemoryRequirements

    global __cuMipmappedArrayGetMemoryRequirements
    data["__cuMipmappedArrayGetMemoryRequirements"] = <intptr_t>__cuMipmappedArrayGetMemoryRequirements

    global __cuArrayGetPlane
    data["__cuArrayGetPlane"] = <intptr_t>__cuArrayGetPlane

    global __cuArrayDestroy
    data["__cuArrayDestroy"] = <intptr_t>__cuArrayDestroy

    global __cuArray3DCreate_v2
    data["__cuArray3DCreate_v2"] = <intptr_t>__cuArray3DCreate_v2

    global __cuArray3DGetDescriptor_v2
    data["__cuArray3DGetDescriptor_v2"] = <intptr_t>__cuArray3DGetDescriptor_v2

    global __cuMipmappedArrayCreate
    data["__cuMipmappedArrayCreate"] = <intptr_t>__cuMipmappedArrayCreate

    global __cuMipmappedArrayGetLevel
    data["__cuMipmappedArrayGetLevel"] = <intptr_t>__cuMipmappedArrayGetLevel

    global __cuMipmappedArrayDestroy
    data["__cuMipmappedArrayDestroy"] = <intptr_t>__cuMipmappedArrayDestroy

    global __cuMemGetHandleForAddressRange
    data["__cuMemGetHandleForAddressRange"] = <intptr_t>__cuMemGetHandleForAddressRange

    global __cuMemBatchDecompressAsync
    data["__cuMemBatchDecompressAsync"] = <intptr_t>__cuMemBatchDecompressAsync

    global __cuMemAddressReserve
    data["__cuMemAddressReserve"] = <intptr_t>__cuMemAddressReserve

    global __cuMemAddressFree
    data["__cuMemAddressFree"] = <intptr_t>__cuMemAddressFree

    global __cuMemCreate
    data["__cuMemCreate"] = <intptr_t>__cuMemCreate

    global __cuMemRelease
    data["__cuMemRelease"] = <intptr_t>__cuMemRelease

    global __cuMemMap
    data["__cuMemMap"] = <intptr_t>__cuMemMap

    global __cuMemMapArrayAsync
    data["__cuMemMapArrayAsync"] = <intptr_t>__cuMemMapArrayAsync

    global __cuMemUnmap
    data["__cuMemUnmap"] = <intptr_t>__cuMemUnmap

    global __cuMemSetAccess
    data["__cuMemSetAccess"] = <intptr_t>__cuMemSetAccess

    global __cuMemGetAccess
    data["__cuMemGetAccess"] = <intptr_t>__cuMemGetAccess

    global __cuMemExportToShareableHandle
    data["__cuMemExportToShareableHandle"] = <intptr_t>__cuMemExportToShareableHandle

    global __cuMemImportFromShareableHandle
    data["__cuMemImportFromShareableHandle"] = <intptr_t>__cuMemImportFromShareableHandle

    global __cuMemGetAllocationGranularity
    data["__cuMemGetAllocationGranularity"] = <intptr_t>__cuMemGetAllocationGranularity

    global __cuMemGetAllocationPropertiesFromHandle
    data["__cuMemGetAllocationPropertiesFromHandle"] = <intptr_t>__cuMemGetAllocationPropertiesFromHandle

    global __cuMemRetainAllocationHandle
    data["__cuMemRetainAllocationHandle"] = <intptr_t>__cuMemRetainAllocationHandle

    global __cuMemFreeAsync
    data["__cuMemFreeAsync"] = <intptr_t>__cuMemFreeAsync

    global __cuMemAllocAsync
    data["__cuMemAllocAsync"] = <intptr_t>__cuMemAllocAsync

    global __cuMemPoolTrimTo
    data["__cuMemPoolTrimTo"] = <intptr_t>__cuMemPoolTrimTo

    global __cuMemPoolSetAttribute
    data["__cuMemPoolSetAttribute"] = <intptr_t>__cuMemPoolSetAttribute

    global __cuMemPoolGetAttribute
    data["__cuMemPoolGetAttribute"] = <intptr_t>__cuMemPoolGetAttribute

    global __cuMemPoolSetAccess
    data["__cuMemPoolSetAccess"] = <intptr_t>__cuMemPoolSetAccess

    global __cuMemPoolGetAccess
    data["__cuMemPoolGetAccess"] = <intptr_t>__cuMemPoolGetAccess

    global __cuMemPoolCreate
    data["__cuMemPoolCreate"] = <intptr_t>__cuMemPoolCreate

    global __cuMemPoolDestroy
    data["__cuMemPoolDestroy"] = <intptr_t>__cuMemPoolDestroy

    global __cuMemAllocFromPoolAsync
    data["__cuMemAllocFromPoolAsync"] = <intptr_t>__cuMemAllocFromPoolAsync

    global __cuMemPoolExportToShareableHandle
    data["__cuMemPoolExportToShareableHandle"] = <intptr_t>__cuMemPoolExportToShareableHandle

    global __cuMemPoolImportFromShareableHandle
    data["__cuMemPoolImportFromShareableHandle"] = <intptr_t>__cuMemPoolImportFromShareableHandle

    global __cuMemPoolExportPointer
    data["__cuMemPoolExportPointer"] = <intptr_t>__cuMemPoolExportPointer

    global __cuMemPoolImportPointer
    data["__cuMemPoolImportPointer"] = <intptr_t>__cuMemPoolImportPointer

    global __cuMulticastCreate
    data["__cuMulticastCreate"] = <intptr_t>__cuMulticastCreate

    global __cuMulticastAddDevice
    data["__cuMulticastAddDevice"] = <intptr_t>__cuMulticastAddDevice

    global __cuMulticastBindMem
    data["__cuMulticastBindMem"] = <intptr_t>__cuMulticastBindMem

    global __cuMulticastBindAddr
    data["__cuMulticastBindAddr"] = <intptr_t>__cuMulticastBindAddr

    global __cuMulticastUnbind
    data["__cuMulticastUnbind"] = <intptr_t>__cuMulticastUnbind

    global __cuMulticastGetGranularity
    data["__cuMulticastGetGranularity"] = <intptr_t>__cuMulticastGetGranularity

    global __cuPointerGetAttribute
    data["__cuPointerGetAttribute"] = <intptr_t>__cuPointerGetAttribute

    global __cuMemPrefetchAsync_v2
    data["__cuMemPrefetchAsync_v2"] = <intptr_t>__cuMemPrefetchAsync_v2

    global __cuMemAdvise_v2
    data["__cuMemAdvise_v2"] = <intptr_t>__cuMemAdvise_v2

    global __cuMemRangeGetAttribute
    data["__cuMemRangeGetAttribute"] = <intptr_t>__cuMemRangeGetAttribute

    global __cuMemRangeGetAttributes
    data["__cuMemRangeGetAttributes"] = <intptr_t>__cuMemRangeGetAttributes

    global __cuPointerSetAttribute
    data["__cuPointerSetAttribute"] = <intptr_t>__cuPointerSetAttribute

    global __cuPointerGetAttributes
    data["__cuPointerGetAttributes"] = <intptr_t>__cuPointerGetAttributes

    global __cuStreamCreate
    data["__cuStreamCreate"] = <intptr_t>__cuStreamCreate

    global __cuStreamCreateWithPriority
    data["__cuStreamCreateWithPriority"] = <intptr_t>__cuStreamCreateWithPriority

    global __cuStreamGetPriority
    data["__cuStreamGetPriority"] = <intptr_t>__cuStreamGetPriority

    global __cuStreamGetDevice
    data["__cuStreamGetDevice"] = <intptr_t>__cuStreamGetDevice

    global __cuStreamGetFlags
    data["__cuStreamGetFlags"] = <intptr_t>__cuStreamGetFlags

    global __cuStreamGetId
    data["__cuStreamGetId"] = <intptr_t>__cuStreamGetId

    global __cuStreamGetCtx
    data["__cuStreamGetCtx"] = <intptr_t>__cuStreamGetCtx

    global __cuStreamGetCtx_v2
    data["__cuStreamGetCtx_v2"] = <intptr_t>__cuStreamGetCtx_v2

    global __cuStreamWaitEvent
    data["__cuStreamWaitEvent"] = <intptr_t>__cuStreamWaitEvent

    global __cuStreamAddCallback
    data["__cuStreamAddCallback"] = <intptr_t>__cuStreamAddCallback

    global __cuStreamBeginCapture_v2
    data["__cuStreamBeginCapture_v2"] = <intptr_t>__cuStreamBeginCapture_v2

    global __cuStreamBeginCaptureToGraph
    data["__cuStreamBeginCaptureToGraph"] = <intptr_t>__cuStreamBeginCaptureToGraph

    global __cuThreadExchangeStreamCaptureMode
    data["__cuThreadExchangeStreamCaptureMode"] = <intptr_t>__cuThreadExchangeStreamCaptureMode

    global __cuStreamEndCapture
    data["__cuStreamEndCapture"] = <intptr_t>__cuStreamEndCapture

    global __cuStreamIsCapturing
    data["__cuStreamIsCapturing"] = <intptr_t>__cuStreamIsCapturing

    global __cuStreamGetCaptureInfo_v3
    data["__cuStreamGetCaptureInfo_v3"] = <intptr_t>__cuStreamGetCaptureInfo_v3

    global __cuStreamUpdateCaptureDependencies_v2
    data["__cuStreamUpdateCaptureDependencies_v2"] = <intptr_t>__cuStreamUpdateCaptureDependencies_v2

    global __cuStreamAttachMemAsync
    data["__cuStreamAttachMemAsync"] = <intptr_t>__cuStreamAttachMemAsync

    global __cuStreamQuery
    data["__cuStreamQuery"] = <intptr_t>__cuStreamQuery

    global __cuStreamSynchronize
    data["__cuStreamSynchronize"] = <intptr_t>__cuStreamSynchronize

    global __cuStreamDestroy_v2
    data["__cuStreamDestroy_v2"] = <intptr_t>__cuStreamDestroy_v2

    global __cuStreamCopyAttributes
    data["__cuStreamCopyAttributes"] = <intptr_t>__cuStreamCopyAttributes

    global __cuStreamGetAttribute
    data["__cuStreamGetAttribute"] = <intptr_t>__cuStreamGetAttribute

    global __cuStreamSetAttribute
    data["__cuStreamSetAttribute"] = <intptr_t>__cuStreamSetAttribute

    global __cuEventCreate
    data["__cuEventCreate"] = <intptr_t>__cuEventCreate

    global __cuEventRecord
    data["__cuEventRecord"] = <intptr_t>__cuEventRecord

    global __cuEventRecordWithFlags
    data["__cuEventRecordWithFlags"] = <intptr_t>__cuEventRecordWithFlags

    global __cuEventQuery
    data["__cuEventQuery"] = <intptr_t>__cuEventQuery

    global __cuEventSynchronize
    data["__cuEventSynchronize"] = <intptr_t>__cuEventSynchronize

    global __cuEventDestroy_v2
    data["__cuEventDestroy_v2"] = <intptr_t>__cuEventDestroy_v2

    global __cuEventElapsedTime_v2
    data["__cuEventElapsedTime_v2"] = <intptr_t>__cuEventElapsedTime_v2

    global __cuImportExternalMemory
    data["__cuImportExternalMemory"] = <intptr_t>__cuImportExternalMemory

    global __cuExternalMemoryGetMappedBuffer
    data["__cuExternalMemoryGetMappedBuffer"] = <intptr_t>__cuExternalMemoryGetMappedBuffer

    global __cuExternalMemoryGetMappedMipmappedArray
    data["__cuExternalMemoryGetMappedMipmappedArray"] = <intptr_t>__cuExternalMemoryGetMappedMipmappedArray

    global __cuDestroyExternalMemory
    data["__cuDestroyExternalMemory"] = <intptr_t>__cuDestroyExternalMemory

    global __cuImportExternalSemaphore
    data["__cuImportExternalSemaphore"] = <intptr_t>__cuImportExternalSemaphore

    global __cuSignalExternalSemaphoresAsync
    data["__cuSignalExternalSemaphoresAsync"] = <intptr_t>__cuSignalExternalSemaphoresAsync

    global __cuWaitExternalSemaphoresAsync
    data["__cuWaitExternalSemaphoresAsync"] = <intptr_t>__cuWaitExternalSemaphoresAsync

    global __cuDestroyExternalSemaphore
    data["__cuDestroyExternalSemaphore"] = <intptr_t>__cuDestroyExternalSemaphore

    global __cuStreamWaitValue32_v2
    data["__cuStreamWaitValue32_v2"] = <intptr_t>__cuStreamWaitValue32_v2

    global __cuStreamWaitValue64_v2
    data["__cuStreamWaitValue64_v2"] = <intptr_t>__cuStreamWaitValue64_v2

    global __cuStreamWriteValue32_v2
    data["__cuStreamWriteValue32_v2"] = <intptr_t>__cuStreamWriteValue32_v2

    global __cuStreamWriteValue64_v2
    data["__cuStreamWriteValue64_v2"] = <intptr_t>__cuStreamWriteValue64_v2

    global __cuStreamBatchMemOp_v2
    data["__cuStreamBatchMemOp_v2"] = <intptr_t>__cuStreamBatchMemOp_v2

    global __cuFuncGetAttribute
    data["__cuFuncGetAttribute"] = <intptr_t>__cuFuncGetAttribute

    global __cuFuncSetAttribute
    data["__cuFuncSetAttribute"] = <intptr_t>__cuFuncSetAttribute

    global __cuFuncSetCacheConfig
    data["__cuFuncSetCacheConfig"] = <intptr_t>__cuFuncSetCacheConfig

    global __cuFuncGetModule
    data["__cuFuncGetModule"] = <intptr_t>__cuFuncGetModule

    global __cuFuncGetName
    data["__cuFuncGetName"] = <intptr_t>__cuFuncGetName

    global __cuFuncGetParamInfo
    data["__cuFuncGetParamInfo"] = <intptr_t>__cuFuncGetParamInfo

    global __cuFuncIsLoaded
    data["__cuFuncIsLoaded"] = <intptr_t>__cuFuncIsLoaded

    global __cuFuncLoad
    data["__cuFuncLoad"] = <intptr_t>__cuFuncLoad

    global __cuLaunchKernel
    data["__cuLaunchKernel"] = <intptr_t>__cuLaunchKernel

    global __cuLaunchKernelEx
    data["__cuLaunchKernelEx"] = <intptr_t>__cuLaunchKernelEx

    global __cuLaunchCooperativeKernel
    data["__cuLaunchCooperativeKernel"] = <intptr_t>__cuLaunchCooperativeKernel

    global __cuLaunchCooperativeKernelMultiDevice
    data["__cuLaunchCooperativeKernelMultiDevice"] = <intptr_t>__cuLaunchCooperativeKernelMultiDevice

    global __cuLaunchHostFunc
    data["__cuLaunchHostFunc"] = <intptr_t>__cuLaunchHostFunc

    global __cuFuncSetBlockShape
    data["__cuFuncSetBlockShape"] = <intptr_t>__cuFuncSetBlockShape

    global __cuFuncSetSharedSize
    data["__cuFuncSetSharedSize"] = <intptr_t>__cuFuncSetSharedSize

    global __cuParamSetSize
    data["__cuParamSetSize"] = <intptr_t>__cuParamSetSize

    global __cuParamSeti
    data["__cuParamSeti"] = <intptr_t>__cuParamSeti

    global __cuParamSetf
    data["__cuParamSetf"] = <intptr_t>__cuParamSetf

    global __cuParamSetv
    data["__cuParamSetv"] = <intptr_t>__cuParamSetv

    global __cuLaunch
    data["__cuLaunch"] = <intptr_t>__cuLaunch

    global __cuLaunchGrid
    data["__cuLaunchGrid"] = <intptr_t>__cuLaunchGrid

    global __cuLaunchGridAsync
    data["__cuLaunchGridAsync"] = <intptr_t>__cuLaunchGridAsync

    global __cuParamSetTexRef
    data["__cuParamSetTexRef"] = <intptr_t>__cuParamSetTexRef

    global __cuFuncSetSharedMemConfig
    data["__cuFuncSetSharedMemConfig"] = <intptr_t>__cuFuncSetSharedMemConfig

    global __cuGraphCreate
    data["__cuGraphCreate"] = <intptr_t>__cuGraphCreate

    global __cuGraphAddKernelNode_v2
    data["__cuGraphAddKernelNode_v2"] = <intptr_t>__cuGraphAddKernelNode_v2

    global __cuGraphKernelNodeGetParams_v2
    data["__cuGraphKernelNodeGetParams_v2"] = <intptr_t>__cuGraphKernelNodeGetParams_v2

    global __cuGraphKernelNodeSetParams_v2
    data["__cuGraphKernelNodeSetParams_v2"] = <intptr_t>__cuGraphKernelNodeSetParams_v2

    global __cuGraphAddMemcpyNode
    data["__cuGraphAddMemcpyNode"] = <intptr_t>__cuGraphAddMemcpyNode

    global __cuGraphMemcpyNodeGetParams
    data["__cuGraphMemcpyNodeGetParams"] = <intptr_t>__cuGraphMemcpyNodeGetParams

    global __cuGraphMemcpyNodeSetParams
    data["__cuGraphMemcpyNodeSetParams"] = <intptr_t>__cuGraphMemcpyNodeSetParams

    global __cuGraphAddMemsetNode
    data["__cuGraphAddMemsetNode"] = <intptr_t>__cuGraphAddMemsetNode

    global __cuGraphMemsetNodeGetParams
    data["__cuGraphMemsetNodeGetParams"] = <intptr_t>__cuGraphMemsetNodeGetParams

    global __cuGraphMemsetNodeSetParams
    data["__cuGraphMemsetNodeSetParams"] = <intptr_t>__cuGraphMemsetNodeSetParams

    global __cuGraphAddHostNode
    data["__cuGraphAddHostNode"] = <intptr_t>__cuGraphAddHostNode

    global __cuGraphHostNodeGetParams
    data["__cuGraphHostNodeGetParams"] = <intptr_t>__cuGraphHostNodeGetParams

    global __cuGraphHostNodeSetParams
    data["__cuGraphHostNodeSetParams"] = <intptr_t>__cuGraphHostNodeSetParams

    global __cuGraphAddChildGraphNode
    data["__cuGraphAddChildGraphNode"] = <intptr_t>__cuGraphAddChildGraphNode

    global __cuGraphChildGraphNodeGetGraph
    data["__cuGraphChildGraphNodeGetGraph"] = <intptr_t>__cuGraphChildGraphNodeGetGraph

    global __cuGraphAddEmptyNode
    data["__cuGraphAddEmptyNode"] = <intptr_t>__cuGraphAddEmptyNode

    global __cuGraphAddEventRecordNode
    data["__cuGraphAddEventRecordNode"] = <intptr_t>__cuGraphAddEventRecordNode

    global __cuGraphEventRecordNodeGetEvent
    data["__cuGraphEventRecordNodeGetEvent"] = <intptr_t>__cuGraphEventRecordNodeGetEvent

    global __cuGraphEventRecordNodeSetEvent
    data["__cuGraphEventRecordNodeSetEvent"] = <intptr_t>__cuGraphEventRecordNodeSetEvent

    global __cuGraphAddEventWaitNode
    data["__cuGraphAddEventWaitNode"] = <intptr_t>__cuGraphAddEventWaitNode

    global __cuGraphEventWaitNodeGetEvent
    data["__cuGraphEventWaitNodeGetEvent"] = <intptr_t>__cuGraphEventWaitNodeGetEvent

    global __cuGraphEventWaitNodeSetEvent
    data["__cuGraphEventWaitNodeSetEvent"] = <intptr_t>__cuGraphEventWaitNodeSetEvent

    global __cuGraphAddExternalSemaphoresSignalNode
    data["__cuGraphAddExternalSemaphoresSignalNode"] = <intptr_t>__cuGraphAddExternalSemaphoresSignalNode

    global __cuGraphExternalSemaphoresSignalNodeGetParams
    data["__cuGraphExternalSemaphoresSignalNodeGetParams"] = <intptr_t>__cuGraphExternalSemaphoresSignalNodeGetParams

    global __cuGraphExternalSemaphoresSignalNodeSetParams
    data["__cuGraphExternalSemaphoresSignalNodeSetParams"] = <intptr_t>__cuGraphExternalSemaphoresSignalNodeSetParams

    global __cuGraphAddExternalSemaphoresWaitNode
    data["__cuGraphAddExternalSemaphoresWaitNode"] = <intptr_t>__cuGraphAddExternalSemaphoresWaitNode

    global __cuGraphExternalSemaphoresWaitNodeGetParams
    data["__cuGraphExternalSemaphoresWaitNodeGetParams"] = <intptr_t>__cuGraphExternalSemaphoresWaitNodeGetParams

    global __cuGraphExternalSemaphoresWaitNodeSetParams
    data["__cuGraphExternalSemaphoresWaitNodeSetParams"] = <intptr_t>__cuGraphExternalSemaphoresWaitNodeSetParams

    global __cuGraphAddBatchMemOpNode
    data["__cuGraphAddBatchMemOpNode"] = <intptr_t>__cuGraphAddBatchMemOpNode

    global __cuGraphBatchMemOpNodeGetParams
    data["__cuGraphBatchMemOpNodeGetParams"] = <intptr_t>__cuGraphBatchMemOpNodeGetParams

    global __cuGraphBatchMemOpNodeSetParams
    data["__cuGraphBatchMemOpNodeSetParams"] = <intptr_t>__cuGraphBatchMemOpNodeSetParams

    global __cuGraphExecBatchMemOpNodeSetParams
    data["__cuGraphExecBatchMemOpNodeSetParams"] = <intptr_t>__cuGraphExecBatchMemOpNodeSetParams

    global __cuGraphAddMemAllocNode
    data["__cuGraphAddMemAllocNode"] = <intptr_t>__cuGraphAddMemAllocNode

    global __cuGraphMemAllocNodeGetParams
    data["__cuGraphMemAllocNodeGetParams"] = <intptr_t>__cuGraphMemAllocNodeGetParams

    global __cuGraphAddMemFreeNode
    data["__cuGraphAddMemFreeNode"] = <intptr_t>__cuGraphAddMemFreeNode

    global __cuGraphMemFreeNodeGetParams
    data["__cuGraphMemFreeNodeGetParams"] = <intptr_t>__cuGraphMemFreeNodeGetParams

    global __cuDeviceGraphMemTrim
    data["__cuDeviceGraphMemTrim"] = <intptr_t>__cuDeviceGraphMemTrim

    global __cuDeviceGetGraphMemAttribute
    data["__cuDeviceGetGraphMemAttribute"] = <intptr_t>__cuDeviceGetGraphMemAttribute

    global __cuDeviceSetGraphMemAttribute
    data["__cuDeviceSetGraphMemAttribute"] = <intptr_t>__cuDeviceSetGraphMemAttribute

    global __cuGraphClone
    data["__cuGraphClone"] = <intptr_t>__cuGraphClone

    global __cuGraphNodeFindInClone
    data["__cuGraphNodeFindInClone"] = <intptr_t>__cuGraphNodeFindInClone

    global __cuGraphNodeGetType
    data["__cuGraphNodeGetType"] = <intptr_t>__cuGraphNodeGetType

    global __cuGraphGetNodes
    data["__cuGraphGetNodes"] = <intptr_t>__cuGraphGetNodes

    global __cuGraphGetRootNodes
    data["__cuGraphGetRootNodes"] = <intptr_t>__cuGraphGetRootNodes

    global __cuGraphGetEdges_v2
    data["__cuGraphGetEdges_v2"] = <intptr_t>__cuGraphGetEdges_v2

    global __cuGraphNodeGetDependencies_v2
    data["__cuGraphNodeGetDependencies_v2"] = <intptr_t>__cuGraphNodeGetDependencies_v2

    global __cuGraphNodeGetDependentNodes_v2
    data["__cuGraphNodeGetDependentNodes_v2"] = <intptr_t>__cuGraphNodeGetDependentNodes_v2

    global __cuGraphAddDependencies_v2
    data["__cuGraphAddDependencies_v2"] = <intptr_t>__cuGraphAddDependencies_v2

    global __cuGraphRemoveDependencies_v2
    data["__cuGraphRemoveDependencies_v2"] = <intptr_t>__cuGraphRemoveDependencies_v2

    global __cuGraphDestroyNode
    data["__cuGraphDestroyNode"] = <intptr_t>__cuGraphDestroyNode

    global __cuGraphInstantiateWithFlags
    data["__cuGraphInstantiateWithFlags"] = <intptr_t>__cuGraphInstantiateWithFlags

    global __cuGraphInstantiateWithParams
    data["__cuGraphInstantiateWithParams"] = <intptr_t>__cuGraphInstantiateWithParams

    global __cuGraphExecGetFlags
    data["__cuGraphExecGetFlags"] = <intptr_t>__cuGraphExecGetFlags

    global __cuGraphExecKernelNodeSetParams_v2
    data["__cuGraphExecKernelNodeSetParams_v2"] = <intptr_t>__cuGraphExecKernelNodeSetParams_v2

    global __cuGraphExecMemcpyNodeSetParams
    data["__cuGraphExecMemcpyNodeSetParams"] = <intptr_t>__cuGraphExecMemcpyNodeSetParams

    global __cuGraphExecMemsetNodeSetParams
    data["__cuGraphExecMemsetNodeSetParams"] = <intptr_t>__cuGraphExecMemsetNodeSetParams

    global __cuGraphExecHostNodeSetParams
    data["__cuGraphExecHostNodeSetParams"] = <intptr_t>__cuGraphExecHostNodeSetParams

    global __cuGraphExecChildGraphNodeSetParams
    data["__cuGraphExecChildGraphNodeSetParams"] = <intptr_t>__cuGraphExecChildGraphNodeSetParams

    global __cuGraphExecEventRecordNodeSetEvent
    data["__cuGraphExecEventRecordNodeSetEvent"] = <intptr_t>__cuGraphExecEventRecordNodeSetEvent

    global __cuGraphExecEventWaitNodeSetEvent
    data["__cuGraphExecEventWaitNodeSetEvent"] = <intptr_t>__cuGraphExecEventWaitNodeSetEvent

    global __cuGraphExecExternalSemaphoresSignalNodeSetParams
    data["__cuGraphExecExternalSemaphoresSignalNodeSetParams"] = <intptr_t>__cuGraphExecExternalSemaphoresSignalNodeSetParams

    global __cuGraphExecExternalSemaphoresWaitNodeSetParams
    data["__cuGraphExecExternalSemaphoresWaitNodeSetParams"] = <intptr_t>__cuGraphExecExternalSemaphoresWaitNodeSetParams

    global __cuGraphNodeSetEnabled
    data["__cuGraphNodeSetEnabled"] = <intptr_t>__cuGraphNodeSetEnabled

    global __cuGraphNodeGetEnabled
    data["__cuGraphNodeGetEnabled"] = <intptr_t>__cuGraphNodeGetEnabled

    global __cuGraphUpload
    data["__cuGraphUpload"] = <intptr_t>__cuGraphUpload

    global __cuGraphLaunch
    data["__cuGraphLaunch"] = <intptr_t>__cuGraphLaunch

    global __cuGraphExecDestroy
    data["__cuGraphExecDestroy"] = <intptr_t>__cuGraphExecDestroy

    global __cuGraphDestroy
    data["__cuGraphDestroy"] = <intptr_t>__cuGraphDestroy

    global __cuGraphExecUpdate_v2
    data["__cuGraphExecUpdate_v2"] = <intptr_t>__cuGraphExecUpdate_v2

    global __cuGraphKernelNodeCopyAttributes
    data["__cuGraphKernelNodeCopyAttributes"] = <intptr_t>__cuGraphKernelNodeCopyAttributes

    global __cuGraphKernelNodeGetAttribute
    data["__cuGraphKernelNodeGetAttribute"] = <intptr_t>__cuGraphKernelNodeGetAttribute

    global __cuGraphKernelNodeSetAttribute
    data["__cuGraphKernelNodeSetAttribute"] = <intptr_t>__cuGraphKernelNodeSetAttribute

    global __cuGraphDebugDotPrint
    data["__cuGraphDebugDotPrint"] = <intptr_t>__cuGraphDebugDotPrint

    global __cuUserObjectCreate
    data["__cuUserObjectCreate"] = <intptr_t>__cuUserObjectCreate

    global __cuUserObjectRetain
    data["__cuUserObjectRetain"] = <intptr_t>__cuUserObjectRetain

    global __cuUserObjectRelease
    data["__cuUserObjectRelease"] = <intptr_t>__cuUserObjectRelease

    global __cuGraphRetainUserObject
    data["__cuGraphRetainUserObject"] = <intptr_t>__cuGraphRetainUserObject

    global __cuGraphReleaseUserObject
    data["__cuGraphReleaseUserObject"] = <intptr_t>__cuGraphReleaseUserObject

    global __cuGraphAddNode_v2
    data["__cuGraphAddNode_v2"] = <intptr_t>__cuGraphAddNode_v2

    global __cuGraphNodeSetParams
    data["__cuGraphNodeSetParams"] = <intptr_t>__cuGraphNodeSetParams

    global __cuGraphExecNodeSetParams
    data["__cuGraphExecNodeSetParams"] = <intptr_t>__cuGraphExecNodeSetParams

    global __cuGraphConditionalHandleCreate
    data["__cuGraphConditionalHandleCreate"] = <intptr_t>__cuGraphConditionalHandleCreate

    global __cuOccupancyMaxActiveBlocksPerMultiprocessor
    data["__cuOccupancyMaxActiveBlocksPerMultiprocessor"] = <intptr_t>__cuOccupancyMaxActiveBlocksPerMultiprocessor

    global __cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags
    data["__cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags"] = <intptr_t>__cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags

    global __cuOccupancyMaxPotentialBlockSize
    data["__cuOccupancyMaxPotentialBlockSize"] = <intptr_t>__cuOccupancyMaxPotentialBlockSize

    global __cuOccupancyMaxPotentialBlockSizeWithFlags
    data["__cuOccupancyMaxPotentialBlockSizeWithFlags"] = <intptr_t>__cuOccupancyMaxPotentialBlockSizeWithFlags

    global __cuOccupancyAvailableDynamicSMemPerBlock
    data["__cuOccupancyAvailableDynamicSMemPerBlock"] = <intptr_t>__cuOccupancyAvailableDynamicSMemPerBlock

    global __cuOccupancyMaxPotentialClusterSize
    data["__cuOccupancyMaxPotentialClusterSize"] = <intptr_t>__cuOccupancyMaxPotentialClusterSize

    global __cuOccupancyMaxActiveClusters
    data["__cuOccupancyMaxActiveClusters"] = <intptr_t>__cuOccupancyMaxActiveClusters

    global __cuTexRefSetArray
    data["__cuTexRefSetArray"] = <intptr_t>__cuTexRefSetArray

    global __cuTexRefSetMipmappedArray
    data["__cuTexRefSetMipmappedArray"] = <intptr_t>__cuTexRefSetMipmappedArray

    global __cuTexRefSetAddress_v2
    data["__cuTexRefSetAddress_v2"] = <intptr_t>__cuTexRefSetAddress_v2

    global __cuTexRefSetAddress2D_v3
    data["__cuTexRefSetAddress2D_v3"] = <intptr_t>__cuTexRefSetAddress2D_v3

    global __cuTexRefSetFormat
    data["__cuTexRefSetFormat"] = <intptr_t>__cuTexRefSetFormat

    global __cuTexRefSetAddressMode
    data["__cuTexRefSetAddressMode"] = <intptr_t>__cuTexRefSetAddressMode

    global __cuTexRefSetFilterMode
    data["__cuTexRefSetFilterMode"] = <intptr_t>__cuTexRefSetFilterMode

    global __cuTexRefSetMipmapFilterMode
    data["__cuTexRefSetMipmapFilterMode"] = <intptr_t>__cuTexRefSetMipmapFilterMode

    global __cuTexRefSetMipmapLevelBias
    data["__cuTexRefSetMipmapLevelBias"] = <intptr_t>__cuTexRefSetMipmapLevelBias

    global __cuTexRefSetMipmapLevelClamp
    data["__cuTexRefSetMipmapLevelClamp"] = <intptr_t>__cuTexRefSetMipmapLevelClamp

    global __cuTexRefSetMaxAnisotropy
    data["__cuTexRefSetMaxAnisotropy"] = <intptr_t>__cuTexRefSetMaxAnisotropy

    global __cuTexRefSetBorderColor
    data["__cuTexRefSetBorderColor"] = <intptr_t>__cuTexRefSetBorderColor

    global __cuTexRefSetFlags
    data["__cuTexRefSetFlags"] = <intptr_t>__cuTexRefSetFlags

    global __cuTexRefGetAddress_v2
    data["__cuTexRefGetAddress_v2"] = <intptr_t>__cuTexRefGetAddress_v2

    global __cuTexRefGetArray
    data["__cuTexRefGetArray"] = <intptr_t>__cuTexRefGetArray

    global __cuTexRefGetMipmappedArray
    data["__cuTexRefGetMipmappedArray"] = <intptr_t>__cuTexRefGetMipmappedArray

    global __cuTexRefGetAddressMode
    data["__cuTexRefGetAddressMode"] = <intptr_t>__cuTexRefGetAddressMode

    global __cuTexRefGetFilterMode
    data["__cuTexRefGetFilterMode"] = <intptr_t>__cuTexRefGetFilterMode

    global __cuTexRefGetFormat
    data["__cuTexRefGetFormat"] = <intptr_t>__cuTexRefGetFormat

    global __cuTexRefGetMipmapFilterMode
    data["__cuTexRefGetMipmapFilterMode"] = <intptr_t>__cuTexRefGetMipmapFilterMode

    global __cuTexRefGetMipmapLevelBias
    data["__cuTexRefGetMipmapLevelBias"] = <intptr_t>__cuTexRefGetMipmapLevelBias

    global __cuTexRefGetMipmapLevelClamp
    data["__cuTexRefGetMipmapLevelClamp"] = <intptr_t>__cuTexRefGetMipmapLevelClamp

    global __cuTexRefGetMaxAnisotropy
    data["__cuTexRefGetMaxAnisotropy"] = <intptr_t>__cuTexRefGetMaxAnisotropy

    global __cuTexRefGetBorderColor
    data["__cuTexRefGetBorderColor"] = <intptr_t>__cuTexRefGetBorderColor

    global __cuTexRefGetFlags
    data["__cuTexRefGetFlags"] = <intptr_t>__cuTexRefGetFlags

    global __cuTexRefCreate
    data["__cuTexRefCreate"] = <intptr_t>__cuTexRefCreate

    global __cuTexRefDestroy
    data["__cuTexRefDestroy"] = <intptr_t>__cuTexRefDestroy

    global __cuSurfRefSetArray
    data["__cuSurfRefSetArray"] = <intptr_t>__cuSurfRefSetArray

    global __cuSurfRefGetArray
    data["__cuSurfRefGetArray"] = <intptr_t>__cuSurfRefGetArray

    global __cuTexObjectCreate
    data["__cuTexObjectCreate"] = <intptr_t>__cuTexObjectCreate

    global __cuTexObjectDestroy
    data["__cuTexObjectDestroy"] = <intptr_t>__cuTexObjectDestroy

    global __cuTexObjectGetResourceDesc
    data["__cuTexObjectGetResourceDesc"] = <intptr_t>__cuTexObjectGetResourceDesc

    global __cuTexObjectGetTextureDesc
    data["__cuTexObjectGetTextureDesc"] = <intptr_t>__cuTexObjectGetTextureDesc

    global __cuTexObjectGetResourceViewDesc
    data["__cuTexObjectGetResourceViewDesc"] = <intptr_t>__cuTexObjectGetResourceViewDesc

    global __cuSurfObjectCreate
    data["__cuSurfObjectCreate"] = <intptr_t>__cuSurfObjectCreate

    global __cuSurfObjectDestroy
    data["__cuSurfObjectDestroy"] = <intptr_t>__cuSurfObjectDestroy

    global __cuSurfObjectGetResourceDesc
    data["__cuSurfObjectGetResourceDesc"] = <intptr_t>__cuSurfObjectGetResourceDesc

    global __cuTensorMapEncodeTiled
    data["__cuTensorMapEncodeTiled"] = <intptr_t>__cuTensorMapEncodeTiled

    global __cuTensorMapEncodeIm2col
    data["__cuTensorMapEncodeIm2col"] = <intptr_t>__cuTensorMapEncodeIm2col

    global __cuTensorMapEncodeIm2colWide
    data["__cuTensorMapEncodeIm2colWide"] = <intptr_t>__cuTensorMapEncodeIm2colWide

    global __cuTensorMapReplaceAddress
    data["__cuTensorMapReplaceAddress"] = <intptr_t>__cuTensorMapReplaceAddress

    global __cuDeviceCanAccessPeer
    data["__cuDeviceCanAccessPeer"] = <intptr_t>__cuDeviceCanAccessPeer

    global __cuCtxEnablePeerAccess
    data["__cuCtxEnablePeerAccess"] = <intptr_t>__cuCtxEnablePeerAccess

    global __cuCtxDisablePeerAccess
    data["__cuCtxDisablePeerAccess"] = <intptr_t>__cuCtxDisablePeerAccess

    global __cuDeviceGetP2PAttribute
    data["__cuDeviceGetP2PAttribute"] = <intptr_t>__cuDeviceGetP2PAttribute

    global __cuGraphicsUnregisterResource
    data["__cuGraphicsUnregisterResource"] = <intptr_t>__cuGraphicsUnregisterResource

    global __cuGraphicsSubResourceGetMappedArray
    data["__cuGraphicsSubResourceGetMappedArray"] = <intptr_t>__cuGraphicsSubResourceGetMappedArray

    global __cuGraphicsResourceGetMappedMipmappedArray
    data["__cuGraphicsResourceGetMappedMipmappedArray"] = <intptr_t>__cuGraphicsResourceGetMappedMipmappedArray

    global __cuGraphicsResourceGetMappedPointer_v2
    data["__cuGraphicsResourceGetMappedPointer_v2"] = <intptr_t>__cuGraphicsResourceGetMappedPointer_v2

    global __cuGraphicsResourceSetMapFlags_v2
    data["__cuGraphicsResourceSetMapFlags_v2"] = <intptr_t>__cuGraphicsResourceSetMapFlags_v2

    global __cuGraphicsMapResources
    data["__cuGraphicsMapResources"] = <intptr_t>__cuGraphicsMapResources

    global __cuGraphicsUnmapResources
    data["__cuGraphicsUnmapResources"] = <intptr_t>__cuGraphicsUnmapResources

    global __cuGetProcAddress_v2
    data["__cuGetProcAddress_v2"] = <intptr_t>__cuGetProcAddress_v2

    global __cuCoredumpGetAttribute
    data["__cuCoredumpGetAttribute"] = <intptr_t>__cuCoredumpGetAttribute

    global __cuCoredumpGetAttributeGlobal
    data["__cuCoredumpGetAttributeGlobal"] = <intptr_t>__cuCoredumpGetAttributeGlobal

    global __cuCoredumpSetAttribute
    data["__cuCoredumpSetAttribute"] = <intptr_t>__cuCoredumpSetAttribute

    global __cuCoredumpSetAttributeGlobal
    data["__cuCoredumpSetAttributeGlobal"] = <intptr_t>__cuCoredumpSetAttributeGlobal

    global __cuGetExportTable
    data["__cuGetExportTable"] = <intptr_t>__cuGetExportTable

    global __cuGreenCtxCreate
    data["__cuGreenCtxCreate"] = <intptr_t>__cuGreenCtxCreate

    global __cuGreenCtxDestroy
    data["__cuGreenCtxDestroy"] = <intptr_t>__cuGreenCtxDestroy

    global __cuCtxFromGreenCtx
    data["__cuCtxFromGreenCtx"] = <intptr_t>__cuCtxFromGreenCtx

    global __cuDeviceGetDevResource
    data["__cuDeviceGetDevResource"] = <intptr_t>__cuDeviceGetDevResource

    global __cuCtxGetDevResource
    data["__cuCtxGetDevResource"] = <intptr_t>__cuCtxGetDevResource

    global __cuGreenCtxGetDevResource
    data["__cuGreenCtxGetDevResource"] = <intptr_t>__cuGreenCtxGetDevResource

    global __cuDevSmResourceSplitByCount
    data["__cuDevSmResourceSplitByCount"] = <intptr_t>__cuDevSmResourceSplitByCount

    global __cuDevResourceGenerateDesc
    data["__cuDevResourceGenerateDesc"] = <intptr_t>__cuDevResourceGenerateDesc

    global __cuGreenCtxRecordEvent
    data["__cuGreenCtxRecordEvent"] = <intptr_t>__cuGreenCtxRecordEvent

    global __cuGreenCtxWaitEvent
    data["__cuGreenCtxWaitEvent"] = <intptr_t>__cuGreenCtxWaitEvent

    global __cuStreamGetGreenCtx
    data["__cuStreamGetGreenCtx"] = <intptr_t>__cuStreamGetGreenCtx

    global __cuGreenCtxStreamCreate
    data["__cuGreenCtxStreamCreate"] = <intptr_t>__cuGreenCtxStreamCreate

    global __cuLogsRegisterCallback
    data["__cuLogsRegisterCallback"] = <intptr_t>__cuLogsRegisterCallback

    global __cuLogsUnregisterCallback
    data["__cuLogsUnregisterCallback"] = <intptr_t>__cuLogsUnregisterCallback

    global __cuLogsCurrent
    data["__cuLogsCurrent"] = <intptr_t>__cuLogsCurrent

    global __cuLogsDumpToFile
    data["__cuLogsDumpToFile"] = <intptr_t>__cuLogsDumpToFile

    global __cuLogsDumpToMemory
    data["__cuLogsDumpToMemory"] = <intptr_t>__cuLogsDumpToMemory

    global __cuCheckpointProcessGetRestoreThreadId
    data["__cuCheckpointProcessGetRestoreThreadId"] = <intptr_t>__cuCheckpointProcessGetRestoreThreadId

    global __cuCheckpointProcessGetState
    data["__cuCheckpointProcessGetState"] = <intptr_t>__cuCheckpointProcessGetState

    global __cuCheckpointProcessLock
    data["__cuCheckpointProcessLock"] = <intptr_t>__cuCheckpointProcessLock

    global __cuCheckpointProcessCheckpoint
    data["__cuCheckpointProcessCheckpoint"] = <intptr_t>__cuCheckpointProcessCheckpoint

    global __cuCheckpointProcessRestore
    data["__cuCheckpointProcessRestore"] = <intptr_t>__cuCheckpointProcessRestore

    global __cuCheckpointProcessUnlock
    data["__cuCheckpointProcessUnlock"] = <intptr_t>__cuCheckpointProcessUnlock

    global __cuGraphicsEGLRegisterImage
    data["__cuGraphicsEGLRegisterImage"] = <intptr_t>__cuGraphicsEGLRegisterImage

    global __cuEGLStreamConsumerConnect
    data["__cuEGLStreamConsumerConnect"] = <intptr_t>__cuEGLStreamConsumerConnect

    global __cuEGLStreamConsumerConnectWithFlags
    data["__cuEGLStreamConsumerConnectWithFlags"] = <intptr_t>__cuEGLStreamConsumerConnectWithFlags

    global __cuEGLStreamConsumerDisconnect
    data["__cuEGLStreamConsumerDisconnect"] = <intptr_t>__cuEGLStreamConsumerDisconnect

    global __cuEGLStreamConsumerAcquireFrame
    data["__cuEGLStreamConsumerAcquireFrame"] = <intptr_t>__cuEGLStreamConsumerAcquireFrame

    global __cuEGLStreamConsumerReleaseFrame
    data["__cuEGLStreamConsumerReleaseFrame"] = <intptr_t>__cuEGLStreamConsumerReleaseFrame

    global __cuEGLStreamProducerConnect
    data["__cuEGLStreamProducerConnect"] = <intptr_t>__cuEGLStreamProducerConnect

    global __cuEGLStreamProducerDisconnect
    data["__cuEGLStreamProducerDisconnect"] = <intptr_t>__cuEGLStreamProducerDisconnect

    global __cuEGLStreamProducerPresentFrame
    data["__cuEGLStreamProducerPresentFrame"] = <intptr_t>__cuEGLStreamProducerPresentFrame

    global __cuEGLStreamProducerReturnFrame
    data["__cuEGLStreamProducerReturnFrame"] = <intptr_t>__cuEGLStreamProducerReturnFrame

    global __cuGraphicsResourceGetMappedEglFrame
    data["__cuGraphicsResourceGetMappedEglFrame"] = <intptr_t>__cuGraphicsResourceGetMappedEglFrame

    global __cuEventCreateFromEGLSync
    data["__cuEventCreateFromEGLSync"] = <intptr_t>__cuEventCreateFromEGLSync

    global __cuGraphicsGLRegisterBuffer
    data["__cuGraphicsGLRegisterBuffer"] = <intptr_t>__cuGraphicsGLRegisterBuffer

    global __cuGraphicsGLRegisterImage
    data["__cuGraphicsGLRegisterImage"] = <intptr_t>__cuGraphicsGLRegisterImage

    global __cuGLGetDevices_v2
    data["__cuGLGetDevices_v2"] = <intptr_t>__cuGLGetDevices_v2

    global __cuGLCtxCreate_v2
    data["__cuGLCtxCreate_v2"] = <intptr_t>__cuGLCtxCreate_v2

    global __cuGLInit
    data["__cuGLInit"] = <intptr_t>__cuGLInit

    global __cuGLRegisterBufferObject
    data["__cuGLRegisterBufferObject"] = <intptr_t>__cuGLRegisterBufferObject

    global __cuGLMapBufferObject_v2
    data["__cuGLMapBufferObject_v2"] = <intptr_t>__cuGLMapBufferObject_v2

    global __cuGLUnmapBufferObject
    data["__cuGLUnmapBufferObject"] = <intptr_t>__cuGLUnmapBufferObject

    global __cuGLUnregisterBufferObject
    data["__cuGLUnregisterBufferObject"] = <intptr_t>__cuGLUnregisterBufferObject

    global __cuGLSetBufferObjectMapFlags
    data["__cuGLSetBufferObjectMapFlags"] = <intptr_t>__cuGLSetBufferObjectMapFlags

    global __cuGLMapBufferObjectAsync_v2
    data["__cuGLMapBufferObjectAsync_v2"] = <intptr_t>__cuGLMapBufferObjectAsync_v2

    global __cuGLUnmapBufferObjectAsync
    data["__cuGLUnmapBufferObjectAsync"] = <intptr_t>__cuGLUnmapBufferObjectAsync

    global __cuProfilerInitialize
    data["__cuProfilerInitialize"] = <intptr_t>__cuProfilerInitialize

    global __cuProfilerStart
    data["__cuProfilerStart"] = <intptr_t>__cuProfilerStart

    global __cuProfilerStop
    data["__cuProfilerStop"] = <intptr_t>__cuProfilerStop

    global __cuVDPAUGetDevice
    data["__cuVDPAUGetDevice"] = <intptr_t>__cuVDPAUGetDevice

    global __cuVDPAUCtxCreate_v2
    data["__cuVDPAUCtxCreate_v2"] = <intptr_t>__cuVDPAUCtxCreate_v2

    global __cuGraphicsVDPAURegisterVideoSurface
    data["__cuGraphicsVDPAURegisterVideoSurface"] = <intptr_t>__cuGraphicsVDPAURegisterVideoSurface

    global __cuGraphicsVDPAURegisterOutputSurface
    data["__cuGraphicsVDPAURegisterOutputSurface"] = <intptr_t>__cuGraphicsVDPAURegisterOutputSurface

    global __cuDeviceGetHostAtomicCapabilities
    data["__cuDeviceGetHostAtomicCapabilities"] = <intptr_t>__cuDeviceGetHostAtomicCapabilities

    global __cuCtxGetDevice_v2
    data["__cuCtxGetDevice_v2"] = <intptr_t>__cuCtxGetDevice_v2

    global __cuCtxSynchronize_v2
    data["__cuCtxSynchronize_v2"] = <intptr_t>__cuCtxSynchronize_v2

    global __cuMemcpyBatchAsync_v2
    data["__cuMemcpyBatchAsync_v2"] = <intptr_t>__cuMemcpyBatchAsync_v2

    global __cuMemcpy3DBatchAsync_v2
    data["__cuMemcpy3DBatchAsync_v2"] = <intptr_t>__cuMemcpy3DBatchAsync_v2

    global __cuMemGetDefaultMemPool
    data["__cuMemGetDefaultMemPool"] = <intptr_t>__cuMemGetDefaultMemPool

    global __cuMemGetMemPool
    data["__cuMemGetMemPool"] = <intptr_t>__cuMemGetMemPool

    global __cuMemSetMemPool
    data["__cuMemSetMemPool"] = <intptr_t>__cuMemSetMemPool

    global __cuMemPrefetchBatchAsync
    data["__cuMemPrefetchBatchAsync"] = <intptr_t>__cuMemPrefetchBatchAsync

    global __cuMemDiscardBatchAsync
    data["__cuMemDiscardBatchAsync"] = <intptr_t>__cuMemDiscardBatchAsync

    global __cuMemDiscardAndPrefetchBatchAsync
    data["__cuMemDiscardAndPrefetchBatchAsync"] = <intptr_t>__cuMemDiscardAndPrefetchBatchAsync

    global __cuDeviceGetP2PAtomicCapabilities
    data["__cuDeviceGetP2PAtomicCapabilities"] = <intptr_t>__cuDeviceGetP2PAtomicCapabilities

    global __cuGreenCtxGetId
    data["__cuGreenCtxGetId"] = <intptr_t>__cuGreenCtxGetId

    global __cuMulticastBindMem_v2
    data["__cuMulticastBindMem_v2"] = <intptr_t>__cuMulticastBindMem_v2

    global __cuMulticastBindAddr_v2
    data["__cuMulticastBindAddr_v2"] = <intptr_t>__cuMulticastBindAddr_v2

    global __cuGraphNodeGetContainingGraph
    data["__cuGraphNodeGetContainingGraph"] = <intptr_t>__cuGraphNodeGetContainingGraph

    global __cuGraphNodeGetLocalId
    data["__cuGraphNodeGetLocalId"] = <intptr_t>__cuGraphNodeGetLocalId

    global __cuGraphNodeGetToolsId
    data["__cuGraphNodeGetToolsId"] = <intptr_t>__cuGraphNodeGetToolsId

    global __cuGraphGetId
    data["__cuGraphGetId"] = <intptr_t>__cuGraphGetId

    global __cuGraphExecGetId
    data["__cuGraphExecGetId"] = <intptr_t>__cuGraphExecGetId

    global __cuDevSmResourceSplit
    data["__cuDevSmResourceSplit"] = <intptr_t>__cuDevSmResourceSplit

    global __cuStreamGetDevResource
    data["__cuStreamGetDevResource"] = <intptr_t>__cuStreamGetDevResource

    global __cuKernelGetParamCount
    data["__cuKernelGetParamCount"] = <intptr_t>__cuKernelGetParamCount

    global __cuMemcpyWithAttributesAsync
    data["__cuMemcpyWithAttributesAsync"] = <intptr_t>__cuMemcpyWithAttributesAsync

    global __cuMemcpy3DWithAttributesAsync
    data["__cuMemcpy3DWithAttributesAsync"] = <intptr_t>__cuMemcpy3DWithAttributesAsync

    global __cuStreamBeginCaptureToCig
    data["__cuStreamBeginCaptureToCig"] = <intptr_t>__cuStreamBeginCaptureToCig

    global __cuStreamEndCaptureToCig
    data["__cuStreamEndCaptureToCig"] = <intptr_t>__cuStreamEndCaptureToCig

    global __cuFuncGetParamCount
    data["__cuFuncGetParamCount"] = <intptr_t>__cuFuncGetParamCount

    global __cuLaunchHostFunc_v2
    data["__cuLaunchHostFunc_v2"] = <intptr_t>__cuLaunchHostFunc_v2

    global __cuGraphNodeGetParams
    data["__cuGraphNodeGetParams"] = <intptr_t>__cuGraphNodeGetParams

    global __cuCoredumpRegisterStartCallback
    data["__cuCoredumpRegisterStartCallback"] = <intptr_t>__cuCoredumpRegisterStartCallback

    global __cuCoredumpRegisterCompleteCallback
    data["__cuCoredumpRegisterCompleteCallback"] = <intptr_t>__cuCoredumpRegisterCompleteCallback

    global __cuCoredumpDeregisterStartCallback
    data["__cuCoredumpDeregisterStartCallback"] = <intptr_t>__cuCoredumpDeregisterStartCallback

    global __cuCoredumpDeregisterCompleteCallback
    data["__cuCoredumpDeregisterCompleteCallback"] = <intptr_t>__cuCoredumpDeregisterCompleteCallback

    func_ptrs = data
    return data


cpdef _inspect_function_pointer(str name):
    global func_ptrs
    if func_ptrs is None:
        func_ptrs = _inspect_function_pointers()
    return func_ptrs[name]


###############################################################################
# Wrapper functions
###############################################################################

cdef CUresult _cuGetErrorString(CUresult error, const char** pStr) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuGetErrorString
    _check_or_init_driver()
    if __cuGetErrorString == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGetErrorString is not found")
    return (<CUresult (*)(CUresult, const char**) noexcept nogil>__cuGetErrorString)(
        error, pStr)


cdef CUresult _cuGetErrorName(CUresult error, const char** pStr) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuGetErrorName
    _check_or_init_driver()
    if __cuGetErrorName == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGetErrorName is not found")
    return (<CUresult (*)(CUresult, const char**) noexcept nogil>__cuGetErrorName)(
        error, pStr)


cdef CUresult _cuInit(unsigned int Flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuInit
    _check_or_init_driver()
    if __cuInit == NULL:
        with gil:
            raise FunctionNotFoundError("function cuInit is not found")
    return (<CUresult (*)(unsigned int) noexcept nogil>__cuInit)(
        Flags)


cdef CUresult _cuDriverGetVersion(int* driverVersion) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuDriverGetVersion
    _check_or_init_driver()
    if __cuDriverGetVersion == NULL:
        with gil:
            raise FunctionNotFoundError("function cuDriverGetVersion is not found")
    return (<CUresult (*)(int*) noexcept nogil>__cuDriverGetVersion)(
        driverVersion)


cdef CUresult _cuDeviceGet(CUdevice* device, int ordinal) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuDeviceGet
    _check_or_init_driver()
    if __cuDeviceGet == NULL:
        with gil:
            raise FunctionNotFoundError("function cuDeviceGet is not found")
    return (<CUresult (*)(CUdevice*, int) noexcept nogil>__cuDeviceGet)(
        device, ordinal)


cdef CUresult _cuDeviceGetCount(int* count) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuDeviceGetCount
    _check_or_init_driver()
    if __cuDeviceGetCount == NULL:
        with gil:
            raise FunctionNotFoundError("function cuDeviceGetCount is not found")
    return (<CUresult (*)(int*) noexcept nogil>__cuDeviceGetCount)(
        count)


cdef CUresult _cuDeviceGetName(char* name, int len, CUdevice dev) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuDeviceGetName
    _check_or_init_driver()
    if __cuDeviceGetName == NULL:
        with gil:
            raise FunctionNotFoundError("function cuDeviceGetName is not found")
    return (<CUresult (*)(char*, int, CUdevice) noexcept nogil>__cuDeviceGetName)(
        name, len, dev)


cdef CUresult _cuDeviceGetUuid_v2(CUuuid* uuid, CUdevice dev) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuDeviceGetUuid_v2
    _check_or_init_driver()
    if __cuDeviceGetUuid_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuDeviceGetUuid_v2 is not found")
    return (<CUresult (*)(CUuuid*, CUdevice) noexcept nogil>__cuDeviceGetUuid_v2)(
        uuid, dev)


cdef CUresult _cuDeviceGetLuid(char* luid, unsigned int* deviceNodeMask, CUdevice dev) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuDeviceGetLuid
    _check_or_init_driver()
    if __cuDeviceGetLuid == NULL:
        with gil:
            raise FunctionNotFoundError("function cuDeviceGetLuid is not found")
    return (<CUresult (*)(char*, unsigned int*, CUdevice) noexcept nogil>__cuDeviceGetLuid)(
        luid, deviceNodeMask, dev)


cdef CUresult _cuDeviceTotalMem_v2(size_t* bytes, CUdevice dev) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuDeviceTotalMem_v2
    _check_or_init_driver()
    if __cuDeviceTotalMem_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuDeviceTotalMem_v2 is not found")
    return (<CUresult (*)(size_t*, CUdevice) noexcept nogil>__cuDeviceTotalMem_v2)(
        bytes, dev)


cdef CUresult _cuDeviceGetTexture1DLinearMaxWidth(size_t* maxWidthInElements, CUarray_format format, unsigned numChannels, CUdevice dev) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuDeviceGetTexture1DLinearMaxWidth
    _check_or_init_driver()
    if __cuDeviceGetTexture1DLinearMaxWidth == NULL:
        with gil:
            raise FunctionNotFoundError("function cuDeviceGetTexture1DLinearMaxWidth is not found")
    return (<CUresult (*)(size_t*, CUarray_format, unsigned, CUdevice) noexcept nogil>__cuDeviceGetTexture1DLinearMaxWidth)(
        maxWidthInElements, format, numChannels, dev)


cdef CUresult _cuDeviceGetAttribute(int* pi, CUdevice_attribute attrib, CUdevice dev) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuDeviceGetAttribute
    _check_or_init_driver()
    if __cuDeviceGetAttribute == NULL:
        with gil:
            raise FunctionNotFoundError("function cuDeviceGetAttribute is not found")
    return (<CUresult (*)(int*, CUdevice_attribute, CUdevice) noexcept nogil>__cuDeviceGetAttribute)(
        pi, attrib, dev)


cdef CUresult _cuDeviceGetNvSciSyncAttributes(void* nvSciSyncAttrList, CUdevice dev, int flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuDeviceGetNvSciSyncAttributes
    _check_or_init_driver()
    if __cuDeviceGetNvSciSyncAttributes == NULL:
        with gil:
            raise FunctionNotFoundError("function cuDeviceGetNvSciSyncAttributes is not found")
    return (<CUresult (*)(void*, CUdevice, int) noexcept nogil>__cuDeviceGetNvSciSyncAttributes)(
        nvSciSyncAttrList, dev, flags)


cdef CUresult _cuDeviceSetMemPool(CUdevice dev, CUmemoryPool pool) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuDeviceSetMemPool
    _check_or_init_driver()
    if __cuDeviceSetMemPool == NULL:
        with gil:
            raise FunctionNotFoundError("function cuDeviceSetMemPool is not found")
    return (<CUresult (*)(CUdevice, CUmemoryPool) noexcept nogil>__cuDeviceSetMemPool)(
        dev, pool)


cdef CUresult _cuDeviceGetMemPool(CUmemoryPool* pool, CUdevice dev) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuDeviceGetMemPool
    _check_or_init_driver()
    if __cuDeviceGetMemPool == NULL:
        with gil:
            raise FunctionNotFoundError("function cuDeviceGetMemPool is not found")
    return (<CUresult (*)(CUmemoryPool*, CUdevice) noexcept nogil>__cuDeviceGetMemPool)(
        pool, dev)


cdef CUresult _cuDeviceGetDefaultMemPool(CUmemoryPool* pool_out, CUdevice dev) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuDeviceGetDefaultMemPool
    _check_or_init_driver()
    if __cuDeviceGetDefaultMemPool == NULL:
        with gil:
            raise FunctionNotFoundError("function cuDeviceGetDefaultMemPool is not found")
    return (<CUresult (*)(CUmemoryPool*, CUdevice) noexcept nogil>__cuDeviceGetDefaultMemPool)(
        pool_out, dev)


cdef CUresult _cuDeviceGetExecAffinitySupport(int* pi, CUexecAffinityType type, CUdevice dev) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuDeviceGetExecAffinitySupport
    _check_or_init_driver()
    if __cuDeviceGetExecAffinitySupport == NULL:
        with gil:
            raise FunctionNotFoundError("function cuDeviceGetExecAffinitySupport is not found")
    return (<CUresult (*)(int*, CUexecAffinityType, CUdevice) noexcept nogil>__cuDeviceGetExecAffinitySupport)(
        pi, type, dev)


cdef CUresult _cuFlushGPUDirectRDMAWrites(CUflushGPUDirectRDMAWritesTarget target, CUflushGPUDirectRDMAWritesScope scope) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuFlushGPUDirectRDMAWrites
    _check_or_init_driver()
    if __cuFlushGPUDirectRDMAWrites == NULL:
        with gil:
            raise FunctionNotFoundError("function cuFlushGPUDirectRDMAWrites is not found")
    return (<CUresult (*)(CUflushGPUDirectRDMAWritesTarget, CUflushGPUDirectRDMAWritesScope) noexcept nogil>__cuFlushGPUDirectRDMAWrites)(
        target, scope)


cdef CUresult _cuDeviceGetProperties(CUdevprop* prop, CUdevice dev) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuDeviceGetProperties
    _check_or_init_driver()
    if __cuDeviceGetProperties == NULL:
        with gil:
            raise FunctionNotFoundError("function cuDeviceGetProperties is not found")
    return (<CUresult (*)(CUdevprop*, CUdevice) noexcept nogil>__cuDeviceGetProperties)(
        prop, dev)


cdef CUresult _cuDeviceComputeCapability(int* major, int* minor, CUdevice dev) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuDeviceComputeCapability
    _check_or_init_driver()
    if __cuDeviceComputeCapability == NULL:
        with gil:
            raise FunctionNotFoundError("function cuDeviceComputeCapability is not found")
    return (<CUresult (*)(int*, int*, CUdevice) noexcept nogil>__cuDeviceComputeCapability)(
        major, minor, dev)


cdef CUresult _cuDevicePrimaryCtxRetain(CUcontext* pctx, CUdevice dev) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuDevicePrimaryCtxRetain
    _check_or_init_driver()
    if __cuDevicePrimaryCtxRetain == NULL:
        with gil:
            raise FunctionNotFoundError("function cuDevicePrimaryCtxRetain is not found")
    return (<CUresult (*)(CUcontext*, CUdevice) noexcept nogil>__cuDevicePrimaryCtxRetain)(
        pctx, dev)


cdef CUresult _cuDevicePrimaryCtxRelease_v2(CUdevice dev) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuDevicePrimaryCtxRelease_v2
    _check_or_init_driver()
    if __cuDevicePrimaryCtxRelease_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuDevicePrimaryCtxRelease_v2 is not found")
    return (<CUresult (*)(CUdevice) noexcept nogil>__cuDevicePrimaryCtxRelease_v2)(
        dev)


cdef CUresult _cuDevicePrimaryCtxSetFlags_v2(CUdevice dev, unsigned int flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuDevicePrimaryCtxSetFlags_v2
    _check_or_init_driver()
    if __cuDevicePrimaryCtxSetFlags_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuDevicePrimaryCtxSetFlags_v2 is not found")
    return (<CUresult (*)(CUdevice, unsigned int) noexcept nogil>__cuDevicePrimaryCtxSetFlags_v2)(
        dev, flags)


cdef CUresult _cuDevicePrimaryCtxGetState(CUdevice dev, unsigned int* flags, int* active) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuDevicePrimaryCtxGetState
    _check_or_init_driver()
    if __cuDevicePrimaryCtxGetState == NULL:
        with gil:
            raise FunctionNotFoundError("function cuDevicePrimaryCtxGetState is not found")
    return (<CUresult (*)(CUdevice, unsigned int*, int*) noexcept nogil>__cuDevicePrimaryCtxGetState)(
        dev, flags, active)


cdef CUresult _cuDevicePrimaryCtxReset_v2(CUdevice dev) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuDevicePrimaryCtxReset_v2
    _check_or_init_driver()
    if __cuDevicePrimaryCtxReset_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuDevicePrimaryCtxReset_v2 is not found")
    return (<CUresult (*)(CUdevice) noexcept nogil>__cuDevicePrimaryCtxReset_v2)(
        dev)


cdef CUresult _cuCtxCreate_v4(CUcontext* pctx, CUctxCreateParams* ctxCreateParams, unsigned int flags, CUdevice dev) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuCtxCreate_v4
    _check_or_init_driver()
    if __cuCtxCreate_v4 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuCtxCreate_v4 is not found")
    return (<CUresult (*)(CUcontext*, CUctxCreateParams*, unsigned int, CUdevice) noexcept nogil>__cuCtxCreate_v4)(
        pctx, ctxCreateParams, flags, dev)


cdef CUresult _cuCtxDestroy_v2(CUcontext ctx) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuCtxDestroy_v2
    _check_or_init_driver()
    if __cuCtxDestroy_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuCtxDestroy_v2 is not found")
    return (<CUresult (*)(CUcontext) noexcept nogil>__cuCtxDestroy_v2)(
        ctx)


cdef CUresult _cuCtxPushCurrent_v2(CUcontext ctx) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuCtxPushCurrent_v2
    _check_or_init_driver()
    if __cuCtxPushCurrent_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuCtxPushCurrent_v2 is not found")
    return (<CUresult (*)(CUcontext) noexcept nogil>__cuCtxPushCurrent_v2)(
        ctx)


cdef CUresult _cuCtxPopCurrent_v2(CUcontext* pctx) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuCtxPopCurrent_v2
    _check_or_init_driver()
    if __cuCtxPopCurrent_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuCtxPopCurrent_v2 is not found")
    return (<CUresult (*)(CUcontext*) noexcept nogil>__cuCtxPopCurrent_v2)(
        pctx)


cdef CUresult _cuCtxSetCurrent(CUcontext ctx) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuCtxSetCurrent
    _check_or_init_driver()
    if __cuCtxSetCurrent == NULL:
        with gil:
            raise FunctionNotFoundError("function cuCtxSetCurrent is not found")
    return (<CUresult (*)(CUcontext) noexcept nogil>__cuCtxSetCurrent)(
        ctx)


cdef CUresult _cuCtxGetCurrent(CUcontext* pctx) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuCtxGetCurrent
    _check_or_init_driver()
    if __cuCtxGetCurrent == NULL:
        with gil:
            raise FunctionNotFoundError("function cuCtxGetCurrent is not found")
    return (<CUresult (*)(CUcontext*) noexcept nogil>__cuCtxGetCurrent)(
        pctx)


cdef CUresult _cuCtxGetDevice(CUdevice* device) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuCtxGetDevice
    _check_or_init_driver()
    if __cuCtxGetDevice == NULL:
        with gil:
            raise FunctionNotFoundError("function cuCtxGetDevice is not found")
    return (<CUresult (*)(CUdevice*) noexcept nogil>__cuCtxGetDevice)(
        device)


cdef CUresult _cuCtxGetFlags(unsigned int* flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuCtxGetFlags
    _check_or_init_driver()
    if __cuCtxGetFlags == NULL:
        with gil:
            raise FunctionNotFoundError("function cuCtxGetFlags is not found")
    return (<CUresult (*)(unsigned int*) noexcept nogil>__cuCtxGetFlags)(
        flags)


cdef CUresult _cuCtxSetFlags(unsigned int flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuCtxSetFlags
    _check_or_init_driver()
    if __cuCtxSetFlags == NULL:
        with gil:
            raise FunctionNotFoundError("function cuCtxSetFlags is not found")
    return (<CUresult (*)(unsigned int) noexcept nogil>__cuCtxSetFlags)(
        flags)


cdef CUresult _cuCtxGetId(CUcontext ctx, unsigned long long* ctxId) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuCtxGetId
    _check_or_init_driver()
    if __cuCtxGetId == NULL:
        with gil:
            raise FunctionNotFoundError("function cuCtxGetId is not found")
    return (<CUresult (*)(CUcontext, unsigned long long*) noexcept nogil>__cuCtxGetId)(
        ctx, ctxId)


cdef CUresult _cuCtxSynchronize() except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuCtxSynchronize
    _check_or_init_driver()
    if __cuCtxSynchronize == NULL:
        with gil:
            raise FunctionNotFoundError("function cuCtxSynchronize is not found")
    return (<CUresult (*)() noexcept nogil>__cuCtxSynchronize)(
        )


cdef CUresult _cuCtxSetLimit(CUlimit limit, size_t value) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuCtxSetLimit
    _check_or_init_driver()
    if __cuCtxSetLimit == NULL:
        with gil:
            raise FunctionNotFoundError("function cuCtxSetLimit is not found")
    return (<CUresult (*)(CUlimit, size_t) noexcept nogil>__cuCtxSetLimit)(
        limit, value)


cdef CUresult _cuCtxGetLimit(size_t* pvalue, CUlimit limit) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuCtxGetLimit
    _check_or_init_driver()
    if __cuCtxGetLimit == NULL:
        with gil:
            raise FunctionNotFoundError("function cuCtxGetLimit is not found")
    return (<CUresult (*)(size_t*, CUlimit) noexcept nogil>__cuCtxGetLimit)(
        pvalue, limit)


cdef CUresult _cuCtxGetCacheConfig(CUfunc_cache* pconfig) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuCtxGetCacheConfig
    _check_or_init_driver()
    if __cuCtxGetCacheConfig == NULL:
        with gil:
            raise FunctionNotFoundError("function cuCtxGetCacheConfig is not found")
    return (<CUresult (*)(CUfunc_cache*) noexcept nogil>__cuCtxGetCacheConfig)(
        pconfig)


cdef CUresult _cuCtxSetCacheConfig(CUfunc_cache config) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuCtxSetCacheConfig
    _check_or_init_driver()
    if __cuCtxSetCacheConfig == NULL:
        with gil:
            raise FunctionNotFoundError("function cuCtxSetCacheConfig is not found")
    return (<CUresult (*)(CUfunc_cache) noexcept nogil>__cuCtxSetCacheConfig)(
        config)


cdef CUresult _cuCtxGetApiVersion(CUcontext ctx, unsigned int* version) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuCtxGetApiVersion
    _check_or_init_driver()
    if __cuCtxGetApiVersion == NULL:
        with gil:
            raise FunctionNotFoundError("function cuCtxGetApiVersion is not found")
    return (<CUresult (*)(CUcontext, unsigned int*) noexcept nogil>__cuCtxGetApiVersion)(
        ctx, version)


cdef CUresult _cuCtxGetStreamPriorityRange(int* leastPriority, int* greatestPriority) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuCtxGetStreamPriorityRange
    _check_or_init_driver()
    if __cuCtxGetStreamPriorityRange == NULL:
        with gil:
            raise FunctionNotFoundError("function cuCtxGetStreamPriorityRange is not found")
    return (<CUresult (*)(int*, int*) noexcept nogil>__cuCtxGetStreamPriorityRange)(
        leastPriority, greatestPriority)


cdef CUresult _cuCtxResetPersistingL2Cache() except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuCtxResetPersistingL2Cache
    _check_or_init_driver()
    if __cuCtxResetPersistingL2Cache == NULL:
        with gil:
            raise FunctionNotFoundError("function cuCtxResetPersistingL2Cache is not found")
    return (<CUresult (*)() noexcept nogil>__cuCtxResetPersistingL2Cache)(
        )


cdef CUresult _cuCtxGetExecAffinity(CUexecAffinityParam* pExecAffinity, CUexecAffinityType type) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuCtxGetExecAffinity
    _check_or_init_driver()
    if __cuCtxGetExecAffinity == NULL:
        with gil:
            raise FunctionNotFoundError("function cuCtxGetExecAffinity is not found")
    return (<CUresult (*)(CUexecAffinityParam*, CUexecAffinityType) noexcept nogil>__cuCtxGetExecAffinity)(
        pExecAffinity, type)


cdef CUresult _cuCtxRecordEvent(CUcontext hCtx, CUevent hEvent) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuCtxRecordEvent
    _check_or_init_driver()
    if __cuCtxRecordEvent == NULL:
        with gil:
            raise FunctionNotFoundError("function cuCtxRecordEvent is not found")
    return (<CUresult (*)(CUcontext, CUevent) noexcept nogil>__cuCtxRecordEvent)(
        hCtx, hEvent)


cdef CUresult _cuCtxWaitEvent(CUcontext hCtx, CUevent hEvent) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuCtxWaitEvent
    _check_or_init_driver()
    if __cuCtxWaitEvent == NULL:
        with gil:
            raise FunctionNotFoundError("function cuCtxWaitEvent is not found")
    return (<CUresult (*)(CUcontext, CUevent) noexcept nogil>__cuCtxWaitEvent)(
        hCtx, hEvent)


cdef CUresult _cuCtxAttach(CUcontext* pctx, unsigned int flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuCtxAttach
    _check_or_init_driver()
    if __cuCtxAttach == NULL:
        with gil:
            raise FunctionNotFoundError("function cuCtxAttach is not found")
    return (<CUresult (*)(CUcontext*, unsigned int) noexcept nogil>__cuCtxAttach)(
        pctx, flags)


cdef CUresult _cuCtxDetach(CUcontext ctx) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuCtxDetach
    _check_or_init_driver()
    if __cuCtxDetach == NULL:
        with gil:
            raise FunctionNotFoundError("function cuCtxDetach is not found")
    return (<CUresult (*)(CUcontext) noexcept nogil>__cuCtxDetach)(
        ctx)


cdef CUresult _cuCtxGetSharedMemConfig(CUsharedconfig* pConfig) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuCtxGetSharedMemConfig
    _check_or_init_driver()
    if __cuCtxGetSharedMemConfig == NULL:
        with gil:
            raise FunctionNotFoundError("function cuCtxGetSharedMemConfig is not found")
    return (<CUresult (*)(CUsharedconfig*) noexcept nogil>__cuCtxGetSharedMemConfig)(
        pConfig)


cdef CUresult _cuCtxSetSharedMemConfig(CUsharedconfig config) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuCtxSetSharedMemConfig
    _check_or_init_driver()
    if __cuCtxSetSharedMemConfig == NULL:
        with gil:
            raise FunctionNotFoundError("function cuCtxSetSharedMemConfig is not found")
    return (<CUresult (*)(CUsharedconfig) noexcept nogil>__cuCtxSetSharedMemConfig)(
        config)


cdef CUresult _cuModuleLoad(CUmodule* module, const char* fname) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuModuleLoad
    _check_or_init_driver()
    if __cuModuleLoad == NULL:
        with gil:
            raise FunctionNotFoundError("function cuModuleLoad is not found")
    return (<CUresult (*)(CUmodule*, const char*) noexcept nogil>__cuModuleLoad)(
        module, fname)


cdef CUresult _cuModuleLoadData(CUmodule* module, const void* image) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuModuleLoadData
    _check_or_init_driver()
    if __cuModuleLoadData == NULL:
        with gil:
            raise FunctionNotFoundError("function cuModuleLoadData is not found")
    return (<CUresult (*)(CUmodule*, const void*) noexcept nogil>__cuModuleLoadData)(
        module, image)


cdef CUresult _cuModuleLoadDataEx(CUmodule* module, const void* image, unsigned int numOptions, CUjit_option* options, void** optionValues) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuModuleLoadDataEx
    _check_or_init_driver()
    if __cuModuleLoadDataEx == NULL:
        with gil:
            raise FunctionNotFoundError("function cuModuleLoadDataEx is not found")
    return (<CUresult (*)(CUmodule*, const void*, unsigned int, CUjit_option*, void**) noexcept nogil>__cuModuleLoadDataEx)(
        module, image, numOptions, options, optionValues)


cdef CUresult _cuModuleLoadFatBinary(CUmodule* module, const void* fatCubin) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuModuleLoadFatBinary
    _check_or_init_driver()
    if __cuModuleLoadFatBinary == NULL:
        with gil:
            raise FunctionNotFoundError("function cuModuleLoadFatBinary is not found")
    return (<CUresult (*)(CUmodule*, const void*) noexcept nogil>__cuModuleLoadFatBinary)(
        module, fatCubin)


cdef CUresult _cuModuleUnload(CUmodule hmod) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuModuleUnload
    _check_or_init_driver()
    if __cuModuleUnload == NULL:
        with gil:
            raise FunctionNotFoundError("function cuModuleUnload is not found")
    return (<CUresult (*)(CUmodule) noexcept nogil>__cuModuleUnload)(
        hmod)


cdef CUresult _cuModuleGetLoadingMode(CUmoduleLoadingMode* mode) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuModuleGetLoadingMode
    _check_or_init_driver()
    if __cuModuleGetLoadingMode == NULL:
        with gil:
            raise FunctionNotFoundError("function cuModuleGetLoadingMode is not found")
    return (<CUresult (*)(CUmoduleLoadingMode*) noexcept nogil>__cuModuleGetLoadingMode)(
        mode)


cdef CUresult _cuModuleGetFunction(CUfunction* hfunc, CUmodule hmod, const char* name) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuModuleGetFunction
    _check_or_init_driver()
    if __cuModuleGetFunction == NULL:
        with gil:
            raise FunctionNotFoundError("function cuModuleGetFunction is not found")
    return (<CUresult (*)(CUfunction*, CUmodule, const char*) noexcept nogil>__cuModuleGetFunction)(
        hfunc, hmod, name)


cdef CUresult _cuModuleGetFunctionCount(unsigned int* count, CUmodule mod) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuModuleGetFunctionCount
    _check_or_init_driver()
    if __cuModuleGetFunctionCount == NULL:
        with gil:
            raise FunctionNotFoundError("function cuModuleGetFunctionCount is not found")
    return (<CUresult (*)(unsigned int*, CUmodule) noexcept nogil>__cuModuleGetFunctionCount)(
        count, mod)


cdef CUresult _cuModuleEnumerateFunctions(CUfunction* functions, unsigned int numFunctions, CUmodule mod) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuModuleEnumerateFunctions
    _check_or_init_driver()
    if __cuModuleEnumerateFunctions == NULL:
        with gil:
            raise FunctionNotFoundError("function cuModuleEnumerateFunctions is not found")
    return (<CUresult (*)(CUfunction*, unsigned int, CUmodule) noexcept nogil>__cuModuleEnumerateFunctions)(
        functions, numFunctions, mod)


cdef CUresult _cuModuleGetGlobal_v2(CUdeviceptr* dptr, size_t* bytes, CUmodule hmod, const char* name) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuModuleGetGlobal_v2
    _check_or_init_driver()
    if __cuModuleGetGlobal_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuModuleGetGlobal_v2 is not found")
    return (<CUresult (*)(CUdeviceptr*, size_t*, CUmodule, const char*) noexcept nogil>__cuModuleGetGlobal_v2)(
        dptr, bytes, hmod, name)


cdef CUresult _cuLinkCreate_v2(unsigned int numOptions, CUjit_option* options, void** optionValues, CUlinkState* stateOut) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuLinkCreate_v2
    _check_or_init_driver()
    if __cuLinkCreate_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuLinkCreate_v2 is not found")
    return (<CUresult (*)(unsigned int, CUjit_option*, void**, CUlinkState*) noexcept nogil>__cuLinkCreate_v2)(
        numOptions, options, optionValues, stateOut)


cdef CUresult _cuLinkAddData_v2(CUlinkState state, CUjitInputType type, void* data, size_t size, const char* name, unsigned int numOptions, CUjit_option* options, void** optionValues) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuLinkAddData_v2
    _check_or_init_driver()
    if __cuLinkAddData_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuLinkAddData_v2 is not found")
    return (<CUresult (*)(CUlinkState, CUjitInputType, void*, size_t, const char*, unsigned int, CUjit_option*, void**) noexcept nogil>__cuLinkAddData_v2)(
        state, type, data, size, name, numOptions, options, optionValues)


cdef CUresult _cuLinkAddFile_v2(CUlinkState state, CUjitInputType type, const char* path, unsigned int numOptions, CUjit_option* options, void** optionValues) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuLinkAddFile_v2
    _check_or_init_driver()
    if __cuLinkAddFile_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuLinkAddFile_v2 is not found")
    return (<CUresult (*)(CUlinkState, CUjitInputType, const char*, unsigned int, CUjit_option*, void**) noexcept nogil>__cuLinkAddFile_v2)(
        state, type, path, numOptions, options, optionValues)


cdef CUresult _cuLinkComplete(CUlinkState state, void** cubinOut, size_t* sizeOut) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuLinkComplete
    _check_or_init_driver()
    if __cuLinkComplete == NULL:
        with gil:
            raise FunctionNotFoundError("function cuLinkComplete is not found")
    return (<CUresult (*)(CUlinkState, void**, size_t*) noexcept nogil>__cuLinkComplete)(
        state, cubinOut, sizeOut)


cdef CUresult _cuLinkDestroy(CUlinkState state) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuLinkDestroy
    _check_or_init_driver()
    if __cuLinkDestroy == NULL:
        with gil:
            raise FunctionNotFoundError("function cuLinkDestroy is not found")
    return (<CUresult (*)(CUlinkState) noexcept nogil>__cuLinkDestroy)(
        state)


cdef CUresult _cuModuleGetTexRef(CUtexref* pTexRef, CUmodule hmod, const char* name) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuModuleGetTexRef
    _check_or_init_driver()
    if __cuModuleGetTexRef == NULL:
        with gil:
            raise FunctionNotFoundError("function cuModuleGetTexRef is not found")
    return (<CUresult (*)(CUtexref*, CUmodule, const char*) noexcept nogil>__cuModuleGetTexRef)(
        pTexRef, hmod, name)


cdef CUresult _cuModuleGetSurfRef(CUsurfref* pSurfRef, CUmodule hmod, const char* name) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuModuleGetSurfRef
    _check_or_init_driver()
    if __cuModuleGetSurfRef == NULL:
        with gil:
            raise FunctionNotFoundError("function cuModuleGetSurfRef is not found")
    return (<CUresult (*)(CUsurfref*, CUmodule, const char*) noexcept nogil>__cuModuleGetSurfRef)(
        pSurfRef, hmod, name)


cdef CUresult _cuLibraryLoadData(CUlibrary* library, const void* code, CUjit_option* jitOptions, void** jitOptionsValues, unsigned int numJitOptions, CUlibraryOption* libraryOptions, void** libraryOptionValues, unsigned int numLibraryOptions) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuLibraryLoadData
    _check_or_init_driver()
    if __cuLibraryLoadData == NULL:
        with gil:
            raise FunctionNotFoundError("function cuLibraryLoadData is not found")
    return (<CUresult (*)(CUlibrary*, const void*, CUjit_option*, void**, unsigned int, CUlibraryOption*, void**, unsigned int) noexcept nogil>__cuLibraryLoadData)(
        library, code, jitOptions, jitOptionsValues, numJitOptions, libraryOptions, libraryOptionValues, numLibraryOptions)


cdef CUresult _cuLibraryLoadFromFile(CUlibrary* library, const char* fileName, CUjit_option* jitOptions, void** jitOptionsValues, unsigned int numJitOptions, CUlibraryOption* libraryOptions, void** libraryOptionValues, unsigned int numLibraryOptions) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuLibraryLoadFromFile
    _check_or_init_driver()
    if __cuLibraryLoadFromFile == NULL:
        with gil:
            raise FunctionNotFoundError("function cuLibraryLoadFromFile is not found")
    return (<CUresult (*)(CUlibrary*, const char*, CUjit_option*, void**, unsigned int, CUlibraryOption*, void**, unsigned int) noexcept nogil>__cuLibraryLoadFromFile)(
        library, fileName, jitOptions, jitOptionsValues, numJitOptions, libraryOptions, libraryOptionValues, numLibraryOptions)


cdef CUresult _cuLibraryUnload(CUlibrary library) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuLibraryUnload
    _check_or_init_driver()
    if __cuLibraryUnload == NULL:
        with gil:
            raise FunctionNotFoundError("function cuLibraryUnload is not found")
    return (<CUresult (*)(CUlibrary) noexcept nogil>__cuLibraryUnload)(
        library)


cdef CUresult _cuLibraryGetKernel(CUkernel* pKernel, CUlibrary library, const char* name) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuLibraryGetKernel
    _check_or_init_driver()
    if __cuLibraryGetKernel == NULL:
        with gil:
            raise FunctionNotFoundError("function cuLibraryGetKernel is not found")
    return (<CUresult (*)(CUkernel*, CUlibrary, const char*) noexcept nogil>__cuLibraryGetKernel)(
        pKernel, library, name)


cdef CUresult _cuLibraryGetKernelCount(unsigned int* count, CUlibrary lib) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuLibraryGetKernelCount
    _check_or_init_driver()
    if __cuLibraryGetKernelCount == NULL:
        with gil:
            raise FunctionNotFoundError("function cuLibraryGetKernelCount is not found")
    return (<CUresult (*)(unsigned int*, CUlibrary) noexcept nogil>__cuLibraryGetKernelCount)(
        count, lib)


cdef CUresult _cuLibraryEnumerateKernels(CUkernel* kernels, unsigned int numKernels, CUlibrary lib) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuLibraryEnumerateKernels
    _check_or_init_driver()
    if __cuLibraryEnumerateKernels == NULL:
        with gil:
            raise FunctionNotFoundError("function cuLibraryEnumerateKernels is not found")
    return (<CUresult (*)(CUkernel*, unsigned int, CUlibrary) noexcept nogil>__cuLibraryEnumerateKernels)(
        kernels, numKernels, lib)


cdef CUresult _cuLibraryGetModule(CUmodule* pMod, CUlibrary library) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuLibraryGetModule
    _check_or_init_driver()
    if __cuLibraryGetModule == NULL:
        with gil:
            raise FunctionNotFoundError("function cuLibraryGetModule is not found")
    return (<CUresult (*)(CUmodule*, CUlibrary) noexcept nogil>__cuLibraryGetModule)(
        pMod, library)


cdef CUresult _cuKernelGetFunction(CUfunction* pFunc, CUkernel kernel) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuKernelGetFunction
    _check_or_init_driver()
    if __cuKernelGetFunction == NULL:
        with gil:
            raise FunctionNotFoundError("function cuKernelGetFunction is not found")
    return (<CUresult (*)(CUfunction*, CUkernel) noexcept nogil>__cuKernelGetFunction)(
        pFunc, kernel)


cdef CUresult _cuKernelGetLibrary(CUlibrary* pLib, CUkernel kernel) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuKernelGetLibrary
    _check_or_init_driver()
    if __cuKernelGetLibrary == NULL:
        with gil:
            raise FunctionNotFoundError("function cuKernelGetLibrary is not found")
    return (<CUresult (*)(CUlibrary*, CUkernel) noexcept nogil>__cuKernelGetLibrary)(
        pLib, kernel)


cdef CUresult _cuLibraryGetGlobal(CUdeviceptr* dptr, size_t* bytes, CUlibrary library, const char* name) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuLibraryGetGlobal
    _check_or_init_driver()
    if __cuLibraryGetGlobal == NULL:
        with gil:
            raise FunctionNotFoundError("function cuLibraryGetGlobal is not found")
    return (<CUresult (*)(CUdeviceptr*, size_t*, CUlibrary, const char*) noexcept nogil>__cuLibraryGetGlobal)(
        dptr, bytes, library, name)


cdef CUresult _cuLibraryGetManaged(CUdeviceptr* dptr, size_t* bytes, CUlibrary library, const char* name) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuLibraryGetManaged
    _check_or_init_driver()
    if __cuLibraryGetManaged == NULL:
        with gil:
            raise FunctionNotFoundError("function cuLibraryGetManaged is not found")
    return (<CUresult (*)(CUdeviceptr*, size_t*, CUlibrary, const char*) noexcept nogil>__cuLibraryGetManaged)(
        dptr, bytes, library, name)


cdef CUresult _cuLibraryGetUnifiedFunction(void** fptr, CUlibrary library, const char* symbol) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuLibraryGetUnifiedFunction
    _check_or_init_driver()
    if __cuLibraryGetUnifiedFunction == NULL:
        with gil:
            raise FunctionNotFoundError("function cuLibraryGetUnifiedFunction is not found")
    return (<CUresult (*)(void**, CUlibrary, const char*) noexcept nogil>__cuLibraryGetUnifiedFunction)(
        fptr, library, symbol)


cdef CUresult _cuKernelGetAttribute(int* pi, CUfunction_attribute attrib, CUkernel kernel, CUdevice dev) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuKernelGetAttribute
    _check_or_init_driver()
    if __cuKernelGetAttribute == NULL:
        with gil:
            raise FunctionNotFoundError("function cuKernelGetAttribute is not found")
    return (<CUresult (*)(int*, CUfunction_attribute, CUkernel, CUdevice) noexcept nogil>__cuKernelGetAttribute)(
        pi, attrib, kernel, dev)


cdef CUresult _cuKernelSetAttribute(CUfunction_attribute attrib, int val, CUkernel kernel, CUdevice dev) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuKernelSetAttribute
    _check_or_init_driver()
    if __cuKernelSetAttribute == NULL:
        with gil:
            raise FunctionNotFoundError("function cuKernelSetAttribute is not found")
    return (<CUresult (*)(CUfunction_attribute, int, CUkernel, CUdevice) noexcept nogil>__cuKernelSetAttribute)(
        attrib, val, kernel, dev)


cdef CUresult _cuKernelSetCacheConfig(CUkernel kernel, CUfunc_cache config, CUdevice dev) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuKernelSetCacheConfig
    _check_or_init_driver()
    if __cuKernelSetCacheConfig == NULL:
        with gil:
            raise FunctionNotFoundError("function cuKernelSetCacheConfig is not found")
    return (<CUresult (*)(CUkernel, CUfunc_cache, CUdevice) noexcept nogil>__cuKernelSetCacheConfig)(
        kernel, config, dev)


cdef CUresult _cuKernelGetName(const char** name, CUkernel hfunc) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuKernelGetName
    _check_or_init_driver()
    if __cuKernelGetName == NULL:
        with gil:
            raise FunctionNotFoundError("function cuKernelGetName is not found")
    return (<CUresult (*)(const char**, CUkernel) noexcept nogil>__cuKernelGetName)(
        name, hfunc)


cdef CUresult _cuKernelGetParamInfo(CUkernel kernel, size_t paramIndex, size_t* paramOffset, size_t* paramSize) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuKernelGetParamInfo
    _check_or_init_driver()
    if __cuKernelGetParamInfo == NULL:
        with gil:
            raise FunctionNotFoundError("function cuKernelGetParamInfo is not found")
    return (<CUresult (*)(CUkernel, size_t, size_t*, size_t*) noexcept nogil>__cuKernelGetParamInfo)(
        kernel, paramIndex, paramOffset, paramSize)


cdef CUresult _cuMemGetInfo_v2(size_t* free, size_t* total) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuMemGetInfo_v2
    _check_or_init_driver()
    if __cuMemGetInfo_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemGetInfo_v2 is not found")
    return (<CUresult (*)(size_t*, size_t*) noexcept nogil>__cuMemGetInfo_v2)(
        free, total)


cdef CUresult _cuMemAlloc_v2(CUdeviceptr* dptr, size_t bytesize) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuMemAlloc_v2
    _check_or_init_driver()
    if __cuMemAlloc_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemAlloc_v2 is not found")
    return (<CUresult (*)(CUdeviceptr*, size_t) noexcept nogil>__cuMemAlloc_v2)(
        dptr, bytesize)


cdef CUresult _cuMemAllocPitch_v2(CUdeviceptr* dptr, size_t* pPitch, size_t WidthInBytes, size_t Height, unsigned int ElementSizeBytes) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuMemAllocPitch_v2
    _check_or_init_driver()
    if __cuMemAllocPitch_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemAllocPitch_v2 is not found")
    return (<CUresult (*)(CUdeviceptr*, size_t*, size_t, size_t, unsigned int) noexcept nogil>__cuMemAllocPitch_v2)(
        dptr, pPitch, WidthInBytes, Height, ElementSizeBytes)


cdef CUresult _cuMemFree_v2(CUdeviceptr dptr) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuMemFree_v2
    _check_or_init_driver()
    if __cuMemFree_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemFree_v2 is not found")
    return (<CUresult (*)(CUdeviceptr) noexcept nogil>__cuMemFree_v2)(
        dptr)


cdef CUresult _cuMemGetAddressRange_v2(CUdeviceptr* pbase, size_t* psize, CUdeviceptr dptr) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuMemGetAddressRange_v2
    _check_or_init_driver()
    if __cuMemGetAddressRange_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemGetAddressRange_v2 is not found")
    return (<CUresult (*)(CUdeviceptr*, size_t*, CUdeviceptr) noexcept nogil>__cuMemGetAddressRange_v2)(
        pbase, psize, dptr)


cdef CUresult _cuMemAllocHost_v2(void** pp, size_t bytesize) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuMemAllocHost_v2
    _check_or_init_driver()
    if __cuMemAllocHost_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemAllocHost_v2 is not found")
    return (<CUresult (*)(void**, size_t) noexcept nogil>__cuMemAllocHost_v2)(
        pp, bytesize)


cdef CUresult _cuMemFreeHost(void* p) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuMemFreeHost
    _check_or_init_driver()
    if __cuMemFreeHost == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemFreeHost is not found")
    return (<CUresult (*)(void*) noexcept nogil>__cuMemFreeHost)(
        p)


cdef CUresult _cuMemHostAlloc(void** pp, size_t bytesize, unsigned int Flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuMemHostAlloc
    _check_or_init_driver()
    if __cuMemHostAlloc == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemHostAlloc is not found")
    return (<CUresult (*)(void**, size_t, unsigned int) noexcept nogil>__cuMemHostAlloc)(
        pp, bytesize, Flags)


cdef CUresult _cuMemHostGetDevicePointer_v2(CUdeviceptr* pdptr, void* p, unsigned int Flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuMemHostGetDevicePointer_v2
    _check_or_init_driver()
    if __cuMemHostGetDevicePointer_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemHostGetDevicePointer_v2 is not found")
    return (<CUresult (*)(CUdeviceptr*, void*, unsigned int) noexcept nogil>__cuMemHostGetDevicePointer_v2)(
        pdptr, p, Flags)


cdef CUresult _cuMemHostGetFlags(unsigned int* pFlags, void* p) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuMemHostGetFlags
    _check_or_init_driver()
    if __cuMemHostGetFlags == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemHostGetFlags is not found")
    return (<CUresult (*)(unsigned int*, void*) noexcept nogil>__cuMemHostGetFlags)(
        pFlags, p)


cdef CUresult _cuMemAllocManaged(CUdeviceptr* dptr, size_t bytesize, unsigned int flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuMemAllocManaged
    _check_or_init_driver()
    if __cuMemAllocManaged == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemAllocManaged is not found")
    return (<CUresult (*)(CUdeviceptr*, size_t, unsigned int) noexcept nogil>__cuMemAllocManaged)(
        dptr, bytesize, flags)


cdef CUresult _cuDeviceRegisterAsyncNotification(CUdevice device, CUasyncCallback callbackFunc, void* userData, CUasyncCallbackHandle* callback) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuDeviceRegisterAsyncNotification
    _check_or_init_driver()
    if __cuDeviceRegisterAsyncNotification == NULL:
        with gil:
            raise FunctionNotFoundError("function cuDeviceRegisterAsyncNotification is not found")
    return (<CUresult (*)(CUdevice, CUasyncCallback, void*, CUasyncCallbackHandle*) noexcept nogil>__cuDeviceRegisterAsyncNotification)(
        device, callbackFunc, userData, callback)


cdef CUresult _cuDeviceUnregisterAsyncNotification(CUdevice device, CUasyncCallbackHandle callback) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuDeviceUnregisterAsyncNotification
    _check_or_init_driver()
    if __cuDeviceUnregisterAsyncNotification == NULL:
        with gil:
            raise FunctionNotFoundError("function cuDeviceUnregisterAsyncNotification is not found")
    return (<CUresult (*)(CUdevice, CUasyncCallbackHandle) noexcept nogil>__cuDeviceUnregisterAsyncNotification)(
        device, callback)


cdef CUresult _cuDeviceGetByPCIBusId(CUdevice* dev, const char* pciBusId) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuDeviceGetByPCIBusId
    _check_or_init_driver()
    if __cuDeviceGetByPCIBusId == NULL:
        with gil:
            raise FunctionNotFoundError("function cuDeviceGetByPCIBusId is not found")
    return (<CUresult (*)(CUdevice*, const char*) noexcept nogil>__cuDeviceGetByPCIBusId)(
        dev, pciBusId)


cdef CUresult _cuDeviceGetPCIBusId(char* pciBusId, int len, CUdevice dev) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuDeviceGetPCIBusId
    _check_or_init_driver()
    if __cuDeviceGetPCIBusId == NULL:
        with gil:
            raise FunctionNotFoundError("function cuDeviceGetPCIBusId is not found")
    return (<CUresult (*)(char*, int, CUdevice) noexcept nogil>__cuDeviceGetPCIBusId)(
        pciBusId, len, dev)


cdef CUresult _cuIpcGetEventHandle(CUipcEventHandle* pHandle, CUevent event) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuIpcGetEventHandle
    _check_or_init_driver()
    if __cuIpcGetEventHandle == NULL:
        with gil:
            raise FunctionNotFoundError("function cuIpcGetEventHandle is not found")
    return (<CUresult (*)(CUipcEventHandle*, CUevent) noexcept nogil>__cuIpcGetEventHandle)(
        pHandle, event)


cdef CUresult _cuIpcOpenEventHandle(CUevent* phEvent, CUipcEventHandle handle) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuIpcOpenEventHandle
    _check_or_init_driver()
    if __cuIpcOpenEventHandle == NULL:
        with gil:
            raise FunctionNotFoundError("function cuIpcOpenEventHandle is not found")
    return (<CUresult (*)(CUevent*, CUipcEventHandle) noexcept nogil>__cuIpcOpenEventHandle)(
        phEvent, handle)


cdef CUresult _cuIpcGetMemHandle(CUipcMemHandle* pHandle, CUdeviceptr dptr) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuIpcGetMemHandle
    _check_or_init_driver()
    if __cuIpcGetMemHandle == NULL:
        with gil:
            raise FunctionNotFoundError("function cuIpcGetMemHandle is not found")
    return (<CUresult (*)(CUipcMemHandle*, CUdeviceptr) noexcept nogil>__cuIpcGetMemHandle)(
        pHandle, dptr)


cdef CUresult _cuIpcOpenMemHandle_v2(CUdeviceptr* pdptr, CUipcMemHandle handle, unsigned int Flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuIpcOpenMemHandle_v2
    _check_or_init_driver()
    if __cuIpcOpenMemHandle_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuIpcOpenMemHandle_v2 is not found")
    return (<CUresult (*)(CUdeviceptr*, CUipcMemHandle, unsigned int) noexcept nogil>__cuIpcOpenMemHandle_v2)(
        pdptr, handle, Flags)


cdef CUresult _cuIpcCloseMemHandle(CUdeviceptr dptr) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuIpcCloseMemHandle
    _check_or_init_driver()
    if __cuIpcCloseMemHandle == NULL:
        with gil:
            raise FunctionNotFoundError("function cuIpcCloseMemHandle is not found")
    return (<CUresult (*)(CUdeviceptr) noexcept nogil>__cuIpcCloseMemHandle)(
        dptr)


cdef CUresult _cuMemHostRegister_v2(void* p, size_t bytesize, unsigned int Flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuMemHostRegister_v2
    _check_or_init_driver()
    if __cuMemHostRegister_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemHostRegister_v2 is not found")
    return (<CUresult (*)(void*, size_t, unsigned int) noexcept nogil>__cuMemHostRegister_v2)(
        p, bytesize, Flags)


cdef CUresult _cuMemHostUnregister(void* p) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuMemHostUnregister
    _check_or_init_driver()
    if __cuMemHostUnregister == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemHostUnregister is not found")
    return (<CUresult (*)(void*) noexcept nogil>__cuMemHostUnregister)(
        p)


cdef CUresult _cuMemcpy(CUdeviceptr dst, CUdeviceptr src, size_t ByteCount) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuMemcpy
    _check_or_init_driver()
    if __cuMemcpy == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemcpy is not found")
    return (<CUresult (*)(CUdeviceptr, CUdeviceptr, size_t) noexcept nogil>__cuMemcpy)(
        dst, src, ByteCount)


cdef CUresult _cuMemcpyPeer(CUdeviceptr dstDevice, CUcontext dstContext, CUdeviceptr srcDevice, CUcontext srcContext, size_t ByteCount) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuMemcpyPeer
    _check_or_init_driver()
    if __cuMemcpyPeer == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemcpyPeer is not found")
    return (<CUresult (*)(CUdeviceptr, CUcontext, CUdeviceptr, CUcontext, size_t) noexcept nogil>__cuMemcpyPeer)(
        dstDevice, dstContext, srcDevice, srcContext, ByteCount)


cdef CUresult _cuMemcpyHtoD_v2(CUdeviceptr dstDevice, const void* srcHost, size_t ByteCount) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuMemcpyHtoD_v2
    _check_or_init_driver()
    if __cuMemcpyHtoD_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemcpyHtoD_v2 is not found")
    return (<CUresult (*)(CUdeviceptr, const void*, size_t) noexcept nogil>__cuMemcpyHtoD_v2)(
        dstDevice, srcHost, ByteCount)


cdef CUresult _cuMemcpyDtoH_v2(void* dstHost, CUdeviceptr srcDevice, size_t ByteCount) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuMemcpyDtoH_v2
    _check_or_init_driver()
    if __cuMemcpyDtoH_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemcpyDtoH_v2 is not found")
    return (<CUresult (*)(void*, CUdeviceptr, size_t) noexcept nogil>__cuMemcpyDtoH_v2)(
        dstHost, srcDevice, ByteCount)


cdef CUresult _cuMemcpyDtoD_v2(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuMemcpyDtoD_v2
    _check_or_init_driver()
    if __cuMemcpyDtoD_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemcpyDtoD_v2 is not found")
    return (<CUresult (*)(CUdeviceptr, CUdeviceptr, size_t) noexcept nogil>__cuMemcpyDtoD_v2)(
        dstDevice, srcDevice, ByteCount)


cdef CUresult _cuMemcpyDtoA_v2(CUarray dstArray, size_t dstOffset, CUdeviceptr srcDevice, size_t ByteCount) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuMemcpyDtoA_v2
    _check_or_init_driver()
    if __cuMemcpyDtoA_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemcpyDtoA_v2 is not found")
    return (<CUresult (*)(CUarray, size_t, CUdeviceptr, size_t) noexcept nogil>__cuMemcpyDtoA_v2)(
        dstArray, dstOffset, srcDevice, ByteCount)


cdef CUresult _cuMemcpyAtoD_v2(CUdeviceptr dstDevice, CUarray srcArray, size_t srcOffset, size_t ByteCount) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuMemcpyAtoD_v2
    _check_or_init_driver()
    if __cuMemcpyAtoD_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemcpyAtoD_v2 is not found")
    return (<CUresult (*)(CUdeviceptr, CUarray, size_t, size_t) noexcept nogil>__cuMemcpyAtoD_v2)(
        dstDevice, srcArray, srcOffset, ByteCount)


cdef CUresult _cuMemcpyHtoA_v2(CUarray dstArray, size_t dstOffset, const void* srcHost, size_t ByteCount) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuMemcpyHtoA_v2
    _check_or_init_driver()
    if __cuMemcpyHtoA_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemcpyHtoA_v2 is not found")
    return (<CUresult (*)(CUarray, size_t, const void*, size_t) noexcept nogil>__cuMemcpyHtoA_v2)(
        dstArray, dstOffset, srcHost, ByteCount)


cdef CUresult _cuMemcpyAtoH_v2(void* dstHost, CUarray srcArray, size_t srcOffset, size_t ByteCount) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuMemcpyAtoH_v2
    _check_or_init_driver()
    if __cuMemcpyAtoH_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemcpyAtoH_v2 is not found")
    return (<CUresult (*)(void*, CUarray, size_t, size_t) noexcept nogil>__cuMemcpyAtoH_v2)(
        dstHost, srcArray, srcOffset, ByteCount)


cdef CUresult _cuMemcpyAtoA_v2(CUarray dstArray, size_t dstOffset, CUarray srcArray, size_t srcOffset, size_t ByteCount) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuMemcpyAtoA_v2
    _check_or_init_driver()
    if __cuMemcpyAtoA_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemcpyAtoA_v2 is not found")
    return (<CUresult (*)(CUarray, size_t, CUarray, size_t, size_t) noexcept nogil>__cuMemcpyAtoA_v2)(
        dstArray, dstOffset, srcArray, srcOffset, ByteCount)


cdef CUresult _cuMemcpy2D_v2(const CUDA_MEMCPY2D* pCopy) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuMemcpy2D_v2
    _check_or_init_driver()
    if __cuMemcpy2D_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemcpy2D_v2 is not found")
    return (<CUresult (*)(const CUDA_MEMCPY2D*) noexcept nogil>__cuMemcpy2D_v2)(
        pCopy)


cdef CUresult _cuMemcpy2DUnaligned_v2(const CUDA_MEMCPY2D* pCopy) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuMemcpy2DUnaligned_v2
    _check_or_init_driver()
    if __cuMemcpy2DUnaligned_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemcpy2DUnaligned_v2 is not found")
    return (<CUresult (*)(const CUDA_MEMCPY2D*) noexcept nogil>__cuMemcpy2DUnaligned_v2)(
        pCopy)


cdef CUresult _cuMemcpy3D_v2(const CUDA_MEMCPY3D* pCopy) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuMemcpy3D_v2
    _check_or_init_driver()
    if __cuMemcpy3D_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemcpy3D_v2 is not found")
    return (<CUresult (*)(const CUDA_MEMCPY3D*) noexcept nogil>__cuMemcpy3D_v2)(
        pCopy)


cdef CUresult _cuMemcpy3DPeer(const CUDA_MEMCPY3D_PEER* pCopy) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuMemcpy3DPeer
    _check_or_init_driver()
    if __cuMemcpy3DPeer == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemcpy3DPeer is not found")
    return (<CUresult (*)(const CUDA_MEMCPY3D_PEER*) noexcept nogil>__cuMemcpy3DPeer)(
        pCopy)


cdef CUresult _cuMemcpyAsync(CUdeviceptr dst, CUdeviceptr src, size_t ByteCount, CUstream hStream) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuMemcpyAsync
    _check_or_init_driver()
    if __cuMemcpyAsync == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemcpyAsync is not found")
    return (<CUresult (*)(CUdeviceptr, CUdeviceptr, size_t, CUstream) noexcept nogil>__cuMemcpyAsync)(
        dst, src, ByteCount, hStream)


cdef CUresult _cuMemcpyPeerAsync(CUdeviceptr dstDevice, CUcontext dstContext, CUdeviceptr srcDevice, CUcontext srcContext, size_t ByteCount, CUstream hStream) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuMemcpyPeerAsync
    _check_or_init_driver()
    if __cuMemcpyPeerAsync == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemcpyPeerAsync is not found")
    return (<CUresult (*)(CUdeviceptr, CUcontext, CUdeviceptr, CUcontext, size_t, CUstream) noexcept nogil>__cuMemcpyPeerAsync)(
        dstDevice, dstContext, srcDevice, srcContext, ByteCount, hStream)


cdef CUresult _cuMemcpyHtoDAsync_v2(CUdeviceptr dstDevice, const void* srcHost, size_t ByteCount, CUstream hStream) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuMemcpyHtoDAsync_v2
    _check_or_init_driver()
    if __cuMemcpyHtoDAsync_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemcpyHtoDAsync_v2 is not found")
    return (<CUresult (*)(CUdeviceptr, const void*, size_t, CUstream) noexcept nogil>__cuMemcpyHtoDAsync_v2)(
        dstDevice, srcHost, ByteCount, hStream)


cdef CUresult _cuMemcpyDtoHAsync_v2(void* dstHost, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuMemcpyDtoHAsync_v2
    _check_or_init_driver()
    if __cuMemcpyDtoHAsync_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemcpyDtoHAsync_v2 is not found")
    return (<CUresult (*)(void*, CUdeviceptr, size_t, CUstream) noexcept nogil>__cuMemcpyDtoHAsync_v2)(
        dstHost, srcDevice, ByteCount, hStream)


cdef CUresult _cuMemcpyDtoDAsync_v2(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuMemcpyDtoDAsync_v2
    _check_or_init_driver()
    if __cuMemcpyDtoDAsync_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemcpyDtoDAsync_v2 is not found")
    return (<CUresult (*)(CUdeviceptr, CUdeviceptr, size_t, CUstream) noexcept nogil>__cuMemcpyDtoDAsync_v2)(
        dstDevice, srcDevice, ByteCount, hStream)


cdef CUresult _cuMemcpyHtoAAsync_v2(CUarray dstArray, size_t dstOffset, const void* srcHost, size_t ByteCount, CUstream hStream) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuMemcpyHtoAAsync_v2
    _check_or_init_driver()
    if __cuMemcpyHtoAAsync_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemcpyHtoAAsync_v2 is not found")
    return (<CUresult (*)(CUarray, size_t, const void*, size_t, CUstream) noexcept nogil>__cuMemcpyHtoAAsync_v2)(
        dstArray, dstOffset, srcHost, ByteCount, hStream)


cdef CUresult _cuMemcpyAtoHAsync_v2(void* dstHost, CUarray srcArray, size_t srcOffset, size_t ByteCount, CUstream hStream) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuMemcpyAtoHAsync_v2
    _check_or_init_driver()
    if __cuMemcpyAtoHAsync_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemcpyAtoHAsync_v2 is not found")
    return (<CUresult (*)(void*, CUarray, size_t, size_t, CUstream) noexcept nogil>__cuMemcpyAtoHAsync_v2)(
        dstHost, srcArray, srcOffset, ByteCount, hStream)


cdef CUresult _cuMemcpy2DAsync_v2(const CUDA_MEMCPY2D* pCopy, CUstream hStream) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuMemcpy2DAsync_v2
    _check_or_init_driver()
    if __cuMemcpy2DAsync_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemcpy2DAsync_v2 is not found")
    return (<CUresult (*)(const CUDA_MEMCPY2D*, CUstream) noexcept nogil>__cuMemcpy2DAsync_v2)(
        pCopy, hStream)


cdef CUresult _cuMemcpy3DAsync_v2(const CUDA_MEMCPY3D* pCopy, CUstream hStream) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuMemcpy3DAsync_v2
    _check_or_init_driver()
    if __cuMemcpy3DAsync_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemcpy3DAsync_v2 is not found")
    return (<CUresult (*)(const CUDA_MEMCPY3D*, CUstream) noexcept nogil>__cuMemcpy3DAsync_v2)(
        pCopy, hStream)


cdef CUresult _cuMemcpy3DPeerAsync(const CUDA_MEMCPY3D_PEER* pCopy, CUstream hStream) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuMemcpy3DPeerAsync
    _check_or_init_driver()
    if __cuMemcpy3DPeerAsync == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemcpy3DPeerAsync is not found")
    return (<CUresult (*)(const CUDA_MEMCPY3D_PEER*, CUstream) noexcept nogil>__cuMemcpy3DPeerAsync)(
        pCopy, hStream)


cdef CUresult _cuMemsetD8_v2(CUdeviceptr dstDevice, unsigned char uc, size_t N) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuMemsetD8_v2
    _check_or_init_driver()
    if __cuMemsetD8_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemsetD8_v2 is not found")
    return (<CUresult (*)(CUdeviceptr, unsigned char, size_t) noexcept nogil>__cuMemsetD8_v2)(
        dstDevice, uc, N)


cdef CUresult _cuMemsetD16_v2(CUdeviceptr dstDevice, unsigned short us, size_t N) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuMemsetD16_v2
    _check_or_init_driver()
    if __cuMemsetD16_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemsetD16_v2 is not found")
    return (<CUresult (*)(CUdeviceptr, unsigned short, size_t) noexcept nogil>__cuMemsetD16_v2)(
        dstDevice, us, N)


cdef CUresult _cuMemsetD32_v2(CUdeviceptr dstDevice, unsigned int ui, size_t N) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuMemsetD32_v2
    _check_or_init_driver()
    if __cuMemsetD32_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemsetD32_v2 is not found")
    return (<CUresult (*)(CUdeviceptr, unsigned int, size_t) noexcept nogil>__cuMemsetD32_v2)(
        dstDevice, ui, N)


cdef CUresult _cuMemsetD2D8_v2(CUdeviceptr dstDevice, size_t dstPitch, unsigned char uc, size_t Width, size_t Height) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuMemsetD2D8_v2
    _check_or_init_driver()
    if __cuMemsetD2D8_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemsetD2D8_v2 is not found")
    return (<CUresult (*)(CUdeviceptr, size_t, unsigned char, size_t, size_t) noexcept nogil>__cuMemsetD2D8_v2)(
        dstDevice, dstPitch, uc, Width, Height)


cdef CUresult _cuMemsetD2D16_v2(CUdeviceptr dstDevice, size_t dstPitch, unsigned short us, size_t Width, size_t Height) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuMemsetD2D16_v2
    _check_or_init_driver()
    if __cuMemsetD2D16_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemsetD2D16_v2 is not found")
    return (<CUresult (*)(CUdeviceptr, size_t, unsigned short, size_t, size_t) noexcept nogil>__cuMemsetD2D16_v2)(
        dstDevice, dstPitch, us, Width, Height)


cdef CUresult _cuMemsetD2D32_v2(CUdeviceptr dstDevice, size_t dstPitch, unsigned int ui, size_t Width, size_t Height) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuMemsetD2D32_v2
    _check_or_init_driver()
    if __cuMemsetD2D32_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemsetD2D32_v2 is not found")
    return (<CUresult (*)(CUdeviceptr, size_t, unsigned int, size_t, size_t) noexcept nogil>__cuMemsetD2D32_v2)(
        dstDevice, dstPitch, ui, Width, Height)


cdef CUresult _cuMemsetD8Async(CUdeviceptr dstDevice, unsigned char uc, size_t N, CUstream hStream) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuMemsetD8Async
    _check_or_init_driver()
    if __cuMemsetD8Async == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemsetD8Async is not found")
    return (<CUresult (*)(CUdeviceptr, unsigned char, size_t, CUstream) noexcept nogil>__cuMemsetD8Async)(
        dstDevice, uc, N, hStream)


cdef CUresult _cuMemsetD16Async(CUdeviceptr dstDevice, unsigned short us, size_t N, CUstream hStream) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuMemsetD16Async
    _check_or_init_driver()
    if __cuMemsetD16Async == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemsetD16Async is not found")
    return (<CUresult (*)(CUdeviceptr, unsigned short, size_t, CUstream) noexcept nogil>__cuMemsetD16Async)(
        dstDevice, us, N, hStream)


cdef CUresult _cuMemsetD32Async(CUdeviceptr dstDevice, unsigned int ui, size_t N, CUstream hStream) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuMemsetD32Async
    _check_or_init_driver()
    if __cuMemsetD32Async == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemsetD32Async is not found")
    return (<CUresult (*)(CUdeviceptr, unsigned int, size_t, CUstream) noexcept nogil>__cuMemsetD32Async)(
        dstDevice, ui, N, hStream)


cdef CUresult _cuMemsetD2D8Async(CUdeviceptr dstDevice, size_t dstPitch, unsigned char uc, size_t Width, size_t Height, CUstream hStream) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuMemsetD2D8Async
    _check_or_init_driver()
    if __cuMemsetD2D8Async == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemsetD2D8Async is not found")
    return (<CUresult (*)(CUdeviceptr, size_t, unsigned char, size_t, size_t, CUstream) noexcept nogil>__cuMemsetD2D8Async)(
        dstDevice, dstPitch, uc, Width, Height, hStream)


cdef CUresult _cuMemsetD2D16Async(CUdeviceptr dstDevice, size_t dstPitch, unsigned short us, size_t Width, size_t Height, CUstream hStream) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuMemsetD2D16Async
    _check_or_init_driver()
    if __cuMemsetD2D16Async == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemsetD2D16Async is not found")
    return (<CUresult (*)(CUdeviceptr, size_t, unsigned short, size_t, size_t, CUstream) noexcept nogil>__cuMemsetD2D16Async)(
        dstDevice, dstPitch, us, Width, Height, hStream)


cdef CUresult _cuMemsetD2D32Async(CUdeviceptr dstDevice, size_t dstPitch, unsigned int ui, size_t Width, size_t Height, CUstream hStream) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuMemsetD2D32Async
    _check_or_init_driver()
    if __cuMemsetD2D32Async == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemsetD2D32Async is not found")
    return (<CUresult (*)(CUdeviceptr, size_t, unsigned int, size_t, size_t, CUstream) noexcept nogil>__cuMemsetD2D32Async)(
        dstDevice, dstPitch, ui, Width, Height, hStream)


cdef CUresult _cuArrayCreate_v2(CUarray* pHandle, const CUDA_ARRAY_DESCRIPTOR* pAllocateArray) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuArrayCreate_v2
    _check_or_init_driver()
    if __cuArrayCreate_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuArrayCreate_v2 is not found")
    return (<CUresult (*)(CUarray*, const CUDA_ARRAY_DESCRIPTOR*) noexcept nogil>__cuArrayCreate_v2)(
        pHandle, pAllocateArray)


cdef CUresult _cuArrayGetDescriptor_v2(CUDA_ARRAY_DESCRIPTOR* pArrayDescriptor, CUarray hArray) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuArrayGetDescriptor_v2
    _check_or_init_driver()
    if __cuArrayGetDescriptor_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuArrayGetDescriptor_v2 is not found")
    return (<CUresult (*)(CUDA_ARRAY_DESCRIPTOR*, CUarray) noexcept nogil>__cuArrayGetDescriptor_v2)(
        pArrayDescriptor, hArray)


cdef CUresult _cuArrayGetSparseProperties(CUDA_ARRAY_SPARSE_PROPERTIES* sparseProperties, CUarray array) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuArrayGetSparseProperties
    _check_or_init_driver()
    if __cuArrayGetSparseProperties == NULL:
        with gil:
            raise FunctionNotFoundError("function cuArrayGetSparseProperties is not found")
    return (<CUresult (*)(CUDA_ARRAY_SPARSE_PROPERTIES*, CUarray) noexcept nogil>__cuArrayGetSparseProperties)(
        sparseProperties, array)


cdef CUresult _cuMipmappedArrayGetSparseProperties(CUDA_ARRAY_SPARSE_PROPERTIES* sparseProperties, CUmipmappedArray mipmap) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuMipmappedArrayGetSparseProperties
    _check_or_init_driver()
    if __cuMipmappedArrayGetSparseProperties == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMipmappedArrayGetSparseProperties is not found")
    return (<CUresult (*)(CUDA_ARRAY_SPARSE_PROPERTIES*, CUmipmappedArray) noexcept nogil>__cuMipmappedArrayGetSparseProperties)(
        sparseProperties, mipmap)


cdef CUresult _cuArrayGetMemoryRequirements(CUDA_ARRAY_MEMORY_REQUIREMENTS* memoryRequirements, CUarray array, CUdevice device) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuArrayGetMemoryRequirements
    _check_or_init_driver()
    if __cuArrayGetMemoryRequirements == NULL:
        with gil:
            raise FunctionNotFoundError("function cuArrayGetMemoryRequirements is not found")
    return (<CUresult (*)(CUDA_ARRAY_MEMORY_REQUIREMENTS*, CUarray, CUdevice) noexcept nogil>__cuArrayGetMemoryRequirements)(
        memoryRequirements, array, device)


cdef CUresult _cuMipmappedArrayGetMemoryRequirements(CUDA_ARRAY_MEMORY_REQUIREMENTS* memoryRequirements, CUmipmappedArray mipmap, CUdevice device) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuMipmappedArrayGetMemoryRequirements
    _check_or_init_driver()
    if __cuMipmappedArrayGetMemoryRequirements == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMipmappedArrayGetMemoryRequirements is not found")
    return (<CUresult (*)(CUDA_ARRAY_MEMORY_REQUIREMENTS*, CUmipmappedArray, CUdevice) noexcept nogil>__cuMipmappedArrayGetMemoryRequirements)(
        memoryRequirements, mipmap, device)


cdef CUresult _cuArrayGetPlane(CUarray* pPlaneArray, CUarray hArray, unsigned int planeIdx) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuArrayGetPlane
    _check_or_init_driver()
    if __cuArrayGetPlane == NULL:
        with gil:
            raise FunctionNotFoundError("function cuArrayGetPlane is not found")
    return (<CUresult (*)(CUarray*, CUarray, unsigned int) noexcept nogil>__cuArrayGetPlane)(
        pPlaneArray, hArray, planeIdx)


cdef CUresult _cuArrayDestroy(CUarray hArray) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuArrayDestroy
    _check_or_init_driver()
    if __cuArrayDestroy == NULL:
        with gil:
            raise FunctionNotFoundError("function cuArrayDestroy is not found")
    return (<CUresult (*)(CUarray) noexcept nogil>__cuArrayDestroy)(
        hArray)


cdef CUresult _cuArray3DCreate_v2(CUarray* pHandle, const CUDA_ARRAY3D_DESCRIPTOR* pAllocateArray) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuArray3DCreate_v2
    _check_or_init_driver()
    if __cuArray3DCreate_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuArray3DCreate_v2 is not found")
    return (<CUresult (*)(CUarray*, const CUDA_ARRAY3D_DESCRIPTOR*) noexcept nogil>__cuArray3DCreate_v2)(
        pHandle, pAllocateArray)


cdef CUresult _cuArray3DGetDescriptor_v2(CUDA_ARRAY3D_DESCRIPTOR* pArrayDescriptor, CUarray hArray) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuArray3DGetDescriptor_v2
    _check_or_init_driver()
    if __cuArray3DGetDescriptor_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuArray3DGetDescriptor_v2 is not found")
    return (<CUresult (*)(CUDA_ARRAY3D_DESCRIPTOR*, CUarray) noexcept nogil>__cuArray3DGetDescriptor_v2)(
        pArrayDescriptor, hArray)


cdef CUresult _cuMipmappedArrayCreate(CUmipmappedArray* pHandle, const CUDA_ARRAY3D_DESCRIPTOR* pMipmappedArrayDesc, unsigned int numMipmapLevels) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuMipmappedArrayCreate
    _check_or_init_driver()
    if __cuMipmappedArrayCreate == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMipmappedArrayCreate is not found")
    return (<CUresult (*)(CUmipmappedArray*, const CUDA_ARRAY3D_DESCRIPTOR*, unsigned int) noexcept nogil>__cuMipmappedArrayCreate)(
        pHandle, pMipmappedArrayDesc, numMipmapLevels)


cdef CUresult _cuMipmappedArrayGetLevel(CUarray* pLevelArray, CUmipmappedArray hMipmappedArray, unsigned int level) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuMipmappedArrayGetLevel
    _check_or_init_driver()
    if __cuMipmappedArrayGetLevel == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMipmappedArrayGetLevel is not found")
    return (<CUresult (*)(CUarray*, CUmipmappedArray, unsigned int) noexcept nogil>__cuMipmappedArrayGetLevel)(
        pLevelArray, hMipmappedArray, level)


cdef CUresult _cuMipmappedArrayDestroy(CUmipmappedArray hMipmappedArray) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuMipmappedArrayDestroy
    _check_or_init_driver()
    if __cuMipmappedArrayDestroy == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMipmappedArrayDestroy is not found")
    return (<CUresult (*)(CUmipmappedArray) noexcept nogil>__cuMipmappedArrayDestroy)(
        hMipmappedArray)


cdef CUresult _cuMemGetHandleForAddressRange(void* handle, CUdeviceptr dptr, size_t size, CUmemRangeHandleType handleType, unsigned long long flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuMemGetHandleForAddressRange
    _check_or_init_driver()
    if __cuMemGetHandleForAddressRange == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemGetHandleForAddressRange is not found")
    return (<CUresult (*)(void*, CUdeviceptr, size_t, CUmemRangeHandleType, unsigned long long) noexcept nogil>__cuMemGetHandleForAddressRange)(
        handle, dptr, size, handleType, flags)


cdef CUresult _cuMemBatchDecompressAsync(CUmemDecompressParams* paramsArray, size_t count, unsigned int flags, size_t* errorIndex, CUstream stream) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuMemBatchDecompressAsync
    _check_or_init_driver()
    if __cuMemBatchDecompressAsync == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemBatchDecompressAsync is not found")
    return (<CUresult (*)(CUmemDecompressParams*, size_t, unsigned int, size_t*, CUstream) noexcept nogil>__cuMemBatchDecompressAsync)(
        paramsArray, count, flags, errorIndex, stream)


cdef CUresult _cuMemAddressReserve(CUdeviceptr* ptr, size_t size, size_t alignment, CUdeviceptr addr, unsigned long long flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuMemAddressReserve
    _check_or_init_driver()
    if __cuMemAddressReserve == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemAddressReserve is not found")
    return (<CUresult (*)(CUdeviceptr*, size_t, size_t, CUdeviceptr, unsigned long long) noexcept nogil>__cuMemAddressReserve)(
        ptr, size, alignment, addr, flags)


cdef CUresult _cuMemAddressFree(CUdeviceptr ptr, size_t size) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuMemAddressFree
    _check_or_init_driver()
    if __cuMemAddressFree == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemAddressFree is not found")
    return (<CUresult (*)(CUdeviceptr, size_t) noexcept nogil>__cuMemAddressFree)(
        ptr, size)


cdef CUresult _cuMemCreate(CUmemGenericAllocationHandle* handle, size_t size, const CUmemAllocationProp* prop, unsigned long long flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuMemCreate
    _check_or_init_driver()
    if __cuMemCreate == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemCreate is not found")
    return (<CUresult (*)(CUmemGenericAllocationHandle*, size_t, const CUmemAllocationProp*, unsigned long long) noexcept nogil>__cuMemCreate)(
        handle, size, prop, flags)


cdef CUresult _cuMemRelease(CUmemGenericAllocationHandle handle) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuMemRelease
    _check_or_init_driver()
    if __cuMemRelease == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemRelease is not found")
    return (<CUresult (*)(CUmemGenericAllocationHandle) noexcept nogil>__cuMemRelease)(
        handle)


cdef CUresult _cuMemMap(CUdeviceptr ptr, size_t size, size_t offset, CUmemGenericAllocationHandle handle, unsigned long long flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuMemMap
    _check_or_init_driver()
    if __cuMemMap == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemMap is not found")
    return (<CUresult (*)(CUdeviceptr, size_t, size_t, CUmemGenericAllocationHandle, unsigned long long) noexcept nogil>__cuMemMap)(
        ptr, size, offset, handle, flags)


cdef CUresult _cuMemMapArrayAsync(CUarrayMapInfo* mapInfoList, unsigned int count, CUstream hStream) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuMemMapArrayAsync
    _check_or_init_driver()
    if __cuMemMapArrayAsync == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemMapArrayAsync is not found")
    return (<CUresult (*)(CUarrayMapInfo*, unsigned int, CUstream) noexcept nogil>__cuMemMapArrayAsync)(
        mapInfoList, count, hStream)


cdef CUresult _cuMemUnmap(CUdeviceptr ptr, size_t size) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuMemUnmap
    _check_or_init_driver()
    if __cuMemUnmap == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemUnmap is not found")
    return (<CUresult (*)(CUdeviceptr, size_t) noexcept nogil>__cuMemUnmap)(
        ptr, size)


cdef CUresult _cuMemSetAccess(CUdeviceptr ptr, size_t size, const CUmemAccessDesc* desc, size_t count) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuMemSetAccess
    _check_or_init_driver()
    if __cuMemSetAccess == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemSetAccess is not found")
    return (<CUresult (*)(CUdeviceptr, size_t, const CUmemAccessDesc*, size_t) noexcept nogil>__cuMemSetAccess)(
        ptr, size, desc, count)


cdef CUresult _cuMemGetAccess(unsigned long long* flags, const CUmemLocation* location, CUdeviceptr ptr) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuMemGetAccess
    _check_or_init_driver()
    if __cuMemGetAccess == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemGetAccess is not found")
    return (<CUresult (*)(unsigned long long*, const CUmemLocation*, CUdeviceptr) noexcept nogil>__cuMemGetAccess)(
        flags, location, ptr)


cdef CUresult _cuMemExportToShareableHandle(void* shareableHandle, CUmemGenericAllocationHandle handle, CUmemAllocationHandleType handleType, unsigned long long flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuMemExportToShareableHandle
    _check_or_init_driver()
    if __cuMemExportToShareableHandle == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemExportToShareableHandle is not found")
    return (<CUresult (*)(void*, CUmemGenericAllocationHandle, CUmemAllocationHandleType, unsigned long long) noexcept nogil>__cuMemExportToShareableHandle)(
        shareableHandle, handle, handleType, flags)


cdef CUresult _cuMemImportFromShareableHandle(CUmemGenericAllocationHandle* handle, void* osHandle, CUmemAllocationHandleType shHandleType) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuMemImportFromShareableHandle
    _check_or_init_driver()
    if __cuMemImportFromShareableHandle == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemImportFromShareableHandle is not found")
    return (<CUresult (*)(CUmemGenericAllocationHandle*, void*, CUmemAllocationHandleType) noexcept nogil>__cuMemImportFromShareableHandle)(
        handle, osHandle, shHandleType)


cdef CUresult _cuMemGetAllocationGranularity(size_t* granularity, const CUmemAllocationProp* prop, CUmemAllocationGranularity_flags option) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuMemGetAllocationGranularity
    _check_or_init_driver()
    if __cuMemGetAllocationGranularity == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemGetAllocationGranularity is not found")
    return (<CUresult (*)(size_t*, const CUmemAllocationProp*, CUmemAllocationGranularity_flags) noexcept nogil>__cuMemGetAllocationGranularity)(
        granularity, prop, option)


cdef CUresult _cuMemGetAllocationPropertiesFromHandle(CUmemAllocationProp* prop, CUmemGenericAllocationHandle handle) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuMemGetAllocationPropertiesFromHandle
    _check_or_init_driver()
    if __cuMemGetAllocationPropertiesFromHandle == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemGetAllocationPropertiesFromHandle is not found")
    return (<CUresult (*)(CUmemAllocationProp*, CUmemGenericAllocationHandle) noexcept nogil>__cuMemGetAllocationPropertiesFromHandle)(
        prop, handle)


cdef CUresult _cuMemRetainAllocationHandle(CUmemGenericAllocationHandle* handle, void* addr) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuMemRetainAllocationHandle
    _check_or_init_driver()
    if __cuMemRetainAllocationHandle == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemRetainAllocationHandle is not found")
    return (<CUresult (*)(CUmemGenericAllocationHandle*, void*) noexcept nogil>__cuMemRetainAllocationHandle)(
        handle, addr)


cdef CUresult _cuMemFreeAsync(CUdeviceptr dptr, CUstream hStream) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuMemFreeAsync
    _check_or_init_driver()
    if __cuMemFreeAsync == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemFreeAsync is not found")
    return (<CUresult (*)(CUdeviceptr, CUstream) noexcept nogil>__cuMemFreeAsync)(
        dptr, hStream)


cdef CUresult _cuMemAllocAsync(CUdeviceptr* dptr, size_t bytesize, CUstream hStream) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuMemAllocAsync
    _check_or_init_driver()
    if __cuMemAllocAsync == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemAllocAsync is not found")
    return (<CUresult (*)(CUdeviceptr*, size_t, CUstream) noexcept nogil>__cuMemAllocAsync)(
        dptr, bytesize, hStream)


cdef CUresult _cuMemPoolTrimTo(CUmemoryPool pool, size_t minBytesToKeep) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuMemPoolTrimTo
    _check_or_init_driver()
    if __cuMemPoolTrimTo == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemPoolTrimTo is not found")
    return (<CUresult (*)(CUmemoryPool, size_t) noexcept nogil>__cuMemPoolTrimTo)(
        pool, minBytesToKeep)


cdef CUresult _cuMemPoolSetAttribute(CUmemoryPool pool, CUmemPool_attribute attr, void* value) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuMemPoolSetAttribute
    _check_or_init_driver()
    if __cuMemPoolSetAttribute == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemPoolSetAttribute is not found")
    return (<CUresult (*)(CUmemoryPool, CUmemPool_attribute, void*) noexcept nogil>__cuMemPoolSetAttribute)(
        pool, attr, value)


cdef CUresult _cuMemPoolGetAttribute(CUmemoryPool pool, CUmemPool_attribute attr, void* value) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuMemPoolGetAttribute
    _check_or_init_driver()
    if __cuMemPoolGetAttribute == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemPoolGetAttribute is not found")
    return (<CUresult (*)(CUmemoryPool, CUmemPool_attribute, void*) noexcept nogil>__cuMemPoolGetAttribute)(
        pool, attr, value)


cdef CUresult _cuMemPoolSetAccess(CUmemoryPool pool, const CUmemAccessDesc* map, size_t count) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuMemPoolSetAccess
    _check_or_init_driver()
    if __cuMemPoolSetAccess == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemPoolSetAccess is not found")
    return (<CUresult (*)(CUmemoryPool, const CUmemAccessDesc*, size_t) noexcept nogil>__cuMemPoolSetAccess)(
        pool, map, count)


cdef CUresult _cuMemPoolGetAccess(CUmemAccess_flags* flags, CUmemoryPool memPool, CUmemLocation* location) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuMemPoolGetAccess
    _check_or_init_driver()
    if __cuMemPoolGetAccess == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemPoolGetAccess is not found")
    return (<CUresult (*)(CUmemAccess_flags*, CUmemoryPool, CUmemLocation*) noexcept nogil>__cuMemPoolGetAccess)(
        flags, memPool, location)


cdef CUresult _cuMemPoolCreate(CUmemoryPool* pool, const CUmemPoolProps* poolProps) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuMemPoolCreate
    _check_or_init_driver()
    if __cuMemPoolCreate == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemPoolCreate is not found")
    return (<CUresult (*)(CUmemoryPool*, const CUmemPoolProps*) noexcept nogil>__cuMemPoolCreate)(
        pool, poolProps)


cdef CUresult _cuMemPoolDestroy(CUmemoryPool pool) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuMemPoolDestroy
    _check_or_init_driver()
    if __cuMemPoolDestroy == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemPoolDestroy is not found")
    return (<CUresult (*)(CUmemoryPool) noexcept nogil>__cuMemPoolDestroy)(
        pool)


cdef CUresult _cuMemAllocFromPoolAsync(CUdeviceptr* dptr, size_t bytesize, CUmemoryPool pool, CUstream hStream) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuMemAllocFromPoolAsync
    _check_or_init_driver()
    if __cuMemAllocFromPoolAsync == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemAllocFromPoolAsync is not found")
    return (<CUresult (*)(CUdeviceptr*, size_t, CUmemoryPool, CUstream) noexcept nogil>__cuMemAllocFromPoolAsync)(
        dptr, bytesize, pool, hStream)


cdef CUresult _cuMemPoolExportToShareableHandle(void* handle_out, CUmemoryPool pool, CUmemAllocationHandleType handleType, unsigned long long flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuMemPoolExportToShareableHandle
    _check_or_init_driver()
    if __cuMemPoolExportToShareableHandle == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemPoolExportToShareableHandle is not found")
    return (<CUresult (*)(void*, CUmemoryPool, CUmemAllocationHandleType, unsigned long long) noexcept nogil>__cuMemPoolExportToShareableHandle)(
        handle_out, pool, handleType, flags)


cdef CUresult _cuMemPoolImportFromShareableHandle(CUmemoryPool* pool_out, void* handle, CUmemAllocationHandleType handleType, unsigned long long flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuMemPoolImportFromShareableHandle
    _check_or_init_driver()
    if __cuMemPoolImportFromShareableHandle == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemPoolImportFromShareableHandle is not found")
    return (<CUresult (*)(CUmemoryPool*, void*, CUmemAllocationHandleType, unsigned long long) noexcept nogil>__cuMemPoolImportFromShareableHandle)(
        pool_out, handle, handleType, flags)


cdef CUresult _cuMemPoolExportPointer(CUmemPoolPtrExportData* shareData_out, CUdeviceptr ptr) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuMemPoolExportPointer
    _check_or_init_driver()
    if __cuMemPoolExportPointer == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemPoolExportPointer is not found")
    return (<CUresult (*)(CUmemPoolPtrExportData*, CUdeviceptr) noexcept nogil>__cuMemPoolExportPointer)(
        shareData_out, ptr)


cdef CUresult _cuMemPoolImportPointer(CUdeviceptr* ptr_out, CUmemoryPool pool, CUmemPoolPtrExportData* shareData) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuMemPoolImportPointer
    _check_or_init_driver()
    if __cuMemPoolImportPointer == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemPoolImportPointer is not found")
    return (<CUresult (*)(CUdeviceptr*, CUmemoryPool, CUmemPoolPtrExportData*) noexcept nogil>__cuMemPoolImportPointer)(
        ptr_out, pool, shareData)


cdef CUresult _cuMulticastCreate(CUmemGenericAllocationHandle* mcHandle, const CUmulticastObjectProp* prop) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuMulticastCreate
    _check_or_init_driver()
    if __cuMulticastCreate == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMulticastCreate is not found")
    return (<CUresult (*)(CUmemGenericAllocationHandle*, const CUmulticastObjectProp*) noexcept nogil>__cuMulticastCreate)(
        mcHandle, prop)


cdef CUresult _cuMulticastAddDevice(CUmemGenericAllocationHandle mcHandle, CUdevice dev) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuMulticastAddDevice
    _check_or_init_driver()
    if __cuMulticastAddDevice == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMulticastAddDevice is not found")
    return (<CUresult (*)(CUmemGenericAllocationHandle, CUdevice) noexcept nogil>__cuMulticastAddDevice)(
        mcHandle, dev)


cdef CUresult _cuMulticastBindMem(CUmemGenericAllocationHandle mcHandle, size_t mcOffset, CUmemGenericAllocationHandle memHandle, size_t memOffset, size_t size, unsigned long long flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuMulticastBindMem
    _check_or_init_driver()
    if __cuMulticastBindMem == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMulticastBindMem is not found")
    return (<CUresult (*)(CUmemGenericAllocationHandle, size_t, CUmemGenericAllocationHandle, size_t, size_t, unsigned long long) noexcept nogil>__cuMulticastBindMem)(
        mcHandle, mcOffset, memHandle, memOffset, size, flags)


cdef CUresult _cuMulticastBindAddr(CUmemGenericAllocationHandle mcHandle, size_t mcOffset, CUdeviceptr memptr, size_t size, unsigned long long flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuMulticastBindAddr
    _check_or_init_driver()
    if __cuMulticastBindAddr == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMulticastBindAddr is not found")
    return (<CUresult (*)(CUmemGenericAllocationHandle, size_t, CUdeviceptr, size_t, unsigned long long) noexcept nogil>__cuMulticastBindAddr)(
        mcHandle, mcOffset, memptr, size, flags)


cdef CUresult _cuMulticastUnbind(CUmemGenericAllocationHandle mcHandle, CUdevice dev, size_t mcOffset, size_t size) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuMulticastUnbind
    _check_or_init_driver()
    if __cuMulticastUnbind == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMulticastUnbind is not found")
    return (<CUresult (*)(CUmemGenericAllocationHandle, CUdevice, size_t, size_t) noexcept nogil>__cuMulticastUnbind)(
        mcHandle, dev, mcOffset, size)


cdef CUresult _cuMulticastGetGranularity(size_t* granularity, const CUmulticastObjectProp* prop, CUmulticastGranularity_flags option) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuMulticastGetGranularity
    _check_or_init_driver()
    if __cuMulticastGetGranularity == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMulticastGetGranularity is not found")
    return (<CUresult (*)(size_t*, const CUmulticastObjectProp*, CUmulticastGranularity_flags) noexcept nogil>__cuMulticastGetGranularity)(
        granularity, prop, option)


cdef CUresult _cuPointerGetAttribute(void* data, CUpointer_attribute attribute, CUdeviceptr ptr) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuPointerGetAttribute
    _check_or_init_driver()
    if __cuPointerGetAttribute == NULL:
        with gil:
            raise FunctionNotFoundError("function cuPointerGetAttribute is not found")
    return (<CUresult (*)(void*, CUpointer_attribute, CUdeviceptr) noexcept nogil>__cuPointerGetAttribute)(
        data, attribute, ptr)


cdef CUresult _cuMemPrefetchAsync_v2(CUdeviceptr devPtr, size_t count, CUmemLocation location, unsigned int flags, CUstream hStream) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuMemPrefetchAsync_v2
    _check_or_init_driver()
    if __cuMemPrefetchAsync_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemPrefetchAsync_v2 is not found")
    return (<CUresult (*)(CUdeviceptr, size_t, CUmemLocation, unsigned int, CUstream) noexcept nogil>__cuMemPrefetchAsync_v2)(
        devPtr, count, location, flags, hStream)


cdef CUresult _cuMemAdvise_v2(CUdeviceptr devPtr, size_t count, CUmem_advise advice, CUmemLocation location) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuMemAdvise_v2
    _check_or_init_driver()
    if __cuMemAdvise_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemAdvise_v2 is not found")
    return (<CUresult (*)(CUdeviceptr, size_t, CUmem_advise, CUmemLocation) noexcept nogil>__cuMemAdvise_v2)(
        devPtr, count, advice, location)


cdef CUresult _cuMemRangeGetAttribute(void* data, size_t dataSize, CUmem_range_attribute attribute, CUdeviceptr devPtr, size_t count) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuMemRangeGetAttribute
    _check_or_init_driver()
    if __cuMemRangeGetAttribute == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemRangeGetAttribute is not found")
    return (<CUresult (*)(void*, size_t, CUmem_range_attribute, CUdeviceptr, size_t) noexcept nogil>__cuMemRangeGetAttribute)(
        data, dataSize, attribute, devPtr, count)


cdef CUresult _cuMemRangeGetAttributes(void** data, size_t* dataSizes, CUmem_range_attribute* attributes, size_t numAttributes, CUdeviceptr devPtr, size_t count) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuMemRangeGetAttributes
    _check_or_init_driver()
    if __cuMemRangeGetAttributes == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemRangeGetAttributes is not found")
    return (<CUresult (*)(void**, size_t*, CUmem_range_attribute*, size_t, CUdeviceptr, size_t) noexcept nogil>__cuMemRangeGetAttributes)(
        data, dataSizes, attributes, numAttributes, devPtr, count)


cdef CUresult _cuPointerSetAttribute(const void* value, CUpointer_attribute attribute, CUdeviceptr ptr) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuPointerSetAttribute
    _check_or_init_driver()
    if __cuPointerSetAttribute == NULL:
        with gil:
            raise FunctionNotFoundError("function cuPointerSetAttribute is not found")
    return (<CUresult (*)(const void*, CUpointer_attribute, CUdeviceptr) noexcept nogil>__cuPointerSetAttribute)(
        value, attribute, ptr)


cdef CUresult _cuPointerGetAttributes(unsigned int numAttributes, CUpointer_attribute* attributes, void** data, CUdeviceptr ptr) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuPointerGetAttributes
    _check_or_init_driver()
    if __cuPointerGetAttributes == NULL:
        with gil:
            raise FunctionNotFoundError("function cuPointerGetAttributes is not found")
    return (<CUresult (*)(unsigned int, CUpointer_attribute*, void**, CUdeviceptr) noexcept nogil>__cuPointerGetAttributes)(
        numAttributes, attributes, data, ptr)


cdef CUresult _cuStreamCreate(CUstream* phStream, unsigned int Flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuStreamCreate
    _check_or_init_driver()
    if __cuStreamCreate == NULL:
        with gil:
            raise FunctionNotFoundError("function cuStreamCreate is not found")
    return (<CUresult (*)(CUstream*, unsigned int) noexcept nogil>__cuStreamCreate)(
        phStream, Flags)


cdef CUresult _cuStreamCreateWithPriority(CUstream* phStream, unsigned int flags, int priority) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuStreamCreateWithPriority
    _check_or_init_driver()
    if __cuStreamCreateWithPriority == NULL:
        with gil:
            raise FunctionNotFoundError("function cuStreamCreateWithPriority is not found")
    return (<CUresult (*)(CUstream*, unsigned int, int) noexcept nogil>__cuStreamCreateWithPriority)(
        phStream, flags, priority)


cdef CUresult _cuStreamGetPriority(CUstream hStream, int* priority) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuStreamGetPriority
    _check_or_init_driver()
    if __cuStreamGetPriority == NULL:
        with gil:
            raise FunctionNotFoundError("function cuStreamGetPriority is not found")
    return (<CUresult (*)(CUstream, int*) noexcept nogil>__cuStreamGetPriority)(
        hStream, priority)


cdef CUresult _cuStreamGetDevice(CUstream hStream, CUdevice* device) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuStreamGetDevice
    _check_or_init_driver()
    if __cuStreamGetDevice == NULL:
        with gil:
            raise FunctionNotFoundError("function cuStreamGetDevice is not found")
    return (<CUresult (*)(CUstream, CUdevice*) noexcept nogil>__cuStreamGetDevice)(
        hStream, device)


cdef CUresult _cuStreamGetFlags(CUstream hStream, unsigned int* flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuStreamGetFlags
    _check_or_init_driver()
    if __cuStreamGetFlags == NULL:
        with gil:
            raise FunctionNotFoundError("function cuStreamGetFlags is not found")
    return (<CUresult (*)(CUstream, unsigned int*) noexcept nogil>__cuStreamGetFlags)(
        hStream, flags)


cdef CUresult _cuStreamGetId(CUstream hStream, unsigned long long* streamId) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuStreamGetId
    _check_or_init_driver()
    if __cuStreamGetId == NULL:
        with gil:
            raise FunctionNotFoundError("function cuStreamGetId is not found")
    return (<CUresult (*)(CUstream, unsigned long long*) noexcept nogil>__cuStreamGetId)(
        hStream, streamId)


cdef CUresult _cuStreamGetCtx(CUstream hStream, CUcontext* pctx) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuStreamGetCtx
    _check_or_init_driver()
    if __cuStreamGetCtx == NULL:
        with gil:
            raise FunctionNotFoundError("function cuStreamGetCtx is not found")
    return (<CUresult (*)(CUstream, CUcontext*) noexcept nogil>__cuStreamGetCtx)(
        hStream, pctx)


cdef CUresult _cuStreamGetCtx_v2(CUstream hStream, CUcontext* pCtx, CUgreenCtx* pGreenCtx) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuStreamGetCtx_v2
    _check_or_init_driver()
    if __cuStreamGetCtx_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuStreamGetCtx_v2 is not found")
    return (<CUresult (*)(CUstream, CUcontext*, CUgreenCtx*) noexcept nogil>__cuStreamGetCtx_v2)(
        hStream, pCtx, pGreenCtx)


cdef CUresult _cuStreamWaitEvent(CUstream hStream, CUevent hEvent, unsigned int Flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuStreamWaitEvent
    _check_or_init_driver()
    if __cuStreamWaitEvent == NULL:
        with gil:
            raise FunctionNotFoundError("function cuStreamWaitEvent is not found")
    return (<CUresult (*)(CUstream, CUevent, unsigned int) noexcept nogil>__cuStreamWaitEvent)(
        hStream, hEvent, Flags)


cdef CUresult _cuStreamAddCallback(CUstream hStream, CUstreamCallback callback, void* userData, unsigned int flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuStreamAddCallback
    _check_or_init_driver()
    if __cuStreamAddCallback == NULL:
        with gil:
            raise FunctionNotFoundError("function cuStreamAddCallback is not found")
    return (<CUresult (*)(CUstream, CUstreamCallback, void*, unsigned int) noexcept nogil>__cuStreamAddCallback)(
        hStream, callback, userData, flags)


cdef CUresult _cuStreamBeginCapture_v2(CUstream hStream, CUstreamCaptureMode mode) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuStreamBeginCapture_v2
    _check_or_init_driver()
    if __cuStreamBeginCapture_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuStreamBeginCapture_v2 is not found")
    return (<CUresult (*)(CUstream, CUstreamCaptureMode) noexcept nogil>__cuStreamBeginCapture_v2)(
        hStream, mode)


cdef CUresult _cuStreamBeginCaptureToGraph(CUstream hStream, CUgraph hGraph, const CUgraphNode* dependencies, const CUgraphEdgeData* dependencyData, size_t numDependencies, CUstreamCaptureMode mode) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuStreamBeginCaptureToGraph
    _check_or_init_driver()
    if __cuStreamBeginCaptureToGraph == NULL:
        with gil:
            raise FunctionNotFoundError("function cuStreamBeginCaptureToGraph is not found")
    return (<CUresult (*)(CUstream, CUgraph, const CUgraphNode*, const CUgraphEdgeData*, size_t, CUstreamCaptureMode) noexcept nogil>__cuStreamBeginCaptureToGraph)(
        hStream, hGraph, dependencies, dependencyData, numDependencies, mode)


cdef CUresult _cuThreadExchangeStreamCaptureMode(CUstreamCaptureMode* mode) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuThreadExchangeStreamCaptureMode
    _check_or_init_driver()
    if __cuThreadExchangeStreamCaptureMode == NULL:
        with gil:
            raise FunctionNotFoundError("function cuThreadExchangeStreamCaptureMode is not found")
    return (<CUresult (*)(CUstreamCaptureMode*) noexcept nogil>__cuThreadExchangeStreamCaptureMode)(
        mode)


cdef CUresult _cuStreamEndCapture(CUstream hStream, CUgraph* phGraph) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuStreamEndCapture
    _check_or_init_driver()
    if __cuStreamEndCapture == NULL:
        with gil:
            raise FunctionNotFoundError("function cuStreamEndCapture is not found")
    return (<CUresult (*)(CUstream, CUgraph*) noexcept nogil>__cuStreamEndCapture)(
        hStream, phGraph)


cdef CUresult _cuStreamIsCapturing(CUstream hStream, CUstreamCaptureStatus* captureStatus) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuStreamIsCapturing
    _check_or_init_driver()
    if __cuStreamIsCapturing == NULL:
        with gil:
            raise FunctionNotFoundError("function cuStreamIsCapturing is not found")
    return (<CUresult (*)(CUstream, CUstreamCaptureStatus*) noexcept nogil>__cuStreamIsCapturing)(
        hStream, captureStatus)


cdef CUresult _cuStreamGetCaptureInfo_v3(CUstream hStream, CUstreamCaptureStatus* captureStatus_out, cuuint64_t* id_out, CUgraph* graph_out, const CUgraphNode** dependencies_out, const CUgraphEdgeData** edgeData_out, size_t* numDependencies_out) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuStreamGetCaptureInfo_v3
    _check_or_init_driver()
    if __cuStreamGetCaptureInfo_v3 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuStreamGetCaptureInfo_v3 is not found")
    return (<CUresult (*)(CUstream, CUstreamCaptureStatus*, cuuint64_t*, CUgraph*, const CUgraphNode**, const CUgraphEdgeData**, size_t*) noexcept nogil>__cuStreamGetCaptureInfo_v3)(
        hStream, captureStatus_out, id_out, graph_out, dependencies_out, edgeData_out, numDependencies_out)


cdef CUresult _cuStreamUpdateCaptureDependencies_v2(CUstream hStream, CUgraphNode* dependencies, const CUgraphEdgeData* dependencyData, size_t numDependencies, unsigned int flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuStreamUpdateCaptureDependencies_v2
    _check_or_init_driver()
    if __cuStreamUpdateCaptureDependencies_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuStreamUpdateCaptureDependencies_v2 is not found")
    return (<CUresult (*)(CUstream, CUgraphNode*, const CUgraphEdgeData*, size_t, unsigned int) noexcept nogil>__cuStreamUpdateCaptureDependencies_v2)(
        hStream, dependencies, dependencyData, numDependencies, flags)


cdef CUresult _cuStreamAttachMemAsync(CUstream hStream, CUdeviceptr dptr, size_t length, unsigned int flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuStreamAttachMemAsync
    _check_or_init_driver()
    if __cuStreamAttachMemAsync == NULL:
        with gil:
            raise FunctionNotFoundError("function cuStreamAttachMemAsync is not found")
    return (<CUresult (*)(CUstream, CUdeviceptr, size_t, unsigned int) noexcept nogil>__cuStreamAttachMemAsync)(
        hStream, dptr, length, flags)


cdef CUresult _cuStreamQuery(CUstream hStream) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuStreamQuery
    _check_or_init_driver()
    if __cuStreamQuery == NULL:
        with gil:
            raise FunctionNotFoundError("function cuStreamQuery is not found")
    return (<CUresult (*)(CUstream) noexcept nogil>__cuStreamQuery)(
        hStream)


cdef CUresult _cuStreamSynchronize(CUstream hStream) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuStreamSynchronize
    _check_or_init_driver()
    if __cuStreamSynchronize == NULL:
        with gil:
            raise FunctionNotFoundError("function cuStreamSynchronize is not found")
    return (<CUresult (*)(CUstream) noexcept nogil>__cuStreamSynchronize)(
        hStream)


cdef CUresult _cuStreamDestroy_v2(CUstream hStream) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuStreamDestroy_v2
    _check_or_init_driver()
    if __cuStreamDestroy_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuStreamDestroy_v2 is not found")
    return (<CUresult (*)(CUstream) noexcept nogil>__cuStreamDestroy_v2)(
        hStream)


cdef CUresult _cuStreamCopyAttributes(CUstream dst, CUstream src) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuStreamCopyAttributes
    _check_or_init_driver()
    if __cuStreamCopyAttributes == NULL:
        with gil:
            raise FunctionNotFoundError("function cuStreamCopyAttributes is not found")
    return (<CUresult (*)(CUstream, CUstream) noexcept nogil>__cuStreamCopyAttributes)(
        dst, src)


cdef CUresult _cuStreamGetAttribute(CUstream hStream, CUstreamAttrID attr, CUstreamAttrValue* value_out) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuStreamGetAttribute
    _check_or_init_driver()
    if __cuStreamGetAttribute == NULL:
        with gil:
            raise FunctionNotFoundError("function cuStreamGetAttribute is not found")
    return (<CUresult (*)(CUstream, CUstreamAttrID, CUstreamAttrValue*) noexcept nogil>__cuStreamGetAttribute)(
        hStream, attr, value_out)


cdef CUresult _cuStreamSetAttribute(CUstream hStream, CUstreamAttrID attr, const CUstreamAttrValue* value) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuStreamSetAttribute
    _check_or_init_driver()
    if __cuStreamSetAttribute == NULL:
        with gil:
            raise FunctionNotFoundError("function cuStreamSetAttribute is not found")
    return (<CUresult (*)(CUstream, CUstreamAttrID, const CUstreamAttrValue*) noexcept nogil>__cuStreamSetAttribute)(
        hStream, attr, value)


cdef CUresult _cuEventCreate(CUevent* phEvent, unsigned int Flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuEventCreate
    _check_or_init_driver()
    if __cuEventCreate == NULL:
        with gil:
            raise FunctionNotFoundError("function cuEventCreate is not found")
    return (<CUresult (*)(CUevent*, unsigned int) noexcept nogil>__cuEventCreate)(
        phEvent, Flags)


cdef CUresult _cuEventRecord(CUevent hEvent, CUstream hStream) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuEventRecord
    _check_or_init_driver()
    if __cuEventRecord == NULL:
        with gil:
            raise FunctionNotFoundError("function cuEventRecord is not found")
    return (<CUresult (*)(CUevent, CUstream) noexcept nogil>__cuEventRecord)(
        hEvent, hStream)


cdef CUresult _cuEventRecordWithFlags(CUevent hEvent, CUstream hStream, unsigned int flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuEventRecordWithFlags
    _check_or_init_driver()
    if __cuEventRecordWithFlags == NULL:
        with gil:
            raise FunctionNotFoundError("function cuEventRecordWithFlags is not found")
    return (<CUresult (*)(CUevent, CUstream, unsigned int) noexcept nogil>__cuEventRecordWithFlags)(
        hEvent, hStream, flags)


cdef CUresult _cuEventQuery(CUevent hEvent) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuEventQuery
    _check_or_init_driver()
    if __cuEventQuery == NULL:
        with gil:
            raise FunctionNotFoundError("function cuEventQuery is not found")
    return (<CUresult (*)(CUevent) noexcept nogil>__cuEventQuery)(
        hEvent)


cdef CUresult _cuEventSynchronize(CUevent hEvent) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuEventSynchronize
    _check_or_init_driver()
    if __cuEventSynchronize == NULL:
        with gil:
            raise FunctionNotFoundError("function cuEventSynchronize is not found")
    return (<CUresult (*)(CUevent) noexcept nogil>__cuEventSynchronize)(
        hEvent)


cdef CUresult _cuEventDestroy_v2(CUevent hEvent) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuEventDestroy_v2
    _check_or_init_driver()
    if __cuEventDestroy_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuEventDestroy_v2 is not found")
    return (<CUresult (*)(CUevent) noexcept nogil>__cuEventDestroy_v2)(
        hEvent)


cdef CUresult _cuEventElapsedTime_v2(float* pMilliseconds, CUevent hStart, CUevent hEnd) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuEventElapsedTime_v2
    _check_or_init_driver()
    if __cuEventElapsedTime_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuEventElapsedTime_v2 is not found")
    return (<CUresult (*)(float*, CUevent, CUevent) noexcept nogil>__cuEventElapsedTime_v2)(
        pMilliseconds, hStart, hEnd)


cdef CUresult _cuImportExternalMemory(CUexternalMemory* extMem_out, const CUDA_EXTERNAL_MEMORY_HANDLE_DESC* memHandleDesc) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuImportExternalMemory
    _check_or_init_driver()
    if __cuImportExternalMemory == NULL:
        with gil:
            raise FunctionNotFoundError("function cuImportExternalMemory is not found")
    return (<CUresult (*)(CUexternalMemory*, const CUDA_EXTERNAL_MEMORY_HANDLE_DESC*) noexcept nogil>__cuImportExternalMemory)(
        extMem_out, memHandleDesc)


cdef CUresult _cuExternalMemoryGetMappedBuffer(CUdeviceptr* devPtr, CUexternalMemory extMem, const CUDA_EXTERNAL_MEMORY_BUFFER_DESC* bufferDesc) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuExternalMemoryGetMappedBuffer
    _check_or_init_driver()
    if __cuExternalMemoryGetMappedBuffer == NULL:
        with gil:
            raise FunctionNotFoundError("function cuExternalMemoryGetMappedBuffer is not found")
    return (<CUresult (*)(CUdeviceptr*, CUexternalMemory, const CUDA_EXTERNAL_MEMORY_BUFFER_DESC*) noexcept nogil>__cuExternalMemoryGetMappedBuffer)(
        devPtr, extMem, bufferDesc)


cdef CUresult _cuExternalMemoryGetMappedMipmappedArray(CUmipmappedArray* mipmap, CUexternalMemory extMem, const CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC* mipmapDesc) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuExternalMemoryGetMappedMipmappedArray
    _check_or_init_driver()
    if __cuExternalMemoryGetMappedMipmappedArray == NULL:
        with gil:
            raise FunctionNotFoundError("function cuExternalMemoryGetMappedMipmappedArray is not found")
    return (<CUresult (*)(CUmipmappedArray*, CUexternalMemory, const CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC*) noexcept nogil>__cuExternalMemoryGetMappedMipmappedArray)(
        mipmap, extMem, mipmapDesc)


cdef CUresult _cuDestroyExternalMemory(CUexternalMemory extMem) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuDestroyExternalMemory
    _check_or_init_driver()
    if __cuDestroyExternalMemory == NULL:
        with gil:
            raise FunctionNotFoundError("function cuDestroyExternalMemory is not found")
    return (<CUresult (*)(CUexternalMemory) noexcept nogil>__cuDestroyExternalMemory)(
        extMem)


cdef CUresult _cuImportExternalSemaphore(CUexternalSemaphore* extSem_out, const CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC* semHandleDesc) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuImportExternalSemaphore
    _check_or_init_driver()
    if __cuImportExternalSemaphore == NULL:
        with gil:
            raise FunctionNotFoundError("function cuImportExternalSemaphore is not found")
    return (<CUresult (*)(CUexternalSemaphore*, const CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC*) noexcept nogil>__cuImportExternalSemaphore)(
        extSem_out, semHandleDesc)


cdef CUresult _cuSignalExternalSemaphoresAsync(const CUexternalSemaphore* extSemArray, const CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS* paramsArray, unsigned int numExtSems, CUstream stream) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuSignalExternalSemaphoresAsync
    _check_or_init_driver()
    if __cuSignalExternalSemaphoresAsync == NULL:
        with gil:
            raise FunctionNotFoundError("function cuSignalExternalSemaphoresAsync is not found")
    return (<CUresult (*)(const CUexternalSemaphore*, const CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS*, unsigned int, CUstream) noexcept nogil>__cuSignalExternalSemaphoresAsync)(
        extSemArray, paramsArray, numExtSems, stream)


cdef CUresult _cuWaitExternalSemaphoresAsync(const CUexternalSemaphore* extSemArray, const CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS* paramsArray, unsigned int numExtSems, CUstream stream) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuWaitExternalSemaphoresAsync
    _check_or_init_driver()
    if __cuWaitExternalSemaphoresAsync == NULL:
        with gil:
            raise FunctionNotFoundError("function cuWaitExternalSemaphoresAsync is not found")
    return (<CUresult (*)(const CUexternalSemaphore*, const CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS*, unsigned int, CUstream) noexcept nogil>__cuWaitExternalSemaphoresAsync)(
        extSemArray, paramsArray, numExtSems, stream)


cdef CUresult _cuDestroyExternalSemaphore(CUexternalSemaphore extSem) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuDestroyExternalSemaphore
    _check_or_init_driver()
    if __cuDestroyExternalSemaphore == NULL:
        with gil:
            raise FunctionNotFoundError("function cuDestroyExternalSemaphore is not found")
    return (<CUresult (*)(CUexternalSemaphore) noexcept nogil>__cuDestroyExternalSemaphore)(
        extSem)


cdef CUresult _cuStreamWaitValue32_v2(CUstream stream, CUdeviceptr addr, cuuint32_t value, unsigned int flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuStreamWaitValue32_v2
    _check_or_init_driver()
    if __cuStreamWaitValue32_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuStreamWaitValue32_v2 is not found")
    return (<CUresult (*)(CUstream, CUdeviceptr, cuuint32_t, unsigned int) noexcept nogil>__cuStreamWaitValue32_v2)(
        stream, addr, value, flags)


cdef CUresult _cuStreamWaitValue64_v2(CUstream stream, CUdeviceptr addr, cuuint64_t value, unsigned int flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuStreamWaitValue64_v2
    _check_or_init_driver()
    if __cuStreamWaitValue64_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuStreamWaitValue64_v2 is not found")
    return (<CUresult (*)(CUstream, CUdeviceptr, cuuint64_t, unsigned int) noexcept nogil>__cuStreamWaitValue64_v2)(
        stream, addr, value, flags)


cdef CUresult _cuStreamWriteValue32_v2(CUstream stream, CUdeviceptr addr, cuuint32_t value, unsigned int flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuStreamWriteValue32_v2
    _check_or_init_driver()
    if __cuStreamWriteValue32_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuStreamWriteValue32_v2 is not found")
    return (<CUresult (*)(CUstream, CUdeviceptr, cuuint32_t, unsigned int) noexcept nogil>__cuStreamWriteValue32_v2)(
        stream, addr, value, flags)


cdef CUresult _cuStreamWriteValue64_v2(CUstream stream, CUdeviceptr addr, cuuint64_t value, unsigned int flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuStreamWriteValue64_v2
    _check_or_init_driver()
    if __cuStreamWriteValue64_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuStreamWriteValue64_v2 is not found")
    return (<CUresult (*)(CUstream, CUdeviceptr, cuuint64_t, unsigned int) noexcept nogil>__cuStreamWriteValue64_v2)(
        stream, addr, value, flags)


cdef CUresult _cuStreamBatchMemOp_v2(CUstream stream, unsigned int count, CUstreamBatchMemOpParams* paramArray, unsigned int flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuStreamBatchMemOp_v2
    _check_or_init_driver()
    if __cuStreamBatchMemOp_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuStreamBatchMemOp_v2 is not found")
    return (<CUresult (*)(CUstream, unsigned int, CUstreamBatchMemOpParams*, unsigned int) noexcept nogil>__cuStreamBatchMemOp_v2)(
        stream, count, paramArray, flags)


cdef CUresult _cuFuncGetAttribute(int* pi, CUfunction_attribute attrib, CUfunction hfunc) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuFuncGetAttribute
    _check_or_init_driver()
    if __cuFuncGetAttribute == NULL:
        with gil:
            raise FunctionNotFoundError("function cuFuncGetAttribute is not found")
    return (<CUresult (*)(int*, CUfunction_attribute, CUfunction) noexcept nogil>__cuFuncGetAttribute)(
        pi, attrib, hfunc)


cdef CUresult _cuFuncSetAttribute(CUfunction hfunc, CUfunction_attribute attrib, int value) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuFuncSetAttribute
    _check_or_init_driver()
    if __cuFuncSetAttribute == NULL:
        with gil:
            raise FunctionNotFoundError("function cuFuncSetAttribute is not found")
    return (<CUresult (*)(CUfunction, CUfunction_attribute, int) noexcept nogil>__cuFuncSetAttribute)(
        hfunc, attrib, value)


cdef CUresult _cuFuncSetCacheConfig(CUfunction hfunc, CUfunc_cache config) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuFuncSetCacheConfig
    _check_or_init_driver()
    if __cuFuncSetCacheConfig == NULL:
        with gil:
            raise FunctionNotFoundError("function cuFuncSetCacheConfig is not found")
    return (<CUresult (*)(CUfunction, CUfunc_cache) noexcept nogil>__cuFuncSetCacheConfig)(
        hfunc, config)


cdef CUresult _cuFuncGetModule(CUmodule* hmod, CUfunction hfunc) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuFuncGetModule
    _check_or_init_driver()
    if __cuFuncGetModule == NULL:
        with gil:
            raise FunctionNotFoundError("function cuFuncGetModule is not found")
    return (<CUresult (*)(CUmodule*, CUfunction) noexcept nogil>__cuFuncGetModule)(
        hmod, hfunc)


cdef CUresult _cuFuncGetName(const char** name, CUfunction hfunc) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuFuncGetName
    _check_or_init_driver()
    if __cuFuncGetName == NULL:
        with gil:
            raise FunctionNotFoundError("function cuFuncGetName is not found")
    return (<CUresult (*)(const char**, CUfunction) noexcept nogil>__cuFuncGetName)(
        name, hfunc)


cdef CUresult _cuFuncGetParamInfo(CUfunction func, size_t paramIndex, size_t* paramOffset, size_t* paramSize) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuFuncGetParamInfo
    _check_or_init_driver()
    if __cuFuncGetParamInfo == NULL:
        with gil:
            raise FunctionNotFoundError("function cuFuncGetParamInfo is not found")
    return (<CUresult (*)(CUfunction, size_t, size_t*, size_t*) noexcept nogil>__cuFuncGetParamInfo)(
        func, paramIndex, paramOffset, paramSize)


cdef CUresult _cuFuncIsLoaded(CUfunctionLoadingState* state, CUfunction function) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuFuncIsLoaded
    _check_or_init_driver()
    if __cuFuncIsLoaded == NULL:
        with gil:
            raise FunctionNotFoundError("function cuFuncIsLoaded is not found")
    return (<CUresult (*)(CUfunctionLoadingState*, CUfunction) noexcept nogil>__cuFuncIsLoaded)(
        state, function)


cdef CUresult _cuFuncLoad(CUfunction function) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuFuncLoad
    _check_or_init_driver()
    if __cuFuncLoad == NULL:
        with gil:
            raise FunctionNotFoundError("function cuFuncLoad is not found")
    return (<CUresult (*)(CUfunction) noexcept nogil>__cuFuncLoad)(
        function)


cdef CUresult _cuLaunchKernel(CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, unsigned int sharedMemBytes, CUstream hStream, void** kernelParams, void** extra) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuLaunchKernel
    _check_or_init_driver()
    if __cuLaunchKernel == NULL:
        with gil:
            raise FunctionNotFoundError("function cuLaunchKernel is not found")
    return (<CUresult (*)(CUfunction, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, CUstream, void**, void**) noexcept nogil>__cuLaunchKernel)(
        f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, kernelParams, extra)


cdef CUresult _cuLaunchKernelEx(const CUlaunchConfig* config, CUfunction f, void** kernelParams, void** extra) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuLaunchKernelEx
    _check_or_init_driver()
    if __cuLaunchKernelEx == NULL:
        with gil:
            raise FunctionNotFoundError("function cuLaunchKernelEx is not found")
    return (<CUresult (*)(const CUlaunchConfig*, CUfunction, void**, void**) noexcept nogil>__cuLaunchKernelEx)(
        config, f, kernelParams, extra)


cdef CUresult _cuLaunchCooperativeKernel(CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, unsigned int sharedMemBytes, CUstream hStream, void** kernelParams) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuLaunchCooperativeKernel
    _check_or_init_driver()
    if __cuLaunchCooperativeKernel == NULL:
        with gil:
            raise FunctionNotFoundError("function cuLaunchCooperativeKernel is not found")
    return (<CUresult (*)(CUfunction, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, CUstream, void**) noexcept nogil>__cuLaunchCooperativeKernel)(
        f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, kernelParams)


cdef CUresult _cuLaunchCooperativeKernelMultiDevice(CUDA_LAUNCH_PARAMS* launchParamsList, unsigned int numDevices, unsigned int flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuLaunchCooperativeKernelMultiDevice
    _check_or_init_driver()
    if __cuLaunchCooperativeKernelMultiDevice == NULL:
        with gil:
            raise FunctionNotFoundError("function cuLaunchCooperativeKernelMultiDevice is not found")
    return (<CUresult (*)(CUDA_LAUNCH_PARAMS*, unsigned int, unsigned int) noexcept nogil>__cuLaunchCooperativeKernelMultiDevice)(
        launchParamsList, numDevices, flags)


cdef CUresult _cuLaunchHostFunc(CUstream hStream, CUhostFn fn, void* userData) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuLaunchHostFunc
    _check_or_init_driver()
    if __cuLaunchHostFunc == NULL:
        with gil:
            raise FunctionNotFoundError("function cuLaunchHostFunc is not found")
    return (<CUresult (*)(CUstream, CUhostFn, void*) noexcept nogil>__cuLaunchHostFunc)(
        hStream, fn, userData)


cdef CUresult _cuFuncSetBlockShape(CUfunction hfunc, int x, int y, int z) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuFuncSetBlockShape
    _check_or_init_driver()
    if __cuFuncSetBlockShape == NULL:
        with gil:
            raise FunctionNotFoundError("function cuFuncSetBlockShape is not found")
    return (<CUresult (*)(CUfunction, int, int, int) noexcept nogil>__cuFuncSetBlockShape)(
        hfunc, x, y, z)


cdef CUresult _cuFuncSetSharedSize(CUfunction hfunc, unsigned int bytes) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuFuncSetSharedSize
    _check_or_init_driver()
    if __cuFuncSetSharedSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cuFuncSetSharedSize is not found")
    return (<CUresult (*)(CUfunction, unsigned int) noexcept nogil>__cuFuncSetSharedSize)(
        hfunc, bytes)


cdef CUresult _cuParamSetSize(CUfunction hfunc, unsigned int numbytes) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuParamSetSize
    _check_or_init_driver()
    if __cuParamSetSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cuParamSetSize is not found")
    return (<CUresult (*)(CUfunction, unsigned int) noexcept nogil>__cuParamSetSize)(
        hfunc, numbytes)


cdef CUresult _cuParamSeti(CUfunction hfunc, int offset, unsigned int value) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuParamSeti
    _check_or_init_driver()
    if __cuParamSeti == NULL:
        with gil:
            raise FunctionNotFoundError("function cuParamSeti is not found")
    return (<CUresult (*)(CUfunction, int, unsigned int) noexcept nogil>__cuParamSeti)(
        hfunc, offset, value)


cdef CUresult _cuParamSetf(CUfunction hfunc, int offset, float value) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuParamSetf
    _check_or_init_driver()
    if __cuParamSetf == NULL:
        with gil:
            raise FunctionNotFoundError("function cuParamSetf is not found")
    return (<CUresult (*)(CUfunction, int, float) noexcept nogil>__cuParamSetf)(
        hfunc, offset, value)


cdef CUresult _cuParamSetv(CUfunction hfunc, int offset, void* ptr, unsigned int numbytes) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuParamSetv
    _check_or_init_driver()
    if __cuParamSetv == NULL:
        with gil:
            raise FunctionNotFoundError("function cuParamSetv is not found")
    return (<CUresult (*)(CUfunction, int, void*, unsigned int) noexcept nogil>__cuParamSetv)(
        hfunc, offset, ptr, numbytes)


cdef CUresult _cuLaunch(CUfunction f) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuLaunch
    _check_or_init_driver()
    if __cuLaunch == NULL:
        with gil:
            raise FunctionNotFoundError("function cuLaunch is not found")
    return (<CUresult (*)(CUfunction) noexcept nogil>__cuLaunch)(
        f)


cdef CUresult _cuLaunchGrid(CUfunction f, int grid_width, int grid_height) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuLaunchGrid
    _check_or_init_driver()
    if __cuLaunchGrid == NULL:
        with gil:
            raise FunctionNotFoundError("function cuLaunchGrid is not found")
    return (<CUresult (*)(CUfunction, int, int) noexcept nogil>__cuLaunchGrid)(
        f, grid_width, grid_height)


cdef CUresult _cuLaunchGridAsync(CUfunction f, int grid_width, int grid_height, CUstream hStream) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuLaunchGridAsync
    _check_or_init_driver()
    if __cuLaunchGridAsync == NULL:
        with gil:
            raise FunctionNotFoundError("function cuLaunchGridAsync is not found")
    return (<CUresult (*)(CUfunction, int, int, CUstream) noexcept nogil>__cuLaunchGridAsync)(
        f, grid_width, grid_height, hStream)


cdef CUresult _cuParamSetTexRef(CUfunction hfunc, int texunit, CUtexref hTexRef) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuParamSetTexRef
    _check_or_init_driver()
    if __cuParamSetTexRef == NULL:
        with gil:
            raise FunctionNotFoundError("function cuParamSetTexRef is not found")
    return (<CUresult (*)(CUfunction, int, CUtexref) noexcept nogil>__cuParamSetTexRef)(
        hfunc, texunit, hTexRef)


cdef CUresult _cuFuncSetSharedMemConfig(CUfunction hfunc, CUsharedconfig config) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuFuncSetSharedMemConfig
    _check_or_init_driver()
    if __cuFuncSetSharedMemConfig == NULL:
        with gil:
            raise FunctionNotFoundError("function cuFuncSetSharedMemConfig is not found")
    return (<CUresult (*)(CUfunction, CUsharedconfig) noexcept nogil>__cuFuncSetSharedMemConfig)(
        hfunc, config)


cdef CUresult _cuGraphCreate(CUgraph* phGraph, unsigned int flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuGraphCreate
    _check_or_init_driver()
    if __cuGraphCreate == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphCreate is not found")
    return (<CUresult (*)(CUgraph*, unsigned int) noexcept nogil>__cuGraphCreate)(
        phGraph, flags)


cdef CUresult _cuGraphAddKernelNode_v2(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, size_t numDependencies, const CUDA_KERNEL_NODE_PARAMS* nodeParams) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuGraphAddKernelNode_v2
    _check_or_init_driver()
    if __cuGraphAddKernelNode_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphAddKernelNode_v2 is not found")
    return (<CUresult (*)(CUgraphNode*, CUgraph, const CUgraphNode*, size_t, const CUDA_KERNEL_NODE_PARAMS*) noexcept nogil>__cuGraphAddKernelNode_v2)(
        phGraphNode, hGraph, dependencies, numDependencies, nodeParams)


cdef CUresult _cuGraphKernelNodeGetParams_v2(CUgraphNode hNode, CUDA_KERNEL_NODE_PARAMS* nodeParams) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuGraphKernelNodeGetParams_v2
    _check_or_init_driver()
    if __cuGraphKernelNodeGetParams_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphKernelNodeGetParams_v2 is not found")
    return (<CUresult (*)(CUgraphNode, CUDA_KERNEL_NODE_PARAMS*) noexcept nogil>__cuGraphKernelNodeGetParams_v2)(
        hNode, nodeParams)


cdef CUresult _cuGraphKernelNodeSetParams_v2(CUgraphNode hNode, const CUDA_KERNEL_NODE_PARAMS* nodeParams) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuGraphKernelNodeSetParams_v2
    _check_or_init_driver()
    if __cuGraphKernelNodeSetParams_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphKernelNodeSetParams_v2 is not found")
    return (<CUresult (*)(CUgraphNode, const CUDA_KERNEL_NODE_PARAMS*) noexcept nogil>__cuGraphKernelNodeSetParams_v2)(
        hNode, nodeParams)


cdef CUresult _cuGraphAddMemcpyNode(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, size_t numDependencies, const CUDA_MEMCPY3D* copyParams, CUcontext ctx) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuGraphAddMemcpyNode
    _check_or_init_driver()
    if __cuGraphAddMemcpyNode == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphAddMemcpyNode is not found")
    return (<CUresult (*)(CUgraphNode*, CUgraph, const CUgraphNode*, size_t, const CUDA_MEMCPY3D*, CUcontext) noexcept nogil>__cuGraphAddMemcpyNode)(
        phGraphNode, hGraph, dependencies, numDependencies, copyParams, ctx)


cdef CUresult _cuGraphMemcpyNodeGetParams(CUgraphNode hNode, CUDA_MEMCPY3D* nodeParams) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuGraphMemcpyNodeGetParams
    _check_or_init_driver()
    if __cuGraphMemcpyNodeGetParams == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphMemcpyNodeGetParams is not found")
    return (<CUresult (*)(CUgraphNode, CUDA_MEMCPY3D*) noexcept nogil>__cuGraphMemcpyNodeGetParams)(
        hNode, nodeParams)


cdef CUresult _cuGraphMemcpyNodeSetParams(CUgraphNode hNode, const CUDA_MEMCPY3D* nodeParams) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuGraphMemcpyNodeSetParams
    _check_or_init_driver()
    if __cuGraphMemcpyNodeSetParams == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphMemcpyNodeSetParams is not found")
    return (<CUresult (*)(CUgraphNode, const CUDA_MEMCPY3D*) noexcept nogil>__cuGraphMemcpyNodeSetParams)(
        hNode, nodeParams)


cdef CUresult _cuGraphAddMemsetNode(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, size_t numDependencies, const CUDA_MEMSET_NODE_PARAMS* memsetParams, CUcontext ctx) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuGraphAddMemsetNode
    _check_or_init_driver()
    if __cuGraphAddMemsetNode == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphAddMemsetNode is not found")
    return (<CUresult (*)(CUgraphNode*, CUgraph, const CUgraphNode*, size_t, const CUDA_MEMSET_NODE_PARAMS*, CUcontext) noexcept nogil>__cuGraphAddMemsetNode)(
        phGraphNode, hGraph, dependencies, numDependencies, memsetParams, ctx)


cdef CUresult _cuGraphMemsetNodeGetParams(CUgraphNode hNode, CUDA_MEMSET_NODE_PARAMS* nodeParams) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuGraphMemsetNodeGetParams
    _check_or_init_driver()
    if __cuGraphMemsetNodeGetParams == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphMemsetNodeGetParams is not found")
    return (<CUresult (*)(CUgraphNode, CUDA_MEMSET_NODE_PARAMS*) noexcept nogil>__cuGraphMemsetNodeGetParams)(
        hNode, nodeParams)


cdef CUresult _cuGraphMemsetNodeSetParams(CUgraphNode hNode, const CUDA_MEMSET_NODE_PARAMS* nodeParams) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuGraphMemsetNodeSetParams
    _check_or_init_driver()
    if __cuGraphMemsetNodeSetParams == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphMemsetNodeSetParams is not found")
    return (<CUresult (*)(CUgraphNode, const CUDA_MEMSET_NODE_PARAMS*) noexcept nogil>__cuGraphMemsetNodeSetParams)(
        hNode, nodeParams)


cdef CUresult _cuGraphAddHostNode(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, size_t numDependencies, const CUDA_HOST_NODE_PARAMS* nodeParams) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuGraphAddHostNode
    _check_or_init_driver()
    if __cuGraphAddHostNode == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphAddHostNode is not found")
    return (<CUresult (*)(CUgraphNode*, CUgraph, const CUgraphNode*, size_t, const CUDA_HOST_NODE_PARAMS*) noexcept nogil>__cuGraphAddHostNode)(
        phGraphNode, hGraph, dependencies, numDependencies, nodeParams)


cdef CUresult _cuGraphHostNodeGetParams(CUgraphNode hNode, CUDA_HOST_NODE_PARAMS* nodeParams) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuGraphHostNodeGetParams
    _check_or_init_driver()
    if __cuGraphHostNodeGetParams == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphHostNodeGetParams is not found")
    return (<CUresult (*)(CUgraphNode, CUDA_HOST_NODE_PARAMS*) noexcept nogil>__cuGraphHostNodeGetParams)(
        hNode, nodeParams)


cdef CUresult _cuGraphHostNodeSetParams(CUgraphNode hNode, const CUDA_HOST_NODE_PARAMS* nodeParams) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuGraphHostNodeSetParams
    _check_or_init_driver()
    if __cuGraphHostNodeSetParams == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphHostNodeSetParams is not found")
    return (<CUresult (*)(CUgraphNode, const CUDA_HOST_NODE_PARAMS*) noexcept nogil>__cuGraphHostNodeSetParams)(
        hNode, nodeParams)


cdef CUresult _cuGraphAddChildGraphNode(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, size_t numDependencies, CUgraph childGraph) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuGraphAddChildGraphNode
    _check_or_init_driver()
    if __cuGraphAddChildGraphNode == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphAddChildGraphNode is not found")
    return (<CUresult (*)(CUgraphNode*, CUgraph, const CUgraphNode*, size_t, CUgraph) noexcept nogil>__cuGraphAddChildGraphNode)(
        phGraphNode, hGraph, dependencies, numDependencies, childGraph)


cdef CUresult _cuGraphChildGraphNodeGetGraph(CUgraphNode hNode, CUgraph* phGraph) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuGraphChildGraphNodeGetGraph
    _check_or_init_driver()
    if __cuGraphChildGraphNodeGetGraph == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphChildGraphNodeGetGraph is not found")
    return (<CUresult (*)(CUgraphNode, CUgraph*) noexcept nogil>__cuGraphChildGraphNodeGetGraph)(
        hNode, phGraph)


cdef CUresult _cuGraphAddEmptyNode(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, size_t numDependencies) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuGraphAddEmptyNode
    _check_or_init_driver()
    if __cuGraphAddEmptyNode == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphAddEmptyNode is not found")
    return (<CUresult (*)(CUgraphNode*, CUgraph, const CUgraphNode*, size_t) noexcept nogil>__cuGraphAddEmptyNode)(
        phGraphNode, hGraph, dependencies, numDependencies)


cdef CUresult _cuGraphAddEventRecordNode(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, size_t numDependencies, CUevent event) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuGraphAddEventRecordNode
    _check_or_init_driver()
    if __cuGraphAddEventRecordNode == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphAddEventRecordNode is not found")
    return (<CUresult (*)(CUgraphNode*, CUgraph, const CUgraphNode*, size_t, CUevent) noexcept nogil>__cuGraphAddEventRecordNode)(
        phGraphNode, hGraph, dependencies, numDependencies, event)


cdef CUresult _cuGraphEventRecordNodeGetEvent(CUgraphNode hNode, CUevent* event_out) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuGraphEventRecordNodeGetEvent
    _check_or_init_driver()
    if __cuGraphEventRecordNodeGetEvent == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphEventRecordNodeGetEvent is not found")
    return (<CUresult (*)(CUgraphNode, CUevent*) noexcept nogil>__cuGraphEventRecordNodeGetEvent)(
        hNode, event_out)


cdef CUresult _cuGraphEventRecordNodeSetEvent(CUgraphNode hNode, CUevent event) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuGraphEventRecordNodeSetEvent
    _check_or_init_driver()
    if __cuGraphEventRecordNodeSetEvent == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphEventRecordNodeSetEvent is not found")
    return (<CUresult (*)(CUgraphNode, CUevent) noexcept nogil>__cuGraphEventRecordNodeSetEvent)(
        hNode, event)


cdef CUresult _cuGraphAddEventWaitNode(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, size_t numDependencies, CUevent event) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuGraphAddEventWaitNode
    _check_or_init_driver()
    if __cuGraphAddEventWaitNode == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphAddEventWaitNode is not found")
    return (<CUresult (*)(CUgraphNode*, CUgraph, const CUgraphNode*, size_t, CUevent) noexcept nogil>__cuGraphAddEventWaitNode)(
        phGraphNode, hGraph, dependencies, numDependencies, event)


cdef CUresult _cuGraphEventWaitNodeGetEvent(CUgraphNode hNode, CUevent* event_out) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuGraphEventWaitNodeGetEvent
    _check_or_init_driver()
    if __cuGraphEventWaitNodeGetEvent == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphEventWaitNodeGetEvent is not found")
    return (<CUresult (*)(CUgraphNode, CUevent*) noexcept nogil>__cuGraphEventWaitNodeGetEvent)(
        hNode, event_out)


cdef CUresult _cuGraphEventWaitNodeSetEvent(CUgraphNode hNode, CUevent event) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuGraphEventWaitNodeSetEvent
    _check_or_init_driver()
    if __cuGraphEventWaitNodeSetEvent == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphEventWaitNodeSetEvent is not found")
    return (<CUresult (*)(CUgraphNode, CUevent) noexcept nogil>__cuGraphEventWaitNodeSetEvent)(
        hNode, event)


cdef CUresult _cuGraphAddExternalSemaphoresSignalNode(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, size_t numDependencies, const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS* nodeParams) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuGraphAddExternalSemaphoresSignalNode
    _check_or_init_driver()
    if __cuGraphAddExternalSemaphoresSignalNode == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphAddExternalSemaphoresSignalNode is not found")
    return (<CUresult (*)(CUgraphNode*, CUgraph, const CUgraphNode*, size_t, const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS*) noexcept nogil>__cuGraphAddExternalSemaphoresSignalNode)(
        phGraphNode, hGraph, dependencies, numDependencies, nodeParams)


cdef CUresult _cuGraphExternalSemaphoresSignalNodeGetParams(CUgraphNode hNode, CUDA_EXT_SEM_SIGNAL_NODE_PARAMS* params_out) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuGraphExternalSemaphoresSignalNodeGetParams
    _check_or_init_driver()
    if __cuGraphExternalSemaphoresSignalNodeGetParams == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphExternalSemaphoresSignalNodeGetParams is not found")
    return (<CUresult (*)(CUgraphNode, CUDA_EXT_SEM_SIGNAL_NODE_PARAMS*) noexcept nogil>__cuGraphExternalSemaphoresSignalNodeGetParams)(
        hNode, params_out)


cdef CUresult _cuGraphExternalSemaphoresSignalNodeSetParams(CUgraphNode hNode, const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS* nodeParams) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuGraphExternalSemaphoresSignalNodeSetParams
    _check_or_init_driver()
    if __cuGraphExternalSemaphoresSignalNodeSetParams == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphExternalSemaphoresSignalNodeSetParams is not found")
    return (<CUresult (*)(CUgraphNode, const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS*) noexcept nogil>__cuGraphExternalSemaphoresSignalNodeSetParams)(
        hNode, nodeParams)


cdef CUresult _cuGraphAddExternalSemaphoresWaitNode(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, size_t numDependencies, const CUDA_EXT_SEM_WAIT_NODE_PARAMS* nodeParams) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuGraphAddExternalSemaphoresWaitNode
    _check_or_init_driver()
    if __cuGraphAddExternalSemaphoresWaitNode == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphAddExternalSemaphoresWaitNode is not found")
    return (<CUresult (*)(CUgraphNode*, CUgraph, const CUgraphNode*, size_t, const CUDA_EXT_SEM_WAIT_NODE_PARAMS*) noexcept nogil>__cuGraphAddExternalSemaphoresWaitNode)(
        phGraphNode, hGraph, dependencies, numDependencies, nodeParams)


cdef CUresult _cuGraphExternalSemaphoresWaitNodeGetParams(CUgraphNode hNode, CUDA_EXT_SEM_WAIT_NODE_PARAMS* params_out) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuGraphExternalSemaphoresWaitNodeGetParams
    _check_or_init_driver()
    if __cuGraphExternalSemaphoresWaitNodeGetParams == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphExternalSemaphoresWaitNodeGetParams is not found")
    return (<CUresult (*)(CUgraphNode, CUDA_EXT_SEM_WAIT_NODE_PARAMS*) noexcept nogil>__cuGraphExternalSemaphoresWaitNodeGetParams)(
        hNode, params_out)


cdef CUresult _cuGraphExternalSemaphoresWaitNodeSetParams(CUgraphNode hNode, const CUDA_EXT_SEM_WAIT_NODE_PARAMS* nodeParams) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuGraphExternalSemaphoresWaitNodeSetParams
    _check_or_init_driver()
    if __cuGraphExternalSemaphoresWaitNodeSetParams == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphExternalSemaphoresWaitNodeSetParams is not found")
    return (<CUresult (*)(CUgraphNode, const CUDA_EXT_SEM_WAIT_NODE_PARAMS*) noexcept nogil>__cuGraphExternalSemaphoresWaitNodeSetParams)(
        hNode, nodeParams)


cdef CUresult _cuGraphAddBatchMemOpNode(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, size_t numDependencies, const CUDA_BATCH_MEM_OP_NODE_PARAMS* nodeParams) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuGraphAddBatchMemOpNode
    _check_or_init_driver()
    if __cuGraphAddBatchMemOpNode == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphAddBatchMemOpNode is not found")
    return (<CUresult (*)(CUgraphNode*, CUgraph, const CUgraphNode*, size_t, const CUDA_BATCH_MEM_OP_NODE_PARAMS*) noexcept nogil>__cuGraphAddBatchMemOpNode)(
        phGraphNode, hGraph, dependencies, numDependencies, nodeParams)


cdef CUresult _cuGraphBatchMemOpNodeGetParams(CUgraphNode hNode, CUDA_BATCH_MEM_OP_NODE_PARAMS* nodeParams_out) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuGraphBatchMemOpNodeGetParams
    _check_or_init_driver()
    if __cuGraphBatchMemOpNodeGetParams == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphBatchMemOpNodeGetParams is not found")
    return (<CUresult (*)(CUgraphNode, CUDA_BATCH_MEM_OP_NODE_PARAMS*) noexcept nogil>__cuGraphBatchMemOpNodeGetParams)(
        hNode, nodeParams_out)


cdef CUresult _cuGraphBatchMemOpNodeSetParams(CUgraphNode hNode, const CUDA_BATCH_MEM_OP_NODE_PARAMS* nodeParams) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuGraphBatchMemOpNodeSetParams
    _check_or_init_driver()
    if __cuGraphBatchMemOpNodeSetParams == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphBatchMemOpNodeSetParams is not found")
    return (<CUresult (*)(CUgraphNode, const CUDA_BATCH_MEM_OP_NODE_PARAMS*) noexcept nogil>__cuGraphBatchMemOpNodeSetParams)(
        hNode, nodeParams)


cdef CUresult _cuGraphExecBatchMemOpNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_BATCH_MEM_OP_NODE_PARAMS* nodeParams) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuGraphExecBatchMemOpNodeSetParams
    _check_or_init_driver()
    if __cuGraphExecBatchMemOpNodeSetParams == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphExecBatchMemOpNodeSetParams is not found")
    return (<CUresult (*)(CUgraphExec, CUgraphNode, const CUDA_BATCH_MEM_OP_NODE_PARAMS*) noexcept nogil>__cuGraphExecBatchMemOpNodeSetParams)(
        hGraphExec, hNode, nodeParams)


cdef CUresult _cuGraphAddMemAllocNode(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, size_t numDependencies, CUDA_MEM_ALLOC_NODE_PARAMS* nodeParams) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuGraphAddMemAllocNode
    _check_or_init_driver()
    if __cuGraphAddMemAllocNode == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphAddMemAllocNode is not found")
    return (<CUresult (*)(CUgraphNode*, CUgraph, const CUgraphNode*, size_t, CUDA_MEM_ALLOC_NODE_PARAMS*) noexcept nogil>__cuGraphAddMemAllocNode)(
        phGraphNode, hGraph, dependencies, numDependencies, nodeParams)


cdef CUresult _cuGraphMemAllocNodeGetParams(CUgraphNode hNode, CUDA_MEM_ALLOC_NODE_PARAMS* params_out) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuGraphMemAllocNodeGetParams
    _check_or_init_driver()
    if __cuGraphMemAllocNodeGetParams == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphMemAllocNodeGetParams is not found")
    return (<CUresult (*)(CUgraphNode, CUDA_MEM_ALLOC_NODE_PARAMS*) noexcept nogil>__cuGraphMemAllocNodeGetParams)(
        hNode, params_out)


cdef CUresult _cuGraphAddMemFreeNode(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, size_t numDependencies, CUdeviceptr dptr) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuGraphAddMemFreeNode
    _check_or_init_driver()
    if __cuGraphAddMemFreeNode == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphAddMemFreeNode is not found")
    return (<CUresult (*)(CUgraphNode*, CUgraph, const CUgraphNode*, size_t, CUdeviceptr) noexcept nogil>__cuGraphAddMemFreeNode)(
        phGraphNode, hGraph, dependencies, numDependencies, dptr)


cdef CUresult _cuGraphMemFreeNodeGetParams(CUgraphNode hNode, CUdeviceptr* dptr_out) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuGraphMemFreeNodeGetParams
    _check_or_init_driver()
    if __cuGraphMemFreeNodeGetParams == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphMemFreeNodeGetParams is not found")
    return (<CUresult (*)(CUgraphNode, CUdeviceptr*) noexcept nogil>__cuGraphMemFreeNodeGetParams)(
        hNode, dptr_out)


cdef CUresult _cuDeviceGraphMemTrim(CUdevice device) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuDeviceGraphMemTrim
    _check_or_init_driver()
    if __cuDeviceGraphMemTrim == NULL:
        with gil:
            raise FunctionNotFoundError("function cuDeviceGraphMemTrim is not found")
    return (<CUresult (*)(CUdevice) noexcept nogil>__cuDeviceGraphMemTrim)(
        device)


cdef CUresult _cuDeviceGetGraphMemAttribute(CUdevice device, CUgraphMem_attribute attr, void* value) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuDeviceGetGraphMemAttribute
    _check_or_init_driver()
    if __cuDeviceGetGraphMemAttribute == NULL:
        with gil:
            raise FunctionNotFoundError("function cuDeviceGetGraphMemAttribute is not found")
    return (<CUresult (*)(CUdevice, CUgraphMem_attribute, void*) noexcept nogil>__cuDeviceGetGraphMemAttribute)(
        device, attr, value)


cdef CUresult _cuDeviceSetGraphMemAttribute(CUdevice device, CUgraphMem_attribute attr, void* value) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuDeviceSetGraphMemAttribute
    _check_or_init_driver()
    if __cuDeviceSetGraphMemAttribute == NULL:
        with gil:
            raise FunctionNotFoundError("function cuDeviceSetGraphMemAttribute is not found")
    return (<CUresult (*)(CUdevice, CUgraphMem_attribute, void*) noexcept nogil>__cuDeviceSetGraphMemAttribute)(
        device, attr, value)


cdef CUresult _cuGraphClone(CUgraph* phGraphClone, CUgraph originalGraph) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuGraphClone
    _check_or_init_driver()
    if __cuGraphClone == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphClone is not found")
    return (<CUresult (*)(CUgraph*, CUgraph) noexcept nogil>__cuGraphClone)(
        phGraphClone, originalGraph)


cdef CUresult _cuGraphNodeFindInClone(CUgraphNode* phNode, CUgraphNode hOriginalNode, CUgraph hClonedGraph) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuGraphNodeFindInClone
    _check_or_init_driver()
    if __cuGraphNodeFindInClone == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphNodeFindInClone is not found")
    return (<CUresult (*)(CUgraphNode*, CUgraphNode, CUgraph) noexcept nogil>__cuGraphNodeFindInClone)(
        phNode, hOriginalNode, hClonedGraph)


cdef CUresult _cuGraphNodeGetType(CUgraphNode hNode, CUgraphNodeType* type) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuGraphNodeGetType
    _check_or_init_driver()
    if __cuGraphNodeGetType == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphNodeGetType is not found")
    return (<CUresult (*)(CUgraphNode, CUgraphNodeType*) noexcept nogil>__cuGraphNodeGetType)(
        hNode, type)


cdef CUresult _cuGraphGetNodes(CUgraph hGraph, CUgraphNode* nodes, size_t* numNodes) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuGraphGetNodes
    _check_or_init_driver()
    if __cuGraphGetNodes == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphGetNodes is not found")
    return (<CUresult (*)(CUgraph, CUgraphNode*, size_t*) noexcept nogil>__cuGraphGetNodes)(
        hGraph, nodes, numNodes)


cdef CUresult _cuGraphGetRootNodes(CUgraph hGraph, CUgraphNode* rootNodes, size_t* numRootNodes) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuGraphGetRootNodes
    _check_or_init_driver()
    if __cuGraphGetRootNodes == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphGetRootNodes is not found")
    return (<CUresult (*)(CUgraph, CUgraphNode*, size_t*) noexcept nogil>__cuGraphGetRootNodes)(
        hGraph, rootNodes, numRootNodes)


cdef CUresult _cuGraphGetEdges_v2(CUgraph hGraph, CUgraphNode* from_, CUgraphNode* to, CUgraphEdgeData* edgeData, size_t* numEdges) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuGraphGetEdges_v2
    _check_or_init_driver()
    if __cuGraphGetEdges_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphGetEdges_v2 is not found")
    return (<CUresult (*)(CUgraph, CUgraphNode*, CUgraphNode*, CUgraphEdgeData*, size_t*) noexcept nogil>__cuGraphGetEdges_v2)(
        hGraph, from_, to, edgeData, numEdges)


cdef CUresult _cuGraphNodeGetDependencies_v2(CUgraphNode hNode, CUgraphNode* dependencies, CUgraphEdgeData* edgeData, size_t* numDependencies) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuGraphNodeGetDependencies_v2
    _check_or_init_driver()
    if __cuGraphNodeGetDependencies_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphNodeGetDependencies_v2 is not found")
    return (<CUresult (*)(CUgraphNode, CUgraphNode*, CUgraphEdgeData*, size_t*) noexcept nogil>__cuGraphNodeGetDependencies_v2)(
        hNode, dependencies, edgeData, numDependencies)


cdef CUresult _cuGraphNodeGetDependentNodes_v2(CUgraphNode hNode, CUgraphNode* dependentNodes, CUgraphEdgeData* edgeData, size_t* numDependentNodes) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuGraphNodeGetDependentNodes_v2
    _check_or_init_driver()
    if __cuGraphNodeGetDependentNodes_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphNodeGetDependentNodes_v2 is not found")
    return (<CUresult (*)(CUgraphNode, CUgraphNode*, CUgraphEdgeData*, size_t*) noexcept nogil>__cuGraphNodeGetDependentNodes_v2)(
        hNode, dependentNodes, edgeData, numDependentNodes)


cdef CUresult _cuGraphAddDependencies_v2(CUgraph hGraph, const CUgraphNode* from_, const CUgraphNode* to, const CUgraphEdgeData* edgeData, size_t numDependencies) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuGraphAddDependencies_v2
    _check_or_init_driver()
    if __cuGraphAddDependencies_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphAddDependencies_v2 is not found")
    return (<CUresult (*)(CUgraph, const CUgraphNode*, const CUgraphNode*, const CUgraphEdgeData*, size_t) noexcept nogil>__cuGraphAddDependencies_v2)(
        hGraph, from_, to, edgeData, numDependencies)


cdef CUresult _cuGraphRemoveDependencies_v2(CUgraph hGraph, const CUgraphNode* from_, const CUgraphNode* to, const CUgraphEdgeData* edgeData, size_t numDependencies) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuGraphRemoveDependencies_v2
    _check_or_init_driver()
    if __cuGraphRemoveDependencies_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphRemoveDependencies_v2 is not found")
    return (<CUresult (*)(CUgraph, const CUgraphNode*, const CUgraphNode*, const CUgraphEdgeData*, size_t) noexcept nogil>__cuGraphRemoveDependencies_v2)(
        hGraph, from_, to, edgeData, numDependencies)


cdef CUresult _cuGraphDestroyNode(CUgraphNode hNode) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuGraphDestroyNode
    _check_or_init_driver()
    if __cuGraphDestroyNode == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphDestroyNode is not found")
    return (<CUresult (*)(CUgraphNode) noexcept nogil>__cuGraphDestroyNode)(
        hNode)


cdef CUresult _cuGraphInstantiateWithFlags(CUgraphExec* phGraphExec, CUgraph hGraph, unsigned long long flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuGraphInstantiateWithFlags
    _check_or_init_driver()
    if __cuGraphInstantiateWithFlags == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphInstantiateWithFlags is not found")
    return (<CUresult (*)(CUgraphExec*, CUgraph, unsigned long long) noexcept nogil>__cuGraphInstantiateWithFlags)(
        phGraphExec, hGraph, flags)


cdef CUresult _cuGraphInstantiateWithParams(CUgraphExec* phGraphExec, CUgraph hGraph, CUDA_GRAPH_INSTANTIATE_PARAMS* instantiateParams) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuGraphInstantiateWithParams
    _check_or_init_driver()
    if __cuGraphInstantiateWithParams == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphInstantiateWithParams is not found")
    return (<CUresult (*)(CUgraphExec*, CUgraph, CUDA_GRAPH_INSTANTIATE_PARAMS*) noexcept nogil>__cuGraphInstantiateWithParams)(
        phGraphExec, hGraph, instantiateParams)


cdef CUresult _cuGraphExecGetFlags(CUgraphExec hGraphExec, cuuint64_t* flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuGraphExecGetFlags
    _check_or_init_driver()
    if __cuGraphExecGetFlags == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphExecGetFlags is not found")
    return (<CUresult (*)(CUgraphExec, cuuint64_t*) noexcept nogil>__cuGraphExecGetFlags)(
        hGraphExec, flags)


cdef CUresult _cuGraphExecKernelNodeSetParams_v2(CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_KERNEL_NODE_PARAMS* nodeParams) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuGraphExecKernelNodeSetParams_v2
    _check_or_init_driver()
    if __cuGraphExecKernelNodeSetParams_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphExecKernelNodeSetParams_v2 is not found")
    return (<CUresult (*)(CUgraphExec, CUgraphNode, const CUDA_KERNEL_NODE_PARAMS*) noexcept nogil>__cuGraphExecKernelNodeSetParams_v2)(
        hGraphExec, hNode, nodeParams)


cdef CUresult _cuGraphExecMemcpyNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_MEMCPY3D* copyParams, CUcontext ctx) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuGraphExecMemcpyNodeSetParams
    _check_or_init_driver()
    if __cuGraphExecMemcpyNodeSetParams == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphExecMemcpyNodeSetParams is not found")
    return (<CUresult (*)(CUgraphExec, CUgraphNode, const CUDA_MEMCPY3D*, CUcontext) noexcept nogil>__cuGraphExecMemcpyNodeSetParams)(
        hGraphExec, hNode, copyParams, ctx)


cdef CUresult _cuGraphExecMemsetNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_MEMSET_NODE_PARAMS* memsetParams, CUcontext ctx) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuGraphExecMemsetNodeSetParams
    _check_or_init_driver()
    if __cuGraphExecMemsetNodeSetParams == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphExecMemsetNodeSetParams is not found")
    return (<CUresult (*)(CUgraphExec, CUgraphNode, const CUDA_MEMSET_NODE_PARAMS*, CUcontext) noexcept nogil>__cuGraphExecMemsetNodeSetParams)(
        hGraphExec, hNode, memsetParams, ctx)


cdef CUresult _cuGraphExecHostNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_HOST_NODE_PARAMS* nodeParams) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuGraphExecHostNodeSetParams
    _check_or_init_driver()
    if __cuGraphExecHostNodeSetParams == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphExecHostNodeSetParams is not found")
    return (<CUresult (*)(CUgraphExec, CUgraphNode, const CUDA_HOST_NODE_PARAMS*) noexcept nogil>__cuGraphExecHostNodeSetParams)(
        hGraphExec, hNode, nodeParams)


cdef CUresult _cuGraphExecChildGraphNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, CUgraph childGraph) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuGraphExecChildGraphNodeSetParams
    _check_or_init_driver()
    if __cuGraphExecChildGraphNodeSetParams == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphExecChildGraphNodeSetParams is not found")
    return (<CUresult (*)(CUgraphExec, CUgraphNode, CUgraph) noexcept nogil>__cuGraphExecChildGraphNodeSetParams)(
        hGraphExec, hNode, childGraph)


cdef CUresult _cuGraphExecEventRecordNodeSetEvent(CUgraphExec hGraphExec, CUgraphNode hNode, CUevent event) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuGraphExecEventRecordNodeSetEvent
    _check_or_init_driver()
    if __cuGraphExecEventRecordNodeSetEvent == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphExecEventRecordNodeSetEvent is not found")
    return (<CUresult (*)(CUgraphExec, CUgraphNode, CUevent) noexcept nogil>__cuGraphExecEventRecordNodeSetEvent)(
        hGraphExec, hNode, event)


cdef CUresult _cuGraphExecEventWaitNodeSetEvent(CUgraphExec hGraphExec, CUgraphNode hNode, CUevent event) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuGraphExecEventWaitNodeSetEvent
    _check_or_init_driver()
    if __cuGraphExecEventWaitNodeSetEvent == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphExecEventWaitNodeSetEvent is not found")
    return (<CUresult (*)(CUgraphExec, CUgraphNode, CUevent) noexcept nogil>__cuGraphExecEventWaitNodeSetEvent)(
        hGraphExec, hNode, event)


cdef CUresult _cuGraphExecExternalSemaphoresSignalNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS* nodeParams) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuGraphExecExternalSemaphoresSignalNodeSetParams
    _check_or_init_driver()
    if __cuGraphExecExternalSemaphoresSignalNodeSetParams == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphExecExternalSemaphoresSignalNodeSetParams is not found")
    return (<CUresult (*)(CUgraphExec, CUgraphNode, const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS*) noexcept nogil>__cuGraphExecExternalSemaphoresSignalNodeSetParams)(
        hGraphExec, hNode, nodeParams)


cdef CUresult _cuGraphExecExternalSemaphoresWaitNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_EXT_SEM_WAIT_NODE_PARAMS* nodeParams) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuGraphExecExternalSemaphoresWaitNodeSetParams
    _check_or_init_driver()
    if __cuGraphExecExternalSemaphoresWaitNodeSetParams == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphExecExternalSemaphoresWaitNodeSetParams is not found")
    return (<CUresult (*)(CUgraphExec, CUgraphNode, const CUDA_EXT_SEM_WAIT_NODE_PARAMS*) noexcept nogil>__cuGraphExecExternalSemaphoresWaitNodeSetParams)(
        hGraphExec, hNode, nodeParams)


cdef CUresult _cuGraphNodeSetEnabled(CUgraphExec hGraphExec, CUgraphNode hNode, unsigned int isEnabled) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuGraphNodeSetEnabled
    _check_or_init_driver()
    if __cuGraphNodeSetEnabled == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphNodeSetEnabled is not found")
    return (<CUresult (*)(CUgraphExec, CUgraphNode, unsigned int) noexcept nogil>__cuGraphNodeSetEnabled)(
        hGraphExec, hNode, isEnabled)


cdef CUresult _cuGraphNodeGetEnabled(CUgraphExec hGraphExec, CUgraphNode hNode, unsigned int* isEnabled) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuGraphNodeGetEnabled
    _check_or_init_driver()
    if __cuGraphNodeGetEnabled == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphNodeGetEnabled is not found")
    return (<CUresult (*)(CUgraphExec, CUgraphNode, unsigned int*) noexcept nogil>__cuGraphNodeGetEnabled)(
        hGraphExec, hNode, isEnabled)


cdef CUresult _cuGraphUpload(CUgraphExec hGraphExec, CUstream hStream) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuGraphUpload
    _check_or_init_driver()
    if __cuGraphUpload == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphUpload is not found")
    return (<CUresult (*)(CUgraphExec, CUstream) noexcept nogil>__cuGraphUpload)(
        hGraphExec, hStream)


cdef CUresult _cuGraphLaunch(CUgraphExec hGraphExec, CUstream hStream) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuGraphLaunch
    _check_or_init_driver()
    if __cuGraphLaunch == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphLaunch is not found")
    return (<CUresult (*)(CUgraphExec, CUstream) noexcept nogil>__cuGraphLaunch)(
        hGraphExec, hStream)


cdef CUresult _cuGraphExecDestroy(CUgraphExec hGraphExec) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuGraphExecDestroy
    _check_or_init_driver()
    if __cuGraphExecDestroy == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphExecDestroy is not found")
    return (<CUresult (*)(CUgraphExec) noexcept nogil>__cuGraphExecDestroy)(
        hGraphExec)


cdef CUresult _cuGraphDestroy(CUgraph hGraph) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuGraphDestroy
    _check_or_init_driver()
    if __cuGraphDestroy == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphDestroy is not found")
    return (<CUresult (*)(CUgraph) noexcept nogil>__cuGraphDestroy)(
        hGraph)


cdef CUresult _cuGraphExecUpdate_v2(CUgraphExec hGraphExec, CUgraph hGraph, CUgraphExecUpdateResultInfo* resultInfo) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuGraphExecUpdate_v2
    _check_or_init_driver()
    if __cuGraphExecUpdate_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphExecUpdate_v2 is not found")
    return (<CUresult (*)(CUgraphExec, CUgraph, CUgraphExecUpdateResultInfo*) noexcept nogil>__cuGraphExecUpdate_v2)(
        hGraphExec, hGraph, resultInfo)


cdef CUresult _cuGraphKernelNodeCopyAttributes(CUgraphNode dst, CUgraphNode src) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuGraphKernelNodeCopyAttributes
    _check_or_init_driver()
    if __cuGraphKernelNodeCopyAttributes == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphKernelNodeCopyAttributes is not found")
    return (<CUresult (*)(CUgraphNode, CUgraphNode) noexcept nogil>__cuGraphKernelNodeCopyAttributes)(
        dst, src)


cdef CUresult _cuGraphKernelNodeGetAttribute(CUgraphNode hNode, CUkernelNodeAttrID attr, CUkernelNodeAttrValue* value_out) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuGraphKernelNodeGetAttribute
    _check_or_init_driver()
    if __cuGraphKernelNodeGetAttribute == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphKernelNodeGetAttribute is not found")
    return (<CUresult (*)(CUgraphNode, CUkernelNodeAttrID, CUkernelNodeAttrValue*) noexcept nogil>__cuGraphKernelNodeGetAttribute)(
        hNode, attr, value_out)


cdef CUresult _cuGraphKernelNodeSetAttribute(CUgraphNode hNode, CUkernelNodeAttrID attr, const CUkernelNodeAttrValue* value) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuGraphKernelNodeSetAttribute
    _check_or_init_driver()
    if __cuGraphKernelNodeSetAttribute == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphKernelNodeSetAttribute is not found")
    return (<CUresult (*)(CUgraphNode, CUkernelNodeAttrID, const CUkernelNodeAttrValue*) noexcept nogil>__cuGraphKernelNodeSetAttribute)(
        hNode, attr, value)


cdef CUresult _cuGraphDebugDotPrint(CUgraph hGraph, const char* path, unsigned int flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuGraphDebugDotPrint
    _check_or_init_driver()
    if __cuGraphDebugDotPrint == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphDebugDotPrint is not found")
    return (<CUresult (*)(CUgraph, const char*, unsigned int) noexcept nogil>__cuGraphDebugDotPrint)(
        hGraph, path, flags)


cdef CUresult _cuUserObjectCreate(CUuserObject* object_out, void* ptr, CUhostFn destroy, unsigned int initialRefcount, unsigned int flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuUserObjectCreate
    _check_or_init_driver()
    if __cuUserObjectCreate == NULL:
        with gil:
            raise FunctionNotFoundError("function cuUserObjectCreate is not found")
    return (<CUresult (*)(CUuserObject*, void*, CUhostFn, unsigned int, unsigned int) noexcept nogil>__cuUserObjectCreate)(
        object_out, ptr, destroy, initialRefcount, flags)


cdef CUresult _cuUserObjectRetain(CUuserObject object, unsigned int count) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuUserObjectRetain
    _check_or_init_driver()
    if __cuUserObjectRetain == NULL:
        with gil:
            raise FunctionNotFoundError("function cuUserObjectRetain is not found")
    return (<CUresult (*)(CUuserObject, unsigned int) noexcept nogil>__cuUserObjectRetain)(
        object, count)


cdef CUresult _cuUserObjectRelease(CUuserObject object, unsigned int count) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuUserObjectRelease
    _check_or_init_driver()
    if __cuUserObjectRelease == NULL:
        with gil:
            raise FunctionNotFoundError("function cuUserObjectRelease is not found")
    return (<CUresult (*)(CUuserObject, unsigned int) noexcept nogil>__cuUserObjectRelease)(
        object, count)


cdef CUresult _cuGraphRetainUserObject(CUgraph graph, CUuserObject object, unsigned int count, unsigned int flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuGraphRetainUserObject
    _check_or_init_driver()
    if __cuGraphRetainUserObject == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphRetainUserObject is not found")
    return (<CUresult (*)(CUgraph, CUuserObject, unsigned int, unsigned int) noexcept nogil>__cuGraphRetainUserObject)(
        graph, object, count, flags)


cdef CUresult _cuGraphReleaseUserObject(CUgraph graph, CUuserObject object, unsigned int count) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuGraphReleaseUserObject
    _check_or_init_driver()
    if __cuGraphReleaseUserObject == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphReleaseUserObject is not found")
    return (<CUresult (*)(CUgraph, CUuserObject, unsigned int) noexcept nogil>__cuGraphReleaseUserObject)(
        graph, object, count)


cdef CUresult _cuGraphAddNode_v2(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, const CUgraphEdgeData* dependencyData, size_t numDependencies, CUgraphNodeParams* nodeParams) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuGraphAddNode_v2
    _check_or_init_driver()
    if __cuGraphAddNode_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphAddNode_v2 is not found")
    return (<CUresult (*)(CUgraphNode*, CUgraph, const CUgraphNode*, const CUgraphEdgeData*, size_t, CUgraphNodeParams*) noexcept nogil>__cuGraphAddNode_v2)(
        phGraphNode, hGraph, dependencies, dependencyData, numDependencies, nodeParams)


cdef CUresult _cuGraphNodeSetParams(CUgraphNode hNode, CUgraphNodeParams* nodeParams) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuGraphNodeSetParams
    _check_or_init_driver()
    if __cuGraphNodeSetParams == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphNodeSetParams is not found")
    return (<CUresult (*)(CUgraphNode, CUgraphNodeParams*) noexcept nogil>__cuGraphNodeSetParams)(
        hNode, nodeParams)


cdef CUresult _cuGraphExecNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, CUgraphNodeParams* nodeParams) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuGraphExecNodeSetParams
    _check_or_init_driver()
    if __cuGraphExecNodeSetParams == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphExecNodeSetParams is not found")
    return (<CUresult (*)(CUgraphExec, CUgraphNode, CUgraphNodeParams*) noexcept nogil>__cuGraphExecNodeSetParams)(
        hGraphExec, hNode, nodeParams)


cdef CUresult _cuGraphConditionalHandleCreate(CUgraphConditionalHandle* pHandle_out, CUgraph hGraph, CUcontext ctx, unsigned int defaultLaunchValue, unsigned int flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuGraphConditionalHandleCreate
    _check_or_init_driver()
    if __cuGraphConditionalHandleCreate == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphConditionalHandleCreate is not found")
    return (<CUresult (*)(CUgraphConditionalHandle*, CUgraph, CUcontext, unsigned int, unsigned int) noexcept nogil>__cuGraphConditionalHandleCreate)(
        pHandle_out, hGraph, ctx, defaultLaunchValue, flags)


cdef CUresult _cuOccupancyMaxActiveBlocksPerMultiprocessor(int* numBlocks, CUfunction func, int blockSize, size_t dynamicSMemSize) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuOccupancyMaxActiveBlocksPerMultiprocessor
    _check_or_init_driver()
    if __cuOccupancyMaxActiveBlocksPerMultiprocessor == NULL:
        with gil:
            raise FunctionNotFoundError("function cuOccupancyMaxActiveBlocksPerMultiprocessor is not found")
    return (<CUresult (*)(int*, CUfunction, int, size_t) noexcept nogil>__cuOccupancyMaxActiveBlocksPerMultiprocessor)(
        numBlocks, func, blockSize, dynamicSMemSize)


cdef CUresult _cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int* numBlocks, CUfunction func, int blockSize, size_t dynamicSMemSize, unsigned int flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags
    _check_or_init_driver()
    if __cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags == NULL:
        with gil:
            raise FunctionNotFoundError("function cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags is not found")
    return (<CUresult (*)(int*, CUfunction, int, size_t, unsigned int) noexcept nogil>__cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags)(
        numBlocks, func, blockSize, dynamicSMemSize, flags)


cdef CUresult _cuOccupancyMaxPotentialBlockSize(int* minGridSize, int* blockSize, CUfunction func, CUoccupancyB2DSize blockSizeToDynamicSMemSize, size_t dynamicSMemSize, int blockSizeLimit) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuOccupancyMaxPotentialBlockSize
    _check_or_init_driver()
    if __cuOccupancyMaxPotentialBlockSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cuOccupancyMaxPotentialBlockSize is not found")
    return (<CUresult (*)(int*, int*, CUfunction, CUoccupancyB2DSize, size_t, int) noexcept nogil>__cuOccupancyMaxPotentialBlockSize)(
        minGridSize, blockSize, func, blockSizeToDynamicSMemSize, dynamicSMemSize, blockSizeLimit)


cdef CUresult _cuOccupancyMaxPotentialBlockSizeWithFlags(int* minGridSize, int* blockSize, CUfunction func, CUoccupancyB2DSize blockSizeToDynamicSMemSize, size_t dynamicSMemSize, int blockSizeLimit, unsigned int flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuOccupancyMaxPotentialBlockSizeWithFlags
    _check_or_init_driver()
    if __cuOccupancyMaxPotentialBlockSizeWithFlags == NULL:
        with gil:
            raise FunctionNotFoundError("function cuOccupancyMaxPotentialBlockSizeWithFlags is not found")
    return (<CUresult (*)(int*, int*, CUfunction, CUoccupancyB2DSize, size_t, int, unsigned int) noexcept nogil>__cuOccupancyMaxPotentialBlockSizeWithFlags)(
        minGridSize, blockSize, func, blockSizeToDynamicSMemSize, dynamicSMemSize, blockSizeLimit, flags)


cdef CUresult _cuOccupancyAvailableDynamicSMemPerBlock(size_t* dynamicSmemSize, CUfunction func, int numBlocks, int blockSize) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuOccupancyAvailableDynamicSMemPerBlock
    _check_or_init_driver()
    if __cuOccupancyAvailableDynamicSMemPerBlock == NULL:
        with gil:
            raise FunctionNotFoundError("function cuOccupancyAvailableDynamicSMemPerBlock is not found")
    return (<CUresult (*)(size_t*, CUfunction, int, int) noexcept nogil>__cuOccupancyAvailableDynamicSMemPerBlock)(
        dynamicSmemSize, func, numBlocks, blockSize)


cdef CUresult _cuOccupancyMaxPotentialClusterSize(int* clusterSize, CUfunction func, const CUlaunchConfig* config) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuOccupancyMaxPotentialClusterSize
    _check_or_init_driver()
    if __cuOccupancyMaxPotentialClusterSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cuOccupancyMaxPotentialClusterSize is not found")
    return (<CUresult (*)(int*, CUfunction, const CUlaunchConfig*) noexcept nogil>__cuOccupancyMaxPotentialClusterSize)(
        clusterSize, func, config)


cdef CUresult _cuOccupancyMaxActiveClusters(int* numClusters, CUfunction func, const CUlaunchConfig* config) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuOccupancyMaxActiveClusters
    _check_or_init_driver()
    if __cuOccupancyMaxActiveClusters == NULL:
        with gil:
            raise FunctionNotFoundError("function cuOccupancyMaxActiveClusters is not found")
    return (<CUresult (*)(int*, CUfunction, const CUlaunchConfig*) noexcept nogil>__cuOccupancyMaxActiveClusters)(
        numClusters, func, config)


cdef CUresult _cuTexRefSetArray(CUtexref hTexRef, CUarray hArray, unsigned int Flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuTexRefSetArray
    _check_or_init_driver()
    if __cuTexRefSetArray == NULL:
        with gil:
            raise FunctionNotFoundError("function cuTexRefSetArray is not found")
    return (<CUresult (*)(CUtexref, CUarray, unsigned int) noexcept nogil>__cuTexRefSetArray)(
        hTexRef, hArray, Flags)


cdef CUresult _cuTexRefSetMipmappedArray(CUtexref hTexRef, CUmipmappedArray hMipmappedArray, unsigned int Flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuTexRefSetMipmappedArray
    _check_or_init_driver()
    if __cuTexRefSetMipmappedArray == NULL:
        with gil:
            raise FunctionNotFoundError("function cuTexRefSetMipmappedArray is not found")
    return (<CUresult (*)(CUtexref, CUmipmappedArray, unsigned int) noexcept nogil>__cuTexRefSetMipmappedArray)(
        hTexRef, hMipmappedArray, Flags)


cdef CUresult _cuTexRefSetAddress_v2(size_t* ByteOffset, CUtexref hTexRef, CUdeviceptr dptr, size_t bytes) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuTexRefSetAddress_v2
    _check_or_init_driver()
    if __cuTexRefSetAddress_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuTexRefSetAddress_v2 is not found")
    return (<CUresult (*)(size_t*, CUtexref, CUdeviceptr, size_t) noexcept nogil>__cuTexRefSetAddress_v2)(
        ByteOffset, hTexRef, dptr, bytes)


cdef CUresult _cuTexRefSetAddress2D_v3(CUtexref hTexRef, const CUDA_ARRAY_DESCRIPTOR* desc, CUdeviceptr dptr, size_t Pitch) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuTexRefSetAddress2D_v3
    _check_or_init_driver()
    if __cuTexRefSetAddress2D_v3 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuTexRefSetAddress2D_v3 is not found")
    return (<CUresult (*)(CUtexref, const CUDA_ARRAY_DESCRIPTOR*, CUdeviceptr, size_t) noexcept nogil>__cuTexRefSetAddress2D_v3)(
        hTexRef, desc, dptr, Pitch)


cdef CUresult _cuTexRefSetFormat(CUtexref hTexRef, CUarray_format fmt, int NumPackedComponents) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuTexRefSetFormat
    _check_or_init_driver()
    if __cuTexRefSetFormat == NULL:
        with gil:
            raise FunctionNotFoundError("function cuTexRefSetFormat is not found")
    return (<CUresult (*)(CUtexref, CUarray_format, int) noexcept nogil>__cuTexRefSetFormat)(
        hTexRef, fmt, NumPackedComponents)


cdef CUresult _cuTexRefSetAddressMode(CUtexref hTexRef, int dim, CUaddress_mode am) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuTexRefSetAddressMode
    _check_or_init_driver()
    if __cuTexRefSetAddressMode == NULL:
        with gil:
            raise FunctionNotFoundError("function cuTexRefSetAddressMode is not found")
    return (<CUresult (*)(CUtexref, int, CUaddress_mode) noexcept nogil>__cuTexRefSetAddressMode)(
        hTexRef, dim, am)


cdef CUresult _cuTexRefSetFilterMode(CUtexref hTexRef, CUfilter_mode fm) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuTexRefSetFilterMode
    _check_or_init_driver()
    if __cuTexRefSetFilterMode == NULL:
        with gil:
            raise FunctionNotFoundError("function cuTexRefSetFilterMode is not found")
    return (<CUresult (*)(CUtexref, CUfilter_mode) noexcept nogil>__cuTexRefSetFilterMode)(
        hTexRef, fm)


cdef CUresult _cuTexRefSetMipmapFilterMode(CUtexref hTexRef, CUfilter_mode fm) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuTexRefSetMipmapFilterMode
    _check_or_init_driver()
    if __cuTexRefSetMipmapFilterMode == NULL:
        with gil:
            raise FunctionNotFoundError("function cuTexRefSetMipmapFilterMode is not found")
    return (<CUresult (*)(CUtexref, CUfilter_mode) noexcept nogil>__cuTexRefSetMipmapFilterMode)(
        hTexRef, fm)


cdef CUresult _cuTexRefSetMipmapLevelBias(CUtexref hTexRef, float bias) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuTexRefSetMipmapLevelBias
    _check_or_init_driver()
    if __cuTexRefSetMipmapLevelBias == NULL:
        with gil:
            raise FunctionNotFoundError("function cuTexRefSetMipmapLevelBias is not found")
    return (<CUresult (*)(CUtexref, float) noexcept nogil>__cuTexRefSetMipmapLevelBias)(
        hTexRef, bias)


cdef CUresult _cuTexRefSetMipmapLevelClamp(CUtexref hTexRef, float minMipmapLevelClamp, float maxMipmapLevelClamp) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuTexRefSetMipmapLevelClamp
    _check_or_init_driver()
    if __cuTexRefSetMipmapLevelClamp == NULL:
        with gil:
            raise FunctionNotFoundError("function cuTexRefSetMipmapLevelClamp is not found")
    return (<CUresult (*)(CUtexref, float, float) noexcept nogil>__cuTexRefSetMipmapLevelClamp)(
        hTexRef, minMipmapLevelClamp, maxMipmapLevelClamp)


cdef CUresult _cuTexRefSetMaxAnisotropy(CUtexref hTexRef, unsigned int maxAniso) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuTexRefSetMaxAnisotropy
    _check_or_init_driver()
    if __cuTexRefSetMaxAnisotropy == NULL:
        with gil:
            raise FunctionNotFoundError("function cuTexRefSetMaxAnisotropy is not found")
    return (<CUresult (*)(CUtexref, unsigned int) noexcept nogil>__cuTexRefSetMaxAnisotropy)(
        hTexRef, maxAniso)


cdef CUresult _cuTexRefSetBorderColor(CUtexref hTexRef, float* pBorderColor) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuTexRefSetBorderColor
    _check_or_init_driver()
    if __cuTexRefSetBorderColor == NULL:
        with gil:
            raise FunctionNotFoundError("function cuTexRefSetBorderColor is not found")
    return (<CUresult (*)(CUtexref, float*) noexcept nogil>__cuTexRefSetBorderColor)(
        hTexRef, pBorderColor)


cdef CUresult _cuTexRefSetFlags(CUtexref hTexRef, unsigned int Flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuTexRefSetFlags
    _check_or_init_driver()
    if __cuTexRefSetFlags == NULL:
        with gil:
            raise FunctionNotFoundError("function cuTexRefSetFlags is not found")
    return (<CUresult (*)(CUtexref, unsigned int) noexcept nogil>__cuTexRefSetFlags)(
        hTexRef, Flags)


cdef CUresult _cuTexRefGetAddress_v2(CUdeviceptr* pdptr, CUtexref hTexRef) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuTexRefGetAddress_v2
    _check_or_init_driver()
    if __cuTexRefGetAddress_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuTexRefGetAddress_v2 is not found")
    return (<CUresult (*)(CUdeviceptr*, CUtexref) noexcept nogil>__cuTexRefGetAddress_v2)(
        pdptr, hTexRef)


cdef CUresult _cuTexRefGetArray(CUarray* phArray, CUtexref hTexRef) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuTexRefGetArray
    _check_or_init_driver()
    if __cuTexRefGetArray == NULL:
        with gil:
            raise FunctionNotFoundError("function cuTexRefGetArray is not found")
    return (<CUresult (*)(CUarray*, CUtexref) noexcept nogil>__cuTexRefGetArray)(
        phArray, hTexRef)


cdef CUresult _cuTexRefGetMipmappedArray(CUmipmappedArray* phMipmappedArray, CUtexref hTexRef) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuTexRefGetMipmappedArray
    _check_or_init_driver()
    if __cuTexRefGetMipmappedArray == NULL:
        with gil:
            raise FunctionNotFoundError("function cuTexRefGetMipmappedArray is not found")
    return (<CUresult (*)(CUmipmappedArray*, CUtexref) noexcept nogil>__cuTexRefGetMipmappedArray)(
        phMipmappedArray, hTexRef)


cdef CUresult _cuTexRefGetAddressMode(CUaddress_mode* pam, CUtexref hTexRef, int dim) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuTexRefGetAddressMode
    _check_or_init_driver()
    if __cuTexRefGetAddressMode == NULL:
        with gil:
            raise FunctionNotFoundError("function cuTexRefGetAddressMode is not found")
    return (<CUresult (*)(CUaddress_mode*, CUtexref, int) noexcept nogil>__cuTexRefGetAddressMode)(
        pam, hTexRef, dim)


cdef CUresult _cuTexRefGetFilterMode(CUfilter_mode* pfm, CUtexref hTexRef) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuTexRefGetFilterMode
    _check_or_init_driver()
    if __cuTexRefGetFilterMode == NULL:
        with gil:
            raise FunctionNotFoundError("function cuTexRefGetFilterMode is not found")
    return (<CUresult (*)(CUfilter_mode*, CUtexref) noexcept nogil>__cuTexRefGetFilterMode)(
        pfm, hTexRef)


cdef CUresult _cuTexRefGetFormat(CUarray_format* pFormat, int* pNumChannels, CUtexref hTexRef) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuTexRefGetFormat
    _check_or_init_driver()
    if __cuTexRefGetFormat == NULL:
        with gil:
            raise FunctionNotFoundError("function cuTexRefGetFormat is not found")
    return (<CUresult (*)(CUarray_format*, int*, CUtexref) noexcept nogil>__cuTexRefGetFormat)(
        pFormat, pNumChannels, hTexRef)


cdef CUresult _cuTexRefGetMipmapFilterMode(CUfilter_mode* pfm, CUtexref hTexRef) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuTexRefGetMipmapFilterMode
    _check_or_init_driver()
    if __cuTexRefGetMipmapFilterMode == NULL:
        with gil:
            raise FunctionNotFoundError("function cuTexRefGetMipmapFilterMode is not found")
    return (<CUresult (*)(CUfilter_mode*, CUtexref) noexcept nogil>__cuTexRefGetMipmapFilterMode)(
        pfm, hTexRef)


cdef CUresult _cuTexRefGetMipmapLevelBias(float* pbias, CUtexref hTexRef) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuTexRefGetMipmapLevelBias
    _check_or_init_driver()
    if __cuTexRefGetMipmapLevelBias == NULL:
        with gil:
            raise FunctionNotFoundError("function cuTexRefGetMipmapLevelBias is not found")
    return (<CUresult (*)(float*, CUtexref) noexcept nogil>__cuTexRefGetMipmapLevelBias)(
        pbias, hTexRef)


cdef CUresult _cuTexRefGetMipmapLevelClamp(float* pminMipmapLevelClamp, float* pmaxMipmapLevelClamp, CUtexref hTexRef) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuTexRefGetMipmapLevelClamp
    _check_or_init_driver()
    if __cuTexRefGetMipmapLevelClamp == NULL:
        with gil:
            raise FunctionNotFoundError("function cuTexRefGetMipmapLevelClamp is not found")
    return (<CUresult (*)(float*, float*, CUtexref) noexcept nogil>__cuTexRefGetMipmapLevelClamp)(
        pminMipmapLevelClamp, pmaxMipmapLevelClamp, hTexRef)


cdef CUresult _cuTexRefGetMaxAnisotropy(int* pmaxAniso, CUtexref hTexRef) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuTexRefGetMaxAnisotropy
    _check_or_init_driver()
    if __cuTexRefGetMaxAnisotropy == NULL:
        with gil:
            raise FunctionNotFoundError("function cuTexRefGetMaxAnisotropy is not found")
    return (<CUresult (*)(int*, CUtexref) noexcept nogil>__cuTexRefGetMaxAnisotropy)(
        pmaxAniso, hTexRef)


cdef CUresult _cuTexRefGetBorderColor(float* pBorderColor, CUtexref hTexRef) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuTexRefGetBorderColor
    _check_or_init_driver()
    if __cuTexRefGetBorderColor == NULL:
        with gil:
            raise FunctionNotFoundError("function cuTexRefGetBorderColor is not found")
    return (<CUresult (*)(float*, CUtexref) noexcept nogil>__cuTexRefGetBorderColor)(
        pBorderColor, hTexRef)


cdef CUresult _cuTexRefGetFlags(unsigned int* pFlags, CUtexref hTexRef) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuTexRefGetFlags
    _check_or_init_driver()
    if __cuTexRefGetFlags == NULL:
        with gil:
            raise FunctionNotFoundError("function cuTexRefGetFlags is not found")
    return (<CUresult (*)(unsigned int*, CUtexref) noexcept nogil>__cuTexRefGetFlags)(
        pFlags, hTexRef)


cdef CUresult _cuTexRefCreate(CUtexref* pTexRef) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuTexRefCreate
    _check_or_init_driver()
    if __cuTexRefCreate == NULL:
        with gil:
            raise FunctionNotFoundError("function cuTexRefCreate is not found")
    return (<CUresult (*)(CUtexref*) noexcept nogil>__cuTexRefCreate)(
        pTexRef)


cdef CUresult _cuTexRefDestroy(CUtexref hTexRef) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuTexRefDestroy
    _check_or_init_driver()
    if __cuTexRefDestroy == NULL:
        with gil:
            raise FunctionNotFoundError("function cuTexRefDestroy is not found")
    return (<CUresult (*)(CUtexref) noexcept nogil>__cuTexRefDestroy)(
        hTexRef)


cdef CUresult _cuSurfRefSetArray(CUsurfref hSurfRef, CUarray hArray, unsigned int Flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuSurfRefSetArray
    _check_or_init_driver()
    if __cuSurfRefSetArray == NULL:
        with gil:
            raise FunctionNotFoundError("function cuSurfRefSetArray is not found")
    return (<CUresult (*)(CUsurfref, CUarray, unsigned int) noexcept nogil>__cuSurfRefSetArray)(
        hSurfRef, hArray, Flags)


cdef CUresult _cuSurfRefGetArray(CUarray* phArray, CUsurfref hSurfRef) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuSurfRefGetArray
    _check_or_init_driver()
    if __cuSurfRefGetArray == NULL:
        with gil:
            raise FunctionNotFoundError("function cuSurfRefGetArray is not found")
    return (<CUresult (*)(CUarray*, CUsurfref) noexcept nogil>__cuSurfRefGetArray)(
        phArray, hSurfRef)


cdef CUresult _cuTexObjectCreate(CUtexObject* pTexObject, const CUDA_RESOURCE_DESC* pResDesc, const CUDA_TEXTURE_DESC* pTexDesc, const CUDA_RESOURCE_VIEW_DESC* pResViewDesc) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuTexObjectCreate
    _check_or_init_driver()
    if __cuTexObjectCreate == NULL:
        with gil:
            raise FunctionNotFoundError("function cuTexObjectCreate is not found")
    return (<CUresult (*)(CUtexObject*, const CUDA_RESOURCE_DESC*, const CUDA_TEXTURE_DESC*, const CUDA_RESOURCE_VIEW_DESC*) noexcept nogil>__cuTexObjectCreate)(
        pTexObject, pResDesc, pTexDesc, pResViewDesc)


cdef CUresult _cuTexObjectDestroy(CUtexObject texObject) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuTexObjectDestroy
    _check_or_init_driver()
    if __cuTexObjectDestroy == NULL:
        with gil:
            raise FunctionNotFoundError("function cuTexObjectDestroy is not found")
    return (<CUresult (*)(CUtexObject) noexcept nogil>__cuTexObjectDestroy)(
        texObject)


cdef CUresult _cuTexObjectGetResourceDesc(CUDA_RESOURCE_DESC* pResDesc, CUtexObject texObject) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuTexObjectGetResourceDesc
    _check_or_init_driver()
    if __cuTexObjectGetResourceDesc == NULL:
        with gil:
            raise FunctionNotFoundError("function cuTexObjectGetResourceDesc is not found")
    return (<CUresult (*)(CUDA_RESOURCE_DESC*, CUtexObject) noexcept nogil>__cuTexObjectGetResourceDesc)(
        pResDesc, texObject)


cdef CUresult _cuTexObjectGetTextureDesc(CUDA_TEXTURE_DESC* pTexDesc, CUtexObject texObject) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuTexObjectGetTextureDesc
    _check_or_init_driver()
    if __cuTexObjectGetTextureDesc == NULL:
        with gil:
            raise FunctionNotFoundError("function cuTexObjectGetTextureDesc is not found")
    return (<CUresult (*)(CUDA_TEXTURE_DESC*, CUtexObject) noexcept nogil>__cuTexObjectGetTextureDesc)(
        pTexDesc, texObject)


cdef CUresult _cuTexObjectGetResourceViewDesc(CUDA_RESOURCE_VIEW_DESC* pResViewDesc, CUtexObject texObject) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuTexObjectGetResourceViewDesc
    _check_or_init_driver()
    if __cuTexObjectGetResourceViewDesc == NULL:
        with gil:
            raise FunctionNotFoundError("function cuTexObjectGetResourceViewDesc is not found")
    return (<CUresult (*)(CUDA_RESOURCE_VIEW_DESC*, CUtexObject) noexcept nogil>__cuTexObjectGetResourceViewDesc)(
        pResViewDesc, texObject)


cdef CUresult _cuSurfObjectCreate(CUsurfObject* pSurfObject, const CUDA_RESOURCE_DESC* pResDesc) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuSurfObjectCreate
    _check_or_init_driver()
    if __cuSurfObjectCreate == NULL:
        with gil:
            raise FunctionNotFoundError("function cuSurfObjectCreate is not found")
    return (<CUresult (*)(CUsurfObject*, const CUDA_RESOURCE_DESC*) noexcept nogil>__cuSurfObjectCreate)(
        pSurfObject, pResDesc)


cdef CUresult _cuSurfObjectDestroy(CUsurfObject surfObject) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuSurfObjectDestroy
    _check_or_init_driver()
    if __cuSurfObjectDestroy == NULL:
        with gil:
            raise FunctionNotFoundError("function cuSurfObjectDestroy is not found")
    return (<CUresult (*)(CUsurfObject) noexcept nogil>__cuSurfObjectDestroy)(
        surfObject)


cdef CUresult _cuSurfObjectGetResourceDesc(CUDA_RESOURCE_DESC* pResDesc, CUsurfObject surfObject) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuSurfObjectGetResourceDesc
    _check_or_init_driver()
    if __cuSurfObjectGetResourceDesc == NULL:
        with gil:
            raise FunctionNotFoundError("function cuSurfObjectGetResourceDesc is not found")
    return (<CUresult (*)(CUDA_RESOURCE_DESC*, CUsurfObject) noexcept nogil>__cuSurfObjectGetResourceDesc)(
        pResDesc, surfObject)


cdef CUresult _cuTensorMapEncodeTiled(CUtensorMap* tensorMap, CUtensorMapDataType tensorDataType, cuuint32_t tensorRank, void* globalAddress, const cuuint64_t* globalDim, const cuuint64_t* globalStrides, const cuuint32_t* boxDim, const cuuint32_t* elementStrides, CUtensorMapInterleave interleave, CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2Promotion, CUtensorMapFloatOOBfill oobFill) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuTensorMapEncodeTiled
    _check_or_init_driver()
    if __cuTensorMapEncodeTiled == NULL:
        with gil:
            raise FunctionNotFoundError("function cuTensorMapEncodeTiled is not found")
    return (<CUresult (*)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*, const cuuint64_t*, const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave, CUtensorMapSwizzle, CUtensorMapL2promotion, CUtensorMapFloatOOBfill) noexcept nogil>__cuTensorMapEncodeTiled)(
        tensorMap, tensorDataType, tensorRank, globalAddress, globalDim, globalStrides, boxDim, elementStrides, interleave, swizzle, l2Promotion, oobFill)


cdef CUresult _cuTensorMapEncodeIm2col(CUtensorMap* tensorMap, CUtensorMapDataType tensorDataType, cuuint32_t tensorRank, void* globalAddress, const cuuint64_t* globalDim, const cuuint64_t* globalStrides, const int* pixelBoxLowerCorner, const int* pixelBoxUpperCorner, cuuint32_t channelsPerPixel, cuuint32_t pixelsPerColumn, const cuuint32_t* elementStrides, CUtensorMapInterleave interleave, CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2Promotion, CUtensorMapFloatOOBfill oobFill) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuTensorMapEncodeIm2col
    _check_or_init_driver()
    if __cuTensorMapEncodeIm2col == NULL:
        with gil:
            raise FunctionNotFoundError("function cuTensorMapEncodeIm2col is not found")
    return (<CUresult (*)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*, const cuuint64_t*, const int*, const int*, cuuint32_t, cuuint32_t, const cuuint32_t*, CUtensorMapInterleave, CUtensorMapSwizzle, CUtensorMapL2promotion, CUtensorMapFloatOOBfill) noexcept nogil>__cuTensorMapEncodeIm2col)(
        tensorMap, tensorDataType, tensorRank, globalAddress, globalDim, globalStrides, pixelBoxLowerCorner, pixelBoxUpperCorner, channelsPerPixel, pixelsPerColumn, elementStrides, interleave, swizzle, l2Promotion, oobFill)


cdef CUresult _cuTensorMapEncodeIm2colWide(CUtensorMap* tensorMap, CUtensorMapDataType tensorDataType, cuuint32_t tensorRank, void* globalAddress, const cuuint64_t* globalDim, const cuuint64_t* globalStrides, int pixelBoxLowerCornerWidth, int pixelBoxUpperCornerWidth, cuuint32_t channelsPerPixel, cuuint32_t pixelsPerColumn, const cuuint32_t* elementStrides, CUtensorMapInterleave interleave, CUtensorMapIm2ColWideMode mode, CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2Promotion, CUtensorMapFloatOOBfill oobFill) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuTensorMapEncodeIm2colWide
    _check_or_init_driver()
    if __cuTensorMapEncodeIm2colWide == NULL:
        with gil:
            raise FunctionNotFoundError("function cuTensorMapEncodeIm2colWide is not found")
    return (<CUresult (*)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*, const cuuint64_t*, int, int, cuuint32_t, cuuint32_t, const cuuint32_t*, CUtensorMapInterleave, CUtensorMapIm2ColWideMode, CUtensorMapSwizzle, CUtensorMapL2promotion, CUtensorMapFloatOOBfill) noexcept nogil>__cuTensorMapEncodeIm2colWide)(
        tensorMap, tensorDataType, tensorRank, globalAddress, globalDim, globalStrides, pixelBoxLowerCornerWidth, pixelBoxUpperCornerWidth, channelsPerPixel, pixelsPerColumn, elementStrides, interleave, mode, swizzle, l2Promotion, oobFill)


cdef CUresult _cuTensorMapReplaceAddress(CUtensorMap* tensorMap, void* globalAddress) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuTensorMapReplaceAddress
    _check_or_init_driver()
    if __cuTensorMapReplaceAddress == NULL:
        with gil:
            raise FunctionNotFoundError("function cuTensorMapReplaceAddress is not found")
    return (<CUresult (*)(CUtensorMap*, void*) noexcept nogil>__cuTensorMapReplaceAddress)(
        tensorMap, globalAddress)


cdef CUresult _cuDeviceCanAccessPeer(int* canAccessPeer, CUdevice dev, CUdevice peerDev) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuDeviceCanAccessPeer
    _check_or_init_driver()
    if __cuDeviceCanAccessPeer == NULL:
        with gil:
            raise FunctionNotFoundError("function cuDeviceCanAccessPeer is not found")
    return (<CUresult (*)(int*, CUdevice, CUdevice) noexcept nogil>__cuDeviceCanAccessPeer)(
        canAccessPeer, dev, peerDev)


cdef CUresult _cuCtxEnablePeerAccess(CUcontext peerContext, unsigned int Flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuCtxEnablePeerAccess
    _check_or_init_driver()
    if __cuCtxEnablePeerAccess == NULL:
        with gil:
            raise FunctionNotFoundError("function cuCtxEnablePeerAccess is not found")
    return (<CUresult (*)(CUcontext, unsigned int) noexcept nogil>__cuCtxEnablePeerAccess)(
        peerContext, Flags)


cdef CUresult _cuCtxDisablePeerAccess(CUcontext peerContext) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuCtxDisablePeerAccess
    _check_or_init_driver()
    if __cuCtxDisablePeerAccess == NULL:
        with gil:
            raise FunctionNotFoundError("function cuCtxDisablePeerAccess is not found")
    return (<CUresult (*)(CUcontext) noexcept nogil>__cuCtxDisablePeerAccess)(
        peerContext)


cdef CUresult _cuDeviceGetP2PAttribute(int* value, CUdevice_P2PAttribute attrib, CUdevice srcDevice, CUdevice dstDevice) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuDeviceGetP2PAttribute
    _check_or_init_driver()
    if __cuDeviceGetP2PAttribute == NULL:
        with gil:
            raise FunctionNotFoundError("function cuDeviceGetP2PAttribute is not found")
    return (<CUresult (*)(int*, CUdevice_P2PAttribute, CUdevice, CUdevice) noexcept nogil>__cuDeviceGetP2PAttribute)(
        value, attrib, srcDevice, dstDevice)


cdef CUresult _cuGraphicsUnregisterResource(CUgraphicsResource resource) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuGraphicsUnregisterResource
    _check_or_init_driver()
    if __cuGraphicsUnregisterResource == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphicsUnregisterResource is not found")
    return (<CUresult (*)(CUgraphicsResource) noexcept nogil>__cuGraphicsUnregisterResource)(
        resource)


cdef CUresult _cuGraphicsSubResourceGetMappedArray(CUarray* pArray, CUgraphicsResource resource, unsigned int arrayIndex, unsigned int mipLevel) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuGraphicsSubResourceGetMappedArray
    _check_or_init_driver()
    if __cuGraphicsSubResourceGetMappedArray == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphicsSubResourceGetMappedArray is not found")
    return (<CUresult (*)(CUarray*, CUgraphicsResource, unsigned int, unsigned int) noexcept nogil>__cuGraphicsSubResourceGetMappedArray)(
        pArray, resource, arrayIndex, mipLevel)


cdef CUresult _cuGraphicsResourceGetMappedMipmappedArray(CUmipmappedArray* pMipmappedArray, CUgraphicsResource resource) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuGraphicsResourceGetMappedMipmappedArray
    _check_or_init_driver()
    if __cuGraphicsResourceGetMappedMipmappedArray == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphicsResourceGetMappedMipmappedArray is not found")
    return (<CUresult (*)(CUmipmappedArray*, CUgraphicsResource) noexcept nogil>__cuGraphicsResourceGetMappedMipmappedArray)(
        pMipmappedArray, resource)


cdef CUresult _cuGraphicsResourceGetMappedPointer_v2(CUdeviceptr* pDevPtr, size_t* pSize, CUgraphicsResource resource) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuGraphicsResourceGetMappedPointer_v2
    _check_or_init_driver()
    if __cuGraphicsResourceGetMappedPointer_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphicsResourceGetMappedPointer_v2 is not found")
    return (<CUresult (*)(CUdeviceptr*, size_t*, CUgraphicsResource) noexcept nogil>__cuGraphicsResourceGetMappedPointer_v2)(
        pDevPtr, pSize, resource)


cdef CUresult _cuGraphicsResourceSetMapFlags_v2(CUgraphicsResource resource, unsigned int flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuGraphicsResourceSetMapFlags_v2
    _check_or_init_driver()
    if __cuGraphicsResourceSetMapFlags_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphicsResourceSetMapFlags_v2 is not found")
    return (<CUresult (*)(CUgraphicsResource, unsigned int) noexcept nogil>__cuGraphicsResourceSetMapFlags_v2)(
        resource, flags)


cdef CUresult _cuGraphicsMapResources(unsigned int count, CUgraphicsResource* resources, CUstream hStream) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuGraphicsMapResources
    _check_or_init_driver()
    if __cuGraphicsMapResources == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphicsMapResources is not found")
    return (<CUresult (*)(unsigned int, CUgraphicsResource*, CUstream) noexcept nogil>__cuGraphicsMapResources)(
        count, resources, hStream)


cdef CUresult _cuGraphicsUnmapResources(unsigned int count, CUgraphicsResource* resources, CUstream hStream) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuGraphicsUnmapResources
    _check_or_init_driver()
    if __cuGraphicsUnmapResources == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphicsUnmapResources is not found")
    return (<CUresult (*)(unsigned int, CUgraphicsResource*, CUstream) noexcept nogil>__cuGraphicsUnmapResources)(
        count, resources, hStream)


cdef CUresult _cuGetProcAddress_v2(const char* symbol, void** pfn, int cudaVersion, cuuint64_t flags, CUdriverProcAddressQueryResult* symbolStatus) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuGetProcAddress_v2
    _check_or_init_driver()
    if __cuGetProcAddress_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGetProcAddress_v2 is not found")
    return (<CUresult (*)(const char*, void**, int, cuuint64_t, CUdriverProcAddressQueryResult*) noexcept nogil>__cuGetProcAddress_v2)(
        symbol, pfn, cudaVersion, flags, symbolStatus)


cdef CUresult _cuCoredumpGetAttribute(CUcoredumpSettings attrib, void* value, size_t* size) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuCoredumpGetAttribute
    _check_or_init_driver()
    if __cuCoredumpGetAttribute == NULL:
        with gil:
            raise FunctionNotFoundError("function cuCoredumpGetAttribute is not found")
    return (<CUresult (*)(CUcoredumpSettings, void*, size_t*) noexcept nogil>__cuCoredumpGetAttribute)(
        attrib, value, size)


cdef CUresult _cuCoredumpGetAttributeGlobal(CUcoredumpSettings attrib, void* value, size_t* size) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuCoredumpGetAttributeGlobal
    _check_or_init_driver()
    if __cuCoredumpGetAttributeGlobal == NULL:
        with gil:
            raise FunctionNotFoundError("function cuCoredumpGetAttributeGlobal is not found")
    return (<CUresult (*)(CUcoredumpSettings, void*, size_t*) noexcept nogil>__cuCoredumpGetAttributeGlobal)(
        attrib, value, size)


cdef CUresult _cuCoredumpSetAttribute(CUcoredumpSettings attrib, void* value, size_t* size) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuCoredumpSetAttribute
    _check_or_init_driver()
    if __cuCoredumpSetAttribute == NULL:
        with gil:
            raise FunctionNotFoundError("function cuCoredumpSetAttribute is not found")
    return (<CUresult (*)(CUcoredumpSettings, void*, size_t*) noexcept nogil>__cuCoredumpSetAttribute)(
        attrib, value, size)


cdef CUresult _cuCoredumpSetAttributeGlobal(CUcoredumpSettings attrib, void* value, size_t* size) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuCoredumpSetAttributeGlobal
    _check_or_init_driver()
    if __cuCoredumpSetAttributeGlobal == NULL:
        with gil:
            raise FunctionNotFoundError("function cuCoredumpSetAttributeGlobal is not found")
    return (<CUresult (*)(CUcoredumpSettings, void*, size_t*) noexcept nogil>__cuCoredumpSetAttributeGlobal)(
        attrib, value, size)


cdef CUresult _cuGetExportTable(const void** ppExportTable, const CUuuid* pExportTableId) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuGetExportTable
    _check_or_init_driver()
    if __cuGetExportTable == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGetExportTable is not found")
    return (<CUresult (*)(const void**, const CUuuid*) noexcept nogil>__cuGetExportTable)(
        ppExportTable, pExportTableId)


cdef CUresult _cuGreenCtxCreate(CUgreenCtx* phCtx, CUdevResourceDesc desc, CUdevice dev, unsigned int flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuGreenCtxCreate
    _check_or_init_driver()
    if __cuGreenCtxCreate == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGreenCtxCreate is not found")
    return (<CUresult (*)(CUgreenCtx*, CUdevResourceDesc, CUdevice, unsigned int) noexcept nogil>__cuGreenCtxCreate)(
        phCtx, desc, dev, flags)


cdef CUresult _cuGreenCtxDestroy(CUgreenCtx hCtx) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuGreenCtxDestroy
    _check_or_init_driver()
    if __cuGreenCtxDestroy == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGreenCtxDestroy is not found")
    return (<CUresult (*)(CUgreenCtx) noexcept nogil>__cuGreenCtxDestroy)(
        hCtx)


cdef CUresult _cuCtxFromGreenCtx(CUcontext* pContext, CUgreenCtx hCtx) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuCtxFromGreenCtx
    _check_or_init_driver()
    if __cuCtxFromGreenCtx == NULL:
        with gil:
            raise FunctionNotFoundError("function cuCtxFromGreenCtx is not found")
    return (<CUresult (*)(CUcontext*, CUgreenCtx) noexcept nogil>__cuCtxFromGreenCtx)(
        pContext, hCtx)


cdef CUresult _cuDeviceGetDevResource(CUdevice device, CUdevResource* resource, CUdevResourceType type) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuDeviceGetDevResource
    _check_or_init_driver()
    if __cuDeviceGetDevResource == NULL:
        with gil:
            raise FunctionNotFoundError("function cuDeviceGetDevResource is not found")
    return (<CUresult (*)(CUdevice, CUdevResource*, CUdevResourceType) noexcept nogil>__cuDeviceGetDevResource)(
        device, resource, type)


cdef CUresult _cuCtxGetDevResource(CUcontext hCtx, CUdevResource* resource, CUdevResourceType type) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuCtxGetDevResource
    _check_or_init_driver()
    if __cuCtxGetDevResource == NULL:
        with gil:
            raise FunctionNotFoundError("function cuCtxGetDevResource is not found")
    return (<CUresult (*)(CUcontext, CUdevResource*, CUdevResourceType) noexcept nogil>__cuCtxGetDevResource)(
        hCtx, resource, type)


cdef CUresult _cuGreenCtxGetDevResource(CUgreenCtx hCtx, CUdevResource* resource, CUdevResourceType type) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuGreenCtxGetDevResource
    _check_or_init_driver()
    if __cuGreenCtxGetDevResource == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGreenCtxGetDevResource is not found")
    return (<CUresult (*)(CUgreenCtx, CUdevResource*, CUdevResourceType) noexcept nogil>__cuGreenCtxGetDevResource)(
        hCtx, resource, type)


cdef CUresult _cuDevSmResourceSplitByCount(CUdevResource* result, unsigned int* nbGroups, const CUdevResource* input, CUdevResource* remainder, unsigned int flags, unsigned int minCount) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuDevSmResourceSplitByCount
    _check_or_init_driver()
    if __cuDevSmResourceSplitByCount == NULL:
        with gil:
            raise FunctionNotFoundError("function cuDevSmResourceSplitByCount is not found")
    return (<CUresult (*)(CUdevResource*, unsigned int*, const CUdevResource*, CUdevResource*, unsigned int, unsigned int) noexcept nogil>__cuDevSmResourceSplitByCount)(
        result, nbGroups, input, remainder, flags, minCount)


cdef CUresult _cuDevResourceGenerateDesc(CUdevResourceDesc* phDesc, CUdevResource* resources, unsigned int nbResources) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuDevResourceGenerateDesc
    _check_or_init_driver()
    if __cuDevResourceGenerateDesc == NULL:
        with gil:
            raise FunctionNotFoundError("function cuDevResourceGenerateDesc is not found")
    return (<CUresult (*)(CUdevResourceDesc*, CUdevResource*, unsigned int) noexcept nogil>__cuDevResourceGenerateDesc)(
        phDesc, resources, nbResources)


cdef CUresult _cuGreenCtxRecordEvent(CUgreenCtx hCtx, CUevent hEvent) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuGreenCtxRecordEvent
    _check_or_init_driver()
    if __cuGreenCtxRecordEvent == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGreenCtxRecordEvent is not found")
    return (<CUresult (*)(CUgreenCtx, CUevent) noexcept nogil>__cuGreenCtxRecordEvent)(
        hCtx, hEvent)


cdef CUresult _cuGreenCtxWaitEvent(CUgreenCtx hCtx, CUevent hEvent) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuGreenCtxWaitEvent
    _check_or_init_driver()
    if __cuGreenCtxWaitEvent == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGreenCtxWaitEvent is not found")
    return (<CUresult (*)(CUgreenCtx, CUevent) noexcept nogil>__cuGreenCtxWaitEvent)(
        hCtx, hEvent)


cdef CUresult _cuStreamGetGreenCtx(CUstream hStream, CUgreenCtx* phCtx) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuStreamGetGreenCtx
    _check_or_init_driver()
    if __cuStreamGetGreenCtx == NULL:
        with gil:
            raise FunctionNotFoundError("function cuStreamGetGreenCtx is not found")
    return (<CUresult (*)(CUstream, CUgreenCtx*) noexcept nogil>__cuStreamGetGreenCtx)(
        hStream, phCtx)


cdef CUresult _cuGreenCtxStreamCreate(CUstream* phStream, CUgreenCtx greenCtx, unsigned int flags, int priority) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuGreenCtxStreamCreate
    _check_or_init_driver()
    if __cuGreenCtxStreamCreate == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGreenCtxStreamCreate is not found")
    return (<CUresult (*)(CUstream*, CUgreenCtx, unsigned int, int) noexcept nogil>__cuGreenCtxStreamCreate)(
        phStream, greenCtx, flags, priority)


cdef CUresult _cuLogsRegisterCallback(CUlogsCallback callbackFunc, void* userData, CUlogsCallbackHandle* callback_out) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuLogsRegisterCallback
    _check_or_init_driver()
    if __cuLogsRegisterCallback == NULL:
        with gil:
            raise FunctionNotFoundError("function cuLogsRegisterCallback is not found")
    return (<CUresult (*)(CUlogsCallback, void*, CUlogsCallbackHandle*) noexcept nogil>__cuLogsRegisterCallback)(
        callbackFunc, userData, callback_out)


cdef CUresult _cuLogsUnregisterCallback(CUlogsCallbackHandle callback) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuLogsUnregisterCallback
    _check_or_init_driver()
    if __cuLogsUnregisterCallback == NULL:
        with gil:
            raise FunctionNotFoundError("function cuLogsUnregisterCallback is not found")
    return (<CUresult (*)(CUlogsCallbackHandle) noexcept nogil>__cuLogsUnregisterCallback)(
        callback)


cdef CUresult _cuLogsCurrent(CUlogIterator* iterator_out, unsigned int flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuLogsCurrent
    _check_or_init_driver()
    if __cuLogsCurrent == NULL:
        with gil:
            raise FunctionNotFoundError("function cuLogsCurrent is not found")
    return (<CUresult (*)(CUlogIterator*, unsigned int) noexcept nogil>__cuLogsCurrent)(
        iterator_out, flags)


cdef CUresult _cuLogsDumpToFile(CUlogIterator* iterator, const char* pathToFile, unsigned int flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuLogsDumpToFile
    _check_or_init_driver()
    if __cuLogsDumpToFile == NULL:
        with gil:
            raise FunctionNotFoundError("function cuLogsDumpToFile is not found")
    return (<CUresult (*)(CUlogIterator*, const char*, unsigned int) noexcept nogil>__cuLogsDumpToFile)(
        iterator, pathToFile, flags)


cdef CUresult _cuLogsDumpToMemory(CUlogIterator* iterator, char* buffer, size_t* size, unsigned int flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuLogsDumpToMemory
    _check_or_init_driver()
    if __cuLogsDumpToMemory == NULL:
        with gil:
            raise FunctionNotFoundError("function cuLogsDumpToMemory is not found")
    return (<CUresult (*)(CUlogIterator*, char*, size_t*, unsigned int) noexcept nogil>__cuLogsDumpToMemory)(
        iterator, buffer, size, flags)


cdef CUresult _cuCheckpointProcessGetRestoreThreadId(int pid, int* tid) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuCheckpointProcessGetRestoreThreadId
    _check_or_init_driver()
    if __cuCheckpointProcessGetRestoreThreadId == NULL:
        with gil:
            raise FunctionNotFoundError("function cuCheckpointProcessGetRestoreThreadId is not found")
    return (<CUresult (*)(int, int*) noexcept nogil>__cuCheckpointProcessGetRestoreThreadId)(
        pid, tid)


cdef CUresult _cuCheckpointProcessGetState(int pid, CUprocessState* state) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuCheckpointProcessGetState
    _check_or_init_driver()
    if __cuCheckpointProcessGetState == NULL:
        with gil:
            raise FunctionNotFoundError("function cuCheckpointProcessGetState is not found")
    return (<CUresult (*)(int, CUprocessState*) noexcept nogil>__cuCheckpointProcessGetState)(
        pid, state)


cdef CUresult _cuCheckpointProcessLock(int pid, CUcheckpointLockArgs* args) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuCheckpointProcessLock
    _check_or_init_driver()
    if __cuCheckpointProcessLock == NULL:
        with gil:
            raise FunctionNotFoundError("function cuCheckpointProcessLock is not found")
    return (<CUresult (*)(int, CUcheckpointLockArgs*) noexcept nogil>__cuCheckpointProcessLock)(
        pid, args)


cdef CUresult _cuCheckpointProcessCheckpoint(int pid, CUcheckpointCheckpointArgs* args) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuCheckpointProcessCheckpoint
    _check_or_init_driver()
    if __cuCheckpointProcessCheckpoint == NULL:
        with gil:
            raise FunctionNotFoundError("function cuCheckpointProcessCheckpoint is not found")
    return (<CUresult (*)(int, CUcheckpointCheckpointArgs*) noexcept nogil>__cuCheckpointProcessCheckpoint)(
        pid, args)


cdef CUresult _cuCheckpointProcessRestore(int pid, CUcheckpointRestoreArgs* args) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuCheckpointProcessRestore
    _check_or_init_driver()
    if __cuCheckpointProcessRestore == NULL:
        with gil:
            raise FunctionNotFoundError("function cuCheckpointProcessRestore is not found")
    return (<CUresult (*)(int, CUcheckpointRestoreArgs*) noexcept nogil>__cuCheckpointProcessRestore)(
        pid, args)


cdef CUresult _cuCheckpointProcessUnlock(int pid, CUcheckpointUnlockArgs* args) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuCheckpointProcessUnlock
    _check_or_init_driver()
    if __cuCheckpointProcessUnlock == NULL:
        with gil:
            raise FunctionNotFoundError("function cuCheckpointProcessUnlock is not found")
    return (<CUresult (*)(int, CUcheckpointUnlockArgs*) noexcept nogil>__cuCheckpointProcessUnlock)(
        pid, args)


cdef CUresult _cuGraphicsEGLRegisterImage(CUgraphicsResource* pCudaResource, EGLImageKHR image, unsigned int flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuGraphicsEGLRegisterImage
    _check_or_init_driver()
    if __cuGraphicsEGLRegisterImage == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphicsEGLRegisterImage is not found")
    return (<CUresult (*)(CUgraphicsResource*, EGLImageKHR, unsigned int) noexcept nogil>__cuGraphicsEGLRegisterImage)(
        pCudaResource, image, flags)


cdef CUresult _cuEGLStreamConsumerConnect(CUeglStreamConnection* conn, EGLStreamKHR stream) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuEGLStreamConsumerConnect
    _check_or_init_driver()
    if __cuEGLStreamConsumerConnect == NULL:
        with gil:
            raise FunctionNotFoundError("function cuEGLStreamConsumerConnect is not found")
    return (<CUresult (*)(CUeglStreamConnection*, EGLStreamKHR) noexcept nogil>__cuEGLStreamConsumerConnect)(
        conn, stream)


cdef CUresult _cuEGLStreamConsumerConnectWithFlags(CUeglStreamConnection* conn, EGLStreamKHR stream, unsigned int flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuEGLStreamConsumerConnectWithFlags
    _check_or_init_driver()
    if __cuEGLStreamConsumerConnectWithFlags == NULL:
        with gil:
            raise FunctionNotFoundError("function cuEGLStreamConsumerConnectWithFlags is not found")
    return (<CUresult (*)(CUeglStreamConnection*, EGLStreamKHR, unsigned int) noexcept nogil>__cuEGLStreamConsumerConnectWithFlags)(
        conn, stream, flags)


cdef CUresult _cuEGLStreamConsumerDisconnect(CUeglStreamConnection* conn) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuEGLStreamConsumerDisconnect
    _check_or_init_driver()
    if __cuEGLStreamConsumerDisconnect == NULL:
        with gil:
            raise FunctionNotFoundError("function cuEGLStreamConsumerDisconnect is not found")
    return (<CUresult (*)(CUeglStreamConnection*) noexcept nogil>__cuEGLStreamConsumerDisconnect)(
        conn)


cdef CUresult _cuEGLStreamConsumerAcquireFrame(CUeglStreamConnection* conn, CUgraphicsResource* pCudaResource, CUstream* pStream, unsigned int timeout) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuEGLStreamConsumerAcquireFrame
    _check_or_init_driver()
    if __cuEGLStreamConsumerAcquireFrame == NULL:
        with gil:
            raise FunctionNotFoundError("function cuEGLStreamConsumerAcquireFrame is not found")
    return (<CUresult (*)(CUeglStreamConnection*, CUgraphicsResource*, CUstream*, unsigned int) noexcept nogil>__cuEGLStreamConsumerAcquireFrame)(
        conn, pCudaResource, pStream, timeout)


cdef CUresult _cuEGLStreamConsumerReleaseFrame(CUeglStreamConnection* conn, CUgraphicsResource pCudaResource, CUstream* pStream) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuEGLStreamConsumerReleaseFrame
    _check_or_init_driver()
    if __cuEGLStreamConsumerReleaseFrame == NULL:
        with gil:
            raise FunctionNotFoundError("function cuEGLStreamConsumerReleaseFrame is not found")
    return (<CUresult (*)(CUeglStreamConnection*, CUgraphicsResource, CUstream*) noexcept nogil>__cuEGLStreamConsumerReleaseFrame)(
        conn, pCudaResource, pStream)


cdef CUresult _cuEGLStreamProducerConnect(CUeglStreamConnection* conn, EGLStreamKHR stream, EGLint width, EGLint height) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuEGLStreamProducerConnect
    _check_or_init_driver()
    if __cuEGLStreamProducerConnect == NULL:
        with gil:
            raise FunctionNotFoundError("function cuEGLStreamProducerConnect is not found")
    return (<CUresult (*)(CUeglStreamConnection*, EGLStreamKHR, EGLint, EGLint) noexcept nogil>__cuEGLStreamProducerConnect)(
        conn, stream, width, height)


cdef CUresult _cuEGLStreamProducerDisconnect(CUeglStreamConnection* conn) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuEGLStreamProducerDisconnect
    _check_or_init_driver()
    if __cuEGLStreamProducerDisconnect == NULL:
        with gil:
            raise FunctionNotFoundError("function cuEGLStreamProducerDisconnect is not found")
    return (<CUresult (*)(CUeglStreamConnection*) noexcept nogil>__cuEGLStreamProducerDisconnect)(
        conn)


cdef CUresult _cuEGLStreamProducerPresentFrame(CUeglStreamConnection* conn, CUeglFrame eglframe, CUstream* pStream) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuEGLStreamProducerPresentFrame
    _check_or_init_driver()
    if __cuEGLStreamProducerPresentFrame == NULL:
        with gil:
            raise FunctionNotFoundError("function cuEGLStreamProducerPresentFrame is not found")
    return (<CUresult (*)(CUeglStreamConnection*, CUeglFrame, CUstream*) noexcept nogil>__cuEGLStreamProducerPresentFrame)(
        conn, eglframe, pStream)


cdef CUresult _cuEGLStreamProducerReturnFrame(CUeglStreamConnection* conn, CUeglFrame* eglframe, CUstream* pStream) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuEGLStreamProducerReturnFrame
    _check_or_init_driver()
    if __cuEGLStreamProducerReturnFrame == NULL:
        with gil:
            raise FunctionNotFoundError("function cuEGLStreamProducerReturnFrame is not found")
    return (<CUresult (*)(CUeglStreamConnection*, CUeglFrame*, CUstream*) noexcept nogil>__cuEGLStreamProducerReturnFrame)(
        conn, eglframe, pStream)


cdef CUresult _cuGraphicsResourceGetMappedEglFrame(CUeglFrame* eglFrame, CUgraphicsResource resource, unsigned int index, unsigned int mipLevel) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuGraphicsResourceGetMappedEglFrame
    _check_or_init_driver()
    if __cuGraphicsResourceGetMappedEglFrame == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphicsResourceGetMappedEglFrame is not found")
    return (<CUresult (*)(CUeglFrame*, CUgraphicsResource, unsigned int, unsigned int) noexcept nogil>__cuGraphicsResourceGetMappedEglFrame)(
        eglFrame, resource, index, mipLevel)


cdef CUresult _cuEventCreateFromEGLSync(CUevent* phEvent, EGLSyncKHR eglSync, unsigned int flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuEventCreateFromEGLSync
    _check_or_init_driver()
    if __cuEventCreateFromEGLSync == NULL:
        with gil:
            raise FunctionNotFoundError("function cuEventCreateFromEGLSync is not found")
    return (<CUresult (*)(CUevent*, EGLSyncKHR, unsigned int) noexcept nogil>__cuEventCreateFromEGLSync)(
        phEvent, eglSync, flags)


cdef CUresult _cuGraphicsGLRegisterBuffer(CUgraphicsResource* pCudaResource, GLuint buffer, unsigned int Flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuGraphicsGLRegisterBuffer
    _check_or_init_driver()
    if __cuGraphicsGLRegisterBuffer == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphicsGLRegisterBuffer is not found")
    return (<CUresult (*)(CUgraphicsResource*, GLuint, unsigned int) noexcept nogil>__cuGraphicsGLRegisterBuffer)(
        pCudaResource, buffer, Flags)


cdef CUresult _cuGraphicsGLRegisterImage(CUgraphicsResource* pCudaResource, GLuint image, GLenum target, unsigned int Flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuGraphicsGLRegisterImage
    _check_or_init_driver()
    if __cuGraphicsGLRegisterImage == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphicsGLRegisterImage is not found")
    return (<CUresult (*)(CUgraphicsResource*, GLuint, GLenum, unsigned int) noexcept nogil>__cuGraphicsGLRegisterImage)(
        pCudaResource, image, target, Flags)


cdef CUresult _cuGLGetDevices_v2(unsigned int* pCudaDeviceCount, CUdevice* pCudaDevices, unsigned int cudaDeviceCount, CUGLDeviceList deviceList) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuGLGetDevices_v2
    _check_or_init_driver()
    if __cuGLGetDevices_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGLGetDevices_v2 is not found")
    return (<CUresult (*)(unsigned int*, CUdevice*, unsigned int, CUGLDeviceList) noexcept nogil>__cuGLGetDevices_v2)(
        pCudaDeviceCount, pCudaDevices, cudaDeviceCount, deviceList)


cdef CUresult _cuGLCtxCreate_v2(CUcontext* pCtx, unsigned int Flags, CUdevice device) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuGLCtxCreate_v2
    _check_or_init_driver()
    if __cuGLCtxCreate_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGLCtxCreate_v2 is not found")
    return (<CUresult (*)(CUcontext*, unsigned int, CUdevice) noexcept nogil>__cuGLCtxCreate_v2)(
        pCtx, Flags, device)


cdef CUresult _cuGLInit() except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuGLInit
    _check_or_init_driver()
    if __cuGLInit == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGLInit is not found")
    return (<CUresult (*)() noexcept nogil>__cuGLInit)(
        )


cdef CUresult _cuGLRegisterBufferObject(GLuint buffer) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuGLRegisterBufferObject
    _check_or_init_driver()
    if __cuGLRegisterBufferObject == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGLRegisterBufferObject is not found")
    return (<CUresult (*)(GLuint) noexcept nogil>__cuGLRegisterBufferObject)(
        buffer)


cdef CUresult _cuGLMapBufferObject_v2(CUdeviceptr* dptr, size_t* size, GLuint buffer) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuGLMapBufferObject_v2
    _check_or_init_driver()
    if __cuGLMapBufferObject_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGLMapBufferObject_v2 is not found")
    return (<CUresult (*)(CUdeviceptr*, size_t*, GLuint) noexcept nogil>__cuGLMapBufferObject_v2)(
        dptr, size, buffer)


cdef CUresult _cuGLUnmapBufferObject(GLuint buffer) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuGLUnmapBufferObject
    _check_or_init_driver()
    if __cuGLUnmapBufferObject == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGLUnmapBufferObject is not found")
    return (<CUresult (*)(GLuint) noexcept nogil>__cuGLUnmapBufferObject)(
        buffer)


cdef CUresult _cuGLUnregisterBufferObject(GLuint buffer) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuGLUnregisterBufferObject
    _check_or_init_driver()
    if __cuGLUnregisterBufferObject == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGLUnregisterBufferObject is not found")
    return (<CUresult (*)(GLuint) noexcept nogil>__cuGLUnregisterBufferObject)(
        buffer)


cdef CUresult _cuGLSetBufferObjectMapFlags(GLuint buffer, unsigned int Flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuGLSetBufferObjectMapFlags
    _check_or_init_driver()
    if __cuGLSetBufferObjectMapFlags == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGLSetBufferObjectMapFlags is not found")
    return (<CUresult (*)(GLuint, unsigned int) noexcept nogil>__cuGLSetBufferObjectMapFlags)(
        buffer, Flags)


cdef CUresult _cuGLMapBufferObjectAsync_v2(CUdeviceptr* dptr, size_t* size, GLuint buffer, CUstream hStream) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuGLMapBufferObjectAsync_v2
    _check_or_init_driver()
    if __cuGLMapBufferObjectAsync_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGLMapBufferObjectAsync_v2 is not found")
    return (<CUresult (*)(CUdeviceptr*, size_t*, GLuint, CUstream) noexcept nogil>__cuGLMapBufferObjectAsync_v2)(
        dptr, size, buffer, hStream)


cdef CUresult _cuGLUnmapBufferObjectAsync(GLuint buffer, CUstream hStream) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuGLUnmapBufferObjectAsync
    _check_or_init_driver()
    if __cuGLUnmapBufferObjectAsync == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGLUnmapBufferObjectAsync is not found")
    return (<CUresult (*)(GLuint, CUstream) noexcept nogil>__cuGLUnmapBufferObjectAsync)(
        buffer, hStream)


cdef CUresult _cuProfilerInitialize(const char* configFile, const char* outputFile, CUoutput_mode outputMode) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuProfilerInitialize
    _check_or_init_driver()
    if __cuProfilerInitialize == NULL:
        with gil:
            raise FunctionNotFoundError("function cuProfilerInitialize is not found")
    return (<CUresult (*)(const char*, const char*, CUoutput_mode) noexcept nogil>__cuProfilerInitialize)(
        configFile, outputFile, outputMode)


cdef CUresult _cuProfilerStart() except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuProfilerStart
    _check_or_init_driver()
    if __cuProfilerStart == NULL:
        with gil:
            raise FunctionNotFoundError("function cuProfilerStart is not found")
    return (<CUresult (*)() noexcept nogil>__cuProfilerStart)(
        )


cdef CUresult _cuProfilerStop() except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuProfilerStop
    _check_or_init_driver()
    if __cuProfilerStop == NULL:
        with gil:
            raise FunctionNotFoundError("function cuProfilerStop is not found")
    return (<CUresult (*)() noexcept nogil>__cuProfilerStop)(
        )


cdef CUresult _cuVDPAUGetDevice(CUdevice* pDevice, VdpDevice vdpDevice, VdpGetProcAddress* vdpGetProcAddress) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuVDPAUGetDevice
    _check_or_init_driver()
    if __cuVDPAUGetDevice == NULL:
        with gil:
            raise FunctionNotFoundError("function cuVDPAUGetDevice is not found")
    return (<CUresult (*)(CUdevice*, VdpDevice, VdpGetProcAddress*) noexcept nogil>__cuVDPAUGetDevice)(
        pDevice, vdpDevice, vdpGetProcAddress)


cdef CUresult _cuVDPAUCtxCreate_v2(CUcontext* pCtx, unsigned int flags, CUdevice device, VdpDevice vdpDevice, VdpGetProcAddress* vdpGetProcAddress) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuVDPAUCtxCreate_v2
    _check_or_init_driver()
    if __cuVDPAUCtxCreate_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuVDPAUCtxCreate_v2 is not found")
    return (<CUresult (*)(CUcontext*, unsigned int, CUdevice, VdpDevice, VdpGetProcAddress*) noexcept nogil>__cuVDPAUCtxCreate_v2)(
        pCtx, flags, device, vdpDevice, vdpGetProcAddress)


cdef CUresult _cuGraphicsVDPAURegisterVideoSurface(CUgraphicsResource* pCudaResource, VdpVideoSurface vdpSurface, unsigned int flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuGraphicsVDPAURegisterVideoSurface
    _check_or_init_driver()
    if __cuGraphicsVDPAURegisterVideoSurface == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphicsVDPAURegisterVideoSurface is not found")
    return (<CUresult (*)(CUgraphicsResource*, VdpVideoSurface, unsigned int) noexcept nogil>__cuGraphicsVDPAURegisterVideoSurface)(
        pCudaResource, vdpSurface, flags)


cdef CUresult _cuGraphicsVDPAURegisterOutputSurface(CUgraphicsResource* pCudaResource, VdpOutputSurface vdpSurface, unsigned int flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuGraphicsVDPAURegisterOutputSurface
    _check_or_init_driver()
    if __cuGraphicsVDPAURegisterOutputSurface == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphicsVDPAURegisterOutputSurface is not found")
    return (<CUresult (*)(CUgraphicsResource*, VdpOutputSurface, unsigned int) noexcept nogil>__cuGraphicsVDPAURegisterOutputSurface)(
        pCudaResource, vdpSurface, flags)


cdef CUresult _cuDeviceGetHostAtomicCapabilities(unsigned int* capabilities, const CUatomicOperation* operations, unsigned int count, CUdevice dev) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuDeviceGetHostAtomicCapabilities
    _check_or_init_driver()
    if __cuDeviceGetHostAtomicCapabilities == NULL:
        with gil:
            raise FunctionNotFoundError("function cuDeviceGetHostAtomicCapabilities is not found")
    return (<CUresult (*)(unsigned int*, const CUatomicOperation*, unsigned int, CUdevice) noexcept nogil>__cuDeviceGetHostAtomicCapabilities)(
        capabilities, operations, count, dev)


cdef CUresult _cuCtxGetDevice_v2(CUdevice* device, CUcontext ctx) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuCtxGetDevice_v2
    _check_or_init_driver()
    if __cuCtxGetDevice_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuCtxGetDevice_v2 is not found")
    return (<CUresult (*)(CUdevice*, CUcontext) noexcept nogil>__cuCtxGetDevice_v2)(
        device, ctx)


cdef CUresult _cuCtxSynchronize_v2(CUcontext ctx) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuCtxSynchronize_v2
    _check_or_init_driver()
    if __cuCtxSynchronize_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuCtxSynchronize_v2 is not found")
    return (<CUresult (*)(CUcontext) noexcept nogil>__cuCtxSynchronize_v2)(
        ctx)


cdef CUresult _cuMemcpyBatchAsync_v2(CUdeviceptr* dsts, CUdeviceptr* srcs, size_t* sizes, size_t count, CUmemcpyAttributes* attrs, size_t* attrsIdxs, size_t numAttrs, CUstream hStream) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuMemcpyBatchAsync_v2
    _check_or_init_driver()
    if __cuMemcpyBatchAsync_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemcpyBatchAsync_v2 is not found")
    return (<CUresult (*)(CUdeviceptr*, CUdeviceptr*, size_t*, size_t, CUmemcpyAttributes*, size_t*, size_t, CUstream) noexcept nogil>__cuMemcpyBatchAsync_v2)(
        dsts, srcs, sizes, count, attrs, attrsIdxs, numAttrs, hStream)


cdef CUresult _cuMemcpy3DBatchAsync_v2(size_t numOps, CUDA_MEMCPY3D_BATCH_OP* opList, unsigned long long flags, CUstream hStream) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuMemcpy3DBatchAsync_v2
    _check_or_init_driver()
    if __cuMemcpy3DBatchAsync_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemcpy3DBatchAsync_v2 is not found")
    return (<CUresult (*)(size_t, CUDA_MEMCPY3D_BATCH_OP*, unsigned long long, CUstream) noexcept nogil>__cuMemcpy3DBatchAsync_v2)(
        numOps, opList, flags, hStream)


cdef CUresult _cuMemGetDefaultMemPool(CUmemoryPool* pool_out, CUmemLocation* location, CUmemAllocationType type) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuMemGetDefaultMemPool
    _check_or_init_driver()
    if __cuMemGetDefaultMemPool == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemGetDefaultMemPool is not found")
    return (<CUresult (*)(CUmemoryPool*, CUmemLocation*, CUmemAllocationType) noexcept nogil>__cuMemGetDefaultMemPool)(
        pool_out, location, type)


cdef CUresult _cuMemGetMemPool(CUmemoryPool* pool, CUmemLocation* location, CUmemAllocationType type) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuMemGetMemPool
    _check_or_init_driver()
    if __cuMemGetMemPool == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemGetMemPool is not found")
    return (<CUresult (*)(CUmemoryPool*, CUmemLocation*, CUmemAllocationType) noexcept nogil>__cuMemGetMemPool)(
        pool, location, type)


cdef CUresult _cuMemSetMemPool(CUmemLocation* location, CUmemAllocationType type, CUmemoryPool pool) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuMemSetMemPool
    _check_or_init_driver()
    if __cuMemSetMemPool == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemSetMemPool is not found")
    return (<CUresult (*)(CUmemLocation*, CUmemAllocationType, CUmemoryPool) noexcept nogil>__cuMemSetMemPool)(
        location, type, pool)


cdef CUresult _cuMemPrefetchBatchAsync(CUdeviceptr* dptrs, size_t* sizes, size_t count, CUmemLocation* prefetchLocs, size_t* prefetchLocIdxs, size_t numPrefetchLocs, unsigned long long flags, CUstream hStream) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuMemPrefetchBatchAsync
    _check_or_init_driver()
    if __cuMemPrefetchBatchAsync == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemPrefetchBatchAsync is not found")
    return (<CUresult (*)(CUdeviceptr*, size_t*, size_t, CUmemLocation*, size_t*, size_t, unsigned long long, CUstream) noexcept nogil>__cuMemPrefetchBatchAsync)(
        dptrs, sizes, count, prefetchLocs, prefetchLocIdxs, numPrefetchLocs, flags, hStream)


cdef CUresult _cuMemDiscardBatchAsync(CUdeviceptr* dptrs, size_t* sizes, size_t count, unsigned long long flags, CUstream hStream) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuMemDiscardBatchAsync
    _check_or_init_driver()
    if __cuMemDiscardBatchAsync == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemDiscardBatchAsync is not found")
    return (<CUresult (*)(CUdeviceptr*, size_t*, size_t, unsigned long long, CUstream) noexcept nogil>__cuMemDiscardBatchAsync)(
        dptrs, sizes, count, flags, hStream)


cdef CUresult _cuMemDiscardAndPrefetchBatchAsync(CUdeviceptr* dptrs, size_t* sizes, size_t count, CUmemLocation* prefetchLocs, size_t* prefetchLocIdxs, size_t numPrefetchLocs, unsigned long long flags, CUstream hStream) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuMemDiscardAndPrefetchBatchAsync
    _check_or_init_driver()
    if __cuMemDiscardAndPrefetchBatchAsync == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemDiscardAndPrefetchBatchAsync is not found")
    return (<CUresult (*)(CUdeviceptr*, size_t*, size_t, CUmemLocation*, size_t*, size_t, unsigned long long, CUstream) noexcept nogil>__cuMemDiscardAndPrefetchBatchAsync)(
        dptrs, sizes, count, prefetchLocs, prefetchLocIdxs, numPrefetchLocs, flags, hStream)


cdef CUresult _cuDeviceGetP2PAtomicCapabilities(unsigned int* capabilities, const CUatomicOperation* operations, unsigned int count, CUdevice srcDevice, CUdevice dstDevice) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuDeviceGetP2PAtomicCapabilities
    _check_or_init_driver()
    if __cuDeviceGetP2PAtomicCapabilities == NULL:
        with gil:
            raise FunctionNotFoundError("function cuDeviceGetP2PAtomicCapabilities is not found")
    return (<CUresult (*)(unsigned int*, const CUatomicOperation*, unsigned int, CUdevice, CUdevice) noexcept nogil>__cuDeviceGetP2PAtomicCapabilities)(
        capabilities, operations, count, srcDevice, dstDevice)


cdef CUresult _cuGreenCtxGetId(CUgreenCtx greenCtx, unsigned long long* greenCtxId) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuGreenCtxGetId
    _check_or_init_driver()
    if __cuGreenCtxGetId == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGreenCtxGetId is not found")
    return (<CUresult (*)(CUgreenCtx, unsigned long long*) noexcept nogil>__cuGreenCtxGetId)(
        greenCtx, greenCtxId)


cdef CUresult _cuMulticastBindMem_v2(CUmemGenericAllocationHandle mcHandle, CUdevice dev, size_t mcOffset, CUmemGenericAllocationHandle memHandle, size_t memOffset, size_t size, unsigned long long flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuMulticastBindMem_v2
    _check_or_init_driver()
    if __cuMulticastBindMem_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMulticastBindMem_v2 is not found")
    return (<CUresult (*)(CUmemGenericAllocationHandle, CUdevice, size_t, CUmemGenericAllocationHandle, size_t, size_t, unsigned long long) noexcept nogil>__cuMulticastBindMem_v2)(
        mcHandle, dev, mcOffset, memHandle, memOffset, size, flags)


cdef CUresult _cuMulticastBindAddr_v2(CUmemGenericAllocationHandle mcHandle, CUdevice dev, size_t mcOffset, CUdeviceptr memptr, size_t size, unsigned long long flags) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuMulticastBindAddr_v2
    _check_or_init_driver()
    if __cuMulticastBindAddr_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMulticastBindAddr_v2 is not found")
    return (<CUresult (*)(CUmemGenericAllocationHandle, CUdevice, size_t, CUdeviceptr, size_t, unsigned long long) noexcept nogil>__cuMulticastBindAddr_v2)(
        mcHandle, dev, mcOffset, memptr, size, flags)


cdef CUresult _cuGraphNodeGetContainingGraph(CUgraphNode hNode, CUgraph* phGraph) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuGraphNodeGetContainingGraph
    _check_or_init_driver()
    if __cuGraphNodeGetContainingGraph == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphNodeGetContainingGraph is not found")
    return (<CUresult (*)(CUgraphNode, CUgraph*) noexcept nogil>__cuGraphNodeGetContainingGraph)(
        hNode, phGraph)


cdef CUresult _cuGraphNodeGetLocalId(CUgraphNode hNode, unsigned int* nodeId) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuGraphNodeGetLocalId
    _check_or_init_driver()
    if __cuGraphNodeGetLocalId == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphNodeGetLocalId is not found")
    return (<CUresult (*)(CUgraphNode, unsigned int*) noexcept nogil>__cuGraphNodeGetLocalId)(
        hNode, nodeId)


cdef CUresult _cuGraphNodeGetToolsId(CUgraphNode hNode, unsigned long long* toolsNodeId) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuGraphNodeGetToolsId
    _check_or_init_driver()
    if __cuGraphNodeGetToolsId == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphNodeGetToolsId is not found")
    return (<CUresult (*)(CUgraphNode, unsigned long long*) noexcept nogil>__cuGraphNodeGetToolsId)(
        hNode, toolsNodeId)


cdef CUresult _cuGraphGetId(CUgraph hGraph, unsigned int* graphId) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuGraphGetId
    _check_or_init_driver()
    if __cuGraphGetId == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphGetId is not found")
    return (<CUresult (*)(CUgraph, unsigned int*) noexcept nogil>__cuGraphGetId)(
        hGraph, graphId)


cdef CUresult _cuGraphExecGetId(CUgraphExec hGraphExec, unsigned int* graphId) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuGraphExecGetId
    _check_or_init_driver()
    if __cuGraphExecGetId == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphExecGetId is not found")
    return (<CUresult (*)(CUgraphExec, unsigned int*) noexcept nogil>__cuGraphExecGetId)(
        hGraphExec, graphId)


cdef CUresult _cuDevSmResourceSplit(CUdevResource* result, unsigned int nbGroups, const CUdevResource* input, CUdevResource* remainder, unsigned int flags, CU_DEV_SM_RESOURCE_GROUP_PARAMS* groupParams) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuDevSmResourceSplit
    _check_or_init_driver()
    if __cuDevSmResourceSplit == NULL:
        with gil:
            raise FunctionNotFoundError("function cuDevSmResourceSplit is not found")
    return (<CUresult (*)(CUdevResource*, unsigned int, const CUdevResource*, CUdevResource*, unsigned int, CU_DEV_SM_RESOURCE_GROUP_PARAMS*) noexcept nogil>__cuDevSmResourceSplit)(
        result, nbGroups, input, remainder, flags, groupParams)


cdef CUresult _cuStreamGetDevResource(CUstream hStream, CUdevResource* resource, CUdevResourceType type) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuStreamGetDevResource
    _check_or_init_driver()
    if __cuStreamGetDevResource == NULL:
        with gil:
            raise FunctionNotFoundError("function cuStreamGetDevResource is not found")
    return (<CUresult (*)(CUstream, CUdevResource*, CUdevResourceType) noexcept nogil>__cuStreamGetDevResource)(
        hStream, resource, type)


cdef CUresult _cuKernelGetParamCount(CUkernel kernel, size_t* paramCount) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuKernelGetParamCount
    _check_or_init_driver()
    if __cuKernelGetParamCount == NULL:
        with gil:
            raise FunctionNotFoundError("function cuKernelGetParamCount is not found")
    return (<CUresult (*)(CUkernel, size_t*) noexcept nogil>__cuKernelGetParamCount)(
        kernel, paramCount)


cdef CUresult _cuMemcpyWithAttributesAsync(CUdeviceptr dst, CUdeviceptr src, size_t size, CUmemcpyAttributes* attr, CUstream hStream) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuMemcpyWithAttributesAsync
    _check_or_init_driver()
    if __cuMemcpyWithAttributesAsync == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemcpyWithAttributesAsync is not found")
    return (<CUresult (*)(CUdeviceptr, CUdeviceptr, size_t, CUmemcpyAttributes*, CUstream) noexcept nogil>__cuMemcpyWithAttributesAsync)(
        dst, src, size, attr, hStream)


cdef CUresult _cuMemcpy3DWithAttributesAsync(CUDA_MEMCPY3D_BATCH_OP* op, unsigned long long flags, CUstream hStream) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuMemcpy3DWithAttributesAsync
    _check_or_init_driver()
    if __cuMemcpy3DWithAttributesAsync == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemcpy3DWithAttributesAsync is not found")
    return (<CUresult (*)(CUDA_MEMCPY3D_BATCH_OP*, unsigned long long, CUstream) noexcept nogil>__cuMemcpy3DWithAttributesAsync)(
        op, flags, hStream)


cdef CUresult _cuStreamBeginCaptureToCig(CUstream hStream, CUstreamCigCaptureParams* streamCigCaptureParams) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuStreamBeginCaptureToCig
    _check_or_init_driver()
    if __cuStreamBeginCaptureToCig == NULL:
        with gil:
            raise FunctionNotFoundError("function cuStreamBeginCaptureToCig is not found")
    return (<CUresult (*)(CUstream, CUstreamCigCaptureParams*) noexcept nogil>__cuStreamBeginCaptureToCig)(
        hStream, streamCigCaptureParams)


cdef CUresult _cuStreamEndCaptureToCig(CUstream hStream) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuStreamEndCaptureToCig
    _check_or_init_driver()
    if __cuStreamEndCaptureToCig == NULL:
        with gil:
            raise FunctionNotFoundError("function cuStreamEndCaptureToCig is not found")
    return (<CUresult (*)(CUstream) noexcept nogil>__cuStreamEndCaptureToCig)(
        hStream)


cdef CUresult _cuFuncGetParamCount(CUfunction func, size_t* paramCount) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuFuncGetParamCount
    _check_or_init_driver()
    if __cuFuncGetParamCount == NULL:
        with gil:
            raise FunctionNotFoundError("function cuFuncGetParamCount is not found")
    return (<CUresult (*)(CUfunction, size_t*) noexcept nogil>__cuFuncGetParamCount)(
        func, paramCount)


cdef CUresult _cuLaunchHostFunc_v2(CUstream hStream, CUhostFn fn, void* userData, unsigned int syncMode) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuLaunchHostFunc_v2
    _check_or_init_driver()
    if __cuLaunchHostFunc_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuLaunchHostFunc_v2 is not found")
    return (<CUresult (*)(CUstream, CUhostFn, void*, unsigned int) noexcept nogil>__cuLaunchHostFunc_v2)(
        hStream, fn, userData, syncMode)


cdef CUresult _cuGraphNodeGetParams(CUgraphNode hNode, CUgraphNodeParams* nodeParams) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuGraphNodeGetParams
    _check_or_init_driver()
    if __cuGraphNodeGetParams == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphNodeGetParams is not found")
    return (<CUresult (*)(CUgraphNode, CUgraphNodeParams*) noexcept nogil>__cuGraphNodeGetParams)(
        hNode, nodeParams)


cdef CUresult _cuCoredumpRegisterStartCallback(CUcoredumpStatusCallback callback, void* userData, CUcoredumpCallbackHandle* callbackOut) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuCoredumpRegisterStartCallback
    _check_or_init_driver()
    if __cuCoredumpRegisterStartCallback == NULL:
        with gil:
            raise FunctionNotFoundError("function cuCoredumpRegisterStartCallback is not found")
    return (<CUresult (*)(CUcoredumpStatusCallback, void*, CUcoredumpCallbackHandle*) noexcept nogil>__cuCoredumpRegisterStartCallback)(
        callback, userData, callbackOut)


cdef CUresult _cuCoredumpRegisterCompleteCallback(CUcoredumpStatusCallback callback, void* userData, CUcoredumpCallbackHandle* callbackOut) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuCoredumpRegisterCompleteCallback
    _check_or_init_driver()
    if __cuCoredumpRegisterCompleteCallback == NULL:
        with gil:
            raise FunctionNotFoundError("function cuCoredumpRegisterCompleteCallback is not found")
    return (<CUresult (*)(CUcoredumpStatusCallback, void*, CUcoredumpCallbackHandle*) noexcept nogil>__cuCoredumpRegisterCompleteCallback)(
        callback, userData, callbackOut)


cdef CUresult _cuCoredumpDeregisterStartCallback(CUcoredumpCallbackHandle callback) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuCoredumpDeregisterStartCallback
    _check_or_init_driver()
    if __cuCoredumpDeregisterStartCallback == NULL:
        with gil:
            raise FunctionNotFoundError("function cuCoredumpDeregisterStartCallback is not found")
    return (<CUresult (*)(CUcoredumpCallbackHandle) noexcept nogil>__cuCoredumpDeregisterStartCallback)(
        callback)


cdef CUresult _cuCoredumpDeregisterCompleteCallback(CUcoredumpCallbackHandle callback) except?<CUresult>_CURESULT_INTERNAL_LOADING_ERROR nogil:
    global __cuCoredumpDeregisterCompleteCallback
    _check_or_init_driver()
    if __cuCoredumpDeregisterCompleteCallback == NULL:
        with gil:
            raise FunctionNotFoundError("function cuCoredumpDeregisterCompleteCallback is not found")
    return (<CUresult (*)(CUcoredumpCallbackHandle) noexcept nogil>__cuCoredumpDeregisterCompleteCallback)(
        callback)
