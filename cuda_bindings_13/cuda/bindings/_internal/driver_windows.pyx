# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
#
# This code was automatically generated across versions from 12.9.0 to 13.5.0. Do not modify it directly.
# CYTHON-BINDINGS-GENERATED-DO-NOT-MODIFY-THIS-FILE: format=1; content-sha256=0af7e354161a1b02160f05a035142fb5042cba981abc1c661dedc9706e1523d8


# <<<< PREAMBLE CONTENT >>>>

from libc.stdint cimport intptr_t

cdef extern from * nogil:
    """
    #if defined(_MSC_VER) && !defined(__clang__)
        #include <intrin.h>
        static __forceinline int atomic_int_load(int *p) {
            int v = *(int volatile *)p; _ReadBarrier(); return v;
        }
        static __forceinline void atomic_int_store(int *p, int v) {
            _WriteBarrier(); *(int volatile *)p = v;
        }
    #elif defined(__cplusplus)
        /* GCC/Clang __atomic builtins work in any C++ standard without headers */
        static inline int atomic_int_load(int *p) {
            return __atomic_load_n(p, __ATOMIC_ACQUIRE);
        }
        static inline void atomic_int_store(int *p, int v) {
            __atomic_store_n(p, v, __ATOMIC_RELEASE);
        }
    #else
        #include <stdatomic.h>
        static inline int atomic_int_load(int *p) {
            return (int)atomic_load_explicit((atomic_int *)p, memory_order_acquire);
        }
        static inline void atomic_int_store(int *p, int v) {
            atomic_store_explicit((atomic_int *)p, v, memory_order_release);
        }
    #endif

    """
    cdef int _cyb_atomic_int_load "atomic_int_load"(int *p) nogil
    cdef void _cyb_atomic_int_store "atomic_int_store"(int *p, int v) nogil

cdef extern from "<windows.h>":
    ctypedef void* HMODULE
    void* _cyb_GetProcAddress "GetProcAddress"(HMODULE, const char*) nogil

from os import getenv as _cyb_getenv
import threading as _cyb_threading

ctypedef int (*_cyb_cuGetProcAddress_v2_T)(const char *, void **, int, cuuint64_t, CUdriverProcAddressQueryResult *)except?CUDA_ERROR_NOT_FOUND nogil

cdef int _cyb___py_driver_init = 0
cdef dict _cyb_func_ptrs = None
cdef object _cyb_symbol_lock = _cyb_threading.Lock()

# <<<< END OF PREAMBLE CONTENT >>>>

from libc.stdint cimport uintptr_t

from .utils import FunctionNotFoundError, NotSupportedError
from cuda.pathfinder import load_nvidia_dynamic_lib


###############################################################################
# Wrapper init
###############################################################################


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
cdef void* __cuCtxCreate_v2 = NULL
cdef void* __cuCtxCreate_v3 = NULL
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
cdef void* __cuStreamGetCaptureInfo_v2 = NULL
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
cdef void* __cuLogicalEndpointIdReserve = NULL
cdef void* __cuLogicalEndpointIdRelease = NULL
cdef void* __cuLogicalEndpointCreate = NULL
cdef void* __cuLogicalEndpointAddDevice = NULL
cdef void* __cuLogicalEndpointDestroy = NULL
cdef void* __cuLogicalEndpointBindAddr = NULL
cdef void* __cuLogicalEndpointBindMem = NULL
cdef void* __cuLogicalEndpointUnbind = NULL
cdef void* __cuLogicalEndpointExport = NULL
cdef void* __cuLogicalEndpointImport = NULL
cdef void* __cuLogicalEndpointGetLimits = NULL
cdef void* __cuLogicalEndpointQuery = NULL
cdef void* __cuStreamBeginRecaptureToGraph = NULL
cdef void* __cuDeviceGetFabricClusterUuid = NULL
cdef void* __cuDeviceGetCliqueCount = NULL
cdef void* __cuDeviceGetCliqueInfo = NULL
cdef void* __cuMemGetLocationInfo = NULL
cdef void* __cuGraphAddNode_v3 = NULL
cdef void* __cuGraphNodeSetParams_v2 = NULL
cdef void* __cuCheckpointOperationComplete = NULL
cdef void* __cuFabricCpyBatchAsync = NULL
cdef void* __cuTensorMapSetAllocationClass = NULL

cdef enum: _cyb_driver_n_functions = 526
cdef void** _cyb_driver_slots[_cyb_driver_n_functions]
cdef const char* _cyb_driver_names[_cyb_driver_n_functions]
_cyb_driver_slots[0] = &__cuGetErrorString; _cyb_driver_names[0] = b'cuGetErrorString'
_cyb_driver_slots[1] = &__cuGetErrorName; _cyb_driver_names[1] = b'cuGetErrorName'
_cyb_driver_slots[2] = &__cuInit; _cyb_driver_names[2] = b'cuInit'
_cyb_driver_slots[3] = &__cuDriverGetVersion; _cyb_driver_names[3] = b'cuDriverGetVersion'
_cyb_driver_slots[4] = &__cuDeviceGet; _cyb_driver_names[4] = b'cuDeviceGet'
_cyb_driver_slots[5] = &__cuDeviceGetCount; _cyb_driver_names[5] = b'cuDeviceGetCount'
_cyb_driver_slots[6] = &__cuDeviceGetName; _cyb_driver_names[6] = b'cuDeviceGetName'
_cyb_driver_slots[7] = &__cuDeviceGetUuid_v2; _cyb_driver_names[7] = b'cuDeviceGetUuid_v2'
_cyb_driver_slots[8] = &__cuDeviceGetLuid; _cyb_driver_names[8] = b'cuDeviceGetLuid'
_cyb_driver_slots[9] = &__cuDeviceTotalMem_v2; _cyb_driver_names[9] = b'cuDeviceTotalMem_v2'
_cyb_driver_slots[10] = &__cuDeviceGetTexture1DLinearMaxWidth; _cyb_driver_names[10] = b'cuDeviceGetTexture1DLinearMaxWidth'
_cyb_driver_slots[11] = &__cuDeviceGetAttribute; _cyb_driver_names[11] = b'cuDeviceGetAttribute'
_cyb_driver_slots[12] = &__cuDeviceGetNvSciSyncAttributes; _cyb_driver_names[12] = b'cuDeviceGetNvSciSyncAttributes'
_cyb_driver_slots[13] = &__cuDeviceSetMemPool; _cyb_driver_names[13] = b'cuDeviceSetMemPool'
_cyb_driver_slots[14] = &__cuDeviceGetMemPool; _cyb_driver_names[14] = b'cuDeviceGetMemPool'
_cyb_driver_slots[15] = &__cuDeviceGetDefaultMemPool; _cyb_driver_names[15] = b'cuDeviceGetDefaultMemPool'
_cyb_driver_slots[16] = &__cuDeviceGetExecAffinitySupport; _cyb_driver_names[16] = b'cuDeviceGetExecAffinitySupport'
_cyb_driver_slots[17] = &__cuFlushGPUDirectRDMAWrites; _cyb_driver_names[17] = b'cuFlushGPUDirectRDMAWrites'
_cyb_driver_slots[18] = &__cuDeviceGetProperties; _cyb_driver_names[18] = b'cuDeviceGetProperties'
_cyb_driver_slots[19] = &__cuDeviceComputeCapability; _cyb_driver_names[19] = b'cuDeviceComputeCapability'
_cyb_driver_slots[20] = &__cuDevicePrimaryCtxRetain; _cyb_driver_names[20] = b'cuDevicePrimaryCtxRetain'
_cyb_driver_slots[21] = &__cuDevicePrimaryCtxRelease_v2; _cyb_driver_names[21] = b'cuDevicePrimaryCtxRelease_v2'
_cyb_driver_slots[22] = &__cuDevicePrimaryCtxSetFlags_v2; _cyb_driver_names[22] = b'cuDevicePrimaryCtxSetFlags_v2'
_cyb_driver_slots[23] = &__cuDevicePrimaryCtxGetState; _cyb_driver_names[23] = b'cuDevicePrimaryCtxGetState'
_cyb_driver_slots[24] = &__cuDevicePrimaryCtxReset_v2; _cyb_driver_names[24] = b'cuDevicePrimaryCtxReset_v2'
_cyb_driver_slots[25] = &__cuCtxCreate_v2; _cyb_driver_names[25] = b'cuCtxCreate_v2'
_cyb_driver_slots[26] = &__cuCtxCreate_v3; _cyb_driver_names[26] = b'cuCtxCreate_v3'
_cyb_driver_slots[27] = &__cuCtxCreate_v4; _cyb_driver_names[27] = b'cuCtxCreate_v4'
_cyb_driver_slots[28] = &__cuCtxDestroy_v2; _cyb_driver_names[28] = b'cuCtxDestroy_v2'
_cyb_driver_slots[29] = &__cuCtxPushCurrent_v2; _cyb_driver_names[29] = b'cuCtxPushCurrent_v2'
_cyb_driver_slots[30] = &__cuCtxPopCurrent_v2; _cyb_driver_names[30] = b'cuCtxPopCurrent_v2'
_cyb_driver_slots[31] = &__cuCtxSetCurrent; _cyb_driver_names[31] = b'cuCtxSetCurrent'
_cyb_driver_slots[32] = &__cuCtxGetCurrent; _cyb_driver_names[32] = b'cuCtxGetCurrent'
_cyb_driver_slots[33] = &__cuCtxGetDevice; _cyb_driver_names[33] = b'cuCtxGetDevice'
_cyb_driver_slots[34] = &__cuCtxGetFlags; _cyb_driver_names[34] = b'cuCtxGetFlags'
_cyb_driver_slots[35] = &__cuCtxSetFlags; _cyb_driver_names[35] = b'cuCtxSetFlags'
_cyb_driver_slots[36] = &__cuCtxGetId; _cyb_driver_names[36] = b'cuCtxGetId'
_cyb_driver_slots[37] = &__cuCtxSynchronize; _cyb_driver_names[37] = b'cuCtxSynchronize'
_cyb_driver_slots[38] = &__cuCtxSetLimit; _cyb_driver_names[38] = b'cuCtxSetLimit'
_cyb_driver_slots[39] = &__cuCtxGetLimit; _cyb_driver_names[39] = b'cuCtxGetLimit'
_cyb_driver_slots[40] = &__cuCtxGetCacheConfig; _cyb_driver_names[40] = b'cuCtxGetCacheConfig'
_cyb_driver_slots[41] = &__cuCtxSetCacheConfig; _cyb_driver_names[41] = b'cuCtxSetCacheConfig'
_cyb_driver_slots[42] = &__cuCtxGetApiVersion; _cyb_driver_names[42] = b'cuCtxGetApiVersion'
_cyb_driver_slots[43] = &__cuCtxGetStreamPriorityRange; _cyb_driver_names[43] = b'cuCtxGetStreamPriorityRange'
_cyb_driver_slots[44] = &__cuCtxResetPersistingL2Cache; _cyb_driver_names[44] = b'cuCtxResetPersistingL2Cache'
_cyb_driver_slots[45] = &__cuCtxGetExecAffinity; _cyb_driver_names[45] = b'cuCtxGetExecAffinity'
_cyb_driver_slots[46] = &__cuCtxRecordEvent; _cyb_driver_names[46] = b'cuCtxRecordEvent'
_cyb_driver_slots[47] = &__cuCtxWaitEvent; _cyb_driver_names[47] = b'cuCtxWaitEvent'
_cyb_driver_slots[48] = &__cuCtxAttach; _cyb_driver_names[48] = b'cuCtxAttach'
_cyb_driver_slots[49] = &__cuCtxDetach; _cyb_driver_names[49] = b'cuCtxDetach'
_cyb_driver_slots[50] = &__cuCtxGetSharedMemConfig; _cyb_driver_names[50] = b'cuCtxGetSharedMemConfig'
_cyb_driver_slots[51] = &__cuCtxSetSharedMemConfig; _cyb_driver_names[51] = b'cuCtxSetSharedMemConfig'
_cyb_driver_slots[52] = &__cuModuleLoad; _cyb_driver_names[52] = b'cuModuleLoad'
_cyb_driver_slots[53] = &__cuModuleLoadData; _cyb_driver_names[53] = b'cuModuleLoadData'
_cyb_driver_slots[54] = &__cuModuleLoadDataEx; _cyb_driver_names[54] = b'cuModuleLoadDataEx'
_cyb_driver_slots[55] = &__cuModuleLoadFatBinary; _cyb_driver_names[55] = b'cuModuleLoadFatBinary'
_cyb_driver_slots[56] = &__cuModuleUnload; _cyb_driver_names[56] = b'cuModuleUnload'
_cyb_driver_slots[57] = &__cuModuleGetLoadingMode; _cyb_driver_names[57] = b'cuModuleGetLoadingMode'
_cyb_driver_slots[58] = &__cuModuleGetFunction; _cyb_driver_names[58] = b'cuModuleGetFunction'
_cyb_driver_slots[59] = &__cuModuleGetFunctionCount; _cyb_driver_names[59] = b'cuModuleGetFunctionCount'
_cyb_driver_slots[60] = &__cuModuleEnumerateFunctions; _cyb_driver_names[60] = b'cuModuleEnumerateFunctions'
_cyb_driver_slots[61] = &__cuModuleGetGlobal_v2; _cyb_driver_names[61] = b'cuModuleGetGlobal_v2'
_cyb_driver_slots[62] = &__cuLinkCreate_v2; _cyb_driver_names[62] = b'cuLinkCreate_v2'
_cyb_driver_slots[63] = &__cuLinkAddData_v2; _cyb_driver_names[63] = b'cuLinkAddData_v2'
_cyb_driver_slots[64] = &__cuLinkAddFile_v2; _cyb_driver_names[64] = b'cuLinkAddFile_v2'
_cyb_driver_slots[65] = &__cuLinkComplete; _cyb_driver_names[65] = b'cuLinkComplete'
_cyb_driver_slots[66] = &__cuLinkDestroy; _cyb_driver_names[66] = b'cuLinkDestroy'
_cyb_driver_slots[67] = &__cuModuleGetTexRef; _cyb_driver_names[67] = b'cuModuleGetTexRef'
_cyb_driver_slots[68] = &__cuModuleGetSurfRef; _cyb_driver_names[68] = b'cuModuleGetSurfRef'
_cyb_driver_slots[69] = &__cuLibraryLoadData; _cyb_driver_names[69] = b'cuLibraryLoadData'
_cyb_driver_slots[70] = &__cuLibraryLoadFromFile; _cyb_driver_names[70] = b'cuLibraryLoadFromFile'
_cyb_driver_slots[71] = &__cuLibraryUnload; _cyb_driver_names[71] = b'cuLibraryUnload'
_cyb_driver_slots[72] = &__cuLibraryGetKernel; _cyb_driver_names[72] = b'cuLibraryGetKernel'
_cyb_driver_slots[73] = &__cuLibraryGetKernelCount; _cyb_driver_names[73] = b'cuLibraryGetKernelCount'
_cyb_driver_slots[74] = &__cuLibraryEnumerateKernels; _cyb_driver_names[74] = b'cuLibraryEnumerateKernels'
_cyb_driver_slots[75] = &__cuLibraryGetModule; _cyb_driver_names[75] = b'cuLibraryGetModule'
_cyb_driver_slots[76] = &__cuKernelGetFunction; _cyb_driver_names[76] = b'cuKernelGetFunction'
_cyb_driver_slots[77] = &__cuKernelGetLibrary; _cyb_driver_names[77] = b'cuKernelGetLibrary'
_cyb_driver_slots[78] = &__cuLibraryGetGlobal; _cyb_driver_names[78] = b'cuLibraryGetGlobal'
_cyb_driver_slots[79] = &__cuLibraryGetManaged; _cyb_driver_names[79] = b'cuLibraryGetManaged'
_cyb_driver_slots[80] = &__cuLibraryGetUnifiedFunction; _cyb_driver_names[80] = b'cuLibraryGetUnifiedFunction'
_cyb_driver_slots[81] = &__cuKernelGetAttribute; _cyb_driver_names[81] = b'cuKernelGetAttribute'
_cyb_driver_slots[82] = &__cuKernelSetAttribute; _cyb_driver_names[82] = b'cuKernelSetAttribute'
_cyb_driver_slots[83] = &__cuKernelSetCacheConfig; _cyb_driver_names[83] = b'cuKernelSetCacheConfig'
_cyb_driver_slots[84] = &__cuKernelGetName; _cyb_driver_names[84] = b'cuKernelGetName'
_cyb_driver_slots[85] = &__cuKernelGetParamInfo; _cyb_driver_names[85] = b'cuKernelGetParamInfo'
_cyb_driver_slots[86] = &__cuMemGetInfo_v2; _cyb_driver_names[86] = b'cuMemGetInfo_v2'
_cyb_driver_slots[87] = &__cuMemAlloc_v2; _cyb_driver_names[87] = b'cuMemAlloc_v2'
_cyb_driver_slots[88] = &__cuMemAllocPitch_v2; _cyb_driver_names[88] = b'cuMemAllocPitch_v2'
_cyb_driver_slots[89] = &__cuMemFree_v2; _cyb_driver_names[89] = b'cuMemFree_v2'
_cyb_driver_slots[90] = &__cuMemGetAddressRange_v2; _cyb_driver_names[90] = b'cuMemGetAddressRange_v2'
_cyb_driver_slots[91] = &__cuMemAllocHost_v2; _cyb_driver_names[91] = b'cuMemAllocHost_v2'
_cyb_driver_slots[92] = &__cuMemFreeHost; _cyb_driver_names[92] = b'cuMemFreeHost'
_cyb_driver_slots[93] = &__cuMemHostAlloc; _cyb_driver_names[93] = b'cuMemHostAlloc'
_cyb_driver_slots[94] = &__cuMemHostGetDevicePointer_v2; _cyb_driver_names[94] = b'cuMemHostGetDevicePointer_v2'
_cyb_driver_slots[95] = &__cuMemHostGetFlags; _cyb_driver_names[95] = b'cuMemHostGetFlags'
_cyb_driver_slots[96] = &__cuMemAllocManaged; _cyb_driver_names[96] = b'cuMemAllocManaged'
_cyb_driver_slots[97] = &__cuDeviceRegisterAsyncNotification; _cyb_driver_names[97] = b'cuDeviceRegisterAsyncNotification'
_cyb_driver_slots[98] = &__cuDeviceUnregisterAsyncNotification; _cyb_driver_names[98] = b'cuDeviceUnregisterAsyncNotification'
_cyb_driver_slots[99] = &__cuDeviceGetByPCIBusId; _cyb_driver_names[99] = b'cuDeviceGetByPCIBusId'
_cyb_driver_slots[100] = &__cuDeviceGetPCIBusId; _cyb_driver_names[100] = b'cuDeviceGetPCIBusId'
_cyb_driver_slots[101] = &__cuIpcGetEventHandle; _cyb_driver_names[101] = b'cuIpcGetEventHandle'
_cyb_driver_slots[102] = &__cuIpcOpenEventHandle; _cyb_driver_names[102] = b'cuIpcOpenEventHandle'
_cyb_driver_slots[103] = &__cuIpcGetMemHandle; _cyb_driver_names[103] = b'cuIpcGetMemHandle'
_cyb_driver_slots[104] = &__cuIpcOpenMemHandle_v2; _cyb_driver_names[104] = b'cuIpcOpenMemHandle_v2'
_cyb_driver_slots[105] = &__cuIpcCloseMemHandle; _cyb_driver_names[105] = b'cuIpcCloseMemHandle'
_cyb_driver_slots[106] = &__cuMemHostRegister_v2; _cyb_driver_names[106] = b'cuMemHostRegister_v2'
_cyb_driver_slots[107] = &__cuMemHostUnregister; _cyb_driver_names[107] = b'cuMemHostUnregister'
_cyb_driver_slots[108] = &__cuMemcpy; _cyb_driver_names[108] = b'cuMemcpy'
_cyb_driver_slots[109] = &__cuMemcpyPeer; _cyb_driver_names[109] = b'cuMemcpyPeer'
_cyb_driver_slots[110] = &__cuMemcpyHtoD_v2; _cyb_driver_names[110] = b'cuMemcpyHtoD_v2'
_cyb_driver_slots[111] = &__cuMemcpyDtoH_v2; _cyb_driver_names[111] = b'cuMemcpyDtoH_v2'
_cyb_driver_slots[112] = &__cuMemcpyDtoD_v2; _cyb_driver_names[112] = b'cuMemcpyDtoD_v2'
_cyb_driver_slots[113] = &__cuMemcpyDtoA_v2; _cyb_driver_names[113] = b'cuMemcpyDtoA_v2'
_cyb_driver_slots[114] = &__cuMemcpyAtoD_v2; _cyb_driver_names[114] = b'cuMemcpyAtoD_v2'
_cyb_driver_slots[115] = &__cuMemcpyHtoA_v2; _cyb_driver_names[115] = b'cuMemcpyHtoA_v2'
_cyb_driver_slots[116] = &__cuMemcpyAtoH_v2; _cyb_driver_names[116] = b'cuMemcpyAtoH_v2'
_cyb_driver_slots[117] = &__cuMemcpyAtoA_v2; _cyb_driver_names[117] = b'cuMemcpyAtoA_v2'
_cyb_driver_slots[118] = &__cuMemcpy2D_v2; _cyb_driver_names[118] = b'cuMemcpy2D_v2'
_cyb_driver_slots[119] = &__cuMemcpy2DUnaligned_v2; _cyb_driver_names[119] = b'cuMemcpy2DUnaligned_v2'
_cyb_driver_slots[120] = &__cuMemcpy3D_v2; _cyb_driver_names[120] = b'cuMemcpy3D_v2'
_cyb_driver_slots[121] = &__cuMemcpy3DPeer; _cyb_driver_names[121] = b'cuMemcpy3DPeer'
_cyb_driver_slots[122] = &__cuMemcpyAsync; _cyb_driver_names[122] = b'cuMemcpyAsync'
_cyb_driver_slots[123] = &__cuMemcpyPeerAsync; _cyb_driver_names[123] = b'cuMemcpyPeerAsync'
_cyb_driver_slots[124] = &__cuMemcpyHtoDAsync_v2; _cyb_driver_names[124] = b'cuMemcpyHtoDAsync_v2'
_cyb_driver_slots[125] = &__cuMemcpyDtoHAsync_v2; _cyb_driver_names[125] = b'cuMemcpyDtoHAsync_v2'
_cyb_driver_slots[126] = &__cuMemcpyDtoDAsync_v2; _cyb_driver_names[126] = b'cuMemcpyDtoDAsync_v2'
_cyb_driver_slots[127] = &__cuMemcpyHtoAAsync_v2; _cyb_driver_names[127] = b'cuMemcpyHtoAAsync_v2'
_cyb_driver_slots[128] = &__cuMemcpyAtoHAsync_v2; _cyb_driver_names[128] = b'cuMemcpyAtoHAsync_v2'
_cyb_driver_slots[129] = &__cuMemcpy2DAsync_v2; _cyb_driver_names[129] = b'cuMemcpy2DAsync_v2'
_cyb_driver_slots[130] = &__cuMemcpy3DAsync_v2; _cyb_driver_names[130] = b'cuMemcpy3DAsync_v2'
_cyb_driver_slots[131] = &__cuMemcpy3DPeerAsync; _cyb_driver_names[131] = b'cuMemcpy3DPeerAsync'
_cyb_driver_slots[132] = &__cuMemsetD8_v2; _cyb_driver_names[132] = b'cuMemsetD8_v2'
_cyb_driver_slots[133] = &__cuMemsetD16_v2; _cyb_driver_names[133] = b'cuMemsetD16_v2'
_cyb_driver_slots[134] = &__cuMemsetD32_v2; _cyb_driver_names[134] = b'cuMemsetD32_v2'
_cyb_driver_slots[135] = &__cuMemsetD2D8_v2; _cyb_driver_names[135] = b'cuMemsetD2D8_v2'
_cyb_driver_slots[136] = &__cuMemsetD2D16_v2; _cyb_driver_names[136] = b'cuMemsetD2D16_v2'
_cyb_driver_slots[137] = &__cuMemsetD2D32_v2; _cyb_driver_names[137] = b'cuMemsetD2D32_v2'
_cyb_driver_slots[138] = &__cuMemsetD8Async; _cyb_driver_names[138] = b'cuMemsetD8Async'
_cyb_driver_slots[139] = &__cuMemsetD16Async; _cyb_driver_names[139] = b'cuMemsetD16Async'
_cyb_driver_slots[140] = &__cuMemsetD32Async; _cyb_driver_names[140] = b'cuMemsetD32Async'
_cyb_driver_slots[141] = &__cuMemsetD2D8Async; _cyb_driver_names[141] = b'cuMemsetD2D8Async'
_cyb_driver_slots[142] = &__cuMemsetD2D16Async; _cyb_driver_names[142] = b'cuMemsetD2D16Async'
_cyb_driver_slots[143] = &__cuMemsetD2D32Async; _cyb_driver_names[143] = b'cuMemsetD2D32Async'
_cyb_driver_slots[144] = &__cuArrayCreate_v2; _cyb_driver_names[144] = b'cuArrayCreate_v2'
_cyb_driver_slots[145] = &__cuArrayGetDescriptor_v2; _cyb_driver_names[145] = b'cuArrayGetDescriptor_v2'
_cyb_driver_slots[146] = &__cuArrayGetSparseProperties; _cyb_driver_names[146] = b'cuArrayGetSparseProperties'
_cyb_driver_slots[147] = &__cuMipmappedArrayGetSparseProperties; _cyb_driver_names[147] = b'cuMipmappedArrayGetSparseProperties'
_cyb_driver_slots[148] = &__cuArrayGetMemoryRequirements; _cyb_driver_names[148] = b'cuArrayGetMemoryRequirements'
_cyb_driver_slots[149] = &__cuMipmappedArrayGetMemoryRequirements; _cyb_driver_names[149] = b'cuMipmappedArrayGetMemoryRequirements'
_cyb_driver_slots[150] = &__cuArrayGetPlane; _cyb_driver_names[150] = b'cuArrayGetPlane'
_cyb_driver_slots[151] = &__cuArrayDestroy; _cyb_driver_names[151] = b'cuArrayDestroy'
_cyb_driver_slots[152] = &__cuArray3DCreate_v2; _cyb_driver_names[152] = b'cuArray3DCreate_v2'
_cyb_driver_slots[153] = &__cuArray3DGetDescriptor_v2; _cyb_driver_names[153] = b'cuArray3DGetDescriptor_v2'
_cyb_driver_slots[154] = &__cuMipmappedArrayCreate; _cyb_driver_names[154] = b'cuMipmappedArrayCreate'
_cyb_driver_slots[155] = &__cuMipmappedArrayGetLevel; _cyb_driver_names[155] = b'cuMipmappedArrayGetLevel'
_cyb_driver_slots[156] = &__cuMipmappedArrayDestroy; _cyb_driver_names[156] = b'cuMipmappedArrayDestroy'
_cyb_driver_slots[157] = &__cuMemGetHandleForAddressRange; _cyb_driver_names[157] = b'cuMemGetHandleForAddressRange'
_cyb_driver_slots[158] = &__cuMemBatchDecompressAsync; _cyb_driver_names[158] = b'cuMemBatchDecompressAsync'
_cyb_driver_slots[159] = &__cuMemAddressReserve; _cyb_driver_names[159] = b'cuMemAddressReserve'
_cyb_driver_slots[160] = &__cuMemAddressFree; _cyb_driver_names[160] = b'cuMemAddressFree'
_cyb_driver_slots[161] = &__cuMemCreate; _cyb_driver_names[161] = b'cuMemCreate'
_cyb_driver_slots[162] = &__cuMemRelease; _cyb_driver_names[162] = b'cuMemRelease'
_cyb_driver_slots[163] = &__cuMemMap; _cyb_driver_names[163] = b'cuMemMap'
_cyb_driver_slots[164] = &__cuMemMapArrayAsync; _cyb_driver_names[164] = b'cuMemMapArrayAsync'
_cyb_driver_slots[165] = &__cuMemUnmap; _cyb_driver_names[165] = b'cuMemUnmap'
_cyb_driver_slots[166] = &__cuMemSetAccess; _cyb_driver_names[166] = b'cuMemSetAccess'
_cyb_driver_slots[167] = &__cuMemGetAccess; _cyb_driver_names[167] = b'cuMemGetAccess'
_cyb_driver_slots[168] = &__cuMemExportToShareableHandle; _cyb_driver_names[168] = b'cuMemExportToShareableHandle'
_cyb_driver_slots[169] = &__cuMemImportFromShareableHandle; _cyb_driver_names[169] = b'cuMemImportFromShareableHandle'
_cyb_driver_slots[170] = &__cuMemGetAllocationGranularity; _cyb_driver_names[170] = b'cuMemGetAllocationGranularity'
_cyb_driver_slots[171] = &__cuMemGetAllocationPropertiesFromHandle; _cyb_driver_names[171] = b'cuMemGetAllocationPropertiesFromHandle'
_cyb_driver_slots[172] = &__cuMemRetainAllocationHandle; _cyb_driver_names[172] = b'cuMemRetainAllocationHandle'
_cyb_driver_slots[173] = &__cuMemFreeAsync; _cyb_driver_names[173] = b'cuMemFreeAsync'
_cyb_driver_slots[174] = &__cuMemAllocAsync; _cyb_driver_names[174] = b'cuMemAllocAsync'
_cyb_driver_slots[175] = &__cuMemPoolTrimTo; _cyb_driver_names[175] = b'cuMemPoolTrimTo'
_cyb_driver_slots[176] = &__cuMemPoolSetAttribute; _cyb_driver_names[176] = b'cuMemPoolSetAttribute'
_cyb_driver_slots[177] = &__cuMemPoolGetAttribute; _cyb_driver_names[177] = b'cuMemPoolGetAttribute'
_cyb_driver_slots[178] = &__cuMemPoolSetAccess; _cyb_driver_names[178] = b'cuMemPoolSetAccess'
_cyb_driver_slots[179] = &__cuMemPoolGetAccess; _cyb_driver_names[179] = b'cuMemPoolGetAccess'
_cyb_driver_slots[180] = &__cuMemPoolCreate; _cyb_driver_names[180] = b'cuMemPoolCreate'
_cyb_driver_slots[181] = &__cuMemPoolDestroy; _cyb_driver_names[181] = b'cuMemPoolDestroy'
_cyb_driver_slots[182] = &__cuMemAllocFromPoolAsync; _cyb_driver_names[182] = b'cuMemAllocFromPoolAsync'
_cyb_driver_slots[183] = &__cuMemPoolExportToShareableHandle; _cyb_driver_names[183] = b'cuMemPoolExportToShareableHandle'
_cyb_driver_slots[184] = &__cuMemPoolImportFromShareableHandle; _cyb_driver_names[184] = b'cuMemPoolImportFromShareableHandle'
_cyb_driver_slots[185] = &__cuMemPoolExportPointer; _cyb_driver_names[185] = b'cuMemPoolExportPointer'
_cyb_driver_slots[186] = &__cuMemPoolImportPointer; _cyb_driver_names[186] = b'cuMemPoolImportPointer'
_cyb_driver_slots[187] = &__cuMulticastCreate; _cyb_driver_names[187] = b'cuMulticastCreate'
_cyb_driver_slots[188] = &__cuMulticastAddDevice; _cyb_driver_names[188] = b'cuMulticastAddDevice'
_cyb_driver_slots[189] = &__cuMulticastBindMem; _cyb_driver_names[189] = b'cuMulticastBindMem'
_cyb_driver_slots[190] = &__cuMulticastBindAddr; _cyb_driver_names[190] = b'cuMulticastBindAddr'
_cyb_driver_slots[191] = &__cuMulticastUnbind; _cyb_driver_names[191] = b'cuMulticastUnbind'
_cyb_driver_slots[192] = &__cuMulticastGetGranularity; _cyb_driver_names[192] = b'cuMulticastGetGranularity'
_cyb_driver_slots[193] = &__cuPointerGetAttribute; _cyb_driver_names[193] = b'cuPointerGetAttribute'
_cyb_driver_slots[194] = &__cuMemPrefetchAsync_v2; _cyb_driver_names[194] = b'cuMemPrefetchAsync_v2'
_cyb_driver_slots[195] = &__cuMemAdvise_v2; _cyb_driver_names[195] = b'cuMemAdvise_v2'
_cyb_driver_slots[196] = &__cuMemRangeGetAttribute; _cyb_driver_names[196] = b'cuMemRangeGetAttribute'
_cyb_driver_slots[197] = &__cuMemRangeGetAttributes; _cyb_driver_names[197] = b'cuMemRangeGetAttributes'
_cyb_driver_slots[198] = &__cuPointerSetAttribute; _cyb_driver_names[198] = b'cuPointerSetAttribute'
_cyb_driver_slots[199] = &__cuPointerGetAttributes; _cyb_driver_names[199] = b'cuPointerGetAttributes'
_cyb_driver_slots[200] = &__cuStreamCreate; _cyb_driver_names[200] = b'cuStreamCreate'
_cyb_driver_slots[201] = &__cuStreamCreateWithPriority; _cyb_driver_names[201] = b'cuStreamCreateWithPriority'
_cyb_driver_slots[202] = &__cuStreamGetPriority; _cyb_driver_names[202] = b'cuStreamGetPriority'
_cyb_driver_slots[203] = &__cuStreamGetDevice; _cyb_driver_names[203] = b'cuStreamGetDevice'
_cyb_driver_slots[204] = &__cuStreamGetFlags; _cyb_driver_names[204] = b'cuStreamGetFlags'
_cyb_driver_slots[205] = &__cuStreamGetId; _cyb_driver_names[205] = b'cuStreamGetId'
_cyb_driver_slots[206] = &__cuStreamGetCtx; _cyb_driver_names[206] = b'cuStreamGetCtx'
_cyb_driver_slots[207] = &__cuStreamGetCtx_v2; _cyb_driver_names[207] = b'cuStreamGetCtx_v2'
_cyb_driver_slots[208] = &__cuStreamWaitEvent; _cyb_driver_names[208] = b'cuStreamWaitEvent'
_cyb_driver_slots[209] = &__cuStreamAddCallback; _cyb_driver_names[209] = b'cuStreamAddCallback'
_cyb_driver_slots[210] = &__cuStreamBeginCapture_v2; _cyb_driver_names[210] = b'cuStreamBeginCapture_v2'
_cyb_driver_slots[211] = &__cuStreamBeginCaptureToGraph; _cyb_driver_names[211] = b'cuStreamBeginCaptureToGraph'
_cyb_driver_slots[212] = &__cuThreadExchangeStreamCaptureMode; _cyb_driver_names[212] = b'cuThreadExchangeStreamCaptureMode'
_cyb_driver_slots[213] = &__cuStreamEndCapture; _cyb_driver_names[213] = b'cuStreamEndCapture'
_cyb_driver_slots[214] = &__cuStreamIsCapturing; _cyb_driver_names[214] = b'cuStreamIsCapturing'
_cyb_driver_slots[215] = &__cuStreamGetCaptureInfo_v2; _cyb_driver_names[215] = b'cuStreamGetCaptureInfo_v2'
_cyb_driver_slots[216] = &__cuStreamGetCaptureInfo_v3; _cyb_driver_names[216] = b'cuStreamGetCaptureInfo_v3'
_cyb_driver_slots[217] = &__cuStreamUpdateCaptureDependencies_v2; _cyb_driver_names[217] = b'cuStreamUpdateCaptureDependencies_v2'
_cyb_driver_slots[218] = &__cuStreamAttachMemAsync; _cyb_driver_names[218] = b'cuStreamAttachMemAsync'
_cyb_driver_slots[219] = &__cuStreamQuery; _cyb_driver_names[219] = b'cuStreamQuery'
_cyb_driver_slots[220] = &__cuStreamSynchronize; _cyb_driver_names[220] = b'cuStreamSynchronize'
_cyb_driver_slots[221] = &__cuStreamDestroy_v2; _cyb_driver_names[221] = b'cuStreamDestroy_v2'
_cyb_driver_slots[222] = &__cuStreamCopyAttributes; _cyb_driver_names[222] = b'cuStreamCopyAttributes'
_cyb_driver_slots[223] = &__cuStreamGetAttribute; _cyb_driver_names[223] = b'cuStreamGetAttribute'
_cyb_driver_slots[224] = &__cuStreamSetAttribute; _cyb_driver_names[224] = b'cuStreamSetAttribute'
_cyb_driver_slots[225] = &__cuEventCreate; _cyb_driver_names[225] = b'cuEventCreate'
_cyb_driver_slots[226] = &__cuEventRecord; _cyb_driver_names[226] = b'cuEventRecord'
_cyb_driver_slots[227] = &__cuEventRecordWithFlags; _cyb_driver_names[227] = b'cuEventRecordWithFlags'
_cyb_driver_slots[228] = &__cuEventQuery; _cyb_driver_names[228] = b'cuEventQuery'
_cyb_driver_slots[229] = &__cuEventSynchronize; _cyb_driver_names[229] = b'cuEventSynchronize'
_cyb_driver_slots[230] = &__cuEventDestroy_v2; _cyb_driver_names[230] = b'cuEventDestroy_v2'
_cyb_driver_slots[231] = &__cuEventElapsedTime_v2; _cyb_driver_names[231] = b'cuEventElapsedTime_v2'
_cyb_driver_slots[232] = &__cuImportExternalMemory; _cyb_driver_names[232] = b'cuImportExternalMemory'
_cyb_driver_slots[233] = &__cuExternalMemoryGetMappedBuffer; _cyb_driver_names[233] = b'cuExternalMemoryGetMappedBuffer'
_cyb_driver_slots[234] = &__cuExternalMemoryGetMappedMipmappedArray; _cyb_driver_names[234] = b'cuExternalMemoryGetMappedMipmappedArray'
_cyb_driver_slots[235] = &__cuDestroyExternalMemory; _cyb_driver_names[235] = b'cuDestroyExternalMemory'
_cyb_driver_slots[236] = &__cuImportExternalSemaphore; _cyb_driver_names[236] = b'cuImportExternalSemaphore'
_cyb_driver_slots[237] = &__cuSignalExternalSemaphoresAsync; _cyb_driver_names[237] = b'cuSignalExternalSemaphoresAsync'
_cyb_driver_slots[238] = &__cuWaitExternalSemaphoresAsync; _cyb_driver_names[238] = b'cuWaitExternalSemaphoresAsync'
_cyb_driver_slots[239] = &__cuDestroyExternalSemaphore; _cyb_driver_names[239] = b'cuDestroyExternalSemaphore'
_cyb_driver_slots[240] = &__cuStreamWaitValue32_v2; _cyb_driver_names[240] = b'cuStreamWaitValue32_v2'
_cyb_driver_slots[241] = &__cuStreamWaitValue64_v2; _cyb_driver_names[241] = b'cuStreamWaitValue64_v2'
_cyb_driver_slots[242] = &__cuStreamWriteValue32_v2; _cyb_driver_names[242] = b'cuStreamWriteValue32_v2'
_cyb_driver_slots[243] = &__cuStreamWriteValue64_v2; _cyb_driver_names[243] = b'cuStreamWriteValue64_v2'
_cyb_driver_slots[244] = &__cuStreamBatchMemOp_v2; _cyb_driver_names[244] = b'cuStreamBatchMemOp_v2'
_cyb_driver_slots[245] = &__cuFuncGetAttribute; _cyb_driver_names[245] = b'cuFuncGetAttribute'
_cyb_driver_slots[246] = &__cuFuncSetAttribute; _cyb_driver_names[246] = b'cuFuncSetAttribute'
_cyb_driver_slots[247] = &__cuFuncSetCacheConfig; _cyb_driver_names[247] = b'cuFuncSetCacheConfig'
_cyb_driver_slots[248] = &__cuFuncGetModule; _cyb_driver_names[248] = b'cuFuncGetModule'
_cyb_driver_slots[249] = &__cuFuncGetName; _cyb_driver_names[249] = b'cuFuncGetName'
_cyb_driver_slots[250] = &__cuFuncGetParamInfo; _cyb_driver_names[250] = b'cuFuncGetParamInfo'
_cyb_driver_slots[251] = &__cuFuncIsLoaded; _cyb_driver_names[251] = b'cuFuncIsLoaded'
_cyb_driver_slots[252] = &__cuFuncLoad; _cyb_driver_names[252] = b'cuFuncLoad'
_cyb_driver_slots[253] = &__cuLaunchKernel; _cyb_driver_names[253] = b'cuLaunchKernel'
_cyb_driver_slots[254] = &__cuLaunchKernelEx; _cyb_driver_names[254] = b'cuLaunchKernelEx'
_cyb_driver_slots[255] = &__cuLaunchCooperativeKernel; _cyb_driver_names[255] = b'cuLaunchCooperativeKernel'
_cyb_driver_slots[256] = &__cuLaunchCooperativeKernelMultiDevice; _cyb_driver_names[256] = b'cuLaunchCooperativeKernelMultiDevice'
_cyb_driver_slots[257] = &__cuLaunchHostFunc; _cyb_driver_names[257] = b'cuLaunchHostFunc'
_cyb_driver_slots[258] = &__cuFuncSetBlockShape; _cyb_driver_names[258] = b'cuFuncSetBlockShape'
_cyb_driver_slots[259] = &__cuFuncSetSharedSize; _cyb_driver_names[259] = b'cuFuncSetSharedSize'
_cyb_driver_slots[260] = &__cuParamSetSize; _cyb_driver_names[260] = b'cuParamSetSize'
_cyb_driver_slots[261] = &__cuParamSeti; _cyb_driver_names[261] = b'cuParamSeti'
_cyb_driver_slots[262] = &__cuParamSetf; _cyb_driver_names[262] = b'cuParamSetf'
_cyb_driver_slots[263] = &__cuParamSetv; _cyb_driver_names[263] = b'cuParamSetv'
_cyb_driver_slots[264] = &__cuLaunch; _cyb_driver_names[264] = b'cuLaunch'
_cyb_driver_slots[265] = &__cuLaunchGrid; _cyb_driver_names[265] = b'cuLaunchGrid'
_cyb_driver_slots[266] = &__cuLaunchGridAsync; _cyb_driver_names[266] = b'cuLaunchGridAsync'
_cyb_driver_slots[267] = &__cuParamSetTexRef; _cyb_driver_names[267] = b'cuParamSetTexRef'
_cyb_driver_slots[268] = &__cuFuncSetSharedMemConfig; _cyb_driver_names[268] = b'cuFuncSetSharedMemConfig'
_cyb_driver_slots[269] = &__cuGraphCreate; _cyb_driver_names[269] = b'cuGraphCreate'
_cyb_driver_slots[270] = &__cuGraphAddKernelNode_v2; _cyb_driver_names[270] = b'cuGraphAddKernelNode_v2'
_cyb_driver_slots[271] = &__cuGraphKernelNodeGetParams_v2; _cyb_driver_names[271] = b'cuGraphKernelNodeGetParams_v2'
_cyb_driver_slots[272] = &__cuGraphKernelNodeSetParams_v2; _cyb_driver_names[272] = b'cuGraphKernelNodeSetParams_v2'
_cyb_driver_slots[273] = &__cuGraphAddMemcpyNode; _cyb_driver_names[273] = b'cuGraphAddMemcpyNode'
_cyb_driver_slots[274] = &__cuGraphMemcpyNodeGetParams; _cyb_driver_names[274] = b'cuGraphMemcpyNodeGetParams'
_cyb_driver_slots[275] = &__cuGraphMemcpyNodeSetParams; _cyb_driver_names[275] = b'cuGraphMemcpyNodeSetParams'
_cyb_driver_slots[276] = &__cuGraphAddMemsetNode; _cyb_driver_names[276] = b'cuGraphAddMemsetNode'
_cyb_driver_slots[277] = &__cuGraphMemsetNodeGetParams; _cyb_driver_names[277] = b'cuGraphMemsetNodeGetParams'
_cyb_driver_slots[278] = &__cuGraphMemsetNodeSetParams; _cyb_driver_names[278] = b'cuGraphMemsetNodeSetParams'
_cyb_driver_slots[279] = &__cuGraphAddHostNode; _cyb_driver_names[279] = b'cuGraphAddHostNode'
_cyb_driver_slots[280] = &__cuGraphHostNodeGetParams; _cyb_driver_names[280] = b'cuGraphHostNodeGetParams'
_cyb_driver_slots[281] = &__cuGraphHostNodeSetParams; _cyb_driver_names[281] = b'cuGraphHostNodeSetParams'
_cyb_driver_slots[282] = &__cuGraphAddChildGraphNode; _cyb_driver_names[282] = b'cuGraphAddChildGraphNode'
_cyb_driver_slots[283] = &__cuGraphChildGraphNodeGetGraph; _cyb_driver_names[283] = b'cuGraphChildGraphNodeGetGraph'
_cyb_driver_slots[284] = &__cuGraphAddEmptyNode; _cyb_driver_names[284] = b'cuGraphAddEmptyNode'
_cyb_driver_slots[285] = &__cuGraphAddEventRecordNode; _cyb_driver_names[285] = b'cuGraphAddEventRecordNode'
_cyb_driver_slots[286] = &__cuGraphEventRecordNodeGetEvent; _cyb_driver_names[286] = b'cuGraphEventRecordNodeGetEvent'
_cyb_driver_slots[287] = &__cuGraphEventRecordNodeSetEvent; _cyb_driver_names[287] = b'cuGraphEventRecordNodeSetEvent'
_cyb_driver_slots[288] = &__cuGraphAddEventWaitNode; _cyb_driver_names[288] = b'cuGraphAddEventWaitNode'
_cyb_driver_slots[289] = &__cuGraphEventWaitNodeGetEvent; _cyb_driver_names[289] = b'cuGraphEventWaitNodeGetEvent'
_cyb_driver_slots[290] = &__cuGraphEventWaitNodeSetEvent; _cyb_driver_names[290] = b'cuGraphEventWaitNodeSetEvent'
_cyb_driver_slots[291] = &__cuGraphAddExternalSemaphoresSignalNode; _cyb_driver_names[291] = b'cuGraphAddExternalSemaphoresSignalNode'
_cyb_driver_slots[292] = &__cuGraphExternalSemaphoresSignalNodeGetParams; _cyb_driver_names[292] = b'cuGraphExternalSemaphoresSignalNodeGetParams'
_cyb_driver_slots[293] = &__cuGraphExternalSemaphoresSignalNodeSetParams; _cyb_driver_names[293] = b'cuGraphExternalSemaphoresSignalNodeSetParams'
_cyb_driver_slots[294] = &__cuGraphAddExternalSemaphoresWaitNode; _cyb_driver_names[294] = b'cuGraphAddExternalSemaphoresWaitNode'
_cyb_driver_slots[295] = &__cuGraphExternalSemaphoresWaitNodeGetParams; _cyb_driver_names[295] = b'cuGraphExternalSemaphoresWaitNodeGetParams'
_cyb_driver_slots[296] = &__cuGraphExternalSemaphoresWaitNodeSetParams; _cyb_driver_names[296] = b'cuGraphExternalSemaphoresWaitNodeSetParams'
_cyb_driver_slots[297] = &__cuGraphAddBatchMemOpNode; _cyb_driver_names[297] = b'cuGraphAddBatchMemOpNode'
_cyb_driver_slots[298] = &__cuGraphBatchMemOpNodeGetParams; _cyb_driver_names[298] = b'cuGraphBatchMemOpNodeGetParams'
_cyb_driver_slots[299] = &__cuGraphBatchMemOpNodeSetParams; _cyb_driver_names[299] = b'cuGraphBatchMemOpNodeSetParams'
_cyb_driver_slots[300] = &__cuGraphExecBatchMemOpNodeSetParams; _cyb_driver_names[300] = b'cuGraphExecBatchMemOpNodeSetParams'
_cyb_driver_slots[301] = &__cuGraphAddMemAllocNode; _cyb_driver_names[301] = b'cuGraphAddMemAllocNode'
_cyb_driver_slots[302] = &__cuGraphMemAllocNodeGetParams; _cyb_driver_names[302] = b'cuGraphMemAllocNodeGetParams'
_cyb_driver_slots[303] = &__cuGraphAddMemFreeNode; _cyb_driver_names[303] = b'cuGraphAddMemFreeNode'
_cyb_driver_slots[304] = &__cuGraphMemFreeNodeGetParams; _cyb_driver_names[304] = b'cuGraphMemFreeNodeGetParams'
_cyb_driver_slots[305] = &__cuDeviceGraphMemTrim; _cyb_driver_names[305] = b'cuDeviceGraphMemTrim'
_cyb_driver_slots[306] = &__cuDeviceGetGraphMemAttribute; _cyb_driver_names[306] = b'cuDeviceGetGraphMemAttribute'
_cyb_driver_slots[307] = &__cuDeviceSetGraphMemAttribute; _cyb_driver_names[307] = b'cuDeviceSetGraphMemAttribute'
_cyb_driver_slots[308] = &__cuGraphClone; _cyb_driver_names[308] = b'cuGraphClone'
_cyb_driver_slots[309] = &__cuGraphNodeFindInClone; _cyb_driver_names[309] = b'cuGraphNodeFindInClone'
_cyb_driver_slots[310] = &__cuGraphNodeGetType; _cyb_driver_names[310] = b'cuGraphNodeGetType'
_cyb_driver_slots[311] = &__cuGraphGetNodes; _cyb_driver_names[311] = b'cuGraphGetNodes'
_cyb_driver_slots[312] = &__cuGraphGetRootNodes; _cyb_driver_names[312] = b'cuGraphGetRootNodes'
_cyb_driver_slots[313] = &__cuGraphGetEdges_v2; _cyb_driver_names[313] = b'cuGraphGetEdges_v2'
_cyb_driver_slots[314] = &__cuGraphNodeGetDependencies_v2; _cyb_driver_names[314] = b'cuGraphNodeGetDependencies_v2'
_cyb_driver_slots[315] = &__cuGraphNodeGetDependentNodes_v2; _cyb_driver_names[315] = b'cuGraphNodeGetDependentNodes_v2'
_cyb_driver_slots[316] = &__cuGraphAddDependencies_v2; _cyb_driver_names[316] = b'cuGraphAddDependencies_v2'
_cyb_driver_slots[317] = &__cuGraphRemoveDependencies_v2; _cyb_driver_names[317] = b'cuGraphRemoveDependencies_v2'
_cyb_driver_slots[318] = &__cuGraphDestroyNode; _cyb_driver_names[318] = b'cuGraphDestroyNode'
_cyb_driver_slots[319] = &__cuGraphInstantiateWithFlags; _cyb_driver_names[319] = b'cuGraphInstantiateWithFlags'
_cyb_driver_slots[320] = &__cuGraphInstantiateWithParams; _cyb_driver_names[320] = b'cuGraphInstantiateWithParams'
_cyb_driver_slots[321] = &__cuGraphExecGetFlags; _cyb_driver_names[321] = b'cuGraphExecGetFlags'
_cyb_driver_slots[322] = &__cuGraphExecKernelNodeSetParams_v2; _cyb_driver_names[322] = b'cuGraphExecKernelNodeSetParams_v2'
_cyb_driver_slots[323] = &__cuGraphExecMemcpyNodeSetParams; _cyb_driver_names[323] = b'cuGraphExecMemcpyNodeSetParams'
_cyb_driver_slots[324] = &__cuGraphExecMemsetNodeSetParams; _cyb_driver_names[324] = b'cuGraphExecMemsetNodeSetParams'
_cyb_driver_slots[325] = &__cuGraphExecHostNodeSetParams; _cyb_driver_names[325] = b'cuGraphExecHostNodeSetParams'
_cyb_driver_slots[326] = &__cuGraphExecChildGraphNodeSetParams; _cyb_driver_names[326] = b'cuGraphExecChildGraphNodeSetParams'
_cyb_driver_slots[327] = &__cuGraphExecEventRecordNodeSetEvent; _cyb_driver_names[327] = b'cuGraphExecEventRecordNodeSetEvent'
_cyb_driver_slots[328] = &__cuGraphExecEventWaitNodeSetEvent; _cyb_driver_names[328] = b'cuGraphExecEventWaitNodeSetEvent'
_cyb_driver_slots[329] = &__cuGraphExecExternalSemaphoresSignalNodeSetParams; _cyb_driver_names[329] = b'cuGraphExecExternalSemaphoresSignalNodeSetParams'
_cyb_driver_slots[330] = &__cuGraphExecExternalSemaphoresWaitNodeSetParams; _cyb_driver_names[330] = b'cuGraphExecExternalSemaphoresWaitNodeSetParams'
_cyb_driver_slots[331] = &__cuGraphNodeSetEnabled; _cyb_driver_names[331] = b'cuGraphNodeSetEnabled'
_cyb_driver_slots[332] = &__cuGraphNodeGetEnabled; _cyb_driver_names[332] = b'cuGraphNodeGetEnabled'
_cyb_driver_slots[333] = &__cuGraphUpload; _cyb_driver_names[333] = b'cuGraphUpload'
_cyb_driver_slots[334] = &__cuGraphLaunch; _cyb_driver_names[334] = b'cuGraphLaunch'
_cyb_driver_slots[335] = &__cuGraphExecDestroy; _cyb_driver_names[335] = b'cuGraphExecDestroy'
_cyb_driver_slots[336] = &__cuGraphDestroy; _cyb_driver_names[336] = b'cuGraphDestroy'
_cyb_driver_slots[337] = &__cuGraphExecUpdate_v2; _cyb_driver_names[337] = b'cuGraphExecUpdate_v2'
_cyb_driver_slots[338] = &__cuGraphKernelNodeCopyAttributes; _cyb_driver_names[338] = b'cuGraphKernelNodeCopyAttributes'
_cyb_driver_slots[339] = &__cuGraphKernelNodeGetAttribute; _cyb_driver_names[339] = b'cuGraphKernelNodeGetAttribute'
_cyb_driver_slots[340] = &__cuGraphKernelNodeSetAttribute; _cyb_driver_names[340] = b'cuGraphKernelNodeSetAttribute'
_cyb_driver_slots[341] = &__cuGraphDebugDotPrint; _cyb_driver_names[341] = b'cuGraphDebugDotPrint'
_cyb_driver_slots[342] = &__cuUserObjectCreate; _cyb_driver_names[342] = b'cuUserObjectCreate'
_cyb_driver_slots[343] = &__cuUserObjectRetain; _cyb_driver_names[343] = b'cuUserObjectRetain'
_cyb_driver_slots[344] = &__cuUserObjectRelease; _cyb_driver_names[344] = b'cuUserObjectRelease'
_cyb_driver_slots[345] = &__cuGraphRetainUserObject; _cyb_driver_names[345] = b'cuGraphRetainUserObject'
_cyb_driver_slots[346] = &__cuGraphReleaseUserObject; _cyb_driver_names[346] = b'cuGraphReleaseUserObject'
_cyb_driver_slots[347] = &__cuGraphAddNode_v2; _cyb_driver_names[347] = b'cuGraphAddNode_v2'
_cyb_driver_slots[348] = &__cuGraphNodeSetParams; _cyb_driver_names[348] = b'cuGraphNodeSetParams'
_cyb_driver_slots[349] = &__cuGraphExecNodeSetParams; _cyb_driver_names[349] = b'cuGraphExecNodeSetParams'
_cyb_driver_slots[350] = &__cuGraphConditionalHandleCreate; _cyb_driver_names[350] = b'cuGraphConditionalHandleCreate'
_cyb_driver_slots[351] = &__cuOccupancyMaxActiveBlocksPerMultiprocessor; _cyb_driver_names[351] = b'cuOccupancyMaxActiveBlocksPerMultiprocessor'
_cyb_driver_slots[352] = &__cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags; _cyb_driver_names[352] = b'cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags'
_cyb_driver_slots[353] = &__cuOccupancyMaxPotentialBlockSize; _cyb_driver_names[353] = b'cuOccupancyMaxPotentialBlockSize'
_cyb_driver_slots[354] = &__cuOccupancyMaxPotentialBlockSizeWithFlags; _cyb_driver_names[354] = b'cuOccupancyMaxPotentialBlockSizeWithFlags'
_cyb_driver_slots[355] = &__cuOccupancyAvailableDynamicSMemPerBlock; _cyb_driver_names[355] = b'cuOccupancyAvailableDynamicSMemPerBlock'
_cyb_driver_slots[356] = &__cuOccupancyMaxPotentialClusterSize; _cyb_driver_names[356] = b'cuOccupancyMaxPotentialClusterSize'
_cyb_driver_slots[357] = &__cuOccupancyMaxActiveClusters; _cyb_driver_names[357] = b'cuOccupancyMaxActiveClusters'
_cyb_driver_slots[358] = &__cuTexRefSetArray; _cyb_driver_names[358] = b'cuTexRefSetArray'
_cyb_driver_slots[359] = &__cuTexRefSetMipmappedArray; _cyb_driver_names[359] = b'cuTexRefSetMipmappedArray'
_cyb_driver_slots[360] = &__cuTexRefSetAddress_v2; _cyb_driver_names[360] = b'cuTexRefSetAddress_v2'
_cyb_driver_slots[361] = &__cuTexRefSetAddress2D_v3; _cyb_driver_names[361] = b'cuTexRefSetAddress2D_v3'
_cyb_driver_slots[362] = &__cuTexRefSetFormat; _cyb_driver_names[362] = b'cuTexRefSetFormat'
_cyb_driver_slots[363] = &__cuTexRefSetAddressMode; _cyb_driver_names[363] = b'cuTexRefSetAddressMode'
_cyb_driver_slots[364] = &__cuTexRefSetFilterMode; _cyb_driver_names[364] = b'cuTexRefSetFilterMode'
_cyb_driver_slots[365] = &__cuTexRefSetMipmapFilterMode; _cyb_driver_names[365] = b'cuTexRefSetMipmapFilterMode'
_cyb_driver_slots[366] = &__cuTexRefSetMipmapLevelBias; _cyb_driver_names[366] = b'cuTexRefSetMipmapLevelBias'
_cyb_driver_slots[367] = &__cuTexRefSetMipmapLevelClamp; _cyb_driver_names[367] = b'cuTexRefSetMipmapLevelClamp'
_cyb_driver_slots[368] = &__cuTexRefSetMaxAnisotropy; _cyb_driver_names[368] = b'cuTexRefSetMaxAnisotropy'
_cyb_driver_slots[369] = &__cuTexRefSetBorderColor; _cyb_driver_names[369] = b'cuTexRefSetBorderColor'
_cyb_driver_slots[370] = &__cuTexRefSetFlags; _cyb_driver_names[370] = b'cuTexRefSetFlags'
_cyb_driver_slots[371] = &__cuTexRefGetAddress_v2; _cyb_driver_names[371] = b'cuTexRefGetAddress_v2'
_cyb_driver_slots[372] = &__cuTexRefGetArray; _cyb_driver_names[372] = b'cuTexRefGetArray'
_cyb_driver_slots[373] = &__cuTexRefGetMipmappedArray; _cyb_driver_names[373] = b'cuTexRefGetMipmappedArray'
_cyb_driver_slots[374] = &__cuTexRefGetAddressMode; _cyb_driver_names[374] = b'cuTexRefGetAddressMode'
_cyb_driver_slots[375] = &__cuTexRefGetFilterMode; _cyb_driver_names[375] = b'cuTexRefGetFilterMode'
_cyb_driver_slots[376] = &__cuTexRefGetFormat; _cyb_driver_names[376] = b'cuTexRefGetFormat'
_cyb_driver_slots[377] = &__cuTexRefGetMipmapFilterMode; _cyb_driver_names[377] = b'cuTexRefGetMipmapFilterMode'
_cyb_driver_slots[378] = &__cuTexRefGetMipmapLevelBias; _cyb_driver_names[378] = b'cuTexRefGetMipmapLevelBias'
_cyb_driver_slots[379] = &__cuTexRefGetMipmapLevelClamp; _cyb_driver_names[379] = b'cuTexRefGetMipmapLevelClamp'
_cyb_driver_slots[380] = &__cuTexRefGetMaxAnisotropy; _cyb_driver_names[380] = b'cuTexRefGetMaxAnisotropy'
_cyb_driver_slots[381] = &__cuTexRefGetBorderColor; _cyb_driver_names[381] = b'cuTexRefGetBorderColor'
_cyb_driver_slots[382] = &__cuTexRefGetFlags; _cyb_driver_names[382] = b'cuTexRefGetFlags'
_cyb_driver_slots[383] = &__cuTexRefCreate; _cyb_driver_names[383] = b'cuTexRefCreate'
_cyb_driver_slots[384] = &__cuTexRefDestroy; _cyb_driver_names[384] = b'cuTexRefDestroy'
_cyb_driver_slots[385] = &__cuSurfRefSetArray; _cyb_driver_names[385] = b'cuSurfRefSetArray'
_cyb_driver_slots[386] = &__cuSurfRefGetArray; _cyb_driver_names[386] = b'cuSurfRefGetArray'
_cyb_driver_slots[387] = &__cuTexObjectCreate; _cyb_driver_names[387] = b'cuTexObjectCreate'
_cyb_driver_slots[388] = &__cuTexObjectDestroy; _cyb_driver_names[388] = b'cuTexObjectDestroy'
_cyb_driver_slots[389] = &__cuTexObjectGetResourceDesc; _cyb_driver_names[389] = b'cuTexObjectGetResourceDesc'
_cyb_driver_slots[390] = &__cuTexObjectGetTextureDesc; _cyb_driver_names[390] = b'cuTexObjectGetTextureDesc'
_cyb_driver_slots[391] = &__cuTexObjectGetResourceViewDesc; _cyb_driver_names[391] = b'cuTexObjectGetResourceViewDesc'
_cyb_driver_slots[392] = &__cuSurfObjectCreate; _cyb_driver_names[392] = b'cuSurfObjectCreate'
_cyb_driver_slots[393] = &__cuSurfObjectDestroy; _cyb_driver_names[393] = b'cuSurfObjectDestroy'
_cyb_driver_slots[394] = &__cuSurfObjectGetResourceDesc; _cyb_driver_names[394] = b'cuSurfObjectGetResourceDesc'
_cyb_driver_slots[395] = &__cuTensorMapEncodeTiled; _cyb_driver_names[395] = b'cuTensorMapEncodeTiled'
_cyb_driver_slots[396] = &__cuTensorMapEncodeIm2col; _cyb_driver_names[396] = b'cuTensorMapEncodeIm2col'
_cyb_driver_slots[397] = &__cuTensorMapEncodeIm2colWide; _cyb_driver_names[397] = b'cuTensorMapEncodeIm2colWide'
_cyb_driver_slots[398] = &__cuTensorMapReplaceAddress; _cyb_driver_names[398] = b'cuTensorMapReplaceAddress'
_cyb_driver_slots[399] = &__cuDeviceCanAccessPeer; _cyb_driver_names[399] = b'cuDeviceCanAccessPeer'
_cyb_driver_slots[400] = &__cuCtxEnablePeerAccess; _cyb_driver_names[400] = b'cuCtxEnablePeerAccess'
_cyb_driver_slots[401] = &__cuCtxDisablePeerAccess; _cyb_driver_names[401] = b'cuCtxDisablePeerAccess'
_cyb_driver_slots[402] = &__cuDeviceGetP2PAttribute; _cyb_driver_names[402] = b'cuDeviceGetP2PAttribute'
_cyb_driver_slots[403] = &__cuGraphicsUnregisterResource; _cyb_driver_names[403] = b'cuGraphicsUnregisterResource'
_cyb_driver_slots[404] = &__cuGraphicsSubResourceGetMappedArray; _cyb_driver_names[404] = b'cuGraphicsSubResourceGetMappedArray'
_cyb_driver_slots[405] = &__cuGraphicsResourceGetMappedMipmappedArray; _cyb_driver_names[405] = b'cuGraphicsResourceGetMappedMipmappedArray'
_cyb_driver_slots[406] = &__cuGraphicsResourceGetMappedPointer_v2; _cyb_driver_names[406] = b'cuGraphicsResourceGetMappedPointer_v2'
_cyb_driver_slots[407] = &__cuGraphicsResourceSetMapFlags_v2; _cyb_driver_names[407] = b'cuGraphicsResourceSetMapFlags_v2'
_cyb_driver_slots[408] = &__cuGraphicsMapResources; _cyb_driver_names[408] = b'cuGraphicsMapResources'
_cyb_driver_slots[409] = &__cuGraphicsUnmapResources; _cyb_driver_names[409] = b'cuGraphicsUnmapResources'
_cyb_driver_slots[410] = &__cuGetProcAddress_v2; _cyb_driver_names[410] = b'cuGetProcAddress_v2'
_cyb_driver_slots[411] = &__cuCoredumpGetAttribute; _cyb_driver_names[411] = b'cuCoredumpGetAttribute'
_cyb_driver_slots[412] = &__cuCoredumpGetAttributeGlobal; _cyb_driver_names[412] = b'cuCoredumpGetAttributeGlobal'
_cyb_driver_slots[413] = &__cuCoredumpSetAttribute; _cyb_driver_names[413] = b'cuCoredumpSetAttribute'
_cyb_driver_slots[414] = &__cuCoredumpSetAttributeGlobal; _cyb_driver_names[414] = b'cuCoredumpSetAttributeGlobal'
_cyb_driver_slots[415] = &__cuGetExportTable; _cyb_driver_names[415] = b'cuGetExportTable'
_cyb_driver_slots[416] = &__cuGreenCtxCreate; _cyb_driver_names[416] = b'cuGreenCtxCreate'
_cyb_driver_slots[417] = &__cuGreenCtxDestroy; _cyb_driver_names[417] = b'cuGreenCtxDestroy'
_cyb_driver_slots[418] = &__cuCtxFromGreenCtx; _cyb_driver_names[418] = b'cuCtxFromGreenCtx'
_cyb_driver_slots[419] = &__cuDeviceGetDevResource; _cyb_driver_names[419] = b'cuDeviceGetDevResource'
_cyb_driver_slots[420] = &__cuCtxGetDevResource; _cyb_driver_names[420] = b'cuCtxGetDevResource'
_cyb_driver_slots[421] = &__cuGreenCtxGetDevResource; _cyb_driver_names[421] = b'cuGreenCtxGetDevResource'
_cyb_driver_slots[422] = &__cuDevSmResourceSplitByCount; _cyb_driver_names[422] = b'cuDevSmResourceSplitByCount'
_cyb_driver_slots[423] = &__cuDevResourceGenerateDesc; _cyb_driver_names[423] = b'cuDevResourceGenerateDesc'
_cyb_driver_slots[424] = &__cuGreenCtxRecordEvent; _cyb_driver_names[424] = b'cuGreenCtxRecordEvent'
_cyb_driver_slots[425] = &__cuGreenCtxWaitEvent; _cyb_driver_names[425] = b'cuGreenCtxWaitEvent'
_cyb_driver_slots[426] = &__cuStreamGetGreenCtx; _cyb_driver_names[426] = b'cuStreamGetGreenCtx'
_cyb_driver_slots[427] = &__cuGreenCtxStreamCreate; _cyb_driver_names[427] = b'cuGreenCtxStreamCreate'
_cyb_driver_slots[428] = &__cuLogsRegisterCallback; _cyb_driver_names[428] = b'cuLogsRegisterCallback'
_cyb_driver_slots[429] = &__cuLogsUnregisterCallback; _cyb_driver_names[429] = b'cuLogsUnregisterCallback'
_cyb_driver_slots[430] = &__cuLogsCurrent; _cyb_driver_names[430] = b'cuLogsCurrent'
_cyb_driver_slots[431] = &__cuLogsDumpToFile; _cyb_driver_names[431] = b'cuLogsDumpToFile'
_cyb_driver_slots[432] = &__cuLogsDumpToMemory; _cyb_driver_names[432] = b'cuLogsDumpToMemory'
_cyb_driver_slots[433] = &__cuCheckpointProcessGetRestoreThreadId; _cyb_driver_names[433] = b'cuCheckpointProcessGetRestoreThreadId'
_cyb_driver_slots[434] = &__cuCheckpointProcessGetState; _cyb_driver_names[434] = b'cuCheckpointProcessGetState'
_cyb_driver_slots[435] = &__cuCheckpointProcessLock; _cyb_driver_names[435] = b'cuCheckpointProcessLock'
_cyb_driver_slots[436] = &__cuCheckpointProcessCheckpoint; _cyb_driver_names[436] = b'cuCheckpointProcessCheckpoint'
_cyb_driver_slots[437] = &__cuCheckpointProcessRestore; _cyb_driver_names[437] = b'cuCheckpointProcessRestore'
_cyb_driver_slots[438] = &__cuCheckpointProcessUnlock; _cyb_driver_names[438] = b'cuCheckpointProcessUnlock'
_cyb_driver_slots[439] = &__cuGraphicsEGLRegisterImage; _cyb_driver_names[439] = b'cuGraphicsEGLRegisterImage'
_cyb_driver_slots[440] = &__cuEGLStreamConsumerConnect; _cyb_driver_names[440] = b'cuEGLStreamConsumerConnect'
_cyb_driver_slots[441] = &__cuEGLStreamConsumerConnectWithFlags; _cyb_driver_names[441] = b'cuEGLStreamConsumerConnectWithFlags'
_cyb_driver_slots[442] = &__cuEGLStreamConsumerDisconnect; _cyb_driver_names[442] = b'cuEGLStreamConsumerDisconnect'
_cyb_driver_slots[443] = &__cuEGLStreamConsumerAcquireFrame; _cyb_driver_names[443] = b'cuEGLStreamConsumerAcquireFrame'
_cyb_driver_slots[444] = &__cuEGLStreamConsumerReleaseFrame; _cyb_driver_names[444] = b'cuEGLStreamConsumerReleaseFrame'
_cyb_driver_slots[445] = &__cuEGLStreamProducerConnect; _cyb_driver_names[445] = b'cuEGLStreamProducerConnect'
_cyb_driver_slots[446] = &__cuEGLStreamProducerDisconnect; _cyb_driver_names[446] = b'cuEGLStreamProducerDisconnect'
_cyb_driver_slots[447] = &__cuEGLStreamProducerPresentFrame; _cyb_driver_names[447] = b'cuEGLStreamProducerPresentFrame'
_cyb_driver_slots[448] = &__cuEGLStreamProducerReturnFrame; _cyb_driver_names[448] = b'cuEGLStreamProducerReturnFrame'
_cyb_driver_slots[449] = &__cuGraphicsResourceGetMappedEglFrame; _cyb_driver_names[449] = b'cuGraphicsResourceGetMappedEglFrame'
_cyb_driver_slots[450] = &__cuEventCreateFromEGLSync; _cyb_driver_names[450] = b'cuEventCreateFromEGLSync'
_cyb_driver_slots[451] = &__cuGraphicsGLRegisterBuffer; _cyb_driver_names[451] = b'cuGraphicsGLRegisterBuffer'
_cyb_driver_slots[452] = &__cuGraphicsGLRegisterImage; _cyb_driver_names[452] = b'cuGraphicsGLRegisterImage'
_cyb_driver_slots[453] = &__cuGLGetDevices_v2; _cyb_driver_names[453] = b'cuGLGetDevices_v2'
_cyb_driver_slots[454] = &__cuGLCtxCreate_v2; _cyb_driver_names[454] = b'cuGLCtxCreate_v2'
_cyb_driver_slots[455] = &__cuGLInit; _cyb_driver_names[455] = b'cuGLInit'
_cyb_driver_slots[456] = &__cuGLRegisterBufferObject; _cyb_driver_names[456] = b'cuGLRegisterBufferObject'
_cyb_driver_slots[457] = &__cuGLMapBufferObject_v2; _cyb_driver_names[457] = b'cuGLMapBufferObject_v2'
_cyb_driver_slots[458] = &__cuGLUnmapBufferObject; _cyb_driver_names[458] = b'cuGLUnmapBufferObject'
_cyb_driver_slots[459] = &__cuGLUnregisterBufferObject; _cyb_driver_names[459] = b'cuGLUnregisterBufferObject'
_cyb_driver_slots[460] = &__cuGLSetBufferObjectMapFlags; _cyb_driver_names[460] = b'cuGLSetBufferObjectMapFlags'
_cyb_driver_slots[461] = &__cuGLMapBufferObjectAsync_v2; _cyb_driver_names[461] = b'cuGLMapBufferObjectAsync_v2'
_cyb_driver_slots[462] = &__cuGLUnmapBufferObjectAsync; _cyb_driver_names[462] = b'cuGLUnmapBufferObjectAsync'
_cyb_driver_slots[463] = &__cuProfilerInitialize; _cyb_driver_names[463] = b'cuProfilerInitialize'
_cyb_driver_slots[464] = &__cuProfilerStart; _cyb_driver_names[464] = b'cuProfilerStart'
_cyb_driver_slots[465] = &__cuProfilerStop; _cyb_driver_names[465] = b'cuProfilerStop'
_cyb_driver_slots[466] = &__cuVDPAUGetDevice; _cyb_driver_names[466] = b'cuVDPAUGetDevice'
_cyb_driver_slots[467] = &__cuVDPAUCtxCreate_v2; _cyb_driver_names[467] = b'cuVDPAUCtxCreate_v2'
_cyb_driver_slots[468] = &__cuGraphicsVDPAURegisterVideoSurface; _cyb_driver_names[468] = b'cuGraphicsVDPAURegisterVideoSurface'
_cyb_driver_slots[469] = &__cuGraphicsVDPAURegisterOutputSurface; _cyb_driver_names[469] = b'cuGraphicsVDPAURegisterOutputSurface'
_cyb_driver_slots[470] = &__cuDeviceGetHostAtomicCapabilities; _cyb_driver_names[470] = b'cuDeviceGetHostAtomicCapabilities'
_cyb_driver_slots[471] = &__cuCtxGetDevice_v2; _cyb_driver_names[471] = b'cuCtxGetDevice_v2'
_cyb_driver_slots[472] = &__cuCtxSynchronize_v2; _cyb_driver_names[472] = b'cuCtxSynchronize_v2'
_cyb_driver_slots[473] = &__cuMemcpyBatchAsync_v2; _cyb_driver_names[473] = b'cuMemcpyBatchAsync_v2'
_cyb_driver_slots[474] = &__cuMemcpy3DBatchAsync_v2; _cyb_driver_names[474] = b'cuMemcpy3DBatchAsync_v2'
_cyb_driver_slots[475] = &__cuMemGetDefaultMemPool; _cyb_driver_names[475] = b'cuMemGetDefaultMemPool'
_cyb_driver_slots[476] = &__cuMemGetMemPool; _cyb_driver_names[476] = b'cuMemGetMemPool'
_cyb_driver_slots[477] = &__cuMemSetMemPool; _cyb_driver_names[477] = b'cuMemSetMemPool'
_cyb_driver_slots[478] = &__cuMemPrefetchBatchAsync; _cyb_driver_names[478] = b'cuMemPrefetchBatchAsync'
_cyb_driver_slots[479] = &__cuMemDiscardBatchAsync; _cyb_driver_names[479] = b'cuMemDiscardBatchAsync'
_cyb_driver_slots[480] = &__cuMemDiscardAndPrefetchBatchAsync; _cyb_driver_names[480] = b'cuMemDiscardAndPrefetchBatchAsync'
_cyb_driver_slots[481] = &__cuDeviceGetP2PAtomicCapabilities; _cyb_driver_names[481] = b'cuDeviceGetP2PAtomicCapabilities'
_cyb_driver_slots[482] = &__cuGreenCtxGetId; _cyb_driver_names[482] = b'cuGreenCtxGetId'
_cyb_driver_slots[483] = &__cuMulticastBindMem_v2; _cyb_driver_names[483] = b'cuMulticastBindMem_v2'
_cyb_driver_slots[484] = &__cuMulticastBindAddr_v2; _cyb_driver_names[484] = b'cuMulticastBindAddr_v2'
_cyb_driver_slots[485] = &__cuGraphNodeGetContainingGraph; _cyb_driver_names[485] = b'cuGraphNodeGetContainingGraph'
_cyb_driver_slots[486] = &__cuGraphNodeGetLocalId; _cyb_driver_names[486] = b'cuGraphNodeGetLocalId'
_cyb_driver_slots[487] = &__cuGraphNodeGetToolsId; _cyb_driver_names[487] = b'cuGraphNodeGetToolsId'
_cyb_driver_slots[488] = &__cuGraphGetId; _cyb_driver_names[488] = b'cuGraphGetId'
_cyb_driver_slots[489] = &__cuGraphExecGetId; _cyb_driver_names[489] = b'cuGraphExecGetId'
_cyb_driver_slots[490] = &__cuDevSmResourceSplit; _cyb_driver_names[490] = b'cuDevSmResourceSplit'
_cyb_driver_slots[491] = &__cuStreamGetDevResource; _cyb_driver_names[491] = b'cuStreamGetDevResource'
_cyb_driver_slots[492] = &__cuKernelGetParamCount; _cyb_driver_names[492] = b'cuKernelGetParamCount'
_cyb_driver_slots[493] = &__cuMemcpyWithAttributesAsync; _cyb_driver_names[493] = b'cuMemcpyWithAttributesAsync'
_cyb_driver_slots[494] = &__cuMemcpy3DWithAttributesAsync; _cyb_driver_names[494] = b'cuMemcpy3DWithAttributesAsync'
_cyb_driver_slots[495] = &__cuStreamBeginCaptureToCig; _cyb_driver_names[495] = b'cuStreamBeginCaptureToCig'
_cyb_driver_slots[496] = &__cuStreamEndCaptureToCig; _cyb_driver_names[496] = b'cuStreamEndCaptureToCig'
_cyb_driver_slots[497] = &__cuFuncGetParamCount; _cyb_driver_names[497] = b'cuFuncGetParamCount'
_cyb_driver_slots[498] = &__cuLaunchHostFunc_v2; _cyb_driver_names[498] = b'cuLaunchHostFunc_v2'
_cyb_driver_slots[499] = &__cuGraphNodeGetParams; _cyb_driver_names[499] = b'cuGraphNodeGetParams'
_cyb_driver_slots[500] = &__cuCoredumpRegisterStartCallback; _cyb_driver_names[500] = b'cuCoredumpRegisterStartCallback'
_cyb_driver_slots[501] = &__cuCoredumpRegisterCompleteCallback; _cyb_driver_names[501] = b'cuCoredumpRegisterCompleteCallback'
_cyb_driver_slots[502] = &__cuCoredumpDeregisterStartCallback; _cyb_driver_names[502] = b'cuCoredumpDeregisterStartCallback'
_cyb_driver_slots[503] = &__cuCoredumpDeregisterCompleteCallback; _cyb_driver_names[503] = b'cuCoredumpDeregisterCompleteCallback'
_cyb_driver_slots[504] = &__cuLogicalEndpointIdReserve; _cyb_driver_names[504] = b'cuLogicalEndpointIdReserve'
_cyb_driver_slots[505] = &__cuLogicalEndpointIdRelease; _cyb_driver_names[505] = b'cuLogicalEndpointIdRelease'
_cyb_driver_slots[506] = &__cuLogicalEndpointCreate; _cyb_driver_names[506] = b'cuLogicalEndpointCreate'
_cyb_driver_slots[507] = &__cuLogicalEndpointAddDevice; _cyb_driver_names[507] = b'cuLogicalEndpointAddDevice'
_cyb_driver_slots[508] = &__cuLogicalEndpointDestroy; _cyb_driver_names[508] = b'cuLogicalEndpointDestroy'
_cyb_driver_slots[509] = &__cuLogicalEndpointBindAddr; _cyb_driver_names[509] = b'cuLogicalEndpointBindAddr'
_cyb_driver_slots[510] = &__cuLogicalEndpointBindMem; _cyb_driver_names[510] = b'cuLogicalEndpointBindMem'
_cyb_driver_slots[511] = &__cuLogicalEndpointUnbind; _cyb_driver_names[511] = b'cuLogicalEndpointUnbind'
_cyb_driver_slots[512] = &__cuLogicalEndpointExport; _cyb_driver_names[512] = b'cuLogicalEndpointExport'
_cyb_driver_slots[513] = &__cuLogicalEndpointImport; _cyb_driver_names[513] = b'cuLogicalEndpointImport'
_cyb_driver_slots[514] = &__cuLogicalEndpointGetLimits; _cyb_driver_names[514] = b'cuLogicalEndpointGetLimits'
_cyb_driver_slots[515] = &__cuLogicalEndpointQuery; _cyb_driver_names[515] = b'cuLogicalEndpointQuery'
_cyb_driver_slots[516] = &__cuStreamBeginRecaptureToGraph; _cyb_driver_names[516] = b'cuStreamBeginRecaptureToGraph'
_cyb_driver_slots[517] = &__cuDeviceGetFabricClusterUuid; _cyb_driver_names[517] = b'cuDeviceGetFabricClusterUuid'
_cyb_driver_slots[518] = &__cuDeviceGetCliqueCount; _cyb_driver_names[518] = b'cuDeviceGetCliqueCount'
_cyb_driver_slots[519] = &__cuDeviceGetCliqueInfo; _cyb_driver_names[519] = b'cuDeviceGetCliqueInfo'
_cyb_driver_slots[520] = &__cuMemGetLocationInfo; _cyb_driver_names[520] = b'cuMemGetLocationInfo'
_cyb_driver_slots[521] = &__cuGraphAddNode_v3; _cyb_driver_names[521] = b'cuGraphAddNode_v3'
_cyb_driver_slots[522] = &__cuGraphNodeSetParams_v2; _cyb_driver_names[522] = b'cuGraphNodeSetParams_v2'
_cyb_driver_slots[523] = &__cuCheckpointOperationComplete; _cyb_driver_names[523] = b'cuCheckpointOperationComplete'
_cyb_driver_slots[524] = &__cuFabricCpyBatchAsync; _cyb_driver_names[524] = b'cuFabricCpyBatchAsync'
_cyb_driver_slots[525] = &__cuTensorMapSetAllocationClass; _cyb_driver_names[525] = b'cuTensorMapSetAllocationClass'

cdef int _init_driver() except -1 nogil:
    global _cyb___py_driver_init
    cdef uintptr_t handle = 0
    cdef int ptds_mode
    cdef _cyb_cuGetProcAddress_v2_T cuGetProcAddress_v2
    with gil, _cyb_symbol_lock:
        if _cyb___py_driver_init: return 0

        handle = load_library()
        if handle == 0:
            raise RuntimeError('Failed to open cuda')
        # Get latest __cuGetProcAddress_v2
        cuGetProcAddress_v2 = <_cyb_cuGetProcAddress_v2_T>_cyb_GetProcAddress(
            <HMODULE>handle, 'cuGetProcAddress_v2'
        )
        if cuGetProcAddress_v2 == NULL:
            raise RuntimeError("Failed to get cuGetProcAddress_v2")
        if bool(int(_cyb_getenv('CUDA_PYTHON_CUDA_PER_THREAD_DEFAULT_STREAM', default=0))):
            ptds_mode = CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM
        else:
            ptds_mode = CU_GET_PROC_ADDRESS_DEFAULT
        global __cuGetErrorString
        cuGetProcAddress_v2('cuGetErrorString', <void **>&__cuGetErrorString, 6000, ptds_mode, NULL)

        global __cuGetErrorName
        cuGetProcAddress_v2('cuGetErrorName', <void **>&__cuGetErrorName, 6000, ptds_mode, NULL)

        global __cuInit
        cuGetProcAddress_v2('cuInit', <void **>&__cuInit, 2000, ptds_mode, NULL)

        global __cuDriverGetVersion
        cuGetProcAddress_v2('cuDriverGetVersion', <void **>&__cuDriverGetVersion, 2020, ptds_mode, NULL)

        global __cuDeviceGet
        cuGetProcAddress_v2('cuDeviceGet', <void **>&__cuDeviceGet, 2000, ptds_mode, NULL)

        global __cuDeviceGetCount
        cuGetProcAddress_v2('cuDeviceGetCount', <void **>&__cuDeviceGetCount, 2000, ptds_mode, NULL)

        global __cuDeviceGetName
        cuGetProcAddress_v2('cuDeviceGetName', <void **>&__cuDeviceGetName, 2000, ptds_mode, NULL)

        global __cuDeviceGetUuid_v2
        cuGetProcAddress_v2('cuDeviceGetUuid', <void **>&__cuDeviceGetUuid_v2, 11040, ptds_mode, NULL)

        global __cuDeviceGetLuid
        cuGetProcAddress_v2('cuDeviceGetLuid', <void **>&__cuDeviceGetLuid, 10000, ptds_mode, NULL)

        global __cuDeviceTotalMem_v2
        cuGetProcAddress_v2('cuDeviceTotalMem', <void **>&__cuDeviceTotalMem_v2, 3020, ptds_mode, NULL)

        global __cuDeviceGetTexture1DLinearMaxWidth
        cuGetProcAddress_v2('cuDeviceGetTexture1DLinearMaxWidth', <void **>&__cuDeviceGetTexture1DLinearMaxWidth, 11010, ptds_mode, NULL)

        global __cuDeviceGetAttribute
        cuGetProcAddress_v2('cuDeviceGetAttribute', <void **>&__cuDeviceGetAttribute, 2000, ptds_mode, NULL)

        global __cuDeviceGetNvSciSyncAttributes
        cuGetProcAddress_v2('cuDeviceGetNvSciSyncAttributes', <void **>&__cuDeviceGetNvSciSyncAttributes, 10020, ptds_mode, NULL)

        global __cuDeviceSetMemPool
        cuGetProcAddress_v2('cuDeviceSetMemPool', <void **>&__cuDeviceSetMemPool, 11020, ptds_mode, NULL)

        global __cuDeviceGetMemPool
        cuGetProcAddress_v2('cuDeviceGetMemPool', <void **>&__cuDeviceGetMemPool, 11020, ptds_mode, NULL)

        global __cuDeviceGetDefaultMemPool
        cuGetProcAddress_v2('cuDeviceGetDefaultMemPool', <void **>&__cuDeviceGetDefaultMemPool, 11020, ptds_mode, NULL)

        global __cuDeviceGetExecAffinitySupport
        cuGetProcAddress_v2('cuDeviceGetExecAffinitySupport', <void **>&__cuDeviceGetExecAffinitySupport, 11040, ptds_mode, NULL)

        global __cuFlushGPUDirectRDMAWrites
        cuGetProcAddress_v2('cuFlushGPUDirectRDMAWrites', <void **>&__cuFlushGPUDirectRDMAWrites, 11030, ptds_mode, NULL)

        global __cuDeviceGetProperties
        cuGetProcAddress_v2('cuDeviceGetProperties', <void **>&__cuDeviceGetProperties, 2000, ptds_mode, NULL)

        global __cuDeviceComputeCapability
        cuGetProcAddress_v2('cuDeviceComputeCapability', <void **>&__cuDeviceComputeCapability, 2000, ptds_mode, NULL)

        global __cuDevicePrimaryCtxRetain
        cuGetProcAddress_v2('cuDevicePrimaryCtxRetain', <void **>&__cuDevicePrimaryCtxRetain, 7000, ptds_mode, NULL)

        global __cuDevicePrimaryCtxRelease_v2
        cuGetProcAddress_v2('cuDevicePrimaryCtxRelease', <void **>&__cuDevicePrimaryCtxRelease_v2, 11000, ptds_mode, NULL)

        global __cuDevicePrimaryCtxSetFlags_v2
        cuGetProcAddress_v2('cuDevicePrimaryCtxSetFlags', <void **>&__cuDevicePrimaryCtxSetFlags_v2, 11000, ptds_mode, NULL)

        global __cuDevicePrimaryCtxGetState
        cuGetProcAddress_v2('cuDevicePrimaryCtxGetState', <void **>&__cuDevicePrimaryCtxGetState, 7000, ptds_mode, NULL)

        global __cuDevicePrimaryCtxReset_v2
        cuGetProcAddress_v2('cuDevicePrimaryCtxReset', <void **>&__cuDevicePrimaryCtxReset_v2, 11000, ptds_mode, NULL)

        global __cuCtxCreate_v2
        cuGetProcAddress_v2('cuCtxCreate', <void **>&__cuCtxCreate_v2, 3020, ptds_mode, NULL)

        global __cuCtxCreate_v3
        cuGetProcAddress_v2('cuCtxCreate', <void **>&__cuCtxCreate_v3, 11040, ptds_mode, NULL)

        global __cuCtxCreate_v4
        cuGetProcAddress_v2('cuCtxCreate', <void **>&__cuCtxCreate_v4, 12050, ptds_mode, NULL)

        global __cuCtxDestroy_v2
        cuGetProcAddress_v2('cuCtxDestroy', <void **>&__cuCtxDestroy_v2, 4000, ptds_mode, NULL)

        global __cuCtxPushCurrent_v2
        cuGetProcAddress_v2('cuCtxPushCurrent', <void **>&__cuCtxPushCurrent_v2, 4000, ptds_mode, NULL)

        global __cuCtxPopCurrent_v2
        cuGetProcAddress_v2('cuCtxPopCurrent', <void **>&__cuCtxPopCurrent_v2, 4000, ptds_mode, NULL)

        global __cuCtxSetCurrent
        cuGetProcAddress_v2('cuCtxSetCurrent', <void **>&__cuCtxSetCurrent, 4000, ptds_mode, NULL)

        global __cuCtxGetCurrent
        cuGetProcAddress_v2('cuCtxGetCurrent', <void **>&__cuCtxGetCurrent, 4000, ptds_mode, NULL)

        global __cuCtxGetDevice
        cuGetProcAddress_v2('cuCtxGetDevice', <void **>&__cuCtxGetDevice, 2000, ptds_mode, NULL)

        global __cuCtxGetFlags
        cuGetProcAddress_v2('cuCtxGetFlags', <void **>&__cuCtxGetFlags, 7000, ptds_mode, NULL)

        global __cuCtxSetFlags
        cuGetProcAddress_v2('cuCtxSetFlags', <void **>&__cuCtxSetFlags, 12010, ptds_mode, NULL)

        global __cuCtxGetId
        cuGetProcAddress_v2('cuCtxGetId', <void **>&__cuCtxGetId, 12000, ptds_mode, NULL)

        global __cuCtxSynchronize
        cuGetProcAddress_v2('cuCtxSynchronize', <void **>&__cuCtxSynchronize, 2000, ptds_mode, NULL)

        global __cuCtxSetLimit
        cuGetProcAddress_v2('cuCtxSetLimit', <void **>&__cuCtxSetLimit, 3010, ptds_mode, NULL)

        global __cuCtxGetLimit
        cuGetProcAddress_v2('cuCtxGetLimit', <void **>&__cuCtxGetLimit, 3010, ptds_mode, NULL)

        global __cuCtxGetCacheConfig
        cuGetProcAddress_v2('cuCtxGetCacheConfig', <void **>&__cuCtxGetCacheConfig, 3020, ptds_mode, NULL)

        global __cuCtxSetCacheConfig
        cuGetProcAddress_v2('cuCtxSetCacheConfig', <void **>&__cuCtxSetCacheConfig, 3020, ptds_mode, NULL)

        global __cuCtxGetApiVersion
        cuGetProcAddress_v2('cuCtxGetApiVersion', <void **>&__cuCtxGetApiVersion, 3020, ptds_mode, NULL)

        global __cuCtxGetStreamPriorityRange
        cuGetProcAddress_v2('cuCtxGetStreamPriorityRange', <void **>&__cuCtxGetStreamPriorityRange, 5050, ptds_mode, NULL)

        global __cuCtxResetPersistingL2Cache
        cuGetProcAddress_v2('cuCtxResetPersistingL2Cache', <void **>&__cuCtxResetPersistingL2Cache, 11000, ptds_mode, NULL)

        global __cuCtxGetExecAffinity
        cuGetProcAddress_v2('cuCtxGetExecAffinity', <void **>&__cuCtxGetExecAffinity, 11040, ptds_mode, NULL)

        global __cuCtxRecordEvent
        cuGetProcAddress_v2('cuCtxRecordEvent', <void **>&__cuCtxRecordEvent, 12050, ptds_mode, NULL)

        global __cuCtxWaitEvent
        cuGetProcAddress_v2('cuCtxWaitEvent', <void **>&__cuCtxWaitEvent, 12050, ptds_mode, NULL)

        global __cuCtxAttach
        cuGetProcAddress_v2('cuCtxAttach', <void **>&__cuCtxAttach, 2000, ptds_mode, NULL)

        global __cuCtxDetach
        cuGetProcAddress_v2('cuCtxDetach', <void **>&__cuCtxDetach, 2000, ptds_mode, NULL)

        global __cuCtxGetSharedMemConfig
        cuGetProcAddress_v2('cuCtxGetSharedMemConfig', <void **>&__cuCtxGetSharedMemConfig, 4020, ptds_mode, NULL)

        global __cuCtxSetSharedMemConfig
        cuGetProcAddress_v2('cuCtxSetSharedMemConfig', <void **>&__cuCtxSetSharedMemConfig, 4020, ptds_mode, NULL)

        global __cuModuleLoad
        cuGetProcAddress_v2('cuModuleLoad', <void **>&__cuModuleLoad, 2000, ptds_mode, NULL)

        global __cuModuleLoadData
        cuGetProcAddress_v2('cuModuleLoadData', <void **>&__cuModuleLoadData, 2000, ptds_mode, NULL)

        global __cuModuleLoadDataEx
        cuGetProcAddress_v2('cuModuleLoadDataEx', <void **>&__cuModuleLoadDataEx, 2010, ptds_mode, NULL)

        global __cuModuleLoadFatBinary
        cuGetProcAddress_v2('cuModuleLoadFatBinary', <void **>&__cuModuleLoadFatBinary, 2000, ptds_mode, NULL)

        global __cuModuleUnload
        cuGetProcAddress_v2('cuModuleUnload', <void **>&__cuModuleUnload, 2000, ptds_mode, NULL)

        global __cuModuleGetLoadingMode
        cuGetProcAddress_v2('cuModuleGetLoadingMode', <void **>&__cuModuleGetLoadingMode, 11070, ptds_mode, NULL)

        global __cuModuleGetFunction
        cuGetProcAddress_v2('cuModuleGetFunction', <void **>&__cuModuleGetFunction, 2000, ptds_mode, NULL)

        global __cuModuleGetFunctionCount
        cuGetProcAddress_v2('cuModuleGetFunctionCount', <void **>&__cuModuleGetFunctionCount, 12040, ptds_mode, NULL)

        global __cuModuleEnumerateFunctions
        cuGetProcAddress_v2('cuModuleEnumerateFunctions', <void **>&__cuModuleEnumerateFunctions, 12040, ptds_mode, NULL)

        global __cuModuleGetGlobal_v2
        cuGetProcAddress_v2('cuModuleGetGlobal', <void **>&__cuModuleGetGlobal_v2, 3020, ptds_mode, NULL)

        global __cuLinkCreate_v2
        cuGetProcAddress_v2('cuLinkCreate', <void **>&__cuLinkCreate_v2, 6050, ptds_mode, NULL)

        global __cuLinkAddData_v2
        cuGetProcAddress_v2('cuLinkAddData', <void **>&__cuLinkAddData_v2, 6050, ptds_mode, NULL)

        global __cuLinkAddFile_v2
        cuGetProcAddress_v2('cuLinkAddFile', <void **>&__cuLinkAddFile_v2, 6050, ptds_mode, NULL)

        global __cuLinkComplete
        cuGetProcAddress_v2('cuLinkComplete', <void **>&__cuLinkComplete, 5050, ptds_mode, NULL)

        global __cuLinkDestroy
        cuGetProcAddress_v2('cuLinkDestroy', <void **>&__cuLinkDestroy, 5050, ptds_mode, NULL)

        global __cuModuleGetTexRef
        cuGetProcAddress_v2('cuModuleGetTexRef', <void **>&__cuModuleGetTexRef, 2000, ptds_mode, NULL)

        global __cuModuleGetSurfRef
        cuGetProcAddress_v2('cuModuleGetSurfRef', <void **>&__cuModuleGetSurfRef, 3000, ptds_mode, NULL)

        global __cuLibraryLoadData
        cuGetProcAddress_v2('cuLibraryLoadData', <void **>&__cuLibraryLoadData, 12000, ptds_mode, NULL)

        global __cuLibraryLoadFromFile
        cuGetProcAddress_v2('cuLibraryLoadFromFile', <void **>&__cuLibraryLoadFromFile, 12000, ptds_mode, NULL)

        global __cuLibraryUnload
        cuGetProcAddress_v2('cuLibraryUnload', <void **>&__cuLibraryUnload, 12000, ptds_mode, NULL)

        global __cuLibraryGetKernel
        cuGetProcAddress_v2('cuLibraryGetKernel', <void **>&__cuLibraryGetKernel, 12000, ptds_mode, NULL)

        global __cuLibraryGetKernelCount
        cuGetProcAddress_v2('cuLibraryGetKernelCount', <void **>&__cuLibraryGetKernelCount, 12040, ptds_mode, NULL)

        global __cuLibraryEnumerateKernels
        cuGetProcAddress_v2('cuLibraryEnumerateKernels', <void **>&__cuLibraryEnumerateKernels, 12040, ptds_mode, NULL)

        global __cuLibraryGetModule
        cuGetProcAddress_v2('cuLibraryGetModule', <void **>&__cuLibraryGetModule, 12000, ptds_mode, NULL)

        global __cuKernelGetFunction
        cuGetProcAddress_v2('cuKernelGetFunction', <void **>&__cuKernelGetFunction, 12000, ptds_mode, NULL)

        global __cuKernelGetLibrary
        cuGetProcAddress_v2('cuKernelGetLibrary', <void **>&__cuKernelGetLibrary, 12050, ptds_mode, NULL)

        global __cuLibraryGetGlobal
        cuGetProcAddress_v2('cuLibraryGetGlobal', <void **>&__cuLibraryGetGlobal, 12000, ptds_mode, NULL)

        global __cuLibraryGetManaged
        cuGetProcAddress_v2('cuLibraryGetManaged', <void **>&__cuLibraryGetManaged, 12000, ptds_mode, NULL)

        global __cuLibraryGetUnifiedFunction
        cuGetProcAddress_v2('cuLibraryGetUnifiedFunction', <void **>&__cuLibraryGetUnifiedFunction, 12000, ptds_mode, NULL)

        global __cuKernelGetAttribute
        cuGetProcAddress_v2('cuKernelGetAttribute', <void **>&__cuKernelGetAttribute, 12000, ptds_mode, NULL)

        global __cuKernelSetAttribute
        cuGetProcAddress_v2('cuKernelSetAttribute', <void **>&__cuKernelSetAttribute, 12000, ptds_mode, NULL)

        global __cuKernelSetCacheConfig
        cuGetProcAddress_v2('cuKernelSetCacheConfig', <void **>&__cuKernelSetCacheConfig, 12000, ptds_mode, NULL)

        global __cuKernelGetName
        cuGetProcAddress_v2('cuKernelGetName', <void **>&__cuKernelGetName, 12030, ptds_mode, NULL)

        global __cuKernelGetParamInfo
        cuGetProcAddress_v2('cuKernelGetParamInfo', <void **>&__cuKernelGetParamInfo, 12040, ptds_mode, NULL)

        global __cuMemGetInfo_v2
        cuGetProcAddress_v2('cuMemGetInfo', <void **>&__cuMemGetInfo_v2, 3020, ptds_mode, NULL)

        global __cuMemAlloc_v2
        cuGetProcAddress_v2('cuMemAlloc', <void **>&__cuMemAlloc_v2, 3020, ptds_mode, NULL)

        global __cuMemAllocPitch_v2
        cuGetProcAddress_v2('cuMemAllocPitch', <void **>&__cuMemAllocPitch_v2, 3020, ptds_mode, NULL)

        global __cuMemFree_v2
        cuGetProcAddress_v2('cuMemFree', <void **>&__cuMemFree_v2, 3020, ptds_mode, NULL)

        global __cuMemGetAddressRange_v2
        cuGetProcAddress_v2('cuMemGetAddressRange', <void **>&__cuMemGetAddressRange_v2, 3020, ptds_mode, NULL)

        global __cuMemAllocHost_v2
        cuGetProcAddress_v2('cuMemAllocHost', <void **>&__cuMemAllocHost_v2, 3020, ptds_mode, NULL)

        global __cuMemFreeHost
        cuGetProcAddress_v2('cuMemFreeHost', <void **>&__cuMemFreeHost, 2000, ptds_mode, NULL)

        global __cuMemHostAlloc
        cuGetProcAddress_v2('cuMemHostAlloc', <void **>&__cuMemHostAlloc, 2020, ptds_mode, NULL)

        global __cuMemHostGetDevicePointer_v2
        cuGetProcAddress_v2('cuMemHostGetDevicePointer', <void **>&__cuMemHostGetDevicePointer_v2, 3020, ptds_mode, NULL)

        global __cuMemHostGetFlags
        cuGetProcAddress_v2('cuMemHostGetFlags', <void **>&__cuMemHostGetFlags, 2030, ptds_mode, NULL)

        global __cuMemAllocManaged
        cuGetProcAddress_v2('cuMemAllocManaged', <void **>&__cuMemAllocManaged, 6000, ptds_mode, NULL)

        global __cuDeviceRegisterAsyncNotification
        cuGetProcAddress_v2('cuDeviceRegisterAsyncNotification', <void **>&__cuDeviceRegisterAsyncNotification, 12040, ptds_mode, NULL)

        global __cuDeviceUnregisterAsyncNotification
        cuGetProcAddress_v2('cuDeviceUnregisterAsyncNotification', <void **>&__cuDeviceUnregisterAsyncNotification, 12040, ptds_mode, NULL)

        global __cuDeviceGetByPCIBusId
        cuGetProcAddress_v2('cuDeviceGetByPCIBusId', <void **>&__cuDeviceGetByPCIBusId, 4010, ptds_mode, NULL)

        global __cuDeviceGetPCIBusId
        cuGetProcAddress_v2('cuDeviceGetPCIBusId', <void **>&__cuDeviceGetPCIBusId, 4010, ptds_mode, NULL)

        global __cuIpcGetEventHandle
        cuGetProcAddress_v2('cuIpcGetEventHandle', <void **>&__cuIpcGetEventHandle, 4010, ptds_mode, NULL)

        global __cuIpcOpenEventHandle
        cuGetProcAddress_v2('cuIpcOpenEventHandle', <void **>&__cuIpcOpenEventHandle, 4010, ptds_mode, NULL)

        global __cuIpcGetMemHandle
        cuGetProcAddress_v2('cuIpcGetMemHandle', <void **>&__cuIpcGetMemHandle, 4010, ptds_mode, NULL)

        global __cuIpcOpenMemHandle_v2
        cuGetProcAddress_v2('cuIpcOpenMemHandle', <void **>&__cuIpcOpenMemHandle_v2, 11000, ptds_mode, NULL)

        global __cuIpcCloseMemHandle
        cuGetProcAddress_v2('cuIpcCloseMemHandle', <void **>&__cuIpcCloseMemHandle, 4010, ptds_mode, NULL)

        global __cuMemHostRegister_v2
        cuGetProcAddress_v2('cuMemHostRegister', <void **>&__cuMemHostRegister_v2, 6050, ptds_mode, NULL)

        global __cuMemHostUnregister
        cuGetProcAddress_v2('cuMemHostUnregister', <void **>&__cuMemHostUnregister, 4000, ptds_mode, NULL)

        global __cuMemcpy
        cuGetProcAddress_v2('cuMemcpy', <void **>&__cuMemcpy, 7000 if ptds_mode == CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM else 4000, ptds_mode, NULL)

        global __cuMemcpyPeer
        cuGetProcAddress_v2('cuMemcpyPeer', <void **>&__cuMemcpyPeer, 7000 if ptds_mode == CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM else 4000, ptds_mode, NULL)

        global __cuMemcpyHtoD_v2
        cuGetProcAddress_v2('cuMemcpyHtoD', <void **>&__cuMemcpyHtoD_v2, 7000 if ptds_mode == CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM else 3020, ptds_mode, NULL)

        global __cuMemcpyDtoH_v2
        cuGetProcAddress_v2('cuMemcpyDtoH', <void **>&__cuMemcpyDtoH_v2, 7000 if ptds_mode == CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM else 3020, ptds_mode, NULL)

        global __cuMemcpyDtoD_v2
        cuGetProcAddress_v2('cuMemcpyDtoD', <void **>&__cuMemcpyDtoD_v2, 7000 if ptds_mode == CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM else 3020, ptds_mode, NULL)

        global __cuMemcpyDtoA_v2
        cuGetProcAddress_v2('cuMemcpyDtoA', <void **>&__cuMemcpyDtoA_v2, 7000 if ptds_mode == CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM else 3020, ptds_mode, NULL)

        global __cuMemcpyAtoD_v2
        cuGetProcAddress_v2('cuMemcpyAtoD', <void **>&__cuMemcpyAtoD_v2, 7000 if ptds_mode == CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM else 3020, ptds_mode, NULL)

        global __cuMemcpyHtoA_v2
        cuGetProcAddress_v2('cuMemcpyHtoA', <void **>&__cuMemcpyHtoA_v2, 7000 if ptds_mode == CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM else 3020, ptds_mode, NULL)

        global __cuMemcpyAtoH_v2
        cuGetProcAddress_v2('cuMemcpyAtoH', <void **>&__cuMemcpyAtoH_v2, 7000 if ptds_mode == CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM else 3020, ptds_mode, NULL)

        global __cuMemcpyAtoA_v2
        cuGetProcAddress_v2('cuMemcpyAtoA', <void **>&__cuMemcpyAtoA_v2, 7000 if ptds_mode == CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM else 3020, ptds_mode, NULL)

        global __cuMemcpy2D_v2
        cuGetProcAddress_v2('cuMemcpy2D', <void **>&__cuMemcpy2D_v2, 7000 if ptds_mode == CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM else 3020, ptds_mode, NULL)

        global __cuMemcpy2DUnaligned_v2
        cuGetProcAddress_v2('cuMemcpy2DUnaligned', <void **>&__cuMemcpy2DUnaligned_v2, 7000 if ptds_mode == CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM else 3020, ptds_mode, NULL)

        global __cuMemcpy3D_v2
        cuGetProcAddress_v2('cuMemcpy3D', <void **>&__cuMemcpy3D_v2, 7000 if ptds_mode == CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM else 3020, ptds_mode, NULL)

        global __cuMemcpy3DPeer
        cuGetProcAddress_v2('cuMemcpy3DPeer', <void **>&__cuMemcpy3DPeer, 7000 if ptds_mode == CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM else 4000, ptds_mode, NULL)

        global __cuMemcpyAsync
        cuGetProcAddress_v2('cuMemcpyAsync', <void **>&__cuMemcpyAsync, 7000 if ptds_mode == CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM else 4000, ptds_mode, NULL)

        global __cuMemcpyPeerAsync
        cuGetProcAddress_v2('cuMemcpyPeerAsync', <void **>&__cuMemcpyPeerAsync, 7000 if ptds_mode == CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM else 4000, ptds_mode, NULL)

        global __cuMemcpyHtoDAsync_v2
        cuGetProcAddress_v2('cuMemcpyHtoDAsync', <void **>&__cuMemcpyHtoDAsync_v2, 7000 if ptds_mode == CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM else 3020, ptds_mode, NULL)

        global __cuMemcpyDtoHAsync_v2
        cuGetProcAddress_v2('cuMemcpyDtoHAsync', <void **>&__cuMemcpyDtoHAsync_v2, 7000 if ptds_mode == CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM else 3020, ptds_mode, NULL)

        global __cuMemcpyDtoDAsync_v2
        cuGetProcAddress_v2('cuMemcpyDtoDAsync', <void **>&__cuMemcpyDtoDAsync_v2, 7000 if ptds_mode == CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM else 3020, ptds_mode, NULL)

        global __cuMemcpyHtoAAsync_v2
        cuGetProcAddress_v2('cuMemcpyHtoAAsync', <void **>&__cuMemcpyHtoAAsync_v2, 7000 if ptds_mode == CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM else 3020, ptds_mode, NULL)

        global __cuMemcpyAtoHAsync_v2
        cuGetProcAddress_v2('cuMemcpyAtoHAsync', <void **>&__cuMemcpyAtoHAsync_v2, 7000 if ptds_mode == CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM else 3020, ptds_mode, NULL)

        global __cuMemcpy2DAsync_v2
        cuGetProcAddress_v2('cuMemcpy2DAsync', <void **>&__cuMemcpy2DAsync_v2, 7000 if ptds_mode == CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM else 3020, ptds_mode, NULL)

        global __cuMemcpy3DAsync_v2
        cuGetProcAddress_v2('cuMemcpy3DAsync', <void **>&__cuMemcpy3DAsync_v2, 7000 if ptds_mode == CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM else 3020, ptds_mode, NULL)

        global __cuMemcpy3DPeerAsync
        cuGetProcAddress_v2('cuMemcpy3DPeerAsync', <void **>&__cuMemcpy3DPeerAsync, 7000 if ptds_mode == CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM else 4000, ptds_mode, NULL)

        global __cuMemsetD8_v2
        cuGetProcAddress_v2('cuMemsetD8', <void **>&__cuMemsetD8_v2, 7000 if ptds_mode == CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM else 3020, ptds_mode, NULL)

        global __cuMemsetD16_v2
        cuGetProcAddress_v2('cuMemsetD16', <void **>&__cuMemsetD16_v2, 7000 if ptds_mode == CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM else 3020, ptds_mode, NULL)

        global __cuMemsetD32_v2
        cuGetProcAddress_v2('cuMemsetD32', <void **>&__cuMemsetD32_v2, 7000 if ptds_mode == CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM else 3020, ptds_mode, NULL)

        global __cuMemsetD2D8_v2
        cuGetProcAddress_v2('cuMemsetD2D8', <void **>&__cuMemsetD2D8_v2, 7000 if ptds_mode == CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM else 3020, ptds_mode, NULL)

        global __cuMemsetD2D16_v2
        cuGetProcAddress_v2('cuMemsetD2D16', <void **>&__cuMemsetD2D16_v2, 7000 if ptds_mode == CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM else 3020, ptds_mode, NULL)

        global __cuMemsetD2D32_v2
        cuGetProcAddress_v2('cuMemsetD2D32', <void **>&__cuMemsetD2D32_v2, 7000 if ptds_mode == CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM else 3020, ptds_mode, NULL)

        global __cuMemsetD8Async
        cuGetProcAddress_v2('cuMemsetD8Async', <void **>&__cuMemsetD8Async, 7000 if ptds_mode == CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM else 3020, ptds_mode, NULL)

        global __cuMemsetD16Async
        cuGetProcAddress_v2('cuMemsetD16Async', <void **>&__cuMemsetD16Async, 7000 if ptds_mode == CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM else 3020, ptds_mode, NULL)

        global __cuMemsetD32Async
        cuGetProcAddress_v2('cuMemsetD32Async', <void **>&__cuMemsetD32Async, 7000 if ptds_mode == CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM else 3020, ptds_mode, NULL)

        global __cuMemsetD2D8Async
        cuGetProcAddress_v2('cuMemsetD2D8Async', <void **>&__cuMemsetD2D8Async, 7000 if ptds_mode == CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM else 3020, ptds_mode, NULL)

        global __cuMemsetD2D16Async
        cuGetProcAddress_v2('cuMemsetD2D16Async', <void **>&__cuMemsetD2D16Async, 7000 if ptds_mode == CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM else 3020, ptds_mode, NULL)

        global __cuMemsetD2D32Async
        cuGetProcAddress_v2('cuMemsetD2D32Async', <void **>&__cuMemsetD2D32Async, 7000 if ptds_mode == CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM else 3020, ptds_mode, NULL)

        global __cuArrayCreate_v2
        cuGetProcAddress_v2('cuArrayCreate', <void **>&__cuArrayCreate_v2, 3020, ptds_mode, NULL)

        global __cuArrayGetDescriptor_v2
        cuGetProcAddress_v2('cuArrayGetDescriptor', <void **>&__cuArrayGetDescriptor_v2, 3020, ptds_mode, NULL)

        global __cuArrayGetSparseProperties
        cuGetProcAddress_v2('cuArrayGetSparseProperties', <void **>&__cuArrayGetSparseProperties, 11010, ptds_mode, NULL)

        global __cuMipmappedArrayGetSparseProperties
        cuGetProcAddress_v2('cuMipmappedArrayGetSparseProperties', <void **>&__cuMipmappedArrayGetSparseProperties, 11010, ptds_mode, NULL)

        global __cuArrayGetMemoryRequirements
        cuGetProcAddress_v2('cuArrayGetMemoryRequirements', <void **>&__cuArrayGetMemoryRequirements, 11060, ptds_mode, NULL)

        global __cuMipmappedArrayGetMemoryRequirements
        cuGetProcAddress_v2('cuMipmappedArrayGetMemoryRequirements', <void **>&__cuMipmappedArrayGetMemoryRequirements, 11060, ptds_mode, NULL)

        global __cuArrayGetPlane
        cuGetProcAddress_v2('cuArrayGetPlane', <void **>&__cuArrayGetPlane, 11020, ptds_mode, NULL)

        global __cuArrayDestroy
        cuGetProcAddress_v2('cuArrayDestroy', <void **>&__cuArrayDestroy, 2000, ptds_mode, NULL)

        global __cuArray3DCreate_v2
        cuGetProcAddress_v2('cuArray3DCreate', <void **>&__cuArray3DCreate_v2, 3020, ptds_mode, NULL)

        global __cuArray3DGetDescriptor_v2
        cuGetProcAddress_v2('cuArray3DGetDescriptor', <void **>&__cuArray3DGetDescriptor_v2, 3020, ptds_mode, NULL)

        global __cuMipmappedArrayCreate
        cuGetProcAddress_v2('cuMipmappedArrayCreate', <void **>&__cuMipmappedArrayCreate, 5000, ptds_mode, NULL)

        global __cuMipmappedArrayGetLevel
        cuGetProcAddress_v2('cuMipmappedArrayGetLevel', <void **>&__cuMipmappedArrayGetLevel, 5000, ptds_mode, NULL)

        global __cuMipmappedArrayDestroy
        cuGetProcAddress_v2('cuMipmappedArrayDestroy', <void **>&__cuMipmappedArrayDestroy, 5000, ptds_mode, NULL)

        global __cuMemGetHandleForAddressRange
        cuGetProcAddress_v2('cuMemGetHandleForAddressRange', <void **>&__cuMemGetHandleForAddressRange, 11070, ptds_mode, NULL)

        global __cuMemBatchDecompressAsync
        cuGetProcAddress_v2('cuMemBatchDecompressAsync', <void **>&__cuMemBatchDecompressAsync, 12060, ptds_mode, NULL)

        global __cuMemAddressReserve
        cuGetProcAddress_v2('cuMemAddressReserve', <void **>&__cuMemAddressReserve, 10020, ptds_mode, NULL)

        global __cuMemAddressFree
        cuGetProcAddress_v2('cuMemAddressFree', <void **>&__cuMemAddressFree, 10020, ptds_mode, NULL)

        global __cuMemCreate
        cuGetProcAddress_v2('cuMemCreate', <void **>&__cuMemCreate, 10020, ptds_mode, NULL)

        global __cuMemRelease
        cuGetProcAddress_v2('cuMemRelease', <void **>&__cuMemRelease, 10020, ptds_mode, NULL)

        global __cuMemMap
        cuGetProcAddress_v2('cuMemMap', <void **>&__cuMemMap, 10020, ptds_mode, NULL)

        global __cuMemMapArrayAsync
        cuGetProcAddress_v2('cuMemMapArrayAsync', <void **>&__cuMemMapArrayAsync, 11010, ptds_mode, NULL)

        global __cuMemUnmap
        cuGetProcAddress_v2('cuMemUnmap', <void **>&__cuMemUnmap, 10020, ptds_mode, NULL)

        global __cuMemSetAccess
        cuGetProcAddress_v2('cuMemSetAccess', <void **>&__cuMemSetAccess, 10020, ptds_mode, NULL)

        global __cuMemGetAccess
        cuGetProcAddress_v2('cuMemGetAccess', <void **>&__cuMemGetAccess, 10020, ptds_mode, NULL)

        global __cuMemExportToShareableHandle
        cuGetProcAddress_v2('cuMemExportToShareableHandle', <void **>&__cuMemExportToShareableHandle, 10020, ptds_mode, NULL)

        global __cuMemImportFromShareableHandle
        cuGetProcAddress_v2('cuMemImportFromShareableHandle', <void **>&__cuMemImportFromShareableHandle, 10020, ptds_mode, NULL)

        global __cuMemGetAllocationGranularity
        cuGetProcAddress_v2('cuMemGetAllocationGranularity', <void **>&__cuMemGetAllocationGranularity, 10020, ptds_mode, NULL)

        global __cuMemGetAllocationPropertiesFromHandle
        cuGetProcAddress_v2('cuMemGetAllocationPropertiesFromHandle', <void **>&__cuMemGetAllocationPropertiesFromHandle, 10020, ptds_mode, NULL)

        global __cuMemRetainAllocationHandle
        cuGetProcAddress_v2('cuMemRetainAllocationHandle', <void **>&__cuMemRetainAllocationHandle, 11000, ptds_mode, NULL)

        global __cuMemFreeAsync
        cuGetProcAddress_v2('cuMemFreeAsync', <void **>&__cuMemFreeAsync, 11020, ptds_mode, NULL)

        global __cuMemAllocAsync
        cuGetProcAddress_v2('cuMemAllocAsync', <void **>&__cuMemAllocAsync, 11020, ptds_mode, NULL)

        global __cuMemPoolTrimTo
        cuGetProcAddress_v2('cuMemPoolTrimTo', <void **>&__cuMemPoolTrimTo, 11020, ptds_mode, NULL)

        global __cuMemPoolSetAttribute
        cuGetProcAddress_v2('cuMemPoolSetAttribute', <void **>&__cuMemPoolSetAttribute, 11020, ptds_mode, NULL)

        global __cuMemPoolGetAttribute
        cuGetProcAddress_v2('cuMemPoolGetAttribute', <void **>&__cuMemPoolGetAttribute, 11020, ptds_mode, NULL)

        global __cuMemPoolSetAccess
        cuGetProcAddress_v2('cuMemPoolSetAccess', <void **>&__cuMemPoolSetAccess, 11020, ptds_mode, NULL)

        global __cuMemPoolGetAccess
        cuGetProcAddress_v2('cuMemPoolGetAccess', <void **>&__cuMemPoolGetAccess, 11020, ptds_mode, NULL)

        global __cuMemPoolCreate
        cuGetProcAddress_v2('cuMemPoolCreate', <void **>&__cuMemPoolCreate, 11020, ptds_mode, NULL)

        global __cuMemPoolDestroy
        cuGetProcAddress_v2('cuMemPoolDestroy', <void **>&__cuMemPoolDestroy, 11020, ptds_mode, NULL)

        global __cuMemAllocFromPoolAsync
        cuGetProcAddress_v2('cuMemAllocFromPoolAsync', <void **>&__cuMemAllocFromPoolAsync, 11020, ptds_mode, NULL)

        global __cuMemPoolExportToShareableHandle
        cuGetProcAddress_v2('cuMemPoolExportToShareableHandle', <void **>&__cuMemPoolExportToShareableHandle, 11020, ptds_mode, NULL)

        global __cuMemPoolImportFromShareableHandle
        cuGetProcAddress_v2('cuMemPoolImportFromShareableHandle', <void **>&__cuMemPoolImportFromShareableHandle, 11020, ptds_mode, NULL)

        global __cuMemPoolExportPointer
        cuGetProcAddress_v2('cuMemPoolExportPointer', <void **>&__cuMemPoolExportPointer, 11020, ptds_mode, NULL)

        global __cuMemPoolImportPointer
        cuGetProcAddress_v2('cuMemPoolImportPointer', <void **>&__cuMemPoolImportPointer, 11020, ptds_mode, NULL)

        global __cuMulticastCreate
        cuGetProcAddress_v2('cuMulticastCreate', <void **>&__cuMulticastCreate, 12010, ptds_mode, NULL)

        global __cuMulticastAddDevice
        cuGetProcAddress_v2('cuMulticastAddDevice', <void **>&__cuMulticastAddDevice, 12010, ptds_mode, NULL)

        global __cuMulticastBindMem
        cuGetProcAddress_v2('cuMulticastBindMem', <void **>&__cuMulticastBindMem, 12010, ptds_mode, NULL)

        global __cuMulticastBindAddr
        cuGetProcAddress_v2('cuMulticastBindAddr', <void **>&__cuMulticastBindAddr, 12010, ptds_mode, NULL)

        global __cuMulticastUnbind
        cuGetProcAddress_v2('cuMulticastUnbind', <void **>&__cuMulticastUnbind, 12010, ptds_mode, NULL)

        global __cuMulticastGetGranularity
        cuGetProcAddress_v2('cuMulticastGetGranularity', <void **>&__cuMulticastGetGranularity, 12010, ptds_mode, NULL)

        global __cuPointerGetAttribute
        cuGetProcAddress_v2('cuPointerGetAttribute', <void **>&__cuPointerGetAttribute, 4000, ptds_mode, NULL)

        global __cuMemPrefetchAsync_v2
        cuGetProcAddress_v2('cuMemPrefetchAsync', <void **>&__cuMemPrefetchAsync_v2, 12020, ptds_mode, NULL)

        global __cuMemAdvise_v2
        cuGetProcAddress_v2('cuMemAdvise', <void **>&__cuMemAdvise_v2, 12020, ptds_mode, NULL)

        global __cuMemRangeGetAttribute
        cuGetProcAddress_v2('cuMemRangeGetAttribute', <void **>&__cuMemRangeGetAttribute, 8000, ptds_mode, NULL)

        global __cuMemRangeGetAttributes
        cuGetProcAddress_v2('cuMemRangeGetAttributes', <void **>&__cuMemRangeGetAttributes, 8000, ptds_mode, NULL)

        global __cuPointerSetAttribute
        cuGetProcAddress_v2('cuPointerSetAttribute', <void **>&__cuPointerSetAttribute, 6000, ptds_mode, NULL)

        global __cuPointerGetAttributes
        cuGetProcAddress_v2('cuPointerGetAttributes', <void **>&__cuPointerGetAttributes, 7000, ptds_mode, NULL)

        global __cuStreamCreate
        cuGetProcAddress_v2('cuStreamCreate', <void **>&__cuStreamCreate, 2000, ptds_mode, NULL)

        global __cuStreamCreateWithPriority
        cuGetProcAddress_v2('cuStreamCreateWithPriority', <void **>&__cuStreamCreateWithPriority, 5050, ptds_mode, NULL)

        global __cuStreamGetPriority
        cuGetProcAddress_v2('cuStreamGetPriority', <void **>&__cuStreamGetPriority, 7000 if ptds_mode == CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM else 5050, ptds_mode, NULL)

        global __cuStreamGetDevice
        cuGetProcAddress_v2('cuStreamGetDevice', <void **>&__cuStreamGetDevice, 12080, ptds_mode, NULL)

        global __cuStreamGetFlags
        cuGetProcAddress_v2('cuStreamGetFlags', <void **>&__cuStreamGetFlags, 7000 if ptds_mode == CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM else 5050, ptds_mode, NULL)

        global __cuStreamGetId
        cuGetProcAddress_v2('cuStreamGetId', <void **>&__cuStreamGetId, 12000, ptds_mode, NULL)

        global __cuStreamGetCtx
        cuGetProcAddress_v2('cuStreamGetCtx', <void **>&__cuStreamGetCtx, 9020, ptds_mode, NULL)

        global __cuStreamGetCtx_v2
        cuGetProcAddress_v2('cuStreamGetCtx', <void **>&__cuStreamGetCtx_v2, 12050, ptds_mode, NULL)

        global __cuStreamWaitEvent
        cuGetProcAddress_v2('cuStreamWaitEvent', <void **>&__cuStreamWaitEvent, 7000 if ptds_mode == CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM else 3020, ptds_mode, NULL)

        global __cuStreamAddCallback
        cuGetProcAddress_v2('cuStreamAddCallback', <void **>&__cuStreamAddCallback, 7000 if ptds_mode == CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM else 5000, ptds_mode, NULL)

        global __cuStreamBeginCapture_v2
        cuGetProcAddress_v2('cuStreamBeginCapture', <void **>&__cuStreamBeginCapture_v2, 10010, ptds_mode, NULL)

        global __cuStreamBeginCaptureToGraph
        cuGetProcAddress_v2('cuStreamBeginCaptureToGraph', <void **>&__cuStreamBeginCaptureToGraph, 12030, ptds_mode, NULL)

        global __cuThreadExchangeStreamCaptureMode
        cuGetProcAddress_v2('cuThreadExchangeStreamCaptureMode', <void **>&__cuThreadExchangeStreamCaptureMode, 10010, ptds_mode, NULL)

        global __cuStreamEndCapture
        cuGetProcAddress_v2('cuStreamEndCapture', <void **>&__cuStreamEndCapture, 10000, ptds_mode, NULL)

        global __cuStreamIsCapturing
        cuGetProcAddress_v2('cuStreamIsCapturing', <void **>&__cuStreamIsCapturing, 10000, ptds_mode, NULL)

        global __cuStreamGetCaptureInfo_v2
        cuGetProcAddress_v2('cuStreamGetCaptureInfo', <void **>&__cuStreamGetCaptureInfo_v2, 11030, ptds_mode, NULL)

        global __cuStreamGetCaptureInfo_v3
        cuGetProcAddress_v2('cuStreamGetCaptureInfo', <void **>&__cuStreamGetCaptureInfo_v3, 12030, ptds_mode, NULL)

        global __cuStreamUpdateCaptureDependencies_v2
        cuGetProcAddress_v2('cuStreamUpdateCaptureDependencies', <void **>&__cuStreamUpdateCaptureDependencies_v2, 12030, ptds_mode, NULL)

        global __cuStreamAttachMemAsync
        cuGetProcAddress_v2('cuStreamAttachMemAsync', <void **>&__cuStreamAttachMemAsync, 7000 if ptds_mode == CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM else 6000, ptds_mode, NULL)

        global __cuStreamQuery
        cuGetProcAddress_v2('cuStreamQuery', <void **>&__cuStreamQuery, 7000 if ptds_mode == CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM else 2000, ptds_mode, NULL)

        global __cuStreamSynchronize
        cuGetProcAddress_v2('cuStreamSynchronize', <void **>&__cuStreamSynchronize, 7000 if ptds_mode == CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM else 2000, ptds_mode, NULL)

        global __cuStreamDestroy_v2
        cuGetProcAddress_v2('cuStreamDestroy', <void **>&__cuStreamDestroy_v2, 4000, ptds_mode, NULL)

        global __cuStreamCopyAttributes
        cuGetProcAddress_v2('cuStreamCopyAttributes', <void **>&__cuStreamCopyAttributes, 11000, ptds_mode, NULL)

        global __cuStreamGetAttribute
        cuGetProcAddress_v2('cuStreamGetAttribute', <void **>&__cuStreamGetAttribute, 11000, ptds_mode, NULL)

        global __cuStreamSetAttribute
        cuGetProcAddress_v2('cuStreamSetAttribute', <void **>&__cuStreamSetAttribute, 11000, ptds_mode, NULL)

        global __cuEventCreate
        cuGetProcAddress_v2('cuEventCreate', <void **>&__cuEventCreate, 2000, ptds_mode, NULL)

        global __cuEventRecord
        cuGetProcAddress_v2('cuEventRecord', <void **>&__cuEventRecord, 7000 if ptds_mode == CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM else 2000, ptds_mode, NULL)

        global __cuEventRecordWithFlags
        cuGetProcAddress_v2('cuEventRecordWithFlags', <void **>&__cuEventRecordWithFlags, 11010, ptds_mode, NULL)

        global __cuEventQuery
        cuGetProcAddress_v2('cuEventQuery', <void **>&__cuEventQuery, 2000, ptds_mode, NULL)

        global __cuEventSynchronize
        cuGetProcAddress_v2('cuEventSynchronize', <void **>&__cuEventSynchronize, 2000, ptds_mode, NULL)

        global __cuEventDestroy_v2
        cuGetProcAddress_v2('cuEventDestroy', <void **>&__cuEventDestroy_v2, 4000, ptds_mode, NULL)

        global __cuEventElapsedTime_v2
        cuGetProcAddress_v2('cuEventElapsedTime', <void **>&__cuEventElapsedTime_v2, 12080, ptds_mode, NULL)

        global __cuImportExternalMemory
        cuGetProcAddress_v2('cuImportExternalMemory', <void **>&__cuImportExternalMemory, 10000, ptds_mode, NULL)

        global __cuExternalMemoryGetMappedBuffer
        cuGetProcAddress_v2('cuExternalMemoryGetMappedBuffer', <void **>&__cuExternalMemoryGetMappedBuffer, 10000, ptds_mode, NULL)

        global __cuExternalMemoryGetMappedMipmappedArray
        cuGetProcAddress_v2('cuExternalMemoryGetMappedMipmappedArray', <void **>&__cuExternalMemoryGetMappedMipmappedArray, 10000, ptds_mode, NULL)

        global __cuDestroyExternalMemory
        cuGetProcAddress_v2('cuDestroyExternalMemory', <void **>&__cuDestroyExternalMemory, 10000, ptds_mode, NULL)

        global __cuImportExternalSemaphore
        cuGetProcAddress_v2('cuImportExternalSemaphore', <void **>&__cuImportExternalSemaphore, 10000, ptds_mode, NULL)

        global __cuSignalExternalSemaphoresAsync
        cuGetProcAddress_v2('cuSignalExternalSemaphoresAsync', <void **>&__cuSignalExternalSemaphoresAsync, 10000, ptds_mode, NULL)

        global __cuWaitExternalSemaphoresAsync
        cuGetProcAddress_v2('cuWaitExternalSemaphoresAsync', <void **>&__cuWaitExternalSemaphoresAsync, 10000, ptds_mode, NULL)

        global __cuDestroyExternalSemaphore
        cuGetProcAddress_v2('cuDestroyExternalSemaphore', <void **>&__cuDestroyExternalSemaphore, 10000, ptds_mode, NULL)

        global __cuStreamWaitValue32_v2
        cuGetProcAddress_v2('cuStreamWaitValue32', <void **>&__cuStreamWaitValue32_v2, 11070, ptds_mode, NULL)

        global __cuStreamWaitValue64_v2
        cuGetProcAddress_v2('cuStreamWaitValue64', <void **>&__cuStreamWaitValue64_v2, 11070, ptds_mode, NULL)

        global __cuStreamWriteValue32_v2
        cuGetProcAddress_v2('cuStreamWriteValue32', <void **>&__cuStreamWriteValue32_v2, 11070, ptds_mode, NULL)

        global __cuStreamWriteValue64_v2
        cuGetProcAddress_v2('cuStreamWriteValue64', <void **>&__cuStreamWriteValue64_v2, 11070, ptds_mode, NULL)

        global __cuStreamBatchMemOp_v2
        cuGetProcAddress_v2('cuStreamBatchMemOp', <void **>&__cuStreamBatchMemOp_v2, 11070, ptds_mode, NULL)

        global __cuFuncGetAttribute
        cuGetProcAddress_v2('cuFuncGetAttribute', <void **>&__cuFuncGetAttribute, 2020, ptds_mode, NULL)

        global __cuFuncSetAttribute
        cuGetProcAddress_v2('cuFuncSetAttribute', <void **>&__cuFuncSetAttribute, 9000, ptds_mode, NULL)

        global __cuFuncSetCacheConfig
        cuGetProcAddress_v2('cuFuncSetCacheConfig', <void **>&__cuFuncSetCacheConfig, 3000, ptds_mode, NULL)

        global __cuFuncGetModule
        cuGetProcAddress_v2('cuFuncGetModule', <void **>&__cuFuncGetModule, 11000, ptds_mode, NULL)

        global __cuFuncGetName
        cuGetProcAddress_v2('cuFuncGetName', <void **>&__cuFuncGetName, 12030, ptds_mode, NULL)

        global __cuFuncGetParamInfo
        cuGetProcAddress_v2('cuFuncGetParamInfo', <void **>&__cuFuncGetParamInfo, 12040, ptds_mode, NULL)

        global __cuFuncIsLoaded
        cuGetProcAddress_v2('cuFuncIsLoaded', <void **>&__cuFuncIsLoaded, 12040, ptds_mode, NULL)

        global __cuFuncLoad
        cuGetProcAddress_v2('cuFuncLoad', <void **>&__cuFuncLoad, 12040, ptds_mode, NULL)

        global __cuLaunchKernel
        cuGetProcAddress_v2('cuLaunchKernel', <void **>&__cuLaunchKernel, 7000 if ptds_mode == CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM else 4000, ptds_mode, NULL)

        global __cuLaunchKernelEx
        cuGetProcAddress_v2('cuLaunchKernelEx', <void **>&__cuLaunchKernelEx, 11060, ptds_mode, NULL)

        global __cuLaunchCooperativeKernel
        cuGetProcAddress_v2('cuLaunchCooperativeKernel', <void **>&__cuLaunchCooperativeKernel, 9000, ptds_mode, NULL)

        global __cuLaunchCooperativeKernelMultiDevice
        cuGetProcAddress_v2('cuLaunchCooperativeKernelMultiDevice', <void **>&__cuLaunchCooperativeKernelMultiDevice, 9000, ptds_mode, NULL)

        global __cuLaunchHostFunc
        cuGetProcAddress_v2('cuLaunchHostFunc', <void **>&__cuLaunchHostFunc, 10000, ptds_mode, NULL)

        global __cuFuncSetBlockShape
        cuGetProcAddress_v2('cuFuncSetBlockShape', <void **>&__cuFuncSetBlockShape, 2000, ptds_mode, NULL)

        global __cuFuncSetSharedSize
        cuGetProcAddress_v2('cuFuncSetSharedSize', <void **>&__cuFuncSetSharedSize, 2000, ptds_mode, NULL)

        global __cuParamSetSize
        cuGetProcAddress_v2('cuParamSetSize', <void **>&__cuParamSetSize, 2000, ptds_mode, NULL)

        global __cuParamSeti
        cuGetProcAddress_v2('cuParamSeti', <void **>&__cuParamSeti, 2000, ptds_mode, NULL)

        global __cuParamSetf
        cuGetProcAddress_v2('cuParamSetf', <void **>&__cuParamSetf, 2000, ptds_mode, NULL)

        global __cuParamSetv
        cuGetProcAddress_v2('cuParamSetv', <void **>&__cuParamSetv, 2000, ptds_mode, NULL)

        global __cuLaunch
        cuGetProcAddress_v2('cuLaunch', <void **>&__cuLaunch, 2000, ptds_mode, NULL)

        global __cuLaunchGrid
        cuGetProcAddress_v2('cuLaunchGrid', <void **>&__cuLaunchGrid, 2000, ptds_mode, NULL)

        global __cuLaunchGridAsync
        cuGetProcAddress_v2('cuLaunchGridAsync', <void **>&__cuLaunchGridAsync, 2000, ptds_mode, NULL)

        global __cuParamSetTexRef
        cuGetProcAddress_v2('cuParamSetTexRef', <void **>&__cuParamSetTexRef, 2000, ptds_mode, NULL)

        global __cuFuncSetSharedMemConfig
        cuGetProcAddress_v2('cuFuncSetSharedMemConfig', <void **>&__cuFuncSetSharedMemConfig, 4020, ptds_mode, NULL)

        global __cuGraphCreate
        cuGetProcAddress_v2('cuGraphCreate', <void **>&__cuGraphCreate, 10000, ptds_mode, NULL)

        global __cuGraphAddKernelNode_v2
        cuGetProcAddress_v2('cuGraphAddKernelNode', <void **>&__cuGraphAddKernelNode_v2, 12000, ptds_mode, NULL)

        global __cuGraphKernelNodeGetParams_v2
        cuGetProcAddress_v2('cuGraphKernelNodeGetParams', <void **>&__cuGraphKernelNodeGetParams_v2, 12000, ptds_mode, NULL)

        global __cuGraphKernelNodeSetParams_v2
        cuGetProcAddress_v2('cuGraphKernelNodeSetParams', <void **>&__cuGraphKernelNodeSetParams_v2, 12000, ptds_mode, NULL)

        global __cuGraphAddMemcpyNode
        cuGetProcAddress_v2('cuGraphAddMemcpyNode', <void **>&__cuGraphAddMemcpyNode, 10000, ptds_mode, NULL)

        global __cuGraphMemcpyNodeGetParams
        cuGetProcAddress_v2('cuGraphMemcpyNodeGetParams', <void **>&__cuGraphMemcpyNodeGetParams, 10000, ptds_mode, NULL)

        global __cuGraphMemcpyNodeSetParams
        cuGetProcAddress_v2('cuGraphMemcpyNodeSetParams', <void **>&__cuGraphMemcpyNodeSetParams, 10000, ptds_mode, NULL)

        global __cuGraphAddMemsetNode
        cuGetProcAddress_v2('cuGraphAddMemsetNode', <void **>&__cuGraphAddMemsetNode, 10000, ptds_mode, NULL)

        global __cuGraphMemsetNodeGetParams
        cuGetProcAddress_v2('cuGraphMemsetNodeGetParams', <void **>&__cuGraphMemsetNodeGetParams, 10000, ptds_mode, NULL)

        global __cuGraphMemsetNodeSetParams
        cuGetProcAddress_v2('cuGraphMemsetNodeSetParams', <void **>&__cuGraphMemsetNodeSetParams, 10000, ptds_mode, NULL)

        global __cuGraphAddHostNode
        cuGetProcAddress_v2('cuGraphAddHostNode', <void **>&__cuGraphAddHostNode, 10000, ptds_mode, NULL)

        global __cuGraphHostNodeGetParams
        cuGetProcAddress_v2('cuGraphHostNodeGetParams', <void **>&__cuGraphHostNodeGetParams, 10000, ptds_mode, NULL)

        global __cuGraphHostNodeSetParams
        cuGetProcAddress_v2('cuGraphHostNodeSetParams', <void **>&__cuGraphHostNodeSetParams, 10000, ptds_mode, NULL)

        global __cuGraphAddChildGraphNode
        cuGetProcAddress_v2('cuGraphAddChildGraphNode', <void **>&__cuGraphAddChildGraphNode, 10000, ptds_mode, NULL)

        global __cuGraphChildGraphNodeGetGraph
        cuGetProcAddress_v2('cuGraphChildGraphNodeGetGraph', <void **>&__cuGraphChildGraphNodeGetGraph, 10000, ptds_mode, NULL)

        global __cuGraphAddEmptyNode
        cuGetProcAddress_v2('cuGraphAddEmptyNode', <void **>&__cuGraphAddEmptyNode, 10000, ptds_mode, NULL)

        global __cuGraphAddEventRecordNode
        cuGetProcAddress_v2('cuGraphAddEventRecordNode', <void **>&__cuGraphAddEventRecordNode, 11010, ptds_mode, NULL)

        global __cuGraphEventRecordNodeGetEvent
        cuGetProcAddress_v2('cuGraphEventRecordNodeGetEvent', <void **>&__cuGraphEventRecordNodeGetEvent, 11010, ptds_mode, NULL)

        global __cuGraphEventRecordNodeSetEvent
        cuGetProcAddress_v2('cuGraphEventRecordNodeSetEvent', <void **>&__cuGraphEventRecordNodeSetEvent, 11010, ptds_mode, NULL)

        global __cuGraphAddEventWaitNode
        cuGetProcAddress_v2('cuGraphAddEventWaitNode', <void **>&__cuGraphAddEventWaitNode, 11010, ptds_mode, NULL)

        global __cuGraphEventWaitNodeGetEvent
        cuGetProcAddress_v2('cuGraphEventWaitNodeGetEvent', <void **>&__cuGraphEventWaitNodeGetEvent, 11010, ptds_mode, NULL)

        global __cuGraphEventWaitNodeSetEvent
        cuGetProcAddress_v2('cuGraphEventWaitNodeSetEvent', <void **>&__cuGraphEventWaitNodeSetEvent, 11010, ptds_mode, NULL)

        global __cuGraphAddExternalSemaphoresSignalNode
        cuGetProcAddress_v2('cuGraphAddExternalSemaphoresSignalNode', <void **>&__cuGraphAddExternalSemaphoresSignalNode, 11020, ptds_mode, NULL)

        global __cuGraphExternalSemaphoresSignalNodeGetParams
        cuGetProcAddress_v2('cuGraphExternalSemaphoresSignalNodeGetParams', <void **>&__cuGraphExternalSemaphoresSignalNodeGetParams, 11020, ptds_mode, NULL)

        global __cuGraphExternalSemaphoresSignalNodeSetParams
        cuGetProcAddress_v2('cuGraphExternalSemaphoresSignalNodeSetParams', <void **>&__cuGraphExternalSemaphoresSignalNodeSetParams, 11020, ptds_mode, NULL)

        global __cuGraphAddExternalSemaphoresWaitNode
        cuGetProcAddress_v2('cuGraphAddExternalSemaphoresWaitNode', <void **>&__cuGraphAddExternalSemaphoresWaitNode, 11020, ptds_mode, NULL)

        global __cuGraphExternalSemaphoresWaitNodeGetParams
        cuGetProcAddress_v2('cuGraphExternalSemaphoresWaitNodeGetParams', <void **>&__cuGraphExternalSemaphoresWaitNodeGetParams, 11020, ptds_mode, NULL)

        global __cuGraphExternalSemaphoresWaitNodeSetParams
        cuGetProcAddress_v2('cuGraphExternalSemaphoresWaitNodeSetParams', <void **>&__cuGraphExternalSemaphoresWaitNodeSetParams, 11020, ptds_mode, NULL)

        global __cuGraphAddBatchMemOpNode
        cuGetProcAddress_v2('cuGraphAddBatchMemOpNode', <void **>&__cuGraphAddBatchMemOpNode, 11070, ptds_mode, NULL)

        global __cuGraphBatchMemOpNodeGetParams
        cuGetProcAddress_v2('cuGraphBatchMemOpNodeGetParams', <void **>&__cuGraphBatchMemOpNodeGetParams, 11070, ptds_mode, NULL)

        global __cuGraphBatchMemOpNodeSetParams
        cuGetProcAddress_v2('cuGraphBatchMemOpNodeSetParams', <void **>&__cuGraphBatchMemOpNodeSetParams, 11070, ptds_mode, NULL)

        global __cuGraphExecBatchMemOpNodeSetParams
        cuGetProcAddress_v2('cuGraphExecBatchMemOpNodeSetParams', <void **>&__cuGraphExecBatchMemOpNodeSetParams, 11070, ptds_mode, NULL)

        global __cuGraphAddMemAllocNode
        cuGetProcAddress_v2('cuGraphAddMemAllocNode', <void **>&__cuGraphAddMemAllocNode, 11040, ptds_mode, NULL)

        global __cuGraphMemAllocNodeGetParams
        cuGetProcAddress_v2('cuGraphMemAllocNodeGetParams', <void **>&__cuGraphMemAllocNodeGetParams, 11040, ptds_mode, NULL)

        global __cuGraphAddMemFreeNode
        cuGetProcAddress_v2('cuGraphAddMemFreeNode', <void **>&__cuGraphAddMemFreeNode, 11040, ptds_mode, NULL)

        global __cuGraphMemFreeNodeGetParams
        cuGetProcAddress_v2('cuGraphMemFreeNodeGetParams', <void **>&__cuGraphMemFreeNodeGetParams, 11040, ptds_mode, NULL)

        global __cuDeviceGraphMemTrim
        cuGetProcAddress_v2('cuDeviceGraphMemTrim', <void **>&__cuDeviceGraphMemTrim, 11040, ptds_mode, NULL)

        global __cuDeviceGetGraphMemAttribute
        cuGetProcAddress_v2('cuDeviceGetGraphMemAttribute', <void **>&__cuDeviceGetGraphMemAttribute, 11040, ptds_mode, NULL)

        global __cuDeviceSetGraphMemAttribute
        cuGetProcAddress_v2('cuDeviceSetGraphMemAttribute', <void **>&__cuDeviceSetGraphMemAttribute, 11040, ptds_mode, NULL)

        global __cuGraphClone
        cuGetProcAddress_v2('cuGraphClone', <void **>&__cuGraphClone, 10000, ptds_mode, NULL)

        global __cuGraphNodeFindInClone
        cuGetProcAddress_v2('cuGraphNodeFindInClone', <void **>&__cuGraphNodeFindInClone, 10000, ptds_mode, NULL)

        global __cuGraphNodeGetType
        cuGetProcAddress_v2('cuGraphNodeGetType', <void **>&__cuGraphNodeGetType, 10000, ptds_mode, NULL)

        global __cuGraphGetNodes
        cuGetProcAddress_v2('cuGraphGetNodes', <void **>&__cuGraphGetNodes, 10000, ptds_mode, NULL)

        global __cuGraphGetRootNodes
        cuGetProcAddress_v2('cuGraphGetRootNodes', <void **>&__cuGraphGetRootNodes, 10000, ptds_mode, NULL)

        global __cuGraphGetEdges_v2
        cuGetProcAddress_v2('cuGraphGetEdges', <void **>&__cuGraphGetEdges_v2, 12030, ptds_mode, NULL)

        global __cuGraphNodeGetDependencies_v2
        cuGetProcAddress_v2('cuGraphNodeGetDependencies', <void **>&__cuGraphNodeGetDependencies_v2, 12030, ptds_mode, NULL)

        global __cuGraphNodeGetDependentNodes_v2
        cuGetProcAddress_v2('cuGraphNodeGetDependentNodes', <void **>&__cuGraphNodeGetDependentNodes_v2, 12030, ptds_mode, NULL)

        global __cuGraphAddDependencies_v2
        cuGetProcAddress_v2('cuGraphAddDependencies', <void **>&__cuGraphAddDependencies_v2, 12030, ptds_mode, NULL)

        global __cuGraphRemoveDependencies_v2
        cuGetProcAddress_v2('cuGraphRemoveDependencies', <void **>&__cuGraphRemoveDependencies_v2, 12030, ptds_mode, NULL)

        global __cuGraphDestroyNode
        cuGetProcAddress_v2('cuGraphDestroyNode', <void **>&__cuGraphDestroyNode, 10000, ptds_mode, NULL)

        global __cuGraphInstantiateWithFlags
        cuGetProcAddress_v2('cuGraphInstantiateWithFlags', <void **>&__cuGraphInstantiateWithFlags, 11040, ptds_mode, NULL)

        global __cuGraphInstantiateWithParams
        cuGetProcAddress_v2('cuGraphInstantiateWithParams', <void **>&__cuGraphInstantiateWithParams, 12000, ptds_mode, NULL)

        global __cuGraphExecGetFlags
        cuGetProcAddress_v2('cuGraphExecGetFlags', <void **>&__cuGraphExecGetFlags, 12000, ptds_mode, NULL)

        global __cuGraphExecKernelNodeSetParams_v2
        cuGetProcAddress_v2('cuGraphExecKernelNodeSetParams', <void **>&__cuGraphExecKernelNodeSetParams_v2, 12000, ptds_mode, NULL)

        global __cuGraphExecMemcpyNodeSetParams
        cuGetProcAddress_v2('cuGraphExecMemcpyNodeSetParams', <void **>&__cuGraphExecMemcpyNodeSetParams, 10020, ptds_mode, NULL)

        global __cuGraphExecMemsetNodeSetParams
        cuGetProcAddress_v2('cuGraphExecMemsetNodeSetParams', <void **>&__cuGraphExecMemsetNodeSetParams, 10020, ptds_mode, NULL)

        global __cuGraphExecHostNodeSetParams
        cuGetProcAddress_v2('cuGraphExecHostNodeSetParams', <void **>&__cuGraphExecHostNodeSetParams, 10020, ptds_mode, NULL)

        global __cuGraphExecChildGraphNodeSetParams
        cuGetProcAddress_v2('cuGraphExecChildGraphNodeSetParams', <void **>&__cuGraphExecChildGraphNodeSetParams, 11010, ptds_mode, NULL)

        global __cuGraphExecEventRecordNodeSetEvent
        cuGetProcAddress_v2('cuGraphExecEventRecordNodeSetEvent', <void **>&__cuGraphExecEventRecordNodeSetEvent, 11010, ptds_mode, NULL)

        global __cuGraphExecEventWaitNodeSetEvent
        cuGetProcAddress_v2('cuGraphExecEventWaitNodeSetEvent', <void **>&__cuGraphExecEventWaitNodeSetEvent, 11010, ptds_mode, NULL)

        global __cuGraphExecExternalSemaphoresSignalNodeSetParams
        cuGetProcAddress_v2('cuGraphExecExternalSemaphoresSignalNodeSetParams', <void **>&__cuGraphExecExternalSemaphoresSignalNodeSetParams, 11020, ptds_mode, NULL)

        global __cuGraphExecExternalSemaphoresWaitNodeSetParams
        cuGetProcAddress_v2('cuGraphExecExternalSemaphoresWaitNodeSetParams', <void **>&__cuGraphExecExternalSemaphoresWaitNodeSetParams, 11020, ptds_mode, NULL)

        global __cuGraphNodeSetEnabled
        cuGetProcAddress_v2('cuGraphNodeSetEnabled', <void **>&__cuGraphNodeSetEnabled, 11060, ptds_mode, NULL)

        global __cuGraphNodeGetEnabled
        cuGetProcAddress_v2('cuGraphNodeGetEnabled', <void **>&__cuGraphNodeGetEnabled, 11060, ptds_mode, NULL)

        global __cuGraphUpload
        cuGetProcAddress_v2('cuGraphUpload', <void **>&__cuGraphUpload, 11010, ptds_mode, NULL)

        global __cuGraphLaunch
        cuGetProcAddress_v2('cuGraphLaunch', <void **>&__cuGraphLaunch, 10000, ptds_mode, NULL)

        global __cuGraphExecDestroy
        cuGetProcAddress_v2('cuGraphExecDestroy', <void **>&__cuGraphExecDestroy, 10000, ptds_mode, NULL)

        global __cuGraphDestroy
        cuGetProcAddress_v2('cuGraphDestroy', <void **>&__cuGraphDestroy, 10000, ptds_mode, NULL)

        global __cuGraphExecUpdate_v2
        cuGetProcAddress_v2('cuGraphExecUpdate', <void **>&__cuGraphExecUpdate_v2, 12000, ptds_mode, NULL)

        global __cuGraphKernelNodeCopyAttributes
        cuGetProcAddress_v2('cuGraphKernelNodeCopyAttributes', <void **>&__cuGraphKernelNodeCopyAttributes, 11000, ptds_mode, NULL)

        global __cuGraphKernelNodeGetAttribute
        cuGetProcAddress_v2('cuGraphKernelNodeGetAttribute', <void **>&__cuGraphKernelNodeGetAttribute, 11000, ptds_mode, NULL)

        global __cuGraphKernelNodeSetAttribute
        cuGetProcAddress_v2('cuGraphKernelNodeSetAttribute', <void **>&__cuGraphKernelNodeSetAttribute, 11000, ptds_mode, NULL)

        global __cuGraphDebugDotPrint
        cuGetProcAddress_v2('cuGraphDebugDotPrint', <void **>&__cuGraphDebugDotPrint, 11030, ptds_mode, NULL)

        global __cuUserObjectCreate
        cuGetProcAddress_v2('cuUserObjectCreate', <void **>&__cuUserObjectCreate, 11030, ptds_mode, NULL)

        global __cuUserObjectRetain
        cuGetProcAddress_v2('cuUserObjectRetain', <void **>&__cuUserObjectRetain, 11030, ptds_mode, NULL)

        global __cuUserObjectRelease
        cuGetProcAddress_v2('cuUserObjectRelease', <void **>&__cuUserObjectRelease, 11030, ptds_mode, NULL)

        global __cuGraphRetainUserObject
        cuGetProcAddress_v2('cuGraphRetainUserObject', <void **>&__cuGraphRetainUserObject, 11030, ptds_mode, NULL)

        global __cuGraphReleaseUserObject
        cuGetProcAddress_v2('cuGraphReleaseUserObject', <void **>&__cuGraphReleaseUserObject, 11030, ptds_mode, NULL)

        global __cuGraphAddNode_v2
        cuGetProcAddress_v2('cuGraphAddNode', <void **>&__cuGraphAddNode_v2, 12030, ptds_mode, NULL)

        global __cuGraphNodeSetParams
        cuGetProcAddress_v2('cuGraphNodeSetParams', <void **>&__cuGraphNodeSetParams, 12020, ptds_mode, NULL)

        global __cuGraphExecNodeSetParams
        cuGetProcAddress_v2('cuGraphExecNodeSetParams', <void **>&__cuGraphExecNodeSetParams, 12020, ptds_mode, NULL)

        global __cuGraphConditionalHandleCreate
        cuGetProcAddress_v2('cuGraphConditionalHandleCreate', <void **>&__cuGraphConditionalHandleCreate, 12030, ptds_mode, NULL)

        global __cuOccupancyMaxActiveBlocksPerMultiprocessor
        cuGetProcAddress_v2('cuOccupancyMaxActiveBlocksPerMultiprocessor', <void **>&__cuOccupancyMaxActiveBlocksPerMultiprocessor, 6050, ptds_mode, NULL)

        global __cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags
        cuGetProcAddress_v2('cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags', <void **>&__cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags, 7000, ptds_mode, NULL)

        global __cuOccupancyMaxPotentialBlockSize
        cuGetProcAddress_v2('cuOccupancyMaxPotentialBlockSize', <void **>&__cuOccupancyMaxPotentialBlockSize, 6050, ptds_mode, NULL)

        global __cuOccupancyMaxPotentialBlockSizeWithFlags
        cuGetProcAddress_v2('cuOccupancyMaxPotentialBlockSizeWithFlags', <void **>&__cuOccupancyMaxPotentialBlockSizeWithFlags, 7000, ptds_mode, NULL)

        global __cuOccupancyAvailableDynamicSMemPerBlock
        cuGetProcAddress_v2('cuOccupancyAvailableDynamicSMemPerBlock', <void **>&__cuOccupancyAvailableDynamicSMemPerBlock, 10020, ptds_mode, NULL)

        global __cuOccupancyMaxPotentialClusterSize
        cuGetProcAddress_v2('cuOccupancyMaxPotentialClusterSize', <void **>&__cuOccupancyMaxPotentialClusterSize, 11070, ptds_mode, NULL)

        global __cuOccupancyMaxActiveClusters
        cuGetProcAddress_v2('cuOccupancyMaxActiveClusters', <void **>&__cuOccupancyMaxActiveClusters, 11070, ptds_mode, NULL)

        global __cuTexRefSetArray
        cuGetProcAddress_v2('cuTexRefSetArray', <void **>&__cuTexRefSetArray, 2000, ptds_mode, NULL)

        global __cuTexRefSetMipmappedArray
        cuGetProcAddress_v2('cuTexRefSetMipmappedArray', <void **>&__cuTexRefSetMipmappedArray, 5000, ptds_mode, NULL)

        global __cuTexRefSetAddress_v2
        cuGetProcAddress_v2('cuTexRefSetAddress', <void **>&__cuTexRefSetAddress_v2, 3020, ptds_mode, NULL)

        global __cuTexRefSetAddress2D_v3
        cuGetProcAddress_v2('cuTexRefSetAddress2D', <void **>&__cuTexRefSetAddress2D_v3, 4010, ptds_mode, NULL)

        global __cuTexRefSetFormat
        cuGetProcAddress_v2('cuTexRefSetFormat', <void **>&__cuTexRefSetFormat, 2000, ptds_mode, NULL)

        global __cuTexRefSetAddressMode
        cuGetProcAddress_v2('cuTexRefSetAddressMode', <void **>&__cuTexRefSetAddressMode, 2000, ptds_mode, NULL)

        global __cuTexRefSetFilterMode
        cuGetProcAddress_v2('cuTexRefSetFilterMode', <void **>&__cuTexRefSetFilterMode, 2000, ptds_mode, NULL)

        global __cuTexRefSetMipmapFilterMode
        cuGetProcAddress_v2('cuTexRefSetMipmapFilterMode', <void **>&__cuTexRefSetMipmapFilterMode, 5000, ptds_mode, NULL)

        global __cuTexRefSetMipmapLevelBias
        cuGetProcAddress_v2('cuTexRefSetMipmapLevelBias', <void **>&__cuTexRefSetMipmapLevelBias, 5000, ptds_mode, NULL)

        global __cuTexRefSetMipmapLevelClamp
        cuGetProcAddress_v2('cuTexRefSetMipmapLevelClamp', <void **>&__cuTexRefSetMipmapLevelClamp, 5000, ptds_mode, NULL)

        global __cuTexRefSetMaxAnisotropy
        cuGetProcAddress_v2('cuTexRefSetMaxAnisotropy', <void **>&__cuTexRefSetMaxAnisotropy, 5000, ptds_mode, NULL)

        global __cuTexRefSetBorderColor
        cuGetProcAddress_v2('cuTexRefSetBorderColor', <void **>&__cuTexRefSetBorderColor, 8000, ptds_mode, NULL)

        global __cuTexRefSetFlags
        cuGetProcAddress_v2('cuTexRefSetFlags', <void **>&__cuTexRefSetFlags, 2000, ptds_mode, NULL)

        global __cuTexRefGetAddress_v2
        cuGetProcAddress_v2('cuTexRefGetAddress', <void **>&__cuTexRefGetAddress_v2, 3020, ptds_mode, NULL)

        global __cuTexRefGetArray
        cuGetProcAddress_v2('cuTexRefGetArray', <void **>&__cuTexRefGetArray, 2000, ptds_mode, NULL)

        global __cuTexRefGetMipmappedArray
        cuGetProcAddress_v2('cuTexRefGetMipmappedArray', <void **>&__cuTexRefGetMipmappedArray, 5000, ptds_mode, NULL)

        global __cuTexRefGetAddressMode
        cuGetProcAddress_v2('cuTexRefGetAddressMode', <void **>&__cuTexRefGetAddressMode, 2000, ptds_mode, NULL)

        global __cuTexRefGetFilterMode
        cuGetProcAddress_v2('cuTexRefGetFilterMode', <void **>&__cuTexRefGetFilterMode, 2000, ptds_mode, NULL)

        global __cuTexRefGetFormat
        cuGetProcAddress_v2('cuTexRefGetFormat', <void **>&__cuTexRefGetFormat, 2000, ptds_mode, NULL)

        global __cuTexRefGetMipmapFilterMode
        cuGetProcAddress_v2('cuTexRefGetMipmapFilterMode', <void **>&__cuTexRefGetMipmapFilterMode, 5000, ptds_mode, NULL)

        global __cuTexRefGetMipmapLevelBias
        cuGetProcAddress_v2('cuTexRefGetMipmapLevelBias', <void **>&__cuTexRefGetMipmapLevelBias, 5000, ptds_mode, NULL)

        global __cuTexRefGetMipmapLevelClamp
        cuGetProcAddress_v2('cuTexRefGetMipmapLevelClamp', <void **>&__cuTexRefGetMipmapLevelClamp, 5000, ptds_mode, NULL)

        global __cuTexRefGetMaxAnisotropy
        cuGetProcAddress_v2('cuTexRefGetMaxAnisotropy', <void **>&__cuTexRefGetMaxAnisotropy, 5000, ptds_mode, NULL)

        global __cuTexRefGetBorderColor
        cuGetProcAddress_v2('cuTexRefGetBorderColor', <void **>&__cuTexRefGetBorderColor, 8000, ptds_mode, NULL)

        global __cuTexRefGetFlags
        cuGetProcAddress_v2('cuTexRefGetFlags', <void **>&__cuTexRefGetFlags, 2000, ptds_mode, NULL)

        global __cuTexRefCreate
        cuGetProcAddress_v2('cuTexRefCreate', <void **>&__cuTexRefCreate, 2000, ptds_mode, NULL)

        global __cuTexRefDestroy
        cuGetProcAddress_v2('cuTexRefDestroy', <void **>&__cuTexRefDestroy, 2000, ptds_mode, NULL)

        global __cuSurfRefSetArray
        cuGetProcAddress_v2('cuSurfRefSetArray', <void **>&__cuSurfRefSetArray, 3000, ptds_mode, NULL)

        global __cuSurfRefGetArray
        cuGetProcAddress_v2('cuSurfRefGetArray', <void **>&__cuSurfRefGetArray, 3000, ptds_mode, NULL)

        global __cuTexObjectCreate
        cuGetProcAddress_v2('cuTexObjectCreate', <void **>&__cuTexObjectCreate, 5000, ptds_mode, NULL)

        global __cuTexObjectDestroy
        cuGetProcAddress_v2('cuTexObjectDestroy', <void **>&__cuTexObjectDestroy, 5000, ptds_mode, NULL)

        global __cuTexObjectGetResourceDesc
        cuGetProcAddress_v2('cuTexObjectGetResourceDesc', <void **>&__cuTexObjectGetResourceDesc, 5000, ptds_mode, NULL)

        global __cuTexObjectGetTextureDesc
        cuGetProcAddress_v2('cuTexObjectGetTextureDesc', <void **>&__cuTexObjectGetTextureDesc, 5000, ptds_mode, NULL)

        global __cuTexObjectGetResourceViewDesc
        cuGetProcAddress_v2('cuTexObjectGetResourceViewDesc', <void **>&__cuTexObjectGetResourceViewDesc, 5000, ptds_mode, NULL)

        global __cuSurfObjectCreate
        cuGetProcAddress_v2('cuSurfObjectCreate', <void **>&__cuSurfObjectCreate, 5000, ptds_mode, NULL)

        global __cuSurfObjectDestroy
        cuGetProcAddress_v2('cuSurfObjectDestroy', <void **>&__cuSurfObjectDestroy, 5000, ptds_mode, NULL)

        global __cuSurfObjectGetResourceDesc
        cuGetProcAddress_v2('cuSurfObjectGetResourceDesc', <void **>&__cuSurfObjectGetResourceDesc, 5000, ptds_mode, NULL)

        global __cuTensorMapEncodeTiled
        cuGetProcAddress_v2('cuTensorMapEncodeTiled', <void **>&__cuTensorMapEncodeTiled, 12000, ptds_mode, NULL)

        global __cuTensorMapEncodeIm2col
        cuGetProcAddress_v2('cuTensorMapEncodeIm2col', <void **>&__cuTensorMapEncodeIm2col, 12000, ptds_mode, NULL)

        global __cuTensorMapEncodeIm2colWide
        cuGetProcAddress_v2('cuTensorMapEncodeIm2colWide', <void **>&__cuTensorMapEncodeIm2colWide, 12080, ptds_mode, NULL)

        global __cuTensorMapReplaceAddress
        cuGetProcAddress_v2('cuTensorMapReplaceAddress', <void **>&__cuTensorMapReplaceAddress, 12000, ptds_mode, NULL)

        global __cuDeviceCanAccessPeer
        cuGetProcAddress_v2('cuDeviceCanAccessPeer', <void **>&__cuDeviceCanAccessPeer, 4000, ptds_mode, NULL)

        global __cuCtxEnablePeerAccess
        cuGetProcAddress_v2('cuCtxEnablePeerAccess', <void **>&__cuCtxEnablePeerAccess, 4000, ptds_mode, NULL)

        global __cuCtxDisablePeerAccess
        cuGetProcAddress_v2('cuCtxDisablePeerAccess', <void **>&__cuCtxDisablePeerAccess, 4000, ptds_mode, NULL)

        global __cuDeviceGetP2PAttribute
        cuGetProcAddress_v2('cuDeviceGetP2PAttribute', <void **>&__cuDeviceGetP2PAttribute, 8000, ptds_mode, NULL)

        global __cuGraphicsUnregisterResource
        cuGetProcAddress_v2('cuGraphicsUnregisterResource', <void **>&__cuGraphicsUnregisterResource, 3000, ptds_mode, NULL)

        global __cuGraphicsSubResourceGetMappedArray
        cuGetProcAddress_v2('cuGraphicsSubResourceGetMappedArray', <void **>&__cuGraphicsSubResourceGetMappedArray, 3000, ptds_mode, NULL)

        global __cuGraphicsResourceGetMappedMipmappedArray
        cuGetProcAddress_v2('cuGraphicsResourceGetMappedMipmappedArray', <void **>&__cuGraphicsResourceGetMappedMipmappedArray, 5000, ptds_mode, NULL)

        global __cuGraphicsResourceGetMappedPointer_v2
        cuGetProcAddress_v2('cuGraphicsResourceGetMappedPointer', <void **>&__cuGraphicsResourceGetMappedPointer_v2, 3020, ptds_mode, NULL)

        global __cuGraphicsResourceSetMapFlags_v2
        cuGetProcAddress_v2('cuGraphicsResourceSetMapFlags', <void **>&__cuGraphicsResourceSetMapFlags_v2, 6050, ptds_mode, NULL)

        global __cuGraphicsMapResources
        cuGetProcAddress_v2('cuGraphicsMapResources', <void **>&__cuGraphicsMapResources, 7000 if ptds_mode == CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM else 3000, ptds_mode, NULL)

        global __cuGraphicsUnmapResources
        cuGetProcAddress_v2('cuGraphicsUnmapResources', <void **>&__cuGraphicsUnmapResources, 7000 if ptds_mode == CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM else 3000, ptds_mode, NULL)

        global __cuGetProcAddress_v2
        cuGetProcAddress_v2('cuGetProcAddress', <void **>&__cuGetProcAddress_v2, 12000, ptds_mode, NULL)

        global __cuCoredumpGetAttribute
        cuGetProcAddress_v2('cuCoredumpGetAttribute', <void **>&__cuCoredumpGetAttribute, 12010, ptds_mode, NULL)

        global __cuCoredumpGetAttributeGlobal
        cuGetProcAddress_v2('cuCoredumpGetAttributeGlobal', <void **>&__cuCoredumpGetAttributeGlobal, 12010, ptds_mode, NULL)

        global __cuCoredumpSetAttribute
        cuGetProcAddress_v2('cuCoredumpSetAttribute', <void **>&__cuCoredumpSetAttribute, 12010, ptds_mode, NULL)

        global __cuCoredumpSetAttributeGlobal
        cuGetProcAddress_v2('cuCoredumpSetAttributeGlobal', <void **>&__cuCoredumpSetAttributeGlobal, 12010, ptds_mode, NULL)

        global __cuGetExportTable
        cuGetProcAddress_v2('cuGetExportTable', <void **>&__cuGetExportTable, 3000, ptds_mode, NULL)

        global __cuGreenCtxCreate
        cuGetProcAddress_v2('cuGreenCtxCreate', <void **>&__cuGreenCtxCreate, 12040, ptds_mode, NULL)

        global __cuGreenCtxDestroy
        cuGetProcAddress_v2('cuGreenCtxDestroy', <void **>&__cuGreenCtxDestroy, 12040, ptds_mode, NULL)

        global __cuCtxFromGreenCtx
        cuGetProcAddress_v2('cuCtxFromGreenCtx', <void **>&__cuCtxFromGreenCtx, 12040, ptds_mode, NULL)

        global __cuDeviceGetDevResource
        cuGetProcAddress_v2('cuDeviceGetDevResource', <void **>&__cuDeviceGetDevResource, 12040, ptds_mode, NULL)

        global __cuCtxGetDevResource
        cuGetProcAddress_v2('cuCtxGetDevResource', <void **>&__cuCtxGetDevResource, 12040, ptds_mode, NULL)

        global __cuGreenCtxGetDevResource
        cuGetProcAddress_v2('cuGreenCtxGetDevResource', <void **>&__cuGreenCtxGetDevResource, 12040, ptds_mode, NULL)

        global __cuDevSmResourceSplitByCount
        cuGetProcAddress_v2('cuDevSmResourceSplitByCount', <void **>&__cuDevSmResourceSplitByCount, 12040, ptds_mode, NULL)

        global __cuDevResourceGenerateDesc
        cuGetProcAddress_v2('cuDevResourceGenerateDesc', <void **>&__cuDevResourceGenerateDesc, 12040, ptds_mode, NULL)

        global __cuGreenCtxRecordEvent
        cuGetProcAddress_v2('cuGreenCtxRecordEvent', <void **>&__cuGreenCtxRecordEvent, 12040, ptds_mode, NULL)

        global __cuGreenCtxWaitEvent
        cuGetProcAddress_v2('cuGreenCtxWaitEvent', <void **>&__cuGreenCtxWaitEvent, 12040, ptds_mode, NULL)

        global __cuStreamGetGreenCtx
        cuGetProcAddress_v2('cuStreamGetGreenCtx', <void **>&__cuStreamGetGreenCtx, 12040, ptds_mode, NULL)

        global __cuGreenCtxStreamCreate
        cuGetProcAddress_v2('cuGreenCtxStreamCreate', <void **>&__cuGreenCtxStreamCreate, 12050, ptds_mode, NULL)

        global __cuLogsRegisterCallback
        cuGetProcAddress_v2('cuLogsRegisterCallback', <void **>&__cuLogsRegisterCallback, 12090, ptds_mode, NULL)

        global __cuLogsUnregisterCallback
        cuGetProcAddress_v2('cuLogsUnregisterCallback', <void **>&__cuLogsUnregisterCallback, 12090, ptds_mode, NULL)

        global __cuLogsCurrent
        cuGetProcAddress_v2('cuLogsCurrent', <void **>&__cuLogsCurrent, 12090, ptds_mode, NULL)

        global __cuLogsDumpToFile
        cuGetProcAddress_v2('cuLogsDumpToFile', <void **>&__cuLogsDumpToFile, 12090, ptds_mode, NULL)

        global __cuLogsDumpToMemory
        cuGetProcAddress_v2('cuLogsDumpToMemory', <void **>&__cuLogsDumpToMemory, 12090, ptds_mode, NULL)

        global __cuCheckpointProcessGetRestoreThreadId
        cuGetProcAddress_v2('cuCheckpointProcessGetRestoreThreadId', <void **>&__cuCheckpointProcessGetRestoreThreadId, 12080, ptds_mode, NULL)

        global __cuCheckpointProcessGetState
        cuGetProcAddress_v2('cuCheckpointProcessGetState', <void **>&__cuCheckpointProcessGetState, 12080, ptds_mode, NULL)

        global __cuCheckpointProcessLock
        cuGetProcAddress_v2('cuCheckpointProcessLock', <void **>&__cuCheckpointProcessLock, 12080, ptds_mode, NULL)

        global __cuCheckpointProcessCheckpoint
        cuGetProcAddress_v2('cuCheckpointProcessCheckpoint', <void **>&__cuCheckpointProcessCheckpoint, 12080, ptds_mode, NULL)

        global __cuCheckpointProcessRestore
        cuGetProcAddress_v2('cuCheckpointProcessRestore', <void **>&__cuCheckpointProcessRestore, 12080, ptds_mode, NULL)

        global __cuCheckpointProcessUnlock
        cuGetProcAddress_v2('cuCheckpointProcessUnlock', <void **>&__cuCheckpointProcessUnlock, 12080, ptds_mode, NULL)

        global __cuGraphicsEGLRegisterImage
        cuGetProcAddress_v2('cuGraphicsEGLRegisterImage', <void **>&__cuGraphicsEGLRegisterImage, 7000, ptds_mode, NULL)

        global __cuEGLStreamConsumerConnect
        cuGetProcAddress_v2('cuEGLStreamConsumerConnect', <void **>&__cuEGLStreamConsumerConnect, 7000, ptds_mode, NULL)

        global __cuEGLStreamConsumerConnectWithFlags
        cuGetProcAddress_v2('cuEGLStreamConsumerConnectWithFlags', <void **>&__cuEGLStreamConsumerConnectWithFlags, 8000, ptds_mode, NULL)

        global __cuEGLStreamConsumerDisconnect
        cuGetProcAddress_v2('cuEGLStreamConsumerDisconnect', <void **>&__cuEGLStreamConsumerDisconnect, 7000, ptds_mode, NULL)

        global __cuEGLStreamConsumerAcquireFrame
        cuGetProcAddress_v2('cuEGLStreamConsumerAcquireFrame', <void **>&__cuEGLStreamConsumerAcquireFrame, 7000, ptds_mode, NULL)

        global __cuEGLStreamConsumerReleaseFrame
        cuGetProcAddress_v2('cuEGLStreamConsumerReleaseFrame', <void **>&__cuEGLStreamConsumerReleaseFrame, 7000, ptds_mode, NULL)

        global __cuEGLStreamProducerConnect
        cuGetProcAddress_v2('cuEGLStreamProducerConnect', <void **>&__cuEGLStreamProducerConnect, 7000, ptds_mode, NULL)

        global __cuEGLStreamProducerDisconnect
        cuGetProcAddress_v2('cuEGLStreamProducerDisconnect', <void **>&__cuEGLStreamProducerDisconnect, 7000, ptds_mode, NULL)

        global __cuEGLStreamProducerPresentFrame
        cuGetProcAddress_v2('cuEGLStreamProducerPresentFrame', <void **>&__cuEGLStreamProducerPresentFrame, 7000, ptds_mode, NULL)

        global __cuEGLStreamProducerReturnFrame
        cuGetProcAddress_v2('cuEGLStreamProducerReturnFrame', <void **>&__cuEGLStreamProducerReturnFrame, 7000, ptds_mode, NULL)

        global __cuGraphicsResourceGetMappedEglFrame
        cuGetProcAddress_v2('cuGraphicsResourceGetMappedEglFrame', <void **>&__cuGraphicsResourceGetMappedEglFrame, 7000, ptds_mode, NULL)

        global __cuEventCreateFromEGLSync
        cuGetProcAddress_v2('cuEventCreateFromEGLSync', <void **>&__cuEventCreateFromEGLSync, 9000, ptds_mode, NULL)

        global __cuGraphicsGLRegisterBuffer
        cuGetProcAddress_v2('cuGraphicsGLRegisterBuffer', <void **>&__cuGraphicsGLRegisterBuffer, 3000, ptds_mode, NULL)

        global __cuGraphicsGLRegisterImage
        cuGetProcAddress_v2('cuGraphicsGLRegisterImage', <void **>&__cuGraphicsGLRegisterImage, 3000, ptds_mode, NULL)

        global __cuGLGetDevices_v2
        cuGetProcAddress_v2('cuGLGetDevices', <void **>&__cuGLGetDevices_v2, 6050, ptds_mode, NULL)

        global __cuGLCtxCreate_v2
        cuGetProcAddress_v2('cuGLCtxCreate', <void **>&__cuGLCtxCreate_v2, 3020, ptds_mode, NULL)

        global __cuGLInit
        cuGetProcAddress_v2('cuGLInit', <void **>&__cuGLInit, 2000, ptds_mode, NULL)

        global __cuGLRegisterBufferObject
        cuGetProcAddress_v2('cuGLRegisterBufferObject', <void **>&__cuGLRegisterBufferObject, 2000, ptds_mode, NULL)

        global __cuGLMapBufferObject_v2
        cuGetProcAddress_v2('cuGLMapBufferObject', <void **>&__cuGLMapBufferObject_v2, 7000 if ptds_mode == CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM else 3020, ptds_mode, NULL)

        global __cuGLUnmapBufferObject
        cuGetProcAddress_v2('cuGLUnmapBufferObject', <void **>&__cuGLUnmapBufferObject, 2000, ptds_mode, NULL)

        global __cuGLUnregisterBufferObject
        cuGetProcAddress_v2('cuGLUnregisterBufferObject', <void **>&__cuGLUnregisterBufferObject, 2000, ptds_mode, NULL)

        global __cuGLSetBufferObjectMapFlags
        cuGetProcAddress_v2('cuGLSetBufferObjectMapFlags', <void **>&__cuGLSetBufferObjectMapFlags, 2030, ptds_mode, NULL)

        global __cuGLMapBufferObjectAsync_v2
        cuGetProcAddress_v2('cuGLMapBufferObjectAsync', <void **>&__cuGLMapBufferObjectAsync_v2, 7000 if ptds_mode == CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM else 3020, ptds_mode, NULL)

        global __cuGLUnmapBufferObjectAsync
        cuGetProcAddress_v2('cuGLUnmapBufferObjectAsync', <void **>&__cuGLUnmapBufferObjectAsync, 2030, ptds_mode, NULL)

        global __cuProfilerInitialize
        cuGetProcAddress_v2('cuProfilerInitialize', <void **>&__cuProfilerInitialize, 4000, ptds_mode, NULL)

        global __cuProfilerStart
        cuGetProcAddress_v2('cuProfilerStart', <void **>&__cuProfilerStart, 4000, ptds_mode, NULL)

        global __cuProfilerStop
        cuGetProcAddress_v2('cuProfilerStop', <void **>&__cuProfilerStop, 4000, ptds_mode, NULL)

        global __cuVDPAUGetDevice
        cuGetProcAddress_v2('cuVDPAUGetDevice', <void **>&__cuVDPAUGetDevice, 3010, ptds_mode, NULL)

        global __cuVDPAUCtxCreate_v2
        cuGetProcAddress_v2('cuVDPAUCtxCreate', <void **>&__cuVDPAUCtxCreate_v2, 3020, ptds_mode, NULL)

        global __cuGraphicsVDPAURegisterVideoSurface
        cuGetProcAddress_v2('cuGraphicsVDPAURegisterVideoSurface', <void **>&__cuGraphicsVDPAURegisterVideoSurface, 3010, ptds_mode, NULL)

        global __cuGraphicsVDPAURegisterOutputSurface
        cuGetProcAddress_v2('cuGraphicsVDPAURegisterOutputSurface', <void **>&__cuGraphicsVDPAURegisterOutputSurface, 3010, ptds_mode, NULL)

        global __cuDeviceGetHostAtomicCapabilities
        cuGetProcAddress_v2('cuDeviceGetHostAtomicCapabilities', <void **>&__cuDeviceGetHostAtomicCapabilities, 13000, ptds_mode, NULL)

        global __cuCtxGetDevice_v2
        cuGetProcAddress_v2('cuCtxGetDevice', <void **>&__cuCtxGetDevice_v2, 13000, ptds_mode, NULL)

        global __cuCtxSynchronize_v2
        cuGetProcAddress_v2('cuCtxSynchronize', <void **>&__cuCtxSynchronize_v2, 13000, ptds_mode, NULL)

        global __cuMemcpyBatchAsync_v2
        cuGetProcAddress_v2('cuMemcpyBatchAsync', <void **>&__cuMemcpyBatchAsync_v2, 13000, ptds_mode, NULL)

        global __cuMemcpy3DBatchAsync_v2
        cuGetProcAddress_v2('cuMemcpy3DBatchAsync', <void **>&__cuMemcpy3DBatchAsync_v2, 13000, ptds_mode, NULL)

        global __cuMemGetDefaultMemPool
        cuGetProcAddress_v2('cuMemGetDefaultMemPool', <void **>&__cuMemGetDefaultMemPool, 13000, ptds_mode, NULL)

        global __cuMemGetMemPool
        cuGetProcAddress_v2('cuMemGetMemPool', <void **>&__cuMemGetMemPool, 13000, ptds_mode, NULL)

        global __cuMemSetMemPool
        cuGetProcAddress_v2('cuMemSetMemPool', <void **>&__cuMemSetMemPool, 13000, ptds_mode, NULL)

        global __cuMemPrefetchBatchAsync
        cuGetProcAddress_v2('cuMemPrefetchBatchAsync', <void **>&__cuMemPrefetchBatchAsync, 13000, ptds_mode, NULL)

        global __cuMemDiscardBatchAsync
        cuGetProcAddress_v2('cuMemDiscardBatchAsync', <void **>&__cuMemDiscardBatchAsync, 13000, ptds_mode, NULL)

        global __cuMemDiscardAndPrefetchBatchAsync
        cuGetProcAddress_v2('cuMemDiscardAndPrefetchBatchAsync', <void **>&__cuMemDiscardAndPrefetchBatchAsync, 13000, ptds_mode, NULL)

        global __cuDeviceGetP2PAtomicCapabilities
        cuGetProcAddress_v2('cuDeviceGetP2PAtomicCapabilities', <void **>&__cuDeviceGetP2PAtomicCapabilities, 13000, ptds_mode, NULL)

        global __cuGreenCtxGetId
        cuGetProcAddress_v2('cuGreenCtxGetId', <void **>&__cuGreenCtxGetId, 13000, ptds_mode, NULL)

        global __cuMulticastBindMem_v2
        cuGetProcAddress_v2('cuMulticastBindMem', <void **>&__cuMulticastBindMem_v2, 13010, ptds_mode, NULL)

        global __cuMulticastBindAddr_v2
        cuGetProcAddress_v2('cuMulticastBindAddr', <void **>&__cuMulticastBindAddr_v2, 13010, ptds_mode, NULL)

        global __cuGraphNodeGetContainingGraph
        cuGetProcAddress_v2('cuGraphNodeGetContainingGraph', <void **>&__cuGraphNodeGetContainingGraph, 13010, ptds_mode, NULL)

        global __cuGraphNodeGetLocalId
        cuGetProcAddress_v2('cuGraphNodeGetLocalId', <void **>&__cuGraphNodeGetLocalId, 13010, ptds_mode, NULL)

        global __cuGraphNodeGetToolsId
        cuGetProcAddress_v2('cuGraphNodeGetToolsId', <void **>&__cuGraphNodeGetToolsId, 13010, ptds_mode, NULL)

        global __cuGraphGetId
        cuGetProcAddress_v2('cuGraphGetId', <void **>&__cuGraphGetId, 13010, ptds_mode, NULL)

        global __cuGraphExecGetId
        cuGetProcAddress_v2('cuGraphExecGetId', <void **>&__cuGraphExecGetId, 13010, ptds_mode, NULL)

        global __cuDevSmResourceSplit
        cuGetProcAddress_v2('cuDevSmResourceSplit', <void **>&__cuDevSmResourceSplit, 13010, ptds_mode, NULL)

        global __cuStreamGetDevResource
        cuGetProcAddress_v2('cuStreamGetDevResource', <void **>&__cuStreamGetDevResource, 13010, ptds_mode, NULL)

        global __cuKernelGetParamCount
        cuGetProcAddress_v2('cuKernelGetParamCount', <void **>&__cuKernelGetParamCount, 13020, ptds_mode, NULL)

        global __cuMemcpyWithAttributesAsync
        cuGetProcAddress_v2('cuMemcpyWithAttributesAsync', <void **>&__cuMemcpyWithAttributesAsync, 13020, ptds_mode, NULL)

        global __cuMemcpy3DWithAttributesAsync
        cuGetProcAddress_v2('cuMemcpy3DWithAttributesAsync', <void **>&__cuMemcpy3DWithAttributesAsync, 13020, ptds_mode, NULL)

        global __cuStreamBeginCaptureToCig
        cuGetProcAddress_v2('cuStreamBeginCaptureToCig', <void **>&__cuStreamBeginCaptureToCig, 13020, ptds_mode, NULL)

        global __cuStreamEndCaptureToCig
        cuGetProcAddress_v2('cuStreamEndCaptureToCig', <void **>&__cuStreamEndCaptureToCig, 13020, ptds_mode, NULL)

        global __cuFuncGetParamCount
        cuGetProcAddress_v2('cuFuncGetParamCount', <void **>&__cuFuncGetParamCount, 13020, ptds_mode, NULL)

        global __cuLaunchHostFunc_v2
        cuGetProcAddress_v2('cuLaunchHostFunc', <void **>&__cuLaunchHostFunc_v2, 13020, ptds_mode, NULL)

        global __cuGraphNodeGetParams
        cuGetProcAddress_v2('cuGraphNodeGetParams', <void **>&__cuGraphNodeGetParams, 13020, ptds_mode, NULL)

        global __cuCoredumpRegisterStartCallback
        cuGetProcAddress_v2('cuCoredumpRegisterStartCallback', <void **>&__cuCoredumpRegisterStartCallback, 13020, ptds_mode, NULL)

        global __cuCoredumpRegisterCompleteCallback
        cuGetProcAddress_v2('cuCoredumpRegisterCompleteCallback', <void **>&__cuCoredumpRegisterCompleteCallback, 13020, ptds_mode, NULL)

        global __cuCoredumpDeregisterStartCallback
        cuGetProcAddress_v2('cuCoredumpDeregisterStartCallback', <void **>&__cuCoredumpDeregisterStartCallback, 13020, ptds_mode, NULL)

        global __cuCoredumpDeregisterCompleteCallback
        cuGetProcAddress_v2('cuCoredumpDeregisterCompleteCallback', <void **>&__cuCoredumpDeregisterCompleteCallback, 13020, ptds_mode, NULL)

        global __cuLogicalEndpointIdReserve
        cuGetProcAddress_v2('cuLogicalEndpointIdReserve', <void **>&__cuLogicalEndpointIdReserve, 13030, ptds_mode, NULL)

        global __cuLogicalEndpointIdRelease
        cuGetProcAddress_v2('cuLogicalEndpointIdRelease', <void **>&__cuLogicalEndpointIdRelease, 13030, ptds_mode, NULL)

        global __cuLogicalEndpointCreate
        cuGetProcAddress_v2('cuLogicalEndpointCreate', <void **>&__cuLogicalEndpointCreate, 13030, ptds_mode, NULL)

        global __cuLogicalEndpointAddDevice
        cuGetProcAddress_v2('cuLogicalEndpointAddDevice', <void **>&__cuLogicalEndpointAddDevice, 13030, ptds_mode, NULL)

        global __cuLogicalEndpointDestroy
        cuGetProcAddress_v2('cuLogicalEndpointDestroy', <void **>&__cuLogicalEndpointDestroy, 13030, ptds_mode, NULL)

        global __cuLogicalEndpointBindAddr
        cuGetProcAddress_v2('cuLogicalEndpointBindAddr', <void **>&__cuLogicalEndpointBindAddr, 13030, ptds_mode, NULL)

        global __cuLogicalEndpointBindMem
        cuGetProcAddress_v2('cuLogicalEndpointBindMem', <void **>&__cuLogicalEndpointBindMem, 13030, ptds_mode, NULL)

        global __cuLogicalEndpointUnbind
        cuGetProcAddress_v2('cuLogicalEndpointUnbind', <void **>&__cuLogicalEndpointUnbind, 13030, ptds_mode, NULL)

        global __cuLogicalEndpointExport
        cuGetProcAddress_v2('cuLogicalEndpointExport', <void **>&__cuLogicalEndpointExport, 13030, ptds_mode, NULL)

        global __cuLogicalEndpointImport
        cuGetProcAddress_v2('cuLogicalEndpointImport', <void **>&__cuLogicalEndpointImport, 13030, ptds_mode, NULL)

        global __cuLogicalEndpointGetLimits
        cuGetProcAddress_v2('cuLogicalEndpointGetLimits', <void **>&__cuLogicalEndpointGetLimits, 13030, ptds_mode, NULL)

        global __cuLogicalEndpointQuery
        cuGetProcAddress_v2('cuLogicalEndpointQuery', <void **>&__cuLogicalEndpointQuery, 13030, ptds_mode, NULL)

        global __cuStreamBeginRecaptureToGraph
        cuGetProcAddress_v2('cuStreamBeginRecaptureToGraph', <void **>&__cuStreamBeginRecaptureToGraph, 13030, ptds_mode, NULL)

        global __cuDeviceGetFabricClusterUuid
        cuGetProcAddress_v2('cuDeviceGetFabricClusterUuid', <void **>&__cuDeviceGetFabricClusterUuid, 13041, ptds_mode, NULL)

        global __cuDeviceGetCliqueCount
        cuGetProcAddress_v2('cuDeviceGetCliqueCount', <void **>&__cuDeviceGetCliqueCount, 13041, ptds_mode, NULL)

        global __cuDeviceGetCliqueInfo
        cuGetProcAddress_v2('cuDeviceGetCliqueInfo', <void **>&__cuDeviceGetCliqueInfo, 13041, ptds_mode, NULL)

        global __cuMemGetLocationInfo
        cuGetProcAddress_v2('cuMemGetLocationInfo', <void **>&__cuMemGetLocationInfo, 13041, ptds_mode, NULL)

        global __cuGraphAddNode_v3
        cuGetProcAddress_v2('cuGraphAddNode', <void **>&__cuGraphAddNode_v3, 13041, ptds_mode, NULL)

        global __cuGraphNodeSetParams_v2
        cuGetProcAddress_v2('cuGraphNodeSetParams', <void **>&__cuGraphNodeSetParams_v2, 13041, ptds_mode, NULL)

        global __cuCheckpointOperationComplete
        cuGetProcAddress_v2('cuCheckpointOperationComplete', <void **>&__cuCheckpointOperationComplete, 13041, ptds_mode, NULL)

        global __cuFabricCpyBatchAsync
        cuGetProcAddress_v2('cuFabricCpyBatchAsync', <void **>&__cuFabricCpyBatchAsync, 13050, ptds_mode, NULL)

        global __cuTensorMapSetAllocationClass
        cuGetProcAddress_v2('cuTensorMapSetAllocationClass', <void **>&__cuTensorMapSetAllocationClass, 13050, ptds_mode, NULL)

        _cyb_atomic_int_store(<int *>&_cyb___py_driver_init, 1)
        return 0

cdef inline int _check_or_init_driver() except -1 nogil:
    if _cyb_atomic_int_load(<int *>&_cyb___py_driver_init):
        return 0

    return _init_driver()


cpdef dict _inspect_function_pointers():
    global _cyb_func_ptrs
    if _cyb_func_ptrs is not None:
        return _cyb_func_ptrs

    _check_or_init_driver()
    cdef dict data = {}
    cdef Py_ssize_t i
    for i in range(_cyb_driver_n_functions):
        data['__' + _cyb_driver_names[i].decode()] = <intptr_t>_cyb_driver_slots[i][0]
    _cyb_func_ptrs = data
    return data


cpdef _inspect_function_pointer(str name):
    global _cyb_func_ptrs
    if _cyb_func_ptrs is None:
        _cyb_func_ptrs = _inspect_function_pointers()
    return _cyb_func_ptrs[name]




cdef uintptr_t load_library() except* with gil:
    cdef uintptr_t handle = load_nvidia_dynamic_lib("cuda")._handle_uint
    return handle


###############################################################################
# Wrapper functions
###############################################################################

cdef CUresult _cuGetErrorString(CUresult error, const char** pStr) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuGetErrorString
    _check_or_init_driver()
    if __cuGetErrorString == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGetErrorString is not found")
    return (<CUresult (*)(CUresult, const char**) noexcept nogil>__cuGetErrorString)(
        error, pStr)


cdef CUresult _cuGetErrorName(CUresult error, const char** pStr) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuGetErrorName
    _check_or_init_driver()
    if __cuGetErrorName == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGetErrorName is not found")
    return (<CUresult (*)(CUresult, const char**) noexcept nogil>__cuGetErrorName)(
        error, pStr)


cdef CUresult _cuInit(unsigned int Flags) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuInit
    _check_or_init_driver()
    if __cuInit == NULL:
        with gil:
            raise FunctionNotFoundError("function cuInit is not found")
    return (<CUresult (*)(unsigned int) noexcept nogil>__cuInit)(
        Flags)


cdef CUresult _cuDriverGetVersion(int* driverVersion) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuDriverGetVersion
    _check_or_init_driver()
    if __cuDriverGetVersion == NULL:
        with gil:
            raise FunctionNotFoundError("function cuDriverGetVersion is not found")
    return (<CUresult (*)(int*) noexcept nogil>__cuDriverGetVersion)(
        driverVersion)


cdef CUresult _cuDeviceGet(CUdevice* device, int ordinal) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuDeviceGet
    _check_or_init_driver()
    if __cuDeviceGet == NULL:
        with gil:
            raise FunctionNotFoundError("function cuDeviceGet is not found")
    return (<CUresult (*)(CUdevice*, int) noexcept nogil>__cuDeviceGet)(
        device, ordinal)


cdef CUresult _cuDeviceGetCount(int* count) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuDeviceGetCount
    _check_or_init_driver()
    if __cuDeviceGetCount == NULL:
        with gil:
            raise FunctionNotFoundError("function cuDeviceGetCount is not found")
    return (<CUresult (*)(int*) noexcept nogil>__cuDeviceGetCount)(
        count)


cdef CUresult _cuDeviceGetName(char* name, int len, CUdevice dev) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuDeviceGetName
    _check_or_init_driver()
    if __cuDeviceGetName == NULL:
        with gil:
            raise FunctionNotFoundError("function cuDeviceGetName is not found")
    return (<CUresult (*)(char*, int, CUdevice) noexcept nogil>__cuDeviceGetName)(
        name, len, dev)


cdef CUresult _cuDeviceGetUuid_v2(CUuuid* uuid, CUdevice dev) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuDeviceGetUuid_v2
    _check_or_init_driver()
    if __cuDeviceGetUuid_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuDeviceGetUuid_v2 is not found")
    return (<CUresult (*)(CUuuid*, CUdevice) noexcept nogil>__cuDeviceGetUuid_v2)(
        uuid, dev)


cdef CUresult _cuDeviceGetLuid(char* luid, unsigned int* deviceNodeMask, CUdevice dev) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuDeviceGetLuid
    _check_or_init_driver()
    if __cuDeviceGetLuid == NULL:
        with gil:
            raise FunctionNotFoundError("function cuDeviceGetLuid is not found")
    return (<CUresult (*)(char*, unsigned int*, CUdevice) noexcept nogil>__cuDeviceGetLuid)(
        luid, deviceNodeMask, dev)


cdef CUresult _cuDeviceTotalMem_v2(size_t* bytes, CUdevice dev) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuDeviceTotalMem_v2
    _check_or_init_driver()
    if __cuDeviceTotalMem_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuDeviceTotalMem_v2 is not found")
    return (<CUresult (*)(size_t*, CUdevice) noexcept nogil>__cuDeviceTotalMem_v2)(
        bytes, dev)


cdef CUresult _cuDeviceGetTexture1DLinearMaxWidth(size_t* maxWidthInElements, CUarray_format format, unsigned numChannels, CUdevice dev) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuDeviceGetTexture1DLinearMaxWidth
    _check_or_init_driver()
    if __cuDeviceGetTexture1DLinearMaxWidth == NULL:
        with gil:
            raise FunctionNotFoundError("function cuDeviceGetTexture1DLinearMaxWidth is not found")
    return (<CUresult (*)(size_t*, CUarray_format, unsigned, CUdevice) noexcept nogil>__cuDeviceGetTexture1DLinearMaxWidth)(
        maxWidthInElements, format, numChannels, dev)


cdef CUresult _cuDeviceGetAttribute(int* pi, CUdevice_attribute attrib, CUdevice dev) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuDeviceGetAttribute
    _check_or_init_driver()
    if __cuDeviceGetAttribute == NULL:
        with gil:
            raise FunctionNotFoundError("function cuDeviceGetAttribute is not found")
    return (<CUresult (*)(int*, CUdevice_attribute, CUdevice) noexcept nogil>__cuDeviceGetAttribute)(
        pi, attrib, dev)


cdef CUresult _cuDeviceGetNvSciSyncAttributes(void* nvSciSyncAttrList, CUdevice dev, int flags) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuDeviceGetNvSciSyncAttributes
    _check_or_init_driver()
    if __cuDeviceGetNvSciSyncAttributes == NULL:
        with gil:
            raise FunctionNotFoundError("function cuDeviceGetNvSciSyncAttributes is not found")
    return (<CUresult (*)(void*, CUdevice, int) noexcept nogil>__cuDeviceGetNvSciSyncAttributes)(
        nvSciSyncAttrList, dev, flags)


cdef CUresult _cuDeviceSetMemPool(CUdevice dev, CUmemoryPool pool) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuDeviceSetMemPool
    _check_or_init_driver()
    if __cuDeviceSetMemPool == NULL:
        with gil:
            raise FunctionNotFoundError("function cuDeviceSetMemPool is not found")
    return (<CUresult (*)(CUdevice, CUmemoryPool) noexcept nogil>__cuDeviceSetMemPool)(
        dev, pool)


cdef CUresult _cuDeviceGetMemPool(CUmemoryPool* pool, CUdevice dev) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuDeviceGetMemPool
    _check_or_init_driver()
    if __cuDeviceGetMemPool == NULL:
        with gil:
            raise FunctionNotFoundError("function cuDeviceGetMemPool is not found")
    return (<CUresult (*)(CUmemoryPool*, CUdevice) noexcept nogil>__cuDeviceGetMemPool)(
        pool, dev)


cdef CUresult _cuDeviceGetDefaultMemPool(CUmemoryPool* pool_out, CUdevice dev) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuDeviceGetDefaultMemPool
    _check_or_init_driver()
    if __cuDeviceGetDefaultMemPool == NULL:
        with gil:
            raise FunctionNotFoundError("function cuDeviceGetDefaultMemPool is not found")
    return (<CUresult (*)(CUmemoryPool*, CUdevice) noexcept nogil>__cuDeviceGetDefaultMemPool)(
        pool_out, dev)


cdef CUresult _cuDeviceGetExecAffinitySupport(int* pi, CUexecAffinityType type, CUdevice dev) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuDeviceGetExecAffinitySupport
    _check_or_init_driver()
    if __cuDeviceGetExecAffinitySupport == NULL:
        with gil:
            raise FunctionNotFoundError("function cuDeviceGetExecAffinitySupport is not found")
    return (<CUresult (*)(int*, CUexecAffinityType, CUdevice) noexcept nogil>__cuDeviceGetExecAffinitySupport)(
        pi, type, dev)


cdef CUresult _cuFlushGPUDirectRDMAWrites(CUflushGPUDirectRDMAWritesTarget target, CUflushGPUDirectRDMAWritesScope scope) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuFlushGPUDirectRDMAWrites
    _check_or_init_driver()
    if __cuFlushGPUDirectRDMAWrites == NULL:
        with gil:
            raise FunctionNotFoundError("function cuFlushGPUDirectRDMAWrites is not found")
    return (<CUresult (*)(CUflushGPUDirectRDMAWritesTarget, CUflushGPUDirectRDMAWritesScope) noexcept nogil>__cuFlushGPUDirectRDMAWrites)(
        target, scope)


cdef CUresult _cuDeviceGetProperties(CUdevprop* prop, CUdevice dev) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuDeviceGetProperties
    _check_or_init_driver()
    if __cuDeviceGetProperties == NULL:
        with gil:
            raise FunctionNotFoundError("function cuDeviceGetProperties is not found")
    return (<CUresult (*)(CUdevprop*, CUdevice) noexcept nogil>__cuDeviceGetProperties)(
        prop, dev)


cdef CUresult _cuDeviceComputeCapability(int* major, int* minor, CUdevice dev) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuDeviceComputeCapability
    _check_or_init_driver()
    if __cuDeviceComputeCapability == NULL:
        with gil:
            raise FunctionNotFoundError("function cuDeviceComputeCapability is not found")
    return (<CUresult (*)(int*, int*, CUdevice) noexcept nogil>__cuDeviceComputeCapability)(
        major, minor, dev)


cdef CUresult _cuDevicePrimaryCtxRetain(CUcontext* pctx, CUdevice dev) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuDevicePrimaryCtxRetain
    _check_or_init_driver()
    if __cuDevicePrimaryCtxRetain == NULL:
        with gil:
            raise FunctionNotFoundError("function cuDevicePrimaryCtxRetain is not found")
    return (<CUresult (*)(CUcontext*, CUdevice) noexcept nogil>__cuDevicePrimaryCtxRetain)(
        pctx, dev)


cdef CUresult _cuDevicePrimaryCtxRelease_v2(CUdevice dev) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuDevicePrimaryCtxRelease_v2
    _check_or_init_driver()
    if __cuDevicePrimaryCtxRelease_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuDevicePrimaryCtxRelease_v2 is not found")
    return (<CUresult (*)(CUdevice) noexcept nogil>__cuDevicePrimaryCtxRelease_v2)(
        dev)


cdef CUresult _cuDevicePrimaryCtxSetFlags_v2(CUdevice dev, unsigned int flags) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuDevicePrimaryCtxSetFlags_v2
    _check_or_init_driver()
    if __cuDevicePrimaryCtxSetFlags_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuDevicePrimaryCtxSetFlags_v2 is not found")
    return (<CUresult (*)(CUdevice, unsigned int) noexcept nogil>__cuDevicePrimaryCtxSetFlags_v2)(
        dev, flags)


cdef CUresult _cuDevicePrimaryCtxGetState(CUdevice dev, unsigned int* flags, int* active) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuDevicePrimaryCtxGetState
    _check_or_init_driver()
    if __cuDevicePrimaryCtxGetState == NULL:
        with gil:
            raise FunctionNotFoundError("function cuDevicePrimaryCtxGetState is not found")
    return (<CUresult (*)(CUdevice, unsigned int*, int*) noexcept nogil>__cuDevicePrimaryCtxGetState)(
        dev, flags, active)


cdef CUresult _cuDevicePrimaryCtxReset_v2(CUdevice dev) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuDevicePrimaryCtxReset_v2
    _check_or_init_driver()
    if __cuDevicePrimaryCtxReset_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuDevicePrimaryCtxReset_v2 is not found")
    return (<CUresult (*)(CUdevice) noexcept nogil>__cuDevicePrimaryCtxReset_v2)(
        dev)


cdef CUresult _cuCtxCreate_v2(CUcontext* pctx, unsigned int flags, CUdevice dev) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuCtxCreate_v2
    _check_or_init_driver()
    if __cuCtxCreate_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuCtxCreate_v2 is not found")
    return (<CUresult (*)(CUcontext*, unsigned int, CUdevice) noexcept nogil>__cuCtxCreate_v2)(
        pctx, flags, dev)


cdef CUresult _cuCtxCreate_v3(CUcontext* pctx, CUexecAffinityParam* paramsArray, int numParams, unsigned int flags, CUdevice dev) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuCtxCreate_v3
    _check_or_init_driver()
    if __cuCtxCreate_v3 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuCtxCreate_v3 is not found")
    return (<CUresult (*)(CUcontext*, CUexecAffinityParam*, int, unsigned int, CUdevice) noexcept nogil>__cuCtxCreate_v3)(
        pctx, paramsArray, numParams, flags, dev)


cdef CUresult _cuCtxCreate_v4(CUcontext* pctx, CUctxCreateParams* ctxCreateParams, unsigned int flags, CUdevice dev) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuCtxCreate_v4
    _check_or_init_driver()
    if __cuCtxCreate_v4 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuCtxCreate_v4 is not found")
    return (<CUresult (*)(CUcontext*, CUctxCreateParams*, unsigned int, CUdevice) noexcept nogil>__cuCtxCreate_v4)(
        pctx, ctxCreateParams, flags, dev)


cdef CUresult _cuCtxDestroy_v2(CUcontext ctx) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuCtxDestroy_v2
    _check_or_init_driver()
    if __cuCtxDestroy_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuCtxDestroy_v2 is not found")
    return (<CUresult (*)(CUcontext) noexcept nogil>__cuCtxDestroy_v2)(
        ctx)


cdef CUresult _cuCtxPushCurrent_v2(CUcontext ctx) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuCtxPushCurrent_v2
    _check_or_init_driver()
    if __cuCtxPushCurrent_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuCtxPushCurrent_v2 is not found")
    return (<CUresult (*)(CUcontext) noexcept nogil>__cuCtxPushCurrent_v2)(
        ctx)


cdef CUresult _cuCtxPopCurrent_v2(CUcontext* pctx) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuCtxPopCurrent_v2
    _check_or_init_driver()
    if __cuCtxPopCurrent_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuCtxPopCurrent_v2 is not found")
    return (<CUresult (*)(CUcontext*) noexcept nogil>__cuCtxPopCurrent_v2)(
        pctx)


cdef CUresult _cuCtxSetCurrent(CUcontext ctx) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuCtxSetCurrent
    _check_or_init_driver()
    if __cuCtxSetCurrent == NULL:
        with gil:
            raise FunctionNotFoundError("function cuCtxSetCurrent is not found")
    return (<CUresult (*)(CUcontext) noexcept nogil>__cuCtxSetCurrent)(
        ctx)


cdef CUresult _cuCtxGetCurrent(CUcontext* pctx) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuCtxGetCurrent
    _check_or_init_driver()
    if __cuCtxGetCurrent == NULL:
        with gil:
            raise FunctionNotFoundError("function cuCtxGetCurrent is not found")
    return (<CUresult (*)(CUcontext*) noexcept nogil>__cuCtxGetCurrent)(
        pctx)


cdef CUresult _cuCtxGetDevice(CUdevice* device) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuCtxGetDevice
    _check_or_init_driver()
    if __cuCtxGetDevice == NULL:
        with gil:
            raise FunctionNotFoundError("function cuCtxGetDevice is not found")
    return (<CUresult (*)(CUdevice*) noexcept nogil>__cuCtxGetDevice)(
        device)


cdef CUresult _cuCtxGetFlags(unsigned int* flags) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuCtxGetFlags
    _check_or_init_driver()
    if __cuCtxGetFlags == NULL:
        with gil:
            raise FunctionNotFoundError("function cuCtxGetFlags is not found")
    return (<CUresult (*)(unsigned int*) noexcept nogil>__cuCtxGetFlags)(
        flags)


cdef CUresult _cuCtxSetFlags(unsigned int flags) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuCtxSetFlags
    _check_or_init_driver()
    if __cuCtxSetFlags == NULL:
        with gil:
            raise FunctionNotFoundError("function cuCtxSetFlags is not found")
    return (<CUresult (*)(unsigned int) noexcept nogil>__cuCtxSetFlags)(
        flags)


cdef CUresult _cuCtxGetId(CUcontext ctx, unsigned long long* ctxId) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuCtxGetId
    _check_or_init_driver()
    if __cuCtxGetId == NULL:
        with gil:
            raise FunctionNotFoundError("function cuCtxGetId is not found")
    return (<CUresult (*)(CUcontext, unsigned long long*) noexcept nogil>__cuCtxGetId)(
        ctx, ctxId)


cdef CUresult _cuCtxSynchronize() except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuCtxSynchronize
    _check_or_init_driver()
    if __cuCtxSynchronize == NULL:
        with gil:
            raise FunctionNotFoundError("function cuCtxSynchronize is not found")
    return (<CUresult (*)() noexcept nogil>__cuCtxSynchronize)(
        )


cdef CUresult _cuCtxSetLimit(CUlimit limit, size_t value) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuCtxSetLimit
    _check_or_init_driver()
    if __cuCtxSetLimit == NULL:
        with gil:
            raise FunctionNotFoundError("function cuCtxSetLimit is not found")
    return (<CUresult (*)(CUlimit, size_t) noexcept nogil>__cuCtxSetLimit)(
        limit, value)


cdef CUresult _cuCtxGetLimit(size_t* pvalue, CUlimit limit) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuCtxGetLimit
    _check_or_init_driver()
    if __cuCtxGetLimit == NULL:
        with gil:
            raise FunctionNotFoundError("function cuCtxGetLimit is not found")
    return (<CUresult (*)(size_t*, CUlimit) noexcept nogil>__cuCtxGetLimit)(
        pvalue, limit)


cdef CUresult _cuCtxGetCacheConfig(CUfunc_cache* pconfig) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuCtxGetCacheConfig
    _check_or_init_driver()
    if __cuCtxGetCacheConfig == NULL:
        with gil:
            raise FunctionNotFoundError("function cuCtxGetCacheConfig is not found")
    return (<CUresult (*)(CUfunc_cache*) noexcept nogil>__cuCtxGetCacheConfig)(
        pconfig)


cdef CUresult _cuCtxSetCacheConfig(CUfunc_cache config) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuCtxSetCacheConfig
    _check_or_init_driver()
    if __cuCtxSetCacheConfig == NULL:
        with gil:
            raise FunctionNotFoundError("function cuCtxSetCacheConfig is not found")
    return (<CUresult (*)(CUfunc_cache) noexcept nogil>__cuCtxSetCacheConfig)(
        config)


cdef CUresult _cuCtxGetApiVersion(CUcontext ctx, unsigned int* version) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuCtxGetApiVersion
    _check_or_init_driver()
    if __cuCtxGetApiVersion == NULL:
        with gil:
            raise FunctionNotFoundError("function cuCtxGetApiVersion is not found")
    return (<CUresult (*)(CUcontext, unsigned int*) noexcept nogil>__cuCtxGetApiVersion)(
        ctx, version)


cdef CUresult _cuCtxGetStreamPriorityRange(int* leastPriority, int* greatestPriority) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuCtxGetStreamPriorityRange
    _check_or_init_driver()
    if __cuCtxGetStreamPriorityRange == NULL:
        with gil:
            raise FunctionNotFoundError("function cuCtxGetStreamPriorityRange is not found")
    return (<CUresult (*)(int*, int*) noexcept nogil>__cuCtxGetStreamPriorityRange)(
        leastPriority, greatestPriority)


cdef CUresult _cuCtxResetPersistingL2Cache() except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuCtxResetPersistingL2Cache
    _check_or_init_driver()
    if __cuCtxResetPersistingL2Cache == NULL:
        with gil:
            raise FunctionNotFoundError("function cuCtxResetPersistingL2Cache is not found")
    return (<CUresult (*)() noexcept nogil>__cuCtxResetPersistingL2Cache)(
        )


cdef CUresult _cuCtxGetExecAffinity(CUexecAffinityParam* pExecAffinity, CUexecAffinityType type) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuCtxGetExecAffinity
    _check_or_init_driver()
    if __cuCtxGetExecAffinity == NULL:
        with gil:
            raise FunctionNotFoundError("function cuCtxGetExecAffinity is not found")
    return (<CUresult (*)(CUexecAffinityParam*, CUexecAffinityType) noexcept nogil>__cuCtxGetExecAffinity)(
        pExecAffinity, type)


cdef CUresult _cuCtxRecordEvent(CUcontext hCtx, CUevent hEvent) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuCtxRecordEvent
    _check_or_init_driver()
    if __cuCtxRecordEvent == NULL:
        with gil:
            raise FunctionNotFoundError("function cuCtxRecordEvent is not found")
    return (<CUresult (*)(CUcontext, CUevent) noexcept nogil>__cuCtxRecordEvent)(
        hCtx, hEvent)


cdef CUresult _cuCtxWaitEvent(CUcontext hCtx, CUevent hEvent) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuCtxWaitEvent
    _check_or_init_driver()
    if __cuCtxWaitEvent == NULL:
        with gil:
            raise FunctionNotFoundError("function cuCtxWaitEvent is not found")
    return (<CUresult (*)(CUcontext, CUevent) noexcept nogil>__cuCtxWaitEvent)(
        hCtx, hEvent)


cdef CUresult _cuCtxAttach(CUcontext* pctx, unsigned int flags) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuCtxAttach
    _check_or_init_driver()
    if __cuCtxAttach == NULL:
        with gil:
            raise FunctionNotFoundError("function cuCtxAttach is not found")
    return (<CUresult (*)(CUcontext*, unsigned int) noexcept nogil>__cuCtxAttach)(
        pctx, flags)


cdef CUresult _cuCtxDetach(CUcontext ctx) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuCtxDetach
    _check_or_init_driver()
    if __cuCtxDetach == NULL:
        with gil:
            raise FunctionNotFoundError("function cuCtxDetach is not found")
    return (<CUresult (*)(CUcontext) noexcept nogil>__cuCtxDetach)(
        ctx)


cdef CUresult _cuCtxGetSharedMemConfig(CUsharedconfig* pConfig) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuCtxGetSharedMemConfig
    _check_or_init_driver()
    if __cuCtxGetSharedMemConfig == NULL:
        with gil:
            raise FunctionNotFoundError("function cuCtxGetSharedMemConfig is not found")
    return (<CUresult (*)(CUsharedconfig*) noexcept nogil>__cuCtxGetSharedMemConfig)(
        pConfig)


cdef CUresult _cuCtxSetSharedMemConfig(CUsharedconfig config) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuCtxSetSharedMemConfig
    _check_or_init_driver()
    if __cuCtxSetSharedMemConfig == NULL:
        with gil:
            raise FunctionNotFoundError("function cuCtxSetSharedMemConfig is not found")
    return (<CUresult (*)(CUsharedconfig) noexcept nogil>__cuCtxSetSharedMemConfig)(
        config)


cdef CUresult _cuModuleLoad(CUmodule* module, const char* fname) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuModuleLoad
    _check_or_init_driver()
    if __cuModuleLoad == NULL:
        with gil:
            raise FunctionNotFoundError("function cuModuleLoad is not found")
    return (<CUresult (*)(CUmodule*, const char*) noexcept nogil>__cuModuleLoad)(
        module, fname)


cdef CUresult _cuModuleLoadData(CUmodule* module, const void* image) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuModuleLoadData
    _check_or_init_driver()
    if __cuModuleLoadData == NULL:
        with gil:
            raise FunctionNotFoundError("function cuModuleLoadData is not found")
    return (<CUresult (*)(CUmodule*, const void*) noexcept nogil>__cuModuleLoadData)(
        module, image)


cdef CUresult _cuModuleLoadDataEx(CUmodule* module, const void* image, unsigned int numOptions, CUjit_option* options, void** optionValues) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuModuleLoadDataEx
    _check_or_init_driver()
    if __cuModuleLoadDataEx == NULL:
        with gil:
            raise FunctionNotFoundError("function cuModuleLoadDataEx is not found")
    return (<CUresult (*)(CUmodule*, const void*, unsigned int, CUjit_option*, void**) noexcept nogil>__cuModuleLoadDataEx)(
        module, image, numOptions, options, optionValues)


cdef CUresult _cuModuleLoadFatBinary(CUmodule* module, const void* fatCubin) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuModuleLoadFatBinary
    _check_or_init_driver()
    if __cuModuleLoadFatBinary == NULL:
        with gil:
            raise FunctionNotFoundError("function cuModuleLoadFatBinary is not found")
    return (<CUresult (*)(CUmodule*, const void*) noexcept nogil>__cuModuleLoadFatBinary)(
        module, fatCubin)


cdef CUresult _cuModuleUnload(CUmodule hmod) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuModuleUnload
    _check_or_init_driver()
    if __cuModuleUnload == NULL:
        with gil:
            raise FunctionNotFoundError("function cuModuleUnload is not found")
    return (<CUresult (*)(CUmodule) noexcept nogil>__cuModuleUnload)(
        hmod)


cdef CUresult _cuModuleGetLoadingMode(CUmoduleLoadingMode* mode) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuModuleGetLoadingMode
    _check_or_init_driver()
    if __cuModuleGetLoadingMode == NULL:
        with gil:
            raise FunctionNotFoundError("function cuModuleGetLoadingMode is not found")
    return (<CUresult (*)(CUmoduleLoadingMode*) noexcept nogil>__cuModuleGetLoadingMode)(
        mode)


cdef CUresult _cuModuleGetFunction(CUfunction* hfunc, CUmodule hmod, const char* name) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuModuleGetFunction
    _check_or_init_driver()
    if __cuModuleGetFunction == NULL:
        with gil:
            raise FunctionNotFoundError("function cuModuleGetFunction is not found")
    return (<CUresult (*)(CUfunction*, CUmodule, const char*) noexcept nogil>__cuModuleGetFunction)(
        hfunc, hmod, name)


cdef CUresult _cuModuleGetFunctionCount(unsigned int* count, CUmodule mod) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuModuleGetFunctionCount
    _check_or_init_driver()
    if __cuModuleGetFunctionCount == NULL:
        with gil:
            raise FunctionNotFoundError("function cuModuleGetFunctionCount is not found")
    return (<CUresult (*)(unsigned int*, CUmodule) noexcept nogil>__cuModuleGetFunctionCount)(
        count, mod)


cdef CUresult _cuModuleEnumerateFunctions(CUfunction* functions, unsigned int numFunctions, CUmodule mod) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuModuleEnumerateFunctions
    _check_or_init_driver()
    if __cuModuleEnumerateFunctions == NULL:
        with gil:
            raise FunctionNotFoundError("function cuModuleEnumerateFunctions is not found")
    return (<CUresult (*)(CUfunction*, unsigned int, CUmodule) noexcept nogil>__cuModuleEnumerateFunctions)(
        functions, numFunctions, mod)


cdef CUresult _cuModuleGetGlobal_v2(CUdeviceptr* dptr, size_t* bytes, CUmodule hmod, const char* name) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuModuleGetGlobal_v2
    _check_or_init_driver()
    if __cuModuleGetGlobal_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuModuleGetGlobal_v2 is not found")
    return (<CUresult (*)(CUdeviceptr*, size_t*, CUmodule, const char*) noexcept nogil>__cuModuleGetGlobal_v2)(
        dptr, bytes, hmod, name)


cdef CUresult _cuLinkCreate_v2(unsigned int numOptions, CUjit_option* options, void** optionValues, CUlinkState* stateOut) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuLinkCreate_v2
    _check_or_init_driver()
    if __cuLinkCreate_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuLinkCreate_v2 is not found")
    return (<CUresult (*)(unsigned int, CUjit_option*, void**, CUlinkState*) noexcept nogil>__cuLinkCreate_v2)(
        numOptions, options, optionValues, stateOut)


cdef CUresult _cuLinkAddData_v2(CUlinkState state, CUjitInputType type, void* data, size_t size, const char* name, unsigned int numOptions, CUjit_option* options, void** optionValues) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuLinkAddData_v2
    _check_or_init_driver()
    if __cuLinkAddData_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuLinkAddData_v2 is not found")
    return (<CUresult (*)(CUlinkState, CUjitInputType, void*, size_t, const char*, unsigned int, CUjit_option*, void**) noexcept nogil>__cuLinkAddData_v2)(
        state, type, data, size, name, numOptions, options, optionValues)


cdef CUresult _cuLinkAddFile_v2(CUlinkState state, CUjitInputType type, const char* path, unsigned int numOptions, CUjit_option* options, void** optionValues) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuLinkAddFile_v2
    _check_or_init_driver()
    if __cuLinkAddFile_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuLinkAddFile_v2 is not found")
    return (<CUresult (*)(CUlinkState, CUjitInputType, const char*, unsigned int, CUjit_option*, void**) noexcept nogil>__cuLinkAddFile_v2)(
        state, type, path, numOptions, options, optionValues)


cdef CUresult _cuLinkComplete(CUlinkState state, void** cubinOut, size_t* sizeOut) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuLinkComplete
    _check_or_init_driver()
    if __cuLinkComplete == NULL:
        with gil:
            raise FunctionNotFoundError("function cuLinkComplete is not found")
    return (<CUresult (*)(CUlinkState, void**, size_t*) noexcept nogil>__cuLinkComplete)(
        state, cubinOut, sizeOut)


cdef CUresult _cuLinkDestroy(CUlinkState state) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuLinkDestroy
    _check_or_init_driver()
    if __cuLinkDestroy == NULL:
        with gil:
            raise FunctionNotFoundError("function cuLinkDestroy is not found")
    return (<CUresult (*)(CUlinkState) noexcept nogil>__cuLinkDestroy)(
        state)


cdef CUresult _cuModuleGetTexRef(CUtexref* pTexRef, CUmodule hmod, const char* name) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuModuleGetTexRef
    _check_or_init_driver()
    if __cuModuleGetTexRef == NULL:
        with gil:
            raise FunctionNotFoundError("function cuModuleGetTexRef is not found")
    return (<CUresult (*)(CUtexref*, CUmodule, const char*) noexcept nogil>__cuModuleGetTexRef)(
        pTexRef, hmod, name)


cdef CUresult _cuModuleGetSurfRef(CUsurfref* pSurfRef, CUmodule hmod, const char* name) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuModuleGetSurfRef
    _check_or_init_driver()
    if __cuModuleGetSurfRef == NULL:
        with gil:
            raise FunctionNotFoundError("function cuModuleGetSurfRef is not found")
    return (<CUresult (*)(CUsurfref*, CUmodule, const char*) noexcept nogil>__cuModuleGetSurfRef)(
        pSurfRef, hmod, name)


cdef CUresult _cuLibraryLoadData(CUlibrary* library, const void* code, CUjit_option* jitOptions, void** jitOptionsValues, unsigned int numJitOptions, CUlibraryOption* libraryOptions, void** libraryOptionValues, unsigned int numLibraryOptions) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuLibraryLoadData
    _check_or_init_driver()
    if __cuLibraryLoadData == NULL:
        with gil:
            raise FunctionNotFoundError("function cuLibraryLoadData is not found")
    return (<CUresult (*)(CUlibrary*, const void*, CUjit_option*, void**, unsigned int, CUlibraryOption*, void**, unsigned int) noexcept nogil>__cuLibraryLoadData)(
        library, code, jitOptions, jitOptionsValues, numJitOptions, libraryOptions, libraryOptionValues, numLibraryOptions)


cdef CUresult _cuLibraryLoadFromFile(CUlibrary* library, const char* fileName, CUjit_option* jitOptions, void** jitOptionsValues, unsigned int numJitOptions, CUlibraryOption* libraryOptions, void** libraryOptionValues, unsigned int numLibraryOptions) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuLibraryLoadFromFile
    _check_or_init_driver()
    if __cuLibraryLoadFromFile == NULL:
        with gil:
            raise FunctionNotFoundError("function cuLibraryLoadFromFile is not found")
    return (<CUresult (*)(CUlibrary*, const char*, CUjit_option*, void**, unsigned int, CUlibraryOption*, void**, unsigned int) noexcept nogil>__cuLibraryLoadFromFile)(
        library, fileName, jitOptions, jitOptionsValues, numJitOptions, libraryOptions, libraryOptionValues, numLibraryOptions)


cdef CUresult _cuLibraryUnload(CUlibrary library) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuLibraryUnload
    _check_or_init_driver()
    if __cuLibraryUnload == NULL:
        with gil:
            raise FunctionNotFoundError("function cuLibraryUnload is not found")
    return (<CUresult (*)(CUlibrary) noexcept nogil>__cuLibraryUnload)(
        library)


cdef CUresult _cuLibraryGetKernel(CUkernel* pKernel, CUlibrary library, const char* name) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuLibraryGetKernel
    _check_or_init_driver()
    if __cuLibraryGetKernel == NULL:
        with gil:
            raise FunctionNotFoundError("function cuLibraryGetKernel is not found")
    return (<CUresult (*)(CUkernel*, CUlibrary, const char*) noexcept nogil>__cuLibraryGetKernel)(
        pKernel, library, name)


cdef CUresult _cuLibraryGetKernelCount(unsigned int* count, CUlibrary lib) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuLibraryGetKernelCount
    _check_or_init_driver()
    if __cuLibraryGetKernelCount == NULL:
        with gil:
            raise FunctionNotFoundError("function cuLibraryGetKernelCount is not found")
    return (<CUresult (*)(unsigned int*, CUlibrary) noexcept nogil>__cuLibraryGetKernelCount)(
        count, lib)


cdef CUresult _cuLibraryEnumerateKernels(CUkernel* kernels, unsigned int numKernels, CUlibrary lib) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuLibraryEnumerateKernels
    _check_or_init_driver()
    if __cuLibraryEnumerateKernels == NULL:
        with gil:
            raise FunctionNotFoundError("function cuLibraryEnumerateKernels is not found")
    return (<CUresult (*)(CUkernel*, unsigned int, CUlibrary) noexcept nogil>__cuLibraryEnumerateKernels)(
        kernels, numKernels, lib)


cdef CUresult _cuLibraryGetModule(CUmodule* pMod, CUlibrary library) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuLibraryGetModule
    _check_or_init_driver()
    if __cuLibraryGetModule == NULL:
        with gil:
            raise FunctionNotFoundError("function cuLibraryGetModule is not found")
    return (<CUresult (*)(CUmodule*, CUlibrary) noexcept nogil>__cuLibraryGetModule)(
        pMod, library)


cdef CUresult _cuKernelGetFunction(CUfunction* pFunc, CUkernel kernel) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuKernelGetFunction
    _check_or_init_driver()
    if __cuKernelGetFunction == NULL:
        with gil:
            raise FunctionNotFoundError("function cuKernelGetFunction is not found")
    return (<CUresult (*)(CUfunction*, CUkernel) noexcept nogil>__cuKernelGetFunction)(
        pFunc, kernel)


cdef CUresult _cuKernelGetLibrary(CUlibrary* pLib, CUkernel kernel) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuKernelGetLibrary
    _check_or_init_driver()
    if __cuKernelGetLibrary == NULL:
        with gil:
            raise FunctionNotFoundError("function cuKernelGetLibrary is not found")
    return (<CUresult (*)(CUlibrary*, CUkernel) noexcept nogil>__cuKernelGetLibrary)(
        pLib, kernel)


cdef CUresult _cuLibraryGetGlobal(CUdeviceptr* dptr, size_t* bytes, CUlibrary library, const char* name) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuLibraryGetGlobal
    _check_or_init_driver()
    if __cuLibraryGetGlobal == NULL:
        with gil:
            raise FunctionNotFoundError("function cuLibraryGetGlobal is not found")
    return (<CUresult (*)(CUdeviceptr*, size_t*, CUlibrary, const char*) noexcept nogil>__cuLibraryGetGlobal)(
        dptr, bytes, library, name)


cdef CUresult _cuLibraryGetManaged(CUdeviceptr* dptr, size_t* bytes, CUlibrary library, const char* name) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuLibraryGetManaged
    _check_or_init_driver()
    if __cuLibraryGetManaged == NULL:
        with gil:
            raise FunctionNotFoundError("function cuLibraryGetManaged is not found")
    return (<CUresult (*)(CUdeviceptr*, size_t*, CUlibrary, const char*) noexcept nogil>__cuLibraryGetManaged)(
        dptr, bytes, library, name)


cdef CUresult _cuLibraryGetUnifiedFunction(void** fptr, CUlibrary library, const char* symbol) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuLibraryGetUnifiedFunction
    _check_or_init_driver()
    if __cuLibraryGetUnifiedFunction == NULL:
        with gil:
            raise FunctionNotFoundError("function cuLibraryGetUnifiedFunction is not found")
    return (<CUresult (*)(void**, CUlibrary, const char*) noexcept nogil>__cuLibraryGetUnifiedFunction)(
        fptr, library, symbol)


cdef CUresult _cuKernelGetAttribute(int* pi, CUfunction_attribute attrib, CUkernel kernel, CUdevice dev) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuKernelGetAttribute
    _check_or_init_driver()
    if __cuKernelGetAttribute == NULL:
        with gil:
            raise FunctionNotFoundError("function cuKernelGetAttribute is not found")
    return (<CUresult (*)(int*, CUfunction_attribute, CUkernel, CUdevice) noexcept nogil>__cuKernelGetAttribute)(
        pi, attrib, kernel, dev)


cdef CUresult _cuKernelSetAttribute(CUfunction_attribute attrib, int val, CUkernel kernel, CUdevice dev) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuKernelSetAttribute
    _check_or_init_driver()
    if __cuKernelSetAttribute == NULL:
        with gil:
            raise FunctionNotFoundError("function cuKernelSetAttribute is not found")
    return (<CUresult (*)(CUfunction_attribute, int, CUkernel, CUdevice) noexcept nogil>__cuKernelSetAttribute)(
        attrib, val, kernel, dev)


cdef CUresult _cuKernelSetCacheConfig(CUkernel kernel, CUfunc_cache config, CUdevice dev) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuKernelSetCacheConfig
    _check_or_init_driver()
    if __cuKernelSetCacheConfig == NULL:
        with gil:
            raise FunctionNotFoundError("function cuKernelSetCacheConfig is not found")
    return (<CUresult (*)(CUkernel, CUfunc_cache, CUdevice) noexcept nogil>__cuKernelSetCacheConfig)(
        kernel, config, dev)


cdef CUresult _cuKernelGetName(const char** name, CUkernel hfunc) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuKernelGetName
    _check_or_init_driver()
    if __cuKernelGetName == NULL:
        with gil:
            raise FunctionNotFoundError("function cuKernelGetName is not found")
    return (<CUresult (*)(const char**, CUkernel) noexcept nogil>__cuKernelGetName)(
        name, hfunc)


cdef CUresult _cuKernelGetParamInfo(CUkernel kernel, size_t paramIndex, size_t* paramOffset, size_t* paramSize) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuKernelGetParamInfo
    _check_or_init_driver()
    if __cuKernelGetParamInfo == NULL:
        with gil:
            raise FunctionNotFoundError("function cuKernelGetParamInfo is not found")
    return (<CUresult (*)(CUkernel, size_t, size_t*, size_t*) noexcept nogil>__cuKernelGetParamInfo)(
        kernel, paramIndex, paramOffset, paramSize)


cdef CUresult _cuMemGetInfo_v2(size_t* free, size_t* total) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuMemGetInfo_v2
    _check_or_init_driver()
    if __cuMemGetInfo_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemGetInfo_v2 is not found")
    return (<CUresult (*)(size_t*, size_t*) noexcept nogil>__cuMemGetInfo_v2)(
        free, total)


cdef CUresult _cuMemAlloc_v2(CUdeviceptr* dptr, size_t bytesize) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuMemAlloc_v2
    _check_or_init_driver()
    if __cuMemAlloc_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemAlloc_v2 is not found")
    return (<CUresult (*)(CUdeviceptr*, size_t) noexcept nogil>__cuMemAlloc_v2)(
        dptr, bytesize)


cdef CUresult _cuMemAllocPitch_v2(CUdeviceptr* dptr, size_t* pPitch, size_t WidthInBytes, size_t Height, unsigned int ElementSizeBytes) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuMemAllocPitch_v2
    _check_or_init_driver()
    if __cuMemAllocPitch_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemAllocPitch_v2 is not found")
    return (<CUresult (*)(CUdeviceptr*, size_t*, size_t, size_t, unsigned int) noexcept nogil>__cuMemAllocPitch_v2)(
        dptr, pPitch, WidthInBytes, Height, ElementSizeBytes)


cdef CUresult _cuMemFree_v2(CUdeviceptr dptr) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuMemFree_v2
    _check_or_init_driver()
    if __cuMemFree_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemFree_v2 is not found")
    return (<CUresult (*)(CUdeviceptr) noexcept nogil>__cuMemFree_v2)(
        dptr)


cdef CUresult _cuMemGetAddressRange_v2(CUdeviceptr* pbase, size_t* psize, CUdeviceptr dptr) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuMemGetAddressRange_v2
    _check_or_init_driver()
    if __cuMemGetAddressRange_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemGetAddressRange_v2 is not found")
    return (<CUresult (*)(CUdeviceptr*, size_t*, CUdeviceptr) noexcept nogil>__cuMemGetAddressRange_v2)(
        pbase, psize, dptr)


cdef CUresult _cuMemAllocHost_v2(void** pp, size_t bytesize) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuMemAllocHost_v2
    _check_or_init_driver()
    if __cuMemAllocHost_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemAllocHost_v2 is not found")
    return (<CUresult (*)(void**, size_t) noexcept nogil>__cuMemAllocHost_v2)(
        pp, bytesize)


cdef CUresult _cuMemFreeHost(void* p) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuMemFreeHost
    _check_or_init_driver()
    if __cuMemFreeHost == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemFreeHost is not found")
    return (<CUresult (*)(void*) noexcept nogil>__cuMemFreeHost)(
        p)


cdef CUresult _cuMemHostAlloc(void** pp, size_t bytesize, unsigned int Flags) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuMemHostAlloc
    _check_or_init_driver()
    if __cuMemHostAlloc == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemHostAlloc is not found")
    return (<CUresult (*)(void**, size_t, unsigned int) noexcept nogil>__cuMemHostAlloc)(
        pp, bytesize, Flags)


cdef CUresult _cuMemHostGetDevicePointer_v2(CUdeviceptr* pdptr, void* p, unsigned int Flags) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuMemHostGetDevicePointer_v2
    _check_or_init_driver()
    if __cuMemHostGetDevicePointer_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemHostGetDevicePointer_v2 is not found")
    return (<CUresult (*)(CUdeviceptr*, void*, unsigned int) noexcept nogil>__cuMemHostGetDevicePointer_v2)(
        pdptr, p, Flags)


cdef CUresult _cuMemHostGetFlags(unsigned int* pFlags, void* p) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuMemHostGetFlags
    _check_or_init_driver()
    if __cuMemHostGetFlags == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemHostGetFlags is not found")
    return (<CUresult (*)(unsigned int*, void*) noexcept nogil>__cuMemHostGetFlags)(
        pFlags, p)


cdef CUresult _cuMemAllocManaged(CUdeviceptr* dptr, size_t bytesize, unsigned int flags) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuMemAllocManaged
    _check_or_init_driver()
    if __cuMemAllocManaged == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemAllocManaged is not found")
    return (<CUresult (*)(CUdeviceptr*, size_t, unsigned int) noexcept nogil>__cuMemAllocManaged)(
        dptr, bytesize, flags)


cdef CUresult _cuDeviceRegisterAsyncNotification(CUdevice device, CUasyncCallback callbackFunc, void* userData, CUasyncCallbackHandle* callback) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuDeviceRegisterAsyncNotification
    _check_or_init_driver()
    if __cuDeviceRegisterAsyncNotification == NULL:
        with gil:
            raise FunctionNotFoundError("function cuDeviceRegisterAsyncNotification is not found")
    return (<CUresult (*)(CUdevice, CUasyncCallback, void*, CUasyncCallbackHandle*) noexcept nogil>__cuDeviceRegisterAsyncNotification)(
        device, callbackFunc, userData, callback)


cdef CUresult _cuDeviceUnregisterAsyncNotification(CUdevice device, CUasyncCallbackHandle callback) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuDeviceUnregisterAsyncNotification
    _check_or_init_driver()
    if __cuDeviceUnregisterAsyncNotification == NULL:
        with gil:
            raise FunctionNotFoundError("function cuDeviceUnregisterAsyncNotification is not found")
    return (<CUresult (*)(CUdevice, CUasyncCallbackHandle) noexcept nogil>__cuDeviceUnregisterAsyncNotification)(
        device, callback)


cdef CUresult _cuDeviceGetByPCIBusId(CUdevice* dev, const char* pciBusId) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuDeviceGetByPCIBusId
    _check_or_init_driver()
    if __cuDeviceGetByPCIBusId == NULL:
        with gil:
            raise FunctionNotFoundError("function cuDeviceGetByPCIBusId is not found")
    return (<CUresult (*)(CUdevice*, const char*) noexcept nogil>__cuDeviceGetByPCIBusId)(
        dev, pciBusId)


cdef CUresult _cuDeviceGetPCIBusId(char* pciBusId, int len, CUdevice dev) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuDeviceGetPCIBusId
    _check_or_init_driver()
    if __cuDeviceGetPCIBusId == NULL:
        with gil:
            raise FunctionNotFoundError("function cuDeviceGetPCIBusId is not found")
    return (<CUresult (*)(char*, int, CUdevice) noexcept nogil>__cuDeviceGetPCIBusId)(
        pciBusId, len, dev)


cdef CUresult _cuIpcGetEventHandle(CUipcEventHandle* pHandle, CUevent event) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuIpcGetEventHandle
    _check_or_init_driver()
    if __cuIpcGetEventHandle == NULL:
        with gil:
            raise FunctionNotFoundError("function cuIpcGetEventHandle is not found")
    return (<CUresult (*)(CUipcEventHandle*, CUevent) noexcept nogil>__cuIpcGetEventHandle)(
        pHandle, event)


cdef CUresult _cuIpcOpenEventHandle(CUevent* phEvent, CUipcEventHandle handle) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuIpcOpenEventHandle
    _check_or_init_driver()
    if __cuIpcOpenEventHandle == NULL:
        with gil:
            raise FunctionNotFoundError("function cuIpcOpenEventHandle is not found")
    return (<CUresult (*)(CUevent*, CUipcEventHandle) noexcept nogil>__cuIpcOpenEventHandle)(
        phEvent, handle)


cdef CUresult _cuIpcGetMemHandle(CUipcMemHandle* pHandle, CUdeviceptr dptr) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuIpcGetMemHandle
    _check_or_init_driver()
    if __cuIpcGetMemHandle == NULL:
        with gil:
            raise FunctionNotFoundError("function cuIpcGetMemHandle is not found")
    return (<CUresult (*)(CUipcMemHandle*, CUdeviceptr) noexcept nogil>__cuIpcGetMemHandle)(
        pHandle, dptr)


cdef CUresult _cuIpcOpenMemHandle_v2(CUdeviceptr* pdptr, CUipcMemHandle handle, unsigned int Flags) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuIpcOpenMemHandle_v2
    _check_or_init_driver()
    if __cuIpcOpenMemHandle_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuIpcOpenMemHandle_v2 is not found")
    return (<CUresult (*)(CUdeviceptr*, CUipcMemHandle, unsigned int) noexcept nogil>__cuIpcOpenMemHandle_v2)(
        pdptr, handle, Flags)


cdef CUresult _cuIpcCloseMemHandle(CUdeviceptr dptr) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuIpcCloseMemHandle
    _check_or_init_driver()
    if __cuIpcCloseMemHandle == NULL:
        with gil:
            raise FunctionNotFoundError("function cuIpcCloseMemHandle is not found")
    return (<CUresult (*)(CUdeviceptr) noexcept nogil>__cuIpcCloseMemHandle)(
        dptr)


cdef CUresult _cuMemHostRegister_v2(void* p, size_t bytesize, unsigned int Flags) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuMemHostRegister_v2
    _check_or_init_driver()
    if __cuMemHostRegister_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemHostRegister_v2 is not found")
    return (<CUresult (*)(void*, size_t, unsigned int) noexcept nogil>__cuMemHostRegister_v2)(
        p, bytesize, Flags)


cdef CUresult _cuMemHostUnregister(void* p) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuMemHostUnregister
    _check_or_init_driver()
    if __cuMemHostUnregister == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemHostUnregister is not found")
    return (<CUresult (*)(void*) noexcept nogil>__cuMemHostUnregister)(
        p)


cdef CUresult _cuMemcpy(CUdeviceptr dst, CUdeviceptr src, size_t ByteCount) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuMemcpy
    _check_or_init_driver()
    if __cuMemcpy == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemcpy is not found")
    return (<CUresult (*)(CUdeviceptr, CUdeviceptr, size_t) noexcept nogil>__cuMemcpy)(
        dst, src, ByteCount)


cdef CUresult _cuMemcpyPeer(CUdeviceptr dstDevice, CUcontext dstContext, CUdeviceptr srcDevice, CUcontext srcContext, size_t ByteCount) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuMemcpyPeer
    _check_or_init_driver()
    if __cuMemcpyPeer == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemcpyPeer is not found")
    return (<CUresult (*)(CUdeviceptr, CUcontext, CUdeviceptr, CUcontext, size_t) noexcept nogil>__cuMemcpyPeer)(
        dstDevice, dstContext, srcDevice, srcContext, ByteCount)


cdef CUresult _cuMemcpyHtoD_v2(CUdeviceptr dstDevice, const void* srcHost, size_t ByteCount) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuMemcpyHtoD_v2
    _check_or_init_driver()
    if __cuMemcpyHtoD_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemcpyHtoD_v2 is not found")
    return (<CUresult (*)(CUdeviceptr, const void*, size_t) noexcept nogil>__cuMemcpyHtoD_v2)(
        dstDevice, srcHost, ByteCount)


cdef CUresult _cuMemcpyDtoH_v2(void* dstHost, CUdeviceptr srcDevice, size_t ByteCount) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuMemcpyDtoH_v2
    _check_or_init_driver()
    if __cuMemcpyDtoH_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemcpyDtoH_v2 is not found")
    return (<CUresult (*)(void*, CUdeviceptr, size_t) noexcept nogil>__cuMemcpyDtoH_v2)(
        dstHost, srcDevice, ByteCount)


cdef CUresult _cuMemcpyDtoD_v2(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuMemcpyDtoD_v2
    _check_or_init_driver()
    if __cuMemcpyDtoD_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemcpyDtoD_v2 is not found")
    return (<CUresult (*)(CUdeviceptr, CUdeviceptr, size_t) noexcept nogil>__cuMemcpyDtoD_v2)(
        dstDevice, srcDevice, ByteCount)


cdef CUresult _cuMemcpyDtoA_v2(CUarray dstArray, size_t dstOffset, CUdeviceptr srcDevice, size_t ByteCount) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuMemcpyDtoA_v2
    _check_or_init_driver()
    if __cuMemcpyDtoA_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemcpyDtoA_v2 is not found")
    return (<CUresult (*)(CUarray, size_t, CUdeviceptr, size_t) noexcept nogil>__cuMemcpyDtoA_v2)(
        dstArray, dstOffset, srcDevice, ByteCount)


cdef CUresult _cuMemcpyAtoD_v2(CUdeviceptr dstDevice, CUarray srcArray, size_t srcOffset, size_t ByteCount) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuMemcpyAtoD_v2
    _check_or_init_driver()
    if __cuMemcpyAtoD_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemcpyAtoD_v2 is not found")
    return (<CUresult (*)(CUdeviceptr, CUarray, size_t, size_t) noexcept nogil>__cuMemcpyAtoD_v2)(
        dstDevice, srcArray, srcOffset, ByteCount)


cdef CUresult _cuMemcpyHtoA_v2(CUarray dstArray, size_t dstOffset, const void* srcHost, size_t ByteCount) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuMemcpyHtoA_v2
    _check_or_init_driver()
    if __cuMemcpyHtoA_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemcpyHtoA_v2 is not found")
    return (<CUresult (*)(CUarray, size_t, const void*, size_t) noexcept nogil>__cuMemcpyHtoA_v2)(
        dstArray, dstOffset, srcHost, ByteCount)


cdef CUresult _cuMemcpyAtoH_v2(void* dstHost, CUarray srcArray, size_t srcOffset, size_t ByteCount) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuMemcpyAtoH_v2
    _check_or_init_driver()
    if __cuMemcpyAtoH_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemcpyAtoH_v2 is not found")
    return (<CUresult (*)(void*, CUarray, size_t, size_t) noexcept nogil>__cuMemcpyAtoH_v2)(
        dstHost, srcArray, srcOffset, ByteCount)


cdef CUresult _cuMemcpyAtoA_v2(CUarray dstArray, size_t dstOffset, CUarray srcArray, size_t srcOffset, size_t ByteCount) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuMemcpyAtoA_v2
    _check_or_init_driver()
    if __cuMemcpyAtoA_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemcpyAtoA_v2 is not found")
    return (<CUresult (*)(CUarray, size_t, CUarray, size_t, size_t) noexcept nogil>__cuMemcpyAtoA_v2)(
        dstArray, dstOffset, srcArray, srcOffset, ByteCount)


cdef CUresult _cuMemcpy2D_v2(const CUDA_MEMCPY2D* pCopy) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuMemcpy2D_v2
    _check_or_init_driver()
    if __cuMemcpy2D_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemcpy2D_v2 is not found")
    return (<CUresult (*)(const CUDA_MEMCPY2D*) noexcept nogil>__cuMemcpy2D_v2)(
        pCopy)


cdef CUresult _cuMemcpy2DUnaligned_v2(const CUDA_MEMCPY2D* pCopy) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuMemcpy2DUnaligned_v2
    _check_or_init_driver()
    if __cuMemcpy2DUnaligned_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemcpy2DUnaligned_v2 is not found")
    return (<CUresult (*)(const CUDA_MEMCPY2D*) noexcept nogil>__cuMemcpy2DUnaligned_v2)(
        pCopy)


cdef CUresult _cuMemcpy3D_v2(const CUDA_MEMCPY3D* pCopy) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuMemcpy3D_v2
    _check_or_init_driver()
    if __cuMemcpy3D_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemcpy3D_v2 is not found")
    return (<CUresult (*)(const CUDA_MEMCPY3D*) noexcept nogil>__cuMemcpy3D_v2)(
        pCopy)


cdef CUresult _cuMemcpy3DPeer(const CUDA_MEMCPY3D_PEER* pCopy) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuMemcpy3DPeer
    _check_or_init_driver()
    if __cuMemcpy3DPeer == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemcpy3DPeer is not found")
    return (<CUresult (*)(const CUDA_MEMCPY3D_PEER*) noexcept nogil>__cuMemcpy3DPeer)(
        pCopy)


cdef CUresult _cuMemcpyAsync(CUdeviceptr dst, CUdeviceptr src, size_t ByteCount, CUstream hStream) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuMemcpyAsync
    _check_or_init_driver()
    if __cuMemcpyAsync == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemcpyAsync is not found")
    return (<CUresult (*)(CUdeviceptr, CUdeviceptr, size_t, CUstream) noexcept nogil>__cuMemcpyAsync)(
        dst, src, ByteCount, hStream)


cdef CUresult _cuMemcpyPeerAsync(CUdeviceptr dstDevice, CUcontext dstContext, CUdeviceptr srcDevice, CUcontext srcContext, size_t ByteCount, CUstream hStream) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuMemcpyPeerAsync
    _check_or_init_driver()
    if __cuMemcpyPeerAsync == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemcpyPeerAsync is not found")
    return (<CUresult (*)(CUdeviceptr, CUcontext, CUdeviceptr, CUcontext, size_t, CUstream) noexcept nogil>__cuMemcpyPeerAsync)(
        dstDevice, dstContext, srcDevice, srcContext, ByteCount, hStream)


cdef CUresult _cuMemcpyHtoDAsync_v2(CUdeviceptr dstDevice, const void* srcHost, size_t ByteCount, CUstream hStream) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuMemcpyHtoDAsync_v2
    _check_or_init_driver()
    if __cuMemcpyHtoDAsync_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemcpyHtoDAsync_v2 is not found")
    return (<CUresult (*)(CUdeviceptr, const void*, size_t, CUstream) noexcept nogil>__cuMemcpyHtoDAsync_v2)(
        dstDevice, srcHost, ByteCount, hStream)


cdef CUresult _cuMemcpyDtoHAsync_v2(void* dstHost, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuMemcpyDtoHAsync_v2
    _check_or_init_driver()
    if __cuMemcpyDtoHAsync_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemcpyDtoHAsync_v2 is not found")
    return (<CUresult (*)(void*, CUdeviceptr, size_t, CUstream) noexcept nogil>__cuMemcpyDtoHAsync_v2)(
        dstHost, srcDevice, ByteCount, hStream)


cdef CUresult _cuMemcpyDtoDAsync_v2(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuMemcpyDtoDAsync_v2
    _check_or_init_driver()
    if __cuMemcpyDtoDAsync_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemcpyDtoDAsync_v2 is not found")
    return (<CUresult (*)(CUdeviceptr, CUdeviceptr, size_t, CUstream) noexcept nogil>__cuMemcpyDtoDAsync_v2)(
        dstDevice, srcDevice, ByteCount, hStream)


cdef CUresult _cuMemcpyHtoAAsync_v2(CUarray dstArray, size_t dstOffset, const void* srcHost, size_t ByteCount, CUstream hStream) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuMemcpyHtoAAsync_v2
    _check_or_init_driver()
    if __cuMemcpyHtoAAsync_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemcpyHtoAAsync_v2 is not found")
    return (<CUresult (*)(CUarray, size_t, const void*, size_t, CUstream) noexcept nogil>__cuMemcpyHtoAAsync_v2)(
        dstArray, dstOffset, srcHost, ByteCount, hStream)


cdef CUresult _cuMemcpyAtoHAsync_v2(void* dstHost, CUarray srcArray, size_t srcOffset, size_t ByteCount, CUstream hStream) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuMemcpyAtoHAsync_v2
    _check_or_init_driver()
    if __cuMemcpyAtoHAsync_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemcpyAtoHAsync_v2 is not found")
    return (<CUresult (*)(void*, CUarray, size_t, size_t, CUstream) noexcept nogil>__cuMemcpyAtoHAsync_v2)(
        dstHost, srcArray, srcOffset, ByteCount, hStream)


cdef CUresult _cuMemcpy2DAsync_v2(const CUDA_MEMCPY2D* pCopy, CUstream hStream) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuMemcpy2DAsync_v2
    _check_or_init_driver()
    if __cuMemcpy2DAsync_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemcpy2DAsync_v2 is not found")
    return (<CUresult (*)(const CUDA_MEMCPY2D*, CUstream) noexcept nogil>__cuMemcpy2DAsync_v2)(
        pCopy, hStream)


cdef CUresult _cuMemcpy3DAsync_v2(const CUDA_MEMCPY3D* pCopy, CUstream hStream) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuMemcpy3DAsync_v2
    _check_or_init_driver()
    if __cuMemcpy3DAsync_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemcpy3DAsync_v2 is not found")
    return (<CUresult (*)(const CUDA_MEMCPY3D*, CUstream) noexcept nogil>__cuMemcpy3DAsync_v2)(
        pCopy, hStream)


cdef CUresult _cuMemcpy3DPeerAsync(const CUDA_MEMCPY3D_PEER* pCopy, CUstream hStream) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuMemcpy3DPeerAsync
    _check_or_init_driver()
    if __cuMemcpy3DPeerAsync == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemcpy3DPeerAsync is not found")
    return (<CUresult (*)(const CUDA_MEMCPY3D_PEER*, CUstream) noexcept nogil>__cuMemcpy3DPeerAsync)(
        pCopy, hStream)


cdef CUresult _cuMemsetD8_v2(CUdeviceptr dstDevice, unsigned char uc, size_t N) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuMemsetD8_v2
    _check_or_init_driver()
    if __cuMemsetD8_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemsetD8_v2 is not found")
    return (<CUresult (*)(CUdeviceptr, unsigned char, size_t) noexcept nogil>__cuMemsetD8_v2)(
        dstDevice, uc, N)


cdef CUresult _cuMemsetD16_v2(CUdeviceptr dstDevice, unsigned short us, size_t N) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuMemsetD16_v2
    _check_or_init_driver()
    if __cuMemsetD16_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemsetD16_v2 is not found")
    return (<CUresult (*)(CUdeviceptr, unsigned short, size_t) noexcept nogil>__cuMemsetD16_v2)(
        dstDevice, us, N)


cdef CUresult _cuMemsetD32_v2(CUdeviceptr dstDevice, unsigned int ui, size_t N) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuMemsetD32_v2
    _check_or_init_driver()
    if __cuMemsetD32_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemsetD32_v2 is not found")
    return (<CUresult (*)(CUdeviceptr, unsigned int, size_t) noexcept nogil>__cuMemsetD32_v2)(
        dstDevice, ui, N)


cdef CUresult _cuMemsetD2D8_v2(CUdeviceptr dstDevice, size_t dstPitch, unsigned char uc, size_t Width, size_t Height) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuMemsetD2D8_v2
    _check_or_init_driver()
    if __cuMemsetD2D8_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemsetD2D8_v2 is not found")
    return (<CUresult (*)(CUdeviceptr, size_t, unsigned char, size_t, size_t) noexcept nogil>__cuMemsetD2D8_v2)(
        dstDevice, dstPitch, uc, Width, Height)


cdef CUresult _cuMemsetD2D16_v2(CUdeviceptr dstDevice, size_t dstPitch, unsigned short us, size_t Width, size_t Height) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuMemsetD2D16_v2
    _check_or_init_driver()
    if __cuMemsetD2D16_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemsetD2D16_v2 is not found")
    return (<CUresult (*)(CUdeviceptr, size_t, unsigned short, size_t, size_t) noexcept nogil>__cuMemsetD2D16_v2)(
        dstDevice, dstPitch, us, Width, Height)


cdef CUresult _cuMemsetD2D32_v2(CUdeviceptr dstDevice, size_t dstPitch, unsigned int ui, size_t Width, size_t Height) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuMemsetD2D32_v2
    _check_or_init_driver()
    if __cuMemsetD2D32_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemsetD2D32_v2 is not found")
    return (<CUresult (*)(CUdeviceptr, size_t, unsigned int, size_t, size_t) noexcept nogil>__cuMemsetD2D32_v2)(
        dstDevice, dstPitch, ui, Width, Height)


cdef CUresult _cuMemsetD8Async(CUdeviceptr dstDevice, unsigned char uc, size_t N, CUstream hStream) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuMemsetD8Async
    _check_or_init_driver()
    if __cuMemsetD8Async == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemsetD8Async is not found")
    return (<CUresult (*)(CUdeviceptr, unsigned char, size_t, CUstream) noexcept nogil>__cuMemsetD8Async)(
        dstDevice, uc, N, hStream)


cdef CUresult _cuMemsetD16Async(CUdeviceptr dstDevice, unsigned short us, size_t N, CUstream hStream) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuMemsetD16Async
    _check_or_init_driver()
    if __cuMemsetD16Async == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemsetD16Async is not found")
    return (<CUresult (*)(CUdeviceptr, unsigned short, size_t, CUstream) noexcept nogil>__cuMemsetD16Async)(
        dstDevice, us, N, hStream)


cdef CUresult _cuMemsetD32Async(CUdeviceptr dstDevice, unsigned int ui, size_t N, CUstream hStream) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuMemsetD32Async
    _check_or_init_driver()
    if __cuMemsetD32Async == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemsetD32Async is not found")
    return (<CUresult (*)(CUdeviceptr, unsigned int, size_t, CUstream) noexcept nogil>__cuMemsetD32Async)(
        dstDevice, ui, N, hStream)


cdef CUresult _cuMemsetD2D8Async(CUdeviceptr dstDevice, size_t dstPitch, unsigned char uc, size_t Width, size_t Height, CUstream hStream) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuMemsetD2D8Async
    _check_or_init_driver()
    if __cuMemsetD2D8Async == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemsetD2D8Async is not found")
    return (<CUresult (*)(CUdeviceptr, size_t, unsigned char, size_t, size_t, CUstream) noexcept nogil>__cuMemsetD2D8Async)(
        dstDevice, dstPitch, uc, Width, Height, hStream)


cdef CUresult _cuMemsetD2D16Async(CUdeviceptr dstDevice, size_t dstPitch, unsigned short us, size_t Width, size_t Height, CUstream hStream) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuMemsetD2D16Async
    _check_or_init_driver()
    if __cuMemsetD2D16Async == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemsetD2D16Async is not found")
    return (<CUresult (*)(CUdeviceptr, size_t, unsigned short, size_t, size_t, CUstream) noexcept nogil>__cuMemsetD2D16Async)(
        dstDevice, dstPitch, us, Width, Height, hStream)


cdef CUresult _cuMemsetD2D32Async(CUdeviceptr dstDevice, size_t dstPitch, unsigned int ui, size_t Width, size_t Height, CUstream hStream) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuMemsetD2D32Async
    _check_or_init_driver()
    if __cuMemsetD2D32Async == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemsetD2D32Async is not found")
    return (<CUresult (*)(CUdeviceptr, size_t, unsigned int, size_t, size_t, CUstream) noexcept nogil>__cuMemsetD2D32Async)(
        dstDevice, dstPitch, ui, Width, Height, hStream)


cdef CUresult _cuArrayCreate_v2(CUarray* pHandle, const CUDA_ARRAY_DESCRIPTOR* pAllocateArray) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuArrayCreate_v2
    _check_or_init_driver()
    if __cuArrayCreate_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuArrayCreate_v2 is not found")
    return (<CUresult (*)(CUarray*, const CUDA_ARRAY_DESCRIPTOR*) noexcept nogil>__cuArrayCreate_v2)(
        pHandle, pAllocateArray)


cdef CUresult _cuArrayGetDescriptor_v2(CUDA_ARRAY_DESCRIPTOR* pArrayDescriptor, CUarray hArray) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuArrayGetDescriptor_v2
    _check_or_init_driver()
    if __cuArrayGetDescriptor_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuArrayGetDescriptor_v2 is not found")
    return (<CUresult (*)(CUDA_ARRAY_DESCRIPTOR*, CUarray) noexcept nogil>__cuArrayGetDescriptor_v2)(
        pArrayDescriptor, hArray)


cdef CUresult _cuArrayGetSparseProperties(CUDA_ARRAY_SPARSE_PROPERTIES* sparseProperties, CUarray array) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuArrayGetSparseProperties
    _check_or_init_driver()
    if __cuArrayGetSparseProperties == NULL:
        with gil:
            raise FunctionNotFoundError("function cuArrayGetSparseProperties is not found")
    return (<CUresult (*)(CUDA_ARRAY_SPARSE_PROPERTIES*, CUarray) noexcept nogil>__cuArrayGetSparseProperties)(
        sparseProperties, array)


cdef CUresult _cuMipmappedArrayGetSparseProperties(CUDA_ARRAY_SPARSE_PROPERTIES* sparseProperties, CUmipmappedArray mipmap) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuMipmappedArrayGetSparseProperties
    _check_or_init_driver()
    if __cuMipmappedArrayGetSparseProperties == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMipmappedArrayGetSparseProperties is not found")
    return (<CUresult (*)(CUDA_ARRAY_SPARSE_PROPERTIES*, CUmipmappedArray) noexcept nogil>__cuMipmappedArrayGetSparseProperties)(
        sparseProperties, mipmap)


cdef CUresult _cuArrayGetMemoryRequirements(CUDA_ARRAY_MEMORY_REQUIREMENTS* memoryRequirements, CUarray array, CUdevice device) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuArrayGetMemoryRequirements
    _check_or_init_driver()
    if __cuArrayGetMemoryRequirements == NULL:
        with gil:
            raise FunctionNotFoundError("function cuArrayGetMemoryRequirements is not found")
    return (<CUresult (*)(CUDA_ARRAY_MEMORY_REQUIREMENTS*, CUarray, CUdevice) noexcept nogil>__cuArrayGetMemoryRequirements)(
        memoryRequirements, array, device)


cdef CUresult _cuMipmappedArrayGetMemoryRequirements(CUDA_ARRAY_MEMORY_REQUIREMENTS* memoryRequirements, CUmipmappedArray mipmap, CUdevice device) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuMipmappedArrayGetMemoryRequirements
    _check_or_init_driver()
    if __cuMipmappedArrayGetMemoryRequirements == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMipmappedArrayGetMemoryRequirements is not found")
    return (<CUresult (*)(CUDA_ARRAY_MEMORY_REQUIREMENTS*, CUmipmappedArray, CUdevice) noexcept nogil>__cuMipmappedArrayGetMemoryRequirements)(
        memoryRequirements, mipmap, device)


cdef CUresult _cuArrayGetPlane(CUarray* pPlaneArray, CUarray hArray, unsigned int planeIdx) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuArrayGetPlane
    _check_or_init_driver()
    if __cuArrayGetPlane == NULL:
        with gil:
            raise FunctionNotFoundError("function cuArrayGetPlane is not found")
    return (<CUresult (*)(CUarray*, CUarray, unsigned int) noexcept nogil>__cuArrayGetPlane)(
        pPlaneArray, hArray, planeIdx)


cdef CUresult _cuArrayDestroy(CUarray hArray) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuArrayDestroy
    _check_or_init_driver()
    if __cuArrayDestroy == NULL:
        with gil:
            raise FunctionNotFoundError("function cuArrayDestroy is not found")
    return (<CUresult (*)(CUarray) noexcept nogil>__cuArrayDestroy)(
        hArray)


cdef CUresult _cuArray3DCreate_v2(CUarray* pHandle, const CUDA_ARRAY3D_DESCRIPTOR* pAllocateArray) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuArray3DCreate_v2
    _check_or_init_driver()
    if __cuArray3DCreate_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuArray3DCreate_v2 is not found")
    return (<CUresult (*)(CUarray*, const CUDA_ARRAY3D_DESCRIPTOR*) noexcept nogil>__cuArray3DCreate_v2)(
        pHandle, pAllocateArray)


cdef CUresult _cuArray3DGetDescriptor_v2(CUDA_ARRAY3D_DESCRIPTOR* pArrayDescriptor, CUarray hArray) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuArray3DGetDescriptor_v2
    _check_or_init_driver()
    if __cuArray3DGetDescriptor_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuArray3DGetDescriptor_v2 is not found")
    return (<CUresult (*)(CUDA_ARRAY3D_DESCRIPTOR*, CUarray) noexcept nogil>__cuArray3DGetDescriptor_v2)(
        pArrayDescriptor, hArray)


cdef CUresult _cuMipmappedArrayCreate(CUmipmappedArray* pHandle, const CUDA_ARRAY3D_DESCRIPTOR* pMipmappedArrayDesc, unsigned int numMipmapLevels) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuMipmappedArrayCreate
    _check_or_init_driver()
    if __cuMipmappedArrayCreate == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMipmappedArrayCreate is not found")
    return (<CUresult (*)(CUmipmappedArray*, const CUDA_ARRAY3D_DESCRIPTOR*, unsigned int) noexcept nogil>__cuMipmappedArrayCreate)(
        pHandle, pMipmappedArrayDesc, numMipmapLevels)


cdef CUresult _cuMipmappedArrayGetLevel(CUarray* pLevelArray, CUmipmappedArray hMipmappedArray, unsigned int level) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuMipmappedArrayGetLevel
    _check_or_init_driver()
    if __cuMipmappedArrayGetLevel == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMipmappedArrayGetLevel is not found")
    return (<CUresult (*)(CUarray*, CUmipmappedArray, unsigned int) noexcept nogil>__cuMipmappedArrayGetLevel)(
        pLevelArray, hMipmappedArray, level)


cdef CUresult _cuMipmappedArrayDestroy(CUmipmappedArray hMipmappedArray) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuMipmappedArrayDestroy
    _check_or_init_driver()
    if __cuMipmappedArrayDestroy == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMipmappedArrayDestroy is not found")
    return (<CUresult (*)(CUmipmappedArray) noexcept nogil>__cuMipmappedArrayDestroy)(
        hMipmappedArray)


cdef CUresult _cuMemGetHandleForAddressRange(void* handle, CUdeviceptr dptr, size_t size, CUmemRangeHandleType handleType, unsigned long long flags) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuMemGetHandleForAddressRange
    _check_or_init_driver()
    if __cuMemGetHandleForAddressRange == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemGetHandleForAddressRange is not found")
    return (<CUresult (*)(void*, CUdeviceptr, size_t, CUmemRangeHandleType, unsigned long long) noexcept nogil>__cuMemGetHandleForAddressRange)(
        handle, dptr, size, handleType, flags)


cdef CUresult _cuMemBatchDecompressAsync(CUmemDecompressParams* paramsArray, size_t count, unsigned int flags, size_t* errorIndex, CUstream stream) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuMemBatchDecompressAsync
    _check_or_init_driver()
    if __cuMemBatchDecompressAsync == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemBatchDecompressAsync is not found")
    return (<CUresult (*)(CUmemDecompressParams*, size_t, unsigned int, size_t*, CUstream) noexcept nogil>__cuMemBatchDecompressAsync)(
        paramsArray, count, flags, errorIndex, stream)


cdef CUresult _cuMemAddressReserve(CUdeviceptr* ptr, size_t size, size_t alignment, CUdeviceptr addr, unsigned long long flags) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuMemAddressReserve
    _check_or_init_driver()
    if __cuMemAddressReserve == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemAddressReserve is not found")
    return (<CUresult (*)(CUdeviceptr*, size_t, size_t, CUdeviceptr, unsigned long long) noexcept nogil>__cuMemAddressReserve)(
        ptr, size, alignment, addr, flags)


cdef CUresult _cuMemAddressFree(CUdeviceptr ptr, size_t size) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuMemAddressFree
    _check_or_init_driver()
    if __cuMemAddressFree == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemAddressFree is not found")
    return (<CUresult (*)(CUdeviceptr, size_t) noexcept nogil>__cuMemAddressFree)(
        ptr, size)


cdef CUresult _cuMemCreate(CUmemGenericAllocationHandle* handle, size_t size, const CUmemAllocationProp* prop, unsigned long long flags) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuMemCreate
    _check_or_init_driver()
    if __cuMemCreate == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemCreate is not found")
    return (<CUresult (*)(CUmemGenericAllocationHandle*, size_t, const CUmemAllocationProp*, unsigned long long) noexcept nogil>__cuMemCreate)(
        handle, size, prop, flags)


cdef CUresult _cuMemRelease(CUmemGenericAllocationHandle handle) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuMemRelease
    _check_or_init_driver()
    if __cuMemRelease == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemRelease is not found")
    return (<CUresult (*)(CUmemGenericAllocationHandle) noexcept nogil>__cuMemRelease)(
        handle)


cdef CUresult _cuMemMap(CUdeviceptr ptr, size_t size, size_t offset, CUmemGenericAllocationHandle handle, unsigned long long flags) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuMemMap
    _check_or_init_driver()
    if __cuMemMap == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemMap is not found")
    return (<CUresult (*)(CUdeviceptr, size_t, size_t, CUmemGenericAllocationHandle, unsigned long long) noexcept nogil>__cuMemMap)(
        ptr, size, offset, handle, flags)


cdef CUresult _cuMemMapArrayAsync(CUarrayMapInfo* mapInfoList, unsigned int count, CUstream hStream) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuMemMapArrayAsync
    _check_or_init_driver()
    if __cuMemMapArrayAsync == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemMapArrayAsync is not found")
    return (<CUresult (*)(CUarrayMapInfo*, unsigned int, CUstream) noexcept nogil>__cuMemMapArrayAsync)(
        mapInfoList, count, hStream)


cdef CUresult _cuMemUnmap(CUdeviceptr ptr, size_t size) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuMemUnmap
    _check_or_init_driver()
    if __cuMemUnmap == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemUnmap is not found")
    return (<CUresult (*)(CUdeviceptr, size_t) noexcept nogil>__cuMemUnmap)(
        ptr, size)


cdef CUresult _cuMemSetAccess(CUdeviceptr ptr, size_t size, const CUmemAccessDesc* desc, size_t count) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuMemSetAccess
    _check_or_init_driver()
    if __cuMemSetAccess == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemSetAccess is not found")
    return (<CUresult (*)(CUdeviceptr, size_t, const CUmemAccessDesc*, size_t) noexcept nogil>__cuMemSetAccess)(
        ptr, size, desc, count)


cdef CUresult _cuMemGetAccess(unsigned long long* flags, const CUmemLocation* location, CUdeviceptr ptr) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuMemGetAccess
    _check_or_init_driver()
    if __cuMemGetAccess == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemGetAccess is not found")
    return (<CUresult (*)(unsigned long long*, const CUmemLocation*, CUdeviceptr) noexcept nogil>__cuMemGetAccess)(
        flags, location, ptr)


cdef CUresult _cuMemExportToShareableHandle(void* shareableHandle, CUmemGenericAllocationHandle handle, CUmemAllocationHandleType handleType, unsigned long long flags) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuMemExportToShareableHandle
    _check_or_init_driver()
    if __cuMemExportToShareableHandle == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemExportToShareableHandle is not found")
    return (<CUresult (*)(void*, CUmemGenericAllocationHandle, CUmemAllocationHandleType, unsigned long long) noexcept nogil>__cuMemExportToShareableHandle)(
        shareableHandle, handle, handleType, flags)


cdef CUresult _cuMemImportFromShareableHandle(CUmemGenericAllocationHandle* handle, void* osHandle, CUmemAllocationHandleType shHandleType) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuMemImportFromShareableHandle
    _check_or_init_driver()
    if __cuMemImportFromShareableHandle == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemImportFromShareableHandle is not found")
    return (<CUresult (*)(CUmemGenericAllocationHandle*, void*, CUmemAllocationHandleType) noexcept nogil>__cuMemImportFromShareableHandle)(
        handle, osHandle, shHandleType)


cdef CUresult _cuMemGetAllocationGranularity(size_t* granularity, const CUmemAllocationProp* prop, CUmemAllocationGranularity_flags option) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuMemGetAllocationGranularity
    _check_or_init_driver()
    if __cuMemGetAllocationGranularity == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemGetAllocationGranularity is not found")
    return (<CUresult (*)(size_t*, const CUmemAllocationProp*, CUmemAllocationGranularity_flags) noexcept nogil>__cuMemGetAllocationGranularity)(
        granularity, prop, option)


cdef CUresult _cuMemGetAllocationPropertiesFromHandle(CUmemAllocationProp* prop, CUmemGenericAllocationHandle handle) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuMemGetAllocationPropertiesFromHandle
    _check_or_init_driver()
    if __cuMemGetAllocationPropertiesFromHandle == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemGetAllocationPropertiesFromHandle is not found")
    return (<CUresult (*)(CUmemAllocationProp*, CUmemGenericAllocationHandle) noexcept nogil>__cuMemGetAllocationPropertiesFromHandle)(
        prop, handle)


cdef CUresult _cuMemRetainAllocationHandle(CUmemGenericAllocationHandle* handle, void* addr) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuMemRetainAllocationHandle
    _check_or_init_driver()
    if __cuMemRetainAllocationHandle == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemRetainAllocationHandle is not found")
    return (<CUresult (*)(CUmemGenericAllocationHandle*, void*) noexcept nogil>__cuMemRetainAllocationHandle)(
        handle, addr)


cdef CUresult _cuMemFreeAsync(CUdeviceptr dptr, CUstream hStream) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuMemFreeAsync
    _check_or_init_driver()
    if __cuMemFreeAsync == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemFreeAsync is not found")
    return (<CUresult (*)(CUdeviceptr, CUstream) noexcept nogil>__cuMemFreeAsync)(
        dptr, hStream)


cdef CUresult _cuMemAllocAsync(CUdeviceptr* dptr, size_t bytesize, CUstream hStream) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuMemAllocAsync
    _check_or_init_driver()
    if __cuMemAllocAsync == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemAllocAsync is not found")
    return (<CUresult (*)(CUdeviceptr*, size_t, CUstream) noexcept nogil>__cuMemAllocAsync)(
        dptr, bytesize, hStream)


cdef CUresult _cuMemPoolTrimTo(CUmemoryPool pool, size_t minBytesToKeep) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuMemPoolTrimTo
    _check_or_init_driver()
    if __cuMemPoolTrimTo == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemPoolTrimTo is not found")
    return (<CUresult (*)(CUmemoryPool, size_t) noexcept nogil>__cuMemPoolTrimTo)(
        pool, minBytesToKeep)


cdef CUresult _cuMemPoolSetAttribute(CUmemoryPool pool, CUmemPool_attribute attr, void* value) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuMemPoolSetAttribute
    _check_or_init_driver()
    if __cuMemPoolSetAttribute == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemPoolSetAttribute is not found")
    return (<CUresult (*)(CUmemoryPool, CUmemPool_attribute, void*) noexcept nogil>__cuMemPoolSetAttribute)(
        pool, attr, value)


cdef CUresult _cuMemPoolGetAttribute(CUmemoryPool pool, CUmemPool_attribute attr, void* value) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuMemPoolGetAttribute
    _check_or_init_driver()
    if __cuMemPoolGetAttribute == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemPoolGetAttribute is not found")
    return (<CUresult (*)(CUmemoryPool, CUmemPool_attribute, void*) noexcept nogil>__cuMemPoolGetAttribute)(
        pool, attr, value)


cdef CUresult _cuMemPoolSetAccess(CUmemoryPool pool, const CUmemAccessDesc* map, size_t count) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuMemPoolSetAccess
    _check_or_init_driver()
    if __cuMemPoolSetAccess == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemPoolSetAccess is not found")
    return (<CUresult (*)(CUmemoryPool, const CUmemAccessDesc*, size_t) noexcept nogil>__cuMemPoolSetAccess)(
        pool, map, count)


cdef CUresult _cuMemPoolGetAccess(CUmemAccess_flags* flags, CUmemoryPool memPool, CUmemLocation* location) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuMemPoolGetAccess
    _check_or_init_driver()
    if __cuMemPoolGetAccess == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemPoolGetAccess is not found")
    return (<CUresult (*)(CUmemAccess_flags*, CUmemoryPool, CUmemLocation*) noexcept nogil>__cuMemPoolGetAccess)(
        flags, memPool, location)


cdef CUresult _cuMemPoolCreate(CUmemoryPool* pool, const CUmemPoolProps* poolProps) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuMemPoolCreate
    _check_or_init_driver()
    if __cuMemPoolCreate == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemPoolCreate is not found")
    return (<CUresult (*)(CUmemoryPool*, const CUmemPoolProps*) noexcept nogil>__cuMemPoolCreate)(
        pool, poolProps)


cdef CUresult _cuMemPoolDestroy(CUmemoryPool pool) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuMemPoolDestroy
    _check_or_init_driver()
    if __cuMemPoolDestroy == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemPoolDestroy is not found")
    return (<CUresult (*)(CUmemoryPool) noexcept nogil>__cuMemPoolDestroy)(
        pool)


cdef CUresult _cuMemAllocFromPoolAsync(CUdeviceptr* dptr, size_t bytesize, CUmemoryPool pool, CUstream hStream) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuMemAllocFromPoolAsync
    _check_or_init_driver()
    if __cuMemAllocFromPoolAsync == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemAllocFromPoolAsync is not found")
    return (<CUresult (*)(CUdeviceptr*, size_t, CUmemoryPool, CUstream) noexcept nogil>__cuMemAllocFromPoolAsync)(
        dptr, bytesize, pool, hStream)


cdef CUresult _cuMemPoolExportToShareableHandle(void* handle_out, CUmemoryPool pool, CUmemAllocationHandleType handleType, unsigned long long flags) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuMemPoolExportToShareableHandle
    _check_or_init_driver()
    if __cuMemPoolExportToShareableHandle == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemPoolExportToShareableHandle is not found")
    return (<CUresult (*)(void*, CUmemoryPool, CUmemAllocationHandleType, unsigned long long) noexcept nogil>__cuMemPoolExportToShareableHandle)(
        handle_out, pool, handleType, flags)


cdef CUresult _cuMemPoolImportFromShareableHandle(CUmemoryPool* pool_out, void* handle, CUmemAllocationHandleType handleType, unsigned long long flags) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuMemPoolImportFromShareableHandle
    _check_or_init_driver()
    if __cuMemPoolImportFromShareableHandle == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemPoolImportFromShareableHandle is not found")
    return (<CUresult (*)(CUmemoryPool*, void*, CUmemAllocationHandleType, unsigned long long) noexcept nogil>__cuMemPoolImportFromShareableHandle)(
        pool_out, handle, handleType, flags)


cdef CUresult _cuMemPoolExportPointer(CUmemPoolPtrExportData* shareData_out, CUdeviceptr ptr) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuMemPoolExportPointer
    _check_or_init_driver()
    if __cuMemPoolExportPointer == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemPoolExportPointer is not found")
    return (<CUresult (*)(CUmemPoolPtrExportData*, CUdeviceptr) noexcept nogil>__cuMemPoolExportPointer)(
        shareData_out, ptr)


cdef CUresult _cuMemPoolImportPointer(CUdeviceptr* ptr_out, CUmemoryPool pool, CUmemPoolPtrExportData* shareData) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuMemPoolImportPointer
    _check_or_init_driver()
    if __cuMemPoolImportPointer == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemPoolImportPointer is not found")
    return (<CUresult (*)(CUdeviceptr*, CUmemoryPool, CUmemPoolPtrExportData*) noexcept nogil>__cuMemPoolImportPointer)(
        ptr_out, pool, shareData)


cdef CUresult _cuMulticastCreate(CUmemGenericAllocationHandle* mcHandle, const CUmulticastObjectProp* prop) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuMulticastCreate
    _check_or_init_driver()
    if __cuMulticastCreate == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMulticastCreate is not found")
    return (<CUresult (*)(CUmemGenericAllocationHandle*, const CUmulticastObjectProp*) noexcept nogil>__cuMulticastCreate)(
        mcHandle, prop)


cdef CUresult _cuMulticastAddDevice(CUmemGenericAllocationHandle mcHandle, CUdevice dev) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuMulticastAddDevice
    _check_or_init_driver()
    if __cuMulticastAddDevice == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMulticastAddDevice is not found")
    return (<CUresult (*)(CUmemGenericAllocationHandle, CUdevice) noexcept nogil>__cuMulticastAddDevice)(
        mcHandle, dev)


cdef CUresult _cuMulticastBindMem(CUmemGenericAllocationHandle mcHandle, size_t mcOffset, CUmemGenericAllocationHandle memHandle, size_t memOffset, size_t size, unsigned long long flags) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuMulticastBindMem
    _check_or_init_driver()
    if __cuMulticastBindMem == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMulticastBindMem is not found")
    return (<CUresult (*)(CUmemGenericAllocationHandle, size_t, CUmemGenericAllocationHandle, size_t, size_t, unsigned long long) noexcept nogil>__cuMulticastBindMem)(
        mcHandle, mcOffset, memHandle, memOffset, size, flags)


cdef CUresult _cuMulticastBindAddr(CUmemGenericAllocationHandle mcHandle, size_t mcOffset, CUdeviceptr memptr, size_t size, unsigned long long flags) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuMulticastBindAddr
    _check_or_init_driver()
    if __cuMulticastBindAddr == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMulticastBindAddr is not found")
    return (<CUresult (*)(CUmemGenericAllocationHandle, size_t, CUdeviceptr, size_t, unsigned long long) noexcept nogil>__cuMulticastBindAddr)(
        mcHandle, mcOffset, memptr, size, flags)


cdef CUresult _cuMulticastUnbind(CUmemGenericAllocationHandle mcHandle, CUdevice dev, size_t mcOffset, size_t size) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuMulticastUnbind
    _check_or_init_driver()
    if __cuMulticastUnbind == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMulticastUnbind is not found")
    return (<CUresult (*)(CUmemGenericAllocationHandle, CUdevice, size_t, size_t) noexcept nogil>__cuMulticastUnbind)(
        mcHandle, dev, mcOffset, size)


cdef CUresult _cuMulticastGetGranularity(size_t* granularity, const CUmulticastObjectProp* prop, CUmulticastGranularity_flags option) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuMulticastGetGranularity
    _check_or_init_driver()
    if __cuMulticastGetGranularity == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMulticastGetGranularity is not found")
    return (<CUresult (*)(size_t*, const CUmulticastObjectProp*, CUmulticastGranularity_flags) noexcept nogil>__cuMulticastGetGranularity)(
        granularity, prop, option)


cdef CUresult _cuPointerGetAttribute(void* data, CUpointer_attribute attribute, CUdeviceptr ptr) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuPointerGetAttribute
    _check_or_init_driver()
    if __cuPointerGetAttribute == NULL:
        with gil:
            raise FunctionNotFoundError("function cuPointerGetAttribute is not found")
    return (<CUresult (*)(void*, CUpointer_attribute, CUdeviceptr) noexcept nogil>__cuPointerGetAttribute)(
        data, attribute, ptr)


cdef CUresult _cuMemPrefetchAsync_v2(CUdeviceptr devPtr, size_t count, CUmemLocation location, unsigned int flags, CUstream hStream) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuMemPrefetchAsync_v2
    _check_or_init_driver()
    if __cuMemPrefetchAsync_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemPrefetchAsync_v2 is not found")
    return (<CUresult (*)(CUdeviceptr, size_t, CUmemLocation, unsigned int, CUstream) noexcept nogil>__cuMemPrefetchAsync_v2)(
        devPtr, count, location, flags, hStream)


cdef CUresult _cuMemAdvise_v2(CUdeviceptr devPtr, size_t count, CUmem_advise advice, CUmemLocation location) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuMemAdvise_v2
    _check_or_init_driver()
    if __cuMemAdvise_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemAdvise_v2 is not found")
    return (<CUresult (*)(CUdeviceptr, size_t, CUmem_advise, CUmemLocation) noexcept nogil>__cuMemAdvise_v2)(
        devPtr, count, advice, location)


cdef CUresult _cuMemRangeGetAttribute(void* data, size_t dataSize, CUmem_range_attribute attribute, CUdeviceptr devPtr, size_t count) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuMemRangeGetAttribute
    _check_or_init_driver()
    if __cuMemRangeGetAttribute == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemRangeGetAttribute is not found")
    return (<CUresult (*)(void*, size_t, CUmem_range_attribute, CUdeviceptr, size_t) noexcept nogil>__cuMemRangeGetAttribute)(
        data, dataSize, attribute, devPtr, count)


cdef CUresult _cuMemRangeGetAttributes(void** data, size_t* dataSizes, CUmem_range_attribute* attributes, size_t numAttributes, CUdeviceptr devPtr, size_t count) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuMemRangeGetAttributes
    _check_or_init_driver()
    if __cuMemRangeGetAttributes == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemRangeGetAttributes is not found")
    return (<CUresult (*)(void**, size_t*, CUmem_range_attribute*, size_t, CUdeviceptr, size_t) noexcept nogil>__cuMemRangeGetAttributes)(
        data, dataSizes, attributes, numAttributes, devPtr, count)


cdef CUresult _cuPointerSetAttribute(const void* value, CUpointer_attribute attribute, CUdeviceptr ptr) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuPointerSetAttribute
    _check_or_init_driver()
    if __cuPointerSetAttribute == NULL:
        with gil:
            raise FunctionNotFoundError("function cuPointerSetAttribute is not found")
    return (<CUresult (*)(const void*, CUpointer_attribute, CUdeviceptr) noexcept nogil>__cuPointerSetAttribute)(
        value, attribute, ptr)


cdef CUresult _cuPointerGetAttributes(unsigned int numAttributes, CUpointer_attribute* attributes, void** data, CUdeviceptr ptr) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuPointerGetAttributes
    _check_or_init_driver()
    if __cuPointerGetAttributes == NULL:
        with gil:
            raise FunctionNotFoundError("function cuPointerGetAttributes is not found")
    return (<CUresult (*)(unsigned int, CUpointer_attribute*, void**, CUdeviceptr) noexcept nogil>__cuPointerGetAttributes)(
        numAttributes, attributes, data, ptr)


cdef CUresult _cuStreamCreate(CUstream* phStream, unsigned int Flags) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuStreamCreate
    _check_or_init_driver()
    if __cuStreamCreate == NULL:
        with gil:
            raise FunctionNotFoundError("function cuStreamCreate is not found")
    return (<CUresult (*)(CUstream*, unsigned int) noexcept nogil>__cuStreamCreate)(
        phStream, Flags)


cdef CUresult _cuStreamCreateWithPriority(CUstream* phStream, unsigned int flags, int priority) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuStreamCreateWithPriority
    _check_or_init_driver()
    if __cuStreamCreateWithPriority == NULL:
        with gil:
            raise FunctionNotFoundError("function cuStreamCreateWithPriority is not found")
    return (<CUresult (*)(CUstream*, unsigned int, int) noexcept nogil>__cuStreamCreateWithPriority)(
        phStream, flags, priority)


cdef CUresult _cuStreamGetPriority(CUstream hStream, int* priority) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuStreamGetPriority
    _check_or_init_driver()
    if __cuStreamGetPriority == NULL:
        with gil:
            raise FunctionNotFoundError("function cuStreamGetPriority is not found")
    return (<CUresult (*)(CUstream, int*) noexcept nogil>__cuStreamGetPriority)(
        hStream, priority)


cdef CUresult _cuStreamGetDevice(CUstream hStream, CUdevice* device) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuStreamGetDevice
    _check_or_init_driver()
    if __cuStreamGetDevice == NULL:
        with gil:
            raise FunctionNotFoundError("function cuStreamGetDevice is not found")
    return (<CUresult (*)(CUstream, CUdevice*) noexcept nogil>__cuStreamGetDevice)(
        hStream, device)


cdef CUresult _cuStreamGetFlags(CUstream hStream, unsigned int* flags) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuStreamGetFlags
    _check_or_init_driver()
    if __cuStreamGetFlags == NULL:
        with gil:
            raise FunctionNotFoundError("function cuStreamGetFlags is not found")
    return (<CUresult (*)(CUstream, unsigned int*) noexcept nogil>__cuStreamGetFlags)(
        hStream, flags)


cdef CUresult _cuStreamGetId(CUstream hStream, unsigned long long* streamId) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuStreamGetId
    _check_or_init_driver()
    if __cuStreamGetId == NULL:
        with gil:
            raise FunctionNotFoundError("function cuStreamGetId is not found")
    return (<CUresult (*)(CUstream, unsigned long long*) noexcept nogil>__cuStreamGetId)(
        hStream, streamId)


cdef CUresult _cuStreamGetCtx(CUstream hStream, CUcontext* pctx) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuStreamGetCtx
    _check_or_init_driver()
    if __cuStreamGetCtx == NULL:
        with gil:
            raise FunctionNotFoundError("function cuStreamGetCtx is not found")
    return (<CUresult (*)(CUstream, CUcontext*) noexcept nogil>__cuStreamGetCtx)(
        hStream, pctx)


cdef CUresult _cuStreamGetCtx_v2(CUstream hStream, CUcontext* pCtx, CUgreenCtx* pGreenCtx) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuStreamGetCtx_v2
    _check_or_init_driver()
    if __cuStreamGetCtx_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuStreamGetCtx_v2 is not found")
    return (<CUresult (*)(CUstream, CUcontext*, CUgreenCtx*) noexcept nogil>__cuStreamGetCtx_v2)(
        hStream, pCtx, pGreenCtx)


cdef CUresult _cuStreamWaitEvent(CUstream hStream, CUevent hEvent, unsigned int Flags) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuStreamWaitEvent
    _check_or_init_driver()
    if __cuStreamWaitEvent == NULL:
        with gil:
            raise FunctionNotFoundError("function cuStreamWaitEvent is not found")
    return (<CUresult (*)(CUstream, CUevent, unsigned int) noexcept nogil>__cuStreamWaitEvent)(
        hStream, hEvent, Flags)


cdef CUresult _cuStreamAddCallback(CUstream hStream, CUstreamCallback callback, void* userData, unsigned int flags) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuStreamAddCallback
    _check_or_init_driver()
    if __cuStreamAddCallback == NULL:
        with gil:
            raise FunctionNotFoundError("function cuStreamAddCallback is not found")
    return (<CUresult (*)(CUstream, CUstreamCallback, void*, unsigned int) noexcept nogil>__cuStreamAddCallback)(
        hStream, callback, userData, flags)


cdef CUresult _cuStreamBeginCapture_v2(CUstream hStream, CUstreamCaptureMode mode) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuStreamBeginCapture_v2
    _check_or_init_driver()
    if __cuStreamBeginCapture_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuStreamBeginCapture_v2 is not found")
    return (<CUresult (*)(CUstream, CUstreamCaptureMode) noexcept nogil>__cuStreamBeginCapture_v2)(
        hStream, mode)


cdef CUresult _cuStreamBeginCaptureToGraph(CUstream hStream, CUgraph hGraph, const CUgraphNode* dependencies, const CUgraphEdgeData* dependencyData, size_t numDependencies, CUstreamCaptureMode mode) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuStreamBeginCaptureToGraph
    _check_or_init_driver()
    if __cuStreamBeginCaptureToGraph == NULL:
        with gil:
            raise FunctionNotFoundError("function cuStreamBeginCaptureToGraph is not found")
    return (<CUresult (*)(CUstream, CUgraph, const CUgraphNode*, const CUgraphEdgeData*, size_t, CUstreamCaptureMode) noexcept nogil>__cuStreamBeginCaptureToGraph)(
        hStream, hGraph, dependencies, dependencyData, numDependencies, mode)


cdef CUresult _cuThreadExchangeStreamCaptureMode(CUstreamCaptureMode* mode) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuThreadExchangeStreamCaptureMode
    _check_or_init_driver()
    if __cuThreadExchangeStreamCaptureMode == NULL:
        with gil:
            raise FunctionNotFoundError("function cuThreadExchangeStreamCaptureMode is not found")
    return (<CUresult (*)(CUstreamCaptureMode*) noexcept nogil>__cuThreadExchangeStreamCaptureMode)(
        mode)


cdef CUresult _cuStreamEndCapture(CUstream hStream, CUgraph* phGraph) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuStreamEndCapture
    _check_or_init_driver()
    if __cuStreamEndCapture == NULL:
        with gil:
            raise FunctionNotFoundError("function cuStreamEndCapture is not found")
    return (<CUresult (*)(CUstream, CUgraph*) noexcept nogil>__cuStreamEndCapture)(
        hStream, phGraph)


cdef CUresult _cuStreamIsCapturing(CUstream hStream, CUstreamCaptureStatus* captureStatus) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuStreamIsCapturing
    _check_or_init_driver()
    if __cuStreamIsCapturing == NULL:
        with gil:
            raise FunctionNotFoundError("function cuStreamIsCapturing is not found")
    return (<CUresult (*)(CUstream, CUstreamCaptureStatus*) noexcept nogil>__cuStreamIsCapturing)(
        hStream, captureStatus)


cdef CUresult _cuStreamGetCaptureInfo_v2(CUstream hStream, CUstreamCaptureStatus* captureStatus_out, cuuint64_t* id_out, CUgraph* graph_out, const CUgraphNode** dependencies_out, size_t* numDependencies_out) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuStreamGetCaptureInfo_v2
    _check_or_init_driver()
    if __cuStreamGetCaptureInfo_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuStreamGetCaptureInfo_v2 is not found")
    return (<CUresult (*)(CUstream, CUstreamCaptureStatus*, cuuint64_t*, CUgraph*, const CUgraphNode**, size_t*) noexcept nogil>__cuStreamGetCaptureInfo_v2)(
        hStream, captureStatus_out, id_out, graph_out, dependencies_out, numDependencies_out)


cdef CUresult _cuStreamGetCaptureInfo_v3(CUstream hStream, CUstreamCaptureStatus* captureStatus_out, cuuint64_t* id_out, CUgraph* graph_out, const CUgraphNode** dependencies_out, const CUgraphEdgeData** edgeData_out, size_t* numDependencies_out) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuStreamGetCaptureInfo_v3
    _check_or_init_driver()
    if __cuStreamGetCaptureInfo_v3 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuStreamGetCaptureInfo_v3 is not found")
    return (<CUresult (*)(CUstream, CUstreamCaptureStatus*, cuuint64_t*, CUgraph*, const CUgraphNode**, const CUgraphEdgeData**, size_t*) noexcept nogil>__cuStreamGetCaptureInfo_v3)(
        hStream, captureStatus_out, id_out, graph_out, dependencies_out, edgeData_out, numDependencies_out)


cdef CUresult _cuStreamUpdateCaptureDependencies_v2(CUstream hStream, CUgraphNode* dependencies, const CUgraphEdgeData* dependencyData, size_t numDependencies, unsigned int flags) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuStreamUpdateCaptureDependencies_v2
    _check_or_init_driver()
    if __cuStreamUpdateCaptureDependencies_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuStreamUpdateCaptureDependencies_v2 is not found")
    return (<CUresult (*)(CUstream, CUgraphNode*, const CUgraphEdgeData*, size_t, unsigned int) noexcept nogil>__cuStreamUpdateCaptureDependencies_v2)(
        hStream, dependencies, dependencyData, numDependencies, flags)


cdef CUresult _cuStreamAttachMemAsync(CUstream hStream, CUdeviceptr dptr, size_t length, unsigned int flags) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuStreamAttachMemAsync
    _check_or_init_driver()
    if __cuStreamAttachMemAsync == NULL:
        with gil:
            raise FunctionNotFoundError("function cuStreamAttachMemAsync is not found")
    return (<CUresult (*)(CUstream, CUdeviceptr, size_t, unsigned int) noexcept nogil>__cuStreamAttachMemAsync)(
        hStream, dptr, length, flags)


cdef CUresult _cuStreamQuery(CUstream hStream) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuStreamQuery
    _check_or_init_driver()
    if __cuStreamQuery == NULL:
        with gil:
            raise FunctionNotFoundError("function cuStreamQuery is not found")
    return (<CUresult (*)(CUstream) noexcept nogil>__cuStreamQuery)(
        hStream)


cdef CUresult _cuStreamSynchronize(CUstream hStream) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuStreamSynchronize
    _check_or_init_driver()
    if __cuStreamSynchronize == NULL:
        with gil:
            raise FunctionNotFoundError("function cuStreamSynchronize is not found")
    return (<CUresult (*)(CUstream) noexcept nogil>__cuStreamSynchronize)(
        hStream)


cdef CUresult _cuStreamDestroy_v2(CUstream hStream) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuStreamDestroy_v2
    _check_or_init_driver()
    if __cuStreamDestroy_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuStreamDestroy_v2 is not found")
    return (<CUresult (*)(CUstream) noexcept nogil>__cuStreamDestroy_v2)(
        hStream)


cdef CUresult _cuStreamCopyAttributes(CUstream dst, CUstream src) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuStreamCopyAttributes
    _check_or_init_driver()
    if __cuStreamCopyAttributes == NULL:
        with gil:
            raise FunctionNotFoundError("function cuStreamCopyAttributes is not found")
    return (<CUresult (*)(CUstream, CUstream) noexcept nogil>__cuStreamCopyAttributes)(
        dst, src)


cdef CUresult _cuStreamGetAttribute(CUstream hStream, CUstreamAttrID attr, CUstreamAttrValue* value_out) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuStreamGetAttribute
    _check_or_init_driver()
    if __cuStreamGetAttribute == NULL:
        with gil:
            raise FunctionNotFoundError("function cuStreamGetAttribute is not found")
    return (<CUresult (*)(CUstream, CUstreamAttrID, CUstreamAttrValue*) noexcept nogil>__cuStreamGetAttribute)(
        hStream, attr, value_out)


cdef CUresult _cuStreamSetAttribute(CUstream hStream, CUstreamAttrID attr, const CUstreamAttrValue* value) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuStreamSetAttribute
    _check_or_init_driver()
    if __cuStreamSetAttribute == NULL:
        with gil:
            raise FunctionNotFoundError("function cuStreamSetAttribute is not found")
    return (<CUresult (*)(CUstream, CUstreamAttrID, const CUstreamAttrValue*) noexcept nogil>__cuStreamSetAttribute)(
        hStream, attr, value)


cdef CUresult _cuEventCreate(CUevent* phEvent, unsigned int Flags) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuEventCreate
    _check_or_init_driver()
    if __cuEventCreate == NULL:
        with gil:
            raise FunctionNotFoundError("function cuEventCreate is not found")
    return (<CUresult (*)(CUevent*, unsigned int) noexcept nogil>__cuEventCreate)(
        phEvent, Flags)


cdef CUresult _cuEventRecord(CUevent hEvent, CUstream hStream) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuEventRecord
    _check_or_init_driver()
    if __cuEventRecord == NULL:
        with gil:
            raise FunctionNotFoundError("function cuEventRecord is not found")
    return (<CUresult (*)(CUevent, CUstream) noexcept nogil>__cuEventRecord)(
        hEvent, hStream)


cdef CUresult _cuEventRecordWithFlags(CUevent hEvent, CUstream hStream, unsigned int flags) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuEventRecordWithFlags
    _check_or_init_driver()
    if __cuEventRecordWithFlags == NULL:
        with gil:
            raise FunctionNotFoundError("function cuEventRecordWithFlags is not found")
    return (<CUresult (*)(CUevent, CUstream, unsigned int) noexcept nogil>__cuEventRecordWithFlags)(
        hEvent, hStream, flags)


cdef CUresult _cuEventQuery(CUevent hEvent) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuEventQuery
    _check_or_init_driver()
    if __cuEventQuery == NULL:
        with gil:
            raise FunctionNotFoundError("function cuEventQuery is not found")
    return (<CUresult (*)(CUevent) noexcept nogil>__cuEventQuery)(
        hEvent)


cdef CUresult _cuEventSynchronize(CUevent hEvent) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuEventSynchronize
    _check_or_init_driver()
    if __cuEventSynchronize == NULL:
        with gil:
            raise FunctionNotFoundError("function cuEventSynchronize is not found")
    return (<CUresult (*)(CUevent) noexcept nogil>__cuEventSynchronize)(
        hEvent)


cdef CUresult _cuEventDestroy_v2(CUevent hEvent) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuEventDestroy_v2
    _check_or_init_driver()
    if __cuEventDestroy_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuEventDestroy_v2 is not found")
    return (<CUresult (*)(CUevent) noexcept nogil>__cuEventDestroy_v2)(
        hEvent)


cdef CUresult _cuEventElapsedTime_v2(float* pMilliseconds, CUevent hStart, CUevent hEnd) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuEventElapsedTime_v2
    _check_or_init_driver()
    if __cuEventElapsedTime_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuEventElapsedTime_v2 is not found")
    return (<CUresult (*)(float*, CUevent, CUevent) noexcept nogil>__cuEventElapsedTime_v2)(
        pMilliseconds, hStart, hEnd)


cdef CUresult _cuImportExternalMemory(CUexternalMemory* extMem_out, const CUDA_EXTERNAL_MEMORY_HANDLE_DESC* memHandleDesc) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuImportExternalMemory
    _check_or_init_driver()
    if __cuImportExternalMemory == NULL:
        with gil:
            raise FunctionNotFoundError("function cuImportExternalMemory is not found")
    return (<CUresult (*)(CUexternalMemory*, const CUDA_EXTERNAL_MEMORY_HANDLE_DESC*) noexcept nogil>__cuImportExternalMemory)(
        extMem_out, memHandleDesc)


cdef CUresult _cuExternalMemoryGetMappedBuffer(CUdeviceptr* devPtr, CUexternalMemory extMem, const CUDA_EXTERNAL_MEMORY_BUFFER_DESC* bufferDesc) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuExternalMemoryGetMappedBuffer
    _check_or_init_driver()
    if __cuExternalMemoryGetMappedBuffer == NULL:
        with gil:
            raise FunctionNotFoundError("function cuExternalMemoryGetMappedBuffer is not found")
    return (<CUresult (*)(CUdeviceptr*, CUexternalMemory, const CUDA_EXTERNAL_MEMORY_BUFFER_DESC*) noexcept nogil>__cuExternalMemoryGetMappedBuffer)(
        devPtr, extMem, bufferDesc)


cdef CUresult _cuExternalMemoryGetMappedMipmappedArray(CUmipmappedArray* mipmap, CUexternalMemory extMem, const CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC* mipmapDesc) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuExternalMemoryGetMappedMipmappedArray
    _check_or_init_driver()
    if __cuExternalMemoryGetMappedMipmappedArray == NULL:
        with gil:
            raise FunctionNotFoundError("function cuExternalMemoryGetMappedMipmappedArray is not found")
    return (<CUresult (*)(CUmipmappedArray*, CUexternalMemory, const CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC*) noexcept nogil>__cuExternalMemoryGetMappedMipmappedArray)(
        mipmap, extMem, mipmapDesc)


cdef CUresult _cuDestroyExternalMemory(CUexternalMemory extMem) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuDestroyExternalMemory
    _check_or_init_driver()
    if __cuDestroyExternalMemory == NULL:
        with gil:
            raise FunctionNotFoundError("function cuDestroyExternalMemory is not found")
    return (<CUresult (*)(CUexternalMemory) noexcept nogil>__cuDestroyExternalMemory)(
        extMem)


cdef CUresult _cuImportExternalSemaphore(CUexternalSemaphore* extSem_out, const CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC* semHandleDesc) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuImportExternalSemaphore
    _check_or_init_driver()
    if __cuImportExternalSemaphore == NULL:
        with gil:
            raise FunctionNotFoundError("function cuImportExternalSemaphore is not found")
    return (<CUresult (*)(CUexternalSemaphore*, const CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC*) noexcept nogil>__cuImportExternalSemaphore)(
        extSem_out, semHandleDesc)


cdef CUresult _cuSignalExternalSemaphoresAsync(const CUexternalSemaphore* extSemArray, const CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS* paramsArray, unsigned int numExtSems, CUstream stream) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuSignalExternalSemaphoresAsync
    _check_or_init_driver()
    if __cuSignalExternalSemaphoresAsync == NULL:
        with gil:
            raise FunctionNotFoundError("function cuSignalExternalSemaphoresAsync is not found")
    return (<CUresult (*)(const CUexternalSemaphore*, const CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS*, unsigned int, CUstream) noexcept nogil>__cuSignalExternalSemaphoresAsync)(
        extSemArray, paramsArray, numExtSems, stream)


cdef CUresult _cuWaitExternalSemaphoresAsync(const CUexternalSemaphore* extSemArray, const CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS* paramsArray, unsigned int numExtSems, CUstream stream) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuWaitExternalSemaphoresAsync
    _check_or_init_driver()
    if __cuWaitExternalSemaphoresAsync == NULL:
        with gil:
            raise FunctionNotFoundError("function cuWaitExternalSemaphoresAsync is not found")
    return (<CUresult (*)(const CUexternalSemaphore*, const CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS*, unsigned int, CUstream) noexcept nogil>__cuWaitExternalSemaphoresAsync)(
        extSemArray, paramsArray, numExtSems, stream)


cdef CUresult _cuDestroyExternalSemaphore(CUexternalSemaphore extSem) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuDestroyExternalSemaphore
    _check_or_init_driver()
    if __cuDestroyExternalSemaphore == NULL:
        with gil:
            raise FunctionNotFoundError("function cuDestroyExternalSemaphore is not found")
    return (<CUresult (*)(CUexternalSemaphore) noexcept nogil>__cuDestroyExternalSemaphore)(
        extSem)


cdef CUresult _cuStreamWaitValue32_v2(CUstream stream, CUdeviceptr addr, cuuint32_t value, unsigned int flags) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuStreamWaitValue32_v2
    _check_or_init_driver()
    if __cuStreamWaitValue32_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuStreamWaitValue32_v2 is not found")
    return (<CUresult (*)(CUstream, CUdeviceptr, cuuint32_t, unsigned int) noexcept nogil>__cuStreamWaitValue32_v2)(
        stream, addr, value, flags)


cdef CUresult _cuStreamWaitValue64_v2(CUstream stream, CUdeviceptr addr, cuuint64_t value, unsigned int flags) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuStreamWaitValue64_v2
    _check_or_init_driver()
    if __cuStreamWaitValue64_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuStreamWaitValue64_v2 is not found")
    return (<CUresult (*)(CUstream, CUdeviceptr, cuuint64_t, unsigned int) noexcept nogil>__cuStreamWaitValue64_v2)(
        stream, addr, value, flags)


cdef CUresult _cuStreamWriteValue32_v2(CUstream stream, CUdeviceptr addr, cuuint32_t value, unsigned int flags) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuStreamWriteValue32_v2
    _check_or_init_driver()
    if __cuStreamWriteValue32_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuStreamWriteValue32_v2 is not found")
    return (<CUresult (*)(CUstream, CUdeviceptr, cuuint32_t, unsigned int) noexcept nogil>__cuStreamWriteValue32_v2)(
        stream, addr, value, flags)


cdef CUresult _cuStreamWriteValue64_v2(CUstream stream, CUdeviceptr addr, cuuint64_t value, unsigned int flags) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuStreamWriteValue64_v2
    _check_or_init_driver()
    if __cuStreamWriteValue64_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuStreamWriteValue64_v2 is not found")
    return (<CUresult (*)(CUstream, CUdeviceptr, cuuint64_t, unsigned int) noexcept nogil>__cuStreamWriteValue64_v2)(
        stream, addr, value, flags)


cdef CUresult _cuStreamBatchMemOp_v2(CUstream stream, unsigned int count, CUstreamBatchMemOpParams* paramArray, unsigned int flags) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuStreamBatchMemOp_v2
    _check_or_init_driver()
    if __cuStreamBatchMemOp_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuStreamBatchMemOp_v2 is not found")
    return (<CUresult (*)(CUstream, unsigned int, CUstreamBatchMemOpParams*, unsigned int) noexcept nogil>__cuStreamBatchMemOp_v2)(
        stream, count, paramArray, flags)


cdef CUresult _cuFuncGetAttribute(int* pi, CUfunction_attribute attrib, CUfunction hfunc) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuFuncGetAttribute
    _check_or_init_driver()
    if __cuFuncGetAttribute == NULL:
        with gil:
            raise FunctionNotFoundError("function cuFuncGetAttribute is not found")
    return (<CUresult (*)(int*, CUfunction_attribute, CUfunction) noexcept nogil>__cuFuncGetAttribute)(
        pi, attrib, hfunc)


cdef CUresult _cuFuncSetAttribute(CUfunction hfunc, CUfunction_attribute attrib, int value) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuFuncSetAttribute
    _check_or_init_driver()
    if __cuFuncSetAttribute == NULL:
        with gil:
            raise FunctionNotFoundError("function cuFuncSetAttribute is not found")
    return (<CUresult (*)(CUfunction, CUfunction_attribute, int) noexcept nogil>__cuFuncSetAttribute)(
        hfunc, attrib, value)


cdef CUresult _cuFuncSetCacheConfig(CUfunction hfunc, CUfunc_cache config) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuFuncSetCacheConfig
    _check_or_init_driver()
    if __cuFuncSetCacheConfig == NULL:
        with gil:
            raise FunctionNotFoundError("function cuFuncSetCacheConfig is not found")
    return (<CUresult (*)(CUfunction, CUfunc_cache) noexcept nogil>__cuFuncSetCacheConfig)(
        hfunc, config)


cdef CUresult _cuFuncGetModule(CUmodule* hmod, CUfunction hfunc) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuFuncGetModule
    _check_or_init_driver()
    if __cuFuncGetModule == NULL:
        with gil:
            raise FunctionNotFoundError("function cuFuncGetModule is not found")
    return (<CUresult (*)(CUmodule*, CUfunction) noexcept nogil>__cuFuncGetModule)(
        hmod, hfunc)


cdef CUresult _cuFuncGetName(const char** name, CUfunction hfunc) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuFuncGetName
    _check_or_init_driver()
    if __cuFuncGetName == NULL:
        with gil:
            raise FunctionNotFoundError("function cuFuncGetName is not found")
    return (<CUresult (*)(const char**, CUfunction) noexcept nogil>__cuFuncGetName)(
        name, hfunc)


cdef CUresult _cuFuncGetParamInfo(CUfunction func, size_t paramIndex, size_t* paramOffset, size_t* paramSize) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuFuncGetParamInfo
    _check_or_init_driver()
    if __cuFuncGetParamInfo == NULL:
        with gil:
            raise FunctionNotFoundError("function cuFuncGetParamInfo is not found")
    return (<CUresult (*)(CUfunction, size_t, size_t*, size_t*) noexcept nogil>__cuFuncGetParamInfo)(
        func, paramIndex, paramOffset, paramSize)


cdef CUresult _cuFuncIsLoaded(CUfunctionLoadingState* state, CUfunction function) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuFuncIsLoaded
    _check_or_init_driver()
    if __cuFuncIsLoaded == NULL:
        with gil:
            raise FunctionNotFoundError("function cuFuncIsLoaded is not found")
    return (<CUresult (*)(CUfunctionLoadingState*, CUfunction) noexcept nogil>__cuFuncIsLoaded)(
        state, function)


cdef CUresult _cuFuncLoad(CUfunction function) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuFuncLoad
    _check_or_init_driver()
    if __cuFuncLoad == NULL:
        with gil:
            raise FunctionNotFoundError("function cuFuncLoad is not found")
    return (<CUresult (*)(CUfunction) noexcept nogil>__cuFuncLoad)(
        function)


cdef CUresult _cuLaunchKernel(CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, unsigned int sharedMemBytes, CUstream hStream, void** kernelParams, void** extra) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuLaunchKernel
    _check_or_init_driver()
    if __cuLaunchKernel == NULL:
        with gil:
            raise FunctionNotFoundError("function cuLaunchKernel is not found")
    return (<CUresult (*)(CUfunction, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, CUstream, void**, void**) noexcept nogil>__cuLaunchKernel)(
        f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, kernelParams, extra)


cdef CUresult _cuLaunchKernelEx(const CUlaunchConfig* config, CUfunction f, void** kernelParams, void** extra) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuLaunchKernelEx
    _check_or_init_driver()
    if __cuLaunchKernelEx == NULL:
        with gil:
            raise FunctionNotFoundError("function cuLaunchKernelEx is not found")
    return (<CUresult (*)(const CUlaunchConfig*, CUfunction, void**, void**) noexcept nogil>__cuLaunchKernelEx)(
        config, f, kernelParams, extra)


cdef CUresult _cuLaunchCooperativeKernel(CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, unsigned int sharedMemBytes, CUstream hStream, void** kernelParams) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuLaunchCooperativeKernel
    _check_or_init_driver()
    if __cuLaunchCooperativeKernel == NULL:
        with gil:
            raise FunctionNotFoundError("function cuLaunchCooperativeKernel is not found")
    return (<CUresult (*)(CUfunction, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, CUstream, void**) noexcept nogil>__cuLaunchCooperativeKernel)(
        f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, kernelParams)


cdef CUresult _cuLaunchCooperativeKernelMultiDevice(CUDA_LAUNCH_PARAMS* launchParamsList, unsigned int numDevices, unsigned int flags) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuLaunchCooperativeKernelMultiDevice
    _check_or_init_driver()
    if __cuLaunchCooperativeKernelMultiDevice == NULL:
        with gil:
            raise FunctionNotFoundError("function cuLaunchCooperativeKernelMultiDevice is not found")
    return (<CUresult (*)(CUDA_LAUNCH_PARAMS*, unsigned int, unsigned int) noexcept nogil>__cuLaunchCooperativeKernelMultiDevice)(
        launchParamsList, numDevices, flags)


cdef CUresult _cuLaunchHostFunc(CUstream hStream, CUhostFn fn, void* userData) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuLaunchHostFunc
    _check_or_init_driver()
    if __cuLaunchHostFunc == NULL:
        with gil:
            raise FunctionNotFoundError("function cuLaunchHostFunc is not found")
    return (<CUresult (*)(CUstream, CUhostFn, void*) noexcept nogil>__cuLaunchHostFunc)(
        hStream, fn, userData)


cdef CUresult _cuFuncSetBlockShape(CUfunction hfunc, int x, int y, int z) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuFuncSetBlockShape
    _check_or_init_driver()
    if __cuFuncSetBlockShape == NULL:
        with gil:
            raise FunctionNotFoundError("function cuFuncSetBlockShape is not found")
    return (<CUresult (*)(CUfunction, int, int, int) noexcept nogil>__cuFuncSetBlockShape)(
        hfunc, x, y, z)


cdef CUresult _cuFuncSetSharedSize(CUfunction hfunc, unsigned int bytes) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuFuncSetSharedSize
    _check_or_init_driver()
    if __cuFuncSetSharedSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cuFuncSetSharedSize is not found")
    return (<CUresult (*)(CUfunction, unsigned int) noexcept nogil>__cuFuncSetSharedSize)(
        hfunc, bytes)


cdef CUresult _cuParamSetSize(CUfunction hfunc, unsigned int numbytes) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuParamSetSize
    _check_or_init_driver()
    if __cuParamSetSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cuParamSetSize is not found")
    return (<CUresult (*)(CUfunction, unsigned int) noexcept nogil>__cuParamSetSize)(
        hfunc, numbytes)


cdef CUresult _cuParamSeti(CUfunction hfunc, int offset, unsigned int value) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuParamSeti
    _check_or_init_driver()
    if __cuParamSeti == NULL:
        with gil:
            raise FunctionNotFoundError("function cuParamSeti is not found")
    return (<CUresult (*)(CUfunction, int, unsigned int) noexcept nogil>__cuParamSeti)(
        hfunc, offset, value)


cdef CUresult _cuParamSetf(CUfunction hfunc, int offset, float value) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuParamSetf
    _check_or_init_driver()
    if __cuParamSetf == NULL:
        with gil:
            raise FunctionNotFoundError("function cuParamSetf is not found")
    return (<CUresult (*)(CUfunction, int, float) noexcept nogil>__cuParamSetf)(
        hfunc, offset, value)


cdef CUresult _cuParamSetv(CUfunction hfunc, int offset, void* ptr, unsigned int numbytes) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuParamSetv
    _check_or_init_driver()
    if __cuParamSetv == NULL:
        with gil:
            raise FunctionNotFoundError("function cuParamSetv is not found")
    return (<CUresult (*)(CUfunction, int, void*, unsigned int) noexcept nogil>__cuParamSetv)(
        hfunc, offset, ptr, numbytes)


cdef CUresult _cuLaunch(CUfunction f) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuLaunch
    _check_or_init_driver()
    if __cuLaunch == NULL:
        with gil:
            raise FunctionNotFoundError("function cuLaunch is not found")
    return (<CUresult (*)(CUfunction) noexcept nogil>__cuLaunch)(
        f)


cdef CUresult _cuLaunchGrid(CUfunction f, int grid_width, int grid_height) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuLaunchGrid
    _check_or_init_driver()
    if __cuLaunchGrid == NULL:
        with gil:
            raise FunctionNotFoundError("function cuLaunchGrid is not found")
    return (<CUresult (*)(CUfunction, int, int) noexcept nogil>__cuLaunchGrid)(
        f, grid_width, grid_height)


cdef CUresult _cuLaunchGridAsync(CUfunction f, int grid_width, int grid_height, CUstream hStream) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuLaunchGridAsync
    _check_or_init_driver()
    if __cuLaunchGridAsync == NULL:
        with gil:
            raise FunctionNotFoundError("function cuLaunchGridAsync is not found")
    return (<CUresult (*)(CUfunction, int, int, CUstream) noexcept nogil>__cuLaunchGridAsync)(
        f, grid_width, grid_height, hStream)


cdef CUresult _cuParamSetTexRef(CUfunction hfunc, int texunit, CUtexref hTexRef) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuParamSetTexRef
    _check_or_init_driver()
    if __cuParamSetTexRef == NULL:
        with gil:
            raise FunctionNotFoundError("function cuParamSetTexRef is not found")
    return (<CUresult (*)(CUfunction, int, CUtexref) noexcept nogil>__cuParamSetTexRef)(
        hfunc, texunit, hTexRef)


cdef CUresult _cuFuncSetSharedMemConfig(CUfunction hfunc, CUsharedconfig config) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuFuncSetSharedMemConfig
    _check_or_init_driver()
    if __cuFuncSetSharedMemConfig == NULL:
        with gil:
            raise FunctionNotFoundError("function cuFuncSetSharedMemConfig is not found")
    return (<CUresult (*)(CUfunction, CUsharedconfig) noexcept nogil>__cuFuncSetSharedMemConfig)(
        hfunc, config)


cdef CUresult _cuGraphCreate(CUgraph* phGraph, unsigned int flags) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuGraphCreate
    _check_or_init_driver()
    if __cuGraphCreate == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphCreate is not found")
    return (<CUresult (*)(CUgraph*, unsigned int) noexcept nogil>__cuGraphCreate)(
        phGraph, flags)


cdef CUresult _cuGraphAddKernelNode_v2(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, size_t numDependencies, const CUDA_KERNEL_NODE_PARAMS* nodeParams) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuGraphAddKernelNode_v2
    _check_or_init_driver()
    if __cuGraphAddKernelNode_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphAddKernelNode_v2 is not found")
    return (<CUresult (*)(CUgraphNode*, CUgraph, const CUgraphNode*, size_t, const CUDA_KERNEL_NODE_PARAMS*) noexcept nogil>__cuGraphAddKernelNode_v2)(
        phGraphNode, hGraph, dependencies, numDependencies, nodeParams)


cdef CUresult _cuGraphKernelNodeGetParams_v2(CUgraphNode hNode, CUDA_KERNEL_NODE_PARAMS* nodeParams) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuGraphKernelNodeGetParams_v2
    _check_or_init_driver()
    if __cuGraphKernelNodeGetParams_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphKernelNodeGetParams_v2 is not found")
    return (<CUresult (*)(CUgraphNode, CUDA_KERNEL_NODE_PARAMS*) noexcept nogil>__cuGraphKernelNodeGetParams_v2)(
        hNode, nodeParams)


cdef CUresult _cuGraphKernelNodeSetParams_v2(CUgraphNode hNode, const CUDA_KERNEL_NODE_PARAMS* nodeParams) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuGraphKernelNodeSetParams_v2
    _check_or_init_driver()
    if __cuGraphKernelNodeSetParams_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphKernelNodeSetParams_v2 is not found")
    return (<CUresult (*)(CUgraphNode, const CUDA_KERNEL_NODE_PARAMS*) noexcept nogil>__cuGraphKernelNodeSetParams_v2)(
        hNode, nodeParams)


cdef CUresult _cuGraphAddMemcpyNode(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, size_t numDependencies, const CUDA_MEMCPY3D* copyParams, CUcontext ctx) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuGraphAddMemcpyNode
    _check_or_init_driver()
    if __cuGraphAddMemcpyNode == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphAddMemcpyNode is not found")
    return (<CUresult (*)(CUgraphNode*, CUgraph, const CUgraphNode*, size_t, const CUDA_MEMCPY3D*, CUcontext) noexcept nogil>__cuGraphAddMemcpyNode)(
        phGraphNode, hGraph, dependencies, numDependencies, copyParams, ctx)


cdef CUresult _cuGraphMemcpyNodeGetParams(CUgraphNode hNode, CUDA_MEMCPY3D* nodeParams) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuGraphMemcpyNodeGetParams
    _check_or_init_driver()
    if __cuGraphMemcpyNodeGetParams == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphMemcpyNodeGetParams is not found")
    return (<CUresult (*)(CUgraphNode, CUDA_MEMCPY3D*) noexcept nogil>__cuGraphMemcpyNodeGetParams)(
        hNode, nodeParams)


cdef CUresult _cuGraphMemcpyNodeSetParams(CUgraphNode hNode, const CUDA_MEMCPY3D* nodeParams) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuGraphMemcpyNodeSetParams
    _check_or_init_driver()
    if __cuGraphMemcpyNodeSetParams == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphMemcpyNodeSetParams is not found")
    return (<CUresult (*)(CUgraphNode, const CUDA_MEMCPY3D*) noexcept nogil>__cuGraphMemcpyNodeSetParams)(
        hNode, nodeParams)


cdef CUresult _cuGraphAddMemsetNode(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, size_t numDependencies, const CUDA_MEMSET_NODE_PARAMS* memsetParams, CUcontext ctx) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuGraphAddMemsetNode
    _check_or_init_driver()
    if __cuGraphAddMemsetNode == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphAddMemsetNode is not found")
    return (<CUresult (*)(CUgraphNode*, CUgraph, const CUgraphNode*, size_t, const CUDA_MEMSET_NODE_PARAMS*, CUcontext) noexcept nogil>__cuGraphAddMemsetNode)(
        phGraphNode, hGraph, dependencies, numDependencies, memsetParams, ctx)


cdef CUresult _cuGraphMemsetNodeGetParams(CUgraphNode hNode, CUDA_MEMSET_NODE_PARAMS* nodeParams) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuGraphMemsetNodeGetParams
    _check_or_init_driver()
    if __cuGraphMemsetNodeGetParams == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphMemsetNodeGetParams is not found")
    return (<CUresult (*)(CUgraphNode, CUDA_MEMSET_NODE_PARAMS*) noexcept nogil>__cuGraphMemsetNodeGetParams)(
        hNode, nodeParams)


cdef CUresult _cuGraphMemsetNodeSetParams(CUgraphNode hNode, const CUDA_MEMSET_NODE_PARAMS* nodeParams) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuGraphMemsetNodeSetParams
    _check_or_init_driver()
    if __cuGraphMemsetNodeSetParams == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphMemsetNodeSetParams is not found")
    return (<CUresult (*)(CUgraphNode, const CUDA_MEMSET_NODE_PARAMS*) noexcept nogil>__cuGraphMemsetNodeSetParams)(
        hNode, nodeParams)


cdef CUresult _cuGraphAddHostNode(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, size_t numDependencies, const CUDA_HOST_NODE_PARAMS* nodeParams) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuGraphAddHostNode
    _check_or_init_driver()
    if __cuGraphAddHostNode == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphAddHostNode is not found")
    return (<CUresult (*)(CUgraphNode*, CUgraph, const CUgraphNode*, size_t, const CUDA_HOST_NODE_PARAMS*) noexcept nogil>__cuGraphAddHostNode)(
        phGraphNode, hGraph, dependencies, numDependencies, nodeParams)


cdef CUresult _cuGraphHostNodeGetParams(CUgraphNode hNode, CUDA_HOST_NODE_PARAMS* nodeParams) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuGraphHostNodeGetParams
    _check_or_init_driver()
    if __cuGraphHostNodeGetParams == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphHostNodeGetParams is not found")
    return (<CUresult (*)(CUgraphNode, CUDA_HOST_NODE_PARAMS*) noexcept nogil>__cuGraphHostNodeGetParams)(
        hNode, nodeParams)


cdef CUresult _cuGraphHostNodeSetParams(CUgraphNode hNode, const CUDA_HOST_NODE_PARAMS* nodeParams) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuGraphHostNodeSetParams
    _check_or_init_driver()
    if __cuGraphHostNodeSetParams == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphHostNodeSetParams is not found")
    return (<CUresult (*)(CUgraphNode, const CUDA_HOST_NODE_PARAMS*) noexcept nogil>__cuGraphHostNodeSetParams)(
        hNode, nodeParams)


cdef CUresult _cuGraphAddChildGraphNode(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, size_t numDependencies, CUgraph childGraph) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuGraphAddChildGraphNode
    _check_or_init_driver()
    if __cuGraphAddChildGraphNode == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphAddChildGraphNode is not found")
    return (<CUresult (*)(CUgraphNode*, CUgraph, const CUgraphNode*, size_t, CUgraph) noexcept nogil>__cuGraphAddChildGraphNode)(
        phGraphNode, hGraph, dependencies, numDependencies, childGraph)


cdef CUresult _cuGraphChildGraphNodeGetGraph(CUgraphNode hNode, CUgraph* phGraph) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuGraphChildGraphNodeGetGraph
    _check_or_init_driver()
    if __cuGraphChildGraphNodeGetGraph == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphChildGraphNodeGetGraph is not found")
    return (<CUresult (*)(CUgraphNode, CUgraph*) noexcept nogil>__cuGraphChildGraphNodeGetGraph)(
        hNode, phGraph)


cdef CUresult _cuGraphAddEmptyNode(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, size_t numDependencies) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuGraphAddEmptyNode
    _check_or_init_driver()
    if __cuGraphAddEmptyNode == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphAddEmptyNode is not found")
    return (<CUresult (*)(CUgraphNode*, CUgraph, const CUgraphNode*, size_t) noexcept nogil>__cuGraphAddEmptyNode)(
        phGraphNode, hGraph, dependencies, numDependencies)


cdef CUresult _cuGraphAddEventRecordNode(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, size_t numDependencies, CUevent event) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuGraphAddEventRecordNode
    _check_or_init_driver()
    if __cuGraphAddEventRecordNode == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphAddEventRecordNode is not found")
    return (<CUresult (*)(CUgraphNode*, CUgraph, const CUgraphNode*, size_t, CUevent) noexcept nogil>__cuGraphAddEventRecordNode)(
        phGraphNode, hGraph, dependencies, numDependencies, event)


cdef CUresult _cuGraphEventRecordNodeGetEvent(CUgraphNode hNode, CUevent* event_out) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuGraphEventRecordNodeGetEvent
    _check_or_init_driver()
    if __cuGraphEventRecordNodeGetEvent == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphEventRecordNodeGetEvent is not found")
    return (<CUresult (*)(CUgraphNode, CUevent*) noexcept nogil>__cuGraphEventRecordNodeGetEvent)(
        hNode, event_out)


cdef CUresult _cuGraphEventRecordNodeSetEvent(CUgraphNode hNode, CUevent event) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuGraphEventRecordNodeSetEvent
    _check_or_init_driver()
    if __cuGraphEventRecordNodeSetEvent == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphEventRecordNodeSetEvent is not found")
    return (<CUresult (*)(CUgraphNode, CUevent) noexcept nogil>__cuGraphEventRecordNodeSetEvent)(
        hNode, event)


cdef CUresult _cuGraphAddEventWaitNode(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, size_t numDependencies, CUevent event) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuGraphAddEventWaitNode
    _check_or_init_driver()
    if __cuGraphAddEventWaitNode == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphAddEventWaitNode is not found")
    return (<CUresult (*)(CUgraphNode*, CUgraph, const CUgraphNode*, size_t, CUevent) noexcept nogil>__cuGraphAddEventWaitNode)(
        phGraphNode, hGraph, dependencies, numDependencies, event)


cdef CUresult _cuGraphEventWaitNodeGetEvent(CUgraphNode hNode, CUevent* event_out) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuGraphEventWaitNodeGetEvent
    _check_or_init_driver()
    if __cuGraphEventWaitNodeGetEvent == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphEventWaitNodeGetEvent is not found")
    return (<CUresult (*)(CUgraphNode, CUevent*) noexcept nogil>__cuGraphEventWaitNodeGetEvent)(
        hNode, event_out)


cdef CUresult _cuGraphEventWaitNodeSetEvent(CUgraphNode hNode, CUevent event) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuGraphEventWaitNodeSetEvent
    _check_or_init_driver()
    if __cuGraphEventWaitNodeSetEvent == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphEventWaitNodeSetEvent is not found")
    return (<CUresult (*)(CUgraphNode, CUevent) noexcept nogil>__cuGraphEventWaitNodeSetEvent)(
        hNode, event)


cdef CUresult _cuGraphAddExternalSemaphoresSignalNode(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, size_t numDependencies, const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS* nodeParams) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuGraphAddExternalSemaphoresSignalNode
    _check_or_init_driver()
    if __cuGraphAddExternalSemaphoresSignalNode == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphAddExternalSemaphoresSignalNode is not found")
    return (<CUresult (*)(CUgraphNode*, CUgraph, const CUgraphNode*, size_t, const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS*) noexcept nogil>__cuGraphAddExternalSemaphoresSignalNode)(
        phGraphNode, hGraph, dependencies, numDependencies, nodeParams)


cdef CUresult _cuGraphExternalSemaphoresSignalNodeGetParams(CUgraphNode hNode, CUDA_EXT_SEM_SIGNAL_NODE_PARAMS* params_out) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuGraphExternalSemaphoresSignalNodeGetParams
    _check_or_init_driver()
    if __cuGraphExternalSemaphoresSignalNodeGetParams == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphExternalSemaphoresSignalNodeGetParams is not found")
    return (<CUresult (*)(CUgraphNode, CUDA_EXT_SEM_SIGNAL_NODE_PARAMS*) noexcept nogil>__cuGraphExternalSemaphoresSignalNodeGetParams)(
        hNode, params_out)


cdef CUresult _cuGraphExternalSemaphoresSignalNodeSetParams(CUgraphNode hNode, const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS* nodeParams) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuGraphExternalSemaphoresSignalNodeSetParams
    _check_or_init_driver()
    if __cuGraphExternalSemaphoresSignalNodeSetParams == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphExternalSemaphoresSignalNodeSetParams is not found")
    return (<CUresult (*)(CUgraphNode, const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS*) noexcept nogil>__cuGraphExternalSemaphoresSignalNodeSetParams)(
        hNode, nodeParams)


cdef CUresult _cuGraphAddExternalSemaphoresWaitNode(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, size_t numDependencies, const CUDA_EXT_SEM_WAIT_NODE_PARAMS* nodeParams) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuGraphAddExternalSemaphoresWaitNode
    _check_or_init_driver()
    if __cuGraphAddExternalSemaphoresWaitNode == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphAddExternalSemaphoresWaitNode is not found")
    return (<CUresult (*)(CUgraphNode*, CUgraph, const CUgraphNode*, size_t, const CUDA_EXT_SEM_WAIT_NODE_PARAMS*) noexcept nogil>__cuGraphAddExternalSemaphoresWaitNode)(
        phGraphNode, hGraph, dependencies, numDependencies, nodeParams)


cdef CUresult _cuGraphExternalSemaphoresWaitNodeGetParams(CUgraphNode hNode, CUDA_EXT_SEM_WAIT_NODE_PARAMS* params_out) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuGraphExternalSemaphoresWaitNodeGetParams
    _check_or_init_driver()
    if __cuGraphExternalSemaphoresWaitNodeGetParams == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphExternalSemaphoresWaitNodeGetParams is not found")
    return (<CUresult (*)(CUgraphNode, CUDA_EXT_SEM_WAIT_NODE_PARAMS*) noexcept nogil>__cuGraphExternalSemaphoresWaitNodeGetParams)(
        hNode, params_out)


cdef CUresult _cuGraphExternalSemaphoresWaitNodeSetParams(CUgraphNode hNode, const CUDA_EXT_SEM_WAIT_NODE_PARAMS* nodeParams) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuGraphExternalSemaphoresWaitNodeSetParams
    _check_or_init_driver()
    if __cuGraphExternalSemaphoresWaitNodeSetParams == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphExternalSemaphoresWaitNodeSetParams is not found")
    return (<CUresult (*)(CUgraphNode, const CUDA_EXT_SEM_WAIT_NODE_PARAMS*) noexcept nogil>__cuGraphExternalSemaphoresWaitNodeSetParams)(
        hNode, nodeParams)


cdef CUresult _cuGraphAddBatchMemOpNode(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, size_t numDependencies, const CUDA_BATCH_MEM_OP_NODE_PARAMS* nodeParams) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuGraphAddBatchMemOpNode
    _check_or_init_driver()
    if __cuGraphAddBatchMemOpNode == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphAddBatchMemOpNode is not found")
    return (<CUresult (*)(CUgraphNode*, CUgraph, const CUgraphNode*, size_t, const CUDA_BATCH_MEM_OP_NODE_PARAMS*) noexcept nogil>__cuGraphAddBatchMemOpNode)(
        phGraphNode, hGraph, dependencies, numDependencies, nodeParams)


cdef CUresult _cuGraphBatchMemOpNodeGetParams(CUgraphNode hNode, CUDA_BATCH_MEM_OP_NODE_PARAMS* nodeParams_out) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuGraphBatchMemOpNodeGetParams
    _check_or_init_driver()
    if __cuGraphBatchMemOpNodeGetParams == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphBatchMemOpNodeGetParams is not found")
    return (<CUresult (*)(CUgraphNode, CUDA_BATCH_MEM_OP_NODE_PARAMS*) noexcept nogil>__cuGraphBatchMemOpNodeGetParams)(
        hNode, nodeParams_out)


cdef CUresult _cuGraphBatchMemOpNodeSetParams(CUgraphNode hNode, const CUDA_BATCH_MEM_OP_NODE_PARAMS* nodeParams) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuGraphBatchMemOpNodeSetParams
    _check_or_init_driver()
    if __cuGraphBatchMemOpNodeSetParams == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphBatchMemOpNodeSetParams is not found")
    return (<CUresult (*)(CUgraphNode, const CUDA_BATCH_MEM_OP_NODE_PARAMS*) noexcept nogil>__cuGraphBatchMemOpNodeSetParams)(
        hNode, nodeParams)


cdef CUresult _cuGraphExecBatchMemOpNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_BATCH_MEM_OP_NODE_PARAMS* nodeParams) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuGraphExecBatchMemOpNodeSetParams
    _check_or_init_driver()
    if __cuGraphExecBatchMemOpNodeSetParams == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphExecBatchMemOpNodeSetParams is not found")
    return (<CUresult (*)(CUgraphExec, CUgraphNode, const CUDA_BATCH_MEM_OP_NODE_PARAMS*) noexcept nogil>__cuGraphExecBatchMemOpNodeSetParams)(
        hGraphExec, hNode, nodeParams)


cdef CUresult _cuGraphAddMemAllocNode(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, size_t numDependencies, CUDA_MEM_ALLOC_NODE_PARAMS* nodeParams) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuGraphAddMemAllocNode
    _check_or_init_driver()
    if __cuGraphAddMemAllocNode == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphAddMemAllocNode is not found")
    return (<CUresult (*)(CUgraphNode*, CUgraph, const CUgraphNode*, size_t, CUDA_MEM_ALLOC_NODE_PARAMS*) noexcept nogil>__cuGraphAddMemAllocNode)(
        phGraphNode, hGraph, dependencies, numDependencies, nodeParams)


cdef CUresult _cuGraphMemAllocNodeGetParams(CUgraphNode hNode, CUDA_MEM_ALLOC_NODE_PARAMS* params_out) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuGraphMemAllocNodeGetParams
    _check_or_init_driver()
    if __cuGraphMemAllocNodeGetParams == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphMemAllocNodeGetParams is not found")
    return (<CUresult (*)(CUgraphNode, CUDA_MEM_ALLOC_NODE_PARAMS*) noexcept nogil>__cuGraphMemAllocNodeGetParams)(
        hNode, params_out)


cdef CUresult _cuGraphAddMemFreeNode(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, size_t numDependencies, CUdeviceptr dptr) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuGraphAddMemFreeNode
    _check_or_init_driver()
    if __cuGraphAddMemFreeNode == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphAddMemFreeNode is not found")
    return (<CUresult (*)(CUgraphNode*, CUgraph, const CUgraphNode*, size_t, CUdeviceptr) noexcept nogil>__cuGraphAddMemFreeNode)(
        phGraphNode, hGraph, dependencies, numDependencies, dptr)


cdef CUresult _cuGraphMemFreeNodeGetParams(CUgraphNode hNode, CUdeviceptr* dptr_out) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuGraphMemFreeNodeGetParams
    _check_or_init_driver()
    if __cuGraphMemFreeNodeGetParams == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphMemFreeNodeGetParams is not found")
    return (<CUresult (*)(CUgraphNode, CUdeviceptr*) noexcept nogil>__cuGraphMemFreeNodeGetParams)(
        hNode, dptr_out)


cdef CUresult _cuDeviceGraphMemTrim(CUdevice device) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuDeviceGraphMemTrim
    _check_or_init_driver()
    if __cuDeviceGraphMemTrim == NULL:
        with gil:
            raise FunctionNotFoundError("function cuDeviceGraphMemTrim is not found")
    return (<CUresult (*)(CUdevice) noexcept nogil>__cuDeviceGraphMemTrim)(
        device)


cdef CUresult _cuDeviceGetGraphMemAttribute(CUdevice device, CUgraphMem_attribute attr, void* value) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuDeviceGetGraphMemAttribute
    _check_or_init_driver()
    if __cuDeviceGetGraphMemAttribute == NULL:
        with gil:
            raise FunctionNotFoundError("function cuDeviceGetGraphMemAttribute is not found")
    return (<CUresult (*)(CUdevice, CUgraphMem_attribute, void*) noexcept nogil>__cuDeviceGetGraphMemAttribute)(
        device, attr, value)


cdef CUresult _cuDeviceSetGraphMemAttribute(CUdevice device, CUgraphMem_attribute attr, void* value) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuDeviceSetGraphMemAttribute
    _check_or_init_driver()
    if __cuDeviceSetGraphMemAttribute == NULL:
        with gil:
            raise FunctionNotFoundError("function cuDeviceSetGraphMemAttribute is not found")
    return (<CUresult (*)(CUdevice, CUgraphMem_attribute, void*) noexcept nogil>__cuDeviceSetGraphMemAttribute)(
        device, attr, value)


cdef CUresult _cuGraphClone(CUgraph* phGraphClone, CUgraph originalGraph) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuGraphClone
    _check_or_init_driver()
    if __cuGraphClone == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphClone is not found")
    return (<CUresult (*)(CUgraph*, CUgraph) noexcept nogil>__cuGraphClone)(
        phGraphClone, originalGraph)


cdef CUresult _cuGraphNodeFindInClone(CUgraphNode* phNode, CUgraphNode hOriginalNode, CUgraph hClonedGraph) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuGraphNodeFindInClone
    _check_or_init_driver()
    if __cuGraphNodeFindInClone == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphNodeFindInClone is not found")
    return (<CUresult (*)(CUgraphNode*, CUgraphNode, CUgraph) noexcept nogil>__cuGraphNodeFindInClone)(
        phNode, hOriginalNode, hClonedGraph)


cdef CUresult _cuGraphNodeGetType(CUgraphNode hNode, CUgraphNodeType* type) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuGraphNodeGetType
    _check_or_init_driver()
    if __cuGraphNodeGetType == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphNodeGetType is not found")
    return (<CUresult (*)(CUgraphNode, CUgraphNodeType*) noexcept nogil>__cuGraphNodeGetType)(
        hNode, type)


cdef CUresult _cuGraphGetNodes(CUgraph hGraph, CUgraphNode* nodes, size_t* numNodes) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuGraphGetNodes
    _check_or_init_driver()
    if __cuGraphGetNodes == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphGetNodes is not found")
    return (<CUresult (*)(CUgraph, CUgraphNode*, size_t*) noexcept nogil>__cuGraphGetNodes)(
        hGraph, nodes, numNodes)


cdef CUresult _cuGraphGetRootNodes(CUgraph hGraph, CUgraphNode* rootNodes, size_t* numRootNodes) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuGraphGetRootNodes
    _check_or_init_driver()
    if __cuGraphGetRootNodes == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphGetRootNodes is not found")
    return (<CUresult (*)(CUgraph, CUgraphNode*, size_t*) noexcept nogil>__cuGraphGetRootNodes)(
        hGraph, rootNodes, numRootNodes)


cdef CUresult _cuGraphGetEdges_v2(CUgraph hGraph, CUgraphNode* from_, CUgraphNode* to, CUgraphEdgeData* edgeData, size_t* numEdges) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuGraphGetEdges_v2
    _check_or_init_driver()
    if __cuGraphGetEdges_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphGetEdges_v2 is not found")
    return (<CUresult (*)(CUgraph, CUgraphNode*, CUgraphNode*, CUgraphEdgeData*, size_t*) noexcept nogil>__cuGraphGetEdges_v2)(
        hGraph, from_, to, edgeData, numEdges)


cdef CUresult _cuGraphNodeGetDependencies_v2(CUgraphNode hNode, CUgraphNode* dependencies, CUgraphEdgeData* edgeData, size_t* numDependencies) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuGraphNodeGetDependencies_v2
    _check_or_init_driver()
    if __cuGraphNodeGetDependencies_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphNodeGetDependencies_v2 is not found")
    return (<CUresult (*)(CUgraphNode, CUgraphNode*, CUgraphEdgeData*, size_t*) noexcept nogil>__cuGraphNodeGetDependencies_v2)(
        hNode, dependencies, edgeData, numDependencies)


cdef CUresult _cuGraphNodeGetDependentNodes_v2(CUgraphNode hNode, CUgraphNode* dependentNodes, CUgraphEdgeData* edgeData, size_t* numDependentNodes) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuGraphNodeGetDependentNodes_v2
    _check_or_init_driver()
    if __cuGraphNodeGetDependentNodes_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphNodeGetDependentNodes_v2 is not found")
    return (<CUresult (*)(CUgraphNode, CUgraphNode*, CUgraphEdgeData*, size_t*) noexcept nogil>__cuGraphNodeGetDependentNodes_v2)(
        hNode, dependentNodes, edgeData, numDependentNodes)


cdef CUresult _cuGraphAddDependencies_v2(CUgraph hGraph, const CUgraphNode* from_, const CUgraphNode* to, const CUgraphEdgeData* edgeData, size_t numDependencies) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuGraphAddDependencies_v2
    _check_or_init_driver()
    if __cuGraphAddDependencies_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphAddDependencies_v2 is not found")
    return (<CUresult (*)(CUgraph, const CUgraphNode*, const CUgraphNode*, const CUgraphEdgeData*, size_t) noexcept nogil>__cuGraphAddDependencies_v2)(
        hGraph, from_, to, edgeData, numDependencies)


cdef CUresult _cuGraphRemoveDependencies_v2(CUgraph hGraph, const CUgraphNode* from_, const CUgraphNode* to, const CUgraphEdgeData* edgeData, size_t numDependencies) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuGraphRemoveDependencies_v2
    _check_or_init_driver()
    if __cuGraphRemoveDependencies_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphRemoveDependencies_v2 is not found")
    return (<CUresult (*)(CUgraph, const CUgraphNode*, const CUgraphNode*, const CUgraphEdgeData*, size_t) noexcept nogil>__cuGraphRemoveDependencies_v2)(
        hGraph, from_, to, edgeData, numDependencies)


cdef CUresult _cuGraphDestroyNode(CUgraphNode hNode) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuGraphDestroyNode
    _check_or_init_driver()
    if __cuGraphDestroyNode == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphDestroyNode is not found")
    return (<CUresult (*)(CUgraphNode) noexcept nogil>__cuGraphDestroyNode)(
        hNode)


cdef CUresult _cuGraphInstantiateWithFlags(CUgraphExec* phGraphExec, CUgraph hGraph, unsigned long long flags) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuGraphInstantiateWithFlags
    _check_or_init_driver()
    if __cuGraphInstantiateWithFlags == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphInstantiateWithFlags is not found")
    return (<CUresult (*)(CUgraphExec*, CUgraph, unsigned long long) noexcept nogil>__cuGraphInstantiateWithFlags)(
        phGraphExec, hGraph, flags)


cdef CUresult _cuGraphInstantiateWithParams(CUgraphExec* phGraphExec, CUgraph hGraph, CUDA_GRAPH_INSTANTIATE_PARAMS* instantiateParams) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuGraphInstantiateWithParams
    _check_or_init_driver()
    if __cuGraphInstantiateWithParams == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphInstantiateWithParams is not found")
    return (<CUresult (*)(CUgraphExec*, CUgraph, CUDA_GRAPH_INSTANTIATE_PARAMS*) noexcept nogil>__cuGraphInstantiateWithParams)(
        phGraphExec, hGraph, instantiateParams)


cdef CUresult _cuGraphExecGetFlags(CUgraphExec hGraphExec, cuuint64_t* flags) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuGraphExecGetFlags
    _check_or_init_driver()
    if __cuGraphExecGetFlags == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphExecGetFlags is not found")
    return (<CUresult (*)(CUgraphExec, cuuint64_t*) noexcept nogil>__cuGraphExecGetFlags)(
        hGraphExec, flags)


cdef CUresult _cuGraphExecKernelNodeSetParams_v2(CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_KERNEL_NODE_PARAMS* nodeParams) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuGraphExecKernelNodeSetParams_v2
    _check_or_init_driver()
    if __cuGraphExecKernelNodeSetParams_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphExecKernelNodeSetParams_v2 is not found")
    return (<CUresult (*)(CUgraphExec, CUgraphNode, const CUDA_KERNEL_NODE_PARAMS*) noexcept nogil>__cuGraphExecKernelNodeSetParams_v2)(
        hGraphExec, hNode, nodeParams)


cdef CUresult _cuGraphExecMemcpyNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_MEMCPY3D* copyParams, CUcontext ctx) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuGraphExecMemcpyNodeSetParams
    _check_or_init_driver()
    if __cuGraphExecMemcpyNodeSetParams == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphExecMemcpyNodeSetParams is not found")
    return (<CUresult (*)(CUgraphExec, CUgraphNode, const CUDA_MEMCPY3D*, CUcontext) noexcept nogil>__cuGraphExecMemcpyNodeSetParams)(
        hGraphExec, hNode, copyParams, ctx)


cdef CUresult _cuGraphExecMemsetNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_MEMSET_NODE_PARAMS* memsetParams, CUcontext ctx) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuGraphExecMemsetNodeSetParams
    _check_or_init_driver()
    if __cuGraphExecMemsetNodeSetParams == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphExecMemsetNodeSetParams is not found")
    return (<CUresult (*)(CUgraphExec, CUgraphNode, const CUDA_MEMSET_NODE_PARAMS*, CUcontext) noexcept nogil>__cuGraphExecMemsetNodeSetParams)(
        hGraphExec, hNode, memsetParams, ctx)


cdef CUresult _cuGraphExecHostNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_HOST_NODE_PARAMS* nodeParams) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuGraphExecHostNodeSetParams
    _check_or_init_driver()
    if __cuGraphExecHostNodeSetParams == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphExecHostNodeSetParams is not found")
    return (<CUresult (*)(CUgraphExec, CUgraphNode, const CUDA_HOST_NODE_PARAMS*) noexcept nogil>__cuGraphExecHostNodeSetParams)(
        hGraphExec, hNode, nodeParams)


cdef CUresult _cuGraphExecChildGraphNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, CUgraph childGraph) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuGraphExecChildGraphNodeSetParams
    _check_or_init_driver()
    if __cuGraphExecChildGraphNodeSetParams == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphExecChildGraphNodeSetParams is not found")
    return (<CUresult (*)(CUgraphExec, CUgraphNode, CUgraph) noexcept nogil>__cuGraphExecChildGraphNodeSetParams)(
        hGraphExec, hNode, childGraph)


cdef CUresult _cuGraphExecEventRecordNodeSetEvent(CUgraphExec hGraphExec, CUgraphNode hNode, CUevent event) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuGraphExecEventRecordNodeSetEvent
    _check_or_init_driver()
    if __cuGraphExecEventRecordNodeSetEvent == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphExecEventRecordNodeSetEvent is not found")
    return (<CUresult (*)(CUgraphExec, CUgraphNode, CUevent) noexcept nogil>__cuGraphExecEventRecordNodeSetEvent)(
        hGraphExec, hNode, event)


cdef CUresult _cuGraphExecEventWaitNodeSetEvent(CUgraphExec hGraphExec, CUgraphNode hNode, CUevent event) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuGraphExecEventWaitNodeSetEvent
    _check_or_init_driver()
    if __cuGraphExecEventWaitNodeSetEvent == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphExecEventWaitNodeSetEvent is not found")
    return (<CUresult (*)(CUgraphExec, CUgraphNode, CUevent) noexcept nogil>__cuGraphExecEventWaitNodeSetEvent)(
        hGraphExec, hNode, event)


cdef CUresult _cuGraphExecExternalSemaphoresSignalNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS* nodeParams) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuGraphExecExternalSemaphoresSignalNodeSetParams
    _check_or_init_driver()
    if __cuGraphExecExternalSemaphoresSignalNodeSetParams == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphExecExternalSemaphoresSignalNodeSetParams is not found")
    return (<CUresult (*)(CUgraphExec, CUgraphNode, const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS*) noexcept nogil>__cuGraphExecExternalSemaphoresSignalNodeSetParams)(
        hGraphExec, hNode, nodeParams)


cdef CUresult _cuGraphExecExternalSemaphoresWaitNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_EXT_SEM_WAIT_NODE_PARAMS* nodeParams) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuGraphExecExternalSemaphoresWaitNodeSetParams
    _check_or_init_driver()
    if __cuGraphExecExternalSemaphoresWaitNodeSetParams == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphExecExternalSemaphoresWaitNodeSetParams is not found")
    return (<CUresult (*)(CUgraphExec, CUgraphNode, const CUDA_EXT_SEM_WAIT_NODE_PARAMS*) noexcept nogil>__cuGraphExecExternalSemaphoresWaitNodeSetParams)(
        hGraphExec, hNode, nodeParams)


cdef CUresult _cuGraphNodeSetEnabled(CUgraphExec hGraphExec, CUgraphNode hNode, unsigned int isEnabled) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuGraphNodeSetEnabled
    _check_or_init_driver()
    if __cuGraphNodeSetEnabled == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphNodeSetEnabled is not found")
    return (<CUresult (*)(CUgraphExec, CUgraphNode, unsigned int) noexcept nogil>__cuGraphNodeSetEnabled)(
        hGraphExec, hNode, isEnabled)


cdef CUresult _cuGraphNodeGetEnabled(CUgraphExec hGraphExec, CUgraphNode hNode, unsigned int* isEnabled) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuGraphNodeGetEnabled
    _check_or_init_driver()
    if __cuGraphNodeGetEnabled == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphNodeGetEnabled is not found")
    return (<CUresult (*)(CUgraphExec, CUgraphNode, unsigned int*) noexcept nogil>__cuGraphNodeGetEnabled)(
        hGraphExec, hNode, isEnabled)


cdef CUresult _cuGraphUpload(CUgraphExec hGraphExec, CUstream hStream) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuGraphUpload
    _check_or_init_driver()
    if __cuGraphUpload == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphUpload is not found")
    return (<CUresult (*)(CUgraphExec, CUstream) noexcept nogil>__cuGraphUpload)(
        hGraphExec, hStream)


cdef CUresult _cuGraphLaunch(CUgraphExec hGraphExec, CUstream hStream) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuGraphLaunch
    _check_or_init_driver()
    if __cuGraphLaunch == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphLaunch is not found")
    return (<CUresult (*)(CUgraphExec, CUstream) noexcept nogil>__cuGraphLaunch)(
        hGraphExec, hStream)


cdef CUresult _cuGraphExecDestroy(CUgraphExec hGraphExec) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuGraphExecDestroy
    _check_or_init_driver()
    if __cuGraphExecDestroy == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphExecDestroy is not found")
    return (<CUresult (*)(CUgraphExec) noexcept nogil>__cuGraphExecDestroy)(
        hGraphExec)


cdef CUresult _cuGraphDestroy(CUgraph hGraph) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuGraphDestroy
    _check_or_init_driver()
    if __cuGraphDestroy == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphDestroy is not found")
    return (<CUresult (*)(CUgraph) noexcept nogil>__cuGraphDestroy)(
        hGraph)


cdef CUresult _cuGraphExecUpdate_v2(CUgraphExec hGraphExec, CUgraph hGraph, CUgraphExecUpdateResultInfo* resultInfo) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuGraphExecUpdate_v2
    _check_or_init_driver()
    if __cuGraphExecUpdate_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphExecUpdate_v2 is not found")
    return (<CUresult (*)(CUgraphExec, CUgraph, CUgraphExecUpdateResultInfo*) noexcept nogil>__cuGraphExecUpdate_v2)(
        hGraphExec, hGraph, resultInfo)


cdef CUresult _cuGraphKernelNodeCopyAttributes(CUgraphNode dst, CUgraphNode src) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuGraphKernelNodeCopyAttributes
    _check_or_init_driver()
    if __cuGraphKernelNodeCopyAttributes == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphKernelNodeCopyAttributes is not found")
    return (<CUresult (*)(CUgraphNode, CUgraphNode) noexcept nogil>__cuGraphKernelNodeCopyAttributes)(
        dst, src)


cdef CUresult _cuGraphKernelNodeGetAttribute(CUgraphNode hNode, CUkernelNodeAttrID attr, CUkernelNodeAttrValue* value_out) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuGraphKernelNodeGetAttribute
    _check_or_init_driver()
    if __cuGraphKernelNodeGetAttribute == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphKernelNodeGetAttribute is not found")
    return (<CUresult (*)(CUgraphNode, CUkernelNodeAttrID, CUkernelNodeAttrValue*) noexcept nogil>__cuGraphKernelNodeGetAttribute)(
        hNode, attr, value_out)


cdef CUresult _cuGraphKernelNodeSetAttribute(CUgraphNode hNode, CUkernelNodeAttrID attr, const CUkernelNodeAttrValue* value) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuGraphKernelNodeSetAttribute
    _check_or_init_driver()
    if __cuGraphKernelNodeSetAttribute == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphKernelNodeSetAttribute is not found")
    return (<CUresult (*)(CUgraphNode, CUkernelNodeAttrID, const CUkernelNodeAttrValue*) noexcept nogil>__cuGraphKernelNodeSetAttribute)(
        hNode, attr, value)


cdef CUresult _cuGraphDebugDotPrint(CUgraph hGraph, const char* path, unsigned int flags) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuGraphDebugDotPrint
    _check_or_init_driver()
    if __cuGraphDebugDotPrint == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphDebugDotPrint is not found")
    return (<CUresult (*)(CUgraph, const char*, unsigned int) noexcept nogil>__cuGraphDebugDotPrint)(
        hGraph, path, flags)


cdef CUresult _cuUserObjectCreate(CUuserObject* object_out, void* ptr, CUhostFn destroy, unsigned int initialRefcount, unsigned int flags) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuUserObjectCreate
    _check_or_init_driver()
    if __cuUserObjectCreate == NULL:
        with gil:
            raise FunctionNotFoundError("function cuUserObjectCreate is not found")
    return (<CUresult (*)(CUuserObject*, void*, CUhostFn, unsigned int, unsigned int) noexcept nogil>__cuUserObjectCreate)(
        object_out, ptr, destroy, initialRefcount, flags)


cdef CUresult _cuUserObjectRetain(CUuserObject object, unsigned int count) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuUserObjectRetain
    _check_or_init_driver()
    if __cuUserObjectRetain == NULL:
        with gil:
            raise FunctionNotFoundError("function cuUserObjectRetain is not found")
    return (<CUresult (*)(CUuserObject, unsigned int) noexcept nogil>__cuUserObjectRetain)(
        object, count)


cdef CUresult _cuUserObjectRelease(CUuserObject object, unsigned int count) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuUserObjectRelease
    _check_or_init_driver()
    if __cuUserObjectRelease == NULL:
        with gil:
            raise FunctionNotFoundError("function cuUserObjectRelease is not found")
    return (<CUresult (*)(CUuserObject, unsigned int) noexcept nogil>__cuUserObjectRelease)(
        object, count)


cdef CUresult _cuGraphRetainUserObject(CUgraph graph, CUuserObject object, unsigned int count, unsigned int flags) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuGraphRetainUserObject
    _check_or_init_driver()
    if __cuGraphRetainUserObject == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphRetainUserObject is not found")
    return (<CUresult (*)(CUgraph, CUuserObject, unsigned int, unsigned int) noexcept nogil>__cuGraphRetainUserObject)(
        graph, object, count, flags)


cdef CUresult _cuGraphReleaseUserObject(CUgraph graph, CUuserObject object, unsigned int count) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuGraphReleaseUserObject
    _check_or_init_driver()
    if __cuGraphReleaseUserObject == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphReleaseUserObject is not found")
    return (<CUresult (*)(CUgraph, CUuserObject, unsigned int) noexcept nogil>__cuGraphReleaseUserObject)(
        graph, object, count)


cdef CUresult _cuGraphAddNode_v2(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, const CUgraphEdgeData* dependencyData, size_t numDependencies, CUgraphNodeParams* nodeParams) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuGraphAddNode_v2
    _check_or_init_driver()
    if __cuGraphAddNode_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphAddNode_v2 is not found")
    return (<CUresult (*)(CUgraphNode*, CUgraph, const CUgraphNode*, const CUgraphEdgeData*, size_t, CUgraphNodeParams*) noexcept nogil>__cuGraphAddNode_v2)(
        phGraphNode, hGraph, dependencies, dependencyData, numDependencies, nodeParams)


cdef CUresult _cuGraphNodeSetParams(CUgraphNode hNode, CUgraphNodeParams* nodeParams) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuGraphNodeSetParams
    _check_or_init_driver()
    if __cuGraphNodeSetParams == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphNodeSetParams is not found")
    return (<CUresult (*)(CUgraphNode, CUgraphNodeParams*) noexcept nogil>__cuGraphNodeSetParams)(
        hNode, nodeParams)


cdef CUresult _cuGraphExecNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, CUgraphNodeParams* nodeParams) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuGraphExecNodeSetParams
    _check_or_init_driver()
    if __cuGraphExecNodeSetParams == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphExecNodeSetParams is not found")
    return (<CUresult (*)(CUgraphExec, CUgraphNode, CUgraphNodeParams*) noexcept nogil>__cuGraphExecNodeSetParams)(
        hGraphExec, hNode, nodeParams)


cdef CUresult _cuGraphConditionalHandleCreate(CUgraphConditionalHandle* pHandle_out, CUgraph hGraph, CUcontext ctx, unsigned int defaultLaunchValue, unsigned int flags) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuGraphConditionalHandleCreate
    _check_or_init_driver()
    if __cuGraphConditionalHandleCreate == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphConditionalHandleCreate is not found")
    return (<CUresult (*)(CUgraphConditionalHandle*, CUgraph, CUcontext, unsigned int, unsigned int) noexcept nogil>__cuGraphConditionalHandleCreate)(
        pHandle_out, hGraph, ctx, defaultLaunchValue, flags)


cdef CUresult _cuOccupancyMaxActiveBlocksPerMultiprocessor(int* numBlocks, CUfunction func, int blockSize, size_t dynamicSMemSize) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuOccupancyMaxActiveBlocksPerMultiprocessor
    _check_or_init_driver()
    if __cuOccupancyMaxActiveBlocksPerMultiprocessor == NULL:
        with gil:
            raise FunctionNotFoundError("function cuOccupancyMaxActiveBlocksPerMultiprocessor is not found")
    return (<CUresult (*)(int*, CUfunction, int, size_t) noexcept nogil>__cuOccupancyMaxActiveBlocksPerMultiprocessor)(
        numBlocks, func, blockSize, dynamicSMemSize)


cdef CUresult _cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int* numBlocks, CUfunction func, int blockSize, size_t dynamicSMemSize, unsigned int flags) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags
    _check_or_init_driver()
    if __cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags == NULL:
        with gil:
            raise FunctionNotFoundError("function cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags is not found")
    return (<CUresult (*)(int*, CUfunction, int, size_t, unsigned int) noexcept nogil>__cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags)(
        numBlocks, func, blockSize, dynamicSMemSize, flags)


cdef CUresult _cuOccupancyMaxPotentialBlockSize(int* minGridSize, int* blockSize, CUfunction func, CUoccupancyB2DSize blockSizeToDynamicSMemSize, size_t dynamicSMemSize, int blockSizeLimit) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuOccupancyMaxPotentialBlockSize
    _check_or_init_driver()
    if __cuOccupancyMaxPotentialBlockSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cuOccupancyMaxPotentialBlockSize is not found")
    return (<CUresult (*)(int*, int*, CUfunction, CUoccupancyB2DSize, size_t, int) noexcept nogil>__cuOccupancyMaxPotentialBlockSize)(
        minGridSize, blockSize, func, blockSizeToDynamicSMemSize, dynamicSMemSize, blockSizeLimit)


cdef CUresult _cuOccupancyMaxPotentialBlockSizeWithFlags(int* minGridSize, int* blockSize, CUfunction func, CUoccupancyB2DSize blockSizeToDynamicSMemSize, size_t dynamicSMemSize, int blockSizeLimit, unsigned int flags) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuOccupancyMaxPotentialBlockSizeWithFlags
    _check_or_init_driver()
    if __cuOccupancyMaxPotentialBlockSizeWithFlags == NULL:
        with gil:
            raise FunctionNotFoundError("function cuOccupancyMaxPotentialBlockSizeWithFlags is not found")
    return (<CUresult (*)(int*, int*, CUfunction, CUoccupancyB2DSize, size_t, int, unsigned int) noexcept nogil>__cuOccupancyMaxPotentialBlockSizeWithFlags)(
        minGridSize, blockSize, func, blockSizeToDynamicSMemSize, dynamicSMemSize, blockSizeLimit, flags)


cdef CUresult _cuOccupancyAvailableDynamicSMemPerBlock(size_t* dynamicSmemSize, CUfunction func, int numBlocks, int blockSize) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuOccupancyAvailableDynamicSMemPerBlock
    _check_or_init_driver()
    if __cuOccupancyAvailableDynamicSMemPerBlock == NULL:
        with gil:
            raise FunctionNotFoundError("function cuOccupancyAvailableDynamicSMemPerBlock is not found")
    return (<CUresult (*)(size_t*, CUfunction, int, int) noexcept nogil>__cuOccupancyAvailableDynamicSMemPerBlock)(
        dynamicSmemSize, func, numBlocks, blockSize)


cdef CUresult _cuOccupancyMaxPotentialClusterSize(int* clusterSize, CUfunction func, const CUlaunchConfig* config) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuOccupancyMaxPotentialClusterSize
    _check_or_init_driver()
    if __cuOccupancyMaxPotentialClusterSize == NULL:
        with gil:
            raise FunctionNotFoundError("function cuOccupancyMaxPotentialClusterSize is not found")
    return (<CUresult (*)(int*, CUfunction, const CUlaunchConfig*) noexcept nogil>__cuOccupancyMaxPotentialClusterSize)(
        clusterSize, func, config)


cdef CUresult _cuOccupancyMaxActiveClusters(int* numClusters, CUfunction func, const CUlaunchConfig* config) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuOccupancyMaxActiveClusters
    _check_or_init_driver()
    if __cuOccupancyMaxActiveClusters == NULL:
        with gil:
            raise FunctionNotFoundError("function cuOccupancyMaxActiveClusters is not found")
    return (<CUresult (*)(int*, CUfunction, const CUlaunchConfig*) noexcept nogil>__cuOccupancyMaxActiveClusters)(
        numClusters, func, config)


cdef CUresult _cuTexRefSetArray(CUtexref hTexRef, CUarray hArray, unsigned int Flags) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuTexRefSetArray
    _check_or_init_driver()
    if __cuTexRefSetArray == NULL:
        with gil:
            raise FunctionNotFoundError("function cuTexRefSetArray is not found")
    return (<CUresult (*)(CUtexref, CUarray, unsigned int) noexcept nogil>__cuTexRefSetArray)(
        hTexRef, hArray, Flags)


cdef CUresult _cuTexRefSetMipmappedArray(CUtexref hTexRef, CUmipmappedArray hMipmappedArray, unsigned int Flags) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuTexRefSetMipmappedArray
    _check_or_init_driver()
    if __cuTexRefSetMipmappedArray == NULL:
        with gil:
            raise FunctionNotFoundError("function cuTexRefSetMipmappedArray is not found")
    return (<CUresult (*)(CUtexref, CUmipmappedArray, unsigned int) noexcept nogil>__cuTexRefSetMipmappedArray)(
        hTexRef, hMipmappedArray, Flags)


cdef CUresult _cuTexRefSetAddress_v2(size_t* ByteOffset, CUtexref hTexRef, CUdeviceptr dptr, size_t bytes) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuTexRefSetAddress_v2
    _check_or_init_driver()
    if __cuTexRefSetAddress_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuTexRefSetAddress_v2 is not found")
    return (<CUresult (*)(size_t*, CUtexref, CUdeviceptr, size_t) noexcept nogil>__cuTexRefSetAddress_v2)(
        ByteOffset, hTexRef, dptr, bytes)


cdef CUresult _cuTexRefSetAddress2D_v3(CUtexref hTexRef, const CUDA_ARRAY_DESCRIPTOR* desc, CUdeviceptr dptr, size_t Pitch) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuTexRefSetAddress2D_v3
    _check_or_init_driver()
    if __cuTexRefSetAddress2D_v3 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuTexRefSetAddress2D_v3 is not found")
    return (<CUresult (*)(CUtexref, const CUDA_ARRAY_DESCRIPTOR*, CUdeviceptr, size_t) noexcept nogil>__cuTexRefSetAddress2D_v3)(
        hTexRef, desc, dptr, Pitch)


cdef CUresult _cuTexRefSetFormat(CUtexref hTexRef, CUarray_format fmt, int NumPackedComponents) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuTexRefSetFormat
    _check_or_init_driver()
    if __cuTexRefSetFormat == NULL:
        with gil:
            raise FunctionNotFoundError("function cuTexRefSetFormat is not found")
    return (<CUresult (*)(CUtexref, CUarray_format, int) noexcept nogil>__cuTexRefSetFormat)(
        hTexRef, fmt, NumPackedComponents)


cdef CUresult _cuTexRefSetAddressMode(CUtexref hTexRef, int dim, CUaddress_mode am) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuTexRefSetAddressMode
    _check_or_init_driver()
    if __cuTexRefSetAddressMode == NULL:
        with gil:
            raise FunctionNotFoundError("function cuTexRefSetAddressMode is not found")
    return (<CUresult (*)(CUtexref, int, CUaddress_mode) noexcept nogil>__cuTexRefSetAddressMode)(
        hTexRef, dim, am)


cdef CUresult _cuTexRefSetFilterMode(CUtexref hTexRef, CUfilter_mode fm) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuTexRefSetFilterMode
    _check_or_init_driver()
    if __cuTexRefSetFilterMode == NULL:
        with gil:
            raise FunctionNotFoundError("function cuTexRefSetFilterMode is not found")
    return (<CUresult (*)(CUtexref, CUfilter_mode) noexcept nogil>__cuTexRefSetFilterMode)(
        hTexRef, fm)


cdef CUresult _cuTexRefSetMipmapFilterMode(CUtexref hTexRef, CUfilter_mode fm) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuTexRefSetMipmapFilterMode
    _check_or_init_driver()
    if __cuTexRefSetMipmapFilterMode == NULL:
        with gil:
            raise FunctionNotFoundError("function cuTexRefSetMipmapFilterMode is not found")
    return (<CUresult (*)(CUtexref, CUfilter_mode) noexcept nogil>__cuTexRefSetMipmapFilterMode)(
        hTexRef, fm)


cdef CUresult _cuTexRefSetMipmapLevelBias(CUtexref hTexRef, float bias) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuTexRefSetMipmapLevelBias
    _check_or_init_driver()
    if __cuTexRefSetMipmapLevelBias == NULL:
        with gil:
            raise FunctionNotFoundError("function cuTexRefSetMipmapLevelBias is not found")
    return (<CUresult (*)(CUtexref, float) noexcept nogil>__cuTexRefSetMipmapLevelBias)(
        hTexRef, bias)


cdef CUresult _cuTexRefSetMipmapLevelClamp(CUtexref hTexRef, float minMipmapLevelClamp, float maxMipmapLevelClamp) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuTexRefSetMipmapLevelClamp
    _check_or_init_driver()
    if __cuTexRefSetMipmapLevelClamp == NULL:
        with gil:
            raise FunctionNotFoundError("function cuTexRefSetMipmapLevelClamp is not found")
    return (<CUresult (*)(CUtexref, float, float) noexcept nogil>__cuTexRefSetMipmapLevelClamp)(
        hTexRef, minMipmapLevelClamp, maxMipmapLevelClamp)


cdef CUresult _cuTexRefSetMaxAnisotropy(CUtexref hTexRef, unsigned int maxAniso) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuTexRefSetMaxAnisotropy
    _check_or_init_driver()
    if __cuTexRefSetMaxAnisotropy == NULL:
        with gil:
            raise FunctionNotFoundError("function cuTexRefSetMaxAnisotropy is not found")
    return (<CUresult (*)(CUtexref, unsigned int) noexcept nogil>__cuTexRefSetMaxAnisotropy)(
        hTexRef, maxAniso)


cdef CUresult _cuTexRefSetBorderColor(CUtexref hTexRef, float* pBorderColor) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuTexRefSetBorderColor
    _check_or_init_driver()
    if __cuTexRefSetBorderColor == NULL:
        with gil:
            raise FunctionNotFoundError("function cuTexRefSetBorderColor is not found")
    return (<CUresult (*)(CUtexref, float*) noexcept nogil>__cuTexRefSetBorderColor)(
        hTexRef, pBorderColor)


cdef CUresult _cuTexRefSetFlags(CUtexref hTexRef, unsigned int Flags) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuTexRefSetFlags
    _check_or_init_driver()
    if __cuTexRefSetFlags == NULL:
        with gil:
            raise FunctionNotFoundError("function cuTexRefSetFlags is not found")
    return (<CUresult (*)(CUtexref, unsigned int) noexcept nogil>__cuTexRefSetFlags)(
        hTexRef, Flags)


cdef CUresult _cuTexRefGetAddress_v2(CUdeviceptr* pdptr, CUtexref hTexRef) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuTexRefGetAddress_v2
    _check_or_init_driver()
    if __cuTexRefGetAddress_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuTexRefGetAddress_v2 is not found")
    return (<CUresult (*)(CUdeviceptr*, CUtexref) noexcept nogil>__cuTexRefGetAddress_v2)(
        pdptr, hTexRef)


cdef CUresult _cuTexRefGetArray(CUarray* phArray, CUtexref hTexRef) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuTexRefGetArray
    _check_or_init_driver()
    if __cuTexRefGetArray == NULL:
        with gil:
            raise FunctionNotFoundError("function cuTexRefGetArray is not found")
    return (<CUresult (*)(CUarray*, CUtexref) noexcept nogil>__cuTexRefGetArray)(
        phArray, hTexRef)


cdef CUresult _cuTexRefGetMipmappedArray(CUmipmappedArray* phMipmappedArray, CUtexref hTexRef) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuTexRefGetMipmappedArray
    _check_or_init_driver()
    if __cuTexRefGetMipmappedArray == NULL:
        with gil:
            raise FunctionNotFoundError("function cuTexRefGetMipmappedArray is not found")
    return (<CUresult (*)(CUmipmappedArray*, CUtexref) noexcept nogil>__cuTexRefGetMipmappedArray)(
        phMipmappedArray, hTexRef)


cdef CUresult _cuTexRefGetAddressMode(CUaddress_mode* pam, CUtexref hTexRef, int dim) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuTexRefGetAddressMode
    _check_or_init_driver()
    if __cuTexRefGetAddressMode == NULL:
        with gil:
            raise FunctionNotFoundError("function cuTexRefGetAddressMode is not found")
    return (<CUresult (*)(CUaddress_mode*, CUtexref, int) noexcept nogil>__cuTexRefGetAddressMode)(
        pam, hTexRef, dim)


cdef CUresult _cuTexRefGetFilterMode(CUfilter_mode* pfm, CUtexref hTexRef) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuTexRefGetFilterMode
    _check_or_init_driver()
    if __cuTexRefGetFilterMode == NULL:
        with gil:
            raise FunctionNotFoundError("function cuTexRefGetFilterMode is not found")
    return (<CUresult (*)(CUfilter_mode*, CUtexref) noexcept nogil>__cuTexRefGetFilterMode)(
        pfm, hTexRef)


cdef CUresult _cuTexRefGetFormat(CUarray_format* pFormat, int* pNumChannels, CUtexref hTexRef) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuTexRefGetFormat
    _check_or_init_driver()
    if __cuTexRefGetFormat == NULL:
        with gil:
            raise FunctionNotFoundError("function cuTexRefGetFormat is not found")
    return (<CUresult (*)(CUarray_format*, int*, CUtexref) noexcept nogil>__cuTexRefGetFormat)(
        pFormat, pNumChannels, hTexRef)


cdef CUresult _cuTexRefGetMipmapFilterMode(CUfilter_mode* pfm, CUtexref hTexRef) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuTexRefGetMipmapFilterMode
    _check_or_init_driver()
    if __cuTexRefGetMipmapFilterMode == NULL:
        with gil:
            raise FunctionNotFoundError("function cuTexRefGetMipmapFilterMode is not found")
    return (<CUresult (*)(CUfilter_mode*, CUtexref) noexcept nogil>__cuTexRefGetMipmapFilterMode)(
        pfm, hTexRef)


cdef CUresult _cuTexRefGetMipmapLevelBias(float* pbias, CUtexref hTexRef) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuTexRefGetMipmapLevelBias
    _check_or_init_driver()
    if __cuTexRefGetMipmapLevelBias == NULL:
        with gil:
            raise FunctionNotFoundError("function cuTexRefGetMipmapLevelBias is not found")
    return (<CUresult (*)(float*, CUtexref) noexcept nogil>__cuTexRefGetMipmapLevelBias)(
        pbias, hTexRef)


cdef CUresult _cuTexRefGetMipmapLevelClamp(float* pminMipmapLevelClamp, float* pmaxMipmapLevelClamp, CUtexref hTexRef) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuTexRefGetMipmapLevelClamp
    _check_or_init_driver()
    if __cuTexRefGetMipmapLevelClamp == NULL:
        with gil:
            raise FunctionNotFoundError("function cuTexRefGetMipmapLevelClamp is not found")
    return (<CUresult (*)(float*, float*, CUtexref) noexcept nogil>__cuTexRefGetMipmapLevelClamp)(
        pminMipmapLevelClamp, pmaxMipmapLevelClamp, hTexRef)


cdef CUresult _cuTexRefGetMaxAnisotropy(int* pmaxAniso, CUtexref hTexRef) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuTexRefGetMaxAnisotropy
    _check_or_init_driver()
    if __cuTexRefGetMaxAnisotropy == NULL:
        with gil:
            raise FunctionNotFoundError("function cuTexRefGetMaxAnisotropy is not found")
    return (<CUresult (*)(int*, CUtexref) noexcept nogil>__cuTexRefGetMaxAnisotropy)(
        pmaxAniso, hTexRef)


cdef CUresult _cuTexRefGetBorderColor(float* pBorderColor, CUtexref hTexRef) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuTexRefGetBorderColor
    _check_or_init_driver()
    if __cuTexRefGetBorderColor == NULL:
        with gil:
            raise FunctionNotFoundError("function cuTexRefGetBorderColor is not found")
    return (<CUresult (*)(float*, CUtexref) noexcept nogil>__cuTexRefGetBorderColor)(
        pBorderColor, hTexRef)


cdef CUresult _cuTexRefGetFlags(unsigned int* pFlags, CUtexref hTexRef) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuTexRefGetFlags
    _check_or_init_driver()
    if __cuTexRefGetFlags == NULL:
        with gil:
            raise FunctionNotFoundError("function cuTexRefGetFlags is not found")
    return (<CUresult (*)(unsigned int*, CUtexref) noexcept nogil>__cuTexRefGetFlags)(
        pFlags, hTexRef)


cdef CUresult _cuTexRefCreate(CUtexref* pTexRef) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuTexRefCreate
    _check_or_init_driver()
    if __cuTexRefCreate == NULL:
        with gil:
            raise FunctionNotFoundError("function cuTexRefCreate is not found")
    return (<CUresult (*)(CUtexref*) noexcept nogil>__cuTexRefCreate)(
        pTexRef)


cdef CUresult _cuTexRefDestroy(CUtexref hTexRef) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuTexRefDestroy
    _check_or_init_driver()
    if __cuTexRefDestroy == NULL:
        with gil:
            raise FunctionNotFoundError("function cuTexRefDestroy is not found")
    return (<CUresult (*)(CUtexref) noexcept nogil>__cuTexRefDestroy)(
        hTexRef)


cdef CUresult _cuSurfRefSetArray(CUsurfref hSurfRef, CUarray hArray, unsigned int Flags) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuSurfRefSetArray
    _check_or_init_driver()
    if __cuSurfRefSetArray == NULL:
        with gil:
            raise FunctionNotFoundError("function cuSurfRefSetArray is not found")
    return (<CUresult (*)(CUsurfref, CUarray, unsigned int) noexcept nogil>__cuSurfRefSetArray)(
        hSurfRef, hArray, Flags)


cdef CUresult _cuSurfRefGetArray(CUarray* phArray, CUsurfref hSurfRef) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuSurfRefGetArray
    _check_or_init_driver()
    if __cuSurfRefGetArray == NULL:
        with gil:
            raise FunctionNotFoundError("function cuSurfRefGetArray is not found")
    return (<CUresult (*)(CUarray*, CUsurfref) noexcept nogil>__cuSurfRefGetArray)(
        phArray, hSurfRef)


cdef CUresult _cuTexObjectCreate(CUtexObject* pTexObject, const CUDA_RESOURCE_DESC* pResDesc, const CUDA_TEXTURE_DESC* pTexDesc, const CUDA_RESOURCE_VIEW_DESC* pResViewDesc) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuTexObjectCreate
    _check_or_init_driver()
    if __cuTexObjectCreate == NULL:
        with gil:
            raise FunctionNotFoundError("function cuTexObjectCreate is not found")
    return (<CUresult (*)(CUtexObject*, const CUDA_RESOURCE_DESC*, const CUDA_TEXTURE_DESC*, const CUDA_RESOURCE_VIEW_DESC*) noexcept nogil>__cuTexObjectCreate)(
        pTexObject, pResDesc, pTexDesc, pResViewDesc)


cdef CUresult _cuTexObjectDestroy(CUtexObject texObject) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuTexObjectDestroy
    _check_or_init_driver()
    if __cuTexObjectDestroy == NULL:
        with gil:
            raise FunctionNotFoundError("function cuTexObjectDestroy is not found")
    return (<CUresult (*)(CUtexObject) noexcept nogil>__cuTexObjectDestroy)(
        texObject)


cdef CUresult _cuTexObjectGetResourceDesc(CUDA_RESOURCE_DESC* pResDesc, CUtexObject texObject) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuTexObjectGetResourceDesc
    _check_or_init_driver()
    if __cuTexObjectGetResourceDesc == NULL:
        with gil:
            raise FunctionNotFoundError("function cuTexObjectGetResourceDesc is not found")
    return (<CUresult (*)(CUDA_RESOURCE_DESC*, CUtexObject) noexcept nogil>__cuTexObjectGetResourceDesc)(
        pResDesc, texObject)


cdef CUresult _cuTexObjectGetTextureDesc(CUDA_TEXTURE_DESC* pTexDesc, CUtexObject texObject) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuTexObjectGetTextureDesc
    _check_or_init_driver()
    if __cuTexObjectGetTextureDesc == NULL:
        with gil:
            raise FunctionNotFoundError("function cuTexObjectGetTextureDesc is not found")
    return (<CUresult (*)(CUDA_TEXTURE_DESC*, CUtexObject) noexcept nogil>__cuTexObjectGetTextureDesc)(
        pTexDesc, texObject)


cdef CUresult _cuTexObjectGetResourceViewDesc(CUDA_RESOURCE_VIEW_DESC* pResViewDesc, CUtexObject texObject) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuTexObjectGetResourceViewDesc
    _check_or_init_driver()
    if __cuTexObjectGetResourceViewDesc == NULL:
        with gil:
            raise FunctionNotFoundError("function cuTexObjectGetResourceViewDesc is not found")
    return (<CUresult (*)(CUDA_RESOURCE_VIEW_DESC*, CUtexObject) noexcept nogil>__cuTexObjectGetResourceViewDesc)(
        pResViewDesc, texObject)


cdef CUresult _cuSurfObjectCreate(CUsurfObject* pSurfObject, const CUDA_RESOURCE_DESC* pResDesc) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuSurfObjectCreate
    _check_or_init_driver()
    if __cuSurfObjectCreate == NULL:
        with gil:
            raise FunctionNotFoundError("function cuSurfObjectCreate is not found")
    return (<CUresult (*)(CUsurfObject*, const CUDA_RESOURCE_DESC*) noexcept nogil>__cuSurfObjectCreate)(
        pSurfObject, pResDesc)


cdef CUresult _cuSurfObjectDestroy(CUsurfObject surfObject) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuSurfObjectDestroy
    _check_or_init_driver()
    if __cuSurfObjectDestroy == NULL:
        with gil:
            raise FunctionNotFoundError("function cuSurfObjectDestroy is not found")
    return (<CUresult (*)(CUsurfObject) noexcept nogil>__cuSurfObjectDestroy)(
        surfObject)


cdef CUresult _cuSurfObjectGetResourceDesc(CUDA_RESOURCE_DESC* pResDesc, CUsurfObject surfObject) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuSurfObjectGetResourceDesc
    _check_or_init_driver()
    if __cuSurfObjectGetResourceDesc == NULL:
        with gil:
            raise FunctionNotFoundError("function cuSurfObjectGetResourceDesc is not found")
    return (<CUresult (*)(CUDA_RESOURCE_DESC*, CUsurfObject) noexcept nogil>__cuSurfObjectGetResourceDesc)(
        pResDesc, surfObject)


cdef CUresult _cuTensorMapEncodeTiled(CUtensorMap* tensorMap, CUtensorMapDataType tensorDataType, cuuint32_t tensorRank, void* globalAddress, const cuuint64_t* globalDim, const cuuint64_t* globalStrides, const cuuint32_t* boxDim, const cuuint32_t* elementStrides, CUtensorMapInterleave interleave, CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2Promotion, CUtensorMapFloatOOBfill oobFill) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuTensorMapEncodeTiled
    _check_or_init_driver()
    if __cuTensorMapEncodeTiled == NULL:
        with gil:
            raise FunctionNotFoundError("function cuTensorMapEncodeTiled is not found")
    return (<CUresult (*)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*, const cuuint64_t*, const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave, CUtensorMapSwizzle, CUtensorMapL2promotion, CUtensorMapFloatOOBfill) noexcept nogil>__cuTensorMapEncodeTiled)(
        tensorMap, tensorDataType, tensorRank, globalAddress, globalDim, globalStrides, boxDim, elementStrides, interleave, swizzle, l2Promotion, oobFill)


cdef CUresult _cuTensorMapEncodeIm2col(CUtensorMap* tensorMap, CUtensorMapDataType tensorDataType, cuuint32_t tensorRank, void* globalAddress, const cuuint64_t* globalDim, const cuuint64_t* globalStrides, const int* pixelBoxLowerCorner, const int* pixelBoxUpperCorner, cuuint32_t channelsPerPixel, cuuint32_t pixelsPerColumn, const cuuint32_t* elementStrides, CUtensorMapInterleave interleave, CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2Promotion, CUtensorMapFloatOOBfill oobFill) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuTensorMapEncodeIm2col
    _check_or_init_driver()
    if __cuTensorMapEncodeIm2col == NULL:
        with gil:
            raise FunctionNotFoundError("function cuTensorMapEncodeIm2col is not found")
    return (<CUresult (*)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*, const cuuint64_t*, const int*, const int*, cuuint32_t, cuuint32_t, const cuuint32_t*, CUtensorMapInterleave, CUtensorMapSwizzle, CUtensorMapL2promotion, CUtensorMapFloatOOBfill) noexcept nogil>__cuTensorMapEncodeIm2col)(
        tensorMap, tensorDataType, tensorRank, globalAddress, globalDim, globalStrides, pixelBoxLowerCorner, pixelBoxUpperCorner, channelsPerPixel, pixelsPerColumn, elementStrides, interleave, swizzle, l2Promotion, oobFill)


cdef CUresult _cuTensorMapEncodeIm2colWide(CUtensorMap* tensorMap, CUtensorMapDataType tensorDataType, cuuint32_t tensorRank, void* globalAddress, const cuuint64_t* globalDim, const cuuint64_t* globalStrides, int pixelBoxLowerCornerWidth, int pixelBoxUpperCornerWidth, cuuint32_t channelsPerPixel, cuuint32_t pixelsPerColumn, const cuuint32_t* elementStrides, CUtensorMapInterleave interleave, CUtensorMapIm2ColWideMode mode, CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2Promotion, CUtensorMapFloatOOBfill oobFill) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuTensorMapEncodeIm2colWide
    _check_or_init_driver()
    if __cuTensorMapEncodeIm2colWide == NULL:
        with gil:
            raise FunctionNotFoundError("function cuTensorMapEncodeIm2colWide is not found")
    return (<CUresult (*)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*, const cuuint64_t*, int, int, cuuint32_t, cuuint32_t, const cuuint32_t*, CUtensorMapInterleave, CUtensorMapIm2ColWideMode, CUtensorMapSwizzle, CUtensorMapL2promotion, CUtensorMapFloatOOBfill) noexcept nogil>__cuTensorMapEncodeIm2colWide)(
        tensorMap, tensorDataType, tensorRank, globalAddress, globalDim, globalStrides, pixelBoxLowerCornerWidth, pixelBoxUpperCornerWidth, channelsPerPixel, pixelsPerColumn, elementStrides, interleave, mode, swizzle, l2Promotion, oobFill)


cdef CUresult _cuTensorMapReplaceAddress(CUtensorMap* tensorMap, void* globalAddress) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuTensorMapReplaceAddress
    _check_or_init_driver()
    if __cuTensorMapReplaceAddress == NULL:
        with gil:
            raise FunctionNotFoundError("function cuTensorMapReplaceAddress is not found")
    return (<CUresult (*)(CUtensorMap*, void*) noexcept nogil>__cuTensorMapReplaceAddress)(
        tensorMap, globalAddress)


cdef CUresult _cuDeviceCanAccessPeer(int* canAccessPeer, CUdevice dev, CUdevice peerDev) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuDeviceCanAccessPeer
    _check_or_init_driver()
    if __cuDeviceCanAccessPeer == NULL:
        with gil:
            raise FunctionNotFoundError("function cuDeviceCanAccessPeer is not found")
    return (<CUresult (*)(int*, CUdevice, CUdevice) noexcept nogil>__cuDeviceCanAccessPeer)(
        canAccessPeer, dev, peerDev)


cdef CUresult _cuCtxEnablePeerAccess(CUcontext peerContext, unsigned int Flags) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuCtxEnablePeerAccess
    _check_or_init_driver()
    if __cuCtxEnablePeerAccess == NULL:
        with gil:
            raise FunctionNotFoundError("function cuCtxEnablePeerAccess is not found")
    return (<CUresult (*)(CUcontext, unsigned int) noexcept nogil>__cuCtxEnablePeerAccess)(
        peerContext, Flags)


cdef CUresult _cuCtxDisablePeerAccess(CUcontext peerContext) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuCtxDisablePeerAccess
    _check_or_init_driver()
    if __cuCtxDisablePeerAccess == NULL:
        with gil:
            raise FunctionNotFoundError("function cuCtxDisablePeerAccess is not found")
    return (<CUresult (*)(CUcontext) noexcept nogil>__cuCtxDisablePeerAccess)(
        peerContext)


cdef CUresult _cuDeviceGetP2PAttribute(int* value, CUdevice_P2PAttribute attrib, CUdevice srcDevice, CUdevice dstDevice) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuDeviceGetP2PAttribute
    _check_or_init_driver()
    if __cuDeviceGetP2PAttribute == NULL:
        with gil:
            raise FunctionNotFoundError("function cuDeviceGetP2PAttribute is not found")
    return (<CUresult (*)(int*, CUdevice_P2PAttribute, CUdevice, CUdevice) noexcept nogil>__cuDeviceGetP2PAttribute)(
        value, attrib, srcDevice, dstDevice)


cdef CUresult _cuGraphicsUnregisterResource(CUgraphicsResource resource) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuGraphicsUnregisterResource
    _check_or_init_driver()
    if __cuGraphicsUnregisterResource == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphicsUnregisterResource is not found")
    return (<CUresult (*)(CUgraphicsResource) noexcept nogil>__cuGraphicsUnregisterResource)(
        resource)


cdef CUresult _cuGraphicsSubResourceGetMappedArray(CUarray* pArray, CUgraphicsResource resource, unsigned int arrayIndex, unsigned int mipLevel) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuGraphicsSubResourceGetMappedArray
    _check_or_init_driver()
    if __cuGraphicsSubResourceGetMappedArray == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphicsSubResourceGetMappedArray is not found")
    return (<CUresult (*)(CUarray*, CUgraphicsResource, unsigned int, unsigned int) noexcept nogil>__cuGraphicsSubResourceGetMappedArray)(
        pArray, resource, arrayIndex, mipLevel)


cdef CUresult _cuGraphicsResourceGetMappedMipmappedArray(CUmipmappedArray* pMipmappedArray, CUgraphicsResource resource) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuGraphicsResourceGetMappedMipmappedArray
    _check_or_init_driver()
    if __cuGraphicsResourceGetMappedMipmappedArray == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphicsResourceGetMappedMipmappedArray is not found")
    return (<CUresult (*)(CUmipmappedArray*, CUgraphicsResource) noexcept nogil>__cuGraphicsResourceGetMappedMipmappedArray)(
        pMipmappedArray, resource)


cdef CUresult _cuGraphicsResourceGetMappedPointer_v2(CUdeviceptr* pDevPtr, size_t* pSize, CUgraphicsResource resource) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuGraphicsResourceGetMappedPointer_v2
    _check_or_init_driver()
    if __cuGraphicsResourceGetMappedPointer_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphicsResourceGetMappedPointer_v2 is not found")
    return (<CUresult (*)(CUdeviceptr*, size_t*, CUgraphicsResource) noexcept nogil>__cuGraphicsResourceGetMappedPointer_v2)(
        pDevPtr, pSize, resource)


cdef CUresult _cuGraphicsResourceSetMapFlags_v2(CUgraphicsResource resource, unsigned int flags) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuGraphicsResourceSetMapFlags_v2
    _check_or_init_driver()
    if __cuGraphicsResourceSetMapFlags_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphicsResourceSetMapFlags_v2 is not found")
    return (<CUresult (*)(CUgraphicsResource, unsigned int) noexcept nogil>__cuGraphicsResourceSetMapFlags_v2)(
        resource, flags)


cdef CUresult _cuGraphicsMapResources(unsigned int count, CUgraphicsResource* resources, CUstream hStream) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuGraphicsMapResources
    _check_or_init_driver()
    if __cuGraphicsMapResources == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphicsMapResources is not found")
    return (<CUresult (*)(unsigned int, CUgraphicsResource*, CUstream) noexcept nogil>__cuGraphicsMapResources)(
        count, resources, hStream)


cdef CUresult _cuGraphicsUnmapResources(unsigned int count, CUgraphicsResource* resources, CUstream hStream) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuGraphicsUnmapResources
    _check_or_init_driver()
    if __cuGraphicsUnmapResources == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphicsUnmapResources is not found")
    return (<CUresult (*)(unsigned int, CUgraphicsResource*, CUstream) noexcept nogil>__cuGraphicsUnmapResources)(
        count, resources, hStream)


cdef CUresult _cuGetProcAddress_v2(const char* symbol, void** pfn, int cudaVersion, cuuint64_t flags, CUdriverProcAddressQueryResult* symbolStatus) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuGetProcAddress_v2
    _check_or_init_driver()
    if __cuGetProcAddress_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGetProcAddress_v2 is not found")
    return (<CUresult (*)(const char*, void**, int, cuuint64_t, CUdriverProcAddressQueryResult*) noexcept nogil>__cuGetProcAddress_v2)(
        symbol, pfn, cudaVersion, flags, symbolStatus)


cdef CUresult _cuCoredumpGetAttribute(CUcoredumpSettings attrib, void* value, size_t* size) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuCoredumpGetAttribute
    _check_or_init_driver()
    if __cuCoredumpGetAttribute == NULL:
        with gil:
            raise FunctionNotFoundError("function cuCoredumpGetAttribute is not found")
    return (<CUresult (*)(CUcoredumpSettings, void*, size_t*) noexcept nogil>__cuCoredumpGetAttribute)(
        attrib, value, size)


cdef CUresult _cuCoredumpGetAttributeGlobal(CUcoredumpSettings attrib, void* value, size_t* size) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuCoredumpGetAttributeGlobal
    _check_or_init_driver()
    if __cuCoredumpGetAttributeGlobal == NULL:
        with gil:
            raise FunctionNotFoundError("function cuCoredumpGetAttributeGlobal is not found")
    return (<CUresult (*)(CUcoredumpSettings, void*, size_t*) noexcept nogil>__cuCoredumpGetAttributeGlobal)(
        attrib, value, size)


cdef CUresult _cuCoredumpSetAttribute(CUcoredumpSettings attrib, void* value, size_t* size) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuCoredumpSetAttribute
    _check_or_init_driver()
    if __cuCoredumpSetAttribute == NULL:
        with gil:
            raise FunctionNotFoundError("function cuCoredumpSetAttribute is not found")
    return (<CUresult (*)(CUcoredumpSettings, void*, size_t*) noexcept nogil>__cuCoredumpSetAttribute)(
        attrib, value, size)


cdef CUresult _cuCoredumpSetAttributeGlobal(CUcoredumpSettings attrib, void* value, size_t* size) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuCoredumpSetAttributeGlobal
    _check_or_init_driver()
    if __cuCoredumpSetAttributeGlobal == NULL:
        with gil:
            raise FunctionNotFoundError("function cuCoredumpSetAttributeGlobal is not found")
    return (<CUresult (*)(CUcoredumpSettings, void*, size_t*) noexcept nogil>__cuCoredumpSetAttributeGlobal)(
        attrib, value, size)


cdef CUresult _cuGetExportTable(const void** ppExportTable, const CUuuid* pExportTableId) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuGetExportTable
    _check_or_init_driver()
    if __cuGetExportTable == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGetExportTable is not found")
    return (<CUresult (*)(const void**, const CUuuid*) noexcept nogil>__cuGetExportTable)(
        ppExportTable, pExportTableId)


cdef CUresult _cuGreenCtxCreate(CUgreenCtx* phCtx, CUdevResourceDesc desc, CUdevice dev, unsigned int flags) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuGreenCtxCreate
    _check_or_init_driver()
    if __cuGreenCtxCreate == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGreenCtxCreate is not found")
    return (<CUresult (*)(CUgreenCtx*, CUdevResourceDesc, CUdevice, unsigned int) noexcept nogil>__cuGreenCtxCreate)(
        phCtx, desc, dev, flags)


cdef CUresult _cuGreenCtxDestroy(CUgreenCtx hCtx) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuGreenCtxDestroy
    _check_or_init_driver()
    if __cuGreenCtxDestroy == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGreenCtxDestroy is not found")
    return (<CUresult (*)(CUgreenCtx) noexcept nogil>__cuGreenCtxDestroy)(
        hCtx)


cdef CUresult _cuCtxFromGreenCtx(CUcontext* pContext, CUgreenCtx hCtx) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuCtxFromGreenCtx
    _check_or_init_driver()
    if __cuCtxFromGreenCtx == NULL:
        with gil:
            raise FunctionNotFoundError("function cuCtxFromGreenCtx is not found")
    return (<CUresult (*)(CUcontext*, CUgreenCtx) noexcept nogil>__cuCtxFromGreenCtx)(
        pContext, hCtx)


cdef CUresult _cuDeviceGetDevResource(CUdevice device, CUdevResource* resource, CUdevResourceType type) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuDeviceGetDevResource
    _check_or_init_driver()
    if __cuDeviceGetDevResource == NULL:
        with gil:
            raise FunctionNotFoundError("function cuDeviceGetDevResource is not found")
    return (<CUresult (*)(CUdevice, CUdevResource*, CUdevResourceType) noexcept nogil>__cuDeviceGetDevResource)(
        device, resource, type)


cdef CUresult _cuCtxGetDevResource(CUcontext hCtx, CUdevResource* resource, CUdevResourceType type) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuCtxGetDevResource
    _check_or_init_driver()
    if __cuCtxGetDevResource == NULL:
        with gil:
            raise FunctionNotFoundError("function cuCtxGetDevResource is not found")
    return (<CUresult (*)(CUcontext, CUdevResource*, CUdevResourceType) noexcept nogil>__cuCtxGetDevResource)(
        hCtx, resource, type)


cdef CUresult _cuGreenCtxGetDevResource(CUgreenCtx hCtx, CUdevResource* resource, CUdevResourceType type) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuGreenCtxGetDevResource
    _check_or_init_driver()
    if __cuGreenCtxGetDevResource == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGreenCtxGetDevResource is not found")
    return (<CUresult (*)(CUgreenCtx, CUdevResource*, CUdevResourceType) noexcept nogil>__cuGreenCtxGetDevResource)(
        hCtx, resource, type)


cdef CUresult _cuDevSmResourceSplitByCount(CUdevResource* result, unsigned int* nbGroups, const CUdevResource* input, CUdevResource* remainder, unsigned int flags, unsigned int minCount) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuDevSmResourceSplitByCount
    _check_or_init_driver()
    if __cuDevSmResourceSplitByCount == NULL:
        with gil:
            raise FunctionNotFoundError("function cuDevSmResourceSplitByCount is not found")
    return (<CUresult (*)(CUdevResource*, unsigned int*, const CUdevResource*, CUdevResource*, unsigned int, unsigned int) noexcept nogil>__cuDevSmResourceSplitByCount)(
        result, nbGroups, input, remainder, flags, minCount)


cdef CUresult _cuDevResourceGenerateDesc(CUdevResourceDesc* phDesc, CUdevResource* resources, unsigned int nbResources) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuDevResourceGenerateDesc
    _check_or_init_driver()
    if __cuDevResourceGenerateDesc == NULL:
        with gil:
            raise FunctionNotFoundError("function cuDevResourceGenerateDesc is not found")
    return (<CUresult (*)(CUdevResourceDesc*, CUdevResource*, unsigned int) noexcept nogil>__cuDevResourceGenerateDesc)(
        phDesc, resources, nbResources)


cdef CUresult _cuGreenCtxRecordEvent(CUgreenCtx hCtx, CUevent hEvent) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuGreenCtxRecordEvent
    _check_or_init_driver()
    if __cuGreenCtxRecordEvent == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGreenCtxRecordEvent is not found")
    return (<CUresult (*)(CUgreenCtx, CUevent) noexcept nogil>__cuGreenCtxRecordEvent)(
        hCtx, hEvent)


cdef CUresult _cuGreenCtxWaitEvent(CUgreenCtx hCtx, CUevent hEvent) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuGreenCtxWaitEvent
    _check_or_init_driver()
    if __cuGreenCtxWaitEvent == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGreenCtxWaitEvent is not found")
    return (<CUresult (*)(CUgreenCtx, CUevent) noexcept nogil>__cuGreenCtxWaitEvent)(
        hCtx, hEvent)


cdef CUresult _cuStreamGetGreenCtx(CUstream hStream, CUgreenCtx* phCtx) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuStreamGetGreenCtx
    _check_or_init_driver()
    if __cuStreamGetGreenCtx == NULL:
        with gil:
            raise FunctionNotFoundError("function cuStreamGetGreenCtx is not found")
    return (<CUresult (*)(CUstream, CUgreenCtx*) noexcept nogil>__cuStreamGetGreenCtx)(
        hStream, phCtx)


cdef CUresult _cuGreenCtxStreamCreate(CUstream* phStream, CUgreenCtx greenCtx, unsigned int flags, int priority) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuGreenCtxStreamCreate
    _check_or_init_driver()
    if __cuGreenCtxStreamCreate == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGreenCtxStreamCreate is not found")
    return (<CUresult (*)(CUstream*, CUgreenCtx, unsigned int, int) noexcept nogil>__cuGreenCtxStreamCreate)(
        phStream, greenCtx, flags, priority)


cdef CUresult _cuLogsRegisterCallback(CUlogsCallback callbackFunc, void* userData, CUlogsCallbackHandle* callback_out) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuLogsRegisterCallback
    _check_or_init_driver()
    if __cuLogsRegisterCallback == NULL:
        with gil:
            raise FunctionNotFoundError("function cuLogsRegisterCallback is not found")
    return (<CUresult (*)(CUlogsCallback, void*, CUlogsCallbackHandle*) noexcept nogil>__cuLogsRegisterCallback)(
        callbackFunc, userData, callback_out)


cdef CUresult _cuLogsUnregisterCallback(CUlogsCallbackHandle callback) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuLogsUnregisterCallback
    _check_or_init_driver()
    if __cuLogsUnregisterCallback == NULL:
        with gil:
            raise FunctionNotFoundError("function cuLogsUnregisterCallback is not found")
    return (<CUresult (*)(CUlogsCallbackHandle) noexcept nogil>__cuLogsUnregisterCallback)(
        callback)


cdef CUresult _cuLogsCurrent(CUlogIterator* iterator_out, unsigned int flags) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuLogsCurrent
    _check_or_init_driver()
    if __cuLogsCurrent == NULL:
        with gil:
            raise FunctionNotFoundError("function cuLogsCurrent is not found")
    return (<CUresult (*)(CUlogIterator*, unsigned int) noexcept nogil>__cuLogsCurrent)(
        iterator_out, flags)


cdef CUresult _cuLogsDumpToFile(CUlogIterator* iterator, const char* pathToFile, unsigned int flags) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuLogsDumpToFile
    _check_or_init_driver()
    if __cuLogsDumpToFile == NULL:
        with gil:
            raise FunctionNotFoundError("function cuLogsDumpToFile is not found")
    return (<CUresult (*)(CUlogIterator*, const char*, unsigned int) noexcept nogil>__cuLogsDumpToFile)(
        iterator, pathToFile, flags)


cdef CUresult _cuLogsDumpToMemory(CUlogIterator* iterator, char* buffer, size_t* size, unsigned int flags) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuLogsDumpToMemory
    _check_or_init_driver()
    if __cuLogsDumpToMemory == NULL:
        with gil:
            raise FunctionNotFoundError("function cuLogsDumpToMemory is not found")
    return (<CUresult (*)(CUlogIterator*, char*, size_t*, unsigned int) noexcept nogil>__cuLogsDumpToMemory)(
        iterator, buffer, size, flags)


cdef CUresult _cuCheckpointProcessGetRestoreThreadId(int pid, int* tid) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuCheckpointProcessGetRestoreThreadId
    _check_or_init_driver()
    if __cuCheckpointProcessGetRestoreThreadId == NULL:
        with gil:
            raise FunctionNotFoundError("function cuCheckpointProcessGetRestoreThreadId is not found")
    return (<CUresult (*)(int, int*) noexcept nogil>__cuCheckpointProcessGetRestoreThreadId)(
        pid, tid)


cdef CUresult _cuCheckpointProcessGetState(int pid, CUprocessState* state) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuCheckpointProcessGetState
    _check_or_init_driver()
    if __cuCheckpointProcessGetState == NULL:
        with gil:
            raise FunctionNotFoundError("function cuCheckpointProcessGetState is not found")
    return (<CUresult (*)(int, CUprocessState*) noexcept nogil>__cuCheckpointProcessGetState)(
        pid, state)


cdef CUresult _cuCheckpointProcessLock(int pid, CUcheckpointLockArgs* args) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuCheckpointProcessLock
    _check_or_init_driver()
    if __cuCheckpointProcessLock == NULL:
        with gil:
            raise FunctionNotFoundError("function cuCheckpointProcessLock is not found")
    return (<CUresult (*)(int, CUcheckpointLockArgs*) noexcept nogil>__cuCheckpointProcessLock)(
        pid, args)


cdef CUresult _cuCheckpointProcessCheckpoint(int pid, CUcheckpointCheckpointArgs* args) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuCheckpointProcessCheckpoint
    _check_or_init_driver()
    if __cuCheckpointProcessCheckpoint == NULL:
        with gil:
            raise FunctionNotFoundError("function cuCheckpointProcessCheckpoint is not found")
    return (<CUresult (*)(int, CUcheckpointCheckpointArgs*) noexcept nogil>__cuCheckpointProcessCheckpoint)(
        pid, args)


cdef CUresult _cuCheckpointProcessRestore(int pid, CUcheckpointRestoreArgs* args) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuCheckpointProcessRestore
    _check_or_init_driver()
    if __cuCheckpointProcessRestore == NULL:
        with gil:
            raise FunctionNotFoundError("function cuCheckpointProcessRestore is not found")
    return (<CUresult (*)(int, CUcheckpointRestoreArgs*) noexcept nogil>__cuCheckpointProcessRestore)(
        pid, args)


cdef CUresult _cuCheckpointProcessUnlock(int pid, CUcheckpointUnlockArgs* args) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuCheckpointProcessUnlock
    _check_or_init_driver()
    if __cuCheckpointProcessUnlock == NULL:
        with gil:
            raise FunctionNotFoundError("function cuCheckpointProcessUnlock is not found")
    return (<CUresult (*)(int, CUcheckpointUnlockArgs*) noexcept nogil>__cuCheckpointProcessUnlock)(
        pid, args)


cdef CUresult _cuGraphicsEGLRegisterImage(CUgraphicsResource* pCudaResource, EGLImageKHR image, unsigned int flags) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuGraphicsEGLRegisterImage
    _check_or_init_driver()
    if __cuGraphicsEGLRegisterImage == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphicsEGLRegisterImage is not found")
    return (<CUresult (*)(CUgraphicsResource*, EGLImageKHR, unsigned int) noexcept nogil>__cuGraphicsEGLRegisterImage)(
        pCudaResource, image, flags)


cdef CUresult _cuEGLStreamConsumerConnect(CUeglStreamConnection* conn, EGLStreamKHR stream) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuEGLStreamConsumerConnect
    _check_or_init_driver()
    if __cuEGLStreamConsumerConnect == NULL:
        with gil:
            raise FunctionNotFoundError("function cuEGLStreamConsumerConnect is not found")
    return (<CUresult (*)(CUeglStreamConnection*, EGLStreamKHR) noexcept nogil>__cuEGLStreamConsumerConnect)(
        conn, stream)


cdef CUresult _cuEGLStreamConsumerConnectWithFlags(CUeglStreamConnection* conn, EGLStreamKHR stream, unsigned int flags) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuEGLStreamConsumerConnectWithFlags
    _check_or_init_driver()
    if __cuEGLStreamConsumerConnectWithFlags == NULL:
        with gil:
            raise FunctionNotFoundError("function cuEGLStreamConsumerConnectWithFlags is not found")
    return (<CUresult (*)(CUeglStreamConnection*, EGLStreamKHR, unsigned int) noexcept nogil>__cuEGLStreamConsumerConnectWithFlags)(
        conn, stream, flags)


cdef CUresult _cuEGLStreamConsumerDisconnect(CUeglStreamConnection* conn) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuEGLStreamConsumerDisconnect
    _check_or_init_driver()
    if __cuEGLStreamConsumerDisconnect == NULL:
        with gil:
            raise FunctionNotFoundError("function cuEGLStreamConsumerDisconnect is not found")
    return (<CUresult (*)(CUeglStreamConnection*) noexcept nogil>__cuEGLStreamConsumerDisconnect)(
        conn)


cdef CUresult _cuEGLStreamConsumerAcquireFrame(CUeglStreamConnection* conn, CUgraphicsResource* pCudaResource, CUstream* pStream, unsigned int timeout) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuEGLStreamConsumerAcquireFrame
    _check_or_init_driver()
    if __cuEGLStreamConsumerAcquireFrame == NULL:
        with gil:
            raise FunctionNotFoundError("function cuEGLStreamConsumerAcquireFrame is not found")
    return (<CUresult (*)(CUeglStreamConnection*, CUgraphicsResource*, CUstream*, unsigned int) noexcept nogil>__cuEGLStreamConsumerAcquireFrame)(
        conn, pCudaResource, pStream, timeout)


cdef CUresult _cuEGLStreamConsumerReleaseFrame(CUeglStreamConnection* conn, CUgraphicsResource pCudaResource, CUstream* pStream) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuEGLStreamConsumerReleaseFrame
    _check_or_init_driver()
    if __cuEGLStreamConsumerReleaseFrame == NULL:
        with gil:
            raise FunctionNotFoundError("function cuEGLStreamConsumerReleaseFrame is not found")
    return (<CUresult (*)(CUeglStreamConnection*, CUgraphicsResource, CUstream*) noexcept nogil>__cuEGLStreamConsumerReleaseFrame)(
        conn, pCudaResource, pStream)


cdef CUresult _cuEGLStreamProducerConnect(CUeglStreamConnection* conn, EGLStreamKHR stream, EGLint width, EGLint height) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuEGLStreamProducerConnect
    _check_or_init_driver()
    if __cuEGLStreamProducerConnect == NULL:
        with gil:
            raise FunctionNotFoundError("function cuEGLStreamProducerConnect is not found")
    return (<CUresult (*)(CUeglStreamConnection*, EGLStreamKHR, EGLint, EGLint) noexcept nogil>__cuEGLStreamProducerConnect)(
        conn, stream, width, height)


cdef CUresult _cuEGLStreamProducerDisconnect(CUeglStreamConnection* conn) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuEGLStreamProducerDisconnect
    _check_or_init_driver()
    if __cuEGLStreamProducerDisconnect == NULL:
        with gil:
            raise FunctionNotFoundError("function cuEGLStreamProducerDisconnect is not found")
    return (<CUresult (*)(CUeglStreamConnection*) noexcept nogil>__cuEGLStreamProducerDisconnect)(
        conn)


cdef CUresult _cuEGLStreamProducerPresentFrame(CUeglStreamConnection* conn, CUeglFrame eglframe, CUstream* pStream) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuEGLStreamProducerPresentFrame
    _check_or_init_driver()
    if __cuEGLStreamProducerPresentFrame == NULL:
        with gil:
            raise FunctionNotFoundError("function cuEGLStreamProducerPresentFrame is not found")
    return (<CUresult (*)(CUeglStreamConnection*, CUeglFrame, CUstream*) noexcept nogil>__cuEGLStreamProducerPresentFrame)(
        conn, eglframe, pStream)


cdef CUresult _cuEGLStreamProducerReturnFrame(CUeglStreamConnection* conn, CUeglFrame* eglframe, CUstream* pStream) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuEGLStreamProducerReturnFrame
    _check_or_init_driver()
    if __cuEGLStreamProducerReturnFrame == NULL:
        with gil:
            raise FunctionNotFoundError("function cuEGLStreamProducerReturnFrame is not found")
    return (<CUresult (*)(CUeglStreamConnection*, CUeglFrame*, CUstream*) noexcept nogil>__cuEGLStreamProducerReturnFrame)(
        conn, eglframe, pStream)


cdef CUresult _cuGraphicsResourceGetMappedEglFrame(CUeglFrame* eglFrame, CUgraphicsResource resource, unsigned int index, unsigned int mipLevel) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuGraphicsResourceGetMappedEglFrame
    _check_or_init_driver()
    if __cuGraphicsResourceGetMappedEglFrame == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphicsResourceGetMappedEglFrame is not found")
    return (<CUresult (*)(CUeglFrame*, CUgraphicsResource, unsigned int, unsigned int) noexcept nogil>__cuGraphicsResourceGetMappedEglFrame)(
        eglFrame, resource, index, mipLevel)


cdef CUresult _cuEventCreateFromEGLSync(CUevent* phEvent, EGLSyncKHR eglSync, unsigned int flags) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuEventCreateFromEGLSync
    _check_or_init_driver()
    if __cuEventCreateFromEGLSync == NULL:
        with gil:
            raise FunctionNotFoundError("function cuEventCreateFromEGLSync is not found")
    return (<CUresult (*)(CUevent*, EGLSyncKHR, unsigned int) noexcept nogil>__cuEventCreateFromEGLSync)(
        phEvent, eglSync, flags)


cdef CUresult _cuGraphicsGLRegisterBuffer(CUgraphicsResource* pCudaResource, GLuint buffer, unsigned int Flags) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuGraphicsGLRegisterBuffer
    _check_or_init_driver()
    if __cuGraphicsGLRegisterBuffer == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphicsGLRegisterBuffer is not found")
    return (<CUresult (*)(CUgraphicsResource*, GLuint, unsigned int) noexcept nogil>__cuGraphicsGLRegisterBuffer)(
        pCudaResource, buffer, Flags)


cdef CUresult _cuGraphicsGLRegisterImage(CUgraphicsResource* pCudaResource, GLuint image, GLenum target, unsigned int Flags) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuGraphicsGLRegisterImage
    _check_or_init_driver()
    if __cuGraphicsGLRegisterImage == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphicsGLRegisterImage is not found")
    return (<CUresult (*)(CUgraphicsResource*, GLuint, GLenum, unsigned int) noexcept nogil>__cuGraphicsGLRegisterImage)(
        pCudaResource, image, target, Flags)


cdef CUresult _cuGLGetDevices_v2(unsigned int* pCudaDeviceCount, CUdevice* pCudaDevices, unsigned int cudaDeviceCount, CUGLDeviceList deviceList) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuGLGetDevices_v2
    _check_or_init_driver()
    if __cuGLGetDevices_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGLGetDevices_v2 is not found")
    return (<CUresult (*)(unsigned int*, CUdevice*, unsigned int, CUGLDeviceList) noexcept nogil>__cuGLGetDevices_v2)(
        pCudaDeviceCount, pCudaDevices, cudaDeviceCount, deviceList)


cdef CUresult _cuGLCtxCreate_v2(CUcontext* pCtx, unsigned int Flags, CUdevice device) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuGLCtxCreate_v2
    _check_or_init_driver()
    if __cuGLCtxCreate_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGLCtxCreate_v2 is not found")
    return (<CUresult (*)(CUcontext*, unsigned int, CUdevice) noexcept nogil>__cuGLCtxCreate_v2)(
        pCtx, Flags, device)


cdef CUresult _cuGLInit() except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuGLInit
    _check_or_init_driver()
    if __cuGLInit == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGLInit is not found")
    return (<CUresult (*)() noexcept nogil>__cuGLInit)(
        )


cdef CUresult _cuGLRegisterBufferObject(GLuint buffer) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuGLRegisterBufferObject
    _check_or_init_driver()
    if __cuGLRegisterBufferObject == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGLRegisterBufferObject is not found")
    return (<CUresult (*)(GLuint) noexcept nogil>__cuGLRegisterBufferObject)(
        buffer)


cdef CUresult _cuGLMapBufferObject_v2(CUdeviceptr* dptr, size_t* size, GLuint buffer) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuGLMapBufferObject_v2
    _check_or_init_driver()
    if __cuGLMapBufferObject_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGLMapBufferObject_v2 is not found")
    return (<CUresult (*)(CUdeviceptr*, size_t*, GLuint) noexcept nogil>__cuGLMapBufferObject_v2)(
        dptr, size, buffer)


cdef CUresult _cuGLUnmapBufferObject(GLuint buffer) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuGLUnmapBufferObject
    _check_or_init_driver()
    if __cuGLUnmapBufferObject == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGLUnmapBufferObject is not found")
    return (<CUresult (*)(GLuint) noexcept nogil>__cuGLUnmapBufferObject)(
        buffer)


cdef CUresult _cuGLUnregisterBufferObject(GLuint buffer) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuGLUnregisterBufferObject
    _check_or_init_driver()
    if __cuGLUnregisterBufferObject == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGLUnregisterBufferObject is not found")
    return (<CUresult (*)(GLuint) noexcept nogil>__cuGLUnregisterBufferObject)(
        buffer)


cdef CUresult _cuGLSetBufferObjectMapFlags(GLuint buffer, unsigned int Flags) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuGLSetBufferObjectMapFlags
    _check_or_init_driver()
    if __cuGLSetBufferObjectMapFlags == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGLSetBufferObjectMapFlags is not found")
    return (<CUresult (*)(GLuint, unsigned int) noexcept nogil>__cuGLSetBufferObjectMapFlags)(
        buffer, Flags)


cdef CUresult _cuGLMapBufferObjectAsync_v2(CUdeviceptr* dptr, size_t* size, GLuint buffer, CUstream hStream) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuGLMapBufferObjectAsync_v2
    _check_or_init_driver()
    if __cuGLMapBufferObjectAsync_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGLMapBufferObjectAsync_v2 is not found")
    return (<CUresult (*)(CUdeviceptr*, size_t*, GLuint, CUstream) noexcept nogil>__cuGLMapBufferObjectAsync_v2)(
        dptr, size, buffer, hStream)


cdef CUresult _cuGLUnmapBufferObjectAsync(GLuint buffer, CUstream hStream) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuGLUnmapBufferObjectAsync
    _check_or_init_driver()
    if __cuGLUnmapBufferObjectAsync == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGLUnmapBufferObjectAsync is not found")
    return (<CUresult (*)(GLuint, CUstream) noexcept nogil>__cuGLUnmapBufferObjectAsync)(
        buffer, hStream)


cdef CUresult _cuProfilerInitialize(const char* configFile, const char* outputFile, CUoutput_mode outputMode) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuProfilerInitialize
    _check_or_init_driver()
    if __cuProfilerInitialize == NULL:
        with gil:
            raise FunctionNotFoundError("function cuProfilerInitialize is not found")
    return (<CUresult (*)(const char*, const char*, CUoutput_mode) noexcept nogil>__cuProfilerInitialize)(
        configFile, outputFile, outputMode)


cdef CUresult _cuProfilerStart() except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuProfilerStart
    _check_or_init_driver()
    if __cuProfilerStart == NULL:
        with gil:
            raise FunctionNotFoundError("function cuProfilerStart is not found")
    return (<CUresult (*)() noexcept nogil>__cuProfilerStart)(
        )


cdef CUresult _cuProfilerStop() except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuProfilerStop
    _check_or_init_driver()
    if __cuProfilerStop == NULL:
        with gil:
            raise FunctionNotFoundError("function cuProfilerStop is not found")
    return (<CUresult (*)() noexcept nogil>__cuProfilerStop)(
        )


cdef CUresult _cuVDPAUGetDevice(CUdevice* pDevice, VdpDevice vdpDevice, VdpGetProcAddress* vdpGetProcAddress) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuVDPAUGetDevice
    _check_or_init_driver()
    if __cuVDPAUGetDevice == NULL:
        with gil:
            raise FunctionNotFoundError("function cuVDPAUGetDevice is not found")
    return (<CUresult (*)(CUdevice*, VdpDevice, VdpGetProcAddress*) noexcept nogil>__cuVDPAUGetDevice)(
        pDevice, vdpDevice, vdpGetProcAddress)


cdef CUresult _cuVDPAUCtxCreate_v2(CUcontext* pCtx, unsigned int flags, CUdevice device, VdpDevice vdpDevice, VdpGetProcAddress* vdpGetProcAddress) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuVDPAUCtxCreate_v2
    _check_or_init_driver()
    if __cuVDPAUCtxCreate_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuVDPAUCtxCreate_v2 is not found")
    return (<CUresult (*)(CUcontext*, unsigned int, CUdevice, VdpDevice, VdpGetProcAddress*) noexcept nogil>__cuVDPAUCtxCreate_v2)(
        pCtx, flags, device, vdpDevice, vdpGetProcAddress)


cdef CUresult _cuGraphicsVDPAURegisterVideoSurface(CUgraphicsResource* pCudaResource, VdpVideoSurface vdpSurface, unsigned int flags) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuGraphicsVDPAURegisterVideoSurface
    _check_or_init_driver()
    if __cuGraphicsVDPAURegisterVideoSurface == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphicsVDPAURegisterVideoSurface is not found")
    return (<CUresult (*)(CUgraphicsResource*, VdpVideoSurface, unsigned int) noexcept nogil>__cuGraphicsVDPAURegisterVideoSurface)(
        pCudaResource, vdpSurface, flags)


cdef CUresult _cuGraphicsVDPAURegisterOutputSurface(CUgraphicsResource* pCudaResource, VdpOutputSurface vdpSurface, unsigned int flags) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuGraphicsVDPAURegisterOutputSurface
    _check_or_init_driver()
    if __cuGraphicsVDPAURegisterOutputSurface == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphicsVDPAURegisterOutputSurface is not found")
    return (<CUresult (*)(CUgraphicsResource*, VdpOutputSurface, unsigned int) noexcept nogil>__cuGraphicsVDPAURegisterOutputSurface)(
        pCudaResource, vdpSurface, flags)


cdef CUresult _cuDeviceGetHostAtomicCapabilities(unsigned int* capabilities, const CUatomicOperation* operations, unsigned int count, CUdevice dev) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuDeviceGetHostAtomicCapabilities
    _check_or_init_driver()
    if __cuDeviceGetHostAtomicCapabilities == NULL:
        with gil:
            raise FunctionNotFoundError("function cuDeviceGetHostAtomicCapabilities is not found")
    return (<CUresult (*)(unsigned int*, const CUatomicOperation*, unsigned int, CUdevice) noexcept nogil>__cuDeviceGetHostAtomicCapabilities)(
        capabilities, operations, count, dev)


cdef CUresult _cuCtxGetDevice_v2(CUdevice* device, CUcontext ctx) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuCtxGetDevice_v2
    _check_or_init_driver()
    if __cuCtxGetDevice_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuCtxGetDevice_v2 is not found")
    return (<CUresult (*)(CUdevice*, CUcontext) noexcept nogil>__cuCtxGetDevice_v2)(
        device, ctx)


cdef CUresult _cuCtxSynchronize_v2(CUcontext ctx) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuCtxSynchronize_v2
    _check_or_init_driver()
    if __cuCtxSynchronize_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuCtxSynchronize_v2 is not found")
    return (<CUresult (*)(CUcontext) noexcept nogil>__cuCtxSynchronize_v2)(
        ctx)


cdef CUresult _cuMemcpyBatchAsync_v2(CUdeviceptr* dsts, CUdeviceptr* srcs, size_t* sizes, size_t count, CUmemcpyAttributes* attrs, size_t* attrsIdxs, size_t numAttrs, CUstream hStream) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuMemcpyBatchAsync_v2
    _check_or_init_driver()
    if __cuMemcpyBatchAsync_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemcpyBatchAsync_v2 is not found")
    return (<CUresult (*)(CUdeviceptr*, CUdeviceptr*, size_t*, size_t, CUmemcpyAttributes*, size_t*, size_t, CUstream) noexcept nogil>__cuMemcpyBatchAsync_v2)(
        dsts, srcs, sizes, count, attrs, attrsIdxs, numAttrs, hStream)


cdef CUresult _cuMemcpy3DBatchAsync_v2(size_t numOps, CUDA_MEMCPY3D_BATCH_OP* opList, unsigned long long flags, CUstream hStream) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuMemcpy3DBatchAsync_v2
    _check_or_init_driver()
    if __cuMemcpy3DBatchAsync_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemcpy3DBatchAsync_v2 is not found")
    return (<CUresult (*)(size_t, CUDA_MEMCPY3D_BATCH_OP*, unsigned long long, CUstream) noexcept nogil>__cuMemcpy3DBatchAsync_v2)(
        numOps, opList, flags, hStream)


cdef CUresult _cuMemGetDefaultMemPool(CUmemoryPool* pool_out, CUmemLocation* location, CUmemAllocationType type) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuMemGetDefaultMemPool
    _check_or_init_driver()
    if __cuMemGetDefaultMemPool == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemGetDefaultMemPool is not found")
    return (<CUresult (*)(CUmemoryPool*, CUmemLocation*, CUmemAllocationType) noexcept nogil>__cuMemGetDefaultMemPool)(
        pool_out, location, type)


cdef CUresult _cuMemGetMemPool(CUmemoryPool* pool, CUmemLocation* location, CUmemAllocationType type) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuMemGetMemPool
    _check_or_init_driver()
    if __cuMemGetMemPool == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemGetMemPool is not found")
    return (<CUresult (*)(CUmemoryPool*, CUmemLocation*, CUmemAllocationType) noexcept nogil>__cuMemGetMemPool)(
        pool, location, type)


cdef CUresult _cuMemSetMemPool(CUmemLocation* location, CUmemAllocationType type, CUmemoryPool pool) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuMemSetMemPool
    _check_or_init_driver()
    if __cuMemSetMemPool == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemSetMemPool is not found")
    return (<CUresult (*)(CUmemLocation*, CUmemAllocationType, CUmemoryPool) noexcept nogil>__cuMemSetMemPool)(
        location, type, pool)


cdef CUresult _cuMemPrefetchBatchAsync(CUdeviceptr* dptrs, size_t* sizes, size_t count, CUmemLocation* prefetchLocs, size_t* prefetchLocIdxs, size_t numPrefetchLocs, unsigned long long flags, CUstream hStream) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuMemPrefetchBatchAsync
    _check_or_init_driver()
    if __cuMemPrefetchBatchAsync == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemPrefetchBatchAsync is not found")
    return (<CUresult (*)(CUdeviceptr*, size_t*, size_t, CUmemLocation*, size_t*, size_t, unsigned long long, CUstream) noexcept nogil>__cuMemPrefetchBatchAsync)(
        dptrs, sizes, count, prefetchLocs, prefetchLocIdxs, numPrefetchLocs, flags, hStream)


cdef CUresult _cuMemDiscardBatchAsync(CUdeviceptr* dptrs, size_t* sizes, size_t count, unsigned long long flags, CUstream hStream) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuMemDiscardBatchAsync
    _check_or_init_driver()
    if __cuMemDiscardBatchAsync == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemDiscardBatchAsync is not found")
    return (<CUresult (*)(CUdeviceptr*, size_t*, size_t, unsigned long long, CUstream) noexcept nogil>__cuMemDiscardBatchAsync)(
        dptrs, sizes, count, flags, hStream)


cdef CUresult _cuMemDiscardAndPrefetchBatchAsync(CUdeviceptr* dptrs, size_t* sizes, size_t count, CUmemLocation* prefetchLocs, size_t* prefetchLocIdxs, size_t numPrefetchLocs, unsigned long long flags, CUstream hStream) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuMemDiscardAndPrefetchBatchAsync
    _check_or_init_driver()
    if __cuMemDiscardAndPrefetchBatchAsync == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemDiscardAndPrefetchBatchAsync is not found")
    return (<CUresult (*)(CUdeviceptr*, size_t*, size_t, CUmemLocation*, size_t*, size_t, unsigned long long, CUstream) noexcept nogil>__cuMemDiscardAndPrefetchBatchAsync)(
        dptrs, sizes, count, prefetchLocs, prefetchLocIdxs, numPrefetchLocs, flags, hStream)


cdef CUresult _cuDeviceGetP2PAtomicCapabilities(unsigned int* capabilities, const CUatomicOperation* operations, unsigned int count, CUdevice srcDevice, CUdevice dstDevice) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuDeviceGetP2PAtomicCapabilities
    _check_or_init_driver()
    if __cuDeviceGetP2PAtomicCapabilities == NULL:
        with gil:
            raise FunctionNotFoundError("function cuDeviceGetP2PAtomicCapabilities is not found")
    return (<CUresult (*)(unsigned int*, const CUatomicOperation*, unsigned int, CUdevice, CUdevice) noexcept nogil>__cuDeviceGetP2PAtomicCapabilities)(
        capabilities, operations, count, srcDevice, dstDevice)


cdef CUresult _cuGreenCtxGetId(CUgreenCtx greenCtx, unsigned long long* greenCtxId) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuGreenCtxGetId
    _check_or_init_driver()
    if __cuGreenCtxGetId == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGreenCtxGetId is not found")
    return (<CUresult (*)(CUgreenCtx, unsigned long long*) noexcept nogil>__cuGreenCtxGetId)(
        greenCtx, greenCtxId)


cdef CUresult _cuMulticastBindMem_v2(CUmemGenericAllocationHandle mcHandle, CUdevice dev, size_t mcOffset, CUmemGenericAllocationHandle memHandle, size_t memOffset, size_t size, unsigned long long flags) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuMulticastBindMem_v2
    _check_or_init_driver()
    if __cuMulticastBindMem_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMulticastBindMem_v2 is not found")
    return (<CUresult (*)(CUmemGenericAllocationHandle, CUdevice, size_t, CUmemGenericAllocationHandle, size_t, size_t, unsigned long long) noexcept nogil>__cuMulticastBindMem_v2)(
        mcHandle, dev, mcOffset, memHandle, memOffset, size, flags)


cdef CUresult _cuMulticastBindAddr_v2(CUmemGenericAllocationHandle mcHandle, CUdevice dev, size_t mcOffset, CUdeviceptr memptr, size_t size, unsigned long long flags) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuMulticastBindAddr_v2
    _check_or_init_driver()
    if __cuMulticastBindAddr_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMulticastBindAddr_v2 is not found")
    return (<CUresult (*)(CUmemGenericAllocationHandle, CUdevice, size_t, CUdeviceptr, size_t, unsigned long long) noexcept nogil>__cuMulticastBindAddr_v2)(
        mcHandle, dev, mcOffset, memptr, size, flags)


cdef CUresult _cuGraphNodeGetContainingGraph(CUgraphNode hNode, CUgraph* phGraph) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuGraphNodeGetContainingGraph
    _check_or_init_driver()
    if __cuGraphNodeGetContainingGraph == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphNodeGetContainingGraph is not found")
    return (<CUresult (*)(CUgraphNode, CUgraph*) noexcept nogil>__cuGraphNodeGetContainingGraph)(
        hNode, phGraph)


cdef CUresult _cuGraphNodeGetLocalId(CUgraphNode hNode, unsigned int* nodeId) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuGraphNodeGetLocalId
    _check_or_init_driver()
    if __cuGraphNodeGetLocalId == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphNodeGetLocalId is not found")
    return (<CUresult (*)(CUgraphNode, unsigned int*) noexcept nogil>__cuGraphNodeGetLocalId)(
        hNode, nodeId)


cdef CUresult _cuGraphNodeGetToolsId(CUgraphNode hNode, unsigned long long* toolsNodeId) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuGraphNodeGetToolsId
    _check_or_init_driver()
    if __cuGraphNodeGetToolsId == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphNodeGetToolsId is not found")
    return (<CUresult (*)(CUgraphNode, unsigned long long*) noexcept nogil>__cuGraphNodeGetToolsId)(
        hNode, toolsNodeId)


cdef CUresult _cuGraphGetId(CUgraph hGraph, unsigned int* graphId) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuGraphGetId
    _check_or_init_driver()
    if __cuGraphGetId == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphGetId is not found")
    return (<CUresult (*)(CUgraph, unsigned int*) noexcept nogil>__cuGraphGetId)(
        hGraph, graphId)


cdef CUresult _cuGraphExecGetId(CUgraphExec hGraphExec, unsigned int* graphId) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuGraphExecGetId
    _check_or_init_driver()
    if __cuGraphExecGetId == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphExecGetId is not found")
    return (<CUresult (*)(CUgraphExec, unsigned int*) noexcept nogil>__cuGraphExecGetId)(
        hGraphExec, graphId)


cdef CUresult _cuDevSmResourceSplit(CUdevResource* result, unsigned int nbGroups, const CUdevResource* input, CUdevResource* remainder, unsigned int flags, CU_DEV_SM_RESOURCE_GROUP_PARAMS* groupParams) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuDevSmResourceSplit
    _check_or_init_driver()
    if __cuDevSmResourceSplit == NULL:
        with gil:
            raise FunctionNotFoundError("function cuDevSmResourceSplit is not found")
    return (<CUresult (*)(CUdevResource*, unsigned int, const CUdevResource*, CUdevResource*, unsigned int, CU_DEV_SM_RESOURCE_GROUP_PARAMS*) noexcept nogil>__cuDevSmResourceSplit)(
        result, nbGroups, input, remainder, flags, groupParams)


cdef CUresult _cuStreamGetDevResource(CUstream hStream, CUdevResource* resource, CUdevResourceType type) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuStreamGetDevResource
    _check_or_init_driver()
    if __cuStreamGetDevResource == NULL:
        with gil:
            raise FunctionNotFoundError("function cuStreamGetDevResource is not found")
    return (<CUresult (*)(CUstream, CUdevResource*, CUdevResourceType) noexcept nogil>__cuStreamGetDevResource)(
        hStream, resource, type)


cdef CUresult _cuKernelGetParamCount(CUkernel kernel, size_t* paramCount) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuKernelGetParamCount
    _check_or_init_driver()
    if __cuKernelGetParamCount == NULL:
        with gil:
            raise FunctionNotFoundError("function cuKernelGetParamCount is not found")
    return (<CUresult (*)(CUkernel, size_t*) noexcept nogil>__cuKernelGetParamCount)(
        kernel, paramCount)


cdef CUresult _cuMemcpyWithAttributesAsync(CUdeviceptr dst, CUdeviceptr src, size_t size, CUmemcpyAttributes* attr, CUstream hStream) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuMemcpyWithAttributesAsync
    _check_or_init_driver()
    if __cuMemcpyWithAttributesAsync == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemcpyWithAttributesAsync is not found")
    return (<CUresult (*)(CUdeviceptr, CUdeviceptr, size_t, CUmemcpyAttributes*, CUstream) noexcept nogil>__cuMemcpyWithAttributesAsync)(
        dst, src, size, attr, hStream)


cdef CUresult _cuMemcpy3DWithAttributesAsync(CUDA_MEMCPY3D_BATCH_OP* op, unsigned long long flags, CUstream hStream) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuMemcpy3DWithAttributesAsync
    _check_or_init_driver()
    if __cuMemcpy3DWithAttributesAsync == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemcpy3DWithAttributesAsync is not found")
    return (<CUresult (*)(CUDA_MEMCPY3D_BATCH_OP*, unsigned long long, CUstream) noexcept nogil>__cuMemcpy3DWithAttributesAsync)(
        op, flags, hStream)


cdef CUresult _cuStreamBeginCaptureToCig(CUstream hStream, CUstreamCigCaptureParams* streamCigCaptureParams) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuStreamBeginCaptureToCig
    _check_or_init_driver()
    if __cuStreamBeginCaptureToCig == NULL:
        with gil:
            raise FunctionNotFoundError("function cuStreamBeginCaptureToCig is not found")
    return (<CUresult (*)(CUstream, CUstreamCigCaptureParams*) noexcept nogil>__cuStreamBeginCaptureToCig)(
        hStream, streamCigCaptureParams)


cdef CUresult _cuStreamEndCaptureToCig(CUstream hStream) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuStreamEndCaptureToCig
    _check_or_init_driver()
    if __cuStreamEndCaptureToCig == NULL:
        with gil:
            raise FunctionNotFoundError("function cuStreamEndCaptureToCig is not found")
    return (<CUresult (*)(CUstream) noexcept nogil>__cuStreamEndCaptureToCig)(
        hStream)


cdef CUresult _cuFuncGetParamCount(CUfunction func, size_t* paramCount) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuFuncGetParamCount
    _check_or_init_driver()
    if __cuFuncGetParamCount == NULL:
        with gil:
            raise FunctionNotFoundError("function cuFuncGetParamCount is not found")
    return (<CUresult (*)(CUfunction, size_t*) noexcept nogil>__cuFuncGetParamCount)(
        func, paramCount)


cdef CUresult _cuLaunchHostFunc_v2(CUstream hStream, CUhostFn fn, void* userData, unsigned int syncMode) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuLaunchHostFunc_v2
    _check_or_init_driver()
    if __cuLaunchHostFunc_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuLaunchHostFunc_v2 is not found")
    return (<CUresult (*)(CUstream, CUhostFn, void*, unsigned int) noexcept nogil>__cuLaunchHostFunc_v2)(
        hStream, fn, userData, syncMode)


cdef CUresult _cuGraphNodeGetParams(CUgraphNode hNode, CUgraphNodeParams* nodeParams) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuGraphNodeGetParams
    _check_or_init_driver()
    if __cuGraphNodeGetParams == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphNodeGetParams is not found")
    return (<CUresult (*)(CUgraphNode, CUgraphNodeParams*) noexcept nogil>__cuGraphNodeGetParams)(
        hNode, nodeParams)


cdef CUresult _cuCoredumpRegisterStartCallback(CUcoredumpStatusCallback callback, void* userData, CUcoredumpCallbackHandle* callbackOut) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuCoredumpRegisterStartCallback
    _check_or_init_driver()
    if __cuCoredumpRegisterStartCallback == NULL:
        with gil:
            raise FunctionNotFoundError("function cuCoredumpRegisterStartCallback is not found")
    return (<CUresult (*)(CUcoredumpStatusCallback, void*, CUcoredumpCallbackHandle*) noexcept nogil>__cuCoredumpRegisterStartCallback)(
        callback, userData, callbackOut)


cdef CUresult _cuCoredumpRegisterCompleteCallback(CUcoredumpStatusCallback callback, void* userData, CUcoredumpCallbackHandle* callbackOut) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuCoredumpRegisterCompleteCallback
    _check_or_init_driver()
    if __cuCoredumpRegisterCompleteCallback == NULL:
        with gil:
            raise FunctionNotFoundError("function cuCoredumpRegisterCompleteCallback is not found")
    return (<CUresult (*)(CUcoredumpStatusCallback, void*, CUcoredumpCallbackHandle*) noexcept nogil>__cuCoredumpRegisterCompleteCallback)(
        callback, userData, callbackOut)


cdef CUresult _cuCoredumpDeregisterStartCallback(CUcoredumpCallbackHandle callback) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuCoredumpDeregisterStartCallback
    _check_or_init_driver()
    if __cuCoredumpDeregisterStartCallback == NULL:
        with gil:
            raise FunctionNotFoundError("function cuCoredumpDeregisterStartCallback is not found")
    return (<CUresult (*)(CUcoredumpCallbackHandle) noexcept nogil>__cuCoredumpDeregisterStartCallback)(
        callback)


cdef CUresult _cuCoredumpDeregisterCompleteCallback(CUcoredumpCallbackHandle callback) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuCoredumpDeregisterCompleteCallback
    _check_or_init_driver()
    if __cuCoredumpDeregisterCompleteCallback == NULL:
        with gil:
            raise FunctionNotFoundError("function cuCoredumpDeregisterCompleteCallback is not found")
    return (<CUresult (*)(CUcoredumpCallbackHandle) noexcept nogil>__cuCoredumpDeregisterCompleteCallback)(
        callback)


cdef CUresult _cuLogicalEndpointIdReserve(CUlogicalEndpointId* baseLeId, cuuint32_t count) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuLogicalEndpointIdReserve
    _check_or_init_driver()
    if __cuLogicalEndpointIdReserve == NULL:
        with gil:
            raise FunctionNotFoundError("function cuLogicalEndpointIdReserve is not found")
    return (<CUresult (*)(CUlogicalEndpointId*, cuuint32_t) noexcept nogil>__cuLogicalEndpointIdReserve)(
        baseLeId, count)


cdef CUresult _cuLogicalEndpointIdRelease(CUlogicalEndpointId baseLeId, cuuint32_t count) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuLogicalEndpointIdRelease
    _check_or_init_driver()
    if __cuLogicalEndpointIdRelease == NULL:
        with gil:
            raise FunctionNotFoundError("function cuLogicalEndpointIdRelease is not found")
    return (<CUresult (*)(CUlogicalEndpointId, cuuint32_t) noexcept nogil>__cuLogicalEndpointIdRelease)(
        baseLeId, count)


cdef CUresult _cuLogicalEndpointCreate(CUlogicalEndpointId leId, const CUlogicalEndpointProp* prop) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuLogicalEndpointCreate
    _check_or_init_driver()
    if __cuLogicalEndpointCreate == NULL:
        with gil:
            raise FunctionNotFoundError("function cuLogicalEndpointCreate is not found")
    return (<CUresult (*)(CUlogicalEndpointId, const CUlogicalEndpointProp*) noexcept nogil>__cuLogicalEndpointCreate)(
        leId, prop)


cdef CUresult _cuLogicalEndpointAddDevice(CUlogicalEndpointId leId, CUdevice dev) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuLogicalEndpointAddDevice
    _check_or_init_driver()
    if __cuLogicalEndpointAddDevice == NULL:
        with gil:
            raise FunctionNotFoundError("function cuLogicalEndpointAddDevice is not found")
    return (<CUresult (*)(CUlogicalEndpointId, CUdevice) noexcept nogil>__cuLogicalEndpointAddDevice)(
        leId, dev)


cdef CUresult _cuLogicalEndpointDestroy(CUlogicalEndpointId leId) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuLogicalEndpointDestroy
    _check_or_init_driver()
    if __cuLogicalEndpointDestroy == NULL:
        with gil:
            raise FunctionNotFoundError("function cuLogicalEndpointDestroy is not found")
    return (<CUresult (*)(CUlogicalEndpointId) noexcept nogil>__cuLogicalEndpointDestroy)(
        leId)


cdef CUresult _cuLogicalEndpointBindAddr(CUlogicalEndpointId leId, CUdevice dev, cuuint64_t offset, void* ptr, cuuint64_t size, unsigned long long flags) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuLogicalEndpointBindAddr
    _check_or_init_driver()
    if __cuLogicalEndpointBindAddr == NULL:
        with gil:
            raise FunctionNotFoundError("function cuLogicalEndpointBindAddr is not found")
    return (<CUresult (*)(CUlogicalEndpointId, CUdevice, cuuint64_t, void*, cuuint64_t, unsigned long long) noexcept nogil>__cuLogicalEndpointBindAddr)(
        leId, dev, offset, ptr, size, flags)


cdef CUresult _cuLogicalEndpointBindMem(CUlogicalEndpointId leId, CUdevice dev, cuuint64_t offset, CUmemGenericAllocationHandle memHandle, cuuint64_t memOffset, cuuint64_t size, unsigned long long flags) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuLogicalEndpointBindMem
    _check_or_init_driver()
    if __cuLogicalEndpointBindMem == NULL:
        with gil:
            raise FunctionNotFoundError("function cuLogicalEndpointBindMem is not found")
    return (<CUresult (*)(CUlogicalEndpointId, CUdevice, cuuint64_t, CUmemGenericAllocationHandle, cuuint64_t, cuuint64_t, unsigned long long) noexcept nogil>__cuLogicalEndpointBindMem)(
        leId, dev, offset, memHandle, memOffset, size, flags)


cdef CUresult _cuLogicalEndpointUnbind(CUlogicalEndpointId leId, CUdevice dev, cuuint64_t offset, cuuint64_t size) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuLogicalEndpointUnbind
    _check_or_init_driver()
    if __cuLogicalEndpointUnbind == NULL:
        with gil:
            raise FunctionNotFoundError("function cuLogicalEndpointUnbind is not found")
    return (<CUresult (*)(CUlogicalEndpointId, CUdevice, cuuint64_t, cuuint64_t) noexcept nogil>__cuLogicalEndpointUnbind)(
        leId, dev, offset, size)


cdef CUresult _cuLogicalEndpointExport(void* handle, CUlogicalEndpointId leId, CUlogicalEndpointIpcHandleType handleType) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuLogicalEndpointExport
    _check_or_init_driver()
    if __cuLogicalEndpointExport == NULL:
        with gil:
            raise FunctionNotFoundError("function cuLogicalEndpointExport is not found")
    return (<CUresult (*)(void*, CUlogicalEndpointId, CUlogicalEndpointIpcHandleType) noexcept nogil>__cuLogicalEndpointExport)(
        handle, leId, handleType)


cdef CUresult _cuLogicalEndpointImport(CUlogicalEndpointId leId, const void* handle, CUlogicalEndpointIpcHandleType handleType) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuLogicalEndpointImport
    _check_or_init_driver()
    if __cuLogicalEndpointImport == NULL:
        with gil:
            raise FunctionNotFoundError("function cuLogicalEndpointImport is not found")
    return (<CUresult (*)(CUlogicalEndpointId, const void*, CUlogicalEndpointIpcHandleType) noexcept nogil>__cuLogicalEndpointImport)(
        leId, handle, handleType)


cdef CUresult _cuLogicalEndpointGetLimits(cuuint64_t* bindAlignment, cuuint64_t* maxSize, const CUlogicalEndpointProp* prop) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuLogicalEndpointGetLimits
    _check_or_init_driver()
    if __cuLogicalEndpointGetLimits == NULL:
        with gil:
            raise FunctionNotFoundError("function cuLogicalEndpointGetLimits is not found")
    return (<CUresult (*)(cuuint64_t*, cuuint64_t*, const CUlogicalEndpointProp*) noexcept nogil>__cuLogicalEndpointGetLimits)(
        bindAlignment, maxSize, prop)


cdef CUresult _cuLogicalEndpointQuery(CUlogicalEndpointId leId, cuuint32_t count, int* queryStatus) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuLogicalEndpointQuery
    _check_or_init_driver()
    if __cuLogicalEndpointQuery == NULL:
        with gil:
            raise FunctionNotFoundError("function cuLogicalEndpointQuery is not found")
    return (<CUresult (*)(CUlogicalEndpointId, cuuint32_t, int*) noexcept nogil>__cuLogicalEndpointQuery)(
        leId, count, queryStatus)


cdef CUresult _cuStreamBeginRecaptureToGraph(CUstream hStream, CUstreamCaptureMode mode, CUgraph hGraph, CUgraphRecaptureCallback callbackFunc, void* userData) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuStreamBeginRecaptureToGraph
    _check_or_init_driver()
    if __cuStreamBeginRecaptureToGraph == NULL:
        with gil:
            raise FunctionNotFoundError("function cuStreamBeginRecaptureToGraph is not found")
    return (<CUresult (*)(CUstream, CUstreamCaptureMode, CUgraph, CUgraphRecaptureCallback, void*) noexcept nogil>__cuStreamBeginRecaptureToGraph)(
        hStream, mode, hGraph, callbackFunc, userData)


cdef CUresult _cuDeviceGetFabricClusterUuid(CUuuid* uuid, CUdevice dev) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuDeviceGetFabricClusterUuid
    _check_or_init_driver()
    if __cuDeviceGetFabricClusterUuid == NULL:
        with gil:
            raise FunctionNotFoundError("function cuDeviceGetFabricClusterUuid is not found")
    return (<CUresult (*)(CUuuid*, CUdevice) noexcept nogil>__cuDeviceGetFabricClusterUuid)(
        uuid, dev)


cdef CUresult _cuDeviceGetCliqueCount(size_t* count, CUdevice dev) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuDeviceGetCliqueCount
    _check_or_init_driver()
    if __cuDeviceGetCliqueCount == NULL:
        with gil:
            raise FunctionNotFoundError("function cuDeviceGetCliqueCount is not found")
    return (<CUresult (*)(size_t*, CUdevice) noexcept nogil>__cuDeviceGetCliqueCount)(
        count, dev)


cdef CUresult _cuDeviceGetCliqueInfo(CUcliqueInfo* cliqueInfo, size_t* count, CUdevice dev) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuDeviceGetCliqueInfo
    _check_or_init_driver()
    if __cuDeviceGetCliqueInfo == NULL:
        with gil:
            raise FunctionNotFoundError("function cuDeviceGetCliqueInfo is not found")
    return (<CUresult (*)(CUcliqueInfo*, size_t*, CUdevice) noexcept nogil>__cuDeviceGetCliqueInfo)(
        cliqueInfo, count, dev)


cdef CUresult _cuMemGetLocationInfo(CUdeviceptr ptr, size_t size, size_t summaryGranularity, size_t samplingGranularity, CUmemLocation* location_out) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuMemGetLocationInfo
    _check_or_init_driver()
    if __cuMemGetLocationInfo == NULL:
        with gil:
            raise FunctionNotFoundError("function cuMemGetLocationInfo is not found")
    return (<CUresult (*)(CUdeviceptr, size_t, size_t, size_t, CUmemLocation*) noexcept nogil>__cuMemGetLocationInfo)(
        ptr, size, summaryGranularity, samplingGranularity, location_out)


cdef CUresult _cuGraphAddNode_v3(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, const CUgraphEdgeData* dependencyData, size_t numDependencies, CUgraphNodeParams* nodeParams) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuGraphAddNode_v3
    _check_or_init_driver()
    if __cuGraphAddNode_v3 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphAddNode_v3 is not found")
    return (<CUresult (*)(CUgraphNode*, CUgraph, const CUgraphNode*, const CUgraphEdgeData*, size_t, CUgraphNodeParams*) noexcept nogil>__cuGraphAddNode_v3)(
        phGraphNode, hGraph, dependencies, dependencyData, numDependencies, nodeParams)


cdef CUresult _cuGraphNodeSetParams_v2(CUgraphNode hNode, CUgraphNodeParams* nodeParams) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuGraphNodeSetParams_v2
    _check_or_init_driver()
    if __cuGraphNodeSetParams_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cuGraphNodeSetParams_v2 is not found")
    return (<CUresult (*)(CUgraphNode, CUgraphNodeParams*) noexcept nogil>__cuGraphNodeSetParams_v2)(
        hNode, nodeParams)


cdef CUresult _cuCheckpointOperationComplete(CUcheckpointOperationHandle handle) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuCheckpointOperationComplete
    _check_or_init_driver()
    if __cuCheckpointOperationComplete == NULL:
        with gil:
            raise FunctionNotFoundError("function cuCheckpointOperationComplete is not found")
    return (<CUresult (*)(CUcheckpointOperationHandle) noexcept nogil>__cuCheckpointOperationComplete)(
        handle)


cdef CUresult _cuFabricCpyBatchAsync(size_t numOps, CUDA_FABRIC_COPY_BATCH_OP* opList, unsigned long long flags, CUstream hStream) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuFabricCpyBatchAsync
    _check_or_init_driver()
    if __cuFabricCpyBatchAsync == NULL:
        with gil:
            raise FunctionNotFoundError("function cuFabricCpyBatchAsync is not found")
    return (<CUresult (*)(size_t, CUDA_FABRIC_COPY_BATCH_OP*, unsigned long long, CUstream) noexcept nogil>__cuFabricCpyBatchAsync)(
        numOps, opList, flags, hStream)


cdef CUresult _cuTensorMapSetAllocationClass(CUtensorMap* tensorMap, CUtensorMapAllocClass allocClass, CUtensorMapAllocClass* outAllocClass) except ?CUDA_ERROR_NOT_FOUND nogil:
    global __cuTensorMapSetAllocationClass
    _check_or_init_driver()
    if __cuTensorMapSetAllocationClass == NULL:
        with gil:
            raise FunctionNotFoundError("function cuTensorMapSetAllocationClass is not found")
    return (<CUresult (*)(CUtensorMap*, CUtensorMapAllocClass, CUtensorMapAllocClass*) noexcept nogil>__cuTensorMapSetAllocationClass)(
        tensorMap, allocClass, outAllocClass)
