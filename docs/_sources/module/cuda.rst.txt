----
cuda
----

Data types used by CUDA driver
------------------------------

    
    

    
  
.. autoenum:: cuda.cuda.CUipcMem_flags
.. autoenum:: cuda.cuda.CUmemAttach_flags
.. autoenum:: cuda.cuda.CUctx_flags
.. autoenum:: cuda.cuda.CUstream_flags
.. autoenum:: cuda.cuda.CUevent_flags
.. autoenum:: cuda.cuda.CUevent_record_flags
.. autoenum:: cuda.cuda.CUevent_wait_flags
.. autoenum:: cuda.cuda.CUstreamWaitValue_flags
.. autoenum:: cuda.cuda.CUstreamWriteValue_flags
.. autoenum:: cuda.cuda.CUstreamBatchMemOpType
.. autoenum:: cuda.cuda.CUoccupancy_flags
.. autoenum:: cuda.cuda.CUstreamUpdateCaptureDependencies_flags
.. autoenum:: cuda.cuda.CUarray_format
.. autoenum:: cuda.cuda.CUaddress_mode
.. autoenum:: cuda.cuda.CUfilter_mode
.. autoenum:: cuda.cuda.CUdevice_attribute
.. autoenum:: cuda.cuda.CUpointer_attribute
.. autoenum:: cuda.cuda.CUfunction_attribute
.. autoenum:: cuda.cuda.CUfunc_cache
.. autoenum:: cuda.cuda.CUsharedconfig
.. autoenum:: cuda.cuda.CUshared_carveout
.. autoenum:: cuda.cuda.CUmemorytype
.. autoenum:: cuda.cuda.CUcomputemode
.. autoenum:: cuda.cuda.CUmem_advise
.. autoenum:: cuda.cuda.CUmem_range_attribute
.. autoenum:: cuda.cuda.CUjit_option
.. autoenum:: cuda.cuda.CUjit_target
.. autoenum:: cuda.cuda.CUjit_fallback
.. autoenum:: cuda.cuda.CUjit_cacheMode
.. autoenum:: cuda.cuda.CUjitInputType
.. autoenum:: cuda.cuda.CUgraphicsRegisterFlags
.. autoenum:: cuda.cuda.CUgraphicsMapResourceFlags
.. autoenum:: cuda.cuda.CUarray_cubemap_face
.. autoenum:: cuda.cuda.CUlimit
.. autoenum:: cuda.cuda.CUresourcetype
.. autoenum:: cuda.cuda.CUaccessProperty
.. autoenum:: cuda.cuda.CUgraphNodeType
.. autoenum:: cuda.cuda.CUsynchronizationPolicy
.. autoenum:: cuda.cuda.CUkernelNodeAttrID
.. autoenum:: cuda.cuda.CUstreamCaptureStatus
.. autoenum:: cuda.cuda.CUstreamCaptureMode
.. autoenum:: cuda.cuda.CUstreamAttrID
.. autoenum:: cuda.cuda.CUdriverProcAddress_flags
.. autoenum:: cuda.cuda.CUexecAffinityType
.. autoenum:: cuda.cuda.CUresult
.. autoenum:: cuda.cuda.CUdevice_P2PAttribute
.. autoenum:: cuda.cuda.CUresourceViewFormat
.. autoenum:: cuda.cuda.CUDA_POINTER_ATTRIBUTE_ACCESS_FLAGS
.. autoenum:: cuda.cuda.CUexternalMemoryHandleType
.. autoenum:: cuda.cuda.CUexternalSemaphoreHandleType
.. autoenum:: cuda.cuda.CUmemAllocationHandleType
.. autoenum:: cuda.cuda.CUmemAccess_flags
.. autoenum:: cuda.cuda.CUmemLocationType
.. autoenum:: cuda.cuda.CUmemAllocationType
.. autoenum:: cuda.cuda.CUmemAllocationGranularity_flags
.. autoenum:: cuda.cuda.CUarraySparseSubresourceType
.. autoenum:: cuda.cuda.CUmemOperationType
.. autoenum:: cuda.cuda.CUmemHandleType
.. autoenum:: cuda.cuda.CUmemAllocationCompType
.. autoenum:: cuda.cuda.CUgraphExecUpdateResult
.. autoenum:: cuda.cuda.CUmemPool_attribute
.. autoenum:: cuda.cuda.CUgraphMem_attribute
.. autoenum:: cuda.cuda.CUflushGPUDirectRDMAWritesOptions
.. autoenum:: cuda.cuda.CUGPUDirectRDMAWritesOrdering
.. autoenum:: cuda.cuda.CUflushGPUDirectRDMAWritesScope
.. autoenum:: cuda.cuda.CUflushGPUDirectRDMAWritesTarget
.. autoenum:: cuda.cuda.CUgraphDebugDot_flags
.. autoenum:: cuda.cuda.CUuserObject_flags
.. autoenum:: cuda.cuda.CUuserObjectRetain_flags
.. autoenum:: cuda.cuda.CUgraphInstantiate_flags
.. autoenum:: cuda.cuda.CUeglFrameType
.. autoenum:: cuda.cuda.CUeglResourceLocationFlags
.. autoenum:: cuda.cuda.CUeglColorFormat
.. autoclass:: cuda.cuda.CUdeviceptr_v2
.. autoclass:: cuda.cuda.CUdeviceptr
.. autoclass:: cuda.cuda.CUdevice_v1
.. autoclass:: cuda.cuda.CUdevice
.. autoclass:: cuda.cuda.CUcontext
.. autoclass:: cuda.cuda.CUmodule
.. autoclass:: cuda.cuda.CUfunction
.. autoclass:: cuda.cuda.CUarray
.. autoclass:: cuda.cuda.CUmipmappedArray
.. autoclass:: cuda.cuda.CUtexref
.. autoclass:: cuda.cuda.CUsurfref
.. autoclass:: cuda.cuda.CUevent
.. autoclass:: cuda.cuda.CUstream
.. autoclass:: cuda.cuda.CUgraphicsResource
.. autoclass:: cuda.cuda.CUtexObject_v1
.. autoclass:: cuda.cuda.CUtexObject
.. autoclass:: cuda.cuda.CUsurfObject_v1
.. autoclass:: cuda.cuda.CUsurfObject
.. autoclass:: cuda.cuda.CUexternalMemory
.. autoclass:: cuda.cuda.CUexternalSemaphore
.. autoclass:: cuda.cuda.CUgraph
.. autoclass:: cuda.cuda.CUgraphNode
.. autoclass:: cuda.cuda.CUgraphExec
.. autoclass:: cuda.cuda.CUmemoryPool
.. autoclass:: cuda.cuda.CUuserObject
.. autoclass:: cuda.cuda.CUuuid
.. autoclass:: cuda.cuda.CUipcEventHandle_v1
.. autoclass:: cuda.cuda.CUipcEventHandle
.. autoclass:: cuda.cuda.CUipcMemHandle_v1
.. autoclass:: cuda.cuda.CUipcMemHandle
.. autoclass:: cuda.cuda.CUstreamBatchMemOpParams_v1
.. autoclass:: cuda.cuda.CUstreamBatchMemOpParams
.. autoclass:: cuda.cuda.CUdevprop_v1
.. autoclass:: cuda.cuda.CUdevprop
.. autoclass:: cuda.cuda.CUlinkState
.. autoclass:: cuda.cuda.CUhostFn
.. autoclass:: cuda.cuda.CUaccessPolicyWindow_v1
.. autoclass:: cuda.cuda.CUaccessPolicyWindow
.. autoclass:: cuda.cuda.CUDA_KERNEL_NODE_PARAMS_v1
.. autoclass:: cuda.cuda.CUDA_KERNEL_NODE_PARAMS
.. autoclass:: cuda.cuda.CUDA_MEMSET_NODE_PARAMS_v1
.. autoclass:: cuda.cuda.CUDA_MEMSET_NODE_PARAMS
.. autoclass:: cuda.cuda.CUDA_HOST_NODE_PARAMS_v1
.. autoclass:: cuda.cuda.CUDA_HOST_NODE_PARAMS
.. autoclass:: cuda.cuda.CUkernelNodeAttrValue_v1
.. autoclass:: cuda.cuda.CUkernelNodeAttrValue
.. autoclass:: cuda.cuda.CUstreamAttrValue_v1
.. autoclass:: cuda.cuda.CUstreamAttrValue
.. autoclass:: cuda.cuda.CUexecAffinitySmCount_v1
.. autoclass:: cuda.cuda.CUexecAffinitySmCount
.. autoclass:: cuda.cuda.CUexecAffinityParam_v1
.. autoclass:: cuda.cuda.CUexecAffinityParam
.. autoclass:: cuda.cuda.CUstreamCallback
.. autoclass:: cuda.cuda.CUoccupancyB2DSize
.. autoclass:: cuda.cuda.CUDA_MEMCPY2D_v2
.. autoclass:: cuda.cuda.CUDA_MEMCPY2D
.. autoclass:: cuda.cuda.CUDA_MEMCPY3D_v2
.. autoclass:: cuda.cuda.CUDA_MEMCPY3D
.. autoclass:: cuda.cuda.CUDA_MEMCPY3D_PEER_v1
.. autoclass:: cuda.cuda.CUDA_MEMCPY3D_PEER
.. autoclass:: cuda.cuda.CUDA_ARRAY_DESCRIPTOR_v2
.. autoclass:: cuda.cuda.CUDA_ARRAY_DESCRIPTOR
.. autoclass:: cuda.cuda.CUDA_ARRAY3D_DESCRIPTOR_v2
.. autoclass:: cuda.cuda.CUDA_ARRAY3D_DESCRIPTOR
.. autoclass:: cuda.cuda.CUDA_ARRAY_SPARSE_PROPERTIES_v1
.. autoclass:: cuda.cuda.CUDA_ARRAY_SPARSE_PROPERTIES
.. autoclass:: cuda.cuda.CUDA_ARRAY_MEMORY_REQUIREMENTS_v1
.. autoclass:: cuda.cuda.CUDA_ARRAY_MEMORY_REQUIREMENTS
.. autoclass:: cuda.cuda.CUDA_RESOURCE_DESC_v1
.. autoclass:: cuda.cuda.CUDA_RESOURCE_DESC
.. autoclass:: cuda.cuda.CUDA_TEXTURE_DESC_v1
.. autoclass:: cuda.cuda.CUDA_TEXTURE_DESC
.. autoclass:: cuda.cuda.CUDA_RESOURCE_VIEW_DESC_v1
.. autoclass:: cuda.cuda.CUDA_RESOURCE_VIEW_DESC
.. autoclass:: cuda.cuda.CUDA_POINTER_ATTRIBUTE_P2P_TOKENS_v1
.. autoclass:: cuda.cuda.CUDA_POINTER_ATTRIBUTE_P2P_TOKENS
.. autoclass:: cuda.cuda.CUDA_LAUNCH_PARAMS_v1
.. autoclass:: cuda.cuda.CUDA_LAUNCH_PARAMS
.. autoclass:: cuda.cuda.CUDA_EXTERNAL_MEMORY_HANDLE_DESC_v1
.. autoclass:: cuda.cuda.CUDA_EXTERNAL_MEMORY_HANDLE_DESC
.. autoclass:: cuda.cuda.CUDA_EXTERNAL_MEMORY_BUFFER_DESC_v1
.. autoclass:: cuda.cuda.CUDA_EXTERNAL_MEMORY_BUFFER_DESC
.. autoclass:: cuda.cuda.CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC_v1
.. autoclass:: cuda.cuda.CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC
.. autoclass:: cuda.cuda.CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_v1
.. autoclass:: cuda.cuda.CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC
.. autoclass:: cuda.cuda.CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_v1
.. autoclass:: cuda.cuda.CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS
.. autoclass:: cuda.cuda.CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_v1
.. autoclass:: cuda.cuda.CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS
.. autoclass:: cuda.cuda.CUDA_EXT_SEM_SIGNAL_NODE_PARAMS_v1
.. autoclass:: cuda.cuda.CUDA_EXT_SEM_SIGNAL_NODE_PARAMS
.. autoclass:: cuda.cuda.CUDA_EXT_SEM_WAIT_NODE_PARAMS_v1
.. autoclass:: cuda.cuda.CUDA_EXT_SEM_WAIT_NODE_PARAMS
.. autoclass:: cuda.cuda.CUmemGenericAllocationHandle_v1
.. autoclass:: cuda.cuda.CUmemGenericAllocationHandle
.. autoclass:: cuda.cuda.CUarrayMapInfo_v1
.. autoclass:: cuda.cuda.CUarrayMapInfo
.. autoclass:: cuda.cuda.CUmemLocation_v1
.. autoclass:: cuda.cuda.CUmemLocation
.. autoclass:: cuda.cuda.CUmemAllocationProp_v1
.. autoclass:: cuda.cuda.CUmemAllocationProp
.. autoclass:: cuda.cuda.CUmemAccessDesc_v1
.. autoclass:: cuda.cuda.CUmemAccessDesc
.. autoclass:: cuda.cuda.CUmemPoolProps_v1
.. autoclass:: cuda.cuda.CUmemPoolProps
.. autoclass:: cuda.cuda.CUmemPoolPtrExportData_v1
.. autoclass:: cuda.cuda.CUmemPoolPtrExportData
.. autoclass:: cuda.cuda.CUDA_MEM_ALLOC_NODE_PARAMS
.. autoclass:: cuda.cuda.CUeglFrame_v1
.. autoclass:: cuda.cuda.CUeglFrame
.. autoclass:: cuda.cuda.CUeglStreamConnection

Error Handling
--------------

    
    


This section describes the error handling functions of the low-level CUDA driver application programming interface. 
    

  
.. autofunction:: cuda.cuda.cuGetErrorString
.. autofunction:: cuda.cuda.cuGetErrorName

Initialization
--------------

    
    


This section describes the initialization functions of the low-level CUDA driver application programming interface. 
    

  
.. autofunction:: cuda.cuda.cuInit

Version Management
------------------

    
    


This section describes the version management functions of the low-level CUDA driver application programming interface. 
    

  
.. autofunction:: cuda.cuda.cuDriverGetVersion

Device Management
-----------------

    
    


This section describes the device management functions of the low-level CUDA driver application programming interface. 
    

  
.. autofunction:: cuda.cuda.cuDeviceGet
.. autofunction:: cuda.cuda.cuDeviceGetCount
.. autofunction:: cuda.cuda.cuDeviceGetName
.. autofunction:: cuda.cuda.cuDeviceGetUuid
.. autofunction:: cuda.cuda.cuDeviceGetUuid_v2
.. autofunction:: cuda.cuda.cuDeviceGetLuid
.. autofunction:: cuda.cuda.cuDeviceTotalMem
.. autofunction:: cuda.cuda.cuDeviceGetTexture1DLinearMaxWidth
.. autofunction:: cuda.cuda.cuDeviceGetAttribute
.. autofunction:: cuda.cuda.cuDeviceGetNvSciSyncAttributes
.. autofunction:: cuda.cuda.cuDeviceSetMemPool
.. autofunction:: cuda.cuda.cuDeviceGetMemPool
.. autofunction:: cuda.cuda.cuDeviceGetDefaultMemPool
.. autofunction:: cuda.cuda.cuFlushGPUDirectRDMAWrites

Primary Context Management
--------------------------

    
    


This section describes the primary context management functions of the low-level CUDA driver application programming interface.

The primary context is unique per device and shared with the CUDA runtime API. These functions allow integration with other libraries using CUDA. 
    

  
.. autofunction:: cuda.cuda.cuDevicePrimaryCtxRetain
.. autofunction:: cuda.cuda.cuDevicePrimaryCtxRelease
.. autofunction:: cuda.cuda.cuDevicePrimaryCtxSetFlags
.. autofunction:: cuda.cuda.cuDevicePrimaryCtxGetState
.. autofunction:: cuda.cuda.cuDevicePrimaryCtxReset

Context Management
------------------

    
    


This section describes the context management functions of the low-level CUDA driver application programming interface.

Please note that some functions are described in Primary Context Management section. 
    

  
.. autofunction:: cuda.cuda.cuCtxCreate
.. autofunction:: cuda.cuda.cuCtxCreate_v3
.. autofunction:: cuda.cuda.cuCtxDestroy
.. autofunction:: cuda.cuda.cuCtxPushCurrent
.. autofunction:: cuda.cuda.cuCtxPopCurrent
.. autofunction:: cuda.cuda.cuCtxSetCurrent
.. autofunction:: cuda.cuda.cuCtxGetCurrent
.. autofunction:: cuda.cuda.cuCtxGetDevice
.. autofunction:: cuda.cuda.cuCtxGetFlags
.. autofunction:: cuda.cuda.cuCtxSynchronize
.. autofunction:: cuda.cuda.cuCtxSetLimit
.. autofunction:: cuda.cuda.cuCtxGetLimit
.. autofunction:: cuda.cuda.cuCtxGetCacheConfig
.. autofunction:: cuda.cuda.cuCtxSetCacheConfig
.. autofunction:: cuda.cuda.cuCtxGetSharedMemConfig
.. autofunction:: cuda.cuda.cuCtxSetSharedMemConfig
.. autofunction:: cuda.cuda.cuCtxGetApiVersion
.. autofunction:: cuda.cuda.cuCtxGetStreamPriorityRange
.. autofunction:: cuda.cuda.cuCtxResetPersistingL2Cache
.. autofunction:: cuda.cuda.cuCtxGetExecAffinity

Module Management
-----------------

    
    


This section describes the module management functions of the low-level CUDA driver application programming interface. 
    

  
.. autofunction:: cuda.cuda.cuModuleLoad
.. autofunction:: cuda.cuda.cuModuleLoadData
.. autofunction:: cuda.cuda.cuModuleLoadDataEx
.. autofunction:: cuda.cuda.cuModuleLoadFatBinary
.. autofunction:: cuda.cuda.cuModuleUnload
.. autofunction:: cuda.cuda.cuModuleGetFunction
.. autofunction:: cuda.cuda.cuModuleGetGlobal
.. autofunction:: cuda.cuda.cuModuleGetTexRef
.. autofunction:: cuda.cuda.cuModuleGetSurfRef
.. autofunction:: cuda.cuda.cuLinkCreate
.. autofunction:: cuda.cuda.cuLinkAddData
.. autofunction:: cuda.cuda.cuLinkAddFile
.. autofunction:: cuda.cuda.cuLinkComplete
.. autofunction:: cuda.cuda.cuLinkDestroy

Memory Management
-----------------

    
    


This section describes the memory management functions of the low-level CUDA driver application programming interface. 
    

  
.. autofunction:: cuda.cuda.cuMemGetInfo
.. autofunction:: cuda.cuda.cuMemAlloc
.. autofunction:: cuda.cuda.cuMemAllocPitch
.. autofunction:: cuda.cuda.cuMemFree
.. autofunction:: cuda.cuda.cuMemGetAddressRange
.. autofunction:: cuda.cuda.cuMemAllocHost
.. autofunction:: cuda.cuda.cuMemFreeHost
.. autofunction:: cuda.cuda.cuMemHostAlloc
.. autofunction:: cuda.cuda.cuMemHostGetDevicePointer
.. autofunction:: cuda.cuda.cuMemHostGetFlags
.. autofunction:: cuda.cuda.cuMemAllocManaged
.. autofunction:: cuda.cuda.cuDeviceGetByPCIBusId
.. autofunction:: cuda.cuda.cuDeviceGetPCIBusId
.. autofunction:: cuda.cuda.cuIpcGetEventHandle
.. autofunction:: cuda.cuda.cuIpcOpenEventHandle
.. autofunction:: cuda.cuda.cuIpcGetMemHandle
.. autofunction:: cuda.cuda.cuIpcOpenMemHandle
.. autofunction:: cuda.cuda.cuIpcCloseMemHandle
.. autofunction:: cuda.cuda.cuMemHostRegister
.. autofunction:: cuda.cuda.cuMemHostUnregister
.. autofunction:: cuda.cuda.cuMemcpy
.. autofunction:: cuda.cuda.cuMemcpyPeer
.. autofunction:: cuda.cuda.cuMemcpyHtoD
.. autofunction:: cuda.cuda.cuMemcpyDtoH
.. autofunction:: cuda.cuda.cuMemcpyDtoD
.. autofunction:: cuda.cuda.cuMemcpyDtoA
.. autofunction:: cuda.cuda.cuMemcpyAtoD
.. autofunction:: cuda.cuda.cuMemcpyHtoA
.. autofunction:: cuda.cuda.cuMemcpyAtoH
.. autofunction:: cuda.cuda.cuMemcpyAtoA
.. autofunction:: cuda.cuda.cuMemcpy2D
.. autofunction:: cuda.cuda.cuMemcpy2DUnaligned
.. autofunction:: cuda.cuda.cuMemcpy3D
.. autofunction:: cuda.cuda.cuMemcpy3DPeer
.. autofunction:: cuda.cuda.cuMemcpyAsync
.. autofunction:: cuda.cuda.cuMemcpyPeerAsync
.. autofunction:: cuda.cuda.cuMemcpyHtoDAsync
.. autofunction:: cuda.cuda.cuMemcpyDtoHAsync
.. autofunction:: cuda.cuda.cuMemcpyDtoDAsync
.. autofunction:: cuda.cuda.cuMemcpyHtoAAsync
.. autofunction:: cuda.cuda.cuMemcpyAtoHAsync
.. autofunction:: cuda.cuda.cuMemcpy2DAsync
.. autofunction:: cuda.cuda.cuMemcpy3DAsync
.. autofunction:: cuda.cuda.cuMemcpy3DPeerAsync
.. autofunction:: cuda.cuda.cuMemsetD8
.. autofunction:: cuda.cuda.cuMemsetD16
.. autofunction:: cuda.cuda.cuMemsetD32
.. autofunction:: cuda.cuda.cuMemsetD2D8
.. autofunction:: cuda.cuda.cuMemsetD2D16
.. autofunction:: cuda.cuda.cuMemsetD2D32
.. autofunction:: cuda.cuda.cuMemsetD8Async
.. autofunction:: cuda.cuda.cuMemsetD16Async
.. autofunction:: cuda.cuda.cuMemsetD32Async
.. autofunction:: cuda.cuda.cuMemsetD2D8Async
.. autofunction:: cuda.cuda.cuMemsetD2D16Async
.. autofunction:: cuda.cuda.cuMemsetD2D32Async
.. autofunction:: cuda.cuda.cuArrayCreate
.. autofunction:: cuda.cuda.cuArrayGetDescriptor
.. autofunction:: cuda.cuda.cuArrayGetSparseProperties
.. autofunction:: cuda.cuda.cuMipmappedArrayGetSparseProperties
.. autofunction:: cuda.cuda.cuArrayGetMemoryRequirements
.. autofunction:: cuda.cuda.cuMipmappedArrayGetMemoryRequirements
.. autofunction:: cuda.cuda.cuArrayGetPlane
.. autofunction:: cuda.cuda.cuArrayDestroy
.. autofunction:: cuda.cuda.cuArray3DCreate
.. autofunction:: cuda.cuda.cuArray3DGetDescriptor
.. autofunction:: cuda.cuda.cuMipmappedArrayCreate
.. autofunction:: cuda.cuda.cuMipmappedArrayGetLevel
.. autofunction:: cuda.cuda.cuMipmappedArrayDestroy

Virtual Memory Management
-------------------------

    
    


This section describes the virtual memory management functions of the low-level CUDA driver application programming interface. 
    

  
.. autofunction:: cuda.cuda.cuMemAddressReserve
.. autofunction:: cuda.cuda.cuMemAddressFree
.. autofunction:: cuda.cuda.cuMemCreate
.. autofunction:: cuda.cuda.cuMemRelease
.. autofunction:: cuda.cuda.cuMemMap
.. autofunction:: cuda.cuda.cuMemMapArrayAsync
.. autofunction:: cuda.cuda.cuMemUnmap
.. autofunction:: cuda.cuda.cuMemSetAccess
.. autofunction:: cuda.cuda.cuMemGetAccess
.. autofunction:: cuda.cuda.cuMemExportToShareableHandle
.. autofunction:: cuda.cuda.cuMemImportFromShareableHandle
.. autofunction:: cuda.cuda.cuMemGetAllocationGranularity
.. autofunction:: cuda.cuda.cuMemGetAllocationPropertiesFromHandle
.. autofunction:: cuda.cuda.cuMemRetainAllocationHandle

Stream Ordered Memory Allocator
-------------------------------

    
    


This section describes the stream ordered memory allocator exposed by the low-level CUDA driver application programming interface.


**overview**

The asynchronous allocator allows the user to allocate and free in stream order. All asynchronous accesses of the allocation must happen between the stream executions of the allocation and the free. If the memory is accessed outside of the promised stream order, a use before allocation / use after free error will cause undefined behavior.
The allocator is free to reallocate the memory as long as it can guarantee that compliant memory accesses will not overlap temporally. The allocator may refer to internal stream ordering as well as inter-stream dependencies (such as CUDA events and null stream dependencies) when establishing the temporal guarantee. The allocator may also insert inter-stream dependencies to establish the temporal guarantee.


**Supported Platforms**

Whether or not a device supports the integrated stream ordered memory allocator may be queried by calling cuDeviceGetAttribute() with the device attribute CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED 

    
  
.. autofunction:: cuda.cuda.cuMemFreeAsync
.. autofunction:: cuda.cuda.cuMemAllocAsync
.. autofunction:: cuda.cuda.cuMemPoolTrimTo
.. autofunction:: cuda.cuda.cuMemPoolSetAttribute
.. autofunction:: cuda.cuda.cuMemPoolGetAttribute
.. autofunction:: cuda.cuda.cuMemPoolSetAccess
.. autofunction:: cuda.cuda.cuMemPoolGetAccess
.. autofunction:: cuda.cuda.cuMemPoolCreate
.. autofunction:: cuda.cuda.cuMemPoolDestroy
.. autofunction:: cuda.cuda.cuMemAllocFromPoolAsync
.. autofunction:: cuda.cuda.cuMemPoolExportToShareableHandle
.. autofunction:: cuda.cuda.cuMemPoolImportFromShareableHandle
.. autofunction:: cuda.cuda.cuMemPoolExportPointer
.. autofunction:: cuda.cuda.cuMemPoolImportPointer

Unified Addressing
------------------

    
    


This section describes the unified addressing functions of the low-level CUDA driver application programming interface.


**Overview**

CUDA devices can share a unified address space with the host. For these devices there is no distinction between a device pointer and a host pointer -- the same pointer value may be used to access memory from the host program and from a kernel running on the device (with exceptions enumerated below).


**Supported Platforms**

Whether or not a device supports unified addressing may be queried by calling cuDeviceGetAttribute() with the device attribute CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING.
Unified addressing is automatically enabled in 64-bit processes


**Looking Up Information from Pointer Values**

It is possible to look up information about the memory which backs a pointer value. For instance, one may want to know if a pointer points to host or device memory. As another example, in the case of device memory, one may want to know on which CUDA device the memory resides. These properties may be queried using the function cuPointerGetAttribute()
Since pointers are unique, it is not necessary to specify information about the pointers specified to the various copy functions in the CUDA API. The function cuMemcpy() may be used to perform a copy between two pointers, ignoring whether they point to host or device memory (making cuMemcpyHtoD(), cuMemcpyDtoD(), and cuMemcpyDtoH() unnecessary for devices supporting unified addressing). For multidimensional copies, the memory type CU_MEMORYTYPE_UNIFIED may be used to specify that the CUDA driver should infer the location of the pointer from its value.


**Automatic Mapping of Host Allocated Host Memory**

All host memory allocated in all contexts using cuMemAllocHost() and cuMemHostAlloc() is always directly accessible from all contexts on all devices that support unified addressing. This is the case regardless of whether or not the flags CU_MEMHOSTALLOC_PORTABLE and CU_MEMHOSTALLOC_DEVICEMAP are specified.
The pointer value through which allocated host memory may be accessed in kernels on all devices that support unified addressing is the same as the pointer value through which that memory is accessed on the host, so it is not necessary to call cuMemHostGetDevicePointer() to get the device pointer for these allocations.
Note that this is not the case for memory allocated using the flag CU_MEMHOSTALLOC_WRITECOMBINED, as discussed below.


**Automatic Registration of Peer Memory**

Upon enabling direct access from a context that supports unified addressing to another peer context that supports unified addressing using cuCtxEnablePeerAccess() all memory allocated in the peer context using cuMemAlloc() and cuMemAllocPitch() will immediately be accessible by the current context. The device pointer value through which any peer memory may be accessed in the current context is the same pointer value through which that memory may be accessed in the peer context.


**Exceptions, Disjoint Addressing**

Not all memory may be accessed on devices through the same pointer value through which they are accessed on the host. These exceptions are host memory registered using cuMemHostRegister() and host memory allocated using the flag CU_MEMHOSTALLOC_WRITECOMBINED. For these exceptions, there exists a distinct host and device address for the memory. The device address is guaranteed to not overlap any valid host pointer range and is guaranteed to have the same value across all contexts that support unified addressing.
This device address may be queried using cuMemHostGetDevicePointer() when a context using unified addressing is current. Either the host or the unified device pointer value may be used to refer to this memory through cuMemcpy() and similar functions using the CU_MEMORYTYPE_UNIFIED memory type. 

    
  
.. autofunction:: cuda.cuda.cuPointerGetAttribute
.. autofunction:: cuda.cuda.cuMemPrefetchAsync
.. autofunction:: cuda.cuda.cuMemAdvise
.. autofunction:: cuda.cuda.cuMemRangeGetAttribute
.. autofunction:: cuda.cuda.cuMemRangeGetAttributes
.. autofunction:: cuda.cuda.cuPointerSetAttribute
.. autofunction:: cuda.cuda.cuPointerGetAttributes

Stream Management
-----------------

    
    


This section describes the stream management functions of the low-level CUDA driver application programming interface. 
    

  
.. autofunction:: cuda.cuda.cuStreamCreate
.. autofunction:: cuda.cuda.cuStreamCreateWithPriority
.. autofunction:: cuda.cuda.cuStreamGetPriority
.. autofunction:: cuda.cuda.cuStreamGetFlags
.. autofunction:: cuda.cuda.cuStreamGetCtx
.. autofunction:: cuda.cuda.cuStreamWaitEvent
.. autofunction:: cuda.cuda.cuStreamAddCallback
.. autofunction:: cuda.cuda.cuStreamBeginCapture
.. autofunction:: cuda.cuda.cuThreadExchangeStreamCaptureMode
.. autofunction:: cuda.cuda.cuStreamEndCapture
.. autofunction:: cuda.cuda.cuStreamIsCapturing
.. autofunction:: cuda.cuda.cuStreamGetCaptureInfo
.. autofunction:: cuda.cuda.cuStreamGetCaptureInfo_v2
.. autofunction:: cuda.cuda.cuStreamUpdateCaptureDependencies
.. autofunction:: cuda.cuda.cuStreamAttachMemAsync
.. autofunction:: cuda.cuda.cuStreamQuery
.. autofunction:: cuda.cuda.cuStreamSynchronize
.. autofunction:: cuda.cuda.cuStreamDestroy
.. autofunction:: cuda.cuda.cuStreamCopyAttributes
.. autofunction:: cuda.cuda.cuStreamGetAttribute
.. autofunction:: cuda.cuda.cuStreamSetAttribute

Event Management
----------------

    
    


This section describes the event management functions of the low-level CUDA driver application programming interface. 
    

  
.. autofunction:: cuda.cuda.cuEventCreate
.. autofunction:: cuda.cuda.cuEventRecord
.. autofunction:: cuda.cuda.cuEventRecordWithFlags
.. autofunction:: cuda.cuda.cuEventQuery
.. autofunction:: cuda.cuda.cuEventSynchronize
.. autofunction:: cuda.cuda.cuEventDestroy
.. autofunction:: cuda.cuda.cuEventElapsedTime

External Resource Interoperability
----------------------------------

    
    


This section describes the external resource interoperability functions of the low-level CUDA driver application programming interface. 
    

  
.. autofunction:: cuda.cuda.cuImportExternalMemory
.. autofunction:: cuda.cuda.cuExternalMemoryGetMappedBuffer
.. autofunction:: cuda.cuda.cuExternalMemoryGetMappedMipmappedArray
.. autofunction:: cuda.cuda.cuDestroyExternalMemory
.. autofunction:: cuda.cuda.cuImportExternalSemaphore
.. autofunction:: cuda.cuda.cuSignalExternalSemaphoresAsync
.. autofunction:: cuda.cuda.cuWaitExternalSemaphoresAsync
.. autofunction:: cuda.cuda.cuDestroyExternalSemaphore

Stream memory operations
------------------------

    
    


This section describes the stream memory operations of the low-level CUDA driver application programming interface.

The whole set of operations is disabled by default. Users are required to explicitly enable them, e.g. on Linux by passing the kernel module parameter shown below: modprobe nvidia NVreg_EnableStreamMemOPs=1 There is currently no way to enable these operations on other operating systems.

Users can programmatically query whether the device supports these operations with cuDeviceGetAttribute() and CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS.

Support for the CU_STREAM_WAIT_VALUE_NOR flag can be queried with CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR.

Support for the cuStreamWriteValue64() and cuStreamWaitValue64() functions, as well as for the CU_STREAM_MEM_OP_WAIT_VALUE_64 and CU_STREAM_MEM_OP_WRITE_VALUE_64 flags, can be queried with CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS.

Support for both CU_STREAM_WAIT_VALUE_FLUSH and CU_STREAM_MEM_OP_FLUSH_REMOTE_WRITES requires dedicated platform hardware features and can be queried with cuDeviceGetAttribute() and CU_DEVICE_ATTRIBUTE_CAN_FLUSH_REMOTE_WRITES.

Note that all memory pointers passed as parameters to these operations are device pointers. Where necessary a device pointer should be obtained, for example with cuMemHostGetDevicePointer().

None of the operations accepts pointers to managed memory buffers (cuMemAllocManaged). 
    

  
.. autofunction:: cuda.cuda.cuStreamWaitValue32
.. autofunction:: cuda.cuda.cuStreamWaitValue64
.. autofunction:: cuda.cuda.cuStreamWriteValue32
.. autofunction:: cuda.cuda.cuStreamWriteValue64
.. autofunction:: cuda.cuda.cuStreamBatchMemOp

Execution Control
-----------------

    
    


This section describes the execution control functions of the low-level CUDA driver application programming interface. 
    

  
.. autofunction:: cuda.cuda.cuFuncGetAttribute
.. autofunction:: cuda.cuda.cuFuncSetAttribute
.. autofunction:: cuda.cuda.cuFuncSetCacheConfig
.. autofunction:: cuda.cuda.cuFuncSetSharedMemConfig
.. autofunction:: cuda.cuda.cuFuncGetModule
.. autofunction:: cuda.cuda.cuLaunchKernel
.. autofunction:: cuda.cuda.cuLaunchCooperativeKernel
.. autofunction:: cuda.cuda.cuLaunchCooperativeKernelMultiDevice
.. autofunction:: cuda.cuda.cuLaunchHostFunc

Graph Management
----------------

    
    


This section describes the graph management functions of the low-level CUDA driver application programming interface. 
    

  
.. autofunction:: cuda.cuda.cuGraphCreate
.. autofunction:: cuda.cuda.cuGraphAddKernelNode
.. autofunction:: cuda.cuda.cuGraphKernelNodeGetParams
.. autofunction:: cuda.cuda.cuGraphKernelNodeSetParams
.. autofunction:: cuda.cuda.cuGraphAddMemcpyNode
.. autofunction:: cuda.cuda.cuGraphMemcpyNodeGetParams
.. autofunction:: cuda.cuda.cuGraphMemcpyNodeSetParams
.. autofunction:: cuda.cuda.cuGraphAddMemsetNode
.. autofunction:: cuda.cuda.cuGraphMemsetNodeGetParams
.. autofunction:: cuda.cuda.cuGraphMemsetNodeSetParams
.. autofunction:: cuda.cuda.cuGraphAddHostNode
.. autofunction:: cuda.cuda.cuGraphHostNodeGetParams
.. autofunction:: cuda.cuda.cuGraphHostNodeSetParams
.. autofunction:: cuda.cuda.cuGraphAddChildGraphNode
.. autofunction:: cuda.cuda.cuGraphChildGraphNodeGetGraph
.. autofunction:: cuda.cuda.cuGraphAddEmptyNode
.. autofunction:: cuda.cuda.cuGraphAddEventRecordNode
.. autofunction:: cuda.cuda.cuGraphEventRecordNodeGetEvent
.. autofunction:: cuda.cuda.cuGraphEventRecordNodeSetEvent
.. autofunction:: cuda.cuda.cuGraphAddEventWaitNode
.. autofunction:: cuda.cuda.cuGraphEventWaitNodeGetEvent
.. autofunction:: cuda.cuda.cuGraphEventWaitNodeSetEvent
.. autofunction:: cuda.cuda.cuGraphAddExternalSemaphoresSignalNode
.. autofunction:: cuda.cuda.cuGraphExternalSemaphoresSignalNodeGetParams
.. autofunction:: cuda.cuda.cuGraphExternalSemaphoresSignalNodeSetParams
.. autofunction:: cuda.cuda.cuGraphAddExternalSemaphoresWaitNode
.. autofunction:: cuda.cuda.cuGraphExternalSemaphoresWaitNodeGetParams
.. autofunction:: cuda.cuda.cuGraphExternalSemaphoresWaitNodeSetParams
.. autofunction:: cuda.cuda.cuGraphAddMemAllocNode
.. autofunction:: cuda.cuda.cuGraphMemAllocNodeGetParams
.. autofunction:: cuda.cuda.cuGraphAddMemFreeNode
.. autofunction:: cuda.cuda.cuGraphMemFreeNodeGetParams
.. autofunction:: cuda.cuda.cuDeviceGraphMemTrim
.. autofunction:: cuda.cuda.cuDeviceGetGraphMemAttribute
.. autofunction:: cuda.cuda.cuDeviceSetGraphMemAttribute
.. autofunction:: cuda.cuda.cuGraphClone
.. autofunction:: cuda.cuda.cuGraphNodeFindInClone
.. autofunction:: cuda.cuda.cuGraphNodeGetType
.. autofunction:: cuda.cuda.cuGraphGetNodes
.. autofunction:: cuda.cuda.cuGraphGetRootNodes
.. autofunction:: cuda.cuda.cuGraphGetEdges
.. autofunction:: cuda.cuda.cuGraphNodeGetDependencies
.. autofunction:: cuda.cuda.cuGraphNodeGetDependentNodes
.. autofunction:: cuda.cuda.cuGraphAddDependencies
.. autofunction:: cuda.cuda.cuGraphRemoveDependencies
.. autofunction:: cuda.cuda.cuGraphDestroyNode
.. autofunction:: cuda.cuda.cuGraphInstantiate
.. autofunction:: cuda.cuda.cuGraphInstantiateWithFlags
.. autofunction:: cuda.cuda.cuGraphExecKernelNodeSetParams
.. autofunction:: cuda.cuda.cuGraphExecMemcpyNodeSetParams
.. autofunction:: cuda.cuda.cuGraphExecMemsetNodeSetParams
.. autofunction:: cuda.cuda.cuGraphExecHostNodeSetParams
.. autofunction:: cuda.cuda.cuGraphExecChildGraphNodeSetParams
.. autofunction:: cuda.cuda.cuGraphExecEventRecordNodeSetEvent
.. autofunction:: cuda.cuda.cuGraphExecEventWaitNodeSetEvent
.. autofunction:: cuda.cuda.cuGraphExecExternalSemaphoresSignalNodeSetParams
.. autofunction:: cuda.cuda.cuGraphExecExternalSemaphoresWaitNodeSetParams
.. autofunction:: cuda.cuda.cuGraphNodeSetEnabled
.. autofunction:: cuda.cuda.cuGraphNodeGetEnabled
.. autofunction:: cuda.cuda.cuGraphUpload
.. autofunction:: cuda.cuda.cuGraphLaunch
.. autofunction:: cuda.cuda.cuGraphExecDestroy
.. autofunction:: cuda.cuda.cuGraphDestroy
.. autofunction:: cuda.cuda.cuGraphExecUpdate
.. autofunction:: cuda.cuda.cuGraphKernelNodeCopyAttributes
.. autofunction:: cuda.cuda.cuGraphKernelNodeGetAttribute
.. autofunction:: cuda.cuda.cuGraphKernelNodeSetAttribute
.. autofunction:: cuda.cuda.cuGraphDebugDotPrint
.. autofunction:: cuda.cuda.cuUserObjectCreate
.. autofunction:: cuda.cuda.cuUserObjectRetain
.. autofunction:: cuda.cuda.cuUserObjectRelease
.. autofunction:: cuda.cuda.cuGraphRetainUserObject
.. autofunction:: cuda.cuda.cuGraphReleaseUserObject

Occupancy
---------

    
    


This section describes the occupancy calculation functions of the low-level CUDA driver application programming interface. 
    

  
.. autofunction:: cuda.cuda.cuOccupancyMaxActiveBlocksPerMultiprocessor
.. autofunction:: cuda.cuda.cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags
.. autofunction:: cuda.cuda.cuOccupancyMaxPotentialBlockSize
.. autofunction:: cuda.cuda.cuOccupancyMaxPotentialBlockSizeWithFlags
.. autofunction:: cuda.cuda.cuOccupancyAvailableDynamicSMemPerBlock

Texture Object Management
-------------------------

    
    


This section describes the texture object management functions of the low-level CUDA driver application programming interface. The texture object API is only supported on devices of compute capability 3.0 or higher. 
    

  
.. autofunction:: cuda.cuda.cuTexObjectCreate
.. autofunction:: cuda.cuda.cuTexObjectDestroy
.. autofunction:: cuda.cuda.cuTexObjectGetResourceDesc
.. autofunction:: cuda.cuda.cuTexObjectGetTextureDesc
.. autofunction:: cuda.cuda.cuTexObjectGetResourceViewDesc

Surface Object Management
-------------------------

    
    


This section describes the surface object management functions of the low-level CUDA driver application programming interface. The surface object API is only supported on devices of compute capability 3.0 or higher. 
    

  
.. autofunction:: cuda.cuda.cuSurfObjectCreate
.. autofunction:: cuda.cuda.cuSurfObjectDestroy
.. autofunction:: cuda.cuda.cuSurfObjectGetResourceDesc

Peer Context Memory Access
--------------------------

    
    


This section describes the direct peer context memory access functions of the low-level CUDA driver application programming interface. 
    

  
.. autofunction:: cuda.cuda.cuDeviceCanAccessPeer
.. autofunction:: cuda.cuda.cuCtxEnablePeerAccess
.. autofunction:: cuda.cuda.cuCtxDisablePeerAccess
.. autofunction:: cuda.cuda.cuDeviceGetP2PAttribute

Graphics Interoperability
-------------------------

    
    


This section describes the graphics interoperability functions of the low-level CUDA driver application programming interface. 
    

  
.. autofunction:: cuda.cuda.cuGraphicsUnregisterResource
.. autofunction:: cuda.cuda.cuGraphicsSubResourceGetMappedArray
.. autofunction:: cuda.cuda.cuGraphicsResourceGetMappedMipmappedArray
.. autofunction:: cuda.cuda.cuGraphicsResourceGetMappedPointer
.. autofunction:: cuda.cuda.cuGraphicsResourceSetMapFlags
.. autofunction:: cuda.cuda.cuGraphicsMapResources
.. autofunction:: cuda.cuda.cuGraphicsUnmapResources

Driver Entry Point Access
-------------------------

    
    


This section describes the driver entry point access functions of the low-level CUDA driver application programming interface. 
    

  
.. autofunction:: cuda.cuda.cuGetProcAddress

EGL Interoperability
--------------------

    
    


This section describes the EGL interoperability functions of the low-level CUDA driver application programming interface. 
    

  
.. autofunction:: cuda.cuda.cuGraphicsEGLRegisterImage
.. autofunction:: cuda.cuda.cuEGLStreamConsumerConnect
.. autofunction:: cuda.cuda.cuEGLStreamConsumerConnectWithFlags
.. autofunction:: cuda.cuda.cuEGLStreamConsumerDisconnect
.. autofunction:: cuda.cuda.cuEGLStreamConsumerAcquireFrame
.. autofunction:: cuda.cuda.cuEGLStreamConsumerReleaseFrame
.. autofunction:: cuda.cuda.cuEGLStreamProducerConnect
.. autofunction:: cuda.cuda.cuEGLStreamProducerDisconnect
.. autofunction:: cuda.cuda.cuEGLStreamProducerPresentFrame
.. autofunction:: cuda.cuda.cuEGLStreamProducerReturnFrame
.. autofunction:: cuda.cuda.cuGraphicsResourceGetMappedEglFrame
.. autofunction:: cuda.cuda.cuEventCreateFromEGLSync

OpenGL Interoperability
-----------------------

    
    


This section describes the OpenGL interoperability functions of the low-level CUDA driver application programming interface. Note that mapping of OpenGL resources is performed with the graphics API agnostic, resource mapping interface described in Graphics Interoperability. 
    

  
.. autoenum:: cuda.cuda.CUGLDeviceList
.. autofunction:: cuda.cuda.cuGraphicsGLRegisterBuffer
.. autofunction:: cuda.cuda.cuGraphicsGLRegisterImage
.. autofunction:: cuda.cuda.cuGLGetDevices

Profiler Control
----------------

    
    


This section describes the profiler control functions of the low-level CUDA driver application programming interface. 
    

  
.. autofunction:: cuda.cuda.cuProfilerStart
.. autofunction:: cuda.cuda.cuProfilerStop

VDPAU Interoperability
----------------------

    
    


This section describes the VDPAU interoperability functions of the low-level CUDA driver application programming interface. 
    

  
.. autofunction:: cuda.cuda.cuVDPAUGetDevice
.. autofunction:: cuda.cuda.cuVDPAUCtxCreate
.. autofunction:: cuda.cuda.cuGraphicsVDPAURegisterVideoSurface
.. autofunction:: cuda.cuda.cuGraphicsVDPAURegisterOutputSurface
