------
cudart
------

Profiler Control
----------------

    
    


This section describes the profiler control functions of the CUDA runtime application programming interface. 
    

  
.. autofunction:: cuda.cudart.cudaProfilerStart
.. autofunction:: cuda.cudart.cudaProfilerStop

Device Management
-----------------

    
    

impl_private


This section describes the device management functions of the CUDA runtime application programming interface. 
    

  
.. autofunction:: cuda.cudart.cudaDeviceReset
.. autofunction:: cuda.cudart.cudaDeviceSynchronize
.. autofunction:: cuda.cudart.cudaDeviceSetLimit
.. autofunction:: cuda.cudart.cudaDeviceGetLimit
.. autofunction:: cuda.cudart.cudaDeviceGetTexture1DLinearMaxWidth
.. autofunction:: cuda.cudart.cudaDeviceGetCacheConfig
.. autofunction:: cuda.cudart.cudaDeviceGetStreamPriorityRange
.. autofunction:: cuda.cudart.cudaDeviceSetCacheConfig
.. autofunction:: cuda.cudart.cudaDeviceGetSharedMemConfig
.. autofunction:: cuda.cudart.cudaDeviceSetSharedMemConfig
.. autofunction:: cuda.cudart.cudaDeviceGetByPCIBusId
.. autofunction:: cuda.cudart.cudaDeviceGetPCIBusId
.. autofunction:: cuda.cudart.cudaIpcGetEventHandle
.. autofunction:: cuda.cudart.cudaIpcOpenEventHandle
.. autofunction:: cuda.cudart.cudaIpcGetMemHandle
.. autofunction:: cuda.cudart.cudaIpcOpenMemHandle
.. autofunction:: cuda.cudart.cudaIpcCloseMemHandle
.. autofunction:: cuda.cudart.cudaDeviceFlushGPUDirectRDMAWrites
.. autofunction:: cuda.cudart.cudaGetDeviceCount
.. autofunction:: cuda.cudart.cudaGetDeviceProperties
.. autofunction:: cuda.cudart.cudaDeviceGetAttribute
.. autofunction:: cuda.cudart.cudaDeviceGetDefaultMemPool
.. autofunction:: cuda.cudart.cudaDeviceSetMemPool
.. autofunction:: cuda.cudart.cudaDeviceGetMemPool
.. autofunction:: cuda.cudart.cudaDeviceGetNvSciSyncAttributes
.. autofunction:: cuda.cudart.cudaDeviceGetP2PAttribute
.. autofunction:: cuda.cudart.cudaChooseDevice
.. autofunction:: cuda.cudart.cudaSetDevice
.. autofunction:: cuda.cudart.cudaGetDevice
.. autofunction:: cuda.cudart.cudaSetDeviceFlags
.. autofunction:: cuda.cudart.cudaGetDeviceFlags

Error Handling
--------------

    
    


This section describes the error handling functions of the CUDA runtime application programming interface. 
    

  
.. autofunction:: cuda.cudart.cudaGetLastError
.. autofunction:: cuda.cudart.cudaPeekAtLastError
.. autofunction:: cuda.cudart.cudaGetErrorName
.. autofunction:: cuda.cudart.cudaGetErrorString

Stream Management
-----------------

    
    


This section describes the stream management functions of the CUDA runtime application programming interface. 
    

  
.. autoclass:: cuda.cudart.cudaStreamCallback_t
.. autofunction:: cuda.cudart.cudaStreamCreate
.. autofunction:: cuda.cudart.cudaStreamCreateWithFlags
.. autofunction:: cuda.cudart.cudaStreamCreateWithPriority
.. autofunction:: cuda.cudart.cudaStreamGetPriority
.. autofunction:: cuda.cudart.cudaStreamGetFlags
.. autofunction:: cuda.cudart.cudaCtxResetPersistingL2Cache
.. autofunction:: cuda.cudart.cudaStreamCopyAttributes
.. autofunction:: cuda.cudart.cudaStreamGetAttribute
.. autofunction:: cuda.cudart.cudaStreamSetAttribute
.. autofunction:: cuda.cudart.cudaStreamDestroy
.. autofunction:: cuda.cudart.cudaStreamWaitEvent
.. autofunction:: cuda.cudart.cudaStreamAddCallback
.. autofunction:: cuda.cudart.cudaStreamSynchronize
.. autofunction:: cuda.cudart.cudaStreamQuery
.. autofunction:: cuda.cudart.cudaStreamAttachMemAsync
.. autofunction:: cuda.cudart.cudaStreamBeginCapture
.. autofunction:: cuda.cudart.cudaThreadExchangeStreamCaptureMode
.. autofunction:: cuda.cudart.cudaStreamEndCapture
.. autofunction:: cuda.cudart.cudaStreamIsCapturing
.. autofunction:: cuda.cudart.cudaStreamGetCaptureInfo
.. autofunction:: cuda.cudart.cudaStreamGetCaptureInfo_v2
.. autofunction:: cuda.cudart.cudaStreamUpdateCaptureDependencies

Event Management
----------------

    
    


This section describes the event management functions of the CUDA runtime application programming interface. 
    

  
.. autofunction:: cuda.cudart.cudaEventCreate
.. autofunction:: cuda.cudart.cudaEventCreateWithFlags
.. autofunction:: cuda.cudart.cudaEventRecord
.. autofunction:: cuda.cudart.cudaEventRecordWithFlags
.. autofunction:: cuda.cudart.cudaEventQuery
.. autofunction:: cuda.cudart.cudaEventSynchronize
.. autofunction:: cuda.cudart.cudaEventDestroy
.. autofunction:: cuda.cudart.cudaEventElapsedTime

External Resource Interoperability
----------------------------------

    
    


This section describes the external resource interoperability functions of the CUDA runtime application programming interface. 
    

  
.. autofunction:: cuda.cudart.cudaImportExternalMemory
.. autofunction:: cuda.cudart.cudaExternalMemoryGetMappedBuffer
.. autofunction:: cuda.cudart.cudaExternalMemoryGetMappedMipmappedArray
.. autofunction:: cuda.cudart.cudaDestroyExternalMemory
.. autofunction:: cuda.cudart.cudaImportExternalSemaphore
.. autofunction:: cuda.cudart.cudaSignalExternalSemaphoresAsync
.. autofunction:: cuda.cudart.cudaWaitExternalSemaphoresAsync
.. autofunction:: cuda.cudart.cudaDestroyExternalSemaphore

Execution Control
-----------------

    
    


This section describes the execution control functions of the CUDA runtime application programming interface.

Some functions have overloaded C++ API template versions documented separately in the C++ API Routines module. 
    

  
.. autofunction:: cuda.cudart.cudaFuncSetCacheConfig
.. autofunction:: cuda.cudart.cudaFuncSetSharedMemConfig
.. autofunction:: cuda.cudart.cudaFuncGetAttributes
.. autofunction:: cuda.cudart.cudaFuncSetAttribute
.. autofunction:: cuda.cudart.cudaSetDoubleForDevice
.. autofunction:: cuda.cudart.cudaSetDoubleForHost
.. autofunction:: cuda.cudart.cudaLaunchHostFunc

Occupancy
---------

    
    


This section describes the occupancy calculation functions of the CUDA runtime application programming interface.

Besides the occupancy calculator functions (cudaOccupancyMaxActiveBlocksPerMultiprocessor and cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags), there are also C++ only occupancy-based launch configuration functions documented in C++ API Routines module.

See cudaOccupancyMaxPotentialBlockSize (C++ API), cudaOccupancyMaxPotentialBlockSize (C++ API), cudaOccupancyMaxPotentialBlockSizeVariableSMem (C++ API), cudaOccupancyMaxPotentialBlockSizeVariableSMem (C++ API) cudaOccupancyAvailableDynamicSMemPerBlock (C++ API), 
    

  
.. autofunction:: cuda.cudart.cudaOccupancyMaxActiveBlocksPerMultiprocessor
.. autofunction:: cuda.cudart.cudaOccupancyAvailableDynamicSMemPerBlock
.. autofunction:: cuda.cudart.cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags

Memory Management
-----------------

    
    


This section describes the memory management functions of the CUDA runtime application programming interface.

Some functions have overloaded C++ API template versions documented separately in the C++ API Routines module. 
    

  
.. autofunction:: cuda.cudart.cudaMallocManaged
.. autofunction:: cuda.cudart.cudaMalloc
.. autofunction:: cuda.cudart.cudaMallocHost
.. autofunction:: cuda.cudart.cudaMallocPitch
.. autofunction:: cuda.cudart.cudaMallocArray
.. autofunction:: cuda.cudart.cudaFree
.. autofunction:: cuda.cudart.cudaFreeHost
.. autofunction:: cuda.cudart.cudaFreeArray
.. autofunction:: cuda.cudart.cudaFreeMipmappedArray
.. autofunction:: cuda.cudart.cudaHostAlloc
.. autofunction:: cuda.cudart.cudaHostRegister
.. autofunction:: cuda.cudart.cudaHostUnregister
.. autofunction:: cuda.cudart.cudaHostGetDevicePointer
.. autofunction:: cuda.cudart.cudaHostGetFlags
.. autofunction:: cuda.cudart.cudaMalloc3D
.. autofunction:: cuda.cudart.cudaMalloc3DArray
.. autofunction:: cuda.cudart.cudaMallocMipmappedArray
.. autofunction:: cuda.cudart.cudaGetMipmappedArrayLevel
.. autofunction:: cuda.cudart.cudaMemcpy3D
.. autofunction:: cuda.cudart.cudaMemcpy3DPeer
.. autofunction:: cuda.cudart.cudaMemcpy3DAsync
.. autofunction:: cuda.cudart.cudaMemcpy3DPeerAsync
.. autofunction:: cuda.cudart.cudaMemGetInfo
.. autofunction:: cuda.cudart.cudaArrayGetInfo
.. autofunction:: cuda.cudart.cudaArrayGetPlane
.. autofunction:: cuda.cudart.cudaArrayGetMemoryRequirements
.. autofunction:: cuda.cudart.cudaMipmappedArrayGetMemoryRequirements
.. autofunction:: cuda.cudart.cudaArrayGetSparseProperties
.. autofunction:: cuda.cudart.cudaMipmappedArrayGetSparseProperties
.. autofunction:: cuda.cudart.cudaMemcpy
.. autofunction:: cuda.cudart.cudaMemcpyPeer
.. autofunction:: cuda.cudart.cudaMemcpy2D
.. autofunction:: cuda.cudart.cudaMemcpy2DToArray
.. autofunction:: cuda.cudart.cudaMemcpy2DFromArray
.. autofunction:: cuda.cudart.cudaMemcpy2DArrayToArray
.. autofunction:: cuda.cudart.cudaMemcpyAsync
.. autofunction:: cuda.cudart.cudaMemcpyPeerAsync
.. autofunction:: cuda.cudart.cudaMemcpy2DAsync
.. autofunction:: cuda.cudart.cudaMemcpy2DToArrayAsync
.. autofunction:: cuda.cudart.cudaMemcpy2DFromArrayAsync
.. autofunction:: cuda.cudart.cudaMemset
.. autofunction:: cuda.cudart.cudaMemset2D
.. autofunction:: cuda.cudart.cudaMemset3D
.. autofunction:: cuda.cudart.cudaMemsetAsync
.. autofunction:: cuda.cudart.cudaMemset2DAsync
.. autofunction:: cuda.cudart.cudaMemset3DAsync
.. autofunction:: cuda.cudart.cudaMemPrefetchAsync
.. autofunction:: cuda.cudart.cudaMemAdvise
.. autofunction:: cuda.cudart.cudaMemRangeGetAttribute
.. autofunction:: cuda.cudart.cudaMemRangeGetAttributes

Stream Ordered Memory Allocator
-------------------------------

    
    



**overview**

The asynchronous allocator allows the user to allocate and free in stream order. All asynchronous accesses of the allocation must happen between the stream executions of the allocation and the free. If the memory is accessed outside of the promised stream order, a use before allocation / use after free error will cause undefined behavior.
The allocator is free to reallocate the memory as long as it can guarantee that compliant memory accesses will not overlap temporally. The allocator may refer to internal stream ordering as well as inter-stream dependencies (such as CUDA events and null stream dependencies) when establishing the temporal guarantee. The allocator may also insert inter-stream dependencies to establish the temporal guarantee.


**Supported Platforms**

Whether or not a device supports the integrated stream ordered memory allocator may be queried by calling cudaDeviceGetAttribute() with the device attribute ::cudaDevAttrMemoryPoolsSupported. 

    
  
.. autofunction:: cuda.cudart.cudaMallocAsync
.. autofunction:: cuda.cudart.cudaFreeAsync
.. autofunction:: cuda.cudart.cudaMemPoolTrimTo
.. autofunction:: cuda.cudart.cudaMemPoolSetAttribute
.. autofunction:: cuda.cudart.cudaMemPoolGetAttribute
.. autofunction:: cuda.cudart.cudaMemPoolSetAccess
.. autofunction:: cuda.cudart.cudaMemPoolGetAccess
.. autofunction:: cuda.cudart.cudaMemPoolCreate
.. autofunction:: cuda.cudart.cudaMemPoolDestroy
.. autofunction:: cuda.cudart.cudaMallocFromPoolAsync
.. autofunction:: cuda.cudart.cudaMemPoolExportToShareableHandle
.. autofunction:: cuda.cudart.cudaMemPoolImportFromShareableHandle
.. autofunction:: cuda.cudart.cudaMemPoolExportPointer
.. autofunction:: cuda.cudart.cudaMemPoolImportPointer

Unified Addressing
------------------

    
    


This section describes the unified addressing functions of the CUDA runtime application programming interface.


**Overview**

CUDA devices can share a unified address space with the host. 
 For these devices there is no distinction between a device pointer and a host pointer -- the same pointer value may be used to access memory from the host program and from a kernel running on the device (with exceptions enumerated below).


**Supported Platforms**

Whether or not a device supports unified addressing may be queried by calling cudaGetDeviceProperties() with the device property cudaDeviceProp::unifiedAddressing.
Unified addressing is automatically enabled in 64-bit processes .


**Looking Up Information from Pointer Values**

It is possible to look up information about the memory which backs a pointer value. For instance, one may want to know if a pointer points to host or device memory. As another example, in the case of device memory, one may want to know on which CUDA device the memory resides. These properties may be queried using the function cudaPointerGetAttributes()
Since pointers are unique, it is not necessary to specify information about the pointers specified to cudaMemcpy() and other copy functions. 
 The copy direction ::cudaMemcpyDefault may be used to specify that the CUDA runtime should infer the location of the pointer from its value.


**Automatic Mapping of Host Allocated Host Memory**

All host memory allocated through all devices using cudaMallocHost() and cudaHostAlloc() is always directly accessible from all devices that support unified addressing. This is the case regardless of whether or not the flags cudaHostAllocPortable and cudaHostAllocMapped are specified.
The pointer value through which allocated host memory may be accessed in kernels on all devices that support unified addressing is the same as the pointer value through which that memory is accessed on the host. It is not necessary to call cudaHostGetDevicePointer() to get the device pointer for these allocations. 

Note that this is not the case for memory allocated using the flag cudaHostAllocWriteCombined, as discussed below.


**Direct Access of Peer Memory**

Upon enabling direct access from a device that supports unified addressing to another peer device that supports unified addressing using cudaDeviceEnablePeerAccess() all memory allocated in the peer device using cudaMalloc() and cudaMallocPitch() will immediately be accessible by the current device. The device pointer value through which any peer's memory may be accessed in the current device is the same pointer value through which that memory may be accessed from the peer device.


**Exceptions, Disjoint Addressing**

Not all memory may be accessed on devices through the same pointer value through which they are accessed on the host. These exceptions are host memory registered using cudaHostRegister() and host memory allocated using the flag cudaHostAllocWriteCombined. For these exceptions, there exists a distinct host and device address for the memory. The device address is guaranteed to not overlap any valid host pointer range and is guaranteed to have the same value across all devices that support unified addressing. 

This device address may be queried using cudaHostGetDevicePointer() when a device using unified addressing is current. Either the host or the unified device pointer value may be used to refer to this memory in cudaMemcpy() and similar functions using the ::cudaMemcpyDefault memory direction. 

    
  
.. autofunction:: cuda.cudart.cudaPointerGetAttributes

Peer Device Memory Access
-------------------------

    
    


This section describes the peer device memory access functions of the CUDA runtime application programming interface. 
    

  
.. autofunction:: cuda.cudart.cudaDeviceCanAccessPeer
.. autofunction:: cuda.cudart.cudaDeviceEnablePeerAccess
.. autofunction:: cuda.cudart.cudaDeviceDisablePeerAccess

OpenGL Interoperability
-----------------------

    
    

impl_private

This section describes the OpenGL interoperability functions of the CUDA runtime application programming interface. Note that mapping of OpenGL resources is performed with the graphics API agnostic, resource mapping interface described in Graphics Interopability. 
    

  
.. autoenum:: cuda.cudart.cudaGLDeviceList
.. autofunction:: cuda.cudart.cudaGLGetDevices
.. autofunction:: cuda.cudart.cudaGraphicsGLRegisterImage
.. autofunction:: cuda.cudart.cudaGraphicsGLRegisterBuffer

Direct3D 9 Interoperability
---------------------------

    
    

    
  

Direct3D 10 Interoperability
----------------------------

    
    

    
  

Direct3D 11 Interoperability
----------------------------

    
    

    
  

VDPAU Interoperability
----------------------

    
    

This section describes the VDPAU interoperability functions of the CUDA runtime application programming interface. 
    

  
.. autofunction:: cuda.cudart.cudaVDPAUGetDevice
.. autofunction:: cuda.cudart.cudaVDPAUSetVDPAUDevice
.. autofunction:: cuda.cudart.cudaGraphicsVDPAURegisterVideoSurface
.. autofunction:: cuda.cudart.cudaGraphicsVDPAURegisterOutputSurface

EGL Interoperability
--------------------

    
    

This section describes the EGL interoperability functions of the CUDA runtime application programming interface. 
    

  
.. autofunction:: cuda.cudart.cudaGraphicsEGLRegisterImage
.. autofunction:: cuda.cudart.cudaEGLStreamConsumerConnect
.. autofunction:: cuda.cudart.cudaEGLStreamConsumerConnectWithFlags
.. autofunction:: cuda.cudart.cudaEGLStreamConsumerDisconnect
.. autofunction:: cuda.cudart.cudaEGLStreamConsumerAcquireFrame
.. autofunction:: cuda.cudart.cudaEGLStreamConsumerReleaseFrame
.. autofunction:: cuda.cudart.cudaEGLStreamProducerConnect
.. autofunction:: cuda.cudart.cudaEGLStreamProducerDisconnect
.. autofunction:: cuda.cudart.cudaEGLStreamProducerPresentFrame
.. autofunction:: cuda.cudart.cudaEGLStreamProducerReturnFrame
.. autofunction:: cuda.cudart.cudaGraphicsResourceGetMappedEglFrame
.. autofunction:: cuda.cudart.cudaEventCreateFromEGLSync

Graphics Interoperability
-------------------------

    
    


This section describes the graphics interoperability functions of the CUDA runtime application programming interface. 
    

  
.. autofunction:: cuda.cudart.cudaGraphicsUnregisterResource
.. autofunction:: cuda.cudart.cudaGraphicsResourceSetMapFlags
.. autofunction:: cuda.cudart.cudaGraphicsMapResources
.. autofunction:: cuda.cudart.cudaGraphicsUnmapResources
.. autofunction:: cuda.cudart.cudaGraphicsResourceGetMappedPointer
.. autofunction:: cuda.cudart.cudaGraphicsSubResourceGetMappedArray
.. autofunction:: cuda.cudart.cudaGraphicsResourceGetMappedMipmappedArray

Texture Object Management
-------------------------

    
    


This section describes the low level texture object management functions of the CUDA runtime application programming interface. The texture object API is only supported on devices of compute capability 3.0 or higher. 
    

  
.. autofunction:: cuda.cudart.cudaGetChannelDesc
.. autofunction:: cuda.cudart.cudaCreateChannelDesc
.. autofunction:: cuda.cudart.cudaCreateTextureObject
.. autofunction:: cuda.cudart.cudaDestroyTextureObject
.. autofunction:: cuda.cudart.cudaGetTextureObjectResourceDesc
.. autofunction:: cuda.cudart.cudaGetTextureObjectTextureDesc
.. autofunction:: cuda.cudart.cudaGetTextureObjectResourceViewDesc

Surface Object Management
-------------------------

    
    


This section describes the low level texture object management functions of the CUDA runtime application programming interface. The surface object API is only supported on devices of compute capability 3.0 or higher. 
    

  
.. autofunction:: cuda.cudart.cudaCreateSurfaceObject
.. autofunction:: cuda.cudart.cudaDestroySurfaceObject
.. autofunction:: cuda.cudart.cudaGetSurfaceObjectResourceDesc

Version Management
------------------

    
    

    
  
.. autofunction:: cuda.cudart.cudaDriverGetVersion
.. autofunction:: cuda.cudart.cudaRuntimeGetVersion

Graph Management
----------------

    
    


This section describes the graph management functions of CUDA runtime application programming interface. 
    

  
.. autofunction:: cuda.cudart.cudaGraphCreate
.. autofunction:: cuda.cudart.cudaGraphAddKernelNode
.. autofunction:: cuda.cudart.cudaGraphKernelNodeGetParams
.. autofunction:: cuda.cudart.cudaGraphKernelNodeSetParams
.. autofunction:: cuda.cudart.cudaGraphKernelNodeCopyAttributes
.. autofunction:: cuda.cudart.cudaGraphKernelNodeGetAttribute
.. autofunction:: cuda.cudart.cudaGraphKernelNodeSetAttribute
.. autofunction:: cuda.cudart.cudaGraphAddMemcpyNode
.. autofunction:: cuda.cudart.cudaGraphAddMemcpyNode1D
.. autofunction:: cuda.cudart.cudaGraphMemcpyNodeGetParams
.. autofunction:: cuda.cudart.cudaGraphMemcpyNodeSetParams
.. autofunction:: cuda.cudart.cudaGraphMemcpyNodeSetParams1D
.. autofunction:: cuda.cudart.cudaGraphAddMemsetNode
.. autofunction:: cuda.cudart.cudaGraphMemsetNodeGetParams
.. autofunction:: cuda.cudart.cudaGraphMemsetNodeSetParams
.. autofunction:: cuda.cudart.cudaGraphAddHostNode
.. autofunction:: cuda.cudart.cudaGraphHostNodeGetParams
.. autofunction:: cuda.cudart.cudaGraphHostNodeSetParams
.. autofunction:: cuda.cudart.cudaGraphAddChildGraphNode
.. autofunction:: cuda.cudart.cudaGraphChildGraphNodeGetGraph
.. autofunction:: cuda.cudart.cudaGraphAddEmptyNode
.. autofunction:: cuda.cudart.cudaGraphAddEventRecordNode
.. autofunction:: cuda.cudart.cudaGraphEventRecordNodeGetEvent
.. autofunction:: cuda.cudart.cudaGraphEventRecordNodeSetEvent
.. autofunction:: cuda.cudart.cudaGraphAddEventWaitNode
.. autofunction:: cuda.cudart.cudaGraphEventWaitNodeGetEvent
.. autofunction:: cuda.cudart.cudaGraphEventWaitNodeSetEvent
.. autofunction:: cuda.cudart.cudaGraphAddExternalSemaphoresSignalNode
.. autofunction:: cuda.cudart.cudaGraphExternalSemaphoresSignalNodeGetParams
.. autofunction:: cuda.cudart.cudaGraphExternalSemaphoresSignalNodeSetParams
.. autofunction:: cuda.cudart.cudaGraphAddExternalSemaphoresWaitNode
.. autofunction:: cuda.cudart.cudaGraphExternalSemaphoresWaitNodeGetParams
.. autofunction:: cuda.cudart.cudaGraphExternalSemaphoresWaitNodeSetParams
.. autofunction:: cuda.cudart.cudaGraphAddMemAllocNode
.. autofunction:: cuda.cudart.cudaGraphMemAllocNodeGetParams
.. autofunction:: cuda.cudart.cudaGraphAddMemFreeNode
.. autofunction:: cuda.cudart.cudaGraphMemFreeNodeGetParams
.. autofunction:: cuda.cudart.cudaDeviceGraphMemTrim
.. autofunction:: cuda.cudart.cudaDeviceGetGraphMemAttribute
.. autofunction:: cuda.cudart.cudaDeviceSetGraphMemAttribute
.. autofunction:: cuda.cudart.cudaGraphClone
.. autofunction:: cuda.cudart.cudaGraphNodeFindInClone
.. autofunction:: cuda.cudart.cudaGraphNodeGetType
.. autofunction:: cuda.cudart.cudaGraphGetNodes
.. autofunction:: cuda.cudart.cudaGraphGetRootNodes
.. autofunction:: cuda.cudart.cudaGraphGetEdges
.. autofunction:: cuda.cudart.cudaGraphNodeGetDependencies
.. autofunction:: cuda.cudart.cudaGraphNodeGetDependentNodes
.. autofunction:: cuda.cudart.cudaGraphAddDependencies
.. autofunction:: cuda.cudart.cudaGraphRemoveDependencies
.. autofunction:: cuda.cudart.cudaGraphDestroyNode
.. autofunction:: cuda.cudart.cudaGraphInstantiate
.. autofunction:: cuda.cudart.cudaGraphInstantiateWithFlags
.. autofunction:: cuda.cudart.cudaGraphExecKernelNodeSetParams
.. autofunction:: cuda.cudart.cudaGraphExecMemcpyNodeSetParams
.. autofunction:: cuda.cudart.cudaGraphExecMemcpyNodeSetParams1D
.. autofunction:: cuda.cudart.cudaGraphExecMemsetNodeSetParams
.. autofunction:: cuda.cudart.cudaGraphExecHostNodeSetParams
.. autofunction:: cuda.cudart.cudaGraphExecChildGraphNodeSetParams
.. autofunction:: cuda.cudart.cudaGraphExecEventRecordNodeSetEvent
.. autofunction:: cuda.cudart.cudaGraphExecEventWaitNodeSetEvent
.. autofunction:: cuda.cudart.cudaGraphExecExternalSemaphoresSignalNodeSetParams
.. autofunction:: cuda.cudart.cudaGraphExecExternalSemaphoresWaitNodeSetParams
.. autofunction:: cuda.cudart.cudaGraphNodeSetEnabled
.. autofunction:: cuda.cudart.cudaGraphNodeGetEnabled
.. autofunction:: cuda.cudart.cudaGraphExecUpdate
.. autofunction:: cuda.cudart.cudaGraphUpload
.. autofunction:: cuda.cudart.cudaGraphLaunch
.. autofunction:: cuda.cudart.cudaGraphExecDestroy
.. autofunction:: cuda.cudart.cudaGraphDestroy
.. autofunction:: cuda.cudart.cudaGraphDebugDotPrint
.. autofunction:: cuda.cudart.cudaUserObjectCreate
.. autofunction:: cuda.cudart.cudaUserObjectRetain
.. autofunction:: cuda.cudart.cudaUserObjectRelease
.. autofunction:: cuda.cudart.cudaGraphRetainUserObject
.. autofunction:: cuda.cudart.cudaGraphReleaseUserObject

Driver Entry Point Access
-------------------------

    
    


This section describes the driver entry point access functions of CUDA runtime application programming interface. 
    

  
.. autofunction:: cuda.cudart.cudaGetDriverEntryPoint

C++ API Routines
----------------

C++-style interface built on top of CUDA runtime API. 
    
    

impl_private


This section describes the C++ high level API functions of the CUDA runtime application programming interface. To use these functions, your application needs to be compiled with the ``nvcc``\  compiler. 
    

  

Interactions with the CUDA Driver API
-------------------------------------

    
    


This section describes the interactions between the CUDA Driver API and the CUDA Runtime API


**Primary Contexts**

There exists a one to one relationship between CUDA devices in the CUDA Runtime API and ::CUcontext s in the CUDA Driver API within a process. The specific context which the CUDA Runtime API uses for a device is called the device's primary context. From the perspective of the CUDA Runtime API, a device and its primary context are synonymous.


**Initialization and Tear-Down**

CUDA Runtime API calls operate on the CUDA Driver API ::CUcontext which is current to to the calling host thread. 

The function cudaSetDevice() makes the primary context for the specified device current to the calling thread by calling ::cuCtxSetCurrent().
The CUDA Runtime API will automatically initialize the primary context for a device at the first CUDA Runtime API call which requires an active context. If no ::CUcontext is current to the calling thread when a CUDA Runtime API call which requires an active context is made, then the primary context for a device will be selected, made current to the calling thread, and initialized.
The context which the CUDA Runtime API initializes will be initialized using the parameters specified by the CUDA Runtime API functions cudaSetDeviceFlags(), ::cudaD3D9SetDirect3DDevice(), ::cudaD3D10SetDirect3DDevice(), ::cudaD3D11SetDirect3DDevice(), cudaGLSetGLDevice(), and cudaVDPAUSetVDPAUDevice(). Note that these functions will fail with ::cudaErrorSetOnActiveProcess if they are called when the primary context for the specified device has already been initialized. (or if the current device has already been initialized, in the case of cudaSetDeviceFlags()).
Primary contexts will remain active until they are explicitly deinitialized using cudaDeviceReset(). The function cudaDeviceReset() will deinitialize the primary context for the calling thread's current device immediately. The context will remain current to all of the threads that it was current to. The next CUDA Runtime API call on any thread which requires an active context will trigger the reinitialization of that device's primary context.
Note that primary contexts are shared resources. It is recommended that the primary context not be reset except just before exit or to recover from an unspecified launch failure.


**Context Interoperability**

Note that the use of multiple ::CUcontext s per device within a single process will substantially degrade performance and is strongly discouraged. Instead, it is highly recommended that the implicit one-to-one device-to-context mapping for the process provided by the CUDA Runtime API be used.
If a non-primary ::CUcontext created by the CUDA Driver API is current to a thread then the CUDA Runtime API calls to that thread will operate on that ::CUcontext, with some exceptions listed below. Interoperability between data types is discussed in the following sections.
The function cudaPointerGetAttributes() will return the error ::cudaErrorIncompatibleDriverContext if the pointer being queried was allocated by a non-primary context. The function cudaDeviceEnablePeerAccess() and the rest of the peer access API may not be called when a non-primary ::CUcontext is current. 
 To use the pointer query and peer access APIs with a context created using the CUDA Driver API, it is necessary that the CUDA Driver API be used to access these features.
All CUDA Runtime API state (e.g, global variables' addresses and values) travels with its underlying ::CUcontext. In particular, if a ::CUcontext is moved from one thread to another then all CUDA Runtime API state will move to that thread as well.
Please note that attaching to legacy contexts (those with a version of 3010 as returned by ::cuCtxGetApiVersion()) is not possible. The CUDA Runtime will return ::cudaErrorIncompatibleDriverContext in such cases.


**Interactions between CUstream and cudaStream_t**

The types ::CUstream and cudaStream_t are identical and may be used interchangeably.


**Interactions between CUevent and cudaEvent_t**

The types ::CUevent and cudaEvent_t are identical and may be used interchangeably.


**Interactions between CUarray and cudaArray_t**

The types ::CUarray and struct ::cudaArray * represent the same data type and may be used interchangeably by casting the two types between each other.
In order to use a ::CUarray in a CUDA Runtime API function which takes a struct ::cudaArray *, it is necessary to explicitly cast the ::CUarray to a struct ::cudaArray *.
In order to use a struct ::cudaArray * in a CUDA Driver API function which takes a ::CUarray, it is necessary to explicitly cast the struct ::cudaArray * to a ::CUarray .


**Interactions between CUgraphicsResource and cudaGraphicsResource_t**

The types ::CUgraphicsResource and cudaGraphicsResource_t represent the same data type and may be used interchangeably by casting the two types between each other.
In order to use a ::CUgraphicsResource in a CUDA Runtime API function which takes a cudaGraphicsResource_t, it is necessary to explicitly cast the ::CUgraphicsResource to a cudaGraphicsResource_t.
In order to use a cudaGraphicsResource_t in a CUDA Driver API function which takes a ::CUgraphicsResource, it is necessary to explicitly cast the cudaGraphicsResource_t to a ::CUgraphicsResource.


**Interactions between CUtexObject * and cudaTextureObject_t**

The types ::CUtexObject * and cudaTextureObject_t represent the same data type and may be used interchangeably by casting the two types between each other.
In order to use a ::CUtexObject * in a CUDA Runtime API function which takes a cudaTextureObject_t, it is necessary to explicitly cast the ::CUtexObject * to a cudaTextureObject_t.
In order to use a cudaTextureObject_t in a CUDA Driver API function which takes a ::CUtexObject *, it is necessary to explicitly cast the cudaTextureObject_t to a ::CUtexObject *.


**Interactions between CUsurfObject * and cudaSurfaceObject_t**

The types ::CUsurfObject * and cudaSurfaceObject_t represent the same data type and may be used interchangeably by casting the two types between each other.
In order to use a ::CUsurfObject * in a CUDA Runtime API function which takes a ::cudaSurfaceObjec_t, it is necessary to explicitly cast the ::CUsurfObject * to a cudaSurfaceObject_t.
In order to use a cudaSurfaceObject_t in a CUDA Driver API function which takes a ::CUsurfObject *, it is necessary to explicitly cast the cudaSurfaceObject_t to a ::CUsurfObject *.


**Interactions between CUfunction and cudaFunction_t**

The types ::CUfunction and cudaFunction_t represent the same data type and may be used interchangeably by casting the two types between each other.
In order to use a cudaFunction_t in a CUDA Driver API function which takes a ::CUfunction, it is necessary to explicitly cast the cudaFunction_t to a ::CUfunction. 

    
  

Data types used by CUDA Runtime
-------------------------------

    
    

    
  
.. autoenum:: cuda.cudart.cudaEglFrameType
.. autoenum:: cuda.cudart.cudaEglResourceLocationFlags
.. autoenum:: cuda.cudart.cudaEglColorFormat
.. autoclass:: cuda.cudart.cudaEglPlaneDesc
.. autoclass:: cuda.cudart.cudaEglFrame
.. autoclass:: cuda.cudart.cudaEglStreamConnection
.. autoclass:: cuda.cudart.cudaArray_t
.. autoclass:: cuda.cudart.cudaArray_const_t
.. autoclass:: cuda.cudart.cudaMipmappedArray_t
.. autoclass:: cuda.cudart.cudaMipmappedArray_const_t
.. autoclass:: cuda.cudart.cudaHostFn_t
.. autoclass:: cuda.cudart.CUuuid
.. autoclass:: cuda.cudart.cudaUUID_t
.. autoclass:: cuda.cudart.cudaIpcEventHandle_t
.. autoclass:: cuda.cudart.cudaIpcMemHandle_t
.. autoclass:: cuda.cudart.cudaStream_t
.. autoclass:: cuda.cudart.cudaEvent_t
.. autoclass:: cuda.cudart.cudaGraphicsResource_t
.. autoclass:: cuda.cudart.cudaExternalMemory_t
.. autoclass:: cuda.cudart.cudaExternalSemaphore_t
.. autoclass:: cuda.cudart.cudaGraph_t
.. autoclass:: cuda.cudart.cudaGraphNode_t
.. autoclass:: cuda.cudart.cudaUserObject_t
.. autoclass:: cuda.cudart.cudaFunction_t
.. autoclass:: cuda.cudart.cudaMemPool_t
.. autoclass:: cuda.cudart.cudaGraphExec_t
.. autoclass:: cuda.cudart.cudaSurfaceObject_t
.. autoclass:: cuda.cudart.cudaTextureObject_t
