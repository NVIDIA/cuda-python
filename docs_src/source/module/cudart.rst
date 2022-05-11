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
.. autofunction:: cuda.cudart.make_cudaPitchedPtr
.. autofunction:: cuda.cudart.make_cudaPos
.. autofunction:: cuda.cudart.make_cudaExtent

Stream Ordered Memory Allocator
-------------------------------

**overview**



The asynchronous allocator allows the user to allocate and free in stream order. All asynchronous accesses of the allocation must happen between the stream executions of the allocation and the free. If the memory is accessed outside of the promised stream order, a use before allocation / use after free error will cause undefined behavior.

The allocator is free to reallocate the memory as long as it can guarantee that compliant memory accesses will not overlap temporally. The allocator may refer to internal stream ordering as well as inter-stream dependencies (such as CUDA events and null stream dependencies) when establishing the temporal guarantee. The allocator may also insert inter-stream dependencies to establish the temporal guarantee.





**Supported Platforms**



Whether or not a device supports the integrated stream ordered memory allocator may be queried by calling cudaDeviceGetAttribute() with the device attribute cudaDevAttrMemoryPoolsSupported.

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

 The copy direction cudaMemcpyDefault may be used to specify that the CUDA runtime should infer the location of the pointer from its value.





**Automatic Mapping of Host Allocated Host Memory**



All host memory allocated through all devices using cudaMallocHost() and cudaHostAlloc() is always directly accessible from all devices that support unified addressing. This is the case regardless of whether or not the flags cudaHostAllocPortable and cudaHostAllocMapped are specified.

The pointer value through which allocated host memory may be accessed in kernels on all devices that support unified addressing is the same as the pointer value through which that memory is accessed on the host. It is not necessary to call cudaHostGetDevicePointer() to get the device pointer for these allocations. 



Note that this is not the case for memory allocated using the flag cudaHostAllocWriteCombined, as discussed below.





**Direct Access of Peer Memory**



Upon enabling direct access from a device that supports unified addressing to another peer device that supports unified addressing using cudaDeviceEnablePeerAccess() all memory allocated in the peer device using cudaMalloc() and cudaMallocPitch() will immediately be accessible by the current device. The device pointer value through which any peer's memory may be accessed in the current device is the same pointer value through which that memory may be accessed from the peer device.





**Exceptions, Disjoint Addressing**



Not all memory may be accessed on devices through the same pointer value through which they are accessed on the host. These exceptions are host memory registered using cudaHostRegister() and host memory allocated using the flag cudaHostAllocWriteCombined. For these exceptions, there exists a distinct host and device address for the memory. The device address is guaranteed to not overlap any valid host pointer range and is guaranteed to have the same value across all devices that support unified addressing. 



This device address may be queried using cudaHostGetDevicePointer() when a device using unified addressing is current. Either the host or the unified device pointer value may be used to refer to this memory in cudaMemcpy() and similar functions using the cudaMemcpyDefault memory direction.

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

.. autoclass:: cuda.cudart.cudaGLDeviceList

    .. autoattribute:: cuda.cudart.cudaGLDeviceList.cudaGLDeviceListAll


        The CUDA devices for all GPUs used by the current OpenGL context


    .. autoattribute:: cuda.cudart.cudaGLDeviceList.cudaGLDeviceListCurrentFrame


        The CUDA devices for the GPUs used by the current OpenGL context in its currently rendering frame


    .. autoattribute:: cuda.cudart.cudaGLDeviceList.cudaGLDeviceListNextFrame


        The CUDA devices for the GPUs to be used by the current OpenGL context in the next frame

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

The context which the CUDA Runtime API initializes will be initialized using the parameters specified by the CUDA Runtime API functions cudaSetDeviceFlags(), ::cudaD3D9SetDirect3DDevice(), ::cudaD3D10SetDirect3DDevice(), ::cudaD3D11SetDirect3DDevice(), cudaGLSetGLDevice(), and cudaVDPAUSetVDPAUDevice(). Note that these functions will fail with cudaErrorSetOnActiveProcess if they are called when the primary context for the specified device has already been initialized. (or if the current device has already been initialized, in the case of cudaSetDeviceFlags()).

Primary contexts will remain active until they are explicitly deinitialized using cudaDeviceReset(). The function cudaDeviceReset() will deinitialize the primary context for the calling thread's current device immediately. The context will remain current to all of the threads that it was current to. The next CUDA Runtime API call on any thread which requires an active context will trigger the reinitialization of that device's primary context.

Note that primary contexts are shared resources. It is recommended that the primary context not be reset except just before exit or to recover from an unspecified launch failure.





**Context Interoperability**



Note that the use of multiple ::CUcontext s per device within a single process will substantially degrade performance and is strongly discouraged. Instead, it is highly recommended that the implicit one-to-one device-to-context mapping for the process provided by the CUDA Runtime API be used.

If a non-primary ::CUcontext created by the CUDA Driver API is current to a thread then the CUDA Runtime API calls to that thread will operate on that ::CUcontext, with some exceptions listed below. Interoperability between data types is discussed in the following sections.

The function cudaPointerGetAttributes() will return the error cudaErrorIncompatibleDriverContext if the pointer being queried was allocated by a non-primary context. The function cudaDeviceEnablePeerAccess() and the rest of the peer access API may not be called when a non-primary ::CUcontext is current. 

 To use the pointer query and peer access APIs with a context created using the CUDA Driver API, it is necessary that the CUDA Driver API be used to access these features.

All CUDA Runtime API state (e.g, global variables' addresses and values) travels with its underlying ::CUcontext. In particular, if a ::CUcontext is moved from one thread to another then all CUDA Runtime API state will move to that thread as well.

Please note that attaching to legacy contexts (those with a version of 3010 as returned by ::cuCtxGetApiVersion()) is not possible. The CUDA Runtime will return cudaErrorIncompatibleDriverContext in such cases.





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





**Interactions between CUtexObject and cudaTextureObject_t**



The types ::CUtexObject and cudaTextureObject_t represent the same data type and may be used interchangeably by casting the two types between each other.

In order to use a ::CUtexObject in a CUDA Runtime API function which takes a cudaTextureObject_t, it is necessary to explicitly cast the ::CUtexObject to a cudaTextureObject_t.

In order to use a cudaTextureObject_t in a CUDA Driver API function which takes a ::CUtexObject, it is necessary to explicitly cast the cudaTextureObject_t to a ::CUtexObject.





**Interactions between CUsurfObject and cudaSurfaceObject_t**



The types ::CUsurfObject and cudaSurfaceObject_t represent the same data type and may be used interchangeably by casting the two types between each other.

In order to use a ::CUsurfObject in a CUDA Runtime API function which takes a cudaSurfaceObject_t, it is necessary to explicitly cast the ::CUsurfObject to a cudaSurfaceObject_t.

In order to use a cudaSurfaceObject_t in a CUDA Driver API function which takes a ::CUsurfObject, it is necessary to explicitly cast the cudaSurfaceObject_t to a ::CUsurfObject.





**Interactions between CUfunction and cudaFunction_t**



The types ::CUfunction and cudaFunction_t represent the same data type and may be used interchangeably by casting the two types between each other.

In order to use a cudaFunction_t in a CUDA Driver API function which takes a ::CUfunction, it is necessary to explicitly cast the cudaFunction_t to a ::CUfunction.


Data types used by CUDA Runtime
-------------------------------



.. autoclass:: cuda.cudart.cudaEglPlaneDesc_st
.. autoclass:: cuda.cudart.cudaEglFrame_st
.. autoclass:: cuda.cudart.cudaChannelFormatDesc
.. autoclass:: cuda.cudart.cudaArraySparseProperties
.. autoclass:: cuda.cudart.cudaArrayMemoryRequirements
.. autoclass:: cuda.cudart.cudaPitchedPtr
.. autoclass:: cuda.cudart.cudaExtent
.. autoclass:: cuda.cudart.cudaPos
.. autoclass:: cuda.cudart.cudaMemcpy3DParms
.. autoclass:: cuda.cudart.cudaMemcpy3DPeerParms
.. autoclass:: cuda.cudart.cudaMemsetParams
.. autoclass:: cuda.cudart.cudaAccessPolicyWindow
.. autoclass:: cuda.cudart.cudaHostNodeParams
.. autoclass:: cuda.cudart.cudaResourceDesc
.. autoclass:: cuda.cudart.cudaResourceViewDesc
.. autoclass:: cuda.cudart.cudaPointerAttributes
.. autoclass:: cuda.cudart.cudaFuncAttributes
.. autoclass:: cuda.cudart.cudaMemLocation
.. autoclass:: cuda.cudart.cudaMemAccessDesc
.. autoclass:: cuda.cudart.cudaMemPoolProps
.. autoclass:: cuda.cudart.cudaMemPoolPtrExportData
.. autoclass:: cuda.cudart.cudaMemAllocNodeParams
.. autoclass:: cuda.cudart.CUuuid_st
.. autoclass:: cuda.cudart.cudaDeviceProp
.. autoclass:: cuda.cudart.cudaIpcEventHandle_st
.. autoclass:: cuda.cudart.cudaIpcMemHandle_st
.. autoclass:: cuda.cudart.cudaExternalMemoryHandleDesc
.. autoclass:: cuda.cudart.cudaExternalMemoryBufferDesc
.. autoclass:: cuda.cudart.cudaExternalMemoryMipmappedArrayDesc
.. autoclass:: cuda.cudart.cudaExternalSemaphoreHandleDesc
.. autoclass:: cuda.cudart.cudaExternalSemaphoreSignalParams_v1
.. autoclass:: cuda.cudart.cudaExternalSemaphoreWaitParams_v1
.. autoclass:: cuda.cudart.cudaExternalSemaphoreSignalParams
.. autoclass:: cuda.cudart.cudaExternalSemaphoreWaitParams
.. autoclass:: cuda.cudart.cudaLaunchParams
.. autoclass:: cuda.cudart.cudaKernelNodeParams
.. autoclass:: cuda.cudart.cudaExternalSemaphoreSignalNodeParams
.. autoclass:: cuda.cudart.cudaExternalSemaphoreWaitNodeParams
.. autoclass:: cuda.cudart.cudaStreamAttrValue
.. autoclass:: cuda.cudart.cudaKernelNodeAttrValue
.. autoclass:: cuda.cudart.surfaceReference
.. autoclass:: cuda.cudart.textureReference
.. autoclass:: cuda.cudart.cudaTextureDesc
.. autoclass:: cuda.cudart.cudaEglFrameType

    .. autoattribute:: cuda.cudart.cudaEglFrameType.cudaEglFrameTypeArray


        Frame type CUDA array


    .. autoattribute:: cuda.cudart.cudaEglFrameType.cudaEglFrameTypePitch


        Frame type CUDA pointer

.. autoclass:: cuda.cudart.cudaEglResourceLocationFlags

    .. autoattribute:: cuda.cudart.cudaEglResourceLocationFlags.cudaEglResourceLocationSysmem


        Resource location sysmem


    .. autoattribute:: cuda.cudart.cudaEglResourceLocationFlags.cudaEglResourceLocationVidmem


        Resource location vidmem

.. autoclass:: cuda.cudart.cudaEglColorFormat

    .. autoattribute:: cuda.cudart.cudaEglColorFormat.cudaEglColorFormatYUV420Planar


        Y, U, V in three surfaces, each in a separate surface, U/V width = 1/2 Y width, U/V height = 1/2 Y height.


    .. autoattribute:: cuda.cudart.cudaEglColorFormat.cudaEglColorFormatYUV420SemiPlanar


        Y, UV in two surfaces (UV as one surface) with VU byte ordering, width, height ratio same as YUV420Planar.


    .. autoattribute:: cuda.cudart.cudaEglColorFormat.cudaEglColorFormatYUV422Planar


        Y, U, V each in a separate surface, U/V width = 1/2 Y width, U/V height = Y height.


    .. autoattribute:: cuda.cudart.cudaEglColorFormat.cudaEglColorFormatYUV422SemiPlanar


        Y, UV in two surfaces with VU byte ordering, width, height ratio same as YUV422Planar.


    .. autoattribute:: cuda.cudart.cudaEglColorFormat.cudaEglColorFormatARGB


        R/G/B/A four channels in one surface with BGRA byte ordering.


    .. autoattribute:: cuda.cudart.cudaEglColorFormat.cudaEglColorFormatRGBA


        R/G/B/A four channels in one surface with ABGR byte ordering.


    .. autoattribute:: cuda.cudart.cudaEglColorFormat.cudaEglColorFormatL


        single luminance channel in one surface.


    .. autoattribute:: cuda.cudart.cudaEglColorFormat.cudaEglColorFormatR


        single color channel in one surface.


    .. autoattribute:: cuda.cudart.cudaEglColorFormat.cudaEglColorFormatYUV444Planar


        Y, U, V in three surfaces, each in a separate surface, U/V width = Y width, U/V height = Y height.


    .. autoattribute:: cuda.cudart.cudaEglColorFormat.cudaEglColorFormatYUV444SemiPlanar


        Y, UV in two surfaces (UV as one surface) with VU byte ordering, width, height ratio same as YUV444Planar.


    .. autoattribute:: cuda.cudart.cudaEglColorFormat.cudaEglColorFormatYUYV422


        Y, U, V in one surface, interleaved as UYVY in one channel.


    .. autoattribute:: cuda.cudart.cudaEglColorFormat.cudaEglColorFormatUYVY422


        Y, U, V in one surface, interleaved as YUYV in one channel.


    .. autoattribute:: cuda.cudart.cudaEglColorFormat.cudaEglColorFormatABGR


        R/G/B/A four channels in one surface with RGBA byte ordering.


    .. autoattribute:: cuda.cudart.cudaEglColorFormat.cudaEglColorFormatBGRA


        R/G/B/A four channels in one surface with ARGB byte ordering.


    .. autoattribute:: cuda.cudart.cudaEglColorFormat.cudaEglColorFormatA


        Alpha color format - one channel in one surface.


    .. autoattribute:: cuda.cudart.cudaEglColorFormat.cudaEglColorFormatRG


        R/G color format - two channels in one surface with GR byte ordering


    .. autoattribute:: cuda.cudart.cudaEglColorFormat.cudaEglColorFormatAYUV


        Y, U, V, A four channels in one surface, interleaved as VUYA.


    .. autoattribute:: cuda.cudart.cudaEglColorFormat.cudaEglColorFormatYVU444SemiPlanar


        Y, VU in two surfaces (VU as one surface) with UV byte ordering, U/V width = Y width, U/V height = Y height.


    .. autoattribute:: cuda.cudart.cudaEglColorFormat.cudaEglColorFormatYVU422SemiPlanar


        Y, VU in two surfaces (VU as one surface) with UV byte ordering, U/V width = 1/2 Y width, U/V height = Y height.


    .. autoattribute:: cuda.cudart.cudaEglColorFormat.cudaEglColorFormatYVU420SemiPlanar


        Y, VU in two surfaces (VU as one surface) with UV byte ordering, U/V width = 1/2 Y width, U/V height = 1/2 Y height.


    .. autoattribute:: cuda.cudart.cudaEglColorFormat.cudaEglColorFormatY10V10U10_444SemiPlanar


        Y10, V10U10 in two surfaces (VU as one surface) with UV byte ordering, U/V width = Y width, U/V height = Y height.


    .. autoattribute:: cuda.cudart.cudaEglColorFormat.cudaEglColorFormatY10V10U10_420SemiPlanar


        Y10, V10U10 in two surfaces (VU as one surface) with UV byte ordering, U/V width = 1/2 Y width, U/V height = 1/2 Y height.


    .. autoattribute:: cuda.cudart.cudaEglColorFormat.cudaEglColorFormatY12V12U12_444SemiPlanar


        Y12, V12U12 in two surfaces (VU as one surface) with UV byte ordering, U/V width = Y width, U/V height = Y height.


    .. autoattribute:: cuda.cudart.cudaEglColorFormat.cudaEglColorFormatY12V12U12_420SemiPlanar


        Y12, V12U12 in two surfaces (VU as one surface) with UV byte ordering, U/V width = 1/2 Y width, U/V height = 1/2 Y height.


    .. autoattribute:: cuda.cudart.cudaEglColorFormat.cudaEglColorFormatVYUY_ER


        Extended Range Y, U, V in one surface, interleaved as YVYU in one channel.


    .. autoattribute:: cuda.cudart.cudaEglColorFormat.cudaEglColorFormatUYVY_ER


        Extended Range Y, U, V in one surface, interleaved as YUYV in one channel.


    .. autoattribute:: cuda.cudart.cudaEglColorFormat.cudaEglColorFormatYUYV_ER


        Extended Range Y, U, V in one surface, interleaved as UYVY in one channel.


    .. autoattribute:: cuda.cudart.cudaEglColorFormat.cudaEglColorFormatYVYU_ER


        Extended Range Y, U, V in one surface, interleaved as VYUY in one channel.


    .. autoattribute:: cuda.cudart.cudaEglColorFormat.cudaEglColorFormatYUVA_ER


        Extended Range Y, U, V, A four channels in one surface, interleaved as AVUY.


    .. autoattribute:: cuda.cudart.cudaEglColorFormat.cudaEglColorFormatAYUV_ER


        Extended Range Y, U, V, A four channels in one surface, interleaved as VUYA.


    .. autoattribute:: cuda.cudart.cudaEglColorFormat.cudaEglColorFormatYUV444Planar_ER


        Extended Range Y, U, V in three surfaces, U/V width = Y width, U/V height = Y height.


    .. autoattribute:: cuda.cudart.cudaEglColorFormat.cudaEglColorFormatYUV422Planar_ER


        Extended Range Y, U, V in three surfaces, U/V width = 1/2 Y width, U/V height = Y height.


    .. autoattribute:: cuda.cudart.cudaEglColorFormat.cudaEglColorFormatYUV420Planar_ER


        Extended Range Y, U, V in three surfaces, U/V width = 1/2 Y width, U/V height = 1/2 Y height.


    .. autoattribute:: cuda.cudart.cudaEglColorFormat.cudaEglColorFormatYUV444SemiPlanar_ER


        Extended Range Y, UV in two surfaces (UV as one surface) with VU byte ordering, U/V width = Y width, U/V height = Y height.


    .. autoattribute:: cuda.cudart.cudaEglColorFormat.cudaEglColorFormatYUV422SemiPlanar_ER


        Extended Range Y, UV in two surfaces (UV as one surface) with VU byte ordering, U/V width = 1/2 Y width, U/V height = Y height.


    .. autoattribute:: cuda.cudart.cudaEglColorFormat.cudaEglColorFormatYUV420SemiPlanar_ER


        Extended Range Y, UV in two surfaces (UV as one surface) with VU byte ordering, U/V width = 1/2 Y width, U/V height = 1/2 Y height.


    .. autoattribute:: cuda.cudart.cudaEglColorFormat.cudaEglColorFormatYVU444Planar_ER


        Extended Range Y, V, U in three surfaces, U/V width = Y width, U/V height = Y height.


    .. autoattribute:: cuda.cudart.cudaEglColorFormat.cudaEglColorFormatYVU422Planar_ER


        Extended Range Y, V, U in three surfaces, U/V width = 1/2 Y width, U/V height = Y height.


    .. autoattribute:: cuda.cudart.cudaEglColorFormat.cudaEglColorFormatYVU420Planar_ER


        Extended Range Y, V, U in three surfaces, U/V width = 1/2 Y width, U/V height = 1/2 Y height.


    .. autoattribute:: cuda.cudart.cudaEglColorFormat.cudaEglColorFormatYVU444SemiPlanar_ER


        Extended Range Y, VU in two surfaces (VU as one surface) with UV byte ordering, U/V width = Y width, U/V height = Y height.


    .. autoattribute:: cuda.cudart.cudaEglColorFormat.cudaEglColorFormatYVU422SemiPlanar_ER


        Extended Range Y, VU in two surfaces (VU as one surface) with UV byte ordering, U/V width = 1/2 Y width, U/V height = Y height.


    .. autoattribute:: cuda.cudart.cudaEglColorFormat.cudaEglColorFormatYVU420SemiPlanar_ER


        Extended Range Y, VU in two surfaces (VU as one surface) with UV byte ordering, U/V width = 1/2 Y width, U/V height = 1/2 Y height.


    .. autoattribute:: cuda.cudart.cudaEglColorFormat.cudaEglColorFormatBayerRGGB


        Bayer format - one channel in one surface with interleaved RGGB ordering.


    .. autoattribute:: cuda.cudart.cudaEglColorFormat.cudaEglColorFormatBayerBGGR


        Bayer format - one channel in one surface with interleaved BGGR ordering.


    .. autoattribute:: cuda.cudart.cudaEglColorFormat.cudaEglColorFormatBayerGRBG


        Bayer format - one channel in one surface with interleaved GRBG ordering.


    .. autoattribute:: cuda.cudart.cudaEglColorFormat.cudaEglColorFormatBayerGBRG


        Bayer format - one channel in one surface with interleaved GBRG ordering.


    .. autoattribute:: cuda.cudart.cudaEglColorFormat.cudaEglColorFormatBayer10RGGB


        Bayer10 format - one channel in one surface with interleaved RGGB ordering. Out of 16 bits, 10 bits used 6 bits No-op.


    .. autoattribute:: cuda.cudart.cudaEglColorFormat.cudaEglColorFormatBayer10BGGR


        Bayer10 format - one channel in one surface with interleaved BGGR ordering. Out of 16 bits, 10 bits used 6 bits No-op.


    .. autoattribute:: cuda.cudart.cudaEglColorFormat.cudaEglColorFormatBayer10GRBG


        Bayer10 format - one channel in one surface with interleaved GRBG ordering. Out of 16 bits, 10 bits used 6 bits No-op.


    .. autoattribute:: cuda.cudart.cudaEglColorFormat.cudaEglColorFormatBayer10GBRG


        Bayer10 format - one channel in one surface with interleaved GBRG ordering. Out of 16 bits, 10 bits used 6 bits No-op.


    .. autoattribute:: cuda.cudart.cudaEglColorFormat.cudaEglColorFormatBayer12RGGB


        Bayer12 format - one channel in one surface with interleaved RGGB ordering. Out of 16 bits, 12 bits used 4 bits No-op.


    .. autoattribute:: cuda.cudart.cudaEglColorFormat.cudaEglColorFormatBayer12BGGR


        Bayer12 format - one channel in one surface with interleaved BGGR ordering. Out of 16 bits, 12 bits used 4 bits No-op.


    .. autoattribute:: cuda.cudart.cudaEglColorFormat.cudaEglColorFormatBayer12GRBG


        Bayer12 format - one channel in one surface with interleaved GRBG ordering. Out of 16 bits, 12 bits used 4 bits No-op.


    .. autoattribute:: cuda.cudart.cudaEglColorFormat.cudaEglColorFormatBayer12GBRG


        Bayer12 format - one channel in one surface with interleaved GBRG ordering. Out of 16 bits, 12 bits used 4 bits No-op.


    .. autoattribute:: cuda.cudart.cudaEglColorFormat.cudaEglColorFormatBayer14RGGB


        Bayer14 format - one channel in one surface with interleaved RGGB ordering. Out of 16 bits, 14 bits used 2 bits No-op.


    .. autoattribute:: cuda.cudart.cudaEglColorFormat.cudaEglColorFormatBayer14BGGR


        Bayer14 format - one channel in one surface with interleaved BGGR ordering. Out of 16 bits, 14 bits used 2 bits No-op.


    .. autoattribute:: cuda.cudart.cudaEglColorFormat.cudaEglColorFormatBayer14GRBG


        Bayer14 format - one channel in one surface with interleaved GRBG ordering. Out of 16 bits, 14 bits used 2 bits No-op.


    .. autoattribute:: cuda.cudart.cudaEglColorFormat.cudaEglColorFormatBayer14GBRG


        Bayer14 format - one channel in one surface with interleaved GBRG ordering. Out of 16 bits, 14 bits used 2 bits No-op.


    .. autoattribute:: cuda.cudart.cudaEglColorFormat.cudaEglColorFormatBayer20RGGB


        Bayer20 format - one channel in one surface with interleaved RGGB ordering. Out of 32 bits, 20 bits used 12 bits No-op.


    .. autoattribute:: cuda.cudart.cudaEglColorFormat.cudaEglColorFormatBayer20BGGR


        Bayer20 format - one channel in one surface with interleaved BGGR ordering. Out of 32 bits, 20 bits used 12 bits No-op.


    .. autoattribute:: cuda.cudart.cudaEglColorFormat.cudaEglColorFormatBayer20GRBG


        Bayer20 format - one channel in one surface with interleaved GRBG ordering. Out of 32 bits, 20 bits used 12 bits No-op.


    .. autoattribute:: cuda.cudart.cudaEglColorFormat.cudaEglColorFormatBayer20GBRG


        Bayer20 format - one channel in one surface with interleaved GBRG ordering. Out of 32 bits, 20 bits used 12 bits No-op.


    .. autoattribute:: cuda.cudart.cudaEglColorFormat.cudaEglColorFormatYVU444Planar


        Y, V, U in three surfaces, each in a separate surface, U/V width = Y width, U/V height = Y height.


    .. autoattribute:: cuda.cudart.cudaEglColorFormat.cudaEglColorFormatYVU422Planar


        Y, V, U in three surfaces, each in a separate surface, U/V width = 1/2 Y width, U/V height = Y height.


    .. autoattribute:: cuda.cudart.cudaEglColorFormat.cudaEglColorFormatYVU420Planar


        Y, V, U in three surfaces, each in a separate surface, U/V width = 1/2 Y width, U/V height = 1/2 Y height.


    .. autoattribute:: cuda.cudart.cudaEglColorFormat.cudaEglColorFormatBayerIspRGGB


        Nvidia proprietary Bayer ISP format - one channel in one surface with interleaved RGGB ordering and mapped to opaque integer datatype.


    .. autoattribute:: cuda.cudart.cudaEglColorFormat.cudaEglColorFormatBayerIspBGGR


        Nvidia proprietary Bayer ISP format - one channel in one surface with interleaved BGGR ordering and mapped to opaque integer datatype.


    .. autoattribute:: cuda.cudart.cudaEglColorFormat.cudaEglColorFormatBayerIspGRBG


        Nvidia proprietary Bayer ISP format - one channel in one surface with interleaved GRBG ordering and mapped to opaque integer datatype.


    .. autoattribute:: cuda.cudart.cudaEglColorFormat.cudaEglColorFormatBayerIspGBRG


        Nvidia proprietary Bayer ISP format - one channel in one surface with interleaved GBRG ordering and mapped to opaque integer datatype.


    .. autoattribute:: cuda.cudart.cudaEglColorFormat.cudaEglColorFormatBayerBCCR


        Bayer format - one channel in one surface with interleaved BCCR ordering.


    .. autoattribute:: cuda.cudart.cudaEglColorFormat.cudaEglColorFormatBayerRCCB


        Bayer format - one channel in one surface with interleaved RCCB ordering.


    .. autoattribute:: cuda.cudart.cudaEglColorFormat.cudaEglColorFormatBayerCRBC


        Bayer format - one channel in one surface with interleaved CRBC ordering.


    .. autoattribute:: cuda.cudart.cudaEglColorFormat.cudaEglColorFormatBayerCBRC


        Bayer format - one channel in one surface with interleaved CBRC ordering.


    .. autoattribute:: cuda.cudart.cudaEglColorFormat.cudaEglColorFormatBayer10CCCC


        Bayer10 format - one channel in one surface with interleaved CCCC ordering. Out of 16 bits, 10 bits used 6 bits No-op.


    .. autoattribute:: cuda.cudart.cudaEglColorFormat.cudaEglColorFormatBayer12BCCR


        Bayer12 format - one channel in one surface with interleaved BCCR ordering. Out of 16 bits, 12 bits used 4 bits No-op.


    .. autoattribute:: cuda.cudart.cudaEglColorFormat.cudaEglColorFormatBayer12RCCB


        Bayer12 format - one channel in one surface with interleaved RCCB ordering. Out of 16 bits, 12 bits used 4 bits No-op.


    .. autoattribute:: cuda.cudart.cudaEglColorFormat.cudaEglColorFormatBayer12CRBC


        Bayer12 format - one channel in one surface with interleaved CRBC ordering. Out of 16 bits, 12 bits used 4 bits No-op.


    .. autoattribute:: cuda.cudart.cudaEglColorFormat.cudaEglColorFormatBayer12CBRC


        Bayer12 format - one channel in one surface with interleaved CBRC ordering. Out of 16 bits, 12 bits used 4 bits No-op.


    .. autoattribute:: cuda.cudart.cudaEglColorFormat.cudaEglColorFormatBayer12CCCC


        Bayer12 format - one channel in one surface with interleaved CCCC ordering. Out of 16 bits, 12 bits used 4 bits No-op.


    .. autoattribute:: cuda.cudart.cudaEglColorFormat.cudaEglColorFormatY


        Color format for single Y plane.


    .. autoattribute:: cuda.cudart.cudaEglColorFormat.cudaEglColorFormatYUV420SemiPlanar_2020


        Y, UV in two surfaces (UV as one surface) U/V width = 1/2 Y width, U/V height = 1/2 Y height.


    .. autoattribute:: cuda.cudart.cudaEglColorFormat.cudaEglColorFormatYVU420SemiPlanar_2020


        Y, VU in two surfaces (VU as one surface) U/V width = 1/2 Y width, U/V height = 1/2 Y height.


    .. autoattribute:: cuda.cudart.cudaEglColorFormat.cudaEglColorFormatYUV420Planar_2020


        Y, U, V in three surfaces, each in a separate surface, U/V width = 1/2 Y width, U/V height = 1/2 Y height.


    .. autoattribute:: cuda.cudart.cudaEglColorFormat.cudaEglColorFormatYVU420Planar_2020


        Y, V, U in three surfaces, each in a separate surface, U/V width = 1/2 Y width, U/V height = 1/2 Y height.


    .. autoattribute:: cuda.cudart.cudaEglColorFormat.cudaEglColorFormatYUV420SemiPlanar_709


        Y, UV in two surfaces (UV as one surface) U/V width = 1/2 Y width, U/V height = 1/2 Y height.


    .. autoattribute:: cuda.cudart.cudaEglColorFormat.cudaEglColorFormatYVU420SemiPlanar_709


        Y, VU in two surfaces (VU as one surface) U/V width = 1/2 Y width, U/V height = 1/2 Y height.


    .. autoattribute:: cuda.cudart.cudaEglColorFormat.cudaEglColorFormatYUV420Planar_709


        Y, U, V in three surfaces, each in a separate surface, U/V width = 1/2 Y width, U/V height = 1/2 Y height.


    .. autoattribute:: cuda.cudart.cudaEglColorFormat.cudaEglColorFormatYVU420Planar_709


        Y, V, U in three surfaces, each in a separate surface, U/V width = 1/2 Y width, U/V height = 1/2 Y height.


    .. autoattribute:: cuda.cudart.cudaEglColorFormat.cudaEglColorFormatY10V10U10_420SemiPlanar_709


        Y10, V10U10 in two surfaces (VU as one surface) U/V width = 1/2 Y width, U/V height = 1/2 Y height.


    .. autoattribute:: cuda.cudart.cudaEglColorFormat.cudaEglColorFormatY10V10U10_420SemiPlanar_2020


        Y10, V10U10 in two surfaces (VU as one surface) U/V width = 1/2 Y width, U/V height = 1/2 Y height.


    .. autoattribute:: cuda.cudart.cudaEglColorFormat.cudaEglColorFormatY10V10U10_422SemiPlanar_2020


        Y10, V10U10 in two surfaces (VU as one surface) U/V width = 1/2 Y width, U/V height = Y height.


    .. autoattribute:: cuda.cudart.cudaEglColorFormat.cudaEglColorFormatY10V10U10_422SemiPlanar


        Y10, V10U10 in two surfaces (VU as one surface) U/V width = 1/2 Y width, U/V height = Y height.


    .. autoattribute:: cuda.cudart.cudaEglColorFormat.cudaEglColorFormatY10V10U10_422SemiPlanar_709


        Y10, V10U10 in two surfaces (VU as one surface) U/V width = 1/2 Y width, U/V height = Y height.


    .. autoattribute:: cuda.cudart.cudaEglColorFormat.cudaEglColorFormatY_ER


        Extended Range Color format for single Y plane.


    .. autoattribute:: cuda.cudart.cudaEglColorFormat.cudaEglColorFormatY_709_ER


        Extended Range Color format for single Y plane.


    .. autoattribute:: cuda.cudart.cudaEglColorFormat.cudaEglColorFormatY10_ER


        Extended Range Color format for single Y10 plane.


    .. autoattribute:: cuda.cudart.cudaEglColorFormat.cudaEglColorFormatY10_709_ER


        Extended Range Color format for single Y10 plane.


    .. autoattribute:: cuda.cudart.cudaEglColorFormat.cudaEglColorFormatY12_ER


        Extended Range Color format for single Y12 plane.


    .. autoattribute:: cuda.cudart.cudaEglColorFormat.cudaEglColorFormatY12_709_ER


        Extended Range Color format for single Y12 plane.


    .. autoattribute:: cuda.cudart.cudaEglColorFormat.cudaEglColorFormatYUVA


        Y, U, V, A four channels in one surface, interleaved as AVUY.


    .. autoattribute:: cuda.cudart.cudaEglColorFormat.cudaEglColorFormatYVYU


        Y, U, V in one surface, interleaved as YVYU in one channel.


    .. autoattribute:: cuda.cudart.cudaEglColorFormat.cudaEglColorFormatVYUY


        Y, U, V in one surface, interleaved as VYUY in one channel.


    .. autoattribute:: cuda.cudart.cudaEglColorFormat.cudaEglColorFormatY10V10U10_420SemiPlanar_ER


        Extended Range Y10, V10U10 in two surfaces (VU as one surface) U/V width = 1/2 Y width, U/V height = 1/2 Y height.


    .. autoattribute:: cuda.cudart.cudaEglColorFormat.cudaEglColorFormatY10V10U10_420SemiPlanar_709_ER


        Extended Range Y10, V10U10 in two surfaces (VU as one surface) U/V width = 1/2 Y width, U/V height = 1/2 Y height.


    .. autoattribute:: cuda.cudart.cudaEglColorFormat.cudaEglColorFormatY10V10U10_444SemiPlanar_ER


        Extended Range Y10, V10U10 in two surfaces (VU as one surface) U/V width = Y width, U/V height = Y height.


    .. autoattribute:: cuda.cudart.cudaEglColorFormat.cudaEglColorFormatY10V10U10_444SemiPlanar_709_ER


        Extended Range Y10, V10U10 in two surfaces (VU as one surface) U/V width = Y width, U/V height = Y height.


    .. autoattribute:: cuda.cudart.cudaEglColorFormat.cudaEglColorFormatY12V12U12_420SemiPlanar_ER


        Extended Range Y12, V12U12 in two surfaces (VU as one surface) U/V width = 1/2 Y width, U/V height = 1/2 Y height.


    .. autoattribute:: cuda.cudart.cudaEglColorFormat.cudaEglColorFormatY12V12U12_420SemiPlanar_709_ER


        Extended Range Y12, V12U12 in two surfaces (VU as one surface) U/V width = 1/2 Y width, U/V height = 1/2 Y height.


    .. autoattribute:: cuda.cudart.cudaEglColorFormat.cudaEglColorFormatY12V12U12_444SemiPlanar_ER


        Extended Range Y12, V12U12 in two surfaces (VU as one surface) U/V width = Y width, U/V height = Y height.


    .. autoattribute:: cuda.cudart.cudaEglColorFormat.cudaEglColorFormatY12V12U12_444SemiPlanar_709_ER


        Extended Range Y12, V12U12 in two surfaces (VU as one surface) U/V width = Y width, U/V height = Y height.

.. autoclass:: cuda.cudart.cudaError_t

    .. autoattribute:: cuda.cudart.cudaError_t.cudaSuccess


        The API call returned with no errors. In the case of query calls, this also means that the operation being queried is complete (see :py:obj:`~.cudaEventQuery()` and :py:obj:`~.cudaStreamQuery()`).


    .. autoattribute:: cuda.cudart.cudaError_t.cudaErrorInvalidValue


        This indicates that one or more of the parameters passed to the API call is not within an acceptable range of values.


    .. autoattribute:: cuda.cudart.cudaError_t.cudaErrorMemoryAllocation


        The API call failed because it was unable to allocate enough memory to perform the requested operation.


    .. autoattribute:: cuda.cudart.cudaError_t.cudaErrorInitializationError


        The API call failed because the CUDA driver and runtime could not be initialized.


    .. autoattribute:: cuda.cudart.cudaError_t.cudaErrorCudartUnloading


        This indicates that a CUDA Runtime API call cannot be executed because it is being called during process shut down, at a point in time after CUDA driver has been unloaded.


    .. autoattribute:: cuda.cudart.cudaError_t.cudaErrorProfilerDisabled


        This indicates profiler is not initialized for this run. This can happen when the application is running with external profiling tools like visual profiler.


    .. autoattribute:: cuda.cudart.cudaError_t.cudaErrorProfilerNotInitialized


        [Deprecated]


    .. autoattribute:: cuda.cudart.cudaError_t.cudaErrorProfilerAlreadyStarted


        [Deprecated]


    .. autoattribute:: cuda.cudart.cudaError_t.cudaErrorProfilerAlreadyStopped


        [Deprecated]


    .. autoattribute:: cuda.cudart.cudaError_t.cudaErrorInvalidConfiguration


        This indicates that a kernel launch is requesting resources that can never be satisfied by the current device. Requesting more shared memory per block than the device supports will trigger this error, as will requesting too many threads or blocks. See :py:obj:`~.cudaDeviceProp` for more device limitations.


    .. autoattribute:: cuda.cudart.cudaError_t.cudaErrorInvalidPitchValue


        This indicates that one or more of the pitch-related parameters passed to the API call is not within the acceptable range for pitch.


    .. autoattribute:: cuda.cudart.cudaError_t.cudaErrorInvalidSymbol


        This indicates that the symbol name/identifier passed to the API call is not a valid name or identifier.


    .. autoattribute:: cuda.cudart.cudaError_t.cudaErrorInvalidHostPointer


        This indicates that at least one host pointer passed to the API call is not a valid host pointer. [Deprecated]


    .. autoattribute:: cuda.cudart.cudaError_t.cudaErrorInvalidDevicePointer


        This indicates that at least one device pointer passed to the API call is not a valid device pointer. [Deprecated]


    .. autoattribute:: cuda.cudart.cudaError_t.cudaErrorInvalidTexture


        This indicates that the texture passed to the API call is not a valid texture.


    .. autoattribute:: cuda.cudart.cudaError_t.cudaErrorInvalidTextureBinding


        This indicates that the texture binding is not valid. This occurs if you call :py:obj:`~.cudaGetTextureAlignmentOffset()` with an unbound texture.


    .. autoattribute:: cuda.cudart.cudaError_t.cudaErrorInvalidChannelDescriptor


        This indicates that the channel descriptor passed to the API call is not valid. This occurs if the format is not one of the formats specified by :py:obj:`~.cudaChannelFormatKind`, or if one of the dimensions is invalid.


    .. autoattribute:: cuda.cudart.cudaError_t.cudaErrorInvalidMemcpyDirection


        This indicates that the direction of the memcpy passed to the API call is not one of the types specified by :py:obj:`~.cudaMemcpyKind`.


    .. autoattribute:: cuda.cudart.cudaError_t.cudaErrorAddressOfConstant


        This indicated that the user has taken the address of a constant variable, which was forbidden up until the CUDA 3.1 release. [Deprecated]


    .. autoattribute:: cuda.cudart.cudaError_t.cudaErrorTextureFetchFailed


        This indicated that a texture fetch was not able to be performed. This was previously used for device emulation of texture operations. [Deprecated]


    .. autoattribute:: cuda.cudart.cudaError_t.cudaErrorTextureNotBound


        This indicated that a texture was not bound for access. This was previously used for device emulation of texture operations. [Deprecated]


    .. autoattribute:: cuda.cudart.cudaError_t.cudaErrorSynchronizationError


        This indicated that a synchronization operation had failed. This was previously used for some device emulation functions. [Deprecated]


    .. autoattribute:: cuda.cudart.cudaError_t.cudaErrorInvalidFilterSetting


        This indicates that a non-float texture was being accessed with linear filtering. This is not supported by CUDA.


    .. autoattribute:: cuda.cudart.cudaError_t.cudaErrorInvalidNormSetting


        This indicates that an attempt was made to read a non-float texture as a normalized float. This is not supported by CUDA.


    .. autoattribute:: cuda.cudart.cudaError_t.cudaErrorMixedDeviceExecution


        Mixing of device and device emulation code was not allowed. [Deprecated]


    .. autoattribute:: cuda.cudart.cudaError_t.cudaErrorNotYetImplemented


        This indicates that the API call is not yet implemented. Production releases of CUDA will never return this error. [Deprecated]


    .. autoattribute:: cuda.cudart.cudaError_t.cudaErrorMemoryValueTooLarge


        This indicated that an emulated device pointer exceeded the 32-bit address range. [Deprecated]


    .. autoattribute:: cuda.cudart.cudaError_t.cudaErrorStubLibrary


        This indicates that the CUDA driver that the application has loaded is a stub library. Applications that run with the stub rather than a real driver loaded will result in CUDA API returning this error.


    .. autoattribute:: cuda.cudart.cudaError_t.cudaErrorInsufficientDriver


        This indicates that the installed NVIDIA CUDA driver is older than the CUDA runtime library. This is not a supported configuration. Users should install an updated NVIDIA display driver to allow the application to run.


    .. autoattribute:: cuda.cudart.cudaError_t.cudaErrorCallRequiresNewerDriver


        This indicates that the API call requires a newer CUDA driver than the one currently installed. Users should install an updated NVIDIA CUDA driver to allow the API call to succeed.


    .. autoattribute:: cuda.cudart.cudaError_t.cudaErrorInvalidSurface


        This indicates that the surface passed to the API call is not a valid surface.


    .. autoattribute:: cuda.cudart.cudaError_t.cudaErrorDuplicateVariableName


        This indicates that multiple global or constant variables (across separate CUDA source files in the application) share the same string name.


    .. autoattribute:: cuda.cudart.cudaError_t.cudaErrorDuplicateTextureName


        This indicates that multiple textures (across separate CUDA source files in the application) share the same string name.


    .. autoattribute:: cuda.cudart.cudaError_t.cudaErrorDuplicateSurfaceName


        This indicates that multiple surfaces (across separate CUDA source files in the application) share the same string name.


    .. autoattribute:: cuda.cudart.cudaError_t.cudaErrorDevicesUnavailable


        This indicates that all CUDA devices are busy or unavailable at the current time. Devices are often busy/unavailable due to use of :py:obj:`~.cudaComputeModeProhibited`, :py:obj:`~.cudaComputeModeExclusiveProcess`, or when long running CUDA kernels have filled up the GPU and are blocking new work from starting. They can also be unavailable due to memory constraints on a device that already has active CUDA work being performed.


    .. autoattribute:: cuda.cudart.cudaError_t.cudaErrorIncompatibleDriverContext


        This indicates that the current context is not compatible with this the CUDA Runtime. This can only occur if you are using CUDA Runtime/Driver interoperability and have created an existing Driver context using the driver API. The Driver context may be incompatible either because the Driver context was created using an older version of the API, because the Runtime API call expects a primary driver context and the Driver context is not primary, or because the Driver context has been destroyed. Please see :py:obj:`~.Interactions`with the CUDA Driver API" for more information.


    .. autoattribute:: cuda.cudart.cudaError_t.cudaErrorMissingConfiguration


        The device function being invoked (usually via :py:obj:`~.cudaLaunchKernel()`) was not previously configured via the :py:obj:`~.cudaConfigureCall()` function.


    .. autoattribute:: cuda.cudart.cudaError_t.cudaErrorPriorLaunchFailure


        This indicated that a previous kernel launch failed. This was previously used for device emulation of kernel launches. [Deprecated]


    .. autoattribute:: cuda.cudart.cudaError_t.cudaErrorLaunchMaxDepthExceeded


        This error indicates that a device runtime grid launch did not occur because the depth of the child grid would exceed the maximum supported number of nested grid launches.


    .. autoattribute:: cuda.cudart.cudaError_t.cudaErrorLaunchFileScopedTex


        This error indicates that a grid launch did not occur because the kernel uses file-scoped textures which are unsupported by the device runtime. Kernels launched via the device runtime only support textures created with the Texture Object API's.


    .. autoattribute:: cuda.cudart.cudaError_t.cudaErrorLaunchFileScopedSurf


        This error indicates that a grid launch did not occur because the kernel uses file-scoped surfaces which are unsupported by the device runtime. Kernels launched via the device runtime only support surfaces created with the Surface Object API's.


    .. autoattribute:: cuda.cudart.cudaError_t.cudaErrorSyncDepthExceeded


        This error indicates that a call to :py:obj:`~.cudaDeviceSynchronize` made from the device runtime failed because the call was made at grid depth greater than than either the default (2 levels of grids) or user specified device limit :py:obj:`~.cudaLimitDevRuntimeSyncDepth`. To be able to synchronize on launched grids at a greater depth successfully, the maximum nested depth at which :py:obj:`~.cudaDeviceSynchronize` will be called must be specified with the :py:obj:`~.cudaLimitDevRuntimeSyncDepth` limit to the :py:obj:`~.cudaDeviceSetLimit` api before the host-side launch of a kernel using the device runtime. Keep in mind that additional levels of sync depth require the runtime to reserve large amounts of device memory that cannot be used for user allocations.


    .. autoattribute:: cuda.cudart.cudaError_t.cudaErrorLaunchPendingCountExceeded


        This error indicates that a device runtime grid launch failed because the launch would exceed the limit :py:obj:`~.cudaLimitDevRuntimePendingLaunchCount`. For this launch to proceed successfully, :py:obj:`~.cudaDeviceSetLimit` must be called to set the :py:obj:`~.cudaLimitDevRuntimePendingLaunchCount` to be higher than the upper bound of outstanding launches that can be issued to the device runtime. Keep in mind that raising the limit of pending device runtime launches will require the runtime to reserve device memory that cannot be used for user allocations.


    .. autoattribute:: cuda.cudart.cudaError_t.cudaErrorInvalidDeviceFunction


        The requested device function does not exist or is not compiled for the proper device architecture.


    .. autoattribute:: cuda.cudart.cudaError_t.cudaErrorNoDevice


        This indicates that no CUDA-capable devices were detected by the installed CUDA driver.


    .. autoattribute:: cuda.cudart.cudaError_t.cudaErrorInvalidDevice


        This indicates that the device ordinal supplied by the user does not correspond to a valid CUDA device or that the action requested is invalid for the specified device.


    .. autoattribute:: cuda.cudart.cudaError_t.cudaErrorDeviceNotLicensed


        This indicates that the device doesn't have a valid Grid License.


    .. autoattribute:: cuda.cudart.cudaError_t.cudaErrorSoftwareValidityNotEstablished


        By default, the CUDA runtime may perform a minimal set of self-tests, as well as CUDA driver tests, to establish the validity of both. Introduced in CUDA 11.2, this error return indicates that at least one of these tests has failed and the validity of either the runtime or the driver could not be established.


    .. autoattribute:: cuda.cudart.cudaError_t.cudaErrorStartupFailure


        This indicates an internal startup failure in the CUDA runtime.


    .. autoattribute:: cuda.cudart.cudaError_t.cudaErrorInvalidKernelImage


        This indicates that the device kernel image is invalid.


    .. autoattribute:: cuda.cudart.cudaError_t.cudaErrorDeviceUninitialized


        This most frequently indicates that there is no context bound to the current thread. This can also be returned if the context passed to an API call is not a valid handle (such as a context that has had :py:obj:`~.cuCtxDestroy()` invoked on it). This can also be returned if a user mixes different API versions (i.e. 3010 context with 3020 API calls). See :py:obj:`~.cuCtxGetApiVersion()` for more details.


    .. autoattribute:: cuda.cudart.cudaError_t.cudaErrorMapBufferObjectFailed


        This indicates that the buffer object could not be mapped.


    .. autoattribute:: cuda.cudart.cudaError_t.cudaErrorUnmapBufferObjectFailed


        This indicates that the buffer object could not be unmapped.


    .. autoattribute:: cuda.cudart.cudaError_t.cudaErrorArrayIsMapped


        This indicates that the specified array is currently mapped and thus cannot be destroyed.


    .. autoattribute:: cuda.cudart.cudaError_t.cudaErrorAlreadyMapped


        This indicates that the resource is already mapped.


    .. autoattribute:: cuda.cudart.cudaError_t.cudaErrorNoKernelImageForDevice


        This indicates that there is no kernel image available that is suitable for the device. This can occur when a user specifies code generation options for a particular CUDA source file that do not include the corresponding device configuration.


    .. autoattribute:: cuda.cudart.cudaError_t.cudaErrorAlreadyAcquired


        This indicates that a resource has already been acquired.


    .. autoattribute:: cuda.cudart.cudaError_t.cudaErrorNotMapped


        This indicates that a resource is not mapped.


    .. autoattribute:: cuda.cudart.cudaError_t.cudaErrorNotMappedAsArray


        This indicates that a mapped resource is not available for access as an array.


    .. autoattribute:: cuda.cudart.cudaError_t.cudaErrorNotMappedAsPointer


        This indicates that a mapped resource is not available for access as a pointer.


    .. autoattribute:: cuda.cudart.cudaError_t.cudaErrorECCUncorrectable


        This indicates that an uncorrectable ECC error was detected during execution.


    .. autoattribute:: cuda.cudart.cudaError_t.cudaErrorUnsupportedLimit


        This indicates that the :py:obj:`~.cudaLimit` passed to the API call is not supported by the active device.


    .. autoattribute:: cuda.cudart.cudaError_t.cudaErrorDeviceAlreadyInUse


        This indicates that a call tried to access an exclusive-thread device that is already in use by a different thread.


    .. autoattribute:: cuda.cudart.cudaError_t.cudaErrorPeerAccessUnsupported


        This error indicates that P2P access is not supported across the given devices.


    .. autoattribute:: cuda.cudart.cudaError_t.cudaErrorInvalidPtx


        A PTX compilation failed. The runtime may fall back to compiling PTX if an application does not contain a suitable binary for the current device.


    .. autoattribute:: cuda.cudart.cudaError_t.cudaErrorInvalidGraphicsContext


        This indicates an error with the OpenGL or DirectX context.


    .. autoattribute:: cuda.cudart.cudaError_t.cudaErrorNvlinkUncorrectable


        This indicates that an uncorrectable NVLink error was detected during the execution.


    .. autoattribute:: cuda.cudart.cudaError_t.cudaErrorJitCompilerNotFound


        This indicates that the PTX JIT compiler library was not found. The JIT Compiler library is used for PTX compilation. The runtime may fall back to compiling PTX if an application does not contain a suitable binary for the current device.


    .. autoattribute:: cuda.cudart.cudaError_t.cudaErrorUnsupportedPtxVersion


        This indicates that the provided PTX was compiled with an unsupported toolchain. The most common reason for this, is the PTX was generated by a compiler newer than what is supported by the CUDA driver and PTX JIT compiler.


    .. autoattribute:: cuda.cudart.cudaError_t.cudaErrorJitCompilationDisabled


        This indicates that the JIT compilation was disabled. The JIT compilation compiles PTX. The runtime may fall back to compiling PTX if an application does not contain a suitable binary for the current device.


    .. autoattribute:: cuda.cudart.cudaError_t.cudaErrorUnsupportedExecAffinity


        This indicates that the provided execution affinity is not supported by the device.


    .. autoattribute:: cuda.cudart.cudaError_t.cudaErrorInvalidSource


        This indicates that the device kernel source is invalid.


    .. autoattribute:: cuda.cudart.cudaError_t.cudaErrorFileNotFound


        This indicates that the file specified was not found.


    .. autoattribute:: cuda.cudart.cudaError_t.cudaErrorSharedObjectSymbolNotFound


        This indicates that a link to a shared object failed to resolve.


    .. autoattribute:: cuda.cudart.cudaError_t.cudaErrorSharedObjectInitFailed


        This indicates that initialization of a shared object failed.


    .. autoattribute:: cuda.cudart.cudaError_t.cudaErrorOperatingSystem


        This error indicates that an OS call failed.


    .. autoattribute:: cuda.cudart.cudaError_t.cudaErrorInvalidResourceHandle


        This indicates that a resource handle passed to the API call was not valid. Resource handles are opaque types like :py:obj:`~.cudaStream_t` and :py:obj:`~.cudaEvent_t`.


    .. autoattribute:: cuda.cudart.cudaError_t.cudaErrorIllegalState


        This indicates that a resource required by the API call is not in a valid state to perform the requested operation.


    .. autoattribute:: cuda.cudart.cudaError_t.cudaErrorSymbolNotFound


        This indicates that a named symbol was not found. Examples of symbols are global/constant variable names, driver function names, texture names, and surface names.


    .. autoattribute:: cuda.cudart.cudaError_t.cudaErrorNotReady


        This indicates that asynchronous operations issued previously have not completed yet. This result is not actually an error, but must be indicated differently than :py:obj:`~.cudaSuccess` (which indicates completion). Calls that may return this value include :py:obj:`~.cudaEventQuery()` and :py:obj:`~.cudaStreamQuery()`.


    .. autoattribute:: cuda.cudart.cudaError_t.cudaErrorIllegalAddress


        The device encountered a load or store instruction on an invalid memory address. This leaves the process in an inconsistent state and any further CUDA work will return the same error. To continue using CUDA, the process must be terminated and relaunched.


    .. autoattribute:: cuda.cudart.cudaError_t.cudaErrorLaunchOutOfResources


        This indicates that a launch did not occur because it did not have appropriate resources. Although this error is similar to :py:obj:`~.cudaErrorInvalidConfiguration`, this error usually indicates that the user has attempted to pass too many arguments to the device kernel, or the kernel launch specifies too many threads for the kernel's register count.


    .. autoattribute:: cuda.cudart.cudaError_t.cudaErrorLaunchTimeout


        This indicates that the device kernel took too long to execute. This can only occur if timeouts are enabled - see the device property :py:obj:`~.kernelExecTimeoutEnabled` for more information. This leaves the process in an inconsistent state and any further CUDA work will return the same error. To continue using CUDA, the process must be terminated and relaunched.


    .. autoattribute:: cuda.cudart.cudaError_t.cudaErrorLaunchIncompatibleTexturing


        This error indicates a kernel launch that uses an incompatible texturing mode.


    .. autoattribute:: cuda.cudart.cudaError_t.cudaErrorPeerAccessAlreadyEnabled


        This error indicates that a call to :py:obj:`~.cudaDeviceEnablePeerAccess()` is trying to re-enable peer addressing on from a context which has already had peer addressing enabled.


    .. autoattribute:: cuda.cudart.cudaError_t.cudaErrorPeerAccessNotEnabled


        This error indicates that :py:obj:`~.cudaDeviceDisablePeerAccess()` is trying to disable peer addressing which has not been enabled yet via :py:obj:`~.cudaDeviceEnablePeerAccess()`.


    .. autoattribute:: cuda.cudart.cudaError_t.cudaErrorSetOnActiveProcess


        This indicates that the user has called :py:obj:`~.cudaSetValidDevices()`, :py:obj:`~.cudaSetDeviceFlags()`, :py:obj:`~.cudaD3D9SetDirect3DDevice()`, :py:obj:`~.cudaD3D10SetDirect3DDevice`, :py:obj:`~.cudaD3D11SetDirect3DDevice()`, or :py:obj:`~.cudaVDPAUSetVDPAUDevice()` after initializing the CUDA runtime by calling non-device management operations (allocating memory and launching kernels are examples of non-device management operations). This error can also be returned if using runtime/driver interoperability and there is an existing :py:obj:`~.CUcontext` active on the host thread.


    .. autoattribute:: cuda.cudart.cudaError_t.cudaErrorContextIsDestroyed


        This error indicates that the context current to the calling thread has been destroyed using :py:obj:`~.cuCtxDestroy`, or is a primary context which has not yet been initialized.


    .. autoattribute:: cuda.cudart.cudaError_t.cudaErrorAssert


        An assert triggered in device code during kernel execution. The device cannot be used again. All existing allocations are invalid. To continue using CUDA, the process must be terminated and relaunched.


    .. autoattribute:: cuda.cudart.cudaError_t.cudaErrorTooManyPeers


        This error indicates that the hardware resources required to enable peer access have been exhausted for one or more of the devices passed to :py:obj:`~.cudaEnablePeerAccess()`.


    .. autoattribute:: cuda.cudart.cudaError_t.cudaErrorHostMemoryAlreadyRegistered


        This error indicates that the memory range passed to :py:obj:`~.cudaHostRegister()` has already been registered.


    .. autoattribute:: cuda.cudart.cudaError_t.cudaErrorHostMemoryNotRegistered


        This error indicates that the pointer passed to :py:obj:`~.cudaHostUnregister()` does not correspond to any currently registered memory region.


    .. autoattribute:: cuda.cudart.cudaError_t.cudaErrorHardwareStackError


        Device encountered an error in the call stack during kernel execution, possibly due to stack corruption or exceeding the stack size limit. This leaves the process in an inconsistent state and any further CUDA work will return the same error. To continue using CUDA, the process must be terminated and relaunched.


    .. autoattribute:: cuda.cudart.cudaError_t.cudaErrorIllegalInstruction


        The device encountered an illegal instruction during kernel execution This leaves the process in an inconsistent state and any further CUDA work will return the same error. To continue using CUDA, the process must be terminated and relaunched.


    .. autoattribute:: cuda.cudart.cudaError_t.cudaErrorMisalignedAddress


        The device encountered a load or store instruction on a memory address which is not aligned. This leaves the process in an inconsistent state and any further CUDA work will return the same error. To continue using CUDA, the process must be terminated and relaunched.


    .. autoattribute:: cuda.cudart.cudaError_t.cudaErrorInvalidAddressSpace


        While executing a kernel, the device encountered an instruction which can only operate on memory locations in certain address spaces (global, shared, or local), but was supplied a memory address not belonging to an allowed address space. This leaves the process in an inconsistent state and any further CUDA work will return the same error. To continue using CUDA, the process must be terminated and relaunched.


    .. autoattribute:: cuda.cudart.cudaError_t.cudaErrorInvalidPc


        The device encountered an invalid program counter. This leaves the process in an inconsistent state and any further CUDA work will return the same error. To continue using CUDA, the process must be terminated and relaunched.


    .. autoattribute:: cuda.cudart.cudaError_t.cudaErrorLaunchFailure


        An exception occurred on the device while executing a kernel. Common causes include dereferencing an invalid device pointer and accessing out of bounds shared memory. Less common cases can be system specific - more information about these cases can be found in the system specific user guide. This leaves the process in an inconsistent state and any further CUDA work will return the same error. To continue using CUDA, the process must be terminated and relaunched.


    .. autoattribute:: cuda.cudart.cudaError_t.cudaErrorCooperativeLaunchTooLarge


        This error indicates that the number of blocks launched per grid for a kernel that was launched via either :py:obj:`~.cudaLaunchCooperativeKernel` or :py:obj:`~.cudaLaunchCooperativeKernelMultiDevice` exceeds the maximum number of blocks as allowed by :py:obj:`~.cudaOccupancyMaxActiveBlocksPerMultiprocessor` or :py:obj:`~.cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags` times the number of multiprocessors as specified by the device attribute :py:obj:`~.cudaDevAttrMultiProcessorCount`.


    .. autoattribute:: cuda.cudart.cudaError_t.cudaErrorNotPermitted


        This error indicates the attempted operation is not permitted.


    .. autoattribute:: cuda.cudart.cudaError_t.cudaErrorNotSupported


        This error indicates the attempted operation is not supported on the current system or device.


    .. autoattribute:: cuda.cudart.cudaError_t.cudaErrorSystemNotReady


        This error indicates that the system is not yet ready to start any CUDA work. To continue using CUDA, verify the system configuration is in a valid state and all required driver daemons are actively running. More information about this error can be found in the system specific user guide.


    .. autoattribute:: cuda.cudart.cudaError_t.cudaErrorSystemDriverMismatch


        This error indicates that there is a mismatch between the versions of the display driver and the CUDA driver. Refer to the compatibility documentation for supported versions.


    .. autoattribute:: cuda.cudart.cudaError_t.cudaErrorCompatNotSupportedOnDevice


        This error indicates that the system was upgraded to run with forward compatibility but the visible hardware detected by CUDA does not support this configuration. Refer to the compatibility documentation for the supported hardware matrix or ensure that only supported hardware is visible during initialization via the CUDA_VISIBLE_DEVICES environment variable.


    .. autoattribute:: cuda.cudart.cudaError_t.cudaErrorMpsConnectionFailed


        This error indicates that the MPS client failed to connect to the MPS control daemon or the MPS server.


    .. autoattribute:: cuda.cudart.cudaError_t.cudaErrorMpsRpcFailure


        This error indicates that the remote procedural call between the MPS server and the MPS client failed.


    .. autoattribute:: cuda.cudart.cudaError_t.cudaErrorMpsServerNotReady


        This error indicates that the MPS server is not ready to accept new MPS client requests. This error can be returned when the MPS server is in the process of recovering from a fatal failure.


    .. autoattribute:: cuda.cudart.cudaError_t.cudaErrorMpsMaxClientsReached


        This error indicates that the hardware resources required to create MPS client have been exhausted.


    .. autoattribute:: cuda.cudart.cudaError_t.cudaErrorMpsMaxConnectionsReached


        This error indicates the the hardware resources required to device connections have been exhausted.


    .. autoattribute:: cuda.cudart.cudaError_t.cudaErrorStreamCaptureUnsupported


        The operation is not permitted when the stream is capturing.


    .. autoattribute:: cuda.cudart.cudaError_t.cudaErrorStreamCaptureInvalidated


        The current capture sequence on the stream has been invalidated due to a previous error.


    .. autoattribute:: cuda.cudart.cudaError_t.cudaErrorStreamCaptureMerge


        The operation would have resulted in a merge of two independent capture sequences.


    .. autoattribute:: cuda.cudart.cudaError_t.cudaErrorStreamCaptureUnmatched


        The capture was not initiated in this stream.


    .. autoattribute:: cuda.cudart.cudaError_t.cudaErrorStreamCaptureUnjoined


        The capture sequence contains a fork that was not joined to the primary stream.


    .. autoattribute:: cuda.cudart.cudaError_t.cudaErrorStreamCaptureIsolation


        A dependency would have been created which crosses the capture sequence boundary. Only implicit in-stream ordering dependencies are allowed to cross the boundary.


    .. autoattribute:: cuda.cudart.cudaError_t.cudaErrorStreamCaptureImplicit


        The operation would have resulted in a disallowed implicit dependency on a current capture sequence from cudaStreamLegacy.


    .. autoattribute:: cuda.cudart.cudaError_t.cudaErrorCapturedEvent


        The operation is not permitted on an event which was last recorded in a capturing stream.


    .. autoattribute:: cuda.cudart.cudaError_t.cudaErrorStreamCaptureWrongThread


        A stream capture sequence not initiated with the :py:obj:`~.cudaStreamCaptureModeRelaxed` argument to :py:obj:`~.cudaStreamBeginCapture` was passed to :py:obj:`~.cudaStreamEndCapture` in a different thread.


    .. autoattribute:: cuda.cudart.cudaError_t.cudaErrorTimeout


        This indicates that the wait operation has timed out.


    .. autoattribute:: cuda.cudart.cudaError_t.cudaErrorGraphExecUpdateFailure


        This error indicates that the graph update was not performed because it included changes which violated constraints specific to instantiated graph update.


    .. autoattribute:: cuda.cudart.cudaError_t.cudaErrorExternalDevice


        This indicates that an async error has occurred in a device outside of CUDA. If CUDA was waiting for an external device's signal before consuming shared data, the external device signaled an error indicating that the data is not valid for consumption. This leaves the process in an inconsistent state and any further CUDA work will return the same error. To continue using CUDA, the process must be terminated and relaunched.


    .. autoattribute:: cuda.cudart.cudaError_t.cudaErrorUnknown


        This indicates that an unknown internal error has occurred.


    .. autoattribute:: cuda.cudart.cudaError_t.cudaErrorApiFailureBase


        Any unhandled CUDA driver error is added to this value and returned via the runtime. Production releases of CUDA should not return such errors. [Deprecated]

.. autoclass:: cuda.cudart.cudaChannelFormatKind

    .. autoattribute:: cuda.cudart.cudaChannelFormatKind.cudaChannelFormatKindSigned


        Signed channel format


    .. autoattribute:: cuda.cudart.cudaChannelFormatKind.cudaChannelFormatKindUnsigned


        Unsigned channel format


    .. autoattribute:: cuda.cudart.cudaChannelFormatKind.cudaChannelFormatKindFloat


        Float channel format


    .. autoattribute:: cuda.cudart.cudaChannelFormatKind.cudaChannelFormatKindNone


        No channel format


    .. autoattribute:: cuda.cudart.cudaChannelFormatKind.cudaChannelFormatKindNV12


        Unsigned 8-bit integers, planar 4:2:0 YUV format


    .. autoattribute:: cuda.cudart.cudaChannelFormatKind.cudaChannelFormatKindUnsignedNormalized8X1


        1 channel unsigned 8-bit normalized integer


    .. autoattribute:: cuda.cudart.cudaChannelFormatKind.cudaChannelFormatKindUnsignedNormalized8X2


        2 channel unsigned 8-bit normalized integer


    .. autoattribute:: cuda.cudart.cudaChannelFormatKind.cudaChannelFormatKindUnsignedNormalized8X4


        4 channel unsigned 8-bit normalized integer


    .. autoattribute:: cuda.cudart.cudaChannelFormatKind.cudaChannelFormatKindUnsignedNormalized16X1


        1 channel unsigned 16-bit normalized integer


    .. autoattribute:: cuda.cudart.cudaChannelFormatKind.cudaChannelFormatKindUnsignedNormalized16X2


        2 channel unsigned 16-bit normalized integer


    .. autoattribute:: cuda.cudart.cudaChannelFormatKind.cudaChannelFormatKindUnsignedNormalized16X4


        4 channel unsigned 16-bit normalized integer


    .. autoattribute:: cuda.cudart.cudaChannelFormatKind.cudaChannelFormatKindSignedNormalized8X1


        1 channel signed 8-bit normalized integer


    .. autoattribute:: cuda.cudart.cudaChannelFormatKind.cudaChannelFormatKindSignedNormalized8X2


        2 channel signed 8-bit normalized integer


    .. autoattribute:: cuda.cudart.cudaChannelFormatKind.cudaChannelFormatKindSignedNormalized8X4


        4 channel signed 8-bit normalized integer


    .. autoattribute:: cuda.cudart.cudaChannelFormatKind.cudaChannelFormatKindSignedNormalized16X1


        1 channel signed 16-bit normalized integer


    .. autoattribute:: cuda.cudart.cudaChannelFormatKind.cudaChannelFormatKindSignedNormalized16X2


        2 channel signed 16-bit normalized integer


    .. autoattribute:: cuda.cudart.cudaChannelFormatKind.cudaChannelFormatKindSignedNormalized16X4


        4 channel signed 16-bit normalized integer


    .. autoattribute:: cuda.cudart.cudaChannelFormatKind.cudaChannelFormatKindUnsignedBlockCompressed1


        4 channel unsigned normalized block-compressed (BC1 compression) format


    .. autoattribute:: cuda.cudart.cudaChannelFormatKind.cudaChannelFormatKindUnsignedBlockCompressed1SRGB


        4 channel unsigned normalized block-compressed (BC1 compression) format with sRGB encoding


    .. autoattribute:: cuda.cudart.cudaChannelFormatKind.cudaChannelFormatKindUnsignedBlockCompressed2


        4 channel unsigned normalized block-compressed (BC2 compression) format


    .. autoattribute:: cuda.cudart.cudaChannelFormatKind.cudaChannelFormatKindUnsignedBlockCompressed2SRGB


        4 channel unsigned normalized block-compressed (BC2 compression) format with sRGB encoding


    .. autoattribute:: cuda.cudart.cudaChannelFormatKind.cudaChannelFormatKindUnsignedBlockCompressed3


        4 channel unsigned normalized block-compressed (BC3 compression) format


    .. autoattribute:: cuda.cudart.cudaChannelFormatKind.cudaChannelFormatKindUnsignedBlockCompressed3SRGB


        4 channel unsigned normalized block-compressed (BC3 compression) format with sRGB encoding


    .. autoattribute:: cuda.cudart.cudaChannelFormatKind.cudaChannelFormatKindUnsignedBlockCompressed4


        1 channel unsigned normalized block-compressed (BC4 compression) format


    .. autoattribute:: cuda.cudart.cudaChannelFormatKind.cudaChannelFormatKindSignedBlockCompressed4


        1 channel signed normalized block-compressed (BC4 compression) format


    .. autoattribute:: cuda.cudart.cudaChannelFormatKind.cudaChannelFormatKindUnsignedBlockCompressed5


        2 channel unsigned normalized block-compressed (BC5 compression) format


    .. autoattribute:: cuda.cudart.cudaChannelFormatKind.cudaChannelFormatKindSignedBlockCompressed5


        2 channel signed normalized block-compressed (BC5 compression) format


    .. autoattribute:: cuda.cudart.cudaChannelFormatKind.cudaChannelFormatKindUnsignedBlockCompressed6H


        3 channel unsigned half-float block-compressed (BC6H compression) format


    .. autoattribute:: cuda.cudart.cudaChannelFormatKind.cudaChannelFormatKindSignedBlockCompressed6H


        3 channel signed half-float block-compressed (BC6H compression) format


    .. autoattribute:: cuda.cudart.cudaChannelFormatKind.cudaChannelFormatKindUnsignedBlockCompressed7


        4 channel unsigned normalized block-compressed (BC7 compression) format


    .. autoattribute:: cuda.cudart.cudaChannelFormatKind.cudaChannelFormatKindUnsignedBlockCompressed7SRGB


        4 channel unsigned normalized block-compressed (BC7 compression) format with sRGB encoding

.. autoclass:: cuda.cudart.cudaMemoryType

    .. autoattribute:: cuda.cudart.cudaMemoryType.cudaMemoryTypeUnregistered


        Unregistered memory


    .. autoattribute:: cuda.cudart.cudaMemoryType.cudaMemoryTypeHost


        Host memory


    .. autoattribute:: cuda.cudart.cudaMemoryType.cudaMemoryTypeDevice


        Device memory


    .. autoattribute:: cuda.cudart.cudaMemoryType.cudaMemoryTypeManaged


        Managed memory

.. autoclass:: cuda.cudart.cudaMemcpyKind

    .. autoattribute:: cuda.cudart.cudaMemcpyKind.cudaMemcpyHostToHost


        Host -> Host


    .. autoattribute:: cuda.cudart.cudaMemcpyKind.cudaMemcpyHostToDevice


        Host -> Device


    .. autoattribute:: cuda.cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost


        Device -> Host


    .. autoattribute:: cuda.cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice


        Device -> Device


    .. autoattribute:: cuda.cudart.cudaMemcpyKind.cudaMemcpyDefault


        Direction of the transfer is inferred from the pointer values. Requires unified virtual addressing

.. autoclass:: cuda.cudart.cudaAccessProperty

    .. autoattribute:: cuda.cudart.cudaAccessProperty.cudaAccessPropertyNormal


        Normal cache persistence.


    .. autoattribute:: cuda.cudart.cudaAccessProperty.cudaAccessPropertyStreaming


        Streaming access is less likely to persit from cache.


    .. autoattribute:: cuda.cudart.cudaAccessProperty.cudaAccessPropertyPersisting


        Persisting access is more likely to persist in cache.

.. autoclass:: cuda.cudart.cudaStreamCaptureStatus

    .. autoattribute:: cuda.cudart.cudaStreamCaptureStatus.cudaStreamCaptureStatusNone


        Stream is not capturing


    .. autoattribute:: cuda.cudart.cudaStreamCaptureStatus.cudaStreamCaptureStatusActive


        Stream is actively capturing


    .. autoattribute:: cuda.cudart.cudaStreamCaptureStatus.cudaStreamCaptureStatusInvalidated


        Stream is part of a capture sequence that has been invalidated, but not terminated

.. autoclass:: cuda.cudart.cudaStreamCaptureMode

    .. autoattribute:: cuda.cudart.cudaStreamCaptureMode.cudaStreamCaptureModeGlobal


    .. autoattribute:: cuda.cudart.cudaStreamCaptureMode.cudaStreamCaptureModeThreadLocal


    .. autoattribute:: cuda.cudart.cudaStreamCaptureMode.cudaStreamCaptureModeRelaxed

.. autoclass:: cuda.cudart.cudaSynchronizationPolicy

    .. autoattribute:: cuda.cudart.cudaSynchronizationPolicy.cudaSyncPolicyAuto


    .. autoattribute:: cuda.cudart.cudaSynchronizationPolicy.cudaSyncPolicySpin


    .. autoattribute:: cuda.cudart.cudaSynchronizationPolicy.cudaSyncPolicyYield


    .. autoattribute:: cuda.cudart.cudaSynchronizationPolicy.cudaSyncPolicyBlockingSync

.. autoclass:: cuda.cudart.cudaStreamUpdateCaptureDependenciesFlags

    .. autoattribute:: cuda.cudart.cudaStreamUpdateCaptureDependenciesFlags.cudaStreamAddCaptureDependencies


        Add new nodes to the dependency set


    .. autoattribute:: cuda.cudart.cudaStreamUpdateCaptureDependenciesFlags.cudaStreamSetCaptureDependencies


        Replace the dependency set with the new nodes

.. autoclass:: cuda.cudart.cudaUserObjectFlags

    .. autoattribute:: cuda.cudart.cudaUserObjectFlags.cudaUserObjectNoDestructorSync


        Indicates the destructor execution is not synchronized by any CUDA handle.

.. autoclass:: cuda.cudart.cudaUserObjectRetainFlags

    .. autoattribute:: cuda.cudart.cudaUserObjectRetainFlags.cudaGraphUserObjectMove


        Transfer references from the caller rather than creating new references.

.. autoclass:: cuda.cudart.cudaGraphicsRegisterFlags

    .. autoattribute:: cuda.cudart.cudaGraphicsRegisterFlags.cudaGraphicsRegisterFlagsNone


        Default


    .. autoattribute:: cuda.cudart.cudaGraphicsRegisterFlags.cudaGraphicsRegisterFlagsReadOnly


        CUDA will not write to this resource


    .. autoattribute:: cuda.cudart.cudaGraphicsRegisterFlags.cudaGraphicsRegisterFlagsWriteDiscard


        CUDA will only write to and will not read from this resource


    .. autoattribute:: cuda.cudart.cudaGraphicsRegisterFlags.cudaGraphicsRegisterFlagsSurfaceLoadStore


        CUDA will bind this resource to a surface reference


    .. autoattribute:: cuda.cudart.cudaGraphicsRegisterFlags.cudaGraphicsRegisterFlagsTextureGather


        CUDA will perform texture gather operations on this resource

.. autoclass:: cuda.cudart.cudaGraphicsMapFlags

    .. autoattribute:: cuda.cudart.cudaGraphicsMapFlags.cudaGraphicsMapFlagsNone


        Default; Assume resource can be read/written


    .. autoattribute:: cuda.cudart.cudaGraphicsMapFlags.cudaGraphicsMapFlagsReadOnly


        CUDA will not write to this resource


    .. autoattribute:: cuda.cudart.cudaGraphicsMapFlags.cudaGraphicsMapFlagsWriteDiscard


        CUDA will only write to and will not read from this resource

.. autoclass:: cuda.cudart.cudaGraphicsCubeFace

    .. autoattribute:: cuda.cudart.cudaGraphicsCubeFace.cudaGraphicsCubeFacePositiveX


        Positive X face of cubemap


    .. autoattribute:: cuda.cudart.cudaGraphicsCubeFace.cudaGraphicsCubeFaceNegativeX


        Negative X face of cubemap


    .. autoattribute:: cuda.cudart.cudaGraphicsCubeFace.cudaGraphicsCubeFacePositiveY


        Positive Y face of cubemap


    .. autoattribute:: cuda.cudart.cudaGraphicsCubeFace.cudaGraphicsCubeFaceNegativeY


        Negative Y face of cubemap


    .. autoattribute:: cuda.cudart.cudaGraphicsCubeFace.cudaGraphicsCubeFacePositiveZ


        Positive Z face of cubemap


    .. autoattribute:: cuda.cudart.cudaGraphicsCubeFace.cudaGraphicsCubeFaceNegativeZ


        Negative Z face of cubemap

.. autoclass:: cuda.cudart.cudaResourceType

    .. autoattribute:: cuda.cudart.cudaResourceType.cudaResourceTypeArray


        Array resource


    .. autoattribute:: cuda.cudart.cudaResourceType.cudaResourceTypeMipmappedArray


        Mipmapped array resource


    .. autoattribute:: cuda.cudart.cudaResourceType.cudaResourceTypeLinear


        Linear resource


    .. autoattribute:: cuda.cudart.cudaResourceType.cudaResourceTypePitch2D


        Pitch 2D resource

.. autoclass:: cuda.cudart.cudaResourceViewFormat

    .. autoattribute:: cuda.cudart.cudaResourceViewFormat.cudaResViewFormatNone


        No resource view format (use underlying resource format)


    .. autoattribute:: cuda.cudart.cudaResourceViewFormat.cudaResViewFormatUnsignedChar1


        1 channel unsigned 8-bit integers


    .. autoattribute:: cuda.cudart.cudaResourceViewFormat.cudaResViewFormatUnsignedChar2


        2 channel unsigned 8-bit integers


    .. autoattribute:: cuda.cudart.cudaResourceViewFormat.cudaResViewFormatUnsignedChar4


        4 channel unsigned 8-bit integers


    .. autoattribute:: cuda.cudart.cudaResourceViewFormat.cudaResViewFormatSignedChar1


        1 channel signed 8-bit integers


    .. autoattribute:: cuda.cudart.cudaResourceViewFormat.cudaResViewFormatSignedChar2


        2 channel signed 8-bit integers


    .. autoattribute:: cuda.cudart.cudaResourceViewFormat.cudaResViewFormatSignedChar4


        4 channel signed 8-bit integers


    .. autoattribute:: cuda.cudart.cudaResourceViewFormat.cudaResViewFormatUnsignedShort1


        1 channel unsigned 16-bit integers


    .. autoattribute:: cuda.cudart.cudaResourceViewFormat.cudaResViewFormatUnsignedShort2


        2 channel unsigned 16-bit integers


    .. autoattribute:: cuda.cudart.cudaResourceViewFormat.cudaResViewFormatUnsignedShort4


        4 channel unsigned 16-bit integers


    .. autoattribute:: cuda.cudart.cudaResourceViewFormat.cudaResViewFormatSignedShort1


        1 channel signed 16-bit integers


    .. autoattribute:: cuda.cudart.cudaResourceViewFormat.cudaResViewFormatSignedShort2


        2 channel signed 16-bit integers


    .. autoattribute:: cuda.cudart.cudaResourceViewFormat.cudaResViewFormatSignedShort4


        4 channel signed 16-bit integers


    .. autoattribute:: cuda.cudart.cudaResourceViewFormat.cudaResViewFormatUnsignedInt1


        1 channel unsigned 32-bit integers


    .. autoattribute:: cuda.cudart.cudaResourceViewFormat.cudaResViewFormatUnsignedInt2


        2 channel unsigned 32-bit integers


    .. autoattribute:: cuda.cudart.cudaResourceViewFormat.cudaResViewFormatUnsignedInt4


        4 channel unsigned 32-bit integers


    .. autoattribute:: cuda.cudart.cudaResourceViewFormat.cudaResViewFormatSignedInt1


        1 channel signed 32-bit integers


    .. autoattribute:: cuda.cudart.cudaResourceViewFormat.cudaResViewFormatSignedInt2


        2 channel signed 32-bit integers


    .. autoattribute:: cuda.cudart.cudaResourceViewFormat.cudaResViewFormatSignedInt4


        4 channel signed 32-bit integers


    .. autoattribute:: cuda.cudart.cudaResourceViewFormat.cudaResViewFormatHalf1


        1 channel 16-bit floating point


    .. autoattribute:: cuda.cudart.cudaResourceViewFormat.cudaResViewFormatHalf2


        2 channel 16-bit floating point


    .. autoattribute:: cuda.cudart.cudaResourceViewFormat.cudaResViewFormatHalf4


        4 channel 16-bit floating point


    .. autoattribute:: cuda.cudart.cudaResourceViewFormat.cudaResViewFormatFloat1


        1 channel 32-bit floating point


    .. autoattribute:: cuda.cudart.cudaResourceViewFormat.cudaResViewFormatFloat2


        2 channel 32-bit floating point


    .. autoattribute:: cuda.cudart.cudaResourceViewFormat.cudaResViewFormatFloat4


        4 channel 32-bit floating point


    .. autoattribute:: cuda.cudart.cudaResourceViewFormat.cudaResViewFormatUnsignedBlockCompressed1


        Block compressed 1


    .. autoattribute:: cuda.cudart.cudaResourceViewFormat.cudaResViewFormatUnsignedBlockCompressed2


        Block compressed 2


    .. autoattribute:: cuda.cudart.cudaResourceViewFormat.cudaResViewFormatUnsignedBlockCompressed3


        Block compressed 3


    .. autoattribute:: cuda.cudart.cudaResourceViewFormat.cudaResViewFormatUnsignedBlockCompressed4


        Block compressed 4 unsigned


    .. autoattribute:: cuda.cudart.cudaResourceViewFormat.cudaResViewFormatSignedBlockCompressed4


        Block compressed 4 signed


    .. autoattribute:: cuda.cudart.cudaResourceViewFormat.cudaResViewFormatUnsignedBlockCompressed5


        Block compressed 5 unsigned


    .. autoattribute:: cuda.cudart.cudaResourceViewFormat.cudaResViewFormatSignedBlockCompressed5


        Block compressed 5 signed


    .. autoattribute:: cuda.cudart.cudaResourceViewFormat.cudaResViewFormatUnsignedBlockCompressed6H


        Block compressed 6 unsigned half-float


    .. autoattribute:: cuda.cudart.cudaResourceViewFormat.cudaResViewFormatSignedBlockCompressed6H


        Block compressed 6 signed half-float


    .. autoattribute:: cuda.cudart.cudaResourceViewFormat.cudaResViewFormatUnsignedBlockCompressed7


        Block compressed 7

.. autoclass:: cuda.cudart.cudaFuncAttribute

    .. autoattribute:: cuda.cudart.cudaFuncAttribute.cudaFuncAttributeMaxDynamicSharedMemorySize


        Maximum dynamic shared memory size


    .. autoattribute:: cuda.cudart.cudaFuncAttribute.cudaFuncAttributePreferredSharedMemoryCarveout


        Preferred shared memory-L1 cache split


    .. autoattribute:: cuda.cudart.cudaFuncAttribute.cudaFuncAttributeMax

.. autoclass:: cuda.cudart.cudaFuncCache

    .. autoattribute:: cuda.cudart.cudaFuncCache.cudaFuncCachePreferNone


        Default function cache configuration, no preference


    .. autoattribute:: cuda.cudart.cudaFuncCache.cudaFuncCachePreferShared


        Prefer larger shared memory and smaller L1 cache


    .. autoattribute:: cuda.cudart.cudaFuncCache.cudaFuncCachePreferL1


        Prefer larger L1 cache and smaller shared memory


    .. autoattribute:: cuda.cudart.cudaFuncCache.cudaFuncCachePreferEqual


        Prefer equal size L1 cache and shared memory

.. autoclass:: cuda.cudart.cudaSharedMemConfig

    .. autoattribute:: cuda.cudart.cudaSharedMemConfig.cudaSharedMemBankSizeDefault


    .. autoattribute:: cuda.cudart.cudaSharedMemConfig.cudaSharedMemBankSizeFourByte


    .. autoattribute:: cuda.cudart.cudaSharedMemConfig.cudaSharedMemBankSizeEightByte

.. autoclass:: cuda.cudart.cudaSharedCarveout

    .. autoattribute:: cuda.cudart.cudaSharedCarveout.cudaSharedmemCarveoutDefault


        No preference for shared memory or L1 (default)


    .. autoattribute:: cuda.cudart.cudaSharedCarveout.cudaSharedmemCarveoutMaxShared


        Prefer maximum available shared memory, minimum L1 cache


    .. autoattribute:: cuda.cudart.cudaSharedCarveout.cudaSharedmemCarveoutMaxL1


        Prefer maximum available L1 cache, minimum shared memory

.. autoclass:: cuda.cudart.cudaComputeMode

    .. autoattribute:: cuda.cudart.cudaComputeMode.cudaComputeModeDefault


        Default compute mode (Multiple threads can use :py:obj:`~.cudaSetDevice()` with this device)


    .. autoattribute:: cuda.cudart.cudaComputeMode.cudaComputeModeExclusive


        Compute-exclusive-thread mode (Only one thread in one process will be able to use :py:obj:`~.cudaSetDevice()` with this device)


    .. autoattribute:: cuda.cudart.cudaComputeMode.cudaComputeModeProhibited


        Compute-prohibited mode (No threads can use :py:obj:`~.cudaSetDevice()` with this device)


    .. autoattribute:: cuda.cudart.cudaComputeMode.cudaComputeModeExclusiveProcess


        Compute-exclusive-process mode (Many threads in one process will be able to use :py:obj:`~.cudaSetDevice()` with this device)

.. autoclass:: cuda.cudart.cudaLimit

    .. autoattribute:: cuda.cudart.cudaLimit.cudaLimitStackSize


        GPU thread stack size


    .. autoattribute:: cuda.cudart.cudaLimit.cudaLimitPrintfFifoSize


        GPU printf FIFO size


    .. autoattribute:: cuda.cudart.cudaLimit.cudaLimitMallocHeapSize


        GPU malloc heap size


    .. autoattribute:: cuda.cudart.cudaLimit.cudaLimitDevRuntimeSyncDepth


        GPU device runtime synchronize depth


    .. autoattribute:: cuda.cudart.cudaLimit.cudaLimitDevRuntimePendingLaunchCount


        GPU device runtime pending launch count


    .. autoattribute:: cuda.cudart.cudaLimit.cudaLimitMaxL2FetchGranularity


        A value between 0 and 128 that indicates the maximum fetch granularity of L2 (in Bytes). This is a hint


    .. autoattribute:: cuda.cudart.cudaLimit.cudaLimitPersistingL2CacheSize


        A size in bytes for L2 persisting lines cache size

.. autoclass:: cuda.cudart.cudaMemoryAdvise

    .. autoattribute:: cuda.cudart.cudaMemoryAdvise.cudaMemAdviseSetReadMostly


        Data will mostly be read and only occassionally be written to


    .. autoattribute:: cuda.cudart.cudaMemoryAdvise.cudaMemAdviseUnsetReadMostly


        Undo the effect of :py:obj:`~.cudaMemAdviseSetReadMostly`


    .. autoattribute:: cuda.cudart.cudaMemoryAdvise.cudaMemAdviseSetPreferredLocation


        Set the preferred location for the data as the specified device


    .. autoattribute:: cuda.cudart.cudaMemoryAdvise.cudaMemAdviseUnsetPreferredLocation


        Clear the preferred location for the data


    .. autoattribute:: cuda.cudart.cudaMemoryAdvise.cudaMemAdviseSetAccessedBy


        Data will be accessed by the specified device, so prevent page faults as much as possible


    .. autoattribute:: cuda.cudart.cudaMemoryAdvise.cudaMemAdviseUnsetAccessedBy


        Let the Unified Memory subsystem decide on the page faulting policy for the specified device

.. autoclass:: cuda.cudart.cudaMemRangeAttribute

    .. autoattribute:: cuda.cudart.cudaMemRangeAttribute.cudaMemRangeAttributeReadMostly


        Whether the range will mostly be read and only occassionally be written to


    .. autoattribute:: cuda.cudart.cudaMemRangeAttribute.cudaMemRangeAttributePreferredLocation


        The preferred location of the range


    .. autoattribute:: cuda.cudart.cudaMemRangeAttribute.cudaMemRangeAttributeAccessedBy


        Memory range has :py:obj:`~.cudaMemAdviseSetAccessedBy` set for specified device


    .. autoattribute:: cuda.cudart.cudaMemRangeAttribute.cudaMemRangeAttributeLastPrefetchLocation


        The last location to which the range was prefetched

.. autoclass:: cuda.cudart.cudaOutputMode_t

    .. autoattribute:: cuda.cudart.cudaOutputMode_t.cudaKeyValuePair


        Output mode Key-Value pair format.


    .. autoattribute:: cuda.cudart.cudaOutputMode_t.cudaCSV


        Output mode Comma separated values format.

.. autoclass:: cuda.cudart.cudaFlushGPUDirectRDMAWritesOptions

    .. autoattribute:: cuda.cudart.cudaFlushGPUDirectRDMAWritesOptions.cudaFlushGPUDirectRDMAWritesOptionHost


        :py:obj:`~.cudaDeviceFlushGPUDirectRDMAWrites()` and its CUDA Driver API counterpart are supported on the device.


    .. autoattribute:: cuda.cudart.cudaFlushGPUDirectRDMAWritesOptions.cudaFlushGPUDirectRDMAWritesOptionMemOps


        The :py:obj:`~.CU_STREAM_WAIT_VALUE_FLUSH` flag and the :py:obj:`~.CU_STREAM_MEM_OP_FLUSH_REMOTE_WRITES` MemOp are supported on the CUDA device.

.. autoclass:: cuda.cudart.cudaGPUDirectRDMAWritesOrdering

    .. autoattribute:: cuda.cudart.cudaGPUDirectRDMAWritesOrdering.cudaGPUDirectRDMAWritesOrderingNone


        The device does not natively support ordering of GPUDirect RDMA writes. :py:obj:`~.cudaFlushGPUDirectRDMAWrites()` can be leveraged if supported.


    .. autoattribute:: cuda.cudart.cudaGPUDirectRDMAWritesOrdering.cudaGPUDirectRDMAWritesOrderingOwner


        Natively, the device can consistently consume GPUDirect RDMA writes, although other CUDA devices may not.


    .. autoattribute:: cuda.cudart.cudaGPUDirectRDMAWritesOrdering.cudaGPUDirectRDMAWritesOrderingAllDevices


        Any CUDA device in the system can consistently consume GPUDirect RDMA writes to this device.

.. autoclass:: cuda.cudart.cudaFlushGPUDirectRDMAWritesScope

    .. autoattribute:: cuda.cudart.cudaFlushGPUDirectRDMAWritesScope.cudaFlushGPUDirectRDMAWritesToOwner


        Blocks until remote writes are visible to the CUDA device context owning the data.


    .. autoattribute:: cuda.cudart.cudaFlushGPUDirectRDMAWritesScope.cudaFlushGPUDirectRDMAWritesToAllDevices


        Blocks until remote writes are visible to all CUDA device contexts.

.. autoclass:: cuda.cudart.cudaFlushGPUDirectRDMAWritesTarget

    .. autoattribute:: cuda.cudart.cudaFlushGPUDirectRDMAWritesTarget.cudaFlushGPUDirectRDMAWritesTargetCurrentDevice


        Sets the target for :py:obj:`~.cudaDeviceFlushGPUDirectRDMAWrites()` to the currently active CUDA device context.

.. autoclass:: cuda.cudart.cudaDeviceAttr

    .. autoattribute:: cuda.cudart.cudaDeviceAttr.cudaDevAttrMaxThreadsPerBlock


        Maximum number of threads per block


    .. autoattribute:: cuda.cudart.cudaDeviceAttr.cudaDevAttrMaxBlockDimX


        Maximum block dimension X


    .. autoattribute:: cuda.cudart.cudaDeviceAttr.cudaDevAttrMaxBlockDimY


        Maximum block dimension Y


    .. autoattribute:: cuda.cudart.cudaDeviceAttr.cudaDevAttrMaxBlockDimZ


        Maximum block dimension Z


    .. autoattribute:: cuda.cudart.cudaDeviceAttr.cudaDevAttrMaxGridDimX


        Maximum grid dimension X


    .. autoattribute:: cuda.cudart.cudaDeviceAttr.cudaDevAttrMaxGridDimY


        Maximum grid dimension Y


    .. autoattribute:: cuda.cudart.cudaDeviceAttr.cudaDevAttrMaxGridDimZ


        Maximum grid dimension Z


    .. autoattribute:: cuda.cudart.cudaDeviceAttr.cudaDevAttrMaxSharedMemoryPerBlock


        Maximum shared memory available per block in bytes


    .. autoattribute:: cuda.cudart.cudaDeviceAttr.cudaDevAttrTotalConstantMemory


        Memory available on device for constant variables in a CUDA C kernel in bytes


    .. autoattribute:: cuda.cudart.cudaDeviceAttr.cudaDevAttrWarpSize


        Warp size in threads


    .. autoattribute:: cuda.cudart.cudaDeviceAttr.cudaDevAttrMaxPitch


        Maximum pitch in bytes allowed by memory copies


    .. autoattribute:: cuda.cudart.cudaDeviceAttr.cudaDevAttrMaxRegistersPerBlock


        Maximum number of 32-bit registers available per block


    .. autoattribute:: cuda.cudart.cudaDeviceAttr.cudaDevAttrClockRate


        Peak clock frequency in kilohertz


    .. autoattribute:: cuda.cudart.cudaDeviceAttr.cudaDevAttrTextureAlignment


        Alignment requirement for textures


    .. autoattribute:: cuda.cudart.cudaDeviceAttr.cudaDevAttrGpuOverlap


        Device can possibly copy memory and execute a kernel concurrently


    .. autoattribute:: cuda.cudart.cudaDeviceAttr.cudaDevAttrMultiProcessorCount


        Number of multiprocessors on device


    .. autoattribute:: cuda.cudart.cudaDeviceAttr.cudaDevAttrKernelExecTimeout


        Specifies whether there is a run time limit on kernels


    .. autoattribute:: cuda.cudart.cudaDeviceAttr.cudaDevAttrIntegrated


        Device is integrated with host memory


    .. autoattribute:: cuda.cudart.cudaDeviceAttr.cudaDevAttrCanMapHostMemory


        Device can map host memory into CUDA address space


    .. autoattribute:: cuda.cudart.cudaDeviceAttr.cudaDevAttrComputeMode


        Compute mode (See :py:obj:`~.cudaComputeMode` for details)


    .. autoattribute:: cuda.cudart.cudaDeviceAttr.cudaDevAttrMaxTexture1DWidth


        Maximum 1D texture width


    .. autoattribute:: cuda.cudart.cudaDeviceAttr.cudaDevAttrMaxTexture2DWidth


        Maximum 2D texture width


    .. autoattribute:: cuda.cudart.cudaDeviceAttr.cudaDevAttrMaxTexture2DHeight


        Maximum 2D texture height


    .. autoattribute:: cuda.cudart.cudaDeviceAttr.cudaDevAttrMaxTexture3DWidth


        Maximum 3D texture width


    .. autoattribute:: cuda.cudart.cudaDeviceAttr.cudaDevAttrMaxTexture3DHeight


        Maximum 3D texture height


    .. autoattribute:: cuda.cudart.cudaDeviceAttr.cudaDevAttrMaxTexture3DDepth


        Maximum 3D texture depth


    .. autoattribute:: cuda.cudart.cudaDeviceAttr.cudaDevAttrMaxTexture2DLayeredWidth


        Maximum 2D layered texture width


    .. autoattribute:: cuda.cudart.cudaDeviceAttr.cudaDevAttrMaxTexture2DLayeredHeight


        Maximum 2D layered texture height


    .. autoattribute:: cuda.cudart.cudaDeviceAttr.cudaDevAttrMaxTexture2DLayeredLayers


        Maximum layers in a 2D layered texture


    .. autoattribute:: cuda.cudart.cudaDeviceAttr.cudaDevAttrSurfaceAlignment


        Alignment requirement for surfaces


    .. autoattribute:: cuda.cudart.cudaDeviceAttr.cudaDevAttrConcurrentKernels


        Device can possibly execute multiple kernels concurrently


    .. autoattribute:: cuda.cudart.cudaDeviceAttr.cudaDevAttrEccEnabled


        Device has ECC support enabled


    .. autoattribute:: cuda.cudart.cudaDeviceAttr.cudaDevAttrPciBusId


        PCI bus ID of the device


    .. autoattribute:: cuda.cudart.cudaDeviceAttr.cudaDevAttrPciDeviceId


        PCI device ID of the device


    .. autoattribute:: cuda.cudart.cudaDeviceAttr.cudaDevAttrTccDriver


        Device is using TCC driver model


    .. autoattribute:: cuda.cudart.cudaDeviceAttr.cudaDevAttrMemoryClockRate


        Peak memory clock frequency in kilohertz


    .. autoattribute:: cuda.cudart.cudaDeviceAttr.cudaDevAttrGlobalMemoryBusWidth


        Global memory bus width in bits


    .. autoattribute:: cuda.cudart.cudaDeviceAttr.cudaDevAttrL2CacheSize


        Size of L2 cache in bytes


    .. autoattribute:: cuda.cudart.cudaDeviceAttr.cudaDevAttrMaxThreadsPerMultiProcessor


        Maximum resident threads per multiprocessor


    .. autoattribute:: cuda.cudart.cudaDeviceAttr.cudaDevAttrAsyncEngineCount


        Number of asynchronous engines


    .. autoattribute:: cuda.cudart.cudaDeviceAttr.cudaDevAttrUnifiedAddressing


        Device shares a unified address space with the host


    .. autoattribute:: cuda.cudart.cudaDeviceAttr.cudaDevAttrMaxTexture1DLayeredWidth


        Maximum 1D layered texture width


    .. autoattribute:: cuda.cudart.cudaDeviceAttr.cudaDevAttrMaxTexture1DLayeredLayers


        Maximum layers in a 1D layered texture


    .. autoattribute:: cuda.cudart.cudaDeviceAttr.cudaDevAttrMaxTexture2DGatherWidth


        Maximum 2D texture width if cudaArrayTextureGather is set


    .. autoattribute:: cuda.cudart.cudaDeviceAttr.cudaDevAttrMaxTexture2DGatherHeight


        Maximum 2D texture height if cudaArrayTextureGather is set


    .. autoattribute:: cuda.cudart.cudaDeviceAttr.cudaDevAttrMaxTexture3DWidthAlt


        Alternate maximum 3D texture width


    .. autoattribute:: cuda.cudart.cudaDeviceAttr.cudaDevAttrMaxTexture3DHeightAlt


        Alternate maximum 3D texture height


    .. autoattribute:: cuda.cudart.cudaDeviceAttr.cudaDevAttrMaxTexture3DDepthAlt


        Alternate maximum 3D texture depth


    .. autoattribute:: cuda.cudart.cudaDeviceAttr.cudaDevAttrPciDomainId


        PCI domain ID of the device


    .. autoattribute:: cuda.cudart.cudaDeviceAttr.cudaDevAttrTexturePitchAlignment


        Pitch alignment requirement for textures


    .. autoattribute:: cuda.cudart.cudaDeviceAttr.cudaDevAttrMaxTextureCubemapWidth


        Maximum cubemap texture width/height


    .. autoattribute:: cuda.cudart.cudaDeviceAttr.cudaDevAttrMaxTextureCubemapLayeredWidth


        Maximum cubemap layered texture width/height


    .. autoattribute:: cuda.cudart.cudaDeviceAttr.cudaDevAttrMaxTextureCubemapLayeredLayers


        Maximum layers in a cubemap layered texture


    .. autoattribute:: cuda.cudart.cudaDeviceAttr.cudaDevAttrMaxSurface1DWidth


        Maximum 1D surface width


    .. autoattribute:: cuda.cudart.cudaDeviceAttr.cudaDevAttrMaxSurface2DWidth


        Maximum 2D surface width


    .. autoattribute:: cuda.cudart.cudaDeviceAttr.cudaDevAttrMaxSurface2DHeight


        Maximum 2D surface height


    .. autoattribute:: cuda.cudart.cudaDeviceAttr.cudaDevAttrMaxSurface3DWidth


        Maximum 3D surface width


    .. autoattribute:: cuda.cudart.cudaDeviceAttr.cudaDevAttrMaxSurface3DHeight


        Maximum 3D surface height


    .. autoattribute:: cuda.cudart.cudaDeviceAttr.cudaDevAttrMaxSurface3DDepth


        Maximum 3D surface depth


    .. autoattribute:: cuda.cudart.cudaDeviceAttr.cudaDevAttrMaxSurface1DLayeredWidth


        Maximum 1D layered surface width


    .. autoattribute:: cuda.cudart.cudaDeviceAttr.cudaDevAttrMaxSurface1DLayeredLayers


        Maximum layers in a 1D layered surface


    .. autoattribute:: cuda.cudart.cudaDeviceAttr.cudaDevAttrMaxSurface2DLayeredWidth


        Maximum 2D layered surface width


    .. autoattribute:: cuda.cudart.cudaDeviceAttr.cudaDevAttrMaxSurface2DLayeredHeight


        Maximum 2D layered surface height


    .. autoattribute:: cuda.cudart.cudaDeviceAttr.cudaDevAttrMaxSurface2DLayeredLayers


        Maximum layers in a 2D layered surface


    .. autoattribute:: cuda.cudart.cudaDeviceAttr.cudaDevAttrMaxSurfaceCubemapWidth


        Maximum cubemap surface width


    .. autoattribute:: cuda.cudart.cudaDeviceAttr.cudaDevAttrMaxSurfaceCubemapLayeredWidth


        Maximum cubemap layered surface width


    .. autoattribute:: cuda.cudart.cudaDeviceAttr.cudaDevAttrMaxSurfaceCubemapLayeredLayers


        Maximum layers in a cubemap layered surface


    .. autoattribute:: cuda.cudart.cudaDeviceAttr.cudaDevAttrMaxTexture1DLinearWidth


        Maximum 1D linear texture width


    .. autoattribute:: cuda.cudart.cudaDeviceAttr.cudaDevAttrMaxTexture2DLinearWidth


        Maximum 2D linear texture width


    .. autoattribute:: cuda.cudart.cudaDeviceAttr.cudaDevAttrMaxTexture2DLinearHeight


        Maximum 2D linear texture height


    .. autoattribute:: cuda.cudart.cudaDeviceAttr.cudaDevAttrMaxTexture2DLinearPitch


        Maximum 2D linear texture pitch in bytes


    .. autoattribute:: cuda.cudart.cudaDeviceAttr.cudaDevAttrMaxTexture2DMipmappedWidth


        Maximum mipmapped 2D texture width


    .. autoattribute:: cuda.cudart.cudaDeviceAttr.cudaDevAttrMaxTexture2DMipmappedHeight


        Maximum mipmapped 2D texture height


    .. autoattribute:: cuda.cudart.cudaDeviceAttr.cudaDevAttrComputeCapabilityMajor


        Major compute capability version number


    .. autoattribute:: cuda.cudart.cudaDeviceAttr.cudaDevAttrComputeCapabilityMinor


        Minor compute capability version number


    .. autoattribute:: cuda.cudart.cudaDeviceAttr.cudaDevAttrMaxTexture1DMipmappedWidth


        Maximum mipmapped 1D texture width


    .. autoattribute:: cuda.cudart.cudaDeviceAttr.cudaDevAttrStreamPrioritiesSupported


        Device supports stream priorities


    .. autoattribute:: cuda.cudart.cudaDeviceAttr.cudaDevAttrGlobalL1CacheSupported


        Device supports caching globals in L1


    .. autoattribute:: cuda.cudart.cudaDeviceAttr.cudaDevAttrLocalL1CacheSupported


        Device supports caching locals in L1


    .. autoattribute:: cuda.cudart.cudaDeviceAttr.cudaDevAttrMaxSharedMemoryPerMultiprocessor


        Maximum shared memory available per multiprocessor in bytes


    .. autoattribute:: cuda.cudart.cudaDeviceAttr.cudaDevAttrMaxRegistersPerMultiprocessor


        Maximum number of 32-bit registers available per multiprocessor


    .. autoattribute:: cuda.cudart.cudaDeviceAttr.cudaDevAttrManagedMemory


        Device can allocate managed memory on this system


    .. autoattribute:: cuda.cudart.cudaDeviceAttr.cudaDevAttrIsMultiGpuBoard


        Device is on a multi-GPU board


    .. autoattribute:: cuda.cudart.cudaDeviceAttr.cudaDevAttrMultiGpuBoardGroupID


        Unique identifier for a group of devices on the same multi-GPU board


    .. autoattribute:: cuda.cudart.cudaDeviceAttr.cudaDevAttrHostNativeAtomicSupported


        Link between the device and the host supports native atomic operations


    .. autoattribute:: cuda.cudart.cudaDeviceAttr.cudaDevAttrSingleToDoublePrecisionPerfRatio


        Ratio of single precision performance (in floating-point operations per second) to double precision performance


    .. autoattribute:: cuda.cudart.cudaDeviceAttr.cudaDevAttrPageableMemoryAccess


        Device supports coherently accessing pageable memory without calling cudaHostRegister on it


    .. autoattribute:: cuda.cudart.cudaDeviceAttr.cudaDevAttrConcurrentManagedAccess


        Device can coherently access managed memory concurrently with the CPU


    .. autoattribute:: cuda.cudart.cudaDeviceAttr.cudaDevAttrComputePreemptionSupported


        Device supports Compute Preemption


    .. autoattribute:: cuda.cudart.cudaDeviceAttr.cudaDevAttrCanUseHostPointerForRegisteredMem


        Device can access host registered memory at the same virtual address as the CPU


    .. autoattribute:: cuda.cudart.cudaDeviceAttr.cudaDevAttrReserved92


    .. autoattribute:: cuda.cudart.cudaDeviceAttr.cudaDevAttrReserved93


    .. autoattribute:: cuda.cudart.cudaDeviceAttr.cudaDevAttrReserved94


    .. autoattribute:: cuda.cudart.cudaDeviceAttr.cudaDevAttrCooperativeLaunch


        Device supports launching cooperative kernels via :py:obj:`~.cudaLaunchCooperativeKernel`


    .. autoattribute:: cuda.cudart.cudaDeviceAttr.cudaDevAttrCooperativeMultiDeviceLaunch


        Deprecated, cudaLaunchCooperativeKernelMultiDevice is deprecated.


    .. autoattribute:: cuda.cudart.cudaDeviceAttr.cudaDevAttrMaxSharedMemoryPerBlockOptin


        The maximum optin shared memory per block. This value may vary by chip. See :py:obj:`~.cudaFuncSetAttribute`


    .. autoattribute:: cuda.cudart.cudaDeviceAttr.cudaDevAttrCanFlushRemoteWrites


        Device supports flushing of outstanding remote writes.


    .. autoattribute:: cuda.cudart.cudaDeviceAttr.cudaDevAttrHostRegisterSupported


        Device supports host memory registration via :py:obj:`~.cudaHostRegister`.


    .. autoattribute:: cuda.cudart.cudaDeviceAttr.cudaDevAttrPageableMemoryAccessUsesHostPageTables


        Device accesses pageable memory via the host's page tables.


    .. autoattribute:: cuda.cudart.cudaDeviceAttr.cudaDevAttrDirectManagedMemAccessFromHost


        Host can directly access managed memory on the device without migration.


    .. autoattribute:: cuda.cudart.cudaDeviceAttr.cudaDevAttrMaxBlocksPerMultiprocessor


        Maximum number of blocks per multiprocessor


    .. autoattribute:: cuda.cudart.cudaDeviceAttr.cudaDevAttrMaxPersistingL2CacheSize


        Maximum L2 persisting lines capacity setting in bytes.


    .. autoattribute:: cuda.cudart.cudaDeviceAttr.cudaDevAttrMaxAccessPolicyWindowSize


        Maximum value of :py:obj:`~.cudaAccessPolicyWindow.num_bytes`.


    .. autoattribute:: cuda.cudart.cudaDeviceAttr.cudaDevAttrReservedSharedMemoryPerBlock


        Shared memory reserved by CUDA driver per block in bytes


    .. autoattribute:: cuda.cudart.cudaDeviceAttr.cudaDevAttrSparseCudaArraySupported


        Device supports sparse CUDA arrays and sparse CUDA mipmapped arrays


    .. autoattribute:: cuda.cudart.cudaDeviceAttr.cudaDevAttrHostRegisterReadOnlySupported


        Device supports using the :py:obj:`~.cudaHostRegister` flag cudaHostRegisterReadOnly to register memory that must be mapped as read-only to the GPU


    .. autoattribute:: cuda.cudart.cudaDeviceAttr.cudaDevAttrTimelineSemaphoreInteropSupported


        External timeline semaphore interop is supported on the device


    .. autoattribute:: cuda.cudart.cudaDeviceAttr.cudaDevAttrMaxTimelineSemaphoreInteropSupported


        Deprecated, External timeline semaphore interop is supported on the device


    .. autoattribute:: cuda.cudart.cudaDeviceAttr.cudaDevAttrMemoryPoolsSupported


        Device supports using the :py:obj:`~.cudaMallocAsync` and :py:obj:`~.cudaMemPool` family of APIs


    .. autoattribute:: cuda.cudart.cudaDeviceAttr.cudaDevAttrGPUDirectRDMASupported


        Device supports GPUDirect RDMA APIs, like nvidia_p2p_get_pages (see https://docs.nvidia.com/cuda/gpudirect-rdma for more information)


    .. autoattribute:: cuda.cudart.cudaDeviceAttr.cudaDevAttrGPUDirectRDMAFlushWritesOptions


        The returned attribute shall be interpreted as a bitmask, where the individual bits are listed in the :py:obj:`~.cudaFlushGPUDirectRDMAWritesOptions` enum


    .. autoattribute:: cuda.cudart.cudaDeviceAttr.cudaDevAttrGPUDirectRDMAWritesOrdering


        GPUDirect RDMA writes to the device do not need to be flushed for consumers within the scope indicated by the returned attribute. See :py:obj:`~.cudaGPUDirectRDMAWritesOrdering` for the numerical values returned here.


    .. autoattribute:: cuda.cudart.cudaDeviceAttr.cudaDevAttrMemoryPoolSupportedHandleTypes


        Handle types supported with mempool based IPC


    .. autoattribute:: cuda.cudart.cudaDeviceAttr.cudaDevAttrDeferredMappingCudaArraySupported


        Device supports deferred mapping CUDA arrays and CUDA mipmapped arrays


    .. autoattribute:: cuda.cudart.cudaDeviceAttr.cudaDevAttrMax

.. autoclass:: cuda.cudart.cudaMemPoolAttr

    .. autoattribute:: cuda.cudart.cudaMemPoolAttr.cudaMemPoolReuseFollowEventDependencies


        (value type = int) Allow cuMemAllocAsync to use memory asynchronously freed in another streams as long as a stream ordering dependency of the allocating stream on the free action exists. Cuda events and null stream interactions can create the required stream ordered dependencies. (default enabled)


    .. autoattribute:: cuda.cudart.cudaMemPoolAttr.cudaMemPoolReuseAllowOpportunistic


        (value type = int) Allow reuse of already completed frees when there is no dependency between the free and allocation. (default enabled)


    .. autoattribute:: cuda.cudart.cudaMemPoolAttr.cudaMemPoolReuseAllowInternalDependencies


        (value type = int) Allow cuMemAllocAsync to insert new stream dependencies in order to establish the stream ordering required to reuse a piece of memory released by cuFreeAsync (default enabled).


    .. autoattribute:: cuda.cudart.cudaMemPoolAttr.cudaMemPoolAttrReleaseThreshold


        (value type = cuuint64_t) Amount of reserved memory in bytes to hold onto before trying to release memory back to the OS. When more than the release threshold bytes of memory are held by the memory pool, the allocator will try to release memory back to the OS on the next call to stream, event or context synchronize. (default 0)


    .. autoattribute:: cuda.cudart.cudaMemPoolAttr.cudaMemPoolAttrReservedMemCurrent


        (value type = cuuint64_t) Amount of backing memory currently allocated for the mempool.


    .. autoattribute:: cuda.cudart.cudaMemPoolAttr.cudaMemPoolAttrReservedMemHigh


        (value type = cuuint64_t) High watermark of backing memory allocated for the mempool since the last time it was reset. High watermark can only be reset to zero.


    .. autoattribute:: cuda.cudart.cudaMemPoolAttr.cudaMemPoolAttrUsedMemCurrent


        (value type = cuuint64_t) Amount of memory from the pool that is currently in use by the application.


    .. autoattribute:: cuda.cudart.cudaMemPoolAttr.cudaMemPoolAttrUsedMemHigh


        (value type = cuuint64_t) High watermark of the amount of memory from the pool that was in use by the application since the last time it was reset. High watermark can only be reset to zero.

.. autoclass:: cuda.cudart.cudaMemLocationType

    .. autoattribute:: cuda.cudart.cudaMemLocationType.cudaMemLocationTypeInvalid


    .. autoattribute:: cuda.cudart.cudaMemLocationType.cudaMemLocationTypeDevice


        Location is a device location, thus id is a device ordinal

.. autoclass:: cuda.cudart.cudaMemAccessFlags

    .. autoattribute:: cuda.cudart.cudaMemAccessFlags.cudaMemAccessFlagsProtNone


        Default, make the address range not accessible


    .. autoattribute:: cuda.cudart.cudaMemAccessFlags.cudaMemAccessFlagsProtRead


        Make the address range read accessible


    .. autoattribute:: cuda.cudart.cudaMemAccessFlags.cudaMemAccessFlagsProtReadWrite


        Make the address range read-write accessible

.. autoclass:: cuda.cudart.cudaMemAllocationType

    .. autoattribute:: cuda.cudart.cudaMemAllocationType.cudaMemAllocationTypeInvalid


    .. autoattribute:: cuda.cudart.cudaMemAllocationType.cudaMemAllocationTypePinned


        This allocation type is 'pinned', i.e. cannot migrate from its current location while the application is actively using it


    .. autoattribute:: cuda.cudart.cudaMemAllocationType.cudaMemAllocationTypeMax

.. autoclass:: cuda.cudart.cudaMemAllocationHandleType

    .. autoattribute:: cuda.cudart.cudaMemAllocationHandleType.cudaMemHandleTypeNone


        Does not allow any export mechanism. >


    .. autoattribute:: cuda.cudart.cudaMemAllocationHandleType.cudaMemHandleTypePosixFileDescriptor


        Allows a file descriptor to be used for exporting. Permitted only on POSIX systems. (int)


    .. autoattribute:: cuda.cudart.cudaMemAllocationHandleType.cudaMemHandleTypeWin32


        Allows a Win32 NT handle to be used for exporting. (HANDLE)


    .. autoattribute:: cuda.cudart.cudaMemAllocationHandleType.cudaMemHandleTypeWin32Kmt


        Allows a Win32 KMT handle to be used for exporting. (D3DKMT_HANDLE)

.. autoclass:: cuda.cudart.cudaGraphMemAttributeType

    .. autoattribute:: cuda.cudart.cudaGraphMemAttributeType.cudaGraphMemAttrUsedMemCurrent


        (value type = cuuint64_t) Amount of memory, in bytes, currently associated with graphs.


    .. autoattribute:: cuda.cudart.cudaGraphMemAttributeType.cudaGraphMemAttrUsedMemHigh


        (value type = cuuint64_t) High watermark of memory, in bytes, associated with graphs since the last time it was reset. High watermark can only be reset to zero.


    .. autoattribute:: cuda.cudart.cudaGraphMemAttributeType.cudaGraphMemAttrReservedMemCurrent


        (value type = cuuint64_t) Amount of memory, in bytes, currently allocated for use by the CUDA graphs asynchronous allocator.


    .. autoattribute:: cuda.cudart.cudaGraphMemAttributeType.cudaGraphMemAttrReservedMemHigh


        (value type = cuuint64_t) High watermark of memory, in bytes, currently allocated for use by the CUDA graphs asynchronous allocator.

.. autoclass:: cuda.cudart.cudaDeviceP2PAttr

    .. autoattribute:: cuda.cudart.cudaDeviceP2PAttr.cudaDevP2PAttrPerformanceRank


        A relative value indicating the performance of the link between two devices


    .. autoattribute:: cuda.cudart.cudaDeviceP2PAttr.cudaDevP2PAttrAccessSupported


        Peer access is enabled


    .. autoattribute:: cuda.cudart.cudaDeviceP2PAttr.cudaDevP2PAttrNativeAtomicSupported


        Native atomic operation over the link supported


    .. autoattribute:: cuda.cudart.cudaDeviceP2PAttr.cudaDevP2PAttrCudaArrayAccessSupported


        Accessing CUDA arrays over the link supported

.. autoclass:: cuda.cudart.cudaExternalMemoryHandleType

    .. autoattribute:: cuda.cudart.cudaExternalMemoryHandleType.cudaExternalMemoryHandleTypeOpaqueFd


        Handle is an opaque file descriptor


    .. autoattribute:: cuda.cudart.cudaExternalMemoryHandleType.cudaExternalMemoryHandleTypeOpaqueWin32


        Handle is an opaque shared NT handle


    .. autoattribute:: cuda.cudart.cudaExternalMemoryHandleType.cudaExternalMemoryHandleTypeOpaqueWin32Kmt


        Handle is an opaque, globally shared handle


    .. autoattribute:: cuda.cudart.cudaExternalMemoryHandleType.cudaExternalMemoryHandleTypeD3D12Heap


        Handle is a D3D12 heap object


    .. autoattribute:: cuda.cudart.cudaExternalMemoryHandleType.cudaExternalMemoryHandleTypeD3D12Resource


        Handle is a D3D12 committed resource


    .. autoattribute:: cuda.cudart.cudaExternalMemoryHandleType.cudaExternalMemoryHandleTypeD3D11Resource


        Handle is a shared NT handle to a D3D11 resource


    .. autoattribute:: cuda.cudart.cudaExternalMemoryHandleType.cudaExternalMemoryHandleTypeD3D11ResourceKmt


        Handle is a globally shared handle to a D3D11 resource


    .. autoattribute:: cuda.cudart.cudaExternalMemoryHandleType.cudaExternalMemoryHandleTypeNvSciBuf


        Handle is an NvSciBuf object

.. autoclass:: cuda.cudart.cudaExternalSemaphoreHandleType

    .. autoattribute:: cuda.cudart.cudaExternalSemaphoreHandleType.cudaExternalSemaphoreHandleTypeOpaqueFd


        Handle is an opaque file descriptor


    .. autoattribute:: cuda.cudart.cudaExternalSemaphoreHandleType.cudaExternalSemaphoreHandleTypeOpaqueWin32


        Handle is an opaque shared NT handle


    .. autoattribute:: cuda.cudart.cudaExternalSemaphoreHandleType.cudaExternalSemaphoreHandleTypeOpaqueWin32Kmt


        Handle is an opaque, globally shared handle


    .. autoattribute:: cuda.cudart.cudaExternalSemaphoreHandleType.cudaExternalSemaphoreHandleTypeD3D12Fence


        Handle is a shared NT handle referencing a D3D12 fence object


    .. autoattribute:: cuda.cudart.cudaExternalSemaphoreHandleType.cudaExternalSemaphoreHandleTypeD3D11Fence


        Handle is a shared NT handle referencing a D3D11 fence object


    .. autoattribute:: cuda.cudart.cudaExternalSemaphoreHandleType.cudaExternalSemaphoreHandleTypeNvSciSync


        Opaque handle to NvSciSync Object


    .. autoattribute:: cuda.cudart.cudaExternalSemaphoreHandleType.cudaExternalSemaphoreHandleTypeKeyedMutex


        Handle is a shared NT handle referencing a D3D11 keyed mutex object


    .. autoattribute:: cuda.cudart.cudaExternalSemaphoreHandleType.cudaExternalSemaphoreHandleTypeKeyedMutexKmt


        Handle is a shared KMT handle referencing a D3D11 keyed mutex object


    .. autoattribute:: cuda.cudart.cudaExternalSemaphoreHandleType.cudaExternalSemaphoreHandleTypeTimelineSemaphoreFd


        Handle is an opaque handle file descriptor referencing a timeline semaphore


    .. autoattribute:: cuda.cudart.cudaExternalSemaphoreHandleType.cudaExternalSemaphoreHandleTypeTimelineSemaphoreWin32


        Handle is an opaque handle file descriptor referencing a timeline semaphore

.. autoclass:: cuda.cudart.cudaCGScope

    .. autoattribute:: cuda.cudart.cudaCGScope.cudaCGScopeInvalid


        Invalid cooperative group scope


    .. autoattribute:: cuda.cudart.cudaCGScope.cudaCGScopeGrid


        Scope represented by a grid_group


    .. autoattribute:: cuda.cudart.cudaCGScope.cudaCGScopeMultiGrid


        Scope represented by a multi_grid_group

.. autoclass:: cuda.cudart.cudaGraphNodeType

    .. autoattribute:: cuda.cudart.cudaGraphNodeType.cudaGraphNodeTypeKernel


        GPU kernel node


    .. autoattribute:: cuda.cudart.cudaGraphNodeType.cudaGraphNodeTypeMemcpy


        Memcpy node


    .. autoattribute:: cuda.cudart.cudaGraphNodeType.cudaGraphNodeTypeMemset


        Memset node


    .. autoattribute:: cuda.cudart.cudaGraphNodeType.cudaGraphNodeTypeHost


        Host (executable) node


    .. autoattribute:: cuda.cudart.cudaGraphNodeType.cudaGraphNodeTypeGraph


        Node which executes an embedded graph


    .. autoattribute:: cuda.cudart.cudaGraphNodeType.cudaGraphNodeTypeEmpty


        Empty (no-op) node


    .. autoattribute:: cuda.cudart.cudaGraphNodeType.cudaGraphNodeTypeWaitEvent


        External event wait node


    .. autoattribute:: cuda.cudart.cudaGraphNodeType.cudaGraphNodeTypeEventRecord


        External event record node


    .. autoattribute:: cuda.cudart.cudaGraphNodeType.cudaGraphNodeTypeExtSemaphoreSignal


        External semaphore signal node


    .. autoattribute:: cuda.cudart.cudaGraphNodeType.cudaGraphNodeTypeExtSemaphoreWait


        External semaphore wait node


    .. autoattribute:: cuda.cudart.cudaGraphNodeType.cudaGraphNodeTypeMemAlloc


        Memory allocation node


    .. autoattribute:: cuda.cudart.cudaGraphNodeType.cudaGraphNodeTypeMemFree


        Memory free node


    .. autoattribute:: cuda.cudart.cudaGraphNodeType.cudaGraphNodeTypeCount

.. autoclass:: cuda.cudart.cudaGraphExecUpdateResult

    .. autoattribute:: cuda.cudart.cudaGraphExecUpdateResult.cudaGraphExecUpdateSuccess


        The update succeeded


    .. autoattribute:: cuda.cudart.cudaGraphExecUpdateResult.cudaGraphExecUpdateError


        The update failed for an unexpected reason which is described in the return value of the function


    .. autoattribute:: cuda.cudart.cudaGraphExecUpdateResult.cudaGraphExecUpdateErrorTopologyChanged


        The update failed because the topology changed


    .. autoattribute:: cuda.cudart.cudaGraphExecUpdateResult.cudaGraphExecUpdateErrorNodeTypeChanged


        The update failed because a node type changed


    .. autoattribute:: cuda.cudart.cudaGraphExecUpdateResult.cudaGraphExecUpdateErrorFunctionChanged


        The update failed because the function of a kernel node changed (CUDA driver < 11.2)


    .. autoattribute:: cuda.cudart.cudaGraphExecUpdateResult.cudaGraphExecUpdateErrorParametersChanged


        The update failed because the parameters changed in a way that is not supported


    .. autoattribute:: cuda.cudart.cudaGraphExecUpdateResult.cudaGraphExecUpdateErrorNotSupported


        The update failed because something about the node is not supported


    .. autoattribute:: cuda.cudart.cudaGraphExecUpdateResult.cudaGraphExecUpdateErrorUnsupportedFunctionChange


        The update failed because the function of a kernel node changed in an unsupported way


    .. autoattribute:: cuda.cudart.cudaGraphExecUpdateResult.cudaGraphExecUpdateErrorAttributesChanged


        The update failed because the node attributes changed in a way that is not supported

.. autoclass:: cuda.cudart.cudaGetDriverEntryPointFlags

    .. autoattribute:: cuda.cudart.cudaGetDriverEntryPointFlags.cudaEnableDefault


        Default search mode for driver symbols.


    .. autoattribute:: cuda.cudart.cudaGetDriverEntryPointFlags.cudaEnableLegacyStream


        Search for legacy versions of driver symbols.


    .. autoattribute:: cuda.cudart.cudaGetDriverEntryPointFlags.cudaEnablePerThreadDefaultStream


        Search for per-thread versions of driver symbols.

.. autoclass:: cuda.cudart.cudaGraphDebugDotFlags

    .. autoattribute:: cuda.cudart.cudaGraphDebugDotFlags.cudaGraphDebugDotFlagsVerbose


        Output all debug data as if every debug flag is enabled


    .. autoattribute:: cuda.cudart.cudaGraphDebugDotFlags.cudaGraphDebugDotFlagsKernelNodeParams


        Adds :py:obj:`~.cudaKernelNodeParams` to output


    .. autoattribute:: cuda.cudart.cudaGraphDebugDotFlags.cudaGraphDebugDotFlagsMemcpyNodeParams


        Adds :py:obj:`~.cudaMemcpy3DParms` to output


    .. autoattribute:: cuda.cudart.cudaGraphDebugDotFlags.cudaGraphDebugDotFlagsMemsetNodeParams


        Adds :py:obj:`~.cudaMemsetParams` to output


    .. autoattribute:: cuda.cudart.cudaGraphDebugDotFlags.cudaGraphDebugDotFlagsHostNodeParams


        Adds :py:obj:`~.cudaHostNodeParams` to output


    .. autoattribute:: cuda.cudart.cudaGraphDebugDotFlags.cudaGraphDebugDotFlagsEventNodeParams


        Adds cudaEvent_t handle from record and wait nodes to output


    .. autoattribute:: cuda.cudart.cudaGraphDebugDotFlags.cudaGraphDebugDotFlagsExtSemasSignalNodeParams


        Adds :py:obj:`~.cudaExternalSemaphoreSignalNodeParams` values to output


    .. autoattribute:: cuda.cudart.cudaGraphDebugDotFlags.cudaGraphDebugDotFlagsExtSemasWaitNodeParams


        Adds :py:obj:`~.cudaExternalSemaphoreWaitNodeParams` to output


    .. autoattribute:: cuda.cudart.cudaGraphDebugDotFlags.cudaGraphDebugDotFlagsKernelNodeAttributes


        Adds cudaKernelNodeAttrID values to output


    .. autoattribute:: cuda.cudart.cudaGraphDebugDotFlags.cudaGraphDebugDotFlagsHandles


        Adds node handles and every kernel function handle to output

.. autoclass:: cuda.cudart.cudaGraphInstantiateFlags

    .. autoattribute:: cuda.cudart.cudaGraphInstantiateFlags.cudaGraphInstantiateFlagAutoFreeOnLaunch


        Automatically free memory allocated in a graph before relaunching.


    .. autoattribute:: cuda.cudart.cudaGraphInstantiateFlags.cudaGraphInstantiateFlagUseNodePriority


        Run the graph using the per-node priority attributes rather than the priority of the stream it is launched into.

.. autoclass:: cuda.cudart.cudaStreamAttrID

    .. autoattribute:: cuda.cudart.cudaStreamAttrID.cudaStreamAttributeAccessPolicyWindow


        Identifier for :py:obj:`~.cudaStreamAttrValue`::accessPolicyWindow.


    .. autoattribute:: cuda.cudart.cudaStreamAttrID.cudaStreamAttributeSynchronizationPolicy


        :py:obj:`~.cudaSynchronizationPolicy` for work queued up in this stream

.. autoclass:: cuda.cudart.cudaKernelNodeAttrID

    .. autoattribute:: cuda.cudart.cudaKernelNodeAttrID.cudaKernelNodeAttributeAccessPolicyWindow


        Identifier for :py:obj:`~.cudaKernelNodeAttrValue.accessPolicyWindow`.


    .. autoattribute:: cuda.cudart.cudaKernelNodeAttrID.cudaKernelNodeAttributeCooperative


        Allows a kernel node to be cooperative (see :py:obj:`~.cudaLaunchCooperativeKernel`).


    .. autoattribute:: cuda.cudart.cudaKernelNodeAttrID.cudaKernelNodeAttributePriority


        Sets the priority of the kernel.

.. autoclass:: cuda.cudart.cudaSurfaceBoundaryMode

    .. autoattribute:: cuda.cudart.cudaSurfaceBoundaryMode.cudaBoundaryModeZero


        Zero boundary mode


    .. autoattribute:: cuda.cudart.cudaSurfaceBoundaryMode.cudaBoundaryModeClamp


        Clamp boundary mode


    .. autoattribute:: cuda.cudart.cudaSurfaceBoundaryMode.cudaBoundaryModeTrap


        Trap boundary mode

.. autoclass:: cuda.cudart.cudaSurfaceFormatMode

    .. autoattribute:: cuda.cudart.cudaSurfaceFormatMode.cudaFormatModeForced


        Forced format mode


    .. autoattribute:: cuda.cudart.cudaSurfaceFormatMode.cudaFormatModeAuto


        Auto format mode

.. autoclass:: cuda.cudart.cudaTextureAddressMode

    .. autoattribute:: cuda.cudart.cudaTextureAddressMode.cudaAddressModeWrap


        Wrapping address mode


    .. autoattribute:: cuda.cudart.cudaTextureAddressMode.cudaAddressModeClamp


        Clamp to edge address mode


    .. autoattribute:: cuda.cudart.cudaTextureAddressMode.cudaAddressModeMirror


        Mirror address mode


    .. autoattribute:: cuda.cudart.cudaTextureAddressMode.cudaAddressModeBorder


        Border address mode

.. autoclass:: cuda.cudart.cudaTextureFilterMode

    .. autoattribute:: cuda.cudart.cudaTextureFilterMode.cudaFilterModePoint


        Point filter mode


    .. autoattribute:: cuda.cudart.cudaTextureFilterMode.cudaFilterModeLinear


        Linear filter mode

.. autoclass:: cuda.cudart.cudaTextureReadMode

    .. autoattribute:: cuda.cudart.cudaTextureReadMode.cudaReadModeElementType


        Read texture as specified element type


    .. autoattribute:: cuda.cudart.cudaTextureReadMode.cudaReadModeNormalizedFloat


        Read texture as normalized float

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
.. autoclass:: cuda.cudart.cudaStreamAttrValue
.. autoclass:: cuda.cudart.cudaKernelNodeAttrValue
.. autoclass:: cuda.cudart.cudaSurfaceObject_t
.. autoclass:: cuda.cudart.cudaTextureObject_t
.. autoattribute:: cuda.cudart.CUDA_EGL_MAX_PLANES

    Maximum number of planes per frame

.. autoattribute:: cuda.cudart.cudaHostAllocDefault

    Default page-locked allocation flag

.. autoattribute:: cuda.cudart.cudaHostAllocPortable

    Pinned memory accessible by all CUDA contexts

.. autoattribute:: cuda.cudart.cudaHostAllocMapped

    Map allocation into device space

.. autoattribute:: cuda.cudart.cudaHostAllocWriteCombined

    Write-combined memory

.. autoattribute:: cuda.cudart.cudaHostRegisterDefault

    Default host memory registration flag

.. autoattribute:: cuda.cudart.cudaHostRegisterPortable

    Pinned memory accessible by all CUDA contexts

.. autoattribute:: cuda.cudart.cudaHostRegisterMapped

    Map registered memory into device space

.. autoattribute:: cuda.cudart.cudaHostRegisterIoMemory

    Memory-mapped I/O space

.. autoattribute:: cuda.cudart.cudaHostRegisterReadOnly

    Memory-mapped read-only

.. autoattribute:: cuda.cudart.cudaPeerAccessDefault

    Default peer addressing enable flag

.. autoattribute:: cuda.cudart.cudaStreamDefault

    Default stream flag

.. autoattribute:: cuda.cudart.cudaStreamNonBlocking

    Stream does not synchronize with stream 0 (the NULL stream)

.. autoattribute:: cuda.cudart.cudaStreamLegacy

    Legacy stream handle



    Stream handle that can be passed as a cudaStream_t to use an implicit stream with legacy synchronization behavior.



    See details of the \link_sync_behavior

.. autoattribute:: cuda.cudart.cudaStreamPerThread

    Per-thread stream handle



    Stream handle that can be passed as a cudaStream_t to use an implicit stream with per-thread synchronization behavior.



    See details of the \link_sync_behavior

.. autoattribute:: cuda.cudart.cudaEventDefault

    Default event flag

.. autoattribute:: cuda.cudart.cudaEventBlockingSync

    Event uses blocking synchronization

.. autoattribute:: cuda.cudart.cudaEventDisableTiming

    Event will not record timing data

.. autoattribute:: cuda.cudart.cudaEventInterprocess

    Event is suitable for interprocess use. cudaEventDisableTiming must be set

.. autoattribute:: cuda.cudart.cudaEventRecordDefault

    Default event record flag

.. autoattribute:: cuda.cudart.cudaEventRecordExternal

    Event is captured in the graph as an external event node when performing stream capture

.. autoattribute:: cuda.cudart.cudaEventWaitDefault

    Default event wait flag

.. autoattribute:: cuda.cudart.cudaEventWaitExternal

    Event is captured in the graph as an external event node when performing stream capture

.. autoattribute:: cuda.cudart.cudaDeviceScheduleAuto

    Device flag - Automatic scheduling

.. autoattribute:: cuda.cudart.cudaDeviceScheduleSpin

    Device flag - Spin default scheduling

.. autoattribute:: cuda.cudart.cudaDeviceScheduleYield

    Device flag - Yield default scheduling

.. autoattribute:: cuda.cudart.cudaDeviceScheduleBlockingSync

    Device flag - Use blocking synchronization

.. autoattribute:: cuda.cudart.cudaDeviceBlockingSync

    Device flag - Use blocking synchronization [Deprecated]

.. autoattribute:: cuda.cudart.cudaDeviceScheduleMask

    Device schedule flags mask

.. autoattribute:: cuda.cudart.cudaDeviceMapHost

    Device flag - Support mapped pinned allocations

.. autoattribute:: cuda.cudart.cudaDeviceLmemResizeToMax

    Device flag - Keep local memory allocation after launch

.. autoattribute:: cuda.cudart.cudaDeviceMask

    Device flags mask

.. autoattribute:: cuda.cudart.cudaArrayDefault

    Default CUDA array allocation flag

.. autoattribute:: cuda.cudart.cudaArrayLayered

    Must be set in cudaMalloc3DArray to create a layered CUDA array

.. autoattribute:: cuda.cudart.cudaArraySurfaceLoadStore

    Must be set in cudaMallocArray or cudaMalloc3DArray in order to bind surfaces to the CUDA array

.. autoattribute:: cuda.cudart.cudaArrayCubemap

    Must be set in cudaMalloc3DArray to create a cubemap CUDA array

.. autoattribute:: cuda.cudart.cudaArrayTextureGather

    Must be set in cudaMallocArray or cudaMalloc3DArray in order to perform texture gather operations on the CUDA array

.. autoattribute:: cuda.cudart.cudaArrayColorAttachment

    Must be set in cudaExternalMemoryGetMappedMipmappedArray if the mipmapped array is used as a color target in a graphics API

.. autoattribute:: cuda.cudart.cudaArraySparse

    Must be set in cudaMallocArray, cudaMalloc3DArray or cudaMallocMipmappedArray in order to create a sparse CUDA array or CUDA mipmapped array

.. autoattribute:: cuda.cudart.cudaArrayDeferredMapping

    Must be set in cudaMallocArray, cudaMalloc3DArray or cudaMallocMipmappedArray in order to create a deferred mapping CUDA array or CUDA mipmapped array

.. autoattribute:: cuda.cudart.cudaIpcMemLazyEnablePeerAccess

    Automatically enable peer access between remote devices as needed

.. autoattribute:: cuda.cudart.cudaMemAttachGlobal

    Memory can be accessed by any stream on any device

.. autoattribute:: cuda.cudart.cudaMemAttachHost

    Memory cannot be accessed by any stream on any device

.. autoattribute:: cuda.cudart.cudaMemAttachSingle

    Memory can only be accessed by a single stream on the associated device

.. autoattribute:: cuda.cudart.cudaOccupancyDefault

    Default behavior

.. autoattribute:: cuda.cudart.cudaOccupancyDisableCachingOverride

    Assume global caching is enabled and cannot be automatically turned off

.. autoattribute:: cuda.cudart.cudaCpuDeviceId

    Device id that represents the CPU

.. autoattribute:: cuda.cudart.cudaInvalidDeviceId

    Device id that represents an invalid device

.. autoattribute:: cuda.cudart.cudaCooperativeLaunchMultiDeviceNoPreSync

    If set, each kernel launched as part of :py:obj:`~.cudaLaunchCooperativeKernelMultiDevice` only waits for prior work in the stream corresponding to that GPU to complete before the kernel begins execution.

.. autoattribute:: cuda.cudart.cudaCooperativeLaunchMultiDeviceNoPostSync

    If set, any subsequent work pushed in a stream that participated in a call to :py:obj:`~.cudaLaunchCooperativeKernelMultiDevice` will only wait for the kernel launched on the GPU corresponding to that stream to complete before it begins execution.

.. autoattribute:: cuda.cudart.cudaArraySparsePropertiesSingleMipTail

    Indicates that the layered sparse CUDA array or CUDA mipmapped array has a single mip tail region for all layers

.. autoattribute:: cuda.cudart.CUDART_CB
.. autoattribute:: cuda.cudart.CU_UUID_HAS_BEEN_DEFINED

    CUDA UUID types

.. autoattribute:: cuda.cudart.cudaDevicePropDontCare

    Empty device properties

.. autoattribute:: cuda.cudart.CUDA_IPC_HANDLE_SIZE

    CUDA IPC Handle Size

.. autoattribute:: cuda.cudart.cudaExternalMemoryDedicated

    Indicates that the external memory object is a dedicated resource

.. autoattribute:: cuda.cudart.cudaExternalSemaphoreSignalSkipNvSciBufMemSync

    When the /p flags parameter of :py:obj:`~.cudaExternalSemaphoreSignalParams` contains this flag, it indicates that signaling an external semaphore object should skip performing appropriate memory synchronization operations over all the external memory objects that are imported as :py:obj:`~.cudaExternalMemoryHandleTypeNvSciBuf`, which otherwise are performed by default to ensure data coherency with other importers of the same NvSciBuf memory objects.

.. autoattribute:: cuda.cudart.cudaExternalSemaphoreWaitSkipNvSciBufMemSync

    When the /p flags parameter of :py:obj:`~.cudaExternalSemaphoreWaitParams` contains this flag, it indicates that waiting an external semaphore object should skip performing appropriate memory synchronization operations over all the external memory objects that are imported as :py:obj:`~.cudaExternalMemoryHandleTypeNvSciBuf`, which otherwise are performed by default to ensure data coherency with other importers of the same NvSciBuf memory objects.

.. autoattribute:: cuda.cudart.cudaNvSciSyncAttrSignal

    When /p flags of :py:obj:`~.cudaDeviceGetNvSciSyncAttributes` is set to this, it indicates that application need signaler specific NvSciSyncAttr to be filled by :py:obj:`~.cudaDeviceGetNvSciSyncAttributes`.

.. autoattribute:: cuda.cudart.cudaNvSciSyncAttrWait

    When /p flags of :py:obj:`~.cudaDeviceGetNvSciSyncAttributes` is set to this, it indicates that application need waiter specific NvSciSyncAttr to be filled by :py:obj:`~.cudaDeviceGetNvSciSyncAttributes`.

.. autoattribute:: cuda.cudart.cudaSurfaceType1D
.. autoattribute:: cuda.cudart.cudaSurfaceType2D
.. autoattribute:: cuda.cudart.cudaSurfaceType3D
.. autoattribute:: cuda.cudart.cudaSurfaceTypeCubemap
.. autoattribute:: cuda.cudart.cudaSurfaceType1DLayered
.. autoattribute:: cuda.cudart.cudaSurfaceType2DLayered
.. autoattribute:: cuda.cudart.cudaSurfaceTypeCubemapLayered
.. autoattribute:: cuda.cudart.cudaTextureType1D
.. autoattribute:: cuda.cudart.cudaTextureType2D
.. autoattribute:: cuda.cudart.cudaTextureType3D
.. autoattribute:: cuda.cudart.cudaTextureTypeCubemap
.. autoattribute:: cuda.cudart.cudaTextureType1DLayered
.. autoattribute:: cuda.cudart.cudaTextureType2DLayered
.. autoattribute:: cuda.cudart.cudaTextureTypeCubemapLayered
