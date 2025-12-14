.. SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

-------
runtime
-------

Profiler Control
----------------

This section describes the profiler control functions of the CUDA runtime application programming interface.

.. autofunction:: cuda.bindings.runtime.cudaProfilerStart
.. autofunction:: cuda.bindings.runtime.cudaProfilerStop

Device Management
-----------------

impl_private







This section describes the device management functions of the CUDA runtime application programming interface.

.. autofunction:: cuda.bindings.runtime.cudaDeviceReset
.. autofunction:: cuda.bindings.runtime.cudaDeviceSynchronize
.. autofunction:: cuda.bindings.runtime.cudaDeviceSetLimit
.. autofunction:: cuda.bindings.runtime.cudaDeviceGetLimit
.. autofunction:: cuda.bindings.runtime.cudaDeviceGetTexture1DLinearMaxWidth
.. autofunction:: cuda.bindings.runtime.cudaDeviceGetCacheConfig
.. autofunction:: cuda.bindings.runtime.cudaDeviceGetStreamPriorityRange
.. autofunction:: cuda.bindings.runtime.cudaDeviceSetCacheConfig
.. autofunction:: cuda.bindings.runtime.cudaDeviceGetByPCIBusId
.. autofunction:: cuda.bindings.runtime.cudaDeviceGetPCIBusId
.. autofunction:: cuda.bindings.runtime.cudaIpcGetEventHandle
.. autofunction:: cuda.bindings.runtime.cudaIpcOpenEventHandle
.. autofunction:: cuda.bindings.runtime.cudaIpcGetMemHandle
.. autofunction:: cuda.bindings.runtime.cudaIpcOpenMemHandle
.. autofunction:: cuda.bindings.runtime.cudaIpcCloseMemHandle
.. autofunction:: cuda.bindings.runtime.cudaDeviceFlushGPUDirectRDMAWrites
.. autofunction:: cuda.bindings.runtime.cudaDeviceRegisterAsyncNotification
.. autofunction:: cuda.bindings.runtime.cudaDeviceUnregisterAsyncNotification
.. autofunction:: cuda.bindings.runtime.cudaGetDeviceCount
.. autofunction:: cuda.bindings.runtime.cudaGetDeviceProperties
.. autofunction:: cuda.bindings.runtime.cudaDeviceGetAttribute
.. autofunction:: cuda.bindings.runtime.cudaDeviceGetHostAtomicCapabilities
.. autofunction:: cuda.bindings.runtime.cudaDeviceGetDefaultMemPool
.. autofunction:: cuda.bindings.runtime.cudaDeviceSetMemPool
.. autofunction:: cuda.bindings.runtime.cudaDeviceGetMemPool
.. autofunction:: cuda.bindings.runtime.cudaDeviceGetNvSciSyncAttributes
.. autofunction:: cuda.bindings.runtime.cudaDeviceGetP2PAttribute
.. autofunction:: cuda.bindings.runtime.cudaDeviceGetP2PAtomicCapabilities
.. autofunction:: cuda.bindings.runtime.cudaChooseDevice
.. autofunction:: cuda.bindings.runtime.cudaInitDevice
.. autofunction:: cuda.bindings.runtime.cudaSetDevice
.. autofunction:: cuda.bindings.runtime.cudaGetDevice
.. autofunction:: cuda.bindings.runtime.cudaSetDeviceFlags
.. autofunction:: cuda.bindings.runtime.cudaGetDeviceFlags

Error Handling
--------------

This section describes the error handling functions of the CUDA runtime application programming interface.

.. autofunction:: cuda.bindings.runtime.cudaGetLastError
.. autofunction:: cuda.bindings.runtime.cudaPeekAtLastError
.. autofunction:: cuda.bindings.runtime.cudaGetErrorName
.. autofunction:: cuda.bindings.runtime.cudaGetErrorString

Stream Management
-----------------

This section describes the stream management functions of the CUDA runtime application programming interface.

.. autoclass:: cuda.bindings.runtime.cudaStreamCallback_t
.. autofunction:: cuda.bindings.runtime.cudaStreamCreate
.. autofunction:: cuda.bindings.runtime.cudaStreamCreateWithFlags
.. autofunction:: cuda.bindings.runtime.cudaStreamCreateWithPriority
.. autofunction:: cuda.bindings.runtime.cudaStreamGetPriority
.. autofunction:: cuda.bindings.runtime.cudaStreamGetFlags
.. autofunction:: cuda.bindings.runtime.cudaStreamGetId
.. autofunction:: cuda.bindings.runtime.cudaStreamGetDevice
.. autofunction:: cuda.bindings.runtime.cudaCtxResetPersistingL2Cache
.. autofunction:: cuda.bindings.runtime.cudaStreamCopyAttributes
.. autofunction:: cuda.bindings.runtime.cudaStreamGetAttribute
.. autofunction:: cuda.bindings.runtime.cudaStreamSetAttribute
.. autofunction:: cuda.bindings.runtime.cudaStreamDestroy
.. autofunction:: cuda.bindings.runtime.cudaStreamWaitEvent
.. autofunction:: cuda.bindings.runtime.cudaStreamAddCallback
.. autofunction:: cuda.bindings.runtime.cudaStreamSynchronize
.. autofunction:: cuda.bindings.runtime.cudaStreamQuery
.. autofunction:: cuda.bindings.runtime.cudaStreamAttachMemAsync
.. autofunction:: cuda.bindings.runtime.cudaStreamBeginCapture
.. autofunction:: cuda.bindings.runtime.cudaStreamBeginCaptureToGraph
.. autofunction:: cuda.bindings.runtime.cudaThreadExchangeStreamCaptureMode
.. autofunction:: cuda.bindings.runtime.cudaStreamEndCapture
.. autofunction:: cuda.bindings.runtime.cudaStreamIsCapturing
.. autofunction:: cuda.bindings.runtime.cudaStreamGetCaptureInfo
.. autofunction:: cuda.bindings.runtime.cudaStreamUpdateCaptureDependencies

Event Management
----------------

This section describes the event management functions of the CUDA runtime application programming interface.

.. autofunction:: cuda.bindings.runtime.cudaEventCreate
.. autofunction:: cuda.bindings.runtime.cudaEventCreateWithFlags
.. autofunction:: cuda.bindings.runtime.cudaEventRecord
.. autofunction:: cuda.bindings.runtime.cudaEventRecordWithFlags
.. autofunction:: cuda.bindings.runtime.cudaEventQuery
.. autofunction:: cuda.bindings.runtime.cudaEventSynchronize
.. autofunction:: cuda.bindings.runtime.cudaEventDestroy
.. autofunction:: cuda.bindings.runtime.cudaEventElapsedTime

External Resource Interoperability
----------------------------------

This section describes the external resource interoperability functions of the CUDA runtime application programming interface.

.. autofunction:: cuda.bindings.runtime.cudaImportExternalMemory
.. autofunction:: cuda.bindings.runtime.cudaExternalMemoryGetMappedBuffer
.. autofunction:: cuda.bindings.runtime.cudaExternalMemoryGetMappedMipmappedArray
.. autofunction:: cuda.bindings.runtime.cudaDestroyExternalMemory
.. autofunction:: cuda.bindings.runtime.cudaImportExternalSemaphore
.. autofunction:: cuda.bindings.runtime.cudaSignalExternalSemaphoresAsync
.. autofunction:: cuda.bindings.runtime.cudaWaitExternalSemaphoresAsync
.. autofunction:: cuda.bindings.runtime.cudaDestroyExternalSemaphore

Execution Control
-----------------

This section describes the execution control functions of the CUDA runtime application programming interface.



Some functions have overloaded C++ API template versions documented separately in the C++ API Routines module.

.. autofunction:: cuda.bindings.runtime.cudaFuncSetCacheConfig
.. autofunction:: cuda.bindings.runtime.cudaFuncGetAttributes
.. autofunction:: cuda.bindings.runtime.cudaFuncSetAttribute
.. autofunction:: cuda.bindings.runtime.cudaLaunchHostFunc

Occupancy
---------

This section describes the occupancy calculation functions of the CUDA runtime application programming interface.



Besides the occupancy calculator functions (cudaOccupancyMaxActiveBlocksPerMultiprocessor and cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags), there are also C++ only occupancy-based launch configuration functions documented in C++ API Routines module.



See cudaOccupancyMaxPotentialBlockSize (C++ API), cudaOccupancyMaxPotentialBlockSize (C++ API), cudaOccupancyMaxPotentialBlockSizeVariableSMem (C++ API), cudaOccupancyMaxPotentialBlockSizeVariableSMem (C++ API) cudaOccupancyAvailableDynamicSMemPerBlock (C++ API),

.. autofunction:: cuda.bindings.runtime.cudaOccupancyMaxActiveBlocksPerMultiprocessor
.. autofunction:: cuda.bindings.runtime.cudaOccupancyAvailableDynamicSMemPerBlock
.. autofunction:: cuda.bindings.runtime.cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags

Memory Management
-----------------

This section describes the memory management functions of the CUDA runtime application programming interface.



Some functions have overloaded C++ API template versions documented separately in the C++ API Routines module.

.. autofunction:: cuda.bindings.runtime.cudaMallocManaged
.. autofunction:: cuda.bindings.runtime.cudaMalloc
.. autofunction:: cuda.bindings.runtime.cudaMallocHost
.. autofunction:: cuda.bindings.runtime.cudaMallocPitch
.. autofunction:: cuda.bindings.runtime.cudaMallocArray
.. autofunction:: cuda.bindings.runtime.cudaFree
.. autofunction:: cuda.bindings.runtime.cudaFreeHost
.. autofunction:: cuda.bindings.runtime.cudaFreeArray
.. autofunction:: cuda.bindings.runtime.cudaFreeMipmappedArray
.. autofunction:: cuda.bindings.runtime.cudaHostAlloc
.. autofunction:: cuda.bindings.runtime.cudaHostRegister
.. autofunction:: cuda.bindings.runtime.cudaHostUnregister
.. autofunction:: cuda.bindings.runtime.cudaHostGetDevicePointer
.. autofunction:: cuda.bindings.runtime.cudaHostGetFlags
.. autofunction:: cuda.bindings.runtime.cudaMalloc3D
.. autofunction:: cuda.bindings.runtime.cudaMalloc3DArray
.. autofunction:: cuda.bindings.runtime.cudaMallocMipmappedArray
.. autofunction:: cuda.bindings.runtime.cudaGetMipmappedArrayLevel
.. autofunction:: cuda.bindings.runtime.cudaMemcpy3D
.. autofunction:: cuda.bindings.runtime.cudaMemcpy3DPeer
.. autofunction:: cuda.bindings.runtime.cudaMemcpy3DAsync
.. autofunction:: cuda.bindings.runtime.cudaMemcpy3DPeerAsync
.. autofunction:: cuda.bindings.runtime.cudaMemGetInfo
.. autofunction:: cuda.bindings.runtime.cudaArrayGetInfo
.. autofunction:: cuda.bindings.runtime.cudaArrayGetPlane
.. autofunction:: cuda.bindings.runtime.cudaArrayGetMemoryRequirements
.. autofunction:: cuda.bindings.runtime.cudaMipmappedArrayGetMemoryRequirements
.. autofunction:: cuda.bindings.runtime.cudaArrayGetSparseProperties
.. autofunction:: cuda.bindings.runtime.cudaMipmappedArrayGetSparseProperties
.. autofunction:: cuda.bindings.runtime.cudaMemcpy
.. autofunction:: cuda.bindings.runtime.cudaMemcpyPeer
.. autofunction:: cuda.bindings.runtime.cudaMemcpy2D
.. autofunction:: cuda.bindings.runtime.cudaMemcpy2DToArray
.. autofunction:: cuda.bindings.runtime.cudaMemcpy2DFromArray
.. autofunction:: cuda.bindings.runtime.cudaMemcpy2DArrayToArray
.. autofunction:: cuda.bindings.runtime.cudaMemcpyAsync
.. autofunction:: cuda.bindings.runtime.cudaMemcpyPeerAsync
.. autofunction:: cuda.bindings.runtime.cudaMemcpyBatchAsync
.. autofunction:: cuda.bindings.runtime.cudaMemcpy3DBatchAsync
.. autofunction:: cuda.bindings.runtime.cudaMemcpy2DAsync
.. autofunction:: cuda.bindings.runtime.cudaMemcpy2DToArrayAsync
.. autofunction:: cuda.bindings.runtime.cudaMemcpy2DFromArrayAsync
.. autofunction:: cuda.bindings.runtime.cudaMemset
.. autofunction:: cuda.bindings.runtime.cudaMemset2D
.. autofunction:: cuda.bindings.runtime.cudaMemset3D
.. autofunction:: cuda.bindings.runtime.cudaMemsetAsync
.. autofunction:: cuda.bindings.runtime.cudaMemset2DAsync
.. autofunction:: cuda.bindings.runtime.cudaMemset3DAsync
.. autofunction:: cuda.bindings.runtime.cudaMemPrefetchAsync
.. autofunction:: cuda.bindings.runtime.cudaMemPrefetchBatchAsync
.. autofunction:: cuda.bindings.runtime.cudaMemDiscardBatchAsync
.. autofunction:: cuda.bindings.runtime.cudaMemDiscardAndPrefetchBatchAsync
.. autofunction:: cuda.bindings.runtime.cudaMemAdvise
.. autofunction:: cuda.bindings.runtime.cudaMemRangeGetAttribute
.. autofunction:: cuda.bindings.runtime.cudaMemRangeGetAttributes
.. autofunction:: cuda.bindings.runtime.make_cudaPitchedPtr
.. autofunction:: cuda.bindings.runtime.make_cudaPos
.. autofunction:: cuda.bindings.runtime.make_cudaExtent

Stream Ordered Memory Allocator
-------------------------------

**overview**



The asynchronous allocator allows the user to allocate and free in stream order. All asynchronous accesses of the allocation must happen between the stream executions of the allocation and the free. If the memory is accessed outside of the promised stream order, a use before allocation / use after free error will cause undefined behavior.

The allocator is free to reallocate the memory as long as it can guarantee that compliant memory accesses will not overlap temporally. The allocator may refer to internal stream ordering as well as inter-stream dependencies (such as CUDA events and null stream dependencies) when establishing the temporal guarantee. The allocator may also insert inter-stream dependencies to establish the temporal guarantee.





**Supported Platforms**



Whether or not a device supports the integrated stream ordered memory allocator may be queried by calling cudaDeviceGetAttribute() with the device attribute cudaDevAttrMemoryPoolsSupported.

.. autofunction:: cuda.bindings.runtime.cudaMallocAsync
.. autofunction:: cuda.bindings.runtime.cudaFreeAsync
.. autofunction:: cuda.bindings.runtime.cudaMemPoolTrimTo
.. autofunction:: cuda.bindings.runtime.cudaMemPoolSetAttribute
.. autofunction:: cuda.bindings.runtime.cudaMemPoolGetAttribute
.. autofunction:: cuda.bindings.runtime.cudaMemPoolSetAccess
.. autofunction:: cuda.bindings.runtime.cudaMemPoolGetAccess
.. autofunction:: cuda.bindings.runtime.cudaMemPoolCreate
.. autofunction:: cuda.bindings.runtime.cudaMemPoolDestroy
.. autofunction:: cuda.bindings.runtime.cudaMemGetDefaultMemPool
.. autofunction:: cuda.bindings.runtime.cudaMemGetMemPool
.. autofunction:: cuda.bindings.runtime.cudaMemSetMemPool
.. autofunction:: cuda.bindings.runtime.cudaMallocFromPoolAsync
.. autofunction:: cuda.bindings.runtime.cudaMemPoolExportToShareableHandle
.. autofunction:: cuda.bindings.runtime.cudaMemPoolImportFromShareableHandle
.. autofunction:: cuda.bindings.runtime.cudaMemPoolExportPointer
.. autofunction:: cuda.bindings.runtime.cudaMemPoolImportPointer

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

.. autofunction:: cuda.bindings.runtime.cudaPointerGetAttributes

Peer Device Memory Access
-------------------------

This section describes the peer device memory access functions of the CUDA runtime application programming interface.

.. autofunction:: cuda.bindings.runtime.cudaDeviceCanAccessPeer
.. autofunction:: cuda.bindings.runtime.cudaDeviceEnablePeerAccess
.. autofunction:: cuda.bindings.runtime.cudaDeviceDisablePeerAccess

OpenGL Interoperability
-----------------------

impl_private



This section describes the OpenGL interoperability functions of the CUDA runtime application programming interface. Note that mapping of OpenGL resources is performed with the graphics API agnostic, resource mapping interface described in Graphics Interopability.

.. autoclass:: cuda.bindings.runtime.cudaGLDeviceList

    .. autoattribute:: cuda.bindings.runtime.cudaGLDeviceList.cudaGLDeviceListAll


        The CUDA devices for all GPUs used by the current OpenGL context


    .. autoattribute:: cuda.bindings.runtime.cudaGLDeviceList.cudaGLDeviceListCurrentFrame


        The CUDA devices for the GPUs used by the current OpenGL context in its currently rendering frame


    .. autoattribute:: cuda.bindings.runtime.cudaGLDeviceList.cudaGLDeviceListNextFrame


        The CUDA devices for the GPUs to be used by the current OpenGL context in the next frame

.. autofunction:: cuda.bindings.runtime.cudaGLGetDevices
.. autofunction:: cuda.bindings.runtime.cudaGraphicsGLRegisterImage
.. autofunction:: cuda.bindings.runtime.cudaGraphicsGLRegisterBuffer

Direct3D 9 Interoperability
---------------------------




Direct3D 10 Interoperability
----------------------------




Direct3D 11 Interoperability
----------------------------




VDPAU Interoperability
----------------------

This section describes the VDPAU interoperability functions of the CUDA runtime application programming interface.

.. autofunction:: cuda.bindings.runtime.cudaVDPAUGetDevice
.. autofunction:: cuda.bindings.runtime.cudaVDPAUSetVDPAUDevice
.. autofunction:: cuda.bindings.runtime.cudaGraphicsVDPAURegisterVideoSurface
.. autofunction:: cuda.bindings.runtime.cudaGraphicsVDPAURegisterOutputSurface

EGL Interoperability
--------------------

This section describes the EGL interoperability functions of the CUDA runtime application programming interface.

.. autofunction:: cuda.bindings.runtime.cudaGraphicsEGLRegisterImage
.. autofunction:: cuda.bindings.runtime.cudaEGLStreamConsumerConnect
.. autofunction:: cuda.bindings.runtime.cudaEGLStreamConsumerConnectWithFlags
.. autofunction:: cuda.bindings.runtime.cudaEGLStreamConsumerDisconnect
.. autofunction:: cuda.bindings.runtime.cudaEGLStreamConsumerAcquireFrame
.. autofunction:: cuda.bindings.runtime.cudaEGLStreamConsumerReleaseFrame
.. autofunction:: cuda.bindings.runtime.cudaEGLStreamProducerConnect
.. autofunction:: cuda.bindings.runtime.cudaEGLStreamProducerDisconnect
.. autofunction:: cuda.bindings.runtime.cudaEGLStreamProducerPresentFrame
.. autofunction:: cuda.bindings.runtime.cudaEGLStreamProducerReturnFrame
.. autofunction:: cuda.bindings.runtime.cudaGraphicsResourceGetMappedEglFrame
.. autofunction:: cuda.bindings.runtime.cudaEventCreateFromEGLSync

Graphics Interoperability
-------------------------

This section describes the graphics interoperability functions of the CUDA runtime application programming interface.

.. autofunction:: cuda.bindings.runtime.cudaGraphicsUnregisterResource
.. autofunction:: cuda.bindings.runtime.cudaGraphicsResourceSetMapFlags
.. autofunction:: cuda.bindings.runtime.cudaGraphicsMapResources
.. autofunction:: cuda.bindings.runtime.cudaGraphicsUnmapResources
.. autofunction:: cuda.bindings.runtime.cudaGraphicsResourceGetMappedPointer
.. autofunction:: cuda.bindings.runtime.cudaGraphicsSubResourceGetMappedArray
.. autofunction:: cuda.bindings.runtime.cudaGraphicsResourceGetMappedMipmappedArray

Texture Object Management
-------------------------

This section describes the low level texture object management functions of the CUDA runtime application programming interface. The texture object API is only supported on devices of compute capability 3.0 or higher.

.. autofunction:: cuda.bindings.runtime.cudaGetChannelDesc
.. autofunction:: cuda.bindings.runtime.cudaCreateChannelDesc
.. autofunction:: cuda.bindings.runtime.cudaCreateTextureObject
.. autofunction:: cuda.bindings.runtime.cudaDestroyTextureObject
.. autofunction:: cuda.bindings.runtime.cudaGetTextureObjectResourceDesc
.. autofunction:: cuda.bindings.runtime.cudaGetTextureObjectTextureDesc
.. autofunction:: cuda.bindings.runtime.cudaGetTextureObjectResourceViewDesc

Surface Object Management
-------------------------

This section describes the low level texture object management functions of the CUDA runtime application programming interface. The surface object API is only supported on devices of compute capability 3.0 or higher.

.. autofunction:: cuda.bindings.runtime.cudaCreateSurfaceObject
.. autofunction:: cuda.bindings.runtime.cudaDestroySurfaceObject
.. autofunction:: cuda.bindings.runtime.cudaGetSurfaceObjectResourceDesc

Version Management
------------------



.. autofunction:: cuda.bindings.runtime.cudaDriverGetVersion
.. autofunction:: cuda.bindings.runtime.cudaRuntimeGetVersion
.. autofunction:: cuda.bindings.runtime.getLocalRuntimeVersion

Error Log Management Functions
------------------------------

This section describes the error log management functions of the CUDA runtime application programming interface. The Error Log Management interface will operate on both the CUDA Driver and CUDA Runtime.

.. autoclass:: cuda.bindings.runtime.cudaLogsCallback_t
.. autofunction:: cuda.bindings.runtime.cudaLogsRegisterCallback
.. autofunction:: cuda.bindings.runtime.cudaLogsUnregisterCallback
.. autofunction:: cuda.bindings.runtime.cudaLogsCurrent
.. autofunction:: cuda.bindings.runtime.cudaLogsDumpToFile
.. autofunction:: cuda.bindings.runtime.cudaLogsDumpToMemory

Graph Management
----------------

This section describes the graph management functions of CUDA runtime application programming interface.

.. autofunction:: cuda.bindings.runtime.cudaGraphCreate
.. autofunction:: cuda.bindings.runtime.cudaGraphAddKernelNode
.. autofunction:: cuda.bindings.runtime.cudaGraphKernelNodeGetParams
.. autofunction:: cuda.bindings.runtime.cudaGraphKernelNodeSetParams
.. autofunction:: cuda.bindings.runtime.cudaGraphKernelNodeCopyAttributes
.. autofunction:: cuda.bindings.runtime.cudaGraphKernelNodeGetAttribute
.. autofunction:: cuda.bindings.runtime.cudaGraphKernelNodeSetAttribute
.. autofunction:: cuda.bindings.runtime.cudaGraphAddMemcpyNode
.. autofunction:: cuda.bindings.runtime.cudaGraphAddMemcpyNode1D
.. autofunction:: cuda.bindings.runtime.cudaGraphMemcpyNodeGetParams
.. autofunction:: cuda.bindings.runtime.cudaGraphMemcpyNodeSetParams
.. autofunction:: cuda.bindings.runtime.cudaGraphMemcpyNodeSetParams1D
.. autofunction:: cuda.bindings.runtime.cudaGraphAddMemsetNode
.. autofunction:: cuda.bindings.runtime.cudaGraphMemsetNodeGetParams
.. autofunction:: cuda.bindings.runtime.cudaGraphMemsetNodeSetParams
.. autofunction:: cuda.bindings.runtime.cudaGraphAddHostNode
.. autofunction:: cuda.bindings.runtime.cudaGraphHostNodeGetParams
.. autofunction:: cuda.bindings.runtime.cudaGraphHostNodeSetParams
.. autofunction:: cuda.bindings.runtime.cudaGraphAddChildGraphNode
.. autofunction:: cuda.bindings.runtime.cudaGraphChildGraphNodeGetGraph
.. autofunction:: cuda.bindings.runtime.cudaGraphAddEmptyNode
.. autofunction:: cuda.bindings.runtime.cudaGraphAddEventRecordNode
.. autofunction:: cuda.bindings.runtime.cudaGraphEventRecordNodeGetEvent
.. autofunction:: cuda.bindings.runtime.cudaGraphEventRecordNodeSetEvent
.. autofunction:: cuda.bindings.runtime.cudaGraphAddEventWaitNode
.. autofunction:: cuda.bindings.runtime.cudaGraphEventWaitNodeGetEvent
.. autofunction:: cuda.bindings.runtime.cudaGraphEventWaitNodeSetEvent
.. autofunction:: cuda.bindings.runtime.cudaGraphAddExternalSemaphoresSignalNode
.. autofunction:: cuda.bindings.runtime.cudaGraphExternalSemaphoresSignalNodeGetParams
.. autofunction:: cuda.bindings.runtime.cudaGraphExternalSemaphoresSignalNodeSetParams
.. autofunction:: cuda.bindings.runtime.cudaGraphAddExternalSemaphoresWaitNode
.. autofunction:: cuda.bindings.runtime.cudaGraphExternalSemaphoresWaitNodeGetParams
.. autofunction:: cuda.bindings.runtime.cudaGraphExternalSemaphoresWaitNodeSetParams
.. autofunction:: cuda.bindings.runtime.cudaGraphAddMemAllocNode
.. autofunction:: cuda.bindings.runtime.cudaGraphMemAllocNodeGetParams
.. autofunction:: cuda.bindings.runtime.cudaGraphAddMemFreeNode
.. autofunction:: cuda.bindings.runtime.cudaGraphMemFreeNodeGetParams
.. autofunction:: cuda.bindings.runtime.cudaDeviceGraphMemTrim
.. autofunction:: cuda.bindings.runtime.cudaDeviceGetGraphMemAttribute
.. autofunction:: cuda.bindings.runtime.cudaDeviceSetGraphMemAttribute
.. autofunction:: cuda.bindings.runtime.cudaGraphClone
.. autofunction:: cuda.bindings.runtime.cudaGraphNodeFindInClone
.. autofunction:: cuda.bindings.runtime.cudaGraphNodeGetType
.. autofunction:: cuda.bindings.runtime.cudaGraphNodeGetContainingGraph
.. autofunction:: cuda.bindings.runtime.cudaGraphNodeGetLocalId
.. autofunction:: cuda.bindings.runtime.cudaGraphNodeGetToolsId
.. autofunction:: cuda.bindings.runtime.cudaGraphGetId
.. autofunction:: cuda.bindings.runtime.cudaGraphExecGetId
.. autofunction:: cuda.bindings.runtime.cudaGraphGetNodes
.. autofunction:: cuda.bindings.runtime.cudaGraphGetRootNodes
.. autofunction:: cuda.bindings.runtime.cudaGraphGetEdges
.. autofunction:: cuda.bindings.runtime.cudaGraphNodeGetDependencies
.. autofunction:: cuda.bindings.runtime.cudaGraphNodeGetDependentNodes
.. autofunction:: cuda.bindings.runtime.cudaGraphAddDependencies
.. autofunction:: cuda.bindings.runtime.cudaGraphRemoveDependencies
.. autofunction:: cuda.bindings.runtime.cudaGraphDestroyNode
.. autofunction:: cuda.bindings.runtime.cudaGraphInstantiate
.. autofunction:: cuda.bindings.runtime.cudaGraphInstantiateWithFlags
.. autofunction:: cuda.bindings.runtime.cudaGraphInstantiateWithParams
.. autofunction:: cuda.bindings.runtime.cudaGraphExecGetFlags
.. autofunction:: cuda.bindings.runtime.cudaGraphExecKernelNodeSetParams
.. autofunction:: cuda.bindings.runtime.cudaGraphExecMemcpyNodeSetParams
.. autofunction:: cuda.bindings.runtime.cudaGraphExecMemcpyNodeSetParams1D
.. autofunction:: cuda.bindings.runtime.cudaGraphExecMemsetNodeSetParams
.. autofunction:: cuda.bindings.runtime.cudaGraphExecHostNodeSetParams
.. autofunction:: cuda.bindings.runtime.cudaGraphExecChildGraphNodeSetParams
.. autofunction:: cuda.bindings.runtime.cudaGraphExecEventRecordNodeSetEvent
.. autofunction:: cuda.bindings.runtime.cudaGraphExecEventWaitNodeSetEvent
.. autofunction:: cuda.bindings.runtime.cudaGraphExecExternalSemaphoresSignalNodeSetParams
.. autofunction:: cuda.bindings.runtime.cudaGraphExecExternalSemaphoresWaitNodeSetParams
.. autofunction:: cuda.bindings.runtime.cudaGraphNodeSetEnabled
.. autofunction:: cuda.bindings.runtime.cudaGraphNodeGetEnabled
.. autofunction:: cuda.bindings.runtime.cudaGraphExecUpdate
.. autofunction:: cuda.bindings.runtime.cudaGraphUpload
.. autofunction:: cuda.bindings.runtime.cudaGraphLaunch
.. autofunction:: cuda.bindings.runtime.cudaGraphExecDestroy
.. autofunction:: cuda.bindings.runtime.cudaGraphDestroy
.. autofunction:: cuda.bindings.runtime.cudaGraphDebugDotPrint
.. autofunction:: cuda.bindings.runtime.cudaUserObjectCreate
.. autofunction:: cuda.bindings.runtime.cudaUserObjectRetain
.. autofunction:: cuda.bindings.runtime.cudaUserObjectRelease
.. autofunction:: cuda.bindings.runtime.cudaGraphRetainUserObject
.. autofunction:: cuda.bindings.runtime.cudaGraphReleaseUserObject
.. autofunction:: cuda.bindings.runtime.cudaGraphAddNode
.. autofunction:: cuda.bindings.runtime.cudaGraphNodeSetParams
.. autofunction:: cuda.bindings.runtime.cudaGraphExecNodeSetParams
.. autofunction:: cuda.bindings.runtime.cudaGraphConditionalHandleCreate
.. autofunction:: cuda.bindings.runtime.cudaGraphConditionalHandleCreate_v2

Driver Entry Point Access
-------------------------

This section describes the driver entry point access functions of CUDA runtime application programming interface.

.. autofunction:: cuda.bindings.runtime.cudaGetDriverEntryPoint
.. autofunction:: cuda.bindings.runtime.cudaGetDriverEntryPointByVersion

Library Management
------------------

This section describes the library management functions of the CUDA runtime application programming interface.

.. autofunction:: cuda.bindings.runtime.cudaLibraryLoadData
.. autofunction:: cuda.bindings.runtime.cudaLibraryLoadFromFile
.. autofunction:: cuda.bindings.runtime.cudaLibraryUnload
.. autofunction:: cuda.bindings.runtime.cudaLibraryGetKernel
.. autofunction:: cuda.bindings.runtime.cudaLibraryGetGlobal
.. autofunction:: cuda.bindings.runtime.cudaLibraryGetManaged
.. autofunction:: cuda.bindings.runtime.cudaLibraryGetUnifiedFunction
.. autofunction:: cuda.bindings.runtime.cudaLibraryGetKernelCount
.. autofunction:: cuda.bindings.runtime.cudaLibraryEnumerateKernels
.. autofunction:: cuda.bindings.runtime.cudaKernelSetAttributeForDevice

Execution Context Management
----------------------------

This section describes the execution context management functions of the CUDA runtime application programming interface.





**Overview**



A CUDA execution context cudaExecutionContext_t serves as an abstraction for the contexts exposed by the CUDA Runtime, specifically green contexts and the primary context, and provides a unified programming model and API interface for contexts in the Runtime.

There are two primary ways today to obtain an execution context:

- cudaDeviceGetExecutionCtx: Returns the execution context that corresponds to the primary context of the specified device.







- cudaGreenCtxCreate: Creates a green context with the specified resources and returns an execution context.









Once you have an execution context at hand, you can perform context-level operations via the CUDA Runtime APIs. This includes:

- Submitting work via streams created with cudaExecutionCtxStreamCreate.







- Querying context via cudaExecutionCtxGetDevResource, cudaExecutionCtxGetDevice, etc.







- Synchronizing and tracking context-level operations via cudaExecutionCtxSynchronize, cudaExecutionCtxRecordEvent, cudaExecutionCtxWaitEvent.







- Performing context-level graph node operations via cudaGraphAddNode by specifying the context in ``nodeParams``\ . Note that individual node creation APIs, such as cudaGraphAddKernelNode, do not support specifying an execution context.









Note: The above APIs take in an explicit cudaExecutionContext_t handle and ignores the context that is current to the calling thread. This enables explicit context-based programming without relying on thread-local state. If no context is specified, the APIs return cudaErrorInvalidValue.

Note: Developers should treat cudaExecutionContext_t as an opaque handle and avoid assumptions about its underlying representation. The CUDA Runtime does not provide a way to convert this handle into driver-level contexts, such as ::CUcontext or ::CUgreenCtx.





**Lifetime of CUDA Resources**



The lifetime of CUDA resources (memory, streams, events, modules, etc) is not tied to the lifetime of the execution context. Their lifetime is tied to the device against which they were created. As such, usage of cudaDeviceReset() should be avoided to persist the lifetime of these resources.





**APIs Operating on Current Context**



The CUDA runtime does not provide a way to set an execution context as current. Since, the majority of the runtime APIs operate on the current context, we document below how the developer can work with these APIs.



**APIs Operating on Device Resources**



To work with these APIs (for example, cudaMalloc, cudaEventCreate, etc), developers are expected to call cudaSetDevice() prior to invoking them. Doing so does not impact functional correctness as these APIs operate on resources that are device-wide. If users have a context handle at hand, they can get the device handle from the context handle using cudaExecutionCtxGetDevice().





**APIs Operating on Context Resources**



These APIs (for example, cudaLaunchKernel, cudaMemcpyAsync, cudaMemsetAsync, etc) take in a stream and resources are inferred from the context bound to the stream at creation. See cudaExecutionCtxStreamCreate for more details. Developers are expected to use the stream-based APIs for context awareness and always pass an explicit stream handle to ensure context-awareness, and avoid reliance on the default NULL stream, which implicitly binds to the current context.







**Green Contexts**



Green contexts are a lightweight alternative to traditional contexts, that can be used to select a subset of device resources. This allows the developer to, for example, select SMs from distinct spatial partitions of the GPU and target them via CUDA stream operations, kernel launches, etc.

Here are the broad initial steps to follow to get started:

- (1) Start with an initial set of resources. For SM resources, they can be fetched via cudaDeviceGetDevResource. In case of workqueues, a new configuration can be used or an existing one queried via the cudaDeviceGetDevResource API.







- (2) Modify these resources by either partitioning them (in case of SMs) or changing the configuration (in case of workqueues). To partition SMs, we recommend cudaDevSmResourceSplit. Changing the workqueue configuration can be done directly in place.







- (3) Finalize the specification of resources by creating a descriptor via cudaDevResourceGenerateDesc.







- (4) Create a green context via cudaGreenCtxCreate. This provisions the resource, such as workqueues (until this step it was only a configuration specification).







- (5) Create a stream via cudaExecutionCtxStreamCreate, and use it throughout your application.









SMs

There are two possible partition operations - with cudaDevSmResourceSplitByCount the partitions created have to follow default SM count granularity requirements, so it will often be rounded up and aligned to a default value. On the other hand, cudaDevSmResourceSplit is explicit and allows for creation of non-equal groups. It will not round up automatically - instead it is the developer’s responsibility to query and set the correct values. These requirements can be queried with cudaDeviceGetDevResource to determine the alignment granularity (sm.smCoscheduledAlignment). A general guideline on the default values for each compute architecture:

- On Compute Architecture 7.X, 8.X, and all Tegra SoC:





  - The smCount must be a multiple of 2.







  - The alignment (and default value of coscheduledSmCount) is 2.









- On Compute Architecture 9.0+:





  - The smCount must be a multiple of 8, or coscheduledSmCount if provided.







  - The alignment (and default value of coscheduledSmCount) is 8. While the maximum value for coscheduled SM count is 32 on all Compute Architecture 9.0+, it's recommended to follow cluster size requirements. The portable cluster size and the max cluster size should be used in order to benefit from this co-scheduling.











Workqueues

For ``cudaDevResourceTypeWorkqueueConfig``\ , the resource specifies the expected maximum number of concurrent stream-ordered workloads via the ``wqConcurrencyLimit``\  field. The ``sharingScope``\  field determines how workqueue resources are shared:

- ``cudaDevWorkqueueConfigScopeDeviceCtx:``\  Use all shared workqueue resources across all contexts (default driver behavior).







- ``cudaDevWorkqueueConfigScopeGreenCtxBalanced:``\  When possible, use non-overlapping workqueue resources with other balanced green contexts.









The maximum concurrency limit depends on ::CUDA_DEVICE_MAX_CONNECTIONS and can be queried from the device via cudaDeviceGetDevResource. Configurations may exceed this concurrency limit, but the driver will not guarantee that work submission remains non-overlapping.

For ``cudaDevResourceTypeWorkqueue``\ , the resource represents a pre-existing workqueue that can be retrieved from existing execution contexts. This allows reusing workqueue resources across different execution contexts.

On Concurrency

Even if the green contexts have disjoint SM partitions, it is not guaranteed that the kernels launched in them will run concurrently or have forward progress guarantees. This is due to other resources that could cause a dependency. Using a combination of disjoint SMs and ``cudaDevWorkqueueConfigScopeGreenCtxBalanced``\  workqueue configurations can provide the best chance of avoiding interference. More resources will be added in the future to provide stronger guarantees.

Additionally, there are two known scenarios, where its possible for the workload to run on more SMs than was provisioned (but never less).



- On Volta+ MPS: When ``CUDA_MPS_ACTIVE_THREAD_PERCENTAGE``\  is used, the set of SMs that are used for running kernels can be scaled up to the value of SMs used for the MPS client.







- On Compute Architecture 9.x: When a module with dynamic parallelism (CDP) is loaded, all future kernels running under green contexts may use and share an additional set of 2 SMs.

.. autofunction:: cuda.bindings.runtime.cudaDeviceGetDevResource
.. autofunction:: cuda.bindings.runtime.cudaDevSmResourceSplitByCount
.. autofunction:: cuda.bindings.runtime.cudaDevSmResourceSplit
.. autofunction:: cuda.bindings.runtime.cudaDevResourceGenerateDesc
.. autofunction:: cuda.bindings.runtime.cudaGreenCtxCreate
.. autofunction:: cuda.bindings.runtime.cudaExecutionCtxDestroy
.. autofunction:: cuda.bindings.runtime.cudaExecutionCtxGetDevResource
.. autofunction:: cuda.bindings.runtime.cudaExecutionCtxGetDevice
.. autofunction:: cuda.bindings.runtime.cudaExecutionCtxGetId
.. autofunction:: cuda.bindings.runtime.cudaExecutionCtxStreamCreate
.. autofunction:: cuda.bindings.runtime.cudaExecutionCtxSynchronize
.. autofunction:: cuda.bindings.runtime.cudaStreamGetDevResource
.. autofunction:: cuda.bindings.runtime.cudaExecutionCtxRecordEvent
.. autofunction:: cuda.bindings.runtime.cudaExecutionCtxWaitEvent
.. autofunction:: cuda.bindings.runtime.cudaDeviceGetExecutionCtx

C++ API Routines
----------------
C++-style interface built on top of CUDA runtime API.
impl_private







This section describes the C++ high level API functions of the CUDA runtime application programming interface. To use these functions, your application needs to be compiled with the ``nvcc``\  compiler.


Interactions with the CUDA Driver API
-------------------------------------

This section describes the interactions between the CUDA Driver API and the CUDA Runtime API





**Execution Contexts**



The CUDA Runtime provides cudaExecutionContext_t as an abstraction over driver-level contexts—specifically, green contexts and the primary context.

There are two primary ways to obtain an execution context:

- cudaDeviceGetExecutionCtx: Returns the execution context that corresponds to the primary context of the specified device.







- cudaGreenCtxCreate: Creates a green context with the specified resources and returns an execution context.









Note: Developers should treat cudaExecutionContext_t as an opaque handle and avoid assumptions about its underlying representation. The CUDA Runtime does not provide a way to convert this handle into a ::CUcontext or ::CUgreenCtx.





**Primary Context (aka Device Execution Context)**



The primary context is the default execution context associated with a device in the Runtime. It can be obtained via a call to cudaDeviceGetExecutionCtx(). There is a one-to-one mapping between CUDA devices in the runtime and their primary contexts within a process.

From the CUDA Runtime’s perspective, a device and its primary context are functionally synonymous.

Unless explicitly overridden, either by making a different context current via the Driver API (e.g., ::cuCtxSetCurrent()) or by using an explicit execution context handle, the Runtime will implicitly initialize and use the primary context for API calls as needed.





**Initialization and Tear-Down**



Unless an explicit execution context is specified (see “Execution Context Management” for APIs), CUDA Runtime API calls operate on the CUDA Driver ::CUcontext which is current to the calling host thread. If no ::CUcontext is current to the calling thread when a CUDA Runtime API call which requires an active context is made, then the primary context (device execution context) for a device will be selected, made current to the calling thread, and initialized. The context will be initialized using the parameters specified by the CUDA Runtime API functions cudaSetDeviceFlags(), ::cudaD3D9SetDirect3DDevice(), ::cudaD3D10SetDirect3DDevice(), ::cudaD3D11SetDirect3DDevice(), cudaGLSetGLDevice(), and cudaVDPAUSetVDPAUDevice(). Note that these functions will fail with cudaErrorSetOnActiveProcess if they are called when the primary context for the specified device has already been initialized, except for cudaSetDeviceFlags() which will simply overwrite the previous settings.

The function cudaInitDevice() ensures that the primary context is initialized for the requested device but does not make it current to the calling thread.

The function cudaSetDevice() initializes the primary context for the specified device and makes it current to the calling thread by calling ::cuCtxSetCurrent().

Primary contexts will remain active until they are explicitly deinitialized using cudaDeviceReset(). The function cudaDeviceReset() will deinitialize the primary context for the calling thread's current device immediately. The context will remain current to all of the threads that it was current to. The next CUDA Runtime API call on any thread which requires an active context will trigger the reinitialization of that device's primary context.

Note that primary contexts are shared resources. It is recommended that the primary context not be reset except just before exit or to recover from an unspecified launch failure.





**CUcontext Interoperability**



Note that the use of multiple ::CUcontext s per device within a single process will substantially degrade performance and is strongly discouraged. Instead, it is highly recommended to either use execution contexts cudaExecutionContext_t or the implicit one-to-one device-to-primary context mapping for the process provided by the CUDA Runtime API.

If a non-primary ::CUcontext created by the CUDA Driver API is current to a thread then the CUDA Runtime API calls to that thread will operate on that ::CUcontext, with some exceptions listed below. Interoperability between data types is discussed in the following sections.

The function cudaDeviceEnablePeerAccess() and the rest of the peer access API may not be called when a non-primary CUcontext is current. To use the peer access APIs with a context created using the CUDA Driver API, it is necessary that the CUDA Driver API be used to access these features.

All CUDA Runtime API state (e.g, global variables' addresses and values) travels with its underlying ::CUcontext. In particular, if a ::CUcontext is moved from one thread to another then all CUDA Runtime API state will move to that thread as well.

Please note that attaching to legacy CUcontext (those with a version of 3010 as returned by ::cuCtxGetApiVersion()) is not possible. The CUDA Runtime will return cudaErrorIncompatibleDriverContext in such cases.





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





**Interactions between CUkernel and cudaKernel_t**



The types ::CUkernel and cudaKernel_t represent the same data type and may be used interchangeably by casting the two types between each other.

In order to use a cudaKernel_t in a CUDA Driver API function which takes a ::CUkernel, it is necessary to explicitly cast the cudaKernel_t to a ::CUkernel.

.. autofunction:: cuda.bindings.runtime.cudaGetKernel

Data types used by CUDA Runtime
-------------------------------



.. autoclass:: cuda.bindings.runtime.cudaEglPlaneDesc_st
.. autoclass:: cuda.bindings.runtime.cudaEglFrame_st
.. autoclass:: cuda.bindings.runtime.cudaChannelFormatDesc
.. autoclass:: cuda.bindings.runtime.cudaArraySparseProperties
.. autoclass:: cuda.bindings.runtime.cudaArrayMemoryRequirements
.. autoclass:: cuda.bindings.runtime.cudaPitchedPtr
.. autoclass:: cuda.bindings.runtime.cudaExtent
.. autoclass:: cuda.bindings.runtime.cudaPos
.. autoclass:: cuda.bindings.runtime.cudaMemcpy3DParms
.. autoclass:: cuda.bindings.runtime.cudaMemcpyNodeParams
.. autoclass:: cuda.bindings.runtime.cudaMemcpy3DPeerParms
.. autoclass:: cuda.bindings.runtime.cudaMemsetParams
.. autoclass:: cuda.bindings.runtime.cudaMemsetParamsV2
.. autoclass:: cuda.bindings.runtime.cudaAccessPolicyWindow
.. autoclass:: cuda.bindings.runtime.cudaHostNodeParams
.. autoclass:: cuda.bindings.runtime.cudaHostNodeParamsV2
.. autoclass:: cuda.bindings.runtime.cudaResourceDesc
.. autoclass:: cuda.bindings.runtime.cudaResourceViewDesc
.. autoclass:: cuda.bindings.runtime.cudaPointerAttributes
.. autoclass:: cuda.bindings.runtime.cudaFuncAttributes
.. autoclass:: cuda.bindings.runtime.cudaMemLocation
.. autoclass:: cuda.bindings.runtime.cudaMemAccessDesc
.. autoclass:: cuda.bindings.runtime.cudaMemPoolProps
.. autoclass:: cuda.bindings.runtime.cudaMemPoolPtrExportData
.. autoclass:: cuda.bindings.runtime.cudaMemAllocNodeParams
.. autoclass:: cuda.bindings.runtime.cudaMemAllocNodeParamsV2
.. autoclass:: cuda.bindings.runtime.cudaMemFreeNodeParams
.. autoclass:: cuda.bindings.runtime.cudaMemcpyAttributes
.. autoclass:: cuda.bindings.runtime.cudaOffset3D
.. autoclass:: cuda.bindings.runtime.cudaMemcpy3DOperand
.. autoclass:: cuda.bindings.runtime.cudaMemcpy3DBatchOp
.. autoclass:: cuda.bindings.runtime.CUuuid_st
.. autoclass:: cuda.bindings.runtime.cudaDeviceProp
.. autoclass:: cuda.bindings.runtime.cudaIpcEventHandle_st
.. autoclass:: cuda.bindings.runtime.cudaIpcMemHandle_st
.. autoclass:: cuda.bindings.runtime.cudaMemFabricHandle_st
.. autoclass:: cuda.bindings.runtime.cudaExternalMemoryHandleDesc
.. autoclass:: cuda.bindings.runtime.cudaExternalMemoryBufferDesc
.. autoclass:: cuda.bindings.runtime.cudaExternalMemoryMipmappedArrayDesc
.. autoclass:: cuda.bindings.runtime.cudaExternalSemaphoreHandleDesc
.. autoclass:: cuda.bindings.runtime.cudaExternalSemaphoreSignalParams
.. autoclass:: cuda.bindings.runtime.cudaExternalSemaphoreWaitParams
.. autoclass:: cuda.bindings.runtime.cudaDevSmResource
.. autoclass:: cuda.bindings.runtime.cudaDevWorkqueueConfigResource
.. autoclass:: cuda.bindings.runtime.cudaDevWorkqueueResource
.. autoclass:: cuda.bindings.runtime.cudaDevSmResourceGroupParams_st
.. autoclass:: cuda.bindings.runtime.cudaDevResource_st
.. autoclass:: cuda.bindings.runtime.cudalibraryHostUniversalFunctionAndDataTable
.. autoclass:: cuda.bindings.runtime.cudaKernelNodeParams
.. autoclass:: cuda.bindings.runtime.cudaKernelNodeParamsV2
.. autoclass:: cuda.bindings.runtime.cudaExternalSemaphoreSignalNodeParams
.. autoclass:: cuda.bindings.runtime.cudaExternalSemaphoreSignalNodeParamsV2
.. autoclass:: cuda.bindings.runtime.cudaExternalSemaphoreWaitNodeParams
.. autoclass:: cuda.bindings.runtime.cudaExternalSemaphoreWaitNodeParamsV2
.. autoclass:: cuda.bindings.runtime.cudaConditionalNodeParams
.. autoclass:: cuda.bindings.runtime.cudaChildGraphNodeParams
.. autoclass:: cuda.bindings.runtime.cudaEventRecordNodeParams
.. autoclass:: cuda.bindings.runtime.cudaEventWaitNodeParams
.. autoclass:: cuda.bindings.runtime.cudaGraphNodeParams
.. autoclass:: cuda.bindings.runtime.cudaGraphEdgeData_st
.. autoclass:: cuda.bindings.runtime.cudaGraphInstantiateParams_st
.. autoclass:: cuda.bindings.runtime.cudaGraphExecUpdateResultInfo_st
.. autoclass:: cuda.bindings.runtime.cudaGraphKernelNodeUpdate
.. autoclass:: cuda.bindings.runtime.cudaLaunchMemSyncDomainMap_st
.. autoclass:: cuda.bindings.runtime.cudaLaunchAttributeValue
.. autoclass:: cuda.bindings.runtime.cudaLaunchAttribute_st
.. autoclass:: cuda.bindings.runtime.cudaAsyncNotificationInfo
.. autoclass:: cuda.bindings.runtime.cudaTextureDesc
.. autoclass:: cuda.bindings.runtime.cudaEglFrameType

    .. autoattribute:: cuda.bindings.runtime.cudaEglFrameType.cudaEglFrameTypeArray


        Frame type CUDA array


    .. autoattribute:: cuda.bindings.runtime.cudaEglFrameType.cudaEglFrameTypePitch


        Frame type CUDA pointer

.. autoclass:: cuda.bindings.runtime.cudaEglResourceLocationFlags

    .. autoattribute:: cuda.bindings.runtime.cudaEglResourceLocationFlags.cudaEglResourceLocationSysmem


        Resource location sysmem


    .. autoattribute:: cuda.bindings.runtime.cudaEglResourceLocationFlags.cudaEglResourceLocationVidmem


        Resource location vidmem

.. autoclass:: cuda.bindings.runtime.cudaEglColorFormat

    .. autoattribute:: cuda.bindings.runtime.cudaEglColorFormat.cudaEglColorFormatYUV420Planar


        Y, U, V in three surfaces, each in a separate surface, U/V width = 1/2 Y width, U/V height = 1/2 Y height.


    .. autoattribute:: cuda.bindings.runtime.cudaEglColorFormat.cudaEglColorFormatYUV420SemiPlanar


        Y, UV in two surfaces (UV as one surface) with VU byte ordering, width, height ratio same as YUV420Planar.


    .. autoattribute:: cuda.bindings.runtime.cudaEglColorFormat.cudaEglColorFormatYUV422Planar


        Y, U, V each in a separate surface, U/V width = 1/2 Y width, U/V height = Y height.


    .. autoattribute:: cuda.bindings.runtime.cudaEglColorFormat.cudaEglColorFormatYUV422SemiPlanar


        Y, UV in two surfaces with VU byte ordering, width, height ratio same as YUV422Planar.


    .. autoattribute:: cuda.bindings.runtime.cudaEglColorFormat.cudaEglColorFormatARGB


        R/G/B/A four channels in one surface with BGRA byte ordering.


    .. autoattribute:: cuda.bindings.runtime.cudaEglColorFormat.cudaEglColorFormatRGBA


        R/G/B/A four channels in one surface with ABGR byte ordering.


    .. autoattribute:: cuda.bindings.runtime.cudaEglColorFormat.cudaEglColorFormatL


        single luminance channel in one surface.


    .. autoattribute:: cuda.bindings.runtime.cudaEglColorFormat.cudaEglColorFormatR


        single color channel in one surface.


    .. autoattribute:: cuda.bindings.runtime.cudaEglColorFormat.cudaEglColorFormatYUV444Planar


        Y, U, V in three surfaces, each in a separate surface, U/V width = Y width, U/V height = Y height.


    .. autoattribute:: cuda.bindings.runtime.cudaEglColorFormat.cudaEglColorFormatYUV444SemiPlanar


        Y, UV in two surfaces (UV as one surface) with VU byte ordering, width, height ratio same as YUV444Planar.


    .. autoattribute:: cuda.bindings.runtime.cudaEglColorFormat.cudaEglColorFormatYUYV422


        Y, U, V in one surface, interleaved as UYVY in one channel.


    .. autoattribute:: cuda.bindings.runtime.cudaEglColorFormat.cudaEglColorFormatUYVY422


        Y, U, V in one surface, interleaved as YUYV in one channel.


    .. autoattribute:: cuda.bindings.runtime.cudaEglColorFormat.cudaEglColorFormatABGR


        R/G/B/A four channels in one surface with RGBA byte ordering.


    .. autoattribute:: cuda.bindings.runtime.cudaEglColorFormat.cudaEglColorFormatBGRA


        R/G/B/A four channels in one surface with ARGB byte ordering.


    .. autoattribute:: cuda.bindings.runtime.cudaEglColorFormat.cudaEglColorFormatA


        Alpha color format - one channel in one surface.


    .. autoattribute:: cuda.bindings.runtime.cudaEglColorFormat.cudaEglColorFormatRG


        R/G color format - two channels in one surface with GR byte ordering


    .. autoattribute:: cuda.bindings.runtime.cudaEglColorFormat.cudaEglColorFormatAYUV


        Y, U, V, A four channels in one surface, interleaved as VUYA.


    .. autoattribute:: cuda.bindings.runtime.cudaEglColorFormat.cudaEglColorFormatYVU444SemiPlanar


        Y, VU in two surfaces (VU as one surface) with UV byte ordering, U/V width = Y width, U/V height = Y height.


    .. autoattribute:: cuda.bindings.runtime.cudaEglColorFormat.cudaEglColorFormatYVU422SemiPlanar


        Y, VU in two surfaces (VU as one surface) with UV byte ordering, U/V width = 1/2 Y width, U/V height = Y height.


    .. autoattribute:: cuda.bindings.runtime.cudaEglColorFormat.cudaEglColorFormatYVU420SemiPlanar


        Y, VU in two surfaces (VU as one surface) with UV byte ordering, U/V width = 1/2 Y width, U/V height = 1/2 Y height.


    .. autoattribute:: cuda.bindings.runtime.cudaEglColorFormat.cudaEglColorFormatY10V10U10_444SemiPlanar


        Y10, V10U10 in two surfaces (VU as one surface) with UV byte ordering, U/V width = Y width, U/V height = Y height.


    .. autoattribute:: cuda.bindings.runtime.cudaEglColorFormat.cudaEglColorFormatY10V10U10_420SemiPlanar


        Y10, V10U10 in two surfaces (VU as one surface) with UV byte ordering, U/V width = 1/2 Y width, U/V height = 1/2 Y height.


    .. autoattribute:: cuda.bindings.runtime.cudaEglColorFormat.cudaEglColorFormatY12V12U12_444SemiPlanar


        Y12, V12U12 in two surfaces (VU as one surface) with UV byte ordering, U/V width = Y width, U/V height = Y height.


    .. autoattribute:: cuda.bindings.runtime.cudaEglColorFormat.cudaEglColorFormatY12V12U12_420SemiPlanar


        Y12, V12U12 in two surfaces (VU as one surface) with UV byte ordering, U/V width = 1/2 Y width, U/V height = 1/2 Y height.


    .. autoattribute:: cuda.bindings.runtime.cudaEglColorFormat.cudaEglColorFormatVYUY_ER


        Extended Range Y, U, V in one surface, interleaved as YVYU in one channel.


    .. autoattribute:: cuda.bindings.runtime.cudaEglColorFormat.cudaEglColorFormatUYVY_ER


        Extended Range Y, U, V in one surface, interleaved as YUYV in one channel.


    .. autoattribute:: cuda.bindings.runtime.cudaEglColorFormat.cudaEglColorFormatYUYV_ER


        Extended Range Y, U, V in one surface, interleaved as UYVY in one channel.


    .. autoattribute:: cuda.bindings.runtime.cudaEglColorFormat.cudaEglColorFormatYVYU_ER


        Extended Range Y, U, V in one surface, interleaved as VYUY in one channel.


    .. autoattribute:: cuda.bindings.runtime.cudaEglColorFormat.cudaEglColorFormatYUVA_ER


        Extended Range Y, U, V, A four channels in one surface, interleaved as AVUY.


    .. autoattribute:: cuda.bindings.runtime.cudaEglColorFormat.cudaEglColorFormatAYUV_ER


        Extended Range Y, U, V, A four channels in one surface, interleaved as VUYA.


    .. autoattribute:: cuda.bindings.runtime.cudaEglColorFormat.cudaEglColorFormatYUV444Planar_ER


        Extended Range Y, U, V in three surfaces, U/V width = Y width, U/V height = Y height.


    .. autoattribute:: cuda.bindings.runtime.cudaEglColorFormat.cudaEglColorFormatYUV422Planar_ER


        Extended Range Y, U, V in three surfaces, U/V width = 1/2 Y width, U/V height = Y height.


    .. autoattribute:: cuda.bindings.runtime.cudaEglColorFormat.cudaEglColorFormatYUV420Planar_ER


        Extended Range Y, U, V in three surfaces, U/V width = 1/2 Y width, U/V height = 1/2 Y height.


    .. autoattribute:: cuda.bindings.runtime.cudaEglColorFormat.cudaEglColorFormatYUV444SemiPlanar_ER


        Extended Range Y, UV in two surfaces (UV as one surface) with VU byte ordering, U/V width = Y width, U/V height = Y height.


    .. autoattribute:: cuda.bindings.runtime.cudaEglColorFormat.cudaEglColorFormatYUV422SemiPlanar_ER


        Extended Range Y, UV in two surfaces (UV as one surface) with VU byte ordering, U/V width = 1/2 Y width, U/V height = Y height.


    .. autoattribute:: cuda.bindings.runtime.cudaEglColorFormat.cudaEglColorFormatYUV420SemiPlanar_ER


        Extended Range Y, UV in two surfaces (UV as one surface) with VU byte ordering, U/V width = 1/2 Y width, U/V height = 1/2 Y height.


    .. autoattribute:: cuda.bindings.runtime.cudaEglColorFormat.cudaEglColorFormatYVU444Planar_ER


        Extended Range Y, V, U in three surfaces, U/V width = Y width, U/V height = Y height.


    .. autoattribute:: cuda.bindings.runtime.cudaEglColorFormat.cudaEglColorFormatYVU422Planar_ER


        Extended Range Y, V, U in three surfaces, U/V width = 1/2 Y width, U/V height = Y height.


    .. autoattribute:: cuda.bindings.runtime.cudaEglColorFormat.cudaEglColorFormatYVU420Planar_ER


        Extended Range Y, V, U in three surfaces, U/V width = 1/2 Y width, U/V height = 1/2 Y height.


    .. autoattribute:: cuda.bindings.runtime.cudaEglColorFormat.cudaEglColorFormatYVU444SemiPlanar_ER


        Extended Range Y, VU in two surfaces (VU as one surface) with UV byte ordering, U/V width = Y width, U/V height = Y height.


    .. autoattribute:: cuda.bindings.runtime.cudaEglColorFormat.cudaEglColorFormatYVU422SemiPlanar_ER


        Extended Range Y, VU in two surfaces (VU as one surface) with UV byte ordering, U/V width = 1/2 Y width, U/V height = Y height.


    .. autoattribute:: cuda.bindings.runtime.cudaEglColorFormat.cudaEglColorFormatYVU420SemiPlanar_ER


        Extended Range Y, VU in two surfaces (VU as one surface) with UV byte ordering, U/V width = 1/2 Y width, U/V height = 1/2 Y height.


    .. autoattribute:: cuda.bindings.runtime.cudaEglColorFormat.cudaEglColorFormatBayerRGGB


        Bayer format - one channel in one surface with interleaved RGGB ordering.


    .. autoattribute:: cuda.bindings.runtime.cudaEglColorFormat.cudaEglColorFormatBayerBGGR


        Bayer format - one channel in one surface with interleaved BGGR ordering.


    .. autoattribute:: cuda.bindings.runtime.cudaEglColorFormat.cudaEglColorFormatBayerGRBG


        Bayer format - one channel in one surface with interleaved GRBG ordering.


    .. autoattribute:: cuda.bindings.runtime.cudaEglColorFormat.cudaEglColorFormatBayerGBRG


        Bayer format - one channel in one surface with interleaved GBRG ordering.


    .. autoattribute:: cuda.bindings.runtime.cudaEglColorFormat.cudaEglColorFormatBayer10RGGB


        Bayer10 format - one channel in one surface with interleaved RGGB ordering. Out of 16 bits, 10 bits used 6 bits No-op.


    .. autoattribute:: cuda.bindings.runtime.cudaEglColorFormat.cudaEglColorFormatBayer10BGGR


        Bayer10 format - one channel in one surface with interleaved BGGR ordering. Out of 16 bits, 10 bits used 6 bits No-op.


    .. autoattribute:: cuda.bindings.runtime.cudaEglColorFormat.cudaEglColorFormatBayer10GRBG


        Bayer10 format - one channel in one surface with interleaved GRBG ordering. Out of 16 bits, 10 bits used 6 bits No-op.


    .. autoattribute:: cuda.bindings.runtime.cudaEglColorFormat.cudaEglColorFormatBayer10GBRG


        Bayer10 format - one channel in one surface with interleaved GBRG ordering. Out of 16 bits, 10 bits used 6 bits No-op.


    .. autoattribute:: cuda.bindings.runtime.cudaEglColorFormat.cudaEglColorFormatBayer12RGGB


        Bayer12 format - one channel in one surface with interleaved RGGB ordering. Out of 16 bits, 12 bits used 4 bits No-op.


    .. autoattribute:: cuda.bindings.runtime.cudaEglColorFormat.cudaEglColorFormatBayer12BGGR


        Bayer12 format - one channel in one surface with interleaved BGGR ordering. Out of 16 bits, 12 bits used 4 bits No-op.


    .. autoattribute:: cuda.bindings.runtime.cudaEglColorFormat.cudaEglColorFormatBayer12GRBG


        Bayer12 format - one channel in one surface with interleaved GRBG ordering. Out of 16 bits, 12 bits used 4 bits No-op.


    .. autoattribute:: cuda.bindings.runtime.cudaEglColorFormat.cudaEglColorFormatBayer12GBRG


        Bayer12 format - one channel in one surface with interleaved GBRG ordering. Out of 16 bits, 12 bits used 4 bits No-op.


    .. autoattribute:: cuda.bindings.runtime.cudaEglColorFormat.cudaEglColorFormatBayer14RGGB


        Bayer14 format - one channel in one surface with interleaved RGGB ordering. Out of 16 bits, 14 bits used 2 bits No-op.


    .. autoattribute:: cuda.bindings.runtime.cudaEglColorFormat.cudaEglColorFormatBayer14BGGR


        Bayer14 format - one channel in one surface with interleaved BGGR ordering. Out of 16 bits, 14 bits used 2 bits No-op.


    .. autoattribute:: cuda.bindings.runtime.cudaEglColorFormat.cudaEglColorFormatBayer14GRBG


        Bayer14 format - one channel in one surface with interleaved GRBG ordering. Out of 16 bits, 14 bits used 2 bits No-op.


    .. autoattribute:: cuda.bindings.runtime.cudaEglColorFormat.cudaEglColorFormatBayer14GBRG


        Bayer14 format - one channel in one surface with interleaved GBRG ordering. Out of 16 bits, 14 bits used 2 bits No-op.


    .. autoattribute:: cuda.bindings.runtime.cudaEglColorFormat.cudaEglColorFormatBayer20RGGB


        Bayer20 format - one channel in one surface with interleaved RGGB ordering. Out of 32 bits, 20 bits used 12 bits No-op.


    .. autoattribute:: cuda.bindings.runtime.cudaEglColorFormat.cudaEglColorFormatBayer20BGGR


        Bayer20 format - one channel in one surface with interleaved BGGR ordering. Out of 32 bits, 20 bits used 12 bits No-op.


    .. autoattribute:: cuda.bindings.runtime.cudaEglColorFormat.cudaEglColorFormatBayer20GRBG


        Bayer20 format - one channel in one surface with interleaved GRBG ordering. Out of 32 bits, 20 bits used 12 bits No-op.


    .. autoattribute:: cuda.bindings.runtime.cudaEglColorFormat.cudaEglColorFormatBayer20GBRG


        Bayer20 format - one channel in one surface with interleaved GBRG ordering. Out of 32 bits, 20 bits used 12 bits No-op.


    .. autoattribute:: cuda.bindings.runtime.cudaEglColorFormat.cudaEglColorFormatYVU444Planar


        Y, V, U in three surfaces, each in a separate surface, U/V width = Y width, U/V height = Y height.


    .. autoattribute:: cuda.bindings.runtime.cudaEglColorFormat.cudaEglColorFormatYVU422Planar


        Y, V, U in three surfaces, each in a separate surface, U/V width = 1/2 Y width, U/V height = Y height.


    .. autoattribute:: cuda.bindings.runtime.cudaEglColorFormat.cudaEglColorFormatYVU420Planar


        Y, V, U in three surfaces, each in a separate surface, U/V width = 1/2 Y width, U/V height = 1/2 Y height.


    .. autoattribute:: cuda.bindings.runtime.cudaEglColorFormat.cudaEglColorFormatBayerIspRGGB


        Nvidia proprietary Bayer ISP format - one channel in one surface with interleaved RGGB ordering and mapped to opaque integer datatype.


    .. autoattribute:: cuda.bindings.runtime.cudaEglColorFormat.cudaEglColorFormatBayerIspBGGR


        Nvidia proprietary Bayer ISP format - one channel in one surface with interleaved BGGR ordering and mapped to opaque integer datatype.


    .. autoattribute:: cuda.bindings.runtime.cudaEglColorFormat.cudaEglColorFormatBayerIspGRBG


        Nvidia proprietary Bayer ISP format - one channel in one surface with interleaved GRBG ordering and mapped to opaque integer datatype.


    .. autoattribute:: cuda.bindings.runtime.cudaEglColorFormat.cudaEglColorFormatBayerIspGBRG


        Nvidia proprietary Bayer ISP format - one channel in one surface with interleaved GBRG ordering and mapped to opaque integer datatype.


    .. autoattribute:: cuda.bindings.runtime.cudaEglColorFormat.cudaEglColorFormatBayerBCCR


        Bayer format - one channel in one surface with interleaved BCCR ordering.


    .. autoattribute:: cuda.bindings.runtime.cudaEglColorFormat.cudaEglColorFormatBayerRCCB


        Bayer format - one channel in one surface with interleaved RCCB ordering.


    .. autoattribute:: cuda.bindings.runtime.cudaEglColorFormat.cudaEglColorFormatBayerCRBC


        Bayer format - one channel in one surface with interleaved CRBC ordering.


    .. autoattribute:: cuda.bindings.runtime.cudaEglColorFormat.cudaEglColorFormatBayerCBRC


        Bayer format - one channel in one surface with interleaved CBRC ordering.


    .. autoattribute:: cuda.bindings.runtime.cudaEglColorFormat.cudaEglColorFormatBayer10CCCC


        Bayer10 format - one channel in one surface with interleaved CCCC ordering. Out of 16 bits, 10 bits used 6 bits No-op.


    .. autoattribute:: cuda.bindings.runtime.cudaEglColorFormat.cudaEglColorFormatBayer12BCCR


        Bayer12 format - one channel in one surface with interleaved BCCR ordering. Out of 16 bits, 12 bits used 4 bits No-op.


    .. autoattribute:: cuda.bindings.runtime.cudaEglColorFormat.cudaEglColorFormatBayer12RCCB


        Bayer12 format - one channel in one surface with interleaved RCCB ordering. Out of 16 bits, 12 bits used 4 bits No-op.


    .. autoattribute:: cuda.bindings.runtime.cudaEglColorFormat.cudaEglColorFormatBayer12CRBC


        Bayer12 format - one channel in one surface with interleaved CRBC ordering. Out of 16 bits, 12 bits used 4 bits No-op.


    .. autoattribute:: cuda.bindings.runtime.cudaEglColorFormat.cudaEglColorFormatBayer12CBRC


        Bayer12 format - one channel in one surface with interleaved CBRC ordering. Out of 16 bits, 12 bits used 4 bits No-op.


    .. autoattribute:: cuda.bindings.runtime.cudaEglColorFormat.cudaEglColorFormatBayer12CCCC


        Bayer12 format - one channel in one surface with interleaved CCCC ordering. Out of 16 bits, 12 bits used 4 bits No-op.


    .. autoattribute:: cuda.bindings.runtime.cudaEglColorFormat.cudaEglColorFormatY


        Color format for single Y plane.


    .. autoattribute:: cuda.bindings.runtime.cudaEglColorFormat.cudaEglColorFormatYUV420SemiPlanar_2020


        Y, UV in two surfaces (UV as one surface) U/V width = 1/2 Y width, U/V height = 1/2 Y height.


    .. autoattribute:: cuda.bindings.runtime.cudaEglColorFormat.cudaEglColorFormatYVU420SemiPlanar_2020


        Y, VU in two surfaces (VU as one surface) U/V width = 1/2 Y width, U/V height = 1/2 Y height.


    .. autoattribute:: cuda.bindings.runtime.cudaEglColorFormat.cudaEglColorFormatYUV420Planar_2020


        Y, U, V in three surfaces, each in a separate surface, U/V width = 1/2 Y width, U/V height = 1/2 Y height.


    .. autoattribute:: cuda.bindings.runtime.cudaEglColorFormat.cudaEglColorFormatYVU420Planar_2020


        Y, V, U in three surfaces, each in a separate surface, U/V width = 1/2 Y width, U/V height = 1/2 Y height.


    .. autoattribute:: cuda.bindings.runtime.cudaEglColorFormat.cudaEglColorFormatYUV420SemiPlanar_709


        Y, UV in two surfaces (UV as one surface) U/V width = 1/2 Y width, U/V height = 1/2 Y height.


    .. autoattribute:: cuda.bindings.runtime.cudaEglColorFormat.cudaEglColorFormatYVU420SemiPlanar_709


        Y, VU in two surfaces (VU as one surface) U/V width = 1/2 Y width, U/V height = 1/2 Y height.


    .. autoattribute:: cuda.bindings.runtime.cudaEglColorFormat.cudaEglColorFormatYUV420Planar_709


        Y, U, V in three surfaces, each in a separate surface, U/V width = 1/2 Y width, U/V height = 1/2 Y height.


    .. autoattribute:: cuda.bindings.runtime.cudaEglColorFormat.cudaEglColorFormatYVU420Planar_709


        Y, V, U in three surfaces, each in a separate surface, U/V width = 1/2 Y width, U/V height = 1/2 Y height.


    .. autoattribute:: cuda.bindings.runtime.cudaEglColorFormat.cudaEglColorFormatY10V10U10_420SemiPlanar_709


        Y10, V10U10 in two surfaces (VU as one surface) U/V width = 1/2 Y width, U/V height = 1/2 Y height.


    .. autoattribute:: cuda.bindings.runtime.cudaEglColorFormat.cudaEglColorFormatY10V10U10_420SemiPlanar_2020


        Y10, V10U10 in two surfaces (VU as one surface) U/V width = 1/2 Y width, U/V height = 1/2 Y height.


    .. autoattribute:: cuda.bindings.runtime.cudaEglColorFormat.cudaEglColorFormatY10V10U10_422SemiPlanar_2020


        Y10, V10U10 in two surfaces (VU as one surface) U/V width = 1/2 Y width, U/V height = Y height.


    .. autoattribute:: cuda.bindings.runtime.cudaEglColorFormat.cudaEglColorFormatY10V10U10_422SemiPlanar


        Y10, V10U10 in two surfaces (VU as one surface) U/V width = 1/2 Y width, U/V height = Y height.


    .. autoattribute:: cuda.bindings.runtime.cudaEglColorFormat.cudaEglColorFormatY10V10U10_422SemiPlanar_709


        Y10, V10U10 in two surfaces (VU as one surface) U/V width = 1/2 Y width, U/V height = Y height.


    .. autoattribute:: cuda.bindings.runtime.cudaEglColorFormat.cudaEglColorFormatY_ER


        Extended Range Color format for single Y plane.


    .. autoattribute:: cuda.bindings.runtime.cudaEglColorFormat.cudaEglColorFormatY_709_ER


        Extended Range Color format for single Y plane.


    .. autoattribute:: cuda.bindings.runtime.cudaEglColorFormat.cudaEglColorFormatY10_ER


        Extended Range Color format for single Y10 plane.


    .. autoattribute:: cuda.bindings.runtime.cudaEglColorFormat.cudaEglColorFormatY10_709_ER


        Extended Range Color format for single Y10 plane.


    .. autoattribute:: cuda.bindings.runtime.cudaEglColorFormat.cudaEglColorFormatY12_ER


        Extended Range Color format for single Y12 plane.


    .. autoattribute:: cuda.bindings.runtime.cudaEglColorFormat.cudaEglColorFormatY12_709_ER


        Extended Range Color format for single Y12 plane.


    .. autoattribute:: cuda.bindings.runtime.cudaEglColorFormat.cudaEglColorFormatYUVA


        Y, U, V, A four channels in one surface, interleaved as AVUY.


    .. autoattribute:: cuda.bindings.runtime.cudaEglColorFormat.cudaEglColorFormatYVYU


        Y, U, V in one surface, interleaved as YVYU in one channel.


    .. autoattribute:: cuda.bindings.runtime.cudaEglColorFormat.cudaEglColorFormatVYUY


        Y, U, V in one surface, interleaved as VYUY in one channel.


    .. autoattribute:: cuda.bindings.runtime.cudaEglColorFormat.cudaEglColorFormatY10V10U10_420SemiPlanar_ER


        Extended Range Y10, V10U10 in two surfaces (VU as one surface) U/V width = 1/2 Y width, U/V height = 1/2 Y height.


    .. autoattribute:: cuda.bindings.runtime.cudaEglColorFormat.cudaEglColorFormatY10V10U10_420SemiPlanar_709_ER


        Extended Range Y10, V10U10 in two surfaces (VU as one surface) U/V width = 1/2 Y width, U/V height = 1/2 Y height.


    .. autoattribute:: cuda.bindings.runtime.cudaEglColorFormat.cudaEglColorFormatY10V10U10_444SemiPlanar_ER


        Extended Range Y10, V10U10 in two surfaces (VU as one surface) U/V width = Y width, U/V height = Y height.


    .. autoattribute:: cuda.bindings.runtime.cudaEglColorFormat.cudaEglColorFormatY10V10U10_444SemiPlanar_709_ER


        Extended Range Y10, V10U10 in two surfaces (VU as one surface) U/V width = Y width, U/V height = Y height.


    .. autoattribute:: cuda.bindings.runtime.cudaEglColorFormat.cudaEglColorFormatY12V12U12_420SemiPlanar_ER


        Extended Range Y12, V12U12 in two surfaces (VU as one surface) U/V width = 1/2 Y width, U/V height = 1/2 Y height.


    .. autoattribute:: cuda.bindings.runtime.cudaEglColorFormat.cudaEglColorFormatY12V12U12_420SemiPlanar_709_ER


        Extended Range Y12, V12U12 in two surfaces (VU as one surface) U/V width = 1/2 Y width, U/V height = 1/2 Y height.


    .. autoattribute:: cuda.bindings.runtime.cudaEglColorFormat.cudaEglColorFormatY12V12U12_444SemiPlanar_ER


        Extended Range Y12, V12U12 in two surfaces (VU as one surface) U/V width = Y width, U/V height = Y height.


    .. autoattribute:: cuda.bindings.runtime.cudaEglColorFormat.cudaEglColorFormatY12V12U12_444SemiPlanar_709_ER


        Extended Range Y12, V12U12 in two surfaces (VU as one surface) U/V width = Y width, U/V height = Y height.


    .. autoattribute:: cuda.bindings.runtime.cudaEglColorFormat.cudaEglColorFormatUYVY709


        Y, U, V in one surface, interleaved as UYVY in one channel.


    .. autoattribute:: cuda.bindings.runtime.cudaEglColorFormat.cudaEglColorFormatUYVY709_ER


        Extended Range Y, U, V in one surface, interleaved as UYVY in one channel.


    .. autoattribute:: cuda.bindings.runtime.cudaEglColorFormat.cudaEglColorFormatUYVY2020


        Y, U, V in one surface, interleaved as UYVY in one channel.

.. autoclass:: cuda.bindings.runtime.cudaError_t

    .. autoattribute:: cuda.bindings.runtime.cudaError_t.cudaSuccess


        The API call returned with no errors. In the case of query calls, this also means that the operation being queried is complete (see :py:obj:`~.cudaEventQuery()` and :py:obj:`~.cudaStreamQuery()`).


    .. autoattribute:: cuda.bindings.runtime.cudaError_t.cudaErrorInvalidValue


        This indicates that one or more of the parameters passed to the API call is not within an acceptable range of values.


    .. autoattribute:: cuda.bindings.runtime.cudaError_t.cudaErrorMemoryAllocation


        The API call failed because it was unable to allocate enough memory or other resources to perform the requested operation.


    .. autoattribute:: cuda.bindings.runtime.cudaError_t.cudaErrorInitializationError


        The API call failed because the CUDA driver and runtime could not be initialized.


    .. autoattribute:: cuda.bindings.runtime.cudaError_t.cudaErrorCudartUnloading


        This indicates that a CUDA Runtime API call cannot be executed because it is being called during process shut down, at a point in time after CUDA driver has been unloaded.


    .. autoattribute:: cuda.bindings.runtime.cudaError_t.cudaErrorProfilerDisabled


        This indicates profiler is not initialized for this run. This can happen when the application is running with external profiling tools like visual profiler.


    .. autoattribute:: cuda.bindings.runtime.cudaError_t.cudaErrorProfilerNotInitialized


        [Deprecated]


    .. autoattribute:: cuda.bindings.runtime.cudaError_t.cudaErrorProfilerAlreadyStarted


        [Deprecated]


    .. autoattribute:: cuda.bindings.runtime.cudaError_t.cudaErrorProfilerAlreadyStopped


        [Deprecated]


    .. autoattribute:: cuda.bindings.runtime.cudaError_t.cudaErrorInvalidConfiguration


        This indicates that a kernel launch is requesting resources that can never be satisfied by the current device. Requesting more shared memory per block than the device supports will trigger this error, as will requesting too many threads or blocks. See :py:obj:`~.cudaDeviceProp` for more device limitations.


    .. autoattribute:: cuda.bindings.runtime.cudaError_t.cudaErrorInvalidPitchValue


        This indicates that one or more of the pitch-related parameters passed to the API call is not within the acceptable range for pitch.


    .. autoattribute:: cuda.bindings.runtime.cudaError_t.cudaErrorInvalidSymbol


        This indicates that the symbol name/identifier passed to the API call is not a valid name or identifier.


    .. autoattribute:: cuda.bindings.runtime.cudaError_t.cudaErrorInvalidHostPointer


        This indicates that at least one host pointer passed to the API call is not a valid host pointer. [Deprecated]


    .. autoattribute:: cuda.bindings.runtime.cudaError_t.cudaErrorInvalidDevicePointer


        This indicates that at least one device pointer passed to the API call is not a valid device pointer. [Deprecated]


    .. autoattribute:: cuda.bindings.runtime.cudaError_t.cudaErrorInvalidTexture


        This indicates that the texture passed to the API call is not a valid texture.


    .. autoattribute:: cuda.bindings.runtime.cudaError_t.cudaErrorInvalidTextureBinding


        This indicates that the texture binding is not valid. This occurs if you call :py:obj:`~.cudaGetTextureAlignmentOffset()` with an unbound texture.


    .. autoattribute:: cuda.bindings.runtime.cudaError_t.cudaErrorInvalidChannelDescriptor


        This indicates that the channel descriptor passed to the API call is not valid. This occurs if the format is not one of the formats specified by :py:obj:`~.cudaChannelFormatKind`, or if one of the dimensions is invalid.


    .. autoattribute:: cuda.bindings.runtime.cudaError_t.cudaErrorInvalidMemcpyDirection


        This indicates that the direction of the memcpy passed to the API call is not one of the types specified by :py:obj:`~.cudaMemcpyKind`.


    .. autoattribute:: cuda.bindings.runtime.cudaError_t.cudaErrorAddressOfConstant


        This indicated that the user has taken the address of a constant variable, which was forbidden up until the CUDA 3.1 release. [Deprecated]


    .. autoattribute:: cuda.bindings.runtime.cudaError_t.cudaErrorTextureFetchFailed


        This indicated that a texture fetch was not able to be performed. This was previously used for device emulation of texture operations. [Deprecated]


    .. autoattribute:: cuda.bindings.runtime.cudaError_t.cudaErrorTextureNotBound


        This indicated that a texture was not bound for access. This was previously used for device emulation of texture operations. [Deprecated]


    .. autoattribute:: cuda.bindings.runtime.cudaError_t.cudaErrorSynchronizationError


        This indicated that a synchronization operation had failed. This was previously used for some device emulation functions. [Deprecated]


    .. autoattribute:: cuda.bindings.runtime.cudaError_t.cudaErrorInvalidFilterSetting


        This indicates that a non-float texture was being accessed with linear filtering. This is not supported by CUDA.


    .. autoattribute:: cuda.bindings.runtime.cudaError_t.cudaErrorInvalidNormSetting


        This indicates that an attempt was made to read an unsupported data type as a normalized float. This is not supported by CUDA.


    .. autoattribute:: cuda.bindings.runtime.cudaError_t.cudaErrorMixedDeviceExecution


        Mixing of device and device emulation code was not allowed. [Deprecated]


    .. autoattribute:: cuda.bindings.runtime.cudaError_t.cudaErrorNotYetImplemented


        This indicates that the API call is not yet implemented. Production releases of CUDA will never return this error. [Deprecated]


    .. autoattribute:: cuda.bindings.runtime.cudaError_t.cudaErrorMemoryValueTooLarge


        This indicated that an emulated device pointer exceeded the 32-bit address range. [Deprecated]


    .. autoattribute:: cuda.bindings.runtime.cudaError_t.cudaErrorStubLibrary


        This indicates that the CUDA driver that the application has loaded is a stub library. Applications that run with the stub rather than a real driver loaded will result in CUDA API returning this error.


    .. autoattribute:: cuda.bindings.runtime.cudaError_t.cudaErrorInsufficientDriver


        This indicates that the installed NVIDIA CUDA driver is older than the CUDA runtime library. This is not a supported configuration. Users should install an updated NVIDIA display driver to allow the application to run.


    .. autoattribute:: cuda.bindings.runtime.cudaError_t.cudaErrorCallRequiresNewerDriver


        This indicates that the API call requires a newer CUDA driver than the one currently installed. Users should install an updated NVIDIA CUDA driver to allow the API call to succeed.


    .. autoattribute:: cuda.bindings.runtime.cudaError_t.cudaErrorInvalidSurface


        This indicates that the surface passed to the API call is not a valid surface.


    .. autoattribute:: cuda.bindings.runtime.cudaError_t.cudaErrorDuplicateVariableName


        This indicates that multiple global or constant variables (across separate CUDA source files in the application) share the same string name.


    .. autoattribute:: cuda.bindings.runtime.cudaError_t.cudaErrorDuplicateTextureName


        This indicates that multiple textures (across separate CUDA source files in the application) share the same string name.


    .. autoattribute:: cuda.bindings.runtime.cudaError_t.cudaErrorDuplicateSurfaceName


        This indicates that multiple surfaces (across separate CUDA source files in the application) share the same string name.


    .. autoattribute:: cuda.bindings.runtime.cudaError_t.cudaErrorDevicesUnavailable


        This indicates that all CUDA devices are busy or unavailable at the current time. Devices are often busy/unavailable due to use of :py:obj:`~.cudaComputeModeProhibited`, :py:obj:`~.cudaComputeModeExclusiveProcess`, or when long running CUDA kernels have filled up the GPU and are blocking new work from starting. They can also be unavailable due to memory constraints on a device that already has active CUDA work being performed.


    .. autoattribute:: cuda.bindings.runtime.cudaError_t.cudaErrorIncompatibleDriverContext


        This indicates that the current context is not compatible with this the CUDA Runtime. This can only occur if you are using CUDA Runtime/Driver interoperability and have created an existing Driver context using the driver API. The Driver context may be incompatible either because the Driver context was created using an older version of the API, because the Runtime API call expects a primary driver context and the Driver context is not primary, or because the Driver context has been destroyed. Please see :py:obj:`~.Interactions`with the CUDA Driver API" for more information.


    .. autoattribute:: cuda.bindings.runtime.cudaError_t.cudaErrorMissingConfiguration


        The device function being invoked (usually via :py:obj:`~.cudaLaunchKernel()`) was not previously configured via the :py:obj:`~.cudaConfigureCall()` function.


    .. autoattribute:: cuda.bindings.runtime.cudaError_t.cudaErrorPriorLaunchFailure


        This indicated that a previous kernel launch failed. This was previously used for device emulation of kernel launches. [Deprecated]


    .. autoattribute:: cuda.bindings.runtime.cudaError_t.cudaErrorLaunchMaxDepthExceeded


        This error indicates that a device runtime grid launch did not occur because the depth of the child grid would exceed the maximum supported number of nested grid launches.


    .. autoattribute:: cuda.bindings.runtime.cudaError_t.cudaErrorLaunchFileScopedTex


        This error indicates that a grid launch did not occur because the kernel uses file-scoped textures which are unsupported by the device runtime. Kernels launched via the device runtime only support textures created with the Texture Object API's.


    .. autoattribute:: cuda.bindings.runtime.cudaError_t.cudaErrorLaunchFileScopedSurf


        This error indicates that a grid launch did not occur because the kernel uses file-scoped surfaces which are unsupported by the device runtime. Kernels launched via the device runtime only support surfaces created with the Surface Object API's.


    .. autoattribute:: cuda.bindings.runtime.cudaError_t.cudaErrorSyncDepthExceeded


        This error indicates that a call to :py:obj:`~.cudaDeviceSynchronize` made from the device runtime failed because the call was made at grid depth greater than than either the default (2 levels of grids) or user specified device limit :py:obj:`~.cudaLimitDevRuntimeSyncDepth`. To be able to synchronize on launched grids at a greater depth successfully, the maximum nested depth at which :py:obj:`~.cudaDeviceSynchronize` will be called must be specified with the :py:obj:`~.cudaLimitDevRuntimeSyncDepth` limit to the :py:obj:`~.cudaDeviceSetLimit` api before the host-side launch of a kernel using the device runtime. Keep in mind that additional levels of sync depth require the runtime to reserve large amounts of device memory that cannot be used for user allocations. Note that :py:obj:`~.cudaDeviceSynchronize` made from device runtime is only supported on devices of compute capability < 9.0.


    .. autoattribute:: cuda.bindings.runtime.cudaError_t.cudaErrorLaunchPendingCountExceeded


        This error indicates that a device runtime grid launch failed because the launch would exceed the limit :py:obj:`~.cudaLimitDevRuntimePendingLaunchCount`. For this launch to proceed successfully, :py:obj:`~.cudaDeviceSetLimit` must be called to set the :py:obj:`~.cudaLimitDevRuntimePendingLaunchCount` to be higher than the upper bound of outstanding launches that can be issued to the device runtime. Keep in mind that raising the limit of pending device runtime launches will require the runtime to reserve device memory that cannot be used for user allocations.


    .. autoattribute:: cuda.bindings.runtime.cudaError_t.cudaErrorInvalidDeviceFunction


        The requested device function does not exist or is not compiled for the proper device architecture.


    .. autoattribute:: cuda.bindings.runtime.cudaError_t.cudaErrorNoDevice


        This indicates that no CUDA-capable devices were detected by the installed CUDA driver.


    .. autoattribute:: cuda.bindings.runtime.cudaError_t.cudaErrorInvalidDevice


        This indicates that the device ordinal supplied by the user does not correspond to a valid CUDA device or that the action requested is invalid for the specified device.


    .. autoattribute:: cuda.bindings.runtime.cudaError_t.cudaErrorDeviceNotLicensed


        This indicates that the device doesn't have a valid Grid License.


    .. autoattribute:: cuda.bindings.runtime.cudaError_t.cudaErrorSoftwareValidityNotEstablished


        By default, the CUDA runtime may perform a minimal set of self-tests, as well as CUDA driver tests, to establish the validity of both. Introduced in CUDA 11.2, this error return indicates that at least one of these tests has failed and the validity of either the runtime or the driver could not be established.


    .. autoattribute:: cuda.bindings.runtime.cudaError_t.cudaErrorStartupFailure


        This indicates an internal startup failure in the CUDA runtime.


    .. autoattribute:: cuda.bindings.runtime.cudaError_t.cudaErrorInvalidKernelImage


        This indicates that the device kernel image is invalid.


    .. autoattribute:: cuda.bindings.runtime.cudaError_t.cudaErrorDeviceUninitialized


        This most frequently indicates that there is no context bound to the current thread. This can also be returned if the context passed to an API call is not a valid handle (such as a context that has had :py:obj:`~.cuCtxDestroy()` invoked on it). This can also be returned if a user mixes different API versions (i.e. 3010 context with 3020 API calls). See :py:obj:`~.cuCtxGetApiVersion()` for more details.


    .. autoattribute:: cuda.bindings.runtime.cudaError_t.cudaErrorMapBufferObjectFailed


        This indicates that the buffer object could not be mapped.


    .. autoattribute:: cuda.bindings.runtime.cudaError_t.cudaErrorUnmapBufferObjectFailed


        This indicates that the buffer object could not be unmapped.


    .. autoattribute:: cuda.bindings.runtime.cudaError_t.cudaErrorArrayIsMapped


        This indicates that the specified array is currently mapped and thus cannot be destroyed.


    .. autoattribute:: cuda.bindings.runtime.cudaError_t.cudaErrorAlreadyMapped


        This indicates that the resource is already mapped.


    .. autoattribute:: cuda.bindings.runtime.cudaError_t.cudaErrorNoKernelImageForDevice


        This indicates that there is no kernel image available that is suitable for the device. This can occur when a user specifies code generation options for a particular CUDA source file that do not include the corresponding device configuration.


    .. autoattribute:: cuda.bindings.runtime.cudaError_t.cudaErrorAlreadyAcquired


        This indicates that a resource has already been acquired.


    .. autoattribute:: cuda.bindings.runtime.cudaError_t.cudaErrorNotMapped


        This indicates that a resource is not mapped.


    .. autoattribute:: cuda.bindings.runtime.cudaError_t.cudaErrorNotMappedAsArray


        This indicates that a mapped resource is not available for access as an array.


    .. autoattribute:: cuda.bindings.runtime.cudaError_t.cudaErrorNotMappedAsPointer


        This indicates that a mapped resource is not available for access as a pointer.


    .. autoattribute:: cuda.bindings.runtime.cudaError_t.cudaErrorECCUncorrectable


        This indicates that an uncorrectable ECC error was detected during execution.


    .. autoattribute:: cuda.bindings.runtime.cudaError_t.cudaErrorUnsupportedLimit


        This indicates that the :py:obj:`~.cudaLimit` passed to the API call is not supported by the active device.


    .. autoattribute:: cuda.bindings.runtime.cudaError_t.cudaErrorDeviceAlreadyInUse


        This indicates that a call tried to access an exclusive-thread device that is already in use by a different thread.


    .. autoattribute:: cuda.bindings.runtime.cudaError_t.cudaErrorPeerAccessUnsupported


        This error indicates that P2P access is not supported across the given devices.


    .. autoattribute:: cuda.bindings.runtime.cudaError_t.cudaErrorInvalidPtx


        A PTX compilation failed. The runtime may fall back to compiling PTX if an application does not contain a suitable binary for the current device.


    .. autoattribute:: cuda.bindings.runtime.cudaError_t.cudaErrorInvalidGraphicsContext


        This indicates an error with the OpenGL or DirectX context.


    .. autoattribute:: cuda.bindings.runtime.cudaError_t.cudaErrorNvlinkUncorrectable


        This indicates that an uncorrectable NVLink error was detected during the execution.


    .. autoattribute:: cuda.bindings.runtime.cudaError_t.cudaErrorJitCompilerNotFound


        This indicates that the PTX JIT compiler library was not found. The JIT Compiler library is used for PTX compilation. The runtime may fall back to compiling PTX if an application does not contain a suitable binary for the current device.


    .. autoattribute:: cuda.bindings.runtime.cudaError_t.cudaErrorUnsupportedPtxVersion


        This indicates that the provided PTX was compiled with an unsupported toolchain. The most common reason for this, is the PTX was generated by a compiler newer than what is supported by the CUDA driver and PTX JIT compiler.


    .. autoattribute:: cuda.bindings.runtime.cudaError_t.cudaErrorJitCompilationDisabled


        This indicates that the JIT compilation was disabled. The JIT compilation compiles PTX. The runtime may fall back to compiling PTX if an application does not contain a suitable binary for the current device.


    .. autoattribute:: cuda.bindings.runtime.cudaError_t.cudaErrorUnsupportedExecAffinity


        This indicates that the provided execution affinity is not supported by the device.


    .. autoattribute:: cuda.bindings.runtime.cudaError_t.cudaErrorUnsupportedDevSideSync


        This indicates that the code to be compiled by the PTX JIT contains unsupported call to cudaDeviceSynchronize.


    .. autoattribute:: cuda.bindings.runtime.cudaError_t.cudaErrorContained


        This indicates that an exception occurred on the device that is now contained by the GPU's error containment capability. Common causes are - a. Certain types of invalid accesses of peer GPU memory over nvlink b. Certain classes of hardware errors This leaves the process in an inconsistent state and any further CUDA work will return the same error. To continue using CUDA, the process must be terminated and relaunched.


    .. autoattribute:: cuda.bindings.runtime.cudaError_t.cudaErrorInvalidSource


        This indicates that the device kernel source is invalid.


    .. autoattribute:: cuda.bindings.runtime.cudaError_t.cudaErrorFileNotFound


        This indicates that the file specified was not found.


    .. autoattribute:: cuda.bindings.runtime.cudaError_t.cudaErrorSharedObjectSymbolNotFound


        This indicates that a link to a shared object failed to resolve.


    .. autoattribute:: cuda.bindings.runtime.cudaError_t.cudaErrorSharedObjectInitFailed


        This indicates that initialization of a shared object failed.


    .. autoattribute:: cuda.bindings.runtime.cudaError_t.cudaErrorOperatingSystem


        This error indicates that an OS call failed.


    .. autoattribute:: cuda.bindings.runtime.cudaError_t.cudaErrorInvalidResourceHandle


        This indicates that a resource handle passed to the API call was not valid. Resource handles are opaque types like :py:obj:`~.cudaStream_t` and :py:obj:`~.cudaEvent_t`.


    .. autoattribute:: cuda.bindings.runtime.cudaError_t.cudaErrorIllegalState


        This indicates that a resource required by the API call is not in a valid state to perform the requested operation.


    .. autoattribute:: cuda.bindings.runtime.cudaError_t.cudaErrorLossyQuery


        This indicates an attempt was made to introspect an object in a way that would discard semantically important information. This is either due to the object using funtionality newer than the API version used to introspect it or omission of optional return arguments.


    .. autoattribute:: cuda.bindings.runtime.cudaError_t.cudaErrorSymbolNotFound


        This indicates that a named symbol was not found. Examples of symbols are global/constant variable names, driver function names, texture names, and surface names.


    .. autoattribute:: cuda.bindings.runtime.cudaError_t.cudaErrorNotReady


        This indicates that asynchronous operations issued previously have not completed yet. This result is not actually an error, but must be indicated differently than :py:obj:`~.cudaSuccess` (which indicates completion). Calls that may return this value include :py:obj:`~.cudaEventQuery()` and :py:obj:`~.cudaStreamQuery()`.


    .. autoattribute:: cuda.bindings.runtime.cudaError_t.cudaErrorIllegalAddress


        The device encountered a load or store instruction on an invalid memory address. This leaves the process in an inconsistent state and any further CUDA work will return the same error. To continue using CUDA, the process must be terminated and relaunched.


    .. autoattribute:: cuda.bindings.runtime.cudaError_t.cudaErrorLaunchOutOfResources


        This indicates that a launch did not occur because it did not have appropriate resources. Although this error is similar to :py:obj:`~.cudaErrorInvalidConfiguration`, this error usually indicates that the user has attempted to pass too many arguments to the device kernel, or the kernel launch specifies too many threads for the kernel's register count.


    .. autoattribute:: cuda.bindings.runtime.cudaError_t.cudaErrorLaunchTimeout


        This indicates that the device kernel took too long to execute. This can only occur if timeouts are enabled - see the device attribute :py:obj:`~.cudaDevAttrKernelExecTimeout` for more information. This leaves the process in an inconsistent state and any further CUDA work will return the same error. To continue using CUDA, the process must be terminated and relaunched.


    .. autoattribute:: cuda.bindings.runtime.cudaError_t.cudaErrorLaunchIncompatibleTexturing


        This error indicates a kernel launch that uses an incompatible texturing mode.


    .. autoattribute:: cuda.bindings.runtime.cudaError_t.cudaErrorPeerAccessAlreadyEnabled


        This error indicates that a call to :py:obj:`~.cudaDeviceEnablePeerAccess()` is trying to re-enable peer addressing on from a context which has already had peer addressing enabled.


    .. autoattribute:: cuda.bindings.runtime.cudaError_t.cudaErrorPeerAccessNotEnabled


        This error indicates that :py:obj:`~.cudaDeviceDisablePeerAccess()` is trying to disable peer addressing which has not been enabled yet via :py:obj:`~.cudaDeviceEnablePeerAccess()`.


    .. autoattribute:: cuda.bindings.runtime.cudaError_t.cudaErrorSetOnActiveProcess


        This indicates that the user has called :py:obj:`~.cudaSetValidDevices()`, :py:obj:`~.cudaSetDeviceFlags()`, :py:obj:`~.cudaD3D9SetDirect3DDevice()`, :py:obj:`~.cudaD3D10SetDirect3DDevice`, :py:obj:`~.cudaD3D11SetDirect3DDevice()`, or :py:obj:`~.cudaVDPAUSetVDPAUDevice()` after initializing the CUDA runtime by calling non-device management operations (allocating memory and launching kernels are examples of non-device management operations). This error can also be returned if using runtime/driver interoperability and there is an existing :py:obj:`~.CUcontext` active on the host thread.


    .. autoattribute:: cuda.bindings.runtime.cudaError_t.cudaErrorContextIsDestroyed


        This error indicates that the context current to the calling thread has been destroyed using :py:obj:`~.cuCtxDestroy`, or is a primary context which has not yet been initialized.


    .. autoattribute:: cuda.bindings.runtime.cudaError_t.cudaErrorAssert


        An assert triggered in device code during kernel execution. The device cannot be used again. All existing allocations are invalid. To continue using CUDA, the process must be terminated and relaunched.


    .. autoattribute:: cuda.bindings.runtime.cudaError_t.cudaErrorTooManyPeers


        This error indicates that the hardware resources required to enable peer access have been exhausted for one or more of the devices passed to :py:obj:`~.cudaEnablePeerAccess()`.


    .. autoattribute:: cuda.bindings.runtime.cudaError_t.cudaErrorHostMemoryAlreadyRegistered


        This error indicates that the memory range passed to :py:obj:`~.cudaHostRegister()` has already been registered.


    .. autoattribute:: cuda.bindings.runtime.cudaError_t.cudaErrorHostMemoryNotRegistered


        This error indicates that the pointer passed to :py:obj:`~.cudaHostUnregister()` does not correspond to any currently registered memory region.


    .. autoattribute:: cuda.bindings.runtime.cudaError_t.cudaErrorHardwareStackError


        Device encountered an error in the call stack during kernel execution, possibly due to stack corruption or exceeding the stack size limit. This leaves the process in an inconsistent state and any further CUDA work will return the same error. To continue using CUDA, the process must be terminated and relaunched.


    .. autoattribute:: cuda.bindings.runtime.cudaError_t.cudaErrorIllegalInstruction


        The device encountered an illegal instruction during kernel execution This leaves the process in an inconsistent state and any further CUDA work will return the same error. To continue using CUDA, the process must be terminated and relaunched.


    .. autoattribute:: cuda.bindings.runtime.cudaError_t.cudaErrorMisalignedAddress


        The device encountered a load or store instruction on a memory address which is not aligned. This leaves the process in an inconsistent state and any further CUDA work will return the same error. To continue using CUDA, the process must be terminated and relaunched.


    .. autoattribute:: cuda.bindings.runtime.cudaError_t.cudaErrorInvalidAddressSpace


        While executing a kernel, the device encountered an instruction which can only operate on memory locations in certain address spaces (global, shared, or local), but was supplied a memory address not belonging to an allowed address space. This leaves the process in an inconsistent state and any further CUDA work will return the same error. To continue using CUDA, the process must be terminated and relaunched.


    .. autoattribute:: cuda.bindings.runtime.cudaError_t.cudaErrorInvalidPc


        The device encountered an invalid program counter. This leaves the process in an inconsistent state and any further CUDA work will return the same error. To continue using CUDA, the process must be terminated and relaunched.


    .. autoattribute:: cuda.bindings.runtime.cudaError_t.cudaErrorLaunchFailure


        An exception occurred on the device while executing a kernel. Common causes include dereferencing an invalid device pointer and accessing out of bounds shared memory. Less common cases can be system specific - more information about these cases can be found in the system specific user guide. This leaves the process in an inconsistent state and any further CUDA work will return the same error. To continue using CUDA, the process must be terminated and relaunched.


    .. autoattribute:: cuda.bindings.runtime.cudaError_t.cudaErrorCooperativeLaunchTooLarge


        This error indicates that the number of blocks launched per grid for a kernel that was launched via either :py:obj:`~.cudaLaunchCooperativeKernel` exceeds the maximum number of blocks as allowed by :py:obj:`~.cudaOccupancyMaxActiveBlocksPerMultiprocessor` or :py:obj:`~.cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags` times the number of multiprocessors as specified by the device attribute :py:obj:`~.cudaDevAttrMultiProcessorCount`.


    .. autoattribute:: cuda.bindings.runtime.cudaError_t.cudaErrorTensorMemoryLeak


        An exception occurred on the device while exiting a kernel using tensor memory: the tensor memory was not completely deallocated. This leaves the process in an inconsistent state and any further CUDA work will return the same error. To continue using CUDA, the process must be terminated and relaunched.


    .. autoattribute:: cuda.bindings.runtime.cudaError_t.cudaErrorNotPermitted


        This error indicates the attempted operation is not permitted.


    .. autoattribute:: cuda.bindings.runtime.cudaError_t.cudaErrorNotSupported


        This error indicates the attempted operation is not supported on the current system or device.


    .. autoattribute:: cuda.bindings.runtime.cudaError_t.cudaErrorSystemNotReady


        This error indicates that the system is not yet ready to start any CUDA work. To continue using CUDA, verify the system configuration is in a valid state and all required driver daemons are actively running. More information about this error can be found in the system specific user guide.


    .. autoattribute:: cuda.bindings.runtime.cudaError_t.cudaErrorSystemDriverMismatch


        This error indicates that there is a mismatch between the versions of the display driver and the CUDA driver. Refer to the compatibility documentation for supported versions.


    .. autoattribute:: cuda.bindings.runtime.cudaError_t.cudaErrorCompatNotSupportedOnDevice


        This error indicates that the system was upgraded to run with forward compatibility but the visible hardware detected by CUDA does not support this configuration. Refer to the compatibility documentation for the supported hardware matrix or ensure that only supported hardware is visible during initialization via the CUDA_VISIBLE_DEVICES environment variable.


    .. autoattribute:: cuda.bindings.runtime.cudaError_t.cudaErrorMpsConnectionFailed


        This error indicates that the MPS client failed to connect to the MPS control daemon or the MPS server.


    .. autoattribute:: cuda.bindings.runtime.cudaError_t.cudaErrorMpsRpcFailure


        This error indicates that the remote procedural call between the MPS server and the MPS client failed.


    .. autoattribute:: cuda.bindings.runtime.cudaError_t.cudaErrorMpsServerNotReady


        This error indicates that the MPS server is not ready to accept new MPS client requests. This error can be returned when the MPS server is in the process of recovering from a fatal failure.


    .. autoattribute:: cuda.bindings.runtime.cudaError_t.cudaErrorMpsMaxClientsReached


        This error indicates that the hardware resources required to create MPS client have been exhausted.


    .. autoattribute:: cuda.bindings.runtime.cudaError_t.cudaErrorMpsMaxConnectionsReached


        This error indicates the the hardware resources required to device connections have been exhausted.


    .. autoattribute:: cuda.bindings.runtime.cudaError_t.cudaErrorMpsClientTerminated


        This error indicates that the MPS client has been terminated by the server. To continue using CUDA, the process must be terminated and relaunched.


    .. autoattribute:: cuda.bindings.runtime.cudaError_t.cudaErrorCdpNotSupported


        This error indicates, that the program is using CUDA Dynamic Parallelism, but the current configuration, like MPS, does not support it.


    .. autoattribute:: cuda.bindings.runtime.cudaError_t.cudaErrorCdpVersionMismatch


        This error indicates, that the program contains an unsupported interaction between different versions of CUDA Dynamic Parallelism.


    .. autoattribute:: cuda.bindings.runtime.cudaError_t.cudaErrorStreamCaptureUnsupported


        The operation is not permitted when the stream is capturing.


    .. autoattribute:: cuda.bindings.runtime.cudaError_t.cudaErrorStreamCaptureInvalidated


        The current capture sequence on the stream has been invalidated due to a previous error.


    .. autoattribute:: cuda.bindings.runtime.cudaError_t.cudaErrorStreamCaptureMerge


        The operation would have resulted in a merge of two independent capture sequences.


    .. autoattribute:: cuda.bindings.runtime.cudaError_t.cudaErrorStreamCaptureUnmatched


        The capture was not initiated in this stream.


    .. autoattribute:: cuda.bindings.runtime.cudaError_t.cudaErrorStreamCaptureUnjoined


        The capture sequence contains a fork that was not joined to the primary stream.


    .. autoattribute:: cuda.bindings.runtime.cudaError_t.cudaErrorStreamCaptureIsolation


        A dependency would have been created which crosses the capture sequence boundary. Only implicit in-stream ordering dependencies are allowed to cross the boundary.


    .. autoattribute:: cuda.bindings.runtime.cudaError_t.cudaErrorStreamCaptureImplicit


        The operation would have resulted in a disallowed implicit dependency on a current capture sequence from cudaStreamLegacy.


    .. autoattribute:: cuda.bindings.runtime.cudaError_t.cudaErrorCapturedEvent


        The operation is not permitted on an event which was last recorded in a capturing stream.


    .. autoattribute:: cuda.bindings.runtime.cudaError_t.cudaErrorStreamCaptureWrongThread


        A stream capture sequence not initiated with the :py:obj:`~.cudaStreamCaptureModeRelaxed` argument to :py:obj:`~.cudaStreamBeginCapture` was passed to :py:obj:`~.cudaStreamEndCapture` in a different thread.


    .. autoattribute:: cuda.bindings.runtime.cudaError_t.cudaErrorTimeout


        This indicates that the wait operation has timed out.


    .. autoattribute:: cuda.bindings.runtime.cudaError_t.cudaErrorGraphExecUpdateFailure


        This error indicates that the graph update was not performed because it included changes which violated constraints specific to instantiated graph update.


    .. autoattribute:: cuda.bindings.runtime.cudaError_t.cudaErrorExternalDevice


        This indicates that an async error has occurred in a device outside of CUDA. If CUDA was waiting for an external device's signal before consuming shared data, the external device signaled an error indicating that the data is not valid for consumption. This leaves the process in an inconsistent state and any further CUDA work will return the same error. To continue using CUDA, the process must be terminated and relaunched.


    .. autoattribute:: cuda.bindings.runtime.cudaError_t.cudaErrorInvalidClusterSize


        This indicates that a kernel launch error has occurred due to cluster misconfiguration.


    .. autoattribute:: cuda.bindings.runtime.cudaError_t.cudaErrorFunctionNotLoaded


        Indiciates a function handle is not loaded when calling an API that requires a loaded function.


    .. autoattribute:: cuda.bindings.runtime.cudaError_t.cudaErrorInvalidResourceType


        This error indicates one or more resources passed in are not valid resource types for the operation.


    .. autoattribute:: cuda.bindings.runtime.cudaError_t.cudaErrorInvalidResourceConfiguration


        This error indicates one or more resources are insufficient or non-applicable for the operation.


    .. autoattribute:: cuda.bindings.runtime.cudaError_t.cudaErrorStreamDetached


        This error indicates that the requested operation is not permitted because the stream is in a detached state. This can occur if the green context associated with the stream has been destroyed, limiting the stream's operational capabilities.


    .. autoattribute:: cuda.bindings.runtime.cudaError_t.cudaErrorUnknown


        This indicates that an unknown internal error has occurred.


    .. autoattribute:: cuda.bindings.runtime.cudaError_t.cudaErrorApiFailureBase

.. autoclass:: cuda.bindings.runtime.cudaChannelFormatKind

    .. autoattribute:: cuda.bindings.runtime.cudaChannelFormatKind.cudaChannelFormatKindSigned


        Signed channel format


    .. autoattribute:: cuda.bindings.runtime.cudaChannelFormatKind.cudaChannelFormatKindUnsigned


        Unsigned channel format


    .. autoattribute:: cuda.bindings.runtime.cudaChannelFormatKind.cudaChannelFormatKindFloat


        Float channel format


    .. autoattribute:: cuda.bindings.runtime.cudaChannelFormatKind.cudaChannelFormatKindNone


        No channel format


    .. autoattribute:: cuda.bindings.runtime.cudaChannelFormatKind.cudaChannelFormatKindNV12


        Unsigned 8-bit integers, planar 4:2:0 YUV format


    .. autoattribute:: cuda.bindings.runtime.cudaChannelFormatKind.cudaChannelFormatKindUnsignedNormalized8X1


        1 channel unsigned 8-bit normalized integer


    .. autoattribute:: cuda.bindings.runtime.cudaChannelFormatKind.cudaChannelFormatKindUnsignedNormalized8X2


        2 channel unsigned 8-bit normalized integer


    .. autoattribute:: cuda.bindings.runtime.cudaChannelFormatKind.cudaChannelFormatKindUnsignedNormalized8X4


        4 channel unsigned 8-bit normalized integer


    .. autoattribute:: cuda.bindings.runtime.cudaChannelFormatKind.cudaChannelFormatKindUnsignedNormalized16X1


        1 channel unsigned 16-bit normalized integer


    .. autoattribute:: cuda.bindings.runtime.cudaChannelFormatKind.cudaChannelFormatKindUnsignedNormalized16X2


        2 channel unsigned 16-bit normalized integer


    .. autoattribute:: cuda.bindings.runtime.cudaChannelFormatKind.cudaChannelFormatKindUnsignedNormalized16X4


        4 channel unsigned 16-bit normalized integer


    .. autoattribute:: cuda.bindings.runtime.cudaChannelFormatKind.cudaChannelFormatKindSignedNormalized8X1


        1 channel signed 8-bit normalized integer


    .. autoattribute:: cuda.bindings.runtime.cudaChannelFormatKind.cudaChannelFormatKindSignedNormalized8X2


        2 channel signed 8-bit normalized integer


    .. autoattribute:: cuda.bindings.runtime.cudaChannelFormatKind.cudaChannelFormatKindSignedNormalized8X4


        4 channel signed 8-bit normalized integer


    .. autoattribute:: cuda.bindings.runtime.cudaChannelFormatKind.cudaChannelFormatKindSignedNormalized16X1


        1 channel signed 16-bit normalized integer


    .. autoattribute:: cuda.bindings.runtime.cudaChannelFormatKind.cudaChannelFormatKindSignedNormalized16X2


        2 channel signed 16-bit normalized integer


    .. autoattribute:: cuda.bindings.runtime.cudaChannelFormatKind.cudaChannelFormatKindSignedNormalized16X4


        4 channel signed 16-bit normalized integer


    .. autoattribute:: cuda.bindings.runtime.cudaChannelFormatKind.cudaChannelFormatKindUnsignedBlockCompressed1


        4 channel unsigned normalized block-compressed (BC1 compression) format


    .. autoattribute:: cuda.bindings.runtime.cudaChannelFormatKind.cudaChannelFormatKindUnsignedBlockCompressed1SRGB


        4 channel unsigned normalized block-compressed (BC1 compression) format with sRGB encoding


    .. autoattribute:: cuda.bindings.runtime.cudaChannelFormatKind.cudaChannelFormatKindUnsignedBlockCompressed2


        4 channel unsigned normalized block-compressed (BC2 compression) format


    .. autoattribute:: cuda.bindings.runtime.cudaChannelFormatKind.cudaChannelFormatKindUnsignedBlockCompressed2SRGB


        4 channel unsigned normalized block-compressed (BC2 compression) format with sRGB encoding


    .. autoattribute:: cuda.bindings.runtime.cudaChannelFormatKind.cudaChannelFormatKindUnsignedBlockCompressed3


        4 channel unsigned normalized block-compressed (BC3 compression) format


    .. autoattribute:: cuda.bindings.runtime.cudaChannelFormatKind.cudaChannelFormatKindUnsignedBlockCompressed3SRGB


        4 channel unsigned normalized block-compressed (BC3 compression) format with sRGB encoding


    .. autoattribute:: cuda.bindings.runtime.cudaChannelFormatKind.cudaChannelFormatKindUnsignedBlockCompressed4


        1 channel unsigned normalized block-compressed (BC4 compression) format


    .. autoattribute:: cuda.bindings.runtime.cudaChannelFormatKind.cudaChannelFormatKindSignedBlockCompressed4


        1 channel signed normalized block-compressed (BC4 compression) format


    .. autoattribute:: cuda.bindings.runtime.cudaChannelFormatKind.cudaChannelFormatKindUnsignedBlockCompressed5


        2 channel unsigned normalized block-compressed (BC5 compression) format


    .. autoattribute:: cuda.bindings.runtime.cudaChannelFormatKind.cudaChannelFormatKindSignedBlockCompressed5


        2 channel signed normalized block-compressed (BC5 compression) format


    .. autoattribute:: cuda.bindings.runtime.cudaChannelFormatKind.cudaChannelFormatKindUnsignedBlockCompressed6H


        3 channel unsigned half-float block-compressed (BC6H compression) format


    .. autoattribute:: cuda.bindings.runtime.cudaChannelFormatKind.cudaChannelFormatKindSignedBlockCompressed6H


        3 channel signed half-float block-compressed (BC6H compression) format


    .. autoattribute:: cuda.bindings.runtime.cudaChannelFormatKind.cudaChannelFormatKindUnsignedBlockCompressed7


        4 channel unsigned normalized block-compressed (BC7 compression) format


    .. autoattribute:: cuda.bindings.runtime.cudaChannelFormatKind.cudaChannelFormatKindUnsignedBlockCompressed7SRGB


        4 channel unsigned normalized block-compressed (BC7 compression) format with sRGB encoding


    .. autoattribute:: cuda.bindings.runtime.cudaChannelFormatKind.cudaChannelFormatKindUnsignedNormalized1010102


        4 channel unsigned normalized (10-bit, 10-bit, 10-bit, 2-bit) format

.. autoclass:: cuda.bindings.runtime.cudaMemoryType

    .. autoattribute:: cuda.bindings.runtime.cudaMemoryType.cudaMemoryTypeUnregistered


        Unregistered memory


    .. autoattribute:: cuda.bindings.runtime.cudaMemoryType.cudaMemoryTypeHost


        Host memory


    .. autoattribute:: cuda.bindings.runtime.cudaMemoryType.cudaMemoryTypeDevice


        Device memory


    .. autoattribute:: cuda.bindings.runtime.cudaMemoryType.cudaMemoryTypeManaged


        Managed memory

.. autoclass:: cuda.bindings.runtime.cudaMemcpyKind

    .. autoattribute:: cuda.bindings.runtime.cudaMemcpyKind.cudaMemcpyHostToHost


        Host -> Host


    .. autoattribute:: cuda.bindings.runtime.cudaMemcpyKind.cudaMemcpyHostToDevice


        Host -> Device


    .. autoattribute:: cuda.bindings.runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost


        Device -> Host


    .. autoattribute:: cuda.bindings.runtime.cudaMemcpyKind.cudaMemcpyDeviceToDevice


        Device -> Device


    .. autoattribute:: cuda.bindings.runtime.cudaMemcpyKind.cudaMemcpyDefault


        Direction of the transfer is inferred from the pointer values. Requires unified virtual addressing

.. autoclass:: cuda.bindings.runtime.cudaAccessProperty

    .. autoattribute:: cuda.bindings.runtime.cudaAccessProperty.cudaAccessPropertyNormal


        Normal cache persistence.


    .. autoattribute:: cuda.bindings.runtime.cudaAccessProperty.cudaAccessPropertyStreaming


        Streaming access is less likely to persit from cache.


    .. autoattribute:: cuda.bindings.runtime.cudaAccessProperty.cudaAccessPropertyPersisting


        Persisting access is more likely to persist in cache.

.. autoclass:: cuda.bindings.runtime.cudaStreamCaptureStatus

    .. autoattribute:: cuda.bindings.runtime.cudaStreamCaptureStatus.cudaStreamCaptureStatusNone


        Stream is not capturing


    .. autoattribute:: cuda.bindings.runtime.cudaStreamCaptureStatus.cudaStreamCaptureStatusActive


        Stream is actively capturing


    .. autoattribute:: cuda.bindings.runtime.cudaStreamCaptureStatus.cudaStreamCaptureStatusInvalidated


        Stream is part of a capture sequence that has been invalidated, but not terminated

.. autoclass:: cuda.bindings.runtime.cudaStreamCaptureMode

    .. autoattribute:: cuda.bindings.runtime.cudaStreamCaptureMode.cudaStreamCaptureModeGlobal


    .. autoattribute:: cuda.bindings.runtime.cudaStreamCaptureMode.cudaStreamCaptureModeThreadLocal


    .. autoattribute:: cuda.bindings.runtime.cudaStreamCaptureMode.cudaStreamCaptureModeRelaxed

.. autoclass:: cuda.bindings.runtime.cudaSynchronizationPolicy

    .. autoattribute:: cuda.bindings.runtime.cudaSynchronizationPolicy.cudaSyncPolicyAuto


    .. autoattribute:: cuda.bindings.runtime.cudaSynchronizationPolicy.cudaSyncPolicySpin


    .. autoattribute:: cuda.bindings.runtime.cudaSynchronizationPolicy.cudaSyncPolicyYield


    .. autoattribute:: cuda.bindings.runtime.cudaSynchronizationPolicy.cudaSyncPolicyBlockingSync

.. autoclass:: cuda.bindings.runtime.cudaClusterSchedulingPolicy

    .. autoattribute:: cuda.bindings.runtime.cudaClusterSchedulingPolicy.cudaClusterSchedulingPolicyDefault


        the default policy


    .. autoattribute:: cuda.bindings.runtime.cudaClusterSchedulingPolicy.cudaClusterSchedulingPolicySpread


        spread the blocks within a cluster to the SMs


    .. autoattribute:: cuda.bindings.runtime.cudaClusterSchedulingPolicy.cudaClusterSchedulingPolicyLoadBalancing


        allow the hardware to load-balance the blocks in a cluster to the SMs

.. autoclass:: cuda.bindings.runtime.cudaStreamUpdateCaptureDependenciesFlags

    .. autoattribute:: cuda.bindings.runtime.cudaStreamUpdateCaptureDependenciesFlags.cudaStreamAddCaptureDependencies


        Add new nodes to the dependency set


    .. autoattribute:: cuda.bindings.runtime.cudaStreamUpdateCaptureDependenciesFlags.cudaStreamSetCaptureDependencies


        Replace the dependency set with the new nodes

.. autoclass:: cuda.bindings.runtime.cudaUserObjectFlags

    .. autoattribute:: cuda.bindings.runtime.cudaUserObjectFlags.cudaUserObjectNoDestructorSync


        Indicates the destructor execution is not synchronized by any CUDA handle.

.. autoclass:: cuda.bindings.runtime.cudaUserObjectRetainFlags

    .. autoattribute:: cuda.bindings.runtime.cudaUserObjectRetainFlags.cudaGraphUserObjectMove


        Transfer references from the caller rather than creating new references.

.. autoclass:: cuda.bindings.runtime.cudaGraphicsRegisterFlags

    .. autoattribute:: cuda.bindings.runtime.cudaGraphicsRegisterFlags.cudaGraphicsRegisterFlagsNone


        Default


    .. autoattribute:: cuda.bindings.runtime.cudaGraphicsRegisterFlags.cudaGraphicsRegisterFlagsReadOnly


        CUDA will not write to this resource


    .. autoattribute:: cuda.bindings.runtime.cudaGraphicsRegisterFlags.cudaGraphicsRegisterFlagsWriteDiscard


        CUDA will only write to and will not read from this resource


    .. autoattribute:: cuda.bindings.runtime.cudaGraphicsRegisterFlags.cudaGraphicsRegisterFlagsSurfaceLoadStore


        CUDA will bind this resource to a surface reference


    .. autoattribute:: cuda.bindings.runtime.cudaGraphicsRegisterFlags.cudaGraphicsRegisterFlagsTextureGather


        CUDA will perform texture gather operations on this resource

.. autoclass:: cuda.bindings.runtime.cudaGraphicsMapFlags

    .. autoattribute:: cuda.bindings.runtime.cudaGraphicsMapFlags.cudaGraphicsMapFlagsNone


        Default; Assume resource can be read/written


    .. autoattribute:: cuda.bindings.runtime.cudaGraphicsMapFlags.cudaGraphicsMapFlagsReadOnly


        CUDA will not write to this resource


    .. autoattribute:: cuda.bindings.runtime.cudaGraphicsMapFlags.cudaGraphicsMapFlagsWriteDiscard


        CUDA will only write to and will not read from this resource

.. autoclass:: cuda.bindings.runtime.cudaGraphicsCubeFace

    .. autoattribute:: cuda.bindings.runtime.cudaGraphicsCubeFace.cudaGraphicsCubeFacePositiveX


        Positive X face of cubemap


    .. autoattribute:: cuda.bindings.runtime.cudaGraphicsCubeFace.cudaGraphicsCubeFaceNegativeX


        Negative X face of cubemap


    .. autoattribute:: cuda.bindings.runtime.cudaGraphicsCubeFace.cudaGraphicsCubeFacePositiveY


        Positive Y face of cubemap


    .. autoattribute:: cuda.bindings.runtime.cudaGraphicsCubeFace.cudaGraphicsCubeFaceNegativeY


        Negative Y face of cubemap


    .. autoattribute:: cuda.bindings.runtime.cudaGraphicsCubeFace.cudaGraphicsCubeFacePositiveZ


        Positive Z face of cubemap


    .. autoattribute:: cuda.bindings.runtime.cudaGraphicsCubeFace.cudaGraphicsCubeFaceNegativeZ


        Negative Z face of cubemap

.. autoclass:: cuda.bindings.runtime.cudaResourceType

    .. autoattribute:: cuda.bindings.runtime.cudaResourceType.cudaResourceTypeArray


        Array resource


    .. autoattribute:: cuda.bindings.runtime.cudaResourceType.cudaResourceTypeMipmappedArray


        Mipmapped array resource


    .. autoattribute:: cuda.bindings.runtime.cudaResourceType.cudaResourceTypeLinear


        Linear resource


    .. autoattribute:: cuda.bindings.runtime.cudaResourceType.cudaResourceTypePitch2D


        Pitch 2D resource

.. autoclass:: cuda.bindings.runtime.cudaResourceViewFormat

    .. autoattribute:: cuda.bindings.runtime.cudaResourceViewFormat.cudaResViewFormatNone


        No resource view format (use underlying resource format)


    .. autoattribute:: cuda.bindings.runtime.cudaResourceViewFormat.cudaResViewFormatUnsignedChar1


        1 channel unsigned 8-bit integers


    .. autoattribute:: cuda.bindings.runtime.cudaResourceViewFormat.cudaResViewFormatUnsignedChar2


        2 channel unsigned 8-bit integers


    .. autoattribute:: cuda.bindings.runtime.cudaResourceViewFormat.cudaResViewFormatUnsignedChar4


        4 channel unsigned 8-bit integers


    .. autoattribute:: cuda.bindings.runtime.cudaResourceViewFormat.cudaResViewFormatSignedChar1


        1 channel signed 8-bit integers


    .. autoattribute:: cuda.bindings.runtime.cudaResourceViewFormat.cudaResViewFormatSignedChar2


        2 channel signed 8-bit integers


    .. autoattribute:: cuda.bindings.runtime.cudaResourceViewFormat.cudaResViewFormatSignedChar4


        4 channel signed 8-bit integers


    .. autoattribute:: cuda.bindings.runtime.cudaResourceViewFormat.cudaResViewFormatUnsignedShort1


        1 channel unsigned 16-bit integers


    .. autoattribute:: cuda.bindings.runtime.cudaResourceViewFormat.cudaResViewFormatUnsignedShort2


        2 channel unsigned 16-bit integers


    .. autoattribute:: cuda.bindings.runtime.cudaResourceViewFormat.cudaResViewFormatUnsignedShort4


        4 channel unsigned 16-bit integers


    .. autoattribute:: cuda.bindings.runtime.cudaResourceViewFormat.cudaResViewFormatSignedShort1


        1 channel signed 16-bit integers


    .. autoattribute:: cuda.bindings.runtime.cudaResourceViewFormat.cudaResViewFormatSignedShort2


        2 channel signed 16-bit integers


    .. autoattribute:: cuda.bindings.runtime.cudaResourceViewFormat.cudaResViewFormatSignedShort4


        4 channel signed 16-bit integers


    .. autoattribute:: cuda.bindings.runtime.cudaResourceViewFormat.cudaResViewFormatUnsignedInt1


        1 channel unsigned 32-bit integers


    .. autoattribute:: cuda.bindings.runtime.cudaResourceViewFormat.cudaResViewFormatUnsignedInt2


        2 channel unsigned 32-bit integers


    .. autoattribute:: cuda.bindings.runtime.cudaResourceViewFormat.cudaResViewFormatUnsignedInt4


        4 channel unsigned 32-bit integers


    .. autoattribute:: cuda.bindings.runtime.cudaResourceViewFormat.cudaResViewFormatSignedInt1


        1 channel signed 32-bit integers


    .. autoattribute:: cuda.bindings.runtime.cudaResourceViewFormat.cudaResViewFormatSignedInt2


        2 channel signed 32-bit integers


    .. autoattribute:: cuda.bindings.runtime.cudaResourceViewFormat.cudaResViewFormatSignedInt4


        4 channel signed 32-bit integers


    .. autoattribute:: cuda.bindings.runtime.cudaResourceViewFormat.cudaResViewFormatHalf1


        1 channel 16-bit floating point


    .. autoattribute:: cuda.bindings.runtime.cudaResourceViewFormat.cudaResViewFormatHalf2


        2 channel 16-bit floating point


    .. autoattribute:: cuda.bindings.runtime.cudaResourceViewFormat.cudaResViewFormatHalf4


        4 channel 16-bit floating point


    .. autoattribute:: cuda.bindings.runtime.cudaResourceViewFormat.cudaResViewFormatFloat1


        1 channel 32-bit floating point


    .. autoattribute:: cuda.bindings.runtime.cudaResourceViewFormat.cudaResViewFormatFloat2


        2 channel 32-bit floating point


    .. autoattribute:: cuda.bindings.runtime.cudaResourceViewFormat.cudaResViewFormatFloat4


        4 channel 32-bit floating point


    .. autoattribute:: cuda.bindings.runtime.cudaResourceViewFormat.cudaResViewFormatUnsignedBlockCompressed1


        Block compressed 1


    .. autoattribute:: cuda.bindings.runtime.cudaResourceViewFormat.cudaResViewFormatUnsignedBlockCompressed2


        Block compressed 2


    .. autoattribute:: cuda.bindings.runtime.cudaResourceViewFormat.cudaResViewFormatUnsignedBlockCompressed3


        Block compressed 3


    .. autoattribute:: cuda.bindings.runtime.cudaResourceViewFormat.cudaResViewFormatUnsignedBlockCompressed4


        Block compressed 4 unsigned


    .. autoattribute:: cuda.bindings.runtime.cudaResourceViewFormat.cudaResViewFormatSignedBlockCompressed4


        Block compressed 4 signed


    .. autoattribute:: cuda.bindings.runtime.cudaResourceViewFormat.cudaResViewFormatUnsignedBlockCompressed5


        Block compressed 5 unsigned


    .. autoattribute:: cuda.bindings.runtime.cudaResourceViewFormat.cudaResViewFormatSignedBlockCompressed5


        Block compressed 5 signed


    .. autoattribute:: cuda.bindings.runtime.cudaResourceViewFormat.cudaResViewFormatUnsignedBlockCompressed6H


        Block compressed 6 unsigned half-float


    .. autoattribute:: cuda.bindings.runtime.cudaResourceViewFormat.cudaResViewFormatSignedBlockCompressed6H


        Block compressed 6 signed half-float


    .. autoattribute:: cuda.bindings.runtime.cudaResourceViewFormat.cudaResViewFormatUnsignedBlockCompressed7


        Block compressed 7

.. autoclass:: cuda.bindings.runtime.cudaFuncAttribute

    .. autoattribute:: cuda.bindings.runtime.cudaFuncAttribute.cudaFuncAttributeMaxDynamicSharedMemorySize


        Maximum dynamic shared memory size


    .. autoattribute:: cuda.bindings.runtime.cudaFuncAttribute.cudaFuncAttributePreferredSharedMemoryCarveout


        Preferred shared memory-L1 cache split


    .. autoattribute:: cuda.bindings.runtime.cudaFuncAttribute.cudaFuncAttributeClusterDimMustBeSet


        Indicator to enforce valid cluster dimension specification on kernel launch


    .. autoattribute:: cuda.bindings.runtime.cudaFuncAttribute.cudaFuncAttributeRequiredClusterWidth


        Required cluster width


    .. autoattribute:: cuda.bindings.runtime.cudaFuncAttribute.cudaFuncAttributeRequiredClusterHeight


        Required cluster height


    .. autoattribute:: cuda.bindings.runtime.cudaFuncAttribute.cudaFuncAttributeRequiredClusterDepth


        Required cluster depth


    .. autoattribute:: cuda.bindings.runtime.cudaFuncAttribute.cudaFuncAttributeNonPortableClusterSizeAllowed


        Whether non-portable cluster scheduling policy is supported


    .. autoattribute:: cuda.bindings.runtime.cudaFuncAttribute.cudaFuncAttributeClusterSchedulingPolicyPreference


        Required cluster scheduling policy preference


    .. autoattribute:: cuda.bindings.runtime.cudaFuncAttribute.cudaFuncAttributeMax

.. autoclass:: cuda.bindings.runtime.cudaFuncCache

    .. autoattribute:: cuda.bindings.runtime.cudaFuncCache.cudaFuncCachePreferNone


        Default function cache configuration, no preference


    .. autoattribute:: cuda.bindings.runtime.cudaFuncCache.cudaFuncCachePreferShared


        Prefer larger shared memory and smaller L1 cache


    .. autoattribute:: cuda.bindings.runtime.cudaFuncCache.cudaFuncCachePreferL1


        Prefer larger L1 cache and smaller shared memory


    .. autoattribute:: cuda.bindings.runtime.cudaFuncCache.cudaFuncCachePreferEqual


        Prefer equal size L1 cache and shared memory

.. autoclass:: cuda.bindings.runtime.cudaSharedMemConfig

    .. autoattribute:: cuda.bindings.runtime.cudaSharedMemConfig.cudaSharedMemBankSizeDefault


    .. autoattribute:: cuda.bindings.runtime.cudaSharedMemConfig.cudaSharedMemBankSizeFourByte


    .. autoattribute:: cuda.bindings.runtime.cudaSharedMemConfig.cudaSharedMemBankSizeEightByte

.. autoclass:: cuda.bindings.runtime.cudaSharedCarveout

    .. autoattribute:: cuda.bindings.runtime.cudaSharedCarveout.cudaSharedmemCarveoutDefault


        No preference for shared memory or L1 (default)


    .. autoattribute:: cuda.bindings.runtime.cudaSharedCarveout.cudaSharedmemCarveoutMaxShared


        Prefer maximum available shared memory, minimum L1 cache


    .. autoattribute:: cuda.bindings.runtime.cudaSharedCarveout.cudaSharedmemCarveoutMaxL1


        Prefer maximum available L1 cache, minimum shared memory

.. autoclass:: cuda.bindings.runtime.cudaComputeMode

    .. autoattribute:: cuda.bindings.runtime.cudaComputeMode.cudaComputeModeDefault


        Default compute mode (Multiple threads can use :py:obj:`~.cudaSetDevice()` with this device)


    .. autoattribute:: cuda.bindings.runtime.cudaComputeMode.cudaComputeModeExclusive


        Compute-exclusive-thread mode (Only one thread in one process will be able to use :py:obj:`~.cudaSetDevice()` with this device)


    .. autoattribute:: cuda.bindings.runtime.cudaComputeMode.cudaComputeModeProhibited


        Compute-prohibited mode (No threads can use :py:obj:`~.cudaSetDevice()` with this device)


    .. autoattribute:: cuda.bindings.runtime.cudaComputeMode.cudaComputeModeExclusiveProcess


        Compute-exclusive-process mode (Many threads in one process will be able to use :py:obj:`~.cudaSetDevice()` with this device)

.. autoclass:: cuda.bindings.runtime.cudaLimit

    .. autoattribute:: cuda.bindings.runtime.cudaLimit.cudaLimitStackSize


        GPU thread stack size


    .. autoattribute:: cuda.bindings.runtime.cudaLimit.cudaLimitPrintfFifoSize


        GPU printf FIFO size


    .. autoattribute:: cuda.bindings.runtime.cudaLimit.cudaLimitMallocHeapSize


        GPU malloc heap size


    .. autoattribute:: cuda.bindings.runtime.cudaLimit.cudaLimitDevRuntimeSyncDepth


        GPU device runtime synchronize depth


    .. autoattribute:: cuda.bindings.runtime.cudaLimit.cudaLimitDevRuntimePendingLaunchCount


        GPU device runtime pending launch count


    .. autoattribute:: cuda.bindings.runtime.cudaLimit.cudaLimitMaxL2FetchGranularity


        A value between 0 and 128 that indicates the maximum fetch granularity of L2 (in Bytes). This is a hint


    .. autoattribute:: cuda.bindings.runtime.cudaLimit.cudaLimitPersistingL2CacheSize


        A size in bytes for L2 persisting lines cache size

.. autoclass:: cuda.bindings.runtime.cudaMemoryAdvise

    .. autoattribute:: cuda.bindings.runtime.cudaMemoryAdvise.cudaMemAdviseSetReadMostly


        Data will mostly be read and only occassionally be written to


    .. autoattribute:: cuda.bindings.runtime.cudaMemoryAdvise.cudaMemAdviseUnsetReadMostly


        Undo the effect of :py:obj:`~.cudaMemAdviseSetReadMostly`


    .. autoattribute:: cuda.bindings.runtime.cudaMemoryAdvise.cudaMemAdviseSetPreferredLocation


        Set the preferred location for the data as the specified device


    .. autoattribute:: cuda.bindings.runtime.cudaMemoryAdvise.cudaMemAdviseUnsetPreferredLocation


        Clear the preferred location for the data


    .. autoattribute:: cuda.bindings.runtime.cudaMemoryAdvise.cudaMemAdviseSetAccessedBy


        Data will be accessed by the specified device, so prevent page faults as much as possible


    .. autoattribute:: cuda.bindings.runtime.cudaMemoryAdvise.cudaMemAdviseUnsetAccessedBy


        Let the Unified Memory subsystem decide on the page faulting policy for the specified device

.. autoclass:: cuda.bindings.runtime.cudaMemRangeAttribute

    .. autoattribute:: cuda.bindings.runtime.cudaMemRangeAttribute.cudaMemRangeAttributeReadMostly


        Whether the range will mostly be read and only occassionally be written to


    .. autoattribute:: cuda.bindings.runtime.cudaMemRangeAttribute.cudaMemRangeAttributePreferredLocation


        The preferred location of the range


    .. autoattribute:: cuda.bindings.runtime.cudaMemRangeAttribute.cudaMemRangeAttributeAccessedBy


        Memory range has :py:obj:`~.cudaMemAdviseSetAccessedBy` set for specified device


    .. autoattribute:: cuda.bindings.runtime.cudaMemRangeAttribute.cudaMemRangeAttributeLastPrefetchLocation


        The last location to which the range was prefetched


    .. autoattribute:: cuda.bindings.runtime.cudaMemRangeAttribute.cudaMemRangeAttributePreferredLocationType


        The preferred location type of the range


    .. autoattribute:: cuda.bindings.runtime.cudaMemRangeAttribute.cudaMemRangeAttributePreferredLocationId


        The preferred location id of the range


    .. autoattribute:: cuda.bindings.runtime.cudaMemRangeAttribute.cudaMemRangeAttributeLastPrefetchLocationType


        The last location type to which the range was prefetched


    .. autoattribute:: cuda.bindings.runtime.cudaMemRangeAttribute.cudaMemRangeAttributeLastPrefetchLocationId


        The last location id to which the range was prefetched

.. autoclass:: cuda.bindings.runtime.cudaFlushGPUDirectRDMAWritesOptions

    .. autoattribute:: cuda.bindings.runtime.cudaFlushGPUDirectRDMAWritesOptions.cudaFlushGPUDirectRDMAWritesOptionHost


        :py:obj:`~.cudaDeviceFlushGPUDirectRDMAWrites()` and its CUDA Driver API counterpart are supported on the device.


    .. autoattribute:: cuda.bindings.runtime.cudaFlushGPUDirectRDMAWritesOptions.cudaFlushGPUDirectRDMAWritesOptionMemOps


        The :py:obj:`~.CU_STREAM_WAIT_VALUE_FLUSH` flag and the :py:obj:`~.CU_STREAM_MEM_OP_FLUSH_REMOTE_WRITES` MemOp are supported on the CUDA device.

.. autoclass:: cuda.bindings.runtime.cudaGPUDirectRDMAWritesOrdering

    .. autoattribute:: cuda.bindings.runtime.cudaGPUDirectRDMAWritesOrdering.cudaGPUDirectRDMAWritesOrderingNone


        The device does not natively support ordering of GPUDirect RDMA writes. :py:obj:`~.cudaFlushGPUDirectRDMAWrites()` can be leveraged if supported.


    .. autoattribute:: cuda.bindings.runtime.cudaGPUDirectRDMAWritesOrdering.cudaGPUDirectRDMAWritesOrderingOwner


        Natively, the device can consistently consume GPUDirect RDMA writes, although other CUDA devices may not.


    .. autoattribute:: cuda.bindings.runtime.cudaGPUDirectRDMAWritesOrdering.cudaGPUDirectRDMAWritesOrderingAllDevices


        Any CUDA device in the system can consistently consume GPUDirect RDMA writes to this device.

.. autoclass:: cuda.bindings.runtime.cudaFlushGPUDirectRDMAWritesScope

    .. autoattribute:: cuda.bindings.runtime.cudaFlushGPUDirectRDMAWritesScope.cudaFlushGPUDirectRDMAWritesToOwner


        Blocks until remote writes are visible to the CUDA device context owning the data.


    .. autoattribute:: cuda.bindings.runtime.cudaFlushGPUDirectRDMAWritesScope.cudaFlushGPUDirectRDMAWritesToAllDevices


        Blocks until remote writes are visible to all CUDA device contexts.

.. autoclass:: cuda.bindings.runtime.cudaFlushGPUDirectRDMAWritesTarget

    .. autoattribute:: cuda.bindings.runtime.cudaFlushGPUDirectRDMAWritesTarget.cudaFlushGPUDirectRDMAWritesTargetCurrentDevice


        Sets the target for :py:obj:`~.cudaDeviceFlushGPUDirectRDMAWrites()` to the currently active CUDA device context.

.. autoclass:: cuda.bindings.runtime.cudaDeviceAttr

    .. autoattribute:: cuda.bindings.runtime.cudaDeviceAttr.cudaDevAttrMaxThreadsPerBlock


        Maximum number of threads per block


    .. autoattribute:: cuda.bindings.runtime.cudaDeviceAttr.cudaDevAttrMaxBlockDimX


        Maximum block dimension X


    .. autoattribute:: cuda.bindings.runtime.cudaDeviceAttr.cudaDevAttrMaxBlockDimY


        Maximum block dimension Y


    .. autoattribute:: cuda.bindings.runtime.cudaDeviceAttr.cudaDevAttrMaxBlockDimZ


        Maximum block dimension Z


    .. autoattribute:: cuda.bindings.runtime.cudaDeviceAttr.cudaDevAttrMaxGridDimX


        Maximum grid dimension X


    .. autoattribute:: cuda.bindings.runtime.cudaDeviceAttr.cudaDevAttrMaxGridDimY


        Maximum grid dimension Y


    .. autoattribute:: cuda.bindings.runtime.cudaDeviceAttr.cudaDevAttrMaxGridDimZ


        Maximum grid dimension Z


    .. autoattribute:: cuda.bindings.runtime.cudaDeviceAttr.cudaDevAttrMaxSharedMemoryPerBlock


        Maximum shared memory available per block in bytes


    .. autoattribute:: cuda.bindings.runtime.cudaDeviceAttr.cudaDevAttrTotalConstantMemory


        Memory available on device for constant variables in a CUDA C kernel in bytes


    .. autoattribute:: cuda.bindings.runtime.cudaDeviceAttr.cudaDevAttrWarpSize


        Warp size in threads


    .. autoattribute:: cuda.bindings.runtime.cudaDeviceAttr.cudaDevAttrMaxPitch


        Maximum pitch in bytes allowed by memory copies


    .. autoattribute:: cuda.bindings.runtime.cudaDeviceAttr.cudaDevAttrMaxRegistersPerBlock


        Maximum number of 32-bit registers available per block


    .. autoattribute:: cuda.bindings.runtime.cudaDeviceAttr.cudaDevAttrClockRate


        Peak clock frequency in kilohertz


    .. autoattribute:: cuda.bindings.runtime.cudaDeviceAttr.cudaDevAttrTextureAlignment


        Alignment requirement for textures


    .. autoattribute:: cuda.bindings.runtime.cudaDeviceAttr.cudaDevAttrGpuOverlap


        Device can possibly copy memory and execute a kernel concurrently


    .. autoattribute:: cuda.bindings.runtime.cudaDeviceAttr.cudaDevAttrMultiProcessorCount


        Number of multiprocessors on device


    .. autoattribute:: cuda.bindings.runtime.cudaDeviceAttr.cudaDevAttrKernelExecTimeout


        Specifies whether there is a run time limit on kernels


    .. autoattribute:: cuda.bindings.runtime.cudaDeviceAttr.cudaDevAttrIntegrated


        Device is integrated with host memory


    .. autoattribute:: cuda.bindings.runtime.cudaDeviceAttr.cudaDevAttrCanMapHostMemory


        Device can map host memory into CUDA address space


    .. autoattribute:: cuda.bindings.runtime.cudaDeviceAttr.cudaDevAttrComputeMode


        Compute mode (See :py:obj:`~.cudaComputeMode` for details)


    .. autoattribute:: cuda.bindings.runtime.cudaDeviceAttr.cudaDevAttrMaxTexture1DWidth


        Maximum 1D texture width


    .. autoattribute:: cuda.bindings.runtime.cudaDeviceAttr.cudaDevAttrMaxTexture2DWidth


        Maximum 2D texture width


    .. autoattribute:: cuda.bindings.runtime.cudaDeviceAttr.cudaDevAttrMaxTexture2DHeight


        Maximum 2D texture height


    .. autoattribute:: cuda.bindings.runtime.cudaDeviceAttr.cudaDevAttrMaxTexture3DWidth


        Maximum 3D texture width


    .. autoattribute:: cuda.bindings.runtime.cudaDeviceAttr.cudaDevAttrMaxTexture3DHeight


        Maximum 3D texture height


    .. autoattribute:: cuda.bindings.runtime.cudaDeviceAttr.cudaDevAttrMaxTexture3DDepth


        Maximum 3D texture depth


    .. autoattribute:: cuda.bindings.runtime.cudaDeviceAttr.cudaDevAttrMaxTexture2DLayeredWidth


        Maximum 2D layered texture width


    .. autoattribute:: cuda.bindings.runtime.cudaDeviceAttr.cudaDevAttrMaxTexture2DLayeredHeight


        Maximum 2D layered texture height


    .. autoattribute:: cuda.bindings.runtime.cudaDeviceAttr.cudaDevAttrMaxTexture2DLayeredLayers


        Maximum layers in a 2D layered texture


    .. autoattribute:: cuda.bindings.runtime.cudaDeviceAttr.cudaDevAttrSurfaceAlignment


        Alignment requirement for surfaces


    .. autoattribute:: cuda.bindings.runtime.cudaDeviceAttr.cudaDevAttrConcurrentKernels


        Device can possibly execute multiple kernels concurrently


    .. autoattribute:: cuda.bindings.runtime.cudaDeviceAttr.cudaDevAttrEccEnabled


        Device has ECC support enabled


    .. autoattribute:: cuda.bindings.runtime.cudaDeviceAttr.cudaDevAttrPciBusId


        PCI bus ID of the device


    .. autoattribute:: cuda.bindings.runtime.cudaDeviceAttr.cudaDevAttrPciDeviceId


        PCI device ID of the device


    .. autoattribute:: cuda.bindings.runtime.cudaDeviceAttr.cudaDevAttrTccDriver


        Device is using TCC driver model


    .. autoattribute:: cuda.bindings.runtime.cudaDeviceAttr.cudaDevAttrMemoryClockRate


        Peak memory clock frequency in kilohertz


    .. autoattribute:: cuda.bindings.runtime.cudaDeviceAttr.cudaDevAttrGlobalMemoryBusWidth


        Global memory bus width in bits


    .. autoattribute:: cuda.bindings.runtime.cudaDeviceAttr.cudaDevAttrL2CacheSize


        Size of L2 cache in bytes


    .. autoattribute:: cuda.bindings.runtime.cudaDeviceAttr.cudaDevAttrMaxThreadsPerMultiProcessor


        Maximum resident threads per multiprocessor


    .. autoattribute:: cuda.bindings.runtime.cudaDeviceAttr.cudaDevAttrAsyncEngineCount


        Number of asynchronous engines


    .. autoattribute:: cuda.bindings.runtime.cudaDeviceAttr.cudaDevAttrUnifiedAddressing


        Device shares a unified address space with the host


    .. autoattribute:: cuda.bindings.runtime.cudaDeviceAttr.cudaDevAttrMaxTexture1DLayeredWidth


        Maximum 1D layered texture width


    .. autoattribute:: cuda.bindings.runtime.cudaDeviceAttr.cudaDevAttrMaxTexture1DLayeredLayers


        Maximum layers in a 1D layered texture


    .. autoattribute:: cuda.bindings.runtime.cudaDeviceAttr.cudaDevAttrMaxTexture2DGatherWidth


        Maximum 2D texture width if cudaArrayTextureGather is set


    .. autoattribute:: cuda.bindings.runtime.cudaDeviceAttr.cudaDevAttrMaxTexture2DGatherHeight


        Maximum 2D texture height if cudaArrayTextureGather is set


    .. autoattribute:: cuda.bindings.runtime.cudaDeviceAttr.cudaDevAttrMaxTexture3DWidthAlt


        Alternate maximum 3D texture width


    .. autoattribute:: cuda.bindings.runtime.cudaDeviceAttr.cudaDevAttrMaxTexture3DHeightAlt


        Alternate maximum 3D texture height


    .. autoattribute:: cuda.bindings.runtime.cudaDeviceAttr.cudaDevAttrMaxTexture3DDepthAlt


        Alternate maximum 3D texture depth


    .. autoattribute:: cuda.bindings.runtime.cudaDeviceAttr.cudaDevAttrPciDomainId


        PCI domain ID of the device


    .. autoattribute:: cuda.bindings.runtime.cudaDeviceAttr.cudaDevAttrTexturePitchAlignment


        Pitch alignment requirement for textures


    .. autoattribute:: cuda.bindings.runtime.cudaDeviceAttr.cudaDevAttrMaxTextureCubemapWidth


        Maximum cubemap texture width/height


    .. autoattribute:: cuda.bindings.runtime.cudaDeviceAttr.cudaDevAttrMaxTextureCubemapLayeredWidth


        Maximum cubemap layered texture width/height


    .. autoattribute:: cuda.bindings.runtime.cudaDeviceAttr.cudaDevAttrMaxTextureCubemapLayeredLayers


        Maximum layers in a cubemap layered texture


    .. autoattribute:: cuda.bindings.runtime.cudaDeviceAttr.cudaDevAttrMaxSurface1DWidth


        Maximum 1D surface width


    .. autoattribute:: cuda.bindings.runtime.cudaDeviceAttr.cudaDevAttrMaxSurface2DWidth


        Maximum 2D surface width


    .. autoattribute:: cuda.bindings.runtime.cudaDeviceAttr.cudaDevAttrMaxSurface2DHeight


        Maximum 2D surface height


    .. autoattribute:: cuda.bindings.runtime.cudaDeviceAttr.cudaDevAttrMaxSurface3DWidth


        Maximum 3D surface width


    .. autoattribute:: cuda.bindings.runtime.cudaDeviceAttr.cudaDevAttrMaxSurface3DHeight


        Maximum 3D surface height


    .. autoattribute:: cuda.bindings.runtime.cudaDeviceAttr.cudaDevAttrMaxSurface3DDepth


        Maximum 3D surface depth


    .. autoattribute:: cuda.bindings.runtime.cudaDeviceAttr.cudaDevAttrMaxSurface1DLayeredWidth


        Maximum 1D layered surface width


    .. autoattribute:: cuda.bindings.runtime.cudaDeviceAttr.cudaDevAttrMaxSurface1DLayeredLayers


        Maximum layers in a 1D layered surface


    .. autoattribute:: cuda.bindings.runtime.cudaDeviceAttr.cudaDevAttrMaxSurface2DLayeredWidth


        Maximum 2D layered surface width


    .. autoattribute:: cuda.bindings.runtime.cudaDeviceAttr.cudaDevAttrMaxSurface2DLayeredHeight


        Maximum 2D layered surface height


    .. autoattribute:: cuda.bindings.runtime.cudaDeviceAttr.cudaDevAttrMaxSurface2DLayeredLayers


        Maximum layers in a 2D layered surface


    .. autoattribute:: cuda.bindings.runtime.cudaDeviceAttr.cudaDevAttrMaxSurfaceCubemapWidth


        Maximum cubemap surface width


    .. autoattribute:: cuda.bindings.runtime.cudaDeviceAttr.cudaDevAttrMaxSurfaceCubemapLayeredWidth


        Maximum cubemap layered surface width


    .. autoattribute:: cuda.bindings.runtime.cudaDeviceAttr.cudaDevAttrMaxSurfaceCubemapLayeredLayers


        Maximum layers in a cubemap layered surface


    .. autoattribute:: cuda.bindings.runtime.cudaDeviceAttr.cudaDevAttrMaxTexture1DLinearWidth


        Maximum 1D linear texture width


    .. autoattribute:: cuda.bindings.runtime.cudaDeviceAttr.cudaDevAttrMaxTexture2DLinearWidth


        Maximum 2D linear texture width


    .. autoattribute:: cuda.bindings.runtime.cudaDeviceAttr.cudaDevAttrMaxTexture2DLinearHeight


        Maximum 2D linear texture height


    .. autoattribute:: cuda.bindings.runtime.cudaDeviceAttr.cudaDevAttrMaxTexture2DLinearPitch


        Maximum 2D linear texture pitch in bytes


    .. autoattribute:: cuda.bindings.runtime.cudaDeviceAttr.cudaDevAttrMaxTexture2DMipmappedWidth


        Maximum mipmapped 2D texture width


    .. autoattribute:: cuda.bindings.runtime.cudaDeviceAttr.cudaDevAttrMaxTexture2DMipmappedHeight


        Maximum mipmapped 2D texture height


    .. autoattribute:: cuda.bindings.runtime.cudaDeviceAttr.cudaDevAttrComputeCapabilityMajor


        Major compute capability version number


    .. autoattribute:: cuda.bindings.runtime.cudaDeviceAttr.cudaDevAttrComputeCapabilityMinor


        Minor compute capability version number


    .. autoattribute:: cuda.bindings.runtime.cudaDeviceAttr.cudaDevAttrMaxTexture1DMipmappedWidth


        Maximum mipmapped 1D texture width


    .. autoattribute:: cuda.bindings.runtime.cudaDeviceAttr.cudaDevAttrStreamPrioritiesSupported


        Device supports stream priorities


    .. autoattribute:: cuda.bindings.runtime.cudaDeviceAttr.cudaDevAttrGlobalL1CacheSupported


        Device supports caching globals in L1


    .. autoattribute:: cuda.bindings.runtime.cudaDeviceAttr.cudaDevAttrLocalL1CacheSupported


        Device supports caching locals in L1


    .. autoattribute:: cuda.bindings.runtime.cudaDeviceAttr.cudaDevAttrMaxSharedMemoryPerMultiprocessor


        Maximum shared memory available per multiprocessor in bytes


    .. autoattribute:: cuda.bindings.runtime.cudaDeviceAttr.cudaDevAttrMaxRegistersPerMultiprocessor


        Maximum number of 32-bit registers available per multiprocessor


    .. autoattribute:: cuda.bindings.runtime.cudaDeviceAttr.cudaDevAttrManagedMemory


        Device can allocate managed memory on this system


    .. autoattribute:: cuda.bindings.runtime.cudaDeviceAttr.cudaDevAttrIsMultiGpuBoard


        Device is on a multi-GPU board


    .. autoattribute:: cuda.bindings.runtime.cudaDeviceAttr.cudaDevAttrMultiGpuBoardGroupID


        Unique identifier for a group of devices on the same multi-GPU board


    .. autoattribute:: cuda.bindings.runtime.cudaDeviceAttr.cudaDevAttrHostNativeAtomicSupported


        Link between the device and the host supports native atomic operations


    .. autoattribute:: cuda.bindings.runtime.cudaDeviceAttr.cudaDevAttrSingleToDoublePrecisionPerfRatio


        Ratio of single precision performance (in floating-point operations per second) to double precision performance


    .. autoattribute:: cuda.bindings.runtime.cudaDeviceAttr.cudaDevAttrPageableMemoryAccess


        Device supports coherently accessing pageable memory without calling cudaHostRegister on it


    .. autoattribute:: cuda.bindings.runtime.cudaDeviceAttr.cudaDevAttrConcurrentManagedAccess


        Device can coherently access managed memory concurrently with the CPU


    .. autoattribute:: cuda.bindings.runtime.cudaDeviceAttr.cudaDevAttrComputePreemptionSupported


        Device supports Compute Preemption


    .. autoattribute:: cuda.bindings.runtime.cudaDeviceAttr.cudaDevAttrCanUseHostPointerForRegisteredMem


        Device can access host registered memory at the same virtual address as the CPU


    .. autoattribute:: cuda.bindings.runtime.cudaDeviceAttr.cudaDevAttrReserved92


    .. autoattribute:: cuda.bindings.runtime.cudaDeviceAttr.cudaDevAttrReserved93


    .. autoattribute:: cuda.bindings.runtime.cudaDeviceAttr.cudaDevAttrReserved94


    .. autoattribute:: cuda.bindings.runtime.cudaDeviceAttr.cudaDevAttrCooperativeLaunch


        Device supports launching cooperative kernels via :py:obj:`~.cudaLaunchCooperativeKernel`


    .. autoattribute:: cuda.bindings.runtime.cudaDeviceAttr.cudaDevAttrReserved96


    .. autoattribute:: cuda.bindings.runtime.cudaDeviceAttr.cudaDevAttrMaxSharedMemoryPerBlockOptin


        The maximum optin shared memory per block. This value may vary by chip. See :py:obj:`~.cudaFuncSetAttribute`


    .. autoattribute:: cuda.bindings.runtime.cudaDeviceAttr.cudaDevAttrCanFlushRemoteWrites


        Device supports flushing of outstanding remote writes.


    .. autoattribute:: cuda.bindings.runtime.cudaDeviceAttr.cudaDevAttrHostRegisterSupported


        Device supports host memory registration via :py:obj:`~.cudaHostRegister`.


    .. autoattribute:: cuda.bindings.runtime.cudaDeviceAttr.cudaDevAttrPageableMemoryAccessUsesHostPageTables


        Device accesses pageable memory via the host's page tables.


    .. autoattribute:: cuda.bindings.runtime.cudaDeviceAttr.cudaDevAttrDirectManagedMemAccessFromHost


        Host can directly access managed memory on the device without migration.


    .. autoattribute:: cuda.bindings.runtime.cudaDeviceAttr.cudaDevAttrMaxBlocksPerMultiprocessor


        Maximum number of blocks per multiprocessor


    .. autoattribute:: cuda.bindings.runtime.cudaDeviceAttr.cudaDevAttrMaxPersistingL2CacheSize


        Maximum L2 persisting lines capacity setting in bytes.


    .. autoattribute:: cuda.bindings.runtime.cudaDeviceAttr.cudaDevAttrMaxAccessPolicyWindowSize


        Maximum value of :py:obj:`~.cudaAccessPolicyWindow.num_bytes`.


    .. autoattribute:: cuda.bindings.runtime.cudaDeviceAttr.cudaDevAttrReservedSharedMemoryPerBlock


        Shared memory reserved by CUDA driver per block in bytes


    .. autoattribute:: cuda.bindings.runtime.cudaDeviceAttr.cudaDevAttrSparseCudaArraySupported


        Device supports sparse CUDA arrays and sparse CUDA mipmapped arrays


    .. autoattribute:: cuda.bindings.runtime.cudaDeviceAttr.cudaDevAttrHostRegisterReadOnlySupported


        Device supports using the :py:obj:`~.cudaHostRegister` flag cudaHostRegisterReadOnly to register memory that must be mapped as read-only to the GPU


    .. autoattribute:: cuda.bindings.runtime.cudaDeviceAttr.cudaDevAttrTimelineSemaphoreInteropSupported


        External timeline semaphore interop is supported on the device


    .. autoattribute:: cuda.bindings.runtime.cudaDeviceAttr.cudaDevAttrMemoryPoolsSupported


        Device supports using the :py:obj:`~.cudaMallocAsync` and :py:obj:`~.cudaMemPool` family of APIs


    .. autoattribute:: cuda.bindings.runtime.cudaDeviceAttr.cudaDevAttrGPUDirectRDMASupported


        Device supports GPUDirect RDMA APIs, like nvidia_p2p_get_pages (see https://docs.nvidia.com/cuda/gpudirect-rdma for more information)


    .. autoattribute:: cuda.bindings.runtime.cudaDeviceAttr.cudaDevAttrGPUDirectRDMAFlushWritesOptions


        The returned attribute shall be interpreted as a bitmask, where the individual bits are listed in the :py:obj:`~.cudaFlushGPUDirectRDMAWritesOptions` enum


    .. autoattribute:: cuda.bindings.runtime.cudaDeviceAttr.cudaDevAttrGPUDirectRDMAWritesOrdering


        GPUDirect RDMA writes to the device do not need to be flushed for consumers within the scope indicated by the returned attribute. See :py:obj:`~.cudaGPUDirectRDMAWritesOrdering` for the numerical values returned here.


    .. autoattribute:: cuda.bindings.runtime.cudaDeviceAttr.cudaDevAttrMemoryPoolSupportedHandleTypes


        Handle types supported with mempool based IPC


    .. autoattribute:: cuda.bindings.runtime.cudaDeviceAttr.cudaDevAttrClusterLaunch


        Indicates device supports cluster launch


    .. autoattribute:: cuda.bindings.runtime.cudaDeviceAttr.cudaDevAttrDeferredMappingCudaArraySupported


        Device supports deferred mapping CUDA arrays and CUDA mipmapped arrays


    .. autoattribute:: cuda.bindings.runtime.cudaDeviceAttr.cudaDevAttrReserved122


    .. autoattribute:: cuda.bindings.runtime.cudaDeviceAttr.cudaDevAttrReserved123


    .. autoattribute:: cuda.bindings.runtime.cudaDeviceAttr.cudaDevAttrReserved124


    .. autoattribute:: cuda.bindings.runtime.cudaDeviceAttr.cudaDevAttrIpcEventSupport


        Device supports IPC Events.


    .. autoattribute:: cuda.bindings.runtime.cudaDeviceAttr.cudaDevAttrMemSyncDomainCount


        Number of memory synchronization domains the device supports.


    .. autoattribute:: cuda.bindings.runtime.cudaDeviceAttr.cudaDevAttrReserved127


    .. autoattribute:: cuda.bindings.runtime.cudaDeviceAttr.cudaDevAttrReserved128


    .. autoattribute:: cuda.bindings.runtime.cudaDeviceAttr.cudaDevAttrReserved129


    .. autoattribute:: cuda.bindings.runtime.cudaDeviceAttr.cudaDevAttrNumaConfig


        NUMA configuration of a device: value is of type :py:obj:`~.cudaDeviceNumaConfig` enum


    .. autoattribute:: cuda.bindings.runtime.cudaDeviceAttr.cudaDevAttrNumaId


        NUMA node ID of the GPU memory


    .. autoattribute:: cuda.bindings.runtime.cudaDeviceAttr.cudaDevAttrReserved132


    .. autoattribute:: cuda.bindings.runtime.cudaDeviceAttr.cudaDevAttrMpsEnabled


        Contexts created on this device will be shared via MPS


    .. autoattribute:: cuda.bindings.runtime.cudaDeviceAttr.cudaDevAttrHostNumaId


        NUMA ID of the host node closest to the device or -1 when system does not support NUMA


    .. autoattribute:: cuda.bindings.runtime.cudaDeviceAttr.cudaDevAttrD3D12CigSupported


        Device supports CIG with D3D12.


    .. autoattribute:: cuda.bindings.runtime.cudaDeviceAttr.cudaDevAttrVulkanCigSupported


        Device supports CIG with Vulkan.


    .. autoattribute:: cuda.bindings.runtime.cudaDeviceAttr.cudaDevAttrGpuPciDeviceId


        The combined 16-bit PCI device ID and 16-bit PCI vendor ID.


    .. autoattribute:: cuda.bindings.runtime.cudaDeviceAttr.cudaDevAttrGpuPciSubsystemId


        The combined 16-bit PCI subsystem ID and 16-bit PCI subsystem vendor ID.


    .. autoattribute:: cuda.bindings.runtime.cudaDeviceAttr.cudaDevAttrReserved141


    .. autoattribute:: cuda.bindings.runtime.cudaDeviceAttr.cudaDevAttrHostNumaMemoryPoolsSupported


        Device supports HOST_NUMA location with the :py:obj:`~.cudaMallocAsync` and :py:obj:`~.cudaMemPool` family of APIs


    .. autoattribute:: cuda.bindings.runtime.cudaDeviceAttr.cudaDevAttrHostNumaMultinodeIpcSupported


        Device supports HostNuma location IPC between nodes in a multi-node system.


    .. autoattribute:: cuda.bindings.runtime.cudaDeviceAttr.cudaDevAttrHostMemoryPoolsSupported


        Device suports HOST location with the :py:obj:`~.cuMemAllocAsync` and :py:obj:`~.cuMemPool` family of APIs


    .. autoattribute:: cuda.bindings.runtime.cudaDeviceAttr.cudaDevAttrReserved145


    .. autoattribute:: cuda.bindings.runtime.cudaDeviceAttr.cudaDevAttrOnlyPartialHostNativeAtomicSupported


        Link between the device and the host supports only some native atomic operations


    .. autoattribute:: cuda.bindings.runtime.cudaDeviceAttr.cudaDevAttrMax

.. autoclass:: cuda.bindings.runtime.cudaMemPoolAttr

    .. autoattribute:: cuda.bindings.runtime.cudaMemPoolAttr.cudaMemPoolReuseFollowEventDependencies


        (value type = int) Allow cuMemAllocAsync to use memory asynchronously freed in another streams as long as a stream ordering dependency of the allocating stream on the free action exists. Cuda events and null stream interactions can create the required stream ordered dependencies. (default enabled)


    .. autoattribute:: cuda.bindings.runtime.cudaMemPoolAttr.cudaMemPoolReuseAllowOpportunistic


        (value type = int) Allow reuse of already completed frees when there is no dependency between the free and allocation. (default enabled)


    .. autoattribute:: cuda.bindings.runtime.cudaMemPoolAttr.cudaMemPoolReuseAllowInternalDependencies


        (value type = int) Allow cuMemAllocAsync to insert new stream dependencies in order to establish the stream ordering required to reuse a piece of memory released by cuFreeAsync (default enabled).


    .. autoattribute:: cuda.bindings.runtime.cudaMemPoolAttr.cudaMemPoolAttrReleaseThreshold


        (value type = cuuint64_t) Amount of reserved memory in bytes to hold onto before trying to release memory back to the OS. When more than the release threshold bytes of memory are held by the memory pool, the allocator will try to release memory back to the OS on the next call to stream, event or context synchronize. (default 0)


    .. autoattribute:: cuda.bindings.runtime.cudaMemPoolAttr.cudaMemPoolAttrReservedMemCurrent


        (value type = cuuint64_t) Amount of backing memory currently allocated for the mempool.


    .. autoattribute:: cuda.bindings.runtime.cudaMemPoolAttr.cudaMemPoolAttrReservedMemHigh


        (value type = cuuint64_t) High watermark of backing memory allocated for the mempool since the last time it was reset. High watermark can only be reset to zero.


    .. autoattribute:: cuda.bindings.runtime.cudaMemPoolAttr.cudaMemPoolAttrUsedMemCurrent


        (value type = cuuint64_t) Amount of memory from the pool that is currently in use by the application.


    .. autoattribute:: cuda.bindings.runtime.cudaMemPoolAttr.cudaMemPoolAttrUsedMemHigh


        (value type = cuuint64_t) High watermark of the amount of memory from the pool that was in use by the application since the last time it was reset. High watermark can only be reset to zero.

.. autoclass:: cuda.bindings.runtime.cudaMemLocationType

    .. autoattribute:: cuda.bindings.runtime.cudaMemLocationType.cudaMemLocationTypeInvalid


    .. autoattribute:: cuda.bindings.runtime.cudaMemLocationType.cudaMemLocationTypeNone


        Location is unspecified. This is used when creating a managed memory pool to indicate no preferred location for the pool


    .. autoattribute:: cuda.bindings.runtime.cudaMemLocationType.cudaMemLocationTypeDevice


        Location is a device location, thus id is a device ordinal


    .. autoattribute:: cuda.bindings.runtime.cudaMemLocationType.cudaMemLocationTypeHost


        Location is host, id is ignored


    .. autoattribute:: cuda.bindings.runtime.cudaMemLocationType.cudaMemLocationTypeHostNuma


        Location is a host NUMA node, thus id is a host NUMA node id


    .. autoattribute:: cuda.bindings.runtime.cudaMemLocationType.cudaMemLocationTypeHostNumaCurrent


        Location is the host NUMA node closest to the current thread's CPU, id is ignored

.. autoclass:: cuda.bindings.runtime.cudaMemAccessFlags

    .. autoattribute:: cuda.bindings.runtime.cudaMemAccessFlags.cudaMemAccessFlagsProtNone


        Default, make the address range not accessible


    .. autoattribute:: cuda.bindings.runtime.cudaMemAccessFlags.cudaMemAccessFlagsProtRead


        Make the address range read accessible


    .. autoattribute:: cuda.bindings.runtime.cudaMemAccessFlags.cudaMemAccessFlagsProtReadWrite


        Make the address range read-write accessible

.. autoclass:: cuda.bindings.runtime.cudaMemAllocationType

    .. autoattribute:: cuda.bindings.runtime.cudaMemAllocationType.cudaMemAllocationTypeInvalid


    .. autoattribute:: cuda.bindings.runtime.cudaMemAllocationType.cudaMemAllocationTypePinned


        This allocation type is 'pinned', i.e. cannot migrate from its current location while the application is actively using it


    .. autoattribute:: cuda.bindings.runtime.cudaMemAllocationType.cudaMemAllocationTypeManaged


        This allocation type is managed memory


    .. autoattribute:: cuda.bindings.runtime.cudaMemAllocationType.cudaMemAllocationTypeMax

.. autoclass:: cuda.bindings.runtime.cudaMemAllocationHandleType

    .. autoattribute:: cuda.bindings.runtime.cudaMemAllocationHandleType.cudaMemHandleTypeNone


        Does not allow any export mechanism. >


    .. autoattribute:: cuda.bindings.runtime.cudaMemAllocationHandleType.cudaMemHandleTypePosixFileDescriptor


        Allows a file descriptor to be used for exporting. Permitted only on POSIX systems. (int)


    .. autoattribute:: cuda.bindings.runtime.cudaMemAllocationHandleType.cudaMemHandleTypeWin32


        Allows a Win32 NT handle to be used for exporting. (HANDLE)


    .. autoattribute:: cuda.bindings.runtime.cudaMemAllocationHandleType.cudaMemHandleTypeWin32Kmt


        Allows a Win32 KMT handle to be used for exporting. (D3DKMT_HANDLE)


    .. autoattribute:: cuda.bindings.runtime.cudaMemAllocationHandleType.cudaMemHandleTypeFabric


        Allows a fabric handle to be used for exporting. (cudaMemFabricHandle_t)

.. autoclass:: cuda.bindings.runtime.cudaGraphMemAttributeType

    .. autoattribute:: cuda.bindings.runtime.cudaGraphMemAttributeType.cudaGraphMemAttrUsedMemCurrent


        (value type = cuuint64_t) Amount of memory, in bytes, currently associated with graphs.


    .. autoattribute:: cuda.bindings.runtime.cudaGraphMemAttributeType.cudaGraphMemAttrUsedMemHigh


        (value type = cuuint64_t) High watermark of memory, in bytes, associated with graphs since the last time it was reset. High watermark can only be reset to zero.


    .. autoattribute:: cuda.bindings.runtime.cudaGraphMemAttributeType.cudaGraphMemAttrReservedMemCurrent


        (value type = cuuint64_t) Amount of memory, in bytes, currently allocated for use by the CUDA graphs asynchronous allocator.


    .. autoattribute:: cuda.bindings.runtime.cudaGraphMemAttributeType.cudaGraphMemAttrReservedMemHigh


        (value type = cuuint64_t) High watermark of memory, in bytes, currently allocated for use by the CUDA graphs asynchronous allocator.

.. autoclass:: cuda.bindings.runtime.cudaMemcpyFlags

    .. autoattribute:: cuda.bindings.runtime.cudaMemcpyFlags.cudaMemcpyFlagDefault


    .. autoattribute:: cuda.bindings.runtime.cudaMemcpyFlags.cudaMemcpyFlagPreferOverlapWithCompute


        Hint to the driver to try and overlap the copy with compute work on the SMs.

.. autoclass:: cuda.bindings.runtime.cudaMemcpySrcAccessOrder

    .. autoattribute:: cuda.bindings.runtime.cudaMemcpySrcAccessOrder.cudaMemcpySrcAccessOrderInvalid


        Default invalid.


    .. autoattribute:: cuda.bindings.runtime.cudaMemcpySrcAccessOrder.cudaMemcpySrcAccessOrderStream


        Indicates that access to the source pointer must be in stream order.


    .. autoattribute:: cuda.bindings.runtime.cudaMemcpySrcAccessOrder.cudaMemcpySrcAccessOrderDuringApiCall


        Indicates that access to the source pointer can be out of stream order and all accesses must be complete before the API call returns. This flag is suited for ephemeral sources (ex., stack variables) when it's known that no prior operations in the stream can be accessing the memory and also that the lifetime of the memory is limited to the scope that the source variable was declared in. Specifying this flag allows the driver to optimize the copy and removes the need for the user to synchronize the stream after the API call.


    .. autoattribute:: cuda.bindings.runtime.cudaMemcpySrcAccessOrder.cudaMemcpySrcAccessOrderAny


        Indicates that access to the source pointer can be out of stream order and the accesses can happen even after the API call returns. This flag is suited for host pointers allocated outside CUDA (ex., via malloc) when it's known that no prior operations in the stream can be accessing the memory. Specifying this flag allows the driver to optimize the copy on certain platforms.


    .. autoattribute:: cuda.bindings.runtime.cudaMemcpySrcAccessOrder.cudaMemcpySrcAccessOrderMax

.. autoclass:: cuda.bindings.runtime.cudaMemcpy3DOperandType

    .. autoattribute:: cuda.bindings.runtime.cudaMemcpy3DOperandType.cudaMemcpyOperandTypePointer


        Memcpy operand is a valid pointer.


    .. autoattribute:: cuda.bindings.runtime.cudaMemcpy3DOperandType.cudaMemcpyOperandTypeArray


        Memcpy operand is a CUarray.


    .. autoattribute:: cuda.bindings.runtime.cudaMemcpy3DOperandType.cudaMemcpyOperandTypeMax

.. autoclass:: cuda.bindings.runtime.cudaDeviceP2PAttr

    .. autoattribute:: cuda.bindings.runtime.cudaDeviceP2PAttr.cudaDevP2PAttrPerformanceRank


        A relative value indicating the performance of the link between two devices


    .. autoattribute:: cuda.bindings.runtime.cudaDeviceP2PAttr.cudaDevP2PAttrAccessSupported


        Peer access is enabled


    .. autoattribute:: cuda.bindings.runtime.cudaDeviceP2PAttr.cudaDevP2PAttrNativeAtomicSupported


        Native atomic operation over the link supported


    .. autoattribute:: cuda.bindings.runtime.cudaDeviceP2PAttr.cudaDevP2PAttrCudaArrayAccessSupported


        Accessing CUDA arrays over the link supported


    .. autoattribute:: cuda.bindings.runtime.cudaDeviceP2PAttr.cudaDevP2PAttrOnlyPartialNativeAtomicSupported


        Only some CUDA-valid atomic operations over the link are supported.

.. autoclass:: cuda.bindings.runtime.cudaAtomicOperation

    .. autoattribute:: cuda.bindings.runtime.cudaAtomicOperation.cudaAtomicOperationIntegerAdd


    .. autoattribute:: cuda.bindings.runtime.cudaAtomicOperation.cudaAtomicOperationIntegerMin


    .. autoattribute:: cuda.bindings.runtime.cudaAtomicOperation.cudaAtomicOperationIntegerMax


    .. autoattribute:: cuda.bindings.runtime.cudaAtomicOperation.cudaAtomicOperationIntegerIncrement


    .. autoattribute:: cuda.bindings.runtime.cudaAtomicOperation.cudaAtomicOperationIntegerDecrement


    .. autoattribute:: cuda.bindings.runtime.cudaAtomicOperation.cudaAtomicOperationAnd


    .. autoattribute:: cuda.bindings.runtime.cudaAtomicOperation.cudaAtomicOperationOr


    .. autoattribute:: cuda.bindings.runtime.cudaAtomicOperation.cudaAtomicOperationXOR


    .. autoattribute:: cuda.bindings.runtime.cudaAtomicOperation.cudaAtomicOperationExchange


    .. autoattribute:: cuda.bindings.runtime.cudaAtomicOperation.cudaAtomicOperationCAS


    .. autoattribute:: cuda.bindings.runtime.cudaAtomicOperation.cudaAtomicOperationFloatAdd


    .. autoattribute:: cuda.bindings.runtime.cudaAtomicOperation.cudaAtomicOperationFloatMin


    .. autoattribute:: cuda.bindings.runtime.cudaAtomicOperation.cudaAtomicOperationFloatMax

.. autoclass:: cuda.bindings.runtime.cudaAtomicOperationCapability

    .. autoattribute:: cuda.bindings.runtime.cudaAtomicOperationCapability.cudaAtomicCapabilitySigned


    .. autoattribute:: cuda.bindings.runtime.cudaAtomicOperationCapability.cudaAtomicCapabilityUnsigned


    .. autoattribute:: cuda.bindings.runtime.cudaAtomicOperationCapability.cudaAtomicCapabilityReduction


    .. autoattribute:: cuda.bindings.runtime.cudaAtomicOperationCapability.cudaAtomicCapabilityScalar32


    .. autoattribute:: cuda.bindings.runtime.cudaAtomicOperationCapability.cudaAtomicCapabilityScalar64


    .. autoattribute:: cuda.bindings.runtime.cudaAtomicOperationCapability.cudaAtomicCapabilityScalar128


    .. autoattribute:: cuda.bindings.runtime.cudaAtomicOperationCapability.cudaAtomicCapabilityVector32x4

.. autoclass:: cuda.bindings.runtime.cudaExternalMemoryHandleType

    .. autoattribute:: cuda.bindings.runtime.cudaExternalMemoryHandleType.cudaExternalMemoryHandleTypeOpaqueFd


        Handle is an opaque file descriptor


    .. autoattribute:: cuda.bindings.runtime.cudaExternalMemoryHandleType.cudaExternalMemoryHandleTypeOpaqueWin32


        Handle is an opaque shared NT handle


    .. autoattribute:: cuda.bindings.runtime.cudaExternalMemoryHandleType.cudaExternalMemoryHandleTypeOpaqueWin32Kmt


        Handle is an opaque, globally shared handle


    .. autoattribute:: cuda.bindings.runtime.cudaExternalMemoryHandleType.cudaExternalMemoryHandleTypeD3D12Heap


        Handle is a D3D12 heap object


    .. autoattribute:: cuda.bindings.runtime.cudaExternalMemoryHandleType.cudaExternalMemoryHandleTypeD3D12Resource


        Handle is a D3D12 committed resource


    .. autoattribute:: cuda.bindings.runtime.cudaExternalMemoryHandleType.cudaExternalMemoryHandleTypeD3D11Resource


        Handle is a shared NT handle to a D3D11 resource


    .. autoattribute:: cuda.bindings.runtime.cudaExternalMemoryHandleType.cudaExternalMemoryHandleTypeD3D11ResourceKmt


        Handle is a globally shared handle to a D3D11 resource


    .. autoattribute:: cuda.bindings.runtime.cudaExternalMemoryHandleType.cudaExternalMemoryHandleTypeNvSciBuf


        Handle is an NvSciBuf object

.. autoclass:: cuda.bindings.runtime.cudaExternalSemaphoreHandleType

    .. autoattribute:: cuda.bindings.runtime.cudaExternalSemaphoreHandleType.cudaExternalSemaphoreHandleTypeOpaqueFd


        Handle is an opaque file descriptor


    .. autoattribute:: cuda.bindings.runtime.cudaExternalSemaphoreHandleType.cudaExternalSemaphoreHandleTypeOpaqueWin32


        Handle is an opaque shared NT handle


    .. autoattribute:: cuda.bindings.runtime.cudaExternalSemaphoreHandleType.cudaExternalSemaphoreHandleTypeOpaqueWin32Kmt


        Handle is an opaque, globally shared handle


    .. autoattribute:: cuda.bindings.runtime.cudaExternalSemaphoreHandleType.cudaExternalSemaphoreHandleTypeD3D12Fence


        Handle is a shared NT handle referencing a D3D12 fence object


    .. autoattribute:: cuda.bindings.runtime.cudaExternalSemaphoreHandleType.cudaExternalSemaphoreHandleTypeD3D11Fence


        Handle is a shared NT handle referencing a D3D11 fence object


    .. autoattribute:: cuda.bindings.runtime.cudaExternalSemaphoreHandleType.cudaExternalSemaphoreHandleTypeNvSciSync


        Opaque handle to NvSciSync Object


    .. autoattribute:: cuda.bindings.runtime.cudaExternalSemaphoreHandleType.cudaExternalSemaphoreHandleTypeKeyedMutex


        Handle is a shared NT handle referencing a D3D11 keyed mutex object


    .. autoattribute:: cuda.bindings.runtime.cudaExternalSemaphoreHandleType.cudaExternalSemaphoreHandleTypeKeyedMutexKmt


        Handle is a shared KMT handle referencing a D3D11 keyed mutex object


    .. autoattribute:: cuda.bindings.runtime.cudaExternalSemaphoreHandleType.cudaExternalSemaphoreHandleTypeTimelineSemaphoreFd


        Handle is an opaque handle file descriptor referencing a timeline semaphore


    .. autoattribute:: cuda.bindings.runtime.cudaExternalSemaphoreHandleType.cudaExternalSemaphoreHandleTypeTimelineSemaphoreWin32


        Handle is an opaque handle file descriptor referencing a timeline semaphore

.. autoclass:: cuda.bindings.runtime.cudaDevSmResourceGroup_flags

    .. autoattribute:: cuda.bindings.runtime.cudaDevSmResourceGroup_flags.cudaDevSmResourceGroupDefault


    .. autoattribute:: cuda.bindings.runtime.cudaDevSmResourceGroup_flags.cudaDevSmResourceGroupBackfill

.. autoclass:: cuda.bindings.runtime.cudaDevSmResourceSplitByCount_flags

    .. autoattribute:: cuda.bindings.runtime.cudaDevSmResourceSplitByCount_flags.cudaDevSmResourceSplitIgnoreSmCoscheduling


    .. autoattribute:: cuda.bindings.runtime.cudaDevSmResourceSplitByCount_flags.cudaDevSmResourceSplitMaxPotentialClusterSize

.. autoclass:: cuda.bindings.runtime.cudaDevResourceType

    .. autoattribute:: cuda.bindings.runtime.cudaDevResourceType.cudaDevResourceTypeInvalid


    .. autoattribute:: cuda.bindings.runtime.cudaDevResourceType.cudaDevResourceTypeSm


        Streaming multiprocessors related information


    .. autoattribute:: cuda.bindings.runtime.cudaDevResourceType.cudaDevResourceTypeWorkqueueConfig


        Workqueue configuration related information


    .. autoattribute:: cuda.bindings.runtime.cudaDevResourceType.cudaDevResourceTypeWorkqueue


        Pre-existing workqueue related information

.. autoclass:: cuda.bindings.runtime.cudaDevWorkqueueConfigScope

    .. autoattribute:: cuda.bindings.runtime.cudaDevWorkqueueConfigScope.cudaDevWorkqueueConfigScopeDeviceCtx


        Use all shared workqueue resources on the device. Default driver behaviour.


    .. autoattribute:: cuda.bindings.runtime.cudaDevWorkqueueConfigScope.cudaDevWorkqueueConfigScopeGreenCtxBalanced


        When possible, use non-overlapping workqueue resources with other balanced green contexts.

.. autoclass:: cuda.bindings.runtime.cudaJitOption

    .. autoattribute:: cuda.bindings.runtime.cudaJitOption.cudaJitMaxRegisters


        Max number of registers that a thread may use.

        Option type: unsigned int

        Applies to: compiler only


    .. autoattribute:: cuda.bindings.runtime.cudaJitOption.cudaJitThreadsPerBlock


        IN: Specifies minimum number of threads per block to target compilation for

        OUT: Returns the number of threads the compiler actually targeted. This restricts the resource utilization of the compiler (e.g. max registers) such that a block with the given number of threads should be able to launch based on register limitations. Note, this option does not currently take into account any other resource limitations, such as shared memory utilization.

        Option type: unsigned int

        Applies to: compiler only


    .. autoattribute:: cuda.bindings.runtime.cudaJitOption.cudaJitWallTime


        Overwrites the option value with the total wall clock time, in milliseconds, spent in the compiler and linker

        Option type: float

        Applies to: compiler and linker


    .. autoattribute:: cuda.bindings.runtime.cudaJitOption.cudaJitInfoLogBuffer


        Pointer to a buffer in which to print any log messages that are informational in nature (the buffer size is specified via option :py:obj:`~.cudaJitInfoLogBufferSizeBytes`)

        Option type: char *

        Applies to: compiler and linker


    .. autoattribute:: cuda.bindings.runtime.cudaJitOption.cudaJitInfoLogBufferSizeBytes


        IN: Log buffer size in bytes. Log messages will be capped at this size (including null terminator)

        OUT: Amount of log buffer filled with messages

        Option type: unsigned int

        Applies to: compiler and linker


    .. autoattribute:: cuda.bindings.runtime.cudaJitOption.cudaJitErrorLogBuffer


        Pointer to a buffer in which to print any log messages that reflect errors (the buffer size is specified via option :py:obj:`~.cudaJitErrorLogBufferSizeBytes`)

        Option type: char *

        Applies to: compiler and linker


    .. autoattribute:: cuda.bindings.runtime.cudaJitOption.cudaJitErrorLogBufferSizeBytes


        IN: Log buffer size in bytes. Log messages will be capped at this size (including null terminator)

        OUT: Amount of log buffer filled with messages

        Option type: unsigned int

        Applies to: compiler and linker


    .. autoattribute:: cuda.bindings.runtime.cudaJitOption.cudaJitOptimizationLevel


        Level of optimizations to apply to generated code (0 - 4), with 4 being the default and highest level of optimizations.

        Option type: unsigned int

        Applies to: compiler only


    .. autoattribute:: cuda.bindings.runtime.cudaJitOption.cudaJitFallbackStrategy


        Specifies choice of fallback strategy if matching cubin is not found. Choice is based on supplied :py:obj:`~.cudaJit_Fallback`. Option type: unsigned int for enumerated type :py:obj:`~.cudaJit_Fallback`

        Applies to: compiler only


    .. autoattribute:: cuda.bindings.runtime.cudaJitOption.cudaJitGenerateDebugInfo


        Specifies whether to create debug information in output (-g) (0: false, default)

        Option type: int

        Applies to: compiler and linker


    .. autoattribute:: cuda.bindings.runtime.cudaJitOption.cudaJitLogVerbose


        Generate verbose log messages (0: false, default)

        Option type: int

        Applies to: compiler and linker


    .. autoattribute:: cuda.bindings.runtime.cudaJitOption.cudaJitGenerateLineInfo


        Generate line number information (-lineinfo) (0: false, default)

        Option type: int

        Applies to: compiler only


    .. autoattribute:: cuda.bindings.runtime.cudaJitOption.cudaJitCacheMode


        Specifies whether to enable caching explicitly (-dlcm) 

        Choice is based on supplied :py:obj:`~.cudaJit_CacheMode`.

        Option type: unsigned int for enumerated type :py:obj:`~.cudaJit_CacheMode`

        Applies to: compiler only


    .. autoattribute:: cuda.bindings.runtime.cudaJitOption.cudaJitPositionIndependentCode


        Generate position independent code (0: false)

        Option type: int

        Applies to: compiler only


    .. autoattribute:: cuda.bindings.runtime.cudaJitOption.cudaJitMinCtaPerSm


        This option hints to the JIT compiler the minimum number of CTAs from the kernel’s grid to be mapped to a SM. This option is ignored when used together with :py:obj:`~.cudaJitMaxRegisters` or :py:obj:`~.cudaJitThreadsPerBlock`. Optimizations based on this option need :py:obj:`~.cudaJitMaxThreadsPerBlock` to be specified as well. For kernels already using PTX directive .minnctapersm, this option will be ignored by default. Use :py:obj:`~.cudaJitOverrideDirectiveValues` to let this option take precedence over the PTX directive. Option type: unsigned int

        Applies to: compiler only


    .. autoattribute:: cuda.bindings.runtime.cudaJitOption.cudaJitMaxThreadsPerBlock


        Maximum number threads in a thread block, computed as the product of the maximum extent specifed for each dimension of the block. This limit is guaranteed not to be exeeded in any invocation of the kernel. Exceeding the the maximum number of threads results in runtime error or kernel launch failure. For kernels already using PTX directive .maxntid, this option will be ignored by default. Use :py:obj:`~.cudaJitOverrideDirectiveValues` to let this option take precedence over the PTX directive. Option type: int

        Applies to: compiler only


    .. autoattribute:: cuda.bindings.runtime.cudaJitOption.cudaJitOverrideDirectiveValues


        This option lets the values specified using :py:obj:`~.cudaJitMaxRegisters`, :py:obj:`~.cudaJitThreadsPerBlock`, :py:obj:`~.cudaJitMaxThreadsPerBlock` and :py:obj:`~.cudaJitMinCtaPerSm` take precedence over any PTX directives. (0: Disable, default; 1: Enable) Option type: int

        Applies to: compiler only

.. autoclass:: cuda.bindings.runtime.cudaLibraryOption

    .. autoattribute:: cuda.bindings.runtime.cudaLibraryOption.cudaLibraryHostUniversalFunctionAndDataTable


    .. autoattribute:: cuda.bindings.runtime.cudaLibraryOption.cudaLibraryBinaryIsPreserved


        Specifes that the argument `code` passed to :py:obj:`~.cudaLibraryLoadData()` will be preserved. Specifying this option will let the driver know that `code` can be accessed at any point until :py:obj:`~.cudaLibraryUnload()`. The default behavior is for the driver to allocate and maintain its own copy of `code`. Note that this is only a memory usage optimization hint and the driver can choose to ignore it if required. Specifying this option with :py:obj:`~.cudaLibraryLoadFromFile()` is invalid and will return :py:obj:`~.cudaErrorInvalidValue`.

.. autoclass:: cuda.bindings.runtime.cudaJit_CacheMode

    .. autoattribute:: cuda.bindings.runtime.cudaJit_CacheMode.cudaJitCacheOptionNone


        Compile with no -dlcm flag specified


    .. autoattribute:: cuda.bindings.runtime.cudaJit_CacheMode.cudaJitCacheOptionCG


        Compile with L1 cache disabled


    .. autoattribute:: cuda.bindings.runtime.cudaJit_CacheMode.cudaJitCacheOptionCA


        Compile with L1 cache enabled

.. autoclass:: cuda.bindings.runtime.cudaJit_Fallback

    .. autoattribute:: cuda.bindings.runtime.cudaJit_Fallback.cudaPreferPtx


        Prefer to compile ptx if exact binary match not found


    .. autoattribute:: cuda.bindings.runtime.cudaJit_Fallback.cudaPreferBinary


        Prefer to fall back to compatible binary code if exact match not found

.. autoclass:: cuda.bindings.runtime.cudaCGScope

    .. autoattribute:: cuda.bindings.runtime.cudaCGScope.cudaCGScopeInvalid


        Invalid cooperative group scope


    .. autoattribute:: cuda.bindings.runtime.cudaCGScope.cudaCGScopeGrid


        Scope represented by a grid_group


    .. autoattribute:: cuda.bindings.runtime.cudaCGScope.cudaCGScopeReserved


        Reserved

.. autoclass:: cuda.bindings.runtime.cudaGraphConditionalHandleFlags

    .. autoattribute:: cuda.bindings.runtime.cudaGraphConditionalHandleFlags.cudaGraphCondAssignDefault


        Apply default handle value when graph is launched.

.. autoclass:: cuda.bindings.runtime.cudaGraphConditionalNodeType

    .. autoattribute:: cuda.bindings.runtime.cudaGraphConditionalNodeType.cudaGraphCondTypeIf


        Conditional 'if/else' Node. Body[0] executed if condition is non-zero. If `size` == 2, an optional ELSE graph is created and this is executed if the condition is zero.


    .. autoattribute:: cuda.bindings.runtime.cudaGraphConditionalNodeType.cudaGraphCondTypeWhile


        Conditional 'while' Node. Body executed repeatedly while condition value is non-zero.


    .. autoattribute:: cuda.bindings.runtime.cudaGraphConditionalNodeType.cudaGraphCondTypeSwitch


        Conditional 'switch' Node. Body[n] is executed once, where 'n' is the value of the condition. If the condition does not match a body index, no body is launched.

.. autoclass:: cuda.bindings.runtime.cudaGraphNodeType

    .. autoattribute:: cuda.bindings.runtime.cudaGraphNodeType.cudaGraphNodeTypeKernel


        GPU kernel node


    .. autoattribute:: cuda.bindings.runtime.cudaGraphNodeType.cudaGraphNodeTypeMemcpy


        Memcpy node


    .. autoattribute:: cuda.bindings.runtime.cudaGraphNodeType.cudaGraphNodeTypeMemset


        Memset node


    .. autoattribute:: cuda.bindings.runtime.cudaGraphNodeType.cudaGraphNodeTypeHost


        Host (executable) node


    .. autoattribute:: cuda.bindings.runtime.cudaGraphNodeType.cudaGraphNodeTypeGraph


        Node which executes an embedded graph


    .. autoattribute:: cuda.bindings.runtime.cudaGraphNodeType.cudaGraphNodeTypeEmpty


        Empty (no-op) node


    .. autoattribute:: cuda.bindings.runtime.cudaGraphNodeType.cudaGraphNodeTypeWaitEvent


        External event wait node


    .. autoattribute:: cuda.bindings.runtime.cudaGraphNodeType.cudaGraphNodeTypeEventRecord


        External event record node


    .. autoattribute:: cuda.bindings.runtime.cudaGraphNodeType.cudaGraphNodeTypeExtSemaphoreSignal


        External semaphore signal node


    .. autoattribute:: cuda.bindings.runtime.cudaGraphNodeType.cudaGraphNodeTypeExtSemaphoreWait


        External semaphore wait node


    .. autoattribute:: cuda.bindings.runtime.cudaGraphNodeType.cudaGraphNodeTypeMemAlloc


        Memory allocation node


    .. autoattribute:: cuda.bindings.runtime.cudaGraphNodeType.cudaGraphNodeTypeMemFree


        Memory free node


    .. autoattribute:: cuda.bindings.runtime.cudaGraphNodeType.cudaGraphNodeTypeConditional


        Conditional node                                    May be used to implement a conditional execution path or loop

                                           inside of a graph. The graph(s) contained within the body of the conditional node

                                           can be selectively executed or iterated upon based on the value of a conditional

                                           variable.



                                           Handles must be created in advance of creating the node

                                           using :py:obj:`~.cudaGraphConditionalHandleCreate`.



                                           The following restrictions apply to graphs which contain conditional nodes:

                                             The graph cannot be used in a child node.

                                             Only one instantiation of the graph may exist at any point in time.

                                             The graph cannot be cloned.



                                           To set the control value, supply a default value when creating the handle and/or

                                           call :py:obj:`~.cudaGraphSetConditional` from device code.


    .. autoattribute:: cuda.bindings.runtime.cudaGraphNodeType.cudaGraphNodeTypeCount

.. autoclass:: cuda.bindings.runtime.cudaGraphChildGraphNodeOwnership

    .. autoattribute:: cuda.bindings.runtime.cudaGraphChildGraphNodeOwnership.cudaGraphChildGraphOwnershipClone


        Default behavior for a child graph node. Child graph is cloned into the parent and memory allocation/free nodes can't be present in the child graph.


    .. autoattribute:: cuda.bindings.runtime.cudaGraphChildGraphNodeOwnership.cudaGraphChildGraphOwnershipMove


        The child graph is moved to the parent. The handle to the child graph is owned by the parent and will be destroyed when the parent is destroyed.



        The following restrictions apply to child graphs after they have been moved: Cannot be independently instantiated or destroyed; Cannot be added as a child graph of a separate parent graph; Cannot be used as an argument to cudaGraphExecUpdate; Cannot have additional memory allocation or free nodes added.

.. autoclass:: cuda.bindings.runtime.cudaGraphDependencyType

    .. autoattribute:: cuda.bindings.runtime.cudaGraphDependencyType.cudaGraphDependencyTypeDefault


        This is an ordinary dependency.


    .. autoattribute:: cuda.bindings.runtime.cudaGraphDependencyType.cudaGraphDependencyTypeProgrammatic


        This dependency type allows the downstream node to use `cudaGridDependencySynchronize()`. It may only be used between kernel nodes, and must be used with either the :py:obj:`~.cudaGraphKernelNodePortProgrammatic` or :py:obj:`~.cudaGraphKernelNodePortLaunchCompletion` outgoing port.

.. autoclass:: cuda.bindings.runtime.cudaGraphExecUpdateResult

    .. autoattribute:: cuda.bindings.runtime.cudaGraphExecUpdateResult.cudaGraphExecUpdateSuccess


        The update succeeded


    .. autoattribute:: cuda.bindings.runtime.cudaGraphExecUpdateResult.cudaGraphExecUpdateError


        The update failed for an unexpected reason which is described in the return value of the function


    .. autoattribute:: cuda.bindings.runtime.cudaGraphExecUpdateResult.cudaGraphExecUpdateErrorTopologyChanged


        The update failed because the topology changed


    .. autoattribute:: cuda.bindings.runtime.cudaGraphExecUpdateResult.cudaGraphExecUpdateErrorNodeTypeChanged


        The update failed because a node type changed


    .. autoattribute:: cuda.bindings.runtime.cudaGraphExecUpdateResult.cudaGraphExecUpdateErrorFunctionChanged


        The update failed because the function of a kernel node changed (CUDA driver < 11.2)


    .. autoattribute:: cuda.bindings.runtime.cudaGraphExecUpdateResult.cudaGraphExecUpdateErrorParametersChanged


        The update failed because the parameters changed in a way that is not supported


    .. autoattribute:: cuda.bindings.runtime.cudaGraphExecUpdateResult.cudaGraphExecUpdateErrorNotSupported


        The update failed because something about the node is not supported


    .. autoattribute:: cuda.bindings.runtime.cudaGraphExecUpdateResult.cudaGraphExecUpdateErrorUnsupportedFunctionChange


        The update failed because the function of a kernel node changed in an unsupported way


    .. autoattribute:: cuda.bindings.runtime.cudaGraphExecUpdateResult.cudaGraphExecUpdateErrorAttributesChanged


        The update failed because the node attributes changed in a way that is not supported

.. autoclass:: cuda.bindings.runtime.cudaGraphInstantiateResult

    .. autoattribute:: cuda.bindings.runtime.cudaGraphInstantiateResult.cudaGraphInstantiateSuccess


        Instantiation succeeded


    .. autoattribute:: cuda.bindings.runtime.cudaGraphInstantiateResult.cudaGraphInstantiateError


        Instantiation failed for an unexpected reason which is described in the return value of the function


    .. autoattribute:: cuda.bindings.runtime.cudaGraphInstantiateResult.cudaGraphInstantiateInvalidStructure


        Instantiation failed due to invalid structure, such as cycles


    .. autoattribute:: cuda.bindings.runtime.cudaGraphInstantiateResult.cudaGraphInstantiateNodeOperationNotSupported


        Instantiation for device launch failed because the graph contained an unsupported operation


    .. autoattribute:: cuda.bindings.runtime.cudaGraphInstantiateResult.cudaGraphInstantiateMultipleDevicesNotSupported


        Instantiation for device launch failed due to the nodes belonging to different contexts


    .. autoattribute:: cuda.bindings.runtime.cudaGraphInstantiateResult.cudaGraphInstantiateConditionalHandleUnused


        One or more conditional handles are not associated with conditional nodes

.. autoclass:: cuda.bindings.runtime.cudaGraphKernelNodeField

    .. autoattribute:: cuda.bindings.runtime.cudaGraphKernelNodeField.cudaGraphKernelNodeFieldInvalid


        Invalid field


    .. autoattribute:: cuda.bindings.runtime.cudaGraphKernelNodeField.cudaGraphKernelNodeFieldGridDim


        Grid dimension update


    .. autoattribute:: cuda.bindings.runtime.cudaGraphKernelNodeField.cudaGraphKernelNodeFieldParam


        Kernel parameter update


    .. autoattribute:: cuda.bindings.runtime.cudaGraphKernelNodeField.cudaGraphKernelNodeFieldEnabled


        Node enable/disable

.. autoclass:: cuda.bindings.runtime.cudaGetDriverEntryPointFlags

    .. autoattribute:: cuda.bindings.runtime.cudaGetDriverEntryPointFlags.cudaEnableDefault


        Default search mode for driver symbols.


    .. autoattribute:: cuda.bindings.runtime.cudaGetDriverEntryPointFlags.cudaEnableLegacyStream


        Search for legacy versions of driver symbols.


    .. autoattribute:: cuda.bindings.runtime.cudaGetDriverEntryPointFlags.cudaEnablePerThreadDefaultStream


        Search for per-thread versions of driver symbols.

.. autoclass:: cuda.bindings.runtime.cudaDriverEntryPointQueryResult

    .. autoattribute:: cuda.bindings.runtime.cudaDriverEntryPointQueryResult.cudaDriverEntryPointSuccess


        Search for symbol found a match


    .. autoattribute:: cuda.bindings.runtime.cudaDriverEntryPointQueryResult.cudaDriverEntryPointSymbolNotFound


        Search for symbol was not found


    .. autoattribute:: cuda.bindings.runtime.cudaDriverEntryPointQueryResult.cudaDriverEntryPointVersionNotSufficent


        Search for symbol was found but version wasn't great enough

.. autoclass:: cuda.bindings.runtime.cudaGraphDebugDotFlags

    .. autoattribute:: cuda.bindings.runtime.cudaGraphDebugDotFlags.cudaGraphDebugDotFlagsVerbose


        Output all debug data as if every debug flag is enabled


    .. autoattribute:: cuda.bindings.runtime.cudaGraphDebugDotFlags.cudaGraphDebugDotFlagsKernelNodeParams


        Adds :py:obj:`~.cudaKernelNodeParams` to output


    .. autoattribute:: cuda.bindings.runtime.cudaGraphDebugDotFlags.cudaGraphDebugDotFlagsMemcpyNodeParams


        Adds :py:obj:`~.cudaMemcpy3DParms` to output


    .. autoattribute:: cuda.bindings.runtime.cudaGraphDebugDotFlags.cudaGraphDebugDotFlagsMemsetNodeParams


        Adds :py:obj:`~.cudaMemsetParams` to output


    .. autoattribute:: cuda.bindings.runtime.cudaGraphDebugDotFlags.cudaGraphDebugDotFlagsHostNodeParams


        Adds :py:obj:`~.cudaHostNodeParams` to output


    .. autoattribute:: cuda.bindings.runtime.cudaGraphDebugDotFlags.cudaGraphDebugDotFlagsEventNodeParams


        Adds cudaEvent_t handle from record and wait nodes to output


    .. autoattribute:: cuda.bindings.runtime.cudaGraphDebugDotFlags.cudaGraphDebugDotFlagsExtSemasSignalNodeParams


        Adds :py:obj:`~.cudaExternalSemaphoreSignalNodeParams` values to output


    .. autoattribute:: cuda.bindings.runtime.cudaGraphDebugDotFlags.cudaGraphDebugDotFlagsExtSemasWaitNodeParams


        Adds :py:obj:`~.cudaExternalSemaphoreWaitNodeParams` to output


    .. autoattribute:: cuda.bindings.runtime.cudaGraphDebugDotFlags.cudaGraphDebugDotFlagsKernelNodeAttributes


        Adds cudaKernelNodeAttrID values to output


    .. autoattribute:: cuda.bindings.runtime.cudaGraphDebugDotFlags.cudaGraphDebugDotFlagsHandles


        Adds node handles and every kernel function handle to output


    .. autoattribute:: cuda.bindings.runtime.cudaGraphDebugDotFlags.cudaGraphDebugDotFlagsConditionalNodeParams


        Adds :py:obj:`~.cudaConditionalNodeParams` to output

.. autoclass:: cuda.bindings.runtime.cudaGraphInstantiateFlags

    .. autoattribute:: cuda.bindings.runtime.cudaGraphInstantiateFlags.cudaGraphInstantiateFlagAutoFreeOnLaunch


        Automatically free memory allocated in a graph before relaunching.


    .. autoattribute:: cuda.bindings.runtime.cudaGraphInstantiateFlags.cudaGraphInstantiateFlagUpload


        Automatically upload the graph after instantiation. Only supported by 

         :py:obj:`~.cudaGraphInstantiateWithParams`. The upload will be performed using the 

         stream provided in `instantiateParams`.


    .. autoattribute:: cuda.bindings.runtime.cudaGraphInstantiateFlags.cudaGraphInstantiateFlagDeviceLaunch


        Instantiate the graph to be launchable from the device. This flag can only 

         be used on platforms which support unified addressing. This flag cannot be 

         used in conjunction with cudaGraphInstantiateFlagAutoFreeOnLaunch.


    .. autoattribute:: cuda.bindings.runtime.cudaGraphInstantiateFlags.cudaGraphInstantiateFlagUseNodePriority


        Run the graph using the per-node priority attributes rather than the priority of the stream it is launched into.

.. autoclass:: cuda.bindings.runtime.cudaLaunchMemSyncDomain

    .. autoattribute:: cuda.bindings.runtime.cudaLaunchMemSyncDomain.cudaLaunchMemSyncDomainDefault


        Launch kernels in the default domain


    .. autoattribute:: cuda.bindings.runtime.cudaLaunchMemSyncDomain.cudaLaunchMemSyncDomainRemote


        Launch kernels in the remote domain

.. autoclass:: cuda.bindings.runtime.cudaLaunchAttributeID

    .. autoattribute:: cuda.bindings.runtime.cudaLaunchAttributeID.cudaLaunchAttributeIgnore


        Ignored entry, for convenient composition


    .. autoattribute:: cuda.bindings.runtime.cudaLaunchAttributeID.cudaLaunchAttributeAccessPolicyWindow


        Valid for streams, graph nodes, launches. See :py:obj:`~.cudaLaunchAttributeValue.accessPolicyWindow`.


    .. autoattribute:: cuda.bindings.runtime.cudaLaunchAttributeID.cudaLaunchAttributeCooperative


        Valid for graph nodes, launches. See :py:obj:`~.cudaLaunchAttributeValue.cooperative`.


    .. autoattribute:: cuda.bindings.runtime.cudaLaunchAttributeID.cudaLaunchAttributeSynchronizationPolicy


        Valid for streams. See :py:obj:`~.cudaLaunchAttributeValue.syncPolicy`.


    .. autoattribute:: cuda.bindings.runtime.cudaLaunchAttributeID.cudaLaunchAttributeClusterDimension


        Valid for graph nodes, launches. See :py:obj:`~.cudaLaunchAttributeValue.clusterDim`.


    .. autoattribute:: cuda.bindings.runtime.cudaLaunchAttributeID.cudaLaunchAttributeClusterSchedulingPolicyPreference


        Valid for graph nodes, launches. See :py:obj:`~.cudaLaunchAttributeValue.clusterSchedulingPolicyPreference`.


    .. autoattribute:: cuda.bindings.runtime.cudaLaunchAttributeID.cudaLaunchAttributeProgrammaticStreamSerialization


        Valid for launches. Setting :py:obj:`~.cudaLaunchAttributeValue.programmaticStreamSerializationAllowed` to non-0 signals that the kernel will use programmatic means to resolve its stream dependency, so that the CUDA runtime should opportunistically allow the grid's execution to overlap with the previous kernel in the stream, if that kernel requests the overlap. The dependent launches can choose to wait on the dependency using the programmatic sync (cudaGridDependencySynchronize() or equivalent PTX instructions).


    .. autoattribute:: cuda.bindings.runtime.cudaLaunchAttributeID.cudaLaunchAttributeProgrammaticEvent


        Valid for launches. Set :py:obj:`~.cudaLaunchAttributeValue.programmaticEvent` to record the event. Event recorded through this launch attribute is guaranteed to only trigger after all block in the associated kernel trigger the event. A block can trigger the event programmatically in a future CUDA release. A trigger can also be inserted at the beginning of each block's execution if triggerAtBlockStart is set to non-0. The dependent launches can choose to wait on the dependency using the programmatic sync (cudaGridDependencySynchronize() or equivalent PTX instructions). Note that dependents (including the CPU thread calling :py:obj:`~.cudaEventSynchronize()`) are not guaranteed to observe the release precisely when it is released. For example, :py:obj:`~.cudaEventSynchronize()` may only observe the event trigger long after the associated kernel has completed. This recording type is primarily meant for establishing programmatic dependency between device tasks. Note also this type of dependency allows, but does not guarantee, concurrent execution of tasks. 

         The event supplied must not be an interprocess or interop event. The event must disable timing (i.e. must be created with the :py:obj:`~.cudaEventDisableTiming` flag set).


    .. autoattribute:: cuda.bindings.runtime.cudaLaunchAttributeID.cudaLaunchAttributePriority


        Valid for streams, graph nodes, launches. See :py:obj:`~.cudaLaunchAttributeValue.priority`.


    .. autoattribute:: cuda.bindings.runtime.cudaLaunchAttributeID.cudaLaunchAttributeMemSyncDomainMap


        Valid for streams, graph nodes, launches. See :py:obj:`~.cudaLaunchAttributeValue.memSyncDomainMap`.


    .. autoattribute:: cuda.bindings.runtime.cudaLaunchAttributeID.cudaLaunchAttributeMemSyncDomain


        Valid for streams, graph nodes, launches. See :py:obj:`~.cudaLaunchAttributeValue.memSyncDomain`.


    .. autoattribute:: cuda.bindings.runtime.cudaLaunchAttributeID.cudaLaunchAttributePreferredClusterDimension


        Valid for graph nodes and launches. Set :py:obj:`~.cudaLaunchAttributeValue.preferredClusterDim` to allow the kernel launch to specify a preferred substitute cluster dimension. Blocks may be grouped according to either the dimensions specified with this attribute (grouped into a "preferred substitute cluster"), or the one specified with :py:obj:`~.cudaLaunchAttributeClusterDimension` attribute (grouped into a "regular cluster"). The cluster dimensions of a "preferred substitute cluster" shall be an integer multiple greater than zero of the regular cluster dimensions. The device will attempt - on a best-effort basis - to group thread blocks into preferred clusters over grouping them into regular clusters. When it deems necessary (primarily when the device temporarily runs out of physical resources to launch the larger preferred clusters), the device may switch to launch the regular clusters instead to attempt to utilize as much of the physical device resources as possible. 

         Each type of cluster will have its enumeration / coordinate setup as if the grid consists solely of its type of cluster. For example, if the preferred substitute cluster dimensions double the regular cluster dimensions, there might be simultaneously a regular cluster indexed at (1,0,0), and a preferred cluster indexed at (1,0,0). In this example, the preferred substitute cluster (1,0,0) replaces regular clusters (2,0,0) and (3,0,0) and groups their blocks. 

         This attribute will only take effect when a regular cluster dimension has been specified. The preferred substitute cluster dimension must be an integer multiple greater than zero of the regular cluster dimension and must divide the grid. It must also be no more than `maxBlocksPerCluster`, if it is set in the kernel's `__launch_bounds__`. Otherwise it must be less than the maximum value the driver can support. Otherwise, setting this attribute to a value physically unable to fit on any particular device is permitted.


    .. autoattribute:: cuda.bindings.runtime.cudaLaunchAttributeID.cudaLaunchAttributeLaunchCompletionEvent


        Valid for launches. Set :py:obj:`~.cudaLaunchAttributeValue.launchCompletionEvent` to record the event. 

         Nominally, the event is triggered once all blocks of the kernel have begun execution. Currently this is a best effort. If a kernel B has a launch completion dependency on a kernel A, B may wait until A is complete. Alternatively, blocks of B may begin before all blocks of A have begun, for example if B can claim execution resources unavailable to A (e.g. they run on different GPUs) or if B is a higher priority than A. Exercise caution if such an ordering inversion could lead to deadlock. 

         A launch completion event is nominally similar to a programmatic event with `triggerAtBlockStart` set except that it is not visible to `cudaGridDependencySynchronize()` and can be used with compute capability less than 9.0. 

         The event supplied must not be an interprocess or interop event. The event must disable timing (i.e. must be created with the :py:obj:`~.cudaEventDisableTiming` flag set).


    .. autoattribute:: cuda.bindings.runtime.cudaLaunchAttributeID.cudaLaunchAttributeDeviceUpdatableKernelNode


        Valid for graph nodes, launches. This attribute is graphs-only, and passing it to a launch in a non-capturing stream will result in an error. 

         :cudaLaunchAttributeValue::deviceUpdatableKernelNode::deviceUpdatable can only be set to 0 or 1. Setting the field to 1 indicates that the corresponding kernel node should be device-updatable. On success, a handle will be returned via :py:obj:`~.cudaLaunchAttributeValue`::deviceUpdatableKernelNode::devNode which can be passed to the various device-side update functions to update the node's kernel parameters from within another kernel. For more information on the types of device updates that can be made, as well as the relevant limitations thereof, see :py:obj:`~.cudaGraphKernelNodeUpdatesApply`. 

         Nodes which are device-updatable have additional restrictions compared to regular kernel nodes. Firstly, device-updatable nodes cannot be removed from their graph via :py:obj:`~.cudaGraphDestroyNode`. Additionally, once opted-in to this functionality, a node cannot opt out, and any attempt to set the deviceUpdatable attribute to 0 will result in an error. Device-updatable kernel nodes also cannot have their attributes copied to/from another kernel node via :py:obj:`~.cudaGraphKernelNodeCopyAttributes`. Graphs containing one or more device-updatable nodes also do not allow multiple instantiation, and neither the graph nor its instantiated version can be passed to :py:obj:`~.cudaGraphExecUpdate`. 

         If a graph contains device-updatable nodes and updates those nodes from the device from within the graph, the graph must be uploaded with :py:obj:`~.cuGraphUpload` before it is launched. For such a graph, if host-side executable graph updates are made to the device-updatable nodes, the graph must be uploaded before it is launched again.


    .. autoattribute:: cuda.bindings.runtime.cudaLaunchAttributeID.cudaLaunchAttributePreferredSharedMemoryCarveout


        Valid for launches. On devices where the L1 cache and shared memory use the same hardware resources, setting :py:obj:`~.cudaLaunchAttributeValue.sharedMemCarveout` to a percentage between 0-100 signals sets the shared memory carveout preference in percent of the total shared memory for that kernel launch. This attribute takes precedence over :py:obj:`~.cudaFuncAttributePreferredSharedMemoryCarveout`. This is only a hint, and the driver can choose a different configuration if required for the launch.


    .. autoattribute:: cuda.bindings.runtime.cudaLaunchAttributeID.cudaLaunchAttributeNvlinkUtilCentricScheduling


        Valid for streams, graph nodes, launches. This attribute is a hint to the CUDA runtime that the launch should attempt to make the kernel maximize its NVLINK utilization. 



         When possible to honor this hint, CUDA will assume each block in the grid launch will carry out an even amount of NVLINK traffic, and make a best-effort attempt to adjust the kernel launch based on that assumption. 

         This attribute is a hint only. CUDA makes no functional or performance guarantee. Its applicability can be affected by many different factors, including driver version (i.e. CUDA doesn't guarantee the performance characteristics will be maintained between driver versions or a driver update could alter or regress previously observed perf characteristics.) It also doesn't guarantee a successful result, i.e. applying the attribute may not improve the performance of either the targeted kernel or the encapsulating application. 

         Valid values for :py:obj:`~.cudaLaunchAttributeValue.nvlinkUtilCentricScheduling` are 0 (disabled) and 1 (enabled).

.. autoclass:: cuda.bindings.runtime.cudaDeviceNumaConfig

    .. autoattribute:: cuda.bindings.runtime.cudaDeviceNumaConfig.cudaDeviceNumaConfigNone


        The GPU is not a NUMA node


    .. autoattribute:: cuda.bindings.runtime.cudaDeviceNumaConfig.cudaDeviceNumaConfigNumaNode


        The GPU is a NUMA node, cudaDevAttrNumaId contains its NUMA ID

.. autoclass:: cuda.bindings.runtime.cudaAsyncNotificationType

    .. autoattribute:: cuda.bindings.runtime.cudaAsyncNotificationType.cudaAsyncNotificationTypeOverBudget


        Sent when the process has exceeded its device memory budget

.. autoclass:: cuda.bindings.runtime.cudaLogLevel

    .. autoattribute:: cuda.bindings.runtime.cudaLogLevel.cudaLogLevelError


    .. autoattribute:: cuda.bindings.runtime.cudaLogLevel.cudaLogLevelWarning

.. autoclass:: cuda.bindings.runtime.cudaSurfaceBoundaryMode

    .. autoattribute:: cuda.bindings.runtime.cudaSurfaceBoundaryMode.cudaBoundaryModeZero


        Zero boundary mode


    .. autoattribute:: cuda.bindings.runtime.cudaSurfaceBoundaryMode.cudaBoundaryModeClamp


        Clamp boundary mode


    .. autoattribute:: cuda.bindings.runtime.cudaSurfaceBoundaryMode.cudaBoundaryModeTrap


        Trap boundary mode

.. autoclass:: cuda.bindings.runtime.cudaSurfaceFormatMode

    .. autoattribute:: cuda.bindings.runtime.cudaSurfaceFormatMode.cudaFormatModeForced


        Forced format mode


    .. autoattribute:: cuda.bindings.runtime.cudaSurfaceFormatMode.cudaFormatModeAuto


        Auto format mode

.. autoclass:: cuda.bindings.runtime.cudaTextureAddressMode

    .. autoattribute:: cuda.bindings.runtime.cudaTextureAddressMode.cudaAddressModeWrap


        Wrapping address mode


    .. autoattribute:: cuda.bindings.runtime.cudaTextureAddressMode.cudaAddressModeClamp


        Clamp to edge address mode


    .. autoattribute:: cuda.bindings.runtime.cudaTextureAddressMode.cudaAddressModeMirror


        Mirror address mode


    .. autoattribute:: cuda.bindings.runtime.cudaTextureAddressMode.cudaAddressModeBorder


        Border address mode

.. autoclass:: cuda.bindings.runtime.cudaTextureFilterMode

    .. autoattribute:: cuda.bindings.runtime.cudaTextureFilterMode.cudaFilterModePoint


        Point filter mode


    .. autoattribute:: cuda.bindings.runtime.cudaTextureFilterMode.cudaFilterModeLinear


        Linear filter mode

.. autoclass:: cuda.bindings.runtime.cudaTextureReadMode

    .. autoattribute:: cuda.bindings.runtime.cudaTextureReadMode.cudaReadModeElementType


        Read texture as specified element type


    .. autoattribute:: cuda.bindings.runtime.cudaTextureReadMode.cudaReadModeNormalizedFloat


        Read texture as normalized float

.. autoclass:: cuda.bindings.runtime.cudaEglPlaneDesc
.. autoclass:: cuda.bindings.runtime.cudaEglFrame
.. autoclass:: cuda.bindings.runtime.cudaEglStreamConnection
.. autoclass:: cuda.bindings.runtime.cudaDevResourceDesc_t
.. autoclass:: cuda.bindings.runtime.cudaExecutionContext_t
.. autoclass:: cuda.bindings.runtime.cudaArray_t
.. autoclass:: cuda.bindings.runtime.cudaArray_const_t
.. autoclass:: cuda.bindings.runtime.cudaMipmappedArray_t
.. autoclass:: cuda.bindings.runtime.cudaMipmappedArray_const_t
.. autoclass:: cuda.bindings.runtime.cudaHostFn_t
.. autoclass:: cuda.bindings.runtime.CUuuid
.. autoclass:: cuda.bindings.runtime.cudaUUID_t
.. autoclass:: cuda.bindings.runtime.cudaIpcEventHandle_t
.. autoclass:: cuda.bindings.runtime.cudaIpcMemHandle_t
.. autoclass:: cuda.bindings.runtime.cudaMemFabricHandle_t
.. autoclass:: cuda.bindings.runtime.cudaDevSmResourceGroupParams
.. autoclass:: cuda.bindings.runtime.cudaDevResource
.. autoclass:: cuda.bindings.runtime.cudaStream_t
.. autoclass:: cuda.bindings.runtime.cudaEvent_t
.. autoclass:: cuda.bindings.runtime.cudaGraphicsResource_t
.. autoclass:: cuda.bindings.runtime.cudaExternalMemory_t
.. autoclass:: cuda.bindings.runtime.cudaExternalSemaphore_t
.. autoclass:: cuda.bindings.runtime.cudaGraph_t
.. autoclass:: cuda.bindings.runtime.cudaGraphNode_t
.. autoclass:: cuda.bindings.runtime.cudaUserObject_t
.. autoclass:: cuda.bindings.runtime.cudaGraphConditionalHandle
.. autoclass:: cuda.bindings.runtime.cudaFunction_t
.. autoclass:: cuda.bindings.runtime.cudaKernel_t
.. autoclass:: cuda.bindings.runtime.cudaLibrary_t
.. autoclass:: cuda.bindings.runtime.cudaMemPool_t
.. autoclass:: cuda.bindings.runtime.cudaGraphEdgeData
.. autoclass:: cuda.bindings.runtime.cudaGraphExec_t
.. autoclass:: cuda.bindings.runtime.cudaGraphInstantiateParams
.. autoclass:: cuda.bindings.runtime.cudaGraphExecUpdateResultInfo
.. autoclass:: cuda.bindings.runtime.cudaGraphDeviceNode_t
.. autoclass:: cuda.bindings.runtime.cudaLaunchMemSyncDomainMap
.. autoclass:: cuda.bindings.runtime.cudaLaunchAttributeValue
.. autoclass:: cuda.bindings.runtime.cudaLaunchAttribute
.. autoclass:: cuda.bindings.runtime.cudaAsyncCallbackHandle_t
.. autoclass:: cuda.bindings.runtime.cudaAsyncNotificationInfo_t
.. autoclass:: cuda.bindings.runtime.cudaAsyncCallback
.. autoclass:: cuda.bindings.runtime.cudaLogsCallbackHandle
.. autoclass:: cuda.bindings.runtime.cudaLogIterator
.. autoclass:: cuda.bindings.runtime.cudaSurfaceObject_t
.. autoclass:: cuda.bindings.runtime.cudaTextureObject_t
.. autoattribute:: cuda.bindings.runtime.CUDA_EGL_MAX_PLANES

    Maximum number of planes per frame

.. autoattribute:: cuda.bindings.runtime.cudaHostAllocDefault

    Default page-locked allocation flag

.. autoattribute:: cuda.bindings.runtime.cudaHostAllocPortable

    Pinned memory accessible by all CUDA contexts

.. autoattribute:: cuda.bindings.runtime.cudaHostAllocMapped

    Map allocation into device space

.. autoattribute:: cuda.bindings.runtime.cudaHostAllocWriteCombined

    Write-combined memory

.. autoattribute:: cuda.bindings.runtime.cudaHostRegisterDefault

    Default host memory registration flag

.. autoattribute:: cuda.bindings.runtime.cudaHostRegisterPortable

    Pinned memory accessible by all CUDA contexts

.. autoattribute:: cuda.bindings.runtime.cudaHostRegisterMapped

    Map registered memory into device space

.. autoattribute:: cuda.bindings.runtime.cudaHostRegisterIoMemory

    Memory-mapped I/O space

.. autoattribute:: cuda.bindings.runtime.cudaHostRegisterReadOnly

    Memory-mapped read-only

.. autoattribute:: cuda.bindings.runtime.cudaPeerAccessDefault

    Default peer addressing enable flag

.. autoattribute:: cuda.bindings.runtime.cudaStreamDefault

    Default stream flag

.. autoattribute:: cuda.bindings.runtime.cudaStreamNonBlocking

    Stream does not synchronize with stream 0 (the NULL stream)

.. autoattribute:: cuda.bindings.runtime.cudaStreamLegacy

    Legacy stream handle



    Stream handle that can be passed as a cudaStream_t to use an implicit stream with legacy synchronization behavior.



    See details of the \link_sync_behavior

.. autoattribute:: cuda.bindings.runtime.cudaStreamPerThread

    Per-thread stream handle



    Stream handle that can be passed as a cudaStream_t to use an implicit stream with per-thread synchronization behavior.



    See details of the \link_sync_behavior

.. autoattribute:: cuda.bindings.runtime.cudaEventDefault

    Default event flag

.. autoattribute:: cuda.bindings.runtime.cudaEventBlockingSync

    Event uses blocking synchronization

.. autoattribute:: cuda.bindings.runtime.cudaEventDisableTiming

    Event will not record timing data

.. autoattribute:: cuda.bindings.runtime.cudaEventInterprocess

    Event is suitable for interprocess use. cudaEventDisableTiming must be set

.. autoattribute:: cuda.bindings.runtime.cudaEventRecordDefault

    Default event record flag

.. autoattribute:: cuda.bindings.runtime.cudaEventRecordExternal

    Event is captured in the graph as an external event node when performing stream capture

.. autoattribute:: cuda.bindings.runtime.cudaEventWaitDefault

    Default event wait flag

.. autoattribute:: cuda.bindings.runtime.cudaEventWaitExternal

    Event is captured in the graph as an external event node when performing stream capture

.. autoattribute:: cuda.bindings.runtime.cudaDeviceScheduleAuto

    Device flag - Automatic scheduling

.. autoattribute:: cuda.bindings.runtime.cudaDeviceScheduleSpin

    Device flag - Spin default scheduling

.. autoattribute:: cuda.bindings.runtime.cudaDeviceScheduleYield

    Device flag - Yield default scheduling

.. autoattribute:: cuda.bindings.runtime.cudaDeviceScheduleBlockingSync

    Device flag - Use blocking synchronization

.. autoattribute:: cuda.bindings.runtime.cudaDeviceBlockingSync

    Device flag - Use blocking synchronization [Deprecated]

.. autoattribute:: cuda.bindings.runtime.cudaDeviceScheduleMask

    Device schedule flags mask

.. autoattribute:: cuda.bindings.runtime.cudaDeviceMapHost

    Device flag - Support mapped pinned allocations

.. autoattribute:: cuda.bindings.runtime.cudaDeviceLmemResizeToMax

    Device flag - Keep local memory allocation after launch

.. autoattribute:: cuda.bindings.runtime.cudaDeviceSyncMemops

    Device flag - Ensure synchronous memory operations on this context will synchronize

.. autoattribute:: cuda.bindings.runtime.cudaDeviceMask

    Device flags mask

.. autoattribute:: cuda.bindings.runtime.cudaArrayDefault

    Default CUDA array allocation flag

.. autoattribute:: cuda.bindings.runtime.cudaArrayLayered

    Must be set in cudaMalloc3DArray to create a layered CUDA array

.. autoattribute:: cuda.bindings.runtime.cudaArraySurfaceLoadStore

    Must be set in cudaMallocArray or cudaMalloc3DArray in order to bind surfaces to the CUDA array

.. autoattribute:: cuda.bindings.runtime.cudaArrayCubemap

    Must be set in cudaMalloc3DArray to create a cubemap CUDA array

.. autoattribute:: cuda.bindings.runtime.cudaArrayTextureGather

    Must be set in cudaMallocArray or cudaMalloc3DArray in order to perform texture gather operations on the CUDA array

.. autoattribute:: cuda.bindings.runtime.cudaArrayColorAttachment

    Must be set in cudaExternalMemoryGetMappedMipmappedArray if the mipmapped array is used as a color target in a graphics API

.. autoattribute:: cuda.bindings.runtime.cudaArraySparse

    Must be set in cudaMallocArray, cudaMalloc3DArray or cudaMallocMipmappedArray in order to create a sparse CUDA array or CUDA mipmapped array

.. autoattribute:: cuda.bindings.runtime.cudaArrayDeferredMapping

    Must be set in cudaMallocArray, cudaMalloc3DArray or cudaMallocMipmappedArray in order to create a deferred mapping CUDA array or CUDA mipmapped array

.. autoattribute:: cuda.bindings.runtime.cudaIpcMemLazyEnablePeerAccess

    Automatically enable peer access between remote devices as needed

.. autoattribute:: cuda.bindings.runtime.cudaMemAttachGlobal

    Memory can be accessed by any stream on any device

.. autoattribute:: cuda.bindings.runtime.cudaMemAttachHost

    Memory cannot be accessed by any stream on any device

.. autoattribute:: cuda.bindings.runtime.cudaMemAttachSingle

    Memory can only be accessed by a single stream on the associated device

.. autoattribute:: cuda.bindings.runtime.cudaOccupancyDefault

    Default behavior

.. autoattribute:: cuda.bindings.runtime.cudaOccupancyDisableCachingOverride

    Assume global caching is enabled and cannot be automatically turned off

.. autoattribute:: cuda.bindings.runtime.cudaCpuDeviceId

    Device id that represents the CPU

.. autoattribute:: cuda.bindings.runtime.cudaInvalidDeviceId

    Device id that represents an invalid device

.. autoattribute:: cuda.bindings.runtime.cudaInitDeviceFlagsAreValid

    Tell the CUDA runtime that DeviceFlags is being set in cudaInitDevice call

.. autoattribute:: cuda.bindings.runtime.cudaArraySparsePropertiesSingleMipTail

    Indicates that the layered sparse CUDA array or CUDA mipmapped array has a single mip tail region for all layers

.. autoattribute:: cuda.bindings.runtime.CUDART_CB
.. autoattribute:: cuda.bindings.runtime.cudaMemPoolCreateUsageHwDecompress

    This flag, if set, indicates that the memory will be used as a buffer for hardware accelerated decompression.

.. autoattribute:: cuda.bindings.runtime.CU_UUID_HAS_BEEN_DEFINED

    CUDA UUID types

.. autoattribute:: cuda.bindings.runtime.CUDA_IPC_HANDLE_SIZE

    CUDA IPC Handle Size

.. autoattribute:: cuda.bindings.runtime.cudaExternalMemoryDedicated

    Indicates that the external memory object is a dedicated resource

.. autoattribute:: cuda.bindings.runtime.cudaExternalSemaphoreSignalSkipNvSciBufMemSync

    When the /p flags parameter of :py:obj:`~.cudaExternalSemaphoreSignalParams` contains this flag, it indicates that signaling an external semaphore object should skip performing appropriate memory synchronization operations over all the external memory objects that are imported as :py:obj:`~.cudaExternalMemoryHandleTypeNvSciBuf`, which otherwise are performed by default to ensure data coherency with other importers of the same NvSciBuf memory objects.

.. autoattribute:: cuda.bindings.runtime.cudaExternalSemaphoreWaitSkipNvSciBufMemSync

    When the /p flags parameter of :py:obj:`~.cudaExternalSemaphoreWaitParams` contains this flag, it indicates that waiting an external semaphore object should skip performing appropriate memory synchronization operations over all the external memory objects that are imported as :py:obj:`~.cudaExternalMemoryHandleTypeNvSciBuf`, which otherwise are performed by default to ensure data coherency with other importers of the same NvSciBuf memory objects.

.. autoattribute:: cuda.bindings.runtime.cudaNvSciSyncAttrSignal

    When /p flags of :py:obj:`~.cudaDeviceGetNvSciSyncAttributes` is set to this, it indicates that application need signaler specific NvSciSyncAttr to be filled by :py:obj:`~.cudaDeviceGetNvSciSyncAttributes`.

.. autoattribute:: cuda.bindings.runtime.cudaNvSciSyncAttrWait

    When /p flags of :py:obj:`~.cudaDeviceGetNvSciSyncAttributes` is set to this, it indicates that application need waiter specific NvSciSyncAttr to be filled by :py:obj:`~.cudaDeviceGetNvSciSyncAttributes`.

.. autoattribute:: cuda.bindings.runtime.RESOURCE_ABI_BYTES
.. autoattribute:: cuda.bindings.runtime.cudaGraphKernelNodePortDefault

    This port activates when the kernel has finished executing.

.. autoattribute:: cuda.bindings.runtime.cudaGraphKernelNodePortProgrammatic

    This port activates when all blocks of the kernel have performed cudaTriggerProgrammaticLaunchCompletion() or have terminated. It must be used with edge type :py:obj:`~.cudaGraphDependencyTypeProgrammatic`. See also :py:obj:`~.cudaLaunchAttributeProgrammaticEvent`.

.. autoattribute:: cuda.bindings.runtime.cudaGraphKernelNodePortLaunchCompletion

    This port activates when all blocks of the kernel have begun execution. See also :py:obj:`~.cudaLaunchAttributeLaunchCompletionEvent`.

.. autoattribute:: cuda.bindings.runtime.cudaStreamAttrID
.. autoattribute:: cuda.bindings.runtime.cudaStreamAttributeAccessPolicyWindow
.. autoattribute:: cuda.bindings.runtime.cudaStreamAttributeSynchronizationPolicy
.. autoattribute:: cuda.bindings.runtime.cudaStreamAttributeMemSyncDomainMap
.. autoattribute:: cuda.bindings.runtime.cudaStreamAttributeMemSyncDomain
.. autoattribute:: cuda.bindings.runtime.cudaStreamAttributePriority
.. autoattribute:: cuda.bindings.runtime.cudaStreamAttrValue
.. autoattribute:: cuda.bindings.runtime.cudaKernelNodeAttrID
.. autoattribute:: cuda.bindings.runtime.cudaKernelNodeAttributeAccessPolicyWindow
.. autoattribute:: cuda.bindings.runtime.cudaKernelNodeAttributeCooperative
.. autoattribute:: cuda.bindings.runtime.cudaKernelNodeAttributePriority
.. autoattribute:: cuda.bindings.runtime.cudaKernelNodeAttributeClusterDimension
.. autoattribute:: cuda.bindings.runtime.cudaKernelNodeAttributeClusterSchedulingPolicyPreference
.. autoattribute:: cuda.bindings.runtime.cudaKernelNodeAttributeMemSyncDomainMap
.. autoattribute:: cuda.bindings.runtime.cudaKernelNodeAttributeMemSyncDomain
.. autoattribute:: cuda.bindings.runtime.cudaKernelNodeAttributePreferredSharedMemoryCarveout
.. autoattribute:: cuda.bindings.runtime.cudaKernelNodeAttributeDeviceUpdatableKernelNode
.. autoattribute:: cuda.bindings.runtime.cudaKernelNodeAttributeNvlinkUtilCentricScheduling
.. autoattribute:: cuda.bindings.runtime.cudaKernelNodeAttrValue
.. autoattribute:: cuda.bindings.runtime.cudaSurfaceType1D
.. autoattribute:: cuda.bindings.runtime.cudaSurfaceType2D
.. autoattribute:: cuda.bindings.runtime.cudaSurfaceType3D
.. autoattribute:: cuda.bindings.runtime.cudaSurfaceTypeCubemap
.. autoattribute:: cuda.bindings.runtime.cudaSurfaceType1DLayered
.. autoattribute:: cuda.bindings.runtime.cudaSurfaceType2DLayered
.. autoattribute:: cuda.bindings.runtime.cudaSurfaceTypeCubemapLayered
.. autoattribute:: cuda.bindings.runtime.cudaTextureType1D
.. autoattribute:: cuda.bindings.runtime.cudaTextureType2D
.. autoattribute:: cuda.bindings.runtime.cudaTextureType3D
.. autoattribute:: cuda.bindings.runtime.cudaTextureTypeCubemap
.. autoattribute:: cuda.bindings.runtime.cudaTextureType1DLayered
.. autoattribute:: cuda.bindings.runtime.cudaTextureType2DLayered
.. autoattribute:: cuda.bindings.runtime.cudaTextureTypeCubemapLayered
