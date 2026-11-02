.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

.. currentmodule:: cuda.bindings._v2.driver

---------
driver_v2
---------

.. seealso::

   See the :doc:`migration guide <v2_migration>` for the differences between this module and the legacy :doc:`driver <driver>` module.

Data types used by CUDA driver
------------------------------

.. autosummary::
   :toctree: generated/driver_v2/
   :nosignatures:

   Uuid
   MemFabricHandle_v1
   IpcEventHandle_v1
   IpcMemHandle_v1
   StreamBatchMemOpParams_v1
   BatchMemOpNodeParams_v1
   BatchMemOpNodeParams_v2
   AsyncNotificationInfo
   AccessPolicyWindow_v1
   KernelNodeParams_v2
   KernelNodeParams_v3
   MemSetNodeParams_v1
   MemSetNodeParams_v2
   HostNodeParams_v1
   HostNodeParams_v2
   ConditionalNodeParams
   GraphEdgeData
   GraphInstantiateParams
   LaunchMemSyncDomainMap
   LaunchAttributeValue
   LaunchAttribute
   LaunchConfig
   ExecAffinitySmCount_v1
   ExecAffinityParam_v1
   CtxCigParam
   CtxCreateParams
   StreamCigParam
   StreamCigCaptureParams
   LibraryHostUniversalFunctionAndDataTable
   MemCpy2d_v2
   MemCpy3d_v2
   MemCpy3dPeer_v1
   MemCpyNodeParams
   ArrayDescriptor_v2
   Array3dDescriptor_v2
   ArraySparseProperties_v1
   ArrayMemoryRequirements_v1
   ResourceDesc_v1
   TextureDesc_v1
   ResourceViewDesc_v1
   TensorMap
   PointerAttributeP2PTokens_v1
   ExternalMemoryHandleDesc_v1
   ExternalMemoryBufferDesc_v1
   ExternalMemoryMipmappedArrayDesc_v1
   ExternalSemaphoreHandleDesc_v1
   ExternalSemaphoreSignalParams_v1
   ExternalSemaphoreWaitParams_v1
   ExtSemSignalNodeParams_v1
   ExtSemSignalNodeParams_v2
   ExtSemWaitNodeParams_v1
   ExtSemWaitNodeParams_v2
   ArrayMapInfo_v1
   MemLocation_v1
   MemAllocationProp_v1
   MulticastObjectProp_v1
   MemAccessDesc_v1
   GraphExecUpdateResultInfo_v1
   MemPoolProps_v1
   MemPoolPtrExportData_v1
   MemCpyAttributes_v1
   Offset3D_v1
   Extent3D_v1
   MemCpy3dOperand_v1
   MemCpy3dBatchOp_v1
   MemAllocNodeParams_v1
   MemAllocNodeParams_v2
   MemFreeNodeParams
   ChildGraphNodeParams
   EventRecordNodeParams
   EventWaitNodeParams
   GraphNodeParams
   CheckpointLockArgs
   CheckpointCheckpointArgs
   CheckpointGpuPair
   CheckpointRestoreArgs
   CheckpointUnlockArgs
   IpcMemFlags
   MemAttachFlags
   CtxFlags
   EventSchedFlags
   ContextFlags
   HostTaskSyncMode
   StreamFlags
   EventFlags
   EventRecordFlags
   EventWaitFlags
   AtomicOperation
   AtomicOperationCapability
   StreamWaitValueFlags
   StreamWriteValueFlags
   StreamBatchMemOpType
   StreamMemoryBarrierFlags
   StreamAtomicReductionOpType
   StreamAtomicReductionDataType
   OccupancyFlags
   StreamUpdateCaptureDependenciesFlags
   AsyncNotificationType
   ArrayFormat
   AddressMode
   FilterMode
   DeviceAttribute
   PointerAttribute
   FunctionAttribute
   FuncCache
   SharedCarveout
   MemoryType
   ComputeMode
   MemAdvise
   MemRangeAttribute
   JitOption
   JitTarget
   JitFallback
   JitCacheMode
   JitInputType
   GraphicsRegisterFlags
   GraphicsMapResourceFlags
   ArrayCubemapFace
   Limit
   ResourceType
   AccessProperty
   GraphConditionalNodeType
   GraphNodeType
   GraphDependencyType
   GraphInstantiateResult
   SynchronizationPolicy
   ClusterSchedulingPolicy
   LaunchMemSyncDomain
   LaunchAttributePortableClusterMode
   SharedMemoryMode
   LaunchAttributeID
   StreamCaptureStatus
   StreamCaptureMode
   DriverProcAddressFlags
   DriverProcAddressQueryResult
   ExecAffinityType
   CigDataType
   StreamCigDataType
   LibraryOption
   Result
   DeviceP2PAttribute
   ResourceViewFormat
   TensorMapDataType
   TensorMapInterleave
   TensorMapSwizzle
   TensorMapL2Promotion
   TensorMapFloatOOBFill
   TensorMapIm2ColWideMode
   PointerAttributeAccessFlags
   ExternalMemoryHandleType
   ExternalSemaphoreHandleType
   MemAllocationHandleType
   MemAccessFlags
   MemLocationType
   MemAllocationType
   MemAllocationGranularityFlags
   MemRangeHandleType
   MemRangeFlags
   ArraySparseSubresourceType
   MemOperationType
   MemHandleType
   MemAllocationCompType
   MulticastGranularityFlags
   GraphExecUpdateResult
   MemPoolAttribute
   MemCpyFlags
   MemCpySrcAccessOrder
   MemCpy3DOperandType
   GraphMemAttribute
   GraphChildGraphNodeOwnership
   FlushGPUDirectRDMAWritesOptions
   GPUDirectRDMAWritesOrdering
   FlushGPUDirectRDMAWritesScope
   FlushGPUDirectRDMAWritesTarget
   GraphDebugDotFlags
   UserObjectFlags
   UserObjectRetainFlags
   GraphInstantiateFlags
   DeviceNumaConfig
   ProcessState
   EglFrameType
   EglResourceLocationFlags
   EglColorFormat
   Array3dFlags
   ArraySparsePropertiesFlags
   ComputeTargetBase
   ExternalMemoryFlags
   ExternalSemaphoreSignalFlags
   ExternalSemaphoreWaitFlags
   GLMapFlags
   GraphConditionalHandleFlags
   GraphKernelNodePort
   MemCreateUsage
   MemHostAllocFlags
   MemHostRegisterFlags
   MemPoolCreateUsage
   NvSciSyncAttr
   OutputMode
   SpecialDevice
   SpecialStream
   DevSmResourceGroupParams
   StreamMemOpAtomicReductionParams
   StreamMemOpFlushRemoteWritesParams
   StreamMemOpMemoryBarrierParams
   StreamMemOpWaitValueParams
   StreamMemOpWriteValueParams
   DevProp_v1
   LaunchParams_v1
   CooperativeLaunchMultiDeviceFlags
   LaunchParam
   NvclEventFlags
   ParamTexRef
   SharedConfig
   TexRefArrayFlags
   TexRefFlags

Error Handling
--------------

This section describes the error handling functions of the low-level CUDA driver application programming interface.

.. autosummary::
   :toctree: generated/driver_v2/
   :nosignatures:

   AlreadyAcquiredError
   AlreadyMappedError
   ArrayIsMappedError
   AssertError
   CallRequiresNewerDriverError
   CapturedEventError
   CdpNotSupportedError
   CdpVersionMismatchError
   CompatNotSupportedOnDeviceError
   ContainedError
   ContextAlreadyCurrentError
   ContextAlreadyInUseError
   ContextIsDestroyedError
   CooperativeLaunchTooLargeError
   DeinitializedError
   DeviceNotLicensedError
   DeviceUnavailableError
   DriverError
   EccUncorrectableError
   ExternalDeviceError
   FabricNotReadyError
   DriverFileNotFoundError
   FunctionNotLoadedError
   GraphExecUpdateFailureError
   GraphRecaptureFailureError
   HardwareStackError
   HostMemoryAlreadyRegisteredError
   HostMemoryNotRegisteredError
   IllegalAddressError
   IllegalInstructionError
   IllegalStateError
   InsufficientLoaderVersionError
   InvalidAddressSpaceError
   InvalidClusterSizeError
   InvalidContextError
   InvalidDeviceError
   InvalidGraphicsContextError
   InvalidHandleError
   InvalidImageError
   InvalidPcError
   InvalidPtxError
   InvalidResourceConfigurationError
   InvalidResourceTypeError
   InvalidSourceError
   InvalidValueError
   JitCompilationDisabledError
   JitCompilerNotFoundError
   KeyRotationError
   LaunchFailedError
   LaunchIncompatibleTexturingError
   LaunchOutOfResourcesError
   LaunchTimeoutError
   LossyQueryError
   MapFailedError
   MisalignedAddressError
   MpsClientTerminatedError
   MpsConnectionFailedError
   MpsMaxClientsReachedError
   MpsMaxConnectionsReachedError
   MpsRpcFailureError
   MpsServerNotReadyError
   MulticastResourceFullError
   NoBinaryForGpuError
   NoDeviceError
   NotFoundError
   NotInitializedError
   NotMappedAsArrayError
   NotMappedAsPointerError
   NotMappedError
   NotPermittedError
   NotReadyError
   NotSupportedError
   NvlinkUncorrectableError
   OperatingSystemError
   OutOfMemoryError
   PeerAccessAlreadyEnabledError
   PeerAccessNotEnabledError
   PeerAccessUnsupportedError
   PrimaryContextActiveError
   ProfilerAlreadyStartedError
   ProfilerAlreadyStoppedError
   ProfilerDisabledError
   ProfilerNotInitializedError
   SharedObjectInitFailedError
   SharedObjectSymbolNotFoundError
   StreamCaptureImplicitError
   StreamCaptureInvalidatedError
   StreamCaptureIsolationError
   StreamCaptureMergeError
   StreamCaptureUnjoinedError
   StreamCaptureUnmatchedError
   StreamCaptureUnsupportedError
   StreamCaptureWrongThreadError
   StreamDetachedError
   StubLibraryError
   SystemDriverMismatchError
   SystemNotReadyError
   TensorMemoryLeakError
   DriverTimeoutError
   TooManyPeersError
   UnknownError
   UnmapFailedError
   UnsupportedDevsideSyncError
   UnsupportedExecAffinityError
   UnsupportedLimitError
   UnsupportedPtxVersionError

.. autosummary::
   :toctree: generated/driver_v2/
   :nosignatures:

   get_error_string
   get_error_name

Initialization
--------------

This section describes the initialization functions of the low-level CUDA driver application programming interface.

.. autosummary::
   :toctree: generated/driver_v2/
   :nosignatures:

   init

Version Management
------------------

This section describes the version management functions of the low-level CUDA driver application programming interface.

.. autosummary::
   :toctree: generated/driver_v2/
   :nosignatures:

   driver_get_version

Device Management
-----------------

This section describes the device management functions of the low-level CUDA driver application programming interface.

.. autosummary::
   :toctree: generated/driver_v2/
   :nosignatures:

   device_get
   device_get_count
   device_get_name
   device_get_uuid_v2
   device_get_luid
   device_total_mem_v2
   device_get_texture_1d_linear_max_width
   device_get_attribute
   device_get_properties
   device_compute_capability
   device_get_host_atomic_capabilities
   device_get_nv_sci_sync_attributes
   device_set_mem_pool
   device_get_mem_pool
   device_get_default_mem_pool
   device_get_exec_affinity_support
   flush_gpu_direct_rdma_writes

Primary Context Management
--------------------------

This section describes the primary context management functions of the low-level CUDA driver application programming interface.

The primary context is unique per device and shared with the CUDA runtime API. These functions allow integration with other libraries using CUDA.

.. autosummary::
   :toctree: generated/driver_v2/
   :nosignatures:

   device_primary_ctx_retain
   device_primary_ctx_release_v2
   device_primary_ctx_set_flags_v2
   device_primary_ctx_get_state
   device_primary_ctx_reset_v2

Context Management
------------------

This section describes the context management functions of the low-level CUDA driver application programming interface.

Please note that some functions are described in Primary Context Management section.

.. autosummary::
   :toctree: generated/driver_v2/
   :nosignatures:

   ctx_create_v4
   ctx_destroy_v2
   ctx_push_current_v2
   ctx_pop_current_v2
   ctx_set_current
   ctx_get_current
   ctx_get_device
   ctx_get_device_v2
   ctx_get_flags
   ctx_set_flags
   ctx_get_id
   ctx_synchronize
   ctx_synchronize_v2
   ctx_set_limit
   ctx_get_limit
   ctx_get_cache_config
   ctx_set_cache_config
   ctx_get_api_version
   ctx_get_stream_priority_range
   ctx_reset_persisting_l2cache
   ctx_get_exec_affinity
   ctx_record_event
   ctx_wait_event

Module Management
-----------------

This section describes the module management functions of the low-level CUDA driver application programming interface.

.. autosummary::
   :toctree: generated/driver_v2/
   :nosignatures:

   ModuleLoadingMode

.. autosummary::
   :toctree: generated/driver_v2/
   :nosignatures:

   module_load
   module_load_data
   module_load_data_ex
   module_load_fat_binary
   module_unload
   module_get_loading_mode
   module_get_function
   module_get_function_count
   module_enumerate_functions
   module_get_global_v2
   link_create_v2
   link_add_data_v2
   link_add_file_v2
   link_complete
   link_destroy

Library Management
------------------

This section describes the library management functions of the low-level CUDA driver application programming interface.

.. autosummary::
   :toctree: generated/driver_v2/
   :nosignatures:

   library_load_data
   library_load_from_file
   library_unload
   library_get_kernel
   library_get_kernel_count
   library_enumerate_kernels
   library_get_module
   kernel_get_function
   kernel_get_library
   library_get_global
   library_get_managed
   library_get_unified_function
   kernel_get_attribute
   kernel_set_attribute
   kernel_set_cache_config
   kernel_get_name
   kernel_get_param_info
   kernel_get_param_count

Memory Management
-----------------

This section describes the memory management functions of the low-level CUDA driver application programming interface.

.. autosummary::
   :toctree: generated/driver_v2/
   :nosignatures:

   MemDecompressParams
   MemDecompressAlgorithm

.. autosummary::
   :toctree: generated/driver_v2/
   :nosignatures:

   mem_get_info_v2
   mem_alloc_v2
   mem_alloc_pitch_v2
   mem_free_v2
   mem_get_address_range_v2
   mem_alloc_host_v2
   mem_free_host
   mem_host_alloc
   mem_host_get_device_pointer_v2
   mem_host_get_flags
   mem_alloc_managed
   device_register_async_notification
   device_unregister_async_notification
   device_get_by_pci_bus_id
   device_get_pci_bus_id
   ipc_get_event_handle
   ipc_open_event_handle
   ipc_get_mem_handle
   ipc_open_mem_handle_v2
   ipc_close_mem_handle
   mem_host_register_v2
   mem_host_unregister
   cu_memcpy
   memcpy_peer
   memcpy_htod_v2
   memcpy_dtoh_v2
   memcpy_dtod_v2
   memcpy_dtoa_v2
   memcpy_atod_v2
   memcpy_htoa_v2
   memcpy_atoh_v2
   memcpy_atoa_v2
   memcpy_2d_v2
   memcpy_2d_unaligned_v2
   memcpy_3d_v2
   memcpy_3d_peer
   memcpy_async
   memcpy_peer_async
   memcpy_htod_async_v2
   memcpy_dtoh_async_v2
   memcpy_dtod_async_v2
   memcpy_htoa_async_v2
   memcpy_atoh_async_v2
   memcpy_2d_async_v2
   memcpy_3d_async_v2
   memcpy_3d_peer_async
   memcpy_batch_async
   memcpy_batch_async_v2
   memcpy_3d_batch_async_v2
   memcpy_with_attributes_async
   memcpy_3d_with_attributes_async
   memset_d8_v2
   memset_d16_v2
   memset_d32_v2
   memset_d2d8_v2
   memset_d2d16_v2
   memset_d2d32_v2
   memset_d8_async
   memset_d16_async
   memset_d32_async
   memset_d2d8_async
   memset_d2d16_async
   memset_d2d32_async
   array_create_v2
   array_get_descriptor_v2
   array_get_sparse_properties
   mipmapped_array_get_sparse_properties
   array_get_memory_requirements
   mipmapped_array_get_memory_requirements
   array_get_plane
   array_destroy
   array_3d_create_v2
   array_3d_get_descriptor_v2
   mipmapped_array_create
   mipmapped_array_get_level
   mipmapped_array_destroy
   mem_get_handle_for_address_range
   mem_batch_decompress_async

Virtual Memory Management
-------------------------

This section describes the virtual memory management functions of the low-level CUDA driver application programming interface.

.. autosummary::
   :toctree: generated/driver_v2/
   :nosignatures:

   mem_address_reserve
   mem_address_free
   mem_create
   mem_release
   mem_map
   mem_map_array_async
   mem_unmap
   mem_set_access
   mem_get_access
   mem_export_to_shareable_handle
   mem_import_from_shareable_handle
   mem_get_allocation_granularity
   mem_get_allocation_properties_from_handle
   mem_retain_allocation_handle

Stream Ordered Memory Allocator
-------------------------------

This section describes the stream ordered memory allocator exposed by the low-level CUDA driver application programming interface.

**overview**

The asynchronous allocator allows the user to allocate and free in stream order. All asynchronous accesses of the allocation must happen between the stream executions of the allocation and the free. If the memory is accessed outside of the promised stream order, a use before allocation / use after free error will cause undefined behavior.

The allocator is free to reallocate the memory as long as it can guarantee that compliant memory accesses will not overlap temporally. The allocator may refer to internal stream ordering as well as inter-stream dependencies (such as CUDA events and null stream dependencies) when establishing the temporal guarantee. The allocator may also insert inter-stream dependencies to establish the temporal guarantee.

**Supported Platforms**

Whether or not a device supports the integrated stream ordered memory allocator may be queried by calling cuDeviceGetAttribute() with the device attribute CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED

.. autosummary::
   :toctree: generated/driver_v2/
   :nosignatures:

   mem_free_async
   mem_alloc_async
   mem_pool_trim_to
   mem_pool_set_attribute
   mem_pool_get_attribute
   mem_pool_set_access
   mem_pool_get_access
   mem_pool_create
   mem_pool_destroy
   mem_get_default_mem_pool
   mem_get_mem_pool
   mem_set_mem_pool
   mem_alloc_from_pool_async
   mem_pool_export_to_shareable_handle
   mem_pool_import_from_shareable_handle
   mem_pool_export_pointer
   mem_pool_import_pointer

Multicast Object Management
---------------------------

This section describes the CUDA multicast object operations exposed by the low-level CUDA driver application programming interface.

**overview**

A multicast object created via cuMulticastCreate enables certain memory operations to be broadcast to a team of devices. Devices can be added to a multicast object via cuMulticastAddDevice. Memory can be bound on each participating device via cuMulticastBindMem, cuMulticastBindMem_v2, cuMulticastBindAddr, or cuMulticastBindAddr_v2. Multicast objects can be mapped into a device's virtual address space using the virtual memmory management APIs (see cuMemMap and cuMemSetAccess).

**Supported Platforms**

Support for multicast on a specific device can be queried using the device attribute CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED

.. autosummary::
   :toctree: generated/driver_v2/
   :nosignatures:

   multicast_create
   multicast_add_device
   multicast_bind_mem
   multicast_bind_mem_v2
   multicast_bind_addr
   multicast_bind_addr_v2
   multicast_unbind
   multicast_get_granularity

Fabric Clique Information
-------------------------

This section describes functions and data structures to retrieve fabric clique information.

.. autosummary::
   :toctree: generated/driver_v2/
   :nosignatures:

   CliqueInfo
   CliqueType

.. autosummary::
   :toctree: generated/driver_v2/
   :nosignatures:

   device_get_fabric_cluster_uuid
   device_get_clique_count
   device_get_clique_info

Logical Endpoint
----------------

This section describes the logical endpoint functions of the low-level CUDA driver application programming interface.

.. autosummary::
   :toctree: generated/driver_v2/
   :nosignatures:

   LogicalEndpointFabricHandle
   LogicalEndpointProp
   LogicalEndpointIpcHandleType
   LogicalEndpointType
   LogicalEndpointFlag

.. autosummary::
   :toctree: generated/driver_v2/
   :nosignatures:

   logical_endpoint_id_reserve
   logical_endpoint_id_release
   logical_endpoint_create
   logical_endpoint_add_device
   logical_endpoint_destroy
   logical_endpoint_bind_addr
   logical_endpoint_bind_mem
   logical_endpoint_unbind
   logical_endpoint_export
   logical_endpoint_import
   logical_endpoint_get_limits
   logical_endpoint_query

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

.. autosummary::
   :toctree: generated/driver_v2/
   :nosignatures:

   pointer_get_attribute
   mem_prefetch_async_v2
   mem_advise_v2
   mem_prefetch_batch_async
   mem_discard_batch_async
   mem_discard_and_prefetch_batch_async
   mem_get_location_info
   mem_range_get_attribute
   mem_range_get_attributes
   pointer_set_attribute
   pointer_get_attributes

Stream Management
-----------------

This section describes the stream management functions of the low-level CUDA driver application programming interface.

.. autosummary::
   :toctree: generated/driver_v2/
   :nosignatures:

   GraphRecaptureStatus

.. autosummary::
   :toctree: generated/driver_v2/
   :nosignatures:

   stream_create
   stream_create_with_priority
   stream_begin_capture_to_cig
   stream_end_capture_to_cig
   stream_get_priority
   stream_get_device
   stream_get_flags
   stream_get_id
   stream_get_ctx
   stream_get_ctx_v2
   stream_wait_event
   stream_add_callback
   stream_begin_capture_v2
   stream_begin_recapture_to_graph
   stream_begin_capture_to_graph
   thread_exchange_stream_capture_mode
   stream_end_capture
   stream_is_capturing
   stream_get_capture_info_v2
   stream_get_capture_info_v3
   stream_update_capture_dependencies_v2
   stream_attach_mem_async
   stream_query
   stream_synchronize
   stream_destroy_v2
   stream_copy_attributes
   stream_get_attribute
   stream_set_attribute

Event Management
----------------

This section describes the event management functions of the low-level CUDA driver application programming interface.

.. autosummary::
   :toctree: generated/driver_v2/
   :nosignatures:

   event_create
   event_record
   event_record_with_flags
   event_query
   event_synchronize
   event_destroy_v2
   event_elapsed_time_v2

External Resource Interoperability
----------------------------------

This section describes the external resource interoperability functions of the low-level CUDA driver application programming interface.

.. autosummary::
   :toctree: generated/driver_v2/
   :nosignatures:

   import_external_memory
   external_memory_get_mapped_buffer
   external_memory_get_mapped_mipmapped_array
   destroy_external_memory
   import_external_semaphore
   signal_external_semaphores_async
   wait_external_semaphores_async
   destroy_external_semaphore

Stream Memory Operations
------------------------

This section describes the stream memory operations of the low-level CUDA driver application programming interface.

Support for the CU_STREAM_WAIT_VALUE_NOR flag can be queried with ::CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR_V2.

Support for the cuStreamWriteValue64() and cuStreamWaitValue64() functions, as well as for the CU_STREAM_MEM_OP_WAIT_VALUE_64 and CU_STREAM_MEM_OP_WRITE_VALUE_64 flags, can be queried with CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS.

Support for both CU_STREAM_WAIT_VALUE_FLUSH and CU_STREAM_MEM_OP_FLUSH_REMOTE_WRITES requires dedicated platform hardware features and can be queried with cuDeviceGetAttribute() and CU_DEVICE_ATTRIBUTE_CAN_FLUSH_REMOTE_WRITES.

Note that all memory pointers passed as parameters to these operations are device pointers. Where necessary a device pointer should be obtained, for example with cuMemHostGetDevicePointer().

None of the operations accepts pointers to managed memory buffers (cuMemAllocManaged).

Warning: Improper use of these APIs may deadlock the application. Synchronization ordering established through these APIs is not visible to CUDA. CUDA tasks that are (even indirectly) ordered by these APIs should also have that order expressed with CUDA-visible dependencies such as events. This ensures that the scheduler does not serialize them in an improper order.

.. autosummary::
   :toctree: generated/driver_v2/
   :nosignatures:

   stream_wait_value32_v2
   stream_wait_value64_v2
   stream_write_value32_v2
   stream_write_value64_v2
   stream_batch_mem_op_v2

Execution Control
-----------------

This section describes the execution control functions of the low-level CUDA driver application programming interface.

.. autosummary::
   :toctree: generated/driver_v2/
   :nosignatures:

   FunctionLoadingState

.. autosummary::
   :toctree: generated/driver_v2/
   :nosignatures:

   func_get_attribute
   func_set_attribute
   func_set_cache_config
   func_get_module
   func_get_name
   func_get_param_info
   func_get_param_count
   func_is_loaded
   func_load
   launch_kernel
   launch_kernel_ex
   launch_cooperative_kernel
   launch_cooperative_kernel_multi_device
   launch_host_func
   launch_host_func_v2

Graph Management
----------------

This section describes the graph management functions of the low-level CUDA driver application programming interface.

.. autosummary::
   :toctree: generated/driver_v2/
   :nosignatures:

   graph_create
   graph_add_kernel_node_v2
   graph_kernel_node_get_params_v2
   graph_kernel_node_set_params_v2
   graph_add_memcpy_node
   graph_memcpy_node_get_params
   graph_memcpy_node_set_params
   graph_add_memset_node
   graph_memset_node_get_params
   graph_memset_node_set_params
   graph_add_host_node
   graph_host_node_get_params
   graph_host_node_set_params
   graph_add_child_graph_node
   graph_child_graph_node_get_graph
   graph_add_empty_node
   graph_add_event_record_node
   graph_event_record_node_get_event
   graph_event_record_node_set_event
   graph_add_event_wait_node
   graph_event_wait_node_get_event
   graph_event_wait_node_set_event
   graph_add_external_semaphores_signal_node
   graph_external_semaphores_signal_node_get_params
   graph_external_semaphores_signal_node_set_params
   graph_add_external_semaphores_wait_node
   graph_external_semaphores_wait_node_get_params
   graph_external_semaphores_wait_node_set_params
   graph_add_batch_mem_op_node
   graph_batch_mem_op_node_get_params
   graph_batch_mem_op_node_set_params
   graph_exec_batch_mem_op_node_set_params
   graph_add_mem_alloc_node
   graph_mem_alloc_node_get_params
   graph_add_mem_free_node
   graph_mem_free_node_get_params
   device_graph_mem_trim
   device_get_graph_mem_attribute
   device_set_graph_mem_attribute
   graph_clone
   graph_node_find_in_clone
   graph_node_get_type
   graph_node_get_containing_graph
   graph_node_get_local_id
   graph_node_get_tools_id
   graph_get_id
   graph_exec_get_id
   graph_get_nodes
   graph_get_root_nodes
   graph_get_edges
   graph_get_edges_v2
   graph_node_get_dependencies
   graph_node_get_dependencies_v2
   graph_node_get_dependent_nodes
   graph_node_get_dependent_nodes_v2
   graph_add_dependencies_v2
   graph_remove_dependencies_v2
   graph_destroy_node
   graph_instantiate
   graph_instantiate_with_flags
   graph_instantiate_with_params
   graph_exec_get_flags
   graph_exec_kernel_node_set_params_v2
   graph_exec_memcpy_node_set_params
   graph_exec_memset_node_set_params
   graph_exec_host_node_set_params
   graph_exec_child_graph_node_set_params
   graph_exec_event_record_node_set_event
   graph_exec_event_wait_node_set_event
   graph_exec_external_semaphores_signal_node_set_params
   graph_exec_external_semaphores_wait_node_set_params
   graph_node_set_enabled
   graph_node_get_enabled
   graph_upload
   graph_launch
   graph_exec_destroy
   graph_destroy
   graph_exec_update_v2
   graph_kernel_node_copy_attributes
   graph_kernel_node_get_attribute
   graph_kernel_node_set_attribute
   graph_debug_dot_print
   user_object_create
   user_object_retain
   user_object_release
   graph_retain_user_object
   graph_release_user_object
   graph_add_node_v2
   graph_add_node_v3
   graph_node_set_params
   graph_node_set_params_v2
   graph_node_get_params
   graph_exec_node_set_params
   graph_conditional_handle_create

Occupancy
---------

This section describes the occupancy calculation functions of the low-level CUDA driver application programming interface.

.. autosummary::
   :toctree: generated/driver_v2/
   :nosignatures:

   occupancy_max_active_blocks_per_multiprocessor
   occupancy_max_active_blocks_per_multiprocessor_with_flags
   occupancy_max_potential_block_size
   occupancy_max_potential_block_size_with_flags
   occupancy_available_dynamic_smem_per_block
   occupancy_max_potential_cluster_size
   occupancy_max_active_clusters

Texture Object Management
-------------------------

This section describes the texture object management functions of the low-level CUDA driver application programming interface. The texture object API is only supported on devices of compute capability 3.0 or higher.

.. autosummary::
   :toctree: generated/driver_v2/
   :nosignatures:

   tex_object_create
   tex_object_destroy
   tex_object_get_resource_desc
   tex_object_get_texture_desc
   tex_object_get_resource_view_desc

Surface Object Management
-------------------------

This section describes the surface object management functions of the low-level CUDA driver application programming interface. The surface object API is only supported on devices of compute capability 3.0 or higher.

.. autosummary::
   :toctree: generated/driver_v2/
   :nosignatures:

   surf_object_create
   surf_object_destroy
   surf_object_get_resource_desc

Tensor Map Object Managment
---------------------------

This section describes the tensor map object management functions of the low-level CUDA driver application programming interface. The tensor core API is only supported on devices of compute capability 9.0 or higher.

.. autosummary::
   :toctree: generated/driver_v2/
   :nosignatures:

   tensor_map_encode_tiled
   tensor_map_encode_im2col
   tensor_map_encode_im2col_wide
   tensor_map_replace_address

Peer Context Memory Access
--------------------------

This section describes the direct peer context memory access functions of the low-level CUDA driver application programming interface.

.. autosummary::
   :toctree: generated/driver_v2/
   :nosignatures:

   device_can_access_peer
   ctx_enable_peer_access
   ctx_disable_peer_access
   device_get_p2p_attribute
   device_get_p2p_atomic_capabilities

Graphics Interoperability
-------------------------

This section describes the graphics interoperability functions of the low-level CUDA driver application programming interface.

.. autosummary::
   :toctree: generated/driver_v2/
   :nosignatures:

   graphics_unregister_resource
   graphics_sub_resource_get_mapped_array
   graphics_resource_get_mapped_mipmapped_array
   graphics_resource_get_mapped_pointer_v2
   graphics_resource_set_map_flags_v2
   graphics_map_resources
   graphics_unmap_resources

Driver Entry Point Access
-------------------------

This section describes the driver entry point access functions of the low-level CUDA driver application programming interface.

.. autosummary::
   :toctree: generated/driver_v2/
   :nosignatures:

   get_proc_address_v2
   get_export_table

Coredump Attributes Control API
-------------------------------

This section describes the coredump attribute control functions of the low-level CUDA driver application programming interface.

.. autosummary::
   :toctree: generated/driver_v2/
   :nosignatures:

   CoredumpSettings
   CoredumpGenerationFlags

.. autosummary::
   :toctree: generated/driver_v2/
   :nosignatures:

   coredump_get_attribute
   coredump_get_attribute_global
   coredump_set_attribute
   coredump_set_attribute_global
   coredump_register_start_callback
   coredump_register_complete_callback
   coredump_deregister_start_callback
   coredump_deregister_complete_callback

Green Contexts
--------------

This section describes the APIs for creation and manipulation of green contexts in the CUDA driver. Green contexts are a lightweight alternative to traditional contexts, that can be used to select a subset of device resources. This allows the developer to, for example, select SMs from distinct spatial partitions of the GPU and target them via CUDA stream operations, kernel launches, etc.

Here are the broad initial steps to follow to get started:

- (1) Start with an initial set of resources. For SM resources, they can be fetched via cuDeviceGetDevResource. In case of workqueues, a new configuration can be used or an existing one queried via the cuDeviceGetDevResource API.

- (2) Modify these resources by either partitioning them (in case of SMs) or changing the configuration (in case of workqueues). To partition SMs, we recommend cuDevSmResourceSplit. Changing the workqueue configuration can be done directly in place.

- (3) Finalize the specification of resources by creating a descriptor via cuDevResourceGenerateDesc.

- (4) Create a green context via cuGreenCtxCreate. This provisions the resource, such as workqueues (until this step it was only a configuration specification).

- (5) Create a stream via cuGreenCtxStreamCreate, and use it throughout your application.

SMs

There are two possible partition operations - with cuDevSmResourceSplitByCount the partitions created have to follow default SM count granularity requirements, so it will often be rounded up and aligned to a default value. On the other hand, cuDevSmResourceSplit is explicit and allows for creation of non-equal groups. It will not round up automatically - instead it is the developer’s responsibility to query and set the correct values. These requirements can be queried with cuDeviceGetDevResource to determine the alignment granularity (sm.smCoscheduledAlignment). A general guideline on the default values for each compute architecture:

- On all architectures,

  - Portable code should set smCount to a multiple of the device's alignment granularity (sm.smCoscheduledAlignment).

- On Compute Architecture 7.X, 8.X, and all Tegra SoC:

  - The smCount must be a multiple of 2.

  - The alignment (and default value of coscheduledSmCount) is 2.

- On Compute Architecture 9.0+:

  - The smCount must be a multiple of coscheduledSmCount if provided.

  - The alignment is 8.

While the maximum value for coscheduled SM count is 32 on all Compute Architecture 9.0+, it's recommended to follow cluster size requirements. The portable cluster size and the max cluster size should be used in order to benefit from this co-scheduling.

Workqueues

For ``CU_DEV_RESOURCE_TYPE_WORKQUEUE_CONFIG``, the resource specifies the expected maximum number of concurrent stream-ordered workloads via the ``wqConcurrencyLimit`` field. The ``sharingScope`` field determines how workqueue resources are shared:

- ``CU_WORKQUEUE_SCOPE_DEVICE_CTX:`` Use all shared workqueue resources across all contexts (default driver behavior).

- ``CU_WORKQUEUE_SCOPE_GREEN_CTX_BALANCED:`` When possible, use non-overlapping workqueue resources with other balanced green contexts.

The maximum concurrency limit depends on ::CUDA_DEVICE_MAX_CONNECTIONS and can be queried from the primary context via cuCtxGetDevResource. Configurations may exceed this concurrency limit, but the driver will not guarantee that work submission remains non-overlapping.

For ``CU_DEV_RESOURCE_TYPE_WORKQUEUE``, the resource represents a pre-existing workqueue that can be retrieved from existing contexts or green contexts. This allows reusing workqueue resources across different green contexts.

On Concurrency

Even if the green contexts have disjoint SM partitions, it is not guaranteed that the kernels launched in them will run concurrently or have forward progress guarantees. This is due to other resources that could cause a dependency. Using a combination of disjoint SMs and CU_WORKQUEUE_SCOPE_GREEN_CTX_BALANCED workqueue configurations can provide the best chance of avoiding interference. More resources will be added in the future to provide stronger guarantees.

Additionally, there are two known scenarios, where its possible for the workload to run on more SMs than was provisioned (but never less).

- On Volta+ MPS: When ``CUDA_MPS_ACTIVE_THREAD_PERCENTAGE`` is used, the set of SMs that are used for running kernels can be scaled up to the value of SMs used for the MPS client.

- On Compute Architecture 9.x: When a module with dynamic parallelism (CDP) is loaded, all future kernels running under green contexts may use and share an additional set of 2 SMs.

Memory Copy Operations

Green context restrictions apply to memory copy operations only when the copy is performed using a green context. For cross-device copies, green context restrictions may not be applied.

.. autosummary::
   :toctree: generated/driver_v2/
   :nosignatures:

   DevSmResource
   DevWorkqueueConfigResource
   DevWorkqueueResource
   DevResource_v1
   GreenCtxCreateFlags
   DevSmResourceGroupFlags
   DevSmResourceSplitByCountFlags
   DevResourceType
   DevWorkqueueConfigScope

.. autosummary::
   :toctree: generated/driver_v2/
   :nosignatures:

   green_ctx_create
   green_ctx_destroy
   ctx_from_green_ctx
   device_get_dev_resource
   ctx_get_dev_resource
   green_ctx_get_dev_resource
   dev_sm_resource_split_by_count
   dev_sm_resource_split
   dev_resource_generate_desc
   green_ctx_record_event
   green_ctx_wait_event
   stream_get_green_ctx
   green_ctx_stream_create
   green_ctx_get_id
   stream_get_dev_resource

Error Log Management Functions
------------------------------

This section describes the error log management functions of the low-level CUDA driver application programming interface.

.. autosummary::
   :toctree: generated/driver_v2/
   :nosignatures:

   LogLevel

.. autosummary::
   :toctree: generated/driver_v2/
   :nosignatures:

   logs_register_callback
   logs_unregister_callback
   logs_current
   logs_dump_to_file
   logs_dump_to_memory

CUDA Checkpointing
------------------

CUDA API versioning support

This sections describes the checkpoint and restore functions of the low-level CUDA driver application programming interface.

The CUDA checkpoint and restore API's provide a way to save and restore GPU state for full process checkpoints when used with CPU side process checkpointing solutions. They can also be used to pause GPU work and suspend a CUDA process to allow other applications to make use of GPU resources.

Checkpoint and restore capabilities are currently restricted to Linux.

.. autosummary::
   :toctree: generated/driver_v2/
   :nosignatures:

   checkpoint_process_get_restore_thread_id
   checkpoint_process_get_state
   checkpoint_process_lock
   checkpoint_process_checkpoint
   checkpoint_process_restore
   checkpoint_operation_complete
   checkpoint_process_unlock

Profiler Control
----------------

This section describes the profiler control functions of the low-level CUDA driver application programming interface.

.. autosummary::
   :toctree: generated/driver_v2/
   :nosignatures:

   profiler_start
   profiler_stop

EGL Interoperability
--------------------

This section describes the EGL interoperability functions of the low-level CUDA driver application programming interface.

.. autosummary::
   :toctree: generated/driver_v2/
   :nosignatures:

   graphics_egl_register_image
   egl_stream_consumer_connect
   egl_stream_consumer_connect_with_flags
   egl_stream_consumer_disconnect
   egl_stream_consumer_acquire_frame
   egl_stream_consumer_release_frame
   egl_stream_producer_connect
   egl_stream_producer_disconnect
   egl_stream_producer_present_frame
   egl_stream_producer_return_frame
   graphics_resource_get_mapped_egl_frame
   event_create_from_egl_sync

OpenGL Interoperability
-----------------------

This section describes the OpenGL interoperability functions of the low-level CUDA driver application programming interface. Note that mapping of OpenGL resources is performed with the graphics API agnostic, resource mapping interface described in Graphics Interoperability.

.. autosummary::
   :toctree: generated/driver_v2/
   :nosignatures:

   graphics_gl_register_buffer
   graphics_gl_register_image
   gl_get_devices_v2

VDPAU Interoperability
----------------------

This section describes the VDPAU interoperability functions of the low-level CUDA driver application programming interface.

.. autosummary::
   :toctree: generated/driver_v2/
   :nosignatures:

   vdpau_get_device
   vdpau_ctx_create_v2
   graphics_vdpau_register_video_surface
   graphics_vdpau_register_output_surface

Deprecated Context Management
-----------------------------

This section describes the deprecated context management functions of the low-level CUDA driver application programming interface.

.. autosummary::
   :toctree: generated/driver_v2/
   :nosignatures:

   ctx_attach
   ctx_detach
   ctx_get_shared_mem_config
   ctx_set_shared_mem_config

Deprecated Module Management
----------------------------

This section describes the deprecated module management functions of the low-level CUDA driver application programming interface.

.. autosummary::
   :toctree: generated/driver_v2/
   :nosignatures:

   module_get_tex_ref
   module_get_surf_ref

Deprecated Execution Control
----------------------------

This section describes the deprecated execution control functions of the low-level CUDA driver application programming interface.

.. autosummary::
   :toctree: generated/driver_v2/
   :nosignatures:

   func_set_block_shape
   func_set_shared_size
   func_set_shared_mem_config
   param_set_size
   param_seti
   param_setf
   param_setv
   param_set_tex_ref
   launch
   launch_grid
   launch_grid_async

Deprecated Texture Reference Management
---------------------------------------

This section describes the deprecated texture reference management functions of the low-level CUDA driver application programming interface.

.. autosummary::
   :toctree: generated/driver_v2/
   :nosignatures:

   tex_ref_create
   tex_ref_destroy
   tex_ref_set_array
   tex_ref_set_mipmapped_array
   tex_ref_set_address_v2
   tex_ref_set_address2d_v3
   tex_ref_set_format
   tex_ref_set_address_mode
   tex_ref_set_filter_mode
   tex_ref_set_mipmap_filter_mode
   tex_ref_set_mipmap_level_bias
   tex_ref_set_mipmap_level_clamp
   tex_ref_set_max_anisotropy
   tex_ref_set_border_color
   tex_ref_set_flags
   tex_ref_get_address_v2
   tex_ref_get_array
   tex_ref_get_mipmapped_array
   tex_ref_get_address_mode
   tex_ref_get_filter_mode
   tex_ref_get_format
   tex_ref_get_mipmap_filter_mode
   tex_ref_get_mipmap_level_bias
   tex_ref_get_mipmap_level_clamp
   tex_ref_get_max_anisotropy
   tex_ref_get_border_color
   tex_ref_get_flags

Deprecated Surface Reference Management
---------------------------------------

This section describes the deprecated surface reference management functions of the low-level CUDA driver application programming interface.

.. autosummary::
   :toctree: generated/driver_v2/
   :nosignatures:

   surf_ref_set_array
   surf_ref_get_array
