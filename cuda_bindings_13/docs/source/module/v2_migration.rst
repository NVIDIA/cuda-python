.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

.. _v2-migration:

------------------------------------
Migrating to the ``_v2`` Bindings
------------------------------------

The ``cuda.bindings._v2`` package provides new bindings for the CUDA driver API
(:doc:`driver_v2 <driver_v2>`) and NVRTC (:doc:`nvrtc_v2 <nvrtc_v2>`). They cover the
same C APIs as the existing :doc:`driver <driver>` and :doc:`nvrtc <nvrtc>`
modules but follow a more Pythonic design. This guide lists what changed so that
existing code can be ported.

Overview of changes
===================

Errors are raised as exceptions
-------------------------------

Legacy functions return a tuple that starts with a result code, such as
``(CUresult.CUDA_SUCCESS, value)``. The new functions never return a result code.
A failing call raises an exception instead. The exceptions derive from ``DriverError`` for the
driver and ``NvrtcError`` for NVRTC, and there is one subclass per error code, for
example ``InvalidDeviceError``. The numeric status is available as ``e.status``
(a plain ``int``).

Return values
-------------

A function with no outputs returns ``None``, a function with one output returns that
value directly, and a function with several outputs returns a ``tuple``. The order of
the outputs is the same as in the legacy API. Functions whose signature or return value changes beyond this are
noted in the tables below.

Naming
------

- Functions use ``snake_case`` with the ``cu``/``nvrtc`` prefix removed
  (``cuStreamCreate`` becomes ``stream_create``). Driver functions keep the version suffix of
  the C symbol (``mem_alloc_v2``), whereas the legacy module drops it.
- Parameters and struct members use ``snake_case``. Names that would clash with a Python keyword or
  builtin, or with a name reserved by the generated classes, get a trailing underscore (``input_``, ``flags_``).
- Enumeration classes and structs use ``CamelCase`` with the ``CU``/``nvrtc`` prefix removed.
  Structs keep their C version suffix (``MemLocation_v1``) and there is a single class per
  versioned C type instead of several aliases.
- Enumeration members lose their common C prefix (``CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK``
  becomes ``DeviceAttribute.MAX_THREADS_PER_BLOCK``), and the old full names are not kept as aliases.
  Enumerations with a single member keep the part of the C name that identifies the member, because there is no
  second member to share a prefix with. For example ``CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS`` becomes
  ``IpcMemFlags.IPC_MEM_LAZY_ENABLE_PEER_ACCESS`` and ``CU_USER_OBJECT_NO_DESTRUCTOR_SYNC`` becomes
  ``UserObjectFlags.USER_OBJECT_NO_DESTRUCTOR_SYNC``. The *Prefix removed* columns below list what was removed in each case.

Handles are plain integers
--------------------------

Handle types such as ``CUcontext``, ``CUstream``, ``CUevent``, ``CUmodule``, ``CUdeviceptr`` and
``nvrtcProgram`` are no longer classes. Functions return them as ``int`` and accept only ``int``.
Legacy functions accepted ``None`` for a null handle. The new functions raise ``TypeError`` for ``None``;
pass ``0`` instead (for example ``stream_synchronize(0)`` for the default stream).
Use ``int(handle)`` to convert a legacy handle object.

Enumerations
------------

Enumeration classes are ``int`` subclasses. Functions accept either the enumeration members or plain integers.
Their members expose a description of the value automatically, so the individual members
are no longer documented one by one.

Structs
-------

Struct classes are generated extension types that wrap a C struct. They expose their address through the ``.ptr``
property (replacing ``getPtr()``), and can be created from existing memory with the
``from_ptr()``, ``from_data()`` (which takes a NumPy array of the matching ``*_dtype``) and ``from_buffer()`` class methods. Where a function takes a struct, the argument
is either a raw address (an ``int``) or the struct object itself. Wrapper objects of the legacy classes cannot be
passed in their place.

Strings and constants
---------------------

- ``get_error_string`` and ``get_error_name`` return ``str``, where the legacy functions returned ``bytes``.
- The ``#define`` constants that the legacy driver module exposed as module attributes (``CU_MEMHOSTALLOC_PORTABLE`` and
  so on) are now members of enumerations grouped by purpose, for example ``MemHostAllocFlags.PORTABLE``. Only a
  few ungrouped constants remain at module level, keeping their C names: ``CUDA_VERSION``, ``CU_IPC_HANDLE_SIZE``,
  ``CU_TENSOR_MAP_NUM_QWORDS``, ``CU_LAUNCH_KERNEL_REQUIRED_BLOCK_DIM``, ``CUDA_EGL_INFINITE_TIMEOUT`` and ``MAX_PLANES``.

Driver
======

The ``cuda.bindings._v2.driver`` module replaces ``cuda.bindings.driver``. The deprecated driver APIs (texture references, surface references, ``cuLaunch*`` and ``cuParam*`` functions, ``cuCtxAttach``, and so on) are also available and follow the same rules. They are listed in the *Deprecated functions* table below.

Functions
---------

Where the C symbol has a version suffix (``cuCtxCreate_v2``), the legacy module exposes only the unsuffixed name, while the new module keeps the suffix, so one legacy function can map to several new ones. Only changes to parameters and return values beyond the general rules above are noted. Renamed parameters are not listed.

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Legacy
     - New
     - Notes
   * - ``cuArray3DCreate``
     - ``array_3d_create_v2``
     -
   * - ``cuArray3DGetDescriptor``
     - ``array_3d_get_descriptor_v2``
     -
   * - ``cuArrayCreate``
     - ``array_create_v2``
     -
   * - ``cuArrayDestroy``
     - ``array_destroy``
     -
   * - ``cuArrayGetDescriptor``
     - ``array_get_descriptor_v2``
     -
   * - ``cuArrayGetMemoryRequirements``
     - ``array_get_memory_requirements``
     -
   * - ``cuArrayGetPlane``
     - ``array_get_plane``
     -
   * - ``cuArrayGetSparseProperties``
     - ``array_get_sparse_properties``
     -
   * - ``cuCheckpointOperationComplete``
     - ``checkpoint_operation_complete``
     -
   * - ``cuCheckpointProcessCheckpoint``
     - ``checkpoint_process_checkpoint``
     -
   * - ``cuCheckpointProcessGetRestoreThreadId``
     - ``checkpoint_process_get_restore_thread_id``
     -
   * - ``cuCheckpointProcessGetState``
     - ``checkpoint_process_get_state``
     -
   * - ``cuCheckpointProcessLock``
     - ``checkpoint_process_lock``
     -
   * - ``cuCheckpointProcessRestore``
     - ``checkpoint_process_restore``
     -
   * - ``cuCheckpointProcessUnlock``
     - ``checkpoint_process_unlock``
     -
   * - ``cuCoredumpDeregisterCompleteCallback``
     - ``coredump_deregister_complete_callback``
     -
   * - ``cuCoredumpDeregisterStartCallback``
     - ``coredump_deregister_start_callback``
     -
   * - ``cuCoredumpGetAttribute``
     - ``coredump_get_attribute``
     - Returns only the value (``bytes``, ``bool`` or ``int``, depending on the attribute). The legacy function also returned ``size``.
   * - ``cuCoredumpGetAttributeGlobal``
     - ``coredump_get_attribute_global``
     - Returns only the value (``bytes``, ``bool`` or ``int``, depending on the attribute). The legacy function also returned ``size``.
   * - ``cuCoredumpRegisterCompleteCallback``
     - ``coredump_register_complete_callback``
     -
   * - ``cuCoredumpRegisterStartCallback``
     - ``coredump_register_start_callback``
     -
   * - ``cuCoredumpSetAttribute``
     - ``coredump_set_attribute``
     - Returns ``None``. The legacy function also returned ``size``.
   * - ``cuCoredumpSetAttributeGlobal``
     - ``coredump_set_attribute_global``
     - Returns ``None``. The legacy function also returned ``size``.
   * - ``cuCtxCreate``
     - ``ctx_create_v4``
     -
   * - ``cuCtxDestroy``
     - ``ctx_destroy_v2``
     -
   * - ``cuCtxDisablePeerAccess``
     - ``ctx_disable_peer_access``
     -
   * - ``cuCtxEnablePeerAccess``
     - ``ctx_enable_peer_access``
     -
   * - ``cuCtxFromGreenCtx``
     - ``ctx_from_green_ctx``
     -
   * - ``cuCtxGetApiVersion``
     - ``ctx_get_api_version``
     -
   * - ``cuCtxGetCacheConfig``
     - ``ctx_get_cache_config``
     -
   * - ``cuCtxGetCurrent``
     - ``ctx_get_current``
     -
   * - ``cuCtxGetDevResource``
     - ``ctx_get_dev_resource``
     -
   * - ``cuCtxGetDevice``
     - ``ctx_get_device``
     -
   * - ``cuCtxGetDevice``
     - ``ctx_get_device_v2``
     - Parameters added: ``ctx``.
   * - ``cuCtxGetExecAffinity``
     - ``ctx_get_exec_affinity``
     -
   * - ``cuCtxGetFlags``
     - ``ctx_get_flags``
     -
   * - ``cuCtxGetId``
     - ``ctx_get_id``
     -
   * - ``cuCtxGetLimit``
     - ``ctx_get_limit``
     -
   * - ``cuCtxGetStreamPriorityRange``
     - ``ctx_get_stream_priority_range``
     -
   * - ``cuCtxPopCurrent``
     - ``ctx_pop_current_v2``
     -
   * - ``cuCtxPushCurrent``
     - ``ctx_push_current_v2``
     -
   * - ``cuCtxRecordEvent``
     - ``ctx_record_event``
     -
   * - ``cuCtxResetPersistingL2Cache``
     - ``ctx_reset_persisting_l2cache``
     -
   * - ``cuCtxSetCacheConfig``
     - ``ctx_set_cache_config``
     -
   * - ``cuCtxSetCurrent``
     - ``ctx_set_current``
     -
   * - ``cuCtxSetFlags``
     - ``ctx_set_flags``
     -
   * - ``cuCtxSetLimit``
     - ``ctx_set_limit``
     -
   * - ``cuCtxSynchronize``
     - ``ctx_synchronize``
     -
   * - ``cuCtxSynchronize``
     - ``ctx_synchronize_v2``
     - Parameters added: ``ctx``.
   * - ``cuCtxWaitEvent``
     - ``ctx_wait_event``
     -
   * - ``cuDestroyExternalMemory``
     - ``destroy_external_memory``
     -
   * - ``cuDestroyExternalSemaphore``
     - ``destroy_external_semaphore``
     -
   * - ``cuDevResourceGenerateDesc``
     - ``dev_resource_generate_desc``
     -
   * - ``cuDevSmResourceSplit``
     - ``dev_sm_resource_split``
     -
   * - ``cuDevSmResourceSplitByCount``
     - ``dev_sm_resource_split_by_count``
     - Parameters removed: ``nbGroups``. Returns ``(resources, remainder)``. The group count is no longer returned.
   * - ``cuDeviceCanAccessPeer``
     - ``device_can_access_peer``
     -
   * - ``cuDeviceGet``
     - ``device_get``
     -
   * - ``cuDeviceGetAttribute``
     - ``device_get_attribute``
     -
   * - ``cuDeviceGetByPCIBusId``
     - ``device_get_by_pci_bus_id``
     -
   * - ``cuDeviceGetCliqueCount``
     - ``device_get_clique_count``
     -
   * - ``cuDeviceGetCliqueInfo``
     - ``device_get_clique_info``
     - Parameters removed: ``count``. Returns the array of ``CliqueInfo`` directly. The count is no longer returned.
   * - ``cuDeviceGetCount``
     - ``device_get_count``
     -
   * - ``cuDeviceGetDefaultMemPool``
     - ``device_get_default_mem_pool``
     -
   * - ``cuDeviceGetDevResource``
     - ``device_get_dev_resource``
     -
   * - ``cuDeviceGetExecAffinitySupport``
     - ``device_get_exec_affinity_support``
     -
   * - ``cuDeviceGetFabricClusterUuid``
     - ``device_get_fabric_cluster_uuid``
     -
   * - ``cuDeviceGetGraphMemAttribute``
     - ``device_get_graph_mem_attribute``
     -
   * - ``cuDeviceGetHostAtomicCapabilities``
     - ``device_get_host_atomic_capabilities``
     - Parameters removed: ``count``.
   * - ``cuDeviceGetLuid``
     - ``device_get_luid``
     -
   * - ``cuDeviceGetMemPool``
     - ``device_get_mem_pool``
     -
   * - ``cuDeviceGetName``
     - ``device_get_name``
     -
   * - ``cuDeviceGetNvSciSyncAttributes``
     - ``device_get_nv_sci_sync_attributes``
     - Returns the populated ``nv_sci_sync_attr_list`` handle.
   * - ``cuDeviceGetP2PAtomicCapabilities``
     - ``device_get_p2p_atomic_capabilities``
     -
   * - ``cuDeviceGetP2PAttribute``
     - ``device_get_p2p_attribute``
     -
   * - ``cuDeviceGetPCIBusId``
     - ``device_get_pci_bus_id``
     -
   * - ``cuDeviceGetTexture1DLinearMaxWidth``
     - ``device_get_texture_1d_linear_max_width``
     -
   * - ``cuDeviceGetUuid``
     - ``device_get_uuid_v2``
     -
   * - ``cuDeviceGraphMemTrim``
     - ``device_graph_mem_trim``
     -
   * - ``cuDevicePrimaryCtxGetState``
     - ``device_primary_ctx_get_state``
     -
   * - ``cuDevicePrimaryCtxRelease``
     - ``device_primary_ctx_release_v2``
     -
   * - ``cuDevicePrimaryCtxReset``
     - ``device_primary_ctx_reset_v2``
     -
   * - ``cuDevicePrimaryCtxRetain``
     - ``device_primary_ctx_retain``
     -
   * - ``cuDevicePrimaryCtxSetFlags``
     - ``device_primary_ctx_set_flags_v2``
     -
   * - ``cuDeviceRegisterAsyncNotification``
     - ``device_register_async_notification``
     -
   * - ``cuDeviceSetGraphMemAttribute``
     - ``device_set_graph_mem_attribute``
     -
   * - ``cuDeviceSetMemPool``
     - ``device_set_mem_pool``
     -
   * - ``cuDeviceTotalMem``
     - ``device_total_mem_v2``
     -
   * - ``cuDeviceUnregisterAsyncNotification``
     - ``device_unregister_async_notification``
     -
   * - ``cuDriverGetVersion``
     - ``driver_get_version``
     -
   * - ``cuEGLStreamConsumerAcquireFrame``
     - ``egl_stream_consumer_acquire_frame``
     - Parameters removed: ``pCudaResource``. Returns the CUDA resource handle.
   * - ``cuEGLStreamConsumerConnect``
     - ``egl_stream_consumer_connect``
     -
   * - ``cuEGLStreamConsumerConnectWithFlags``
     - ``egl_stream_consumer_connect_with_flags``
     -
   * - ``cuEGLStreamConsumerDisconnect``
     - ``egl_stream_consumer_disconnect``
     -
   * - ``cuEGLStreamConsumerReleaseFrame``
     - ``egl_stream_consumer_release_frame``
     -
   * - ``cuEGLStreamProducerConnect``
     - ``egl_stream_producer_connect``
     -
   * - ``cuEGLStreamProducerDisconnect``
     - ``egl_stream_producer_disconnect``
     -
   * - ``cuEGLStreamProducerPresentFrame``
     - ``egl_stream_producer_present_frame``
     -
   * - ``cuEGLStreamProducerReturnFrame``
     - ``egl_stream_producer_return_frame``
     -
   * - ``cuEventCreate``
     - ``event_create``
     -
   * - ``cuEventCreateFromEGLSync``
     - ``event_create_from_egl_sync``
     -
   * - ``cuEventDestroy``
     - ``event_destroy_v2``
     -
   * - ``cuEventElapsedTime``
     - ``event_elapsed_time_v2``
     -
   * - ``cuEventQuery``
     - ``event_query``
     -
   * - ``cuEventRecord``
     - ``event_record``
     -
   * - ``cuEventRecordWithFlags``
     - ``event_record_with_flags``
     -
   * - ``cuEventSynchronize``
     - ``event_synchronize``
     -
   * - ``cuExternalMemoryGetMappedBuffer``
     - ``external_memory_get_mapped_buffer``
     -
   * - ``cuExternalMemoryGetMappedMipmappedArray``
     - ``external_memory_get_mapped_mipmapped_array``
     -
   * - ``cuFlushGPUDirectRDMAWrites``
     - ``flush_gpu_direct_rdma_writes``
     -
   * - ``cuFuncGetAttribute``
     - ``func_get_attribute``
     -
   * - ``cuFuncGetModule``
     - ``func_get_module``
     -
   * - ``cuFuncGetName``
     - ``func_get_name``
     -
   * - ``cuFuncGetParamCount``
     - ``func_get_param_count``
     -
   * - ``cuFuncGetParamInfo``
     - ``func_get_param_info``
     -
   * - ``cuFuncIsLoaded``
     - ``func_is_loaded``
     -
   * - ``cuFuncLoad``
     - ``func_load``
     -
   * - ``cuFuncSetAttribute``
     - ``func_set_attribute``
     -
   * - ``cuFuncSetCacheConfig``
     - ``func_set_cache_config``
     -
   * - ``cuGLGetDevices``
     - ``gl_get_devices_v2``
     - Parameters removed: ``cudaDeviceCount``. Returns a NumPy ``int32`` array of device ordinals. The device count is no longer returned.
   * - ``cuGetErrorName``
     - ``get_error_name``
     -
   * - ``cuGetErrorString``
     - ``get_error_string``
     -
   * - ``cuGetExportTable``
     - ``get_export_table``
     -
   * - ``cuGetProcAddress``
     - ``get_proc_address_v2``
     -
   * - ``cuGraphAddBatchMemOpNode``
     - ``graph_add_batch_mem_op_node``
     -
   * - ``cuGraphAddChildGraphNode``
     - ``graph_add_child_graph_node``
     -
   * - ``cuGraphAddDependencies``
     - ``graph_add_dependencies_v2``
     -
   * - ``cuGraphAddEmptyNode``
     - ``graph_add_empty_node``
     -
   * - ``cuGraphAddEventRecordNode``
     - ``graph_add_event_record_node``
     -
   * - ``cuGraphAddEventWaitNode``
     - ``graph_add_event_wait_node``
     -
   * - ``cuGraphAddExternalSemaphoresSignalNode``
     - ``graph_add_external_semaphores_signal_node``
     -
   * - ``cuGraphAddExternalSemaphoresWaitNode``
     - ``graph_add_external_semaphores_wait_node``
     -
   * - ``cuGraphAddHostNode``
     - ``graph_add_host_node``
     -
   * - ``cuGraphAddKernelNode``
     - ``graph_add_kernel_node_v2``
     -
   * - ``cuGraphAddMemAllocNode``
     - ``graph_add_mem_alloc_node``
     -
   * - ``cuGraphAddMemFreeNode``
     - ``graph_add_mem_free_node``
     -
   * - ``cuGraphAddMemcpyNode``
     - ``graph_add_memcpy_node``
     -
   * - ``cuGraphAddMemsetNode``
     - ``graph_add_memset_node``
     -
   * - ``cuGraphAddNode``
     - ``graph_add_node_v2``
     -
   * - ``cuGraphAddNode``
     - ``graph_add_node_v3``
     -
   * - ``cuGraphBatchMemOpNodeGetParams``
     - ``graph_batch_mem_op_node_get_params``
     -
   * - ``cuGraphBatchMemOpNodeSetParams``
     - ``graph_batch_mem_op_node_set_params``
     -
   * - ``cuGraphChildGraphNodeGetGraph``
     - ``graph_child_graph_node_get_graph``
     -
   * - ``cuGraphClone``
     - ``graph_clone``
     -
   * - ``cuGraphConditionalHandleCreate``
     - ``graph_conditional_handle_create``
     -
   * - ``cuGraphCreate``
     - ``graph_create``
     -
   * - ``cuGraphDebugDotPrint``
     - ``graph_debug_dot_print``
     -
   * - ``cuGraphDestroy``
     - ``graph_destroy``
     -
   * - ``cuGraphDestroyNode``
     - ``graph_destroy_node``
     -
   * - ``cuGraphEventRecordNodeGetEvent``
     - ``graph_event_record_node_get_event``
     -
   * - ``cuGraphEventRecordNodeSetEvent``
     - ``graph_event_record_node_set_event``
     -
   * - ``cuGraphEventWaitNodeGetEvent``
     - ``graph_event_wait_node_get_event``
     -
   * - ``cuGraphEventWaitNodeSetEvent``
     - ``graph_event_wait_node_set_event``
     -
   * - ``cuGraphExecBatchMemOpNodeSetParams``
     - ``graph_exec_batch_mem_op_node_set_params``
     -
   * - ``cuGraphExecChildGraphNodeSetParams``
     - ``graph_exec_child_graph_node_set_params``
     -
   * - ``cuGraphExecDestroy``
     - ``graph_exec_destroy``
     -
   * - ``cuGraphExecEventRecordNodeSetEvent``
     - ``graph_exec_event_record_node_set_event``
     -
   * - ``cuGraphExecEventWaitNodeSetEvent``
     - ``graph_exec_event_wait_node_set_event``
     -
   * - ``cuGraphExecExternalSemaphoresSignalNodeSetParams``
     - ``graph_exec_external_semaphores_signal_node_set_params``
     -
   * - ``cuGraphExecExternalSemaphoresWaitNodeSetParams``
     - ``graph_exec_external_semaphores_wait_node_set_params``
     -
   * - ``cuGraphExecGetFlags``
     - ``graph_exec_get_flags``
     -
   * - ``cuGraphExecGetId``
     - ``graph_exec_get_id``
     -
   * - ``cuGraphExecHostNodeSetParams``
     - ``graph_exec_host_node_set_params``
     -
   * - ``cuGraphExecKernelNodeSetParams``
     - ``graph_exec_kernel_node_set_params_v2``
     -
   * - ``cuGraphExecMemcpyNodeSetParams``
     - ``graph_exec_memcpy_node_set_params``
     -
   * - ``cuGraphExecMemsetNodeSetParams``
     - ``graph_exec_memset_node_set_params``
     -
   * - ``cuGraphExecNodeSetParams``
     - ``graph_exec_node_set_params``
     -
   * - ``cuGraphExecUpdate``
     - ``graph_exec_update_v2``
     -
   * - ``cuGraphExternalSemaphoresSignalNodeGetParams``
     - ``graph_external_semaphores_signal_node_get_params``
     -
   * - ``cuGraphExternalSemaphoresSignalNodeSetParams``
     - ``graph_external_semaphores_signal_node_set_params``
     -
   * - ``cuGraphExternalSemaphoresWaitNodeGetParams``
     - ``graph_external_semaphores_wait_node_get_params``
     -
   * - ``cuGraphExternalSemaphoresWaitNodeSetParams``
     - ``graph_external_semaphores_wait_node_set_params``
     -
   * - ``cuGraphGetEdges``
     - ``graph_get_edges``
     - Parameters removed: ``numEdges``. Returns ``(from, to, edge_data)``. The edge count is no longer returned.
   * - ``cuGraphGetEdges``
     - ``graph_get_edges_v2``
     - Parameters removed: ``numEdges``. Returns ``(from, to, edge_data)``. The edge count is no longer returned.
   * - ``cuGraphGetId``
     - ``graph_get_id``
     -
   * - ``cuGraphGetNodes``
     - ``graph_get_nodes``
     - Parameters removed: ``numNodes``. Returns an array of node handles. The count is no longer returned.
   * - ``cuGraphGetRootNodes``
     - ``graph_get_root_nodes``
     - Parameters removed: ``numRootNodes``. Returns an array of node handles. The count is no longer returned.
   * - ``cuGraphHostNodeGetParams``
     - ``graph_host_node_get_params``
     -
   * - ``cuGraphHostNodeSetParams``
     - ``graph_host_node_set_params``
     -
   * - ``cuGraphInstantiate``
     - ``graph_instantiate``
     -
   * - ``cuGraphInstantiateWithParams``
     - ``graph_instantiate_with_params``
     -
   * - ``cuGraphKernelNodeCopyAttributes``
     - ``graph_kernel_node_copy_attributes``
     -
   * - ``cuGraphKernelNodeGetAttribute``
     - ``graph_kernel_node_get_attribute``
     -
   * - ``cuGraphKernelNodeGetParams``
     - ``graph_kernel_node_get_params_v2``
     -
   * - ``cuGraphKernelNodeSetAttribute``
     - ``graph_kernel_node_set_attribute``
     -
   * - ``cuGraphKernelNodeSetParams``
     - ``graph_kernel_node_set_params_v2``
     -
   * - ``cuGraphLaunch``
     - ``graph_launch``
     -
   * - ``cuGraphMemAllocNodeGetParams``
     - ``graph_mem_alloc_node_get_params``
     -
   * - ``cuGraphMemFreeNodeGetParams``
     - ``graph_mem_free_node_get_params``
     -
   * - ``cuGraphMemcpyNodeGetParams``
     - ``graph_memcpy_node_get_params``
     -
   * - ``cuGraphMemcpyNodeSetParams``
     - ``graph_memcpy_node_set_params``
     -
   * - ``cuGraphMemsetNodeGetParams``
     - ``graph_memset_node_get_params``
     -
   * - ``cuGraphMemsetNodeSetParams``
     - ``graph_memset_node_set_params``
     -
   * - ``cuGraphNodeFindInClone``
     - ``graph_node_find_in_clone``
     -
   * - ``cuGraphNodeGetContainingGraph``
     - ``graph_node_get_containing_graph``
     -
   * - ``cuGraphNodeGetDependencies``
     - ``graph_node_get_dependencies``
     - Parameters removed: ``numDependencies``. Returns ``(dependencies, edge_data)``. The count is no longer returned.
   * - ``cuGraphNodeGetDependencies``
     - ``graph_node_get_dependencies_v2``
     - Parameters removed: ``numDependencies``. Returns ``(dependencies, edge_data)``. The count is no longer returned.
   * - ``cuGraphNodeGetDependentNodes``
     - ``graph_node_get_dependent_nodes``
     - Parameters removed: ``numDependentNodes``. Returns ``(dependent_nodes, edge_data)``. The count is no longer returned.
   * - ``cuGraphNodeGetDependentNodes``
     - ``graph_node_get_dependent_nodes_v2``
     - Parameters removed: ``numDependentNodes``. Returns ``(dependent_nodes, edge_data)``. The count is no longer returned.
   * - ``cuGraphNodeGetEnabled``
     - ``graph_node_get_enabled``
     -
   * - ``cuGraphNodeGetLocalId``
     - ``graph_node_get_local_id``
     -
   * - ``cuGraphNodeGetParams``
     - ``graph_node_get_params``
     -
   * - ``cuGraphNodeGetToolsId``
     - ``graph_node_get_tools_id``
     -
   * - ``cuGraphNodeGetType``
     - ``graph_node_get_type``
     -
   * - ``cuGraphNodeSetEnabled``
     - ``graph_node_set_enabled``
     -
   * - ``cuGraphNodeSetParams``
     - ``graph_node_set_params``
     -
   * - ``cuGraphNodeSetParams``
     - ``graph_node_set_params_v2``
     -
   * - ``cuGraphReleaseUserObject``
     - ``graph_release_user_object``
     -
   * - ``cuGraphRemoveDependencies``
     - ``graph_remove_dependencies_v2``
     -
   * - ``cuGraphRetainUserObject``
     - ``graph_retain_user_object``
     -
   * - ``cuGraphUpload``
     - ``graph_upload``
     -
   * - ``cuGraphicsEGLRegisterImage``
     - ``graphics_egl_register_image``
     -
   * - ``cuGraphicsGLRegisterBuffer``
     - ``graphics_gl_register_buffer``
     -
   * - ``cuGraphicsGLRegisterImage``
     - ``graphics_gl_register_image``
     -
   * - ``cuGraphicsMapResources``
     - ``graphics_map_resources``
     -
   * - ``cuGraphicsResourceGetMappedEglFrame``
     - ``graphics_resource_get_mapped_egl_frame``
     - Output ``eglFrame`` is now an input: the caller supplies the address of a buffer to fill.
   * - ``cuGraphicsResourceGetMappedMipmappedArray``
     - ``graphics_resource_get_mapped_mipmapped_array``
     -
   * - ``cuGraphicsResourceGetMappedPointer``
     - ``graphics_resource_get_mapped_pointer_v2``
     -
   * - ``cuGraphicsResourceSetMapFlags``
     - ``graphics_resource_set_map_flags_v2``
     -
   * - ``cuGraphicsSubResourceGetMappedArray``
     - ``graphics_sub_resource_get_mapped_array``
     -
   * - ``cuGraphicsUnmapResources``
     - ``graphics_unmap_resources``
     -
   * - ``cuGraphicsUnregisterResource``
     - ``graphics_unregister_resource``
     -
   * - ``cuGraphicsVDPAURegisterOutputSurface``
     - ``graphics_vdpau_register_output_surface``
     -
   * - ``cuGraphicsVDPAURegisterVideoSurface``
     - ``graphics_vdpau_register_video_surface``
     -
   * - ``cuGreenCtxCreate``
     - ``green_ctx_create``
     -
   * - ``cuGreenCtxDestroy``
     - ``green_ctx_destroy``
     -
   * - ``cuGreenCtxGetDevResource``
     - ``green_ctx_get_dev_resource``
     -
   * - ``cuGreenCtxGetId``
     - ``green_ctx_get_id``
     -
   * - ``cuGreenCtxRecordEvent``
     - ``green_ctx_record_event``
     -
   * - ``cuGreenCtxStreamCreate``
     - ``green_ctx_stream_create``
     -
   * - ``cuGreenCtxWaitEvent``
     - ``green_ctx_wait_event``
     -
   * - ``cuImportExternalMemory``
     - ``import_external_memory``
     -
   * - ``cuImportExternalSemaphore``
     - ``import_external_semaphore``
     -
   * - ``cuInit``
     - ``init``
     -
   * - ``cuIpcCloseMemHandle``
     - ``ipc_close_mem_handle``
     -
   * - ``cuIpcGetEventHandle``
     - ``ipc_get_event_handle``
     -
   * - ``cuIpcGetMemHandle``
     - ``ipc_get_mem_handle``
     -
   * - ``cuIpcOpenEventHandle``
     - ``ipc_open_event_handle``
     -
   * - ``cuIpcOpenMemHandle``
     - ``ipc_open_mem_handle_v2``
     -
   * - ``cuKernelGetAttribute``
     - ``kernel_get_attribute``
     -
   * - ``cuKernelGetFunction``
     - ``kernel_get_function``
     -
   * - ``cuKernelGetLibrary``
     - ``kernel_get_library``
     -
   * - ``cuKernelGetName``
     - ``kernel_get_name``
     -
   * - ``cuKernelGetParamCount``
     - ``kernel_get_param_count``
     -
   * - ``cuKernelGetParamInfo``
     - ``kernel_get_param_info``
     -
   * - ``cuKernelSetAttribute``
     - ``kernel_set_attribute``
     -
   * - ``cuKernelSetCacheConfig``
     - ``kernel_set_cache_config``
     -
   * - ``cuLaunchCooperativeKernel``
     - ``launch_cooperative_kernel``
     -
   * - ``cuLaunchCooperativeKernelMultiDevice``
     - ``launch_cooperative_kernel_multi_device``
     -
   * - ``cuLaunchHostFunc``
     - ``launch_host_func``
     -
   * - ``cuLaunchHostFunc``
     - ``launch_host_func_v2``
     - Parameters added: ``sync_mode``.
   * - ``cuLaunchKernel``
     - ``launch_kernel``
     -
   * - ``cuLaunchKernelEx``
     - ``launch_kernel_ex``
     -
   * - ``cuLibraryEnumerateKernels``
     - ``library_enumerate_kernels``
     - Parameters removed: ``numKernels``. Returns an array of kernel handles.
   * - ``cuLibraryGetGlobal``
     - ``library_get_global``
     -
   * - ``cuLibraryGetKernel``
     - ``library_get_kernel``
     -
   * - ``cuLibraryGetKernelCount``
     - ``library_get_kernel_count``
     -
   * - ``cuLibraryGetManaged``
     - ``library_get_managed``
     -
   * - ``cuLibraryGetModule``
     - ``library_get_module``
     -
   * - ``cuLibraryGetUnifiedFunction``
     - ``library_get_unified_function``
     -
   * - ``cuLibraryLoadData``
     - ``library_load_data``
     -
   * - ``cuLibraryLoadFromFile``
     - ``library_load_from_file``
     -
   * - ``cuLibraryUnload``
     - ``library_unload``
     -
   * - ``cuLinkAddData``
     - ``link_add_data_v2``
     -
   * - ``cuLinkAddFile``
     - ``link_add_file_v2``
     -
   * - ``cuLinkComplete``
     - ``link_complete``
     - Returns the linked image as ``bytes``. ``sizeOut`` is no longer returned.
   * - ``cuLinkCreate``
     - ``link_create_v2``
     -
   * - ``cuLinkDestroy``
     - ``link_destroy``
     -
   * - ``cuLogicalEndpointAddDevice``
     - ``logical_endpoint_add_device``
     -
   * - ``cuLogicalEndpointBindAddr``
     - ``logical_endpoint_bind_addr``
     -
   * - ``cuLogicalEndpointBindMem``
     - ``logical_endpoint_bind_mem``
     -
   * - ``cuLogicalEndpointCreate``
     - ``logical_endpoint_create``
     -
   * - ``cuLogicalEndpointDestroy``
     - ``logical_endpoint_destroy``
     -
   * - ``cuLogicalEndpointExport``
     - ``logical_endpoint_export``
     -
   * - ``cuLogicalEndpointGetLimits``
     - ``logical_endpoint_get_limits``
     -
   * - ``cuLogicalEndpointIdRelease``
     - ``logical_endpoint_id_release``
     -
   * - ``cuLogicalEndpointIdReserve``
     - ``logical_endpoint_id_reserve``
     -
   * - ``cuLogicalEndpointImport``
     - ``logical_endpoint_import``
     -
   * - ``cuLogicalEndpointQuery``
     - ``logical_endpoint_query``
     -
   * - ``cuLogicalEndpointUnbind``
     - ``logical_endpoint_unbind``
     -
   * - ``cuLogsCurrent``
     - ``logs_current``
     -
   * - ``cuLogsDumpToFile``
     - ``logs_dump_to_file``
     -
   * - ``cuLogsDumpToMemory``
     - ``logs_dump_to_memory``
     -
   * - ``cuLogsRegisterCallback``
     - ``logs_register_callback``
     -
   * - ``cuLogsUnregisterCallback``
     - ``logs_unregister_callback``
     -
   * - ``cuMemAddressFree``
     - ``mem_address_free``
     -
   * - ``cuMemAddressReserve``
     - ``mem_address_reserve``
     -
   * - ``cuMemAdvise``
     - ``mem_advise_v2``
     -
   * - ``cuMemAlloc``
     - ``mem_alloc_v2``
     -
   * - ``cuMemAllocAsync``
     - ``mem_alloc_async``
     -
   * - ``cuMemAllocFromPoolAsync``
     - ``mem_alloc_from_pool_async``
     -
   * - ``cuMemAllocHost``
     - ``mem_alloc_host_v2``
     -
   * - ``cuMemAllocManaged``
     - ``mem_alloc_managed``
     -
   * - ``cuMemAllocPitch``
     - ``mem_alloc_pitch_v2``
     -
   * - ``cuMemBatchDecompressAsync``
     - ``mem_batch_decompress_async``
     -
   * - ``cuMemCreate``
     - ``mem_create``
     -
   * - ``cuMemDiscardAndPrefetchBatchAsync``
     - ``mem_discard_and_prefetch_batch_async``
     -
   * - ``cuMemDiscardBatchAsync``
     - ``mem_discard_batch_async``
     -
   * - ``cuMemExportToShareableHandle``
     - ``mem_export_to_shareable_handle``
     -
   * - ``cuMemFree``
     - ``mem_free_v2``
     -
   * - ``cuMemFreeAsync``
     - ``mem_free_async``
     -
   * - ``cuMemFreeHost``
     - ``mem_free_host``
     -
   * - ``cuMemGetAccess``
     - ``mem_get_access``
     -
   * - ``cuMemGetAddressRange``
     - ``mem_get_address_range_v2``
     -
   * - ``cuMemGetAllocationGranularity``
     - ``mem_get_allocation_granularity``
     -
   * - ``cuMemGetAllocationPropertiesFromHandle``
     - ``mem_get_allocation_properties_from_handle``
     -
   * - ``cuMemGetDefaultMemPool``
     - ``mem_get_default_mem_pool``
     -
   * - ``cuMemGetHandleForAddressRange``
     - ``mem_get_handle_for_address_range``
     -
   * - ``cuMemGetInfo``
     - ``mem_get_info_v2``
     -
   * - ``cuMemGetLocationInfo``
     - ``mem_get_location_info``
     - Parameters removed: ``location_out``. Returns the array of ``MemLocation_v1`` entries.
   * - ``cuMemGetMemPool``
     - ``mem_get_mem_pool``
     -
   * - ``cuMemHostAlloc``
     - ``mem_host_alloc``
     -
   * - ``cuMemHostGetDevicePointer``
     - ``mem_host_get_device_pointer_v2``
     -
   * - ``cuMemHostGetFlags``
     - ``mem_host_get_flags``
     -
   * - ``cuMemHostRegister``
     - ``mem_host_register_v2``
     -
   * - ``cuMemHostUnregister``
     - ``mem_host_unregister``
     -
   * - ``cuMemImportFromShareableHandle``
     - ``mem_import_from_shareable_handle``
     -
   * - ``cuMemMap``
     - ``mem_map``
     -
   * - ``cuMemMapArrayAsync``
     - ``mem_map_array_async``
     -
   * - ``cuMemPoolCreate``
     - ``mem_pool_create``
     -
   * - ``cuMemPoolDestroy``
     - ``mem_pool_destroy``
     -
   * - ``cuMemPoolExportPointer``
     - ``mem_pool_export_pointer``
     -
   * - ``cuMemPoolExportToShareableHandle``
     - ``mem_pool_export_to_shareable_handle``
     -
   * - ``cuMemPoolGetAccess``
     - ``mem_pool_get_access``
     -
   * - ``cuMemPoolGetAttribute``
     - ``mem_pool_get_attribute``
     -
   * - ``cuMemPoolImportFromShareableHandle``
     - ``mem_pool_import_from_shareable_handle``
     -
   * - ``cuMemPoolImportPointer``
     - ``mem_pool_import_pointer``
     -
   * - ``cuMemPoolSetAccess``
     - ``mem_pool_set_access``
     -
   * - ``cuMemPoolSetAttribute``
     - ``mem_pool_set_attribute``
     -
   * - ``cuMemPoolTrimTo``
     - ``mem_pool_trim_to``
     -
   * - ``cuMemPrefetchAsync``
     - ``mem_prefetch_async_v2``
     -
   * - ``cuMemPrefetchBatchAsync``
     - ``mem_prefetch_batch_async``
     -
   * - ``cuMemRangeGetAttribute``
     - ``mem_range_get_attribute``
     -
   * - ``cuMemRangeGetAttributes``
     - ``mem_range_get_attributes``
     - Parameters removed: ``numAttributes``.
   * - ``cuMemRelease``
     - ``mem_release``
     -
   * - ``cuMemRetainAllocationHandle``
     - ``mem_retain_allocation_handle``
     -
   * - ``cuMemSetAccess``
     - ``mem_set_access``
     -
   * - ``cuMemSetMemPool``
     - ``mem_set_mem_pool``
     -
   * - ``cuMemUnmap``
     - ``mem_unmap``
     -
   * - ``cuMemcpy``
     - ``cu_memcpy``
     -
   * - ``cuMemcpy2D``
     - ``memcpy_2d_v2``
     -
   * - ``cuMemcpy2DAsync``
     - ``memcpy_2d_async_v2``
     -
   * - ``cuMemcpy2DUnaligned``
     - ``memcpy_2d_unaligned_v2``
     -
   * - ``cuMemcpy3D``
     - ``memcpy_3d_v2``
     -
   * - ``cuMemcpy3DAsync``
     - ``memcpy_3d_async_v2``
     -
   * - ``cuMemcpy3DBatchAsync``
     - ``memcpy_3d_batch_async_v2``
     -
   * - ``cuMemcpy3DPeer``
     - ``memcpy_3d_peer``
     -
   * - ``cuMemcpy3DPeerAsync``
     - ``memcpy_3d_peer_async``
     -
   * - ``cuMemcpy3DWithAttributesAsync``
     - ``memcpy_3d_with_attributes_async``
     -
   * - ``cuMemcpyAsync``
     - ``memcpy_async``
     -
   * - ``cuMemcpyAtoA``
     - ``memcpy_atoa_v2``
     -
   * - ``cuMemcpyAtoD``
     - ``memcpy_atod_v2``
     -
   * - ``cuMemcpyAtoH``
     - ``memcpy_atoh_v2``
     -
   * - ``cuMemcpyAtoHAsync``
     - ``memcpy_atoh_async_v2``
     -
   * - ``cuMemcpyBatchAsync``
     - ``memcpy_batch_async``
     - Parameters removed: ``count``, ``numAttrs``.
   * - ``cuMemcpyBatchAsync``
     - ``memcpy_batch_async_v2``
     -
   * - ``cuMemcpyDtoA``
     - ``memcpy_dtoa_v2``
     -
   * - ``cuMemcpyDtoD``
     - ``memcpy_dtod_v2``
     -
   * - ``cuMemcpyDtoDAsync``
     - ``memcpy_dtod_async_v2``
     -
   * - ``cuMemcpyDtoH``
     - ``memcpy_dtoh_v2``
     -
   * - ``cuMemcpyDtoHAsync``
     - ``memcpy_dtoh_async_v2``
     -
   * - ``cuMemcpyHtoA``
     - ``memcpy_htoa_v2``
     -
   * - ``cuMemcpyHtoAAsync``
     - ``memcpy_htoa_async_v2``
     -
   * - ``cuMemcpyHtoD``
     - ``memcpy_htod_v2``
     -
   * - ``cuMemcpyHtoDAsync``
     - ``memcpy_htod_async_v2``
     -
   * - ``cuMemcpyPeer``
     - ``memcpy_peer``
     -
   * - ``cuMemcpyPeerAsync``
     - ``memcpy_peer_async``
     -
   * - ``cuMemcpyWithAttributesAsync``
     - ``memcpy_with_attributes_async``
     -
   * - ``cuMemsetD16``
     - ``memset_d16_v2``
     -
   * - ``cuMemsetD16Async``
     - ``memset_d16_async``
     -
   * - ``cuMemsetD2D16``
     - ``memset_d2d16_v2``
     -
   * - ``cuMemsetD2D16Async``
     - ``memset_d2d16_async``
     -
   * - ``cuMemsetD2D32``
     - ``memset_d2d32_v2``
     -
   * - ``cuMemsetD2D32Async``
     - ``memset_d2d32_async``
     -
   * - ``cuMemsetD2D8``
     - ``memset_d2d8_v2``
     -
   * - ``cuMemsetD2D8Async``
     - ``memset_d2d8_async``
     -
   * - ``cuMemsetD32``
     - ``memset_d32_v2``
     -
   * - ``cuMemsetD32Async``
     - ``memset_d32_async``
     -
   * - ``cuMemsetD8``
     - ``memset_d8_v2``
     -
   * - ``cuMemsetD8Async``
     - ``memset_d8_async``
     -
   * - ``cuMipmappedArrayCreate``
     - ``mipmapped_array_create``
     -
   * - ``cuMipmappedArrayDestroy``
     - ``mipmapped_array_destroy``
     -
   * - ``cuMipmappedArrayGetLevel``
     - ``mipmapped_array_get_level``
     -
   * - ``cuMipmappedArrayGetMemoryRequirements``
     - ``mipmapped_array_get_memory_requirements``
     -
   * - ``cuMipmappedArrayGetSparseProperties``
     - ``mipmapped_array_get_sparse_properties``
     -
   * - ``cuModuleEnumerateFunctions``
     - ``module_enumerate_functions``
     - Parameters removed: ``numFunctions``. Returns an array of function handles.
   * - ``cuModuleGetFunction``
     - ``module_get_function``
     -
   * - ``cuModuleGetFunctionCount``
     - ``module_get_function_count``
     -
   * - ``cuModuleGetGlobal``
     - ``module_get_global_v2``
     -
   * - ``cuModuleGetLoadingMode``
     - ``module_get_loading_mode``
     -
   * - ``cuModuleLoad``
     - ``module_load``
     -
   * - ``cuModuleLoadData``
     - ``module_load_data``
     -
   * - ``cuModuleLoadDataEx``
     - ``module_load_data_ex``
     -
   * - ``cuModuleLoadFatBinary``
     - ``module_load_fat_binary``
     -
   * - ``cuModuleUnload``
     - ``module_unload``
     -
   * - ``cuMulticastAddDevice``
     - ``multicast_add_device``
     -
   * - ``cuMulticastBindAddr``
     - ``multicast_bind_addr``
     -
   * - ``cuMulticastBindAddr``
     - ``multicast_bind_addr_v2``
     - Parameters added: ``dev``.
   * - ``cuMulticastBindMem``
     - ``multicast_bind_mem``
     -
   * - ``cuMulticastBindMem``
     - ``multicast_bind_mem_v2``
     - Parameters added: ``dev``.
   * - ``cuMulticastCreate``
     - ``multicast_create``
     -
   * - ``cuMulticastGetGranularity``
     - ``multicast_get_granularity``
     -
   * - ``cuMulticastUnbind``
     - ``multicast_unbind``
     -
   * - ``cuOccupancyAvailableDynamicSMemPerBlock``
     - ``occupancy_available_dynamic_smem_per_block``
     -
   * - ``cuOccupancyMaxActiveBlocksPerMultiprocessor``
     - ``occupancy_max_active_blocks_per_multiprocessor``
     -
   * - ``cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags``
     - ``occupancy_max_active_blocks_per_multiprocessor_with_flags``
     -
   * - ``cuOccupancyMaxActiveClusters``
     - ``occupancy_max_active_clusters``
     -
   * - ``cuOccupancyMaxPotentialBlockSize``
     - ``occupancy_max_potential_block_size``
     -
   * - ``cuOccupancyMaxPotentialBlockSizeWithFlags``
     - ``occupancy_max_potential_block_size_with_flags``
     -
   * - ``cuOccupancyMaxPotentialClusterSize``
     - ``occupancy_max_potential_cluster_size``
     -
   * - ``cuPointerGetAttribute``
     - ``pointer_get_attribute``
     -
   * - ``cuPointerGetAttributes``
     - ``pointer_get_attributes``
     - Parameters removed: ``numAttributes``.
   * - ``cuPointerSetAttribute``
     - ``pointer_set_attribute``
     -
   * - ``cuProfilerStart``
     - ``profiler_start``
     -
   * - ``cuProfilerStop``
     - ``profiler_stop``
     -
   * - ``cuSignalExternalSemaphoresAsync``
     - ``signal_external_semaphores_async``
     -
   * - ``cuStreamAddCallback``
     - ``stream_add_callback``
     -
   * - ``cuStreamAttachMemAsync``
     - ``stream_attach_mem_async``
     -
   * - ``cuStreamBatchMemOp``
     - ``stream_batch_mem_op_v2``
     -
   * - ``cuStreamBeginCapture``
     - ``stream_begin_capture_v2``
     -
   * - ``cuStreamBeginCaptureToCig``
     - ``stream_begin_capture_to_cig``
     -
   * - ``cuStreamBeginCaptureToGraph``
     - ``stream_begin_capture_to_graph``
     -
   * - ``cuStreamBeginRecaptureToGraph``
     - ``stream_begin_recapture_to_graph``
     -
   * - ``cuStreamCopyAttributes``
     - ``stream_copy_attributes``
     -
   * - ``cuStreamCreate``
     - ``stream_create``
     -
   * - ``cuStreamCreateWithPriority``
     - ``stream_create_with_priority``
     -
   * - ``cuStreamDestroy``
     - ``stream_destroy_v2``
     -
   * - ``cuStreamEndCapture``
     - ``stream_end_capture``
     -
   * - ``cuStreamEndCaptureToCig``
     - ``stream_end_capture_to_cig``
     -
   * - ``cuStreamGetAttribute``
     - ``stream_get_attribute``
     -
   * - ``cuStreamGetCaptureInfo``
     - ``stream_get_capture_info_v2``
     - Returns a 4-tuple. The dependency count is no longer returned and edge data is not included.
   * - ``cuStreamGetCaptureInfo``
     - ``stream_get_capture_info_v3``
     - Returns a 5-tuple. The dependency count is no longer returned.
   * - ``cuStreamGetCtx``
     - ``stream_get_ctx``
     -
   * - ``cuStreamGetCtx``
     - ``stream_get_ctx_v2``
     - Returns a 2-tuple ``(ctx, green_ctx)``. The legacy function returned only the context.
   * - ``cuStreamGetDevResource``
     - ``stream_get_dev_resource``
     -
   * - ``cuStreamGetDevice``
     - ``stream_get_device``
     -
   * - ``cuStreamGetFlags``
     - ``stream_get_flags``
     -
   * - ``cuStreamGetGreenCtx``
     - ``stream_get_green_ctx``
     -
   * - ``cuStreamGetId``
     - ``stream_get_id``
     -
   * - ``cuStreamGetPriority``
     - ``stream_get_priority``
     -
   * - ``cuStreamIsCapturing``
     - ``stream_is_capturing``
     -
   * - ``cuStreamQuery``
     - ``stream_query``
     -
   * - ``cuStreamSetAttribute``
     - ``stream_set_attribute``
     -
   * - ``cuStreamSynchronize``
     - ``stream_synchronize``
     -
   * - ``cuStreamUpdateCaptureDependencies``
     - ``stream_update_capture_dependencies_v2``
     -
   * - ``cuStreamWaitEvent``
     - ``stream_wait_event``
     -
   * - ``cuStreamWaitValue32``
     - ``stream_wait_value32_v2``
     -
   * - ``cuStreamWaitValue64``
     - ``stream_wait_value64_v2``
     -
   * - ``cuStreamWriteValue32``
     - ``stream_write_value32_v2``
     -
   * - ``cuStreamWriteValue64``
     - ``stream_write_value64_v2``
     -
   * - ``cuSurfObjectCreate``
     - ``surf_object_create``
     -
   * - ``cuSurfObjectDestroy``
     - ``surf_object_destroy``
     -
   * - ``cuSurfObjectGetResourceDesc``
     - ``surf_object_get_resource_desc``
     -
   * - ``cuTensorMapEncodeIm2col``
     - ``tensor_map_encode_im2col``
     -
   * - ``cuTensorMapEncodeIm2colWide``
     - ``tensor_map_encode_im2col_wide``
     -
   * - ``cuTensorMapEncodeTiled``
     - ``tensor_map_encode_tiled``
     -
   * - ``cuTensorMapReplaceAddress``
     - ``tensor_map_replace_address``
     -
   * - ``cuTexObjectCreate``
     - ``tex_object_create``
     -
   * - ``cuTexObjectDestroy``
     - ``tex_object_destroy``
     -
   * - ``cuTexObjectGetResourceDesc``
     - ``tex_object_get_resource_desc``
     -
   * - ``cuTexObjectGetResourceViewDesc``
     - ``tex_object_get_resource_view_desc``
     -
   * - ``cuTexObjectGetTextureDesc``
     - ``tex_object_get_texture_desc``
     -
   * - ``cuTexRefSetAddressMode``
     - ``tex_ref_set_address_mode``
     -
   * - ``cuTexRefSetFilterMode``
     - ``tex_ref_set_filter_mode``
     -
   * - ``cuTexRefSetMipmapFilterMode``
     - ``tex_ref_set_mipmap_filter_mode``
     -
   * - ``cuThreadExchangeStreamCaptureMode``
     - ``thread_exchange_stream_capture_mode``
     -
   * - ``cuUserObjectCreate``
     - ``user_object_create``
     -
   * - ``cuUserObjectRelease``
     - ``user_object_release``
     -
   * - ``cuUserObjectRetain``
     - ``user_object_retain``
     -
   * - ``cuVDPAUCtxCreate``
     - ``vdpau_ctx_create_v2``
     -
   * - ``cuVDPAUGetDevice``
     - ``vdpau_get_device``
     -
   * - ``cuWaitExternalSemaphoresAsync``
     - ``wait_external_semaphores_async``
     -
   * - ``cuDeviceComputeCapability``
     - ``device_compute_capability``
     -
   * - ``cuDeviceGetProperties``
     - ``device_get_properties``
     -
   * - ``cuLaunchCooperativeKernelMultiDevice``
     - ``launch_cooperative_kernel_multi_device``
     -
   * - *(none)*
     - ``graph_instantiate_with_flags``
     - Legacy exposes this C function as ``cuGraphInstantiate``.

Deprecated functions
--------------------

These functions are deprecated in the CUDA driver API but are still available in the new module. The general rules above apply. Texture and surface references are handles, so they are ``int`` values like the other handle types.

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Legacy
     - New
     - Notes
   * - ``cuCtxAttach``
     - ``ctx_attach``
     -
   * - ``cuCtxDetach``
     - ``ctx_detach``
     -
   * - ``cuCtxGetSharedMemConfig``
     - ``ctx_get_shared_mem_config``
     -
   * - ``cuCtxSetSharedMemConfig``
     - ``ctx_set_shared_mem_config``
     -
   * - ``cuFuncSetBlockShape``
     - ``func_set_block_shape``
     -
   * - ``cuFuncSetSharedMemConfig``
     - ``func_set_shared_mem_config``
     -
   * - ``cuFuncSetSharedSize``
     - ``func_set_shared_size``
     -
   * - ``cuLaunch``
     - ``launch``
     -
   * - ``cuLaunchGrid``
     - ``launch_grid``
     -
   * - ``cuLaunchGridAsync``
     - ``launch_grid_async``
     -
   * - ``cuModuleGetSurfRef``
     - ``module_get_surf_ref``
     -
   * - ``cuModuleGetTexRef``
     - ``module_get_tex_ref``
     -
   * - ``cuParamSetSize``
     - ``param_set_size``
     -
   * - ``cuParamSetTexRef``
     - ``param_set_tex_ref``
     -
   * - ``cuParamSetf``
     - ``param_setf``
     -
   * - ``cuParamSeti``
     - ``param_seti``
     -
   * - ``cuParamSetv``
     - ``param_setv``
     -
   * - ``cuSurfRefGetArray``
     - ``surf_ref_get_array``
     -
   * - ``cuSurfRefSetArray``
     - ``surf_ref_set_array``
     -
   * - ``cuTexRefCreate``
     - ``tex_ref_create``
     -
   * - ``cuTexRefDestroy``
     - ``tex_ref_destroy``
     -
   * - ``cuTexRefGetAddress``
     - ``tex_ref_get_address_v2``
     -
   * - ``cuTexRefGetAddressMode``
     - ``tex_ref_get_address_mode``
     -
   * - ``cuTexRefGetArray``
     - ``tex_ref_get_array``
     -
   * - ``cuTexRefGetBorderColor``
     - ``tex_ref_get_border_color``
     -
   * - ``cuTexRefGetFilterMode``
     - ``tex_ref_get_filter_mode``
     -
   * - ``cuTexRefGetFlags``
     - ``tex_ref_get_flags``
     -
   * - ``cuTexRefGetFormat``
     - ``tex_ref_get_format``
     -
   * - ``cuTexRefGetMaxAnisotropy``
     - ``tex_ref_get_max_anisotropy``
     -
   * - ``cuTexRefGetMipmapFilterMode``
     - ``tex_ref_get_mipmap_filter_mode``
     -
   * - ``cuTexRefGetMipmapLevelBias``
     - ``tex_ref_get_mipmap_level_bias``
     -
   * - ``cuTexRefGetMipmapLevelClamp``
     - ``tex_ref_get_mipmap_level_clamp``
     -
   * - ``cuTexRefGetMipmappedArray``
     - ``tex_ref_get_mipmapped_array``
     -
   * - ``cuTexRefSetAddress``
     - ``tex_ref_set_address_v2``
     -
   * - ``cuTexRefSetAddress2D``
     - ``tex_ref_set_address2d_v3``
     -
   * - ``cuTexRefSetAddressMode``
     - ``tex_ref_set_address_mode``
     -
   * - ``cuTexRefSetArray``
     - ``tex_ref_set_array``
     -
   * - ``cuTexRefSetBorderColor``
     - ``tex_ref_set_border_color``
     -
   * - ``cuTexRefSetFilterMode``
     - ``tex_ref_set_filter_mode``
     -
   * - ``cuTexRefSetFlags``
     - ``tex_ref_set_flags``
     -
   * - ``cuTexRefSetFormat``
     - ``tex_ref_set_format``
     -
   * - ``cuTexRefSetMaxAnisotropy``
     - ``tex_ref_set_max_anisotropy``
     -
   * - ``cuTexRefSetMipmapFilterMode``
     - ``tex_ref_set_mipmap_filter_mode``
     -
   * - ``cuTexRefSetMipmapLevelBias``
     - ``tex_ref_set_mipmap_level_bias``
     -
   * - ``cuTexRefSetMipmapLevelClamp``
     - ``tex_ref_set_mipmap_level_clamp``
     -
   * - ``cuTexRefSetMipmappedArray``
     - ``tex_ref_set_mipmapped_array``
     -

Classes
-------

The legacy module exposes several aliases for the same class (``X``, ``X_st``, ``X_v1``). Only one legacy name per class is shown. Handle types such as ``CUcontext`` and ``CUstream`` have no class in the new module (see above). Each class is followed by its renamed members. Members that are not listed keep their name apart from the general rename to ``snake_case``.

.. rubric:: ``CUaccessPolicyWindow_v1`` -> ``AccessPolicyWindow_v1``

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Legacy member
     - New member
   * - ``hitProp``
     - ``hit_prop``
   * - ``hitRatio``
     - ``hit_ratio``
   * - ``missProp``
     - ``miss_prop``

.. rubric:: ``CUarrayMapInfo_v1`` -> ``ArrayMapInfo_v1``

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Legacy member
     - New member
   * - ``deviceBitMask``
     - ``device_bit_mask``
   * - ``flags``
     - ``flags_``
   * - ``memHandle``
     - ``mem_handle``
   * - ``memHandleType``
     - ``mem_handle_type``
   * - ``memOperationType``
     - ``mem_operation_type``
   * - ``resourceType``
     - ``resource_type``
   * - ``subresourceType``
     - ``subresource_type``

.. rubric:: ``CUasyncNotificationInfo`` -> ``AsyncNotificationInfo``

.. rubric:: ``CUcheckpointCheckpointArgs`` -> ``CheckpointCheckpointArgs``

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Legacy member
     - New member
   * - ``customStorageInfo_out``
     - ``custom_storage_info_out``

.. rubric:: ``CUcheckpointGpuPair`` -> ``CheckpointGpuPair``

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Legacy member
     - New member
   * - ``newUuid``
     - ``new_uuid``
   * - ``oldUuid``
     - ``old_uuid``

.. rubric:: ``CUcheckpointLockArgs`` -> ``CheckpointLockArgs``

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Legacy member
     - New member
   * - ``timeoutMs``
     - ``timeout_ms``

.. rubric:: ``CUcheckpointRestoreArgs`` -> ``CheckpointRestoreArgs``

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Legacy member
     - New member
   * - ``customStorageInfo_out``
     - ``custom_storage_info_out``
   * - ``gpuPairs``
     - ``gpu_pairs``
   * - ``gpuPairsCount``
     - ``gpu_pairs_count``

.. rubric:: ``CUcheckpointUnlockArgs`` -> ``CheckpointUnlockArgs``

.. rubric:: ``CUcliqueInfo`` -> ``CliqueInfo``

.. rubric:: ``CUctxCigParam`` -> ``CtxCigParam``

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Legacy member
     - New member
   * - ``sharedData``
     - ``shared_data``
   * - ``sharedDataType``
     - ``shared_data_type``

.. rubric:: ``CUctxCreateParams`` -> ``CtxCreateParams``

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Legacy member
     - New member
   * - ``cigParams``
     - ``cig_params``
   * - ``execAffinityParams``
     - ``exec_affinity_params``
   * - ``numExecAffinityParams``
     - ``num_exec_affinity_params``

.. rubric:: ``CUDA_ARRAY3D_DESCRIPTOR_v2`` -> ``Array3dDescriptor_v2``

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Legacy member
     - New member
   * - ``Depth``
     - ``depth``
   * - ``Flags``
     - ``flags_``
   * - ``Format``
     - ``format``
   * - ``Height``
     - ``height``
   * - ``NumChannels``
     - ``num_channels``
   * - ``Width``
     - ``width``

.. rubric:: ``CUDA_ARRAY_DESCRIPTOR_v2`` -> ``ArrayDescriptor_v2``

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Legacy member
     - New member
   * - ``Format``
     - ``format``
   * - ``Height``
     - ``height``
   * - ``NumChannels``
     - ``num_channels``
   * - ``Width``
     - ``width``

.. rubric:: ``CUDA_ARRAY_MEMORY_REQUIREMENTS_v1`` -> ``ArrayMemoryRequirements_v1``

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Legacy member
     - New member
   * - ``size``
     - ``size_``

.. rubric:: ``CUDA_ARRAY_SPARSE_PROPERTIES_v1`` -> ``ArraySparseProperties_v1``

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Legacy member
     - New member
   * - ``flags``
     - ``flags_``
   * - ``miptailFirstLevel``
     - ``miptail_first_level``
   * - ``miptailSize``
     - ``miptail_size``
   * - ``tileExtent``
     - ``tile_extent``

.. rubric:: ``CUDA_BATCH_MEM_OP_NODE_PARAMS_v1`` -> ``BatchMemOpNodeParams_v1``

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Legacy member
     - New member
   * - ``flags``
     - ``flags_``
   * - ``paramArray``
     - ``param_array``

.. rubric:: ``CUDA_BATCH_MEM_OP_NODE_PARAMS_v2`` -> ``BatchMemOpNodeParams_v2``

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Legacy member
     - New member
   * - ``flags``
     - ``flags_``
   * - ``paramArray``
     - ``param_array``

.. rubric:: ``CUDA_CHILD_GRAPH_NODE_PARAMS`` -> ``ChildGraphNodeParams``

.. rubric:: ``CUDA_CONDITIONAL_NODE_PARAMS`` -> ``ConditionalNodeParams``

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Legacy member
     - New member
   * - ``phGraph_out``
     - ``ph_graph_out``
   * - ``size``
     - ``size_``

.. rubric:: ``CUDA_EVENT_RECORD_NODE_PARAMS`` -> ``EventRecordNodeParams``

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Legacy member
     - New member
   * - ``gCtx``
     - ``g_ctx``

.. rubric:: ``CUDA_EVENT_WAIT_NODE_PARAMS`` -> ``EventWaitNodeParams``

.. rubric:: ``CUDA_EXT_SEM_SIGNAL_NODE_PARAMS_v1`` -> ``ExtSemSignalNodeParams_v1``

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Legacy member
     - New member
   * - ``extSemArray``
     - ``ext_sem_array``
   * - ``numExtSems``
     - ``num_ext_sems``
   * - ``paramsArray``
     - ``params_array``

.. rubric:: ``CUDA_EXT_SEM_SIGNAL_NODE_PARAMS_v2`` -> ``ExtSemSignalNodeParams_v2``

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Legacy member
     - New member
   * - ``extSemArray``
     - ``ext_sem_array``
   * - ``gCtx``
     - ``g_ctx``
   * - ``numExtSems``
     - ``num_ext_sems``
   * - ``paramsArray``
     - ``params_array``

.. rubric:: ``CUDA_EXT_SEM_WAIT_NODE_PARAMS_v1`` -> ``ExtSemWaitNodeParams_v1``

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Legacy member
     - New member
   * - ``extSemArray``
     - ``ext_sem_array``
   * - ``numExtSems``
     - ``num_ext_sems``
   * - ``paramsArray``
     - ``params_array``

.. rubric:: ``CUDA_EXT_SEM_WAIT_NODE_PARAMS_v2`` -> ``ExtSemWaitNodeParams_v2``

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Legacy member
     - New member
   * - ``extSemArray``
     - ``ext_sem_array``
   * - ``gCtx``
     - ``g_ctx``
   * - ``numExtSems``
     - ``num_ext_sems``
   * - ``paramsArray``
     - ``params_array``

.. rubric:: ``CUDA_EXTERNAL_MEMORY_BUFFER_DESC_v1`` -> ``ExternalMemoryBufferDesc_v1``

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Legacy member
     - New member
   * - ``flags``
     - ``flags_``
   * - ``size``
     - ``size_``

.. rubric:: ``CUDA_EXTERNAL_MEMORY_HANDLE_DESC_v1`` -> ``ExternalMemoryHandleDesc_v1``

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Legacy member
     - New member
   * - ``flags``
     - ``flags_``
   * - ``size``
     - ``size_``

.. rubric:: ``CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC_v1`` -> ``ExternalMemoryMipmappedArrayDesc_v1``

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Legacy member
     - New member
   * - ``arrayDesc``
     - ``array_desc``
   * - ``numLevels``
     - ``num_levels``

.. rubric:: ``CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_v1`` -> ``ExternalSemaphoreHandleDesc_v1``

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Legacy member
     - New member
   * - ``flags``
     - ``flags_``

.. rubric:: ``CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_v1`` -> ``ExternalSemaphoreSignalParams_v1``

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Legacy member
     - New member
   * - ``flags``
     - ``flags_``

.. rubric:: ``CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_v1`` -> ``ExternalSemaphoreWaitParams_v1``

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Legacy member
     - New member
   * - ``flags``
     - ``flags_``

.. rubric:: ``CUDA_GRAPH_INSTANTIATE_PARAMS`` -> ``GraphInstantiateParams``

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Legacy member
     - New member
   * - ``flags``
     - ``flags_``
   * - ``hErrNode_out``
     - ``h_err_node_out``
   * - ``hUploadStream``
     - ``h_upload_stream``

.. rubric:: ``CUDA_HOST_NODE_PARAMS_v1`` -> ``HostNodeParams_v1``

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Legacy member
     - New member
   * - ``userData``
     - ``user_data``

.. rubric:: ``CUDA_HOST_NODE_PARAMS_v2`` -> ``HostNodeParams_v2``

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Legacy member
     - New member
   * - ``gCtx``
     - ``g_ctx``
   * - ``syncMode``
     - ``sync_mode``
   * - ``userData``
     - ``user_data``

.. rubric:: ``CUDA_KERNEL_NODE_PARAMS`` -> ``KernelNodeParams_v3``

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Legacy member
     - New member
   * - ``blockDimX``
     - ``block_dim_x``
   * - ``blockDimY``
     - ``block_dim_y``
   * - ``blockDimZ``
     - ``block_dim_z``
   * - ``gridDimX``
     - ``grid_dim_x``
   * - ``gridDimY``
     - ``grid_dim_y``
   * - ``gridDimZ``
     - ``grid_dim_z``
   * - ``kernelParams``
     - ``kernel_params``
   * - ``sharedMemBytes``
     - ``shared_mem_bytes``

.. rubric:: ``CUDA_KERNEL_NODE_PARAMS_v2`` -> ``KernelNodeParams_v2``

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Legacy member
     - New member
   * - ``blockDimX``
     - ``block_dim_x``
   * - ``blockDimY``
     - ``block_dim_y``
   * - ``blockDimZ``
     - ``block_dim_z``
   * - ``gridDimX``
     - ``grid_dim_x``
   * - ``gridDimY``
     - ``grid_dim_y``
   * - ``gridDimZ``
     - ``grid_dim_z``
   * - ``kernelParams``
     - ``kernel_params``
   * - ``sharedMemBytes``
     - ``shared_mem_bytes``

.. rubric:: ``CUDA_MEM_ALLOC_NODE_PARAMS_v1`` -> ``MemAllocNodeParams_v1``

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Legacy member
     - New member
   * - ``accessDescCount``
     - ``access_desc_count``
   * - ``accessDescs``
     - ``access_descs``
   * - ``poolProps``
     - ``pool_props``

.. rubric:: ``CUDA_MEM_ALLOC_NODE_PARAMS_v2`` -> ``MemAllocNodeParams_v2``

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Legacy member
     - New member
   * - ``accessDescCount``
     - ``access_desc_count``
   * - ``accessDescs``
     - ``access_descs``
   * - ``poolProps``
     - ``pool_props``

.. rubric:: ``CUDA_MEM_FREE_NODE_PARAMS`` -> ``MemFreeNodeParams``

.. rubric:: ``CUDA_MEMCPY2D_v2`` -> ``MemCpy2d_v2``

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Legacy member
     - New member
   * - ``Height``
     - ``height``
   * - ``WidthInBytes``
     - ``width_in_bytes``
   * - ``dstArray``
     - ``dst_array``
   * - ``dstDevice``
     - ``dst_device``
   * - ``dstHost``
     - ``dst_host``
   * - ``dstMemoryType``
     - ``dst_memory_type``
   * - ``dstPitch``
     - ``dst_pitch``
   * - ``dstXInBytes``
     - ``dst_x_in_bytes``
   * - ``dstY``
     - ``dst_y``
   * - ``srcArray``
     - ``src_array``
   * - ``srcDevice``
     - ``src_device``
   * - ``srcHost``
     - ``src_host``
   * - ``srcMemoryType``
     - ``src_memory_type``
   * - ``srcPitch``
     - ``src_pitch``
   * - ``srcXInBytes``
     - ``src_x_in_bytes``
   * - ``srcY``
     - ``src_y``

.. rubric:: ``CUDA_MEMCPY3D_BATCH_OP_v1`` -> ``MemCpy3dBatchOp_v1``

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Legacy member
     - New member
   * - ``flags``
     - ``flags_``
   * - ``srcAccessOrder``
     - ``src_access_order``

.. rubric:: ``CUDA_MEMCPY3D_PEER_v1`` -> ``MemCpy3dPeer_v1``

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Legacy member
     - New member
   * - ``Depth``
     - ``depth``
   * - ``Height``
     - ``height``
   * - ``WidthInBytes``
     - ``width_in_bytes``
   * - ``dstArray``
     - ``dst_array``
   * - ``dstContext``
     - ``dst_context``
   * - ``dstDevice``
     - ``dst_device``
   * - ``dstHeight``
     - ``dst_height``
   * - ``dstHost``
     - ``dst_host``
   * - ``dstLOD``
     - ``dst_lod``
   * - ``dstMemoryType``
     - ``dst_memory_type``
   * - ``dstPitch``
     - ``dst_pitch``
   * - ``dstXInBytes``
     - ``dst_x_in_bytes``
   * - ``dstY``
     - ``dst_y``
   * - ``dstZ``
     - ``dst_z``
   * - ``srcArray``
     - ``src_array``
   * - ``srcContext``
     - ``src_context``
   * - ``srcDevice``
     - ``src_device``
   * - ``srcHeight``
     - ``src_height``
   * - ``srcHost``
     - ``src_host``
   * - ``srcLOD``
     - ``src_lod``
   * - ``srcMemoryType``
     - ``src_memory_type``
   * - ``srcPitch``
     - ``src_pitch``
   * - ``srcXInBytes``
     - ``src_x_in_bytes``
   * - ``srcY``
     - ``src_y``
   * - ``srcZ``
     - ``src_z``

.. rubric:: ``CUDA_MEMCPY3D_v2`` -> ``MemCpy3d_v2``

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Legacy member
     - New member
   * - ``Depth``
     - ``depth``
   * - ``Height``
     - ``height``
   * - ``WidthInBytes``
     - ``width_in_bytes``
   * - ``dstArray``
     - ``dst_array``
   * - ``dstDevice``
     - ``dst_device``
   * - ``dstHeight``
     - ``dst_height``
   * - ``dstHost``
     - ``dst_host``
   * - ``dstLOD``
     - ``dst_lod``
   * - ``dstMemoryType``
     - ``dst_memory_type``
   * - ``dstPitch``
     - ``dst_pitch``
   * - ``dstXInBytes``
     - ``dst_x_in_bytes``
   * - ``dstY``
     - ``dst_y``
   * - ``dstZ``
     - ``dst_z``
   * - ``srcArray``
     - ``src_array``
   * - ``srcDevice``
     - ``src_device``
   * - ``srcHeight``
     - ``src_height``
   * - ``srcHost``
     - ``src_host``
   * - ``srcLOD``
     - ``src_lod``
   * - ``srcMemoryType``
     - ``src_memory_type``
   * - ``srcPitch``
     - ``src_pitch``
   * - ``srcXInBytes``
     - ``src_x_in_bytes``
   * - ``srcY``
     - ``src_y``
   * - ``srcZ``
     - ``src_z``

.. rubric:: ``CUDA_MEMCPY_NODE_PARAMS`` -> ``MemCpyNodeParams``

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Legacy member
     - New member
   * - ``copyCtx``
     - ``copy_ctx``
   * - ``copyParams``
     - ``copy_params``
   * - ``flags``
     - ``flags_``

.. rubric:: ``CUDA_MEMSET_NODE_PARAMS_v1`` -> ``MemSetNodeParams_v1``

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Legacy member
     - New member
   * - ``elementSize``
     - ``element_size``

.. rubric:: ``CUDA_MEMSET_NODE_PARAMS_v2`` -> ``MemSetNodeParams_v2``

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Legacy member
     - New member
   * - ``elementSize``
     - ``element_size``

.. rubric:: ``CUDA_POINTER_ATTRIBUTE_P2P_TOKENS_v1`` -> ``PointerAttributeP2PTokens_v1``

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Legacy member
     - New member
   * - ``p2pToken``
     - ``p2p_token``
   * - ``vaSpaceToken``
     - ``va_space_token``

.. rubric:: ``CUDA_RESOURCE_DESC_v1`` -> ``ResourceDesc_v1``

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Legacy member
     - New member
   * - ``flags``
     - ``flags_``
   * - ``resType``
     - ``res_type``

.. rubric:: ``CUDA_RESOURCE_VIEW_DESC_v1`` -> ``ResourceViewDesc_v1``

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Legacy member
     - New member
   * - ``firstLayer``
     - ``first_layer``
   * - ``firstMipmapLevel``
     - ``first_mipmap_level``
   * - ``lastLayer``
     - ``last_layer``
   * - ``lastMipmapLevel``
     - ``last_mipmap_level``

.. rubric:: ``CUDA_TEXTURE_DESC_v1`` -> ``TextureDesc_v1``

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Legacy member
     - New member
   * - ``addressMode``
     - ``address_mode``
   * - ``borderColor``
     - ``border_color``
   * - ``filterMode``
     - ``filter_mode``
   * - ``flags``
     - ``flags_``
   * - ``maxAnisotropy``
     - ``max_anisotropy``
   * - ``maxMipmapLevelClamp``
     - ``max_mipmap_level_clamp``
   * - ``minMipmapLevelClamp``
     - ``min_mipmap_level_clamp``
   * - ``mipmapFilterMode``
     - ``mipmap_filter_mode``
   * - ``mipmapLevelBias``
     - ``mipmap_level_bias``

.. rubric:: ``CUdevResource_v1`` -> ``DevResource_v1``

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Legacy member
     - New member
   * - ``nextResource``
     - ``next_resource``
   * - ``wqConfig``
     - ``wq_config``

.. rubric:: ``CUdevSmResource`` -> ``DevSmResource``

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Legacy member
     - New member
   * - ``flags``
     - ``flags_``
   * - ``localityDomainId``
     - ``locality_domain_id``
   * - ``minSmPartitionSize``
     - ``min_sm_partition_size``
   * - ``smCoscheduledAlignment``
     - ``sm_coscheduled_alignment``
   * - ``smCount``
     - ``sm_count``

.. rubric:: ``CUdevWorkqueueConfigResource`` -> ``DevWorkqueueConfigResource``

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Legacy member
     - New member
   * - ``device``
     - ``device_``
   * - ``sharingScope``
     - ``sharing_scope``
   * - ``wqConcurrencyLimit``
     - ``wq_concurrency_limit``

.. rubric:: ``CUdevWorkqueueResource`` -> ``DevWorkqueueResource``

.. rubric:: ``CUexecAffinityParam_v1`` -> ``ExecAffinityParam_v1``

.. rubric:: ``CUexecAffinitySmCount_v1`` -> ``ExecAffinitySmCount_v1``

.. rubric:: ``CUextent3D_v1`` -> ``Extent3D_v1``

.. rubric:: ``CUgraphEdgeData`` -> ``GraphEdgeData``

.. rubric:: ``CUgraphExecUpdateResultInfo_v1`` -> ``GraphExecUpdateResultInfo_v1``

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Legacy member
     - New member
   * - ``errorFromNode``
     - ``error_from_node``
   * - ``errorNode``
     - ``error_node``

.. rubric:: ``CUgraphNodeParams`` -> ``GraphNodeParams``

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Legacy member
     - New member
   * - ``asBytes``
     - ``as_bytes``
   * - ``eventRecord``
     - ``event_record``
   * - ``eventWait``
     - ``event_wait``
   * - ``extSemSignal``
     - ``ext_sem_signal``
   * - ``extSemWait``
     - ``ext_sem_wait``
   * - ``memOp``
     - ``mem_op``

.. rubric:: ``CUipcEventHandle_v1`` -> ``IpcEventHandle_v1``

.. rubric:: ``CUipcMemHandle_v1`` -> ``IpcMemHandle_v1``

.. rubric:: ``CUlaunchAttribute`` -> ``LaunchAttribute``

.. rubric:: ``CUlaunchAttributeValue`` -> ``LaunchAttributeValue``

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Legacy member
     - New member
   * - ``accessPolicyWindow``
     - ``access_policy_window``
   * - ``clusterDim``
     - ``cluster_dim``
   * - ``clusterSchedulingPolicyPreference``
     - ``cluster_scheduling_policy_preference``
   * - ``deviceUpdatableKernelNode``
     - ``device_updatable_kernel_node``
   * - ``launchCompletionEvent``
     - ``launch_completion_event``
   * - ``memSyncDomain``
     - ``mem_sync_domain``
   * - ``memSyncDomainMap``
     - ``mem_sync_domain_map``
   * - ``nvlinkUtilCentricScheduling``
     - ``nvlink_util_centric_scheduling``
   * - ``portableClusterSizeMode``
     - ``portable_cluster_size_mode``
   * - ``preferredClusterDim``
     - ``preferred_cluster_dim``
   * - ``programmaticEvent``
     - ``programmatic_event``
   * - ``programmaticStreamSerializationAllowed``
     - ``programmatic_stream_serialization_allowed``
   * - ``sharedMemCarveout``
     - ``shared_mem_carveout``
   * - ``sharedMemoryMode``
     - ``shared_memory_mode``
   * - ``syncPolicy``
     - ``sync_policy``

.. rubric:: ``CUlaunchConfig`` -> ``LaunchConfig``

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Legacy member
     - New member
   * - ``blockDimX``
     - ``block_dim_x``
   * - ``blockDimY``
     - ``block_dim_y``
   * - ``blockDimZ``
     - ``block_dim_z``
   * - ``gridDimX``
     - ``grid_dim_x``
   * - ``gridDimY``
     - ``grid_dim_y``
   * - ``gridDimZ``
     - ``grid_dim_z``
   * - ``hStream``
     - ``h_stream``
   * - ``numAttrs``
     - ``num_attrs``
   * - ``sharedMemBytes``
     - ``shared_mem_bytes``

.. rubric:: ``CUlaunchMemSyncDomainMap`` -> ``LaunchMemSyncDomainMap``

.. rubric:: ``CUlibraryHostUniversalFunctionAndDataTable`` -> ``LibraryHostUniversalFunctionAndDataTable``

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Legacy member
     - New member
   * - ``dataTable``
     - ``data_table``
   * - ``dataWindowSize``
     - ``data_window_size``
   * - ``functionTable``
     - ``function_table``
   * - ``functionWindowSize``
     - ``function_window_size``

.. rubric:: ``CUlogicalEndpointFabricHandle`` -> ``LogicalEndpointFabricHandle``

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Legacy member
     - New member
   * - ``data``
     - ``data_``

.. rubric:: ``CUlogicalEndpointProp`` -> ``LogicalEndpointProp``

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Legacy member
     - New member
   * - ``flags``
     - ``flags_``
   * - ``ipcHandleTypes``
     - ``ipc_handle_types``
   * - ``size``
     - ``size_``

.. rubric:: ``CUmemAccessDesc_v1`` -> ``MemAccessDesc_v1``

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Legacy member
     - New member
   * - ``flags``
     - ``flags_``

.. rubric:: ``CUmemAllocationProp_v1`` -> ``MemAllocationProp_v1``

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Legacy member
     - New member
   * - ``allocFlags``
     - ``alloc_flags``
   * - ``requestedHandleTypes``
     - ``requested_handle_types``
   * - ``win32HandleMetaData``
     - ``win32handle_meta_data``

.. rubric:: ``CUmemcpy3DOperand_v1`` -> ``MemCpy3dOperand_v1``

.. rubric:: ``CUmemcpyAttributes_v1`` -> ``MemCpyAttributes_v1``

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Legacy member
     - New member
   * - ``dstLocHint``
     - ``dst_loc_hint``
   * - ``flags``
     - ``flags_``
   * - ``srcAccessOrder``
     - ``src_access_order``
   * - ``srcLocHint``
     - ``src_loc_hint``

.. rubric:: ``CUmemDecompressParams`` -> ``MemDecompressParams``

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Legacy member
     - New member
   * - ``dstActBytes``
     - ``dst_act_bytes``
   * - ``dstNumBytes``
     - ``dst_num_bytes``
   * - ``srcNumBytes``
     - ``src_num_bytes``

.. rubric:: ``CUmemFabricHandle_v1`` -> ``MemFabricHandle_v1``

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Legacy member
     - New member
   * - ``data``
     - ``data_``

.. rubric:: ``CUmemLocation_v1`` -> ``MemLocation_v1``

.. rubric:: ``CUmemPoolProps_v1`` -> ``MemPoolProps_v1``

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Legacy member
     - New member
   * - ``allocType``
     - ``alloc_type``
   * - ``gpuDirectRDMACapable``
     - ``gpu_direct_rdma_capable``
   * - ``handleTypes``
     - ``handle_types``
   * - ``maxSize``
     - ``max_size``
   * - ``win32SecurityAttributes``
     - ``win32security_attributes``

.. rubric:: ``CUmemPoolPtrExportData_v1`` -> ``MemPoolPtrExportData_v1``

.. rubric:: ``CUmulticastObjectProp_v1`` -> ``MulticastObjectProp_v1``

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Legacy member
     - New member
   * - ``flags``
     - ``flags_``
   * - ``handleTypes``
     - ``handle_types``
   * - ``numDevices``
     - ``num_devices``
   * - ``size``
     - ``size_``

.. rubric:: ``CUoffset3D_v1`` -> ``Offset3D_v1``

.. rubric:: ``CUstreamBatchMemOpParams_v1`` -> ``StreamBatchMemOpParams_v1``

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Legacy member
     - New member
   * - ``atomicReduction``
     - ``atomic_reduction``
   * - ``flushRemoteWrites``
     - ``flush_remote_writes``
   * - ``memoryBarrier``
     - ``memory_barrier``
   * - ``waitValue``
     - ``wait_value``
   * - ``writeValue``
     - ``write_value``

.. rubric:: ``CUstreamCigCaptureParams`` -> ``StreamCigCaptureParams``

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Legacy member
     - New member
   * - ``streamCigParams``
     - ``stream_cig_params``

.. rubric:: ``CUstreamCigParam`` -> ``StreamCigParam``

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Legacy member
     - New member
   * - ``streamSharedData``
     - ``stream_shared_data``
   * - ``streamSharedDataType``
     - ``stream_shared_data_type``

.. rubric:: ``CUstreamMemOpAtomicReductionParams_st`` -> ``StreamMemOpAtomicReductionParams``

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Legacy member
     - New member
   * - ``dataType``
     - ``data_type``
   * - ``flags``
     - ``flags_``
   * - ``reductionOp``
     - ``reduction_op``

.. rubric:: ``CUstreamMemOpFlushRemoteWritesParams_st`` -> ``StreamMemOpFlushRemoteWritesParams``

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Legacy member
     - New member
   * - ``flags``
     - ``flags_``

.. rubric:: ``CUstreamMemOpMemoryBarrierParams_st`` -> ``StreamMemOpMemoryBarrierParams``

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Legacy member
     - New member
   * - ``flags``
     - ``flags_``

.. rubric:: ``CUstreamMemOpWaitValueParams_st`` -> ``StreamMemOpWaitValueParams``

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Legacy member
     - New member
   * - ``flags``
     - ``flags_``

.. rubric:: ``CUstreamMemOpWriteValueParams_st`` -> ``StreamMemOpWriteValueParams``

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Legacy member
     - New member
   * - ``flags``
     - ``flags_``

.. rubric:: ``CUtensorMap`` -> ``TensorMap``

.. rubric:: ``CUuuid`` -> ``Uuid``

.. rubric:: ``CUdevprop_v1`` -> ``DevProp_v1``

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Legacy member
     - New member
   * - ``SIMDWidth``
     - ``simd_width``

.. rubric:: ``CUDA_LAUNCH_PARAMS_v1`` -> ``LaunchParams_v1``

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Legacy member
     - New member
   * - ``blockDimX``
     - ``block_dim_x``
   * - ``blockDimY``
     - ``block_dim_y``
   * - ``blockDimZ``
     - ``block_dim_z``
   * - ``gridDimX``
     - ``grid_dim_x``
   * - ``gridDimY``
     - ``grid_dim_y``
   * - ``gridDimZ``
     - ``grid_dim_z``
   * - ``hStream``
     - ``h_stream``
   * - ``kernelParams``
     - ``kernel_params``
   * - ``sharedMemBytes``
     - ``shared_mem_bytes``

Enumerations
------------

Enumeration members lose the common C prefix shown in the *Prefix removed* column. For example ``CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK`` becomes ``DeviceAttribute.MAX_THREADS_PER_BLOCK``. The old full C names are not kept as aliases. Members that changed in any other way are listed under the enumeration.

.. list-table::
   :header-rows: 1
   :widths: 38 32 30

   * - Legacy
     - New
     - Prefix removed
   * - ``cl_context_flags``
     - ``ContextFlags``
     - ``NVCL_CTX_SCHED_``
   * - ``CUaccessProperty``
     - ``AccessProperty``
     - ``CU_ACCESS_PROPERTY_``
   * - ``CUaddress_mode``
     - ``AddressMode``
     - ``CU_TR_ADDRESS_MODE_``
   * - ``CUarray_cubemap_face``
     - ``ArrayCubemapFace``
     - ``CU_CUBEMAP_FACE_``
   * - ``CUarray_format``
     - ``ArrayFormat``
     - ``CU_AD_FORMAT_``
   * - ``CUarraySparseSubresourceType``
     - ``ArraySparseSubresourceType``
     - ``CU_ARRAY_SPARSE_SUBRESOURCE_TYPE_``
   * - ``CUasyncNotificationType``
     - ``AsyncNotificationType``
     - ``CU_ASYNC_NOTIFICATION_TYPE_``
   * - ``CUatomicOperation``
     - ``AtomicOperation``
     - ``CU_ATOMIC_OPERATION_``
   * - ``CUatomicOperationCapability``
     - ``AtomicOperationCapability``
     - ``CU_ATOMIC_CAPABILITY_``
   * - ``CUcigDataType``
     - ``CigDataType``
     - ``CIG_DATA_TYPE_``
   * - ``CUcliqueType``
     - ``CliqueType``
     - ``CU_CLIQUE_TYPE_``
   * - ``CUclusterSchedulingPolicy``
     - ``ClusterSchedulingPolicy``
     - ``CU_CLUSTER_SCHEDULING_POLICY_``
   * - ``CUcomputemode``
     - ``ComputeMode``
     - ``CU_COMPUTEMODE_``
   * - ``CUCoredumpGenerationFlags``
     - ``CoredumpGenerationFlags``
     - ``CU_COREDUMP_``
   * - ``CUcoredumpSettings``
     - ``CoredumpSettings``
     - ``CU_COREDUMP_``
   * - ``CUctx_flags``
     - ``CtxFlags``
     - ``CU_CTX_``
   * - ``CUDA_POINTER_ATTRIBUTE_ACCESS_FLAGS``
     - ``PointerAttributeAccessFlags``
     - ``CU_POINTER_ATTRIBUTE_ACCESS_FLAG_``
   * - ``CUdevice_attribute``
     - ``DeviceAttribute``
     - ``CU_DEVICE_ATTRIBUTE_``
   * - ``CUdevice_P2PAttribute``
     - ``DeviceP2PAttribute``
     - ``CU_DEVICE_P2P_ATTRIBUTE_``
   * - ``CUdeviceNumaConfig``
     - ``DeviceNumaConfig``
     - ``CU_DEVICE_NUMA_CONFIG_``
   * - ``CUdevResourceType``
     - ``DevResourceType``
     - ``CU_DEV_RESOURCE_TYPE_``
   * - ``CUdevSmResourceGroup_flags``
     - ``DevSmResourceGroupFlags``
     - ``CU_DEV_SM_RESOURCE_GROUP_``
   * - ``CUdevSmResourceSplitByCount_flags``
     - ``DevSmResourceSplitByCountFlags``
     - ``CU_DEV_SM_RESOURCE_SPLIT_``
   * - ``CUdevWorkqueueConfigScope``
     - ``DevWorkqueueConfigScope``
     - ``CU_WORKQUEUE_SCOPE_``
   * - ``CUdriverProcAddress_flags``
     - ``DriverProcAddressFlags``
     - ``CU_GET_PROC_ADDRESS_``
   * - ``CUdriverProcAddressQueryResult``
     - ``DriverProcAddressQueryResult``
     - ``CU_GET_PROC_ADDRESS_``
   * - ``CUeglColorFormat``
     - ``EglColorFormat``
     - ``CU_EGL_COLOR_FORMAT_``
   * - ``CUeglFrameType``
     - ``EglFrameType``
     - ``CU_EGL_FRAME_TYPE_``
   * - ``CUeglResourceLocationFlags``
     - ``EglResourceLocationFlags``
     - ``CU_EGL_RESOURCE_LOCATION_``
   * - ``CUevent_flags``
     - ``EventFlags``
     - ``CU_EVENT_``
   * - ``CUevent_record_flags``
     - ``EventRecordFlags``
     - ``CU_EVENT_RECORD_``
   * - ``CUevent_sched_flags``
     - ``EventSchedFlags``
     - ``CU_EVENT_SCHED_``
   * - ``CUevent_wait_flags``
     - ``EventWaitFlags``
     - ``CU_EVENT_WAIT_``
   * - ``CUexecAffinityType``
     - ``ExecAffinityType``
     - ``CU_EXEC_AFFINITY_TYPE_``
   * - ``CUexternalMemoryHandleType``
     - ``ExternalMemoryHandleType``
     - ``CU_EXTERNAL_MEMORY_HANDLE_TYPE_``
   * - ``CUexternalSemaphoreHandleType``
     - ``ExternalSemaphoreHandleType``
     - ``CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_``
   * - ``CUfilter_mode``
     - ``FilterMode``
     - ``CU_TR_FILTER_MODE_``
   * - ``CUflushGPUDirectRDMAWritesOptions``
     - ``FlushGPUDirectRDMAWritesOptions``
     - ``CU_FLUSH_GPU_DIRECT_RDMA_WRITES_OPTION_``
   * - ``CUflushGPUDirectRDMAWritesScope``
     - ``FlushGPUDirectRDMAWritesScope``
     - ``CU_FLUSH_GPU_DIRECT_RDMA_WRITES_TO_``
   * - ``CUflushGPUDirectRDMAWritesTarget``
     - ``FlushGPUDirectRDMAWritesTarget``
     - ``CU_FLUSH_GPU_DIRECT_RDMA_WRITES_TARGET_``
   * - ``CUfunc_cache``
     - ``FuncCache``
     - ``CU_FUNC_CACHE_PREFER_``
   * - ``CUfunction_attribute``
     - ``FunctionAttribute``
     - ``CU_FUNC_ATTRIBUTE_``
   * - ``CUfunctionLoadingState``
     - ``FunctionLoadingState``
     - ``CU_FUNCTION_LOADING_STATE_``
   * - ``CUGLmap_flags``
     - ``GLMapFlags``
     - ``CU_GL_MAP_RESOURCE_FLAGS_``
   * - ``CUGPUDirectRDMAWritesOrdering``
     - ``GPUDirectRDMAWritesOrdering``
     - ``CU_GPU_DIRECT_RDMA_WRITES_ORDERING_``
   * - ``CUgraphChildGraphNodeOwnership``
     - ``GraphChildGraphNodeOwnership``
     - ``CU_GRAPH_CHILD_GRAPH_OWNERSHIP_``
   * - ``CUgraphConditionalNodeType``
     - ``GraphConditionalNodeType``
     - ``CU_GRAPH_COND_TYPE_``
   * - ``CUgraphDebugDot_flags``
     - ``GraphDebugDotFlags``
     - ``CU_GRAPH_DEBUG_DOT_FLAGS_``
   * - ``CUgraphDependencyType``
     - ``GraphDependencyType``
     - ``CU_GRAPH_DEPENDENCY_TYPE_``
   * - ``CUgraphExecUpdateResult``
     - ``GraphExecUpdateResult``
     - ``CU_GRAPH_EXEC_UPDATE_``
   * - ``CUgraphicsMapResourceFlags``
     - ``GraphicsMapResourceFlags``
     - ``CU_GRAPHICS_MAP_RESOURCE_FLAGS_``
   * - ``CUgraphicsRegisterFlags``
     - ``GraphicsRegisterFlags``
     - ``CU_GRAPHICS_REGISTER_FLAGS_``
   * - ``CUgraphInstantiate_flags``
     - ``GraphInstantiateFlags``
     - ``CUDA_GRAPH_INSTANTIATE_FLAG_``
   * - ``CUgraphInstantiateResult``
     - ``GraphInstantiateResult``
     - ``CUDA_GRAPH_INSTANTIATE_``
   * - ``CUgraphMem_attribute``
     - ``GraphMemAttribute``
     - ``CU_GRAPH_MEM_ATTR_``
   * - ``CUgraphNodeType``
     - ``GraphNodeType``
     - ``CU_GRAPH_NODE_TYPE_``
   * - ``CUgraphRecaptureStatus``
     - ``GraphRecaptureStatus``
     - ``CU_GRAPH_RECAPTURE_``
   * - ``CUgreenCtxCreate_flags``
     - ``GreenCtxCreateFlags``
     - ``CU_GREEN_CTX_``
   * - ``CUhostTaskSyncMode``
     - ``HostTaskSyncMode``
     - ``CU_HOST_TASK_``
   * - ``CUipcMem_flags``
     - ``IpcMemFlags``
     - ``CU_``
   * - ``CUjit_cacheMode``
     - ``JitCacheMode``
     - ``CU_JIT_CACHE_OPTION_``
   * - ``CUjit_fallback``
     - ``JitFallback``
     - ``CU_PREFER_``
   * - ``CUjit_option``
     - ``JitOption``
     - ``CU_JIT_``
   * - ``CUjit_target``
     - ``JitTarget``
     - ``CU_``
   * - ``CUjitInputType``
     - ``JitInputType``
     - ``CU_JIT_``
   * - ``CUlaunchAttributeID``
     - ``LaunchAttributeID``
     - ``CU_LAUNCH_ATTRIBUTE_``
   * - ``CUlaunchAttributePortableClusterMode``
     - ``LaunchAttributePortableClusterMode``
     - ``CU_LAUNCH_PORTABLE_CLUSTER_MODE_``
   * - ``CUlaunchMemSyncDomain``
     - ``LaunchMemSyncDomain``
     - ``CU_LAUNCH_MEM_SYNC_DOMAIN_``
   * - ``CUlibraryOption``
     - ``LibraryOption``
     - ``CU_LIBRARY_``
   * - ``CUlimit``
     - ``Limit``
     - ``CU_LIMIT_``
   * - ``CUlogicalEndpointFlag``
     - ``LogicalEndpointFlag``
     - ``CU_LOGICAL_ENDPOINT_FLAG_``
   * - ``CUlogicalEndpointIpcHandleType``
     - ``LogicalEndpointIpcHandleType``
     - ``CU_LOGICAL_ENDPOINT_IPC_HANDLE_TYPE_``
   * - ``CUlogicalEndpointType``
     - ``LogicalEndpointType``
     - ``CU_LOGICAL_ENDPOINT_TYPE_``
   * - ``CUlogLevel``
     - ``LogLevel``
     - ``CU_LOG_LEVEL_``
   * - ``CUmem_advise``
     - ``MemAdvise``
     - ``CU_MEM_ADVISE_``
   * - ``CUmem_range_attribute``
     - ``MemRangeAttribute``
     - ``CU_MEM_RANGE_ATTRIBUTE_``
   * - ``CUmemAccess_flags``
     - ``MemAccessFlags``
     - ``CU_MEM_ACCESS_FLAGS_PROT_``
   * - ``CUmemAllocationCompType``
     - ``MemAllocationCompType``
     - ``CU_MEM_ALLOCATION_COMP_``
   * - ``CUmemAllocationGranularity_flags``
     - ``MemAllocationGranularityFlags``
     - ``CU_MEM_ALLOC_GRANULARITY_``
   * - ``CUmemAllocationHandleType``
     - ``MemAllocationHandleType``
     - ``CU_MEM_HANDLE_TYPE_``
   * - ``CUmemAllocationType``
     - ``MemAllocationType``
     - ``CU_MEM_ALLOCATION_TYPE_``
   * - ``CUmemAttach_flags``
     - ``MemAttachFlags``
     - ``CU_MEM_ATTACH_``
   * - ``CUmemcpy3DOperandType``
     - ``MemCpy3DOperandType``
     - ``CU_MEMCPY_OPERAND_TYPE_``
   * - ``CUmemcpyFlags``
     - ``MemCpyFlags``
     - ``CU_MEMCPY_FLAG_``
   * - ``CUmemcpySrcAccessOrder``
     - ``MemCpySrcAccessOrder``
     - ``CU_MEMCPY_SRC_ACCESS_ORDER_``
   * - ``CUmemDecompressAlgorithm``
     - ``MemDecompressAlgorithm``
     - ``CU_MEM_DECOMPRESS_``
   * - ``CUmemHandleType``
     - ``MemHandleType``
     - ``CU_MEM_HANDLE_TYPE_``
   * - ``CUmemLocationType``
     - ``MemLocationType``
     - ``CU_MEM_LOCATION_TYPE_``
   * - ``CUmemOperationType``
     - ``MemOperationType``
     - ``CU_MEM_OPERATION_TYPE_``
   * - ``CUmemorytype``
     - ``MemoryType``
     - ``CU_MEMORYTYPE_``
   * - ``CUmemPool_attribute``
     - ``MemPoolAttribute``
     - ``CU_MEMPOOL_ATTR_``
   * - ``CUmemRangeFlags``
     - ``MemRangeFlags``
     - ``CU_``
   * - ``CUmemRangeHandleType``
     - ``MemRangeHandleType``
     - ``CU_MEM_RANGE_HANDLE_TYPE_``
   * - ``CUmoduleLoadingMode``
     - ``ModuleLoadingMode``
     - ``CU_MODULE_``
   * - ``CUmulticastGranularity_flags``
     - ``MulticastGranularityFlags``
     - ``CU_MULTICAST_GRANULARITY_``
   * - ``CUoccupancy_flags``
     - ``OccupancyFlags``
     - ``CU_OCCUPANCY_``
   * - ``CUoutput_mode``
     - ``OutputMode``
     - ``CU_OUT_``
   * - ``CUpointer_attribute``
     - ``PointerAttribute``
     - ``CU_POINTER_ATTRIBUTE_``
   * - ``CUprocessState``
     - ``ProcessState``
     - ``CU_PROCESS_STATE_``
   * - ``CUresourcetype``
     - ``ResourceType``
     - ``CU_RESOURCE_TYPE_``
   * - ``CUresourceViewFormat``
     - ``ResourceViewFormat``
     - ``CU_RES_VIEW_FORMAT_``
   * - ``CUresult``
     - ``Result``
     - ``CUDA_``
   * - ``CUshared_carveout``
     - ``SharedCarveout``
     - ``CU_SHAREDMEM_CARVEOUT_``
   * - ``CUsharedMemoryMode``
     - ``SharedMemoryMode``
     - ``CU_SHARED_MEMORY_MODE_``
   * - ``CUstream_flags``
     - ``StreamFlags``
     - ``CU_STREAM_``
   * - ``CUstreamAtomicReductionDataType``
     - ``StreamAtomicReductionDataType``
     - ``CU_``
   * - ``CUstreamAtomicReductionOpType``
     - ``StreamAtomicReductionOpType``
     - ``CU_STREAM_ATOMIC_REDUCTION_OP_``
   * - ``CUstreamBatchMemOpType``
     - ``StreamBatchMemOpType``
     - ``CU_STREAM_MEM_OP_``
   * - ``CUstreamCaptureMode``
     - ``StreamCaptureMode``
     - ``CU_STREAM_CAPTURE_MODE_``
   * - ``CUstreamCaptureStatus``
     - ``StreamCaptureStatus``
     - ``CU_STREAM_CAPTURE_STATUS_``
   * - ``CUstreamCigDataType``
     - ``StreamCigDataType``
     - ``STREAM_CIG_DATA_TYPE_``
   * - ``CUstreamMemoryBarrier_flags``
     - ``StreamMemoryBarrierFlags``
     - ``CU_STREAM_MEMORY_BARRIER_TYPE_``
   * - ``CUstreamUpdateCaptureDependencies_flags``
     - ``StreamUpdateCaptureDependenciesFlags``
     - ``CU_STREAM_``
   * - ``CUstreamWaitValue_flags``
     - ``StreamWaitValueFlags``
     - ``CU_STREAM_WAIT_VALUE_``
   * - ``CUstreamWriteValue_flags``
     - ``StreamWriteValueFlags``
     - ``CU_STREAM_WRITE_VALUE_``
   * - ``CUsynchronizationPolicy``
     - ``SynchronizationPolicy``
     - ``CU_SYNC_POLICY_``
   * - ``CUtensorMapDataType``
     - ``TensorMapDataType``
     - ``CU_``
   * - ``CUtensorMapFloatOOBfill``
     - ``TensorMapFloatOOBFill``
     - ``CU_TENSOR_MAP_FLOAT_OOB_FILL_``
   * - ``CUtensorMapIm2ColWideMode``
     - ``TensorMapIm2ColWideMode``
     - ``CU_TENSOR_MAP_IM2COL_WIDE_MODE_``
   * - ``CUtensorMapInterleave``
     - ``TensorMapInterleave``
     - ``CU_``
   * - ``CUtensorMapL2promotion``
     - ``TensorMapL2Promotion``
     - ``CU_TENSOR_MAP_L2_PROMOTION_``
   * - ``CUtensorMapSwizzle``
     - ``TensorMapSwizzle``
     - ``CU_``
   * - ``CUuserObject_flags``
     - ``UserObjectFlags``
     - ``CU_``
   * - ``CUuserObjectRetain_flags``
     - ``UserObjectRetainFlags``
     - ``CU_``
   * - ``cl_event_flags``
     - ``NvclEventFlags``
     - ``NVCL_EVENT_SCHED_``
   * - ``CUsharedconfig``
     - ``SharedConfig``
     - ``CU_SHARED_MEM_CONFIG_``
   * - *(module constants ``CUDA_COOPERATIVE_LAUNCH_MULTI_DEVICE_*``)*
     - ``CooperativeLaunchMultiDeviceFlags``
     - ``CUDA_COOPERATIVE_LAUNCH_MULTI_DEVICE_``
   * - *(module constants ``CU_LAUNCH_PARAM_*``)*
     - ``LaunchParam``
     - ``CU_LAUNCH_PARAM_``, and the ``_AS_INT`` suffix
   * - *(module constant ``CU_PARAM_TR_DEFAULT``)*
     - ``ParamTexRef``
     - ``CU_PARAM_TR_``
   * - *(module constants ``CU_TRSA_*``)*
     - ``TexRefArrayFlags``
     - ``CU_TRSA_``
   * - *(module constants ``CU_TRSF_*``)*
     - ``TexRefFlags``
     - ``CU_TRSF_``

.. rubric:: ``CUgraphConditionalNodeType`` -> ``GraphConditionalNodeType``

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Legacy
     - New
   * - ``CU_GRAPH_COND_TYPE_IF``
     - ``IF_``

NVRTC
=====

The ``cuda.bindings._v2.nvrtc`` module replaces ``cuda.bindings.nvrtc``.

Functions
---------

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Legacy
     - New
     - Notes
   * - ``nvrtcVersion``
     - ``version``
     - Returns ``(major, minor)`` as a tuple.
   * - ``nvrtcGetNumSupportedArchs``
     - ``get_num_supported_archs``
     -
   * - ``nvrtcGetSupportedArchs``
     - ``get_supported_archs``
     - Returns the sorted architectures as an integer array (legacy returned a ``list[int]``).
   * - ``nvrtcCreateProgram``
     - ``create_program``
     - Parameter ``numHeaders`` removed (the number of headers is taken from ``headers``). ``headers`` and ``include_names`` default to ``None``. Returns the program handle as an ``int``.
   * - ``nvrtcDestroyProgram``
     - ``destroy_program``
     -
   * - ``nvrtcCompileProgram``
     - ``compile_program``
     - Parameter ``numOptions`` removed (taken from ``options``). ``options`` defaults to ``None``.
   * - ``nvrtcGetPTXSize``
     - ``get_ptx_size``
     -
   * - ``nvrtcGetPTX``
     - ``get_ptx``
     - Parameter ``ptx`` (a caller-allocated buffer) removed. Returns the result as ``bytes``.
   * - ``nvrtcGetCUBINSize``
     - ``get_cubin_size``
     -
   * - ``nvrtcGetCUBIN``
     - ``get_cubin``
     - Parameter ``cubin`` (a caller-allocated buffer) removed. Returns the result as ``bytes``.
   * - ``nvrtcGetLTOIRSize``
     - ``get_ltoir_size``
     -
   * - ``nvrtcGetLTOIR``
     - ``get_ltoir``
     - Parameter ``LTOIR`` (a caller-allocated buffer) removed. Returns the result as ``bytes``.
   * - ``nvrtcGetOptiXIRSize``
     - ``get_optix_ir_size``
     -
   * - ``nvrtcGetOptiXIR``
     - ``get_optix_ir``
     - Parameter ``optixir`` (a caller-allocated buffer) removed. Returns the result as ``bytes``.
   * - ``nvrtcGetTileIRSize``
     - ``get_tile_ir_size``
     -
   * - ``nvrtcGetTileIR``
     - ``get_tile_ir``
     - Parameter ``TileIR`` (a caller-allocated buffer) removed. Returns the result as ``bytes``.
   * - ``nvrtcGetProgramLogSize``
     - ``get_program_log_size``
     -
   * - ``nvrtcGetProgramLog``
     - ``get_program_log``
     - Parameter ``log`` (a caller-allocated buffer) removed. Returns the result as ``bytes``.
   * - ``nvrtcAddNameExpression``
     - ``add_name_expression``
     -
   * - ``nvrtcGetLoweredName``
     - ``get_lowered_name``
     - Returns the lowered name as ``bytes``.
   * - ``nvrtcSetFlowCallback``
     - ``set_flow_callback``
     - ``callback`` and ``payload`` are plain integers (a C function pointer and an opaque pointer).
   * - ``nvrtcGetPCHHeapSize``
     - ``get_pch_heap_size``
     -
   * - ``nvrtcSetPCHHeapSize``
     - ``set_pch_heap_size``
     -
   * - ``nvrtcGetPCHCreateStatus``
     - ``get_pch_create_status``
     - Returns the status as an ``int`` holding a ``Result`` value (for example ``Result.ERROR_NO_PCH_CREATE_ATTEMPTED``) instead of a result code.
   * - ``nvrtcGetPCHHeapSizeRequired``
     - ``get_pch_heap_size_required``
     -
   * - ``nvrtcInstallBundledHeaders``
     - ``install_bundled_headers``
     - ``flags`` takes ``InstallHeadersFlag`` values. Returns the optional error message (``bytes`` or ``None``).
   * - ``nvrtcGetBundledHeadersInfo``
     - ``get_bundled_headers_info``
     - Returns ``(BundledHeadersInfo, error_log)`` as a tuple, with ``error_log`` set to ``None`` on success.
   * - ``nvrtcRemoveBundledHeaders``
     - ``remove_bundled_headers``
     - Returns the optional error message (``bytes`` or ``None``).
   * - ``nvrtcGetErrorString``
     - ``get_error_string``
     - Takes the result code as an ``int``. Returns ``str`` (legacy returned ``bytes``).

Classes
-------

.. rubric:: ``nvrtcBundledHeadersInfo`` -> ``BundledHeadersInfo``

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Legacy member
     - New member
   * - ``compressedSize``
     - ``compressed_size``
   * - ``cudaVersionMajor``
     - ``cuda_version_major``
   * - ``cudaVersionMinor``
     - ``cuda_version_minor``
   * - ``numFiles``
     - ``num_files``
   * - ``uncompressedSize``
     - ``uncompressed_size``
   * - ``getPtr()``
     - ``.ptr``

.. rubric:: ``nvrtcProgram`` (removed)

Programs are plain ``int`` handles.

Enumerations
------------

Members lose the prefix shown in the *Prefix removed* column. No members changed in any other way. ``NVRTC_SUCCESS`` becomes ``Result.SUCCESS`` and ``NVRTC_ERROR_OUT_OF_MEMORY`` becomes ``Result.ERROR_OUT_OF_MEMORY``. The installation flags were plain module-level constants in the legacy module and are now members of the ``InstallHeadersFlag`` enumeration.

.. list-table::
   :header-rows: 1
   :widths: 38 32 30

   * - Legacy
     - New
     - Prefix removed
   * - ``nvrtcResult``
     - ``Result``
     - ``NVRTC_``
   * - ``NVRTC_INSTALL_HEADERS_*`` (module constants)
     - ``InstallHeadersFlag``
     - ``NVRTC_INSTALL_HEADERS_``
