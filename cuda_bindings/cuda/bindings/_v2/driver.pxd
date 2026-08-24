# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
#
# This code was automatically generated across versions from 12.9.0 to 13.3.0. Do not modify it directly.
# CYTHON-BINDINGS-GENERATED-DO-NOT-MODIFY-THIS-FILE: format=1; content-sha256=43b4128088be54bdc2833ba7c4d5f0198c68a6fbcb97b62d2201a186a15e3bb7


# <<<< PREAMBLE CONTENT >>>>

from libc.stdint cimport (
    intptr_t,
    uint32_t,
    uint64_t,
)


# <<<< END OF PREAMBLE CONTENT >>>>

from libc.stdint cimport intptr_t

from ..cydriver cimport *
# Named cimport so enum class bodies can use cydriver.TYPE.CONST syntax, which
# causes Cython to emit C-level PyLong_From_TYPE() instead of a Python global
# name lookup that fails at module init time.
cimport cuda.bindings.cydriver as cydriver


###############################################################################
# Types
###############################################################################

ctypedef CUcontext Context
ctypedef CUmodule Module
ctypedef CUfunction Function
ctypedef CUlibrary Library
ctypedef CUkernel Kernel
ctypedef CUarray Array
ctypedef CUmipmappedArray MipmappedArray
ctypedef CUtexref Texref
ctypedef CUsurfref Surfref
ctypedef CUevent Event
ctypedef CUstream Stream
ctypedef CUgraphicsResource GraphicsResource
ctypedef CUexternalMemory ExternalMemory
ctypedef CUexternalSemaphore ExternalSemaphore
ctypedef CUgraph Graph
ctypedef CUgraphNode GraphNode
ctypedef CUgraphExec GraphExec
ctypedef CUmemoryPool MemoryPool
ctypedef CUuserObject UserObject
ctypedef CUgraphDeviceNode GraphDeviceNode
ctypedef CUasyncCallbackHandle AsyncCallbackHandle
ctypedef CUgreenCtx GreenCtx
ctypedef CUlinkState LinkState
ctypedef CUdevResourceDesc DevResourceDesc
ctypedef CUlogsCallbackHandle LogsCallbackHandle
ctypedef CUcoredumpCallbackHandle CoredumpCallbackHandle
ctypedef CUhostFn hostFn
ctypedef CUoccupancyB2DSize occupancyB2DSize
ctypedef CUlogsCallback logsCallback
ctypedef CUDA_KERNEL_NODE_PARAMS_v1 KernelNodeParams_v1
ctypedef CUstreamCallback streamCallback
ctypedef CUstreamCigCaptureParams StreamCigCaptureParams
ctypedef CUcoredumpStatusCallback coredumpStatusCallback
ctypedef CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC_v1 ExternalMemoryMipmappedArrayDesc_v1
ctypedef CUlaunchAttributeValue LaunchAttributeValue
ctypedef CUcheckpointRestoreArgs CheckpointRestoreArgs
ctypedef CUasyncNotificationInfo AsyncNotificationInfo
ctypedef CUDA_EXTERNAL_MEMORY_HANDLE_DESC_v1 ExternalMemoryHandleDesc_v1
ctypedef CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_v1 ExternalSemaphoreHandleDesc_v1
ctypedef CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_v1 ExternalSemaphoreSignalParams_v1
ctypedef CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_v1 ExternalSemaphoreWaitParams_v1
ctypedef CUlaunchAttribute LaunchAttribute
ctypedef CUasyncCallback asyncCallback
ctypedef CUDA_RESOURCE_DESC_v1 ResourceDesc_v1
ctypedef CUlogicalEndpointProp LogicalEndpointProp
ctypedef CUmemcpy3DOperand_v1 Memcpy3DOperand_v1
ctypedef CUDA_MEMCPY3D_BATCH_OP_v1 Memcpy3dBatchOp_v1
ctypedef CUgraphRecaptureCallback graphRecaptureCallback


###############################################################################
# Enum
###############################################################################

ctypedef CUipcMem_flags _IpcMemFlags
ctypedef CUmemAttach_flags _MemAttachFlags
ctypedef CUctx_flags _CtxFlags
ctypedef CUevent_sched_flags _EventSchedFlags
ctypedef CUevent_flags _EventFlags
ctypedef cl_context_flags _ContextFlags
ctypedef CUstream_flags _StreamFlags
ctypedef CUevent_record_flags _EventRecordFlags
ctypedef CUevent_wait_flags _EventWaitFlags
ctypedef CUstreamWaitValue_flags _StreamWaitValueFlags
ctypedef CUstreamWriteValue_flags _StreamWriteValueFlags
ctypedef CUstreamBatchMemOpType _StreamBatchMemOpType
ctypedef CUstreamMemoryBarrier_flags _StreamMemoryBarrierFlags
ctypedef CUoccupancy_flags _OccupancyFlags
ctypedef CUstreamUpdateCaptureDependencies_flags _StreamUpdateCaptureDependenciesFlags
ctypedef CUasyncNotificationType _AsyncNotificationType
ctypedef CUarray_format _ArrayFormat
ctypedef CUaddress_mode _AddressMode
ctypedef CUfilter_mode _FilterMode
ctypedef CUdevice_attribute _DeviceAttribute
ctypedef CUpointer_attribute _PointerAttribute
ctypedef CUfunction_attribute _FunctionAttribute
ctypedef CUfunc_cache _FuncCache
ctypedef CUsharedconfig _Sharedconfig
ctypedef CUshared_carveout _SharedCarveout
ctypedef CUmemorytype _Memorytype
ctypedef CUcomputemode _Computemode
ctypedef CUmem_advise _MemAdvise
ctypedef CUmem_range_attribute _MemRangeAttribute
ctypedef CUjit_option _JitOption
ctypedef CUjit_target _JitTarget
ctypedef CUjit_fallback _JitFallback
ctypedef CUjit_cacheMode _JitCacheMode
ctypedef CUjitInputType _JitInputType
ctypedef CUgraphicsRegisterFlags _GraphicsRegisterFlags
ctypedef CUgraphicsMapResourceFlags _GraphicsMapResourceFlags
ctypedef CUarray_cubemap_face _ArrayCubemapFace
ctypedef CUlimit _Limit
ctypedef CUresourcetype _Resourcetype
ctypedef CUaccessProperty _AccessProperty
ctypedef CUgraphConditionalNodeType _GraphConditionalNodeType
ctypedef CUgraphNodeType _GraphNodeType
ctypedef CUgraphDependencyType _GraphDependencyType
ctypedef CUgraphInstantiateResult _GraphInstantiateResult
ctypedef CUsynchronizationPolicy _SynchronizationPolicy
ctypedef CUclusterSchedulingPolicy _ClusterSchedulingPolicy
ctypedef CUlaunchMemSyncDomain _LaunchMemSyncDomain
ctypedef CUlaunchAttributeID _LaunchAttributeID
ctypedef CUstreamCaptureStatus _StreamCaptureStatus
ctypedef CUstreamCaptureMode _StreamCaptureMode
ctypedef CUdriverProcAddress_flags _DriverProcAddressFlags
ctypedef CUdriverProcAddressQueryResult _DriverProcAddressQueryResult
ctypedef CUexecAffinityType _ExecAffinityType
ctypedef CUcigDataType _CigDataType
ctypedef CUlibraryOption _LibraryOption
ctypedef CUresult _Result
ctypedef CUdevice_P2PAttribute _DeviceP2PAttribute
ctypedef CUresourceViewFormat _ResourceViewFormat
ctypedef CUtensorMapDataType _TensorMapDataType
ctypedef CUtensorMapInterleave _TensorMapInterleave
ctypedef CUtensorMapSwizzle _TensorMapSwizzle
ctypedef CUtensorMapL2promotion _TensorMapL2promotion
ctypedef CUtensorMapFloatOOBfill _TensorMapFloatOOBfill
ctypedef CUtensorMapIm2ColWideMode _TensorMapIm2ColWideMode
ctypedef CUDA_POINTER_ATTRIBUTE_ACCESS_FLAGS _PointerAttributeAccessFlags
ctypedef CUexternalMemoryHandleType _ExternalMemoryHandleType
ctypedef CUexternalSemaphoreHandleType _ExternalSemaphoreHandleType
ctypedef CUmemAllocationHandleType _MemAllocationHandleType
ctypedef CUmemAccess_flags _MemAccessFlags
ctypedef CUmemLocationType _MemLocationType
ctypedef CUmemAllocationType _MemAllocationType
ctypedef CUmemAllocationGranularity_flags _MemAllocationGranularityFlags
ctypedef CUmemRangeHandleType _MemRangeHandleType
ctypedef CUmemRangeFlags _MemRangeFlags
ctypedef CUarraySparseSubresourceType _ArraySparseSubresourceType
ctypedef CUmemOperationType _MemOperationType
ctypedef CUmemHandleType _MemHandleType
ctypedef CUmemAllocationCompType _MemAllocationCompType
ctypedef CUmulticastGranularity_flags _MulticastGranularityFlags
ctypedef CUgraphExecUpdateResult _GraphExecUpdateResult
ctypedef CUmemPool_attribute _MemPoolAttribute
ctypedef CUmemcpyFlags _MemcpyFlags
ctypedef CUmemcpySrcAccessOrder _MemcpySrcAccessOrder
ctypedef CUmemcpy3DOperandType _Memcpy3DOperandType
ctypedef CUgraphMem_attribute _GraphMemAttribute
ctypedef CUgraphChildGraphNodeOwnership _GraphChildGraphNodeOwnership
ctypedef CUflushGPUDirectRDMAWritesOptions _FlushGPUDirectRDMAWritesOptions
ctypedef CUGPUDirectRDMAWritesOrdering _GPUDirectRDMAWritesOrdering
ctypedef CUflushGPUDirectRDMAWritesScope _FlushGPUDirectRDMAWritesScope
ctypedef CUflushGPUDirectRDMAWritesTarget _FlushGPUDirectRDMAWritesTarget
ctypedef CUgraphDebugDot_flags _GraphDebugDotFlags
ctypedef CUuserObject_flags _UserObjectFlags
ctypedef CUuserObjectRetain_flags _UserObjectRetainFlags
ctypedef CUgraphInstantiate_flags _GraphInstantiateFlags
ctypedef CUdeviceNumaConfig _DeviceNumaConfig
ctypedef CUprocessState _ProcessState
ctypedef CUmoduleLoadingMode _ModuleLoadingMode
ctypedef CUmemDecompressAlgorithm _MemDecompressAlgorithm
ctypedef CUfunctionLoadingState _FunctionLoadingState
ctypedef CUcoredumpSettings _CoredumpSettings
ctypedef CUCoredumpGenerationFlags _CoredumpGenerationFlags
ctypedef CUgreenCtxCreate_flags _GreenCtxCreateFlags
ctypedef CUdevResourceType _DevResourceType
ctypedef CUlogLevel _LogLevel
ctypedef CUeglFrameType _EglFrameType
ctypedef CUeglResourceLocationFlags _EglResourceLocationFlags
ctypedef CUeglColorFormat _EglColorFormat
ctypedef CUGLmap_flags _GLmapFlags
ctypedef CUoutput_mode _OutputMode
ctypedef CUatomicOperation _AtomicOperation
ctypedef CUatomicOperationCapability _AtomicOperationCapability
ctypedef CUstreamAtomicReductionOpType _StreamAtomicReductionOpType
ctypedef CUstreamAtomicReductionDataType _StreamAtomicReductionDataType
ctypedef CUdevSmResourceGroup_flags _DevSmResourceGroupFlags
ctypedef CUdevSmResourceSplitByCount_flags _DevSmResourceSplitByCountFlags
ctypedef CUdevWorkqueueConfigScope _DevWorkqueueConfigScope
ctypedef CUhostTaskSyncMode _HostTaskSyncMode
ctypedef CUlaunchAttributePortableClusterMode _LaunchAttributePortableClusterMode
ctypedef CUsharedMemoryMode _SharedMemoryMode
ctypedef CUstreamCigDataType _StreamCigDataType
ctypedef CUlogicalEndpointIpcHandleType _LogicalEndpointIpcHandleType
ctypedef CUlogicalEndpointType _LogicalEndpointType
ctypedef CUlogicalEndpointFlag _LogicalEndpointFlag
ctypedef CUgraphRecaptureStatus _GraphRecaptureStatus


###############################################################################
# Functions
###############################################################################

cpdef str get_error_string(int error)
cpdef str get_error_name(int error)
cpdef object device_get_host_atomic_capabilities(object operations, int dev)
cpdef tuple graph_get_edges(intptr_t h_graph)
cpdef tuple graph_node_get_dependencies(intptr_t h_node)
cpdef tuple graph_node_get_dependent_nodes(intptr_t h_node)
cpdef egl_stream_producer_present_frame(intptr_t conn, intptr_t eglframe, intptr_t p_stream)
cpdef egl_stream_producer_return_frame(intptr_t conn, intptr_t eglframe, intptr_t p_stream)
cpdef graphics_resource_get_mapped_egl_frame(intptr_t egl_frame, intptr_t resource, unsigned int index, unsigned int mip_level)
cpdef intptr_t device_get_nv_sci_sync_attributes(intptr_t nv_sci_sync_attr_list, int dev, int flags)
cpdef object gl_get_devices_v2(int device_list)
cpdef launch_kernel(intptr_t f, unsigned int grid_dim_x, unsigned int grid_dim_y, unsigned int grid_dim_z, unsigned int block_dim_x, unsigned int block_dim_y, unsigned int block_dim_z, unsigned int shared_mem_bytes, intptr_t h_stream, kernel_params, intptr_t extra)
cpdef launch_kernel_ex(config, intptr_t f, kernel_params, intptr_t extra)
cpdef launch_cooperative_kernel(intptr_t f, unsigned int grid_dim_x, unsigned int grid_dim_y, unsigned int grid_dim_z, unsigned int block_dim_x, unsigned int block_dim_y, unsigned int block_dim_z, unsigned int shared_mem_bytes, intptr_t h_stream, kernel_params)

cpdef init(unsigned int flags)
cpdef int driver_get_version() except? -1
cpdef int device_get(int ordinal) except? -1
cpdef int device_get_count() except? -1
cpdef bytes device_get_name(int len, int dev)
cpdef object device_get_uuid_v2(int dev)
cpdef tuple device_get_luid(int dev)
cpdef size_t device_total_mem_v2(int dev) except? 0
cpdef size_t device_get_texture_1d_linear_max_width(int format, unsigned num_channels, int dev) except? 0
cpdef int device_get_attribute(int attrib, int dev) except? -1
cpdef device_set_mem_pool(int dev, intptr_t pool)
cpdef intptr_t device_get_mem_pool(int dev) except? 0
cpdef intptr_t device_get_default_mem_pool(int dev) except? 0
cpdef int device_get_exec_affinity_support(int type, int dev) except? -1
cpdef flush_gpu_direct_rdma_writes(int target, int scope)
cpdef object device_get_properties(int dev)
cpdef tuple device_compute_capability(int dev)
cpdef intptr_t device_primary_ctx_retain(int dev) except? 0
cpdef device_primary_ctx_release_v2(int dev)
cpdef device_primary_ctx_set_flags_v2(int dev, unsigned int flags)
cpdef tuple device_primary_ctx_get_state(int dev)
cpdef device_primary_ctx_reset_v2(int dev)
cpdef intptr_t ctx_create_v2(unsigned int flags, int dev) except? 0
cpdef intptr_t ctx_create_v3(params_array, int num_params, unsigned int flags, int dev) except? 0
cpdef intptr_t ctx_create_v4(ctx_create_params, unsigned int flags, int dev) except? 0
cpdef ctx_destroy_v2(intptr_t ctx)
cpdef ctx_push_current_v2(intptr_t ctx)
cpdef intptr_t ctx_pop_current_v2() except? 0
cpdef ctx_set_current(intptr_t ctx)
cpdef intptr_t ctx_get_current() except? 0
cpdef int ctx_get_device() except? -1
cpdef unsigned int ctx_get_flags() except? 0
cpdef ctx_set_flags(unsigned int flags)
cpdef unsigned long long ctx_get_id(intptr_t ctx) except? 0
cpdef ctx_synchronize()
cpdef ctx_set_limit(int limit, size_t value)
cpdef size_t ctx_get_limit(int limit) except? 0
cpdef int ctx_get_cache_config() except? -1
cpdef ctx_set_cache_config(int config)
cpdef unsigned int ctx_get_api_version(intptr_t ctx) except? 0
cpdef tuple ctx_get_stream_priority_range()
cpdef ctx_reset_persisting_l2cache()
cpdef object ctx_get_exec_affinity(int type)
cpdef ctx_record_event(intptr_t h_ctx, intptr_t h_event)
cpdef ctx_wait_event(intptr_t h_ctx, intptr_t h_event)
cpdef intptr_t ctx_attach(unsigned int flags) except? 0
cpdef ctx_detach(intptr_t ctx)
cpdef int ctx_get_shared_mem_config() except? -1
cpdef ctx_set_shared_mem_config(int config)
cpdef intptr_t module_load(fname) except? 0
cpdef intptr_t module_load_data(image) except? 0
cpdef intptr_t module_load_data_ex(image, unsigned int num_options, intptr_t options, intptr_t option_values) except? 0
cpdef intptr_t module_load_fat_binary(fat_cubin) except? 0
cpdef module_unload(intptr_t hmod)
cpdef int module_get_loading_mode() except? -1
cpdef intptr_t module_get_function(intptr_t hmod, name) except? 0
cpdef unsigned int module_get_function_count(intptr_t mod) except? 0
cpdef object module_enumerate_functions(intptr_t mod)
cpdef tuple module_get_global_v2(intptr_t hmod, name)
cpdef intptr_t link_create_v2(unsigned int num_options, intptr_t options, intptr_t option_values) except? 0
cpdef link_add_data_v2(intptr_t state, int type, intptr_t data, size_t size, name, unsigned int num_options, intptr_t options, intptr_t option_values)
cpdef link_add_file_v2(intptr_t state, int type, path, unsigned int num_options, intptr_t options, intptr_t option_values)
cpdef bytes link_complete(intptr_t state)
cpdef link_destroy(intptr_t state)
cpdef intptr_t module_get_tex_ref(intptr_t hmod, name) except? 0
cpdef intptr_t module_get_surf_ref(intptr_t hmod, name) except? 0
cpdef intptr_t library_load_data(code, intptr_t jit_options, intptr_t jit_options_values, unsigned int num_jit_options, intptr_t library_options, intptr_t library_option_values, unsigned int num_library_options) except? 0
cpdef intptr_t library_load_from_file(file_name, intptr_t jit_options, intptr_t jit_options_values, unsigned int num_jit_options, intptr_t library_options, intptr_t library_option_values, unsigned int num_library_options) except? 0
cpdef library_unload(intptr_t library)
cpdef intptr_t library_get_kernel(intptr_t library, name) except? 0
cpdef unsigned int library_get_kernel_count(intptr_t lib) except? 0
cpdef object library_enumerate_kernels(intptr_t lib)
cpdef intptr_t library_get_module(intptr_t library) except? 0
cpdef intptr_t kernel_get_function(intptr_t kernel) except? 0
cpdef intptr_t kernel_get_library(intptr_t kernel) except? 0
cpdef tuple library_get_global(intptr_t library, name)
cpdef tuple library_get_managed(intptr_t library, name)
cpdef intptr_t library_get_unified_function(intptr_t library, symbol) except? 0
cpdef int kernel_get_attribute(int attrib, intptr_t kernel, int dev) except? -1
cpdef kernel_set_attribute(int attrib, int val, intptr_t kernel, int dev)
cpdef kernel_set_cache_config(intptr_t kernel, int config, int dev)
cpdef tuple kernel_get_param_info(intptr_t kernel, size_t param_index)
cpdef tuple mem_get_info_v2()
cpdef unsigned long long mem_alloc_v2(size_t bytesize) except? 0
cpdef tuple mem_alloc_pitch_v2(size_t width_in_bytes, size_t height, unsigned int element_size_bytes)
cpdef mem_free_v2(unsigned long long dptr)
cpdef tuple mem_get_address_range_v2(unsigned long long dptr)
cpdef intptr_t mem_alloc_host_v2(size_t bytesize) except? 0
cpdef mem_free_host(p)
cpdef intptr_t mem_host_alloc(size_t bytesize, unsigned int flags) except? 0
cpdef unsigned long long mem_host_get_device_pointer_v2(intptr_t p, unsigned int flags) except? 0
cpdef unsigned int mem_host_get_flags(intptr_t p) except? 0
cpdef unsigned long long mem_alloc_managed(size_t bytesize, unsigned int flags) except? 0
cpdef intptr_t device_register_async_notification(int device, intptr_t callback_func, intptr_t user_data) except? 0
cpdef device_unregister_async_notification(int device, intptr_t callback)
cpdef int device_get_by_pci_bus_id(pci_bus_id) except? -1
cpdef bytes device_get_pci_bus_id(int len, int dev)
cpdef object ipc_get_event_handle(intptr_t event)
cpdef intptr_t ipc_open_event_handle(handle) except? 0
cpdef object ipc_get_mem_handle(unsigned long long dptr)
cpdef unsigned long long ipc_open_mem_handle_v2(handle, unsigned int flags) except? 0
cpdef ipc_close_mem_handle(unsigned long long dptr)
cpdef mem_host_register_v2(p, size_t bytesize, unsigned int flags)
cpdef mem_host_unregister(p)
cpdef cu_memcpy(unsigned long long dst, unsigned long long src, size_t byte_count)
cpdef memcpy_peer(unsigned long long dst_device, intptr_t dst_context, unsigned long long src_device, intptr_t src_context, size_t byte_count)
cpdef memcpy_htod_v2(unsigned long long dst_device, src_host, size_t byte_count)
cpdef memcpy_dtoh_v2(dst_host, unsigned long long src_device, size_t byte_count)
cpdef memcpy_dtod_v2(unsigned long long dst_device, unsigned long long src_device, size_t byte_count)
cpdef memcpy_dtoa_v2(intptr_t dst_array, size_t dst_offset, unsigned long long src_device, size_t byte_count)
cpdef memcpy_atod_v2(unsigned long long dst_device, intptr_t src_array, size_t src_offset, size_t byte_count)
cpdef memcpy_htoa_v2(intptr_t dst_array, size_t dst_offset, src_host, size_t byte_count)
cpdef memcpy_atoh_v2(dst_host, intptr_t src_array, size_t src_offset, size_t byte_count)
cpdef memcpy_atoa_v2(intptr_t dst_array, size_t dst_offset, intptr_t src_array, size_t src_offset, size_t byte_count)
cpdef memcpy_2d_v2(p_copy)
cpdef memcpy_2d_unaligned_v2(p_copy)
cpdef memcpy_3d_v2(p_copy)
cpdef memcpy_3d_peer(p_copy)
cpdef memcpy_async(unsigned long long dst, unsigned long long src, size_t byte_count, intptr_t h_stream)
cpdef memcpy_peer_async(unsigned long long dst_device, intptr_t dst_context, unsigned long long src_device, intptr_t src_context, size_t byte_count, intptr_t h_stream)
cpdef memcpy_htod_async_v2(unsigned long long dst_device, src_host, size_t byte_count, intptr_t h_stream)
cpdef memcpy_dtoh_async_v2(dst_host, unsigned long long src_device, size_t byte_count, intptr_t h_stream)
cpdef memcpy_dtod_async_v2(unsigned long long dst_device, unsigned long long src_device, size_t byte_count, intptr_t h_stream)
cpdef memcpy_htoa_async_v2(intptr_t dst_array, size_t dst_offset, src_host, size_t byte_count, intptr_t h_stream)
cpdef memcpy_atoh_async_v2(dst_host, intptr_t src_array, size_t src_offset, size_t byte_count, intptr_t h_stream)
cpdef memcpy_2d_async_v2(p_copy, intptr_t h_stream)
cpdef memcpy_3d_async_v2(p_copy, intptr_t h_stream)
cpdef memcpy_3d_peer_async(p_copy, intptr_t h_stream)
cpdef memset_d8_v2(unsigned long long dst_device, unsigned char uc, size_t n)
cpdef memset_d16_v2(unsigned long long dst_device, unsigned short us, size_t n)
cpdef memset_d32_v2(unsigned long long dst_device, unsigned int ui, size_t n)
cpdef memset_d2d8_v2(unsigned long long dst_device, size_t dst_pitch, unsigned char uc, size_t width, size_t height)
cpdef memset_d2d16_v2(unsigned long long dst_device, size_t dst_pitch, unsigned short us, size_t width, size_t height)
cpdef memset_d2d32_v2(unsigned long long dst_device, size_t dst_pitch, unsigned int ui, size_t width, size_t height)
cpdef memset_d8_async(unsigned long long dst_device, unsigned char uc, size_t n, intptr_t h_stream)
cpdef memset_d16_async(unsigned long long dst_device, unsigned short us, size_t n, intptr_t h_stream)
cpdef memset_d32_async(unsigned long long dst_device, unsigned int ui, size_t n, intptr_t h_stream)
cpdef memset_d2d8_async(unsigned long long dst_device, size_t dst_pitch, unsigned char uc, size_t width, size_t height, intptr_t h_stream)
cpdef memset_d2d16_async(unsigned long long dst_device, size_t dst_pitch, unsigned short us, size_t width, size_t height, intptr_t h_stream)
cpdef memset_d2d32_async(unsigned long long dst_device, size_t dst_pitch, unsigned int ui, size_t width, size_t height, intptr_t h_stream)
cpdef intptr_t array_create_v2(p_allocate_array) except? 0
cpdef object array_get_descriptor_v2(intptr_t h_array)
cpdef object array_get_sparse_properties(intptr_t array)
cpdef object mipmapped_array_get_sparse_properties(intptr_t mipmap)
cpdef object array_get_memory_requirements(intptr_t array, int device)
cpdef object mipmapped_array_get_memory_requirements(intptr_t mipmap, int device)
cpdef intptr_t array_get_plane(intptr_t h_array, unsigned int plane_idx) except? 0
cpdef array_destroy(intptr_t h_array)
cpdef intptr_t array_3d_create_v2(p_allocate_array) except? 0
cpdef object array_3d_get_descriptor_v2(intptr_t h_array)
cpdef intptr_t mipmapped_array_create(p_mipmapped_array_desc, unsigned int num_mipmap_levels) except? 0
cpdef intptr_t mipmapped_array_get_level(intptr_t h_mipmapped_array, unsigned int level) except? 0
cpdef mipmapped_array_destroy(intptr_t h_mipmapped_array)
cpdef mem_get_handle_for_address_range(intptr_t handle, unsigned long long dptr, size_t size, int handle_type, unsigned long long flags)
cpdef mem_batch_decompress_async(params_array, size_t count, unsigned int flags, intptr_t error_index, intptr_t stream)
cpdef unsigned long long mem_address_reserve(size_t size, size_t alignment, unsigned long long addr, unsigned long long flags) except? 0
cpdef mem_address_free(unsigned long long ptr, size_t size)
cpdef unsigned long long mem_create(size_t size, prop, unsigned long long flags) except? 0
cpdef mem_release(unsigned long long handle)
cpdef mem_map(unsigned long long ptr, size_t size, size_t offset, unsigned long long handle, unsigned long long flags)
cpdef mem_map_array_async(map_info_list, unsigned int count, intptr_t h_stream)
cpdef mem_unmap(unsigned long long ptr, size_t size)
cpdef mem_set_access(unsigned long long ptr, size_t size, desc, size_t count)
cpdef unsigned long long mem_get_access(location, unsigned long long ptr) except? 0
cpdef mem_export_to_shareable_handle(intptr_t shareable_handle, unsigned long long handle, int handle_type, unsigned long long flags)
cpdef unsigned long long mem_import_from_shareable_handle(intptr_t os_handle, int sh_handle_type) except? 0
cpdef size_t mem_get_allocation_granularity(prop, int option) except? 0
cpdef mem_get_allocation_properties_from_handle(prop, unsigned long long handle)
cpdef unsigned long long mem_retain_allocation_handle(intptr_t addr) except? 0
cpdef mem_free_async(unsigned long long dptr, intptr_t h_stream)
cpdef unsigned long long mem_alloc_async(size_t bytesize, intptr_t h_stream) except? 0
cpdef mem_pool_trim_to(intptr_t pool, size_t min_bytes_to_keep)
cpdef mem_pool_set_attribute(intptr_t pool, int attr, intptr_t value)
cpdef mem_pool_get_attribute(intptr_t pool, int attr, intptr_t value)
cpdef mem_pool_set_access(intptr_t pool, map, size_t count)
cpdef int mem_pool_get_access(intptr_t mem_pool, location) except? 0
cpdef intptr_t mem_pool_create(pool_props) except? 0
cpdef mem_pool_destroy(intptr_t pool)
cpdef unsigned long long mem_alloc_from_pool_async(size_t bytesize, intptr_t pool, intptr_t h_stream) except? 0
cpdef mem_pool_export_to_shareable_handle(intptr_t handle_out, intptr_t pool, int handle_type, unsigned long long flags)
cpdef intptr_t mem_pool_import_from_shareable_handle(intptr_t handle, int handle_type, unsigned long long flags) except? 0
cpdef object mem_pool_export_pointer(unsigned long long ptr)
cpdef unsigned long long mem_pool_import_pointer(intptr_t pool, share_data) except? 0
cpdef unsigned long long multicast_create(prop) except? 0
cpdef multicast_add_device(unsigned long long mc_handle, int dev)
cpdef multicast_bind_mem(unsigned long long mc_handle, size_t mc_offset, unsigned long long mem_handle, size_t mem_offset, size_t size, unsigned long long flags)
cpdef multicast_bind_addr(unsigned long long mc_handle, size_t mc_offset, unsigned long long memptr, size_t size, unsigned long long flags)
cpdef multicast_unbind(unsigned long long mc_handle, int dev, size_t mc_offset, size_t size)
cpdef size_t multicast_get_granularity(prop, int option) except? 0
cpdef pointer_get_attribute(intptr_t data, int attribute, unsigned long long ptr)
cpdef mem_prefetch_async_v2(unsigned long long dev_ptr, size_t count, location, unsigned int flags, intptr_t h_stream)
cpdef mem_advise_v2(unsigned long long dev_ptr, size_t count, int advice, location)
cpdef mem_range_get_attribute(intptr_t data, size_t data_size, int attribute, unsigned long long dev_ptr, size_t count)
cpdef mem_range_get_attributes(intptr_t data, intptr_t data_sizes, intptr_t attributes, size_t num_attributes, unsigned long long dev_ptr, size_t count)
cpdef pointer_set_attribute(value, int attribute, unsigned long long ptr)
cpdef pointer_get_attributes(unsigned int num_attributes, intptr_t attributes, intptr_t data, unsigned long long ptr)
cpdef intptr_t stream_create(unsigned int flags) except? 0
cpdef intptr_t stream_create_with_priority(unsigned int flags, int priority) except? 0
cpdef int stream_get_priority(intptr_t h_stream) except? -1
cpdef int stream_get_device(intptr_t h_stream) except? -1
cpdef unsigned int stream_get_flags(intptr_t h_stream) except? 0
cpdef unsigned long long stream_get_id(intptr_t h_stream) except? 0
cpdef intptr_t stream_get_ctx(intptr_t h_stream) except? 0
cpdef tuple stream_get_ctx_v2(intptr_t h_stream)
cpdef stream_wait_event(intptr_t h_stream, intptr_t h_event, unsigned int flags)
cpdef stream_add_callback(intptr_t h_stream, intptr_t callback, intptr_t user_data, unsigned int flags)
cpdef stream_begin_capture_v2(intptr_t h_stream, int mode)
cpdef stream_begin_capture_to_graph(intptr_t h_stream, intptr_t h_graph, intptr_t dependencies, dependency_data, size_t num_dependencies, int mode)
cpdef int thread_exchange_stream_capture_mode() except? -1
cpdef intptr_t stream_end_capture(intptr_t h_stream) except? 0
cpdef int stream_is_capturing(intptr_t h_stream) except? -1
cpdef tuple stream_get_capture_info_v2(intptr_t h_stream)
cpdef tuple stream_get_capture_info_v3(intptr_t h_stream)
cpdef stream_update_capture_dependencies_v2(intptr_t h_stream, intptr_t dependencies, dependency_data, size_t num_dependencies, unsigned int flags)
cpdef stream_attach_mem_async(intptr_t h_stream, unsigned long long dptr, size_t length, unsigned int flags)
cpdef stream_query(intptr_t h_stream)
cpdef stream_synchronize(intptr_t h_stream)
cpdef stream_destroy_v2(intptr_t h_stream)
cpdef stream_copy_attributes(intptr_t dst, intptr_t src)
cpdef stream_get_attribute(intptr_t h_stream, int attr, intptr_t value_out)
cpdef stream_set_attribute(intptr_t h_stream, int attr, intptr_t value)
cpdef intptr_t event_create(unsigned int flags) except? 0
cpdef event_record(intptr_t h_event, intptr_t h_stream)
cpdef event_record_with_flags(intptr_t h_event, intptr_t h_stream, unsigned int flags)
cpdef event_query(intptr_t h_event)
cpdef event_synchronize(intptr_t h_event)
cpdef event_destroy_v2(intptr_t h_event)
cpdef float event_elapsed_time_v2(intptr_t h_start, intptr_t h_end) except? -1.0
cpdef intptr_t import_external_memory(intptr_t mem_handle_desc) except? 0
cpdef unsigned long long external_memory_get_mapped_buffer(intptr_t ext_mem, buffer_desc) except? 0
cpdef intptr_t external_memory_get_mapped_mipmapped_array(intptr_t ext_mem, intptr_t mipmap_desc) except? 0
cpdef destroy_external_memory(intptr_t ext_mem)
cpdef intptr_t import_external_semaphore(intptr_t sem_handle_desc) except? 0
cpdef signal_external_semaphores_async(intptr_t ext_sem_array, intptr_t params_array, unsigned int num_ext_sems, intptr_t stream)
cpdef wait_external_semaphores_async(intptr_t ext_sem_array, intptr_t params_array, unsigned int num_ext_sems, intptr_t stream)
cpdef destroy_external_semaphore(intptr_t ext_sem)
cpdef stream_wait_value32_v2(intptr_t stream, unsigned long long addr, uint64_t value, unsigned int flags)
cpdef stream_wait_value64_v2(intptr_t stream, unsigned long long addr, uint64_t value, unsigned int flags)
cpdef stream_write_value32_v2(intptr_t stream, unsigned long long addr, uint64_t value, unsigned int flags)
cpdef stream_write_value64_v2(intptr_t stream, unsigned long long addr, uint64_t value, unsigned int flags)
cpdef stream_batch_mem_op_v2(intptr_t stream, unsigned int count, param_array, unsigned int flags)
cpdef int func_get_attribute(int attrib, intptr_t hfunc) except? -1
cpdef func_set_attribute(intptr_t hfunc, int attrib, int value)
cpdef func_set_cache_config(intptr_t hfunc, int config)
cpdef intptr_t func_get_module(intptr_t hfunc) except? 0
cpdef tuple func_get_param_info(intptr_t func, size_t param_index)
cpdef int func_is_loaded(intptr_t function) except? -1
cpdef func_load(intptr_t function)
cpdef launch_cooperative_kernel_multi_device(launch_params_list, unsigned int num_devices, unsigned int flags)
cpdef launch_host_func(intptr_t h_stream, intptr_t fn, intptr_t user_data)
cpdef func_set_block_shape(intptr_t hfunc, int x, int y, int z)
cpdef func_set_shared_size(intptr_t hfunc, unsigned int bytes)
cpdef param_set_size(intptr_t hfunc, unsigned int numbytes)
cpdef param_seti(intptr_t hfunc, int offset, unsigned int value)
cpdef param_setf(intptr_t hfunc, int offset, float value)
cpdef param_setv(intptr_t hfunc, int offset, intptr_t ptr, unsigned int numbytes)
cpdef launch(intptr_t f)
cpdef launch_grid(intptr_t f, int grid_width, int grid_height)
cpdef launch_grid_async(intptr_t f, int grid_width, int grid_height, intptr_t h_stream)
cpdef param_set_tex_ref(intptr_t hfunc, int texunit, intptr_t h_tex_ref)
cpdef func_set_shared_mem_config(intptr_t hfunc, int config)
cpdef intptr_t graph_create(unsigned int flags) except? 0
cpdef intptr_t graph_add_kernel_node_v2(intptr_t h_graph, intptr_t dependencies, size_t num_dependencies, node_params) except? 0
cpdef graph_kernel_node_get_params_v2(intptr_t h_node, node_params)
cpdef graph_kernel_node_set_params_v2(intptr_t h_node, node_params)
cpdef intptr_t graph_add_memcpy_node(intptr_t h_graph, intptr_t dependencies, size_t num_dependencies, copy_params, intptr_t ctx) except? 0
cpdef graph_memcpy_node_get_params(intptr_t h_node, node_params)
cpdef graph_memcpy_node_set_params(intptr_t h_node, node_params)
cpdef intptr_t graph_add_memset_node(intptr_t h_graph, intptr_t dependencies, size_t num_dependencies, memset_params, intptr_t ctx) except? 0
cpdef object graph_memset_node_get_params(intptr_t h_node)
cpdef graph_memset_node_set_params(intptr_t h_node, node_params)
cpdef intptr_t graph_add_host_node(intptr_t h_graph, intptr_t dependencies, size_t num_dependencies, node_params) except? 0
cpdef graph_host_node_get_params(intptr_t h_node, node_params)
cpdef graph_host_node_set_params(intptr_t h_node, node_params)
cpdef intptr_t graph_add_child_graph_node(intptr_t h_graph, intptr_t dependencies, size_t num_dependencies, intptr_t child_graph) except? 0
cpdef intptr_t graph_child_graph_node_get_graph(intptr_t h_node) except? 0
cpdef intptr_t graph_add_empty_node(intptr_t h_graph, intptr_t dependencies, size_t num_dependencies) except? 0
cpdef intptr_t graph_add_event_record_node(intptr_t h_graph, intptr_t dependencies, size_t num_dependencies, intptr_t event) except? 0
cpdef intptr_t graph_event_record_node_get_event(intptr_t h_node) except? 0
cpdef graph_event_record_node_set_event(intptr_t h_node, intptr_t event)
cpdef intptr_t graph_add_event_wait_node(intptr_t h_graph, intptr_t dependencies, size_t num_dependencies, intptr_t event) except? 0
cpdef intptr_t graph_event_wait_node_get_event(intptr_t h_node) except? 0
cpdef graph_event_wait_node_set_event(intptr_t h_node, intptr_t event)
cpdef intptr_t graph_add_external_semaphores_signal_node(intptr_t h_graph, intptr_t dependencies, size_t num_dependencies, node_params) except? 0
cpdef graph_external_semaphores_signal_node_get_params(intptr_t h_node, params_out)
cpdef graph_external_semaphores_signal_node_set_params(intptr_t h_node, node_params)
cpdef intptr_t graph_add_external_semaphores_wait_node(intptr_t h_graph, intptr_t dependencies, size_t num_dependencies, node_params) except? 0
cpdef graph_external_semaphores_wait_node_get_params(intptr_t h_node, params_out)
cpdef graph_external_semaphores_wait_node_set_params(intptr_t h_node, node_params)
cpdef intptr_t graph_add_batch_mem_op_node(intptr_t h_graph, intptr_t dependencies, size_t num_dependencies, node_params) except? 0
cpdef graph_batch_mem_op_node_get_params(intptr_t h_node, node_params_out)
cpdef graph_batch_mem_op_node_set_params(intptr_t h_node, node_params)
cpdef graph_exec_batch_mem_op_node_set_params(intptr_t h_graph_exec, intptr_t h_node, node_params)
cpdef intptr_t graph_add_mem_alloc_node(intptr_t h_graph, intptr_t dependencies, size_t num_dependencies, node_params) except? 0
cpdef graph_mem_alloc_node_get_params(intptr_t h_node, params_out)
cpdef intptr_t graph_add_mem_free_node(intptr_t h_graph, intptr_t dependencies, size_t num_dependencies, unsigned long long dptr) except? 0
cpdef unsigned long long graph_mem_free_node_get_params(intptr_t h_node) except? 0
cpdef device_graph_mem_trim(int device)
cpdef device_get_graph_mem_attribute(int device, int attr, intptr_t value)
cpdef device_set_graph_mem_attribute(int device, int attr, intptr_t value)
cpdef intptr_t graph_clone(intptr_t original_graph) except? 0
cpdef intptr_t graph_node_find_in_clone(intptr_t h_original_node, intptr_t h_cloned_graph) except? 0
cpdef int graph_node_get_type(intptr_t h_node) except? -1
cpdef object graph_get_nodes(intptr_t h_graph)
cpdef object graph_get_root_nodes(intptr_t h_graph)
cpdef tuple graph_get_edges_v2(intptr_t h_graph)
cpdef tuple graph_node_get_dependencies_v2(intptr_t h_node)
cpdef tuple graph_node_get_dependent_nodes_v2(intptr_t h_node)
cpdef graph_add_dependencies_v2(intptr_t h_graph, intptr_t from_, intptr_t to, edge_data, size_t num_dependencies)
cpdef graph_remove_dependencies_v2(intptr_t h_graph, intptr_t from_, intptr_t to, edge_data, size_t num_dependencies)
cpdef graph_destroy_node(intptr_t h_node)
cpdef intptr_t graph_instantiate_with_flags(intptr_t h_graph, unsigned long long flags) except? 0
cpdef intptr_t graph_instantiate_with_params(intptr_t h_graph, instantiate_params) except? 0
cpdef uint64_t graph_exec_get_flags(intptr_t h_graph_exec) except? 0
cpdef graph_exec_kernel_node_set_params_v2(intptr_t h_graph_exec, intptr_t h_node, node_params)
cpdef graph_exec_memcpy_node_set_params(intptr_t h_graph_exec, intptr_t h_node, copy_params, intptr_t ctx)
cpdef graph_exec_memset_node_set_params(intptr_t h_graph_exec, intptr_t h_node, memset_params, intptr_t ctx)
cpdef graph_exec_host_node_set_params(intptr_t h_graph_exec, intptr_t h_node, node_params)
cpdef graph_exec_child_graph_node_set_params(intptr_t h_graph_exec, intptr_t h_node, intptr_t child_graph)
cpdef graph_exec_event_record_node_set_event(intptr_t h_graph_exec, intptr_t h_node, intptr_t event)
cpdef graph_exec_event_wait_node_set_event(intptr_t h_graph_exec, intptr_t h_node, intptr_t event)
cpdef graph_exec_external_semaphores_signal_node_set_params(intptr_t h_graph_exec, intptr_t h_node, node_params)
cpdef graph_exec_external_semaphores_wait_node_set_params(intptr_t h_graph_exec, intptr_t h_node, node_params)
cpdef graph_node_set_enabled(intptr_t h_graph_exec, intptr_t h_node, unsigned int is_enabled)
cpdef unsigned int graph_node_get_enabled(intptr_t h_graph_exec, intptr_t h_node) except? 0
cpdef graph_upload(intptr_t h_graph_exec, intptr_t h_stream)
cpdef graph_launch(intptr_t h_graph_exec, intptr_t h_stream)
cpdef graph_exec_destroy(intptr_t h_graph_exec)
cpdef graph_destroy(intptr_t h_graph)
cpdef graph_exec_update_v2(intptr_t h_graph_exec, intptr_t h_graph, result_info)
cpdef graph_kernel_node_copy_attributes(intptr_t dst, intptr_t src)
cpdef graph_kernel_node_get_attribute(intptr_t h_node, int attr, intptr_t value_out)
cpdef graph_kernel_node_set_attribute(intptr_t h_node, int attr, intptr_t value)
cpdef graph_debug_dot_print(intptr_t h_graph, path, unsigned int flags)
cpdef intptr_t user_object_create(ptr, intptr_t destroy, unsigned int initial_refcount, unsigned int flags) except? 0
cpdef user_object_retain(intptr_t object, unsigned int count)
cpdef user_object_release(intptr_t object, unsigned int count)
cpdef graph_retain_user_object(intptr_t graph, intptr_t object, unsigned int count, unsigned int flags)
cpdef graph_release_user_object(intptr_t graph, intptr_t object, unsigned int count)
cpdef intptr_t graph_add_node_v2(intptr_t h_graph, intptr_t dependencies, dependency_data, size_t num_dependencies, node_params) except? 0
cpdef graph_node_set_params(intptr_t h_node, node_params)
cpdef graph_exec_node_set_params(intptr_t h_graph_exec, intptr_t h_node, node_params)
cpdef uint64_t graph_conditional_handle_create(intptr_t h_graph, intptr_t ctx, unsigned int default_launch_value, unsigned int flags) except? 0
cpdef int occupancy_max_active_blocks_per_multiprocessor(intptr_t func, int block_size, size_t dynamic_s_mem_size) except? -1
cpdef int occupancy_max_active_blocks_per_multiprocessor_with_flags(intptr_t func, int block_size, size_t dynamic_s_mem_size, unsigned int flags) except? -1
cpdef tuple occupancy_max_potential_block_size(intptr_t func, intptr_t block_size_to_dynamic_s_mem_size, size_t dynamic_s_mem_size, int block_size_limit)
cpdef tuple occupancy_max_potential_block_size_with_flags(intptr_t func, intptr_t block_size_to_dynamic_s_mem_size, size_t dynamic_s_mem_size, int block_size_limit, unsigned int flags)
cpdef size_t occupancy_available_dynamic_smem_per_block(intptr_t func, int num_blocks, int block_size) except? 0
cpdef int occupancy_max_potential_cluster_size(intptr_t func, config) except? -1
cpdef int occupancy_max_active_clusters(intptr_t func, config) except? -1
cpdef tex_ref_set_array(intptr_t h_tex_ref, intptr_t h_array, unsigned int flags)
cpdef tex_ref_set_mipmapped_array(intptr_t h_tex_ref, intptr_t h_mipmapped_array, unsigned int flags)
cpdef size_t tex_ref_set_address_v2(intptr_t h_tex_ref, unsigned long long dptr, size_t bytes) except? 0
cpdef tex_ref_set_address2d_v3(intptr_t h_tex_ref, desc, unsigned long long dptr, size_t pitch)
cpdef tex_ref_set_format(intptr_t h_tex_ref, int fmt, int num_packed_components)
cpdef tex_ref_set_address_mode(intptr_t h_tex_ref, int dim, int am)
cpdef tex_ref_set_filter_mode(intptr_t h_tex_ref, int fm)
cpdef tex_ref_set_mipmap_filter_mode(intptr_t h_tex_ref, int fm)
cpdef tex_ref_set_mipmap_level_bias(intptr_t h_tex_ref, float bias)
cpdef tex_ref_set_mipmap_level_clamp(intptr_t h_tex_ref, float min_mipmap_level_clamp, float max_mipmap_level_clamp)
cpdef tex_ref_set_max_anisotropy(intptr_t h_tex_ref, unsigned int max_aniso)
cpdef tex_ref_set_border_color(intptr_t h_tex_ref, intptr_t p_border_color)
cpdef tex_ref_set_flags(intptr_t h_tex_ref, unsigned int flags)
cpdef unsigned long long tex_ref_get_address_v2(intptr_t h_tex_ref) except? 0
cpdef intptr_t tex_ref_get_array(intptr_t h_tex_ref) except? 0
cpdef intptr_t tex_ref_get_mipmapped_array(intptr_t h_tex_ref) except? 0
cpdef int tex_ref_get_address_mode(intptr_t h_tex_ref, int dim) except? -1
cpdef int tex_ref_get_filter_mode(intptr_t h_tex_ref) except? -1
cpdef tuple tex_ref_get_format(intptr_t h_tex_ref)
cpdef int tex_ref_get_mipmap_filter_mode(intptr_t h_tex_ref) except? -1
cpdef float tex_ref_get_mipmap_level_bias(intptr_t h_tex_ref) except? -1.0
cpdef tuple tex_ref_get_mipmap_level_clamp(intptr_t h_tex_ref)
cpdef int tex_ref_get_max_anisotropy(intptr_t h_tex_ref) except? -1
cpdef tex_ref_get_border_color(intptr_t p_border_color, intptr_t h_tex_ref)
cpdef unsigned int tex_ref_get_flags(intptr_t h_tex_ref) except? 0
cpdef intptr_t tex_ref_create() except? 0
cpdef tex_ref_destroy(intptr_t h_tex_ref)
cpdef surf_ref_set_array(intptr_t h_surf_ref, intptr_t h_array, unsigned int flags)
cpdef intptr_t surf_ref_get_array(intptr_t h_surf_ref) except? 0
cpdef unsigned long long tex_object_create(intptr_t p_res_desc, p_tex_desc, p_res_view_desc) except? 0
cpdef tex_object_destroy(unsigned long long tex_object)
cpdef tex_object_get_resource_desc(intptr_t p_res_desc, unsigned long long tex_object)
cpdef tex_object_get_texture_desc(p_tex_desc, unsigned long long tex_object)
cpdef object tex_object_get_resource_view_desc(unsigned long long tex_object)
cpdef unsigned long long surf_object_create(intptr_t p_res_desc) except? 0
cpdef surf_object_destroy(unsigned long long surf_object)
cpdef surf_object_get_resource_desc(intptr_t p_res_desc, unsigned long long surf_object)
cpdef tensor_map_encode_tiled(tensor_map, int tensor_data_type, uint64_t tensor_rank, intptr_t global_address, intptr_t global_dim, intptr_t global_strides, intptr_t box_dim, intptr_t element_strides, int interleave, int swizzle, int l2promotion, int oob_fill)
cpdef tensor_map_encode_im2col(tensor_map, int tensor_data_type, uint64_t tensor_rank, intptr_t global_address, intptr_t global_dim, intptr_t global_strides, intptr_t pixel_box_lower_corner, intptr_t pixel_box_upper_corner, uint64_t channels_per_pixel, uint64_t pixels_per_column, intptr_t element_strides, int interleave, int swizzle, int l2promotion, int oob_fill)
cpdef tensor_map_encode_im2col_wide(tensor_map, int tensor_data_type, uint64_t tensor_rank, intptr_t global_address, intptr_t global_dim, intptr_t global_strides, int pixel_box_lower_corner_width, int pixel_box_upper_corner_width, uint64_t channels_per_pixel, uint64_t pixels_per_column, intptr_t element_strides, int interleave, int mode, int swizzle, int l2promotion, int oob_fill)
cpdef tensor_map_replace_address(tensor_map, intptr_t global_address)
cpdef int device_can_access_peer(int dev, int peer_dev) except? -1
cpdef ctx_enable_peer_access(intptr_t peer_context, unsigned int flags)
cpdef ctx_disable_peer_access(intptr_t peer_context)
cpdef int device_get_p2p_attribute(int attrib, int src_device, int dst_device) except? -1
cpdef graphics_unregister_resource(intptr_t resource)
cpdef intptr_t graphics_sub_resource_get_mapped_array(intptr_t resource, unsigned int array_index, unsigned int mip_level) except? 0
cpdef intptr_t graphics_resource_get_mapped_mipmapped_array(intptr_t resource) except? 0
cpdef tuple graphics_resource_get_mapped_pointer_v2(intptr_t resource)
cpdef graphics_resource_set_map_flags_v2(intptr_t resource, unsigned int flags)
cpdef graphics_map_resources(unsigned int count, intptr_t resources, intptr_t h_stream)
cpdef graphics_unmap_resources(unsigned int count, intptr_t resources, intptr_t h_stream)
cpdef get_proc_address_v2(symbol, intptr_t pfn, int cuda_version, uint64_t flags, intptr_t symbol_status)
cpdef coredump_get_attribute(int attrib, intptr_t value, intptr_t size)
cpdef coredump_get_attribute_global(int attrib, intptr_t value, intptr_t size)
cpdef coredump_set_attribute(int attrib, intptr_t value, intptr_t size)
cpdef coredump_set_attribute_global(int attrib, intptr_t value, intptr_t size)
cpdef intptr_t get_export_table(p_export_table_id) except? 0
cpdef intptr_t green_ctx_create(intptr_t desc, int dev, unsigned int flags) except? 0
cpdef green_ctx_destroy(intptr_t h_ctx)
cpdef intptr_t ctx_from_green_ctx(intptr_t h_ctx) except? 0
cpdef device_get_dev_resource(int device, resource, int type)
cpdef ctx_get_dev_resource(intptr_t h_ctx, resource, int type)
cpdef green_ctx_get_dev_resource(intptr_t h_ctx, resource, int type)
cpdef dev_sm_resource_split_by_count(result, intptr_t nb_groups, input, remainder, unsigned int flags, unsigned int min_count)
cpdef intptr_t dev_resource_generate_desc(resources, unsigned int nb_resources) except? 0
cpdef green_ctx_record_event(intptr_t h_ctx, intptr_t h_event)
cpdef green_ctx_wait_event(intptr_t h_ctx, intptr_t h_event)
cpdef intptr_t stream_get_green_ctx(intptr_t h_stream) except? 0
cpdef intptr_t green_ctx_stream_create(intptr_t green_ctx, unsigned int flags, int priority) except? 0
cpdef intptr_t logs_register_callback(intptr_t callback_func, intptr_t user_data) except? 0
cpdef logs_unregister_callback(intptr_t callback)
cpdef unsigned int logs_current(unsigned int flags) except? 0
cpdef logs_dump_to_file(intptr_t iterator, path_to_file, unsigned int flags)
cpdef logs_dump_to_memory(intptr_t iterator, intptr_t buffer, intptr_t size, unsigned int flags)
cpdef int checkpoint_process_get_restore_thread_id(int pid) except? -1
cpdef int checkpoint_process_get_state(int pid) except? -1
cpdef checkpoint_process_lock(int pid, args)
cpdef checkpoint_process_checkpoint(int pid, args)
cpdef checkpoint_process_restore(int pid, intptr_t args)
cpdef checkpoint_process_unlock(int pid, args)
cpdef graphics_egl_register_image(intptr_t p_cuda_resource, intptr_t image, unsigned int flags)
cpdef intptr_t egl_stream_consumer_connect(intptr_t stream) except? 0
cpdef intptr_t egl_stream_consumer_connect_with_flags(intptr_t stream, unsigned int flags) except? 0
cpdef egl_stream_consumer_disconnect(intptr_t conn)
cpdef intptr_t egl_stream_consumer_acquire_frame(intptr_t conn, intptr_t p_stream, unsigned int timeout) except? 0
cpdef egl_stream_consumer_release_frame(intptr_t conn, intptr_t p_cuda_resource, intptr_t p_stream)
cpdef intptr_t egl_stream_producer_connect(intptr_t stream, unsigned int width, unsigned int height) except? 0
cpdef egl_stream_producer_disconnect(intptr_t conn)
cpdef intptr_t event_create_from_egl_sync(intptr_t egl_sync, unsigned int flags) except? 0
cpdef intptr_t graphics_gl_register_buffer(GLuint buffer, unsigned int flags) except? 0
cpdef intptr_t graphics_gl_register_image(GLuint image, GLenum target, unsigned int flags) except? 0
cpdef profiler_start()
cpdef profiler_stop()
cpdef int vdpau_get_device(VdpDevice vdp_device, intptr_t vdp_get_proc_address) except? -1
cpdef intptr_t vdpau_ctx_create_v2(unsigned int flags, int device, VdpDevice vdp_device, intptr_t vdp_get_proc_address) except? 0
cpdef intptr_t graphics_vdpau_register_video_surface(VdpVideoSurface vdp_surface, unsigned int flags) except? 0
cpdef intptr_t graphics_vdpau_register_output_surface(VdpOutputSurface vdp_surface, unsigned int flags) except? 0
cpdef int ctx_get_device_v2(intptr_t ctx) except? -1
cpdef ctx_synchronize_v2(intptr_t ctx)
cpdef memcpy_batch_async_v2(intptr_t dsts, intptr_t srcs, intptr_t sizes, size_t count, attrs, intptr_t attrs_idxs, size_t num_attrs, intptr_t h_stream)
cpdef memcpy_3d_batch_async_v2(size_t num_ops, intptr_t op_list, unsigned long long flags, intptr_t h_stream)
cpdef intptr_t mem_get_default_mem_pool(location, int type) except? 0
cpdef intptr_t mem_get_mem_pool(location, int type) except? 0
cpdef mem_set_mem_pool(location, int type, intptr_t pool)
cpdef mem_prefetch_batch_async(intptr_t dptrs, intptr_t sizes, size_t count, prefetch_locs, intptr_t prefetch_loc_idxs, size_t num_prefetch_locs, unsigned long long flags, intptr_t h_stream)
cpdef mem_discard_batch_async(intptr_t dptrs, intptr_t sizes, size_t count, unsigned long long flags, intptr_t h_stream)
cpdef mem_discard_and_prefetch_batch_async(intptr_t dptrs, intptr_t sizes, size_t count, prefetch_locs, intptr_t prefetch_loc_idxs, size_t num_prefetch_locs, unsigned long long flags, intptr_t h_stream)
cpdef unsigned int device_get_p2p_atomic_capabilities(intptr_t operations, unsigned int count, int src_device, int dst_device) except? 0
cpdef unsigned long long green_ctx_get_id(intptr_t green_ctx) except? 0
cpdef multicast_bind_mem_v2(unsigned long long mc_handle, int dev, size_t mc_offset, unsigned long long mem_handle, size_t mem_offset, size_t size, unsigned long long flags)
cpdef multicast_bind_addr_v2(unsigned long long mc_handle, int dev, size_t mc_offset, unsigned long long memptr, size_t size, unsigned long long flags)
cpdef intptr_t graph_node_get_containing_graph(intptr_t h_node) except? 0
cpdef unsigned int graph_node_get_local_id(intptr_t h_node) except? 0
cpdef unsigned long long graph_node_get_tools_id(intptr_t h_node) except? 0
cpdef unsigned int graph_get_id(intptr_t h_graph) except? 0
cpdef unsigned int graph_exec_get_id(intptr_t h_graph_exec) except? 0
cpdef dev_sm_resource_split(result, unsigned int nb_groups, input, remainder, unsigned int flags, group_params)
cpdef stream_get_dev_resource(intptr_t h_stream, resource, int type)
cpdef size_t kernel_get_param_count(intptr_t kernel) except? 0
cpdef memcpy_with_attributes_async(unsigned long long dst, unsigned long long src, size_t size, attr, intptr_t h_stream)
cpdef memcpy_3d_with_attributes_async(intptr_t op, unsigned long long flags, intptr_t h_stream)
cpdef stream_begin_capture_to_cig(intptr_t h_stream, intptr_t stream_cig_capture_params)
cpdef stream_end_capture_to_cig(intptr_t h_stream)
cpdef size_t func_get_param_count(intptr_t func) except? 0
cpdef launch_host_func_v2(intptr_t h_stream, intptr_t fn, intptr_t user_data, unsigned int sync_mode)
cpdef graph_node_get_params(intptr_t h_node, node_params)
cpdef intptr_t coredump_register_start_callback(intptr_t callback, intptr_t user_data) except? 0
cpdef intptr_t coredump_register_complete_callback(intptr_t callback, intptr_t user_data) except? 0
cpdef coredump_deregister_start_callback(intptr_t callback)
cpdef coredump_deregister_complete_callback(intptr_t callback)
cpdef uint32_t logical_endpoint_id_reserve(uint64_t count) except? 0
cpdef logical_endpoint_id_release(uint32_t base_le_id, uint64_t count)
cpdef logical_endpoint_create(uint32_t le_id, intptr_t prop)
cpdef logical_endpoint_add_device(uint32_t le_id, int dev)
cpdef logical_endpoint_destroy(uint32_t le_id)
cpdef logical_endpoint_bind_addr(uint32_t le_id, int dev, uint64_t offset, intptr_t ptr, uint64_t size, unsigned long long flags)
cpdef logical_endpoint_bind_mem(uint32_t le_id, int dev, uint64_t offset, unsigned long long mem_handle, uint64_t mem_offset, uint64_t size, unsigned long long flags)
cpdef logical_endpoint_unbind(uint32_t le_id, int dev, uint64_t offset, uint64_t size)
cpdef logical_endpoint_export(intptr_t handle, uint32_t le_id, int handle_type)
cpdef logical_endpoint_import(uint32_t le_id, handle, int handle_type)
cpdef tuple logical_endpoint_get_limits(intptr_t prop)
cpdef logical_endpoint_query(uint32_t le_id, uint64_t count, intptr_t query_status)
cpdef stream_begin_recapture_to_graph(intptr_t h_stream, int mode, intptr_t h_graph, intptr_t callback_func, intptr_t user_data)
