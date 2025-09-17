.. SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

------
driver
------

Data types used by CUDA driver
------------------------------



.. autoclass:: cuda.bindings.driver.CUuuid_st
.. autoclass:: cuda.bindings.driver.CUmemFabricHandle_st
.. autoclass:: cuda.bindings.driver.CUipcEventHandle_st
.. autoclass:: cuda.bindings.driver.CUipcMemHandle_st
.. autoclass:: cuda.bindings.driver.CUstreamBatchMemOpParams_union
.. autoclass:: cuda.bindings.driver.CUDA_BATCH_MEM_OP_NODE_PARAMS_v1_st
.. autoclass:: cuda.bindings.driver.CUDA_BATCH_MEM_OP_NODE_PARAMS_v2_st
.. autoclass:: cuda.bindings.driver.CUasyncNotificationInfo_st
.. autoclass:: cuda.bindings.driver.CUdevprop_st
.. autoclass:: cuda.bindings.driver.CUaccessPolicyWindow_st
.. autoclass:: cuda.bindings.driver.CUDA_KERNEL_NODE_PARAMS_st
.. autoclass:: cuda.bindings.driver.CUDA_KERNEL_NODE_PARAMS_v2_st
.. autoclass:: cuda.bindings.driver.CUDA_KERNEL_NODE_PARAMS_v3_st
.. autoclass:: cuda.bindings.driver.CUDA_MEMSET_NODE_PARAMS_st
.. autoclass:: cuda.bindings.driver.CUDA_MEMSET_NODE_PARAMS_v2_st
.. autoclass:: cuda.bindings.driver.CUDA_HOST_NODE_PARAMS_st
.. autoclass:: cuda.bindings.driver.CUDA_HOST_NODE_PARAMS_v2_st
.. autoclass:: cuda.bindings.driver.CUDA_CONDITIONAL_NODE_PARAMS
.. autoclass:: cuda.bindings.driver.CUgraphEdgeData_st
.. autoclass:: cuda.bindings.driver.CUDA_GRAPH_INSTANTIATE_PARAMS_st
.. autoclass:: cuda.bindings.driver.CUlaunchMemSyncDomainMap_st
.. autoclass:: cuda.bindings.driver.CUlaunchAttributeValue_union
.. autoclass:: cuda.bindings.driver.CUlaunchAttribute_st
.. autoclass:: cuda.bindings.driver.CUlaunchConfig_st
.. autoclass:: cuda.bindings.driver.CUexecAffinitySmCount_st
.. autoclass:: cuda.bindings.driver.CUexecAffinityParam_st
.. autoclass:: cuda.bindings.driver.CUctxCigParam_st
.. autoclass:: cuda.bindings.driver.CUctxCreateParams_st
.. autoclass:: cuda.bindings.driver.CUlibraryHostUniversalFunctionAndDataTable_st
.. autoclass:: cuda.bindings.driver.CUDA_MEMCPY2D_st
.. autoclass:: cuda.bindings.driver.CUDA_MEMCPY3D_st
.. autoclass:: cuda.bindings.driver.CUDA_MEMCPY3D_PEER_st
.. autoclass:: cuda.bindings.driver.CUDA_MEMCPY_NODE_PARAMS_st
.. autoclass:: cuda.bindings.driver.CUDA_ARRAY_DESCRIPTOR_st
.. autoclass:: cuda.bindings.driver.CUDA_ARRAY3D_DESCRIPTOR_st
.. autoclass:: cuda.bindings.driver.CUDA_ARRAY_SPARSE_PROPERTIES_st
.. autoclass:: cuda.bindings.driver.CUDA_ARRAY_MEMORY_REQUIREMENTS_st
.. autoclass:: cuda.bindings.driver.CUDA_RESOURCE_DESC_st
.. autoclass:: cuda.bindings.driver.CUDA_TEXTURE_DESC_st
.. autoclass:: cuda.bindings.driver.CUDA_RESOURCE_VIEW_DESC_st
.. autoclass:: cuda.bindings.driver.CUtensorMap_st
.. autoclass:: cuda.bindings.driver.CUDA_POINTER_ATTRIBUTE_P2P_TOKENS_st
.. autoclass:: cuda.bindings.driver.CUDA_LAUNCH_PARAMS_st
.. autoclass:: cuda.bindings.driver.CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st
.. autoclass:: cuda.bindings.driver.CUDA_EXTERNAL_MEMORY_BUFFER_DESC_st
.. autoclass:: cuda.bindings.driver.CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC_st
.. autoclass:: cuda.bindings.driver.CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_st
.. autoclass:: cuda.bindings.driver.CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st
.. autoclass:: cuda.bindings.driver.CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st
.. autoclass:: cuda.bindings.driver.CUDA_EXT_SEM_SIGNAL_NODE_PARAMS_st
.. autoclass:: cuda.bindings.driver.CUDA_EXT_SEM_SIGNAL_NODE_PARAMS_v2_st
.. autoclass:: cuda.bindings.driver.CUDA_EXT_SEM_WAIT_NODE_PARAMS_st
.. autoclass:: cuda.bindings.driver.CUDA_EXT_SEM_WAIT_NODE_PARAMS_v2_st
.. autoclass:: cuda.bindings.driver.CUarrayMapInfo_st
.. autoclass:: cuda.bindings.driver.CUmemLocation_st
.. autoclass:: cuda.bindings.driver.CUmemAllocationProp_st
.. autoclass:: cuda.bindings.driver.CUmulticastObjectProp_st
.. autoclass:: cuda.bindings.driver.CUmemAccessDesc_st
.. autoclass:: cuda.bindings.driver.CUgraphExecUpdateResultInfo_st
.. autoclass:: cuda.bindings.driver.CUmemPoolProps_st
.. autoclass:: cuda.bindings.driver.CUmemPoolPtrExportData_st
.. autoclass:: cuda.bindings.driver.CUmemcpyAttributes_st
.. autoclass:: cuda.bindings.driver.CUoffset3D_st
.. autoclass:: cuda.bindings.driver.CUextent3D_st
.. autoclass:: cuda.bindings.driver.CUmemcpy3DOperand_st
.. autoclass:: cuda.bindings.driver.CUDA_MEMCPY3D_BATCH_OP_st
.. autoclass:: cuda.bindings.driver.CUDA_MEM_ALLOC_NODE_PARAMS_v1_st
.. autoclass:: cuda.bindings.driver.CUDA_MEM_ALLOC_NODE_PARAMS_v2_st
.. autoclass:: cuda.bindings.driver.CUDA_MEM_FREE_NODE_PARAMS_st
.. autoclass:: cuda.bindings.driver.CUDA_CHILD_GRAPH_NODE_PARAMS_st
.. autoclass:: cuda.bindings.driver.CUDA_EVENT_RECORD_NODE_PARAMS_st
.. autoclass:: cuda.bindings.driver.CUDA_EVENT_WAIT_NODE_PARAMS_st
.. autoclass:: cuda.bindings.driver.CUgraphNodeParams_st
.. autoclass:: cuda.bindings.driver.CUcheckpointLockArgs_st
.. autoclass:: cuda.bindings.driver.CUcheckpointCheckpointArgs_st
.. autoclass:: cuda.bindings.driver.CUcheckpointGpuPair_st
.. autoclass:: cuda.bindings.driver.CUcheckpointRestoreArgs_st
.. autoclass:: cuda.bindings.driver.CUcheckpointUnlockArgs_st
.. autoclass:: cuda.bindings.driver.CUeglFrame_st
.. autoclass:: cuda.bindings.driver.CUipcMem_flags

    .. autoattribute:: cuda.bindings.driver.CUipcMem_flags.CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS


        Automatically enable peer access between remote devices as needed

.. autoclass:: cuda.bindings.driver.CUmemAttach_flags

    .. autoattribute:: cuda.bindings.driver.CUmemAttach_flags.CU_MEM_ATTACH_GLOBAL


        Memory can be accessed by any stream on any device


    .. autoattribute:: cuda.bindings.driver.CUmemAttach_flags.CU_MEM_ATTACH_HOST


        Memory cannot be accessed by any stream on any device


    .. autoattribute:: cuda.bindings.driver.CUmemAttach_flags.CU_MEM_ATTACH_SINGLE


        Memory can only be accessed by a single stream on the associated device

.. autoclass:: cuda.bindings.driver.CUctx_flags

    .. autoattribute:: cuda.bindings.driver.CUctx_flags.CU_CTX_SCHED_AUTO


        Automatic scheduling


    .. autoattribute:: cuda.bindings.driver.CUctx_flags.CU_CTX_SCHED_SPIN


        Set spin as default scheduling


    .. autoattribute:: cuda.bindings.driver.CUctx_flags.CU_CTX_SCHED_YIELD


        Set yield as default scheduling


    .. autoattribute:: cuda.bindings.driver.CUctx_flags.CU_CTX_SCHED_BLOCKING_SYNC


        Set blocking synchronization as default scheduling


    .. autoattribute:: cuda.bindings.driver.CUctx_flags.CU_CTX_BLOCKING_SYNC


        Set blocking synchronization as default scheduling [Deprecated]


    .. autoattribute:: cuda.bindings.driver.CUctx_flags.CU_CTX_SCHED_MASK


    .. autoattribute:: cuda.bindings.driver.CUctx_flags.CU_CTX_MAP_HOST


        [Deprecated]


    .. autoattribute:: cuda.bindings.driver.CUctx_flags.CU_CTX_LMEM_RESIZE_TO_MAX


        Keep local memory allocation after launch


    .. autoattribute:: cuda.bindings.driver.CUctx_flags.CU_CTX_COREDUMP_ENABLE


        Trigger coredumps from exceptions in this context


    .. autoattribute:: cuda.bindings.driver.CUctx_flags.CU_CTX_USER_COREDUMP_ENABLE


        Enable user pipe to trigger coredumps in this context


    .. autoattribute:: cuda.bindings.driver.CUctx_flags.CU_CTX_SYNC_MEMOPS


        Ensure synchronous memory operations on this context will synchronize


    .. autoattribute:: cuda.bindings.driver.CUctx_flags.CU_CTX_FLAGS_MASK

.. autoclass:: cuda.bindings.driver.CUevent_sched_flags

    .. autoattribute:: cuda.bindings.driver.CUevent_sched_flags.CU_EVENT_SCHED_AUTO


        Automatic scheduling


    .. autoattribute:: cuda.bindings.driver.CUevent_sched_flags.CU_EVENT_SCHED_SPIN


        Set spin as default scheduling


    .. autoattribute:: cuda.bindings.driver.CUevent_sched_flags.CU_EVENT_SCHED_YIELD


        Set yield as default scheduling


    .. autoattribute:: cuda.bindings.driver.CUevent_sched_flags.CU_EVENT_SCHED_BLOCKING_SYNC


        Set blocking synchronization as default scheduling

.. autoclass:: cuda.bindings.driver.cl_event_flags

    .. autoattribute:: cuda.bindings.driver.cl_event_flags.NVCL_EVENT_SCHED_AUTO


        Automatic scheduling


    .. autoattribute:: cuda.bindings.driver.cl_event_flags.NVCL_EVENT_SCHED_SPIN


        Set spin as default scheduling


    .. autoattribute:: cuda.bindings.driver.cl_event_flags.NVCL_EVENT_SCHED_YIELD


        Set yield as default scheduling


    .. autoattribute:: cuda.bindings.driver.cl_event_flags.NVCL_EVENT_SCHED_BLOCKING_SYNC


        Set blocking synchronization as default scheduling

.. autoclass:: cuda.bindings.driver.cl_context_flags

    .. autoattribute:: cuda.bindings.driver.cl_context_flags.NVCL_CTX_SCHED_AUTO


        Automatic scheduling


    .. autoattribute:: cuda.bindings.driver.cl_context_flags.NVCL_CTX_SCHED_SPIN


        Set spin as default scheduling


    .. autoattribute:: cuda.bindings.driver.cl_context_flags.NVCL_CTX_SCHED_YIELD


        Set yield as default scheduling


    .. autoattribute:: cuda.bindings.driver.cl_context_flags.NVCL_CTX_SCHED_BLOCKING_SYNC


        Set blocking synchronization as default scheduling

.. autoclass:: cuda.bindings.driver.CUstream_flags

    .. autoattribute:: cuda.bindings.driver.CUstream_flags.CU_STREAM_DEFAULT


        Default stream flag


    .. autoattribute:: cuda.bindings.driver.CUstream_flags.CU_STREAM_NON_BLOCKING


        Stream does not synchronize with stream 0 (the NULL stream)

.. autoclass:: cuda.bindings.driver.CUevent_flags

    .. autoattribute:: cuda.bindings.driver.CUevent_flags.CU_EVENT_DEFAULT


        Default event flag


    .. autoattribute:: cuda.bindings.driver.CUevent_flags.CU_EVENT_BLOCKING_SYNC


        Event uses blocking synchronization


    .. autoattribute:: cuda.bindings.driver.CUevent_flags.CU_EVENT_DISABLE_TIMING


        Event will not record timing data


    .. autoattribute:: cuda.bindings.driver.CUevent_flags.CU_EVENT_INTERPROCESS


        Event is suitable for interprocess use. CU_EVENT_DISABLE_TIMING must be set

.. autoclass:: cuda.bindings.driver.CUevent_record_flags

    .. autoattribute:: cuda.bindings.driver.CUevent_record_flags.CU_EVENT_RECORD_DEFAULT


        Default event record flag


    .. autoattribute:: cuda.bindings.driver.CUevent_record_flags.CU_EVENT_RECORD_EXTERNAL


        When using stream capture, create an event record node instead of the default behavior. This flag is invalid when used outside of capture.

.. autoclass:: cuda.bindings.driver.CUevent_wait_flags

    .. autoattribute:: cuda.bindings.driver.CUevent_wait_flags.CU_EVENT_WAIT_DEFAULT


        Default event wait flag


    .. autoattribute:: cuda.bindings.driver.CUevent_wait_flags.CU_EVENT_WAIT_EXTERNAL


        When using stream capture, create an event wait node instead of the default behavior. This flag is invalid when used outside of capture.

.. autoclass:: cuda.bindings.driver.CUstreamWaitValue_flags

    .. autoattribute:: cuda.bindings.driver.CUstreamWaitValue_flags.CU_STREAM_WAIT_VALUE_GEQ


        Wait until (int32_t)(*addr - value) >= 0 (or int64_t for 64 bit values). Note this is a cyclic comparison which ignores wraparound. (Default behavior.)


    .. autoattribute:: cuda.bindings.driver.CUstreamWaitValue_flags.CU_STREAM_WAIT_VALUE_EQ


        Wait until *addr == value.


    .. autoattribute:: cuda.bindings.driver.CUstreamWaitValue_flags.CU_STREAM_WAIT_VALUE_AND


        Wait until (*addr & value) != 0.


    .. autoattribute:: cuda.bindings.driver.CUstreamWaitValue_flags.CU_STREAM_WAIT_VALUE_NOR


        Wait until ~(*addr | value) != 0. Support for this operation can be queried with :py:obj:`~.cuDeviceGetAttribute()` and :py:obj:`~.CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR`.


    .. autoattribute:: cuda.bindings.driver.CUstreamWaitValue_flags.CU_STREAM_WAIT_VALUE_FLUSH


        Follow the wait operation with a flush of outstanding remote writes. This means that, if a remote write operation is guaranteed to have reached the device before the wait can be satisfied, that write is guaranteed to be visible to downstream device work. The device is permitted to reorder remote writes internally. For example, this flag would be required if two remote writes arrive in a defined order, the wait is satisfied by the second write, and downstream work needs to observe the first write. Support for this operation is restricted to selected platforms and can be queried with :py:obj:`~.CU_DEVICE_ATTRIBUTE_CAN_FLUSH_REMOTE_WRITES`.

.. autoclass:: cuda.bindings.driver.CUstreamWriteValue_flags

    .. autoattribute:: cuda.bindings.driver.CUstreamWriteValue_flags.CU_STREAM_WRITE_VALUE_DEFAULT


        Default behavior


    .. autoattribute:: cuda.bindings.driver.CUstreamWriteValue_flags.CU_STREAM_WRITE_VALUE_NO_MEMORY_BARRIER


        Permits the write to be reordered with writes which were issued before it, as a performance optimization. Normally, :py:obj:`~.cuStreamWriteValue32` will provide a memory fence before the write, which has similar semantics to __threadfence_system() but is scoped to the stream rather than a CUDA thread. This flag is not supported in the v2 API.

.. autoclass:: cuda.bindings.driver.CUstreamBatchMemOpType

    .. autoattribute:: cuda.bindings.driver.CUstreamBatchMemOpType.CU_STREAM_MEM_OP_WAIT_VALUE_32


        Represents a :py:obj:`~.cuStreamWaitValue32` operation


    .. autoattribute:: cuda.bindings.driver.CUstreamBatchMemOpType.CU_STREAM_MEM_OP_WRITE_VALUE_32


        Represents a :py:obj:`~.cuStreamWriteValue32` operation


    .. autoattribute:: cuda.bindings.driver.CUstreamBatchMemOpType.CU_STREAM_MEM_OP_WAIT_VALUE_64


        Represents a :py:obj:`~.cuStreamWaitValue64` operation


    .. autoattribute:: cuda.bindings.driver.CUstreamBatchMemOpType.CU_STREAM_MEM_OP_WRITE_VALUE_64


        Represents a :py:obj:`~.cuStreamWriteValue64` operation


    .. autoattribute:: cuda.bindings.driver.CUstreamBatchMemOpType.CU_STREAM_MEM_OP_BARRIER


        Insert a memory barrier of the specified type


    .. autoattribute:: cuda.bindings.driver.CUstreamBatchMemOpType.CU_STREAM_MEM_OP_FLUSH_REMOTE_WRITES


        This has the same effect as :py:obj:`~.CU_STREAM_WAIT_VALUE_FLUSH`, but as a standalone operation.

.. autoclass:: cuda.bindings.driver.CUstreamMemoryBarrier_flags

    .. autoattribute:: cuda.bindings.driver.CUstreamMemoryBarrier_flags.CU_STREAM_MEMORY_BARRIER_TYPE_SYS


        System-wide memory barrier.


    .. autoattribute:: cuda.bindings.driver.CUstreamMemoryBarrier_flags.CU_STREAM_MEMORY_BARRIER_TYPE_GPU


        Limit memory barrier scope to the GPU.

.. autoclass:: cuda.bindings.driver.CUoccupancy_flags

    .. autoattribute:: cuda.bindings.driver.CUoccupancy_flags.CU_OCCUPANCY_DEFAULT


        Default behavior


    .. autoattribute:: cuda.bindings.driver.CUoccupancy_flags.CU_OCCUPANCY_DISABLE_CACHING_OVERRIDE


        Assume global caching is enabled and cannot be automatically turned off

.. autoclass:: cuda.bindings.driver.CUstreamUpdateCaptureDependencies_flags

    .. autoattribute:: cuda.bindings.driver.CUstreamUpdateCaptureDependencies_flags.CU_STREAM_ADD_CAPTURE_DEPENDENCIES


        Add new nodes to the dependency set


    .. autoattribute:: cuda.bindings.driver.CUstreamUpdateCaptureDependencies_flags.CU_STREAM_SET_CAPTURE_DEPENDENCIES


        Replace the dependency set with the new nodes

.. autoclass:: cuda.bindings.driver.CUasyncNotificationType

    .. autoattribute:: cuda.bindings.driver.CUasyncNotificationType.CU_ASYNC_NOTIFICATION_TYPE_OVER_BUDGET


        Sent when the process has exceeded its device memory budget

.. autoclass:: cuda.bindings.driver.CUarray_format

    .. autoattribute:: cuda.bindings.driver.CUarray_format.CU_AD_FORMAT_UNSIGNED_INT8


        Unsigned 8-bit integers


    .. autoattribute:: cuda.bindings.driver.CUarray_format.CU_AD_FORMAT_UNSIGNED_INT16


        Unsigned 16-bit integers


    .. autoattribute:: cuda.bindings.driver.CUarray_format.CU_AD_FORMAT_UNSIGNED_INT32


        Unsigned 32-bit integers


    .. autoattribute:: cuda.bindings.driver.CUarray_format.CU_AD_FORMAT_SIGNED_INT8


        Signed 8-bit integers


    .. autoattribute:: cuda.bindings.driver.CUarray_format.CU_AD_FORMAT_SIGNED_INT16


        Signed 16-bit integers


    .. autoattribute:: cuda.bindings.driver.CUarray_format.CU_AD_FORMAT_SIGNED_INT32


        Signed 32-bit integers


    .. autoattribute:: cuda.bindings.driver.CUarray_format.CU_AD_FORMAT_HALF


        16-bit floating point


    .. autoattribute:: cuda.bindings.driver.CUarray_format.CU_AD_FORMAT_FLOAT


        32-bit floating point


    .. autoattribute:: cuda.bindings.driver.CUarray_format.CU_AD_FORMAT_NV12


        8-bit YUV planar format, with 4:2:0 sampling


    .. autoattribute:: cuda.bindings.driver.CUarray_format.CU_AD_FORMAT_UNORM_INT8X1


        1 channel unsigned 8-bit normalized integer


    .. autoattribute:: cuda.bindings.driver.CUarray_format.CU_AD_FORMAT_UNORM_INT8X2


        2 channel unsigned 8-bit normalized integer


    .. autoattribute:: cuda.bindings.driver.CUarray_format.CU_AD_FORMAT_UNORM_INT8X4


        4 channel unsigned 8-bit normalized integer


    .. autoattribute:: cuda.bindings.driver.CUarray_format.CU_AD_FORMAT_UNORM_INT16X1


        1 channel unsigned 16-bit normalized integer


    .. autoattribute:: cuda.bindings.driver.CUarray_format.CU_AD_FORMAT_UNORM_INT16X2


        2 channel unsigned 16-bit normalized integer


    .. autoattribute:: cuda.bindings.driver.CUarray_format.CU_AD_FORMAT_UNORM_INT16X4


        4 channel unsigned 16-bit normalized integer


    .. autoattribute:: cuda.bindings.driver.CUarray_format.CU_AD_FORMAT_SNORM_INT8X1


        1 channel signed 8-bit normalized integer


    .. autoattribute:: cuda.bindings.driver.CUarray_format.CU_AD_FORMAT_SNORM_INT8X2


        2 channel signed 8-bit normalized integer


    .. autoattribute:: cuda.bindings.driver.CUarray_format.CU_AD_FORMAT_SNORM_INT8X4


        4 channel signed 8-bit normalized integer


    .. autoattribute:: cuda.bindings.driver.CUarray_format.CU_AD_FORMAT_SNORM_INT16X1


        1 channel signed 16-bit normalized integer


    .. autoattribute:: cuda.bindings.driver.CUarray_format.CU_AD_FORMAT_SNORM_INT16X2


        2 channel signed 16-bit normalized integer


    .. autoattribute:: cuda.bindings.driver.CUarray_format.CU_AD_FORMAT_SNORM_INT16X4


        4 channel signed 16-bit normalized integer


    .. autoattribute:: cuda.bindings.driver.CUarray_format.CU_AD_FORMAT_BC1_UNORM


        4 channel unsigned normalized block-compressed (BC1 compression) format


    .. autoattribute:: cuda.bindings.driver.CUarray_format.CU_AD_FORMAT_BC1_UNORM_SRGB


        4 channel unsigned normalized block-compressed (BC1 compression) format with sRGB encoding


    .. autoattribute:: cuda.bindings.driver.CUarray_format.CU_AD_FORMAT_BC2_UNORM


        4 channel unsigned normalized block-compressed (BC2 compression) format


    .. autoattribute:: cuda.bindings.driver.CUarray_format.CU_AD_FORMAT_BC2_UNORM_SRGB


        4 channel unsigned normalized block-compressed (BC2 compression) format with sRGB encoding


    .. autoattribute:: cuda.bindings.driver.CUarray_format.CU_AD_FORMAT_BC3_UNORM


        4 channel unsigned normalized block-compressed (BC3 compression) format


    .. autoattribute:: cuda.bindings.driver.CUarray_format.CU_AD_FORMAT_BC3_UNORM_SRGB


        4 channel unsigned normalized block-compressed (BC3 compression) format with sRGB encoding


    .. autoattribute:: cuda.bindings.driver.CUarray_format.CU_AD_FORMAT_BC4_UNORM


        1 channel unsigned normalized block-compressed (BC4 compression) format


    .. autoattribute:: cuda.bindings.driver.CUarray_format.CU_AD_FORMAT_BC4_SNORM


        1 channel signed normalized block-compressed (BC4 compression) format


    .. autoattribute:: cuda.bindings.driver.CUarray_format.CU_AD_FORMAT_BC5_UNORM


        2 channel unsigned normalized block-compressed (BC5 compression) format


    .. autoattribute:: cuda.bindings.driver.CUarray_format.CU_AD_FORMAT_BC5_SNORM


        2 channel signed normalized block-compressed (BC5 compression) format


    .. autoattribute:: cuda.bindings.driver.CUarray_format.CU_AD_FORMAT_BC6H_UF16


        3 channel unsigned half-float block-compressed (BC6H compression) format


    .. autoattribute:: cuda.bindings.driver.CUarray_format.CU_AD_FORMAT_BC6H_SF16


        3 channel signed half-float block-compressed (BC6H compression) format


    .. autoattribute:: cuda.bindings.driver.CUarray_format.CU_AD_FORMAT_BC7_UNORM


        4 channel unsigned normalized block-compressed (BC7 compression) format


    .. autoattribute:: cuda.bindings.driver.CUarray_format.CU_AD_FORMAT_BC7_UNORM_SRGB


        4 channel unsigned normalized block-compressed (BC7 compression) format with sRGB encoding


    .. autoattribute:: cuda.bindings.driver.CUarray_format.CU_AD_FORMAT_P010


        10-bit YUV planar format, with 4:2:0 sampling


    .. autoattribute:: cuda.bindings.driver.CUarray_format.CU_AD_FORMAT_P016


        16-bit YUV planar format, with 4:2:0 sampling


    .. autoattribute:: cuda.bindings.driver.CUarray_format.CU_AD_FORMAT_NV16


        8-bit YUV planar format, with 4:2:2 sampling


    .. autoattribute:: cuda.bindings.driver.CUarray_format.CU_AD_FORMAT_P210


        10-bit YUV planar format, with 4:2:2 sampling


    .. autoattribute:: cuda.bindings.driver.CUarray_format.CU_AD_FORMAT_P216


        16-bit YUV planar format, with 4:2:2 sampling


    .. autoattribute:: cuda.bindings.driver.CUarray_format.CU_AD_FORMAT_YUY2


        2 channel, 8-bit YUV packed planar format, with 4:2:2 sampling


    .. autoattribute:: cuda.bindings.driver.CUarray_format.CU_AD_FORMAT_Y210


        2 channel, 10-bit YUV packed planar format, with 4:2:2 sampling


    .. autoattribute:: cuda.bindings.driver.CUarray_format.CU_AD_FORMAT_Y216


        2 channel, 16-bit YUV packed planar format, with 4:2:2 sampling


    .. autoattribute:: cuda.bindings.driver.CUarray_format.CU_AD_FORMAT_AYUV


        4 channel, 8-bit YUV packed planar format, with 4:4:4 sampling


    .. autoattribute:: cuda.bindings.driver.CUarray_format.CU_AD_FORMAT_Y410


        10-bit YUV packed planar format, with 4:4:4 sampling


    .. autoattribute:: cuda.bindings.driver.CUarray_format.CU_AD_FORMAT_Y416


        4 channel, 12-bit YUV packed planar format, with 4:4:4 sampling


    .. autoattribute:: cuda.bindings.driver.CUarray_format.CU_AD_FORMAT_Y444_PLANAR8


        3 channel 8-bit YUV planar format, with 4:4:4 sampling


    .. autoattribute:: cuda.bindings.driver.CUarray_format.CU_AD_FORMAT_Y444_PLANAR10


        3 channel 10-bit YUV planar format, with 4:4:4 sampling


    .. autoattribute:: cuda.bindings.driver.CUarray_format.CU_AD_FORMAT_YUV444_8bit_SemiPlanar


        3 channel 8-bit YUV semi-planar format, with 4:4:4 sampling


    .. autoattribute:: cuda.bindings.driver.CUarray_format.CU_AD_FORMAT_YUV444_16bit_SemiPlanar


        3 channel 16-bit YUV semi-planar format, with 4:4:4 sampling


    .. autoattribute:: cuda.bindings.driver.CUarray_format.CU_AD_FORMAT_UNORM_INT_101010_2


        4 channel unorm R10G10B10A2 RGB format


    .. autoattribute:: cuda.bindings.driver.CUarray_format.CU_AD_FORMAT_MAX

.. autoclass:: cuda.bindings.driver.CUaddress_mode

    .. autoattribute:: cuda.bindings.driver.CUaddress_mode.CU_TR_ADDRESS_MODE_WRAP


        Wrapping address mode


    .. autoattribute:: cuda.bindings.driver.CUaddress_mode.CU_TR_ADDRESS_MODE_CLAMP


        Clamp to edge address mode


    .. autoattribute:: cuda.bindings.driver.CUaddress_mode.CU_TR_ADDRESS_MODE_MIRROR


        Mirror address mode


    .. autoattribute:: cuda.bindings.driver.CUaddress_mode.CU_TR_ADDRESS_MODE_BORDER


        Border address mode

.. autoclass:: cuda.bindings.driver.CUfilter_mode

    .. autoattribute:: cuda.bindings.driver.CUfilter_mode.CU_TR_FILTER_MODE_POINT


        Point filter mode


    .. autoattribute:: cuda.bindings.driver.CUfilter_mode.CU_TR_FILTER_MODE_LINEAR


        Linear filter mode

.. autoclass:: cuda.bindings.driver.CUdevice_attribute

    .. autoattribute:: cuda.bindings.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK


        Maximum number of threads per block


    .. autoattribute:: cuda.bindings.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X


        Maximum block dimension X


    .. autoattribute:: cuda.bindings.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y


        Maximum block dimension Y


    .. autoattribute:: cuda.bindings.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z


        Maximum block dimension Z


    .. autoattribute:: cuda.bindings.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X


        Maximum grid dimension X


    .. autoattribute:: cuda.bindings.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y


        Maximum grid dimension Y


    .. autoattribute:: cuda.bindings.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z


        Maximum grid dimension Z


    .. autoattribute:: cuda.bindings.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK


        Maximum shared memory available per block in bytes


    .. autoattribute:: cuda.bindings.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_SHARED_MEMORY_PER_BLOCK


        Deprecated, use CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK


    .. autoattribute:: cuda.bindings.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY


        Memory available on device for constant variables in a CUDA C kernel in bytes


    .. autoattribute:: cuda.bindings.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_WARP_SIZE


        Warp size in threads


    .. autoattribute:: cuda.bindings.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_PITCH


        Maximum pitch in bytes allowed by memory copies


    .. autoattribute:: cuda.bindings.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK


        Maximum number of 32-bit registers available per block


    .. autoattribute:: cuda.bindings.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_REGISTERS_PER_BLOCK


        Deprecated, use CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK


    .. autoattribute:: cuda.bindings.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_CLOCK_RATE


        Typical clock frequency in kilohertz


    .. autoattribute:: cuda.bindings.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT


        Alignment requirement for textures


    .. autoattribute:: cuda.bindings.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_GPU_OVERLAP


        Device can possibly copy memory and execute a kernel concurrently. Deprecated. Use instead CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT.


    .. autoattribute:: cuda.bindings.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT


        Number of multiprocessors on device


    .. autoattribute:: cuda.bindings.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT


        Specifies whether there is a run time limit on kernels


    .. autoattribute:: cuda.bindings.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_INTEGRATED


        Device is integrated with host memory


    .. autoattribute:: cuda.bindings.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY


        Device can map host memory into CUDA address space


    .. autoattribute:: cuda.bindings.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_MODE


        Compute mode (See :py:obj:`~.CUcomputemode` for details)


    .. autoattribute:: cuda.bindings.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH


        Maximum 1D texture width


    .. autoattribute:: cuda.bindings.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH


        Maximum 2D texture width


    .. autoattribute:: cuda.bindings.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT


        Maximum 2D texture height


    .. autoattribute:: cuda.bindings.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH


        Maximum 3D texture width


    .. autoattribute:: cuda.bindings.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT


        Maximum 3D texture height


    .. autoattribute:: cuda.bindings.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH


        Maximum 3D texture depth


    .. autoattribute:: cuda.bindings.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH


        Maximum 2D layered texture width


    .. autoattribute:: cuda.bindings.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT


        Maximum 2D layered texture height


    .. autoattribute:: cuda.bindings.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS


        Maximum layers in a 2D layered texture


    .. autoattribute:: cuda.bindings.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_WIDTH


        Deprecated, use CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH


    .. autoattribute:: cuda.bindings.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_HEIGHT


        Deprecated, use CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT


    .. autoattribute:: cuda.bindings.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_NUMSLICES


        Deprecated, use CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS


    .. autoattribute:: cuda.bindings.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT


        Alignment requirement for surfaces


    .. autoattribute:: cuda.bindings.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS


        Device can possibly execute multiple kernels concurrently


    .. autoattribute:: cuda.bindings.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_ECC_ENABLED


        Device has ECC support enabled


    .. autoattribute:: cuda.bindings.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_PCI_BUS_ID


        PCI bus ID of the device


    .. autoattribute:: cuda.bindings.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID


        PCI device ID of the device


    .. autoattribute:: cuda.bindings.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_TCC_DRIVER


        Device is using TCC driver model


    .. autoattribute:: cuda.bindings.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE


        Peak memory clock frequency in kilohertz


    .. autoattribute:: cuda.bindings.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH


        Global memory bus width in bits


    .. autoattribute:: cuda.bindings.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE


        Size of L2 cache in bytes


    .. autoattribute:: cuda.bindings.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR


        Maximum resident threads per multiprocessor


    .. autoattribute:: cuda.bindings.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT


        Number of asynchronous engines


    .. autoattribute:: cuda.bindings.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING


        Device shares a unified address space with the host


    .. autoattribute:: cuda.bindings.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH


        Maximum 1D layered texture width


    .. autoattribute:: cuda.bindings.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS


        Maximum layers in a 1D layered texture


    .. autoattribute:: cuda.bindings.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_CAN_TEX2D_GATHER


        Deprecated, do not use.


    .. autoattribute:: cuda.bindings.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_WIDTH


        Maximum 2D texture width if CUDA_ARRAY3D_TEXTURE_GATHER is set


    .. autoattribute:: cuda.bindings.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_HEIGHT


        Maximum 2D texture height if CUDA_ARRAY3D_TEXTURE_GATHER is set


    .. autoattribute:: cuda.bindings.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH_ALTERNATE


        Alternate maximum 3D texture width


    .. autoattribute:: cuda.bindings.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT_ALTERNATE


        Alternate maximum 3D texture height


    .. autoattribute:: cuda.bindings.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH_ALTERNATE


        Alternate maximum 3D texture depth


    .. autoattribute:: cuda.bindings.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID


        PCI domain ID of the device


    .. autoattribute:: cuda.bindings.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT


        Pitch alignment requirement for textures


    .. autoattribute:: cuda.bindings.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_WIDTH


        Maximum cubemap texture width/height


    .. autoattribute:: cuda.bindings.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_WIDTH


        Maximum cubemap layered texture width/height


    .. autoattribute:: cuda.bindings.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_LAYERS


        Maximum layers in a cubemap layered texture


    .. autoattribute:: cuda.bindings.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_WIDTH


        Maximum 1D surface width


    .. autoattribute:: cuda.bindings.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_WIDTH


        Maximum 2D surface width


    .. autoattribute:: cuda.bindings.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_HEIGHT


        Maximum 2D surface height


    .. autoattribute:: cuda.bindings.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_WIDTH


        Maximum 3D surface width


    .. autoattribute:: cuda.bindings.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_HEIGHT


        Maximum 3D surface height


    .. autoattribute:: cuda.bindings.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_DEPTH


        Maximum 3D surface depth


    .. autoattribute:: cuda.bindings.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_WIDTH


        Maximum 1D layered surface width


    .. autoattribute:: cuda.bindings.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_LAYERS


        Maximum layers in a 1D layered surface


    .. autoattribute:: cuda.bindings.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_WIDTH


        Maximum 2D layered surface width


    .. autoattribute:: cuda.bindings.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_HEIGHT


        Maximum 2D layered surface height


    .. autoattribute:: cuda.bindings.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_LAYERS


        Maximum layers in a 2D layered surface


    .. autoattribute:: cuda.bindings.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_WIDTH


        Maximum cubemap surface width


    .. autoattribute:: cuda.bindings.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_WIDTH


        Maximum cubemap layered surface width


    .. autoattribute:: cuda.bindings.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_LAYERS


        Maximum layers in a cubemap layered surface


    .. autoattribute:: cuda.bindings.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LINEAR_WIDTH


        Deprecated, do not use. Use cudaDeviceGetTexture1DLinearMaxWidth() or :py:obj:`~.cuDeviceGetTexture1DLinearMaxWidth()` instead.


    .. autoattribute:: cuda.bindings.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_WIDTH


        Maximum 2D linear texture width


    .. autoattribute:: cuda.bindings.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_HEIGHT


        Maximum 2D linear texture height


    .. autoattribute:: cuda.bindings.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_PITCH


        Maximum 2D linear texture pitch in bytes


    .. autoattribute:: cuda.bindings.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_WIDTH


        Maximum mipmapped 2D texture width


    .. autoattribute:: cuda.bindings.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_HEIGHT


        Maximum mipmapped 2D texture height


    .. autoattribute:: cuda.bindings.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR


        Major compute capability version number


    .. autoattribute:: cuda.bindings.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR


        Minor compute capability version number


    .. autoattribute:: cuda.bindings.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH


        Maximum mipmapped 1D texture width


    .. autoattribute:: cuda.bindings.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED


        Device supports stream priorities


    .. autoattribute:: cuda.bindings.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED


        Device supports caching globals in L1


    .. autoattribute:: cuda.bindings.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED


        Device supports caching locals in L1


    .. autoattribute:: cuda.bindings.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR


        Maximum shared memory available per multiprocessor in bytes


    .. autoattribute:: cuda.bindings.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR


        Maximum number of 32-bit registers available per multiprocessor


    .. autoattribute:: cuda.bindings.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY


        Device can allocate managed memory on this system


    .. autoattribute:: cuda.bindings.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD


        Device is on a multi-GPU board


    .. autoattribute:: cuda.bindings.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID


        Unique id for a group of devices on the same multi-GPU board


    .. autoattribute:: cuda.bindings.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED


        Link between the device and the host supports all native atomic operations


    .. autoattribute:: cuda.bindings.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO


        Ratio of single precision performance (in floating-point operations per second) to double precision performance


    .. autoattribute:: cuda.bindings.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS


        Device supports coherently accessing pageable memory without calling cudaHostRegister on it


    .. autoattribute:: cuda.bindings.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS


        Device can coherently access managed memory concurrently with the CPU


    .. autoattribute:: cuda.bindings.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED


        Device supports compute preemption.


    .. autoattribute:: cuda.bindings.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM


        Device can access host registered memory at the same virtual address as the CPU


    .. autoattribute:: cuda.bindings.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS_V1


        Deprecated, along with v1 MemOps API, :py:obj:`~.cuStreamBatchMemOp` and related APIs are supported.


    .. autoattribute:: cuda.bindings.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS_V1


        Deprecated, along with v1 MemOps API, 64-bit operations are supported in :py:obj:`~.cuStreamBatchMemOp` and related APIs.


    .. autoattribute:: cuda.bindings.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR_V1


        Deprecated, along with v1 MemOps API, :py:obj:`~.CU_STREAM_WAIT_VALUE_NOR` is supported.


    .. autoattribute:: cuda.bindings.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH


        Device supports launching cooperative kernels via :py:obj:`~.cuLaunchCooperativeKernel`


    .. autoattribute:: cuda.bindings.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COOPERATIVE_MULTI_DEVICE_LAUNCH


        Deprecated, :py:obj:`~.cuLaunchCooperativeKernelMultiDevice` is deprecated.


    .. autoattribute:: cuda.bindings.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN


        Maximum optin shared memory per block


    .. autoattribute:: cuda.bindings.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_CAN_FLUSH_REMOTE_WRITES


        The :py:obj:`~.CU_STREAM_WAIT_VALUE_FLUSH` flag and the :py:obj:`~.CU_STREAM_MEM_OP_FLUSH_REMOTE_WRITES` MemOp are supported on the device. See :py:obj:`~.Stream Memory Operations` for additional details.


    .. autoattribute:: cuda.bindings.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_HOST_REGISTER_SUPPORTED


        Device supports host memory registration via :py:obj:`~.cudaHostRegister`.


    .. autoattribute:: cuda.bindings.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES


        Device accesses pageable memory via the host's page tables.


    .. autoattribute:: cuda.bindings.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_DIRECT_MANAGED_MEM_ACCESS_FROM_HOST


        The host can directly access managed memory on the device without migration.


    .. autoattribute:: cuda.bindings.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED


        Deprecated, Use CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED


    .. autoattribute:: cuda.bindings.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED


        Device supports virtual memory management APIs like :py:obj:`~.cuMemAddressReserve`, :py:obj:`~.cuMemCreate`, :py:obj:`~.cuMemMap` and related APIs


    .. autoattribute:: cuda.bindings.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED


        Device supports exporting memory to a posix file descriptor with :py:obj:`~.cuMemExportToShareableHandle`, if requested via :py:obj:`~.cuMemCreate`


    .. autoattribute:: cuda.bindings.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_HANDLE_SUPPORTED


        Device supports exporting memory to a Win32 NT handle with :py:obj:`~.cuMemExportToShareableHandle`, if requested via :py:obj:`~.cuMemCreate`


    .. autoattribute:: cuda.bindings.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_KMT_HANDLE_SUPPORTED


        Device supports exporting memory to a Win32 KMT handle with :py:obj:`~.cuMemExportToShareableHandle`, if requested via :py:obj:`~.cuMemCreate`


    .. autoattribute:: cuda.bindings.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR


        Maximum number of blocks per multiprocessor


    .. autoattribute:: cuda.bindings.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_GENERIC_COMPRESSION_SUPPORTED


        Device supports compression of memory


    .. autoattribute:: cuda.bindings.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_PERSISTING_L2_CACHE_SIZE


        Maximum L2 persisting lines capacity setting in bytes.


    .. autoattribute:: cuda.bindings.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_ACCESS_POLICY_WINDOW_SIZE


        Maximum value of :py:obj:`~.CUaccessPolicyWindow.num_bytes`.


    .. autoattribute:: cuda.bindings.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED


        Device supports specifying the GPUDirect RDMA flag with :py:obj:`~.cuMemCreate`


    .. autoattribute:: cuda.bindings.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_RESERVED_SHARED_MEMORY_PER_BLOCK


        Shared memory reserved by CUDA driver per block in bytes


    .. autoattribute:: cuda.bindings.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_SPARSE_CUDA_ARRAY_SUPPORTED


        Device supports sparse CUDA arrays and sparse CUDA mipmapped arrays


    .. autoattribute:: cuda.bindings.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_READ_ONLY_HOST_REGISTER_SUPPORTED


        Device supports using the :py:obj:`~.cuMemHostRegister` flag :py:obj:`~.CU_MEMHOSTERGISTER_READ_ONLY` to register memory that must be mapped as read-only to the GPU


    .. autoattribute:: cuda.bindings.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_TIMELINE_SEMAPHORE_INTEROP_SUPPORTED


        External timeline semaphore interop is supported on the device


    .. autoattribute:: cuda.bindings.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED


        Device supports using the :py:obj:`~.cuMemAllocAsync` and :py:obj:`~.cuMemPool` family of APIs


    .. autoattribute:: cuda.bindings.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED


        Device supports GPUDirect RDMA APIs, like nvidia_p2p_get_pages (see https://docs.nvidia.com/cuda/gpudirect-rdma for more information)


    .. autoattribute:: cuda.bindings.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_FLUSH_WRITES_OPTIONS


        The returned attribute shall be interpreted as a bitmask, where the individual bits are described by the :py:obj:`~.CUflushGPUDirectRDMAWritesOptions` enum


    .. autoattribute:: cuda.bindings.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WRITES_ORDERING


        GPUDirect RDMA writes to the device do not need to be flushed for consumers within the scope indicated by the returned attribute. See :py:obj:`~.CUGPUDirectRDMAWritesOrdering` for the numerical values returned here.


    .. autoattribute:: cuda.bindings.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MEMPOOL_SUPPORTED_HANDLE_TYPES


        Handle types supported with mempool based IPC


    .. autoattribute:: cuda.bindings.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_CLUSTER_LAUNCH


        Indicates device supports cluster launch


    .. autoattribute:: cuda.bindings.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_DEFERRED_MAPPING_CUDA_ARRAY_SUPPORTED


        Device supports deferred mapping CUDA arrays and CUDA mipmapped arrays


    .. autoattribute:: cuda.bindings.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS


        64-bit operations are supported in :py:obj:`~.cuStreamBatchMemOp` and related MemOp APIs.


    .. autoattribute:: cuda.bindings.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR


        :py:obj:`~.CU_STREAM_WAIT_VALUE_NOR` is supported by MemOp APIs.


    .. autoattribute:: cuda.bindings.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED


        Device supports buffer sharing with dma_buf mechanism.


    .. autoattribute:: cuda.bindings.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_IPC_EVENT_SUPPORTED


        Device supports IPC Events.


    .. autoattribute:: cuda.bindings.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MEM_SYNC_DOMAIN_COUNT


        Number of memory domains the device supports.


    .. autoattribute:: cuda.bindings.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_TENSOR_MAP_ACCESS_SUPPORTED


        Device supports accessing memory using Tensor Map.


    .. autoattribute:: cuda.bindings.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED


        Device supports exporting memory to a fabric handle with :py:obj:`~.cuMemExportToShareableHandle()` or requested with :py:obj:`~.cuMemCreate()`


    .. autoattribute:: cuda.bindings.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_UNIFIED_FUNCTION_POINTERS


        Device supports unified function pointers.


    .. autoattribute:: cuda.bindings.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_NUMA_CONFIG


        NUMA configuration of a device: value is of type :py:obj:`~.CUdeviceNumaConfig` enum


    .. autoattribute:: cuda.bindings.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_NUMA_ID


        NUMA node ID of the GPU memory


    .. autoattribute:: cuda.bindings.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED


        Device supports switch multicast and reduction operations.


    .. autoattribute:: cuda.bindings.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MPS_ENABLED


        Indicates if contexts created on this device will be shared via MPS


    .. autoattribute:: cuda.bindings.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID


        NUMA ID of the host node closest to the device. Returns -1 when system does not support NUMA.


    .. autoattribute:: cuda.bindings.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_D3D12_CIG_SUPPORTED


        Device supports CIG with D3D12.


    .. autoattribute:: cuda.bindings.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MEM_DECOMPRESS_ALGORITHM_MASK


        The returned valued shall be interpreted as a bitmask, where the individual bits are described by the :py:obj:`~.CUmemDecompressAlgorithm` enum.


    .. autoattribute:: cuda.bindings.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MEM_DECOMPRESS_MAXIMUM_LENGTH


        The returned valued is the maximum length in bytes of a single decompress operation that is allowed.


    .. autoattribute:: cuda.bindings.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_VULKAN_CIG_SUPPORTED


        Device supports CIG with Vulkan.


    .. autoattribute:: cuda.bindings.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_GPU_PCI_DEVICE_ID


        The combined 16-bit PCI device ID and 16-bit PCI vendor ID.


    .. autoattribute:: cuda.bindings.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_GPU_PCI_SUBSYSTEM_ID


        The combined 16-bit PCI subsystem ID and 16-bit PCI subsystem vendor ID.


    .. autoattribute:: cuda.bindings.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_HOST_NUMA_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED


        Device supports HOST_NUMA location with the virtual memory management APIs like :py:obj:`~.cuMemCreate`, :py:obj:`~.cuMemMap` and related APIs


    .. autoattribute:: cuda.bindings.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_HOST_NUMA_MEMORY_POOLS_SUPPORTED


        Device supports HOST_NUMA location with the :py:obj:`~.cuMemAllocAsync` and :py:obj:`~.cuMemPool` family of APIs


    .. autoattribute:: cuda.bindings.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_HOST_NUMA_MULTINODE_IPC_SUPPORTED


        Device supports HOST_NUMA location IPC between nodes in a multi-node system.


    .. autoattribute:: cuda.bindings.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_HOST_MEMORY_POOLS_SUPPORTED


        Device suports HOST location with the :py:obj:`~.cuMemAllocAsync` and :py:obj:`~.cuMemPool` family of APIs


    .. autoattribute:: cuda.bindings.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_HOST_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED


        Device supports HOST location with the virtual memory management APIs like :py:obj:`~.cuMemCreate`, :py:obj:`~.cuMemMap` and related APIs


    .. autoattribute:: cuda.bindings.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_HOST_ALLOC_DMA_BUF_SUPPORTED


        Device supports page-locked host memory buffer sharing with dma_buf mechanism.


    .. autoattribute:: cuda.bindings.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_ONLY_PARTIAL_HOST_NATIVE_ATOMIC_SUPPORTED


        Link between the device and the host supports only some native atomic operations


    .. autoattribute:: cuda.bindings.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX

.. autoclass:: cuda.bindings.driver.CUpointer_attribute

    .. autoattribute:: cuda.bindings.driver.CUpointer_attribute.CU_POINTER_ATTRIBUTE_CONTEXT


        The :py:obj:`~.CUcontext` on which a pointer was allocated or registered


    .. autoattribute:: cuda.bindings.driver.CUpointer_attribute.CU_POINTER_ATTRIBUTE_MEMORY_TYPE


        The :py:obj:`~.CUmemorytype` describing the physical location of a pointer


    .. autoattribute:: cuda.bindings.driver.CUpointer_attribute.CU_POINTER_ATTRIBUTE_DEVICE_POINTER


        The address at which a pointer's memory may be accessed on the device


    .. autoattribute:: cuda.bindings.driver.CUpointer_attribute.CU_POINTER_ATTRIBUTE_HOST_POINTER


        The address at which a pointer's memory may be accessed on the host


    .. autoattribute:: cuda.bindings.driver.CUpointer_attribute.CU_POINTER_ATTRIBUTE_P2P_TOKENS


        A pair of tokens for use with the nv-p2p.h Linux kernel interface


    .. autoattribute:: cuda.bindings.driver.CUpointer_attribute.CU_POINTER_ATTRIBUTE_SYNC_MEMOPS


        Synchronize every synchronous memory operation initiated on this region


    .. autoattribute:: cuda.bindings.driver.CUpointer_attribute.CU_POINTER_ATTRIBUTE_BUFFER_ID


        A process-wide unique ID for an allocated memory region


    .. autoattribute:: cuda.bindings.driver.CUpointer_attribute.CU_POINTER_ATTRIBUTE_IS_MANAGED


        Indicates if the pointer points to managed memory


    .. autoattribute:: cuda.bindings.driver.CUpointer_attribute.CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL


        A device ordinal of a device on which a pointer was allocated or registered


    .. autoattribute:: cuda.bindings.driver.CUpointer_attribute.CU_POINTER_ATTRIBUTE_IS_LEGACY_CUDA_IPC_CAPABLE


        1 if this pointer maps to an allocation that is suitable for :py:obj:`~.cudaIpcGetMemHandle`, 0 otherwise


    .. autoattribute:: cuda.bindings.driver.CUpointer_attribute.CU_POINTER_ATTRIBUTE_RANGE_START_ADDR


        Starting address for this requested pointer


    .. autoattribute:: cuda.bindings.driver.CUpointer_attribute.CU_POINTER_ATTRIBUTE_RANGE_SIZE


        Size of the address range for this requested pointer


    .. autoattribute:: cuda.bindings.driver.CUpointer_attribute.CU_POINTER_ATTRIBUTE_MAPPED


        1 if this pointer is in a valid address range that is mapped to a backing allocation, 0 otherwise


    .. autoattribute:: cuda.bindings.driver.CUpointer_attribute.CU_POINTER_ATTRIBUTE_ALLOWED_HANDLE_TYPES


        Bitmask of allowed :py:obj:`~.CUmemAllocationHandleType` for this allocation


    .. autoattribute:: cuda.bindings.driver.CUpointer_attribute.CU_POINTER_ATTRIBUTE_IS_GPU_DIRECT_RDMA_CAPABLE


        1 if the memory this pointer is referencing can be used with the GPUDirect RDMA API


    .. autoattribute:: cuda.bindings.driver.CUpointer_attribute.CU_POINTER_ATTRIBUTE_ACCESS_FLAGS


        Returns the access flags the device associated with the current context has on the corresponding memory referenced by the pointer given


    .. autoattribute:: cuda.bindings.driver.CUpointer_attribute.CU_POINTER_ATTRIBUTE_MEMPOOL_HANDLE


        Returns the mempool handle for the allocation if it was allocated from a mempool. Otherwise returns NULL.


    .. autoattribute:: cuda.bindings.driver.CUpointer_attribute.CU_POINTER_ATTRIBUTE_MAPPING_SIZE


        Size of the actual underlying mapping that the pointer belongs to


    .. autoattribute:: cuda.bindings.driver.CUpointer_attribute.CU_POINTER_ATTRIBUTE_MAPPING_BASE_ADDR


        The start address of the mapping that the pointer belongs to


    .. autoattribute:: cuda.bindings.driver.CUpointer_attribute.CU_POINTER_ATTRIBUTE_MEMORY_BLOCK_ID


        A process-wide unique id corresponding to the physical allocation the pointer belongs to


    .. autoattribute:: cuda.bindings.driver.CUpointer_attribute.CU_POINTER_ATTRIBUTE_IS_HW_DECOMPRESS_CAPABLE


        Returns in `*data` a boolean that indicates whether the pointer points to memory that is capable to be used for hardware accelerated decompression.

.. autoclass:: cuda.bindings.driver.CUfunction_attribute

    .. autoattribute:: cuda.bindings.driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK


        The maximum number of threads per block, beyond which a launch of the function would fail. This number depends on both the function and the device on which the function is currently loaded.


    .. autoattribute:: cuda.bindings.driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES


        The size in bytes of statically-allocated shared memory required by this function. This does not include dynamically-allocated shared memory requested by the user at runtime.


    .. autoattribute:: cuda.bindings.driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES


        The size in bytes of user-allocated constant memory required by this function.


    .. autoattribute:: cuda.bindings.driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES


        The size in bytes of local memory used by each thread of this function.


    .. autoattribute:: cuda.bindings.driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_NUM_REGS


        The number of registers used by each thread of this function.


    .. autoattribute:: cuda.bindings.driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_PTX_VERSION


        The PTX virtual architecture version for which the function was compiled. This value is the major PTX version * 10 + the minor PTX version, so a PTX version 1.3 function would return the value 13. Note that this may return the undefined value of 0 for cubins compiled prior to CUDA 3.0.


    .. autoattribute:: cuda.bindings.driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_BINARY_VERSION


        The binary architecture version for which the function was compiled. This value is the major binary version * 10 + the minor binary version, so a binary version 1.3 function would return the value 13. Note that this will return a value of 10 for legacy cubins that do not have a properly-encoded binary architecture version.


    .. autoattribute:: cuda.bindings.driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_CACHE_MODE_CA


        The attribute to indicate whether the function has been compiled with user specified option "-Xptxas --dlcm=ca" set .


    .. autoattribute:: cuda.bindings.driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES


        The maximum size in bytes of dynamically-allocated shared memory that can be used by this function. If the user-specified dynamic shared memory size is larger than this value, the launch will fail. The default value of this attribute is :py:obj:`~.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK` - :py:obj:`~.CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES`, except when :py:obj:`~.CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES` is greater than :py:obj:`~.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK`, then the default value of this attribute is 0. The value can be increased to :py:obj:`~.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN` - :py:obj:`~.CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES`. See :py:obj:`~.cuFuncSetAttribute`, :py:obj:`~.cuKernelSetAttribute`


    .. autoattribute:: cuda.bindings.driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT


        On devices where the L1 cache and shared memory use the same hardware resources, this sets the shared memory carveout preference, in percent of the total shared memory. Refer to :py:obj:`~.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR`. This is only a hint, and the driver can choose a different ratio if required to execute the function. See :py:obj:`~.cuFuncSetAttribute`, :py:obj:`~.cuKernelSetAttribute`


    .. autoattribute:: cuda.bindings.driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_CLUSTER_SIZE_MUST_BE_SET


        If this attribute is set, the kernel must launch with a valid cluster size specified. See :py:obj:`~.cuFuncSetAttribute`, :py:obj:`~.cuKernelSetAttribute`


    .. autoattribute:: cuda.bindings.driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_WIDTH


        The required cluster width in blocks. The values must either all be 0 or all be positive. The validity of the cluster dimensions is otherwise checked at launch time.



        If the value is set during compile time, it cannot be set at runtime. Setting it at runtime will return CUDA_ERROR_NOT_PERMITTED. See :py:obj:`~.cuFuncSetAttribute`, :py:obj:`~.cuKernelSetAttribute`


    .. autoattribute:: cuda.bindings.driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_HEIGHT


        The required cluster height in blocks. The values must either all be 0 or all be positive. The validity of the cluster dimensions is otherwise checked at launch time.



        If the value is set during compile time, it cannot be set at runtime. Setting it at runtime should return CUDA_ERROR_NOT_PERMITTED. See :py:obj:`~.cuFuncSetAttribute`, :py:obj:`~.cuKernelSetAttribute`


    .. autoattribute:: cuda.bindings.driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_DEPTH


        The required cluster depth in blocks. The values must either all be 0 or all be positive. The validity of the cluster dimensions is otherwise checked at launch time.



        If the value is set during compile time, it cannot be set at runtime. Setting it at runtime should return CUDA_ERROR_NOT_PERMITTED. See :py:obj:`~.cuFuncSetAttribute`, :py:obj:`~.cuKernelSetAttribute`


    .. autoattribute:: cuda.bindings.driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED


        Whether the function can be launched with non-portable cluster size. 1 is allowed, 0 is disallowed. A non-portable cluster size may only function on the specific SKUs the program is tested on. The launch might fail if the program is run on a different hardware platform.



        CUDA API provides cudaOccupancyMaxActiveClusters to assist with checking whether the desired size can be launched on the current device.



        Portable Cluster Size



        A portable cluster size is guaranteed to be functional on all compute capabilities higher than the target compute capability. The portable cluster size for sm_90 is 8 blocks per cluster. This value may increase for future compute capabilities.



        The specific hardware unit may support higher cluster sizes that’s not guaranteed to be portable. See :py:obj:`~.cuFuncSetAttribute`, :py:obj:`~.cuKernelSetAttribute`


    .. autoattribute:: cuda.bindings.driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE


        The block scheduling policy of a function. The value type is CUclusterSchedulingPolicy / cudaClusterSchedulingPolicy. See :py:obj:`~.cuFuncSetAttribute`, :py:obj:`~.cuKernelSetAttribute`


    .. autoattribute:: cuda.bindings.driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX

.. autoclass:: cuda.bindings.driver.CUfunc_cache

    .. autoattribute:: cuda.bindings.driver.CUfunc_cache.CU_FUNC_CACHE_PREFER_NONE


        no preference for shared memory or L1 (default)


    .. autoattribute:: cuda.bindings.driver.CUfunc_cache.CU_FUNC_CACHE_PREFER_SHARED


        prefer larger shared memory and smaller L1 cache


    .. autoattribute:: cuda.bindings.driver.CUfunc_cache.CU_FUNC_CACHE_PREFER_L1


        prefer larger L1 cache and smaller shared memory


    .. autoattribute:: cuda.bindings.driver.CUfunc_cache.CU_FUNC_CACHE_PREFER_EQUAL


        prefer equal sized L1 cache and shared memory

.. autoclass:: cuda.bindings.driver.CUsharedconfig

    .. autoattribute:: cuda.bindings.driver.CUsharedconfig.CU_SHARED_MEM_CONFIG_DEFAULT_BANK_SIZE


        set default shared memory bank size


    .. autoattribute:: cuda.bindings.driver.CUsharedconfig.CU_SHARED_MEM_CONFIG_FOUR_BYTE_BANK_SIZE


        set shared memory bank width to four bytes


    .. autoattribute:: cuda.bindings.driver.CUsharedconfig.CU_SHARED_MEM_CONFIG_EIGHT_BYTE_BANK_SIZE


        set shared memory bank width to eight bytes

.. autoclass:: cuda.bindings.driver.CUshared_carveout

    .. autoattribute:: cuda.bindings.driver.CUshared_carveout.CU_SHAREDMEM_CARVEOUT_DEFAULT


        No preference for shared memory or L1 (default)


    .. autoattribute:: cuda.bindings.driver.CUshared_carveout.CU_SHAREDMEM_CARVEOUT_MAX_SHARED


        Prefer maximum available shared memory, minimum L1 cache


    .. autoattribute:: cuda.bindings.driver.CUshared_carveout.CU_SHAREDMEM_CARVEOUT_MAX_L1


        Prefer maximum available L1 cache, minimum shared memory

.. autoclass:: cuda.bindings.driver.CUmemorytype

    .. autoattribute:: cuda.bindings.driver.CUmemorytype.CU_MEMORYTYPE_HOST


        Host memory


    .. autoattribute:: cuda.bindings.driver.CUmemorytype.CU_MEMORYTYPE_DEVICE


        Device memory


    .. autoattribute:: cuda.bindings.driver.CUmemorytype.CU_MEMORYTYPE_ARRAY


        Array memory


    .. autoattribute:: cuda.bindings.driver.CUmemorytype.CU_MEMORYTYPE_UNIFIED


        Unified device or host memory

.. autoclass:: cuda.bindings.driver.CUcomputemode

    .. autoattribute:: cuda.bindings.driver.CUcomputemode.CU_COMPUTEMODE_DEFAULT


        Default compute mode (Multiple contexts allowed per device)


    .. autoattribute:: cuda.bindings.driver.CUcomputemode.CU_COMPUTEMODE_PROHIBITED


        Compute-prohibited mode (No contexts can be created on this device at this time)


    .. autoattribute:: cuda.bindings.driver.CUcomputemode.CU_COMPUTEMODE_EXCLUSIVE_PROCESS


        Compute-exclusive-process mode (Only one context used by a single process can be present on this device at a time)

.. autoclass:: cuda.bindings.driver.CUmem_advise

    .. autoattribute:: cuda.bindings.driver.CUmem_advise.CU_MEM_ADVISE_SET_READ_MOSTLY


        Data will mostly be read and only occasionally be written to


    .. autoattribute:: cuda.bindings.driver.CUmem_advise.CU_MEM_ADVISE_UNSET_READ_MOSTLY


        Undo the effect of :py:obj:`~.CU_MEM_ADVISE_SET_READ_MOSTLY`


    .. autoattribute:: cuda.bindings.driver.CUmem_advise.CU_MEM_ADVISE_SET_PREFERRED_LOCATION


        Set the preferred location for the data as the specified device


    .. autoattribute:: cuda.bindings.driver.CUmem_advise.CU_MEM_ADVISE_UNSET_PREFERRED_LOCATION


        Clear the preferred location for the data


    .. autoattribute:: cuda.bindings.driver.CUmem_advise.CU_MEM_ADVISE_SET_ACCESSED_BY


        Data will be accessed by the specified device, so prevent page faults as much as possible


    .. autoattribute:: cuda.bindings.driver.CUmem_advise.CU_MEM_ADVISE_UNSET_ACCESSED_BY


        Let the Unified Memory subsystem decide on the page faulting policy for the specified device

.. autoclass:: cuda.bindings.driver.CUmem_range_attribute

    .. autoattribute:: cuda.bindings.driver.CUmem_range_attribute.CU_MEM_RANGE_ATTRIBUTE_READ_MOSTLY


        Whether the range will mostly be read and only occasionally be written to


    .. autoattribute:: cuda.bindings.driver.CUmem_range_attribute.CU_MEM_RANGE_ATTRIBUTE_PREFERRED_LOCATION


        The preferred location of the range


    .. autoattribute:: cuda.bindings.driver.CUmem_range_attribute.CU_MEM_RANGE_ATTRIBUTE_ACCESSED_BY


        Memory range has :py:obj:`~.CU_MEM_ADVISE_SET_ACCESSED_BY` set for specified device


    .. autoattribute:: cuda.bindings.driver.CUmem_range_attribute.CU_MEM_RANGE_ATTRIBUTE_LAST_PREFETCH_LOCATION


        The last location to which the range was prefetched


    .. autoattribute:: cuda.bindings.driver.CUmem_range_attribute.CU_MEM_RANGE_ATTRIBUTE_PREFERRED_LOCATION_TYPE


        The preferred location type of the range


    .. autoattribute:: cuda.bindings.driver.CUmem_range_attribute.CU_MEM_RANGE_ATTRIBUTE_PREFERRED_LOCATION_ID


        The preferred location id of the range


    .. autoattribute:: cuda.bindings.driver.CUmem_range_attribute.CU_MEM_RANGE_ATTRIBUTE_LAST_PREFETCH_LOCATION_TYPE


        The last location type to which the range was prefetched


    .. autoattribute:: cuda.bindings.driver.CUmem_range_attribute.CU_MEM_RANGE_ATTRIBUTE_LAST_PREFETCH_LOCATION_ID


        The last location id to which the range was prefetched

.. autoclass:: cuda.bindings.driver.CUjit_option

    .. autoattribute:: cuda.bindings.driver.CUjit_option.CU_JIT_MAX_REGISTERS


        Max number of registers that a thread may use.

        Option type: unsigned int

        Applies to: compiler only


    .. autoattribute:: cuda.bindings.driver.CUjit_option.CU_JIT_THREADS_PER_BLOCK


        IN: Specifies minimum number of threads per block to target compilation for

        OUT: Returns the number of threads the compiler actually targeted. This restricts the resource utilization of the compiler (e.g. max registers) such that a block with the given number of threads should be able to launch based on register limitations. Note, this option does not currently take into account any other resource limitations, such as shared memory utilization.

        Cannot be combined with :py:obj:`~.CU_JIT_TARGET`.

        Option type: unsigned int

        Applies to: compiler only


    .. autoattribute:: cuda.bindings.driver.CUjit_option.CU_JIT_WALL_TIME


        Overwrites the option value with the total wall clock time, in milliseconds, spent in the compiler and linker

        Option type: float

        Applies to: compiler and linker


    .. autoattribute:: cuda.bindings.driver.CUjit_option.CU_JIT_INFO_LOG_BUFFER


        Pointer to a buffer in which to print any log messages that are informational in nature (the buffer size is specified via option :py:obj:`~.CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES`)

        Option type: char *

        Applies to: compiler and linker


    .. autoattribute:: cuda.bindings.driver.CUjit_option.CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES


        IN: Log buffer size in bytes. Log messages will be capped at this size (including null terminator)

        OUT: Amount of log buffer filled with messages

        Option type: unsigned int

        Applies to: compiler and linker


    .. autoattribute:: cuda.bindings.driver.CUjit_option.CU_JIT_ERROR_LOG_BUFFER


        Pointer to a buffer in which to print any log messages that reflect errors (the buffer size is specified via option :py:obj:`~.CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES`)

        Option type: char *

        Applies to: compiler and linker


    .. autoattribute:: cuda.bindings.driver.CUjit_option.CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES


        IN: Log buffer size in bytes. Log messages will be capped at this size (including null terminator)

        OUT: Amount of log buffer filled with messages

        Option type: unsigned int

        Applies to: compiler and linker


    .. autoattribute:: cuda.bindings.driver.CUjit_option.CU_JIT_OPTIMIZATION_LEVEL


        Level of optimizations to apply to generated code (0 - 4), with 4 being the default and highest level of optimizations.

        Option type: unsigned int

        Applies to: compiler only


    .. autoattribute:: cuda.bindings.driver.CUjit_option.CU_JIT_TARGET_FROM_CUCONTEXT


        No option value required. Determines the target based on the current attached context (default)

        Option type: No option value needed

        Applies to: compiler and linker


    .. autoattribute:: cuda.bindings.driver.CUjit_option.CU_JIT_TARGET


        Target is chosen based on supplied :py:obj:`~.CUjit_target`. Cannot be combined with :py:obj:`~.CU_JIT_THREADS_PER_BLOCK`.

        Option type: unsigned int for enumerated type :py:obj:`~.CUjit_target`

        Applies to: compiler and linker


    .. autoattribute:: cuda.bindings.driver.CUjit_option.CU_JIT_FALLBACK_STRATEGY


        Specifies choice of fallback strategy if matching cubin is not found. Choice is based on supplied :py:obj:`~.CUjit_fallback`. This option cannot be used with cuLink* APIs as the linker requires exact matches.

        Option type: unsigned int for enumerated type :py:obj:`~.CUjit_fallback`

        Applies to: compiler only


    .. autoattribute:: cuda.bindings.driver.CUjit_option.CU_JIT_GENERATE_DEBUG_INFO


        Specifies whether to create debug information in output (-g) (0: false, default)

        Option type: int

        Applies to: compiler and linker


    .. autoattribute:: cuda.bindings.driver.CUjit_option.CU_JIT_LOG_VERBOSE


        Generate verbose log messages (0: false, default)

        Option type: int

        Applies to: compiler and linker


    .. autoattribute:: cuda.bindings.driver.CUjit_option.CU_JIT_GENERATE_LINE_INFO


        Generate line number information (-lineinfo) (0: false, default)

        Option type: int

        Applies to: compiler only


    .. autoattribute:: cuda.bindings.driver.CUjit_option.CU_JIT_CACHE_MODE


        Specifies whether to enable caching explicitly (-dlcm) 

        Choice is based on supplied :py:obj:`~.CUjit_cacheMode_enum`.

        Option type: unsigned int for enumerated type :py:obj:`~.CUjit_cacheMode_enum`

        Applies to: compiler only


    .. autoattribute:: cuda.bindings.driver.CUjit_option.CU_JIT_NEW_SM3X_OPT


        [Deprecated]


    .. autoattribute:: cuda.bindings.driver.CUjit_option.CU_JIT_FAST_COMPILE


        This jit option is used for internal purpose only.


    .. autoattribute:: cuda.bindings.driver.CUjit_option.CU_JIT_GLOBAL_SYMBOL_NAMES


        Array of device symbol names that will be relocated to the corresponding host addresses stored in :py:obj:`~.CU_JIT_GLOBAL_SYMBOL_ADDRESSES`.

        Must contain :py:obj:`~.CU_JIT_GLOBAL_SYMBOL_COUNT` entries.

        When loading a device module, driver will relocate all encountered unresolved symbols to the host addresses.

        It is only allowed to register symbols that correspond to unresolved global variables.

        It is illegal to register the same device symbol at multiple addresses.

        Option type: const char **

        Applies to: dynamic linker only


    .. autoattribute:: cuda.bindings.driver.CUjit_option.CU_JIT_GLOBAL_SYMBOL_ADDRESSES


        Array of host addresses that will be used to relocate corresponding device symbols stored in :py:obj:`~.CU_JIT_GLOBAL_SYMBOL_NAMES`.

        Must contain :py:obj:`~.CU_JIT_GLOBAL_SYMBOL_COUNT` entries.

        Option type: void **

        Applies to: dynamic linker only


    .. autoattribute:: cuda.bindings.driver.CUjit_option.CU_JIT_GLOBAL_SYMBOL_COUNT


        Number of entries in :py:obj:`~.CU_JIT_GLOBAL_SYMBOL_NAMES` and :py:obj:`~.CU_JIT_GLOBAL_SYMBOL_ADDRESSES` arrays.

        Option type: unsigned int

        Applies to: dynamic linker only


    .. autoattribute:: cuda.bindings.driver.CUjit_option.CU_JIT_LTO


        [Deprecated]



        Only valid with LTO-IR compiled with toolkits prior to CUDA 12.0


    .. autoattribute:: cuda.bindings.driver.CUjit_option.CU_JIT_FTZ


        [Deprecated]



        Only valid with LTO-IR compiled with toolkits prior to CUDA 12.0


    .. autoattribute:: cuda.bindings.driver.CUjit_option.CU_JIT_PREC_DIV


        [Deprecated]



        Only valid with LTO-IR compiled with toolkits prior to CUDA 12.0


    .. autoattribute:: cuda.bindings.driver.CUjit_option.CU_JIT_PREC_SQRT


        [Deprecated]



        Only valid with LTO-IR compiled with toolkits prior to CUDA 12.0


    .. autoattribute:: cuda.bindings.driver.CUjit_option.CU_JIT_FMA


        [Deprecated]



        Only valid with LTO-IR compiled with toolkits prior to CUDA 12.0


    .. autoattribute:: cuda.bindings.driver.CUjit_option.CU_JIT_REFERENCED_KERNEL_NAMES


        [Deprecated]



        Only valid with LTO-IR compiled with toolkits prior to CUDA 12.0


    .. autoattribute:: cuda.bindings.driver.CUjit_option.CU_JIT_REFERENCED_KERNEL_COUNT


        [Deprecated]



        Only valid with LTO-IR compiled with toolkits prior to CUDA 12.0


    .. autoattribute:: cuda.bindings.driver.CUjit_option.CU_JIT_REFERENCED_VARIABLE_NAMES


        [Deprecated]



        Only valid with LTO-IR compiled with toolkits prior to CUDA 12.0


    .. autoattribute:: cuda.bindings.driver.CUjit_option.CU_JIT_REFERENCED_VARIABLE_COUNT


        [Deprecated]



        Only valid with LTO-IR compiled with toolkits prior to CUDA 12.0


    .. autoattribute:: cuda.bindings.driver.CUjit_option.CU_JIT_OPTIMIZE_UNUSED_DEVICE_VARIABLES


        [Deprecated]



        Only valid with LTO-IR compiled with toolkits prior to CUDA 12.0


    .. autoattribute:: cuda.bindings.driver.CUjit_option.CU_JIT_POSITION_INDEPENDENT_CODE


        Generate position independent code (0: false)

        Option type: int

        Applies to: compiler only


    .. autoattribute:: cuda.bindings.driver.CUjit_option.CU_JIT_MIN_CTA_PER_SM


        This option hints to the JIT compiler the minimum number of CTAs from the kernel’s grid to be mapped to a SM. This option is ignored when used together with :py:obj:`~.CU_JIT_MAX_REGISTERS` or :py:obj:`~.CU_JIT_THREADS_PER_BLOCK`. Optimizations based on this option need :py:obj:`~.CU_JIT_MAX_THREADS_PER_BLOCK` to be specified as well. For kernels already using PTX directive .minnctapersm, this option will be ignored by default. Use :py:obj:`~.CU_JIT_OVERRIDE_DIRECTIVE_VALUES` to let this option take precedence over the PTX directive. Option type: unsigned int

        Applies to: compiler only


    .. autoattribute:: cuda.bindings.driver.CUjit_option.CU_JIT_MAX_THREADS_PER_BLOCK


        Maximum number threads in a thread block, computed as the product of the maximum extent specifed for each dimension of the block. This limit is guaranteed not to be exeeded in any invocation of the kernel. Exceeding the the maximum number of threads results in runtime error or kernel launch failure. For kernels already using PTX directive .maxntid, this option will be ignored by default. Use :py:obj:`~.CU_JIT_OVERRIDE_DIRECTIVE_VALUES` to let this option take precedence over the PTX directive. Option type: int

        Applies to: compiler only


    .. autoattribute:: cuda.bindings.driver.CUjit_option.CU_JIT_OVERRIDE_DIRECTIVE_VALUES


        This option lets the values specified using :py:obj:`~.CU_JIT_MAX_REGISTERS`, :py:obj:`~.CU_JIT_THREADS_PER_BLOCK`, :py:obj:`~.CU_JIT_MAX_THREADS_PER_BLOCK` and :py:obj:`~.CU_JIT_MIN_CTA_PER_SM` take precedence over any PTX directives. (0: Disable, default; 1: Enable) Option type: int

        Applies to: compiler only


    .. autoattribute:: cuda.bindings.driver.CUjit_option.CU_JIT_SPLIT_COMPILE


        This option specifies the maximum number of concurrent threads to use when running compiler optimizations. If the specified value is 1, the option will be ignored. If the specified value is 0, the number of threads will match the number of CPUs on the underlying machine. Otherwise, if the option is N, then up to N threads will be used. Option type: unsigned int

        Applies to: compiler only


    .. autoattribute:: cuda.bindings.driver.CUjit_option.CU_JIT_NUM_OPTIONS

.. autoclass:: cuda.bindings.driver.CUjit_target

    .. autoattribute:: cuda.bindings.driver.CUjit_target.CU_TARGET_COMPUTE_30


        Compute device class 3.0


    .. autoattribute:: cuda.bindings.driver.CUjit_target.CU_TARGET_COMPUTE_32


        Compute device class 3.2


    .. autoattribute:: cuda.bindings.driver.CUjit_target.CU_TARGET_COMPUTE_35


        Compute device class 3.5


    .. autoattribute:: cuda.bindings.driver.CUjit_target.CU_TARGET_COMPUTE_37


        Compute device class 3.7


    .. autoattribute:: cuda.bindings.driver.CUjit_target.CU_TARGET_COMPUTE_50


        Compute device class 5.0


    .. autoattribute:: cuda.bindings.driver.CUjit_target.CU_TARGET_COMPUTE_52


        Compute device class 5.2


    .. autoattribute:: cuda.bindings.driver.CUjit_target.CU_TARGET_COMPUTE_53


        Compute device class 5.3


    .. autoattribute:: cuda.bindings.driver.CUjit_target.CU_TARGET_COMPUTE_60


        Compute device class 6.0.


    .. autoattribute:: cuda.bindings.driver.CUjit_target.CU_TARGET_COMPUTE_61


        Compute device class 6.1.


    .. autoattribute:: cuda.bindings.driver.CUjit_target.CU_TARGET_COMPUTE_62


        Compute device class 6.2.


    .. autoattribute:: cuda.bindings.driver.CUjit_target.CU_TARGET_COMPUTE_70


        Compute device class 7.0.


    .. autoattribute:: cuda.bindings.driver.CUjit_target.CU_TARGET_COMPUTE_72


        Compute device class 7.2.


    .. autoattribute:: cuda.bindings.driver.CUjit_target.CU_TARGET_COMPUTE_75


        Compute device class 7.5.


    .. autoattribute:: cuda.bindings.driver.CUjit_target.CU_TARGET_COMPUTE_80


        Compute device class 8.0.


    .. autoattribute:: cuda.bindings.driver.CUjit_target.CU_TARGET_COMPUTE_86


        Compute device class 8.6.


    .. autoattribute:: cuda.bindings.driver.CUjit_target.CU_TARGET_COMPUTE_87


        Compute device class 8.7.


    .. autoattribute:: cuda.bindings.driver.CUjit_target.CU_TARGET_COMPUTE_89


        Compute device class 8.9.


    .. autoattribute:: cuda.bindings.driver.CUjit_target.CU_TARGET_COMPUTE_90


        Compute device class 9.0.


    .. autoattribute:: cuda.bindings.driver.CUjit_target.CU_TARGET_COMPUTE_100


        Compute device class 10.0.


    .. autoattribute:: cuda.bindings.driver.CUjit_target.CU_TARGET_COMPUTE_110


        Compute device class 11.0.


    .. autoattribute:: cuda.bindings.driver.CUjit_target.CU_TARGET_COMPUTE_103


        Compute device class 10.3.


    .. autoattribute:: cuda.bindings.driver.CUjit_target.CU_TARGET_COMPUTE_120


        Compute device class 12.0.


    .. autoattribute:: cuda.bindings.driver.CUjit_target.CU_TARGET_COMPUTE_121


        Compute device class 12.1. Compute device class 9.0. with accelerated features.


    .. autoattribute:: cuda.bindings.driver.CUjit_target.CU_TARGET_COMPUTE_90A


        Compute device class 10.0. with accelerated features.


    .. autoattribute:: cuda.bindings.driver.CUjit_target.CU_TARGET_COMPUTE_100A


        Compute device class 11.0 with accelerated features.


    .. autoattribute:: cuda.bindings.driver.CUjit_target.CU_TARGET_COMPUTE_110A


        Compute device class 10.3. with accelerated features.


    .. autoattribute:: cuda.bindings.driver.CUjit_target.CU_TARGET_COMPUTE_103A


        Compute device class 12.0. with accelerated features.


    .. autoattribute:: cuda.bindings.driver.CUjit_target.CU_TARGET_COMPUTE_120A


        Compute device class 12.1. with accelerated features.


    .. autoattribute:: cuda.bindings.driver.CUjit_target.CU_TARGET_COMPUTE_121A


        Compute device class 10.x with family features.


    .. autoattribute:: cuda.bindings.driver.CUjit_target.CU_TARGET_COMPUTE_100F


        Compute device class 11.0 with family features.


    .. autoattribute:: cuda.bindings.driver.CUjit_target.CU_TARGET_COMPUTE_110F


        Compute device class 10.3. with family features.


    .. autoattribute:: cuda.bindings.driver.CUjit_target.CU_TARGET_COMPUTE_103F


        Compute device class 12.0. with family features.


    .. autoattribute:: cuda.bindings.driver.CUjit_target.CU_TARGET_COMPUTE_120F


        Compute device class 12.1. with family features.


    .. autoattribute:: cuda.bindings.driver.CUjit_target.CU_TARGET_COMPUTE_121F

.. autoclass:: cuda.bindings.driver.CUjit_fallback

    .. autoattribute:: cuda.bindings.driver.CUjit_fallback.CU_PREFER_PTX


        Prefer to compile ptx if exact binary match not found


    .. autoattribute:: cuda.bindings.driver.CUjit_fallback.CU_PREFER_BINARY


        Prefer to fall back to compatible binary code if exact match not found

.. autoclass:: cuda.bindings.driver.CUjit_cacheMode

    .. autoattribute:: cuda.bindings.driver.CUjit_cacheMode.CU_JIT_CACHE_OPTION_NONE


        Compile with no -dlcm flag specified


    .. autoattribute:: cuda.bindings.driver.CUjit_cacheMode.CU_JIT_CACHE_OPTION_CG


        Compile with L1 cache disabled


    .. autoattribute:: cuda.bindings.driver.CUjit_cacheMode.CU_JIT_CACHE_OPTION_CA


        Compile with L1 cache enabled

.. autoclass:: cuda.bindings.driver.CUjitInputType

    .. autoattribute:: cuda.bindings.driver.CUjitInputType.CU_JIT_INPUT_CUBIN


        Compiled device-class-specific device code

        Applicable options: none


    .. autoattribute:: cuda.bindings.driver.CUjitInputType.CU_JIT_INPUT_PTX


        PTX source code

        Applicable options: PTX compiler options


    .. autoattribute:: cuda.bindings.driver.CUjitInputType.CU_JIT_INPUT_FATBINARY


        Bundle of multiple cubins and/or PTX of some device code

        Applicable options: PTX compiler options, :py:obj:`~.CU_JIT_FALLBACK_STRATEGY`


    .. autoattribute:: cuda.bindings.driver.CUjitInputType.CU_JIT_INPUT_OBJECT


        Host object with embedded device code

        Applicable options: PTX compiler options, :py:obj:`~.CU_JIT_FALLBACK_STRATEGY`


    .. autoattribute:: cuda.bindings.driver.CUjitInputType.CU_JIT_INPUT_LIBRARY


        Archive of host objects with embedded device code

        Applicable options: PTX compiler options, :py:obj:`~.CU_JIT_FALLBACK_STRATEGY`


    .. autoattribute:: cuda.bindings.driver.CUjitInputType.CU_JIT_INPUT_NVVM


        [Deprecated]



        Only valid with LTO-IR compiled with toolkits prior to CUDA 12.0


    .. autoattribute:: cuda.bindings.driver.CUjitInputType.CU_JIT_NUM_INPUT_TYPES

.. autoclass:: cuda.bindings.driver.CUgraphicsRegisterFlags

    .. autoattribute:: cuda.bindings.driver.CUgraphicsRegisterFlags.CU_GRAPHICS_REGISTER_FLAGS_NONE


    .. autoattribute:: cuda.bindings.driver.CUgraphicsRegisterFlags.CU_GRAPHICS_REGISTER_FLAGS_READ_ONLY


    .. autoattribute:: cuda.bindings.driver.CUgraphicsRegisterFlags.CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD


    .. autoattribute:: cuda.bindings.driver.CUgraphicsRegisterFlags.CU_GRAPHICS_REGISTER_FLAGS_SURFACE_LDST


    .. autoattribute:: cuda.bindings.driver.CUgraphicsRegisterFlags.CU_GRAPHICS_REGISTER_FLAGS_TEXTURE_GATHER

.. autoclass:: cuda.bindings.driver.CUgraphicsMapResourceFlags

    .. autoattribute:: cuda.bindings.driver.CUgraphicsMapResourceFlags.CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE


    .. autoattribute:: cuda.bindings.driver.CUgraphicsMapResourceFlags.CU_GRAPHICS_MAP_RESOURCE_FLAGS_READ_ONLY


    .. autoattribute:: cuda.bindings.driver.CUgraphicsMapResourceFlags.CU_GRAPHICS_MAP_RESOURCE_FLAGS_WRITE_DISCARD

.. autoclass:: cuda.bindings.driver.CUarray_cubemap_face

    .. autoattribute:: cuda.bindings.driver.CUarray_cubemap_face.CU_CUBEMAP_FACE_POSITIVE_X


        Positive X face of cubemap


    .. autoattribute:: cuda.bindings.driver.CUarray_cubemap_face.CU_CUBEMAP_FACE_NEGATIVE_X


        Negative X face of cubemap


    .. autoattribute:: cuda.bindings.driver.CUarray_cubemap_face.CU_CUBEMAP_FACE_POSITIVE_Y


        Positive Y face of cubemap


    .. autoattribute:: cuda.bindings.driver.CUarray_cubemap_face.CU_CUBEMAP_FACE_NEGATIVE_Y


        Negative Y face of cubemap


    .. autoattribute:: cuda.bindings.driver.CUarray_cubemap_face.CU_CUBEMAP_FACE_POSITIVE_Z


        Positive Z face of cubemap


    .. autoattribute:: cuda.bindings.driver.CUarray_cubemap_face.CU_CUBEMAP_FACE_NEGATIVE_Z


        Negative Z face of cubemap

.. autoclass:: cuda.bindings.driver.CUlimit

    .. autoattribute:: cuda.bindings.driver.CUlimit.CU_LIMIT_STACK_SIZE


        GPU thread stack size


    .. autoattribute:: cuda.bindings.driver.CUlimit.CU_LIMIT_PRINTF_FIFO_SIZE


        GPU printf FIFO size


    .. autoattribute:: cuda.bindings.driver.CUlimit.CU_LIMIT_MALLOC_HEAP_SIZE


        GPU malloc heap size


    .. autoattribute:: cuda.bindings.driver.CUlimit.CU_LIMIT_DEV_RUNTIME_SYNC_DEPTH


        GPU device runtime launch synchronize depth


    .. autoattribute:: cuda.bindings.driver.CUlimit.CU_LIMIT_DEV_RUNTIME_PENDING_LAUNCH_COUNT


        GPU device runtime pending launch count


    .. autoattribute:: cuda.bindings.driver.CUlimit.CU_LIMIT_MAX_L2_FETCH_GRANULARITY


        A value between 0 and 128 that indicates the maximum fetch granularity of L2 (in Bytes). This is a hint


    .. autoattribute:: cuda.bindings.driver.CUlimit.CU_LIMIT_PERSISTING_L2_CACHE_SIZE


        A size in bytes for L2 persisting lines cache size


    .. autoattribute:: cuda.bindings.driver.CUlimit.CU_LIMIT_SHMEM_SIZE


        A maximum size in bytes of shared memory available to CUDA kernels on a CIG context. Can only be queried, cannot be set


    .. autoattribute:: cuda.bindings.driver.CUlimit.CU_LIMIT_CIG_ENABLED


        A non-zero value indicates this CUDA context is a CIG-enabled context. Can only be queried, cannot be set


    .. autoattribute:: cuda.bindings.driver.CUlimit.CU_LIMIT_CIG_SHMEM_FALLBACK_ENABLED


        When set to zero, CUDA will fail to launch a kernel on a CIG context, instead of using the fallback path, if the kernel uses more shared memory than available


    .. autoattribute:: cuda.bindings.driver.CUlimit.CU_LIMIT_MAX

.. autoclass:: cuda.bindings.driver.CUresourcetype

    .. autoattribute:: cuda.bindings.driver.CUresourcetype.CU_RESOURCE_TYPE_ARRAY


        Array resource


    .. autoattribute:: cuda.bindings.driver.CUresourcetype.CU_RESOURCE_TYPE_MIPMAPPED_ARRAY


        Mipmapped array resource


    .. autoattribute:: cuda.bindings.driver.CUresourcetype.CU_RESOURCE_TYPE_LINEAR


        Linear resource


    .. autoattribute:: cuda.bindings.driver.CUresourcetype.CU_RESOURCE_TYPE_PITCH2D


        Pitch 2D resource

.. autoclass:: cuda.bindings.driver.CUaccessProperty

    .. autoattribute:: cuda.bindings.driver.CUaccessProperty.CU_ACCESS_PROPERTY_NORMAL


        Normal cache persistence.


    .. autoattribute:: cuda.bindings.driver.CUaccessProperty.CU_ACCESS_PROPERTY_STREAMING


        Streaming access is less likely to persit from cache.


    .. autoattribute:: cuda.bindings.driver.CUaccessProperty.CU_ACCESS_PROPERTY_PERSISTING


        Persisting access is more likely to persist in cache.

.. autoclass:: cuda.bindings.driver.CUgraphConditionalNodeType

    .. autoattribute:: cuda.bindings.driver.CUgraphConditionalNodeType.CU_GRAPH_COND_TYPE_IF


        Conditional 'if/else' Node. Body[0] executed if condition is non-zero. If `size` == 2, an optional ELSE graph is created and this is executed if the condition is zero.


    .. autoattribute:: cuda.bindings.driver.CUgraphConditionalNodeType.CU_GRAPH_COND_TYPE_WHILE


        Conditional 'while' Node. Body executed repeatedly while condition value is non-zero.


    .. autoattribute:: cuda.bindings.driver.CUgraphConditionalNodeType.CU_GRAPH_COND_TYPE_SWITCH


        Conditional 'switch' Node. Body[n] is executed once, where 'n' is the value of the condition. If the condition does not match a body index, no body is launched.

.. autoclass:: cuda.bindings.driver.CUgraphNodeType

    .. autoattribute:: cuda.bindings.driver.CUgraphNodeType.CU_GRAPH_NODE_TYPE_KERNEL


        GPU kernel node


    .. autoattribute:: cuda.bindings.driver.CUgraphNodeType.CU_GRAPH_NODE_TYPE_MEMCPY


        Memcpy node


    .. autoattribute:: cuda.bindings.driver.CUgraphNodeType.CU_GRAPH_NODE_TYPE_MEMSET


        Memset node


    .. autoattribute:: cuda.bindings.driver.CUgraphNodeType.CU_GRAPH_NODE_TYPE_HOST


        Host (executable) node


    .. autoattribute:: cuda.bindings.driver.CUgraphNodeType.CU_GRAPH_NODE_TYPE_GRAPH


        Node which executes an embedded graph


    .. autoattribute:: cuda.bindings.driver.CUgraphNodeType.CU_GRAPH_NODE_TYPE_EMPTY


        Empty (no-op) node


    .. autoattribute:: cuda.bindings.driver.CUgraphNodeType.CU_GRAPH_NODE_TYPE_WAIT_EVENT


        External event wait node


    .. autoattribute:: cuda.bindings.driver.CUgraphNodeType.CU_GRAPH_NODE_TYPE_EVENT_RECORD


        External event record node


    .. autoattribute:: cuda.bindings.driver.CUgraphNodeType.CU_GRAPH_NODE_TYPE_EXT_SEMAS_SIGNAL


        External semaphore signal node


    .. autoattribute:: cuda.bindings.driver.CUgraphNodeType.CU_GRAPH_NODE_TYPE_EXT_SEMAS_WAIT


        External semaphore wait node


    .. autoattribute:: cuda.bindings.driver.CUgraphNodeType.CU_GRAPH_NODE_TYPE_MEM_ALLOC


        Memory Allocation Node


    .. autoattribute:: cuda.bindings.driver.CUgraphNodeType.CU_GRAPH_NODE_TYPE_MEM_FREE


        Memory Free Node


    .. autoattribute:: cuda.bindings.driver.CUgraphNodeType.CU_GRAPH_NODE_TYPE_BATCH_MEM_OP


        Batch MemOp Node See :py:obj:`~.cuStreamBatchMemOp` and :py:obj:`~.CUstreamBatchMemOpType` for what these nodes can do.


    .. autoattribute:: cuda.bindings.driver.CUgraphNodeType.CU_GRAPH_NODE_TYPE_CONDITIONAL


        Conditional Node                                         May be used to implement a conditional execution path or loop

                                                inside of a graph. The graph(s) contained within the body of the conditional node

                                                can be selectively executed or iterated upon based on the value of a conditional

                                                variable.



                                                Handles must be created in advance of creating the node

                                                using :py:obj:`~.cuGraphConditionalHandleCreate`.



                                                The following restrictions apply to graphs which contain conditional nodes:

                                                 The graph cannot be used in a child node.

                                                 Only one instantiation of the graph may exist at any point in time.

                                                 The graph cannot be cloned.



                                                To set the control value, supply a default value when creating the handle and/or

                                                call :py:obj:`~.cudaGraphSetConditional` from device code.

.. autoclass:: cuda.bindings.driver.CUgraphDependencyType

    .. autoattribute:: cuda.bindings.driver.CUgraphDependencyType.CU_GRAPH_DEPENDENCY_TYPE_DEFAULT


        This is an ordinary dependency.


    .. autoattribute:: cuda.bindings.driver.CUgraphDependencyType.CU_GRAPH_DEPENDENCY_TYPE_PROGRAMMATIC


        This dependency type allows the downstream node to use `cudaGridDependencySynchronize()`. It may only be used between kernel nodes, and must be used with either the :py:obj:`~.CU_GRAPH_KERNEL_NODE_PORT_PROGRAMMATIC` or :py:obj:`~.CU_GRAPH_KERNEL_NODE_PORT_LAUNCH_ORDER` outgoing port.

.. autoclass:: cuda.bindings.driver.CUgraphInstantiateResult

    .. autoattribute:: cuda.bindings.driver.CUgraphInstantiateResult.CUDA_GRAPH_INSTANTIATE_SUCCESS


        Instantiation succeeded


    .. autoattribute:: cuda.bindings.driver.CUgraphInstantiateResult.CUDA_GRAPH_INSTANTIATE_ERROR


        Instantiation failed for an unexpected reason which is described in the return value of the function


    .. autoattribute:: cuda.bindings.driver.CUgraphInstantiateResult.CUDA_GRAPH_INSTANTIATE_INVALID_STRUCTURE


        Instantiation failed due to invalid structure, such as cycles


    .. autoattribute:: cuda.bindings.driver.CUgraphInstantiateResult.CUDA_GRAPH_INSTANTIATE_NODE_OPERATION_NOT_SUPPORTED


        Instantiation for device launch failed because the graph contained an unsupported operation


    .. autoattribute:: cuda.bindings.driver.CUgraphInstantiateResult.CUDA_GRAPH_INSTANTIATE_MULTIPLE_CTXS_NOT_SUPPORTED


        Instantiation for device launch failed due to the nodes belonging to different contexts


    .. autoattribute:: cuda.bindings.driver.CUgraphInstantiateResult.CUDA_GRAPH_INSTANTIATE_CONDITIONAL_HANDLE_UNUSED


        One or more conditional handles are not associated with conditional nodes

.. autoclass:: cuda.bindings.driver.CUsynchronizationPolicy

    .. autoattribute:: cuda.bindings.driver.CUsynchronizationPolicy.CU_SYNC_POLICY_AUTO


    .. autoattribute:: cuda.bindings.driver.CUsynchronizationPolicy.CU_SYNC_POLICY_SPIN


    .. autoattribute:: cuda.bindings.driver.CUsynchronizationPolicy.CU_SYNC_POLICY_YIELD


    .. autoattribute:: cuda.bindings.driver.CUsynchronizationPolicy.CU_SYNC_POLICY_BLOCKING_SYNC

.. autoclass:: cuda.bindings.driver.CUclusterSchedulingPolicy

    .. autoattribute:: cuda.bindings.driver.CUclusterSchedulingPolicy.CU_CLUSTER_SCHEDULING_POLICY_DEFAULT


        the default policy


    .. autoattribute:: cuda.bindings.driver.CUclusterSchedulingPolicy.CU_CLUSTER_SCHEDULING_POLICY_SPREAD


        spread the blocks within a cluster to the SMs


    .. autoattribute:: cuda.bindings.driver.CUclusterSchedulingPolicy.CU_CLUSTER_SCHEDULING_POLICY_LOAD_BALANCING


        allow the hardware to load-balance the blocks in a cluster to the SMs

.. autoclass:: cuda.bindings.driver.CUlaunchMemSyncDomain

    .. autoattribute:: cuda.bindings.driver.CUlaunchMemSyncDomain.CU_LAUNCH_MEM_SYNC_DOMAIN_DEFAULT


        Launch kernels in the default domain


    .. autoattribute:: cuda.bindings.driver.CUlaunchMemSyncDomain.CU_LAUNCH_MEM_SYNC_DOMAIN_REMOTE


        Launch kernels in the remote domain

.. autoclass:: cuda.bindings.driver.CUlaunchAttributeID

    .. autoattribute:: cuda.bindings.driver.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_IGNORE


        Ignored entry, for convenient composition


    .. autoattribute:: cuda.bindings.driver.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_ACCESS_POLICY_WINDOW


        Valid for streams, graph nodes, launches. See :py:obj:`~.CUlaunchAttributeValue.accessPolicyWindow`.


    .. autoattribute:: cuda.bindings.driver.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_COOPERATIVE


        Valid for graph nodes, launches. See :py:obj:`~.CUlaunchAttributeValue.cooperative`.


    .. autoattribute:: cuda.bindings.driver.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_SYNCHRONIZATION_POLICY


        Valid for streams. See :py:obj:`~.CUlaunchAttributeValue.syncPolicy`.


    .. autoattribute:: cuda.bindings.driver.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION


        Valid for graph nodes, launches. See :py:obj:`~.CUlaunchAttributeValue.clusterDim`.


    .. autoattribute:: cuda.bindings.driver.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE


        Valid for graph nodes, launches. See :py:obj:`~.CUlaunchAttributeValue.clusterSchedulingPolicyPreference`.


    .. autoattribute:: cuda.bindings.driver.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION


        Valid for launches. Setting :py:obj:`~.CUlaunchAttributeValue.programmaticStreamSerializationAllowed` to non-0 signals that the kernel will use programmatic means to resolve its stream dependency, so that the CUDA runtime should opportunistically allow the grid's execution to overlap with the previous kernel in the stream, if that kernel requests the overlap. The dependent launches can choose to wait on the dependency using the programmatic sync (cudaGridDependencySynchronize() or equivalent PTX instructions).


    .. autoattribute:: cuda.bindings.driver.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_EVENT


        Valid for launches. Set :py:obj:`~.CUlaunchAttributeValue.programmaticEvent` to record the event. Event recorded through this launch attribute is guaranteed to only trigger after all block in the associated kernel trigger the event. A block can trigger the event through PTX launchdep.release or CUDA builtin function cudaTriggerProgrammaticLaunchCompletion(). A trigger can also be inserted at the beginning of each block's execution if triggerAtBlockStart is set to non-0. The dependent launches can choose to wait on the dependency using the programmatic sync (cudaGridDependencySynchronize() or equivalent PTX instructions). Note that dependents (including the CPU thread calling :py:obj:`~.cuEventSynchronize()`) are not guaranteed to observe the release precisely when it is released. For example, :py:obj:`~.cuEventSynchronize()` may only observe the event trigger long after the associated kernel has completed. This recording type is primarily meant for establishing programmatic dependency between device tasks. Note also this type of dependency allows, but does not guarantee, concurrent execution of tasks. 

         The event supplied must not be an interprocess or interop event. The event must disable timing (i.e. must be created with the :py:obj:`~.CU_EVENT_DISABLE_TIMING` flag set).


    .. autoattribute:: cuda.bindings.driver.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_PRIORITY


        Valid for streams, graph nodes, launches. See :py:obj:`~.CUlaunchAttributeValue.priority`.


    .. autoattribute:: cuda.bindings.driver.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_MEM_SYNC_DOMAIN_MAP


        Valid for streams, graph nodes, launches. See :py:obj:`~.CUlaunchAttributeValue.memSyncDomainMap`.


    .. autoattribute:: cuda.bindings.driver.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_MEM_SYNC_DOMAIN


        Valid for streams, graph nodes, launches. See :py:obj:`~.CUlaunchAttributeValue.memSyncDomain`.


    .. autoattribute:: cuda.bindings.driver.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_PREFERRED_CLUSTER_DIMENSION


        Valid for graph nodes, launches. Set :py:obj:`~.CUlaunchAttributeValue.preferredClusterDim` to allow the kernel launch to specify a preferred substitute cluster dimension. Blocks may be grouped according to either the dimensions specified with this attribute (grouped into a "preferred substitute cluster"), or the one specified with :py:obj:`~.CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION` attribute (grouped into a "regular cluster"). The cluster dimensions of a "preferred substitute cluster" shall be an integer multiple greater than zero of the regular cluster dimensions. The device will attempt - on a best-effort basis - to group thread blocks into preferred clusters over grouping them into regular clusters. When it deems necessary (primarily when the device temporarily runs out of physical resources to launch the larger preferred clusters), the device may switch to launch the regular clusters instead to attempt to utilize as much of the physical device resources as possible. 

         Each type of cluster will have its enumeration / coordinate setup as if the grid consists solely of its type of cluster. For example, if the preferred substitute cluster dimensions double the regular cluster dimensions, there might be simultaneously a regular cluster indexed at (1,0,0), and a preferred cluster indexed at (1,0,0). In this example, the preferred substitute cluster (1,0,0) replaces regular clusters (2,0,0) and (3,0,0) and groups their blocks. 

         This attribute will only take effect when a regular cluster dimension has been specified. The preferred substitute cluster dimension must be an integer multiple greater than zero of the regular cluster dimension and must divide the grid. It must also be no more than `maxBlocksPerCluster`, if it is set in the kernel's `__launch_bounds__`. Otherwise it must be less than the maximum value the driver can support. Otherwise, setting this attribute to a value physically unable to fit on any particular device is permitted.


    .. autoattribute:: cuda.bindings.driver.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_LAUNCH_COMPLETION_EVENT


        Valid for launches. Set :py:obj:`~.CUlaunchAttributeValue.launchCompletionEvent` to record the event. 

         Nominally, the event is triggered once all blocks of the kernel have begun execution. Currently this is a best effort. If a kernel B has a launch completion dependency on a kernel A, B may wait until A is complete. Alternatively, blocks of B may begin before all blocks of A have begun, for example if B can claim execution resources unavailable to A (e.g. they run on different GPUs) or if B is a higher priority than A. Exercise caution if such an ordering inversion could lead to deadlock. 

         A launch completion event is nominally similar to a programmatic event with `triggerAtBlockStart` set except that it is not visible to `cudaGridDependencySynchronize()` and can be used with compute capability less than 9.0. 

         The event supplied must not be an interprocess or interop event. The event must disable timing (i.e. must be created with the :py:obj:`~.CU_EVENT_DISABLE_TIMING` flag set).


    .. autoattribute:: cuda.bindings.driver.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_DEVICE_UPDATABLE_KERNEL_NODE


        Valid for graph nodes, launches. This attribute is graphs-only, and passing it to a launch in a non-capturing stream will result in an error. 

         :py:obj:`~.CUlaunchAttributeValue`::deviceUpdatableKernelNode::deviceUpdatable can only be set to 0 or 1. Setting the field to 1 indicates that the corresponding kernel node should be device-updatable. On success, a handle will be returned via :py:obj:`~.CUlaunchAttributeValue`::deviceUpdatableKernelNode::devNode which can be passed to the various device-side update functions to update the node's kernel parameters from within another kernel. For more information on the types of device updates that can be made, as well as the relevant limitations thereof, see :py:obj:`~.cudaGraphKernelNodeUpdatesApply`. 

         Nodes which are device-updatable have additional restrictions compared to regular kernel nodes. Firstly, device-updatable nodes cannot be removed from their graph via :py:obj:`~.cuGraphDestroyNode`. Additionally, once opted-in to this functionality, a node cannot opt out, and any attempt to set the deviceUpdatable attribute to 0 will result in an error. Device-updatable kernel nodes also cannot have their attributes copied to/from another kernel node via :py:obj:`~.cuGraphKernelNodeCopyAttributes`. Graphs containing one or more device-updatable nodes also do not allow multiple instantiation, and neither the graph nor its instantiated version can be passed to :py:obj:`~.cuGraphExecUpdate`. 

         If a graph contains device-updatable nodes and updates those nodes from the device from within the graph, the graph must be uploaded with :py:obj:`~.cuGraphUpload` before it is launched. For such a graph, if host-side executable graph updates are made to the device-updatable nodes, the graph must be uploaded before it is launched again.


    .. autoattribute:: cuda.bindings.driver.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT


        Valid for launches. On devices where the L1 cache and shared memory use the same hardware resources, setting :py:obj:`~.CUlaunchAttributeValue.sharedMemCarveout` to a percentage between 0-100 signals the CUDA driver to set the shared memory carveout preference, in percent of the total shared memory for that kernel launch. This attribute takes precedence over :py:obj:`~.CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT`. This is only a hint, and the CUDA driver can choose a different configuration if required for the launch.


    .. autoattribute:: cuda.bindings.driver.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_NVLINK_UTIL_CENTRIC_SCHEDULING


        Valid for streams, graph nodes, launches. This attribute is a hint to the CUDA runtime that the launch should attempt to make the kernel maximize its NVLINK utilization. 



         When possible to honor this hint, CUDA will assume each block in the grid launch will carry out an even amount of NVLINK traffic, and make a best-effort attempt to adjust the kernel launch based on that assumption. 

         This attribute is a hint only. CUDA makes no functional or performance guarantee. Its applicability can be affected by many different factors, including driver version (i.e. CUDA doesn't guarantee the performance characteristics will be maintained between driver versions or a driver update could alter or regress previously observed perf characteristics.) It also doesn't guarantee a successful result, i.e. applying the attribute may not improve the performance of either the targeted kernel or the encapsulating application. 

         Valid values for :py:obj:`~.CUlaunchAttributeValue`::nvlinkUtilCentricScheduling are 0 (disabled) and 1 (enabled).

.. autoclass:: cuda.bindings.driver.CUstreamCaptureStatus

    .. autoattribute:: cuda.bindings.driver.CUstreamCaptureStatus.CU_STREAM_CAPTURE_STATUS_NONE


        Stream is not capturing


    .. autoattribute:: cuda.bindings.driver.CUstreamCaptureStatus.CU_STREAM_CAPTURE_STATUS_ACTIVE


        Stream is actively capturing


    .. autoattribute:: cuda.bindings.driver.CUstreamCaptureStatus.CU_STREAM_CAPTURE_STATUS_INVALIDATED


        Stream is part of a capture sequence that has been invalidated, but not terminated

.. autoclass:: cuda.bindings.driver.CUstreamCaptureMode

    .. autoattribute:: cuda.bindings.driver.CUstreamCaptureMode.CU_STREAM_CAPTURE_MODE_GLOBAL


    .. autoattribute:: cuda.bindings.driver.CUstreamCaptureMode.CU_STREAM_CAPTURE_MODE_THREAD_LOCAL


    .. autoattribute:: cuda.bindings.driver.CUstreamCaptureMode.CU_STREAM_CAPTURE_MODE_RELAXED

.. autoclass:: cuda.bindings.driver.CUdriverProcAddress_flags

    .. autoattribute:: cuda.bindings.driver.CUdriverProcAddress_flags.CU_GET_PROC_ADDRESS_DEFAULT


        Default search mode for driver symbols.


    .. autoattribute:: cuda.bindings.driver.CUdriverProcAddress_flags.CU_GET_PROC_ADDRESS_LEGACY_STREAM


        Search for legacy versions of driver symbols.


    .. autoattribute:: cuda.bindings.driver.CUdriverProcAddress_flags.CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM


        Search for per-thread versions of driver symbols.

.. autoclass:: cuda.bindings.driver.CUdriverProcAddressQueryResult

    .. autoattribute:: cuda.bindings.driver.CUdriverProcAddressQueryResult.CU_GET_PROC_ADDRESS_SUCCESS


        Symbol was succesfully found


    .. autoattribute:: cuda.bindings.driver.CUdriverProcAddressQueryResult.CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND


        Symbol was not found in search


    .. autoattribute:: cuda.bindings.driver.CUdriverProcAddressQueryResult.CU_GET_PROC_ADDRESS_VERSION_NOT_SUFFICIENT


        Symbol was found but version supplied was not sufficient

.. autoclass:: cuda.bindings.driver.CUexecAffinityType

    .. autoattribute:: cuda.bindings.driver.CUexecAffinityType.CU_EXEC_AFFINITY_TYPE_SM_COUNT


        Create a context with limited SMs.


    .. autoattribute:: cuda.bindings.driver.CUexecAffinityType.CU_EXEC_AFFINITY_TYPE_MAX

.. autoclass:: cuda.bindings.driver.CUcigDataType

    .. autoattribute:: cuda.bindings.driver.CUcigDataType.CIG_DATA_TYPE_D3D12_COMMAND_QUEUE


    .. autoattribute:: cuda.bindings.driver.CUcigDataType.CIG_DATA_TYPE_NV_BLOB


        D3D12 Command Queue Handle

.. autoclass:: cuda.bindings.driver.CUlibraryOption

    .. autoattribute:: cuda.bindings.driver.CUlibraryOption.CU_LIBRARY_HOST_UNIVERSAL_FUNCTION_AND_DATA_TABLE


    .. autoattribute:: cuda.bindings.driver.CUlibraryOption.CU_LIBRARY_BINARY_IS_PRESERVED


        Specifes that the argument `code` passed to :py:obj:`~.cuLibraryLoadData()` will be preserved. Specifying this option will let the driver know that `code` can be accessed at any point until :py:obj:`~.cuLibraryUnload()`. The default behavior is for the driver to allocate and maintain its own copy of `code`. Note that this is only a memory usage optimization hint and the driver can choose to ignore it if required. Specifying this option with :py:obj:`~.cuLibraryLoadFromFile()` is invalid and will return :py:obj:`~.CUDA_ERROR_INVALID_VALUE`.


    .. autoattribute:: cuda.bindings.driver.CUlibraryOption.CU_LIBRARY_NUM_OPTIONS

.. autoclass:: cuda.bindings.driver.CUresult

    .. autoattribute:: cuda.bindings.driver.CUresult.CUDA_SUCCESS


        The API call returned with no errors. In the case of query calls, this also means that the operation being queried is complete (see :py:obj:`~.cuEventQuery()` and :py:obj:`~.cuStreamQuery()`).


    .. autoattribute:: cuda.bindings.driver.CUresult.CUDA_ERROR_INVALID_VALUE


        This indicates that one or more of the parameters passed to the API call is not within an acceptable range of values.


    .. autoattribute:: cuda.bindings.driver.CUresult.CUDA_ERROR_OUT_OF_MEMORY


        The API call failed because it was unable to allocate enough memory or other resources to perform the requested operation.


    .. autoattribute:: cuda.bindings.driver.CUresult.CUDA_ERROR_NOT_INITIALIZED


        This indicates that the CUDA driver has not been initialized with :py:obj:`~.cuInit()` or that initialization has failed.


    .. autoattribute:: cuda.bindings.driver.CUresult.CUDA_ERROR_DEINITIALIZED


        This indicates that the CUDA driver is in the process of shutting down.


    .. autoattribute:: cuda.bindings.driver.CUresult.CUDA_ERROR_PROFILER_DISABLED


        This indicates profiler is not initialized for this run. This can happen when the application is running with external profiling tools like visual profiler.


    .. autoattribute:: cuda.bindings.driver.CUresult.CUDA_ERROR_PROFILER_NOT_INITIALIZED


        [Deprecated]


    .. autoattribute:: cuda.bindings.driver.CUresult.CUDA_ERROR_PROFILER_ALREADY_STARTED


        [Deprecated]


    .. autoattribute:: cuda.bindings.driver.CUresult.CUDA_ERROR_PROFILER_ALREADY_STOPPED


        [Deprecated]


    .. autoattribute:: cuda.bindings.driver.CUresult.CUDA_ERROR_STUB_LIBRARY


        This indicates that the CUDA driver that the application has loaded is a stub library. Applications that run with the stub rather than a real driver loaded will result in CUDA API returning this error.


    .. autoattribute:: cuda.bindings.driver.CUresult.CUDA_ERROR_CALL_REQUIRES_NEWER_DRIVER


        This indicates that the API call requires a newer CUDA driver than the one currently installed. Users should install an updated NVIDIA CUDA driver to allow the API call to succeed.


    .. autoattribute:: cuda.bindings.driver.CUresult.CUDA_ERROR_DEVICE_UNAVAILABLE


        This indicates that requested CUDA device is unavailable at the current time. Devices are often unavailable due to use of :py:obj:`~.CU_COMPUTEMODE_EXCLUSIVE_PROCESS` or :py:obj:`~.CU_COMPUTEMODE_PROHIBITED`.


    .. autoattribute:: cuda.bindings.driver.CUresult.CUDA_ERROR_NO_DEVICE


        This indicates that no CUDA-capable devices were detected by the installed CUDA driver.


    .. autoattribute:: cuda.bindings.driver.CUresult.CUDA_ERROR_INVALID_DEVICE


        This indicates that the device ordinal supplied by the user does not correspond to a valid CUDA device or that the action requested is invalid for the specified device.


    .. autoattribute:: cuda.bindings.driver.CUresult.CUDA_ERROR_DEVICE_NOT_LICENSED


        This error indicates that the Grid license is not applied.


    .. autoattribute:: cuda.bindings.driver.CUresult.CUDA_ERROR_INVALID_IMAGE


        This indicates that the device kernel image is invalid. This can also indicate an invalid CUDA module.


    .. autoattribute:: cuda.bindings.driver.CUresult.CUDA_ERROR_INVALID_CONTEXT


        This most frequently indicates that there is no context bound to the current thread. This can also be returned if the context passed to an API call is not a valid handle (such as a context that has had :py:obj:`~.cuCtxDestroy()` invoked on it). This can also be returned if a user mixes different API versions (i.e. 3010 context with 3020 API calls). See :py:obj:`~.cuCtxGetApiVersion()` for more details. This can also be returned if the green context passed to an API call was not converted to a :py:obj:`~.CUcontext` using :py:obj:`~.cuCtxFromGreenCtx` API.


    .. autoattribute:: cuda.bindings.driver.CUresult.CUDA_ERROR_CONTEXT_ALREADY_CURRENT


        This indicated that the context being supplied as a parameter to the API call was already the active context. [Deprecated]


    .. autoattribute:: cuda.bindings.driver.CUresult.CUDA_ERROR_MAP_FAILED


        This indicates that a map or register operation has failed.


    .. autoattribute:: cuda.bindings.driver.CUresult.CUDA_ERROR_UNMAP_FAILED


        This indicates that an unmap or unregister operation has failed.


    .. autoattribute:: cuda.bindings.driver.CUresult.CUDA_ERROR_ARRAY_IS_MAPPED


        This indicates that the specified array is currently mapped and thus cannot be destroyed.


    .. autoattribute:: cuda.bindings.driver.CUresult.CUDA_ERROR_ALREADY_MAPPED


        This indicates that the resource is already mapped.


    .. autoattribute:: cuda.bindings.driver.CUresult.CUDA_ERROR_NO_BINARY_FOR_GPU


        This indicates that there is no kernel image available that is suitable for the device. This can occur when a user specifies code generation options for a particular CUDA source file that do not include the corresponding device configuration.


    .. autoattribute:: cuda.bindings.driver.CUresult.CUDA_ERROR_ALREADY_ACQUIRED


        This indicates that a resource has already been acquired.


    .. autoattribute:: cuda.bindings.driver.CUresult.CUDA_ERROR_NOT_MAPPED


        This indicates that a resource is not mapped.


    .. autoattribute:: cuda.bindings.driver.CUresult.CUDA_ERROR_NOT_MAPPED_AS_ARRAY


        This indicates that a mapped resource is not available for access as an array.


    .. autoattribute:: cuda.bindings.driver.CUresult.CUDA_ERROR_NOT_MAPPED_AS_POINTER


        This indicates that a mapped resource is not available for access as a pointer.


    .. autoattribute:: cuda.bindings.driver.CUresult.CUDA_ERROR_ECC_UNCORRECTABLE


        This indicates that an uncorrectable ECC error was detected during execution.


    .. autoattribute:: cuda.bindings.driver.CUresult.CUDA_ERROR_UNSUPPORTED_LIMIT


        This indicates that the :py:obj:`~.CUlimit` passed to the API call is not supported by the active device.


    .. autoattribute:: cuda.bindings.driver.CUresult.CUDA_ERROR_CONTEXT_ALREADY_IN_USE


        This indicates that the :py:obj:`~.CUcontext` passed to the API call can only be bound to a single CPU thread at a time but is already bound to a CPU thread.


    .. autoattribute:: cuda.bindings.driver.CUresult.CUDA_ERROR_PEER_ACCESS_UNSUPPORTED


        This indicates that peer access is not supported across the given devices.


    .. autoattribute:: cuda.bindings.driver.CUresult.CUDA_ERROR_INVALID_PTX


        This indicates that a PTX JIT compilation failed.


    .. autoattribute:: cuda.bindings.driver.CUresult.CUDA_ERROR_INVALID_GRAPHICS_CONTEXT


        This indicates an error with OpenGL or DirectX context.


    .. autoattribute:: cuda.bindings.driver.CUresult.CUDA_ERROR_NVLINK_UNCORRECTABLE


        This indicates that an uncorrectable NVLink error was detected during the execution.


    .. autoattribute:: cuda.bindings.driver.CUresult.CUDA_ERROR_JIT_COMPILER_NOT_FOUND


        This indicates that the PTX JIT compiler library was not found.


    .. autoattribute:: cuda.bindings.driver.CUresult.CUDA_ERROR_UNSUPPORTED_PTX_VERSION


        This indicates that the provided PTX was compiled with an unsupported toolchain.


    .. autoattribute:: cuda.bindings.driver.CUresult.CUDA_ERROR_JIT_COMPILATION_DISABLED


        This indicates that the PTX JIT compilation was disabled.


    .. autoattribute:: cuda.bindings.driver.CUresult.CUDA_ERROR_UNSUPPORTED_EXEC_AFFINITY


        This indicates that the :py:obj:`~.CUexecAffinityType` passed to the API call is not supported by the active device.


    .. autoattribute:: cuda.bindings.driver.CUresult.CUDA_ERROR_UNSUPPORTED_DEVSIDE_SYNC


        This indicates that the code to be compiled by the PTX JIT contains unsupported call to cudaDeviceSynchronize.


    .. autoattribute:: cuda.bindings.driver.CUresult.CUDA_ERROR_CONTAINED


        This indicates that an exception occurred on the device that is now contained by the GPU's error containment capability. Common causes are - a. Certain types of invalid accesses of peer GPU memory over nvlink b. Certain classes of hardware errors This leaves the process in an inconsistent state and any further CUDA work will return the same error. To continue using CUDA, the process must be terminated and relaunched.


    .. autoattribute:: cuda.bindings.driver.CUresult.CUDA_ERROR_INVALID_SOURCE


        This indicates that the device kernel source is invalid. This includes compilation/linker errors encountered in device code or user error.


    .. autoattribute:: cuda.bindings.driver.CUresult.CUDA_ERROR_FILE_NOT_FOUND


        This indicates that the file specified was not found.


    .. autoattribute:: cuda.bindings.driver.CUresult.CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND


        This indicates that a link to a shared object failed to resolve.


    .. autoattribute:: cuda.bindings.driver.CUresult.CUDA_ERROR_SHARED_OBJECT_INIT_FAILED


        This indicates that initialization of a shared object failed.


    .. autoattribute:: cuda.bindings.driver.CUresult.CUDA_ERROR_OPERATING_SYSTEM


        This indicates that an OS call failed.


    .. autoattribute:: cuda.bindings.driver.CUresult.CUDA_ERROR_INVALID_HANDLE


        This indicates that a resource handle passed to the API call was not valid. Resource handles are opaque types like :py:obj:`~.CUstream` and :py:obj:`~.CUevent`.


    .. autoattribute:: cuda.bindings.driver.CUresult.CUDA_ERROR_ILLEGAL_STATE


        This indicates that a resource required by the API call is not in a valid state to perform the requested operation.


    .. autoattribute:: cuda.bindings.driver.CUresult.CUDA_ERROR_LOSSY_QUERY


        This indicates an attempt was made to introspect an object in a way that would discard semantically important information. This is either due to the object using funtionality newer than the API version used to introspect it or omission of optional return arguments.


    .. autoattribute:: cuda.bindings.driver.CUresult.CUDA_ERROR_NOT_FOUND


        This indicates that a named symbol was not found. Examples of symbols are global/constant variable names, driver function names, texture names, and surface names.


    .. autoattribute:: cuda.bindings.driver.CUresult.CUDA_ERROR_NOT_READY


        This indicates that asynchronous operations issued previously have not completed yet. This result is not actually an error, but must be indicated differently than :py:obj:`~.CUDA_SUCCESS` (which indicates completion). Calls that may return this value include :py:obj:`~.cuEventQuery()` and :py:obj:`~.cuStreamQuery()`.


    .. autoattribute:: cuda.bindings.driver.CUresult.CUDA_ERROR_ILLEGAL_ADDRESS


        While executing a kernel, the device encountered a load or store instruction on an invalid memory address. This leaves the process in an inconsistent state and any further CUDA work will return the same error. To continue using CUDA, the process must be terminated and relaunched.


    .. autoattribute:: cuda.bindings.driver.CUresult.CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES


        This indicates that a launch did not occur because it did not have appropriate resources. This error usually indicates that the user has attempted to pass too many arguments to the device kernel, or the kernel launch specifies too many threads for the kernel's register count. Passing arguments of the wrong size (i.e. a 64-bit pointer when a 32-bit int is expected) is equivalent to passing too many arguments and can also result in this error.


    .. autoattribute:: cuda.bindings.driver.CUresult.CUDA_ERROR_LAUNCH_TIMEOUT


        This indicates that the device kernel took too long to execute. This can only occur if timeouts are enabled - see the device attribute :py:obj:`~.CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT` for more information. This leaves the process in an inconsistent state and any further CUDA work will return the same error. To continue using CUDA, the process must be terminated and relaunched.


    .. autoattribute:: cuda.bindings.driver.CUresult.CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING


        This error indicates a kernel launch that uses an incompatible texturing mode.


    .. autoattribute:: cuda.bindings.driver.CUresult.CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED


        This error indicates that a call to :py:obj:`~.cuCtxEnablePeerAccess()` is trying to re-enable peer access to a context which has already had peer access to it enabled.


    .. autoattribute:: cuda.bindings.driver.CUresult.CUDA_ERROR_PEER_ACCESS_NOT_ENABLED


        This error indicates that :py:obj:`~.cuCtxDisablePeerAccess()` is trying to disable peer access which has not been enabled yet via :py:obj:`~.cuCtxEnablePeerAccess()`.


    .. autoattribute:: cuda.bindings.driver.CUresult.CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE


        This error indicates that the primary context for the specified device has already been initialized.


    .. autoattribute:: cuda.bindings.driver.CUresult.CUDA_ERROR_CONTEXT_IS_DESTROYED


        This error indicates that the context current to the calling thread has been destroyed using :py:obj:`~.cuCtxDestroy`, or is a primary context which has not yet been initialized.


    .. autoattribute:: cuda.bindings.driver.CUresult.CUDA_ERROR_ASSERT


        A device-side assert triggered during kernel execution. The context cannot be used anymore, and must be destroyed. All existing device memory allocations from this context are invalid and must be reconstructed if the program is to continue using CUDA.


    .. autoattribute:: cuda.bindings.driver.CUresult.CUDA_ERROR_TOO_MANY_PEERS


        This error indicates that the hardware resources required to enable peer access have been exhausted for one or more of the devices passed to :py:obj:`~.cuCtxEnablePeerAccess()`.


    .. autoattribute:: cuda.bindings.driver.CUresult.CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED


        This error indicates that the memory range passed to :py:obj:`~.cuMemHostRegister()` has already been registered.


    .. autoattribute:: cuda.bindings.driver.CUresult.CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED


        This error indicates that the pointer passed to :py:obj:`~.cuMemHostUnregister()` does not correspond to any currently registered memory region.


    .. autoattribute:: cuda.bindings.driver.CUresult.CUDA_ERROR_HARDWARE_STACK_ERROR


        While executing a kernel, the device encountered a stack error. This can be due to stack corruption or exceeding the stack size limit. This leaves the process in an inconsistent state and any further CUDA work will return the same error. To continue using CUDA, the process must be terminated and relaunched.


    .. autoattribute:: cuda.bindings.driver.CUresult.CUDA_ERROR_ILLEGAL_INSTRUCTION


        While executing a kernel, the device encountered an illegal instruction. This leaves the process in an inconsistent state and any further CUDA work will return the same error. To continue using CUDA, the process must be terminated and relaunched.


    .. autoattribute:: cuda.bindings.driver.CUresult.CUDA_ERROR_MISALIGNED_ADDRESS


        While executing a kernel, the device encountered a load or store instruction on a memory address which is not aligned. This leaves the process in an inconsistent state and any further CUDA work will return the same error. To continue using CUDA, the process must be terminated and relaunched.


    .. autoattribute:: cuda.bindings.driver.CUresult.CUDA_ERROR_INVALID_ADDRESS_SPACE


        While executing a kernel, the device encountered an instruction which can only operate on memory locations in certain address spaces (global, shared, or local), but was supplied a memory address not belonging to an allowed address space. This leaves the process in an inconsistent state and any further CUDA work will return the same error. To continue using CUDA, the process must be terminated and relaunched.


    .. autoattribute:: cuda.bindings.driver.CUresult.CUDA_ERROR_INVALID_PC


        While executing a kernel, the device program counter wrapped its address space. This leaves the process in an inconsistent state and any further CUDA work will return the same error. To continue using CUDA, the process must be terminated and relaunched.


    .. autoattribute:: cuda.bindings.driver.CUresult.CUDA_ERROR_LAUNCH_FAILED


        An exception occurred on the device while executing a kernel. Common causes include dereferencing an invalid device pointer and accessing out of bounds shared memory. Less common cases can be system specific - more information about these cases can be found in the system specific user guide. This leaves the process in an inconsistent state and any further CUDA work will return the same error. To continue using CUDA, the process must be terminated and relaunched.


    .. autoattribute:: cuda.bindings.driver.CUresult.CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE


        This error indicates that the number of blocks launched per grid for a kernel that was launched via either :py:obj:`~.cuLaunchCooperativeKernel` or :py:obj:`~.cuLaunchCooperativeKernelMultiDevice` exceeds the maximum number of blocks as allowed by :py:obj:`~.cuOccupancyMaxActiveBlocksPerMultiprocessor` or :py:obj:`~.cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags` times the number of multiprocessors as specified by the device attribute :py:obj:`~.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT`.


    .. autoattribute:: cuda.bindings.driver.CUresult.CUDA_ERROR_TENSOR_MEMORY_LEAK


        An exception occurred on the device while exiting a kernel using tensor memory: the tensor memory was not completely deallocated. This leaves the process in an inconsistent state and any further CUDA work will return the same error. To continue using CUDA, the process must be terminated and relaunched.


    .. autoattribute:: cuda.bindings.driver.CUresult.CUDA_ERROR_NOT_PERMITTED


        This error indicates that the attempted operation is not permitted.


    .. autoattribute:: cuda.bindings.driver.CUresult.CUDA_ERROR_NOT_SUPPORTED


        This error indicates that the attempted operation is not supported on the current system or device.


    .. autoattribute:: cuda.bindings.driver.CUresult.CUDA_ERROR_SYSTEM_NOT_READY


        This error indicates that the system is not yet ready to start any CUDA work. To continue using CUDA, verify the system configuration is in a valid state and all required driver daemons are actively running. More information about this error can be found in the system specific user guide.


    .. autoattribute:: cuda.bindings.driver.CUresult.CUDA_ERROR_SYSTEM_DRIVER_MISMATCH


        This error indicates that there is a mismatch between the versions of the display driver and the CUDA driver. Refer to the compatibility documentation for supported versions.


    .. autoattribute:: cuda.bindings.driver.CUresult.CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE


        This error indicates that the system was upgraded to run with forward compatibility but the visible hardware detected by CUDA does not support this configuration. Refer to the compatibility documentation for the supported hardware matrix or ensure that only supported hardware is visible during initialization via the CUDA_VISIBLE_DEVICES environment variable.


    .. autoattribute:: cuda.bindings.driver.CUresult.CUDA_ERROR_MPS_CONNECTION_FAILED


        This error indicates that the MPS client failed to connect to the MPS control daemon or the MPS server.


    .. autoattribute:: cuda.bindings.driver.CUresult.CUDA_ERROR_MPS_RPC_FAILURE


        This error indicates that the remote procedural call between the MPS server and the MPS client failed.


    .. autoattribute:: cuda.bindings.driver.CUresult.CUDA_ERROR_MPS_SERVER_NOT_READY


        This error indicates that the MPS server is not ready to accept new MPS client requests. This error can be returned when the MPS server is in the process of recovering from a fatal failure.


    .. autoattribute:: cuda.bindings.driver.CUresult.CUDA_ERROR_MPS_MAX_CLIENTS_REACHED


        This error indicates that the hardware resources required to create MPS client have been exhausted.


    .. autoattribute:: cuda.bindings.driver.CUresult.CUDA_ERROR_MPS_MAX_CONNECTIONS_REACHED


        This error indicates the the hardware resources required to support device connections have been exhausted.


    .. autoattribute:: cuda.bindings.driver.CUresult.CUDA_ERROR_MPS_CLIENT_TERMINATED


        This error indicates that the MPS client has been terminated by the server. To continue using CUDA, the process must be terminated and relaunched.


    .. autoattribute:: cuda.bindings.driver.CUresult.CUDA_ERROR_CDP_NOT_SUPPORTED


        This error indicates that the module is using CUDA Dynamic Parallelism, but the current configuration, like MPS, does not support it.


    .. autoattribute:: cuda.bindings.driver.CUresult.CUDA_ERROR_CDP_VERSION_MISMATCH


        This error indicates that a module contains an unsupported interaction between different versions of CUDA Dynamic Parallelism.


    .. autoattribute:: cuda.bindings.driver.CUresult.CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED


        This error indicates that the operation is not permitted when the stream is capturing.


    .. autoattribute:: cuda.bindings.driver.CUresult.CUDA_ERROR_STREAM_CAPTURE_INVALIDATED


        This error indicates that the current capture sequence on the stream has been invalidated due to a previous error.


    .. autoattribute:: cuda.bindings.driver.CUresult.CUDA_ERROR_STREAM_CAPTURE_MERGE


        This error indicates that the operation would have resulted in a merge of two independent capture sequences.


    .. autoattribute:: cuda.bindings.driver.CUresult.CUDA_ERROR_STREAM_CAPTURE_UNMATCHED


        This error indicates that the capture was not initiated in this stream.


    .. autoattribute:: cuda.bindings.driver.CUresult.CUDA_ERROR_STREAM_CAPTURE_UNJOINED


        This error indicates that the capture sequence contains a fork that was not joined to the primary stream.


    .. autoattribute:: cuda.bindings.driver.CUresult.CUDA_ERROR_STREAM_CAPTURE_ISOLATION


        This error indicates that a dependency would have been created which crosses the capture sequence boundary. Only implicit in-stream ordering dependencies are allowed to cross the boundary.


    .. autoattribute:: cuda.bindings.driver.CUresult.CUDA_ERROR_STREAM_CAPTURE_IMPLICIT


        This error indicates a disallowed implicit dependency on a current capture sequence from cudaStreamLegacy.


    .. autoattribute:: cuda.bindings.driver.CUresult.CUDA_ERROR_CAPTURED_EVENT


        This error indicates that the operation is not permitted on an event which was last recorded in a capturing stream.


    .. autoattribute:: cuda.bindings.driver.CUresult.CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD


        A stream capture sequence not initiated with the :py:obj:`~.CU_STREAM_CAPTURE_MODE_RELAXED` argument to :py:obj:`~.cuStreamBeginCapture` was passed to :py:obj:`~.cuStreamEndCapture` in a different thread.


    .. autoattribute:: cuda.bindings.driver.CUresult.CUDA_ERROR_TIMEOUT


        This error indicates that the timeout specified for the wait operation has lapsed.


    .. autoattribute:: cuda.bindings.driver.CUresult.CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE


        This error indicates that the graph update was not performed because it included changes which violated constraints specific to instantiated graph update.


    .. autoattribute:: cuda.bindings.driver.CUresult.CUDA_ERROR_EXTERNAL_DEVICE


        This indicates that an async error has occurred in a device outside of CUDA. If CUDA was waiting for an external device's signal before consuming shared data, the external device signaled an error indicating that the data is not valid for consumption. This leaves the process in an inconsistent state and any further CUDA work will return the same error. To continue using CUDA, the process must be terminated and relaunched.


    .. autoattribute:: cuda.bindings.driver.CUresult.CUDA_ERROR_INVALID_CLUSTER_SIZE


        Indicates a kernel launch error due to cluster misconfiguration.


    .. autoattribute:: cuda.bindings.driver.CUresult.CUDA_ERROR_FUNCTION_NOT_LOADED


        Indiciates a function handle is not loaded when calling an API that requires a loaded function.


    .. autoattribute:: cuda.bindings.driver.CUresult.CUDA_ERROR_INVALID_RESOURCE_TYPE


        This error indicates one or more resources passed in are not valid resource types for the operation.


    .. autoattribute:: cuda.bindings.driver.CUresult.CUDA_ERROR_INVALID_RESOURCE_CONFIGURATION


        This error indicates one or more resources are insufficient or non-applicable for the operation.


    .. autoattribute:: cuda.bindings.driver.CUresult.CUDA_ERROR_KEY_ROTATION


        This error indicates that an error happened during the key rotation sequence.


    .. autoattribute:: cuda.bindings.driver.CUresult.CUDA_ERROR_UNKNOWN


        This indicates that an unknown internal error has occurred.

.. autoclass:: cuda.bindings.driver.CUdevice_P2PAttribute

    .. autoattribute:: cuda.bindings.driver.CUdevice_P2PAttribute.CU_DEVICE_P2P_ATTRIBUTE_PERFORMANCE_RANK


        A relative value indicating the performance of the link between two devices


    .. autoattribute:: cuda.bindings.driver.CUdevice_P2PAttribute.CU_DEVICE_P2P_ATTRIBUTE_ACCESS_SUPPORTED


        P2P Access is enable


    .. autoattribute:: cuda.bindings.driver.CUdevice_P2PAttribute.CU_DEVICE_P2P_ATTRIBUTE_NATIVE_ATOMIC_SUPPORTED


        All CUDA-valid atomic operation over the link are supported


    .. autoattribute:: cuda.bindings.driver.CUdevice_P2PAttribute.CU_DEVICE_P2P_ATTRIBUTE_ACCESS_ACCESS_SUPPORTED


        [Deprecated]


    .. autoattribute:: cuda.bindings.driver.CUdevice_P2PAttribute.CU_DEVICE_P2P_ATTRIBUTE_CUDA_ARRAY_ACCESS_SUPPORTED


        Accessing CUDA arrays over the link supported


    .. autoattribute:: cuda.bindings.driver.CUdevice_P2PAttribute.CU_DEVICE_P2P_ATTRIBUTE_ONLY_PARTIAL_NATIVE_ATOMIC_SUPPORTED


        Only some CUDA-valid atomic operations over the link are supported.

.. autoclass:: cuda.bindings.driver.CUatomicOperation

    .. autoattribute:: cuda.bindings.driver.CUatomicOperation.CU_ATOMIC_OPERATION_INTEGER_ADD


    .. autoattribute:: cuda.bindings.driver.CUatomicOperation.CU_ATOMIC_OPERATION_INTEGER_MIN


    .. autoattribute:: cuda.bindings.driver.CUatomicOperation.CU_ATOMIC_OPERATION_INTEGER_MAX


    .. autoattribute:: cuda.bindings.driver.CUatomicOperation.CU_ATOMIC_OPERATION_INTEGER_INCREMENT


    .. autoattribute:: cuda.bindings.driver.CUatomicOperation.CU_ATOMIC_OPERATION_INTEGER_DECREMENT


    .. autoattribute:: cuda.bindings.driver.CUatomicOperation.CU_ATOMIC_OPERATION_AND


    .. autoattribute:: cuda.bindings.driver.CUatomicOperation.CU_ATOMIC_OPERATION_OR


    .. autoattribute:: cuda.bindings.driver.CUatomicOperation.CU_ATOMIC_OPERATION_XOR


    .. autoattribute:: cuda.bindings.driver.CUatomicOperation.CU_ATOMIC_OPERATION_EXCHANGE


    .. autoattribute:: cuda.bindings.driver.CUatomicOperation.CU_ATOMIC_OPERATION_CAS


    .. autoattribute:: cuda.bindings.driver.CUatomicOperation.CU_ATOMIC_OPERATION_FLOAT_ADD


    .. autoattribute:: cuda.bindings.driver.CUatomicOperation.CU_ATOMIC_OPERATION_FLOAT_MIN


    .. autoattribute:: cuda.bindings.driver.CUatomicOperation.CU_ATOMIC_OPERATION_FLOAT_MAX


    .. autoattribute:: cuda.bindings.driver.CUatomicOperation.CU_ATOMIC_OPERATION_MAX

.. autoclass:: cuda.bindings.driver.CUatomicOperationCapability

    .. autoattribute:: cuda.bindings.driver.CUatomicOperationCapability.CU_ATOMIC_CAPABILITY_SIGNED


    .. autoattribute:: cuda.bindings.driver.CUatomicOperationCapability.CU_ATOMIC_CAPABILITY_UNSIGNED


    .. autoattribute:: cuda.bindings.driver.CUatomicOperationCapability.CU_ATOMIC_CAPABILITY_REDUCTION


    .. autoattribute:: cuda.bindings.driver.CUatomicOperationCapability.CU_ATOMIC_CAPABILITY_SCALAR_32


    .. autoattribute:: cuda.bindings.driver.CUatomicOperationCapability.CU_ATOMIC_CAPABILITY_SCALAR_64


    .. autoattribute:: cuda.bindings.driver.CUatomicOperationCapability.CU_ATOMIC_CAPABILITY_SCALAR_128


    .. autoattribute:: cuda.bindings.driver.CUatomicOperationCapability.CU_ATOMIC_CAPABILITY_VECTOR_32x4

.. autoclass:: cuda.bindings.driver.CUresourceViewFormat

    .. autoattribute:: cuda.bindings.driver.CUresourceViewFormat.CU_RES_VIEW_FORMAT_NONE


        No resource view format (use underlying resource format)


    .. autoattribute:: cuda.bindings.driver.CUresourceViewFormat.CU_RES_VIEW_FORMAT_UINT_1X8


        1 channel unsigned 8-bit integers


    .. autoattribute:: cuda.bindings.driver.CUresourceViewFormat.CU_RES_VIEW_FORMAT_UINT_2X8


        2 channel unsigned 8-bit integers


    .. autoattribute:: cuda.bindings.driver.CUresourceViewFormat.CU_RES_VIEW_FORMAT_UINT_4X8


        4 channel unsigned 8-bit integers


    .. autoattribute:: cuda.bindings.driver.CUresourceViewFormat.CU_RES_VIEW_FORMAT_SINT_1X8


        1 channel signed 8-bit integers


    .. autoattribute:: cuda.bindings.driver.CUresourceViewFormat.CU_RES_VIEW_FORMAT_SINT_2X8


        2 channel signed 8-bit integers


    .. autoattribute:: cuda.bindings.driver.CUresourceViewFormat.CU_RES_VIEW_FORMAT_SINT_4X8


        4 channel signed 8-bit integers


    .. autoattribute:: cuda.bindings.driver.CUresourceViewFormat.CU_RES_VIEW_FORMAT_UINT_1X16


        1 channel unsigned 16-bit integers


    .. autoattribute:: cuda.bindings.driver.CUresourceViewFormat.CU_RES_VIEW_FORMAT_UINT_2X16


        2 channel unsigned 16-bit integers


    .. autoattribute:: cuda.bindings.driver.CUresourceViewFormat.CU_RES_VIEW_FORMAT_UINT_4X16


        4 channel unsigned 16-bit integers


    .. autoattribute:: cuda.bindings.driver.CUresourceViewFormat.CU_RES_VIEW_FORMAT_SINT_1X16


        1 channel signed 16-bit integers


    .. autoattribute:: cuda.bindings.driver.CUresourceViewFormat.CU_RES_VIEW_FORMAT_SINT_2X16


        2 channel signed 16-bit integers


    .. autoattribute:: cuda.bindings.driver.CUresourceViewFormat.CU_RES_VIEW_FORMAT_SINT_4X16


        4 channel signed 16-bit integers


    .. autoattribute:: cuda.bindings.driver.CUresourceViewFormat.CU_RES_VIEW_FORMAT_UINT_1X32


        1 channel unsigned 32-bit integers


    .. autoattribute:: cuda.bindings.driver.CUresourceViewFormat.CU_RES_VIEW_FORMAT_UINT_2X32


        2 channel unsigned 32-bit integers


    .. autoattribute:: cuda.bindings.driver.CUresourceViewFormat.CU_RES_VIEW_FORMAT_UINT_4X32


        4 channel unsigned 32-bit integers


    .. autoattribute:: cuda.bindings.driver.CUresourceViewFormat.CU_RES_VIEW_FORMAT_SINT_1X32


        1 channel signed 32-bit integers


    .. autoattribute:: cuda.bindings.driver.CUresourceViewFormat.CU_RES_VIEW_FORMAT_SINT_2X32


        2 channel signed 32-bit integers


    .. autoattribute:: cuda.bindings.driver.CUresourceViewFormat.CU_RES_VIEW_FORMAT_SINT_4X32


        4 channel signed 32-bit integers


    .. autoattribute:: cuda.bindings.driver.CUresourceViewFormat.CU_RES_VIEW_FORMAT_FLOAT_1X16


        1 channel 16-bit floating point


    .. autoattribute:: cuda.bindings.driver.CUresourceViewFormat.CU_RES_VIEW_FORMAT_FLOAT_2X16


        2 channel 16-bit floating point


    .. autoattribute:: cuda.bindings.driver.CUresourceViewFormat.CU_RES_VIEW_FORMAT_FLOAT_4X16


        4 channel 16-bit floating point


    .. autoattribute:: cuda.bindings.driver.CUresourceViewFormat.CU_RES_VIEW_FORMAT_FLOAT_1X32


        1 channel 32-bit floating point


    .. autoattribute:: cuda.bindings.driver.CUresourceViewFormat.CU_RES_VIEW_FORMAT_FLOAT_2X32


        2 channel 32-bit floating point


    .. autoattribute:: cuda.bindings.driver.CUresourceViewFormat.CU_RES_VIEW_FORMAT_FLOAT_4X32


        4 channel 32-bit floating point


    .. autoattribute:: cuda.bindings.driver.CUresourceViewFormat.CU_RES_VIEW_FORMAT_UNSIGNED_BC1


        Block compressed 1


    .. autoattribute:: cuda.bindings.driver.CUresourceViewFormat.CU_RES_VIEW_FORMAT_UNSIGNED_BC2


        Block compressed 2


    .. autoattribute:: cuda.bindings.driver.CUresourceViewFormat.CU_RES_VIEW_FORMAT_UNSIGNED_BC3


        Block compressed 3


    .. autoattribute:: cuda.bindings.driver.CUresourceViewFormat.CU_RES_VIEW_FORMAT_UNSIGNED_BC4


        Block compressed 4 unsigned


    .. autoattribute:: cuda.bindings.driver.CUresourceViewFormat.CU_RES_VIEW_FORMAT_SIGNED_BC4


        Block compressed 4 signed


    .. autoattribute:: cuda.bindings.driver.CUresourceViewFormat.CU_RES_VIEW_FORMAT_UNSIGNED_BC5


        Block compressed 5 unsigned


    .. autoattribute:: cuda.bindings.driver.CUresourceViewFormat.CU_RES_VIEW_FORMAT_SIGNED_BC5


        Block compressed 5 signed


    .. autoattribute:: cuda.bindings.driver.CUresourceViewFormat.CU_RES_VIEW_FORMAT_UNSIGNED_BC6H


        Block compressed 6 unsigned half-float


    .. autoattribute:: cuda.bindings.driver.CUresourceViewFormat.CU_RES_VIEW_FORMAT_SIGNED_BC6H


        Block compressed 6 signed half-float


    .. autoattribute:: cuda.bindings.driver.CUresourceViewFormat.CU_RES_VIEW_FORMAT_UNSIGNED_BC7


        Block compressed 7

.. autoclass:: cuda.bindings.driver.CUtensorMapDataType

    .. autoattribute:: cuda.bindings.driver.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT8


    .. autoattribute:: cuda.bindings.driver.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT16


    .. autoattribute:: cuda.bindings.driver.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT32


    .. autoattribute:: cuda.bindings.driver.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_INT32


    .. autoattribute:: cuda.bindings.driver.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT64


    .. autoattribute:: cuda.bindings.driver.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_INT64


    .. autoattribute:: cuda.bindings.driver.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_FLOAT16


    .. autoattribute:: cuda.bindings.driver.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_FLOAT32


    .. autoattribute:: cuda.bindings.driver.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_FLOAT64


    .. autoattribute:: cuda.bindings.driver.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_BFLOAT16


    .. autoattribute:: cuda.bindings.driver.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_FLOAT32_FTZ


    .. autoattribute:: cuda.bindings.driver.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_TFLOAT32


    .. autoattribute:: cuda.bindings.driver.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_TFLOAT32_FTZ


    .. autoattribute:: cuda.bindings.driver.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B


    .. autoattribute:: cuda.bindings.driver.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B


    .. autoattribute:: cuda.bindings.driver.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_16U6_ALIGN16B

.. autoclass:: cuda.bindings.driver.CUtensorMapInterleave

    .. autoattribute:: cuda.bindings.driver.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE


    .. autoattribute:: cuda.bindings.driver.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_16B


    .. autoattribute:: cuda.bindings.driver.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_32B

.. autoclass:: cuda.bindings.driver.CUtensorMapSwizzle

    .. autoattribute:: cuda.bindings.driver.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_NONE


    .. autoattribute:: cuda.bindings.driver.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_32B


    .. autoattribute:: cuda.bindings.driver.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_64B


    .. autoattribute:: cuda.bindings.driver.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B


    .. autoattribute:: cuda.bindings.driver.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B_ATOM_32B


    .. autoattribute:: cuda.bindings.driver.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B_ATOM_32B_FLIP_8B


    .. autoattribute:: cuda.bindings.driver.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B_ATOM_64B

.. autoclass:: cuda.bindings.driver.CUtensorMapL2promotion

    .. autoattribute:: cuda.bindings.driver.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_NONE


    .. autoattribute:: cuda.bindings.driver.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_L2_64B


    .. autoattribute:: cuda.bindings.driver.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_L2_128B


    .. autoattribute:: cuda.bindings.driver.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_L2_256B

.. autoclass:: cuda.bindings.driver.CUtensorMapFloatOOBfill

    .. autoattribute:: cuda.bindings.driver.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE


    .. autoattribute:: cuda.bindings.driver.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA

.. autoclass:: cuda.bindings.driver.CUtensorMapIm2ColWideMode

    .. autoattribute:: cuda.bindings.driver.CUtensorMapIm2ColWideMode.CU_TENSOR_MAP_IM2COL_WIDE_MODE_W


    .. autoattribute:: cuda.bindings.driver.CUtensorMapIm2ColWideMode.CU_TENSOR_MAP_IM2COL_WIDE_MODE_W128

.. autoclass:: cuda.bindings.driver.CUDA_POINTER_ATTRIBUTE_ACCESS_FLAGS

    .. autoattribute:: cuda.bindings.driver.CUDA_POINTER_ATTRIBUTE_ACCESS_FLAGS.CU_POINTER_ATTRIBUTE_ACCESS_FLAG_NONE


        No access, meaning the device cannot access this memory at all, thus must be staged through accessible memory in order to complete certain operations


    .. autoattribute:: cuda.bindings.driver.CUDA_POINTER_ATTRIBUTE_ACCESS_FLAGS.CU_POINTER_ATTRIBUTE_ACCESS_FLAG_READ


        Read-only access, meaning writes to this memory are considered invalid accesses and thus return error in that case.


    .. autoattribute:: cuda.bindings.driver.CUDA_POINTER_ATTRIBUTE_ACCESS_FLAGS.CU_POINTER_ATTRIBUTE_ACCESS_FLAG_READWRITE


        Read-write access, the device has full read-write access to the memory

.. autoclass:: cuda.bindings.driver.CUexternalMemoryHandleType

    .. autoattribute:: cuda.bindings.driver.CUexternalMemoryHandleType.CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD


        Handle is an opaque file descriptor


    .. autoattribute:: cuda.bindings.driver.CUexternalMemoryHandleType.CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32


        Handle is an opaque shared NT handle


    .. autoattribute:: cuda.bindings.driver.CUexternalMemoryHandleType.CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT


        Handle is an opaque, globally shared handle


    .. autoattribute:: cuda.bindings.driver.CUexternalMemoryHandleType.CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_HEAP


        Handle is a D3D12 heap object


    .. autoattribute:: cuda.bindings.driver.CUexternalMemoryHandleType.CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE


        Handle is a D3D12 committed resource


    .. autoattribute:: cuda.bindings.driver.CUexternalMemoryHandleType.CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_RESOURCE


        Handle is a shared NT handle to a D3D11 resource


    .. autoattribute:: cuda.bindings.driver.CUexternalMemoryHandleType.CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_RESOURCE_KMT


        Handle is a globally shared handle to a D3D11 resource


    .. autoattribute:: cuda.bindings.driver.CUexternalMemoryHandleType.CU_EXTERNAL_MEMORY_HANDLE_TYPE_NVSCIBUF


        Handle is an NvSciBuf object


    .. autoattribute:: cuda.bindings.driver.CUexternalMemoryHandleType.CU_EXTERNAL_MEMORY_HANDLE_TYPE_DMABUF_FD


        Handle is a dma_buf file descriptor

.. autoclass:: cuda.bindings.driver.CUexternalSemaphoreHandleType

    .. autoattribute:: cuda.bindings.driver.CUexternalSemaphoreHandleType.CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD


        Handle is an opaque file descriptor


    .. autoattribute:: cuda.bindings.driver.CUexternalSemaphoreHandleType.CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32


        Handle is an opaque shared NT handle


    .. autoattribute:: cuda.bindings.driver.CUexternalSemaphoreHandleType.CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_KMT


        Handle is an opaque, globally shared handle


    .. autoattribute:: cuda.bindings.driver.CUexternalSemaphoreHandleType.CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D12_FENCE


        Handle is a shared NT handle referencing a D3D12 fence object


    .. autoattribute:: cuda.bindings.driver.CUexternalSemaphoreHandleType.CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D11_FENCE


        Handle is a shared NT handle referencing a D3D11 fence object


    .. autoattribute:: cuda.bindings.driver.CUexternalSemaphoreHandleType.CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_NVSCISYNC


        Opaque handle to NvSciSync Object


    .. autoattribute:: cuda.bindings.driver.CUexternalSemaphoreHandleType.CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D11_KEYED_MUTEX


        Handle is a shared NT handle referencing a D3D11 keyed mutex object


    .. autoattribute:: cuda.bindings.driver.CUexternalSemaphoreHandleType.CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D11_KEYED_MUTEX_KMT


        Handle is a globally shared handle referencing a D3D11 keyed mutex object


    .. autoattribute:: cuda.bindings.driver.CUexternalSemaphoreHandleType.CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_TIMELINE_SEMAPHORE_FD


        Handle is an opaque file descriptor referencing a timeline semaphore


    .. autoattribute:: cuda.bindings.driver.CUexternalSemaphoreHandleType.CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_TIMELINE_SEMAPHORE_WIN32


        Handle is an opaque shared NT handle referencing a timeline semaphore

.. autoclass:: cuda.bindings.driver.CUmemAllocationHandleType

    .. autoattribute:: cuda.bindings.driver.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_NONE


        Does not allow any export mechanism. >


    .. autoattribute:: cuda.bindings.driver.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR


        Allows a file descriptor to be used for exporting. Permitted only on POSIX systems. (int)


    .. autoattribute:: cuda.bindings.driver.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_WIN32


        Allows a Win32 NT handle to be used for exporting. (HANDLE)


    .. autoattribute:: cuda.bindings.driver.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_WIN32_KMT


        Allows a Win32 KMT handle to be used for exporting. (D3DKMT_HANDLE)


    .. autoattribute:: cuda.bindings.driver.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC


        Allows a fabric handle to be used for exporting. (CUmemFabricHandle)


    .. autoattribute:: cuda.bindings.driver.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_MAX

.. autoclass:: cuda.bindings.driver.CUmemAccess_flags

    .. autoattribute:: cuda.bindings.driver.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_NONE


        Default, make the address range not accessible


    .. autoattribute:: cuda.bindings.driver.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READ


        Make the address range read accessible


    .. autoattribute:: cuda.bindings.driver.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE


        Make the address range read-write accessible


    .. autoattribute:: cuda.bindings.driver.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_MAX

.. autoclass:: cuda.bindings.driver.CUmemLocationType

    .. autoattribute:: cuda.bindings.driver.CUmemLocationType.CU_MEM_LOCATION_TYPE_INVALID


    .. autoattribute:: cuda.bindings.driver.CUmemLocationType.CU_MEM_LOCATION_TYPE_NONE


        Location is unspecified. This is used when creating a managed memory pool to indicate no preferred location for the pool


    .. autoattribute:: cuda.bindings.driver.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE


        Location is a device location, thus id is a device ordinal


    .. autoattribute:: cuda.bindings.driver.CUmemLocationType.CU_MEM_LOCATION_TYPE_HOST


        Location is host, id is ignored


    .. autoattribute:: cuda.bindings.driver.CUmemLocationType.CU_MEM_LOCATION_TYPE_HOST_NUMA


        Location is a host NUMA node, thus id is a host NUMA node id


    .. autoattribute:: cuda.bindings.driver.CUmemLocationType.CU_MEM_LOCATION_TYPE_HOST_NUMA_CURRENT


        Location is a host NUMA node of the current thread, id is ignored


    .. autoattribute:: cuda.bindings.driver.CUmemLocationType.CU_MEM_LOCATION_TYPE_MAX

.. autoclass:: cuda.bindings.driver.CUmemAllocationType

    .. autoattribute:: cuda.bindings.driver.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_INVALID


    .. autoattribute:: cuda.bindings.driver.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED


        This allocation type is 'pinned', i.e. cannot migrate from its current location while the application is actively using it


    .. autoattribute:: cuda.bindings.driver.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_MANAGED


        This allocation type is managed memory


    .. autoattribute:: cuda.bindings.driver.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_MAX

.. autoclass:: cuda.bindings.driver.CUmemAllocationGranularity_flags

    .. autoattribute:: cuda.bindings.driver.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_MINIMUM


        Minimum required granularity for allocation


    .. autoattribute:: cuda.bindings.driver.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_RECOMMENDED


        Recommended granularity for allocation for best performance

.. autoclass:: cuda.bindings.driver.CUmemRangeHandleType

    .. autoattribute:: cuda.bindings.driver.CUmemRangeHandleType.CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD


    .. autoattribute:: cuda.bindings.driver.CUmemRangeHandleType.CU_MEM_RANGE_HANDLE_TYPE_MAX

.. autoclass:: cuda.bindings.driver.CUmemRangeFlags

    .. autoattribute:: cuda.bindings.driver.CUmemRangeFlags.CU_MEM_RANGE_FLAG_DMA_BUF_MAPPING_TYPE_PCIE


        Indicates that DMA_BUF handle should be mapped via PCIe BAR1

.. autoclass:: cuda.bindings.driver.CUarraySparseSubresourceType

    .. autoattribute:: cuda.bindings.driver.CUarraySparseSubresourceType.CU_ARRAY_SPARSE_SUBRESOURCE_TYPE_SPARSE_LEVEL


    .. autoattribute:: cuda.bindings.driver.CUarraySparseSubresourceType.CU_ARRAY_SPARSE_SUBRESOURCE_TYPE_MIPTAIL

.. autoclass:: cuda.bindings.driver.CUmemOperationType

    .. autoattribute:: cuda.bindings.driver.CUmemOperationType.CU_MEM_OPERATION_TYPE_MAP


    .. autoattribute:: cuda.bindings.driver.CUmemOperationType.CU_MEM_OPERATION_TYPE_UNMAP

.. autoclass:: cuda.bindings.driver.CUmemHandleType

    .. autoattribute:: cuda.bindings.driver.CUmemHandleType.CU_MEM_HANDLE_TYPE_GENERIC

.. autoclass:: cuda.bindings.driver.CUmemAllocationCompType

    .. autoattribute:: cuda.bindings.driver.CUmemAllocationCompType.CU_MEM_ALLOCATION_COMP_NONE


        Allocating non-compressible memory


    .. autoattribute:: cuda.bindings.driver.CUmemAllocationCompType.CU_MEM_ALLOCATION_COMP_GENERIC


        Allocating compressible memory

.. autoclass:: cuda.bindings.driver.CUmulticastGranularity_flags

    .. autoattribute:: cuda.bindings.driver.CUmulticastGranularity_flags.CU_MULTICAST_GRANULARITY_MINIMUM


        Minimum required granularity


    .. autoattribute:: cuda.bindings.driver.CUmulticastGranularity_flags.CU_MULTICAST_GRANULARITY_RECOMMENDED


        Recommended granularity for best performance

.. autoclass:: cuda.bindings.driver.CUgraphExecUpdateResult

    .. autoattribute:: cuda.bindings.driver.CUgraphExecUpdateResult.CU_GRAPH_EXEC_UPDATE_SUCCESS


        The update succeeded


    .. autoattribute:: cuda.bindings.driver.CUgraphExecUpdateResult.CU_GRAPH_EXEC_UPDATE_ERROR


        The update failed for an unexpected reason which is described in the return value of the function


    .. autoattribute:: cuda.bindings.driver.CUgraphExecUpdateResult.CU_GRAPH_EXEC_UPDATE_ERROR_TOPOLOGY_CHANGED


        The update failed because the topology changed


    .. autoattribute:: cuda.bindings.driver.CUgraphExecUpdateResult.CU_GRAPH_EXEC_UPDATE_ERROR_NODE_TYPE_CHANGED


        The update failed because a node type changed


    .. autoattribute:: cuda.bindings.driver.CUgraphExecUpdateResult.CU_GRAPH_EXEC_UPDATE_ERROR_FUNCTION_CHANGED


        The update failed because the function of a kernel node changed (CUDA driver < 11.2)


    .. autoattribute:: cuda.bindings.driver.CUgraphExecUpdateResult.CU_GRAPH_EXEC_UPDATE_ERROR_PARAMETERS_CHANGED


        The update failed because the parameters changed in a way that is not supported


    .. autoattribute:: cuda.bindings.driver.CUgraphExecUpdateResult.CU_GRAPH_EXEC_UPDATE_ERROR_NOT_SUPPORTED


        The update failed because something about the node is not supported


    .. autoattribute:: cuda.bindings.driver.CUgraphExecUpdateResult.CU_GRAPH_EXEC_UPDATE_ERROR_UNSUPPORTED_FUNCTION_CHANGE


        The update failed because the function of a kernel node changed in an unsupported way


    .. autoattribute:: cuda.bindings.driver.CUgraphExecUpdateResult.CU_GRAPH_EXEC_UPDATE_ERROR_ATTRIBUTES_CHANGED


        The update failed because the node attributes changed in a way that is not supported

.. autoclass:: cuda.bindings.driver.CUmemPool_attribute

    .. autoattribute:: cuda.bindings.driver.CUmemPool_attribute.CU_MEMPOOL_ATTR_REUSE_FOLLOW_EVENT_DEPENDENCIES


        (value type = int) Allow cuMemAllocAsync to use memory asynchronously freed in another streams as long as a stream ordering dependency of the allocating stream on the free action exists. Cuda events and null stream interactions can create the required stream ordered dependencies. (default enabled)


    .. autoattribute:: cuda.bindings.driver.CUmemPool_attribute.CU_MEMPOOL_ATTR_REUSE_ALLOW_OPPORTUNISTIC


        (value type = int) Allow reuse of already completed frees when there is no dependency between the free and allocation. (default enabled)


    .. autoattribute:: cuda.bindings.driver.CUmemPool_attribute.CU_MEMPOOL_ATTR_REUSE_ALLOW_INTERNAL_DEPENDENCIES


        (value type = int) Allow cuMemAllocAsync to insert new stream dependencies in order to establish the stream ordering required to reuse a piece of memory released by cuFreeAsync (default enabled).


    .. autoattribute:: cuda.bindings.driver.CUmemPool_attribute.CU_MEMPOOL_ATTR_RELEASE_THRESHOLD


        (value type = cuuint64_t) Amount of reserved memory in bytes to hold onto before trying to release memory back to the OS. When more than the release threshold bytes of memory are held by the memory pool, the allocator will try to release memory back to the OS on the next call to stream, event or context synchronize. (default 0)


    .. autoattribute:: cuda.bindings.driver.CUmemPool_attribute.CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT


        (value type = cuuint64_t) Amount of backing memory currently allocated for the mempool.


    .. autoattribute:: cuda.bindings.driver.CUmemPool_attribute.CU_MEMPOOL_ATTR_RESERVED_MEM_HIGH


        (value type = cuuint64_t) High watermark of backing memory allocated for the mempool since the last time it was reset. High watermark can only be reset to zero.


    .. autoattribute:: cuda.bindings.driver.CUmemPool_attribute.CU_MEMPOOL_ATTR_USED_MEM_CURRENT


        (value type = cuuint64_t) Amount of memory from the pool that is currently in use by the application.


    .. autoattribute:: cuda.bindings.driver.CUmemPool_attribute.CU_MEMPOOL_ATTR_USED_MEM_HIGH


        (value type = cuuint64_t) High watermark of the amount of memory from the pool that was in use by the application since the last time it was reset. High watermark can only be reset to zero.

.. autoclass:: cuda.bindings.driver.CUmemcpyFlags

    .. autoattribute:: cuda.bindings.driver.CUmemcpyFlags.CU_MEMCPY_FLAG_DEFAULT


    .. autoattribute:: cuda.bindings.driver.CUmemcpyFlags.CU_MEMCPY_FLAG_PREFER_OVERLAP_WITH_COMPUTE


        Hint to the driver to try and overlap the copy with compute work on the SMs.

.. autoclass:: cuda.bindings.driver.CUmemcpySrcAccessOrder

    .. autoattribute:: cuda.bindings.driver.CUmemcpySrcAccessOrder.CU_MEMCPY_SRC_ACCESS_ORDER_INVALID


        Default invalid.


    .. autoattribute:: cuda.bindings.driver.CUmemcpySrcAccessOrder.CU_MEMCPY_SRC_ACCESS_ORDER_STREAM


        Indicates that access to the source pointer must be in stream order.


    .. autoattribute:: cuda.bindings.driver.CUmemcpySrcAccessOrder.CU_MEMCPY_SRC_ACCESS_ORDER_DURING_API_CALL


        Indicates that access to the source pointer can be out of stream order and all accesses must be complete before the API call returns. This flag is suited for ephemeral sources (ex., stack variables) when it's known that no prior operations in the stream can be accessing the memory and also that the lifetime of the memory is limited to the scope that the source variable was declared in. Specifying this flag allows the driver to optimize the copy and removes the need for the user to synchronize the stream after the API call.


    .. autoattribute:: cuda.bindings.driver.CUmemcpySrcAccessOrder.CU_MEMCPY_SRC_ACCESS_ORDER_ANY


        Indicates that access to the source pointer can be out of stream order and the accesses can happen even after the API call returns. This flag is suited for host pointers allocated outside CUDA (ex., via malloc) when it's known that no prior operations in the stream can be accessing the memory. Specifying this flag allows the driver to optimize the copy on certain platforms.


    .. autoattribute:: cuda.bindings.driver.CUmemcpySrcAccessOrder.CU_MEMCPY_SRC_ACCESS_ORDER_MAX

.. autoclass:: cuda.bindings.driver.CUmemcpy3DOperandType

    .. autoattribute:: cuda.bindings.driver.CUmemcpy3DOperandType.CU_MEMCPY_OPERAND_TYPE_POINTER


        Memcpy operand is a valid pointer.


    .. autoattribute:: cuda.bindings.driver.CUmemcpy3DOperandType.CU_MEMCPY_OPERAND_TYPE_ARRAY


        Memcpy operand is a CUarray.


    .. autoattribute:: cuda.bindings.driver.CUmemcpy3DOperandType.CU_MEMCPY_OPERAND_TYPE_MAX

.. autoclass:: cuda.bindings.driver.CUgraphMem_attribute

    .. autoattribute:: cuda.bindings.driver.CUgraphMem_attribute.CU_GRAPH_MEM_ATTR_USED_MEM_CURRENT


        (value type = cuuint64_t) Amount of memory, in bytes, currently associated with graphs


    .. autoattribute:: cuda.bindings.driver.CUgraphMem_attribute.CU_GRAPH_MEM_ATTR_USED_MEM_HIGH


        (value type = cuuint64_t) High watermark of memory, in bytes, associated with graphs since the last time it was reset. High watermark can only be reset to zero.


    .. autoattribute:: cuda.bindings.driver.CUgraphMem_attribute.CU_GRAPH_MEM_ATTR_RESERVED_MEM_CURRENT


        (value type = cuuint64_t) Amount of memory, in bytes, currently allocated for use by the CUDA graphs asynchronous allocator.


    .. autoattribute:: cuda.bindings.driver.CUgraphMem_attribute.CU_GRAPH_MEM_ATTR_RESERVED_MEM_HIGH


        (value type = cuuint64_t) High watermark of memory, in bytes, currently allocated for use by the CUDA graphs asynchronous allocator.

.. autoclass:: cuda.bindings.driver.CUgraphChildGraphNodeOwnership

    .. autoattribute:: cuda.bindings.driver.CUgraphChildGraphNodeOwnership.CU_GRAPH_CHILD_GRAPH_OWNERSHIP_CLONE


        Default behavior for a child graph node. Child graph is cloned into the parent and memory allocation/free nodes can't be present in the child graph.


    .. autoattribute:: cuda.bindings.driver.CUgraphChildGraphNodeOwnership.CU_GRAPH_CHILD_GRAPH_OWNERSHIP_MOVE


        The child graph is moved to the parent. The handle to the child graph is owned by the parent and will be destroyed when the parent is destroyed.



        The following restrictions apply to child graphs after they have been moved: Cannot be independently instantiated or destroyed; Cannot be added as a child graph of a separate parent graph; Cannot be used as an argument to cuGraphExecUpdate; Cannot have additional memory allocation or free nodes added.

.. autoclass:: cuda.bindings.driver.CUflushGPUDirectRDMAWritesOptions

    .. autoattribute:: cuda.bindings.driver.CUflushGPUDirectRDMAWritesOptions.CU_FLUSH_GPU_DIRECT_RDMA_WRITES_OPTION_HOST


        :py:obj:`~.cuFlushGPUDirectRDMAWrites()` and its CUDA Runtime API counterpart are supported on the device.


    .. autoattribute:: cuda.bindings.driver.CUflushGPUDirectRDMAWritesOptions.CU_FLUSH_GPU_DIRECT_RDMA_WRITES_OPTION_MEMOPS


        The :py:obj:`~.CU_STREAM_WAIT_VALUE_FLUSH` flag and the :py:obj:`~.CU_STREAM_MEM_OP_FLUSH_REMOTE_WRITES` MemOp are supported on the device.

.. autoclass:: cuda.bindings.driver.CUGPUDirectRDMAWritesOrdering

    .. autoattribute:: cuda.bindings.driver.CUGPUDirectRDMAWritesOrdering.CU_GPU_DIRECT_RDMA_WRITES_ORDERING_NONE


        The device does not natively support ordering of remote writes. :py:obj:`~.cuFlushGPUDirectRDMAWrites()` can be leveraged if supported.


    .. autoattribute:: cuda.bindings.driver.CUGPUDirectRDMAWritesOrdering.CU_GPU_DIRECT_RDMA_WRITES_ORDERING_OWNER


        Natively, the device can consistently consume remote writes, although other CUDA devices may not.


    .. autoattribute:: cuda.bindings.driver.CUGPUDirectRDMAWritesOrdering.CU_GPU_DIRECT_RDMA_WRITES_ORDERING_ALL_DEVICES


        Any CUDA device in the system can consistently consume remote writes to this device.

.. autoclass:: cuda.bindings.driver.CUflushGPUDirectRDMAWritesScope

    .. autoattribute:: cuda.bindings.driver.CUflushGPUDirectRDMAWritesScope.CU_FLUSH_GPU_DIRECT_RDMA_WRITES_TO_OWNER


        Blocks until remote writes are visible to the CUDA device context owning the data.


    .. autoattribute:: cuda.bindings.driver.CUflushGPUDirectRDMAWritesScope.CU_FLUSH_GPU_DIRECT_RDMA_WRITES_TO_ALL_DEVICES


        Blocks until remote writes are visible to all CUDA device contexts.

.. autoclass:: cuda.bindings.driver.CUflushGPUDirectRDMAWritesTarget

    .. autoattribute:: cuda.bindings.driver.CUflushGPUDirectRDMAWritesTarget.CU_FLUSH_GPU_DIRECT_RDMA_WRITES_TARGET_CURRENT_CTX


        Sets the target for :py:obj:`~.cuFlushGPUDirectRDMAWrites()` to the currently active CUDA device context.

.. autoclass:: cuda.bindings.driver.CUgraphDebugDot_flags

    .. autoattribute:: cuda.bindings.driver.CUgraphDebugDot_flags.CU_GRAPH_DEBUG_DOT_FLAGS_VERBOSE


        Output all debug data as if every debug flag is enabled


    .. autoattribute:: cuda.bindings.driver.CUgraphDebugDot_flags.CU_GRAPH_DEBUG_DOT_FLAGS_RUNTIME_TYPES


        Use CUDA Runtime structures for output


    .. autoattribute:: cuda.bindings.driver.CUgraphDebugDot_flags.CU_GRAPH_DEBUG_DOT_FLAGS_KERNEL_NODE_PARAMS


        Adds CUDA_KERNEL_NODE_PARAMS values to output


    .. autoattribute:: cuda.bindings.driver.CUgraphDebugDot_flags.CU_GRAPH_DEBUG_DOT_FLAGS_MEMCPY_NODE_PARAMS


        Adds CUDA_MEMCPY3D values to output


    .. autoattribute:: cuda.bindings.driver.CUgraphDebugDot_flags.CU_GRAPH_DEBUG_DOT_FLAGS_MEMSET_NODE_PARAMS


        Adds CUDA_MEMSET_NODE_PARAMS values to output


    .. autoattribute:: cuda.bindings.driver.CUgraphDebugDot_flags.CU_GRAPH_DEBUG_DOT_FLAGS_HOST_NODE_PARAMS


        Adds CUDA_HOST_NODE_PARAMS values to output


    .. autoattribute:: cuda.bindings.driver.CUgraphDebugDot_flags.CU_GRAPH_DEBUG_DOT_FLAGS_EVENT_NODE_PARAMS


        Adds CUevent handle from record and wait nodes to output


    .. autoattribute:: cuda.bindings.driver.CUgraphDebugDot_flags.CU_GRAPH_DEBUG_DOT_FLAGS_EXT_SEMAS_SIGNAL_NODE_PARAMS


        Adds CUDA_EXT_SEM_SIGNAL_NODE_PARAMS values to output


    .. autoattribute:: cuda.bindings.driver.CUgraphDebugDot_flags.CU_GRAPH_DEBUG_DOT_FLAGS_EXT_SEMAS_WAIT_NODE_PARAMS


        Adds CUDA_EXT_SEM_WAIT_NODE_PARAMS values to output


    .. autoattribute:: cuda.bindings.driver.CUgraphDebugDot_flags.CU_GRAPH_DEBUG_DOT_FLAGS_KERNEL_NODE_ATTRIBUTES


        Adds CUkernelNodeAttrValue values to output


    .. autoattribute:: cuda.bindings.driver.CUgraphDebugDot_flags.CU_GRAPH_DEBUG_DOT_FLAGS_HANDLES


        Adds node handles and every kernel function handle to output


    .. autoattribute:: cuda.bindings.driver.CUgraphDebugDot_flags.CU_GRAPH_DEBUG_DOT_FLAGS_MEM_ALLOC_NODE_PARAMS


        Adds memory alloc node parameters to output


    .. autoattribute:: cuda.bindings.driver.CUgraphDebugDot_flags.CU_GRAPH_DEBUG_DOT_FLAGS_MEM_FREE_NODE_PARAMS


        Adds memory free node parameters to output


    .. autoattribute:: cuda.bindings.driver.CUgraphDebugDot_flags.CU_GRAPH_DEBUG_DOT_FLAGS_BATCH_MEM_OP_NODE_PARAMS


        Adds batch mem op node parameters to output


    .. autoattribute:: cuda.bindings.driver.CUgraphDebugDot_flags.CU_GRAPH_DEBUG_DOT_FLAGS_EXTRA_TOPO_INFO


        Adds edge numbering information


    .. autoattribute:: cuda.bindings.driver.CUgraphDebugDot_flags.CU_GRAPH_DEBUG_DOT_FLAGS_CONDITIONAL_NODE_PARAMS


        Adds conditional node parameters to output

.. autoclass:: cuda.bindings.driver.CUuserObject_flags

    .. autoattribute:: cuda.bindings.driver.CUuserObject_flags.CU_USER_OBJECT_NO_DESTRUCTOR_SYNC


        Indicates the destructor execution is not synchronized by any CUDA handle.

.. autoclass:: cuda.bindings.driver.CUuserObjectRetain_flags

    .. autoattribute:: cuda.bindings.driver.CUuserObjectRetain_flags.CU_GRAPH_USER_OBJECT_MOVE


        Transfer references from the caller rather than creating new references.

.. autoclass:: cuda.bindings.driver.CUgraphInstantiate_flags

    .. autoattribute:: cuda.bindings.driver.CUgraphInstantiate_flags.CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH


        Automatically free memory allocated in a graph before relaunching.


    .. autoattribute:: cuda.bindings.driver.CUgraphInstantiate_flags.CUDA_GRAPH_INSTANTIATE_FLAG_UPLOAD


        Automatically upload the graph after instantiation. Only supported by :py:obj:`~.cuGraphInstantiateWithParams`. The upload will be performed using the stream provided in `instantiateParams`.


    .. autoattribute:: cuda.bindings.driver.CUgraphInstantiate_flags.CUDA_GRAPH_INSTANTIATE_FLAG_DEVICE_LAUNCH


        Instantiate the graph to be launchable from the device. This flag can only be used on platforms which support unified addressing. This flag cannot be used in conjunction with CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH.


    .. autoattribute:: cuda.bindings.driver.CUgraphInstantiate_flags.CUDA_GRAPH_INSTANTIATE_FLAG_USE_NODE_PRIORITY


        Run the graph using the per-node priority attributes rather than the priority of the stream it is launched into.

.. autoclass:: cuda.bindings.driver.CUdeviceNumaConfig

    .. autoattribute:: cuda.bindings.driver.CUdeviceNumaConfig.CU_DEVICE_NUMA_CONFIG_NONE


        The GPU is not a NUMA node


    .. autoattribute:: cuda.bindings.driver.CUdeviceNumaConfig.CU_DEVICE_NUMA_CONFIG_NUMA_NODE


        The GPU is a NUMA node, CU_DEVICE_ATTRIBUTE_NUMA_ID contains its NUMA ID

.. autoclass:: cuda.bindings.driver.CUprocessState

    .. autoattribute:: cuda.bindings.driver.CUprocessState.CU_PROCESS_STATE_RUNNING


        Default process state


    .. autoattribute:: cuda.bindings.driver.CUprocessState.CU_PROCESS_STATE_LOCKED


        CUDA API locks are taken so further CUDA API calls will block


    .. autoattribute:: cuda.bindings.driver.CUprocessState.CU_PROCESS_STATE_CHECKPOINTED


        Application memory contents have been checkpointed and underlying allocations and device handles have been released


    .. autoattribute:: cuda.bindings.driver.CUprocessState.CU_PROCESS_STATE_FAILED


        Application entered an uncorrectable error during the checkpoint/restore process

.. autoclass:: cuda.bindings.driver.CUeglFrameType

    .. autoattribute:: cuda.bindings.driver.CUeglFrameType.CU_EGL_FRAME_TYPE_ARRAY


        Frame type CUDA array


    .. autoattribute:: cuda.bindings.driver.CUeglFrameType.CU_EGL_FRAME_TYPE_PITCH


        Frame type pointer

.. autoclass:: cuda.bindings.driver.CUeglResourceLocationFlags

    .. autoattribute:: cuda.bindings.driver.CUeglResourceLocationFlags.CU_EGL_RESOURCE_LOCATION_SYSMEM


        Resource location sysmem


    .. autoattribute:: cuda.bindings.driver.CUeglResourceLocationFlags.CU_EGL_RESOURCE_LOCATION_VIDMEM


        Resource location vidmem

.. autoclass:: cuda.bindings.driver.CUeglColorFormat

    .. autoattribute:: cuda.bindings.driver.CUeglColorFormat.CU_EGL_COLOR_FORMAT_YUV420_PLANAR


        Y, U, V in three surfaces, each in a separate surface, U/V width = 1/2 Y width, U/V height = 1/2 Y height.


    .. autoattribute:: cuda.bindings.driver.CUeglColorFormat.CU_EGL_COLOR_FORMAT_YUV420_SEMIPLANAR


        Y, UV in two surfaces (UV as one surface) with VU byte ordering, width, height ratio same as YUV420Planar.


    .. autoattribute:: cuda.bindings.driver.CUeglColorFormat.CU_EGL_COLOR_FORMAT_YUV422_PLANAR


        Y, U, V each in a separate surface, U/V width = 1/2 Y width, U/V height = Y height.


    .. autoattribute:: cuda.bindings.driver.CUeglColorFormat.CU_EGL_COLOR_FORMAT_YUV422_SEMIPLANAR


        Y, UV in two surfaces with VU byte ordering, width, height ratio same as YUV422Planar.


    .. autoattribute:: cuda.bindings.driver.CUeglColorFormat.CU_EGL_COLOR_FORMAT_RGB


        R/G/B three channels in one surface with BGR byte ordering. Only pitch linear format supported.


    .. autoattribute:: cuda.bindings.driver.CUeglColorFormat.CU_EGL_COLOR_FORMAT_BGR


        R/G/B three channels in one surface with RGB byte ordering. Only pitch linear format supported.


    .. autoattribute:: cuda.bindings.driver.CUeglColorFormat.CU_EGL_COLOR_FORMAT_ARGB


        R/G/B/A four channels in one surface with BGRA byte ordering.


    .. autoattribute:: cuda.bindings.driver.CUeglColorFormat.CU_EGL_COLOR_FORMAT_RGBA


        R/G/B/A four channels in one surface with ABGR byte ordering.


    .. autoattribute:: cuda.bindings.driver.CUeglColorFormat.CU_EGL_COLOR_FORMAT_L


        single luminance channel in one surface.


    .. autoattribute:: cuda.bindings.driver.CUeglColorFormat.CU_EGL_COLOR_FORMAT_R


        single color channel in one surface.


    .. autoattribute:: cuda.bindings.driver.CUeglColorFormat.CU_EGL_COLOR_FORMAT_YUV444_PLANAR


        Y, U, V in three surfaces, each in a separate surface, U/V width = Y width, U/V height = Y height.


    .. autoattribute:: cuda.bindings.driver.CUeglColorFormat.CU_EGL_COLOR_FORMAT_YUV444_SEMIPLANAR


        Y, UV in two surfaces (UV as one surface) with VU byte ordering, width, height ratio same as YUV444Planar.


    .. autoattribute:: cuda.bindings.driver.CUeglColorFormat.CU_EGL_COLOR_FORMAT_YUYV_422


        Y, U, V in one surface, interleaved as UYVY in one channel.


    .. autoattribute:: cuda.bindings.driver.CUeglColorFormat.CU_EGL_COLOR_FORMAT_UYVY_422


        Y, U, V in one surface, interleaved as YUYV in one channel.


    .. autoattribute:: cuda.bindings.driver.CUeglColorFormat.CU_EGL_COLOR_FORMAT_ABGR


        R/G/B/A four channels in one surface with RGBA byte ordering.


    .. autoattribute:: cuda.bindings.driver.CUeglColorFormat.CU_EGL_COLOR_FORMAT_BGRA


        R/G/B/A four channels in one surface with ARGB byte ordering.


    .. autoattribute:: cuda.bindings.driver.CUeglColorFormat.CU_EGL_COLOR_FORMAT_A


        Alpha color format - one channel in one surface.


    .. autoattribute:: cuda.bindings.driver.CUeglColorFormat.CU_EGL_COLOR_FORMAT_RG


        R/G color format - two channels in one surface with GR byte ordering


    .. autoattribute:: cuda.bindings.driver.CUeglColorFormat.CU_EGL_COLOR_FORMAT_AYUV


        Y, U, V, A four channels in one surface, interleaved as VUYA.


    .. autoattribute:: cuda.bindings.driver.CUeglColorFormat.CU_EGL_COLOR_FORMAT_YVU444_SEMIPLANAR


        Y, VU in two surfaces (VU as one surface) with UV byte ordering, U/V width = Y width, U/V height = Y height.


    .. autoattribute:: cuda.bindings.driver.CUeglColorFormat.CU_EGL_COLOR_FORMAT_YVU422_SEMIPLANAR


        Y, VU in two surfaces (VU as one surface) with UV byte ordering, U/V width = 1/2 Y width, U/V height = Y height.


    .. autoattribute:: cuda.bindings.driver.CUeglColorFormat.CU_EGL_COLOR_FORMAT_YVU420_SEMIPLANAR


        Y, VU in two surfaces (VU as one surface) with UV byte ordering, U/V width = 1/2 Y width, U/V height = 1/2 Y height.


    .. autoattribute:: cuda.bindings.driver.CUeglColorFormat.CU_EGL_COLOR_FORMAT_Y10V10U10_444_SEMIPLANAR


        Y10, V10U10 in two surfaces (VU as one surface) with UV byte ordering, U/V width = Y width, U/V height = Y height.


    .. autoattribute:: cuda.bindings.driver.CUeglColorFormat.CU_EGL_COLOR_FORMAT_Y10V10U10_420_SEMIPLANAR


        Y10, V10U10 in two surfaces (VU as one surface) with UV byte ordering, U/V width = 1/2 Y width, U/V height = 1/2 Y height.


    .. autoattribute:: cuda.bindings.driver.CUeglColorFormat.CU_EGL_COLOR_FORMAT_Y12V12U12_444_SEMIPLANAR


        Y12, V12U12 in two surfaces (VU as one surface) with UV byte ordering, U/V width = Y width, U/V height = Y height.


    .. autoattribute:: cuda.bindings.driver.CUeglColorFormat.CU_EGL_COLOR_FORMAT_Y12V12U12_420_SEMIPLANAR


        Y12, V12U12 in two surfaces (VU as one surface) with UV byte ordering, U/V width = 1/2 Y width, U/V height = 1/2 Y height.


    .. autoattribute:: cuda.bindings.driver.CUeglColorFormat.CU_EGL_COLOR_FORMAT_VYUY_ER


        Extended Range Y, U, V in one surface, interleaved as YVYU in one channel.


    .. autoattribute:: cuda.bindings.driver.CUeglColorFormat.CU_EGL_COLOR_FORMAT_UYVY_ER


        Extended Range Y, U, V in one surface, interleaved as YUYV in one channel.


    .. autoattribute:: cuda.bindings.driver.CUeglColorFormat.CU_EGL_COLOR_FORMAT_YUYV_ER


        Extended Range Y, U, V in one surface, interleaved as UYVY in one channel.


    .. autoattribute:: cuda.bindings.driver.CUeglColorFormat.CU_EGL_COLOR_FORMAT_YVYU_ER


        Extended Range Y, U, V in one surface, interleaved as VYUY in one channel.


    .. autoattribute:: cuda.bindings.driver.CUeglColorFormat.CU_EGL_COLOR_FORMAT_YUV_ER


        Extended Range Y, U, V three channels in one surface, interleaved as VUY. Only pitch linear format supported.


    .. autoattribute:: cuda.bindings.driver.CUeglColorFormat.CU_EGL_COLOR_FORMAT_YUVA_ER


        Extended Range Y, U, V, A four channels in one surface, interleaved as AVUY.


    .. autoattribute:: cuda.bindings.driver.CUeglColorFormat.CU_EGL_COLOR_FORMAT_AYUV_ER


        Extended Range Y, U, V, A four channels in one surface, interleaved as VUYA.


    .. autoattribute:: cuda.bindings.driver.CUeglColorFormat.CU_EGL_COLOR_FORMAT_YUV444_PLANAR_ER


        Extended Range Y, U, V in three surfaces, U/V width = Y width, U/V height = Y height.


    .. autoattribute:: cuda.bindings.driver.CUeglColorFormat.CU_EGL_COLOR_FORMAT_YUV422_PLANAR_ER


        Extended Range Y, U, V in three surfaces, U/V width = 1/2 Y width, U/V height = Y height.


    .. autoattribute:: cuda.bindings.driver.CUeglColorFormat.CU_EGL_COLOR_FORMAT_YUV420_PLANAR_ER


        Extended Range Y, U, V in three surfaces, U/V width = 1/2 Y width, U/V height = 1/2 Y height.


    .. autoattribute:: cuda.bindings.driver.CUeglColorFormat.CU_EGL_COLOR_FORMAT_YUV444_SEMIPLANAR_ER


        Extended Range Y, UV in two surfaces (UV as one surface) with VU byte ordering, U/V width = Y width, U/V height = Y height.


    .. autoattribute:: cuda.bindings.driver.CUeglColorFormat.CU_EGL_COLOR_FORMAT_YUV422_SEMIPLANAR_ER


        Extended Range Y, UV in two surfaces (UV as one surface) with VU byte ordering, U/V width = 1/2 Y width, U/V height = Y height.


    .. autoattribute:: cuda.bindings.driver.CUeglColorFormat.CU_EGL_COLOR_FORMAT_YUV420_SEMIPLANAR_ER


        Extended Range Y, UV in two surfaces (UV as one surface) with VU byte ordering, U/V width = 1/2 Y width, U/V height = 1/2 Y height.


    .. autoattribute:: cuda.bindings.driver.CUeglColorFormat.CU_EGL_COLOR_FORMAT_YVU444_PLANAR_ER


        Extended Range Y, V, U in three surfaces, U/V width = Y width, U/V height = Y height.


    .. autoattribute:: cuda.bindings.driver.CUeglColorFormat.CU_EGL_COLOR_FORMAT_YVU422_PLANAR_ER


        Extended Range Y, V, U in three surfaces, U/V width = 1/2 Y width, U/V height = Y height.


    .. autoattribute:: cuda.bindings.driver.CUeglColorFormat.CU_EGL_COLOR_FORMAT_YVU420_PLANAR_ER


        Extended Range Y, V, U in three surfaces, U/V width = 1/2 Y width, U/V height = 1/2 Y height.


    .. autoattribute:: cuda.bindings.driver.CUeglColorFormat.CU_EGL_COLOR_FORMAT_YVU444_SEMIPLANAR_ER


        Extended Range Y, VU in two surfaces (VU as one surface) with UV byte ordering, U/V width = Y width, U/V height = Y height.


    .. autoattribute:: cuda.bindings.driver.CUeglColorFormat.CU_EGL_COLOR_FORMAT_YVU422_SEMIPLANAR_ER


        Extended Range Y, VU in two surfaces (VU as one surface) with UV byte ordering, U/V width = 1/2 Y width, U/V height = Y height.


    .. autoattribute:: cuda.bindings.driver.CUeglColorFormat.CU_EGL_COLOR_FORMAT_YVU420_SEMIPLANAR_ER


        Extended Range Y, VU in two surfaces (VU as one surface) with UV byte ordering, U/V width = 1/2 Y width, U/V height = 1/2 Y height.


    .. autoattribute:: cuda.bindings.driver.CUeglColorFormat.CU_EGL_COLOR_FORMAT_BAYER_RGGB


        Bayer format - one channel in one surface with interleaved RGGB ordering.


    .. autoattribute:: cuda.bindings.driver.CUeglColorFormat.CU_EGL_COLOR_FORMAT_BAYER_BGGR


        Bayer format - one channel in one surface with interleaved BGGR ordering.


    .. autoattribute:: cuda.bindings.driver.CUeglColorFormat.CU_EGL_COLOR_FORMAT_BAYER_GRBG


        Bayer format - one channel in one surface with interleaved GRBG ordering.


    .. autoattribute:: cuda.bindings.driver.CUeglColorFormat.CU_EGL_COLOR_FORMAT_BAYER_GBRG


        Bayer format - one channel in one surface with interleaved GBRG ordering.


    .. autoattribute:: cuda.bindings.driver.CUeglColorFormat.CU_EGL_COLOR_FORMAT_BAYER10_RGGB


        Bayer10 format - one channel in one surface with interleaved RGGB ordering. Out of 16 bits, 10 bits used 6 bits No-op.


    .. autoattribute:: cuda.bindings.driver.CUeglColorFormat.CU_EGL_COLOR_FORMAT_BAYER10_BGGR


        Bayer10 format - one channel in one surface with interleaved BGGR ordering. Out of 16 bits, 10 bits used 6 bits No-op.


    .. autoattribute:: cuda.bindings.driver.CUeglColorFormat.CU_EGL_COLOR_FORMAT_BAYER10_GRBG


        Bayer10 format - one channel in one surface with interleaved GRBG ordering. Out of 16 bits, 10 bits used 6 bits No-op.


    .. autoattribute:: cuda.bindings.driver.CUeglColorFormat.CU_EGL_COLOR_FORMAT_BAYER10_GBRG


        Bayer10 format - one channel in one surface with interleaved GBRG ordering. Out of 16 bits, 10 bits used 6 bits No-op.


    .. autoattribute:: cuda.bindings.driver.CUeglColorFormat.CU_EGL_COLOR_FORMAT_BAYER12_RGGB


        Bayer12 format - one channel in one surface with interleaved RGGB ordering. Out of 16 bits, 12 bits used 4 bits No-op.


    .. autoattribute:: cuda.bindings.driver.CUeglColorFormat.CU_EGL_COLOR_FORMAT_BAYER12_BGGR


        Bayer12 format - one channel in one surface with interleaved BGGR ordering. Out of 16 bits, 12 bits used 4 bits No-op.


    .. autoattribute:: cuda.bindings.driver.CUeglColorFormat.CU_EGL_COLOR_FORMAT_BAYER12_GRBG


        Bayer12 format - one channel in one surface with interleaved GRBG ordering. Out of 16 bits, 12 bits used 4 bits No-op.


    .. autoattribute:: cuda.bindings.driver.CUeglColorFormat.CU_EGL_COLOR_FORMAT_BAYER12_GBRG


        Bayer12 format - one channel in one surface with interleaved GBRG ordering. Out of 16 bits, 12 bits used 4 bits No-op.


    .. autoattribute:: cuda.bindings.driver.CUeglColorFormat.CU_EGL_COLOR_FORMAT_BAYER14_RGGB


        Bayer14 format - one channel in one surface with interleaved RGGB ordering. Out of 16 bits, 14 bits used 2 bits No-op.


    .. autoattribute:: cuda.bindings.driver.CUeglColorFormat.CU_EGL_COLOR_FORMAT_BAYER14_BGGR


        Bayer14 format - one channel in one surface with interleaved BGGR ordering. Out of 16 bits, 14 bits used 2 bits No-op.


    .. autoattribute:: cuda.bindings.driver.CUeglColorFormat.CU_EGL_COLOR_FORMAT_BAYER14_GRBG


        Bayer14 format - one channel in one surface with interleaved GRBG ordering. Out of 16 bits, 14 bits used 2 bits No-op.


    .. autoattribute:: cuda.bindings.driver.CUeglColorFormat.CU_EGL_COLOR_FORMAT_BAYER14_GBRG


        Bayer14 format - one channel in one surface with interleaved GBRG ordering. Out of 16 bits, 14 bits used 2 bits No-op.


    .. autoattribute:: cuda.bindings.driver.CUeglColorFormat.CU_EGL_COLOR_FORMAT_BAYER20_RGGB


        Bayer20 format - one channel in one surface with interleaved RGGB ordering. Out of 32 bits, 20 bits used 12 bits No-op.


    .. autoattribute:: cuda.bindings.driver.CUeglColorFormat.CU_EGL_COLOR_FORMAT_BAYER20_BGGR


        Bayer20 format - one channel in one surface with interleaved BGGR ordering. Out of 32 bits, 20 bits used 12 bits No-op.


    .. autoattribute:: cuda.bindings.driver.CUeglColorFormat.CU_EGL_COLOR_FORMAT_BAYER20_GRBG


        Bayer20 format - one channel in one surface with interleaved GRBG ordering. Out of 32 bits, 20 bits used 12 bits No-op.


    .. autoattribute:: cuda.bindings.driver.CUeglColorFormat.CU_EGL_COLOR_FORMAT_BAYER20_GBRG


        Bayer20 format - one channel in one surface with interleaved GBRG ordering. Out of 32 bits, 20 bits used 12 bits No-op.


    .. autoattribute:: cuda.bindings.driver.CUeglColorFormat.CU_EGL_COLOR_FORMAT_YVU444_PLANAR


        Y, V, U in three surfaces, each in a separate surface, U/V width = Y width, U/V height = Y height.


    .. autoattribute:: cuda.bindings.driver.CUeglColorFormat.CU_EGL_COLOR_FORMAT_YVU422_PLANAR


        Y, V, U in three surfaces, each in a separate surface, U/V width = 1/2 Y width, U/V height = Y height.


    .. autoattribute:: cuda.bindings.driver.CUeglColorFormat.CU_EGL_COLOR_FORMAT_YVU420_PLANAR


        Y, V, U in three surfaces, each in a separate surface, U/V width = 1/2 Y width, U/V height = 1/2 Y height.


    .. autoattribute:: cuda.bindings.driver.CUeglColorFormat.CU_EGL_COLOR_FORMAT_BAYER_ISP_RGGB


        Nvidia proprietary Bayer ISP format - one channel in one surface with interleaved RGGB ordering and mapped to opaque integer datatype.


    .. autoattribute:: cuda.bindings.driver.CUeglColorFormat.CU_EGL_COLOR_FORMAT_BAYER_ISP_BGGR


        Nvidia proprietary Bayer ISP format - one channel in one surface with interleaved BGGR ordering and mapped to opaque integer datatype.


    .. autoattribute:: cuda.bindings.driver.CUeglColorFormat.CU_EGL_COLOR_FORMAT_BAYER_ISP_GRBG


        Nvidia proprietary Bayer ISP format - one channel in one surface with interleaved GRBG ordering and mapped to opaque integer datatype.


    .. autoattribute:: cuda.bindings.driver.CUeglColorFormat.CU_EGL_COLOR_FORMAT_BAYER_ISP_GBRG


        Nvidia proprietary Bayer ISP format - one channel in one surface with interleaved GBRG ordering and mapped to opaque integer datatype.


    .. autoattribute:: cuda.bindings.driver.CUeglColorFormat.CU_EGL_COLOR_FORMAT_BAYER_BCCR


        Bayer format - one channel in one surface with interleaved BCCR ordering.


    .. autoattribute:: cuda.bindings.driver.CUeglColorFormat.CU_EGL_COLOR_FORMAT_BAYER_RCCB


        Bayer format - one channel in one surface with interleaved RCCB ordering.


    .. autoattribute:: cuda.bindings.driver.CUeglColorFormat.CU_EGL_COLOR_FORMAT_BAYER_CRBC


        Bayer format - one channel in one surface with interleaved CRBC ordering.


    .. autoattribute:: cuda.bindings.driver.CUeglColorFormat.CU_EGL_COLOR_FORMAT_BAYER_CBRC


        Bayer format - one channel in one surface with interleaved CBRC ordering.


    .. autoattribute:: cuda.bindings.driver.CUeglColorFormat.CU_EGL_COLOR_FORMAT_BAYER10_CCCC


        Bayer10 format - one channel in one surface with interleaved CCCC ordering. Out of 16 bits, 10 bits used 6 bits No-op.


    .. autoattribute:: cuda.bindings.driver.CUeglColorFormat.CU_EGL_COLOR_FORMAT_BAYER12_BCCR


        Bayer12 format - one channel in one surface with interleaved BCCR ordering. Out of 16 bits, 12 bits used 4 bits No-op.


    .. autoattribute:: cuda.bindings.driver.CUeglColorFormat.CU_EGL_COLOR_FORMAT_BAYER12_RCCB


        Bayer12 format - one channel in one surface with interleaved RCCB ordering. Out of 16 bits, 12 bits used 4 bits No-op.


    .. autoattribute:: cuda.bindings.driver.CUeglColorFormat.CU_EGL_COLOR_FORMAT_BAYER12_CRBC


        Bayer12 format - one channel in one surface with interleaved CRBC ordering. Out of 16 bits, 12 bits used 4 bits No-op.


    .. autoattribute:: cuda.bindings.driver.CUeglColorFormat.CU_EGL_COLOR_FORMAT_BAYER12_CBRC


        Bayer12 format - one channel in one surface with interleaved CBRC ordering. Out of 16 bits, 12 bits used 4 bits No-op.


    .. autoattribute:: cuda.bindings.driver.CUeglColorFormat.CU_EGL_COLOR_FORMAT_BAYER12_CCCC


        Bayer12 format - one channel in one surface with interleaved CCCC ordering. Out of 16 bits, 12 bits used 4 bits No-op.


    .. autoattribute:: cuda.bindings.driver.CUeglColorFormat.CU_EGL_COLOR_FORMAT_Y


        Color format for single Y plane.


    .. autoattribute:: cuda.bindings.driver.CUeglColorFormat.CU_EGL_COLOR_FORMAT_YUV420_SEMIPLANAR_2020


        Y, UV in two surfaces (UV as one surface) U/V width = 1/2 Y width, U/V height = 1/2 Y height.


    .. autoattribute:: cuda.bindings.driver.CUeglColorFormat.CU_EGL_COLOR_FORMAT_YVU420_SEMIPLANAR_2020


        Y, VU in two surfaces (VU as one surface) U/V width = 1/2 Y width, U/V height = 1/2 Y height.


    .. autoattribute:: cuda.bindings.driver.CUeglColorFormat.CU_EGL_COLOR_FORMAT_YUV420_PLANAR_2020


        Y, U, V each in a separate surface, U/V width = 1/2 Y width, U/V height= 1/2 Y height.


    .. autoattribute:: cuda.bindings.driver.CUeglColorFormat.CU_EGL_COLOR_FORMAT_YVU420_PLANAR_2020


        Y, V, U each in a separate surface, U/V width = 1/2 Y width, U/V height = 1/2 Y height.


    .. autoattribute:: cuda.bindings.driver.CUeglColorFormat.CU_EGL_COLOR_FORMAT_YUV420_SEMIPLANAR_709


        Y, UV in two surfaces (UV as one surface) U/V width = 1/2 Y width, U/V height = 1/2 Y height.


    .. autoattribute:: cuda.bindings.driver.CUeglColorFormat.CU_EGL_COLOR_FORMAT_YVU420_SEMIPLANAR_709


        Y, VU in two surfaces (VU as one surface) U/V width = 1/2 Y width, U/V height = 1/2 Y height.


    .. autoattribute:: cuda.bindings.driver.CUeglColorFormat.CU_EGL_COLOR_FORMAT_YUV420_PLANAR_709


        Y, U, V each in a separate surface, U/V width = 1/2 Y width, U/V height = 1/2 Y height.


    .. autoattribute:: cuda.bindings.driver.CUeglColorFormat.CU_EGL_COLOR_FORMAT_YVU420_PLANAR_709


        Y, V, U each in a separate surface, U/V width = 1/2 Y width, U/V height = 1/2 Y height.


    .. autoattribute:: cuda.bindings.driver.CUeglColorFormat.CU_EGL_COLOR_FORMAT_Y10V10U10_420_SEMIPLANAR_709


        Y10, V10U10 in two surfaces (VU as one surface), U/V width = 1/2 Y width, U/V height = 1/2 Y height.


    .. autoattribute:: cuda.bindings.driver.CUeglColorFormat.CU_EGL_COLOR_FORMAT_Y10V10U10_420_SEMIPLANAR_2020


        Y10, V10U10 in two surfaces (VU as one surface), U/V width = 1/2 Y width, U/V height = 1/2 Y height.


    .. autoattribute:: cuda.bindings.driver.CUeglColorFormat.CU_EGL_COLOR_FORMAT_Y10V10U10_422_SEMIPLANAR_2020


        Y10, V10U10 in two surfaces(VU as one surface) U/V width = 1/2 Y width, U/V height = Y height.


    .. autoattribute:: cuda.bindings.driver.CUeglColorFormat.CU_EGL_COLOR_FORMAT_Y10V10U10_422_SEMIPLANAR


        Y10, V10U10 in two surfaces(VU as one surface) U/V width = 1/2 Y width, U/V height = Y height.


    .. autoattribute:: cuda.bindings.driver.CUeglColorFormat.CU_EGL_COLOR_FORMAT_Y10V10U10_422_SEMIPLANAR_709


        Y10, V10U10 in two surfaces(VU as one surface) U/V width = 1/2 Y width, U/V height = Y height.


    .. autoattribute:: cuda.bindings.driver.CUeglColorFormat.CU_EGL_COLOR_FORMAT_Y_ER


        Extended Range Color format for single Y plane.


    .. autoattribute:: cuda.bindings.driver.CUeglColorFormat.CU_EGL_COLOR_FORMAT_Y_709_ER


        Extended Range Color format for single Y plane.


    .. autoattribute:: cuda.bindings.driver.CUeglColorFormat.CU_EGL_COLOR_FORMAT_Y10_ER


        Extended Range Color format for single Y10 plane.


    .. autoattribute:: cuda.bindings.driver.CUeglColorFormat.CU_EGL_COLOR_FORMAT_Y10_709_ER


        Extended Range Color format for single Y10 plane.


    .. autoattribute:: cuda.bindings.driver.CUeglColorFormat.CU_EGL_COLOR_FORMAT_Y12_ER


        Extended Range Color format for single Y12 plane.


    .. autoattribute:: cuda.bindings.driver.CUeglColorFormat.CU_EGL_COLOR_FORMAT_Y12_709_ER


        Extended Range Color format for single Y12 plane.


    .. autoattribute:: cuda.bindings.driver.CUeglColorFormat.CU_EGL_COLOR_FORMAT_YUVA


        Y, U, V, A four channels in one surface, interleaved as AVUY.


    .. autoattribute:: cuda.bindings.driver.CUeglColorFormat.CU_EGL_COLOR_FORMAT_YUV


        Y, U, V three channels in one surface, interleaved as VUY. Only pitch linear format supported.


    .. autoattribute:: cuda.bindings.driver.CUeglColorFormat.CU_EGL_COLOR_FORMAT_YVYU


        Y, U, V in one surface, interleaved as YVYU in one channel.


    .. autoattribute:: cuda.bindings.driver.CUeglColorFormat.CU_EGL_COLOR_FORMAT_VYUY


        Y, U, V in one surface, interleaved as VYUY in one channel.


    .. autoattribute:: cuda.bindings.driver.CUeglColorFormat.CU_EGL_COLOR_FORMAT_Y10V10U10_420_SEMIPLANAR_ER


        Extended Range Y10, V10U10 in two surfaces(VU as one surface) U/V width = 1/2 Y width, U/V height = 1/2 Y height.


    .. autoattribute:: cuda.bindings.driver.CUeglColorFormat.CU_EGL_COLOR_FORMAT_Y10V10U10_420_SEMIPLANAR_709_ER


        Extended Range Y10, V10U10 in two surfaces(VU as one surface) U/V width = 1/2 Y width, U/V height = 1/2 Y height.


    .. autoattribute:: cuda.bindings.driver.CUeglColorFormat.CU_EGL_COLOR_FORMAT_Y10V10U10_444_SEMIPLANAR_ER


        Extended Range Y10, V10U10 in two surfaces (VU as one surface) U/V width = Y width, U/V height = Y height.


    .. autoattribute:: cuda.bindings.driver.CUeglColorFormat.CU_EGL_COLOR_FORMAT_Y10V10U10_444_SEMIPLANAR_709_ER


        Extended Range Y10, V10U10 in two surfaces (VU as one surface) U/V width = Y width, U/V height = Y height.


    .. autoattribute:: cuda.bindings.driver.CUeglColorFormat.CU_EGL_COLOR_FORMAT_Y12V12U12_420_SEMIPLANAR_ER


        Extended Range Y12, V12U12 in two surfaces (VU as one surface) U/V width = 1/2 Y width, U/V height = 1/2 Y height.


    .. autoattribute:: cuda.bindings.driver.CUeglColorFormat.CU_EGL_COLOR_FORMAT_Y12V12U12_420_SEMIPLANAR_709_ER


        Extended Range Y12, V12U12 in two surfaces (VU as one surface) U/V width = 1/2 Y width, U/V height = 1/2 Y height.


    .. autoattribute:: cuda.bindings.driver.CUeglColorFormat.CU_EGL_COLOR_FORMAT_Y12V12U12_444_SEMIPLANAR_ER


        Extended Range Y12, V12U12 in two surfaces (VU as one surface) U/V width = Y width, U/V height = Y height.


    .. autoattribute:: cuda.bindings.driver.CUeglColorFormat.CU_EGL_COLOR_FORMAT_Y12V12U12_444_SEMIPLANAR_709_ER


        Extended Range Y12, V12U12 in two surfaces (VU as one surface) U/V width = Y width, U/V height = Y height.


    .. autoattribute:: cuda.bindings.driver.CUeglColorFormat.CU_EGL_COLOR_FORMAT_UYVY_709


        Y, U, V in one surface, interleaved as UYVY in one channel.


    .. autoattribute:: cuda.bindings.driver.CUeglColorFormat.CU_EGL_COLOR_FORMAT_UYVY_709_ER


        Extended Range Y, U, V in one surface, interleaved as UYVY in one channel.


    .. autoattribute:: cuda.bindings.driver.CUeglColorFormat.CU_EGL_COLOR_FORMAT_UYVY_2020


        Y, U, V in one surface, interleaved as UYVY in one channel.


    .. autoattribute:: cuda.bindings.driver.CUeglColorFormat.CU_EGL_COLOR_FORMAT_MAX

.. autoclass:: cuda.bindings.driver.CUdeviceptr_v2
.. autoclass:: cuda.bindings.driver.CUdeviceptr
.. autoclass:: cuda.bindings.driver.CUdevice_v1
.. autoclass:: cuda.bindings.driver.CUdevice
.. autoclass:: cuda.bindings.driver.CUcontext
.. autoclass:: cuda.bindings.driver.CUmodule
.. autoclass:: cuda.bindings.driver.CUfunction
.. autoclass:: cuda.bindings.driver.CUlibrary
.. autoclass:: cuda.bindings.driver.CUkernel
.. autoclass:: cuda.bindings.driver.CUarray
.. autoclass:: cuda.bindings.driver.CUmipmappedArray
.. autoclass:: cuda.bindings.driver.CUtexref
.. autoclass:: cuda.bindings.driver.CUsurfref
.. autoclass:: cuda.bindings.driver.CUevent
.. autoclass:: cuda.bindings.driver.CUstream
.. autoclass:: cuda.bindings.driver.CUgraphicsResource
.. autoclass:: cuda.bindings.driver.CUtexObject_v1
.. autoclass:: cuda.bindings.driver.CUtexObject
.. autoclass:: cuda.bindings.driver.CUsurfObject_v1
.. autoclass:: cuda.bindings.driver.CUsurfObject
.. autoclass:: cuda.bindings.driver.CUexternalMemory
.. autoclass:: cuda.bindings.driver.CUexternalSemaphore
.. autoclass:: cuda.bindings.driver.CUgraph
.. autoclass:: cuda.bindings.driver.CUgraphNode
.. autoclass:: cuda.bindings.driver.CUgraphExec
.. autoclass:: cuda.bindings.driver.CUmemoryPool
.. autoclass:: cuda.bindings.driver.CUuserObject
.. autoclass:: cuda.bindings.driver.CUgraphConditionalHandle
.. autoclass:: cuda.bindings.driver.CUgraphDeviceNode
.. autoclass:: cuda.bindings.driver.CUasyncCallbackHandle
.. autoclass:: cuda.bindings.driver.CUgreenCtx
.. autoclass:: cuda.bindings.driver.CUuuid
.. autoclass:: cuda.bindings.driver.CUmemFabricHandle_v1
.. autoclass:: cuda.bindings.driver.CUmemFabricHandle
.. autoclass:: cuda.bindings.driver.CUipcEventHandle_v1
.. autoclass:: cuda.bindings.driver.CUipcEventHandle
.. autoclass:: cuda.bindings.driver.CUipcMemHandle_v1
.. autoclass:: cuda.bindings.driver.CUipcMemHandle
.. autoclass:: cuda.bindings.driver.CUstreamBatchMemOpParams_v1
.. autoclass:: cuda.bindings.driver.CUstreamBatchMemOpParams
.. autoclass:: cuda.bindings.driver.CUDA_BATCH_MEM_OP_NODE_PARAMS_v1
.. autoclass:: cuda.bindings.driver.CUDA_BATCH_MEM_OP_NODE_PARAMS
.. autoclass:: cuda.bindings.driver.CUDA_BATCH_MEM_OP_NODE_PARAMS_v2
.. autoclass:: cuda.bindings.driver.CUasyncNotificationInfo
.. autoclass:: cuda.bindings.driver.CUasyncCallback
.. autoclass:: cuda.bindings.driver.CUdevprop_v1
.. autoclass:: cuda.bindings.driver.CUdevprop
.. autoclass:: cuda.bindings.driver.CUlinkState
.. autoclass:: cuda.bindings.driver.CUhostFn
.. autoclass:: cuda.bindings.driver.CUaccessPolicyWindow_v1
.. autoclass:: cuda.bindings.driver.CUaccessPolicyWindow
.. autoclass:: cuda.bindings.driver.CUDA_KERNEL_NODE_PARAMS_v1
.. autoclass:: cuda.bindings.driver.CUDA_KERNEL_NODE_PARAMS_v2
.. autoclass:: cuda.bindings.driver.CUDA_KERNEL_NODE_PARAMS
.. autoclass:: cuda.bindings.driver.CUDA_KERNEL_NODE_PARAMS_v3
.. autoclass:: cuda.bindings.driver.CUDA_MEMSET_NODE_PARAMS_v1
.. autoclass:: cuda.bindings.driver.CUDA_MEMSET_NODE_PARAMS
.. autoclass:: cuda.bindings.driver.CUDA_MEMSET_NODE_PARAMS_v2
.. autoclass:: cuda.bindings.driver.CUDA_HOST_NODE_PARAMS_v1
.. autoclass:: cuda.bindings.driver.CUDA_HOST_NODE_PARAMS
.. autoclass:: cuda.bindings.driver.CUDA_HOST_NODE_PARAMS_v2
.. autoclass:: cuda.bindings.driver.CUDA_CONDITIONAL_NODE_PARAMS
.. autoclass:: cuda.bindings.driver.CUgraphEdgeData
.. autoclass:: cuda.bindings.driver.CUDA_GRAPH_INSTANTIATE_PARAMS
.. autoclass:: cuda.bindings.driver.CUlaunchMemSyncDomainMap
.. autoclass:: cuda.bindings.driver.CUlaunchAttributeValue
.. autoclass:: cuda.bindings.driver.CUlaunchAttribute
.. autoclass:: cuda.bindings.driver.CUlaunchConfig
.. autoclass:: cuda.bindings.driver.CUkernelNodeAttrID
.. autoclass:: cuda.bindings.driver.CUkernelNodeAttrValue_v1
.. autoclass:: cuda.bindings.driver.CUkernelNodeAttrValue
.. autoclass:: cuda.bindings.driver.CUstreamAttrID
.. autoclass:: cuda.bindings.driver.CUstreamAttrValue_v1
.. autoclass:: cuda.bindings.driver.CUstreamAttrValue
.. autoclass:: cuda.bindings.driver.CUexecAffinitySmCount_v1
.. autoclass:: cuda.bindings.driver.CUexecAffinitySmCount
.. autoclass:: cuda.bindings.driver.CUexecAffinityParam_v1
.. autoclass:: cuda.bindings.driver.CUexecAffinityParam
.. autoclass:: cuda.bindings.driver.CUctxCigParam
.. autoclass:: cuda.bindings.driver.CUctxCreateParams
.. autoclass:: cuda.bindings.driver.CUlibraryHostUniversalFunctionAndDataTable
.. autoclass:: cuda.bindings.driver.CUstreamCallback
.. autoclass:: cuda.bindings.driver.CUoccupancyB2DSize
.. autoclass:: cuda.bindings.driver.CUDA_MEMCPY2D_v2
.. autoclass:: cuda.bindings.driver.CUDA_MEMCPY2D
.. autoclass:: cuda.bindings.driver.CUDA_MEMCPY3D_v2
.. autoclass:: cuda.bindings.driver.CUDA_MEMCPY3D
.. autoclass:: cuda.bindings.driver.CUDA_MEMCPY3D_PEER_v1
.. autoclass:: cuda.bindings.driver.CUDA_MEMCPY3D_PEER
.. autoclass:: cuda.bindings.driver.CUDA_MEMCPY_NODE_PARAMS
.. autoclass:: cuda.bindings.driver.CUDA_ARRAY_DESCRIPTOR_v2
.. autoclass:: cuda.bindings.driver.CUDA_ARRAY_DESCRIPTOR
.. autoclass:: cuda.bindings.driver.CUDA_ARRAY3D_DESCRIPTOR_v2
.. autoclass:: cuda.bindings.driver.CUDA_ARRAY3D_DESCRIPTOR
.. autoclass:: cuda.bindings.driver.CUDA_ARRAY_SPARSE_PROPERTIES_v1
.. autoclass:: cuda.bindings.driver.CUDA_ARRAY_SPARSE_PROPERTIES
.. autoclass:: cuda.bindings.driver.CUDA_ARRAY_MEMORY_REQUIREMENTS_v1
.. autoclass:: cuda.bindings.driver.CUDA_ARRAY_MEMORY_REQUIREMENTS
.. autoclass:: cuda.bindings.driver.CUDA_RESOURCE_DESC_v1
.. autoclass:: cuda.bindings.driver.CUDA_RESOURCE_DESC
.. autoclass:: cuda.bindings.driver.CUDA_TEXTURE_DESC_v1
.. autoclass:: cuda.bindings.driver.CUDA_TEXTURE_DESC
.. autoclass:: cuda.bindings.driver.CUDA_RESOURCE_VIEW_DESC_v1
.. autoclass:: cuda.bindings.driver.CUDA_RESOURCE_VIEW_DESC
.. autoclass:: cuda.bindings.driver.CUtensorMap
.. autoclass:: cuda.bindings.driver.CUDA_POINTER_ATTRIBUTE_P2P_TOKENS_v1
.. autoclass:: cuda.bindings.driver.CUDA_POINTER_ATTRIBUTE_P2P_TOKENS
.. autoclass:: cuda.bindings.driver.CUDA_LAUNCH_PARAMS_v1
.. autoclass:: cuda.bindings.driver.CUDA_LAUNCH_PARAMS
.. autoclass:: cuda.bindings.driver.CUDA_EXTERNAL_MEMORY_HANDLE_DESC_v1
.. autoclass:: cuda.bindings.driver.CUDA_EXTERNAL_MEMORY_HANDLE_DESC
.. autoclass:: cuda.bindings.driver.CUDA_EXTERNAL_MEMORY_BUFFER_DESC_v1
.. autoclass:: cuda.bindings.driver.CUDA_EXTERNAL_MEMORY_BUFFER_DESC
.. autoclass:: cuda.bindings.driver.CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC_v1
.. autoclass:: cuda.bindings.driver.CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC
.. autoclass:: cuda.bindings.driver.CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_v1
.. autoclass:: cuda.bindings.driver.CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC
.. autoclass:: cuda.bindings.driver.CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_v1
.. autoclass:: cuda.bindings.driver.CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS
.. autoclass:: cuda.bindings.driver.CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_v1
.. autoclass:: cuda.bindings.driver.CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS
.. autoclass:: cuda.bindings.driver.CUDA_EXT_SEM_SIGNAL_NODE_PARAMS_v1
.. autoclass:: cuda.bindings.driver.CUDA_EXT_SEM_SIGNAL_NODE_PARAMS
.. autoclass:: cuda.bindings.driver.CUDA_EXT_SEM_SIGNAL_NODE_PARAMS_v2
.. autoclass:: cuda.bindings.driver.CUDA_EXT_SEM_WAIT_NODE_PARAMS_v1
.. autoclass:: cuda.bindings.driver.CUDA_EXT_SEM_WAIT_NODE_PARAMS
.. autoclass:: cuda.bindings.driver.CUDA_EXT_SEM_WAIT_NODE_PARAMS_v2
.. autoclass:: cuda.bindings.driver.CUmemGenericAllocationHandle_v1
.. autoclass:: cuda.bindings.driver.CUmemGenericAllocationHandle
.. autoclass:: cuda.bindings.driver.CUarrayMapInfo_v1
.. autoclass:: cuda.bindings.driver.CUarrayMapInfo
.. autoclass:: cuda.bindings.driver.CUmemLocation_v1
.. autoclass:: cuda.bindings.driver.CUmemLocation
.. autoclass:: cuda.bindings.driver.CUmemAllocationProp_v1
.. autoclass:: cuda.bindings.driver.CUmemAllocationProp
.. autoclass:: cuda.bindings.driver.CUmulticastObjectProp_v1
.. autoclass:: cuda.bindings.driver.CUmulticastObjectProp
.. autoclass:: cuda.bindings.driver.CUmemAccessDesc_v1
.. autoclass:: cuda.bindings.driver.CUmemAccessDesc
.. autoclass:: cuda.bindings.driver.CUgraphExecUpdateResultInfo_v1
.. autoclass:: cuda.bindings.driver.CUgraphExecUpdateResultInfo
.. autoclass:: cuda.bindings.driver.CUmemPoolProps_v1
.. autoclass:: cuda.bindings.driver.CUmemPoolProps
.. autoclass:: cuda.bindings.driver.CUmemPoolPtrExportData_v1
.. autoclass:: cuda.bindings.driver.CUmemPoolPtrExportData
.. autoclass:: cuda.bindings.driver.CUmemcpyAttributes_v1
.. autoclass:: cuda.bindings.driver.CUmemcpyAttributes
.. autoclass:: cuda.bindings.driver.CUoffset3D_v1
.. autoclass:: cuda.bindings.driver.CUoffset3D
.. autoclass:: cuda.bindings.driver.CUextent3D_v1
.. autoclass:: cuda.bindings.driver.CUextent3D
.. autoclass:: cuda.bindings.driver.CUmemcpy3DOperand_v1
.. autoclass:: cuda.bindings.driver.CUmemcpy3DOperand
.. autoclass:: cuda.bindings.driver.CUDA_MEMCPY3D_BATCH_OP_v1
.. autoclass:: cuda.bindings.driver.CUDA_MEMCPY3D_BATCH_OP
.. autoclass:: cuda.bindings.driver.CUDA_MEM_ALLOC_NODE_PARAMS_v1
.. autoclass:: cuda.bindings.driver.CUDA_MEM_ALLOC_NODE_PARAMS
.. autoclass:: cuda.bindings.driver.CUDA_MEM_ALLOC_NODE_PARAMS_v2
.. autoclass:: cuda.bindings.driver.CUDA_MEM_FREE_NODE_PARAMS
.. autoclass:: cuda.bindings.driver.CUDA_CHILD_GRAPH_NODE_PARAMS
.. autoclass:: cuda.bindings.driver.CUDA_EVENT_RECORD_NODE_PARAMS
.. autoclass:: cuda.bindings.driver.CUDA_EVENT_WAIT_NODE_PARAMS
.. autoclass:: cuda.bindings.driver.CUgraphNodeParams
.. autoclass:: cuda.bindings.driver.CUcheckpointLockArgs
.. autoclass:: cuda.bindings.driver.CUcheckpointCheckpointArgs
.. autoclass:: cuda.bindings.driver.CUcheckpointGpuPair
.. autoclass:: cuda.bindings.driver.CUcheckpointRestoreArgs
.. autoclass:: cuda.bindings.driver.CUcheckpointUnlockArgs
.. autoclass:: cuda.bindings.driver.CUeglFrame_v1
.. autoclass:: cuda.bindings.driver.CUeglFrame
.. autoclass:: cuda.bindings.driver.CUeglStreamConnection
.. autoattribute:: cuda.bindings.driver.CUDA_VERSION

    CUDA API version number

.. autoattribute:: cuda.bindings.driver.CU_UUID_HAS_BEEN_DEFINED

    CUDA UUID types

.. autoattribute:: cuda.bindings.driver.CU_IPC_HANDLE_SIZE

    CUDA IPC handle size

.. autoattribute:: cuda.bindings.driver.CU_STREAM_LEGACY

    Legacy stream handle



    Stream handle that can be passed as a CUstream to use an implicit stream with legacy synchronization behavior.



    See details of the \link_sync_behavior

.. autoattribute:: cuda.bindings.driver.CU_STREAM_PER_THREAD

    Per-thread stream handle



    Stream handle that can be passed as a CUstream to use an implicit stream with per-thread synchronization behavior.



    See details of the \link_sync_behavior

.. autoattribute:: cuda.bindings.driver.CU_COMPUTE_ACCELERATED_TARGET_BASE
.. autoattribute:: cuda.bindings.driver.CU_COMPUTE_FAMILY_TARGET_BASE
.. autoattribute:: cuda.bindings.driver.CUDA_CB
.. autoattribute:: cuda.bindings.driver.CU_GRAPH_COND_ASSIGN_DEFAULT

    Conditional node handle flags Default value is applied when graph is launched.

.. autoattribute:: cuda.bindings.driver.CU_GRAPH_KERNEL_NODE_PORT_DEFAULT

    This port activates when the kernel has finished executing.

.. autoattribute:: cuda.bindings.driver.CU_GRAPH_KERNEL_NODE_PORT_PROGRAMMATIC

    This port activates when all blocks of the kernel have performed cudaTriggerProgrammaticLaunchCompletion() or have terminated. It must be used with edge type :py:obj:`~.CU_GRAPH_DEPENDENCY_TYPE_PROGRAMMATIC`. See also :py:obj:`~.CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_EVENT`.

.. autoattribute:: cuda.bindings.driver.CU_GRAPH_KERNEL_NODE_PORT_LAUNCH_ORDER

    This port activates when all blocks of the kernel have begun execution. See also :py:obj:`~.CU_LAUNCH_ATTRIBUTE_LAUNCH_COMPLETION_EVENT`.

.. autoattribute:: cuda.bindings.driver.CU_KERNEL_NODE_ATTRIBUTE_ACCESS_POLICY_WINDOW
.. autoattribute:: cuda.bindings.driver.CU_KERNEL_NODE_ATTRIBUTE_COOPERATIVE
.. autoattribute:: cuda.bindings.driver.CU_KERNEL_NODE_ATTRIBUTE_CLUSTER_DIMENSION
.. autoattribute:: cuda.bindings.driver.CU_KERNEL_NODE_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE
.. autoattribute:: cuda.bindings.driver.CU_KERNEL_NODE_ATTRIBUTE_PRIORITY
.. autoattribute:: cuda.bindings.driver.CU_KERNEL_NODE_ATTRIBUTE_MEM_SYNC_DOMAIN_MAP
.. autoattribute:: cuda.bindings.driver.CU_KERNEL_NODE_ATTRIBUTE_MEM_SYNC_DOMAIN
.. autoattribute:: cuda.bindings.driver.CU_KERNEL_NODE_ATTRIBUTE_PREFERRED_CLUSTER_DIMENSION
.. autoattribute:: cuda.bindings.driver.CU_KERNEL_NODE_ATTRIBUTE_DEVICE_UPDATABLE_KERNEL_NODE
.. autoattribute:: cuda.bindings.driver.CU_KERNEL_NODE_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT
.. autoattribute:: cuda.bindings.driver.CU_STREAM_ATTRIBUTE_ACCESS_POLICY_WINDOW
.. autoattribute:: cuda.bindings.driver.CU_STREAM_ATTRIBUTE_SYNCHRONIZATION_POLICY
.. autoattribute:: cuda.bindings.driver.CU_STREAM_ATTRIBUTE_PRIORITY
.. autoattribute:: cuda.bindings.driver.CU_STREAM_ATTRIBUTE_MEM_SYNC_DOMAIN_MAP
.. autoattribute:: cuda.bindings.driver.CU_STREAM_ATTRIBUTE_MEM_SYNC_DOMAIN
.. autoattribute:: cuda.bindings.driver.CU_MEMHOSTALLOC_PORTABLE

    If set, host memory is portable between CUDA contexts. Flag for :py:obj:`~.cuMemHostAlloc()`

.. autoattribute:: cuda.bindings.driver.CU_MEMHOSTALLOC_DEVICEMAP

    If set, host memory is mapped into CUDA address space and :py:obj:`~.cuMemHostGetDevicePointer()` may be called on the host pointer. Flag for :py:obj:`~.cuMemHostAlloc()`

.. autoattribute:: cuda.bindings.driver.CU_MEMHOSTALLOC_WRITECOMBINED

    If set, host memory is allocated as write-combined - fast to write, faster to DMA, slow to read except via SSE4 streaming load instruction (MOVNTDQA). Flag for :py:obj:`~.cuMemHostAlloc()`

.. autoattribute:: cuda.bindings.driver.CU_MEMHOSTREGISTER_PORTABLE

    If set, host memory is portable between CUDA contexts. Flag for :py:obj:`~.cuMemHostRegister()`

.. autoattribute:: cuda.bindings.driver.CU_MEMHOSTREGISTER_DEVICEMAP

    If set, host memory is mapped into CUDA address space and :py:obj:`~.cuMemHostGetDevicePointer()` may be called on the host pointer. Flag for :py:obj:`~.cuMemHostRegister()`

.. autoattribute:: cuda.bindings.driver.CU_MEMHOSTREGISTER_IOMEMORY

    If set, the passed memory pointer is treated as pointing to some memory-mapped I/O space, e.g. belonging to a third-party PCIe device. On Windows the flag is a no-op. On Linux that memory is marked as non cache-coherent for the GPU and is expected to be physically contiguous. It may return :py:obj:`~.CUDA_ERROR_NOT_PERMITTED` if run as an unprivileged user, :py:obj:`~.CUDA_ERROR_NOT_SUPPORTED` on older Linux kernel versions. On all other platforms, it is not supported and :py:obj:`~.CUDA_ERROR_NOT_SUPPORTED` is returned. Flag for :py:obj:`~.cuMemHostRegister()`

.. autoattribute:: cuda.bindings.driver.CU_MEMHOSTREGISTER_READ_ONLY

    If set, the passed memory pointer is treated as pointing to memory that is considered read-only by the device. On platforms without :py:obj:`~.CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES`, this flag is required in order to register memory mapped to the CPU as read-only. Support for the use of this flag can be queried from the device attribute :py:obj:`~.CU_DEVICE_ATTRIBUTE_READ_ONLY_HOST_REGISTER_SUPPORTED`. Using this flag with a current context associated with a device that does not have this attribute set will cause :py:obj:`~.cuMemHostRegister` to error with :py:obj:`~.CUDA_ERROR_NOT_SUPPORTED`.

.. autoattribute:: cuda.bindings.driver.CU_ARRAY_SPARSE_PROPERTIES_SINGLE_MIPTAIL

    Indicates that the layered sparse CUDA array or CUDA mipmapped array has a single mip tail region for all layers

.. autoattribute:: cuda.bindings.driver.CU_TENSOR_MAP_NUM_QWORDS

    Size of tensor map descriptor

.. autoattribute:: cuda.bindings.driver.CUDA_EXTERNAL_MEMORY_DEDICATED

    Indicates that the external memory object is a dedicated resource

.. autoattribute:: cuda.bindings.driver.CUDA_EXTERNAL_SEMAPHORE_SIGNAL_SKIP_NVSCIBUF_MEMSYNC

    When the `flags` parameter of :py:obj:`~.CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS` contains this flag, it indicates that signaling an external semaphore object should skip performing appropriate memory synchronization operations over all the external memory objects that are imported as :py:obj:`~.CU_EXTERNAL_MEMORY_HANDLE_TYPE_NVSCIBUF`, which otherwise are performed by default to ensure data coherency with other importers of the same NvSciBuf memory objects.

.. autoattribute:: cuda.bindings.driver.CUDA_EXTERNAL_SEMAPHORE_WAIT_SKIP_NVSCIBUF_MEMSYNC

    When the `flags` parameter of :py:obj:`~.CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS` contains this flag, it indicates that waiting on an external semaphore object should skip performing appropriate memory synchronization operations over all the external memory objects that are imported as :py:obj:`~.CU_EXTERNAL_MEMORY_HANDLE_TYPE_NVSCIBUF`, which otherwise are performed by default to ensure data coherency with other importers of the same NvSciBuf memory objects.

.. autoattribute:: cuda.bindings.driver.CUDA_NVSCISYNC_ATTR_SIGNAL

    When `flags` of :py:obj:`~.cuDeviceGetNvSciSyncAttributes` is set to this, it indicates that application needs signaler specific NvSciSyncAttr to be filled by :py:obj:`~.cuDeviceGetNvSciSyncAttributes`.

.. autoattribute:: cuda.bindings.driver.CUDA_NVSCISYNC_ATTR_WAIT

    When `flags` of :py:obj:`~.cuDeviceGetNvSciSyncAttributes` is set to this, it indicates that application needs waiter specific NvSciSyncAttr to be filled by :py:obj:`~.cuDeviceGetNvSciSyncAttributes`.

.. autoattribute:: cuda.bindings.driver.CU_MEM_CREATE_USAGE_TILE_POOL

    This flag if set indicates that the memory will be used as a tile pool.

.. autoattribute:: cuda.bindings.driver.CU_MEM_CREATE_USAGE_HW_DECOMPRESS

    This flag, if set, indicates that the memory will be used as a buffer for hardware accelerated decompression.

.. autoattribute:: cuda.bindings.driver.CU_MEM_POOL_CREATE_USAGE_HW_DECOMPRESS

    This flag, if set, indicates that the memory will be used as a buffer for hardware accelerated decompression.

.. autoattribute:: cuda.bindings.driver.CUDA_COOPERATIVE_LAUNCH_MULTI_DEVICE_NO_PRE_LAUNCH_SYNC

    If set, each kernel launched as part of :py:obj:`~.cuLaunchCooperativeKernelMultiDevice` only waits for prior work in the stream corresponding to that GPU to complete before the kernel begins execution.

.. autoattribute:: cuda.bindings.driver.CUDA_COOPERATIVE_LAUNCH_MULTI_DEVICE_NO_POST_LAUNCH_SYNC

    If set, any subsequent work pushed in a stream that participated in a call to :py:obj:`~.cuLaunchCooperativeKernelMultiDevice` will only wait for the kernel launched on the GPU corresponding to that stream to complete before it begins execution.

.. autoattribute:: cuda.bindings.driver.CUDA_ARRAY3D_LAYERED

    If set, the CUDA array is a collection of layers, where each layer is either a 1D or a 2D array and the Depth member of CUDA_ARRAY3D_DESCRIPTOR specifies the number of layers, not the depth of a 3D array.

.. autoattribute:: cuda.bindings.driver.CUDA_ARRAY3D_2DARRAY

    Deprecated, use CUDA_ARRAY3D_LAYERED

.. autoattribute:: cuda.bindings.driver.CUDA_ARRAY3D_SURFACE_LDST

    This flag must be set in order to bind a surface reference to the CUDA array

.. autoattribute:: cuda.bindings.driver.CUDA_ARRAY3D_CUBEMAP

    If set, the CUDA array is a collection of six 2D arrays, representing faces of a cube. The width of such a CUDA array must be equal to its height, and Depth must be six. If :py:obj:`~.CUDA_ARRAY3D_LAYERED` flag is also set, then the CUDA array is a collection of cubemaps and Depth must be a multiple of six.

.. autoattribute:: cuda.bindings.driver.CUDA_ARRAY3D_TEXTURE_GATHER

    This flag must be set in order to perform texture gather operations on a CUDA array.

.. autoattribute:: cuda.bindings.driver.CUDA_ARRAY3D_DEPTH_TEXTURE

    This flag if set indicates that the CUDA array is a DEPTH_TEXTURE.

.. autoattribute:: cuda.bindings.driver.CUDA_ARRAY3D_COLOR_ATTACHMENT

    This flag indicates that the CUDA array may be bound as a color target in an external graphics API

.. autoattribute:: cuda.bindings.driver.CUDA_ARRAY3D_SPARSE

    This flag if set indicates that the CUDA array or CUDA mipmapped array is a sparse CUDA array or CUDA mipmapped array respectively

.. autoattribute:: cuda.bindings.driver.CUDA_ARRAY3D_DEFERRED_MAPPING

    This flag if set indicates that the CUDA array or CUDA mipmapped array will allow deferred memory mapping

.. autoattribute:: cuda.bindings.driver.CUDA_ARRAY3D_VIDEO_ENCODE_DECODE

    This flag indicates that the CUDA array will be used for hardware accelerated video encode/decode operations.

.. autoattribute:: cuda.bindings.driver.CU_TRSA_OVERRIDE_FORMAT

    Override the texref format with a format inferred from the array. Flag for :py:obj:`~.cuTexRefSetArray()`

.. autoattribute:: cuda.bindings.driver.CU_TRSF_READ_AS_INTEGER

    Read the texture as integers rather than promoting the values to floats in the range [0,1]. Flag for :py:obj:`~.cuTexRefSetFlags()` and :py:obj:`~.cuTexObjectCreate()`

.. autoattribute:: cuda.bindings.driver.CU_TRSF_NORMALIZED_COORDINATES

    Use normalized texture coordinates in the range [0,1) instead of [0,dim). Flag for :py:obj:`~.cuTexRefSetFlags()` and :py:obj:`~.cuTexObjectCreate()`

.. autoattribute:: cuda.bindings.driver.CU_TRSF_SRGB

    Perform sRGB->linear conversion during texture read. Flag for :py:obj:`~.cuTexRefSetFlags()` and :py:obj:`~.cuTexObjectCreate()`

.. autoattribute:: cuda.bindings.driver.CU_TRSF_DISABLE_TRILINEAR_OPTIMIZATION

    Disable any trilinear filtering optimizations. Flag for :py:obj:`~.cuTexRefSetFlags()` and :py:obj:`~.cuTexObjectCreate()`

.. autoattribute:: cuda.bindings.driver.CU_TRSF_SEAMLESS_CUBEMAP

    Enable seamless cube map filtering. Flag for :py:obj:`~.cuTexObjectCreate()`

.. autoattribute:: cuda.bindings.driver.CU_LAUNCH_KERNEL_REQUIRED_BLOCK_DIM

    Launch with the required block dimension.

.. autoattribute:: cuda.bindings.driver.CU_LAUNCH_PARAM_END_AS_INT

    C++ compile time constant for CU_LAUNCH_PARAM_END

.. autoattribute:: cuda.bindings.driver.CU_LAUNCH_PARAM_END

    End of array terminator for the `extra` parameter to :py:obj:`~.cuLaunchKernel`

.. autoattribute:: cuda.bindings.driver.CU_LAUNCH_PARAM_BUFFER_POINTER_AS_INT

    C++ compile time constant for CU_LAUNCH_PARAM_BUFFER_POINTER

.. autoattribute:: cuda.bindings.driver.CU_LAUNCH_PARAM_BUFFER_POINTER

    Indicator that the next value in the `extra` parameter to :py:obj:`~.cuLaunchKernel` will be a pointer to a buffer containing all kernel parameters used for launching kernel `f`. This buffer needs to honor all alignment/padding requirements of the individual parameters. If :py:obj:`~.CU_LAUNCH_PARAM_BUFFER_SIZE` is not also specified in the `extra` array, then :py:obj:`~.CU_LAUNCH_PARAM_BUFFER_POINTER` will have no effect.

.. autoattribute:: cuda.bindings.driver.CU_LAUNCH_PARAM_BUFFER_SIZE_AS_INT

    C++ compile time constant for CU_LAUNCH_PARAM_BUFFER_SIZE

.. autoattribute:: cuda.bindings.driver.CU_LAUNCH_PARAM_BUFFER_SIZE

    Indicator that the next value in the `extra` parameter to :py:obj:`~.cuLaunchKernel` will be a pointer to a size_t which contains the size of the buffer specified with :py:obj:`~.CU_LAUNCH_PARAM_BUFFER_POINTER`. It is required that :py:obj:`~.CU_LAUNCH_PARAM_BUFFER_POINTER` also be specified in the `extra` array if the value associated with :py:obj:`~.CU_LAUNCH_PARAM_BUFFER_SIZE` is not zero.

.. autoattribute:: cuda.bindings.driver.CU_PARAM_TR_DEFAULT

    For texture references loaded into the module, use default texunit from texture reference.

.. autoattribute:: cuda.bindings.driver.CU_DEVICE_CPU

    Device that represents the CPU

.. autoattribute:: cuda.bindings.driver.CU_DEVICE_INVALID

    Device that represents an invalid device

.. autoattribute:: cuda.bindings.driver.MAX_PLANES

    Maximum number of planes per frame

.. autoattribute:: cuda.bindings.driver.CUDA_EGL_INFINITE_TIMEOUT

    Indicates that timeout for :py:obj:`~.cuEGLStreamConsumerAcquireFrame` is infinite.


Error Handling
--------------

This section describes the error handling functions of the low-level CUDA driver application programming interface.

.. autofunction:: cuda.bindings.driver.cuGetErrorString
.. autofunction:: cuda.bindings.driver.cuGetErrorName

Initialization
--------------

This section describes the initialization functions of the low-level CUDA driver application programming interface.

.. autofunction:: cuda.bindings.driver.cuInit

Version Management
------------------

This section describes the version management functions of the low-level CUDA driver application programming interface.

.. autofunction:: cuda.bindings.driver.cuDriverGetVersion

Device Management
-----------------

This section describes the device management functions of the low-level CUDA driver application programming interface.

.. autofunction:: cuda.bindings.driver.cuDeviceGet
.. autofunction:: cuda.bindings.driver.cuDeviceGetCount
.. autofunction:: cuda.bindings.driver.cuDeviceGetName
.. autofunction:: cuda.bindings.driver.cuDeviceGetUuid
.. autofunction:: cuda.bindings.driver.cuDeviceGetLuid
.. autofunction:: cuda.bindings.driver.cuDeviceTotalMem
.. autofunction:: cuda.bindings.driver.cuDeviceGetTexture1DLinearMaxWidth
.. autofunction:: cuda.bindings.driver.cuDeviceGetAttribute
.. autofunction:: cuda.bindings.driver.cuDeviceGetHostAtomicCapabilities
.. autofunction:: cuda.bindings.driver.cuDeviceGetNvSciSyncAttributes
.. autofunction:: cuda.bindings.driver.cuDeviceSetMemPool
.. autofunction:: cuda.bindings.driver.cuDeviceGetMemPool
.. autofunction:: cuda.bindings.driver.cuDeviceGetDefaultMemPool
.. autofunction:: cuda.bindings.driver.cuDeviceGetExecAffinitySupport
.. autofunction:: cuda.bindings.driver.cuFlushGPUDirectRDMAWrites

Primary Context Management
--------------------------

This section describes the primary context management functions of the low-level CUDA driver application programming interface.



The primary context is unique per device and shared with the CUDA runtime API. These functions allow integration with other libraries using CUDA.

.. autofunction:: cuda.bindings.driver.cuDevicePrimaryCtxRetain
.. autofunction:: cuda.bindings.driver.cuDevicePrimaryCtxRelease
.. autofunction:: cuda.bindings.driver.cuDevicePrimaryCtxSetFlags
.. autofunction:: cuda.bindings.driver.cuDevicePrimaryCtxGetState
.. autofunction:: cuda.bindings.driver.cuDevicePrimaryCtxReset

Context Management
------------------

This section describes the context management functions of the low-level CUDA driver application programming interface.



Please note that some functions are described in Primary Context Management section.

.. autofunction:: cuda.bindings.driver.cuCtxCreate
.. autofunction:: cuda.bindings.driver.cuCtxDestroy
.. autofunction:: cuda.bindings.driver.cuCtxPushCurrent
.. autofunction:: cuda.bindings.driver.cuCtxPopCurrent
.. autofunction:: cuda.bindings.driver.cuCtxSetCurrent
.. autofunction:: cuda.bindings.driver.cuCtxGetCurrent
.. autofunction:: cuda.bindings.driver.cuCtxGetDevice
.. autofunction:: cuda.bindings.driver.cuCtxGetDevice_v2
.. autofunction:: cuda.bindings.driver.cuCtxGetFlags
.. autofunction:: cuda.bindings.driver.cuCtxSetFlags
.. autofunction:: cuda.bindings.driver.cuCtxGetId
.. autofunction:: cuda.bindings.driver.cuCtxSynchronize
.. autofunction:: cuda.bindings.driver.cuCtxSynchronize_v2
.. autofunction:: cuda.bindings.driver.cuCtxSetLimit
.. autofunction:: cuda.bindings.driver.cuCtxGetLimit
.. autofunction:: cuda.bindings.driver.cuCtxGetCacheConfig
.. autofunction:: cuda.bindings.driver.cuCtxSetCacheConfig
.. autofunction:: cuda.bindings.driver.cuCtxGetApiVersion
.. autofunction:: cuda.bindings.driver.cuCtxGetStreamPriorityRange
.. autofunction:: cuda.bindings.driver.cuCtxResetPersistingL2Cache
.. autofunction:: cuda.bindings.driver.cuCtxGetExecAffinity
.. autofunction:: cuda.bindings.driver.cuCtxRecordEvent
.. autofunction:: cuda.bindings.driver.cuCtxWaitEvent

Module Management
-----------------

This section describes the module management functions of the low-level CUDA driver application programming interface.

.. autoclass:: cuda.bindings.driver.CUmoduleLoadingMode

    .. autoattribute:: cuda.bindings.driver.CUmoduleLoadingMode.CU_MODULE_EAGER_LOADING


        Lazy Kernel Loading is not enabled


    .. autoattribute:: cuda.bindings.driver.CUmoduleLoadingMode.CU_MODULE_LAZY_LOADING


        Lazy Kernel Loading is enabled

.. autofunction:: cuda.bindings.driver.cuModuleLoad
.. autofunction:: cuda.bindings.driver.cuModuleLoadData
.. autofunction:: cuda.bindings.driver.cuModuleLoadDataEx
.. autofunction:: cuda.bindings.driver.cuModuleLoadFatBinary
.. autofunction:: cuda.bindings.driver.cuModuleUnload
.. autofunction:: cuda.bindings.driver.cuModuleGetLoadingMode
.. autofunction:: cuda.bindings.driver.cuModuleGetFunction
.. autofunction:: cuda.bindings.driver.cuModuleGetFunctionCount
.. autofunction:: cuda.bindings.driver.cuModuleEnumerateFunctions
.. autofunction:: cuda.bindings.driver.cuModuleGetGlobal
.. autofunction:: cuda.bindings.driver.cuLinkCreate
.. autofunction:: cuda.bindings.driver.cuLinkAddData
.. autofunction:: cuda.bindings.driver.cuLinkAddFile
.. autofunction:: cuda.bindings.driver.cuLinkComplete
.. autofunction:: cuda.bindings.driver.cuLinkDestroy

Library Management
------------------

This section describes the library management functions of the low-level CUDA driver application programming interface.

.. autofunction:: cuda.bindings.driver.cuLibraryLoadData
.. autofunction:: cuda.bindings.driver.cuLibraryLoadFromFile
.. autofunction:: cuda.bindings.driver.cuLibraryUnload
.. autofunction:: cuda.bindings.driver.cuLibraryGetKernel
.. autofunction:: cuda.bindings.driver.cuLibraryGetKernelCount
.. autofunction:: cuda.bindings.driver.cuLibraryEnumerateKernels
.. autofunction:: cuda.bindings.driver.cuLibraryGetModule
.. autofunction:: cuda.bindings.driver.cuKernelGetFunction
.. autofunction:: cuda.bindings.driver.cuKernelGetLibrary
.. autofunction:: cuda.bindings.driver.cuLibraryGetGlobal
.. autofunction:: cuda.bindings.driver.cuLibraryGetManaged
.. autofunction:: cuda.bindings.driver.cuLibraryGetUnifiedFunction
.. autofunction:: cuda.bindings.driver.cuKernelGetAttribute
.. autofunction:: cuda.bindings.driver.cuKernelSetAttribute
.. autofunction:: cuda.bindings.driver.cuKernelSetCacheConfig
.. autofunction:: cuda.bindings.driver.cuKernelGetName
.. autofunction:: cuda.bindings.driver.cuKernelGetParamInfo

Memory Management
-----------------

This section describes the memory management functions of the low-level CUDA driver application programming interface.

.. autoclass:: cuda.bindings.driver.CUmemDecompressParams_st
.. autoclass:: cuda.bindings.driver.CUmemDecompressAlgorithm

    .. autoattribute:: cuda.bindings.driver.CUmemDecompressAlgorithm.CU_MEM_DECOMPRESS_UNSUPPORTED


        Decompression is unsupported.


    .. autoattribute:: cuda.bindings.driver.CUmemDecompressAlgorithm.CU_MEM_DECOMPRESS_ALGORITHM_DEFLATE


        Deflate is supported.


    .. autoattribute:: cuda.bindings.driver.CUmemDecompressAlgorithm.CU_MEM_DECOMPRESS_ALGORITHM_SNAPPY


        Snappy is supported.


    .. autoattribute:: cuda.bindings.driver.CUmemDecompressAlgorithm.CU_MEM_DECOMPRESS_ALGORITHM_LZ4


        LZ4 is supported.

.. autoclass:: cuda.bindings.driver.CUmemDecompressParams
.. autofunction:: cuda.bindings.driver.cuMemGetInfo
.. autofunction:: cuda.bindings.driver.cuMemAlloc
.. autofunction:: cuda.bindings.driver.cuMemAllocPitch
.. autofunction:: cuda.bindings.driver.cuMemFree
.. autofunction:: cuda.bindings.driver.cuMemGetAddressRange
.. autofunction:: cuda.bindings.driver.cuMemAllocHost
.. autofunction:: cuda.bindings.driver.cuMemFreeHost
.. autofunction:: cuda.bindings.driver.cuMemHostAlloc
.. autofunction:: cuda.bindings.driver.cuMemHostGetDevicePointer
.. autofunction:: cuda.bindings.driver.cuMemHostGetFlags
.. autofunction:: cuda.bindings.driver.cuMemAllocManaged
.. autofunction:: cuda.bindings.driver.cuDeviceRegisterAsyncNotification
.. autofunction:: cuda.bindings.driver.cuDeviceUnregisterAsyncNotification
.. autofunction:: cuda.bindings.driver.cuDeviceGetByPCIBusId
.. autofunction:: cuda.bindings.driver.cuDeviceGetPCIBusId
.. autofunction:: cuda.bindings.driver.cuIpcGetEventHandle
.. autofunction:: cuda.bindings.driver.cuIpcOpenEventHandle
.. autofunction:: cuda.bindings.driver.cuIpcGetMemHandle
.. autofunction:: cuda.bindings.driver.cuIpcOpenMemHandle
.. autofunction:: cuda.bindings.driver.cuIpcCloseMemHandle
.. autofunction:: cuda.bindings.driver.cuMemHostRegister
.. autofunction:: cuda.bindings.driver.cuMemHostUnregister
.. autofunction:: cuda.bindings.driver.cuMemcpy
.. autofunction:: cuda.bindings.driver.cuMemcpyPeer
.. autofunction:: cuda.bindings.driver.cuMemcpyHtoD
.. autofunction:: cuda.bindings.driver.cuMemcpyDtoH
.. autofunction:: cuda.bindings.driver.cuMemcpyDtoD
.. autofunction:: cuda.bindings.driver.cuMemcpyDtoA
.. autofunction:: cuda.bindings.driver.cuMemcpyAtoD
.. autofunction:: cuda.bindings.driver.cuMemcpyHtoA
.. autofunction:: cuda.bindings.driver.cuMemcpyAtoH
.. autofunction:: cuda.bindings.driver.cuMemcpyAtoA
.. autofunction:: cuda.bindings.driver.cuMemcpy2D
.. autofunction:: cuda.bindings.driver.cuMemcpy2DUnaligned
.. autofunction:: cuda.bindings.driver.cuMemcpy3D
.. autofunction:: cuda.bindings.driver.cuMemcpy3DPeer
.. autofunction:: cuda.bindings.driver.cuMemcpyAsync
.. autofunction:: cuda.bindings.driver.cuMemcpyPeerAsync
.. autofunction:: cuda.bindings.driver.cuMemcpyHtoDAsync
.. autofunction:: cuda.bindings.driver.cuMemcpyDtoHAsync
.. autofunction:: cuda.bindings.driver.cuMemcpyDtoDAsync
.. autofunction:: cuda.bindings.driver.cuMemcpyHtoAAsync
.. autofunction:: cuda.bindings.driver.cuMemcpyAtoHAsync
.. autofunction:: cuda.bindings.driver.cuMemcpy2DAsync
.. autofunction:: cuda.bindings.driver.cuMemcpy3DAsync
.. autofunction:: cuda.bindings.driver.cuMemcpy3DPeerAsync
.. autofunction:: cuda.bindings.driver.cuMemcpyBatchAsync
.. autofunction:: cuda.bindings.driver.cuMemcpy3DBatchAsync
.. autofunction:: cuda.bindings.driver.cuMemsetD8
.. autofunction:: cuda.bindings.driver.cuMemsetD16
.. autofunction:: cuda.bindings.driver.cuMemsetD32
.. autofunction:: cuda.bindings.driver.cuMemsetD2D8
.. autofunction:: cuda.bindings.driver.cuMemsetD2D16
.. autofunction:: cuda.bindings.driver.cuMemsetD2D32
.. autofunction:: cuda.bindings.driver.cuMemsetD8Async
.. autofunction:: cuda.bindings.driver.cuMemsetD16Async
.. autofunction:: cuda.bindings.driver.cuMemsetD32Async
.. autofunction:: cuda.bindings.driver.cuMemsetD2D8Async
.. autofunction:: cuda.bindings.driver.cuMemsetD2D16Async
.. autofunction:: cuda.bindings.driver.cuMemsetD2D32Async
.. autofunction:: cuda.bindings.driver.cuArrayCreate
.. autofunction:: cuda.bindings.driver.cuArrayGetDescriptor
.. autofunction:: cuda.bindings.driver.cuArrayGetSparseProperties
.. autofunction:: cuda.bindings.driver.cuMipmappedArrayGetSparseProperties
.. autofunction:: cuda.bindings.driver.cuArrayGetMemoryRequirements
.. autofunction:: cuda.bindings.driver.cuMipmappedArrayGetMemoryRequirements
.. autofunction:: cuda.bindings.driver.cuArrayGetPlane
.. autofunction:: cuda.bindings.driver.cuArrayDestroy
.. autofunction:: cuda.bindings.driver.cuArray3DCreate
.. autofunction:: cuda.bindings.driver.cuArray3DGetDescriptor
.. autofunction:: cuda.bindings.driver.cuMipmappedArrayCreate
.. autofunction:: cuda.bindings.driver.cuMipmappedArrayGetLevel
.. autofunction:: cuda.bindings.driver.cuMipmappedArrayDestroy
.. autofunction:: cuda.bindings.driver.cuMemGetHandleForAddressRange
.. autofunction:: cuda.bindings.driver.cuMemBatchDecompressAsync

Virtual Memory Management
-------------------------

This section describes the virtual memory management functions of the low-level CUDA driver application programming interface.

.. autofunction:: cuda.bindings.driver.cuMemAddressReserve
.. autofunction:: cuda.bindings.driver.cuMemAddressFree
.. autofunction:: cuda.bindings.driver.cuMemCreate
.. autofunction:: cuda.bindings.driver.cuMemRelease
.. autofunction:: cuda.bindings.driver.cuMemMap
.. autofunction:: cuda.bindings.driver.cuMemMapArrayAsync
.. autofunction:: cuda.bindings.driver.cuMemUnmap
.. autofunction:: cuda.bindings.driver.cuMemSetAccess
.. autofunction:: cuda.bindings.driver.cuMemGetAccess
.. autofunction:: cuda.bindings.driver.cuMemExportToShareableHandle
.. autofunction:: cuda.bindings.driver.cuMemImportFromShareableHandle
.. autofunction:: cuda.bindings.driver.cuMemGetAllocationGranularity
.. autofunction:: cuda.bindings.driver.cuMemGetAllocationPropertiesFromHandle
.. autofunction:: cuda.bindings.driver.cuMemRetainAllocationHandle

Stream Ordered Memory Allocator
-------------------------------

This section describes the stream ordered memory allocator exposed by the low-level CUDA driver application programming interface.





**overview**



The asynchronous allocator allows the user to allocate and free in stream order. All asynchronous accesses of the allocation must happen between the stream executions of the allocation and the free. If the memory is accessed outside of the promised stream order, a use before allocation / use after free error will cause undefined behavior.

The allocator is free to reallocate the memory as long as it can guarantee that compliant memory accesses will not overlap temporally. The allocator may refer to internal stream ordering as well as inter-stream dependencies (such as CUDA events and null stream dependencies) when establishing the temporal guarantee. The allocator may also insert inter-stream dependencies to establish the temporal guarantee.





**Supported Platforms**



Whether or not a device supports the integrated stream ordered memory allocator may be queried by calling cuDeviceGetAttribute() with the device attribute CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED

.. autofunction:: cuda.bindings.driver.cuMemFreeAsync
.. autofunction:: cuda.bindings.driver.cuMemAllocAsync
.. autofunction:: cuda.bindings.driver.cuMemPoolTrimTo
.. autofunction:: cuda.bindings.driver.cuMemPoolSetAttribute
.. autofunction:: cuda.bindings.driver.cuMemPoolGetAttribute
.. autofunction:: cuda.bindings.driver.cuMemPoolSetAccess
.. autofunction:: cuda.bindings.driver.cuMemPoolGetAccess
.. autofunction:: cuda.bindings.driver.cuMemPoolCreate
.. autofunction:: cuda.bindings.driver.cuMemPoolDestroy
.. autofunction:: cuda.bindings.driver.cuMemGetDefaultMemPool
.. autofunction:: cuda.bindings.driver.cuMemGetMemPool
.. autofunction:: cuda.bindings.driver.cuMemSetMemPool
.. autofunction:: cuda.bindings.driver.cuMemAllocFromPoolAsync
.. autofunction:: cuda.bindings.driver.cuMemPoolExportToShareableHandle
.. autofunction:: cuda.bindings.driver.cuMemPoolImportFromShareableHandle
.. autofunction:: cuda.bindings.driver.cuMemPoolExportPointer
.. autofunction:: cuda.bindings.driver.cuMemPoolImportPointer

Multicast Object Management
---------------------------

This section describes the CUDA multicast object operations exposed by the low-level CUDA driver application programming interface.





**overview**



A multicast object created via cuMulticastCreate enables certain memory operations to be broadcast to a team of devices. Devices can be added to a multicast object via cuMulticastAddDevice. Memory can be bound on each participating device via either cuMulticastBindMem or cuMulticastBindAddr. Multicast objects can be mapped into a device's virtual address space using the virtual memmory management APIs (see cuMemMap and cuMemSetAccess).





**Supported Platforms**



Support for multicast on a specific device can be queried using the device attribute CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED

.. autofunction:: cuda.bindings.driver.cuMulticastCreate
.. autofunction:: cuda.bindings.driver.cuMulticastAddDevice
.. autofunction:: cuda.bindings.driver.cuMulticastBindMem
.. autofunction:: cuda.bindings.driver.cuMulticastBindAddr
.. autofunction:: cuda.bindings.driver.cuMulticastUnbind
.. autofunction:: cuda.bindings.driver.cuMulticastGetGranularity

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

.. autofunction:: cuda.bindings.driver.cuPointerGetAttribute
.. autofunction:: cuda.bindings.driver.cuMemPrefetchAsync
.. autofunction:: cuda.bindings.driver.cuMemAdvise
.. autofunction:: cuda.bindings.driver.cuMemPrefetchBatchAsync
.. autofunction:: cuda.bindings.driver.cuMemDiscardBatchAsync
.. autofunction:: cuda.bindings.driver.cuMemDiscardAndPrefetchBatchAsync
.. autofunction:: cuda.bindings.driver.cuMemRangeGetAttribute
.. autofunction:: cuda.bindings.driver.cuMemRangeGetAttributes
.. autofunction:: cuda.bindings.driver.cuPointerSetAttribute
.. autofunction:: cuda.bindings.driver.cuPointerGetAttributes

Stream Management
-----------------

This section describes the stream management functions of the low-level CUDA driver application programming interface.

.. autofunction:: cuda.bindings.driver.cuStreamCreate
.. autofunction:: cuda.bindings.driver.cuStreamCreateWithPriority
.. autofunction:: cuda.bindings.driver.cuStreamGetPriority
.. autofunction:: cuda.bindings.driver.cuStreamGetDevice
.. autofunction:: cuda.bindings.driver.cuStreamGetFlags
.. autofunction:: cuda.bindings.driver.cuStreamGetId
.. autofunction:: cuda.bindings.driver.cuStreamGetCtx
.. autofunction:: cuda.bindings.driver.cuStreamGetCtx_v2
.. autofunction:: cuda.bindings.driver.cuStreamWaitEvent
.. autofunction:: cuda.bindings.driver.cuStreamAddCallback
.. autofunction:: cuda.bindings.driver.cuStreamBeginCapture
.. autofunction:: cuda.bindings.driver.cuStreamBeginCaptureToGraph
.. autofunction:: cuda.bindings.driver.cuThreadExchangeStreamCaptureMode
.. autofunction:: cuda.bindings.driver.cuStreamEndCapture
.. autofunction:: cuda.bindings.driver.cuStreamIsCapturing
.. autofunction:: cuda.bindings.driver.cuStreamGetCaptureInfo
.. autofunction:: cuda.bindings.driver.cuStreamUpdateCaptureDependencies
.. autofunction:: cuda.bindings.driver.cuStreamAttachMemAsync
.. autofunction:: cuda.bindings.driver.cuStreamQuery
.. autofunction:: cuda.bindings.driver.cuStreamSynchronize
.. autofunction:: cuda.bindings.driver.cuStreamDestroy
.. autofunction:: cuda.bindings.driver.cuStreamCopyAttributes
.. autofunction:: cuda.bindings.driver.cuStreamGetAttribute
.. autofunction:: cuda.bindings.driver.cuStreamSetAttribute

Event Management
----------------

This section describes the event management functions of the low-level CUDA driver application programming interface.

.. autofunction:: cuda.bindings.driver.cuEventCreate
.. autofunction:: cuda.bindings.driver.cuEventRecord
.. autofunction:: cuda.bindings.driver.cuEventRecordWithFlags
.. autofunction:: cuda.bindings.driver.cuEventQuery
.. autofunction:: cuda.bindings.driver.cuEventSynchronize
.. autofunction:: cuda.bindings.driver.cuEventDestroy
.. autofunction:: cuda.bindings.driver.cuEventElapsedTime

External Resource Interoperability
----------------------------------

This section describes the external resource interoperability functions of the low-level CUDA driver application programming interface.

.. autofunction:: cuda.bindings.driver.cuImportExternalMemory
.. autofunction:: cuda.bindings.driver.cuExternalMemoryGetMappedBuffer
.. autofunction:: cuda.bindings.driver.cuExternalMemoryGetMappedMipmappedArray
.. autofunction:: cuda.bindings.driver.cuDestroyExternalMemory
.. autofunction:: cuda.bindings.driver.cuImportExternalSemaphore
.. autofunction:: cuda.bindings.driver.cuSignalExternalSemaphoresAsync
.. autofunction:: cuda.bindings.driver.cuWaitExternalSemaphoresAsync
.. autofunction:: cuda.bindings.driver.cuDestroyExternalSemaphore

Stream Memory Operations
------------------------

This section describes the stream memory operations of the low-level CUDA driver application programming interface.



Support for the CU_STREAM_WAIT_VALUE_NOR flag can be queried with ::CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR_V2.



Support for the cuStreamWriteValue64() and cuStreamWaitValue64() functions, as well as for the CU_STREAM_MEM_OP_WAIT_VALUE_64 and CU_STREAM_MEM_OP_WRITE_VALUE_64 flags, can be queried with CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS.



Support for both CU_STREAM_WAIT_VALUE_FLUSH and CU_STREAM_MEM_OP_FLUSH_REMOTE_WRITES requires dedicated platform hardware features and can be queried with cuDeviceGetAttribute() and CU_DEVICE_ATTRIBUTE_CAN_FLUSH_REMOTE_WRITES.



Note that all memory pointers passed as parameters to these operations are device pointers. Where necessary a device pointer should be obtained, for example with cuMemHostGetDevicePointer().



None of the operations accepts pointers to managed memory buffers (cuMemAllocManaged).



Warning: Improper use of these APIs may deadlock the application. Synchronization ordering established through these APIs is not visible to CUDA. CUDA tasks that are (even indirectly) ordered by these APIs should also have that order expressed with CUDA-visible dependencies such as events. This ensures that the scheduler does not serialize them in an improper order.

.. autofunction:: cuda.bindings.driver.cuStreamWaitValue32
.. autofunction:: cuda.bindings.driver.cuStreamWaitValue64
.. autofunction:: cuda.bindings.driver.cuStreamWriteValue32
.. autofunction:: cuda.bindings.driver.cuStreamWriteValue64
.. autofunction:: cuda.bindings.driver.cuStreamBatchMemOp

Execution Control
-----------------

This section describes the execution control functions of the low-level CUDA driver application programming interface.

.. autoclass:: cuda.bindings.driver.CUfunctionLoadingState

    .. autoattribute:: cuda.bindings.driver.CUfunctionLoadingState.CU_FUNCTION_LOADING_STATE_UNLOADED


    .. autoattribute:: cuda.bindings.driver.CUfunctionLoadingState.CU_FUNCTION_LOADING_STATE_LOADED


    .. autoattribute:: cuda.bindings.driver.CUfunctionLoadingState.CU_FUNCTION_LOADING_STATE_MAX

.. autofunction:: cuda.bindings.driver.cuFuncGetAttribute
.. autofunction:: cuda.bindings.driver.cuFuncSetAttribute
.. autofunction:: cuda.bindings.driver.cuFuncSetCacheConfig
.. autofunction:: cuda.bindings.driver.cuFuncGetModule
.. autofunction:: cuda.bindings.driver.cuFuncGetName
.. autofunction:: cuda.bindings.driver.cuFuncGetParamInfo
.. autofunction:: cuda.bindings.driver.cuFuncIsLoaded
.. autofunction:: cuda.bindings.driver.cuFuncLoad
.. autofunction:: cuda.bindings.driver.cuLaunchKernel
.. autofunction:: cuda.bindings.driver.cuLaunchKernelEx
.. autofunction:: cuda.bindings.driver.cuLaunchCooperativeKernel
.. autofunction:: cuda.bindings.driver.cuLaunchCooperativeKernelMultiDevice
.. autofunction:: cuda.bindings.driver.cuLaunchHostFunc

Graph Management
----------------

This section describes the graph management functions of the low-level CUDA driver application programming interface.

.. autofunction:: cuda.bindings.driver.cuGraphCreate
.. autofunction:: cuda.bindings.driver.cuGraphAddKernelNode
.. autofunction:: cuda.bindings.driver.cuGraphKernelNodeGetParams
.. autofunction:: cuda.bindings.driver.cuGraphKernelNodeSetParams
.. autofunction:: cuda.bindings.driver.cuGraphAddMemcpyNode
.. autofunction:: cuda.bindings.driver.cuGraphMemcpyNodeGetParams
.. autofunction:: cuda.bindings.driver.cuGraphMemcpyNodeSetParams
.. autofunction:: cuda.bindings.driver.cuGraphAddMemsetNode
.. autofunction:: cuda.bindings.driver.cuGraphMemsetNodeGetParams
.. autofunction:: cuda.bindings.driver.cuGraphMemsetNodeSetParams
.. autofunction:: cuda.bindings.driver.cuGraphAddHostNode
.. autofunction:: cuda.bindings.driver.cuGraphHostNodeGetParams
.. autofunction:: cuda.bindings.driver.cuGraphHostNodeSetParams
.. autofunction:: cuda.bindings.driver.cuGraphAddChildGraphNode
.. autofunction:: cuda.bindings.driver.cuGraphChildGraphNodeGetGraph
.. autofunction:: cuda.bindings.driver.cuGraphAddEmptyNode
.. autofunction:: cuda.bindings.driver.cuGraphAddEventRecordNode
.. autofunction:: cuda.bindings.driver.cuGraphEventRecordNodeGetEvent
.. autofunction:: cuda.bindings.driver.cuGraphEventRecordNodeSetEvent
.. autofunction:: cuda.bindings.driver.cuGraphAddEventWaitNode
.. autofunction:: cuda.bindings.driver.cuGraphEventWaitNodeGetEvent
.. autofunction:: cuda.bindings.driver.cuGraphEventWaitNodeSetEvent
.. autofunction:: cuda.bindings.driver.cuGraphAddExternalSemaphoresSignalNode
.. autofunction:: cuda.bindings.driver.cuGraphExternalSemaphoresSignalNodeGetParams
.. autofunction:: cuda.bindings.driver.cuGraphExternalSemaphoresSignalNodeSetParams
.. autofunction:: cuda.bindings.driver.cuGraphAddExternalSemaphoresWaitNode
.. autofunction:: cuda.bindings.driver.cuGraphExternalSemaphoresWaitNodeGetParams
.. autofunction:: cuda.bindings.driver.cuGraphExternalSemaphoresWaitNodeSetParams
.. autofunction:: cuda.bindings.driver.cuGraphAddBatchMemOpNode
.. autofunction:: cuda.bindings.driver.cuGraphBatchMemOpNodeGetParams
.. autofunction:: cuda.bindings.driver.cuGraphBatchMemOpNodeSetParams
.. autofunction:: cuda.bindings.driver.cuGraphExecBatchMemOpNodeSetParams
.. autofunction:: cuda.bindings.driver.cuGraphAddMemAllocNode
.. autofunction:: cuda.bindings.driver.cuGraphMemAllocNodeGetParams
.. autofunction:: cuda.bindings.driver.cuGraphAddMemFreeNode
.. autofunction:: cuda.bindings.driver.cuGraphMemFreeNodeGetParams
.. autofunction:: cuda.bindings.driver.cuDeviceGraphMemTrim
.. autofunction:: cuda.bindings.driver.cuDeviceGetGraphMemAttribute
.. autofunction:: cuda.bindings.driver.cuDeviceSetGraphMemAttribute
.. autofunction:: cuda.bindings.driver.cuGraphClone
.. autofunction:: cuda.bindings.driver.cuGraphNodeFindInClone
.. autofunction:: cuda.bindings.driver.cuGraphNodeGetType
.. autofunction:: cuda.bindings.driver.cuGraphGetNodes
.. autofunction:: cuda.bindings.driver.cuGraphGetRootNodes
.. autofunction:: cuda.bindings.driver.cuGraphGetEdges
.. autofunction:: cuda.bindings.driver.cuGraphNodeGetDependencies
.. autofunction:: cuda.bindings.driver.cuGraphNodeGetDependentNodes
.. autofunction:: cuda.bindings.driver.cuGraphAddDependencies
.. autofunction:: cuda.bindings.driver.cuGraphRemoveDependencies
.. autofunction:: cuda.bindings.driver.cuGraphDestroyNode
.. autofunction:: cuda.bindings.driver.cuGraphInstantiate
.. autofunction:: cuda.bindings.driver.cuGraphInstantiateWithParams
.. autofunction:: cuda.bindings.driver.cuGraphExecGetFlags
.. autofunction:: cuda.bindings.driver.cuGraphExecKernelNodeSetParams
.. autofunction:: cuda.bindings.driver.cuGraphExecMemcpyNodeSetParams
.. autofunction:: cuda.bindings.driver.cuGraphExecMemsetNodeSetParams
.. autofunction:: cuda.bindings.driver.cuGraphExecHostNodeSetParams
.. autofunction:: cuda.bindings.driver.cuGraphExecChildGraphNodeSetParams
.. autofunction:: cuda.bindings.driver.cuGraphExecEventRecordNodeSetEvent
.. autofunction:: cuda.bindings.driver.cuGraphExecEventWaitNodeSetEvent
.. autofunction:: cuda.bindings.driver.cuGraphExecExternalSemaphoresSignalNodeSetParams
.. autofunction:: cuda.bindings.driver.cuGraphExecExternalSemaphoresWaitNodeSetParams
.. autofunction:: cuda.bindings.driver.cuGraphNodeSetEnabled
.. autofunction:: cuda.bindings.driver.cuGraphNodeGetEnabled
.. autofunction:: cuda.bindings.driver.cuGraphUpload
.. autofunction:: cuda.bindings.driver.cuGraphLaunch
.. autofunction:: cuda.bindings.driver.cuGraphExecDestroy
.. autofunction:: cuda.bindings.driver.cuGraphDestroy
.. autofunction:: cuda.bindings.driver.cuGraphExecUpdate
.. autofunction:: cuda.bindings.driver.cuGraphKernelNodeCopyAttributes
.. autofunction:: cuda.bindings.driver.cuGraphKernelNodeGetAttribute
.. autofunction:: cuda.bindings.driver.cuGraphKernelNodeSetAttribute
.. autofunction:: cuda.bindings.driver.cuGraphDebugDotPrint
.. autofunction:: cuda.bindings.driver.cuUserObjectCreate
.. autofunction:: cuda.bindings.driver.cuUserObjectRetain
.. autofunction:: cuda.bindings.driver.cuUserObjectRelease
.. autofunction:: cuda.bindings.driver.cuGraphRetainUserObject
.. autofunction:: cuda.bindings.driver.cuGraphReleaseUserObject
.. autofunction:: cuda.bindings.driver.cuGraphAddNode
.. autofunction:: cuda.bindings.driver.cuGraphNodeSetParams
.. autofunction:: cuda.bindings.driver.cuGraphExecNodeSetParams
.. autofunction:: cuda.bindings.driver.cuGraphConditionalHandleCreate

Occupancy
---------

This section describes the occupancy calculation functions of the low-level CUDA driver application programming interface.

.. autofunction:: cuda.bindings.driver.cuOccupancyMaxActiveBlocksPerMultiprocessor
.. autofunction:: cuda.bindings.driver.cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags
.. autofunction:: cuda.bindings.driver.cuOccupancyMaxPotentialBlockSize
.. autofunction:: cuda.bindings.driver.cuOccupancyMaxPotentialBlockSizeWithFlags
.. autofunction:: cuda.bindings.driver.cuOccupancyAvailableDynamicSMemPerBlock
.. autofunction:: cuda.bindings.driver.cuOccupancyMaxPotentialClusterSize
.. autofunction:: cuda.bindings.driver.cuOccupancyMaxActiveClusters

Texture Object Management
-------------------------

This section describes the texture object management functions of the low-level CUDA driver application programming interface. The texture object API is only supported on devices of compute capability 3.0 or higher.

.. autofunction:: cuda.bindings.driver.cuTexObjectCreate
.. autofunction:: cuda.bindings.driver.cuTexObjectDestroy
.. autofunction:: cuda.bindings.driver.cuTexObjectGetResourceDesc
.. autofunction:: cuda.bindings.driver.cuTexObjectGetTextureDesc
.. autofunction:: cuda.bindings.driver.cuTexObjectGetResourceViewDesc

Surface Object Management
-------------------------

This section describes the surface object management functions of the low-level CUDA driver application programming interface. The surface object API is only supported on devices of compute capability 3.0 or higher.

.. autofunction:: cuda.bindings.driver.cuSurfObjectCreate
.. autofunction:: cuda.bindings.driver.cuSurfObjectDestroy
.. autofunction:: cuda.bindings.driver.cuSurfObjectGetResourceDesc

Tensor Map Object Managment
---------------------------

This section describes the tensor map object management functions of the low-level CUDA driver application programming interface. The tensor core API is only supported on devices of compute capability 9.0 or higher.

.. autofunction:: cuda.bindings.driver.cuTensorMapEncodeTiled
.. autofunction:: cuda.bindings.driver.cuTensorMapEncodeIm2col
.. autofunction:: cuda.bindings.driver.cuTensorMapEncodeIm2colWide
.. autofunction:: cuda.bindings.driver.cuTensorMapReplaceAddress

Peer Context Memory Access
--------------------------

This section describes the direct peer context memory access functions of the low-level CUDA driver application programming interface.

.. autofunction:: cuda.bindings.driver.cuDeviceCanAccessPeer
.. autofunction:: cuda.bindings.driver.cuCtxEnablePeerAccess
.. autofunction:: cuda.bindings.driver.cuCtxDisablePeerAccess
.. autofunction:: cuda.bindings.driver.cuDeviceGetP2PAttribute
.. autofunction:: cuda.bindings.driver.cuDeviceGetP2PAtomicCapabilities

Graphics Interoperability
-------------------------

This section describes the graphics interoperability functions of the low-level CUDA driver application programming interface.

.. autofunction:: cuda.bindings.driver.cuGraphicsUnregisterResource
.. autofunction:: cuda.bindings.driver.cuGraphicsSubResourceGetMappedArray
.. autofunction:: cuda.bindings.driver.cuGraphicsResourceGetMappedMipmappedArray
.. autofunction:: cuda.bindings.driver.cuGraphicsResourceGetMappedPointer
.. autofunction:: cuda.bindings.driver.cuGraphicsResourceSetMapFlags
.. autofunction:: cuda.bindings.driver.cuGraphicsMapResources
.. autofunction:: cuda.bindings.driver.cuGraphicsUnmapResources

Driver Entry Point Access
-------------------------

This section describes the driver entry point access functions of the low-level CUDA driver application programming interface.

.. autofunction:: cuda.bindings.driver.cuGetProcAddress

Coredump Attributes Control API
-------------------------------

This section describes the coredump attribute control functions of the low-level CUDA driver application programming interface.

.. autoclass:: cuda.bindings.driver.CUcoredumpSettings

    .. autoattribute:: cuda.bindings.driver.CUcoredumpSettings.CU_COREDUMP_ENABLE_ON_EXCEPTION


    .. autoattribute:: cuda.bindings.driver.CUcoredumpSettings.CU_COREDUMP_TRIGGER_HOST


    .. autoattribute:: cuda.bindings.driver.CUcoredumpSettings.CU_COREDUMP_LIGHTWEIGHT


    .. autoattribute:: cuda.bindings.driver.CUcoredumpSettings.CU_COREDUMP_ENABLE_USER_TRIGGER


    .. autoattribute:: cuda.bindings.driver.CUcoredumpSettings.CU_COREDUMP_FILE


    .. autoattribute:: cuda.bindings.driver.CUcoredumpSettings.CU_COREDUMP_PIPE


    .. autoattribute:: cuda.bindings.driver.CUcoredumpSettings.CU_COREDUMP_GENERATION_FLAGS


    .. autoattribute:: cuda.bindings.driver.CUcoredumpSettings.CU_COREDUMP_MAX

.. autoclass:: cuda.bindings.driver.CUCoredumpGenerationFlags

    .. autoattribute:: cuda.bindings.driver.CUCoredumpGenerationFlags.CU_COREDUMP_DEFAULT_FLAGS


    .. autoattribute:: cuda.bindings.driver.CUCoredumpGenerationFlags.CU_COREDUMP_SKIP_NONRELOCATED_ELF_IMAGES


    .. autoattribute:: cuda.bindings.driver.CUCoredumpGenerationFlags.CU_COREDUMP_SKIP_GLOBAL_MEMORY


    .. autoattribute:: cuda.bindings.driver.CUCoredumpGenerationFlags.CU_COREDUMP_SKIP_SHARED_MEMORY


    .. autoattribute:: cuda.bindings.driver.CUCoredumpGenerationFlags.CU_COREDUMP_SKIP_LOCAL_MEMORY


    .. autoattribute:: cuda.bindings.driver.CUCoredumpGenerationFlags.CU_COREDUMP_SKIP_ABORT


    .. autoattribute:: cuda.bindings.driver.CUCoredumpGenerationFlags.CU_COREDUMP_SKIP_CONSTBANK_MEMORY


    .. autoattribute:: cuda.bindings.driver.CUCoredumpGenerationFlags.CU_COREDUMP_LIGHTWEIGHT_FLAGS

.. autofunction:: cuda.bindings.driver.cuCoredumpGetAttribute
.. autofunction:: cuda.bindings.driver.cuCoredumpGetAttributeGlobal
.. autofunction:: cuda.bindings.driver.cuCoredumpSetAttribute
.. autofunction:: cuda.bindings.driver.cuCoredumpSetAttributeGlobal

Green Contexts
--------------

This section describes the APIs for creation and manipulation of green contexts in the CUDA driver. Green contexts are a lightweight alternative to traditional contexts, with the ability to pass in a set of resources that they should be initialized with. This allows the developer to represent distinct spatial partitions of the GPU, provision resources for them, and target them via the same programming model that CUDA exposes (streams, kernel launches, etc.).



There are 4 main steps to using these new set of APIs.

- (1) Start with an initial set of resources, for example via cuDeviceGetDevResource. Only SM type is supported today.







- (2) Partition this set of resources by providing them as input to a partition API, for example: cuDevSmResourceSplitByCount.







- (3) Finalize the specification of resources by creating a descriptor via cuDevResourceGenerateDesc.







- (4) Provision the resources and create a green context via cuGreenCtxCreate.











For ``CU_DEV_RESOURCE_TYPE_SM``\ , the partitions created have minimum SM count requirements, often rounding up and aligning the minCount provided to cuDevSmResourceSplitByCount. These requirements can be queried with cuDeviceGetDevResource from step (1) above to determine the minimum partition size (``sm.minSmPartitionSize``\ ) and alignment granularity (``sm.smCoscheduledAlignment``\ ).



While it's recommended to use cuDeviceGetDevResource for accurate information, here is a guideline for each compute architecture:

- On Compute Architecture 6.X: The minimum count is 2 SMs and must be a multiple of 2.







- On Compute Architecture 7.X: The minimum count is 2 SMs and must be a multiple of 2.







- On Compute Architecture 8.X: The minimum count is 4 SMs and must be a multiple of 2.







- On Compute Architecture 9.0+: The minimum count is 8 SMs and must be a multiple of 8.











In the future, flags can be provided to tradeoff functional and performance characteristics versus finer grained SM partitions.



Even if the green contexts have disjoint SM partitions, it is not guaranteed that the kernels launched in them will run concurrently or have forward progress guarantees. This is due to other resources (like HW connections, see ::CUDA_DEVICE_MAX_CONNECTIONS) that could cause a dependency. Additionally, in certain scenarios, it is possible for the workload to run on more SMs than was provisioned (but never less). The following are two scenarios which can exhibit this behavior:

- On Volta+ MPS: When ``CUDA_MPS_ACTIVE_THREAD_PERCENTAGE``\  is used, the set of SMs that are used for running kernels can be scaled up to the value of SMs used for the MPS client.







- On Compute Architecture 9.x: When a module with dynamic parallelism (CDP) is loaded, all future kernels running under green contexts may use and share an additional set of 2 SMs.

.. autoclass:: cuda.bindings.driver.CUdevSmResource_st
.. autoclass:: cuda.bindings.driver.CUdevResource_st
.. autoclass:: cuda.bindings.driver.CUdevSmResource
.. autoclass:: cuda.bindings.driver.CUdevResource
.. autoclass:: cuda.bindings.driver.CUgreenCtxCreate_flags

    .. autoattribute:: cuda.bindings.driver.CUgreenCtxCreate_flags.CU_GREEN_CTX_DEFAULT_STREAM


        Required. Creates a default stream to use inside the green context

.. autoclass:: cuda.bindings.driver.CUdevSmResourceSplit_flags

    .. autoattribute:: cuda.bindings.driver.CUdevSmResourceSplit_flags.CU_DEV_SM_RESOURCE_SPLIT_IGNORE_SM_COSCHEDULING


    .. autoattribute:: cuda.bindings.driver.CUdevSmResourceSplit_flags.CU_DEV_SM_RESOURCE_SPLIT_MAX_POTENTIAL_CLUSTER_SIZE

.. autoclass:: cuda.bindings.driver.CUdevResourceType

    .. autoattribute:: cuda.bindings.driver.CUdevResourceType.CU_DEV_RESOURCE_TYPE_INVALID


    .. autoattribute:: cuda.bindings.driver.CUdevResourceType.CU_DEV_RESOURCE_TYPE_SM


        Streaming multiprocessors related information

.. autoclass:: cuda.bindings.driver.CUdevResourceDesc
.. autoclass:: cuda.bindings.driver.CUdevSmResource
.. autofunction:: cuda.bindings.driver._CONCAT_OUTER
.. autofunction:: cuda.bindings.driver.cuGreenCtxCreate
.. autofunction:: cuda.bindings.driver.cuGreenCtxDestroy
.. autofunction:: cuda.bindings.driver.cuCtxFromGreenCtx
.. autofunction:: cuda.bindings.driver.cuDeviceGetDevResource
.. autofunction:: cuda.bindings.driver.cuCtxGetDevResource
.. autofunction:: cuda.bindings.driver.cuGreenCtxGetDevResource
.. autofunction:: cuda.bindings.driver.cuDevSmResourceSplitByCount
.. autofunction:: cuda.bindings.driver.cuDevResourceGenerateDesc
.. autofunction:: cuda.bindings.driver.cuGreenCtxRecordEvent
.. autofunction:: cuda.bindings.driver.cuGreenCtxWaitEvent
.. autofunction:: cuda.bindings.driver.cuStreamGetGreenCtx
.. autofunction:: cuda.bindings.driver.cuGreenCtxStreamCreate
.. autofunction:: cuda.bindings.driver.cuGreenCtxGetId
.. autoattribute:: cuda.bindings.driver.RESOURCE_ABI_VERSION
.. autoattribute:: cuda.bindings.driver.RESOURCE_ABI_EXTERNAL_BYTES
.. autoattribute:: cuda.bindings.driver._CONCAT_INNER
.. autoattribute:: cuda.bindings.driver._CONCAT_OUTER

Error Log Management Functions
------------------------------

This section describes the error log management functions of the low-level CUDA driver application programming interface.

.. autoclass:: cuda.bindings.driver.CUlogLevel

    .. autoattribute:: cuda.bindings.driver.CUlogLevel.CU_LOG_LEVEL_ERROR


    .. autoattribute:: cuda.bindings.driver.CUlogLevel.CU_LOG_LEVEL_WARNING

.. autoclass:: cuda.bindings.driver.CUlogsCallbackHandle
.. autoclass:: cuda.bindings.driver.CUlogsCallback
.. autoclass:: cuda.bindings.driver.CUlogIterator
.. autofunction:: cuda.bindings.driver.cuLogsRegisterCallback
.. autofunction:: cuda.bindings.driver.cuLogsUnregisterCallback
.. autofunction:: cuda.bindings.driver.cuLogsCurrent
.. autofunction:: cuda.bindings.driver.cuLogsDumpToFile
.. autofunction:: cuda.bindings.driver.cuLogsDumpToMemory

CUDA Checkpointing
------------------

CUDA API versioning support







This sections describes the checkpoint and restore functions of the low-level CUDA driver application programming interface.



The CUDA checkpoint and restore API's provide a way to save and restore GPU state for full process checkpoints when used with CPU side process checkpointing solutions. They can also be used to pause GPU work and suspend a CUDA process to allow other applications to make use of GPU resources.



Checkpoint and restore capabilities are currently restricted to Linux.

.. autofunction:: cuda.bindings.driver.cuCheckpointProcessGetRestoreThreadId
.. autofunction:: cuda.bindings.driver.cuCheckpointProcessGetState
.. autofunction:: cuda.bindings.driver.cuCheckpointProcessLock
.. autofunction:: cuda.bindings.driver.cuCheckpointProcessCheckpoint
.. autofunction:: cuda.bindings.driver.cuCheckpointProcessRestore
.. autofunction:: cuda.bindings.driver.cuCheckpointProcessUnlock

EGL Interoperability
--------------------

This section describes the EGL interoperability functions of the low-level CUDA driver application programming interface.

.. autofunction:: cuda.bindings.driver.cuGraphicsEGLRegisterImage
.. autofunction:: cuda.bindings.driver.cuEGLStreamConsumerConnect
.. autofunction:: cuda.bindings.driver.cuEGLStreamConsumerConnectWithFlags
.. autofunction:: cuda.bindings.driver.cuEGLStreamConsumerDisconnect
.. autofunction:: cuda.bindings.driver.cuEGLStreamConsumerAcquireFrame
.. autofunction:: cuda.bindings.driver.cuEGLStreamConsumerReleaseFrame
.. autofunction:: cuda.bindings.driver.cuEGLStreamProducerConnect
.. autofunction:: cuda.bindings.driver.cuEGLStreamProducerDisconnect
.. autofunction:: cuda.bindings.driver.cuEGLStreamProducerPresentFrame
.. autofunction:: cuda.bindings.driver.cuEGLStreamProducerReturnFrame
.. autofunction:: cuda.bindings.driver.cuGraphicsResourceGetMappedEglFrame
.. autofunction:: cuda.bindings.driver.cuEventCreateFromEGLSync

OpenGL Interoperability
-----------------------

This section describes the OpenGL interoperability functions of the low-level CUDA driver application programming interface. Note that mapping of OpenGL resources is performed with the graphics API agnostic, resource mapping interface described in Graphics Interoperability.

.. autoclass:: cuda.bindings.driver.CUGLDeviceList

    .. autoattribute:: cuda.bindings.driver.CUGLDeviceList.CU_GL_DEVICE_LIST_ALL


        The CUDA devices for all GPUs used by the current OpenGL context


    .. autoattribute:: cuda.bindings.driver.CUGLDeviceList.CU_GL_DEVICE_LIST_CURRENT_FRAME


        The CUDA devices for the GPUs used by the current OpenGL context in its currently rendering frame


    .. autoattribute:: cuda.bindings.driver.CUGLDeviceList.CU_GL_DEVICE_LIST_NEXT_FRAME


        The CUDA devices for the GPUs to be used by the current OpenGL context in the next frame

.. autofunction:: cuda.bindings.driver.cuGraphicsGLRegisterBuffer
.. autofunction:: cuda.bindings.driver.cuGraphicsGLRegisterImage
.. autofunction:: cuda.bindings.driver.cuGLGetDevices

Profiler Control
----------------

This section describes the profiler control functions of the low-level CUDA driver application programming interface.

.. autofunction:: cuda.bindings.driver.cuProfilerStart
.. autofunction:: cuda.bindings.driver.cuProfilerStop

VDPAU Interoperability
----------------------

This section describes the VDPAU interoperability functions of the low-level CUDA driver application programming interface.

.. autofunction:: cuda.bindings.driver.cuVDPAUGetDevice
.. autofunction:: cuda.bindings.driver.cuVDPAUCtxCreate
.. autofunction:: cuda.bindings.driver.cuGraphicsVDPAURegisterVideoSurface
.. autofunction:: cuda.bindings.driver.cuGraphicsVDPAURegisterOutputSurface
