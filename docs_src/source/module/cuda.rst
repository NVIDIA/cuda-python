----
cuda
----

Data types used by CUDA driver
------------------------------



.. autoclass:: cuda.cuda.CUuuid_st
.. autoclass:: cuda.cuda.CUipcEventHandle_st
.. autoclass:: cuda.cuda.CUipcMemHandle_st
.. autoclass:: cuda.cuda.CUstreamBatchMemOpParams_union
.. autoclass:: cuda.cuda.CUDA_BATCH_MEM_OP_NODE_PARAMS_st
.. autoclass:: cuda.cuda.CUdevprop_st
.. autoclass:: cuda.cuda.CUaccessPolicyWindow_st
.. autoclass:: cuda.cuda.CUDA_KERNEL_NODE_PARAMS_st
.. autoclass:: cuda.cuda.CUDA_MEMSET_NODE_PARAMS_st
.. autoclass:: cuda.cuda.CUDA_HOST_NODE_PARAMS_st
.. autoclass:: cuda.cuda.CUkernelNodeAttrValue_union
.. autoclass:: cuda.cuda.CUstreamAttrValue_union
.. autoclass:: cuda.cuda.CUexecAffinitySmCount_st
.. autoclass:: cuda.cuda.CUexecAffinityParam_st
.. autoclass:: cuda.cuda.CUDA_MEMCPY2D_st
.. autoclass:: cuda.cuda.CUDA_MEMCPY3D_st
.. autoclass:: cuda.cuda.CUDA_MEMCPY3D_PEER_st
.. autoclass:: cuda.cuda.CUDA_ARRAY_DESCRIPTOR_st
.. autoclass:: cuda.cuda.CUDA_ARRAY3D_DESCRIPTOR_st
.. autoclass:: cuda.cuda.CUDA_ARRAY_SPARSE_PROPERTIES_st
.. autoclass:: cuda.cuda.CUDA_ARRAY_MEMORY_REQUIREMENTS_st
.. autoclass:: cuda.cuda.CUDA_RESOURCE_DESC_st
.. autoclass:: cuda.cuda.CUDA_TEXTURE_DESC_st
.. autoclass:: cuda.cuda.CUDA_RESOURCE_VIEW_DESC_st
.. autoclass:: cuda.cuda.CUDA_POINTER_ATTRIBUTE_P2P_TOKENS_st
.. autoclass:: cuda.cuda.CUDA_LAUNCH_PARAMS_st
.. autoclass:: cuda.cuda.CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st
.. autoclass:: cuda.cuda.CUDA_EXTERNAL_MEMORY_BUFFER_DESC_st
.. autoclass:: cuda.cuda.CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC_st
.. autoclass:: cuda.cuda.CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_st
.. autoclass:: cuda.cuda.CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st
.. autoclass:: cuda.cuda.CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st
.. autoclass:: cuda.cuda.CUDA_EXT_SEM_SIGNAL_NODE_PARAMS_st
.. autoclass:: cuda.cuda.CUDA_EXT_SEM_WAIT_NODE_PARAMS_st
.. autoclass:: cuda.cuda.CUarrayMapInfo_st
.. autoclass:: cuda.cuda.CUmemLocation_st
.. autoclass:: cuda.cuda.CUmemAllocationProp_st
.. autoclass:: cuda.cuda.CUmemAccessDesc_st
.. autoclass:: cuda.cuda.CUmemPoolProps_st
.. autoclass:: cuda.cuda.CUmemPoolPtrExportData_st
.. autoclass:: cuda.cuda.CUDA_MEM_ALLOC_NODE_PARAMS_st
.. autoclass:: cuda.cuda.CUeglFrame_st
.. autoclass:: cuda.cuda.CUipcMem_flags

    .. autoattribute:: cuda.cuda.CUipcMem_flags.CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS


        Automatically enable peer access between remote devices as needed

.. autoclass:: cuda.cuda.CUmemAttach_flags

    .. autoattribute:: cuda.cuda.CUmemAttach_flags.CU_MEM_ATTACH_GLOBAL


        Memory can be accessed by any stream on any device


    .. autoattribute:: cuda.cuda.CUmemAttach_flags.CU_MEM_ATTACH_HOST


        Memory cannot be accessed by any stream on any device


    .. autoattribute:: cuda.cuda.CUmemAttach_flags.CU_MEM_ATTACH_SINGLE


        Memory can only be accessed by a single stream on the associated device

.. autoclass:: cuda.cuda.CUctx_flags

    .. autoattribute:: cuda.cuda.CUctx_flags.CU_CTX_SCHED_AUTO


        Automatic scheduling


    .. autoattribute:: cuda.cuda.CUctx_flags.CU_CTX_SCHED_SPIN


        Set spin as default scheduling


    .. autoattribute:: cuda.cuda.CUctx_flags.CU_CTX_SCHED_YIELD


        Set yield as default scheduling


    .. autoattribute:: cuda.cuda.CUctx_flags.CU_CTX_SCHED_BLOCKING_SYNC


        Set blocking synchronization as default scheduling


    .. autoattribute:: cuda.cuda.CUctx_flags.CU_CTX_BLOCKING_SYNC


        Set blocking synchronization as default scheduling [Deprecated]


    .. autoattribute:: cuda.cuda.CUctx_flags.CU_CTX_SCHED_MASK


    .. autoattribute:: cuda.cuda.CUctx_flags.CU_CTX_MAP_HOST


        [Deprecated]


    .. autoattribute:: cuda.cuda.CUctx_flags.CU_CTX_LMEM_RESIZE_TO_MAX


        Keep local memory allocation after launch


    .. autoattribute:: cuda.cuda.CUctx_flags.CU_CTX_FLAGS_MASK

.. autoclass:: cuda.cuda.CUstream_flags

    .. autoattribute:: cuda.cuda.CUstream_flags.CU_STREAM_DEFAULT


        Default stream flag


    .. autoattribute:: cuda.cuda.CUstream_flags.CU_STREAM_NON_BLOCKING


        Stream does not synchronize with stream 0 (the NULL stream)

.. autoclass:: cuda.cuda.CUevent_flags

    .. autoattribute:: cuda.cuda.CUevent_flags.CU_EVENT_DEFAULT


        Default event flag


    .. autoattribute:: cuda.cuda.CUevent_flags.CU_EVENT_BLOCKING_SYNC


        Event uses blocking synchronization


    .. autoattribute:: cuda.cuda.CUevent_flags.CU_EVENT_DISABLE_TIMING


        Event will not record timing data


    .. autoattribute:: cuda.cuda.CUevent_flags.CU_EVENT_INTERPROCESS


        Event is suitable for interprocess use. CU_EVENT_DISABLE_TIMING must be set

.. autoclass:: cuda.cuda.CUevent_record_flags

    .. autoattribute:: cuda.cuda.CUevent_record_flags.CU_EVENT_RECORD_DEFAULT


        Default event record flag


    .. autoattribute:: cuda.cuda.CUevent_record_flags.CU_EVENT_RECORD_EXTERNAL


        When using stream capture, create an event record node instead of the default behavior. This flag is invalid when used outside of capture.

.. autoclass:: cuda.cuda.CUevent_wait_flags

    .. autoattribute:: cuda.cuda.CUevent_wait_flags.CU_EVENT_WAIT_DEFAULT


        Default event wait flag


    .. autoattribute:: cuda.cuda.CUevent_wait_flags.CU_EVENT_WAIT_EXTERNAL


        When using stream capture, create an event wait node instead of the default behavior. This flag is invalid when used outside of capture.

.. autoclass:: cuda.cuda.CUstreamWaitValue_flags

    .. autoattribute:: cuda.cuda.CUstreamWaitValue_flags.CU_STREAM_WAIT_VALUE_GEQ


        Wait until (int32_t)(*addr - value) >= 0 (or int64_t for 64 bit values). Note this is a cyclic comparison which ignores wraparound. (Default behavior.)


    .. autoattribute:: cuda.cuda.CUstreamWaitValue_flags.CU_STREAM_WAIT_VALUE_EQ


        Wait until *addr == value.


    .. autoattribute:: cuda.cuda.CUstreamWaitValue_flags.CU_STREAM_WAIT_VALUE_AND


        Wait until (*addr & value) != 0.


    .. autoattribute:: cuda.cuda.CUstreamWaitValue_flags.CU_STREAM_WAIT_VALUE_NOR


        Wait until ~(*addr | value) != 0. Support for this operation can be queried with :py:obj:`~.cuDeviceGetAttribute()` and :py:obj:`~.CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR`.


    .. autoattribute:: cuda.cuda.CUstreamWaitValue_flags.CU_STREAM_WAIT_VALUE_FLUSH


        Follow the wait operation with a flush of outstanding remote writes. This means that, if a remote write operation is guaranteed to have reached the device before the wait can be satisfied, that write is guaranteed to be visible to downstream device work. The device is permitted to reorder remote writes internally. For example, this flag would be required if two remote writes arrive in a defined order, the wait is satisfied by the second write, and downstream work needs to observe the first write. Support for this operation is restricted to selected platforms and can be queried with :py:obj:`~.CU_DEVICE_ATTRIBUTE_CAN_USE_WAIT_VALUE_FLUSH`.

.. autoclass:: cuda.cuda.CUstreamWriteValue_flags

    .. autoattribute:: cuda.cuda.CUstreamWriteValue_flags.CU_STREAM_WRITE_VALUE_DEFAULT


        Default behavior


    .. autoattribute:: cuda.cuda.CUstreamWriteValue_flags.CU_STREAM_WRITE_VALUE_NO_MEMORY_BARRIER


        Permits the write to be reordered with writes which were issued before it, as a performance optimization. Normally, :py:obj:`~.cuStreamWriteValue32` will provide a memory fence before the write, which has similar semantics to __threadfence_system() but is scoped to the stream rather than a CUDA thread. This flag is not supported in the v2 API.

.. autoclass:: cuda.cuda.CUstreamBatchMemOpType

    .. autoattribute:: cuda.cuda.CUstreamBatchMemOpType.CU_STREAM_MEM_OP_WAIT_VALUE_32


        Represents a :py:obj:`~.cuStreamWaitValue32` operation


    .. autoattribute:: cuda.cuda.CUstreamBatchMemOpType.CU_STREAM_MEM_OP_WRITE_VALUE_32


        Represents a :py:obj:`~.cuStreamWriteValue32` operation


    .. autoattribute:: cuda.cuda.CUstreamBatchMemOpType.CU_STREAM_MEM_OP_WAIT_VALUE_64


        Represents a :py:obj:`~.cuStreamWaitValue64` operation


    .. autoattribute:: cuda.cuda.CUstreamBatchMemOpType.CU_STREAM_MEM_OP_WRITE_VALUE_64


        Represents a :py:obj:`~.cuStreamWriteValue64` operation


    .. autoattribute:: cuda.cuda.CUstreamBatchMemOpType.CU_STREAM_MEM_OP_BARRIER


        Insert a memory barrier of the specified type


    .. autoattribute:: cuda.cuda.CUstreamBatchMemOpType.CU_STREAM_MEM_OP_FLUSH_REMOTE_WRITES


        This has the same effect as :py:obj:`~.CU_STREAM_WAIT_VALUE_FLUSH`, but as a standalone operation.

.. autoclass:: cuda.cuda.CUstreamMemoryBarrier_flags

    .. autoattribute:: cuda.cuda.CUstreamMemoryBarrier_flags.CU_STREAM_MEMORY_BARRIER_TYPE_SYS


        System-wide memory barrier.


    .. autoattribute:: cuda.cuda.CUstreamMemoryBarrier_flags.CU_STREAM_MEMORY_BARRIER_TYPE_GPU


        Limit memory barrier scope to the GPU.

.. autoclass:: cuda.cuda.CUoccupancy_flags

    .. autoattribute:: cuda.cuda.CUoccupancy_flags.CU_OCCUPANCY_DEFAULT


        Default behavior


    .. autoattribute:: cuda.cuda.CUoccupancy_flags.CU_OCCUPANCY_DISABLE_CACHING_OVERRIDE


        Assume global caching is enabled and cannot be automatically turned off

.. autoclass:: cuda.cuda.CUstreamUpdateCaptureDependencies_flags

    .. autoattribute:: cuda.cuda.CUstreamUpdateCaptureDependencies_flags.CU_STREAM_ADD_CAPTURE_DEPENDENCIES


        Add new nodes to the dependency set


    .. autoattribute:: cuda.cuda.CUstreamUpdateCaptureDependencies_flags.CU_STREAM_SET_CAPTURE_DEPENDENCIES


        Replace the dependency set with the new nodes

.. autoclass:: cuda.cuda.CUarray_format

    .. autoattribute:: cuda.cuda.CUarray_format.CU_AD_FORMAT_UNSIGNED_INT8


        Unsigned 8-bit integers


    .. autoattribute:: cuda.cuda.CUarray_format.CU_AD_FORMAT_UNSIGNED_INT16


        Unsigned 16-bit integers


    .. autoattribute:: cuda.cuda.CUarray_format.CU_AD_FORMAT_UNSIGNED_INT32


        Unsigned 32-bit integers


    .. autoattribute:: cuda.cuda.CUarray_format.CU_AD_FORMAT_SIGNED_INT8


        Signed 8-bit integers


    .. autoattribute:: cuda.cuda.CUarray_format.CU_AD_FORMAT_SIGNED_INT16


        Signed 16-bit integers


    .. autoattribute:: cuda.cuda.CUarray_format.CU_AD_FORMAT_SIGNED_INT32


        Signed 32-bit integers


    .. autoattribute:: cuda.cuda.CUarray_format.CU_AD_FORMAT_HALF


        16-bit floating point


    .. autoattribute:: cuda.cuda.CUarray_format.CU_AD_FORMAT_FLOAT


        32-bit floating point


    .. autoattribute:: cuda.cuda.CUarray_format.CU_AD_FORMAT_NV12


        8-bit YUV planar format, with 4:2:0 sampling


    .. autoattribute:: cuda.cuda.CUarray_format.CU_AD_FORMAT_UNORM_INT8X1


        1 channel unsigned 8-bit normalized integer


    .. autoattribute:: cuda.cuda.CUarray_format.CU_AD_FORMAT_UNORM_INT8X2


        2 channel unsigned 8-bit normalized integer


    .. autoattribute:: cuda.cuda.CUarray_format.CU_AD_FORMAT_UNORM_INT8X4


        4 channel unsigned 8-bit normalized integer


    .. autoattribute:: cuda.cuda.CUarray_format.CU_AD_FORMAT_UNORM_INT16X1


        1 channel unsigned 16-bit normalized integer


    .. autoattribute:: cuda.cuda.CUarray_format.CU_AD_FORMAT_UNORM_INT16X2


        2 channel unsigned 16-bit normalized integer


    .. autoattribute:: cuda.cuda.CUarray_format.CU_AD_FORMAT_UNORM_INT16X4


        4 channel unsigned 16-bit normalized integer


    .. autoattribute:: cuda.cuda.CUarray_format.CU_AD_FORMAT_SNORM_INT8X1


        1 channel signed 8-bit normalized integer


    .. autoattribute:: cuda.cuda.CUarray_format.CU_AD_FORMAT_SNORM_INT8X2


        2 channel signed 8-bit normalized integer


    .. autoattribute:: cuda.cuda.CUarray_format.CU_AD_FORMAT_SNORM_INT8X4


        4 channel signed 8-bit normalized integer


    .. autoattribute:: cuda.cuda.CUarray_format.CU_AD_FORMAT_SNORM_INT16X1


        1 channel signed 16-bit normalized integer


    .. autoattribute:: cuda.cuda.CUarray_format.CU_AD_FORMAT_SNORM_INT16X2


        2 channel signed 16-bit normalized integer


    .. autoattribute:: cuda.cuda.CUarray_format.CU_AD_FORMAT_SNORM_INT16X4


        4 channel signed 16-bit normalized integer


    .. autoattribute:: cuda.cuda.CUarray_format.CU_AD_FORMAT_BC1_UNORM


        4 channel unsigned normalized block-compressed (BC1 compression) format


    .. autoattribute:: cuda.cuda.CUarray_format.CU_AD_FORMAT_BC1_UNORM_SRGB


        4 channel unsigned normalized block-compressed (BC1 compression) format with sRGB encoding


    .. autoattribute:: cuda.cuda.CUarray_format.CU_AD_FORMAT_BC2_UNORM


        4 channel unsigned normalized block-compressed (BC2 compression) format


    .. autoattribute:: cuda.cuda.CUarray_format.CU_AD_FORMAT_BC2_UNORM_SRGB


        4 channel unsigned normalized block-compressed (BC2 compression) format with sRGB encoding


    .. autoattribute:: cuda.cuda.CUarray_format.CU_AD_FORMAT_BC3_UNORM


        4 channel unsigned normalized block-compressed (BC3 compression) format


    .. autoattribute:: cuda.cuda.CUarray_format.CU_AD_FORMAT_BC3_UNORM_SRGB


        4 channel unsigned normalized block-compressed (BC3 compression) format with sRGB encoding


    .. autoattribute:: cuda.cuda.CUarray_format.CU_AD_FORMAT_BC4_UNORM


        1 channel unsigned normalized block-compressed (BC4 compression) format


    .. autoattribute:: cuda.cuda.CUarray_format.CU_AD_FORMAT_BC4_SNORM


        1 channel signed normalized block-compressed (BC4 compression) format


    .. autoattribute:: cuda.cuda.CUarray_format.CU_AD_FORMAT_BC5_UNORM


        2 channel unsigned normalized block-compressed (BC5 compression) format


    .. autoattribute:: cuda.cuda.CUarray_format.CU_AD_FORMAT_BC5_SNORM


        2 channel signed normalized block-compressed (BC5 compression) format


    .. autoattribute:: cuda.cuda.CUarray_format.CU_AD_FORMAT_BC6H_UF16


        3 channel unsigned half-float block-compressed (BC6H compression) format


    .. autoattribute:: cuda.cuda.CUarray_format.CU_AD_FORMAT_BC6H_SF16


        3 channel signed half-float block-compressed (BC6H compression) format


    .. autoattribute:: cuda.cuda.CUarray_format.CU_AD_FORMAT_BC7_UNORM


        4 channel unsigned normalized block-compressed (BC7 compression) format


    .. autoattribute:: cuda.cuda.CUarray_format.CU_AD_FORMAT_BC7_UNORM_SRGB


        4 channel unsigned normalized block-compressed (BC7 compression) format with sRGB encoding

.. autoclass:: cuda.cuda.CUaddress_mode

    .. autoattribute:: cuda.cuda.CUaddress_mode.CU_TR_ADDRESS_MODE_WRAP


        Wrapping address mode


    .. autoattribute:: cuda.cuda.CUaddress_mode.CU_TR_ADDRESS_MODE_CLAMP


        Clamp to edge address mode


    .. autoattribute:: cuda.cuda.CUaddress_mode.CU_TR_ADDRESS_MODE_MIRROR


        Mirror address mode


    .. autoattribute:: cuda.cuda.CUaddress_mode.CU_TR_ADDRESS_MODE_BORDER


        Border address mode

.. autoclass:: cuda.cuda.CUfilter_mode

    .. autoattribute:: cuda.cuda.CUfilter_mode.CU_TR_FILTER_MODE_POINT


        Point filter mode


    .. autoattribute:: cuda.cuda.CUfilter_mode.CU_TR_FILTER_MODE_LINEAR


        Linear filter mode

.. autoclass:: cuda.cuda.CUdevice_attribute

    .. autoattribute:: cuda.cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK


        Maximum number of threads per block


    .. autoattribute:: cuda.cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X


        Maximum block dimension X


    .. autoattribute:: cuda.cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y


        Maximum block dimension Y


    .. autoattribute:: cuda.cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z


        Maximum block dimension Z


    .. autoattribute:: cuda.cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X


        Maximum grid dimension X


    .. autoattribute:: cuda.cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y


        Maximum grid dimension Y


    .. autoattribute:: cuda.cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z


        Maximum grid dimension Z


    .. autoattribute:: cuda.cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK


        Maximum shared memory available per block in bytes


    .. autoattribute:: cuda.cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_SHARED_MEMORY_PER_BLOCK


        Deprecated, use CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK


    .. autoattribute:: cuda.cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY


        Memory available on device for constant variables in a CUDA C kernel in bytes


    .. autoattribute:: cuda.cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_WARP_SIZE


        Warp size in threads


    .. autoattribute:: cuda.cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_PITCH


        Maximum pitch in bytes allowed by memory copies


    .. autoattribute:: cuda.cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK


        Maximum number of 32-bit registers available per block


    .. autoattribute:: cuda.cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_REGISTERS_PER_BLOCK


        Deprecated, use CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK


    .. autoattribute:: cuda.cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_CLOCK_RATE


        Typical clock frequency in kilohertz


    .. autoattribute:: cuda.cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT


        Alignment requirement for textures


    .. autoattribute:: cuda.cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_GPU_OVERLAP


        Device can possibly copy memory and execute a kernel concurrently. Deprecated. Use instead CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT.


    .. autoattribute:: cuda.cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT


        Number of multiprocessors on device


    .. autoattribute:: cuda.cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT


        Specifies whether there is a run time limit on kernels


    .. autoattribute:: cuda.cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_INTEGRATED


        Device is integrated with host memory


    .. autoattribute:: cuda.cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY


        Device can map host memory into CUDA address space


    .. autoattribute:: cuda.cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_MODE


        Compute mode (See :py:obj:`~.CUcomputemode` for details)


    .. autoattribute:: cuda.cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH


        Maximum 1D texture width


    .. autoattribute:: cuda.cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH


        Maximum 2D texture width


    .. autoattribute:: cuda.cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT


        Maximum 2D texture height


    .. autoattribute:: cuda.cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH


        Maximum 3D texture width


    .. autoattribute:: cuda.cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT


        Maximum 3D texture height


    .. autoattribute:: cuda.cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH


        Maximum 3D texture depth


    .. autoattribute:: cuda.cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH


        Maximum 2D layered texture width


    .. autoattribute:: cuda.cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT


        Maximum 2D layered texture height


    .. autoattribute:: cuda.cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS


        Maximum layers in a 2D layered texture


    .. autoattribute:: cuda.cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_WIDTH


        Deprecated, use CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH


    .. autoattribute:: cuda.cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_HEIGHT


        Deprecated, use CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT


    .. autoattribute:: cuda.cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_NUMSLICES


        Deprecated, use CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS


    .. autoattribute:: cuda.cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT


        Alignment requirement for surfaces


    .. autoattribute:: cuda.cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS


        Device can possibly execute multiple kernels concurrently


    .. autoattribute:: cuda.cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_ECC_ENABLED


        Device has ECC support enabled


    .. autoattribute:: cuda.cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_PCI_BUS_ID


        PCI bus ID of the device


    .. autoattribute:: cuda.cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID


        PCI device ID of the device


    .. autoattribute:: cuda.cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_TCC_DRIVER


        Device is using TCC driver model


    .. autoattribute:: cuda.cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE


        Peak memory clock frequency in kilohertz


    .. autoattribute:: cuda.cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH


        Global memory bus width in bits


    .. autoattribute:: cuda.cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE


        Size of L2 cache in bytes


    .. autoattribute:: cuda.cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR


        Maximum resident threads per multiprocessor


    .. autoattribute:: cuda.cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT


        Number of asynchronous engines


    .. autoattribute:: cuda.cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING


        Device shares a unified address space with the host


    .. autoattribute:: cuda.cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH


        Maximum 1D layered texture width


    .. autoattribute:: cuda.cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS


        Maximum layers in a 1D layered texture


    .. autoattribute:: cuda.cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_CAN_TEX2D_GATHER


        Deprecated, do not use.


    .. autoattribute:: cuda.cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_WIDTH


        Maximum 2D texture width if CUDA_ARRAY3D_TEXTURE_GATHER is set


    .. autoattribute:: cuda.cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_HEIGHT


        Maximum 2D texture height if CUDA_ARRAY3D_TEXTURE_GATHER is set


    .. autoattribute:: cuda.cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH_ALTERNATE


        Alternate maximum 3D texture width


    .. autoattribute:: cuda.cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT_ALTERNATE


        Alternate maximum 3D texture height


    .. autoattribute:: cuda.cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH_ALTERNATE


        Alternate maximum 3D texture depth


    .. autoattribute:: cuda.cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID


        PCI domain ID of the device


    .. autoattribute:: cuda.cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT


        Pitch alignment requirement for textures


    .. autoattribute:: cuda.cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_WIDTH


        Maximum cubemap texture width/height


    .. autoattribute:: cuda.cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_WIDTH


        Maximum cubemap layered texture width/height


    .. autoattribute:: cuda.cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_LAYERS


        Maximum layers in a cubemap layered texture


    .. autoattribute:: cuda.cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_WIDTH


        Maximum 1D surface width


    .. autoattribute:: cuda.cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_WIDTH


        Maximum 2D surface width


    .. autoattribute:: cuda.cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_HEIGHT


        Maximum 2D surface height


    .. autoattribute:: cuda.cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_WIDTH


        Maximum 3D surface width


    .. autoattribute:: cuda.cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_HEIGHT


        Maximum 3D surface height


    .. autoattribute:: cuda.cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_DEPTH


        Maximum 3D surface depth


    .. autoattribute:: cuda.cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_WIDTH


        Maximum 1D layered surface width


    .. autoattribute:: cuda.cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_LAYERS


        Maximum layers in a 1D layered surface


    .. autoattribute:: cuda.cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_WIDTH


        Maximum 2D layered surface width


    .. autoattribute:: cuda.cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_HEIGHT


        Maximum 2D layered surface height


    .. autoattribute:: cuda.cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_LAYERS


        Maximum layers in a 2D layered surface


    .. autoattribute:: cuda.cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_WIDTH


        Maximum cubemap surface width


    .. autoattribute:: cuda.cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_WIDTH


        Maximum cubemap layered surface width


    .. autoattribute:: cuda.cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_LAYERS


        Maximum layers in a cubemap layered surface


    .. autoattribute:: cuda.cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LINEAR_WIDTH


        Deprecated, do not use. Use cudaDeviceGetTexture1DLinearMaxWidth() or :py:obj:`~.cuDeviceGetTexture1DLinearMaxWidth()` instead.


    .. autoattribute:: cuda.cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_WIDTH


        Maximum 2D linear texture width


    .. autoattribute:: cuda.cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_HEIGHT


        Maximum 2D linear texture height


    .. autoattribute:: cuda.cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_PITCH


        Maximum 2D linear texture pitch in bytes


    .. autoattribute:: cuda.cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_WIDTH


        Maximum mipmapped 2D texture width


    .. autoattribute:: cuda.cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_HEIGHT


        Maximum mipmapped 2D texture height


    .. autoattribute:: cuda.cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR


        Major compute capability version number


    .. autoattribute:: cuda.cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR


        Minor compute capability version number


    .. autoattribute:: cuda.cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH


        Maximum mipmapped 1D texture width


    .. autoattribute:: cuda.cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED


        Device supports stream priorities


    .. autoattribute:: cuda.cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED


        Device supports caching globals in L1


    .. autoattribute:: cuda.cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED


        Device supports caching locals in L1


    .. autoattribute:: cuda.cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR


        Maximum shared memory available per multiprocessor in bytes


    .. autoattribute:: cuda.cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR


        Maximum number of 32-bit registers available per multiprocessor


    .. autoattribute:: cuda.cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY


        Device can allocate managed memory on this system


    .. autoattribute:: cuda.cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD


        Device is on a multi-GPU board


    .. autoattribute:: cuda.cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID


        Unique id for a group of devices on the same multi-GPU board


    .. autoattribute:: cuda.cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED


        Link between the device and the host supports native atomic operations (this is a placeholder attribute, and is not supported on any current hardware)


    .. autoattribute:: cuda.cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO


        Ratio of single precision performance (in floating-point operations per second) to double precision performance


    .. autoattribute:: cuda.cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS


        Device supports coherently accessing pageable memory without calling cudaHostRegister on it


    .. autoattribute:: cuda.cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS


        Device can coherently access managed memory concurrently with the CPU


    .. autoattribute:: cuda.cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED


        Device supports compute preemption.


    .. autoattribute:: cuda.cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM


        Device can access host registered memory at the same virtual address as the CPU


    .. autoattribute:: cuda.cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS


        :py:obj:`~.cuStreamBatchMemOp` and related APIs are supported.


    .. autoattribute:: cuda.cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS


        64-bit operations are supported in :py:obj:`~.cuStreamBatchMemOp` and related APIs.


    .. autoattribute:: cuda.cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR


        :py:obj:`~.CU_STREAM_WAIT_VALUE_NOR` is supported.


    .. autoattribute:: cuda.cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH


        Device supports launching cooperative kernels via :py:obj:`~.cuLaunchCooperativeKernel`


    .. autoattribute:: cuda.cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COOPERATIVE_MULTI_DEVICE_LAUNCH


        Deprecated, :py:obj:`~.cuLaunchCooperativeKernelMultiDevice` is deprecated.


    .. autoattribute:: cuda.cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN


        Maximum optin shared memory per block


    .. autoattribute:: cuda.cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_CAN_FLUSH_REMOTE_WRITES


        The :py:obj:`~.CU_STREAM_WAIT_VALUE_FLUSH` flag and the :py:obj:`~.CU_STREAM_MEM_OP_FLUSH_REMOTE_WRITES` MemOp are supported on the device. See :py:obj:`~.Stream Memory Operations` for additional details.


    .. autoattribute:: cuda.cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_HOST_REGISTER_SUPPORTED


        Device supports host memory registration via :py:obj:`~.cudaHostRegister`.


    .. autoattribute:: cuda.cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES


        Device accesses pageable memory via the host's page tables.


    .. autoattribute:: cuda.cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_DIRECT_MANAGED_MEM_ACCESS_FROM_HOST


        The host can directly access managed memory on the device without migration.


    .. autoattribute:: cuda.cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED


        Deprecated, Use CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED


    .. autoattribute:: cuda.cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED


        Device supports virtual memory management APIs like :py:obj:`~.cuMemAddressReserve`, :py:obj:`~.cuMemCreate`, :py:obj:`~.cuMemMap` and related APIs


    .. autoattribute:: cuda.cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED


        Device supports exporting memory to a posix file descriptor with :py:obj:`~.cuMemExportToShareableHandle`, if requested via :py:obj:`~.cuMemCreate`


    .. autoattribute:: cuda.cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_HANDLE_SUPPORTED


        Device supports exporting memory to a Win32 NT handle with :py:obj:`~.cuMemExportToShareableHandle`, if requested via :py:obj:`~.cuMemCreate`


    .. autoattribute:: cuda.cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_KMT_HANDLE_SUPPORTED


        Device supports exporting memory to a Win32 KMT handle with :py:obj:`~.cuMemExportToShareableHandle`, if requested via :py:obj:`~.cuMemCreate`


    .. autoattribute:: cuda.cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR


        Maximum number of blocks per multiprocessor


    .. autoattribute:: cuda.cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_GENERIC_COMPRESSION_SUPPORTED


        Device supports compression of memory


    .. autoattribute:: cuda.cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_PERSISTING_L2_CACHE_SIZE


        Maximum L2 persisting lines capacity setting in bytes.


    .. autoattribute:: cuda.cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_ACCESS_POLICY_WINDOW_SIZE


        Maximum value of :py:obj:`~.CUaccessPolicyWindow.num_bytes`.


    .. autoattribute:: cuda.cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED


        Device supports specifying the GPUDirect RDMA flag with :py:obj:`~.cuMemCreate`


    .. autoattribute:: cuda.cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_RESERVED_SHARED_MEMORY_PER_BLOCK


        Shared memory reserved by CUDA driver per block in bytes


    .. autoattribute:: cuda.cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_SPARSE_CUDA_ARRAY_SUPPORTED


        Device supports sparse CUDA arrays and sparse CUDA mipmapped arrays


    .. autoattribute:: cuda.cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_READ_ONLY_HOST_REGISTER_SUPPORTED


        Device supports using the :py:obj:`~.cuMemHostRegister` flag :py:obj:`~.CU_MEMHOSTERGISTER_READ_ONLY` to register memory that must be mapped as read-only to the GPU


    .. autoattribute:: cuda.cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_TIMELINE_SEMAPHORE_INTEROP_SUPPORTED


        External timeline semaphore interop is supported on the device


    .. autoattribute:: cuda.cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED


        Device supports using the :py:obj:`~.cuMemAllocAsync` and :py:obj:`~.cuMemPool` family of APIs


    .. autoattribute:: cuda.cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED


        Device supports GPUDirect RDMA APIs, like nvidia_p2p_get_pages (see https://docs.nvidia.com/cuda/gpudirect-rdma for more information)


    .. autoattribute:: cuda.cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_FLUSH_WRITES_OPTIONS


        The returned attribute shall be interpreted as a bitmask, where the individual bits are described by the :py:obj:`~.CUflushGPUDirectRDMAWritesOptions` enum


    .. autoattribute:: cuda.cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WRITES_ORDERING


        GPUDirect RDMA writes to the device do not need to be flushed for consumers within the scope indicated by the returned attribute. See :py:obj:`~.CUGPUDirectRDMAWritesOrdering` for the numerical values returned here.


    .. autoattribute:: cuda.cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MEMPOOL_SUPPORTED_HANDLE_TYPES


        Handle types supported with mempool based IPC


    .. autoattribute:: cuda.cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_DEFERRED_MAPPING_CUDA_ARRAY_SUPPORTED


        Device supports deferred mapping CUDA arrays and CUDA mipmapped arrays


    .. autoattribute:: cuda.cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS_V2


        64-bit operations are supported in :py:obj:`~.cuStreamBatchMemOp_v2` and related v2 MemOp APIs.


    .. autoattribute:: cuda.cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR_V2


        :py:obj:`~.CU_STREAM_WAIT_VALUE_NOR` is supported by v2 MemOp APIs.


    .. autoattribute:: cuda.cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED


        Device supports buffer sharing with dma_buf mechanism.


    .. autoattribute:: cuda.cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX

.. autoclass:: cuda.cuda.CUpointer_attribute

    .. autoattribute:: cuda.cuda.CUpointer_attribute.CU_POINTER_ATTRIBUTE_CONTEXT


        The :py:obj:`~.CUcontext` on which a pointer was allocated or registered


    .. autoattribute:: cuda.cuda.CUpointer_attribute.CU_POINTER_ATTRIBUTE_MEMORY_TYPE


        The :py:obj:`~.CUmemorytype` describing the physical location of a pointer


    .. autoattribute:: cuda.cuda.CUpointer_attribute.CU_POINTER_ATTRIBUTE_DEVICE_POINTER


        The address at which a pointer's memory may be accessed on the device


    .. autoattribute:: cuda.cuda.CUpointer_attribute.CU_POINTER_ATTRIBUTE_HOST_POINTER


        The address at which a pointer's memory may be accessed on the host


    .. autoattribute:: cuda.cuda.CUpointer_attribute.CU_POINTER_ATTRIBUTE_P2P_TOKENS


        A pair of tokens for use with the nv-p2p.h Linux kernel interface


    .. autoattribute:: cuda.cuda.CUpointer_attribute.CU_POINTER_ATTRIBUTE_SYNC_MEMOPS


        Synchronize every synchronous memory operation initiated on this region


    .. autoattribute:: cuda.cuda.CUpointer_attribute.CU_POINTER_ATTRIBUTE_BUFFER_ID


        A process-wide unique ID for an allocated memory region


    .. autoattribute:: cuda.cuda.CUpointer_attribute.CU_POINTER_ATTRIBUTE_IS_MANAGED


        Indicates if the pointer points to managed memory


    .. autoattribute:: cuda.cuda.CUpointer_attribute.CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL


        A device ordinal of a device on which a pointer was allocated or registered


    .. autoattribute:: cuda.cuda.CUpointer_attribute.CU_POINTER_ATTRIBUTE_IS_LEGACY_CUDA_IPC_CAPABLE


        1 if this pointer maps to an allocation that is suitable for :py:obj:`~.cudaIpcGetMemHandle`, 0 otherwise


    .. autoattribute:: cuda.cuda.CUpointer_attribute.CU_POINTER_ATTRIBUTE_RANGE_START_ADDR


        Starting address for this requested pointer


    .. autoattribute:: cuda.cuda.CUpointer_attribute.CU_POINTER_ATTRIBUTE_RANGE_SIZE


        Size of the address range for this requested pointer


    .. autoattribute:: cuda.cuda.CUpointer_attribute.CU_POINTER_ATTRIBUTE_MAPPED


        1 if this pointer is in a valid address range that is mapped to a backing allocation, 0 otherwise


    .. autoattribute:: cuda.cuda.CUpointer_attribute.CU_POINTER_ATTRIBUTE_ALLOWED_HANDLE_TYPES


        Bitmask of allowed :py:obj:`~.CUmemAllocationHandleType` for this allocation


    .. autoattribute:: cuda.cuda.CUpointer_attribute.CU_POINTER_ATTRIBUTE_IS_GPU_DIRECT_RDMA_CAPABLE


        1 if the memory this pointer is referencing can be used with the GPUDirect RDMA API


    .. autoattribute:: cuda.cuda.CUpointer_attribute.CU_POINTER_ATTRIBUTE_ACCESS_FLAGS


        Returns the access flags the device associated with the current context has on the corresponding memory referenced by the pointer given


    .. autoattribute:: cuda.cuda.CUpointer_attribute.CU_POINTER_ATTRIBUTE_MEMPOOL_HANDLE


        Returns the mempool handle for the allocation if it was allocated from a mempool. Otherwise returns NULL.


    .. autoattribute:: cuda.cuda.CUpointer_attribute.CU_POINTER_ATTRIBUTE_MAPPING_SIZE


        Size of the actual underlying mapping that the pointer belongs to


    .. autoattribute:: cuda.cuda.CUpointer_attribute.CU_POINTER_ATTRIBUTE_MAPPING_BASE_ADDR


        The start address of the mapping that the pointer belongs to


    .. autoattribute:: cuda.cuda.CUpointer_attribute.CU_POINTER_ATTRIBUTE_MEMORY_BLOCK_ID


        A process-wide unique id corresponding to the physical allocation the pointer belongs to

.. autoclass:: cuda.cuda.CUfunction_attribute

    .. autoattribute:: cuda.cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK


        The maximum number of threads per block, beyond which a launch of the function would fail. This number depends on both the function and the device on which the function is currently loaded.


    .. autoattribute:: cuda.cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES


        The size in bytes of statically-allocated shared memory required by this function. This does not include dynamically-allocated shared memory requested by the user at runtime.


    .. autoattribute:: cuda.cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES


        The size in bytes of user-allocated constant memory required by this function.


    .. autoattribute:: cuda.cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES


        The size in bytes of local memory used by each thread of this function.


    .. autoattribute:: cuda.cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_NUM_REGS


        The number of registers used by each thread of this function.


    .. autoattribute:: cuda.cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_PTX_VERSION


        The PTX virtual architecture version for which the function was compiled. This value is the major PTX version * 10 + the minor PTX version, so a PTX version 1.3 function would return the value 13. Note that this may return the undefined value of 0 for cubins compiled prior to CUDA 3.0.


    .. autoattribute:: cuda.cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_BINARY_VERSION


        The binary architecture version for which the function was compiled. This value is the major binary version * 10 + the minor binary version, so a binary version 1.3 function would return the value 13. Note that this will return a value of 10 for legacy cubins that do not have a properly-encoded binary architecture version.


    .. autoattribute:: cuda.cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_CACHE_MODE_CA


        The attribute to indicate whether the function has been compiled with user specified option "-Xptxas --dlcm=ca" set .


    .. autoattribute:: cuda.cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES


        The maximum size in bytes of dynamically-allocated shared memory that can be used by this function. If the user-specified dynamic shared memory size is larger than this value, the launch will fail. See :py:obj:`~.cuFuncSetAttribute`


    .. autoattribute:: cuda.cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT


        On devices where the L1 cache and shared memory use the same hardware resources, this sets the shared memory carveout preference, in percent of the total shared memory. Refer to :py:obj:`~.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR`. This is only a hint, and the driver can choose a different ratio if required to execute the function. See :py:obj:`~.cuFuncSetAttribute`


    .. autoattribute:: cuda.cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX

.. autoclass:: cuda.cuda.CUfunc_cache

    .. autoattribute:: cuda.cuda.CUfunc_cache.CU_FUNC_CACHE_PREFER_NONE


        no preference for shared memory or L1 (default)


    .. autoattribute:: cuda.cuda.CUfunc_cache.CU_FUNC_CACHE_PREFER_SHARED


        prefer larger shared memory and smaller L1 cache


    .. autoattribute:: cuda.cuda.CUfunc_cache.CU_FUNC_CACHE_PREFER_L1


        prefer larger L1 cache and smaller shared memory


    .. autoattribute:: cuda.cuda.CUfunc_cache.CU_FUNC_CACHE_PREFER_EQUAL


        prefer equal sized L1 cache and shared memory

.. autoclass:: cuda.cuda.CUsharedconfig

    .. autoattribute:: cuda.cuda.CUsharedconfig.CU_SHARED_MEM_CONFIG_DEFAULT_BANK_SIZE


        set default shared memory bank size


    .. autoattribute:: cuda.cuda.CUsharedconfig.CU_SHARED_MEM_CONFIG_FOUR_BYTE_BANK_SIZE


        set shared memory bank width to four bytes


    .. autoattribute:: cuda.cuda.CUsharedconfig.CU_SHARED_MEM_CONFIG_EIGHT_BYTE_BANK_SIZE


        set shared memory bank width to eight bytes

.. autoclass:: cuda.cuda.CUshared_carveout

    .. autoattribute:: cuda.cuda.CUshared_carveout.CU_SHAREDMEM_CARVEOUT_DEFAULT


        No preference for shared memory or L1 (default)


    .. autoattribute:: cuda.cuda.CUshared_carveout.CU_SHAREDMEM_CARVEOUT_MAX_SHARED


        Prefer maximum available shared memory, minimum L1 cache


    .. autoattribute:: cuda.cuda.CUshared_carveout.CU_SHAREDMEM_CARVEOUT_MAX_L1


        Prefer maximum available L1 cache, minimum shared memory

.. autoclass:: cuda.cuda.CUmemorytype

    .. autoattribute:: cuda.cuda.CUmemorytype.CU_MEMORYTYPE_HOST


        Host memory


    .. autoattribute:: cuda.cuda.CUmemorytype.CU_MEMORYTYPE_DEVICE


        Device memory


    .. autoattribute:: cuda.cuda.CUmemorytype.CU_MEMORYTYPE_ARRAY


        Array memory


    .. autoattribute:: cuda.cuda.CUmemorytype.CU_MEMORYTYPE_UNIFIED


        Unified device or host memory

.. autoclass:: cuda.cuda.CUcomputemode

    .. autoattribute:: cuda.cuda.CUcomputemode.CU_COMPUTEMODE_DEFAULT


        Default compute mode (Multiple contexts allowed per device)


    .. autoattribute:: cuda.cuda.CUcomputemode.CU_COMPUTEMODE_PROHIBITED


        Compute-prohibited mode (No contexts can be created on this device at this time)


    .. autoattribute:: cuda.cuda.CUcomputemode.CU_COMPUTEMODE_EXCLUSIVE_PROCESS


        Compute-exclusive-process mode (Only one context used by a single process can be present on this device at a time)

.. autoclass:: cuda.cuda.CUmem_advise

    .. autoattribute:: cuda.cuda.CUmem_advise.CU_MEM_ADVISE_SET_READ_MOSTLY


        Data will mostly be read and only occassionally be written to


    .. autoattribute:: cuda.cuda.CUmem_advise.CU_MEM_ADVISE_UNSET_READ_MOSTLY


        Undo the effect of :py:obj:`~.CU_MEM_ADVISE_SET_READ_MOSTLY`


    .. autoattribute:: cuda.cuda.CUmem_advise.CU_MEM_ADVISE_SET_PREFERRED_LOCATION


        Set the preferred location for the data as the specified device


    .. autoattribute:: cuda.cuda.CUmem_advise.CU_MEM_ADVISE_UNSET_PREFERRED_LOCATION


        Clear the preferred location for the data


    .. autoattribute:: cuda.cuda.CUmem_advise.CU_MEM_ADVISE_SET_ACCESSED_BY


        Data will be accessed by the specified device, so prevent page faults as much as possible


    .. autoattribute:: cuda.cuda.CUmem_advise.CU_MEM_ADVISE_UNSET_ACCESSED_BY


        Let the Unified Memory subsystem decide on the page faulting policy for the specified device

.. autoclass:: cuda.cuda.CUmem_range_attribute

    .. autoattribute:: cuda.cuda.CUmem_range_attribute.CU_MEM_RANGE_ATTRIBUTE_READ_MOSTLY


        Whether the range will mostly be read and only occassionally be written to


    .. autoattribute:: cuda.cuda.CUmem_range_attribute.CU_MEM_RANGE_ATTRIBUTE_PREFERRED_LOCATION


        The preferred location of the range


    .. autoattribute:: cuda.cuda.CUmem_range_attribute.CU_MEM_RANGE_ATTRIBUTE_ACCESSED_BY


        Memory range has :py:obj:`~.CU_MEM_ADVISE_SET_ACCESSED_BY` set for specified device


    .. autoattribute:: cuda.cuda.CUmem_range_attribute.CU_MEM_RANGE_ATTRIBUTE_LAST_PREFETCH_LOCATION


        The last location to which the range was prefetched

.. autoclass:: cuda.cuda.CUjit_option

    .. autoattribute:: cuda.cuda.CUjit_option.CU_JIT_MAX_REGISTERS


        Max number of registers that a thread may use.

        Option type: unsigned int

        Applies to: compiler only


    .. autoattribute:: cuda.cuda.CUjit_option.CU_JIT_THREADS_PER_BLOCK


        IN: Specifies minimum number of threads per block to target compilation for

        OUT: Returns the number of threads the compiler actually targeted. This restricts the resource utilization fo the compiler (e.g. max registers) such that a block with the given number of threads should be able to launch based on register limitations. Note, this option does not currently take into account any other resource limitations, such as shared memory utilization.

        Cannot be combined with :py:obj:`~.CU_JIT_TARGET`.

        Option type: unsigned int

        Applies to: compiler only


    .. autoattribute:: cuda.cuda.CUjit_option.CU_JIT_WALL_TIME


        Overwrites the option value with the total wall clock time, in milliseconds, spent in the compiler and linker

        Option type: float

        Applies to: compiler and linker


    .. autoattribute:: cuda.cuda.CUjit_option.CU_JIT_INFO_LOG_BUFFER


        Pointer to a buffer in which to print any log messages that are informational in nature (the buffer size is specified via option :py:obj:`~.CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES`)

        Option type: char *

        Applies to: compiler and linker


    .. autoattribute:: cuda.cuda.CUjit_option.CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES


        IN: Log buffer size in bytes. Log messages will be capped at this size (including null terminator)

        OUT: Amount of log buffer filled with messages

        Option type: unsigned int

        Applies to: compiler and linker


    .. autoattribute:: cuda.cuda.CUjit_option.CU_JIT_ERROR_LOG_BUFFER


        Pointer to a buffer in which to print any log messages that reflect errors (the buffer size is specified via option :py:obj:`~.CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES`)

        Option type: char *

        Applies to: compiler and linker


    .. autoattribute:: cuda.cuda.CUjit_option.CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES


        IN: Log buffer size in bytes. Log messages will be capped at this size (including null terminator)

        OUT: Amount of log buffer filled with messages

        Option type: unsigned int

        Applies to: compiler and linker


    .. autoattribute:: cuda.cuda.CUjit_option.CU_JIT_OPTIMIZATION_LEVEL


        Level of optimizations to apply to generated code (0 - 4), with 4 being the default and highest level of optimizations.

        Option type: unsigned int

        Applies to: compiler only


    .. autoattribute:: cuda.cuda.CUjit_option.CU_JIT_TARGET_FROM_CUCONTEXT


        No option value required. Determines the target based on the current attached context (default)

        Option type: No option value needed

        Applies to: compiler and linker


    .. autoattribute:: cuda.cuda.CUjit_option.CU_JIT_TARGET


        Target is chosen based on supplied :py:obj:`~.CUjit_target`. Cannot be combined with :py:obj:`~.CU_JIT_THREADS_PER_BLOCK`.

        Option type: unsigned int for enumerated type :py:obj:`~.CUjit_target`

        Applies to: compiler and linker


    .. autoattribute:: cuda.cuda.CUjit_option.CU_JIT_FALLBACK_STRATEGY


        Specifies choice of fallback strategy if matching cubin is not found. Choice is based on supplied :py:obj:`~.CUjit_fallback`. This option cannot be used with cuLink* APIs as the linker requires exact matches.

        Option type: unsigned int for enumerated type :py:obj:`~.CUjit_fallback`

        Applies to: compiler only


    .. autoattribute:: cuda.cuda.CUjit_option.CU_JIT_GENERATE_DEBUG_INFO


        Specifies whether to create debug information in output (-g) (0: false, default)

        Option type: int

        Applies to: compiler and linker


    .. autoattribute:: cuda.cuda.CUjit_option.CU_JIT_LOG_VERBOSE


        Generate verbose log messages (0: false, default)

        Option type: int

        Applies to: compiler and linker


    .. autoattribute:: cuda.cuda.CUjit_option.CU_JIT_GENERATE_LINE_INFO


        Generate line number information (-lineinfo) (0: false, default)

        Option type: int

        Applies to: compiler only


    .. autoattribute:: cuda.cuda.CUjit_option.CU_JIT_CACHE_MODE


        Specifies whether to enable caching explicitly (-dlcm) 

        Choice is based on supplied :py:obj:`~.CUjit_cacheMode_enum`.

        Option type: unsigned int for enumerated type :py:obj:`~.CUjit_cacheMode_enum`

        Applies to: compiler only


    .. autoattribute:: cuda.cuda.CUjit_option.CU_JIT_NEW_SM3X_OPT


        [Deprecated]


    .. autoattribute:: cuda.cuda.CUjit_option.CU_JIT_FAST_COMPILE


        This jit option is used for internal purpose only.


    .. autoattribute:: cuda.cuda.CUjit_option.CU_JIT_GLOBAL_SYMBOL_NAMES


        Array of device symbol names that will be relocated to the corresponing host addresses stored in :py:obj:`~.CU_JIT_GLOBAL_SYMBOL_ADDRESSES`.

        Must contain :py:obj:`~.CU_JIT_GLOBAL_SYMBOL_COUNT` entries.

        When loding a device module, driver will relocate all encountered unresolved symbols to the host addresses.

        It is only allowed to register symbols that correspond to unresolved global variables.

        It is illegal to register the same device symbol at multiple addresses.

        Option type: const char **

        Applies to: dynamic linker only


    .. autoattribute:: cuda.cuda.CUjit_option.CU_JIT_GLOBAL_SYMBOL_ADDRESSES


        Array of host addresses that will be used to relocate corresponding device symbols stored in :py:obj:`~.CU_JIT_GLOBAL_SYMBOL_NAMES`.

        Must contain :py:obj:`~.CU_JIT_GLOBAL_SYMBOL_COUNT` entries.

        Option type: void **

        Applies to: dynamic linker only


    .. autoattribute:: cuda.cuda.CUjit_option.CU_JIT_GLOBAL_SYMBOL_COUNT


        Number of entries in :py:obj:`~.CU_JIT_GLOBAL_SYMBOL_NAMES` and :py:obj:`~.CU_JIT_GLOBAL_SYMBOL_ADDRESSES` arrays.

        Option type: unsigned int

        Applies to: dynamic linker only


    .. autoattribute:: cuda.cuda.CUjit_option.CU_JIT_LTO


        Enable link-time optimization (-dlto) for device code (Disabled by default).

        This option is not supported on 32-bit platforms.

        Option type: int

        Applies to: compiler and linker


    .. autoattribute:: cuda.cuda.CUjit_option.CU_JIT_FTZ


        Control single-precision denormals (-ftz) support (0: false, default). 1 : flushes denormal values to zero 0 : preserves denormal values Option type: int

        Applies to: link-time optimization specified with CU_JIT_LTO


    .. autoattribute:: cuda.cuda.CUjit_option.CU_JIT_PREC_DIV


        Control single-precision floating-point division and reciprocals (-prec-div) support (1: true, default). 1 : Enables the IEEE round-to-nearest mode 0 : Enables the fast approximation mode Option type: int

        Applies to: link-time optimization specified with CU_JIT_LTO


    .. autoattribute:: cuda.cuda.CUjit_option.CU_JIT_PREC_SQRT


        Control single-precision floating-point square root (-prec-sqrt) support (1: true, default). 1 : Enables the IEEE round-to-nearest mode 0 : Enables the fast approximation mode Option type: int

        Applies to: link-time optimization specified with CU_JIT_LTO


    .. autoattribute:: cuda.cuda.CUjit_option.CU_JIT_FMA


        Enable/Disable the contraction of floating-point multiplies and adds/subtracts into floating-point multiply-add (-fma) operations (1: Enable, default; 0: Disable). Option type: int

        Applies to: link-time optimization specified with CU_JIT_LTO


    .. autoattribute:: cuda.cuda.CUjit_option.CU_JIT_REFERENCED_KERNEL_NAMES


        Array of kernel names that should be preserved at link time while others can be removed.

        Must contain :py:obj:`~.CU_JIT_REFERENCED_KERNEL_COUNT` entries.

        Note that kernel names can be mangled by the compiler in which case the mangled name needs to be specified.

        Wildcard "*" can be used to represent zero or more characters instead of specifying the full or mangled name.

        It is important to note that the wildcard "*" is also added implicitly. For example, specifying "foo" will match "foobaz", "barfoo", "barfoobaz" and thus preserve all kernels with those names. This can be avoided by providing a more specific name like "barfoobaz".

        Option type: const char **

        Applies to: dynamic linker only


    .. autoattribute:: cuda.cuda.CUjit_option.CU_JIT_REFERENCED_KERNEL_COUNT


        Number of entries in :py:obj:`~.CU_JIT_REFERENCED_KERNEL_NAMES` array.

        Option type: unsigned int

        Applies to: dynamic linker only


    .. autoattribute:: cuda.cuda.CUjit_option.CU_JIT_REFERENCED_VARIABLE_NAMES


        Array of variable names (device and/or constant) that should be preserved at link time while others can be removed.

        Must contain :py:obj:`~.CU_JIT_REFERENCED_VARIABLE_COUNT` entries.

        Note that variable names can be mangled by the compiler in which case the mangled name needs to be specified.

        Wildcard "*" can be used to represent zero or more characters instead of specifying the full or mangled name.

        It is important to note that the wildcard "*" is also added implicitly. For example, specifying "foo" will match "foobaz", "barfoo", "barfoobaz" and thus preserve all variables with those names. This can be avoided by providing a more specific name like "barfoobaz".

        Option type: const char **

        Applies to: link-time optimization specified with CU_JIT_LTO


    .. autoattribute:: cuda.cuda.CUjit_option.CU_JIT_REFERENCED_VARIABLE_COUNT


        Number of entries in :py:obj:`~.CU_JIT_REFERENCED_VARIABLE_NAMES` array.

        Option type: unsigned int

        Applies to: link-time optimization specified with CU_JIT_LTO


    .. autoattribute:: cuda.cuda.CUjit_option.CU_JIT_OPTIMIZE_UNUSED_DEVICE_VARIABLES


        This option serves as a hint to enable the JIT compiler/linker to remove constant (constant) and device (device) variables unreferenced in device code (Disabled by default).

        Note that host references to constant and device variables using APIs like :py:obj:`~.cuModuleGetGlobal()` with this option specified may result in undefined behavior unless the variables are explicitly specified using :py:obj:`~.CU_JIT_REFERENCED_VARIABLE_NAMES`.

        Option type: int

        Applies to: link-time optimization specified with CU_JIT_LTO


    .. autoattribute:: cuda.cuda.CUjit_option.CU_JIT_NUM_OPTIONS

.. autoclass:: cuda.cuda.CUjit_target

    .. autoattribute:: cuda.cuda.CUjit_target.CU_TARGET_COMPUTE_20


        Compute device class 2.0


    .. autoattribute:: cuda.cuda.CUjit_target.CU_TARGET_COMPUTE_21


        Compute device class 2.1


    .. autoattribute:: cuda.cuda.CUjit_target.CU_TARGET_COMPUTE_30


        Compute device class 3.0


    .. autoattribute:: cuda.cuda.CUjit_target.CU_TARGET_COMPUTE_32


        Compute device class 3.2


    .. autoattribute:: cuda.cuda.CUjit_target.CU_TARGET_COMPUTE_35


        Compute device class 3.5


    .. autoattribute:: cuda.cuda.CUjit_target.CU_TARGET_COMPUTE_37


        Compute device class 3.7


    .. autoattribute:: cuda.cuda.CUjit_target.CU_TARGET_COMPUTE_50


        Compute device class 5.0


    .. autoattribute:: cuda.cuda.CUjit_target.CU_TARGET_COMPUTE_52


        Compute device class 5.2


    .. autoattribute:: cuda.cuda.CUjit_target.CU_TARGET_COMPUTE_53


        Compute device class 5.3


    .. autoattribute:: cuda.cuda.CUjit_target.CU_TARGET_COMPUTE_60


        Compute device class 6.0.


    .. autoattribute:: cuda.cuda.CUjit_target.CU_TARGET_COMPUTE_61


        Compute device class 6.1.


    .. autoattribute:: cuda.cuda.CUjit_target.CU_TARGET_COMPUTE_62


        Compute device class 6.2.


    .. autoattribute:: cuda.cuda.CUjit_target.CU_TARGET_COMPUTE_70


        Compute device class 7.0.


    .. autoattribute:: cuda.cuda.CUjit_target.CU_TARGET_COMPUTE_72


        Compute device class 7.2.


    .. autoattribute:: cuda.cuda.CUjit_target.CU_TARGET_COMPUTE_75


        Compute device class 7.5.


    .. autoattribute:: cuda.cuda.CUjit_target.CU_TARGET_COMPUTE_80


        Compute device class 8.0.


    .. autoattribute:: cuda.cuda.CUjit_target.CU_TARGET_COMPUTE_86


        Compute device class 8.6.

.. autoclass:: cuda.cuda.CUjit_fallback

    .. autoattribute:: cuda.cuda.CUjit_fallback.CU_PREFER_PTX


        Prefer to compile ptx if exact binary match not found


    .. autoattribute:: cuda.cuda.CUjit_fallback.CU_PREFER_BINARY


        Prefer to fall back to compatible binary code if exact match not found

.. autoclass:: cuda.cuda.CUjit_cacheMode

    .. autoattribute:: cuda.cuda.CUjit_cacheMode.CU_JIT_CACHE_OPTION_NONE


        Compile with no -dlcm flag specified


    .. autoattribute:: cuda.cuda.CUjit_cacheMode.CU_JIT_CACHE_OPTION_CG


        Compile with L1 cache disabled


    .. autoattribute:: cuda.cuda.CUjit_cacheMode.CU_JIT_CACHE_OPTION_CA


        Compile with L1 cache enabled

.. autoclass:: cuda.cuda.CUjitInputType

    .. autoattribute:: cuda.cuda.CUjitInputType.CU_JIT_INPUT_CUBIN


        Compiled device-class-specific device code

        Applicable options: none


    .. autoattribute:: cuda.cuda.CUjitInputType.CU_JIT_INPUT_PTX


        PTX source code

        Applicable options: PTX compiler options


    .. autoattribute:: cuda.cuda.CUjitInputType.CU_JIT_INPUT_FATBINARY


        Bundle of multiple cubins and/or PTX of some device code

        Applicable options: PTX compiler options, :py:obj:`~.CU_JIT_FALLBACK_STRATEGY`


    .. autoattribute:: cuda.cuda.CUjitInputType.CU_JIT_INPUT_OBJECT


        Host object with embedded device code

        Applicable options: PTX compiler options, :py:obj:`~.CU_JIT_FALLBACK_STRATEGY`


    .. autoattribute:: cuda.cuda.CUjitInputType.CU_JIT_INPUT_LIBRARY


        Archive of host objects with embedded device code

        Applicable options: PTX compiler options, :py:obj:`~.CU_JIT_FALLBACK_STRATEGY`


    .. autoattribute:: cuda.cuda.CUjitInputType.CU_JIT_INPUT_NVVM


        High-level intermediate code for link-time optimization

        Applicable options: NVVM compiler options, PTX compiler options


    .. autoattribute:: cuda.cuda.CUjitInputType.CU_JIT_NUM_INPUT_TYPES

.. autoclass:: cuda.cuda.CUgraphicsRegisterFlags

    .. autoattribute:: cuda.cuda.CUgraphicsRegisterFlags.CU_GRAPHICS_REGISTER_FLAGS_NONE


    .. autoattribute:: cuda.cuda.CUgraphicsRegisterFlags.CU_GRAPHICS_REGISTER_FLAGS_READ_ONLY


    .. autoattribute:: cuda.cuda.CUgraphicsRegisterFlags.CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD


    .. autoattribute:: cuda.cuda.CUgraphicsRegisterFlags.CU_GRAPHICS_REGISTER_FLAGS_SURFACE_LDST


    .. autoattribute:: cuda.cuda.CUgraphicsRegisterFlags.CU_GRAPHICS_REGISTER_FLAGS_TEXTURE_GATHER

.. autoclass:: cuda.cuda.CUgraphicsMapResourceFlags

    .. autoattribute:: cuda.cuda.CUgraphicsMapResourceFlags.CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE


    .. autoattribute:: cuda.cuda.CUgraphicsMapResourceFlags.CU_GRAPHICS_MAP_RESOURCE_FLAGS_READ_ONLY


    .. autoattribute:: cuda.cuda.CUgraphicsMapResourceFlags.CU_GRAPHICS_MAP_RESOURCE_FLAGS_WRITE_DISCARD

.. autoclass:: cuda.cuda.CUarray_cubemap_face

    .. autoattribute:: cuda.cuda.CUarray_cubemap_face.CU_CUBEMAP_FACE_POSITIVE_X


        Positive X face of cubemap


    .. autoattribute:: cuda.cuda.CUarray_cubemap_face.CU_CUBEMAP_FACE_NEGATIVE_X


        Negative X face of cubemap


    .. autoattribute:: cuda.cuda.CUarray_cubemap_face.CU_CUBEMAP_FACE_POSITIVE_Y


        Positive Y face of cubemap


    .. autoattribute:: cuda.cuda.CUarray_cubemap_face.CU_CUBEMAP_FACE_NEGATIVE_Y


        Negative Y face of cubemap


    .. autoattribute:: cuda.cuda.CUarray_cubemap_face.CU_CUBEMAP_FACE_POSITIVE_Z


        Positive Z face of cubemap


    .. autoattribute:: cuda.cuda.CUarray_cubemap_face.CU_CUBEMAP_FACE_NEGATIVE_Z


        Negative Z face of cubemap

.. autoclass:: cuda.cuda.CUlimit

    .. autoattribute:: cuda.cuda.CUlimit.CU_LIMIT_STACK_SIZE


        GPU thread stack size


    .. autoattribute:: cuda.cuda.CUlimit.CU_LIMIT_PRINTF_FIFO_SIZE


        GPU printf FIFO size


    .. autoattribute:: cuda.cuda.CUlimit.CU_LIMIT_MALLOC_HEAP_SIZE


        GPU malloc heap size


    .. autoattribute:: cuda.cuda.CUlimit.CU_LIMIT_DEV_RUNTIME_SYNC_DEPTH


        GPU device runtime launch synchronize depth


    .. autoattribute:: cuda.cuda.CUlimit.CU_LIMIT_DEV_RUNTIME_PENDING_LAUNCH_COUNT


        GPU device runtime pending launch count


    .. autoattribute:: cuda.cuda.CUlimit.CU_LIMIT_MAX_L2_FETCH_GRANULARITY


        A value between 0 and 128 that indicates the maximum fetch granularity of L2 (in Bytes). This is a hint


    .. autoattribute:: cuda.cuda.CUlimit.CU_LIMIT_PERSISTING_L2_CACHE_SIZE


        A size in bytes for L2 persisting lines cache size


    .. autoattribute:: cuda.cuda.CUlimit.CU_LIMIT_MAX

.. autoclass:: cuda.cuda.CUresourcetype

    .. autoattribute:: cuda.cuda.CUresourcetype.CU_RESOURCE_TYPE_ARRAY


        Array resoure


    .. autoattribute:: cuda.cuda.CUresourcetype.CU_RESOURCE_TYPE_MIPMAPPED_ARRAY


        Mipmapped array resource


    .. autoattribute:: cuda.cuda.CUresourcetype.CU_RESOURCE_TYPE_LINEAR


        Linear resource


    .. autoattribute:: cuda.cuda.CUresourcetype.CU_RESOURCE_TYPE_PITCH2D


        Pitch 2D resource

.. autoclass:: cuda.cuda.CUaccessProperty

    .. autoattribute:: cuda.cuda.CUaccessProperty.CU_ACCESS_PROPERTY_NORMAL


        Normal cache persistence.


    .. autoattribute:: cuda.cuda.CUaccessProperty.CU_ACCESS_PROPERTY_STREAMING


        Streaming access is less likely to persit from cache.


    .. autoattribute:: cuda.cuda.CUaccessProperty.CU_ACCESS_PROPERTY_PERSISTING


        Persisting access is more likely to persist in cache.

.. autoclass:: cuda.cuda.CUgraphNodeType

    .. autoattribute:: cuda.cuda.CUgraphNodeType.CU_GRAPH_NODE_TYPE_KERNEL


        GPU kernel node


    .. autoattribute:: cuda.cuda.CUgraphNodeType.CU_GRAPH_NODE_TYPE_MEMCPY


        Memcpy node


    .. autoattribute:: cuda.cuda.CUgraphNodeType.CU_GRAPH_NODE_TYPE_MEMSET


        Memset node


    .. autoattribute:: cuda.cuda.CUgraphNodeType.CU_GRAPH_NODE_TYPE_HOST


        Host (executable) node


    .. autoattribute:: cuda.cuda.CUgraphNodeType.CU_GRAPH_NODE_TYPE_GRAPH


        Node which executes an embedded graph


    .. autoattribute:: cuda.cuda.CUgraphNodeType.CU_GRAPH_NODE_TYPE_EMPTY


        Empty (no-op) node


    .. autoattribute:: cuda.cuda.CUgraphNodeType.CU_GRAPH_NODE_TYPE_WAIT_EVENT


        External event wait node


    .. autoattribute:: cuda.cuda.CUgraphNodeType.CU_GRAPH_NODE_TYPE_EVENT_RECORD


        External event record node


    .. autoattribute:: cuda.cuda.CUgraphNodeType.CU_GRAPH_NODE_TYPE_EXT_SEMAS_SIGNAL


        External semaphore signal node


    .. autoattribute:: cuda.cuda.CUgraphNodeType.CU_GRAPH_NODE_TYPE_EXT_SEMAS_WAIT


        External semaphore wait node


    .. autoattribute:: cuda.cuda.CUgraphNodeType.CU_GRAPH_NODE_TYPE_MEM_ALLOC


        Memory Allocation Node


    .. autoattribute:: cuda.cuda.CUgraphNodeType.CU_GRAPH_NODE_TYPE_MEM_FREE


        Memory Free Node


    .. autoattribute:: cuda.cuda.CUgraphNodeType.CU_GRAPH_NODE_TYPE_BATCH_MEM_OP


        Batch MemOp Node

.. autoclass:: cuda.cuda.CUsynchronizationPolicy

    .. autoattribute:: cuda.cuda.CUsynchronizationPolicy.CU_SYNC_POLICY_AUTO


    .. autoattribute:: cuda.cuda.CUsynchronizationPolicy.CU_SYNC_POLICY_SPIN


    .. autoattribute:: cuda.cuda.CUsynchronizationPolicy.CU_SYNC_POLICY_YIELD


    .. autoattribute:: cuda.cuda.CUsynchronizationPolicy.CU_SYNC_POLICY_BLOCKING_SYNC

.. autoclass:: cuda.cuda.CUkernelNodeAttrID

    .. autoattribute:: cuda.cuda.CUkernelNodeAttrID.CU_KERNEL_NODE_ATTRIBUTE_ACCESS_POLICY_WINDOW


        Identifier for :py:obj:`~.CUkernelNodeAttrValue.accessPolicyWindow`.


    .. autoattribute:: cuda.cuda.CUkernelNodeAttrID.CU_KERNEL_NODE_ATTRIBUTE_COOPERATIVE


        Allows a kernel node to be cooperative (see :py:obj:`~.cuLaunchCooperativeKernel`).


    .. autoattribute:: cuda.cuda.CUkernelNodeAttrID.CU_KERNEL_NODE_ATTRIBUTE_PRIORITY


        Sets the priority of the kernel.

.. autoclass:: cuda.cuda.CUstreamCaptureStatus

    .. autoattribute:: cuda.cuda.CUstreamCaptureStatus.CU_STREAM_CAPTURE_STATUS_NONE


        Stream is not capturing


    .. autoattribute:: cuda.cuda.CUstreamCaptureStatus.CU_STREAM_CAPTURE_STATUS_ACTIVE


        Stream is actively capturing


    .. autoattribute:: cuda.cuda.CUstreamCaptureStatus.CU_STREAM_CAPTURE_STATUS_INVALIDATED


        Stream is part of a capture sequence that has been invalidated, but not terminated

.. autoclass:: cuda.cuda.CUstreamCaptureMode

    .. autoattribute:: cuda.cuda.CUstreamCaptureMode.CU_STREAM_CAPTURE_MODE_GLOBAL


    .. autoattribute:: cuda.cuda.CUstreamCaptureMode.CU_STREAM_CAPTURE_MODE_THREAD_LOCAL


    .. autoattribute:: cuda.cuda.CUstreamCaptureMode.CU_STREAM_CAPTURE_MODE_RELAXED

.. autoclass:: cuda.cuda.CUstreamAttrID

    .. autoattribute:: cuda.cuda.CUstreamAttrID.CU_STREAM_ATTRIBUTE_ACCESS_POLICY_WINDOW


        Identifier for :py:obj:`~.CUstreamAttrValue.accessPolicyWindow`.


    .. autoattribute:: cuda.cuda.CUstreamAttrID.CU_STREAM_ATTRIBUTE_SYNCHRONIZATION_POLICY


        :py:obj:`~.CUsynchronizationPolicy` for work queued up in this stream

.. autoclass:: cuda.cuda.CUdriverProcAddress_flags

    .. autoattribute:: cuda.cuda.CUdriverProcAddress_flags.CU_GET_PROC_ADDRESS_DEFAULT


        Default search mode for driver symbols.


    .. autoattribute:: cuda.cuda.CUdriverProcAddress_flags.CU_GET_PROC_ADDRESS_LEGACY_STREAM


        Search for legacy versions of driver symbols.


    .. autoattribute:: cuda.cuda.CUdriverProcAddress_flags.CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM


        Search for per-thread versions of driver symbols.

.. autoclass:: cuda.cuda.CUexecAffinityType

    .. autoattribute:: cuda.cuda.CUexecAffinityType.CU_EXEC_AFFINITY_TYPE_SM_COUNT


        Create a context with limited SMs.


    .. autoattribute:: cuda.cuda.CUexecAffinityType.CU_EXEC_AFFINITY_TYPE_MAX

.. autoclass:: cuda.cuda.CUresult

    .. autoattribute:: cuda.cuda.CUresult.CUDA_SUCCESS


        The API call returned with no errors. In the case of query calls, this also means that the operation being queried is complete (see :py:obj:`~.cuEventQuery()` and :py:obj:`~.cuStreamQuery()`).


    .. autoattribute:: cuda.cuda.CUresult.CUDA_ERROR_INVALID_VALUE


        This indicates that one or more of the parameters passed to the API call is not within an acceptable range of values.


    .. autoattribute:: cuda.cuda.CUresult.CUDA_ERROR_OUT_OF_MEMORY


        The API call failed because it was unable to allocate enough memory to perform the requested operation.


    .. autoattribute:: cuda.cuda.CUresult.CUDA_ERROR_NOT_INITIALIZED


        This indicates that the CUDA driver has not been initialized with :py:obj:`~.cuInit()` or that initialization has failed.


    .. autoattribute:: cuda.cuda.CUresult.CUDA_ERROR_DEINITIALIZED


        This indicates that the CUDA driver is in the process of shutting down.


    .. autoattribute:: cuda.cuda.CUresult.CUDA_ERROR_PROFILER_DISABLED


        This indicates profiler is not initialized for this run. This can happen when the application is running with external profiling tools like visual profiler.


    .. autoattribute:: cuda.cuda.CUresult.CUDA_ERROR_PROFILER_NOT_INITIALIZED


        [Deprecated]


    .. autoattribute:: cuda.cuda.CUresult.CUDA_ERROR_PROFILER_ALREADY_STARTED


        [Deprecated]


    .. autoattribute:: cuda.cuda.CUresult.CUDA_ERROR_PROFILER_ALREADY_STOPPED


        [Deprecated]


    .. autoattribute:: cuda.cuda.CUresult.CUDA_ERROR_STUB_LIBRARY


        This indicates that the CUDA driver that the application has loaded is a stub library. Applications that run with the stub rather than a real driver loaded will result in CUDA API returning this error.


    .. autoattribute:: cuda.cuda.CUresult.CUDA_ERROR_DEVICE_UNAVAILABLE


        This indicates that requested CUDA device is unavailable at the current time. Devices are often unavailable due to use of :py:obj:`~.CU_COMPUTEMODE_EXCLUSIVE_PROCESS` or :py:obj:`~.CU_COMPUTEMODE_PROHIBITED`.


    .. autoattribute:: cuda.cuda.CUresult.CUDA_ERROR_NO_DEVICE


        This indicates that no CUDA-capable devices were detected by the installed CUDA driver.


    .. autoattribute:: cuda.cuda.CUresult.CUDA_ERROR_INVALID_DEVICE


        This indicates that the device ordinal supplied by the user does not correspond to a valid CUDA device or that the action requested is invalid for the specified device.


    .. autoattribute:: cuda.cuda.CUresult.CUDA_ERROR_DEVICE_NOT_LICENSED


        This error indicates that the Grid license is not applied.


    .. autoattribute:: cuda.cuda.CUresult.CUDA_ERROR_INVALID_IMAGE


        This indicates that the device kernel image is invalid. This can also indicate an invalid CUDA module.


    .. autoattribute:: cuda.cuda.CUresult.CUDA_ERROR_INVALID_CONTEXT


        This most frequently indicates that there is no context bound to the current thread. This can also be returned if the context passed to an API call is not a valid handle (such as a context that has had :py:obj:`~.cuCtxDestroy()` invoked on it). This can also be returned if a user mixes different API versions (i.e. 3010 context with 3020 API calls). See :py:obj:`~.cuCtxGetApiVersion()` for more details.


    .. autoattribute:: cuda.cuda.CUresult.CUDA_ERROR_CONTEXT_ALREADY_CURRENT


        This indicated that the context being supplied as a parameter to the API call was already the active context. [Deprecated]


    .. autoattribute:: cuda.cuda.CUresult.CUDA_ERROR_MAP_FAILED


        This indicates that a map or register operation has failed.


    .. autoattribute:: cuda.cuda.CUresult.CUDA_ERROR_UNMAP_FAILED


        This indicates that an unmap or unregister operation has failed.


    .. autoattribute:: cuda.cuda.CUresult.CUDA_ERROR_ARRAY_IS_MAPPED


        This indicates that the specified array is currently mapped and thus cannot be destroyed.


    .. autoattribute:: cuda.cuda.CUresult.CUDA_ERROR_ALREADY_MAPPED


        This indicates that the resource is already mapped.


    .. autoattribute:: cuda.cuda.CUresult.CUDA_ERROR_NO_BINARY_FOR_GPU


        This indicates that there is no kernel image available that is suitable for the device. This can occur when a user specifies code generation options for a particular CUDA source file that do not include the corresponding device configuration.


    .. autoattribute:: cuda.cuda.CUresult.CUDA_ERROR_ALREADY_ACQUIRED


        This indicates that a resource has already been acquired.


    .. autoattribute:: cuda.cuda.CUresult.CUDA_ERROR_NOT_MAPPED


        This indicates that a resource is not mapped.


    .. autoattribute:: cuda.cuda.CUresult.CUDA_ERROR_NOT_MAPPED_AS_ARRAY


        This indicates that a mapped resource is not available for access as an array.


    .. autoattribute:: cuda.cuda.CUresult.CUDA_ERROR_NOT_MAPPED_AS_POINTER


        This indicates that a mapped resource is not available for access as a pointer.


    .. autoattribute:: cuda.cuda.CUresult.CUDA_ERROR_ECC_UNCORRECTABLE


        This indicates that an uncorrectable ECC error was detected during execution.


    .. autoattribute:: cuda.cuda.CUresult.CUDA_ERROR_UNSUPPORTED_LIMIT


        This indicates that the :py:obj:`~.CUlimit` passed to the API call is not supported by the active device.


    .. autoattribute:: cuda.cuda.CUresult.CUDA_ERROR_CONTEXT_ALREADY_IN_USE


        This indicates that the :py:obj:`~.CUcontext` passed to the API call can only be bound to a single CPU thread at a time but is already bound to a CPU thread.


    .. autoattribute:: cuda.cuda.CUresult.CUDA_ERROR_PEER_ACCESS_UNSUPPORTED


        This indicates that peer access is not supported across the given devices.


    .. autoattribute:: cuda.cuda.CUresult.CUDA_ERROR_INVALID_PTX


        This indicates that a PTX JIT compilation failed.


    .. autoattribute:: cuda.cuda.CUresult.CUDA_ERROR_INVALID_GRAPHICS_CONTEXT


        This indicates an error with OpenGL or DirectX context.


    .. autoattribute:: cuda.cuda.CUresult.CUDA_ERROR_NVLINK_UNCORRECTABLE


        This indicates that an uncorrectable NVLink error was detected during the execution.


    .. autoattribute:: cuda.cuda.CUresult.CUDA_ERROR_JIT_COMPILER_NOT_FOUND


        This indicates that the PTX JIT compiler library was not found.


    .. autoattribute:: cuda.cuda.CUresult.CUDA_ERROR_UNSUPPORTED_PTX_VERSION


        This indicates that the provided PTX was compiled with an unsupported toolchain.


    .. autoattribute:: cuda.cuda.CUresult.CUDA_ERROR_JIT_COMPILATION_DISABLED


        This indicates that the PTX JIT compilation was disabled.


    .. autoattribute:: cuda.cuda.CUresult.CUDA_ERROR_UNSUPPORTED_EXEC_AFFINITY


        This indicates that the :py:obj:`~.CUexecAffinityType` passed to the API call is not supported by the active device.


    .. autoattribute:: cuda.cuda.CUresult.CUDA_ERROR_INVALID_SOURCE


        This indicates that the device kernel source is invalid. This includes compilation/linker errors encountered in device code or user error.


    .. autoattribute:: cuda.cuda.CUresult.CUDA_ERROR_FILE_NOT_FOUND


        This indicates that the file specified was not found.


    .. autoattribute:: cuda.cuda.CUresult.CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND


        This indicates that a link to a shared object failed to resolve.


    .. autoattribute:: cuda.cuda.CUresult.CUDA_ERROR_SHARED_OBJECT_INIT_FAILED


        This indicates that initialization of a shared object failed.


    .. autoattribute:: cuda.cuda.CUresult.CUDA_ERROR_OPERATING_SYSTEM


        This indicates that an OS call failed.


    .. autoattribute:: cuda.cuda.CUresult.CUDA_ERROR_INVALID_HANDLE


        This indicates that a resource handle passed to the API call was not valid. Resource handles are opaque types like :py:obj:`~.CUstream` and :py:obj:`~.CUevent`.


    .. autoattribute:: cuda.cuda.CUresult.CUDA_ERROR_ILLEGAL_STATE


        This indicates that a resource required by the API call is not in a valid state to perform the requested operation.


    .. autoattribute:: cuda.cuda.CUresult.CUDA_ERROR_NOT_FOUND


        This indicates that a named symbol was not found. Examples of symbols are global/constant variable names, driver function names, texture names, and surface names.


    .. autoattribute:: cuda.cuda.CUresult.CUDA_ERROR_NOT_READY


        This indicates that asynchronous operations issued previously have not completed yet. This result is not actually an error, but must be indicated differently than :py:obj:`~.CUDA_SUCCESS` (which indicates completion). Calls that may return this value include :py:obj:`~.cuEventQuery()` and :py:obj:`~.cuStreamQuery()`.


    .. autoattribute:: cuda.cuda.CUresult.CUDA_ERROR_ILLEGAL_ADDRESS


        While executing a kernel, the device encountered a load or store instruction on an invalid memory address. This leaves the process in an inconsistent state and any further CUDA work will return the same error. To continue using CUDA, the process must be terminated and relaunched.


    .. autoattribute:: cuda.cuda.CUresult.CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES


        This indicates that a launch did not occur because it did not have appropriate resources. This error usually indicates that the user has attempted to pass too many arguments to the device kernel, or the kernel launch specifies too many threads for the kernel's register count. Passing arguments of the wrong size (i.e. a 64-bit pointer when a 32-bit int is expected) is equivalent to passing too many arguments and can also result in this error.


    .. autoattribute:: cuda.cuda.CUresult.CUDA_ERROR_LAUNCH_TIMEOUT


        This indicates that the device kernel took too long to execute. This can only occur if timeouts are enabled - see the device attribute :py:obj:`~.CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT` for more information. This leaves the process in an inconsistent state and any further CUDA work will return the same error. To continue using CUDA, the process must be terminated and relaunched.


    .. autoattribute:: cuda.cuda.CUresult.CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING


        This error indicates a kernel launch that uses an incompatible texturing mode.


    .. autoattribute:: cuda.cuda.CUresult.CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED


        This error indicates that a call to :py:obj:`~.cuCtxEnablePeerAccess()` is trying to re-enable peer access to a context which has already had peer access to it enabled.


    .. autoattribute:: cuda.cuda.CUresult.CUDA_ERROR_PEER_ACCESS_NOT_ENABLED


        This error indicates that :py:obj:`~.cuCtxDisablePeerAccess()` is trying to disable peer access which has not been enabled yet via :py:obj:`~.cuCtxEnablePeerAccess()`.


    .. autoattribute:: cuda.cuda.CUresult.CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE


        This error indicates that the primary context for the specified device has already been initialized.


    .. autoattribute:: cuda.cuda.CUresult.CUDA_ERROR_CONTEXT_IS_DESTROYED


        This error indicates that the context current to the calling thread has been destroyed using :py:obj:`~.cuCtxDestroy`, or is a primary context which has not yet been initialized.


    .. autoattribute:: cuda.cuda.CUresult.CUDA_ERROR_ASSERT


        A device-side assert triggered during kernel execution. The context cannot be used anymore, and must be destroyed. All existing device memory allocations from this context are invalid and must be reconstructed if the program is to continue using CUDA.


    .. autoattribute:: cuda.cuda.CUresult.CUDA_ERROR_TOO_MANY_PEERS


        This error indicates that the hardware resources required to enable peer access have been exhausted for one or more of the devices passed to :py:obj:`~.cuCtxEnablePeerAccess()`.


    .. autoattribute:: cuda.cuda.CUresult.CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED


        This error indicates that the memory range passed to :py:obj:`~.cuMemHostRegister()` has already been registered.


    .. autoattribute:: cuda.cuda.CUresult.CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED


        This error indicates that the pointer passed to :py:obj:`~.cuMemHostUnregister()` does not correspond to any currently registered memory region.


    .. autoattribute:: cuda.cuda.CUresult.CUDA_ERROR_HARDWARE_STACK_ERROR


        While executing a kernel, the device encountered a stack error. This can be due to stack corruption or exceeding the stack size limit. This leaves the process in an inconsistent state and any further CUDA work will return the same error. To continue using CUDA, the process must be terminated and relaunched.


    .. autoattribute:: cuda.cuda.CUresult.CUDA_ERROR_ILLEGAL_INSTRUCTION


        While executing a kernel, the device encountered an illegal instruction. This leaves the process in an inconsistent state and any further CUDA work will return the same error. To continue using CUDA, the process must be terminated and relaunched.


    .. autoattribute:: cuda.cuda.CUresult.CUDA_ERROR_MISALIGNED_ADDRESS


        While executing a kernel, the device encountered a load or store instruction on a memory address which is not aligned. This leaves the process in an inconsistent state and any further CUDA work will return the same error. To continue using CUDA, the process must be terminated and relaunched.


    .. autoattribute:: cuda.cuda.CUresult.CUDA_ERROR_INVALID_ADDRESS_SPACE


        While executing a kernel, the device encountered an instruction which can only operate on memory locations in certain address spaces (global, shared, or local), but was supplied a memory address not belonging to an allowed address space. This leaves the process in an inconsistent state and any further CUDA work will return the same error. To continue using CUDA, the process must be terminated and relaunched.


    .. autoattribute:: cuda.cuda.CUresult.CUDA_ERROR_INVALID_PC


        While executing a kernel, the device program counter wrapped its address space. This leaves the process in an inconsistent state and any further CUDA work will return the same error. To continue using CUDA, the process must be terminated and relaunched.


    .. autoattribute:: cuda.cuda.CUresult.CUDA_ERROR_LAUNCH_FAILED


        An exception occurred on the device while executing a kernel. Common causes include dereferencing an invalid device pointer and accessing out of bounds shared memory. Less common cases can be system specific - more information about these cases can be found in the system specific user guide. This leaves the process in an inconsistent state and any further CUDA work will return the same error. To continue using CUDA, the process must be terminated and relaunched.


    .. autoattribute:: cuda.cuda.CUresult.CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE


        This error indicates that the number of blocks launched per grid for a kernel that was launched via either :py:obj:`~.cuLaunchCooperativeKernel` or :py:obj:`~.cuLaunchCooperativeKernelMultiDevice` exceeds the maximum number of blocks as allowed by :py:obj:`~.cuOccupancyMaxActiveBlocksPerMultiprocessor` or :py:obj:`~.cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags` times the number of multiprocessors as specified by the device attribute :py:obj:`~.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT`.


    .. autoattribute:: cuda.cuda.CUresult.CUDA_ERROR_NOT_PERMITTED


        This error indicates that the attempted operation is not permitted.


    .. autoattribute:: cuda.cuda.CUresult.CUDA_ERROR_NOT_SUPPORTED


        This error indicates that the attempted operation is not supported on the current system or device.


    .. autoattribute:: cuda.cuda.CUresult.CUDA_ERROR_SYSTEM_NOT_READY


        This error indicates that the system is not yet ready to start any CUDA work. To continue using CUDA, verify the system configuration is in a valid state and all required driver daemons are actively running. More information about this error can be found in the system specific user guide.


    .. autoattribute:: cuda.cuda.CUresult.CUDA_ERROR_SYSTEM_DRIVER_MISMATCH


        This error indicates that there is a mismatch between the versions of the display driver and the CUDA driver. Refer to the compatibility documentation for supported versions.


    .. autoattribute:: cuda.cuda.CUresult.CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE


        This error indicates that the system was upgraded to run with forward compatibility but the visible hardware detected by CUDA does not support this configuration. Refer to the compatibility documentation for the supported hardware matrix or ensure that only supported hardware is visible during initialization via the CUDA_VISIBLE_DEVICES environment variable.


    .. autoattribute:: cuda.cuda.CUresult.CUDA_ERROR_MPS_CONNECTION_FAILED


        This error indicates that the MPS client failed to connect to the MPS control daemon or the MPS server.


    .. autoattribute:: cuda.cuda.CUresult.CUDA_ERROR_MPS_RPC_FAILURE


        This error indicates that the remote procedural call between the MPS server and the MPS client failed.


    .. autoattribute:: cuda.cuda.CUresult.CUDA_ERROR_MPS_SERVER_NOT_READY


        This error indicates that the MPS server is not ready to accept new MPS client requests. This error can be returned when the MPS server is in the process of recovering from a fatal failure.


    .. autoattribute:: cuda.cuda.CUresult.CUDA_ERROR_MPS_MAX_CLIENTS_REACHED


        This error indicates that the hardware resources required to create MPS client have been exhausted.


    .. autoattribute:: cuda.cuda.CUresult.CUDA_ERROR_MPS_MAX_CONNECTIONS_REACHED


        This error indicates the the hardware resources required to support device connections have been exhausted.


    .. autoattribute:: cuda.cuda.CUresult.CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED


        This error indicates that the operation is not permitted when the stream is capturing.


    .. autoattribute:: cuda.cuda.CUresult.CUDA_ERROR_STREAM_CAPTURE_INVALIDATED


        This error indicates that the current capture sequence on the stream has been invalidated due to a previous error.


    .. autoattribute:: cuda.cuda.CUresult.CUDA_ERROR_STREAM_CAPTURE_MERGE


        This error indicates that the operation would have resulted in a merge of two independent capture sequences.


    .. autoattribute:: cuda.cuda.CUresult.CUDA_ERROR_STREAM_CAPTURE_UNMATCHED


        This error indicates that the capture was not initiated in this stream.


    .. autoattribute:: cuda.cuda.CUresult.CUDA_ERROR_STREAM_CAPTURE_UNJOINED


        This error indicates that the capture sequence contains a fork that was not joined to the primary stream.


    .. autoattribute:: cuda.cuda.CUresult.CUDA_ERROR_STREAM_CAPTURE_ISOLATION


        This error indicates that a dependency would have been created which crosses the capture sequence boundary. Only implicit in-stream ordering dependencies are allowed to cross the boundary.


    .. autoattribute:: cuda.cuda.CUresult.CUDA_ERROR_STREAM_CAPTURE_IMPLICIT


        This error indicates a disallowed implicit dependency on a current capture sequence from cudaStreamLegacy.


    .. autoattribute:: cuda.cuda.CUresult.CUDA_ERROR_CAPTURED_EVENT


        This error indicates that the operation is not permitted on an event which was last recorded in a capturing stream.


    .. autoattribute:: cuda.cuda.CUresult.CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD


        A stream capture sequence not initiated with the :py:obj:`~.CU_STREAM_CAPTURE_MODE_RELAXED` argument to :py:obj:`~.cuStreamBeginCapture` was passed to :py:obj:`~.cuStreamEndCapture` in a different thread.


    .. autoattribute:: cuda.cuda.CUresult.CUDA_ERROR_TIMEOUT


        This error indicates that the timeout specified for the wait operation has lapsed.


    .. autoattribute:: cuda.cuda.CUresult.CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE


        This error indicates that the graph update was not performed because it included changes which violated constraints specific to instantiated graph update.


    .. autoattribute:: cuda.cuda.CUresult.CUDA_ERROR_EXTERNAL_DEVICE


        This indicates that an async error has occurred in a device outside of CUDA. If CUDA was waiting for an external device's signal before consuming shared data, the external device signaled an error indicating that the data is not valid for consumption. This leaves the process in an inconsistent state and any further CUDA work will return the same error. To continue using CUDA, the process must be terminated and relaunched.


    .. autoattribute:: cuda.cuda.CUresult.CUDA_ERROR_UNKNOWN


        This indicates that an unknown internal error has occurred.

.. autoclass:: cuda.cuda.CUdevice_P2PAttribute

    .. autoattribute:: cuda.cuda.CUdevice_P2PAttribute.CU_DEVICE_P2P_ATTRIBUTE_PERFORMANCE_RANK


        A relative value indicating the performance of the link between two devices


    .. autoattribute:: cuda.cuda.CUdevice_P2PAttribute.CU_DEVICE_P2P_ATTRIBUTE_ACCESS_SUPPORTED


        P2P Access is enable


    .. autoattribute:: cuda.cuda.CUdevice_P2PAttribute.CU_DEVICE_P2P_ATTRIBUTE_NATIVE_ATOMIC_SUPPORTED


        Atomic operation over the link supported


    .. autoattribute:: cuda.cuda.CUdevice_P2PAttribute.CU_DEVICE_P2P_ATTRIBUTE_ACCESS_ACCESS_SUPPORTED


        [Deprecated]


    .. autoattribute:: cuda.cuda.CUdevice_P2PAttribute.CU_DEVICE_P2P_ATTRIBUTE_CUDA_ARRAY_ACCESS_SUPPORTED


        Accessing CUDA arrays over the link supported

.. autoclass:: cuda.cuda.CUresourceViewFormat

    .. autoattribute:: cuda.cuda.CUresourceViewFormat.CU_RES_VIEW_FORMAT_NONE


        No resource view format (use underlying resource format)


    .. autoattribute:: cuda.cuda.CUresourceViewFormat.CU_RES_VIEW_FORMAT_UINT_1X8


        1 channel unsigned 8-bit integers


    .. autoattribute:: cuda.cuda.CUresourceViewFormat.CU_RES_VIEW_FORMAT_UINT_2X8


        2 channel unsigned 8-bit integers


    .. autoattribute:: cuda.cuda.CUresourceViewFormat.CU_RES_VIEW_FORMAT_UINT_4X8


        4 channel unsigned 8-bit integers


    .. autoattribute:: cuda.cuda.CUresourceViewFormat.CU_RES_VIEW_FORMAT_SINT_1X8


        1 channel signed 8-bit integers


    .. autoattribute:: cuda.cuda.CUresourceViewFormat.CU_RES_VIEW_FORMAT_SINT_2X8


        2 channel signed 8-bit integers


    .. autoattribute:: cuda.cuda.CUresourceViewFormat.CU_RES_VIEW_FORMAT_SINT_4X8


        4 channel signed 8-bit integers


    .. autoattribute:: cuda.cuda.CUresourceViewFormat.CU_RES_VIEW_FORMAT_UINT_1X16


        1 channel unsigned 16-bit integers


    .. autoattribute:: cuda.cuda.CUresourceViewFormat.CU_RES_VIEW_FORMAT_UINT_2X16


        2 channel unsigned 16-bit integers


    .. autoattribute:: cuda.cuda.CUresourceViewFormat.CU_RES_VIEW_FORMAT_UINT_4X16


        4 channel unsigned 16-bit integers


    .. autoattribute:: cuda.cuda.CUresourceViewFormat.CU_RES_VIEW_FORMAT_SINT_1X16


        1 channel signed 16-bit integers


    .. autoattribute:: cuda.cuda.CUresourceViewFormat.CU_RES_VIEW_FORMAT_SINT_2X16


        2 channel signed 16-bit integers


    .. autoattribute:: cuda.cuda.CUresourceViewFormat.CU_RES_VIEW_FORMAT_SINT_4X16


        4 channel signed 16-bit integers


    .. autoattribute:: cuda.cuda.CUresourceViewFormat.CU_RES_VIEW_FORMAT_UINT_1X32


        1 channel unsigned 32-bit integers


    .. autoattribute:: cuda.cuda.CUresourceViewFormat.CU_RES_VIEW_FORMAT_UINT_2X32


        2 channel unsigned 32-bit integers


    .. autoattribute:: cuda.cuda.CUresourceViewFormat.CU_RES_VIEW_FORMAT_UINT_4X32


        4 channel unsigned 32-bit integers


    .. autoattribute:: cuda.cuda.CUresourceViewFormat.CU_RES_VIEW_FORMAT_SINT_1X32


        1 channel signed 32-bit integers


    .. autoattribute:: cuda.cuda.CUresourceViewFormat.CU_RES_VIEW_FORMAT_SINT_2X32


        2 channel signed 32-bit integers


    .. autoattribute:: cuda.cuda.CUresourceViewFormat.CU_RES_VIEW_FORMAT_SINT_4X32


        4 channel signed 32-bit integers


    .. autoattribute:: cuda.cuda.CUresourceViewFormat.CU_RES_VIEW_FORMAT_FLOAT_1X16


        1 channel 16-bit floating point


    .. autoattribute:: cuda.cuda.CUresourceViewFormat.CU_RES_VIEW_FORMAT_FLOAT_2X16


        2 channel 16-bit floating point


    .. autoattribute:: cuda.cuda.CUresourceViewFormat.CU_RES_VIEW_FORMAT_FLOAT_4X16


        4 channel 16-bit floating point


    .. autoattribute:: cuda.cuda.CUresourceViewFormat.CU_RES_VIEW_FORMAT_FLOAT_1X32


        1 channel 32-bit floating point


    .. autoattribute:: cuda.cuda.CUresourceViewFormat.CU_RES_VIEW_FORMAT_FLOAT_2X32


        2 channel 32-bit floating point


    .. autoattribute:: cuda.cuda.CUresourceViewFormat.CU_RES_VIEW_FORMAT_FLOAT_4X32


        4 channel 32-bit floating point


    .. autoattribute:: cuda.cuda.CUresourceViewFormat.CU_RES_VIEW_FORMAT_UNSIGNED_BC1


        Block compressed 1


    .. autoattribute:: cuda.cuda.CUresourceViewFormat.CU_RES_VIEW_FORMAT_UNSIGNED_BC2


        Block compressed 2


    .. autoattribute:: cuda.cuda.CUresourceViewFormat.CU_RES_VIEW_FORMAT_UNSIGNED_BC3


        Block compressed 3


    .. autoattribute:: cuda.cuda.CUresourceViewFormat.CU_RES_VIEW_FORMAT_UNSIGNED_BC4


        Block compressed 4 unsigned


    .. autoattribute:: cuda.cuda.CUresourceViewFormat.CU_RES_VIEW_FORMAT_SIGNED_BC4


        Block compressed 4 signed


    .. autoattribute:: cuda.cuda.CUresourceViewFormat.CU_RES_VIEW_FORMAT_UNSIGNED_BC5


        Block compressed 5 unsigned


    .. autoattribute:: cuda.cuda.CUresourceViewFormat.CU_RES_VIEW_FORMAT_SIGNED_BC5


        Block compressed 5 signed


    .. autoattribute:: cuda.cuda.CUresourceViewFormat.CU_RES_VIEW_FORMAT_UNSIGNED_BC6H


        Block compressed 6 unsigned half-float


    .. autoattribute:: cuda.cuda.CUresourceViewFormat.CU_RES_VIEW_FORMAT_SIGNED_BC6H


        Block compressed 6 signed half-float


    .. autoattribute:: cuda.cuda.CUresourceViewFormat.CU_RES_VIEW_FORMAT_UNSIGNED_BC7


        Block compressed 7

.. autoclass:: cuda.cuda.CUDA_POINTER_ATTRIBUTE_ACCESS_FLAGS

    .. autoattribute:: cuda.cuda.CUDA_POINTER_ATTRIBUTE_ACCESS_FLAGS.CU_POINTER_ATTRIBUTE_ACCESS_FLAG_NONE


        No access, meaning the device cannot access this memory at all, thus must be staged through accessible memory in order to complete certain operations


    .. autoattribute:: cuda.cuda.CUDA_POINTER_ATTRIBUTE_ACCESS_FLAGS.CU_POINTER_ATTRIBUTE_ACCESS_FLAG_READ


        Read-only access, meaning writes to this memory are considered invalid accesses and thus return error in that case.


    .. autoattribute:: cuda.cuda.CUDA_POINTER_ATTRIBUTE_ACCESS_FLAGS.CU_POINTER_ATTRIBUTE_ACCESS_FLAG_READWRITE


        Read-write access, the device has full read-write access to the memory

.. autoclass:: cuda.cuda.CUexternalMemoryHandleType

    .. autoattribute:: cuda.cuda.CUexternalMemoryHandleType.CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD


        Handle is an opaque file descriptor


    .. autoattribute:: cuda.cuda.CUexternalMemoryHandleType.CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32


        Handle is an opaque shared NT handle


    .. autoattribute:: cuda.cuda.CUexternalMemoryHandleType.CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT


        Handle is an opaque, globally shared handle


    .. autoattribute:: cuda.cuda.CUexternalMemoryHandleType.CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_HEAP


        Handle is a D3D12 heap object


    .. autoattribute:: cuda.cuda.CUexternalMemoryHandleType.CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE


        Handle is a D3D12 committed resource


    .. autoattribute:: cuda.cuda.CUexternalMemoryHandleType.CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_RESOURCE


        Handle is a shared NT handle to a D3D11 resource


    .. autoattribute:: cuda.cuda.CUexternalMemoryHandleType.CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_RESOURCE_KMT


        Handle is a globally shared handle to a D3D11 resource


    .. autoattribute:: cuda.cuda.CUexternalMemoryHandleType.CU_EXTERNAL_MEMORY_HANDLE_TYPE_NVSCIBUF


        Handle is an NvSciBuf object

.. autoclass:: cuda.cuda.CUexternalSemaphoreHandleType

    .. autoattribute:: cuda.cuda.CUexternalSemaphoreHandleType.CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD


        Handle is an opaque file descriptor


    .. autoattribute:: cuda.cuda.CUexternalSemaphoreHandleType.CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32


        Handle is an opaque shared NT handle


    .. autoattribute:: cuda.cuda.CUexternalSemaphoreHandleType.CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_KMT


        Handle is an opaque, globally shared handle


    .. autoattribute:: cuda.cuda.CUexternalSemaphoreHandleType.CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D12_FENCE


        Handle is a shared NT handle referencing a D3D12 fence object


    .. autoattribute:: cuda.cuda.CUexternalSemaphoreHandleType.CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D11_FENCE


        Handle is a shared NT handle referencing a D3D11 fence object


    .. autoattribute:: cuda.cuda.CUexternalSemaphoreHandleType.CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_NVSCISYNC


        Opaque handle to NvSciSync Object


    .. autoattribute:: cuda.cuda.CUexternalSemaphoreHandleType.CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D11_KEYED_MUTEX


        Handle is a shared NT handle referencing a D3D11 keyed mutex object


    .. autoattribute:: cuda.cuda.CUexternalSemaphoreHandleType.CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D11_KEYED_MUTEX_KMT


        Handle is a globally shared handle referencing a D3D11 keyed mutex object


    .. autoattribute:: cuda.cuda.CUexternalSemaphoreHandleType.CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_TIMELINE_SEMAPHORE_FD


        Handle is an opaque file descriptor referencing a timeline semaphore


    .. autoattribute:: cuda.cuda.CUexternalSemaphoreHandleType.CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_TIMELINE_SEMAPHORE_WIN32


        Handle is an opaque shared NT handle referencing a timeline semaphore

.. autoclass:: cuda.cuda.CUmemAllocationHandleType

    .. autoattribute:: cuda.cuda.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_NONE


        Does not allow any export mechanism. >


    .. autoattribute:: cuda.cuda.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR


        Allows a file descriptor to be used for exporting. Permitted only on POSIX systems. (int)


    .. autoattribute:: cuda.cuda.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_WIN32


        Allows a Win32 NT handle to be used for exporting. (HANDLE)


    .. autoattribute:: cuda.cuda.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_WIN32_KMT


        Allows a Win32 KMT handle to be used for exporting. (D3DKMT_HANDLE)


    .. autoattribute:: cuda.cuda.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_MAX

.. autoclass:: cuda.cuda.CUmemAccess_flags

    .. autoattribute:: cuda.cuda.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_NONE


        Default, make the address range not accessible


    .. autoattribute:: cuda.cuda.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READ


        Make the address range read accessible


    .. autoattribute:: cuda.cuda.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE


        Make the address range read-write accessible


    .. autoattribute:: cuda.cuda.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_MAX

.. autoclass:: cuda.cuda.CUmemLocationType

    .. autoattribute:: cuda.cuda.CUmemLocationType.CU_MEM_LOCATION_TYPE_INVALID


    .. autoattribute:: cuda.cuda.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE


        Location is a device location, thus id is a device ordinal


    .. autoattribute:: cuda.cuda.CUmemLocationType.CU_MEM_LOCATION_TYPE_MAX

.. autoclass:: cuda.cuda.CUmemAllocationType

    .. autoattribute:: cuda.cuda.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_INVALID


    .. autoattribute:: cuda.cuda.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED


        This allocation type is 'pinned', i.e. cannot migrate from its current location while the application is actively using it


    .. autoattribute:: cuda.cuda.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_MAX

.. autoclass:: cuda.cuda.CUmemAllocationGranularity_flags

    .. autoattribute:: cuda.cuda.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_MINIMUM


        Minimum required granularity for allocation


    .. autoattribute:: cuda.cuda.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_RECOMMENDED


        Recommended granularity for allocation for best performance

.. autoclass:: cuda.cuda.CUmemRangeHandleType

    .. autoattribute:: cuda.cuda.CUmemRangeHandleType.CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD


    .. autoattribute:: cuda.cuda.CUmemRangeHandleType.CU_MEM_RANGE_HANDLE_TYPE_MAX

.. autoclass:: cuda.cuda.CUarraySparseSubresourceType

    .. autoattribute:: cuda.cuda.CUarraySparseSubresourceType.CU_ARRAY_SPARSE_SUBRESOURCE_TYPE_SPARSE_LEVEL


    .. autoattribute:: cuda.cuda.CUarraySparseSubresourceType.CU_ARRAY_SPARSE_SUBRESOURCE_TYPE_MIPTAIL

.. autoclass:: cuda.cuda.CUmemOperationType

    .. autoattribute:: cuda.cuda.CUmemOperationType.CU_MEM_OPERATION_TYPE_MAP


    .. autoattribute:: cuda.cuda.CUmemOperationType.CU_MEM_OPERATION_TYPE_UNMAP

.. autoclass:: cuda.cuda.CUmemHandleType

    .. autoattribute:: cuda.cuda.CUmemHandleType.CU_MEM_HANDLE_TYPE_GENERIC

.. autoclass:: cuda.cuda.CUmemAllocationCompType

    .. autoattribute:: cuda.cuda.CUmemAllocationCompType.CU_MEM_ALLOCATION_COMP_NONE


        Allocating non-compressible memory


    .. autoattribute:: cuda.cuda.CUmemAllocationCompType.CU_MEM_ALLOCATION_COMP_GENERIC


        Allocating compressible memory

.. autoclass:: cuda.cuda.CUgraphExecUpdateResult

    .. autoattribute:: cuda.cuda.CUgraphExecUpdateResult.CU_GRAPH_EXEC_UPDATE_SUCCESS


        The update succeeded


    .. autoattribute:: cuda.cuda.CUgraphExecUpdateResult.CU_GRAPH_EXEC_UPDATE_ERROR


        The update failed for an unexpected reason which is described in the return value of the function


    .. autoattribute:: cuda.cuda.CUgraphExecUpdateResult.CU_GRAPH_EXEC_UPDATE_ERROR_TOPOLOGY_CHANGED


        The update failed because the topology changed


    .. autoattribute:: cuda.cuda.CUgraphExecUpdateResult.CU_GRAPH_EXEC_UPDATE_ERROR_NODE_TYPE_CHANGED


        The update failed because a node type changed


    .. autoattribute:: cuda.cuda.CUgraphExecUpdateResult.CU_GRAPH_EXEC_UPDATE_ERROR_FUNCTION_CHANGED


        The update failed because the function of a kernel node changed (CUDA driver < 11.2)


    .. autoattribute:: cuda.cuda.CUgraphExecUpdateResult.CU_GRAPH_EXEC_UPDATE_ERROR_PARAMETERS_CHANGED


        The update failed because the parameters changed in a way that is not supported


    .. autoattribute:: cuda.cuda.CUgraphExecUpdateResult.CU_GRAPH_EXEC_UPDATE_ERROR_NOT_SUPPORTED


        The update failed because something about the node is not supported


    .. autoattribute:: cuda.cuda.CUgraphExecUpdateResult.CU_GRAPH_EXEC_UPDATE_ERROR_UNSUPPORTED_FUNCTION_CHANGE


        The update failed because the function of a kernel node changed in an unsupported way


    .. autoattribute:: cuda.cuda.CUgraphExecUpdateResult.CU_GRAPH_EXEC_UPDATE_ERROR_ATTRIBUTES_CHANGED


        The update failed because the node attributes changed in a way that is not supported

.. autoclass:: cuda.cuda.CUmemPool_attribute

    .. autoattribute:: cuda.cuda.CUmemPool_attribute.CU_MEMPOOL_ATTR_REUSE_FOLLOW_EVENT_DEPENDENCIES


        (value type = int) Allow cuMemAllocAsync to use memory asynchronously freed in another streams as long as a stream ordering dependency of the allocating stream on the free action exists. Cuda events and null stream interactions can create the required stream ordered dependencies. (default enabled)


    .. autoattribute:: cuda.cuda.CUmemPool_attribute.CU_MEMPOOL_ATTR_REUSE_ALLOW_OPPORTUNISTIC


        (value type = int) Allow reuse of already completed frees when there is no dependency between the free and allocation. (default enabled)


    .. autoattribute:: cuda.cuda.CUmemPool_attribute.CU_MEMPOOL_ATTR_REUSE_ALLOW_INTERNAL_DEPENDENCIES


        (value type = int) Allow cuMemAllocAsync to insert new stream dependencies in order to establish the stream ordering required to reuse a piece of memory released by cuFreeAsync (default enabled).


    .. autoattribute:: cuda.cuda.CUmemPool_attribute.CU_MEMPOOL_ATTR_RELEASE_THRESHOLD


        (value type = cuuint64_t) Amount of reserved memory in bytes to hold onto before trying to release memory back to the OS. When more than the release threshold bytes of memory are held by the memory pool, the allocator will try to release memory back to the OS on the next call to stream, event or context synchronize. (default 0)


    .. autoattribute:: cuda.cuda.CUmemPool_attribute.CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT


        (value type = cuuint64_t) Amount of backing memory currently allocated for the mempool.


    .. autoattribute:: cuda.cuda.CUmemPool_attribute.CU_MEMPOOL_ATTR_RESERVED_MEM_HIGH


        (value type = cuuint64_t) High watermark of backing memory allocated for the mempool since the last time it was reset. High watermark can only be reset to zero.


    .. autoattribute:: cuda.cuda.CUmemPool_attribute.CU_MEMPOOL_ATTR_USED_MEM_CURRENT


        (value type = cuuint64_t) Amount of memory from the pool that is currently in use by the application.


    .. autoattribute:: cuda.cuda.CUmemPool_attribute.CU_MEMPOOL_ATTR_USED_MEM_HIGH


        (value type = cuuint64_t) High watermark of the amount of memory from the pool that was in use by the application since the last time it was reset. High watermark can only be reset to zero.

.. autoclass:: cuda.cuda.CUgraphMem_attribute

    .. autoattribute:: cuda.cuda.CUgraphMem_attribute.CU_GRAPH_MEM_ATTR_USED_MEM_CURRENT


        (value type = cuuint64_t) Amount of memory, in bytes, currently associated with graphs


    .. autoattribute:: cuda.cuda.CUgraphMem_attribute.CU_GRAPH_MEM_ATTR_USED_MEM_HIGH


        (value type = cuuint64_t) High watermark of memory, in bytes, associated with graphs since the last time it was reset. High watermark can only be reset to zero.


    .. autoattribute:: cuda.cuda.CUgraphMem_attribute.CU_GRAPH_MEM_ATTR_RESERVED_MEM_CURRENT


        (value type = cuuint64_t) Amount of memory, in bytes, currently allocated for use by the CUDA graphs asynchronous allocator.


    .. autoattribute:: cuda.cuda.CUgraphMem_attribute.CU_GRAPH_MEM_ATTR_RESERVED_MEM_HIGH


        (value type = cuuint64_t) High watermark of memory, in bytes, currently allocated for use by the CUDA graphs asynchronous allocator.

.. autoclass:: cuda.cuda.CUflushGPUDirectRDMAWritesOptions

    .. autoattribute:: cuda.cuda.CUflushGPUDirectRDMAWritesOptions.CU_FLUSH_GPU_DIRECT_RDMA_WRITES_OPTION_HOST


        :py:obj:`~.cuFlushGPUDirectRDMAWrites()` and its CUDA Runtime API counterpart are supported on the device.


    .. autoattribute:: cuda.cuda.CUflushGPUDirectRDMAWritesOptions.CU_FLUSH_GPU_DIRECT_RDMA_WRITES_OPTION_MEMOPS


        The :py:obj:`~.CU_STREAM_WAIT_VALUE_FLUSH` flag and the :py:obj:`~.CU_STREAM_MEM_OP_FLUSH_REMOTE_WRITES` MemOp are supported on the device.

.. autoclass:: cuda.cuda.CUGPUDirectRDMAWritesOrdering

    .. autoattribute:: cuda.cuda.CUGPUDirectRDMAWritesOrdering.CU_GPU_DIRECT_RDMA_WRITES_ORDERING_NONE


        The device does not natively support ordering of remote writes. :py:obj:`~.cuFlushGPUDirectRDMAWrites()` can be leveraged if supported.


    .. autoattribute:: cuda.cuda.CUGPUDirectRDMAWritesOrdering.CU_GPU_DIRECT_RDMA_WRITES_ORDERING_OWNER


        Natively, the device can consistently consume remote writes, although other CUDA devices may not.


    .. autoattribute:: cuda.cuda.CUGPUDirectRDMAWritesOrdering.CU_GPU_DIRECT_RDMA_WRITES_ORDERING_ALL_DEVICES


        Any CUDA device in the system can consistently consume remote writes to this device.

.. autoclass:: cuda.cuda.CUflushGPUDirectRDMAWritesScope

    .. autoattribute:: cuda.cuda.CUflushGPUDirectRDMAWritesScope.CU_FLUSH_GPU_DIRECT_RDMA_WRITES_TO_OWNER


        Blocks until remote writes are visible to the CUDA device context owning the data.


    .. autoattribute:: cuda.cuda.CUflushGPUDirectRDMAWritesScope.CU_FLUSH_GPU_DIRECT_RDMA_WRITES_TO_ALL_DEVICES


        Blocks until remote writes are visible to all CUDA device contexts.

.. autoclass:: cuda.cuda.CUflushGPUDirectRDMAWritesTarget

    .. autoattribute:: cuda.cuda.CUflushGPUDirectRDMAWritesTarget.CU_FLUSH_GPU_DIRECT_RDMA_WRITES_TARGET_CURRENT_CTX


        Sets the target for :py:obj:`~.cuFlushGPUDirectRDMAWrites()` to the currently active CUDA device context.

.. autoclass:: cuda.cuda.CUgraphDebugDot_flags

    .. autoattribute:: cuda.cuda.CUgraphDebugDot_flags.CU_GRAPH_DEBUG_DOT_FLAGS_VERBOSE


    .. autoattribute:: cuda.cuda.CUgraphDebugDot_flags.CU_GRAPH_DEBUG_DOT_FLAGS_RUNTIME_TYPES


        Output all debug data as if every debug flag is enabled


    .. autoattribute:: cuda.cuda.CUgraphDebugDot_flags.CU_GRAPH_DEBUG_DOT_FLAGS_KERNEL_NODE_PARAMS


        Use CUDA Runtime structures for output


    .. autoattribute:: cuda.cuda.CUgraphDebugDot_flags.CU_GRAPH_DEBUG_DOT_FLAGS_MEMCPY_NODE_PARAMS


        Adds CUDA_KERNEL_NODE_PARAMS values to output


    .. autoattribute:: cuda.cuda.CUgraphDebugDot_flags.CU_GRAPH_DEBUG_DOT_FLAGS_MEMSET_NODE_PARAMS


        Adds CUDA_MEMCPY3D values to output


    .. autoattribute:: cuda.cuda.CUgraphDebugDot_flags.CU_GRAPH_DEBUG_DOT_FLAGS_HOST_NODE_PARAMS


        Adds CUDA_MEMSET_NODE_PARAMS values to output


    .. autoattribute:: cuda.cuda.CUgraphDebugDot_flags.CU_GRAPH_DEBUG_DOT_FLAGS_EVENT_NODE_PARAMS


        Adds CUDA_HOST_NODE_PARAMS values to output


    .. autoattribute:: cuda.cuda.CUgraphDebugDot_flags.CU_GRAPH_DEBUG_DOT_FLAGS_EXT_SEMAS_SIGNAL_NODE_PARAMS


        Adds CUevent handle from record and wait nodes to output


    .. autoattribute:: cuda.cuda.CUgraphDebugDot_flags.CU_GRAPH_DEBUG_DOT_FLAGS_EXT_SEMAS_WAIT_NODE_PARAMS


        Adds CUDA_EXT_SEM_SIGNAL_NODE_PARAMS values to output


    .. autoattribute:: cuda.cuda.CUgraphDebugDot_flags.CU_GRAPH_DEBUG_DOT_FLAGS_KERNEL_NODE_ATTRIBUTES


        Adds CUDA_EXT_SEM_WAIT_NODE_PARAMS values to output


    .. autoattribute:: cuda.cuda.CUgraphDebugDot_flags.CU_GRAPH_DEBUG_DOT_FLAGS_HANDLES


        Adds CUkernelNodeAttrValue values to output


    .. autoattribute:: cuda.cuda.CUgraphDebugDot_flags.CU_GRAPH_DEBUG_DOT_FLAGS_MEM_ALLOC_NODE_PARAMS


        Adds node handles and every kernel function handle to output


    .. autoattribute:: cuda.cuda.CUgraphDebugDot_flags.CU_GRAPH_DEBUG_DOT_FLAGS_MEM_FREE_NODE_PARAMS


        Adds memory alloc node parameters to output


    .. autoattribute:: cuda.cuda.CUgraphDebugDot_flags.CU_GRAPH_DEBUG_DOT_FLAGS_BATCH_MEM_OP_NODE_PARAMS

.. autoclass:: cuda.cuda.CUuserObject_flags

    .. autoattribute:: cuda.cuda.CUuserObject_flags.CU_USER_OBJECT_NO_DESTRUCTOR_SYNC


        Indicates the destructor execution is not synchronized by any CUDA handle.

.. autoclass:: cuda.cuda.CUuserObjectRetain_flags

    .. autoattribute:: cuda.cuda.CUuserObjectRetain_flags.CU_GRAPH_USER_OBJECT_MOVE


        Transfer references from the caller rather than creating new references.

.. autoclass:: cuda.cuda.CUgraphInstantiate_flags

    .. autoattribute:: cuda.cuda.CUgraphInstantiate_flags.CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH


        Automatically free memory allocated in a graph before relaunching.


    .. autoattribute:: cuda.cuda.CUgraphInstantiate_flags.CUDA_GRAPH_INSTANTIATE_FLAG_USE_NODE_PRIORITY


        Run the graph using the per-node priority attributes rather than the priority of the stream it is launched into.

.. autoclass:: cuda.cuda.CUeglFrameType

    .. autoattribute:: cuda.cuda.CUeglFrameType.CU_EGL_FRAME_TYPE_ARRAY


        Frame type CUDA array


    .. autoattribute:: cuda.cuda.CUeglFrameType.CU_EGL_FRAME_TYPE_PITCH


        Frame type pointer

.. autoclass:: cuda.cuda.CUeglResourceLocationFlags

    .. autoattribute:: cuda.cuda.CUeglResourceLocationFlags.CU_EGL_RESOURCE_LOCATION_SYSMEM


        Resource location sysmem


    .. autoattribute:: cuda.cuda.CUeglResourceLocationFlags.CU_EGL_RESOURCE_LOCATION_VIDMEM


        Resource location vidmem

.. autoclass:: cuda.cuda.CUeglColorFormat

    .. autoattribute:: cuda.cuda.CUeglColorFormat.CU_EGL_COLOR_FORMAT_YUV420_PLANAR


        Y, U, V in three surfaces, each in a separate surface, U/V width = 1/2 Y width, U/V height = 1/2 Y height.


    .. autoattribute:: cuda.cuda.CUeglColorFormat.CU_EGL_COLOR_FORMAT_YUV420_SEMIPLANAR


        Y, UV in two surfaces (UV as one surface) with VU byte ordering, width, height ratio same as YUV420Planar.


    .. autoattribute:: cuda.cuda.CUeglColorFormat.CU_EGL_COLOR_FORMAT_YUV422_PLANAR


        Y, U, V each in a separate surface, U/V width = 1/2 Y width, U/V height = Y height.


    .. autoattribute:: cuda.cuda.CUeglColorFormat.CU_EGL_COLOR_FORMAT_YUV422_SEMIPLANAR


        Y, UV in two surfaces with VU byte ordering, width, height ratio same as YUV422Planar.


    .. autoattribute:: cuda.cuda.CUeglColorFormat.CU_EGL_COLOR_FORMAT_RGB


        R/G/B three channels in one surface with BGR byte ordering. Only pitch linear format supported.


    .. autoattribute:: cuda.cuda.CUeglColorFormat.CU_EGL_COLOR_FORMAT_BGR


        R/G/B three channels in one surface with RGB byte ordering. Only pitch linear format supported.


    .. autoattribute:: cuda.cuda.CUeglColorFormat.CU_EGL_COLOR_FORMAT_ARGB


        R/G/B/A four channels in one surface with BGRA byte ordering.


    .. autoattribute:: cuda.cuda.CUeglColorFormat.CU_EGL_COLOR_FORMAT_RGBA


        R/G/B/A four channels in one surface with ABGR byte ordering.


    .. autoattribute:: cuda.cuda.CUeglColorFormat.CU_EGL_COLOR_FORMAT_L


        single luminance channel in one surface.


    .. autoattribute:: cuda.cuda.CUeglColorFormat.CU_EGL_COLOR_FORMAT_R


        single color channel in one surface.


    .. autoattribute:: cuda.cuda.CUeglColorFormat.CU_EGL_COLOR_FORMAT_YUV444_PLANAR


        Y, U, V in three surfaces, each in a separate surface, U/V width = Y width, U/V height = Y height.


    .. autoattribute:: cuda.cuda.CUeglColorFormat.CU_EGL_COLOR_FORMAT_YUV444_SEMIPLANAR


        Y, UV in two surfaces (UV as one surface) with VU byte ordering, width, height ratio same as YUV444Planar.


    .. autoattribute:: cuda.cuda.CUeglColorFormat.CU_EGL_COLOR_FORMAT_YUYV_422


        Y, U, V in one surface, interleaved as UYVY in one channel.


    .. autoattribute:: cuda.cuda.CUeglColorFormat.CU_EGL_COLOR_FORMAT_UYVY_422


        Y, U, V in one surface, interleaved as YUYV in one channel.


    .. autoattribute:: cuda.cuda.CUeglColorFormat.CU_EGL_COLOR_FORMAT_ABGR


        R/G/B/A four channels in one surface with RGBA byte ordering.


    .. autoattribute:: cuda.cuda.CUeglColorFormat.CU_EGL_COLOR_FORMAT_BGRA


        R/G/B/A four channels in one surface with ARGB byte ordering.


    .. autoattribute:: cuda.cuda.CUeglColorFormat.CU_EGL_COLOR_FORMAT_A


        Alpha color format - one channel in one surface.


    .. autoattribute:: cuda.cuda.CUeglColorFormat.CU_EGL_COLOR_FORMAT_RG


        R/G color format - two channels in one surface with GR byte ordering


    .. autoattribute:: cuda.cuda.CUeglColorFormat.CU_EGL_COLOR_FORMAT_AYUV


        Y, U, V, A four channels in one surface, interleaved as VUYA.


    .. autoattribute:: cuda.cuda.CUeglColorFormat.CU_EGL_COLOR_FORMAT_YVU444_SEMIPLANAR


        Y, VU in two surfaces (VU as one surface) with UV byte ordering, U/V width = Y width, U/V height = Y height.


    .. autoattribute:: cuda.cuda.CUeglColorFormat.CU_EGL_COLOR_FORMAT_YVU422_SEMIPLANAR


        Y, VU in two surfaces (VU as one surface) with UV byte ordering, U/V width = 1/2 Y width, U/V height = Y height.


    .. autoattribute:: cuda.cuda.CUeglColorFormat.CU_EGL_COLOR_FORMAT_YVU420_SEMIPLANAR


        Y, VU in two surfaces (VU as one surface) with UV byte ordering, U/V width = 1/2 Y width, U/V height = 1/2 Y height.


    .. autoattribute:: cuda.cuda.CUeglColorFormat.CU_EGL_COLOR_FORMAT_Y10V10U10_444_SEMIPLANAR


        Y10, V10U10 in two surfaces (VU as one surface) with UV byte ordering, U/V width = Y width, U/V height = Y height.


    .. autoattribute:: cuda.cuda.CUeglColorFormat.CU_EGL_COLOR_FORMAT_Y10V10U10_420_SEMIPLANAR


        Y10, V10U10 in two surfaces (VU as one surface) with UV byte ordering, U/V width = 1/2 Y width, U/V height = 1/2 Y height.


    .. autoattribute:: cuda.cuda.CUeglColorFormat.CU_EGL_COLOR_FORMAT_Y12V12U12_444_SEMIPLANAR


        Y12, V12U12 in two surfaces (VU as one surface) with UV byte ordering, U/V width = Y width, U/V height = Y height.


    .. autoattribute:: cuda.cuda.CUeglColorFormat.CU_EGL_COLOR_FORMAT_Y12V12U12_420_SEMIPLANAR


        Y12, V12U12 in two surfaces (VU as one surface) with UV byte ordering, U/V width = 1/2 Y width, U/V height = 1/2 Y height.


    .. autoattribute:: cuda.cuda.CUeglColorFormat.CU_EGL_COLOR_FORMAT_VYUY_ER


        Extended Range Y, U, V in one surface, interleaved as YVYU in one channel.


    .. autoattribute:: cuda.cuda.CUeglColorFormat.CU_EGL_COLOR_FORMAT_UYVY_ER


        Extended Range Y, U, V in one surface, interleaved as YUYV in one channel.


    .. autoattribute:: cuda.cuda.CUeglColorFormat.CU_EGL_COLOR_FORMAT_YUYV_ER


        Extended Range Y, U, V in one surface, interleaved as UYVY in one channel.


    .. autoattribute:: cuda.cuda.CUeglColorFormat.CU_EGL_COLOR_FORMAT_YVYU_ER


        Extended Range Y, U, V in one surface, interleaved as VYUY in one channel.


    .. autoattribute:: cuda.cuda.CUeglColorFormat.CU_EGL_COLOR_FORMAT_YUV_ER


        Extended Range Y, U, V three channels in one surface, interleaved as VUY. Only pitch linear format supported.


    .. autoattribute:: cuda.cuda.CUeglColorFormat.CU_EGL_COLOR_FORMAT_YUVA_ER


        Extended Range Y, U, V, A four channels in one surface, interleaved as AVUY.


    .. autoattribute:: cuda.cuda.CUeglColorFormat.CU_EGL_COLOR_FORMAT_AYUV_ER


        Extended Range Y, U, V, A four channels in one surface, interleaved as VUYA.


    .. autoattribute:: cuda.cuda.CUeglColorFormat.CU_EGL_COLOR_FORMAT_YUV444_PLANAR_ER


        Extended Range Y, U, V in three surfaces, U/V width = Y width, U/V height = Y height.


    .. autoattribute:: cuda.cuda.CUeglColorFormat.CU_EGL_COLOR_FORMAT_YUV422_PLANAR_ER


        Extended Range Y, U, V in three surfaces, U/V width = 1/2 Y width, U/V height = Y height.


    .. autoattribute:: cuda.cuda.CUeglColorFormat.CU_EGL_COLOR_FORMAT_YUV420_PLANAR_ER


        Extended Range Y, U, V in three surfaces, U/V width = 1/2 Y width, U/V height = 1/2 Y height.


    .. autoattribute:: cuda.cuda.CUeglColorFormat.CU_EGL_COLOR_FORMAT_YUV444_SEMIPLANAR_ER


        Extended Range Y, UV in two surfaces (UV as one surface) with VU byte ordering, U/V width = Y width, U/V height = Y height.


    .. autoattribute:: cuda.cuda.CUeglColorFormat.CU_EGL_COLOR_FORMAT_YUV422_SEMIPLANAR_ER


        Extended Range Y, UV in two surfaces (UV as one surface) with VU byte ordering, U/V width = 1/2 Y width, U/V height = Y height.


    .. autoattribute:: cuda.cuda.CUeglColorFormat.CU_EGL_COLOR_FORMAT_YUV420_SEMIPLANAR_ER


        Extended Range Y, UV in two surfaces (UV as one surface) with VU byte ordering, U/V width = 1/2 Y width, U/V height = 1/2 Y height.


    .. autoattribute:: cuda.cuda.CUeglColorFormat.CU_EGL_COLOR_FORMAT_YVU444_PLANAR_ER


        Extended Range Y, V, U in three surfaces, U/V width = Y width, U/V height = Y height.


    .. autoattribute:: cuda.cuda.CUeglColorFormat.CU_EGL_COLOR_FORMAT_YVU422_PLANAR_ER


        Extended Range Y, V, U in three surfaces, U/V width = 1/2 Y width, U/V height = Y height.


    .. autoattribute:: cuda.cuda.CUeglColorFormat.CU_EGL_COLOR_FORMAT_YVU420_PLANAR_ER


        Extended Range Y, V, U in three surfaces, U/V width = 1/2 Y width, U/V height = 1/2 Y height.


    .. autoattribute:: cuda.cuda.CUeglColorFormat.CU_EGL_COLOR_FORMAT_YVU444_SEMIPLANAR_ER


        Extended Range Y, VU in two surfaces (VU as one surface) with UV byte ordering, U/V width = Y width, U/V height = Y height.


    .. autoattribute:: cuda.cuda.CUeglColorFormat.CU_EGL_COLOR_FORMAT_YVU422_SEMIPLANAR_ER


        Extended Range Y, VU in two surfaces (VU as one surface) with UV byte ordering, U/V width = 1/2 Y width, U/V height = Y height.


    .. autoattribute:: cuda.cuda.CUeglColorFormat.CU_EGL_COLOR_FORMAT_YVU420_SEMIPLANAR_ER


        Extended Range Y, VU in two surfaces (VU as one surface) with UV byte ordering, U/V width = 1/2 Y width, U/V height = 1/2 Y height.


    .. autoattribute:: cuda.cuda.CUeglColorFormat.CU_EGL_COLOR_FORMAT_BAYER_RGGB


        Bayer format - one channel in one surface with interleaved RGGB ordering.


    .. autoattribute:: cuda.cuda.CUeglColorFormat.CU_EGL_COLOR_FORMAT_BAYER_BGGR


        Bayer format - one channel in one surface with interleaved BGGR ordering.


    .. autoattribute:: cuda.cuda.CUeglColorFormat.CU_EGL_COLOR_FORMAT_BAYER_GRBG


        Bayer format - one channel in one surface with interleaved GRBG ordering.


    .. autoattribute:: cuda.cuda.CUeglColorFormat.CU_EGL_COLOR_FORMAT_BAYER_GBRG


        Bayer format - one channel in one surface with interleaved GBRG ordering.


    .. autoattribute:: cuda.cuda.CUeglColorFormat.CU_EGL_COLOR_FORMAT_BAYER10_RGGB


        Bayer10 format - one channel in one surface with interleaved RGGB ordering. Out of 16 bits, 10 bits used 6 bits No-op.


    .. autoattribute:: cuda.cuda.CUeglColorFormat.CU_EGL_COLOR_FORMAT_BAYER10_BGGR


        Bayer10 format - one channel in one surface with interleaved BGGR ordering. Out of 16 bits, 10 bits used 6 bits No-op.


    .. autoattribute:: cuda.cuda.CUeglColorFormat.CU_EGL_COLOR_FORMAT_BAYER10_GRBG


        Bayer10 format - one channel in one surface with interleaved GRBG ordering. Out of 16 bits, 10 bits used 6 bits No-op.


    .. autoattribute:: cuda.cuda.CUeglColorFormat.CU_EGL_COLOR_FORMAT_BAYER10_GBRG


        Bayer10 format - one channel in one surface with interleaved GBRG ordering. Out of 16 bits, 10 bits used 6 bits No-op.


    .. autoattribute:: cuda.cuda.CUeglColorFormat.CU_EGL_COLOR_FORMAT_BAYER12_RGGB


        Bayer12 format - one channel in one surface with interleaved RGGB ordering. Out of 16 bits, 12 bits used 4 bits No-op.


    .. autoattribute:: cuda.cuda.CUeglColorFormat.CU_EGL_COLOR_FORMAT_BAYER12_BGGR


        Bayer12 format - one channel in one surface with interleaved BGGR ordering. Out of 16 bits, 12 bits used 4 bits No-op.


    .. autoattribute:: cuda.cuda.CUeglColorFormat.CU_EGL_COLOR_FORMAT_BAYER12_GRBG


        Bayer12 format - one channel in one surface with interleaved GRBG ordering. Out of 16 bits, 12 bits used 4 bits No-op.


    .. autoattribute:: cuda.cuda.CUeglColorFormat.CU_EGL_COLOR_FORMAT_BAYER12_GBRG


        Bayer12 format - one channel in one surface with interleaved GBRG ordering. Out of 16 bits, 12 bits used 4 bits No-op.


    .. autoattribute:: cuda.cuda.CUeglColorFormat.CU_EGL_COLOR_FORMAT_BAYER14_RGGB


        Bayer14 format - one channel in one surface with interleaved RGGB ordering. Out of 16 bits, 14 bits used 2 bits No-op.


    .. autoattribute:: cuda.cuda.CUeglColorFormat.CU_EGL_COLOR_FORMAT_BAYER14_BGGR


        Bayer14 format - one channel in one surface with interleaved BGGR ordering. Out of 16 bits, 14 bits used 2 bits No-op.


    .. autoattribute:: cuda.cuda.CUeglColorFormat.CU_EGL_COLOR_FORMAT_BAYER14_GRBG


        Bayer14 format - one channel in one surface with interleaved GRBG ordering. Out of 16 bits, 14 bits used 2 bits No-op.


    .. autoattribute:: cuda.cuda.CUeglColorFormat.CU_EGL_COLOR_FORMAT_BAYER14_GBRG


        Bayer14 format - one channel in one surface with interleaved GBRG ordering. Out of 16 bits, 14 bits used 2 bits No-op.


    .. autoattribute:: cuda.cuda.CUeglColorFormat.CU_EGL_COLOR_FORMAT_BAYER20_RGGB


        Bayer20 format - one channel in one surface with interleaved RGGB ordering. Out of 32 bits, 20 bits used 12 bits No-op.


    .. autoattribute:: cuda.cuda.CUeglColorFormat.CU_EGL_COLOR_FORMAT_BAYER20_BGGR


        Bayer20 format - one channel in one surface with interleaved BGGR ordering. Out of 32 bits, 20 bits used 12 bits No-op.


    .. autoattribute:: cuda.cuda.CUeglColorFormat.CU_EGL_COLOR_FORMAT_BAYER20_GRBG


        Bayer20 format - one channel in one surface with interleaved GRBG ordering. Out of 32 bits, 20 bits used 12 bits No-op.


    .. autoattribute:: cuda.cuda.CUeglColorFormat.CU_EGL_COLOR_FORMAT_BAYER20_GBRG


        Bayer20 format - one channel in one surface with interleaved GBRG ordering. Out of 32 bits, 20 bits used 12 bits No-op.


    .. autoattribute:: cuda.cuda.CUeglColorFormat.CU_EGL_COLOR_FORMAT_YVU444_PLANAR


        Y, V, U in three surfaces, each in a separate surface, U/V width = Y width, U/V height = Y height.


    .. autoattribute:: cuda.cuda.CUeglColorFormat.CU_EGL_COLOR_FORMAT_YVU422_PLANAR


        Y, V, U in three surfaces, each in a separate surface, U/V width = 1/2 Y width, U/V height = Y height.


    .. autoattribute:: cuda.cuda.CUeglColorFormat.CU_EGL_COLOR_FORMAT_YVU420_PLANAR


        Y, V, U in three surfaces, each in a separate surface, U/V width = 1/2 Y width, U/V height = 1/2 Y height.


    .. autoattribute:: cuda.cuda.CUeglColorFormat.CU_EGL_COLOR_FORMAT_BAYER_ISP_RGGB


        Nvidia proprietary Bayer ISP format - one channel in one surface with interleaved RGGB ordering and mapped to opaque integer datatype.


    .. autoattribute:: cuda.cuda.CUeglColorFormat.CU_EGL_COLOR_FORMAT_BAYER_ISP_BGGR


        Nvidia proprietary Bayer ISP format - one channel in one surface with interleaved BGGR ordering and mapped to opaque integer datatype.


    .. autoattribute:: cuda.cuda.CUeglColorFormat.CU_EGL_COLOR_FORMAT_BAYER_ISP_GRBG


        Nvidia proprietary Bayer ISP format - one channel in one surface with interleaved GRBG ordering and mapped to opaque integer datatype.


    .. autoattribute:: cuda.cuda.CUeglColorFormat.CU_EGL_COLOR_FORMAT_BAYER_ISP_GBRG


        Nvidia proprietary Bayer ISP format - one channel in one surface with interleaved GBRG ordering and mapped to opaque integer datatype.


    .. autoattribute:: cuda.cuda.CUeglColorFormat.CU_EGL_COLOR_FORMAT_BAYER_BCCR


        Bayer format - one channel in one surface with interleaved BCCR ordering.


    .. autoattribute:: cuda.cuda.CUeglColorFormat.CU_EGL_COLOR_FORMAT_BAYER_RCCB


        Bayer format - one channel in one surface with interleaved RCCB ordering.


    .. autoattribute:: cuda.cuda.CUeglColorFormat.CU_EGL_COLOR_FORMAT_BAYER_CRBC


        Bayer format - one channel in one surface with interleaved CRBC ordering.


    .. autoattribute:: cuda.cuda.CUeglColorFormat.CU_EGL_COLOR_FORMAT_BAYER_CBRC


        Bayer format - one channel in one surface with interleaved CBRC ordering.


    .. autoattribute:: cuda.cuda.CUeglColorFormat.CU_EGL_COLOR_FORMAT_BAYER10_CCCC


        Bayer10 format - one channel in one surface with interleaved CCCC ordering. Out of 16 bits, 10 bits used 6 bits No-op.


    .. autoattribute:: cuda.cuda.CUeglColorFormat.CU_EGL_COLOR_FORMAT_BAYER12_BCCR


        Bayer12 format - one channel in one surface with interleaved BCCR ordering. Out of 16 bits, 12 bits used 4 bits No-op.


    .. autoattribute:: cuda.cuda.CUeglColorFormat.CU_EGL_COLOR_FORMAT_BAYER12_RCCB


        Bayer12 format - one channel in one surface with interleaved RCCB ordering. Out of 16 bits, 12 bits used 4 bits No-op.


    .. autoattribute:: cuda.cuda.CUeglColorFormat.CU_EGL_COLOR_FORMAT_BAYER12_CRBC


        Bayer12 format - one channel in one surface with interleaved CRBC ordering. Out of 16 bits, 12 bits used 4 bits No-op.


    .. autoattribute:: cuda.cuda.CUeglColorFormat.CU_EGL_COLOR_FORMAT_BAYER12_CBRC


        Bayer12 format - one channel in one surface with interleaved CBRC ordering. Out of 16 bits, 12 bits used 4 bits No-op.


    .. autoattribute:: cuda.cuda.CUeglColorFormat.CU_EGL_COLOR_FORMAT_BAYER12_CCCC


        Bayer12 format - one channel in one surface with interleaved CCCC ordering. Out of 16 bits, 12 bits used 4 bits No-op.


    .. autoattribute:: cuda.cuda.CUeglColorFormat.CU_EGL_COLOR_FORMAT_Y


        Color format for single Y plane.


    .. autoattribute:: cuda.cuda.CUeglColorFormat.CU_EGL_COLOR_FORMAT_YUV420_SEMIPLANAR_2020


        Y, UV in two surfaces (UV as one surface) U/V width = 1/2 Y width, U/V height = 1/2 Y height.


    .. autoattribute:: cuda.cuda.CUeglColorFormat.CU_EGL_COLOR_FORMAT_YVU420_SEMIPLANAR_2020


        Y, VU in two surfaces (VU as one surface) U/V width = 1/2 Y width, U/V height = 1/2 Y height.


    .. autoattribute:: cuda.cuda.CUeglColorFormat.CU_EGL_COLOR_FORMAT_YUV420_PLANAR_2020


        Y, U, V each in a separate surface, U/V width = 1/2 Y width, U/V height= 1/2 Y height.


    .. autoattribute:: cuda.cuda.CUeglColorFormat.CU_EGL_COLOR_FORMAT_YVU420_PLANAR_2020


        Y, V, U each in a separate surface, U/V width = 1/2 Y width, U/V height = 1/2 Y height.


    .. autoattribute:: cuda.cuda.CUeglColorFormat.CU_EGL_COLOR_FORMAT_YUV420_SEMIPLANAR_709


        Y, UV in two surfaces (UV as one surface) U/V width = 1/2 Y width, U/V height = 1/2 Y height.


    .. autoattribute:: cuda.cuda.CUeglColorFormat.CU_EGL_COLOR_FORMAT_YVU420_SEMIPLANAR_709


        Y, VU in two surfaces (VU as one surface) U/V width = 1/2 Y width, U/V height = 1/2 Y height.


    .. autoattribute:: cuda.cuda.CUeglColorFormat.CU_EGL_COLOR_FORMAT_YUV420_PLANAR_709


        Y, U, V each in a separate surface, U/V width = 1/2 Y width, U/V height = 1/2 Y height.


    .. autoattribute:: cuda.cuda.CUeglColorFormat.CU_EGL_COLOR_FORMAT_YVU420_PLANAR_709


        Y, V, U each in a separate surface, U/V width = 1/2 Y width, U/V height = 1/2 Y height.


    .. autoattribute:: cuda.cuda.CUeglColorFormat.CU_EGL_COLOR_FORMAT_Y10V10U10_420_SEMIPLANAR_709


        Y10, V10U10 in two surfaces (VU as one surface), U/V width = 1/2 Y width, U/V height = 1/2 Y height.


    .. autoattribute:: cuda.cuda.CUeglColorFormat.CU_EGL_COLOR_FORMAT_Y10V10U10_420_SEMIPLANAR_2020


        Y10, V10U10 in two surfaces (VU as one surface), U/V width = 1/2 Y width, U/V height = 1/2 Y height.


    .. autoattribute:: cuda.cuda.CUeglColorFormat.CU_EGL_COLOR_FORMAT_Y10V10U10_422_SEMIPLANAR_2020


        Y10, V10U10 in two surfaces(VU as one surface) U/V width = 1/2 Y width, U/V height = Y height.


    .. autoattribute:: cuda.cuda.CUeglColorFormat.CU_EGL_COLOR_FORMAT_Y10V10U10_422_SEMIPLANAR


        Y10, V10U10 in two surfaces(VU as one surface) U/V width = 1/2 Y width, U/V height = Y height.


    .. autoattribute:: cuda.cuda.CUeglColorFormat.CU_EGL_COLOR_FORMAT_Y10V10U10_422_SEMIPLANAR_709


        Y10, V10U10 in two surfaces(VU as one surface) U/V width = 1/2 Y width, U/V height = Y height.


    .. autoattribute:: cuda.cuda.CUeglColorFormat.CU_EGL_COLOR_FORMAT_Y_ER


        Extended Range Color format for single Y plane.


    .. autoattribute:: cuda.cuda.CUeglColorFormat.CU_EGL_COLOR_FORMAT_Y_709_ER


        Extended Range Color format for single Y plane.


    .. autoattribute:: cuda.cuda.CUeglColorFormat.CU_EGL_COLOR_FORMAT_Y10_ER


        Extended Range Color format for single Y10 plane.


    .. autoattribute:: cuda.cuda.CUeglColorFormat.CU_EGL_COLOR_FORMAT_Y10_709_ER


        Extended Range Color format for single Y10 plane.


    .. autoattribute:: cuda.cuda.CUeglColorFormat.CU_EGL_COLOR_FORMAT_Y12_ER


        Extended Range Color format for single Y12 plane.


    .. autoattribute:: cuda.cuda.CUeglColorFormat.CU_EGL_COLOR_FORMAT_Y12_709_ER


        Extended Range Color format for single Y12 plane.


    .. autoattribute:: cuda.cuda.CUeglColorFormat.CU_EGL_COLOR_FORMAT_YUVA


        Y, U, V, A four channels in one surface, interleaved as AVUY.


    .. autoattribute:: cuda.cuda.CUeglColorFormat.CU_EGL_COLOR_FORMAT_YUV


        Y, U, V three channels in one surface, interleaved as VUY. Only pitch linear format supported.


    .. autoattribute:: cuda.cuda.CUeglColorFormat.CU_EGL_COLOR_FORMAT_YVYU


        Y, U, V in one surface, interleaved as YVYU in one channel.


    .. autoattribute:: cuda.cuda.CUeglColorFormat.CU_EGL_COLOR_FORMAT_VYUY


        Y, U, V in one surface, interleaved as VYUY in one channel.


    .. autoattribute:: cuda.cuda.CUeglColorFormat.CU_EGL_COLOR_FORMAT_Y10V10U10_420_SEMIPLANAR_ER


        Extended Range Y10, V10U10 in two surfaces(VU as one surface) U/V width = 1/2 Y width, U/V height = 1/2 Y height.


    .. autoattribute:: cuda.cuda.CUeglColorFormat.CU_EGL_COLOR_FORMAT_Y10V10U10_420_SEMIPLANAR_709_ER


        Extended Range Y10, V10U10 in two surfaces(VU as one surface) U/V width = 1/2 Y width, U/V height = 1/2 Y height.


    .. autoattribute:: cuda.cuda.CUeglColorFormat.CU_EGL_COLOR_FORMAT_Y10V10U10_444_SEMIPLANAR_ER


        Extended Range Y10, V10U10 in two surfaces (VU as one surface) U/V width = Y width, U/V height = Y height.


    .. autoattribute:: cuda.cuda.CUeglColorFormat.CU_EGL_COLOR_FORMAT_Y10V10U10_444_SEMIPLANAR_709_ER


        Extended Range Y10, V10U10 in two surfaces (VU as one surface) U/V width = Y width, U/V height = Y height.


    .. autoattribute:: cuda.cuda.CUeglColorFormat.CU_EGL_COLOR_FORMAT_Y12V12U12_420_SEMIPLANAR_ER


        Extended Range Y12, V12U12 in two surfaces (VU as one surface) U/V width = 1/2 Y width, U/V height = 1/2 Y height.


    .. autoattribute:: cuda.cuda.CUeglColorFormat.CU_EGL_COLOR_FORMAT_Y12V12U12_420_SEMIPLANAR_709_ER


        Extended Range Y12, V12U12 in two surfaces (VU as one surface) U/V width = 1/2 Y width, U/V height = 1/2 Y height.


    .. autoattribute:: cuda.cuda.CUeglColorFormat.CU_EGL_COLOR_FORMAT_Y12V12U12_444_SEMIPLANAR_ER


        Extended Range Y12, V12U12 in two surfaces (VU as one surface) U/V width = Y width, U/V height = Y height.


    .. autoattribute:: cuda.cuda.CUeglColorFormat.CU_EGL_COLOR_FORMAT_Y12V12U12_444_SEMIPLANAR_709_ER


        Extended Range Y12, V12U12 in two surfaces (VU as one surface) U/V width = Y width, U/V height = Y height.


    .. autoattribute:: cuda.cuda.CUeglColorFormat.CU_EGL_COLOR_FORMAT_MAX

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
.. autoclass:: cuda.cuda.CUDA_BATCH_MEM_OP_NODE_PARAMS
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
.. autoattribute:: cuda.cuda.CUDA_VERSION

    CUDA API version number

.. autoattribute:: cuda.cuda.CU_UUID_HAS_BEEN_DEFINED

    CUDA UUID types

.. autoattribute:: cuda.cuda.CU_IPC_HANDLE_SIZE

    CUDA IPC handle size

.. autoattribute:: cuda.cuda.CU_STREAM_LEGACY

    Legacy stream handle



    Stream handle that can be passed as a CUstream to use an implicit stream with legacy synchronization behavior.



    See details of the \link_sync_behavior

.. autoattribute:: cuda.cuda.CU_STREAM_PER_THREAD

    Per-thread stream handle



    Stream handle that can be passed as a CUstream to use an implicit stream with per-thread synchronization behavior.



    See details of the \link_sync_behavior

.. autoattribute:: cuda.cuda.CUDA_CB
.. autoattribute:: cuda.cuda.CU_MEMHOSTALLOC_PORTABLE

    If set, host memory is portable between CUDA contexts. Flag for :py:obj:`~.cuMemHostAlloc()`

.. autoattribute:: cuda.cuda.CU_MEMHOSTALLOC_DEVICEMAP

    If set, host memory is mapped into CUDA address space and :py:obj:`~.cuMemHostGetDevicePointer()` may be called on the host pointer. Flag for :py:obj:`~.cuMemHostAlloc()`

.. autoattribute:: cuda.cuda.CU_MEMHOSTALLOC_WRITECOMBINED

    If set, host memory is allocated as write-combined - fast to write, faster to DMA, slow to read except via SSE4 streaming load instruction (MOVNTDQA). Flag for :py:obj:`~.cuMemHostAlloc()`

.. autoattribute:: cuda.cuda.CU_MEMHOSTREGISTER_PORTABLE

    If set, host memory is portable between CUDA contexts. Flag for :py:obj:`~.cuMemHostRegister()`

.. autoattribute:: cuda.cuda.CU_MEMHOSTREGISTER_DEVICEMAP

    If set, host memory is mapped into CUDA address space and :py:obj:`~.cuMemHostGetDevicePointer()` may be called on the host pointer. Flag for :py:obj:`~.cuMemHostRegister()`

.. autoattribute:: cuda.cuda.CU_MEMHOSTREGISTER_IOMEMORY

    If set, the passed memory pointer is treated as pointing to some memory-mapped I/O space, e.g. belonging to a third-party PCIe device. On Windows the flag is a no-op. On Linux that memory is marked as non cache-coherent for the GPU and is expected to be physically contiguous. It may return :py:obj:`~.CUDA_ERROR_NOT_PERMITTED` if run as an unprivileged user, :py:obj:`~.CUDA_ERROR_NOT_SUPPORTED` on older Linux kernel versions. On all other platforms, it is not supported and :py:obj:`~.CUDA_ERROR_NOT_SUPPORTED` is returned. Flag for :py:obj:`~.cuMemHostRegister()`

.. autoattribute:: cuda.cuda.CU_MEMHOSTREGISTER_READ_ONLY

    If set, the passed memory pointer is treated as pointing to memory that is considered read-only by the device. On platforms without :py:obj:`~.CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES`, this flag is required in order to register memory mapped to the CPU as read-only. Support for the use of this flag can be queried from the device attribute :py:obj:`~.CU_DEVICE_ATTRIBUTE_READ_ONLY_HOST_REGISTER_SUPPORTED`. Using this flag with a current context associated with a device that does not have this attribute set will cause :py:obj:`~.cuMemHostRegister` to error with :py:obj:`~.CUDA_ERROR_NOT_SUPPORTED`.

.. autoattribute:: cuda.cuda.CU_ARRAY_SPARSE_PROPERTIES_SINGLE_MIPTAIL

    Indicates that the layered sparse CUDA array or CUDA mipmapped array has a single mip tail region for all layers

.. autoattribute:: cuda.cuda.CUDA_EXTERNAL_MEMORY_DEDICATED

    Indicates that the external memory object is a dedicated resource

.. autoattribute:: cuda.cuda.CUDA_EXTERNAL_SEMAPHORE_SIGNAL_SKIP_NVSCIBUF_MEMSYNC

    When the `flags` parameter of :py:obj:`~.CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS` contains this flag, it indicates that signaling an external semaphore object should skip performing appropriate memory synchronization operations over all the external memory objects that are imported as :py:obj:`~.CU_EXTERNAL_MEMORY_HANDLE_TYPE_NVSCIBUF`, which otherwise are performed by default to ensure data coherency with other importers of the same NvSciBuf memory objects.

.. autoattribute:: cuda.cuda.CUDA_EXTERNAL_SEMAPHORE_WAIT_SKIP_NVSCIBUF_MEMSYNC

    When the `flags` parameter of :py:obj:`~.CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS` contains this flag, it indicates that waiting on an external semaphore object should skip performing appropriate memory synchronization operations over all the external memory objects that are imported as :py:obj:`~.CU_EXTERNAL_MEMORY_HANDLE_TYPE_NVSCIBUF`, which otherwise are performed by default to ensure data coherency with other importers of the same NvSciBuf memory objects.

.. autoattribute:: cuda.cuda.CUDA_NVSCISYNC_ATTR_SIGNAL

    When `flags` of :py:obj:`~.cuDeviceGetNvSciSyncAttributes` is set to this, it indicates that application needs signaler specific NvSciSyncAttr to be filled by :py:obj:`~.cuDeviceGetNvSciSyncAttributes`.

.. autoattribute:: cuda.cuda.CUDA_NVSCISYNC_ATTR_WAIT

    When `flags` of :py:obj:`~.cuDeviceGetNvSciSyncAttributes` is set to this, it indicates that application needs waiter specific NvSciSyncAttr to be filled by :py:obj:`~.cuDeviceGetNvSciSyncAttributes`.

.. autoattribute:: cuda.cuda.CU_MEM_CREATE_USAGE_TILE_POOL

    This flag if set indicates that the memory will be used as a tile pool.

.. autoattribute:: cuda.cuda.CUDA_COOPERATIVE_LAUNCH_MULTI_DEVICE_NO_PRE_LAUNCH_SYNC

    If set, each kernel launched as part of :py:obj:`~.cuLaunchCooperativeKernelMultiDevice` only waits for prior work in the stream corresponding to that GPU to complete before the kernel begins execution.

.. autoattribute:: cuda.cuda.CUDA_COOPERATIVE_LAUNCH_MULTI_DEVICE_NO_POST_LAUNCH_SYNC

    If set, any subsequent work pushed in a stream that participated in a call to :py:obj:`~.cuLaunchCooperativeKernelMultiDevice` will only wait for the kernel launched on the GPU corresponding to that stream to complete before it begins execution.

.. autoattribute:: cuda.cuda.CUDA_ARRAY3D_LAYERED

    If set, the CUDA array is a collection of layers, where each layer is either a 1D or a 2D array and the Depth member of CUDA_ARRAY3D_DESCRIPTOR specifies the number of layers, not the depth of a 3D array.

.. autoattribute:: cuda.cuda.CUDA_ARRAY3D_2DARRAY

    Deprecated, use CUDA_ARRAY3D_LAYERED

.. autoattribute:: cuda.cuda.CUDA_ARRAY3D_SURFACE_LDST

    This flag must be set in order to bind a surface reference to the CUDA array

.. autoattribute:: cuda.cuda.CUDA_ARRAY3D_CUBEMAP

    If set, the CUDA array is a collection of six 2D arrays, representing faces of a cube. The width of such a CUDA array must be equal to its height, and Depth must be six. If :py:obj:`~.CUDA_ARRAY3D_LAYERED` flag is also set, then the CUDA array is a collection of cubemaps and Depth must be a multiple of six.

.. autoattribute:: cuda.cuda.CUDA_ARRAY3D_TEXTURE_GATHER

    This flag must be set in order to perform texture gather operations on a CUDA array.

.. autoattribute:: cuda.cuda.CUDA_ARRAY3D_DEPTH_TEXTURE

    This flag if set indicates that the CUDA array is a DEPTH_TEXTURE.

.. autoattribute:: cuda.cuda.CUDA_ARRAY3D_COLOR_ATTACHMENT

    This flag indicates that the CUDA array may be bound as a color target in an external graphics API

.. autoattribute:: cuda.cuda.CUDA_ARRAY3D_SPARSE

    This flag if set indicates that the CUDA array or CUDA mipmapped array is a sparse CUDA array or CUDA mipmapped array respectively

.. autoattribute:: cuda.cuda.CUDA_ARRAY3D_DEFERRED_MAPPING

    This flag if set indicates that the CUDA array or CUDA mipmapped array will allow deferred memory mapping

.. autoattribute:: cuda.cuda.CU_TRSA_OVERRIDE_FORMAT

    Override the texref format with a format inferred from the array. Flag for :py:obj:`~.cuTexRefSetArray()`

.. autoattribute:: cuda.cuda.CU_TRSF_READ_AS_INTEGER

    Read the texture as integers rather than promoting the values to floats in the range [0,1]. Flag for :py:obj:`~.cuTexRefSetFlags()` and :py:obj:`~.cuTexObjectCreate()`

.. autoattribute:: cuda.cuda.CU_TRSF_NORMALIZED_COORDINATES

    Use normalized texture coordinates in the range [0,1) instead of [0,dim). Flag for :py:obj:`~.cuTexRefSetFlags()` and :py:obj:`~.cuTexObjectCreate()`

.. autoattribute:: cuda.cuda.CU_TRSF_SRGB

    Perform sRGB->linear conversion during texture read. Flag for :py:obj:`~.cuTexRefSetFlags()` and :py:obj:`~.cuTexObjectCreate()`

.. autoattribute:: cuda.cuda.CU_TRSF_DISABLE_TRILINEAR_OPTIMIZATION

    Disable any trilinear filtering optimizations. Flag for :py:obj:`~.cuTexRefSetFlags()` and :py:obj:`~.cuTexObjectCreate()`

.. autoattribute:: cuda.cuda.CU_TRSF_SEAMLESS_CUBEMAP

    Enable seamless cube map filtering. Flag for :py:obj:`~.cuTexObjectCreate()`

.. autoattribute:: cuda.cuda.CU_LAUNCH_PARAM_END_AS_INT

    C++ compile time constant for CU_LAUNCH_PARAM_END

.. autoattribute:: cuda.cuda.CU_LAUNCH_PARAM_END

    End of array terminator for the `extra` parameter to :py:obj:`~.cuLaunchKernel`

.. autoattribute:: cuda.cuda.CU_LAUNCH_PARAM_BUFFER_POINTER_AS_INT

    C++ compile time constant for CU_LAUNCH_PARAM_BUFFER_POINTER

.. autoattribute:: cuda.cuda.CU_LAUNCH_PARAM_BUFFER_POINTER

    Indicator that the next value in the `extra` parameter to :py:obj:`~.cuLaunchKernel` will be a pointer to a buffer containing all kernel parameters used for launching kernel `f`. This buffer needs to honor all alignment/padding requirements of the individual parameters. If :py:obj:`~.CU_LAUNCH_PARAM_BUFFER_SIZE` is not also specified in the `extra` array, then :py:obj:`~.CU_LAUNCH_PARAM_BUFFER_POINTER` will have no effect.

.. autoattribute:: cuda.cuda.CU_LAUNCH_PARAM_BUFFER_SIZE_AS_INT

    C++ compile time constant for CU_LAUNCH_PARAM_BUFFER_SIZE

.. autoattribute:: cuda.cuda.CU_LAUNCH_PARAM_BUFFER_SIZE

    Indicator that the next value in the `extra` parameter to :py:obj:`~.cuLaunchKernel` will be a pointer to a size_t which contains the size of the buffer specified with :py:obj:`~.CU_LAUNCH_PARAM_BUFFER_POINTER`. It is required that :py:obj:`~.CU_LAUNCH_PARAM_BUFFER_POINTER` also be specified in the `extra` array if the value associated with :py:obj:`~.CU_LAUNCH_PARAM_BUFFER_SIZE` is not zero.

.. autoattribute:: cuda.cuda.CU_PARAM_TR_DEFAULT

    For texture references loaded into the module, use default texunit from texture reference.

.. autoattribute:: cuda.cuda.CU_DEVICE_CPU

    Device that represents the CPU

.. autoattribute:: cuda.cuda.CU_DEVICE_INVALID

    Device that represents an invalid device

.. autoattribute:: cuda.cuda.MAX_PLANES

    Maximum number of planes per frame

.. autoattribute:: cuda.cuda.CUDA_EGL_INFINITE_TIMEOUT

    Indicates that timeout for :py:obj:`~.cuEGLStreamConsumerAcquireFrame` is infinite.


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

Stream Memory Operations
------------------------

This section describes the stream memory operations of the low-level CUDA driver application programming interface.



There are two versions of these APIs, a legacy version and a newer V2 version.



V1:



The V1 API is disabled by default. Users are required to explicitly enable it, e.g. on Linux by passing the kernel module parameter shown below: modprobe nvidia NVreg_EnableStreamMemOPs=1 There is currently no way to enable these operations on other operating systems.



Users can programmatically query whether the device supports these operations with cuDeviceGetAttribute() and CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS.



Support for the CU_STREAM_WAIT_VALUE_NOR flag can be queried with CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR.



Support for the cuStreamWriteValue64() and cuStreamWaitValue64() functions, as well as for the CU_STREAM_MEM_OP_WAIT_VALUE_64 and CU_STREAM_MEM_OP_WRITE_VALUE_64 flags, can be queried with CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS.



Support for both CU_STREAM_WAIT_VALUE_FLUSH and CU_STREAM_MEM_OP_FLUSH_REMOTE_WRITES requires dedicated platform hardware features and can be queried with cuDeviceGetAttribute() and CU_DEVICE_ATTRIBUTE_CAN_FLUSH_REMOTE_WRITES.



V2:



The V2 APIs are available by default on all platforms.



Support for the CU_STREAM_WAIT_VALUE_NOR flag can be queried with CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR_V2.



Support for the cuStreamWriteValue64() and cuStreamWaitValue64() functions, as well as for the CU_STREAM_MEM_OP_WAIT_VALUE_64 and CU_STREAM_MEM_OP_WRITE_VALUE_64 flags, can be queried with CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS_V2.



Support for both CU_STREAM_WAIT_VALUE_FLUSH and CU_STREAM_MEM_OP_FLUSH_REMOTE_WRITES requires dedicated platform hardware features and can be queried with cuDeviceGetAttribute() and CU_DEVICE_ATTRIBUTE_CAN_FLUSH_REMOTE_WRITES.



V1 & V2:



Note that all memory pointers passed as parameters to these operations are device pointers. Where necessary a device pointer should be obtained, for example with cuMemHostGetDevicePointer().



None of the operations accepts pointers to managed memory buffers (cuMemAllocManaged).



Warning: Improper use of these APIs may deadlock the application. Synchronization ordering established through these APIs is not visible to CUDA. CUDA tasks that are (even indirectly) ordered by these APIs should also have that order expressed with CUDA-visible dependencies such as events. This ensures that the scheduler does not serialize them in an improper order. For more information, see the Stream Memory Operations section in the programming guide(https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html).

.. autofunction:: cuda.cuda.cuStreamWaitValue32
.. autofunction:: cuda.cuda.cuStreamWaitValue64
.. autofunction:: cuda.cuda.cuStreamWriteValue32
.. autofunction:: cuda.cuda.cuStreamWriteValue64
.. autofunction:: cuda.cuda.cuStreamBatchMemOp
.. autofunction:: cuda.cuda.cuStreamWaitValue32_v2
.. autofunction:: cuda.cuda.cuStreamWaitValue64_v2
.. autofunction:: cuda.cuda.cuStreamWriteValue32_v2
.. autofunction:: cuda.cuda.cuStreamWriteValue64_v2
.. autofunction:: cuda.cuda.cuStreamBatchMemOp_v2

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
.. autofunction:: cuda.cuda.cuGraphAddBatchMemOpNode
.. autofunction:: cuda.cuda.cuGraphBatchMemOpNodeGetParams
.. autofunction:: cuda.cuda.cuGraphBatchMemOpNodeSetParams
.. autofunction:: cuda.cuda.cuGraphExecBatchMemOpNodeSetParams
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

.. autoclass:: cuda.cuda.CUGLDeviceList

    .. autoattribute:: cuda.cuda.CUGLDeviceList.CU_GL_DEVICE_LIST_ALL


        The CUDA devices for all GPUs used by the current OpenGL context


    .. autoattribute:: cuda.cuda.CUGLDeviceList.CU_GL_DEVICE_LIST_CURRENT_FRAME


        The CUDA devices for the GPUs used by the current OpenGL context in its currently rendering frame


    .. autoattribute:: cuda.cuda.CUGLDeviceList.CU_GL_DEVICE_LIST_NEXT_FRAME


        The CUDA devices for the GPUs to be used by the current OpenGL context in the next frame

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
