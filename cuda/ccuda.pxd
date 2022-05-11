# Copyright 2021-2022 NVIDIA Corporation.  All rights reserved.
#
# Please refer to the NVIDIA end user license agreement (EULA) associated
# with this source code for terms and conditions that govern your use of
# this software. Any use, reproduction, disclosure, or distribution of
# this software and related documentation outside the terms of the EULA
# is strictly prohibited.
from libc.stdint cimport uint32_t, uint64_t


ctypedef uint32_t cuuint32_t

ctypedef uint64_t cuuint64_t

ctypedef unsigned long long CUdeviceptr_v2

ctypedef CUdeviceptr_v2 CUdeviceptr

ctypedef int CUdevice_v1

ctypedef CUdevice_v1 CUdevice

cdef extern from "":
    cdef struct CUctx_st:
        pass
ctypedef CUctx_st* CUcontext

cdef extern from "":
    cdef struct CUmod_st:
        pass
ctypedef CUmod_st* CUmodule

cdef extern from "":
    cdef struct CUfunc_st:
        pass
ctypedef CUfunc_st* CUfunction

cdef extern from "":
    cdef struct CUarray_st:
        pass
ctypedef CUarray_st* CUarray

cdef extern from "":
    cdef struct CUmipmappedArray_st:
        pass
ctypedef CUmipmappedArray_st* CUmipmappedArray

cdef extern from "":
    cdef struct CUtexref_st:
        pass
ctypedef CUtexref_st* CUtexref

cdef extern from "":
    cdef struct CUsurfref_st:
        pass
ctypedef CUsurfref_st* CUsurfref

cdef extern from "":
    cdef struct CUevent_st:
        pass
ctypedef CUevent_st* CUevent

cdef extern from "":
    cdef struct CUstream_st:
        pass
ctypedef CUstream_st* CUstream

cdef extern from "":
    cdef struct CUgraphicsResource_st:
        pass
ctypedef CUgraphicsResource_st* CUgraphicsResource

ctypedef unsigned long long CUtexObject_v1

ctypedef CUtexObject_v1 CUtexObject

ctypedef unsigned long long CUsurfObject_v1

ctypedef CUsurfObject_v1 CUsurfObject

cdef extern from "":
    cdef struct CUextMemory_st:
        pass
ctypedef CUextMemory_st* CUexternalMemory

cdef extern from "":
    cdef struct CUextSemaphore_st:
        pass
ctypedef CUextSemaphore_st* CUexternalSemaphore

cdef extern from "":
    cdef struct CUgraph_st:
        pass
ctypedef CUgraph_st* CUgraph

cdef extern from "":
    cdef struct CUgraphNode_st:
        pass
ctypedef CUgraphNode_st* CUgraphNode

cdef extern from "":
    cdef struct CUgraphExec_st:
        pass
ctypedef CUgraphExec_st* CUgraphExec

cdef extern from "":
    cdef struct CUmemPoolHandle_st:
        pass
ctypedef CUmemPoolHandle_st* CUmemoryPool

cdef extern from "":
    cdef struct CUuserObject_st:
        pass
ctypedef CUuserObject_st* CUuserObject

cdef struct CUuuid_st:
    char bytes[16]

ctypedef CUuuid_st CUuuid

cdef struct CUipcEventHandle_st:
    char reserved[64]

ctypedef CUipcEventHandle_st CUipcEventHandle_v1

ctypedef CUipcEventHandle_v1 CUipcEventHandle

cdef struct CUipcMemHandle_st:
    char reserved[64]

ctypedef CUipcMemHandle_st CUipcMemHandle_v1

ctypedef CUipcMemHandle_v1 CUipcMemHandle

cdef enum CUipcMem_flags_enum:
    CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS = 1

ctypedef CUipcMem_flags_enum CUipcMem_flags

cdef enum CUmemAttach_flags_enum:
    CU_MEM_ATTACH_GLOBAL = 1
    CU_MEM_ATTACH_HOST = 2
    CU_MEM_ATTACH_SINGLE = 4

ctypedef CUmemAttach_flags_enum CUmemAttach_flags

cdef enum CUctx_flags_enum:
    CU_CTX_SCHED_AUTO = 0
    CU_CTX_SCHED_SPIN = 1
    CU_CTX_SCHED_YIELD = 2
    CU_CTX_SCHED_BLOCKING_SYNC = 4
    CU_CTX_BLOCKING_SYNC = 4
    CU_CTX_SCHED_MASK = 7
    CU_CTX_MAP_HOST = 8
    CU_CTX_LMEM_RESIZE_TO_MAX = 16
    CU_CTX_FLAGS_MASK = 31

ctypedef CUctx_flags_enum CUctx_flags

cdef enum CUstream_flags_enum:
    CU_STREAM_DEFAULT = 0
    CU_STREAM_NON_BLOCKING = 1

ctypedef CUstream_flags_enum CUstream_flags

cdef enum CUevent_flags_enum:
    CU_EVENT_DEFAULT = 0
    CU_EVENT_BLOCKING_SYNC = 1
    CU_EVENT_DISABLE_TIMING = 2
    CU_EVENT_INTERPROCESS = 4

ctypedef CUevent_flags_enum CUevent_flags

cdef enum CUevent_record_flags_enum:
    CU_EVENT_RECORD_DEFAULT = 0
    CU_EVENT_RECORD_EXTERNAL = 1

ctypedef CUevent_record_flags_enum CUevent_record_flags

cdef enum CUevent_wait_flags_enum:
    CU_EVENT_WAIT_DEFAULT = 0
    CU_EVENT_WAIT_EXTERNAL = 1

ctypedef CUevent_wait_flags_enum CUevent_wait_flags

cdef enum CUstreamWaitValue_flags_enum:
    CU_STREAM_WAIT_VALUE_GEQ = 0
    CU_STREAM_WAIT_VALUE_EQ = 1
    CU_STREAM_WAIT_VALUE_AND = 2
    CU_STREAM_WAIT_VALUE_NOR = 3
    CU_STREAM_WAIT_VALUE_FLUSH = 1073741824

ctypedef CUstreamWaitValue_flags_enum CUstreamWaitValue_flags

cdef enum CUstreamWriteValue_flags_enum:
    CU_STREAM_WRITE_VALUE_DEFAULT = 0
    CU_STREAM_WRITE_VALUE_NO_MEMORY_BARRIER = 1

ctypedef CUstreamWriteValue_flags_enum CUstreamWriteValue_flags

cdef enum CUstreamBatchMemOpType_enum:
    CU_STREAM_MEM_OP_WAIT_VALUE_32 = 1
    CU_STREAM_MEM_OP_WRITE_VALUE_32 = 2
    CU_STREAM_MEM_OP_WAIT_VALUE_64 = 4
    CU_STREAM_MEM_OP_WRITE_VALUE_64 = 5
    CU_STREAM_MEM_OP_BARRIER = 6
    CU_STREAM_MEM_OP_FLUSH_REMOTE_WRITES = 3

ctypedef CUstreamBatchMemOpType_enum CUstreamBatchMemOpType

cdef enum CUstreamMemoryBarrier_flags_enum:
    CU_STREAM_MEMORY_BARRIER_TYPE_SYS = 0
    CU_STREAM_MEMORY_BARRIER_TYPE_GPU = 1

ctypedef CUstreamMemoryBarrier_flags_enum CUstreamMemoryBarrier_flags

cdef struct CUstreamMemOpWaitValueParams_st:
    CUstreamBatchMemOpType operation
    CUdeviceptr address
    cuuint64_t value64
    unsigned int flags
    CUdeviceptr alias

cdef struct CUstreamMemOpWriteValueParams_st:
    CUstreamBatchMemOpType operation
    CUdeviceptr address
    cuuint64_t value64
    unsigned int flags
    CUdeviceptr alias

cdef struct CUstreamMemOpFlushRemoteWritesParams_st:
    CUstreamBatchMemOpType operation
    unsigned int flags

cdef struct CUstreamMemOpMemoryBarrierParams_st:
    CUstreamBatchMemOpType operation
    unsigned int flags

cdef union CUstreamBatchMemOpParams_union:
    CUstreamBatchMemOpType operation
    CUstreamMemOpWaitValueParams_st waitValue
    CUstreamMemOpWriteValueParams_st writeValue
    CUstreamMemOpFlushRemoteWritesParams_st flushRemoteWrites
    CUstreamMemOpMemoryBarrierParams_st memoryBarrier
    cuuint64_t pad[6]

ctypedef CUstreamBatchMemOpParams_union CUstreamBatchMemOpParams_v1

ctypedef CUstreamBatchMemOpParams_v1 CUstreamBatchMemOpParams

cdef struct CUDA_BATCH_MEM_OP_NODE_PARAMS_st:
    CUcontext ctx
    unsigned int count
    CUstreamBatchMemOpParams* paramArray
    unsigned int flags

ctypedef CUDA_BATCH_MEM_OP_NODE_PARAMS_st CUDA_BATCH_MEM_OP_NODE_PARAMS

cdef enum CUoccupancy_flags_enum:
    CU_OCCUPANCY_DEFAULT = 0
    CU_OCCUPANCY_DISABLE_CACHING_OVERRIDE = 1

ctypedef CUoccupancy_flags_enum CUoccupancy_flags

cdef enum CUstreamUpdateCaptureDependencies_flags_enum:
    CU_STREAM_ADD_CAPTURE_DEPENDENCIES = 0
    CU_STREAM_SET_CAPTURE_DEPENDENCIES = 1

ctypedef CUstreamUpdateCaptureDependencies_flags_enum CUstreamUpdateCaptureDependencies_flags

cdef enum CUarray_format_enum:
    CU_AD_FORMAT_UNSIGNED_INT8 = 1
    CU_AD_FORMAT_UNSIGNED_INT16 = 2
    CU_AD_FORMAT_UNSIGNED_INT32 = 3
    CU_AD_FORMAT_SIGNED_INT8 = 8
    CU_AD_FORMAT_SIGNED_INT16 = 9
    CU_AD_FORMAT_SIGNED_INT32 = 10
    CU_AD_FORMAT_HALF = 16
    CU_AD_FORMAT_FLOAT = 32
    CU_AD_FORMAT_NV12 = 176
    CU_AD_FORMAT_UNORM_INT8X1 = 192
    CU_AD_FORMAT_UNORM_INT8X2 = 193
    CU_AD_FORMAT_UNORM_INT8X4 = 194
    CU_AD_FORMAT_UNORM_INT16X1 = 195
    CU_AD_FORMAT_UNORM_INT16X2 = 196
    CU_AD_FORMAT_UNORM_INT16X4 = 197
    CU_AD_FORMAT_SNORM_INT8X1 = 198
    CU_AD_FORMAT_SNORM_INT8X2 = 199
    CU_AD_FORMAT_SNORM_INT8X4 = 200
    CU_AD_FORMAT_SNORM_INT16X1 = 201
    CU_AD_FORMAT_SNORM_INT16X2 = 202
    CU_AD_FORMAT_SNORM_INT16X4 = 203
    CU_AD_FORMAT_BC1_UNORM = 145
    CU_AD_FORMAT_BC1_UNORM_SRGB = 146
    CU_AD_FORMAT_BC2_UNORM = 147
    CU_AD_FORMAT_BC2_UNORM_SRGB = 148
    CU_AD_FORMAT_BC3_UNORM = 149
    CU_AD_FORMAT_BC3_UNORM_SRGB = 150
    CU_AD_FORMAT_BC4_UNORM = 151
    CU_AD_FORMAT_BC4_SNORM = 152
    CU_AD_FORMAT_BC5_UNORM = 153
    CU_AD_FORMAT_BC5_SNORM = 154
    CU_AD_FORMAT_BC6H_UF16 = 155
    CU_AD_FORMAT_BC6H_SF16 = 156
    CU_AD_FORMAT_BC7_UNORM = 157
    CU_AD_FORMAT_BC7_UNORM_SRGB = 158

ctypedef CUarray_format_enum CUarray_format

cdef enum CUaddress_mode_enum:
    CU_TR_ADDRESS_MODE_WRAP = 0
    CU_TR_ADDRESS_MODE_CLAMP = 1
    CU_TR_ADDRESS_MODE_MIRROR = 2
    CU_TR_ADDRESS_MODE_BORDER = 3

ctypedef CUaddress_mode_enum CUaddress_mode

cdef enum CUfilter_mode_enum:
    CU_TR_FILTER_MODE_POINT = 0
    CU_TR_FILTER_MODE_LINEAR = 1

ctypedef CUfilter_mode_enum CUfilter_mode

cdef enum CUdevice_attribute_enum:
    CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 1
    CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X = 2
    CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y = 3
    CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z = 4
    CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X = 5
    CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y = 6
    CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z = 7
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK = 8
    CU_DEVICE_ATTRIBUTE_SHARED_MEMORY_PER_BLOCK = 8
    CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY = 9
    CU_DEVICE_ATTRIBUTE_WARP_SIZE = 10
    CU_DEVICE_ATTRIBUTE_MAX_PITCH = 11
    CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK = 12
    CU_DEVICE_ATTRIBUTE_REGISTERS_PER_BLOCK = 12
    CU_DEVICE_ATTRIBUTE_CLOCK_RATE = 13
    CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT = 14
    CU_DEVICE_ATTRIBUTE_GPU_OVERLAP = 15
    CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16
    CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT = 17
    CU_DEVICE_ATTRIBUTE_INTEGRATED = 18
    CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY = 19
    CU_DEVICE_ATTRIBUTE_COMPUTE_MODE = 20
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH = 21
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH = 22
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT = 23
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH = 24
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT = 25
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH = 26
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH = 27
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT = 28
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS = 29
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_WIDTH = 27
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_HEIGHT = 28
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_NUMSLICES = 29
    CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT = 30
    CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS = 31
    CU_DEVICE_ATTRIBUTE_ECC_ENABLED = 32
    CU_DEVICE_ATTRIBUTE_PCI_BUS_ID = 33
    CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID = 34
    CU_DEVICE_ATTRIBUTE_TCC_DRIVER = 35
    CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE = 36
    CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH = 37
    CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE = 38
    CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR = 39
    CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT = 40
    CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING = 41
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH = 42
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS = 43
    CU_DEVICE_ATTRIBUTE_CAN_TEX2D_GATHER = 44
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_WIDTH = 45
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_HEIGHT = 46
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH_ALTERNATE = 47
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT_ALTERNATE = 48
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH_ALTERNATE = 49
    CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID = 50
    CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT = 51
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_WIDTH = 52
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_WIDTH = 53
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_LAYERS = 54
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_WIDTH = 55
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_WIDTH = 56
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_HEIGHT = 57
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_WIDTH = 58
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_HEIGHT = 59
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_DEPTH = 60
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_WIDTH = 61
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_LAYERS = 62
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_WIDTH = 63
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_HEIGHT = 64
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_LAYERS = 65
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_WIDTH = 66
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_WIDTH = 67
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_LAYERS = 68
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LINEAR_WIDTH = 69
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_WIDTH = 70
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_HEIGHT = 71
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_PITCH = 72
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_WIDTH = 73
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_HEIGHT = 74
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH = 77
    CU_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED = 78
    CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED = 79
    CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED = 80
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR = 81
    CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR = 82
    CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY = 83
    CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD = 84
    CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID = 85
    CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED = 86
    CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO = 87
    CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS = 88
    CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS = 89
    CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED = 90
    CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM = 91
    CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS = 92
    CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS = 93
    CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR = 94
    CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH = 95
    CU_DEVICE_ATTRIBUTE_COOPERATIVE_MULTI_DEVICE_LAUNCH = 96
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN = 97
    CU_DEVICE_ATTRIBUTE_CAN_FLUSH_REMOTE_WRITES = 98
    CU_DEVICE_ATTRIBUTE_HOST_REGISTER_SUPPORTED = 99
    CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES = 100
    CU_DEVICE_ATTRIBUTE_DIRECT_MANAGED_MEM_ACCESS_FROM_HOST = 101
    CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED = 102
    CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED = 102
    CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED = 103
    CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_HANDLE_SUPPORTED = 104
    CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_KMT_HANDLE_SUPPORTED = 105
    CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR = 106
    CU_DEVICE_ATTRIBUTE_GENERIC_COMPRESSION_SUPPORTED = 107
    CU_DEVICE_ATTRIBUTE_MAX_PERSISTING_L2_CACHE_SIZE = 108
    CU_DEVICE_ATTRIBUTE_MAX_ACCESS_POLICY_WINDOW_SIZE = 109
    CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED = 110
    CU_DEVICE_ATTRIBUTE_RESERVED_SHARED_MEMORY_PER_BLOCK = 111
    CU_DEVICE_ATTRIBUTE_SPARSE_CUDA_ARRAY_SUPPORTED = 112
    CU_DEVICE_ATTRIBUTE_READ_ONLY_HOST_REGISTER_SUPPORTED = 113
    CU_DEVICE_ATTRIBUTE_TIMELINE_SEMAPHORE_INTEROP_SUPPORTED = 114
    CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED = 115
    CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED = 116
    CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_FLUSH_WRITES_OPTIONS = 117
    CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WRITES_ORDERING = 118
    CU_DEVICE_ATTRIBUTE_MEMPOOL_SUPPORTED_HANDLE_TYPES = 119
    CU_DEVICE_ATTRIBUTE_DEFERRED_MAPPING_CUDA_ARRAY_SUPPORTED = 121
    CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS_V2 = 122
    CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR_V2 = 123
    CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED = 124
    CU_DEVICE_ATTRIBUTE_MAX = 125

ctypedef CUdevice_attribute_enum CUdevice_attribute

cdef struct CUdevprop_st:
    int maxThreadsPerBlock
    int maxThreadsDim[3]
    int maxGridSize[3]
    int sharedMemPerBlock
    int totalConstantMemory
    int SIMDWidth
    int memPitch
    int regsPerBlock
    int clockRate
    int textureAlign

ctypedef CUdevprop_st CUdevprop_v1

ctypedef CUdevprop_v1 CUdevprop

cdef enum CUpointer_attribute_enum:
    CU_POINTER_ATTRIBUTE_CONTEXT = 1
    CU_POINTER_ATTRIBUTE_MEMORY_TYPE = 2
    CU_POINTER_ATTRIBUTE_DEVICE_POINTER = 3
    CU_POINTER_ATTRIBUTE_HOST_POINTER = 4
    CU_POINTER_ATTRIBUTE_P2P_TOKENS = 5
    CU_POINTER_ATTRIBUTE_SYNC_MEMOPS = 6
    CU_POINTER_ATTRIBUTE_BUFFER_ID = 7
    CU_POINTER_ATTRIBUTE_IS_MANAGED = 8
    CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL = 9
    CU_POINTER_ATTRIBUTE_IS_LEGACY_CUDA_IPC_CAPABLE = 10
    CU_POINTER_ATTRIBUTE_RANGE_START_ADDR = 11
    CU_POINTER_ATTRIBUTE_RANGE_SIZE = 12
    CU_POINTER_ATTRIBUTE_MAPPED = 13
    CU_POINTER_ATTRIBUTE_ALLOWED_HANDLE_TYPES = 14
    CU_POINTER_ATTRIBUTE_IS_GPU_DIRECT_RDMA_CAPABLE = 15
    CU_POINTER_ATTRIBUTE_ACCESS_FLAGS = 16
    CU_POINTER_ATTRIBUTE_MEMPOOL_HANDLE = 17
    CU_POINTER_ATTRIBUTE_MAPPING_SIZE = 18
    CU_POINTER_ATTRIBUTE_MAPPING_BASE_ADDR = 19
    CU_POINTER_ATTRIBUTE_MEMORY_BLOCK_ID = 20

ctypedef CUpointer_attribute_enum CUpointer_attribute

cdef enum CUfunction_attribute_enum:
    CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 0
    CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES = 1
    CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES = 2
    CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES = 3
    CU_FUNC_ATTRIBUTE_NUM_REGS = 4
    CU_FUNC_ATTRIBUTE_PTX_VERSION = 5
    CU_FUNC_ATTRIBUTE_BINARY_VERSION = 6
    CU_FUNC_ATTRIBUTE_CACHE_MODE_CA = 7
    CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = 8
    CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT = 9
    CU_FUNC_ATTRIBUTE_MAX = 10

ctypedef CUfunction_attribute_enum CUfunction_attribute

cdef enum CUfunc_cache_enum:
    CU_FUNC_CACHE_PREFER_NONE = 0
    CU_FUNC_CACHE_PREFER_SHARED = 1
    CU_FUNC_CACHE_PREFER_L1 = 2
    CU_FUNC_CACHE_PREFER_EQUAL = 3

ctypedef CUfunc_cache_enum CUfunc_cache

cdef enum CUsharedconfig_enum:
    CU_SHARED_MEM_CONFIG_DEFAULT_BANK_SIZE = 0
    CU_SHARED_MEM_CONFIG_FOUR_BYTE_BANK_SIZE = 1
    CU_SHARED_MEM_CONFIG_EIGHT_BYTE_BANK_SIZE = 2

ctypedef CUsharedconfig_enum CUsharedconfig

cdef enum CUshared_carveout_enum:
    CU_SHAREDMEM_CARVEOUT_DEFAULT = -1
    CU_SHAREDMEM_CARVEOUT_MAX_SHARED = 100
    CU_SHAREDMEM_CARVEOUT_MAX_L1 = 0

ctypedef CUshared_carveout_enum CUshared_carveout

cdef enum CUmemorytype_enum:
    CU_MEMORYTYPE_HOST = 1
    CU_MEMORYTYPE_DEVICE = 2
    CU_MEMORYTYPE_ARRAY = 3
    CU_MEMORYTYPE_UNIFIED = 4

ctypedef CUmemorytype_enum CUmemorytype

cdef enum CUcomputemode_enum:
    CU_COMPUTEMODE_DEFAULT = 0
    CU_COMPUTEMODE_PROHIBITED = 2
    CU_COMPUTEMODE_EXCLUSIVE_PROCESS = 3

ctypedef CUcomputemode_enum CUcomputemode

cdef enum CUmem_advise_enum:
    CU_MEM_ADVISE_SET_READ_MOSTLY = 1
    CU_MEM_ADVISE_UNSET_READ_MOSTLY = 2
    CU_MEM_ADVISE_SET_PREFERRED_LOCATION = 3
    CU_MEM_ADVISE_UNSET_PREFERRED_LOCATION = 4
    CU_MEM_ADVISE_SET_ACCESSED_BY = 5
    CU_MEM_ADVISE_UNSET_ACCESSED_BY = 6

ctypedef CUmem_advise_enum CUmem_advise

cdef enum CUmem_range_attribute_enum:
    CU_MEM_RANGE_ATTRIBUTE_READ_MOSTLY = 1
    CU_MEM_RANGE_ATTRIBUTE_PREFERRED_LOCATION = 2
    CU_MEM_RANGE_ATTRIBUTE_ACCESSED_BY = 3
    CU_MEM_RANGE_ATTRIBUTE_LAST_PREFETCH_LOCATION = 4

ctypedef CUmem_range_attribute_enum CUmem_range_attribute

cdef enum CUjit_option_enum:
    CU_JIT_MAX_REGISTERS = 0
    CU_JIT_THREADS_PER_BLOCK = 1
    CU_JIT_WALL_TIME = 2
    CU_JIT_INFO_LOG_BUFFER = 3
    CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES = 4
    CU_JIT_ERROR_LOG_BUFFER = 5
    CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES = 6
    CU_JIT_OPTIMIZATION_LEVEL = 7
    CU_JIT_TARGET_FROM_CUCONTEXT = 8
    CU_JIT_TARGET = 9
    CU_JIT_FALLBACK_STRATEGY = 10
    CU_JIT_GENERATE_DEBUG_INFO = 11
    CU_JIT_LOG_VERBOSE = 12
    CU_JIT_GENERATE_LINE_INFO = 13
    CU_JIT_CACHE_MODE = 14
    CU_JIT_NEW_SM3X_OPT = 15
    CU_JIT_FAST_COMPILE = 16
    CU_JIT_GLOBAL_SYMBOL_NAMES = 17
    CU_JIT_GLOBAL_SYMBOL_ADDRESSES = 18
    CU_JIT_GLOBAL_SYMBOL_COUNT = 19
    CU_JIT_LTO = 20
    CU_JIT_FTZ = 21
    CU_JIT_PREC_DIV = 22
    CU_JIT_PREC_SQRT = 23
    CU_JIT_FMA = 24
    CU_JIT_REFERENCED_KERNEL_NAMES = 25
    CU_JIT_REFERENCED_KERNEL_COUNT = 26
    CU_JIT_REFERENCED_VARIABLE_NAMES = 27
    CU_JIT_REFERENCED_VARIABLE_COUNT = 28
    CU_JIT_OPTIMIZE_UNUSED_DEVICE_VARIABLES = 29
    CU_JIT_NUM_OPTIONS = 30

ctypedef CUjit_option_enum CUjit_option

cdef enum CUjit_target_enum:
    CU_TARGET_COMPUTE_20 = 20
    CU_TARGET_COMPUTE_21 = 21
    CU_TARGET_COMPUTE_30 = 30
    CU_TARGET_COMPUTE_32 = 32
    CU_TARGET_COMPUTE_35 = 35
    CU_TARGET_COMPUTE_37 = 37
    CU_TARGET_COMPUTE_50 = 50
    CU_TARGET_COMPUTE_52 = 52
    CU_TARGET_COMPUTE_53 = 53
    CU_TARGET_COMPUTE_60 = 60
    CU_TARGET_COMPUTE_61 = 61
    CU_TARGET_COMPUTE_62 = 62
    CU_TARGET_COMPUTE_70 = 70
    CU_TARGET_COMPUTE_72 = 72
    CU_TARGET_COMPUTE_75 = 75
    CU_TARGET_COMPUTE_80 = 80
    CU_TARGET_COMPUTE_86 = 86
    CU_TARGET_COMPUTE_87 = 87

ctypedef CUjit_target_enum CUjit_target

cdef enum CUjit_fallback_enum:
    CU_PREFER_PTX = 0
    CU_PREFER_BINARY = 1

ctypedef CUjit_fallback_enum CUjit_fallback

cdef enum CUjit_cacheMode_enum:
    CU_JIT_CACHE_OPTION_NONE = 0
    CU_JIT_CACHE_OPTION_CG = 1
    CU_JIT_CACHE_OPTION_CA = 2

ctypedef CUjit_cacheMode_enum CUjit_cacheMode

cdef enum CUjitInputType_enum:
    CU_JIT_INPUT_CUBIN = 0
    CU_JIT_INPUT_PTX = 1
    CU_JIT_INPUT_FATBINARY = 2
    CU_JIT_INPUT_OBJECT = 3
    CU_JIT_INPUT_LIBRARY = 4
    CU_JIT_INPUT_NVVM = 5
    CU_JIT_NUM_INPUT_TYPES = 6

ctypedef CUjitInputType_enum CUjitInputType

cdef extern from "":
    cdef struct CUlinkState_st:
        pass
ctypedef CUlinkState_st* CUlinkState

cdef enum CUgraphicsRegisterFlags_enum:
    CU_GRAPHICS_REGISTER_FLAGS_NONE = 0
    CU_GRAPHICS_REGISTER_FLAGS_READ_ONLY = 1
    CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD = 2
    CU_GRAPHICS_REGISTER_FLAGS_SURFACE_LDST = 4
    CU_GRAPHICS_REGISTER_FLAGS_TEXTURE_GATHER = 8

ctypedef CUgraphicsRegisterFlags_enum CUgraphicsRegisterFlags

cdef enum CUgraphicsMapResourceFlags_enum:
    CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE = 0
    CU_GRAPHICS_MAP_RESOURCE_FLAGS_READ_ONLY = 1
    CU_GRAPHICS_MAP_RESOURCE_FLAGS_WRITE_DISCARD = 2

ctypedef CUgraphicsMapResourceFlags_enum CUgraphicsMapResourceFlags

cdef enum CUarray_cubemap_face_enum:
    CU_CUBEMAP_FACE_POSITIVE_X = 0
    CU_CUBEMAP_FACE_NEGATIVE_X = 1
    CU_CUBEMAP_FACE_POSITIVE_Y = 2
    CU_CUBEMAP_FACE_NEGATIVE_Y = 3
    CU_CUBEMAP_FACE_POSITIVE_Z = 4
    CU_CUBEMAP_FACE_NEGATIVE_Z = 5

ctypedef CUarray_cubemap_face_enum CUarray_cubemap_face

cdef enum CUlimit_enum:
    CU_LIMIT_STACK_SIZE = 0
    CU_LIMIT_PRINTF_FIFO_SIZE = 1
    CU_LIMIT_MALLOC_HEAP_SIZE = 2
    CU_LIMIT_DEV_RUNTIME_SYNC_DEPTH = 3
    CU_LIMIT_DEV_RUNTIME_PENDING_LAUNCH_COUNT = 4
    CU_LIMIT_MAX_L2_FETCH_GRANULARITY = 5
    CU_LIMIT_PERSISTING_L2_CACHE_SIZE = 6
    CU_LIMIT_MAX = 7

ctypedef CUlimit_enum CUlimit

cdef enum CUresourcetype_enum:
    CU_RESOURCE_TYPE_ARRAY = 0
    CU_RESOURCE_TYPE_MIPMAPPED_ARRAY = 1
    CU_RESOURCE_TYPE_LINEAR = 2
    CU_RESOURCE_TYPE_PITCH2D = 3

ctypedef CUresourcetype_enum CUresourcetype

ctypedef void (*CUhostFn)(void* userData)

cdef enum CUaccessProperty_enum:
    CU_ACCESS_PROPERTY_NORMAL = 0
    CU_ACCESS_PROPERTY_STREAMING = 1
    CU_ACCESS_PROPERTY_PERSISTING = 2

ctypedef CUaccessProperty_enum CUaccessProperty

cdef struct CUaccessPolicyWindow_st:
    void* base_ptr
    size_t num_bytes
    float hitRatio
    CUaccessProperty hitProp
    CUaccessProperty missProp

ctypedef CUaccessPolicyWindow_st CUaccessPolicyWindow_v1

ctypedef CUaccessPolicyWindow_v1 CUaccessPolicyWindow

cdef struct CUDA_KERNEL_NODE_PARAMS_st:
    CUfunction func
    unsigned int gridDimX
    unsigned int gridDimY
    unsigned int gridDimZ
    unsigned int blockDimX
    unsigned int blockDimY
    unsigned int blockDimZ
    unsigned int sharedMemBytes
    void** kernelParams
    void** extra

ctypedef CUDA_KERNEL_NODE_PARAMS_st CUDA_KERNEL_NODE_PARAMS_v1

ctypedef CUDA_KERNEL_NODE_PARAMS_v1 CUDA_KERNEL_NODE_PARAMS

cdef struct CUDA_MEMSET_NODE_PARAMS_st:
    CUdeviceptr dst
    size_t pitch
    unsigned int value
    unsigned int elementSize
    size_t width
    size_t height

ctypedef CUDA_MEMSET_NODE_PARAMS_st CUDA_MEMSET_NODE_PARAMS_v1

ctypedef CUDA_MEMSET_NODE_PARAMS_v1 CUDA_MEMSET_NODE_PARAMS

cdef struct CUDA_HOST_NODE_PARAMS_st:
    CUhostFn fn
    void* userData

ctypedef CUDA_HOST_NODE_PARAMS_st CUDA_HOST_NODE_PARAMS_v1

ctypedef CUDA_HOST_NODE_PARAMS_v1 CUDA_HOST_NODE_PARAMS

cdef enum CUgraphNodeType_enum:
    CU_GRAPH_NODE_TYPE_KERNEL = 0
    CU_GRAPH_NODE_TYPE_MEMCPY = 1
    CU_GRAPH_NODE_TYPE_MEMSET = 2
    CU_GRAPH_NODE_TYPE_HOST = 3
    CU_GRAPH_NODE_TYPE_GRAPH = 4
    CU_GRAPH_NODE_TYPE_EMPTY = 5
    CU_GRAPH_NODE_TYPE_WAIT_EVENT = 6
    CU_GRAPH_NODE_TYPE_EVENT_RECORD = 7
    CU_GRAPH_NODE_TYPE_EXT_SEMAS_SIGNAL = 8
    CU_GRAPH_NODE_TYPE_EXT_SEMAS_WAIT = 9
    CU_GRAPH_NODE_TYPE_MEM_ALLOC = 10
    CU_GRAPH_NODE_TYPE_MEM_FREE = 11
    CU_GRAPH_NODE_TYPE_BATCH_MEM_OP = 12

ctypedef CUgraphNodeType_enum CUgraphNodeType

cdef enum CUsynchronizationPolicy_enum:
    CU_SYNC_POLICY_AUTO = 1
    CU_SYNC_POLICY_SPIN = 2
    CU_SYNC_POLICY_YIELD = 3
    CU_SYNC_POLICY_BLOCKING_SYNC = 4

ctypedef CUsynchronizationPolicy_enum CUsynchronizationPolicy

cdef enum CUkernelNodeAttrID_enum:
    CU_KERNEL_NODE_ATTRIBUTE_ACCESS_POLICY_WINDOW = 1
    CU_KERNEL_NODE_ATTRIBUTE_COOPERATIVE = 2
    CU_KERNEL_NODE_ATTRIBUTE_PRIORITY = 8

ctypedef CUkernelNodeAttrID_enum CUkernelNodeAttrID

cdef union CUkernelNodeAttrValue_union:
    CUaccessPolicyWindow accessPolicyWindow
    int cooperative
    int priority

ctypedef CUkernelNodeAttrValue_union CUkernelNodeAttrValue_v1

ctypedef CUkernelNodeAttrValue_v1 CUkernelNodeAttrValue

cdef enum CUstreamCaptureStatus_enum:
    CU_STREAM_CAPTURE_STATUS_NONE = 0
    CU_STREAM_CAPTURE_STATUS_ACTIVE = 1
    CU_STREAM_CAPTURE_STATUS_INVALIDATED = 2

ctypedef CUstreamCaptureStatus_enum CUstreamCaptureStatus

cdef enum CUstreamCaptureMode_enum:
    CU_STREAM_CAPTURE_MODE_GLOBAL = 0
    CU_STREAM_CAPTURE_MODE_THREAD_LOCAL = 1
    CU_STREAM_CAPTURE_MODE_RELAXED = 2

ctypedef CUstreamCaptureMode_enum CUstreamCaptureMode

cdef enum CUstreamAttrID_enum:
    CU_STREAM_ATTRIBUTE_ACCESS_POLICY_WINDOW = 1
    CU_STREAM_ATTRIBUTE_SYNCHRONIZATION_POLICY = 3

ctypedef CUstreamAttrID_enum CUstreamAttrID

cdef union CUstreamAttrValue_union:
    CUaccessPolicyWindow accessPolicyWindow
    CUsynchronizationPolicy syncPolicy

ctypedef CUstreamAttrValue_union CUstreamAttrValue_v1

ctypedef CUstreamAttrValue_v1 CUstreamAttrValue

cdef enum CUdriverProcAddress_flags_enum:
    CU_GET_PROC_ADDRESS_DEFAULT = 0
    CU_GET_PROC_ADDRESS_LEGACY_STREAM = 1
    CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM = 2

ctypedef CUdriverProcAddress_flags_enum CUdriverProcAddress_flags

cdef enum CUexecAffinityType_enum:
    CU_EXEC_AFFINITY_TYPE_SM_COUNT = 0
    CU_EXEC_AFFINITY_TYPE_MAX = 1

ctypedef CUexecAffinityType_enum CUexecAffinityType

cdef struct CUexecAffinitySmCount_st:
    unsigned int val

ctypedef CUexecAffinitySmCount_st CUexecAffinitySmCount_v1

ctypedef CUexecAffinitySmCount_v1 CUexecAffinitySmCount

cdef union _CUexecAffinityParam_v1_CUexecAffinityParam_v1_CUexecAffinityParam_st_param_u:
    CUexecAffinitySmCount smCount

cdef struct CUexecAffinityParam_st:
    CUexecAffinityType type
    _CUexecAffinityParam_v1_CUexecAffinityParam_v1_CUexecAffinityParam_st_param_u param

ctypedef CUexecAffinityParam_st CUexecAffinityParam_v1

ctypedef CUexecAffinityParam_v1 CUexecAffinityParam

cdef enum cudaError_enum:
    CUDA_SUCCESS = 0
    CUDA_ERROR_INVALID_VALUE = 1
    CUDA_ERROR_OUT_OF_MEMORY = 2
    CUDA_ERROR_NOT_INITIALIZED = 3
    CUDA_ERROR_DEINITIALIZED = 4
    CUDA_ERROR_PROFILER_DISABLED = 5
    CUDA_ERROR_PROFILER_NOT_INITIALIZED = 6
    CUDA_ERROR_PROFILER_ALREADY_STARTED = 7
    CUDA_ERROR_PROFILER_ALREADY_STOPPED = 8
    CUDA_ERROR_STUB_LIBRARY = 34
    CUDA_ERROR_DEVICE_UNAVAILABLE = 46
    CUDA_ERROR_NO_DEVICE = 100
    CUDA_ERROR_INVALID_DEVICE = 101
    CUDA_ERROR_DEVICE_NOT_LICENSED = 102
    CUDA_ERROR_INVALID_IMAGE = 200
    CUDA_ERROR_INVALID_CONTEXT = 201
    CUDA_ERROR_CONTEXT_ALREADY_CURRENT = 202
    CUDA_ERROR_MAP_FAILED = 205
    CUDA_ERROR_UNMAP_FAILED = 206
    CUDA_ERROR_ARRAY_IS_MAPPED = 207
    CUDA_ERROR_ALREADY_MAPPED = 208
    CUDA_ERROR_NO_BINARY_FOR_GPU = 209
    CUDA_ERROR_ALREADY_ACQUIRED = 210
    CUDA_ERROR_NOT_MAPPED = 211
    CUDA_ERROR_NOT_MAPPED_AS_ARRAY = 212
    CUDA_ERROR_NOT_MAPPED_AS_POINTER = 213
    CUDA_ERROR_ECC_UNCORRECTABLE = 214
    CUDA_ERROR_UNSUPPORTED_LIMIT = 215
    CUDA_ERROR_CONTEXT_ALREADY_IN_USE = 216
    CUDA_ERROR_PEER_ACCESS_UNSUPPORTED = 217
    CUDA_ERROR_INVALID_PTX = 218
    CUDA_ERROR_INVALID_GRAPHICS_CONTEXT = 219
    CUDA_ERROR_NVLINK_UNCORRECTABLE = 220
    CUDA_ERROR_JIT_COMPILER_NOT_FOUND = 221
    CUDA_ERROR_UNSUPPORTED_PTX_VERSION = 222
    CUDA_ERROR_JIT_COMPILATION_DISABLED = 223
    CUDA_ERROR_UNSUPPORTED_EXEC_AFFINITY = 224
    CUDA_ERROR_INVALID_SOURCE = 300
    CUDA_ERROR_FILE_NOT_FOUND = 301
    CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND = 302
    CUDA_ERROR_SHARED_OBJECT_INIT_FAILED = 303
    CUDA_ERROR_OPERATING_SYSTEM = 304
    CUDA_ERROR_INVALID_HANDLE = 400
    CUDA_ERROR_ILLEGAL_STATE = 401
    CUDA_ERROR_NOT_FOUND = 500
    CUDA_ERROR_NOT_READY = 600
    CUDA_ERROR_ILLEGAL_ADDRESS = 700
    CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES = 701
    CUDA_ERROR_LAUNCH_TIMEOUT = 702
    CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING = 703
    CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED = 704
    CUDA_ERROR_PEER_ACCESS_NOT_ENABLED = 705
    CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE = 708
    CUDA_ERROR_CONTEXT_IS_DESTROYED = 709
    CUDA_ERROR_ASSERT = 710
    CUDA_ERROR_TOO_MANY_PEERS = 711
    CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED = 712
    CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED = 713
    CUDA_ERROR_HARDWARE_STACK_ERROR = 714
    CUDA_ERROR_ILLEGAL_INSTRUCTION = 715
    CUDA_ERROR_MISALIGNED_ADDRESS = 716
    CUDA_ERROR_INVALID_ADDRESS_SPACE = 717
    CUDA_ERROR_INVALID_PC = 718
    CUDA_ERROR_LAUNCH_FAILED = 719
    CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE = 720
    CUDA_ERROR_NOT_PERMITTED = 800
    CUDA_ERROR_NOT_SUPPORTED = 801
    CUDA_ERROR_SYSTEM_NOT_READY = 802
    CUDA_ERROR_SYSTEM_DRIVER_MISMATCH = 803
    CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE = 804
    CUDA_ERROR_MPS_CONNECTION_FAILED = 805
    CUDA_ERROR_MPS_RPC_FAILURE = 806
    CUDA_ERROR_MPS_SERVER_NOT_READY = 807
    CUDA_ERROR_MPS_MAX_CLIENTS_REACHED = 808
    CUDA_ERROR_MPS_MAX_CONNECTIONS_REACHED = 809
    CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED = 900
    CUDA_ERROR_STREAM_CAPTURE_INVALIDATED = 901
    CUDA_ERROR_STREAM_CAPTURE_MERGE = 902
    CUDA_ERROR_STREAM_CAPTURE_UNMATCHED = 903
    CUDA_ERROR_STREAM_CAPTURE_UNJOINED = 904
    CUDA_ERROR_STREAM_CAPTURE_ISOLATION = 905
    CUDA_ERROR_STREAM_CAPTURE_IMPLICIT = 906
    CUDA_ERROR_CAPTURED_EVENT = 907
    CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD = 908
    CUDA_ERROR_TIMEOUT = 909
    CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE = 910
    CUDA_ERROR_EXTERNAL_DEVICE = 911
    CUDA_ERROR_UNKNOWN = 999

ctypedef cudaError_enum CUresult

cdef enum CUdevice_P2PAttribute_enum:
    CU_DEVICE_P2P_ATTRIBUTE_PERFORMANCE_RANK = 1
    CU_DEVICE_P2P_ATTRIBUTE_ACCESS_SUPPORTED = 2
    CU_DEVICE_P2P_ATTRIBUTE_NATIVE_ATOMIC_SUPPORTED = 3
    CU_DEVICE_P2P_ATTRIBUTE_ACCESS_ACCESS_SUPPORTED = 4
    CU_DEVICE_P2P_ATTRIBUTE_CUDA_ARRAY_ACCESS_SUPPORTED = 4

ctypedef CUdevice_P2PAttribute_enum CUdevice_P2PAttribute

ctypedef void (*CUstreamCallback)(CUstream hStream, CUresult status, void* userData)

ctypedef size_t (*CUoccupancyB2DSize)(int blockSize)

cdef struct CUDA_MEMCPY2D_st:
    size_t srcXInBytes
    size_t srcY
    CUmemorytype srcMemoryType
    const void* srcHost
    CUdeviceptr srcDevice
    CUarray srcArray
    size_t srcPitch
    size_t dstXInBytes
    size_t dstY
    CUmemorytype dstMemoryType
    void* dstHost
    CUdeviceptr dstDevice
    CUarray dstArray
    size_t dstPitch
    size_t WidthInBytes
    size_t Height

ctypedef CUDA_MEMCPY2D_st CUDA_MEMCPY2D_v2

ctypedef CUDA_MEMCPY2D_v2 CUDA_MEMCPY2D

cdef struct CUDA_MEMCPY3D_st:
    size_t srcXInBytes
    size_t srcY
    size_t srcZ
    size_t srcLOD
    CUmemorytype srcMemoryType
    const void* srcHost
    CUdeviceptr srcDevice
    CUarray srcArray
    void* reserved0
    size_t srcPitch
    size_t srcHeight
    size_t dstXInBytes
    size_t dstY
    size_t dstZ
    size_t dstLOD
    CUmemorytype dstMemoryType
    void* dstHost
    CUdeviceptr dstDevice
    CUarray dstArray
    void* reserved1
    size_t dstPitch
    size_t dstHeight
    size_t WidthInBytes
    size_t Height
    size_t Depth

ctypedef CUDA_MEMCPY3D_st CUDA_MEMCPY3D_v2

ctypedef CUDA_MEMCPY3D_v2 CUDA_MEMCPY3D

cdef struct CUDA_MEMCPY3D_PEER_st:
    size_t srcXInBytes
    size_t srcY
    size_t srcZ
    size_t srcLOD
    CUmemorytype srcMemoryType
    const void* srcHost
    CUdeviceptr srcDevice
    CUarray srcArray
    CUcontext srcContext
    size_t srcPitch
    size_t srcHeight
    size_t dstXInBytes
    size_t dstY
    size_t dstZ
    size_t dstLOD
    CUmemorytype dstMemoryType
    void* dstHost
    CUdeviceptr dstDevice
    CUarray dstArray
    CUcontext dstContext
    size_t dstPitch
    size_t dstHeight
    size_t WidthInBytes
    size_t Height
    size_t Depth

ctypedef CUDA_MEMCPY3D_PEER_st CUDA_MEMCPY3D_PEER_v1

ctypedef CUDA_MEMCPY3D_PEER_v1 CUDA_MEMCPY3D_PEER

cdef struct CUDA_ARRAY_DESCRIPTOR_st:
    size_t Width
    size_t Height
    CUarray_format Format
    unsigned int NumChannels

ctypedef CUDA_ARRAY_DESCRIPTOR_st CUDA_ARRAY_DESCRIPTOR_v2

ctypedef CUDA_ARRAY_DESCRIPTOR_v2 CUDA_ARRAY_DESCRIPTOR

cdef struct CUDA_ARRAY3D_DESCRIPTOR_st:
    size_t Width
    size_t Height
    size_t Depth
    CUarray_format Format
    unsigned int NumChannels
    unsigned int Flags

ctypedef CUDA_ARRAY3D_DESCRIPTOR_st CUDA_ARRAY3D_DESCRIPTOR_v2

ctypedef CUDA_ARRAY3D_DESCRIPTOR_v2 CUDA_ARRAY3D_DESCRIPTOR

cdef struct _CUDA_ARRAY_SPARSE_PROPERTIES_v1_CUDA_ARRAY_SPARSE_PROPERTIES_v1_CUDA_ARRAY_SPARSE_PROPERTIES_st_tileExtent_s:
    unsigned int width
    unsigned int height
    unsigned int depth

cdef struct CUDA_ARRAY_SPARSE_PROPERTIES_st:
    _CUDA_ARRAY_SPARSE_PROPERTIES_v1_CUDA_ARRAY_SPARSE_PROPERTIES_v1_CUDA_ARRAY_SPARSE_PROPERTIES_st_tileExtent_s tileExtent
    unsigned int miptailFirstLevel
    unsigned long long miptailSize
    unsigned int flags
    unsigned int reserved[4]

ctypedef CUDA_ARRAY_SPARSE_PROPERTIES_st CUDA_ARRAY_SPARSE_PROPERTIES_v1

ctypedef CUDA_ARRAY_SPARSE_PROPERTIES_v1 CUDA_ARRAY_SPARSE_PROPERTIES

cdef struct CUDA_ARRAY_MEMORY_REQUIREMENTS_st:
    size_t size
    size_t alignment
    unsigned int reserved[4]

ctypedef CUDA_ARRAY_MEMORY_REQUIREMENTS_st CUDA_ARRAY_MEMORY_REQUIREMENTS_v1

ctypedef CUDA_ARRAY_MEMORY_REQUIREMENTS_v1 CUDA_ARRAY_MEMORY_REQUIREMENTS

cdef struct _CUDA_RESOURCE_DESC_v1_CUDA_RESOURCE_DESC_v1_CUDA_RESOURCE_DESC_st_res_res_array_s:
    CUarray hArray

cdef struct _CUDA_RESOURCE_DESC_v1_CUDA_RESOURCE_DESC_v1_CUDA_RESOURCE_DESC_st_res_res_mipmap_s:
    CUmipmappedArray hMipmappedArray

cdef struct _CUDA_RESOURCE_DESC_v1_CUDA_RESOURCE_DESC_v1_CUDA_RESOURCE_DESC_st_res_res_linear_s:
    CUdeviceptr devPtr
    CUarray_format format
    unsigned int numChannels
    size_t sizeInBytes

cdef struct _CUDA_RESOURCE_DESC_v1_CUDA_RESOURCE_DESC_v1_CUDA_RESOURCE_DESC_st_res_res_pitch2D_s:
    CUdeviceptr devPtr
    CUarray_format format
    unsigned int numChannels
    size_t width
    size_t height
    size_t pitchInBytes

cdef struct _CUDA_RESOURCE_DESC_v1_CUDA_RESOURCE_DESC_v1_CUDA_RESOURCE_DESC_st_res_res_reserved_s:
    int reserved[32]

cdef union _CUDA_RESOURCE_DESC_v1_CUDA_RESOURCE_DESC_v1_CUDA_RESOURCE_DESC_st_res_u:
    _CUDA_RESOURCE_DESC_v1_CUDA_RESOURCE_DESC_v1_CUDA_RESOURCE_DESC_st_res_res_array_s array
    _CUDA_RESOURCE_DESC_v1_CUDA_RESOURCE_DESC_v1_CUDA_RESOURCE_DESC_st_res_res_mipmap_s mipmap
    _CUDA_RESOURCE_DESC_v1_CUDA_RESOURCE_DESC_v1_CUDA_RESOURCE_DESC_st_res_res_linear_s linear
    _CUDA_RESOURCE_DESC_v1_CUDA_RESOURCE_DESC_v1_CUDA_RESOURCE_DESC_st_res_res_pitch2D_s pitch2D
    _CUDA_RESOURCE_DESC_v1_CUDA_RESOURCE_DESC_v1_CUDA_RESOURCE_DESC_st_res_res_reserved_s reserved

cdef struct CUDA_RESOURCE_DESC_st:
    CUresourcetype resType
    _CUDA_RESOURCE_DESC_v1_CUDA_RESOURCE_DESC_v1_CUDA_RESOURCE_DESC_st_res_u res
    unsigned int flags

ctypedef CUDA_RESOURCE_DESC_st CUDA_RESOURCE_DESC_v1

ctypedef CUDA_RESOURCE_DESC_v1 CUDA_RESOURCE_DESC

cdef struct CUDA_TEXTURE_DESC_st:
    CUaddress_mode addressMode[3]
    CUfilter_mode filterMode
    unsigned int flags
    unsigned int maxAnisotropy
    CUfilter_mode mipmapFilterMode
    float mipmapLevelBias
    float minMipmapLevelClamp
    float maxMipmapLevelClamp
    float borderColor[4]
    int reserved[12]

ctypedef CUDA_TEXTURE_DESC_st CUDA_TEXTURE_DESC_v1

ctypedef CUDA_TEXTURE_DESC_v1 CUDA_TEXTURE_DESC

cdef enum CUresourceViewFormat_enum:
    CU_RES_VIEW_FORMAT_NONE = 0
    CU_RES_VIEW_FORMAT_UINT_1X8 = 1
    CU_RES_VIEW_FORMAT_UINT_2X8 = 2
    CU_RES_VIEW_FORMAT_UINT_4X8 = 3
    CU_RES_VIEW_FORMAT_SINT_1X8 = 4
    CU_RES_VIEW_FORMAT_SINT_2X8 = 5
    CU_RES_VIEW_FORMAT_SINT_4X8 = 6
    CU_RES_VIEW_FORMAT_UINT_1X16 = 7
    CU_RES_VIEW_FORMAT_UINT_2X16 = 8
    CU_RES_VIEW_FORMAT_UINT_4X16 = 9
    CU_RES_VIEW_FORMAT_SINT_1X16 = 10
    CU_RES_VIEW_FORMAT_SINT_2X16 = 11
    CU_RES_VIEW_FORMAT_SINT_4X16 = 12
    CU_RES_VIEW_FORMAT_UINT_1X32 = 13
    CU_RES_VIEW_FORMAT_UINT_2X32 = 14
    CU_RES_VIEW_FORMAT_UINT_4X32 = 15
    CU_RES_VIEW_FORMAT_SINT_1X32 = 16
    CU_RES_VIEW_FORMAT_SINT_2X32 = 17
    CU_RES_VIEW_FORMAT_SINT_4X32 = 18
    CU_RES_VIEW_FORMAT_FLOAT_1X16 = 19
    CU_RES_VIEW_FORMAT_FLOAT_2X16 = 20
    CU_RES_VIEW_FORMAT_FLOAT_4X16 = 21
    CU_RES_VIEW_FORMAT_FLOAT_1X32 = 22
    CU_RES_VIEW_FORMAT_FLOAT_2X32 = 23
    CU_RES_VIEW_FORMAT_FLOAT_4X32 = 24
    CU_RES_VIEW_FORMAT_UNSIGNED_BC1 = 25
    CU_RES_VIEW_FORMAT_UNSIGNED_BC2 = 26
    CU_RES_VIEW_FORMAT_UNSIGNED_BC3 = 27
    CU_RES_VIEW_FORMAT_UNSIGNED_BC4 = 28
    CU_RES_VIEW_FORMAT_SIGNED_BC4 = 29
    CU_RES_VIEW_FORMAT_UNSIGNED_BC5 = 30
    CU_RES_VIEW_FORMAT_SIGNED_BC5 = 31
    CU_RES_VIEW_FORMAT_UNSIGNED_BC6H = 32
    CU_RES_VIEW_FORMAT_SIGNED_BC6H = 33
    CU_RES_VIEW_FORMAT_UNSIGNED_BC7 = 34

ctypedef CUresourceViewFormat_enum CUresourceViewFormat

cdef struct CUDA_RESOURCE_VIEW_DESC_st:
    CUresourceViewFormat format
    size_t width
    size_t height
    size_t depth
    unsigned int firstMipmapLevel
    unsigned int lastMipmapLevel
    unsigned int firstLayer
    unsigned int lastLayer
    unsigned int reserved[16]

ctypedef CUDA_RESOURCE_VIEW_DESC_st CUDA_RESOURCE_VIEW_DESC_v1

ctypedef CUDA_RESOURCE_VIEW_DESC_v1 CUDA_RESOURCE_VIEW_DESC

cdef struct CUDA_POINTER_ATTRIBUTE_P2P_TOKENS_st:
    unsigned long long p2pToken
    unsigned int vaSpaceToken

ctypedef CUDA_POINTER_ATTRIBUTE_P2P_TOKENS_st CUDA_POINTER_ATTRIBUTE_P2P_TOKENS_v1

ctypedef CUDA_POINTER_ATTRIBUTE_P2P_TOKENS_v1 CUDA_POINTER_ATTRIBUTE_P2P_TOKENS

cdef enum CUDA_POINTER_ATTRIBUTE_ACCESS_FLAGS_enum:
    CU_POINTER_ATTRIBUTE_ACCESS_FLAG_NONE = 0
    CU_POINTER_ATTRIBUTE_ACCESS_FLAG_READ = 1
    CU_POINTER_ATTRIBUTE_ACCESS_FLAG_READWRITE = 3

ctypedef CUDA_POINTER_ATTRIBUTE_ACCESS_FLAGS_enum CUDA_POINTER_ATTRIBUTE_ACCESS_FLAGS

cdef struct CUDA_LAUNCH_PARAMS_st:
    CUfunction function
    unsigned int gridDimX
    unsigned int gridDimY
    unsigned int gridDimZ
    unsigned int blockDimX
    unsigned int blockDimY
    unsigned int blockDimZ
    unsigned int sharedMemBytes
    CUstream hStream
    void** kernelParams

ctypedef CUDA_LAUNCH_PARAMS_st CUDA_LAUNCH_PARAMS_v1

ctypedef CUDA_LAUNCH_PARAMS_v1 CUDA_LAUNCH_PARAMS

cdef enum CUexternalMemoryHandleType_enum:
    CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD = 1
    CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32 = 2
    CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT = 3
    CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_HEAP = 4
    CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE = 5
    CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_RESOURCE = 6
    CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_RESOURCE_KMT = 7
    CU_EXTERNAL_MEMORY_HANDLE_TYPE_NVSCIBUF = 8

ctypedef CUexternalMemoryHandleType_enum CUexternalMemoryHandleType

cdef struct _CUDA_EXTERNAL_MEMORY_HANDLE_DESC_v1_CUDA_EXTERNAL_MEMORY_HANDLE_DESC_v1_CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st_handle_handle_win32_s:
    void* handle
    void* name

cdef union _CUDA_EXTERNAL_MEMORY_HANDLE_DESC_v1_CUDA_EXTERNAL_MEMORY_HANDLE_DESC_v1_CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st_handle_u:
    int fd
    _CUDA_EXTERNAL_MEMORY_HANDLE_DESC_v1_CUDA_EXTERNAL_MEMORY_HANDLE_DESC_v1_CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st_handle_handle_win32_s win32
    void* nvSciBufObject

cdef struct CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st:
    CUexternalMemoryHandleType type
    _CUDA_EXTERNAL_MEMORY_HANDLE_DESC_v1_CUDA_EXTERNAL_MEMORY_HANDLE_DESC_v1_CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st_handle_u handle
    unsigned long long size
    unsigned int flags
    unsigned int reserved[16]

ctypedef CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st CUDA_EXTERNAL_MEMORY_HANDLE_DESC_v1

ctypedef CUDA_EXTERNAL_MEMORY_HANDLE_DESC_v1 CUDA_EXTERNAL_MEMORY_HANDLE_DESC

cdef struct CUDA_EXTERNAL_MEMORY_BUFFER_DESC_st:
    unsigned long long offset
    unsigned long long size
    unsigned int flags
    unsigned int reserved[16]

ctypedef CUDA_EXTERNAL_MEMORY_BUFFER_DESC_st CUDA_EXTERNAL_MEMORY_BUFFER_DESC_v1

ctypedef CUDA_EXTERNAL_MEMORY_BUFFER_DESC_v1 CUDA_EXTERNAL_MEMORY_BUFFER_DESC

cdef struct CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC_st:
    unsigned long long offset
    CUDA_ARRAY3D_DESCRIPTOR arrayDesc
    unsigned int numLevels
    unsigned int reserved[16]

ctypedef CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC_st CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC_v1

ctypedef CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC_v1 CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC

cdef enum CUexternalSemaphoreHandleType_enum:
    CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD = 1
    CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32 = 2
    CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_KMT = 3
    CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D12_FENCE = 4
    CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D11_FENCE = 5
    CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_NVSCISYNC = 6
    CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D11_KEYED_MUTEX = 7
    CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D11_KEYED_MUTEX_KMT = 8
    CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_TIMELINE_SEMAPHORE_FD = 9
    CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_TIMELINE_SEMAPHORE_WIN32 = 10

ctypedef CUexternalSemaphoreHandleType_enum CUexternalSemaphoreHandleType

cdef struct _CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_v1_CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_v1_CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_st_handle_handle_win32_s:
    void* handle
    void* name

cdef union _CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_v1_CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_v1_CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_st_handle_u:
    int fd
    _CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_v1_CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_v1_CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_st_handle_handle_win32_s win32
    void* nvSciSyncObj

cdef struct CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_st:
    CUexternalSemaphoreHandleType type
    _CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_v1_CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_v1_CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_st_handle_u handle
    unsigned int flags
    unsigned int reserved[16]

ctypedef CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_st CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_v1

ctypedef CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_v1 CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC

cdef struct _CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_v1_CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_v1_CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st_params_params_fence_s:
    unsigned long long value

cdef union _CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_v1_CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_v1_CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st_params_params_nvSciSync_u:
    void* fence
    unsigned long long reserved

cdef struct _CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_v1_CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_v1_CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st_params_params_keyedMutex_s:
    unsigned long long key

cdef struct _CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_v1_CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_v1_CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st_params_s:
    _CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_v1_CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_v1_CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st_params_params_fence_s fence
    _CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_v1_CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_v1_CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st_params_params_nvSciSync_u nvSciSync
    _CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_v1_CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_v1_CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st_params_params_keyedMutex_s keyedMutex
    unsigned int reserved[12]

cdef struct CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st:
    _CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_v1_CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_v1_CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st_params_s params
    unsigned int flags
    unsigned int reserved[16]

ctypedef CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_v1

ctypedef CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_v1 CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS

cdef struct _CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_v1_CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_v1_CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st_params_params_fence_s:
    unsigned long long value

cdef union _CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_v1_CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_v1_CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st_params_params_nvSciSync_u:
    void* fence
    unsigned long long reserved

cdef struct _CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_v1_CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_v1_CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st_params_params_keyedMutex_s:
    unsigned long long key
    unsigned int timeoutMs

cdef struct _CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_v1_CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_v1_CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st_params_s:
    _CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_v1_CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_v1_CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st_params_params_fence_s fence
    _CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_v1_CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_v1_CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st_params_params_nvSciSync_u nvSciSync
    _CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_v1_CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_v1_CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st_params_params_keyedMutex_s keyedMutex
    unsigned int reserved[10]

cdef struct CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st:
    _CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_v1_CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_v1_CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st_params_s params
    unsigned int flags
    unsigned int reserved[16]

ctypedef CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_v1

ctypedef CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_v1 CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS

cdef struct CUDA_EXT_SEM_SIGNAL_NODE_PARAMS_st:
    CUexternalSemaphore* extSemArray
    const CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS* paramsArray
    unsigned int numExtSems

ctypedef CUDA_EXT_SEM_SIGNAL_NODE_PARAMS_st CUDA_EXT_SEM_SIGNAL_NODE_PARAMS_v1

ctypedef CUDA_EXT_SEM_SIGNAL_NODE_PARAMS_v1 CUDA_EXT_SEM_SIGNAL_NODE_PARAMS

cdef struct CUDA_EXT_SEM_WAIT_NODE_PARAMS_st:
    CUexternalSemaphore* extSemArray
    const CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS* paramsArray
    unsigned int numExtSems

ctypedef CUDA_EXT_SEM_WAIT_NODE_PARAMS_st CUDA_EXT_SEM_WAIT_NODE_PARAMS_v1

ctypedef CUDA_EXT_SEM_WAIT_NODE_PARAMS_v1 CUDA_EXT_SEM_WAIT_NODE_PARAMS

ctypedef unsigned long long CUmemGenericAllocationHandle_v1

ctypedef CUmemGenericAllocationHandle_v1 CUmemGenericAllocationHandle

cdef enum CUmemAllocationHandleType_enum:
    CU_MEM_HANDLE_TYPE_NONE = 0
    CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR = 1
    CU_MEM_HANDLE_TYPE_WIN32 = 2
    CU_MEM_HANDLE_TYPE_WIN32_KMT = 4
    CU_MEM_HANDLE_TYPE_MAX = 2147483647

ctypedef CUmemAllocationHandleType_enum CUmemAllocationHandleType

cdef enum CUmemAccess_flags_enum:
    CU_MEM_ACCESS_FLAGS_PROT_NONE = 0
    CU_MEM_ACCESS_FLAGS_PROT_READ = 1
    CU_MEM_ACCESS_FLAGS_PROT_READWRITE = 3
    CU_MEM_ACCESS_FLAGS_PROT_MAX = 2147483647

ctypedef CUmemAccess_flags_enum CUmemAccess_flags

cdef enum CUmemLocationType_enum:
    CU_MEM_LOCATION_TYPE_INVALID = 0
    CU_MEM_LOCATION_TYPE_DEVICE = 1
    CU_MEM_LOCATION_TYPE_MAX = 2147483647

ctypedef CUmemLocationType_enum CUmemLocationType

cdef enum CUmemAllocationType_enum:
    CU_MEM_ALLOCATION_TYPE_INVALID = 0
    CU_MEM_ALLOCATION_TYPE_PINNED = 1
    CU_MEM_ALLOCATION_TYPE_MAX = 2147483647

ctypedef CUmemAllocationType_enum CUmemAllocationType

cdef enum CUmemAllocationGranularity_flags_enum:
    CU_MEM_ALLOC_GRANULARITY_MINIMUM = 0
    CU_MEM_ALLOC_GRANULARITY_RECOMMENDED = 1

ctypedef CUmemAllocationGranularity_flags_enum CUmemAllocationGranularity_flags

cdef enum CUmemRangeHandleType_enum:
    CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD = 1
    CU_MEM_RANGE_HANDLE_TYPE_MAX = 2147483647

ctypedef CUmemRangeHandleType_enum CUmemRangeHandleType

cdef enum CUarraySparseSubresourceType_enum:
    CU_ARRAY_SPARSE_SUBRESOURCE_TYPE_SPARSE_LEVEL = 0
    CU_ARRAY_SPARSE_SUBRESOURCE_TYPE_MIPTAIL = 1

ctypedef CUarraySparseSubresourceType_enum CUarraySparseSubresourceType

cdef enum CUmemOperationType_enum:
    CU_MEM_OPERATION_TYPE_MAP = 1
    CU_MEM_OPERATION_TYPE_UNMAP = 2

ctypedef CUmemOperationType_enum CUmemOperationType

cdef enum CUmemHandleType_enum:
    CU_MEM_HANDLE_TYPE_GENERIC = 0

ctypedef CUmemHandleType_enum CUmemHandleType

cdef union _CUarrayMapInfo_v1_CUarrayMapInfo_v1_CUarrayMapInfo_st_resource_u:
    CUmipmappedArray mipmap
    CUarray array

cdef struct _CUarrayMapInfo_v1_CUarrayMapInfo_v1_CUarrayMapInfo_st_subresource_subresource_sparseLevel_s:
    unsigned int level
    unsigned int layer
    unsigned int offsetX
    unsigned int offsetY
    unsigned int offsetZ
    unsigned int extentWidth
    unsigned int extentHeight
    unsigned int extentDepth

cdef struct _CUarrayMapInfo_v1_CUarrayMapInfo_v1_CUarrayMapInfo_st_subresource_subresource_miptail_s:
    unsigned int layer
    unsigned long long offset
    unsigned long long size

cdef union _CUarrayMapInfo_v1_CUarrayMapInfo_v1_CUarrayMapInfo_st_subresource_u:
    _CUarrayMapInfo_v1_CUarrayMapInfo_v1_CUarrayMapInfo_st_subresource_subresource_sparseLevel_s sparseLevel
    _CUarrayMapInfo_v1_CUarrayMapInfo_v1_CUarrayMapInfo_st_subresource_subresource_miptail_s miptail

cdef union _CUarrayMapInfo_v1_CUarrayMapInfo_v1_CUarrayMapInfo_st_memHandle_u:
    CUmemGenericAllocationHandle memHandle

cdef struct CUarrayMapInfo_st:
    CUresourcetype resourceType
    _CUarrayMapInfo_v1_CUarrayMapInfo_v1_CUarrayMapInfo_st_resource_u resource
    CUarraySparseSubresourceType subresourceType
    _CUarrayMapInfo_v1_CUarrayMapInfo_v1_CUarrayMapInfo_st_subresource_u subresource
    CUmemOperationType memOperationType
    CUmemHandleType memHandleType
    _CUarrayMapInfo_v1_CUarrayMapInfo_v1_CUarrayMapInfo_st_memHandle_u memHandle
    unsigned long long offset
    unsigned int deviceBitMask
    unsigned int flags
    unsigned int reserved[2]

ctypedef CUarrayMapInfo_st CUarrayMapInfo_v1

ctypedef CUarrayMapInfo_v1 CUarrayMapInfo

cdef struct CUmemLocation_st:
    CUmemLocationType type
    int id

ctypedef CUmemLocation_st CUmemLocation_v1

ctypedef CUmemLocation_v1 CUmemLocation

cdef enum CUmemAllocationCompType_enum:
    CU_MEM_ALLOCATION_COMP_NONE = 0
    CU_MEM_ALLOCATION_COMP_GENERIC = 1

ctypedef CUmemAllocationCompType_enum CUmemAllocationCompType

cdef struct _CUmemAllocationProp_v1_CUmemAllocationProp_v1_CUmemAllocationProp_st_allocFlags_s:
    unsigned char compressionType
    unsigned char gpuDirectRDMACapable
    unsigned short usage
    unsigned char reserved[4]

cdef struct CUmemAllocationProp_st:
    CUmemAllocationType type
    CUmemAllocationHandleType requestedHandleTypes
    CUmemLocation location
    void* win32HandleMetaData
    _CUmemAllocationProp_v1_CUmemAllocationProp_v1_CUmemAllocationProp_st_allocFlags_s allocFlags

ctypedef CUmemAllocationProp_st CUmemAllocationProp_v1

ctypedef CUmemAllocationProp_v1 CUmemAllocationProp

cdef struct CUmemAccessDesc_st:
    CUmemLocation location
    CUmemAccess_flags flags

ctypedef CUmemAccessDesc_st CUmemAccessDesc_v1

ctypedef CUmemAccessDesc_v1 CUmemAccessDesc

cdef enum CUgraphExecUpdateResult_enum:
    CU_GRAPH_EXEC_UPDATE_SUCCESS = 0
    CU_GRAPH_EXEC_UPDATE_ERROR = 1
    CU_GRAPH_EXEC_UPDATE_ERROR_TOPOLOGY_CHANGED = 2
    CU_GRAPH_EXEC_UPDATE_ERROR_NODE_TYPE_CHANGED = 3
    CU_GRAPH_EXEC_UPDATE_ERROR_FUNCTION_CHANGED = 4
    CU_GRAPH_EXEC_UPDATE_ERROR_PARAMETERS_CHANGED = 5
    CU_GRAPH_EXEC_UPDATE_ERROR_NOT_SUPPORTED = 6
    CU_GRAPH_EXEC_UPDATE_ERROR_UNSUPPORTED_FUNCTION_CHANGE = 7
    CU_GRAPH_EXEC_UPDATE_ERROR_ATTRIBUTES_CHANGED = 8

ctypedef CUgraphExecUpdateResult_enum CUgraphExecUpdateResult

cdef enum CUmemPool_attribute_enum:
    CU_MEMPOOL_ATTR_REUSE_FOLLOW_EVENT_DEPENDENCIES = 1
    CU_MEMPOOL_ATTR_REUSE_ALLOW_OPPORTUNISTIC = 2
    CU_MEMPOOL_ATTR_REUSE_ALLOW_INTERNAL_DEPENDENCIES = 3
    CU_MEMPOOL_ATTR_RELEASE_THRESHOLD = 4
    CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT = 5
    CU_MEMPOOL_ATTR_RESERVED_MEM_HIGH = 6
    CU_MEMPOOL_ATTR_USED_MEM_CURRENT = 7
    CU_MEMPOOL_ATTR_USED_MEM_HIGH = 8

ctypedef CUmemPool_attribute_enum CUmemPool_attribute

cdef struct CUmemPoolProps_st:
    CUmemAllocationType allocType
    CUmemAllocationHandleType handleTypes
    CUmemLocation location
    void* win32SecurityAttributes
    unsigned char reserved[64]

ctypedef CUmemPoolProps_st CUmemPoolProps_v1

ctypedef CUmemPoolProps_v1 CUmemPoolProps

cdef struct CUmemPoolPtrExportData_st:
    unsigned char reserved[64]

ctypedef CUmemPoolPtrExportData_st CUmemPoolPtrExportData_v1

ctypedef CUmemPoolPtrExportData_v1 CUmemPoolPtrExportData

cdef struct CUDA_MEM_ALLOC_NODE_PARAMS_st:
    CUmemPoolProps poolProps
    const CUmemAccessDesc* accessDescs
    size_t accessDescCount
    size_t bytesize
    CUdeviceptr dptr

ctypedef CUDA_MEM_ALLOC_NODE_PARAMS_st CUDA_MEM_ALLOC_NODE_PARAMS

cdef enum CUgraphMem_attribute_enum:
    CU_GRAPH_MEM_ATTR_USED_MEM_CURRENT = 0
    CU_GRAPH_MEM_ATTR_USED_MEM_HIGH = 1
    CU_GRAPH_MEM_ATTR_RESERVED_MEM_CURRENT = 2
    CU_GRAPH_MEM_ATTR_RESERVED_MEM_HIGH = 3

ctypedef CUgraphMem_attribute_enum CUgraphMem_attribute

cdef enum CUflushGPUDirectRDMAWritesOptions_enum:
    CU_FLUSH_GPU_DIRECT_RDMA_WRITES_OPTION_HOST = 1
    CU_FLUSH_GPU_DIRECT_RDMA_WRITES_OPTION_MEMOPS = 2

ctypedef CUflushGPUDirectRDMAWritesOptions_enum CUflushGPUDirectRDMAWritesOptions

cdef enum CUGPUDirectRDMAWritesOrdering_enum:
    CU_GPU_DIRECT_RDMA_WRITES_ORDERING_NONE = 0
    CU_GPU_DIRECT_RDMA_WRITES_ORDERING_OWNER = 100
    CU_GPU_DIRECT_RDMA_WRITES_ORDERING_ALL_DEVICES = 200

ctypedef CUGPUDirectRDMAWritesOrdering_enum CUGPUDirectRDMAWritesOrdering

cdef enum CUflushGPUDirectRDMAWritesScope_enum:
    CU_FLUSH_GPU_DIRECT_RDMA_WRITES_TO_OWNER = 100
    CU_FLUSH_GPU_DIRECT_RDMA_WRITES_TO_ALL_DEVICES = 200

ctypedef CUflushGPUDirectRDMAWritesScope_enum CUflushGPUDirectRDMAWritesScope

cdef enum CUflushGPUDirectRDMAWritesTarget_enum:
    CU_FLUSH_GPU_DIRECT_RDMA_WRITES_TARGET_CURRENT_CTX = 0

ctypedef CUflushGPUDirectRDMAWritesTarget_enum CUflushGPUDirectRDMAWritesTarget

cdef enum CUgraphDebugDot_flags_enum:
    CU_GRAPH_DEBUG_DOT_FLAGS_VERBOSE = 1
    CU_GRAPH_DEBUG_DOT_FLAGS_RUNTIME_TYPES = 2
    CU_GRAPH_DEBUG_DOT_FLAGS_KERNEL_NODE_PARAMS = 4
    CU_GRAPH_DEBUG_DOT_FLAGS_MEMCPY_NODE_PARAMS = 8
    CU_GRAPH_DEBUG_DOT_FLAGS_MEMSET_NODE_PARAMS = 16
    CU_GRAPH_DEBUG_DOT_FLAGS_HOST_NODE_PARAMS = 32
    CU_GRAPH_DEBUG_DOT_FLAGS_EVENT_NODE_PARAMS = 64
    CU_GRAPH_DEBUG_DOT_FLAGS_EXT_SEMAS_SIGNAL_NODE_PARAMS = 128
    CU_GRAPH_DEBUG_DOT_FLAGS_EXT_SEMAS_WAIT_NODE_PARAMS = 256
    CU_GRAPH_DEBUG_DOT_FLAGS_KERNEL_NODE_ATTRIBUTES = 512
    CU_GRAPH_DEBUG_DOT_FLAGS_HANDLES = 1024
    CU_GRAPH_DEBUG_DOT_FLAGS_MEM_ALLOC_NODE_PARAMS = 2048
    CU_GRAPH_DEBUG_DOT_FLAGS_MEM_FREE_NODE_PARAMS = 4096
    CU_GRAPH_DEBUG_DOT_FLAGS_BATCH_MEM_OP_NODE_PARAMS = 8192

ctypedef CUgraphDebugDot_flags_enum CUgraphDebugDot_flags

cdef enum CUuserObject_flags_enum:
    CU_USER_OBJECT_NO_DESTRUCTOR_SYNC = 1

ctypedef CUuserObject_flags_enum CUuserObject_flags

cdef enum CUuserObjectRetain_flags_enum:
    CU_GRAPH_USER_OBJECT_MOVE = 1

ctypedef CUuserObjectRetain_flags_enum CUuserObjectRetain_flags

cdef enum CUgraphInstantiate_flags_enum:
    CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH = 1
    CUDA_GRAPH_INSTANTIATE_FLAG_USE_NODE_PRIORITY = 8

ctypedef CUgraphInstantiate_flags_enum CUgraphInstantiate_flags

cdef CUresult cuGetErrorString(CUresult error, const char** pStr) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuGetErrorName(CUresult error, const char** pStr) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuInit(unsigned int Flags) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuDriverGetVersion(int* driverVersion) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuDeviceGet(CUdevice* device, int ordinal) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuDeviceGetCount(int* count) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuDeviceGetName(char* name, int length, CUdevice dev) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuDeviceGetUuid(CUuuid* uuid, CUdevice dev) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuDeviceGetUuid_v2(CUuuid* uuid, CUdevice dev) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuDeviceGetLuid(char* luid, unsigned int* deviceNodeMask, CUdevice dev) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuDeviceTotalMem(size_t* numbytes, CUdevice dev) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuDeviceGetTexture1DLinearMaxWidth(size_t* maxWidthInElements, CUarray_format pformat, unsigned numChannels, CUdevice dev) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuDeviceGetAttribute(int* pi, CUdevice_attribute attrib, CUdevice dev) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuDeviceGetNvSciSyncAttributes(void* nvSciSyncAttrList, CUdevice dev, int flags) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuDeviceSetMemPool(CUdevice dev, CUmemoryPool pool) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuDeviceGetMemPool(CUmemoryPool* pool, CUdevice dev) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuDeviceGetDefaultMemPool(CUmemoryPool* pool_out, CUdevice dev) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuFlushGPUDirectRDMAWrites(CUflushGPUDirectRDMAWritesTarget target, CUflushGPUDirectRDMAWritesScope scope) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuDeviceGetProperties(CUdevprop* prop, CUdevice dev) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuDeviceComputeCapability(int* major, int* minor, CUdevice dev) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuDevicePrimaryCtxRetain(CUcontext* pctx, CUdevice dev) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuDevicePrimaryCtxRelease(CUdevice dev) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuDevicePrimaryCtxSetFlags(CUdevice dev, unsigned int flags) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuDevicePrimaryCtxGetState(CUdevice dev, unsigned int* flags, int* active) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuDevicePrimaryCtxReset(CUdevice dev) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuDeviceGetExecAffinitySupport(int* pi, CUexecAffinityType typename, CUdevice dev) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuCtxCreate(CUcontext* pctx, unsigned int flags, CUdevice dev) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuCtxCreate_v3(CUcontext* pctx, CUexecAffinityParam* paramsArray, int numParams, unsigned int flags, CUdevice dev) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuCtxDestroy(CUcontext ctx) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuCtxPushCurrent(CUcontext ctx) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuCtxPopCurrent(CUcontext* pctx) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuCtxSetCurrent(CUcontext ctx) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuCtxGetCurrent(CUcontext* pctx) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuCtxGetDevice(CUdevice* device) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuCtxGetFlags(unsigned int* flags) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuCtxSynchronize() nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuCtxSetLimit(CUlimit limit, size_t value) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuCtxGetLimit(size_t* pvalue, CUlimit limit) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuCtxGetCacheConfig(CUfunc_cache* pconfig) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuCtxSetCacheConfig(CUfunc_cache config) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuCtxGetSharedMemConfig(CUsharedconfig* pConfig) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuCtxSetSharedMemConfig(CUsharedconfig config) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuCtxGetApiVersion(CUcontext ctx, unsigned int* version) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuCtxGetStreamPriorityRange(int* leastPriority, int* greatestPriority) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuCtxResetPersistingL2Cache() nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuCtxGetExecAffinity(CUexecAffinityParam* pExecAffinity, CUexecAffinityType typename) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuCtxAttach(CUcontext* pctx, unsigned int flags) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuCtxDetach(CUcontext ctx) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuModuleLoad(CUmodule* module, const char* fname) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuModuleLoadData(CUmodule* module, const void* image) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuModuleLoadDataEx(CUmodule* module, const void* image, unsigned int numOptions, CUjit_option* options, void** optionValues) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuModuleLoadFatBinary(CUmodule* module, const void* fatCubin) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuModuleUnload(CUmodule hmod) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuModuleGetFunction(CUfunction* hfunc, CUmodule hmod, const char* name) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuModuleGetGlobal(CUdeviceptr* dptr, size_t* numbytes, CUmodule hmod, const char* name) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuModuleGetTexRef(CUtexref* pTexRef, CUmodule hmod, const char* name) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuModuleGetSurfRef(CUsurfref* pSurfRef, CUmodule hmod, const char* name) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuLinkCreate(unsigned int numOptions, CUjit_option* options, void** optionValues, CUlinkState* stateOut) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuLinkAddData(CUlinkState state, CUjitInputType typename, void* data, size_t size, const char* name, unsigned int numOptions, CUjit_option* options, void** optionValues) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuLinkAddFile(CUlinkState state, CUjitInputType typename, const char* path, unsigned int numOptions, CUjit_option* options, void** optionValues) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuLinkComplete(CUlinkState state, void** cubinOut, size_t* sizeOut) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuLinkDestroy(CUlinkState state) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuMemGetInfo(size_t* free, size_t* total) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuMemAlloc(CUdeviceptr* dptr, size_t bytesize) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuMemAllocPitch(CUdeviceptr* dptr, size_t* pPitch, size_t WidthInBytes, size_t Height, unsigned int ElementSizeBytes) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuMemFree(CUdeviceptr dptr) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuMemGetAddressRange(CUdeviceptr* pbase, size_t* psize, CUdeviceptr dptr) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuMemAllocHost(void** pp, size_t bytesize) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuMemFreeHost(void* p) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuMemHostAlloc(void** pp, size_t bytesize, unsigned int Flags) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuMemHostGetDevicePointer(CUdeviceptr* pdptr, void* p, unsigned int Flags) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuMemHostGetFlags(unsigned int* pFlags, void* p) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuMemAllocManaged(CUdeviceptr* dptr, size_t bytesize, unsigned int flags) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuDeviceGetByPCIBusId(CUdevice* dev, const char* pciBusId) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuDeviceGetPCIBusId(char* pciBusId, int length, CUdevice dev) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuIpcGetEventHandle(CUipcEventHandle* pHandle, CUevent event) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuIpcOpenEventHandle(CUevent* phEvent, CUipcEventHandle handle) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuIpcGetMemHandle(CUipcMemHandle* pHandle, CUdeviceptr dptr) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuIpcOpenMemHandle(CUdeviceptr* pdptr, CUipcMemHandle handle, unsigned int Flags) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuIpcCloseMemHandle(CUdeviceptr dptr) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuMemHostRegister(void* p, size_t bytesize, unsigned int Flags) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuMemHostUnregister(void* p) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuMemcpy(CUdeviceptr dst, CUdeviceptr src, size_t ByteCount) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuMemcpyPeer(CUdeviceptr dstDevice, CUcontext dstContext, CUdeviceptr srcDevice, CUcontext srcContext, size_t ByteCount) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuMemcpyHtoD(CUdeviceptr dstDevice, const void* srcHost, size_t ByteCount) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuMemcpyDtoH(void* dstHost, CUdeviceptr srcDevice, size_t ByteCount) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuMemcpyDtoD(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuMemcpyDtoA(CUarray dstArray, size_t dstOffset, CUdeviceptr srcDevice, size_t ByteCount) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuMemcpyAtoD(CUdeviceptr dstDevice, CUarray srcArray, size_t srcOffset, size_t ByteCount) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuMemcpyHtoA(CUarray dstArray, size_t dstOffset, const void* srcHost, size_t ByteCount) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuMemcpyAtoH(void* dstHost, CUarray srcArray, size_t srcOffset, size_t ByteCount) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuMemcpyAtoA(CUarray dstArray, size_t dstOffset, CUarray srcArray, size_t srcOffset, size_t ByteCount) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuMemcpy2D(const CUDA_MEMCPY2D* pCopy) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuMemcpy2DUnaligned(const CUDA_MEMCPY2D* pCopy) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuMemcpy3D(const CUDA_MEMCPY3D* pCopy) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuMemcpy3DPeer(const CUDA_MEMCPY3D_PEER* pCopy) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuMemcpyAsync(CUdeviceptr dst, CUdeviceptr src, size_t ByteCount, CUstream hStream) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuMemcpyPeerAsync(CUdeviceptr dstDevice, CUcontext dstContext, CUdeviceptr srcDevice, CUcontext srcContext, size_t ByteCount, CUstream hStream) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuMemcpyHtoDAsync(CUdeviceptr dstDevice, const void* srcHost, size_t ByteCount, CUstream hStream) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuMemcpyDtoHAsync(void* dstHost, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuMemcpyDtoDAsync(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuMemcpyHtoAAsync(CUarray dstArray, size_t dstOffset, const void* srcHost, size_t ByteCount, CUstream hStream) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuMemcpyAtoHAsync(void* dstHost, CUarray srcArray, size_t srcOffset, size_t ByteCount, CUstream hStream) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuMemcpy2DAsync(const CUDA_MEMCPY2D* pCopy, CUstream hStream) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuMemcpy3DAsync(const CUDA_MEMCPY3D* pCopy, CUstream hStream) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuMemcpy3DPeerAsync(const CUDA_MEMCPY3D_PEER* pCopy, CUstream hStream) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuMemsetD8(CUdeviceptr dstDevice, unsigned char uc, size_t N) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuMemsetD16(CUdeviceptr dstDevice, unsigned short us, size_t N) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuMemsetD32(CUdeviceptr dstDevice, unsigned int ui, size_t N) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuMemsetD2D8(CUdeviceptr dstDevice, size_t dstPitch, unsigned char uc, size_t Width, size_t Height) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuMemsetD2D16(CUdeviceptr dstDevice, size_t dstPitch, unsigned short us, size_t Width, size_t Height) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuMemsetD2D32(CUdeviceptr dstDevice, size_t dstPitch, unsigned int ui, size_t Width, size_t Height) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuMemsetD8Async(CUdeviceptr dstDevice, unsigned char uc, size_t N, CUstream hStream) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuMemsetD16Async(CUdeviceptr dstDevice, unsigned short us, size_t N, CUstream hStream) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuMemsetD32Async(CUdeviceptr dstDevice, unsigned int ui, size_t N, CUstream hStream) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuMemsetD2D8Async(CUdeviceptr dstDevice, size_t dstPitch, unsigned char uc, size_t Width, size_t Height, CUstream hStream) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuMemsetD2D16Async(CUdeviceptr dstDevice, size_t dstPitch, unsigned short us, size_t Width, size_t Height, CUstream hStream) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuMemsetD2D32Async(CUdeviceptr dstDevice, size_t dstPitch, unsigned int ui, size_t Width, size_t Height, CUstream hStream) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuArrayCreate(CUarray* pHandle, const CUDA_ARRAY_DESCRIPTOR* pAllocateArray) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuArrayGetDescriptor(CUDA_ARRAY_DESCRIPTOR* pArrayDescriptor, CUarray hArray) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuArrayGetSparseProperties(CUDA_ARRAY_SPARSE_PROPERTIES* sparseProperties, CUarray array) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuMipmappedArrayGetSparseProperties(CUDA_ARRAY_SPARSE_PROPERTIES* sparseProperties, CUmipmappedArray mipmap) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuArrayGetMemoryRequirements(CUDA_ARRAY_MEMORY_REQUIREMENTS* memoryRequirements, CUarray array, CUdevice device) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuMipmappedArrayGetMemoryRequirements(CUDA_ARRAY_MEMORY_REQUIREMENTS* memoryRequirements, CUmipmappedArray mipmap, CUdevice device) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuArrayGetPlane(CUarray* pPlaneArray, CUarray hArray, unsigned int planeIdx) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuArrayDestroy(CUarray hArray) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuArray3DCreate(CUarray* pHandle, const CUDA_ARRAY3D_DESCRIPTOR* pAllocateArray) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuArray3DGetDescriptor(CUDA_ARRAY3D_DESCRIPTOR* pArrayDescriptor, CUarray hArray) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuMipmappedArrayCreate(CUmipmappedArray* pHandle, const CUDA_ARRAY3D_DESCRIPTOR* pMipmappedArrayDesc, unsigned int numMipmapLevels) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuMipmappedArrayGetLevel(CUarray* pLevelArray, CUmipmappedArray hMipmappedArray, unsigned int level) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuMipmappedArrayDestroy(CUmipmappedArray hMipmappedArray) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuMemAddressReserve(CUdeviceptr* ptr, size_t size, size_t alignment, CUdeviceptr addr, unsigned long long flags) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuMemAddressFree(CUdeviceptr ptr, size_t size) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuMemCreate(CUmemGenericAllocationHandle* handle, size_t size, const CUmemAllocationProp* prop, unsigned long long flags) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuMemRelease(CUmemGenericAllocationHandle handle) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuMemMap(CUdeviceptr ptr, size_t size, size_t offset, CUmemGenericAllocationHandle handle, unsigned long long flags) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuMemMapArrayAsync(CUarrayMapInfo* mapInfoList, unsigned int count, CUstream hStream) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuMemUnmap(CUdeviceptr ptr, size_t size) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuMemSetAccess(CUdeviceptr ptr, size_t size, const CUmemAccessDesc* desc, size_t count) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuMemGetAccess(unsigned long long* flags, const CUmemLocation* location, CUdeviceptr ptr) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuMemExportToShareableHandle(void* shareableHandle, CUmemGenericAllocationHandle handle, CUmemAllocationHandleType handleType, unsigned long long flags) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuMemImportFromShareableHandle(CUmemGenericAllocationHandle* handle, void* osHandle, CUmemAllocationHandleType shHandleType) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuMemGetAllocationGranularity(size_t* granularity, const CUmemAllocationProp* prop, CUmemAllocationGranularity_flags option) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuMemGetAllocationPropertiesFromHandle(CUmemAllocationProp* prop, CUmemGenericAllocationHandle handle) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuMemRetainAllocationHandle(CUmemGenericAllocationHandle* handle, void* addr) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuMemFreeAsync(CUdeviceptr dptr, CUstream hStream) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuMemAllocAsync(CUdeviceptr* dptr, size_t bytesize, CUstream hStream) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuMemPoolTrimTo(CUmemoryPool pool, size_t minBytesToKeep) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuMemPoolSetAttribute(CUmemoryPool pool, CUmemPool_attribute attr, void* value) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuMemPoolGetAttribute(CUmemoryPool pool, CUmemPool_attribute attr, void* value) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuMemPoolSetAccess(CUmemoryPool pool, const CUmemAccessDesc* map, size_t count) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuMemPoolGetAccess(CUmemAccess_flags* flags, CUmemoryPool memPool, CUmemLocation* location) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuMemPoolCreate(CUmemoryPool* pool, const CUmemPoolProps* poolProps) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuMemPoolDestroy(CUmemoryPool pool) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuMemAllocFromPoolAsync(CUdeviceptr* dptr, size_t bytesize, CUmemoryPool pool, CUstream hStream) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuMemPoolExportToShareableHandle(void* handle_out, CUmemoryPool pool, CUmemAllocationHandleType handleType, unsigned long long flags) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuMemPoolImportFromShareableHandle(CUmemoryPool* pool_out, void* handle, CUmemAllocationHandleType handleType, unsigned long long flags) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuMemPoolExportPointer(CUmemPoolPtrExportData* shareData_out, CUdeviceptr ptr) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuMemPoolImportPointer(CUdeviceptr* ptr_out, CUmemoryPool pool, CUmemPoolPtrExportData* shareData) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuPointerGetAttribute(void* data, CUpointer_attribute attribute, CUdeviceptr ptr) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuMemPrefetchAsync(CUdeviceptr devPtr, size_t count, CUdevice dstDevice, CUstream hStream) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuMemAdvise(CUdeviceptr devPtr, size_t count, CUmem_advise advice, CUdevice device) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuMemRangeGetAttribute(void* data, size_t dataSize, CUmem_range_attribute attribute, CUdeviceptr devPtr, size_t count) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuMemRangeGetAttributes(void** data, size_t* dataSizes, CUmem_range_attribute* attributes, size_t numAttributes, CUdeviceptr devPtr, size_t count) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuPointerSetAttribute(const void* value, CUpointer_attribute attribute, CUdeviceptr ptr) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuPointerGetAttributes(unsigned int numAttributes, CUpointer_attribute* attributes, void** data, CUdeviceptr ptr) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuStreamCreate(CUstream* phStream, unsigned int Flags) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuStreamCreateWithPriority(CUstream* phStream, unsigned int flags, int priority) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuStreamGetPriority(CUstream hStream, int* priority) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuStreamGetFlags(CUstream hStream, unsigned int* flags) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuStreamGetCtx(CUstream hStream, CUcontext* pctx) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuStreamWaitEvent(CUstream hStream, CUevent hEvent, unsigned int Flags) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuStreamAddCallback(CUstream hStream, CUstreamCallback callback, void* userData, unsigned int flags) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuStreamBeginCapture(CUstream hStream, CUstreamCaptureMode mode) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuThreadExchangeStreamCaptureMode(CUstreamCaptureMode* mode) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuStreamEndCapture(CUstream hStream, CUgraph* phGraph) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuStreamIsCapturing(CUstream hStream, CUstreamCaptureStatus* captureStatus) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuStreamGetCaptureInfo(CUstream hStream, CUstreamCaptureStatus* captureStatus_out, cuuint64_t* id_out) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuStreamGetCaptureInfo_v2(CUstream hStream, CUstreamCaptureStatus* captureStatus_out, cuuint64_t* id_out, CUgraph* graph_out, const CUgraphNode** dependencies_out, size_t* numDependencies_out) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuStreamUpdateCaptureDependencies(CUstream hStream, CUgraphNode* dependencies, size_t numDependencies, unsigned int flags) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuStreamAttachMemAsync(CUstream hStream, CUdeviceptr dptr, size_t length, unsigned int flags) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuStreamQuery(CUstream hStream) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuStreamSynchronize(CUstream hStream) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuStreamDestroy(CUstream hStream) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuStreamCopyAttributes(CUstream dst, CUstream src) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuStreamGetAttribute(CUstream hStream, CUstreamAttrID attr, CUstreamAttrValue* value_out) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuStreamSetAttribute(CUstream hStream, CUstreamAttrID attr, const CUstreamAttrValue* value) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuEventCreate(CUevent* phEvent, unsigned int Flags) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuEventRecord(CUevent hEvent, CUstream hStream) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuEventRecordWithFlags(CUevent hEvent, CUstream hStream, unsigned int flags) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuEventQuery(CUevent hEvent) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuEventSynchronize(CUevent hEvent) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuEventDestroy(CUevent hEvent) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuEventElapsedTime(float* pMilliseconds, CUevent hStart, CUevent hEnd) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuImportExternalMemory(CUexternalMemory* extMem_out, const CUDA_EXTERNAL_MEMORY_HANDLE_DESC* memHandleDesc) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuExternalMemoryGetMappedBuffer(CUdeviceptr* devPtr, CUexternalMemory extMem, const CUDA_EXTERNAL_MEMORY_BUFFER_DESC* bufferDesc) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuExternalMemoryGetMappedMipmappedArray(CUmipmappedArray* mipmap, CUexternalMemory extMem, const CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC* mipmapDesc) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuDestroyExternalMemory(CUexternalMemory extMem) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuImportExternalSemaphore(CUexternalSemaphore* extSem_out, const CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC* semHandleDesc) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuSignalExternalSemaphoresAsync(const CUexternalSemaphore* extSemArray, const CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS* paramsArray, unsigned int numExtSems, CUstream stream) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuWaitExternalSemaphoresAsync(const CUexternalSemaphore* extSemArray, const CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS* paramsArray, unsigned int numExtSems, CUstream stream) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuDestroyExternalSemaphore(CUexternalSemaphore extSem) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuStreamWaitValue32(CUstream stream, CUdeviceptr addr, cuuint32_t value, unsigned int flags) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuStreamWaitValue64(CUstream stream, CUdeviceptr addr, cuuint64_t value, unsigned int flags) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuStreamWriteValue32(CUstream stream, CUdeviceptr addr, cuuint32_t value, unsigned int flags) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuStreamWriteValue64(CUstream stream, CUdeviceptr addr, cuuint64_t value, unsigned int flags) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuStreamBatchMemOp(CUstream stream, unsigned int count, CUstreamBatchMemOpParams* paramArray, unsigned int flags) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuStreamWaitValue32_v2(CUstream stream, CUdeviceptr addr, cuuint32_t value, unsigned int flags) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuStreamWaitValue64_v2(CUstream stream, CUdeviceptr addr, cuuint64_t value, unsigned int flags) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuStreamWriteValue32_v2(CUstream stream, CUdeviceptr addr, cuuint32_t value, unsigned int flags) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuStreamWriteValue64_v2(CUstream stream, CUdeviceptr addr, cuuint64_t value, unsigned int flags) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuStreamBatchMemOp_v2(CUstream stream, unsigned int count, CUstreamBatchMemOpParams* paramArray, unsigned int flags) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuFuncGetAttribute(int* pi, CUfunction_attribute attrib, CUfunction hfunc) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuFuncSetAttribute(CUfunction hfunc, CUfunction_attribute attrib, int value) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuFuncSetCacheConfig(CUfunction hfunc, CUfunc_cache config) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuFuncSetSharedMemConfig(CUfunction hfunc, CUsharedconfig config) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuFuncGetModule(CUmodule* hmod, CUfunction hfunc) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuLaunchKernel(CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, unsigned int sharedMemBytes, CUstream hStream, void** kernelParams, void** extra) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuLaunchCooperativeKernel(CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, unsigned int sharedMemBytes, CUstream hStream, void** kernelParams) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuLaunchCooperativeKernelMultiDevice(CUDA_LAUNCH_PARAMS* launchParamsList, unsigned int numDevices, unsigned int flags) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuLaunchHostFunc(CUstream hStream, CUhostFn fn, void* userData) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuFuncSetBlockShape(CUfunction hfunc, int x, int y, int z) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuFuncSetSharedSize(CUfunction hfunc, unsigned int numbytes) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuParamSetSize(CUfunction hfunc, unsigned int numbytes) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuParamSeti(CUfunction hfunc, int offset, unsigned int value) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuParamSetf(CUfunction hfunc, int offset, float value) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuParamSetv(CUfunction hfunc, int offset, void* ptr, unsigned int numbytes) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuLaunch(CUfunction f) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuLaunchGrid(CUfunction f, int grid_width, int grid_height) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuLaunchGridAsync(CUfunction f, int grid_width, int grid_height, CUstream hStream) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuParamSetTexRef(CUfunction hfunc, int texunit, CUtexref hTexRef) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuGraphCreate(CUgraph* phGraph, unsigned int flags) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuGraphAddKernelNode(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, size_t numDependencies, const CUDA_KERNEL_NODE_PARAMS* nodeParams) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuGraphKernelNodeGetParams(CUgraphNode hNode, CUDA_KERNEL_NODE_PARAMS* nodeParams) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuGraphKernelNodeSetParams(CUgraphNode hNode, const CUDA_KERNEL_NODE_PARAMS* nodeParams) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuGraphAddMemcpyNode(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, size_t numDependencies, const CUDA_MEMCPY3D* copyParams, CUcontext ctx) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuGraphMemcpyNodeGetParams(CUgraphNode hNode, CUDA_MEMCPY3D* nodeParams) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuGraphMemcpyNodeSetParams(CUgraphNode hNode, const CUDA_MEMCPY3D* nodeParams) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuGraphAddMemsetNode(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, size_t numDependencies, const CUDA_MEMSET_NODE_PARAMS* memsetParams, CUcontext ctx) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuGraphMemsetNodeGetParams(CUgraphNode hNode, CUDA_MEMSET_NODE_PARAMS* nodeParams) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuGraphMemsetNodeSetParams(CUgraphNode hNode, const CUDA_MEMSET_NODE_PARAMS* nodeParams) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuGraphAddHostNode(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, size_t numDependencies, const CUDA_HOST_NODE_PARAMS* nodeParams) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuGraphHostNodeGetParams(CUgraphNode hNode, CUDA_HOST_NODE_PARAMS* nodeParams) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuGraphHostNodeSetParams(CUgraphNode hNode, const CUDA_HOST_NODE_PARAMS* nodeParams) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuGraphAddChildGraphNode(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, size_t numDependencies, CUgraph childGraph) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuGraphChildGraphNodeGetGraph(CUgraphNode hNode, CUgraph* phGraph) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuGraphAddEmptyNode(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, size_t numDependencies) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuGraphAddEventRecordNode(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, size_t numDependencies, CUevent event) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuGraphEventRecordNodeGetEvent(CUgraphNode hNode, CUevent* event_out) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuGraphEventRecordNodeSetEvent(CUgraphNode hNode, CUevent event) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuGraphAddEventWaitNode(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, size_t numDependencies, CUevent event) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuGraphEventWaitNodeGetEvent(CUgraphNode hNode, CUevent* event_out) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuGraphEventWaitNodeSetEvent(CUgraphNode hNode, CUevent event) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuGraphAddExternalSemaphoresSignalNode(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, size_t numDependencies, const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS* nodeParams) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuGraphExternalSemaphoresSignalNodeGetParams(CUgraphNode hNode, CUDA_EXT_SEM_SIGNAL_NODE_PARAMS* params_out) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuGraphExternalSemaphoresSignalNodeSetParams(CUgraphNode hNode, const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS* nodeParams) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuGraphAddExternalSemaphoresWaitNode(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, size_t numDependencies, const CUDA_EXT_SEM_WAIT_NODE_PARAMS* nodeParams) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuGraphExternalSemaphoresWaitNodeGetParams(CUgraphNode hNode, CUDA_EXT_SEM_WAIT_NODE_PARAMS* params_out) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuGraphExternalSemaphoresWaitNodeSetParams(CUgraphNode hNode, const CUDA_EXT_SEM_WAIT_NODE_PARAMS* nodeParams) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuGraphAddBatchMemOpNode(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, size_t numDependencies, const CUDA_BATCH_MEM_OP_NODE_PARAMS* nodeParams) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuGraphBatchMemOpNodeGetParams(CUgraphNode hNode, CUDA_BATCH_MEM_OP_NODE_PARAMS* nodeParams_out) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuGraphBatchMemOpNodeSetParams(CUgraphNode hNode, const CUDA_BATCH_MEM_OP_NODE_PARAMS* nodeParams) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuGraphExecBatchMemOpNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_BATCH_MEM_OP_NODE_PARAMS* nodeParams) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuGraphAddMemAllocNode(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, size_t numDependencies, CUDA_MEM_ALLOC_NODE_PARAMS* nodeParams) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuGraphMemAllocNodeGetParams(CUgraphNode hNode, CUDA_MEM_ALLOC_NODE_PARAMS* params_out) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuGraphAddMemFreeNode(CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, size_t numDependencies, CUdeviceptr dptr) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuGraphMemFreeNodeGetParams(CUgraphNode hNode, CUdeviceptr* dptr_out) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuDeviceGraphMemTrim(CUdevice device) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuDeviceGetGraphMemAttribute(CUdevice device, CUgraphMem_attribute attr, void* value) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuDeviceSetGraphMemAttribute(CUdevice device, CUgraphMem_attribute attr, void* value) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuGraphClone(CUgraph* phGraphClone, CUgraph originalGraph) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuGraphNodeFindInClone(CUgraphNode* phNode, CUgraphNode hOriginalNode, CUgraph hClonedGraph) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuGraphNodeGetType(CUgraphNode hNode, CUgraphNodeType* typename) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuGraphGetNodes(CUgraph hGraph, CUgraphNode* nodes, size_t* numNodes) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuGraphGetRootNodes(CUgraph hGraph, CUgraphNode* rootNodes, size_t* numRootNodes) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuGraphGetEdges(CUgraph hGraph, CUgraphNode* from_, CUgraphNode* to, size_t* numEdges) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuGraphNodeGetDependencies(CUgraphNode hNode, CUgraphNode* dependencies, size_t* numDependencies) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuGraphNodeGetDependentNodes(CUgraphNode hNode, CUgraphNode* dependentNodes, size_t* numDependentNodes) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuGraphAddDependencies(CUgraph hGraph, const CUgraphNode* from_, const CUgraphNode* to, size_t numDependencies) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuGraphRemoveDependencies(CUgraph hGraph, const CUgraphNode* from_, const CUgraphNode* to, size_t numDependencies) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuGraphDestroyNode(CUgraphNode hNode) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuGraphInstantiate(CUgraphExec* phGraphExec, CUgraph hGraph, CUgraphNode* phErrorNode, char* logBuffer, size_t bufferSize) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuGraphInstantiateWithFlags(CUgraphExec* phGraphExec, CUgraph hGraph, unsigned long long flags) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuGraphExecKernelNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_KERNEL_NODE_PARAMS* nodeParams) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuGraphExecMemcpyNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_MEMCPY3D* copyParams, CUcontext ctx) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuGraphExecMemsetNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_MEMSET_NODE_PARAMS* memsetParams, CUcontext ctx) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuGraphExecHostNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_HOST_NODE_PARAMS* nodeParams) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuGraphExecChildGraphNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, CUgraph childGraph) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuGraphExecEventRecordNodeSetEvent(CUgraphExec hGraphExec, CUgraphNode hNode, CUevent event) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuGraphExecEventWaitNodeSetEvent(CUgraphExec hGraphExec, CUgraphNode hNode, CUevent event) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuGraphExecExternalSemaphoresSignalNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS* nodeParams) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuGraphExecExternalSemaphoresWaitNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_EXT_SEM_WAIT_NODE_PARAMS* nodeParams) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuGraphNodeSetEnabled(CUgraphExec hGraphExec, CUgraphNode hNode, unsigned int isEnabled) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuGraphNodeGetEnabled(CUgraphExec hGraphExec, CUgraphNode hNode, unsigned int* isEnabled) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuGraphUpload(CUgraphExec hGraphExec, CUstream hStream) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuGraphLaunch(CUgraphExec hGraphExec, CUstream hStream) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuGraphExecDestroy(CUgraphExec hGraphExec) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuGraphDestroy(CUgraph hGraph) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuGraphExecUpdate(CUgraphExec hGraphExec, CUgraph hGraph, CUgraphNode* hErrorNode_out, CUgraphExecUpdateResult* updateResult_out) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuGraphKernelNodeCopyAttributes(CUgraphNode dst, CUgraphNode src) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuGraphKernelNodeGetAttribute(CUgraphNode hNode, CUkernelNodeAttrID attr, CUkernelNodeAttrValue* value_out) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuGraphKernelNodeSetAttribute(CUgraphNode hNode, CUkernelNodeAttrID attr, const CUkernelNodeAttrValue* value) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuGraphDebugDotPrint(CUgraph hGraph, const char* path, unsigned int flags) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuUserObjectCreate(CUuserObject* object_out, void* ptr, CUhostFn destroy, unsigned int initialRefcount, unsigned int flags) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuUserObjectRetain(CUuserObject object, unsigned int count) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuUserObjectRelease(CUuserObject object, unsigned int count) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuGraphRetainUserObject(CUgraph graph, CUuserObject object, unsigned int count, unsigned int flags) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuGraphReleaseUserObject(CUgraph graph, CUuserObject object, unsigned int count) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuOccupancyMaxActiveBlocksPerMultiprocessor(int* numBlocks, CUfunction func, int blockSize, size_t dynamicSMemSize) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int* numBlocks, CUfunction func, int blockSize, size_t dynamicSMemSize, unsigned int flags) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuOccupancyMaxPotentialBlockSize(int* minGridSize, int* blockSize, CUfunction func, CUoccupancyB2DSize blockSizeToDynamicSMemSize, size_t dynamicSMemSize, int blockSizeLimit) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuOccupancyMaxPotentialBlockSizeWithFlags(int* minGridSize, int* blockSize, CUfunction func, CUoccupancyB2DSize blockSizeToDynamicSMemSize, size_t dynamicSMemSize, int blockSizeLimit, unsigned int flags) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuOccupancyAvailableDynamicSMemPerBlock(size_t* dynamicSmemSize, CUfunction func, int numBlocks, int blockSize) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuTexRefSetArray(CUtexref hTexRef, CUarray hArray, unsigned int Flags) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuTexRefSetMipmappedArray(CUtexref hTexRef, CUmipmappedArray hMipmappedArray, unsigned int Flags) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuTexRefSetAddress(size_t* ByteOffset, CUtexref hTexRef, CUdeviceptr dptr, size_t numbytes) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuTexRefSetAddress2D(CUtexref hTexRef, const CUDA_ARRAY_DESCRIPTOR* desc, CUdeviceptr dptr, size_t Pitch) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuTexRefSetFormat(CUtexref hTexRef, CUarray_format fmt, int NumPackedComponents) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuTexRefSetAddressMode(CUtexref hTexRef, int dim, CUaddress_mode am) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuTexRefSetFilterMode(CUtexref hTexRef, CUfilter_mode fm) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuTexRefSetMipmapFilterMode(CUtexref hTexRef, CUfilter_mode fm) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuTexRefSetMipmapLevelBias(CUtexref hTexRef, float bias) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuTexRefSetMipmapLevelClamp(CUtexref hTexRef, float minMipmapLevelClamp, float maxMipmapLevelClamp) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuTexRefSetMaxAnisotropy(CUtexref hTexRef, unsigned int maxAniso) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuTexRefSetBorderColor(CUtexref hTexRef, float* pBorderColor) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuTexRefSetFlags(CUtexref hTexRef, unsigned int Flags) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuTexRefGetAddress(CUdeviceptr* pdptr, CUtexref hTexRef) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuTexRefGetArray(CUarray* phArray, CUtexref hTexRef) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuTexRefGetMipmappedArray(CUmipmappedArray* phMipmappedArray, CUtexref hTexRef) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuTexRefGetAddressMode(CUaddress_mode* pam, CUtexref hTexRef, int dim) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuTexRefGetFilterMode(CUfilter_mode* pfm, CUtexref hTexRef) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuTexRefGetFormat(CUarray_format* pFormat, int* pNumChannels, CUtexref hTexRef) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuTexRefGetMipmapFilterMode(CUfilter_mode* pfm, CUtexref hTexRef) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuTexRefGetMipmapLevelBias(float* pbias, CUtexref hTexRef) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuTexRefGetMipmapLevelClamp(float* pminMipmapLevelClamp, float* pmaxMipmapLevelClamp, CUtexref hTexRef) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuTexRefGetMaxAnisotropy(int* pmaxAniso, CUtexref hTexRef) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuTexRefGetBorderColor(float* pBorderColor, CUtexref hTexRef) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuTexRefGetFlags(unsigned int* pFlags, CUtexref hTexRef) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuTexRefCreate(CUtexref* pTexRef) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuTexRefDestroy(CUtexref hTexRef) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuSurfRefSetArray(CUsurfref hSurfRef, CUarray hArray, unsigned int Flags) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuSurfRefGetArray(CUarray* phArray, CUsurfref hSurfRef) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuTexObjectCreate(CUtexObject* pTexObject, const CUDA_RESOURCE_DESC* pResDesc, const CUDA_TEXTURE_DESC* pTexDesc, const CUDA_RESOURCE_VIEW_DESC* pResViewDesc) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuTexObjectDestroy(CUtexObject texObject) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuTexObjectGetResourceDesc(CUDA_RESOURCE_DESC* pResDesc, CUtexObject texObject) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuTexObjectGetTextureDesc(CUDA_TEXTURE_DESC* pTexDesc, CUtexObject texObject) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuTexObjectGetResourceViewDesc(CUDA_RESOURCE_VIEW_DESC* pResViewDesc, CUtexObject texObject) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuSurfObjectCreate(CUsurfObject* pSurfObject, const CUDA_RESOURCE_DESC* pResDesc) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuSurfObjectDestroy(CUsurfObject surfObject) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuSurfObjectGetResourceDesc(CUDA_RESOURCE_DESC* pResDesc, CUsurfObject surfObject) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuDeviceCanAccessPeer(int* canAccessPeer, CUdevice dev, CUdevice peerDev) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuCtxEnablePeerAccess(CUcontext peerContext, unsigned int Flags) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuCtxDisablePeerAccess(CUcontext peerContext) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuDeviceGetP2PAttribute(int* value, CUdevice_P2PAttribute attrib, CUdevice srcDevice, CUdevice dstDevice) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuGraphicsUnregisterResource(CUgraphicsResource resource) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuGraphicsSubResourceGetMappedArray(CUarray* pArray, CUgraphicsResource resource, unsigned int arrayIndex, unsigned int mipLevel) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuGraphicsResourceGetMappedMipmappedArray(CUmipmappedArray* pMipmappedArray, CUgraphicsResource resource) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuGraphicsResourceGetMappedPointer(CUdeviceptr* pDevPtr, size_t* pSize, CUgraphicsResource resource) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuGraphicsResourceSetMapFlags(CUgraphicsResource resource, unsigned int flags) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuGraphicsMapResources(unsigned int count, CUgraphicsResource* resources, CUstream hStream) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuGraphicsUnmapResources(unsigned int count, CUgraphicsResource* resources, CUstream hStream) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuGetProcAddress(const char* symbol, void** pfn, int cudaVersion, cuuint64_t flags) nogil except ?CUDA_ERROR_NOT_FOUND

cdef enum CUmoduleLoadingMode_enum:
    CU_MODULE_EAGER_LOADING = 1
    CU_MODULE_LAZY_LOADING = 2

ctypedef CUmoduleLoadingMode_enum CUmoduleLoadingMode

cdef CUresult cuModuleGetLoadingMode(CUmoduleLoadingMode* mode) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuMemGetHandleForAddressRange(void* handle, CUdeviceptr dptr, size_t size, CUmemRangeHandleType handleType, unsigned long long flags) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuGetExportTable(const void** ppExportTable, const CUuuid* pExportTableId) nogil except ?CUDA_ERROR_NOT_FOUND

cdef enum CUoutput_mode_enum:
    CU_OUT_KEY_VALUE_PAIR = 0
    CU_OUT_CSV = 1

ctypedef CUoutput_mode_enum CUoutput_mode

cdef CUresult cuProfilerInitialize(const char* configFile, const char* outputFile, CUoutput_mode outputMode) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuProfilerStart() nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuProfilerStop() nogil except ?CUDA_ERROR_NOT_FOUND
from libc.stdint cimport uint32_t


ctypedef unsigned int GLenum

ctypedef unsigned int GLuint

cdef extern from "":
    cdef struct void:
        pass
ctypedef void* EGLImageKHR

cdef extern from "":
    cdef struct void:
        pass
ctypedef void* EGLStreamKHR

ctypedef unsigned int EGLint

cdef extern from "":
    cdef struct void:
        pass
ctypedef void* EGLSyncKHR

ctypedef uint32_t VdpDevice

ctypedef unsigned long long VdpGetProcAddress

ctypedef uint32_t VdpVideoSurface

ctypedef uint32_t VdpOutputSurface

cdef CUresult cuVDPAUGetDevice(CUdevice* pDevice, VdpDevice vdpDevice, VdpGetProcAddress* vdpGetProcAddress) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuVDPAUCtxCreate(CUcontext* pCtx, unsigned int flags, CUdevice device, VdpDevice vdpDevice, VdpGetProcAddress* vdpGetProcAddress) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuGraphicsVDPAURegisterVideoSurface(CUgraphicsResource* pCudaResource, VdpVideoSurface vdpSurface, unsigned int flags) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuGraphicsVDPAURegisterOutputSurface(CUgraphicsResource* pCudaResource, VdpOutputSurface vdpSurface, unsigned int flags) nogil except ?CUDA_ERROR_NOT_FOUND

cdef enum CUeglFrameType_enum:
    CU_EGL_FRAME_TYPE_ARRAY = 0
    CU_EGL_FRAME_TYPE_PITCH = 1

ctypedef CUeglFrameType_enum CUeglFrameType

cdef enum CUeglResourceLocationFlags_enum:
    CU_EGL_RESOURCE_LOCATION_SYSMEM = 0
    CU_EGL_RESOURCE_LOCATION_VIDMEM = 1

ctypedef CUeglResourceLocationFlags_enum CUeglResourceLocationFlags

cdef enum CUeglColorFormat_enum:
    CU_EGL_COLOR_FORMAT_YUV420_PLANAR = 0
    CU_EGL_COLOR_FORMAT_YUV420_SEMIPLANAR = 1
    CU_EGL_COLOR_FORMAT_YUV422_PLANAR = 2
    CU_EGL_COLOR_FORMAT_YUV422_SEMIPLANAR = 3
    CU_EGL_COLOR_FORMAT_RGB = 4
    CU_EGL_COLOR_FORMAT_BGR = 5
    CU_EGL_COLOR_FORMAT_ARGB = 6
    CU_EGL_COLOR_FORMAT_RGBA = 7
    CU_EGL_COLOR_FORMAT_L = 8
    CU_EGL_COLOR_FORMAT_R = 9
    CU_EGL_COLOR_FORMAT_YUV444_PLANAR = 10
    CU_EGL_COLOR_FORMAT_YUV444_SEMIPLANAR = 11
    CU_EGL_COLOR_FORMAT_YUYV_422 = 12
    CU_EGL_COLOR_FORMAT_UYVY_422 = 13
    CU_EGL_COLOR_FORMAT_ABGR = 14
    CU_EGL_COLOR_FORMAT_BGRA = 15
    CU_EGL_COLOR_FORMAT_A = 16
    CU_EGL_COLOR_FORMAT_RG = 17
    CU_EGL_COLOR_FORMAT_AYUV = 18
    CU_EGL_COLOR_FORMAT_YVU444_SEMIPLANAR = 19
    CU_EGL_COLOR_FORMAT_YVU422_SEMIPLANAR = 20
    CU_EGL_COLOR_FORMAT_YVU420_SEMIPLANAR = 21
    CU_EGL_COLOR_FORMAT_Y10V10U10_444_SEMIPLANAR = 22
    CU_EGL_COLOR_FORMAT_Y10V10U10_420_SEMIPLANAR = 23
    CU_EGL_COLOR_FORMAT_Y12V12U12_444_SEMIPLANAR = 24
    CU_EGL_COLOR_FORMAT_Y12V12U12_420_SEMIPLANAR = 25
    CU_EGL_COLOR_FORMAT_VYUY_ER = 26
    CU_EGL_COLOR_FORMAT_UYVY_ER = 27
    CU_EGL_COLOR_FORMAT_YUYV_ER = 28
    CU_EGL_COLOR_FORMAT_YVYU_ER = 29
    CU_EGL_COLOR_FORMAT_YUV_ER = 30
    CU_EGL_COLOR_FORMAT_YUVA_ER = 31
    CU_EGL_COLOR_FORMAT_AYUV_ER = 32
    CU_EGL_COLOR_FORMAT_YUV444_PLANAR_ER = 33
    CU_EGL_COLOR_FORMAT_YUV422_PLANAR_ER = 34
    CU_EGL_COLOR_FORMAT_YUV420_PLANAR_ER = 35
    CU_EGL_COLOR_FORMAT_YUV444_SEMIPLANAR_ER = 36
    CU_EGL_COLOR_FORMAT_YUV422_SEMIPLANAR_ER = 37
    CU_EGL_COLOR_FORMAT_YUV420_SEMIPLANAR_ER = 38
    CU_EGL_COLOR_FORMAT_YVU444_PLANAR_ER = 39
    CU_EGL_COLOR_FORMAT_YVU422_PLANAR_ER = 40
    CU_EGL_COLOR_FORMAT_YVU420_PLANAR_ER = 41
    CU_EGL_COLOR_FORMAT_YVU444_SEMIPLANAR_ER = 42
    CU_EGL_COLOR_FORMAT_YVU422_SEMIPLANAR_ER = 43
    CU_EGL_COLOR_FORMAT_YVU420_SEMIPLANAR_ER = 44
    CU_EGL_COLOR_FORMAT_BAYER_RGGB = 45
    CU_EGL_COLOR_FORMAT_BAYER_BGGR = 46
    CU_EGL_COLOR_FORMAT_BAYER_GRBG = 47
    CU_EGL_COLOR_FORMAT_BAYER_GBRG = 48
    CU_EGL_COLOR_FORMAT_BAYER10_RGGB = 49
    CU_EGL_COLOR_FORMAT_BAYER10_BGGR = 50
    CU_EGL_COLOR_FORMAT_BAYER10_GRBG = 51
    CU_EGL_COLOR_FORMAT_BAYER10_GBRG = 52
    CU_EGL_COLOR_FORMAT_BAYER12_RGGB = 53
    CU_EGL_COLOR_FORMAT_BAYER12_BGGR = 54
    CU_EGL_COLOR_FORMAT_BAYER12_GRBG = 55
    CU_EGL_COLOR_FORMAT_BAYER12_GBRG = 56
    CU_EGL_COLOR_FORMAT_BAYER14_RGGB = 57
    CU_EGL_COLOR_FORMAT_BAYER14_BGGR = 58
    CU_EGL_COLOR_FORMAT_BAYER14_GRBG = 59
    CU_EGL_COLOR_FORMAT_BAYER14_GBRG = 60
    CU_EGL_COLOR_FORMAT_BAYER20_RGGB = 61
    CU_EGL_COLOR_FORMAT_BAYER20_BGGR = 62
    CU_EGL_COLOR_FORMAT_BAYER20_GRBG = 63
    CU_EGL_COLOR_FORMAT_BAYER20_GBRG = 64
    CU_EGL_COLOR_FORMAT_YVU444_PLANAR = 65
    CU_EGL_COLOR_FORMAT_YVU422_PLANAR = 66
    CU_EGL_COLOR_FORMAT_YVU420_PLANAR = 67
    CU_EGL_COLOR_FORMAT_BAYER_ISP_RGGB = 68
    CU_EGL_COLOR_FORMAT_BAYER_ISP_BGGR = 69
    CU_EGL_COLOR_FORMAT_BAYER_ISP_GRBG = 70
    CU_EGL_COLOR_FORMAT_BAYER_ISP_GBRG = 71
    CU_EGL_COLOR_FORMAT_BAYER_BCCR = 72
    CU_EGL_COLOR_FORMAT_BAYER_RCCB = 73
    CU_EGL_COLOR_FORMAT_BAYER_CRBC = 74
    CU_EGL_COLOR_FORMAT_BAYER_CBRC = 75
    CU_EGL_COLOR_FORMAT_BAYER10_CCCC = 76
    CU_EGL_COLOR_FORMAT_BAYER12_BCCR = 77
    CU_EGL_COLOR_FORMAT_BAYER12_RCCB = 78
    CU_EGL_COLOR_FORMAT_BAYER12_CRBC = 79
    CU_EGL_COLOR_FORMAT_BAYER12_CBRC = 80
    CU_EGL_COLOR_FORMAT_BAYER12_CCCC = 81
    CU_EGL_COLOR_FORMAT_Y = 82
    CU_EGL_COLOR_FORMAT_YUV420_SEMIPLANAR_2020 = 83
    CU_EGL_COLOR_FORMAT_YVU420_SEMIPLANAR_2020 = 84
    CU_EGL_COLOR_FORMAT_YUV420_PLANAR_2020 = 85
    CU_EGL_COLOR_FORMAT_YVU420_PLANAR_2020 = 86
    CU_EGL_COLOR_FORMAT_YUV420_SEMIPLANAR_709 = 87
    CU_EGL_COLOR_FORMAT_YVU420_SEMIPLANAR_709 = 88
    CU_EGL_COLOR_FORMAT_YUV420_PLANAR_709 = 89
    CU_EGL_COLOR_FORMAT_YVU420_PLANAR_709 = 90
    CU_EGL_COLOR_FORMAT_Y10V10U10_420_SEMIPLANAR_709 = 91
    CU_EGL_COLOR_FORMAT_Y10V10U10_420_SEMIPLANAR_2020 = 92
    CU_EGL_COLOR_FORMAT_Y10V10U10_422_SEMIPLANAR_2020 = 93
    CU_EGL_COLOR_FORMAT_Y10V10U10_422_SEMIPLANAR = 94
    CU_EGL_COLOR_FORMAT_Y10V10U10_422_SEMIPLANAR_709 = 95
    CU_EGL_COLOR_FORMAT_Y_ER = 96
    CU_EGL_COLOR_FORMAT_Y_709_ER = 97
    CU_EGL_COLOR_FORMAT_Y10_ER = 98
    CU_EGL_COLOR_FORMAT_Y10_709_ER = 99
    CU_EGL_COLOR_FORMAT_Y12_ER = 100
    CU_EGL_COLOR_FORMAT_Y12_709_ER = 101
    CU_EGL_COLOR_FORMAT_YUVA = 102
    CU_EGL_COLOR_FORMAT_YUV = 103
    CU_EGL_COLOR_FORMAT_YVYU = 104
    CU_EGL_COLOR_FORMAT_VYUY = 105
    CU_EGL_COLOR_FORMAT_Y10V10U10_420_SEMIPLANAR_ER = 106
    CU_EGL_COLOR_FORMAT_Y10V10U10_420_SEMIPLANAR_709_ER = 107
    CU_EGL_COLOR_FORMAT_Y10V10U10_444_SEMIPLANAR_ER = 108
    CU_EGL_COLOR_FORMAT_Y10V10U10_444_SEMIPLANAR_709_ER = 109
    CU_EGL_COLOR_FORMAT_Y12V12U12_420_SEMIPLANAR_ER = 110
    CU_EGL_COLOR_FORMAT_Y12V12U12_420_SEMIPLANAR_709_ER = 111
    CU_EGL_COLOR_FORMAT_Y12V12U12_444_SEMIPLANAR_ER = 112
    CU_EGL_COLOR_FORMAT_Y12V12U12_444_SEMIPLANAR_709_ER = 113
    CU_EGL_COLOR_FORMAT_MAX = 114

ctypedef CUeglColorFormat_enum CUeglColorFormat

cdef union _CUeglFrame_v1_CUeglFrame_v1_CUeglFrame_st_frame_u:
    CUarray pArray[3]
    void* pPitch[3]

cdef struct CUeglFrame_st:
    _CUeglFrame_v1_CUeglFrame_v1_CUeglFrame_st_frame_u frame
    unsigned int width
    unsigned int height
    unsigned int depth
    unsigned int pitch
    unsigned int planeCount
    unsigned int numChannels
    CUeglFrameType frameType
    CUeglColorFormat eglColorFormat
    CUarray_format cuFormat

ctypedef CUeglFrame_st CUeglFrame_v1

ctypedef CUeglFrame_v1 CUeglFrame

cdef extern from "":
    cdef struct CUeglStreamConnection_st:
        pass
ctypedef CUeglStreamConnection_st* CUeglStreamConnection

cdef CUresult cuGraphicsEGLRegisterImage(CUgraphicsResource* pCudaResource, EGLImageKHR image, unsigned int flags) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuEGLStreamConsumerConnect(CUeglStreamConnection* conn, EGLStreamKHR stream) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuEGLStreamConsumerConnectWithFlags(CUeglStreamConnection* conn, EGLStreamKHR stream, unsigned int flags) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuEGLStreamConsumerDisconnect(CUeglStreamConnection* conn) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuEGLStreamConsumerAcquireFrame(CUeglStreamConnection* conn, CUgraphicsResource* pCudaResource, CUstream* pStream, unsigned int timeout) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuEGLStreamConsumerReleaseFrame(CUeglStreamConnection* conn, CUgraphicsResource pCudaResource, CUstream* pStream) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuEGLStreamProducerConnect(CUeglStreamConnection* conn, EGLStreamKHR stream, EGLint width, EGLint height) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuEGLStreamProducerDisconnect(CUeglStreamConnection* conn) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuEGLStreamProducerPresentFrame(CUeglStreamConnection* conn, CUeglFrame eglframe, CUstream* pStream) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuEGLStreamProducerReturnFrame(CUeglStreamConnection* conn, CUeglFrame* eglframe, CUstream* pStream) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuGraphicsResourceGetMappedEglFrame(CUeglFrame* eglFrame, CUgraphicsResource resource, unsigned int index, unsigned int mipLevel) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuEventCreateFromEGLSync(CUevent* phEvent, EGLSyncKHR eglSync, unsigned int flags) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuGraphicsGLRegisterBuffer(CUgraphicsResource* pCudaResource, GLuint buffer, unsigned int Flags) nogil except ?CUDA_ERROR_NOT_FOUND

cdef CUresult cuGraphicsGLRegisterImage(CUgraphicsResource* pCudaResource, GLuint image, GLenum target, unsigned int Flags) nogil except ?CUDA_ERROR_NOT_FOUND

cdef enum CUGLDeviceList_enum:
    CU_GL_DEVICE_LIST_ALL = 1
    CU_GL_DEVICE_LIST_CURRENT_FRAME = 2
    CU_GL_DEVICE_LIST_NEXT_FRAME = 3

ctypedef CUGLDeviceList_enum CUGLDeviceList

cdef CUresult cuGLGetDevices(unsigned int* pCudaDeviceCount, CUdevice* pCudaDevices, unsigned int cudaDeviceCount, CUGLDeviceList deviceList) nogil except ?CUDA_ERROR_NOT_FOUND

cdef enum CUGLmap_flags_enum:
    CU_GL_MAP_RESOURCE_FLAGS_NONE = 0
    CU_GL_MAP_RESOURCE_FLAGS_READ_ONLY = 1
    CU_GL_MAP_RESOURCE_FLAGS_WRITE_DISCARD = 2

ctypedef CUGLmap_flags_enum CUGLmap_flags

cdef enum: CUDA_VERSION = 11070

cdef enum: CU_IPC_HANDLE_SIZE = 64

cdef enum: CU_STREAM_LEGACY = 1

cdef enum: CU_STREAM_PER_THREAD = 2

cdef enum: CU_MEMHOSTALLOC_PORTABLE = 1

cdef enum: CU_MEMHOSTALLOC_DEVICEMAP = 2

cdef enum: CU_MEMHOSTALLOC_WRITECOMBINED = 4

cdef enum: CU_MEMHOSTREGISTER_PORTABLE = 1

cdef enum: CU_MEMHOSTREGISTER_DEVICEMAP = 2

cdef enum: CU_MEMHOSTREGISTER_IOMEMORY = 4

cdef enum: CU_MEMHOSTREGISTER_READ_ONLY = 8

cdef enum: CU_ARRAY_SPARSE_PROPERTIES_SINGLE_MIPTAIL = 1

cdef enum: CUDA_EXTERNAL_MEMORY_DEDICATED = 1

cdef enum: CUDA_EXTERNAL_SEMAPHORE_SIGNAL_SKIP_NVSCIBUF_MEMSYNC = 1

cdef enum: CUDA_EXTERNAL_SEMAPHORE_WAIT_SKIP_NVSCIBUF_MEMSYNC = 2

cdef enum: CUDA_NVSCISYNC_ATTR_SIGNAL = 1

cdef enum: CUDA_NVSCISYNC_ATTR_WAIT = 2

cdef enum: CU_MEM_CREATE_USAGE_TILE_POOL = 1

cdef enum: CUDA_COOPERATIVE_LAUNCH_MULTI_DEVICE_NO_PRE_LAUNCH_SYNC = 1

cdef enum: CUDA_COOPERATIVE_LAUNCH_MULTI_DEVICE_NO_POST_LAUNCH_SYNC = 2

cdef enum: CUDA_ARRAY3D_LAYERED = 1

cdef enum: CUDA_ARRAY3D_2DARRAY = 1

cdef enum: CUDA_ARRAY3D_SURFACE_LDST = 2

cdef enum: CUDA_ARRAY3D_CUBEMAP = 4

cdef enum: CUDA_ARRAY3D_TEXTURE_GATHER = 8

cdef enum: CUDA_ARRAY3D_DEPTH_TEXTURE = 16

cdef enum: CUDA_ARRAY3D_COLOR_ATTACHMENT = 32

cdef enum: CUDA_ARRAY3D_SPARSE = 64

cdef enum: CUDA_ARRAY3D_DEFERRED_MAPPING = 128

cdef enum: CU_TRSA_OVERRIDE_FORMAT = 1

cdef enum: CU_TRSF_READ_AS_INTEGER = 1

cdef enum: CU_TRSF_NORMALIZED_COORDINATES = 2

cdef enum: CU_TRSF_SRGB = 16

cdef enum: CU_TRSF_DISABLE_TRILINEAR_OPTIMIZATION = 32

cdef enum: CU_TRSF_SEAMLESS_CUBEMAP = 64

cdef enum: CU_LAUNCH_PARAM_END_AS_INT = 0

cdef enum: CU_LAUNCH_PARAM_END = 0

cdef enum: CU_LAUNCH_PARAM_BUFFER_POINTER_AS_INT = 1

cdef enum: CU_LAUNCH_PARAM_BUFFER_POINTER = 1

cdef enum: CU_LAUNCH_PARAM_BUFFER_SIZE_AS_INT = 2

cdef enum: CU_LAUNCH_PARAM_BUFFER_SIZE = 2

cdef enum: CU_PARAM_TR_DEFAULT = -1

cdef enum: CU_DEVICE_CPU = -1

cdef enum: CU_DEVICE_INVALID = -2

cdef enum: MAX_PLANES = 3

cdef enum: CUDA_EGL_INFINITE_TIMEOUT = 4294967295
