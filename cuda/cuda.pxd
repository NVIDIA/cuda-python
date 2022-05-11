# Copyright 2021-2022 NVIDIA Corporation.  All rights reserved.
#
# Please refer to the NVIDIA end user license agreement (EULA) associated
# with this source code for terms and conditions that govern your use of
# this software. Any use, reproduction, disclosure, or distribution of
# this software and related documentation outside the terms of the EULA
# is strictly prohibited.
cimport cuda.ccuda as ccuda
cimport cuda._lib.utils as utils

cdef class CUcontext:
    """

    CUDA context

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef ccuda.CUcontext  _val
    cdef ccuda.CUcontext* _ptr

cdef class CUmodule:
    """

    CUDA module

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef ccuda.CUmodule  _val
    cdef ccuda.CUmodule* _ptr

cdef class CUfunction:
    """

    CUDA function

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef ccuda.CUfunction  _val
    cdef ccuda.CUfunction* _ptr

cdef class CUarray:
    """

    CUDA array

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef ccuda.CUarray  _val
    cdef ccuda.CUarray* _ptr

cdef class CUmipmappedArray:
    """

    CUDA mipmapped array

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef ccuda.CUmipmappedArray  _val
    cdef ccuda.CUmipmappedArray* _ptr

cdef class CUtexref:
    """

    CUDA texture reference

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef ccuda.CUtexref  _val
    cdef ccuda.CUtexref* _ptr

cdef class CUsurfref:
    """

    CUDA surface reference

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef ccuda.CUsurfref  _val
    cdef ccuda.CUsurfref* _ptr

cdef class CUevent:
    """

    CUDA event

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef ccuda.CUevent  _val
    cdef ccuda.CUevent* _ptr

cdef class CUstream:
    """

    CUDA stream

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef ccuda.CUstream  _val
    cdef ccuda.CUstream* _ptr

cdef class CUgraphicsResource:
    """

    CUDA graphics interop resource

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef ccuda.CUgraphicsResource  _val
    cdef ccuda.CUgraphicsResource* _ptr

cdef class CUexternalMemory:
    """

    CUDA external memory

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef ccuda.CUexternalMemory  _val
    cdef ccuda.CUexternalMemory* _ptr

cdef class CUexternalSemaphore:
    """

    CUDA external semaphore

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef ccuda.CUexternalSemaphore  _val
    cdef ccuda.CUexternalSemaphore* _ptr

cdef class CUgraph:
    """

    CUDA graph

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef ccuda.CUgraph  _val
    cdef ccuda.CUgraph* _ptr

cdef class CUgraphNode:
    """

    CUDA graph node

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef ccuda.CUgraphNode  _val
    cdef ccuda.CUgraphNode* _ptr

cdef class CUgraphExec:
    """

    CUDA executable graph

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef ccuda.CUgraphExec  _val
    cdef ccuda.CUgraphExec* _ptr

cdef class CUmemoryPool:
    """

    CUDA memory pool

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef ccuda.CUmemoryPool  _val
    cdef ccuda.CUmemoryPool* _ptr

cdef class CUuserObject:
    """

    CUDA user object for graphs

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef ccuda.CUuserObject  _val
    cdef ccuda.CUuserObject* _ptr

cdef class CUlinkState:
    """

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef ccuda.CUlinkState  _val
    cdef ccuda.CUlinkState* _ptr
    cdef list _keepalive

cdef class EGLImageKHR:
    """

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef ccuda.EGLImageKHR  _val
    cdef ccuda.EGLImageKHR* _ptr

cdef class EGLStreamKHR:
    """

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef ccuda.EGLStreamKHR  _val
    cdef ccuda.EGLStreamKHR* _ptr

cdef class EGLSyncKHR:
    """

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef ccuda.EGLSyncKHR  _val
    cdef ccuda.EGLSyncKHR* _ptr

cdef class CUeglStreamConnection:
    """

    CUDA EGLSream Connection

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef ccuda.CUeglStreamConnection  _val
    cdef ccuda.CUeglStreamConnection* _ptr

cdef class CUhostFn:
    """

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef ccuda.CUhostFn  _val
    cdef ccuda.CUhostFn* _ptr

cdef class CUstreamCallback:
    """

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef ccuda.CUstreamCallback  _val
    cdef ccuda.CUstreamCallback* _ptr

cdef class CUoccupancyB2DSize:
    """

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef ccuda.CUoccupancyB2DSize  _val
    cdef ccuda.CUoccupancyB2DSize* _ptr

cdef class CUuuid_st:
    """

    Attributes
    ----------
    bytes : bytes
        < CUDA definition of UUID

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef ccuda.CUuuid_st  _val
    cdef ccuda.CUuuid_st* _ptr

cdef class CUipcEventHandle_st:
    """
    CUDA IPC event handle

    Attributes
    ----------
    reserved : bytes


    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef ccuda.CUipcEventHandle_st  _val
    cdef ccuda.CUipcEventHandle_st* _ptr

cdef class CUipcMemHandle_st:
    """
    CUDA IPC mem handle

    Attributes
    ----------
    reserved : bytes


    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef ccuda.CUipcMemHandle_st  _val
    cdef ccuda.CUipcMemHandle_st* _ptr

cdef class CUstreamMemOpWaitValueParams_st:
    """

    Attributes
    ----------
    operation : CUstreamBatchMemOpType

    address : CUdeviceptr

    flags : unsigned int

    value64 : cuuint64_t

    alias : CUdeviceptr
        For driver internal use. Initial value is unimportant.

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef ccuda.CUstreamMemOpWaitValueParams_st  _val
    cdef ccuda.CUstreamMemOpWaitValueParams_st* _ptr
    cdef CUdeviceptr _address
    cdef cuuint64_t _value64
    cdef CUdeviceptr _alias

cdef class CUstreamMemOpWriteValueParams_st:
    """

    Attributes
    ----------
    operation : CUstreamBatchMemOpType

    address : CUdeviceptr

    flags : unsigned int

    value64 : cuuint64_t

    alias : CUdeviceptr
        For driver internal use. Initial value is unimportant.

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef ccuda.CUstreamMemOpWriteValueParams_st  _val
    cdef ccuda.CUstreamMemOpWriteValueParams_st* _ptr
    cdef CUdeviceptr _address
    cdef cuuint64_t _value64
    cdef CUdeviceptr _alias

cdef class CUstreamMemOpFlushRemoteWritesParams_st:
    """

    Attributes
    ----------
    operation : CUstreamBatchMemOpType

    flags : unsigned int


    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef ccuda.CUstreamMemOpFlushRemoteWritesParams_st  _val
    cdef ccuda.CUstreamMemOpFlushRemoteWritesParams_st* _ptr

cdef class CUstreamMemOpMemoryBarrierParams_st:
    """

    Attributes
    ----------
    operation : CUstreamBatchMemOpType
        < Only supported in the _v2 API
    flags : unsigned int


    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef ccuda.CUstreamMemOpMemoryBarrierParams_st  _val
    cdef ccuda.CUstreamMemOpMemoryBarrierParams_st* _ptr

cdef class CUstreamBatchMemOpParams_union:
    """
    Per-operation parameters for cuStreamBatchMemOp

    Attributes
    ----------
    operation : CUstreamBatchMemOpType

    waitValue : CUstreamMemOpWaitValueParams_st

    writeValue : CUstreamMemOpWriteValueParams_st

    flushRemoteWrites : CUstreamMemOpFlushRemoteWritesParams_st

    memoryBarrier : CUstreamMemOpMemoryBarrierParams_st

    pad : List[cuuint64_t]


    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef ccuda.CUstreamBatchMemOpParams_union  _val
    cdef ccuda.CUstreamBatchMemOpParams_union* _ptr
    cdef CUstreamMemOpWaitValueParams_st _waitValue
    cdef CUstreamMemOpWriteValueParams_st _writeValue
    cdef CUstreamMemOpFlushRemoteWritesParams_st _flushRemoteWrites
    cdef CUstreamMemOpMemoryBarrierParams_st _memoryBarrier

cdef class CUDA_BATCH_MEM_OP_NODE_PARAMS_st:
    """

    Attributes
    ----------
    ctx : CUcontext

    count : unsigned int

    paramArray : CUstreamBatchMemOpParams

    flags : unsigned int


    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef ccuda.CUDA_BATCH_MEM_OP_NODE_PARAMS_st  _val
    cdef ccuda.CUDA_BATCH_MEM_OP_NODE_PARAMS_st* _ptr
    cdef CUcontext _ctx
    cdef size_t _paramArray_length
    cdef ccuda.CUstreamBatchMemOpParams* _paramArray

cdef class CUdevprop_st:
    """
    Legacy device properties

    Attributes
    ----------
    maxThreadsPerBlock : int
        Maximum number of threads per block
    maxThreadsDim : List[int]
        Maximum size of each dimension of a block
    maxGridSize : List[int]
        Maximum size of each dimension of a grid
    sharedMemPerBlock : int
        Shared memory available per block in bytes
    totalConstantMemory : int
        Constant memory available on device in bytes
    SIMDWidth : int
        Warp size in threads
    memPitch : int
        Maximum pitch in bytes allowed by memory copies
    regsPerBlock : int
        32-bit registers available per block
    clockRate : int
        Clock frequency in kilohertz
    textureAlign : int
        Alignment requirement for textures

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef ccuda.CUdevprop_st  _val
    cdef ccuda.CUdevprop_st* _ptr

cdef class CUaccessPolicyWindow_st:
    """
    Specifies an access policy for a window, a contiguous extent of
    memory beginning at base_ptr and ending at base_ptr + num_bytes.
    num_bytes is limited by
    CU_DEVICE_ATTRIBUTE_MAX_ACCESS_POLICY_WINDOW_SIZE. Partition into
    many segments and assign segments such that: sum of "hit segments"
    / window == approx. ratio. sum of "miss segments" / window ==
    approx 1-ratio. Segments and ratio specifications are fitted to the
    capabilities of the architecture. Accesses in a hit segment apply
    the hitProp access policy. Accesses in a miss segment apply the
    missProp access policy.

    Attributes
    ----------
    base_ptr : Any
        Starting address of the access policy window. CUDA driver may align
        it.
    num_bytes : size_t
        Size in bytes of the window policy. CUDA driver may restrict the
        maximum size and alignment.
    hitRatio : float
        hitRatio specifies percentage of lines assigned hitProp, rest are
        assigned missProp.
    hitProp : CUaccessProperty
        CUaccessProperty set for hit.
    missProp : CUaccessProperty
        CUaccessProperty set for miss. Must be either NORMAL or STREAMING

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef ccuda.CUaccessPolicyWindow_st  _val
    cdef ccuda.CUaccessPolicyWindow_st* _ptr
    cdef utils.HelperInputVoidPtr _cbase_ptr

cdef class CUDA_KERNEL_NODE_PARAMS_st:
    """
    GPU kernel node parameters

    Attributes
    ----------
    func : CUfunction
        Kernel to launch
    gridDimX : unsigned int
        Width of grid in blocks
    gridDimY : unsigned int
        Height of grid in blocks
    gridDimZ : unsigned int
        Depth of grid in blocks
    blockDimX : unsigned int
        X dimension of each thread block
    blockDimY : unsigned int
        Y dimension of each thread block
    blockDimZ : unsigned int
        Z dimension of each thread block
    sharedMemBytes : unsigned int
        Dynamic shared-memory size per thread block in bytes
    kernelParams : Any
        Array of pointers to kernel parameters
    extra : Any
        Extra options

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef ccuda.CUDA_KERNEL_NODE_PARAMS_st  _val
    cdef ccuda.CUDA_KERNEL_NODE_PARAMS_st* _ptr
    cdef CUfunction _func
    cdef utils.HelperKernelParams _ckernelParams

cdef class CUDA_MEMSET_NODE_PARAMS_st:
    """
    Memset node parameters

    Attributes
    ----------
    dst : CUdeviceptr
        Destination device pointer
    pitch : size_t
        Pitch of destination device pointer. Unused if height is 1
    value : unsigned int
        Value to be set
    elementSize : unsigned int
        Size of each element in bytes. Must be 1, 2, or 4.
    width : size_t
        Width of the row in elements
    height : size_t
        Number of rows

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef ccuda.CUDA_MEMSET_NODE_PARAMS_st  _val
    cdef ccuda.CUDA_MEMSET_NODE_PARAMS_st* _ptr
    cdef CUdeviceptr _dst

cdef class CUDA_HOST_NODE_PARAMS_st:
    """
    Host node parameters

    Attributes
    ----------
    fn : CUhostFn
        The function to call when the node executes
    userData : Any
        Argument to pass to the function

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef ccuda.CUDA_HOST_NODE_PARAMS_st  _val
    cdef ccuda.CUDA_HOST_NODE_PARAMS_st* _ptr
    cdef CUhostFn _fn
    cdef utils.HelperInputVoidPtr _cuserData

cdef class CUkernelNodeAttrValue_union:
    """
    Graph kernel node attributes union, used with
    ::cuKernelNodeSetAttribute/::cuKernelNodeGetAttribute

    Attributes
    ----------
    accessPolicyWindow : CUaccessPolicyWindow
        Attribute ::CUaccessPolicyWindow.
    cooperative : int
        Nonzero indicates a cooperative kernel (see
        cuLaunchCooperativeKernel).
    priority : int
        Execution priority of the kernel.

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef ccuda.CUkernelNodeAttrValue_union  _val
    cdef ccuda.CUkernelNodeAttrValue_union* _ptr
    cdef CUaccessPolicyWindow _accessPolicyWindow

cdef class CUstreamAttrValue_union:
    """
    Stream attributes union, used with
    cuStreamSetAttribute/cuStreamGetAttribute

    Attributes
    ----------
    accessPolicyWindow : CUaccessPolicyWindow
        Attribute ::CUaccessPolicyWindow.
    syncPolicy : CUsynchronizationPolicy
        Value for CU_STREAM_ATTRIBUTE_SYNCHRONIZATION_POLICY.

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef ccuda.CUstreamAttrValue_union  _val
    cdef ccuda.CUstreamAttrValue_union* _ptr
    cdef CUaccessPolicyWindow _accessPolicyWindow

cdef class CUexecAffinitySmCount_st:
    """
    Value for CU_EXEC_AFFINITY_TYPE_SM_COUNT

    Attributes
    ----------
    val : unsigned int
        The number of SMs the context is limited to use.

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef ccuda.CUexecAffinitySmCount_st  _val
    cdef ccuda.CUexecAffinitySmCount_st* _ptr

cdef class _CUexecAffinityParam_v1_CUexecAffinityParam_v1_CUexecAffinityParam_st_param_u:
    """

    Attributes
    ----------
    smCount : CUexecAffinitySmCount


    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef ccuda.CUexecAffinityParam_st* _ptr
    cdef CUexecAffinitySmCount _smCount

cdef class CUexecAffinityParam_st:
    """
    Execution Affinity Parameters

    Attributes
    ----------
    type : CUexecAffinityType

    param : _CUexecAffinityParam_v1_CUexecAffinityParam_v1_CUexecAffinityParam_st_param_u


    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef ccuda.CUexecAffinityParam_st  _val
    cdef ccuda.CUexecAffinityParam_st* _ptr
    cdef _CUexecAffinityParam_v1_CUexecAffinityParam_v1_CUexecAffinityParam_st_param_u _param

cdef class CUDA_MEMCPY2D_st:
    """
    2D memory copy parameters

    Attributes
    ----------
    srcXInBytes : size_t
        Source X in bytes
    srcY : size_t
        Source Y
    srcMemoryType : CUmemorytype
        Source memory type (host, device, array)
    srcHost : Any
        Source host pointer
    srcDevice : CUdeviceptr
        Source device pointer
    srcArray : CUarray
        Source array reference
    srcPitch : size_t
        Source pitch (ignored when src is array)
    dstXInBytes : size_t
        Destination X in bytes
    dstY : size_t
        Destination Y
    dstMemoryType : CUmemorytype
        Destination memory type (host, device, array)
    dstHost : Any
        Destination host pointer
    dstDevice : CUdeviceptr
        Destination device pointer
    dstArray : CUarray
        Destination array reference
    dstPitch : size_t
        Destination pitch (ignored when dst is array)
    WidthInBytes : size_t
        Width of 2D memory copy in bytes
    Height : size_t
        Height of 2D memory copy

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef ccuda.CUDA_MEMCPY2D_st  _val
    cdef ccuda.CUDA_MEMCPY2D_st* _ptr
    cdef utils.HelperInputVoidPtr _csrcHost
    cdef CUdeviceptr _srcDevice
    cdef CUarray _srcArray
    cdef utils.HelperInputVoidPtr _cdstHost
    cdef CUdeviceptr _dstDevice
    cdef CUarray _dstArray

cdef class CUDA_MEMCPY3D_st:
    """
    3D memory copy parameters

    Attributes
    ----------
    srcXInBytes : size_t
        Source X in bytes
    srcY : size_t
        Source Y
    srcZ : size_t
        Source Z
    srcLOD : size_t
        Source LOD
    srcMemoryType : CUmemorytype
        Source memory type (host, device, array)
    srcHost : Any
        Source host pointer
    srcDevice : CUdeviceptr
        Source device pointer
    srcArray : CUarray
        Source array reference
    reserved0 : Any
        Must be NULL
    srcPitch : size_t
        Source pitch (ignored when src is array)
    srcHeight : size_t
        Source height (ignored when src is array; may be 0 if Depth==1)
    dstXInBytes : size_t
        Destination X in bytes
    dstY : size_t
        Destination Y
    dstZ : size_t
        Destination Z
    dstLOD : size_t
        Destination LOD
    dstMemoryType : CUmemorytype
        Destination memory type (host, device, array)
    dstHost : Any
        Destination host pointer
    dstDevice : CUdeviceptr
        Destination device pointer
    dstArray : CUarray
        Destination array reference
    reserved1 : Any
        Must be NULL
    dstPitch : size_t
        Destination pitch (ignored when dst is array)
    dstHeight : size_t
        Destination height (ignored when dst is array; may be 0 if
        Depth==1)
    WidthInBytes : size_t
        Width of 3D memory copy in bytes
    Height : size_t
        Height of 3D memory copy
    Depth : size_t
        Depth of 3D memory copy

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef ccuda.CUDA_MEMCPY3D_st  _val
    cdef ccuda.CUDA_MEMCPY3D_st* _ptr
    cdef utils.HelperInputVoidPtr _csrcHost
    cdef CUdeviceptr _srcDevice
    cdef CUarray _srcArray
    cdef utils.HelperInputVoidPtr _creserved0
    cdef utils.HelperInputVoidPtr _cdstHost
    cdef CUdeviceptr _dstDevice
    cdef CUarray _dstArray
    cdef utils.HelperInputVoidPtr _creserved1

cdef class CUDA_MEMCPY3D_PEER_st:
    """
    3D memory cross-context copy parameters

    Attributes
    ----------
    srcXInBytes : size_t
        Source X in bytes
    srcY : size_t
        Source Y
    srcZ : size_t
        Source Z
    srcLOD : size_t
        Source LOD
    srcMemoryType : CUmemorytype
        Source memory type (host, device, array)
    srcHost : Any
        Source host pointer
    srcDevice : CUdeviceptr
        Source device pointer
    srcArray : CUarray
        Source array reference
    srcContext : CUcontext
        Source context (ignored with srcMemoryType is CU_MEMORYTYPE_ARRAY)
    srcPitch : size_t
        Source pitch (ignored when src is array)
    srcHeight : size_t
        Source height (ignored when src is array; may be 0 if Depth==1)
    dstXInBytes : size_t
        Destination X in bytes
    dstY : size_t
        Destination Y
    dstZ : size_t
        Destination Z
    dstLOD : size_t
        Destination LOD
    dstMemoryType : CUmemorytype
        Destination memory type (host, device, array)
    dstHost : Any
        Destination host pointer
    dstDevice : CUdeviceptr
        Destination device pointer
    dstArray : CUarray
        Destination array reference
    dstContext : CUcontext
        Destination context (ignored with dstMemoryType is
        CU_MEMORYTYPE_ARRAY)
    dstPitch : size_t
        Destination pitch (ignored when dst is array)
    dstHeight : size_t
        Destination height (ignored when dst is array; may be 0 if
        Depth==1)
    WidthInBytes : size_t
        Width of 3D memory copy in bytes
    Height : size_t
        Height of 3D memory copy
    Depth : size_t
        Depth of 3D memory copy

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef ccuda.CUDA_MEMCPY3D_PEER_st  _val
    cdef ccuda.CUDA_MEMCPY3D_PEER_st* _ptr
    cdef utils.HelperInputVoidPtr _csrcHost
    cdef CUdeviceptr _srcDevice
    cdef CUarray _srcArray
    cdef CUcontext _srcContext
    cdef utils.HelperInputVoidPtr _cdstHost
    cdef CUdeviceptr _dstDevice
    cdef CUarray _dstArray
    cdef CUcontext _dstContext

cdef class CUDA_ARRAY_DESCRIPTOR_st:
    """
    Array descriptor

    Attributes
    ----------
    Width : size_t
        Width of array
    Height : size_t
        Height of array
    Format : CUarray_format
        Array format
    NumChannels : unsigned int
        Channels per array element

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef ccuda.CUDA_ARRAY_DESCRIPTOR_st  _val
    cdef ccuda.CUDA_ARRAY_DESCRIPTOR_st* _ptr

cdef class CUDA_ARRAY3D_DESCRIPTOR_st:
    """
    3D array descriptor

    Attributes
    ----------
    Width : size_t
        Width of 3D array
    Height : size_t
        Height of 3D array
    Depth : size_t
        Depth of 3D array
    Format : CUarray_format
        Array format
    NumChannels : unsigned int
        Channels per array element
    Flags : unsigned int
        Flags

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef ccuda.CUDA_ARRAY3D_DESCRIPTOR_st  _val
    cdef ccuda.CUDA_ARRAY3D_DESCRIPTOR_st* _ptr

cdef class _CUDA_ARRAY_SPARSE_PROPERTIES_v1_CUDA_ARRAY_SPARSE_PROPERTIES_v1_CUDA_ARRAY_SPARSE_PROPERTIES_st_tileExtent_s:
    """

    Attributes
    ----------
    width : unsigned int

    height : unsigned int

    depth : unsigned int


    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef ccuda.CUDA_ARRAY_SPARSE_PROPERTIES_st* _ptr

cdef class CUDA_ARRAY_SPARSE_PROPERTIES_st:
    """
    CUDA array sparse properties

    Attributes
    ----------
    tileExtent : _CUDA_ARRAY_SPARSE_PROPERTIES_v1_CUDA_ARRAY_SPARSE_PROPERTIES_v1_CUDA_ARRAY_SPARSE_PROPERTIES_st_tileExtent_s

    miptailFirstLevel : unsigned int
        First mip level at which the mip tail begins.
    miptailSize : unsigned long long
        Total size of the mip tail.
    flags : unsigned int
        Flags will either be zero or
        CU_ARRAY_SPARSE_PROPERTIES_SINGLE_MIPTAIL
    reserved : List[unsigned int]


    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef ccuda.CUDA_ARRAY_SPARSE_PROPERTIES_st  _val
    cdef ccuda.CUDA_ARRAY_SPARSE_PROPERTIES_st* _ptr
    cdef _CUDA_ARRAY_SPARSE_PROPERTIES_v1_CUDA_ARRAY_SPARSE_PROPERTIES_v1_CUDA_ARRAY_SPARSE_PROPERTIES_st_tileExtent_s _tileExtent

cdef class CUDA_ARRAY_MEMORY_REQUIREMENTS_st:
    """
    CUDA array memory requirements

    Attributes
    ----------
    size : size_t
        Total required memory size
    alignment : size_t
        alignment requirement
    reserved : List[unsigned int]


    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef ccuda.CUDA_ARRAY_MEMORY_REQUIREMENTS_st  _val
    cdef ccuda.CUDA_ARRAY_MEMORY_REQUIREMENTS_st* _ptr

cdef class _CUDA_RESOURCE_DESC_v1_CUDA_RESOURCE_DESC_v1_CUDA_RESOURCE_DESC_st_res_res_array_s:
    """

    Attributes
    ----------
    hArray : CUarray


    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef ccuda.CUDA_RESOURCE_DESC_st* _ptr
    cdef CUarray _hArray

cdef class _CUDA_RESOURCE_DESC_v1_CUDA_RESOURCE_DESC_v1_CUDA_RESOURCE_DESC_st_res_res_mipmap_s:
    """

    Attributes
    ----------
    hMipmappedArray : CUmipmappedArray


    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef ccuda.CUDA_RESOURCE_DESC_st* _ptr
    cdef CUmipmappedArray _hMipmappedArray

cdef class _CUDA_RESOURCE_DESC_v1_CUDA_RESOURCE_DESC_v1_CUDA_RESOURCE_DESC_st_res_res_linear_s:
    """

    Attributes
    ----------
    devPtr : CUdeviceptr

    format : CUarray_format

    numChannels : unsigned int

    sizeInBytes : size_t


    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef ccuda.CUDA_RESOURCE_DESC_st* _ptr
    cdef CUdeviceptr _devPtr

cdef class _CUDA_RESOURCE_DESC_v1_CUDA_RESOURCE_DESC_v1_CUDA_RESOURCE_DESC_st_res_res_pitch2D_s:
    """

    Attributes
    ----------
    devPtr : CUdeviceptr

    format : CUarray_format

    numChannels : unsigned int

    width : size_t

    height : size_t

    pitchInBytes : size_t


    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef ccuda.CUDA_RESOURCE_DESC_st* _ptr
    cdef CUdeviceptr _devPtr

cdef class _CUDA_RESOURCE_DESC_v1_CUDA_RESOURCE_DESC_v1_CUDA_RESOURCE_DESC_st_res_res_reserved_s:
    """

    Attributes
    ----------
    reserved : List[int]


    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef ccuda.CUDA_RESOURCE_DESC_st* _ptr

cdef class _CUDA_RESOURCE_DESC_v1_CUDA_RESOURCE_DESC_v1_CUDA_RESOURCE_DESC_st_res_u:
    """

    Attributes
    ----------
    array : _CUDA_RESOURCE_DESC_v1_CUDA_RESOURCE_DESC_v1_CUDA_RESOURCE_DESC_st_res_res_array_s

    mipmap : _CUDA_RESOURCE_DESC_v1_CUDA_RESOURCE_DESC_v1_CUDA_RESOURCE_DESC_st_res_res_mipmap_s

    linear : _CUDA_RESOURCE_DESC_v1_CUDA_RESOURCE_DESC_v1_CUDA_RESOURCE_DESC_st_res_res_linear_s

    pitch2D : _CUDA_RESOURCE_DESC_v1_CUDA_RESOURCE_DESC_v1_CUDA_RESOURCE_DESC_st_res_res_pitch2D_s

    reserved : _CUDA_RESOURCE_DESC_v1_CUDA_RESOURCE_DESC_v1_CUDA_RESOURCE_DESC_st_res_res_reserved_s


    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef ccuda.CUDA_RESOURCE_DESC_st* _ptr
    cdef _CUDA_RESOURCE_DESC_v1_CUDA_RESOURCE_DESC_v1_CUDA_RESOURCE_DESC_st_res_res_array_s _array
    cdef _CUDA_RESOURCE_DESC_v1_CUDA_RESOURCE_DESC_v1_CUDA_RESOURCE_DESC_st_res_res_mipmap_s _mipmap
    cdef _CUDA_RESOURCE_DESC_v1_CUDA_RESOURCE_DESC_v1_CUDA_RESOURCE_DESC_st_res_res_linear_s _linear
    cdef _CUDA_RESOURCE_DESC_v1_CUDA_RESOURCE_DESC_v1_CUDA_RESOURCE_DESC_st_res_res_pitch2D_s _pitch2D
    cdef _CUDA_RESOURCE_DESC_v1_CUDA_RESOURCE_DESC_v1_CUDA_RESOURCE_DESC_st_res_res_reserved_s _reserved

cdef class CUDA_RESOURCE_DESC_st:
    """
    CUDA Resource descriptor

    Attributes
    ----------
    resType : CUresourcetype
        Resource type
    res : _CUDA_RESOURCE_DESC_v1_CUDA_RESOURCE_DESC_v1_CUDA_RESOURCE_DESC_st_res_u

    flags : unsigned int
        Flags (must be zero)

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef ccuda.CUDA_RESOURCE_DESC_st  _val
    cdef ccuda.CUDA_RESOURCE_DESC_st* _ptr
    cdef _CUDA_RESOURCE_DESC_v1_CUDA_RESOURCE_DESC_v1_CUDA_RESOURCE_DESC_st_res_u _res

cdef class CUDA_TEXTURE_DESC_st:
    """
    Texture descriptor

    Attributes
    ----------
    addressMode : List[CUaddress_mode]
        Address modes
    filterMode : CUfilter_mode
        Filter mode
    flags : unsigned int
        Flags
    maxAnisotropy : unsigned int
        Maximum anisotropy ratio
    mipmapFilterMode : CUfilter_mode
        Mipmap filter mode
    mipmapLevelBias : float
        Mipmap level bias
    minMipmapLevelClamp : float
        Mipmap minimum level clamp
    maxMipmapLevelClamp : float
        Mipmap maximum level clamp
    borderColor : List[float]
        Border Color
    reserved : List[int]


    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef ccuda.CUDA_TEXTURE_DESC_st  _val
    cdef ccuda.CUDA_TEXTURE_DESC_st* _ptr

cdef class CUDA_RESOURCE_VIEW_DESC_st:
    """
    Resource view descriptor

    Attributes
    ----------
    format : CUresourceViewFormat
        Resource view format
    width : size_t
        Width of the resource view
    height : size_t
        Height of the resource view
    depth : size_t
        Depth of the resource view
    firstMipmapLevel : unsigned int
        First defined mipmap level
    lastMipmapLevel : unsigned int
        Last defined mipmap level
    firstLayer : unsigned int
        First layer index
    lastLayer : unsigned int
        Last layer index
    reserved : List[unsigned int]


    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef ccuda.CUDA_RESOURCE_VIEW_DESC_st  _val
    cdef ccuda.CUDA_RESOURCE_VIEW_DESC_st* _ptr

cdef class CUDA_POINTER_ATTRIBUTE_P2P_TOKENS_st:
    """
    GPU Direct v3 tokens

    Attributes
    ----------
    p2pToken : unsigned long long

    vaSpaceToken : unsigned int


    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef ccuda.CUDA_POINTER_ATTRIBUTE_P2P_TOKENS_st  _val
    cdef ccuda.CUDA_POINTER_ATTRIBUTE_P2P_TOKENS_st* _ptr

cdef class CUDA_LAUNCH_PARAMS_st:
    """
    Kernel launch parameters

    Attributes
    ----------
    function : CUfunction
        Kernel to launch
    gridDimX : unsigned int
        Width of grid in blocks
    gridDimY : unsigned int
        Height of grid in blocks
    gridDimZ : unsigned int
        Depth of grid in blocks
    blockDimX : unsigned int
        X dimension of each thread block
    blockDimY : unsigned int
        Y dimension of each thread block
    blockDimZ : unsigned int
        Z dimension of each thread block
    sharedMemBytes : unsigned int
        Dynamic shared-memory size per thread block in bytes
    hStream : CUstream
        Stream identifier
    kernelParams : Any
        Array of pointers to kernel parameters

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef ccuda.CUDA_LAUNCH_PARAMS_st  _val
    cdef ccuda.CUDA_LAUNCH_PARAMS_st* _ptr
    cdef CUfunction _function
    cdef CUstream _hStream
    cdef utils.HelperKernelParams _ckernelParams

cdef class _CUDA_EXTERNAL_MEMORY_HANDLE_DESC_v1_CUDA_EXTERNAL_MEMORY_HANDLE_DESC_v1_CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st_handle_handle_win32_s:
    """

    Attributes
    ----------
    handle : void

    name : void


    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef ccuda.CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st* _ptr
    cdef utils.HelperInputVoidPtr _chandle
    cdef utils.HelperInputVoidPtr _cname

cdef class _CUDA_EXTERNAL_MEMORY_HANDLE_DESC_v1_CUDA_EXTERNAL_MEMORY_HANDLE_DESC_v1_CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st_handle_u:
    """

    Attributes
    ----------
    fd : int

    win32 : _CUDA_EXTERNAL_MEMORY_HANDLE_DESC_v1_CUDA_EXTERNAL_MEMORY_HANDLE_DESC_v1_CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st_handle_handle_win32_s

    nvSciBufObject : void


    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef ccuda.CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st* _ptr
    cdef _CUDA_EXTERNAL_MEMORY_HANDLE_DESC_v1_CUDA_EXTERNAL_MEMORY_HANDLE_DESC_v1_CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st_handle_handle_win32_s _win32
    cdef utils.HelperInputVoidPtr _cnvSciBufObject

cdef class CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st:
    """
    External memory handle descriptor

    Attributes
    ----------
    type : CUexternalMemoryHandleType
        Type of the handle
    handle : _CUDA_EXTERNAL_MEMORY_HANDLE_DESC_v1_CUDA_EXTERNAL_MEMORY_HANDLE_DESC_v1_CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st_handle_u

    size : unsigned long long
        Size of the memory allocation
    flags : unsigned int
        Flags must either be zero or CUDA_EXTERNAL_MEMORY_DEDICATED
    reserved : List[unsigned int]


    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef ccuda.CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st  _val
    cdef ccuda.CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st* _ptr
    cdef _CUDA_EXTERNAL_MEMORY_HANDLE_DESC_v1_CUDA_EXTERNAL_MEMORY_HANDLE_DESC_v1_CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st_handle_u _handle

cdef class CUDA_EXTERNAL_MEMORY_BUFFER_DESC_st:
    """
    External memory buffer descriptor

    Attributes
    ----------
    offset : unsigned long long
        Offset into the memory object where the buffer's base is
    size : unsigned long long
        Size of the buffer
    flags : unsigned int
        Flags reserved for future use. Must be zero.
    reserved : List[unsigned int]


    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef ccuda.CUDA_EXTERNAL_MEMORY_BUFFER_DESC_st  _val
    cdef ccuda.CUDA_EXTERNAL_MEMORY_BUFFER_DESC_st* _ptr

cdef class CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC_st:
    """
    External memory mipmap descriptor

    Attributes
    ----------
    offset : unsigned long long
        Offset into the memory object where the base level of the mipmap
        chain is.
    arrayDesc : CUDA_ARRAY3D_DESCRIPTOR
        Format, dimension and type of base level of the mipmap chain
    numLevels : unsigned int
        Total number of levels in the mipmap chain
    reserved : List[unsigned int]


    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef ccuda.CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC_st  _val
    cdef ccuda.CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC_st* _ptr
    cdef CUDA_ARRAY3D_DESCRIPTOR _arrayDesc

cdef class _CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_v1_CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_v1_CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_st_handle_handle_win32_s:
    """

    Attributes
    ----------
    handle : void

    name : void


    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef ccuda.CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_st* _ptr
    cdef utils.HelperInputVoidPtr _chandle
    cdef utils.HelperInputVoidPtr _cname

cdef class _CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_v1_CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_v1_CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_st_handle_u:
    """

    Attributes
    ----------
    fd : int

    win32 : _CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_v1_CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_v1_CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_st_handle_handle_win32_s

    nvSciSyncObj : void


    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef ccuda.CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_st* _ptr
    cdef _CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_v1_CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_v1_CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_st_handle_handle_win32_s _win32
    cdef utils.HelperInputVoidPtr _cnvSciSyncObj

cdef class CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_st:
    """
    External semaphore handle descriptor

    Attributes
    ----------
    type : CUexternalSemaphoreHandleType
        Type of the handle
    handle : _CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_v1_CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_v1_CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_st_handle_u

    flags : unsigned int
        Flags reserved for the future. Must be zero.
    reserved : List[unsigned int]


    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef ccuda.CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_st  _val
    cdef ccuda.CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_st* _ptr
    cdef _CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_v1_CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_v1_CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_st_handle_u _handle

cdef class _CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_v1_CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_v1_CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st_params_params_fence_s:
    """

    Attributes
    ----------
    value : unsigned long long


    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef ccuda.CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st* _ptr

cdef class _CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_v1_CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_v1_CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st_params_params_nvSciSync_u:
    """

    Attributes
    ----------
    fence : void

    reserved : unsigned long long


    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef ccuda.CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st* _ptr
    cdef utils.HelperInputVoidPtr _cfence

cdef class _CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_v1_CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_v1_CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st_params_params_keyedMutex_s:
    """

    Attributes
    ----------
    key : unsigned long long


    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef ccuda.CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st* _ptr

cdef class _CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_v1_CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_v1_CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st_params_s:
    """

    Attributes
    ----------
    fence : _CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_v1_CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_v1_CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st_params_params_fence_s

    nvSciSync : _CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_v1_CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_v1_CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st_params_params_nvSciSync_u

    keyedMutex : _CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_v1_CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_v1_CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st_params_params_keyedMutex_s

    reserved : List[unsigned int]


    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef ccuda.CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st* _ptr
    cdef _CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_v1_CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_v1_CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st_params_params_fence_s _fence
    cdef _CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_v1_CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_v1_CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st_params_params_nvSciSync_u _nvSciSync
    cdef _CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_v1_CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_v1_CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st_params_params_keyedMutex_s _keyedMutex

cdef class CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st:
    """
    External semaphore signal parameters

    Attributes
    ----------
    params : _CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_v1_CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_v1_CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st_params_s

    flags : unsigned int
        Only when ::CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS is used to signal
        a CUexternalSemaphore of type
        CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_NVSCISYNC, the valid flag is
        CUDA_EXTERNAL_SEMAPHORE_SIGNAL_SKIP_NVSCIBUF_MEMSYNC which
        indicates that while signaling the CUexternalSemaphore, no memory
        synchronization operations should be performed for any external
        memory object imported as CU_EXTERNAL_MEMORY_HANDLE_TYPE_NVSCIBUF.
        For all other types of CUexternalSemaphore, flags must be zero.
    reserved : List[unsigned int]


    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef ccuda.CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st  _val
    cdef ccuda.CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st* _ptr
    cdef _CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_v1_CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_v1_CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st_params_s _params

cdef class _CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_v1_CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_v1_CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st_params_params_fence_s:
    """

    Attributes
    ----------
    value : unsigned long long


    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef ccuda.CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st* _ptr

cdef class _CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_v1_CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_v1_CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st_params_params_nvSciSync_u:
    """

    Attributes
    ----------
    fence : void

    reserved : unsigned long long


    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef ccuda.CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st* _ptr
    cdef utils.HelperInputVoidPtr _cfence

cdef class _CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_v1_CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_v1_CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st_params_params_keyedMutex_s:
    """

    Attributes
    ----------
    key : unsigned long long

    timeoutMs : unsigned int


    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef ccuda.CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st* _ptr

cdef class _CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_v1_CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_v1_CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st_params_s:
    """

    Attributes
    ----------
    fence : _CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_v1_CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_v1_CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st_params_params_fence_s

    nvSciSync : _CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_v1_CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_v1_CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st_params_params_nvSciSync_u

    keyedMutex : _CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_v1_CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_v1_CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st_params_params_keyedMutex_s

    reserved : List[unsigned int]


    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef ccuda.CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st* _ptr
    cdef _CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_v1_CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_v1_CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st_params_params_fence_s _fence
    cdef _CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_v1_CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_v1_CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st_params_params_nvSciSync_u _nvSciSync
    cdef _CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_v1_CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_v1_CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st_params_params_keyedMutex_s _keyedMutex

cdef class CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st:
    """
    External semaphore wait parameters

    Attributes
    ----------
    params : _CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_v1_CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_v1_CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st_params_s

    flags : unsigned int
        Only when ::CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS is used to wait on
        a CUexternalSemaphore of type
        CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_NVSCISYNC, the valid flag is
        CUDA_EXTERNAL_SEMAPHORE_WAIT_SKIP_NVSCIBUF_MEMSYNC which indicates
        that while waiting for the CUexternalSemaphore, no memory
        synchronization operations should be performed for any external
        memory object imported as CU_EXTERNAL_MEMORY_HANDLE_TYPE_NVSCIBUF.
        For all other types of CUexternalSemaphore, flags must be zero.
    reserved : List[unsigned int]


    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef ccuda.CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st  _val
    cdef ccuda.CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st* _ptr
    cdef _CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_v1_CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_v1_CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st_params_s _params

cdef class CUDA_EXT_SEM_SIGNAL_NODE_PARAMS_st:
    """
    Semaphore signal node parameters

    Attributes
    ----------
    extSemArray : CUexternalSemaphore
        Array of external semaphore handles.
    paramsArray : CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS
        Array of external semaphore signal parameters.
    numExtSems : unsigned int
        Number of handles and parameters supplied in extSemArray and
        paramsArray.

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef ccuda.CUDA_EXT_SEM_SIGNAL_NODE_PARAMS_st  _val
    cdef ccuda.CUDA_EXT_SEM_SIGNAL_NODE_PARAMS_st* _ptr
    cdef size_t _extSemArray_length
    cdef ccuda.CUexternalSemaphore* _extSemArray
    cdef size_t _paramsArray_length
    cdef ccuda.CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS* _paramsArray

cdef class CUDA_EXT_SEM_WAIT_NODE_PARAMS_st:
    """
    Semaphore wait node parameters

    Attributes
    ----------
    extSemArray : CUexternalSemaphore
        Array of external semaphore handles.
    paramsArray : CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS
        Array of external semaphore wait parameters.
    numExtSems : unsigned int
        Number of handles and parameters supplied in extSemArray and
        paramsArray.

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef ccuda.CUDA_EXT_SEM_WAIT_NODE_PARAMS_st  _val
    cdef ccuda.CUDA_EXT_SEM_WAIT_NODE_PARAMS_st* _ptr
    cdef size_t _extSemArray_length
    cdef ccuda.CUexternalSemaphore* _extSemArray
    cdef size_t _paramsArray_length
    cdef ccuda.CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS* _paramsArray

cdef class _CUarrayMapInfo_v1_CUarrayMapInfo_v1_CUarrayMapInfo_st_resource_u:
    """

    Attributes
    ----------
    mipmap : CUmipmappedArray

    array : CUarray


    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef ccuda.CUarrayMapInfo_st* _ptr
    cdef CUmipmappedArray _mipmap
    cdef CUarray _array

cdef class _CUarrayMapInfo_v1_CUarrayMapInfo_v1_CUarrayMapInfo_st_subresource_subresource_sparseLevel_s:
    """

    Attributes
    ----------
    level : unsigned int

    layer : unsigned int

    offsetX : unsigned int

    offsetY : unsigned int

    offsetZ : unsigned int

    extentWidth : unsigned int

    extentHeight : unsigned int

    extentDepth : unsigned int


    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef ccuda.CUarrayMapInfo_st* _ptr

cdef class _CUarrayMapInfo_v1_CUarrayMapInfo_v1_CUarrayMapInfo_st_subresource_subresource_miptail_s:
    """

    Attributes
    ----------
    layer : unsigned int

    offset : unsigned long long

    size : unsigned long long


    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef ccuda.CUarrayMapInfo_st* _ptr

cdef class _CUarrayMapInfo_v1_CUarrayMapInfo_v1_CUarrayMapInfo_st_subresource_u:
    """

    Attributes
    ----------
    sparseLevel : _CUarrayMapInfo_v1_CUarrayMapInfo_v1_CUarrayMapInfo_st_subresource_subresource_sparseLevel_s

    miptail : _CUarrayMapInfo_v1_CUarrayMapInfo_v1_CUarrayMapInfo_st_subresource_subresource_miptail_s


    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef ccuda.CUarrayMapInfo_st* _ptr
    cdef _CUarrayMapInfo_v1_CUarrayMapInfo_v1_CUarrayMapInfo_st_subresource_subresource_sparseLevel_s _sparseLevel
    cdef _CUarrayMapInfo_v1_CUarrayMapInfo_v1_CUarrayMapInfo_st_subresource_subresource_miptail_s _miptail

cdef class _CUarrayMapInfo_v1_CUarrayMapInfo_v1_CUarrayMapInfo_st_memHandle_u:
    """

    Attributes
    ----------
    memHandle : CUmemGenericAllocationHandle


    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef ccuda.CUarrayMapInfo_st* _ptr
    cdef CUmemGenericAllocationHandle _memHandle

cdef class CUarrayMapInfo_st:
    """
    Specifies the CUDA array or CUDA mipmapped array memory mapping
    information

    Attributes
    ----------
    resourceType : CUresourcetype
        Resource type
    resource : _CUarrayMapInfo_v1_CUarrayMapInfo_v1_CUarrayMapInfo_st_resource_u

    subresourceType : CUarraySparseSubresourceType
        Sparse subresource type
    subresource : _CUarrayMapInfo_v1_CUarrayMapInfo_v1_CUarrayMapInfo_st_subresource_u

    memOperationType : CUmemOperationType
        Memory operation type
    memHandleType : CUmemHandleType
        Memory handle type
    memHandle : _CUarrayMapInfo_v1_CUarrayMapInfo_v1_CUarrayMapInfo_st_memHandle_u

    offset : unsigned long long
        Offset within mip tail  Offset within the memory
    deviceBitMask : unsigned int
        Device ordinal bit mask
    flags : unsigned int
        flags for future use, must be zero now.
    reserved : List[unsigned int]
        Reserved for future use, must be zero now.

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef ccuda.CUarrayMapInfo_st  _val
    cdef ccuda.CUarrayMapInfo_st* _ptr
    cdef _CUarrayMapInfo_v1_CUarrayMapInfo_v1_CUarrayMapInfo_st_resource_u _resource
    cdef _CUarrayMapInfo_v1_CUarrayMapInfo_v1_CUarrayMapInfo_st_subresource_u _subresource
    cdef _CUarrayMapInfo_v1_CUarrayMapInfo_v1_CUarrayMapInfo_st_memHandle_u _memHandle

cdef class CUmemLocation_st:
    """
    Specifies a memory location.

    Attributes
    ----------
    type : CUmemLocationType
        Specifies the location type, which modifies the meaning of id.
    id : int
        identifier for a given this location's CUmemLocationType.

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef ccuda.CUmemLocation_st  _val
    cdef ccuda.CUmemLocation_st* _ptr

cdef class _CUmemAllocationProp_v1_CUmemAllocationProp_v1_CUmemAllocationProp_st_allocFlags_s:
    """

    Attributes
    ----------
    compressionType : unsigned char

    gpuDirectRDMACapable : unsigned char

    usage : unsigned short

    reserved : unsigned char


    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef ccuda.CUmemAllocationProp_st* _ptr

cdef class CUmemAllocationProp_st:
    """
    Specifies the allocation properties for a allocation.

    Attributes
    ----------
    type : CUmemAllocationType
        Allocation type
    requestedHandleTypes : CUmemAllocationHandleType
        requested CUmemAllocationHandleType
    location : CUmemLocation
        Location of allocation
    win32HandleMetaData : Any
        Windows-specific POBJECT_ATTRIBUTES required when
        CU_MEM_HANDLE_TYPE_WIN32 is specified. This object atributes
        structure includes security attributes that define the scope of
        which exported allocations may be tranferred to other processes. In
        all other cases, this field is required to be zero.
    allocFlags : _CUmemAllocationProp_v1_CUmemAllocationProp_v1_CUmemAllocationProp_st_allocFlags_s


    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef ccuda.CUmemAllocationProp_st  _val
    cdef ccuda.CUmemAllocationProp_st* _ptr
    cdef CUmemLocation _location
    cdef utils.HelperInputVoidPtr _cwin32HandleMetaData
    cdef _CUmemAllocationProp_v1_CUmemAllocationProp_v1_CUmemAllocationProp_st_allocFlags_s _allocFlags

cdef class CUmemAccessDesc_st:
    """
    Memory access descriptor

    Attributes
    ----------
    location : CUmemLocation
        Location on which the request is to change it's accessibility
    flags : CUmemAccess_flags
        ::CUmemProt accessibility flags to set on the request

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef ccuda.CUmemAccessDesc_st  _val
    cdef ccuda.CUmemAccessDesc_st* _ptr
    cdef CUmemLocation _location

cdef class CUmemPoolProps_st:
    """
    Specifies the properties of allocations made from the pool.

    Attributes
    ----------
    allocType : CUmemAllocationType
        Allocation type. Currently must be specified as
        CU_MEM_ALLOCATION_TYPE_PINNED
    handleTypes : CUmemAllocationHandleType
        Handle types that will be supported by allocations from the pool.
    location : CUmemLocation
        Location where allocations should reside.
    win32SecurityAttributes : Any
        Windows-specific LPSECURITYATTRIBUTES required when
        CU_MEM_HANDLE_TYPE_WIN32 is specified. This security attribute
        defines the scope of which exported allocations may be tranferred
        to other processes. In all other cases, this field is required to
        be zero.
    reserved : bytes
        reserved for future use, must be 0

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef ccuda.CUmemPoolProps_st  _val
    cdef ccuda.CUmemPoolProps_st* _ptr
    cdef CUmemLocation _location
    cdef utils.HelperInputVoidPtr _cwin32SecurityAttributes

cdef class CUmemPoolPtrExportData_st:
    """
    Opaque data for exporting a pool allocation

    Attributes
    ----------
    reserved : bytes


    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef ccuda.CUmemPoolPtrExportData_st  _val
    cdef ccuda.CUmemPoolPtrExportData_st* _ptr

cdef class CUDA_MEM_ALLOC_NODE_PARAMS_st:
    """
    Memory allocation node parameters

    Attributes
    ----------
    poolProps : CUmemPoolProps
        in: location where the allocation should reside (specified in
        ::location). ::handleTypes must be CU_MEM_HANDLE_TYPE_NONE. IPC is
        not supported.
    accessDescs : CUmemAccessDesc
        in: array of memory access descriptors. Used to describe peer GPU
        access
    accessDescCount : size_t
        in: number of memory access descriptors. Must not exceed the number
        of GPUs.
    bytesize : size_t
        in: size in bytes of the requested allocation
    dptr : CUdeviceptr
        out: address of the allocation returned by CUDA

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef ccuda.CUDA_MEM_ALLOC_NODE_PARAMS_st  _val
    cdef ccuda.CUDA_MEM_ALLOC_NODE_PARAMS_st* _ptr
    cdef CUmemPoolProps _poolProps
    cdef size_t _accessDescs_length
    cdef ccuda.CUmemAccessDesc* _accessDescs
    cdef CUdeviceptr _dptr

cdef class _CUeglFrame_v1_CUeglFrame_v1_CUeglFrame_st_frame_u:
    """

    Attributes
    ----------
    pArray : List[CUarray]

    pPitch : List[void]


    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef ccuda.CUeglFrame_st* _ptr

cdef class CUeglFrame_st:
    """
    CUDA EGLFrame structure Descriptor - structure defining one frame
    of EGL.  Each frame may contain one or more planes depending on
    whether the surface * is Multiplanar or not.

    Attributes
    ----------
    frame : _CUeglFrame_v1_CUeglFrame_v1_CUeglFrame_st_frame_u

    width : unsigned int
        Width of first plane
    height : unsigned int
        Height of first plane
    depth : unsigned int
        Depth of first plane
    pitch : unsigned int
        Pitch of first plane
    planeCount : unsigned int
        Number of planes
    numChannels : unsigned int
        Number of channels for the plane
    frameType : CUeglFrameType
        Array or Pitch
    eglColorFormat : CUeglColorFormat
        CUDA EGL Color Format
    cuFormat : CUarray_format
        CUDA Array Format

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef ccuda.CUeglFrame_st  _val
    cdef ccuda.CUeglFrame_st* _ptr
    cdef _CUeglFrame_v1_CUeglFrame_v1_CUeglFrame_st_frame_u _frame

cdef class CUuuid(CUuuid_st):
    """

    Attributes
    ----------
    bytes : bytes
        < CUDA definition of UUID

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    pass

cdef class CUipcEventHandle_v1(CUipcEventHandle_st):
    """
    CUDA IPC event handle

    Attributes
    ----------
    reserved : bytes


    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    pass

cdef class CUipcEventHandle(CUipcEventHandle_st):
    """
    CUDA IPC event handle

    Attributes
    ----------
    reserved : bytes


    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    pass

cdef class CUipcMemHandle_v1(CUipcMemHandle_st):
    """
    CUDA IPC mem handle

    Attributes
    ----------
    reserved : bytes


    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    pass

cdef class CUipcMemHandle(CUipcMemHandle_st):
    """
    CUDA IPC mem handle

    Attributes
    ----------
    reserved : bytes


    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    pass

cdef class CUstreamBatchMemOpParams_v1(CUstreamBatchMemOpParams_union):
    """
    Per-operation parameters for cuStreamBatchMemOp

    Attributes
    ----------
    operation : CUstreamBatchMemOpType

    waitValue : CUstreamMemOpWaitValueParams_st

    writeValue : CUstreamMemOpWriteValueParams_st

    flushRemoteWrites : CUstreamMemOpFlushRemoteWritesParams_st

    memoryBarrier : CUstreamMemOpMemoryBarrierParams_st

    pad : List[cuuint64_t]


    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    pass

cdef class CUstreamBatchMemOpParams(CUstreamBatchMemOpParams_union):
    """
    Per-operation parameters for cuStreamBatchMemOp

    Attributes
    ----------
    operation : CUstreamBatchMemOpType

    waitValue : CUstreamMemOpWaitValueParams_st

    writeValue : CUstreamMemOpWriteValueParams_st

    flushRemoteWrites : CUstreamMemOpFlushRemoteWritesParams_st

    memoryBarrier : CUstreamMemOpMemoryBarrierParams_st

    pad : List[cuuint64_t]


    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    pass

cdef class CUDA_BATCH_MEM_OP_NODE_PARAMS(CUDA_BATCH_MEM_OP_NODE_PARAMS_st):
    """

    Attributes
    ----------
    ctx : CUcontext

    count : unsigned int

    paramArray : CUstreamBatchMemOpParams

    flags : unsigned int


    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    pass

cdef class CUdevprop_v1(CUdevprop_st):
    """
    Legacy device properties

    Attributes
    ----------
    maxThreadsPerBlock : int
        Maximum number of threads per block
    maxThreadsDim : List[int]
        Maximum size of each dimension of a block
    maxGridSize : List[int]
        Maximum size of each dimension of a grid
    sharedMemPerBlock : int
        Shared memory available per block in bytes
    totalConstantMemory : int
        Constant memory available on device in bytes
    SIMDWidth : int
        Warp size in threads
    memPitch : int
        Maximum pitch in bytes allowed by memory copies
    regsPerBlock : int
        32-bit registers available per block
    clockRate : int
        Clock frequency in kilohertz
    textureAlign : int
        Alignment requirement for textures

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    pass

cdef class CUdevprop(CUdevprop_st):
    """
    Legacy device properties

    Attributes
    ----------
    maxThreadsPerBlock : int
        Maximum number of threads per block
    maxThreadsDim : List[int]
        Maximum size of each dimension of a block
    maxGridSize : List[int]
        Maximum size of each dimension of a grid
    sharedMemPerBlock : int
        Shared memory available per block in bytes
    totalConstantMemory : int
        Constant memory available on device in bytes
    SIMDWidth : int
        Warp size in threads
    memPitch : int
        Maximum pitch in bytes allowed by memory copies
    regsPerBlock : int
        32-bit registers available per block
    clockRate : int
        Clock frequency in kilohertz
    textureAlign : int
        Alignment requirement for textures

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    pass

cdef class CUaccessPolicyWindow_v1(CUaccessPolicyWindow_st):
    """
    Specifies an access policy for a window, a contiguous extent of
    memory beginning at base_ptr and ending at base_ptr + num_bytes.
    num_bytes is limited by
    CU_DEVICE_ATTRIBUTE_MAX_ACCESS_POLICY_WINDOW_SIZE. Partition into
    many segments and assign segments such that: sum of "hit segments"
    / window == approx. ratio. sum of "miss segments" / window ==
    approx 1-ratio. Segments and ratio specifications are fitted to the
    capabilities of the architecture. Accesses in a hit segment apply
    the hitProp access policy. Accesses in a miss segment apply the
    missProp access policy.

    Attributes
    ----------
    base_ptr : Any
        Starting address of the access policy window. CUDA driver may align
        it.
    num_bytes : size_t
        Size in bytes of the window policy. CUDA driver may restrict the
        maximum size and alignment.
    hitRatio : float
        hitRatio specifies percentage of lines assigned hitProp, rest are
        assigned missProp.
    hitProp : CUaccessProperty
        CUaccessProperty set for hit.
    missProp : CUaccessProperty
        CUaccessProperty set for miss. Must be either NORMAL or STREAMING

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    pass

cdef class CUaccessPolicyWindow(CUaccessPolicyWindow_st):
    """
    Specifies an access policy for a window, a contiguous extent of
    memory beginning at base_ptr and ending at base_ptr + num_bytes.
    num_bytes is limited by
    CU_DEVICE_ATTRIBUTE_MAX_ACCESS_POLICY_WINDOW_SIZE. Partition into
    many segments and assign segments such that: sum of "hit segments"
    / window == approx. ratio. sum of "miss segments" / window ==
    approx 1-ratio. Segments and ratio specifications are fitted to the
    capabilities of the architecture. Accesses in a hit segment apply
    the hitProp access policy. Accesses in a miss segment apply the
    missProp access policy.

    Attributes
    ----------
    base_ptr : Any
        Starting address of the access policy window. CUDA driver may align
        it.
    num_bytes : size_t
        Size in bytes of the window policy. CUDA driver may restrict the
        maximum size and alignment.
    hitRatio : float
        hitRatio specifies percentage of lines assigned hitProp, rest are
        assigned missProp.
    hitProp : CUaccessProperty
        CUaccessProperty set for hit.
    missProp : CUaccessProperty
        CUaccessProperty set for miss. Must be either NORMAL or STREAMING

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    pass

cdef class CUDA_KERNEL_NODE_PARAMS_v1(CUDA_KERNEL_NODE_PARAMS_st):
    """
    GPU kernel node parameters

    Attributes
    ----------
    func : CUfunction
        Kernel to launch
    gridDimX : unsigned int
        Width of grid in blocks
    gridDimY : unsigned int
        Height of grid in blocks
    gridDimZ : unsigned int
        Depth of grid in blocks
    blockDimX : unsigned int
        X dimension of each thread block
    blockDimY : unsigned int
        Y dimension of each thread block
    blockDimZ : unsigned int
        Z dimension of each thread block
    sharedMemBytes : unsigned int
        Dynamic shared-memory size per thread block in bytes
    kernelParams : Any
        Array of pointers to kernel parameters
    extra : Any
        Extra options

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    pass

cdef class CUDA_KERNEL_NODE_PARAMS(CUDA_KERNEL_NODE_PARAMS_st):
    """
    GPU kernel node parameters

    Attributes
    ----------
    func : CUfunction
        Kernel to launch
    gridDimX : unsigned int
        Width of grid in blocks
    gridDimY : unsigned int
        Height of grid in blocks
    gridDimZ : unsigned int
        Depth of grid in blocks
    blockDimX : unsigned int
        X dimension of each thread block
    blockDimY : unsigned int
        Y dimension of each thread block
    blockDimZ : unsigned int
        Z dimension of each thread block
    sharedMemBytes : unsigned int
        Dynamic shared-memory size per thread block in bytes
    kernelParams : Any
        Array of pointers to kernel parameters
    extra : Any
        Extra options

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    pass

cdef class CUDA_MEMSET_NODE_PARAMS_v1(CUDA_MEMSET_NODE_PARAMS_st):
    """
    Memset node parameters

    Attributes
    ----------
    dst : CUdeviceptr
        Destination device pointer
    pitch : size_t
        Pitch of destination device pointer. Unused if height is 1
    value : unsigned int
        Value to be set
    elementSize : unsigned int
        Size of each element in bytes. Must be 1, 2, or 4.
    width : size_t
        Width of the row in elements
    height : size_t
        Number of rows

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    pass

cdef class CUDA_MEMSET_NODE_PARAMS(CUDA_MEMSET_NODE_PARAMS_st):
    """
    Memset node parameters

    Attributes
    ----------
    dst : CUdeviceptr
        Destination device pointer
    pitch : size_t
        Pitch of destination device pointer. Unused if height is 1
    value : unsigned int
        Value to be set
    elementSize : unsigned int
        Size of each element in bytes. Must be 1, 2, or 4.
    width : size_t
        Width of the row in elements
    height : size_t
        Number of rows

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    pass

cdef class CUDA_HOST_NODE_PARAMS_v1(CUDA_HOST_NODE_PARAMS_st):
    """
    Host node parameters

    Attributes
    ----------
    fn : CUhostFn
        The function to call when the node executes
    userData : Any
        Argument to pass to the function

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    pass

cdef class CUDA_HOST_NODE_PARAMS(CUDA_HOST_NODE_PARAMS_st):
    """
    Host node parameters

    Attributes
    ----------
    fn : CUhostFn
        The function to call when the node executes
    userData : Any
        Argument to pass to the function

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    pass

cdef class CUkernelNodeAttrValue_v1(CUkernelNodeAttrValue_union):
    """
    Graph kernel node attributes union, used with
    ::cuKernelNodeSetAttribute/::cuKernelNodeGetAttribute

    Attributes
    ----------
    accessPolicyWindow : CUaccessPolicyWindow
        Attribute ::CUaccessPolicyWindow.
    cooperative : int
        Nonzero indicates a cooperative kernel (see
        cuLaunchCooperativeKernel).
    priority : int
        Execution priority of the kernel.

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    pass

cdef class CUkernelNodeAttrValue(CUkernelNodeAttrValue_union):
    """
    Graph kernel node attributes union, used with
    ::cuKernelNodeSetAttribute/::cuKernelNodeGetAttribute

    Attributes
    ----------
    accessPolicyWindow : CUaccessPolicyWindow
        Attribute ::CUaccessPolicyWindow.
    cooperative : int
        Nonzero indicates a cooperative kernel (see
        cuLaunchCooperativeKernel).
    priority : int
        Execution priority of the kernel.

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    pass

cdef class CUstreamAttrValue_v1(CUstreamAttrValue_union):
    """
    Stream attributes union, used with
    cuStreamSetAttribute/cuStreamGetAttribute

    Attributes
    ----------
    accessPolicyWindow : CUaccessPolicyWindow
        Attribute ::CUaccessPolicyWindow.
    syncPolicy : CUsynchronizationPolicy
        Value for CU_STREAM_ATTRIBUTE_SYNCHRONIZATION_POLICY.

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    pass

cdef class CUstreamAttrValue(CUstreamAttrValue_union):
    """
    Stream attributes union, used with
    cuStreamSetAttribute/cuStreamGetAttribute

    Attributes
    ----------
    accessPolicyWindow : CUaccessPolicyWindow
        Attribute ::CUaccessPolicyWindow.
    syncPolicy : CUsynchronizationPolicy
        Value for CU_STREAM_ATTRIBUTE_SYNCHRONIZATION_POLICY.

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    pass

cdef class CUexecAffinitySmCount_v1(CUexecAffinitySmCount_st):
    """
    Value for CU_EXEC_AFFINITY_TYPE_SM_COUNT

    Attributes
    ----------
    val : unsigned int
        The number of SMs the context is limited to use.

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    pass

cdef class CUexecAffinitySmCount(CUexecAffinitySmCount_st):
    """
    Value for CU_EXEC_AFFINITY_TYPE_SM_COUNT

    Attributes
    ----------
    val : unsigned int
        The number of SMs the context is limited to use.

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    pass

cdef class CUexecAffinityParam_v1(CUexecAffinityParam_st):
    """
    Execution Affinity Parameters

    Attributes
    ----------
    type : CUexecAffinityType

    param : _CUexecAffinityParam_v1_CUexecAffinityParam_v1_CUexecAffinityParam_st_param_u


    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    pass

cdef class CUexecAffinityParam(CUexecAffinityParam_st):
    """
    Execution Affinity Parameters

    Attributes
    ----------
    type : CUexecAffinityType

    param : _CUexecAffinityParam_v1_CUexecAffinityParam_v1_CUexecAffinityParam_st_param_u


    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    pass

cdef class CUDA_MEMCPY2D_v2(CUDA_MEMCPY2D_st):
    """
    2D memory copy parameters

    Attributes
    ----------
    srcXInBytes : size_t
        Source X in bytes
    srcY : size_t
        Source Y
    srcMemoryType : CUmemorytype
        Source memory type (host, device, array)
    srcHost : Any
        Source host pointer
    srcDevice : CUdeviceptr
        Source device pointer
    srcArray : CUarray
        Source array reference
    srcPitch : size_t
        Source pitch (ignored when src is array)
    dstXInBytes : size_t
        Destination X in bytes
    dstY : size_t
        Destination Y
    dstMemoryType : CUmemorytype
        Destination memory type (host, device, array)
    dstHost : Any
        Destination host pointer
    dstDevice : CUdeviceptr
        Destination device pointer
    dstArray : CUarray
        Destination array reference
    dstPitch : size_t
        Destination pitch (ignored when dst is array)
    WidthInBytes : size_t
        Width of 2D memory copy in bytes
    Height : size_t
        Height of 2D memory copy

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    pass

cdef class CUDA_MEMCPY2D(CUDA_MEMCPY2D_st):
    """
    2D memory copy parameters

    Attributes
    ----------
    srcXInBytes : size_t
        Source X in bytes
    srcY : size_t
        Source Y
    srcMemoryType : CUmemorytype
        Source memory type (host, device, array)
    srcHost : Any
        Source host pointer
    srcDevice : CUdeviceptr
        Source device pointer
    srcArray : CUarray
        Source array reference
    srcPitch : size_t
        Source pitch (ignored when src is array)
    dstXInBytes : size_t
        Destination X in bytes
    dstY : size_t
        Destination Y
    dstMemoryType : CUmemorytype
        Destination memory type (host, device, array)
    dstHost : Any
        Destination host pointer
    dstDevice : CUdeviceptr
        Destination device pointer
    dstArray : CUarray
        Destination array reference
    dstPitch : size_t
        Destination pitch (ignored when dst is array)
    WidthInBytes : size_t
        Width of 2D memory copy in bytes
    Height : size_t
        Height of 2D memory copy

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    pass

cdef class CUDA_MEMCPY3D_v2(CUDA_MEMCPY3D_st):
    """
    3D memory copy parameters

    Attributes
    ----------
    srcXInBytes : size_t
        Source X in bytes
    srcY : size_t
        Source Y
    srcZ : size_t
        Source Z
    srcLOD : size_t
        Source LOD
    srcMemoryType : CUmemorytype
        Source memory type (host, device, array)
    srcHost : Any
        Source host pointer
    srcDevice : CUdeviceptr
        Source device pointer
    srcArray : CUarray
        Source array reference
    reserved0 : Any
        Must be NULL
    srcPitch : size_t
        Source pitch (ignored when src is array)
    srcHeight : size_t
        Source height (ignored when src is array; may be 0 if Depth==1)
    dstXInBytes : size_t
        Destination X in bytes
    dstY : size_t
        Destination Y
    dstZ : size_t
        Destination Z
    dstLOD : size_t
        Destination LOD
    dstMemoryType : CUmemorytype
        Destination memory type (host, device, array)
    dstHost : Any
        Destination host pointer
    dstDevice : CUdeviceptr
        Destination device pointer
    dstArray : CUarray
        Destination array reference
    reserved1 : Any
        Must be NULL
    dstPitch : size_t
        Destination pitch (ignored when dst is array)
    dstHeight : size_t
        Destination height (ignored when dst is array; may be 0 if
        Depth==1)
    WidthInBytes : size_t
        Width of 3D memory copy in bytes
    Height : size_t
        Height of 3D memory copy
    Depth : size_t
        Depth of 3D memory copy

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    pass

cdef class CUDA_MEMCPY3D(CUDA_MEMCPY3D_st):
    """
    3D memory copy parameters

    Attributes
    ----------
    srcXInBytes : size_t
        Source X in bytes
    srcY : size_t
        Source Y
    srcZ : size_t
        Source Z
    srcLOD : size_t
        Source LOD
    srcMemoryType : CUmemorytype
        Source memory type (host, device, array)
    srcHost : Any
        Source host pointer
    srcDevice : CUdeviceptr
        Source device pointer
    srcArray : CUarray
        Source array reference
    reserved0 : Any
        Must be NULL
    srcPitch : size_t
        Source pitch (ignored when src is array)
    srcHeight : size_t
        Source height (ignored when src is array; may be 0 if Depth==1)
    dstXInBytes : size_t
        Destination X in bytes
    dstY : size_t
        Destination Y
    dstZ : size_t
        Destination Z
    dstLOD : size_t
        Destination LOD
    dstMemoryType : CUmemorytype
        Destination memory type (host, device, array)
    dstHost : Any
        Destination host pointer
    dstDevice : CUdeviceptr
        Destination device pointer
    dstArray : CUarray
        Destination array reference
    reserved1 : Any
        Must be NULL
    dstPitch : size_t
        Destination pitch (ignored when dst is array)
    dstHeight : size_t
        Destination height (ignored when dst is array; may be 0 if
        Depth==1)
    WidthInBytes : size_t
        Width of 3D memory copy in bytes
    Height : size_t
        Height of 3D memory copy
    Depth : size_t
        Depth of 3D memory copy

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    pass

cdef class CUDA_MEMCPY3D_PEER_v1(CUDA_MEMCPY3D_PEER_st):
    """
    3D memory cross-context copy parameters

    Attributes
    ----------
    srcXInBytes : size_t
        Source X in bytes
    srcY : size_t
        Source Y
    srcZ : size_t
        Source Z
    srcLOD : size_t
        Source LOD
    srcMemoryType : CUmemorytype
        Source memory type (host, device, array)
    srcHost : Any
        Source host pointer
    srcDevice : CUdeviceptr
        Source device pointer
    srcArray : CUarray
        Source array reference
    srcContext : CUcontext
        Source context (ignored with srcMemoryType is CU_MEMORYTYPE_ARRAY)
    srcPitch : size_t
        Source pitch (ignored when src is array)
    srcHeight : size_t
        Source height (ignored when src is array; may be 0 if Depth==1)
    dstXInBytes : size_t
        Destination X in bytes
    dstY : size_t
        Destination Y
    dstZ : size_t
        Destination Z
    dstLOD : size_t
        Destination LOD
    dstMemoryType : CUmemorytype
        Destination memory type (host, device, array)
    dstHost : Any
        Destination host pointer
    dstDevice : CUdeviceptr
        Destination device pointer
    dstArray : CUarray
        Destination array reference
    dstContext : CUcontext
        Destination context (ignored with dstMemoryType is
        CU_MEMORYTYPE_ARRAY)
    dstPitch : size_t
        Destination pitch (ignored when dst is array)
    dstHeight : size_t
        Destination height (ignored when dst is array; may be 0 if
        Depth==1)
    WidthInBytes : size_t
        Width of 3D memory copy in bytes
    Height : size_t
        Height of 3D memory copy
    Depth : size_t
        Depth of 3D memory copy

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    pass

cdef class CUDA_MEMCPY3D_PEER(CUDA_MEMCPY3D_PEER_st):
    """
    3D memory cross-context copy parameters

    Attributes
    ----------
    srcXInBytes : size_t
        Source X in bytes
    srcY : size_t
        Source Y
    srcZ : size_t
        Source Z
    srcLOD : size_t
        Source LOD
    srcMemoryType : CUmemorytype
        Source memory type (host, device, array)
    srcHost : Any
        Source host pointer
    srcDevice : CUdeviceptr
        Source device pointer
    srcArray : CUarray
        Source array reference
    srcContext : CUcontext
        Source context (ignored with srcMemoryType is CU_MEMORYTYPE_ARRAY)
    srcPitch : size_t
        Source pitch (ignored when src is array)
    srcHeight : size_t
        Source height (ignored when src is array; may be 0 if Depth==1)
    dstXInBytes : size_t
        Destination X in bytes
    dstY : size_t
        Destination Y
    dstZ : size_t
        Destination Z
    dstLOD : size_t
        Destination LOD
    dstMemoryType : CUmemorytype
        Destination memory type (host, device, array)
    dstHost : Any
        Destination host pointer
    dstDevice : CUdeviceptr
        Destination device pointer
    dstArray : CUarray
        Destination array reference
    dstContext : CUcontext
        Destination context (ignored with dstMemoryType is
        CU_MEMORYTYPE_ARRAY)
    dstPitch : size_t
        Destination pitch (ignored when dst is array)
    dstHeight : size_t
        Destination height (ignored when dst is array; may be 0 if
        Depth==1)
    WidthInBytes : size_t
        Width of 3D memory copy in bytes
    Height : size_t
        Height of 3D memory copy
    Depth : size_t
        Depth of 3D memory copy

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    pass

cdef class CUDA_ARRAY_DESCRIPTOR_v2(CUDA_ARRAY_DESCRIPTOR_st):
    """
    Array descriptor

    Attributes
    ----------
    Width : size_t
        Width of array
    Height : size_t
        Height of array
    Format : CUarray_format
        Array format
    NumChannels : unsigned int
        Channels per array element

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    pass

cdef class CUDA_ARRAY_DESCRIPTOR(CUDA_ARRAY_DESCRIPTOR_st):
    """
    Array descriptor

    Attributes
    ----------
    Width : size_t
        Width of array
    Height : size_t
        Height of array
    Format : CUarray_format
        Array format
    NumChannels : unsigned int
        Channels per array element

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    pass

cdef class CUDA_ARRAY3D_DESCRIPTOR_v2(CUDA_ARRAY3D_DESCRIPTOR_st):
    """
    3D array descriptor

    Attributes
    ----------
    Width : size_t
        Width of 3D array
    Height : size_t
        Height of 3D array
    Depth : size_t
        Depth of 3D array
    Format : CUarray_format
        Array format
    NumChannels : unsigned int
        Channels per array element
    Flags : unsigned int
        Flags

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    pass

cdef class CUDA_ARRAY3D_DESCRIPTOR(CUDA_ARRAY3D_DESCRIPTOR_st):
    """
    3D array descriptor

    Attributes
    ----------
    Width : size_t
        Width of 3D array
    Height : size_t
        Height of 3D array
    Depth : size_t
        Depth of 3D array
    Format : CUarray_format
        Array format
    NumChannels : unsigned int
        Channels per array element
    Flags : unsigned int
        Flags

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    pass

cdef class CUDA_ARRAY_SPARSE_PROPERTIES_v1(CUDA_ARRAY_SPARSE_PROPERTIES_st):
    """
    CUDA array sparse properties

    Attributes
    ----------
    tileExtent : _CUDA_ARRAY_SPARSE_PROPERTIES_v1_CUDA_ARRAY_SPARSE_PROPERTIES_v1_CUDA_ARRAY_SPARSE_PROPERTIES_st_tileExtent_s

    miptailFirstLevel : unsigned int
        First mip level at which the mip tail begins.
    miptailSize : unsigned long long
        Total size of the mip tail.
    flags : unsigned int
        Flags will either be zero or
        CU_ARRAY_SPARSE_PROPERTIES_SINGLE_MIPTAIL
    reserved : List[unsigned int]


    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    pass

cdef class CUDA_ARRAY_SPARSE_PROPERTIES(CUDA_ARRAY_SPARSE_PROPERTIES_st):
    """
    CUDA array sparse properties

    Attributes
    ----------
    tileExtent : _CUDA_ARRAY_SPARSE_PROPERTIES_v1_CUDA_ARRAY_SPARSE_PROPERTIES_v1_CUDA_ARRAY_SPARSE_PROPERTIES_st_tileExtent_s

    miptailFirstLevel : unsigned int
        First mip level at which the mip tail begins.
    miptailSize : unsigned long long
        Total size of the mip tail.
    flags : unsigned int
        Flags will either be zero or
        CU_ARRAY_SPARSE_PROPERTIES_SINGLE_MIPTAIL
    reserved : List[unsigned int]


    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    pass

cdef class CUDA_ARRAY_MEMORY_REQUIREMENTS_v1(CUDA_ARRAY_MEMORY_REQUIREMENTS_st):
    """
    CUDA array memory requirements

    Attributes
    ----------
    size : size_t
        Total required memory size
    alignment : size_t
        alignment requirement
    reserved : List[unsigned int]


    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    pass

cdef class CUDA_ARRAY_MEMORY_REQUIREMENTS(CUDA_ARRAY_MEMORY_REQUIREMENTS_st):
    """
    CUDA array memory requirements

    Attributes
    ----------
    size : size_t
        Total required memory size
    alignment : size_t
        alignment requirement
    reserved : List[unsigned int]


    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    pass

cdef class CUDA_RESOURCE_DESC_v1(CUDA_RESOURCE_DESC_st):
    """
    CUDA Resource descriptor

    Attributes
    ----------
    resType : CUresourcetype
        Resource type
    res : _CUDA_RESOURCE_DESC_v1_CUDA_RESOURCE_DESC_v1_CUDA_RESOURCE_DESC_st_res_u

    flags : unsigned int
        Flags (must be zero)

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    pass

cdef class CUDA_RESOURCE_DESC(CUDA_RESOURCE_DESC_st):
    """
    CUDA Resource descriptor

    Attributes
    ----------
    resType : CUresourcetype
        Resource type
    res : _CUDA_RESOURCE_DESC_v1_CUDA_RESOURCE_DESC_v1_CUDA_RESOURCE_DESC_st_res_u

    flags : unsigned int
        Flags (must be zero)

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    pass

cdef class CUDA_TEXTURE_DESC_v1(CUDA_TEXTURE_DESC_st):
    """
    Texture descriptor

    Attributes
    ----------
    addressMode : List[CUaddress_mode]
        Address modes
    filterMode : CUfilter_mode
        Filter mode
    flags : unsigned int
        Flags
    maxAnisotropy : unsigned int
        Maximum anisotropy ratio
    mipmapFilterMode : CUfilter_mode
        Mipmap filter mode
    mipmapLevelBias : float
        Mipmap level bias
    minMipmapLevelClamp : float
        Mipmap minimum level clamp
    maxMipmapLevelClamp : float
        Mipmap maximum level clamp
    borderColor : List[float]
        Border Color
    reserved : List[int]


    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    pass

cdef class CUDA_TEXTURE_DESC(CUDA_TEXTURE_DESC_st):
    """
    Texture descriptor

    Attributes
    ----------
    addressMode : List[CUaddress_mode]
        Address modes
    filterMode : CUfilter_mode
        Filter mode
    flags : unsigned int
        Flags
    maxAnisotropy : unsigned int
        Maximum anisotropy ratio
    mipmapFilterMode : CUfilter_mode
        Mipmap filter mode
    mipmapLevelBias : float
        Mipmap level bias
    minMipmapLevelClamp : float
        Mipmap minimum level clamp
    maxMipmapLevelClamp : float
        Mipmap maximum level clamp
    borderColor : List[float]
        Border Color
    reserved : List[int]


    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    pass

cdef class CUDA_RESOURCE_VIEW_DESC_v1(CUDA_RESOURCE_VIEW_DESC_st):
    """
    Resource view descriptor

    Attributes
    ----------
    format : CUresourceViewFormat
        Resource view format
    width : size_t
        Width of the resource view
    height : size_t
        Height of the resource view
    depth : size_t
        Depth of the resource view
    firstMipmapLevel : unsigned int
        First defined mipmap level
    lastMipmapLevel : unsigned int
        Last defined mipmap level
    firstLayer : unsigned int
        First layer index
    lastLayer : unsigned int
        Last layer index
    reserved : List[unsigned int]


    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    pass

cdef class CUDA_RESOURCE_VIEW_DESC(CUDA_RESOURCE_VIEW_DESC_st):
    """
    Resource view descriptor

    Attributes
    ----------
    format : CUresourceViewFormat
        Resource view format
    width : size_t
        Width of the resource view
    height : size_t
        Height of the resource view
    depth : size_t
        Depth of the resource view
    firstMipmapLevel : unsigned int
        First defined mipmap level
    lastMipmapLevel : unsigned int
        Last defined mipmap level
    firstLayer : unsigned int
        First layer index
    lastLayer : unsigned int
        Last layer index
    reserved : List[unsigned int]


    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    pass

cdef class CUDA_POINTER_ATTRIBUTE_P2P_TOKENS_v1(CUDA_POINTER_ATTRIBUTE_P2P_TOKENS_st):
    """
    GPU Direct v3 tokens

    Attributes
    ----------
    p2pToken : unsigned long long

    vaSpaceToken : unsigned int


    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    pass

cdef class CUDA_POINTER_ATTRIBUTE_P2P_TOKENS(CUDA_POINTER_ATTRIBUTE_P2P_TOKENS_st):
    """
    GPU Direct v3 tokens

    Attributes
    ----------
    p2pToken : unsigned long long

    vaSpaceToken : unsigned int


    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    pass

cdef class CUDA_LAUNCH_PARAMS_v1(CUDA_LAUNCH_PARAMS_st):
    """
    Kernel launch parameters

    Attributes
    ----------
    function : CUfunction
        Kernel to launch
    gridDimX : unsigned int
        Width of grid in blocks
    gridDimY : unsigned int
        Height of grid in blocks
    gridDimZ : unsigned int
        Depth of grid in blocks
    blockDimX : unsigned int
        X dimension of each thread block
    blockDimY : unsigned int
        Y dimension of each thread block
    blockDimZ : unsigned int
        Z dimension of each thread block
    sharedMemBytes : unsigned int
        Dynamic shared-memory size per thread block in bytes
    hStream : CUstream
        Stream identifier
    kernelParams : Any
        Array of pointers to kernel parameters

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    pass

cdef class CUDA_LAUNCH_PARAMS(CUDA_LAUNCH_PARAMS_st):
    """
    Kernel launch parameters

    Attributes
    ----------
    function : CUfunction
        Kernel to launch
    gridDimX : unsigned int
        Width of grid in blocks
    gridDimY : unsigned int
        Height of grid in blocks
    gridDimZ : unsigned int
        Depth of grid in blocks
    blockDimX : unsigned int
        X dimension of each thread block
    blockDimY : unsigned int
        Y dimension of each thread block
    blockDimZ : unsigned int
        Z dimension of each thread block
    sharedMemBytes : unsigned int
        Dynamic shared-memory size per thread block in bytes
    hStream : CUstream
        Stream identifier
    kernelParams : Any
        Array of pointers to kernel parameters

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    pass

cdef class CUDA_EXTERNAL_MEMORY_HANDLE_DESC_v1(CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st):
    """
    External memory handle descriptor

    Attributes
    ----------
    type : CUexternalMemoryHandleType
        Type of the handle
    handle : _CUDA_EXTERNAL_MEMORY_HANDLE_DESC_v1_CUDA_EXTERNAL_MEMORY_HANDLE_DESC_v1_CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st_handle_u

    size : unsigned long long
        Size of the memory allocation
    flags : unsigned int
        Flags must either be zero or CUDA_EXTERNAL_MEMORY_DEDICATED
    reserved : List[unsigned int]


    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    pass

cdef class CUDA_EXTERNAL_MEMORY_HANDLE_DESC(CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st):
    """
    External memory handle descriptor

    Attributes
    ----------
    type : CUexternalMemoryHandleType
        Type of the handle
    handle : _CUDA_EXTERNAL_MEMORY_HANDLE_DESC_v1_CUDA_EXTERNAL_MEMORY_HANDLE_DESC_v1_CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st_handle_u

    size : unsigned long long
        Size of the memory allocation
    flags : unsigned int
        Flags must either be zero or CUDA_EXTERNAL_MEMORY_DEDICATED
    reserved : List[unsigned int]


    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    pass

cdef class CUDA_EXTERNAL_MEMORY_BUFFER_DESC_v1(CUDA_EXTERNAL_MEMORY_BUFFER_DESC_st):
    """
    External memory buffer descriptor

    Attributes
    ----------
    offset : unsigned long long
        Offset into the memory object where the buffer's base is
    size : unsigned long long
        Size of the buffer
    flags : unsigned int
        Flags reserved for future use. Must be zero.
    reserved : List[unsigned int]


    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    pass

cdef class CUDA_EXTERNAL_MEMORY_BUFFER_DESC(CUDA_EXTERNAL_MEMORY_BUFFER_DESC_st):
    """
    External memory buffer descriptor

    Attributes
    ----------
    offset : unsigned long long
        Offset into the memory object where the buffer's base is
    size : unsigned long long
        Size of the buffer
    flags : unsigned int
        Flags reserved for future use. Must be zero.
    reserved : List[unsigned int]


    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    pass

cdef class CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC_v1(CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC_st):
    """
    External memory mipmap descriptor

    Attributes
    ----------
    offset : unsigned long long
        Offset into the memory object where the base level of the mipmap
        chain is.
    arrayDesc : CUDA_ARRAY3D_DESCRIPTOR
        Format, dimension and type of base level of the mipmap chain
    numLevels : unsigned int
        Total number of levels in the mipmap chain
    reserved : List[unsigned int]


    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    pass

cdef class CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC(CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC_st):
    """
    External memory mipmap descriptor

    Attributes
    ----------
    offset : unsigned long long
        Offset into the memory object where the base level of the mipmap
        chain is.
    arrayDesc : CUDA_ARRAY3D_DESCRIPTOR
        Format, dimension and type of base level of the mipmap chain
    numLevels : unsigned int
        Total number of levels in the mipmap chain
    reserved : List[unsigned int]


    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    pass

cdef class CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_v1(CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_st):
    """
    External semaphore handle descriptor

    Attributes
    ----------
    type : CUexternalSemaphoreHandleType
        Type of the handle
    handle : _CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_v1_CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_v1_CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_st_handle_u

    flags : unsigned int
        Flags reserved for the future. Must be zero.
    reserved : List[unsigned int]


    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    pass

cdef class CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC(CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_st):
    """
    External semaphore handle descriptor

    Attributes
    ----------
    type : CUexternalSemaphoreHandleType
        Type of the handle
    handle : _CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_v1_CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_v1_CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_st_handle_u

    flags : unsigned int
        Flags reserved for the future. Must be zero.
    reserved : List[unsigned int]


    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    pass

cdef class CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_v1(CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st):
    """
    External semaphore signal parameters

    Attributes
    ----------
    params : _CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_v1_CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_v1_CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st_params_s

    flags : unsigned int
        Only when ::CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS is used to signal
        a CUexternalSemaphore of type
        CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_NVSCISYNC, the valid flag is
        CUDA_EXTERNAL_SEMAPHORE_SIGNAL_SKIP_NVSCIBUF_MEMSYNC which
        indicates that while signaling the CUexternalSemaphore, no memory
        synchronization operations should be performed for any external
        memory object imported as CU_EXTERNAL_MEMORY_HANDLE_TYPE_NVSCIBUF.
        For all other types of CUexternalSemaphore, flags must be zero.
    reserved : List[unsigned int]


    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    pass

cdef class CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS(CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st):
    """
    External semaphore signal parameters

    Attributes
    ----------
    params : _CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_v1_CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_v1_CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st_params_s

    flags : unsigned int
        Only when ::CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS is used to signal
        a CUexternalSemaphore of type
        CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_NVSCISYNC, the valid flag is
        CUDA_EXTERNAL_SEMAPHORE_SIGNAL_SKIP_NVSCIBUF_MEMSYNC which
        indicates that while signaling the CUexternalSemaphore, no memory
        synchronization operations should be performed for any external
        memory object imported as CU_EXTERNAL_MEMORY_HANDLE_TYPE_NVSCIBUF.
        For all other types of CUexternalSemaphore, flags must be zero.
    reserved : List[unsigned int]


    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    pass

cdef class CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_v1(CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st):
    """
    External semaphore wait parameters

    Attributes
    ----------
    params : _CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_v1_CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_v1_CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st_params_s

    flags : unsigned int
        Only when ::CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS is used to wait on
        a CUexternalSemaphore of type
        CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_NVSCISYNC, the valid flag is
        CUDA_EXTERNAL_SEMAPHORE_WAIT_SKIP_NVSCIBUF_MEMSYNC which indicates
        that while waiting for the CUexternalSemaphore, no memory
        synchronization operations should be performed for any external
        memory object imported as CU_EXTERNAL_MEMORY_HANDLE_TYPE_NVSCIBUF.
        For all other types of CUexternalSemaphore, flags must be zero.
    reserved : List[unsigned int]


    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    pass

cdef class CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS(CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st):
    """
    External semaphore wait parameters

    Attributes
    ----------
    params : _CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_v1_CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_v1_CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st_params_s

    flags : unsigned int
        Only when ::CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS is used to wait on
        a CUexternalSemaphore of type
        CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_NVSCISYNC, the valid flag is
        CUDA_EXTERNAL_SEMAPHORE_WAIT_SKIP_NVSCIBUF_MEMSYNC which indicates
        that while waiting for the CUexternalSemaphore, no memory
        synchronization operations should be performed for any external
        memory object imported as CU_EXTERNAL_MEMORY_HANDLE_TYPE_NVSCIBUF.
        For all other types of CUexternalSemaphore, flags must be zero.
    reserved : List[unsigned int]


    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    pass

cdef class CUDA_EXT_SEM_SIGNAL_NODE_PARAMS_v1(CUDA_EXT_SEM_SIGNAL_NODE_PARAMS_st):
    """
    Semaphore signal node parameters

    Attributes
    ----------
    extSemArray : CUexternalSemaphore
        Array of external semaphore handles.
    paramsArray : CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS
        Array of external semaphore signal parameters.
    numExtSems : unsigned int
        Number of handles and parameters supplied in extSemArray and
        paramsArray.

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    pass

cdef class CUDA_EXT_SEM_SIGNAL_NODE_PARAMS(CUDA_EXT_SEM_SIGNAL_NODE_PARAMS_st):
    """
    Semaphore signal node parameters

    Attributes
    ----------
    extSemArray : CUexternalSemaphore
        Array of external semaphore handles.
    paramsArray : CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS
        Array of external semaphore signal parameters.
    numExtSems : unsigned int
        Number of handles and parameters supplied in extSemArray and
        paramsArray.

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    pass

cdef class CUDA_EXT_SEM_WAIT_NODE_PARAMS_v1(CUDA_EXT_SEM_WAIT_NODE_PARAMS_st):
    """
    Semaphore wait node parameters

    Attributes
    ----------
    extSemArray : CUexternalSemaphore
        Array of external semaphore handles.
    paramsArray : CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS
        Array of external semaphore wait parameters.
    numExtSems : unsigned int
        Number of handles and parameters supplied in extSemArray and
        paramsArray.

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    pass

cdef class CUDA_EXT_SEM_WAIT_NODE_PARAMS(CUDA_EXT_SEM_WAIT_NODE_PARAMS_st):
    """
    Semaphore wait node parameters

    Attributes
    ----------
    extSemArray : CUexternalSemaphore
        Array of external semaphore handles.
    paramsArray : CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS
        Array of external semaphore wait parameters.
    numExtSems : unsigned int
        Number of handles and parameters supplied in extSemArray and
        paramsArray.

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    pass

cdef class CUarrayMapInfo_v1(CUarrayMapInfo_st):
    """
    Specifies the CUDA array or CUDA mipmapped array memory mapping
    information

    Attributes
    ----------
    resourceType : CUresourcetype
        Resource type
    resource : _CUarrayMapInfo_v1_CUarrayMapInfo_v1_CUarrayMapInfo_st_resource_u

    subresourceType : CUarraySparseSubresourceType
        Sparse subresource type
    subresource : _CUarrayMapInfo_v1_CUarrayMapInfo_v1_CUarrayMapInfo_st_subresource_u

    memOperationType : CUmemOperationType
        Memory operation type
    memHandleType : CUmemHandleType
        Memory handle type
    memHandle : _CUarrayMapInfo_v1_CUarrayMapInfo_v1_CUarrayMapInfo_st_memHandle_u

    offset : unsigned long long
        Offset within mip tail  Offset within the memory
    deviceBitMask : unsigned int
        Device ordinal bit mask
    flags : unsigned int
        flags for future use, must be zero now.
    reserved : List[unsigned int]
        Reserved for future use, must be zero now.

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    pass

cdef class CUarrayMapInfo(CUarrayMapInfo_st):
    """
    Specifies the CUDA array or CUDA mipmapped array memory mapping
    information

    Attributes
    ----------
    resourceType : CUresourcetype
        Resource type
    resource : _CUarrayMapInfo_v1_CUarrayMapInfo_v1_CUarrayMapInfo_st_resource_u

    subresourceType : CUarraySparseSubresourceType
        Sparse subresource type
    subresource : _CUarrayMapInfo_v1_CUarrayMapInfo_v1_CUarrayMapInfo_st_subresource_u

    memOperationType : CUmemOperationType
        Memory operation type
    memHandleType : CUmemHandleType
        Memory handle type
    memHandle : _CUarrayMapInfo_v1_CUarrayMapInfo_v1_CUarrayMapInfo_st_memHandle_u

    offset : unsigned long long
        Offset within mip tail  Offset within the memory
    deviceBitMask : unsigned int
        Device ordinal bit mask
    flags : unsigned int
        flags for future use, must be zero now.
    reserved : List[unsigned int]
        Reserved for future use, must be zero now.

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    pass

cdef class CUmemLocation_v1(CUmemLocation_st):
    """
    Specifies a memory location.

    Attributes
    ----------
    type : CUmemLocationType
        Specifies the location type, which modifies the meaning of id.
    id : int
        identifier for a given this location's CUmemLocationType.

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    pass

cdef class CUmemLocation(CUmemLocation_st):
    """
    Specifies a memory location.

    Attributes
    ----------
    type : CUmemLocationType
        Specifies the location type, which modifies the meaning of id.
    id : int
        identifier for a given this location's CUmemLocationType.

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    pass

cdef class CUmemAllocationProp_v1(CUmemAllocationProp_st):
    """
    Specifies the allocation properties for a allocation.

    Attributes
    ----------
    type : CUmemAllocationType
        Allocation type
    requestedHandleTypes : CUmemAllocationHandleType
        requested CUmemAllocationHandleType
    location : CUmemLocation
        Location of allocation
    win32HandleMetaData : Any
        Windows-specific POBJECT_ATTRIBUTES required when
        CU_MEM_HANDLE_TYPE_WIN32 is specified. This object atributes
        structure includes security attributes that define the scope of
        which exported allocations may be tranferred to other processes. In
        all other cases, this field is required to be zero.
    allocFlags : _CUmemAllocationProp_v1_CUmemAllocationProp_v1_CUmemAllocationProp_st_allocFlags_s


    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    pass

cdef class CUmemAllocationProp(CUmemAllocationProp_st):
    """
    Specifies the allocation properties for a allocation.

    Attributes
    ----------
    type : CUmemAllocationType
        Allocation type
    requestedHandleTypes : CUmemAllocationHandleType
        requested CUmemAllocationHandleType
    location : CUmemLocation
        Location of allocation
    win32HandleMetaData : Any
        Windows-specific POBJECT_ATTRIBUTES required when
        CU_MEM_HANDLE_TYPE_WIN32 is specified. This object atributes
        structure includes security attributes that define the scope of
        which exported allocations may be tranferred to other processes. In
        all other cases, this field is required to be zero.
    allocFlags : _CUmemAllocationProp_v1_CUmemAllocationProp_v1_CUmemAllocationProp_st_allocFlags_s


    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    pass

cdef class CUmemAccessDesc_v1(CUmemAccessDesc_st):
    """
    Memory access descriptor

    Attributes
    ----------
    location : CUmemLocation
        Location on which the request is to change it's accessibility
    flags : CUmemAccess_flags
        ::CUmemProt accessibility flags to set on the request

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    pass

cdef class CUmemAccessDesc(CUmemAccessDesc_st):
    """
    Memory access descriptor

    Attributes
    ----------
    location : CUmemLocation
        Location on which the request is to change it's accessibility
    flags : CUmemAccess_flags
        ::CUmemProt accessibility flags to set on the request

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    pass

cdef class CUmemPoolProps_v1(CUmemPoolProps_st):
    """
    Specifies the properties of allocations made from the pool.

    Attributes
    ----------
    allocType : CUmemAllocationType
        Allocation type. Currently must be specified as
        CU_MEM_ALLOCATION_TYPE_PINNED
    handleTypes : CUmemAllocationHandleType
        Handle types that will be supported by allocations from the pool.
    location : CUmemLocation
        Location where allocations should reside.
    win32SecurityAttributes : Any
        Windows-specific LPSECURITYATTRIBUTES required when
        CU_MEM_HANDLE_TYPE_WIN32 is specified. This security attribute
        defines the scope of which exported allocations may be tranferred
        to other processes. In all other cases, this field is required to
        be zero.
    reserved : bytes
        reserved for future use, must be 0

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    pass

cdef class CUmemPoolProps(CUmemPoolProps_st):
    """
    Specifies the properties of allocations made from the pool.

    Attributes
    ----------
    allocType : CUmemAllocationType
        Allocation type. Currently must be specified as
        CU_MEM_ALLOCATION_TYPE_PINNED
    handleTypes : CUmemAllocationHandleType
        Handle types that will be supported by allocations from the pool.
    location : CUmemLocation
        Location where allocations should reside.
    win32SecurityAttributes : Any
        Windows-specific LPSECURITYATTRIBUTES required when
        CU_MEM_HANDLE_TYPE_WIN32 is specified. This security attribute
        defines the scope of which exported allocations may be tranferred
        to other processes. In all other cases, this field is required to
        be zero.
    reserved : bytes
        reserved for future use, must be 0

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    pass

cdef class CUmemPoolPtrExportData_v1(CUmemPoolPtrExportData_st):
    """
    Opaque data for exporting a pool allocation

    Attributes
    ----------
    reserved : bytes


    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    pass

cdef class CUmemPoolPtrExportData(CUmemPoolPtrExportData_st):
    """
    Opaque data for exporting a pool allocation

    Attributes
    ----------
    reserved : bytes


    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    pass

cdef class CUDA_MEM_ALLOC_NODE_PARAMS(CUDA_MEM_ALLOC_NODE_PARAMS_st):
    """
    Memory allocation node parameters

    Attributes
    ----------
    poolProps : CUmemPoolProps
        in: location where the allocation should reside (specified in
        ::location). ::handleTypes must be CU_MEM_HANDLE_TYPE_NONE. IPC is
        not supported.
    accessDescs : CUmemAccessDesc
        in: array of memory access descriptors. Used to describe peer GPU
        access
    accessDescCount : size_t
        in: number of memory access descriptors. Must not exceed the number
        of GPUs.
    bytesize : size_t
        in: size in bytes of the requested allocation
    dptr : CUdeviceptr
        out: address of the allocation returned by CUDA

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    pass

cdef class CUeglFrame_v1(CUeglFrame_st):
    """
    CUDA EGLFrame structure Descriptor - structure defining one frame
    of EGL.  Each frame may contain one or more planes depending on
    whether the surface * is Multiplanar or not.

    Attributes
    ----------
    frame : _CUeglFrame_v1_CUeglFrame_v1_CUeglFrame_st_frame_u

    width : unsigned int
        Width of first plane
    height : unsigned int
        Height of first plane
    depth : unsigned int
        Depth of first plane
    pitch : unsigned int
        Pitch of first plane
    planeCount : unsigned int
        Number of planes
    numChannels : unsigned int
        Number of channels for the plane
    frameType : CUeglFrameType
        Array or Pitch
    eglColorFormat : CUeglColorFormat
        CUDA EGL Color Format
    cuFormat : CUarray_format
        CUDA Array Format

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    pass

cdef class CUeglFrame(CUeglFrame_st):
    """
    CUDA EGLFrame structure Descriptor - structure defining one frame
    of EGL.  Each frame may contain one or more planes depending on
    whether the surface * is Multiplanar or not.

    Attributes
    ----------
    frame : _CUeglFrame_v1_CUeglFrame_v1_CUeglFrame_st_frame_u

    width : unsigned int
        Width of first plane
    height : unsigned int
        Height of first plane
    depth : unsigned int
        Depth of first plane
    pitch : unsigned int
        Pitch of first plane
    planeCount : unsigned int
        Number of planes
    numChannels : unsigned int
        Number of channels for the plane
    frameType : CUeglFrameType
        Array or Pitch
    eglColorFormat : CUeglColorFormat
        CUDA EGL Color Format
    cuFormat : CUarray_format
        CUDA Array Format

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    pass

cdef class cuuint32_t:
    """

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef ccuda.cuuint32_t  _val
    cdef ccuda.cuuint32_t* _ptr

cdef class cuuint64_t:
    """

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef ccuda.cuuint64_t  _val
    cdef ccuda.cuuint64_t* _ptr

cdef class CUdeviceptr_v2:
    """

    CUDA device pointer CUdeviceptr is defined as an unsigned integer type whose size matches the size of a pointer on the target platform.

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef ccuda.CUdeviceptr_v2  _val
    cdef ccuda.CUdeviceptr_v2* _ptr

cdef class CUdeviceptr:
    """

    CUDA device pointer

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef ccuda.CUdeviceptr  _val
    cdef ccuda.CUdeviceptr* _ptr

cdef class CUdevice_v1:
    """

    CUDA device

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef ccuda.CUdevice_v1  _val
    cdef ccuda.CUdevice_v1* _ptr

cdef class CUdevice:
    """

    CUDA device

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef ccuda.CUdevice  _val
    cdef ccuda.CUdevice* _ptr

cdef class CUtexObject_v1:
    """

    An opaque value that represents a CUDA texture object

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef ccuda.CUtexObject_v1  _val
    cdef ccuda.CUtexObject_v1* _ptr

cdef class CUtexObject:
    """

    An opaque value that represents a CUDA texture object

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef ccuda.CUtexObject  _val
    cdef ccuda.CUtexObject* _ptr

cdef class CUsurfObject_v1:
    """

    An opaque value that represents a CUDA surface object

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef ccuda.CUsurfObject_v1  _val
    cdef ccuda.CUsurfObject_v1* _ptr

cdef class CUsurfObject:
    """

    An opaque value that represents a CUDA surface object

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef ccuda.CUsurfObject  _val
    cdef ccuda.CUsurfObject* _ptr

cdef class CUmemGenericAllocationHandle_v1:
    """

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef ccuda.CUmemGenericAllocationHandle_v1  _val
    cdef ccuda.CUmemGenericAllocationHandle_v1* _ptr

cdef class CUmemGenericAllocationHandle:
    """

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef ccuda.CUmemGenericAllocationHandle  _val
    cdef ccuda.CUmemGenericAllocationHandle* _ptr

cdef class GLenum:
    """

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef ccuda.GLenum  _val
    cdef ccuda.GLenum* _ptr

cdef class GLuint:
    """

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef ccuda.GLuint  _val
    cdef ccuda.GLuint* _ptr

cdef class EGLint:
    """

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef ccuda.EGLint  _val
    cdef ccuda.EGLint* _ptr

cdef class VdpDevice:
    """

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef ccuda.VdpDevice  _val
    cdef ccuda.VdpDevice* _ptr

cdef class VdpGetProcAddress:
    """

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef ccuda.VdpGetProcAddress  _val
    cdef ccuda.VdpGetProcAddress* _ptr

cdef class VdpVideoSurface:
    """

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef ccuda.VdpVideoSurface  _val
    cdef ccuda.VdpVideoSurface* _ptr

cdef class VdpOutputSurface:
    """

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef ccuda.VdpOutputSurface  _val
    cdef ccuda.VdpOutputSurface* _ptr
