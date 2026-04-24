# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

# This code was automatically generated with version 13.2.0, generator version 0.3.1.dev1630+gadce055ea.d20260422. Do not modify it directly.
cimport cuda.bindings.cydriver as cydriver

include "_lib/utils.pxd"

cdef class CUcontext:
    """

    A regular context handle

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef cydriver.CUcontext  _pvt_val
    cdef cydriver.CUcontext* _pvt_ptr

cdef class CUmodule:
    """

    CUDA module

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef cydriver.CUmodule  _pvt_val
    cdef cydriver.CUmodule* _pvt_ptr

cdef class CUfunction:
    """

    CUDA function

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef cydriver.CUfunction  _pvt_val
    cdef cydriver.CUfunction* _pvt_ptr

cdef class CUlibrary:
    """

    CUDA library

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef cydriver.CUlibrary  _pvt_val
    cdef cydriver.CUlibrary* _pvt_ptr

cdef class CUkernel:
    """

    CUDA kernel

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef cydriver.CUkernel  _pvt_val
    cdef cydriver.CUkernel* _pvt_ptr

cdef class CUarray:
    """

    CUDA array

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef cydriver.CUarray  _pvt_val
    cdef cydriver.CUarray* _pvt_ptr

cdef class CUmipmappedArray:
    """

    CUDA mipmapped array

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef cydriver.CUmipmappedArray  _pvt_val
    cdef cydriver.CUmipmappedArray* _pvt_ptr

cdef class CUtexref:
    """

    CUDA texture reference

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef cydriver.CUtexref  _pvt_val
    cdef cydriver.CUtexref* _pvt_ptr

cdef class CUsurfref:
    """

    CUDA surface reference

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef cydriver.CUsurfref  _pvt_val
    cdef cydriver.CUsurfref* _pvt_ptr

cdef class CUevent:
    """

    CUDA event

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef cydriver.CUevent  _pvt_val
    cdef cydriver.CUevent* _pvt_ptr

cdef class CUstream:
    """

    CUDA stream

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef cydriver.CUstream  _pvt_val
    cdef cydriver.CUstream* _pvt_ptr

cdef class CUgraphicsResource:
    """

    CUDA graphics interop resource

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef cydriver.CUgraphicsResource  _pvt_val
    cdef cydriver.CUgraphicsResource* _pvt_ptr

cdef class CUexternalMemory:
    """

    CUDA external memory

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef cydriver.CUexternalMemory  _pvt_val
    cdef cydriver.CUexternalMemory* _pvt_ptr

cdef class CUexternalSemaphore:
    """

    CUDA external semaphore

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef cydriver.CUexternalSemaphore  _pvt_val
    cdef cydriver.CUexternalSemaphore* _pvt_ptr

cdef class CUgraph:
    """

    CUDA graph

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef cydriver.CUgraph  _pvt_val
    cdef cydriver.CUgraph* _pvt_ptr

cdef class CUgraphNode:
    """

    CUDA graph node

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef cydriver.CUgraphNode  _pvt_val
    cdef cydriver.CUgraphNode* _pvt_ptr

cdef class CUgraphExec:
    """

    CUDA executable graph

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef cydriver.CUgraphExec  _pvt_val
    cdef cydriver.CUgraphExec* _pvt_ptr

cdef class CUmemoryPool:
    """

    CUDA memory pool

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef cydriver.CUmemoryPool  _pvt_val
    cdef cydriver.CUmemoryPool* _pvt_ptr

cdef class CUuserObject:
    """

    CUDA user object for graphs

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef cydriver.CUuserObject  _pvt_val
    cdef cydriver.CUuserObject* _pvt_ptr

cdef class CUgraphDeviceNode:
    """

    CUDA graph device node handle

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef cydriver.CUgraphDeviceNode  _pvt_val
    cdef cydriver.CUgraphDeviceNode* _pvt_ptr

cdef class CUasyncCallbackHandle:
    """

    CUDA async notification callback handle

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef cydriver.CUasyncCallbackHandle  _pvt_val
    cdef cydriver.CUasyncCallbackHandle* _pvt_ptr

cdef class CUgreenCtx:
    """

    A green context handle. This handle can be used safely from only one CPU thread at a time. Created via cuGreenCtxCreate

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef cydriver.CUgreenCtx  _pvt_val
    cdef cydriver.CUgreenCtx* _pvt_ptr

cdef class CUlinkState:
    """

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef cydriver.CUlinkState  _pvt_val
    cdef cydriver.CUlinkState* _pvt_ptr
    cdef list _keepalive

cdef class CUcoredumpCallbackHandle:
    """ Opaque handle representing a registered coredump status callback.

    This handle is returned when registering a callback and must be provided when deregistering the callback.

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef cydriver.CUcoredumpCallbackHandle  _pvt_val
    cdef cydriver.CUcoredumpCallbackHandle* _pvt_ptr

cdef class CUdevResourceDesc:
    """

    An opaque descriptor handle. The descriptor encapsulates multiple created and configured resources. Created via cuDevResourceGenerateDesc

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef cydriver.CUdevResourceDesc  _pvt_val
    cdef cydriver.CUdevResourceDesc* _pvt_ptr

cdef class CUlogsCallbackHandle:
    """

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef cydriver.CUlogsCallbackHandle  _pvt_val
    cdef cydriver.CUlogsCallbackHandle* _pvt_ptr

cdef class CUeglStreamConnection:
    """

    CUDA EGLSream Connection

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef cydriver.CUeglStreamConnection  _pvt_val
    cdef cydriver.CUeglStreamConnection* _pvt_ptr

cdef class EGLImageKHR:
    """

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef cydriver.EGLImageKHR  _pvt_val
    cdef cydriver.EGLImageKHR* _pvt_ptr

cdef class EGLStreamKHR:
    """

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef cydriver.EGLStreamKHR  _pvt_val
    cdef cydriver.EGLStreamKHR* _pvt_ptr

cdef class EGLSyncKHR:
    """

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef cydriver.EGLSyncKHR  _pvt_val
    cdef cydriver.EGLSyncKHR* _pvt_ptr

cdef class CUasyncCallback:
    """

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef cydriver.CUasyncCallback  _pvt_val
    cdef cydriver.CUasyncCallback* _pvt_ptr

cdef class CUhostFn:
    """

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef cydriver.CUhostFn  _pvt_val
    cdef cydriver.CUhostFn* _pvt_ptr

cdef class CUstreamCallback:
    """

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef cydriver.CUstreamCallback  _pvt_val
    cdef cydriver.CUstreamCallback* _pvt_ptr

cdef class CUoccupancyB2DSize:
    """

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef cydriver.CUoccupancyB2DSize  _pvt_val
    cdef cydriver.CUoccupancyB2DSize* _pvt_ptr

cdef class CUcoredumpStatusCallback:
    """

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef cydriver.CUcoredumpStatusCallback  _pvt_val
    cdef cydriver.CUcoredumpStatusCallback* _pvt_ptr

cdef class CUlogsCallback:
    """

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef cydriver.CUlogsCallback  _pvt_val
    cdef cydriver.CUlogsCallback* _pvt_ptr

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
    cdef cydriver.CUuuid_st _pvt_val
    cdef cydriver.CUuuid_st* _pvt_ptr

cdef class CUmemFabricHandle_st:
    """
    Fabric handle - An opaque handle representing a memory allocation
    that can be exported to processes in same or different nodes. For
    IPC between processes on different nodes they must be connected via
    the NVSwitch fabric.

    Attributes
    ----------

    data : bytes



    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cydriver.CUmemFabricHandle_st _pvt_val
    cdef cydriver.CUmemFabricHandle_st* _pvt_ptr

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
    cdef cydriver.CUipcEventHandle_st _pvt_val
    cdef cydriver.CUipcEventHandle_st* _pvt_ptr

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
    cdef cydriver.CUipcMemHandle_st _pvt_val
    cdef cydriver.CUipcMemHandle_st* _pvt_ptr

cdef class CUstreamMemOpWaitValueParams_st:
    """
    Attributes
    ----------

    operation : CUstreamBatchMemOpType



    address : CUdeviceptr



    value : cuuint32_t



    value64 : cuuint64_t



    flags : unsigned int
        See CUstreamWaitValue_flags.


    alias : CUdeviceptr
        For driver internal use. Initial value is unimportant.


    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cydriver.CUstreamBatchMemOpParams_union* _pvt_ptr

    cdef CUdeviceptr _address


    cdef cuuint32_t _value


    cdef cuuint64_t _value64


    cdef CUdeviceptr _alias


cdef class CUstreamMemOpWriteValueParams_st:
    """
    Attributes
    ----------

    operation : CUstreamBatchMemOpType



    address : CUdeviceptr



    value : cuuint32_t



    value64 : cuuint64_t



    flags : unsigned int
        See CUstreamWriteValue_flags.


    alias : CUdeviceptr
        For driver internal use. Initial value is unimportant.


    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cydriver.CUstreamBatchMemOpParams_union* _pvt_ptr

    cdef CUdeviceptr _address


    cdef cuuint32_t _value


    cdef cuuint64_t _value64


    cdef CUdeviceptr _alias


cdef class CUstreamMemOpFlushRemoteWritesParams_st:
    """
    Attributes
    ----------

    operation : CUstreamBatchMemOpType



    flags : unsigned int
        Must be 0.


    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cydriver.CUstreamBatchMemOpParams_union* _pvt_ptr

cdef class CUstreamMemOpMemoryBarrierParams_st:
    """
    Attributes
    ----------

    operation : CUstreamBatchMemOpType
        < Only supported in the _v2 API


    flags : unsigned int
        See CUstreamMemoryBarrier_flags


    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cydriver.CUstreamBatchMemOpParams_union* _pvt_ptr

cdef class CUstreamMemOpAtomicReductionParams_st:
    """
    Attributes
    ----------

    operation : CUstreamBatchMemOpType



    flags : unsigned int
        Must be 0


    reductionOp : CUstreamAtomicReductionOpType
        See CUstreamAtomicReductionOpType


    dataType : CUstreamAtomicReductionDataType
        See CUstreamAtomicReductionDataType


    address : CUdeviceptr
        The address the atomic operation will be operated on


    value : cuuint64_t
        The operand value the atomic operation will operate with


    alias : CUdeviceptr
        For driver internal use. Initial value is unimportant.


    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cydriver.CUstreamBatchMemOpParams_union* _pvt_ptr

    cdef CUdeviceptr _address


    cdef cuuint64_t _value


    cdef CUdeviceptr _alias


cdef class CUstreamBatchMemOpParams_union:
    """
    Per-operation parameters for cuStreamBatchMemOp

    Attributes
    ----------

    operation : CUstreamBatchMemOpType
        Operation. This is the first field of all the union elemets and
        acts as a TAG to determine which union member is valid.


    waitValue : CUstreamMemOpWaitValueParams_st
        Params for CU_STREAM_MEM_OP_WAIT_VALUE_32 and
        CU_STREAM_MEM_OP_WAIT_VALUE_64 operations.


    writeValue : CUstreamMemOpWriteValueParams_st
        Params for CU_STREAM_MEM_OP_WRITE_VALUE_32 and
        CU_STREAM_MEM_OP_WRITE_VALUE_64 operations.


    flushRemoteWrites : CUstreamMemOpFlushRemoteWritesParams_st
        Params for CU_STREAM_MEM_OP_FLUSH_REMOTE_WRITES operations.


    memoryBarrier : CUstreamMemOpMemoryBarrierParams_st
        Params for CU_STREAM_MEM_OP_BARRIER operations.


    atomicReduction : CUstreamMemOpAtomicReductionParams_st



    pad : list[cuuint64_t]



    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cydriver.CUstreamBatchMemOpParams_union _pvt_val
    cdef cydriver.CUstreamBatchMemOpParams_union* _pvt_ptr

    cdef CUstreamMemOpWaitValueParams_st _waitValue


    cdef CUstreamMemOpWriteValueParams_st _writeValue


    cdef CUstreamMemOpFlushRemoteWritesParams_st _flushRemoteWrites


    cdef CUstreamMemOpMemoryBarrierParams_st _memoryBarrier


    cdef CUstreamMemOpAtomicReductionParams_st _atomicReduction


cdef class CUDA_BATCH_MEM_OP_NODE_PARAMS_v1_st:
    """
    Batch memory operation node parameters  Used in the legacy
    cuGraphAddBatchMemOpNode api. New code should use cuGraphAddNode()

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
    cdef cydriver.CUDA_BATCH_MEM_OP_NODE_PARAMS_v1_st _pvt_val
    cdef cydriver.CUDA_BATCH_MEM_OP_NODE_PARAMS_v1_st* _pvt_ptr

    cdef CUcontext _ctx


    cdef size_t _paramArray_length
    cdef cydriver.CUstreamBatchMemOpParams* _paramArray


cdef class CUDA_BATCH_MEM_OP_NODE_PARAMS_v2_st:
    """
    Batch memory operation node parameters

    Attributes
    ----------

    ctx : CUcontext
        Context to use for the operations.


    count : unsigned int
        Number of operations in paramArray.


    paramArray : CUstreamBatchMemOpParams
        Array of batch memory operations.


    flags : unsigned int
        Flags to control the node.


    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cydriver.CUDA_BATCH_MEM_OP_NODE_PARAMS_v2_st _pvt_val
    cdef cydriver.CUDA_BATCH_MEM_OP_NODE_PARAMS_v2_st* _pvt_ptr

    cdef CUcontext _ctx


    cdef size_t _paramArray_length
    cdef cydriver.CUstreamBatchMemOpParams* _paramArray


cdef class anon_struct0:
    """
    Attributes
    ----------

    bytesOverBudget : unsigned long long



    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cydriver.CUasyncNotificationInfo_st* _pvt_ptr

cdef class anon_union2:
    """
    Attributes
    ----------

    overBudget : anon_struct0



    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cydriver.CUasyncNotificationInfo_st* _pvt_ptr

    cdef anon_struct0 _overBudget


cdef class CUasyncNotificationInfo_st:
    """
    Information passed to the user via the async notification callback

    Attributes
    ----------

    type : CUasyncNotificationType
        The type of notification being sent


    info : anon_union2
        Information about the notification. `typename` must be checked in
        order to interpret this field.


    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cydriver.CUasyncNotificationInfo_st* _val_ptr
    cdef cydriver.CUasyncNotificationInfo_st* _pvt_ptr

    cdef anon_union2 _info


cdef class CUdevprop_st:
    """
    Legacy device properties

    Attributes
    ----------

    maxThreadsPerBlock : int
        Maximum number of threads per block


    maxThreadsDim : list[int]
        Maximum size of each dimension of a block


    maxGridSize : list[int]
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
    cdef cydriver.CUdevprop_st _pvt_val
    cdef cydriver.CUdevprop_st* _pvt_ptr

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
    cdef cydriver.CUaccessPolicyWindow_st _pvt_val
    cdef cydriver.CUaccessPolicyWindow_st* _pvt_ptr

    cdef _HelperInputVoidPtr _cybase_ptr


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
    cdef cydriver.CUDA_KERNEL_NODE_PARAMS_st _pvt_val
    cdef cydriver.CUDA_KERNEL_NODE_PARAMS_st* _pvt_ptr

    cdef CUfunction _func


    cdef _HelperKernelParams _cykernelParams


cdef class CUDA_KERNEL_NODE_PARAMS_v2_st:
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


    kern : CUkernel
        Kernel to launch, will only be referenced if func is NULL


    ctx : CUcontext
        Context for the kernel task to run in. The value NULL will indicate
        the current context should be used by the api. This field is
        ignored if func is set.


    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cydriver.CUDA_KERNEL_NODE_PARAMS_v2_st _pvt_val
    cdef cydriver.CUDA_KERNEL_NODE_PARAMS_v2_st* _pvt_ptr

    cdef CUfunction _func


    cdef _HelperKernelParams _cykernelParams


    cdef CUkernel _kern


    cdef CUcontext _ctx


cdef class CUDA_KERNEL_NODE_PARAMS_v3_st:
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


    kern : CUkernel
        Kernel to launch, will only be referenced if func is NULL


    ctx : CUcontext
        Context for the kernel task to run in. The value NULL will indicate
        the current context should be used by the api. This field is
        ignored if func is set.


    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cydriver.CUDA_KERNEL_NODE_PARAMS_v3_st _pvt_val
    cdef cydriver.CUDA_KERNEL_NODE_PARAMS_v3_st* _pvt_ptr

    cdef CUfunction _func


    cdef _HelperKernelParams _cykernelParams


    cdef CUkernel _kern


    cdef CUcontext _ctx


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
    cdef cydriver.CUDA_MEMSET_NODE_PARAMS_st _pvt_val
    cdef cydriver.CUDA_MEMSET_NODE_PARAMS_st* _pvt_ptr

    cdef CUdeviceptr _dst


cdef class CUDA_MEMSET_NODE_PARAMS_v2_st:
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


    ctx : CUcontext
        Context on which to run the node


    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cydriver.CUDA_MEMSET_NODE_PARAMS_v2_st _pvt_val
    cdef cydriver.CUDA_MEMSET_NODE_PARAMS_v2_st* _pvt_ptr

    cdef CUdeviceptr _dst


    cdef CUcontext _ctx


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
    cdef cydriver.CUDA_HOST_NODE_PARAMS_st _pvt_val
    cdef cydriver.CUDA_HOST_NODE_PARAMS_st* _pvt_ptr

    cdef CUhostFn _fn


    cdef _HelperInputVoidPtr _cyuserData


cdef class CUDA_HOST_NODE_PARAMS_v2_st:
    """
    Host node parameters

    Attributes
    ----------

    fn : CUhostFn
        The function to call when the node executes


    userData : Any
        Argument to pass to the function


    syncMode : unsigned int
        The sync mode to use for the host task


    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cydriver.CUDA_HOST_NODE_PARAMS_v2_st _pvt_val
    cdef cydriver.CUDA_HOST_NODE_PARAMS_v2_st* _pvt_ptr

    cdef CUhostFn _fn


    cdef _HelperInputVoidPtr _cyuserData


cdef class CUDA_CONDITIONAL_NODE_PARAMS:
    """
    Conditional node parameters

    Attributes
    ----------

    handle : CUgraphConditionalHandle
        Conditional node handle. Handles must be created in advance of
        creating the node using cuGraphConditionalHandleCreate.


    type : CUgraphConditionalNodeType
        Type of conditional node.


    size : unsigned int
        Size of graph output array. Allowed values are 1 for
        CU_GRAPH_COND_TYPE_WHILE, 1 or 2 for CU_GRAPH_COND_TYPE_IF, or any
        value greater than zero for CU_GRAPH_COND_TYPE_SWITCH.


    phGraph_out : CUgraph
        CUDA-owned array populated with conditional node child graphs
        during creation of the node. Valid for the lifetime of the
        conditional node. The contents of the graph(s) are subject to the
        following constraints:   - Allowed node types are kernel nodes,
        empty nodes, child graphs, memsets, memcopies, and conditionals.
        This applies recursively to child graphs and conditional bodies.
        - All kernels, including kernels in nested conditionals or child
        graphs at any level, must belong to the same CUDA context.
        These graphs may be populated using graph node creation APIs or
        cuStreamBeginCaptureToGraph.  CU_GRAPH_COND_TYPE_IF: phGraph_out[0]
        is executed when the condition is non-zero. If `size` == 2,
        phGraph_out[1] will be executed when the condition is zero.
        CU_GRAPH_COND_TYPE_WHILE: phGraph_out[0] is executed as long as the
        condition is non-zero. CU_GRAPH_COND_TYPE_SWITCH: phGraph_out[n] is
        executed when the condition is equal to n. If the condition >=
        `size`, no body graph is executed.


    ctx : CUcontext
        Context on which to run the node. Must match context used to create
        the handle and all body nodes.


    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cydriver.CUDA_CONDITIONAL_NODE_PARAMS _pvt_val
    cdef cydriver.CUDA_CONDITIONAL_NODE_PARAMS* _pvt_ptr

    cdef CUgraphConditionalHandle _handle


    cdef size_t _phGraph_out_length
    cdef cydriver.CUgraph* _phGraph_out


    cdef CUcontext _ctx


cdef class CUgraphEdgeData_st:
    """
    Optional annotation for edges in a CUDA graph. Note, all edges
    implicitly have annotations and default to a zero-initialized value
    if not specified. A zero-initialized struct indicates a standard
    full serialization of two nodes with memory visibility.

    Attributes
    ----------

    from_port : bytes
        This indicates when the dependency is triggered from the upstream
        node on the edge. The meaning is specfic to the node type. A value
        of 0 in all cases means full completion of the upstream node, with
        memory visibility to the downstream node or portion thereof
        (indicated by `to_port`).   Only kernel nodes define non-zero
        ports. A kernel node can use the following output port types:
        CU_GRAPH_KERNEL_NODE_PORT_DEFAULT,
        CU_GRAPH_KERNEL_NODE_PORT_PROGRAMMATIC, or
        CU_GRAPH_KERNEL_NODE_PORT_LAUNCH_ORDER.


    to_port : bytes
        This indicates what portion of the downstream node is dependent on
        the upstream node or portion thereof (indicated by `from_port`).
        The meaning is specific to the node type. A value of 0 in all cases
        means the entirety of the downstream node is dependent on the
        upstream work.   Currently no node types define non-zero ports.
        Accordingly, this field must be set to zero.


    type : bytes
        This should be populated with a value from CUgraphDependencyType.
        (It is typed as char due to compiler-specific layout of bitfields.)
        See CUgraphDependencyType.


    reserved : bytes
        These bytes are unused and must be zeroed. This ensures
        compatibility if additional fields are added in the future.


    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cydriver.CUgraphEdgeData_st _pvt_val
    cdef cydriver.CUgraphEdgeData_st* _pvt_ptr

cdef class CUDA_GRAPH_INSTANTIATE_PARAMS_st:
    """
    Graph instantiation parameters

    Attributes
    ----------

    flags : cuuint64_t
        Instantiation flags


    hUploadStream : CUstream
        Upload stream


    hErrNode_out : CUgraphNode
        The node which caused instantiation to fail, if any


    result_out : CUgraphInstantiateResult
        Whether instantiation was successful. If it failed, the reason why


    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cydriver.CUDA_GRAPH_INSTANTIATE_PARAMS_st _pvt_val
    cdef cydriver.CUDA_GRAPH_INSTANTIATE_PARAMS_st* _pvt_ptr

    cdef cuuint64_t _flags


    cdef CUstream _hUploadStream


    cdef CUgraphNode _hErrNode_out


cdef class CUlaunchMemSyncDomainMap_st:
    """
    Memory Synchronization Domain map  See ::cudaLaunchMemSyncDomain.
    By default, kernels are launched in domain 0. Kernel launched with
    CU_LAUNCH_MEM_SYNC_DOMAIN_REMOTE will have a different domain ID.
    User may also alter the domain ID with CUlaunchMemSyncDomainMap for
    a specific stream / graph node / kernel launch. See
    CU_LAUNCH_ATTRIBUTE_MEM_SYNC_DOMAIN_MAP.  Domain ID range is
    available through CU_DEVICE_ATTRIBUTE_MEM_SYNC_DOMAIN_COUNT.

    Attributes
    ----------

    default_ : bytes
        The default domain ID to use for designated kernels


    remote : bytes
        The remote domain ID to use for designated kernels


    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cydriver.CUlaunchMemSyncDomainMap_st _pvt_val
    cdef cydriver.CUlaunchMemSyncDomainMap_st* _pvt_ptr

cdef class anon_struct1:
    """
    Attributes
    ----------

    x : unsigned int



    y : unsigned int



    z : unsigned int



    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cydriver.CUlaunchAttributeValue_union* _pvt_ptr

cdef class anon_struct2:
    """
    Attributes
    ----------

    event : CUevent



    flags : int



    triggerAtBlockStart : int



    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cydriver.CUlaunchAttributeValue_union* _pvt_ptr

    cdef CUevent _event


cdef class anon_struct3:
    """
    Attributes
    ----------

    event : CUevent



    flags : int



    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cydriver.CUlaunchAttributeValue_union* _pvt_ptr

    cdef CUevent _event


cdef class anon_struct4:
    """
    Attributes
    ----------

    x : unsigned int



    y : unsigned int



    z : unsigned int



    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cydriver.CUlaunchAttributeValue_union* _pvt_ptr

cdef class anon_struct5:
    """
    Attributes
    ----------

    deviceUpdatable : int



    devNode : CUgraphDeviceNode



    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cydriver.CUlaunchAttributeValue_union* _pvt_ptr

    cdef CUgraphDeviceNode _devNode


cdef class CUlaunchAttributeValue_union:
    """
    Launch attributes union; used as value field of CUlaunchAttribute

    Attributes
    ----------

    pad : bytes



    accessPolicyWindow : CUaccessPolicyWindow
        Value of launch attribute CU_LAUNCH_ATTRIBUTE_ACCESS_POLICY_WINDOW.


    cooperative : int
        Value of launch attribute CU_LAUNCH_ATTRIBUTE_COOPERATIVE. Nonzero
        indicates a cooperative kernel (see cuLaunchCooperativeKernel).


    syncPolicy : CUsynchronizationPolicy
        Value of launch attribute
        CU_LAUNCH_ATTRIBUTE_SYNCHRONIZATION_POLICY. CUsynchronizationPolicy
        for work queued up in this stream


    clusterDim : anon_struct1
        Value of launch attribute CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION
        that represents the desired cluster dimensions for the kernel.
        Opaque type with the following fields: - `x` - The X dimension of
        the cluster, in blocks. Must be a divisor of the grid X dimension.
        - `y` - The Y dimension of the cluster, in blocks. Must be a
        divisor of the grid Y dimension.    - `z` - The Z dimension of the
        cluster, in blocks. Must be a divisor of the grid Z dimension.


    clusterSchedulingPolicyPreference : CUclusterSchedulingPolicy
        Value of launch attribute
        CU_LAUNCH_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE. Cluster
        scheduling policy preference for the kernel.


    programmaticStreamSerializationAllowed : int
        Value of launch attribute
        CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION.


    programmaticEvent : anon_struct2
        Value of launch attribute CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_EVENT
        with the following fields: - `CUevent` event - Event to fire when
        all blocks trigger it.    - `Event` record flags, see
        cuEventRecordWithFlags. Does not accept :CU_EVENT_RECORD_EXTERNAL.
        - `triggerAtBlockStart` - If this is set to non-0, each block
        launch will automatically trigger the event.


    launchCompletionEvent : anon_struct3
        Value of launch attribute
        CU_LAUNCH_ATTRIBUTE_LAUNCH_COMPLETION_EVENT with the following
        fields: - `CUevent` event - Event to fire when the last block
        launches    - `int` flags; - Event record flags, see
        cuEventRecordWithFlags. Does not accept CU_EVENT_RECORD_EXTERNAL.


    priority : int
        Value of launch attribute CU_LAUNCH_ATTRIBUTE_PRIORITY. Execution
        priority of the kernel.


    memSyncDomainMap : CUlaunchMemSyncDomainMap
        Value of launch attribute CU_LAUNCH_ATTRIBUTE_MEM_SYNC_DOMAIN_MAP.
        See CUlaunchMemSyncDomainMap.


    memSyncDomain : CUlaunchMemSyncDomain
        Value of launch attribute CU_LAUNCH_ATTRIBUTE_MEM_SYNC_DOMAIN.
        See::CUlaunchMemSyncDomain


    preferredClusterDim : anon_struct4
        Value of launch attribute
        CU_LAUNCH_ATTRIBUTE_PREFERRED_CLUSTER_DIMENSION that represents the
        desired preferred cluster dimensions for the kernel. Opaque type
        with the following fields: - `x` - The X dimension of the preferred
        cluster, in blocks. Must be a divisor of the grid X dimension, and
        must be a multiple of the `x` field of
        CUlaunchAttributeValue::clusterDim.    - `y` - The Y dimension of
        the preferred cluster, in blocks. Must be a divisor of the grid Y
        dimension, and must be a multiple of the `y` field of
        CUlaunchAttributeValue::clusterDim.    - `z` - The Z dimension of
        the preferred cluster, in blocks. Must be equal to the `z` field of
        CUlaunchAttributeValue::clusterDim.


    deviceUpdatableKernelNode : anon_struct5
        Value of launch attribute
        CU_LAUNCH_ATTRIBUTE_DEVICE_UPDATABLE_KERNEL_NODE. with the
        following fields: - `int` deviceUpdatable - Whether or not the
        resulting kernel node should be device-updatable.    -
        `CUgraphDeviceNode` devNode - Returns a handle to pass to the
        various device-side update functions.


    sharedMemCarveout : unsigned int
        Value of launch attribute
        CU_LAUNCH_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT.


    nvlinkUtilCentricScheduling : unsigned int



    portableClusterSizeMode : CUlaunchAttributePortableClusterMode
        Value of launch attribute
        CU_LAUNCH_ATTRIBUTE_PORTABLE_CLUSTER_SIZE_MODE.


    sharedMemoryMode : CUsharedMemoryMode
        Value of launch attribute CU_LAUNCH_ATTRIBUTE_SHARED_MEMORY_MODE.
        See CUsharedMemoryMode for acceptable values.


    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cydriver.CUlaunchAttributeValue_union _pvt_val
    cdef cydriver.CUlaunchAttributeValue_union* _pvt_ptr

    cdef CUaccessPolicyWindow _accessPolicyWindow


    cdef anon_struct1 _clusterDim


    cdef anon_struct2 _programmaticEvent


    cdef anon_struct3 _launchCompletionEvent


    cdef CUlaunchMemSyncDomainMap _memSyncDomainMap


    cdef anon_struct4 _preferredClusterDim


    cdef anon_struct5 _deviceUpdatableKernelNode


cdef class CUlaunchAttribute_st:
    """
    Launch attribute

    Attributes
    ----------

    id : CUlaunchAttributeID
        Attribute to set


    value : CUlaunchAttributeValue
        Value of the attribute


    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cydriver.CUlaunchAttribute_st _pvt_val
    cdef cydriver.CUlaunchAttribute_st* _pvt_ptr

    cdef CUlaunchAttributeValue _value


cdef class CUlaunchConfig_st:
    """
    CUDA extensible launch configuration

    Attributes
    ----------

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


    attrs : CUlaunchAttribute
        List of attributes; nullable if CUlaunchConfig::numAttrs == 0


    numAttrs : unsigned int
        Number of attributes populated in CUlaunchConfig::attrs


    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cydriver.CUlaunchConfig_st _pvt_val
    cdef cydriver.CUlaunchConfig_st* _pvt_ptr

    cdef CUstream _hStream


    cdef size_t _attrs_length
    cdef cydriver.CUlaunchAttribute* _attrs


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
    cdef cydriver.CUexecAffinitySmCount_st _pvt_val
    cdef cydriver.CUexecAffinitySmCount_st* _pvt_ptr

cdef class anon_union3:
    """
    Attributes
    ----------

    smCount : CUexecAffinitySmCount



    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cydriver.CUexecAffinityParam_st* _pvt_ptr

    cdef CUexecAffinitySmCount _smCount


cdef class CUexecAffinityParam_st:
    """
    Execution Affinity Parameters

    Attributes
    ----------

    type : CUexecAffinityType
        Type of execution affinity.


    param : anon_union3



    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cydriver.CUexecAffinityParam_st* _val_ptr
    cdef cydriver.CUexecAffinityParam_st* _pvt_ptr

    cdef anon_union3 _param


cdef class CUctxCigParam_st:
    """
    CIG Context Create Params

    Attributes
    ----------

    sharedDataType : CUcigDataType
        Type of shared data from graphics client (D3D12 or Vulkan).


    sharedData : Any
        Graphics client data handle (ID3D12CommandQueue or Nvidia specific
        data blob).


    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cydriver.CUctxCigParam_st _pvt_val
    cdef cydriver.CUctxCigParam_st* _pvt_ptr

    cdef _HelperInputVoidPtr _cysharedData


cdef class CUctxCreateParams_st:
    """
    Params for creating CUDA context. Both execAffinityParams and
    cigParams cannot be non-NULL at the same time. If both are NULL,
    the context will be created as a regular CUDA context.

    Attributes
    ----------

    execAffinityParams : CUexecAffinityParam
        Array of execution affinity parameters to limit context resources
        (e.g., SM count). Only supported Volta+ MPS. Mutually exclusive
        with cigParams.


    numExecAffinityParams : int
        Number of elements in execAffinityParams array. Must be 0 if
        execAffinityParams is NULL.


    cigParams : CUctxCigParam
        CIG (CUDA in Graphics) parameters for sharing data from
        D3D12/Vulkan graphics clients. Mutually exclusive with
        execAffinityParams.


    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cydriver.CUctxCreateParams_st _pvt_val
    cdef cydriver.CUctxCreateParams_st* _pvt_ptr

    cdef size_t _execAffinityParams_length
    cdef cydriver.CUexecAffinityParam* _execAffinityParams


    cdef size_t _cigParams_length
    cdef cydriver.CUctxCigParam* _cigParams


cdef class CUstreamCigParam_st:
    """
    CIG Stream Capture Params

    Attributes
    ----------

    streamSharedDataType : CUstreamCigDataType
        Type of shared data from graphics client (D3D12).


    streamSharedData : Any
        Graphics client data handle
        (ID3D12CommandList/ID3D12GraphicsCommandList).


    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cydriver.CUstreamCigParam_st _pvt_val
    cdef cydriver.CUstreamCigParam_st* _pvt_ptr

    cdef _HelperInputVoidPtr _cystreamSharedData


cdef class CUstreamCigCaptureParams_st:
    """
    Params for capturing CUDA stream to CIG streamCigParams must be
    non-NULL.

    Attributes
    ----------

    streamCigParams : CUstreamCigParam
        CIG (CUDA in Graphics) parameters for sharing command list data
        from D3D12 graphics clients.


    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cydriver.CUstreamCigCaptureParams_st _pvt_val
    cdef cydriver.CUstreamCigCaptureParams_st* _pvt_ptr

    cdef size_t _streamCigParams_length
    cdef cydriver.CUstreamCigParam* _streamCigParams


cdef class CUlibraryHostUniversalFunctionAndDataTable_st:
    """
    Attributes
    ----------

    functionTable : Any



    functionWindowSize : size_t



    dataTable : Any



    dataWindowSize : size_t



    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cydriver.CUlibraryHostUniversalFunctionAndDataTable_st _pvt_val
    cdef cydriver.CUlibraryHostUniversalFunctionAndDataTable_st* _pvt_ptr

    cdef _HelperInputVoidPtr _cyfunctionTable


    cdef _HelperInputVoidPtr _cydataTable


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
    cdef cydriver.CUDA_MEMCPY2D_st _pvt_val
    cdef cydriver.CUDA_MEMCPY2D_st* _pvt_ptr

    cdef _HelperInputVoidPtr _cysrcHost


    cdef CUdeviceptr _srcDevice


    cdef CUarray _srcArray


    cdef _HelperInputVoidPtr _cydstHost


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
    cdef cydriver.CUDA_MEMCPY3D_st _pvt_val
    cdef cydriver.CUDA_MEMCPY3D_st* _pvt_ptr

    cdef _HelperInputVoidPtr _cysrcHost


    cdef CUdeviceptr _srcDevice


    cdef CUarray _srcArray


    cdef _HelperInputVoidPtr _cyreserved0


    cdef _HelperInputVoidPtr _cydstHost


    cdef CUdeviceptr _dstDevice


    cdef CUarray _dstArray


    cdef _HelperInputVoidPtr _cyreserved1


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
    cdef cydriver.CUDA_MEMCPY3D_PEER_st _pvt_val
    cdef cydriver.CUDA_MEMCPY3D_PEER_st* _pvt_ptr

    cdef _HelperInputVoidPtr _cysrcHost


    cdef CUdeviceptr _srcDevice


    cdef CUarray _srcArray


    cdef CUcontext _srcContext


    cdef _HelperInputVoidPtr _cydstHost


    cdef CUdeviceptr _dstDevice


    cdef CUarray _dstArray


    cdef CUcontext _dstContext


cdef class CUDA_MEMCPY_NODE_PARAMS_st:
    """
    Memcpy node parameters

    Attributes
    ----------

    flags : int
        Must be zero


    reserved : int
        Must be zero


    copyCtx : CUcontext
        Context on which to run the node


    copyParams : CUDA_MEMCPY3D
        Parameters for the memory copy


    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cydriver.CUDA_MEMCPY_NODE_PARAMS_st _pvt_val
    cdef cydriver.CUDA_MEMCPY_NODE_PARAMS_st* _pvt_ptr

    cdef CUcontext _copyCtx


    cdef CUDA_MEMCPY3D _copyParams


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
    cdef cydriver.CUDA_ARRAY_DESCRIPTOR_st _pvt_val
    cdef cydriver.CUDA_ARRAY_DESCRIPTOR_st* _pvt_ptr

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
    cdef cydriver.CUDA_ARRAY3D_DESCRIPTOR_st _pvt_val
    cdef cydriver.CUDA_ARRAY3D_DESCRIPTOR_st* _pvt_ptr

cdef class anon_struct6:
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
    cdef cydriver.CUDA_ARRAY_SPARSE_PROPERTIES_st* _pvt_ptr

cdef class CUDA_ARRAY_SPARSE_PROPERTIES_st:
    """
    CUDA array sparse properties

    Attributes
    ----------

    tileExtent : anon_struct6



    miptailFirstLevel : unsigned int
        First mip level at which the mip tail begins.


    miptailSize : unsigned long long
        Total size of the mip tail.


    flags : unsigned int
        Flags will either be zero or
        CU_ARRAY_SPARSE_PROPERTIES_SINGLE_MIPTAIL


    reserved : list[unsigned int]



    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cydriver.CUDA_ARRAY_SPARSE_PROPERTIES_st _pvt_val
    cdef cydriver.CUDA_ARRAY_SPARSE_PROPERTIES_st* _pvt_ptr

    cdef anon_struct6 _tileExtent


cdef class CUDA_ARRAY_MEMORY_REQUIREMENTS_st:
    """
    CUDA array memory requirements

    Attributes
    ----------

    size : size_t
        Total required memory size


    alignment : size_t
        alignment requirement


    reserved : list[unsigned int]



    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cydriver.CUDA_ARRAY_MEMORY_REQUIREMENTS_st _pvt_val
    cdef cydriver.CUDA_ARRAY_MEMORY_REQUIREMENTS_st* _pvt_ptr

cdef class anon_struct7:
    """
    Attributes
    ----------

    hArray : CUarray



    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cydriver.CUDA_RESOURCE_DESC_st* _pvt_ptr

    cdef CUarray _hArray


cdef class anon_struct8:
    """
    Attributes
    ----------

    hMipmappedArray : CUmipmappedArray



    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cydriver.CUDA_RESOURCE_DESC_st* _pvt_ptr

    cdef CUmipmappedArray _hMipmappedArray


cdef class anon_struct9:
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
    cdef cydriver.CUDA_RESOURCE_DESC_st* _pvt_ptr

    cdef CUdeviceptr _devPtr


cdef class anon_struct10:
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
    cdef cydriver.CUDA_RESOURCE_DESC_st* _pvt_ptr

    cdef CUdeviceptr _devPtr


cdef class anon_struct11:
    """
    Attributes
    ----------

    reserved : list[int]



    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cydriver.CUDA_RESOURCE_DESC_st* _pvt_ptr

cdef class anon_union4:
    """
    Attributes
    ----------

    array : anon_struct7



    mipmap : anon_struct8



    linear : anon_struct9



    pitch2D : anon_struct10



    reserved : anon_struct11



    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cydriver.CUDA_RESOURCE_DESC_st* _pvt_ptr

    cdef anon_struct7 _array


    cdef anon_struct8 _mipmap


    cdef anon_struct9 _linear


    cdef anon_struct10 _pitch2D


    cdef anon_struct11 _reserved


cdef class CUDA_RESOURCE_DESC_st:
    """
    CUDA Resource descriptor

    Attributes
    ----------

    resType : CUresourcetype
        Resource type


    res : anon_union4



    flags : unsigned int
        Flags (must be zero)


    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cydriver.CUDA_RESOURCE_DESC_st* _val_ptr
    cdef cydriver.CUDA_RESOURCE_DESC_st* _pvt_ptr

    cdef anon_union4 _res


cdef class CUDA_TEXTURE_DESC_st:
    """
    Texture descriptor

    Attributes
    ----------

    addressMode : list[CUaddress_mode]
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


    borderColor : list[float]
        Border Color


    reserved : list[int]



    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cydriver.CUDA_TEXTURE_DESC_st _pvt_val
    cdef cydriver.CUDA_TEXTURE_DESC_st* _pvt_ptr

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


    reserved : list[unsigned int]



    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cydriver.CUDA_RESOURCE_VIEW_DESC_st _pvt_val
    cdef cydriver.CUDA_RESOURCE_VIEW_DESC_st* _pvt_ptr

cdef class CUtensorMap_st:
    """
    Tensor map descriptor. Requires compiler support for aligning to
    128 bytes.

    Attributes
    ----------

    opaque : list[cuuint64_t]



    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cydriver.CUtensorMap_st _pvt_val
    cdef cydriver.CUtensorMap_st* _pvt_ptr

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
    cdef cydriver.CUDA_POINTER_ATTRIBUTE_P2P_TOKENS_st _pvt_val
    cdef cydriver.CUDA_POINTER_ATTRIBUTE_P2P_TOKENS_st* _pvt_ptr

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
    cdef cydriver.CUDA_LAUNCH_PARAMS_st _pvt_val
    cdef cydriver.CUDA_LAUNCH_PARAMS_st* _pvt_ptr

    cdef CUfunction _function


    cdef CUstream _hStream


    cdef _HelperKernelParams _cykernelParams


cdef class anon_struct12:
    """
    Attributes
    ----------

    handle : Any



    name : Any



    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cydriver.CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st* _pvt_ptr

    cdef _HelperInputVoidPtr _cyhandle


    cdef _HelperInputVoidPtr _cyname


cdef class anon_union5:
    """
    Attributes
    ----------

    fd : int



    win32 : anon_struct12



    nvSciBufObject : Any



    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cydriver.CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st* _pvt_ptr

    cdef anon_struct12 _win32


    cdef _HelperInputVoidPtr _cynvSciBufObject


cdef class CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st:
    """
    External memory handle descriptor

    Attributes
    ----------

    type : CUexternalMemoryHandleType
        Type of the handle


    handle : anon_union5



    size : unsigned long long
        Size of the memory allocation


    flags : unsigned int
        Flags must either be zero or CUDA_EXTERNAL_MEMORY_DEDICATED


    reserved : list[unsigned int]



    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cydriver.CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st* _val_ptr
    cdef cydriver.CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st* _pvt_ptr

    cdef anon_union5 _handle


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


    reserved : list[unsigned int]



    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cydriver.CUDA_EXTERNAL_MEMORY_BUFFER_DESC_st _pvt_val
    cdef cydriver.CUDA_EXTERNAL_MEMORY_BUFFER_DESC_st* _pvt_ptr

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


    reserved : list[unsigned int]



    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cydriver.CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC_st _pvt_val
    cdef cydriver.CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC_st* _pvt_ptr

    cdef CUDA_ARRAY3D_DESCRIPTOR _arrayDesc


cdef class anon_struct13:
    """
    Attributes
    ----------

    handle : Any



    name : Any



    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cydriver.CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_st* _pvt_ptr

    cdef _HelperInputVoidPtr _cyhandle


    cdef _HelperInputVoidPtr _cyname


cdef class anon_union6:
    """
    Attributes
    ----------

    fd : int



    win32 : anon_struct13



    nvSciSyncObj : Any



    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cydriver.CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_st* _pvt_ptr

    cdef anon_struct13 _win32


    cdef _HelperInputVoidPtr _cynvSciSyncObj


cdef class CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_st:
    """
    External semaphore handle descriptor

    Attributes
    ----------

    type : CUexternalSemaphoreHandleType
        Type of the handle


    handle : anon_union6



    flags : unsigned int
        Flags reserved for the future. Must be zero.


    reserved : list[unsigned int]



    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cydriver.CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_st* _val_ptr
    cdef cydriver.CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_st* _pvt_ptr

    cdef anon_union6 _handle


cdef class anon_struct14:
    """
    Attributes
    ----------

    value : unsigned long long



    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cydriver.CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st* _pvt_ptr

cdef class anon_union7:
    """
    Attributes
    ----------

    fence : Any



    reserved : unsigned long long



    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cydriver.CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st* _pvt_ptr

    cdef _HelperInputVoidPtr _cyfence


cdef class anon_struct15:
    """
    Attributes
    ----------

    key : unsigned long long



    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cydriver.CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st* _pvt_ptr

cdef class anon_struct16:
    """
    Attributes
    ----------

    fence : anon_struct14



    nvSciSync : anon_union7



    keyedMutex : anon_struct15



    reserved : list[unsigned int]



    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cydriver.CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st* _pvt_ptr

    cdef anon_struct14 _fence


    cdef anon_union7 _nvSciSync


    cdef anon_struct15 _keyedMutex


cdef class CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st:
    """
    External semaphore signal parameters

    Attributes
    ----------

    params : anon_struct16



    flags : unsigned int
        Only when CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS is used to signal a
        CUexternalSemaphore of type
        CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_NVSCISYNC, the valid flag is
        CUDA_EXTERNAL_SEMAPHORE_SIGNAL_SKIP_NVSCIBUF_MEMSYNC which
        indicates that while signaling the CUexternalSemaphore, no memory
        synchronization operations should be performed for any external
        memory object imported as CU_EXTERNAL_MEMORY_HANDLE_TYPE_NVSCIBUF.
        For all other types of CUexternalSemaphore, flags must be zero.


    reserved : list[unsigned int]



    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cydriver.CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st _pvt_val
    cdef cydriver.CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st* _pvt_ptr

    cdef anon_struct16 _params


cdef class anon_struct17:
    """
    Attributes
    ----------

    value : unsigned long long



    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cydriver.CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st* _pvt_ptr

cdef class anon_union8:
    """
    Attributes
    ----------

    fence : Any



    reserved : unsigned long long



    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cydriver.CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st* _pvt_ptr

    cdef _HelperInputVoidPtr _cyfence


cdef class anon_struct18:
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
    cdef cydriver.CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st* _pvt_ptr

cdef class anon_struct19:
    """
    Attributes
    ----------

    fence : anon_struct17



    nvSciSync : anon_union8



    keyedMutex : anon_struct18



    reserved : list[unsigned int]



    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cydriver.CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st* _pvt_ptr

    cdef anon_struct17 _fence


    cdef anon_union8 _nvSciSync


    cdef anon_struct18 _keyedMutex


cdef class CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st:
    """
    External semaphore wait parameters

    Attributes
    ----------

    params : anon_struct19



    flags : unsigned int
        Only when CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS is used to wait on a
        CUexternalSemaphore of type
        CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_NVSCISYNC, the valid flag is
        CUDA_EXTERNAL_SEMAPHORE_WAIT_SKIP_NVSCIBUF_MEMSYNC which indicates
        that while waiting for the CUexternalSemaphore, no memory
        synchronization operations should be performed for any external
        memory object imported as CU_EXTERNAL_MEMORY_HANDLE_TYPE_NVSCIBUF.
        For all other types of CUexternalSemaphore, flags must be zero.


    reserved : list[unsigned int]



    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cydriver.CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st _pvt_val
    cdef cydriver.CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st* _pvt_ptr

    cdef anon_struct19 _params


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
    cdef cydriver.CUDA_EXT_SEM_SIGNAL_NODE_PARAMS_st _pvt_val
    cdef cydriver.CUDA_EXT_SEM_SIGNAL_NODE_PARAMS_st* _pvt_ptr

    cdef size_t _extSemArray_length
    cdef cydriver.CUexternalSemaphore* _extSemArray


    cdef size_t _paramsArray_length
    cdef cydriver.CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS* _paramsArray


cdef class CUDA_EXT_SEM_SIGNAL_NODE_PARAMS_v2_st:
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
    cdef cydriver.CUDA_EXT_SEM_SIGNAL_NODE_PARAMS_v2_st _pvt_val
    cdef cydriver.CUDA_EXT_SEM_SIGNAL_NODE_PARAMS_v2_st* _pvt_ptr

    cdef size_t _extSemArray_length
    cdef cydriver.CUexternalSemaphore* _extSemArray


    cdef size_t _paramsArray_length
    cdef cydriver.CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS* _paramsArray


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
    cdef cydriver.CUDA_EXT_SEM_WAIT_NODE_PARAMS_st _pvt_val
    cdef cydriver.CUDA_EXT_SEM_WAIT_NODE_PARAMS_st* _pvt_ptr

    cdef size_t _extSemArray_length
    cdef cydriver.CUexternalSemaphore* _extSemArray


    cdef size_t _paramsArray_length
    cdef cydriver.CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS* _paramsArray


cdef class CUDA_EXT_SEM_WAIT_NODE_PARAMS_v2_st:
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
    cdef cydriver.CUDA_EXT_SEM_WAIT_NODE_PARAMS_v2_st _pvt_val
    cdef cydriver.CUDA_EXT_SEM_WAIT_NODE_PARAMS_v2_st* _pvt_ptr

    cdef size_t _extSemArray_length
    cdef cydriver.CUexternalSemaphore* _extSemArray


    cdef size_t _paramsArray_length
    cdef cydriver.CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS* _paramsArray


cdef class anon_union9:
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
    cdef cydriver.CUarrayMapInfo_st* _pvt_ptr

    cdef CUmipmappedArray _mipmap


    cdef CUarray _array


cdef class anon_struct20:
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
    cdef cydriver.CUarrayMapInfo_st* _pvt_ptr

cdef class anon_struct21:
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
    cdef cydriver.CUarrayMapInfo_st* _pvt_ptr

cdef class anon_union10:
    """
    Attributes
    ----------

    sparseLevel : anon_struct20



    miptail : anon_struct21



    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cydriver.CUarrayMapInfo_st* _pvt_ptr

    cdef anon_struct20 _sparseLevel


    cdef anon_struct21 _miptail


cdef class anon_union11:
    """
    Attributes
    ----------

    memHandle : CUmemGenericAllocationHandle



    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cydriver.CUarrayMapInfo_st* _pvt_ptr

    cdef CUmemGenericAllocationHandle _memHandle


cdef class CUarrayMapInfo_st:
    """
    Specifies the CUDA array or CUDA mipmapped array memory mapping
    information

    Attributes
    ----------

    resourceType : CUresourcetype
        Resource type


    resource : anon_union9



    subresourceType : CUarraySparseSubresourceType
        Sparse subresource type


    subresource : anon_union10



    memOperationType : CUmemOperationType
        Memory operation type


    memHandleType : CUmemHandleType
        Memory handle type


    memHandle : anon_union11



    offset : unsigned long long
        Offset within mip tail  Offset within the memory


    deviceBitMask : unsigned int
        Device ordinal bit mask


    flags : unsigned int
        flags for future use, must be zero now.


    reserved : list[unsigned int]
        Reserved for future use, must be zero now.


    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cydriver.CUarrayMapInfo_st* _val_ptr
    cdef cydriver.CUarrayMapInfo_st* _pvt_ptr

    cdef anon_union9 _resource


    cdef anon_union10 _subresource


    cdef anon_union11 _memHandle


cdef class CUmemLocation_st:
    """
    Specifies a memory location.

    Attributes
    ----------

    type : CUmemLocationType
        Specifies the location type, which modifies the meaning of id.


    id : int



    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cydriver.CUmemLocation_st* _val_ptr
    cdef cydriver.CUmemLocation_st* _pvt_ptr

cdef class anon_struct22:
    """
    Attributes
    ----------

    compressionType : bytes



    gpuDirectRDMACapable : bytes



    usage : unsigned short



    reserved : bytes



    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cydriver.CUmemAllocationProp_st* _pvt_ptr

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
        CU_MEM_HANDLE_TYPE_WIN32 is specified. This object attributes
        structure includes security attributes that define the scope of
        which exported allocations may be transferred to other processes.
        In all other cases, this field is required to be zero.


    allocFlags : anon_struct22



    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cydriver.CUmemAllocationProp_st _pvt_val
    cdef cydriver.CUmemAllocationProp_st* _pvt_ptr

    cdef CUmemLocation _location


    cdef _HelperInputVoidPtr _cywin32HandleMetaData


    cdef anon_struct22 _allocFlags


cdef class CUmulticastObjectProp_st:
    """
    Specifies the properties for a multicast object.

    Attributes
    ----------

    numDevices : unsigned int
        The number of devices in the multicast team that will bind memory
        to this object


    size : size_t
        The maximum amount of memory that can be bound to this multicast
        object per device


    handleTypes : unsigned long long
        Bitmask of exportable handle types (see CUmemAllocationHandleType)
        for this object


    flags : unsigned long long
        Flags for future use, must be zero now


    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cydriver.CUmulticastObjectProp_st _pvt_val
    cdef cydriver.CUmulticastObjectProp_st* _pvt_ptr

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
    cdef cydriver.CUmemAccessDesc_st _pvt_val
    cdef cydriver.CUmemAccessDesc_st* _pvt_ptr

    cdef CUmemLocation _location


cdef class CUgraphExecUpdateResultInfo_st:
    """
    Result information returned by cuGraphExecUpdate

    Attributes
    ----------

    result : CUgraphExecUpdateResult
        Gives more specific detail when a cuda graph update fails.


    errorNode : CUgraphNode
        The "to node" of the error edge when the topologies do not match.
        The error node when the error is associated with a specific node.
        NULL when the error is generic.


    errorFromNode : CUgraphNode
        The from node of error edge when the topologies do not match.
        Otherwise NULL.


    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cydriver.CUgraphExecUpdateResultInfo_st _pvt_val
    cdef cydriver.CUgraphExecUpdateResultInfo_st* _pvt_ptr

    cdef CUgraphNode _errorNode


    cdef CUgraphNode _errorFromNode


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
        defines the scope of which exported allocations may be transferred
        to other processes. In all other cases, this field is required to
        be zero.


    maxSize : size_t
        Maximum pool size. When set to 0, defaults to a system dependent
        value.


    usage : unsigned short
        Bitmask indicating intended usage for the pool.


    reserved : bytes
        reserved for future use, must be 0


    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cydriver.CUmemPoolProps_st _pvt_val
    cdef cydriver.CUmemPoolProps_st* _pvt_ptr

    cdef CUmemLocation _location


    cdef _HelperInputVoidPtr _cywin32SecurityAttributes


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
    cdef cydriver.CUmemPoolPtrExportData_st _pvt_val
    cdef cydriver.CUmemPoolPtrExportData_st* _pvt_ptr

cdef class CUmemcpyAttributes_st:
    """
    Attributes specific to copies within a batch. For more details on
    usage see cuMemcpyBatchAsync.

    Attributes
    ----------

    srcAccessOrder : CUmemcpySrcAccessOrder
        Source access ordering to be observed for copies with this
        attribute.


    srcLocHint : CUmemLocation
        Hint location for the source operand. Ignored when the pointers are
        not managed memory or memory allocated outside CUDA.


    dstLocHint : CUmemLocation
        Hint location for the destination operand. Ignored when the
        pointers are not managed memory or memory allocated outside CUDA.


    flags : unsigned int
        Additional flags for copies with this attribute. See CUmemcpyFlags


    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cydriver.CUmemcpyAttributes_st _pvt_val
    cdef cydriver.CUmemcpyAttributes_st* _pvt_ptr

    cdef CUmemLocation _srcLocHint


    cdef CUmemLocation _dstLocHint


cdef class CUoffset3D_st:
    """
    Struct representing a 3D offset

    Attributes
    ----------

    x : size_t



    y : size_t



    z : size_t



    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cydriver.CUoffset3D_st _pvt_val
    cdef cydriver.CUoffset3D_st* _pvt_ptr

cdef class CUextent3D_st:
    """
    Struct representing width/height/depth of a CUarray in elements

    Attributes
    ----------

    width : size_t



    height : size_t



    depth : size_t



    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cydriver.CUextent3D_st _pvt_val
    cdef cydriver.CUextent3D_st* _pvt_ptr

cdef class anon_struct23:
    """
    Attributes
    ----------

    ptr : CUdeviceptr



    rowLength : size_t



    layerHeight : size_t



    locHint : CUmemLocation



    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cydriver.CUmemcpy3DOperand_st* _pvt_ptr

    cdef CUdeviceptr _ptr


    cdef CUmemLocation _locHint


cdef class anon_struct24:
    """
    Attributes
    ----------

    array : CUarray



    offset : CUoffset3D



    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cydriver.CUmemcpy3DOperand_st* _pvt_ptr

    cdef CUarray _array


    cdef CUoffset3D _offset


cdef class anon_union13:
    """
    Attributes
    ----------

    ptr : anon_struct23



    array : anon_struct24



    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cydriver.CUmemcpy3DOperand_st* _pvt_ptr

    cdef anon_struct23 _ptr


    cdef anon_struct24 _array


cdef class CUmemcpy3DOperand_st:
    """
    Struct representing an operand for copy with cuMemcpy3DBatchAsync

    Attributes
    ----------

    type : CUmemcpy3DOperandType



    op : anon_union13



    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cydriver.CUmemcpy3DOperand_st* _val_ptr
    cdef cydriver.CUmemcpy3DOperand_st* _pvt_ptr

    cdef anon_union13 _op


cdef class CUDA_MEMCPY3D_BATCH_OP_st:
    """
    Attributes
    ----------

    src : CUmemcpy3DOperand
        Source memcpy operand.


    dst : CUmemcpy3DOperand
        Destination memcpy operand.


    extent : CUextent3D
        Extents of the memcpy between src and dst. The width, height and
        depth components must not be 0.


    srcAccessOrder : CUmemcpySrcAccessOrder
        Source access ordering to be observed for copy from src to dst.


    flags : unsigned int
        Additional flags for copies with this attribute. See CUmemcpyFlags


    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cydriver.CUDA_MEMCPY3D_BATCH_OP_st _pvt_val
    cdef cydriver.CUDA_MEMCPY3D_BATCH_OP_st* _pvt_ptr

    cdef CUmemcpy3DOperand _src


    cdef CUmemcpy3DOperand _dst


    cdef CUextent3D _extent


cdef class CUDA_MEM_ALLOC_NODE_PARAMS_v1_st:
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
    cdef cydriver.CUDA_MEM_ALLOC_NODE_PARAMS_v1_st _pvt_val
    cdef cydriver.CUDA_MEM_ALLOC_NODE_PARAMS_v1_st* _pvt_ptr

    cdef CUmemPoolProps _poolProps


    cdef size_t _accessDescs_length
    cdef cydriver.CUmemAccessDesc* _accessDescs


    cdef CUdeviceptr _dptr


cdef class CUDA_MEM_ALLOC_NODE_PARAMS_v2_st:
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
    cdef cydriver.CUDA_MEM_ALLOC_NODE_PARAMS_v2_st _pvt_val
    cdef cydriver.CUDA_MEM_ALLOC_NODE_PARAMS_v2_st* _pvt_ptr

    cdef CUmemPoolProps _poolProps


    cdef size_t _accessDescs_length
    cdef cydriver.CUmemAccessDesc* _accessDescs


    cdef CUdeviceptr _dptr


cdef class CUDA_MEM_FREE_NODE_PARAMS_st:
    """
    Memory free node parameters

    Attributes
    ----------

    dptr : CUdeviceptr
        in: the pointer to free


    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cydriver.CUDA_MEM_FREE_NODE_PARAMS_st _pvt_val
    cdef cydriver.CUDA_MEM_FREE_NODE_PARAMS_st* _pvt_ptr

    cdef CUdeviceptr _dptr


cdef class CUDA_CHILD_GRAPH_NODE_PARAMS_st:
    """
    Child graph node parameters

    Attributes
    ----------

    graph : CUgraph
        The child graph to clone into the node for node creation, or a
        handle to the graph owned by the node for node query. The graph
        must not contain conditional nodes. Graphs containing memory
        allocation or memory free nodes must set the ownership to be moved
        to the parent.


    ownership : CUgraphChildGraphNodeOwnership
        The ownership relationship of the child graph node.


    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cydriver.CUDA_CHILD_GRAPH_NODE_PARAMS_st _pvt_val
    cdef cydriver.CUDA_CHILD_GRAPH_NODE_PARAMS_st* _pvt_ptr

    cdef CUgraph _graph


cdef class CUDA_EVENT_RECORD_NODE_PARAMS_st:
    """
    Event record node parameters

    Attributes
    ----------

    event : CUevent
        The event to record when the node executes


    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cydriver.CUDA_EVENT_RECORD_NODE_PARAMS_st _pvt_val
    cdef cydriver.CUDA_EVENT_RECORD_NODE_PARAMS_st* _pvt_ptr

    cdef CUevent _event


cdef class CUDA_EVENT_WAIT_NODE_PARAMS_st:
    """
    Event wait node parameters

    Attributes
    ----------

    event : CUevent
        The event to wait on from the node


    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cydriver.CUDA_EVENT_WAIT_NODE_PARAMS_st _pvt_val
    cdef cydriver.CUDA_EVENT_WAIT_NODE_PARAMS_st* _pvt_ptr

    cdef CUevent _event


cdef class CUgraphNodeParams_st:
    """
    Graph node parameters. See cuGraphAddNode.

    Attributes
    ----------

    type : CUgraphNodeType
        Type of the node


    reserved0 : list[int]
        Reserved. Must be zero.


    reserved1 : list[long long]
        Padding. Unused bytes must be zero.


    kernel : CUDA_KERNEL_NODE_PARAMS_v3
        Kernel node parameters.


    memcpy : CUDA_MEMCPY_NODE_PARAMS
        Memcpy node parameters.


    memset : CUDA_MEMSET_NODE_PARAMS_v2
        Memset node parameters.


    host : CUDA_HOST_NODE_PARAMS_v2
        Host node parameters.


    graph : CUDA_CHILD_GRAPH_NODE_PARAMS
        Child graph node parameters.


    eventWait : CUDA_EVENT_WAIT_NODE_PARAMS
        Event wait node parameters.


    eventRecord : CUDA_EVENT_RECORD_NODE_PARAMS
        Event record node parameters.


    extSemSignal : CUDA_EXT_SEM_SIGNAL_NODE_PARAMS_v2
        External semaphore signal node parameters.


    extSemWait : CUDA_EXT_SEM_WAIT_NODE_PARAMS_v2
        External semaphore wait node parameters.


    alloc : CUDA_MEM_ALLOC_NODE_PARAMS_v2
        Memory allocation node parameters.


    free : CUDA_MEM_FREE_NODE_PARAMS
        Memory free node parameters.


    memOp : CUDA_BATCH_MEM_OP_NODE_PARAMS_v2
        MemOp node parameters.


    conditional : CUDA_CONDITIONAL_NODE_PARAMS
        Conditional node parameters.


    reserved2 : long long
        Reserved bytes. Must be zero.


    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cydriver.CUgraphNodeParams_st* _val_ptr
    cdef cydriver.CUgraphNodeParams_st* _pvt_ptr

    cdef CUDA_KERNEL_NODE_PARAMS_v3 _kernel


    cdef CUDA_MEMCPY_NODE_PARAMS _memcpy


    cdef CUDA_MEMSET_NODE_PARAMS_v2 _memset


    cdef CUDA_HOST_NODE_PARAMS_v2 _host


    cdef CUDA_CHILD_GRAPH_NODE_PARAMS _graph


    cdef CUDA_EVENT_WAIT_NODE_PARAMS _eventWait


    cdef CUDA_EVENT_RECORD_NODE_PARAMS _eventRecord


    cdef CUDA_EXT_SEM_SIGNAL_NODE_PARAMS_v2 _extSemSignal


    cdef CUDA_EXT_SEM_WAIT_NODE_PARAMS_v2 _extSemWait


    cdef CUDA_MEM_ALLOC_NODE_PARAMS_v2 _alloc


    cdef CUDA_MEM_FREE_NODE_PARAMS _free


    cdef CUDA_BATCH_MEM_OP_NODE_PARAMS_v2 _memOp


    cdef CUDA_CONDITIONAL_NODE_PARAMS _conditional


cdef class CUcheckpointLockArgs_st:
    """
    CUDA checkpoint optional lock arguments

    Attributes
    ----------

    timeoutMs : unsigned int
        Timeout in milliseconds to attempt to lock the process, 0 indicates
        no timeout


    reserved0 : unsigned int
        Reserved for future use, must be zero


    reserved1 : list[cuuint64_t]
        Reserved for future use, must be zeroed


    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cydriver.CUcheckpointLockArgs_st _pvt_val
    cdef cydriver.CUcheckpointLockArgs_st* _pvt_ptr

cdef class CUcheckpointCheckpointArgs_st:
    """
    CUDA checkpoint optional checkpoint arguments

    Attributes
    ----------

    reserved : list[cuuint64_t]
        Reserved for future use, must be zeroed


    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cydriver.CUcheckpointCheckpointArgs_st _pvt_val
    cdef cydriver.CUcheckpointCheckpointArgs_st* _pvt_ptr

cdef class CUcheckpointGpuPair_st:
    """
    CUDA checkpoint GPU UUID pairs for device remapping during restore

    Attributes
    ----------

    oldUuid : CUuuid
        UUID of the GPU that was checkpointed


    newUuid : CUuuid
        UUID of the GPU to restore onto


    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cydriver.CUcheckpointGpuPair_st _pvt_val
    cdef cydriver.CUcheckpointGpuPair_st* _pvt_ptr

    cdef CUuuid _oldUuid


    cdef CUuuid _newUuid


cdef class CUcheckpointRestoreArgs_st:
    """
    CUDA checkpoint optional restore arguments

    Attributes
    ----------

    gpuPairs : CUcheckpointGpuPair
        Pointer to array of gpu pairs that indicate how to remap GPUs
        during restore


    gpuPairsCount : unsigned int
        Number of gpu pairs to remap


    reserved : bytes
        Reserved for future use, must be zeroed


    reserved1 : cuuint64_t
        Reserved for future use, must be zeroed


    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cydriver.CUcheckpointRestoreArgs_st _pvt_val
    cdef cydriver.CUcheckpointRestoreArgs_st* _pvt_ptr

    cdef size_t _gpuPairs_length
    cdef cydriver.CUcheckpointGpuPair* _gpuPairs


    cdef cuuint64_t _reserved1


cdef class CUcheckpointUnlockArgs_st:
    """
    CUDA checkpoint optional unlock arguments

    Attributes
    ----------

    reserved : list[cuuint64_t]
        Reserved for future use, must be zeroed


    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cydriver.CUcheckpointUnlockArgs_st _pvt_val
    cdef cydriver.CUcheckpointUnlockArgs_st* _pvt_ptr

cdef class CUmemDecompressParams_st:
    """
    Structure describing the parameters that compose a single
    decompression operation.

    Attributes
    ----------

    srcNumBytes : size_t
        The number of bytes to be read and decompressed from
        CUmemDecompressParams_st.src.


    dstNumBytes : size_t
        The number of bytes that the decompression operation will be
        expected to write to CUmemDecompressParams_st.dst. This value is
        optional; if present, it may be used by the CUDA driver as a
        heuristic for scheduling the individual decompression operations.


    dstActBytes : cuuint32_t
        After the decompression operation has completed, the actual number
        of bytes written to CUmemDecompressParams.dst will be recorded as a
        32-bit unsigned integer in the memory at this address.


    src : Any
        Pointer to a buffer of at least
        CUmemDecompressParams_st.srcNumBytes compressed bytes.


    dst : Any
        Pointer to a buffer where the decompressed data will be written.
        The number of bytes written to this location will be recorded in
        the memory pointed to by CUmemDecompressParams_st.dstActBytes


    algo : CUmemDecompressAlgorithm
        The decompression algorithm to use.


    padding : bytes



    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cydriver.CUmemDecompressParams_st _pvt_val
    cdef cydriver.CUmemDecompressParams_st* _pvt_ptr

    cdef _HelperInputVoidPtr _cysrc


    cdef _HelperInputVoidPtr _cydst


cdef class CUdevSmResource_st:
    """
    Attributes
    ----------

    smCount : unsigned int
        The amount of streaming multiprocessors available in this resource.


    minSmPartitionSize : unsigned int
        The minimum number of streaming multiprocessors required to
        partition this resource.


    smCoscheduledAlignment : unsigned int
        The number of streaming multiprocessors in this resource that are
        guaranteed to be co-scheduled on the same GPU processing cluster.
        smCount will be a multiple of this value, unless the backfill flag
        is set.


    flags : unsigned int
        The flags set on this SM resource. For possible values see
        CUdevSmResourceGroup_flags.


    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cydriver.CUdevSmResource_st _pvt_val
    cdef cydriver.CUdevSmResource_st* _pvt_ptr

cdef class CUdevWorkqueueConfigResource_st:
    """
    Attributes
    ----------

    device : CUdevice
        The device on which the workqueue resources are available


    wqConcurrencyLimit : unsigned int
        The expected maximum number of concurrent stream-ordered workloads


    sharingScope : CUdevWorkqueueConfigScope
        The sharing scope for the workqueue resources


    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cydriver.CUdevWorkqueueConfigResource_st _pvt_val
    cdef cydriver.CUdevWorkqueueConfigResource_st* _pvt_ptr

    cdef CUdevice _device


cdef class CUdevWorkqueueResource_st:
    """
    Attributes
    ----------

    reserved : bytes
        Reserved for future use


    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cydriver.CUdevWorkqueueResource_st _pvt_val
    cdef cydriver.CUdevWorkqueueResource_st* _pvt_ptr

cdef class CU_DEV_SM_RESOURCE_GROUP_PARAMS_st:
    """
    Attributes
    ----------

    smCount : unsigned int
        The amount of SMs available in this resource.


    coscheduledSmCount : unsigned int
        The amount of co-scheduled SMs grouped together for locality
        purposes.


    preferredCoscheduledSmCount : unsigned int
        When possible, combine co-scheduled groups together into larger
        groups of this size.


    flags : unsigned int
        The flags set on this SM resource group. For possible values see
        CUdevSmResourceGroup_flags.


    reserved : list[unsigned int]



    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cydriver.CU_DEV_SM_RESOURCE_GROUP_PARAMS_st _pvt_val
    cdef cydriver.CU_DEV_SM_RESOURCE_GROUP_PARAMS_st* _pvt_ptr

cdef class CUdevResource_st:
    """
    Attributes
    ----------

    type : CUdevResourceType
        Type of resource, dictates which union field was last set


    _internal_padding : bytes



    sm : CUdevSmResource
        Resource corresponding to CU_DEV_RESOURCE_TYPE_SM `typename`.


    wqConfig : CUdevWorkqueueConfigResource
        Resource corresponding to CU_DEV_RESOURCE_TYPE_WORKQUEUE_CONFIG
        `typename`.


    wq : CUdevWorkqueueResource
        Resource corresponding to CU_DEV_RESOURCE_TYPE_WORKQUEUE
        `typename`.


    _oversize : bytes



    nextResource : CUdevResource_st



    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cydriver.CUdevResource_st* _val_ptr
    cdef cydriver.CUdevResource_st* _pvt_ptr

    cdef CUdevSmResource _sm


    cdef CUdevWorkqueueConfigResource _wqConfig


    cdef CUdevWorkqueueResource _wq


    cdef size_t _nextResource_length
    cdef cydriver.CUdevResource_st* _nextResource


cdef class anon_union16:
    """
    Attributes
    ----------

    pArray : list[CUarray]



    pPitch : list[Any]



    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cydriver.CUeglFrame_st* _pvt_ptr

cdef class CUeglFrame_st:
    """
    CUDA EGLFrame structure Descriptor - structure defining one frame
    of EGL.  Each frame may contain one or more planes depending on
    whether the surface * is Multiplanar or not.

    Attributes
    ----------

    frame : anon_union16



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
    cdef cydriver.CUeglFrame_st* _val_ptr
    cdef cydriver.CUeglFrame_st* _pvt_ptr

    cdef anon_union16 _frame


cdef class CUdeviceptr:
    """

    CUDA device pointer CUdeviceptr is defined as an unsigned integer type whose size matches the size of a pointer on the target platform.

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef cydriver.CUdeviceptr  _pvt_val
    cdef cydriver.CUdeviceptr* _pvt_ptr

cdef class CUdevice:
    """

    CUDA device

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef cydriver.CUdevice  _pvt_val
    cdef cydriver.CUdevice* _pvt_ptr

cdef class CUtexObject:
    """

    An opaque value that represents a CUDA texture object

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef cydriver.CUtexObject  _pvt_val
    cdef cydriver.CUtexObject* _pvt_ptr

cdef class CUsurfObject:
    """

    An opaque value that represents a CUDA surface object

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef cydriver.CUsurfObject  _pvt_val
    cdef cydriver.CUsurfObject* _pvt_ptr

cdef class CUgraphConditionalHandle:
    """

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef cydriver.CUgraphConditionalHandle  _pvt_val
    cdef cydriver.CUgraphConditionalHandle* _pvt_ptr

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

cdef class CUmemFabricHandle_v1(CUmemFabricHandle_st):
    """
    Fabric handle - An opaque handle representing a memory allocation
    that can be exported to processes in same or different nodes. For
    IPC between processes on different nodes they must be connected via
    the NVSwitch fabric.

    Attributes
    ----------

    data : bytes



    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    pass

cdef class CUmemFabricHandle(CUmemFabricHandle_v1):
    """
    Fabric handle - An opaque handle representing a memory allocation
    that can be exported to processes in same or different nodes. For
    IPC between processes on different nodes they must be connected via
    the NVSwitch fabric.

    Attributes
    ----------

    data : bytes



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

cdef class CUipcEventHandle(CUipcEventHandle_v1):
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

cdef class CUipcMemHandle(CUipcMemHandle_v1):
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
        Operation. This is the first field of all the union elemets and
        acts as a TAG to determine which union member is valid.


    waitValue : CUstreamMemOpWaitValueParams_st
        Params for CU_STREAM_MEM_OP_WAIT_VALUE_32 and
        CU_STREAM_MEM_OP_WAIT_VALUE_64 operations.


    writeValue : CUstreamMemOpWriteValueParams_st
        Params for CU_STREAM_MEM_OP_WRITE_VALUE_32 and
        CU_STREAM_MEM_OP_WRITE_VALUE_64 operations.


    flushRemoteWrites : CUstreamMemOpFlushRemoteWritesParams_st
        Params for CU_STREAM_MEM_OP_FLUSH_REMOTE_WRITES operations.


    memoryBarrier : CUstreamMemOpMemoryBarrierParams_st
        Params for CU_STREAM_MEM_OP_BARRIER operations.


    atomicReduction : CUstreamMemOpAtomicReductionParams_st



    pad : list[cuuint64_t]



    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    pass

cdef class CUstreamBatchMemOpParams(CUstreamBatchMemOpParams_v1):
    """
    Per-operation parameters for cuStreamBatchMemOp

    Attributes
    ----------

    operation : CUstreamBatchMemOpType
        Operation. This is the first field of all the union elemets and
        acts as a TAG to determine which union member is valid.


    waitValue : CUstreamMemOpWaitValueParams_st
        Params for CU_STREAM_MEM_OP_WAIT_VALUE_32 and
        CU_STREAM_MEM_OP_WAIT_VALUE_64 operations.


    writeValue : CUstreamMemOpWriteValueParams_st
        Params for CU_STREAM_MEM_OP_WRITE_VALUE_32 and
        CU_STREAM_MEM_OP_WRITE_VALUE_64 operations.


    flushRemoteWrites : CUstreamMemOpFlushRemoteWritesParams_st
        Params for CU_STREAM_MEM_OP_FLUSH_REMOTE_WRITES operations.


    memoryBarrier : CUstreamMemOpMemoryBarrierParams_st
        Params for CU_STREAM_MEM_OP_BARRIER operations.


    atomicReduction : CUstreamMemOpAtomicReductionParams_st



    pad : list[cuuint64_t]



    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    pass

cdef class CUDA_BATCH_MEM_OP_NODE_PARAMS_v1(CUDA_BATCH_MEM_OP_NODE_PARAMS_v1_st):
    """
    Batch memory operation node parameters  Used in the legacy
    cuGraphAddBatchMemOpNode api. New code should use cuGraphAddNode()

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

cdef class CUDA_BATCH_MEM_OP_NODE_PARAMS(CUDA_BATCH_MEM_OP_NODE_PARAMS_v1):
    """
    Batch memory operation node parameters  Used in the legacy
    cuGraphAddBatchMemOpNode api. New code should use cuGraphAddNode()

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

cdef class CUDA_BATCH_MEM_OP_NODE_PARAMS_v2(CUDA_BATCH_MEM_OP_NODE_PARAMS_v2_st):
    """
    Batch memory operation node parameters

    Attributes
    ----------

    ctx : CUcontext
        Context to use for the operations.


    count : unsigned int
        Number of operations in paramArray.


    paramArray : CUstreamBatchMemOpParams
        Array of batch memory operations.


    flags : unsigned int
        Flags to control the node.


    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    pass

cdef class CUasyncNotificationInfo(CUasyncNotificationInfo_st):
    """
    Information passed to the user via the async notification callback

    Attributes
    ----------

    type : CUasyncNotificationType
        The type of notification being sent


    info : anon_union2
        Information about the notification. `typename` must be checked in
        order to interpret this field.


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


    maxThreadsDim : list[int]
        Maximum size of each dimension of a block


    maxGridSize : list[int]
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

cdef class CUdevprop(CUdevprop_v1):
    """
    Legacy device properties

    Attributes
    ----------

    maxThreadsPerBlock : int
        Maximum number of threads per block


    maxThreadsDim : list[int]
        Maximum size of each dimension of a block


    maxGridSize : list[int]
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

cdef class CUaccessPolicyWindow(CUaccessPolicyWindow_v1):
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

cdef class CUDA_KERNEL_NODE_PARAMS_v2(CUDA_KERNEL_NODE_PARAMS_v2_st):
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


    kern : CUkernel
        Kernel to launch, will only be referenced if func is NULL


    ctx : CUcontext
        Context for the kernel task to run in. The value NULL will indicate
        the current context should be used by the api. This field is
        ignored if func is set.


    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    pass

cdef class CUDA_KERNEL_NODE_PARAMS(CUDA_KERNEL_NODE_PARAMS_v2):
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


    kern : CUkernel
        Kernel to launch, will only be referenced if func is NULL


    ctx : CUcontext
        Context for the kernel task to run in. The value NULL will indicate
        the current context should be used by the api. This field is
        ignored if func is set.


    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    pass

cdef class CUDA_KERNEL_NODE_PARAMS_v3(CUDA_KERNEL_NODE_PARAMS_v3_st):
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


    kern : CUkernel
        Kernel to launch, will only be referenced if func is NULL


    ctx : CUcontext
        Context for the kernel task to run in. The value NULL will indicate
        the current context should be used by the api. This field is
        ignored if func is set.


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

cdef class CUDA_MEMSET_NODE_PARAMS(CUDA_MEMSET_NODE_PARAMS_v1):
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

cdef class CUDA_MEMSET_NODE_PARAMS_v2(CUDA_MEMSET_NODE_PARAMS_v2_st):
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


    ctx : CUcontext
        Context on which to run the node


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

cdef class CUDA_HOST_NODE_PARAMS(CUDA_HOST_NODE_PARAMS_v1):
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

cdef class CUDA_HOST_NODE_PARAMS_v2(CUDA_HOST_NODE_PARAMS_v2_st):
    """
    Host node parameters

    Attributes
    ----------

    fn : CUhostFn
        The function to call when the node executes


    userData : Any
        Argument to pass to the function


    syncMode : unsigned int
        The sync mode to use for the host task


    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    pass

cdef class CUgraphEdgeData(CUgraphEdgeData_st):
    """
    Optional annotation for edges in a CUDA graph. Note, all edges
    implicitly have annotations and default to a zero-initialized value
    if not specified. A zero-initialized struct indicates a standard
    full serialization of two nodes with memory visibility.

    Attributes
    ----------

    from_port : bytes
        This indicates when the dependency is triggered from the upstream
        node on the edge. The meaning is specfic to the node type. A value
        of 0 in all cases means full completion of the upstream node, with
        memory visibility to the downstream node or portion thereof
        (indicated by `to_port`).   Only kernel nodes define non-zero
        ports. A kernel node can use the following output port types:
        CU_GRAPH_KERNEL_NODE_PORT_DEFAULT,
        CU_GRAPH_KERNEL_NODE_PORT_PROGRAMMATIC, or
        CU_GRAPH_KERNEL_NODE_PORT_LAUNCH_ORDER.


    to_port : bytes
        This indicates what portion of the downstream node is dependent on
        the upstream node or portion thereof (indicated by `from_port`).
        The meaning is specific to the node type. A value of 0 in all cases
        means the entirety of the downstream node is dependent on the
        upstream work.   Currently no node types define non-zero ports.
        Accordingly, this field must be set to zero.


    type : bytes
        This should be populated with a value from CUgraphDependencyType.
        (It is typed as char due to compiler-specific layout of bitfields.)
        See CUgraphDependencyType.


    reserved : bytes
        These bytes are unused and must be zeroed. This ensures
        compatibility if additional fields are added in the future.


    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    pass

cdef class CUDA_GRAPH_INSTANTIATE_PARAMS(CUDA_GRAPH_INSTANTIATE_PARAMS_st):
    """
    Graph instantiation parameters

    Attributes
    ----------

    flags : cuuint64_t
        Instantiation flags


    hUploadStream : CUstream
        Upload stream


    hErrNode_out : CUgraphNode
        The node which caused instantiation to fail, if any


    result_out : CUgraphInstantiateResult
        Whether instantiation was successful. If it failed, the reason why


    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    pass

cdef class CUlaunchMemSyncDomainMap(CUlaunchMemSyncDomainMap_st):
    """
    Memory Synchronization Domain map  See ::cudaLaunchMemSyncDomain.
    By default, kernels are launched in domain 0. Kernel launched with
    CU_LAUNCH_MEM_SYNC_DOMAIN_REMOTE will have a different domain ID.
    User may also alter the domain ID with CUlaunchMemSyncDomainMap for
    a specific stream / graph node / kernel launch. See
    CU_LAUNCH_ATTRIBUTE_MEM_SYNC_DOMAIN_MAP.  Domain ID range is
    available through CU_DEVICE_ATTRIBUTE_MEM_SYNC_DOMAIN_COUNT.

    Attributes
    ----------

    default_ : bytes
        The default domain ID to use for designated kernels


    remote : bytes
        The remote domain ID to use for designated kernels


    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    pass

cdef class CUlaunchAttributeValue(CUlaunchAttributeValue_union):
    """
    Launch attributes union; used as value field of CUlaunchAttribute

    Attributes
    ----------

    pad : bytes



    accessPolicyWindow : CUaccessPolicyWindow
        Value of launch attribute CU_LAUNCH_ATTRIBUTE_ACCESS_POLICY_WINDOW.


    cooperative : int
        Value of launch attribute CU_LAUNCH_ATTRIBUTE_COOPERATIVE. Nonzero
        indicates a cooperative kernel (see cuLaunchCooperativeKernel).


    syncPolicy : CUsynchronizationPolicy
        Value of launch attribute
        CU_LAUNCH_ATTRIBUTE_SYNCHRONIZATION_POLICY. CUsynchronizationPolicy
        for work queued up in this stream


    clusterDim : anon_struct1
        Value of launch attribute CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION
        that represents the desired cluster dimensions for the kernel.
        Opaque type with the following fields: - `x` - The X dimension of
        the cluster, in blocks. Must be a divisor of the grid X dimension.
        - `y` - The Y dimension of the cluster, in blocks. Must be a
        divisor of the grid Y dimension.    - `z` - The Z dimension of the
        cluster, in blocks. Must be a divisor of the grid Z dimension.


    clusterSchedulingPolicyPreference : CUclusterSchedulingPolicy
        Value of launch attribute
        CU_LAUNCH_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE. Cluster
        scheduling policy preference for the kernel.


    programmaticStreamSerializationAllowed : int
        Value of launch attribute
        CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION.


    programmaticEvent : anon_struct2
        Value of launch attribute CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_EVENT
        with the following fields: - `CUevent` event - Event to fire when
        all blocks trigger it.    - `Event` record flags, see
        cuEventRecordWithFlags. Does not accept :CU_EVENT_RECORD_EXTERNAL.
        - `triggerAtBlockStart` - If this is set to non-0, each block
        launch will automatically trigger the event.


    launchCompletionEvent : anon_struct3
        Value of launch attribute
        CU_LAUNCH_ATTRIBUTE_LAUNCH_COMPLETION_EVENT with the following
        fields: - `CUevent` event - Event to fire when the last block
        launches    - `int` flags; - Event record flags, see
        cuEventRecordWithFlags. Does not accept CU_EVENT_RECORD_EXTERNAL.


    priority : int
        Value of launch attribute CU_LAUNCH_ATTRIBUTE_PRIORITY. Execution
        priority of the kernel.


    memSyncDomainMap : CUlaunchMemSyncDomainMap
        Value of launch attribute CU_LAUNCH_ATTRIBUTE_MEM_SYNC_DOMAIN_MAP.
        See CUlaunchMemSyncDomainMap.


    memSyncDomain : CUlaunchMemSyncDomain
        Value of launch attribute CU_LAUNCH_ATTRIBUTE_MEM_SYNC_DOMAIN.
        See::CUlaunchMemSyncDomain


    preferredClusterDim : anon_struct4
        Value of launch attribute
        CU_LAUNCH_ATTRIBUTE_PREFERRED_CLUSTER_DIMENSION that represents the
        desired preferred cluster dimensions for the kernel. Opaque type
        with the following fields: - `x` - The X dimension of the preferred
        cluster, in blocks. Must be a divisor of the grid X dimension, and
        must be a multiple of the `x` field of
        CUlaunchAttributeValue::clusterDim.    - `y` - The Y dimension of
        the preferred cluster, in blocks. Must be a divisor of the grid Y
        dimension, and must be a multiple of the `y` field of
        CUlaunchAttributeValue::clusterDim.    - `z` - The Z dimension of
        the preferred cluster, in blocks. Must be equal to the `z` field of
        CUlaunchAttributeValue::clusterDim.


    deviceUpdatableKernelNode : anon_struct5
        Value of launch attribute
        CU_LAUNCH_ATTRIBUTE_DEVICE_UPDATABLE_KERNEL_NODE. with the
        following fields: - `int` deviceUpdatable - Whether or not the
        resulting kernel node should be device-updatable.    -
        `CUgraphDeviceNode` devNode - Returns a handle to pass to the
        various device-side update functions.


    sharedMemCarveout : unsigned int
        Value of launch attribute
        CU_LAUNCH_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT.


    nvlinkUtilCentricScheduling : unsigned int



    portableClusterSizeMode : CUlaunchAttributePortableClusterMode
        Value of launch attribute
        CU_LAUNCH_ATTRIBUTE_PORTABLE_CLUSTER_SIZE_MODE.


    sharedMemoryMode : CUsharedMemoryMode
        Value of launch attribute CU_LAUNCH_ATTRIBUTE_SHARED_MEMORY_MODE.
        See CUsharedMemoryMode for acceptable values.


    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    pass

cdef class CUlaunchAttribute(CUlaunchAttribute_st):
    """
    Launch attribute

    Attributes
    ----------

    id : CUlaunchAttributeID
        Attribute to set


    value : CUlaunchAttributeValue
        Value of the attribute


    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    pass

cdef class CUlaunchConfig(CUlaunchConfig_st):
    """
    CUDA extensible launch configuration

    Attributes
    ----------

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


    attrs : CUlaunchAttribute
        List of attributes; nullable if CUlaunchConfig::numAttrs == 0


    numAttrs : unsigned int
        Number of attributes populated in CUlaunchConfig::attrs


    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    pass

cdef class CUkernelNodeAttrValue_v1(CUlaunchAttributeValue):
    """
    Launch attributes union; used as value field of CUlaunchAttribute

    Attributes
    ----------

    pad : bytes



    accessPolicyWindow : CUaccessPolicyWindow
        Value of launch attribute CU_LAUNCH_ATTRIBUTE_ACCESS_POLICY_WINDOW.


    cooperative : int
        Value of launch attribute CU_LAUNCH_ATTRIBUTE_COOPERATIVE. Nonzero
        indicates a cooperative kernel (see cuLaunchCooperativeKernel).


    syncPolicy : CUsynchronizationPolicy
        Value of launch attribute
        CU_LAUNCH_ATTRIBUTE_SYNCHRONIZATION_POLICY. CUsynchronizationPolicy
        for work queued up in this stream


    clusterDim : anon_struct1
        Value of launch attribute CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION
        that represents the desired cluster dimensions for the kernel.
        Opaque type with the following fields: - `x` - The X dimension of
        the cluster, in blocks. Must be a divisor of the grid X dimension.
        - `y` - The Y dimension of the cluster, in blocks. Must be a
        divisor of the grid Y dimension.    - `z` - The Z dimension of the
        cluster, in blocks. Must be a divisor of the grid Z dimension.


    clusterSchedulingPolicyPreference : CUclusterSchedulingPolicy
        Value of launch attribute
        CU_LAUNCH_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE. Cluster
        scheduling policy preference for the kernel.


    programmaticStreamSerializationAllowed : int
        Value of launch attribute
        CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION.


    programmaticEvent : anon_struct2
        Value of launch attribute CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_EVENT
        with the following fields: - `CUevent` event - Event to fire when
        all blocks trigger it.    - `Event` record flags, see
        cuEventRecordWithFlags. Does not accept :CU_EVENT_RECORD_EXTERNAL.
        - `triggerAtBlockStart` - If this is set to non-0, each block
        launch will automatically trigger the event.


    launchCompletionEvent : anon_struct3
        Value of launch attribute
        CU_LAUNCH_ATTRIBUTE_LAUNCH_COMPLETION_EVENT with the following
        fields: - `CUevent` event - Event to fire when the last block
        launches    - `int` flags; - Event record flags, see
        cuEventRecordWithFlags. Does not accept CU_EVENT_RECORD_EXTERNAL.


    priority : int
        Value of launch attribute CU_LAUNCH_ATTRIBUTE_PRIORITY. Execution
        priority of the kernel.


    memSyncDomainMap : CUlaunchMemSyncDomainMap
        Value of launch attribute CU_LAUNCH_ATTRIBUTE_MEM_SYNC_DOMAIN_MAP.
        See CUlaunchMemSyncDomainMap.


    memSyncDomain : CUlaunchMemSyncDomain
        Value of launch attribute CU_LAUNCH_ATTRIBUTE_MEM_SYNC_DOMAIN.
        See::CUlaunchMemSyncDomain


    preferredClusterDim : anon_struct4
        Value of launch attribute
        CU_LAUNCH_ATTRIBUTE_PREFERRED_CLUSTER_DIMENSION that represents the
        desired preferred cluster dimensions for the kernel. Opaque type
        with the following fields: - `x` - The X dimension of the preferred
        cluster, in blocks. Must be a divisor of the grid X dimension, and
        must be a multiple of the `x` field of
        CUlaunchAttributeValue::clusterDim.    - `y` - The Y dimension of
        the preferred cluster, in blocks. Must be a divisor of the grid Y
        dimension, and must be a multiple of the `y` field of
        CUlaunchAttributeValue::clusterDim.    - `z` - The Z dimension of
        the preferred cluster, in blocks. Must be equal to the `z` field of
        CUlaunchAttributeValue::clusterDim.


    deviceUpdatableKernelNode : anon_struct5
        Value of launch attribute
        CU_LAUNCH_ATTRIBUTE_DEVICE_UPDATABLE_KERNEL_NODE. with the
        following fields: - `int` deviceUpdatable - Whether or not the
        resulting kernel node should be device-updatable.    -
        `CUgraphDeviceNode` devNode - Returns a handle to pass to the
        various device-side update functions.


    sharedMemCarveout : unsigned int
        Value of launch attribute
        CU_LAUNCH_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT.


    nvlinkUtilCentricScheduling : unsigned int



    portableClusterSizeMode : CUlaunchAttributePortableClusterMode
        Value of launch attribute
        CU_LAUNCH_ATTRIBUTE_PORTABLE_CLUSTER_SIZE_MODE.


    sharedMemoryMode : CUsharedMemoryMode
        Value of launch attribute CU_LAUNCH_ATTRIBUTE_SHARED_MEMORY_MODE.
        See CUsharedMemoryMode for acceptable values.


    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    pass

cdef class CUkernelNodeAttrValue(CUkernelNodeAttrValue_v1):
    """
    Launch attributes union; used as value field of CUlaunchAttribute

    Attributes
    ----------

    pad : bytes



    accessPolicyWindow : CUaccessPolicyWindow
        Value of launch attribute CU_LAUNCH_ATTRIBUTE_ACCESS_POLICY_WINDOW.


    cooperative : int
        Value of launch attribute CU_LAUNCH_ATTRIBUTE_COOPERATIVE. Nonzero
        indicates a cooperative kernel (see cuLaunchCooperativeKernel).


    syncPolicy : CUsynchronizationPolicy
        Value of launch attribute
        CU_LAUNCH_ATTRIBUTE_SYNCHRONIZATION_POLICY. CUsynchronizationPolicy
        for work queued up in this stream


    clusterDim : anon_struct1
        Value of launch attribute CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION
        that represents the desired cluster dimensions for the kernel.
        Opaque type with the following fields: - `x` - The X dimension of
        the cluster, in blocks. Must be a divisor of the grid X dimension.
        - `y` - The Y dimension of the cluster, in blocks. Must be a
        divisor of the grid Y dimension.    - `z` - The Z dimension of the
        cluster, in blocks. Must be a divisor of the grid Z dimension.


    clusterSchedulingPolicyPreference : CUclusterSchedulingPolicy
        Value of launch attribute
        CU_LAUNCH_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE. Cluster
        scheduling policy preference for the kernel.


    programmaticStreamSerializationAllowed : int
        Value of launch attribute
        CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION.


    programmaticEvent : anon_struct2
        Value of launch attribute CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_EVENT
        with the following fields: - `CUevent` event - Event to fire when
        all blocks trigger it.    - `Event` record flags, see
        cuEventRecordWithFlags. Does not accept :CU_EVENT_RECORD_EXTERNAL.
        - `triggerAtBlockStart` - If this is set to non-0, each block
        launch will automatically trigger the event.


    launchCompletionEvent : anon_struct3
        Value of launch attribute
        CU_LAUNCH_ATTRIBUTE_LAUNCH_COMPLETION_EVENT with the following
        fields: - `CUevent` event - Event to fire when the last block
        launches    - `int` flags; - Event record flags, see
        cuEventRecordWithFlags. Does not accept CU_EVENT_RECORD_EXTERNAL.


    priority : int
        Value of launch attribute CU_LAUNCH_ATTRIBUTE_PRIORITY. Execution
        priority of the kernel.


    memSyncDomainMap : CUlaunchMemSyncDomainMap
        Value of launch attribute CU_LAUNCH_ATTRIBUTE_MEM_SYNC_DOMAIN_MAP.
        See CUlaunchMemSyncDomainMap.


    memSyncDomain : CUlaunchMemSyncDomain
        Value of launch attribute CU_LAUNCH_ATTRIBUTE_MEM_SYNC_DOMAIN.
        See::CUlaunchMemSyncDomain


    preferredClusterDim : anon_struct4
        Value of launch attribute
        CU_LAUNCH_ATTRIBUTE_PREFERRED_CLUSTER_DIMENSION that represents the
        desired preferred cluster dimensions for the kernel. Opaque type
        with the following fields: - `x` - The X dimension of the preferred
        cluster, in blocks. Must be a divisor of the grid X dimension, and
        must be a multiple of the `x` field of
        CUlaunchAttributeValue::clusterDim.    - `y` - The Y dimension of
        the preferred cluster, in blocks. Must be a divisor of the grid Y
        dimension, and must be a multiple of the `y` field of
        CUlaunchAttributeValue::clusterDim.    - `z` - The Z dimension of
        the preferred cluster, in blocks. Must be equal to the `z` field of
        CUlaunchAttributeValue::clusterDim.


    deviceUpdatableKernelNode : anon_struct5
        Value of launch attribute
        CU_LAUNCH_ATTRIBUTE_DEVICE_UPDATABLE_KERNEL_NODE. with the
        following fields: - `int` deviceUpdatable - Whether or not the
        resulting kernel node should be device-updatable.    -
        `CUgraphDeviceNode` devNode - Returns a handle to pass to the
        various device-side update functions.


    sharedMemCarveout : unsigned int
        Value of launch attribute
        CU_LAUNCH_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT.


    nvlinkUtilCentricScheduling : unsigned int



    portableClusterSizeMode : CUlaunchAttributePortableClusterMode
        Value of launch attribute
        CU_LAUNCH_ATTRIBUTE_PORTABLE_CLUSTER_SIZE_MODE.


    sharedMemoryMode : CUsharedMemoryMode
        Value of launch attribute CU_LAUNCH_ATTRIBUTE_SHARED_MEMORY_MODE.
        See CUsharedMemoryMode for acceptable values.


    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    pass

cdef class CUstreamAttrValue_v1(CUlaunchAttributeValue):
    """
    Launch attributes union; used as value field of CUlaunchAttribute

    Attributes
    ----------

    pad : bytes



    accessPolicyWindow : CUaccessPolicyWindow
        Value of launch attribute CU_LAUNCH_ATTRIBUTE_ACCESS_POLICY_WINDOW.


    cooperative : int
        Value of launch attribute CU_LAUNCH_ATTRIBUTE_COOPERATIVE. Nonzero
        indicates a cooperative kernel (see cuLaunchCooperativeKernel).


    syncPolicy : CUsynchronizationPolicy
        Value of launch attribute
        CU_LAUNCH_ATTRIBUTE_SYNCHRONIZATION_POLICY. CUsynchronizationPolicy
        for work queued up in this stream


    clusterDim : anon_struct1
        Value of launch attribute CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION
        that represents the desired cluster dimensions for the kernel.
        Opaque type with the following fields: - `x` - The X dimension of
        the cluster, in blocks. Must be a divisor of the grid X dimension.
        - `y` - The Y dimension of the cluster, in blocks. Must be a
        divisor of the grid Y dimension.    - `z` - The Z dimension of the
        cluster, in blocks. Must be a divisor of the grid Z dimension.


    clusterSchedulingPolicyPreference : CUclusterSchedulingPolicy
        Value of launch attribute
        CU_LAUNCH_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE. Cluster
        scheduling policy preference for the kernel.


    programmaticStreamSerializationAllowed : int
        Value of launch attribute
        CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION.


    programmaticEvent : anon_struct2
        Value of launch attribute CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_EVENT
        with the following fields: - `CUevent` event - Event to fire when
        all blocks trigger it.    - `Event` record flags, see
        cuEventRecordWithFlags. Does not accept :CU_EVENT_RECORD_EXTERNAL.
        - `triggerAtBlockStart` - If this is set to non-0, each block
        launch will automatically trigger the event.


    launchCompletionEvent : anon_struct3
        Value of launch attribute
        CU_LAUNCH_ATTRIBUTE_LAUNCH_COMPLETION_EVENT with the following
        fields: - `CUevent` event - Event to fire when the last block
        launches    - `int` flags; - Event record flags, see
        cuEventRecordWithFlags. Does not accept CU_EVENT_RECORD_EXTERNAL.


    priority : int
        Value of launch attribute CU_LAUNCH_ATTRIBUTE_PRIORITY. Execution
        priority of the kernel.


    memSyncDomainMap : CUlaunchMemSyncDomainMap
        Value of launch attribute CU_LAUNCH_ATTRIBUTE_MEM_SYNC_DOMAIN_MAP.
        See CUlaunchMemSyncDomainMap.


    memSyncDomain : CUlaunchMemSyncDomain
        Value of launch attribute CU_LAUNCH_ATTRIBUTE_MEM_SYNC_DOMAIN.
        See::CUlaunchMemSyncDomain


    preferredClusterDim : anon_struct4
        Value of launch attribute
        CU_LAUNCH_ATTRIBUTE_PREFERRED_CLUSTER_DIMENSION that represents the
        desired preferred cluster dimensions for the kernel. Opaque type
        with the following fields: - `x` - The X dimension of the preferred
        cluster, in blocks. Must be a divisor of the grid X dimension, and
        must be a multiple of the `x` field of
        CUlaunchAttributeValue::clusterDim.    - `y` - The Y dimension of
        the preferred cluster, in blocks. Must be a divisor of the grid Y
        dimension, and must be a multiple of the `y` field of
        CUlaunchAttributeValue::clusterDim.    - `z` - The Z dimension of
        the preferred cluster, in blocks. Must be equal to the `z` field of
        CUlaunchAttributeValue::clusterDim.


    deviceUpdatableKernelNode : anon_struct5
        Value of launch attribute
        CU_LAUNCH_ATTRIBUTE_DEVICE_UPDATABLE_KERNEL_NODE. with the
        following fields: - `int` deviceUpdatable - Whether or not the
        resulting kernel node should be device-updatable.    -
        `CUgraphDeviceNode` devNode - Returns a handle to pass to the
        various device-side update functions.


    sharedMemCarveout : unsigned int
        Value of launch attribute
        CU_LAUNCH_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT.


    nvlinkUtilCentricScheduling : unsigned int



    portableClusterSizeMode : CUlaunchAttributePortableClusterMode
        Value of launch attribute
        CU_LAUNCH_ATTRIBUTE_PORTABLE_CLUSTER_SIZE_MODE.


    sharedMemoryMode : CUsharedMemoryMode
        Value of launch attribute CU_LAUNCH_ATTRIBUTE_SHARED_MEMORY_MODE.
        See CUsharedMemoryMode for acceptable values.


    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    pass

cdef class CUstreamAttrValue(CUstreamAttrValue_v1):
    """
    Launch attributes union; used as value field of CUlaunchAttribute

    Attributes
    ----------

    pad : bytes



    accessPolicyWindow : CUaccessPolicyWindow
        Value of launch attribute CU_LAUNCH_ATTRIBUTE_ACCESS_POLICY_WINDOW.


    cooperative : int
        Value of launch attribute CU_LAUNCH_ATTRIBUTE_COOPERATIVE. Nonzero
        indicates a cooperative kernel (see cuLaunchCooperativeKernel).


    syncPolicy : CUsynchronizationPolicy
        Value of launch attribute
        CU_LAUNCH_ATTRIBUTE_SYNCHRONIZATION_POLICY. CUsynchronizationPolicy
        for work queued up in this stream


    clusterDim : anon_struct1
        Value of launch attribute CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION
        that represents the desired cluster dimensions for the kernel.
        Opaque type with the following fields: - `x` - The X dimension of
        the cluster, in blocks. Must be a divisor of the grid X dimension.
        - `y` - The Y dimension of the cluster, in blocks. Must be a
        divisor of the grid Y dimension.    - `z` - The Z dimension of the
        cluster, in blocks. Must be a divisor of the grid Z dimension.


    clusterSchedulingPolicyPreference : CUclusterSchedulingPolicy
        Value of launch attribute
        CU_LAUNCH_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE. Cluster
        scheduling policy preference for the kernel.


    programmaticStreamSerializationAllowed : int
        Value of launch attribute
        CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION.


    programmaticEvent : anon_struct2
        Value of launch attribute CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_EVENT
        with the following fields: - `CUevent` event - Event to fire when
        all blocks trigger it.    - `Event` record flags, see
        cuEventRecordWithFlags. Does not accept :CU_EVENT_RECORD_EXTERNAL.
        - `triggerAtBlockStart` - If this is set to non-0, each block
        launch will automatically trigger the event.


    launchCompletionEvent : anon_struct3
        Value of launch attribute
        CU_LAUNCH_ATTRIBUTE_LAUNCH_COMPLETION_EVENT with the following
        fields: - `CUevent` event - Event to fire when the last block
        launches    - `int` flags; - Event record flags, see
        cuEventRecordWithFlags. Does not accept CU_EVENT_RECORD_EXTERNAL.


    priority : int
        Value of launch attribute CU_LAUNCH_ATTRIBUTE_PRIORITY. Execution
        priority of the kernel.


    memSyncDomainMap : CUlaunchMemSyncDomainMap
        Value of launch attribute CU_LAUNCH_ATTRIBUTE_MEM_SYNC_DOMAIN_MAP.
        See CUlaunchMemSyncDomainMap.


    memSyncDomain : CUlaunchMemSyncDomain
        Value of launch attribute CU_LAUNCH_ATTRIBUTE_MEM_SYNC_DOMAIN.
        See::CUlaunchMemSyncDomain


    preferredClusterDim : anon_struct4
        Value of launch attribute
        CU_LAUNCH_ATTRIBUTE_PREFERRED_CLUSTER_DIMENSION that represents the
        desired preferred cluster dimensions for the kernel. Opaque type
        with the following fields: - `x` - The X dimension of the preferred
        cluster, in blocks. Must be a divisor of the grid X dimension, and
        must be a multiple of the `x` field of
        CUlaunchAttributeValue::clusterDim.    - `y` - The Y dimension of
        the preferred cluster, in blocks. Must be a divisor of the grid Y
        dimension, and must be a multiple of the `y` field of
        CUlaunchAttributeValue::clusterDim.    - `z` - The Z dimension of
        the preferred cluster, in blocks. Must be equal to the `z` field of
        CUlaunchAttributeValue::clusterDim.


    deviceUpdatableKernelNode : anon_struct5
        Value of launch attribute
        CU_LAUNCH_ATTRIBUTE_DEVICE_UPDATABLE_KERNEL_NODE. with the
        following fields: - `int` deviceUpdatable - Whether or not the
        resulting kernel node should be device-updatable.    -
        `CUgraphDeviceNode` devNode - Returns a handle to pass to the
        various device-side update functions.


    sharedMemCarveout : unsigned int
        Value of launch attribute
        CU_LAUNCH_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT.


    nvlinkUtilCentricScheduling : unsigned int



    portableClusterSizeMode : CUlaunchAttributePortableClusterMode
        Value of launch attribute
        CU_LAUNCH_ATTRIBUTE_PORTABLE_CLUSTER_SIZE_MODE.


    sharedMemoryMode : CUsharedMemoryMode
        Value of launch attribute CU_LAUNCH_ATTRIBUTE_SHARED_MEMORY_MODE.
        See CUsharedMemoryMode for acceptable values.


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

cdef class CUexecAffinitySmCount(CUexecAffinitySmCount_v1):
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
        Type of execution affinity.


    param : anon_union3



    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    pass

cdef class CUexecAffinityParam(CUexecAffinityParam_v1):
    """
    Execution Affinity Parameters

    Attributes
    ----------

    type : CUexecAffinityType
        Type of execution affinity.


    param : anon_union3



    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    pass

cdef class CUctxCigParam(CUctxCigParam_st):
    """
    CIG Context Create Params

    Attributes
    ----------

    sharedDataType : CUcigDataType
        Type of shared data from graphics client (D3D12 or Vulkan).


    sharedData : Any
        Graphics client data handle (ID3D12CommandQueue or Nvidia specific
        data blob).


    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    pass

cdef class CUctxCreateParams(CUctxCreateParams_st):
    """
    Params for creating CUDA context. Both execAffinityParams and
    cigParams cannot be non-NULL at the same time. If both are NULL,
    the context will be created as a regular CUDA context.

    Attributes
    ----------

    execAffinityParams : CUexecAffinityParam
        Array of execution affinity parameters to limit context resources
        (e.g., SM count). Only supported Volta+ MPS. Mutually exclusive
        with cigParams.


    numExecAffinityParams : int
        Number of elements in execAffinityParams array. Must be 0 if
        execAffinityParams is NULL.


    cigParams : CUctxCigParam
        CIG (CUDA in Graphics) parameters for sharing data from
        D3D12/Vulkan graphics clients. Mutually exclusive with
        execAffinityParams.


    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    pass

cdef class CUstreamCigParam(CUstreamCigParam_st):
    """
    CIG Stream Capture Params

    Attributes
    ----------

    streamSharedDataType : CUstreamCigDataType
        Type of shared data from graphics client (D3D12).


    streamSharedData : Any
        Graphics client data handle
        (ID3D12CommandList/ID3D12GraphicsCommandList).


    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    pass

cdef class CUstreamCigCaptureParams(CUstreamCigCaptureParams_st):
    """
    Params for capturing CUDA stream to CIG streamCigParams must be
    non-NULL.

    Attributes
    ----------

    streamCigParams : CUstreamCigParam
        CIG (CUDA in Graphics) parameters for sharing command list data
        from D3D12 graphics clients.


    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    pass

cdef class CUlibraryHostUniversalFunctionAndDataTable(CUlibraryHostUniversalFunctionAndDataTable_st):
    """
    Attributes
    ----------

    functionTable : Any



    functionWindowSize : size_t



    dataTable : Any



    dataWindowSize : size_t



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

cdef class CUDA_MEMCPY2D(CUDA_MEMCPY2D_v2):
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

cdef class CUDA_MEMCPY3D(CUDA_MEMCPY3D_v2):
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

cdef class CUDA_MEMCPY3D_PEER(CUDA_MEMCPY3D_PEER_v1):
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

cdef class CUDA_MEMCPY_NODE_PARAMS(CUDA_MEMCPY_NODE_PARAMS_st):
    """
    Memcpy node parameters

    Attributes
    ----------

    flags : int
        Must be zero


    reserved : int
        Must be zero


    copyCtx : CUcontext
        Context on which to run the node


    copyParams : CUDA_MEMCPY3D
        Parameters for the memory copy


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

cdef class CUDA_ARRAY_DESCRIPTOR(CUDA_ARRAY_DESCRIPTOR_v2):
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

cdef class CUDA_ARRAY3D_DESCRIPTOR(CUDA_ARRAY3D_DESCRIPTOR_v2):
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

    tileExtent : anon_struct6



    miptailFirstLevel : unsigned int
        First mip level at which the mip tail begins.


    miptailSize : unsigned long long
        Total size of the mip tail.


    flags : unsigned int
        Flags will either be zero or
        CU_ARRAY_SPARSE_PROPERTIES_SINGLE_MIPTAIL


    reserved : list[unsigned int]



    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    pass

cdef class CUDA_ARRAY_SPARSE_PROPERTIES(CUDA_ARRAY_SPARSE_PROPERTIES_v1):
    """
    CUDA array sparse properties

    Attributes
    ----------

    tileExtent : anon_struct6



    miptailFirstLevel : unsigned int
        First mip level at which the mip tail begins.


    miptailSize : unsigned long long
        Total size of the mip tail.


    flags : unsigned int
        Flags will either be zero or
        CU_ARRAY_SPARSE_PROPERTIES_SINGLE_MIPTAIL


    reserved : list[unsigned int]



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


    reserved : list[unsigned int]



    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    pass

cdef class CUDA_ARRAY_MEMORY_REQUIREMENTS(CUDA_ARRAY_MEMORY_REQUIREMENTS_v1):
    """
    CUDA array memory requirements

    Attributes
    ----------

    size : size_t
        Total required memory size


    alignment : size_t
        alignment requirement


    reserved : list[unsigned int]



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


    res : anon_union4



    flags : unsigned int
        Flags (must be zero)


    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    pass

cdef class CUDA_RESOURCE_DESC(CUDA_RESOURCE_DESC_v1):
    """
    CUDA Resource descriptor

    Attributes
    ----------

    resType : CUresourcetype
        Resource type


    res : anon_union4



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

    addressMode : list[CUaddress_mode]
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


    borderColor : list[float]
        Border Color


    reserved : list[int]



    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    pass

cdef class CUDA_TEXTURE_DESC(CUDA_TEXTURE_DESC_v1):
    """
    Texture descriptor

    Attributes
    ----------

    addressMode : list[CUaddress_mode]
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


    borderColor : list[float]
        Border Color


    reserved : list[int]



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


    reserved : list[unsigned int]



    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    pass

cdef class CUDA_RESOURCE_VIEW_DESC(CUDA_RESOURCE_VIEW_DESC_v1):
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


    reserved : list[unsigned int]



    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    pass

cdef class CUtensorMap(CUtensorMap_st):
    """
    Tensor map descriptor. Requires compiler support for aligning to
    128 bytes.

    Attributes
    ----------

    opaque : list[cuuint64_t]



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

cdef class CUDA_POINTER_ATTRIBUTE_P2P_TOKENS(CUDA_POINTER_ATTRIBUTE_P2P_TOKENS_v1):
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

cdef class CUDA_LAUNCH_PARAMS(CUDA_LAUNCH_PARAMS_v1):
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


    handle : anon_union5



    size : unsigned long long
        Size of the memory allocation


    flags : unsigned int
        Flags must either be zero or CUDA_EXTERNAL_MEMORY_DEDICATED


    reserved : list[unsigned int]



    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    pass

cdef class CUDA_EXTERNAL_MEMORY_HANDLE_DESC(CUDA_EXTERNAL_MEMORY_HANDLE_DESC_v1):
    """
    External memory handle descriptor

    Attributes
    ----------

    type : CUexternalMemoryHandleType
        Type of the handle


    handle : anon_union5



    size : unsigned long long
        Size of the memory allocation


    flags : unsigned int
        Flags must either be zero or CUDA_EXTERNAL_MEMORY_DEDICATED


    reserved : list[unsigned int]



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


    reserved : list[unsigned int]



    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    pass

cdef class CUDA_EXTERNAL_MEMORY_BUFFER_DESC(CUDA_EXTERNAL_MEMORY_BUFFER_DESC_v1):
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


    reserved : list[unsigned int]



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


    reserved : list[unsigned int]



    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    pass

cdef class CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC(CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC_v1):
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


    reserved : list[unsigned int]



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


    handle : anon_union6



    flags : unsigned int
        Flags reserved for the future. Must be zero.


    reserved : list[unsigned int]



    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    pass

cdef class CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC(CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_v1):
    """
    External semaphore handle descriptor

    Attributes
    ----------

    type : CUexternalSemaphoreHandleType
        Type of the handle


    handle : anon_union6



    flags : unsigned int
        Flags reserved for the future. Must be zero.


    reserved : list[unsigned int]



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

    params : anon_struct16



    flags : unsigned int
        Only when CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS is used to signal a
        CUexternalSemaphore of type
        CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_NVSCISYNC, the valid flag is
        CUDA_EXTERNAL_SEMAPHORE_SIGNAL_SKIP_NVSCIBUF_MEMSYNC which
        indicates that while signaling the CUexternalSemaphore, no memory
        synchronization operations should be performed for any external
        memory object imported as CU_EXTERNAL_MEMORY_HANDLE_TYPE_NVSCIBUF.
        For all other types of CUexternalSemaphore, flags must be zero.


    reserved : list[unsigned int]



    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    pass

cdef class CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS(CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_v1):
    """
    External semaphore signal parameters

    Attributes
    ----------

    params : anon_struct16



    flags : unsigned int
        Only when CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS is used to signal a
        CUexternalSemaphore of type
        CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_NVSCISYNC, the valid flag is
        CUDA_EXTERNAL_SEMAPHORE_SIGNAL_SKIP_NVSCIBUF_MEMSYNC which
        indicates that while signaling the CUexternalSemaphore, no memory
        synchronization operations should be performed for any external
        memory object imported as CU_EXTERNAL_MEMORY_HANDLE_TYPE_NVSCIBUF.
        For all other types of CUexternalSemaphore, flags must be zero.


    reserved : list[unsigned int]



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

    params : anon_struct19



    flags : unsigned int
        Only when CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS is used to wait on a
        CUexternalSemaphore of type
        CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_NVSCISYNC, the valid flag is
        CUDA_EXTERNAL_SEMAPHORE_WAIT_SKIP_NVSCIBUF_MEMSYNC which indicates
        that while waiting for the CUexternalSemaphore, no memory
        synchronization operations should be performed for any external
        memory object imported as CU_EXTERNAL_MEMORY_HANDLE_TYPE_NVSCIBUF.
        For all other types of CUexternalSemaphore, flags must be zero.


    reserved : list[unsigned int]



    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    pass

cdef class CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS(CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_v1):
    """
    External semaphore wait parameters

    Attributes
    ----------

    params : anon_struct19



    flags : unsigned int
        Only when CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS is used to wait on a
        CUexternalSemaphore of type
        CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_NVSCISYNC, the valid flag is
        CUDA_EXTERNAL_SEMAPHORE_WAIT_SKIP_NVSCIBUF_MEMSYNC which indicates
        that while waiting for the CUexternalSemaphore, no memory
        synchronization operations should be performed for any external
        memory object imported as CU_EXTERNAL_MEMORY_HANDLE_TYPE_NVSCIBUF.
        For all other types of CUexternalSemaphore, flags must be zero.


    reserved : list[unsigned int]



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

cdef class CUDA_EXT_SEM_SIGNAL_NODE_PARAMS(CUDA_EXT_SEM_SIGNAL_NODE_PARAMS_v1):
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

cdef class CUDA_EXT_SEM_SIGNAL_NODE_PARAMS_v2(CUDA_EXT_SEM_SIGNAL_NODE_PARAMS_v2_st):
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

cdef class CUDA_EXT_SEM_WAIT_NODE_PARAMS(CUDA_EXT_SEM_WAIT_NODE_PARAMS_v1):
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

cdef class CUDA_EXT_SEM_WAIT_NODE_PARAMS_v2(CUDA_EXT_SEM_WAIT_NODE_PARAMS_v2_st):
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

cdef class CUmemGenericAllocationHandle:
    """

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef cydriver.CUmemGenericAllocationHandle  _pvt_val
    cdef cydriver.CUmemGenericAllocationHandle* _pvt_ptr

cdef class CUarrayMapInfo_v1(CUarrayMapInfo_st):
    """
    Specifies the CUDA array or CUDA mipmapped array memory mapping
    information

    Attributes
    ----------

    resourceType : CUresourcetype
        Resource type


    resource : anon_union9



    subresourceType : CUarraySparseSubresourceType
        Sparse subresource type


    subresource : anon_union10



    memOperationType : CUmemOperationType
        Memory operation type


    memHandleType : CUmemHandleType
        Memory handle type


    memHandle : anon_union11



    offset : unsigned long long
        Offset within mip tail  Offset within the memory


    deviceBitMask : unsigned int
        Device ordinal bit mask


    flags : unsigned int
        flags for future use, must be zero now.


    reserved : list[unsigned int]
        Reserved for future use, must be zero now.


    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    pass

cdef class CUarrayMapInfo(CUarrayMapInfo_v1):
    """
    Specifies the CUDA array or CUDA mipmapped array memory mapping
    information

    Attributes
    ----------

    resourceType : CUresourcetype
        Resource type


    resource : anon_union9



    subresourceType : CUarraySparseSubresourceType
        Sparse subresource type


    subresource : anon_union10



    memOperationType : CUmemOperationType
        Memory operation type


    memHandleType : CUmemHandleType
        Memory handle type


    memHandle : anon_union11



    offset : unsigned long long
        Offset within mip tail  Offset within the memory


    deviceBitMask : unsigned int
        Device ordinal bit mask


    flags : unsigned int
        flags for future use, must be zero now.


    reserved : list[unsigned int]
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



    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    pass

cdef class CUmemLocation(CUmemLocation_v1):
    """
    Specifies a memory location.

    Attributes
    ----------

    type : CUmemLocationType
        Specifies the location type, which modifies the meaning of id.


    id : int



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
        CU_MEM_HANDLE_TYPE_WIN32 is specified. This object attributes
        structure includes security attributes that define the scope of
        which exported allocations may be transferred to other processes.
        In all other cases, this field is required to be zero.


    allocFlags : anon_struct22



    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    pass

cdef class CUmemAllocationProp(CUmemAllocationProp_v1):
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
        CU_MEM_HANDLE_TYPE_WIN32 is specified. This object attributes
        structure includes security attributes that define the scope of
        which exported allocations may be transferred to other processes.
        In all other cases, this field is required to be zero.


    allocFlags : anon_struct22



    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    pass

cdef class CUmulticastObjectProp_v1(CUmulticastObjectProp_st):
    """
    Specifies the properties for a multicast object.

    Attributes
    ----------

    numDevices : unsigned int
        The number of devices in the multicast team that will bind memory
        to this object


    size : size_t
        The maximum amount of memory that can be bound to this multicast
        object per device


    handleTypes : unsigned long long
        Bitmask of exportable handle types (see CUmemAllocationHandleType)
        for this object


    flags : unsigned long long
        Flags for future use, must be zero now


    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    pass

cdef class CUmulticastObjectProp(CUmulticastObjectProp_v1):
    """
    Specifies the properties for a multicast object.

    Attributes
    ----------

    numDevices : unsigned int
        The number of devices in the multicast team that will bind memory
        to this object


    size : size_t
        The maximum amount of memory that can be bound to this multicast
        object per device


    handleTypes : unsigned long long
        Bitmask of exportable handle types (see CUmemAllocationHandleType)
        for this object


    flags : unsigned long long
        Flags for future use, must be zero now


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

cdef class CUmemAccessDesc(CUmemAccessDesc_v1):
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

cdef class CUgraphExecUpdateResultInfo_v1(CUgraphExecUpdateResultInfo_st):
    """
    Result information returned by cuGraphExecUpdate

    Attributes
    ----------

    result : CUgraphExecUpdateResult
        Gives more specific detail when a cuda graph update fails.


    errorNode : CUgraphNode
        The "to node" of the error edge when the topologies do not match.
        The error node when the error is associated with a specific node.
        NULL when the error is generic.


    errorFromNode : CUgraphNode
        The from node of error edge when the topologies do not match.
        Otherwise NULL.


    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    pass

cdef class CUgraphExecUpdateResultInfo(CUgraphExecUpdateResultInfo_v1):
    """
    Result information returned by cuGraphExecUpdate

    Attributes
    ----------

    result : CUgraphExecUpdateResult
        Gives more specific detail when a cuda graph update fails.


    errorNode : CUgraphNode
        The "to node" of the error edge when the topologies do not match.
        The error node when the error is associated with a specific node.
        NULL when the error is generic.


    errorFromNode : CUgraphNode
        The from node of error edge when the topologies do not match.
        Otherwise NULL.


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
        defines the scope of which exported allocations may be transferred
        to other processes. In all other cases, this field is required to
        be zero.


    maxSize : size_t
        Maximum pool size. When set to 0, defaults to a system dependent
        value.


    usage : unsigned short
        Bitmask indicating intended usage for the pool.


    reserved : bytes
        reserved for future use, must be 0


    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    pass

cdef class CUmemPoolProps(CUmemPoolProps_v1):
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
        defines the scope of which exported allocations may be transferred
        to other processes. In all other cases, this field is required to
        be zero.


    maxSize : size_t
        Maximum pool size. When set to 0, defaults to a system dependent
        value.


    usage : unsigned short
        Bitmask indicating intended usage for the pool.


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

cdef class CUmemPoolPtrExportData(CUmemPoolPtrExportData_v1):
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

cdef class CUmemcpyAttributes_v1(CUmemcpyAttributes_st):
    """
    Attributes specific to copies within a batch. For more details on
    usage see cuMemcpyBatchAsync.

    Attributes
    ----------

    srcAccessOrder : CUmemcpySrcAccessOrder
        Source access ordering to be observed for copies with this
        attribute.


    srcLocHint : CUmemLocation
        Hint location for the source operand. Ignored when the pointers are
        not managed memory or memory allocated outside CUDA.


    dstLocHint : CUmemLocation
        Hint location for the destination operand. Ignored when the
        pointers are not managed memory or memory allocated outside CUDA.


    flags : unsigned int
        Additional flags for copies with this attribute. See CUmemcpyFlags


    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    pass

cdef class CUmemcpyAttributes(CUmemcpyAttributes_v1):
    """
    Attributes specific to copies within a batch. For more details on
    usage see cuMemcpyBatchAsync.

    Attributes
    ----------

    srcAccessOrder : CUmemcpySrcAccessOrder
        Source access ordering to be observed for copies with this
        attribute.


    srcLocHint : CUmemLocation
        Hint location for the source operand. Ignored when the pointers are
        not managed memory or memory allocated outside CUDA.


    dstLocHint : CUmemLocation
        Hint location for the destination operand. Ignored when the
        pointers are not managed memory or memory allocated outside CUDA.


    flags : unsigned int
        Additional flags for copies with this attribute. See CUmemcpyFlags


    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    pass

cdef class CUoffset3D_v1(CUoffset3D_st):
    """
    Struct representing a 3D offset

    Attributes
    ----------

    x : size_t



    y : size_t



    z : size_t



    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    pass

cdef class CUoffset3D(CUoffset3D_v1):
    """
    Struct representing a 3D offset

    Attributes
    ----------

    x : size_t



    y : size_t



    z : size_t



    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    pass

cdef class CUextent3D_v1(CUextent3D_st):
    """
    Struct representing width/height/depth of a CUarray in elements

    Attributes
    ----------

    width : size_t



    height : size_t



    depth : size_t



    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    pass

cdef class CUextent3D(CUextent3D_v1):
    """
    Struct representing width/height/depth of a CUarray in elements

    Attributes
    ----------

    width : size_t



    height : size_t



    depth : size_t



    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    pass

cdef class CUmemcpy3DOperand_v1(CUmemcpy3DOperand_st):
    """
    Struct representing an operand for copy with cuMemcpy3DBatchAsync

    Attributes
    ----------

    type : CUmemcpy3DOperandType



    op : anon_union13



    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    pass

cdef class CUmemcpy3DOperand(CUmemcpy3DOperand_v1):
    """
    Struct representing an operand for copy with cuMemcpy3DBatchAsync

    Attributes
    ----------

    type : CUmemcpy3DOperandType



    op : anon_union13



    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    pass

cdef class CUDA_MEMCPY3D_BATCH_OP_v1(CUDA_MEMCPY3D_BATCH_OP_st):
    """
    Attributes
    ----------

    src : CUmemcpy3DOperand
        Source memcpy operand.


    dst : CUmemcpy3DOperand
        Destination memcpy operand.


    extent : CUextent3D
        Extents of the memcpy between src and dst. The width, height and
        depth components must not be 0.


    srcAccessOrder : CUmemcpySrcAccessOrder
        Source access ordering to be observed for copy from src to dst.


    flags : unsigned int
        Additional flags for copies with this attribute. See CUmemcpyFlags


    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    pass

cdef class CUDA_MEMCPY3D_BATCH_OP(CUDA_MEMCPY3D_BATCH_OP_v1):
    """
    Attributes
    ----------

    src : CUmemcpy3DOperand
        Source memcpy operand.


    dst : CUmemcpy3DOperand
        Destination memcpy operand.


    extent : CUextent3D
        Extents of the memcpy between src and dst. The width, height and
        depth components must not be 0.


    srcAccessOrder : CUmemcpySrcAccessOrder
        Source access ordering to be observed for copy from src to dst.


    flags : unsigned int
        Additional flags for copies with this attribute. See CUmemcpyFlags


    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    pass

cdef class CUDA_MEM_ALLOC_NODE_PARAMS_v1(CUDA_MEM_ALLOC_NODE_PARAMS_v1_st):
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

cdef class CUDA_MEM_ALLOC_NODE_PARAMS(CUDA_MEM_ALLOC_NODE_PARAMS_v1):
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

cdef class CUDA_MEM_ALLOC_NODE_PARAMS_v2(CUDA_MEM_ALLOC_NODE_PARAMS_v2_st):
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

cdef class CUDA_MEM_FREE_NODE_PARAMS(CUDA_MEM_FREE_NODE_PARAMS_st):
    """
    Memory free node parameters

    Attributes
    ----------

    dptr : CUdeviceptr
        in: the pointer to free


    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    pass

cdef class CUDA_CHILD_GRAPH_NODE_PARAMS(CUDA_CHILD_GRAPH_NODE_PARAMS_st):
    """
    Child graph node parameters

    Attributes
    ----------

    graph : CUgraph
        The child graph to clone into the node for node creation, or a
        handle to the graph owned by the node for node query. The graph
        must not contain conditional nodes. Graphs containing memory
        allocation or memory free nodes must set the ownership to be moved
        to the parent.


    ownership : CUgraphChildGraphNodeOwnership
        The ownership relationship of the child graph node.


    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    pass

cdef class CUDA_EVENT_RECORD_NODE_PARAMS(CUDA_EVENT_RECORD_NODE_PARAMS_st):
    """
    Event record node parameters

    Attributes
    ----------

    event : CUevent
        The event to record when the node executes


    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    pass

cdef class CUDA_EVENT_WAIT_NODE_PARAMS(CUDA_EVENT_WAIT_NODE_PARAMS_st):
    """
    Event wait node parameters

    Attributes
    ----------

    event : CUevent
        The event to wait on from the node


    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    pass

cdef class CUgraphNodeParams(CUgraphNodeParams_st):
    """
    Graph node parameters. See cuGraphAddNode.

    Attributes
    ----------

    type : CUgraphNodeType
        Type of the node


    reserved0 : list[int]
        Reserved. Must be zero.


    reserved1 : list[long long]
        Padding. Unused bytes must be zero.


    kernel : CUDA_KERNEL_NODE_PARAMS_v3
        Kernel node parameters.


    memcpy : CUDA_MEMCPY_NODE_PARAMS
        Memcpy node parameters.


    memset : CUDA_MEMSET_NODE_PARAMS_v2
        Memset node parameters.


    host : CUDA_HOST_NODE_PARAMS_v2
        Host node parameters.


    graph : CUDA_CHILD_GRAPH_NODE_PARAMS
        Child graph node parameters.


    eventWait : CUDA_EVENT_WAIT_NODE_PARAMS
        Event wait node parameters.


    eventRecord : CUDA_EVENT_RECORD_NODE_PARAMS
        Event record node parameters.


    extSemSignal : CUDA_EXT_SEM_SIGNAL_NODE_PARAMS_v2
        External semaphore signal node parameters.


    extSemWait : CUDA_EXT_SEM_WAIT_NODE_PARAMS_v2
        External semaphore wait node parameters.


    alloc : CUDA_MEM_ALLOC_NODE_PARAMS_v2
        Memory allocation node parameters.


    free : CUDA_MEM_FREE_NODE_PARAMS
        Memory free node parameters.


    memOp : CUDA_BATCH_MEM_OP_NODE_PARAMS_v2
        MemOp node parameters.


    conditional : CUDA_CONDITIONAL_NODE_PARAMS
        Conditional node parameters.


    reserved2 : long long
        Reserved bytes. Must be zero.


    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    pass

cdef class CUcheckpointLockArgs(CUcheckpointLockArgs_st):
    """
    CUDA checkpoint optional lock arguments

    Attributes
    ----------

    timeoutMs : unsigned int
        Timeout in milliseconds to attempt to lock the process, 0 indicates
        no timeout


    reserved0 : unsigned int
        Reserved for future use, must be zero


    reserved1 : list[cuuint64_t]
        Reserved for future use, must be zeroed


    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    pass

cdef class CUcheckpointCheckpointArgs(CUcheckpointCheckpointArgs_st):
    """
    CUDA checkpoint optional checkpoint arguments

    Attributes
    ----------

    reserved : list[cuuint64_t]
        Reserved for future use, must be zeroed


    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    pass

cdef class CUcheckpointGpuPair(CUcheckpointGpuPair_st):
    """
    CUDA checkpoint GPU UUID pairs for device remapping during restore

    Attributes
    ----------

    oldUuid : CUuuid
        UUID of the GPU that was checkpointed


    newUuid : CUuuid
        UUID of the GPU to restore onto


    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    pass

cdef class CUcheckpointRestoreArgs(CUcheckpointRestoreArgs_st):
    """
    CUDA checkpoint optional restore arguments

    Attributes
    ----------

    gpuPairs : CUcheckpointGpuPair
        Pointer to array of gpu pairs that indicate how to remap GPUs
        during restore


    gpuPairsCount : unsigned int
        Number of gpu pairs to remap


    reserved : bytes
        Reserved for future use, must be zeroed


    reserved1 : cuuint64_t
        Reserved for future use, must be zeroed


    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    pass

cdef class CUcheckpointUnlockArgs(CUcheckpointUnlockArgs_st):
    """
    CUDA checkpoint optional unlock arguments

    Attributes
    ----------

    reserved : list[cuuint64_t]
        Reserved for future use, must be zeroed


    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    pass

cdef class CUmemDecompressParams(CUmemDecompressParams_st):
    """
    Structure describing the parameters that compose a single
    decompression operation.

    Attributes
    ----------

    srcNumBytes : size_t
        The number of bytes to be read and decompressed from
        CUmemDecompressParams_st.src.


    dstNumBytes : size_t
        The number of bytes that the decompression operation will be
        expected to write to CUmemDecompressParams_st.dst. This value is
        optional; if present, it may be used by the CUDA driver as a
        heuristic for scheduling the individual decompression operations.


    dstActBytes : cuuint32_t
        After the decompression operation has completed, the actual number
        of bytes written to CUmemDecompressParams.dst will be recorded as a
        32-bit unsigned integer in the memory at this address.


    src : Any
        Pointer to a buffer of at least
        CUmemDecompressParams_st.srcNumBytes compressed bytes.


    dst : Any
        Pointer to a buffer where the decompressed data will be written.
        The number of bytes written to this location will be recorded in
        the memory pointed to by CUmemDecompressParams_st.dstActBytes


    algo : CUmemDecompressAlgorithm
        The decompression algorithm to use.


    padding : bytes



    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    pass

cdef class CUdevSmResource(CUdevSmResource_st):
    """
    Attributes
    ----------

    smCount : unsigned int
        The amount of streaming multiprocessors available in this resource.


    minSmPartitionSize : unsigned int
        The minimum number of streaming multiprocessors required to
        partition this resource.


    smCoscheduledAlignment : unsigned int
        The number of streaming multiprocessors in this resource that are
        guaranteed to be co-scheduled on the same GPU processing cluster.
        smCount will be a multiple of this value, unless the backfill flag
        is set.


    flags : unsigned int
        The flags set on this SM resource. For possible values see
        CUdevSmResourceGroup_flags.


    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    pass

cdef class CUdevWorkqueueConfigResource(CUdevWorkqueueConfigResource_st):
    """
    Attributes
    ----------

    device : CUdevice
        The device on which the workqueue resources are available


    wqConcurrencyLimit : unsigned int
        The expected maximum number of concurrent stream-ordered workloads


    sharingScope : CUdevWorkqueueConfigScope
        The sharing scope for the workqueue resources


    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    pass

cdef class CUdevWorkqueueResource(CUdevWorkqueueResource_st):
    """
    Attributes
    ----------

    reserved : bytes
        Reserved for future use


    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    pass

cdef class CU_DEV_SM_RESOURCE_GROUP_PARAMS(CU_DEV_SM_RESOURCE_GROUP_PARAMS_st):
    """
    Attributes
    ----------

    smCount : unsigned int
        The amount of SMs available in this resource.


    coscheduledSmCount : unsigned int
        The amount of co-scheduled SMs grouped together for locality
        purposes.


    preferredCoscheduledSmCount : unsigned int
        When possible, combine co-scheduled groups together into larger
        groups of this size.


    flags : unsigned int
        The flags set on this SM resource group. For possible values see
        CUdevSmResourceGroup_flags.


    reserved : list[unsigned int]



    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    pass

cdef class CUdevResource_v1(CUdevResource_st):
    """
    Attributes
    ----------

    type : CUdevResourceType
        Type of resource, dictates which union field was last set


    _internal_padding : bytes



    sm : CUdevSmResource
        Resource corresponding to CU_DEV_RESOURCE_TYPE_SM `typename`.


    wqConfig : CUdevWorkqueueConfigResource
        Resource corresponding to CU_DEV_RESOURCE_TYPE_WORKQUEUE_CONFIG
        `typename`.


    wq : CUdevWorkqueueResource
        Resource corresponding to CU_DEV_RESOURCE_TYPE_WORKQUEUE
        `typename`.


    _oversize : bytes



    nextResource : CUdevResource_st



    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    pass

cdef class CUdevResource(CUdevResource_v1):
    """
    Attributes
    ----------

    type : CUdevResourceType
        Type of resource, dictates which union field was last set


    _internal_padding : bytes



    sm : CUdevSmResource
        Resource corresponding to CU_DEV_RESOURCE_TYPE_SM `typename`.


    wqConfig : CUdevWorkqueueConfigResource
        Resource corresponding to CU_DEV_RESOURCE_TYPE_WORKQUEUE_CONFIG
        `typename`.


    wq : CUdevWorkqueueResource
        Resource corresponding to CU_DEV_RESOURCE_TYPE_WORKQUEUE
        `typename`.


    _oversize : bytes



    nextResource : CUdevResource_st



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

    frame : anon_union16



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

cdef class CUeglFrame(CUeglFrame_v1):
    """
    CUDA EGLFrame structure Descriptor - structure defining one frame
    of EGL.  Each frame may contain one or more planes depending on
    whether the surface * is Multiplanar or not.

    Attributes
    ----------

    frame : anon_union16



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
    cdef cydriver.cuuint32_t  _pvt_val
    cdef cydriver.cuuint32_t* _pvt_ptr

cdef class cuuint64_t:
    """

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef cydriver.cuuint64_t  _pvt_val
    cdef cydriver.cuuint64_t* _pvt_ptr

cdef class CUdeviceptr_v2:
    """

    CUDA device pointer CUdeviceptr is defined as an unsigned integer type whose size matches the size of a pointer on the target platform.

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef cydriver.CUdeviceptr_v2  _pvt_val
    cdef cydriver.CUdeviceptr_v2* _pvt_ptr

cdef class CUdevice_v1:
    """

    CUDA device

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef cydriver.CUdevice_v1  _pvt_val
    cdef cydriver.CUdevice_v1* _pvt_ptr

cdef class CUtexObject_v1:
    """

    An opaque value that represents a CUDA texture object

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef cydriver.CUtexObject_v1  _pvt_val
    cdef cydriver.CUtexObject_v1* _pvt_ptr

cdef class CUsurfObject_v1:
    """

    An opaque value that represents a CUDA surface object

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef cydriver.CUsurfObject_v1  _pvt_val
    cdef cydriver.CUsurfObject_v1* _pvt_ptr

cdef class CUmemGenericAllocationHandle_v1:
    """

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef cydriver.CUmemGenericAllocationHandle_v1  _pvt_val
    cdef cydriver.CUmemGenericAllocationHandle_v1* _pvt_ptr

cdef class CUlogIterator:
    """

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef cydriver.CUlogIterator  _pvt_val
    cdef cydriver.CUlogIterator* _pvt_ptr

cdef class GLenum:
    """

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef cydriver.GLenum  _pvt_val
    cdef cydriver.GLenum* _pvt_ptr

cdef class GLuint:
    """

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef cydriver.GLuint  _pvt_val
    cdef cydriver.GLuint* _pvt_ptr

cdef class EGLint:
    """

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef cydriver.EGLint  _pvt_val
    cdef cydriver.EGLint* _pvt_ptr

cdef class VdpDevice:
    """

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef cydriver.VdpDevice  _pvt_val
    cdef cydriver.VdpDevice* _pvt_ptr

cdef class VdpGetProcAddress:
    """

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef cydriver.VdpGetProcAddress  _pvt_val
    cdef cydriver.VdpGetProcAddress* _pvt_ptr

cdef class VdpVideoSurface:
    """

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef cydriver.VdpVideoSurface  _pvt_val
    cdef cydriver.VdpVideoSurface* _pvt_ptr

cdef class VdpOutputSurface:
    """

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef cydriver.VdpOutputSurface  _pvt_val
    cdef cydriver.VdpOutputSurface* _pvt_ptr
