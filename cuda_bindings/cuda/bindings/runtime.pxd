# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# This code was automatically generated with version 13.3.0. Do not modify it directly.
# CYTHON-BINDINGS-GENERATED-DO-NOT-MODIFY-THIS-FILE: format=1; content-sha256=ea44514ba11796759754ddfb1ae93c9b7e1d510a09f7349a82a2201dd0c0c159
cimport cuda.bindings.cyruntime as cyruntime

include "_lib/utils.pxd"
cimport cuda.bindings.driver as driver

cdef class cudaDevResourceDesc_t:
    """

    An opaque descriptor handle. The descriptor encapsulates multiple created and configured resources. Created via ::cudaDeviceResourceGenerateDesc

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef cyruntime.cudaDevResourceDesc_t  _pvt_val
    cdef cyruntime.cudaDevResourceDesc_t* _pvt_ptr

cdef class cudaExecutionContext_t:
    """

    An opaque handle to a CUDA execution context. It represents an execution context created via CUDA Runtime APIs such as cudaGreenCtxCreate.

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef cyruntime.cudaExecutionContext_t  _pvt_val
    cdef cyruntime.cudaExecutionContext_t* _pvt_ptr

cdef class cudaArray_t:
    """

    CUDA array

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef cyruntime.cudaArray_t  _pvt_val
    cdef cyruntime.cudaArray_t* _pvt_ptr

cdef class cudaArray_const_t:
    """

    CUDA array (as source copy argument)

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef cyruntime.cudaArray_const_t  _pvt_val
    cdef cyruntime.cudaArray_const_t* _pvt_ptr

cdef class cudaMipmappedArray_t:
    """

    CUDA mipmapped array

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef cyruntime.cudaMipmappedArray_t  _pvt_val
    cdef cyruntime.cudaMipmappedArray_t* _pvt_ptr

cdef class cudaMipmappedArray_const_t:
    """

    CUDA mipmapped array (as source argument)

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef cyruntime.cudaMipmappedArray_const_t  _pvt_val
    cdef cyruntime.cudaMipmappedArray_const_t* _pvt_ptr

cdef class cudaGraphicsResource_t:
    """

    CUDA graphics resource types

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef cyruntime.cudaGraphicsResource_t  _pvt_val
    cdef cyruntime.cudaGraphicsResource_t* _pvt_ptr

cdef class cudaExternalMemory_t:
    """

    CUDA external memory

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef cyruntime.cudaExternalMemory_t  _pvt_val
    cdef cyruntime.cudaExternalMemory_t* _pvt_ptr

cdef class cudaExternalSemaphore_t:
    """

    CUDA external semaphore

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef cyruntime.cudaExternalSemaphore_t  _pvt_val
    cdef cyruntime.cudaExternalSemaphore_t* _pvt_ptr

cdef class cudaKernel_t:
    """

    CUDA kernel

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef cyruntime.cudaKernel_t  _pvt_val
    cdef cyruntime.cudaKernel_t* _pvt_ptr

cdef class cudaLibrary_t:
    """

    CUDA library

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef cyruntime.cudaLibrary_t  _pvt_val
    cdef cyruntime.cudaLibrary_t* _pvt_ptr

cdef class cudaGraphDeviceNode_t:
    """

    CUDA device node handle for device-side node update

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef cyruntime.cudaGraphDeviceNode_t  _pvt_val
    cdef cyruntime.cudaGraphDeviceNode_t* _pvt_ptr

cdef class cudaAsyncCallbackHandle_t:
    """

    CUDA async callback handle

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef cyruntime.cudaAsyncCallbackHandle_t  _pvt_val
    cdef cyruntime.cudaAsyncCallbackHandle_t* _pvt_ptr

cdef class cudaLogsCallbackHandle:
    """

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef cyruntime.cudaLogsCallbackHandle  _pvt_val
    cdef cyruntime.cudaLogsCallbackHandle* _pvt_ptr

cdef class EGLImageKHR:
    """

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef cyruntime.EGLImageKHR  _pvt_val
    cdef cyruntime.EGLImageKHR* _pvt_ptr

cdef class EGLStreamKHR:
    """

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef cyruntime.EGLStreamKHR  _pvt_val
    cdef cyruntime.EGLStreamKHR* _pvt_ptr

cdef class EGLSyncKHR:
    """

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef cyruntime.EGLSyncKHR  _pvt_val
    cdef cyruntime.EGLSyncKHR* _pvt_ptr

cdef class cudaHostFn_t:
    """

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef cyruntime.cudaHostFn_t  _pvt_val
    cdef cyruntime.cudaHostFn_t* _pvt_ptr

cdef class cudaAsyncCallback:
    """

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef cyruntime.cudaAsyncCallback  _pvt_val
    cdef cyruntime.cudaAsyncCallback* _pvt_ptr

cdef class cudaStreamCallback_t:
    """

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef cyruntime.cudaStreamCallback_t  _pvt_val
    cdef cyruntime.cudaStreamCallback_t* _pvt_ptr

cdef class cudaGraphRecaptureCallback_t:
    """

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef cyruntime.cudaGraphRecaptureCallback_t  _pvt_val
    cdef cyruntime.cudaGraphRecaptureCallback_t* _pvt_ptr

cdef class cudaLogsCallback_t:
    """

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef cyruntime.cudaLogsCallback_t  _pvt_val
    cdef cyruntime.cudaLogsCallback_t* _pvt_ptr

cdef class dim3:
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
    cdef cyruntime.dim3 _pvt_val
    cdef cyruntime.dim3* _pvt_ptr

cdef class cudaChannelFormatDesc:
    """
    CUDA Channel format descriptor

    Attributes
    ----------

    x : int
        x


    y : int
        y


    z : int
        z


    w : int
        w


    f : cudaChannelFormatKind
        Channel format kind


    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cyruntime.cudaChannelFormatDesc _pvt_val
    cdef cyruntime.cudaChannelFormatDesc* _pvt_ptr

cdef class anon_struct0:
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
    cdef cyruntime.cudaArraySparseProperties* _pvt_ptr

cdef class cudaArraySparseProperties:
    """
    Sparse CUDA array and CUDA mipmapped array properties

    Attributes
    ----------

    tileExtent : anon_struct0



    miptailFirstLevel : unsigned int
        First mip level at which the mip tail begins


    miptailSize : unsigned long long
        Total size of the mip tail.


    flags : unsigned int
        Flags will either be zero or cudaArraySparsePropertiesSingleMipTail


    reserved : list[unsigned int]



    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cyruntime.cudaArraySparseProperties _pvt_val
    cdef cyruntime.cudaArraySparseProperties* _pvt_ptr

    cdef anon_struct0 _tileExtent


cdef class cudaArrayMemoryRequirements:
    """
    CUDA array and CUDA mipmapped array memory requirements

    Attributes
    ----------

    size : size_t
        Total size of the array.


    alignment : size_t
        Alignment necessary for mapping the array.


    reserved : list[unsigned int]



    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cyruntime.cudaArrayMemoryRequirements _pvt_val
    cdef cyruntime.cudaArrayMemoryRequirements* _pvt_ptr

cdef class cudaPitchedPtr:
    """
    CUDA Pitched memory pointer  make_cudaPitchedPtr

    Attributes
    ----------

    ptr : Any
        Pointer to allocated memory


    pitch : size_t
        Pitch of allocated memory in bytes


    xsize : size_t
        Logical width of allocation in elements


    ysize : size_t
        Logical height of allocation in elements


    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cyruntime.cudaPitchedPtr _pvt_val
    cdef cyruntime.cudaPitchedPtr* _pvt_ptr

    cdef _HelperInputVoidPtr _cyptr


cdef class cudaExtent:
    """
    CUDA extent  make_cudaExtent

    Attributes
    ----------

    width : size_t
        Width in elements when referring to array memory, in bytes when
        referring to linear memory


    height : size_t
        Height in elements


    depth : size_t
        Depth in elements


    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cyruntime.cudaExtent _pvt_val
    cdef cyruntime.cudaExtent* _pvt_ptr

cdef class cudaPos:
    """
    CUDA 3D position  make_cudaPos

    Attributes
    ----------

    x : size_t
        x


    y : size_t
        y


    z : size_t
        z


    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cyruntime.cudaPos _pvt_val
    cdef cyruntime.cudaPos* _pvt_ptr

cdef class cudaMemcpy3DParms:
    """
    CUDA 3D memory copying parameters

    Attributes
    ----------

    srcArray : cudaArray_t
        Source memory address


    srcPos : cudaPos
        Source position offset


    srcPtr : cudaPitchedPtr
        Pitched source memory address


    dstArray : cudaArray_t
        Destination memory address


    dstPos : cudaPos
        Destination position offset


    dstPtr : cudaPitchedPtr
        Pitched destination memory address


    extent : cudaExtent
        Requested memory copy size


    kind : cudaMemcpyKind
        Type of transfer


    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cyruntime.cudaMemcpy3DParms _pvt_val
    cdef cyruntime.cudaMemcpy3DParms* _pvt_ptr

    cdef cudaArray_t _srcArray


    cdef cudaPos _srcPos


    cdef cudaPitchedPtr _srcPtr


    cdef cudaArray_t _dstArray


    cdef cudaPos _dstPos


    cdef cudaPitchedPtr _dstPtr


    cdef cudaExtent _extent


cdef class cudaMemcpyNodeParams:
    """
    Memcpy node parameters

    Attributes
    ----------

    flags : int
        Must be zero


    reserved : int
        Must be zero


    ctx : cudaExecutionContext_t
        Context in which to run the memcpy. If NULL will try to use the
        current context.


    copyParams : cudaMemcpy3DParms
        Parameters for the memory copy


    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cyruntime.cudaMemcpyNodeParams _pvt_val
    cdef cyruntime.cudaMemcpyNodeParams* _pvt_ptr

    cdef cudaExecutionContext_t _ctx


    cdef cudaMemcpy3DParms _copyParams


cdef class cudaMemcpy3DPeerParms:
    """
    CUDA 3D cross-device memory copying parameters

    Attributes
    ----------

    srcArray : cudaArray_t
        Source memory address


    srcPos : cudaPos
        Source position offset


    srcPtr : cudaPitchedPtr
        Pitched source memory address


    srcDevice : int
        Source device


    dstArray : cudaArray_t
        Destination memory address


    dstPos : cudaPos
        Destination position offset


    dstPtr : cudaPitchedPtr
        Pitched destination memory address


    dstDevice : int
        Destination device


    extent : cudaExtent
        Requested memory copy size


    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cyruntime.cudaMemcpy3DPeerParms _pvt_val
    cdef cyruntime.cudaMemcpy3DPeerParms* _pvt_ptr

    cdef cudaArray_t _srcArray


    cdef cudaPos _srcPos


    cdef cudaPitchedPtr _srcPtr


    cdef cudaArray_t _dstArray


    cdef cudaPos _dstPos


    cdef cudaPitchedPtr _dstPtr


    cdef cudaExtent _extent


cdef class cudaMemsetParams:
    """
    CUDA Memset node parameters

    Attributes
    ----------

    dst : Any
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
    cdef cyruntime.cudaMemsetParams _pvt_val
    cdef cyruntime.cudaMemsetParams* _pvt_ptr

    cdef _HelperInputVoidPtr _cydst


cdef class cudaMemsetParamsV2:
    """
    CUDA Memset node parameters

    Attributes
    ----------

    dst : Any
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


    ctx : cudaExecutionContext_t
        Context in which to run the memset. If NULL will try to use the
        current context.


    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cyruntime.cudaMemsetParamsV2 _pvt_val
    cdef cyruntime.cudaMemsetParamsV2* _pvt_ptr

    cdef _HelperInputVoidPtr _cydst


    cdef cudaExecutionContext_t _ctx


cdef class cudaAccessPolicyWindow:
    """
    Specifies an access policy for a window, a contiguous extent of
    memory beginning at base_ptr and ending at base_ptr + num_bytes.
    Partition into many segments and assign segments such that. sum of
    "hit segments" / window == approx. ratio. sum of "miss segments" /
    window == approx 1-ratio. Segments and ratio specifications are
    fitted to the capabilities of the architecture. Accesses in a hit
    segment apply the hitProp access policy. Accesses in a miss segment
    apply the missProp access policy.

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


    hitProp : cudaAccessProperty
        ::CUaccessProperty set for hit.


    missProp : cudaAccessProperty
        ::CUaccessProperty set for miss. Must be either NORMAL or
        STREAMING.


    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cyruntime.cudaAccessPolicyWindow _pvt_val
    cdef cyruntime.cudaAccessPolicyWindow* _pvt_ptr

    cdef _HelperInputVoidPtr _cybase_ptr


cdef class cudaHostNodeParams:
    """
    CUDA host node parameters

    Attributes
    ----------

    fn : cudaHostFn_t
        The function to call when the node executes


    userData : Any
        Argument to pass to the function


    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cyruntime.cudaHostNodeParams _pvt_val
    cdef cyruntime.cudaHostNodeParams* _pvt_ptr

    cdef cudaHostFn_t _fn


    cdef _HelperInputVoidPtr _cyuserData


cdef class cudaHostNodeParamsV2:
    """
    CUDA host node parameters

    Attributes
    ----------

    fn : cudaHostFn_t
        The function to call when the node executes


    userData : Any
        Argument to pass to the function


    syncMode : unsigned int
        The synchronization mode to use for the host task


    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cyruntime.cudaHostNodeParamsV2 _pvt_val
    cdef cyruntime.cudaHostNodeParamsV2* _pvt_ptr

    cdef cudaHostFn_t _fn


    cdef _HelperInputVoidPtr _cyuserData


cdef class anon_struct1:
    """
    Attributes
    ----------

    array : cudaArray_t



    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cyruntime.cudaResourceDesc* _pvt_ptr

    cdef cudaArray_t _array


cdef class anon_struct2:
    """
    Attributes
    ----------

    mipmap : cudaMipmappedArray_t



    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cyruntime.cudaResourceDesc* _pvt_ptr

    cdef cudaMipmappedArray_t _mipmap


cdef class anon_struct3:
    """
    Attributes
    ----------

    devPtr : Any



    desc : cudaChannelFormatDesc



    sizeInBytes : size_t



    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cyruntime.cudaResourceDesc* _pvt_ptr

    cdef _HelperInputVoidPtr _cydevPtr


    cdef cudaChannelFormatDesc _desc


cdef class anon_struct4:
    """
    Attributes
    ----------

    devPtr : Any



    desc : cudaChannelFormatDesc



    width : size_t



    height : size_t



    pitchInBytes : size_t



    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cyruntime.cudaResourceDesc* _pvt_ptr

    cdef _HelperInputVoidPtr _cydevPtr


    cdef cudaChannelFormatDesc _desc


cdef class anon_struct5:
    """
    Attributes
    ----------

    reserved : list[int]



    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cyruntime.cudaResourceDesc* _pvt_ptr

cdef class anon_union0:
    """
    Attributes
    ----------

    array : anon_struct1



    mipmap : anon_struct2



    linear : anon_struct3



    pitch2D : anon_struct4



    reserved : anon_struct5



    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cyruntime.cudaResourceDesc* _pvt_ptr

    cdef anon_struct1 _array


    cdef anon_struct2 _mipmap


    cdef anon_struct3 _linear


    cdef anon_struct4 _pitch2D


    cdef anon_struct5 _reserved


cdef class cudaResourceDesc:
    """
    CUDA resource descriptor

    Attributes
    ----------

    resType : cudaResourceType
        Resource type


    res : anon_union0



    flags : unsigned int
        Flags (must be zero)


    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cyruntime.cudaResourceDesc* _val_ptr
    cdef cyruntime.cudaResourceDesc* _pvt_ptr

    cdef anon_union0 _res


cdef class cudaResourceViewDesc:
    """
    CUDA resource view descriptor

    Attributes
    ----------

    format : cudaResourceViewFormat
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
        Must be zero


    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cyruntime.cudaResourceViewDesc _pvt_val
    cdef cyruntime.cudaResourceViewDesc* _pvt_ptr

cdef class cudaPointerAttributes:
    """
    CUDA pointer attributes

    Attributes
    ----------

    type : cudaMemoryType
        The type of memory - cudaMemoryTypeUnregistered,
        cudaMemoryTypeHost, cudaMemoryTypeDevice or cudaMemoryTypeManaged.


    device : int
        The device against which the memory was allocated or registered. If
        the memory type is cudaMemoryTypeDevice then this identifies the
        device on which the memory referred physically resides. If the
        memory type is cudaMemoryTypeHost or::cudaMemoryTypeManaged then
        this identifies the device which was current when the memory was
        allocated or registered (and if that device is deinitialized then
        this allocation will vanish with that device's state).


    devicePointer : Any
        The address which may be dereferenced on the current device to
        access the memory or NULL if no such address exists.


    hostPointer : Any
        The address which may be dereferenced on the host to access the
        memory or NULL if no such address exists.  CUDA doesn't check if
        unregistered memory is allocated so this field may contain invalid
        pointer if an invalid pointer has been passed to CUDA.


    reserved : list[long]
        Must be zero


    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cyruntime.cudaPointerAttributes _pvt_val
    cdef cyruntime.cudaPointerAttributes* _pvt_ptr

    cdef _HelperInputVoidPtr _cydevicePointer


    cdef _HelperInputVoidPtr _cyhostPointer


cdef class cudaFuncAttributes:
    """
    CUDA function attributes

    Attributes
    ----------

    sharedSizeBytes : size_t
        The size in bytes of statically-allocated shared memory per block
        required by this function. This does not include dynamically-
        allocated shared memory requested by the user at runtime.


    constSizeBytes : size_t
        The size in bytes of user-allocated constant memory required by
        this function.


    localSizeBytes : size_t
        The size in bytes of local memory used by each thread of this
        function.


    maxThreadsPerBlock : int
        The maximum number of threads per block, beyond which a launch of
        the function would fail. This number depends on both the function
        and the device on which the function is currently loaded.


    numRegs : int
        The number of registers used by each thread of this function.


    ptxVersion : int
        The PTX virtual architecture version for which the function was
        compiled. This value is the major PTX version * 10 + the minor PTX
        version, so a PTX version 1.3 function would return the value 13.


    binaryVersion : int
        The binary architecture version for which the function was
        compiled. This value is the major binary version * 10 + the minor
        binary version, so a binary version 1.3 function would return the
        value 13.


    cacheModeCA : int
        The attribute to indicate whether the function has been compiled
        with user specified option "-Xptxas --dlcm=ca" set.


    maxDynamicSharedSizeBytes : int
        The maximum size in bytes of dynamic shared memory per block for
        this function. Any launch must have a dynamic shared memory size
        smaller than this value.


    preferredShmemCarveout : int
        On devices where the L1 cache and shared memory use the same
        hardware resources, this sets the shared memory carveout
        preference, in percent of the maximum shared memory. Refer to
        cudaDevAttrMaxSharedMemoryPerMultiprocessor. This is only a hint,
        and the driver can choose a different ratio if required to execute
        the function. See cudaFuncSetAttribute


    clusterDimMustBeSet : int
        If this attribute is set, the kernel must launch with a valid
        cluster dimension specified.


    requiredClusterWidth : int
        The required cluster width/height/depth in blocks. The values must
        either all be 0 or all be positive. The validity of the cluster
        dimensions is otherwise checked at launch time.  If the value is
        set during compile time, it cannot be set at runtime. Setting it at
        runtime should return cudaErrorNotPermitted. See
        cudaFuncSetAttribute


    requiredClusterHeight : int



    requiredClusterDepth : int



    clusterSchedulingPolicyPreference : int
        The block scheduling policy of a function. See cudaFuncSetAttribute


    nonPortableClusterSizeAllowed : int
        Whether the function can be launched with non-portable cluster
        size. 1 is allowed, 0 is disallowed. A non-portable cluster size
        may only function on the specific SKUs the program is tested on.
        The launch might fail if the program is run on a different hardware
        platform.  CUDA API provides cudaOccupancyMaxActiveClusters to
        assist with checking whether the desired size can be launched on
        the current device.  Portable Cluster Size  A portable cluster size
        is guaranteed to be functional on all compute capabilities higher
        than the target compute capability. The portable cluster size for
        sm_90 is 8 blocks per cluster. This value may increase for future
        compute capabilities.  The specific hardware unit may support
        higher cluster sizes that’s not guaranteed to be portable. See
        cudaFuncSetAttribute


    deviceNodeUpdateStatus : int
        Whether the function can be updated on device. 1 means device node
        update is supported, 0 is unsupported or driver is too old to check
        the value.


    reserved1 : int



    reserved : list[int]
        Reserved for future use.


    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cyruntime.cudaFuncAttributes _pvt_val
    cdef cyruntime.cudaFuncAttributes* _pvt_ptr

cdef class cudaMemLocation:
    """
    Specifies a memory location.  To specify a gpu, set type =
    cudaMemLocationTypeDevice and set id = the gpu's device ordinal. To
    specify a cpu NUMA node, set type = cudaMemLocationTypeHostNuma and
    set id = host NUMA node id.

    Attributes
    ----------

    type : cudaMemLocationType
        Specifies the location type, which modifies the meaning of id.


    id : int
        Identifier for cudaMemLocationType::cudaMemLocationTypeDevice,
        cudaMemLocationType::cudaMemLocationTypeHost, or
        cudaMemLocationType::cudaMemLocationTypeHostNuma.


    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cyruntime.cudaMemLocation* _val_ptr
    cdef cyruntime.cudaMemLocation* _pvt_ptr

cdef class cudaMemAccessDesc:
    """
    Memory access descriptor

    Attributes
    ----------

    location : cudaMemLocation
        Location on which the request is to change it's accessibility


    flags : cudaMemAccessFlags
        ::CUmemProt accessibility flags to set on the request


    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cyruntime.cudaMemAccessDesc _pvt_val
    cdef cyruntime.cudaMemAccessDesc* _pvt_ptr

    cdef cudaMemLocation _location


cdef class cudaMemPoolProps:
    """
    Specifies the properties of allocations made from the pool.

    Attributes
    ----------

    allocType : cudaMemAllocationType
        Allocation type. Currently must be specified as
        cudaMemAllocationTypePinned


    handleTypes : cudaMemAllocationHandleType
        Handle types that will be supported by allocations from the pool.


    location : cudaMemLocation
        Location allocations should reside.


    win32SecurityAttributes : Any
        Windows-specific LPSECURITYATTRIBUTES required when
        cudaMemHandleTypeWin32 is specified. This security attribute
        defines the scope of which exported allocations may be tranferred
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
    cdef cyruntime.cudaMemPoolProps _pvt_val
    cdef cyruntime.cudaMemPoolProps* _pvt_ptr

    cdef cudaMemLocation _location


    cdef _HelperInputVoidPtr _cywin32SecurityAttributes


cdef class cudaMemPoolPtrExportData:
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
    cdef cyruntime.cudaMemPoolPtrExportData _pvt_val
    cdef cyruntime.cudaMemPoolPtrExportData* _pvt_ptr

cdef class cudaMemAllocNodeParams:
    """
    Memory allocation node parameters

    Attributes
    ----------

    poolProps : cudaMemPoolProps
        in: location where the allocation should reside (specified in
        ::location). ::handleTypes must be cudaMemHandleTypeNone. IPC is
        not supported. in: array of memory access descriptors. Used to
        describe peer GPU access


    accessDescs : cudaMemAccessDesc
        in: number of memory access descriptors. Must not exceed the number
        of GPUs.


    accessDescCount : size_t
        in: Number of `accessDescs`s


    bytesize : size_t
        in: size in bytes of the requested allocation


    dptr : Any
        out: address of the allocation returned by CUDA


    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cyruntime.cudaMemAllocNodeParams _pvt_val
    cdef cyruntime.cudaMemAllocNodeParams* _pvt_ptr

    cdef cudaMemPoolProps _poolProps


    cdef size_t _accessDescs_length
    cdef cyruntime.cudaMemAccessDesc* _accessDescs


    cdef _HelperInputVoidPtr _cydptr


cdef class cudaMemAllocNodeParamsV2:
    """
    Memory allocation node parameters

    Attributes
    ----------

    poolProps : cudaMemPoolProps
        in: location where the allocation should reside (specified in
        ::location). ::handleTypes must be cudaMemHandleTypeNone. IPC is
        not supported. in: array of memory access descriptors. Used to
        describe peer GPU access


    accessDescs : cudaMemAccessDesc
        in: number of memory access descriptors. Must not exceed the number
        of GPUs.


    accessDescCount : size_t
        in: Number of `accessDescs`s


    bytesize : size_t
        in: size in bytes of the requested allocation


    dptr : Any
        out: address of the allocation returned by CUDA


    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cyruntime.cudaMemAllocNodeParamsV2 _pvt_val
    cdef cyruntime.cudaMemAllocNodeParamsV2* _pvt_ptr

    cdef cudaMemPoolProps _poolProps


    cdef size_t _accessDescs_length
    cdef cyruntime.cudaMemAccessDesc* _accessDescs


    cdef _HelperInputVoidPtr _cydptr


cdef class cudaMemFreeNodeParams:
    """
    Memory free node parameters

    Attributes
    ----------

    dptr : Any
        in: the pointer to free


    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cyruntime.cudaMemFreeNodeParams _pvt_val
    cdef cyruntime.cudaMemFreeNodeParams* _pvt_ptr

    cdef _HelperInputVoidPtr _cydptr


cdef class cudaMemcpyAttributes:
    """
    Attributes specific to copies within a batch. For more details on
    usage see cudaMemcpyBatchAsync.

    Attributes
    ----------

    srcAccessOrder : cudaMemcpySrcAccessOrder
        Source access ordering to be observed for copies with this
        attribute.


    srcLocHint : cudaMemLocation
        Hint location for the source operand. Ignored when the pointers are
        not managed memory or memory allocated outside CUDA.


    dstLocHint : cudaMemLocation
        Hint location for the destination operand. Ignored when the
        pointers are not managed memory or memory allocated outside CUDA.


    flags : unsigned int
        Additional flags for copies with this attribute. See
        cudaMemcpyFlags.


    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cyruntime.cudaMemcpyAttributes _pvt_val
    cdef cyruntime.cudaMemcpyAttributes* _pvt_ptr

    cdef cudaMemLocation _srcLocHint


    cdef cudaMemLocation _dstLocHint


cdef class cudaOffset3D:
    """
    Struct representing offset into a cudaArray_t in elements

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
    cdef cyruntime.cudaOffset3D _pvt_val
    cdef cyruntime.cudaOffset3D* _pvt_ptr

cdef class anon_struct6:
    """
    Attributes
    ----------

    ptr : Any



    rowLength : size_t



    layerHeight : size_t



    locHint : cudaMemLocation



    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cyruntime.cudaMemcpy3DOperand* _pvt_ptr

    cdef _HelperInputVoidPtr _cyptr


    cdef cudaMemLocation _locHint


cdef class anon_struct7:
    """
    Attributes
    ----------

    array : cudaArray_t



    offset : cudaOffset3D



    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cyruntime.cudaMemcpy3DOperand* _pvt_ptr

    cdef cudaArray_t _array


    cdef cudaOffset3D _offset


cdef class anon_union2:
    """
    Attributes
    ----------

    ptr : anon_struct6



    array : anon_struct7



    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cyruntime.cudaMemcpy3DOperand* _pvt_ptr

    cdef anon_struct6 _ptr


    cdef anon_struct7 _array


cdef class cudaMemcpy3DOperand:
    """
    Struct representing an operand for copy with cudaMemcpy3DBatchAsync

    Attributes
    ----------

    type : cudaMemcpy3DOperandType



    op : anon_union2



    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cyruntime.cudaMemcpy3DOperand* _val_ptr
    cdef cyruntime.cudaMemcpy3DOperand* _pvt_ptr

    cdef anon_union2 _op


cdef class cudaMemcpy3DBatchOp:
    """
    Attributes
    ----------

    src : cudaMemcpy3DOperand
        Source memcpy operand.


    dst : cudaMemcpy3DOperand
        Destination memcpy operand.


    extent : cudaExtent
        Extents of the memcpy between src and dst. The width, height and
        depth components must not be 0.


    srcAccessOrder : cudaMemcpySrcAccessOrder
        Source access ordering to be observed for copy from src to dst.


    flags : unsigned int
        Additional flags for copy from src to dst. See cudaMemcpyFlags.


    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cyruntime.cudaMemcpy3DBatchOp _pvt_val
    cdef cyruntime.cudaMemcpy3DBatchOp* _pvt_ptr

    cdef cudaMemcpy3DOperand _src


    cdef cudaMemcpy3DOperand _dst


    cdef cudaExtent _extent


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
    cdef cyruntime.CUuuid_st _pvt_val
    cdef cyruntime.CUuuid_st* _pvt_ptr

cdef class cudaDeviceProp:
    """
    CUDA device properties

    Attributes
    ----------

    name : bytes
        ASCII string identifying device


    uuid : cudaUUID_t
        16-byte unique identifier


    luid : bytes
        8-byte locally unique identifier. Value is undefined on TCC and
        non-Windows platforms


    luidDeviceNodeMask : unsigned int
        LUID device node mask. Value is undefined on TCC and non-Windows
        platforms


    totalGlobalMem : size_t
        Global memory available on device in bytes


    sharedMemPerBlock : size_t
        Shared memory available per block in bytes


    regsPerBlock : int
        32-bit registers available per block


    warpSize : int
        Warp size in threads


    memPitch : size_t
        Maximum pitch in bytes allowed by memory copies


    maxThreadsPerBlock : int
        Maximum number of threads per block


    maxThreadsDim : list[int]
        Maximum size of each dimension of a block


    maxGridSize : list[int]
        Maximum size of each dimension of a grid


    totalConstMem : size_t
        Constant memory available on device in bytes


    major : int
        Major compute capability


    minor : int
        Minor compute capability


    textureAlignment : size_t
        Alignment requirement for textures


    texturePitchAlignment : size_t
        Pitch alignment requirement for texture references bound to pitched
        memory


    multiProcessorCount : int
        Number of multiprocessors on device


    integrated : int
        Device is integrated as opposed to discrete


    canMapHostMemory : int
        Device can map host memory with
        cudaHostAlloc/cudaHostGetDevicePointer


    maxTexture1D : int
        Maximum 1D texture size


    maxTexture1DMipmap : int
        Maximum 1D mipmapped texture size


    maxTexture2D : list[int]
        Maximum 2D texture dimensions


    maxTexture2DMipmap : list[int]
        Maximum 2D mipmapped texture dimensions


    maxTexture2DLinear : list[int]
        Maximum dimensions (width, height, pitch) for 2D textures bound to
        pitched memory


    maxTexture2DGather : list[int]
        Maximum 2D texture dimensions if texture gather operations have to
        be performed


    maxTexture3D : list[int]
        Maximum 3D texture dimensions


    maxTexture3DAlt : list[int]
        Maximum alternate 3D texture dimensions


    maxTextureCubemap : int
        Maximum Cubemap texture dimensions


    maxTexture1DLayered : list[int]
        Maximum 1D layered texture dimensions


    maxTexture2DLayered : list[int]
        Maximum 2D layered texture dimensions


    maxTextureCubemapLayered : list[int]
        Maximum Cubemap layered texture dimensions


    maxSurface1D : int
        Maximum 1D surface size


    maxSurface2D : list[int]
        Maximum 2D surface dimensions


    maxSurface3D : list[int]
        Maximum 3D surface dimensions


    maxSurface1DLayered : list[int]
        Maximum 1D layered surface dimensions


    maxSurface2DLayered : list[int]
        Maximum 2D layered surface dimensions


    maxSurfaceCubemap : int
        Maximum Cubemap surface dimensions


    maxSurfaceCubemapLayered : list[int]
        Maximum Cubemap layered surface dimensions


    surfaceAlignment : size_t
        Alignment requirements for surfaces


    concurrentKernels : int
        Device can possibly execute multiple kernels concurrently


    ECCEnabled : int
        Device has ECC support enabled


    pciBusID : int
        PCI bus ID of the device


    pciDeviceID : int
        PCI device ID of the device


    pciDomainID : int
        PCI domain ID of the device


    tccDriver : int
        1 if device is a Tesla device using TCC driver, 0 otherwise


    asyncEngineCount : int
        Number of asynchronous engines


    unifiedAddressing : int
        Device shares a unified address space with the host


    memoryBusWidth : int
        Global memory bus width in bits


    l2CacheSize : int
        Size of L2 cache in bytes


    persistingL2CacheMaxSize : int
        Device's maximum l2 persisting lines capacity setting in bytes


    maxThreadsPerMultiProcessor : int
        Maximum resident threads per multiprocessor


    streamPrioritiesSupported : int
        Device supports stream priorities


    globalL1CacheSupported : int
        Device supports caching globals in L1


    localL1CacheSupported : int
        Device supports caching locals in L1


    sharedMemPerMultiprocessor : size_t
        Shared memory available per multiprocessor in bytes


    regsPerMultiprocessor : int
        32-bit registers available per multiprocessor


    managedMemory : int
        Device supports allocating managed memory on this system


    isMultiGpuBoard : int
        Device is on a multi-GPU board


    multiGpuBoardGroupID : int
        Unique identifier for a group of devices on the same multi-GPU
        board


    hostNativeAtomicSupported : int
        Link between the device and the host supports native atomic
        operations


    pageableMemoryAccess : int
        Device supports coherently accessing pageable memory without
        calling cudaHostRegister on it


    concurrentManagedAccess : int
        Device can coherently access managed memory concurrently with the
        CPU


    computePreemptionSupported : int
        Device supports Compute Preemption


    canUseHostPointerForRegisteredMem : int
        Device can access host registered memory at the same virtual
        address as the CPU


    cooperativeLaunch : int
        Device supports launching cooperative kernels via
        cudaLaunchCooperativeKernel


    sharedMemPerBlockOptin : size_t
        Per device maximum shared memory per block usable by special opt in


    pageableMemoryAccessUsesHostPageTables : int
        Device accesses pageable memory via the host's page tables


    directManagedMemAccessFromHost : int
        Host can directly access managed memory on the device without
        migration.


    maxBlocksPerMultiProcessor : int
        Maximum number of resident blocks per multiprocessor


    accessPolicyMaxWindowSize : int
        The maximum value of cudaAccessPolicyWindow::num_bytes.


    reservedSharedMemPerBlock : size_t
        Shared memory reserved by CUDA driver per block in bytes


    hostRegisterSupported : int
        Device supports host memory registration via cudaHostRegister.


    sparseCudaArraySupported : int
        1 if the device supports sparse CUDA arrays and sparse CUDA
        mipmapped arrays, 0 otherwise


    hostRegisterReadOnlySupported : int
        Device supports using the cudaHostRegister flag
        cudaHostRegisterReadOnly to register memory that must be mapped as
        read-only to the GPU


    timelineSemaphoreInteropSupported : int
        External timeline semaphore interop is supported on the device


    memoryPoolsSupported : int
        1 if the device supports using the cudaMallocAsync and cudaMemPool
        family of APIs, 0 otherwise


    gpuDirectRDMASupported : int
        1 if the device supports GPUDirect RDMA APIs, 0 otherwise


    gpuDirectRDMAFlushWritesOptions : unsigned int
        Bitmask to be interpreted according to the
        cudaFlushGPUDirectRDMAWritesOptions enum


    gpuDirectRDMAWritesOrdering : int
        See the cudaGPUDirectRDMAWritesOrdering enum for numerical values


    memoryPoolSupportedHandleTypes : unsigned int
        Bitmask of handle types supported with mempool-based IPC


    deferredMappingCudaArraySupported : int
        1 if the device supports deferred mapping CUDA arrays and CUDA
        mipmapped arrays


    ipcEventSupported : int
        Device supports IPC Events.


    clusterLaunch : int
        Indicates device supports cluster launch


    unifiedFunctionPointers : int
        Indicates device supports unified pointers


    deviceNumaConfig : int
        NUMA configuration of a device: value is of type
        cudaDeviceNumaConfig enum


    deviceNumaId : int
        NUMA node ID of the GPU memory


    mpsEnabled : int
        Indicates if contexts created on this device will be shared via MPS


    hostNumaId : int
        NUMA ID of the host node closest to the device or -1 when system
        does not support NUMA


    gpuPciDeviceID : unsigned int
        The combined 16-bit PCI device ID and 16-bit PCI vendor ID


    gpuPciSubsystemID : unsigned int
        The combined 16-bit PCI subsystem ID and 16-bit PCI subsystem
        vendor ID


    hostNumaMultinodeIpcSupported : int
        1 if the device supports HostNuma location IPC between nodes in a
        multi-node system.


    reserved : list[int]
        Reserved for future use


    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cyruntime.cudaDeviceProp _pvt_val
    cdef cyruntime.cudaDeviceProp* _pvt_ptr

    cdef cudaUUID_t _uuid


cdef class cudaIpcEventHandle_st:
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
    cdef cyruntime.cudaIpcEventHandle_st _pvt_val
    cdef cyruntime.cudaIpcEventHandle_st* _pvt_ptr

cdef class cudaIpcMemHandle_st:
    """
    CUDA IPC memory handle

    Attributes
    ----------

    reserved : bytes



    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cyruntime.cudaIpcMemHandle_st _pvt_val
    cdef cyruntime.cudaIpcMemHandle_st* _pvt_ptr

cdef class cudaMemFabricHandle_st:
    """
    Attributes
    ----------

    reserved : bytes



    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cyruntime.cudaMemFabricHandle_st _pvt_val
    cdef cyruntime.cudaMemFabricHandle_st* _pvt_ptr

cdef class anon_struct8:
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
    cdef cyruntime.cudaExternalMemoryHandleDesc* _pvt_ptr

    cdef _HelperInputVoidPtr _cyhandle


    cdef _HelperInputVoidPtr _cyname


cdef class anon_union3:
    """
    Attributes
    ----------

    fd : int



    win32 : anon_struct8



    nvSciBufObject : Any



    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cyruntime.cudaExternalMemoryHandleDesc* _pvt_ptr

    cdef anon_struct8 _win32


    cdef _HelperInputVoidPtr _cynvSciBufObject


cdef class cudaExternalMemoryHandleDesc:
    """
    External memory handle descriptor

    Attributes
    ----------

    type : cudaExternalMemoryHandleType
        Type of the handle


    handle : anon_union3



    size : unsigned long long
        Size of the memory allocation


    flags : unsigned int
        Flags must either be zero or cudaExternalMemoryDedicated


    reserved : list[unsigned int]
        Must be zero


    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cyruntime.cudaExternalMemoryHandleDesc* _val_ptr
    cdef cyruntime.cudaExternalMemoryHandleDesc* _pvt_ptr

    cdef anon_union3 _handle


cdef class cudaExternalMemoryBufferDesc:
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
        Must be zero


    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cyruntime.cudaExternalMemoryBufferDesc _pvt_val
    cdef cyruntime.cudaExternalMemoryBufferDesc* _pvt_ptr

cdef class cudaExternalMemoryMipmappedArrayDesc:
    """
    External memory mipmap descriptor

    Attributes
    ----------

    offset : unsigned long long
        Offset into the memory object where the base level of the mipmap
        chain is.


    formatDesc : cudaChannelFormatDesc
        Format of base level of the mipmap chain


    extent : cudaExtent
        Dimensions of base level of the mipmap chain


    flags : unsigned int
        Flags associated with CUDA mipmapped arrays. See
        cudaMallocMipmappedArray


    numLevels : unsigned int
        Total number of levels in the mipmap chain


    reserved : list[unsigned int]
        Must be zero


    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cyruntime.cudaExternalMemoryMipmappedArrayDesc _pvt_val
    cdef cyruntime.cudaExternalMemoryMipmappedArrayDesc* _pvt_ptr

    cdef cudaChannelFormatDesc _formatDesc


    cdef cudaExtent _extent


cdef class anon_struct9:
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
    cdef cyruntime.cudaExternalSemaphoreHandleDesc* _pvt_ptr

    cdef _HelperInputVoidPtr _cyhandle


    cdef _HelperInputVoidPtr _cyname


cdef class anon_union4:
    """
    Attributes
    ----------

    fd : int



    win32 : anon_struct9



    nvSciSyncObj : Any



    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cyruntime.cudaExternalSemaphoreHandleDesc* _pvt_ptr

    cdef anon_struct9 _win32


    cdef _HelperInputVoidPtr _cynvSciSyncObj


cdef class cudaExternalSemaphoreHandleDesc:
    """
    External semaphore handle descriptor

    Attributes
    ----------

    type : cudaExternalSemaphoreHandleType
        Type of the handle


    handle : anon_union4



    flags : unsigned int
        Flags reserved for the future. Must be zero.


    reserved : list[unsigned int]
        Must be zero


    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cyruntime.cudaExternalSemaphoreHandleDesc* _val_ptr
    cdef cyruntime.cudaExternalSemaphoreHandleDesc* _pvt_ptr

    cdef anon_union4 _handle


cdef class anon_struct10:
    """
    Attributes
    ----------

    value : unsigned long long



    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cyruntime.cudaExternalSemaphoreSignalParams* _pvt_ptr

cdef class anon_union5:
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
    cdef cyruntime.cudaExternalSemaphoreSignalParams* _pvt_ptr

    cdef _HelperInputVoidPtr _cyfence


cdef class anon_struct11:
    """
    Attributes
    ----------

    key : unsigned long long



    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cyruntime.cudaExternalSemaphoreSignalParams* _pvt_ptr

cdef class anon_struct12:
    """
    Attributes
    ----------

    fence : anon_struct10



    nvSciSync : anon_union5



    keyedMutex : anon_struct11



    reserved : list[unsigned int]



    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cyruntime.cudaExternalSemaphoreSignalParams* _pvt_ptr

    cdef anon_struct10 _fence


    cdef anon_union5 _nvSciSync


    cdef anon_struct11 _keyedMutex


cdef class cudaExternalSemaphoreSignalParams:
    """
    External semaphore signal parameters, compatible with driver type

    Attributes
    ----------

    params : anon_struct12



    flags : unsigned int
        Only when cudaExternalSemaphoreSignalParams is used to signal a
        cudaExternalSemaphore_t of type
        cudaExternalSemaphoreHandleTypeNvSciSync, the valid flag is
        cudaExternalSemaphoreSignalSkipNvSciBufMemSync: which indicates
        that while signaling the cudaExternalSemaphore_t, no memory
        synchronization operations should be performed for any external
        memory object imported as cudaExternalMemoryHandleTypeNvSciBuf. For
        all other types of cudaExternalSemaphore_t, flags must be zero.


    reserved : list[unsigned int]



    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cyruntime.cudaExternalSemaphoreSignalParams _pvt_val
    cdef cyruntime.cudaExternalSemaphoreSignalParams* _pvt_ptr

    cdef anon_struct12 _params


cdef class anon_struct13:
    """
    Attributes
    ----------

    value : unsigned long long



    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cyruntime.cudaExternalSemaphoreWaitParams* _pvt_ptr

cdef class anon_union6:
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
    cdef cyruntime.cudaExternalSemaphoreWaitParams* _pvt_ptr

    cdef _HelperInputVoidPtr _cyfence


cdef class anon_struct14:
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
    cdef cyruntime.cudaExternalSemaphoreWaitParams* _pvt_ptr

cdef class anon_struct15:
    """
    Attributes
    ----------

    fence : anon_struct13



    nvSciSync : anon_union6



    keyedMutex : anon_struct14



    reserved : list[unsigned int]



    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cyruntime.cudaExternalSemaphoreWaitParams* _pvt_ptr

    cdef anon_struct13 _fence


    cdef anon_union6 _nvSciSync


    cdef anon_struct14 _keyedMutex


cdef class cudaExternalSemaphoreWaitParams:
    """
    External semaphore wait parameters, compatible with driver type

    Attributes
    ----------

    params : anon_struct15



    flags : unsigned int
        Only when cudaExternalSemaphoreSignalParams is used to signal a
        cudaExternalSemaphore_t of type
        cudaExternalSemaphoreHandleTypeNvSciSync, the valid flag is
        cudaExternalSemaphoreSignalSkipNvSciBufMemSync: which indicates
        that while waiting for the cudaExternalSemaphore_t, no memory
        synchronization operations should be performed for any external
        memory object imported as cudaExternalMemoryHandleTypeNvSciBuf. For
        all other types of cudaExternalSemaphore_t, flags must be zero.


    reserved : list[unsigned int]



    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cyruntime.cudaExternalSemaphoreWaitParams _pvt_val
    cdef cyruntime.cudaExternalSemaphoreWaitParams* _pvt_ptr

    cdef anon_struct15 _params


cdef class cudaDevSmResource:
    """
    Data for SM-related resources All parameters in this structure are
    OUTPUT only. Do not write to any of the fields in this structure.

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
        The flags set on this SM resource. For available flags see
        cudaDevSmResourceGroup_flags.


    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cyruntime.cudaDevSmResource _pvt_val
    cdef cyruntime.cudaDevSmResource* _pvt_ptr

cdef class cudaDevWorkqueueConfigResource:
    """
    Data for workqueue configuration related resources

    Attributes
    ----------

    device : int
        The device on which the workqueue resources are available


    wqConcurrencyLimit : unsigned int
        The expected maximum number of concurrent stream-ordered workloads


    sharingScope : cudaDevWorkqueueConfigScope
        The sharing scope for the workqueue resources


    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cyruntime.cudaDevWorkqueueConfigResource _pvt_val
    cdef cyruntime.cudaDevWorkqueueConfigResource* _pvt_ptr

cdef class cudaDevWorkqueueResource:
    """
    Handle to a pre-existing workqueue related resource

    Attributes
    ----------

    reserved : bytes
        Reserved for future use


    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cyruntime.cudaDevWorkqueueResource _pvt_val
    cdef cyruntime.cudaDevWorkqueueResource* _pvt_ptr

cdef class cudaDevSmResourceGroupParams_st:
    """
    Input data for splitting SMs

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
        Combination of `cudaDevSmResourceGroup_flags` values to indicate
        this this group is created.


    reserved : list[unsigned int]
        Reserved for future use - ensure this is zero initialized.


    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cyruntime.cudaDevSmResourceGroupParams_st _pvt_val
    cdef cyruntime.cudaDevSmResourceGroupParams_st* _pvt_ptr

cdef class cudaDevResource_st:
    """
    A tagged union describing different resources identified by the
    type field. This structure should not be directly modified outside
    of the API that created it. struct enumcudaDevResourceTypetype;
    union structcudaDevSmResourcesm;
    structcudaDevWorkqueueConfigResourcewqConfig;
    structcudaDevWorkqueueResourcewq; ; ;  - If `typename` is
    `cudaDevResourceTypeInvalid`, this resoure is not valid and cannot
    be further accessed.    - If `typename` is `cudaDevResourceTypeSm`,
    the cudaDevSmResource structure `sm` is filled in. For example,
    `sm.smCount` will reflect the amount of streaming multiprocessors
    available in this resource.    - If `typename` is
    `cudaDevResourceTypeWorkqueueConfig`, the
    cudaDevWorkqueueConfigResource structure `wqConfig` is filled in.
    - If `typename` is `cudaDevResourceTypeWorkqueue`, the
    cudaDevWorkqueueResource structure `wq` is filled in.

    Attributes
    ----------

    type : cudaDevResourceType
        Type of resource, dictates which union field was last set


    _internal_padding : bytes



    sm : cudaDevSmResource
        Resource corresponding to cudaDevResourceTypeSm `typename`.


    wqConfig : cudaDevWorkqueueConfigResource
        Resource corresponding to cudaDevResourceTypeWorkqueueConfig
        `typename`.


    wq : cudaDevWorkqueueResource
        Resource corresponding to cudaDevResourceTypeWorkqueue `typename`.


    _oversize : bytes



    nextResource : cudaDevResource_st



    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cyruntime.cudaDevResource_st* _val_ptr
    cdef cyruntime.cudaDevResource_st* _pvt_ptr

    cdef cudaDevSmResource _sm


    cdef cudaDevWorkqueueConfigResource _wqConfig


    cdef cudaDevWorkqueueResource _wq


    cdef size_t _nextResource_length
    cdef cyruntime.cudaDevResource_st* _nextResource


cdef class cudalibraryHostUniversalFunctionAndDataTable:
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
    cdef cyruntime.cudalibraryHostUniversalFunctionAndDataTable _pvt_val
    cdef cyruntime.cudalibraryHostUniversalFunctionAndDataTable* _pvt_ptr

    cdef _HelperInputVoidPtr _cyfunctionTable


    cdef _HelperInputVoidPtr _cydataTable


cdef class cudaKernelNodeParams:
    """
    CUDA GPU kernel node parameters

    Attributes
    ----------

    func : Any
        Kernel to launch


    gridDim : dim3
        Grid dimensions


    blockDim : dim3
        Block dimensions


    sharedMemBytes : unsigned int
        Dynamic shared-memory size per thread block in bytes


    kernelParams : Any
        Array of pointers to individual kernel arguments


    extra : Any
        Pointer to kernel arguments in the "extra" format


    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cyruntime.cudaKernelNodeParams _pvt_val
    cdef cyruntime.cudaKernelNodeParams* _pvt_ptr

    cdef _HelperInputVoidPtr _cyfunc


    cdef dim3 _gridDim


    cdef dim3 _blockDim


    cdef _HelperKernelParams _cykernelParams


cdef class cudaKernelNodeParamsV2:
    """
    CUDA GPU kernel node parameters

    Attributes
    ----------

    func : Any
        functionType = cudaKernelFucntionTypeDevice


    kern : cudaKernel_t
        functionType = cudaKernelFucntionTypeKernel


    cuFunc : cudaFunction_t
        functionType = cudaKernelFucntionTypeFunction


    gridDim : dim3
        Grid dimensions


    blockDim : dim3
        Block dimensions


    sharedMemBytes : unsigned int
        Dynamic shared-memory size per thread block in bytes


    kernelParams : Any
        Array of pointers to individual kernel arguments


    extra : Any
        Pointer to kernel arguments in the "extra" format


    ctx : cudaExecutionContext_t
        Context in which to run the kernel. If NULL will try to use the
        current context.


    functionType : cudaKernelFunctionType
        Type of handle passed in the func/kern/cuFunc union above


    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cyruntime.cudaKernelNodeParamsV2* _val_ptr
    cdef cyruntime.cudaKernelNodeParamsV2* _pvt_ptr

    cdef _HelperInputVoidPtr _cyfunc


    cdef cudaKernel_t _kern


    cdef cudaFunction_t _cuFunc


    cdef dim3 _gridDim


    cdef dim3 _blockDim


    cdef _HelperKernelParams _cykernelParams


    cdef cudaExecutionContext_t _ctx


cdef class cudaExternalSemaphoreSignalNodeParams:
    """
    External semaphore signal node parameters

    Attributes
    ----------

    extSemArray : cudaExternalSemaphore_t
        Array of external semaphore handles.


    paramsArray : cudaExternalSemaphoreSignalParams
        Array of external semaphore signal parameters.


    numExtSems : unsigned int
        Number of handles and parameters supplied in extSemArray and
        paramsArray.


    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cyruntime.cudaExternalSemaphoreSignalNodeParams _pvt_val
    cdef cyruntime.cudaExternalSemaphoreSignalNodeParams* _pvt_ptr

    cdef size_t _extSemArray_length
    cdef cyruntime.cudaExternalSemaphore_t* _extSemArray


    cdef size_t _paramsArray_length
    cdef cyruntime.cudaExternalSemaphoreSignalParams* _paramsArray


cdef class cudaExternalSemaphoreSignalNodeParamsV2:
    """
    External semaphore signal node parameters

    Attributes
    ----------

    extSemArray : cudaExternalSemaphore_t
        Array of external semaphore handles.


    paramsArray : cudaExternalSemaphoreSignalParams
        Array of external semaphore signal parameters.


    numExtSems : unsigned int
        Number of handles and parameters supplied in extSemArray and
        paramsArray.


    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cyruntime.cudaExternalSemaphoreSignalNodeParamsV2 _pvt_val
    cdef cyruntime.cudaExternalSemaphoreSignalNodeParamsV2* _pvt_ptr

    cdef size_t _extSemArray_length
    cdef cyruntime.cudaExternalSemaphore_t* _extSemArray


    cdef size_t _paramsArray_length
    cdef cyruntime.cudaExternalSemaphoreSignalParams* _paramsArray


cdef class cudaExternalSemaphoreWaitNodeParams:
    """
    External semaphore wait node parameters

    Attributes
    ----------

    extSemArray : cudaExternalSemaphore_t
        Array of external semaphore handles.


    paramsArray : cudaExternalSemaphoreWaitParams
        Array of external semaphore wait parameters.


    numExtSems : unsigned int
        Number of handles and parameters supplied in extSemArray and
        paramsArray.


    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cyruntime.cudaExternalSemaphoreWaitNodeParams _pvt_val
    cdef cyruntime.cudaExternalSemaphoreWaitNodeParams* _pvt_ptr

    cdef size_t _extSemArray_length
    cdef cyruntime.cudaExternalSemaphore_t* _extSemArray


    cdef size_t _paramsArray_length
    cdef cyruntime.cudaExternalSemaphoreWaitParams* _paramsArray


cdef class cudaExternalSemaphoreWaitNodeParamsV2:
    """
    External semaphore wait node parameters

    Attributes
    ----------

    extSemArray : cudaExternalSemaphore_t
        Array of external semaphore handles.


    paramsArray : cudaExternalSemaphoreWaitParams
        Array of external semaphore wait parameters.


    numExtSems : unsigned int
        Number of handles and parameters supplied in extSemArray and
        paramsArray.


    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cyruntime.cudaExternalSemaphoreWaitNodeParamsV2 _pvt_val
    cdef cyruntime.cudaExternalSemaphoreWaitNodeParamsV2* _pvt_ptr

    cdef size_t _extSemArray_length
    cdef cyruntime.cudaExternalSemaphore_t* _extSemArray


    cdef size_t _paramsArray_length
    cdef cyruntime.cudaExternalSemaphoreWaitParams* _paramsArray


cdef class cudaConditionalNodeParams:
    """
    CUDA conditional node parameters

    Attributes
    ----------

    handle : cudaGraphConditionalHandle
        Conditional node handle. Handles must be created in advance of
        creating the node using cudaGraphConditionalHandleCreate.


    type : cudaGraphConditionalNodeType
        Type of conditional node.


    size : unsigned int
        Size of graph output array. Allowed values are 1 for
        cudaGraphCondTypeWhile, 1 or 2 for cudaGraphCondTypeIf, or any
        value greater than zero for cudaGraphCondTypeSwitch.


    phGraph_out : cudaGraph_t
        CUDA-owned array populated with conditional node child graphs
        during creation of the node. Valid for the lifetime of the
        conditional node. The contents of the graph(s) are subject to the
        following constraints:   - Allowed node types are kernel nodes,
        empty nodes, child graphs, memsets, memcopies, and conditionals.
        This applies recursively to child graphs and conditional bodies.
        - All kernels, including kernels in nested conditionals or child
        graphs at any level, must belong to the same CUDA context.
        These graphs may be populated using graph node creation APIs or
        cudaStreamBeginCaptureToGraph. cudaGraphCondTypeIf: phGraph_out[0]
        is executed when the condition is non-zero. If `size` == 2,
        phGraph_out[1] will be executed when the condition is zero.
        cudaGraphCondTypeWhile: phGraph_out[0] is executed as long as the
        condition is non-zero. cudaGraphCondTypeSwitch: phGraph_out[n] is
        executed when the condition is equal to n. If the condition >=
        `size`, no body graph is executed.


    ctx : cudaExecutionContext_t
        CUDA Execution Context


    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cyruntime.cudaConditionalNodeParams _pvt_val
    cdef cyruntime.cudaConditionalNodeParams* _pvt_ptr

    cdef cudaGraphConditionalHandle _handle


    cdef size_t _phGraph_out_length
    cdef cyruntime.cudaGraph_t* _phGraph_out


    cdef cudaExecutionContext_t _ctx


cdef class cudaChildGraphNodeParams:
    """
    Child graph node parameters

    Attributes
    ----------

    graph : cudaGraph_t
        The child graph to clone into the node for node creation, or a
        handle to the graph owned by the node for node query. The graph
        must not contain conditional nodes. Graphs containing memory
        allocation or memory free nodes must set the ownership to be moved
        to the parent.


    ownership : cudaGraphChildGraphNodeOwnership
        The ownership relationship of the child graph node.


    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cyruntime.cudaChildGraphNodeParams _pvt_val
    cdef cyruntime.cudaChildGraphNodeParams* _pvt_ptr

    cdef cudaGraph_t _graph


cdef class cudaEventRecordNodeParams:
    """
    Event record node parameters

    Attributes
    ----------

    event : cudaEvent_t
        The event to record when the node executes


    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cyruntime.cudaEventRecordNodeParams _pvt_val
    cdef cyruntime.cudaEventRecordNodeParams* _pvt_ptr

    cdef cudaEvent_t _event


cdef class cudaEventWaitNodeParams:
    """
    Event wait node parameters

    Attributes
    ----------

    event : cudaEvent_t
        The event to wait on from the node


    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cyruntime.cudaEventWaitNodeParams _pvt_val
    cdef cyruntime.cudaEventWaitNodeParams* _pvt_ptr

    cdef cudaEvent_t _event


cdef class cudaGraphNodeParams:
    """
    Graph node parameters. See cudaGraphAddNode.

    Attributes
    ----------

    type : cudaGraphNodeType
        Type of the node


    reserved0 : list[int]
        Reserved. Must be zero.


    reserved1 : list[long long]
        Padding. Unused bytes must be zero.


    kernel : cudaKernelNodeParamsV2
        Kernel node parameters.


    memcpy : cudaMemcpyNodeParams
        Memcpy node parameters.


    memset : cudaMemsetParamsV2
        Memset node parameters.


    host : cudaHostNodeParamsV2
        Host node parameters.


    graph : cudaChildGraphNodeParams
        Child graph node parameters.


    eventWait : cudaEventWaitNodeParams
        Event wait node parameters.


    eventRecord : cudaEventRecordNodeParams
        Event record node parameters.


    extSemSignal : cudaExternalSemaphoreSignalNodeParamsV2
        External semaphore signal node parameters.


    extSemWait : cudaExternalSemaphoreWaitNodeParamsV2
        External semaphore wait node parameters.


    alloc : cudaMemAllocNodeParamsV2
        Memory allocation node parameters.


    free : cudaMemFreeNodeParams
        Memory free node parameters.


    conditional : cudaConditionalNodeParams
        Conditional node parameters.


    reserved2 : long long
        Reserved bytes. Must be zero.


    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cyruntime.cudaGraphNodeParams* _val_ptr
    cdef cyruntime.cudaGraphNodeParams* _pvt_ptr

    cdef cudaKernelNodeParamsV2 _kernel


    cdef cudaMemcpyNodeParams _memcpy


    cdef cudaMemsetParamsV2 _memset


    cdef cudaHostNodeParamsV2 _host


    cdef cudaChildGraphNodeParams _graph


    cdef cudaEventWaitNodeParams _eventWait


    cdef cudaEventRecordNodeParams _eventRecord


    cdef cudaExternalSemaphoreSignalNodeParamsV2 _extSemSignal


    cdef cudaExternalSemaphoreWaitNodeParamsV2 _extSemWait


    cdef cudaMemAllocNodeParamsV2 _alloc


    cdef cudaMemFreeNodeParams _free


    cdef cudaConditionalNodeParams _conditional


cdef class cudaGraphEdgeData_st:
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
        cudaGraphKernelNodePortDefault,
        cudaGraphKernelNodePortProgrammatic, or
        cudaGraphKernelNodePortLaunchCompletion.


    to_port : bytes
        This indicates what portion of the downstream node is dependent on
        the upstream node or portion thereof (indicated by `from_port`).
        The meaning is specific to the node type. A value of 0 in all cases
        means the entirety of the downstream node is dependent on the
        upstream work.   Currently no node types define non-zero ports.
        Accordingly, this field must be set to zero.


    type : bytes
        This should be populated with a value from cudaGraphDependencyType.
        (It is typed as char due to compiler-specific layout of bitfields.)
        See cudaGraphDependencyType.


    reserved : bytes
        These bytes are unused and must be zeroed. This ensures
        compatibility if additional fields are added in the future.


    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cyruntime.cudaGraphEdgeData_st _pvt_val
    cdef cyruntime.cudaGraphEdgeData_st* _pvt_ptr

cdef class cudaGraphInstantiateParams_st:
    """
    Graph instantiation parameters

    Attributes
    ----------

    flags : unsigned long long
        Instantiation flags


    uploadStream : cudaStream_t
        Upload stream


    errNode_out : cudaGraphNode_t
        The node which caused instantiation to fail, if any


    result_out : cudaGraphInstantiateResult
        Whether instantiation was successful. If it failed, the reason why


    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cyruntime.cudaGraphInstantiateParams_st _pvt_val
    cdef cyruntime.cudaGraphInstantiateParams_st* _pvt_ptr

    cdef cudaStream_t _uploadStream


    cdef cudaGraphNode_t _errNode_out


cdef class cudaGraphExecUpdateResultInfo_st:
    """
    Result information returned by cudaGraphExecUpdate

    Attributes
    ----------

    result : cudaGraphExecUpdateResult
        Gives more specific detail when a cuda graph update fails.


    errorNode : cudaGraphNode_t
        The "to node" of the error edge when the topologies do not match.
        The error node when the error is associated with a specific node.
        NULL when the error is generic.


    errorFromNode : cudaGraphNode_t
        The from node of error edge when the topologies do not match.
        Otherwise NULL.


    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cyruntime.cudaGraphExecUpdateResultInfo_st _pvt_val
    cdef cyruntime.cudaGraphExecUpdateResultInfo_st* _pvt_ptr

    cdef cudaGraphNode_t _errorNode


    cdef cudaGraphNode_t _errorFromNode


cdef class anon_struct16:
    """
    Attributes
    ----------

    pValue : Any



    offset : size_t



    size : size_t



    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cyruntime.cudaGraphKernelNodeUpdate* _pvt_ptr

    cdef _HelperInputVoidPtr _cypValue


cdef class anon_union10:
    """
    Attributes
    ----------

    gridDim : dim3



    param : anon_struct16



    isEnabled : unsigned int



    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cyruntime.cudaGraphKernelNodeUpdate* _pvt_ptr

    cdef dim3 _gridDim


    cdef anon_struct16 _param


cdef class cudaGraphKernelNodeUpdate:
    """
    Struct to specify a single node update to pass as part of a larger
    array to ::cudaGraphKernelNodeUpdatesApply

    Attributes
    ----------

    node : cudaGraphDeviceNode_t
        Node to update


    field : cudaGraphKernelNodeField
        Which type of update to apply. Determines how updateData is
        interpreted


    updateData : anon_union10
        Update data to apply. Which field is used depends on field's value


    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cyruntime.cudaGraphKernelNodeUpdate* _val_ptr
    cdef cyruntime.cudaGraphKernelNodeUpdate* _pvt_ptr

    cdef cudaGraphDeviceNode_t _node


    cdef anon_union10 _updateData


cdef class cudaLaunchMemSyncDomainMap_st:
    """
    Memory Synchronization Domain map  See cudaLaunchMemSyncDomain.  By
    default, kernels are launched in domain 0. Kernel launched with
    cudaLaunchMemSyncDomainRemote will have a different domain ID. User
    may also alter the domain ID with cudaLaunchMemSyncDomainMap for a
    specific stream / graph node / kernel launch. See
    cudaLaunchAttributeMemSyncDomainMap.  Domain ID range is available
    through cudaDevAttrMemSyncDomainCount.

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
    cdef cyruntime.cudaLaunchMemSyncDomainMap_st _pvt_val
    cdef cyruntime.cudaLaunchMemSyncDomainMap_st* _pvt_ptr

cdef class anon_struct17:
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
    cdef cyruntime.cudaLaunchAttributeValue* _pvt_ptr

cdef class anon_struct18:
    """
    Attributes
    ----------

    event : cudaEvent_t



    flags : int



    triggerAtBlockStart : int



    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cyruntime.cudaLaunchAttributeValue* _pvt_ptr

    cdef cudaEvent_t _event


cdef class anon_struct19:
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
    cdef cyruntime.cudaLaunchAttributeValue* _pvt_ptr

cdef class anon_struct20:
    """
    Attributes
    ----------

    event : cudaEvent_t



    flags : int



    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cyruntime.cudaLaunchAttributeValue* _pvt_ptr

    cdef cudaEvent_t _event


cdef class anon_struct21:
    """
    Attributes
    ----------

    deviceUpdatable : int



    devNode : cudaGraphDeviceNode_t



    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cyruntime.cudaLaunchAttributeValue* _pvt_ptr

    cdef cudaGraphDeviceNode_t _devNode


cdef class cudaLaunchAttributeValue:
    """
    Launch attributes union; used as value field of cudaLaunchAttribute

    Attributes
    ----------

    pad : bytes



    accessPolicyWindow : cudaAccessPolicyWindow
        Value of launch attribute cudaLaunchAttributeAccessPolicyWindow.


    cooperative : int
        Value of launch attribute cudaLaunchAttributeCooperative. Nonzero
        indicates a cooperative kernel (see cudaLaunchCooperativeKernel).


    syncPolicy : cudaSynchronizationPolicy
        Value of launch attribute cudaLaunchAttributeSynchronizationPolicy.
        cudaSynchronizationPolicy for work queued up in this stream.


    clusterDim : anon_struct17
        Value of launch attribute cudaLaunchAttributeClusterDimension that
        represents the desired cluster dimensions for the kernel. Opaque
        type with the following fields: - `x` - The X dimension of the
        cluster, in blocks. Must be a divisor of the grid X dimension.    -
        `y` - The Y dimension of the cluster, in blocks. Must be a divisor
        of the grid Y dimension.    - `z` - The Z dimension of the cluster,
        in blocks. Must be a divisor of the grid Z dimension.


    clusterSchedulingPolicyPreference : cudaClusterSchedulingPolicy
        Value of launch attribute
        cudaLaunchAttributeClusterSchedulingPolicyPreference. Cluster
        scheduling policy preference for the kernel.


    programmaticStreamSerializationAllowed : int
        Value of launch attribute
        cudaLaunchAttributeProgrammaticStreamSerialization.


    programmaticEvent : anon_struct18
        Value of launch attribute cudaLaunchAttributeProgrammaticEvent with
        the following fields: - `cudaEvent_t` event - Event to fire when
        all blocks trigger it.    - `int` flags; - Event record flags, see
        cudaEventRecordWithFlags. Does not accept cudaEventRecordExternal.
        - `int` triggerAtBlockStart - If this is set to non-0, each block
        launch will automatically trigger the event.


    priority : int
        Value of launch attribute cudaLaunchAttributePriority. Execution
        priority of the kernel.


    memSyncDomainMap : cudaLaunchMemSyncDomainMap
        Value of launch attribute cudaLaunchAttributeMemSyncDomainMap. See
        cudaLaunchMemSyncDomainMap.


    memSyncDomain : cudaLaunchMemSyncDomain
        Value of launch attribute cudaLaunchAttributeMemSyncDomain. See
        cudaLaunchMemSyncDomain.


    preferredClusterDim : anon_struct19
        Value of launch attribute
        cudaLaunchAttributePreferredClusterDimension that represents the
        desired preferred cluster dimensions for the kernel. Opaque type
        with the following fields: - `x` - The X dimension of the preferred
        cluster, in blocks. Must be a divisor of the grid X dimension, and
        must be a multiple of the `x` field of
        cudaLaunchAttributeValue::clusterDim.    - `y` - The Y dimension of
        the preferred cluster, in blocks. Must be a divisor of the grid Y
        dimension, and must be a multiple of the `y` field of
        cudaLaunchAttributeValue::clusterDim.    - `z` - The Z dimension of
        the preferred cluster, in blocks. Must be equal to the `z` field of
        cudaLaunchAttributeValue::clusterDim.


    launchCompletionEvent : anon_struct20
        Value of launch attribute cudaLaunchAttributeLaunchCompletionEvent
        with the following fields: - `cudaEvent_t` event - Event to fire
        when the last block launches.    - `int` flags - Event record
        flags, see cudaEventRecordWithFlags. Does not accept
        cudaEventRecordExternal.


    deviceUpdatableKernelNode : anon_struct21
        Value of launch attribute
        cudaLaunchAttributeDeviceUpdatableKernelNode with the following
        fields: - `int` deviceUpdatable - Whether or not the resulting
        kernel node should be device-updatable.    -
        `cudaGraphDeviceNode_t` devNode - Returns a handle to pass to the
        various device-side update functions.


    sharedMemCarveout : unsigned int
        Value of launch attribute
        cudaLaunchAttributePreferredSharedMemoryCarveout.


    nvlinkUtilCentricScheduling : unsigned int
        Value of launch attribute
        cudaLaunchAttributeNvlinkUtilCentricScheduling.


    portableClusterSizeMode : cudaLaunchAttributePortableClusterMode
        Value of launch attribute
        cudaLaunchAttributePortableClusterSizeMode


    sharedMemoryMode : cudaSharedMemoryMode
        Value of launch attribute cudaLaunchAttributeSharedMemoryMode. See
        cudaSharedMemoryMode for acceptable values.


    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cyruntime.cudaLaunchAttributeValue _pvt_val
    cdef cyruntime.cudaLaunchAttributeValue* _pvt_ptr

    cdef cudaAccessPolicyWindow _accessPolicyWindow


    cdef anon_struct17 _clusterDim


    cdef anon_struct18 _programmaticEvent


    cdef cudaLaunchMemSyncDomainMap _memSyncDomainMap


    cdef anon_struct19 _preferredClusterDim


    cdef anon_struct20 _launchCompletionEvent


    cdef anon_struct21 _deviceUpdatableKernelNode


cdef class cudaLaunchAttribute_st:
    """
    Launch attribute

    Attributes
    ----------

    id : cudaLaunchAttributeID
        Attribute to set


    val : cudaLaunchAttributeValue
        Value of the attribute


    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cyruntime.cudaLaunchAttribute_st _pvt_val
    cdef cyruntime.cudaLaunchAttribute_st* _pvt_ptr

    cdef cudaLaunchAttributeValue _val


cdef class anon_struct22:
    """
    Attributes
    ----------

    bytesOverBudget : unsigned long long



    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cyruntime.cudaAsyncNotificationInfo* _pvt_ptr

cdef class anon_union11:
    """
    Attributes
    ----------

    overBudget : anon_struct22



    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cyruntime.cudaAsyncNotificationInfo* _pvt_ptr

    cdef anon_struct22 _overBudget


cdef class cudaAsyncNotificationInfo:
    """
    Information describing an async notification event

    Attributes
    ----------

    type : cudaAsyncNotificationType
        The type of notification being sent


    info : anon_union11
        Information about the notification. `typename` must be checked in
        order to interpret this field.


    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cyruntime.cudaAsyncNotificationInfo* _val_ptr
    cdef cyruntime.cudaAsyncNotificationInfo* _pvt_ptr

    cdef anon_union11 _info


cdef class cudaTextureDesc:
    """
    CUDA texture descriptor

    Attributes
    ----------

    addressMode : list[cudaTextureAddressMode]
        Texture address mode for up to 3 dimensions


    filterMode : cudaTextureFilterMode
        Texture filter mode


    readMode : cudaTextureReadMode
        Texture read mode


    sRGB : int
        Perform sRGB->linear conversion during texture read


    borderColor : list[float]
        Texture Border Color


    normalizedCoords : int
        Indicates whether texture reads are normalized or not


    maxAnisotropy : unsigned int
        Limit to the anisotropy ratio


    mipmapFilterMode : cudaTextureFilterMode
        Mipmap filter mode


    mipmapLevelBias : float
        Offset applied to the supplied mipmap level


    minMipmapLevelClamp : float
        Lower end of the mipmap level range to clamp access to


    maxMipmapLevelClamp : float
        Upper end of the mipmap level range to clamp access to


    disableTrilinearOptimization : int
        Disable any trilinear filtering optimizations.


    seamlessCubemap : int
        Enable seamless cube map filtering.


    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cyruntime.cudaTextureDesc _pvt_val
    cdef cyruntime.cudaTextureDesc* _pvt_ptr

cdef class cudaGraphRecaptureCallbackData:
    """
    Struct of user callback data that is invoked when node parameter
    mismatches are detected while recapturing to an existing graph

    Attributes
    ----------

    callbackFunc : cudaGraphRecaptureCallback_t
        Callback function that will be invoked


    userData : Any
        Generic pointer that is passed to the callback function


    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cyruntime.cudaGraphRecaptureCallbackData _pvt_val
    cdef cyruntime.cudaGraphRecaptureCallbackData* _pvt_ptr

    cdef cudaGraphRecaptureCallback_t _callbackFunc


    cdef _HelperInputVoidPtr _cyuserData


cdef class cudaEglPlaneDesc_st:
    """
    CUDA EGL Plane Descriptor - structure defining each plane of a CUDA
    EGLFrame

    Attributes
    ----------

    width : unsigned int
        Width of plane


    height : unsigned int
        Height of plane


    depth : unsigned int
        Depth of plane


    pitch : unsigned int
        Pitch of plane


    numChannels : unsigned int
        Number of channels for the plane


    channelDesc : cudaChannelFormatDesc
        Channel Format Descriptor


    reserved : list[unsigned int]
        Reserved for future use


    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cyruntime.cudaEglPlaneDesc_st _pvt_val
    cdef cyruntime.cudaEglPlaneDesc_st* _pvt_ptr

    cdef cudaChannelFormatDesc _channelDesc


cdef class anon_union12:
    """
    Attributes
    ----------

    pArray : list[cudaArray_t]



    pPitch : list[cudaPitchedPtr]



    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cyruntime.cudaEglFrame_st* _pvt_ptr

cdef class cudaEglFrame_st:
    """
    CUDA EGLFrame Descriptor - structure defining one frame of EGL.
    Each frame may contain one or more planes depending on whether the
    surface is Multiplanar or not. Each plane of EGLFrame is
    represented by cudaEglPlaneDesc which is defined as:
    typedefstructcudaEglPlaneDesc_st unsignedintwidth;
    unsignedintheight; unsignedintdepth; unsignedintpitch;
    unsignedintnumChannels; structcudaChannelFormatDescchannelDesc;
    unsignedintreserved[4]; cudaEglPlaneDesc;

    Attributes
    ----------

    frame : anon_union12



    planeDesc : list[cudaEglPlaneDesc]
        CUDA EGL Plane Descriptor cudaEglPlaneDesc


    planeCount : unsigned int
        Number of planes


    frameType : cudaEglFrameType
        Array or Pitch


    eglColorFormat : cudaEglColorFormat
        CUDA EGL Color Format


    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    cdef cyruntime.cudaEglFrame_st* _val_ptr
    cdef cyruntime.cudaEglFrame_st* _pvt_ptr

    cdef anon_union12 _frame


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

cdef class cudaUUID_t(CUuuid_st):
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

cdef class cudaIpcEventHandle_t(cudaIpcEventHandle_st):
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

cdef class cudaIpcMemHandle_t(cudaIpcMemHandle_st):
    """
    CUDA IPC memory handle

    Attributes
    ----------

    reserved : bytes



    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    pass

cdef class cudaMemFabricHandle_t(cudaMemFabricHandle_st):
    """
    Attributes
    ----------

    reserved : bytes



    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    pass

cdef class cudaDevSmResourceGroupParams(cudaDevSmResourceGroupParams_st):
    """
    Input data for splitting SMs

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
        Combination of `cudaDevSmResourceGroup_flags` values to indicate
        this this group is created.


    reserved : list[unsigned int]
        Reserved for future use - ensure this is zero initialized.


    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    pass

cdef class cudaDevResource(cudaDevResource_st):
    """
    A tagged union describing different resources identified by the
    type field. This structure should not be directly modified outside
    of the API that created it. struct enumcudaDevResourceTypetype;
    union structcudaDevSmResourcesm;
    structcudaDevWorkqueueConfigResourcewqConfig;
    structcudaDevWorkqueueResourcewq; ; ;  - If `typename` is
    `cudaDevResourceTypeInvalid`, this resoure is not valid and cannot
    be further accessed.    - If `typename` is `cudaDevResourceTypeSm`,
    the cudaDevSmResource structure `sm` is filled in. For example,
    `sm.smCount` will reflect the amount of streaming multiprocessors
    available in this resource.    - If `typename` is
    `cudaDevResourceTypeWorkqueueConfig`, the
    cudaDevWorkqueueConfigResource structure `wqConfig` is filled in.
    - If `typename` is `cudaDevResourceTypeWorkqueue`, the
    cudaDevWorkqueueResource structure `wq` is filled in.

    Attributes
    ----------

    type : cudaDevResourceType
        Type of resource, dictates which union field was last set


    _internal_padding : bytes



    sm : cudaDevSmResource
        Resource corresponding to cudaDevResourceTypeSm `typename`.


    wqConfig : cudaDevWorkqueueConfigResource
        Resource corresponding to cudaDevResourceTypeWorkqueueConfig
        `typename`.


    wq : cudaDevWorkqueueResource
        Resource corresponding to cudaDevResourceTypeWorkqueue `typename`.


    _oversize : bytes



    nextResource : cudaDevResource_st



    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    pass

cdef class cudaGraphEdgeData(cudaGraphEdgeData_st):
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
        cudaGraphKernelNodePortDefault,
        cudaGraphKernelNodePortProgrammatic, or
        cudaGraphKernelNodePortLaunchCompletion.


    to_port : bytes
        This indicates what portion of the downstream node is dependent on
        the upstream node or portion thereof (indicated by `from_port`).
        The meaning is specific to the node type. A value of 0 in all cases
        means the entirety of the downstream node is dependent on the
        upstream work.   Currently no node types define non-zero ports.
        Accordingly, this field must be set to zero.


    type : bytes
        This should be populated with a value from cudaGraphDependencyType.
        (It is typed as char due to compiler-specific layout of bitfields.)
        See cudaGraphDependencyType.


    reserved : bytes
        These bytes are unused and must be zeroed. This ensures
        compatibility if additional fields are added in the future.


    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    pass

cdef class cudaGraphInstantiateParams(cudaGraphInstantiateParams_st):
    """
    Graph instantiation parameters

    Attributes
    ----------

    flags : unsigned long long
        Instantiation flags


    uploadStream : cudaStream_t
        Upload stream


    errNode_out : cudaGraphNode_t
        The node which caused instantiation to fail, if any


    result_out : cudaGraphInstantiateResult
        Whether instantiation was successful. If it failed, the reason why


    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    pass

cdef class cudaGraphExecUpdateResultInfo(cudaGraphExecUpdateResultInfo_st):
    """
    Result information returned by cudaGraphExecUpdate

    Attributes
    ----------

    result : cudaGraphExecUpdateResult
        Gives more specific detail when a cuda graph update fails.


    errorNode : cudaGraphNode_t
        The "to node" of the error edge when the topologies do not match.
        The error node when the error is associated with a specific node.
        NULL when the error is generic.


    errorFromNode : cudaGraphNode_t
        The from node of error edge when the topologies do not match.
        Otherwise NULL.


    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    pass

cdef class cudaLaunchMemSyncDomainMap(cudaLaunchMemSyncDomainMap_st):
    """
    Memory Synchronization Domain map  See cudaLaunchMemSyncDomain.  By
    default, kernels are launched in domain 0. Kernel launched with
    cudaLaunchMemSyncDomainRemote will have a different domain ID. User
    may also alter the domain ID with cudaLaunchMemSyncDomainMap for a
    specific stream / graph node / kernel launch. See
    cudaLaunchAttributeMemSyncDomainMap.  Domain ID range is available
    through cudaDevAttrMemSyncDomainCount.

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

cdef class cudaLaunchAttribute(cudaLaunchAttribute_st):
    """
    Launch attribute

    Attributes
    ----------

    id : cudaLaunchAttributeID
        Attribute to set


    val : cudaLaunchAttributeValue
        Value of the attribute


    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    pass

cdef class cudaAsyncNotificationInfo_t(cudaAsyncNotificationInfo):
    """
    Information describing an async notification event

    Attributes
    ----------

    type : cudaAsyncNotificationType
        The type of notification being sent


    info : anon_union11
        Information about the notification. `typename` must be checked in
        order to interpret this field.


    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    pass

cdef class cudaStreamAttrValue(cudaLaunchAttributeValue):
    """
    Launch attributes union; used as value field of cudaLaunchAttribute

    Attributes
    ----------

    pad : bytes



    accessPolicyWindow : cudaAccessPolicyWindow
        Value of launch attribute cudaLaunchAttributeAccessPolicyWindow.


    cooperative : int
        Value of launch attribute cudaLaunchAttributeCooperative. Nonzero
        indicates a cooperative kernel (see cudaLaunchCooperativeKernel).


    syncPolicy : cudaSynchronizationPolicy
        Value of launch attribute cudaLaunchAttributeSynchronizationPolicy.
        cudaSynchronizationPolicy for work queued up in this stream.


    clusterDim : anon_struct17
        Value of launch attribute cudaLaunchAttributeClusterDimension that
        represents the desired cluster dimensions for the kernel. Opaque
        type with the following fields: - `x` - The X dimension of the
        cluster, in blocks. Must be a divisor of the grid X dimension.    -
        `y` - The Y dimension of the cluster, in blocks. Must be a divisor
        of the grid Y dimension.    - `z` - The Z dimension of the cluster,
        in blocks. Must be a divisor of the grid Z dimension.


    clusterSchedulingPolicyPreference : cudaClusterSchedulingPolicy
        Value of launch attribute
        cudaLaunchAttributeClusterSchedulingPolicyPreference. Cluster
        scheduling policy preference for the kernel.


    programmaticStreamSerializationAllowed : int
        Value of launch attribute
        cudaLaunchAttributeProgrammaticStreamSerialization.


    programmaticEvent : anon_struct18
        Value of launch attribute cudaLaunchAttributeProgrammaticEvent with
        the following fields: - `cudaEvent_t` event - Event to fire when
        all blocks trigger it.    - `int` flags; - Event record flags, see
        cudaEventRecordWithFlags. Does not accept cudaEventRecordExternal.
        - `int` triggerAtBlockStart - If this is set to non-0, each block
        launch will automatically trigger the event.


    priority : int
        Value of launch attribute cudaLaunchAttributePriority. Execution
        priority of the kernel.


    memSyncDomainMap : cudaLaunchMemSyncDomainMap
        Value of launch attribute cudaLaunchAttributeMemSyncDomainMap. See
        cudaLaunchMemSyncDomainMap.


    memSyncDomain : cudaLaunchMemSyncDomain
        Value of launch attribute cudaLaunchAttributeMemSyncDomain. See
        cudaLaunchMemSyncDomain.


    preferredClusterDim : anon_struct19
        Value of launch attribute
        cudaLaunchAttributePreferredClusterDimension that represents the
        desired preferred cluster dimensions for the kernel. Opaque type
        with the following fields: - `x` - The X dimension of the preferred
        cluster, in blocks. Must be a divisor of the grid X dimension, and
        must be a multiple of the `x` field of
        cudaLaunchAttributeValue::clusterDim.    - `y` - The Y dimension of
        the preferred cluster, in blocks. Must be a divisor of the grid Y
        dimension, and must be a multiple of the `y` field of
        cudaLaunchAttributeValue::clusterDim.    - `z` - The Z dimension of
        the preferred cluster, in blocks. Must be equal to the `z` field of
        cudaLaunchAttributeValue::clusterDim.


    launchCompletionEvent : anon_struct20
        Value of launch attribute cudaLaunchAttributeLaunchCompletionEvent
        with the following fields: - `cudaEvent_t` event - Event to fire
        when the last block launches.    - `int` flags - Event record
        flags, see cudaEventRecordWithFlags. Does not accept
        cudaEventRecordExternal.


    deviceUpdatableKernelNode : anon_struct21
        Value of launch attribute
        cudaLaunchAttributeDeviceUpdatableKernelNode with the following
        fields: - `int` deviceUpdatable - Whether or not the resulting
        kernel node should be device-updatable.    -
        `cudaGraphDeviceNode_t` devNode - Returns a handle to pass to the
        various device-side update functions.


    sharedMemCarveout : unsigned int
        Value of launch attribute
        cudaLaunchAttributePreferredSharedMemoryCarveout.


    nvlinkUtilCentricScheduling : unsigned int
        Value of launch attribute
        cudaLaunchAttributeNvlinkUtilCentricScheduling.


    portableClusterSizeMode : cudaLaunchAttributePortableClusterMode
        Value of launch attribute
        cudaLaunchAttributePortableClusterSizeMode


    sharedMemoryMode : cudaSharedMemoryMode
        Value of launch attribute cudaLaunchAttributeSharedMemoryMode. See
        cudaSharedMemoryMode for acceptable values.


    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    pass

cdef class cudaKernelNodeAttrValue(cudaLaunchAttributeValue):
    """
    Launch attributes union; used as value field of cudaLaunchAttribute

    Attributes
    ----------

    pad : bytes



    accessPolicyWindow : cudaAccessPolicyWindow
        Value of launch attribute cudaLaunchAttributeAccessPolicyWindow.


    cooperative : int
        Value of launch attribute cudaLaunchAttributeCooperative. Nonzero
        indicates a cooperative kernel (see cudaLaunchCooperativeKernel).


    syncPolicy : cudaSynchronizationPolicy
        Value of launch attribute cudaLaunchAttributeSynchronizationPolicy.
        cudaSynchronizationPolicy for work queued up in this stream.


    clusterDim : anon_struct17
        Value of launch attribute cudaLaunchAttributeClusterDimension that
        represents the desired cluster dimensions for the kernel. Opaque
        type with the following fields: - `x` - The X dimension of the
        cluster, in blocks. Must be a divisor of the grid X dimension.    -
        `y` - The Y dimension of the cluster, in blocks. Must be a divisor
        of the grid Y dimension.    - `z` - The Z dimension of the cluster,
        in blocks. Must be a divisor of the grid Z dimension.


    clusterSchedulingPolicyPreference : cudaClusterSchedulingPolicy
        Value of launch attribute
        cudaLaunchAttributeClusterSchedulingPolicyPreference. Cluster
        scheduling policy preference for the kernel.


    programmaticStreamSerializationAllowed : int
        Value of launch attribute
        cudaLaunchAttributeProgrammaticStreamSerialization.


    programmaticEvent : anon_struct18
        Value of launch attribute cudaLaunchAttributeProgrammaticEvent with
        the following fields: - `cudaEvent_t` event - Event to fire when
        all blocks trigger it.    - `int` flags; - Event record flags, see
        cudaEventRecordWithFlags. Does not accept cudaEventRecordExternal.
        - `int` triggerAtBlockStart - If this is set to non-0, each block
        launch will automatically trigger the event.


    priority : int
        Value of launch attribute cudaLaunchAttributePriority. Execution
        priority of the kernel.


    memSyncDomainMap : cudaLaunchMemSyncDomainMap
        Value of launch attribute cudaLaunchAttributeMemSyncDomainMap. See
        cudaLaunchMemSyncDomainMap.


    memSyncDomain : cudaLaunchMemSyncDomain
        Value of launch attribute cudaLaunchAttributeMemSyncDomain. See
        cudaLaunchMemSyncDomain.


    preferredClusterDim : anon_struct19
        Value of launch attribute
        cudaLaunchAttributePreferredClusterDimension that represents the
        desired preferred cluster dimensions for the kernel. Opaque type
        with the following fields: - `x` - The X dimension of the preferred
        cluster, in blocks. Must be a divisor of the grid X dimension, and
        must be a multiple of the `x` field of
        cudaLaunchAttributeValue::clusterDim.    - `y` - The Y dimension of
        the preferred cluster, in blocks. Must be a divisor of the grid Y
        dimension, and must be a multiple of the `y` field of
        cudaLaunchAttributeValue::clusterDim.    - `z` - The Z dimension of
        the preferred cluster, in blocks. Must be equal to the `z` field of
        cudaLaunchAttributeValue::clusterDim.


    launchCompletionEvent : anon_struct20
        Value of launch attribute cudaLaunchAttributeLaunchCompletionEvent
        with the following fields: - `cudaEvent_t` event - Event to fire
        when the last block launches.    - `int` flags - Event record
        flags, see cudaEventRecordWithFlags. Does not accept
        cudaEventRecordExternal.


    deviceUpdatableKernelNode : anon_struct21
        Value of launch attribute
        cudaLaunchAttributeDeviceUpdatableKernelNode with the following
        fields: - `int` deviceUpdatable - Whether or not the resulting
        kernel node should be device-updatable.    -
        `cudaGraphDeviceNode_t` devNode - Returns a handle to pass to the
        various device-side update functions.


    sharedMemCarveout : unsigned int
        Value of launch attribute
        cudaLaunchAttributePreferredSharedMemoryCarveout.


    nvlinkUtilCentricScheduling : unsigned int
        Value of launch attribute
        cudaLaunchAttributeNvlinkUtilCentricScheduling.


    portableClusterSizeMode : cudaLaunchAttributePortableClusterMode
        Value of launch attribute
        cudaLaunchAttributePortableClusterSizeMode


    sharedMemoryMode : cudaSharedMemoryMode
        Value of launch attribute cudaLaunchAttributeSharedMemoryMode. See
        cudaSharedMemoryMode for acceptable values.


    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    pass

cdef class cudaEglPlaneDesc(cudaEglPlaneDesc_st):
    """
    CUDA EGL Plane Descriptor - structure defining each plane of a CUDA
    EGLFrame

    Attributes
    ----------

    width : unsigned int
        Width of plane


    height : unsigned int
        Height of plane


    depth : unsigned int
        Depth of plane


    pitch : unsigned int
        Pitch of plane


    numChannels : unsigned int
        Number of channels for the plane


    channelDesc : cudaChannelFormatDesc
        Channel Format Descriptor


    reserved : list[unsigned int]
        Reserved for future use


    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    pass

cdef class cudaEglFrame(cudaEglFrame_st):
    """
    CUDA EGLFrame Descriptor - structure defining one frame of EGL.
    Each frame may contain one or more planes depending on whether the
    surface is Multiplanar or not. Each plane of EGLFrame is
    represented by cudaEglPlaneDesc which is defined as:
    typedefstructcudaEglPlaneDesc_st unsignedintwidth;
    unsignedintheight; unsignedintdepth; unsignedintpitch;
    unsignedintnumChannels; structcudaChannelFormatDescchannelDesc;
    unsignedintreserved[4]; cudaEglPlaneDesc;

    Attributes
    ----------

    frame : anon_union12



    planeDesc : list[cudaEglPlaneDesc]
        CUDA EGL Plane Descriptor cudaEglPlaneDesc


    planeCount : unsigned int
        Number of planes


    frameType : cudaEglFrameType
        Array or Pitch


    eglColorFormat : cudaEglColorFormat
        CUDA EGL Color Format


    Methods
    -------
    getPtr()
        Get memory address of class instance
    """
    pass

cdef class cudaStream_t(driver.CUstream):
    """

    CUDA stream

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    pass

cdef class cudaEvent_t(driver.CUevent):
    """

    CUDA event types

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    pass

cdef class cudaGraph_t(driver.CUgraph):
    """

    CUDA graph

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    pass

cdef class cudaGraphNode_t(driver.CUgraphNode):
    """

    CUDA graph node.

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    pass

cdef class cudaUserObject_t(driver.CUuserObject):
    """

    CUDA user object for graphs

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    pass

cdef class cudaFunction_t(driver.CUfunction):
    """

    CUDA function

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    pass

cdef class cudaMemPool_t(driver.CUmemoryPool):
    """

    CUDA memory pool

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    pass

cdef class cudaGraphExec_t(driver.CUgraphExec):
    """

    CUDA executable (launchable) graph

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    pass

cdef class cudaEglStreamConnection(driver.CUeglStreamConnection):
    """

    CUDA EGLSream Connection

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    pass

cdef class cudaGraphConditionalHandle:
    """

    CUDA handle for conditional graph nodes

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef cyruntime.cudaGraphConditionalHandle  _pvt_val
    cdef cyruntime.cudaGraphConditionalHandle* _pvt_ptr

cdef class cudaLogIterator:
    """

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef cyruntime.cudaLogIterator  _pvt_val
    cdef cyruntime.cudaLogIterator* _pvt_ptr

cdef class cudaSurfaceObject_t:
    """

    An opaque value that represents a CUDA Surface object

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef cyruntime.cudaSurfaceObject_t  _pvt_val
    cdef cyruntime.cudaSurfaceObject_t* _pvt_ptr

cdef class cudaTextureObject_t:
    """

    An opaque value that represents a CUDA texture object

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef cyruntime.cudaTextureObject_t  _pvt_val
    cdef cyruntime.cudaTextureObject_t* _pvt_ptr

cdef class GLenum:
    """

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef cyruntime.GLenum  _pvt_val
    cdef cyruntime.GLenum* _pvt_ptr

cdef class GLuint:
    """

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef cyruntime.GLuint  _pvt_val
    cdef cyruntime.GLuint* _pvt_ptr

cdef class EGLint:
    """

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef cyruntime.EGLint  _pvt_val
    cdef cyruntime.EGLint* _pvt_ptr

cdef class VdpDevice:
    """

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef cyruntime.VdpDevice  _pvt_val
    cdef cyruntime.VdpDevice* _pvt_ptr

cdef class VdpGetProcAddress:
    """

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef cyruntime.VdpGetProcAddress  _pvt_val
    cdef cyruntime.VdpGetProcAddress* _pvt_ptr

cdef class VdpVideoSurface:
    """

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef cyruntime.VdpVideoSurface  _pvt_val
    cdef cyruntime.VdpVideoSurface* _pvt_ptr

cdef class VdpOutputSurface:
    """

    Methods
    -------
    getPtr()
        Get memory address of class instance

    """
    cdef cyruntime.VdpOutputSurface  _pvt_val
    cdef cyruntime.VdpOutputSurface* _pvt_ptr
