# Copyright 2021 NVIDIA Corporation.  All rights reserved.
#
# Please refer to the NVIDIA end user license agreement (EULA) associated
# with this source code for terms and conditions that govern your use of
# this software. Any use, reproduction, disclosure, or distribution of
# this software and related documentation outside the terms of the EULA
# is strictly prohibited.
cimport cudapython.ccudart as ccudart
cimport cudapython._lib.utils as utils
cimport cudapython.cuda as cuda

cdef class cudaArray_t:
    cdef ccudart.cudaArray_t* _ptr
    cdef bint _ptr_owner

cdef class cudaArray_const_t:
    cdef ccudart.cudaArray_const_t* _ptr
    cdef bint _ptr_owner

cdef class cudaMipmappedArray_t:
    cdef ccudart.cudaMipmappedArray_t* _ptr
    cdef bint _ptr_owner

cdef class cudaMipmappedArray_const_t:
    cdef ccudart.cudaMipmappedArray_const_t* _ptr
    cdef bint _ptr_owner

cdef class cudaGraphicsResource_t:
    cdef ccudart.cudaGraphicsResource_t* _ptr
    cdef bint _ptr_owner

cdef class cudaExternalMemory_t:
    cdef ccudart.cudaExternalMemory_t* _ptr
    cdef bint _ptr_owner

cdef class cudaExternalSemaphore_t:
    cdef ccudart.cudaExternalSemaphore_t* _ptr
    cdef bint _ptr_owner

cdef class cudaHostFn_t:
    cdef ccudart.cudaHostFn_t* _ptr
    cdef bint _ptr_owner

cdef class cudaStreamCallback_t:
    cdef ccudart.cudaStreamCallback_t* _ptr
    cdef bint _ptr_owner

cdef class cudaSurfaceObject_t:
    cdef ccudart.cudaSurfaceObject_t* _ptr
    cdef bint _ptr_owner

cdef class cudaTextureObject_t:
    cdef ccudart.cudaTextureObject_t* _ptr
    cdef bint _ptr_owner

cdef class dim3:
    cdef ccudart.dim3* _ptr
    cdef bint _ptr_owner

cdef class cudaChannelFormatDesc:
    cdef ccudart.cudaChannelFormatDesc* _ptr
    cdef bint _ptr_owner

cdef class _cudaArraySparseProperties_tileExtent_s:
    cdef ccudart.cudaArraySparseProperties* _ptr

cdef class cudaArraySparseProperties:
    cdef ccudart.cudaArraySparseProperties* _ptr
    cdef bint _ptr_owner
    cdef _cudaArraySparseProperties_tileExtent_s _tileExtent

cdef class cudaPitchedPtr:
    cdef ccudart.cudaPitchedPtr* _ptr
    cdef bint _ptr_owner
    cdef utils.HelperInputVoidPtr _cptr

cdef class cudaExtent:
    cdef ccudart.cudaExtent* _ptr
    cdef bint _ptr_owner

cdef class cudaPos:
    cdef ccudart.cudaPos* _ptr
    cdef bint _ptr_owner

cdef class cudaMemcpy3DParms:
    cdef ccudart.cudaMemcpy3DParms* _ptr
    cdef bint _ptr_owner
    cdef cudaArray_t _srcArray
    cdef cudaPos _srcPos
    cdef cudaPitchedPtr _srcPtr
    cdef cudaArray_t _dstArray
    cdef cudaPos _dstPos
    cdef cudaPitchedPtr _dstPtr
    cdef cudaExtent _extent

cdef class cudaMemcpy3DPeerParms:
    cdef ccudart.cudaMemcpy3DPeerParms* _ptr
    cdef bint _ptr_owner
    cdef cudaArray_t _srcArray
    cdef cudaPos _srcPos
    cdef cudaPitchedPtr _srcPtr
    cdef cudaArray_t _dstArray
    cdef cudaPos _dstPos
    cdef cudaPitchedPtr _dstPtr
    cdef cudaExtent _extent

cdef class cudaMemsetParams:
    cdef ccudart.cudaMemsetParams* _ptr
    cdef bint _ptr_owner
    cdef utils.HelperInputVoidPtr _cdst

cdef class cudaAccessPolicyWindow:
    cdef ccudart.cudaAccessPolicyWindow* _ptr
    cdef bint _ptr_owner
    cdef utils.HelperInputVoidPtr _cbase_ptr

cdef class cudaHostNodeParams:
    cdef ccudart.cudaHostNodeParams* _ptr
    cdef bint _ptr_owner
    cdef cudaHostFn_t _fn
    cdef utils.HelperInputVoidPtr _cuserData

cdef class cudaStreamAttrValue:
    cdef ccudart.cudaStreamAttrValue* _ptr
    cdef bint _ptr_owner
    cdef cudaAccessPolicyWindow _accessPolicyWindow

cdef class cudaKernelNodeAttrValue:
    cdef ccudart.cudaKernelNodeAttrValue* _ptr
    cdef bint _ptr_owner
    cdef cudaAccessPolicyWindow _accessPolicyWindow

cdef class _cudaResourceDesc_res_res_array_s:
    cdef ccudart.cudaResourceDesc* _ptr
    cdef cudaArray_t _array

cdef class _cudaResourceDesc_res_res_mipmap_s:
    cdef ccudart.cudaResourceDesc* _ptr
    cdef cudaMipmappedArray_t _mipmap

cdef class _cudaResourceDesc_res_res_linear_s:
    cdef ccudart.cudaResourceDesc* _ptr
    cdef utils.HelperInputVoidPtr _cdevPtr
    cdef cudaChannelFormatDesc _desc

cdef class _cudaResourceDesc_res_res_pitch2D_s:
    cdef ccudart.cudaResourceDesc* _ptr
    cdef utils.HelperInputVoidPtr _cdevPtr
    cdef cudaChannelFormatDesc _desc

cdef class _cudaResourceDesc_res_u:
    cdef ccudart.cudaResourceDesc* _ptr
    cdef _cudaResourceDesc_res_res_array_s _array
    cdef _cudaResourceDesc_res_res_mipmap_s _mipmap
    cdef _cudaResourceDesc_res_res_linear_s _linear
    cdef _cudaResourceDesc_res_res_pitch2D_s _pitch2D

cdef class cudaResourceDesc:
    cdef ccudart.cudaResourceDesc* _ptr
    cdef bint _ptr_owner
    cdef _cudaResourceDesc_res_u _res

cdef class cudaResourceViewDesc:
    cdef ccudart.cudaResourceViewDesc* _ptr
    cdef bint _ptr_owner

cdef class cudaPointerAttributes:
    cdef ccudart.cudaPointerAttributes* _ptr
    cdef bint _ptr_owner
    cdef utils.HelperInputVoidPtr _cdevicePointer
    cdef utils.HelperInputVoidPtr _chostPointer

cdef class cudaFuncAttributes:
    cdef ccudart.cudaFuncAttributes* _ptr
    cdef bint _ptr_owner

cdef class cudaMemLocation:
    cdef ccudart.cudaMemLocation* _ptr
    cdef bint _ptr_owner

cdef class cudaMemAccessDesc:
    cdef ccudart.cudaMemAccessDesc* _ptr
    cdef bint _ptr_owner
    cdef cudaMemLocation _location

cdef class cudaMemPoolProps:
    cdef ccudart.cudaMemPoolProps* _ptr
    cdef bint _ptr_owner
    cdef cudaMemLocation _location
    cdef utils.HelperInputVoidPtr _cwin32SecurityAttributes

cdef class cudaMemPoolPtrExportData:
    cdef ccudart.cudaMemPoolPtrExportData* _ptr
    cdef bint _ptr_owner

cdef class cudaMemAllocNodeParams:
    cdef ccudart.cudaMemAllocNodeParams* _ptr
    cdef bint _ptr_owner
    cdef cudaMemPoolProps _poolProps
    cdef size_t _accessDescs_length
    cdef ccudart.cudaMemAccessDesc* _accessDescs
    cdef utils.HelperInputVoidPtr _cdptr

cdef class CUuuid_st:
    cdef ccudart.CUuuid_st* _ptr
    cdef bint _ptr_owner

cdef class cudaDeviceProp:
    cdef ccudart.cudaDeviceProp* _ptr
    cdef bint _ptr_owner
    cdef cudaUUID_t _uuid

cdef class cudaIpcEventHandle_st:
    cdef ccudart.cudaIpcEventHandle_st* _ptr
    cdef bint _ptr_owner

cdef class cudaIpcMemHandle_st:
    cdef ccudart.cudaIpcMemHandle_st* _ptr
    cdef bint _ptr_owner

cdef class _cudaExternalMemoryHandleDesc_handle_handle_win32_s:
    cdef ccudart.cudaExternalMemoryHandleDesc* _ptr
    cdef utils.HelperInputVoidPtr _chandle
    cdef utils.HelperInputVoidPtr _cname

cdef class _cudaExternalMemoryHandleDesc_handle_u:
    cdef ccudart.cudaExternalMemoryHandleDesc* _ptr
    cdef _cudaExternalMemoryHandleDesc_handle_handle_win32_s _win32
    cdef utils.HelperInputVoidPtr _cnvSciBufObject

cdef class cudaExternalMemoryHandleDesc:
    cdef ccudart.cudaExternalMemoryHandleDesc* _ptr
    cdef bint _ptr_owner
    cdef _cudaExternalMemoryHandleDesc_handle_u _handle

cdef class cudaExternalMemoryBufferDesc:
    cdef ccudart.cudaExternalMemoryBufferDesc* _ptr
    cdef bint _ptr_owner

cdef class cudaExternalMemoryMipmappedArrayDesc:
    cdef ccudart.cudaExternalMemoryMipmappedArrayDesc* _ptr
    cdef bint _ptr_owner
    cdef cudaChannelFormatDesc _formatDesc
    cdef cudaExtent _extent

cdef class _cudaExternalSemaphoreHandleDesc_handle_handle_win32_s:
    cdef ccudart.cudaExternalSemaphoreHandleDesc* _ptr
    cdef utils.HelperInputVoidPtr _chandle
    cdef utils.HelperInputVoidPtr _cname

cdef class _cudaExternalSemaphoreHandleDesc_handle_u:
    cdef ccudart.cudaExternalSemaphoreHandleDesc* _ptr
    cdef _cudaExternalSemaphoreHandleDesc_handle_handle_win32_s _win32
    cdef utils.HelperInputVoidPtr _cnvSciSyncObj

cdef class cudaExternalSemaphoreHandleDesc:
    cdef ccudart.cudaExternalSemaphoreHandleDesc* _ptr
    cdef bint _ptr_owner
    cdef _cudaExternalSemaphoreHandleDesc_handle_u _handle

cdef class _cudaExternalSemaphoreSignalParams_params_params_fence_s:
    cdef ccudart.cudaExternalSemaphoreSignalParams* _ptr

cdef class _cudaExternalSemaphoreSignalParams_params_params_nvSciSync_u:
    cdef ccudart.cudaExternalSemaphoreSignalParams* _ptr
    cdef utils.HelperInputVoidPtr _cfence

cdef class _cudaExternalSemaphoreSignalParams_params_params_keyedMutex_s:
    cdef ccudart.cudaExternalSemaphoreSignalParams* _ptr

cdef class _cudaExternalSemaphoreSignalParams_params_s:
    cdef ccudart.cudaExternalSemaphoreSignalParams* _ptr
    cdef _cudaExternalSemaphoreSignalParams_params_params_fence_s _fence
    cdef _cudaExternalSemaphoreSignalParams_params_params_nvSciSync_u _nvSciSync
    cdef _cudaExternalSemaphoreSignalParams_params_params_keyedMutex_s _keyedMutex

cdef class cudaExternalSemaphoreSignalParams:
    cdef ccudart.cudaExternalSemaphoreSignalParams* _ptr
    cdef bint _ptr_owner
    cdef _cudaExternalSemaphoreSignalParams_params_s _params

cdef class _cudaExternalSemaphoreWaitParams_params_params_fence_s:
    cdef ccudart.cudaExternalSemaphoreWaitParams* _ptr

cdef class _cudaExternalSemaphoreWaitParams_params_params_nvSciSync_u:
    cdef ccudart.cudaExternalSemaphoreWaitParams* _ptr
    cdef utils.HelperInputVoidPtr _cfence

cdef class _cudaExternalSemaphoreWaitParams_params_params_keyedMutex_s:
    cdef ccudart.cudaExternalSemaphoreWaitParams* _ptr

cdef class _cudaExternalSemaphoreWaitParams_params_s:
    cdef ccudart.cudaExternalSemaphoreWaitParams* _ptr
    cdef _cudaExternalSemaphoreWaitParams_params_params_fence_s _fence
    cdef _cudaExternalSemaphoreWaitParams_params_params_nvSciSync_u _nvSciSync
    cdef _cudaExternalSemaphoreWaitParams_params_params_keyedMutex_s _keyedMutex

cdef class cudaExternalSemaphoreWaitParams:
    cdef ccudart.cudaExternalSemaphoreWaitParams* _ptr
    cdef bint _ptr_owner
    cdef _cudaExternalSemaphoreWaitParams_params_s _params

cdef class cudaKernelNodeParams:
    cdef ccudart.cudaKernelNodeParams* _ptr
    cdef bint _ptr_owner
    cdef utils.HelperInputVoidPtr _cfunc
    cdef dim3 _gridDim
    cdef dim3 _blockDim
    cdef utils.HelperKernelParams _ckernelParams

cdef class cudaExternalSemaphoreSignalNodeParams:
    cdef ccudart.cudaExternalSemaphoreSignalNodeParams* _ptr
    cdef bint _ptr_owner
    cdef size_t _extSemArray_length
    cdef ccudart.cudaExternalSemaphore_t* _extSemArray
    cdef size_t _paramsArray_length
    cdef ccudart.cudaExternalSemaphoreSignalParams* _paramsArray

cdef class cudaExternalSemaphoreWaitNodeParams:
    cdef ccudart.cudaExternalSemaphoreWaitNodeParams* _ptr
    cdef bint _ptr_owner
    cdef size_t _extSemArray_length
    cdef ccudart.cudaExternalSemaphore_t* _extSemArray
    cdef size_t _paramsArray_length
    cdef ccudart.cudaExternalSemaphoreWaitParams* _paramsArray

cdef class cudaTextureDesc:
    cdef ccudart.cudaTextureDesc* _ptr
    cdef bint _ptr_owner

cdef class CUuuid(CUuuid_st):
    pass

cdef class cudaUUID_t(CUuuid_st):
    pass

cdef class cudaIpcEventHandle_t(cudaIpcEventHandle_st):
    pass

cdef class cudaIpcMemHandle_t(cudaIpcMemHandle_st):
    pass

cdef class cudaStream_t(cuda.CUstream):
    pass

cdef class cudaEvent_t(cuda.CUevent):
    pass

cdef class cudaGraph_t(cuda.CUgraph):
    pass

cdef class cudaGraphNode_t(cuda.CUgraphNode):
    pass

cdef class cudaUserObject_t(cuda.CUuserObject):
    pass

cdef class cudaFunction_t(cuda.CUfunction):
    pass

cdef class cudaMemPool_t(cuda.CUmemoryPool):
    pass

cdef class cudaGraphExec_t(cuda.CUgraphExec):
    pass
