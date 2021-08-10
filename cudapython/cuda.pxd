# Copyright 2021 NVIDIA Corporation.  All rights reserved.
#
# Please refer to the NVIDIA end user license agreement (EULA) associated
# with this source code for terms and conditions that govern your use of
# this software. Any use, reproduction, disclosure, or distribution of
# this software and related documentation outside the terms of the EULA
# is strictly prohibited.
cimport cudapython.ccuda as ccuda
cimport cudapython._lib.utils as utils

cdef class CUcontext:
    cdef ccuda.CUcontext* _ptr
    cdef bint _ptr_owner

cdef class CUmodule:
    cdef ccuda.CUmodule* _ptr
    cdef bint _ptr_owner

cdef class CUfunction:
    cdef ccuda.CUfunction* _ptr
    cdef bint _ptr_owner

cdef class CUarray:
    cdef ccuda.CUarray* _ptr
    cdef bint _ptr_owner

cdef class CUmipmappedArray:
    cdef ccuda.CUmipmappedArray* _ptr
    cdef bint _ptr_owner

cdef class CUtexref:
    cdef ccuda.CUtexref* _ptr
    cdef bint _ptr_owner

cdef class CUsurfref:
    cdef ccuda.CUsurfref* _ptr
    cdef bint _ptr_owner

cdef class CUevent:
    cdef ccuda.CUevent* _ptr
    cdef bint _ptr_owner

cdef class CUstream:
    cdef ccuda.CUstream* _ptr
    cdef bint _ptr_owner

cdef class CUgraphicsResource:
    cdef ccuda.CUgraphicsResource* _ptr
    cdef bint _ptr_owner

cdef class CUexternalMemory:
    cdef ccuda.CUexternalMemory* _ptr
    cdef bint _ptr_owner

cdef class CUexternalSemaphore:
    cdef ccuda.CUexternalSemaphore* _ptr
    cdef bint _ptr_owner

cdef class CUgraph:
    cdef ccuda.CUgraph* _ptr
    cdef bint _ptr_owner

cdef class CUgraphNode:
    cdef ccuda.CUgraphNode* _ptr
    cdef bint _ptr_owner

cdef class CUgraphExec:
    cdef ccuda.CUgraphExec* _ptr
    cdef bint _ptr_owner

cdef class CUmemoryPool:
    cdef ccuda.CUmemoryPool* _ptr
    cdef bint _ptr_owner

cdef class CUuserObject:
    cdef ccuda.CUuserObject* _ptr
    cdef bint _ptr_owner

cdef class CUlinkState:
    cdef ccuda.CUlinkState* _ptr
    cdef bint _ptr_owner
    cdef list _keepalive

cdef class CUhostFn:
    cdef ccuda.CUhostFn* _ptr
    cdef bint _ptr_owner

cdef class CUstreamCallback:
    cdef ccuda.CUstreamCallback* _ptr
    cdef bint _ptr_owner

cdef class CUoccupancyB2DSize:
    cdef ccuda.CUoccupancyB2DSize* _ptr
    cdef bint _ptr_owner

cdef class cuuint32_t:
    cdef ccuda.cuuint32_t* _ptr
    cdef bint _ptr_owner

cdef class cuuint64_t:
    cdef ccuda.cuuint64_t* _ptr
    cdef bint _ptr_owner

cdef class CUdeviceptr_v2:
    cdef ccuda.CUdeviceptr_v2* _ptr
    cdef bint _ptr_owner

cdef class CUdeviceptr:
    cdef ccuda.CUdeviceptr* _ptr
    cdef bint _ptr_owner

cdef class CUdevice_v1:
    cdef ccuda.CUdevice_v1* _ptr
    cdef bint _ptr_owner

cdef class CUdevice:
    cdef ccuda.CUdevice* _ptr
    cdef bint _ptr_owner

cdef class CUtexObject_v1:
    cdef ccuda.CUtexObject_v1* _ptr
    cdef bint _ptr_owner

cdef class CUtexObject:
    cdef ccuda.CUtexObject* _ptr
    cdef bint _ptr_owner

cdef class CUsurfObject_v1:
    cdef ccuda.CUsurfObject_v1* _ptr
    cdef bint _ptr_owner

cdef class CUsurfObject:
    cdef ccuda.CUsurfObject* _ptr
    cdef bint _ptr_owner

cdef class CUmemGenericAllocationHandle_v1:
    cdef ccuda.CUmemGenericAllocationHandle_v1* _ptr
    cdef bint _ptr_owner

cdef class CUmemGenericAllocationHandle:
    cdef ccuda.CUmemGenericAllocationHandle* _ptr
    cdef bint _ptr_owner

cdef class CUuuid_st:
    cdef ccuda.CUuuid_st* _ptr
    cdef bint _ptr_owner

cdef class CUipcEventHandle_st:
    cdef ccuda.CUipcEventHandle_st* _ptr
    cdef bint _ptr_owner

cdef class CUipcMemHandle_st:
    cdef ccuda.CUipcMemHandle_st* _ptr
    cdef bint _ptr_owner

cdef class CUstreamMemOpWaitValueParams_st:
    cdef ccuda.CUstreamMemOpWaitValueParams_st* _ptr
    cdef bint _ptr_owner
    cdef CUdeviceptr _address
    cdef cuuint64_t _value64
    cdef CUdeviceptr _alias

cdef class CUstreamMemOpWriteValueParams_st:
    cdef ccuda.CUstreamMemOpWriteValueParams_st* _ptr
    cdef bint _ptr_owner
    cdef CUdeviceptr _address
    cdef cuuint64_t _value64
    cdef CUdeviceptr _alias

cdef class CUstreamMemOpFlushRemoteWritesParams_st:
    cdef ccuda.CUstreamMemOpFlushRemoteWritesParams_st* _ptr
    cdef bint _ptr_owner

cdef class CUstreamBatchMemOpParams_union:
    cdef ccuda.CUstreamBatchMemOpParams_union* _ptr
    cdef bint _ptr_owner
    cdef CUstreamMemOpWaitValueParams_st _waitValue
    cdef CUstreamMemOpWriteValueParams_st _writeValue
    cdef CUstreamMemOpFlushRemoteWritesParams_st _flushRemoteWrites
    cdef cuuint64_t _pad

cdef class CUdevprop_st:
    cdef ccuda.CUdevprop_st* _ptr
    cdef bint _ptr_owner

cdef class CUaccessPolicyWindow_st:
    cdef ccuda.CUaccessPolicyWindow_st* _ptr
    cdef bint _ptr_owner
    cdef utils.HelperInputVoidPtr _cbase_ptr

cdef class CUDA_KERNEL_NODE_PARAMS_st:
    cdef ccuda.CUDA_KERNEL_NODE_PARAMS_st* _ptr
    cdef bint _ptr_owner
    cdef CUfunction _func
    cdef utils.HelperKernelParams _ckernelParams

cdef class CUDA_MEMSET_NODE_PARAMS_st:
    cdef ccuda.CUDA_MEMSET_NODE_PARAMS_st* _ptr
    cdef bint _ptr_owner
    cdef CUdeviceptr _dst

cdef class CUDA_HOST_NODE_PARAMS_st:
    cdef ccuda.CUDA_HOST_NODE_PARAMS_st* _ptr
    cdef bint _ptr_owner
    cdef CUhostFn _fn
    cdef utils.HelperInputVoidPtr _cuserData

cdef class CUkernelNodeAttrValue_union:
    cdef ccuda.CUkernelNodeAttrValue_union* _ptr
    cdef bint _ptr_owner
    cdef CUaccessPolicyWindow _accessPolicyWindow

cdef class CUstreamAttrValue_union:
    cdef ccuda.CUstreamAttrValue_union* _ptr
    cdef bint _ptr_owner
    cdef CUaccessPolicyWindow _accessPolicyWindow

cdef class CUexecAffinitySmCount_st:
    cdef ccuda.CUexecAffinitySmCount_st* _ptr
    cdef bint _ptr_owner

cdef class _CUexecAffinityParam_v1_CUexecAffinityParam_v1_CUexecAffinityParam_st_param_u:
    cdef ccuda.CUexecAffinityParam_st* _ptr
    cdef CUexecAffinitySmCount _smCount

cdef class CUexecAffinityParam_st:
    cdef ccuda.CUexecAffinityParam_st* _ptr
    cdef bint _ptr_owner
    cdef _CUexecAffinityParam_v1_CUexecAffinityParam_v1_CUexecAffinityParam_st_param_u _param

cdef class CUDA_MEMCPY2D_st:
    cdef ccuda.CUDA_MEMCPY2D_st* _ptr
    cdef bint _ptr_owner
    cdef utils.HelperInputVoidPtr _csrcHost
    cdef CUdeviceptr _srcDevice
    cdef CUarray _srcArray
    cdef utils.HelperInputVoidPtr _cdstHost
    cdef CUdeviceptr _dstDevice
    cdef CUarray _dstArray

cdef class CUDA_MEMCPY3D_st:
    cdef ccuda.CUDA_MEMCPY3D_st* _ptr
    cdef bint _ptr_owner
    cdef utils.HelperInputVoidPtr _csrcHost
    cdef CUdeviceptr _srcDevice
    cdef CUarray _srcArray
    cdef utils.HelperInputVoidPtr _creserved0
    cdef utils.HelperInputVoidPtr _cdstHost
    cdef CUdeviceptr _dstDevice
    cdef CUarray _dstArray
    cdef utils.HelperInputVoidPtr _creserved1

cdef class CUDA_MEMCPY3D_PEER_st:
    cdef ccuda.CUDA_MEMCPY3D_PEER_st* _ptr
    cdef bint _ptr_owner
    cdef utils.HelperInputVoidPtr _csrcHost
    cdef CUdeviceptr _srcDevice
    cdef CUarray _srcArray
    cdef CUcontext _srcContext
    cdef utils.HelperInputVoidPtr _cdstHost
    cdef CUdeviceptr _dstDevice
    cdef CUarray _dstArray
    cdef CUcontext _dstContext

cdef class CUDA_ARRAY_DESCRIPTOR_st:
    cdef ccuda.CUDA_ARRAY_DESCRIPTOR_st* _ptr
    cdef bint _ptr_owner

cdef class CUDA_ARRAY3D_DESCRIPTOR_st:
    cdef ccuda.CUDA_ARRAY3D_DESCRIPTOR_st* _ptr
    cdef bint _ptr_owner

cdef class _CUDA_ARRAY_SPARSE_PROPERTIES_v1_CUDA_ARRAY_SPARSE_PROPERTIES_v1_CUDA_ARRAY_SPARSE_PROPERTIES_st_tileExtent_s:
    cdef ccuda.CUDA_ARRAY_SPARSE_PROPERTIES_st* _ptr

cdef class CUDA_ARRAY_SPARSE_PROPERTIES_st:
    cdef ccuda.CUDA_ARRAY_SPARSE_PROPERTIES_st* _ptr
    cdef bint _ptr_owner
    cdef _CUDA_ARRAY_SPARSE_PROPERTIES_v1_CUDA_ARRAY_SPARSE_PROPERTIES_v1_CUDA_ARRAY_SPARSE_PROPERTIES_st_tileExtent_s _tileExtent

cdef class _CUDA_RESOURCE_DESC_v1_CUDA_RESOURCE_DESC_v1_CUDA_RESOURCE_DESC_st_res_res_array_s:
    cdef ccuda.CUDA_RESOURCE_DESC_st* _ptr
    cdef CUarray _hArray

cdef class _CUDA_RESOURCE_DESC_v1_CUDA_RESOURCE_DESC_v1_CUDA_RESOURCE_DESC_st_res_res_mipmap_s:
    cdef ccuda.CUDA_RESOURCE_DESC_st* _ptr
    cdef CUmipmappedArray _hMipmappedArray

cdef class _CUDA_RESOURCE_DESC_v1_CUDA_RESOURCE_DESC_v1_CUDA_RESOURCE_DESC_st_res_res_linear_s:
    cdef ccuda.CUDA_RESOURCE_DESC_st* _ptr
    cdef CUdeviceptr _devPtr

cdef class _CUDA_RESOURCE_DESC_v1_CUDA_RESOURCE_DESC_v1_CUDA_RESOURCE_DESC_st_res_res_pitch2D_s:
    cdef ccuda.CUDA_RESOURCE_DESC_st* _ptr
    cdef CUdeviceptr _devPtr

cdef class _CUDA_RESOURCE_DESC_v1_CUDA_RESOURCE_DESC_v1_CUDA_RESOURCE_DESC_st_res_res_reserved_s:
    cdef ccuda.CUDA_RESOURCE_DESC_st* _ptr

cdef class _CUDA_RESOURCE_DESC_v1_CUDA_RESOURCE_DESC_v1_CUDA_RESOURCE_DESC_st_res_u:
    cdef ccuda.CUDA_RESOURCE_DESC_st* _ptr
    cdef _CUDA_RESOURCE_DESC_v1_CUDA_RESOURCE_DESC_v1_CUDA_RESOURCE_DESC_st_res_res_array_s _array
    cdef _CUDA_RESOURCE_DESC_v1_CUDA_RESOURCE_DESC_v1_CUDA_RESOURCE_DESC_st_res_res_mipmap_s _mipmap
    cdef _CUDA_RESOURCE_DESC_v1_CUDA_RESOURCE_DESC_v1_CUDA_RESOURCE_DESC_st_res_res_linear_s _linear
    cdef _CUDA_RESOURCE_DESC_v1_CUDA_RESOURCE_DESC_v1_CUDA_RESOURCE_DESC_st_res_res_pitch2D_s _pitch2D
    cdef _CUDA_RESOURCE_DESC_v1_CUDA_RESOURCE_DESC_v1_CUDA_RESOURCE_DESC_st_res_res_reserved_s _reserved

cdef class CUDA_RESOURCE_DESC_st:
    cdef ccuda.CUDA_RESOURCE_DESC_st* _ptr
    cdef bint _ptr_owner
    cdef _CUDA_RESOURCE_DESC_v1_CUDA_RESOURCE_DESC_v1_CUDA_RESOURCE_DESC_st_res_u _res

cdef class CUDA_TEXTURE_DESC_st:
    cdef ccuda.CUDA_TEXTURE_DESC_st* _ptr
    cdef bint _ptr_owner

cdef class CUDA_RESOURCE_VIEW_DESC_st:
    cdef ccuda.CUDA_RESOURCE_VIEW_DESC_st* _ptr
    cdef bint _ptr_owner

cdef class CUDA_POINTER_ATTRIBUTE_P2P_TOKENS_st:
    cdef ccuda.CUDA_POINTER_ATTRIBUTE_P2P_TOKENS_st* _ptr
    cdef bint _ptr_owner

cdef class CUDA_LAUNCH_PARAMS_st:
    cdef ccuda.CUDA_LAUNCH_PARAMS_st* _ptr
    cdef bint _ptr_owner
    cdef CUfunction _function
    cdef CUstream _hStream
    cdef utils.HelperKernelParams _ckernelParams

cdef class _CUDA_EXTERNAL_MEMORY_HANDLE_DESC_v1_CUDA_EXTERNAL_MEMORY_HANDLE_DESC_v1_CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st_handle_handle_win32_s:
    cdef ccuda.CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st* _ptr
    cdef utils.HelperInputVoidPtr _chandle
    cdef utils.HelperInputVoidPtr _cname

cdef class _CUDA_EXTERNAL_MEMORY_HANDLE_DESC_v1_CUDA_EXTERNAL_MEMORY_HANDLE_DESC_v1_CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st_handle_u:
    cdef ccuda.CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st* _ptr
    cdef _CUDA_EXTERNAL_MEMORY_HANDLE_DESC_v1_CUDA_EXTERNAL_MEMORY_HANDLE_DESC_v1_CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st_handle_handle_win32_s _win32
    cdef utils.HelperInputVoidPtr _cnvSciBufObject

cdef class CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st:
    cdef ccuda.CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st* _ptr
    cdef bint _ptr_owner
    cdef _CUDA_EXTERNAL_MEMORY_HANDLE_DESC_v1_CUDA_EXTERNAL_MEMORY_HANDLE_DESC_v1_CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st_handle_u _handle

cdef class CUDA_EXTERNAL_MEMORY_BUFFER_DESC_st:
    cdef ccuda.CUDA_EXTERNAL_MEMORY_BUFFER_DESC_st* _ptr
    cdef bint _ptr_owner

cdef class CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC_st:
    cdef ccuda.CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC_st* _ptr
    cdef bint _ptr_owner
    cdef CUDA_ARRAY3D_DESCRIPTOR _arrayDesc

cdef class _CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_v1_CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_v1_CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_st_handle_handle_win32_s:
    cdef ccuda.CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_st* _ptr
    cdef utils.HelperInputVoidPtr _chandle
    cdef utils.HelperInputVoidPtr _cname

cdef class _CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_v1_CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_v1_CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_st_handle_u:
    cdef ccuda.CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_st* _ptr
    cdef _CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_v1_CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_v1_CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_st_handle_handle_win32_s _win32
    cdef utils.HelperInputVoidPtr _cnvSciSyncObj

cdef class CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_st:
    cdef ccuda.CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_st* _ptr
    cdef bint _ptr_owner
    cdef _CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_v1_CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_v1_CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_st_handle_u _handle

cdef class _CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_v1_CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_v1_CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st_params_params_fence_s:
    cdef ccuda.CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st* _ptr

cdef class _CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_v1_CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_v1_CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st_params_params_nvSciSync_u:
    cdef ccuda.CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st* _ptr
    cdef utils.HelperInputVoidPtr _cfence

cdef class _CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_v1_CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_v1_CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st_params_params_keyedMutex_s:
    cdef ccuda.CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st* _ptr

cdef class _CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_v1_CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_v1_CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st_params_s:
    cdef ccuda.CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st* _ptr
    cdef _CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_v1_CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_v1_CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st_params_params_fence_s _fence
    cdef _CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_v1_CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_v1_CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st_params_params_nvSciSync_u _nvSciSync
    cdef _CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_v1_CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_v1_CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st_params_params_keyedMutex_s _keyedMutex

cdef class CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st:
    cdef ccuda.CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st* _ptr
    cdef bint _ptr_owner
    cdef _CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_v1_CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_v1_CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st_params_s _params

cdef class _CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_v1_CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_v1_CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st_params_params_fence_s:
    cdef ccuda.CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st* _ptr

cdef class _CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_v1_CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_v1_CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st_params_params_nvSciSync_u:
    cdef ccuda.CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st* _ptr
    cdef utils.HelperInputVoidPtr _cfence

cdef class _CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_v1_CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_v1_CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st_params_params_keyedMutex_s:
    cdef ccuda.CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st* _ptr

cdef class _CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_v1_CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_v1_CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st_params_s:
    cdef ccuda.CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st* _ptr
    cdef _CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_v1_CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_v1_CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st_params_params_fence_s _fence
    cdef _CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_v1_CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_v1_CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st_params_params_nvSciSync_u _nvSciSync
    cdef _CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_v1_CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_v1_CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st_params_params_keyedMutex_s _keyedMutex

cdef class CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st:
    cdef ccuda.CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st* _ptr
    cdef bint _ptr_owner
    cdef _CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_v1_CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_v1_CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st_params_s _params

cdef class CUDA_EXT_SEM_SIGNAL_NODE_PARAMS_st:
    cdef ccuda.CUDA_EXT_SEM_SIGNAL_NODE_PARAMS_st* _ptr
    cdef bint _ptr_owner
    cdef size_t _extSemArray_length
    cdef ccuda.CUexternalSemaphore* _extSemArray
    cdef size_t _paramsArray_length
    cdef ccuda.CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS* _paramsArray

cdef class CUDA_EXT_SEM_WAIT_NODE_PARAMS_st:
    cdef ccuda.CUDA_EXT_SEM_WAIT_NODE_PARAMS_st* _ptr
    cdef bint _ptr_owner
    cdef size_t _extSemArray_length
    cdef ccuda.CUexternalSemaphore* _extSemArray
    cdef size_t _paramsArray_length
    cdef ccuda.CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS* _paramsArray

cdef class _CUarrayMapInfo_v1_CUarrayMapInfo_v1_CUarrayMapInfo_st_resource_u:
    cdef ccuda.CUarrayMapInfo_st* _ptr
    cdef CUmipmappedArray _mipmap
    cdef CUarray _array

cdef class _CUarrayMapInfo_v1_CUarrayMapInfo_v1_CUarrayMapInfo_st_subresource_subresource_sparseLevel_s:
    cdef ccuda.CUarrayMapInfo_st* _ptr

cdef class _CUarrayMapInfo_v1_CUarrayMapInfo_v1_CUarrayMapInfo_st_subresource_subresource_miptail_s:
    cdef ccuda.CUarrayMapInfo_st* _ptr

cdef class _CUarrayMapInfo_v1_CUarrayMapInfo_v1_CUarrayMapInfo_st_subresource_u:
    cdef ccuda.CUarrayMapInfo_st* _ptr
    cdef _CUarrayMapInfo_v1_CUarrayMapInfo_v1_CUarrayMapInfo_st_subresource_subresource_sparseLevel_s _sparseLevel
    cdef _CUarrayMapInfo_v1_CUarrayMapInfo_v1_CUarrayMapInfo_st_subresource_subresource_miptail_s _miptail

cdef class _CUarrayMapInfo_v1_CUarrayMapInfo_v1_CUarrayMapInfo_st_memHandle_u:
    cdef ccuda.CUarrayMapInfo_st* _ptr
    cdef CUmemGenericAllocationHandle _memHandle

cdef class CUarrayMapInfo_st:
    cdef ccuda.CUarrayMapInfo_st* _ptr
    cdef bint _ptr_owner
    cdef _CUarrayMapInfo_v1_CUarrayMapInfo_v1_CUarrayMapInfo_st_resource_u _resource
    cdef _CUarrayMapInfo_v1_CUarrayMapInfo_v1_CUarrayMapInfo_st_subresource_u _subresource
    cdef _CUarrayMapInfo_v1_CUarrayMapInfo_v1_CUarrayMapInfo_st_memHandle_u _memHandle

cdef class CUmemLocation_st:
    cdef ccuda.CUmemLocation_st* _ptr
    cdef bint _ptr_owner

cdef class _CUmemAllocationProp_v1_CUmemAllocationProp_v1_CUmemAllocationProp_st_allocFlags_s:
    cdef ccuda.CUmemAllocationProp_st* _ptr

cdef class CUmemAllocationProp_st:
    cdef ccuda.CUmemAllocationProp_st* _ptr
    cdef bint _ptr_owner
    cdef CUmemLocation _location
    cdef utils.HelperInputVoidPtr _cwin32HandleMetaData
    cdef _CUmemAllocationProp_v1_CUmemAllocationProp_v1_CUmemAllocationProp_st_allocFlags_s _allocFlags

cdef class CUmemAccessDesc_st:
    cdef ccuda.CUmemAccessDesc_st* _ptr
    cdef bint _ptr_owner
    cdef CUmemLocation _location

cdef class CUmemPoolProps_st:
    cdef ccuda.CUmemPoolProps_st* _ptr
    cdef bint _ptr_owner
    cdef CUmemLocation _location
    cdef utils.HelperInputVoidPtr _cwin32SecurityAttributes

cdef class CUmemPoolPtrExportData_st:
    cdef ccuda.CUmemPoolPtrExportData_st* _ptr
    cdef bint _ptr_owner

cdef class CUDA_MEM_ALLOC_NODE_PARAMS_st:
    cdef ccuda.CUDA_MEM_ALLOC_NODE_PARAMS_st* _ptr
    cdef bint _ptr_owner
    cdef CUmemPoolProps _poolProps
    cdef size_t _accessDescs_length
    cdef ccuda.CUmemAccessDesc* _accessDescs
    cdef CUdeviceptr _dptr

cdef class CUuuid(CUuuid_st):
    pass

cdef class CUipcEventHandle_v1(CUipcEventHandle_st):
    pass

cdef class CUipcEventHandle(CUipcEventHandle_st):
    pass

cdef class CUipcMemHandle_v1(CUipcMemHandle_st):
    pass

cdef class CUipcMemHandle(CUipcMemHandle_st):
    pass

cdef class CUstreamBatchMemOpParams_v1(CUstreamBatchMemOpParams_union):
    pass

cdef class CUstreamBatchMemOpParams(CUstreamBatchMemOpParams_union):
    pass

cdef class CUdevprop_v1(CUdevprop_st):
    pass

cdef class CUdevprop(CUdevprop_st):
    pass

cdef class CUaccessPolicyWindow_v1(CUaccessPolicyWindow_st):
    pass

cdef class CUaccessPolicyWindow(CUaccessPolicyWindow_st):
    pass

cdef class CUDA_KERNEL_NODE_PARAMS_v1(CUDA_KERNEL_NODE_PARAMS_st):
    pass

cdef class CUDA_KERNEL_NODE_PARAMS(CUDA_KERNEL_NODE_PARAMS_st):
    pass

cdef class CUDA_MEMSET_NODE_PARAMS_v1(CUDA_MEMSET_NODE_PARAMS_st):
    pass

cdef class CUDA_MEMSET_NODE_PARAMS(CUDA_MEMSET_NODE_PARAMS_st):
    pass

cdef class CUDA_HOST_NODE_PARAMS_v1(CUDA_HOST_NODE_PARAMS_st):
    pass

cdef class CUDA_HOST_NODE_PARAMS(CUDA_HOST_NODE_PARAMS_st):
    pass

cdef class CUkernelNodeAttrValue_v1(CUkernelNodeAttrValue_union):
    pass

cdef class CUkernelNodeAttrValue(CUkernelNodeAttrValue_union):
    pass

cdef class CUstreamAttrValue_v1(CUstreamAttrValue_union):
    pass

cdef class CUstreamAttrValue(CUstreamAttrValue_union):
    pass

cdef class CUexecAffinitySmCount_v1(CUexecAffinitySmCount_st):
    pass

cdef class CUexecAffinitySmCount(CUexecAffinitySmCount_st):
    pass

cdef class CUexecAffinityParam_v1(CUexecAffinityParam_st):
    pass

cdef class CUexecAffinityParam(CUexecAffinityParam_st):
    pass

cdef class CUDA_MEMCPY2D_v2(CUDA_MEMCPY2D_st):
    pass

cdef class CUDA_MEMCPY2D(CUDA_MEMCPY2D_st):
    pass

cdef class CUDA_MEMCPY3D_v2(CUDA_MEMCPY3D_st):
    pass

cdef class CUDA_MEMCPY3D(CUDA_MEMCPY3D_st):
    pass

cdef class CUDA_MEMCPY3D_PEER_v1(CUDA_MEMCPY3D_PEER_st):
    pass

cdef class CUDA_MEMCPY3D_PEER(CUDA_MEMCPY3D_PEER_st):
    pass

cdef class CUDA_ARRAY_DESCRIPTOR_v2(CUDA_ARRAY_DESCRIPTOR_st):
    pass

cdef class CUDA_ARRAY_DESCRIPTOR(CUDA_ARRAY_DESCRIPTOR_st):
    pass

cdef class CUDA_ARRAY3D_DESCRIPTOR_v2(CUDA_ARRAY3D_DESCRIPTOR_st):
    pass

cdef class CUDA_ARRAY3D_DESCRIPTOR(CUDA_ARRAY3D_DESCRIPTOR_st):
    pass

cdef class CUDA_ARRAY_SPARSE_PROPERTIES_v1(CUDA_ARRAY_SPARSE_PROPERTIES_st):
    pass

cdef class CUDA_ARRAY_SPARSE_PROPERTIES(CUDA_ARRAY_SPARSE_PROPERTIES_st):
    pass

cdef class CUDA_RESOURCE_DESC_v1(CUDA_RESOURCE_DESC_st):
    pass

cdef class CUDA_RESOURCE_DESC(CUDA_RESOURCE_DESC_st):
    pass

cdef class CUDA_TEXTURE_DESC_v1(CUDA_TEXTURE_DESC_st):
    pass

cdef class CUDA_TEXTURE_DESC(CUDA_TEXTURE_DESC_st):
    pass

cdef class CUDA_RESOURCE_VIEW_DESC_v1(CUDA_RESOURCE_VIEW_DESC_st):
    pass

cdef class CUDA_RESOURCE_VIEW_DESC(CUDA_RESOURCE_VIEW_DESC_st):
    pass

cdef class CUDA_POINTER_ATTRIBUTE_P2P_TOKENS_v1(CUDA_POINTER_ATTRIBUTE_P2P_TOKENS_st):
    pass

cdef class CUDA_POINTER_ATTRIBUTE_P2P_TOKENS(CUDA_POINTER_ATTRIBUTE_P2P_TOKENS_st):
    pass

cdef class CUDA_LAUNCH_PARAMS_v1(CUDA_LAUNCH_PARAMS_st):
    pass

cdef class CUDA_LAUNCH_PARAMS(CUDA_LAUNCH_PARAMS_st):
    pass

cdef class CUDA_EXTERNAL_MEMORY_HANDLE_DESC_v1(CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st):
    pass

cdef class CUDA_EXTERNAL_MEMORY_HANDLE_DESC(CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st):
    pass

cdef class CUDA_EXTERNAL_MEMORY_BUFFER_DESC_v1(CUDA_EXTERNAL_MEMORY_BUFFER_DESC_st):
    pass

cdef class CUDA_EXTERNAL_MEMORY_BUFFER_DESC(CUDA_EXTERNAL_MEMORY_BUFFER_DESC_st):
    pass

cdef class CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC_v1(CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC_st):
    pass

cdef class CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC(CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC_st):
    pass

cdef class CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_v1(CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_st):
    pass

cdef class CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC(CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_st):
    pass

cdef class CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_v1(CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st):
    pass

cdef class CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS(CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st):
    pass

cdef class CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_v1(CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st):
    pass

cdef class CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS(CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st):
    pass

cdef class CUDA_EXT_SEM_SIGNAL_NODE_PARAMS_v1(CUDA_EXT_SEM_SIGNAL_NODE_PARAMS_st):
    pass

cdef class CUDA_EXT_SEM_SIGNAL_NODE_PARAMS(CUDA_EXT_SEM_SIGNAL_NODE_PARAMS_st):
    pass

cdef class CUDA_EXT_SEM_WAIT_NODE_PARAMS_v1(CUDA_EXT_SEM_WAIT_NODE_PARAMS_st):
    pass

cdef class CUDA_EXT_SEM_WAIT_NODE_PARAMS(CUDA_EXT_SEM_WAIT_NODE_PARAMS_st):
    pass

cdef class CUarrayMapInfo_v1(CUarrayMapInfo_st):
    pass

cdef class CUarrayMapInfo(CUarrayMapInfo_st):
    pass

cdef class CUmemLocation_v1(CUmemLocation_st):
    pass

cdef class CUmemLocation(CUmemLocation_st):
    pass

cdef class CUmemAllocationProp_v1(CUmemAllocationProp_st):
    pass

cdef class CUmemAllocationProp(CUmemAllocationProp_st):
    pass

cdef class CUmemAccessDesc_v1(CUmemAccessDesc_st):
    pass

cdef class CUmemAccessDesc(CUmemAccessDesc_st):
    pass

cdef class CUmemPoolProps_v1(CUmemPoolProps_st):
    pass

cdef class CUmemPoolProps(CUmemPoolProps_st):
    pass

cdef class CUmemPoolPtrExportData_v1(CUmemPoolPtrExportData_st):
    pass

cdef class CUmemPoolPtrExportData(CUmemPoolPtrExportData_st):
    pass

cdef class CUDA_MEM_ALLOC_NODE_PARAMS(CUDA_MEM_ALLOC_NODE_PARAMS_st):
    pass
