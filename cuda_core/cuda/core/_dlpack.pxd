# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

cimport cpython

from libc cimport stdlib
from libc.stdint cimport uint8_t
from libc.stdint cimport uint16_t
from libc.stdint cimport uint32_t
from libc.stdint cimport int32_t
from libc.stdint cimport int64_t
from libc.stdint cimport uint64_t
from libc.stdint cimport intptr_t


cdef extern from "_include/dlpack.h" nogil:
    """
    #define DLPACK_TENSOR_UNUSED_NAME "dltensor"
    #define DLPACK_VERSIONED_TENSOR_UNUSED_NAME "dltensor_versioned"
    #define DLPACK_TENSOR_USED_NAME "used_dltensor"
    #define DLPACK_VERSIONED_TENSOR_USED_NAME "used_dltensor_versioned"
    """
    ctypedef enum _DLDeviceType "DLDeviceType":
        _kDLCPU "kDLCPU"
        _kDLCUDA "kDLCUDA"
        _kDLCUDAHost "kDLCUDAHost"
        _kDLCUDAManaged "kDLCUDAManaged"
        _kDLTrn "kDLTrn"

    ctypedef struct DLDevice:
        _DLDeviceType device_type
        int32_t device_id

    cdef enum DLDataTypeCode:
        kDLInt
        kDLUInt
        kDLFloat
        kDLBfloat
        kDLComplex
        kDLBool

    ctypedef struct DLDataType:
        uint8_t code
        uint8_t bits
        uint16_t lanes

    ctypedef struct DLTensor:
        void* data
        DLDevice device
        int32_t ndim
        DLDataType dtype
        int64_t* shape
        int64_t* strides
        uint64_t byte_offset

    ctypedef struct DLManagedTensor:
        DLTensor dl_tensor
        void* manager_ctx
        void (*deleter)(DLManagedTensor*)

    ctypedef struct DLPackVersion:
        uint32_t major
        uint32_t minor

    ctypedef struct DLManagedTensorVersioned:
        DLPackVersion version
        void* manager_ctx
        void (*deleter)(DLManagedTensorVersioned*)
        uint64_t flags
        DLTensor dl_tensor

    int DLPACK_MAJOR_VERSION
    int DLPACK_MINOR_VERSION
    int DLPACK_FLAG_BITMASK_READ_ONLY
    int DLPACK_FLAG_BITMASK_IS_COPIED
    int DLPACK_FLAG_BITMASK_IS_SUBBYTE_TYPE_PADDED

    const char* DLPACK_TENSOR_UNUSED_NAME
    const char* DLPACK_VERSIONED_TENSOR_UNUSED_NAME
    const char* DLPACK_TENSOR_USED_NAME
    const char* DLPACK_VERSIONED_TENSOR_USED_NAME


cdef extern from "_include/dlpack.h":
    ctypedef int (*DLPackManagedTensorAllocator)(
        DLTensor* prototype,
        DLManagedTensorVersioned** out,
        void* error_ctx,
        void (*SetError)(void* error_ctx, const char* kind, const char* message) noexcept
    )

    ctypedef int (*DLPackManagedTensorFromPyObjectNoSync)(
        void* py_object,
        DLManagedTensorVersioned** out
    )

    ctypedef int (*DLPackManagedTensorToPyObjectNoSync)(
        DLManagedTensorVersioned* tensor,
        void** out_py_object
    )

    ctypedef int (*DLPackDLTensorFromPyObjectNoSync)(
        void* py_object,
        DLTensor* out
    )

    ctypedef int (*DLPackCurrentWorkStream)(
        _DLDeviceType device_type,
        int32_t device_id,
        void** out_current_stream
    )

    ctypedef struct DLPackExchangeAPIHeader:
        DLPackVersion version
        DLPackExchangeAPIHeader* prev_api

    ctypedef struct DLPackExchangeAPI:
        DLPackExchangeAPIHeader header
        DLPackManagedTensorAllocator managed_tensor_allocator
        DLPackManagedTensorFromPyObjectNoSync managed_tensor_from_py_object_no_sync
        DLPackManagedTensorToPyObjectNoSync managed_tensor_to_py_object_no_sync
        DLPackDLTensorFromPyObjectNoSync dltensor_from_py_object_no_sync
        DLPackCurrentWorkStream current_work_stream
