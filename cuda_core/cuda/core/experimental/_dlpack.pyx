# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from enum import IntEnum
from ._layout cimport StridedLayout


cdef void pycapsule_deleter(object capsule) noexcept:
    cdef DLManagedTensor* dlm_tensor
    cdef DLManagedTensorVersioned* dlm_tensor_ver
    # Do not invoke the deleter on a used capsule.
    if cpython.PyCapsule_IsValid(
            capsule, DLPACK_TENSOR_UNUSED_NAME):
        dlm_tensor = <DLManagedTensor*>(
            cpython.PyCapsule_GetPointer(
                capsule, DLPACK_TENSOR_UNUSED_NAME))
        if dlm_tensor.deleter:
            dlm_tensor.deleter(dlm_tensor)
    elif cpython.PyCapsule_IsValid(
            capsule, DLPACK_VERSIONED_TENSOR_UNUSED_NAME):
        dlm_tensor_ver = <DLManagedTensorVersioned*>(
            cpython.PyCapsule_GetPointer(
                capsule, DLPACK_VERSIONED_TENSOR_UNUSED_NAME))
        if dlm_tensor_ver.deleter:
            dlm_tensor_ver.deleter(dlm_tensor_ver)


cdef void deleter(DLManagedTensor* tensor) noexcept with gil:
    stdlib.free(tensor.dl_tensor.shape)
    if tensor.manager_ctx:
        cpython.Py_DECREF(<object>tensor.manager_ctx)
        tensor.manager_ctx = NULL
    stdlib.free(tensor)


cdef void cleanup(DLManagedTensor* tensor) noexcept with gil:
    if tensor:
        if tensor.dl_tensor.shape:
            stdlib.free(tensor.dl_tensor.shape)
        if tensor.manager_ctx:
            cpython.Py_DECREF(<object>tensor.manager_ctx)
            tensor.manager_ctx = NULL
        stdlib.free(tensor)


cdef void versioned_deleter(DLManagedTensorVersioned* tensor) noexcept with gil:
    stdlib.free(tensor.dl_tensor.shape)
    if tensor.manager_ctx:
        cpython.Py_DECREF(<object>tensor.manager_ctx)
        tensor.manager_ctx = NULL
    stdlib.free(tensor)


cdef void cleanup_versioned(DLManagedTensorVersioned* tensor) noexcept with gil:
    if tensor:
        if tensor.dl_tensor.shape:
            stdlib.free(tensor.dl_tensor.shape)
        if tensor.manager_ctx:
            cpython.Py_DECREF(<object>tensor.manager_ctx)
            tensor.manager_ctx = NULL
        stdlib.free(tensor)


cdef inline int _setup_dl_tensor_device(DLTensor* dl_tensor, object buf) except -1:
    cdef DLDevice* device = &dl_tensor.device
    # buf should be a Buffer instance
    if buf.is_device_accessible and not buf.is_host_accessible:
        device.device_type = _kDLCUDA
        device.device_id = buf.device_id
    elif buf.is_device_accessible and buf.is_host_accessible:
        device.device_type = _kDLCUDAHost
        device.device_id = 0
    elif not buf.is_device_accessible and buf.is_host_accessible:
        device.device_type = _kDLCPU
        device.device_id = 0
    else:  # not buf.is_device_accessible and not buf.is_host_accessible
        raise BufferError("invalid buffer")
    return 0


cdef inline int _setup_dl_tensor_layout(DLTensor* dl_tensor, object buf, StridedLayout layout) except -1:
    cdef int64_t* shape_strides = NULL
    cdef int ndim
    if layout is None:
        dl_tensor.ndim = 1
        shape_strides = <int64_t*>stdlib.malloc(sizeof(int64_t) * 2)
        shape_strides[0] = <int64_t>buf.size
        shape_strides[1] = 1  # redundant
        dl_tensor.shape = shape_strides
        dl_tensor.strides = NULL
        dl_tensor.byte_offset = 0
    else:
        ndim = layout.ndim
        dl_tensor.ndim = ndim
        shape_strides = <int64_t*>stdlib.malloc(sizeof(int64_t) * ndim * 2)
        dl_tensor.shape = shape_strides
        for i in range(ndim):
            shape_strides[i] = layout.base.shape[i]
        if layout.base.strides == NULL:
            dl_tensor.strides = NULL
        else:
            dl_tensor.strides = shape_strides + ndim
            for i in range(ndim):
                dl_tensor.strides[i] = layout.base.strides[i]
        dl_tensor.byte_offset = 0
    return 0


cdef inline int _setup_dl_tensor_dtype(DLTensor* dl_tensor, object dtype) except -1:
    cdef DLDataType* dl_dtype = &dl_tensor.dtype
    if dtype is None:
        dl_dtype.code = <uint8_t>kDLInt
        dl_dtype.lanes = <uint16_t>1
        dl_dtype.bits = <uint8_t>8
        return 0
    cdef uint8_t code
    cdef uint8_t bits
    cdef uint16_t lanes
    code, bits, lanes = dtype
    dl_dtype.code = code
    dl_dtype.bits = bits
    dl_dtype.lanes = lanes
    return 0


cpdef object make_py_capsule(
    object buf,
    bint versioned,
    intptr_t data_ptr,
    StridedLayout layout=None,
    object dtype=None,
):
    cdef DLManagedTensor* dlm_tensor = NULL
    cdef DLManagedTensorVersioned* dlm_tensor_ver = NULL
    cdef DLTensor* dl_tensor = NULL
    cdef void* tensor_ptr
    cdef const char* capsule_name

    try:
        if versioned:
            dlm_tensor_ver = <DLManagedTensorVersioned*>(
                stdlib.malloc(sizeof(DLManagedTensorVersioned)))
            dlm_tensor_ver.dl_tensor.shape = NULL
            dl_tensor = &dlm_tensor_ver.dl_tensor
            dlm_tensor_ver.version.major = DLPACK_MAJOR_VERSION
            dlm_tensor_ver.version.minor = DLPACK_MINOR_VERSION
            cpython.Py_INCREF(buf)
            dlm_tensor_ver.manager_ctx = <void*>buf
            dlm_tensor_ver.deleter = versioned_deleter
            dlm_tensor_ver.flags = 0
            tensor_ptr = dlm_tensor_ver
            capsule_name = DLPACK_VERSIONED_TENSOR_UNUSED_NAME
        else:
            dlm_tensor = <DLManagedTensor*>(
                stdlib.malloc(sizeof(DLManagedTensor)))
            dlm_tensor.dl_tensor.shape = NULL
            dl_tensor = &dlm_tensor.dl_tensor
            cpython.Py_INCREF(buf)
            dlm_tensor.manager_ctx = <void*>buf
            dlm_tensor.deleter = deleter
            tensor_ptr = dlm_tensor
            capsule_name = DLPACK_TENSOR_UNUSED_NAME

        dl_tensor.data = <void*>data_ptr

        _setup_dl_tensor_device(dl_tensor, buf)
        _setup_dl_tensor_layout(dl_tensor, buf, layout)
        _setup_dl_tensor_dtype(dl_tensor, dtype)

        return cpython.PyCapsule_New(tensor_ptr, capsule_name, pycapsule_deleter)
    except:
        cleanup(dlm_tensor)
        cleanup_versioned(dlm_tensor_ver)
        raise


class DLDeviceType(IntEnum):
    kDLCPU = _kDLCPU
    kDLCUDA = _kDLCUDA
    kDLCUDAHost = _kDLCUDAHost
    kDLCUDAManaged = _kDLCUDAManaged
