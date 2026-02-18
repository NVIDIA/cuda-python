# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from enum import IntEnum


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
    if tensor:
        if tensor.dl_tensor.shape:
            stdlib.free(tensor.dl_tensor.shape)
        if tensor.manager_ctx:
            cpython.Py_DECREF(<object>tensor.manager_ctx)
            tensor.manager_ctx = NULL
        stdlib.free(tensor)


cdef void versioned_deleter(DLManagedTensorVersioned* tensor) noexcept with gil:
    if tensor:
        if tensor.dl_tensor.shape:
            stdlib.free(tensor.dl_tensor.shape)
        if tensor.manager_ctx:
            cpython.Py_DECREF(<object>tensor.manager_ctx)
            tensor.manager_ctx = NULL
        stdlib.free(tensor)


cdef inline DLManagedTensorVersioned* allocate_dlm_tensor_versioned() except? NULL:
    cdef DLManagedTensorVersioned* dlm_tensor_ver = NULL
    try:
        dlm_tensor_ver = <DLManagedTensorVersioned*>(
            stdlib.malloc(sizeof(DLManagedTensorVersioned)))
        dlm_tensor_ver.dl_tensor.shape = NULL
        dlm_tensor_ver.manager_ctx = NULL
        return dlm_tensor_ver
    except:
        if dlm_tensor_ver:
            stdlib.free(dlm_tensor_ver)
        raise


cdef inline DLManagedTensor* allocate_dlm_tensor() except? NULL:
    cdef DLManagedTensor* dlm_tensor = NULL
    try:
        dlm_tensor = <DLManagedTensor*>(
            stdlib.malloc(sizeof(DLManagedTensor)))
        dlm_tensor.dl_tensor.shape = NULL
        dlm_tensor.manager_ctx = NULL
        return dlm_tensor
    except:
        if dlm_tensor:
            stdlib.free(dlm_tensor)
        raise


cdef inline int setup_dl_tensor_layout(DLTensor* dl_tensor, object buf) except -1:
    dl_tensor.ndim = 1
    cdef int64_t* shape_strides = \
        <int64_t*>stdlib.malloc(sizeof(int64_t) * 2)
    if shape_strides == NULL:
        raise MemoryError()
    # DLPack v1.2+ requires non-NULL strides for ndim != 0.
    shape_strides[0] = <int64_t>buf.size
    shape_strides[1] = 1
    dl_tensor.shape = shape_strides
    dl_tensor.strides = shape_strides + 1
    dl_tensor.byte_offset = 0
    return 0


cdef inline int setup_dl_tensor_device(DLTensor* dl_tensor, object buf) except -1:
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


cdef inline int setup_dl_tensor_dtype(DLTensor* dl_tensor) except -1 nogil:
    cdef DLDataType* dtype = &dl_tensor.dtype
    dtype.code = <uint8_t>kDLInt
    dtype.lanes = <uint16_t>1
    dtype.bits = <uint8_t>8
    return 0


cpdef object make_py_capsule(object buf, bint versioned):
    cdef DLManagedTensor* dlm_tensor = NULL
    cdef DLManagedTensorVersioned* dlm_tensor_ver = NULL
    cdef DLTensor* dl_tensor
    cdef void* tensor_ptr
    cdef const char* capsule_name
    cdef object ret = None

    try:
        if versioned:
            dlm_tensor_ver = allocate_dlm_tensor_versioned()
            # Transfer the reference to manager_ctx
            cpython.Py_INCREF(buf)
            dlm_tensor_ver.manager_ctx = <void*>buf
            dlm_tensor_ver.deleter = versioned_deleter
            dlm_tensor_ver.version.major = DLPACK_MAJOR_VERSION
            dlm_tensor_ver.version.minor = DLPACK_MINOR_VERSION
            dlm_tensor_ver.flags = 0
            dl_tensor = &dlm_tensor_ver.dl_tensor
            tensor_ptr = dlm_tensor_ver
            capsule_name = DLPACK_VERSIONED_TENSOR_UNUSED_NAME
        else:
            dlm_tensor = allocate_dlm_tensor()
            # Transfer the reference to manager_ctx
            cpython.Py_INCREF(buf)
            dlm_tensor.manager_ctx = <void*>buf
            dlm_tensor.deleter = deleter
            dl_tensor = &dlm_tensor.dl_tensor
            tensor_ptr = dlm_tensor
            capsule_name = DLPACK_TENSOR_UNUSED_NAME

        dl_tensor.data = <void*><intptr_t>(int(buf.handle))
        setup_dl_tensor_layout(dl_tensor, buf)
        setup_dl_tensor_device(dl_tensor, buf)
        setup_dl_tensor_dtype(dl_tensor)
        ret = cpython.PyCapsule_New(tensor_ptr, capsule_name, pycapsule_deleter)
    except:
        if ret is None:
            deleter(dlm_tensor)
            versioned_deleter(dlm_tensor_ver)
        raise
    return ret


class DLDeviceType(IntEnum):
    kDLCPU = _kDLCPU
    kDLCUDA = _kDLCUDA
    kDLCUDAHost = _kDLCUDAHost
    kDLCUDAManaged = _kDLCUDAManaged
