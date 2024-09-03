# distutils: language = c++

cimport cpython  # NOQA

from libc cimport stdlib
from libc.stdint cimport uint8_t
from libc.stdint cimport uint16_t
from libc.stdint cimport int32_t
from libc.stdint cimport int64_t
from libc.stdint cimport uint64_t
from libc.stdint cimport intptr_t
from libcpp.vector cimport vector

from enum import IntEnum


cdef extern from "dlpack.h" nogil:

    ctypedef enum _DLDeviceType "DLDeviceType":
        _kDLCPU "kDLCPU"
        _kDLCUDA "kDLCUDA"
        _kDLCUDAHost "kDLCUDAHost"
        _kDLCUDAManaged "kDLCUDAManaged"

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
        void (*deleter)(DLManagedTensor*)  # noqa: E211


cdef void pycapsule_deleter(object dltensor):
    cdef DLManagedTensor* dlm_tensor
    # Do not invoke the deleter on a used capsule
    if cpython.PyCapsule_IsValid(dltensor, 'dltensor'):
        dlm_tensor = <DLManagedTensor*>cpython.PyCapsule_GetPointer(
            dltensor, 'dltensor')
        dlm_tensor.deleter(dlm_tensor)


cdef void deleter(DLManagedTensor* tensor) with gil:
    if tensor.manager_ctx is NULL:
        return
    stdlib.free(tensor.dl_tensor.shape)
    cpython.Py_DECREF(<object>tensor.manager_ctx)
    tensor.manager_ctx = NULL
    stdlib.free(tensor)


cpdef object make_py_capsule(object buf) except +:
    cdef DLManagedTensor* dlm_tensor = \
        <DLManagedTensor*>stdlib.malloc(sizeof(DLManagedTensor))

    cdef DLTensor* dl_tensor = &dlm_tensor.dl_tensor
    dl_tensor.data = <void*><intptr_t>(int(buf.handle))
    dl_tensor.ndim = 1

    cdef int64_t* shape_strides = \
        <int64_t*>stdlib.malloc(sizeof(int64_t) * 2)
    shape_strides[0] = <int64_t>buf.size
    shape_strides[1] = 1  # redundant
    dl_tensor.shape = shape_strides
    dl_tensor.strides = NULL
    dl_tensor.byte_offset = 0

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

    cdef DLDataType* dtype = &dl_tensor.dtype
    dtype.code = <uint8_t>kDLInt
    dtype.lanes = <uint16_t>1
    dtype.bits = <uint8_t>8

    dlm_tensor.manager_ctx = <void*>buf
    cpython.Py_INCREF(buf)
    dlm_tensor.deleter = deleter

    return cpython.PyCapsule_New(dlm_tensor, 'dltensor', pycapsule_deleter)


class DLDeviceType(IntEnum):
    kDLCPU = _kDLCPU
    kDLCUDA = _kDLCUDA
    kDLCUDAHost = _kDLCUDAHost
    kDLCUDAManaged = _kDLCUDAManaged
