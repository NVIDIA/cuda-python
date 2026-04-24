# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

cimport cuda.bindings.driver as driver
cimport cuda.bindings.cydriver as cydriver
cimport cuda.bindings.cyruntime as cyruntime
from libcpp.vector cimport vector
from cpython.buffer cimport PyBuffer_Release, Py_buffer

cdef class _HelperKernelParams:
    cdef Py_buffer _pybuffer
    cdef bint _pyobj_acquired
    cdef void** _ckernelParams
    cdef char* _ckernelParamsData
    cdef int _length
    cdef bint _malloc_list_created

cdef struct _HelperInputVoidPtrStruct:
    Py_buffer _pybuffer

cdef class _HelperInputVoidPtr:
    cdef _HelperInputVoidPtrStruct _helper
    cdef void* _cptr

cdef void * _helper_input_void_ptr(ptr, _HelperInputVoidPtrStruct *buffer)

cdef inline void * _helper_input_void_ptr_free(_HelperInputVoidPtrStruct *helper):
    if helper[0]._pybuffer.buf != NULL:
        PyBuffer_Release(&helper[0]._pybuffer)



cdef class _HelperCUmemPool_attribute:
    cdef void* _cptr
    cdef cydriver.CUmemPool_attribute_enum _attr
    cdef bint _is_getter

    # Return values
    cdef int _int_val
    cdef driver.cuuint64_t _cuuint64_t_val



cdef class _HelperCUmem_range_attribute:
    cdef void* _cptr
    cdef cydriver.CUmem_range_attribute_enum _attr
    cdef size_t _data_size

    # Return values
    cdef int _int_val # 32 bit integer
    cdef int* _int_val_list # 32 bit integer array



cdef class _HelperCUpointer_attribute:
    cdef void* _cptr
    cdef cydriver.CUpointer_attribute_enum _attr
    cdef bint _is_getter

    # Return values
    cdef driver.CUcontext _ctx
    cdef unsigned int _uint
    cdef int _int
    cdef driver.CUdeviceptr _devptr
    cdef void** _void
    cdef driver.CUDA_POINTER_ATTRIBUTE_P2P_TOKENS _token
    cdef bint _bool
    cdef unsigned long long _ull
    cdef size_t _size
    cdef driver.CUmemoryPool _mempool



cdef class _HelperCUgraphMem_attribute:
    cdef void* _cptr
    cdef cydriver.CUgraphMem_attribute_enum _attr
    cdef bint _is_getter

    # Return values
    cdef driver.cuuint64_t _cuuint64_t_val



cdef class _HelperCUjit_option:
    cdef void* _cptr
    cdef cydriver.CUjit_option_enum _attr

    # Return values
    cdef unsigned int _uint
    cdef float _float
    cdef char* _charstar
    cdef cydriver.CUjit_target_enum _target
    cdef cydriver.CUjit_fallback_enum _fallback
    cdef int _int
    cdef cydriver.CUjit_cacheMode_enum _cacheMode
    cdef vector[char*] _charstarstar # list of names
    cdef _InputVoidPtrPtrHelper _voidstarstar # list of addresses



cdef class _HelperCudaJitOption:
    cdef void* _cptr
    cdef cyruntime.cudaJitOption _attr

    # Return values
    cdef unsigned int _uint
    cdef float _float
    cdef char* _charstar
    cdef cyruntime.cudaJit_Fallback _fallback
    cdef int _int
    cdef cyruntime.cudaJit_CacheMode _cacheMode



cdef class _HelperCUlibraryOption:
    cdef void* _cptr
    cdef cydriver.CUlibraryOption_enum _attr

    # Return values
    cdef unsigned int _uint



cdef class _HelperCudaLibraryOption:
    cdef void* _cptr
    cdef cyruntime.cudaLibraryOption _attr

    # Return values
    cdef unsigned int _uint



cdef class _HelperCUmemAllocationHandleType:
    cdef void* _cptr
    cdef cydriver.CUmemAllocationHandleType_enum _type

    # Return values
    cdef int _int
    cdef void* _handle
    cdef unsigned int _d3dkmt_handle

    cdef driver.CUmemFabricHandle _mem_fabric_handle



cdef class _InputVoidPtrPtrHelper:
    cdef object _references
    cdef void** _cptr



cdef class _HelperCUcoredumpSettings:
    cdef void* _cptr
    cdef cydriver.CUcoredumpSettings_enum _attrib
    cdef bint _is_getter
    cdef size_t _size

    # Return values
    cdef bint _bool
    cdef char* _charstar
