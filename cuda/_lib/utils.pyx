# Copyright 2021-2022 NVIDIA Corporation.  All rights reserved.
#
# Please refer to the NVIDIA end user license agreement (EULA) associated
# with this source code for terms and conditions that govern your use of
# this software. Any use, reproduction, disclosure, or distribution of
# this software and related documentation outside the terms of the EULA
# is strictly prohibited.
from cpython.buffer cimport PyObject_CheckBuffer, PyObject_GetBuffer, PyBuffer_Release, PyBUF_SIMPLE, PyBUF_ANY_CONTIGUOUS
from libc.stdlib cimport calloc, free
from libc.stdint cimport int32_t, uint32_t, int64_t, uint64_t
from libc.stddef cimport wchar_t
from libc.string cimport memcpy
from enum import Enum
from typing import List, Tuple
import ctypes
cimport cuda.ccuda as ccuda
import cuda.cuda as cuda
cimport cuda._lib.param_packer as param_packer

ctypedef unsigned long long void_ptr

cdef void* callocWrapper(length, size):
    cdef void* out = calloc(length, size)
    if out is NULL:
        raise MemoryError('Failed to allocated length x size memory: {}x{}'.format(length, size))
    return out

cdef class HelperKernelParams:
    supported_types = { # excluding void_p and None, which are handled specially
        ctypes.c_bool,
        ctypes.c_char,
        ctypes.c_wchar,
        ctypes.c_byte,
        ctypes.c_ubyte,
        ctypes.c_short,
        ctypes.c_ushort,
        ctypes.c_int,
        ctypes.c_uint,
        ctypes.c_long,
        ctypes.c_ulong,
        ctypes.c_longlong,
        ctypes.c_ulonglong,
        ctypes.c_size_t,
        ctypes.c_float,
        ctypes.c_double
    }

    max_param_size = max(ctypes.sizeof(max(HelperKernelParams.supported_types, key=lambda t:ctypes.sizeof(t))), sizeof(void_ptr))

    def __cinit__(self, kernelParams):
        self._pyobj_acquired = False
        self._malloc_list_created = False
        if kernelParams is None:
            self._ckernelParams = NULL
        elif isinstance(kernelParams, (int)):
            # Easy run, user gave us an already configured void** address
            self._ckernelParams = <void**><void_ptr>kernelParams
        elif PyObject_CheckBuffer(kernelParams):
            # Easy run, get address from Python Buffer Protocol
            err_buffer = PyObject_GetBuffer(kernelParams, &self._pybuffer, PyBUF_SIMPLE | PyBUF_ANY_CONTIGUOUS)
            if err_buffer == -1:
                raise RuntimeError("Argument 'kernelParams' failed to retrieve buffer through Buffer Protocol")
            self._pyobj_acquired = True
            self._ckernelParams = <void**><void_ptr>self._pybuffer.buf
        elif isinstance(kernelParams, (Tuple)) and len(kernelParams) == 2 and isinstance(kernelParams[0], (Tuple)) and isinstance(kernelParams[1], (Tuple)):
            # Hard run, construct and fill out contigues memory using provided kernel values and types based
            if len(kernelParams[0]) != len(kernelParams[1]):
                raise TypeError("Argument 'kernelParams' has tuples with different length")
            if len(kernelParams[0]) != 0:
                self._length = len(kernelParams[0])
                self._ckernelParams = <void**>callocWrapper(len(kernelParams[0]), sizeof(void*))
                self._ckernelParamsData = <char*>callocWrapper(len(kernelParams[0]), HelperKernelParams.max_param_size)
                self._malloc_list_created = True

            idx = 0
            data_idx = 0
            for value, ctype in zip(kernelParams[0], kernelParams[1]):
                if ctype is None:
                    # special cases for None
                    if callable(getattr(value, 'getPtr', None)):
                        self._ckernelParams[idx] = <void*><void_ptr>value.getPtr()
                    elif isinstance(value, (ctypes.Structure)):
                        self._ckernelParams[idx] = <void*><void_ptr>ctypes.addressof(value)
                    elif isinstance(value, (Enum)):
                        self._ckernelParams[idx] = &(self._ckernelParamsData[data_idx])
                        (<int*>self._ckernelParams[idx])[0] = value.value
                        data_idx += sizeof(int)
                    else:
                        raise TypeError("Provided argument is of type {} but expected Type {}, {} or CUDA Binding structure with getPtr() attribute".format(type(value), type(ctypes.Structure), type(ctypes.c_void_p)))
                elif ctype in HelperKernelParams.supported_types:
                    self._ckernelParams[idx] = &(self._ckernelParamsData[data_idx])

                    # handle case where a float is passed as a double
                    if ctype == ctypes.c_double and isinstance(value, ctypes.c_float):
                        value = ctype(value.value)
                    if not isinstance(value, ctype): # make it a ctype
                        size = param_packer.feed(self._ckernelParams[idx], value, ctype)
                        if size == 0: # feed failed
                            value = ctype(value)
                            size = ctypes.sizeof(ctype)
                            addr = <void*>(<void_ptr>ctypes.addressof(value))
                            memcpy(self._ckernelParams[idx], addr, size)
                    else:
                        size = ctypes.sizeof(ctype)
                        addr = <void*>(<void_ptr>ctypes.addressof(value))
                        memcpy(self._ckernelParams[idx], addr, size)
                    data_idx += size
                elif ctype == ctypes.c_void_p:
                    # special cases for void_p
                    if isinstance(value, (int, ctypes.c_void_p)):
                        self._ckernelParams[idx] = &(self._ckernelParamsData[data_idx])
                        (<void_ptr*>self._ckernelParams[idx])[0] = value.value if isinstance(value, (ctypes.c_void_p)) else value
                        data_idx += sizeof(void_ptr)
                    elif callable(getattr(value, 'getPtr', None)):
                        self._ckernelParams[idx] = &(self._ckernelParamsData[data_idx])
                        (<void_ptr*>self._ckernelParams[idx])[0] = value.getPtr()
                        data_idx += sizeof(void_ptr)
                    else:
                        raise TypeError("Provided argument is of type {} but expected Type {}, {} or CUDA Binding structure with getPtr() attribute".format(type(value), type(int), type(ctypes.c_void_p)))
                else:
                    raise TypeError("Unsupported type: " + str(type(ctype)))
                idx += 1
        else:
            raise TypeError("Argument 'kernelParams' is not a valid type: Tuple[Tuple[Any, ...], Tuple[Any, ...]] or PyObject implimenting Buffer Protocol or Int")

    def __dealloc__(self):
        if self._pyobj_acquired is True:
            PyBuffer_Release(&self._pybuffer)
        if self._malloc_list_created is True:
            free(self._ckernelParams)
            free(self._ckernelParamsData)

    @property
    def ckernelParams(self):
        return <void_ptr>self._ckernelParams

cdef class HelperInputVoidPtr:
    def __cinit__(self, ptr):
        self._pyobj_acquired = False
        if ptr is None:
            self._cptr = NULL
        elif isinstance(ptr, (int)):
            # Easy run, user gave us an already configured void** address
            self._cptr = <void*><void_ptr>ptr
        elif PyObject_CheckBuffer(ptr):
            # Easy run, get address from Python Buffer Protocol
            err_buffer = PyObject_GetBuffer(ptr, &self._pybuffer, PyBUF_SIMPLE | PyBUF_ANY_CONTIGUOUS)
            if err_buffer == -1:
                raise RuntimeError("Failed to retrieve buffer through Buffer Protocol")
            self._pyobj_acquired = True
            self._cptr = <void*><void_ptr>self._pybuffer.buf
        else:
            raise TypeError("Provided argument is of type {} but expected Type {}, {} or object with Buffer Protocol".format(type(ptr), type(None), type(int)))

    def __dealloc__(self):
        if self._pyobj_acquired is True:
            PyBuffer_Release(&self._pybuffer)

    @property
    def cptr(self):
        return <void_ptr>self._cptr

cdef class HelperCUmemPool_attribute:
    def __cinit__(self, attr, init_value, is_getter=False):
        self._is_getter = is_getter
        self._attr = attr.value
        if self._attr in (ccuda.CUmemPool_attribute_enum.CU_MEMPOOL_ATTR_REUSE_FOLLOW_EVENT_DEPENDENCIES,
                          ccuda.CUmemPool_attribute_enum.CU_MEMPOOL_ATTR_REUSE_ALLOW_OPPORTUNISTIC,
                          ccuda.CUmemPool_attribute_enum.CU_MEMPOOL_ATTR_REUSE_ALLOW_INTERNAL_DEPENDENCIES):
            self._int_val = init_value
            self._cptr = <void*>&self._int_val
        elif self._attr in (ccuda.CUmemPool_attribute_enum.CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
                            ccuda.CUmemPool_attribute_enum.CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT,
                            ccuda.CUmemPool_attribute_enum.CU_MEMPOOL_ATTR_RESERVED_MEM_HIGH,
                            ccuda.CUmemPool_attribute_enum.CU_MEMPOOL_ATTR_USED_MEM_CURRENT,
                            ccuda.CUmemPool_attribute_enum.CU_MEMPOOL_ATTR_USED_MEM_HIGH):
            if self._is_getter:
                self._cuuint64_t_val = cuda.cuuint64_t()
                self._cptr = <void*><void_ptr>self._cuuint64_t_val.getPtr()
            else:
                self._cptr = <void*><void_ptr>init_value.getPtr()
        else:
            raise TypeError('Unsupported attribute: {}'.format(attr.name))

    def __dealloc__(self):
        pass

    @property
    def cptr(self):
        return <void_ptr>self._cptr

    def pyObj(self):
        assert(self._is_getter == True)
        if self._attr in (ccuda.CUmemPool_attribute_enum.CU_MEMPOOL_ATTR_REUSE_FOLLOW_EVENT_DEPENDENCIES,
                          ccuda.CUmemPool_attribute_enum.CU_MEMPOOL_ATTR_REUSE_ALLOW_OPPORTUNISTIC,
                          ccuda.CUmemPool_attribute_enum.CU_MEMPOOL_ATTR_REUSE_ALLOW_INTERNAL_DEPENDENCIES):
            return self._int_val
        elif self._attr in (ccuda.CUmemPool_attribute_enum.CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
                            ccuda.CUmemPool_attribute_enum.CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT,
                            ccuda.CUmemPool_attribute_enum.CU_MEMPOOL_ATTR_RESERVED_MEM_HIGH,
                            ccuda.CUmemPool_attribute_enum.CU_MEMPOOL_ATTR_USED_MEM_CURRENT,
                            ccuda.CUmemPool_attribute_enum.CU_MEMPOOL_ATTR_USED_MEM_HIGH):
            return self._cuuint64_t_val
        else:
            raise TypeError('Unsupported attribute value: {}'.format(self._attr))

cdef class HelperCUmem_range_attribute:
    def __cinit__(self, attr, data_size):
        self._data_size = data_size
        self._attr = attr.value
        if self._attr in (ccuda.CUmem_range_attribute_enum.CU_MEM_RANGE_ATTRIBUTE_READ_MOSTLY,
                          ccuda.CUmem_range_attribute_enum.CU_MEM_RANGE_ATTRIBUTE_PREFERRED_LOCATION,
                          ccuda.CUmem_range_attribute_enum.CU_MEM_RANGE_ATTRIBUTE_LAST_PREFETCH_LOCATION):
            self._cptr = <void*>&self._int_val
        elif self._attr in (ccuda.CUmem_range_attribute_enum.CU_MEM_RANGE_ATTRIBUTE_ACCESSED_BY,):
            self._cptr = callocWrapper(1, self._data_size)
            self._int_val_list = <int*>self._cptr
        else:
            raise TypeError('Unsupported attribute: {}'.format(attr.name))

    def __dealloc__(self):
        if self._attr in (ccuda.CUmem_range_attribute_enum.CU_MEM_RANGE_ATTRIBUTE_ACCESSED_BY,):
            free(self._cptr)

    @property
    def cptr(self):
        return <void_ptr>self._cptr

    def pyObj(self):
        if self._attr in (ccuda.CUmem_range_attribute_enum.CU_MEM_RANGE_ATTRIBUTE_READ_MOSTLY,
                          ccuda.CUmem_range_attribute_enum.CU_MEM_RANGE_ATTRIBUTE_PREFERRED_LOCATION,
                          ccuda.CUmem_range_attribute_enum.CU_MEM_RANGE_ATTRIBUTE_LAST_PREFETCH_LOCATION):
            return self._int_val
        elif self._attr in (ccuda.CUmem_range_attribute_enum.CU_MEM_RANGE_ATTRIBUTE_ACCESSED_BY,):
            return [self._int_val_list[idx] for idx in range(int(self._data_size/4))]
        else:
            raise TypeError('Unsupported attribute value: {}'.format(self._attr))

cdef class HelperCUpointer_attribute:
    def __cinit__(self, attr, init_value, is_getter=False):
        self._is_getter = is_getter
        self._attr = attr.value
        if self._attr in (ccuda.CUpointer_attribute_enum.CU_POINTER_ATTRIBUTE_CONTEXT,):
            if self._is_getter:
                self._ctx = cuda.CUcontext()
                self._cptr = <void*><void_ptr>self._ctx.getPtr()
            else:
                self._cptr = <void*><void_ptr>init_value.getPtr()
        elif self._attr in (ccuda.CUpointer_attribute_enum.CU_POINTER_ATTRIBUTE_MEMORY_TYPE,
                            ccuda.CUpointer_attribute_enum.CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL,
                            ccuda.CUpointer_attribute_enum.CU_POINTER_ATTRIBUTE_ALLOWED_HANDLE_TYPES,
                            ccuda.CUpointer_attribute_enum.CU_POINTER_ATTRIBUTE_IS_GPU_DIRECT_RDMA_CAPABLE,
                            ccuda.CUpointer_attribute_enum.CU_POINTER_ATTRIBUTE_ACCESS_FLAGS):
            self._uint = init_value
            self._cptr = <void*>&self._uint
        elif self._attr in (ccuda.CUpointer_attribute_enum.CU_POINTER_ATTRIBUTE_DEVICE_POINTER,
                            ccuda.CUpointer_attribute_enum.CU_POINTER_ATTRIBUTE_RANGE_START_ADDR):
            if self._is_getter:
                self._devptr = cuda.CUdeviceptr()
                self._cptr = <void*><void_ptr>self._devptr.getPtr()
            else:
                self._cptr = <void*><void_ptr>init_value.getPtr()
        elif self._attr in (ccuda.CUpointer_attribute_enum.CU_POINTER_ATTRIBUTE_HOST_POINTER,):
            self._void = <void**><void_ptr>init_value
            self._cptr = <void*>&self._void
        elif self._attr in (ccuda.CUpointer_attribute_enum.CU_POINTER_ATTRIBUTE_P2P_TOKENS,):
            if self._is_getter:
                self._token = cuda.CUDA_POINTER_ATTRIBUTE_P2P_TOKENS()
                self._cptr = <void*><void_ptr>self._token.getPtr()
            else:
                self._cptr = <void*><void_ptr>init_value.getPtr()
        elif self._attr in (ccuda.CUpointer_attribute_enum.CU_POINTER_ATTRIBUTE_SYNC_MEMOPS,
                            ccuda.CUpointer_attribute_enum.CU_POINTER_ATTRIBUTE_IS_MANAGED,
                            ccuda.CUpointer_attribute_enum.CU_POINTER_ATTRIBUTE_IS_LEGACY_CUDA_IPC_CAPABLE,
                            ccuda.CUpointer_attribute_enum.CU_POINTER_ATTRIBUTE_MAPPED):
            self._bool = init_value
            self._cptr = <void*>&self._bool
        elif self._attr in (ccuda.CUpointer_attribute_enum.CU_POINTER_ATTRIBUTE_BUFFER_ID,):
            self._ull = init_value
            self._cptr = <void*>&self._ull
        elif self._attr in (ccuda.CUpointer_attribute_enum.CU_POINTER_ATTRIBUTE_RANGE_SIZE,):
            self._size = init_value
            self._cptr = <void*>&self._size
        elif self._attr in (ccuda.CUpointer_attribute_enum.CU_POINTER_ATTRIBUTE_MEMPOOL_HANDLE,):
            if self._is_getter:
                self._mempool = cuda.CUmemoryPool()
                self._cptr = <void*><void_ptr>self._mempool.getPtr()
            else:
                self._cptr = <void*><void_ptr>init_value.getPtr()
        else:
            raise TypeError('Unsupported attribute: {}'.format(attr.name))

    def __dealloc__(self):
        pass

    @property
    def cptr(self):
        return <void_ptr>self._cptr

    def pyObj(self):
        assert(self._is_getter == True)
        if self._attr in (ccuda.CUpointer_attribute_enum.CU_POINTER_ATTRIBUTE_CONTEXT,):
            return self._ctx
        elif self._attr in (ccuda.CUpointer_attribute_enum.CU_POINTER_ATTRIBUTE_MEMORY_TYPE,
                            ccuda.CUpointer_attribute_enum.CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL,
                            ccuda.CUpointer_attribute_enum.CU_POINTER_ATTRIBUTE_ALLOWED_HANDLE_TYPES,
                            ccuda.CUpointer_attribute_enum.CU_POINTER_ATTRIBUTE_IS_GPU_DIRECT_RDMA_CAPABLE,
                            ccuda.CUpointer_attribute_enum.CU_POINTER_ATTRIBUTE_ACCESS_FLAGS):
            return self._uint
        elif self._attr in (ccuda.CUpointer_attribute_enum.CU_POINTER_ATTRIBUTE_DEVICE_POINTER,
                            ccuda.CUpointer_attribute_enum.CU_POINTER_ATTRIBUTE_RANGE_START_ADDR):
            return self._devptr
        elif self._attr in (ccuda.CUpointer_attribute_enum.CU_POINTER_ATTRIBUTE_HOST_POINTER,):
            return <void_ptr>self._void
        elif self._attr in (ccuda.CUpointer_attribute_enum.CU_POINTER_ATTRIBUTE_P2P_TOKENS,):
            return self._token
        elif self._attr in (ccuda.CUpointer_attribute_enum.CU_POINTER_ATTRIBUTE_SYNC_MEMOPS,
                            ccuda.CUpointer_attribute_enum.CU_POINTER_ATTRIBUTE_IS_MANAGED,
                            ccuda.CUpointer_attribute_enum.CU_POINTER_ATTRIBUTE_IS_LEGACY_CUDA_IPC_CAPABLE,
                            ccuda.CUpointer_attribute_enum.CU_POINTER_ATTRIBUTE_MAPPED):
            return self._bool
        elif self._attr in (ccuda.CUpointer_attribute_enum.CU_POINTER_ATTRIBUTE_BUFFER_ID,):
            return self._ull
        elif self._attr in (ccuda.CUpointer_attribute_enum.CU_POINTER_ATTRIBUTE_RANGE_SIZE,):
            return self._size
        elif self._attr in (ccuda.CUpointer_attribute_enum.CU_POINTER_ATTRIBUTE_MEMPOOL_HANDLE,):
            return self._mempool
        else:
            raise TypeError('Unsupported attribute value: {}'.format(self._attr))

cdef class HelperCUgraphMem_attribute:
    def __cinit__(self, attr, init_value, is_getter=False):
        self._is_getter = is_getter
        self._attr = attr.value
        if self._attr in (ccuda.CUgraphMem_attribute_enum.CU_GRAPH_MEM_ATTR_USED_MEM_CURRENT,
                          ccuda.CUgraphMem_attribute_enum.CU_GRAPH_MEM_ATTR_USED_MEM_HIGH,
                          ccuda.CUgraphMem_attribute_enum.CU_GRAPH_MEM_ATTR_RESERVED_MEM_CURRENT,
                          ccuda.CUgraphMem_attribute_enum.CU_GRAPH_MEM_ATTR_RESERVED_MEM_HIGH):
            if self._is_getter:
                self._cuuint64_t_val = cuda.cuuint64_t()
                self._cptr = <void*><void_ptr>self._cuuint64_t_val.getPtr()
            else:
                self._cptr = <void*><void_ptr>init_value.getPtr()
        else:
            raise TypeError('Unsupported attribute: {}'.format(attr.name))

    def __dealloc__(self):
        pass

    @property
    def cptr(self):
        return <void_ptr>self._cptr

    def pyObj(self):
        assert(self._is_getter == True)
        if self._attr in (ccuda.CUgraphMem_attribute_enum.CU_GRAPH_MEM_ATTR_USED_MEM_CURRENT,
                          ccuda.CUgraphMem_attribute_enum.CU_GRAPH_MEM_ATTR_USED_MEM_HIGH,
                          ccuda.CUgraphMem_attribute_enum.CU_GRAPH_MEM_ATTR_RESERVED_MEM_CURRENT,
                          ccuda.CUgraphMem_attribute_enum.CU_GRAPH_MEM_ATTR_RESERVED_MEM_HIGH):
            return self._cuuint64_t_val
        else:
            raise TypeError('Unsupported attribute value: {}'.format(self._attr))

cdef class HelperCUjit_option:
    def __cinit__(self, attr, init_value):
        self._attr = attr.value
        if self._attr in (ccuda.CUjit_option_enum.CU_JIT_MAX_REGISTERS,
                          ccuda.CUjit_option_enum.CU_JIT_THREADS_PER_BLOCK,
                          ccuda.CUjit_option_enum.CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES,
                          ccuda.CUjit_option_enum.CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
                          ccuda.CUjit_option_enum.CU_JIT_OPTIMIZATION_LEVEL,
                          ccuda.CUjit_option_enum.CU_JIT_GLOBAL_SYMBOL_COUNT,
                          ccuda.CUjit_option_enum.CU_JIT_TARGET_FROM_CUCONTEXT,):
            self._uint = init_value
            self._cptr = <void*><void_ptr>self._uint
        elif self._attr in (ccuda.CUjit_option_enum.CU_JIT_WALL_TIME,):
            self._float = init_value
            self._cptr = <void*><void_ptr>self._float
        elif self._attr in (ccuda.CUjit_option_enum.CU_JIT_INFO_LOG_BUFFER,
                            ccuda.CUjit_option_enum.CU_JIT_ERROR_LOG_BUFFER):
            self._charstar = init_value
            self._cptr = <void*><void_ptr>self._charstar
        elif self._attr in (ccuda.CUjit_option_enum.CU_JIT_TARGET,):
            self._target = init_value.value
            self._cptr = <void*><void_ptr>self._target
        elif self._attr in (ccuda.CUjit_option_enum.CU_JIT_FALLBACK_STRATEGY,):
            self._fallback = init_value.value
            self._cptr = <void*><void_ptr>self._fallback
        elif self._attr in (ccuda.CUjit_option_enum.CU_JIT_GENERATE_DEBUG_INFO,
                            ccuda.CUjit_option_enum.CU_JIT_LOG_VERBOSE,
                            ccuda.CUjit_option_enum.CU_JIT_GENERATE_LINE_INFO,
                            ccuda.CUjit_option_enum.CU_JIT_LTO,
                            ccuda.CUjit_option_enum.CU_JIT_FTZ,
                            ccuda.CUjit_option_enum.CU_JIT_PREC_DIV,
                            ccuda.CUjit_option_enum.CU_JIT_PREC_SQRT,
                            ccuda.CUjit_option_enum.CU_JIT_FMA):
            self._int = init_value
            self._cptr = <void*><void_ptr>self._int
        elif self._attr in (ccuda.CUjit_option_enum.CU_JIT_CACHE_MODE,):
            self._cacheMode = init_value.value
            self._cptr = <void*><void_ptr>self._cacheMode
        elif self._attr in (ccuda.CUjit_option_enum.CU_JIT_GLOBAL_SYMBOL_NAMES,):
            self._charstarstar = init_value
            self._cptr = <void*>&self._charstarstar[0]
        elif self._attr in (ccuda.CUjit_option_enum.CU_JIT_GLOBAL_SYMBOL_ADDRESSES,):
            pylist = [HelperInputVoidPtr(val) for val in init_value]
            self._voidstarstar = InputVoidPtrPtrHelper(pylist)
            self._cptr = <void*><void_ptr>self._voidstarstar.cptr
        else:
            raise TypeError('Unsupported attribute: {}'.format(attr.name))

    def __dealloc__(self):
        pass

    @property
    def cptr(self):
        return <void_ptr>self._cptr

cdef class InputVoidPtrPtrHelper:
    def __cinit__(self, lst):
        self._cptr = <void**>callocWrapper(len(lst), sizeof(void*))
        for idx in range(len(lst)):
            self._cptr[idx] = <void*><void_ptr>lst[idx].cptr

    def __dealloc__(self):
        free(self._cptr)

    @property
    def cptr(self):
        return <void_ptr>self._cptr
