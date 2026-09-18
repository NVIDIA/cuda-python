# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from cpython.buffer cimport PyObject_CheckBuffer, PyObject_GetBuffer, PyBuffer_Release, PyBUF_SIMPLE, PyBUF_ANY_CONTIGUOUS
from cpython.mem cimport PyMem_Calloc, PyMem_Free
from libc.stdlib cimport calloc, free
from libc.stdint cimport int8_t
from libc.stddef cimport size_t
from libc.string cimport memcpy
from libcpp cimport bool as cpp_bool
from cuda.bindings._internal._fast_enum import FastEnum as _FastEnum
import ctypes as _ctypes
cimport cuda.bindings.cydriver as cydriver

cdef void* _callocWrapper(length, size):
    cdef void* out = calloc(length, size)
    if out is NULL:
        raise MemoryError('Failed to allocated length x size memory: {}x{}'.format(length, size))
    return out

cdef object ctypes_c_bool = _ctypes.c_bool
cdef object ctypes_c_char = _ctypes.c_char
cdef object ctypes_c_wchar = _ctypes.c_wchar
cdef object ctypes_c_byte = _ctypes.c_byte
cdef object ctypes_c_ubyte = _ctypes.c_ubyte
cdef object ctypes_c_short = _ctypes.c_short
cdef object ctypes_c_ushort = _ctypes.c_ushort
cdef object ctypes_c_int = _ctypes.c_int
cdef object ctypes_c_uint = _ctypes.c_uint
cdef object ctypes_c_long = _ctypes.c_long
cdef object ctypes_c_ulong = _ctypes.c_ulong
cdef object ctypes_c_longlong = _ctypes.c_longlong
cdef object ctypes_c_ulonglong = _ctypes.c_ulonglong
cdef object ctypes_c_size_t = _ctypes.c_size_t
cdef object ctypes_c_float = _ctypes.c_float
cdef object ctypes_c_double = _ctypes.c_double
cdef object ctypes_c_void_p = _ctypes.c_void_p
cdef object ctypes_Structure = _ctypes.Structure

# excluding void_p and None, which are handled specially
cdef object supported_types = {
    ctypes_c_bool,
    ctypes_c_char,
    ctypes_c_wchar,
    ctypes_c_byte,
    ctypes_c_ubyte,
    ctypes_c_short,
    ctypes_c_ushort,
    ctypes_c_int,
    ctypes_c_uint,
    ctypes_c_long,
    ctypes_c_ulong,
    ctypes_c_longlong,
    ctypes_c_ulonglong,
    ctypes_c_size_t,
    ctypes_c_float,
    ctypes_c_double,
}

# A slot wide enough for all `supported_types` (and `void_p`)
cdef size_t MAX_PARAM_SIZE = max(_ctypes.sizeof(t) for t in supported_types)
# For correct access pointers have to fit (appended) and the size
# must be a multiple of the max alignment (guaranteed if power of 2).
assert MAX_PARAM_SIZE % sizeof(void*) == 0
assert ((MAX_PARAM_SIZE - 1) & MAX_PARAM_SIZE) == 0


cdef int _try_specific_types(char* slot, object value, object ctype) except -1:
    """Helper for specific type parsing.  If this fails, conversion goes
    via `ctype(value)`.  This converter is more strict than ctypes
    (e.g. raises for out of bound integers).  It should be extended if a
    specific argument is slow.
    """
    cdef object value_type = type(value)
    if ctype is ctypes_c_int and value_type is int:
        (<int*>slot)[0] = value
        return 1
    if ctype is ctypes_c_bool and value_type is bool:
        (<cpp_bool*>slot)[0] = value
        return 1
    if ctype is ctypes_c_byte and value_type is int:
        (<int8_t*>slot)[0] = value
        return 1
    if ctype is ctypes_c_double:
        if value_type is float:
            (<double*>slot)[0] = value
            return 1
        if isinstance(value, ctypes_c_float):
            # This explicitly allows c_float for double arguments.
            (<double*>slot)[0] = value.value
            return 1
        return 0
    if ctype is ctypes_c_float and value_type is float:
        (<float*>slot)[0] = value
        return 1
    if ctype is ctypes_c_longlong and value_type is int:
        (<long long*>slot)[0] = value
        return 1
    return 0


cdef int _pack_argument(void** ptr, char* slot, object value, object ctype) except -1:
    cdef size_t size
    cdef void* addr
    cdef object getPtr

    if ctype is None:
        getPtr = getattr(value, 'getPtr', None)
        if callable(getPtr):
            ptr[0] = <void*><void_ptr>getPtr()
        elif isinstance(value, ctypes_Structure):
            ptr[0] = <void*><void_ptr>_ctypes.addressof(value)
        elif isinstance(value, _FastEnum):
            ptr[0] = slot
            (<int*>slot)[0] = value  # _FastEnum is an int
        else:
            raise TypeError(f"Provided argument is of type {type(value)} but expected Type {_ctypes.Structure}, {_ctypes.c_void_p} or CUDA Binding structure with getPtr() attribute")
        return 0

    ptr[0] = slot
    if _try_specific_types(slot, value, ctype):
        return 0
    if ctype in supported_types:
        if not isinstance(value, ctype):
            value = ctype(value)
        size = <size_t>_ctypes.sizeof(ctype)
        addr = <void*><void_ptr>_ctypes.addressof(value)
        memcpy(slot, addr, size)
        return 0
    elif ctype is ctypes_c_void_p:
        if isinstance(value, (int, ctypes_c_void_p)):
            (<void_ptr*>slot)[0] = value.value if isinstance(value, ctypes_c_void_p) else value
        else:
            getPtr = getattr(value, 'getPtr', None)
            if callable(getPtr):
                (<void_ptr*>slot)[0] = getPtr()
            else:
                raise TypeError(f"Provided argument is of type {type(value)} but expected Type {int}, {_ctypes.c_void_p} or CUDA Binding structure with getPtr() attribute")
        return 0
    raise TypeError(f"Unsupported type: {ctype!r}")


cdef class _HelperKernelParams:
    def __cinit__(self, kernelParams):
        cdef tuple values, types
        cdef Py_ssize_t i, n
        cdef size_t data_bytes, total
        cdef char* block
        cdef char* slot
        cdef int err_buffer
        cdef void** ptrs

        self._pyobj_acquired = False
        self.ckernelParams = NULL
        self._ckernelParamsData = NULL

        if kernelParams is None:
            pass
        elif isinstance(kernelParams, int):
            # Easy run, user gave us an already configured void** address
            self.ckernelParams = <void**><void_ptr>kernelParams
        elif PyObject_CheckBuffer(kernelParams):
            # Easy run, get address from Python Buffer Protocol
            err_buffer = PyObject_GetBuffer(kernelParams, &self._pybuffer, PyBUF_SIMPLE | PyBUF_ANY_CONTIGUOUS)
            if err_buffer == -1:
                raise RuntimeError("Argument 'kernelParams' failed to retrieve buffer through Buffer Protocol")
            self._pyobj_acquired = True
            self.ckernelParams = <void**><void_ptr>self._pybuffer.buf
        elif (
            isinstance(kernelParams, tuple)
            and len(kernelParams) == 2
            and isinstance(kernelParams[0], tuple)
            and isinstance(kernelParams[1], tuple)
        ):
            # Hard run, construct and fill out contiguous memory using provided kernel values and types
            values = <tuple>kernelParams[0]
            types = <tuple>kernelParams[1]
            n = len(values)
            if n != len(types):
                raise TypeError("Argument 'kernelParams' has tuples with different length")
            if n == 0:
                return
            data_bytes = <size_t>n * MAX_PARAM_SIZE
            total = data_bytes + <size_t>n * sizeof(void*)
            block = <char*>PyMem_Calloc(1, total)
            if block == NULL:
                raise MemoryError('Failed to allocated length x size memory: {}x{}'.format(n, MAX_PARAM_SIZE))
            self._ckernelParamsData = block
            ptrs = <void**>(block + data_bytes)
            self.ckernelParams = ptrs
            for i in range(n):
                slot = block + i * MAX_PARAM_SIZE
                _pack_argument(ptrs + i, slot, values[i], types[i])
        else:
            raise TypeError("Argument 'kernelParams' is not a valid type: tuple[tuple[Any, ...], tuple[Any, ...]] or PyObject implimenting Buffer Protocol or Int")

    def __dealloc__(self):
        if self._pyobj_acquired:
            PyBuffer_Release(&self._pybuffer)
        if self._ckernelParamsData:
            PyMem_Free(self._ckernelParamsData)

cdef class _HelperInputVoidPtr:
    def __cinit__(self, ptr):
        self._cptr = _helper_input_void_ptr(ptr, &self._helper)

    def __dealloc__(self):
        _helper_input_void_ptr_free(&self._helper)

    @property
    def cptr(self):
        return <void_ptr>self._cptr


cdef void * _helper_input_void_ptr(ptr, _HelperInputVoidPtrStruct *helper):
    helper[0]._pybuffer.buf = NULL
    try:
        return <void *><void_ptr>ptr
    except:
        if ptr is None:
            return NULL
        elif PyObject_CheckBuffer(ptr):
            # Easy run, get address from Python Buffer Protocol
            err_buffer = PyObject_GetBuffer(ptr, &helper[0]._pybuffer, PyBUF_SIMPLE | PyBUF_ANY_CONTIGUOUS)
            if err_buffer == -1:
                raise RuntimeError("Failed to retrieve buffer through Buffer Protocol")
            return <void*><void_ptr>(helper[0]._pybuffer.buf)
        else:
            raise TypeError(f"Provided argument is of type {type(ptr)} but expected Type None, int or object with Buffer Protocol")




cdef class _HelperCUmemPool_attribute:
    def __cinit__(self, attr, init_value, is_getter=False):
        self._is_getter = is_getter
        self._attr = attr.value
        if self._attr in (cydriver.CUmemPool_attribute_enum.CU_MEMPOOL_ATTR_REUSE_FOLLOW_EVENT_DEPENDENCIES,
                          cydriver.CUmemPool_attribute_enum.CU_MEMPOOL_ATTR_REUSE_ALLOW_OPPORTUNISTIC,
                          cydriver.CUmemPool_attribute_enum.CU_MEMPOOL_ATTR_REUSE_ALLOW_INTERNAL_DEPENDENCIES,):
            self._int_val = init_value
            self._cptr = <void*>&self._int_val
        elif self._attr in (cydriver.CUmemPool_attribute_enum.CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
                            cydriver.CUmemPool_attribute_enum.CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT,
                            cydriver.CUmemPool_attribute_enum.CU_MEMPOOL_ATTR_RESERVED_MEM_HIGH,
                            cydriver.CUmemPool_attribute_enum.CU_MEMPOOL_ATTR_USED_MEM_CURRENT,
                            cydriver.CUmemPool_attribute_enum.CU_MEMPOOL_ATTR_USED_MEM_HIGH,):
            if self._is_getter:
                self._cuuint64_t_val = _driver["cuuint64_t"]()
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
        if self._attr in (cydriver.CUmemPool_attribute_enum.CU_MEMPOOL_ATTR_REUSE_FOLLOW_EVENT_DEPENDENCIES,
                          cydriver.CUmemPool_attribute_enum.CU_MEMPOOL_ATTR_REUSE_ALLOW_OPPORTUNISTIC,
                          cydriver.CUmemPool_attribute_enum.CU_MEMPOOL_ATTR_REUSE_ALLOW_INTERNAL_DEPENDENCIES,):
            return self._int_val
        elif self._attr in (cydriver.CUmemPool_attribute_enum.CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
                            cydriver.CUmemPool_attribute_enum.CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT,
                            cydriver.CUmemPool_attribute_enum.CU_MEMPOOL_ATTR_RESERVED_MEM_HIGH,
                            cydriver.CUmemPool_attribute_enum.CU_MEMPOOL_ATTR_USED_MEM_CURRENT,
                            cydriver.CUmemPool_attribute_enum.CU_MEMPOOL_ATTR_USED_MEM_HIGH,):
            return self._cuuint64_t_val
        else:
            raise TypeError('Unsupported attribute value: {}'.format(self._attr))



cdef class _HelperCUmem_range_attribute:
    def __cinit__(self, attr, data_size):
        self._data_size = data_size
        self._attr = attr.value
        if self._attr in (cydriver.CUmem_range_attribute_enum.CU_MEM_RANGE_ATTRIBUTE_READ_MOSTLY,
                          cydriver.CUmem_range_attribute_enum.CU_MEM_RANGE_ATTRIBUTE_PREFERRED_LOCATION,
                          cydriver.CUmem_range_attribute_enum.CU_MEM_RANGE_ATTRIBUTE_LAST_PREFETCH_LOCATION,):
            self._cptr = <void*>&self._int_val
        elif self._attr in (cydriver.CUmem_range_attribute_enum.CU_MEM_RANGE_ATTRIBUTE_ACCESSED_BY,):
            self._cptr = _callocWrapper(1, self._data_size)
            self._int_val_list = <int*>self._cptr
        else:
            raise TypeError('Unsupported attribute: {}'.format(attr.name))

    def __dealloc__(self):
        if self._attr in (cydriver.CUmem_range_attribute_enum.CU_MEM_RANGE_ATTRIBUTE_ACCESSED_BY,):
            free(self._cptr)

    @property
    def cptr(self):
        return <void_ptr>self._cptr

    def pyObj(self):
        if self._attr in (cydriver.CUmem_range_attribute_enum.CU_MEM_RANGE_ATTRIBUTE_READ_MOSTLY,
                          cydriver.CUmem_range_attribute_enum.CU_MEM_RANGE_ATTRIBUTE_PREFERRED_LOCATION,
                          cydriver.CUmem_range_attribute_enum.CU_MEM_RANGE_ATTRIBUTE_LAST_PREFETCH_LOCATION,):
            return self._int_val
        elif self._attr in (cydriver.CUmem_range_attribute_enum.CU_MEM_RANGE_ATTRIBUTE_ACCESSED_BY,):
            return [self._int_val_list[idx] for idx in range(int(self._data_size/4))]
        else:
            raise TypeError('Unsupported attribute value: {}'.format(self._attr))



cdef class _HelperCUpointer_attribute:
    def __cinit__(self, attr, init_value, is_getter=False):
        self._is_getter = is_getter
        self._attr = attr.value
        if self._attr in (cydriver.CUpointer_attribute_enum.CU_POINTER_ATTRIBUTE_CONTEXT,):
            if self._is_getter:
                self._ctx = _driver["CUcontext"]()
                self._cptr = <void*><void_ptr>self._ctx.getPtr()
            else:
                self._cptr = <void*><void_ptr>init_value.getPtr()
        elif self._attr in (cydriver.CUpointer_attribute_enum.CU_POINTER_ATTRIBUTE_MEMORY_TYPE,
                            cydriver.CUpointer_attribute_enum.CU_POINTER_ATTRIBUTE_ALLOWED_HANDLE_TYPES,
                            cydriver.CUpointer_attribute_enum.CU_POINTER_ATTRIBUTE_IS_GPU_DIRECT_RDMA_CAPABLE,
                            cydriver.CUpointer_attribute_enum.CU_POINTER_ATTRIBUTE_ACCESS_FLAGS,):
            self._uint = init_value
            self._cptr = <void*>&self._uint
        elif self._attr in (cydriver.CUpointer_attribute_enum.CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL,):
            self._int = init_value
            self._cptr = <void*>&self._int
        elif self._attr in (cydriver.CUpointer_attribute_enum.CU_POINTER_ATTRIBUTE_DEVICE_POINTER,
                            cydriver.CUpointer_attribute_enum.CU_POINTER_ATTRIBUTE_RANGE_START_ADDR,):
            if self._is_getter:
                self._devptr = _driver["CUdeviceptr"]()
                self._cptr = <void*><void_ptr>self._devptr.getPtr()
            else:
                self._cptr = <void*><void_ptr>init_value.getPtr()
        elif self._attr in (cydriver.CUpointer_attribute_enum.CU_POINTER_ATTRIBUTE_HOST_POINTER,):
            self._void = <void**><void_ptr>init_value
            self._cptr = <void*>&self._void
        elif self._attr in (cydriver.CUpointer_attribute_enum.CU_POINTER_ATTRIBUTE_P2P_TOKENS,):
            if self._is_getter:
                self._token = _driver["CUDA_POINTER_ATTRIBUTE_P2P_TOKENS"]()
                self._cptr = <void*><void_ptr>self._token.getPtr()
            else:
                self._cptr = <void*><void_ptr>init_value.getPtr()
        elif self._attr in (cydriver.CUpointer_attribute_enum.CU_POINTER_ATTRIBUTE_SYNC_MEMOPS,
                            cydriver.CUpointer_attribute_enum.CU_POINTER_ATTRIBUTE_IS_MANAGED,
                            cydriver.CUpointer_attribute_enum.CU_POINTER_ATTRIBUTE_IS_LEGACY_CUDA_IPC_CAPABLE,
                            cydriver.CUpointer_attribute_enum.CU_POINTER_ATTRIBUTE_MAPPED,):
            self._bool = init_value
            self._cptr = <void*>&self._bool
        elif self._attr in (cydriver.CUpointer_attribute_enum.CU_POINTER_ATTRIBUTE_BUFFER_ID,):
            self._ull = init_value
            self._cptr = <void*>&self._ull
        elif self._attr in (cydriver.CUpointer_attribute_enum.CU_POINTER_ATTRIBUTE_RANGE_SIZE,):
            self._size = init_value
            self._cptr = <void*>&self._size
        elif self._attr in (cydriver.CUpointer_attribute_enum.CU_POINTER_ATTRIBUTE_MEMPOOL_HANDLE,):
            if self._is_getter:
                self._mempool = _driver["CUmemoryPool"]()
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
        if self._attr in (cydriver.CUpointer_attribute_enum.CU_POINTER_ATTRIBUTE_CONTEXT,):
            return self._ctx
        elif self._attr in (cydriver.CUpointer_attribute_enum.CU_POINTER_ATTRIBUTE_MEMORY_TYPE,
                            cydriver.CUpointer_attribute_enum.CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL,
                            cydriver.CUpointer_attribute_enum.CU_POINTER_ATTRIBUTE_ALLOWED_HANDLE_TYPES,
                            cydriver.CUpointer_attribute_enum.CU_POINTER_ATTRIBUTE_IS_GPU_DIRECT_RDMA_CAPABLE,
                            cydriver.CUpointer_attribute_enum.CU_POINTER_ATTRIBUTE_ACCESS_FLAGS,):
            return self._uint
        elif self._attr in (cydriver.CUpointer_attribute_enum.CU_POINTER_ATTRIBUTE_DEVICE_POINTER,
                            cydriver.CUpointer_attribute_enum.CU_POINTER_ATTRIBUTE_RANGE_START_ADDR,):
            return self._devptr
        elif self._attr in (cydriver.CUpointer_attribute_enum.CU_POINTER_ATTRIBUTE_HOST_POINTER,):
            return <void_ptr>self._void
        elif self._attr in (cydriver.CUpointer_attribute_enum.CU_POINTER_ATTRIBUTE_P2P_TOKENS,):
            return self._token
        elif self._attr in (cydriver.CUpointer_attribute_enum.CU_POINTER_ATTRIBUTE_SYNC_MEMOPS,
                            cydriver.CUpointer_attribute_enum.CU_POINTER_ATTRIBUTE_IS_MANAGED,
                            cydriver.CUpointer_attribute_enum.CU_POINTER_ATTRIBUTE_IS_LEGACY_CUDA_IPC_CAPABLE,
                            cydriver.CUpointer_attribute_enum.CU_POINTER_ATTRIBUTE_MAPPED,):
            return self._bool
        elif self._attr in (cydriver.CUpointer_attribute_enum.CU_POINTER_ATTRIBUTE_BUFFER_ID,):
            return self._ull
        elif self._attr in (cydriver.CUpointer_attribute_enum.CU_POINTER_ATTRIBUTE_RANGE_SIZE,):
            return self._size
        elif self._attr in (cydriver.CUpointer_attribute_enum.CU_POINTER_ATTRIBUTE_MEMPOOL_HANDLE,):
            return self._mempool
        else:
            raise TypeError('Unsupported attribute value: {}'.format(self._attr))



cdef class _HelperCUgraphMem_attribute:
    def __cinit__(self, attr, init_value, is_getter=False):
        self._is_getter = is_getter
        self._attr = attr.value
        if self._attr in (cydriver.CUgraphMem_attribute_enum.CU_GRAPH_MEM_ATTR_USED_MEM_CURRENT,
                          cydriver.CUgraphMem_attribute_enum.CU_GRAPH_MEM_ATTR_USED_MEM_HIGH,
                          cydriver.CUgraphMem_attribute_enum.CU_GRAPH_MEM_ATTR_RESERVED_MEM_CURRENT,
                          cydriver.CUgraphMem_attribute_enum.CU_GRAPH_MEM_ATTR_RESERVED_MEM_HIGH,):
            if self._is_getter:
                self._cuuint64_t_val = _driver["cuuint64_t"]()
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
        if self._attr in (cydriver.CUgraphMem_attribute_enum.CU_GRAPH_MEM_ATTR_USED_MEM_CURRENT,
                          cydriver.CUgraphMem_attribute_enum.CU_GRAPH_MEM_ATTR_USED_MEM_HIGH,
                          cydriver.CUgraphMem_attribute_enum.CU_GRAPH_MEM_ATTR_RESERVED_MEM_CURRENT,
                          cydriver.CUgraphMem_attribute_enum.CU_GRAPH_MEM_ATTR_RESERVED_MEM_HIGH,):
            return self._cuuint64_t_val
        else:
            raise TypeError('Unsupported attribute value: {}'.format(self._attr))



cdef class _HelperCUjit_option:
    def __cinit__(self, attr, init_value):
        self._attr = attr.value
        if self._attr in (cydriver.CUjit_option_enum.CU_JIT_MAX_REGISTERS,
                          cydriver.CUjit_option_enum.CU_JIT_THREADS_PER_BLOCK,
                          cydriver.CUjit_option_enum.CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES,
                          cydriver.CUjit_option_enum.CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
                          cydriver.CUjit_option_enum.CU_JIT_OPTIMIZATION_LEVEL,
                          cydriver.CUjit_option_enum.CU_JIT_GLOBAL_SYMBOL_COUNT,
                          cydriver.CUjit_option_enum.CU_JIT_TARGET_FROM_CUCONTEXT,
                          cydriver.CUjit_option_enum.CU_JIT_REFERENCED_KERNEL_COUNT,
                          cydriver.CUjit_option_enum.CU_JIT_REFERENCED_VARIABLE_COUNT,
                          cydriver.CUjit_option_enum.CU_JIT_MIN_CTA_PER_SM,
                          cydriver.CUjit_option_enum.CU_JIT_SPLIT_COMPILE,):
            self._uint = init_value
            self._cptr = <void*><void_ptr>self._uint
        elif self._attr in (cydriver.CUjit_option_enum.CU_JIT_WALL_TIME,):
            self._float = init_value
            self._cptr = <void*><void_ptr>self._float
        elif self._attr in (cydriver.CUjit_option_enum.CU_JIT_INFO_LOG_BUFFER,
                            cydriver.CUjit_option_enum.CU_JIT_ERROR_LOG_BUFFER):
            self._charstar = init_value
            self._cptr = <void*><void_ptr>self._charstar
        elif self._attr in (cydriver.CUjit_option_enum.CU_JIT_TARGET,):
            self._target = init_value.value
            self._cptr = <void*><void_ptr>self._target
        elif self._attr in (cydriver.CUjit_option_enum.CU_JIT_FALLBACK_STRATEGY,):
            self._fallback = init_value.value
            self._cptr = <void*><void_ptr>self._fallback
        elif self._attr in (cydriver.CUjit_option_enum.CU_JIT_GENERATE_DEBUG_INFO,
                            cydriver.CUjit_option_enum.CU_JIT_LOG_VERBOSE,
                            cydriver.CUjit_option_enum.CU_JIT_GENERATE_LINE_INFO,
                            cydriver.CUjit_option_enum.CU_JIT_LTO,
                            cydriver.CUjit_option_enum.CU_JIT_FTZ,
                            cydriver.CUjit_option_enum.CU_JIT_PREC_DIV,
                            cydriver.CUjit_option_enum.CU_JIT_PREC_SQRT,
                            cydriver.CUjit_option_enum.CU_JIT_FMA,
                            cydriver.CUjit_option_enum.CU_JIT_OPTIMIZE_UNUSED_DEVICE_VARIABLES,):
            self._int = init_value
            self._cptr = <void*><void_ptr>self._int
        elif self._attr in (cydriver.CUjit_option_enum.CU_JIT_CACHE_MODE,):
            self._cacheMode = init_value.value
            self._cptr = <void*><void_ptr>self._cacheMode
        elif self._attr in (cydriver.CUjit_option_enum.CU_JIT_GLOBAL_SYMBOL_NAMES,
                            cydriver.CUjit_option_enum.CU_JIT_REFERENCED_KERNEL_NAMES,
                            cydriver.CUjit_option_enum.CU_JIT_REFERENCED_VARIABLE_NAMES,):
            self._charstarstar = init_value
            self._cptr = <void*>&self._charstarstar[0]
        elif self._attr in (cydriver.CUjit_option_enum.CU_JIT_GLOBAL_SYMBOL_ADDRESSES,):
            pylist = [_HelperInputVoidPtr(val) for val in init_value]
            self._voidstarstar = _InputVoidPtrPtrHelper(pylist)
            self._cptr = <void*><void_ptr>self._voidstarstar.cptr
        else:
            raise TypeError('Unsupported attribute: {}'.format(attr.name))

    def __dealloc__(self):
        pass

    @property
    def cptr(self):
        return <void_ptr>self._cptr




cdef class _HelperCudaJitOption:
    def __cinit__(self, attr, init_value):
        self._attr = attr.value
        if self._attr in (cyruntime.cudaJitOption.cudaJitMaxRegisters,
                          cyruntime.cudaJitOption.cudaJitThreadsPerBlock,
                          cyruntime.cudaJitOption.cudaJitInfoLogBufferSizeBytes,
                          cyruntime.cudaJitOption.cudaJitErrorLogBufferSizeBytes,
                          cyruntime.cudaJitOption.cudaJitOptimizationLevel,
                          cyruntime.cudaJitOption.cudaJitMinCtaPerSm,):
            self._uint = init_value
            self._cptr = <void*><void_ptr>self._uint
        elif self._attr in (cyruntime.cudaJitOption.cudaJitWallTime,):
            self._float = init_value
            self._cptr = <void*><void_ptr>self._float
        elif self._attr in (cyruntime.cudaJitOption.cudaJitInfoLogBuffer,
                            cyruntime.cudaJitOption.cudaJitErrorLogBuffer):
            self._charstar = init_value
            self._cptr = <void*><void_ptr>self._charstar
        elif self._attr in (cyruntime.cudaJitOption.cudaJitFallbackStrategy,):
            self._fallback = init_value.value
            self._cptr = <void*><void_ptr>self._fallback
        elif self._attr in (cyruntime.cudaJitOption.cudaJitGenerateDebugInfo,
                            cyruntime.cudaJitOption.cudaJitLogVerbose,
                            cyruntime.cudaJitOption.cudaJitGenerateLineInfo,
                            cyruntime.cudaJitOption.cudaJitPositionIndependentCode,
                            cyruntime.cudaJitOption.cudaJitMaxThreadsPerBlock,
                            cyruntime.cudaJitOption.cudaJitOverrideDirectiveValues,):
            self._int = init_value
            self._cptr = <void*><void_ptr>self._int
        elif self._attr in (cyruntime.cudaJitOption.cudaJitCacheMode,):
            self._cacheMode = init_value.value
            self._cptr = <void*><void_ptr>self._cacheMode
        else:
            raise TypeError('Unsupported attribute: {}'.format(attr.name))

    def __dealloc__(self):
        pass

    @property
    def cptr(self):
        return <void_ptr>self._cptr




cdef class _HelperCUlibraryOption:
    def __cinit__(self, attr, init_value):
        self._attr = attr.value
        if False:
            pass

        elif self._attr in (cydriver.CUlibraryOption_enum.CU_LIBRARY_HOST_UNIVERSAL_FUNCTION_AND_DATA_TABLE,):
            self._cptr = <void*><void_ptr>init_value.getPtr()


        elif self._attr in (cydriver.CUlibraryOption_enum.CU_LIBRARY_BINARY_IS_PRESERVED,):
            self._uint = init_value
            self._cptr = <void*><void_ptr>self._uint

        else:
            raise TypeError('Unsupported attribute: {}'.format(attr.name))

    def __dealloc__(self):
        pass

    @property
    def cptr(self):
        return <void_ptr>self._cptr




cdef class _HelperCudaLibraryOption:
    def __cinit__(self, attr, init_value):
        self._attr = attr.value
        if False:
            pass

        elif self._attr in (cyruntime.cudaLibraryOption.cudaLibraryHostUniversalFunctionAndDataTable,):
            self._cptr = <void*><void_ptr>init_value.getPtr()


        elif self._attr in (cyruntime.cudaLibraryOption.cudaLibraryBinaryIsPreserved,):
            self._uint = init_value
            self._cptr = <void*><void_ptr>self._uint

        else:
            raise TypeError('Unsupported attribute: {}'.format(attr.name))

    def __dealloc__(self):
        pass

    @property
    def cptr(self):
        return <void_ptr>self._cptr




cdef class _HelperCUmemAllocationHandleType:
    def __cinit__(self, attr):
        self._type = attr.value
        if False:
            pass

        elif self._type in (cydriver.CUmemAllocationHandleType_enum.CU_MEM_HANDLE_TYPE_NONE,):
            self._cptr = <void*>&self._int


        elif self._type in (cydriver.CUmemAllocationHandleType_enum.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,):
            self._cptr = <void*>&self._int


        elif self._type in (cydriver.CUmemAllocationHandleType_enum.CU_MEM_HANDLE_TYPE_WIN32,):
            self._cptr = <void*>&self._handle


        elif self._type in (cydriver.CUmemAllocationHandleType_enum.CU_MEM_HANDLE_TYPE_WIN32_KMT,):
            self._cptr = <void*>&self._d3dkmt_handle


        elif self._type in (cydriver.CUmemAllocationHandleType_enum.CU_MEM_HANDLE_TYPE_FABRIC,):
            self._mem_fabric_handle = _driver["CUmemFabricHandle"]()
            self._cptr = <void*><void_ptr>self._mem_fabric_handle.getPtr()

        else:
            raise TypeError('Unsupported attribute: {}'.format(attr.name))

    def __dealloc__(self):
        pass

    @property
    def cptr(self):
        return <void_ptr>self._cptr

    def pyObj(self):
        if False:
            pass

        elif self._type in (cydriver.CUmemAllocationHandleType_enum.CU_MEM_HANDLE_TYPE_NONE,):
            return self._int


        elif self._type in (cydriver.CUmemAllocationHandleType_enum.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,):
            return self._int


        elif self._type in (cydriver.CUmemAllocationHandleType_enum.CU_MEM_HANDLE_TYPE_WIN32,):
            return <void_ptr>self._handle


        elif self._type in (cydriver.CUmemAllocationHandleType_enum.CU_MEM_HANDLE_TYPE_WIN32_KMT,):
            return self._d3dkmt_handle


        elif self._type in (cydriver.CUmemAllocationHandleType_enum.CU_MEM_HANDLE_TYPE_FABRIC,):
            return self._mem_fabric_handle

        else:
            raise TypeError('Unsupported attribute: {}'.format(self._type))



cdef class _HelperCUlogicalEndpointIpcHandleType:
    def __cinit__(self, attr):
        self._type = attr.value
        if False:
            pass

        elif self._type in (cydriver.CUlogicalEndpointIpcHandleType_enum.CU_LOGICAL_ENDPOINT_IPC_HANDLE_TYPE_NONE,):
            self._cptr = <void*>&self._int


        elif self._type in (cydriver.CUlogicalEndpointIpcHandleType_enum.CU_LOGICAL_ENDPOINT_IPC_HANDLE_TYPE_FABRIC,):
            self._fabric_handle = _driver["CUlogicalEndpointFabricHandle"]()
            self._cptr = <void*><void_ptr>self._fabric_handle.getPtr()

        else:
            raise TypeError('Unsupported attribute: {}'.format(attr.name))

    def __dealloc__(self):
        pass

    @property
    def cptr(self):
        return <void_ptr>self._cptr

    def pyObj(self):
        if False:
            pass

        elif self._type in (cydriver.CUlogicalEndpointIpcHandleType_enum.CU_LOGICAL_ENDPOINT_IPC_HANDLE_TYPE_NONE,):
            return self._int


        elif self._type in (cydriver.CUlogicalEndpointIpcHandleType_enum.CU_LOGICAL_ENDPOINT_IPC_HANDLE_TYPE_FABRIC,):
            return self._fabric_handle

        else:
            raise TypeError('Unsupported attribute: {}'.format(self._type))


cdef class _InputVoidPtrPtrHelper:
    def __cinit__(self, lst):
        # Hold onto references to the original buffers so they
        # won't be free'd behind our back
        self._references = lst
        self._cptr = <void**>_callocWrapper(len(lst), sizeof(void*))
        for idx in range(len(lst)):
            self._cptr[idx] = <void*><void_ptr>lst[idx].cptr

    def __dealloc__(self):
        free(self._cptr)

    @property
    def cptr(self):
        return <void_ptr>self._cptr



cdef class _HelperCUcoredumpSettings:
    def __cinit__(self, attr, init_value, is_getter=False):
        self._is_getter = is_getter
        self._attrib = attr.value
        if self._attrib in (cydriver.CUcoredumpSettings_enum.CU_COREDUMP_FILE,
                          cydriver.CUcoredumpSettings_enum.CU_COREDUMP_PIPE,):
            if self._is_getter:
                self._charstar = <char*>_callocWrapper(1024, 1)
                self._cptr = <void*><void_ptr>self._charstar
                self._size = 1024
            else:
                # Keep a reference so the borrowed _charstar buffer stays alive.
                self._references = init_value
                self._charstar = init_value
                self._cptr = <void*><void_ptr>self._charstar
                self._size = len(init_value)
        elif self._attrib in (cydriver.CUcoredumpSettings_enum.CU_COREDUMP_ENABLE_ON_EXCEPTION,
                            cydriver.CUcoredumpSettings_enum.CU_COREDUMP_TRIGGER_HOST,
                            cydriver.CUcoredumpSettings_enum.CU_COREDUMP_LIGHTWEIGHT,
                            cydriver.CUcoredumpSettings_enum.CU_COREDUMP_ENABLE_USER_TRIGGER,):
            if self._is_getter == False:
                self._bool = init_value

            self._cptr = <void*>&self._bool
            self._size = 1
        else:
            raise TypeError('Unsupported attribute: {}'.format(attr.name))

    def __dealloc__(self):
        # Only the getter path owns heap (the calloc'd 1024-byte buffer). The
        # setter borrows caller bytes and the bool path points at &self._bool,
        # so only free for the getter.
        if self._is_getter:
            free(self._charstar)

    @property
    def cptr(self):
        return <void_ptr>self._cptr

    def size(self):
        return self._size

    def pyObj(self):
        assert(self._is_getter == True)
        if self._attrib in (cydriver.CUcoredumpSettings_enum.CU_COREDUMP_FILE,
                          cydriver.CUcoredumpSettings_enum.CU_COREDUMP_PIPE,):
            return self._charstar
        elif self._attrib in (cydriver.CUcoredumpSettings_enum.CU_COREDUMP_ENABLE_ON_EXCEPTION,
                            cydriver.CUcoredumpSettings_enum.CU_COREDUMP_TRIGGER_HOST,
                            cydriver.CUcoredumpSettings_enum.CU_COREDUMP_LIGHTWEIGHT,
                            cydriver.CUcoredumpSettings_enum.CU_COREDUMP_ENABLE_USER_TRIGGER,):
            return self._bool
        else:
            raise TypeError('Unsupported attribute value: {}'.format(self._attrib))
