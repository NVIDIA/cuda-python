# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE
#
# This code was automatically generated across versions from 12.9.1 to 13.1.1, generator version 0.3.1.dev1322+g646ce84ec. Do not modify it directly.

cimport cython  # NOQA
from libc cimport errno
from ._internal.utils cimport (get_buffer_pointer, get_nested_resource_ptr,
                               nested_resource)
from cuda.bindings._internal._fast_enum import FastEnum as _FastEnum

import cython

from cuda.bindings.driver import CUresult as pyCUresult

from libc.stdlib cimport calloc, free, malloc
from cython cimport view
cimport cpython.buffer
cimport cpython.memoryview
cimport cpython
from libc.string cimport memcmp, memcpy
import numpy as _numpy


cdef __from_data(data, dtype_name, expected_dtype, lowpp_type):
    # _numpy.recarray is a subclass of _numpy.ndarray, so implicitly handled here.
    if isinstance(data, lowpp_type):
        return data
    if not isinstance(data, _numpy.ndarray):
        raise TypeError("data argument must be a NumPy ndarray")
    if data.size != 1:
        raise ValueError("data array must have a size of 1")
    if data.dtype != expected_dtype:
        raise ValueError(f"data array must be of dtype {dtype_name}")
    return lowpp_type.from_ptr(data.ctypes.data, not data.flags.writeable, data)


cdef __from_buffer(buffer, size, lowpp_type):
    cdef Py_buffer view
    if cpython.PyObject_GetBuffer(buffer, &view, cpython.PyBUF_SIMPLE) != 0:
        raise TypeError("buffer argument does not support the buffer protocol")
    try:
        if view.itemsize != 1:
            raise ValueError("buffer itemsize must be 1 byte")
        if view.len != size:
            raise ValueError(f"buffer length must be {size} bytes")
        return lowpp_type.from_ptr(<intptr_t><void *>view.buf, not view.readonly, buffer)
    finally:
        cpython.PyBuffer_Release(&view)


cdef __getbuffer(object self, cpython.Py_buffer *buffer, void *ptr, int size, bint readonly):
    buffer.buf = <char *>ptr
    buffer.format = 'b'
    buffer.internal = NULL
    buffer.itemsize = 1
    buffer.len = size
    buffer.ndim = 1
    buffer.obj = self
    buffer.readonly = readonly
    buffer.shape = &buffer.len
    buffer.strides = &buffer.itemsize
    buffer.suboffsets = NULL

###############################################################################
# POD
###############################################################################

_py_anon_pod1_dtype = _numpy.dtype((
    _numpy.dtype((_numpy.void, sizeof((<CUfileDescr_t*>NULL).handle))),
    {
        "fd": (_numpy.int32, 0),
        "handle": (_numpy.intp, 0),
    }
    ))


cdef class _py_anon_pod1:
    """Empty-initialize an instance of `_anon_pod1`.


    .. seealso:: `_anon_pod1`
    """
    cdef:
        _anon_pod1 *_ptr
        object _owner
        bint _owned
        bint _readonly

    def __init__(self):
        self._ptr = <_anon_pod1 *>calloc(1, sizeof((<CUfileDescr_t*>NULL).handle))
        if self._ptr == NULL:
            raise MemoryError("Error allocating _py_anon_pod1")
        self._owner = None
        self._owned = True
        self._readonly = False

    def __dealloc__(self):
        cdef _anon_pod1 *ptr
        if self._owned and self._ptr != NULL:
            ptr = self._ptr
            self._ptr = NULL
            free(ptr)

    def __repr__(self):
        return f"<{__name__}._py_anon_pod1 object at {hex(id(self))}>"

    @property
    def ptr(self):
        """Get the pointer address to the data as Python :class:`int`."""
        return <intptr_t>(self._ptr)

    cdef intptr_t _get_ptr(self):
        return <intptr_t>(self._ptr)

    def __int__(self):
        return <intptr_t>(self._ptr)

    def __eq__(self, other):
        cdef _py_anon_pod1 other_
        if not isinstance(other, _py_anon_pod1):
            return False
        other_ = other
        return (memcmp(<void *><intptr_t>(self._ptr), <void *><intptr_t>(other_._ptr), sizeof((<CUfileDescr_t*>NULL).handle)) == 0)

    def __getbuffer__(self, Py_buffer *buffer, int flags):
        __getbuffer(self, buffer, <void *>self._ptr, sizeof((<CUfileDescr_t*>NULL).handle), self._readonly)

    def __releasebuffer__(self, Py_buffer *buffer):
        pass

    def __setitem__(self, key, val):
        if key == 0 and isinstance(val, _numpy.ndarray):
            self._ptr = <_anon_pod1 *>malloc(sizeof((<CUfileDescr_t*>NULL).handle))
            if self._ptr == NULL:
                raise MemoryError("Error allocating _py_anon_pod1")
            memcpy(<void*>self._ptr, <void*><intptr_t>val.ctypes.data, sizeof((<CUfileDescr_t*>NULL).handle))
            self._owner = None
            self._owned = True
            self._readonly = not val.flags.writeable
        else:
            setattr(self, key, val)

    @property
    def fd(self):
        """int: """
        return self._ptr[0].fd

    @fd.setter
    def fd(self, val):
        if self._readonly:
            raise ValueError("This _py_anon_pod1 instance is read-only")
        self._ptr[0].fd = val

    @property
    def handle(self):
        """int: """
        return <intptr_t>(self._ptr[0].handle)

    @handle.setter
    def handle(self, val):
        if self._readonly:
            raise ValueError("This _py_anon_pod1 instance is read-only")
        self._ptr[0].handle = <void *><intptr_t>val

    @staticmethod
    def from_buffer(buffer):
        """Create an _py_anon_pod1 instance with the memory from the given buffer."""
        return __from_buffer(buffer, sizeof((<CUfileDescr_t*>NULL).handle), _py_anon_pod1)

    @staticmethod
    def from_data(data):
        """Create an _py_anon_pod1 instance wrapping the given NumPy array.

        Args:
            data (_numpy.ndarray): a single-element array of dtype `_py_anon_pod1_dtype` holding the data.
        """
        return __from_data(data, "_py_anon_pod1_dtype", _py_anon_pod1_dtype, _py_anon_pod1)

    @staticmethod
    def from_ptr(intptr_t ptr, bint readonly=False, object owner=None):
        """Create an _py_anon_pod1 instance wrapping the given pointer.

        Args:
            ptr (intptr_t): pointer address as Python :class:`int` to the data.
            owner (object): The Python object that owns the pointer. If not provided, data will be copied.
            readonly (bool): whether the data is read-only (to the user). default is `False`.
        """
        if ptr == 0:
            raise ValueError("ptr must not be null (0)")
        cdef _py_anon_pod1 obj = _py_anon_pod1.__new__(_py_anon_pod1)
        if owner is None:
            obj._ptr = <_anon_pod1 *>malloc(sizeof((<CUfileDescr_t*>NULL).handle))
            if obj._ptr == NULL:
                raise MemoryError("Error allocating _py_anon_pod1")
            memcpy(<void*>(obj._ptr), <void*>ptr, sizeof((<CUfileDescr_t*>NULL).handle))
            obj._owner = None
            obj._owned = True
        else:
            obj._ptr = <_anon_pod1 *>ptr
            obj._owner = owner
            obj._owned = False
        obj._readonly = readonly
        return obj


cdef _get__py_anon_pod3_dtype_offsets():
    cdef _anon_pod3 pod = _anon_pod3()
    return _numpy.dtype({
        'names': ['dev_ptr_base', 'file_offset', 'dev_ptr_offset', 'size_'],
        'formats': [_numpy.intp, _numpy.int64, _numpy.int64, _numpy.uint64],
        'offsets': [
            (<intptr_t>&(pod.devPtr_base)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.file_offset)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.devPtr_offset)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.size)) - (<intptr_t>&pod),
        ],
        'itemsize': sizeof((<CUfileIOParams_t*>NULL).u.batch),
    })

_py_anon_pod3_dtype = _get__py_anon_pod3_dtype_offsets()

cdef class _py_anon_pod3:
    """Empty-initialize an instance of `_anon_pod3`.


    .. seealso:: `_anon_pod3`
    """
    cdef:
        _anon_pod3 *_ptr
        object _owner
        bint _owned
        bint _readonly

    def __init__(self):
        self._ptr = <_anon_pod3 *>calloc(1, sizeof((<CUfileIOParams_t*>NULL).u.batch))
        if self._ptr == NULL:
            raise MemoryError("Error allocating _py_anon_pod3")
        self._owner = None
        self._owned = True
        self._readonly = False

    def __dealloc__(self):
        cdef _anon_pod3 *ptr
        if self._owned and self._ptr != NULL:
            ptr = self._ptr
            self._ptr = NULL
            free(ptr)

    def __repr__(self):
        return f"<{__name__}._py_anon_pod3 object at {hex(id(self))}>"

    @property
    def ptr(self):
        """Get the pointer address to the data as Python :class:`int`."""
        return <intptr_t>(self._ptr)

    cdef intptr_t _get_ptr(self):
        return <intptr_t>(self._ptr)

    def __int__(self):
        return <intptr_t>(self._ptr)

    def __eq__(self, other):
        cdef _py_anon_pod3 other_
        if not isinstance(other, _py_anon_pod3):
            return False
        other_ = other
        return (memcmp(<void *><intptr_t>(self._ptr), <void *><intptr_t>(other_._ptr), sizeof((<CUfileIOParams_t*>NULL).u.batch)) == 0)

    def __getbuffer__(self, Py_buffer *buffer, int flags):
        __getbuffer(self, buffer, <void *>self._ptr, sizeof((<CUfileIOParams_t*>NULL).u.batch), self._readonly)

    def __releasebuffer__(self, Py_buffer *buffer):
        pass

    def __setitem__(self, key, val):
        if key == 0 and isinstance(val, _numpy.ndarray):
            self._ptr = <_anon_pod3 *>malloc(sizeof((<CUfileIOParams_t*>NULL).u.batch))
            if self._ptr == NULL:
                raise MemoryError("Error allocating _py_anon_pod3")
            memcpy(<void*>self._ptr, <void*><intptr_t>val.ctypes.data, sizeof((<CUfileIOParams_t*>NULL).u.batch))
            self._owner = None
            self._owned = True
            self._readonly = not val.flags.writeable
        else:
            setattr(self, key, val)

    @property
    def dev_ptr_base(self):
        """int: """
        return <intptr_t>(self._ptr[0].devPtr_base)

    @dev_ptr_base.setter
    def dev_ptr_base(self, val):
        if self._readonly:
            raise ValueError("This _py_anon_pod3 instance is read-only")
        self._ptr[0].devPtr_base = <void *><intptr_t>val

    @property
    def file_offset(self):
        """int: """
        return self._ptr[0].file_offset

    @file_offset.setter
    def file_offset(self, val):
        if self._readonly:
            raise ValueError("This _py_anon_pod3 instance is read-only")
        self._ptr[0].file_offset = val

    @property
    def dev_ptr_offset(self):
        """int: """
        return self._ptr[0].devPtr_offset

    @dev_ptr_offset.setter
    def dev_ptr_offset(self, val):
        if self._readonly:
            raise ValueError("This _py_anon_pod3 instance is read-only")
        self._ptr[0].devPtr_offset = val

    @property
    def size_(self):
        """int: """
        return self._ptr[0].size

    @size_.setter
    def size_(self, val):
        if self._readonly:
            raise ValueError("This _py_anon_pod3 instance is read-only")
        self._ptr[0].size = val

    @staticmethod
    def from_buffer(buffer):
        """Create an _py_anon_pod3 instance with the memory from the given buffer."""
        return __from_buffer(buffer, sizeof((<CUfileIOParams_t*>NULL).u.batch), _py_anon_pod3)

    @staticmethod
    def from_data(data):
        """Create an _py_anon_pod3 instance wrapping the given NumPy array.

        Args:
            data (_numpy.ndarray): a single-element array of dtype `_py_anon_pod3_dtype` holding the data.
        """
        return __from_data(data, "_py_anon_pod3_dtype", _py_anon_pod3_dtype, _py_anon_pod3)

    @staticmethod
    def from_ptr(intptr_t ptr, bint readonly=False, object owner=None):
        """Create an _py_anon_pod3 instance wrapping the given pointer.

        Args:
            ptr (intptr_t): pointer address as Python :class:`int` to the data.
            owner (object): The Python object that owns the pointer. If not provided, data will be copied.
            readonly (bool): whether the data is read-only (to the user). default is `False`.
        """
        if ptr == 0:
            raise ValueError("ptr must not be null (0)")
        cdef _py_anon_pod3 obj = _py_anon_pod3.__new__(_py_anon_pod3)
        if owner is None:
            obj._ptr = <_anon_pod3 *>malloc(sizeof((<CUfileIOParams_t*>NULL).u.batch))
            if obj._ptr == NULL:
                raise MemoryError("Error allocating _py_anon_pod3")
            memcpy(<void*>(obj._ptr), <void*>ptr, sizeof((<CUfileIOParams_t*>NULL).u.batch))
            obj._owner = None
            obj._owned = True
        else:
            obj._ptr = <_anon_pod3 *>ptr
            obj._owner = owner
            obj._owned = False
        obj._readonly = readonly
        return obj


cdef _get_io_events_dtype_offsets():
    cdef CUfileIOEvents_t pod = CUfileIOEvents_t()
    return _numpy.dtype({
        'names': ['cookie', 'status', 'ret'],
        'formats': [_numpy.intp, _numpy.int32, _numpy.uint64],
        'offsets': [
            (<intptr_t>&(pod.cookie)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.status)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.ret)) - (<intptr_t>&pod),
        ],
        'itemsize': sizeof(CUfileIOEvents_t),
    })

io_events_dtype = _get_io_events_dtype_offsets()

cdef class IOEvents:
    """Empty-initialize an array of `CUfileIOEvents_t`.

    The resulting object is of length `size` and of dtype `io_events_dtype`.
    If default-constructed, the instance represents a single struct.

    Args:
        size (int): number of structs, default=1.


    .. seealso:: `CUfileIOEvents_t`
    """
    cdef:
        readonly object _data



    def __init__(self, size=1):
        arr = _numpy.empty(size, dtype=io_events_dtype)
        self._data = arr.view(_numpy.recarray)
        assert self._data.itemsize == sizeof(CUfileIOEvents_t), \
            f"itemsize {self._data.itemsize} mismatches struct size { sizeof(CUfileIOEvents_t) }"

    def __repr__(self):
        if self._data.size > 1:
            return f"<{__name__}.IOEvents_Array_{self._data.size} object at {hex(id(self))}>"
        else:
            return f"<{__name__}.IOEvents object at {hex(id(self))}>"

    @property
    def ptr(self):
        """Get the pointer address to the data as Python :class:`int`."""
        return self._data.ctypes.data

    cdef intptr_t _get_ptr(self):
        return self._data.ctypes.data

    def __int__(self):
        if self._data.size > 1:
            raise TypeError("int() argument must be a bytes-like object of size 1. "
                            "To get the pointer address of an array, use .ptr")
        return self._data.ctypes.data

    def __len__(self):
        return self._data.size

    def __eq__(self, other):
        cdef object self_data = self._data
        if (not isinstance(other, IOEvents)) or self_data.size != other._data.size or self_data.dtype != other._data.dtype:
            return False
        return bool((self_data == other._data).all())

    def __getbuffer__(self, Py_buffer *buffer, int flags):
        cpython.PyObject_GetBuffer(self._data, buffer, flags)

    def __releasebuffer__(self, Py_buffer *buffer):
        cpython.PyBuffer_Release(buffer)

    @property
    def cookie(self):
        """Union[~_numpy.intp, int]: """
        if self._data.size == 1:
            return int(self._data.cookie[0])
        return self._data.cookie

    @cookie.setter
    def cookie(self, val):
        self._data.cookie = val

    @property
    def status(self):
        """Union[~_numpy.int32, int]: """
        if self._data.size == 1:
            return int(self._data.status[0])
        return self._data.status

    @status.setter
    def status(self, val):
        self._data.status = val

    @property
    def ret(self):
        """Union[~_numpy.uint64, int]: """
        if self._data.size == 1:
            return int(self._data.ret[0])
        return self._data.ret

    @ret.setter
    def ret(self, val):
        self._data.ret = val

    def __getitem__(self, key):
        cdef ssize_t key_
        cdef ssize_t size
        if isinstance(key, int):
            key_ = key
            size = self._data.size
            if key_ >= size or key_ <= -(size+1):
                raise IndexError("index is out of bounds")
            if key_ < 0:
                key_ += size
            return IOEvents.from_data(self._data[key_:key_+1])
        out = self._data[key]
        if isinstance(out, _numpy.recarray) and out.dtype == io_events_dtype:
            return IOEvents.from_data(out)
        return out

    def __setitem__(self, key, val):
        self._data[key] = val

    @staticmethod
    def from_buffer(buffer):
        """Create an IOEvents instance with the memory from the given buffer."""
        return IOEvents.from_data(_numpy.frombuffer(buffer, dtype=io_events_dtype))

    @staticmethod
    def from_data(data):
        """Create an IOEvents instance wrapping the given NumPy array.

        Args:
            data (_numpy.ndarray): a 1D array of dtype `io_events_dtype` holding the data.
        """
        cdef IOEvents obj = IOEvents.__new__(IOEvents)
        if not isinstance(data, _numpy.ndarray):
            raise TypeError("data argument must be a NumPy ndarray")
        if data.ndim != 1:
            raise ValueError("data array must be 1D")
        if data.dtype != io_events_dtype:
            raise ValueError("data array must be of dtype io_events_dtype")
        obj._data = data.view(_numpy.recarray)

        return obj

    @staticmethod
    def from_ptr(intptr_t ptr, size_t size=1, bint readonly=False):
        """Create an IOEvents instance wrapping the given pointer.

        Args:
            ptr (intptr_t): pointer address as Python :class:`int` to the data.
            size (int): number of structs, default=1.
            readonly (bool): whether the data is read-only (to the user). default is `False`.
        """
        if ptr == 0:
            raise ValueError("ptr must not be null (0)")
        cdef IOEvents obj = IOEvents.__new__(IOEvents)
        cdef flag = cpython.buffer.PyBUF_READ if readonly else cpython.buffer.PyBUF_WRITE
        cdef object buf = cpython.memoryview.PyMemoryView_FromMemory(
            <char*>ptr, sizeof(CUfileIOEvents_t) * size, flag)
        data = _numpy.ndarray(size, buffer=buf, dtype=io_events_dtype)
        obj._data = data.view(_numpy.recarray)

        return obj


cdef _get_op_counter_dtype_offsets():
    cdef CUfileOpCounter_t pod = CUfileOpCounter_t()
    return _numpy.dtype({
        'names': ['ok', 'err'],
        'formats': [_numpy.uint64, _numpy.uint64],
        'offsets': [
            (<intptr_t>&(pod.ok)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.err)) - (<intptr_t>&pod),
        ],
        'itemsize': sizeof(CUfileOpCounter_t),
    })

op_counter_dtype = _get_op_counter_dtype_offsets()

cdef class OpCounter:
    """Empty-initialize an instance of `CUfileOpCounter_t`.


    .. seealso:: `CUfileOpCounter_t`
    """
    cdef:
        CUfileOpCounter_t *_ptr
        object _owner
        bint _owned
        bint _readonly

    def __init__(self):
        self._ptr = <CUfileOpCounter_t *>calloc(1, sizeof(CUfileOpCounter_t))
        if self._ptr == NULL:
            raise MemoryError("Error allocating OpCounter")
        self._owner = None
        self._owned = True
        self._readonly = False

    def __dealloc__(self):
        cdef CUfileOpCounter_t *ptr
        if self._owned and self._ptr != NULL:
            ptr = self._ptr
            self._ptr = NULL
            free(ptr)

    def __repr__(self):
        return f"<{__name__}.OpCounter object at {hex(id(self))}>"

    @property
    def ptr(self):
        """Get the pointer address to the data as Python :class:`int`."""
        return <intptr_t>(self._ptr)

    cdef intptr_t _get_ptr(self):
        return <intptr_t>(self._ptr)

    def __int__(self):
        return <intptr_t>(self._ptr)

    def __eq__(self, other):
        cdef OpCounter other_
        if not isinstance(other, OpCounter):
            return False
        other_ = other
        return (memcmp(<void *><intptr_t>(self._ptr), <void *><intptr_t>(other_._ptr), sizeof(CUfileOpCounter_t)) == 0)

    def __getbuffer__(self, Py_buffer *buffer, int flags):
        __getbuffer(self, buffer, <void *>self._ptr, sizeof(CUfileOpCounter_t), self._readonly)

    def __releasebuffer__(self, Py_buffer *buffer):
        pass

    def __setitem__(self, key, val):
        if key == 0 and isinstance(val, _numpy.ndarray):
            self._ptr = <CUfileOpCounter_t *>malloc(sizeof(CUfileOpCounter_t))
            if self._ptr == NULL:
                raise MemoryError("Error allocating OpCounter")
            memcpy(<void*>self._ptr, <void*><intptr_t>val.ctypes.data, sizeof(CUfileOpCounter_t))
            self._owner = None
            self._owned = True
            self._readonly = not val.flags.writeable
        else:
            setattr(self, key, val)

    @property
    def ok(self):
        """int: """
        return self._ptr[0].ok

    @ok.setter
    def ok(self, val):
        if self._readonly:
            raise ValueError("This OpCounter instance is read-only")
        self._ptr[0].ok = val

    @property
    def err(self):
        """int: """
        return self._ptr[0].err

    @err.setter
    def err(self, val):
        if self._readonly:
            raise ValueError("This OpCounter instance is read-only")
        self._ptr[0].err = val

    @staticmethod
    def from_buffer(buffer):
        """Create an OpCounter instance with the memory from the given buffer."""
        return __from_buffer(buffer, sizeof(CUfileOpCounter_t), OpCounter)

    @staticmethod
    def from_data(data):
        """Create an OpCounter instance wrapping the given NumPy array.

        Args:
            data (_numpy.ndarray): a single-element array of dtype `op_counter_dtype` holding the data.
        """
        return __from_data(data, "op_counter_dtype", op_counter_dtype, OpCounter)

    @staticmethod
    def from_ptr(intptr_t ptr, bint readonly=False, object owner=None):
        """Create an OpCounter instance wrapping the given pointer.

        Args:
            ptr (intptr_t): pointer address as Python :class:`int` to the data.
            owner (object): The Python object that owns the pointer. If not provided, data will be copied.
            readonly (bool): whether the data is read-only (to the user). default is `False`.
        """
        if ptr == 0:
            raise ValueError("ptr must not be null (0)")
        cdef OpCounter obj = OpCounter.__new__(OpCounter)
        if owner is None:
            obj._ptr = <CUfileOpCounter_t *>malloc(sizeof(CUfileOpCounter_t))
            if obj._ptr == NULL:
                raise MemoryError("Error allocating OpCounter")
            memcpy(<void*>(obj._ptr), <void*>ptr, sizeof(CUfileOpCounter_t))
            obj._owner = None
            obj._owned = True
        else:
            obj._ptr = <CUfileOpCounter_t *>ptr
            obj._owner = owner
            obj._owned = False
        obj._readonly = readonly
        return obj


cdef _get_per_gpu_stats_dtype_offsets():
    cdef CUfilePerGpuStats_t pod = CUfilePerGpuStats_t()
    return _numpy.dtype({
        'names': ['uuid', 'read_bytes', 'read_bw_bytes_per_sec', 'read_utilization', 'read_duration_us', 'n_total_reads', 'n_p2p_reads', 'n_nvfs_reads', 'n_posix_reads', 'n_unaligned_reads', 'n_dr_reads', 'n_sparse_regions', 'n_inline_regions', 'n_reads_err', 'writes_bytes', 'write_bw_bytes_per_sec', 'write_utilization', 'write_duration_us', 'n_total_writes', 'n_p2p_writes', 'n_nvfs_writes', 'n_posix_writes', 'n_unaligned_writes', 'n_dr_writes', 'n_writes_err', 'n_mmap', 'n_mmap_ok', 'n_mmap_err', 'n_mmap_free', 'reg_bytes'],
        'formats': [(_numpy.int8, 16), _numpy.uint64, _numpy.uint64, _numpy.uint64, _numpy.uint64, _numpy.uint64, _numpy.uint64, _numpy.uint64, _numpy.uint64, _numpy.uint64, _numpy.uint64, _numpy.uint64, _numpy.uint64, _numpy.uint64, _numpy.uint64, _numpy.uint64, _numpy.uint64, _numpy.uint64, _numpy.uint64, _numpy.uint64, _numpy.uint64, _numpy.uint64, _numpy.uint64, _numpy.uint64, _numpy.uint64, _numpy.uint64, _numpy.uint64, _numpy.uint64, _numpy.uint64, _numpy.uint64],
        'offsets': [
            (<intptr_t>&(pod.uuid)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.read_bytes)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.read_bw_bytes_per_sec)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.read_utilization)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.read_duration_us)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.n_total_reads)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.n_p2p_reads)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.n_nvfs_reads)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.n_posix_reads)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.n_unaligned_reads)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.n_dr_reads)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.n_sparse_regions)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.n_inline_regions)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.n_reads_err)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.writes_bytes)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.write_bw_bytes_per_sec)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.write_utilization)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.write_duration_us)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.n_total_writes)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.n_p2p_writes)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.n_nvfs_writes)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.n_posix_writes)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.n_unaligned_writes)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.n_dr_writes)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.n_writes_err)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.n_mmap)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.n_mmap_ok)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.n_mmap_err)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.n_mmap_free)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.reg_bytes)) - (<intptr_t>&pod),
        ],
        'itemsize': sizeof(CUfilePerGpuStats_t),
    })

per_gpu_stats_dtype = _get_per_gpu_stats_dtype_offsets()

cdef class PerGpuStats:
    """Empty-initialize an array of `CUfilePerGpuStats_t`.

    The resulting object is of length `size` and of dtype `per_gpu_stats_dtype`.
    If default-constructed, the instance represents a single struct.

    Args:
        size (int): number of structs, default=1.


    .. seealso:: `CUfilePerGpuStats_t`
    """
    cdef:
        readonly object _data



    def __init__(self, size=1):
        arr = _numpy.empty(size, dtype=per_gpu_stats_dtype)
        self._data = arr.view(_numpy.recarray)
        assert self._data.itemsize == sizeof(CUfilePerGpuStats_t), \
            f"itemsize {self._data.itemsize} mismatches struct size { sizeof(CUfilePerGpuStats_t) }"

    def __repr__(self):
        if self._data.size > 1:
            return f"<{__name__}.PerGpuStats_Array_{self._data.size} object at {hex(id(self))}>"
        else:
            return f"<{__name__}.PerGpuStats object at {hex(id(self))}>"

    @property
    def ptr(self):
        """Get the pointer address to the data as Python :class:`int`."""
        return self._data.ctypes.data

    cdef intptr_t _get_ptr(self):
        return self._data.ctypes.data

    def __int__(self):
        if self._data.size > 1:
            raise TypeError("int() argument must be a bytes-like object of size 1. "
                            "To get the pointer address of an array, use .ptr")
        return self._data.ctypes.data

    def __len__(self):
        return self._data.size

    def __eq__(self, other):
        cdef object self_data = self._data
        if (not isinstance(other, PerGpuStats)) or self_data.size != other._data.size or self_data.dtype != other._data.dtype:
            return False
        return bool((self_data == other._data).all())

    def __getbuffer__(self, Py_buffer *buffer, int flags):
        cpython.PyObject_GetBuffer(self._data, buffer, flags)

    def __releasebuffer__(self, Py_buffer *buffer):
        cpython.PyBuffer_Release(buffer)

    @property
    def uuid(self):
        """~_numpy.int8: (array of length 16)."""
        return self._data.uuid

    @uuid.setter
    def uuid(self, val):
        self._data.uuid = val

    @property
    def read_bytes(self):
        """Union[~_numpy.uint64, int]: """
        if self._data.size == 1:
            return int(self._data.read_bytes[0])
        return self._data.read_bytes

    @read_bytes.setter
    def read_bytes(self, val):
        self._data.read_bytes = val

    @property
    def read_bw_bytes_per_sec(self):
        """Union[~_numpy.uint64, int]: """
        if self._data.size == 1:
            return int(self._data.read_bw_bytes_per_sec[0])
        return self._data.read_bw_bytes_per_sec

    @read_bw_bytes_per_sec.setter
    def read_bw_bytes_per_sec(self, val):
        self._data.read_bw_bytes_per_sec = val

    @property
    def read_utilization(self):
        """Union[~_numpy.uint64, int]: """
        if self._data.size == 1:
            return int(self._data.read_utilization[0])
        return self._data.read_utilization

    @read_utilization.setter
    def read_utilization(self, val):
        self._data.read_utilization = val

    @property
    def read_duration_us(self):
        """Union[~_numpy.uint64, int]: """
        if self._data.size == 1:
            return int(self._data.read_duration_us[0])
        return self._data.read_duration_us

    @read_duration_us.setter
    def read_duration_us(self, val):
        self._data.read_duration_us = val

    @property
    def n_total_reads(self):
        """Union[~_numpy.uint64, int]: """
        if self._data.size == 1:
            return int(self._data.n_total_reads[0])
        return self._data.n_total_reads

    @n_total_reads.setter
    def n_total_reads(self, val):
        self._data.n_total_reads = val

    @property
    def n_p2p_reads(self):
        """Union[~_numpy.uint64, int]: """
        if self._data.size == 1:
            return int(self._data.n_p2p_reads[0])
        return self._data.n_p2p_reads

    @n_p2p_reads.setter
    def n_p2p_reads(self, val):
        self._data.n_p2p_reads = val

    @property
    def n_nvfs_reads(self):
        """Union[~_numpy.uint64, int]: """
        if self._data.size == 1:
            return int(self._data.n_nvfs_reads[0])
        return self._data.n_nvfs_reads

    @n_nvfs_reads.setter
    def n_nvfs_reads(self, val):
        self._data.n_nvfs_reads = val

    @property
    def n_posix_reads(self):
        """Union[~_numpy.uint64, int]: """
        if self._data.size == 1:
            return int(self._data.n_posix_reads[0])
        return self._data.n_posix_reads

    @n_posix_reads.setter
    def n_posix_reads(self, val):
        self._data.n_posix_reads = val

    @property
    def n_unaligned_reads(self):
        """Union[~_numpy.uint64, int]: """
        if self._data.size == 1:
            return int(self._data.n_unaligned_reads[0])
        return self._data.n_unaligned_reads

    @n_unaligned_reads.setter
    def n_unaligned_reads(self, val):
        self._data.n_unaligned_reads = val

    @property
    def n_dr_reads(self):
        """Union[~_numpy.uint64, int]: """
        if self._data.size == 1:
            return int(self._data.n_dr_reads[0])
        return self._data.n_dr_reads

    @n_dr_reads.setter
    def n_dr_reads(self, val):
        self._data.n_dr_reads = val

    @property
    def n_sparse_regions(self):
        """Union[~_numpy.uint64, int]: """
        if self._data.size == 1:
            return int(self._data.n_sparse_regions[0])
        return self._data.n_sparse_regions

    @n_sparse_regions.setter
    def n_sparse_regions(self, val):
        self._data.n_sparse_regions = val

    @property
    def n_inline_regions(self):
        """Union[~_numpy.uint64, int]: """
        if self._data.size == 1:
            return int(self._data.n_inline_regions[0])
        return self._data.n_inline_regions

    @n_inline_regions.setter
    def n_inline_regions(self, val):
        self._data.n_inline_regions = val

    @property
    def n_reads_err(self):
        """Union[~_numpy.uint64, int]: """
        if self._data.size == 1:
            return int(self._data.n_reads_err[0])
        return self._data.n_reads_err

    @n_reads_err.setter
    def n_reads_err(self, val):
        self._data.n_reads_err = val

    @property
    def writes_bytes(self):
        """Union[~_numpy.uint64, int]: """
        if self._data.size == 1:
            return int(self._data.writes_bytes[0])
        return self._data.writes_bytes

    @writes_bytes.setter
    def writes_bytes(self, val):
        self._data.writes_bytes = val

    @property
    def write_bw_bytes_per_sec(self):
        """Union[~_numpy.uint64, int]: """
        if self._data.size == 1:
            return int(self._data.write_bw_bytes_per_sec[0])
        return self._data.write_bw_bytes_per_sec

    @write_bw_bytes_per_sec.setter
    def write_bw_bytes_per_sec(self, val):
        self._data.write_bw_bytes_per_sec = val

    @property
    def write_utilization(self):
        """Union[~_numpy.uint64, int]: """
        if self._data.size == 1:
            return int(self._data.write_utilization[0])
        return self._data.write_utilization

    @write_utilization.setter
    def write_utilization(self, val):
        self._data.write_utilization = val

    @property
    def write_duration_us(self):
        """Union[~_numpy.uint64, int]: """
        if self._data.size == 1:
            return int(self._data.write_duration_us[0])
        return self._data.write_duration_us

    @write_duration_us.setter
    def write_duration_us(self, val):
        self._data.write_duration_us = val

    @property
    def n_total_writes(self):
        """Union[~_numpy.uint64, int]: """
        if self._data.size == 1:
            return int(self._data.n_total_writes[0])
        return self._data.n_total_writes

    @n_total_writes.setter
    def n_total_writes(self, val):
        self._data.n_total_writes = val

    @property
    def n_p2p_writes(self):
        """Union[~_numpy.uint64, int]: """
        if self._data.size == 1:
            return int(self._data.n_p2p_writes[0])
        return self._data.n_p2p_writes

    @n_p2p_writes.setter
    def n_p2p_writes(self, val):
        self._data.n_p2p_writes = val

    @property
    def n_nvfs_writes(self):
        """Union[~_numpy.uint64, int]: """
        if self._data.size == 1:
            return int(self._data.n_nvfs_writes[0])
        return self._data.n_nvfs_writes

    @n_nvfs_writes.setter
    def n_nvfs_writes(self, val):
        self._data.n_nvfs_writes = val

    @property
    def n_posix_writes(self):
        """Union[~_numpy.uint64, int]: """
        if self._data.size == 1:
            return int(self._data.n_posix_writes[0])
        return self._data.n_posix_writes

    @n_posix_writes.setter
    def n_posix_writes(self, val):
        self._data.n_posix_writes = val

    @property
    def n_unaligned_writes(self):
        """Union[~_numpy.uint64, int]: """
        if self._data.size == 1:
            return int(self._data.n_unaligned_writes[0])
        return self._data.n_unaligned_writes

    @n_unaligned_writes.setter
    def n_unaligned_writes(self, val):
        self._data.n_unaligned_writes = val

    @property
    def n_dr_writes(self):
        """Union[~_numpy.uint64, int]: """
        if self._data.size == 1:
            return int(self._data.n_dr_writes[0])
        return self._data.n_dr_writes

    @n_dr_writes.setter
    def n_dr_writes(self, val):
        self._data.n_dr_writes = val

    @property
    def n_writes_err(self):
        """Union[~_numpy.uint64, int]: """
        if self._data.size == 1:
            return int(self._data.n_writes_err[0])
        return self._data.n_writes_err

    @n_writes_err.setter
    def n_writes_err(self, val):
        self._data.n_writes_err = val

    @property
    def n_mmap(self):
        """Union[~_numpy.uint64, int]: """
        if self._data.size == 1:
            return int(self._data.n_mmap[0])
        return self._data.n_mmap

    @n_mmap.setter
    def n_mmap(self, val):
        self._data.n_mmap = val

    @property
    def n_mmap_ok(self):
        """Union[~_numpy.uint64, int]: """
        if self._data.size == 1:
            return int(self._data.n_mmap_ok[0])
        return self._data.n_mmap_ok

    @n_mmap_ok.setter
    def n_mmap_ok(self, val):
        self._data.n_mmap_ok = val

    @property
    def n_mmap_err(self):
        """Union[~_numpy.uint64, int]: """
        if self._data.size == 1:
            return int(self._data.n_mmap_err[0])
        return self._data.n_mmap_err

    @n_mmap_err.setter
    def n_mmap_err(self, val):
        self._data.n_mmap_err = val

    @property
    def n_mmap_free(self):
        """Union[~_numpy.uint64, int]: """
        if self._data.size == 1:
            return int(self._data.n_mmap_free[0])
        return self._data.n_mmap_free

    @n_mmap_free.setter
    def n_mmap_free(self, val):
        self._data.n_mmap_free = val

    @property
    def reg_bytes(self):
        """Union[~_numpy.uint64, int]: """
        if self._data.size == 1:
            return int(self._data.reg_bytes[0])
        return self._data.reg_bytes

    @reg_bytes.setter
    def reg_bytes(self, val):
        self._data.reg_bytes = val

    def __getitem__(self, key):
        cdef ssize_t key_
        cdef ssize_t size
        if isinstance(key, int):
            key_ = key
            size = self._data.size
            if key_ >= size or key_ <= -(size+1):
                raise IndexError("index is out of bounds")
            if key_ < 0:
                key_ += size
            return PerGpuStats.from_data(self._data[key_:key_+1])
        out = self._data[key]
        if isinstance(out, _numpy.recarray) and out.dtype == per_gpu_stats_dtype:
            return PerGpuStats.from_data(out)
        return out

    def __setitem__(self, key, val):
        self._data[key] = val

    @staticmethod
    def from_buffer(buffer):
        """Create an PerGpuStats instance with the memory from the given buffer."""
        return PerGpuStats.from_data(_numpy.frombuffer(buffer, dtype=per_gpu_stats_dtype))

    @staticmethod
    def from_data(data):
        """Create an PerGpuStats instance wrapping the given NumPy array.

        Args:
            data (_numpy.ndarray): a 1D array of dtype `per_gpu_stats_dtype` holding the data.
        """
        cdef PerGpuStats obj = PerGpuStats.__new__(PerGpuStats)
        if not isinstance(data, _numpy.ndarray):
            raise TypeError("data argument must be a NumPy ndarray")
        if data.ndim != 1:
            raise ValueError("data array must be 1D")
        if data.dtype != per_gpu_stats_dtype:
            raise ValueError("data array must be of dtype per_gpu_stats_dtype")
        obj._data = data.view(_numpy.recarray)

        return obj

    @staticmethod
    def from_ptr(intptr_t ptr, size_t size=1, bint readonly=False):
        """Create an PerGpuStats instance wrapping the given pointer.

        Args:
            ptr (intptr_t): pointer address as Python :class:`int` to the data.
            size (int): number of structs, default=1.
            readonly (bool): whether the data is read-only (to the user). default is `False`.
        """
        if ptr == 0:
            raise ValueError("ptr must not be null (0)")
        cdef PerGpuStats obj = PerGpuStats.__new__(PerGpuStats)
        cdef flag = cpython.buffer.PyBUF_READ if readonly else cpython.buffer.PyBUF_WRITE
        cdef object buf = cpython.memoryview.PyMemoryView_FromMemory(
            <char*>ptr, sizeof(CUfilePerGpuStats_t) * size, flag)
        data = _numpy.ndarray(size, buffer=buf, dtype=per_gpu_stats_dtype)
        obj._data = data.view(_numpy.recarray)

        return obj


cdef _get_descr_dtype_offsets():
    cdef CUfileDescr_t pod = CUfileDescr_t()
    return _numpy.dtype({
        'names': ['type', 'handle', 'fs_ops'],
        'formats': [_numpy.int32, _py_anon_pod1_dtype, _numpy.intp],
        'offsets': [
            (<intptr_t>&(pod.type)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.handle)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.fs_ops)) - (<intptr_t>&pod),
        ],
        'itemsize': sizeof(CUfileDescr_t),
    })

descr_dtype = _get_descr_dtype_offsets()

cdef class Descr:
    """Empty-initialize an array of `CUfileDescr_t`.

    The resulting object is of length `size` and of dtype `descr_dtype`.
    If default-constructed, the instance represents a single struct.

    Args:
        size (int): number of structs, default=1.


    .. seealso:: `CUfileDescr_t`
    """
    cdef:
        readonly object _data



    def __init__(self, size=1):
        arr = _numpy.empty(size, dtype=descr_dtype)
        self._data = arr.view(_numpy.recarray)
        assert self._data.itemsize == sizeof(CUfileDescr_t), \
            f"itemsize {self._data.itemsize} mismatches struct size { sizeof(CUfileDescr_t) }"

    def __repr__(self):
        if self._data.size > 1:
            return f"<{__name__}.Descr_Array_{self._data.size} object at {hex(id(self))}>"
        else:
            return f"<{__name__}.Descr object at {hex(id(self))}>"

    @property
    def ptr(self):
        """Get the pointer address to the data as Python :class:`int`."""
        return self._data.ctypes.data

    cdef intptr_t _get_ptr(self):
        return self._data.ctypes.data

    def __int__(self):
        if self._data.size > 1:
            raise TypeError("int() argument must be a bytes-like object of size 1. "
                            "To get the pointer address of an array, use .ptr")
        return self._data.ctypes.data

    def __len__(self):
        return self._data.size

    def __eq__(self, other):
        cdef object self_data = self._data
        if (not isinstance(other, Descr)) or self_data.size != other._data.size or self_data.dtype != other._data.dtype:
            return False
        return bool((self_data == other._data).all())

    def __getbuffer__(self, Py_buffer *buffer, int flags):
        cpython.PyObject_GetBuffer(self._data, buffer, flags)

    def __releasebuffer__(self, Py_buffer *buffer):
        cpython.PyBuffer_Release(buffer)

    @property
    def type(self):
        """Union[~_numpy.int32, int]: """
        if self._data.size == 1:
            return int(self._data.type[0])
        return self._data.type

    @type.setter
    def type(self, val):
        self._data.type = val

    @property
    def handle(self):
        """_py_anon_pod1_dtype: """
        return self._data.handle

    @handle.setter
    def handle(self, val):
        self._data.handle = val

    @property
    def fs_ops(self):
        """Union[~_numpy.intp, int]: """
        if self._data.size == 1:
            return int(self._data.fs_ops[0])
        return self._data.fs_ops

    @fs_ops.setter
    def fs_ops(self, val):
        self._data.fs_ops = val

    def __getitem__(self, key):
        cdef ssize_t key_
        cdef ssize_t size
        if isinstance(key, int):
            key_ = key
            size = self._data.size
            if key_ >= size or key_ <= -(size+1):
                raise IndexError("index is out of bounds")
            if key_ < 0:
                key_ += size
            return Descr.from_data(self._data[key_:key_+1])
        out = self._data[key]
        if isinstance(out, _numpy.recarray) and out.dtype == descr_dtype:
            return Descr.from_data(out)
        return out

    def __setitem__(self, key, val):
        self._data[key] = val

    @staticmethod
    def from_buffer(buffer):
        """Create an Descr instance with the memory from the given buffer."""
        return Descr.from_data(_numpy.frombuffer(buffer, dtype=descr_dtype))

    @staticmethod
    def from_data(data):
        """Create an Descr instance wrapping the given NumPy array.

        Args:
            data (_numpy.ndarray): a 1D array of dtype `descr_dtype` holding the data.
        """
        cdef Descr obj = Descr.__new__(Descr)
        if not isinstance(data, _numpy.ndarray):
            raise TypeError("data argument must be a NumPy ndarray")
        if data.ndim != 1:
            raise ValueError("data array must be 1D")
        if data.dtype != descr_dtype:
            raise ValueError("data array must be of dtype descr_dtype")
        obj._data = data.view(_numpy.recarray)

        return obj

    @staticmethod
    def from_ptr(intptr_t ptr, size_t size=1, bint readonly=False):
        """Create an Descr instance wrapping the given pointer.

        Args:
            ptr (intptr_t): pointer address as Python :class:`int` to the data.
            size (int): number of structs, default=1.
            readonly (bool): whether the data is read-only (to the user). default is `False`.
        """
        if ptr == 0:
            raise ValueError("ptr must not be null (0)")
        cdef Descr obj = Descr.__new__(Descr)
        cdef flag = cpython.buffer.PyBUF_READ if readonly else cpython.buffer.PyBUF_WRITE
        cdef object buf = cpython.memoryview.PyMemoryView_FromMemory(
            <char*>ptr, sizeof(CUfileDescr_t) * size, flag)
        data = _numpy.ndarray(size, buffer=buf, dtype=descr_dtype)
        obj._data = data.view(_numpy.recarray)

        return obj


_py_anon_pod2_dtype = _numpy.dtype((
    _numpy.dtype((_numpy.void, sizeof((<CUfileIOParams_t*>NULL).u))),
    {
        "batch": (_py_anon_pod3_dtype, 0),
    }
    ))


cdef class _py_anon_pod2:
    """Empty-initialize an instance of `_anon_pod2`.


    .. seealso:: `_anon_pod2`
    """
    cdef:
        _anon_pod2 *_ptr
        object _owner
        bint _owned
        bint _readonly

    def __init__(self):
        self._ptr = <_anon_pod2 *>calloc(1, sizeof((<CUfileIOParams_t*>NULL).u))
        if self._ptr == NULL:
            raise MemoryError("Error allocating _py_anon_pod2")
        self._owner = None
        self._owned = True
        self._readonly = False

    def __dealloc__(self):
        cdef _anon_pod2 *ptr
        if self._owned and self._ptr != NULL:
            ptr = self._ptr
            self._ptr = NULL
            free(ptr)

    def __repr__(self):
        return f"<{__name__}._py_anon_pod2 object at {hex(id(self))}>"

    @property
    def ptr(self):
        """Get the pointer address to the data as Python :class:`int`."""
        return <intptr_t>(self._ptr)

    cdef intptr_t _get_ptr(self):
        return <intptr_t>(self._ptr)

    def __int__(self):
        return <intptr_t>(self._ptr)

    def __eq__(self, other):
        cdef _py_anon_pod2 other_
        if not isinstance(other, _py_anon_pod2):
            return False
        other_ = other
        return (memcmp(<void *><intptr_t>(self._ptr), <void *><intptr_t>(other_._ptr), sizeof((<CUfileIOParams_t*>NULL).u)) == 0)

    def __getbuffer__(self, Py_buffer *buffer, int flags):
        __getbuffer(self, buffer, <void *>self._ptr, sizeof((<CUfileIOParams_t*>NULL).u), self._readonly)

    def __releasebuffer__(self, Py_buffer *buffer):
        pass

    def __setitem__(self, key, val):
        if key == 0 and isinstance(val, _numpy.ndarray):
            self._ptr = <_anon_pod2 *>malloc(sizeof((<CUfileIOParams_t*>NULL).u))
            if self._ptr == NULL:
                raise MemoryError("Error allocating _py_anon_pod2")
            memcpy(<void*>self._ptr, <void*><intptr_t>val.ctypes.data, sizeof((<CUfileIOParams_t*>NULL).u))
            self._owner = None
            self._owned = True
            self._readonly = not val.flags.writeable
        else:
            setattr(self, key, val)

    @property
    def batch(self):
        """_py_anon_pod3: """
        return _py_anon_pod3.from_ptr(<intptr_t>&(self._ptr[0].batch), self._readonly, self)

    @batch.setter
    def batch(self, val):
        if self._readonly:
            raise ValueError("This _py_anon_pod2 instance is read-only")
        cdef _py_anon_pod3 val_ = val
        memcpy(<void *>&(self._ptr[0].batch), <void *>(val_._get_ptr()), sizeof(_anon_pod3) * 1)

    @staticmethod
    def from_buffer(buffer):
        """Create an _py_anon_pod2 instance with the memory from the given buffer."""
        return __from_buffer(buffer, sizeof((<CUfileIOParams_t*>NULL).u), _py_anon_pod2)

    @staticmethod
    def from_data(data):
        """Create an _py_anon_pod2 instance wrapping the given NumPy array.

        Args:
            data (_numpy.ndarray): a single-element array of dtype `_py_anon_pod2_dtype` holding the data.
        """
        return __from_data(data, "_py_anon_pod2_dtype", _py_anon_pod2_dtype, _py_anon_pod2)

    @staticmethod
    def from_ptr(intptr_t ptr, bint readonly=False, object owner=None):
        """Create an _py_anon_pod2 instance wrapping the given pointer.

        Args:
            ptr (intptr_t): pointer address as Python :class:`int` to the data.
            owner (object): The Python object that owns the pointer. If not provided, data will be copied.
            readonly (bool): whether the data is read-only (to the user). default is `False`.
        """
        if ptr == 0:
            raise ValueError("ptr must not be null (0)")
        cdef _py_anon_pod2 obj = _py_anon_pod2.__new__(_py_anon_pod2)
        if owner is None:
            obj._ptr = <_anon_pod2 *>malloc(sizeof((<CUfileIOParams_t*>NULL).u))
            if obj._ptr == NULL:
                raise MemoryError("Error allocating _py_anon_pod2")
            memcpy(<void*>(obj._ptr), <void*>ptr, sizeof((<CUfileIOParams_t*>NULL).u))
            obj._owner = None
            obj._owned = True
        else:
            obj._ptr = <_anon_pod2 *>ptr
            obj._owner = owner
            obj._owned = False
        obj._readonly = readonly
        return obj


cdef _get_stats_level1_dtype_offsets():
    cdef CUfileStatsLevel1_t pod = CUfileStatsLevel1_t()
    return _numpy.dtype({
        'names': ['read_ops', 'write_ops', 'hdl_register_ops', 'hdl_deregister_ops', 'buf_register_ops', 'buf_deregister_ops', 'read_bytes', 'write_bytes', 'read_bw_bytes_per_sec', 'write_bw_bytes_per_sec', 'read_lat_avg_us', 'write_lat_avg_us', 'read_ops_per_sec', 'write_ops_per_sec', 'read_lat_sum_us', 'write_lat_sum_us', 'batch_submit_ops', 'batch_complete_ops', 'batch_setup_ops', 'batch_cancel_ops', 'batch_destroy_ops', 'batch_enqueued_ops', 'batch_posix_enqueued_ops', 'batch_processed_ops', 'batch_posix_processed_ops', 'batch_nvfs_submit_ops', 'batch_p2p_submit_ops', 'batch_aio_submit_ops', 'batch_iouring_submit_ops', 'batch_mixed_io_submit_ops', 'batch_total_submit_ops', 'batch_read_bytes', 'batch_write_bytes', 'batch_read_bw_bytes', 'batch_write_bw_bytes', 'batch_submit_lat_avg_us', 'batch_completion_lat_avg_us', 'batch_submit_ops_per_sec', 'batch_complete_ops_per_sec', 'batch_submit_lat_sum_us', 'batch_completion_lat_sum_us', 'last_batch_read_bytes', 'last_batch_write_bytes'],
        'formats': [op_counter_dtype, op_counter_dtype, op_counter_dtype, op_counter_dtype, op_counter_dtype, op_counter_dtype, _numpy.uint64, _numpy.uint64, _numpy.uint64, _numpy.uint64, _numpy.uint64, _numpy.uint64, _numpy.uint64, _numpy.uint64, _numpy.uint64, _numpy.uint64, op_counter_dtype, op_counter_dtype, op_counter_dtype, op_counter_dtype, op_counter_dtype, op_counter_dtype, op_counter_dtype, op_counter_dtype, op_counter_dtype, op_counter_dtype, op_counter_dtype, op_counter_dtype, op_counter_dtype, op_counter_dtype, op_counter_dtype, _numpy.uint64, _numpy.uint64, _numpy.uint64, _numpy.uint64, _numpy.uint64, _numpy.uint64, _numpy.uint64, _numpy.uint64, _numpy.uint64, _numpy.uint64, _numpy.uint64, _numpy.uint64],
        'offsets': [
            (<intptr_t>&(pod.read_ops)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.write_ops)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.hdl_register_ops)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.hdl_deregister_ops)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.buf_register_ops)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.buf_deregister_ops)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.read_bytes)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.write_bytes)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.read_bw_bytes_per_sec)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.write_bw_bytes_per_sec)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.read_lat_avg_us)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.write_lat_avg_us)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.read_ops_per_sec)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.write_ops_per_sec)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.read_lat_sum_us)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.write_lat_sum_us)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.batch_submit_ops)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.batch_complete_ops)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.batch_setup_ops)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.batch_cancel_ops)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.batch_destroy_ops)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.batch_enqueued_ops)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.batch_posix_enqueued_ops)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.batch_processed_ops)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.batch_posix_processed_ops)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.batch_nvfs_submit_ops)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.batch_p2p_submit_ops)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.batch_aio_submit_ops)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.batch_iouring_submit_ops)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.batch_mixed_io_submit_ops)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.batch_total_submit_ops)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.batch_read_bytes)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.batch_write_bytes)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.batch_read_bw_bytes)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.batch_write_bw_bytes)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.batch_submit_lat_avg_us)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.batch_completion_lat_avg_us)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.batch_submit_ops_per_sec)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.batch_complete_ops_per_sec)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.batch_submit_lat_sum_us)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.batch_completion_lat_sum_us)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.last_batch_read_bytes)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.last_batch_write_bytes)) - (<intptr_t>&pod),
        ],
        'itemsize': sizeof(CUfileStatsLevel1_t),
    })

stats_level1_dtype = _get_stats_level1_dtype_offsets()

cdef class StatsLevel1:
    """Empty-initialize an instance of `CUfileStatsLevel1_t`.


    .. seealso:: `CUfileStatsLevel1_t`
    """
    cdef:
        CUfileStatsLevel1_t *_ptr
        object _owner
        bint _owned
        bint _readonly

    def __init__(self):
        self._ptr = <CUfileStatsLevel1_t *>calloc(1, sizeof(CUfileStatsLevel1_t))
        if self._ptr == NULL:
            raise MemoryError("Error allocating StatsLevel1")
        self._owner = None
        self._owned = True
        self._readonly = False

    def __dealloc__(self):
        cdef CUfileStatsLevel1_t *ptr
        if self._owned and self._ptr != NULL:
            ptr = self._ptr
            self._ptr = NULL
            free(ptr)

    def __repr__(self):
        return f"<{__name__}.StatsLevel1 object at {hex(id(self))}>"

    @property
    def ptr(self):
        """Get the pointer address to the data as Python :class:`int`."""
        return <intptr_t>(self._ptr)

    cdef intptr_t _get_ptr(self):
        return <intptr_t>(self._ptr)

    def __int__(self):
        return <intptr_t>(self._ptr)

    def __eq__(self, other):
        cdef StatsLevel1 other_
        if not isinstance(other, StatsLevel1):
            return False
        other_ = other
        return (memcmp(<void *><intptr_t>(self._ptr), <void *><intptr_t>(other_._ptr), sizeof(CUfileStatsLevel1_t)) == 0)

    def __getbuffer__(self, Py_buffer *buffer, int flags):
        __getbuffer(self, buffer, <void *>self._ptr, sizeof(CUfileStatsLevel1_t), self._readonly)

    def __releasebuffer__(self, Py_buffer *buffer):
        pass

    def __setitem__(self, key, val):
        if key == 0 and isinstance(val, _numpy.ndarray):
            self._ptr = <CUfileStatsLevel1_t *>malloc(sizeof(CUfileStatsLevel1_t))
            if self._ptr == NULL:
                raise MemoryError("Error allocating StatsLevel1")
            memcpy(<void*>self._ptr, <void*><intptr_t>val.ctypes.data, sizeof(CUfileStatsLevel1_t))
            self._owner = None
            self._owned = True
            self._readonly = not val.flags.writeable
        else:
            setattr(self, key, val)

    @property
    def read_ops(self):
        """OpCounter: """
        return OpCounter.from_ptr(<intptr_t>&(self._ptr[0].read_ops), self._readonly, self)

    @read_ops.setter
    def read_ops(self, val):
        if self._readonly:
            raise ValueError("This StatsLevel1 instance is read-only")
        cdef OpCounter val_ = val
        memcpy(<void *>&(self._ptr[0].read_ops), <void *>(val_._get_ptr()), sizeof(CUfileOpCounter_t) * 1)

    @property
    def write_ops(self):
        """OpCounter: """
        return OpCounter.from_ptr(<intptr_t>&(self._ptr[0].write_ops), self._readonly, self)

    @write_ops.setter
    def write_ops(self, val):
        if self._readonly:
            raise ValueError("This StatsLevel1 instance is read-only")
        cdef OpCounter val_ = val
        memcpy(<void *>&(self._ptr[0].write_ops), <void *>(val_._get_ptr()), sizeof(CUfileOpCounter_t) * 1)

    @property
    def hdl_register_ops(self):
        """OpCounter: """
        return OpCounter.from_ptr(<intptr_t>&(self._ptr[0].hdl_register_ops), self._readonly, self)

    @hdl_register_ops.setter
    def hdl_register_ops(self, val):
        if self._readonly:
            raise ValueError("This StatsLevel1 instance is read-only")
        cdef OpCounter val_ = val
        memcpy(<void *>&(self._ptr[0].hdl_register_ops), <void *>(val_._get_ptr()), sizeof(CUfileOpCounter_t) * 1)

    @property
    def hdl_deregister_ops(self):
        """OpCounter: """
        return OpCounter.from_ptr(<intptr_t>&(self._ptr[0].hdl_deregister_ops), self._readonly, self)

    @hdl_deregister_ops.setter
    def hdl_deregister_ops(self, val):
        if self._readonly:
            raise ValueError("This StatsLevel1 instance is read-only")
        cdef OpCounter val_ = val
        memcpy(<void *>&(self._ptr[0].hdl_deregister_ops), <void *>(val_._get_ptr()), sizeof(CUfileOpCounter_t) * 1)

    @property
    def buf_register_ops(self):
        """OpCounter: """
        return OpCounter.from_ptr(<intptr_t>&(self._ptr[0].buf_register_ops), self._readonly, self)

    @buf_register_ops.setter
    def buf_register_ops(self, val):
        if self._readonly:
            raise ValueError("This StatsLevel1 instance is read-only")
        cdef OpCounter val_ = val
        memcpy(<void *>&(self._ptr[0].buf_register_ops), <void *>(val_._get_ptr()), sizeof(CUfileOpCounter_t) * 1)

    @property
    def buf_deregister_ops(self):
        """OpCounter: """
        return OpCounter.from_ptr(<intptr_t>&(self._ptr[0].buf_deregister_ops), self._readonly, self)

    @buf_deregister_ops.setter
    def buf_deregister_ops(self, val):
        if self._readonly:
            raise ValueError("This StatsLevel1 instance is read-only")
        cdef OpCounter val_ = val
        memcpy(<void *>&(self._ptr[0].buf_deregister_ops), <void *>(val_._get_ptr()), sizeof(CUfileOpCounter_t) * 1)

    @property
    def batch_submit_ops(self):
        """OpCounter: """
        return OpCounter.from_ptr(<intptr_t>&(self._ptr[0].batch_submit_ops), self._readonly, self)

    @batch_submit_ops.setter
    def batch_submit_ops(self, val):
        if self._readonly:
            raise ValueError("This StatsLevel1 instance is read-only")
        cdef OpCounter val_ = val
        memcpy(<void *>&(self._ptr[0].batch_submit_ops), <void *>(val_._get_ptr()), sizeof(CUfileOpCounter_t) * 1)

    @property
    def batch_complete_ops(self):
        """OpCounter: """
        return OpCounter.from_ptr(<intptr_t>&(self._ptr[0].batch_complete_ops), self._readonly, self)

    @batch_complete_ops.setter
    def batch_complete_ops(self, val):
        if self._readonly:
            raise ValueError("This StatsLevel1 instance is read-only")
        cdef OpCounter val_ = val
        memcpy(<void *>&(self._ptr[0].batch_complete_ops), <void *>(val_._get_ptr()), sizeof(CUfileOpCounter_t) * 1)

    @property
    def batch_setup_ops(self):
        """OpCounter: """
        return OpCounter.from_ptr(<intptr_t>&(self._ptr[0].batch_setup_ops), self._readonly, self)

    @batch_setup_ops.setter
    def batch_setup_ops(self, val):
        if self._readonly:
            raise ValueError("This StatsLevel1 instance is read-only")
        cdef OpCounter val_ = val
        memcpy(<void *>&(self._ptr[0].batch_setup_ops), <void *>(val_._get_ptr()), sizeof(CUfileOpCounter_t) * 1)

    @property
    def batch_cancel_ops(self):
        """OpCounter: """
        return OpCounter.from_ptr(<intptr_t>&(self._ptr[0].batch_cancel_ops), self._readonly, self)

    @batch_cancel_ops.setter
    def batch_cancel_ops(self, val):
        if self._readonly:
            raise ValueError("This StatsLevel1 instance is read-only")
        cdef OpCounter val_ = val
        memcpy(<void *>&(self._ptr[0].batch_cancel_ops), <void *>(val_._get_ptr()), sizeof(CUfileOpCounter_t) * 1)

    @property
    def batch_destroy_ops(self):
        """OpCounter: """
        return OpCounter.from_ptr(<intptr_t>&(self._ptr[0].batch_destroy_ops), self._readonly, self)

    @batch_destroy_ops.setter
    def batch_destroy_ops(self, val):
        if self._readonly:
            raise ValueError("This StatsLevel1 instance is read-only")
        cdef OpCounter val_ = val
        memcpy(<void *>&(self._ptr[0].batch_destroy_ops), <void *>(val_._get_ptr()), sizeof(CUfileOpCounter_t) * 1)

    @property
    def batch_enqueued_ops(self):
        """OpCounter: """
        return OpCounter.from_ptr(<intptr_t>&(self._ptr[0].batch_enqueued_ops), self._readonly, self)

    @batch_enqueued_ops.setter
    def batch_enqueued_ops(self, val):
        if self._readonly:
            raise ValueError("This StatsLevel1 instance is read-only")
        cdef OpCounter val_ = val
        memcpy(<void *>&(self._ptr[0].batch_enqueued_ops), <void *>(val_._get_ptr()), sizeof(CUfileOpCounter_t) * 1)

    @property
    def batch_posix_enqueued_ops(self):
        """OpCounter: """
        return OpCounter.from_ptr(<intptr_t>&(self._ptr[0].batch_posix_enqueued_ops), self._readonly, self)

    @batch_posix_enqueued_ops.setter
    def batch_posix_enqueued_ops(self, val):
        if self._readonly:
            raise ValueError("This StatsLevel1 instance is read-only")
        cdef OpCounter val_ = val
        memcpy(<void *>&(self._ptr[0].batch_posix_enqueued_ops), <void *>(val_._get_ptr()), sizeof(CUfileOpCounter_t) * 1)

    @property
    def batch_processed_ops(self):
        """OpCounter: """
        return OpCounter.from_ptr(<intptr_t>&(self._ptr[0].batch_processed_ops), self._readonly, self)

    @batch_processed_ops.setter
    def batch_processed_ops(self, val):
        if self._readonly:
            raise ValueError("This StatsLevel1 instance is read-only")
        cdef OpCounter val_ = val
        memcpy(<void *>&(self._ptr[0].batch_processed_ops), <void *>(val_._get_ptr()), sizeof(CUfileOpCounter_t) * 1)

    @property
    def batch_posix_processed_ops(self):
        """OpCounter: """
        return OpCounter.from_ptr(<intptr_t>&(self._ptr[0].batch_posix_processed_ops), self._readonly, self)

    @batch_posix_processed_ops.setter
    def batch_posix_processed_ops(self, val):
        if self._readonly:
            raise ValueError("This StatsLevel1 instance is read-only")
        cdef OpCounter val_ = val
        memcpy(<void *>&(self._ptr[0].batch_posix_processed_ops), <void *>(val_._get_ptr()), sizeof(CUfileOpCounter_t) * 1)

    @property
    def batch_nvfs_submit_ops(self):
        """OpCounter: """
        return OpCounter.from_ptr(<intptr_t>&(self._ptr[0].batch_nvfs_submit_ops), self._readonly, self)

    @batch_nvfs_submit_ops.setter
    def batch_nvfs_submit_ops(self, val):
        if self._readonly:
            raise ValueError("This StatsLevel1 instance is read-only")
        cdef OpCounter val_ = val
        memcpy(<void *>&(self._ptr[0].batch_nvfs_submit_ops), <void *>(val_._get_ptr()), sizeof(CUfileOpCounter_t) * 1)

    @property
    def batch_p2p_submit_ops(self):
        """OpCounter: """
        return OpCounter.from_ptr(<intptr_t>&(self._ptr[0].batch_p2p_submit_ops), self._readonly, self)

    @batch_p2p_submit_ops.setter
    def batch_p2p_submit_ops(self, val):
        if self._readonly:
            raise ValueError("This StatsLevel1 instance is read-only")
        cdef OpCounter val_ = val
        memcpy(<void *>&(self._ptr[0].batch_p2p_submit_ops), <void *>(val_._get_ptr()), sizeof(CUfileOpCounter_t) * 1)

    @property
    def batch_aio_submit_ops(self):
        """OpCounter: """
        return OpCounter.from_ptr(<intptr_t>&(self._ptr[0].batch_aio_submit_ops), self._readonly, self)

    @batch_aio_submit_ops.setter
    def batch_aio_submit_ops(self, val):
        if self._readonly:
            raise ValueError("This StatsLevel1 instance is read-only")
        cdef OpCounter val_ = val
        memcpy(<void *>&(self._ptr[0].batch_aio_submit_ops), <void *>(val_._get_ptr()), sizeof(CUfileOpCounter_t) * 1)

    @property
    def batch_iouring_submit_ops(self):
        """OpCounter: """
        return OpCounter.from_ptr(<intptr_t>&(self._ptr[0].batch_iouring_submit_ops), self._readonly, self)

    @batch_iouring_submit_ops.setter
    def batch_iouring_submit_ops(self, val):
        if self._readonly:
            raise ValueError("This StatsLevel1 instance is read-only")
        cdef OpCounter val_ = val
        memcpy(<void *>&(self._ptr[0].batch_iouring_submit_ops), <void *>(val_._get_ptr()), sizeof(CUfileOpCounter_t) * 1)

    @property
    def batch_mixed_io_submit_ops(self):
        """OpCounter: """
        return OpCounter.from_ptr(<intptr_t>&(self._ptr[0].batch_mixed_io_submit_ops), self._readonly, self)

    @batch_mixed_io_submit_ops.setter
    def batch_mixed_io_submit_ops(self, val):
        if self._readonly:
            raise ValueError("This StatsLevel1 instance is read-only")
        cdef OpCounter val_ = val
        memcpy(<void *>&(self._ptr[0].batch_mixed_io_submit_ops), <void *>(val_._get_ptr()), sizeof(CUfileOpCounter_t) * 1)

    @property
    def batch_total_submit_ops(self):
        """OpCounter: """
        return OpCounter.from_ptr(<intptr_t>&(self._ptr[0].batch_total_submit_ops), self._readonly, self)

    @batch_total_submit_ops.setter
    def batch_total_submit_ops(self, val):
        if self._readonly:
            raise ValueError("This StatsLevel1 instance is read-only")
        cdef OpCounter val_ = val
        memcpy(<void *>&(self._ptr[0].batch_total_submit_ops), <void *>(val_._get_ptr()), sizeof(CUfileOpCounter_t) * 1)

    @property
    def read_bytes(self):
        """int: """
        return self._ptr[0].read_bytes

    @read_bytes.setter
    def read_bytes(self, val):
        if self._readonly:
            raise ValueError("This StatsLevel1 instance is read-only")
        self._ptr[0].read_bytes = val

    @property
    def write_bytes(self):
        """int: """
        return self._ptr[0].write_bytes

    @write_bytes.setter
    def write_bytes(self, val):
        if self._readonly:
            raise ValueError("This StatsLevel1 instance is read-only")
        self._ptr[0].write_bytes = val

    @property
    def read_bw_bytes_per_sec(self):
        """int: """
        return self._ptr[0].read_bw_bytes_per_sec

    @read_bw_bytes_per_sec.setter
    def read_bw_bytes_per_sec(self, val):
        if self._readonly:
            raise ValueError("This StatsLevel1 instance is read-only")
        self._ptr[0].read_bw_bytes_per_sec = val

    @property
    def write_bw_bytes_per_sec(self):
        """int: """
        return self._ptr[0].write_bw_bytes_per_sec

    @write_bw_bytes_per_sec.setter
    def write_bw_bytes_per_sec(self, val):
        if self._readonly:
            raise ValueError("This StatsLevel1 instance is read-only")
        self._ptr[0].write_bw_bytes_per_sec = val

    @property
    def read_lat_avg_us(self):
        """int: """
        return self._ptr[0].read_lat_avg_us

    @read_lat_avg_us.setter
    def read_lat_avg_us(self, val):
        if self._readonly:
            raise ValueError("This StatsLevel1 instance is read-only")
        self._ptr[0].read_lat_avg_us = val

    @property
    def write_lat_avg_us(self):
        """int: """
        return self._ptr[0].write_lat_avg_us

    @write_lat_avg_us.setter
    def write_lat_avg_us(self, val):
        if self._readonly:
            raise ValueError("This StatsLevel1 instance is read-only")
        self._ptr[0].write_lat_avg_us = val

    @property
    def read_ops_per_sec(self):
        """int: """
        return self._ptr[0].read_ops_per_sec

    @read_ops_per_sec.setter
    def read_ops_per_sec(self, val):
        if self._readonly:
            raise ValueError("This StatsLevel1 instance is read-only")
        self._ptr[0].read_ops_per_sec = val

    @property
    def write_ops_per_sec(self):
        """int: """
        return self._ptr[0].write_ops_per_sec

    @write_ops_per_sec.setter
    def write_ops_per_sec(self, val):
        if self._readonly:
            raise ValueError("This StatsLevel1 instance is read-only")
        self._ptr[0].write_ops_per_sec = val

    @property
    def read_lat_sum_us(self):
        """int: """
        return self._ptr[0].read_lat_sum_us

    @read_lat_sum_us.setter
    def read_lat_sum_us(self, val):
        if self._readonly:
            raise ValueError("This StatsLevel1 instance is read-only")
        self._ptr[0].read_lat_sum_us = val

    @property
    def write_lat_sum_us(self):
        """int: """
        return self._ptr[0].write_lat_sum_us

    @write_lat_sum_us.setter
    def write_lat_sum_us(self, val):
        if self._readonly:
            raise ValueError("This StatsLevel1 instance is read-only")
        self._ptr[0].write_lat_sum_us = val

    @property
    def batch_read_bytes(self):
        """int: """
        return self._ptr[0].batch_read_bytes

    @batch_read_bytes.setter
    def batch_read_bytes(self, val):
        if self._readonly:
            raise ValueError("This StatsLevel1 instance is read-only")
        self._ptr[0].batch_read_bytes = val

    @property
    def batch_write_bytes(self):
        """int: """
        return self._ptr[0].batch_write_bytes

    @batch_write_bytes.setter
    def batch_write_bytes(self, val):
        if self._readonly:
            raise ValueError("This StatsLevel1 instance is read-only")
        self._ptr[0].batch_write_bytes = val

    @property
    def batch_read_bw_bytes(self):
        """int: """
        return self._ptr[0].batch_read_bw_bytes

    @batch_read_bw_bytes.setter
    def batch_read_bw_bytes(self, val):
        if self._readonly:
            raise ValueError("This StatsLevel1 instance is read-only")
        self._ptr[0].batch_read_bw_bytes = val

    @property
    def batch_write_bw_bytes(self):
        """int: """
        return self._ptr[0].batch_write_bw_bytes

    @batch_write_bw_bytes.setter
    def batch_write_bw_bytes(self, val):
        if self._readonly:
            raise ValueError("This StatsLevel1 instance is read-only")
        self._ptr[0].batch_write_bw_bytes = val

    @property
    def batch_submit_lat_avg_us(self):
        """int: """
        return self._ptr[0].batch_submit_lat_avg_us

    @batch_submit_lat_avg_us.setter
    def batch_submit_lat_avg_us(self, val):
        if self._readonly:
            raise ValueError("This StatsLevel1 instance is read-only")
        self._ptr[0].batch_submit_lat_avg_us = val

    @property
    def batch_completion_lat_avg_us(self):
        """int: """
        return self._ptr[0].batch_completion_lat_avg_us

    @batch_completion_lat_avg_us.setter
    def batch_completion_lat_avg_us(self, val):
        if self._readonly:
            raise ValueError("This StatsLevel1 instance is read-only")
        self._ptr[0].batch_completion_lat_avg_us = val

    @property
    def batch_submit_ops_per_sec(self):
        """int: """
        return self._ptr[0].batch_submit_ops_per_sec

    @batch_submit_ops_per_sec.setter
    def batch_submit_ops_per_sec(self, val):
        if self._readonly:
            raise ValueError("This StatsLevel1 instance is read-only")
        self._ptr[0].batch_submit_ops_per_sec = val

    @property
    def batch_complete_ops_per_sec(self):
        """int: """
        return self._ptr[0].batch_complete_ops_per_sec

    @batch_complete_ops_per_sec.setter
    def batch_complete_ops_per_sec(self, val):
        if self._readonly:
            raise ValueError("This StatsLevel1 instance is read-only")
        self._ptr[0].batch_complete_ops_per_sec = val

    @property
    def batch_submit_lat_sum_us(self):
        """int: """
        return self._ptr[0].batch_submit_lat_sum_us

    @batch_submit_lat_sum_us.setter
    def batch_submit_lat_sum_us(self, val):
        if self._readonly:
            raise ValueError("This StatsLevel1 instance is read-only")
        self._ptr[0].batch_submit_lat_sum_us = val

    @property
    def batch_completion_lat_sum_us(self):
        """int: """
        return self._ptr[0].batch_completion_lat_sum_us

    @batch_completion_lat_sum_us.setter
    def batch_completion_lat_sum_us(self, val):
        if self._readonly:
            raise ValueError("This StatsLevel1 instance is read-only")
        self._ptr[0].batch_completion_lat_sum_us = val

    @property
    def last_batch_read_bytes(self):
        """int: """
        return self._ptr[0].last_batch_read_bytes

    @last_batch_read_bytes.setter
    def last_batch_read_bytes(self, val):
        if self._readonly:
            raise ValueError("This StatsLevel1 instance is read-only")
        self._ptr[0].last_batch_read_bytes = val

    @property
    def last_batch_write_bytes(self):
        """int: """
        return self._ptr[0].last_batch_write_bytes

    @last_batch_write_bytes.setter
    def last_batch_write_bytes(self, val):
        if self._readonly:
            raise ValueError("This StatsLevel1 instance is read-only")
        self._ptr[0].last_batch_write_bytes = val

    @staticmethod
    def from_buffer(buffer):
        """Create an StatsLevel1 instance with the memory from the given buffer."""
        return __from_buffer(buffer, sizeof(CUfileStatsLevel1_t), StatsLevel1)

    @staticmethod
    def from_data(data):
        """Create an StatsLevel1 instance wrapping the given NumPy array.

        Args:
            data (_numpy.ndarray): a single-element array of dtype `stats_level1_dtype` holding the data.
        """
        return __from_data(data, "stats_level1_dtype", stats_level1_dtype, StatsLevel1)

    @staticmethod
    def from_ptr(intptr_t ptr, bint readonly=False, object owner=None):
        """Create an StatsLevel1 instance wrapping the given pointer.

        Args:
            ptr (intptr_t): pointer address as Python :class:`int` to the data.
            owner (object): The Python object that owns the pointer. If not provided, data will be copied.
            readonly (bool): whether the data is read-only (to the user). default is `False`.
        """
        if ptr == 0:
            raise ValueError("ptr must not be null (0)")
        cdef StatsLevel1 obj = StatsLevel1.__new__(StatsLevel1)
        if owner is None:
            obj._ptr = <CUfileStatsLevel1_t *>malloc(sizeof(CUfileStatsLevel1_t))
            if obj._ptr == NULL:
                raise MemoryError("Error allocating StatsLevel1")
            memcpy(<void*>(obj._ptr), <void*>ptr, sizeof(CUfileStatsLevel1_t))
            obj._owner = None
            obj._owned = True
        else:
            obj._ptr = <CUfileStatsLevel1_t *>ptr
            obj._owner = owner
            obj._owned = False
        obj._readonly = readonly
        return obj


cdef _get_io_params_dtype_offsets():
    cdef CUfileIOParams_t pod = CUfileIOParams_t()
    return _numpy.dtype({
        'names': ['mode', 'u', 'fh', 'opcode', 'cookie'],
        'formats': [_numpy.int32, _py_anon_pod2_dtype, _numpy.intp, _numpy.int32, _numpy.intp],
        'offsets': [
            (<intptr_t>&(pod.mode)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.u)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.fh)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.opcode)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.cookie)) - (<intptr_t>&pod),
        ],
        'itemsize': sizeof(CUfileIOParams_t),
    })

io_params_dtype = _get_io_params_dtype_offsets()

cdef class IOParams:
    """Empty-initialize an array of `CUfileIOParams_t`.

    The resulting object is of length `size` and of dtype `io_params_dtype`.
    If default-constructed, the instance represents a single struct.

    Args:
        size (int): number of structs, default=1.


    .. seealso:: `CUfileIOParams_t`
    """
    cdef:
        readonly object _data



    def __init__(self, size=1):
        arr = _numpy.empty(size, dtype=io_params_dtype)
        self._data = arr.view(_numpy.recarray)
        assert self._data.itemsize == sizeof(CUfileIOParams_t), \
            f"itemsize {self._data.itemsize} mismatches struct size { sizeof(CUfileIOParams_t) }"

    def __repr__(self):
        if self._data.size > 1:
            return f"<{__name__}.IOParams_Array_{self._data.size} object at {hex(id(self))}>"
        else:
            return f"<{__name__}.IOParams object at {hex(id(self))}>"

    @property
    def ptr(self):
        """Get the pointer address to the data as Python :class:`int`."""
        return self._data.ctypes.data

    cdef intptr_t _get_ptr(self):
        return self._data.ctypes.data

    def __int__(self):
        if self._data.size > 1:
            raise TypeError("int() argument must be a bytes-like object of size 1. "
                            "To get the pointer address of an array, use .ptr")
        return self._data.ctypes.data

    def __len__(self):
        return self._data.size

    def __eq__(self, other):
        cdef object self_data = self._data
        if (not isinstance(other, IOParams)) or self_data.size != other._data.size or self_data.dtype != other._data.dtype:
            return False
        return bool((self_data == other._data).all())

    def __getbuffer__(self, Py_buffer *buffer, int flags):
        cpython.PyObject_GetBuffer(self._data, buffer, flags)

    def __releasebuffer__(self, Py_buffer *buffer):
        cpython.PyBuffer_Release(buffer)

    @property
    def mode(self):
        """Union[~_numpy.int32, int]: """
        if self._data.size == 1:
            return int(self._data.mode[0])
        return self._data.mode

    @mode.setter
    def mode(self, val):
        self._data.mode = val

    @property
    def u(self):
        """_py_anon_pod2_dtype: """
        return self._data.u

    @u.setter
    def u(self, val):
        self._data.u = val

    @property
    def fh(self):
        """Union[~_numpy.intp, int]: """
        if self._data.size == 1:
            return int(self._data.fh[0])
        return self._data.fh

    @fh.setter
    def fh(self, val):
        self._data.fh = val

    @property
    def opcode(self):
        """Union[~_numpy.int32, int]: """
        if self._data.size == 1:
            return int(self._data.opcode[0])
        return self._data.opcode

    @opcode.setter
    def opcode(self, val):
        self._data.opcode = val

    @property
    def cookie(self):
        """Union[~_numpy.intp, int]: """
        if self._data.size == 1:
            return int(self._data.cookie[0])
        return self._data.cookie

    @cookie.setter
    def cookie(self, val):
        self._data.cookie = val

    def __getitem__(self, key):
        cdef ssize_t key_
        cdef ssize_t size
        if isinstance(key, int):
            key_ = key
            size = self._data.size
            if key_ >= size or key_ <= -(size+1):
                raise IndexError("index is out of bounds")
            if key_ < 0:
                key_ += size
            return IOParams.from_data(self._data[key_:key_+1])
        out = self._data[key]
        if isinstance(out, _numpy.recarray) and out.dtype == io_params_dtype:
            return IOParams.from_data(out)
        return out

    def __setitem__(self, key, val):
        self._data[key] = val

    @staticmethod
    def from_buffer(buffer):
        """Create an IOParams instance with the memory from the given buffer."""
        return IOParams.from_data(_numpy.frombuffer(buffer, dtype=io_params_dtype))

    @staticmethod
    def from_data(data):
        """Create an IOParams instance wrapping the given NumPy array.

        Args:
            data (_numpy.ndarray): a 1D array of dtype `io_params_dtype` holding the data.
        """
        cdef IOParams obj = IOParams.__new__(IOParams)
        if not isinstance(data, _numpy.ndarray):
            raise TypeError("data argument must be a NumPy ndarray")
        if data.ndim != 1:
            raise ValueError("data array must be 1D")
        if data.dtype != io_params_dtype:
            raise ValueError("data array must be of dtype io_params_dtype")
        obj._data = data.view(_numpy.recarray)

        return obj

    @staticmethod
    def from_ptr(intptr_t ptr, size_t size=1, bint readonly=False):
        """Create an IOParams instance wrapping the given pointer.

        Args:
            ptr (intptr_t): pointer address as Python :class:`int` to the data.
            size (int): number of structs, default=1.
            readonly (bool): whether the data is read-only (to the user). default is `False`.
        """
        if ptr == 0:
            raise ValueError("ptr must not be null (0)")
        cdef IOParams obj = IOParams.__new__(IOParams)
        cdef flag = cpython.buffer.PyBUF_READ if readonly else cpython.buffer.PyBUF_WRITE
        cdef object buf = cpython.memoryview.PyMemoryView_FromMemory(
            <char*>ptr, sizeof(CUfileIOParams_t) * size, flag)
        data = _numpy.ndarray(size, buffer=buf, dtype=io_params_dtype)
        obj._data = data.view(_numpy.recarray)

        return obj


cdef _get_stats_level2_dtype_offsets():
    cdef CUfileStatsLevel2_t pod = CUfileStatsLevel2_t()
    return _numpy.dtype({
        'names': ['basic', 'read_size_kb_hist', 'write_size_kb_hist'],
        'formats': [stats_level1_dtype, (_numpy.uint64, 32), (_numpy.uint64, 32)],
        'offsets': [
            (<intptr_t>&(pod.basic)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.read_size_kb_hist)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.write_size_kb_hist)) - (<intptr_t>&pod),
        ],
        'itemsize': sizeof(CUfileStatsLevel2_t),
    })

stats_level2_dtype = _get_stats_level2_dtype_offsets()

cdef class StatsLevel2:
    """Empty-initialize an instance of `CUfileStatsLevel2_t`.


    .. seealso:: `CUfileStatsLevel2_t`
    """
    cdef:
        CUfileStatsLevel2_t *_ptr
        object _owner
        bint _owned
        bint _readonly

    def __init__(self):
        self._ptr = <CUfileStatsLevel2_t *>calloc(1, sizeof(CUfileStatsLevel2_t))
        if self._ptr == NULL:
            raise MemoryError("Error allocating StatsLevel2")
        self._owner = None
        self._owned = True
        self._readonly = False

    def __dealloc__(self):
        cdef CUfileStatsLevel2_t *ptr
        if self._owned and self._ptr != NULL:
            ptr = self._ptr
            self._ptr = NULL
            free(ptr)

    def __repr__(self):
        return f"<{__name__}.StatsLevel2 object at {hex(id(self))}>"

    @property
    def ptr(self):
        """Get the pointer address to the data as Python :class:`int`."""
        return <intptr_t>(self._ptr)

    cdef intptr_t _get_ptr(self):
        return <intptr_t>(self._ptr)

    def __int__(self):
        return <intptr_t>(self._ptr)

    def __eq__(self, other):
        cdef StatsLevel2 other_
        if not isinstance(other, StatsLevel2):
            return False
        other_ = other
        return (memcmp(<void *><intptr_t>(self._ptr), <void *><intptr_t>(other_._ptr), sizeof(CUfileStatsLevel2_t)) == 0)

    def __getbuffer__(self, Py_buffer *buffer, int flags):
        __getbuffer(self, buffer, <void *>self._ptr, sizeof(CUfileStatsLevel2_t), self._readonly)

    def __releasebuffer__(self, Py_buffer *buffer):
        pass

    def __setitem__(self, key, val):
        if key == 0 and isinstance(val, _numpy.ndarray):
            self._ptr = <CUfileStatsLevel2_t *>malloc(sizeof(CUfileStatsLevel2_t))
            if self._ptr == NULL:
                raise MemoryError("Error allocating StatsLevel2")
            memcpy(<void*>self._ptr, <void*><intptr_t>val.ctypes.data, sizeof(CUfileStatsLevel2_t))
            self._owner = None
            self._owned = True
            self._readonly = not val.flags.writeable
        else:
            setattr(self, key, val)

    @property
    def basic(self):
        """StatsLevel1: """
        return StatsLevel1.from_ptr(<intptr_t>&(self._ptr[0].basic), self._readonly, self)

    @basic.setter
    def basic(self, val):
        if self._readonly:
            raise ValueError("This StatsLevel2 instance is read-only")
        cdef StatsLevel1 val_ = val
        memcpy(<void *>&(self._ptr[0].basic), <void *>(val_._get_ptr()), sizeof(CUfileStatsLevel1_t) * 1)

    @property
    def read_size_kb_hist(self):
        """~_numpy.uint64: (array of length 32)."""
        cdef view.array arr = view.array(shape=(32,), itemsize=sizeof(uint64_t), format="Q", mode="c", allocate_buffer=False)
        arr.data = <char *>(&(self._ptr[0].read_size_kb_hist))
        return _numpy.asarray(arr)

    @read_size_kb_hist.setter
    def read_size_kb_hist(self, val):
        if self._readonly:
            raise ValueError("This StatsLevel2 instance is read-only")
        if len(val) != 32:
            raise ValueError(f"Expected length { 32 } for field read_size_kb_hist, got {len(val)}")
        cdef view.array arr = view.array(shape=(32,), itemsize=sizeof(uint64_t), format="Q", mode="c")
        arr[:] = _numpy.asarray(val, dtype=_numpy.uint64)
        memcpy(<void *>(&(self._ptr[0].read_size_kb_hist)), <void *>(arr.data), sizeof(uint64_t) * len(val))

    @property
    def write_size_kb_hist(self):
        """~_numpy.uint64: (array of length 32)."""
        cdef view.array arr = view.array(shape=(32,), itemsize=sizeof(uint64_t), format="Q", mode="c", allocate_buffer=False)
        arr.data = <char *>(&(self._ptr[0].write_size_kb_hist))
        return _numpy.asarray(arr)

    @write_size_kb_hist.setter
    def write_size_kb_hist(self, val):
        if self._readonly:
            raise ValueError("This StatsLevel2 instance is read-only")
        if len(val) != 32:
            raise ValueError(f"Expected length { 32 } for field write_size_kb_hist, got {len(val)}")
        cdef view.array arr = view.array(shape=(32,), itemsize=sizeof(uint64_t), format="Q", mode="c")
        arr[:] = _numpy.asarray(val, dtype=_numpy.uint64)
        memcpy(<void *>(&(self._ptr[0].write_size_kb_hist)), <void *>(arr.data), sizeof(uint64_t) * len(val))

    @staticmethod
    def from_buffer(buffer):
        """Create an StatsLevel2 instance with the memory from the given buffer."""
        return __from_buffer(buffer, sizeof(CUfileStatsLevel2_t), StatsLevel2)

    @staticmethod
    def from_data(data):
        """Create an StatsLevel2 instance wrapping the given NumPy array.

        Args:
            data (_numpy.ndarray): a single-element array of dtype `stats_level2_dtype` holding the data.
        """
        return __from_data(data, "stats_level2_dtype", stats_level2_dtype, StatsLevel2)

    @staticmethod
    def from_ptr(intptr_t ptr, bint readonly=False, object owner=None):
        """Create an StatsLevel2 instance wrapping the given pointer.

        Args:
            ptr (intptr_t): pointer address as Python :class:`int` to the data.
            owner (object): The Python object that owns the pointer. If not provided, data will be copied.
            readonly (bool): whether the data is read-only (to the user). default is `False`.
        """
        if ptr == 0:
            raise ValueError("ptr must not be null (0)")
        cdef StatsLevel2 obj = StatsLevel2.__new__(StatsLevel2)
        if owner is None:
            obj._ptr = <CUfileStatsLevel2_t *>malloc(sizeof(CUfileStatsLevel2_t))
            if obj._ptr == NULL:
                raise MemoryError("Error allocating StatsLevel2")
            memcpy(<void*>(obj._ptr), <void*>ptr, sizeof(CUfileStatsLevel2_t))
            obj._owner = None
            obj._owned = True
        else:
            obj._ptr = <CUfileStatsLevel2_t *>ptr
            obj._owner = owner
            obj._owned = False
        obj._readonly = readonly
        return obj


cdef _get_stats_level3_dtype_offsets():
    cdef CUfileStatsLevel3_t pod = CUfileStatsLevel3_t()
    return _numpy.dtype({
        'names': ['detailed', 'num_gpus', 'per_gpu_stats'],
        'formats': [stats_level2_dtype, _numpy.uint32, (per_gpu_stats_dtype, 16)],
        'offsets': [
            (<intptr_t>&(pod.detailed)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.num_gpus)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.per_gpu_stats)) - (<intptr_t>&pod),
        ],
        'itemsize': sizeof(CUfileStatsLevel3_t),
    })

stats_level3_dtype = _get_stats_level3_dtype_offsets()

cdef class StatsLevel3:
    """Empty-initialize an instance of `CUfileStatsLevel3_t`.


    .. seealso:: `CUfileStatsLevel3_t`
    """
    cdef:
        CUfileStatsLevel3_t *_ptr
        object _owner
        bint _owned
        bint _readonly

    def __init__(self):
        self._ptr = <CUfileStatsLevel3_t *>calloc(1, sizeof(CUfileStatsLevel3_t))
        if self._ptr == NULL:
            raise MemoryError("Error allocating StatsLevel3")
        self._owner = None
        self._owned = True
        self._readonly = False

    def __dealloc__(self):
        cdef CUfileStatsLevel3_t *ptr
        if self._owned and self._ptr != NULL:
            ptr = self._ptr
            self._ptr = NULL
            free(ptr)

    def __repr__(self):
        return f"<{__name__}.StatsLevel3 object at {hex(id(self))}>"

    @property
    def ptr(self):
        """Get the pointer address to the data as Python :class:`int`."""
        return <intptr_t>(self._ptr)

    cdef intptr_t _get_ptr(self):
        return <intptr_t>(self._ptr)

    def __int__(self):
        return <intptr_t>(self._ptr)

    def __eq__(self, other):
        cdef StatsLevel3 other_
        if not isinstance(other, StatsLevel3):
            return False
        other_ = other
        return (memcmp(<void *><intptr_t>(self._ptr), <void *><intptr_t>(other_._ptr), sizeof(CUfileStatsLevel3_t)) == 0)

    def __getbuffer__(self, Py_buffer *buffer, int flags):
        __getbuffer(self, buffer, <void *>self._ptr, sizeof(CUfileStatsLevel3_t), self._readonly)

    def __releasebuffer__(self, Py_buffer *buffer):
        pass

    def __setitem__(self, key, val):
        if key == 0 and isinstance(val, _numpy.ndarray):
            self._ptr = <CUfileStatsLevel3_t *>malloc(sizeof(CUfileStatsLevel3_t))
            if self._ptr == NULL:
                raise MemoryError("Error allocating StatsLevel3")
            memcpy(<void*>self._ptr, <void*><intptr_t>val.ctypes.data, sizeof(CUfileStatsLevel3_t))
            self._owner = None
            self._owned = True
            self._readonly = not val.flags.writeable
        else:
            setattr(self, key, val)

    @property
    def detailed(self):
        """StatsLevel2: """
        return StatsLevel2.from_ptr(<intptr_t>&(self._ptr[0].detailed), self._readonly, self)

    @detailed.setter
    def detailed(self, val):
        if self._readonly:
            raise ValueError("This StatsLevel3 instance is read-only")
        cdef StatsLevel2 val_ = val
        memcpy(<void *>&(self._ptr[0].detailed), <void *>(val_._get_ptr()), sizeof(CUfileStatsLevel2_t) * 1)

    @property
    def per_gpu_stats(self):
        """PerGpuStats: """
        return PerGpuStats.from_ptr(<intptr_t>&(self._ptr[0].per_gpu_stats), 16, self._readonly)

    @per_gpu_stats.setter
    def per_gpu_stats(self, val):
        if self._readonly:
            raise ValueError("This StatsLevel3 instance is read-only")
        cdef PerGpuStats val_ = val
        if len(val) != 16:
            raise ValueError(f"Expected length { 16 } for field per_gpu_stats, got {len(val)}")
        memcpy(<void *>&(self._ptr[0].per_gpu_stats), <void *>(val_._get_ptr()), sizeof(CUfilePerGpuStats_t) * 16)

    @property
    def num_gpus(self):
        """int: """
        return self._ptr[0].num_gpus

    @num_gpus.setter
    def num_gpus(self, val):
        if self._readonly:
            raise ValueError("This StatsLevel3 instance is read-only")
        self._ptr[0].num_gpus = val

    @staticmethod
    def from_buffer(buffer):
        """Create an StatsLevel3 instance with the memory from the given buffer."""
        return __from_buffer(buffer, sizeof(CUfileStatsLevel3_t), StatsLevel3)

    @staticmethod
    def from_data(data):
        """Create an StatsLevel3 instance wrapping the given NumPy array.

        Args:
            data (_numpy.ndarray): a single-element array of dtype `stats_level3_dtype` holding the data.
        """
        return __from_data(data, "stats_level3_dtype", stats_level3_dtype, StatsLevel3)

    @staticmethod
    def from_ptr(intptr_t ptr, bint readonly=False, object owner=None):
        """Create an StatsLevel3 instance wrapping the given pointer.

        Args:
            ptr (intptr_t): pointer address as Python :class:`int` to the data.
            owner (object): The Python object that owns the pointer. If not provided, data will be copied.
            readonly (bool): whether the data is read-only (to the user). default is `False`.
        """
        if ptr == 0:
            raise ValueError("ptr must not be null (0)")
        cdef StatsLevel3 obj = StatsLevel3.__new__(StatsLevel3)
        if owner is None:
            obj._ptr = <CUfileStatsLevel3_t *>malloc(sizeof(CUfileStatsLevel3_t))
            if obj._ptr == NULL:
                raise MemoryError("Error allocating StatsLevel3")
            memcpy(<void*>(obj._ptr), <void*>ptr, sizeof(CUfileStatsLevel3_t))
            obj._owner = None
            obj._owned = True
        else:
            obj._ptr = <CUfileStatsLevel3_t *>ptr
            obj._owner = owner
            obj._owned = False
        obj._readonly = readonly
        return obj



###############################################################################
# Enum
###############################################################################

class OpError(_FastEnum):
    """
    See `CUfileOpError`.
    """
    SUCCESS = CU_FILE_SUCCESS
    DRIVER_NOT_INITIALIZED = CU_FILE_DRIVER_NOT_INITIALIZED
    DRIVER_INVALID_PROPS = CU_FILE_DRIVER_INVALID_PROPS
    DRIVER_UNSUPPORTED_LIMIT = CU_FILE_DRIVER_UNSUPPORTED_LIMIT
    DRIVER_VERSION_MISMATCH = CU_FILE_DRIVER_VERSION_MISMATCH
    DRIVER_VERSION_READ_ERROR = CU_FILE_DRIVER_VERSION_READ_ERROR
    DRIVER_CLOSING = CU_FILE_DRIVER_CLOSING
    PLATFORM_NOT_SUPPORTED = CU_FILE_PLATFORM_NOT_SUPPORTED
    IO_NOT_SUPPORTED = CU_FILE_IO_NOT_SUPPORTED
    DEVICE_NOT_SUPPORTED = CU_FILE_DEVICE_NOT_SUPPORTED
    NVFS_DRIVER_ERROR = CU_FILE_NVFS_DRIVER_ERROR
    CUDA_DRIVER_ERROR = CU_FILE_CUDA_DRIVER_ERROR
    CUDA_POINTER_INVALID = CU_FILE_CUDA_POINTER_INVALID
    CUDA_MEMORY_TYPE_INVALID = CU_FILE_CUDA_MEMORY_TYPE_INVALID
    CUDA_POINTER_RANGE_ERROR = CU_FILE_CUDA_POINTER_RANGE_ERROR
    CUDA_CONTEXT_MISMATCH = CU_FILE_CUDA_CONTEXT_MISMATCH
    INVALID_MAPPING_SIZE = CU_FILE_INVALID_MAPPING_SIZE
    INVALID_MAPPING_RANGE = CU_FILE_INVALID_MAPPING_RANGE
    INVALID_FILE_TYPE = CU_FILE_INVALID_FILE_TYPE
    INVALID_FILE_OPEN_FLAG = CU_FILE_INVALID_FILE_OPEN_FLAG
    DIO_NOT_SET = CU_FILE_DIO_NOT_SET
    INVALID_VALUE = CU_FILE_INVALID_VALUE
    MEMORY_ALREADY_REGISTERED = CU_FILE_MEMORY_ALREADY_REGISTERED
    MEMORY_NOT_REGISTERED = CU_FILE_MEMORY_NOT_REGISTERED
    PERMISSION_DENIED = CU_FILE_PERMISSION_DENIED
    DRIVER_ALREADY_OPEN = CU_FILE_DRIVER_ALREADY_OPEN
    HANDLE_NOT_REGISTERED = CU_FILE_HANDLE_NOT_REGISTERED
    HANDLE_ALREADY_REGISTERED = CU_FILE_HANDLE_ALREADY_REGISTERED
    DEVICE_NOT_FOUND = CU_FILE_DEVICE_NOT_FOUND
    INTERNAL_ERROR = CU_FILE_INTERNAL_ERROR
    GETNEWFD_FAILED = CU_FILE_GETNEWFD_FAILED
    NVFS_SETUP_ERROR = CU_FILE_NVFS_SETUP_ERROR
    IO_DISABLED = CU_FILE_IO_DISABLED
    BATCH_SUBMIT_FAILED = CU_FILE_BATCH_SUBMIT_FAILED
    GPU_MEMORY_PINNING_FAILED = CU_FILE_GPU_MEMORY_PINNING_FAILED
    BATCH_FULL = CU_FILE_BATCH_FULL
    ASYNC_NOT_SUPPORTED = CU_FILE_ASYNC_NOT_SUPPORTED
    INTERNAL_BATCH_SETUP_ERROR = CU_FILE_INTERNAL_BATCH_SETUP_ERROR
    INTERNAL_BATCH_SUBMIT_ERROR = CU_FILE_INTERNAL_BATCH_SUBMIT_ERROR
    INTERNAL_BATCH_GETSTATUS_ERROR = CU_FILE_INTERNAL_BATCH_GETSTATUS_ERROR
    INTERNAL_BATCH_CANCEL_ERROR = CU_FILE_INTERNAL_BATCH_CANCEL_ERROR
    NOMEM_ERROR = CU_FILE_NOMEM_ERROR
    IO_ERROR = CU_FILE_IO_ERROR
    INTERNAL_BUF_REGISTER_ERROR = CU_FILE_INTERNAL_BUF_REGISTER_ERROR
    HASH_OPR_ERROR = CU_FILE_HASH_OPR_ERROR
    INVALID_CONTEXT_ERROR = CU_FILE_INVALID_CONTEXT_ERROR
    NVFS_INTERNAL_DRIVER_ERROR = CU_FILE_NVFS_INTERNAL_DRIVER_ERROR
    BATCH_NOCOMPAT_ERROR = CU_FILE_BATCH_NOCOMPAT_ERROR
    IO_MAX_ERROR = CU_FILE_IO_MAX_ERROR

class DriverStatusFlags(_FastEnum):
    """
    See `CUfileDriverStatusFlags_t`.
    """
    LUSTRE_SUPPORTED = (CU_FILE_LUSTRE_SUPPORTED, 'Support for DDN LUSTRE')
    WEKAFS_SUPPORTED = (CU_FILE_WEKAFS_SUPPORTED, 'Support for WEKAFS')
    NFS_SUPPORTED = (CU_FILE_NFS_SUPPORTED, 'Support for NFS')
    GPFS_SUPPORTED = CU_FILE_GPFS_SUPPORTED
    NVME_SUPPORTED = (CU_FILE_NVME_SUPPORTED, '< Support for GPFS Support for NVMe')
    NVMEOF_SUPPORTED = (CU_FILE_NVMEOF_SUPPORTED, 'Support for NVMeOF')
    SCSI_SUPPORTED = (CU_FILE_SCSI_SUPPORTED, 'Support for SCSI')
    SCALEFLUX_CSD_SUPPORTED = (CU_FILE_SCALEFLUX_CSD_SUPPORTED, 'Support for Scaleflux CSD')
    NVMESH_SUPPORTED = (CU_FILE_NVMESH_SUPPORTED, 'Support for NVMesh Block Dev')
    BEEGFS_SUPPORTED = (CU_FILE_BEEGFS_SUPPORTED, 'Support for BeeGFS')
    NVME_P2P_SUPPORTED = (CU_FILE_NVME_P2P_SUPPORTED, 'Do not use this macro. This is deprecated now')
    SCATEFS_SUPPORTED = (CU_FILE_SCATEFS_SUPPORTED, 'Support for ScateFS')
    VIRTIOFS_SUPPORTED = (CU_FILE_VIRTIOFS_SUPPORTED, 'Support for VirtioFS')
    MAX_TARGET_TYPES = (CU_FILE_MAX_TARGET_TYPES, 'Maximum FS supported')

class DriverControlFlags(_FastEnum):
    """
    See `CUfileDriverControlFlags_t`.
    """
    USE_POLL_MODE = (CU_FILE_USE_POLL_MODE, 'use POLL mode. properties.use_poll_mode')
    ALLOW_COMPAT_MODE = (CU_FILE_ALLOW_COMPAT_MODE, 'allow COMPATIBILITY mode. properties.allow_compat_mode')

class FeatureFlags(_FastEnum):
    """
    See `CUfileFeatureFlags_t`.
    """
    DYN_ROUTING_SUPPORTED = (CU_FILE_DYN_ROUTING_SUPPORTED, 'Support for Dynamic routing to handle devices across the PCIe bridges')
    BATCH_IO_SUPPORTED = (CU_FILE_BATCH_IO_SUPPORTED, 'Supported')
    STREAMS_SUPPORTED = (CU_FILE_STREAMS_SUPPORTED, 'Supported')
    PARALLEL_IO_SUPPORTED = (CU_FILE_PARALLEL_IO_SUPPORTED, 'Supported')
    P2P_SUPPORTED = (CU_FILE_P2P_SUPPORTED, 'Support for PCI P2PDMA')

class FileHandleType(_FastEnum):
    """
    See `CUfileFileHandleType`.
    """
    OPAQUE_FD = (CU_FILE_HANDLE_TYPE_OPAQUE_FD, 'Linux based fd')
    OPAQUE_WIN32 = (CU_FILE_HANDLE_TYPE_OPAQUE_WIN32, 'Windows based handle (unsupported)')
    USERSPACE_FS = CU_FILE_HANDLE_TYPE_USERSPACE_FS

class Opcode(_FastEnum):
    """
    See `CUfileOpcode_t`.
    """
    READ = CUFILE_READ
    WRITE = CUFILE_WRITE

class Status(_FastEnum):
    """
    See `CUfileStatus_t`.
    """
    WAITING = CUFILE_WAITING
    PENDING = CUFILE_PENDING
    INVALID = CUFILE_INVALID
    CANCELED = CUFILE_CANCELED
    COMPLETE = CUFILE_COMPLETE
    TIMEOUT = CUFILE_TIMEOUT
    FAILED = CUFILE_FAILED

class BatchMode(_FastEnum):
    """
    See `CUfileBatchMode_t`.
    """
    BATCH = CUFILE_BATCH

class SizeTConfigParameter(_FastEnum):
    """
    See `CUFileSizeTConfigParameter_t`.
    """
    PROFILE_STATS = CUFILE_PARAM_PROFILE_STATS
    EXECUTION_MAX_IO_QUEUE_DEPTH = CUFILE_PARAM_EXECUTION_MAX_IO_QUEUE_DEPTH
    EXECUTION_MAX_IO_THREADS = CUFILE_PARAM_EXECUTION_MAX_IO_THREADS
    EXECUTION_MIN_IO_THRESHOLD_SIZE_KB = CUFILE_PARAM_EXECUTION_MIN_IO_THRESHOLD_SIZE_KB
    EXECUTION_MAX_REQUEST_PARALLELISM = CUFILE_PARAM_EXECUTION_MAX_REQUEST_PARALLELISM
    PROPERTIES_MAX_DIRECT_IO_SIZE_KB = CUFILE_PARAM_PROPERTIES_MAX_DIRECT_IO_SIZE_KB
    PROPERTIES_MAX_DEVICE_CACHE_SIZE_KB = CUFILE_PARAM_PROPERTIES_MAX_DEVICE_CACHE_SIZE_KB
    PROPERTIES_PER_BUFFER_CACHE_SIZE_KB = CUFILE_PARAM_PROPERTIES_PER_BUFFER_CACHE_SIZE_KB
    PROPERTIES_MAX_DEVICE_PINNED_MEM_SIZE_KB = CUFILE_PARAM_PROPERTIES_MAX_DEVICE_PINNED_MEM_SIZE_KB
    PROPERTIES_IO_BATCHSIZE = CUFILE_PARAM_PROPERTIES_IO_BATCHSIZE
    POLLTHRESHOLD_SIZE_KB = CUFILE_PARAM_POLLTHRESHOLD_SIZE_KB
    PROPERTIES_BATCH_IO_TIMEOUT_MS = CUFILE_PARAM_PROPERTIES_BATCH_IO_TIMEOUT_MS

class BoolConfigParameter(_FastEnum):
    """
    See `CUFileBoolConfigParameter_t`.
    """
    PROPERTIES_USE_POLL_MODE = CUFILE_PARAM_PROPERTIES_USE_POLL_MODE
    PROPERTIES_ALLOW_COMPAT_MODE = CUFILE_PARAM_PROPERTIES_ALLOW_COMPAT_MODE
    FORCE_COMPAT_MODE = CUFILE_PARAM_FORCE_COMPAT_MODE
    FS_MISC_API_CHECK_AGGRESSIVE = CUFILE_PARAM_FS_MISC_API_CHECK_AGGRESSIVE
    EXECUTION_PARALLEL_IO = CUFILE_PARAM_EXECUTION_PARALLEL_IO
    PROFILE_NVTX = CUFILE_PARAM_PROFILE_NVTX
    PROPERTIES_ALLOW_SYSTEM_MEMORY = CUFILE_PARAM_PROPERTIES_ALLOW_SYSTEM_MEMORY
    USE_PCIP2PDMA = CUFILE_PARAM_USE_PCIP2PDMA
    PREFER_IO_URING = CUFILE_PARAM_PREFER_IO_URING
    FORCE_ODIRECT_MODE = CUFILE_PARAM_FORCE_ODIRECT_MODE
    SKIP_TOPOLOGY_DETECTION = CUFILE_PARAM_SKIP_TOPOLOGY_DETECTION
    STREAM_MEMOPS_BYPASS = CUFILE_PARAM_STREAM_MEMOPS_BYPASS

class StringConfigParameter(_FastEnum):
    """
    See `CUFileStringConfigParameter_t`.
    """
    LOGGING_LEVEL = CUFILE_PARAM_LOGGING_LEVEL
    ENV_LOGFILE_PATH = CUFILE_PARAM_ENV_LOGFILE_PATH
    LOG_DIR = CUFILE_PARAM_LOG_DIR

class ArrayConfigParameter(_FastEnum):
    """
    See `CUFileArrayConfigParameter_t`.
    """
    POSIX_POOL_SLAB_SIZE_KB = CUFILE_PARAM_POSIX_POOL_SLAB_SIZE_KB
    POSIX_POOL_SLAB_COUNT = CUFILE_PARAM_POSIX_POOL_SLAB_COUNT

class P2PFlags(_FastEnum):
    """
    See `CUfileP2PFlags_t`.
    """
    P2PDMA = (CUFILE_P2PDMA, 'Support for PCI P2PDMA')
    NVFS = (CUFILE_NVFS, 'Support for nvidia-fs')
    DMABUF = (CUFILE_DMABUF, 'Support for DMA Buffer')
    C2C = (CUFILE_C2C, 'Support for Chip-to-Chip (Grace-based systems)')
    NVIDIA_PEERMEM = (CUFILE_NVIDIA_PEERMEM, 'Only for IBM Spectrum Scale and WekaFS')


###############################################################################
# Error handling
###############################################################################

ctypedef fused ReturnT:
    CUfileError_t
    ssize_t


class cuFileError(Exception):

    def __init__(self, status, cu_err=None):
        self.status = status
        self.cuda_error = cu_err
        s = OpError(status)
        cdef str err = f"{s.name} ({s.value}): {op_status_error(status)}"
        if cu_err is not None:
            e = pyCUresult(cu_err)
            err += f"; CUDA status: {e.name} ({e.value})"
        super(cuFileError, self).__init__(err)

    def __reduce__(self):
        return (type(self), (self.status, self.cuda_error))


@cython.profile(False)
cdef int check_status(ReturnT status) except 1 nogil:
    if ReturnT is CUfileError_t:
        if status.err != 0 or status.cu_err != 0:
            with gil:
                raise cuFileError(status.err, status.cu_err)
    elif ReturnT is ssize_t:
        if status == -1:
            # note: this assumes cuFile already properly resets errno in each API
            with gil:
                raise cuFileError(errno.errno)
    return 0


###############################################################################
# Wrapper functions
###############################################################################

cpdef intptr_t handle_register(intptr_t descr) except? 0:
    """cuFileHandleRegister is required, and performs extra checking that is memoized to provide increased performance on later cuFile operations.

    Args:
        descr (intptr_t): ``CUfileDescr_t`` file descriptor (OS agnostic).

    Returns:
        intptr_t: ``CUfileHandle_t`` opaque file handle for IO operations.

    .. seealso:: `cuFileHandleRegister`
    """
    cdef Handle fh
    with nogil:
        __status__ = cuFileHandleRegister(&fh, <CUfileDescr_t*>descr)
    check_status(__status__)
    return <intptr_t>fh


cpdef void handle_deregister(intptr_t fh) except*:
    """releases a registered filehandle from cuFile.

    Args:
        fh (intptr_t): ``CUfileHandle_t`` file handle.

    .. seealso:: `cuFileHandleDeregister`
    """
    cuFileHandleDeregister(<Handle>fh)


cpdef buf_register(intptr_t buf_ptr_base, size_t length, int flags):
    """register an existing cudaMalloced memory with cuFile to pin for GPUDirect Storage access or register host allocated memory with cuFile.

    Args:
        buf_ptr_base (intptr_t): buffer pointer allocated.
        length (size_t): size of memory region from the above specified bufPtr.
        flags (int): CU_FILE_RDMA_REGISTER.

    .. seealso:: `cuFileBufRegister`
    """
    with nogil:
        __status__ = cuFileBufRegister(<const void*>buf_ptr_base, length, flags)
    check_status(__status__)


cpdef buf_deregister(intptr_t buf_ptr_base):
    """deregister an already registered device or host memory from cuFile.

    Args:
        buf_ptr_base (intptr_t): buffer pointer to deregister.

    .. seealso:: `cuFileBufDeregister`
    """
    with nogil:
        __status__ = cuFileBufDeregister(<const void*>buf_ptr_base)
    check_status(__status__)


cpdef driver_open():
    """Initialize the cuFile library and open the nvidia-fs driver.

    .. seealso:: `cuFileDriverOpen`
    """
    with nogil:
        __status__ = cuFileDriverOpen()
    check_status(__status__)


cpdef use_count():
    """returns use count of cufile drivers at that moment by the process.

    .. seealso:: `cuFileUseCount`
    """
    with nogil:
        __status__ = cuFileUseCount()
    check_status(__status__)


cpdef driver_get_properties(intptr_t props):
    """Gets the Driver session properties.

    Args:
        props (intptr_t): Properties to set.

    .. seealso:: `cuFileDriverGetProperties`
    """
    with nogil:
        __status__ = cuFileDriverGetProperties(<CUfileDrvProps_t*>props)
    check_status(__status__)


cpdef driver_set_poll_mode(bint poll, size_t poll_threshold_size):
    """Sets whether the Read/Write APIs use polling to do IO operations.

    Args:
        poll (bint): boolean to indicate whether to use poll mode or not.
        poll_threshold_size (size_t): max IO size to use for POLLING mode in KB.

    .. seealso:: `cuFileDriverSetPollMode`
    """
    with nogil:
        __status__ = cuFileDriverSetPollMode(<cpp_bool>poll, poll_threshold_size)
    check_status(__status__)


cpdef driver_set_max_direct_io_size(size_t max_direct_io_size):
    """Control parameter to set max IO size(KB) used by the library to talk to nvidia-fs driver.

    Args:
        max_direct_io_size (size_t): maximum allowed direct io size in KB.

    .. seealso:: `cuFileDriverSetMaxDirectIOSize`
    """
    with nogil:
        __status__ = cuFileDriverSetMaxDirectIOSize(max_direct_io_size)
    check_status(__status__)


cpdef driver_set_max_cache_size(size_t max_cache_size):
    """Control parameter to set maximum GPU memory reserved per device by the library for internal buffering.

    Args:
        max_cache_size (size_t): The maximum GPU buffer space per device used for internal use in KB.

    .. seealso:: `cuFileDriverSetMaxCacheSize`
    """
    with nogil:
        __status__ = cuFileDriverSetMaxCacheSize(max_cache_size)
    check_status(__status__)


cpdef driver_set_max_pinned_mem_size(size_t max_pinned_size):
    """Sets maximum buffer space that is pinned in KB for use by ``cuFileBufRegister``.

    Args:
        max_pinned_size (size_t): maximum buffer space that is pinned in KB.

    .. seealso:: `cuFileDriverSetMaxPinnedMemSize`
    """
    with nogil:
        __status__ = cuFileDriverSetMaxPinnedMemSize(max_pinned_size)
    check_status(__status__)


cpdef intptr_t batch_io_set_up(unsigned nr) except? 0:
    cdef BatchHandle batch_idp
    with nogil:
        __status__ = cuFileBatchIOSetUp(&batch_idp, nr)
    check_status(__status__)
    return <intptr_t>batch_idp


cpdef batch_io_submit(intptr_t batch_idp, unsigned nr, intptr_t iocbp, unsigned int flags):
    with nogil:
        __status__ = cuFileBatchIOSubmit(<BatchHandle>batch_idp, nr, <CUfileIOParams_t*>iocbp, flags)
    check_status(__status__)


cpdef batch_io_get_status(intptr_t batch_idp, unsigned min_nr, intptr_t nr, intptr_t iocbp, intptr_t timeout):
    with nogil:
        __status__ = cuFileBatchIOGetStatus(<BatchHandle>batch_idp, min_nr, <unsigned*>nr, <CUfileIOEvents_t*>iocbp, <timespec*>timeout)
    check_status(__status__)


cpdef batch_io_cancel(intptr_t batch_idp):
    with nogil:
        __status__ = cuFileBatchIOCancel(<BatchHandle>batch_idp)
    check_status(__status__)


cpdef void batch_io_destroy(intptr_t batch_idp) except*:
    cuFileBatchIODestroy(<BatchHandle>batch_idp)


cpdef read_async(intptr_t fh, intptr_t buf_ptr_base, intptr_t size_p, intptr_t file_offset_p, intptr_t buf_ptr_offset_p, intptr_t bytes_read_p, intptr_t stream):
    with nogil:
        __status__ = cuFileReadAsync(<Handle>fh, <void*>buf_ptr_base, <size_t*>size_p, <off_t*>file_offset_p, <off_t*>buf_ptr_offset_p, <ssize_t*>bytes_read_p, <CUstream>stream)
    check_status(__status__)


cpdef write_async(intptr_t fh, intptr_t buf_ptr_base, intptr_t size_p, intptr_t file_offset_p, intptr_t buf_ptr_offset_p, intptr_t bytes_written_p, intptr_t stream):
    with nogil:
        __status__ = cuFileWriteAsync(<Handle>fh, <void*>buf_ptr_base, <size_t*>size_p, <off_t*>file_offset_p, <off_t*>buf_ptr_offset_p, <ssize_t*>bytes_written_p, <CUstream>stream)
    check_status(__status__)


cpdef stream_register(intptr_t stream, unsigned flags):
    with nogil:
        __status__ = cuFileStreamRegister(<CUstream>stream, flags)
    check_status(__status__)


cpdef stream_deregister(intptr_t stream):
    with nogil:
        __status__ = cuFileStreamDeregister(<CUstream>stream)
    check_status(__status__)


cpdef int get_version() except? 0:
    """Get the cuFile library version.

    Returns:
        int: Pointer to an integer where the version will be stored.

    .. seealso:: `cuFileGetVersion`
    """
    cdef int version
    with nogil:
        __status__ = cuFileGetVersion(&version)
    check_status(__status__)
    return version


cpdef size_t get_parameter_size_t(int param) except? 0:
    cdef size_t value
    with nogil:
        __status__ = cuFileGetParameterSizeT(<_SizeTConfigParameter>param, &value)
    check_status(__status__)
    return value


cpdef bint get_parameter_bool(int param) except? 0:
    cdef cpp_bool value
    with nogil:
        __status__ = cuFileGetParameterBool(<_BoolConfigParameter>param, &value)
    check_status(__status__)
    return <bint>value


cpdef str get_parameter_string(int param, int len):
    cdef bytes _desc_str_ = bytes(len)
    cdef char* desc_str = _desc_str_
    with nogil:
        __status__ = cuFileGetParameterString(<_StringConfigParameter>param, desc_str, len)
    check_status(__status__)
    return cpython.PyUnicode_FromString(desc_str)


cpdef set_parameter_size_t(int param, size_t value):
    with nogil:
        __status__ = cuFileSetParameterSizeT(<_SizeTConfigParameter>param, value)
    check_status(__status__)


cpdef set_parameter_bool(int param, bint value):
    with nogil:
        __status__ = cuFileSetParameterBool(<_BoolConfigParameter>param, <cpp_bool>value)
    check_status(__status__)


cpdef set_parameter_string(int param, intptr_t desc_str):
    with nogil:
        __status__ = cuFileSetParameterString(<_StringConfigParameter>param, <const char*>desc_str)
    check_status(__status__)


cpdef tuple get_parameter_min_max_value(int param):
    """Get both the minimum and maximum settable values for a given size_t parameter in a single call.

    Args:
        param (SizeTConfigParameter): CUfile SizeT configuration parameter.

    Returns:
        A 2-tuple containing:

        - size_t: Pointer to store the minimum value.
        - size_t: Pointer to store the maximum value.

    .. seealso:: `cuFileGetParameterMinMaxValue`
    """
    cdef size_t min_value
    cdef size_t max_value
    with nogil:
        __status__ = cuFileGetParameterMinMaxValue(<_SizeTConfigParameter>param, &min_value, &max_value)
    check_status(__status__)
    return (min_value, max_value)


cpdef set_stats_level(int level):
    """Set the level of statistics collection for cuFile operations. This will override the cufile.json settings for stats.

    Args:
        level (int): Statistics level (0 = disabled, 1 = basic, 2 = detailed, 3 = verbose).

    .. seealso:: `cuFileSetStatsLevel`
    """
    with nogil:
        __status__ = cuFileSetStatsLevel(level)
    check_status(__status__)


cpdef int get_stats_level() except? 0:
    """Get the current level of statistics collection for cuFile operations.

    Returns:
        int: Pointer to store the current statistics level.

    .. seealso:: `cuFileGetStatsLevel`
    """
    cdef int level
    with nogil:
        __status__ = cuFileGetStatsLevel(&level)
    check_status(__status__)
    return level


cpdef stats_start():
    """Start collecting cuFile statistics.

    .. seealso:: `cuFileStatsStart`
    """
    with nogil:
        __status__ = cuFileStatsStart()
    check_status(__status__)


cpdef stats_stop():
    """Stop collecting cuFile statistics.

    .. seealso:: `cuFileStatsStop`
    """
    with nogil:
        __status__ = cuFileStatsStop()
    check_status(__status__)


cpdef stats_reset():
    """Reset all cuFile statistics counters.

    .. seealso:: `cuFileStatsReset`
    """
    with nogil:
        __status__ = cuFileStatsReset()
    check_status(__status__)


cpdef get_stats_l1(intptr_t stats):
    """Get Level 1 cuFile statistics.

    Args:
        stats (intptr_t): Pointer to CUfileStatsLevel1_t structure to be filled.

    .. seealso:: `cuFileGetStatsL1`
    """
    with nogil:
        __status__ = cuFileGetStatsL1(<CUfileStatsLevel1_t*>stats)
    check_status(__status__)


cpdef get_stats_l2(intptr_t stats):
    """Get Level 2 cuFile statistics.

    Args:
        stats (intptr_t): Pointer to CUfileStatsLevel2_t structure to be filled.

    .. seealso:: `cuFileGetStatsL2`
    """
    with nogil:
        __status__ = cuFileGetStatsL2(<CUfileStatsLevel2_t*>stats)
    check_status(__status__)


cpdef get_stats_l3(intptr_t stats):
    """Get Level 3 cuFile statistics.

    Args:
        stats (intptr_t): Pointer to CUfileStatsLevel3_t structure to be filled.

    .. seealso:: `cuFileGetStatsL3`
    """
    with nogil:
        __status__ = cuFileGetStatsL3(<CUfileStatsLevel3_t*>stats)
    check_status(__status__)


cpdef size_t get_bar_size_in_kb(int gpu_ind_ex) except? 0:
    cdef size_t bar_size
    with nogil:
        __status__ = cuFileGetBARSizeInKB(gpu_ind_ex, &bar_size)
    check_status(__status__)
    return bar_size


cpdef set_parameter_posix_pool_slab_array(intptr_t size_values, intptr_t count_values, int len):
    """Set both POSIX pool slab size and count parameters as a pair.

    Args:
        size_values (intptr_t): Array of slab sizes in KB.
        count_values (intptr_t): Array of slab counts.
        len (int): Length of both arrays (must be the same).

    .. seealso:: `cuFileSetParameterPosixPoolSlabArray`
    """
    with nogil:
        __status__ = cuFileSetParameterPosixPoolSlabArray(<const size_t*>size_values, <const size_t*>count_values, len)
    check_status(__status__)


cpdef get_parameter_posix_pool_slab_array(intptr_t size_values, intptr_t count_values, int len):
    """Get both POSIX pool slab size and count parameters as a pair.

    Args:
        size_values (intptr_t): Buffer to receive slab sizes in KB.
        count_values (intptr_t): Buffer to receive slab counts.
        len (int): Buffer size (must match the actual parameter length).

    .. seealso:: `cuFileGetParameterPosixPoolSlabArray`
    """
    with nogil:
        __status__ = cuFileGetParameterPosixPoolSlabArray(<size_t*>size_values, <size_t*>count_values, len)
    check_status(__status__)


cpdef str op_status_error(int status):
    """cufileop status string.

    Args:
        status (OpError): the error status to query.

    .. seealso:: `cufileop_status_error`
    """
    cdef bytes _output_
    _output_ = cufileop_status_error(<_OpError>status)
    return _output_.decode()


cpdef driver_close():
    """reset the cuFile library and release the nvidia-fs driver
    """
    with nogil:
        status = cuFileDriverClose_v2()
    check_status(status)

cpdef read(intptr_t fh, intptr_t buf_ptr_base, size_t size, off_t file_offset, off_t buf_ptr_offset):
    """read data from a registered file handle to a specified device or host memory.

    Args:
        fh (intptr_t): ``CUfileHandle_t`` opaque file handle.
        buf_ptr_base (intptr_t): base address of buffer in device or host memory.
        size (size_t): size bytes to read.
        file_offset (off_t): file-offset from begining of the file.
        buf_ptr_offset (off_t): offset relative to the buf_ptr_base pointer to read into.

    Returns:
        ssize_t: number of bytes read on success.

    .. seealso:: `cuFileRead`
    """
    with nogil:
        status = cuFileRead(<Handle>fh, <void*>buf_ptr_base, size, file_offset, buf_ptr_offset)
    check_status(status)
    return status


cpdef write(intptr_t fh, intptr_t buf_ptr_base, size_t size, off_t file_offset, off_t buf_ptr_offset):
    """write data from a specified device or host memory to a registered file handle.

    Args:
        fh (intptr_t): ``CUfileHandle_t`` opaque file handle.
        buf_ptr_base (intptr_t): base address of buffer in device or host memory.
        size (size_t): size bytes to write.
        file_offset (off_t): file-offset from begining of the file.
        buf_ptr_offset (off_t): offset relative to the buf_ptr_base pointer to write from.

    Returns:
        ssize_t: number of bytes written on success.

    .. seealso:: `cuFileWrite`
    """
    with nogil:
        status = cuFileWrite(<Handle>fh, <const void*>buf_ptr_base, size, file_offset, buf_ptr_offset)
    check_status(status)
    return status
