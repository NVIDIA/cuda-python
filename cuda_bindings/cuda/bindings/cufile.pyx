# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE
#
# This code was automatically generated across versions from 12.9.0 to 13.0.1. Do not modify it directly.

cimport cython  # NOQA
from libc cimport errno
from ._internal.utils cimport (get_buffer_pointer, get_nested_resource_ptr,
                               nested_resource)
import numpy as _numpy
from cpython cimport buffer as _buffer
from cpython.memoryview cimport PyMemoryView_FromMemory
from enum import IntEnum as _IntEnum

import cython

from cuda.bindings.driver import CUresult as pyCUresult


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
        readonly object _data

    def __init__(self):
        arr = _numpy.empty(1, dtype=_py_anon_pod1_dtype)
        self._data = arr.view(_numpy.recarray)
        assert self._data.itemsize == sizeof((<CUfileDescr_t*>NULL).handle), \
            f"itemsize {self._data.itemsize} mismatches union size {sizeof((<CUfileDescr_t*>NULL).handle)}"

    def __repr__(self):
        return f"<{__name__}._py_anon_pod1 object at {hex(id(self))}>"

    @property
    def ptr(self):
        """Get the pointer address to the data as Python :class:`int`."""
        return self._data.ctypes.data

    def __int__(self):
        return self._data.ctypes.data

    def __eq__(self, other):
        if not isinstance(other, _py_anon_pod1):
            return False
        if self._data.size != other._data.size:
            return False
        if self._data.dtype != other._data.dtype:
            return False
        return bool((self._data == other._data).all())

    @property
    def fd(self):
        """int: """
        return int(self._data.fd[0])

    @fd.setter
    def fd(self, val):
        self._data.fd = val

    @property
    def handle(self):
        """int: """
        return int(self._data.handle[0])

    @handle.setter
    def handle(self, val):
        self._data.handle = val

    def __setitem__(self, key, val):
        self._data[key] = val

    @staticmethod
    def from_data(data):
        """Create an _py_anon_pod1 instance wrapping the given NumPy array.

        Args:
            data (_numpy.ndarray): a 1D array of dtype `_py_anon_pod1_dtype` holding the data.
        """
        cdef _py_anon_pod1 obj = _py_anon_pod1.__new__(_py_anon_pod1)
        if not isinstance(data, (_numpy.ndarray, _numpy.recarray)):
            raise TypeError("data argument must be a NumPy ndarray")
        if data.ndim != 1:
            raise ValueError("data array must be 1D")
        if data.dtype != _py_anon_pod1_dtype:
            raise ValueError("data array must be of dtype _py_anon_pod1_dtype")
        obj._data = data.view(_numpy.recarray)

        return obj

    @staticmethod
    def from_ptr(intptr_t ptr, bint readonly=False):
        """Create an _py_anon_pod1 instance wrapping the given pointer.

        Args:
            ptr (intptr_t): pointer address as Python :class:`int` to the data.
            readonly (bool): whether the data is read-only (to the user). default is `False`.
        """
        if ptr == 0:
            raise ValueError("ptr must not be null (0)")
        cdef _py_anon_pod1 obj = _py_anon_pod1.__new__(_py_anon_pod1)
        cdef flag = _buffer.PyBUF_READ if readonly else _buffer.PyBUF_WRITE
        cdef object buf = PyMemoryView_FromMemory(
            <char*>ptr, sizeof((<CUfileDescr_t*>NULL).handle), flag)
        data = _numpy.ndarray((1,), buffer=buf,
                              dtype=_py_anon_pod1_dtype)
        obj._data = data.view(_numpy.recarray)

        return obj


_py_anon_pod3_dtype = _numpy.dtype([
    ("dev_ptr_base", _numpy.intp, ),
    ("file_offset", _numpy.int64, ),
    ("dev_ptr_offset", _numpy.int64, ),
    ("size_", _numpy.uint64, ),
    ], align=True)


cdef class _py_anon_pod3:
    """Empty-initialize an instance of `_anon_pod3`.


    .. seealso:: `_anon_pod3`
    """
    cdef:
        readonly object _data

    def __init__(self):
        arr = _numpy.empty(1, dtype=_py_anon_pod3_dtype)
        self._data = arr.view(_numpy.recarray)
        assert self._data.itemsize == sizeof((<CUfileIOParams_t*>NULL).u.batch), \
            f"itemsize {self._data.itemsize} mismatches struct size {sizeof((<CUfileIOParams_t*>NULL).u.batch)}"

    def __repr__(self):
        return f"<{__name__}._py_anon_pod3 object at {hex(id(self))}>"

    @property
    def ptr(self):
        """Get the pointer address to the data as Python :class:`int`."""
        return self._data.ctypes.data

    def __int__(self):
        return self._data.ctypes.data

    def __eq__(self, other):
        if not isinstance(other, _py_anon_pod3):
            return False
        if self._data.size != other._data.size:
            return False
        if self._data.dtype != other._data.dtype:
            return False
        return bool((self._data == other._data).all())

    @property
    def dev_ptr_base(self):
        """int: """
        return int(self._data.dev_ptr_base[0])

    @dev_ptr_base.setter
    def dev_ptr_base(self, val):
        self._data.dev_ptr_base = val

    @property
    def file_offset(self):
        """int: """
        return int(self._data.file_offset[0])

    @file_offset.setter
    def file_offset(self, val):
        self._data.file_offset = val

    @property
    def dev_ptr_offset(self):
        """int: """
        return int(self._data.dev_ptr_offset[0])

    @dev_ptr_offset.setter
    def dev_ptr_offset(self, val):
        self._data.dev_ptr_offset = val

    @property
    def size_(self):
        """int: """
        return int(self._data.size_[0])

    @size_.setter
    def size_(self, val):
        self._data.size_ = val

    def __setitem__(self, key, val):
        self._data[key] = val

    @staticmethod
    def from_data(data):
        """Create an _py_anon_pod3 instance wrapping the given NumPy array.

        Args:
            data (_numpy.ndarray): a 1D array of dtype `_py_anon_pod3_dtype` holding the data.
        """
        cdef _py_anon_pod3 obj = _py_anon_pod3.__new__(_py_anon_pod3)
        if not isinstance(data, (_numpy.ndarray, _numpy.recarray)):
            raise TypeError("data argument must be a NumPy ndarray")
        if data.ndim != 1:
            raise ValueError("data array must be 1D")
        if data.dtype != _py_anon_pod3_dtype:
            raise ValueError("data array must be of dtype _py_anon_pod3_dtype")
        obj._data = data.view(_numpy.recarray)

        return obj

    @staticmethod
    def from_ptr(intptr_t ptr, bint readonly=False):
        """Create an _py_anon_pod3 instance wrapping the given pointer.

        Args:
            ptr (intptr_t): pointer address as Python :class:`int` to the data.
            readonly (bool): whether the data is read-only (to the user). default is `False`.
        """
        if ptr == 0:
            raise ValueError("ptr must not be null (0)")
        cdef _py_anon_pod3 obj = _py_anon_pod3.__new__(_py_anon_pod3)
        cdef flag = _buffer.PyBUF_READ if readonly else _buffer.PyBUF_WRITE
        cdef object buf = PyMemoryView_FromMemory(
            <char*>ptr, sizeof((<CUfileIOParams_t*>NULL).u.batch), flag)
        data = _numpy.ndarray((1,), buffer=buf,
                              dtype=_py_anon_pod3_dtype)
        obj._data = data.view(_numpy.recarray)

        return obj


io_events_dtype = _numpy.dtype([
    ("cookie", _numpy.intp, ),
    ("status", _numpy.int32, ),
    ("ret", _numpy.uint64, ),
    ], align=True)


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
            f"itemsize {self._data.itemsize} mismatches struct size {sizeof(CUfileIOEvents_t)}"

    def __repr__(self):
        if self._data.size > 1:
            return f"<{__name__}.IOEvents_Array_{self._data.size} object at {hex(id(self))}>"
        else:
            return f"<{__name__}.IOEvents object at {hex(id(self))}>"

    @property
    def ptr(self):
        """Get the pointer address to the data as Python :class:`int`."""
        return self._data.ctypes.data

    def __int__(self):
        if self._data.size > 1:
            raise TypeError("int() argument must be a bytes-like object of size 1. "
                            "To get the pointer address of an array, use .ptr")
        return self._data.ctypes.data

    def __len__(self):
        return self._data.size

    def __eq__(self, other):
        if not isinstance(other, IOEvents):
            return False
        if self._data.size != other._data.size:
            return False
        if self._data.dtype != other._data.dtype:
            return False
        return bool((self._data == other._data).all())

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
        if isinstance(key, int):
            size = self._data.size
            if key >= size or key <= -(size+1):
                raise IndexError("index is out of bounds")
            if key < 0:
                key += size
            return IOEvents.from_data(self._data[key:key+1])
        out = self._data[key]
        if isinstance(out, _numpy.recarray) and out.dtype == io_events_dtype:
            return IOEvents.from_data(out)
        return out

    def __setitem__(self, key, val):
        self._data[key] = val

    @staticmethod
    def from_data(data):
        """Create an IOEvents instance wrapping the given NumPy array.

        Args:
            data (_numpy.ndarray): a 1D array of dtype `io_events_dtype` holding the data.
        """
        cdef IOEvents obj = IOEvents.__new__(IOEvents)
        if not isinstance(data, (_numpy.ndarray, _numpy.recarray)):
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
        cdef flag = _buffer.PyBUF_READ if readonly else _buffer.PyBUF_WRITE
        cdef object buf = PyMemoryView_FromMemory(
            <char*>ptr, sizeof(CUfileIOEvents_t) * size, flag)
        data = _numpy.ndarray((size,), buffer=buf,
                              dtype=io_events_dtype)
        obj._data = data.view(_numpy.recarray)

        return obj


descr_dtype = _numpy.dtype([
    ("type", _numpy.int32, ),
    ("handle", _py_anon_pod1_dtype, ),
    ("fs_ops", _numpy.intp, ),
    ], align=True)


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
            f"itemsize {self._data.itemsize} mismatches struct size {sizeof(CUfileDescr_t)}"

    def __repr__(self):
        if self._data.size > 1:
            return f"<{__name__}.Descr_Array_{self._data.size} object at {hex(id(self))}>"
        else:
            return f"<{__name__}.Descr object at {hex(id(self))}>"

    @property
    def ptr(self):
        """Get the pointer address to the data as Python :class:`int`."""
        return self._data.ctypes.data

    def __int__(self):
        if self._data.size > 1:
            raise TypeError("int() argument must be a bytes-like object of size 1. "
                            "To get the pointer address of an array, use .ptr")
        return self._data.ctypes.data

    def __len__(self):
        return self._data.size

    def __eq__(self, other):
        if not isinstance(other, Descr):
            return False
        if self._data.size != other._data.size:
            return False
        if self._data.dtype != other._data.dtype:
            return False
        return bool((self._data == other._data).all())

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
        if isinstance(key, int):
            size = self._data.size
            if key >= size or key <= -(size+1):
                raise IndexError("index is out of bounds")
            if key < 0:
                key += size
            return Descr.from_data(self._data[key:key+1])
        out = self._data[key]
        if isinstance(out, _numpy.recarray) and out.dtype == descr_dtype:
            return Descr.from_data(out)
        return out

    def __setitem__(self, key, val):
        self._data[key] = val

    @staticmethod
    def from_data(data):
        """Create an Descr instance wrapping the given NumPy array.

        Args:
            data (_numpy.ndarray): a 1D array of dtype `descr_dtype` holding the data.
        """
        cdef Descr obj = Descr.__new__(Descr)
        if not isinstance(data, (_numpy.ndarray, _numpy.recarray)):
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
        cdef flag = _buffer.PyBUF_READ if readonly else _buffer.PyBUF_WRITE
        cdef object buf = PyMemoryView_FromMemory(
            <char*>ptr, sizeof(CUfileDescr_t) * size, flag)
        data = _numpy.ndarray((size,), buffer=buf,
                              dtype=descr_dtype)
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
        readonly object _data

        readonly object _batch

    def __init__(self):
        arr = _numpy.empty(1, dtype=_py_anon_pod2_dtype)
        self._data = arr.view(_numpy.recarray)
        assert self._data.itemsize == sizeof((<CUfileIOParams_t*>NULL).u), \
            f"itemsize {self._data.itemsize} mismatches union size {sizeof((<CUfileIOParams_t*>NULL).u)}"

    def __repr__(self):
        return f"<{__name__}._py_anon_pod2 object at {hex(id(self))}>"

    @property
    def ptr(self):
        """Get the pointer address to the data as Python :class:`int`."""
        return self._data.ctypes.data

    def __int__(self):
        return self._data.ctypes.data

    def __eq__(self, other):
        if not isinstance(other, _py_anon_pod2):
            return False
        if self._data.size != other._data.size:
            return False
        if self._data.dtype != other._data.dtype:
            return False
        return bool((self._data == other._data).all())

    @property
    def batch(self):
        """_py_anon_pod3: """
        return self._batch

    def __setitem__(self, key, val):
        self._data[key] = val

    @staticmethod
    def from_data(data):
        """Create an _py_anon_pod2 instance wrapping the given NumPy array.

        Args:
            data (_numpy.ndarray): a 1D array of dtype `_py_anon_pod2_dtype` holding the data.
        """
        cdef _py_anon_pod2 obj = _py_anon_pod2.__new__(_py_anon_pod2)
        if not isinstance(data, (_numpy.ndarray, _numpy.recarray)):
            raise TypeError("data argument must be a NumPy ndarray")
        if data.ndim != 1:
            raise ValueError("data array must be 1D")
        if data.dtype != _py_anon_pod2_dtype:
            raise ValueError("data array must be of dtype _py_anon_pod2_dtype")
        obj._data = data.view(_numpy.recarray)

        batch_addr = obj._data.batch[0].__array_interface__['data'][0]
        obj._batch = _py_anon_pod3.from_ptr(batch_addr)
        return obj

    @staticmethod
    def from_ptr(intptr_t ptr, bint readonly=False):
        """Create an _py_anon_pod2 instance wrapping the given pointer.

        Args:
            ptr (intptr_t): pointer address as Python :class:`int` to the data.
            readonly (bool): whether the data is read-only (to the user). default is `False`.
        """
        if ptr == 0:
            raise ValueError("ptr must not be null (0)")
        cdef _py_anon_pod2 obj = _py_anon_pod2.__new__(_py_anon_pod2)
        cdef flag = _buffer.PyBUF_READ if readonly else _buffer.PyBUF_WRITE
        cdef object buf = PyMemoryView_FromMemory(
            <char*>ptr, sizeof((<CUfileIOParams_t*>NULL).u), flag)
        data = _numpy.ndarray((1,), buffer=buf,
                              dtype=_py_anon_pod2_dtype)
        obj._data = data.view(_numpy.recarray)

        batch_addr = obj._data.batch[0].__array_interface__['data'][0]
        obj._batch = _py_anon_pod3.from_ptr(batch_addr)
        return obj


io_params_dtype = _numpy.dtype([
    ("mode", _numpy.int32, ),
    ("u", _py_anon_pod2_dtype, ),
    ("fh", _numpy.intp, ),
    ("opcode", _numpy.int32, ),
    ("cookie", _numpy.intp, ),
    ], align=True)


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
            f"itemsize {self._data.itemsize} mismatches struct size {sizeof(CUfileIOParams_t)}"

    def __repr__(self):
        if self._data.size > 1:
            return f"<{__name__}.IOParams_Array_{self._data.size} object at {hex(id(self))}>"
        else:
            return f"<{__name__}.IOParams object at {hex(id(self))}>"

    @property
    def ptr(self):
        """Get the pointer address to the data as Python :class:`int`."""
        return self._data.ctypes.data

    def __int__(self):
        if self._data.size > 1:
            raise TypeError("int() argument must be a bytes-like object of size 1. "
                            "To get the pointer address of an array, use .ptr")
        return self._data.ctypes.data

    def __len__(self):
        return self._data.size

    def __eq__(self, other):
        if not isinstance(other, IOParams):
            return False
        if self._data.size != other._data.size:
            return False
        if self._data.dtype != other._data.dtype:
            return False
        return bool((self._data == other._data).all())

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
        if isinstance(key, int):
            size = self._data.size
            if key >= size or key <= -(size+1):
                raise IndexError("index is out of bounds")
            if key < 0:
                key += size
            return IOParams.from_data(self._data[key:key+1])
        out = self._data[key]
        if isinstance(out, _numpy.recarray) and out.dtype == io_params_dtype:
            return IOParams.from_data(out)
        return out

    def __setitem__(self, key, val):
        self._data[key] = val

    @staticmethod
    def from_data(data):
        """Create an IOParams instance wrapping the given NumPy array.

        Args:
            data (_numpy.ndarray): a 1D array of dtype `io_params_dtype` holding the data.
        """
        cdef IOParams obj = IOParams.__new__(IOParams)
        if not isinstance(data, (_numpy.ndarray, _numpy.recarray)):
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
        cdef flag = _buffer.PyBUF_READ if readonly else _buffer.PyBUF_WRITE
        cdef object buf = PyMemoryView_FromMemory(
            <char*>ptr, sizeof(CUfileIOParams_t) * size, flag)
        data = _numpy.ndarray((size,), buffer=buf,
                              dtype=io_params_dtype)
        obj._data = data.view(_numpy.recarray)

        return obj


# Hack: Overwrite the generated descr_dtype, which NumPy deduced the offset wrong.
descr_dtype = _numpy.dtype({
    "names": ['type', 'handle', 'fs_ops'],
    "formats": [_numpy.int32, _py_anon_pod1_dtype, _numpy.intp],
    "offsets": [0, 8, 16],
}, align=True)

# Hack: Overwrite the generated io_params_dtype, which NumPy deduced the offset wrong.
io_params_dtype = _numpy.dtype({
    "names": ['mode', 'u', 'fh', 'opcode', 'cookie'],
    "formats": [_numpy.int32, _py_anon_pod2_dtype, _numpy.intp, _numpy.int32, _numpy.intp],
    "offsets": [0, 8, 40, 48, 56],
}, align=True)


###############################################################################
# Enum
###############################################################################

class OpError(_IntEnum):
    """See `CUfileOpError`."""
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

class DriverStatusFlags(_IntEnum):
    """See `CUfileDriverStatusFlags_t`."""
    LUSTRE_SUPPORTED = CU_FILE_LUSTRE_SUPPORTED
    WEKAFS_SUPPORTED = CU_FILE_WEKAFS_SUPPORTED
    NFS_SUPPORTED = CU_FILE_NFS_SUPPORTED
    GPFS_SUPPORTED = CU_FILE_GPFS_SUPPORTED
    NVME_SUPPORTED = CU_FILE_NVME_SUPPORTED
    NVMEOF_SUPPORTED = CU_FILE_NVMEOF_SUPPORTED
    SCSI_SUPPORTED = CU_FILE_SCSI_SUPPORTED
    SCALEFLUX_CSD_SUPPORTED = CU_FILE_SCALEFLUX_CSD_SUPPORTED
    NVMESH_SUPPORTED = CU_FILE_NVMESH_SUPPORTED
    BEEGFS_SUPPORTED = CU_FILE_BEEGFS_SUPPORTED
    NVME_P2P_SUPPORTED = CU_FILE_NVME_P2P_SUPPORTED
    SCATEFS_SUPPORTED = CU_FILE_SCATEFS_SUPPORTED

class DriverControlFlags(_IntEnum):
    """See `CUfileDriverControlFlags_t`."""
    USE_POLL_MODE = CU_FILE_USE_POLL_MODE
    ALLOW_COMPAT_MODE = CU_FILE_ALLOW_COMPAT_MODE

class FeatureFlags(_IntEnum):
    """See `CUfileFeatureFlags_t`."""
    DYN_ROUTING_SUPPORTED = CU_FILE_DYN_ROUTING_SUPPORTED
    BATCH_IO_SUPPORTED = CU_FILE_BATCH_IO_SUPPORTED
    STREAMS_SUPPORTED = CU_FILE_STREAMS_SUPPORTED
    PARALLEL_IO_SUPPORTED = CU_FILE_PARALLEL_IO_SUPPORTED

class FileHandleType(_IntEnum):
    """See `CUfileFileHandleType`."""
    OPAQUE_FD = CU_FILE_HANDLE_TYPE_OPAQUE_FD
    OPAQUE_WIN32 = CU_FILE_HANDLE_TYPE_OPAQUE_WIN32
    USERSPACE_FS = CU_FILE_HANDLE_TYPE_USERSPACE_FS

class Opcode(_IntEnum):
    """See `CUfileOpcode_t`."""
    READ = CUFILE_READ
    WRITE = CUFILE_WRITE

class Status(_IntEnum):
    """See `CUfileStatus_t`."""
    WAITING = CUFILE_WAITING
    PENDING = CUFILE_PENDING
    INVALID = CUFILE_INVALID
    CANCELED = CUFILE_CANCELED
    COMPLETE = CUFILE_COMPLETE
    TIMEOUT = CUFILE_TIMEOUT
    FAILED = CUFILE_FAILED

class BatchMode(_IntEnum):
    """See `CUfileBatchMode_t`."""
    BATCH = CUFILE_BATCH

class SizeTConfigParameter(_IntEnum):
    """See `CUFileSizeTConfigParameter_t`."""
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

class BoolConfigParameter(_IntEnum):
    """See `CUFileBoolConfigParameter_t`."""
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

class StringConfigParameter(_IntEnum):
    """See `CUFileStringConfigParameter_t`."""
    LOGGING_LEVEL = CUFILE_PARAM_LOGGING_LEVEL
    ENV_LOGFILE_PATH = CUFILE_PARAM_ENV_LOGFILE_PATH
    LOG_DIR = CUFILE_PARAM_LOG_DIR

class ArrayConfigParameter(_IntEnum):
    """See `CUFileArrayConfigParameter_t`."""
    POSIX_POOL_SLAB_SIZE_KB = CUFILE_PARAM_POSIX_POOL_SLAB_SIZE_KB
    POSIX_POOL_SLAB_COUNT = CUFILE_PARAM_POSIX_POOL_SLAB_COUNT


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
        status = cuFileHandleRegister(&fh, <CUfileDescr_t*>descr)
    check_status(status)
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
        status = cuFileBufRegister(<const void*>buf_ptr_base, length, flags)
    check_status(status)


cpdef buf_deregister(intptr_t buf_ptr_base):
    """deregister an already registered device or host memory from cuFile.

    Args:
        buf_ptr_base (intptr_t): buffer pointer to deregister.

    .. seealso:: `cuFileBufDeregister`
    """
    with nogil:
        status = cuFileBufDeregister(<const void*>buf_ptr_base)
    check_status(status)


cpdef read(intptr_t fh, intptr_t buf_ptr_base, size_t size, off_t file_offset, off_t buf_ptr_offset):
    """read data from a registered file handle to a specified device or host memory.

    Args:
        fh (intptr_t): ``CUfileHandle_t`` opaque file handle.
        buf_ptr_base (intptr_t): base address of buffer in device or host memory.
        size (size_t): size bytes to read.
        file_offset (off_t): file-offset from begining of the file.
        buf_ptr_offset (off_t): offset relative to the buf_ptr_base pointer to read into.

    .. seealso:: `cuFileRead`
    """
    with nogil:
        status = cuFileRead(<Handle>fh, <void*>buf_ptr_base, size, file_offset, buf_ptr_offset)
    check_status(status)


cpdef write(intptr_t fh, intptr_t buf_ptr_base, size_t size, off_t file_offset, off_t buf_ptr_offset):
    """write data from a specified device or host memory to a registered file handle.

    Args:
        fh (intptr_t): ``CUfileHandle_t`` opaque file handle.
        buf_ptr_base (intptr_t): base address of buffer in device or host memory.
        size (size_t): size bytes to write.
        file_offset (off_t): file-offset from begining of the file.
        buf_ptr_offset (off_t): offset relative to the buf_ptr_base pointer to write from.

    .. seealso:: `cuFileWrite`
    """
    with nogil:
        status = cuFileWrite(<Handle>fh, <const void*>buf_ptr_base, size, file_offset, buf_ptr_offset)
    check_status(status)


cpdef driver_open():
    """Initialize the cuFile library and open the nvidia-fs driver.

    .. seealso:: `cuFileDriverOpen`
    """
    with nogil:
        status = cuFileDriverOpen()
    check_status(status)


cpdef use_count():
    """returns use count of cufile drivers at that moment by the process.

    .. seealso:: `cuFileUseCount`
    """
    with nogil:
        status = cuFileUseCount()
    check_status(status)


cpdef driver_get_properties(intptr_t props):
    """Gets the Driver session properties.

    Args:
        props (intptr_t): Properties to set.

    .. seealso:: `cuFileDriverGetProperties`
    """
    with nogil:
        status = cuFileDriverGetProperties(<CUfileDrvProps_t*>props)
    check_status(status)


cpdef driver_set_poll_mode(bint poll, size_t poll_threshold_size):
    """Sets whether the Read/Write APIs use polling to do IO operations.

    Args:
        poll (bint): boolean to indicate whether to use poll mode or not.
        poll_threshold_size (size_t): max IO size to use for POLLING mode in KB.

    .. seealso:: `cuFileDriverSetPollMode`
    """
    with nogil:
        status = cuFileDriverSetPollMode(<cpp_bool>poll, poll_threshold_size)
    check_status(status)


cpdef driver_set_max_direct_io_size(size_t max_direct_io_size):
    """Control parameter to set max IO size(KB) used by the library to talk to nvidia-fs driver.

    Args:
        max_direct_io_size (size_t): maximum allowed direct io size in KB.

    .. seealso:: `cuFileDriverSetMaxDirectIOSize`
    """
    with nogil:
        status = cuFileDriverSetMaxDirectIOSize(max_direct_io_size)
    check_status(status)


cpdef driver_set_max_cache_size(size_t max_cache_size):
    """Control parameter to set maximum GPU memory reserved per device by the library for internal buffering.

    Args:
        max_cache_size (size_t): The maximum GPU buffer space per device used for internal use in KB.

    .. seealso:: `cuFileDriverSetMaxCacheSize`
    """
    with nogil:
        status = cuFileDriverSetMaxCacheSize(max_cache_size)
    check_status(status)


cpdef driver_set_max_pinned_mem_size(size_t max_pinned_size):
    """Sets maximum buffer space that is pinned in KB for use by ``cuFileBufRegister``.

    Args:
        max_pinned_size (size_t): maximum buffer space that is pinned in KB.

    .. seealso:: `cuFileDriverSetMaxPinnedMemSize`
    """
    with nogil:
        status = cuFileDriverSetMaxPinnedMemSize(max_pinned_size)
    check_status(status)


cpdef intptr_t batch_io_set_up(unsigned nr) except? 0:
    cdef BatchHandle batch_idp
    with nogil:
        status = cuFileBatchIOSetUp(&batch_idp, nr)
    check_status(status)
    return <intptr_t>batch_idp


cpdef batch_io_submit(intptr_t batch_idp, unsigned nr, intptr_t iocbp, unsigned int flags):
    with nogil:
        status = cuFileBatchIOSubmit(<BatchHandle>batch_idp, nr, <CUfileIOParams_t*>iocbp, flags)
    check_status(status)


cpdef batch_io_get_status(intptr_t batch_idp, unsigned min_nr, intptr_t nr, intptr_t iocbp, intptr_t timeout):
    with nogil:
        status = cuFileBatchIOGetStatus(<BatchHandle>batch_idp, min_nr, <unsigned*>nr, <CUfileIOEvents_t*>iocbp, <timespec*>timeout)
    check_status(status)


cpdef batch_io_cancel(intptr_t batch_idp):
    with nogil:
        status = cuFileBatchIOCancel(<BatchHandle>batch_idp)
    check_status(status)


cpdef void batch_io_destroy(intptr_t batch_idp) except*:
    cuFileBatchIODestroy(<BatchHandle>batch_idp)


cpdef read_async(intptr_t fh, intptr_t buf_ptr_base, intptr_t size_p, intptr_t file_offset_p, intptr_t buf_ptr_offset_p, intptr_t bytes_read_p, intptr_t stream):
    with nogil:
        status = cuFileReadAsync(<Handle>fh, <void*>buf_ptr_base, <size_t*>size_p, <off_t*>file_offset_p, <off_t*>buf_ptr_offset_p, <ssize_t*>bytes_read_p, <void*>stream)
    check_status(status)


cpdef write_async(intptr_t fh, intptr_t buf_ptr_base, intptr_t size_p, intptr_t file_offset_p, intptr_t buf_ptr_offset_p, intptr_t bytes_written_p, intptr_t stream):
    with nogil:
        status = cuFileWriteAsync(<Handle>fh, <void*>buf_ptr_base, <size_t*>size_p, <off_t*>file_offset_p, <off_t*>buf_ptr_offset_p, <ssize_t*>bytes_written_p, <void*>stream)
    check_status(status)


cpdef stream_register(intptr_t stream, unsigned flags):
    with nogil:
        status = cuFileStreamRegister(<void*>stream, flags)
    check_status(status)


cpdef stream_deregister(intptr_t stream):
    with nogil:
        status = cuFileStreamDeregister(<void*>stream)
    check_status(status)


cpdef int get_version() except? 0:
    cdef int version
    with nogil:
        status = cuFileGetVersion(&version)
    check_status(status)
    return version


cpdef size_t get_parameter_size_t(int param) except? 0:
    cdef size_t value
    with nogil:
        status = cuFileGetParameterSizeT(<_SizeTConfigParameter>param, &value)
    check_status(status)
    return value


cpdef bint get_parameter_bool(int param) except? 0:
    cdef cpp_bool value
    with nogil:
        status = cuFileGetParameterBool(<_BoolConfigParameter>param, &value)
    check_status(status)
    return <bint>value


cpdef str get_parameter_string(int param, int len):
    cdef bytes _desc_str_ = bytes(len)
    cdef char* desc_str = _desc_str_
    with nogil:
        status = cuFileGetParameterString(<_StringConfigParameter>param, desc_str, len)
    check_status(status)
    return _desc_str_.decode()


cpdef set_parameter_size_t(int param, size_t value):
    with nogil:
        status = cuFileSetParameterSizeT(<_SizeTConfigParameter>param, value)
    check_status(status)


cpdef set_parameter_bool(int param, bint value):
    with nogil:
        status = cuFileSetParameterBool(<_BoolConfigParameter>param, <cpp_bool>value)
    check_status(status)


cpdef set_parameter_string(int param, intptr_t desc_str):
    with nogil:
        status = cuFileSetParameterString(<_StringConfigParameter>param, <const char*>desc_str)
    check_status(status)


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
