# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

# This code was automatically generated with version 1.5.0, generator version 0.3.1.dev1465+gc5c5c8652. Do not modify it directly.

cimport cython  # NOQA
from libc.stdint cimport intptr_t, uintptr_t

from ._internal.utils cimport get_buffer_pointer

from enum import IntEnum as _IntEnum

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

cdef _get_external_memory_handle_desc_dtype_offsets():
    cdef cudlaExternalMemoryHandleDesc_t pod = cudlaExternalMemoryHandleDesc_t()
    return _numpy.dtype({
        'names': ['ext_buf_object', 'size_'],
        'formats': [_numpy.intp, _numpy.uint64],
        'offsets': [
            (<intptr_t>&(pod.extBufObject)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.size)) - (<intptr_t>&pod),
        ],
        'itemsize': sizeof(cudlaExternalMemoryHandleDesc_t),
    })

external_memory_handle_desc_dtype = _get_external_memory_handle_desc_dtype_offsets()

cdef class ExternalMemoryHandleDesc:
    """Empty-initialize an instance of `cudlaExternalMemoryHandleDesc_t`.


    .. seealso:: `cudlaExternalMemoryHandleDesc_t`
    """
    cdef:
        cudlaExternalMemoryHandleDesc_t *_ptr
        object _owner
        bint _owned
        bint _readonly

    def __init__(self):
        self._ptr = <cudlaExternalMemoryHandleDesc_t *>calloc(1, sizeof(cudlaExternalMemoryHandleDesc_t))
        if self._ptr == NULL:
            raise MemoryError("Error allocating ExternalMemoryHandleDesc")
        self._owner = None
        self._owned = True
        self._readonly = False

    def __dealloc__(self):
        cdef cudlaExternalMemoryHandleDesc_t *ptr
        if self._owned and self._ptr != NULL:
            ptr = self._ptr
            self._ptr = NULL
            free(ptr)

    def __repr__(self):
        return f"<{__name__}.ExternalMemoryHandleDesc object at {hex(id(self))}>"

    @property
    def ptr(self):
        """Get the pointer address to the data as Python :class:`int`."""
        return <intptr_t>(self._ptr)

    cdef intptr_t _get_ptr(self):
        return <intptr_t>(self._ptr)

    def __int__(self):
        return <intptr_t>(self._ptr)

    def __eq__(self, other):
        cdef ExternalMemoryHandleDesc other_
        if not isinstance(other, ExternalMemoryHandleDesc):
            return False
        other_ = other
        return (memcmp(<void *><intptr_t>(self._ptr), <void *><intptr_t>(other_._ptr), sizeof(cudlaExternalMemoryHandleDesc_t)) == 0)

    def __getbuffer__(self, Py_buffer *buffer, int flags):
        __getbuffer(self, buffer, <void *>self._ptr, sizeof(cudlaExternalMemoryHandleDesc_t), self._readonly)

    def __releasebuffer__(self, Py_buffer *buffer):
        pass

    def __setitem__(self, key, val):
        if key == 0 and isinstance(val, _numpy.ndarray):
            self._ptr = <cudlaExternalMemoryHandleDesc_t *>malloc(sizeof(cudlaExternalMemoryHandleDesc_t))
            if self._ptr == NULL:
                raise MemoryError("Error allocating ExternalMemoryHandleDesc")
            memcpy(<void*>self._ptr, <void*><intptr_t>val.ctypes.data, sizeof(cudlaExternalMemoryHandleDesc_t))
            self._owner = None
            self._owned = True
            self._readonly = not val.flags.writeable
        else:
            setattr(self, key, val)

    @property
    def ext_buf_object(self):
        """int: """
        return <intptr_t>(self._ptr[0].extBufObject)

    @ext_buf_object.setter
    def ext_buf_object(self, val):
        if self._readonly:
            raise ValueError("This ExternalMemoryHandleDesc instance is read-only")
        self._ptr[0].extBufObject = <void *><intptr_t>val

    @property
    def size_(self):
        """int: """
        return self._ptr[0].size

    @size_.setter
    def size_(self, val):
        if self._readonly:
            raise ValueError("This ExternalMemoryHandleDesc instance is read-only")
        self._ptr[0].size = val

    @staticmethod
    def from_buffer(buffer):
        """Create an ExternalMemoryHandleDesc instance with the memory from the given buffer."""
        return __from_buffer(buffer, sizeof(cudlaExternalMemoryHandleDesc_t), ExternalMemoryHandleDesc)

    @staticmethod
    def from_data(data):
        """Create an ExternalMemoryHandleDesc instance wrapping the given NumPy array.

        Args:
            data (_numpy.ndarray): a single-element array of dtype `external_memory_handle_desc_dtype` holding the data.
        """
        return __from_data(data, "external_memory_handle_desc_dtype", external_memory_handle_desc_dtype, ExternalMemoryHandleDesc)

    @staticmethod
    def from_ptr(intptr_t ptr, bint readonly=False, object owner=None):
        """Create an ExternalMemoryHandleDesc instance wrapping the given pointer.

        Args:
            ptr (intptr_t): pointer address as Python :class:`int` to the data.
            owner (object): The Python object that owns the pointer. If not provided, data will be copied.
            readonly (bool): whether the data is read-only (to the user). default is `False`.
        """
        if ptr == 0:
            raise ValueError("ptr must not be null (0)")
        cdef ExternalMemoryHandleDesc obj = ExternalMemoryHandleDesc.__new__(ExternalMemoryHandleDesc)
        if owner is None:
            obj._ptr = <cudlaExternalMemoryHandleDesc_t *>malloc(sizeof(cudlaExternalMemoryHandleDesc_t))
            if obj._ptr == NULL:
                raise MemoryError("Error allocating ExternalMemoryHandleDesc")
            memcpy(<void*>(obj._ptr), <void*>ptr, sizeof(cudlaExternalMemoryHandleDesc_t))
            obj._owner = None
            obj._owned = True
        else:
            obj._ptr = <cudlaExternalMemoryHandleDesc_t *>ptr
            obj._owner = owner
            obj._owned = False
        obj._readonly = readonly
        return obj


cdef _get_external_semaphore_handle_desc_dtype_offsets():
    cdef cudlaExternalSemaphoreHandleDesc_t pod = cudlaExternalSemaphoreHandleDesc_t()
    return _numpy.dtype({
        'names': ['ext_sync_object'],
        'formats': [_numpy.intp],
        'offsets': [
            (<intptr_t>&(pod.extSyncObject)) - (<intptr_t>&pod),
        ],
        'itemsize': sizeof(cudlaExternalSemaphoreHandleDesc_t),
    })

external_semaphore_handle_desc_dtype = _get_external_semaphore_handle_desc_dtype_offsets()

cdef class ExternalSemaphoreHandleDesc:
    """Empty-initialize an instance of `cudlaExternalSemaphoreHandleDesc_t`.


    .. seealso:: `cudlaExternalSemaphoreHandleDesc_t`
    """
    cdef:
        cudlaExternalSemaphoreHandleDesc_t *_ptr
        object _owner
        bint _owned
        bint _readonly

    def __init__(self):
        self._ptr = <cudlaExternalSemaphoreHandleDesc_t *>calloc(1, sizeof(cudlaExternalSemaphoreHandleDesc_t))
        if self._ptr == NULL:
            raise MemoryError("Error allocating ExternalSemaphoreHandleDesc")
        self._owner = None
        self._owned = True
        self._readonly = False

    def __dealloc__(self):
        cdef cudlaExternalSemaphoreHandleDesc_t *ptr
        if self._owned and self._ptr != NULL:
            ptr = self._ptr
            self._ptr = NULL
            free(ptr)

    def __repr__(self):
        return f"<{__name__}.ExternalSemaphoreHandleDesc object at {hex(id(self))}>"

    @property
    def ptr(self):
        """Get the pointer address to the data as Python :class:`int`."""
        return <intptr_t>(self._ptr)

    cdef intptr_t _get_ptr(self):
        return <intptr_t>(self._ptr)

    def __int__(self):
        return <intptr_t>(self._ptr)

    def __eq__(self, other):
        cdef ExternalSemaphoreHandleDesc other_
        if not isinstance(other, ExternalSemaphoreHandleDesc):
            return False
        other_ = other
        return (memcmp(<void *><intptr_t>(self._ptr), <void *><intptr_t>(other_._ptr), sizeof(cudlaExternalSemaphoreHandleDesc_t)) == 0)

    def __getbuffer__(self, Py_buffer *buffer, int flags):
        __getbuffer(self, buffer, <void *>self._ptr, sizeof(cudlaExternalSemaphoreHandleDesc_t), self._readonly)

    def __releasebuffer__(self, Py_buffer *buffer):
        pass

    def __setitem__(self, key, val):
        if key == 0 and isinstance(val, _numpy.ndarray):
            self._ptr = <cudlaExternalSemaphoreHandleDesc_t *>malloc(sizeof(cudlaExternalSemaphoreHandleDesc_t))
            if self._ptr == NULL:
                raise MemoryError("Error allocating ExternalSemaphoreHandleDesc")
            memcpy(<void*>self._ptr, <void*><intptr_t>val.ctypes.data, sizeof(cudlaExternalSemaphoreHandleDesc_t))
            self._owner = None
            self._owned = True
            self._readonly = not val.flags.writeable
        else:
            setattr(self, key, val)

    @property
    def ext_sync_object(self):
        """int: """
        return <intptr_t>(self._ptr[0].extSyncObject)

    @ext_sync_object.setter
    def ext_sync_object(self, val):
        if self._readonly:
            raise ValueError("This ExternalSemaphoreHandleDesc instance is read-only")
        self._ptr[0].extSyncObject = <void *><intptr_t>val

    @staticmethod
    def from_buffer(buffer):
        """Create an ExternalSemaphoreHandleDesc instance with the memory from the given buffer."""
        return __from_buffer(buffer, sizeof(cudlaExternalSemaphoreHandleDesc_t), ExternalSemaphoreHandleDesc)

    @staticmethod
    def from_data(data):
        """Create an ExternalSemaphoreHandleDesc instance wrapping the given NumPy array.

        Args:
            data (_numpy.ndarray): a single-element array of dtype `external_semaphore_handle_desc_dtype` holding the data.
        """
        return __from_data(data, "external_semaphore_handle_desc_dtype", external_semaphore_handle_desc_dtype, ExternalSemaphoreHandleDesc)

    @staticmethod
    def from_ptr(intptr_t ptr, bint readonly=False, object owner=None):
        """Create an ExternalSemaphoreHandleDesc instance wrapping the given pointer.

        Args:
            ptr (intptr_t): pointer address as Python :class:`int` to the data.
            owner (object): The Python object that owns the pointer. If not provided, data will be copied.
            readonly (bool): whether the data is read-only (to the user). default is `False`.
        """
        if ptr == 0:
            raise ValueError("ptr must not be null (0)")
        cdef ExternalSemaphoreHandleDesc obj = ExternalSemaphoreHandleDesc.__new__(ExternalSemaphoreHandleDesc)
        if owner is None:
            obj._ptr = <cudlaExternalSemaphoreHandleDesc_t *>malloc(sizeof(cudlaExternalSemaphoreHandleDesc_t))
            if obj._ptr == NULL:
                raise MemoryError("Error allocating ExternalSemaphoreHandleDesc")
            memcpy(<void*>(obj._ptr), <void*>ptr, sizeof(cudlaExternalSemaphoreHandleDesc_t))
            obj._owner = None
            obj._owned = True
        else:
            obj._ptr = <cudlaExternalSemaphoreHandleDesc_t *>ptr
            obj._owner = owner
            obj._owned = False
        obj._readonly = readonly
        return obj


cdef _get_module_tensor_descriptor_dtype_offsets():
    cdef cudlaModuleTensorDescriptor pod = cudlaModuleTensorDescriptor()
    return _numpy.dtype({
        'names': ['name', 'size_', 'n', 'c', 'h', 'w', 'data_format', 'data_type', 'data_category', 'pixel_format', 'pixel_mapping', 'stride'],
        'formats': [(_numpy.int8, 81), _numpy.uint64, _numpy.uint64, _numpy.uint64, _numpy.uint64, _numpy.uint64, _numpy.uint8, _numpy.uint8, _numpy.uint8, _numpy.uint8, _numpy.uint8, (_numpy.uint32, 8)],
        'offsets': [
            (<intptr_t>&(pod.name)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.size)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.n)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.c)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.h)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.w)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.dataFormat)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.dataType)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.dataCategory)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.pixelFormat)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.pixelMapping)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.stride)) - (<intptr_t>&pod),
        ],
        'itemsize': sizeof(cudlaModuleTensorDescriptor),
    })

module_tensor_descriptor_dtype = _get_module_tensor_descriptor_dtype_offsets()

cdef class ModuleTensorDescriptor:
    """Empty-initialize an instance of `cudlaModuleTensorDescriptor`.


    .. seealso:: `cudlaModuleTensorDescriptor`
    """
    cdef:
        cudlaModuleTensorDescriptor *_ptr
        object _owner
        bint _owned
        bint _readonly

    def __init__(self):
        self._ptr = <cudlaModuleTensorDescriptor *>calloc(1, sizeof(cudlaModuleTensorDescriptor))
        if self._ptr == NULL:
            raise MemoryError("Error allocating ModuleTensorDescriptor")
        self._owner = None
        self._owned = True
        self._readonly = False

    def __dealloc__(self):
        cdef cudlaModuleTensorDescriptor *ptr
        if self._owned and self._ptr != NULL:
            ptr = self._ptr
            self._ptr = NULL
            free(ptr)

    def __repr__(self):
        return f"<{__name__}.ModuleTensorDescriptor object at {hex(id(self))}>"

    @property
    def ptr(self):
        """Get the pointer address to the data as Python :class:`int`."""
        return <intptr_t>(self._ptr)

    cdef intptr_t _get_ptr(self):
        return <intptr_t>(self._ptr)

    def __int__(self):
        return <intptr_t>(self._ptr)

    def __eq__(self, other):
        cdef ModuleTensorDescriptor other_
        if not isinstance(other, ModuleTensorDescriptor):
            return False
        other_ = other
        return (memcmp(<void *><intptr_t>(self._ptr), <void *><intptr_t>(other_._ptr), sizeof(cudlaModuleTensorDescriptor)) == 0)

    def __getbuffer__(self, Py_buffer *buffer, int flags):
        __getbuffer(self, buffer, <void *>self._ptr, sizeof(cudlaModuleTensorDescriptor), self._readonly)

    def __releasebuffer__(self, Py_buffer *buffer):
        pass

    def __setitem__(self, key, val):
        if key == 0 and isinstance(val, _numpy.ndarray):
            self._ptr = <cudlaModuleTensorDescriptor *>malloc(sizeof(cudlaModuleTensorDescriptor))
            if self._ptr == NULL:
                raise MemoryError("Error allocating ModuleTensorDescriptor")
            memcpy(<void*>self._ptr, <void*><intptr_t>val.ctypes.data, sizeof(cudlaModuleTensorDescriptor))
            self._owner = None
            self._owned = True
            self._readonly = not val.flags.writeable
        else:
            setattr(self, key, val)

    @property
    def name(self):
        """~_numpy.int8: (array of length 81)."""
        return cpython.PyUnicode_FromString(self._ptr[0].name)

    @name.setter
    def name(self, val):
        if self._readonly:
            raise ValueError("This ModuleTensorDescriptor instance is read-only")
        cdef bytes buf = val.encode()
        if len(buf) >= 81:
            raise ValueError("String too long for field name, max length is 80")
        cdef char *ptr = buf
        memcpy(<void *>(self._ptr[0].name), <void *>ptr, 81)

    @property
    def size_(self):
        """int: """
        return self._ptr[0].size

    @size_.setter
    def size_(self, val):
        if self._readonly:
            raise ValueError("This ModuleTensorDescriptor instance is read-only")
        self._ptr[0].size = val

    @property
    def n(self):
        """int: """
        return self._ptr[0].n

    @n.setter
    def n(self, val):
        if self._readonly:
            raise ValueError("This ModuleTensorDescriptor instance is read-only")
        self._ptr[0].n = val

    @property
    def c(self):
        """int: """
        return self._ptr[0].c

    @c.setter
    def c(self, val):
        if self._readonly:
            raise ValueError("This ModuleTensorDescriptor instance is read-only")
        self._ptr[0].c = val

    @property
    def h(self):
        """int: """
        return self._ptr[0].h

    @h.setter
    def h(self, val):
        if self._readonly:
            raise ValueError("This ModuleTensorDescriptor instance is read-only")
        self._ptr[0].h = val

    @property
    def w(self):
        """int: """
        return self._ptr[0].w

    @w.setter
    def w(self, val):
        if self._readonly:
            raise ValueError("This ModuleTensorDescriptor instance is read-only")
        self._ptr[0].w = val

    @property
    def data_format(self):
        """int: """
        return self._ptr[0].dataFormat

    @data_format.setter
    def data_format(self, val):
        if self._readonly:
            raise ValueError("This ModuleTensorDescriptor instance is read-only")
        self._ptr[0].dataFormat = val

    @property
    def data_type(self):
        """int: """
        return self._ptr[0].dataType

    @data_type.setter
    def data_type(self, val):
        if self._readonly:
            raise ValueError("This ModuleTensorDescriptor instance is read-only")
        self._ptr[0].dataType = val

    @property
    def data_category(self):
        """int: """
        return self._ptr[0].dataCategory

    @data_category.setter
    def data_category(self, val):
        if self._readonly:
            raise ValueError("This ModuleTensorDescriptor instance is read-only")
        self._ptr[0].dataCategory = val

    @property
    def pixel_format(self):
        """int: """
        return self._ptr[0].pixelFormat

    @pixel_format.setter
    def pixel_format(self, val):
        if self._readonly:
            raise ValueError("This ModuleTensorDescriptor instance is read-only")
        self._ptr[0].pixelFormat = val

    @property
    def pixel_mapping(self):
        """int: """
        return self._ptr[0].pixelMapping

    @pixel_mapping.setter
    def pixel_mapping(self, val):
        if self._readonly:
            raise ValueError("This ModuleTensorDescriptor instance is read-only")
        self._ptr[0].pixelMapping = val

    @property
    def stride(self):
        """~_numpy.uint32: (array of length 8)."""
        cdef view.array arr = view.array(shape=(8,), itemsize=sizeof(uint32_t), format="I", mode="c", allocate_buffer=False)
        arr.data = <char *>(&(self._ptr[0].stride))
        return _numpy.asarray(arr)

    @stride.setter
    def stride(self, val):
        if self._readonly:
            raise ValueError("This ModuleTensorDescriptor instance is read-only")
        if len(val) != 8:
            raise ValueError(f"Expected length { 8 } for field stride, got {len(val)}")
        cdef view.array arr = view.array(shape=(8,), itemsize=sizeof(uint32_t), format="I", mode="c")
        arr[:] = _numpy.asarray(val, dtype=_numpy.uint32)
        memcpy(<void *>(&(self._ptr[0].stride)), <void *>(arr.data), sizeof(uint32_t) * len(val))

    @staticmethod
    def from_buffer(buffer):
        """Create an ModuleTensorDescriptor instance with the memory from the given buffer."""
        return __from_buffer(buffer, sizeof(cudlaModuleTensorDescriptor), ModuleTensorDescriptor)

    @staticmethod
    def from_data(data):
        """Create an ModuleTensorDescriptor instance wrapping the given NumPy array.

        Args:
            data (_numpy.ndarray): a single-element array of dtype `module_tensor_descriptor_dtype` holding the data.
        """
        return __from_data(data, "module_tensor_descriptor_dtype", module_tensor_descriptor_dtype, ModuleTensorDescriptor)

    @staticmethod
    def from_ptr(intptr_t ptr, bint readonly=False, object owner=None):
        """Create an ModuleTensorDescriptor instance wrapping the given pointer.

        Args:
            ptr (intptr_t): pointer address as Python :class:`int` to the data.
            owner (object): The Python object that owns the pointer. If not provided, data will be copied.
            readonly (bool): whether the data is read-only (to the user). default is `False`.
        """
        if ptr == 0:
            raise ValueError("ptr must not be null (0)")
        cdef ModuleTensorDescriptor obj = ModuleTensorDescriptor.__new__(ModuleTensorDescriptor)
        if owner is None:
            obj._ptr = <cudlaModuleTensorDescriptor *>malloc(sizeof(cudlaModuleTensorDescriptor))
            if obj._ptr == NULL:
                raise MemoryError("Error allocating ModuleTensorDescriptor")
            memcpy(<void*>(obj._ptr), <void*>ptr, sizeof(cudlaModuleTensorDescriptor))
            obj._owner = None
            obj._owned = True
        else:
            obj._ptr = <cudlaModuleTensorDescriptor *>ptr
            obj._owner = owner
            obj._owned = False
        obj._readonly = readonly
        return obj


cdef _get_fence_dtype_offsets():
    cdef CudlaFence pod = CudlaFence()
    return _numpy.dtype({
        'names': ['fence', 'type'],
        'formats': [_numpy.intp, _numpy.int32],
        'offsets': [
            (<intptr_t>&(pod.fence)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.type)) - (<intptr_t>&pod),
        ],
        'itemsize': sizeof(CudlaFence),
    })

fence_dtype = _get_fence_dtype_offsets()

cdef class Fence:
    """Empty-initialize an instance of `CudlaFence`.


    .. seealso:: `CudlaFence`
    """
    cdef:
        CudlaFence *_ptr
        object _owner
        bint _owned
        bint _readonly

    def __init__(self):
        self._ptr = <CudlaFence *>calloc(1, sizeof(CudlaFence))
        if self._ptr == NULL:
            raise MemoryError("Error allocating Fence")
        self._owner = None
        self._owned = True
        self._readonly = False

    def __dealloc__(self):
        cdef CudlaFence *ptr
        if self._owned and self._ptr != NULL:
            ptr = self._ptr
            self._ptr = NULL
            free(ptr)

    def __repr__(self):
        return f"<{__name__}.Fence object at {hex(id(self))}>"

    @property
    def ptr(self):
        """Get the pointer address to the data as Python :class:`int`."""
        return <intptr_t>(self._ptr)

    cdef intptr_t _get_ptr(self):
        return <intptr_t>(self._ptr)

    def __int__(self):
        return <intptr_t>(self._ptr)

    def __eq__(self, other):
        cdef Fence other_
        if not isinstance(other, Fence):
            return False
        other_ = other
        return (memcmp(<void *><intptr_t>(self._ptr), <void *><intptr_t>(other_._ptr), sizeof(CudlaFence)) == 0)

    def __getbuffer__(self, Py_buffer *buffer, int flags):
        __getbuffer(self, buffer, <void *>self._ptr, sizeof(CudlaFence), self._readonly)

    def __releasebuffer__(self, Py_buffer *buffer):
        pass

    def __setitem__(self, key, val):
        if key == 0 and isinstance(val, _numpy.ndarray):
            self._ptr = <CudlaFence *>malloc(sizeof(CudlaFence))
            if self._ptr == NULL:
                raise MemoryError("Error allocating Fence")
            memcpy(<void*>self._ptr, <void*><intptr_t>val.ctypes.data, sizeof(CudlaFence))
            self._owner = None
            self._owned = True
            self._readonly = not val.flags.writeable
        else:
            setattr(self, key, val)

    @property
    def fence(self):
        """int: """
        return <intptr_t>(self._ptr[0].fence)

    @fence.setter
    def fence(self, val):
        if self._readonly:
            raise ValueError("This Fence instance is read-only")
        self._ptr[0].fence = <void *><intptr_t>val

    @property
    def type(self):
        """int: """
        return <int>(self._ptr[0].type)

    @type.setter
    def type(self, val):
        if self._readonly:
            raise ValueError("This Fence instance is read-only")
        self._ptr[0].type = <cudlaFenceType><int>val

    @staticmethod
    def from_buffer(buffer):
        """Create an Fence instance with the memory from the given buffer."""
        return __from_buffer(buffer, sizeof(CudlaFence), Fence)

    @staticmethod
    def from_data(data):
        """Create an Fence instance wrapping the given NumPy array.

        Args:
            data (_numpy.ndarray): a single-element array of dtype `fence_dtype` holding the data.
        """
        return __from_data(data, "fence_dtype", fence_dtype, Fence)

    @staticmethod
    def from_ptr(intptr_t ptr, bint readonly=False, object owner=None):
        """Create an Fence instance wrapping the given pointer.

        Args:
            ptr (intptr_t): pointer address as Python :class:`int` to the data.
            owner (object): The Python object that owns the pointer. If not provided, data will be copied.
            readonly (bool): whether the data is read-only (to the user). default is `False`.
        """
        if ptr == 0:
            raise ValueError("ptr must not be null (0)")
        cdef Fence obj = Fence.__new__(Fence)
        if owner is None:
            obj._ptr = <CudlaFence *>malloc(sizeof(CudlaFence))
            if obj._ptr == NULL:
                raise MemoryError("Error allocating Fence")
            memcpy(<void*>(obj._ptr), <void*>ptr, sizeof(CudlaFence))
            obj._owner = None
            obj._owned = True
        else:
            obj._ptr = <CudlaFence *>ptr
            obj._owner = owner
            obj._owned = False
        obj._readonly = readonly
        return obj


dev_attribute_dtype = _numpy.dtype((
    _numpy.dtype((_numpy.void, sizeof(cudlaDevAttribute))),
    {
        "unified_addressing_supported": (_numpy.uint8, 0),
        "device_version": (_numpy.uint32, 0),
    }
    ))

cdef class DevAttribute:
    """Empty-initialize an instance of `cudlaDevAttribute`.


    .. seealso:: `cudlaDevAttribute`
    """
    cdef:
        cudlaDevAttribute *_ptr
        object _owner
        bint _owned
        bint _readonly

    def __init__(self):
        self._ptr = <cudlaDevAttribute *>calloc(1, sizeof(cudlaDevAttribute))
        if self._ptr == NULL:
            raise MemoryError("Error allocating DevAttribute")
        self._owner = None
        self._owned = True
        self._readonly = False

    def __dealloc__(self):
        cdef cudlaDevAttribute *ptr
        if self._owned and self._ptr != NULL:
            ptr = self._ptr
            self._ptr = NULL
            free(ptr)

    def __repr__(self):
        return f"<{__name__}.DevAttribute object at {hex(id(self))}>"

    @property
    def ptr(self):
        """Get the pointer address to the data as Python :class:`int`."""
        return <intptr_t>(self._ptr)

    cdef intptr_t _get_ptr(self):
        return <intptr_t>(self._ptr)

    def __int__(self):
        return <intptr_t>(self._ptr)

    def __eq__(self, other):
        cdef DevAttribute other_
        if not isinstance(other, DevAttribute):
            return False
        other_ = other
        return (memcmp(<void *><intptr_t>(self._ptr), <void *><intptr_t>(other_._ptr), sizeof(cudlaDevAttribute)) == 0)

    def __getbuffer__(self, Py_buffer *buffer, int flags):
        __getbuffer(self, buffer, <void *>self._ptr, sizeof(cudlaDevAttribute), self._readonly)

    def __releasebuffer__(self, Py_buffer *buffer):
        pass

    def __setitem__(self, key, val):
        if key == 0 and isinstance(val, _numpy.ndarray):
            self._ptr = <cudlaDevAttribute *>malloc(sizeof(cudlaDevAttribute))
            if self._ptr == NULL:
                raise MemoryError("Error allocating DevAttribute")
            memcpy(<void*>self._ptr, <void*><intptr_t>val.ctypes.data, sizeof(cudlaDevAttribute))
            self._owner = None
            self._owned = True
            self._readonly = not val.flags.writeable
        else:
            setattr(self, key, val)

    @property
    def unified_addressing_supported(self):
        """int: """
        return self._ptr[0].unifiedAddressingSupported

    @unified_addressing_supported.setter
    def unified_addressing_supported(self, val):
        if self._readonly:
            raise ValueError("This DevAttribute instance is read-only")
        self._ptr[0].unifiedAddressingSupported = val

    @property
    def device_version(self):
        """int: """
        return self._ptr[0].deviceVersion

    @device_version.setter
    def device_version(self, val):
        if self._readonly:
            raise ValueError("This DevAttribute instance is read-only")
        self._ptr[0].deviceVersion = val

    @staticmethod
    def from_buffer(buffer):
        """Create an DevAttribute instance with the memory from the given buffer."""
        return __from_buffer(buffer, sizeof(cudlaDevAttribute), DevAttribute)

    @staticmethod
    def from_data(data):
        """Create an DevAttribute instance wrapping the given NumPy array.

        Args:
            data (_numpy.ndarray): a single-element array of dtype `dev_attribute_dtype` holding the data.
        """
        return __from_data(data, "dev_attribute_dtype", dev_attribute_dtype, DevAttribute)

    @staticmethod
    def from_ptr(intptr_t ptr, bint readonly=False, object owner=None):
        """Create an DevAttribute instance wrapping the given pointer.

        Args:
            ptr (intptr_t): pointer address as Python :class:`int` to the data.
            owner (object): The Python object that owns the pointer. If not provided, data will be copied.
            readonly (bool): whether the data is read-only (to the user). default is `False`.
        """
        if ptr == 0:
            raise ValueError("ptr must not be null (0)")
        cdef DevAttribute obj = DevAttribute.__new__(DevAttribute)
        if owner is None:
            obj._ptr = <cudlaDevAttribute *>malloc(sizeof(cudlaDevAttribute))
            if obj._ptr == NULL:
                raise MemoryError("Error allocating DevAttribute")
            memcpy(<void*>(obj._ptr), <void*>ptr, sizeof(cudlaDevAttribute))
            obj._owner = None
            obj._owned = True
        else:
            obj._ptr = <cudlaDevAttribute *>ptr
            obj._owner = owner
            obj._owned = False
        obj._readonly = readonly
        return obj


module_attribute_dtype = _numpy.dtype((
    _numpy.dtype((_numpy.void, sizeof(cudlaModuleAttribute))),
    {
        "num_input_tensors": (_numpy.uint32, 0),
        "num_output_tensors": (_numpy.uint32, 0),
        "input_tensor_desc": (_numpy.intp, 0),
        "output_tensor_desc": (_numpy.intp, 0),
    }
    ))

cdef class ModuleAttribute:
    """Empty-initialize an instance of `cudlaModuleAttribute`.


    .. seealso:: `cudlaModuleAttribute`
    """
    cdef:
        cudlaModuleAttribute *_ptr
        object _owner
        bint _owned
        bint _readonly

    def __init__(self):
        self._ptr = <cudlaModuleAttribute *>calloc(1, sizeof(cudlaModuleAttribute))
        if self._ptr == NULL:
            raise MemoryError("Error allocating ModuleAttribute")
        self._owner = None
        self._owned = True
        self._readonly = False

    def __dealloc__(self):
        cdef cudlaModuleAttribute *ptr
        if self._owned and self._ptr != NULL:
            ptr = self._ptr
            self._ptr = NULL
            free(ptr)

    def __repr__(self):
        return f"<{__name__}.ModuleAttribute object at {hex(id(self))}>"

    @property
    def ptr(self):
        """Get the pointer address to the data as Python :class:`int`."""
        return <intptr_t>(self._ptr)

    cdef intptr_t _get_ptr(self):
        return <intptr_t>(self._ptr)

    def __int__(self):
        return <intptr_t>(self._ptr)

    def __eq__(self, other):
        cdef ModuleAttribute other_
        if not isinstance(other, ModuleAttribute):
            return False
        other_ = other
        return (memcmp(<void *><intptr_t>(self._ptr), <void *><intptr_t>(other_._ptr), sizeof(cudlaModuleAttribute)) == 0)

    def __getbuffer__(self, Py_buffer *buffer, int flags):
        __getbuffer(self, buffer, <void *>self._ptr, sizeof(cudlaModuleAttribute), self._readonly)

    def __releasebuffer__(self, Py_buffer *buffer):
        pass

    def __setitem__(self, key, val):
        if key == 0 and isinstance(val, _numpy.ndarray):
            self._ptr = <cudlaModuleAttribute *>malloc(sizeof(cudlaModuleAttribute))
            if self._ptr == NULL:
                raise MemoryError("Error allocating ModuleAttribute")
            memcpy(<void*>self._ptr, <void*><intptr_t>val.ctypes.data, sizeof(cudlaModuleAttribute))
            self._owner = None
            self._owned = True
            self._readonly = not val.flags.writeable
        else:
            setattr(self, key, val)

    @property
    def num_input_tensors(self):
        """int: """
        return self._ptr[0].numInputTensors

    @num_input_tensors.setter
    def num_input_tensors(self, val):
        if self._readonly:
            raise ValueError("This ModuleAttribute instance is read-only")
        self._ptr[0].numInputTensors = val

    @property
    def num_output_tensors(self):
        """int: """
        return self._ptr[0].numOutputTensors

    @num_output_tensors.setter
    def num_output_tensors(self, val):
        if self._readonly:
            raise ValueError("This ModuleAttribute instance is read-only")
        self._ptr[0].numOutputTensors = val

    @property
    def input_tensor_desc(self):
        """int: """
        return <intptr_t>(self._ptr[0].inputTensorDesc)

    @input_tensor_desc.setter
    def input_tensor_desc(self, val):
        if self._readonly:
            raise ValueError("This ModuleAttribute instance is read-only")
        self._ptr[0].inputTensorDesc = <cudlaModuleTensorDescriptor*><intptr_t>val

    @property
    def output_tensor_desc(self):
        """int: """
        return <intptr_t>(self._ptr[0].outputTensorDesc)

    @output_tensor_desc.setter
    def output_tensor_desc(self, val):
        if self._readonly:
            raise ValueError("This ModuleAttribute instance is read-only")
        self._ptr[0].outputTensorDesc = <cudlaModuleTensorDescriptor*><intptr_t>val

    @staticmethod
    def from_buffer(buffer):
        """Create an ModuleAttribute instance with the memory from the given buffer."""
        return __from_buffer(buffer, sizeof(cudlaModuleAttribute), ModuleAttribute)

    @staticmethod
    def from_data(data):
        """Create an ModuleAttribute instance wrapping the given NumPy array.

        Args:
            data (_numpy.ndarray): a single-element array of dtype `module_attribute_dtype` holding the data.
        """
        return __from_data(data, "module_attribute_dtype", module_attribute_dtype, ModuleAttribute)

    @staticmethod
    def from_ptr(intptr_t ptr, bint readonly=False, object owner=None):
        """Create an ModuleAttribute instance wrapping the given pointer.

        Args:
            ptr (intptr_t): pointer address as Python :class:`int` to the data.
            owner (object): The Python object that owns the pointer. If not provided, data will be copied.
            readonly (bool): whether the data is read-only (to the user). default is `False`.
        """
        if ptr == 0:
            raise ValueError("ptr must not be null (0)")
        cdef ModuleAttribute obj = ModuleAttribute.__new__(ModuleAttribute)
        if owner is None:
            obj._ptr = <cudlaModuleAttribute *>malloc(sizeof(cudlaModuleAttribute))
            if obj._ptr == NULL:
                raise MemoryError("Error allocating ModuleAttribute")
            memcpy(<void*>(obj._ptr), <void*>ptr, sizeof(cudlaModuleAttribute))
            obj._owner = None
            obj._owned = True
        else:
            obj._ptr = <cudlaModuleAttribute *>ptr
            obj._owner = owner
            obj._owned = False
        obj._readonly = readonly
        return obj


cdef _get_wait_events_dtype_offsets():
    cdef cudlaWaitEvents pod = cudlaWaitEvents()
    return _numpy.dtype({
        'names': ['pre_fences', 'num_events'],
        'formats': [_numpy.intp, _numpy.uint32],
        'offsets': [
            (<intptr_t>&(pod.preFences)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.numEvents)) - (<intptr_t>&pod),
        ],
        'itemsize': sizeof(cudlaWaitEvents),
    })

wait_events_dtype = _get_wait_events_dtype_offsets()

cdef class WaitEvents:
    """Empty-initialize an instance of `cudlaWaitEvents`.


    .. seealso:: `cudlaWaitEvents`
    """
    cdef:
        cudlaWaitEvents *_ptr
        object _owner
        bint _owned
        bint _readonly
        dict _refs

    def __init__(self):
        self._ptr = <cudlaWaitEvents *>calloc(1, sizeof(cudlaWaitEvents))
        if self._ptr == NULL:
            raise MemoryError("Error allocating WaitEvents")
        self._owner = None
        self._owned = True
        self._readonly = False
        self._refs = {}

    def __dealloc__(self):
        cdef cudlaWaitEvents *ptr
        if self._owned and self._ptr != NULL:
            ptr = self._ptr
            self._ptr = NULL
            free(ptr)

    def __repr__(self):
        return f"<{__name__}.WaitEvents object at {hex(id(self))}>"

    @property
    def ptr(self):
        """Get the pointer address to the data as Python :class:`int`."""
        return <intptr_t>(self._ptr)

    cdef intptr_t _get_ptr(self):
        return <intptr_t>(self._ptr)

    def __int__(self):
        return <intptr_t>(self._ptr)

    def __eq__(self, other):
        cdef WaitEvents other_
        if not isinstance(other, WaitEvents):
            return False
        other_ = other
        return (memcmp(<void *><intptr_t>(self._ptr), <void *><intptr_t>(other_._ptr), sizeof(cudlaWaitEvents)) == 0)

    def __getbuffer__(self, Py_buffer *buffer, int flags):
        __getbuffer(self, buffer, <void *>self._ptr, sizeof(cudlaWaitEvents), self._readonly)

    def __releasebuffer__(self, Py_buffer *buffer):
        pass

    def __setitem__(self, key, val):
        if key == 0 and isinstance(val, _numpy.ndarray):
            self._ptr = <cudlaWaitEvents *>malloc(sizeof(cudlaWaitEvents))
            if self._ptr == NULL:
                raise MemoryError("Error allocating WaitEvents")
            memcpy(<void*>self._ptr, <void*><intptr_t>val.ctypes.data, sizeof(cudlaWaitEvents))
            self._owner = None
            self._owned = True
            self._readonly = not val.flags.writeable
        else:
            setattr(self, key, val)

    @property
    def pre_fences(self):
        """int: """
        if self._ptr[0].preFences == NULL or self._ptr[0].numEvents == 0:
            return []
        return Fence.from_ptr(<intptr_t>(self._ptr[0].preFences), self._ptr[0].numEvents)

    @pre_fences.setter
    def pre_fences(self, val):
        if self._readonly:
            raise ValueError("This WaitEvents instance is read-only")
        cdef Fence arr = val
        self._ptr[0].preFences = <CudlaFence*><intptr_t>(arr._get_ptr())
        self._ptr[0].numEvents = len(arr)
        self._refs["pre_fences"] = arr

    @staticmethod
    def from_buffer(buffer):
        """Create an WaitEvents instance with the memory from the given buffer."""
        return __from_buffer(buffer, sizeof(cudlaWaitEvents), WaitEvents)

    @staticmethod
    def from_data(data):
        """Create an WaitEvents instance wrapping the given NumPy array.

        Args:
            data (_numpy.ndarray): a single-element array of dtype `wait_events_dtype` holding the data.
        """
        return __from_data(data, "wait_events_dtype", wait_events_dtype, WaitEvents)

    @staticmethod
    def from_ptr(intptr_t ptr, bint readonly=False, object owner=None):
        """Create an WaitEvents instance wrapping the given pointer.

        Args:
            ptr (intptr_t): pointer address as Python :class:`int` to the data.
            owner (object): The Python object that owns the pointer. If not provided, data will be copied.
            readonly (bool): whether the data is read-only (to the user). default is `False`.
        """
        if ptr == 0:
            raise ValueError("ptr must not be null (0)")
        cdef WaitEvents obj = WaitEvents.__new__(WaitEvents)
        if owner is None:
            obj._ptr = <cudlaWaitEvents *>malloc(sizeof(cudlaWaitEvents))
            if obj._ptr == NULL:
                raise MemoryError("Error allocating WaitEvents")
            memcpy(<void*>(obj._ptr), <void*>ptr, sizeof(cudlaWaitEvents))
            obj._owner = None
            obj._owned = True
        else:
            obj._ptr = <cudlaWaitEvents *>ptr
            obj._owner = owner
            obj._owned = False
        obj._readonly = readonly
        obj._refs = {}
        return obj


cdef _get_signal_events_dtype_offsets():
    cdef cudlaSignalEvents pod = cudlaSignalEvents()
    return _numpy.dtype({
        'names': ['dev_ptrs', 'eof_fences', 'num_events'],
        'formats': [_numpy.intp, _numpy.intp, _numpy.uint32],
        'offsets': [
            (<intptr_t>&(pod.devPtrs)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.eofFences)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.numEvents)) - (<intptr_t>&pod),
        ],
        'itemsize': sizeof(cudlaSignalEvents),
    })

signal_events_dtype = _get_signal_events_dtype_offsets()

cdef class SignalEvents:
    """Empty-initialize an instance of `cudlaSignalEvents`.


    .. seealso:: `cudlaSignalEvents`
    """
    cdef:
        cudlaSignalEvents *_ptr
        object _owner
        bint _owned
        bint _readonly
        dict _refs

    def __init__(self):
        self._ptr = <cudlaSignalEvents *>calloc(1, sizeof(cudlaSignalEvents))
        if self._ptr == NULL:
            raise MemoryError("Error allocating SignalEvents")
        self._owner = None
        self._owned = True
        self._readonly = False
        self._refs = {}

    def __dealloc__(self):
        cdef cudlaSignalEvents *ptr
        if self._owned and self._ptr != NULL:
            ptr = self._ptr
            self._ptr = NULL
            free(ptr)

    def __repr__(self):
        return f"<{__name__}.SignalEvents object at {hex(id(self))}>"

    @property
    def ptr(self):
        """Get the pointer address to the data as Python :class:`int`."""
        return <intptr_t>(self._ptr)

    cdef intptr_t _get_ptr(self):
        return <intptr_t>(self._ptr)

    def __int__(self):
        return <intptr_t>(self._ptr)

    def __eq__(self, other):
        cdef SignalEvents other_
        if not isinstance(other, SignalEvents):
            return False
        other_ = other
        return (memcmp(<void *><intptr_t>(self._ptr), <void *><intptr_t>(other_._ptr), sizeof(cudlaSignalEvents)) == 0)

    def __getbuffer__(self, Py_buffer *buffer, int flags):
        __getbuffer(self, buffer, <void *>self._ptr, sizeof(cudlaSignalEvents), self._readonly)

    def __releasebuffer__(self, Py_buffer *buffer):
        pass

    def __setitem__(self, key, val):
        if key == 0 and isinstance(val, _numpy.ndarray):
            self._ptr = <cudlaSignalEvents *>malloc(sizeof(cudlaSignalEvents))
            if self._ptr == NULL:
                raise MemoryError("Error allocating SignalEvents")
            memcpy(<void*>self._ptr, <void*><intptr_t>val.ctypes.data, sizeof(cudlaSignalEvents))
            self._owner = None
            self._owned = True
            self._readonly = not val.flags.writeable
        else:
            setattr(self, key, val)

    @property
    def dev_ptrs(self):
        """int: """
        if self._ptr[0].devPtrs == NULL or self._ptr[0].numEvents == 0:
            return view.array(shape=(1,), itemsize=sizeof(intptr_t), format="q", mode="c")[:0]
        cdef view.array arr = view.array(shape=(self._ptr[0].numEvents,), itemsize=sizeof(intptr_t), format="q", mode="c", allocate_buffer=False)
        arr.data = <char *>(self._ptr[0].devPtrs)
        return arr

    @dev_ptrs.setter
    def dev_ptrs(self, val):
        if self._readonly:
            raise ValueError("This SignalEvents instance is read-only")
        cdef Py_ssize_t _n = len(val)
        self._ptr[0].numEvents = _n
        if _n == 0:
            return
        cdef view.array arr = view.array(shape=(_n,), itemsize=sizeof(intptr_t), format="q", mode="c")
        cdef intptr_t[:] mv = arr
        cdef Py_ssize_t i
        for i in range(_n):
            mv[i] = val[i]
        self._ptr[0].devPtrs = <uint64_t**><intptr_t>(arr.data)
        self._refs["dev_ptrs"] = arr

    @property
    def eof_fences(self):
        """int: """
        if self._ptr[0].eofFences == NULL or self._ptr[0].numEvents == 0:
            return []
        return Fence.from_ptr(<intptr_t>(self._ptr[0].eofFences), self._ptr[0].numEvents)

    @eof_fences.setter
    def eof_fences(self, val):
        if self._readonly:
            raise ValueError("This SignalEvents instance is read-only")
        cdef Fence arr = val
        self._ptr[0].eofFences = <CudlaFence*><intptr_t>(arr._get_ptr())
        self._ptr[0].numEvents = len(arr)
        self._refs["eof_fences"] = arr

    @staticmethod
    def from_buffer(buffer):
        """Create an SignalEvents instance with the memory from the given buffer."""
        return __from_buffer(buffer, sizeof(cudlaSignalEvents), SignalEvents)

    @staticmethod
    def from_data(data):
        """Create an SignalEvents instance wrapping the given NumPy array.

        Args:
            data (_numpy.ndarray): a single-element array of dtype `signal_events_dtype` holding the data.
        """
        return __from_data(data, "signal_events_dtype", signal_events_dtype, SignalEvents)

    @staticmethod
    def from_ptr(intptr_t ptr, bint readonly=False, object owner=None):
        """Create an SignalEvents instance wrapping the given pointer.

        Args:
            ptr (intptr_t): pointer address as Python :class:`int` to the data.
            owner (object): The Python object that owns the pointer. If not provided, data will be copied.
            readonly (bool): whether the data is read-only (to the user). default is `False`.
        """
        if ptr == 0:
            raise ValueError("ptr must not be null (0)")
        cdef SignalEvents obj = SignalEvents.__new__(SignalEvents)
        if owner is None:
            obj._ptr = <cudlaSignalEvents *>malloc(sizeof(cudlaSignalEvents))
            if obj._ptr == NULL:
                raise MemoryError("Error allocating SignalEvents")
            memcpy(<void*>(obj._ptr), <void*>ptr, sizeof(cudlaSignalEvents))
            obj._owner = None
            obj._owned = True
        else:
            obj._ptr = <cudlaSignalEvents *>ptr
            obj._owner = owner
            obj._owned = False
        obj._readonly = readonly
        obj._refs = {}
        return obj


cdef _get_task_dtype_offsets():
    cdef cudlaTask pod = cudlaTask()
    return _numpy.dtype({
        'names': ['module_handle', 'output_tensor', 'num_output_tensors', 'num_input_tensors', 'input_tensor', 'wait_events', 'signal_events'],
        'formats': [_numpy.intp, _numpy.intp, _numpy.uint32, _numpy.uint32, _numpy.intp, _numpy.intp, _numpy.intp],
        'offsets': [
            (<intptr_t>&(pod.moduleHandle)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.outputTensor)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.numOutputTensors)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.numInputTensors)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.inputTensor)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.waitEvents)) - (<intptr_t>&pod),
            (<intptr_t>&(pod.signalEvents)) - (<intptr_t>&pod),
        ],
        'itemsize': sizeof(cudlaTask),
    })

task_dtype = _get_task_dtype_offsets()

cdef class Task:
    """Empty-initialize an instance of `cudlaTask`.


    .. seealso:: `cudlaTask`
    """
    cdef:
        cudlaTask *_ptr
        object _owner
        bint _owned
        bint _readonly
        dict _refs

    def __init__(self):
        self._ptr = <cudlaTask *>calloc(1, sizeof(cudlaTask))
        if self._ptr == NULL:
            raise MemoryError("Error allocating Task")
        self._owner = None
        self._owned = True
        self._readonly = False
        self._refs = {}

    def __dealloc__(self):
        cdef cudlaTask *ptr
        if self._owned and self._ptr != NULL:
            ptr = self._ptr
            self._ptr = NULL
            free(ptr)

    def __repr__(self):
        return f"<{__name__}.Task object at {hex(id(self))}>"

    @property
    def ptr(self):
        """Get the pointer address to the data as Python :class:`int`."""
        return <intptr_t>(self._ptr)

    cdef intptr_t _get_ptr(self):
        return <intptr_t>(self._ptr)

    def __int__(self):
        return <intptr_t>(self._ptr)

    def __eq__(self, other):
        cdef Task other_
        if not isinstance(other, Task):
            return False
        other_ = other
        return (memcmp(<void *><intptr_t>(self._ptr), <void *><intptr_t>(other_._ptr), sizeof(cudlaTask)) == 0)

    def __getbuffer__(self, Py_buffer *buffer, int flags):
        __getbuffer(self, buffer, <void *>self._ptr, sizeof(cudlaTask), self._readonly)

    def __releasebuffer__(self, Py_buffer *buffer):
        pass

    def __setitem__(self, key, val):
        if key == 0 and isinstance(val, _numpy.ndarray):
            self._ptr = <cudlaTask *>malloc(sizeof(cudlaTask))
            if self._ptr == NULL:
                raise MemoryError("Error allocating Task")
            memcpy(<void*>self._ptr, <void*><intptr_t>val.ctypes.data, sizeof(cudlaTask))
            self._owner = None
            self._owned = True
            self._readonly = not val.flags.writeable
        else:
            setattr(self, key, val)

    @property
    def module_handle(self):
        """int: """
        return <intptr_t>(self._ptr[0].moduleHandle)

    @module_handle.setter
    def module_handle(self, val):
        if self._readonly:
            raise ValueError("This Task instance is read-only")
        self._ptr[0].moduleHandle = <cudlaModule><intptr_t>val

    @property
    def output_tensor(self):
        """int: """
        if self._ptr[0].outputTensor == NULL or self._ptr[0].numOutputTensors == 0:
            return view.array(shape=(1,), itemsize=sizeof(intptr_t), format="q", mode="c")[:0]
        cdef view.array arr = view.array(shape=(self._ptr[0].numOutputTensors,), itemsize=sizeof(intptr_t), format="q", mode="c", allocate_buffer=False)
        arr.data = <char *>(self._ptr[0].outputTensor)
        return arr

    @output_tensor.setter
    def output_tensor(self, val):
        if self._readonly:
            raise ValueError("This Task instance is read-only")
        cdef Py_ssize_t _n = len(val)
        self._ptr[0].numOutputTensors = _n
        if _n == 0:
            return
        cdef view.array arr = view.array(shape=(_n,), itemsize=sizeof(intptr_t), format="q", mode="c")
        cdef intptr_t[:] mv = arr
        cdef Py_ssize_t i
        for i in range(_n):
            mv[i] = val[i]
        self._ptr[0].outputTensor = <uint64_t**><intptr_t>(arr.data)
        self._refs["output_tensor"] = arr

    @property
    def input_tensor(self):
        """int: """
        if self._ptr[0].inputTensor == NULL or self._ptr[0].numInputTensors == 0:
            return view.array(shape=(1,), itemsize=sizeof(intptr_t), format="q", mode="c")[:0]
        cdef view.array arr = view.array(shape=(self._ptr[0].numInputTensors,), itemsize=sizeof(intptr_t), format="q", mode="c", allocate_buffer=False)
        arr.data = <char *>(self._ptr[0].inputTensor)
        return arr

    @input_tensor.setter
    def input_tensor(self, val):
        if self._readonly:
            raise ValueError("This Task instance is read-only")
        cdef Py_ssize_t _n = len(val)
        self._ptr[0].numInputTensors = _n
        if _n == 0:
            return
        cdef view.array arr = view.array(shape=(_n,), itemsize=sizeof(intptr_t), format="q", mode="c")
        cdef intptr_t[:] mv = arr
        cdef Py_ssize_t i
        for i in range(_n):
            mv[i] = val[i]
        self._ptr[0].inputTensor = <uint64_t**><intptr_t>(arr.data)
        self._refs["input_tensor"] = arr

    @property
    def wait_events(self):
        """int: """
        return <intptr_t>(self._ptr[0].waitEvents)

    @wait_events.setter
    def wait_events(self, val):
        if self._readonly:
            raise ValueError("This Task instance is read-only")
        self._ptr[0].waitEvents = <cudlaWaitEvents*><intptr_t>val

    @property
    def signal_events(self):
        """int: """
        return <intptr_t>(self._ptr[0].signalEvents)

    @signal_events.setter
    def signal_events(self, val):
        if self._readonly:
            raise ValueError("This Task instance is read-only")
        self._ptr[0].signalEvents = <cudlaSignalEvents*><intptr_t>val

    @staticmethod
    def from_buffer(buffer):
        """Create an Task instance with the memory from the given buffer."""
        return __from_buffer(buffer, sizeof(cudlaTask), Task)

    @staticmethod
    def from_data(data):
        """Create an Task instance wrapping the given NumPy array.

        Args:
            data (_numpy.ndarray): a single-element array of dtype `task_dtype` holding the data.
        """
        return __from_data(data, "task_dtype", task_dtype, Task)

    @staticmethod
    def from_ptr(intptr_t ptr, bint readonly=False, object owner=None):
        """Create an Task instance wrapping the given pointer.

        Args:
            ptr (intptr_t): pointer address as Python :class:`int` to the data.
            owner (object): The Python object that owns the pointer. If not provided, data will be copied.
            readonly (bool): whether the data is read-only (to the user). default is `False`.
        """
        if ptr == 0:
            raise ValueError("ptr must not be null (0)")
        cdef Task obj = Task.__new__(Task)
        if owner is None:
            obj._ptr = <cudlaTask *>malloc(sizeof(cudlaTask))
            if obj._ptr == NULL:
                raise MemoryError("Error allocating Task")
            memcpy(<void*>(obj._ptr), <void*>ptr, sizeof(cudlaTask))
            obj._owner = None
            obj._owned = True
        else:
            obj._ptr = <cudlaTask *>ptr
            obj._owner = owner
            obj._owned = False
        obj._readonly = readonly
        obj._refs = {}
        return obj


###############################################################################
# Enum
###############################################################################

class Status(_IntEnum):
    """
    See `cudlaStatus`.
    """
    Success = cudlaSuccess
    ErrorInvalidParam = cudlaErrorInvalidParam
    ErrorOutOfResources = cudlaErrorOutOfResources
    ErrorCreationFailed = cudlaErrorCreationFailed
    ErrorInvalidAddress = cudlaErrorInvalidAddress
    ErrorOs = cudlaErrorOs
    ErrorCuda = cudlaErrorCuda
    ErrorUmd = cudlaErrorUmd
    ErrorInvalidDevice = cudlaErrorInvalidDevice
    ErrorInvalidAttribute = cudlaErrorInvalidAttribute
    ErrorIncompatibleDlaSWVersion = cudlaErrorIncompatibleDlaSWVersion
    ErrorMemoryRegistered = cudlaErrorMemoryRegistered
    ErrorInvalidModule = cudlaErrorInvalidModule
    ErrorUnsupportedOperation = cudlaErrorUnsupportedOperation
    ErrorNvSci = cudlaErrorNvSci
    ErrorDlaErrInvalidInput = cudlaErrorDlaErrInvalidInput
    ErrorDlaErrInvalidPreAction = cudlaErrorDlaErrInvalidPreAction
    ErrorDlaErrNoMem = cudlaErrorDlaErrNoMem
    ErrorDlaErrProcessorBusy = cudlaErrorDlaErrProcessorBusy
    ErrorDlaErrTaskStatusMismatch = cudlaErrorDlaErrTaskStatusMismatch
    ErrorDlaErrEngineTimeout = cudlaErrorDlaErrEngineTimeout
    ErrorDlaErrDataMismatch = cudlaErrorDlaErrDataMismatch
    ErrorUnknown = cudlaErrorUnknown

class Mode(_IntEnum):
    """
    See `cudlaMode`.
    """
    CUDA_DLA = CUDLA_CUDA_DLA
    STANDALONE = CUDLA_STANDALONE

class ModuleAttributeType(_IntEnum):
    """
    See `cudlaModuleAttributeType`.
    """
    NUM_INPUT_TENSORS = CUDLA_NUM_INPUT_TENSORS
    NUM_OUTPUT_TENSORS = CUDLA_NUM_OUTPUT_TENSORS
    INPUT_TENSOR_DESCRIPTORS = CUDLA_INPUT_TENSOR_DESCRIPTORS
    OUTPUT_TENSOR_DESCRIPTORS = CUDLA_OUTPUT_TENSOR_DESCRIPTORS
    NUM_OUTPUT_TASK_STATISTICS = CUDLA_NUM_OUTPUT_TASK_STATISTICS
    OUTPUT_TASK_STATISTICS_DESCRIPTORS = CUDLA_OUTPUT_TASK_STATISTICS_DESCRIPTORS

class FenceType(_IntEnum):
    """
    See `cudlaFenceType`.
    """
    NVSCISYNC_FENCE = CUDLA_NVSCISYNC_FENCE
    NVSCISYNC_FENCE_SOF = CUDLA_NVSCISYNC_FENCE_SOF

class ModuleLoadFlags(_IntEnum):
    """
    See `cudlaModuleLoadFlags`.
    """
    MODULE_DEFAULT = CUDLA_MODULE_DEFAULT
    MODULE_ENABLE_FAULT_DIAGNOSTICS = CUDLA_MODULE_ENABLE_FAULT_DIAGNOSTICS

class SubmissionFlags(_IntEnum):
    """
    See `cudlaSubmissionFlags`.
    """
    SUBMIT_NOOP = CUDLA_SUBMIT_NOOP
    SUBMIT_SKIP_LOCK_ACQUIRE = CUDLA_SUBMIT_SKIP_LOCK_ACQUIRE
    SUBMIT_DIAGNOSTICS_TASK = CUDLA_SUBMIT_DIAGNOSTICS_TASK

class AccessPermissionFlags(_IntEnum):
    """
    See `cudlaAccessPermissionFlags`.
    """
    READ_WRITE_PERM = CUDLA_READ_WRITE_PERM
    READ_ONLY_PERM = CUDLA_READ_ONLY_PERM
    TASK_STATISTICS = CUDLA_TASK_STATISTICS

class DevAttributeType(_IntEnum):
    """
    See `cudlaDevAttributeType`.
    """
    UNIFIED_ADDRESSING = CUDLA_UNIFIED_ADDRESSING
    DEVICE_VERSION = CUDLA_DEVICE_VERSION


###############################################################################
# Error handling
###############################################################################

class CudlaError(Exception):

    def __init__(self, status):
        self.status = status
        s = Status(status)
        cdef str err = f"{s.name} ({s.value})"
        super(CudlaError, self).__init__(err)

    def __reduce__(self):
        return (type(self), (self.status,))


@cython.profile(False)
cpdef inline check_status(int status):
    if status != 0:
        raise CudlaError(status)


###############################################################################
# Wrapper functions
###############################################################################

cpdef uint64_t get_version() except? -1:
    cdef uint64_t version
    with nogil:
        __status__ = cudlaGetVersion(&version)
    check_status(__status__)
    return version


cpdef uint64_t device_get_count() except? -1:
    cdef uint64_t p_num_devices
    with nogil:
        __status__ = cudlaDeviceGetCount(&p_num_devices)
    check_status(__status__)
    return p_num_devices


cpdef intptr_t create_device(uint64_t device, uint32_t flags) except *:
    cdef DevHandle dev_handle
    if flags == CUDLA_STANDALONE:
        raise CudlaError(cudlaErrorUnsupportedOperation)
    with nogil:
        __status__ = cudlaCreateDevice(<const uint64_t>device, &dev_handle, <const uint32_t>flags)
    check_status(__status__)
    return <intptr_t>dev_handle


cpdef intptr_t mem_register(intptr_t dev_handle, intptr_t ptr, size_t size, uint32_t flags) except *:
    cdef uint64_t* dev_ptr
    with nogil:
        __status__ = cudlaMemRegister(<const DevHandle>dev_handle, <const uint64_t* const>ptr, <const size_t>size, &dev_ptr, <const uint32_t>flags)
    check_status(__status__)
    return <intptr_t>dev_ptr


cpdef intptr_t module_load_from_memory(intptr_t dev_handle, p_module, size_t module_size, uint32_t flags) except *:
    cdef void* _p_module_ = get_buffer_pointer(p_module, module_size, readonly=True)
    cdef Module h_module
    with nogil:
        __status__ = cudlaModuleLoadFromMemory(<const DevHandle>dev_handle, <const uint8_t* const>_p_module_, <const size_t>module_size, &h_module, <const uint32_t>flags)
    check_status(__status__)
    return <intptr_t>h_module


cpdef module_unload(intptr_t h_module, uint32_t flags):
    with nogil:
        __status__ = cudlaModuleUnload(<const Module>h_module, <const uint32_t>flags)
    check_status(__status__)


cpdef submit_task(intptr_t dev_handle, intptr_t ptr_to_tasks, uint32_t num_tasks, intptr_t stream, uint32_t flags):
    with nogil:
        __status__ = cudlaSubmitTask(<const DevHandle>dev_handle, <const cudlaTask* const>ptr_to_tasks, <const uint32_t>num_tasks, <void* const>stream, <const uint32_t>flags)
    check_status(__status__)


cpdef object device_get_attribute(intptr_t dev_handle, int attrib) except *:
    cdef DevAttribute p_attribute_py = DevAttribute()
    cdef cudlaDevAttribute *p_attribute = <cudlaDevAttribute *><intptr_t>(p_attribute_py._get_ptr())
    with nogil:
        __status__ = cudlaDeviceGetAttribute(<const DevHandle>dev_handle, <const _DevAttributeType>attrib, p_attribute)
    check_status(__status__)
    return p_attribute_py


cpdef mem_unregister(intptr_t dev_handle, intptr_t dev_ptr):
    with nogil:
        __status__ = cudlaMemUnregister(<const DevHandle>dev_handle, <const uint64_t* const>dev_ptr)
    check_status(__status__)


cpdef int get_last_error(intptr_t dev_handle) except? 0:
    return <int>cudlaGetLastError(<const DevHandle>dev_handle)


cpdef destroy_device(intptr_t dev_handle):
    with nogil:
        __status__ = cudlaDestroyDevice(<const DevHandle>dev_handle)
    check_status(__status__)


cpdef set_task_timeout_in_ms(intptr_t dev_handle, uint32_t timeout):
    with nogil:
        __status__ = cudlaSetTaskTimeoutInMs(<const DevHandle>dev_handle, <const uint32_t>timeout)
    check_status(__status__)


cpdef module_get_attributes(intptr_t h_module, int attr_type) except *:
    """Query module attributes, interpreting the cudlaModuleAttribute union
    based on the requested attribute type.

    For count attributes (NUM_INPUT_TENSORS, NUM_OUTPUT_TENSORS,
    NUM_OUTPUT_TASK_STATISTICS), returns an int.

    For descriptor attributes (INPUT_TENSOR_DESCRIPTORS,
    OUTPUT_TENSOR_DESCRIPTORS, OUTPUT_TASK_STATISTICS_DESCRIPTORS),
    returns a list of ModuleTensorDescriptor objects.
    """
    cdef int _attr_type = attr_type
    cdef cudlaModuleAttribute count_attr
    cdef cudlaModuleAttribute num_attr
    cdef cudlaModuleAttribute desc_attr
    cdef uint32_t count
    cdef cudlaModuleTensorDescriptor* desc_buf
    cdef uint32_t i
    cdef int num_attr_type

    if _attr_type == CUDLA_NUM_INPUT_TENSORS or _attr_type == CUDLA_NUM_OUTPUT_TENSORS or _attr_type == CUDLA_NUM_OUTPUT_TASK_STATISTICS:
        with nogil:
            __status__ = cudlaModuleGetAttributes(<const Module>h_module, <const _ModuleAttributeType>_attr_type, &count_attr)
        check_status(__status__)
        return <int>(count_attr.numInputTensors)
    elif _attr_type == CUDLA_INPUT_TENSOR_DESCRIPTORS or _attr_type == CUDLA_OUTPUT_TENSOR_DESCRIPTORS or _attr_type == CUDLA_OUTPUT_TASK_STATISTICS_DESCRIPTORS:
        if _attr_type == CUDLA_INPUT_TENSOR_DESCRIPTORS:
            num_attr_type = CUDLA_NUM_INPUT_TENSORS
        elif _attr_type == CUDLA_OUTPUT_TENSOR_DESCRIPTORS:
            num_attr_type = CUDLA_NUM_OUTPUT_TENSORS
        else:
            num_attr_type = CUDLA_NUM_OUTPUT_TASK_STATISTICS
        with nogil:
            __status__ = cudlaModuleGetAttributes(<const Module>h_module, <const _ModuleAttributeType>num_attr_type, &num_attr)
        check_status(__status__)
        count = num_attr.numInputTensors
        desc_buf = <cudlaModuleTensorDescriptor*>malloc(count * sizeof(cudlaModuleTensorDescriptor))
        if desc_buf == NULL:
            raise MemoryError("Failed to allocate descriptor buffer")
        try:
            desc_attr.inputTensorDesc = desc_buf
            with nogil:
                __status__ = cudlaModuleGetAttributes(<const Module>h_module, <const _ModuleAttributeType>_attr_type, &desc_attr)
            check_status(__status__)
            result = []
            for i in range(count):
                result.append(ModuleTensorDescriptor.from_ptr(<intptr_t>&desc_buf[i], readonly=True))
            return result
        finally:
            free(desc_buf)
    else:
        raise ValueError(f"Unknown attribute type: {attr_type}")
