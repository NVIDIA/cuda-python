from __future__ import annotations

import abc
from typing import Optional, Tuple, TypeVar

from cuda import cuda
from cuda.py._stream import default_stream
from cuda.py._utils import handle_return


PyCapsule = TypeVar("PyCapsule")


class Buffer:

    # TODO: handle ownership? (_mr could be None)
    __slots__ = ("_ptr", "_size", "_mr",)

    def __init__(self, ptr, size, mr: MemoryResource=None):
        self._ptr = ptr
        self._size = size
        self._mr = mr

    def __del__(self):
        self.close(default_stream())

    def close(self, stream=None):
        if self._ptr and self._mr is not None:
            if stream is None:
                stream = default_stream()
            self._mr.deallocate(self._ptr, self._size, stream)
            self._ptr = 0
            self._mr = None

    @property
    def handle(self):
        return self._ptr

    @property
    def size(self):
        return self._size

    @property
    def memory_resource(self) -> MemoryResource:
        # Return the memory resource from which this buffer was allocated.
        return self._mr

    @property
    def is_device_accessible(self) -> bool:
        # Check if this buffer can be accessed from GPUs.
        if self._mr is not None:
            return self._mr.is_device_accessible
        raise NotImplementedError

    @property
    def is_host_accessible(self) -> bool:
        # Check if this buffer can be accessed from CPUs.
        if self._mr is not None:
            return self._mr.is_host_accessible
        raise NotImplementedError

    def copy_to(self, dst: Buffer=None, stream=None) -> Buffer:
        # Copy from this buffer to the dst buffer asynchronously on the
        # given stream. The dst buffer is returned. If the dst is not provided,
        # allocate one from self.memory_resource. Raise an exception if the
        # stream is not provided.
        if stream is None:
            raise ValueError("stream must be provided")
        if dst is None:
            if self._mr is None:
                raise ValueError("a destination buffer must be provided")
            dst = self._mr.allocate(self._size, stream)
        if dst._size != self._size:
            raise ValueError("buffer sizes mismatch between src and dst")
        handle_return(
            cuda.cuMemcpyAsync(dst._ptr, self._ptr, self._size, stream._handle))
        return dst

    def copy_from(self, src: Buffer, stream=None):
        # Copy from the src buffer to this buffer asynchronously on the
        # given stream. Raise an exception if the stream is not provided. 
        if stream is None:
            raise ValueError("stream must be provided")
        if src._size != self._size:
            raise ValueError("buffer sizes mismatch between src and dst")
        handle_return(
            cuda.cuMemcpyAsync(self._ptr, src._ptr, self._size, stream._handle))

    def __dlpack__(self, *,
                   stream: int, 
                   max_version: Optional[Tuple[int, int]] = None, 
                   dl_device: Optional[Tuple[int, int]] = None, 
                   copy: Optional[bool] = None) -> PyCapsule:
        # Support for Python-level DLPack protocol.
        # Note that we do not support stream=None on purpose, see the 
        # discussion in GPUMemoryView below.
        raise NotImplementedError("TODO")

    def __dlpack_device__(self) -> Tuple[int, int]:
        # Supporting methond paired with __dlpack__.
        raise NotImplementedError("TODO")

    def __buffer__(self, flags: int, /) -> memoryview:
        # Support for Python-level buffer protocol as per PEP 688. 
        # This raises a BufferError unless: 
        #   1. Python is 3.12+
        #   2. This Buffer object is host accessible 
        raise NotImplementedError("TODO")

    def __release_buffer__(self, buffer: memoryview, /):
        # Supporting methond paired with __buffer__.
        raise NotImplementedError("TODO")


class MemoryResource(abc.ABC):

    __slots__ = ("_handle",)

    @abc.abstractmethod
    def __init__(self, *args, **kwargs):
        ...

    @abc.abstractmethod
    def allocate(self, size, stream=None) -> Buffer:
        ...

    @abc.abstractmethod
    def deallocate(self, ptr, size, stream=None):
        ...

    @property
    @abc.abstractmethod
    def is_device_accessible(self) -> bool:
        # Check if the buffers allocated from this MR can be accessed from
        # GPUs.
        ...

    @property
    @abc.abstractmethod
    def is_host_accessible(self) -> bool:
        # Check if the buffers allocated from this MR can be accessed from
        # CPUs.
        ...


class _DefaultAsyncMempool(MemoryResource):

    def __init__(self, dev_id):
        self._handle = handle_return(cuda.cuDeviceGetMemPool(dev_id))

    def allocate(self, size, stream=None) -> Buffer:
        if stream is None:
            stream = default_stream()
        ptr = handle_return(cuda.cuMemAllocFromPoolAsync(size, self._handle, stream._handle))
        return Buffer(ptr, size, self)

    def deallocate(self, ptr, size, stream=None):
        if stream is None:
            stream = default_stream()
        handle_return(cuda.cuMemFreeAsync(ptr, stream._handle))

    @property
    def is_device_accessible(self) -> bool:
        return True

    @property
    def is_host_accessible(self) -> bool:
        return False


class _DefaultPinnedMemorySource(MemoryResource):

    def __init__(self):
        # TODO: support flags from cuMemHostAlloc?
        self._handle = None

    def allocate(self, size, stream=None) -> Buffer:
        ptr = handle_return(cuda.cuMemHostAlloc(size))
        return Buffer(ptr, size, self)

    def deallocate(self, ptr, size, stream=None):
        handle_return(cuda.cuMemFreeHost(ptr))

    @property
    def is_device_accessible(self) -> bool:
        return True

    @property
    def is_host_accessible(self) -> bool:
        return True
