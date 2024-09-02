import abc

from cuda import cuda
from cuda.py._stream import default_stream
from cuda.py._utils import handle_return


class Buffer:

    # TODO: how about memory properties?
    # TODO: handle ownership (_mr could be None)
    __slots__ = ("_ptr", "_size", "_mr",)

    def __init__(self, ptr, size, mr=None):
        self._ptr = ptr
        self._size = size
        self._mr = mr

    def __del__(self):
        self.close(default_stream())

    def close(self, stream=None):
        if stream is None:
            stream = default_stream()
        if self._ptr and self._mr is not None:
            self._mr.deallocate(self._ptr, self._size, stream)
            self._ptr = 0

    @property
    def ptr(self):
        return self._ptr

    @property
    def size(self):
        return self._size


class MemoryResource(abc.ABC):

    # TODO: how about memory properties?
    __slots__ = ("_handle",)

    @abc.abstractmethod
    def __init__(self):
        ...

    @abc.abstractmethod
    def allocate(self, size, stream=None):
        ...

    @abc.abstractmethod
    def deallocate(self, ptr, size, stream=None):
        ...


class _DefaultAsyncMempool(MemoryResource):

    def __init__(self, dev_id):
        self._handle = handle_return(cuda.cuDeviceGetDefaultMemPool(dev_id))

    def allocate(self, size, stream=None):
        if stream is None:
            stream = default_stream()
        ptr = handle_return(cuda.cuMemAllocFromPoolAsync(size, self._handle, stream._handle))
        return Buffer(ptr, size, self)

    def deallocate(self, ptr, size, stream=None):
        if stream is None:
            stream = default_stream()
        handle_return(cuda.cuMemFreeAsync(ptr, stream._handle))
