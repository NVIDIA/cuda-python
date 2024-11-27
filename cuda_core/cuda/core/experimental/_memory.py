# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

from __future__ import annotations

import abc
import weakref
from typing import Optional, Tuple, TypeVar

from cuda import cuda
from cuda.core.experimental._dlpack import DLDeviceType, make_py_capsule
from cuda.core.experimental._stream import default_stream
from cuda.core.experimental._utils import handle_return

PyCapsule = TypeVar("PyCapsule")


# TODO: define a memory property mixin class and make Buffer and
# MemoryResource both inherit from it


class Buffer:
    """Represent a handle to allocated memory.

    This generic object provides a unified representation for how
    different memory resources are to give access to their memory
    allocations.

    Support for data interchange mechanisms are provided by
    establishing both the DLPack and the Python-level buffer
    protocols.

    Parameters
    ----------
    ptr : Any
        Allocated buffer handle object
    size : Any
        Memory size of the buffer
    mr : :obj:`MemoryResource`, optional
        Memory resource associated with the buffer

    """

    class _MembersNeededForFinalize:
        __slots__ = ("ptr", "size", "mr")

        def __init__(self, buffer_obj, ptr, size, mr):
            self.ptr = ptr
            self.size = size
            self.mr = mr
            weakref.finalize(buffer_obj, self.close)

        def close(self, stream=None):
            if self.ptr and self.mr is not None:
                if stream is None:
                    stream = default_stream()
                self.mr.deallocate(self.ptr, self.size, stream)
                self.ptr = 0
                self.mr = None

    # TODO: handle ownership? (_mr could be None)
    __slots__ = ("__weakref__", "_mnff")

    def __init__(self, ptr, size, mr: MemoryResource = None):
        self._mnff = Buffer._MembersNeededForFinalize(self, ptr, size, mr)

    def close(self, stream=None):
        """Deallocate this buffer asynchronously on the given stream.

        This buffer is released back to their memory resource
        asynchronously on the given stream.

        Parameters
        ----------
        stream : Any, optional
            The stream object with a __cuda_stream__ protocol to
            use for asynchronous deallocation. Defaults to using
            the default stream.

        """
        self._mnff.close(stream)

    @property
    def handle(self):
        """Return the buffer handle object."""
        return self._mnff.ptr

    @property
    def size(self):
        """Return the memory size of this buffer."""
        return self._mnff.size

    @property
    def memory_resource(self) -> MemoryResource:
        """Return the memory resource associated with this buffer."""
        return self._mnff.mr

    @property
    def is_device_accessible(self) -> bool:
        """Return True if this buffer can be accessed by the GPU, otherwise False."""
        if self._mnff.mr is not None:
            return self._mnff.mr.is_device_accessible
        raise NotImplementedError

    @property
    def is_host_accessible(self) -> bool:
        """Return True if this buffer can be accessed by the CPU, otherwise False."""
        if self._mnff.mr is not None:
            return self._mnff.mr.is_host_accessible
        raise NotImplementedError

    @property
    def device_id(self) -> int:
        """Return the device ordinal of this buffer."""
        if self._mnff.mr is not None:
            return self._mnff.mr.device_id
        raise NotImplementedError

    def copy_to(self, dst: Buffer = None, *, stream) -> Buffer:
        """Copy from this buffer to the dst buffer asynchronously on the given stream.

        Copies the data from this buffer to the provided dst buffer.
        If the dst buffer is not provided, then a new buffer is first
        allocated using the associated memory resource before the copy.

        Parameters
        ----------
        dst : :obj:`Buffer`
            Source buffer to copy data from
        stream : Any
            Keyword argument specifying the stream for the
            asynchronous copy

        """
        if stream is None:
            raise ValueError("stream must be provided")
        if dst is None:
            if self._mnff.mr is None:
                raise ValueError("a destination buffer must be provided")
            dst = self._mnff.mr.allocate(self._mnff.size, stream)
        if dst._mnff.size != self._mnff.size:
            raise ValueError("buffer sizes mismatch between src and dst")
        handle_return(cuda.cuMemcpyAsync(dst._mnff.ptr, self._mnff.ptr, self._mnff.size, stream.handle))
        return dst

    def copy_from(self, src: Buffer, *, stream):
        """Copy from the src buffer to this buffer asynchronously on the given stream.

        Parameters
        ----------
        src : :obj:`Buffer`
            Source buffer to copy data from
        stream : Any
            Keyword argument specifying the stream for the
            asynchronous copy

        """
        if stream is None:
            raise ValueError("stream must be provided")
        if src._mnff.size != self._mnff.size:
            raise ValueError("buffer sizes mismatch between src and dst")
        handle_return(cuda.cuMemcpyAsync(self._mnff.ptr, src._mnff.ptr, self._mnff.size, stream.handle))

    def __dlpack__(
        self,
        *,
        stream: Optional[int] = None,
        max_version: Optional[Tuple[int, int]] = None,
        dl_device: Optional[Tuple[int, int]] = None,
        copy: Optional[bool] = None,
    ) -> PyCapsule:
        # Note: we ignore the stream argument entirely (as if it is -1).
        # It is the user's responsibility to maintain stream order.
        if dl_device is not None or copy is True:
            raise BufferError
        if max_version is None:
            versioned = False
        else:
            assert len(max_version) == 2
            versioned = max_version >= (1, 0)
        capsule = make_py_capsule(self, versioned)
        return capsule

    def __dlpack_device__(self) -> Tuple[int, int]:
        if self.is_device_accessible and not self.is_host_accessible:
            return (DLDeviceType.kDLCUDA, self.device_id)
        elif self.is_device_accessible and self.is_host_accessible:
            # TODO: this can also be kDLCUDAManaged, we need more fine-grained checks
            return (DLDeviceType.kDLCUDAHost, 0)
        elif not self.is_device_accessible and self.is_host_accessible:
            return (DLDeviceType.kDLCPU, 0)
        else:  # not self.is_device_accessible and not self.is_host_accessible
            raise BufferError("invalid buffer")

    def __buffer__(self, flags: int, /) -> memoryview:
        # Support for Python-level buffer protocol as per PEP 688.
        # This raises a BufferError unless:
        #   1. Python is 3.12+
        #   2. This Buffer object is host accessible
        raise NotImplementedError("TODO")

    def __release_buffer__(self, buffer: memoryview, /):
        # Supporting method paired with __buffer__.
        raise NotImplementedError("TODO")


class MemoryResource(abc.ABC):
    __slots__ = ("_handle",)

    @abc.abstractmethod
    def __init__(self, *args, **kwargs): ...

    @abc.abstractmethod
    def allocate(self, size, stream=None) -> Buffer: ...

    @abc.abstractmethod
    def deallocate(self, ptr, size, stream=None): ...

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

    @property
    @abc.abstractmethod
    def device_id(self) -> int:
        # Return the device ID if this MR is for single devices. Raise an
        # exception if it is not.
        ...


class _DefaultAsyncMempool(MemoryResource):
    __slots__ = ("_dev_id",)

    def __init__(self, dev_id):
        self._handle = handle_return(cuda.cuDeviceGetMemPool(dev_id))
        self._dev_id = dev_id

    def allocate(self, size, stream=None) -> Buffer:
        if stream is None:
            stream = default_stream()
        ptr = handle_return(cuda.cuMemAllocFromPoolAsync(size, self._handle, stream.handle))
        return Buffer(ptr, size, self)

    def deallocate(self, ptr, size, stream=None):
        if stream is None:
            stream = default_stream()
        handle_return(cuda.cuMemFreeAsync(ptr, stream.handle))

    @property
    def is_device_accessible(self) -> bool:
        return True

    @property
    def is_host_accessible(self) -> bool:
        return False

    @property
    def device_id(self) -> int:
        return self._dev_id


class _DefaultPinnedMemorySource(MemoryResource):
    def __init__(self):
        # TODO: support flags from cuMemHostAlloc?
        self._handle = None

    def allocate(self, size, stream=None) -> Buffer:
        ptr = handle_return(cuda.cuMemAllocHost(size))
        return Buffer(ptr, size, self)

    def deallocate(self, ptr, size, stream=None):
        handle_return(cuda.cuMemFreeHost(ptr))

    @property
    def is_device_accessible(self) -> bool:
        return True

    @property
    def is_host_accessible(self) -> bool:
        return True

    @property
    def device_id(self) -> int:
        raise RuntimeError("the pinned memory resource is not bound to any GPU")
