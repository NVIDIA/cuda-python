# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cuda.core._memory._buffer import DevicePointerT

from cuda.core._memory._buffer import Buffer, MemoryResource
from cuda.core._utils.cuda_utils import (
    _check_driver_error as raise_if_driver_error,
)
from cuda.core._utils.cuda_utils import (
    driver,
)

__all__ = ["LegacyPinnedMemoryResource", "_SynchronousMemoryResource"]


class LegacyPinnedMemoryResource(MemoryResource):
    """Create a pinned memory resource that uses legacy cuMemAllocHost/cudaMallocHost
    APIs.
    """

    # TODO: support creating this MR with flags that are later passed to cuMemHostAlloc?

    def allocate(self, size, stream=None) -> Buffer:
        """Allocate a buffer of the requested size.

        Parameters
        ----------
        size : int
            The size of the buffer to allocate, in bytes.
        stream : Stream, optional
            Currently ignored

        Returns
        -------
        Buffer
            The allocated buffer object, which is accessible on both host and device.
        """
        if stream is None:
            from cuda.core._stream import default_stream

            stream = default_stream()
        err, ptr = driver.cuMemAllocHost(size)
        raise_if_driver_error(err)
        return Buffer._init(ptr, size, self, stream)

    def deallocate(self, ptr: DevicePointerT, size, stream):
        """Deallocate a buffer previously allocated by this resource.

        Parameters
        ----------
        ptr : :obj:`~_memory.DevicePointerT`
            The pointer or handle to the buffer to deallocate.
        size : int
            The size of the buffer to deallocate, in bytes.
        stream : Stream
            The stream on which to perform the deallocation synchronously.
        """
        if stream is not None:
            stream.sync()
        (err,) = driver.cuMemFreeHost(ptr)
        raise_if_driver_error(err)

    @property
    def is_device_accessible(self) -> bool:
        """bool: this memory resource provides device-accessible buffers."""
        return True

    @property
    def is_host_accessible(self) -> bool:
        """bool: this memory resource provides host-accessible buffers."""
        return True

    @property
    def device_id(self) -> int:
        """This memory resource is not bound to any GPU."""
        raise RuntimeError("a pinned memory resource is not bound to any GPU")


class _SynchronousMemoryResource(MemoryResource):
    __slots__ = ("_device_id",)

    def __init__(self, device_id):
        from .._device import Device

        self._device_id = Device(device_id).device_id

    def allocate(self, size, stream=None) -> Buffer:
        if stream is None:
            from cuda.core._stream import default_stream

            stream = default_stream()
        err, ptr = driver.cuMemAlloc(size)
        raise_if_driver_error(err)
        return Buffer._init(ptr, size, self, stream)

    def deallocate(self, ptr, size, stream):
        if stream is not None:
            stream.sync()
        (err,) = driver.cuMemFree(ptr)
        raise_if_driver_error(err)

    @property
    def is_device_accessible(self) -> bool:
        return True

    @property
    def is_host_accessible(self) -> bool:
        return False

    @property
    def device_id(self) -> int:
        return self._device_id
