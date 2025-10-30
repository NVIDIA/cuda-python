# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

from cuda.core.experimental._utils.cuda_utils import driver


@dataclass
class ContextOptions:
    pass  # TODO


cdef class Context:

    cdef:
        readonly object _handle
        int _device_id

    def __init__(self, *args, **kwargs):
        raise RuntimeError("Context objects cannot be instantiated directly. Please use Device or Stream APIs.")

    @classmethod
    def _from_ctx(cls, handle: driver.CUcontext, int device_id):
        cdef Context ctx = Context.__new__(Context)
        ctx._handle = handle
        ctx._device_id = device_id
        return ctx

    def __eq__(self, other):
        """Check equality based on the underlying CUcontext handle address.

        Two Context objects are considered equal if they wrap the same
        underlying CUDA context.

        Parameters
        ----------
        other : object
            Another object to compare with.

        Returns
        -------
        bool or NotImplemented
            True if other is a Context wrapping the same handle, False if not equal,
            NotImplemented if other is not a Context.
        """
        if not isinstance(other, Context):
            return NotImplemented
        return int(self._handle) == int(other._handle)

    def __hash__(self) -> int:
        """Return hash based on the underlying CUcontext handle address.

        This enables Context objects to be used as dictionary keys and in sets.
        Two Context objects wrapping the same underlying CUDA context will hash
        to the same value and be considered equal.

        Returns
        -------
        int
            Hash value based on the context handle address.
        """
        return hash(int(self._handle))
