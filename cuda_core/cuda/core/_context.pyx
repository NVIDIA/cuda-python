# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

from cuda.core._resource_handles cimport (
    ContextHandle,
    as_intptr,
    as_py,
)


__all__ = ['Context', 'ContextOptions']


cdef class Context:
    """CUDA context wrapper.

    Context objects represent CUDA contexts and cannot be instantiated directly.
    Use Device or Stream APIs to obtain context objects.
    """

    def __init__(self, *args, **kwargs):
        raise RuntimeError("Context objects cannot be instantiated directly. Please use Device or Stream APIs.")

    @staticmethod
    cdef Context _from_handle(type cls, ContextHandle h_context, int device_id):
        """Create Context from existing ContextHandle (cdef-only factory)."""
        cdef Context ctx = cls.__new__(cls)
        ctx._h_context = h_context
        ctx._device_id = device_id
        return ctx

    @property
    def handle(self):
        """Return the underlying CUcontext handle."""
        if self._h_context.get() == NULL:
            return None
        return as_py(self._h_context)

    def __eq__(self, other):
        if not isinstance(other, Context):
            return NotImplemented
        cdef Context _other = <Context>other
        return as_intptr(self._h_context) == as_intptr(_other._h_context)

    def __hash__(self) -> int:
        return hash((type(self), as_intptr(self._h_context)))


@dataclass
class ContextOptions:
    """Options for context creation.

    Currently unused, reserved for future use.
    """
    pass  # TODO
