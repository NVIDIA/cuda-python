# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

from cuda.core._resource_handles cimport (
    ContextHandle,
    GreenCtxHandle,
    get_green_ctx_context,
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
        ctx._is_green = False
        return ctx

    @staticmethod
    cdef Context _from_green_ctx(type cls, GreenCtxHandle h_green_ctx, int device_id):
        """Create Context from an owning green context handle."""
        cdef Context ctx = cls.__new__(cls)
        ctx._h_green_ctx = h_green_ctx
        ctx._h_context = get_green_ctx_context(h_green_ctx)
        ctx._device_id = device_id
        ctx._is_green = True
        return ctx

    @property
    def handle(self):
        """Return the underlying CUcontext handle."""
        if self._h_context.get() == NULL:
            return None
        return as_py(self._h_context)

    @property
    def _handle(self):
        return self.handle

    @property
    def is_green(self) -> bool:
        """True if this context was created from device resources."""
        return bool(self._is_green)

    cpdef close(self):
        """Release this context wrapper's underlying CUDA handles."""
        self._h_context.reset()
        self._h_green_ctx.reset()

    def __eq__(self, other):
        if not isinstance(other, Context):
            return NotImplemented
        cdef Context _other = <Context>other
        return as_intptr(self._h_context) == as_intptr(_other._h_context)

    def __hash__(self) -> int:
        return hash(as_intptr(self._h_context))

    def __repr__(self) -> str:
        return f"<Context handle={as_intptr(self._h_context):#x} device={self._device_id}>"


@dataclass
class ContextOptions:
    """Options for context creation.

    Attributes
    ----------
    resources : Sequence[SMResource | WorkqueueResource], optional
        Device resources used to create a green context.
    """
    resources: object = None
