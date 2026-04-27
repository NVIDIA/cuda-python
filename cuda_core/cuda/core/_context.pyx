# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from cuda.bindings cimport cydriver
from cuda.core._device_resources import SMResource, WorkqueueResource
from cuda.core._resource_handles cimport (
    ContextHandle,
    GreenCtxHandle,
    as_cu,
    create_context_handle_from_green_ctx,
    get_context_green_ctx,
    get_last_error,
    as_intptr,
    as_py,
)
from cuda.core._utils.cuda_utils cimport HANDLE_RETURN


__all__ = ['Context', 'ContextOptions']


DeviceResourcesT = Sequence[SMResource | WorkqueueResource]


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

    @staticmethod
    cdef Context _from_green_ctx(type cls, GreenCtxHandle h_green_ctx, int device_id):
        """Create Context from an owning green context handle."""
        cdef ContextHandle h_context = create_context_handle_from_green_ctx(h_green_ctx)
        if not h_context:
            HANDLE_RETURN(get_last_error())
            raise RuntimeError("Failed to create CUDA context view from green context")
        return Context._from_handle(cls, h_context, device_id)

    @property
    def handle(self):
        """Return the underlying CUcontext handle."""
        if not self._h_context:
            return None
        if as_cu(self._h_context) == NULL:
            return None
        return as_py(self._h_context)

    @property
    def _handle(self):
        return self.handle

    @property
    def is_green(self) -> bool:
        """True if this context was created from device resources."""
        if not self._h_context:
            return False
        return get_context_green_ctx(self._h_context).get() != NULL

    cpdef close(self):
        """Release this context wrapper's underlying CUDA handles."""
        cdef cydriver.CUcontext current_ctx
        if self._h_context and as_cu(self._h_context) != NULL:
            with nogil:
                HANDLE_RETURN(cydriver.cuCtxGetCurrent(&current_ctx))
            if current_ctx == as_cu(self._h_context):
                raise RuntimeError(
                    "Cannot close a CUDA context while it is current. "
                    "Restore a previous context before closing this context."
                )
        self._h_context.reset()

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
cdef class ContextOptions:
    """Options for context creation.

    Attributes
    ----------
    resources : :obj:`~_context.DeviceResourcesT`
        Device resources used to create a green context.
    """
    resources: DeviceResourcesT
