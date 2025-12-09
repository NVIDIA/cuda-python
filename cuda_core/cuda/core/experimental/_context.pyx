# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

from libc.stdint cimport uintptr_t

from cuda.bindings cimport cydriver
from cuda.core.experimental._resource_handles cimport (
    create_context_handle_ref,
    get_primary_context,
    get_current_context,
    native,
)
from cuda.core.experimental._utils.cuda_utils import driver
from cuda.core.experimental._utils.cuda_utils cimport HANDLE_RETURN


__all__ = ['Context', 'ContextOptions']


cdef class Context:
    """CUDA context wrapper.

    Context objects represent CUDA contexts and cannot be instantiated directly.
    Use Device or Stream APIs to obtain context objects.
    """

    def __init__(self, *args, **kwargs):
        raise RuntimeError("Context objects cannot be instantiated directly. Please use Device or Stream APIs.")

    @classmethod
    def _from_ctx(cls, handle: driver.CUcontext, int device_id):
        cdef Context ctx = Context.__new__(Context)
        # Convert Python CUcontext to C-level CUcontext and create non-owning ContextHandle
        cdef cydriver.CUcontext c_ctx = <cydriver.CUcontext><uintptr_t>int(handle)
        ctx._h_context = create_context_handle_ref(c_ctx)
        ctx._device_id = device_id
        return ctx

    @property
    def handle(self):
        """Return the underlying CUcontext handle."""
        cdef const cydriver.CUcontext* ptr = self._h_context.get()
        if ptr != NULL:
            return driver.CUcontext(<uintptr_t>(ptr[0]))
        return None

    def __eq__(self, other):
        if not isinstance(other, Context):
            return NotImplemented
        cdef Context _other = <Context>other
        # Compare the actual CUcontext values, not the shared_ptr objects
        # (aliasing constructor creates different addresses even for same CUcontext)
        cdef const cydriver.CUcontext* ptr1 = self._h_context.get()
        cdef const cydriver.CUcontext* ptr2 = _other._h_context.get()
        if ptr1 == NULL or ptr2 == NULL:
            return ptr1 == ptr2
        return ptr1[0] == ptr2[0]

    def __hash__(self) -> int:
        cdef const cydriver.CUcontext* ptr = self._h_context.get()
        if ptr == NULL:
            return hash((type(self), 0))
        return hash((type(self), <uintptr_t>(ptr[0])))


@dataclass
class ContextOptions:
    """Options for context creation.

    Currently unused, reserved for future use.
    """
    pass  # TODO


# get_current_context() and get_primary_context() are now pure C++ functions
# imported from _resource_handles (with thread-local caching in C++)


cdef ContextHandle get_stream_context(cydriver.CUstream stream) except * nogil:
    """Get handle to the context associated with a stream.

    Parameters
    ----------
    stream : CUstream
        Stream handle

    Returns
    -------
    ContextHandle
        Handle to context associated with the stream
    """
    cdef cydriver.CUcontext ctx = NULL
    HANDLE_RETURN(cydriver.cuStreamGetCtx(stream, &ctx))
    return create_context_handle_ref(ctx)


cdef void set_current_context(ContextHandle h_context) except * nogil:
    """Set the current CUDA context from a handle.

    Parameters
    ----------
    h_context : ContextHandle
        Context handle to set as current
    """
    if h_context.get() == NULL:
        with gil:
            raise ValueError("Cannot set NULL context as current")
    HANDLE_RETURN(cydriver.cuCtxSetCurrent(native(h_context)))
