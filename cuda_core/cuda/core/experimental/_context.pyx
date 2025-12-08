# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import threading
from libc.stdint cimport uintptr_t

from cuda.bindings cimport cydriver
from cuda.core.experimental._resource_handles cimport create_context_handle_ref
from cuda.core.experimental._utils.cuda_utils import driver
from cuda.core.experimental._utils.cuda_utils cimport HANDLE_RETURN


__all__ = ['Context', 'ContextOptions']


# Lightweight Python wrapper for ContextHandle (for caching in TLS)
cdef class _ContextHandleWrapper:
    """Internal wrapper to store ContextHandle in Python containers."""
    cdef ContextHandle h_context

    def __cinit__(self):
        pass

    @staticmethod
    cdef _ContextHandleWrapper create(ContextHandle h_context):
        cdef _ContextHandleWrapper wrapper = _ContextHandleWrapper.__new__(_ContextHandleWrapper)
        wrapper.h_context = h_context
        return wrapper


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


cdef ContextHandle get_current_context() except * nogil:
    """Get handle to the current CUDA context.

    Returns
    -------
    ContextHandle
        Handle to current context, or empty handle if no context is bound
    """
    cdef cydriver.CUcontext ctx = NULL
    HANDLE_RETURN(cydriver.cuCtxGetCurrent(&ctx))
    if ctx == NULL:
        return ContextHandle()
    return create_context_handle_ref(ctx)


cdef ContextHandle get_primary_context(int dev_id) except *:
    """Get handle to the primary context for a device.

    Uses thread-local storage to cache primary context handles per device.
    The primary context is lazily initialized on first access.

    Parameters
    ----------
    dev_id : int
        Device ID

    Returns
    -------
    ContextHandle
        Handle to primary context for the device
    """
    cdef int total = 0
    cdef cydriver.CUcontext ctx
    cdef ContextHandle h_context
    cdef _ContextHandleWrapper wrapper

    # Check TLS cache
    try:
        primary_ctxs = _tls.primary_ctxs
    except AttributeError:
        # Initialize primary context cache
        with nogil:
            HANDLE_RETURN(cydriver.cuDeviceGetCount(&total))
        primary_ctxs = _tls.primary_ctxs = [None] * total

    wrapper = primary_ctxs[dev_id]
    if wrapper is not None:
        return wrapper.h_context

    # Acquire primary context (release GIL for driver call)
    with nogil:
        HANDLE_RETURN(cydriver.cuDevicePrimaryCtxRetain(&ctx, dev_id))
        h_context = create_context_handle_ref(ctx)

    # Cache the handle (wrapped in Python object)
    _tls.primary_ctxs[dev_id] = _ContextHandleWrapper.create(h_context)

    return h_context


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
    HANDLE_RETURN(cydriver.cuCtxSetCurrent(h_context.get()[0]))


# Thread-local storage for primary context cache
_tls = threading.local()
