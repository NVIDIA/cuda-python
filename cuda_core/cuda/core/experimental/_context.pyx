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
        ctx._resource_handle = create_context_handle_ref(c_ctx)
        ctx._device_id = device_id
        return ctx

    @property
    def handle(self):
        """Return the underlying CUcontext handle."""
        cdef const cydriver.CUcontext* ptr = self._resource_handle.get()
        if ptr != NULL:
            return driver.CUcontext(<uintptr_t>(ptr[0]))
        return None

    def __eq__(self, other):
        if not isinstance(other, Context):
            return NotImplemented
        cdef Context _other = <Context>other
        # Compare the actual CUcontext values, not the shared_ptr objects
        # (aliasing constructor creates different addresses even for same CUcontext)
        cdef const cydriver.CUcontext* ptr1 = self._resource_handle.get()
        cdef const cydriver.CUcontext* ptr2 = _other._resource_handle.get()
        if ptr1 == NULL or ptr2 == NULL:
            return ptr1 == ptr2
        return ptr1[0] == ptr2[0]

    def __hash__(self) -> int:
        cdef const cydriver.CUcontext* ptr = self._resource_handle.get()
        if ptr == NULL:
            return hash((type(self), 0))
        return hash((type(self), <uintptr_t>(ptr[0])))


@dataclass
class ContextOptions:
    """Options for context creation.

    Currently unused, reserved for future use.
    """
    pass  # TODO


cdef cydriver.CUcontext get_current_context() except?NULL nogil:
    """Get the current CUDA context.

    Returns
    -------
    CUcontext
        Current context handle, or NULL if no context is bound
    """
    cdef cydriver.CUcontext ctx = NULL
    HANDLE_RETURN(cydriver.cuCtxGetCurrent(&ctx))
    return ctx


cdef cydriver.CUcontext get_primary_context(int dev_id) except?NULL:
    """Get the primary context for a device.

    Uses thread-local storage to cache primary contexts per device.
    The primary context is lazily initialized on first access.

    Parameters
    ----------
    dev_id : int
        Device ID

    Returns
    -------
    CUcontext
        Primary context handle for the device, or NULL on error
    """
    cdef int total = 0
    cdef cydriver.CUcontext ctx

    try:
        primary_ctxs = _tls.primary_ctxs
    except AttributeError:
        # Initialize primary context cache
        with nogil:
            HANDLE_RETURN(cydriver.cuDeviceGetCount(&total))
        primary_ctxs = _tls.primary_ctxs = [0] * total

    ctx = <cydriver.CUcontext><uintptr_t>(primary_ctxs[dev_id])
    if ctx == NULL:
        with nogil:
            HANDLE_RETURN(cydriver.cuDevicePrimaryCtxRetain(&ctx, dev_id))
        primary_ctxs[dev_id] = <uintptr_t>(ctx)
    return ctx


cdef cydriver.CUcontext get_stream_context(cydriver.CUstream stream) except?NULL nogil:
    """Get the context associated with a stream.

    Parameters
    ----------
    stream : CUstream
        Stream handle

    Returns
    -------
    CUcontext
        Context handle associated with the stream, or NULL on error
    """
    cdef cydriver.CUcontext ctx = NULL
    HANDLE_RETURN(cydriver.cuStreamGetCtx(stream, &ctx))
    return ctx


cdef void set_current_context(cydriver.CUcontext ctx) except *:
    """Set the current CUDA context.

    Parameters
    ----------
    ctx : CUcontext
        Context handle to set as current
    """
    with nogil:
        HANDLE_RETURN(cydriver.cuCtxSetCurrent(ctx))


# Thread-local storage for primary context cache
_tls = threading.local()
