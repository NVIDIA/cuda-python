# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import threading
from libc.stdint cimport uintptr_t

from cuda.bindings cimport cydriver
from cuda.core.experimental._utils.cuda_utils import driver, CUDAError
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
        ctx._handle = handle
        ctx._device_id = device_id
        return ctx

    def __eq__(self, other):
        if not isinstance(other, Context):
            return NotImplemented
        cdef Context _other = <Context>other
        return int(self._handle) == int(_other._handle)

    def __hash__(self) -> int:
        return hash(int(self._handle))


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
