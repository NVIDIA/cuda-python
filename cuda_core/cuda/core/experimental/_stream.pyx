# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from libc.stdint cimport uintptr_t, INT32_MIN
from libc.stdlib cimport strtol, getenv

from cuda.bindings cimport cydriver

from cuda.core.experimental._event cimport Event as cyEvent
from cuda.core.experimental._utils.cuda_utils cimport (
    check_or_create_options,
    CU_CONTEXT_INVALID,
    get_device_from_ctx,
    HANDLE_RETURN,
)

import cython
import os
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Protocol, Union

if TYPE_CHECKING:
    import cuda.bindings
    from cuda.core.experimental._device import Device
from cuda.core.experimental._context import Context
from cuda.core.experimental._event import Event, EventOptions
from cuda.core.experimental._graph import GraphBuilder
from cuda.core.experimental._utils.clear_error_support import assert_type
from cuda.core.experimental._utils.cuda_utils import (
    driver,
)


@dataclass
cdef class StreamOptions:
    """Customizable :obj:`~_stream.Stream` options.

    Attributes
    ----------
    nonblocking : bool, optional
        Stream does not synchronize with the NULL stream. (Default to True)
    priority : int, optional
        Stream priority where lower number represents a
        higher priority. (Default to lowest priority)

    """

    nonblocking : cython.bint = True
    priority: Optional[int] = None


class IsStreamT(Protocol):
    def __cuda_stream__(self) -> tuple[int, int]:
        """
        For any Python object that is meant to be interpreted as a CUDA stream, the intent
        can be communicated by implementing this protocol that returns a 2-tuple: The protocol
        version number (currently ``0``) and the address of ``cudaStream_t``. Both values
        should be Python `int`.
        """
        ...


cdef cydriver.CUstream _try_to_get_stream_ptr(obj: IsStreamT) except*:
    try:
        cuda_stream_attr = obj.__cuda_stream__
    except AttributeError:
        raise TypeError(f"{type(obj)} object does not have a '__cuda_stream__' attribute") from None

    if callable(cuda_stream_attr):
        info = cuda_stream_attr()
    else:
        info = cuda_stream_attr
        warnings.simplefilter("once", DeprecationWarning)
        warnings.warn(
            "Implementing __cuda_stream__ as an attribute is deprecated; it must be implemented as a method",
            stacklevel=3,
            category=DeprecationWarning,
        )

    try:
        len_info = len(info)
    except TypeError as e:
        raise RuntimeError(f"obj.__cuda_stream__ must return a sequence with 2 elements, got {type(info)}") from e
    if len_info != 2:
        raise RuntimeError(f"obj.__cuda_stream__ must return a sequence with 2 elements, got {len_info} elements")
    if info[0] != 0:
        raise RuntimeError(
            f"The first element of the sequence returned by obj.__cuda_stream__ must be 0, got {repr(info[0])}"
        )
    return <cydriver.CUstream><uintptr_t>(info[1])


cdef class Stream:
    """Represent a queue of GPU operations that are executed in a specific order.

    Applications use streams to control the order of execution for
    GPU work. Work within a single stream are executed sequentially.
    Whereas work across multiple streams can be further controlled
    using stream priorities and :obj:`~_event.Event` managements.

    Advanced users can utilize default streams for enforce complex
    implicit synchronization behaviors.

    Directly creating a :obj:`~_stream.Stream` is not supported due to ambiguity.
    New streams should instead be created through a :obj:`~_device.Device`
    object, or created directly through using an existing handle
    using Stream.from_handle().
    """
    def __cinit__(self):
        self._handle = <cydriver.CUstream>(NULL)
        self._owner = None
        self._builtin = False
        self._nonblocking = -1  # lazy init'd
        self._priority = INT32_MIN  # lazy init'd
        self._device_id = cydriver.CU_DEVICE_INVALID  # lazy init'd
        self._ctx_handle = CU_CONTEXT_INVALID  # lazy init'd

    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "Stream objects cannot be instantiated directly. "
            "Please use Device APIs (create_stream) or other Stream APIs (from_handle)."
        )

    @classmethod
    def _legacy_default(cls):
        cdef Stream self = Stream.__new__(cls)
        self._handle = <cydriver.CUstream>(cydriver.CU_STREAM_LEGACY)
        self._builtin = True
        return self

    @classmethod
    def _per_thread_default(cls):
        cdef Stream self = Stream.__new__(cls)
        self._handle = <cydriver.CUstream>(cydriver.CU_STREAM_PER_THREAD)
        self._builtin = True
        return self

    @classmethod
    def _init(cls, obj: Optional[IsStreamT] = None, options=None, device_id: int = None):
        cdef Stream self = Stream.__new__(cls)

        if obj is not None and options is not None:
            raise ValueError("obj and options cannot be both specified")
        if obj is not None:
            self._handle = _try_to_get_stream_ptr(obj)
            # TODO: check if obj is created under the current context/device
            self._owner = obj
            return self

        cdef StreamOptions opts = check_or_create_options(StreamOptions, options, "Stream options")
        nonblocking = opts.nonblocking
        priority = opts.priority

        flags = cydriver.CUstream_flags.CU_STREAM_NON_BLOCKING if nonblocking else cydriver.CUstream_flags.CU_STREAM_DEFAULT
        # TODO: we might want to consider memoizing high/low per CUDA context and avoid this call
        cdef int high, low
        with nogil:
            HANDLE_RETURN(cydriver.cuCtxGetStreamPriorityRange(&high, &low))
        cdef int prio
        if priority is not None:
            prio = priority
            if not (low <= prio <= high):
                raise ValueError(f"{priority=} is out of range {[low, high]}")
        else:
            prio = high

        cdef cydriver.CUstream s
        with nogil:
            HANDLE_RETURN(cydriver.cuStreamCreateWithPriority(&s, flags, prio))
        self._handle = s
        self._nonblocking = int(nonblocking)
        self._priority = prio
        self._device_id = device_id if device_id is not None else self._device_id
        return self

    def __dealloc__(self):
        self.close()

    cpdef close(self):
        """Destroy the stream.

        Destroy the stream if we own it. Borrowed foreign stream
        object will instead have their references released.

        """
        if self._owner is None:
            if self._handle and not self._builtin:
                with nogil:
                    HANDLE_RETURN(cydriver.cuStreamDestroy(self._handle))
        else:
            self._owner = None
        self._handle = <cydriver.CUstream>(NULL)

    def __cuda_stream__(self) -> tuple[int, int]:
        """Return an instance of a __cuda_stream__ protocol."""
        return (0, <uintptr_t>(self._handle))

    @property
    def handle(self) -> cuda.bindings.driver.CUstream:
        """Return the underlying ``CUstream`` object.

        .. caution::

            This handle is a Python object. To get the memory address of the underlying C
            handle, call ``int(Stream.handle)``.
        """
        return driver.CUstream(<uintptr_t>(self._handle))

    @property
    def is_nonblocking(self) -> bool:
        """Return True if this is a nonblocking stream, otherwise False."""
        cdef unsigned int flags
        if self._nonblocking == -1:
            with nogil:
                HANDLE_RETURN(cydriver.cuStreamGetFlags(self._handle, &flags))
            if flags & cydriver.CUstream_flags.CU_STREAM_NON_BLOCKING:
                self._nonblocking = True
            else:
                self._nonblocking = False
        return bool(self._nonblocking)

    @property
    def priority(self) -> int:
        """Return the stream priority."""
        cdef int prio
        if self._priority == INT32_MIN:
            with nogil:
                HANDLE_RETURN(cydriver.cuStreamGetPriority(self._handle, &prio))
            self._priority = prio
        return self._priority

    def sync(self):
        """Synchronize the stream."""
        with nogil:
            HANDLE_RETURN(cydriver.cuStreamSynchronize(self._handle))

    def record(self, event: Event = None, options: EventOptions = None) -> Event:
        """Record an event onto the stream.

        Creates an Event object (or reuses the given one) by
        recording on the stream.

        Parameters
        ----------
        event : :obj:`~_event.Event`, optional
            Optional event object to be reused for recording.
        options : :obj:`EventOptions`, optional
            Customizable dataclass for event creation options.

        Returns
        -------
        :obj:`~_event.Event`
            Newly created event object.

        """
        # Create an Event object (or reusing the given one) by recording
        # on the stream. Event flags such as disabling timing, nonblocking,
        # and CU_EVENT_RECORD_EXTERNAL, can be set in EventOptions.
        if event is None:
            self._get_device_and_context()
            event = Event._init(<int>(self._device_id), <uintptr_t>(self._ctx_handle), options)
        cdef cydriver.CUevent e = (<cyEvent?>(event))._handle
        with nogil:
            HANDLE_RETURN(cydriver.cuEventRecord(e, self._handle))
        return event

    def wait(self, event_or_stream: Union[Event, Stream]):
        """Wait for a CUDA event or a CUDA stream.

        Waiting for an event or a stream establishes a stream order.

        If a :obj:`~_stream.Stream` is provided, then wait until the stream's
        work is completed. This is done by recording a new :obj:`~_event.Event`
        on the stream and then waiting on it.

        """
        cdef cydriver.CUevent event
        cdef cydriver.CUstream stream

        if isinstance(event_or_stream, Event):
            event = <cydriver.CUevent><uintptr_t>(event_or_stream.handle)
            with nogil:
                # TODO: support flags other than 0?
                HANDLE_RETURN(cydriver.cuStreamWaitEvent(self._handle, event, 0))
        else:
            if isinstance(event_or_stream, Stream):
                stream = <cydriver.CUstream><uintptr_t>(event_or_stream.handle)
            else:
                try:
                    s = Stream._init(obj=event_or_stream)
                except Exception as e:
                    raise ValueError(
                        "only an Event, Stream, or object supporting __cuda_stream__ can be waited,"
                        f" got {type(event_or_stream)}"
                    ) from e
                stream = <cydriver.CUstream><uintptr_t>(s.handle)
            with nogil:
                HANDLE_RETURN(cydriver.cuEventCreate(&event, cydriver.CUevent_flags.CU_EVENT_DISABLE_TIMING))
                HANDLE_RETURN(cydriver.cuEventRecord(event, stream))
                # TODO: support flags other than 0?
                HANDLE_RETURN(cydriver.cuStreamWaitEvent(self._handle, event, 0))
                HANDLE_RETURN(cydriver.cuEventDestroy(event))

    @property
    def device(self) -> Device:
        """Return the :obj:`~_device.Device` singleton associated with this stream.

        Note
        ----
        The current context on the device may differ from this
        stream's context. This case occurs when a different CUDA
        context is set current after a stream is created.

        """
        from cuda.core.experimental._device import Device  # avoid circular import
        self._get_device_and_context()
        return Device(<int>(self._device_id))

    cdef int _get_context(self) except?-1 nogil:
        if self._ctx_handle == CU_CONTEXT_INVALID:
            HANDLE_RETURN(cydriver.cuStreamGetCtx(self._handle, &(self._ctx_handle)))
        return 0

    cdef int _get_device_and_context(self) except?-1:
        cdef cydriver.CUcontext curr_ctx
        if self._device_id == cydriver.CU_DEVICE_INVALID:
            with nogil:
                # Get the current context
                HANDLE_RETURN(cydriver.cuCtxGetCurrent(&curr_ctx))
                # Get the stream's context (self.ctx_handle is populated)
                self._get_context()
                # Get the stream's device (may require a context-switching dance)
                self._device_id = get_device_from_ctx(self._ctx_handle, curr_ctx)
        return 0

    @property
    def context(self) -> Context:
        """Return the :obj:`~_context.Context` associated with this stream."""
        self._get_context()
        self._get_device_and_context()
        return Context._from_ctx(<uintptr_t>(self._ctx_handle), <int>(self._device_id))

    @staticmethod
    def from_handle(handle: int) -> Stream:
        """Create a new :obj:`~_stream.Stream` object from a foreign stream handle.

        Uses a cudaStream_t pointer address represented as a Python int
        to create a new :obj:`~_stream.Stream` object.

        Note
        ----
        Stream lifetime is not managed, foreign object must remain
        alive while this steam is active.

        Parameters
        ----------
        handle : int
            Stream handle representing the address of a foreign
            stream object.

        Returns
        -------
        :obj:`~_stream.Stream`
            Newly created stream object.

        """

        class _stream_holder:
            def __cuda_stream__(self):
                return (0, handle)

        return Stream._init(obj=_stream_holder())

    def create_graph_builder(self) -> GraphBuilder:
        """Create a new :obj:`~_graph.GraphBuilder` object.

        The new graph builder will be associated with this stream.

        Returns
        -------
        :obj:`~_graph.GraphBuilder`
            Newly created graph builder object.

        """
        return GraphBuilder._init(stream=self, is_stream_owner=False)


# c-only python objects, not public
cdef Stream C_LEGACY_DEFAULT_STREAM = Stream._legacy_default()
cdef Stream C_PER_THREAD_DEFAULT_STREAM = Stream._per_thread_default()

# standard python objects, public
LEGACY_DEFAULT_STREAM = C_LEGACY_DEFAULT_STREAM
PER_THREAD_DEFAULT_STREAM = C_PER_THREAD_DEFAULT_STREAM


cdef Stream default_stream():
    """Return the default CUDA :obj:`~_stream.Stream`.

    The type of default stream returned depends on if the environment
    variable CUDA_PYTHON_CUDA_PER_THREAD_DEFAULT_STREAM is set.

    If set, returns a per-thread default stream. Otherwise returns
    the legacy stream.

    """
    # TODO: flip the default
    cdef const char* use_ptds_raw = getenv("CUDA_PYTHON_CUDA_PER_THREAD_DEFAULT_STREAM")

    cdef int use_ptds = 0
    if use_ptds_raw != NULL:
        use_ptds = strtol(use_ptds_raw, NULL, 10)

    # value is non-zero, including for weird stuff like 123foo
    if use_ptds:
        return C_PER_THREAD_DEFAULT_STREAM
    else:
        return C_LEGACY_DEFAULT_STREAM
