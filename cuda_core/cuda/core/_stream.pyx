# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from libc.stdint cimport uintptr_t, INT32_MIN
from libc.stdlib cimport strtol, getenv

from cuda.bindings cimport cydriver

from cuda.core._event cimport Event as cyEvent
from cuda.core._utils.cuda_utils cimport (
    check_or_create_options,
    HANDLE_RETURN,
)

import cython
import warnings
from dataclasses import dataclass
from typing import Protocol

from cuda.core._context cimport Context
from cuda.core._event import Event, EventOptions
from cuda.core._resource_handles cimport (
    ContextHandle,
    EventHandle,
    StreamHandle,
    create_context_handle_ref,
    create_event_handle_noctx,
    create_stream_handle,
    create_stream_handle_with_owner,
    get_current_context,
    get_legacy_stream,
    get_per_thread_stream,
    as_intptr,
    as_cu,
    as_py,
)

from cuda.core._graph import GraphBuilder


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
    priority: int | None = None


class IsStreamT(Protocol):
    def __cuda_stream__(self) -> tuple[int, int]:
        """
        For any Python object that is meant to be interpreted as a CUDA stream, the intent
        can be communicated by implementing this protocol that returns a 2-tuple: The protocol
        version number (currently ``0``) and the address of ``cudaStream_t``. Both values
        should be Python `int`.
        """
        ...


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
    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "Stream objects cannot be instantiated directly. "
            "Please use Device APIs (create_stream) or other Stream APIs (from_handle)."
        )

    @staticmethod
    cdef Stream _from_handle(type cls, StreamHandle h_stream):
        """Create a Stream from an existing StreamHandle (cdef-only factory)."""
        cdef Stream s = cls.__new__(cls)
        s._h_stream = h_stream
        # _h_context is default-initialized to empty ContextHandle by C++
        s._device_id = -1  # lazy init'd (invalid sentinel)
        s._nonblocking = -1  # lazy init'd
        s._priority = INT32_MIN  # lazy init'd
        return s

    @classmethod
    def legacy_default(cls):
        """Return the legacy default stream.

        The legacy default stream is an implicit stream which synchronizes
        with all other streams in the same CUDA context except for non-blocking
        streams. When any operation is launched on the legacy default stream,
        it waits for all previously launched operations in blocking streams to
        complete, and all subsequent operations in blocking streams wait for
        the legacy default stream operation to complete.

        Returns
        -------
        Stream
            The legacy default stream instance for the current context.

        See Also
        --------
        per_thread_default : Per-thread default stream alternative.

        """
        return Stream._from_handle(cls, get_legacy_stream())

    @classmethod
    def per_thread_default(cls):
        """Return the per-thread default stream.

        The per-thread default stream is local to both the calling thread and
        the CUDA context. Unlike the legacy default stream, it does not
        synchronize with other streams and behaves like an explicitly created
        non-blocking stream. This allows for better concurrency in multi-threaded
        applications.

        Returns
        -------
        Stream
            The per-thread default stream instance for the current thread
            and context.

        See Also
        --------
        legacy_default : Legacy default stream alternative.

        """
        return Stream._from_handle(cls, get_per_thread_stream())

    @classmethod
    def _init(cls, obj: IsStreamT | None = None, options=None, device_id: int = None,
              ctx: Context = None):
        cdef StreamHandle h_stream
        cdef cydriver.CUstream borrowed
        cdef ContextHandle h_context
        cdef Stream self

        # Extract context handle if provided
        if ctx is not None:
            h_context = (<Context>ctx)._h_context

        if obj is not None and options is not None:
            raise ValueError("obj and options cannot be both specified")
        if obj is not None:
            # Borrowed stream from foreign object
            # C++ handle prevents owner from being GC'd until handle is released
            # Owner is responsible for keeping the stream's context alive
            borrowed = _handle_from_stream_protocol(obj)
            h_stream = create_stream_handle_with_owner(borrowed, obj)
            return Stream._from_handle(cls, h_stream)

        cdef StreamOptions opts = check_or_create_options(StreamOptions, options, "Stream options")
        nonblocking = opts.nonblocking
        priority = opts.priority

        cdef unsigned int flags = (cydriver.CUstream_flags.CU_STREAM_NON_BLOCKING if nonblocking
                                   else cydriver.CUstream_flags.CU_STREAM_DEFAULT)
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

        # C++ creates the stream and returns owning handle with context dependency
        h_stream = create_stream_handle(h_context, flags, prio)
        if not h_stream:
            raise RuntimeError("Failed to create CUDA stream")
        self = Stream._from_handle(cls, h_stream)
        self._nonblocking = int(nonblocking)
        self._priority = prio
        if device_id is not None:
            self._device_id = device_id
        return self

    cpdef close(self):
        """Destroy the stream.

        Releases the stream handle. For owned streams, this destroys the
        underlying CUDA stream. For borrowed streams, this releases the
        reference and allows the Python owner to be GC'd.
        """
        self._h_stream.reset()

    def __cuda_stream__(self) -> tuple[int, int]:
        """Return an instance of a __cuda_stream__ protocol."""
        return (0, as_intptr(self._h_stream))

    def __hash__(self) -> int:
        return hash(as_intptr(self._h_stream))

    def __eq__(self, other) -> bool:
        if not isinstance(other, Stream):
            return NotImplemented
        return as_intptr(self._h_stream) == as_intptr((<Stream>other)._h_stream)

    def __repr__(self) -> str:
        Stream_ensure_ctx(self)
        return f"<Stream handle={as_intptr(self._h_stream):#x} context={as_intptr(self._h_context):#x}>"

    @property
    def handle(self) -> cuda.bindings.driver.CUstream:
        """Return the underlying ``CUstream`` object.

        .. caution::

            This handle is a Python object. To get the memory address of the underlying C
            handle, call ``int(Stream.handle)``.
        """
        return as_py(self._h_stream)

    @property
    def is_nonblocking(self) -> bool:
        """Return True if this is a nonblocking stream, otherwise False."""
        cdef unsigned int flags
        if self._nonblocking == -1:
            with nogil:
                HANDLE_RETURN(cydriver.cuStreamGetFlags(as_cu(self._h_stream), &flags))
            self._nonblocking = flags & cydriver.CUstream_flags.CU_STREAM_NON_BLOCKING
        return bool(self._nonblocking)

    @property
    def priority(self) -> int:
        """Return the stream priority."""
        cdef int prio
        if self._priority == INT32_MIN:
            with nogil:
                HANDLE_RETURN(cydriver.cuStreamGetPriority(as_cu(self._h_stream), &prio))
            self._priority = prio
        return self._priority

    def sync(self):
        """Synchronize the stream."""
        with nogil:
            HANDLE_RETURN(cydriver.cuStreamSynchronize(as_cu(self._h_stream)))

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
            Stream_ensure_ctx_device(self)
            event = cyEvent._init(cyEvent, self._device_id, self._h_context, options, False)
        elif event.is_ipc_enabled:
            raise TypeError(
                "IPC-enabled events should not be re-recorded, instead create a "
                "new event by supplying options."
            )

        cdef cydriver.CUevent e = as_cu((<cyEvent?>(event))._h_event)
        with nogil:
            HANDLE_RETURN(cydriver.cuEventRecord(e, as_cu(self._h_stream)))
        return event

    def wait(self, event_or_stream: Event | Stream):
        """Wait for a CUDA event or a CUDA stream.

        Waiting for an event or a stream establishes a stream order.

        If a :obj:`~_stream.Stream` is provided, then wait until the stream's
        work is completed. This is done by recording a new :obj:`~_event.Event`
        on the stream and then waiting on it.

        """
        cdef Stream stream
        cdef EventHandle h_event

        # Handle Event directly
        if isinstance(event_or_stream, Event):
            with nogil:
                # TODO: support flags other than 0?
                HANDLE_RETURN(cydriver.cuStreamWaitEvent(
                    as_cu(self._h_stream), as_cu((<cyEvent>event_or_stream)._h_event), 0))
            return

        # Convert to Stream if needed
        if isinstance(event_or_stream, Stream):
            stream = <Stream>event_or_stream
        else:
            try:
                stream = Stream._init(obj=event_or_stream)
            except Exception as e:
                raise ValueError(
                    "only an Event, Stream, or object supporting __cuda_stream__ can be waited,"
                    f" got {type(event_or_stream)}"
                ) from e

        # Wait on stream via temporary event
        with nogil:
            h_event = create_event_handle_noctx(cydriver.CUevent_flags.CU_EVENT_DISABLE_TIMING)
            HANDLE_RETURN(cydriver.cuEventRecord(as_cu(h_event), as_cu(stream._h_stream)))
            # TODO: support flags other than 0?
            HANDLE_RETURN(cydriver.cuStreamWaitEvent(as_cu(self._h_stream), as_cu(h_event), 0))

    @property
    def device(self) -> Device:
        """Return the :obj:`~_device.Device` singleton associated with this stream.

        Note
        ----
        The current context on the device may differ from this
        stream's context. This case occurs when a different CUDA
        context is set current after a stream is created.

        """
        from cuda.core._device import Device  # avoid circular import
        Stream_ensure_ctx_device(self)
        return Device(self._device_id)

    @property
    def context(self) -> Context:
        """Return the :obj:`~_context.Context` associated with this stream."""
        Stream_ensure_ctx(self)
        Stream_ensure_ctx_device(self)
        return Context._from_handle(Context, self._h_context, self._device_id)

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
cdef Stream C_LEGACY_DEFAULT_STREAM = Stream.legacy_default()
cdef Stream C_PER_THREAD_DEFAULT_STREAM = Stream.per_thread_default()

# standard python objects, public
LEGACY_DEFAULT_STREAM = C_LEGACY_DEFAULT_STREAM
PER_THREAD_DEFAULT_STREAM = C_PER_THREAD_DEFAULT_STREAM


cpdef Stream default_stream():
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


cdef inline int Stream_ensure_ctx(Stream self) except?-1 nogil:
    """Ensure the stream's context handle is populated."""
    cdef cydriver.CUcontext ctx
    if not self._h_context:
        HANDLE_RETURN(cydriver.cuStreamGetCtx(as_cu(self._h_stream), &ctx))
        with gil:
            self._h_context = create_context_handle_ref(ctx)
    return 0


cdef inline int Stream_ensure_ctx_device(Stream self) except?-1:
    """Ensure the stream's context and device_id are populated."""
    cdef cydriver.CUcontext ctx
    cdef cydriver.CUdevice target_dev
    cdef bint switch_context

    if self._device_id < 0:
        with nogil:
            # Get device ID from context, switching context temporarily if needed
            Stream_ensure_ctx(self)
            switch_context = (get_current_context() != self._h_context)
            if switch_context:
                HANDLE_RETURN(cydriver.cuCtxPushCurrent(as_cu(self._h_context)))
            HANDLE_RETURN(cydriver.cuCtxGetDevice(&target_dev))
            if switch_context:
                HANDLE_RETURN(cydriver.cuCtxPopCurrent(&ctx))
        self._device_id = <int>target_dev
    return 0


cdef cydriver.CUstream _handle_from_stream_protocol(obj) except*:
    if isinstance(obj, Stream):
        return <cydriver.CUstream><uintptr_t>(obj.handle)

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

# Helper for API functions that accept either Stream or GraphBuilder. Performs
# needed checks and returns the relevant stream.
cdef Stream Stream_accept(arg, bint allow_stream_protocol=False):
    if isinstance(arg, Stream):
        return <Stream>(arg)
    elif isinstance(arg, GraphBuilder):
        return <Stream>(arg.stream)
    elif allow_stream_protocol and hasattr(arg, "__cuda_stream__"):
        stream = Stream._init(arg)
        warnings.warn(
            "Passing foreign stream objects to this function via the "
            "stream protocol is deprecated. Convert the object explicitly "
            "using Stream(obj) instead.",
            stacklevel=2,
            category=DeprecationWarning,
        )
        return <Stream>(stream)
    raise TypeError(f"Stream or GraphBuilder expected, got {type(arg).__name__}")
