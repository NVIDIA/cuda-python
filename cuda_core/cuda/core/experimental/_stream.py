# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import warnings
import weakref
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Protocol, Tuple, Union

if TYPE_CHECKING:
    import cuda.bindings
    from cuda.core.experimental._device import Device
from cuda.core.experimental._context import Context
from cuda.core.experimental._event import Event, EventOptions
from cuda.core.experimental._graph import GraphBuilder
from cuda.core.experimental._utils.clear_error_support import assert_type
from cuda.core.experimental._utils.cuda_utils import (
    check_or_create_options,
    driver,
    get_device_from_ctx,
    handle_return,
    runtime,
)


@dataclass
class StreamOptions:
    """Customizable :obj:`~_stream.Stream` options.

    Attributes
    ----------
    nonblocking : bool, optional
        Stream does not synchronize with the NULL stream. (Default to True)
    priority : int, optional
        Stream priority where lower number represents a
        higher priority. (Default to lowest priority)

    """

    nonblocking: bool = True
    priority: Optional[int] = None


class IsStreamT(Protocol):
    def __cuda_stream__(self) -> Tuple[int, int]:
        """
        For any Python object that is meant to be interpreted as a CUDA stream, the intent
        can be communicated by implementing this protocol that returns a 2-tuple: The protocol
        version number (currently ``0``) and the address of ``cudaStream_t``. Both values
        should be Python `int`.
        """
        ...


def _try_to_get_stream_ptr(obj: IsStreamT):
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
    return driver.CUstream(info[1])


class Stream:
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

    class _MembersNeededForFinalize:
        __slots__ = ("handle", "owner", "builtin")

        def __init__(self, stream_obj, handle, owner, builtin):
            self.handle = handle
            self.owner = owner
            self.builtin = builtin
            weakref.finalize(stream_obj, self.close)

        def close(self):
            if self.owner is None:
                if self.handle and not self.builtin:
                    handle_return(driver.cuStreamDestroy(self.handle))
            else:
                self.owner = None
            self.handle = None

    def __new__(self, *args, **kwargs):
        raise RuntimeError(
            "Stream objects cannot be instantiated directly. "
            "Please use Device APIs (create_stream) or other Stream APIs (from_handle)."
        )

    __slots__ = ("__weakref__", "_mnff", "_nonblocking", "_priority", "_device_id", "_ctx_handle")

    @classmethod
    def _legacy_default(cls):
        self = super().__new__(cls)
        self._mnff = Stream._MembersNeededForFinalize(self, driver.CUstream(driver.CU_STREAM_LEGACY), None, True)
        self._nonblocking = None  # delayed
        self._priority = None  # delayed
        self._device_id = None  # delayed
        self._ctx_handle = None  # delayed
        return self

    @classmethod
    def _per_thread_default(cls):
        self = super().__new__(cls)
        self._mnff = Stream._MembersNeededForFinalize(self, driver.CUstream(driver.CU_STREAM_PER_THREAD), None, True)
        self._nonblocking = None  # delayed
        self._priority = None  # delayed
        self._device_id = None  # delayed
        self._ctx_handle = None  # delayed
        return self

    @classmethod
    def _init(cls, obj: Optional[IsStreamT] = None, *, options: Optional[StreamOptions] = None):
        self = super().__new__(cls)
        self._mnff = Stream._MembersNeededForFinalize(self, None, None, False)

        if obj is not None and options is not None:
            raise ValueError("obj and options cannot be both specified")
        if obj is not None:
            self._mnff.handle = _try_to_get_stream_ptr(obj)
            # TODO: check if obj is created under the current context/device
            self._mnff.owner = obj
            self._nonblocking = None  # delayed
            self._priority = None  # delayed
            self._device_id = None  # delayed
            self._ctx_handle = None  # delayed
            return self

        options = check_or_create_options(StreamOptions, options, "Stream options")
        nonblocking = options.nonblocking
        priority = options.priority

        flags = driver.CUstream_flags.CU_STREAM_NON_BLOCKING if nonblocking else driver.CUstream_flags.CU_STREAM_DEFAULT

        high, low = handle_return(runtime.cudaDeviceGetStreamPriorityRange())
        if priority is not None:
            if not (low <= priority <= high):
                raise ValueError(f"{priority=} is out of range {[low, high]}")
        else:
            priority = high

        self._mnff.handle = handle_return(driver.cuStreamCreateWithPriority(flags, priority))
        self._mnff.owner = None
        self._nonblocking = nonblocking
        self._priority = priority
        # don't defer this because we will have to pay a cost for context
        # switch later
        self._device_id = int(handle_return(driver.cuCtxGetDevice()))
        self._ctx_handle = None  # delayed
        return self

    def close(self):
        """Destroy the stream.

        Destroy the stream if we own it. Borrowed foreign stream
        object will instead have their references released.

        """
        self._mnff.close()

    def __cuda_stream__(self) -> Tuple[int, int]:
        """Return an instance of a __cuda_stream__ protocol."""
        return (0, self.handle)

    @property
    def handle(self) -> cuda.bindings.driver.CUstream:
        """Return the underlying ``CUstream`` object.

        .. caution::

            This handle is a Python object. To get the memory address of the underlying C
            handle, call ``int(Stream.handle)``.
        """
        return self._mnff.handle

    @property
    def is_nonblocking(self) -> bool:
        """Return True if this is a nonblocking stream, otherwise False."""
        if self._nonblocking is None:
            flag = handle_return(driver.cuStreamGetFlags(self._mnff.handle))
            if flag == driver.CUstream_flags.CU_STREAM_NON_BLOCKING:
                self._nonblocking = True
            else:
                self._nonblocking = False
        return self._nonblocking

    @property
    def priority(self) -> int:
        """Return the stream priority."""
        if self._priority is None:
            prio = handle_return(driver.cuStreamGetPriority(self._mnff.handle))
            self._priority = prio
        return self._priority

    def sync(self):
        """Synchronize the stream."""
        handle_return(driver.cuStreamSynchronize(self._mnff.handle))

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
            event = Event._init(self._device_id, self._ctx_handle, options)
        assert_type(event, Event)
        handle_return(driver.cuEventRecord(event.handle, self._mnff.handle))
        return event

    def wait(self, event_or_stream: Union[Event, Stream]):
        """Wait for a CUDA event or a CUDA stream.

        Waiting for an event or a stream establishes a stream order.

        If a :obj:`~_stream.Stream` is provided, then wait until the stream's
        work is completed. This is done by recording a new :obj:`~_event.Event`
        on the stream and then waiting on it.

        """
        if isinstance(event_or_stream, Event):
            event = event_or_stream.handle
            discard_event = False
        else:
            if isinstance(event_or_stream, Stream):
                stream = event_or_stream
            else:
                try:
                    stream = Stream._init(event_or_stream)
                except Exception as e:
                    raise ValueError(
                        "only an Event, Stream, or object supporting __cuda_stream__ can be waited,"
                        f" got {type(event_or_stream)}"
                    ) from e
            event = handle_return(driver.cuEventCreate(driver.CUevent_flags.CU_EVENT_DISABLE_TIMING))
            handle_return(driver.cuEventRecord(event, stream.handle))
            discard_event = True

        # TODO: support flags other than 0?
        handle_return(driver.cuStreamWaitEvent(self._mnff.handle, event, 0))
        if discard_event:
            handle_return(driver.cuEventDestroy(event))

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

        if self._device_id is None:
            # Get the stream context first
            if self._ctx_handle is None:
                self._ctx_handle = handle_return(driver.cuStreamGetCtx(self._mnff.handle))
            self._device_id = get_device_from_ctx(self._ctx_handle)
        return Device(self._device_id)

    @property
    def context(self) -> Context:
        """Return the :obj:`~_context.Context` associated with this stream."""
        if self._ctx_handle is None:
            self._ctx_handle = handle_return(driver.cuStreamGetCtx(self._mnff.handle))
        if self._device_id is None:
            self._device_id = get_device_from_ctx(self._ctx_handle)
        return Context._from_ctx(self._ctx_handle, self._device_id)

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


LEGACY_DEFAULT_STREAM = Stream._legacy_default()
PER_THREAD_DEFAULT_STREAM = Stream._per_thread_default()


def default_stream():
    """Return the default CUDA :obj:`~_stream.Stream`.

    The type of default stream returned depends on if the environment
    variable CUDA_PYTHON_CUDA_PER_THREAD_DEFAULT_STREAM is set.

    If set, returns a per-thread default stream. Otherwise returns
    the legacy stream.

    """
    # TODO: flip the default
    use_ptds = int(os.environ.get("CUDA_PYTHON_CUDA_PER_THREAD_DEFAULT_STREAM", 0))
    if use_ptds:
        return PER_THREAD_DEFAULT_STREAM
    else:
        return LEGACY_DEFAULT_STREAM
