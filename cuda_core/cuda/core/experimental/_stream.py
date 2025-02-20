# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

from __future__ import annotations

import os
import warnings
import weakref
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Tuple, Union

if TYPE_CHECKING:
    from cuda.core.experimental._device import Device
from cuda.core.experimental._context import Context
from cuda.core.experimental._event import Event, EventOptions
from cuda.core.experimental._graph import GraphBuilder
from cuda.core.experimental._utils import check_or_create_options, driver, get_device_from_ctx, handle_return, runtime


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

    __slots__ = ("__weakref__", "_mnff", "_nonblocking", "_priority", "_device_id", "_ctx_handle")

    def __init__(self):
        raise NotImplementedError(
            "directly creating a Stream object can be ambiguous. Please either "
            "call Device.create_stream() or, if a stream pointer is already "
            "available from somewhere else, Stream.from_handle()"
        )

    @staticmethod
    def _init(obj=None, *, options: Optional[StreamOptions] = None):
        self = Stream.__new__(Stream)
        self._mnff = Stream._MembersNeededForFinalize(self, None, None, False)

        if obj is not None and options is not None:
            raise ValueError("obj and options cannot be both specified")
        if obj is not None:
            try:
                info = obj.__cuda_stream__()
            except AttributeError as e:
                raise TypeError(f"{type(obj)} object does not have a '__cuda_stream__' method") from e
            except TypeError:
                info = obj.__cuda_stream__
                warnings.simplefilter("once", DeprecationWarning)
                warnings.warn(
                    "Implementing __cuda_stream__ as an attribute is deprecated; it must be implemented as a method",
                    stacklevel=3,
                    category=DeprecationWarning,
                )

            assert info[0] == 0
            self._mnff.handle = driver.CUstream(info[1])
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
    def handle(self) -> int:
        """Return the underlying cudaStream_t pointer address as Python int."""
        return int(self._mnff.handle)

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
            event = Event._init(options)
        elif not isinstance(event, Event):
            raise TypeError("record only takes an Event object")
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
            if not isinstance(event_or_stream, Stream):
                try:
                    stream = Stream._init(event_or_stream)
                except Exception as e:
                    raise ValueError("only an Event, Stream, or object supporting __cuda_stream__ can be waited") from e
            else:
                stream = event_or_stream
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

    def build_graph(self) -> GraphBuilder:
        return GraphBuilder._init(stream=self)


class _LegacyDefaultStream(Stream):
    def __init__(self):
        self._mnff = Stream._MembersNeededForFinalize(self, driver.CUstream(driver.CU_STREAM_LEGACY), None, True)
        self._nonblocking = None  # delayed
        self._priority = None  # delayed


class _PerThreadDefaultStream(Stream):
    def __init__(self):
        self._mnff = Stream._MembersNeededForFinalize(self, driver.CUstream(driver.CU_STREAM_PER_THREAD), None, True)
        self._nonblocking = None  # delayed
        self._priority = None  # delayed


LEGACY_DEFAULT_STREAM = _LegacyDefaultStream()
PER_THREAD_DEFAULT_STREAM = _PerThreadDefaultStream()


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
