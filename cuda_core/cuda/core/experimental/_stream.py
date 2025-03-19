# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

from __future__ import annotations

import os
import warnings
import weakref
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Tuple, Union, Any

if TYPE_CHECKING:
    import cuda.bindings
    from cuda.core.experimental._device import Device
from cuda.core.experimental._context import Context
from cuda.core.experimental._event import Event, EventOptions
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

        def __init__(self, stream_obj: "Stream", handle: driver.CUstream, owner: Optional[Any], builtin: bool) -> None:
            self.handle = handle
            self.owner = owner
            self.builtin = builtin
            weakref.finalize(stream_obj, self.close)

        def close(self) -> None:
            if self.owner is None:
                if self.handle and not self.builtin:
                    handle_return(driver.cuStreamDestroy(self.handle))
            else:
                self.owner = None
            self.handle = None

    def __new__(self, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError(
            "Stream objects cannot be instantiated directly. "
            "Please use Device APIs (create_stream) or other Stream APIs (from_handle)."
        )

    __slots__ = ("__weakref__", "_mnff", "_nonblocking", "_priority", "_device_id", "_ctx_handle")

    @classmethod
    def _legacy_default(cls) -> "Stream":
        self = super().__new__(cls)
        self._mnff = Stream._MembersNeededForFinalize(self, driver.CUstream(driver.CU_STREAM_LEGACY), None, True)
        self._nonblocking = None  # delayed
        self._priority = None  # delayed
        self._device_id = None  # delayed
        self._ctx_handle = None  # delayed
        return self

    @classmethod
    def _per_thread_default(cls) -> "Stream":
        self = super().__new__(cls)
        self._mnff = Stream._MembersNeededForFinalize(self, driver.CUstream(driver.CU_STREAM_PER_THREAD), None, True)
        self._nonblocking = None  # delayed
        self._priority = None  # delayed
        self._device_id = None  # delayed
        self._ctx_handle = None  # delayed
        return self

    @classmethod
    def _init(cls, obj: Optional[Any] = None, *, options: Optional[StreamOptions] = None) -> "Stream":
        self = super().__new__(cls)
        self._mnff = Stream._MembersNeededForFinalize(self, None, None, False)

        if obj is not None and options is not None:
            raise ValueError("obj and options cannot be both specified")
        if obj is not None:
            cuda_stream_attr = getattr(obj, "__cuda_stream__", None)
            if cuda_stream_attr is None:
                raise TypeError(f"{type(obj)} object does not have a '__cuda_stream__' attribute")
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
            except Exception as e:
                raise RuntimeError(
                    f"obj.__cuda_stream__ must return a sequence with 2 elements, got {type(info)}"
                ) from e
            if len_info != 2:
                raise RuntimeError(
                    f"obj.__cuda_stream__ must return a sequence with 2 elements, got {len_info} elements"
                )
            if info[0] != 0:
                raise RuntimeError(
                    f"The first element of the sequence returned by obj.__cuda_stream__ must be 0, got {repr(info[0])}"
                )

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

    def close(self) -> None:
        """Destroy the stream.

        Destroy the stream if we own it. Borrowed foreign stream
        object will instead have their references released.

        """
        self._mnff.close()

    def __cuda_stream__(self) -> Tuple[int, int]:
        """Return an instance of a __cuda_stream__ protocol."""
        return (0, self.handle)

    @property
    def handle(self) -> driver.CUstream:
        """Return the underlying ``CUstream`` object."""
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

    def sync(self) -> None:
        """Synchronize the stream."""
        handle_return(driver.cuStreamSynchronize(self._mnff.handle))

    def record(self, event: Optional[Event] = None, options: Optional[EventOptions] = None) -> Event:
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
        assert_type(event, Event)
        handle_return(driver.cuEventRecord(event.handle, self._mnff.handle))
        return event

    def wait(self, event_or_stream: Union[Event, "Stream"]) -> None:
        """Wait for a CUDA event or a CUDA stream.

        Waiting for an event or a stream establishes a stream order.

        Parameters
        ----------
        event_or_stream : Union[:obj:`~_event.Event`, :obj:`~_stream.Stream`]
            The event or stream to wait for.

        """
        if isinstance(event_or_stream, Event):
            handle_return(driver.cuStreamWaitEvent(self._mnff.handle, event_or_stream.handle, 0))
        else:
            handle_return(driver.cuStreamWaitStream(self._mnff.handle, event_or_stream.handle, 0))

    @property
    def device(self) -> "Device":
        """Return the device associated with this stream."""
        if self._device_id is None:
            self._device_id = int(handle_return(driver.cuCtxGetDevice()))
        return Device(self._device_id)

    @property
    def context(self) -> Context:
        """Return the context associated with this stream."""
        if self._ctx_handle is None:
            self._ctx_handle = handle_return(driver.cuCtxGetCurrent())
        return Context._from_ctx(self._ctx_handle, self._device_id)

    @staticmethod
    def from_handle(handle: int) -> "Stream":
        """Create a Stream object from an existing CUDA stream handle.

        Parameters
        ----------
        handle : int
            The CUDA stream handle.

        Returns
        -------
        :obj:`~_stream.Stream`
            A new Stream object.

        """
        class _stream_holder:
            def __cuda_stream__(self) -> Tuple[int, int]:
                return (0, handle)

        return Stream._init(_stream_holder())


LEGACY_DEFAULT_STREAM = Stream._legacy_default()
PER_THREAD_DEFAULT_STREAM = Stream._per_thread_default()


def default_stream() -> Stream:
    """Return the default stream.

    The type of default stream returned depends on if the environment
    variable CUDA_PYTHON_CUDA_PER_THREAD_DEFAULT_STREAM is set.

    If set, returns a per-thread default stream. Otherwise returns
    the legacy stream.

    Returns
    -------
    :obj:`~_stream.Stream`
        The default stream.

    """
    if os.environ.get("CUDA_PYTHON_CUDA_PER_THREAD_DEFAULT_STREAM") is not None:
        return Stream._per_thread_default()
    return Stream._legacy_default()
