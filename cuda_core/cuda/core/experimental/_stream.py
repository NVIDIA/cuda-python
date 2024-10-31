# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Optional, Tuple, TYPE_CHECKING, Union

if TYPE_CHECKING:
    from cuda.core.experimental._device import Device
from cuda import cuda, cudart
from cuda.core.experimental._context import Context
from cuda.core.experimental._event import Event, EventOptions
from cuda.core.experimental._utils import check_or_create_options
from cuda.core.experimental._utils import get_device_from_ctx
from cuda.core.experimental._utils import handle_return


@dataclass
class StreamOptions:

    nonblocking: bool = True
    priority: Optional[int] = None


class Stream:

    __slots__ = ("_handle", "_nonblocking", "_priority", "_owner", "_builtin",
                 "_device_id", "_ctx_handle")

    def __init__(self):
        # minimal requirements for the destructor
        self._handle = None
        self._owner = None
        self._builtin = False
        raise NotImplementedError(
            "directly creating a Stream object can be ambiguous. Please either "
            "call Device.create_stream() or, if a stream pointer is already "
            "available from somewhere else, Stream.from_handle()")

    @staticmethod
    def _init(obj=None, *, options: Optional[StreamOptions]=None):
        self = Stream.__new__(Stream)

        # minimal requirements for the destructor
        self._handle = None
        self._owner = None
        self._builtin = False

        if obj is not None and options is not None:
            raise ValueError("obj and options cannot be both specified")
        if obj is not None:
            if not hasattr(obj, "__cuda_stream__"):
                raise ValueError
            info = obj.__cuda_stream__
            assert info[0] == 0
            self._handle = cuda.CUstream(info[1])
            # TODO: check if obj is created under the current context/device
            self._owner = obj
            self._nonblocking = None  # delayed
            self._priority = None  # delayed
            self._device_id = None  # delayed
            self._ctx_handle = None  # delayed
            return self

        options = check_or_create_options(StreamOptions, options, "Stream options")
        nonblocking = options.nonblocking
        priority = options.priority

        if nonblocking:
            flags = cuda.CUstream_flags.CU_STREAM_NON_BLOCKING
        else:
            flags = cuda.CUstream_flags.CU_STREAM_DEFAULT

        if priority is not None:
            high, low = handle_return(
                cudart.cudaDeviceGetStreamPriorityRange())
            if not (low <= priority <= high):
                raise ValueError(f"{priority=} is out of range {[low, high]}")
        else:
            priority = 0

        self._handle = handle_return(
            cuda.cuStreamCreateWithPriority(flags, priority))
        self._owner = None
        self._nonblocking = nonblocking
        self._priority = priority
        # don't defer this because we will have to pay a cost for context
        # switch later
        self._device_id = int(handle_return(cuda.cuCtxGetDevice()))
        self._ctx_handle = None  # delayed
        return self

    def __del__(self):
        self.close()

    def close(self):
        if self._owner is None:
            if self._handle and not self._builtin:
                handle_return(cuda.cuStreamDestroy(self._handle))
        else:
            self._owner = None
        self._handle = None

    @property
    def __cuda_stream__(self) -> Tuple[int, int]:
        return (0, int(self._handle))

    @property
    def handle(self) -> int:
        # Return the underlying cudaStream_t pointer address as Python int.
        return int(self._handle)

    @property
    def is_nonblocking(self) -> bool:
        if self._nonblocking is None:
            flag = handle_return(cuda.cuStreamGetFlags(self._handle))
            if flag == cuda.CUstream_flags.CU_STREAM_NON_BLOCKING:
                self._nonblocking = True
            else:
                self._nonblocking = False
        return self._nonblocking

    @property
    def priority(self) -> int:
        if self._priority is None:
            prio = handle_return(cuda.cuStreamGetPriority(self._handle))
            self._priority = prio
        return self._priority

    def sync(self):
        handle_return(cuda.cuStreamSynchronize(self._handle))

    def record(self, event: Event=None, options: EventOptions=None) -> Event:
        # Create an Event object (or reusing the given one) by recording
        # on the stream. Event flags such as disabling timing, nonblocking,
        # and CU_EVENT_RECORD_EXTERNAL, can be set in EventOptions.
        if event is None:
            event = Event._init(options)
        elif not isinstance(event, Event):
            raise TypeError("record only takes an Event object")
        handle_return(cuda.cuEventRecord(event.handle, self._handle))
        return event

    def wait(self, event_or_stream: Union[Event, Stream]):
        # Wait for a CUDA event or a CUDA stream to establish a stream order.
        #
        # If a Stream instance is provided, the effect is as if an event is
        # recorded on the given stream, and then self waits on the recorded
        # event.
        if isinstance(event_or_stream, Event):
            event = event_or_stream.handle
            discard_event = False
        else:
            if not isinstance(event_or_stream, Stream):
                try:
                    stream = Stream._init(event_or_stream)
                except Exception as e:
                    raise ValueError(
                        "only an Event, Stream, or object supporting "
                        "__cuda_stream__ can be waited") from e
            else:
                stream = event_or_stream
            event = handle_return(
                cuda.cuEventCreate(cuda.CUevent_flags.CU_EVENT_DISABLE_TIMING))
            handle_return(cuda.cuEventRecord(event, stream.handle))
            discard_event = True

        # TODO: support flags other than 0?
        handle_return(cuda.cuStreamWaitEvent(self._handle, event, 0))
        if discard_event:
            handle_return(cuda.cuEventDestroy(event))

    @property
    def device(self) -> Device:
        # Inverse look-up to find on which device this stream instance was
        # created.
        #
        # Note that Stream.device.context might not necessarily agree with
        # Stream.context, in cases where a different CUDA context is set
        # current after a stream was created.
        from cuda.core.experimental._device import Device  # avoid circular import
        if self._device_id is None:
            # Get the stream context first
            if self._ctx_handle is None:
                self._ctx_handle = handle_return(
                    cuda.cuStreamGetCtx(self._handle))
            self._device_id = get_device_from_ctx(self._ctx_handle)
        return Device(self._device_id)

    @property
    def context(self) -> Context:
        # Inverse look-up to find in which CUDA context this stream instance
        # was created
        if self._ctx_handle is None:
            self._ctx_handle = handle_return(
                cuda.cuStreamGetCtx(self._handle))
        if self._device_id is None:
            self._device_id = get_device_from_ctx(self._ctx_handle)
        return Context._from_ctx(self._ctx_handle, self._device_id)

    @staticmethod
    def from_handle(handle: int) -> Stream:
        class _stream_holder:
            @property
            def __cuda_stream__(self):
                return (0, handle)
        return Stream._init(obj=_stream_holder())


class _LegacyDefaultStream(Stream):

    def __init__(self):
        self._handle = cuda.CUstream(cuda.CU_STREAM_LEGACY)
        self._owner = None
        self._nonblocking = None  # delayed
        self._priority = None  # delayed
        self._builtin = True


class _PerThreadDefaultStream(Stream):

    def __init__(self):
        self._handle = cuda.CUstream(cuda.CU_STREAM_PER_THREAD)
        self._owner = None
        self._nonblocking = None  # delayed
        self._priority = None  # delayed
        self._builtin = True


LEGACY_DEFAULT_STREAM = _LegacyDefaultStream()
PER_THREAD_DEFAULT_STREAM = _PerThreadDefaultStream()


def default_stream():
    # TODO: flip the default
    use_ptds = int(os.environ.get('CUDA_PYTHON_CUDA_PER_THREAD_DEFAULT_STREAM', 0))
    if use_ptds:
        return PER_THREAD_DEFAULT_STREAM
    else:
        return LEGACY_DEFAULT_STREAM
