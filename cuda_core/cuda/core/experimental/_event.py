# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

from dataclasses import dataclass
from typing import Optional

from cuda import cuda
from cuda.core.experimental._utils import CUDAError, check_or_create_options, handle_return


@dataclass
class EventOptions:
    """Customizable :obj:`Event` options.

    Attributes
    ----------
    enable_timing : bool, optional
        Event will record timing data. (Default to False)
    busy_waited_sync : bool, optional
        If True, event will use blocking synchronization. When a CPU
        thread calls synchronize, the call will block until the event
        has actually been completed.
        Otherwise, the CPU thread will busy-wait until the event has
        been completed. (Default to False)
    support_ipc : bool, optional
        Event will be suitable for interprocess use.
        Note that enable_timing must be False. (Default to False)

    """

    enable_timing: Optional[bool] = False
    busy_waited_sync: Optional[bool] = False
    support_ipc: Optional[bool] = False


class Event:
    """Represent a record at a specific point of execution within a CUDA stream.

    Applications can asynchronously record events at any point in
    the program. An event keeps a record of all previous work within
    the last recorded stream.

    Events can be used to monitor device's progress, query completion
    of work up to event's record, and help establish dependencies
    between GPU work submissions.

    Directly creating an :obj:`Event` is not supported due to ambiguity,
    and they should instead be created through a :obj:`Stream` object.

    """

    __slots__ = (
        "_handle",
        "_timing_disabled",
        "_busy_waited"
        )

    def __init__(self):
        self._handle = None
        raise NotImplementedError(
            "directly creating an Event object can be ambiguous. Please call " "call Stream.record()."
        )

    @staticmethod
    def _init(options: Optional[EventOptions] = None):
        self = Event.__new__(Event)
        # minimal requirements for the destructor
        self._handle = None

        options = check_or_create_options(EventOptions, options, "Event options")
        flags = 0x0
        self._timing_disabled = False
        self._busy_waited = False
        if not options.enable_timing:
            flags |= cuda.CUevent_flags.CU_EVENT_DISABLE_TIMING
            self._timing_disabled = True
        if options.busy_waited_sync:
            flags |= cuda.CUevent_flags.CU_EVENT_BLOCKING_SYNC
            self._busy_waited = True
        if options.support_ipc:
            raise NotImplementedError("TODO")
        self._handle = handle_return(cuda.cuEventCreate(flags))
        return self

    def __del__(self):
        """Return close(self)"""
        self.close()

    def close(self):
        """Destroy the event."""
        if self._handle:
            handle_return(cuda.cuEventDestroy(self._handle))
            self._handle = None

    @property
    def is_timing_disabled(self) -> bool:
        """Return True if the event does not record timing data, otherwise False."""
        return self._timing_disabled

    @property
    def is_sync_busy_waited(self) -> bool:
        """Return True if the event synchronization would keep the CPU busy-waiting, otherwise False."""
        return self._busy_waited

    @property
    def is_ipc_supported(self) -> bool:
        """Return True if this event can be used as an interprocess event, otherwise False."""
        raise NotImplementedError("TODO")

    def sync(self):
        """Synchronize until the event completes.

        If the event was created with busy_waited_sync, then the
        calling CPU thread will block until the event has been
        completed by the device.
        Otherwise the CPU thread will busy-wait until the event
        has been completed.

        """
        handle_return(cuda.cuEventSynchronize(self._handle))

    @property
    def is_done(self) -> bool:
        """Return True if all captured works have been completed, otherwise False."""
        (result,) = cuda.cuEventQuery(self._handle)
        if result == cuda.CUresult.CUDA_SUCCESS:
            return True
        elif result == cuda.CUresult.CUDA_ERROR_NOT_READY:
            return False
        else:
            raise CUDAError(f"unexpected error: {result}")

    @property
    def handle(self) -> int:
        """Return the underlying cudaEvent_t pointer address as Python int."""
        return int(self._handle)
