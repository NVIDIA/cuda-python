# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

from __future__ import annotations

import weakref
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from cuda.core.experimental._utils.cuda_utils import CUDAError, check_or_create_options, driver, handle_return

if TYPE_CHECKING:
    import cuda.bindings


@dataclass
class EventOptions:
    """Customizable :obj:`~_event.Event` options.

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
    of work up to event's record, help establish dependencies
    between GPU work submissions, and record the elapsed time (in milliseconds)
    on GPU:

    .. code-block:: python

        # To create events and record the timing:
        s = Device().create_stream()
        e1 = Device().create_event({"enable_timing": True})
        e2 = Device().create_event({"enable_timing": True})
        s.record(e1)
        # ... run some GPU works ...
        s.record(e2)
        e2.sync()
        print(f"time = {e2 - e1} milliseconds")

    Directly creating an :obj:`~_event.Event` is not supported due to ambiguity,
    and they should instead be created through a :obj:`~_stream.Stream` object.

    """

    class _MembersNeededForFinalize:
        __slots__ = ("handle",)

        def __init__(self, event_obj, handle):
            self.handle = handle
            weakref.finalize(event_obj, self.close)

        def close(self):
            if self.handle is not None:
                handle_return(driver.cuEventDestroy(self.handle))
                self.handle = None

    def __new__(self, *args, **kwargs):
        raise RuntimeError("Event objects cannot be instantiated directly. Please use Stream APIs (record).")

    __slots__ = ("__weakref__", "_mnff", "_timing_disabled", "_busy_waited")

    @classmethod
    def _init(cls, options: Optional[EventOptions] = None):
        self = super().__new__(cls)
        self._mnff = Event._MembersNeededForFinalize(self, None)

        options = check_or_create_options(EventOptions, options, "Event options")
        flags = 0x0
        self._timing_disabled = False
        self._busy_waited = False
        if not options.enable_timing:
            flags |= driver.CUevent_flags.CU_EVENT_DISABLE_TIMING
            self._timing_disabled = True
        if options.busy_waited_sync:
            flags |= driver.CUevent_flags.CU_EVENT_BLOCKING_SYNC
            self._busy_waited = True
        if options.support_ipc:
            raise NotImplementedError("WIP: https://github.com/NVIDIA/cuda-python/issues/103")
        self._mnff.handle = handle_return(driver.cuEventCreate(flags))
        return self

    def close(self):
        """Destroy the event."""
        self._mnff.close()

    def __isub__(self, other):
        return NotImplemented

    def __rsub__(self, other):
        return NotImplemented

    def __sub__(self, other):
        # return self - other (in milliseconds)
        try:
            timing = handle_return(driver.cuEventElapsedTime(other.handle, self.handle))
        except CUDAError as e:
            raise RuntimeError(
                "Timing capability must be enabled in order to subtract two Events; timing is disabled by default."
            ) from e
        return timing

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
        raise NotImplementedError("WIP: https://github.com/NVIDIA/cuda-python/issues/103")

    def sync(self):
        """Synchronize until the event completes.

        If the event was created with busy_waited_sync, then the
        calling CPU thread will block until the event has been
        completed by the device.
        Otherwise the CPU thread will busy-wait until the event
        has been completed.

        """
        handle_return(driver.cuEventSynchronize(self._mnff.handle))

    @property
    def is_done(self) -> bool:
        """Return True if all captured works have been completed, otherwise False."""
        (result,) = driver.cuEventQuery(self._mnff.handle)
        if result == driver.CUresult.CUDA_SUCCESS:
            return True
        if result == driver.CUresult.CUDA_ERROR_NOT_READY:
            return False
        handle_return(result)

    @property
    def handle(self) -> cuda.bindings.driver.CUevent:
        """Return the underlying CUevent object."""
        return self._mnff.handle
