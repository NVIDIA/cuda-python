# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from libc.stdint cimport uintptr_t

from cuda.bindings cimport cydriver

from cuda.core.experimental._utils.cuda_utils cimport (
    check_or_create_options,
    HANDLE_RETURN
)

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from cuda.core.experimental._context import Context
from cuda.core.experimental._utils.cuda_utils import (
    CUDAError,
    driver,
)
if TYPE_CHECKING:
    import cuda.bindings
    from cuda.core.experimental._device import Device


@dataclass
cdef class EventOptions:
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


cdef class Event:
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
    def __cinit__(self):
        self._handle = <cydriver.CUevent>(NULL)

    def __init__(self, *args, **kwargs):
        raise RuntimeError("Event objects cannot be instantiated directly. Please use Stream APIs (record).")

    @classmethod
    def _init(cls, device_id: int, ctx_handle: Context, options=None):
        cdef Event self = Event.__new__(cls)
        cdef EventOptions opts = check_or_create_options(EventOptions, options, "Event options")
        cdef unsigned int flags = 0x0
        self._timing_disabled = False
        self._busy_waited = False
        if not opts.enable_timing:
            flags |= cydriver.CUevent_flags.CU_EVENT_DISABLE_TIMING
            self._timing_disabled = True
        if opts.busy_waited_sync:
            flags |= cydriver.CUevent_flags.CU_EVENT_BLOCKING_SYNC
            self._busy_waited = True
        if opts.support_ipc:
            raise NotImplementedError("WIP: https://github.com/NVIDIA/cuda-python/issues/103")
        with nogil:
            HANDLE_RETURN(cydriver.cuEventCreate(&self._handle, flags))
        self._device_id = device_id
        self._ctx_handle = ctx_handle
        return self

    cpdef close(self):
        """Destroy the event."""
        if self._handle != NULL:
            with nogil:
                HANDLE_RETURN(cydriver.cuEventDestroy(self._handle))
            self._handle = <cydriver.CUevent>(NULL)

    def __dealloc__(self):
        self.close()

    def __isub__(self, other):
        return NotImplemented

    def __rsub__(self, other):
        return NotImplemented

    def __sub__(self, other: Event):
        # return self - other (in milliseconds)
        cdef float timing
        with nogil:
            err = cydriver.cuEventElapsedTime(&timing, other._handle, self._handle)
        if err == 0:
            return timing
        else:
            if err == cydriver.CUresult.CUDA_ERROR_INVALID_HANDLE:
                if self.is_timing_disabled or other.is_timing_disabled:
                    explanation = (
                        "Both Events must be created with timing enabled in order to subtract them; "
                        "use EventOptions(enable_timing=True) when creating both events."
                    )
                else:
                    explanation = (
                        "Both Events must be recorded before they can be subtracted; "
                        "use Stream.record() to record both events to a stream."
                    )
            elif err == cydriver.CUresult.CUDA_ERROR_NOT_READY:
                explanation = (
                    "One or both events have not completed; "
                    "use Event.sync(), Stream.sync(), or Device.sync() to wait for the events to complete "
                    "before subtracting them."
                )
            else:
                raise CUDAError(err)
            raise RuntimeError(explanation)

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
        with nogil:
            HANDLE_RETURN(cydriver.cuEventSynchronize(self._handle))

    @property
    def is_done(self) -> bool:
        """Return True if all captured works have been completed, otherwise False."""
        with nogil:
            result = cydriver.cuEventQuery(self._handle)
        if result == cydriver.CUresult.CUDA_SUCCESS:
            return True
        if result == cydriver.CUresult.CUDA_ERROR_NOT_READY:
            return False
        HANDLE_RETURN(result)

    @property
    def handle(self) -> cuda.bindings.driver.CUevent:
        """Return the underlying CUevent object.

        .. caution::

            This handle is a Python object. To get the memory address of the underlying C
            handle, call ``int(Event.handle)``.
        """
        return driver.CUevent(<uintptr_t>(self._handle))

    @property
    def device(self) -> Device:
        """Return the :obj:`~_device.Device` singleton associated with this event.

        Note
        ----
        The current context on the device may differ from this
        event's context. This case occurs when a different CUDA
        context is set current after a event is created.

        """

        from cuda.core.experimental._device import Device  # avoid circular import

        return Device(self._device_id)

    @property
    def context(self) -> Context:
        """Return the :obj:`~_context.Context` associated with this event."""
        return Context._from_ctx(self._ctx_handle, self._device_id)
