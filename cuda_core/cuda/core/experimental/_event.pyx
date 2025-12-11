# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

cimport cpython
from libc.stdint cimport uintptr_t
from libc.string cimport memcpy
from cuda.bindings cimport cydriver
from cuda.core.experimental._context cimport Context
from cuda.core.experimental._resource_handles cimport (
    ContextHandle,
    EventHandle,
    create_event_handle,
    create_event_handle_ipc,
    intptr,
    native,
    py,
)
from cuda.core.experimental._utils.cuda_utils cimport (
    check_or_create_options,
    HANDLE_RETURN
)

import cython
from dataclasses import dataclass
import multiprocessing
from typing import TYPE_CHECKING, Optional
from cuda.core.experimental._utils.cuda_utils import (
    CUDAError,
    check_multiprocessing_start_method,
    driver,
)
if TYPE_CHECKING:
    import cuda.bindings


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
    ipc_enabled : bool, optional
        Event will be suitable for interprocess use.
        Note that enable_timing must be False. (Default to False)

    """

    enable_timing: Optional[bool] = False
    busy_waited_sync: Optional[bool] = False
    ipc_enabled: Optional[bool] = False


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

    def __init__(self, *args, **kwargs):
        raise RuntimeError("Event objects cannot be instantiated directly. Please use Stream APIs (record).")

    @staticmethod
    cdef Event _init(type cls, int device_id, ContextHandle h_context, options, bint is_free):
        cdef Event self = cls.__new__(cls)
        cdef EventOptions opts = check_or_create_options(EventOptions, options, "Event options")
        cdef unsigned int flags = 0x0
        self._timing_disabled = False
        self._busy_waited = False
        self._ipc_enabled = False
        self._ipc_descriptor = None
        if not opts.enable_timing:
            flags |= cydriver.CUevent_flags.CU_EVENT_DISABLE_TIMING
            self._timing_disabled = True
        if opts.busy_waited_sync:
            flags |= cydriver.CUevent_flags.CU_EVENT_BLOCKING_SYNC
            self._busy_waited = True
        if opts.ipc_enabled:
            if is_free:
                raise TypeError(
                    "IPC-enabled events must be bound; use Stream.record for creation."
                )
            flags |= cydriver.CUevent_flags.CU_EVENT_INTERPROCESS
            self._ipc_enabled = True
            if not self._timing_disabled:
                raise TypeError("IPC-enabled events cannot use timing.")
        # C++ creates the event and returns owning handle with context dependency
        cdef EventHandle h_event = create_event_handle(h_context, flags)
        if not h_event:
            raise RuntimeError("Failed to create CUDA event")
        self._h_event = h_event
        self._h_context = h_context
        self._device_id = device_id
        if opts.ipc_enabled:
            self.get_ipc_descriptor()
        return self

    cpdef close(self):
        """Destroy the event.

        Releases the event handle. The underlying CUDA event is destroyed
        when the last reference is released.
        """
        self._h_event.reset()

    def __isub__(self, other):
        return NotImplemented

    def __rsub__(self, other):
        return NotImplemented

    def __sub__(self, other: Event):
        # return self - other (in milliseconds)
        cdef float timing
        with nogil:
            err = cydriver.cuEventElapsedTime(&timing, native((<Event>other)._h_event), native(self._h_event))
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

    def __hash__(self) -> int:
        return hash((type(self), intptr(self._h_context), intptr(self._h_event)))

    def __eq__(self, other) -> bool:
        # Note: using isinstance because `Event` can be subclassed.
        if not isinstance(other, Event):
            return NotImplemented
        cdef Event _other = <Event>other
        return intptr(self._h_event) == intptr(_other._h_event)

    def get_ipc_descriptor(self) -> IPCEventDescriptor:
        """Export an event allocated for sharing between processes."""
        if self._ipc_descriptor is not None:
            return self._ipc_descriptor
        if not self.is_ipc_enabled:
            raise RuntimeError("Event is not IPC-enabled")
        cdef cydriver.CUipcEventHandle data
        with nogil:
            HANDLE_RETURN(cydriver.cuIpcGetEventHandle(&data, native(self._h_event)))
        cdef bytes data_b = cpython.PyBytes_FromStringAndSize(<char*>(data.reserved), sizeof(data.reserved))
        self._ipc_descriptor = IPCEventDescriptor._init(data_b, self._busy_waited)
        return self._ipc_descriptor

    @classmethod
    def from_ipc_descriptor(cls, ipc_descriptor: IPCEventDescriptor) -> Event:
        """Import an event that was exported from another process."""
        cdef cydriver.CUipcEventHandle data
        memcpy(data.reserved, <const void*><const char*>(ipc_descriptor._reserved), sizeof(data.reserved))
        cdef Event self = Event.__new__(cls)
        # IPC events: the originating process owns the event and its context
        cdef EventHandle h_event = create_event_handle_ipc(data)
        if not h_event:
            raise RuntimeError("Failed to open IPC event handle")
        self._h_event = h_event
        self._h_context = ContextHandle()
        self._timing_disabled = True
        self._busy_waited = ipc_descriptor._busy_waited
        self._ipc_enabled = True
        self._ipc_descriptor = ipc_descriptor
        self._device_id = -1
        return self

    @property
    def is_ipc_enabled(self) -> bool:
        """Return True if the event can be shared across process boundaries, otherwise False."""
        return self._ipc_enabled

    @property
    def is_timing_disabled(self) -> bool:
        """Return True if the event does not record timing data, otherwise False."""
        return self._timing_disabled

    @property
    def is_sync_busy_waited(self) -> bool:
        """Return True if the event synchronization would keep the CPU busy-waiting, otherwise False."""
        return self._busy_waited

    def sync(self):
        """Synchronize until the event completes.

        If the event was created with busy_waited_sync, then the
        calling CPU thread will block until the event has been
        completed by the device.
        Otherwise the CPU thread will busy-wait until the event
        has been completed.

        """
        with nogil:
            HANDLE_RETURN(cydriver.cuEventSynchronize(native(self._h_event)))

    @property
    def is_done(self) -> bool:
        """Return True if all captured works have been completed, otherwise False."""
        with nogil:
            result = cydriver.cuEventQuery(native(self._h_event))
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
        return py(self._h_event)

    @property
    def device(self) -> Device:
        """Return the :obj:`~_device.Device` singleton associated with this event.

        Note
        ----
        The current context on the device may differ from this
        event's context. This case occurs when a different CUDA
        context is set current after a event is created.

        """
        if self._device_id >= 0:
            from ._device import Device  # avoid circular import
            return Device(self._device_id)

    @property
    def context(self) -> Context:
        """Return the :obj:`~_context.Context` associated with this event."""
        if self._h_context and self._device_id >= 0:
            return Context._from_handle(Context, self._h_context, self._device_id)


cdef class IPCEventDescriptor:
    """Serializable object describing an event that can be shared between processes."""

    cdef:
        bytes _reserved
        bint _busy_waited

    def __init__(self, *arg, **kwargs):
        raise RuntimeError("IPCEventDescriptor objects cannot be instantiated directly. Please use Event APIs.")

    @classmethod
    def _init(cls, reserved: bytes, busy_waited: cython.bint):
        cdef IPCEventDescriptor self = IPCEventDescriptor.__new__(cls)
        self._reserved = reserved
        self._busy_waited = busy_waited
        return self

    def __eq__(self, IPCEventDescriptor rhs):
        # No need to check self._busy_waited.
        return self._reserved == rhs._reserved

    def __reduce__(self):
        return self._init, (self._reserved, self._busy_waited)


def _reduce_event(event):
    check_multiprocessing_start_method()
    return event.from_ipc_descriptor, (event.get_ipc_descriptor(),)

multiprocessing.reduction.register(Event, _reduce_event)
