# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

cimport cpython
from libc.string cimport memcpy
from cuda.bindings cimport cydriver
from cuda.core._context cimport Context
from cuda.core._resource_handles cimport (
    ContextHandle,
    EventHandle,
    create_event_handle,
    create_event_handle_ipc,
    get_event_timing_enabled,
    get_event_uses_blocking_sync,
    get_event_ipc_enabled,
    get_event_device_id,
    get_event_context,
    as_intptr,
    as_cu,
    as_py,
)

from cuda.core._utils.cuda_utils cimport (
    check_or_create_options,
    HANDLE_RETURN
)

import cython
from dataclasses import dataclass
import multiprocessing

from cuda.core._utils.cuda_utils import (
    CUDAError,
    check_multiprocessing_start_method,
)


@dataclass
cdef class EventOptions:
    """Customizable :obj:`~_event.Event` options.

    Attributes
    ----------
    timing_enabled : bool, optional
        Event will record timing data. (Default to False)
    blocking_sync : bool, optional
        If True, the event uses blocking synchronization: a CPU
        thread that calls :meth:`Event.sync` blocks (yields) until
        the event has completed. Otherwise (the default), the CPU
        thread busy-waits until the event has completed.
        (Default to False)
    ipc_enabled : bool, optional
        Event will be suitable for interprocess use.
        Note that timing_enabled must be False. (Default to False)

    """

    timing_enabled: bool | None = False
    blocking_sync: bool | None = False
    ipc_enabled: bool | None = False


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
        e1 = Device().create_event({"timing_enabled": True})
        e2 = Device().create_event({"timing_enabled": True})
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
        cdef bint timing_enabled = True
        cdef bint uses_blocking_sync = False
        cdef bint ipc_enabled = False
        self._ipc_descriptor = None
        if not opts.timing_enabled:
            flags |= cydriver.CUevent_flags.CU_EVENT_DISABLE_TIMING
            timing_enabled = False
        if opts.blocking_sync:
            flags |= cydriver.CUevent_flags.CU_EVENT_BLOCKING_SYNC
            uses_blocking_sync = True
        if opts.ipc_enabled:
            if is_free:
                raise TypeError(
                    "IPC-enabled events must be bound; use Stream.record for creation."
                )
            flags |= cydriver.CUevent_flags.CU_EVENT_INTERPROCESS
            ipc_enabled = True
            if timing_enabled:
                raise TypeError("IPC-enabled events cannot use timing.")
        cdef EventHandle h_event = create_event_handle(
            h_context, flags, timing_enabled, uses_blocking_sync, ipc_enabled, device_id)
        if not h_event:
            raise RuntimeError("Failed to create CUDA event")
        self._h_event = h_event
        if ipc_enabled:
            _ = self.ipc_descriptor  # eagerly populate the descriptor cache
        return self

    @staticmethod
    cdef Event _from_handle(EventHandle h_event):
        """Create an Event wrapping an existing EventHandle.

        Metadata (timing, blocking_sync, ipc, device_id) is read from
        the EventBox via pointer arithmetic — no fields are cached on
        Event.
        """
        cdef Event self = Event.__new__(Event)
        self._h_event = h_event
        self._ipc_descriptor = None
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
            err = cydriver.cuEventElapsedTime(&timing, as_cu((<Event>other)._h_event), as_cu(self._h_event))
        if err == 0:
            return timing
        else:
            if err == cydriver.CUresult.CUDA_ERROR_INVALID_HANDLE:
                if not self.is_timing_enabled or not other.is_timing_enabled:
                    explanation = (
                        "Both Events must be created with timing enabled in order to subtract them; "
                        "use EventOptions(timing_enabled=True) when creating both events."
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
        return hash(as_intptr(self._h_event))

    def __eq__(self, other) -> bool:
        # Note: using isinstance because `Event` can be subclassed.
        if not isinstance(other, Event):
            return NotImplemented
        cdef Event _other = <Event>other
        return as_intptr(self._h_event) == as_intptr(_other._h_event)

    def __repr__(self) -> str:
        return f"<Event handle={as_intptr(self._h_event):#x}>"

    @property
    def ipc_descriptor(self) -> IPCEventDescriptor:
        """Descriptor for sharing this event with other processes."""
        if self._ipc_descriptor is not None:
            return self._ipc_descriptor
        if not self.is_ipc_enabled:
            raise RuntimeError("Event is not IPC-enabled")
        cdef cydriver.CUipcEventHandle data
        with nogil:
            HANDLE_RETURN(cydriver.cuIpcGetEventHandle(&data, as_cu(self._h_event)))
        cdef bytes data_b = cpython.PyBytes_FromStringAndSize(<char*>(data.reserved), sizeof(data.reserved))
        self._ipc_descriptor = IPCEventDescriptor._init(data_b, get_event_uses_blocking_sync(self._h_event))
        return self._ipc_descriptor

    @classmethod
    def from_ipc_descriptor(cls, ipc_descriptor: IPCEventDescriptor) -> Event:
        """Import an event that was exported from another process."""
        cdef cydriver.CUipcEventHandle data
        memcpy(data.reserved, <const void*><const char*>(ipc_descriptor._reserved), sizeof(data.reserved))
        cdef Event self = Event.__new__(cls)
        cdef EventHandle h_event = create_event_handle_ipc(data, ipc_descriptor._uses_blocking_sync)
        if not h_event:
            raise RuntimeError("Failed to open IPC event handle")
        self._h_event = h_event
        self._ipc_descriptor = ipc_descriptor
        return self

    @property
    def is_ipc_enabled(self) -> bool:
        """Return True if the event can be shared across process boundaries, otherwise False."""
        return get_event_ipc_enabled(self._h_event)

    @property
    def is_timing_enabled(self) -> bool:
        """Return True if the event records timing data, otherwise False."""
        return get_event_timing_enabled(self._h_event)

    @property
    def uses_blocking_sync(self) -> bool:
        """Return True if the event uses blocking synchronization (the CPU
        thread blocks on :meth:`sync` instead of busy-waiting), otherwise False.
        """
        return get_event_uses_blocking_sync(self._h_event)

    def sync(self):
        """Synchronize until the event completes.

        If the event was created with ``blocking_sync=True``, the
        calling CPU thread blocks (yields) until the event has been
        completed by the device. Otherwise (the default) the CPU
        thread busy-waits until the event has completed.

        """
        with nogil:
            HANDLE_RETURN(cydriver.cuEventSynchronize(as_cu(self._h_event)))

    @property
    def is_done(self) -> bool:
        """Return True if all captured works have been completed, otherwise False."""
        with nogil:
            result = cydriver.cuEventQuery(as_cu(self._h_event))
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
        return as_py(self._h_event)

    @property
    def device(self) -> Device:
        """Return the :obj:`~_device.Device` singleton associated with this event.

        Note
        ----
        The current context on the device may differ from this
        event's context. This case occurs when a different CUDA
        context is set current after a event is created.

        """
        cdef int dev_id = get_event_device_id(self._h_event)
        if dev_id >= 0:
            from ._device import Device  # avoid circular import
            return Device(dev_id)

    @property
    def context(self) -> Context:
        """Return the :obj:`~_context.Context` associated with this event."""
        cdef ContextHandle h_ctx = get_event_context(self._h_event)
        cdef int dev_id = get_event_device_id(self._h_event)
        if h_ctx and dev_id >= 0:
            return Context._from_handle(Context, h_ctx, dev_id)


cdef class IPCEventDescriptor:
    """Serializable object describing an event that can be shared between processes."""

    cdef:
        bytes _reserved
        bint _uses_blocking_sync

    def __init__(self, *arg, **kwargs):
        raise RuntimeError("IPCEventDescriptor objects cannot be instantiated directly. Please use Event APIs.")

    @staticmethod
    def _init(reserved: bytes, uses_blocking_sync: cython.bint):
        cdef IPCEventDescriptor self = IPCEventDescriptor.__new__(IPCEventDescriptor)
        self._reserved = reserved
        self._uses_blocking_sync = uses_blocking_sync
        return self

    def __eq__(self, IPCEventDescriptor rhs):
        # No need to check self._uses_blocking_sync.
        return self._reserved == rhs._reserved

    def __reduce__(self):
        return IPCEventDescriptor._init, (self._reserved, self._uses_blocking_sync)


def _reduce_event(event):
    check_multiprocessing_start_method()
    return event.from_ipc_descriptor, (event.ipc_descriptor,)

multiprocessing.reduction.register(Event, _reduce_event)
