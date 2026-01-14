# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0


from libc.stdint cimport intptr_t

from cuda.bindings import _nvml as nvml

from ._nvml_context cimport initialize

from . import _device


SystemEventType = nvml.SystemEventType


cdef class SystemEvent:
    """
    Data about a collection of system events.
    """
    def __init__(self, event_data: nvml.SystemEventData_v1):
        assert len(event_data) == 1
        self._event_data = event_data

    @property
    def event_type(self) -> SystemEventType:
        """
        The type of event that was triggered.
        """
        return SystemEventType(self._event_data.event_type)

    @property
    def gpu_id(self) -> int:
        """
        The GPU ID in PCI ID format.
        """
        return self._event_data.gpu_id

    @property
    def device(self) -> _device.Device:
        """
        The device associated with this event.
        """
        return _device.Device(pci_bus_id=self.gpu_id)


cdef class SystemEvents:
    """
    Data about a collection of system events.
    """
    def __init__(self, event_data: nvml.SystemEventData_v1):
        self._event_data = event_data

    def __len__(self):
        return len(self._event_data)

    def __getitem__(self, idx: int) -> SystemEvent:
        return SystemEvent(self._event_data[idx])


cdef class RegisteredSystemEvents:
    """
    Represents a set of events that can be waited on for a specific device.
    """
    cdef intptr_t _event_set

    def __init__(self, events: SystemEventType | int | list[SystemEventType | int]):
        cdef unsigned long long event_bitmask
        if isinstance(events, (int, SystemEventType)):
            event_bitmask = <unsigned long long>int(events)
        elif isinstance(events, list):
            event_bitmask = 0
            for ev in events:
                event_bitmask |= <unsigned long long>int(ev)
        else:
            raise TypeError("events must be an SystemEventType, int, or list of SystemEventType or int")

        initialize()

        self._event_set = nvml.system_event_set_create()
        print("event set:", self._event_set)
        # If this raises, the event needs to be freed and this is handled by
        # this class's __dealloc__ method.
        nvml.system_register_events(event_bitmask, self._event_set)

    def __dealloc__(self):
        nvml.system_event_set_free(self._event_set)

    def wait(self, timeout_ms: int = 0, buffer_size: int = 1) -> SystemEvents:
        """
        Wait for events in the system event set.

        For Fermi™ or newer fully supported devices.

        If some events are ready to be delivered at the time of the call,
        function returns immediately.  If there are no events ready to be
        delivered, function sleeps till event arrives but not longer than
        specified timeout. If timeout passes, a
        :class:`cuda.core.system.TimeoutError` is raised.  This function in
        certain conditions can return before specified timeout passes (e.g. when
        interrupt arrives)

        Parameters
        ----------
        timeout_ms: int
            The timeout in milliseconds. A value of 0 means to wait indefinitely.
        buffer_size: int
            The maximum number of events to retrieve.  Must be at least 1.

        Raises
        ------
        :class:`cuda.core.system.TimeoutError`
            If the timeout expires before an event is received.
        :class:`cuda.core.system.GpuIsLostError`
            If the GPU has fallen off the bus or is otherwise inaccessible.
        """
        return SystemEvents(nvml.system_event_set_wait(self._event_set, timeout_ms, buffer_size))


def register_events(events: SystemEventType | int | list[SystemEventType | int]) -> RegisteredSystemEvents:
    """
    Starts recording of events on test system.

    For Linux only.

    All events that occurred before this call are not recorded.  Wait for events
    using the :meth:`RegisteredSystemEvents.wait` method on the result.

    Examples
    --------
    >>> from cuda.core import system
    >>> events = system.register_events([
    ...     SystemEventType.SYSTEM_EVENT_TYPE_GPU_DRIVER_UNBIND,
    ... ])
    >>> while event := events.wait(timeout_ms=10000):
    ...     print(f"Event {event.event_type} occurred.")

    Parameters
    ----------
    events: SystemEventType, int, or list of SystemEventType or int
        The event type or list of event types to register for this device.

    Returns
    -------
    :class:`RegisteredSystemEvents`
        An object representing the registered events.  Call
        :meth:`RegisteredSystemEvents.wait` on this object to wait for events.

    Raises
    ------
    :class:`cuda.core.system.NotSupportedError`
        None of the requested event types are registered.
    """
    return RegisteredSystemEvents(events)


__all__ = [
    "register_events",
    "RegisteredSystemEvents",
    "SystemEvent",
    "SystemEvents",
    "SystemEventType",
]
