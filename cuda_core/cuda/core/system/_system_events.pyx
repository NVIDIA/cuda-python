# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0


from libc.stdint cimport intptr_t

import sys
if sys.version_info >= (3, 11):
    from enum import StrEnum
else:
    from backports.strenum import StrEnum

from cuda.bindings import nvml

from ._nvml_context cimport initialize

from . import _device


class SystemEventType(StrEnum):
    """
    System event types.
    """
    UNBIND = "unbind"
    BIND = "bind"
cdef dict _SYSTEM_EVENT_TYPE_MAPPING = {
    nvml.SystemEventType.GPU_DRIVER_UNBIND: SystemEventType.UNBIND,
    nvml.SystemEventType.GPU_DRIVER_BIND: SystemEventType.BIND,
}
cdef dict _SYSTEM_EVENT_TYPE_INV_MAPPING = {v: k for k, v in _SYSTEM_EVENT_TYPE_MAPPING.items()}


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
        The :obj:`~SystemEventType` that was triggered.
        """
        return _SYSTEM_EVENT_TYPE_MAPPING[self._event_data.event_type]

    @property
    def gpu_id(self) -> int:
        """
        The GPU ID in PCI ID format.
        """
        return self._event_data.gpu_id

    @property
    def device(self) -> _device.Device:
        """
        The :obj:`~_device.Device` associated with this event.
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
        """
        Get the :obj:`~_system_events.SystemEvent` at the specified index.
        """
        return SystemEvent(self._event_data[idx])


cdef class RegisteredSystemEvents:
    """
    Represents a set of events that can be waited on for a specific device.
    """
    cdef intptr_t _event_set

    def __init__(self, events: SystemEventType | str | list[SystemEventType | str]):
        cdef unsigned long long event_bitmask
        if isinstance(events, (str, SystemEventType)):
            events = [events]

        if isinstance(events, list):
            event_bitmask = 0
            for ev in events:
                try:
                    ev_enum = _SYSTEM_EVENT_TYPE_INV_MAPPING[ev]
                except KeyError:
                    raise ValueError(
                        f"Invalid event type: {ev}. "
                        f"Must be one of {list(SystemEventType.__members__.values())}"
                    ) from None
                event_bitmask |= <unsigned long long>int(ev_enum)
        else:
            raise TypeError("events must be an SystemEventType, str, or list of SystemEventType or str")

        initialize()

        self._event_set = 0
        self._event_set = nvml.system_event_set_create()
        # If this raises, the event needs to be freed and this is handled by
        # this class's __dealloc__ method.
        nvml.system_register_events(event_bitmask, self._event_set)

    def __dealloc__(self):
        if self._event_set != 0:
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

        Returns
        -------
        :obj:`~_system_events.SystemEvents`
            A set of events that were received.  The number of events returned may
            be less than the specified buffer size if fewer events were available.

        Raises
        ------
        :class:`cuda.core.system.TimeoutError`
            If the timeout expires before an event is received.
        :class:`cuda.core.system.GpuIsLostError`
            If the GPU has fallen off the bus or is otherwise inaccessible.
        """
        return SystemEvents(nvml.system_event_set_wait(self._event_set, timeout_ms, buffer_size))


def register_events(events: SystemEventType | str | list[SystemEventType | str]) -> RegisteredSystemEvents:
    """
    Starts recording of events on test system.

    For Linux only.

    All events that occurred before this call are not recorded.  Wait for events
    using the :meth:`RegisteredSystemEvents.wait` method on the result.

    Examples
    --------
    >>> from cuda.core import system
    >>> events = system.register_events([SystemEventType.UNBIND])
    >>> while event := events.wait(timeout_ms=10000):
    ...     print(f"Event {event.event_type} occurred.")

    Parameters
    ----------
    events: SystemEventType, str, or list of SystemEventType or str
        The event type or list of event types to register for this device.

    Returns
    -------
    :obj:`~_system_events.RegisteredSystemEvents`
        An object representing the registered events.  Call
        :meth:`~_system_events.RegisteredSystemEvents.wait` on this object to wait for events.

    Raises
    ------
    :class:`cuda.core.system.NotSupportedError`
        None of the requested event types are registered.
    """
    return RegisteredSystemEvents(events)


__all__ = [
    "register_events",
    "SystemEventType",
]
