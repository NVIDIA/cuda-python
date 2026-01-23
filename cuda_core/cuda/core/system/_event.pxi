# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0


EventType = nvml.EventType


cdef class EventData:
    """
    Data about a single event.
    """
    def __init__(self, event_data: nvml.EventData):
        self._event_data = event_data

    @property
    def device(self) -> Device:
        """
        The device on which the event occurred.
        """
        device = Device.__new__()
        device._handle = self._event_data.device
        return device

    @property
    def event_type(self) -> EventType:
        """
        The type of event that was triggered.
        """
        return EventType(self._event_data.event_type)

    @property
    def event_data(self) -> int:
        """
        Returns Xid error for the device in the event of
        :member:`EventType.EVENT_TYPE_XID_CRITICAL_ERROR`.

        Raises :class:`ValueError` for other event types.
        """
        if self.event_type != EventType.EVENT_TYPE_XID_CRITICAL_ERROR:
            raise ValueError("event_data is only available for Xid critical error events.")
        return self._event_data.event_data

    @property
    def gpu_instance_id(self) -> int:
        """
        The GPU instance ID for MIG devices.

        Only valid for events of type :attr:`EventType.EVENT_TYPE_XID_CRITICAL_ERROR`.

        Raises :class:`ValueError` for other event types.
        """
        if self.event_type != EventType.EVENT_TYPE_XID_CRITICAL_ERROR:
            raise ValueError("gpu_instance_id is only available for Xid critical error events.")
        return self._event_data.gpu_instance_id

    @property
    def compute_instance_id(self) -> int:
        """
        The Compute instance ID for MIG devices.

        Only valid for events of type :attr:`EventType.EVENT_TYPE_XID_CRITICAL_ERROR`.

        Raises :class:`ValueError` for other event types.
        """
        if self.event_type != EventType.EVENT_TYPE_XID_CRITICAL_ERROR:
            raise ValueError("compute_instance_id is only available for Xid critical error events.")
        return self._event_data.compute_instance_id


cdef class DeviceEvents:
    """
    Represents a set of events that can be waited on for a specific device.
    """
    cdef intptr_t _event_set
    cdef intptr_t _device_handle

    def __init__(self, device_handle: intptr_t, events: EventType | int | list[EventType | int]):
        cdef unsigned long long event_bitmask
        if isinstance(events, (int, EventType)):
            event_bitmask = <unsigned long long>int(events)
        elif isinstance(events, list):
            event_bitmask = 0
            for ev in events:
                event_bitmask |= <unsigned long long>int(ev)
        else:
            raise TypeError("events must be an EventType, int, or list of EventType or int")

        self._device_handle = device_handle
        self._event_set = nvml.event_set_create()
        # If this raises, the event needs to be freed and this is handled by
        # this class's __dealloc__ method.
        nvml.device_register_events(self._device_handle, event_bitmask, self._event_set)

    def __dealloc__(self):
        nvml.event_set_free(self._event_set)

    def wait(self, timeout_ms: int = 0) -> EventData:
        """
        Wait for events in the event set.

        For Fermi™ or newer fully supported devices.

        If some events are ready to be delivered at the time of the call,
        function returns immediately.  If there are no events ready to be
        delivered, function sleeps until event arrives but not longer than
        specified timeout. If timeout passes, a
        :class:`cuda.core.system.TimeoutError` is raised. This function in
        certain conditions can return before specified timeout passes (e.g. when
        interrupt arrives).

        On Windows, in case of Xid error, the function returns the most recent
        Xid error type seen by the system.  If there are multiple Xid errors
        generated before ``wait`` is invoked, then the last seen Xid
        error type is returned for all Xid error events.

        On Linux, every Xid error event would return the associated event data
        and other information if applicable.

        In MIG mode, if device handle is provided, the API reports all the
        events for the available instances, only if the caller has appropriate
        privileges. In absence of required privileges, only the events which
        affect all the instances (i.e. whole device) are reported.

        This API does not currently support per-instance event reporting using
        MIG device handles.

        Parameters
        ----------
        timeout_ms: int
            The timeout in milliseconds. A value of 0 means to wait indefinitely.

        Raises
        ------
        :class:`cuda.core.system.TimeoutError`
            If the timeout expires before an event is received.
        :class:`cuda.core.system.GpuIsLostError`
            If the GPU has fallen off the bus or is otherwise inaccessible.
        """
        return EventData(nvml.event_set_wait_v2(self._event_set, timeout_ms))
