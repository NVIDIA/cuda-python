# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0


class ClockId(StrEnum):
    """
    Clock Ids. These are used in combination with :class:`ClockType` to specify a single clock value.
    """
    CURRENT = "current"
    CUSTOMER_BOOST_MAX = "customer_boost_max"
    # APP_CLOCK_TARGET and APP_CLOCK_DEFAULT are deprecated so not included here


ClockId.CURRENT.__doc__ = "Current actual clock value."
ClockId.CUSTOMER_BOOST_MAX.__doc__ = "OEM-defined maximum clock rate"


_CLOCK_ID_MAPPING = {
    ClockId.CURRENT: nvml.ClockId.CURRENT,
    ClockId.CUSTOMER_BOOST_MAX: nvml.ClockId.CUSTOMER_BOOST_MAX,
}


class ClocksEventReasons(StrEnum):
    """
    Reasons for a clocks event.  These are used in combination with :class:`ClockType` to specify the reason for a clocks event.
    """
    NONE = "none"
    GPU_IDLE = "gpu_idle"
    APPLICATIONS_CLOCKS_SETTING = "applications_clocks_setting"
    SW_POWER_CAP = "sw_power_cap"
    HW_SLOWDOWN = "hw_slowdown"
    SYNC_BOOST = "sync_boost"
    SW_THERMAL_SLOWDOWN = "sw_thermal_slowdown"
    HW_THERMAL_SLOWDOWN = "hw_thermal_slowdown"
    HW_POWER_BRAKE_SLOWDOWN = "hw_power_brake_slowdown"
    DISPLAY_CLOCK_SETTING = "display_clock_setting"


_CLOCKS_EVENT_REASONS_MAPPING = {
    nvml.ClocksEventReasons.EVENT_REASON_NONE: ClocksEventReasons.NONE,
    nvml.ClocksEventReasons.EVENT_REASON_GPU_IDLE: ClocksEventReasons.GPU_IDLE,
    nvml.ClocksEventReasons.EVENT_REASON_APPLICATIONS_CLOCKS_SETTING: ClocksEventReasons.APPLICATIONS_CLOCKS_SETTING,
    nvml.ClocksEventReasons.EVENT_REASON_SW_POWER_CAP: ClocksEventReasons.SW_POWER_CAP,
    nvml.ClocksEventReasons.THROTTLE_REASON_HW_SLOWDOWN: ClocksEventReasons.HW_SLOWDOWN,
    nvml.ClocksEventReasons.EVENT_REASON_SYNC_BOOST: ClocksEventReasons.SYNC_BOOST,
    nvml.ClocksEventReasons.EVENT_REASON_SW_THERMAL_SLOWDOWN: ClocksEventReasons.SW_THERMAL_SLOWDOWN,
    nvml.ClocksEventReasons.THROTTLE_REASON_HW_THERMAL_SLOWDOWN: ClocksEventReasons.HW_THERMAL_SLOWDOWN,
    nvml.ClocksEventReasons.THROTTLE_REASON_HW_POWER_BRAKE_SLOWDOWN: ClocksEventReasons.HW_POWER_BRAKE_SLOWDOWN,
    nvml.ClocksEventReasons.EVENT_REASON_DISPLAY_CLOCK_SETTING: ClocksEventReasons.DISPLAY_CLOCK_SETTING,
}


class ClockType(StrEnum):
    """
    Clock types. All speeds are in Mhz.
    """
    GRAPHICS = "graphics"
    SM = "sm"
    MEMORY = "memory"
    VIDEO = "video"


_CLOCK_TYPE_MAPPING = {
    ClockType.GRAPHICS: nvml.ClockType.CLOCK_GRAPHICS,
    ClockType.SM: nvml.ClockType.CLOCK_SM,
    ClockType.MEMORY: nvml.ClockType.CLOCK_MEM,
    ClockType.VIDEO: nvml.ClockType.CLOCK_VIDEO,
}


cdef class ClockOffsets:
    """
    Contains clock offset information.
    """

    cdef object _clock_offset

    def __init__(self, clock_offset: nvml.ClockOffset):
        self._clock_offset = clock_offset

    @property
    def clock_offset_mhz(self) -> int:
        """
        The current clock offset in MHz.
        """
        return self._clock_offset.clock_offset_m_hz

    @property
    def max_offset_mhz(self) -> int:
        """
        The maximum clock offset in MHz.
        """
        return self._clock_offset.max_clock_offset_m_hz

    @property
    def min_offset_mhz(self) -> int:
        """
        The minimum clock offset in MHz.
        """
        return self._clock_offset.min_clock_offset_m_hz


cdef class ClockInfo:
    """
    Accesses various clock information about a device.
    """

    cdef intptr_t _handle
    cdef int _clock_type

    def __init__(self, handle, clock_type: ClockType | str):
        self._handle = handle
        try:
            clock_type = _CLOCK_TYPE_MAPPING[clock_type]
        except KeyError:
            raise ValueError(
                f"Invalid clock type: {clock_type}. "
                f"Must be one of {list(ClockType.__members__.values())}"
            ) from None
        self._clock_type = int(clock_type)

    def get_current_mhz(self, clock_id: ClockId | str = ClockId.CURRENT) -> int:
        """
        Get the current clock speed of a specific clock domain, in MHz.

        For Kepler™ or newer fully supported devices.

        Parameters
        ----------
        clock_id: :class:`ClockId` | str
            The clock ID to query.  Defaults to the current clock value.

        Returns
        -------
        int
            The clock speed in MHz.
        """
        try:
            clock_id = _CLOCK_ID_MAPPING[clock_id]
        except KeyError:
            raise ValueError(
                f"Invalid clock ID: {clock_id}. "
                f"Must be one of {list(ClockId.__members__.values())}"
            ) from None
        return nvml.device_get_clock(self._handle, self._clock_type, clock_id)

    def get_max_mhz(self) -> int:
        """
        Get the maximum clock speed of a specific clock domain, in MHz.

        For Fermi™ or newer fully supported devices.

        Current P0 clocks (reported by :meth:`get_current_mhz` can differ from
        max clocks by a few MHz.

        Returns
        -------
        int
            The maximum clock speed in MHz.
        """
        return nvml.device_get_max_clock_info(self._handle, self._clock_type)

    def get_max_customer_boost_mhz(self) -> int:
        """
        Get the maximum customer boost clock speed of a specific clock, in MHz.

        For Pascal™ or newer fully supported devices.

        Returns
        -------
        int
            The maximum customer boost clock speed in MHz.
        """
        return nvml.device_get_max_customer_boost_clock(self._handle, self._clock_type)

    def get_min_max_clock_of_pstate_mhz(self, pstate: int) -> tuple[int, int]:
        """
        Get the minimum and maximum clock speeds for this clock domain
        at a given performance state (Pstate), in MHz.

        Parameters
        ----------
        pstate: int
            The performance state to query.  Must be an int between 0 and 15,
            where 0 is the highest performance state (P0) and 15 is the lowest
            (P15).

        Returns
        -------
        tuple[int, int]
            A tuple containing the minimum and maximum clock speeds in MHz.
        """
        return nvml.device_get_min_max_clock_of_p_state(self._handle, self._clock_type, _pstate_to_enum(pstate))

    def get_offsets(self, pstate: int) -> ClockOffsets:
        """
        Retrieve min, max and current clock offset of some clock domain for a given Pstate.

        For Maxwell™ or newer fully supported devices.

        Parameters
        ----------
        pstate: int
            The performance state to query.  Must be an int between 0 and 15,
            where 0 is the highest performance state (P0) and 15 is the lowest
            (P15).

        Returns
        -------
        :obj:`~_device.ClockOffsets`
            An object with the min, max and current clock offset.
        """
        return ClockOffsets(nvml.device_get_clock_offsets(self._handle, self._clock_type, _pstate_to_enum(pstate)))
