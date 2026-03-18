# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0


ClockId = nvml.ClockId
ClocksEventReasons = nvml.ClocksEventReasons
ClockType = nvml.ClockType


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

    def __init__(self, handle, clock_type: ClockType):
        self._handle = handle
        self._clock_type = int(clock_type)

    def get_current_mhz(self, clock_id: ClockId = ClockId.CURRENT) -> int:
        """
        Get the current clock speed of a specific clock domain, in MHz.

        For Kepler™ or newer fully supported devices.

        Parameters
        ----------
        clock_id: :class:`ClockId`
            The clock ID to query.

        Returns
        -------
        int
            The clock speed in MHz.
        """
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

    def get_min_max_clock_of_pstate_mhz(self, pstate: Pstates) -> tuple[int, int]:
        """
        Get the minimum and maximum clock speeds for this clock domain
        at a given performance state (Pstate), in MHz.

        Parameters
        ----------
        pstate: :class:`Pstates`
            The performance state to query.

        Returns
        -------
        tuple[int, int]
            A tuple containing the minimum and maximum clock speeds in MHz.
        """
        return nvml.device_get_min_max_clock_of_p_state(self._handle, self._clock_type, pstate)

    def get_offsets(self, pstate: Pstates) -> ClockOffsets:
        """
        Retrieve min, max and current clock offset of some clock domain for a given Pstate.

        For Maxwell™ or newer fully supported devices.

        Parameters
        ----------
        pstate: :class:`Pstates`
            The performance state to query.

        Returns
        -------
        ClockOffsets
            An object with the min, max and current clock offset.
        """
        return ClockOffsets(nvml.device_get_clock_offsets(self._handle, self._clock_type, pstate))
