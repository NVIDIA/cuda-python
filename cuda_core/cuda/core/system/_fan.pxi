# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0


cdef class FanInfo:
    """
    Manages information related to a specific fan on a specific device.
    """

    cdef intptr_t _handle
    cdef int _fan

    def __init__(self, handle: int, fan: int):
        self._handle = handle
        self._fan = fan

    @property
    def speed(self) -> int:
        """
        Get/set the intended operating speed of the device's fan.

        For all discrete products with dedicated fans.

        Note: The reported speed is the intended fan speed.  If the fan is
        physically blocked and unable to spin, the output will not match the
        actual fan speed.

        The fan speed is expressed as a percentage of the product's maximum
        noise tolerance fan speed.  This value may exceed 100% in certain cases.
        """
        return nvml.device_get_fan_speed_v2(self._handle, self._fan)

    @speed.setter
    def speed(self, speed: int):
        nvml.device_set_fan_speed_v2(self._handle, self._fan, speed)

    @property
    def speed_rpm(self) -> int:
        """
        The intended operating speed of the device's fan in rotations per minute
        (RPM).

        For Maxwell™ or newer fully supported devices.

        For all discrete products with dedicated fans.

        Note: The reported speed is the intended fan speed.  If the fan is
        physically blocked and unable to spin, the output will not match the
        actual fan speed.
        """
        return nvml.device_get_fan_speed_rpm(self._handle, self._fan)

    @property
    def target_speed(self) -> int:
        """
        Retrieves the intended target speed of the device's specified fan.

        For all discrete products with dedicated fans.

        Normally, the driver dynamically adjusts the fan based on
        the needs of the GPU.  But when user set fan speed using :property:`speed`
        the driver will attempt to make the fan achieve the setting in
        :property:`speed`.  The actual current speed of the fan
        is reported in :property:`speed`.

        The fan speed is expressed as a percentage of the product's maximum
        noise tolerance fan speed.  This value may exceed 100% in certain cases.
        """
        return nvml.device_get_target_fan_speed(self._handle, self._fan)

    @property
    def min_max_speed(self) -> tuple[int, int]:
        """
        Retrieves the minimum and maximum fan speed all of the device's fans.

        For all discrete products with dedicated fans.

        Returns
        -------
        tuple[int, int]
            A tuple of (min_speed, max_speed)
        """
        return nvml.device_get_min_max_fan_speed(self._handle)

    @property
    def control_policy(self) -> FanControlPolicy:
        """
        The current fan control policy.

        For Maxwell™ or newer fully supported devices.

        For all CUDA-capable discrete products with fans.
        """
        return FanControlPolicy(nvml.device_get_fan_control_policy_v2(self._handle, self._fan))

    def set_default_fan_speed(self):
        """
        Set the speed of the fan control policy to default.

        For all CUDA-capable discrete products with fans.
        """
        nvml.device_set_default_fan_speed_v2(self._handle, self._fan)
