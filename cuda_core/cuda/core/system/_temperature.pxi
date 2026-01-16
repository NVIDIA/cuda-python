# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0


# In cuda.bindings.nvml, this is an anonymous struct inside nvmlThermalSettings_t.


ctypedef struct _ThermalSensor:
    int controller
    int defaultMinTemp
    int defaultMaxTemp
    int currentTemp
    int target


cdef class ThermalSensor:
    cdef:
        _ThermalSensor *_ptr
        object _owner

    def __init__(self, ptr: int, owner: object):
        # ptr points to a part of the numpy buffer held by `_owner`, so we need
        # to maintain a reference to `_owner` to keep it alive.
        self._ptr = <_ThermalSensor *><intptr_t>ptr
        self._owner = owner

    @property
    def controller(self) -> ThermalController:
        return ThermalController(self._ptr[0].controller)

    @property
    def default_min_temp(self) -> int:
        return self._ptr[0].defaultMinTemp

    @property
    def default_max_temp(self) -> int:
        return self._ptr[0].defaultMaxTemp

    @property
    def current_temp(self) -> int:
        return self._ptr[0].currentTemp

    @property
    def target(self) -> ThermalTarget:
        return ThermalTarget(self._ptr[0].target)


cdef class ThermalSettings:
    cdef object _thermal_settings

    def __init__(self, thermal_settings: nvml.ThermalSettings):
        self._thermal_settings = thermal_settings

    def __len__(self):
        # MAX_THERMAL_SENSORS_PER_GPU is 3
        return min(self._thermal_settings.count, 3)

    def __getitem__(self, idx: int) -> nvml.ThermalSensor:
        if idx < 0 or idx >= len(self):
            raise IndexError("Thermal sensor index out of range")
        return ThermalSensor(
            self._thermal_settings.sensor.ptr + idx * sizeof(_ThermalSensor),
            self._thermal_settings
        )


cdef class Temperature:
    cdef intptr_t _handle

    def __init__(self, handle: int):
        self._handle = handle

    def sensor(
        self,
        sensor: TemperatureSensors = TemperatureSensors.TEMPERATURE_GPU
    ) -> int:
        """
        Get the temperature reading from a specific sensor on the device, in
        degrees Celsius.

        Parameters
        ----------
        sensor: :class:`TemperatureSensors`, optional
            The temperature sensor to query.

        Returns
        -------
        int
            The temperature in degrees Celsius.
        """
        return nvml.device_get_temperature_v(self._handle, sensor)

    def threshold(self, threshold_type: TemperatureThresholds) -> int:
        """
        Retrieves the temperature threshold for this GPU with the specified
        threshold type, in degrees Celsius.

        For Kepler™ or newer fully supported devices.

        See :class:`TemperatureThresholds` for possible threshold types.

        Note: This API is no longer the preferred interface for retrieving the
        following temperature thresholds on Ada and later architectures:
        ``NVML_TEMPERATURE_THRESHOLD_SHUTDOWN``,
        ``NVML_TEMPERATURE_THRESHOLD_SLOWDOWN``,
        ``NVML_TEMPERATURE_THRESHOLD_MEM_MAX`` and
        ``NVML_TEMPERATURE_THRESHOLD_GPU_MAX``.

        Support for reading these temperature thresholds for Ada and later
        architectures would be removed from this API in future releases. Please
        use :meth:`get_field_values` with ``NVML_FI_DEV_TEMPERATURE_*`` fields
        to retrieve temperature thresholds on these architectures.
        """
        return nvml.device_get_temperature_threshold(self._handle, threshold_type)

    @property
    def margin(self) -> int:
        """
        The thermal margin temperature (distance to nearest slowdown threshold) for the device.
        """
        return nvml.device_get_margin_temperature(self._handle)

    def thermal_settings(self, sensor_index: ThermalTarget) -> ThermalSettings:
        """
        Used to execute a list of thermal system instructions.

        TODO: The above docstring is from the NVML header, but it doesn't seem to make sense.

        Parameters
        ----------
        sensor_index: ThermalTarget
            The index of the thermal sensor.

        Returns
        -------
        :class:`ThermalSettings`
            The thermal settings for the specified sensor.
        """
        return ThermalSettings(nvml.device_get_thermal_settings(self._handle, sensor_index))
