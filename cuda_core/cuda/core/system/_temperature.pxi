# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0


class TemperatureThresholds(StrEnum):
    """
    Temperature threshold types.
    """
    SHUTDOWN = "shutdown"
    SLOWDOWN = "slowdown"
    MEM_MAX = "mem_max"
    GPU_MAX = "gpu_max"
    ACOUSTIC_MIN = "acoustic_min"
    ACOUSTIC_CURR = "acoustic_curr"
    ACOUSTIC_MAX = "acoustic_max"
    GPS_CURR = "gps_curr"
cdef dict _TEMPERATURE_THRESHOLD_MAPPING = {
    TemperatureThresholds.SHUTDOWN: nvml.TemperatureThresholds.TEMPERATURE_THRESHOLD_SHUTDOWN,
    TemperatureThresholds.SLOWDOWN: nvml.TemperatureThresholds.TEMPERATURE_THRESHOLD_SLOWDOWN,
    TemperatureThresholds.MEM_MAX: nvml.TemperatureThresholds.TEMPERATURE_THRESHOLD_MEM_MAX,
    TemperatureThresholds.GPU_MAX: nvml.TemperatureThresholds.TEMPERATURE_THRESHOLD_GPU_MAX,
    TemperatureThresholds.ACOUSTIC_MIN: nvml.TemperatureThresholds.TEMPERATURE_THRESHOLD_ACOUSTIC_MIN,
    TemperatureThresholds.ACOUSTIC_CURR: nvml.TemperatureThresholds.TEMPERATURE_THRESHOLD_ACOUSTIC_CURR,
    TemperatureThresholds.ACOUSTIC_MAX: nvml.TemperatureThresholds.TEMPERATURE_THRESHOLD_ACOUSTIC_MAX,
    TemperatureThresholds.GPS_CURR: nvml.TemperatureThresholds.TEMPERATURE_THRESHOLD_GPS_CURR,
}


class ThermalController(StrEnum):
    """
    Thermal controller types.
    """
    GPU_INTERNAL = "gpu_internal"
    ADM1032 = "adm1032"
    ADT7461 = "adt7461"
    MAX6649 = "max6649"
    MAX1617 = "max1617"
    LM99 = "lm99"
    LM89 = "lm89"
    LM64 = "lm64"
    G781 = "g781"
    ADT7473 = "adt7473"
    SBMAX6649 = "sbmax6649"
    VBIOSEVT = "vbiosevt"
    OS = "os"
    NVSYSCON_CANOAS = "nvsyscon_canoas"
    NVSYSCON_E551 = "nvsyscon_e551"
    MAX6649R = "max6649r"
    ADT7473S = "adt7473s"
    UNKNOWN = "unknown"
cdef dict _THERMAL_CONTROLLER_MAPPING = {
    nvml.ThermalController.GPU_INTERNAL: ThermalController.GPU_INTERNAL,
    nvml.ThermalController.ADM1032: ThermalController.ADM1032,
    nvml.ThermalController.ADT7461: ThermalController.ADT7461,
    nvml.ThermalController.MAX6649: ThermalController.MAX6649,
    nvml.ThermalController.MAX1617: ThermalController.MAX1617,
    nvml.ThermalController.LM99: ThermalController.LM99,
    nvml.ThermalController.LM89: ThermalController.LM89,
    nvml.ThermalController.LM64: ThermalController.LM64,
    nvml.ThermalController.G781: ThermalController.G781,
    nvml.ThermalController.ADT7473: ThermalController.ADT7473,
    nvml.ThermalController.SBMAX6649: ThermalController.SBMAX6649,
    nvml.ThermalController.VBIOSEVT: ThermalController.VBIOSEVT,
    nvml.ThermalController.OS: ThermalController.OS,
    nvml.ThermalController.NVSYSCON_CANOAS: ThermalController.NVSYSCON_CANOAS,
    nvml.ThermalController.NVSYSCON_E551: ThermalController.NVSYSCON_E551,
    nvml.ThermalController.MAX6649R: ThermalController.MAX6649R,
    nvml.ThermalController.ADT7473S: ThermalController.ADT7473S,
}


class ThermalTarget(StrEnum):
    """
    Thermal sensor targets.
    """
    NONE = "none"
    GPU = "gpu"
    MEMORY = "memory"
    POWER_SUPPLY = "power_supply"
    BOARD = "board"
    VCD_BOARD = "vcd_board"
    VCD_INLET = "vcd_inlet"
    VCD_OUTLET = "vcd_outlet"
    ALL = "all"
ThermalTarget.GPU.__doc__ = "GPU core temperature requires physical GPU handle."
ThermalTarget.MEMORY.__doc__ = "GPU memory temperature requires physical GPU handle."
ThermalTarget.POWER_SUPPLY.__doc__ = "GPU power supply temperature requires physical GPU handle."
ThermalTarget.BOARD.__doc__ = "GPU board ambient temperature requires physical GPU handle."
ThermalTarget.VCD_BOARD.__doc__ = "Visual Computing Device Board temperature requires visual computing device handle."
ThermalTarget.VCD_INLET.__doc__ = "Visual Computing Device Inlet temperature requires visual computing device handle."
ThermalTarget.VCD_OUTLET.__doc__ = "Visual Computing Device Outlet temperature requires visual computing device handle."
cdef dict _THERMAL_TARGET_MAPPING = {
    nvml.ThermalTarget.NONE: ThermalTarget.NONE,
    nvml.ThermalTarget.GPU: ThermalTarget.GPU,
    nvml.ThermalTarget.MEMORY: ThermalTarget.MEMORY,
    nvml.ThermalTarget.POWER_SUPPLY: ThermalTarget.POWER_SUPPLY,
    nvml.ThermalTarget.BOARD: ThermalTarget.BOARD,
    nvml.ThermalTarget.VCD_BOARD: ThermalTarget.VCD_BOARD,
    nvml.ThermalTarget.VCD_INLET: ThermalTarget.VCD_INLET,
    nvml.ThermalTarget.VCD_OUTLET: ThermalTarget.VCD_OUTLET,
    nvml.ThermalTarget.ALL: ThermalTarget.ALL,
}
cdef dict _THERMAL_TARGET_INV_MAPPING = {v: k for k, v in _THERMAL_TARGET_MAPPING.items()}


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
        return _THERMAL_CONTROLLER_MAPPING.get(self._ptr[0].controller, ThermalController.UNKNOWN)

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
        return _THERMAL_TARGET_MAPPING.get(self._ptr[0].target, ThermalTarget.NONE)


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

    def get_sensor(self) -> int:
        """
        Get the temperature reading from a specific sensor on the device, in
        degrees Celsius.

        The only sensor currently supported is the GPU temperature sensor.

        Returns
        -------
        int
            The temperature in degrees Celsius.
        """
        # NOTE: nvml.device_get_temperature_v takes a sensor type from the
        # TemperatorSensors enum, but there is only one value in that enum.  For
        # future compatibility if there are other values for that enum, this if
        # a method, not a property
        return nvml.device_get_temperature_v(self._handle, nvml.TemperatureSensors.TEMPERATURE_GPU)

    def get_threshold(self, threshold_type: TemperatureThresholds | str) -> int:
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
        try:
            threshold_type_enum = _TEMPERATURE_THRESHOLD_MAPPING[threshold_type]
        except KeyError:
            raise ValueError(
                f"Invalid temperature threshold type: {threshold_type}. "
                f"Must be one of {list(TemperatureThresholds.__members__.values())}"
            ) from None
        if threshold_type_enum in (
            nvml.TemperatureThresholds.TEMPERATURE_THRESHOLD_SHUTDOWN,
            nvml.TemperatureThresholds.TEMPERATURE_THRESHOLD_SLOWDOWN,
            nvml.TemperatureThresholds.TEMPERATURE_THRESHOLD_MEM_MAX,
            nvml.TemperatureThresholds.TEMPERATURE_THRESHOLD_GPU_MAX
        ):
            device_arch = nvml.DeviceArch(nvml.device_get_architecture(self._handle))
            if device_arch >= nvml.DeviceArch.ADA:
                warnings.warn(
                    f"{threshold_type} is no longer recommended for Ada and later architectures. "
                    "Use get_field_values with NVML_FI_DEV_TEMPERATURE_* fields to retrieve this "
                    "threshold on these architectures.",
                    DeprecationWarning,
                    stacklevel=2
                )
        return nvml.device_get_temperature_threshold(self._handle, threshold_type_enum)

    @property
    def margin(self) -> int:
        """
        The thermal margin temperature (distance to nearest slowdown threshold) for the device.
        """
        return nvml.device_get_margin_temperature(self._handle)

    def get_thermal_settings(self, sensor_index: ThermalTarget | str) -> ThermalSettings:
        """
        Used to execute a list of thermal system instructions.

        Parameters
        ----------
        sensor_index: ThermalTarget
            The index of the thermal sensor.

        Returns
        -------
        :obj:`~_device.ThermalSettings`
            The thermal settings for the specified sensor.
        """
        # TODO: The above docstring is from the NVML header, but it doesn't seem to make sense.
        try:
            sensor_index_enum = _THERMAL_TARGET_INV_MAPPING[sensor_index]
        except KeyError:
            raise ValueError(
                f"Invalid thermal sensor index: {sensor_index}. "
                f"Must be one of {list(ThermalTarget.__members__.values())}"
            ) from None

        return ThermalSettings(nvml.device_get_thermal_settings(self._handle, sensor_index_enum))
