# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0


class CoolerControl(StrEnum):
    """
    Cooler control type.
    """
    TOGGLE = "toggle"
    VARIABLE = "variable"


CoolerControl.TOGGLE.__doc__ = """
This cooler can only be toggled either ON or OFF (e.g. a switch).
"""
CoolerControl.VARIABLE.__doc__ = """
This cooler's level can be adjusted from some minimum to some maximum (e.g. a knob).
"""


_COOLER_CONTROL_MAPPING = {
    nvml.CoolerControl.THERMAL_COOLER_SIGNAL_TOGGLE: CoolerControl.TOGGLE,
    nvml.CoolerControl.THERMAL_COOLER_SIGNAL_VARIABLE: CoolerControl.VARIABLE,
}


class CoolerTarget(StrEnum):
    """
    Cooler target.
    """
    NONE = "none"
    GPU = "gpu"
    MEMORY = "memory"
    POWER_SUPPLY = "power_supply"
    # THERMAL_GPU_RELATED is a composite target, so it is omitted here and will
    # get returned as 3 separate targets: GPU, MEMORY, and POWER_SUPPLY.


CoolerTarget.NONE.__doc__ = "This cooler controls nothing."
CoolerTarget.GPU.__doc__ = "This cooler can cool the GPU."
CoolerTarget.MEMORY.__doc__ = "This cooler can cool the memory."
CoolerTarget.POWER_SUPPLY.__doc__ = "This cooler can cool the power supply."


_COOLER_TARGET_MAPPING = {
    nvml.CoolerTarget.THERMAL_NONE: CoolerTarget.NONE,
    nvml.CoolerTarget.THERMAL_GPU: CoolerTarget.GPU,
    nvml.CoolerTarget.THERMAL_MEMORY: CoolerTarget.MEMORY,
    nvml.CoolerTarget.THERMAL_POWER_SUPPLY: CoolerTarget.POWER_SUPPLY,
}


cdef class CoolerInfo:
    cdef object _cooler_info

    def __init__(self, cooler_info: nvml.CoolerInfo):
        self._cooler_info = cooler_info

    @property
    def signal_type(self) -> CoolerControl | None:
        """
        The cooler's control signal characteristics.

        The possible types are variable and toggle.
        """
        return _COOLER_CONTROL_MAPPING.get(self._cooler_info.signal_type, None)

    @property
    def target(self) -> list[CoolerTarget]:
        """
        The target that cooler controls.

        Targets may be GPU, Memory, Power Supply, or all of these.  See
        :class:`CoolerTarget` for details.
        """
        cdef uint64_t[1] targets = [self._cooler_info.target]
        output_targets = []
        for target in _unpack_bitmask(targets):
            try:
                output_target = _COOLER_TARGET_MAPPING[1 << target]
            except KeyError:
                raise ValueError(f"Unknown cooler target bit: {1 << target}")
            output_targets.append(output_target)
        return output_targets
