# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from cuda.core._utils.pycompat import StrEnum

__all__ = [
    "AddressingMode",
    "AffinityScope",
    "ClockId",
    "ClockType",
    "ClocksEventReasons",
    "CoolerControl",
    "CoolerTarget",
    "EventType",
    "FanControlPolicy",
    "GpuP2PCapsIndex",
    "GpuP2PStatus",
    "GpuTopologyLevel",
    "InforomObject",
    "SystemEventType",
    "TemperatureThresholds",
    "ThermalController",
    "ThermalTarget",
]


class AddressingMode(StrEnum):
    """
    Addressing mode of a device.

    For Kepler™ or newer fully supported devices.
    """

    HMM = "hmm"
    ATS = "ats"


AddressingMode.HMM.__doc__ = """
    System allocated memory (``malloc``, ``mmap``) is addressable from the device
    (GPU), via software-based mirroring of the CPU's page tables, on the GPU.
"""

AddressingMode.ATS.__doc__ = """
    System allocated memory (``malloc``, ``mmap``) is addressable from the device
    (GPU), via Address Translation Services. This means that there is (effectively)
    a single set of page tables, and the CPU and GPU both use them.
"""


class AffinityScope(StrEnum):
    """
    Scope for affinity queries.
    """

    NODE = "node"
    SOCKET = "socket"


AffinityScope.NODE.__doc__ = """
The NUMA node is the scope of the affinity query.  This is the default scope.
"""

AffinityScope.SOCKET.__doc__ = """
The CPU socket is the scope of the affinity query.
"""


class ClockId(StrEnum):
    """
    Clock Ids. These are used in combination with :class:`ClockType` to specify a single clock value.
    """

    CURRENT = "current"
    CUSTOMER_BOOST_MAX = "customer_boost_max"
    # APP_CLOCK_TARGET and APP_CLOCK_DEFAULT are deprecated so not included here


ClockId.CURRENT.__doc__ = "Current actual clock value."
ClockId.CUSTOMER_BOOST_MAX.__doc__ = "OEM-defined maximum clock rate"


class ClocksEventReasons(StrEnum):
    """
    Reasons for a clocks event.  These are used in combination with :class:`ClockType` to specify the reason
    for a clocks event.
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


class ClockType(StrEnum):
    """
    Clock types. All speeds are in Mhz.
    """

    GRAPHICS = "graphics"
    SM = "sm"
    MEMORY = "memory"
    VIDEO = "video"


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


class EventType(StrEnum):
    """
    Event types that can be waited on with :class:`DeviceEvents`.
    """

    NONE = "none"
    SINGLE_BIT_ECC_ERROR = "single_bit_ecc_error"
    DOUBLE_BIT_ECC_ERROR = "double_bit_ecc_error"
    PSTATE = "pstate"
    XID_CRITICAL_ERROR = "xid_critical_error"
    CLOCK = "clock"
    POWER_SOURCE_CHANGE = "power_source_change"
    MIG_CONFIG_CHANGE = "mig_config_change"
    SINGLE_BIT_ECC_ERROR_STORM = "single_bit_ecc_error_storm"
    DRAM_RETIREMENT_EVENT = "dram_retirement_event"
    DRAM_RETIREMENT_FAILURE = "dram_retirement_failure"
    NON_FATAL_POISON_ERROR = "non_fatal_poison_error"
    FATAL_POISON_ERROR = "fatal_poison_error"
    GPU_UNAVAILABLE_ERROR = "gpu_unavailable_error"
    GPU_RECOVERY_ACTION = "gpu_recovery_action"


EventType.PSTATE.__doc__ = """
Event about PState changes

On Fermi™ architecture, PState changes are also an indicator that GPU is throttling down due to
no work being executed on the GPU, power capping or thermal capping. In a typical situation,
Fermi-based GPU should stay in P0 for the duration of the execution of the compute process.
"""


class FanControlPolicy(StrEnum):
    """
    Fan control policies.
    """

    TEMPERATURE_CONTROLLED = "temperature_controlled"
    MANUAL = "manual"


class GpuP2PCapsIndex(StrEnum):
    """
    GPU peer-to-peer capabilities index.
    """

    READ = "read"
    WRITE = "write"
    NVLINK = "nvlink"
    ATOMICS = "atomics"
    PCI = "pci"
    PROP = "prop"
    UNKNOWN = "unknown"


class GpuP2PStatus(StrEnum):
    """
    GPU peer-to-peer status.
    """

    OK = "ok"
    CHIPSET_NOT_SUPPORTED = "chipset not supported"
    GPU_NOT_SUPPORTED = "GPU not supported"
    IOH_TOPOLOGY_NOT_SUPPORTED = "IOH topology not supported"
    DISABLED_BY_REGKEY = "disabled by regkey"
    NOT_SUPPORTED = "not supported"
    UNKNOWN = "unknown"


class GpuTopologyLevel(StrEnum):
    """
    Represents level relationships within a system between two GPUs.
    """

    INTERNAL = "internal"
    SINGLE = "single"
    MULTIPLE = "multiple"
    HOSTBRIDGE = "hostbridge"
    NODE = "node"
    SYSTEM = "system"


class InforomObject(StrEnum):
    """
    InfoROM objects types.
    """

    OEM = "oem"
    ECC = "ecc"
    POWER = "power"
    DEN = "den"


InforomObject.OEM.__doc__ = "An object defined by OEM."
InforomObject.ECC.__doc__ = "The ECC object determining the level of ECC support."
InforomObject.POWER.__doc__ = "The power management object."
InforomObject.DEN.__doc__ = "DRAM Encryption object."


class SystemEventType(StrEnum):
    """
    System event types.
    """

    UNBIND = "unbind"
    BIND = "bind"


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


# DeviceArch values are derived from cuda.bindings.nvml at definition time, so
# the class can only be defined when nvml is importable.
try:
    from cuda.bindings import nvml as _nvml

    try:
        from cuda.bindings._internal._fast_enum import FastEnum as _FastEnum
    except ImportError:
        from enum import IntEnum as _FastEnum

    # This uses FastEnum instead of StrEnum because the ordering of the values is
    # meaningful, e.g. Kepler "or later"
    class DeviceArch(_FastEnum):
        """
        Device architecture.
        """

        KEPLER = int(_nvml.DeviceArch.KEPLER)
        MAXWELL = int(_nvml.DeviceArch.MAXWELL)
        PASCAL = int(_nvml.DeviceArch.PASCAL)
        VOLTA = int(_nvml.DeviceArch.VOLTA)
        TURING = int(_nvml.DeviceArch.TURING)
        AMPERE = int(_nvml.DeviceArch.AMPERE)
        ADA = int(_nvml.DeviceArch.ADA)
        HOPPER = int(_nvml.DeviceArch.HOPPER)
        BLACKWELL = int(_nvml.DeviceArch.BLACKWELL)
        UNKNOWN = int(_nvml.DeviceArch.UNKNOWN)

    __all__.append("DeviceArch")

    FieldId = _nvml.FieldId

    __all__.append("FieldId")

    del _nvml, _FastEnum

except ImportError:
    pass


del StrEnum
