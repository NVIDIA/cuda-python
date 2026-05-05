# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# Verify that every cuda_binding enum member has a corresponding entry in the
# cuda_core wrapper mappings.  No GPU required; the test only inspects
# mapping dicts at import time, so it runs on any CI host that has a
# compatible cuda.bindings version.

import importlib
import inspect
import pkgutil
import sys

import pytest

import cuda.core
from cuda.core import system

if sys.version_info >= (3, 11):
    from enum import StrEnum
else:
    from backports.strenum import StrEnum

# Each entry is:
#   (cuda_binding_enum, str_enum, mapping_dict, binding_unmapped, str_enum_unmapped)
#
# cuda_binding_enum: the cuda.bindings enum class
# str_enum: the cuda_core StrEnum wrapper class, or None if the mapping does
#   not use a StrEnum (e.g. maps to plain str or tuple)
# mapping_dict: the dict that maps between the two enum types
# binding_unmapped: cuda_binding_enum member names intentionally absent from the mapping
#   (sentinels, deprecated aliases, etc.)
# str_enum_unmapped: StrEnum member names intentionally absent from the mapping
#   (fallback sentinels returned by the wrapper via .get(value, default))
#
# The first test checks that every member of cuda_binding_enum whose name is NOT
# in binding_unmapped appears as either a key or a value of the mapping dict,
# and conversely that every str_enum member not in str_enum_unmapped also
# appears.
_CASES = []

if system.CUDA_BINDINGS_NVML_IS_COMPATIBLE:
    # Populated below only when NVML bindings are compatible, so that importing
    # this module on an incompatible host does not raise ImportError.
    from cuda.bindings import nvml
    from cuda.core.system import _device, _system_events

    _CASES.extend(
        [
            (
                nvml.DeviceAddressingModeType,
                _device.AddressingMode,
                _device._ADDRESSING_MODE_MAPPING,
                # NONE means "no special addressing mode is active"; not a valid target
                {"DEVICE_ADDRESSING_MODE_NONE"},
                set(),
            ),
            (
                nvml.BrandType,
                None,  # maps to plain str, not a StrEnum
                _device._BRAND_TYPE_MAPPING,
                # COUNT is a sentinel, not a real brand
                {"BRAND_COUNT"},
                set(),
            ),
            (
                nvml.GpuP2PStatus,
                _device.GpuP2PStatus,
                _device._GPU_P2P_STATUS_MAPPING,
                # Both the typo'd (SUPPORED) and corrected (SUPPORTED) spellings
                # share the same integer value; the mapping covers both via aliases
                set(),
                set(),
            ),
            (
                nvml.ClocksEventReasons,
                _device.ClocksEventReasons,
                _device._CLOCKS_EVENT_REASONS_MAPPING,
                set(),
                set(),
            ),
            (
                nvml.EventType,
                _device.EventType,
                _device._EVENT_TYPE_MAPPING,
                set(),
                set(),
            ),
            (
                nvml.FanControlPolicy,
                _device.FanControlPolicy,
                _device._FAN_CONTROL_POLICY_MAPPING,
                set(),
                set(),
            ),
            (
                nvml.CoolerControl,
                _device.CoolerControl,
                _device._COOLER_CONTROL_MAPPING,
                # NONE means no signal; COUNT is a sentinel
                {"THERMAL_COOLER_SIGNAL_NONE", "THERMAL_COOLER_SIGNAL_COUNT"},
                set(),
            ),
            (
                nvml.CoolerTarget,
                _device.CoolerTarget,
                _device._COOLER_TARGET_MAPPING,
                # GPU_RELATED is a composite bitmask (GPU | MEMORY | POWER_SUPPLY);
                # the wrapper expands it into individual targets instead of mapping
                # it as a single entry
                {"THERMAL_GPU_RELATED"},
                set(),
            ),
            (
                nvml.ThermalController,
                _device.ThermalController,
                _device._THERMAL_CONTROLLER_MAPPING,
                # NONE and UNKNOWN are both handled by the .get() fallback that
                # returns ThermalController.UNKNOWN when the value is not in the mapping
                {"NONE", "UNKNOWN"},
                # UNKNOWN is the default returned by .get() for unrecognised controllers
                {"UNKNOWN"},
            ),
            (
                nvml.ThermalTarget,
                _device.ThermalTarget,
                _device._THERMAL_TARGET_MAPPING,
                # UNKNOWN is a fallback sentinel; handled by .get()
                {"UNKNOWN"},
                set(),
            ),
            (
                nvml.NvlinkVersion,
                None,  # maps to tuple, not a StrEnum
                _device._NVLINK_VERSION_MAPPING,
                # VERSION_INVALID is a sentinel for "no NvLink present"
                {"VERSION_INVALID"},
                set(),
            ),
            (
                nvml.SystemEventType,
                _system_events.SystemEventType,
                _system_events._SYSTEM_EVENT_TYPE_MAPPING,
                set(),
                set(),
            ),
            (
                nvml.AffinityScope,
                _device.AffinityScope,
                _device._AFFINITY_SCOPE_MAPPING,
                set(),
                set(),
            ),
            (
                nvml.GpuP2PCapsIndex,
                _device.GpuP2PCapsIndex,
                _device._GPU_P2P_CAPS_INDEX_MAPPING,
                # UNKNOWN is returned by the driver when an index is unrecognised;
                # it is not a capability the caller selects
                {"P2P_CAPS_INDEX_UNKNOWN"},
                # UNKNOWN is a driver-side fallback, not a caller-selectable index
                {"UNKNOWN"},
            ),
            (
                nvml.GpuTopologyLevel,
                _device.GpuTopologyLevel,
                _device._GPU_TOPOLOGY_LEVEL_MAPPING,
                set(),
                set(),
            ),
            (
                nvml.ClockId,
                _device.ClockId,
                _device._CLOCK_ID_MAPPING,
                # APP_CLOCK_TARGET and APP_CLOCK_DEFAULT are deprecated; COUNT is a sentinel
                {"APP_CLOCK_TARGET", "APP_CLOCK_DEFAULT", "COUNT"},
                set(),
            ),
            (
                nvml.ClockType,
                _device.ClockType,
                _device._CLOCK_TYPE_MAPPING,
                # COUNT is a sentinel
                {"CLOCK_COUNT"},
                set(),
            ),
            (
                nvml.InforomObject,
                _device.InforomObject,
                _device._INFOROM_OBJECT_MAPPING,
                # COUNT is a sentinel
                {"INFOROM_COUNT"},
                set(),
            ),
            (
                nvml.TemperatureThresholds,
                _device.TemperatureThresholds,
                _device._TEMPERATURE_THRESHOLD_MAPPING,
                # COUNT is a sentinel
                {"TEMPERATURE_THRESHOLD_COUNT"},
                set(),
            ),
        ]
    )


# StrEnum subclasses that intentionally have no associated cuda_binding.
# Add classes here (with a comment explaining why) when a new StrEnum is
# introduced that wraps something other than a cuda_binding enum.
_UNBOUND_STR_ENUMS: frozenset[type] = frozenset()


@pytest.mark.parametrize(
    "binding, str_enum, mapping, binding_unmapped, str_enum_unmapped",
    _CASES,
    ids=[x[0].__name__ for x in _CASES],
)
def test_wrapper_covers_all_binding_members(binding, str_enum, mapping, binding_unmapped, str_enum_unmapped):
    """Every cuda_binding enum member must appear in the wrapper mapping (or be allow-listed).

    Also checks the reverse: every StrEnum wrapper member must appear in the
    mapping (or be listed in the per-entry str_enum_unmapped set).
    """
    required = set(binding.__members__) - binding_unmapped
    # Compare by integer value so that enum aliases (two names, one integer)
    # are treated as covered when the canonical member appears in the mapping.
    covered_values = frozenset(int(m) for m in (*mapping.keys(), *mapping.values()) if isinstance(m, binding))
    missing = {name for name in required if int(binding.__members__[name]) not in covered_values}
    assert not missing, f"{binding.__name__} has members not covered by the wrapper mapping: {missing}"

    # Reverse check: every StrEnum member must also appear in the mapping.
    if str_enum is not None:
        required_str = set(str_enum.__members__) - str_enum_unmapped
        covered_str = {m.name for m in (*mapping.keys(), *mapping.values()) if isinstance(m, str_enum)}
        missing_str = required_str - covered_str
        assert not missing_str, f"{str_enum.__name__} has members not covered by the wrapper mapping: {missing_str}"


def test_all_str_enums_in_cases():
    """Every StrEnum subclass in cuda.core must appear in _CASES or _UNBOUND_STR_ENUMS.

    This ensures that when a new StrEnum wrapper is added to cuda.core, the
    author is prompted to add a binding-coverage entry to _CASES (or explicitly
    declare it as unbound in _UNBOUND_STR_ENUMS).
    """

    def discover_str_enums() -> set[type]:
        """Walk all submodules of cuda.core and return every StrEnum subclass found."""
        found: set[type] = set()
        for _, modname, _ in pkgutil.walk_packages(
            path=cuda.core.__path__,
            prefix=cuda.core.__name__ + ".",
            onerror=lambda _: None,
        ):
            try:
                mod = importlib.import_module(modname)
            except Exception:  # noqa
                continue
            try:
                members = inspect.getmembers(mod, inspect.isclass)
            except Exception:  # noqa
                continue
            for _, obj in members:
                if obj is not StrEnum and issubclass(obj, StrEnum):
                    found.add(obj)
        return found

    covered = {x[1] for x in _CASES if x[1] is not None}
    uncovered = discover_str_enums() - covered - _UNBOUND_STR_ENUMS
    assert not uncovered, (
        f"StrEnum subclasses in cuda.core not covered by _CASES: "
        f"{sorted(c.__qualname__ for c in uncovered)}\n"
        "Add a _CASES entry for each, or add to _UNBOUND_STR_ENUMS if it does not wrap a cuda_binding enum."
    )
