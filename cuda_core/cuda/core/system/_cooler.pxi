# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0


CoolerControl = nvml.CoolerControl
CoolerTarget = nvml.CoolerTarget


cdef class CoolerInfo:
    cdef object _cooler_info

    def __init__(self, cooler_info: nvml.CoolerInfo):
        self._cooler_info = cooler_info

    @property
    def signal_type(self) -> CoolerControl:
        """
        The cooler's control signal characteristics.

        The possible types are restricted, variable and toggle.  See
        :class:`CoolerControl` for details.
        """
        return CoolerControl(self._cooler_info.signal_type)

    @property
    def target(self) -> list[CoolerTarget]:
        """
        The target that cooler controls.

        Targets may be GPU, Memory, Power Supply, or all of these.  See
        :class:`CoolerTarget` for details.
        """
        cdef uint64_t[1] targets = [self._cooler_info.target]
        return [CoolerTarget(1 << ev) for ev in _unpack_bitmask(targets)]
