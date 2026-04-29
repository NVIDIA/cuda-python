# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0


cdef class Utilization:
    """
    Utilization rates for a device.

    For devices with compute capability 2.0 or higher.
    """
    cdef object _utilization

    def __init__(self, utilization: nvml.Utilization):
        self._utilization = utilization

    @property
    def gpu(self) -> int:
        """
        Percent of time over the past sample period during which one or more kernels was executing on the GPU.
        """
        return self._utilization.gpu

    @property
    def memory(self) -> int:
        """
        Percent of time over the past sample period during which global (device) memory was being read or written.
        """
        return self._utilization.memory
