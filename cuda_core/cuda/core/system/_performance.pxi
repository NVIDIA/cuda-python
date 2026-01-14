# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0


# In cuda.bindings.nvml, this is an anonymous struct inside nvmlGpuDynamicPstatesInfo_t.


ctypedef struct _GpuDynamicPstatesUtilization:
    unsigned int bIsPresent
    unsigned int percentage
    unsigned int incThreshold
    unsigned int decThreshold


cdef class GpuDynamicPstatesUtilization:
    cdef:
        _GpuDynamicPstatesUtilization *_ptr
        object _owner

    def __init__(self, ptr: int, owner: object):
        self._ptr = <_GpuDynamicPstatesUtilization *><intptr_t>ptr
        self._owner = owner

    @property
    def is_present(self) -> bool:
        """
        Set if the utilization domain is present on this GPU.
        """
        return bool(self._ptr[0].bIsPresent)

    @property
    def percentage(self) -> int:
        """
        Percentage of time where the domain is considered busy in the last 1-second interval.
        """
        return self._ptr[0].percentage

    @property
    def inc_threshold(self) -> int:
        """
        Utilization threshold that can trigger a perf-increasing P-State change when crossed.
        """
        return self._ptr[0].incThreshold

    @property
    def dec_threshold(self) -> int:
        """
        Utilization threshold that can trigger a perf-decreasing P-State change when crossed.
        """
        return self._ptr[0].decThreshold


cdef class GpuDynamicPstatesInfo:
    """
    Handles performance monitor samples from the device.
    """
    cdef object _gpu_dynamic_pstates_info

    def __init__(self, gpu_dynamic_pstates_info: nvml.GpuDynamicPstatesInfo):
        self._gpu_dynamic_pstates_info = gpu_dynamic_pstates_info

    def __len__(self):
        return nvml.MAX_GPU_UTILIZATIONS

    def __getitem__(self, idx: int) -> GpuDynamicPstatesUtilization:
        if idx < 0 or idx >= nvml.MAX_GPU_UTILIZATIONS:
            raise IndexError("GPU dynamic P-states index out of range")
        return GpuDynamicPstatesUtilization(
            self._gpu_dynamic_pstates_info.utilization.ptr + idx * sizeof(_GpuDynamicPstatesUtilization),
            self._gpu_dynamic_pstates_info
        )
