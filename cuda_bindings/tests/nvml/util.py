# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE


import platform

from cuda.bindings import _nvml as nvml

current_os = platform.system()
if current_os == "VMkernel":
    current_os = "Linux"  # Treat VMkernel as Linux


def is_windows(os=current_os):
    return os == "Windows"


def is_linux(os=current_os):
    return os == "Linux"


def is_vgpu(device):
    """
    Returns True if device in vGPU virtualization mode
    """
    return nvml.device_get_virtualization_mode(device) == nvml.GpuVirtualizationMode.VGPU


def supports_ecc(device):
    try:
        (cur_ecc, pend_ecc) = nvml.device_get_ecc_mode(device)
        return cur_ecc != nvml.EnableState.FEATURE_DISABLED
    except nvml.NotSupportedError as e:
        return False


def supports_nvlink(device):
    fields = nvml.FieldValue(1)
    fields[0].field_id = nvml.FI.DEV_NVLINK_GET_STATE
    return nvml.device_get_field_values(device, fields)[0].nvml_return == nvml.Return.SUCCESS
