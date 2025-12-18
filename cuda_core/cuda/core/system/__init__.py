# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: F403, F405


__all__ = [
    "get_driver_version",
    "get_driver_version_full",
    "get_gpu_driver_version",
    "get_num_devices",
    "get_process_name",
    "HAS_WORKING_NVML",
]


from .system import *

if HAS_WORKING_NVML:
    from cuda.bindings import _nvml

    from ._nvml_context import initialize
    from .device import Device, DeviceArchitecture

    initialize()

    __all__.extend(["Device", "DeviceArchitecture"])
