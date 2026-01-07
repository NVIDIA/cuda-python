# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: F403, F405


# NOTE: We must maintaint that it is always possible to import this module
# without CUDA being installed, and without CUDA being initialized or any
# contexts created, so that a user can use NVML to explore things about their
# system without loading CUDA.


__all__ = [
    "get_driver_version",
    "get_driver_version_full",
    "get_num_devices",
    "get_process_name",
    "CUDA_BINDINGS_NVML_IS_COMPATIBLE",
]


from ._system import *

if CUDA_BINDINGS_NVML_IS_COMPATIBLE:
    from ._device import Device, DeviceArchitecture
    from .exceptions import *
    from .exceptions import __all__ as _exceptions_all

    __all__.extend(
        [
            "get_nvml_version",
            "Device",
            "DeviceArchitecture",
        ]
    )

    __all__.extend(_exceptions_all)
