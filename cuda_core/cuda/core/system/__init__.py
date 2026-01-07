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
    "get_gpu_driver_version",
    "get_num_devices",
    "get_process_name",
    "HAS_WORKING_NVML",
]


from .system import *

if HAS_WORKING_NVML:
    from ._nvml_context import initialize
    from .device import Device, DeviceArchitecture
    from .exceptions import *

    __all__.extend(
        [
            "initialize",
            "get_nvml_version",
            "Device",
            "DeviceArchitecture",
            "UninitializedError",
            "InvalidArgumentError",
            "NotSupportedError",
            "NoPermissionError",
            "AlreadyInitializedError",
            "NotFoundError",
            "InsufficientSizeError",
            "InsufficientPowerError",
            "DriverNotLoadedError",
            "TimeoutError",
            "IrqIssueError",
            "LibraryNotFoundError",
            "FunctionNotFoundError",
            "CorruptedInforomError",
            "GpuIsLostError",
            "ResetRequiredError",
            "OperatingSystemError",
            "LibRmVersionMismatchError",
            "InUseError",
            "MemoryError",
            "NoDataError",
            "VgpuEccNotSupportedError",
            "InsufficientResourcesError",
            "FreqNotSupportedError",
            "ArgumentVersionMismatchError",
            "DeprecatedError",
            "NotReadyError",
            "GpuNotFoundError",
            "InvalidStateError",
            "ResetTypeNotSupportedError",
            "UnknownError",
        ]
    )
