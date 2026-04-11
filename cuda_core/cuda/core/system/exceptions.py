# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0


from cuda.bindings import nvml

NvmlError = nvml.NvmlError
UninitializedError = nvml.UninitializedError
InvalidArgumentError = nvml.InvalidArgumentError
NotSupportedError = nvml.NotSupportedError
NoPermissionError = nvml.NoPermissionError
AlreadyInitializedError = nvml.AlreadyInitializedError
NotFoundError = nvml.NotFoundError
InsufficientSizeError = nvml.InsufficientSizeError
InsufficientPowerError = nvml.InsufficientPowerError
DriverNotLoadedError = nvml.DriverNotLoadedError
TimeoutError = nvml.TimeoutError
IrqIssueError = nvml.IrqIssueError
LibraryNotFoundError = nvml.LibraryNotFoundError
FunctionNotFoundError = nvml.FunctionNotFoundError
CorruptedInforomError = nvml.CorruptedInforomError
GpuIsLostError = nvml.GpuIsLostError
ResetRequiredError = nvml.ResetRequiredError
OperatingSystemError = nvml.OperatingSystemError
LibRmVersionMismatchError = nvml.LibRmVersionMismatchError
InUseError = nvml.InUseError
MemoryError = nvml.MemoryError
NoDataError = nvml.NoDataError
VgpuEccNotSupportedError = nvml.VgpuEccNotSupportedError
InsufficientResourcesError = nvml.InsufficientResourcesError
FreqNotSupportedError = nvml.FreqNotSupportedError
ArgumentVersionMismatchError = nvml.ArgumentVersionMismatchError
DeprecatedError = nvml.DeprecatedError
NotReadyError = nvml.NotReadyError
GpuNotFoundError = nvml.GpuNotFoundError
InvalidStateError = nvml.InvalidStateError
ResetTypeNotSupportedError = nvml.ResetTypeNotSupportedError
UnknownError = nvml.UnknownError


__all__ = [
    "AlreadyInitializedError",
    "ArgumentVersionMismatchError",
    "CorruptedInforomError",
    "DeprecatedError",
    "DriverNotLoadedError",
    "FreqNotSupportedError",
    "FunctionNotFoundError",
    "GpuIsLostError",
    "GpuNotFoundError",
    "InUseError",
    "InsufficientPowerError",
    "InsufficientResourcesError",
    "InsufficientSizeError",
    "InvalidArgumentError",
    "InvalidStateError",
    "IrqIssueError",
    "LibRmVersionMismatchError",
    "LibraryNotFoundError",
    "MemoryError",
    "NoDataError",
    "NoPermissionError",
    "NotFoundError",
    "NotReadyError",
    "NotSupportedError",
    "NvmlError",
    "OperatingSystemError",
    "ResetRequiredError",
    "ResetTypeNotSupportedError",
    "TimeoutError",
    "UninitializedError",
    "UnknownError",
    "VgpuEccNotSupportedError",
]
