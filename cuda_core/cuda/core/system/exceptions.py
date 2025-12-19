# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0


from cuda.bindings import _nvml as nvml

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
