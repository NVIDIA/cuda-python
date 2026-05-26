# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Public type aliases, protocols, and enumerations used in cuda.core API signatures."""

try:
    from enum import StrEnum
except ImportError:
    from backports.strenum import StrEnum
from typing import Literal as _Literal

from cuda.core._context import DeviceResourcesType
from cuda.core._stream import IsStreamType
from cuda.core._utils.cuda_utils import driver

__all__ = [
    "CompilerBackendType",
    "DevicePointerType",
    "DeviceResourcesType",
    "GraphConditionalType",
    "GraphMemoryType",
    "IsStreamType",
    "ManagedMemoryLocationType",
    "ObjectCodeFormatType",
    "PCHStatusType",
    "ProcessStateType",
    "SourceCodeType",
    "VirtualMemoryAccessType",
    "VirtualMemoryAllocationType",
    "VirtualMemoryGranularityType",
    "VirtualMemoryHandleType",
    "VirtualMemoryLocationType",
]


# A type union of :obj:`~driver.CUdeviceptr`, `int` and `None` for hinting
# :attr:`Buffer.handle`.
DevicePointerType = driver.CUdeviceptr | int | None


ProcessStateType = _Literal["running", "locked", "checkpointed", "failed"]


class SourceCodeType(StrEnum):
    CXX = "c++"
    PTX = "ptx"
    NVVM = "nvvm"


class ObjectCodeFormatType(StrEnum):
    PTX = "ptx"
    CUBIN = "cubin"
    LTOIR = "ltoir"
    FATBIN = "fatbin"
    OBJECT = "object"
    LIBRARY = "library"


class CompilerBackendType(StrEnum):
    NVRTC = "NVRTC"
    NVVM = "NVVM"
    NVJITLINK = "nvJitLink"
    DRIVER = "driver"


class PCHStatusType(StrEnum):
    CREATED = "created"
    NOT_ATTEMPTED = "not_attempted"
    FAILED = "failed"


class GraphConditionalType(StrEnum):
    IF = "if"
    WHILE = "while"
    SWITCH = "switch"


class GraphMemoryType(StrEnum):
    DEVICE = "device"
    HOST = "host"
    MANAGED = "managed"


class ManagedMemoryLocationType(StrEnum):
    DEVICE = "device"
    HOST = "host"
    HOST_NUMA = "host_numa"


class VirtualMemoryHandleType(StrEnum):
    POSIX_FD = "posix_fd"
    WIN32_KMT = "win32_kmt"
    FABRIC = "fabric"


class VirtualMemoryLocationType(StrEnum):
    DEVICE = "device"
    HOST = "host"
    HOST_NUMA = "host_numa"
    HOST_NUMA_CURRENT = "host_numa_current"


class VirtualMemoryGranularityType(StrEnum):
    MINIMUM = "minimum"
    RECOMMENDED = "recommended"


class VirtualMemoryAccessType(StrEnum):
    READ_WRITE = "rw"
    READ = "r"


class VirtualMemoryAllocationType(StrEnum):
    PINNED = "pinned"
    MANAGED = "managed"


del StrEnum
