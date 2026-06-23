# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Public type aliases, protocols, and enumerations used in cuda.core API signatures."""

import sys
from typing import TYPE_CHECKING
from typing import Literal as _Literal
from typing import TypeAlias as _TypeAlias

if TYPE_CHECKING:
    # `backports.strenum` ships no type stubs and typeshed conditionally gates
    # `enum.StrEnum` behind `sys.version_info >= (3, 11)`. Declaring a minimal
    # local shape here (mirroring typeshed's 3.11 StrEnum) lets mypy at
    # `python_version = "3.10"` infer subclass members as `Literal[Foo.MEMBER]`
    # rather than bare `str`.
    from enum import Enum

    class StrEnum(str, Enum):
        _value_: str


if not TYPE_CHECKING:
    if sys.version_info >= (3, 11):
        from enum import StrEnum
    else:
        from backports.strenum import StrEnum

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
DevicePointerType: _TypeAlias = driver.CUdeviceptr | int | None


ProcessStateType = _Literal["running", "locked", "checkpointed", "failed"]


class SourceCodeType(StrEnum):
    """Source language passed to :class:`~cuda.core.Program`.

    * ``CXX`` — CUDA C++ source.
    * ``PTX`` — PTX assembly text.
    * ``NVVM`` — NVVM IR (LLVM bitcode).
    """

    CXX = "c++"
    PTX = "ptx"
    NVVM = "nvvm"


class ObjectCodeFormatType(StrEnum):
    """Output format for :meth:`~cuda.core.Program.compile`, :meth:`~cuda.core.Linker.link`, and :meth:`~cuda.core.Program.as_bytes`.

    * ``PTX`` — PTX assembly text.
    * ``CUBIN`` — device-native CUDA binary.
    * ``LTOIR`` — LTO (link-time optimization) IR for later linking.
    * ``FATBIN`` — fat binary bundling multiple device images.
    * ``OBJECT`` — relocatable device object.
    * ``LIBRARY`` — device code library.
    """

    PTX = "ptx"
    CUBIN = "cubin"
    LTOIR = "ltoir"
    FATBIN = "fatbin"
    OBJECT = "object"
    LIBRARY = "library"


class CompilerBackendType(StrEnum):
    """Compiler backend inferred from the program's code type and exposed on :attr:`~cuda.core.Program.backend`.

    * ``NVRTC`` — NVIDIA Runtime Compilation.
    * ``NVVM`` — NVVM LLVM backend.
    * ``NVJITLINK`` — nvJitLink device-side linker.
    * ``DRIVER`` — CUDA driver PTX JIT compiler.
    """

    NVRTC = "NVRTC"
    NVVM = "NVVM"
    NVJITLINK = "nvJitLink"
    DRIVER = "driver"


class PCHStatusType(StrEnum):
    """Precompiled-header (PCH) outcome reported by :meth:`~cuda.core.Program.compile`.

    * ``CREATED`` — PCH was successfully written.
    * ``NOT_ATTEMPTED`` — PCH creation was skipped (backend does not support it or the option was not requested).
    * ``FAILED`` — PCH creation was attempted but failed.
    """

    CREATED = "created"
    NOT_ATTEMPTED = "not_attempted"
    FAILED = "failed"


class GraphConditionalType(StrEnum):
    """Conditional node flavor for :class:`~cuda.core.graph.GraphBuilder`.

    * ``IF`` — body graph executes at most once based on a condition.
    * ``WHILE`` — body graph loops while the condition is true.
    * ``SWITCH`` — selects one child graph by an integer index.
    """

    IF = "if"
    WHILE = "while"
    SWITCH = "switch"


class GraphMemoryType(StrEnum):
    """Memory space for a graph memory-allocation or free node.

    * ``DEVICE`` — GPU device memory.
    * ``HOST`` — pinned host memory.
    * ``MANAGED`` — CUDA managed (unified) memory.
    """

    DEVICE = "device"
    HOST = "host"
    MANAGED = "managed"


class ManagedMemoryLocationType(StrEnum):
    """Destination type for managed-memory prefetch and advise operations.

    * ``DEVICE`` — target a GPU device.
    * ``HOST`` — target the CPU host (any NUMA node).
    * ``HOST_NUMA`` — target a specific host NUMA node.
    """

    DEVICE = "device"
    HOST = "host"
    HOST_NUMA = "host_numa"


class VirtualMemoryHandleType(StrEnum):
    """OS handle type for exporting virtual memory allocations across processes.

    * ``POSIX_FD`` — POSIX file descriptor (Linux).
    * ``WIN32_KMT`` — Win32 kernel-mode handle (Windows).
    * ``FABRIC`` — NVLink/NVSwitch fabric handle for multi-node topologies.
    """

    POSIX_FD = "posix_fd"
    WIN32_KMT = "win32_kmt"
    FABRIC = "fabric"


class VirtualMemoryLocationType(StrEnum):
    """Physical backing location for a virtual memory allocation.

    * ``DEVICE`` — GPU device memory.
    * ``HOST`` — pinned host memory.
    * ``HOST_NUMA`` — host memory pinned to a specific NUMA node.
    * ``HOST_NUMA_CURRENT`` — host memory on the calling thread's NUMA node.
    """

    DEVICE = "device"
    HOST = "host"
    HOST_NUMA = "host_numa"
    HOST_NUMA_CURRENT = "host_numa_current"


class VirtualMemoryGranularityType(StrEnum):
    """Granularity query type for virtual memory allocations.

    * ``MINIMUM`` — smallest allocation size supported by the device.
    * ``RECOMMENDED`` — granularity that yields best performance on the device.
    """

    MINIMUM = "minimum"
    RECOMMENDED = "recommended"


class VirtualMemoryAccessType(StrEnum):
    """Access permissions for a virtual memory mapping.

    * ``READ_WRITE`` — both read and write access.
    * ``READ`` — read-only access.
    """

    READ_WRITE = "rw"
    READ = "r"


class VirtualMemoryAllocationType(StrEnum):
    """Physical memory type for a virtual memory backing allocation.

    * ``PINNED`` — pinned/non-migratable physical allocation (placement via :class:`VirtualMemoryLocationType`).
    * ``MANAGED`` — CUDA managed (unified) memory (CUDA 13+ only).
    """

    PINNED = "pinned"
    MANAGED = "managed"


del StrEnum
