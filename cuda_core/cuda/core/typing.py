# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Public type aliases and protocols used in cuda.core API signatures."""

from cuda.core._memory._buffer import DevicePointerT
from cuda.core._memory._virtual_memory_resource import (
    VirtualMemoryAccessTypeT,
    VirtualMemoryAllocationTypeT,
    VirtualMemoryGranularityT,
    VirtualMemoryHandleTypeT,
    VirtualMemoryLocationTypeT,
)
from cuda.core._stream import IsStreamT

__all__ = [
    "DevicePointerT",
    "IsStreamT",
    "VirtualMemoryAccessTypeT",
    "VirtualMemoryAllocationTypeT",
    "VirtualMemoryGranularityT",
    "VirtualMemoryHandleTypeT",
    "VirtualMemoryLocationTypeT",
]
