# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Public type aliases and protocols used in cuda.core API signatures."""

from typing import Literal as _Literal

from cuda.core._context import DeviceResourcesType
from cuda.core._stream import IsStreamType
from cuda.core._utils.cuda_utils import driver

__all__ = [
    "DevicePointerType",
    "DeviceResourcesType",
    "IsStreamType",
    "ProcessStateType",
]


# A type union of :obj:`~driver.CUdeviceptr`, `int` and `None` for hinting
# :attr:`Buffer.handle`.
DevicePointerType = driver.CUdeviceptr | int | None


ProcessStateType = _Literal["running", "locked", "checkpointed", "failed"]
