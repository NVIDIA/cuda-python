# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Public type aliases and protocols used in cuda.core API signatures."""

from typing import Literal as _Literal

from cuda.core._memory._buffer import DevicePointerT
from cuda.core._stream import IsStreamT

ProcessStateT = _Literal["running", "locked", "checkpointed", "failed"]

__all__ = [
    "DevicePointerT",
    "IsStreamT",
    "ProcessStateT",
]
