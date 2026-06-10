# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""
Compatibility shims for differences between supported Python versions.
"""

import sys

__all__ = ["BufferProtocol", "StrEnum"]


if sys.version_info >= (3, 11):
    from enum import StrEnum
else:
    from backports.strenum import StrEnum


if sys.version_info >= (3, 12):
    from collections.abc import Buffer as BufferProtocol
else:
    BufferProtocol = object
