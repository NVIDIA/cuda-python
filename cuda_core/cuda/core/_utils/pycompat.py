# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""
Compatibility shims for differences between supported Python versions.
"""

import sys

__all__ = ["BufferProtocol"]


if sys.version_info >= (3, 12):
    from collections.abc import Buffer as BufferProtocol
else:
    BufferProtocol = object
