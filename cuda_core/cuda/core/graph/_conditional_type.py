# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

try:
    from enum import StrEnum
except ImportError:
    from backports.strenum import StrEnum

__all__ = ["ConditionalType"]


# Since this enum uses `IF`, which is a reserved keyword in Cython, we have to
# define it in a separate Python file just for this purpose.
class ConditionalType(StrEnum):
    IF = "if"
    WHILE = "while"
    SWITCH = "switch"
