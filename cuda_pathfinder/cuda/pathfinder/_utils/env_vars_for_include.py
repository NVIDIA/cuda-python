# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import sys
from collections.abc import Iterable

IS_WINDOWS = sys.platform == "win32"

# GCC/Clang-style include vars
GCC_VNAMES = ("CPATH", "C_INCLUDE_PATH", "CPLUS_INCLUDE_PATH")

# MSVC: INCLUDE is the canonical header search variable
MSVC_GCC_VNAMES = ("INCLUDE",)

VNAMES: tuple[str, ...] = MSVC_GCC_VNAMES + GCC_VNAMES if IS_WINDOWS else GCC_VNAMES


def iter_env_vars_for_include_dirs() -> Iterable[str]:
    for vname in VNAMES:
        v = os.getenv(vname)
        if v:
            for d in v.split(os.pathsep):
                if d:
                    yield d
