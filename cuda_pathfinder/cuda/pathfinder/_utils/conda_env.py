# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import functools
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

# https://docs.conda.io/projects/conda-build/en/stable/user-guide/environment-variables.html

BUILD_STATES = ("RENDER", "BUILD", "TEST")


@dataclass(frozen=True)
class CondaPrefix:
    env_state: Literal["RENDER", "BUILD", "TEST", "activated"]
    path: Path


@functools.cache
def get_conda_prefix() -> Optional[CondaPrefix]:
    """
    Return the effective conda prefix.
    - RENDER, BUILD, TEST: inside conda-build (host prefix at $PREFIX)
    - activated: user-activated env ($CONDA_PREFIX)
    - None: neither detected
    """
    state = os.getenv("CONDA_BUILD_STATE")
    if state:
        if state in BUILD_STATES:
            p = os.getenv("PREFIX")
            if p:
                return CondaPrefix(state, Path(p))  # type: ignore[arg-type]
        return None

    cp = os.getenv("CONDA_PREFIX")
    if cp:
        return CondaPrefix("activated", Path(cp))

    return None
