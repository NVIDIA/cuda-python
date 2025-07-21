# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Callable, Optional

from cuda.pathfinder._dynamic_libs.supported_nvidia_libs import DIRECT_DEPENDENCIES


class DynamicLibNotFoundError(RuntimeError):
    pass


@dataclass
class LoadedDL:
    abs_path: Optional[str]
    was_already_loaded_from_elsewhere: bool
    _handle_uint: int  # Platform-agnostic unsigned pointer value


def load_dependencies(libname: str, load_func: Callable[[str], LoadedDL]) -> None:
    for dep in DIRECT_DEPENDENCIES.get(libname, ()):
        load_func(dep)
