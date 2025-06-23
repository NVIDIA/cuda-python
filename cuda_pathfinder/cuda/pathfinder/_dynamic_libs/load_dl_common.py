# Copyright 2025 NVIDIA Corporation.  All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

from dataclasses import dataclass
from typing import Callable, Optional

from cuda.path_finder._dynamic_libs.supported_nvidia_libs import (
    DIRECT_DEPENDENCIES,
    IS_WINDOWS,
)

if IS_WINDOWS:
    import pywintypes

    HandleType = pywintypes.HANDLE
else:
    HandleType = int


@dataclass
class LoadedDL:
    handle: HandleType
    abs_path: Optional[str]
    was_already_loaded_from_elsewhere: bool


def load_dependencies(libname: str, load_func: Callable[[str], LoadedDL]) -> None:
    for dep in DIRECT_DEPENDENCIES.get(libname, ()):
        load_func(dep)
