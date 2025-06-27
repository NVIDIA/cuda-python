# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from cuda.pathfinder._dynamic_libs import load_nvidia_dynamic_lib as _load_nvidia_dynamic_lib
from cuda.pathfinder._dynamic_libs.load_dl_common import LoadedDL
from cuda.pathfinder._dynamic_libs.supported_nvidia_libs import SUPPORTED_LIBNAMES as SUPPORTED_NVIDIA_LIBNAMES

__all__ = ["SUPPORTED_NVIDIA_LIBNAMES", "load_nvidia_dynamic_lib"]


def load_nvidia_dynamic_lib(libname: str) -> LoadedDL:
    """Load a NVIDIA dynamic library by name.

    Args:
        libname: The name of the library to load (e.g. "cudart", "nvvm", etc.)

    Returns:
        A LoadedDL object containing the library handle and path

    Raises:
        RuntimeError: If the library cannot be found or loaded
    """
    return _load_nvidia_dynamic_lib.load_lib(libname)
