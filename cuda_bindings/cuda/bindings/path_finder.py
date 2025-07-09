# Copyright 2024-2025 NVIDIA Corporation.  All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

from cuda.pathfinder import SUPPORTED_NVIDIA_LIBNAMES as _SUPPORTED_LIBNAMES
from cuda.pathfinder import load_nvidia_dynamic_lib

__all__ = [
    "_load_nvidia_dynamic_library",
    "_SUPPORTED_LIBNAMES",
]

# All code below this line is for TEMPORARY BACKWARD COMPATIBILITY only.

from dataclasses import dataclass
from typing import Optional

from cuda.pathfinder._dynamic_libs import supported_nvidia_libs

if supported_nvidia_libs.IS_WINDOWS:
    import pywintypes

    from cuda.pathfinder._dynamic_libs.load_dl_windows import POINTER_ADDRESS_SPACE

    def _unsigned_int_to_pywintypes_handle(handle_uint: int) -> pywintypes.HANDLE:
        handle_int = handle_uint - POINTER_ADDRESS_SPACE if handle_uint >= POINTER_ADDRESS_SPACE // 2 else handle_uint
        return pywintypes.HANDLE(handle_int)

    HandleType = pywintypes.HANDLE
else:
    HandleType = int


# Original implementation, before making handle private as _handle_uint.
@dataclass
class LoadedDL:
    handle: HandleType  # type: ignore[valid-type]
    abs_path: Optional[str]
    was_already_loaded_from_elsewhere: bool


def _load_nvidia_dynamic_library(libname: str) -> LoadedDL:
    loaded_dl_uint = load_nvidia_dynamic_lib(libname)
    if supported_nvidia_libs.IS_WINDOWS:
        handle = _unsigned_int_to_pywintypes_handle(loaded_dl_uint._handle_uint)
    else:
        handle = loaded_dl_uint._handle_uint
    return LoadedDL(handle, loaded_dl_uint.abs_path, loaded_dl_uint.was_already_loaded_from_elsewhere)
