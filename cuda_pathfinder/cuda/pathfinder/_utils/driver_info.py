# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ctypes
from collections.abc import Callable
from typing import cast

from cuda.pathfinder._dynamic_libs.load_nvidia_dynamic_lib import (
    load_nvidia_dynamic_lib as _load_nvidia_dynamic_lib,
)
from cuda.pathfinder._utils.platform_aware import IS_WINDOWS


def _query_driver_version() -> int:
    """Return the CUDA driver version from ``cuDriverGetVersion()``."""
    loaded_cuda = _load_nvidia_dynamic_lib("cuda")
    if loaded_cuda.abs_path is None:
        raise RuntimeError('Could not determine an absolute path for the driver library "cuda".')
    if IS_WINDOWS:
        loader_cls_obj = vars(ctypes).get("WinDLL")
        if loader_cls_obj is None:
            raise RuntimeError("ctypes.WinDLL is unavailable on this platform.")
        loader_cls = cast(Callable[[str], ctypes.CDLL], loader_cls_obj)
    else:
        loader_cls = ctypes.CDLL
    driver_lib = loader_cls(loaded_cuda.abs_path)
    cu_driver_get_version = driver_lib.cuDriverGetVersion
    cu_driver_get_version.argtypes = [ctypes.POINTER(ctypes.c_int)]
    cu_driver_get_version.restype = ctypes.c_int
    version = ctypes.c_int()
    status = cu_driver_get_version(ctypes.byref(version))
    if status != 0:
        raise RuntimeError(f"Failed to query CUDA driver version via cuDriverGetVersion() (status={status}).")
    return version.value
