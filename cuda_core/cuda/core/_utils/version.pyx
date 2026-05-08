# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import functools
import importlib.metadata

from cuda.core._utils.cuda_utils import driver, handle_return


@functools.cache
def binding_version() -> tuple[int, int, int]:
    """Return the cuda-bindings version as a (major, minor, patch) triple."""
    try:
        parts = importlib.metadata.version("cuda-bindings").split(".")[:3]
    except importlib.metadata.PackageNotFoundError:
        parts = importlib.metadata.version("cuda-python").split(".")[:3]
    return tuple(int(v) for v in parts)


@functools.cache
def driver_version() -> tuple[int, int, int]:
    """Return the CUDA driver version as a (major, minor, patch) triple."""
    cdef int ver = handle_return(driver.cuDriverGetVersion())
    return (ver // 1000, (ver // 10) % 100, ver % 10)


cdef tuple _cached_binding_version = None
cdef tuple _cached_driver_version = None


cdef tuple cy_binding_version():
    global _cached_binding_version
    if _cached_binding_version is None:
        _cached_binding_version = binding_version()
    return _cached_binding_version


cdef tuple cy_driver_version():
    global _cached_driver_version
    if _cached_driver_version is None:
        _cached_driver_version = driver_version()
    return _cached_driver_version
