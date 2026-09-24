# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import functools
import importlib.metadata
import re

from cuda.core._utils.cuda_utils import driver, handle_return

# The CUDA major series that this build of cuda.core targets, 12 or 13. The
# value comes from the compile-time environment that build_hooks.py sets. The
# installed cuda-bindings has the same major, which cuda/core/__init__.py
# enforces at import. Python modules that must branch on the series, where
# `IF CUDA_CORE_BUILD_MAJOR` is not available, read this constant rather than
# compare binding_version().
BUILD_CUDA_MAJOR: int = CUDA_CORE_BUILD_MAJOR


def _parse_version_triple(version_str: str) -> tuple[int, int, int]:
    """Parse a PEP 440 version string into a (major, minor, patch) triple.

    Strips local-version identifiers and handles pre-release suffixes such as
    ``0b1`` or ``0rc1`` by extracting only the leading integer from each
    release segment.
    """
    parts = version_str.partition("+")[0].split(".")[:3]
    ints = ([int(m.group(1)) if (m := re.match(r"(\d+)", v)) else 0 for v in parts] + [0, 0, 0])[:3]
    return (ints[0], ints[1], ints[2])


@functools.cache
def binding_version() -> tuple[int, int, int]:
    """Return the cuda-bindings version as a (major, minor, patch) triple."""
    try:
        version_str = importlib.metadata.version("cuda-bindings")
    except importlib.metadata.PackageNotFoundError:
        version_str = importlib.metadata.version("cuda-python")
    return _parse_version_triple(version_str)


@functools.cache
def driver_version() -> tuple[int, int, int]:
    """Return the CUDA driver version as a (major, minor, patch) triple."""
    cdef int ver = handle_return(driver.cuDriverGetVersion())
    return (ver // 1000, (ver // 10) % 100, ver % 10)


cdef tuple _cached_driver_version = None


cdef tuple cy_driver_version():
    global _cached_driver_version
    if _cached_driver_version is None:
        _cached_driver_version = driver_version()
    return _cached_driver_version
