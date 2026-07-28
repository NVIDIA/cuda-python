# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public availability queries backed by the dynamic-library descriptor catalog."""

from __future__ import annotations

from cuda.pathfinder._dynamic_libs.descriptor_catalog import WindowsArch
from cuda.pathfinder._dynamic_libs.lib_descriptor import LIB_DESCRIPTORS
from cuda.pathfinder._dynamic_libs.load_dl_common import DynamicLibUnknownError


def windows_supported_arches(libname: str) -> tuple[WindowsArch, ...]:
    """Return the Windows target architectures supported for ``libname``.

    The result is authored descriptor metadata and is independent of the
    current operating system, host architecture, and Python interpreter.
    A known library that is unavailable on Windows returns an empty tuple.

    Args:
        libname: Short NVIDIA dynamic-library name, such as ``"cudart"``.

    Raises:
        DynamicLibUnknownError: If ``libname`` is not in the descriptor catalog.
    """
    desc = LIB_DESCRIPTORS.get(libname)
    if desc is None:
        raise DynamicLibUnknownError(f"Unknown library name: {libname!r}. Known names: {sorted(LIB_DESCRIPTORS)}")
    return desc.supported_windows_arch
