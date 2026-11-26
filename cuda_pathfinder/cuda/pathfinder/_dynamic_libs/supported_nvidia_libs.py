# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Library-name sets derived from the authored descriptor catalog.

The canonical data entry point is :mod:`descriptor_catalog`. This module holds
only the projections that the package itself reads.
"""

from __future__ import annotations

from cuda.pathfinder._dynamic_libs.descriptor_catalog import DESCRIPTOR_CATALOG
from cuda.pathfinder._utils.platform_aware import IS_WINDOWS, IS_WINDOWS_ARM64, IS_WINDOWS_X64

_CTK_DESCRIPTORS = tuple(desc for desc in DESCRIPTOR_CATALOG if desc.packaged_with == "ctk")


SUPPORTED_LIBNAMES_COMMON = tuple(desc.name for desc in _CTK_DESCRIPTORS if desc.linux_sonames and desc.windows_dlls)
SUPPORTED_LIBNAMES_LINUX_ONLY = tuple(
    desc.name for desc in _CTK_DESCRIPTORS if desc.linux_sonames and not desc.windows_dlls
)

if not IS_WINDOWS:
    ALL_AVAILABLE_LIBNAMES = frozenset(desc.name for desc in DESCRIPTOR_CATALOG if desc.linux_sonames)
else:
    assert IS_WINDOWS_X64 != IS_WINDOWS_ARM64
    _current_windows_arch = "x64" if IS_WINDOWS_X64 else "arm64"
    ALL_AVAILABLE_LIBNAMES = frozenset(
        desc.name for desc in DESCRIPTOR_CATALOG if _current_windows_arch in desc.supported_windows_arch
    )

SUPPORTED_LIBNAMES_LINUX = SUPPORTED_LIBNAMES_COMMON + SUPPORTED_LIBNAMES_LINUX_ONLY
SUPPORTED_LIBNAMES_WINDOWS_X64 = tuple(desc.name for desc in _CTK_DESCRIPTORS if "x64" in desc.supported_windows_arch)
SUPPORTED_LIBNAMES_WINDOWS_ARM64 = tuple(
    desc.name for desc in _CTK_DESCRIPTORS if "arm64" in desc.supported_windows_arch
)
if not IS_WINDOWS:
    SUPPORTED_LIBNAMES = SUPPORTED_LIBNAMES_LINUX
elif IS_WINDOWS_X64:
    SUPPORTED_LIBNAMES = SUPPORTED_LIBNAMES_WINDOWS_X64
else:
    assert IS_WINDOWS_ARM64
    SUPPORTED_LIBNAMES = SUPPORTED_LIBNAMES_WINDOWS_ARM64
