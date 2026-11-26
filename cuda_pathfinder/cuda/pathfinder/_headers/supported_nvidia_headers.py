# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CTK header-name tables derived from the authored header descriptor catalog.

The canonical data entry point is :mod:`header_descriptor_catalog`. This module
holds only the projection that the package itself reads.
"""

from __future__ import annotations

from typing import Final

from cuda.pathfinder._headers.header_descriptor_catalog import HEADER_DESCRIPTOR_CATALOG
from cuda.pathfinder._utils.platform_aware import IS_WINDOWS

_CTK_DESCRIPTORS = tuple(desc for desc in HEADER_DESCRIPTOR_CATALOG if desc.packaged_with == "ctk")

SUPPORTED_HEADERS_CTK_COMMON: Final[dict[str, str]] = {
    desc.name: desc.header_basename
    for desc in _CTK_DESCRIPTORS
    if desc.available_on_linux and desc.available_on_windows
}
SUPPORTED_HEADERS_CTK_LINUX_ONLY: Final[dict[str, str]] = {
    desc.name: desc.header_basename
    for desc in _CTK_DESCRIPTORS
    if desc.available_on_linux and not desc.available_on_windows
}
SUPPORTED_HEADERS_CTK_WINDOWS_ONLY: Final[dict[str, str]] = {
    desc.name: desc.header_basename
    for desc in _CTK_DESCRIPTORS
    if desc.available_on_windows and not desc.available_on_linux
}

SUPPORTED_HEADERS_CTK_LINUX = SUPPORTED_HEADERS_CTK_COMMON | SUPPORTED_HEADERS_CTK_LINUX_ONLY
SUPPORTED_HEADERS_CTK_WINDOWS = SUPPORTED_HEADERS_CTK_COMMON | SUPPORTED_HEADERS_CTK_WINDOWS_ONLY
SUPPORTED_HEADERS_CTK: Final[dict[str, str]] = (
    SUPPORTED_HEADERS_CTK_WINDOWS if IS_WINDOWS else SUPPORTED_HEADERS_CTK_LINUX
)
