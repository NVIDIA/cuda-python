# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-header descriptor, registry, and platform-aware accessors.

The canonical authored data lives in :mod:`header_descriptor_catalog`. This
module provides a name-keyed registry and platform-dispatch helpers consumed
by the runtime search path — keeping the search code itself platform-agnostic.
"""

from __future__ import annotations

import glob
import os
from typing import TypeAlias, cast

from cuda.pathfinder._headers.header_descriptor_catalog import (
    HEADER_DESCRIPTOR_CATALOG,
    HeaderDescriptorSpec,
)
from cuda.pathfinder._utils.platform_aware import IS_WINDOWS

HeaderDescriptor: TypeAlias = HeaderDescriptorSpec

#: Canonical registry of all known header libraries.
HEADER_DESCRIPTORS: dict[str, HeaderDescriptor] = {desc.name: desc for desc in HEADER_DESCRIPTOR_CATALOG}


def platform_include_subdirs(desc: HeaderDescriptor) -> tuple[str, ...]:
    """Return the effective include subdirectory search list for the current platform.

    On Windows, Windows-specific subdirs are checked first, followed by the
    common subdirs.  On Linux, only the common subdirs are returned.
    """
    if IS_WINDOWS:
        return cast(tuple[str, ...], desc.include_subdirs_windows + desc.include_subdirs)
    return cast(tuple[str, ...], desc.include_subdirs)


def resolve_conda_anchor(desc: HeaderDescriptor, conda_prefix: str) -> str | None:
    """Resolve the conda anchor point for header search on the current platform.

    Returns the directory that ``_locate_in_anchor_layout`` should use as
    *anchor_point*, or ``None`` if the conda layout is not usable.
    """
    if IS_WINDOWS:
        anchor = os.path.join(conda_prefix, "Library")
        return anchor if os.path.isdir(anchor) else None
    if desc.conda_targets_layout:
        targets_include_path = glob.glob(os.path.join(conda_prefix, "targets", "*", "include"))
        if not targets_include_path or len(targets_include_path) != 1:
            return None
        return os.path.dirname(targets_include_path[0])
    return conda_prefix
