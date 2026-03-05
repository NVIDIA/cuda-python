# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Platform loader seam for OS-specific dynamic linking.

This module provides a small interface that hides the Linux vs Windows
implementation details of:

- already-loaded checks
- system-search loading
- absolute-path loading

The orchestration logic in :mod:`load_nvidia_dynamic_lib` should not need to
branch on platform; it calls through the loader instance exported here.
"""

from __future__ import annotations

from typing import Protocol

from cuda.pathfinder._dynamic_libs.lib_descriptor import LibDescriptor
from cuda.pathfinder._dynamic_libs.load_dl_common import LoadedDL
from cuda.pathfinder._utils.platform_aware import IS_WINDOWS


class PlatformLoader(Protocol):
    def check_if_already_loaded_from_elsewhere(self, desc: LibDescriptor, have_abs_path: bool) -> LoadedDL | None: ...

    def load_with_system_search(self, desc: LibDescriptor) -> LoadedDL | None: ...

    def load_with_abs_path(self, desc: LibDescriptor, found_path: str, found_via: str | None = None) -> LoadedDL: ...


if IS_WINDOWS:
    from cuda.pathfinder._dynamic_libs import load_dl_windows as _impl
else:
    from cuda.pathfinder._dynamic_libs import load_dl_linux as _impl

# The platform modules already expose functions matching the PlatformLoader
# protocol. Wrap in a simple namespace so callers use LOADER.method() syntax.
LOADER: PlatformLoader = _impl
