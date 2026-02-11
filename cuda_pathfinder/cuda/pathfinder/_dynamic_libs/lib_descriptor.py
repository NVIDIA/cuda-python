# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-library descriptor and registry.

Each NVIDIA library known to pathfinder is described by a single
:class:`LibDescriptor` instance.  The :data:`LIB_DESCRIPTORS` dict is the
canonical registry, keyed by short library name (e.g. ``"cudart"``).

This module is intentionally **read-only at runtime** — it assembles
descriptors from the existing data tables in
:mod:`~cuda.pathfinder._dynamic_libs.supported_nvidia_libs` so that all
behavioural contracts are preserved while giving consumers a single object
to query per library.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from cuda.pathfinder._dynamic_libs.supported_nvidia_libs import (
    DIRECT_DEPENDENCIES,
    LIBNAMES_REQUIRING_OS_ADD_DLL_DIRECTORY,
    LIBNAMES_REQUIRING_RTLD_DEEPBIND,
    SITE_PACKAGES_LIBDIRS_LINUX,
    SITE_PACKAGES_LIBDIRS_WINDOWS,
    SUPPORTED_LINUX_SONAMES,
    SUPPORTED_WINDOWS_DLLS,
)
from cuda.pathfinder._utils.platform_aware import IS_WINDOWS

Strategy = Literal["ctk", "other", "driver"]


@dataclass(frozen=True, slots=True)
class LibDescriptor:
    """Immutable description of an NVIDIA library known to pathfinder."""

    name: str
    strategy: Strategy

    # Platform-specific file names used by the system loader.
    linux_sonames: tuple[str, ...] = ()
    windows_dlls: tuple[str, ...] = ()

    # Relative directories under site-packages where pip wheels place the lib.
    site_packages_linux: tuple[str, ...] = ()
    site_packages_windows: tuple[str, ...] = ()

    # Libraries that must be loaded first.
    dependencies: tuple[str, ...] = ()

    # Platform-specific loader quirks.
    requires_add_dll_directory: bool = False
    requires_rtld_deepbind: bool = False

    # --- Derived helpers (not stored, computed on access) ---

    @property
    def sonames(self) -> tuple[str, ...]:
        """Platform-appropriate loader names."""
        return self.windows_dlls if IS_WINDOWS else self.linux_sonames

    @property
    def site_packages_dirs(self) -> tuple[str, ...]:
        """Platform-appropriate site-packages relative directories."""
        return self.site_packages_windows if IS_WINDOWS else self.site_packages_linux


def _classify_lib(name: str) -> Strategy:
    """Determine the search strategy for a library based on which dicts it appears in."""
    from cuda.pathfinder._dynamic_libs.supported_nvidia_libs import (
        SUPPORTED_LIBNAMES,
        SUPPORTED_LINUX_SONAMES_DRIVER,
        SUPPORTED_LINUX_SONAMES_OTHER,
        SUPPORTED_WINDOWS_DLLS_DRIVER,
        SUPPORTED_WINDOWS_DLLS_OTHER,
    )

    if name in SUPPORTED_LIBNAMES:
        return "ctk"
    if name in SUPPORTED_LINUX_SONAMES_DRIVER or name in SUPPORTED_WINDOWS_DLLS_DRIVER:
        return "driver"
    if name in SUPPORTED_LINUX_SONAMES_OTHER or name in SUPPORTED_WINDOWS_DLLS_OTHER:
        return "other"
    return "other"


def _build_registry() -> dict[str, LibDescriptor]:
    """Assemble one LibDescriptor per library from the existing data tables."""
    all_names: set[str] = set()
    all_names.update(SUPPORTED_LINUX_SONAMES)
    all_names.update(SUPPORTED_WINDOWS_DLLS)

    registry: dict[str, LibDescriptor] = {}
    for name in sorted(all_names):
        registry[name] = LibDescriptor(
            name=name,
            strategy=_classify_lib(name),
            linux_sonames=SUPPORTED_LINUX_SONAMES.get(name, ()),
            windows_dlls=SUPPORTED_WINDOWS_DLLS.get(name, ()),
            site_packages_linux=SITE_PACKAGES_LIBDIRS_LINUX.get(name, ()),
            site_packages_windows=SITE_PACKAGES_LIBDIRS_WINDOWS.get(name, ()),
            dependencies=DIRECT_DEPENDENCIES.get(name, ()),
            requires_add_dll_directory=name in LIBNAMES_REQUIRING_OS_ADD_DLL_DIRECTORY,
            requires_rtld_deepbind=name in LIBNAMES_REQUIRING_RTLD_DEEPBIND,
        )
    return registry


#: Canonical registry of all known libraries.
LIB_DESCRIPTORS: dict[str, LibDescriptor] = _build_registry()
