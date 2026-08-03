# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Legacy table exports derived from the authored descriptor catalog.

The canonical data entry point is :mod:`descriptor_catalog`. This module keeps
historical constant names for backward compatibility by deriving them from the
catalog.

The unsuffixed ``SUPPORTED_LIBNAMES_WINDOWS`` and
``SITE_PACKAGES_LIBDIRS_WINDOWS*`` constants retain their historical x64
meaning for compatibility, but are not recommended for new code. Use the
explicit ``*_X64`` or ``*_ARM64`` projection instead. Never combine the two
architecture projections.
"""

from __future__ import annotations

from cuda.pathfinder._dynamic_libs.descriptor_catalog import DESCRIPTOR_CATALOG
from cuda.pathfinder._utils.platform_aware import IS_WINDOWS, IS_WINDOWS_ARM64, IS_WINDOWS_X64

_CTK_DESCRIPTORS = tuple(desc for desc in DESCRIPTOR_CATALOG if desc.packaged_with == "ctk")
_OTHER_DESCRIPTORS = tuple(desc for desc in DESCRIPTOR_CATALOG if desc.packaged_with == "other")
_DRIVER_DESCRIPTORS = tuple(desc for desc in DESCRIPTOR_CATALOG if desc.packaged_with == "driver")
_NON_CTK_DESCRIPTORS = _OTHER_DESCRIPTORS + _DRIVER_DESCRIPTORS

SUPPORTED_LIBNAMES_COMMON = tuple(desc.name for desc in _CTK_DESCRIPTORS if desc.linux_sonames and desc.windows_dlls)
SUPPORTED_LIBNAMES_LINUX_ONLY = tuple(
    desc.name for desc in _CTK_DESCRIPTORS if desc.linux_sonames and not desc.windows_dlls
)
SUPPORTED_LIBNAMES_WINDOWS_ONLY = tuple(
    desc.name for desc in _CTK_DESCRIPTORS if desc.windows_dlls and not desc.linux_sonames
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
# Backward-compatible alias preserves the historical x64 meaning.
SUPPORTED_LIBNAMES_WINDOWS = SUPPORTED_LIBNAMES_WINDOWS_X64
SUPPORTED_LIBNAMES_ALL = SUPPORTED_LIBNAMES_COMMON + SUPPORTED_LIBNAMES_LINUX_ONLY + SUPPORTED_LIBNAMES_WINDOWS_ONLY
if not IS_WINDOWS:
    SUPPORTED_LIBNAMES = SUPPORTED_LIBNAMES_LINUX
elif IS_WINDOWS_X64:
    SUPPORTED_LIBNAMES = SUPPORTED_LIBNAMES_WINDOWS_X64
else:
    assert IS_WINDOWS_ARM64
    SUPPORTED_LIBNAMES = SUPPORTED_LIBNAMES_WINDOWS_ARM64

DIRECT_DEPENDENCIES_CTK = {desc.name: desc.dependencies for desc in _CTK_DESCRIPTORS if desc.dependencies}
DIRECT_DEPENDENCIES = {desc.name: desc.dependencies for desc in DESCRIPTOR_CATALOG if desc.dependencies}

SUPPORTED_LINUX_SONAMES_CTK = {desc.name: desc.linux_sonames for desc in _CTK_DESCRIPTORS if desc.linux_sonames}
SUPPORTED_LINUX_SONAMES_OTHER = {desc.name: desc.linux_sonames for desc in _OTHER_DESCRIPTORS if desc.linux_sonames}
SUPPORTED_LINUX_SONAMES_DRIVER = {desc.name: desc.linux_sonames for desc in _DRIVER_DESCRIPTORS if desc.linux_sonames}
SUPPORTED_LINUX_SONAMES = SUPPORTED_LINUX_SONAMES_CTK | SUPPORTED_LINUX_SONAMES_OTHER | SUPPORTED_LINUX_SONAMES_DRIVER

SUPPORTED_WINDOWS_DLLS_CTK = {desc.name: desc.windows_dlls for desc in _CTK_DESCRIPTORS if desc.windows_dlls}
SUPPORTED_WINDOWS_DLLS_OTHER = {desc.name: desc.windows_dlls for desc in _OTHER_DESCRIPTORS if desc.windows_dlls}
SUPPORTED_WINDOWS_DLLS_DRIVER = {desc.name: desc.windows_dlls for desc in _DRIVER_DESCRIPTORS if desc.windows_dlls}
SUPPORTED_WINDOWS_DLLS = SUPPORTED_WINDOWS_DLLS_CTK | SUPPORTED_WINDOWS_DLLS_OTHER | SUPPORTED_WINDOWS_DLLS_DRIVER

LIBNAMES_REQUIRING_OS_ADD_DLL_DIRECTORY = tuple(
    desc.name for desc in DESCRIPTOR_CATALOG if desc.requires_add_dll_directory and desc.windows_dlls
)
LIBNAMES_REQUIRING_RTLD_DEEPBIND = tuple(
    desc.name for desc in DESCRIPTOR_CATALOG if desc.requires_rtld_deepbind and desc.linux_sonames
)

SITE_PACKAGES_LIBDIRS_LINUX_CTK = {
    desc.name: desc.site_packages_linux for desc in _CTK_DESCRIPTORS if desc.site_packages_linux
}
SITE_PACKAGES_LIBDIRS_LINUX_OTHER = {
    desc.name: desc.site_packages_linux for desc in _NON_CTK_DESCRIPTORS if desc.site_packages_linux
}
SITE_PACKAGES_LIBDIRS_LINUX = SITE_PACKAGES_LIBDIRS_LINUX_CTK | SITE_PACKAGES_LIBDIRS_LINUX_OTHER

# Architecture-specific Windows projections. Keep these separate: combining
# them would make the table unsafe to consume for either process ABI.
SITE_PACKAGES_LIBDIRS_WINDOWS_CTK_X64 = {
    desc.name: desc.site_packages_windows.x64 for desc in _CTK_DESCRIPTORS if desc.site_packages_windows.x64
}
SITE_PACKAGES_LIBDIRS_WINDOWS_CTK_ARM64 = {
    desc.name: desc.site_packages_windows.arm64 for desc in _CTK_DESCRIPTORS if desc.site_packages_windows.arm64
}
SITE_PACKAGES_LIBDIRS_WINDOWS_OTHER_X64 = {
    desc.name: desc.site_packages_windows.x64 for desc in _NON_CTK_DESCRIPTORS if desc.site_packages_windows.x64
}
SITE_PACKAGES_LIBDIRS_WINDOWS_OTHER_ARM64 = {
    desc.name: desc.site_packages_windows.arm64 for desc in _NON_CTK_DESCRIPTORS if desc.site_packages_windows.arm64
}
SITE_PACKAGES_LIBDIRS_WINDOWS_X64 = SITE_PACKAGES_LIBDIRS_WINDOWS_CTK_X64 | SITE_PACKAGES_LIBDIRS_WINDOWS_OTHER_X64
SITE_PACKAGES_LIBDIRS_WINDOWS_ARM64 = (
    SITE_PACKAGES_LIBDIRS_WINDOWS_CTK_ARM64 | SITE_PACKAGES_LIBDIRS_WINDOWS_OTHER_ARM64
)

# Backward-compatible aliases preserve the historical x64 meaning.
SITE_PACKAGES_LIBDIRS_WINDOWS_CTK = SITE_PACKAGES_LIBDIRS_WINDOWS_CTK_X64
SITE_PACKAGES_LIBDIRS_WINDOWS_OTHER = SITE_PACKAGES_LIBDIRS_WINDOWS_OTHER_X64
SITE_PACKAGES_LIBDIRS_WINDOWS = SITE_PACKAGES_LIBDIRS_WINDOWS_X64


def is_suppressed_dll_file(path_basename: str) -> bool:
    if path_basename.startswith("nvrtc"):
        # nvidia_cuda_nvrtc_cu12-12.8.93-py3-none-win_amd64.whl:
        #     nvidia\cuda_nvrtc\bin\
        #         nvrtc-builtins64_128.dll
        #         nvrtc64_120_0.alt.dll
        #         nvrtc64_120_0.dll
        return path_basename.endswith(".alt.dll") or "-builtins" in path_basename
    return path_basename.startswith(("cudart32_", "nvvm32"))
