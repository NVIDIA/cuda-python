# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import functools
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

from cuda.pathfinder._static_libs.locate_file import locate_file, not_found_message
from cuda.pathfinder._utils.platform_aware import IS_WINDOWS
from cuda.pathfinder._utils.windows_arch import windows_python_arch


class StaticLibNotFoundError(RuntimeError):
    """Raised when a static library cannot be found."""


@dataclass(frozen=True)
class LocatedStaticLib:
    """Information about a located static library."""

    name: str
    abs_path: str
    filename: str
    found_via: str


class _StaticLibInfo(TypedDict):
    """Static-library metadata with ordered alternatives searched first-to-last."""

    filename: str
    ctk_rel_paths: tuple[str, ...]
    conda_rel_paths: tuple[str, ...]
    site_packages_dirs: tuple[str, ...]


def _cudadevrt_info() -> _StaticLibInfo:
    if not IS_WINDOWS:
        return {
            "filename": "libcudadevrt.a",
            "ctk_rel_paths": ("lib64", "lib"),
            "conda_rel_paths": ("lib",),
            "site_packages_dirs": ("nvidia/cu13/lib", "nvidia/cuda_runtime/lib"),
        }

    arch_dir = windows_python_arch()
    component_wheel_dirs = ("nvidia/cuda_runtime/lib/x64",) if arch_dir == "x64" else ()
    conda_fallback_dirs = ("lib",) if arch_dir == "x64" else ()
    return {
        "filename": "cudadevrt.lib",
        "ctk_rel_paths": (str(Path("lib", arch_dir)),),
        "conda_rel_paths": (str(Path("lib", arch_dir)), *conda_fallback_dirs),
        "site_packages_dirs": (f"nvidia/cu13/lib/{arch_dir}", *component_wheel_dirs),
    }


_SUPPORTED_STATIC_LIBS_INFO: dict[str, _StaticLibInfo] = {
    "cudadevrt": _cudadevrt_info(),
}

SUPPORTED_STATIC_LIBS: tuple[str, ...] = tuple(sorted(_SUPPORTED_STATIC_LIBS_INFO.keys()))


def locate_static_lib(name: str) -> LocatedStaticLib:
    """Locate a static library by name, returning its resolved absolute path.

    Raises:
        ValueError: If ``name`` is not a supported static library.
        StaticLibNotFoundError: If the static library cannot be found.
    """
    if name not in _SUPPORTED_STATIC_LIBS_INFO:
        raise ValueError(f"Unknown static library: '{name}'. Supported: {', '.join(SUPPORTED_STATIC_LIBS)}")
    info = _SUPPORTED_STATIC_LIBS_INFO[name]
    filename = info["filename"]

    found = locate_file(
        filename,
        site_packages_dirs=info["site_packages_dirs"],
        conda_rel_paths=info["conda_rel_paths"],
        ctk_rel_paths=info["ctk_rel_paths"],
    )
    if found is None:
        raise StaticLibNotFoundError(not_found_message(filename, info["ctk_rel_paths"]))
    abs_path, found_via = found
    return LocatedStaticLib(name=name, abs_path=abs_path, filename=filename, found_via=found_via)


@functools.cache
def find_static_lib(name: str) -> str:
    """Find the resolved absolute path to a static library.

    Raises:
        ValueError: If ``name`` is not a supported static library.
        StaticLibNotFoundError: If the static library cannot be found.

    Windows on ARM (WoA) Note:
        On Windows, this API aims to return the path to a static library whose
        architecture matches the Python interpreter architecture. For example,
        x64 Python running on an Arm64 machine targets the x64 library, while
        native Arm64 Python targets the Arm64 library. This differs from
        ``find_nvidia_binary_utility``, which targets the native machine
        architecture when selecting architecture-specific executables.
    """
    return locate_static_lib(name).abs_path
