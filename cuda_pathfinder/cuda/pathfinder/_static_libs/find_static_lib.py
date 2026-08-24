# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import functools
import os
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn, TypedDict

from cuda.pathfinder._utils.env_vars import get_cuda_path_or_home
from cuda.pathfinder._utils.find_sub_dirs import find_sub_dirs_all_sitepackages
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


def _no_such_file_in_dir(directory: Path, filename: str, error_messages: list[str], attachments: list[str]) -> None:
    error_messages.append(f"No such file: {directory / filename}")
    if directory.is_dir():
        attachments.append(f'  listdir("{directory}"):')
        for node in sorted(node_path.name for node_path in directory.iterdir()):
            attachments.append(f"    {node}")
    else:
        attachments.append(f'  Directory does not exist: "{directory}"')


class _FindStaticLib:
    def __init__(self, name: str) -> None:
        if name not in _SUPPORTED_STATIC_LIBS_INFO:
            raise ValueError(f"Unknown static library: '{name}'. Supported: {', '.join(SUPPORTED_STATIC_LIBS)}")
        self.name: str = name
        self.config: _StaticLibInfo = _SUPPORTED_STATIC_LIBS_INFO[name]
        self.filename: str = self.config["filename"]
        self.ctk_rel_paths: tuple[str, ...] = self.config["ctk_rel_paths"]
        self.conda_rel_paths: tuple[str, ...] = self.config["conda_rel_paths"]
        self.site_packages_dirs: tuple[str, ...] = self.config["site_packages_dirs"]
        self.error_messages: list[str] = []
        self.attachments: list[str] = []

    def try_site_packages(self) -> Path | None:
        for rel_dir in self.site_packages_dirs:
            sub_dir = tuple(rel_dir.split("/"))
            for abs_dir in find_sub_dirs_all_sitepackages(sub_dir):
                file_path = Path(abs_dir, self.filename)
                if file_path.is_file():
                    return file_path
        return None

    def try_with_conda_prefix(self) -> Path | None:
        conda_prefix = os.environ.get("CONDA_PREFIX")
        if not conda_prefix:
            return None

        anchor = Path(conda_prefix, "Library") if IS_WINDOWS else Path(conda_prefix)
        for rel_path in self.conda_rel_paths:
            file_path = anchor / rel_path / self.filename
            if file_path.is_file():
                return file_path
        return None

    def try_with_cuda_home(self) -> Path | None:
        cuda_home = get_cuda_path_or_home()
        if cuda_home is None:
            self.error_messages.append("CUDA_HOME/CUDA_PATH not set")
            return None

        anchor = Path(cuda_home)
        for rel_path in self.ctk_rel_paths:
            file_path = anchor / rel_path / self.filename
            if file_path.is_file():
                return file_path

        _no_such_file_in_dir(
            anchor / self.ctk_rel_paths[0],
            self.filename,
            self.error_messages,
            self.attachments,
        )
        return None

    def raise_not_found_error(self) -> NoReturn:
        err = ", ".join(self.error_messages) if self.error_messages else "No search paths available"
        att = "\n".join(self.attachments) if self.attachments else ""
        raise StaticLibNotFoundError(f'Failure finding "{self.filename}": {err}\n{att}')


def locate_static_lib(name: str) -> LocatedStaticLib:
    """Locate a static library by name.

    Raises:
        ValueError: If ``name`` is not a supported static library.
        StaticLibNotFoundError: If the static library cannot be found.
    """
    finder = _FindStaticLib(name)

    abs_path = finder.try_site_packages()
    if abs_path is not None:
        return LocatedStaticLib(
            name=name,
            abs_path=str(abs_path),
            filename=finder.filename,
            found_via="site-packages",
        )

    abs_path = finder.try_with_conda_prefix()
    if abs_path is not None:
        return LocatedStaticLib(
            name=name,
            abs_path=str(abs_path),
            filename=finder.filename,
            found_via="conda",
        )

    abs_path = finder.try_with_cuda_home()
    if abs_path is not None:
        return LocatedStaticLib(
            name=name,
            abs_path=str(abs_path),
            filename=finder.filename,
            found_via="CUDA_PATH",
        )

    finder.raise_not_found_error()


@functools.cache
def find_static_lib(name: str) -> str:
    """Find the absolute path to a static library.

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
