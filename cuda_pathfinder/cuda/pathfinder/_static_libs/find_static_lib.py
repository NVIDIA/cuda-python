# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import functools
import os
from dataclasses import dataclass
from typing import NoReturn, TypedDict

from cuda.pathfinder._utils.env_vars import get_cuda_home_or_path
from cuda.pathfinder._utils.find_sub_dirs import find_sub_dirs_all_sitepackages
from cuda.pathfinder._utils.platform_aware import IS_WINDOWS


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
    conda_rel_path: str
    site_packages_dirs: tuple[str, ...]


_SUPPORTED_STATIC_LIBS_INFO: dict[str, _StaticLibInfo] = {
    "cudadevrt": {
        "filename": "cudadevrt.lib" if IS_WINDOWS else "libcudadevrt.a",
        "ctk_rel_paths": (os.path.join("lib", "x64"),) if IS_WINDOWS else ("lib64", "lib"),
        "conda_rel_path": os.path.join("lib", "x64") if IS_WINDOWS else "lib",
        "site_packages_dirs": (
            ("nvidia/cu13/lib/x64", "nvidia/cuda_runtime/lib/x64")
            if IS_WINDOWS
            else ("nvidia/cu13/lib", "nvidia/cuda_runtime/lib")
        ),
    },
}

SUPPORTED_STATIC_LIBS: tuple[str, ...] = tuple(sorted(_SUPPORTED_STATIC_LIBS_INFO.keys()))


def _no_such_file_in_dir(dir_path: str, filename: str, error_messages: list[str], attachments: list[str]) -> None:
    error_messages.append(f"No such file: {os.path.join(dir_path, filename)}")
    if os.path.isdir(dir_path):
        attachments.append(f'  listdir("{dir_path}"):')
        for node in sorted(os.listdir(dir_path)):
            attachments.append(f"    {node}")
    else:
        attachments.append(f'  Directory does not exist: "{dir_path}"')


class _FindStaticLib:
    def __init__(self, name: str) -> None:
        if name not in _SUPPORTED_STATIC_LIBS_INFO:
            raise ValueError(f"Unknown static library: '{name}'. Supported: {', '.join(SUPPORTED_STATIC_LIBS)}")
        self.name: str = name
        self.config: _StaticLibInfo = _SUPPORTED_STATIC_LIBS_INFO[name]
        self.filename: str = self.config["filename"]
        self.ctk_rel_paths: tuple[str, ...] = self.config["ctk_rel_paths"]
        self.conda_rel_path: str = self.config["conda_rel_path"]
        self.site_packages_dirs: tuple[str, ...] = self.config["site_packages_dirs"]
        self.error_messages: list[str] = []
        self.attachments: list[str] = []

    def try_site_packages(self) -> str | None:
        for rel_dir in self.site_packages_dirs:
            sub_dir = tuple(rel_dir.split("/"))
            for abs_dir in find_sub_dirs_all_sitepackages(sub_dir):
                file_path = os.path.join(abs_dir, self.filename)
                if os.path.isfile(file_path):
                    return file_path
        return None

    def try_with_conda_prefix(self) -> str | None:
        conda_prefix = os.environ.get("CONDA_PREFIX")
        if not conda_prefix:
            return None

        anchor = os.path.join(conda_prefix, "Library") if IS_WINDOWS else conda_prefix
        file_path = os.path.join(anchor, self.conda_rel_path, self.filename)
        if os.path.isfile(file_path):
            return file_path
        return None

    def try_with_cuda_home(self) -> str | None:
        cuda_home = get_cuda_home_or_path()
        if cuda_home is None:
            self.error_messages.append("CUDA_HOME/CUDA_PATH not set")
            return None

        for rel_path in self.ctk_rel_paths:
            file_path = os.path.join(cuda_home, rel_path, self.filename)
            if os.path.isfile(file_path):
                return file_path

        _no_such_file_in_dir(
            os.path.join(cuda_home, self.ctk_rel_paths[0]),
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
            abs_path=abs_path,
            filename=finder.filename,
            found_via="site-packages",
        )

    abs_path = finder.try_with_conda_prefix()
    if abs_path is not None:
        return LocatedStaticLib(
            name=name,
            abs_path=abs_path,
            filename=finder.filename,
            found_via="conda",
        )

    abs_path = finder.try_with_cuda_home()
    if abs_path is not None:
        return LocatedStaticLib(
            name=name,
            abs_path=abs_path,
            filename=finder.filename,
            found_via="CUDA_HOME",
        )

    finder.raise_not_found_error()


@functools.cache
def find_static_lib(name: str) -> str:
    """Find the absolute path to a static library.

    Raises:
        ValueError: If ``name`` is not a supported static library.
        StaticLibNotFoundError: If the static library cannot be found.
    """
    return locate_static_lib(name).abs_path
