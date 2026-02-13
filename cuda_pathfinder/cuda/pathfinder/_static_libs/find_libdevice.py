# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import functools
import glob
import os
from dataclasses import dataclass

from cuda.pathfinder._dynamic_libs.load_dl_common import DynamicLibNotFoundError
from cuda.pathfinder._dynamic_libs.supported_nvidia_libs import IS_WINDOWS
from cuda.pathfinder._utils.env_vars import get_cuda_home_or_path
from cuda.pathfinder._utils.find_sub_dirs import find_sub_dirs_all_sitepackages


class BitcodeLibNotFoundError(DynamicLibNotFoundError):
    """Raised when a bitcode library cannot be found."""

    pass


@dataclass(frozen=True)
class LocatedBitcodeLib:
    """Information about a located bitcode library."""

    name: str
    abs_path: str
    filename: str


SUPPORTED_BITCODE_LIBS = {
    "device": {
        "filename": "libdevice.10.bc",
        "rel_path": os.path.join("nvvm", "libdevice"),
        "site_packages_dirs": (
            "nvidia/cu13/nvvm/libdevice",  # CTK 13+
            "nvidia/cuda_nvcc/nvvm/libdevice",  # CTK <13
        ),
    },
}

if IS_WINDOWS:
    _COMMON_BASES = [r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA", r"C:\CUDA"]
else:
    _COMMON_BASES = ["/usr/local/cuda", "/opt/cuda"]


def _no_such_file_in_dir(dir_path: str, filename: str, error_messages: list[str], attachments: list[str]) -> None:
    error_messages.append(f"No such file: {os.path.join(dir_path, filename)}")
    if os.path.isdir(dir_path):
        attachments.append(f'  listdir("{dir_path}"):')
        for node in sorted(os.listdir(dir_path)):
            attachments.append(f"    {node}")
    else:
        attachments.append(f'  Directory does not exist: "{dir_path}"')


class _FindBitcodeLib:
    def __init__(self, name: str) -> None:
        if name not in SUPPORTED_BITCODE_LIBS:
            raise ValueError(
                f"Unknown bitcode library: '{name}'. Supported: {', '.join(sorted(SUPPORTED_BITCODE_LIBS.keys()))}"
            )
        self.name = name
        self.config = SUPPORTED_BITCODE_LIBS[name]
        self.filename = self.config["filename"]
        self.rel_path = self.config["rel_path"]
        self.site_packages_dirs = self.config["site_packages_dirs"]
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
        file_path = os.path.join(anchor, self.rel_path, self.filename)
        if os.path.isfile(file_path):
            return file_path
        return None

    def try_with_cuda_home(self) -> str | None:
        cuda_home = get_cuda_home_or_path()
        if cuda_home is None:
            self.error_messages.append("CUDA_HOME/CUDA_PATH not set")
            return None

        file_path = os.path.join(cuda_home, self.rel_path, self.filename)
        if os.path.isfile(file_path):
            return file_path

        _no_such_file_in_dir(
            os.path.join(cuda_home, self.rel_path),
            self.filename,
            self.error_messages,
            self.attachments,
        )
        return None

    def try_common_paths(self) -> str | None:
        for base in _COMMON_BASES:
            file_path = os.path.join(base, self.rel_path, self.filename)
            if os.path.isfile(file_path):
                return file_path
            for versioned in sorted(glob.glob(base + "*"), reverse=True):
                if os.path.isdir(versioned):
                    file_path = os.path.join(versioned, self.rel_path, self.filename)
                    if os.path.isfile(file_path):
                        return file_path
        return None

    def raise_not_found_error(self) -> None:
        err = ", ".join(self.error_messages) if self.error_messages else "No search paths available"
        att = "\n".join(self.attachments) if self.attachments else ""
        raise BitcodeLibNotFoundError(f'Failure finding "{self.filename}": {err}\n{att}')


def locate_bitcode_lib(name: str) -> LocatedBitcodeLib | None:
    """Locate a bitcode library by name."""
    finder = _FindBitcodeLib(name)

    abs_path = finder.try_site_packages()
    if abs_path is None:
        abs_path = finder.try_with_conda_prefix()
    if abs_path is None:
        abs_path = finder.try_with_cuda_home()
    if abs_path is None:
        abs_path = finder.try_common_paths()

    if abs_path is None:
        return None

    return LocatedBitcodeLib(
        name=name,
        abs_path=abs_path,
        filename=finder.filename,
    )


@functools.cache
def find_bitcode_lib(name: str) -> str:
    """Find the absolute path to a bitcode library."""
    result = locate_bitcode_lib(name)
    if result is None:
        config = SUPPORTED_BITCODE_LIBS.get(name, {})
        filename = config.get("filename", name)
        raise BitcodeLibNotFoundError(f"{filename} not found")
    return result.abs_path
