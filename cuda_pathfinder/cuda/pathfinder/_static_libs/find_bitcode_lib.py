# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import functools
import os
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn, TypedDict

from cuda.pathfinder._utils.env_vars import get_cuda_path_or_home
from cuda.pathfinder._utils.find_sub_dirs import find_sub_dirs_all_sitepackages
from cuda.pathfinder._utils.platform_aware import IS_WINDOWS


class BitcodeLibNotFoundError(RuntimeError):
    """Raised when a bitcode library cannot be found."""


@dataclass(frozen=True)
class LocatedBitcodeLib:
    """Information about a located bitcode library."""

    name: str
    abs_path: str
    filename: str
    found_via: str


class _BitcodeLibInfo(TypedDict):
    filename: str
    rel_path: str
    site_packages_dirs: tuple[str, ...]
    available_on_windows: bool


_SUPPORTED_BITCODE_LIBS_INFO: dict[str, _BitcodeLibInfo] = {
    "device": {
        "filename": "libdevice.10.bc",
        "rel_path": "nvvm/libdevice",
        "site_packages_dirs": (
            "nvidia/cu13/nvvm/libdevice",
            "nvidia/cuda_nvcc/nvvm/libdevice",
        ),
        "available_on_windows": True,
    },
    "nccl_device": {
        "filename": "libnccl_device.bc",
        "rel_path": "lib",
        "site_packages_dirs": ("nvidia/nccl/lib",),
        "available_on_windows": False,
    },
    "nvshmem_device": {
        "filename": "libnvshmem_device.bc",
        "rel_path": "lib",
        "site_packages_dirs": ("nvidia/nvshmem/lib",),
        "available_on_windows": False,
    },
}

# Public API: just the supported library names
SUPPORTED_BITCODE_LIBS: tuple[str, ...] = tuple(
    sorted(
        name for name, info in _SUPPORTED_BITCODE_LIBS_INFO.items() if not IS_WINDOWS or info["available_on_windows"]
    )
)


def _no_such_file_in_dir(directory: Path, filename: str, error_messages: list[str], attachments: list[str]) -> None:
    error_messages.append(f"No such file: {directory / filename}")
    if directory.is_dir():
        attachments.append(f'  listdir("{directory}"):')
        for node in sorted(node_path.name for node_path in directory.iterdir()):
            attachments.append(f"    {node}")
    else:
        attachments.append(f'  Directory does not exist: "{directory}"')


class _FindBitcodeLib:
    def __init__(self, name: str) -> None:
        if name not in _SUPPORTED_BITCODE_LIBS_INFO:  # Updated reference
            raise ValueError(f"Unknown bitcode library: '{name}'. Supported: {', '.join(SUPPORTED_BITCODE_LIBS)}")
        self.name: str = name
        self.config: _BitcodeLibInfo = _SUPPORTED_BITCODE_LIBS_INFO[name]  # Updated reference
        self.filename: str = self.config["filename"]
        self.rel_path: str = self.config["rel_path"]
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
        file_path = anchor / self.rel_path / self.filename
        if file_path.is_file():
            return file_path
        return None

    def try_with_cuda_home(self) -> Path | None:
        cuda_home = get_cuda_path_or_home()
        if cuda_home is None:
            self.error_messages.append("CUDA_HOME/CUDA_PATH not set")
            return None

        anchor = Path(cuda_home)
        file_path = anchor / self.rel_path / self.filename
        if file_path.is_file():
            return file_path

        _no_such_file_in_dir(
            anchor / self.rel_path,
            self.filename,
            self.error_messages,
            self.attachments,
        )
        return None

    def raise_not_found_error(self) -> NoReturn:
        err = ", ".join(self.error_messages) if self.error_messages else "No search paths available"
        att = "\n".join(self.attachments) if self.attachments else ""
        raise BitcodeLibNotFoundError(f'Failure finding "{self.filename}": {err}\n{att}')


def locate_bitcode_lib(name: str) -> LocatedBitcodeLib:
    """Locate a bitcode library by name.

    Raises:
        ValueError: If ``name`` is not a supported bitcode library.
        BitcodeLibNotFoundError: If the bitcode library cannot be found.
    """
    finder = _FindBitcodeLib(name)

    abs_path = finder.try_site_packages()
    if abs_path is not None:
        return LocatedBitcodeLib(
            name=name,
            abs_path=str(abs_path),
            filename=finder.filename,
            found_via="site-packages",
        )

    abs_path = finder.try_with_conda_prefix()
    if abs_path is not None:
        return LocatedBitcodeLib(
            name=name,
            abs_path=str(abs_path),
            filename=finder.filename,
            found_via="conda",
        )

    abs_path = finder.try_with_cuda_home()
    if abs_path is not None:
        return LocatedBitcodeLib(
            name=name,
            abs_path=str(abs_path),
            filename=finder.filename,
            found_via="CUDA_PATH",
        )

    finder.raise_not_found_error()


@functools.cache
def find_bitcode_lib(name: str) -> str:
    """Find the absolute path to a bitcode library.

    Raises:
        ValueError: If ``name`` is not a supported bitcode library.
        BitcodeLibNotFoundError: If the bitcode library cannot be found.
    """
    return locate_bitcode_lib(name).abs_path
