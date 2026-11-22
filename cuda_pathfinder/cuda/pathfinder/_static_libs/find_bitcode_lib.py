# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import functools
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn, TypedDict

from cuda.pathfinder._utils.env_vars import get_cuda_path_or_home
from cuda.pathfinder._utils.find_sub_dirs import find_sub_dirs_all_sitepackages
from cuda.pathfinder._utils.platform_aware import IS_WINDOWS

from .._utils.resolve_path import resolve_path


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
    """Bitcode-library metadata with ordered alternatives searched first-to-last.

    File paths are relative to the library directory. ``rel_path`` selects
    that directory for Conda and CUDA installations, and
    ``site_packages_dirs`` does so for site-packages.

    ``{sm_arch}`` expands to the caller's value, such as ``90`` or ``100a``.
    """

    # Keys: None for the default file, "*" for unlisted SMs, or an SM like "100a".
    # Values: relative paths. Entries used with an SM may contain {sm_arch}.
    filenames: dict[str | None, str]
    rel_path: str
    site_packages_dirs: tuple[str, ...]
    available_on_windows: bool


_SUPPORTED_BITCODE_LIBS_INFO: dict[str, _BitcodeLibInfo] = {
    "device": {
        "filenames": {None: "libdevice.10.bc"},
        "rel_path": "nvvm/libdevice",
        "site_packages_dirs": (
            "nvidia/cu13/nvvm/libdevice",
            "nvidia/cuda_nvcc/nvvm/libdevice",
        ),
        "available_on_windows": True,
    },
    "nccl_device": {
        "filenames": {
            None: "libnccl_device.bc",
            "*": "libnccl_device_sm_{sm_arch}.bc",
        },
        "rel_path": "lib",
        "site_packages_dirs": ("nvidia/nccl/lib",),
        "available_on_windows": False,
    },
    "nvshmem_device": {
        "filenames": {
            None: "libnvshmem_device.bc",
            "*": "libnvshmem_device_sm_{sm_arch}.bc",
        },
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
    def __init__(self, name: str, sm_arch: str | None) -> None:
        if name not in _SUPPORTED_BITCODE_LIBS_INFO:  # Updated reference
            raise ValueError(f"Unknown bitcode library: '{name}'. Supported: {', '.join(SUPPORTED_BITCODE_LIBS)}")
        self.name: str = name
        self.sm_arch: str | None = sm_arch
        self.config: _BitcodeLibInfo = _SUPPORTED_BITCODE_LIBS_INFO[name]  # Updated reference
        filenames = self.config["filenames"]
        if sm_arch is None:
            filename = filenames.get(None)
            if filename is None:
                raise BitcodeLibNotFoundError(f"Bitcode library {name!r} has no default file configured")
        else:
            if not isinstance(sm_arch, str) or re.fullmatch(r"[0-9]+[a-z]?", sm_arch) is None:
                raise ValueError("sm_arch must be a string such as '90' or '100a' without the 'sm_' prefix, or None")
            template = filenames.get(sm_arch, filenames.get("*"))
            if template is None:
                raise BitcodeLibNotFoundError(
                    f"Bitcode library {name!r} has no file configured for sm_arch={sm_arch!r}"
                )
            filename = template.format(sm_arch=sm_arch)

        self.filename: str = filename
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
                    return resolve_path(file_path)
        return None

    def try_with_conda_prefix(self) -> Path | None:
        conda_prefix = os.environ.get("CONDA_PREFIX")
        if not conda_prefix:
            return None

        anchor = Path(conda_prefix, "Library") if IS_WINDOWS else Path(conda_prefix)
        file_path = anchor / self.rel_path / self.filename
        if file_path.is_file():
            return resolve_path(file_path)
        return None

    def try_with_cuda_home(self) -> Path | None:
        cuda_home = get_cuda_path_or_home()
        if cuda_home is None:
            self.error_messages.append("CUDA_HOME/CUDA_PATH not set")
            return None

        anchor = Path(cuda_home)
        file_path = anchor / self.rel_path / self.filename
        if file_path.is_file():
            return resolve_path(file_path)

        _no_such_file_in_dir(
            file_path.parent,
            file_path.name,
            self.error_messages,
            self.attachments,
        )
        return None

    def raise_not_found_error(self) -> NoReturn:
        err = ", ".join(self.error_messages) if self.error_messages else "No search paths available"
        att = "\n".join(self.attachments) if self.attachments else ""
        target = f'"{self.filename}"'
        if self.sm_arch is not None:
            target += f" for {self.name!r} with sm_arch={self.sm_arch!r}"
        raise BitcodeLibNotFoundError(f"Failure finding {target}: {err}\n{att}")


def locate_bitcode_lib(name: str, *, sm_arch: str | None = None) -> LocatedBitcodeLib:
    """Locate a bitcode library by name, returning its resolved absolute path.

    Args:
        name: Supported bitcode library name.
        sm_arch: Target SM without the ``sm_`` prefix, such as ``"90"`` or
            ``"100a"``. If specified, search only for the file selected for
            that SM by the library's configuration. The matching file must
            be present in the installation.
            Pass ``None`` or omit this argument to search for the default file.

    Raises:
        ValueError: If ``name`` is not a supported bitcode library.
            Also raised if ``sm_arch`` is invalid.
        BitcodeLibNotFoundError: If the bitcode library cannot be found.
            Also raised if no file is configured for the request.
    """
    finder = _FindBitcodeLib(name, sm_arch)
    filename = Path(finder.filename).name

    abs_path = finder.try_site_packages()
    if abs_path is not None:
        return LocatedBitcodeLib(
            name=name,
            abs_path=str(abs_path),
            filename=filename,
            found_via="site-packages",
        )

    abs_path = finder.try_with_conda_prefix()
    if abs_path is not None:
        return LocatedBitcodeLib(
            name=name,
            abs_path=str(abs_path),
            filename=filename,
            found_via="conda",
        )

    abs_path = finder.try_with_cuda_home()
    if abs_path is not None:
        return LocatedBitcodeLib(
            name=name,
            abs_path=str(abs_path),
            filename=filename,
            found_via="CUDA_PATH",
        )

    finder.raise_not_found_error()


@functools.cache
def find_bitcode_lib(name: str, *, sm_arch: str | None = None) -> str:
    """Find the resolved absolute path to a bitcode library.

    Args:
        name: Supported bitcode library name.
        sm_arch: Target SM without the ``sm_`` prefix, such as ``"90"``. See
            :func:`locate_bitcode_lib` for file selection.

    Raises:
        ValueError: If ``name`` is not a supported bitcode library.
            Also raised if the ``sm_arch`` string has an invalid format.
        BitcodeLibNotFoundError: If the bitcode library cannot be found.
            Also raised if no file is configured for the request.

    Example:
        Find ``libnvshmem_device_sm_100.bc`` if installed.

        >>> from cuda.pathfinder import find_bitcode_lib
        >>> path = find_bitcode_lib("nvshmem_device", sm_arch="100")
    """
    return locate_bitcode_lib(name, sm_arch=sm_arch).abs_path
