# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import functools
from dataclasses import dataclass
from typing import TypedDict

from cuda.pathfinder._static_libs.locate_file import locate_file
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
    """Bitcode-library metadata with ordered alternatives searched first-to-last."""

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


def locate_bitcode_lib(name: str) -> LocatedBitcodeLib:
    """Locate a bitcode library by name, returning its resolved absolute path.

    Raises:
        ValueError: If ``name`` is not a supported bitcode library.
        BitcodeLibNotFoundError: If the bitcode library cannot be found.
    """
    if name not in _SUPPORTED_BITCODE_LIBS_INFO:
        raise ValueError(f"Unknown bitcode library: '{name}'. Supported: {', '.join(SUPPORTED_BITCODE_LIBS)}")
    info = _SUPPORTED_BITCODE_LIBS_INFO[name]
    filename = info["filename"]
    rel_paths = (info["rel_path"],)

    abs_path, found_via = locate_file(
        filename,
        site_packages_dirs=info["site_packages_dirs"],
        conda_rel_paths=rel_paths,
        ctk_rel_paths=rel_paths,
        not_found_error=BitcodeLibNotFoundError,
    )
    return LocatedBitcodeLib(name=name, abs_path=abs_path, filename=filename, found_via=found_via)


@functools.cache
def find_bitcode_lib(name: str) -> str:
    """Find the resolved absolute path to a bitcode library.

    Raises:
        ValueError: If ``name`` is not a supported bitcode library.
        BitcodeLibNotFoundError: If the bitcode library cannot be found.
    """
    return locate_bitcode_lib(name).abs_path
