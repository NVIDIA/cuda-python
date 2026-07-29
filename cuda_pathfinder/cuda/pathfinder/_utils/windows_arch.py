# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sysconfig

WINDOWS_PE_MACHINE_BY_ARCH = {
    "x64": 0x8664,
    "arm64": 0xAA64,
}


class UnsupportedArchError(RuntimeError):
    """Raised when Python reports an unsupported Windows architecture."""

    def __init__(self, platform_tag: str) -> None:
        self.platform_tag = platform_tag
        super().__init__(
            f"Unsupported Windows Python platform tag: {platform_tag!r}; expected 'win-amd64' or 'win-arm64'"
        )


def windows_python_arch() -> str:
    """Return the current Windows Python interpreter architecture."""
    raw_platform_tag = sysconfig.get_platform()
    platform_tag = raw_platform_tag.lower().replace("_", "-")

    if platform_tag == "win-arm64":
        return "arm64"

    if platform_tag == "win-amd64":
        return "x64"

    raise UnsupportedArchError(raw_platform_tag)


def windows_pe_matches_arch(path: str, target_arch: str) -> bool:
    """Return whether a valid PE file has the requested machine architecture."""
    expected_machine = WINDOWS_PE_MACHINE_BY_ARCH.get(target_arch)
    if expected_machine is None:
        raise ValueError(f"Unsupported Windows target architecture: {target_arch!r}")

    try:
        with open(path, "rb") as stream:
            if stream.read(2) != b"MZ":
                return False
            stream.seek(0x3C)
            pe_offset_bytes = stream.read(4)
            if len(pe_offset_bytes) != 4:
                return False
            stream.seek(int.from_bytes(pe_offset_bytes, "little"))
            if stream.read(4) != b"PE\0\0":
                return False
            machine_bytes = stream.read(2)
            if len(machine_bytes) != 2:
                return False
    except OSError:
        return False

    return int.from_bytes(machine_bytes, "little") == expected_machine
