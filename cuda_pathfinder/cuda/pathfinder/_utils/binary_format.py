# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import functools
import sys
from dataclasses import dataclass
from typing import Literal

_COFF_MACHINE_TYPES = {
    0x014C,  # x86
    0x01C0,  # Arm
    0x01C4,  # Armv7
    0x8664,  # x64
    0xAA64,  # Arm64
}


@dataclass(frozen=True, slots=True)
class BinaryFormat:
    """The object format and machine type of a native binary."""

    kind: Literal["coff", "elf"]
    machine: int


def _binary_format_from_object(data: bytes) -> BinaryFormat | None:
    if data.startswith(b"\x7fELF") and len(data) >= 20:
        if data[5] == 1:
            byte_order: Literal["little", "big"] = "little"
        elif data[5] == 2:
            byte_order = "big"
        else:
            return None
        return BinaryFormat("elf", int.from_bytes(data[18:20], byte_order))

    # A regular COFF object starts with Machine. Import objects and bigobj
    # objects use the anonymous-object signature and store Machine at offset 6.
    if data.startswith(b"\x00\x00\xff\xff") and len(data) >= 8:
        machine = int.from_bytes(data[6:8], "little")
        if machine in _COFF_MACHINE_TYPES:
            return BinaryFormat("coff", machine)
    if len(data) >= 20:
        machine = int.from_bytes(data[:2], "little")
        if machine in _COFF_MACHINE_TYPES:
            return BinaryFormat("coff", machine)
    return None


@functools.cache
def python_binary_format() -> BinaryFormat:
    """Read the running Python executable's native object format and machine."""
    try:
        with open(sys.executable, "rb") as stream:
            prefix = stream.read(64)
            binary_format = _binary_format_from_object(prefix)
            if binary_format is not None and binary_format.kind == "elf":
                return binary_format

            if prefix.startswith(b"MZ") and len(prefix) >= 64:
                pe_offset = int.from_bytes(prefix[0x3C:0x40], "little")
                stream.seek(pe_offset)
                if stream.read(4) == b"PE\0\0":
                    machine_bytes = stream.read(2)
                    if len(machine_bytes) == 2:
                        return BinaryFormat("coff", int.from_bytes(machine_bytes, "little"))
    except OSError as exc:
        raise RuntimeError(f"Could not inspect Python executable {sys.executable!r}: {exc}") from exc

    raise RuntimeError(f"Unsupported Python executable binary format: {sys.executable!r}")


def static_archive_matches_binary_format(path: str, expected: BinaryFormat) -> bool:
    """Return whether an ar archive contains an object matching ``expected``."""
    try:
        with open(path, "rb") as stream:
            if stream.read(8) != b"!<arch>\n":
                return False

            while True:
                header = stream.read(60)
                if not header:
                    return False
                if len(header) != 60 or header[58:60] != b"`\n":
                    return False

                member_name = header[:16].decode("ascii", "replace").rstrip()
                member_size = int(header[48:58].decode("ascii").strip())
                member_start = stream.tell()
                object_start = member_start
                object_size = member_size

                if member_name.startswith("#1/"):
                    name_size = int(member_name[3:])
                    if name_size > member_size:
                        return False
                    object_start += name_size
                    object_size -= name_size

                is_index = member_name in ("/", "//", "/SYM64/") or member_name.startswith("__.SYMDEF")
                if not is_index:
                    stream.seek(object_start)
                    actual = _binary_format_from_object(stream.read(min(object_size, 64)))
                    if actual is not None:
                        return actual == expected

                stream.seek(member_start + member_size + member_size % 2)
    except (OSError, ValueError):
        return False
