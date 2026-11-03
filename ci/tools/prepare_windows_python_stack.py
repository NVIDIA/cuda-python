#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Prepare a copy of a Windows Python executable with at least an 8 MiB stack.

Windows prevents writing to a running executable. Write a patched copy, then
have the calling shell replace the original after this interpreter exits.
Patch sys._base_executable rather than the virtual environment's launcher.
"""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

STACK_RESERVE = 8 * 1024 * 1024


def prepare_executable(source: Path, destination: Path) -> tuple[int, int]:
    """Validate a PE image and write a new copy with a larger stack reserve."""
    image = bytearray(source.read_bytes())
    if len(image) < 64 or image[:2] != b"MZ":
        raise ValueError("Not a Windows executable: missing DOS header")
    pe_offset = struct.unpack_from("<I", image, 60)[0]
    if pe_offset < 64 or pe_offset + 24 > len(image) or image[pe_offset : pe_offset + 4] != b"PE\0\0":
        raise ValueError("Missing or truncated PE header")

    optional_offset = pe_offset + 24
    optional_size = struct.unpack_from("<H", image, pe_offset + 20)[0]
    if optional_size < 2 or optional_offset + optional_size > len(image):
        raise ValueError("Truncated PE optional header")
    magic = struct.unpack_from("<H", image, optional_offset)[0]
    if magic == 0x10B:  # PE32
        stack_format, minimum_size = "<I", 96
    elif magic == 0x20B:  # PE32+
        stack_format, minimum_size = "<Q", 112
    else:
        raise ValueError(f"Unsupported PE optional header magic: {magic:#x}")
    if optional_size < minimum_size:
        raise ValueError("Truncated PE optional header")

    # SizeOfStackReserve is at offset 72 in both optional-header formats:
    # https://learn.microsoft.com/en-us/windows/win32/debug/pe-format
    stack_offset = optional_offset + 72
    previous = struct.unpack_from(stack_format, image, stack_offset)[0]
    reserve = max(previous, STACK_RESERVE)
    struct.pack_into(stack_format, image, stack_offset, reserve)
    # Exclusive creation also prevents accidentally overwriting the running
    # source executable, or reusing a stale patched copy from an earlier run.
    with destination.open("xb") as output:
        output.write(image)
    return previous, reserve


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Base Python executable (sys._base_executable)")
    parser.add_argument("destination", type=Path, help="New patched copy; must not already exist")
    args = parser.parse_args(argv)
    previous, reserve = prepare_executable(args.source, args.destination)
    print(f"{args.source}: stack reserve {previous} -> {reserve} bytes; prepared {args.destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
