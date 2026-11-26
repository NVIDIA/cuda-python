# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Copy a Sphinx inventory and rebase its relative URLs for the destination."""

import argparse
import re
import zlib
from os.path import relpath
from pathlib import Path
from urllib.parse import urlsplit

# Match the Sphinx v2 fields around the location. Names and display text can
# contain spaces, and Sphinx expands a trailing $ in the location after loading.
_ENTRY = re.compile(r"^(.+?[ \t]+\S+[ \t]+-?\d+[ \t]+?)(\S*)([ \t]+.*)$", re.MULTILINE)


def copy_rebased_inventory(source: Path, destination: Path) -> None:
    parts = source.read_bytes().split(b"\n", 4)
    if len(parts) != 5 or parts[0] != b"# Sphinx inventory version 2":
        raise ValueError(f"{source}: expected a Sphinx v2 inventory")

    relative_dir = Path(relpath(source.parent, destination.parent)).as_posix()

    def rebase_entry(match: re.Match[str]) -> str:
        prefix, location, suffix = match.groups()
        if location.startswith("/") or urlsplit(location).scheme:
            return match[0]
        return f"{prefix}{relative_dir}/{location}{suffix}"

    header = b"\n".join(parts[:4]) + b"\n"
    entries = zlib.decompress(parts[4]).decode()
    rebased = _ENTRY.sub(rebase_entry, entries)
    destination.write_bytes(header + zlib.compress(rebased.encode()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    copy_rebased_inventory(args.source, args.destination)
