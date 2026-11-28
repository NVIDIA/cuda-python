# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Copy a Sphinx inventory one level up and rebase its relative URLs."""

import argparse
import re
import zlib
from os.path import abspath
from pathlib import Path
from urllib.parse import urlsplit

# Match the Sphinx v2 fields around the location. Names and display text can
# contain spaces, and Sphinx expands a trailing $ in the location after loading.
_ENTRY = re.compile(r"^(.+?[ \t]+\S+[ \t]+-?\d+[ \t]+?)(\S*)([ \t]+.*)$", re.MULTILINE)


def copy_rebased_inventory_one_level_up(source: Path) -> None:
    source = Path(abspath(source))
    destination = source.parent.parent / source.name
    if destination == source:
        raise ValueError(f"{source}: cannot copy an inventory up from the filesystem root")
    parts = source.read_bytes().split(b"\n", 4)
    if len(parts) != 5 or parts[0] != b"# Sphinx inventory version 2":
        raise ValueError(f"{source}: expected a Sphinx v2 inventory")

    url_prefix = source.parent.name + "/"

    def rebase_entry(match: re.Match[str]) -> str:
        prefix, location, suffix = match.groups()
        if location.startswith("/") or urlsplit(location).scheme:
            return match[0]
        return f"{prefix}{url_prefix}{location}{suffix}"

    header = b"\n".join(parts[:4]) + b"\n"
    entries = zlib.decompress(parts[4]).decode()
    rebased = _ENTRY.sub(rebase_entry, entries)
    destination.write_bytes(header + zlib.compress(rebased.encode()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    copy_rebased_inventory_one_level_up(parser.parse_args().source)
