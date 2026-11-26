# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Make staged root Sphinx inventories point into their latest/ directories."""

import argparse
import re
import zlib
from pathlib import Path
from urllib.parse import urlsplit

# Match the Sphinx v2 fields around the location. Names and display text can
# contain spaces, and Sphinx expands a trailing $ in the location after loading.
_ENTRY = re.compile(r"^(.+?[ \t]+\S+[ \t]+-?\d+[ \t]+?)(\S*)([ \t]+.*)$", re.MULTILINE)


def _rebase_entry(match: re.Match[str]) -> str:
    prefix, location, suffix = match.groups()
    if location.startswith("/") or urlsplit(location).scheme:
        return match[0]
    return f"{prefix}latest/{location}{suffix}"


def rebase_root_inventories(docs_root: Path) -> None:
    latest_inventories = sorted(docs_root.glob("*/latest/objects.inv"))
    metapackage_inventory = docs_root / "latest" / "objects.inv"
    if metapackage_inventory.is_file():
        latest_inventories.insert(0, metapackage_inventory)
    if not latest_inventories:
        raise FileNotFoundError(f"No staged latest/objects.inv under {docs_root}")

    for inventory in latest_inventories:
        parts = inventory.read_bytes().split(b"\n", 4)
        if len(parts) != 5 or parts[0] != b"# Sphinx inventory version 2":
            raise ValueError(f"{inventory}: expected a Sphinx v2 inventory")

        header = b"\n".join(parts[:4]) + b"\n"
        entries = zlib.decompress(parts[4]).decode()
        rebased = _ENTRY.sub(_rebase_entry, entries)
        (inventory.parent.parent / "objects.inv").write_bytes(header + zlib.compress(rebased.encode()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("docs_root", type=Path)
    rebase_root_inventories(parser.parse_args().docs_root)
