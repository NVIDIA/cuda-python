#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Scan directories for .so files, extract SONAMEs, update descriptor_catalog.py.

Usage:
    python toolshed/build_pathfinder_sonames.py /path/to/cuda [/more/paths ...]
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _catalog_writer import load_catalog, update_specs, write_catalog


def _extract_soname(path: str) -> str | None:
    try:
        out = subprocess.run(  # noqa: S603
            ["readelf", "-d", path],  # noqa: S607
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    for line in out.stdout.splitlines():
        if "SONAME" in line:
            # Format: 0x000000000000000e (SONAME)  Library soname: [libfoo.so.1]
            start = line.find("[")
            end = line.find("]")
            if start != -1 and end != -1:
                return line[start + 1 : end]
    return None


def _find_sonames(roots: list[str]) -> set[str]:
    sonames: set[str] = set()
    for root in roots:
        for dirpath, _dirnames, filenames in os.walk(root):
            for fname in filenames:
                if ".so" not in fname:
                    continue
                full = os.path.join(dirpath, fname)
                if os.path.islink(full):
                    continue
                soname = _extract_soname(full)
                if soname is not None:
                    sonames.add(soname)
    return sonames


def run(roots: list[str]) -> None:
    sonames_found = _find_sonames(roots)
    catalog = load_catalog()

    updates: dict[str, dict[str, object]] = {}
    matched: set[str] = set()
    for spec in catalog:
        if spec.strategy != "ctk":
            continue
        prefix = "lib" + spec.name + ".so"
        found = tuple(sorted(s for s in sonames_found if s.startswith(prefix)))
        if found:
            updates[spec.name] = {"linux_sonames": found}
            matched.update(found)

    if updates:
        write_catalog(update_specs(catalog, updates))
        for name, upd in sorted(updates.items()):
            print(f"  updated {name}: linux_sonames={upd['linux_sonames']}")
    else:
        print("No matching sonames found.")

    unmatched = sonames_found - matched
    if unmatched:
        print(f"\nSONAMEs not matched to any CTK descriptor ({len(unmatched)}):")
        for s in sorted(unmatched):
            print(f"  {s}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: build_pathfinder_sonames.py <dir> [<dir> ...]", file=sys.stderr)
        sys.exit(1)
    run(roots=sys.argv[1:])
