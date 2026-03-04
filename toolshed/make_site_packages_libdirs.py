#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Parse collected site-packages library paths, update descriptor_catalog.py.

Usage:
    python toolshed/make_site_packages_libdirs.py linux collected_linux.txt
    python toolshed/make_site_packages_libdirs.py windows collected_windows.txt
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Dict, Set

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _catalog_writer import load_catalog, update_specs, write_catalog

_SITE_PACKAGES_RE = re.compile(r"(?i)^.*?/site-packages/")


def _strip_site_packages_prefix(p: str) -> str:
    """Remove any leading '.../site-packages/' (handles '\\' or '/', case-insensitive)."""
    p = p.replace("\\", "/")
    return _SITE_PACKAGES_RE.sub("", p)


def _parse_lines_linux(lines: list[str]) -> Dict[str, Set[str]]:
    d: Dict[str, Set[str]] = {}
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        line = _strip_site_packages_prefix(line)
        dirpath, fname = os.path.split(line)
        # Require something like libNAME.so, libNAME.so.12, libNAME.so.12.1, etc.
        i = fname.find(".so")
        if not fname.startswith("lib") or i == -1:
            continue
        name = fname[3:i]  # e.g. "libnvrtc" -> "nvrtc"
        d.setdefault(name, set()).add(dirpath)
    return d


def _extract_libname_from_dll(fname: str) -> str | None:
    """Return base libname per the heuristic, or None if not a .dll."""
    base = os.path.basename(fname)
    if not base.lower().endswith(".dll"):
        return None
    stem = base[:-4]  # drop ".dll"
    out = []
    for ch in stem:
        if ch == "_" or ch.isdigit():
            break
        out.append(ch)
    name = "".join(out)
    return name or None


def _parse_lines_windows(lines: list[str]) -> Dict[str, Set[str]]:
    """Collect {libname: set(dirnames)} with deduped directories."""
    m: Dict[str, Set[str]] = {}
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        line = _strip_site_packages_prefix(line)
        dirpath, fname = os.path.split(line)
        libname = _extract_libname_from_dll(fname)
        if not libname:
            continue
        m.setdefault(libname, set()).add(dirpath)
    return m


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Update site_packages_* in descriptor_catalog.py from collected library paths"
    )
    ap.add_argument("platform", choices=["linux", "windows"])
    ap.add_argument("path", help="Text file with one library path per line")
    args = ap.parse_args()

    with open(args.path, encoding="utf-8") as f:
        lines = f.read().splitlines()

    if args.platform == "linux":
        parsed = _parse_lines_linux(lines)
        field = "site_packages_linux"
    else:
        parsed = _parse_lines_windows(lines)
        field = "site_packages_windows"

    catalog = load_catalog()
    catalog_names = {spec.name for spec in catalog}

    updates: dict[str, dict[str, object]] = {}
    for name, dirs in parsed.items():
        if name in catalog_names:
            updates[name] = {field: tuple(sorted(dirs))}

    if updates:
        write_catalog(update_specs(catalog, updates))
        for name in sorted(updates):
            print(f"  updated {name}: {field}={updates[name][field]}")
    else:
        print("No matching libraries found.")

    unmatched = set(parsed.keys()) - catalog_names
    if unmatched:
        print(f"\nLibraries not in catalog ({len(unmatched)}):")
        for name in sorted(unmatched):
            print(f"  {name}")


if __name__ == "__main__":
    main()
