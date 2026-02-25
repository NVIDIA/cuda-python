#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Scan 7z listing files for .dll names, update descriptor_catalog.py.

Usage:
    # First generate listings from CTK .exe installers:
    #   for exe in *.exe; do 7z l "$exe" > "${exe%.exe}.txt"; done
    python toolshed/build_pathfinder_dlls.py listing1.txt [listing2.txt ...]
"""

from __future__ import annotations

import collections
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _catalog_writer import load_catalog, update_specs, write_catalog


def _is_suppressed_dll(libname: str, dll: str) -> bool:
    if libname == "cudart":
        if dll.startswith("cudart32_"):
            return True
        if dll == "cudart64_65.dll":
            # PhysX/files/Common/cudart64_65.dll from CTK 6.5, but shipped with CTK 12.0-12.9
            return True
        if dll == "cudart64_101.dll":
            # GFExperience.NvStreamSrv/amd64/server/cudart64_101.dll from CTK 10.1, but shipped with CTK 12.0-12.6
            return True
    elif libname == "nvrtc":
        if dll.endswith(".alt.dll"):
            return True
        if dll.startswith("nvrtc-builtins"):
            return True
    elif libname == "nvvm" and dll == "nvvm32.dll":
        return True
    return False


def _parse_listings(paths: list[str]) -> set[str]:
    dlls: set[str] = set()
    for filename in paths:
        lines_iter = iter(Path(filename).read_text().splitlines())
        for line in lines_iter:
            if line.startswith("-------------------"):
                break
        else:
            raise RuntimeError(f"------------------- NOT FOUND in {filename}")
        for line in lines_iter:
            if line.startswith("-------------------"):
                break
            assert line[52] == " ", line
            assert line[53] != " ", line
            path = line[53:]
            if path.endswith(".dll"):
                dll = path.rsplit("/", 1)[1]
                dlls.add(dll)
        else:
            raise RuntimeError(f"------------------- NOT FOUND in {filename}")
    return dlls


def run(listing_files: list[str]) -> None:
    dlls_from_files = _parse_listings(listing_files)
    catalog = load_catalog()

    # Longest-prefix-first to avoid ambiguous matches (e.g. "cufftw" before "cufft").
    ctk_names = sorted(
        (spec.name for spec in catalog if spec.strategy == "ctk"),
        key=lambda n: (-len(n), n),
    )

    dlls_in_scope: set[str] = set()
    dlls_by_name: dict[str, list[str]] = collections.defaultdict(list)
    suppressed: set[str] = set()

    for name in ctk_names:
        for dll in sorted(dlls_from_files):
            if dll not in dlls_in_scope and dll.startswith(name):
                if _is_suppressed_dll(name, dll):
                    suppressed.add(dll)
                else:
                    dlls_by_name[name].append(dll)
                dlls_in_scope.add(dll)

    updates: dict[str, dict[str, object]] = {}
    for name, dlls in dlls_by_name.items():
        updates[name] = {"windows_dlls": tuple(dlls)}

    if updates:
        write_catalog(update_specs(catalog, updates))
        for name in sorted(updates):
            print(f"  updated {name}: windows_dlls={updates[name]['windows_dlls']}")
    else:
        print("No matching DLLs found.")

    if suppressed:
        print(f"\nSuppressed DLLs ({len(suppressed)}):")
        for dll in sorted(suppressed):
            print(f"  {dll}")

    out_of_scope = dlls_from_files - dlls_in_scope
    if out_of_scope:
        print(f"\nDLLs out of scope ({len(out_of_scope)}):")
        for dll in sorted(out_of_scope):
            print(f"  {dll}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: build_pathfinder_dlls.py <7z-listing.txt> ...", file=sys.stderr)
        sys.exit(1)
    run(listing_files=sys.argv[1:])
