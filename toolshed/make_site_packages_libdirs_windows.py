#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Build a dict mapping Windows CUDA/NVIDIA DLL *base* names to the tuple of directories
they appear in.

Input:
- A file path argument (text file with one DLL path per line), or stdin if omitted.
- Lines may be relative or absolute paths. Blank lines and lines starting with '#'
  are ignored.

Output:
- Prints a valid Python dict literal:
    { "nvrtc": ("nvidia/cu13/bin/x86_64", "nvidia/cuda_nvrtc/bin", ...), ... }

Use the resulting directories to glob like:  f"{dirname}/{libname}*.dll"
"""

import sys
import os
import re
import argparse
from typing import Optional, Dict, Set

_SITE_PACKAGES_RE = re.compile(r"(?i)^.*?/site-packages/")


def strip_site_packages_prefix(p: str) -> str:
    """Remove any leading '.../site-packages/' (handles '\' or '/', case-insensitive)."""
    p = p.replace("\\", "/")
    return _SITE_PACKAGES_RE.sub("", p)


def extract_libname_from_dll(fname: str) -> Optional[str]:
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


def parse_lines_to_map(lines) -> Dict[str, Set[str]]:
    """Collect {libname: set(dirnames)} with deduped directories."""
    m: Dict[str, Set[str]] = {}
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        line = strip_site_packages_prefix(line)
        dirpath, fname = os.path.split(line)
        libname = extract_libname_from_dll(fname)
        if not libname:
            continue
        m.setdefault(libname, set()).add(dirpath)
    return m


def dict_literal(d: Dict[str, Set[str]]) -> str:
    """Pretty, stable dict literal with tuple values (singletons keep trailing comma)."""
    lines = ["{"]
    for k in sorted(d):
        dirs = sorted(d[k])
        tup = (
            "("
            + ", ".join(repr(x) for x in dirs)
            + ("," if len(dirs) == 1 else "")
            + ")"
        )
        lines.append(f"    {k!r}: {tup},")
    lines.append("}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(
        description="Convert a list of Windows DLL paths to {libname: (dir1, dir2, ...)}, "
        "stripping any prefix up to 'site-packages/'."
    )
    ap.add_argument(
        "paths", nargs="*", help="Input file(s); if omitted, read from stdin"
    )
    args = ap.parse_args()

    lines = []
    if args.paths:
        for path in args.paths:
            with open(path, "r", encoding="utf-8") as f:
                # Append as if files were `cat`'d together
                lines.extend(f.read().splitlines())
    else:
        lines = sys.stdin.read().splitlines()

    m = parse_lines_to_map(lines)
    print(dict_literal(m))


if __name__ == "__main__":
    main()
