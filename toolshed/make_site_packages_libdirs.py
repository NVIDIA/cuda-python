#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# For usage see top of collect_site_packages_*_files.*

import os
import re
import argparse
from typing import Optional, Dict, Set

_SITE_PACKAGES_RE = re.compile(r"(?i)^.*?/site-packages/")


def strip_site_packages_prefix(p: str) -> str:
    """Remove any leading '.../site-packages/' (handles '\' or '/', case-insensitive)."""
    p = p.replace("\\", "/")
    return _SITE_PACKAGES_RE.sub("", p)


def parse_lines_linux(lines) -> Dict[str, Set[str]]:
    d = {}  # name -> set of dirs
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        line = strip_site_packages_prefix(line)
        dirpath, fname = os.path.split(line)
        # Require something like libNAME.so, libNAME.so.12, libNAME.so.12.1, etc.
        i = fname.find(".so")
        if not fname.startswith("lib") or i == -1:
            # Skip lines that don't look like shared libs
            continue
        name = fname[:i]  # e.g. "libnvrtc"
        name = name[3:]  # drop leading "lib" -> "nvrtc"
        d.setdefault(name, set()).add(dirpath)
    return d


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


def parse_lines_windows(lines) -> Dict[str, Set[str]]:
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


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Convert a list of site-packages library paths into {name: (dirs, ...)}"
    )
    ap.add_argument(
        "platform", choices=["linux", "windows"], help="Target platform to parse"
    )
    ap.add_argument("path", help="Text file with one library path per line")
    args = ap.parse_args()

    with open(args.path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()

    if args.platform == "linux":
        m = parse_lines_linux(lines)
    else:
        m = parse_lines_windows(lines)
    print(dict_literal(m))


if __name__ == "__main__":
    main()
