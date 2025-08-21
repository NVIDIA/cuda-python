#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# For usage see top of collect_site_packages_so_files.sh

import re
import sys
import os
import argparse

_SITE_PACKAGES_RE = re.compile(r"(?i)^.*?/site-packages/")


def strip_site_packages_prefix(p: str) -> str:
    """Remove any leading '.../site-packages/' (handles '\' or '/', case-insensitive)."""
    p = p.replace("\\", "/")
    return _SITE_PACKAGES_RE.sub("", p)


def parse_lines(lines):
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


def dict_literal(d):
    # Produce a stable, pretty dict literal with tuples
    lines = ["{"]
    for k in sorted(d):
        dirs = sorted(d[k])
        # Ensure single-element tuples have a trailing comma
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
        description="Convert a list of lib paths to a dict {name: (dirs, ...)}"
    )
    ap.add_argument("path", nargs="?", help="Input file (defaults to stdin)")
    args = ap.parse_args()

    if args.path:
        with open(args.path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    else:
        lines = sys.stdin.read().splitlines()

    d = parse_lines(lines)
    print(dict_literal(d))


if __name__ == "__main__":
    main()
