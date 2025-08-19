# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import collections
import fnmatch
import functools
import site
from pathlib import Path


def split_so_version_suffix(so_filename: str) -> tuple[str, str]:
    idx = so_filename.rfind(".so")
    assert idx > 0, so_filename
    idx += 3
    return (so_filename[:idx], so_filename[idx:])


@functools.cache
def find_all_so_files_under_all_site_packages() -> dict[str, dict[str, list[str]]]:
    # Collect candidate site-packages directories (user first, then system).
    dirs: list[Path] = []
    user_sp = site.getusersitepackages()
    if user_sp:
        dirs.append(Path(user_sp))
    dirs.extend(Path(p) for p in site.getsitepackages() if p)

    # Normalize, dedupe, and keep only existing directories. Sort for determinism.
    norm_dirs: list[Path] = []
    seen = set()
    for d in dirs:
        for subdir in ("nvidia", "nvpl"):
            ds = d / subdir
            try:
                p = ds.resolve()
            except Exception:
                p = ds
            if p.exists() and p.is_dir() and p not in seen:
                seen.add(p)
                norm_dirs.append(p)
    norm_dirs.sort()

    # results[so_basename][so_version_suffix]
    results: collections.defaultdict[str, collections.defaultdict[str, list[str]]] = collections.defaultdict(
        lambda: collections.defaultdict(list)
    )

    # Walk each site-packages tree and collect *.so* files.
    for base in norm_dirs:
        # pathlib.rglob is concise and cross-platform
        for path in base.rglob("*"):
            # Quick prune of dirs named __pycache__
            if path.name == "__pycache__":
                continue
            if path.is_file() and fnmatch.fnmatch(path.name, "*.so*"):
                so_basename, so_version_suffix = split_so_version_suffix(path.name)
                results[so_basename][so_version_suffix].append(str(path))

    # Convert to plain dict, to reduce potential for accidents downstream.
    return {k: dict(v) for k, v in results.items()}
