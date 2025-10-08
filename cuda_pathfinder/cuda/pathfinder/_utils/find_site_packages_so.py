# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import collections
import functools
import importlib.metadata
import re

_SO_RE = re.compile(r"\.so(?:$|\.)")  # matches libfoo.so or libfoo.so.1.2.3


def split_so_version_suffix(so_filename: str) -> tuple[str, str]:
    idx = so_filename.rfind(".so")
    assert idx > 0, so_filename
    idx += 3
    return (so_filename[:idx], so_filename[idx:])


@functools.cache
def find_all_so_files_via_metadata() -> dict[str, dict[str, tuple[str, ...]]]:
    results: collections.defaultdict[str, collections.defaultdict[str, list[str]]] = collections.defaultdict(
        lambda: collections.defaultdict(list)
    )

    # sort dists for deterministic output
    for dist in sorted(importlib.metadata.distributions(), key=lambda d: (d.metadata.get("Name", ""), d.version)):
        files = dist.files
        if not files:
            continue
        for relpath in sorted(files, key=lambda p: str(p)):  # deterministic
            relname = relpath.name
            if not _SO_RE.search(relname):
                continue
            so_basename, so_version_suffix = split_so_version_suffix(relname)
            abs_path = str(dist.locate_file(relpath))
            results[so_basename][so_version_suffix].append(abs_path)

    # plain dicts; sort inner lists for stability
    return {k: {kk: tuple(sorted(vv)) for kk, vv in v.items()} for k, v in results.items()}
