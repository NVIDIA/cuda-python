# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import collections
import functools
import importlib.metadata


@functools.cache
def find_all_dll_files_via_metadata() -> dict[str, tuple[str, ...]]:
    results: collections.defaultdict[str, list[str]] = collections.defaultdict(list)

    # sort dists for deterministic output
    for dist in sorted(importlib.metadata.distributions(), key=lambda d: (d.metadata.get("Name", ""), d.version)):
        files = dist.files
        if not files:
            continue
        for relpath in sorted(files, key=lambda p: str(p)):  # deterministic
            relname = relpath.name.lower()
            if not relname.endswith(".dll"):
                continue
            abs_path = str(dist.locate_file(relpath))
            results[relname].append(abs_path)

    # plain dicts; sort inner list for stability
    return {k: tuple(sorted(v)) for k, v in results.items()}
