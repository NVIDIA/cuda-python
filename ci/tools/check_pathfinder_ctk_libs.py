# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check that cuda.pathfinder loads the given libraries from the CTK under CUDA_PATH.

The default cuda.pathfinder tests tolerate libraries that cannot be found, so a
layout mismatch between the mini CTK and pathfinder's search directories would
otherwise go unnoticed. The library may be found through any search step, as
long as it comes from under CUDA_PATH.

Usage: python check_pathfinder_ctk_libs.py LIBNAME [LIBNAME ...]
"""

from __future__ import annotations

import os
import sys

from cuda.pathfinder import load_nvidia_dynamic_lib


def _canonical(path: str) -> str:
    return os.path.normcase(os.path.realpath(path))


def main(libnames: list[str]) -> int:
    root = _canonical(os.environ["CUDA_PATH"])
    failed = False
    for libname in libnames:
        loaded = load_nvidia_dynamic_lib(libname)
        if loaded.abs_path is None:
            print(f"FATAL: cuda.pathfinder did not report a path for {libname}", file=sys.stderr)
            failed = True
            continue
        path = _canonical(loaded.abs_path)
        print(f"{libname}: {path} (found via {loaded.found_via})")
        if not path.startswith(root + os.sep):
            print(f"FATAL: {libname} was not loaded from CUDA_PATH ({root})", file=sys.stderr)
            failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
