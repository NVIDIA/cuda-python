#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import sys
from collections.abc import Sequence

from cuda.pathfinder._dynamic_libs.lib_descriptor import LIB_DESCRIPTORS
from cuda.pathfinder._dynamic_libs.load_dl_common import DynamicLibNotFoundError, LoadedDL
from cuda.pathfinder._dynamic_libs.platform_loader import LOADER


def _probe_canary_abs_path(libname: str) -> str | None:
    desc = LIB_DESCRIPTORS.get(libname)
    if desc is None:
        raise ValueError(f"Unsupported canary library name: {libname!r}")
    try:
        loaded: LoadedDL | None = LOADER.load_with_system_search(desc)
    except DynamicLibNotFoundError:
        return None
    if loaded is None:
        return None
    abs_path = loaded.abs_path
    if not isinstance(abs_path, str):
        return None
    return abs_path


def probe_canary_abs_path_and_print_json(libname: str) -> None:
    print(json.dumps(_probe_canary_abs_path(libname)))


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        raise SystemExit("Usage: python -m cuda.pathfinder._dynamic_libs.canary_probe_subprocess <libname>")
    probe_canary_abs_path_and_print_json(args[0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
