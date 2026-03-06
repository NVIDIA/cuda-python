#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

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
