#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

from cuda.pathfinder._dynamic_libs.load_dl_common import DynamicLibNotFoundError, LoadedDL
from cuda.pathfinder._utils.platform_aware import IS_WINDOWS

if IS_WINDOWS:
    from cuda.pathfinder._dynamic_libs.load_dl_windows import load_with_system_search
else:
    from cuda.pathfinder._dynamic_libs.load_dl_linux import load_with_system_search


def _probe_canary_abs_path(libname: str) -> str | None:
    try:
        loaded: LoadedDL | None = load_with_system_search(libname)
    except DynamicLibNotFoundError:
        return None
    if loaded is None:
        return None
    abs_path = loaded.abs_path
    if not isinstance(abs_path, str):
        return None
    return abs_path


def probe_canary_abs_path_and_print_json(libname: str) -> None:
    print(json.dumps(_probe_canary_abs_path(libname)))  # noqa: T201
