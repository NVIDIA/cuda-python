#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import sys
import traceback
from collections.abc import Sequence

DYNAMIC_LIB_NOT_FOUND_MARKER = "CHILD_LOAD_NVIDIA_DYNAMIC_LIB_HELPER_DYNAMIC_LIB_NOT_FOUND_ERROR:"


def _validate_abs_path(abs_path: str) -> None:
    assert abs_path, f"empty path: {abs_path=!r}"
    assert os.path.isabs(abs_path), f"not absolute: {abs_path=!r}"
    assert os.path.isfile(abs_path), f"not a file: {abs_path=!r}"


def _load_nvidia_dynamic_lib_for_test(libname: str) -> str:
    # Keep imports inside the subprocess body so startup stays focused on the
    # code under test rather than the parent test module.
    from cuda.pathfinder import load_nvidia_dynamic_lib
    from cuda.pathfinder._dynamic_libs.load_nvidia_dynamic_lib import _load_lib_no_cache
    from cuda.pathfinder._dynamic_libs.supported_nvidia_libs import (
        SUPPORTED_LINUX_SONAMES,
        SUPPORTED_WINDOWS_DLLS,
    )
    from cuda.pathfinder._utils.platform_aware import IS_WINDOWS

    loaded_dl_fresh = load_nvidia_dynamic_lib(libname)
    if loaded_dl_fresh.was_already_loaded_from_elsewhere:
        raise RuntimeError("loaded_dl_fresh.was_already_loaded_from_elsewhere")

    _validate_abs_path(loaded_dl_fresh.abs_path)
    assert loaded_dl_fresh.found_via is not None

    loaded_dl_from_cache = load_nvidia_dynamic_lib(libname)
    if loaded_dl_from_cache is not loaded_dl_fresh:
        raise RuntimeError("loaded_dl_from_cache is not loaded_dl_fresh")

    loaded_dl_no_cache = _load_lib_no_cache(libname)
    supported_libs = SUPPORTED_WINDOWS_DLLS if IS_WINDOWS else SUPPORTED_LINUX_SONAMES
    if not loaded_dl_no_cache.was_already_loaded_from_elsewhere and libname in supported_libs:
        raise RuntimeError("not loaded_dl_no_cache.was_already_loaded_from_elsewhere")
    if not os.path.samefile(loaded_dl_no_cache.abs_path, loaded_dl_fresh.abs_path):
        raise RuntimeError(f"not os.path.samefile({loaded_dl_no_cache.abs_path=!r}, {loaded_dl_fresh.abs_path=!r})")
    _validate_abs_path(loaded_dl_no_cache.abs_path)
    return loaded_dl_fresh.abs_path


def probe_load_nvidia_dynamic_lib_and_print_json(libname: str) -> None:
    from cuda.pathfinder import DynamicLibNotFoundError

    try:
        abs_path = _load_nvidia_dynamic_lib_for_test(libname)
    except DynamicLibNotFoundError:
        sys.stdout.write(f"{DYNAMIC_LIB_NOT_FOUND_MARKER}\n")
        traceback.print_exc(file=sys.stdout)
        return
    sys.stdout.write(f"{json.dumps(abs_path)}\n")


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        raise SystemExit("Usage: python -m cuda.pathfinder._testing.load_nvidia_dynamic_lib_subprocess <libname>")
    probe_load_nvidia_dynamic_lib_and_print_json(args[0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
