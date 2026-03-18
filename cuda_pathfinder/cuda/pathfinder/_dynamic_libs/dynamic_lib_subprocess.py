#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import sys
from collections.abc import Sequence

from cuda.pathfinder._dynamic_libs.lib_descriptor import LIB_DESCRIPTORS
from cuda.pathfinder._dynamic_libs.load_dl_common import DynamicLibNotFoundError, LoadedDL
from cuda.pathfinder._dynamic_libs.platform_loader import LOADER
from cuda.pathfinder._dynamic_libs.subprocess_protocol import (
    MODE_CANARY,
    MODE_LOAD,
    STATUS_NOT_FOUND,
    STATUS_OK,
    VALID_MODES,
    format_dynamic_lib_subprocess_payload,
)

# NOTE: The main entrypoint (below) serves both production (canary probe)
# and tests (full loader). Keeping them together ensures a single subprocess
# protocol and CLI surface, so the test subprocess stays aligned with the
# production flow while avoiding a separate test-only module.
# Any production-code impact is negligible since the extra logic only runs
# in the subprocess entrypoint and only in test mode.


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
    abs_path: str | None = loaded.abs_path
    return abs_path


def _validate_abs_path(abs_path: str) -> None:
    assert abs_path, f"empty path: {abs_path=!r}"
    assert os.path.isabs(abs_path), f"not absolute: {abs_path=!r}"
    assert os.path.isfile(abs_path), f"not a file: {abs_path=!r}"


def _load_nvidia_dynamic_lib_for_test(libname: str) -> str:
    """Test-only loader used by the subprocess entrypoint."""
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

    abs_path = loaded_dl_fresh.abs_path
    if not isinstance(abs_path, str):
        raise RuntimeError(f"loaded_dl_fresh.abs_path is not a string: {abs_path!r}")
    _validate_abs_path(abs_path)
    assert loaded_dl_fresh.found_via is not None

    loaded_dl_from_cache = load_nvidia_dynamic_lib(libname)
    if loaded_dl_from_cache is not loaded_dl_fresh:
        raise RuntimeError("loaded_dl_from_cache is not loaded_dl_fresh")

    loaded_dl_no_cache = _load_lib_no_cache(libname)
    supported_libs = SUPPORTED_WINDOWS_DLLS if IS_WINDOWS else SUPPORTED_LINUX_SONAMES
    if not loaded_dl_no_cache.was_already_loaded_from_elsewhere and libname in supported_libs:
        raise RuntimeError("not loaded_dl_no_cache.was_already_loaded_from_elsewhere")
    abs_path_no_cache = loaded_dl_no_cache.abs_path
    if not isinstance(abs_path_no_cache, str):
        raise RuntimeError(f"loaded_dl_no_cache.abs_path is not a string: {abs_path_no_cache!r}")
    if not os.path.samefile(abs_path_no_cache, abs_path):
        raise RuntimeError(f"not os.path.samefile({abs_path_no_cache=!r}, {abs_path=!r})")
    _validate_abs_path(abs_path_no_cache)
    return abs_path


def probe_dynamic_lib_and_print_json(libname: str, mode: str) -> None:
    if mode == MODE_CANARY:
        abs_path = _probe_canary_abs_path(libname)
        status = STATUS_OK if abs_path is not None else STATUS_NOT_FOUND
        print(format_dynamic_lib_subprocess_payload(status, abs_path))
        return

    if mode == MODE_LOAD:
        # Test-only path: exercises full loader behavior in isolation.
        try:
            abs_path = _load_nvidia_dynamic_lib_for_test(libname)
        except DynamicLibNotFoundError as exc:
            error = {
                "type": exc.__class__.__name__,
                "message": str(exc),
            }
            print(format_dynamic_lib_subprocess_payload(STATUS_NOT_FOUND, None, error=error))
            return
        print(format_dynamic_lib_subprocess_payload(STATUS_OK, abs_path))
        return

    raise ValueError(f"Unsupported subprocess probe mode: {mode!r}")


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 2 or args[0] not in VALID_MODES:
        modes = ", ".join(VALID_MODES)
        raise SystemExit(
            f"Usage: python -m cuda.pathfinder._dynamic_libs.dynamic_lib_subprocess <mode> <libname>\nModes: {modes}"
        )
    mode, libname = args
    probe_dynamic_lib_and_print_json(libname, mode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
