# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ctypes
import ctypes.util
import os
import platform
import sys
from contextlib import suppress

__all__ = [
    "IS_LINUX",
    "IS_WINDOWS",
    "IS_WSL",
    "libc",
    "under_compute_sanitizer",
]


# Please keep in sync with the equivalent implementation in
# cuda_core/cuda/core/system/_system.pyx.
def _detect_wsl() -> bool:
    data = ""
    with suppress(Exception), open("/proc/sys/kernel/osrelease") as f:
        data = f.read().lower()
    return "microsoft" in data or "wsl" in data


IS_WSL: bool = _detect_wsl()
IS_WINDOWS: bool = platform.system() == "Windows" or sys.platform.startswith("win")
IS_LINUX: bool = not IS_WINDOWS and not IS_WSL and platform.system() == "Linux"


def _load_libc() -> ctypes.CDLL:
    """Load the C runtime.

    ``libc.so.6`` is the glibc soname specifically: it does not exist on musl
    (Alpine) or on macOS. It is tried first so the library resolved on the
    platforms CI runs on is bit-for-bit what it was before, with
    ``ctypes.util.find_library`` only as a fallback.

    Failing here is expensive out of proportion to the one function that needs
    it: this module is pulled in by ``cuda_core/tests/conftest.py`` through
    ``pytest_plugins``, so an unloadable libc stops the entire suite from being
    *collected*, including every test that never touches ``libc``.
    """
    if IS_WINDOWS:
        return ctypes.CDLL("msvcrt.dll")

    tried = []
    for candidate in ("libc.so.6", ctypes.util.find_library("c")):
        if candidate is None:
            continue
        tried.append(candidate)
        try:
            return ctypes.CDLL(candidate)
        except OSError:
            continue
    raise OSError(f"could not load the C runtime on {platform.system()}; tried {tried}")


libc = _load_libc()


def under_compute_sanitizer() -> bool:
    """Return True if the current process is likely running under compute-sanitizer.

    This is best-effort and primarily intended for CI, where the environment
    is configured by wrapper scripts.
    """
    # Explicit override (if we ever want to set this directly in CI).
    if os.environ.get("CUDA_PYTHON_UNDER_SANITIZER") == "1":
        return True

    # CI sets these when compute-sanitizer is enabled.
    if os.environ.get("SETUP_SANITIZER") == "1":
        return True

    cmd = os.environ.get("SANITIZER_CMD", "")
    if "compute-sanitizer" in cmd or "cuda-memcheck" in cmd:
        return True

    # Secondary signals: depending on how tests are invoked, the wrapper name may
    # appear in argv (e.g. `compute-sanitizer pytest ...`). This is not reliable
    # in general (often argv0 is `python`/`pytest`), but it's cheap and harmless.
    argv0 = os.path.basename(sys.argv[0]) if sys.argv else ""
    if argv0 in ("compute-sanitizer", "cuda-memcheck"):
        return True
    if any(("compute-sanitizer" in a or "cuda-memcheck" in a) for a in sys.argv):
        return True

    # Another common indicator: sanitizer injectors are configured via env vars.
    inj = os.environ.get("CUDA_INJECTION64_PATH", "")
    return "compute-sanitizer" in inj or "cuda-memcheck" in inj


def driver_version_less_than(target):
    from cuda.bindings import driver

    (err,) = driver.cuInit(0)
    assert err == driver.CUresult.CUDA_SUCCESS
    err, version = driver.cuDriverGetVersion()
    assert err == driver.CUresult.CUDA_SUCCESS
    return version < target
