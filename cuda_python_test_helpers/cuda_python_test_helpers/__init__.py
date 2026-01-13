# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ctypes
import os
import platform
import sys
from contextlib import suppress

__all__ = [
    "IS_WINDOWS",
    "IS_WSL",
    "libc",
    "under_compute_sanitizer",
]


def _detect_wsl() -> bool:
    data = ""
    with suppress(Exception), open("/proc/sys/kernel/osrelease") as f:
        data = f.read().lower()
    if "microsoft" in data or "wsl" in data:
        return True
    return any(os.environ.get(k) for k in ("WSL_DISTRO_NAME", "WSL_INTEROP"))


IS_WSL: bool = _detect_wsl()
IS_WINDOWS: bool = platform.system() == "Windows" or sys.platform.startswith("win")

if IS_WINDOWS:
    libc = ctypes.CDLL("msvcrt.dll")
else:
    libc = ctypes.CDLL("libc.so.6")


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
