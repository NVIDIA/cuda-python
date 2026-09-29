# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared toolchain helpers for the cuda.bindings and cuda.core build backends.

This file is the single source of truth for the CUDA_PYTHON_TOOLCHAIN
resolution logic. `cuda_core/_toolchain_shared.py` is a symlink to this file,
so both PEP 517 backends import the same helpers via their own `backend-path`.

Only the per-package `_resolve_toolchain()` flag assembly in each backend is
package-specific (they use different C++ standards and opt levels).
"""

import os
import shutil
import sys

_TOOLCHAINS_LINUX = ("gnu", "llvm")
_TOOLCHAINS_WINDOWS = ("msvc",)
_TOOLCHAIN_COMPILERS = {
    "gnu": ("gcc", "g++"),
    "llvm": ("clang", "clang++"),
    "msvc": (None, None),
}


def _resolve_toolchain_name():
    """Read CUDA_PYTHON_TOOLCHAIN, validate it, return (name, allowed, cc, cxx, explicit).

    The default toolchain (gnu on Linux, msvc on Windows) is the first entry
    of the platform's allowed tuple. cc/cxx are the compiler binaries for the
    toolchain (None for msvc, which distutils discovers via the MSVC env).
    """
    if sys.platform == "win32":
        platform_key, allowed = "win32", _TOOLCHAINS_WINDOWS
    else:
        platform_key, allowed = "linux", _TOOLCHAINS_LINUX
    name = os.environ.get("CUDA_PYTHON_TOOLCHAIN", allowed[0]).strip().lower()
    if name not in allowed:
        raise RuntimeError(
            f"CUDA_PYTHON_TOOLCHAIN={name!r} is not supported on {platform_key}. Valid values: {', '.join(allowed)}."
        )
    cc, cxx = _TOOLCHAIN_COMPILERS[name]
    explicit = bool(os.environ.get("CUDA_PYTHON_TOOLCHAIN", "").strip())
    return name, allowed, cc, cxx, explicit


def _apply_toolchain_env(cc, cxx, explicit):
    """Set CC/CXX/LDSHARED for an explicitly-chosen toolchain.

    The default path (CUDA_PYTHON_TOOLCHAIN unset) intentionally
    does not touch the env, so an externally-set compiler (e.g.
    CC="sccache cc" in CI) keeps working. An explicit CUDA_PYTHON_TOOLCHAIN
    override (incl. =gnu) governs the compiler and overrides CC/CXX/LDSHARED.
    """
    if explicit and cc is not None:
        os.environ["CC"] = cc
        os.environ["CXX"] = cxx
        os.environ["LDSHARED"] = f"{cxx} -shared"


def _check_toolchain_available(name):
    """Preflight: verify the selected toolchain's tools are on PATH.

    No-op for the platform default (distutils discovers those). For llvm,
    probes clang, clang++, and ld.lld so a missing toolchain fails fast with a
    helpful message instead of a cryptic compile error.
    """
    if name != "llvm":
        return
    tools = ("clang", "clang++", "ld.lld")
    missing = [t for t in tools if shutil.which(t) is None]
    if missing:
        raise RuntimeError(
            f"CUDA_PYTHON_TOOLCHAIN=llvm but required tool(s) not found on PATH: "
            f"{', '.join(missing)}. Install clang and lld "
            f"(e.g. `apt install clang lld` or `dnf install clang lld`) "
            f"or set CUDA_PYTHON_TOOLCHAIN=gnu."
        )
