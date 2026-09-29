# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared PEP 517 backend helpers for the cuda.bindings and cuda.core build backends.

This file is the single source of truth for helpers duplicated between the two
backends: CUDA_PYTHON_TOOLCHAIN resolution, the PEP 517 pathfinder-shadowing
workaround, and the extension-ABI-scoped build stamp path helper.
`cuda_core/_build_shared.py` is a symlink to this file, so both backends
resolve the shared code through their own `backend-path`.

Per-package logic (extension list, flag assembly, stamp bookkeeping, PEP 517
hooks) stays in each package's `build_hooks.py`.
"""

import functools
import os
import shutil
import sys
import sysconfig
from pathlib import Path

# -----------------------------------------------------------------------
# CUDA path resolution (via cuda.pathfinder, with a PEP 517 shim)


def _import_get_cuda_path_or_home():
    """Import get_cuda_path_or_home, working around PEP 517 namespace shadowing.

    See https://github.com/NVIDIA/cuda-python/issues/1824 for why this helper is needed.
    """
    try:
        import cuda.pathfinder
    except ModuleNotFoundError as exc:
        if exc.name not in ("cuda", "cuda.pathfinder"):
            raise
        try:
            import cuda
        except ModuleNotFoundError:
            cuda = None

        for p in sys.path:
            sp_cuda = Path(p) / "cuda"
            if (sp_cuda / "pathfinder").is_dir():
                cuda.__path__ = list(cuda.__path__) + [str(sp_cuda)]
                break
        else:
            raise ModuleNotFoundError(
                "cuda-pathfinder is not installed in the build environment. "
                "Ensure 'cuda-pathfinder>=1.5' is in build-system.requires."
            )
        import cuda.pathfinder

    pathfinder_dir = Path(cuda.pathfinder.__file__).parent
    print(
        f"Using cuda-pathfinder {cuda.pathfinder.__version__} from {pathfinder_dir}",
        file=sys.stderr,
    )
    return cuda.pathfinder.get_cuda_path_or_home


@functools.cache
def _get_cuda_path() -> str:
    get_cuda_path_or_home = _import_get_cuda_path_or_home()
    cuda_path = get_cuda_path_or_home()
    if not cuda_path:
        raise RuntimeError("Environment variable CUDA_PATH or CUDA_HOME is not set")
    print("CUDA path:", cuda_path)
    return cuda_path


# -----------------------------------------------------------------------
# Build stamp paths
#
# `Path(__file__)` reflects the path this module was loaded from, and Python
# does not resolve symlinks on it. When cuda_core imports this file via its
# `_build_shared.py` symlink, `_BUILD_DIR` resolves to `cuda_core/build/`;
# imported directly from cuda_bindings, it resolves to `cuda_bindings/build/`.
# So each package's stamp files land under its own build/ directory even
# though the code lives in one file.

_BUILD_DIR = Path(__file__).parent / "build"


def _abi_stamp_path(stem):
    """Return a stamp path scoped to this interpreter's extension ABI."""
    extension_suffix = sysconfig.get_config_var("EXT_SUFFIX")
    if not extension_suffix:
        raise RuntimeError("Python's EXT_SUFFIX build configuration is unavailable")
    return _BUILD_DIR / f"{stem}{extension_suffix}"


# -----------------------------------------------------------------------
# Toolchain selection

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
