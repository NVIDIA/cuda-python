# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build cuda_bindings Cython test extensions in-place.

pixi-build's editable install exposes the `cuda` namespace package via a
PEP 660 finder hook. Python's import machinery honors the hook, but
Cython's filesystem .pxd resolver only walks real directories on sys.path,
so `cimport cuda.bindings.*` fails to locate the .pxd files. We resolve
the namespace package's source root from `cuda.bindings.__file__` and pass
it via `include_path=` so cythonize finds the .pxd tree on every platform.

When CUDA_PYTHON_CYTHON_CACHE_DIR is set, cythonize uses the same cache
namespacing and include-path aliasing as the package build (``_build_shared.py``).
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import Cython
from Cython.Build import cythonize
from setuptools import setup

import cuda.bindings

_COMPILER_DIRECTIVES = {"freethreading_compatible": True}


def _load_build_shared():
    # A PEP 517 backend file, not an installed module. Load it by path so we do
    # not put the package directory on sys.path (that would shadow the
    # installed package). It needs only the standard library.
    path = Path(__file__).resolve().parents[2] / "_build_shared.py"
    spec = importlib.util.spec_from_file_location("cuda_bindings_build_shared", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


build_shared = _load_build_shared()


def _bindings_source_root() -> Path:
    # cuda.bindings.__file__ -> .../<root>/cuda/bindings/__init__.py
    root = Path(cuda.bindings.__file__).resolve().parents[2]
    if not (root / "cuda" / "bindings").is_dir():
        raise RuntimeError(
            f"cuda.bindings source tree not found at {root}; pixi-build editable install layout may have changed."
        )
    return root


def _cythonize_tests(pyx_files):
    cache_path = build_shared._cython_cache_path(
        "cuda-bindings-cython-tests",
        compiler_directives=_COMPILER_DIRECTIVES,
        language_level=3,
        cplus=True,
    )
    cythonize_kwargs = {
        "language_level": 3,
        "nthreads": 1,
        "compiler_directives": _COMPILER_DIRECTIVES,
        "cache": cache_path,
    }
    if cache_path is None:
        return cythonize(
            pyx_files,
            include_path=[str(_bindings_source_root())],
            **cythonize_kwargs,
        )

    # Distinct alias names so a concurrent package build's .cython-stdlib /
    # .cython-bindings symlinks are not replaced. Relative aliases resolve
    # next to _build_shared.py (package root).
    stdlib_target = Path(Cython.__file__).parent / "Includes"
    with (
        build_shared._stable_cython_alias(stdlib_target, Path(".cython-stdlib-tests")) as rel_stdlib,
        build_shared._stable_cython_alias(_bindings_source_root(), Path(".cython-bindings-tests")) as rel_bindings,
    ):
        return cythonize(
            pyx_files,
            include_path=[".", rel_bindings, rel_stdlib],
            **cythonize_kwargs,
        )


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    # Avoid appending the absolute checkout path under build/temp: the
    # concatenated path can exceed Windows' path limit. These files are siblings.
    os.chdir(script_dir)
    pyx_files = sorted(p.name for p in script_dir.glob("test_*.pyx"))
    if not pyx_files:
        raise SystemExit(f"no test_*.pyx files under {script_dir}")

    ext_modules = _cythonize_tests(pyx_files)

    # pytest imports each extension by bare module name (see test_cython.py),
    # so build in-place next to its .pyx regardless of the invoking cwd.
    sys.argv = [sys.argv[0], "build_ext", "--inplace"]
    setup(name="cuda_bindings_cython_tests", ext_modules=ext_modules)


if __name__ == "__main__":
    main()
