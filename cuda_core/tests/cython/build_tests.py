# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build cuda_core Cython test extensions in-place.

pixi-build's editable install exposes the `cuda` namespace package via a
PEP 660 finder hook. Python's import machinery honors the hook, but
Cython's filesystem .pxd resolver only walks real directories on sys.path,
so `cimport cuda.bindings.*` fails to locate the .pxd files. We resolve
the namespace package's source root from `cuda.bindings.__file__` and pass
it via `include_path=` so cythonize finds the .pxd tree on every platform.

When CUDA_PYTHON_CYTHON_CACHE_DIR is set, cythonize uses the same cache
namespacing and include-path aliasing as cuda_core/build_hooks.py.
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


def _load_module(name, path, *, register=False):
    # PEP 517 backend files, not installed modules. Load by path so we do not
    # put the package directory on sys.path (that would shadow the installed
    # package). With ``register`` the module is also entered into sys.modules,
    # which is how build_hooks.py's ``from _build_shared import ...`` finds it.
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    if register:
        sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_build_hooks(modname):
    package_root = Path(__file__).resolve().parents[2]
    _load_module("_build_shared", package_root / "_build_shared.py", register=True)
    return _load_module(modname, package_root / "build_hooks.py")


build_hooks = _load_build_hooks("cuda_core_build_hooks")


def _bindings_source_root() -> Path:
    # cuda.bindings.__file__ -> .../<root>/cuda/bindings/__init__.py
    root = Path(cuda.bindings.__file__).resolve().parents[2]
    if not (root / "cuda" / "bindings").is_dir():
        raise RuntimeError(
            f"cuda.bindings source tree not found at {root}; pixi-build editable install layout may have changed."
        )
    return root


def _cythonize_tests(pyx_files):
    cache_path = build_hooks._cython_cache_path(
        "cuda-core-cython-tests",
        compiler_directives=_COMPILER_DIRECTIVES,
        language_level=3,
        cplus=True,
    )
    cythonize_kwargs = {
        "language_level": 3,
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
    # next to build_hooks.py (package root).
    stdlib_target = Path(Cython.__file__).parent / "Includes"
    with (
        build_hooks._stable_cython_alias(stdlib_target, Path(".cython-stdlib-tests")) as rel_stdlib,
        build_hooks._stable_cython_alias(_bindings_source_root(), Path(".cython-bindings-tests")) as rel_bindings,
    ):
        return cythonize(
            pyx_files,
            include_path=[".", rel_bindings, rel_stdlib],
            **cythonize_kwargs,
        )


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    pyx_files = sorted(p.name for p in script_dir.glob("test_*.pyx"))
    if not pyx_files:
        raise SystemExit(f"no test_*.pyx files under {script_dir}")

    # `build_ext --inplace` places the compiled .so relative to the current
    # working directory, but pixi runs this task from the project root. pytest
    # imports each extension by bare module name (see test_cython.py), which
    # only resolves when the .so sits in tests/cython (the dir pytest puts on
    # sys.path). chdir before cythonize so alias relpaths and the .so location
    # are stable regardless of the invoking cwd.
    os.chdir(script_dir)
    ext_modules = _cythonize_tests(pyx_files)

    sys.argv = [sys.argv[0], "build_ext", "--inplace"]
    setup(name="cuda_core_cython_tests", ext_modules=ext_modules)


if __name__ == "__main__":
    main()
