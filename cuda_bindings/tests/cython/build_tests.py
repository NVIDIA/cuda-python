# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE
"""Build cuda_bindings Cython test extensions in-place.

pixi-build's editable install exposes the `cuda` namespace package via a
PEP 660 finder hook. Python's import machinery honors the hook, but
Cython's filesystem .pxd resolver only walks real directories on sys.path,
so `cimport cuda.bindings.*` fails to locate the .pxd files. We resolve
the namespace package's source root from `cuda.bindings.__file__` and pass
it via `include_path=` so cythonize finds the .pxd tree on every platform.
"""

from __future__ import annotations

import sys
from pathlib import Path

from Cython.Build import cythonize
from setuptools import setup

import cuda.bindings


def _bindings_source_root() -> Path:
    # cuda.bindings.__file__ -> .../<root>/cuda/bindings/__init__.py
    root = Path(cuda.bindings.__file__).resolve().parents[2]
    if not (root / "cuda" / "bindings").is_dir():
        raise RuntimeError(
            f"cuda.bindings source tree not found at {root}; pixi-build editable install layout may have changed."
        )
    return root


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    pyx_files = sorted(str(p) for p in script_dir.glob("test_*.pyx"))
    if not pyx_files:
        raise SystemExit(f"no test_*.pyx files under {script_dir}")

    ext_modules = cythonize(
        pyx_files,
        language_level=3,
        nthreads=1,
        include_path=[str(_bindings_source_root())],
        compiler_directives={"freethreading_compatible": True},
    )

    sys.argv = [sys.argv[0], "build_ext", "--inplace"]
    setup(name="cuda_bindings_cython_tests", ext_modules=ext_modules)


if __name__ == "__main__":
    main()
