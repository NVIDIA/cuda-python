# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build cuda_core Cython test extensions in-place.

The build intentionally provides no explicit Cython include path. Editable
installs must expose their ``.pxd`` trees through physical ``sys.path`` entries
so downstream Cython consumers work without repository-specific setup.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from Cython.Build import cythonize
from setuptools import setup


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    pyx_files = sorted(str(p) for p in script_dir.glob("test_*.pyx"))
    if not pyx_files:
        raise SystemExit(f"no test_*.pyx files under {script_dir}")

    ext_modules = cythonize(
        pyx_files,
        language_level=3,
        compiler_directives={"freethreading_compatible": True},
    )

    # `build_ext --inplace` places the compiled .so relative to the current
    # working directory, but pixi runs this task from the project root. pytest
    # imports each extension by bare module name (see test_cython.py), which
    # only resolves when the .so sits in tests/cython (the dir pytest puts on
    # sys.path). chdir here so the .so lands next to its .pyx regardless of the
    # invoking cwd.
    os.chdir(script_dir)
    sys.argv = [sys.argv[0], "build_ext", "--inplace"]
    setup(name="cuda_core_cython_tests", ext_modules=ext_modules)


if __name__ == "__main__":
    main()
