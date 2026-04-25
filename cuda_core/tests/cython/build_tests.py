# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# The Cython CLI has no --include-path flag, and pixi-build's editable install
# exposes the cuda namespace package through a finder hook that Cython's
# filesystem .pxd resolver does not consult. Locate the package's parent
# directory at runtime and pass it to cythonize() explicitly.

import glob
import os
from pathlib import Path

from Cython.Build import cythonize
from setuptools import setup

import cuda.bindings

HERE = Path(__file__).resolve().parent
CUDA_PKG_PARENT = Path(cuda.bindings.__file__).parents[2]

# `setup(... build_ext --inplace)` resolves the .so destination relative to cwd
# and the module's basename, so chdir into HERE before invoking it.
os.chdir(HERE)

extensions = cythonize(
    sorted(glob.glob("test_*.pyx")),
    language_level=3,
    include_path=[str(CUDA_PKG_PARENT)],
    compiler_directives={"freethreading_compatible": True},
)

setup(ext_modules=extensions, script_args=["build_ext", "--inplace"])
