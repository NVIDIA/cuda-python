# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

import functools
import os
import subprocess
from warnings import warn

import build_hooks
from setuptools import setup
from setuptools.command.build_ext import build_ext as _build_ext

if os.environ.get("PARALLEL_LEVEL") is not None:
    warn(
        "Environment variable PARALLEL_LEVEL is deprecated. Use CUDA_PYTHON_PARALLEL_LEVEL instead",
        DeprecationWarning,
        stacklevel=1,
    )
    nthreads = int(os.environ.get("PARALLEL_LEVEL", "0"))
else:
    nthreads = int(os.environ.get("CUDA_PYTHON_PARALLEL_LEVEL", "0") or "0")


def _is_clang(compiler):
    @functools.lru_cache
    def _check(compiler_cxx):
        try:
            output = subprocess.check_output([*compiler_cxx, "--version"])  # noqa: S603
        except subprocess.CalledProcessError:
            return False
        lines = output.decode().splitlines()
        return len(lines) > 0 and "clang" in lines[0]

    if not hasattr(compiler, "compiler_cxx"):
        return False
    return _check(tuple(compiler.compiler_cxx))


class build_ext(_build_ext):
    def build_extensions(self):
        if nthreads > 0:
            self.parallel = nthreads
        if _is_clang(self.compiler):
            for ext in self.extensions:
                ext.extra_compile_args = [a for a in ext.extra_compile_args if a != "-fno-var-tracking-assignments"]
        super().build_extensions()


setup(
    ext_modules=build_hooks._extensions,
    cmdclass={
        "build_ext": build_ext,
    },
    zip_safe=False,
)
