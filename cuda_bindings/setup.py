# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import functools
import subprocess

import build_hooks
from setuptools import setup
from setuptools.command.build_ext import build_ext as _build_ext
from setuptools.command.build_py import build_py as _build_py

# Shared with build_hooks so the two build entry points parse these knobs
# identically (see build_hooks.env_int for why a bare int() is not enough).
nthreads = build_hooks.parallel_level()

coverage_mode = bool(build_hooks.env_int("CUDA_PYTHON_COVERAGE", 0))


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


class build_py(_build_py):
    def finalize_options(self):
        super().finalize_options()
        if coverage_mode:
            self.package_data.setdefault("", [])
            self.package_data[""] += ["*.pyx", "*.cpp"]


setup(
    ext_modules=build_hooks._extensions,
    cmdclass={
        "build_ext": build_ext,
        "build_py": build_py,
    },
    zip_safe=False,
)
