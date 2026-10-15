# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import os
from warnings import warn

import build_hooks
from setuptools import setup
from setuptools.command.build_ext import build_ext as _build_ext
from setuptools.command.build_py import build_py as _build_py

if os.environ.get("PARALLEL_LEVEL") is not None:
    warn(
        "Environment variable PARALLEL_LEVEL is deprecated. Use CUDA_PYTHON_PARALLEL_LEVEL instead",
        DeprecationWarning,
        stacklevel=1,
    )
    nthreads = int(os.environ.get("PARALLEL_LEVEL", "0"))
else:
    nthreads = int(os.environ.get("CUDA_PYTHON_PARALLEL_LEVEL", "0") or "0")

coverage_mode = bool(int(os.environ.get("CUDA_PYTHON_COVERAGE", "0")))


class build_ext(_build_ext):
    def build_extensions(self):
        if nthreads > 0:
            self.parallel = nthreads
        # A stale .so from a previous toolchain looks perfectly fresh;
        # see build_hooks._check_build_toolchain().
        if build_hooks.force_build_ext:
            self.force = True
        super().build_extensions()
        build_hooks.record_build_toolchain()


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
