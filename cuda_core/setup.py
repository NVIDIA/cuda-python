# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import os

import build_hooks  # our build backend
from setuptools import setup
from setuptools.command.build_ext import build_ext as _build_ext
from setuptools.command.build_py import build_py as _build_py

nthreads = int(os.environ.get("CUDA_PYTHON_PARALLEL_LEVEL", os.cpu_count() // 2))
coverage_mode = bool(int(os.environ.get("CUDA_PYTHON_COVERAGE", "0")))


class build_ext(_build_ext):  # noqa: N801
    def build_extensions(self):
        self.parallel = nthreads
        super().build_extensions()


class build_py(_build_py):  # noqa: N801
    def finalize_options(self):
        super().finalize_options()
        if coverage_mode:
            self.package_data.setdefault("", [])
            self.package_data[""] += ["*.pxi", "*.pyx", "*.cpp"]


setup(
    ext_modules=build_hooks._extensions,
    cmdclass={
        "build_ext": build_ext,
        "build_py": build_py,
    },
    zip_safe=False,
)
