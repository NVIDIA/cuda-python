# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import os

import build_hooks  # our build backend
from setuptools import setup
from setuptools.command.build_ext import build_ext as _build_ext

nthreads = int(os.environ.get("CUDA_PYTHON_PARALLEL_LEVEL", os.cpu_count() // 2))


class build_ext(_build_ext):
    def build_extensions(self):
        self.parallel = nthreads
        super().build_extensions()


setup(
    ext_modules=build_hooks._extensions,
    cmdclass={
        "build_ext": build_ext,
    },
    zip_safe=False,
)
