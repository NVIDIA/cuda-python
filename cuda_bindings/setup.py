# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

import os

import build_hooks
from setuptools import setup
from setuptools.command.build_ext import build_ext as _build_ext

nthreads = int(os.environ.get("CUDA_PYTHON_PARALLEL_LEVEL", "0") or "0")


class build_ext(_build_ext):
    def build_extensions(self):
        if nthreads > 0:
            self.parallel = nthreads
        super().build_extensions()


setup(
    ext_modules=build_hooks._extensions,
    cmdclass={
        "build_ext": build_ext,
    },
    zip_safe=False,
)
