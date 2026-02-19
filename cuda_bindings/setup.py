# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

import os
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
