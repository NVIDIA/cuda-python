# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

import os

from Cython.Build import cythonize
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext as _build_ext


ext_modules = (
    Extension(
        "cuda.core._dlpack",
        sources=["cuda/core/_dlpack.pyx"],
        language="c++",
    ),
    Extension(
        "cuda.core._memoryview",
        sources=["cuda/core/_memoryview.pyx"],
        language="c++",
    ),
    Extension(
        "cuda.core._kernel_arg_handler",
        sources=["cuda/core/_kernel_arg_handler.pyx"],
        language="c++",
    ),
)


class build_ext(_build_ext):

    def build_extensions(self):
        self.parallel = os.cpu_count() // 2
        super().build_extensions()


setup(
    ext_modules=cythonize(ext_modules,
        verbose=True, language_level=3,
        compiler_directives={'embedsignature': True}),
    packages=find_packages(include=['cuda.core', 'cuda.core.*']),
    package_data=dict.fromkeys(
        find_packages(include=["cuda.core.*"]),
        ["*.pxd", "*.pyx", "*.py"],
    ),
    cmdclass = {'build_ext': build_ext,},
    zip_safe=False,
)
