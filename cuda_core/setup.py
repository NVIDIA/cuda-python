# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

from Cython.Build import cythonize
from setuptools import setup, Extension, find_packages


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
)


setup(
    ext_modules=cythonize(ext_modules,
        verbose=True, language_level=3,
        compiler_directives={'embedsignature': True}),
    packages=find_packages(include=['cuda.core', 'cuda.core.*']),
    package_data=dict.fromkeys(
        find_packages(include=["cuda.core.*"]),
        ["*.pxd", "*.pyx", "*.py"],
    ),
    zip_safe=False,
)
