# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

from Cython.Build import cythonize
from setuptools import setup, Extension, find_packages


ext_modules = (
    Extension(
        "cuda.py._dlpack",
        sources=["cuda/py/_dlpack.pyx"],
        language="c++",
    ),
)


setup(
    ext_modules=cythonize(ext_modules,
        verbose=True, language_level=3,
        compiler_directives={'embedsignature': True}),
    packages=find_packages(include=['cuda.py', 'cuda.py.*']),
    package_data=dict.fromkeys(
        find_packages(include=["cuda.py.*"]),
        ["*.pxd", "*.pyx", "*.py"],
    ),
    zip_safe=False,
)
