# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

import os

from Cython.Build import cythonize
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext as _build_ext


CUDA_HOME = os.environ.get("CUDA_HOME", os.environ.get("CUDA_PATH", None))
if not CUDA_HOME:
    raise RuntimeError("Environment variable CUDA_HOME or CUDA_PATH is not set")

CUDA_HOME = CUDA_HOME.split(os.pathsep)

include_path_list = [os.path.join(path, "include") for path in CUDA_HOME]

ext_modules = (
    Extension(
        "cuda.core.experimental._dlpack",
        sources=["cuda/core/experimental/_dlpack.pyx"],
        language="c++",
    ),
    Extension(
        "cuda.core.experimental._stream",
        sources=["cuda/core/experimental/_stream.pyx"],
        include_dirs=include_path_list,
        language="c++",
    ),
    Extension(
        "cuda.core.experimental._memoryview",
        sources=["cuda/core/experimental/_memoryview.pyx"],
        language="c++",
    ),
    Extension(
        "cuda.core.experimental._kernel_arg_handler",
        sources=["cuda/core/experimental/_kernel_arg_handler.pyx"],
        language="c++",
    ),
    Extension(
        "cuda.core.experimental._utils._error_utils",
        sources=["cuda/core/experimental/_utils/_error_utils.pyx"],
        include_dirs=include_path_list,
        language="c++",
    ),
)


class build_ext(_build_ext):
    def build_extensions(self):
        self.parallel = os.cpu_count() // 2
        super().build_extensions()


setup(
    ext_modules=cythonize(ext_modules, verbose=True, emit_linenums=True, language_level=3, compiler_directives={"embedsignature": True}),
    cmdclass={
        "build_ext": build_ext,
    },
    zip_safe=False,
)
