# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import glob
import os

from Cython.Build import cythonize
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext as _build_ext

nthreads = int(os.environ.get("CUDA_PYTHON_PARALLEL_LEVEL", os.cpu_count() // 2))


# It seems setuptools' wildcard support has problems for namespace packages,
# so we explicitly spell out all Extension instances.
root_module = "cuda.core.experimental"
root_path = f"{os.path.sep}".join(root_module.split(".")) + os.path.sep
ext_files = glob.glob(f"{root_path}/**/*.pyx", recursive=True)


def strip_prefix_suffix(filename):
    return filename[len(root_path) : -4]


module_names = (strip_prefix_suffix(f) for f in ext_files)
ext_modules = tuple(
    Extension(
        f"cuda.core.experimental.{mod.replace(os.path.sep, '.')}",
        sources=[f"cuda/core/experimental/{mod}.pyx"],
        language="c++",
    )
    for mod in module_names
)


class build_ext(_build_ext):
    def build_extensions(self):
        self.parallel = nthreads
        super().build_extensions()


setup(
    ext_modules=cythonize(
        ext_modules,
        verbose=True,
        language_level=3,
        compiler_directives={"embedsignature": True, "freethreading_compatible": True},
    ),
    cmdclass={
        "build_ext": build_ext,
    },
    zip_safe=False,
)
