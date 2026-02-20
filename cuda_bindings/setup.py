# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

import functools
import os
import pathlib
import subprocess
from warnings import warn

from setuptools import setup
from setuptools.command.build_ext import build_ext as _build_ext
from setuptools.command.build_py import build_py
from setuptools.command.editable_wheel import _TopLevelFinder, editable_wheel

import build_hooks

if os.environ.get("PARALLEL_LEVEL") is not None:
    warn(
        "Environment variable PARALLEL_LEVEL is deprecated. Use CUDA_PYTHON_PARALLEL_LEVEL instead",
        DeprecationWarning,
        stacklevel=1,
    )
    nthreads = int(os.environ.get("PARALLEL_LEVEL", "0"))
else:
    nthreads = int(os.environ.get("CUDA_PYTHON_PARALLEL_LEVEL", "0") or "0")


def _is_clang(compiler):
    @functools.lru_cache
    def _check(compiler_cxx):
        try:
            output = subprocess.check_output([*compiler_cxx, "--version"])  # noqa: S603
        except subprocess.CalledProcessError:
            return False
        lines = output.decode().splitlines()
        return len(lines) > 0 and "clang" in lines[0]

    if not hasattr(compiler, "compiler_cxx"):
        return False
    return _check(tuple(compiler.compiler_cxx))


class build_ext(_build_ext):
    def build_extensions(self):
        if nthreads > 0:
            self.parallel = nthreads
        if _is_clang(self.compiler):
            for ext in self.extensions:
                ext.extra_compile_args = [a for a in ext.extra_compile_args if a != "-fno-var-tracking-assignments"]
        super().build_extensions()


################################################################################
# Adapted from NVIDIA/numba-cuda
# TODO: Remove this block once we get rid of cuda.__version__ and the .pth files

REDIRECTOR_PTH = "_cuda_bindings_redirector.pth"
REDIRECTOR_PY = "_cuda_bindings_redirector.py"
SITE_PACKAGES = pathlib.Path("site-packages")


class build_py_with_redirector(build_py):  # noqa: N801
    """Include the redirector files in the generated wheel."""

    def copy_redirector_file(self, source, destination="."):
        destination = pathlib.Path(self.build_lib) / destination
        self.copy_file(str(source), str(destination), preserve_mode=0)

    def run(self):
        super().run()
        self.copy_redirector_file(SITE_PACKAGES / REDIRECTOR_PTH)
        self.copy_redirector_file(SITE_PACKAGES / REDIRECTOR_PY)

    def get_source_files(self):
        src = super().get_source_files()
        src.extend(
            [
                str(SITE_PACKAGES / REDIRECTOR_PTH),
                str(SITE_PACKAGES / REDIRECTOR_PY),
            ]
        )
        return src

    def get_output_mapping(self):
        mapping = super().get_output_mapping()
        build_lib = pathlib.Path(self.build_lib)
        mapping[str(build_lib / REDIRECTOR_PTH)] = REDIRECTOR_PTH
        mapping[str(build_lib / REDIRECTOR_PY)] = REDIRECTOR_PY
        return mapping


class TopLevelFinderWithRedirector(_TopLevelFinder):
    """Include the redirector files in the editable wheel."""

    def get_implementation(self):
        for item in super().get_implementation():  # noqa: UP028
            yield item

        with open(SITE_PACKAGES / REDIRECTOR_PTH) as f:
            yield (REDIRECTOR_PTH, f.read())

        with open(SITE_PACKAGES / REDIRECTOR_PY) as f:
            yield (REDIRECTOR_PY, f.read())


class editable_wheel_with_redirector(editable_wheel):
    def _select_strategy(self, name, tag, build_lib):
        # The default mode is "lenient" - others are "strict" and "compat".
        # "compat" is deprecated. "strict" creates a tree of links to files in
        # the repo. It could be implemented, but we only handle the default
        # case for now.
        if self.mode is not None and self.mode != "lenient":
            raise RuntimeError(f"Only lenient mode is supported for editable install. Current mode is {self.mode}")

        return TopLevelFinderWithRedirector(self.distribution, name)


################################################################################

setup(
    ext_modules=build_hooks._extensions,
    cmdclass={
        "build_ext": build_ext,
        "build_py": build_py_with_redirector,
        "editable_wheel": editable_wheel_with_redirector,
    },
    zip_safe=False,
)
