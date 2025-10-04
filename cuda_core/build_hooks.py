# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# This module implements basic PEP 517 backend support, see e.g.
# - https://peps.python.org/pep-0517/
# - https://setuptools.pypa.io/en/latest/build_meta.html#dynamic-build-dependencies-and-other-build-meta-tweaks
# Specifically, there are 5 APIs required to create a proper build backend, see below.
#
# TODO: also implement PEP-660 API hooks

import functools
import glob
import os
import pathlib
import re
import subprocess

from Cython.Build import cythonize
from setuptools import Extension
from setuptools import build_meta as _build_meta

prepare_metadata_for_build_wheel = _build_meta.prepare_metadata_for_build_wheel
build_sdist = _build_meta.build_sdist
get_requires_for_build_sdist = _build_meta.get_requires_for_build_sdist


@functools.cache
def _get_cuda_paths():
    CUDA_PATH = os.environ.get("CUDA_PATH", os.environ.get("CUDA_HOME", None))
    if not CUDA_PATH:
        raise RuntimeError("Environment variable CUDA_PATH or CUDA_HOME is not set")
    CUDA_PATH = CUDA_PATH.split(os.pathsep)
    print("CUDA paths:", CUDA_PATH, flush=True)
    return CUDA_PATH


@functools.cache
def _get_cuda_version_from_cuda_h(cuda_home=None):
    """
    Given CUDA_HOME, try to extract the CUDA_VERSION macro from include/cuda.h.

    Example line in cuda.h:
        #define CUDA_VERSION 13000

    Returns the integer (e.g. 13000) or None if not found / on error.
    """
    if cuda_home is None:
        cuda_home = _get_cuda_paths()[0]

    cuda_h = pathlib.Path(cuda_home) / "include" / "cuda.h"
    if not cuda_h.is_file():
        return None

    try:
        text = cuda_h.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        # Permissions issue, unreadable file, etc.
        return None

    m = re.search(r"^\s*#define\s+CUDA_VERSION\s+(\d+)", text, re.MULTILINE)
    if not m:
        return None
    print(f"CUDA_VERSION from {cuda_h}:", m.group(1), flush=True)
    return m.group(1)


@functools.cache
def _get_proper_cuda_bindings_major_version() -> str:
    # for local development (with/without build isolation)
    try:
        import cuda.bindings

        return cuda.bindings.__version__.split(".")[0]
    except ImportError:
        pass

    # for custom overwrite, e.g. in CI
    cuda_major = os.environ.get("CUDA_CORE_BUILD_MAJOR")
    if cuda_major is not None:
        return cuda_major

    cuda_version = _get_cuda_version_from_cuda_h()
    if cuda_version and len(cuda_version) > 3:
        return cuda_version[:-3]

    # also for local development
    try:
        out = subprocess.run("nvidia-smi", env=os.environ, capture_output=True, check=True)  # noqa: S603, S607
        m = re.search(r"CUDA Version:\s*([\d\.]+)", out.stdout.decode())
        if m:
            return m.group(1).split(".")[0]
    except (FileNotFoundError, subprocess.CalledProcessError):
        # the build machine has no driver installed
        pass

    # default fallback
    return "13"


# used later by setup()
_extensions = None


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    # Customizing this hook is needed because we must defer cythonization until cuda-bindings,
    # now a required build-time dependency that's dynamically installed via the other hook below,
    # is installed. Otherwise, cimport any cuda.bindings modules would fail!

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
            include_dirs=list(os.path.join(root, "include") for root in _get_cuda_paths()),
            language="c++",
        )
        for mod in module_names
    )

    nthreads = int(os.environ.get("CUDA_PYTHON_PARALLEL_LEVEL", os.cpu_count() // 2))
    compile_time_env = {"CUDA_CORE_BUILD_MAJOR": _get_proper_cuda_bindings_major_version()}

    global _extensions
    _extensions = cythonize(
        ext_modules,
        verbose=True,
        language_level=3,
        nthreads=nthreads,
        compiler_directives={"embedsignature": True, "warn.deprecated.IF": False},
        compile_time_env=compile_time_env,
    )

    return _build_meta.build_wheel(wheel_directory, config_settings, metadata_directory)


def get_requires_for_build_wheel(config_settings=None):
    cuda_major = _get_proper_cuda_bindings_major_version()
    cuda_bindings_require = [f"cuda-bindings=={cuda_major}.*"]
    return _build_meta.get_requires_for_build_wheel(config_settings) + cuda_bindings_require
