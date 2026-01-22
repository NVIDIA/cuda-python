# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# This module implements basic PEP 517 backend support, see e.g.
# - https://peps.python.org/pep-0517/
# - https://setuptools.pypa.io/en/latest/build_meta.html#dynamic-build-dependencies-and-other-build-meta-tweaks
# Specifically, there are 5 APIs required to create a proper build backend, see below.

import functools
import glob
import os
import re

from Cython.Build import cythonize
from setuptools import Extension
from setuptools import build_meta as _build_meta

# Import centralized CUDA environment variable handling
# Note: This import may fail at build-dependency-resolution time if cuda-pathfinder
# is not yet installed, but it's guaranteed to be available when _get_cuda_path()
# is actually called (during wheel build time).
try:
    from cuda.pathfinder._utils.env_vars import get_cuda_home_or_path
except ImportError as e:
    raise ImportError(
        "Failed to import cuda.pathfinder. "
        "Please ensure cuda-pathfinder is installed as a build dependency. "
        "If building cuda-core, cuda-pathfinder should be automatically installed. "
        "If this error persists, try: pip install cuda-pathfinder"
    ) from e

prepare_metadata_for_build_editable = _build_meta.prepare_metadata_for_build_editable
prepare_metadata_for_build_wheel = _build_meta.prepare_metadata_for_build_wheel
build_sdist = _build_meta.build_sdist
get_requires_for_build_sdist = _build_meta.get_requires_for_build_sdist

COMPILE_FOR_COVERAGE = bool(int(os.environ.get("CUDA_PYTHON_COVERAGE", "0")))


@functools.cache
def _get_cuda_paths() -> list[str]:
    """Get list of CUDA Toolkit paths from environment variables.
    
    Supports multiple paths separated by os.pathsep (: on Unix, ; on Windows).
    Returns a list of paths for use in include_dirs and library_dirs.
    """
    CUDA_PATH = get_cuda_home_or_path()
    if not CUDA_PATH:
        raise RuntimeError("Environment variable CUDA_PATH or CUDA_HOME is not set")
    CUDA_PATH = CUDA_PATH.split(os.pathsep)
    print("CUDA paths:", CUDA_PATH)
    return CUDA_PATH


@functools.cache
def _determine_cuda_major_version() -> str:
    """Determine the CUDA major version for building cuda.core.

    This version is used for two purposes:
    1. Determining which cuda-bindings version to install as a build dependency
    2. Setting CUDA_CORE_BUILD_MAJOR for Cython compile-time conditionals

    The version is derived from (in order of priority):
    1. CUDA_CORE_BUILD_MAJOR environment variable (explicit override, e.g. in CI)
    2. CUDA_VERSION macro in cuda.h from CUDA_PATH or CUDA_HOME

    Since CUDA_PATH or CUDA_HOME is required for the build (to provide include
    directories), the cuda.h header should always be available.
    """
    # Explicit override, e.g. in CI.
    cuda_major = os.environ.get("CUDA_CORE_BUILD_MAJOR")
    if cuda_major is not None:
        print("CUDA MAJOR VERSION:", cuda_major)
        return cuda_major

    # Derive from the CUDA headers (the authoritative source for what we compile against).
    cuda_path = _get_cuda_paths()
    for root in cuda_path:
        cuda_h = os.path.join(root, "include", "cuda.h")
        try:
            with open(cuda_h, encoding="utf-8") as f:
                for line in f:
                    m = re.match(r"^#\s*define\s+CUDA_VERSION\s+(\d+)\s*$", line)
                    if m:
                        v = int(m.group(1))
                        # CUDA_VERSION is e.g. 12020 for 12.2.
                        cuda_major = str(v // 1000)
                        print("CUDA MAJOR VERSION:", cuda_major)
                        return cuda_major
        except OSError:
            continue

    # CUDA_PATH or CUDA_HOME is required for the build, so we should not reach here
    # in normal circumstances. Raise an error to make the issue clear.
    raise RuntimeError(
        "Cannot determine CUDA major version. "
        "Set CUDA_CORE_BUILD_MAJOR environment variable, or ensure CUDA_PATH or CUDA_HOME "
        "points to a valid CUDA installation with include/cuda.h."
    )


# used later by setup()
_extensions = None


def _build_cuda_core():
    # Customizing the build hooks is needed because we must defer cythonization until cuda-bindings,
    # now a required build-time dependency that's dynamically installed via the other hook below,
    # is installed. Otherwise, cimport any cuda.bindings modules would fail!
    #
    # This function populates "_extensions".
    global _extensions

    # It seems setuptools' wildcard support has problems for namespace packages,
    # so we explicitly spell out all Extension instances.
    def module_names():
        root_path = os.path.sep.join(["cuda", "core", ""])
        for filename in glob.glob(f"{root_path}/**/*.pyx", recursive=True):
            yield filename[len(root_path) : -4]

    def get_sources(mod_name):
        """Get source files for a module, including any .cpp files."""
        sources = [f"cuda/core/{mod_name}.pyx"]

        # Add module-specific .cpp file from _cpp/ directory if it exists
        # Example: _resource_handles.pyx finds _cpp/resource_handles.cpp.
        cpp_file = f"cuda/core/_cpp/{mod_name.lstrip('_')}.cpp"
        if os.path.exists(cpp_file):
            sources.append(cpp_file)

        return sources

    all_include_dirs = list(os.path.join(root, "include") for root in _get_cuda_paths())
    extra_compile_args = []
    if COMPILE_FOR_COVERAGE:
        # CYTHON_TRACE_NOGIL indicates to trace nogil functions.  It is not
        # related to free-threading builds.
        extra_compile_args += ["-DCYTHON_TRACE_NOGIL=1", "-DCYTHON_USE_SYS_MONITORING=0"]

    ext_modules = tuple(
        Extension(
            f"cuda.core.{mod.replace(os.path.sep, '.')}",
            sources=get_sources(mod),
            include_dirs=[
                "cuda/core/_include",
                "cuda/core/_cpp",
            ]
            + all_include_dirs,
            language="c++",
            extra_compile_args=extra_compile_args,
        )
        for mod in module_names()
    )

    nthreads = int(os.environ.get("CUDA_PYTHON_PARALLEL_LEVEL", os.cpu_count() // 2))
    compile_time_env = {"CUDA_CORE_BUILD_MAJOR": int(_determine_cuda_major_version())}
    compiler_directives = {"embedsignature": True, "warn.deprecated.IF": False, "freethreading_compatible": True}
    if COMPILE_FOR_COVERAGE:
        compiler_directives["linetrace"] = True
    _extensions = cythonize(
        ext_modules,
        verbose=True,
        language_level=3,
        nthreads=nthreads,
        compiler_directives=compiler_directives,
        compile_time_env=compile_time_env,
    )

    return


def build_editable(wheel_directory, config_settings=None, metadata_directory=None):
    _build_cuda_core()
    return _build_meta.build_editable(wheel_directory, config_settings, metadata_directory)


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    _build_cuda_core()
    return _build_meta.build_wheel(wheel_directory, config_settings, metadata_directory)


def _get_cuda_bindings_require():
    cuda_major = _determine_cuda_major_version()
    return [f"cuda-bindings=={cuda_major}.*"]


def get_requires_for_build_editable(config_settings=None):
    return _build_meta.get_requires_for_build_editable(config_settings) + _get_cuda_bindings_require()


def get_requires_for_build_wheel(config_settings=None):
    return _build_meta.get_requires_for_build_wheel(config_settings) + _get_cuda_bindings_require()
