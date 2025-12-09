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
import subprocess

from Cython.Build import cythonize
from setuptools import Extension
from setuptools import build_meta as _build_meta

prepare_metadata_for_build_editable = _build_meta.prepare_metadata_for_build_editable
prepare_metadata_for_build_wheel = _build_meta.prepare_metadata_for_build_wheel
build_sdist = _build_meta.build_sdist
get_requires_for_build_sdist = _build_meta.get_requires_for_build_sdist


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


def _build_cuda_core():
    # Customizing the build hooks is needed because we must defer cythonization until cuda-bindings,
    # now a required build-time dependency that's dynamically installed via the other hook below,
    # is installed. Otherwise, cimport any cuda.bindings modules would fail!
    #
    # This function populates "_extensions".
    global _extensions

    # It seems setuptools' wildcard support has problems for namespace packages,
    # so we explicitly spell out all Extension instances.
    root_module = "cuda.core.experimental"
    root_path = f"{os.path.sep}".join(root_module.split(".")) + os.path.sep
    ext_files = glob.glob(f"{root_path}/**/*.pyx", recursive=True)

    def strip_prefix_suffix(filename):
        return filename[len(root_path) : -4]

    module_names = (strip_prefix_suffix(f) for f in ext_files)

    @functools.cache
    def get_cuda_paths():
        CUDA_PATH = os.environ.get("CUDA_PATH", os.environ.get("CUDA_HOME", None))
        if not CUDA_PATH:
            raise RuntimeError("Environment variable CUDA_PATH or CUDA_HOME is not set")
        CUDA_PATH = CUDA_PATH.split(os.pathsep)
        print("CUDA paths:", CUDA_PATH)
        return CUDA_PATH

    @functools.cache
    def get_cuda_library_dirs():
        """Return library search paths for CUDA driver runtime."""

        libdirs = []
        for root in get_cuda_paths():
            for subdir in ("lib64", "lib"):
                candidate = os.path.join(root, subdir)
                if os.path.isdir(candidate):
                    libdirs.append(candidate)
        return libdirs

    def get_sources(mod_name):
        """Get source files for a module, including any .cpp files."""
        sources = [f"cuda/core/experimental/{mod_name}.pyx"]

        # Add module-specific .cpp file from _cpp/ directory if it exists
        cpp_file = f"cuda/core/experimental/_cpp/{mod_name.lstrip('_')}.cpp"
        if os.path.exists(cpp_file):
            sources.append(cpp_file)

        return sources

    def get_extension_kwargs(mod_name):
        """Return Extension kwargs (libraries, library_dirs) per module."""

        # Modules that use CUDA driver APIs need to link against libcuda
        # _resource_handles: contains the C++ implementation that calls CUDA driver
        # _context, _stream, _event, _device: use resource handles and may call CUDA driver directly
        cuda_users = {"_resource_handles", "_context", "_stream", "_event", "_device"}
        kwargs = {}
        if mod_name in cuda_users:
            kwargs["libraries"] = ["cuda"]
            kwargs["library_dirs"] = get_cuda_library_dirs()
        return kwargs

    ext_modules = tuple(
        Extension(
            f"cuda.core.experimental.{mod.replace(os.path.sep, '.')}",
            sources=get_sources(mod),
            include_dirs=[
                "cuda/core/experimental/include",
                "cuda/core/experimental/_cpp",
            ]
            + list(os.path.join(root, "include") for root in get_cuda_paths()),
            language="c++",
            **get_extension_kwargs(mod),
        )
        for mod in module_names
    )

    nthreads = int(os.environ.get("CUDA_PYTHON_PARALLEL_LEVEL", os.cpu_count() // 2))
    compile_time_env = {"CUDA_CORE_BUILD_MAJOR": _get_proper_cuda_bindings_major_version()}
    _extensions = cythonize(
        ext_modules,
        verbose=True,
        language_level=3,
        nthreads=nthreads,
        compiler_directives={"embedsignature": True, "warn.deprecated.IF": False, "freethreading_compatible": True},
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
    cuda_major = _get_proper_cuda_bindings_major_version()
    return [f"cuda-bindings=={cuda_major}.*"]


def get_requires_for_build_editable(config_settings=None):
    return _build_meta.get_requires_for_build_editable(config_settings) + _get_cuda_bindings_require()


def get_requires_for_build_wheel(config_settings=None):
    return _build_meta.get_requires_for_build_wheel(config_settings) + _get_cuda_bindings_require()
