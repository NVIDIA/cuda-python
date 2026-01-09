# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# This module implements basic PEP 517 backend support, see e.g.
# - https://peps.python.org/pep-0517/
# - https://setuptools.pypa.io/en/latest/build_meta.html#dynamic-build-dependencies-and-other-build-meta-tweaks
# Specifically, there are 5 APIs required to create a proper build backend, see below.

import functools
import glob
import importlib.metadata
import json
import os
import re
from pathlib import Path

from Cython.Build import cythonize
from setuptools import Extension
from setuptools import build_meta as _build_meta

prepare_metadata_for_build_editable = _build_meta.prepare_metadata_for_build_editable
prepare_metadata_for_build_wheel = _build_meta.prepare_metadata_for_build_wheel
build_sdist = _build_meta.build_sdist
get_requires_for_build_sdist = _build_meta.get_requires_for_build_sdist

COMPILE_FOR_COVERAGE = bool(int(os.environ.get("CUDA_PYTHON_COVERAGE", "0")))


@functools.cache
def _get_cuda_paths() -> list[str]:
    CUDA_PATH = os.environ.get("CUDA_PATH", os.environ.get("CUDA_HOME", None))
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
    # Validate git tags are available (fail early before setuptools-scm runs)
    import subprocess

    # Check if git is available
    try:
        subprocess.run(["git", "--version"], capture_output=True, check=True, timeout=5)  # noqa: S607
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        raise RuntimeError(
            "Git is not available in PATH. setuptools-scm requires git to determine version from tags.\n"
            "Please ensure git is installed and available in your PATH."
        ) from None

    # Find git repository root (setuptools_scm root='..')
    repo_root = os.path.dirname(os.path.dirname(__file__))

    # Check if git describe works (this is what setuptools-scm uses)
    try:
        result = subprocess.run(
            ["git", "describe", "--tags", "--long", "--match", "cuda-core-v*[0-9]*"],  # noqa: S607
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"git describe failed! This means setuptools-scm will fall back to version '0.1.x'.\n"
                f"\n"
                f"Error: {result.stderr.strip()}\n"
                f"\n"
                f"This usually means:\n"
                f"  1. Git tags are not fetched (run: git fetch --tags)\n"
                f"  2. Running from wrong directory (setuptools_scm root='..')\n"
                f"  3. No matching tags found\n"
                f"\n"
                f"To fix:\n"
                f"  git fetch --tags\n"
                f"\n"
                f"To debug, run: git describe --tags --long --match 'cuda-core-v*[0-9]*'"
            ) from None
    except subprocess.TimeoutExpired:
        raise RuntimeError(
            "git describe command timed out. This may indicate git repository issues.\n"
            "Please check your git repository state."
        ) from None

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

    all_include_dirs = list(os.path.join(root, "include") for root in _get_cuda_paths())
    extra_compile_args = []
    if COMPILE_FOR_COVERAGE:
        # CYTHON_TRACE_NOGIL indicates to trace nogil functions.  It is not
        # related to free-threading builds.
        extra_compile_args += ["-DCYTHON_TRACE_NOGIL=1", "-DCYTHON_USE_SYS_MONITORING=0"]

    ext_modules = tuple(
        Extension(
            f"cuda.core.{mod.replace(os.path.sep, '.')}",
            sources=[f"cuda/core/{mod}.pyx"],
            include_dirs=all_include_dirs,
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


def _check_cuda_bindings_installed():
    """Check if cuda-bindings is installed and validate its version.

    Returns:
        tuple: (is_installed: bool, is_editable: bool, major_version: str | None)
        If not installed, returns (False, False, None)
    """
    try:
        bindings_dist = importlib.metadata.distribution("cuda-bindings")
    except importlib.metadata.PackageNotFoundError:
        return (False, False, None)

    bindings_version = bindings_dist.version
    bindings_major_version = bindings_version.split(".")[0]

    # Check if it's an editable install
    is_editable = False

    # Method 1: Check direct_url.json (PEP 610 - modern editable installs)
    try:
        dist_location = Path(bindings_dist.locate_file(""))
        # dist-info or egg-info directory
        metadata_dir = dist_location.parent
        direct_url_path = metadata_dir / "direct_url.json"
        if direct_url_path.exists():
            with open(direct_url_path) as f:
                direct_url = json.load(f)
                if direct_url.get("dir_info", {}).get("editable"):
                    is_editable = True
    except Exception:  # noqa: S110
        pass

    # Method 2: Check if package is in current repository (another way to detect editable)
    if not is_editable:
        try:
            import cuda.bindings

            bindings_path = Path(cuda.bindings.__file__).resolve()
            repo_root = Path(__file__).resolve().parent.parent.parent
            if repo_root in bindings_path.parents:
                is_editable = True
        except Exception:  # noqa: S110
            pass

    # Method 3: Check for .egg-link file (old editable install method)
    if not is_editable:
        try:
            # Look for .egg-link files in site-packages
            import site

            for site_dir in site.getsitepackages():
                egg_link_path = Path(site_dir) / "cuda-bindings.egg-link"
                if egg_link_path.exists():
                    is_editable = True
                    break
        except Exception:  # noqa: S110
            pass

    return (True, is_editable, bindings_major_version)


def _get_cuda_bindings_require():
    """Determine cuda-bindings build requirement.

    Strategy:
    1. If cuda-bindings is already installed (any version), don't require it (keep existing)
    2. If installed from sources (editable), definitely keep it
    3. Otherwise, if installed version major doesn't match CUDA major, raise error
    4. If not installed, require matching CUDA major version
    """
    bindings_installed, bindings_editable, bindings_major = _check_cuda_bindings_installed()

    # If not installed, require matching CUDA major version
    if not bindings_installed:
        cuda_major = _determine_cuda_major_version()
        return [f"cuda-bindings=={cuda_major}.*"]

    # If installed from sources (editable), keep it (don't require anything)
    if bindings_editable:
        return []

    # If installed but not editable, check version matches CUDA major
    cuda_major = _determine_cuda_major_version()
    if bindings_major != cuda_major:
        raise RuntimeError(
            f"Installed cuda-bindings version has major version {bindings_major}, "
            f"but CUDA major version is {cuda_major}.\n"
            f"This mismatch could cause build or runtime errors.\n"
            f"\n"
            f"To fix:\n"
            f"  1. Uninstall cuda-bindings: pip uninstall cuda-bindings\n"
            f"  2. Or install from sources: pip install -e ./cuda_bindings\n"
            f"  3. Or install matching version: pip install 'cuda-bindings=={cuda_major}.*'"
        )

    # Installed and version matches (or is editable), keep it
    return []


def get_requires_for_build_editable(config_settings=None):
    return _build_meta.get_requires_for_build_editable(config_settings) + _get_cuda_bindings_require()


def get_requires_for_build_wheel(config_settings=None):
    return _build_meta.get_requires_for_build_wheel(config_settings) + _get_cuda_bindings_require()
