# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import sys
import tempfile
import zipfile
from pathlib import Path

from Cython.Build import cythonize
from Cython.Compiler import Options as _CythonOptions
from setuptools import Extension
from setuptools import build_meta as _build_meta

prepare_metadata_for_build_editable = _build_meta.prepare_metadata_for_build_editable
prepare_metadata_for_build_wheel = _build_meta.prepare_metadata_for_build_wheel
build_sdist = _build_meta.build_sdist
get_requires_for_build_sdist = _build_meta.get_requires_for_build_sdist

COMPILE_FOR_COVERAGE = bool(int(os.environ.get("CUDA_PYTHON_COVERAGE", "0")))


# Please keep in sync with the copy in cuda_bindings/build_hooks.py.
def _import_get_cuda_path_or_home():
    """Import get_cuda_path_or_home, working around PEP 517 namespace shadowing.

    See https://github.com/NVIDIA/cuda-python/issues/1824 for why this helper is needed.
    """
    try:
        import cuda.pathfinder
    except ModuleNotFoundError as exc:
        if exc.name not in ("cuda", "cuda.pathfinder"):
            raise
        try:
            import cuda
        except ModuleNotFoundError:
            cuda = None

        for p in sys.path:
            sp_cuda = Path(p) / "cuda"
            if (sp_cuda / "pathfinder").is_dir():
                cuda.__path__ = list(cuda.__path__) + [str(sp_cuda)]
                break
        else:
            raise ModuleNotFoundError(
                "cuda-pathfinder is not installed in the build environment. "
                "Ensure 'cuda-pathfinder>=1.5' is in build-system.requires."
            )
        import cuda.pathfinder

    pathfinder_dir = Path(cuda.pathfinder.__file__).parent
    print(
        f"Using cuda-pathfinder {cuda.pathfinder.__version__} from {pathfinder_dir}",
        file=sys.stderr,
    )
    return cuda.pathfinder.get_cuda_path_or_home


@functools.cache
def _get_cuda_path() -> str:
    get_cuda_path_or_home = _import_get_cuda_path_or_home()
    cuda_path = get_cuda_path_or_home()
    if not cuda_path:
        raise RuntimeError("Environment variable CUDA_PATH or CUDA_HOME is not set")
    print("CUDA path:", cuda_path)
    return cuda_path


@functools.cache
def _determine_cuda_version() -> tuple[int, int]:
    """Determine the CUDA major and minor version for building cuda.core.

    This version is used for two purposes:
    1. Determining which cuda-bindings version to install as a build dependency
    2. Setting CUDA_CORE_BUILD_MAJOR and CUDA_CORE_BUILD_MINOR for Cython
       compile-time conditionals

    The version is derived from (in order of priority):
    1. CUDA_CORE_BUILD_MAJOR (and optionally CUDA_CORE_BUILD_MINOR) environment
       variables (explicit override, e.g. in CI)
    2. CUDA_VERSION macro in cuda.h from CUDA_PATH or CUDA_HOME

    Since CUDA_PATH or CUDA_HOME is required for the build (to provide include
    directories), the cuda.h header should always be available.
    """
    # Explicit override, e.g. in CI.
    cuda_major = os.environ.get("CUDA_CORE_BUILD_MAJOR")
    if cuda_major is not None:
        cuda_minor = int(os.environ.get("CUDA_CORE_BUILD_MINOR", "0"))
        print(f"CUDA VERSION: {cuda_major}.{cuda_minor}")
        return int(cuda_major), cuda_minor

    # Derive from the CUDA headers (the authoritative source for what we compile against).
    cuda_path = _get_cuda_path()
    cuda_h = os.path.join(cuda_path, "include", "cuda.h")
    try:
        with open(cuda_h, encoding="utf-8") as f:
            for line in f:
                m = re.match(r"^#\s*define\s+CUDA_VERSION\s+(\d+)\s*$", line)
                if m:
                    v = int(m.group(1))
                    # CUDA_VERSION is e.g. 12020 for 12.2, 13010 for 13.1.
                    major = v // 1000
                    minor = (v % 1000) // 10
                    print(f"CUDA VERSION: {major}.{minor}")
                    return major, minor
    except OSError:
        pass

    # CUDA_PATH or CUDA_HOME is required for the build, so we should not reach here
    # in normal circumstances. Raise an error to make the issue clear.
    raise RuntimeError(
        "Cannot determine CUDA major version. "
        "Set CUDA_CORE_BUILD_MAJOR environment variable, or ensure CUDA_PATH or CUDA_HOME "
        "points to a valid CUDA installation with include/cuda.h."
    )


def _determine_cuda_major_version() -> str:
    """Return the CUDA major version as a string."""
    major, _ = _determine_cuda_version()
    return str(major)


# used later by setup()
_extensions = None

# Where per-configuration build artifacts live. Anchored to this file rather
# than the cwd, since a project can be built from anywhere.
_BUILD_DIR = Path(__file__).parent / "build"

# Records the CUDA major of the last completed build, so setup.py can force
# build_ext when it changes. Written by record_build_major().
_BUILD_MAJOR_STAMP = _BUILD_DIR / ".build-cuda-major"

force_build_ext = False


def _check_build_major() -> str:
    """Return the CUDA major to key build artifacts by, and set force_build_ext.

    Cython's up-to-date check does not hash ``compile_time_env``, so generated
    sources for one CUDA major would otherwise be reused for another. Keying
    the generated-source directory fixes that, but not the compiled extension:
    in an editable install it lands in the source tree under a name keyed by
    the Python ABI tag alone, with nowhere to record the CUDA major. On a
    cu12 -> cu13 -> cu12 round trip build_ext would find the older cu12
    generated source next to the newer cu13 .so and skip the rebuild, so the
    major is also stamped and build_ext forced whenever it changes.
    """
    global force_build_ext

    cuda_major = _determine_cuda_major_version()
    try:
        previous = _BUILD_MAJOR_STAMP.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        previous = None

    # A missing stamp means the last build's major is unknown, so force too.
    # On a first build that costs nothing: there are no artifacts to reuse.
    if previous != cuda_major:
        print(f"CUDA major of last build: {previous} (building {cuda_major}); forcing a full rebuild")
        force_build_ext = True

    return cuda_major


def record_build_major() -> None:
    """Stamp the CUDA major of the build that just completed.

    setup.py calls this after build_ext succeeds, so that a build which failed
    partway through does not claim outputs it never produced.
    """
    _BUILD_MAJOR_STAMP.parent.mkdir(parents=True, exist_ok=True)
    _BUILD_MAJOR_STAMP.write_text(_determine_cuda_major_version() + "\n", encoding="utf-8")


def _build_cuda_core(debug=False):
    # Customizing the build hooks is needed because we must defer cythonization until cuda-bindings,
    # now a required build-time dependency that's dynamically installed via the other hook below,
    # is installed. Otherwise, cimport any cuda.bindings modules would fail!
    #
    # This function populates "_extensions".
    global _extensions

    # Resolve CUDA first so the pathfinder import repairs PEP 517 namespace shadowing before importing bindings.
    cuda_path = _get_cuda_path()

    # Add cuda-bindings to sys.path so Cython can find .pxd files
    # This is needed for editable installs where meta path finders don't work for Cython
    # We need to add the directory containing the 'cuda' package so Cython can resolve
    # "from cuda.bindings cimport cydriver"
    try:
        import cuda.bindings

        bindings_path = Path(cuda.bindings.__file__).parent  # .../cuda/bindings/
        print(f"Using cuda-bindings {cuda.bindings.__version__} from {bindings_path}", file=sys.stderr)
        cuda_package_dir = bindings_path.parent.parent  # .../cuda_bindings/ (contains cuda/)
        if str(cuda_package_dir) not in sys.path:
            sys.path.insert(0, str(cuda_package_dir))
            print(f"Added cuda-bindings parent path for Cython: {cuda_package_dir}", file=sys.stderr)
    except ImportError:
        # cuda-bindings not available in editable mode, will use installed version
        pass

    _posix_only_modules = frozenset(
        {
            "_utils/_wsl_locale",
        }
    )

    # It seems setuptools' wildcard support has problems for namespace packages,
    # so we explicitly spell out all Extension instances.
    def module_names():
        root_path = os.path.sep.join(["cuda", "core", ""])
        for filename in glob.glob(f"{root_path}/**/*.pyx", recursive=True):
            mod = filename[len(root_path) : -4]
            if sys.platform == "win32" and mod.replace(os.path.sep, "/") in _posix_only_modules:
                continue
            yield mod

    def get_sources(mod_name):
        """Get source files for a module, including any .cpp files."""
        sources = [f"cuda/core/{mod_name}.pyx"]

        # Add module-specific .cpp file from _cpp/ directory if it exists
        # Example: _resource_handles.pyx finds _cpp/resource_handles.cpp.
        cpp_file = f"cuda/core/_cpp/{mod_name.lstrip('_')}.cpp"
        if os.path.exists(cpp_file):
            sources.append(cpp_file)

        return sources

    all_include_dirs = [os.path.join(cuda_path, "include")]
    extra_compile_args = []
    extra_link_args = []
    extra_cythonize_kwargs = {}
    if sys.platform == "win32":
        extra_compile_args += ["/std:c++17"]
        if debug:
            raise RuntimeError("Debuggable builds are not supported on Windows.")
    else:
        extra_compile_args += ["-std=c++17"]
        if debug:
            extra_cythonize_kwargs["gdb_debug"] = True
            extra_compile_args += ["-g", "-O0"]
            extra_compile_args += ["-D _GLIBCXX_ASSERTIONS"]
        else:
            extra_compile_args += ["-O2"]
            extra_link_args += ["-Wl,--strip-all"]
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
            extra_link_args=extra_link_args,
        )
        for mod in module_names()
    )

    # Deliberately after the cuda.bindings import above: this re-enters
    # _get_cuda_path() and reads cuda.h, which must not run before the
    # pathfinder import has repaired PEP 517 namespace shadowing.
    cuda_major = _check_build_major()

    nthreads = int(os.environ.get("CUDA_PYTHON_PARALLEL_LEVEL", os.cpu_count() // 2))
    _, cuda_minor = _determine_cuda_version()
    compile_time_env = {"CUDA_CORE_BUILD_MAJOR": int(cuda_major), "CUDA_CORE_BUILD_MINOR": cuda_minor}
    compiler_directives = {"embedsignature": True, "warn.deprecated.IF": False, "freethreading_compatible": True}
    _CythonOptions.warning_errors = True
    if COMPILE_FOR_COVERAGE:
        compiler_directives["linetrace"] = True
    _extensions = cythonize(
        ext_modules,
        verbose=True,
        language_level=3,
        # CUDA_PYTHON_COVERAGE deliberately generates in-tree so the sources can
        # be packaged; every other build gets its own per-configuration cache,
        # anchored alongside the stamp so both resolve the same from any cwd.
        build_dir="." if COMPILE_FOR_COVERAGE else str(_BUILD_DIR / "cython" / f"cu{cuda_major}"),
        nthreads=nthreads,
        compiler_directives=compiler_directives,
        compile_time_env=compile_time_env,
        **extra_cythonize_kwargs,
    )

    return


def _add_cython_include_paths_to_pth(wheel_path: str) -> None:
    """
    Modify the .pth file in an editable install wheel to add Cython include paths.

    This is needed because Cython cannot find .pxd files through meta path finders,
    it only looks in sys.path directories. By adding direct paths to the .pth file,
    we enable Cython to find .pxd files from editable-installed cuda-bindings.

    See: https://github.com/scikit-build/scikit-build-core/pull/516
    See: https://github.com/cython/cython/issues/7326
    """
    # Find cuda-bindings location
    # When building with pixi path dependencies, cuda-bindings should be importable
    try:
        import cuda.bindings

        bindings_path = Path(cuda.bindings.__file__).parent  # .../cuda/bindings/
        # We need the directory containing the 'cuda' package for Cython imports
        cuda_package_dir = bindings_path.parent.parent  # .../cuda_bindings/ (contains cuda/)
        print(f"Found cuda-bindings at: {bindings_path}", file=sys.stderr)
        print(f"Will add to .pth for Cython: {cuda_package_dir}", file=sys.stderr)
    except ImportError:
        # If cuda-bindings isn't available yet, we can't add the path
        # This might happen in some build scenarios, but it's okay - the
        # wildcard dependency will work in those cases
        print("cuda-bindings not found in current environment, skipping .pth modification")
        return

    # Create a temporary directory for wheel manipulation
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        wheel_file = Path(wheel_path)

        # Extract the wheel
        extract_dir = tmpdir_path / "extracted"
        with zipfile.ZipFile(wheel_file, "r") as zf:
            zf.extractall(extract_dir)

        # Find the .pth file (should be named something like __editable___cuda_core-*.pth)
        pth_files = list(extract_dir.glob("**/*.pth"))
        if not pth_files:
            print("Warning: No .pth file found in editable wheel", file=sys.stderr)
            return

        # Modify each .pth file (usually just one)
        for pth_file in pth_files:
            print(f"Modifying {pth_file.name} to add Cython include paths", file=sys.stderr)

            # Read existing content
            content = pth_file.read_text()

            # Add the cuda-bindings source path to sys.path for Cython
            # This allows Cython to find .pxd files via direct path lookup
            # The path must be the directory containing the 'cuda' package
            path_to_add = str(cuda_package_dir.absolute())

            # Ensure content ends with newline before adding path
            if not content.endswith("\n"):
                content += "\n"

            # Append to the .pth file (after the import hook line)
            if path_to_add not in content:
                pth_file.write_text(content + path_to_add + "\n")
                print(f"Added Cython include path: {cuda_package_dir}", file=sys.stderr)

        # Repackage the wheel
        # Remove the old wheel first
        wheel_file.unlink()

        # Create new wheel with same name
        with zipfile.ZipFile(wheel_file, "w", zipfile.ZIP_DEFLATED) as zf:
            for file_path in extract_dir.rglob("*"):
                if file_path.is_file():
                    arcname = file_path.relative_to(extract_dir)
                    zf.write(file_path, arcname)

        print(f"Successfully patched {wheel_file.name}", file=sys.stderr)


def build_editable(wheel_directory, config_settings=None, metadata_directory=None):
    debug_default = sys.platform != "win32"  # Debug builds not supported on Windows
    debug = config_settings.get("debug", debug_default) if config_settings else debug_default
    _build_cuda_core(debug=debug)
    wheel_name = _build_meta.build_editable(wheel_directory, config_settings, metadata_directory)

    # Patch the .pth file to add Cython include paths
    wheel_path = os.path.join(wheel_directory, wheel_name)
    _add_cython_include_paths_to_pth(wheel_path)

    return wheel_name


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    debug = config_settings.get("debug", False) if config_settings else False
    _build_cuda_core(debug=debug)
    return _build_meta.build_wheel(wheel_directory, config_settings, metadata_directory)


def _get_cuda_bindings_require():
    cuda_major = _determine_cuda_major_version()
    return [f"cuda-bindings=={cuda_major}.*"]


def get_requires_for_build_editable(config_settings=None):
    return _build_meta.get_requires_for_build_editable(config_settings) + _get_cuda_bindings_require()


def get_requires_for_build_wheel(config_settings=None):
    return _build_meta.get_requires_for_build_wheel(config_settings) + _get_cuda_bindings_require()
