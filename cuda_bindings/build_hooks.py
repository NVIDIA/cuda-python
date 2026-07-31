# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# This module implements basic PEP 517 backend support to defer CUDA-dependent
# logic (cythonization) to build time. See:
# - https://peps.python.org/pep-0517/
# - https://setuptools.pypa.io/en/latest/build_meta.html#dynamic-build-dependencies-and-other-build-meta-tweaks
# - https://github.com/NVIDIA/cuda-python/issues/1635

import atexit
import contextlib
import functools
import glob
import os
import re
import shutil
import sys
import sysconfig
import tempfile
from warnings import warn

from setuptools import build_meta as _build_meta
from setuptools.extension import Extension

# Metadata hooks delegate directly to setuptools -- no CUDA needed.
prepare_metadata_for_build_editable = _build_meta.prepare_metadata_for_build_editable
prepare_metadata_for_build_wheel = _build_meta.prepare_metadata_for_build_wheel
build_sdist = _build_meta.build_sdist
get_requires_for_build_sdist = _build_meta.get_requires_for_build_sdist
get_requires_for_build_wheel = _build_meta.get_requires_for_build_wheel
get_requires_for_build_editable = _build_meta.get_requires_for_build_editable

# Populated by _build_cuda_bindings(); consumed by setup.py.
_extensions = None


# Please keep in sync with the copy in cuda_core/build_hooks.py.
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
            sp_cuda = os.path.join(p, "cuda")
            if os.path.isdir(os.path.join(sp_cuda, "pathfinder")):
                cuda.__path__ = list(cuda.__path__) + [sp_cuda]
                break
        else:
            raise ModuleNotFoundError(
                "cuda-pathfinder is not installed in the build environment. "
                "Ensure 'cuda-pathfinder>=1.5' is in build-system.requires."
            )
        import cuda.pathfinder

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
def _determine_cuda_major_version() -> str:
    """Determine the CUDA major version the extensions are compiled against.

    Derived from the CUDA_VERSION macro in cuda.h under CUDA_PATH or CUDA_HOME,
    which is the toolkit whose headers the build actually uses. Unlike
    cuda.core, cuda.bindings has no compile-time CUDA-major conditionals, so
    this is used only to key build artifacts -- see _resolve_build_identity().
    """
    cuda_h = os.path.join(_get_cuda_path(), "include", "cuda.h")
    try:
        with open(cuda_h, encoding="utf-8") as f:
            for line in f:
                m = re.match(r"^#\s*define\s+CUDA_VERSION\s+(\d+)\s*$", line)
                if m:
                    # CUDA_VERSION is e.g. 12020 for 12.2.
                    cuda_major = str(int(m.group(1)) // 1000)
                    print("CUDA MAJOR VERSION:", cuda_major)
                    return cuda_major
    except OSError:
        pass

    # CUDA_PATH or CUDA_HOME is required for the build, so we should not reach
    # here in normal circumstances. Raise an error to make the issue clear.
    raise RuntimeError(
        "Cannot determine CUDA major version. "
        "Ensure CUDA_PATH or CUDA_HOME points to a valid CUDA installation with include/cuda.h."
    )


# -----------------------------------------------------------------------
# Build-configuration identity
#
# Please keep this whole section in sync with the copy in cuda_core/build_hooks.py.

# Set by _build_cuda_bindings(), read by setup.py.
_build_identity = None
force_build_ext = False

_BUILD_IDENTITY_STAMP = os.path.join("build", ".build-identity")


def _resolve_build_identity(debug: bool) -> str:
    """Compute this build's identity and decide whether a rebuild is forced.

    The identity covers the axes that change what cythonize and the compiler
    *produce*, but which neither tool records in its own up-to-date check:

    - CUDA major: selects the CUDA headers the generated C++ is compiled
      against (and, in cuda.core, the Cython ``compile_time_env``).
    - debug: toggles ``gdb_debug``, which changes the cythonize output.
    - coverage: toggles ``linetrace``, which changes the generated C++.

    Python version and platform are deliberately absent: setuptools already
    encodes them in its own ``build/lib.*`` and ``build/temp.*`` directory
    names, so repeating them here would only lengthen paths.

    Keying the generated-source directory by this identity is not sufficient on
    its own. In an editable install the compiled extension lands in the source
    tree under a name keyed by the Python ABI tag alone -- there is no place to
    put the CUDA major -- so on a cu12 -> cu13 -> cu12 round trip build_ext
    would find the (older) cu12 generated source next to the (newer) cu13 .so
    and conclude everything is fresh. Forcing build_ext whenever the identity
    changes is what makes a cu13 output impossible to mistake for a cu12 one.
    """
    global _build_identity, force_build_ext

    identity = f"cu{_determine_cuda_major_version()}"
    if debug:
        identity += "-debug"
    if bool(int(os.environ.get("CUDA_PYTHON_COVERAGE", "0"))):
        identity += "-coverage"

    try:
        with open(_BUILD_IDENTITY_STAMP, encoding="utf-8") as f:
            previous = f.read().strip()
    except FileNotFoundError:
        previous = None

    if previous is not None and previous != identity:
        print(f"Build configuration changed ({previous} -> {identity}); forcing a full rebuild")
        force_build_ext = True

    _build_identity = identity
    return identity


def record_build_identity() -> None:
    """Persist the identity of the build that just completed.

    setup.py calls this after build_ext succeeds, so that a build which failed
    partway through does not claim outputs it never produced.
    """
    if _build_identity is None:
        return
    os.makedirs(os.path.dirname(_BUILD_IDENTITY_STAMP), exist_ok=True)
    with open(_BUILD_IDENTITY_STAMP, "w", encoding="utf-8") as f:
        f.write(_build_identity + "\n")


# -----------------------------------------------------------------------
# Extension preparation helpers


def _rename_architecture_specific_files():
    path = os.path.join("cuda", "bindings", "_internal")
    if sys.platform == "linux":
        src_files = glob.glob(os.path.join(path, "*_linux.pyx"))
    elif sys.platform == "win32":
        src_files = glob.glob(os.path.join(path, "*_windows.pyx"))
    else:
        raise RuntimeError(f"platform is unrecognized: {sys.platform}")
    dst_files = []
    for src in src_files:
        with tempfile.NamedTemporaryFile(delete=False, dir=".") as f:
            shutil.copy2(src, f.name)
            f_name = f.name
        dst = src.replace("_linux", "").replace("_windows", "")
        os.replace(f_name, f"./{dst}")
        dst_files.append(dst)
    return dst_files


def _prep_extensions(sources, libraries, include_dirs, library_dirs, extra_compile_args, extra_link_args):
    pattern = sources[0]
    files = glob.glob(pattern)
    libraries = libraries if libraries else []
    exts = []
    for pyx in files:
        mod_name = pyx.replace(".pyx", "").replace(os.sep, ".").replace("/", ".")
        exts.append(
            Extension(
                mod_name,
                sources=[pyx, *sources[1:]],
                include_dirs=include_dirs,
                library_dirs=library_dirs,
                runtime_library_dirs=[],
                libraries=libraries,
                language="c++",
                extra_compile_args=extra_compile_args,
                extra_link_args=extra_link_args,
            )
        )
    return exts


# -----------------------------------------------------------------------
# Main build function


def _build_cuda_bindings(debug=False):
    """Build all cuda-bindings extensions.

    All CUDA-dependent logic (cythonization) is deferred to this function so
    that metadata queries do not require a CUDA toolkit installation.
    """
    from Cython.Build import cythonize

    global _extensions

    cuda_path = _get_cuda_path()

    if os.environ.get("PARALLEL_LEVEL") is not None:
        warn(
            "Environment variable PARALLEL_LEVEL is deprecated. Use CUDA_PYTHON_PARALLEL_LEVEL instead",
            DeprecationWarning,
            stacklevel=2,
        )
        nthreads = int(os.environ.get("PARALLEL_LEVEL", "0"))
    else:
        nthreads = int(os.environ.get("CUDA_PYTHON_PARALLEL_LEVEL", "0") or "0")

    compile_for_coverage = bool(int(os.environ.get("CUDA_PYTHON_COVERAGE", "0")))
    build_identity = _resolve_build_identity(debug)

    # Prepare compile/link arguments
    include_path_list = [os.path.join(cuda_path, "include")]
    include_dirs = [
        os.path.dirname(sysconfig.get_path("include")),
    ] + include_path_list
    library_dirs = [sysconfig.get_path("platlib"), os.path.join(os.sys.prefix, "lib")]
    if sys.platform == "win32":
        cudalib_subdirs = [r"lib\arm64"] if sysconfig.get_platform() == "win-arm64" else [r"lib\x64"]
    else:
        cudalib_subdirs = ["lib64", "lib"]
    library_dirs.extend(os.path.join(cuda_path, subdir) for subdir in cudalib_subdirs)

    extra_compile_args = []
    extra_link_args = []
    extra_cythonize_kwargs = {}
    if sys.platform == "win32":
        if debug:
            raise RuntimeError("Debuggable builds are not supported on Windows.")
    else:
        extra_compile_args += [
            "-std=c++14",
            "-fpermissive",
            "-Wno-deprecated-declarations",
            "-fno-var-tracking-assignments",
        ]
        if debug:
            extra_cythonize_kwargs["gdb_debug"] = True
            extra_compile_args += ["-g", "-O0"]
            extra_compile_args += ["-D _GLIBCXX_ASSERTIONS"]
        else:
            extra_compile_args += ["-O3"]
            extra_link_args += ["-Wl,--strip-all"]
    if compile_for_coverage:
        # CYTHON_TRACE_NOGIL indicates to trace nogil functions.  It is not
        # related to free-threading builds.
        extra_compile_args += ["-DCYTHON_TRACE_NOGIL=1", "-DCYTHON_USE_SYS_MONITORING=0"]

    # Rename architecture-specific files
    dst_files = _rename_architecture_specific_files()

    @atexit.register
    def _cleanup_dst_files():
        for dst in dst_files:
            with contextlib.suppress(FileNotFoundError):
                os.remove(dst)

    # Build extension list
    extensions = []
    cuda_bindings_files = glob.glob("cuda/bindings/*.pyx") + glob.glob("cuda/bindings/_v2/*.pyx")
    if sys.platform == "win32":
        cuda_bindings_files = [f for f in cuda_bindings_files if "cufile" not in f]

    def get_static_libraries(f):
        if os.path.basename(f) in ("runtime.pyx", "runtime_ptds.pyx"):
            if sys.platform == "linux":
                return ["cudart_static", "rt"]
            else:
                return ["cudart_static"]
        return None

    sources_list = [
        # utils
        (["cuda/bindings/utils/*.pyx"], None),
        # public
        *(([f], None) for f in cuda_bindings_files),
        # internal files used by generated bindings
        (["cuda/bindings/_internal/utils.pyx"], None),
        *(([f], get_static_libraries(f)) for f in dst_files if f.endswith(".pyx")),
    ]

    for sources, libraries in sources_list:
        extensions += _prep_extensions(
            sources, libraries, include_dirs, library_dirs, extra_compile_args, extra_link_args
        )

    # Cythonize
    cython_directives = {"language_level": 3, "embedsignature": True, "binding": True, "freethreading_compatible": True}
    if compile_for_coverage:
        cython_directives["linetrace"] = True

    _extensions = cythonize(
        extensions,
        nthreads=nthreads,
        # CUDA_PYTHON_COVERAGE deliberately generates in-tree so the sources can
        # be packaged; every other build gets its own per-configuration cache.
        build_dir="." if compile_for_coverage else f"build/cython/{build_identity}",
        compiler_directives=cython_directives,
        **extra_cythonize_kwargs,
    )


# -----------------------------------------------------------------------
# PEP 517 build hooks


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    debug = config_settings.get("debug", False) if config_settings else False
    _build_cuda_bindings(debug=debug)
    return _build_meta.build_wheel(wheel_directory, config_settings, metadata_directory)


def build_editable(wheel_directory, config_settings=None, metadata_directory=None):
    debug_default = sys.platform != "win32"  # Debug builds not supported on Windows
    debug = config_settings.get("debug", debug_default) if config_settings else debug_default
    _build_cuda_bindings(debug=debug)
    return _build_meta.build_editable(wheel_directory, config_settings, metadata_directory)
