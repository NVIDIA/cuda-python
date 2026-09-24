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
from pathlib import Path
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

# The generated sources declare the types and functions of one CUDA header set,
# and cydriver.pxd records which. The install docs state the build rule that results.
_CYDRIVER_PXD = Path(__file__).resolve().parent / "cuda" / "bindings" / "cydriver.pxd"
_GENERATED_VERSION_RE = re.compile(r"^cdef enum:\s*CUDA_VERSION\s*=\s*(\d+)\s*$")
_CUDA_H_VERSION_RE = re.compile(r"^#\s*define\s+CUDA_VERSION\s+(\d+)\s*$")
_INSTALL_URL = "https://nvidia.github.io/cuda-python/cuda-bindings/latest/install.html#installing-from-source"


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


# -----------------------------------------------------------------------
# CUDA header check


def _cuda_h_path(cuda_path: str) -> str:
    """The cuda.h under cuda_path, with symlinks such as /usr/local/cuda resolved for messages."""
    return os.path.realpath(os.path.join(cuda_path, "include", "cuda.h"))


def _read_version_macro(path: str, pattern: re.Pattern) -> int | None:
    """The integer on the first line of ``path`` that matches ``pattern``, or None if no line does."""
    with open(path, encoding="utf-8") as f:
        for line in f:
            m = pattern.match(line)
            if m:
                return int(m.group(1))
    return None


def _read_cuda_h_version(cuda_path: str) -> int:
    """The CUDA_VERSION macro of the cuda.h under cuda_path, for example 13040 for 13.4."""
    cuda_h = _cuda_h_path(cuda_path)
    try:
        version = _read_version_macro(cuda_h, _CUDA_H_VERSION_RE)
    except OSError:
        version = None
    if version is None:
        raise RuntimeError(
            f"Cannot read CUDA_VERSION from {cuda_h}. "
            "Ensure CUDA_PATH or CUDA_HOME points to a CUDA Toolkit with include/cuda.h."
        )
    return version


def _generated_cuda_version() -> int:
    """The CUDA_VERSION of the headers that this source tree was generated from.

    Read from cuda/bindings/cydriver.pxd.
    """
    version = _read_version_macro(str(_CYDRIVER_PXD), _GENERATED_VERSION_RE)
    if version is None:
        raise RuntimeError(f"Cannot read CUDA_VERSION from {_CYDRIVER_PXD}")
    return version


def _major_minor(cuda_version: int) -> str:
    """13040 -> \"13.4\"."""
    return f"{cuda_version // 1000}.{cuda_version // 10 % 100}"


def _check_cuda_headers(cuda_path: str) -> None:
    """Reject a toolkit whose cuda.h is not the major.minor that this source tree was generated from.

    Against another minor, the C++ compile fails with a long list of
    redefinition and undeclared-type errors that do not name the cause. See
    https://github.com/NVIDIA/cuda-python/issues/2783. Runs before cythonize,
    which is the first step that touches the source tree.
    """
    generated = _generated_cuda_version()
    needed, found = _major_minor(generated), _major_minor(_read_cuda_h_version(cuda_path))
    if found != needed:
        raise RuntimeError(
            f"This cuda-bindings source tree needs CUDA {needed} headers, but {_cuda_h_path(cuda_path)} is "
            f"CUDA {found}. This is a build-time requirement only: at run time cuda-bindings supports any "
            f"CUDA {generated // 1000}.x toolkit, see {_INSTALL_URL}. Point CUDA_PATH or CUDA_HOME at a "
            f"CUDA {needed} toolkit, or build from cuda-bindings {found}.x sources."
        )


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
    from Cython.Compiler import Options as _CythonOptions

    global _extensions

    cuda_path = _get_cuda_path()
    _check_cuda_headers(cuda_path)

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
            extra_compile_args += ["-g0", "-O3"]
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
    _CythonOptions.warning_errors = True
    cython_directives = {"language_level": 3, "embedsignature": True, "binding": True, "freethreading_compatible": True}
    if compile_for_coverage:
        cython_directives["linetrace"] = True

    _extensions = cythonize(
        extensions,
        nthreads=nthreads,
        build_dir="." if compile_for_coverage else "build/cython",
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
