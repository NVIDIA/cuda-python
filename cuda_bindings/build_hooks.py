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
import shutil
import sys
import sysconfig
import tempfile
from pathlib import Path
from warnings import warn

from setuptools import build_meta as _build_meta
from setuptools.extension import Extension

# Valid toolchains per platform; the first entry is the default.
_TOOLCHAINS_LINUX = ("gnu", "llvm")
_TOOLCHAINS_WINDOWS = ("msvc",)

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
# Toolchain selection


def _resolve_toolchain(debug=False, compile_for_coverage=False):
    """Resolve the C/C++ toolchain from CUDA_PYTHON_TOOLCHAIN.

    Returns (name, cc, cxx, extra_compile_args, extra_link_args). The default
    toolchain (gnu on Linux, msvc on Windows) reproduces the previous build
    behavior and does not touch CC/CXX/LDSHARED, so an externally-set compiler
    (e.g. CC="sccache cc") keeps working. A non-default toolchain (llvm on
    Linux) selects clang/clang++ and lld and sets CC/CXX/LDSHARED so distutils'
    customize_compiler picks them up.
    """
    if sys.platform == "win32":
        platform_key = "win32"
        allowed = _TOOLCHAINS_WINDOWS
    else:
        platform_key = "linux"
        allowed = _TOOLCHAINS_LINUX
    name = os.environ.get("CUDA_PYTHON_TOOLCHAIN", allowed[0]).strip().lower()
    if name not in allowed:
        raise RuntimeError(
            f"CUDA_PYTHON_TOOLCHAIN={name!r} is not supported on {platform_key}. Valid values: {', '.join(allowed)}."
        )

    extra_compile_args = []
    extra_link_args = []
    cc = cxx = None

    if name == "msvc":
        if debug:
            raise RuntimeError("Debuggable builds are not supported on Windows.")
    elif name == "gnu":
        cc, cxx = "cc", "c++"
        extra_compile_args += [
            "-std=c++14",
            "-fpermissive",
            "-Wno-deprecated-declarations",
            "-fno-var-tracking-assignments",
        ]
        if debug:
            extra_compile_args += ["-g", "-O0", "-D _GLIBCXX_ASSERTIONS"]
        else:
            extra_compile_args += ["-g0", "-O3"]
            extra_link_args += ["-Wl,--strip-all"]
    elif name == "llvm":
        cc, cxx = "clang", "clang++"
        # clang rejects -fpermissive and -fno-var-tracking-assignments (gcc-only).
        extra_compile_args += ["-std=c++14", "-Wno-deprecated-declarations"]
        extra_link_args += ["-fuse-ld=lld"]
        if debug:
            extra_compile_args += ["-g", "-O0", "-D _GLIBCXX_ASSERTIONS"]
        else:
            extra_compile_args += ["-g0", "-O3"]
            extra_link_args += ["-Wl,--strip-all"]

    if compile_for_coverage:
        # CYTHON_TRACE_NOGIL indicates to trace nogil functions.  It is not
        # related to free-threading builds.
        extra_compile_args += ["-DCYTHON_TRACE_NOGIL=1", "-DCYTHON_USE_SYS_MONITORING=0"]

    # Only override the env for a non-default toolchain; the default path must
    # defer to any externally-set CC/CXX (e.g. the sccache wrapper in CI).
    if name != allowed[0] and cc is not None:
        os.environ["CC"] = cc
        os.environ["CXX"] = cxx
        os.environ["LDSHARED"] = f"{cxx} -shared"

    return name, cc, cxx, extra_compile_args, extra_link_args


def _check_toolchain_available(name):
    """Preflight: verify the selected toolchain's tools are on PATH.

    No-op for the platform default (distutils discovers those). For llvm,
    probes clang, clang++, and ld.lld so a missing toolchain fails fast with a
    helpful message instead of a cryptic compile error.
    """
    if name != "llvm":
        return
    tools = ("clang", "clang++", "ld.lld")
    missing = [t for t in tools if shutil.which(t) is None]
    if missing:
        raise RuntimeError(
            f"CUDA_PYTHON_TOOLCHAIN=llvm but required tool(s) not found on PATH: "
            f"{', '.join(missing)}. Install clang and lld "
            f"(e.g. `apt install clang lld` or `dnf install clang lld`) "
            f"or set CUDA_PYTHON_TOOLCHAIN=gnu."
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

    # Resolve the C/C++ toolchain (CUDA_PYTHON_TOOLCHAIN). The default (gnu on
    # Linux, msvc on Windows) reproduces the previous build behavior and does
    # not touch CC/CXX, so an externally-set compiler (e.g. sccache) survives.
    toolchain, _cc, _cxx, extra_compile_args, extra_link_args = _resolve_toolchain(
        debug=debug, compile_for_coverage=compile_for_coverage
    )
    _check_toolchain_available(toolchain)
    extra_cythonize_kwargs = {}
    if debug and sys.platform != "win32":
        extra_cythonize_kwargs["gdb_debug"] = True

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
