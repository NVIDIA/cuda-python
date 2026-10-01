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
import hashlib
import os
import re
import shutil
import sys
import sysconfig
import tempfile
import uuid
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

# Note: There is no support guarantee for environment variables like
# CUDA_PYTHON_TOOLCHAIN, CUDA_PYTHON_CYTHON_CACHE_DIR, etc. They may be
# removed or changed in the future.

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
            f"CUDA {found}. This is a build-time requirement only: at run time this cuda-bindings build can be "
            f"used with any CUDA {generated // 1000}.x toolkit, see {_INSTALL_URL}. Point CUDA_PATH or CUDA_HOME "
            f"at a CUDA {needed} toolkit, or build from cuda-bindings {found}.x sources."
        )


# -----------------------------------------------------------------------
# Toolchain selection
#
# There is one shared helper block below, duplicated verbatim in
# cuda_core/build_hooks.py (keep it in sync; enforced by
# toolshed/check_build_hooks_sync.py). It contains the toolchain helpers and
# the Cython cache helpers. Only the per-package _resolve_toolchain() flag
# assembly that follows the shared block is package-specific (it differs
# because the two packages use different C++ standards and opt levels).

# --- begin shared build helpers (keep in sync) ---
_TOOLCHAINS_LINUX = ("gnu", "llvm")
_TOOLCHAINS_WINDOWS = ("msvc",)
_TOOLCHAIN_COMPILERS = {
    "gnu": ("gcc", "g++"),
    "llvm": ("clang", "clang++"),
    "msvc": (None, None),
}


def _resolve_toolchain_name():
    """Read CUDA_PYTHON_TOOLCHAIN, validate it, return (name, allowed, cc, cxx).

    The default toolchain (gnu on Linux, msvc on Windows) is the first entry
    of the platform's allowed tuple. cc/cxx are the compiler binaries for the
    toolchain (None for msvc, which distutils discovers via the MSVC env).
    """
    if sys.platform == "win32":
        platform_key, allowed = "win32", _TOOLCHAINS_WINDOWS
    else:
        platform_key, allowed = "linux", _TOOLCHAINS_LINUX
    name = os.environ.get("CUDA_PYTHON_TOOLCHAIN", allowed[0]).strip().lower()
    if name not in allowed:
        raise RuntimeError(
            f"CUDA_PYTHON_TOOLCHAIN={name!r} is not supported on {platform_key}. Valid values: {', '.join(allowed)}."
        )
    cc, cxx = _TOOLCHAIN_COMPILERS[name]
    explicit = bool(os.environ.get("CUDA_PYTHON_TOOLCHAIN", "").strip())
    return name, allowed, cc, cxx, explicit


def _with_sccache(current, compiler):
    """Keep CC="sccache cc" as CC="sccache clang" when the toolchain picks a compiler."""
    if current:
        launcher = current.split()[0]
        if os.path.basename(launcher) == "sccache":
            return f"{launcher} {compiler}"
    return compiler


def _apply_toolchain_env(cc, cxx, explicit):
    """Set CC/CXX/LDSHARED for an explicitly-chosen toolchain.

    The default path (CUDA_PYTHON_TOOLCHAIN unset) intentionally
    does not touch the env, so an externally-set compiler (e.g.
    CC="sccache cc" in CI) keeps working. An explicit CUDA_PYTHON_TOOLCHAIN
    override (incl. =gnu) governs the compiler. An existing sccache prefix
    is kept (CC="sccache cc" + llvm -> CC="sccache clang").
    """
    if explicit and cc is not None:
        os.environ["CC"] = _with_sccache(os.environ.get("CC", ""), cc)
        os.environ["CXX"] = _with_sccache(os.environ.get("CXX", ""), cxx)
        os.environ["LDSHARED"] = f"{os.environ['CXX']} -shared"


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


# === Cython generated-source cache (opt-in via CUDA_PYTHON_CYTHON_CACHE_DIR) ===
# Workaround for Cython issue #7532: Cython's native cache fingerprint omits
# `compiler_directives`, so builds with different directives (e.g. linetrace
# for coverage) could reuse stale generated C/C++ output. This helper
# namespaces the Cython cache by package and a digest of output-affecting
# build configuration so distinct configurations get distinct caches.
#
# Removal: once cython/cython#7532 is resolved in a released Cython version
# and cuda-python's minimum Cython version includes the fix, this helper
# and its workaround-specific tests can be deleted; cythonize() can then be
# called with `cache=<root>` (or `cache=True`) without per-config namespacing.
# See https://github.com/cython/cython/issues/7532
def _cython_cache_path(
    package,
    *,
    compiler_directives=None,
    compile_time_env=None,
    language_level=None,
    cplus=None,
    debug=False,
    cuda_major=None,
):
    """Return a per-configuration Cython cache directory, or None to disable caching.

    Returns None when CUDA_PYTHON_CYTHON_CACHE_DIR is unset, so cythonize()
    is called without ``cache=`` and existing workflows are unchanged.
    """
    cache_root = os.environ.get("CUDA_PYTHON_CYTHON_CACHE_DIR")
    if not cache_root:
        return None
    if sys.platform == "win32":
        warn(
            "CUDA_PYTHON_CYTHON_CACHE_DIR is set but Cython caching via symlinks "
            "is not supported on Windows; caching will be disabled.",
            stacklevel=2,
        )
        return None

    h = hashlib.sha256()
    h.update(package.encode("utf-8"))
    # The Python version running cythonize affects generated C code
    # (e.g. CYTHON_COMPRESS_STRINGS: zstd on 3.14, zlib on 3.12/3.13).
    h.update(f"python={sys.version_info.major}.{sys.version_info.minor}".encode())

    def _update(name, value):
        h.update(name.encode("utf-8"))
        h.update(repr(value).encode("utf-8"))

    # compiler_directives are not in Cython's native fingerprint (#7532).
    if compiler_directives:
        for key in sorted(compiler_directives):
            _update(f"directive:{key}", compiler_directives[key])
    # compile_time_env, language_level, and cplus are already in Cython's
    # fingerprint, but we include them so the namespace stays correct even
    # if Cython's fingerprint logic changes.
    if compile_time_env:
        for key in sorted(compile_time_env):
            _update(f"compile_time_env:{key}", compile_time_env[key])
    if language_level is not None:
        _update("language_level", language_level)
    if cplus is not None:
        _update("cplus", cplus)
    # debug toggles gdb_debug in cythonize(), which affects generated code.
    _update("debug", debug)
    if cuda_major is not None:
        _update("cuda_major", cuda_major)

    return os.path.join(cache_root, f"{package}-{h.hexdigest()[:16]}")


@contextlib.contextmanager
def _stable_cython_alias(target: Path, alias: Path):
    """Atomically create a stable directory symlink alias for a Cython include tree.

    Cython's cache fingerprint includes the absolute path of each resolved
    .pxd dependency (via ``file_hash()``). PEP 517 build environments install
    dependencies under randomized temporary prefixes, making those paths
    unstable across runs. This context manager creates a fixed, worktree-
    relative symlink so Cython sees a stable lexical path.

    The symlink is created in the *package directory* (the directory containing
    this build_hooks.py), not in the cwd, to keep aliases package-local and
    avoid cross-package races.

    alias must not already exist as a real file or directory; if it is a
    symlink (including a dangling one) it is atomically replaced.

    On exit the alias is removed only if it still points at ``target`` (a
    racing replacement will not be deleted).

    POSIX only: directory symlinks require no elevated privileges on Linux.
    """
    # Resolve the *parent* directory (must exist), then append the name.
    # We deliberately do not follow a symlink that may already sit at alias.
    if not alias.is_absolute():
        alias = Path(__file__).parent / alias
    alias = alias.parent.resolve() / alias.name
    target = target.resolve()

    if alias.exists() and not alias.is_symlink():
        raise RuntimeError(
            f"Cannot create Cython include alias at {alias}: a real file or directory already exists there."
        )

    tmp_alias = alias.with_name(f".{alias.name}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        os.symlink(target, tmp_alias, target_is_directory=True)
        try:
            os.replace(tmp_alias, alias)
        except BaseException:
            tmp_alias.unlink(missing_ok=True)
            raise
        rel = os.path.relpath(alias, start=Path.cwd())
        yield rel
    finally:
        tmp_alias.unlink(missing_ok=True)
        # Only remove the alias we created; leave it alone if something else
        # has already replaced it (readlink will differ).
        try:
            if alias.is_symlink() and Path(os.readlink(alias)).resolve() == target:
                alias.unlink()
        except OSError:
            pass


# --- end shared build helpers ---


def _resolve_toolchain(debug=False, compile_for_coverage=False):
    """Resolve the C/C++ toolchain from CUDA_PYTHON_TOOLCHAIN.

    Returns (name, cc, cxx, extra_compile_args, extra_link_args). The default
    toolchain (gnu on Linux, msvc on Windows) reproduces the previous build
    behavior and does not touch CC/CXX/LDSHARED, so an externally-set compiler
    (e.g. CC="sccache cc") keeps working. A non-default toolchain (llvm on
    Linux) selects clang/clang++ and lld and sets CC/CXX/LDSHARED so distutils'
    customize_compiler picks them up.
    """
    name, _allowed, cc, cxx, explicit = _resolve_toolchain_name()

    extra_compile_args = []
    extra_link_args = []

    if name == "msvc":
        if debug:
            raise RuntimeError("Debuggable builds are not supported on Windows.")
    else:
        # Common Linux compile flags.
        extra_compile_args += ["-std=c++14", "-Wno-deprecated-declarations"]
        # Compiler-specific flags.
        if name == "gnu":
            extra_compile_args += ["-fpermissive", "-fno-var-tracking-assignments"]
        elif name == "llvm":
            extra_link_args += ["-fuse-ld=lld"]
        # Common Linux debug/opt flags.
        if debug:
            extra_compile_args += ["-g", "-O0", "-D _GLIBCXX_ASSERTIONS"]
        else:
            extra_compile_args += ["-g0", "-O3"]
            extra_link_args += ["-Wl,--strip-all"]

    if compile_for_coverage:
        # CYTHON_TRACE_NOGIL indicates to trace nogil functions.  It is not
        # related to free-threading builds.
        extra_compile_args += ["-DCYTHON_TRACE_NOGIL=1", "-DCYTHON_USE_SYS_MONITORING=0"]

    _apply_toolchain_env(cc, cxx, explicit)

    return name, cc, cxx, extra_compile_args, extra_link_args


# -----------------------------------------------------------------------
# Toolchain stamp

_BUILD_DIR = Path(__file__).parent / "build"


def _abi_stamp_path(stem):
    """Return a stamp path scoped to this interpreter's extension ABI."""
    extension_suffix = sysconfig.get_config_var("EXT_SUFFIX")
    if not extension_suffix:
        raise RuntimeError("Python's EXT_SUFFIX build configuration is unavailable")
    return _BUILD_DIR / f"{stem}{extension_suffix}"


# Records the toolchain of the last completed build for this extension ABI,
# so setup.py can force build_ext when it changes. Written by
# record_build_toolchain().
_BUILD_TOOLCHAIN_STAMP = _abi_stamp_path(".build-toolchain")

force_build_ext = False


def _check_build_toolchain(toolchain):
    """Set force_build_ext when the toolchain changed since the last build.

    Setuptools' freshness check does not include the extension flags, so a
    stale .so compiled by a previous toolchain would otherwise be packaged.
    """
    global force_build_ext

    try:
        previous = _BUILD_TOOLCHAIN_STAMP.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        previous = None

    # A missing stamp means the last build's toolchain is unknown, so force too.
    # On a first build that costs nothing: there are no artifacts to reuse.
    if previous != toolchain:
        print(f"Toolchain of last build: {previous} (building {toolchain}); forcing a full rebuild")
        force_build_ext = True


def record_build_toolchain() -> None:
    """Stamp the toolchain of the build that just completed.

    setup.py calls this after build_ext succeeds, so that a build which failed
    partway through does not claim outputs it never produced. Re-derives the
    toolchain name from the environment rather than caching it in a global.
    """
    name, *_ = _resolve_toolchain_name()
    _BUILD_TOOLCHAIN_STAMP.parent.mkdir(parents=True, exist_ok=True)
    _BUILD_TOOLCHAIN_STAMP.write_text(name + "\n", encoding="utf-8")


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
    import Cython
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

    # Force a full rebuild when the toolchain changed since the last successful
    # build, so a stale .so from a previous toolchain is never packaged.
    _check_build_toolchain(toolchain)

    cache_path = _cython_cache_path(
        "cuda-bindings",
        compiler_directives=cython_directives,
        language_level=3,
        cplus=True,
        debug=debug,
    )

    def _do_cythonize(cython_include_path):
        global _extensions
        _extensions = cythonize(
            extensions,
            nthreads=nthreads,
            build_dir="." if compile_for_coverage else "build/cython",
            compiler_directives=cython_directives,
            include_path=cython_include_path,
            cache=cache_path,
            **extra_cythonize_kwargs,
        )

    if cache_path is not None:
        # Alias Cython's bundled .pxd declarations under a stable worktree-relative
        # path so Cython's cache fingerprint sees the same path on every run
        # despite PEP 517 build environments landing under randomized temp prefixes.
        stdlib_target = Path(Cython.__file__).parent / "Includes"
        stdlib_alias = Path(__file__).parent / ".cython-stdlib"
        with _stable_cython_alias(stdlib_target, stdlib_alias) as rel_stdlib:
            _do_cythonize([".", rel_stdlib])
    else:
        _do_cythonize(["."])


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
