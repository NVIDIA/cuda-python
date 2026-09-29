# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# This module implements basic PEP 517 backend support, see e.g.
# - https://peps.python.org/pep-0517/
# - https://setuptools.pypa.io/en/latest/build_meta.html#dynamic-build-dependencies-and-other-build-meta-tweaks
# Specifically, there are 5 APIs required to create a proper build backend, see below.

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
import zipfile
from pathlib import Path
from warnings import warn

import Cython as _Cython
from Cython.Build import cythonize
from Cython.Compiler import Options as _CythonOptions
from setuptools import Extension
from setuptools import build_meta as _build_meta

prepare_metadata_for_build_editable = _build_meta.prepare_metadata_for_build_editable
prepare_metadata_for_build_wheel = _build_meta.prepare_metadata_for_build_wheel
build_sdist = _build_meta.build_sdist
get_requires_for_build_sdist = _build_meta.get_requires_for_build_sdist

# Note: There is no support guarantee for environment variables like CUDA_PYTHON_COVERAGE,
# CUDA_PYTHON_TOOLCHAIN, CUDA_PYTHON_CYTHON_CACHE_DIR, etc. They may be removed
# or changed in the future.
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


# -----------------------------------------------------------------------
# Toolchain selection
#
# There is one shared helper block below, duplicated verbatim in
# cuda_bindings/build_hooks.py (keep it in sync; enforced by
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


def _apply_toolchain_env(cc, cxx, explicit):
    """Set CC/CXX/LDSHARED for an explicitly-chosen toolchain.

    The default path (CUDA_PYTHON_TOOLCHAIN unset) intentionally
    does not touch the env, so an externally-set compiler (e.g.
    CC="sccache cc" in CI) keeps working. An explicit CUDA_PYTHON_TOOLCHAIN
    override (incl. =gnu) governs the compiler and overrides CC/CXX/LDSHARED.
    """
    if explicit and cc is not None:
        os.environ["CC"] = cc
        os.environ["CXX"] = cxx
        os.environ["LDSHARED"] = f"{cxx} -shared"


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
    """Resolve the C/C++ toolchain from CUDA_PYTHON_TOOLCHAIN (cuda.core flags).

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
        extra_compile_args += ["/std:c++17"]
        if debug:
            raise RuntimeError("Debuggable builds are not supported on Windows.")
    else:
        # Common Linux compile flags.
        extra_compile_args += ["-std=c++17"]
        # Compiler-specific flags.
        if name == "llvm":
            extra_link_args += ["-fuse-ld=lld"]
        # Common Linux debug/opt flags.
        if debug:
            extra_compile_args += ["-g", "-O0", "-D _GLIBCXX_ASSERTIONS"]
        else:
            extra_compile_args += ["-g0", "-O2"]
            extra_link_args += ["-Wl,--strip-all"]

    if compile_for_coverage:
        # CYTHON_TRACE_NOGIL indicates to trace nogil functions.  It is not
        # related to free-threading builds.
        extra_compile_args += ["-DCYTHON_TRACE_NOGIL=1", "-DCYTHON_USE_SYS_MONITORING=0"]

    _apply_toolchain_env(cc, cxx, explicit)

    return name, cc, cxx, extra_compile_args, extra_link_args


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
    cuda_path = _get_cuda_path()
    cuda_h = os.path.join(cuda_path, "include", "cuda.h")
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
        pass

    # CUDA_PATH or CUDA_HOME is required for the build, so we should not reach here
    # in normal circumstances. Raise an error to make the issue clear.
    raise RuntimeError(
        "Cannot determine CUDA major version. "
        "Set CUDA_CORE_BUILD_MAJOR environment variable, or ensure CUDA_PATH or CUDA_HOME "
        "points to a valid CUDA installation with include/cuda.h."
    )


# used later by setup()
_extensions = None

# Where per-configuration build artifacts live. Anchored to this file rather
# than the cwd, since a project can be built from anywhere.
_BUILD_DIR = Path(__file__).parent / "build"


def _abi_stamp_path(stem):
    """Return a stamp path scoped to this interpreter's extension ABI."""
    extension_suffix = sysconfig.get_config_var("EXT_SUFFIX")
    if not extension_suffix:
        raise RuntimeError("Python's EXT_SUFFIX build configuration is unavailable")
    return _BUILD_DIR / f"{stem}{extension_suffix}"


# Records the build configuration (CUDA major, toolchain, debug/coverage) of
# the last completed build for this extension ABI, so setup.py can force
# build_ext when it changes. Written by record_build_config() after the
# PEP 517 backend succeeds.
_BUILD_CONFIG_STAMP = _abi_stamp_path(".build-config")

force_build_ext = False


def _build_config_key(cuda_major, toolchain, debug, coverage):
    """Return a stable string key for the build configuration."""
    return f"cu{cuda_major}-{toolchain}-{'debug' if debug else 'opt'}{'-cov' if coverage else ''}"


def _check_build_config(toolchain, debug, coverage):
    """Return (cuda_major, config_key), and force a rebuild when the config changed.

    Cython's up-to-date check does not hash ``compile_time_env`` or the
    extension flags, so generated sources and compiled extensions from a
    previous configuration would otherwise be reused. Keying the generated-
    source directory fixes the generated C++, but not the compiled
    extension: in an editable install it lands in the source tree under a
    name keyed by the Python ABI tag alone. The build configuration (CUDA
    major, toolchain, debug/coverage) is therefore stamped and build_ext
    forced whenever it changes, so a stale .so is never packaged.
    """
    global force_build_ext

    cuda_major = _determine_cuda_major_version()
    key = _build_config_key(cuda_major, toolchain, debug, coverage)
    try:
        previous = _BUILD_CONFIG_STAMP.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        previous = None

    # A missing stamp means the last build's config is unknown, so force too.
    # On a first build that costs nothing: there are no artifacts to reuse.
    if previous != key:
        print(f"Build config of last build: {previous} (building {key}); forcing a full rebuild")
        force_build_ext = True

    return cuda_major, key


def record_build_config(key) -> None:
    """Stamp the exact configuration that `_build_cuda_core()` prepared.

    The PEP 517 hooks call this after the wheel or editable build succeeds,
    passing the key already checked rather than re-deriving from ambient
    state (setuptools' `build_ext.debug` is not `config_settings["debug"]`).
    """
    _BUILD_CONFIG_STAMP.parent.mkdir(parents=True, exist_ok=True)
    _BUILD_CONFIG_STAMP.write_text(key + "\n", encoding="utf-8")


def _relativize_extension_sources(extensions) -> None:
    """Keep absolute source paths out of setuptools' temporary build tree."""
    for extension in extensions:
        extension.sources = [
            os.path.relpath(source, start=Path.cwd()) if os.path.isabs(source) else source
            for source in extension.sources
        ]


def _extension_sources(mod_name):
    """The module's .pyx plus its C++, if any: every .cpp under
    cuda/core/_cpp/<stem>/, or the single legacy file cuda/core/_cpp/<stem>.cpp.
    Example: _tensor_map.pyx compiles _cpp/tensor_map.cpp."""
    sources = [f"cuda/core/{mod_name}.pyx"]
    cpp_stem = Path("cuda", "core", "_cpp", mod_name.lstrip("_"))
    if cpp_stem.is_dir():
        cpp_sources = sorted(str(path) for path in cpp_stem.rglob("*.cpp"))
        if not cpp_sources:
            raise RuntimeError(f"{cpp_stem}/ exists but contains no .cpp files")
        sources.extend(cpp_sources)
    elif cpp_stem.with_suffix(".cpp").is_file():
        sources.append(str(cpp_stem.with_suffix(".cpp")))
    return sources


def _extension_depends():
    """Headers whose edits must rebuild an extension: every header under a
    directory-form module's cuda/core/_cpp/<stem>/ (a single-file module has
    none).

    The same list serves every extension. A module that cimports a
    directory-form module compiles against the header its .pxd names, and
    cythonize copies each `depends` entry into its build directory before
    compiling, so the copied header finds its sibling includes beside it
    (quoted includes resolve next to the copy, not in the source tree).
    Listing the whole directory keeps the rule free of include parsing; the
    cost is that every extension rebuilds when any of these headers changes,
    exactly as editing the one monolithic header did before the split."""
    cpp = Path("cuda", "core", "_cpp")
    return sorted(
        str(path)
        for module_dir in cpp.iterdir()
        if module_dir.is_dir()
        for path in module_dir.rglob("*")
        if path.suffix in (".h", ".hpp")
    )


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
    cuda_package_dir = None
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

    all_include_dirs = [os.path.join(cuda_path, "include")]

    # Resolve the C/C++ toolchain (CUDA_PYTHON_TOOLCHAIN). The default (gnu on
    # Linux, msvc on Windows) reproduces the previous build behavior and does
    # not touch CC/CXX, so an externally-set compiler (e.g. sccache) survives.
    toolchain, _cc, _cxx, extra_compile_args, extra_link_args = _resolve_toolchain(
        debug=debug, compile_for_coverage=COMPILE_FOR_COVERAGE
    )
    _check_toolchain_available(toolchain)
    extra_cythonize_kwargs = {}
    if debug and sys.platform != "win32":
        extra_cythonize_kwargs["gdb_debug"] = True

    depends = _extension_depends()
    ext_modules = tuple(
        Extension(
            f"cuda.core.{mod.replace(os.path.sep, '.')}",
            sources=_extension_sources(mod),
            depends=depends,
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
    cuda_major, config_key = _check_build_config(toolchain, debug, COMPILE_FOR_COVERAGE)

    nthreads = int(os.environ.get("CUDA_PYTHON_PARALLEL_LEVEL", os.cpu_count() // 2))
    compile_time_env = {"CUDA_CORE_BUILD_MAJOR": int(cuda_major)}
    compiler_directives = {"embedsignature": True, "warn.deprecated.IF": False, "freethreading_compatible": True}
    _CythonOptions.warning_errors = True
    if COMPILE_FOR_COVERAGE:
        compiler_directives["linetrace"] = True
    cache_path = _cython_cache_path(
        "cuda-core",
        compiler_directives=compiler_directives,
        compile_time_env=compile_time_env,
        language_level=3,
        cplus=True,
        debug=debug,
        cuda_major=cuda_major,
    )

    def _do_cythonize(cython_include_path):
        global _extensions
        _extensions = cythonize(
            ext_modules,
            verbose=True,
            language_level=3,
            # CUDA_PYTHON_COVERAGE deliberately generates in-tree so the sources can
            # be packaged; every other build gets its own per-configuration cache,
            # anchored alongside the stamp so both resolve the same from any cwd.
            # Cython also copies each extension's extern headers and `depends` under
            # this directory and compiles against the copies. Copies are refreshed by
            # mtime and never deleted, so remove build/ after renaming or deleting a
            # header under _cpp/.
            build_dir="." if COMPILE_FOR_COVERAGE else str(_BUILD_DIR / "cython" / config_key),
            nthreads=nthreads,
            compiler_directives=compiler_directives,
            compile_time_env=compile_time_env,
            include_path=cython_include_path,
            cache=cache_path,
            **extra_cythonize_kwargs,
        )

    if cache_path is not None:
        # Alias both Cython's bundled .pxd declarations and cuda.bindings declarations
        # under stable worktree-relative paths so Cython's cache fingerprint stays
        # stable across PEP 517 builds (which install deps under randomized prefixes).
        stdlib_target = Path(_Cython.__file__).parent / "Includes"
        stdlib_alias = Path(__file__).parent / ".cython-stdlib"
        bindings_alias = Path(__file__).parent / ".cython-bindings"

        with _stable_cython_alias(stdlib_target, stdlib_alias) as rel_stdlib:
            if cuda_package_dir is not None:
                with _stable_cython_alias(cuda_package_dir, bindings_alias) as rel_bindings:
                    _do_cythonize([".", rel_bindings, rel_stdlib])
            else:
                _do_cythonize([".", rel_stdlib])
    else:
        _do_cythonize(["."])
    # Cython returns generated sources under the absolute build_dir above.
    # setuptools mirrors absolute source paths into build/temp, which can push
    # MSVC linker output paths past MAX_PATH in deeper Windows checkouts.
    _relativize_extension_sources(_extensions)

    return config_key


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
    config_key = _build_cuda_core(debug=debug)
    wheel_name = _build_meta.build_editable(wheel_directory, config_settings, metadata_directory)

    # Patch the .pth file to add Cython include paths
    wheel_path = os.path.join(wheel_directory, wheel_name)
    _add_cython_include_paths_to_pth(wheel_path)
    record_build_config(config_key)

    return wheel_name


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    debug = config_settings.get("debug", False) if config_settings else False
    config_key = _build_cuda_core(debug=debug)
    wheel_name = _build_meta.build_wheel(wheel_directory, config_settings, metadata_directory)
    record_build_config(config_key)
    return wheel_name


def _get_cuda_bindings_require():
    cuda_major = _determine_cuda_major_version()
    return [f"cuda-bindings=={cuda_major}.*"]


def get_requires_for_build_editable(config_settings=None):
    return _build_meta.get_requires_for_build_editable(config_settings) + _get_cuda_bindings_require()


def get_requires_for_build_wheel(config_settings=None):
    return _build_meta.get_requires_for_build_wheel(config_settings) + _get_cuda_bindings_require()
