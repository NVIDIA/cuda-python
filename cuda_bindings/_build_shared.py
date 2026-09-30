# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared PEP 517 backend helpers for the cuda.bindings and cuda.core build backends.

This file is the single source of truth for helpers duplicated between the two
backends: CUDA_PYTHON_TOOLCHAIN resolution, the PEP 517 pathfinder-shadowing
workaround, the extension-ABI-scoped build stamp path helper, and the
CUDA_PYTHON_CYTHON_CACHE_DIR opt-in Cython cache helpers.
`cuda_core/_build_shared.py` is a symlink to this file, so both backends
resolve the shared code through their own `backend-path`.

Per-package logic (extension list, flag assembly, stamp bookkeeping, PEP 517
hooks) stays in each package's `build_hooks.py`.
"""

import contextlib
import functools
import hashlib
import os
import shutil
import sys
import sysconfig
import uuid
from pathlib import Path
from warnings import warn

# -----------------------------------------------------------------------
# CUDA path resolution (via cuda.pathfinder, with a PEP 517 shim)


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
# Build stamp paths
#
# `Path(__file__)` reflects the path this module was loaded from, and Python
# does not resolve symlinks on it. When cuda_core imports this file via its
# `_build_shared.py` symlink, `_BUILD_DIR` resolves to `cuda_core/build/`;
# imported directly from cuda_bindings, it resolves to `cuda_bindings/build/`.
# So each package's stamp files land under its own build/ directory even
# though the code lives in one file.

_BUILD_DIR = Path(__file__).parent / "build"


def _abi_stamp_path(stem):
    """Return a stamp path scoped to this interpreter's extension ABI."""
    extension_suffix = sysconfig.get_config_var("EXT_SUFFIX")
    if not extension_suffix:
        raise RuntimeError("Python's EXT_SUFFIX build configuration is unavailable")
    return _BUILD_DIR / f"{stem}{extension_suffix}"


# Set to True by ``check_build_key`` when the current key differs from the
# stamped one. Each ``build_hooks.py`` re-exports this via module-level
# ``__getattr__`` so setup.py's ``build_hooks.force_build_ext`` attribute
# read continues to work transparently.
force_build_ext = False


def check_build_key(stamp, get_key) -> None:
    """Compare ``get_key()`` against the value in ``stamp`` and flip force_build_ext.

    Each backend supplies a package-specific ``get_key`` callable (e.g. the
    toolchain name for cuda.bindings, or a composite ``cu{major}-{toolchain}-
    {opt|debug}[-cov]`` key for cuda.core). A missing stamp counts as a
    change, which forces a rebuild on the first build after this helper is
    introduced. That is the intended cost.
    """
    global force_build_ext
    key = get_key()
    try:
        previous = stamp.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        previous = None
    if previous != key:
        print(f"Build key of last build: {previous} (building {key}); forcing a full rebuild")
        force_build_ext = True


def record_build_key(stamp, get_key) -> None:
    """Stamp ``get_key()`` at ``stamp``, creating parent directories as needed."""
    stamp.parent.mkdir(parents=True, exist_ok=True)
    stamp.write_text(get_key() + "\n", encoding="utf-8")


# -----------------------------------------------------------------------
# Toolchain selection

_TOOLCHAINS_LINUX = ("gnu", "llvm")
_TOOLCHAINS_WINDOWS = ("msvc",)
_TOOLCHAIN_COMPILERS = {
    "gnu": ("gcc", "g++"),
    "llvm": ("clang", "clang++"),
    "msvc": (None, None),
}


def _resolve_toolchain_name():
    """Read CUDA_PYTHON_TOOLCHAIN, validate it, return (name, allowed, cc, cxx, explicit).

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


def _build_flags(name, cxx_std, debug, compile_for_coverage):
    """Compile/link flags for a resolved toolchain (shared across backends).

    ``cxx_std`` is required — the two backends can legitimately differ (bindings
    stays on ``c++14`` to avoid a c++17 variadic-template regression on the
    kernel-launch code paths; core is on ``c++17``), and there is no defensible
    shared default. See https://github.com/NVIDIA/cuda-python/issues/1882.
    """
    extra_compile_args = []
    extra_link_args = []

    if name == "msvc":
        extra_compile_args += [f"/std:c++{cxx_std}"]
    else:
        extra_compile_args += [f"-std=c++{cxx_std}"]
        if name == "llvm":
            extra_link_args += ["-fuse-ld=lld"]
        if debug:
            extra_compile_args += ["-g", "-O0", "-D _GLIBCXX_ASSERTIONS"]
        else:
            extra_compile_args += ["-g0", "-O2"]
            extra_link_args += ["-Wl,--strip-all"]

    if compile_for_coverage:
        # CYTHON_TRACE_NOGIL indicates to trace nogil functions.  It is not
        # related to free-threading builds.
        extra_compile_args += ["-DCYTHON_TRACE_NOGIL=1", "-DCYTHON_USE_SYS_MONITORING=0"]

    return extra_compile_args, extra_link_args


def resolve_toolchain(*, cxx_std, debug=False, compile_for_coverage=False, tweak=None):
    """Resolve the C/C++ toolchain from CUDA_PYTHON_TOOLCHAIN.

    Returns (name, cc, cxx, extra_compile_args, extra_link_args). The default
    toolchain (gnu on Linux, msvc on Windows) does not touch CC/CXX/LDSHARED,
    so an externally-set compiler (e.g. CC="sccache cc") keeps working. An
    explicit CUDA_PYTHON_TOOLCHAIN (llvm on Linux) sets CC/CXX/LDSHARED to
    the toolchain's binaries so distutils' customize_compiler picks them up.

    ``cxx_std`` is required — each backend chooses its own C++ standard.
    ``tweak`` is an optional post-hook ``(name, cargs, largs) -> (cargs, largs)``
    for package-specific flag layering (e.g. bindings adds
    ``-Wno-deprecated-declarations``).
    """
    name, _allowed, cc, cxx, explicit = _resolve_toolchain_name()
    if name == "msvc" and debug:
        raise RuntimeError("Debuggable builds are not supported on Windows.")
    extra_compile_args, extra_link_args = _build_flags(name, cxx_std, debug, compile_for_coverage)
    if tweak is not None:
        extra_compile_args, extra_link_args = tweak(name, extra_compile_args, extra_link_args)
    _apply_toolchain_env(cc, cxx, explicit)
    return name, cc, cxx, extra_compile_args, extra_link_args


# -----------------------------------------------------------------------
# Cython generated-source cache (opt-in via CUDA_PYTHON_CYTHON_CACHE_DIR)
#
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
    this shared file — i.e. cuda_bindings/ or cuda_core/ depending on which
    backend loaded us; Python does not resolve `__file__` through symlinks),
    not in the cwd, to keep aliases package-local and avoid cross-package
    races.

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
