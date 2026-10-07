# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build helpers shared by the cuda-bindings and cuda-core PEP 517 backends.

``cuda_bindings/_build_shared.py`` is canonical. ``cuda_core/_build_shared.py``
is either a symlink to it or a byte-for-byte identical copy. Both backends load
the helper through their package-local path, keeping ``Path(__file__)``-relative
state package-local.

PEP 517 build isolation gives each backend its own ``build_hooks.py`` but not a
shared import path. Both packages declare ``backend-path = ["."]``, which puts
the package directory (and therefore this module) on ``sys.path`` while the
backend runs.

Only what genuinely differs per package is parameterized, via the arguments of
``resolve_toolchain``: the C++ standard, warnings-as-errors, and an optional
``tweak`` hook for flags a single package needs.

Besides the toolchain, this module owns the CUDA path lookup and the machinery that both backends use
around cythonize and build_ext: the opt-in Cython generated-source cache and
the build stamps that force a rebuild when the build configuration changed.

Note: There is no support guarantee for environment variables like
CUDA_PYTHON_TOOLCHAIN. They may be removed or changed in the future.
"""

import contextlib
import functools
import hashlib
import os
import shlex
import shutil
import sys
import sysconfig
import uuid
from pathlib import Path
from warnings import warn

# -----------------------------------------------------------------------
# CUDA path


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


def _with_compiler(command, compiler):
    """Replace the leading compiler on a linker command; keep flags.

    Conda ``LDCXXSHARED`` looks like ``g++ -pthread -B .../python_compiler_compat
    -shared ...``. Only the executable changes so those flags stay on the
    link line. The command is tokenized with shlex so quoted arguments
    survive, and a leading ``env VAR=value`` prefix is preserved. CC/CXX are
    not rewritten this way: they may already be a launcher plus compiler
    (``sccache cc``).
    """
    if not command or not command.strip():
        return compiler
    parts = shlex.split(command)
    # Keep an ``env VAR=value ...`` prefix: setuptools' C++ link step splits it
    # off before it substitutes the compiler, so it still reaches the link line.
    prefix_end = 0
    if parts and os.path.basename(parts[0]) == "env":
        prefix_end = 1
        # Match setuptools' _split_env: any token with ``=`` is an env operand
        # (covers both ``VAR=value`` and ``--unset=VAR`` long options).
        while prefix_end < len(parts) and "=" in parts[prefix_end]:
            prefix_end += 1
    # Everything else before the first flag is the old compiler (or a launcher
    # for it; setuptools takes the launcher from CXX instead).
    i = prefix_end
    while i < len(parts) and not parts[i].startswith("-"):
        i += 1
    return shlex.join([*parts[:prefix_end], compiler, *parts[i:]])


def _with_sccache(current, compiler):
    """Keep a leading sccache token when the toolchain picks a compiler.

    CI sets ``CC="sccache cc"`` or ``CC="/host/.../sccache cc"``. An explicit
    toolchain then becomes ``CC="sccache clang"`` rather than a bare compiler.
    """
    if current:
        launcher = current.split()[0]
        if os.path.basename(launcher) == "sccache":
            return f"{launcher} {compiler}"
    return compiler


def _apply_toolchain_env(cc, cxx, explicit):
    """Set CC/CXX/LDCXXSHARED for an explicitly-chosen toolchain.

    The default path (CUDA_PYTHON_TOOLCHAIN unset) intentionally
    does not touch the env, so an externally-set compiler (e.g.
    CC="sccache cc" in CI) keeps working. An explicit CUDA_PYTHON_TOOLCHAIN
    override (incl. =gnu) sets CC/CXX to the toolchain compiler; an existing
    sccache prefix is kept (CC="sccache cc" + llvm -> CC="sccache clang").
    Extras on LDCXXSHARED (rpath, -pthread, -B, ...) are kept, taken from the
    environment if set there and from sysconfig otherwise; only the compiler
    is swapped. LDSHARED is left unset so distutils rewrites it from CC.
    """
    if explicit and cc is not None:
        os.environ["CC"] = _with_sccache(os.environ.get("CC", ""), cc)
        os.environ["CXX"] = _with_sccache(os.environ.get("CXX", ""), cxx)
        # An LDCXXSHARED the user already exported takes precedence over
        # sysconfig's, as CC/CXX do; either way only the compiler is swapped.
        ldcxxshared = (
            os.environ.get("LDCXXSHARED")
            or sysconfig.get_config_var("LDCXXSHARED")
            or sysconfig.get_config_var("LDSHARED")
        )
        os.environ["LDCXXSHARED"] = _with_compiler(ldcxxshared, cxx) if ldcxxshared else f"{cxx} -shared"


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
# Compiler and linker flags


def _build_flags(name, cxx_std, debug, coverage, warnings_as_errors):
    """Return (extra_compile_args, extra_link_args) for toolchain ``name``.

    The one flag set used by both packages:

    - Linux compile: ``-std=c++{cxx_std}``, then ``-g0 -O2`` (opt) or
      ``-g -O0 -D _GLIBCXX_ASSERTIONS`` (debug).
    - Linux link: ``-fuse-ld=lld`` (llvm) and ``-Wl,--strip-all`` (opt).
    - MSVC compile: ``/std:c++{cxx_std}`` and ``/O2``. Modern setuptools no
      longer forces ``/Ox``, so the optimization level is set explicitly for
      symmetry with Linux ``-O2``. Debug builds are not supported on Windows.
    - Coverage: Cython tracing defines.
    - Warnings-as-errors: ``-Werror`` (Linux) or ``/WX`` (MSVC).
    """
    extra_compile_args = []
    extra_link_args = []

    if name == "msvc":
        if debug:
            raise RuntimeError("Debuggable builds are not supported on Windows.")
        extra_compile_args += [f"/std:c++{cxx_std}", "/O2"]
    else:
        # Common Linux compile flags.
        extra_compile_args += [f"-std=c++{cxx_std}"]
        # Compiler-specific flags.
        if name == "llvm":
            extra_link_args += ["-fuse-ld=lld"]
        # Common Linux debug/opt flags.
        if debug:
            extra_compile_args += ["-g", "-O0", "-D _GLIBCXX_ASSERTIONS"]
        else:
            extra_compile_args += ["-g0", "-O2"]
            extra_link_args += ["-Wl,--strip-all"]

    if coverage:
        # CYTHON_TRACE_NOGIL indicates to trace nogil functions.  It is not
        # related to free-threading builds.
        extra_compile_args += ["-DCYTHON_TRACE_NOGIL=1", "-DCYTHON_USE_SYS_MONITORING=0"]

    if warnings_as_errors:
        # The MSVC exemptions cover warnings that Cython's utility code
        # produces in every module and the .pyx sources cannot fix:
        # - C4551 ("function call missing argument list"), hundreds per
        #   module.
        # - C4244 (narrowing): the overflow-check helpers that
        #   @cython.overflowcheck(True) instantiates for _layout.pxd narrow
        #   int64 to int inside Cython's own code.
        # gcc and clang need no exemption. The one generated warning they
        # report in cuda.core, the unused @overload wrappers of
        # Graph.__getitem__, is silenced by a pragma in
        # cuda/core/graph/_graph_builder.pyx.
        if name == "msvc":
            extra_compile_args += ["/WX", "/wd4551", "/wd4244"]
        else:
            extra_compile_args += ["-Werror"]

    return extra_compile_args, extra_link_args


def resolve_toolchain(*, cxx_std, debug=False, compile_for_coverage=False, warnings_as_errors=False, tweak=None):
    """Resolve the C/C++ toolchain from CUDA_PYTHON_TOOLCHAIN.

    Returns (name, cc, cxx, extra_compile_args, extra_link_args). The default
    toolchain (gnu on Linux, msvc on Windows) reproduces the previous build
    behavior and does not touch CC/CXX/LDCXXSHARED, so an externally-set compiler
    (e.g. CC="sccache cc") keeps working. A non-default toolchain (llvm on
    Linux) selects clang/clang++ and lld and sets CC/CXX/LDCXXSHARED so distutils'
    customize_compiler picks them up.

    The per-package choices are arguments:

    - ``cxx_std`` (required): the C++ standard, e.g. ``14``. There is
      deliberately no shared default; each package picks its own.
    - ``warnings_as_errors``: opt in to ``-Werror`` / ``/WX``.
    - ``tweak``: optional callable ``tweak(name, extra_compile_args,
      extra_link_args)`` returning the adjusted ``(extra_compile_args,
      extra_link_args)`` pair, for the few flags a single package needs that
      do not belong in the shared set.
    """
    name, _allowed, cc, cxx, explicit = _resolve_toolchain_name()

    extra_compile_args, extra_link_args = _build_flags(name, cxx_std, debug, compile_for_coverage, warnings_as_errors)
    if tweak is not None:
        extra_compile_args, extra_link_args = tweak(name, extra_compile_args, extra_link_args)

    _apply_toolchain_env(cc, cxx, explicit)

    return name, cc, cxx, extra_compile_args, extra_link_args


# -----------------------------------------------------------------------
# Cython cache helpers


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

    The symlink is created in the package directory containing this file, not
    in the cwd, to keep aliases package-local and avoid cross-package races.

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


# -----------------------------------------------------------------------
# Build stamps
#
# Setuptools' freshness check covers neither the extension flags nor the
# build configuration, so a stale .so built under another configuration looks
# perfectly fresh. Each backend stamps the key of its last completed build
# (the toolchain for cuda-bindings; CUDA major, toolchain and debug/coverage
# for cuda-core). check_build_key() sets ``force_build_ext`` when the key
# changed, and setup.py hands that to build_ext.

# Where per-configuration build artifacts live. Anchored to this file rather
# than the cwd, since a project can be built from anywhere.
_BUILD_DIR = Path(__file__).parent / "build"

# Set by check_build_key(). Read it as ``build_hooks.force_build_ext``: both
# build_hooks modules re-export it.
force_build_ext = False


def _abi_stamp_path(stem):
    """Return a stamp path scoped to this interpreter's extension ABI."""
    extension_suffix = sysconfig.get_config_var("EXT_SUFFIX")
    if not extension_suffix:
        raise RuntimeError("Python's EXT_SUFFIX build configuration is unavailable")
    return _BUILD_DIR / f"{stem}{extension_suffix}"


def check_build_key(stamp, key, description):
    """Set force_build_ext when ``key`` differs from the one stamped at ``stamp``.

    ``description`` names the key in the message that explains the rebuild.
    """
    global force_build_ext

    try:
        previous = stamp.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        previous = None

    # A missing stamp means the last build's key is unknown, so force too.
    # On a first build that costs nothing: there are no artifacts to reuse.
    if previous != key:
        print(f"{description} of last build: {previous} (building {key}); forcing a full rebuild")
        force_build_ext = True


def record_build_key(stamp, key):
    """Stamp ``key`` at ``stamp``, once the build it describes has completed.

    A build that failed partway through must not claim outputs it never
    produced, so callers record only after the build succeeded.
    """
    stamp.parent.mkdir(parents=True, exist_ok=True)
    stamp.write_text(key + "\n", encoding="utf-8")
