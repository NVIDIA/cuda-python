# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build helpers shared by the cuda-bindings and cuda-core PEP 517 backends.

This is the single source of truth. ``cuda_core/_build_shared.py`` is a symlink
to this file. Python does not dereference symlinks in ``__file__``, so any
``Path(__file__)``-relative location in here resolves under whichever package
loads it.

PEP 517 build isolation gives each backend its own ``build_hooks.py`` but not a
shared import path. Both packages declare ``backend-path = ["."]``, which puts
the package directory (and therefore this module) on ``sys.path`` while the
backend runs.

Only what genuinely differs per package is parameterized, via the arguments of
``resolve_toolchain``: the C++ standard, warnings-as-errors, and an optional
``tweak`` hook for flags a single package needs.

Note: There is no support guarantee for environment variables like
CUDA_PYTHON_TOOLCHAIN. They may be removed or changed in the future.
"""

import os
import shlex
import shutil
import sys
import sysconfig

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
