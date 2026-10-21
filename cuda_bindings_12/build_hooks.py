# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# This module implements basic PEP 517 backend support to defer CUDA-dependent
# logic (header parsing, code generation, cythonization) to build time. See:
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

import _build_shared
from _build_shared import (
    _abi_stamp_path,
    _check_toolchain_available,
    _cython_cache_path,
    _resolve_toolchain_name,
    _stable_cython_alias,
    check_build_key,
    record_build_key,
    resolve_toolchain,
)

# Metadata hooks delegate directly to setuptools -- no CUDA needed.
prepare_metadata_for_build_editable = _build_meta.prepare_metadata_for_build_editable
prepare_metadata_for_build_wheel = _build_meta.prepare_metadata_for_build_wheel
build_sdist = _build_meta.build_sdist
get_requires_for_build_sdist = _build_meta.get_requires_for_build_sdist
get_requires_for_build_wheel = _build_meta.get_requires_for_build_wheel
get_requires_for_build_editable = _build_meta.get_requires_for_build_editable


def __getattr__(name):
    # setup.py reads ``build_hooks.force_build_ext``; the flag itself lives in
    # _build_shared, where check_build_key() sets it.
    if name == "force_build_ext":
        return _build_shared.force_build_ext
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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
_INSTALL_URL = "https://nvidia.github.io/cuda-python/cuda-bindings/12.9.10/install.html#installing-from-source"


@functools.cache
def _get_cuda_paths() -> list[str]:
    CUDA_HOME = os.environ.get("CUDA_HOME", os.environ.get("CUDA_PATH", None))
    if not CUDA_HOME:
        raise RuntimeError("Environment variable CUDA_HOME or CUDA_PATH is not set")
    CUDA_HOME = CUDA_HOME.split(os.pathsep)
    print("CUDA paths:", CUDA_HOME)
    return CUDA_HOME


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


def _tweak_flags(name, extra_compile_args, extra_link_args):
    """cuda-bindings flags that do not belong in the shared set."""
    if name != "msvc":
        # cudaMemcpy*Array* and cudaGetDriverEntryPoint are deprecated but still
        # supported; suppress the resulting warnings so a future -Werror build
        # is not broken by Cython-generated calls we cannot control.
        extra_compile_args = [*extra_compile_args, "-Wno-deprecated-declarations"]
    return extra_compile_args, extra_link_args


def _resolve_toolchain(debug=False, compile_for_coverage=False):
    """Resolve the C/C++ toolchain from CUDA_PYTHON_TOOLCHAIN (cuda.bindings flags).

    See _build_shared.resolve_toolchain() for the return value and the
    environment handling. What is specific to cuda.bindings is declared here.
    """
    return resolve_toolchain(
        # c++14: raising to c++17 costs a measured ~15% on launch_{256,512}_args
        # from gcc's c++17 variadic-template expansion.
        cxx_std=14,
        debug=debug,
        compile_for_coverage=compile_for_coverage,
        tweak=_tweak_flags,
    )


# -----------------------------------------------------------------------
# Toolchain stamp

# Records the toolchain of the last completed build for this extension ABI,
# so setup.py can force build_ext when it changes. Written by
# record_build_toolchain().
_BUILD_TOOLCHAIN_STAMP = _abi_stamp_path(".build-toolchain")


def _check_build_toolchain(toolchain):
    """Set force_build_ext when the toolchain changed since the last build.

    Setuptools' freshness check does not include the extension flags, so a
    stale .so compiled by a previous toolchain would otherwise be packaged.
    """
    check_build_key(_BUILD_TOOLCHAIN_STAMP, toolchain, "Toolchain")


def record_build_toolchain() -> None:
    """Stamp the toolchain of the build that just completed.

    setup.py calls this after build_ext succeeds, so that a build which failed
    partway through does not claim outputs it never produced. Re-derives the
    toolchain name from the environment rather than caching it in a global.
    """
    name, *_ = _resolve_toolchain_name()
    record_build_key(_BUILD_TOOLCHAIN_STAMP, name)


# -----------------------------------------------------------------------
# Header parsing helpers (called only from _build_cuda_bindings)

_REQUIRED_HEADERS = {
    "runtime": [
        "driver_types.h",
        "vector_types.h",
        "cuda_runtime.h",
        "surface_types.h",
        "texture_types.h",
        "library_types.h",
        "cuda_runtime_api.h",
        "device_types.h",
        "driver_functions.h",
        "cuda_profiler_api.h",
    ],
    # nvrtc: headers no longer parsed at build time (pre-generated by cybind).
    # During compilation, Cython will reference C headers that are not
    # explicitly parsed above. These are the known dependencies:
    #
    # - crt/host_defines.h
    # - builtin_types.h
    # - cuda_device_runtime_api.h
}


class _Struct:
    def __init__(self, name, members):
        self._name = name
        self._member_names = []
        self._member_types = []
        self._member_declarators = []
        for var_name, var_type, _ in members:
            base_type = var_type[0]
            base_type = base_type.removeprefix("struct ")
            base_type = base_type.removeprefix("union ")

            self._member_names += [var_name]
            self._member_types += [base_type]
            self._member_declarators += [tuple(var_type[1:])]

    def member_type(self, member_name):
        try:
            return self._member_types[self._member_names.index(member_name)]
        except ValueError:
            return None

    def member_array_length(self, member_name):
        try:
            declarators = self._member_declarators[self._member_names.index(member_name)]
        except ValueError:
            return None

        for declarator in declarators:
            if isinstance(declarator, list) and len(declarator) == 1:
                return declarator[0]
        return None

    def discoverMembers(self, memberDict, prefix, seen=None):
        if seen is None:
            seen = set()
        elif self._name in seen:
            return []

        discovered = []
        next_seen = set(seen)
        next_seen.add(self._name)

        for memberName, memberType in zip(self._member_names, self._member_types):
            if memberName:
                discovered.append(".".join([prefix, memberName]))

            t = memberType.replace("const ", "").replace("volatile ", "").strip().rstrip(" *")
            if t in memberDict and t != self._name:
                discovered += memberDict[t].discoverMembers(
                    memberDict, discovered[-1] if memberName else prefix, next_seen
                )

        return discovered

    def __repr__(self):
        return f"{self._name}: {self._member_names} with types {self._member_types}"


def _fetch_header_paths(required_headers, include_path_list):
    header_dict = {}
    missing_headers = []
    for library, header_list in required_headers.items():
        header_paths = []
        for header in header_list:
            path_candidate = [os.path.join(path, header) for path in include_path_list]
            for path in path_candidate:
                if os.path.exists(path):
                    header_paths += [path]
                    break
            else:
                missing_headers += [header]

        header_dict[library] = header_paths

    if missing_headers:
        error_message = "Couldn't find required headers: "
        error_message += ", ".join(missing_headers)
        cuda_paths = _get_cuda_paths()
        raise RuntimeError(f'{error_message}\nIs CUDA_HOME setup correctly? (CUDA_HOME="{cuda_paths}")')

    return header_dict


def _parse_headers(header_dict, include_path_list, parser_caching):
    from pyclibrary import CParser

    found_types = []
    found_functions = []
    found_values = []
    found_struct = []
    struct_list = {}

    replace = {
        " __device_builtin__ ": " ",
        "CUDARTAPI ": " ",
        "typedef __device_builtin__ enum cudaError cudaError_t;": "typedef cudaError cudaError_t;",
        "typedef __device_builtin__ enum cudaOutputMode cudaOutputMode_t;": "typedef cudaOutputMode cudaOutputMode_t;",
        "typedef enum cudaError cudaError_t;": "typedef cudaError cudaError_t;",
        "typedef enum cudaOutputMode cudaOutputMode_t;": "typedef cudaOutputMode cudaOutputMode_t;",
        "typedef enum cudaDataType_t cudaDataType_t;": "",
        "typedef enum libraryPropertyType_t libraryPropertyType_t;": "",
        "  enum ": "   ",
        ", enum ": ", ",
        "\\(enum ": "(",
        # Since we only support 64 bit architectures, we can inline the sizeof(T*) to 8 and then compute the
        # result in Python. The arithmetic expression is preserved to help with clarity and understanding
        r"char reserved\[52 - sizeof\(CUcheckpointGpuPair \*\)\];": rf"char reserved[{52 - 8}];",
        r"char reserved\[64 - sizeof\(CUcheckpointGpuPair \*\) - sizeof\(unsigned int\)\];": (
            rf"char reserved[{64 - 8 - 4}];"
        ),
    }

    print(f'Parsing headers in "{include_path_list}" (Caching = {parser_caching})', flush=True)
    for library, header_paths in header_dict.items():
        print(f"Parsing {library} headers", flush=True)
        parser = CParser(
            header_paths, cache="./cache_{}".format(library.split(".")[0]) if parser_caching else None, replace=replace
        )

        if library == "driver":
            CUDA_VERSION = parser.defs["macros"].get("CUDA_VERSION", "Unknown")
            print(f"Found CUDA_VERSION: {CUDA_VERSION}", flush=True)

        found_types += {key for key in parser.defs["types"]}
        found_types += {key for key in parser.defs["structs"]}
        found_types += {key for key in parser.defs["unions"]}
        found_types += {key for key in parser.defs["enums"]}
        found_functions += {key for key in parser.defs["functions"]}
        found_values += {key for key in parser.defs["values"]}

        for key, value in parser.defs["structs"].items():
            struct_list[key] = _Struct(key, value["members"])
        for key, value in parser.defs["unions"].items():
            struct_list[key] = _Struct(key, value["members"])

        for key, value in struct_list.items():
            if key.startswith("anon_union") or key.startswith("anon_struct"):
                continue

            found_struct += [key]
            discovered = value.discoverMembers(struct_list, key)
            if discovered:
                found_struct += discovered

    # TODO(#1312): make this work properly
    found_types.append("CUstreamAtomicReductionDataType_enum")

    return found_types, found_functions, found_values, found_struct, struct_list


# -----------------------------------------------------------------------
# Code generation helpers


def _fetch_input_files(path):
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".in")]


def _generate_output(infile, template_vars):
    from Cython import Tempita

    assert infile.endswith(".in")
    outfile = infile[:-3]

    with open(infile, encoding="utf-8") as f:
        pxdcontent = Tempita.Template(f.read()).substitute(template_vars)

    if os.path.exists(outfile):
        with open(outfile, encoding="utf-8") as f:
            if f.read() == pxdcontent:
                print(f"Skipping {infile} (No change)", flush=True)
                return
    with open(outfile, "w", encoding="utf-8") as f:
        print(f"Generating {infile}", flush=True)
        f.write(pxdcontent)


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

    All CUDA-dependent logic (header parsing, code generation, cythonization)
    is deferred to this function so that metadata queries do not require a
    CUDA toolkit installation.
    """
    import Cython
    from Cython.Build import cythonize
    from Cython.Compiler import Options as _CythonOptions

    global _extensions

    cuda_paths = _get_cuda_paths()
    # CUDA 12 supports split toolkit installations. Check the first cuda.h
    # found in the compiler's include-search order, allowing library-only roots.
    cuda_header_root = next((path for path in cuda_paths if os.path.isfile(_cuda_h_path(path))), cuda_paths[0])
    _check_cuda_headers(cuda_header_root)

    if os.environ.get("PARALLEL_LEVEL") is not None:
        warn(
            "Environment variable PARALLEL_LEVEL is deprecated. Use CUDA_PYTHON_PARALLEL_LEVEL instead",
            DeprecationWarning,
            stacklevel=2,
        )
        nthreads = int(os.environ.get("PARALLEL_LEVEL", "0"))
    else:
        nthreads = int(os.environ.get("CUDA_PYTHON_PARALLEL_LEVEL", "0") or "0")

    parser_caching = bool(os.environ.get("CUDA_PYTHON_PARSER_CACHING", False))
    compile_for_coverage = bool(int(os.environ.get("CUDA_PYTHON_COVERAGE", "0")))

    toolchain, _cc, _cxx, extra_compile_args, extra_link_args = _resolve_toolchain(
        debug=debug, compile_for_coverage=compile_for_coverage
    )
    _check_toolchain_available(toolchain)
    extra_cythonize_kwargs = {}
    if debug and sys.platform != "win32":
        extra_cythonize_kwargs["gdb_debug"] = True

    # Parse CUDA headers
    include_path_list = [os.path.join(path, "include") for path in cuda_paths]
    header_dict = _fetch_header_paths(_REQUIRED_HEADERS, include_path_list)
    found_types, found_functions, found_values, found_struct, struct_list = _parse_headers(
        header_dict, include_path_list, parser_caching
    )
    struct_field_types = {}
    struct_field_array_lengths = {}
    for struct_name, struct in struct_list.items():
        for member_name in struct._member_names:
            key = f"{struct_name}.{member_name}"
            struct_field_types[key] = struct.member_type(member_name)
            struct_field_array_lengths[key] = struct.member_array_length(member_name)

    # Generate code from .in templates
    path_list = [
        os.path.join("cuda"),
        os.path.join("cuda", "bindings"),
        os.path.join("cuda", "bindings", "_bindings"),
        os.path.join("cuda", "bindings", "_internal"),
        os.path.join("cuda", "bindings", "_lib"),
        os.path.join("cuda", "bindings", "utils"),
    ]
    input_files = []
    for path in path_list:
        input_files += _fetch_input_files(path)

    import platform

    template_vars = {
        "found_types": found_types,
        "found_functions": found_functions,
        "found_values": found_values,
        "found_struct": found_struct,
        "struct_list": struct_list,
        "struct_field_types": struct_field_types,
        "struct_field_array_lengths": struct_field_array_lengths,
        "os": os,
        "sys": sys,
        "platform": platform,
    }
    for file in input_files:
        _generate_output(file, template_vars)

    # Prepare compile/link arguments
    include_dirs = [
        os.path.dirname(sysconfig.get_path("include")),
    ] + include_path_list
    # Add the build-environment library path in setup.py instead of generated Cython sources.
    library_dirs = []
    cudalib_subdirs = [r"lib\x64"] if sys.platform == "win32" else ["lib64", "lib"]
    library_dirs.extend(os.path.join(prefix, subdir) for prefix in cuda_paths for subdir in cudalib_subdirs)

    # Rename architecture-specific files
    dst_files = _rename_architecture_specific_files()

    @atexit.register
    def _cleanup_dst_files():
        for dst in dst_files:
            with contextlib.suppress(FileNotFoundError):
                os.remove(dst)

    # Build extension list
    extensions = []
    static_runtime_libraries = ["cudart_static", "rt"] if sys.platform == "linux" else ["cudart_static"]
    cuda_bindings_files = glob.glob("cuda/bindings/*.pyx")
    if sys.platform == "win32":
        cuda_bindings_files = [f for f in cuda_bindings_files if "cufile" not in f]
    sources_list = [
        # private
        (["cuda/bindings/_bindings/cyruntime.pyx"], static_runtime_libraries),
        (["cuda/bindings/_bindings/cyruntime_ptds.pyx"], static_runtime_libraries),
        # utils
        (["cuda/bindings/utils/*.pyx"], None),
        # public
        *(([f], None) for f in cuda_bindings_files),
        # public (deprecated, to be removed)
        (["cuda/*.pyx"], None),
        # internal files used by generated bindings
        (["cuda/bindings/_internal/utils.pyx"], None),
        *(([f], None) for f in dst_files if f.endswith(".pyx")),
    ]

    for sources, libraries in sources_list:
        extensions += _prep_extensions(
            sources, libraries, include_dirs, library_dirs, extra_compile_args, extra_link_args
        )

    # Cythonize
    _CythonOptions.warning_errors = True
    cython_directives = dict(language_level=3, embedsignature=True, binding=True, freethreading_compatible=True)
    if compile_for_coverage:
        cython_directives["linetrace"] = True

    # Force a full rebuild when the toolchain changed since the last successful
    # build, so a stale .so from a previous toolchain is never packaged.
    _check_build_toolchain(toolchain)

    cache_path = _cython_cache_path(
        "cuda-bindings-12",
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
