# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

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
        from importlib.metadata import PackageNotFoundError, distribution
        from pathlib import Path

        try:
            dist = distribution("cuda-pathfinder")
        except PackageNotFoundError:
            raise ModuleNotFoundError(
                "cuda-pathfinder is not installed in the build environment. "
                "Ensure 'cuda-pathfinder>=1.5' is in build-system.requires."
            ) from None
        import cuda

        locate_result = str(dist.locate_file(Path("cuda")))
        print(f"[diag] dist._path: {dist._path}", flush=True)
        print(f"[diag] dist._path.parent: {dist._path.parent}", flush=True)
        print(f"[diag] locate_file('cuda'): {locate_result}", flush=True)
        print(f"[diag] locate_file exists: {os.path.isdir(locate_result)}", flush=True)
        print(
            f"[diag] locate_file/pathfinder exists: {os.path.isdir(os.path.join(locate_result, 'pathfinder'))}",
            flush=True,
        )
        print(f"[diag] cuda.__path__ (before): {cuda.__path__}", flush=True)
        print("[diag] sys.path:", flush=True)
        for p in sys.path:
            sp_cuda = os.path.join(p, "cuda")
            has_pf = os.path.isdir(os.path.join(sp_cuda, "pathfinder"))
            print(f"[diag]   {p}  ->  cuda/pathfinder exists: {has_pf}", flush=True)

        site_cuda = locate_result
        cuda_paths = list(cuda.__path__)
        if site_cuda not in cuda_paths:
            cuda.__path__ = cuda_paths + [site_cuda]
        print(f"[diag] cuda.__path__ (after): {cuda.__path__}", flush=True)
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


# -----------------------------------------------------------------------
# Header parsing helpers (called only from _build_cuda_bindings)

_REQUIRED_HEADERS = {
    "driver": [
        "cuda.h",
        "cudaProfiler.h",
    ],
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
    "nvrtc": [
        "nvrtc.h",
    ],
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
        for var_name, var_type, _ in members:
            var_type = var_type[0]
            var_type = var_type.removeprefix("struct ")
            var_type = var_type.removeprefix("union ")

            self._member_names += [var_name]
            self._member_types += [var_type]

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
        cuda_path = _get_cuda_path()
        raise RuntimeError(f'{error_message}\nIs CUDA_PATH setup correctly? (CUDA_PATH="{cuda_path}")')

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

        found_types += set(parser.defs["types"])
        found_types += set(parser.defs["structs"])
        found_types += set(parser.defs["unions"])
        found_types += set(parser.defs["enums"])
        found_functions += set(parser.defs["functions"])
        found_values += set(parser.defs["values"])

        for key, value in parser.defs["structs"].items():
            struct_list[key] = _Struct(key, value["members"])
        for key, value in parser.defs["unions"].items():
            struct_list[key] = _Struct(key, value["members"])

        for key, value in struct_list.items():
            if key.startswith(("anon_union", "anon_struct")):
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


def _build_cuda_bindings(strip=False):
    """Build all cuda-bindings extensions.

    All CUDA-dependent logic (header parsing, code generation, cythonization)
    is deferred to this function so that metadata queries do not require a
    CUDA toolkit installation.
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

    parser_caching = bool(os.environ.get("CUDA_PYTHON_PARSER_CACHING", False))
    compile_for_coverage = bool(int(os.environ.get("CUDA_PYTHON_COVERAGE", "0")))

    # Parse CUDA headers
    include_path_list = [os.path.join(cuda_path, "include")]
    header_dict = _fetch_header_paths(_REQUIRED_HEADERS, include_path_list)
    found_types, found_functions, found_values, found_struct, struct_list = _parse_headers(
        header_dict, include_path_list, parser_caching
    )

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
    library_dirs = [sysconfig.get_path("platlib"), os.path.join(os.sys.prefix, "lib")]
    cudalib_subdirs = [r"lib\x64"] if sys.platform == "win32" else ["lib64", "lib"]
    library_dirs.extend(os.path.join(cuda_path, subdir) for subdir in cudalib_subdirs)

    extra_compile_args = []
    extra_link_args = []
    extra_cythonize_kwargs = {}
    if sys.platform != "win32":
        extra_compile_args += [
            "-std=c++14",
            "-fpermissive",
            "-Wno-deprecated-declarations",
            "-fno-var-tracking-assignments",
        ]
        if "--debug" in sys.argv:
            extra_cythonize_kwargs["gdb_debug"] = True
            extra_compile_args += ["-g", "-O0"]
            extra_compile_args += ["-D _GLIBCXX_ASSERTIONS"]
        else:
            extra_compile_args += ["-O3"]
            if strip and sys.platform == "linux":
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
    static_runtime_libraries = ["cudart_static", "rt"] if sys.platform == "linux" else ["cudart_static"]
    cuda_bindings_files = glob.glob("cuda/bindings/*.pyx")
    if sys.platform == "win32":
        cuda_bindings_files = [f for f in cuda_bindings_files if "cufile" not in f]
    sources_list = [
        # private
        (["cuda/bindings/_bindings/cydriver.pyx", "cuda/bindings/_bindings/loader.cpp"], None),
        (["cuda/bindings/_bindings/cynvrtc.pyx"], None),
        (["cuda/bindings/_bindings/cyruntime.pyx"], static_runtime_libraries),
        (["cuda/bindings/_bindings/cyruntime_ptds.pyx"], static_runtime_libraries),
        # utils
        (["cuda/bindings/utils/*.pyx"], None),
        # public
        *(([f], None) for f in cuda_bindings_files),
        # internal files used by generated bindings
        (["cuda/bindings/_internal/utils.pyx"], None),
        *(([f], None) for f in dst_files if f.endswith(".pyx")),
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
        build_dir="." if compile_for_coverage else "build/cython",
        compiler_directives=cython_directives,
        **extra_cythonize_kwargs,
    )


# -----------------------------------------------------------------------
# PEP 517 build hooks


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    _build_cuda_bindings(strip=True)
    return _build_meta.build_wheel(wheel_directory, config_settings, metadata_directory)


def build_editable(wheel_directory, config_settings=None, metadata_directory=None):
    _build_cuda_bindings(strip=False)
    return _build_meta.build_editable(wheel_directory, config_settings, metadata_directory)
