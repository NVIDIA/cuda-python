# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

import atexit
import contextlib
import glob
import os
import pathlib
import platform
import shutil
import sys
import sysconfig
import tempfile
from warnings import warn

from Cython import Tempita
from Cython.Build import cythonize
from pyclibrary import CParser
from setuptools import find_packages, setup
from setuptools.command.bdist_wheel import bdist_wheel
from setuptools.command.build_ext import build_ext
from setuptools.command.build_py import build_py
from setuptools.command.editable_wheel import _TopLevelFinder, editable_wheel
from setuptools.extension import Extension

# ----------------------------------------------------------------------
# Fetch configuration options

CUDA_HOME = os.environ.get("CUDA_HOME", os.environ.get("CUDA_PATH", None))
if not CUDA_HOME:
    raise RuntimeError("Environment variable CUDA_HOME or CUDA_PATH is not set")

CUDA_HOME = CUDA_HOME.split(os.pathsep)

if os.environ.get("PARALLEL_LEVEL") is not None:
    warn(
        "Environment variable PARALLEL_LEVEL is deprecated. Use CUDA_PYTHON_PARALLEL_LEVEL instead",
        DeprecationWarning,
        stacklevel=1,
    )
    nthreads = int(os.environ.get("PARALLEL_LEVEL", "0"))
else:
    nthreads = int(os.environ.get("CUDA_PYTHON_PARALLEL_LEVEL", "0") or "0")
PARSER_CACHING = os.environ.get("CUDA_PYTHON_PARSER_CACHING", False)
PARSER_CACHING = bool(PARSER_CACHING)

# ----------------------------------------------------------------------
# Parse user-provided CUDA headers

required_headers = {
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


def fetch_header_paths(required_headers, include_path_list):
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

        # Update dictionary with validated paths to headers
        header_dict[library] = header_paths

    if missing_headers:
        error_message = "Couldn't find required headers: "
        error_message += ", ".join([header for header in missing_headers])
        raise RuntimeError(f'{error_message}\nIs CUDA_HOME setup correctly? (CUDA_HOME="{CUDA_HOME}")')

    return header_dict


class Struct:
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

    def discoverMembers(self, memberDict, prefix):
        discovered = []
        for memberName, memberType in zip(self._member_names, self._member_types):
            if memberName:
                discovered += [".".join([prefix, memberName])]
            if memberType in memberDict:
                discovered += memberDict[memberType].discoverMembers(
                    memberDict, discovered[-1] if memberName else prefix
                )
        return discovered

    def __repr__(self):
        return f"{self._name}: {self._member_names} with types {self._member_types}"


def parse_headers(header_dict):
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

    print(f'Parsing headers in "{include_path_list}" (Caching = {PARSER_CACHING})')
    for library, header_paths in header_dict.items():
        print(f"Parsing {library} headers")
        parser = CParser(
            header_paths, cache="./cache_{}".format(library.split(".")[0]) if PARSER_CACHING else None, replace=replace
        )

        if library == "driver":
            CUDA_VERSION = parser.defs["macros"].get("CUDA_VERSION", "Unknown")
            print(f"Found CUDA_VERSION: {CUDA_VERSION}")

        # Combine types with others since they sometimes get tangled
        found_types += {key for key in parser.defs["types"]}
        found_types += {key for key in parser.defs["structs"]}
        found_types += {key for key in parser.defs["unions"]}
        found_types += {key for key in parser.defs["enums"]}
        found_functions += {key for key in parser.defs["functions"]}
        found_values += {key for key in parser.defs["values"]}

        for key, value in parser.defs["structs"].items():
            struct_list[key] = Struct(key, value["members"])
        for key, value in parser.defs["unions"].items():
            struct_list[key] = Struct(key, value["members"])

        for key, value in struct_list.items():
            if key.startswith("anon_union") or key.startswith("anon_struct"):
                continue

            found_struct += [key]
            discovered = value.discoverMembers(struct_list, key)
            if discovered:
                found_struct += discovered

    return found_types, found_functions, found_values, found_struct, struct_list


include_path_list = [os.path.join(path, "include") for path in CUDA_HOME]
header_dict = fetch_header_paths(required_headers, include_path_list)
found_types, found_functions, found_values, found_struct, struct_list = parse_headers(header_dict)

# ----------------------------------------------------------------------
# Generate


def fetch_input_files(path):
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".in")]


def generate_output(infile, local):
    assert infile.endswith(".in")
    outfile = infile[:-3]

    with open(infile) as f:
        pxdcontent = Tempita.Template(f.read()).substitute(local)

    if os.path.exists(outfile):
        with open(outfile) as f:
            if f.read() == pxdcontent:
                print(f"Skipping {infile} (No change)")
                return
    with open(outfile, "w") as f:
        print(f"Generating {infile}")
        f.write(pxdcontent)


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
    input_files += fetch_input_files(path)

for file in input_files:
    generate_output(file, locals())

# ----------------------------------------------------------------------
# Prepare compile arguments

# For Cython
include_dirs = [
    os.path.dirname(sysconfig.get_path("include")),
] + include_path_list
library_dirs = [sysconfig.get_path("platlib"), os.path.join(os.sys.prefix, "lib")]
cudalib_subdirs = [r"lib\x64"] if sys.platform == "win32" else ["lib64", "lib"]
library_dirs.extend(os.path.join(prefix, subdir) for prefix in CUDA_HOME for subdir in cudalib_subdirs)

extra_compile_args = []
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
        extra_compile_args += ["-D _GLIBCXX_ASSERTIONS"]  # libstdc++
    # extra_compile_args += ["-D _LIBCPP_ENABLE_ASSERTIONS"] # Consider: if clang, use libc++ preprocessor macros.
    else:
        extra_compile_args += ["-O3"]

# For Setup
extensions = []
new_extensions = []
cmdclass = {}

# ----------------------------------------------------------------------
# Cythonize


def prep_extensions(sources, libraries):
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
            )
        )
    return exts


# new path for the bindings from cybind
def rename_architecture_specific_files():
    path = os.path.join("cuda", "bindings", "_internal")
    if sys.platform == "linux":
        src_files = glob.glob(os.path.join(path, "*_linux.pyx"))
    elif sys.platform == "win32":
        src_files = glob.glob(os.path.join(path, "*_windows.pyx"))
    else:
        raise RuntimeError(f"platform is unrecognized: {sys.platform}")
    dst_files = []
    for src in src_files:
        # Set up a temporary file; it must be under the cache directory so
        # that atomic moves within the same filesystem can be guaranteed
        with tempfile.NamedTemporaryFile(delete=False, dir=".") as f:
            shutil.copy2(src, f.name)
            f_name = f.name
        dst = src.replace("_linux", "").replace("_windows", "")
        # atomic move with the destination guaranteed to be overwritten
        os.replace(f_name, f"./{dst}")
        dst_files.append(dst)
    return dst_files


dst_files = rename_architecture_specific_files()


@atexit.register
def cleanup_dst_files():
    for dst in dst_files:
        with contextlib.suppress(FileNotFoundError):
            os.remove(dst)


def do_cythonize(extensions):
    return cythonize(
        extensions,
        nthreads=nthreads,
        compiler_directives=dict(language_level=3, embedsignature=True, binding=True),
        **extra_cythonize_kwargs,
    )


static_runtime_libraries = ["cudart_static", "rt"] if sys.platform == "linux" else ["cudart_static"]
cuda_bindings_files = glob.glob("cuda/bindings/*.pyx")
if sys.platform == "win32":
    # cuFILE does not support Windows
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
    extensions += prep_extensions(sources, libraries)

# ---------------------------------------------------------------------
# Custom cmdclass extensions

building_wheel = False


class WheelsBuildExtensions(bdist_wheel):
    def run(self):
        global building_wheel
        building_wheel = True
        super().run()


class ParallelBuildExtensions(build_ext):
    def initialize_options(self):
        super().initialize_options()
        if nthreads > 0:
            self.parallel = nthreads

    def build_extension(self, ext):
        if building_wheel and sys.platform == "linux":
            # Strip binaries to remove debug symbols
            ext.extra_link_args.append("-Wl,--strip-all")
        super().build_extension(ext)


cmdclass = {
    "bdist_wheel": WheelsBuildExtensions,
    "build_ext": ParallelBuildExtensions,
}

# ----------------------------------------------------------------------
# Setup

setup(
    ext_modules=do_cythonize(extensions),
    cmdclass=cmdclass,
    zip_safe=False,
)
