# Copyright 2021-2024 NVIDIA Corporation.  All rights reserved.
#
# Please refer to the NVIDIA end user license agreement (EULA) associated
# with this source code for terms and conditions that govern your use of
# this software. Any use, reproduction, disclosure, or distribution of
# this software and related documentation outside the terms of the EULA
# is strictly prohibited.

import atexit
import contextlib
import glob
import os
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
    os.path.join("cuda", "bindings", "_lib"),
    os.path.join("cuda", "bindings", "_lib", "cyruntime"),
    os.path.join("cuda", "bindings", "_internal"),
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

extra_compile_args = []
extra_cythonize_kwargs = {}
if sys.platform != "win32":
    extra_compile_args += [
        "-std=c++14",
        "-fpermissive",
        "-Wno-deprecated-declarations",
        "-D _GLIBCXX_ASSERTIONS",
        "-fno-var-tracking-assignments",
    ]
    if "--debug" in sys.argv:
        extra_cythonize_kwargs["gdb_debug"] = True
        extra_compile_args += ["-g", "-O0"]
    else:
        extra_compile_args += ["-O3"]

# For Setup
extensions = []
new_extensions = []
cmdclass = {}

# ----------------------------------------------------------------------
# Cythonize


def prep_extensions(sources):
    pattern = sources[0]
    files = glob.glob(pattern)
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
                libraries=[],
                language="c++",
                extra_compile_args=extra_compile_args,
            )
        )
    return exts


# new path for the bindings from cybind
def rename_architecture_specific_files():
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
        compiler_directives=dict(profile=True, language_level=3, embedsignature=True, binding=True),
        **extra_cythonize_kwargs,
    )


sources_list = [
    # private
    ["cuda/bindings/_bindings/cydriver.pyx", "cuda/bindings/_bindings/loader.cpp"],
    ["cuda/bindings/_bindings/cynvrtc.pyx"],
    # utils
    ["cuda/bindings/_lib/utils.pyx", "cuda/bindings/_lib/param_packer.cpp"],
    ["cuda/bindings/_lib/cyruntime/cyruntime.pyx"],
    ["cuda/bindings/_lib/cyruntime/utils.pyx"],
    # public
    ["cuda/bindings/*.pyx"],
    # public (deprecated, to be removed)
    ["cuda/*.pyx"],
    # internal files used by generated bindings
    ["cuda/bindings/_internal/nvjitlink.pyx"],
    ["cuda/bindings/_internal/nvvm.pyx"],
    ["cuda/bindings/_internal/utils.pyx"],
]

for sources in sources_list:
    extensions += prep_extensions(sources)

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
            extra_linker_flags = ["-Wl,--strip-all"]

            # Allow extensions to discover libraries at runtime
            # relative their wheels installation.
            if ext.name == "cuda.bindings._bindings.cynvrtc":
                ldflag = "-Wl,--disable-new-dtags,-rpath,$ORIGIN/../../../nvidia/cuda_nvrtc/lib"
            elif ext.name == "cuda.bindings._internal.nvjitlink":
                ldflag = "-Wl,--disable-new-dtags,-rpath,$ORIGIN/../../../nvidia/nvjitlink/lib"
            elif ext.name == "cuda.bindings._internal.nvvm":
                # from <loc>/site-packages/cuda/bindings/_internal/
                #   to <loc>/site-packages/nvidia/cuda_nvcc/nvvm/lib64/
                rel1 = "$ORIGIN/../../../nvidia/cuda_nvcc/nvvm/lib64"
                # from <loc>/lib/python3.*/site-packages/cuda/bindings/_internal/
                #   to <loc>/lib/nvvm/lib64/
                rel2 = "$ORIGIN/../../../../../../nvvm/lib64"
                ldflag = f"-Wl,--disable-new-dtags,-rpath,{rel1},-rpath,{rel2}"
            else:
                ldflag = None

            if ldflag:
                extra_linker_flags.append(ldflag)
        else:
            extra_linker_flags = []

        ext.extra_link_args += extra_linker_flags
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
