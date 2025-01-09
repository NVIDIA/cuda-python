# Copyright 2021-2024 NVIDIA Corporation.  All rights reserved.
#
# Please refer to the NVIDIA end user license agreement (EULA) associated
# with this source code for terms and conditions that govern your use of
# this software. Any use, reproduction, disclosure, or distribution of
# this software and related documentation outside the terms of the EULA
# is strictly prohibited.

import glob
import os
import platform
import sys
import sysconfig

from Cython import Tempita
from Cython.Build import cythonize
from pyclibrary import CParser
from setuptools import find_packages, setup
from setuptools.extension import Extension
from setuptools.command.bdist_wheel import bdist_wheel
from setuptools.command.build_ext import build_ext
import versioneer


# ----------------------------------------------------------------------
# Fetch configuration options

CUDA_HOME = os.environ.get("CUDA_HOME", os.environ.get("CUDA_PATH", None))
if not CUDA_HOME:
    raise RuntimeError('Environment variable CUDA_HOME or CUDA_PATH is not set')

CUDA_HOME = CUDA_HOME.split(os.pathsep)
nthreads = int(os.environ.get("PARALLEL_LEVEL", "0") or "0")
PARSER_CACHING = os.environ.get("CUDA_PYTHON_PARSER_CACHING", False)
PARSER_CACHING = True if PARSER_CACHING else False

# ----------------------------------------------------------------------
# Parse user-provided CUDA headers

header_dict = {
    'driver' : ['cuda.h',
                'cudaProfiler.h',
                'cudaEGL.h',
                'cudaGL.h',
                'cudaVDPAU.h'],
    'runtime' : ['driver_types.h',
                 'vector_types.h',
                 'cuda_runtime.h',
                 'surface_types.h',
                 'texture_types.h',
                 'library_types.h',
                 'cuda_runtime_api.h',
                 'device_types.h',
                 'driver_functions.h',
                 'cuda_profiler_api.h',
                 'cuda_egl_interop.h',
                 'cuda_gl_interop.h',
                 'cuda_vdpau_interop.h'],
    'nvrtc' : ['nvrtc.h']}

replace = {' __device_builtin__ ':' ',
           'CUDARTAPI ':' ',
           'typedef __device_builtin__ enum cudaError cudaError_t;' : 'typedef cudaError cudaError_t;',
           'typedef __device_builtin__ enum cudaOutputMode cudaOutputMode_t;' : 'typedef cudaOutputMode cudaOutputMode_t;',
           'typedef enum cudaError cudaError_t;' : 'typedef cudaError cudaError_t;',
           'typedef enum cudaOutputMode cudaOutputMode_t;' : 'typedef cudaOutputMode cudaOutputMode_t;',
           'typedef enum cudaDataType_t cudaDataType_t;' : '',
           'typedef enum libraryPropertyType_t libraryPropertyType_t;' : '',
           '  enum ' : '   ',
           ', enum ' : ', ',
           '\\(enum ' : '(',}

found_types = []
found_structs = {}
found_unions = {}
found_functions = []
found_values = []

include_path_list = [os.path.join(path, 'include') for path in CUDA_HOME]
print(f'Parsing headers in "{include_path_list}" (Caching {PARSER_CACHING})')
for library, header_list in header_dict.items():
    header_paths = []
    for header in header_list:
        path_candidate = [os.path.join(path, header) for path in include_path_list]
        for path in path_candidate:
            if os.path.exists(path):
                header_paths += [path]
                break
        if not os.path.exists(path):
            print(f'Missing header {header}')

    print(f'Parsing {library} headers')
    parser = CParser(header_paths,
                     cache='./cache_{}'.format(library.split('.')[0]) if PARSER_CACHING else None,
                     replace=replace)

    if library == 'driver':
        CUDA_VERSION = parser.defs['macros']['CUDA_VERSION'] if 'CUDA_VERSION' in parser.defs['macros'] else 'Unknown'
        print(f'Found CUDA_VERSION: {CUDA_VERSION}')

    # Combine types with others since they sometimes get tangled
    found_types += {key for key in parser.defs['types']}
    found_types += {key for key in parser.defs['structs']}
    found_structs.update(parser.defs['structs'])
    found_types += {key for key in parser.defs['unions']}
    found_unions.update(parser.defs['unions'])
    found_types += {key for key in parser.defs['enums']}
    found_functions += {key for key in parser.defs['functions']}
    found_values += {key for key in parser.defs['values']}

if len(found_functions) == 0:
    raise RuntimeError(f'Parser found no functions. Is CUDA_HOME setup correctly? (CUDA_HOME="{CUDA_HOME}")')

# Unwrap struct and union members
def unwrapMembers(found_dict):
    for key in found_dict:
        members = [var for var, _, _ in found_dict[key]['members']]
        found_dict[key]['members'] = members

unwrapMembers(found_structs)
unwrapMembers(found_unions)

# ----------------------------------------------------------------------
# Generate

def fetch_input_files(path):
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.in')]

def generate_output(infile, local):
    assert infile.endswith('.in')
    outfile = infile[:-3]

    with open(infile) as f:
        pxdcontent = Tempita.Template(f.read()).substitute(local)

    if os.path.exists(outfile):
        with open(outfile) as f:
            if f.read() == pxdcontent:
                print(f'Skipping {infile} (No change)')
                return
    with open(outfile, "w") as f:
        print(f'Generating {infile}')
        f.write(pxdcontent)

path_list = [os.path.join('cuda'),
             os.path.join('cuda', 'bindings'),
             os.path.join('cuda', 'bindings', '_bindings'),
             os.path.join('cuda', 'bindings', '_lib'),
             os.path.join('cuda', 'bindings', '_lib', 'cyruntime')]
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
if sys.platform != 'win32':
    extra_compile_args += [
        '-std=c++14',
        '-fpermissive',
        '-Wno-deprecated-declarations',
        '-D _GLIBCXX_ASSERTIONS',
        '-fno-var-tracking-assignments'
    ]
    if '--debug' in sys.argv:
        extra_cythonize_kwargs['gdb_debug'] = True
        extra_compile_args += ['-g', '-O0']
    else:
        extra_compile_args += ['-O3']

# For Setup
extensions = []
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


def do_cythonize(extensions):
    return cythonize(
        extensions,
        nthreads=nthreads,
        compiler_directives=dict(
            profile=True, language_level=3, embedsignature=True, binding=True
        ),
        **extra_cythonize_kwargs)


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
                ldflag = f"-Wl,--disable-new-dtags,-rpath,$ORIGIN/../../../nvidia/cuda_nvrtc/lib"
            elif ext.name == "cuda.bindings._internal.nvjitlink":
                ldflag = f"-Wl,--disable-new-dtags,-rpath,$ORIGIN/../../../nvidia/nvjitlink/lib"
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


cmdclass = {"build_ext": ParallelBuildExtensions}
cmdclass = versioneer.get_cmdclass(cmdclass)

# ----------------------------------------------------------------------
# Setup

setup(
    version=versioneer.get_version(),
    ext_modules=do_cythonize(extensions),
    cmdclass=cmdclass,
    zip_safe=False,
)
