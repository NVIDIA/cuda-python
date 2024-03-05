# Copyright 2021-2024 NVIDIA Corporation.  All rights reserved.
#
# Please refer to the NVIDIA end user license agreement (EULA) associated
# with this source code for terms and conditions that govern your use of
# this software. Any use, reproduction, disclosure, or distribution of
# this software and related documentation outside the terms of the EULA
# is strictly prohibited.

from Cython import Tempita
from Cython.Build import cythonize
import os
import platform
from pyclibrary import CParser
import sys
import sysconfig
from setuptools import find_packages, setup
from setuptools.extension import Extension
from setuptools.command.build_ext import build_ext
import versioneer

# ----------------------------------------------------------------------
# Fetch configuration options

CUDA_HOME = os.environ.get("CUDA_HOME")
if not CUDA_HOME:
    CUDA_HOME = os.environ.get("CUDA_PATH")
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
             os.path.join('cuda', '_cuda'),
             os.path.join('cuda', '_lib'),
             os.path.join('cuda', '_lib', 'ccudart')]
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
if sys.platform == 'win32':
    from distutils import _msvccompiler
    _msvccompiler.PLAT_TO_VCVARS['win-amd64'] = 'amd64'

extensions = []
cmdclass = {}

# ----------------------------------------------------------------------
# Cythonize

def do_cythonize(sources):
    return cythonize(
                    [
                        Extension(
                            "*",
                            sources=sources,
                            include_dirs=include_dirs,
                            library_dirs=library_dirs,
                            runtime_library_dirs=[],
                            libraries=[],
                            language="c++",
                            extra_compile_args=extra_compile_args,
                        )
                    ],
                    nthreads=nthreads,
                    compiler_directives=dict(
                        profile=True, language_level=3, embedsignature=True, binding=True
                    ),
                    **extra_cythonize_kwargs)

sources_list = [
    # private
    ["cuda/_cuda/*.pyx", "cuda/_cuda/loader.cpp"],
    # utils
    ["cuda/_lib/*.pyx", "cuda/_lib/param_packer.cpp"],
    ["cuda/_lib/ccudart/*.pyx"],
    # public
    ["cuda/*.pyx"],
    # tests
    ["cuda/tests/*.pyx"]]

for sources in sources_list:
    extensions += do_cythonize(sources)

# ---------------------------------------------------------------------
# Custom build_ext command
# Files are build in two steps:
# 1) Cythonized (in the do_cythonize() command)
# 2) Compiled to .o files as part of build_ext
# This class is solely for passing the value of nthreads to build_ext

class ParallelBuildExtensions(build_ext):
    def initialize_options(self):
        build_ext.initialize_options(self)
        if nthreads > 0:
            self.parallel = nthreads

    def finalize_options(self):
        build_ext.finalize_options(self)

cmdclass = {"build_ext": ParallelBuildExtensions}
cmdclass = versioneer.get_cmdclass(cmdclass)

# ----------------------------------------------------------------------
# Setup

setup(
    version=versioneer.get_version(),
    ext_modules=extensions,
    packages=find_packages(include=["cuda", "cuda.*"]),
    package_data=dict.fromkeys(
        find_packages(include=["cuda", "cuda.*"]),
        ["*.pxd", "*.pyx", "*.h", "*.cpp"],
    ),
    cmdclass=cmdclass,
    zip_safe=False,
)
